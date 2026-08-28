use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process;

use ndarray::Array2;
use num_complex::Complex;

use crate::npy_output::{NamedNpz, NpyMeta};
use plotters::coord::Shift;
use plotters::prelude::*;
use rustfft::FftPlanner;
use zip::ZipArchive;

type C32 = Complex<f32>;

const HISTOGRAM_BINS: usize = 256;

pub fn has_histogram_mode(rfi_args: &[String]) -> bool {
    rfi_args
        .iter()
        .any(|value| value.eq_ignore_ascii_case("histogram") || value.eq_ignore_ascii_case("hist"))
}

/// Extract the Rayleigh tail count from `--rfi histogram count:N`.
///
/// The count token is consumed from the RFI argument list so the remaining
/// values continue through the normal numeric-range/NPZ parser.  A bare
/// `histogram`/`hist` keeps the Zig-compatible default of one tail sample.
pub fn parse_histogram_count(rfi_args: &mut Vec<String>) -> io::Result<u64> {
    let histogram = has_histogram_mode(rfi_args);
    let mut count = 1u64;
    let mut count_seen = false;
    let mut filtered = Vec::with_capacity(rfi_args.len());
    let mut index = 0usize;

    while index < rfi_args.len() {
        let current = rfi_args[index].trim().to_string();
        let Some((key, inline_value)) = current.split_once(':') else {
            filtered.push(rfi_args[index].clone());
            index += 1;
            continue;
        };
        if !key.trim().eq_ignore_ascii_case("count") {
            filtered.push(rfi_args[index].clone());
            index += 1;
            continue;
        }
        if !histogram {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--rfi count:N requires --rfi histogram",
            ));
        }
        if count_seen {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--rfi histogram accepts only one count:N subargument",
            ));
        }

        let mut raw_value = inline_value.trim().to_string();
        if raw_value.is_empty() {
            index += 1;
            if index >= rfi_args.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "--rfi histogram count: requires an integer value",
                ));
            }
            raw_value = rfi_args[index].trim().to_string();
        }
        let parsed = raw_value.parse::<u64>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid --rfi histogram count: '{}'", raw_value),
            )
        })?;
        if parsed == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--rfi histogram count must be at least 1",
            ));
        }
        count = parsed;
        count_seen = true;
        index += 1;
    }

    *rfi_args = filtered;
    Ok(count)
}

/// RFI masks exported by the noise-histogram tools.
///
/// `frequency_rate` uses row-major [rate, frequency] coordinates and is
/// applied after the time FFT. `delay_rate` uses [rate, delay] coordinates and
/// is applied after the frequency-to-delay IFFT.
#[derive(Debug, Clone)]
pub struct RfiMask {
    pub frequency_rate: Option<PlaneMask>,
    pub delay_rate: Option<PlaneMask>,
}

#[derive(Debug, Clone)]
pub struct PlaneMask {
    pub axis_x: Vec<f64>,
    pub rate_hz: Vec<f64>,
    pub mask: Vec<bool>,
    pub rows: usize,
    pub cols: usize,
}

impl PlaneMask {
    fn is_valid(&self) -> bool {
        self.rows > 0
            && self.cols > 0
            && self.mask.len() == self.rows.saturating_mul(self.cols)
            && self.rate_hz.len() == self.rows
            && self.axis_x.len() == self.cols
    }

    fn marked_points(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.mask
            .iter()
            .enumerate()
            .filter_map(|(index, &marked)| marked.then_some((index / self.cols, index % self.cols)))
    }
}

impl RfiMask {
    pub fn is_empty(&self) -> bool {
        self.frequency_rate.as_ref().map_or(true, |plane| {
            !plane.is_valid() || !plane.mask.iter().any(|value| *value)
        }) && self.delay_rate.as_ref().map_or(true, |plane| {
            !plane.is_valid() || !plane.mask.iter().any(|value| *value)
        })
    }

    /// Conservative channel mask used by direct coherent-search evaluation.
    pub fn frequency_channel_mask(
        &self,
        channel_count: usize,
        sampling_speed: i32,
        fft_point: i32,
    ) -> Vec<bool> {
        let mut result = vec![false; channel_count];
        let Some(plane) = self
            .frequency_rate
            .as_ref()
            .filter(|value| value.is_valid())
        else {
            return result;
        };
        for (_, source_x) in plane.marked_points() {
            if let Some(channel) = map_frequency_index(
                plane.axis_x[source_x],
                &plane.axis_x,
                channel_count,
                sampling_speed,
                fft_point,
            ) {
                result[channel] = true;
            }
        }
        result
    }

    /// Zero marked frequency-rate cells in a row-major [frequency, rate] FFT
    /// plane. Returns the number of cells changed.
    pub fn apply_frequency_rate(
        &self,
        values: &mut [C32],
        frequency_count: usize,
        rate_count: usize,
        sampling_speed: i32,
        fft_point: i32,
        effective_integ_time: f32,
    ) -> usize {
        let Some(plane) = self
            .frequency_rate
            .as_ref()
            .filter(|value| value.is_valid())
        else {
            return 0;
        };
        if values.len() != frequency_count.saturating_mul(rate_count) {
            return 0;
        }
        let rate_map = build_rate_map(&plane.rate_hz, rate_count, effective_integ_time);
        let mut changed = 0usize;
        for (source_rate, source_x) in plane.marked_points() {
            let Some(target_rate) = rate_map[source_rate] else {
                continue;
            };
            let Some(target_frequency) = map_frequency_index(
                plane.axis_x[source_x],
                &plane.axis_x,
                frequency_count,
                sampling_speed,
                fft_point,
            ) else {
                continue;
            };
            let index = target_frequency * rate_count + target_rate;
            if values[index] != C32::new(0.0, 0.0) {
                values[index] = C32::new(0.0, 0.0);
                changed += 1;
            }
        }
        changed
    }

    /// Test whether a physical delay/rate candidate falls on an imported
    /// delay-rate RFI cell. This is used by direct search paths that do not
    /// materialize an IFFT plane for every candidate.
    pub fn contains_delay_rate(&self, delay: f32, rate: f32) -> bool {
        let Some(plane) = self.delay_rate.as_ref().filter(|value| value.is_valid()) else {
            return false;
        };
        let source_delay = if plane.axis_x.iter().all(|value| *value >= -0.5)
            && plane
                .axis_x
                .iter()
                .all(|value| *value <= plane.cols as f64 + 0.5)
        {
            delay as f64 + plane.cols as f64 / 2.0 - 1.0
        } else {
            delay as f64
        };
        let Some(delay_index) = nearest_axis_index(&plane.axis_x, source_delay) else {
            return false;
        };
        let Some(rate_index) = nearest_axis_index(&plane.rate_hz, rate as f64) else {
            return false;
        };
        plane.mask[rate_index * plane.cols + delay_index]
    }

    /// Zero marked delay-rate cells in a row-major [rate, delay] IFFT plane.
    pub fn apply_delay_rate(
        &self,
        values: &mut [C32],
        rate_count: usize,
        delay_count: usize,
        effective_integ_time: f32,
        fft_point: i32,
    ) -> usize {
        let Some(plane) = self.delay_rate.as_ref().filter(|value| value.is_valid()) else {
            return 0;
        };
        if values.len() != rate_count.saturating_mul(delay_count) {
            return 0;
        }
        let rate_map = build_rate_map(&plane.rate_hz, rate_count, effective_integ_time);
        let mut changed = 0usize;
        for (source_rate, source_x) in plane.marked_points() {
            let Some(target_rate) = rate_map[source_rate] else {
                continue;
            };
            let Some(target_delay) = map_delay_index(
                plane.axis_x[source_x],
                &plane.axis_x,
                delay_count,
                fft_point,
            ) else {
                continue;
            };
            let index = target_rate * delay_count + target_delay;
            if values[index] != C32::new(0.0, 0.0) {
                values[index] = C32::new(0.0, 0.0);
                changed += 1;
            }
        }
        changed
    }
}

/// Load an NPZ RFI product. Numeric `MIN,MAX` arguments remain handled by
/// `parse_rfi_ranges`; this function only reads the `.npz` path.
pub fn load_rfi_npz(path: &Path) -> io::Result<RfiMask> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file).map_err(|error| {
        io::Error::new(io::ErrorKind::InvalidData, format!("invalid NPZ: {error}"))
    })?;
    let axis_x = read_f64_entry(&mut archive, "axis_x.npy")?.unwrap_or_default();
    let rate_hz = read_f64_entry(&mut archive, "rate_hz.npy")?.unwrap_or_default();

    let frequency_rate =
        if let Some(mask) = read_bool_entry(&mut archive, "ifft_rfi_frequency_mask.npy")? {
            let (rows, cols) = read_shape(&mut archive, "ifft_rfi_frequency_mask.npy")?;
            Some(normalize_plane_mask(mask, rows, cols, &axis_x, &rate_hz)?)
        } else if let Some(coordinates) =
            read_f64_entry(&mut archive, "ifft_rfi_frequency_coordinates.npy")?
        {
            let (rows, cols) = (rate_hz.len(), axis_x.len());
            if rows == 0 || cols == 0 || coordinates.len() % 2 != 0 {
                None
            } else {
                let mut mask = vec![false; rows.saturating_mul(cols)];
                for pair in coordinates.chunks_exact(2) {
                    if let (Some(x), Some(rate)) = (
                        nearest_axis_index(&axis_x, pair[0]),
                        nearest_axis_index(&rate_hz, pair[1]),
                    ) {
                        mask[rate * cols + x] = true;
                    }
                }
                Some(PlaneMask {
                    axis_x: axis_x.clone(),
                    rate_hz: rate_hz.clone(),
                    mask,
                    rows,
                    cols,
                })
            }
        } else {
            None
        };

    // The histogram source mask is a delay-rate plane. If an NPZ contains the
    // IFFT-derived frequency mask as well, both are retained.
    let delay_rate = if let Some(mask) = read_bool_entry(&mut archive, "rfi_mask.npy")? {
        let (rows, cols) = read_shape(&mut archive, "rfi_mask.npy")?;
        Some(normalize_plane_mask(mask, rows, cols, &axis_x, &rate_hz)?)
    } else {
        None
    };

    let result = RfiMask {
        frequency_rate,
        delay_rate,
    };
    if result.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "NPZ contains no usable RFI mask (expected ifft_rfi_frequency_mask/coordinates or rfi_mask)",
        ));
    }
    Ok(result)
}

fn normalize_plane_mask(
    mask: Vec<bool>,
    rows: usize,
    cols: usize,
    axis_x: &[f64],
    rate_hz: &[f64],
) -> io::Result<PlaneMask> {
    if rows == 0 || cols == 0 || mask.len() != rows.saturating_mul(cols) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid RFI mask shape",
        ));
    }
    if axis_x.len() != cols || rate_hz.len() != rows {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("RFI mask axes do not match shape ({rows},{cols})"),
        ));
    }
    Ok(PlaneMask {
        axis_x: axis_x.to_vec(),
        rate_hz: rate_hz.to_vec(),
        mask,
        rows,
        cols,
    })
}

#[derive(Debug)]
struct NpyArray {
    descr: String,
    shape: Vec<usize>,
    payload: Vec<u8>,
}

fn read_shape(archive: &mut ZipArchive<File>, name: &str) -> io::Result<(usize, usize)> {
    let array = read_npy_entry(archive, name)?.ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, format!("missing NPZ entry {name}"))
    })?;
    match array.shape.as_slice() {
        [rows, cols] => Ok((*rows, *cols)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{name} must be 2-D"),
        )),
    }
}

fn read_npy_entry(archive: &mut ZipArchive<File>, name: &str) -> io::Result<Option<NpyArray>> {
    let mut entry = match archive.by_name(name) {
        Ok(entry) => entry,
        Err(zip::result::ZipError::FileNotFound) => return Ok(None),
        Err(error) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                error.to_string(),
            ))
        }
    };
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid NPY entry {name}"),
        ));
    }
    let (header_start, header_len) = match bytes[6] {
        1 => (10usize, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated NPY header",
                ));
            }
            (
                12usize,
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            )
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported NPY version",
            ))
        }
    };
    let data_start = header_start
        .checked_add(header_len)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "NPY header overflow"))?;
    if data_start > bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated NPY payload",
        ));
    }
    let header = String::from_utf8_lossy(&bytes[header_start..data_start]);
    let descr = extract_quoted_value(&header, "descr").ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, format!("{name} has no dtype"))
    })?;
    let shape_text = header
        .split("shape")
        .nth(1)
        .and_then(|part| part.split('(').nth(1))
        .and_then(|part| part.split(')').next())
        .unwrap_or("");
    let shape = shape_text
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect::<Vec<_>>();
    Ok(Some(NpyArray {
        descr,
        shape,
        payload: bytes[data_start..].to_vec(),
    }))
}

fn extract_quoted_value(header: &str, key: &str) -> Option<String> {
    let start = header.find(key)?;
    let rest = &header[start + key.len()..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    let quote = rest.as_bytes().first().copied()?;
    if quote != b'\'' && quote != b'"' {
        return None;
    }
    let end = rest[1..].find(quote as char)? + 1;
    Some(rest[1..end].to_string())
}

fn read_f64_entry(archive: &mut ZipArchive<File>, name: &str) -> io::Result<Option<Vec<f64>>> {
    let Some(array) = read_npy_entry(archive, name)? else {
        return Ok(None);
    };
    let item_size = if array.descr.ends_with("f8") {
        8
    } else if array.descr.ends_with("f4") {
        4
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{name} is not floating point"),
        ));
    };
    if array.payload.len() % item_size != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid {name} payload"),
        ));
    }
    let values = array
        .payload
        .chunks_exact(item_size)
        .map(|bytes| {
            if item_size == 8 {
                f64::from_le_bytes(bytes.try_into().unwrap())
            } else {
                f32::from_le_bytes(bytes.try_into().unwrap()) as f64
            }
        })
        .collect();
    Ok(Some(values))
}

fn read_bool_entry(archive: &mut ZipArchive<File>, name: &str) -> io::Result<Option<Vec<bool>>> {
    let Some(array) = read_npy_entry(archive, name)? else {
        return Ok(None);
    };
    if !array.descr.ends_with("b1") && !array.descr.ends_with("u1") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{name} is not bool/uint8"),
        ));
    }
    Ok(Some(
        array.payload.iter().map(|value| *value != 0).collect(),
    ))
}

fn nearest_axis_index(axis: &[f64], value: f64) -> Option<usize> {
    if axis.is_empty() || !value.is_finite() {
        return None;
    }
    let mut best = None;
    for (index, &candidate) in axis.iter().enumerate() {
        if !candidate.is_finite() {
            continue;
        }
        let distance = (candidate - value).abs();
        if best.map_or(true, |(_, best_distance): (usize, f64)| {
            distance < best_distance
        }) {
            best = Some((index, distance));
        }
    }
    best.map(|(index, _)| index)
}

fn build_rate_map(source_axis: &[f64], target_count: usize, integ_time: f32) -> Vec<Option<usize>> {
    let target_axis = crate::utils::rate_cal(target_count as f32, integ_time.max(1.0e-9));
    let target_min = target_axis.iter().copied().fold(f32::INFINITY, f32::min);
    let target_max = target_axis
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let target_step = target_axis
        .windows(2)
        .find_map(|pair| {
            let delta = (pair[1] - pair[0]).abs();
            (delta > 1.0e-12).then_some(delta)
        })
        .unwrap_or(f32::INFINITY);
    source_axis
        .iter()
        .map(|&value| {
            if !value.is_finite()
                || target_axis.is_empty()
                || (value as f32) < target_min - target_step * 0.5
                || (value as f32) > target_max + target_step * 0.5
            {
                return None;
            }
            let mut best = None;
            for (index, &candidate) in target_axis.iter().enumerate() {
                let distance = (candidate - value as f32).abs();
                if best.map_or(true, |(_, best_distance): (usize, f32)| {
                    distance < best_distance
                }) {
                    best = Some((index, distance));
                }
            }
            best.map(|(index, _)| index)
        })
        .collect()
}

fn map_frequency_index(
    value: f64,
    source_axis: &[f64],
    target_count: usize,
    sampling_speed: i32,
    fft_point: i32,
) -> Option<usize> {
    if target_count == 0 || !value.is_finite() {
        return None;
    }
    let source_count = source_axis.len();
    let spacing = source_axis
        .windows(2)
        .find_map(|pair| {
            let delta = (pair[1] - pair[0]).abs();
            (delta > 1.0e-9).then_some(delta)
        })
        .unwrap_or(1.0);
    let index_like = spacing > 0.5 && spacing < 1.5;
    let target = if index_like {
        value * target_count as f64 / source_count.max(1) as f64
    } else {
        let step_mhz = sampling_speed as f64 / fft_point.max(1) as f64 / 1.0e6;
        if step_mhz <= 0.0 {
            return None;
        }
        value / step_mhz
    };
    let index = target.round() as isize;
    (index >= 0 && (index as usize) < target_count).then_some(index as usize)
}

fn map_delay_index(
    value: f64,
    source_axis: &[f64],
    target_count: usize,
    fft_point: i32,
) -> Option<usize> {
    if target_count == 0 || !value.is_finite() {
        return None;
    }
    let source_count = source_axis.len();
    let index_like = source_axis.iter().all(|value| *value >= -0.5)
        && source_axis
            .iter()
            .all(|value| *value <= source_count as f64 + 0.5);
    let target = if index_like {
        value * target_count as f64 / source_count.max(1) as f64
    } else {
        value + target_count as f64 / 2.0 - 1.0
            + (fft_point.max(0) as f64 - target_count as f64) / 2.0
    };
    let index = target.round() as isize;
    (index >= 0 && (index as usize) < target_count).then_some(index as usize)
}

pub fn parse_rfi_ranges(rfi_args: &[String], rbw: f32) -> io::Result<Vec<(usize, usize)>> {
    if rfi_args.is_empty() {
        return Ok(vec![]);
    }
    let mut ranges = Vec::new();
    for rfi_pair in rfi_args {
        // NPZ masks are loaded once by main and applied in the plane matching
        // their stored coordinates. They are not MIN,MAX frequency ranges.
        if rfi_pair.to_ascii_lowercase().ends_with(".npz")
            || has_histogram_mode(std::slice::from_ref(rfi_pair))
        {
            continue;
        }
        let parts: Vec<&str> = rfi_pair.split(',').collect();
        if parts.len() != 2 {
            eprintln!(
                "Invalid RFI format: {}. Expected MIN,MAX, histogram, or an RFI .npz file.",
                rfi_pair
            );
            process::exit(1);
        }

        let min_mhz_int: i32 = parts[0].parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Invalid integer for RFI min: {}", parts[0]),
            )
        })?;
        let max_mhz_int: i32 = parts[1].parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Invalid integer for RFI max: {}", parts[1]),
            )
        })?;

        if min_mhz_int >= max_mhz_int {
            eprintln!(
                "Invalid RFI range: min ({}) >= max ({}).",
                min_mhz_int, max_mhz_int
            );
            process::exit(1);
        }
        let min_chan = (min_mhz_int as f32 / rbw).floor() as usize;
        let max_chan = (max_mhz_int as f32 / rbw).ceil() as usize;
        ranges.push((min_chan, max_chan));
    }
    Ok(ranges)
}

#[derive(Debug, Clone)]
pub struct HistogramRfiResult {
    pub sigma: f32,
    pub sigma_mle: f64,
    pub sigma_initial: f64,
    /// Fitted Rayleigh normalization (effective noise samples).
    pub rayleigh_samples: f64,
    pub reduced_chi_square: f64,
    pub rayleigh_count: u64,
    pub threshold: f32,
    pub frequency_threshold: f32,
    /// Delay/rate coordinates of the selected fringe peak. Its entire delay
    /// column and rate row are protected from histogram/RFI masking.
    pub fringe_peak_rate: usize,
    pub fringe_peak_delay: usize,
    pub fringe_peak: f32,
    pub fringe_peak_bin_count: u64,
    pub samples_ge_peak: u64,
    /// Number of finite cells used by the Rayleigh fit (celestial excluded).
    pub valid_count: usize,
    /// Number of finite cells shown in the all-cell histograms.
    pub hist_valid_count: usize,
    pub candidate_count: usize,
    pub celestial_count: usize,
    pub delay_mask: Vec<bool>,
    pub frequency_mask: Vec<bool>,
    pub delay_shape: (usize, usize),
    pub frequency_shape: (usize, usize),
    pub hist_edges: Vec<f64>,
    pub hist_counts: Vec<u64>,
    pub log_edges: Vec<f64>,
    pub log_counts: Vec<u64>,
    pub expected_counts: Vec<f64>,
    pub log_expected_counts: Vec<f64>,
    pub rfi_hist_counts: Vec<u64>,
    pub rfi_log_counts: Vec<u64>,
    pub celestial_hist_counts: Vec<u64>,
    pub celestial_log_counts: Vec<u64>,
    pub delay_amplitudes: Vec<f32>,
    pub frequency_amplitudes: Vec<f32>,
    pub candidate_mask: Vec<bool>,
    pub celestial_mask: Vec<bool>,
    pub delay_axis: Vec<f64>,
    pub rate_axis: Vec<f64>,
    pub frequency_axis: Vec<f64>,
}

impl HistogramRfiResult {
    pub fn zero_delay_rate(&self, values: &mut Array2<C32>) -> usize {
        if values.dim() != self.delay_shape {
            return 0;
        }
        let Some(slice) = values.as_slice_mut() else {
            return 0;
        };
        let mut count = 0;
        for (index, marked) in self.delay_mask.iter().copied().enumerate() {
            if marked && slice[index] != C32::new(0.0, 0.0) {
                slice[index] = C32::new(0.0, 0.0);
                count += 1;
            }
        }
        count
    }
}

#[derive(Debug, Clone, Copy)]
struct RayleighFit {
    sigma: f64,
    sigma_mle: f64,
    sigma_initial: f64,
    sample_count: f64,
    reduced_chi_square: f64,
}

fn rayleigh_survival(value: f64, sigma: f64) -> f64 {
    if value <= 0.0 {
        1.0
    } else {
        (-value * value / (2.0 * sigma * sigma)).exp()
    }
}

fn rayleigh_bin_probability(lower: f64, upper: f64, sigma: f64) -> f64 {
    if !(sigma > 0.0) || !sigma.is_finite() || !(upper > lower) {
        return 0.0;
    }
    rayleigh_survival(lower.max(0.0), sigma) - rayleigh_survival(upper.max(0.0), sigma)
}

fn rayleigh_histogram_sse(edges: &[f64], counts: &[u64], log_sigma: f64, log_samples: f64) -> f64 {
    let sigma = log_sigma.exp();
    let samples = log_samples.exp();
    if !(sigma > 0.0) || !sigma.is_finite() || !(samples > 0.0) || !samples.is_finite() {
        return f64::INFINITY;
    }
    edges
        .windows(2)
        .zip(counts.iter())
        .map(|(window, &count)| {
            let expected = samples * rayleigh_bin_probability(window[0], window[1], sigma);
            let weight = if count > 1 {
                (count as f64).sqrt()
            } else {
                1.0
            };
            let residual = (count as f64 - expected) / weight;
            residual * residual
        })
        .sum()
}

/// Fit a Rayleigh distribution to binned amplitudes using the same model as
/// noise_hist: log(sigma) and log(sample_count) are fitted simultaneously,
/// with sqrt(observed-count) weighting and exact bin-integrated probabilities.
fn fit_rayleigh_histogram(edges: &[f64], counts: &[u64]) -> RayleighFit {
    let n_bins = counts.len().min(edges.len().saturating_sub(1));
    let total_count: f64 = counts.iter().take(n_bins).map(|&value| value as f64).sum();
    if n_bins == 0 || total_count <= 0.0 || !total_count.is_finite() {
        return RayleighFit {
            sigma: f64::NAN,
            sigma_mle: f64::NAN,
            sigma_initial: f64::NAN,
            sample_count: 0.0,
            reduced_chi_square: f64::NAN,
        };
    }

    let mut second_moment = 0.0;
    let mut mode_index = 0usize;
    let mut mode_count = 0u64;
    for (index, (&count, window)) in counts.iter().zip(edges.windows(2)).take(n_bins).enumerate() {
        let center = 0.5 * (window[0] + window[1]);
        second_moment += count as f64 * center * center;
        if count > mode_count {
            mode_count = count;
            mode_index = index;
        }
    }
    let sigma_mle = (second_moment / (2.0 * total_count)).sqrt();
    let sigma_mode = 0.5 * (edges[mode_index] + edges[mode_index + 1]);
    let sigma_initial = if sigma_mode.is_finite() && sigma_mode > 0.0 {
        sigma_mode
    } else {
        sigma_mle
    };
    if !(sigma_initial > 0.0) || !sigma_initial.is_finite() {
        return RayleighFit {
            sigma: sigma_mle,
            sigma_mle,
            sigma_initial,
            sample_count: total_count,
            reduced_chi_square: f64::NAN,
        };
    }

    let mut log_sigma = sigma_initial.ln();
    let mut log_samples = total_count.ln();
    let mut current_sse = rayleigh_histogram_sse(edges, counts, log_sigma, log_samples);
    let mut lambda = 1.0e-3;

    for _ in 0..500 {
        let sigma = log_sigma.exp();
        let samples = log_samples.exp();
        if !(sigma > 0.0) || !sigma.is_finite() || !(samples > 0.0) || !samples.is_finite() {
            break;
        }
        let sigma_squared = sigma * sigma;
        let mut h00 = 0.0;
        let mut h01 = 0.0;
        let mut h11 = 0.0;
        let mut g0 = 0.0;
        let mut g1 = 0.0;
        for (&count, window) in counts.iter().zip(edges.windows(2)).take(n_bins) {
            let lower = window[0].max(0.0);
            let upper = window[1].max(0.0);
            let lower_survival = rayleigh_survival(lower, sigma);
            let upper_survival = rayleigh_survival(upper, sigma);
            let probability = lower_survival - upper_survival;
            let expected = samples * probability;
            let derivative_probability = lower_survival * lower * lower / sigma_squared
                - upper_survival * upper * upper / sigma_squared;
            let derivative_sigma = samples * derivative_probability;
            let weight = if count > 1 {
                (count as f64).sqrt()
            } else {
                1.0
            };
            let residual = (count as f64 - expected) / weight;
            let jac_sigma = -derivative_sigma / weight;
            let jac_samples = -expected / weight;
            h00 += jac_sigma * jac_sigma;
            h01 += jac_sigma * jac_samples;
            h11 += jac_samples * jac_samples;
            g0 += jac_sigma * residual;
            g1 += jac_samples * residual;
        }
        let d00 = h00 * (1.0 + lambda) + 1.0e-30;
        let d11 = h11 * (1.0 + lambda) + 1.0e-30;
        let determinant = d00 * d11 - h01 * h01;
        if !determinant.is_finite() || determinant.abs() <= 1.0e-30 {
            break;
        }
        let step_sigma = (-g0 * d11 + h01 * g1) / determinant;
        let step_samples = (h01 * g0 - d00 * g1) / determinant;
        if !step_sigma.is_finite() || !step_samples.is_finite() {
            lambda *= 10.0;
            continue;
        }
        let trial_log_sigma = log_sigma + step_sigma;
        let trial_log_samples = log_samples + step_samples;
        let trial_sse = rayleigh_histogram_sse(edges, counts, trial_log_sigma, trial_log_samples);
        if trial_sse.is_finite() && trial_sse < current_sse {
            log_sigma = trial_log_sigma;
            log_samples = trial_log_samples;
            let improvement = current_sse - trial_sse;
            current_sse = trial_sse;
            lambda = (lambda * 0.3).max(1.0e-12);
            if step_sigma.abs().max(step_samples.abs()) < 1.0e-8
                || improvement < 1.0e-12 * current_sse.max(1.0)
            {
                break;
            }
        } else {
            lambda = (lambda * 10.0).min(1.0e30);
        }
    }

    let sigma = log_sigma.exp();
    let sample_count = log_samples.exp();
    let reduced_chi_square = current_sse / (n_bins.saturating_sub(1).max(1) as f64);
    RayleighFit {
        sigma,
        sigma_mle,
        sigma_initial,
        sample_count,
        reduced_chi_square,
    }
}

fn fit_rayleigh_values(values: &[f32], min_positive: f64, max_value: f64) -> RayleighFit {
    if values.is_empty() || !(min_positive > 0.0) || !(max_value >= min_positive) {
        return RayleighFit {
            sigma: f64::NAN,
            sigma_mle: f64::NAN,
            sigma_initial: f64::NAN,
            sample_count: 0.0,
            reduced_chi_square: f64::NAN,
        };
    }
    let log_min = min_positive.log10();
    let log_max = max_value.log10().max(log_min + 1.0e-9);
    let mut edges = Vec::with_capacity(HISTOGRAM_BINS + 1);
    for bin in 0..=HISTOGRAM_BINS {
        let fraction = bin as f64 / HISTOGRAM_BINS as f64;
        edges.push(10.0f64.powf(log_min + fraction * (log_max - log_min)));
    }
    let mut counts = vec![0u64; HISTOGRAM_BINS];
    for &value in values {
        let value = value as f64;
        if !value.is_finite() || value <= 0.0 {
            continue;
        }
        let bin =
            (((value.log10() - log_min) / (log_max - log_min)) * HISTOGRAM_BINS as f64) as usize;
        counts[bin.min(HISTOGRAM_BINS - 1)] += 1;
    }
    fit_rayleigh_histogram(&edges, &counts)
}

/// Detect RFI for the current integration window. Axes and dimensions are
/// derived from the current arrays, so no fixed-length NPZ mask is reused.
pub fn detect_histogram_rfi(
    freq_rate: &mut Array2<C32>,
    delay_rate: &Array2<C32>,
    protect_peak: Option<(usize, usize)>,
    // Frequency-rate row corresponding to the ordinary rate=0 spectrum.
    // It is protected from the derived RFI mask so a celestial spectrum is
    // never zeroed while inspecting the histogram result.
    protect_rate_row: Option<usize>,
    requested_rayleigh_count: u64,
) -> HistogramRfiResult {
    let (rate_count, delay_count) = delay_rate.dim();
    let (frequency_count, frequency_rate_count) = freq_rate.dim();
    let total = rate_count.saturating_mul(delay_count);
    let mut amplitudes = vec![0.0f32; total];
    let mut max_index = 0usize;
    let mut max_value = 0.0f32;
    for (index, value) in delay_rate.iter().enumerate() {
        let amplitude = value.norm();
        amplitudes[index] = amplitude;
        if amplitude.is_finite() && amplitude > max_value {
            max_value = amplitude;
            max_index = index;
        }
    }
    let seed = protect_peak
        .filter(|(rate, delay)| *rate < rate_count && *delay < delay_count)
        .map(|(rate, delay)| rate * delay_count + delay)
        .unwrap_or(max_index);
    let fringe_peak_rate = seed / delay_count.max(1);
    let fringe_peak_delay = seed % delay_count.max(1);
    let is_protected = |index: usize| {
        let rate = index / delay_count.max(1);
        let delay = index % delay_count.max(1);
        rate == fringe_peak_rate || delay == fringe_peak_delay
    };
    // The fringe-peak row and column are signal-bearing structures, not noise
    // samples. Exclude them from both the fit and displayed histograms.
    let mut valid = amplitudes
        .iter()
        .enumerate()
        .filter_map(|(index, amplitude)| {
            (!is_protected(index) && amplitude.is_finite()).then_some(*amplitude)
        })
        .collect::<Vec<_>>();
    valid.sort_by(|a, b| a.total_cmp(b));
    let mut valid_count = valid.len();
    let histogram_min_positive = valid
        .iter()
        .copied()
        .find(|value| *value > 0.0)
        .unwrap_or(1.0) as f64;
    let histogram_max = valid
        .last()
        .copied()
        .unwrap_or(histogram_min_positive as f32)
        .max(histogram_min_positive as f32) as f64;
    // Match the noise_hist tail-count convention while keeping every
    // integration window numerically valid if it has very few finite cells.
    let mut rayleigh_count = requested_rayleigh_count
        .max(1)
        .min(valid_count.saturating_sub(1).max(1) as u64);
    let mut rayleigh_count_f64 = rayleigh_count as f64;
    let mut sigma = if valid_count > 0 {
        let median = valid[valid_count / 2] as f64;
        (median / (2.0f64 * 2.0f64.ln()).sqrt()) as f32
    } else {
        0.0
    };
    if !sigma.is_finite() || sigma <= 0.0 {
        let second = valid
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>();
        sigma = if valid_count > 0 {
            (second / (2.0 * valid_count as f64)).sqrt() as f32
        } else {
            0.0
        };
    }
    let mut rayleigh_samples = valid_count as f64;
    let mut sigma_mle = f64::NAN;
    let mut sigma_initial = f64::NAN;
    let mut reduced_chi_square = f64::NAN;
    for _ in 0..3 {
        if sigma <= 0.0 || !sigma.is_finite() {
            break;
        }
        let tail = (2.0 * ((valid_count.max(2) as f64) / rayleigh_count_f64).ln()).sqrt();
        let cutoff = sigma as f64 * tail;
        let below = valid
            .iter()
            .filter(|value| (**value as f64) < cutoff)
            .collect::<Vec<_>>();
        if below.len() < 16 {
            break;
        }
        let second = below
            .iter()
            .map(|value| (**value as f64).powi(2))
            .sum::<f64>();
        sigma = (second / (2.0 * below.len() as f64)).sqrt() as f32;
    }
    let tail_sigma = if valid_count > 1 {
        (2.0 * ((valid_count as f64) / rayleigh_count_f64).ln()).sqrt()
    } else {
        0.0
    };
    let mut threshold = (sigma as f64 * tail_sigma).max(0.0) as f32;
    let mut delay_mask = vec![false; total];
    for (index, amplitude) in amplitudes.iter().copied().enumerate() {
        delay_mask[index] = threshold > 0.0 && amplitude.is_finite() && amplitude >= threshold;
    }

    let mut celestial = vec![false; total];
    if seed < total && delay_mask.get(seed).copied().unwrap_or(false) {
        let mut queue = vec![seed];
        celestial[seed] = true;
        let mut head = 0usize;
        while head < queue.len() {
            let index = queue[head];
            head += 1;
            let row = index / delay_count.max(1);
            let col = index % delay_count.max(1);
            let r0 = row.saturating_sub(1);
            let r1 = (row + 1).min(rate_count.saturating_sub(1));
            let c0 = col.saturating_sub(1);
            let c1 = (col + 1).min(delay_count.saturating_sub(1));
            for neighbour_row in r0..=r1 {
                for neighbour_col in c0..=c1 {
                    let neighbour = neighbour_row * delay_count + neighbour_col;
                    if delay_mask[neighbour] && !is_protected(neighbour) && !celestial[neighbour] {
                        celestial[neighbour] = true;
                        queue.push(neighbour);
                    }
                }
            }
        }
    }
    // The connected component around the strongest fringe peak is the
    // celestial signal. The finite-band sinc response occupies the complete
    // delay column and rate row through the peak, so mark that cross as
    // celestial and keep it out of the Rayleigh/RFI statistics.
    for index in 0..total {
        if is_protected(index) {
            celestial[index] = true;
        }
    }
    // It must not participate in the Rayleigh fit or in
    // the tail-count threshold; otherwise a bright source inflates sigma and
    // hides genuine RFI.  Keep `valid` (all finite cells) for the displayed
    // histogram, but refit using only the non-celestial cells.
    let mut fit_valid = amplitudes
        .iter()
        .enumerate()
        .filter_map(|(index, amplitude)| {
            (!celestial[index] && !is_protected(index) && amplitude.is_finite())
                .then_some(*amplitude)
        })
        .collect::<Vec<_>>();
    if fit_valid.is_empty() {
        fit_valid = valid.clone();
    }
    fit_valid.sort_by(|a, b| a.total_cmp(b));
    valid_count = fit_valid.len();
    rayleigh_count = requested_rayleigh_count
        .max(1)
        .min(valid_count.saturating_sub(1).max(1) as u64);
    rayleigh_count_f64 = rayleigh_count as f64;
    let fit = fit_rayleigh_values(&fit_valid, histogram_min_positive, histogram_max);
    if fit.sigma.is_finite() && fit.sigma > 0.0 {
        sigma = fit.sigma as f32;
        sigma_mle = fit.sigma_mle;
        sigma_initial = fit.sigma_initial;
        rayleigh_samples = fit.sample_count;
        reduced_chi_square = fit.reduced_chi_square;
    }
    let tail_sigma = if valid_count > 1 {
        (2.0 * ((valid_count as f64) / rayleigh_count_f64).ln()).sqrt()
    } else {
        0.0
    };
    threshold = (sigma as f64 * tail_sigma).max(0.0) as f32;
    delay_mask.fill(false);
    for (index, amplitude) in amplitudes.iter().copied().enumerate() {
        delay_mask[index] = threshold > 0.0 && amplitude.is_finite() && amplitude >= threshold;
    }
    let mut candidate_mask = delay_mask.clone();
    for index in 0..total {
        if celestial[index] {
            // Keep the source visible as a separate class even if the final
            // noise-only threshold is higher than its initial classification.
            candidate_mask[index] = true;
            delay_mask[index] = false;
        }
        if is_protected(index) {
            // Never classify or remove the celestial fringe cross. Its row
            // and column are marked celestial and excluded from RFI masks.
            candidate_mask[index] = false;
            delay_mask[index] = false;
        }
    }

    let mut rfi_frequency = Array2::<C32>::zeros((frequency_count, frequency_rate_count));
    if rate_count == frequency_rate_count && delay_count > 0 && frequency_count > 0 {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(delay_count);
        let half = delay_count / 2;
        for rate in 0..rate_count {
            let mut work = vec![C32::new(0.0, 0.0); delay_count];
            for delay in 0..delay_count {
                if delay_mask[rate * delay_count + delay] {
                    let source = if delay < half {
                        half.saturating_sub(1 + delay)
                    } else {
                        delay_count - 1 - (delay - half)
                    };
                    work[source] = delay_rate[[rate, delay]];
                }
            }
            fft.process(&mut work);
            for frequency in 0..frequency_count.min(delay_count) {
                rfi_frequency[[frequency, rate]] = work[frequency];
            }
        }
    }
    let mut transformed = rfi_frequency
        .iter()
        .map(|value| value.norm())
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    transformed.sort_by(|a, b| a.total_cmp(b));
    let transformed_median = transformed
        .get(transformed.len() / 2)
        .copied()
        .unwrap_or(0.0);
    let transformed_max = transformed.last().copied().unwrap_or(0.0);
    let frequency_threshold = if transformed_median > 0.0 {
        (transformed_median * 3.0).max(transformed_max * 0.02)
    } else {
        transformed_max * 0.02
    };
    let frequency_amplitudes = rfi_frequency
        .iter()
        .map(|value| value.norm())
        .collect::<Vec<_>>();
    let mut frequency_mask = vec![false; frequency_count.saturating_mul(frequency_rate_count)];
    for (index, value) in rfi_frequency.iter().enumerate() {
        let rate_row = index % frequency_rate_count.max(1);
        frequency_mask[index] = frequency_threshold > 0.0
            && value.norm().is_finite()
            && value.norm() >= frequency_threshold
            && Some(rate_row) != protect_rate_row;
    }
    if let Some(slice) = freq_rate.as_slice_mut() {
        for (index, marked) in frequency_mask.iter().copied().enumerate() {
            if marked {
                slice[index] = C32::new(0.0, 0.0);
            }
        }
    }

    let min_positive = histogram_min_positive;
    let max_hist = histogram_max;
    let log_min = min_positive.log10();
    let log_max = max_hist.log10().max(log_min + 1.0e-9);
    let mut hist_edges = Vec::with_capacity(HISTOGRAM_BINS + 1);
    let mut log_edges = Vec::with_capacity(HISTOGRAM_BINS + 1);
    for bin in 0..=HISTOGRAM_BINS {
        let fraction = bin as f64 / HISTOGRAM_BINS as f64;
        hist_edges.push(fraction * max_hist as f64);
        log_edges.push(10.0f64.powf(log_min + fraction * (log_max - log_min)));
    }
    let mut hist_counts = vec![0u64; HISTOGRAM_BINS];
    let mut log_counts = vec![0u64; HISTOGRAM_BINS];
    for amplitude in valid.iter().copied() {
        let linear_bin = ((amplitude as f64 / (max_hist as f64).max(1.0e-30)
            * HISTOGRAM_BINS as f64) as usize)
            .min(HISTOGRAM_BINS - 1);
        hist_counts[linear_bin] += 1;
        let log_bin = (((amplitude as f64).log10() - log_min) / (log_max - log_min).max(1.0e-30)
            * HISTOGRAM_BINS as f64) as usize;
        log_counts[log_bin.min(HISTOGRAM_BINS - 1)] += 1;
    }
    let fringe_peak = amplitudes.get(seed).copied().unwrap_or(max_value);
    let samples_ge_peak = amplitudes
        .iter()
        .filter(|value| value.is_finite() && **value >= fringe_peak)
        .count() as u64;
    let fringe_peak_log_bin = if fringe_peak > 0.0 && fringe_peak.is_finite() {
        (((((fringe_peak as f64).log10() - log_min) / (log_max - log_min).max(1.0e-30))
            * HISTOGRAM_BINS as f64) as usize)
            .min(HISTOGRAM_BINS - 1)
    } else {
        0
    };
    let fringe_peak_bin_count = log_counts.get(fringe_peak_log_bin).copied().unwrap_or(0);
    let rayleigh_cdf = |x: f64| {
        let s = sigma.max(f32::EPSILON) as f64;
        1.0 - (-x * x / (2.0 * s * s)).exp()
    };
    let expected_counts = hist_edges
        .windows(2)
        .map(|window| rayleigh_samples * (rayleigh_cdf(window[1]) - rayleigh_cdf(window[0])))
        .collect::<Vec<_>>();
    let log_expected_counts = log_edges
        .windows(2)
        .map(|window| rayleigh_samples * (rayleigh_cdf(window[1]) - rayleigh_cdf(window[0])))
        .collect::<Vec<_>>();
    let mut rfi_hist_counts = vec![0u64; HISTOGRAM_BINS];
    let mut rfi_log_counts = vec![0u64; HISTOGRAM_BINS];
    let mut celestial_hist_counts = vec![0u64; HISTOGRAM_BINS];
    let mut celestial_log_counts = vec![0u64; HISTOGRAM_BINS];
    for (index, amplitude) in amplitudes.iter().copied().enumerate() {
        if is_protected(index) || !amplitude.is_finite() {
            continue;
        }
        let linear_bin = ((amplitude as f64 / (max_hist as f64).max(1.0e-30)
            * HISTOGRAM_BINS as f64) as usize)
            .min(HISTOGRAM_BINS - 1);
        let log_bin = (((amplitude as f64).log10() - log_min) / (log_max - log_min).max(1.0e-30)
            * HISTOGRAM_BINS as f64) as usize;
        let log_bin = log_bin.min(HISTOGRAM_BINS - 1);
        if celestial[index] {
            celestial_hist_counts[linear_bin] += 1;
            celestial_log_counts[log_bin] += 1;
        } else if candidate_mask[index] {
            rfi_hist_counts[linear_bin] += 1;
            rfi_log_counts[log_bin] += 1;
        }
    }
    let candidate_count = delay_mask.iter().filter(|value| **value).count();
    let celestial_count = celestial.iter().filter(|value| **value).count();
    HistogramRfiResult {
        sigma,
        sigma_mle,
        sigma_initial,
        rayleigh_samples,
        reduced_chi_square,
        rayleigh_count,
        threshold,
        frequency_threshold,
        fringe_peak_rate,
        fringe_peak_delay,
        fringe_peak,
        fringe_peak_bin_count,
        samples_ge_peak,
        valid_count,
        hist_valid_count: valid.len(),
        candidate_count,
        celestial_count,
        delay_mask,
        frequency_mask,
        delay_shape: (rate_count, delay_count),
        frequency_shape: (frequency_count, frequency_rate_count),
        hist_edges,
        hist_counts,
        log_edges,
        log_counts,
        expected_counts,
        log_expected_counts,
        rfi_hist_counts,
        rfi_log_counts,
        celestial_hist_counts,
        celestial_log_counts,
        delay_amplitudes: amplitudes,
        frequency_amplitudes,
        candidate_mask,
        celestial_mask: celestial,
        delay_axis: (0..delay_count).map(|value| value as f64).collect(),
        rate_axis: (0..rate_count).map(|value| value as f64).collect(),
        frequency_axis: (0..frequency_count).map(|value| value as f64).collect(),
    }
}

fn rayleigh_annotation(result: &HistogramRfiResult, bins_label: &str) -> Vec<String> {
    let peak_percent = if result.hist_valid_count > 0 {
        100.0 * result.samples_ge_peak as f64 / result.hist_valid_count as f64
    } else {
        0.0
    };
    vec![
        format!("Samples               : {}", result.hist_valid_count),
        format!("Valid samples         : {}", result.valid_count),
        format!("Bins                  : {} ({bins_label})", HISTOGRAM_BINS),
        format!("Fringe-peak bin count : {}", result.fringe_peak_bin_count),
        format!(
            "Samples >= peak       : {} ({peak_percent:.4}%)",
            result.samples_ge_peak
        ),
        format!("Fringe peak           : {:.6e}", result.fringe_peak),
        format!("Rayleigh sigma        : {:.6e}", result.sigma),
        format!("Rayleigh sigma MLE    : {:.6e}", result.sigma_mle),
        format!("Rayleigh sigma init   : {:.6e}", result.sigma_initial),
        format!("Rayleigh reduced chi2 : {:.6e}", result.reduced_chi_square),
        format!("Rayleigh tail count   : {}", result.rayleigh_count),
        format!("RFI tail threshold    : {:.6e}", result.threshold),
        format!("RFI candidates        : {}", result.candidate_count),
        format!("Celestial candidates  : {}", result.celestial_count),
    ]
}

pub fn write_histogram_products(
    output_dir: &Path,
    basename: &str,
    result: &HistogramRfiResult,
) -> Result<(PathBuf, PathBuf, PathBuf), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;
    let stem = format!("{basename}_hist");
    let png_path = output_dir.join(format!("{stem}.png"));
    let tsv_path = output_dir.join(format!("{stem}.tsv"));
    let npz_path = output_dir.join(format!("{stem}.npz"));
    let root = BitMapBackend::new(&png_path, (1600, 1000)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((2, 2));
    let annotation_linear = rayleigh_annotation(result, "linear");
    let annotation_logx = rayleigh_annotation(result, "log10");
    let annotation_logy = rayleigh_annotation(result, "linear");
    let annotation_logxy = rayleigh_annotation(result, "log10");
    draw_histogram_panel(
        &panels[0],
        "linear",
        &result.hist_edges,
        &result.hist_counts,
        &result.expected_counts,
        &result.rfi_hist_counts,
        &result.celestial_hist_counts,
        result.threshold as f64,
        &annotation_linear,
        false,
        false,
    )?;
    draw_histogram_panel(
        &panels[1],
        "log-x",
        &result.log_edges,
        &result.log_counts,
        &result.log_expected_counts,
        &result.rfi_log_counts,
        &result.celestial_log_counts,
        result.threshold as f64,
        &annotation_logx,
        true,
        false,
    )?;
    draw_histogram_panel(
        &panels[2],
        "log-y",
        &result.hist_edges,
        &result.hist_counts,
        &result.expected_counts,
        &result.rfi_hist_counts,
        &result.celestial_hist_counts,
        result.threshold as f64,
        &annotation_logy,
        false,
        true,
    )?;
    draw_histogram_panel(
        &panels[3],
        "log-x / log-y",
        &result.log_edges,
        &result.log_counts,
        &result.log_expected_counts,
        &result.rfi_log_counts,
        &result.celestial_log_counts,
        result.threshold as f64,
        &annotation_logxy,
        true,
        true,
    )?;
    root.present()?;
    drop(panels);
    drop(root);
    crate::png_compress::compress_png_with_mode(
        &png_path,
        crate::png_compress::CompressQuality::High,
    );

    // Keep the individual files used by the Zig implementation as well as
    // the compact four-panel overview above.  In particular, log-x and
    // log-x/log-y retain the Rayleigh fit curve.
    let individual = [
        (
            "linear",
            &result.hist_edges,
            &result.hist_counts,
            &result.expected_counts,
            &result.rfi_hist_counts,
            &result.celestial_hist_counts,
            result.threshold as f64,
            false,
            false,
        ),
        (
            "logy",
            &result.hist_edges,
            &result.hist_counts,
            &result.expected_counts,
            &result.rfi_hist_counts,
            &result.celestial_hist_counts,
            result.threshold as f64,
            false,
            true,
        ),
        (
            "logx",
            &result.log_edges,
            &result.log_counts,
            &result.log_expected_counts,
            &result.rfi_log_counts,
            &result.celestial_log_counts,
            result.threshold as f64,
            true,
            false,
        ),
        (
            "logxy",
            &result.log_edges,
            &result.log_counts,
            &result.log_expected_counts,
            &result.rfi_log_counts,
            &result.celestial_log_counts,
            result.threshold as f64,
            true,
            true,
        ),
    ];
    for (suffix, edges, counts, expected, rfi_counts, celestial_counts, threshold, log_x, log_y) in
        individual
    {
        let path = output_dir.join(format!("{basename}_hist_{suffix}.png"));
        let annotation = rayleigh_annotation(result, if log_x { "log10" } else { "linear" });
        write_histogram_png(
            &path,
            suffix,
            edges,
            counts,
            expected,
            rfi_counts,
            celestial_counts,
            threshold,
            &annotation,
            log_x,
            log_y,
        )?;
    }
    write_histogram_heatmap(
        &output_dir.join(format!("{basename}_hist_imshow.png")),
        result,
        false,
    )?;
    write_histogram_heatmap(
        &output_dir.join(format!("{basename}_hist_imshow_rfi.png")),
        result,
        true,
    )?;

    let mut file = File::create(&tsv_path)?;
    writeln!(file, "# RFI histogram analysis")?;
    writeln!(
        file,
        "# sigma\tsigma_mle\tsigma_initial\trayleigh_samples\treduced_chi_square\trayleigh_count\tthreshold\tfrequency_threshold\tfringe_peak_rate_index\tfringe_peak_delay_index\tfringe_peak\tfringe_peak_bin_count\tsamples_ge_peak\tfit_valid_count\thistogram_valid_count\tdelay_rfi_count\tcelestial_count"
    )?;
    writeln!(
        file,
        "# {:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{}\t{:.8e}\t{:.8e}\t{}\t{}\t{:.8e}\t{}\t{}\t{}\t{}\t{}\t{}",
        result.sigma,
        result.sigma_mle,
        result.sigma_initial,
        result.rayleigh_samples,
        result.reduced_chi_square,
        result.rayleigh_count,
        result.threshold,
        result.frequency_threshold,
        result.fringe_peak_rate,
        result.fringe_peak_delay,
        result.fringe_peak,
        result.fringe_peak_bin_count,
        result.samples_ge_peak,
        result.valid_count,
        result.hist_valid_count,
        result.candidate_count,
        result.celestial_count
    )?;
    writeln!(
        file,
        "# bin_left\tbin_right\tcount\trfi_count\tcelestial_count\trayleigh_expected"
    )?;
    for (index, count) in result.hist_counts.iter().enumerate() {
        let left = result.hist_edges.get(index).copied().unwrap_or(0.0);
        let right = result.hist_edges.get(index + 1).copied().unwrap_or(left);
        let expected = result.expected_counts.get(index).copied().unwrap_or(0.0);
        let rfi_count = result.rfi_hist_counts.get(index).copied().unwrap_or(0);
        let celestial_count = result
            .celestial_hist_counts
            .get(index)
            .copied()
            .unwrap_or(0);
        writeln!(
            file,
            "{left:.8e}\t{right:.8e}\t{count}\t{rfi_count}\t{celestial_count}\t{expected:.8e}"
        )?;
    }
    writeln!(
        file,
        "# log10_bin_left\tlog10_bin_right\tcount\trfi_count\tcelestial_count\trayleigh_expected"
    )?;
    for (index, count) in result.log_counts.iter().enumerate() {
        let left = result.log_edges.get(index).copied().unwrap_or(1.0).log10();
        let right = result
            .log_edges
            .get(index + 1)
            .copied()
            .unwrap_or(result.log_edges.get(index).copied().unwrap_or(1.0))
            .log10();
        let expected = result
            .log_expected_counts
            .get(index)
            .copied()
            .unwrap_or(0.0);
        let rfi_count = result.rfi_log_counts.get(index).copied().unwrap_or(0);
        let celestial_count = result.celestial_log_counts.get(index).copied().unwrap_or(0);
        writeln!(
            file,
            "{left:.8e}\t{right:.8e}\t{count}\t{rfi_count}\t{celestial_count}\t{expected:.8e}"
        )?;
    }
    let mut npz = NamedNpz::new(NpyMeta::new(
        "rfi_histogram",
        result.frequency_shape.0 as u32 * 2,
        result.delay_shape.0 as u32,
    ));
    npz.add_f64_1d("hist_edges", &result.hist_edges);
    npz.add_f64_1d(
        "hist_counts",
        &result
            .hist_counts
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>(),
    );
    npz.add_f64_1d("log_edges", &result.log_edges);
    npz.add_f64_1d(
        "log_counts",
        &result
            .log_counts
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>(),
    );
    npz.add_f64_1d("rayleigh_expected", &result.expected_counts);
    npz.add_f64_1d("rayleigh_expected_log", &result.log_expected_counts);
    npz.add_f64_1d(
        "rfi_hist_counts",
        &result
            .rfi_hist_counts
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>(),
    );
    npz.add_f64_1d(
        "rfi_log_counts",
        &result
            .rfi_log_counts
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>(),
    );
    npz.add_f64_1d(
        "celestial_hist_counts",
        &result
            .celestial_hist_counts
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>(),
    );
    npz.add_f64_1d(
        "celestial_log_counts",
        &result
            .celestial_log_counts
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>(),
    );
    npz.add_f64_1d("rayleigh_count", &[result.rayleigh_count as f64]);
    npz.add_f64_1d("rayleigh_sigma_mle", &[result.sigma_mle]);
    npz.add_f64_1d("rayleigh_sigma_initial", &[result.sigma_initial]);
    npz.add_f64_1d("rayleigh_samples", &[result.rayleigh_samples]);
    npz.add_f64_1d("rayleigh_reduced_chi_square", &[result.reduced_chi_square]);
    npz.add_f64_1d("fit_valid_count", &[result.valid_count as f64]);
    npz.add_f64_1d("histogram_valid_count", &[result.hist_valid_count as f64]);
    npz.add_f64_1d("rayleigh_sigma", &[result.sigma as f64]);
    npz.add_f64_1d("rfi_threshold", &[result.threshold as f64]);
    npz.add_f64_1d(
        "rfi_frequency_threshold",
        &[result.frequency_threshold as f64],
    );
    npz.add_f64_1d("fringe_peak_rate_index", &[result.fringe_peak_rate as f64]);
    npz.add_f64_1d(
        "fringe_peak_delay_index",
        &[result.fringe_peak_delay as f64],
    );
    npz.add_f64_1d("fringe_peak", &[result.fringe_peak as f64]);
    npz.add_f64_1d(
        "fringe_peak_bin_count",
        &[result.fringe_peak_bin_count as f64],
    );
    npz.add_f64_1d("samples_ge_peak", &[result.samples_ge_peak as f64]);
    npz.add_f64_1d("delay_axis", &result.delay_axis);
    npz.add_f64_1d("rate_hz", &result.rate_axis);
    npz.add_f64_1d("frequency_axis", &result.frequency_axis);
    npz.add_f32_1d("delay_amplitude", &result.delay_amplitudes);
    npz.add_f32_1d("frequency_amplitude", &result.frequency_amplitudes);
    npz.add_u8_2d(
        "delay_rfi_mask",
        result.delay_shape,
        result.delay_mask.iter().map(|value| *value as u8),
    )?;
    npz.add_u8_2d(
        "delay_candidate_mask",
        result.delay_shape,
        result.candidate_mask.iter().map(|value| *value as u8),
    )?;
    npz.add_u8_2d(
        "delay_celestial_mask",
        result.delay_shape,
        result.celestial_mask.iter().map(|value| *value as u8),
    )?;
    npz.add_u8_2d(
        "frequency_rfi_mask",
        result.frequency_shape,
        result.frequency_mask.iter().map(|value| *value as u8),
    )?;
    npz.write(&npz_path)?;
    Ok((png_path, tsv_path, npz_path))
}

fn write_histogram_png(
    path: &Path,
    title: &str,
    edges: &[f64],
    counts: &[u64],
    expected: &[f64],
    rfi_counts: &[u64],
    celestial_counts: &[u64],
    threshold: f64,
    annotation: &[String],
    log_x: bool,
    log_y: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (1280, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    draw_histogram_panel(
        &root,
        title,
        edges,
        counts,
        expected,
        rfi_counts,
        celestial_counts,
        threshold,
        annotation,
        log_x,
        log_y,
    )?;
    root.present()?;
    drop(root);
    crate::png_compress::compress_png_with_mode(path, crate::png_compress::CompressQuality::High);
    Ok(())
}

/// Return the physical boundaries around one sampled axis coordinate.  The
/// histogram products use cell centers, while a heatmap needs finite edges.
fn histogram_cell_bounds(axis: &[f64], index: usize) -> (f64, f64) {
    let center = axis.get(index).copied().unwrap_or(index as f64);
    let previous = index
        .checked_sub(1)
        .and_then(|value| axis.get(value).copied())
        .filter(|value| value.is_finite());
    let next = axis
        .get(index + 1)
        .copied()
        .filter(|value| value.is_finite());
    let default_step = 1.0;
    let left_step = previous
        .map(|value| (center - value).abs())
        .filter(|value| *value > f64::EPSILON)
        .or_else(|| {
            next.map(|value| (value - center).abs())
                .filter(|value| *value > f64::EPSILON)
        })
        .unwrap_or(default_step);
    let right_step = next
        .map(|value| (value - center).abs())
        .filter(|value| *value > f64::EPSILON)
        .or_else(|| {
            previous
                .map(|value| (center - value).abs())
                .filter(|value| *value > f64::EPSILON)
        })
        .unwrap_or(default_step);
    let mut left = previous
        .map(|value| 0.5 * (value + center))
        .unwrap_or(center - 0.5 * left_step);
    let mut right = next
        .map(|value| 0.5 * (value + center))
        .unwrap_or(center + 0.5 * right_step);
    if right <= left {
        let half = 0.5 * default_step;
        left = center - half;
        right = center + half;
    }
    (left, right)
}

fn write_histogram_heatmap(
    path: &Path,
    result: &HistogramRfiResult,
    classified: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (rows, cols) = result.delay_shape;
    if rows == 0 || cols == 0 || result.delay_amplitudes.len() != rows.saturating_mul(cols) {
        return Ok(());
    }
    let x_axis = if result.delay_axis.len() == cols {
        result.delay_axis.as_slice()
    } else {
        // This fallback is only used when a caller writes a result before
        // attaching physical search axes.
        &result.delay_axis
    };
    let y_axis = if result.rate_axis.len() == rows {
        result.rate_axis.as_slice()
    } else {
        &result.rate_axis
    };
    let x_min = x_axis.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = x_axis.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y_axis.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = y_axis.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !x_min.is_finite() || !x_max.is_finite() || !y_min.is_finite() || !y_max.is_finite() {
        return Ok(());
    }
    let x_pad = if (x_max - x_min).abs() > f64::EPSILON {
        (x_max - x_min).abs() * 0.01
    } else {
        0.5
    };
    let y_pad = if (y_max - y_min).abs() > f64::EPSILON {
        (y_max - y_min).abs() * 0.01
    } else {
        0.5
    };
    let mut log_min = f32::INFINITY;
    let mut log_max = f32::NEG_INFINITY;
    for &amplitude in &result.delay_amplitudes {
        if amplitude.is_finite() && amplitude > 0.0 {
            let value = amplitude.log10();
            log_min = log_min.min(value);
            log_max = log_max.max(value);
        }
    }
    if !log_min.is_finite() || !log_max.is_finite() {
        log_min = -12.0;
        log_max = 0.0;
    }
    if (log_max - log_min).abs() < f32::EPSILON {
        log_min -= 0.5;
        log_max += 0.5;
    }

    let root = BitMapBackend::new(path, (1500, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let title = if classified {
        "Delay-rate amplitude: RFI/celestial classification"
    } else {
        "Delay-rate amplitude (all cells)"
    };
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(title, ("sans-serif", 30))
        .x_label_area_size(65)
        .y_label_area_size(85)
        .build_cartesian_2d(
            (x_min - x_pad)..(x_max + x_pad),
            (y_min - y_pad)..(y_max + y_pad),
        )?;
    chart
        .configure_mesh()
        .x_desc("Delay")
        .y_desc("Rate [Hz]")
        .label_style(("sans-serif", 20))
        .axis_desc_style(("sans-serif", 24))
        .draw()?;

    let amplitudes = &result.delay_amplitudes;
    chart.draw_series((0..rows).flat_map(|row| {
        (0..cols).map(move |col| {
            let index = row * cols + col;
            let value = amplitudes.get(index).copied().unwrap_or(0.0).max(0.0);
            let value_log = if value > 0.0 { value.log10() } else { log_min };
            let normalized = ((value_log - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
            let (left, right) = histogram_cell_bounds(x_axis, col);
            let (bottom, top) = histogram_cell_bounds(y_axis, row);
            Rectangle::new(
                [(left, bottom), (right, top)],
                HSLColor((240.0 - 240.0 * normalized as f64) / 360.0, 1.0, 0.5).filled(),
            )
        })
    }))?;
    if classified {
        let candidate = &result.candidate_mask;
        let celestial = &result.celestial_mask;
        chart
            .draw_series((0..rows).flat_map(|row| {
                (0..cols).filter_map(move |col| {
                    let index = row * cols + col;
                    if !candidate.get(index).copied().unwrap_or(false)
                        || celestial.get(index).copied().unwrap_or(false)
                    {
                        return None;
                    }
                    let (left, right) = histogram_cell_bounds(x_axis, col);
                    let (bottom, top) = histogram_cell_bounds(y_axis, row);
                    Some(Rectangle::new(
                        [(left, bottom), (right, top)],
                        RED.stroke_width(2),
                    ))
                })
            }))?
            .label("RFI candidates")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], RED.stroke_width(2)));
        chart
            .draw_series((0..rows).flat_map(|row| {
                (0..cols).filter_map(move |col| {
                    let index = row * cols + col;
                    if !celestial.get(index).copied().unwrap_or(false) {
                        return None;
                    }
                    let (left, right) = histogram_cell_bounds(x_axis, col);
                    let (bottom, top) = histogram_cell_bounds(y_axis, row);
                    Some(Rectangle::new(
                        [(left, bottom), (right, top)],
                        CYAN.stroke_width(2),
                    ))
                })
            }))?
            .label("Celestial fringe cross")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], CYAN.stroke_width(2)));
        let annotation = rayleigh_annotation(result, "delay-rate");
        let box_left = x_min + (x_max - x_min) * 0.02;
        let box_right = x_min + (x_max - x_min) * 0.36;
        let line_step = ((y_max - y_min) / (annotation.len() as f64 + 3.0)).max(1.0e-6);
        let box_top = y_max - (y_max - y_min) * 0.02;
        let box_bottom = (box_top - line_step * (annotation.len() as f64 + 1.0)).max(y_min);
        chart.draw_series(std::iter::once(Rectangle::new(
            [(box_left, box_bottom), (box_right, box_top)],
            WHITE.mix(0.82).filled(),
        )))?;
        for (index, line) in annotation.iter().enumerate() {
            let y = box_top - line_step * (index as f64 + 1.0);
            chart.draw_series(std::iter::once(Text::new(
                line.clone(),
                (box_left + (x_max - x_min) * 0.008, y),
                ("monospace", 13).into_font().color(&BLACK),
            )))?;
        }
        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .background_style(WHITE.mix(0.8))
            .draw()?;
    }
    root.present()?;
    drop(chart);
    drop(root);
    crate::png_compress::compress_png_with_mode(path, crate::png_compress::CompressQuality::High);
    Ok(())
}

fn draw_histogram_panel(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    title: &str,
    edges: &[f64],
    counts: &[u64],
    expected: &[f64],
    rfi_counts: &[u64],
    celestial_counts: &[u64],
    threshold: f64,
    annotation: &[String],
    log_x: bool,
    log_y: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if edges.len() < 2 || counts.is_empty() {
        return Ok(());
    }
    let tx = |value: f64| {
        if log_x {
            value.max(1.0e-30).log10()
        } else {
            value
        }
    };
    let ty = |value: f64| {
        if log_y {
            (value.max(0.0) + 1.0).log10()
        } else {
            value
        }
    };
    let x0 = tx(edges[0]);
    let x1 = tx(*edges.last().unwrap_or(&edges[0]));
    let ymax = counts
        .iter()
        .chain(rfi_counts.iter())
        .chain(celestial_counts.iter())
        .map(|value| ty(*value as f64))
        .fold(0.0, f64::max)
        .max(1.0)
        * 1.15;
    let mut chart = ChartBuilder::on(area)
        .margin(24)
        .caption(format!("Rayleigh histogram: {title}"), ("sans-serif", 28))
        .x_label_area_size(55)
        .y_label_area_size(70)
        .build_cartesian_2d(x0..x1.max(x0 + 1.0e-9), 0.0..ymax)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc(if log_x {
            "log10 amplitude"
        } else {
            "amplitude"
        })
        .y_desc(if log_y { "log10(count+1)" } else { "count" })
        .label_style(("sans-serif", 18))
        .draw()?;
    chart
        .draw_series(edges.windows(2).zip(counts.iter()).map(|(window, count)| {
            Rectangle::new(
                [(tx(window[0]), 0.0), (tx(window[1]), ty(*count as f64))],
                BLUE.mix(0.45).filled(),
            )
        }))?
        .label("All cells")
        .legend(|(x, y)| Rectangle::new([(x, y - 4), (x + 18, y + 4)], BLUE.mix(0.45).filled()));
    if rfi_counts.len() == counts.len() {
        chart
            .draw_series(
                edges
                    .windows(2)
                    .zip(rfi_counts.iter())
                    .map(|(window, count)| {
                        Rectangle::new(
                            [(tx(window[0]), 0.0), (tx(window[1]), ty(*count as f64))],
                            RGBColor(255, 102, 0).mix(0.75).filled(),
                        )
                    }),
            )?
            .label("RFI candidates")
            .legend(|(x, y)| {
                Rectangle::new(
                    [(x, y - 4), (x + 18, y + 4)],
                    RGBColor(255, 102, 0).mix(0.75).filled(),
                )
            });
    }
    if celestial_counts.len() == counts.len() {
        chart
            .draw_series(
                edges
                    .windows(2)
                    .zip(celestial_counts.iter())
                    .map(|(window, count)| {
                        Rectangle::new(
                            [(tx(window[0]), 0.0), (tx(window[1]), ty(*count as f64))],
                            CYAN.mix(0.75).filled(),
                        )
                    }),
            )?
            .label("Known celestial")
            .legend(|(x, y)| {
                Rectangle::new([(x, y - 4), (x + 18, y + 4)], CYAN.mix(0.75).filled())
            });
    }
    if expected.len() == counts.len() {
        chart
            .draw_series(LineSeries::new(
                edges
                    .windows(2)
                    .zip(expected.iter())
                    .map(|(window, value)| (tx(0.5 * (window[0] + window[1])), ty(*value))),
                RED.stroke_width(3),
            ))?
            .label("Rayleigh fit")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));
    }
    if threshold.is_finite()
        && threshold > 0.0
        && threshold >= edges[0]
        && threshold <= *edges.last().unwrap_or(&edges[0])
    {
        let threshold_x = tx(threshold);
        chart
            .draw_series(LineSeries::new(
                [(threshold_x, 0.0), (threshold_x, ymax)],
                BLACK.stroke_width(3),
            ))?
            .label("RFI threshold")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.stroke_width(3)));
    }
    if !annotation.is_empty() {
        let box_left = x0 + (x1 - x0) * 0.02;
        let box_right = x0 + (x1 - x0) * 0.43;
        let line_step = (ymax / (annotation.len() as f64 + 3.0)).max(1.0);
        let box_top = ymax * 0.98;
        let box_bottom = (box_top - line_step * (annotation.len() as f64 + 1.0)).max(0.0);
        chart.draw_series(std::iter::once(Rectangle::new(
            [(box_left, box_bottom), (box_right, box_top)],
            WHITE.mix(0.82).filled(),
        )))?;
        for (index, line) in annotation.iter().enumerate() {
            let y = box_top - line_step * (index as f64 + 1.0);
            chart.draw_series(std::iter::once(Text::new(
                line.clone(),
                (box_left + (x1 - x0) * 0.008, y),
                ("monospace", 12).into_font().color(&BLACK),
            )))?;
        }
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(&BLACK)
        .label_font(("sans-serif", 15))
        .draw()?;
    Ok(())
}
