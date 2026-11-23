use std::collections::BTreeMap;
use std::error::Error;
use std::f64::consts::TAU;
use std::fs::File;
use std::io::{Cursor, Read};
use std::ops::Range;
use std::path::{Path, PathBuf};

use num_complex::{Complex32, Complex64};
use serde::Deserialize;
use zip::ZipArchive;

use crate::bandpass::read_bandpass_file;
use crate::header::{available_cor_sectors, parse_header, CorHeader};
use crate::read::{sector_byte_range, sector_epochs, sector_size, SECTOR_HEADER_SIZE};
use crate::utils::mjd_cal;

// Earlier handoffs stored MJD through integer microseconds. Allow for that
// rounding and for f64 MJD resolution, while requiring a matching epoch.
const EPOCH_TOLERANCE_DAYS: f64 = 5e-6 / 86_400.0;
const FREQUENCY_TOLERANCE_HZ: f64 = 1.0;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(remote = "Complex64")]
struct ComplexParts {
    re: f64,
    im: f64,
}

#[derive(Deserialize)]
#[serde(transparent)]
struct GainSample(#[serde(with = "ComplexParts")] Complex64);

#[derive(Deserialize)]
struct ModelRecord {
    #[serde(default)]
    band: char,
    input_cor: PathBuf,
    start_mjd: f64,
    samples: usize,
    #[serde(default)]
    raw_mjd: Vec<f64>,
    analysis_rows: usize,
    rate_padding: u32,
    rfi_specs: Vec<String>,
    #[serde(default)]
    bandpass_applied: bool,
    integration_s: f64,
    fft_point: usize,
    sampling_speed_hz: f64,
    peak_delay_sample: f64,
    peak_rate_hz: f64,
    manual_delay_sample: f64,
    manual_rate_hz: f64,
    manual_acel_hz_per_s: f64,
    manual_jerk_hz_per_s2: f64,
    manual_snap_hz_per_s3: f64,
    manual_start_time_offset_s: f64,
    search_delay_sample: f64,
    search_rate_hz: f64,
    search_start_time_offset_s: f64,
    #[serde(with = "ComplexParts")]
    raw_model: Complex64,
    #[serde(default)]
    geometric_delay_s: Vec<f64>,
    #[serde(default)]
    frequency_hz: Vec<f64>,
    #[serde(default)]
    reference_frequency_hz: f64,
    #[serde(default)]
    spectral_index: f64,
    #[serde(default)]
    gain_flux_ratio: f64,
    #[serde(default)]
    compact_model_scale: Option<GainSample>,
    #[serde(default)]
    bandpass_real: Vec<f64>,
    #[serde(default)]
    bandpass_imag: Vec<f64>,
    #[serde(default)]
    direct_model_entry: String,
}

#[derive(Deserialize)]
struct GainTransfer {
    band: char,
    midpoint_mjd: f64,
    frequency_hz: Vec<f64>,
    spectrum: Vec<GainSample>,
}

#[derive(Deserialize)]
struct ModelFile {
    product: String,
    format_version: u32,
    records: Vec<ModelRecord>,
    #[serde(default)]
    gain_transfers: Vec<GainTransfer>,
}

pub fn apply_contamination_subtract(
    bytes: &mut [u8],
    input: &Path,
    model_path: &Path,
    bandpass_override_path: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
    let header = parse_header(&mut Cursor::new(&*bytes))?;
    let bandpass = bandpass_override_path.map(read_bandpass_file).transpose()?;
    subtract_model(bytes, &header, input, model_path, bandpass.as_deref())
}

pub(crate) fn subtract_model(
    bytes: &mut [u8],
    header: &CorHeader,
    input: &Path,
    model_path: &Path,
    bandpass_override: Option<&[Complex32]>,
) -> Result<(), Box<dyn Error>> {
    let mut archive = ZipArchive::new(File::open(model_path)?)?;
    let mut model: ModelFile = {
        let mut entry = archive.by_name("metadata_json.npy")?;
        let entry_size = entry.size();
        read_npy_header(&mut entry, entry_size)?;
        let mut payload = Vec::new();
        entry.read_to_end(&mut payload)?;
        serde_json::from_slice(&payload)?
    };
    if model.product != "flux_contamination_subtraction_model"
        || !matches!(model.format_version, 1..=5)
    {
        return Err(format!(
            "unsupported contamination model {} version {}",
            model.product, model.format_version
        )
        .into());
    }
    model.records.retain(|record| {
        record.input_cor == input || record.input_cor.file_name() == input.file_name()
    });
    if model.records.is_empty() {
        return Err(format!(
            "{} contains no records for {}",
            model_path.display(),
            input.display()
        )
        .into());
    }
    let available = available_cor_sectors(header, bytes.len())? as usize;
    let times: Vec<_> = sector_epochs(bytes, header, 0, available)?
        .map(|time| time.map(mjd_cal))
        .collect::<std::io::Result<_>>()?;
    let mut epochs: Vec<_> = times.iter().copied().enumerate().collect();
    if !epochs.is_sorted_by_key(|&(_, time)| time) {
        epochs.sort_by(|a, b| a.1.total_cmp(&b.1));
    }
    let mut windows = Vec::with_capacity(model.records.len());
    for record in &model.records {
        if record.fft_point != header.fft_point as usize
            || (record.sampling_speed_hz - f64::from(header.sampling_speed)).abs()
                > FREQUENCY_TOLERANCE_HZ
        {
            return Err(
                "contamination model FFT size or sampling speed differs from .cor header".into(),
            );
        }
        let range = locate_window(record, &epochs, &times)?;
        let byte_range = sector_byte_range(header, bytes.len(), range.start, range.end)?;
        windows.push((byte_range, record));
    }
    windows.sort_unstable_by_key(|(range, _)| range.start);
    if windows
        .windows(2)
        .any(|pair| pair[0].0.end > pair[1].0.start)
    {
        return Err("overlapping contamination correction sectors".into());
    }

    let mut gains_by_band = BTreeMap::<char, Vec<&GainTransfer>>::new();
    for transfer in &model.gain_transfers {
        gains_by_band
            .entry(transfer.band)
            .or_default()
            .push(transfer);
    }
    for transfers in gains_by_band.values_mut() {
        transfers.sort_by(|a, b| a.midpoint_mjd.total_cmp(&b.midpoint_mjd));
    }
    if bandpass_override.is_none()
        && model.records.iter().any(|record| {
            record.compact_model_scale.is_none()
                && record.direct_model_entry.is_empty()
                && !record.bandpass_applied
                && record.bandpass_real.is_empty()
        })
    {
        eprintln!("#WARN: legacy contamination model window(s) assume a flat complex bandpass; frequency-dependent residuals can remain in the delay plane");
    }
    let mut compact_windows = 0;
    let stride = sector_size(header);
    for (range, record) in windows {
        let window = &mut bytes[range];
        if let Some(scale) = &record.compact_model_scale {
            let gains = gains_by_band
                .get(&record.band)
                .ok_or_else(|| format!("no compact gain-transfer data for {} band", record.band))?;
            subtract_compact(window, stride, header, record, scale.0, gains)?;
            compact_windows += 1;
        } else if !record.direct_model_entry.is_empty() {
            subtract_direct(window, stride, record, &mut archive)?;
        } else {
            subtract_legacy(window, stride, header, record, bandpass_override)?;
        }
    }
    println!(
        "# Contamination correction table: {} window(s) applied in copy-on-write memory from {}",
        model.records.len(),
        model_path.display()
    );
    if compact_windows > 0 {
        println!("# Contamination subtraction method: compact gain/geometry table ({compact_windows}/{})", model.records.len());
    }
    Ok(())
}

fn locate_window(
    record: &ModelRecord,
    epochs: &[(usize, f64)],
    times: &[f64],
) -> Result<Range<usize>, Box<dyn Error>> {
    let upper = epochs.partition_point(|&(_, time)| time < record.start_mjd);
    let first = upper.saturating_sub(1);
    let candidates = &epochs[first..epochs.len().min(upper + 1)];
    let &(start, epoch) = candidates
        .iter()
        .min_by(|a, b| {
            (a.1 - record.start_mjd)
                .abs()
                .total_cmp(&(b.1 - record.start_mjd).abs())
        })
        .filter(|candidate| (candidate.1 - record.start_mjd).abs() <= EPOCH_TOLERANCE_DAYS)
        .ok_or_else(|| format!("no .cor sector at model start MJD {:.12}", record.start_mjd))?;
    let matching = epochs.partition_point(|&(_, time)| time < epoch);
    if epochs
        .get(matching + 1)
        .is_some_and(|&(_, time)| epoch == time)
    {
        return Err(format!("multiple .cor sectors share model start MJD {epoch:.12}").into());
    }
    let end = start
        .checked_add(record.samples)
        .filter(|&end| end <= times.len() && end > start)
        .ok_or("contamination window exceeds .cor payload or is empty")?;
    if !record.raw_mjd.is_empty() {
        if record.raw_mjd.len() != record.samples {
            return Err("contamination model sector time axis has the wrong length".into());
        }
        for (row, (&model_epoch, &actual)) in
            record.raw_mjd.iter().zip(&times[start..end]).enumerate()
        {
            if (model_epoch - actual).abs() > EPOCH_TOLERANCE_DAYS {
                return Err(format!("contamination model sector {} epoch mismatch: model MJD {model_epoch:.12}, .cor MJD {actual:.12}", start + row).into());
            }
        }
    }
    Ok(start..end)
}

fn read_npy_header(entry: &mut impl Read, entry_size: u64) -> std::io::Result<String> {
    use std::io::{Error, ErrorKind::InvalidData};
    let mut prefix = [0u8; 8];
    entry.read_exact(&mut prefix)?;
    if &prefix[..6] != b"\x93NUMPY" {
        return Err(Error::new(InvalidData, "invalid NPY magic"));
    }
    let mut length = [0u8; 4];
    let len_bytes = match prefix[6] {
        1 => 2,
        2 | 3 => 4,
        _ => return Err(Error::new(InvalidData, "unsupported NPY version")),
    };
    entry.read_exact(&mut length[..len_bytes])?;
    let header_len = u32::from_le_bytes(length) as usize;
    if header_len as u64 + (8 + len_bytes) as u64 > entry_size {
        return Err(Error::new(InvalidData, "NPY header exceeds entry size"));
    }
    let mut header = vec![0; header_len];
    entry.read_exact(&mut header)?;
    String::from_utf8(header).map_err(|error| Error::new(InvalidData, error))
}

fn read_direct_component(
    archive: &mut ZipArchive<File>,
    name: &str,
    count: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut entry = archive.by_name(name)?;
    let entry_size = entry.size();
    let header = read_npy_header(&mut entry, entry_size)?;
    if !header.contains("<f4") && !header.contains("=f4") {
        return Err(format!("{name} is not float32").into());
    }
    let mut values = Vec::with_capacity(count);
    let mut buffer = [0u8; 64 * 1024];
    while values.len() < count {
        let end = (count - values.len()).min(buffer.len() / 4) * 4;
        entry.read_exact(&mut buffer[..end])?;
        values.extend(
            buffer[..end]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        );
    }
    if entry.read(&mut buffer[..1])? != 0 {
        return Err(format!("{name} has more than {count} float32 values").into());
    }
    Ok(values)
}

fn visibility_cells(sector: &mut [u8]) -> impl Iterator<Item = &mut [u8]> {
    sector[SECTOR_HEADER_SIZE as usize..].chunks_exact_mut(8)
}

#[inline]
fn subtract_sample(bytes: &mut [u8], model: Complex64) {
    let (re, im) = bytes.split_at_mut(4);
    for (part, correction) in [(re, model.re), (im, model.im)] {
        let original = f64::from(f32::from_le_bytes(part.try_into().unwrap()));
        part.copy_from_slice(&((original - correction) as f32).to_le_bytes());
    }
}

fn subtract_direct(
    window: &mut [u8],
    stride: usize,
    scan: &ModelRecord,
    archive: &mut ZipArchive<File>,
) -> Result<(), Box<dyn Error>> {
    let count = scan.samples * (scan.fft_point / 2);
    let real = read_direct_component(
        archive,
        &format!("{}_real.npy", scan.direct_model_entry),
        count,
    )?;
    let imag = read_direct_component(
        archive,
        &format!("{}_imag.npy", scan.direct_model_entry),
        count,
    )?;
    for (bytes, (re, im)) in window
        .chunks_exact_mut(stride)
        .flat_map(visibility_cells)
        .zip(real.into_iter().zip(imag))
    {
        subtract_sample(bytes, Complex64::new(re.into(), im.into()));
    }
    Ok(())
}

/// Regular correlator channels permit a single phase increment per sector.
/// A nonuniform model axis can still use the exact frequencies.
struct GeometricPhase<'a> {
    frequencies: &'a [f64],
    step_hz: Option<f64>,
}

impl<'a> GeometricPhase<'a> {
    fn new(frequencies: &'a [f64], delays: &[f64]) -> Self {
        let step = frequencies[1] - frequencies[0];
        let max_delay = delays.iter().copied().map(f64::abs).fold(0.0, f64::max);
        let regular = frequencies.iter().enumerate().all(|(channel, &frequency)| {
            let expected = frequencies[0] + step * channel as f64;
            TAU * max_delay * (frequency - expected).abs() < 1e-11
        });
        Self {
            frequencies,
            step_hz: regular.then_some(step),
        }
    }

    fn fill(&self, phases: &mut [Complex64], delay_s: f64, extra_rad: f64) {
        let delay_rad_per_hz = TAU * delay_s;
        if let Some(step_hz) = self.step_hz {
            let step = Complex64::cis(delay_rad_per_hz * step_hz);
            let mut phase = Complex64::cis(delay_rad_per_hz * self.frequencies[0] + extra_rad);
            for value in phases {
                *value = phase;
                phase *= step;
            }
        } else {
            for (value, &frequency) in phases.iter_mut().zip(self.frequencies) {
                *value = Complex64::cis(delay_rad_per_hz * frequency + extra_rad);
            }
        }
    }
}

fn spectral_weights(header: &CorHeader, scan: &ModelRecord) -> Result<Vec<f64>, Box<dyn Error>> {
    let frequencies = &scan.frequency_hz;
    let step_hz = f64::from(header.sampling_speed) / f64::from(header.fft_point);
    if frequencies.len() != scan.fft_point / 2
        || scan.reference_frequency_hz <= 0.0
        || scan.geometric_delay_s.len() < scan.samples
    {
        return Err("invalid contamination geometry, frequency axis or reference frequency".into());
    }
    for (channel, &frequency) in frequencies.iter().enumerate() {
        let expected = header.observing_frequency + channel as f64 * step_hz;
        if frequency < 0.0 || (frequency - expected).abs() > FREQUENCY_TOLERANCE_HZ {
            return Err(format!("contamination frequency {frequency} Hz does not match .cor channel {channel} ({expected} Hz)").into());
        }
    }
    // DC is excluded in fringe analysis and in compact/continuous models.
    let weights: Vec<_> = std::iter::once(0.0)
        .chain(
            frequencies[1..]
                .iter()
                .map(|&hz| (hz / scan.reference_frequency_hz).powf(scan.spectral_index)),
        )
        .collect();
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err("non-finite contamination spectral amplitude".into());
    }
    Ok(weights)
}

struct GainPair {
    indices: (usize, usize),
    phase_delta: f64,
    initial: Vec<Complex64>,
    change: Vec<Complex64>,
}

impl GainPair {
    fn new(
        indices: (usize, usize),
        gains: &[&GainTransfer],
        frequencies: &[f64],
        scale: Complex64,
        weights: &[f64],
    ) -> Result<Self, Box<dyn Error>> {
        let (before, after) = (gains[indices.0], gains[indices.1]);
        for gain in [before, after] {
            if gain.spectrum.len() != frequencies.len()
                || gain.frequency_hz.len() != frequencies.len()
                || gain
                    .frequency_hz
                    .iter()
                    .zip(frequencies)
                    .any(|(a, b)| (a - b).abs() > FREQUENCY_TOLERANCE_HZ)
            {
                return Err("gain and target raw frequency axes differ".into());
            }
        }
        let phase_delta = if indices.0 != indices.1 {
            before.spectrum[1..]
                .iter()
                .zip(&after.spectrum[1..])
                .map(|(a, b)| a.0.conj() * b.0)
                .sum::<Complex64>()
                .arg()
        } else {
            0.0
        };
        let align = Complex64::cis(-phase_delta);
        let (initial, change) = before
            .spectrum
            .iter()
            .zip(&after.spectrum)
            .zip(weights)
            .map(|((a, b), weight)| {
                let factor = scale * weight;
                (a.0 * factor, (b.0 * align - a.0) * factor)
            })
            .unzip();
        Ok(Self {
            indices,
            phase_delta,
            initial,
            change,
        })
    }
}

fn subtract_compact(
    window: &mut [u8],
    stride: usize,
    header: &CorHeader,
    scan: &ModelRecord,
    model_scale: Complex64,
    gains: &[&GainTransfer],
) -> Result<(), Box<dyn Error>> {
    if scan.raw_mjd.len() != scan.samples
        || scan.geometric_delay_s.len() != scan.samples
        || scan.gain_flux_ratio <= 0.0
    {
        return Err("invalid compact contamination correction table record".into());
    }
    let weights = spectral_weights(header, scan)?;
    let geometry = GeometricPhase::new(&scan.frequency_hz, &scan.geometric_delay_s);
    let mut phases = vec![Complex64::default(); weights.len()];
    let scale = model_scale * scan.gain_flux_ratio;
    let mut active: Option<GainPair> = None;
    for (row, sector) in window.chunks_exact_mut(stride).enumerate() {
        let mjd = scan.raw_mjd[row];
        let upper = gains.partition_point(|gain| gain.midpoint_mjd < mjd);
        let after = upper.min(gains.len() - 1);
        let before = if gains[after].midpoint_mjd == mjd {
            after
        } else {
            upper.saturating_sub(1)
        };
        let indices = (before, after);
        if active.as_ref().is_none_or(|pair| pair.indices != indices) {
            active = Some(GainPair::new(
                indices,
                gains,
                &scan.frequency_hz,
                scale,
                &weights,
            )?);
        }
        let pair = active.as_ref().unwrap();
        let start = gains[before].midpoint_mjd;
        let end = gains[after].midpoint_mjd;
        let fraction = if start == end {
            0.0
        } else {
            ((mjd - start) / (end - start)).clamp(0.0, 1.0)
        };
        geometry.fill(
            &mut phases,
            scan.geometric_delay_s[row],
            fraction * pair.phase_delta,
        );
        for (channel, bytes) in visibility_cells(sector).enumerate().skip(1) {
            let gain = pair.initial[channel] + pair.change[channel] * fraction;
            subtract_sample(bytes, gain * phases[channel]);
        }
    }
    Ok(())
}

struct LegacyProjection {
    cell_scale: f64,
    time_phase: Vec<Complex64>,
    channel_phase: Vec<Complex64>, // zero for excluded channels, including DC
}

impl LegacyProjection {
    fn new(scan: &ModelRecord) -> Result<Self, Box<dyn Error>> {
        let rows = scan.analysis_rows.max(scan.samples.next_power_of_two());
        let bins = rows
            .checked_mul(scan.rate_padding.max(1) as usize)
            .ok_or("legacy rate grid overflow")?;
        let dt = scan.integration_s / scan.samples as f64;
        if dt <= 0.0 {
            return Err("invalid legacy model integration time".into());
        }
        let rate_idx = (scan.peak_rate_hz * bins as f64 * dt + bins as f64 / 2.0)
            .round()
            .clamp(0.0, (bins - 1) as f64) as usize;
        let rate_turns_per_row = ((rate_idx + bins / 2) % bins) as f64 / bins as f64;
        let channels = scan.fft_point / 2;
        let delay_idx = (scan.peak_delay_sample + channels as f64 - 1.0)
            .round()
            .clamp(0.0, (scan.fft_point - 1) as f64) as usize;
        let delay_ifft_idx = (scan.fft_point + channels - 1 - delay_idx) % scan.fft_point;
        let delay_turns_per_channel =
            (delay_ifft_idx as f64 - scan.manual_delay_sample - scan.search_delay_sample)
                / scan.fft_point as f64;
        let time_phase = (0..scan.samples)
            .map(|row| {
                let t = row as f64 * dt;
                let manual_t = t + scan.manual_start_time_offset_s;
                let cycles = scan.manual_rate_hz * manual_t
                    + 0.5 * scan.manual_acel_hz_per_s * manual_t.powi(2)
                    + (scan.manual_jerk_hz_per_s2 / 6.0) * manual_t.powi(3)
                    + (scan.manual_snap_hz_per_s3 / 24.0) * manual_t.powi(4)
                    + scan.search_rate_hz * (t + scan.search_start_time_offset_s);
                Complex64::cis(-TAU * (cycles + rate_turns_per_row * row as f64))
            })
            .collect();
        // Older flux tables serialize channel-index ranges here.
        let ranges: Vec<(usize, usize)> = scan
            .rfi_specs
            .iter()
            .filter_map(|spec| {
                let (a, b) = spec.split_once(',')?;
                Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
            })
            .collect();
        let channel_phase = std::iter::once(Complex64::default())
            .chain((1..channels).map(|channel| {
                if ranges.iter().any(|&(a, b)| (a..=b).contains(&channel)) {
                    Complex64::default()
                } else {
                    Complex64::cis(TAU * delay_turns_per_channel * channel as f64)
                }
            }))
            .collect();
        let bandwidth_mhz = scan.sampling_speed_hz / 2e6;
        let cell_scale = 512.0 / (bandwidth_mhz * scan.samples as f64);
        Ok(Self {
            cell_scale,
            time_phase,
            channel_phase,
        })
    }
}

fn raw_frame_bandpass(
    scan: &ModelRecord,
    override_data: Option<&[Complex32]>,
) -> Result<Vec<Complex64>, Box<dyn Error>> {
    let channels = scan.fft_point / 2;
    let mut bandpass: Vec<_> = if scan.bandpass_applied
        || !scan.bandpass_real.is_empty()
        || !scan.bandpass_imag.is_empty()
    {
        if scan.bandpass_real.len() != channels || scan.bandpass_imag.len() != channels {
            return Err("legacy model requires one complex bandpass value per raw channel".into());
        }
        scan.bandpass_real
            .iter()
            .zip(&scan.bandpass_imag)
            .map(|(&re, &im)| Complex64::new(re, im))
            .collect()
    } else {
        override_data
            .unwrap_or_default()
            .iter()
            .map(|v| Complex64::new(v.re.into(), v.im.into()))
            .collect()
    };
    if bandpass.is_empty() {
        return Ok(vec![Complex64::new(1.0, 0.0); channels]);
    }
    if bandpass.len() != channels {
        return Err("bandpass channel count differs from .cor header".into());
    }
    let mean = bandpass.iter().copied().sum::<Complex64>() / channels as f64;
    if !mean.norm_sqr().is_finite() || mean.norm_sqr() <= 1e-18 {
        return Err("stored complex bandpass has a zero mean".into());
    }
    for gain in &mut bandpass {
        *gain = if gain.norm_sqr() > 1e-18 {
            *gain / mean
        } else {
            Complex64::new(1.0, 0.0)
        };
    }
    Ok(bandpass)
}

fn subtract_legacy(
    window: &mut [u8],
    stride: usize,
    header: &CorHeader,
    scan: &ModelRecord,
    bandpass_override: Option<&[Complex32]>,
) -> Result<(), Box<dyn Error>> {
    let projection = LegacyProjection::new(scan)?;
    let active = projection
        .channel_phase
        .iter()
        .filter(|&&p| p != Complex64::default())
        .count();
    if active == 0 {
        return Err("invalid legacy projection: all channels are masked".into());
    }
    if scan.frequency_hz.is_empty() && scan.geometric_delay_s.is_empty() {
        let scale = scan.raw_model / (scan.samples as f64 * active as f64 * projection.cell_scale);
        for (sector, time_phase) in window.chunks_exact_mut(stride).zip(projection.time_phase) {
            for (bytes, &channel_phase) in visibility_cells(sector).zip(&projection.channel_phase) {
                if channel_phase != Complex64::default() {
                    subtract_sample(bytes, scale * (time_phase * channel_phase).conj());
                }
            }
        }
        return Ok(());
    }

    let weights = spectral_weights(header, scan)?;
    let bandpass = raw_frame_bandpass(scan, bandpass_override)?;
    let geometry = GeometricPhase::new(&scan.frequency_hz, &scan.geometric_delay_s);
    let mut phases = vec![Complex64::default(); weights.len()];
    let channel_weights: Vec<_> = projection
        .channel_phase
        .iter()
        .zip(&weights)
        .zip(&bandpass)
        .map(|((&phase, &weight), &bp)| {
            let scalar_gain = if scan.bandpass_applied {
                Complex64::new(1.0, 0.0)
            } else {
                bp
            };
            phase * scalar_gain * weight
        })
        .collect();
    let mut response = Complex64::default();
    for (&delay, &time_phase) in scan.geometric_delay_s.iter().zip(&projection.time_phase) {
        geometry.fill(&mut phases, delay, 0.0);
        let row_response: Complex64 = phases
            .iter()
            .zip(&channel_weights)
            .map(|(a, b)| a * b)
            .sum();
        response += row_response * time_phase;
    }
    response *= projection.cell_scale;
    if !response.norm_sqr().is_finite() || response.norm_sqr() <= f64::MIN_POSITIVE {
        return Err("continuous contamination model has zero coherent response".into());
    }
    let scale = scan.raw_model / response;
    let subtraction_weights: Vec<_> = bandpass
        .iter()
        .zip(&weights)
        .zip(&projection.channel_phase)
        .map(|((&bp, &weight), &phase)| {
            if phase == Complex64::default() {
                Complex64::default()
            } else {
                scale * bp * weight
            }
        })
        .collect();
    for (sector, &delay) in window.chunks_exact_mut(stride).zip(&scan.geometric_delay_s) {
        geometry.fill(&mut phases, delay, 0.0);
        for ((bytes, &phase), &weight) in visibility_cells(sector)
            .zip(&phases)
            .zip(&subtraction_weights)
        {
            if weight != Complex64::default() {
                subtract_sample(bytes, phase * weight);
            }
        }
    }
    Ok(())
}
