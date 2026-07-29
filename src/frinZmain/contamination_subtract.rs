use std::error::Error;
use std::f64::consts::PI;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use zip::ZipArchive;

use crate::bandpass::read_bandpass_file;

const FILE_HEADER: usize = 256;
const SECTOR_HEADER: usize = 128;

#[derive(Debug, Clone, Copy, Deserialize)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn from_polar(r: f64, phase: f64) -> Self {
        Self::new(r * phase.cos(), r * phase.sin())
    }
    fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

impl std::ops::Mul for C64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}
impl std::ops::Div<f64> for C64 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ModelRecord {
    input_cor: PathBuf,
    start_mjd: f64,
    samples: usize,
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
    raw_model: C64,
    #[serde(default)]
    geometric_delay_s: Vec<f64>,
    #[serde(default)]
    frequency_hz: Vec<f64>,
    #[serde(default)]
    reference_frequency_hz: f64,
    #[serde(default)]
    spectral_index: f64,
    #[serde(default)]
    bandpass_real: Vec<f64>,
    #[serde(default)]
    bandpass_imag: Vec<f64>,
    #[serde(default)]
    direct_model_entry: String,
    #[serde(skip)]
    direct_model_real: Vec<f32>,
    #[serde(skip)]
    direct_model_imag: Vec<f32>,
}

#[derive(Deserialize)]
struct ModelFile {
    product: String,
    format_version: u32,
    records: Vec<ModelRecord>,
}

pub fn run_contamination_subtract(
    input: &Path,
    model_path: &Path,
    bandpass_override_path: Option<&Path>,
) -> Result<PathBuf, Box<dyn Error>> {
    let model = read_model(model_path, input)?;
    if model.product != "flux_contamination_subtraction_model"
        || !matches!(model.format_version, 1 | 2 | 3 | 4)
    {
        return Err(format!(
            "unsupported contamination model {} version {}",
            model.product, model.format_version
        )
        .into());
    }
    let input_name = input.file_name();
    let mut records: Vec<ModelRecord> = model
        .records
        .into_iter()
        .filter(|record| record.input_cor == input || record.input_cor.file_name() == input_name)
        .collect();
    if records.is_empty() {
        return Err(format!(
            "{} contains no records for {}",
            model_path.display(),
            input.display()
        )
        .into());
    }
    records.sort_by(|a, b| a.start_mjd.total_cmp(&b.start_mjd));
    for pair in records.windows(2) {
        let separation_s = (pair[1].start_mjd - pair[0].start_mjd) * 86_400.0;
        if separation_s + 1.0e-3 < pair[0].integration_s {
            return Err(format!(
                "overlapping contamination windows are not supported: {:.8} and {:.8}",
                pair[0].start_mjd, pair[1].start_mjd
            )
            .into());
        }
    }
    let bandpass_override = bandpass_override_path.map(read_bandpass_file).transpose()?;
    for record in &records {
        if record.bandpass_applied
            && (record.bandpass_real.len() != record.fft_point / 2
                || record.bandpass_imag.len() != record.fft_point / 2)
        {
            return Err("bandpass-applied contamination model does not contain one complex bandpass value per raw channel; regenerate the handoff and flux model".into());
        }
    }
    if records
        .iter()
        .any(|record| !record.bandpass_applied && record.direct_model_entry.is_empty())
        && bandpass_override.is_none()
        && records
            .iter()
            .any(|record| record.direct_model_entry.is_empty() && record.bandpass_real.is_empty())
    {
        eprintln!(
            "#WARN: contamination subtraction is assuming a flat complex bandpass for model window(s) without stored bandpass data; the integrated scalar is matched, but frequency-dependent residuals can remain in the delay plane"
        );
    }

    let mut bytes = fs::read(input)?;
    for record in &records {
        subtract_one_window(&mut bytes, record, bandpass_override.as_deref())?;
    }
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("visibility");
    let output_dir = input
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("contamisubt");
    fs::create_dir_all(&output_dir)?;
    let output = output_dir.join(format!("{stem}_contamisubt.cor"));
    fs::write(&output, &bytes)?;
    println!("Saved: {}", output.display());
    println!(
        "# Contamination subtraction: {} model window(s) applied from {}",
        records.len(),
        model_path.display()
    );
    let direct_windows = records
        .iter()
        .filter(|record| !record.direct_model_entry.is_empty())
        .count();
    if direct_windows > 0 {
        println!(
            "# Contamination subtraction method: direct raw time-frequency complex model ({}/{})",
            direct_windows,
            records.len()
        );
    }
    if let Some(path) = bandpass_override_path {
        println!(
            "# Contamination subtraction bandpass template: {} (model normalized in the original uncalibrated scalar frame)",
            path.display()
        );
    }
    Ok(output)
}

fn read_model(path: &Path, input: &Path) -> Result<ModelFile, Box<dyn Error>> {
    let file = fs::File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    let mut entry = archive.by_name("metadata_json.npy")?;
    let mut npy = Vec::new();
    entry.read_to_end(&mut npy)?;
    if npy.len() < 10 || &npy[..6] != b"\x93NUMPY" {
        return Err("invalid metadata_json.npy".into());
    }
    let major = npy[6];
    let (header_len, start) = match major {
        1 => (u16::from_le_bytes([npy[8], npy[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes([npy[8], npy[9], npy[10], npy[11]]) as usize,
            12,
        ),
        _ => return Err("unsupported NPY version".into()),
    };
    let payload = &npy[start + header_len..];
    let mut model: ModelFile = serde_json::from_slice(payload)?;
    drop(entry);
    let input_name = input.file_name();
    model
        .records
        .retain(|record| record.input_cor == input || record.input_cor.file_name() == input_name);
    for record in &mut model.records {
        if record.direct_model_entry.is_empty() {
            continue;
        }
        record.direct_model_real = read_f32_npy_entry(
            &mut archive,
            &format!("{}_real.npy", record.direct_model_entry),
        )?;
        record.direct_model_imag = read_f32_npy_entry(
            &mut archive,
            &format!("{}_imag.npy", record.direct_model_entry),
        )?;
    }
    Ok(model)
}

fn read_f32_npy_entry(
    archive: &mut ZipArchive<fs::File>,
    name: &str,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut entry = archive.by_name(name)?;
    let mut npy = Vec::new();
    entry.read_to_end(&mut npy)?;
    if npy.len() < 10 || &npy[..6] != b"\x93NUMPY" {
        return Err(format!("invalid {name}").into());
    }
    let (header_start, header_len) = match npy[6] {
        1 => (10usize, u16::from_le_bytes([npy[8], npy[9]]) as usize),
        2 | 3 => (
            12usize,
            u32::from_le_bytes([npy[8], npy[9], npy[10], npy[11]]) as usize,
        ),
        _ => return Err(format!("unsupported NPY version in {name}").into()),
    };
    let data_start = header_start + header_len;
    if data_start > npy.len() || (npy.len() - data_start) % 4 != 0 {
        return Err(format!("invalid float32 payload in {name}").into());
    }
    let header = String::from_utf8_lossy(&npy[header_start..data_start]);
    if !header.contains("<f4") && !header.contains("=f4") {
        return Err(format!("{name} is not float32").into());
    }
    Ok(npy[data_start..]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect())
}

fn subtract_one_window(
    bytes: &mut [u8],
    scan: &ModelRecord,
    bandpass_override: Option<&[num_complex::Complex<f32>]>,
) -> Result<(), Box<dyn Error>> {
    if scan.fft_point < 4 || scan.fft_point % 2 != 0 || scan.samples == 0 {
        return Err("invalid FFT point or integration length".into());
    }
    let channels = scan.fft_point / 2;
    let sector_size = SECTOR_HEADER + channels * 8;
    if bytes.len() < FILE_HEADER + sector_size {
        return Err("truncated .cor file".into());
    }
    let total_sectors = (bytes.len() - FILE_HEADER) / sector_size;
    let wanted_unix = ((scan.start_mjd - 40_587.0) * 86_400.0).round() as i64;
    let start_sector = (0..total_sectors)
        .min_by_key(|sector| {
            let pos = FILE_HEADER + sector * sector_size;
            let unix = i32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as i64;
            (unix - wanted_unix).abs()
        })
        .ok_or("no sectors in .cor")?;
    if start_sector + scan.samples > total_sectors {
        return Err("integration window exceeds .cor payload".into());
    }

    if !scan.direct_model_entry.is_empty() {
        let expected = scan.samples.saturating_mul(channels);
        if scan.direct_model_real.len() != expected || scan.direct_model_imag.len() != expected {
            return Err(format!(
                "direct model {} shape mismatch: expected {}, real {}, imag {}",
                scan.direct_model_entry,
                expected,
                scan.direct_model_real.len(),
                scan.direct_model_imag.len()
            )
            .into());
        }
        for row in 0..scan.samples {
            for channel in 0..channels {
                let model_idx = row * channels + channel;
                let pos =
                    FILE_HEADER + (start_sector + row) * sector_size + SECTOR_HEADER + channel * 8;
                let re = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap())
                    - scan.direct_model_real[model_idx];
                let im = f32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap())
                    - scan.direct_model_imag[model_idx];
                bytes[pos..pos + 4].copy_from_slice(&re.to_le_bytes());
                bytes[pos + 4..pos + 8].copy_from_slice(&im.to_le_bytes());
            }
        }
        return Ok(());
    }

    let rows = scan.analysis_rows.max(scan.samples.next_power_of_two());
    let rate_bins = rows.saturating_mul(scan.rate_padding.max(1) as usize);
    let dt = scan.integration_s / scan.samples as f64;
    let rate_idx = (scan.peak_rate_hz * rate_bins as f64 * dt + rate_bins as f64 / 2.0)
        .round()
        .clamp(0.0, (rate_bins - 1) as f64) as usize;
    let rate_fft_idx = (rate_idx + rate_bins / 2) % rate_bins;
    let delay_idx = (scan.peak_delay_sample + scan.fft_point as f64 / 2.0 - 1.0)
        .round()
        .clamp(0.0, (scan.fft_point - 1) as f64) as usize;
    let delay_ifft_idx = if delay_idx < scan.fft_point / 2 {
        scan.fft_point / 2 - 1 - delay_idx
    } else {
        scan.fft_point - 1 - (delay_idx - scan.fft_point / 2)
    };
    let bandwidth_mhz = scan.sampling_speed_hz / 2.0 / 1.0e6;
    let scale_factor = scan.fft_point as f64 / scan.samples as f64 * 512.0 / bandwidth_mhz;
    let rfi_ranges: Vec<(usize, usize)> = scan
        .rfi_specs
        .iter()
        .filter_map(|spec| {
            let (a, b) = spec.split_once(',')?;
            Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
        })
        .collect();
    let is_rfi = |channel: usize| {
        rfi_ranges
            .iter()
            .any(|(a, b)| channel >= *a && channel <= *b)
    };
    let active_channels = (1..channels).filter(|channel| !is_rfi(*channel)).count();
    let cell_scale = scale_factor / scan.fft_point as f64;
    let norm2 = scan.samples as f64 * active_channels as f64 * cell_scale * cell_scale;
    if !norm2.is_finite() || norm2 <= 0.0 {
        return Err("invalid frinZ projection normalization".into());
    }

    let projection_weight = |row: usize, channel: usize| {
        let local_t = row as f64 * dt;
        let manual_t = local_t + scan.manual_start_time_offset_s;
        let search_t = local_t + scan.search_start_time_offset_s;
        let time_phase = -2.0
            * PI
            * (scan.manual_rate_hz * manual_t
                + 0.5 * scan.manual_acel_hz_per_s * manual_t.powi(2)
                + scan.manual_jerk_hz_per_s2 / 6.0 * manual_t.powi(3)
                + scan.manual_snap_hz_per_s3 / 24.0 * manual_t.powi(4)
                + scan.search_rate_hz * search_t);
        let rate_phase = -2.0 * PI * rate_fft_idx as f64 * row as f64 / rate_bins as f64;
        let delay_correction_phase =
            -2.0 * PI * (scan.manual_delay_sample + scan.search_delay_sample) * channel as f64
                / scan.fft_point as f64;
        let delay_ifft_phase =
            2.0 * PI * channel as f64 * delay_ifft_idx as f64 / scan.fft_point as f64;
        C64::from_polar(
            cell_scale,
            time_phase + rate_phase + delay_correction_phase + delay_ifft_phase,
        )
    };

    let stored_bandpass: Vec<C64> = scan
        .bandpass_real
        .iter()
        .zip(&scan.bandpass_imag)
        .map(|(re, im)| C64::new(*re, *im))
        .collect();
    let override_bandpass: Vec<C64> = bandpass_override
        .unwrap_or_default()
        .iter()
        .map(|value| C64::new(value.re as f64, value.im as f64))
        .collect();
    let bandpass = if !stored_bandpass.is_empty() {
        Some(stored_bandpass.as_slice())
    } else if !override_bandpass.is_empty() {
        Some(override_bandpass.as_slice())
    } else {
        None
    };
    if let Some(values) = bandpass {
        if values.len() != channels {
            return Err(format!(
                "complex bandpass has {} channels, but .cor/model require {}",
                values.len(),
                channels
            )
            .into());
        }
    }
    let bandpass_mean = if let Some(values) = bandpass {
        let sum = values.iter().fold(C64::new(0.0, 0.0), |acc, value| {
            C64::new(acc.re + value.re, acc.im + value.im)
        });
        let mean = sum / channels as f64;
        if !mean.norm_sqr().is_finite() || mean.norm_sqr() <= 1.0e-18 {
            return Err("stored complex bandpass has a zero mean".into());
        }
        Some(mean)
    } else {
        None
    };
    let raw_frame_bandpass = |channel: usize| {
        let Some(mean) = bandpass_mean else {
            return C64::new(1.0, 0.0);
        };
        let value = bandpass.expect("bandpass mean implies bandpass data")[channel];
        if value.norm_sqr() > 1.0e-18 {
            value / mean
        } else {
            // apply_bandpass_correction leaves such a channel unchanged.
            C64::new(1.0, 0.0)
        }
    };

    if !scan.geometric_delay_s.is_empty() || !scan.frequency_hz.is_empty() {
        if scan.geometric_delay_s.len() < scan.samples {
            return Err("contamination model has too few geometric-delay samples".into());
        }
        if scan.frequency_hz.len() != channels {
            return Err(
                "contamination model frequency axis does not match the .cor channels".into(),
            );
        }
        if !scan.reference_frequency_hz.is_finite() || scan.reference_frequency_hz <= 0.0 {
            return Err("invalid contamination-model reference frequency".into());
        }
        let physical_basis = |row: usize, channel: usize| {
            let frequency_hz = scan.frequency_hz[channel];
            let spectral_amplitude =
                (frequency_hz / scan.reference_frequency_hz).powf(scan.spectral_index);
            C64::from_polar(
                spectral_amplitude,
                2.0 * PI * frequency_hz * scan.geometric_delay_s[row],
            )
        };
        let mut response = C64::new(0.0, 0.0);
        for row in 0..scan.samples {
            for channel in 1..channels {
                if !is_rfi(channel) {
                    let basis = physical_basis(row, channel);
                    let basis_in_scalar_frame = if scan.bandpass_applied {
                        basis
                    } else {
                        basis * raw_frame_bandpass(channel)
                    };
                    response += basis_in_scalar_frame * projection_weight(row, channel);
                }
            }
        }
        if !response.norm_sqr().is_finite() || response.norm_sqr() <= f64::MIN_POSITIVE {
            return Err("continuous contamination model has zero coherent response".into());
        }
        let model_scale = scan.raw_model / response;
        for row in 0..scan.samples {
            for channel in 1..channels {
                if is_rfi(channel) {
                    continue;
                }
                let q = model_scale * physical_basis(row, channel) * raw_frame_bandpass(channel);
                let pos =
                    FILE_HEADER + (start_sector + row) * sector_size + SECTOR_HEADER + channel * 8;
                let re = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as f64 - q.re;
                let im =
                    f32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as f64 - q.im;
                bytes[pos..pos + 4].copy_from_slice(&(re as f32).to_le_bytes());
                bytes[pos + 4..pos + 8].copy_from_slice(&(im as f32).to_le_bytes());
            }
        }
        return Ok(());
    }

    for row in 0..scan.samples {
        for channel in 1..channels {
            if is_rfi(channel) {
                continue;
            }
            let w = projection_weight(row, channel);
            let q = scan.raw_model * C64::new(w.re, -w.im) / norm2;
            let pos =
                FILE_HEADER + (start_sector + row) * sector_size + SECTOR_HEADER + channel * 8;
            let re = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as f64 - q.re;
            let im = f32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as f64 - q.im;
            bytes[pos..pos + 4].copy_from_slice(&(re as f32).to_le_bytes());
            bytes[pos + 4..pos + 8].copy_from_slice(&(im as f32).to_le_bytes());
        }
    }
    Ok(())
}
impl std::ops::Div for C64 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let denominator = rhs.norm_sqr();
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denominator,
            (self.im * rhs.re - self.re * rhs.im) / denominator,
        )
    }
}
impl std::ops::AddAssign for C64 {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}
