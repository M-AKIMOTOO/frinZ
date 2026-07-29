use std::error::Error;
use std::f64::consts::PI;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use zip::ZipArchive;

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
) -> Result<PathBuf, Box<dyn Error>> {
    let model = read_model(model_path)?;
    if model.product != "flux_contamination_subtraction_model" || model.format_version != 1 {
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
    if records.iter().any(|record| record.bandpass_applied) {
        return Err("bandpass-applied contamination models require stored projection weights and are not yet supported".into());
    }

    let mut bytes = fs::read(input)?;
    for record in &records {
        subtract_one_window(&mut bytes, record)?;
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
    Ok(output)
}

fn read_model(path: &Path) -> Result<ModelFile, Box<dyn Error>> {
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
    Ok(serde_json::from_slice(payload)?)
}

fn subtract_one_window(bytes: &mut [u8], scan: &ModelRecord) -> Result<(), Box<dyn Error>> {
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

    for row in 0..scan.samples {
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
        for channel in 1..channels {
            if is_rfi(channel) {
                continue;
            }
            let delay_correction_phase =
                -2.0 * PI * (scan.manual_delay_sample + scan.search_delay_sample) * channel as f64
                    / scan.fft_point as f64;
            let delay_ifft_phase =
                2.0 * PI * channel as f64 * delay_ifft_idx as f64 / scan.fft_point as f64;
            let w = C64::from_polar(
                cell_scale,
                time_phase + rate_phase + delay_correction_phase + delay_ifft_phase,
            );
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
