// Handoff records preserve explicit calibration quantities at the API boundary.
#![allow(clippy::too_many_arguments)]
use std::error::Error;
use std::f64::consts::PI;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use zip::ZipArchive;

use crate::args::Args;
use crate::bandpass::read_bandpass_file;
use crate::header::CorHeader;
use crate::npy_output::{NamedNpz, NpyMeta};
use crate::output::insert_product_before_processing_suffixes;
use crate::utils::uvw_cal;

const C_M_PER_S: f64 = 299_792_458.0;

#[derive(Debug, Serialize)]
struct ContaminationHandoff<'a> {
    product: &'static str,
    format_version: u32,
    note: &'static str,
    input_cor: String,
    source: SourceInfo<'a>,
    baseline: BaselineInfo<'a>,
    spectral_setup: SpectralSetup,
    time_axis: TimeAxis,
    uvw_m: UvwSeries,
    phase_correction: PhaseCorrectionSeries,
    fringe_peak: FringePeak,
    projection: ProjectionInfo,
    visibility: VisibilitySeries,
    flux_usage: FluxUsage,
    #[serde(skip)]
    bandpass_real: Vec<f32>,
    #[serde(skip)]
    bandpass_imag: Vec<f32>,
    /// Unmodified row-major complex visibility read from the source .cor
    /// window, before normalization, rebinning, phase correction, padding,
    /// or bandpass correction.  Shape is
    /// [time_axis.samples_per_visibility, spectral_setup.original_channels].
    #[serde(skip)]
    raw_visibility_real: Vec<f32>,
    #[serde(skip)]
    raw_visibility_imag: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct SourceInfo<'a> {
    name: &'a str,
    phase_center_ra_rad: f64,
    phase_center_dec_rad: f64,
    phase_center_ra_deg: f64,
    phase_center_dec_deg: f64,
}

#[derive(Debug, Serialize)]
struct BaselineInfo<'a> {
    station1: &'a str,
    station2: &'a str,
    station1_xyz_m: [f64; 3],
    station2_xyz_m: [f64; 3],
}

#[derive(Debug, Serialize)]
struct SpectralSetup {
    observing_frequency_hz: f64,
    sampling_speed_hz: i32,
    fft_point: i32,
    channels: usize,
    original_channels: usize,
    bandwidth_hz: f64,
    channel_width_hz: f64,
    frequency_mhz: Vec<f64>,
    wavelength_m: Vec<f64>,
    #[serde(skip)]
    raw_frequency_mhz: Vec<f64>,
    #[serde(skip)]
    raw_wavelength_m: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct TimeAxis {
    start_utc: String,
    effective_integration_time_s: f32,
    sectors: i32,
    samples_per_visibility: i32,
    mjd: Vec<f64>,
    elapsed_s: Vec<f64>,
    #[serde(skip)]
    raw_mjd: Vec<f64>,
    #[serde(skip)]
    raw_elapsed_s: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct UvwSeries {
    description: &'static str,
    u_m: Vec<f64>,
    v_m: Vec<f64>,
    w_m: Vec<f64>,
    du_dt_m_per_s: Vec<f64>,
    dv_dt_m_per_s: Vec<f64>,
    #[serde(skip)]
    raw_u_m: Vec<f64>,
    #[serde(skip)]
    raw_v_m: Vec<f64>,
    #[serde(skip)]
    raw_w_m: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct ContaminationPhaseCorrectionInput {
    pub manual_delay_sample: f32,
    pub manual_rate_hz: f32,
    pub manual_acel_hz_per_s: f32,
    pub manual_jerk_hz_per_s2: f32,
    pub manual_snap_hz_per_s3: f32,
    pub manual_start_time_offset_s: f32,
    pub search_delay_sample: f32,
    pub search_rate_hz: f32,
    pub search_start_time_offset_s: f32,
    pub target_frame_rotation_deg: f32,
}

#[derive(Debug, Serialize)]
struct PhaseCorrectionSeries {
    description: &'static str,
    manual_delay_sample: f32,
    manual_rate_hz: f32,
    manual_acel_hz_per_s: f32,
    manual_jerk_hz_per_s2: f32,
    manual_snap_hz_per_s3: f32,
    manual_start_time_offset_s: f32,
    search_delay_sample: f32,
    search_rate_hz: f32,
    search_start_time_offset_s: f32,
    total_delay_sample_at_reference: f32,
    target_frame_rotation_deg: f32,
    note: &'static str,
}

#[derive(Debug, Serialize)]
struct VisibilitySeries {
    description: &'static str,
    layout: &'static str,
    real: Vec<f32>,
    imag: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct FringePeak {
    description: &'static str,
    delay_sample: f32,
    rate_hz: f32,
    snr: f32,
    noise: f32,
}

#[derive(Debug, Serialize)]
struct ProjectionInfo {
    analysis_rows: usize,
    rate_padding: u32,
    rfi_specs: Vec<String>,
    bandpass_applied: bool,
    exact_peak_indices_available: bool,
}

#[derive(Debug, Serialize)]
struct FluxUsage {
    contamination_cli_for_flux: &'static str,
    phase_model: &'static str,
    amplitude_model: &'static str,
}

pub fn write_contamination_handoff(
    input_path: &Path,
    args: &Args,
    header: &CorHeader,
    frinz_dir: &Path,
    basename: &str,
    window_start_time: DateTime<Utc>,
    effective_integ_time: f32,
    sectors_in_window: i32,
    fringe_peak_visibility: num_complex::Complex<f32>,
    fringe_peak_delay_sample: f32,
    fringe_peak_rate_hz: f32,
    fringe_peak_snr: f32,
    fringe_peak_noise: f32,
    correction: ContaminationPhaseCorrectionInput,
    bandpass_data: Option<&[num_complex::Complex<f32>]>,
    raw_visibility: &[num_complex::Complex<f32>],
) -> Result<PathBuf, Box<dyn Error>> {
    let output_path = contamination_output_path(input_path, args, frinz_dir, basename)?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let handoff = build_handoff(
        input_path,
        header,
        window_start_time,
        effective_integ_time,
        sectors_in_window,
        fringe_peak_visibility,
        fringe_peak_delay_sample,
        fringe_peak_rate_hz,
        fringe_peak_snr,
        fringe_peak_noise,
        correction,
        bandpass_data,
        raw_visibility,
        args,
    );
    let npz_path = output_path;
    write_contamination_npz(&npz_path, &handoff, fringe_peak_visibility)?;
    println!("Contamination NPZ saved to: {}", npz_path.display());
    Ok(npz_path)
}

fn write_contamination_npz(
    path: &Path,
    handoff: &ContaminationHandoff<'_>,
    fringe_peak_visibility: num_complex::Complex<f32>,
) -> Result<(), Box<dyn Error>> {
    let mut npz = NamedNpz::new(NpyMeta::new(
        "contamination",
        handoff.spectral_setup.fft_point as u32,
        handoff.time_axis.sectors.max(0) as u32,
    ));
    npz.add_f64_1d("mjd", &handoff.time_axis.mjd);
    npz.add_f64_1d("elapsed_s", &handoff.time_axis.elapsed_s);
    npz.add_f64_1d("uv_u", &handoff.uvw_m.u_m);
    npz.add_f64_1d("uv_v", &handoff.uvw_m.v_m);
    npz.add_f64_1d("uv_w", &handoff.uvw_m.w_m);
    npz.add_f64_1d("du_dt_m_per_s", &handoff.uvw_m.du_dt_m_per_s);
    npz.add_f64_1d("dv_dt_m_per_s", &handoff.uvw_m.dv_dt_m_per_s);
    npz.add_f64_1d("frequency_mhz", &handoff.spectral_setup.frequency_mhz);
    npz.add_f64_1d("wavelength_m", &handoff.spectral_setup.wavelength_m);
    npz.add_f64_1d("raw_mjd", &handoff.time_axis.raw_mjd);
    npz.add_f64_1d("raw_elapsed_s", &handoff.time_axis.raw_elapsed_s);
    npz.add_f64_1d("raw_uv_u", &handoff.uvw_m.raw_u_m);
    npz.add_f64_1d("raw_uv_v", &handoff.uvw_m.raw_v_m);
    npz.add_f64_1d("raw_uv_w", &handoff.uvw_m.raw_w_m);
    npz.add_f64_1d(
        "raw_frequency_mhz",
        &handoff.spectral_setup.raw_frequency_mhz,
    );
    npz.add_f64_1d("raw_wavelength_m", &handoff.spectral_setup.raw_wavelength_m);
    npz.add_f32_1d("bandpass_real", &handoff.bandpass_real);
    npz.add_f32_1d("bandpass_imag", &handoff.bandpass_imag);
    npz.add_f32_1d("raw_visibility_real", &handoff.raw_visibility_real);
    npz.add_f32_1d("raw_visibility_imag", &handoff.raw_visibility_imag);
    npz.add_complex64_1d("complex_vis", &[fringe_peak_visibility]);
    npz.add_complex64_1d("frinz_complex_vis", &[fringe_peak_visibility]);
    npz.add_f32_1d("visibility_real", &handoff.visibility.real);
    npz.add_f32_1d("visibility_imag", &handoff.visibility.imag);
    npz.add_f64_1d("phase_center_ra_rad", &[handoff.source.phase_center_ra_rad]);
    npz.add_f64_1d(
        "phase_center_dec_rad",
        &[handoff.source.phase_center_dec_rad],
    );
    npz.add_f64_1d(
        "observing_frequency_hz",
        &[handoff.spectral_setup.observing_frequency_hz],
    );
    npz.add_f64_1d(
        "sampling_speed_hz",
        &[handoff.spectral_setup.sampling_speed_hz as f64],
    );
    npz.add_f64_1d(
        "effective_integration_time_s",
        &[handoff.time_axis.effective_integration_time_s as f64],
    );
    npz.add_f64_1d(
        "peak_delay_sample",
        &[handoff.fringe_peak.delay_sample as f64],
    );
    npz.add_f64_1d("peak_rate_hz", &[handoff.fringe_peak.rate_hz as f64]);
    npz.add_f64_1d("peak_snr", &[handoff.fringe_peak.snr as f64]);
    npz.add_f64_1d("peak_noise", &[handoff.fringe_peak.noise as f64]);
    npz.add_u8_1d("source_name", handoff.source.name.as_bytes());
    npz.add_u8_1d("input_cor", handoff.input_cor.as_bytes());
    let metadata = serde_json::to_vec(handoff)?;
    npz.add_u8_1d("metadata_json", &metadata);
    npz.write(path)?;
    Ok(())
}

fn contamination_output_path(
    input_path: &Path,
    args: &Args,
    frinz_dir: &Path,
    basename: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(tokens) = &args.contamination {
        for token in tokens {
            if let Some(value) = token.strip_prefix("output:") {
                return Ok(PathBuf::from(value).with_extension("npz"));
            }
            if let Some(value) = token.strip_prefix("out:") {
                return Ok(PathBuf::from(value).with_extension("npz"));
            }
        }
    }
    let stem = if basename.is_empty() {
        input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("frinZ")
    } else {
        basename
    };
    let output_stem = insert_product_before_processing_suffixes(stem, "contamination");
    Ok(frinz_dir
        .join("contamination")
        .join(format!("{output_stem}.npz")))
}

fn build_handoff<'a>(
    input_path: &Path,
    header: &'a CorHeader,
    window_start_time: DateTime<Utc>,
    effective_integ_time: f32,
    sectors_in_window: i32,
    fringe_peak_visibility: num_complex::Complex<f32>,
    fringe_peak_delay_sample: f32,
    fringe_peak_rate_hz: f32,
    fringe_peak_snr: f32,
    fringe_peak_noise: f32,
    correction: ContaminationPhaseCorrectionInput,
    bandpass_data: Option<&[num_complex::Complex<f32>]>,
    raw_visibility: &[num_complex::Complex<f32>],
    args: &Args,
) -> ContaminationHandoff<'a> {
    let original_channels = (header.fft_point / 2).max(0) as usize;
    let channel_width_hz = if header.fft_point > 0 {
        header.sampling_speed as f64 / header.fft_point as f64
    } else {
        0.0
    };
    let samples_per_visibility = sectors_in_window.max(0);
    let (u, v, w, du_dt, dv_dt) = uvw_cal(
        header.station1_position,
        header.station2_position,
        window_start_time,
        header.source_position_ra,
        header.source_position_dec,
        true,
    );
    let reference_frequency_hz = header.observing_frequency
        + 0.5 * original_channels.saturating_sub(1) as f64 * channel_width_hz;
    let reference_frequency_mhz = reference_frequency_hz / 1.0e6;
    let reference_wavelength_m = if reference_frequency_hz > 0.0 {
        C_M_PER_S / reference_frequency_hz
    } else {
        f64::NAN
    };
    let total_integration_time_s = samples_per_visibility.max(1) as f32 * effective_integ_time;
    let raw_frequency_mhz: Vec<f64> = (0..original_channels)
        .map(|channel| (header.observing_frequency + channel as f64 * channel_width_hz) / 1.0e6)
        .collect();
    let raw_wavelength_m: Vec<f64> = raw_frequency_mhz
        .iter()
        .map(|frequency_mhz| C_M_PER_S / (frequency_mhz * 1.0e6))
        .collect();
    let raw_samples = samples_per_visibility.max(0) as usize;
    let mut raw_mjd = Vec::with_capacity(raw_samples);
    let mut raw_elapsed_s = Vec::with_capacity(raw_samples);
    let mut raw_u_m = Vec::with_capacity(raw_samples);
    let mut raw_v_m = Vec::with_capacity(raw_samples);
    let mut raw_w_m = Vec::with_capacity(raw_samples);
    for sample in 0..raw_samples {
        let elapsed_s = sample as f64 * effective_integ_time as f64;
        let sample_time =
            window_start_time + Duration::nanoseconds((elapsed_s * 1.0e9).round() as i64);
        let (sample_u, sample_v, sample_w, _, _) = uvw_cal(
            header.station1_position,
            header.station2_position,
            sample_time,
            header.source_position_ra,
            header.source_position_dec,
            true,
        );
        raw_mjd.push(datetime_to_mjd(sample_time));
        raw_elapsed_s.push(elapsed_s);
        raw_u_m.push(sample_u);
        raw_v_m.push(sample_v);
        raw_w_m.push(sample_w);
    }

    let bandpass_real = bandpass_data
        .map(|values| values.iter().map(|value| value.re).collect())
        .unwrap_or_default();
    let bandpass_imag = bandpass_data
        .map(|values| values.iter().map(|value| value.im).collect())
        .unwrap_or_default();
    let expected_raw_values = raw_samples.saturating_mul(original_channels);
    assert_eq!(
        raw_visibility.len(),
        expected_raw_values,
        "raw contamination visibility shape does not match the .cor window"
    );
    let raw_visibility_real = raw_visibility.iter().map(|value| value.re).collect();
    let raw_visibility_imag = raw_visibility.iter().map(|value| value.im).collect();
    ContaminationHandoff {
        product: "frinZ_contamination_handoff",
        format_version: 5,
        note: "One NPZ contains the exact complex time-domain fringe scalar and the unmodified time-by-frequency complex visibility read from .cor. flux uses target and gain-calibrator handoffs to construct a contaminant model directly in the original .cor frame.",
        input_cor: input_path.display().to_string(),
        source: SourceInfo {
            name: &header.source_name,
            phase_center_ra_rad: header.source_position_ra,
            phase_center_dec_rad: header.source_position_dec,
            phase_center_ra_deg: header.source_position_ra.to_degrees(),
            phase_center_dec_deg: header.source_position_dec.to_degrees(),
        },
        baseline: BaselineInfo {
            station1: &header.station1_name,
            station2: &header.station2_name,
            station1_xyz_m: header.station1_position,
            station2_xyz_m: header.station2_position,
        },
        spectral_setup: SpectralSetup {
            observing_frequency_hz: header.observing_frequency,
            sampling_speed_hz: header.sampling_speed,
            fft_point: header.fft_point,
            channels: 1,
            original_channels,
            bandwidth_hz: header.sampling_speed as f64 / 2.0,
            channel_width_hz,
            frequency_mhz: vec![reference_frequency_mhz],
            wavelength_m: vec![reference_wavelength_m],
            raw_frequency_mhz,
            raw_wavelength_m,
        },
        time_axis: TimeAxis {
            start_utc: window_start_time.to_rfc3339(),
            effective_integration_time_s: total_integration_time_s,
            sectors: 1,
            samples_per_visibility,
            mjd: vec![datetime_to_mjd(window_start_time)],
            elapsed_s: vec![0.0],
            raw_mjd,
            raw_elapsed_s,
        },
        uvw_m: UvwSeries {
            description: "UVW and derivatives in meters at the start of the analyzed --length window. The scalar visibility integrates forward from this epoch for effective_integration_time_s.",
            u_m: vec![u],
            v_m: vec![v],
            w_m: vec![w],
            du_dt_m_per_s: vec![du_dt],
            dv_dt_m_per_s: vec![dv_dt],
            raw_u_m,
            raw_v_m,
            raw_w_m,
        },
        phase_correction: PhaseCorrectionSeries {
            description: "Manual delay/rate corrections, if specified, have already been applied before frinZ selects the reported time-domain fringe cell. Searched residual rate is referenced to the first sample of the --length window, the same epoch as MJD/UVW. The handoff stores that exact complex scalar.",
            manual_delay_sample: correction.manual_delay_sample,
            manual_rate_hz: correction.manual_rate_hz,
            manual_acel_hz_per_s: correction.manual_acel_hz_per_s,
            manual_jerk_hz_per_s2: correction.manual_jerk_hz_per_s2,
            manual_snap_hz_per_s3: correction.manual_snap_hz_per_s3,
            manual_start_time_offset_s: correction.manual_start_time_offset_s,
            search_delay_sample: correction.search_delay_sample,
            search_rate_hz: correction.search_rate_hz,
            search_start_time_offset_s: correction.search_start_time_offset_s,
            total_delay_sample_at_reference: correction.manual_delay_sample + correction.search_delay_sample,
            target_frame_rotation_deg: correction.target_frame_rotation_deg,
            note: "The stored visibility is the complex value at the selected fringe cell, not a positive real amplitude reconstructed by flux. flux must not re-apply the search delay/rate.",
        },
        fringe_peak: FringePeak {
            description: "Delay/rate cell used for the exact complex time-domain fringe visibility reported by frinZ.",
            delay_sample: fringe_peak_delay_sample,
            rate_hz: fringe_peak_rate_hz,
            snr: fringe_peak_snr,
            noise: fringe_peak_noise,
        },
        projection: ProjectionInfo {
            analysis_rows: (samples_per_visibility.max(1) as usize).next_power_of_two(),
            rate_padding: args.rate_padding.max(1),
            rfi_specs: args.rfi.clone(),
            bandpass_applied: bandpass_data.is_some(),
            exact_peak_indices_available: false,
        },
        visibility: VisibilitySeries {
            description: "Exact complex time-domain fringe value used by the ordinary frinZ output row.",
            layout: "[scan]; one complex scalar per NPZ --length window",
            real: vec![fringe_peak_visibility.re],
            imag: vec![fringe_peak_visibility.im],
        },
        flux_usage: FluxUsage {
            contamination_cli_for_flux: "flux --contamination ra:<hhmmss> dec:<ddmmss> flux:<mJy|Jy> [alpha:<value>] [ref:<MHz>]",
            phase_model: "flux fits all frinZ complex scalars with V_i=A_i*exp(i*theta_target)+S_contam*exp(i*(G_i+theta_contam))+N_i, with A_i>=0 free per epoch and independent constant phases per band; no first-sample phase anchoring or delay/rate reprocessing is used.",
            amplitude_model: "A(nu) is flux density converted to correlation units by flux using the phase-center/gain-calibrator flux calibration products.",
        },
        bandpass_real,
        bandpass_imag,
        raw_visibility_real,
        raw_visibility_imag,
    }
}

fn datetime_to_mjd(dt: DateTime<Utc>) -> f64 {
    let mjd0 = DateTime::<Utc>::from_timestamp(0, 0).unwrap() - Duration::days(40_587);
    let duration = dt.signed_duration_since(mjd0);
    duration.num_microseconds().unwrap_or(0) as f64 / 86_400.0e6
}

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

impl std::ops::Add for C64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
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
impl std::ops::Mul<f64> for C64 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
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
    gain_flux_ratio: f64,
    #[serde(default)]
    compact_model_scale: Option<C64>,
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

#[derive(Debug, Clone, Deserialize)]
struct GainTransfer {
    band: char,
    midpoint_mjd: f64,
    frequency_hz: Vec<f64>,
    spectrum: Vec<C64>,
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
    let model = read_model(model_path, input)?;
    if model.product != "flux_contamination_subtraction_model"
        || !matches!(model.format_version, 1..=5)
    {
        return Err(format!(
            "unsupported contamination model {} version {}",
            model.product, model.format_version
        )
        .into());
    }
    let gain_transfers = model.gain_transfers;
    let mut records = model.records;
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
        if record.compact_model_scale.is_none()
            && record.direct_model_entry.is_empty()
            && record.bandpass_applied
            && (record.bandpass_real.len() != record.fft_point / 2
                || record.bandpass_imag.len() != record.fft_point / 2)
        {
            return Err("bandpass-applied legacy contamination model does not contain one complex bandpass value per raw channel; regenerate the handoff and flux model".into());
        }
    }
    if records.iter().any(|record| {
        record.compact_model_scale.is_none()
            && record.direct_model_entry.is_empty()
            && !record.bandpass_applied
            && record.bandpass_real.is_empty()
    }) && bandpass_override.is_none()
    {
        eprintln!(
            "#WARN: legacy contamination model window(s) assume a flat complex bandpass; frequency-dependent residuals can remain in the delay plane"
        );
    }

    for record in &records {
        subtract_one_window(bytes, record, &gain_transfers, bandpass_override.as_deref())?;
    }
    let compact_windows = records
        .iter()
        .filter(|record| record.compact_model_scale.is_some())
        .count();
    println!(
        "# Contamination correction table: {} window(s) applied in copy-on-write memory from {}",
        records.len(),
        model_path.display()
    );
    if compact_windows > 0 {
        println!(
            "# Contamination subtraction method: compact gain/geometry table ({}/{})",
            compact_windows,
            records.len()
        );
    }
    println!("# No *_contamisubt.cor file is written");
    Ok(())
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

fn bracketing_gain_transfers(
    transfers: &[GainTransfer],
    band: char,
    mjd: f64,
) -> Result<(&GainTransfer, &GainTransfer, f64), Box<dyn Error>> {
    let same_band: Vec<_> = transfers.iter().filter(|item| item.band == band).collect();
    if same_band.is_empty() {
        return Err(format!("no compact gain-transfer data for {band} band").into());
    }
    let before = same_band
        .iter()
        .copied()
        .filter(|item| item.midpoint_mjd <= mjd)
        .max_by(|a, b| a.midpoint_mjd.total_cmp(&b.midpoint_mjd))
        .unwrap_or(same_band[0]);
    let after = same_band
        .iter()
        .copied()
        .filter(|item| item.midpoint_mjd >= mjd)
        .min_by(|a, b| a.midpoint_mjd.total_cmp(&b.midpoint_mjd))
        .unwrap_or(*same_band.last().expect("nonempty gain list"));
    let span = after.midpoint_mjd - before.midpoint_mjd;
    let fraction = if span.abs() < 1.0e-15 {
        0.0
    } else {
        ((mjd - before.midpoint_mjd) / span).clamp(0.0, 1.0)
    };
    Ok((before, after, fraction))
}

fn interpolate_gain_spectrum(
    transfers: &[GainTransfer],
    band: char,
    mjd: f64,
    frequency_hz: &[f64],
) -> Result<Vec<C64>, Box<dyn Error>> {
    let (before, after, fraction) = bracketing_gain_transfers(transfers, band, mjd)?;
    if before.spectrum.len() != frequency_hz.len()
        || after.spectrum.len() != frequency_hz.len()
        || before.frequency_hz.len() != frequency_hz.len()
        || after.frequency_hz.len() != frequency_hz.len()
    {
        return Err("gain and target raw frequency axes differ".into());
    }
    for ((&before_frequency, &after_frequency), &target_frequency) in before
        .frequency_hz
        .iter()
        .zip(&after.frequency_hz)
        .zip(frequency_hz)
    {
        if (before_frequency - target_frequency).abs() > 1.0
            || (after_frequency - target_frequency).abs() > 1.0
        {
            return Err("gain and target frequency values differ".into());
        }
    }
    if std::ptr::eq(before, after) {
        return Ok(before.spectrum.clone());
    }
    let mut cross = C64::new(0.0, 0.0);
    for channel in 1..frequency_hz.len() {
        let left_conjugate = C64::new(before.spectrum[channel].re, -before.spectrum[channel].im);
        cross += left_conjugate * after.spectrum[channel];
    }
    let global_delta = cross.im.atan2(cross.re);
    let align_after = C64::from_polar(1.0, -global_delta);
    let restore_phase = C64::from_polar(1.0, fraction * global_delta);
    Ok(before
        .spectrum
        .iter()
        .zip(&after.spectrum)
        .map(|(left, right)| {
            (*left * (1.0 - fraction) + (*right * align_after) * fraction) * restore_phase
        })
        .collect())
}

fn subtract_one_window(
    bytes: &mut [u8],
    scan: &ModelRecord,
    gain_transfers: &[GainTransfer],
    bandpass_override: Option<&[num_complex::Complex<f32>]>,
) -> Result<(), Box<dyn Error>> {
    if scan.fft_point < 4 || !scan.fft_point.is_multiple_of(2) || scan.samples == 0 {
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

    if let Some(model_scale) = scan.compact_model_scale {
        if scan.band == char::from(0)
            || scan.raw_mjd.len() != scan.samples
            || scan.geometric_delay_s.len() != scan.samples
            || scan.frequency_hz.len() != channels
            || !scan.gain_flux_ratio.is_finite()
            || scan.gain_flux_ratio <= 0.0
            || !scan.reference_frequency_hz.is_finite()
            || scan.reference_frequency_hz <= 0.0
        {
            return Err("invalid compact contamination correction table record".into());
        }
        for row in 0..scan.samples {
            let gain_spectrum = interpolate_gain_spectrum(
                gain_transfers,
                scan.band,
                scan.raw_mjd[row],
                &scan.frequency_hz,
            )?;
            for (channel, (&gain, &frequency_hz)) in gain_spectrum
                .iter()
                .zip(&scan.frequency_hz)
                .enumerate()
                .skip(1)
            {
                let spectral_amplitude =
                    (frequency_hz / scan.reference_frequency_hz).powf(scan.spectral_index);
                let geometric_phase = 2.0 * PI * frequency_hz * scan.geometric_delay_s[row];
                let q = gain
                    * (scan.gain_flux_ratio * spectral_amplitude)
                    * C64::from_polar(1.0, geometric_phase)
                    * model_scale;
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn scalar_handoff_uses_one_start_time_sample() {
        let header = CorHeader {
            sampling_speed: 1_024_000_000,
            observing_frequency: 6_100_000_000.0,
            fft_point: 1024,
            station1_name: "A".to_string(),
            station2_name: "B".to_string(),
            source_name: "TARGET".to_string(),
            source_position_ra: 1.0,
            source_position_dec: 0.5,
            station1_position: [-3_950_000.0, 3_300_000.0, 3_700_000.0],
            station2_position: [-3_950_050.0, 3_300_090.0, 3_700_030.0],
            ..CorHeader::default()
        };
        let start = Utc.with_ymd_and_hms(2025, 11, 10, 14, 46, 0).unwrap();
        let peak = num_complex::Complex::new(0.0125, -0.003);
        let raw_visibility = vec![num_complex::Complex::new(0.0, 0.0); 480 * 512];
        let correction = ContaminationPhaseCorrectionInput {
            manual_delay_sample: 0.0,
            manual_rate_hz: 0.0,
            manual_acel_hz_per_s: 0.0,
            manual_jerk_hz_per_s2: 0.0,
            manual_snap_hz_per_s3: 0.0,
            manual_start_time_offset_s: 0.0,
            search_delay_sample: 1.25,
            search_rate_hz: -0.002,
            search_start_time_offset_s: 0.0,
            target_frame_rotation_deg: 0.0,
        };
        let handoff = build_handoff(
            Path::new("scan.cor"),
            &header,
            start,
            1.0,
            480,
            peak,
            1.25,
            -0.002,
            20.0,
            0.0005,
            correction,
            None,
            &raw_visibility,
            &Args::default(),
        );

        assert_eq!(handoff.format_version, 5);
        assert_eq!(handoff.projection.analysis_rows, 512);
        assert_eq!(handoff.projection.rate_padding, 1);
        assert!(!handoff.projection.bandpass_applied);
        assert_eq!(handoff.spectral_setup.channels, 1);
        assert_eq!(handoff.spectral_setup.original_channels, 512);
        assert_eq!(handoff.time_axis.sectors, 1);
        assert_eq!(handoff.time_axis.samples_per_visibility, 480);
        assert_eq!(handoff.time_axis.effective_integration_time_s, 480.0);
        assert_eq!(handoff.visibility.real, vec![peak.re]);
        assert_eq!(handoff.visibility.imag, vec![peak.im]);
        assert_eq!(handoff.uvw_m.u_m.len(), 1);
        assert_eq!(handoff.time_axis.elapsed_s, vec![0.0]);
        assert_eq!(handoff.phase_correction.search_start_time_offset_s, 0.0);
        assert_eq!(handoff.raw_visibility_real.len(), 480 * 512);
        assert_eq!(handoff.raw_visibility_imag.len(), 480 * 512);
        assert!((handoff.time_axis.mjd[0] - datetime_to_mjd(start)).abs() < 1.0e-12);
    }
}
