use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Utc};
use serde::Serialize;

use crate::args::Args;
use crate::header::CorHeader;
use crate::npy_output::{NamedNpz, NpyMeta};
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
}

#[derive(Debug, Serialize)]
struct TimeAxis {
    start_utc: String,
    effective_integration_time_s: f32,
    sectors: i32,
    samples_per_visibility: i32,
    mjd: Vec<f64>,
    elapsed_s: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct UvwSeries {
    description: &'static str,
    u_m: Vec<f64>,
    v_m: Vec<f64>,
    w_m: Vec<f64>,
    du_dt_m_per_s: Vec<f64>,
    dv_dt_m_per_s: Vec<f64>,
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
    Ok(frinz_dir
        .join("contamination")
        .join(format!("{stem}_contamination.npz")))
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

    ContaminationHandoff {
        product: "frinZ_contamination_handoff",
        format_version: 3,
        note: "One NPZ contains the exact complex time-domain fringe value reported by frinZ for one --length window. flux performs scalar point-source subtraction; full channel spectra are available separately through frinZ --spectrum.",
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
        },
        time_axis: TimeAxis {
            start_utc: window_start_time.to_rfc3339(),
            effective_integration_time_s: total_integration_time_s,
            sectors: 1,
            samples_per_visibility,
            mjd: vec![datetime_to_mjd(window_start_time)],
            elapsed_s: vec![0.0],
        },
        uvw_m: UvwSeries {
            description: "UVW and derivatives in meters at the start of the analyzed --length window. The scalar visibility integrates forward from this epoch for effective_integration_time_s.",
            u_m: vec![u],
            v_m: vec![v],
            w_m: vec![w],
            du_dt_m_per_s: vec![du_dt],
            dv_dt_m_per_s: vec![dv_dt],
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
            bandpass_applied: args.bandpass.is_some(),
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
    }
}

fn datetime_to_mjd(dt: DateTime<Utc>) -> f64 {
    let mjd0 = DateTime::<Utc>::from_timestamp(0, 0).unwrap() - Duration::days(40_587);
    let duration = dt.signed_duration_since(mjd0);
    duration.num_microseconds().unwrap_or(0) as f64 / 86_400.0e6
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
            &Args::default(),
        );

        assert_eq!(handoff.format_version, 3);
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
        assert!((handoff.time_axis.mjd[0] - datetime_to_mjd(start)).abs() < 1.0e-12);
    }
}
