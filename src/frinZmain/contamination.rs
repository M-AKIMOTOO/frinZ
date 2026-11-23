// Handoff records preserve explicit calibration quantities at the API boundary.
#![allow(clippy::too_many_arguments)]
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use num_complex::Complex32;
use serde::Serialize;

use crate::analysis::AnalysisResults;
use crate::args::Args;
use crate::header::CorHeader;
use crate::npy_output::{NamedNpz, NpyMeta};
use crate::output::insert_product_before_processing_suffixes;
use crate::utils::{greenwich_sidereal_angle, mjd_cal, seconds_between, UvwProjection};

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
    pub manual_start_time_offset_s: f64,
    pub search_delay_sample: f64,
    pub search_rate_hz: f64,
    pub search_start_time_offset_s: f64,
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
    manual_start_time_offset_s: f64,
    search_delay_sample: f64,
    search_rate_hz: f64,
    search_start_time_offset_s: f64,
    total_delay_sample_at_reference: f64,
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
    delay_sample: f64,
    rate_hz: f64,
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

pub(crate) struct ContaminationWindow<'a> {
    pub epochs: &'a [DateTime<Utc>],
    pub first_sector: usize,
    pub multiple_windows: bool,
    pub effective_integ_time: f32,
    pub raw_visibility: &'a [Complex32],
    pub native_bandpass: &'a [Complex32],
    pub bandpass_applied: bool,
}

pub(crate) fn write_contamination_handoff(
    input_path: &Path,
    args: &Args,
    header: &CorHeader,
    frinz_dir: &Path,
    basename: &str,
    window: ContaminationWindow<'_>,
    analysis: &AnalysisResults,
    correction: ContaminationPhaseCorrectionInput,
) -> Result<PathBuf, Box<dyn Error>> {
    let output_path = contamination_output_path(input_path, args, frinz_dir, basename, &window);
    if let Some(parent) = output_path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }

    let (cell_delay, cell_rate) = analysis.delay_peak_cell;
    let peak = FringePeak {
        description:
            "Delay/rate cell in the corrected plane used for the complex fringe visibility.",
        delay_sample: f64::from(cell_delay),
        rate_hz: f64::from(cell_rate),
        snr: analysis.delay_snr,
        noise: analysis.delay_noise,
    };
    let visibility = analysis.delay_peak_complex;
    let handoff = build_handoff(
        input_path, header, &window, visibility, peak, correction, args,
    )?;
    write_contamination_npz(&output_path, &handoff, visibility, window)?;
    println!("Contamination NPZ saved to: {}", output_path.display());
    Ok(output_path)
}

fn write_contamination_npz(
    path: &Path,
    handoff: &ContaminationHandoff<'_>,
    fringe_peak_visibility: Complex32,
    window: ContaminationWindow<'_>,
) -> Result<(), Box<dyn Error>> {
    let mut npz = NamedNpz::for_output(
        NpyMeta::new(
            "contamination",
            handoff.spectral_setup.fft_point as u32,
            handoff.time_axis.sectors.max(0) as u32,
        ),
        path,
    );
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
    npz.add_f32_series("bandpass_real", window.native_bandpass.iter().map(|v| v.re));
    npz.add_f32_series("bandpass_imag", window.native_bandpass.iter().map(|v| v.im));
    npz.add_f32_series(
        "raw_visibility_real",
        window.raw_visibility.iter().map(|v| v.re),
    );
    npz.add_f32_series(
        "raw_visibility_imag",
        window.raw_visibility.iter().map(|v| v.im),
    );
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
    npz.add_f64_1d("peak_delay_sample", &[handoff.fringe_peak.delay_sample]);
    npz.add_f64_1d("peak_rate_hz", &[handoff.fringe_peak.rate_hz]);
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
    window: &ContaminationWindow<'_>,
) -> PathBuf {
    if let Some(tokens) = &args.contamination {
        for token in tokens {
            if let Some(value) = token
                .strip_prefix("output:")
                .or_else(|| token.strip_prefix("out:"))
            {
                let path = PathBuf::from(value).with_extension("npz");
                if !window.multiple_windows {
                    return path;
                }
                let stem = path.file_stem().unwrap_or_default().to_string_lossy();
                return path.with_file_name(format!("{stem}_sector{}.npz", window.first_sector));
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
    let nanos = window
        .epochs
        .first()
        .map_or(0, DateTime::timestamp_subsec_nanos);
    let product = if nanos == 0 {
        "contamination".to_owned()
    } else {
        format!("{nanos:09}ns_contamination")
    };
    let output_stem = insert_product_before_processing_suffixes(stem, &product);
    frinz_dir
        .join("contamination")
        .join(format!("{output_stem}.npz"))
}

fn build_handoff<'a>(
    input_path: &Path,
    header: &'a CorHeader,
    window: &ContaminationWindow<'_>,
    fringe_peak_visibility: Complex32,
    fringe_peak: FringePeak,
    correction: ContaminationPhaseCorrectionInput,
    args: &Args,
) -> Result<ContaminationHandoff<'a>, Box<dyn Error>> {
    let window_start_time = *window.epochs.first().ok_or("empty contamination window")?;
    let raw_samples = window.epochs.len();
    let samples_per_visibility = i32::try_from(raw_samples)?;
    let original_channels = (header.fft_point / 2) as usize;
    if window.raw_visibility.len() != raw_samples * original_channels {
        return Err("raw visibility shape does not match the .cor window".into());
    }
    let channel_width_hz = f64::from(header.sampling_speed) / f64::from(header.fft_point);
    let projection = UvwProjection::new(
        header.station1_position,
        header.station2_position,
        header.source_position_ra,
        header.source_position_dec,
        true,
    );
    let (u, v, w, du_dt, dv_dt) = projection.project(greenwich_sidereal_angle(window_start_time));
    let reference_frequency_hz = header.observing_frequency
        + 0.5 * original_channels.saturating_sub(1) as f64 * channel_width_hz;
    let reference_frequency_mhz = reference_frequency_hz / 1.0e6;
    let reference_wavelength_m = if reference_frequency_hz > 0.0 {
        C_M_PER_S / reference_frequency_hz
    } else {
        f64::NAN
    };
    let total_integration_time_s = samples_per_visibility as f32 * window.effective_integ_time;
    let raw_frequency_mhz: Vec<f64> = (0..original_channels)
        .map(|channel| (header.observing_frequency + channel as f64 * channel_width_hz) / 1.0e6)
        .collect();
    let raw_wavelength_m: Vec<f64> = raw_frequency_mhz
        .iter()
        .map(|frequency_mhz| C_M_PER_S / (frequency_mhz * 1.0e6))
        .collect();
    let mut raw_mjd = Vec::with_capacity(raw_samples);
    let mut raw_elapsed_s = Vec::with_capacity(raw_samples);
    let mut raw_u_m = Vec::with_capacity(raw_samples);
    let mut raw_v_m = Vec::with_capacity(raw_samples);
    let mut raw_w_m = Vec::with_capacity(raw_samples);
    for &sample_time in window.epochs {
        let (sample_u, sample_v, sample_w, _, _) =
            projection.project(greenwich_sidereal_angle(sample_time));
        raw_mjd.push(mjd_cal(sample_time));
        raw_elapsed_s.push(seconds_between(sample_time, window_start_time));
        raw_u_m.push(sample_u);
        raw_v_m.push(sample_v);
        raw_w_m.push(sample_w);
    }

    Ok(ContaminationHandoff {
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
            mjd: vec![mjd_cal(window_start_time)],
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
            total_delay_sample_at_reference: f64::from(correction.manual_delay_sample) + correction.search_delay_sample,
            target_frame_rotation_deg: correction.target_frame_rotation_deg,
            note: "The stored visibility is the complex value at the selected fringe cell, not a positive real amplitude reconstructed by flux. flux must not re-apply the search delay/rate.",
        },
        fringe_peak,
        projection: ProjectionInfo {
            analysis_rows: (samples_per_visibility.max(1) as usize).next_power_of_two(),
            rate_padding: args.rate_padding.max(1),
            rfi_specs: crate::rfi::parse_rfi_ranges(&args.rfi, (channel_width_hz * 1e-6) as f32)?
                .into_iter().map(|(first, last)| format!("{first},{last}")).collect(),
            bandpass_applied: window.bandpass_applied,
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
    })
}

mod subtraction;
pub use subtraction::apply_contamination_subtract;
pub(crate) use subtraction::subtract_model;

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};

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
        let start =
            Utc.with_ymd_and_hms(2025, 11, 10, 14, 46, 0).unwrap() + Duration::milliseconds(125);
        let peak = num_complex::Complex::new(0.0125, -0.003);
        let raw_visibility = vec![num_complex::Complex::new(0.0, 0.0); 480 * 512];
        let epochs: Vec<_> = (0..480)
            .map(|sector| {
                start + Duration::seconds(if sector < 240 { sector } else { sector + 375 })
            })
            .collect();
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
            &ContaminationWindow {
                epochs: &epochs,
                first_sector: 0,
                multiple_windows: false,
                effective_integ_time: 1.0,
                raw_visibility: &raw_visibility,
                native_bandpass: &[],
                bandpass_applied: false,
            },
            peak,
            FringePeak {
                description: "cell",
                delay_sample: 0.0,
                rate_hz: 0.0,
                snr: 20.0,
                noise: 0.0005,
            },
            correction,
            &Args::default(),
        )
        .unwrap();

        assert_eq!(handoff.format_version, 5);
        assert_eq!(handoff.projection.analysis_rows, 512);
        assert_eq!(
            handoff.projection.rate_padding,
            crate::args::DEFAULT_RATE_PADDING
        );
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
        assert_eq!(handoff.time_axis.raw_mjd[240], mjd_cal(epochs[240]));
        assert_eq!(handoff.time_axis.raw_elapsed_s[240], 615.0);
        assert_eq!(handoff.time_axis.mjd[0], mjd_cal(start));
    }
}
