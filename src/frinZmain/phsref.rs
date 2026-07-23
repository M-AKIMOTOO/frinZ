// PHSREF module (phsref = phase reference).
// Uses a calibrator scan to estimate phase trends and applies phase-referenced
// correction to target visibilities, with optional fit models and diagnostics.
use std::error::Error;
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use num_complex::Complex;

use crate::args::Args;
use crate::fitting;
use crate::input_support::read_input_bytes;
use crate::npy_output::{npz_sidecar_path, NamedNpz, NpyMeta};
use crate::output::write_phase_corrected_spectrum_binary;
use crate::plot::phase_reference_plot;
use crate::processing::process_cor_file;
use crate::read::{read_sector_header, read_visibility_data};
use crate::utils;

type C32 = Complex<f32>;

#[derive(Debug, Clone)]
enum PhaseFitSpec {
    Poly2,
    Linear,
    Nearest,
    Hybrid,
}

#[derive(Debug, Clone)]
enum ResolvedPhaseFitModel {
    Poly2,
    Linear,
    Nearest,
    Hybrid,
}

impl PhaseFitSpec {
    fn min_data_points(&self) -> usize {
        match self {
            Self::Nearest => 1,
            Self::Linear => 2,
            Self::Poly2 | Self::Hybrid => 3,
        }
    }

    fn describe(&self) -> &'static str {
        match self {
            Self::Poly2 => "poly2",
            Self::Linear => "linear",
            Self::Nearest => "nearest",
            Self::Hybrid => "hybrid",
        }
    }

    fn resolve(&self) -> ResolvedPhaseFitModel {
        match self {
            Self::Poly2 => ResolvedPhaseFitModel::Poly2,
            Self::Linear => ResolvedPhaseFitModel::Linear,
            Self::Nearest => ResolvedPhaseFitModel::Nearest,
            Self::Hybrid => ResolvedPhaseFitModel::Hybrid,
        }
    }
}

fn parse_phase_fit_spec(raw_spec: Option<&str>) -> Result<PhaseFitSpec, String> {
    let mode = raw_spec.unwrap_or("poly2").trim().to_ascii_lowercase();
    match mode.as_str() {
        "poly2" | "quadratic" | "2" => Ok(PhaseFitSpec::Poly2),
        "linear" | "interp" => Ok(PhaseFitSpec::Linear),
        "nearest" => Ok(PhaseFitSpec::Nearest),
        "hybrid" => Ok(PhaseFitSpec::Hybrid),
        _ => Err(format!(
            "Invalid phase-reference mode '{}'. Use poly2, linear, nearest, or hybrid.",
            mode
        )),
    }
}

fn pack_samples(times: &[f64], values: &[f64], prefix: &[f64]) -> Vec<f64> {
    let mut packed = Vec::with_capacity(prefix.len() + times.len() * 2);
    packed.extend_from_slice(prefix);
    for (&time, &value) in times.iter().zip(values) {
        packed.push(time);
        packed.push(value);
    }
    packed
}

fn sample_model(x_sec: f64, packed: &[f64], offset: usize, nearest: bool) -> f64 {
    let samples = &packed[offset..];
    let count = samples.len() / 2;
    if count == 0 {
        return 0.0;
    }
    let at = |index: usize| (samples[index * 2], samples[index * 2 + 1]);
    if x_sec <= at(0).0 {
        return at(0).1;
    }
    if x_sec >= at(count - 1).0 {
        return at(count - 1).1;
    }
    let mut upper = 1usize;
    while upper < count && at(upper).0 < x_sec {
        upper += 1;
    }
    let (t0, y0) = at(upper - 1);
    let (t1, y1) = at(upper);
    if nearest {
        return if x_sec - t0 <= t1 - x_sec { y0 } else { y1 };
    }
    if (t1 - t0).abs() <= f64::EPSILON {
        return y0;
    }
    y0 + (y1 - y0) * ((x_sec - t0) / (t1 - t0))
}

fn evaluate_phase_fit_model(x_sec: f64, coeffs: &[f64], model: &ResolvedPhaseFitModel) -> f64 {
    match model {
        ResolvedPhaseFitModel::Poly2 => coeffs
            .iter()
            .take(3)
            .enumerate()
            .map(|(power, &coefficient)| coefficient * x_sec.powi(power as i32))
            .sum(),
        ResolvedPhaseFitModel::Linear => sample_model(x_sec, coeffs, 0, false),
        ResolvedPhaseFitModel::Nearest => sample_model(x_sec, coeffs, 0, true),
        ResolvedPhaseFitModel::Hybrid => {
            let trend: f64 = coeffs[..3]
                .iter()
                .enumerate()
                .map(|(power, &coefficient)| coefficient * x_sec.powi(power as i32))
                .sum();
            trend + sample_model(x_sec, coeffs, 3, false)
        }
    }
}

pub fn run_phase_reference_analysis(
    args: &Args,
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
) -> Result<(), Box<dyn Error>> {
    let cal_path = PathBuf::from(&args.phase_reference[1]);
    let target_path = PathBuf::from(&args.phase_reference[2]);

    // --- Parse phase_reference arguments ---
    let fit_spec = match parse_phase_fit_spec(args.phase_reference.first().map(|s| s.as_str())) {
        Ok(spec) => spec,
        Err(msg) => {
            eprintln!("Error: {}", msg);
            return Err("Invalid phase fit specification".into());
        }
    };
    let cal_length: i32 = if args.phase_reference.len() > 3 {
        args.phase_reference[3].parse().unwrap_or(0)
    } else {
        args.length // Default to global length or 0
    };
    let target_length: i32 = if args.phase_reference.len() > 4 {
        args.phase_reference[4].parse().unwrap_or(0)
    } else {
        args.length // Default to global length or 0
    };
    let loop_count: i32 = if args.phase_reference.len() > 5 {
        args.phase_reference[5].parse().unwrap_or(1)
    } else {
        args.loop_ // Default to global loop
    };

    // --- Create specific Args for calibrator and target ---
    let mut cal_args = args.clone();
    cal_args.length = cal_length;
    cal_args.loop_ = loop_count;
    cal_args.input = Some(cal_path.clone());

    let mut target_args = args.clone();
    target_args.length = target_length;
    target_args.loop_ = loop_count;
    target_args.input = Some(target_path.clone());

    println!("Running phase reference analysis...");
    println!(
        "Calibrator: {:?} (length: {}s, loop: {})",
        &cal_path,
        if cal_length == 0 {
            "all".to_string()
        } else {
            cal_length.to_string()
        },
        loop_count
    );
    let mut cal_results = process_cor_file(
        &cal_path,
        &cal_args,
        time_flag_ranges,
        pp_flag_ranges,
        false,
    )?;

    println!(
        "Target:     {:?} (length: {}s, loop: {})",
        &target_path,
        if target_length == 0 {
            "all".to_string()
        } else {
            target_length.to_string()
        },
        loop_count
    );
    let mut target_results = process_cor_file(
        &target_path,
        &target_args,
        time_flag_ranges,
        pp_flag_ranges,
        false,
    )?;

    let cal_header = &cal_results.header;
    let target_header = &target_results.header;
    let same_baseline = cal_header.station1_name.trim() == target_header.station1_name.trim()
        && cal_header.station2_name.trim() == target_header.station2_name.trim();
    let same_setup = cal_header.fft_point == target_header.fft_point
        && cal_header.sampling_speed == target_header.sampling_speed
        && (cal_header.observing_frequency - target_header.observing_frequency).abs() <= 1.0;
    if !same_baseline || !same_setup {
        return Err(format!(
            "Calibrator and target must use the same directed baseline and frequency setup: cal={}–{} FFT={} fs={} fobs={:.3}, target={}–{} FFT={} fs={} fobs={:.3}",
            cal_header.station1_name.trim(),
            cal_header.station2_name.trim(),
            cal_header.fft_point,
            cal_header.sampling_speed,
            cal_header.observing_frequency,
            target_header.station1_name.trim(),
            target_header.station2_name.trim(),
            target_header.fft_point,
            target_header.sampling_speed,
            target_header.observing_frequency,
        ).into());
    }

    // --- Phase Unwrapping ---
    utils::unwrap_phase(&mut cal_results.add_plot_phase, false);
    utils::unwrap_phase(&mut target_results.add_plot_phase, false);

    // Store original calibrator phases before fitting
    let original_cal_phases = cal_results.add_plot_phase.clone();
    // Store original target phases before fitting
    let original_target_phases = target_results.add_plot_phase.clone();

    let mut fitted_cal_phases: Vec<f32> = Vec::new(); // To store the fitted curve for calibrator

    // --- Phase Fitting ---
    let min_data_points = fit_spec.min_data_points();
    if cal_results.add_plot_times.is_empty() {
        eprintln!("Error: Calibrator data is empty, cannot proceed with phase fitting.");
        return Err("Empty calibrator data".into());
    }
    let first_time = cal_results.add_plot_times[0];
    if cal_results.add_plot_times.len() < min_data_points {
        eprintln!(
            "Warning: Not enough data points ({}) for {} on calibrator. Need at least {} points. Proceeding without phase fit.",
            cal_results.add_plot_times.len(),
            fit_spec.describe(),
            min_data_points
        );
    } else {
        let cal_times_f64: Vec<f64> = cal_results
            .add_plot_times
            .iter()
            .map(|t| t.signed_duration_since(first_time).num_milliseconds() as f64 / 1000.0)
            .collect();
        let cal_phases_f64: Vec<f64> = cal_results
            .add_plot_phase
            .iter()
            .map(|&p| p as f64)
            .collect();
        let fit_model = fit_spec.resolve();
        let fit_result: Result<Vec<f64>, Box<dyn Error>> = match &fit_model {
            ResolvedPhaseFitModel::Poly2 => {
                fitting::fit_polynomial_least_squares(&cal_times_f64, &cal_phases_f64, 2)
            }
            ResolvedPhaseFitModel::Linear | ResolvedPhaseFitModel::Nearest => {
                Ok(pack_samples(&cal_times_f64, &cal_phases_f64, &[]))
            }
            ResolvedPhaseFitModel::Hybrid => fitting::fit_polynomial_least_squares(
                &cal_times_f64,
                &cal_phases_f64,
                2,
            )
            .map(|trend| {
                let residuals: Vec<f64> = cal_times_f64
                    .iter()
                    .zip(&cal_phases_f64)
                    .map(|(&time, &phase)| {
                        let fitted: f64 = trend
                            .iter()
                            .enumerate()
                            .map(|(power, &coefficient)| coefficient * time.powi(power as i32))
                            .sum();
                        phase - fitted
                    })
                    .collect();
                pack_samples(&cal_times_f64, &residuals, &trend)
            }),
        };

        match fit_result {
            Ok(coeffs) => {
                println!("Phase-reference mode: {}", fit_spec.describe());
                if matches!(
                    fit_model,
                    ResolvedPhaseFitModel::Poly2 | ResolvedPhaseFitModel::Hybrid
                ) {
                    println!("Quadratic trend coefficients: {:?}", &coeffs[..3]);
                }

                // Calculate fitted_cal_phases
                fitted_cal_phases = cal_times_f64
                    .iter()
                    .map(|&t| evaluate_phase_fit_model(t, &coeffs, &fit_model) as f32)
                    .collect();

                // Subtract from calibrator
                for (i, t) in cal_times_f64.iter().enumerate() {
                    let fitted_val = evaluate_phase_fit_model(*t, &coeffs, &fit_model);
                    cal_results.add_plot_phase[i] -= fitted_val as f32;
                }

                // Subtract from target
                if !target_results.add_plot_times.is_empty() {
                    let target_times_f64: Vec<f64> = target_results
                        .add_plot_times
                        .iter()
                        .map(|t| {
                            t.signed_duration_since(first_time).num_milliseconds() as f64 / 1000.0
                        })
                        .collect();
                    let cal_start = cal_times_f64[0];
                    let cal_end = *cal_times_f64.last().unwrap_or(&cal_start);
                    let outside = target_times_f64
                        .iter()
                        .filter(|&&time| time < cal_start || time > cal_end)
                        .count();
                    if outside > 0 {
                        match fit_model {
                            ResolvedPhaseFitModel::Poly2 => eprintln!(
                                "#WARN: {} target epochs are outside the calibrator range [{:.3}, {:.3}] s; quadratic phase is being extrapolated.",
                                outside, cal_start, cal_end
                            ),
                            _ => eprintln!(
                                "#WARN: {} target epochs are outside the calibrator range [{:.3}, {:.3}] s; local residual correction is clamped to the nearest endpoint.",
                                outside, cal_start, cal_end
                            ),
                        }
                    }
                    for (i, t) in target_times_f64.iter().enumerate() {
                        let fitted_val = evaluate_phase_fit_model(*t, &coeffs, &fit_model);
                        target_results.add_plot_phase[i] -= fitted_val as f32;
                    }
                }

                // --- Apply phase correction to target and write to binary file ---
                println!(
                    "\nApplying phase correction to target file and writing to binary output..."
                );

                let target_buffer = read_input_bytes(&target_path)?;

                let mut file_header = vec![0u8; 256];
                let mut cursor = Cursor::new(target_buffer.as_slice());
                cursor.read_exact(&mut file_header)?;

                let mut calibrated_spectra: Vec<Vec<C32>> = Vec::new();
                let mut sector_headers_raw: Vec<Vec<u8>> = Vec::new();

                let num_sectors = target_results.header.number_of_sector;
                for l1 in 0..num_sectors {
                    let (complex_vec, current_obs_time, _effective_integ_time) =
                        read_visibility_data(
                            &mut Cursor::new(target_buffer.as_slice()),
                            &target_results.header,
                            1,
                            l1,
                            0,
                            false,
                            pp_flag_ranges,
                        )?;

                    let sector_headers = read_sector_header(
                        &mut Cursor::new(target_buffer.as_slice()),
                        &target_results.header,
                        1,
                        l1,
                        0,
                        false,
                    )?;
                    sector_headers_raw.push(sector_headers[0].clone());

                    let time_since_start_sec = current_obs_time
                        .signed_duration_since(first_time)
                        .num_milliseconds() as f64
                        / 1000.0;
                    let phase_correction_deg =
                        evaluate_phase_fit_model(time_since_start_sec, &coeffs, &fit_model);
                    let phase_correction_rad = (phase_correction_deg as f32).to_radians();

                    let phase_rotation = Complex::new(0.0, -phase_correction_rad).exp();
                    let calibrated_spectrum: Vec<C32> =
                        complex_vec.iter().map(|c| *c * phase_rotation).collect();
                    calibrated_spectra.push(calibrated_spectrum);
                }

                let target_basename = target_path.file_stem().unwrap().to_str().unwrap();
                let parts: Vec<&str> = target_basename.split('_').collect();
                if parts.len() >= 3 {
                    let new_basename = parts[..3].join("_");
                    let output_filename_str = format!("{}_phsref.cor", new_basename);
                    let phase_reference_dir = target_path.parent().unwrap_or_else(|| Path::new(""));
                    fs::create_dir_all(&phase_reference_dir)?;
                    let output_path = phase_reference_dir.join(output_filename_str);

                    write_phase_corrected_spectrum_binary(
                        &output_path,
                        &file_header,
                        &sector_headers_raw,
                        &calibrated_spectra,
                    )?;
                    println!(
                        "Successfully wrote phase-calibrated data to: {:?}",
                        output_path
                    );
                } else {
                    eprintln!("Warning: Could not generate output filename for calibrated data due to unexpected format of target filename.");
                }
            }
            Err(e) => {
                eprintln!(
                    "Warning: Phase fitting failed ({}): {}",
                    fit_spec.describe(),
                    e
                );
            }
        }
    }

    // --- Plotting ---
    let plot_dir = target_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join("frinZ")
        .join("phase_reference");
    fs::create_dir_all(&plot_dir)?;

    let target_basename = target_path.file_stem().unwrap().to_str().unwrap();
    let parts: Vec<&str> = target_basename.split('_').collect();
    let output_basename = if parts.len() >= 3 {
        parts[..3].join("_")
    } else {
        // Fallback for unexpected filename format
        format!(
            "phsref_{}_{}",
            cal_path.file_stem().unwrap().to_str().unwrap(),
            target_basename
        )
    };
    let plot_filename = format!("{}_phsref.png", output_basename);
    let output_filepath = plot_dir.join(plot_filename);

    phase_reference_plot(
        &cal_results.add_plot_times,
        &original_cal_phases,
        &fitted_cal_phases,
        &target_results.add_plot_times,
        &original_target_phases,
        &target_results.add_plot_phase,
        output_filepath.to_str().unwrap(),
    )?;
    let phase_axis = |times: &[chrono::DateTime<chrono::Utc>]| -> Vec<f64> {
        let Some(epoch) = times.first() else {
            return Vec::new();
        };
        times
            .iter()
            .map(|time| time.signed_duration_since(*epoch).num_milliseconds() as f64 / 1000.0)
            .collect()
    };
    if args.npz {
        let mut npz = NamedNpz::new(NpyMeta::new(
            "phase_reference",
            target_results.header.fft_point as u32,
            target_results.header.number_of_sector as u32,
        ));
        if cal_results.add_plot_times.len() == original_cal_phases.len() {
            let axis = phase_axis(&cal_results.add_plot_times);
            npz.add_f64_1d("calibrator_elapsed_time_s", &axis);
            npz.add_f32_1d("calibrator_phase_deg", &original_cal_phases);
            npz.add_f32_1d("calibrator_fit_phase_deg", &fitted_cal_phases);
        }
        if target_results.add_plot_times.len() == original_target_phases.len() {
            let axis = phase_axis(&target_results.add_plot_times);
            npz.add_f64_1d("target_elapsed_time_s", &axis);
            npz.add_f32_1d("target_phase_deg", &original_target_phases);
            npz.add_f32_1d("target_corrected_phase_deg", &target_results.add_plot_phase);
        }
        npz.write(&npz_sidecar_path(&output_filepath, "phase_reference"))?;
        for legacy_flag in [
            "phase_reference_calibrator",
            "phase_reference_calibrator_fit",
            "phase_reference_target",
            "phase_reference_target_corrected",
        ] {
            let _ = fs::remove_file(npz_sidecar_path(&output_filepath, legacy_flag));
        }
    }

    println!("Phase reference plot saved to: {:?}\n", output_filepath);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_supported_transfer_modes() {
        assert!(matches!(
            parse_phase_fit_spec(Some("poly2")).unwrap(),
            PhaseFitSpec::Poly2
        ));
        assert!(matches!(
            parse_phase_fit_spec(Some("linear")).unwrap(),
            PhaseFitSpec::Linear
        ));
        assert!(matches!(
            parse_phase_fit_spec(Some("nearest")).unwrap(),
            PhaseFitSpec::Nearest
        ));
        assert!(matches!(
            parse_phase_fit_spec(Some("hybrid")).unwrap(),
            PhaseFitSpec::Hybrid
        ));
        assert!(parse_phase_fit_spec(Some("cubic")).is_err());
    }

    #[test]
    fn linear_interpolation_clamps_outside_calibrator_range() {
        let packed = pack_samples(&[0.0, 10.0], &[20.0, 40.0], &[]);
        let model = ResolvedPhaseFitModel::Linear;
        assert_eq!(evaluate_phase_fit_model(-1.0, &packed, &model), 20.0);
        assert_eq!(evaluate_phase_fit_model(5.0, &packed, &model), 30.0);
        assert_eq!(evaluate_phase_fit_model(20.0, &packed, &model), 40.0);
    }

    #[test]
    fn nearest_uses_closest_calibrator_scan() {
        let packed = pack_samples(&[0.0, 10.0], &[20.0, 40.0], &[]);
        let model = ResolvedPhaseFitModel::Nearest;
        assert_eq!(evaluate_phase_fit_model(4.0, &packed, &model), 20.0);
        assert_eq!(evaluate_phase_fit_model(6.0, &packed, &model), 40.0);
    }

    #[test]
    fn hybrid_combines_quadratic_trend_and_interpolated_residual() {
        let packed = pack_samples(&[0.0, 10.0], &[2.0, 4.0], &[1.0, 0.5, 0.0]);
        let model = ResolvedPhaseFitModel::Hybrid;
        assert!((evaluate_phase_fit_model(5.0, &packed, &model) - 6.5).abs() < 1.0e-12);
    }
}
