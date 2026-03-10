// PHSREF module (phsref = phase reference).
// Uses a calibrator scan to estimate phase trends and applies phase-referenced
// correction to target visibilities, with optional fit models and diagnostics.
use std::error::Error;
use std::fs::{self, File};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use num_complex::Complex;

use crate::args::Args;
use crate::fitting;
use crate::output::write_phase_corrected_spectrum_binary;
use crate::plot::phase_reference_plot;
use crate::processing::process_cor_file;
use crate::read::{read_sector_header, read_visibility_data};
use crate::utils;

type C32 = Complex<f32>;

#[derive(Debug, Clone)]
enum PhaseFitSpec {
    Polynomial {
        degree: usize,
    },
    PolynomialPlusSin {
        degree: usize,
        period_sec: Option<f64>,
    },
}

#[derive(Debug, Clone)]
enum ResolvedPhaseFitModel {
    Polynomial { degree: usize },
    PolynomialPlusSin { degree: usize, period_sec: f64 },
}

impl PhaseFitSpec {
    fn min_data_points(&self) -> usize {
        match self {
            Self::Polynomial { degree } => degree + 1,
            Self::PolynomialPlusSin { degree, .. } => degree + 3,
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Polynomial { degree } => format!("polynomial(deg={})", degree),
            Self::PolynomialPlusSin {
                degree,
                period_sec: Some(period_sec),
            } => format!("polynomial+sin(deg={}, period={}s)", degree, period_sec),
            Self::PolynomialPlusSin {
                degree,
                period_sec: None,
            } => format!("polynomial+sin(deg={}, period=auto)", degree),
        }
    }

    fn resolve(self, default_period_sec: f64) -> Result<ResolvedPhaseFitModel, String> {
        match self {
            Self::Polynomial { degree } => Ok(ResolvedPhaseFitModel::Polynomial { degree }),
            Self::PolynomialPlusSin { degree, period_sec } => {
                let fallback = if default_period_sec.is_finite() && default_period_sec > 0.0 {
                    default_period_sec
                } else {
                    1.0
                };
                let period = period_sec.unwrap_or(fallback);
                if !period.is_finite() || period <= 0.0 {
                    return Err(format!("Invalid sinusoid period: {}", period));
                }
                Ok(ResolvedPhaseFitModel::PolynomialPlusSin {
                    degree,
                    period_sec: period,
                })
            }
        }
    }
}

fn parse_phase_fit_spec(raw_spec: Option<&str>) -> Result<PhaseFitSpec, String> {
    let Some(raw) = raw_spec else {
        return Ok(PhaseFitSpec::Polynomial { degree: 1 });
    };

    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(PhaseFitSpec::Polynomial { degree: 1 });
    }

    if let Ok(degree) = trimmed.parse::<usize>() {
        return Ok(PhaseFitSpec::Polynomial { degree });
    }

    let normalized = trimmed.to_ascii_lowercase().replace(' ', "");
    let (model_part, period_sec) = if let Some((lhs, rhs)) = normalized.split_once(':') {
        let period = rhs.parse::<f64>().map_err(|_| {
            format!(
                "Invalid fit spec '{}': period part '{}' must be a positive number in seconds.",
                trimmed, rhs
            )
        })?;
        if !period.is_finite() || period <= 0.0 {
            return Err(format!(
                "Invalid fit spec '{}': period must be > 0.",
                trimmed
            ));
        }
        (lhs, Some(period))
    } else {
        (normalized.as_str(), None)
    };

    let degree = if model_part == "sin" {
        0
    } else if let Some(prefix) = model_part.strip_suffix("+sin") {
        prefix.parse::<usize>().map_err(|_| {
            format!(
                "Invalid fit spec '{}': degree before '+sin' must be non-negative integer.",
                trimmed
            )
        })?
    } else {
        return Err(format!(
            "Invalid fit spec '{}'. Use one of: <deg>, sin, <deg>+sin, <deg>+sin:<period_sec>.",
            trimmed
        ));
    };

    Ok(PhaseFitSpec::PolynomialPlusSin { degree, period_sec })
}

fn evaluate_phase_fit_model(x_sec: f64, coeffs: &[f64], model: &ResolvedPhaseFitModel) -> f64 {
    match model {
        ResolvedPhaseFitModel::Polynomial { degree } => coeffs
            .iter()
            .take(degree + 1)
            .enumerate()
            .map(|(i, &c)| c * x_sec.powi(i as i32))
            .sum(),
        ResolvedPhaseFitModel::PolynomialPlusSin { degree, period_sec } => {
            let omega = 2.0 * std::f64::consts::PI / period_sec;
            let poly_sum: f64 = coeffs
                .iter()
                .take(degree + 1)
                .enumerate()
                .map(|(i, &c)| c * x_sec.powi(i as i32))
                .sum();
            let sin_coeff = coeffs[*degree + 1];
            let cos_coeff = coeffs[*degree + 2];
            poly_sum + sin_coeff * (omega * x_sec).sin() + cos_coeff * (omega * x_sec).cos()
        }
    }
}

pub fn run_phase_reference_analysis(
    args: &Args,
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
) -> Result<(), Box<dyn Error>> {
    let cal_path = PathBuf::from(&args.phase_reference[0]);
    let target_path = PathBuf::from(&args.phase_reference[1]);

    // --- Parse phase_reference arguments ---
    let fit_spec = match parse_phase_fit_spec(args.phase_reference.get(2).map(|s| s.as_str())) {
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
        let cal_duration_sec = cal_times_f64.last().copied().unwrap_or(0.0)
            - cal_times_f64.first().copied().unwrap_or(0.0);
        let fit_model = match fit_spec.clone().resolve(cal_duration_sec) {
            Ok(model) => model,
            Err(msg) => {
                eprintln!("Warning: {}", msg);
                return Err("Failed to resolve phase fit model".into());
            }
        };

        let fit_result = match &fit_model {
            ResolvedPhaseFitModel::Polynomial { degree } => {
                fitting::fit_polynomial_least_squares(&cal_times_f64, &cal_phases_f64, *degree)
            }
            ResolvedPhaseFitModel::PolynomialPlusSin { degree, period_sec } => {
                fitting::fit_polynomial_plus_sinusoid_least_squares(
                    &cal_times_f64,
                    &cal_phases_f64,
                    *degree,
                    *period_sec,
                )
            }
        };

        match fit_result {
            Ok(coeffs) => {
                match &fit_model {
                    ResolvedPhaseFitModel::Polynomial { degree } => {
                        println!(
                            "Polynomial fit (degree {}) to calibrator phase. Coefficients: {:?}",
                            degree, coeffs
                        );
                    }
                    ResolvedPhaseFitModel::PolynomialPlusSin { degree, period_sec } => {
                        println!(
                            "Polynomial+sin fit (degree {}, period {:.3}s) to calibrator phase. Coefficients: {:?}",
                            degree, period_sec, coeffs
                        );
                    }
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
                    for (i, t) in target_times_f64.iter().enumerate() {
                        let fitted_val = evaluate_phase_fit_model(*t, &coeffs, &fit_model);
                        target_results.add_plot_phase[i] -= fitted_val as f32;
                    }
                }

                // --- Apply phase correction to target and write to binary file ---
                println!(
                    "\nApplying phase correction to target file and writing to binary output..."
                );

                let mut target_file = File::open(&target_path)?;
                let mut target_buffer = Vec::new();
                target_file.read_to_end(&mut target_buffer)?;

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

    println!("Phase reference plot saved to: {:?}\n", output_filepath);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_fit_spec_accepts_polynomial_degree() {
        let spec = parse_phase_fit_spec(Some("2")).unwrap();
        match spec {
            PhaseFitSpec::Polynomial { degree } => assert_eq!(degree, 2),
            _ => panic!("Expected polynomial fit spec"),
        }
    }

    #[test]
    fn parse_fit_spec_accepts_poly_plus_sin_with_period() {
        let spec = parse_phase_fit_spec(Some("1+sin:3600")).unwrap();
        match spec {
            PhaseFitSpec::PolynomialPlusSin { degree, period_sec } => {
                assert_eq!(degree, 1);
                assert_eq!(period_sec, Some(3600.0));
            }
            _ => panic!("Expected polynomial+sin fit spec"),
        }
    }

    #[test]
    fn evaluate_phase_fit_model_poly_plus_sin_uses_sin_cos_terms() {
        let model = ResolvedPhaseFitModel::PolynomialPlusSin {
            degree: 1,
            period_sec: 10.0,
        };
        let coeffs = vec![1.0, 2.0, 3.0, 4.0]; // c0, c1, sin, cos
        let y = evaluate_phase_fit_model(0.0, &coeffs, &model);
        assert!((y - 5.0).abs() < 1e-9);
    }
}
