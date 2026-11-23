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
use crate::input_support::open_input_data;
use crate::npy_output::{npz_sidecar_path, NamedNpz, NpyMeta};
use crate::output::write_phase_corrected_spectrum_binary;
use crate::plot::phase_reference_plot;
use crate::processing::{process_cor_file, ProcessResult};
use crate::read::{read_sector_header, read_visibility_data};
use crate::utils;

type C32 = Complex<f32>;

fn elapsed_seconds(time: DateTime<Utc>, epoch: DateTime<Utc>) -> f64 {
    (time.timestamp() - epoch.timestamp()) as f64
        + (f64::from(time.timestamp_subsec_nanos()) - f64::from(epoch.timestamp_subsec_nanos()))
            * 1e-9
}

/// Select the integer turn using the independently measured fringe rates.
/// All phases are in degrees at their own window-start epochs; rates are Hz.
fn connect_phase<T: Copy + Into<f64>>(
    times: &[f64],
    phases: &[f32],
    rates: &[T],
) -> Result<Vec<f64>, Box<dyn Error>> {
    if times.is_empty() || times.len() != phases.len() || times.len() != rates.len() {
        return Err("Phase connection requires matching, nonempty times, phases and rates".into());
    }
    if times.iter().any(|v| !v.is_finite())
        || phases.iter().any(|v| !v.is_finite())
        || rates.iter().any(|&v| !v.into().is_finite())
        || times.windows(2).any(|t| t[1] <= t[0])
    {
        return Err("Phase connection requires finite values and increasing epochs".into());
    }
    let mut connected = vec![phases[0] as f64];
    for i in 1..times.len() {
        let predicted =
            360.0 * 0.5 * (rates[i - 1].into() + rates[i].into()) * (times[i] - times[i - 1]);
        let delta = phases[i] as f64 - phases[i - 1] as f64;
        let turns = ((predicted - delta) / 360.0).round();
        connected.push(connected[i - 1] + delta + 360.0 * turns);
    }
    Ok(connected)
}

fn phase_rotation(phase_deg: f64) -> C32 {
    // Reduce in f64 before conversion, preserving precision over many turns.
    C32::from_polar(1.0, -(phase_deg.rem_euclid(360.0).to_radians() as f32))
}

/// Phase at the lower band edge plus the frequency slope at the same epoch.
/// The time evolution (including rate) is already in phase_deg and delay_s.
#[derive(Debug, Clone, Copy)]
struct FringeCorrection {
    phase_deg: f64,
    delay_s: f64,
}

impl FringeCorrection {
    fn apply(self, spectrum: &mut [C32], sampling_speed_hz: f64, fft_point: usize) {
        let phase_step_deg = 360.0 * sampling_speed_hz / fft_point as f64 * self.delay_s;
        for (channel, value) in spectrum.iter_mut().enumerate() {
            *value *= phase_rotation(self.phase_deg + channel as f64 * phase_step_deg);
        }
    }
}

fn fit_series(
    times: &[f64],
    values: &[f64],
    model: &ResolvedPhaseFitModel,
) -> Result<Vec<f64>, Box<dyn Error>> {
    if values.len() != times.len() || values.iter().any(|v| !v.is_finite()) {
        return Err("Fringe model requires matching finite samples".into());
    }
    match model {
        ResolvedPhaseFitModel::Poly2 => fitting::fit_polynomial_least_squares(times, values, 2),
        ResolvedPhaseFitModel::Linear | ResolvedPhaseFitModel::Nearest => {
            Ok(pack_samples(times, values, &[]))
        }
        ResolvedPhaseFitModel::Hybrid => {
            let trend = fitting::fit_polynomial_least_squares(times, values, 2)?;
            let residuals: Vec<f64> = times
                .iter()
                .zip(values)
                .map(|(&t, &v)| {
                    v - evaluate_phase_fit_model(t, &trend, &ResolvedPhaseFitModel::Poly2)
                })
                .collect();
            Ok(pack_samples(times, &residuals, &trend))
        }
        ResolvedPhaseFitModel::Hermite => {
            Err("Hermite interpolation requires fringe-rate derivatives".into())
        }
    }
}

fn fit_fringe_series(
    times: &[f64],
    values: &[f64],
    slopes: &[f64],
    model: &ResolvedPhaseFitModel,
) -> Result<Vec<f64>, Box<dyn Error>> {
    if !matches!(model, ResolvedPhaseFitModel::Hermite) {
        return fit_series(times, values, model);
    }
    if times.is_empty()
        || times.len() != values.len()
        || times.len() != slopes.len()
        || times
            .iter()
            .chain(values)
            .chain(slopes)
            .any(|v| !v.is_finite())
        || times.windows(2).any(|t| t[1] <= t[0])
    {
        return Err(
            "Hermite interpolation requires finite values, derivatives and increasing epochs"
                .into(),
        );
    }
    Ok(times
        .iter()
        .zip(values)
        .zip(slopes)
        .flat_map(|((&t, &v), &d)| [t, v, d])
        .collect())
}

fn evaluate_hermite(time: f64, packed: &[f64]) -> f64 {
    let n = packed.len() / 3;
    let at = |i: usize| (packed[3 * i], packed[3 * i + 1], packed[3 * i + 2]);
    let (first_t, first_v, first_d) = at(0);
    let (last_t, last_v, last_d) = at(n - 1);
    if time <= first_t {
        return first_v + first_d * (time - first_t);
    }
    if time >= last_t {
        return last_v + last_d * (time - last_t);
    }
    let mut upper = 1;
    while at(upper).0 < time {
        upper += 1;
    }
    let (t0, y0, d0) = at(upper - 1);
    let (t1, y1, d1) = at(upper);
    let h = t1 - t0;
    let u = (time - t0) / h;
    (2.0 * u.powi(3) - 3.0 * u * u + 1.0) * y0
        + (u.powi(3) - 2.0 * u * u + u) * h * d0
        + (-2.0 * u.powi(3) + 3.0 * u * u) * y1
        + (u.powi(3) - u * u) * h * d1
}

fn evaluate_fringe_model(
    time_s: f64,
    phase_coeffs: &[f64],
    delay_coeffs: &[f64],
    model: &ResolvedPhaseFitModel,
) -> FringeCorrection {
    FringeCorrection {
        phase_deg: evaluate_phase_fit_model(time_s, phase_coeffs, model),
        delay_s: evaluate_phase_fit_model(time_s, delay_coeffs, model),
    }
}

/// Undo the reporting frame of static/manual corrections before fitting a
/// model to be applied to the original .cor payload. Match fft.rs, including
/// the delay drift implied by rate at the lower band edge.
fn restore_manual_frame(
    result: &mut ProcessResult,
    args: &Args,
) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let fs = result.header.sampling_speed as f64;
    let nu0 = result.header.observing_frequency;
    let mut delays_s = Vec::with_capacity(result.add_plot_times.len());
    let mut rates_hz = Vec::with_capacity(result.add_plot_times.len());
    for i in 0..result.add_plot_times.len() {
        let t = elapsed_seconds(result.add_plot_times[i], result.obs_time);
        let cycles = args.rate_correct as f64 * t
            + 0.5 * args.acel_correct as f64 * t * t
            + args.jerk_correct as f64 * t.powi(3) / 6.0
            + args.snap_correct as f64 * t.powi(4) / 24.0;
        let rate = args.rate_correct as f64
            + args.acel_correct as f64 * t
            + 0.5 * args.jerk_correct as f64 * t * t
            + args.snap_correct as f64 * t.powi(3) / 6.0;
        result.add_plot_phase[i] = ((result.add_plot_phase[i] as f64 + 360.0 * cycles + 180.0)
            .rem_euclid(360.0)
            - 180.0) as f32;
        let (res_delay, res_rate) = match result.group_delay_fits.get(i) {
            Some(Ok(f)) => (f.delay_samples, f.rate_hz),
            Some(Err(reason)) => {
                return Err(format!(
                    "group delay at {} is unavailable ({reason})",
                    result.add_plot_times[i]
                )
                .into())
            }
            None => (
                f64::from(result.add_plot_res_delay[i]),
                f64::from(result.add_plot_res_rate[i]),
            ),
        };
        let delay = res_delay
            + f64::from(args.delay_correct)
            + if nu0 > 0.0 { fs * cycles / nu0 } else { 0.0 };
        let rate = res_rate + rate;
        delays_s.push(delay / fs);
        rates_hz.push(rate);
        result.add_plot_res_delay[i] = delay as f32;
        result.add_plot_res_rate[i] = rate as f32;
    }
    Ok((delays_s, rates_hz))
}

#[derive(Debug, Clone)]
enum PhaseFitSpec {
    Poly2,
    Linear,
    Nearest,
    Hybrid,
    Hermite,
}

#[derive(Debug, Clone)]
enum ResolvedPhaseFitModel {
    Poly2,
    Linear,
    Nearest,
    Hybrid,
    Hermite,
}

impl PhaseFitSpec {
    fn min_data_points(&self) -> usize {
        match self {
            Self::Nearest => 1,
            Self::Linear | Self::Hermite => 2,
            Self::Poly2 | Self::Hybrid => 3,
        }
    }

    fn describe(&self) -> &'static str {
        match self {
            Self::Poly2 => "poly2",
            Self::Linear => "linear",
            Self::Nearest => "nearest",
            Self::Hybrid => "hybrid",
            Self::Hermite => "hermite",
        }
    }

    fn resolve(&self) -> ResolvedPhaseFitModel {
        match self {
            Self::Poly2 => ResolvedPhaseFitModel::Poly2,
            Self::Linear => ResolvedPhaseFitModel::Linear,
            Self::Nearest => ResolvedPhaseFitModel::Nearest,
            Self::Hybrid => ResolvedPhaseFitModel::Hybrid,
            Self::Hermite => ResolvedPhaseFitModel::Hermite,
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
        "hermite" => Ok(PhaseFitSpec::Hermite),
        _ => Err(format!(
            "Invalid phase-reference mode '{}'. Use poly2, linear, nearest, hybrid, or hermite.",
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
        ResolvedPhaseFitModel::Hermite => evaluate_hermite(x_sec, coeffs),
    }
}

pub fn run_phase_reference_analysis(
    args: &Args,
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
) -> Result<(), Box<dyn Error>> {
    if args.frequency {
        return Err(
            "--phase-reference requires delay-domain fringe solutions; omit --frequency".into(),
        );
    }
    if args.scan_correct.is_some() || args.spike34m.is_some() {
        return Err("--phase-reference cannot reconstruct the raw-data frame after --scan-correct or --spike34; apply those corrections to a .cor file first".into());
    }
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
    // These arrays are calibration inputs, even when no optional plot was requested.
    cal_args.add_plot = true;
    if cal_args.search.is_empty() {
        cal_args.search = vec!["peak".to_string()];
    }

    let mut target_args = args.clone();
    target_args.length = target_length;
    target_args.loop_ = loop_count;
    target_args.input = Some(target_path.clone());
    target_args.add_plot = true;
    if target_args.search.is_empty() {
        target_args.search = vec!["peak".to_string()];
    }

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

    for result in [&cal_results, &target_results] {
        let n = result.add_plot_times.len();
        if [
            result.add_plot_phase.len(),
            result.add_plot_res_delay.len(),
            result.add_plot_res_rate.len(),
        ]
        .iter()
        .any(|&len| len != n)
        {
            return Err(
                "Phase referencing requires phase, delay and rate at every solution epoch".into(),
            );
        }
    }
    let (cal_delays_s, cal_rates) = restore_manual_frame(&mut cal_results, &cal_args)?;
    let (target_delays_s, target_rates) = restore_manual_frame(&mut target_results, &target_args)?;

    // --- Phase Unwrapping ---
    let first_time = *cal_results
        .add_plot_times
        .first()
        .ok_or("Empty calibrator data")?;
    let cal_times_f64: Vec<f64> = cal_results
        .add_plot_times
        .iter()
        .map(|&t| elapsed_seconds(t, first_time))
        .collect();
    let cal_phases_f64 = connect_phase(&cal_times_f64, &cal_results.add_plot_phase, &cal_rates)?;
    cal_results.add_plot_phase = cal_phases_f64.iter().map(|&p| p as f32).collect();
    println!(
        "Phase-reference epoch: {} (first valid calibrator window)",
        first_time.to_rfc3339()
    );
    println!(
        "Calibrator phase connection: fringe-rate assisted, {} samples",
        cal_phases_f64.len()
    );

    // Store original calibrator phases before fitting
    let original_cal_phases = cal_results.add_plot_phase.clone();
    // Store original target phases before fitting
    let target_wrapped_phases = target_results.add_plot_phase.clone();
    let target_times_f64: Vec<f64> = target_results
        .add_plot_times
        .iter()
        .map(|&t| elapsed_seconds(t, first_time))
        .collect();
    let original_target_phases: Vec<f32> = if target_times_f64.is_empty() {
        Vec::new()
    } else {
        connect_phase(&target_times_f64, &target_wrapped_phases, &target_rates)?
            .iter()
            .map(|&p| p as f32)
            .collect()
    };

    let fitted_cal_phases: Vec<f32>;
    let mut cal_fit_residuals = Vec::new();
    let cal_fit_delays_s: Vec<f64>;
    let mut target_model_phases_deg = Vec::new();
    let mut target_model_delays_s = Vec::new();

    // --- Phase Fitting ---
    let min_data_points = fit_spec.min_data_points();
    if cal_results.add_plot_times.is_empty() {
        eprintln!("Error: Calibrator data is empty, cannot proceed with phase fitting.");
        return Err("Empty calibrator data".into());
    }
    if cal_results.add_plot_times.len() < min_data_points {
        return Err(format!(
            "Not enough calibrator windows ({}) for {}: need at least {}. Specify cal_len and a larger loop count.",
            cal_results.add_plot_times.len(),
            fit_spec.describe(),
            min_data_points
        ).into());
    } else {
        let fit_model = fit_spec.resolve();
        let phase_slopes: Vec<f64> = cal_rates.iter().map(|&r| 360.0 * r).collect();
        let delay_slopes: Vec<f64> = cal_rates
            .iter()
            .map(|&r| r / cal_results.header.observing_frequency)
            .collect();
        let fit_result =
            fit_fringe_series(&cal_times_f64, &cal_phases_f64, &phase_slopes, &fit_model);
        let delay_coeffs =
            fit_fringe_series(&cal_times_f64, &cal_delays_s, &delay_slopes, &fit_model)?;

        match fit_result {
            Ok(coeffs) => {
                println!("Phase-reference mode: {}", fit_spec.describe());
                println!("Fringe transfer: phase at {:.3} MHz + delay slope per channel (rate is included in connected phase)", cal_results.header.observing_frequency / 1e6);
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
                cal_fit_delays_s = cal_times_f64
                    .iter()
                    .map(|&t| evaluate_phase_fit_model(t, &delay_coeffs, &fit_model))
                    .collect();

                // Keep fit diagnostics in f64, independently of the plot arrays.
                for (i, t) in cal_times_f64.iter().enumerate() {
                    let fitted_val = evaluate_phase_fit_model(*t, &coeffs, &fit_model);
                    let residual = cal_phases_f64[i] - fitted_val;
                    cal_fit_residuals.push(residual);
                    cal_results.add_plot_phase[i] = residual as f32;
                }
                let rms = (cal_fit_residuals.iter().map(|v| v * v).sum::<f64>()
                    / cal_fit_residuals.len() as f64)
                    .sqrt();
                println!("Calibrator phase-fit residual RMS: {:.3} deg", rms);
                if rms > 90.0 {
                    eprintln!("#WARN: Large calibrator phase residuals; check phase connection across scan gaps and model adequacy before using the calibrated data.");
                }

                // Subtract from target
                if !target_results.add_plot_times.is_empty() {
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
                            ResolvedPhaseFitModel::Hermite => eprintln!(
                                "#WARN: {} target epochs are outside the calibrator range [{:.3}, {:.3}] s; the endpoint phase/delay solution is propagated with its fringe rate.",
                                outside, cal_start, cal_end
                            ),
                            _ => eprintln!(
                                "#WARN: {} target epochs are outside the calibrator range [{:.3}, {:.3}] s; local residual correction is clamped to the nearest endpoint.",
                                outside, cal_start, cal_end
                            ),
                        }
                    }
                    for (i, t) in target_times_f64.iter().enumerate() {
                        let correction =
                            evaluate_fringe_model(*t, &coeffs, &delay_coeffs, &fit_model);
                        target_model_phases_deg.push(correction.phase_deg);
                        target_model_delays_s.push(correction.delay_s);
                        let fitted_val = correction.phase_deg;
                        // Correct wrapped target phases first: independent target unwrap
                        // can choose a different rate alias, particularly on a weak source.
                        target_results.add_plot_phase[i] =
                            ((target_wrapped_phases[i] as f64 - fitted_val + 180.0)
                                .rem_euclid(360.0)
                                - 180.0) as f32;
                    }
                    utils::unwrap_phase(&mut target_results.add_plot_phase, false);
                }

                // --- Apply phase correction to target and write to binary file ---
                println!(
                    "\nApplying phase correction to target file and writing to binary output..."
                );

                let target_buffer = open_input_data(&target_path)?;

                let mut file_header = vec![0u8; 256];
                let mut cursor = Cursor::new(target_buffer.as_slice());
                cursor.read_exact(&mut file_header)?;
                let native_header =
                    crate::header::parse_header(&mut Cursor::new(file_header.as_slice()))?;

                let mut calibrated_spectra: Vec<Vec<C32>> = Vec::new();
                let mut sector_headers_raw: Vec<Vec<u8>> = Vec::new();

                let num_sectors = native_header.number_of_sector;
                for l1 in 0..num_sectors {
                    let (mut complex_vec, current_obs_time, _effective_integ_time) =
                        read_visibility_data(
                            &mut Cursor::new(target_buffer.as_slice()),
                            &native_header,
                            1,
                            l1,
                            0,
                            false,
                            pp_flag_ranges,
                        )?;

                    let sector_headers = read_sector_header(
                        &mut Cursor::new(target_buffer.as_slice()),
                        &native_header,
                        1,
                        l1,
                        0,
                        false,
                    )?;
                    sector_headers_raw.push(sector_headers[0].clone());

                    let time_since_start_sec = elapsed_seconds(current_obs_time, first_time);
                    let correction = evaluate_fringe_model(
                        time_since_start_sec,
                        &coeffs,
                        &delay_coeffs,
                        &fit_model,
                    );
                    correction.apply(
                        &mut complex_vec,
                        native_header.sampling_speed as f64,
                        native_header.fft_point as usize,
                    );
                    calibrated_spectra.push(complex_vec);
                }

                let target_basename = target_path.file_stem().unwrap().to_str().unwrap();
                let parts: Vec<&str> = target_basename.split('_').collect();
                if parts.len() >= 3 {
                    let new_basename = parts[..3].join("_");
                    let observing_frequency = target_results.header.observing_frequency;
                    let band = if (6_600_000_000.0..=7_112_000_000.0).contains(&observing_frequency)
                    {
                        "c"
                    } else if (8_192_000_000.0..=8_704_000_000.0).contains(&observing_frequency) {
                        "x"
                    } else {
                        ""
                    };
                    let output_filename_str = format!("{}_{}phsref.cor", new_basename, band);
                    let phase_reference_dir = target_path.parent().unwrap_or_else(|| Path::new(""));
                    fs::create_dir_all(phase_reference_dir)?;
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
                return Err(
                    format!("Phase fitting failed ({}): {}", fit_spec.describe(), e).into(),
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
        times
            .iter()
            .map(|&time| elapsed_seconds(time, first_time))
            .collect()
    };
    if args.npz {
        let mut npz = NamedNpz::new(NpyMeta::new(
            "phase_reference",
            target_results.header.fft_point as u32,
            target_results.header.number_of_sector as u32,
        ));
        npz.add_f64_1d(
            "reference_epoch_unix_s",
            &[first_time.timestamp() as f64 + first_time.timestamp_subsec_nanos() as f64 / 1e9],
        );
        npz.add_f64_1d("calibrator_connected_phase_deg", &cal_phases_f64);
        npz.add_f64_1d(
            "reference_frequency_hz",
            &[cal_results.header.observing_frequency],
        );
        npz.add_f64_1d("calibrator_delay_s", &cal_delays_s);
        npz.add_f64_1d("calibrator_fit_delay_s", &cal_fit_delays_s);
        npz.add_f64_1d("target_model_phase_deg", &target_model_phases_deg);
        npz.add_f64_1d("target_model_delay_s", &target_model_delays_s);
        if !target_results.group_delay_fits.is_empty() {
            npz.add_f64_1d("target_delay_s", &target_delays_s);
        }
        npz.add_f32_1d("target_delay_samples", &target_results.add_plot_res_delay);
        npz.add_f32_1d("target_rate_hz", &target_results.add_plot_res_rate);
        npz.add_f64_1d("calibrator_fit_residual_deg", &cal_fit_residuals);
        npz.add_f32_1d("calibrator_rate_hz", &cal_results.add_plot_res_rate);
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
    fn full_model_cancels_delay_rate_and_phase_without_double_counting() {
        let fs = 64e6;
        let nu0 = 8.4e9;
        let fft = 64;
        let times = [0.0, 30.0, 1800.0, 1830.0, 3600.0];
        for (model, acceleration) in [
            (ResolvedPhaseFitModel::Linear, 0.0),
            (ResolvedPhaseFitModel::Poly2, 1e-6),
            (ResolvedPhaseFitModel::Hybrid, 1e-6),
            (ResolvedPhaseFitModel::Hermite, 1e-6),
        ] {
            let cycles = |t: f64| 0.037 * t + 0.5 * acceleration * t * t;
            let phase = |t: f64| 42.0 + 360.0 * cycles(t);
            let delay = |t: f64| 1.7 / fs + cycles(t) / nu0;
            let phases: Vec<f32> = times
                .iter()
                .map(|&t| ((phase(t) + 180.0).rem_euclid(360.0) - 180.0) as f32)
                .collect();
            let rates: Vec<f32> = times
                .iter()
                .map(|&t| (0.037 + acceleration * t) as f32)
                .collect();
            let connected = connect_phase(&times, &phases, &rates).unwrap();
            let delays: Vec<f64> = times.iter().map(|&t| delay(t)).collect();
            let phase_slopes: Vec<f64> = rates.iter().map(|&r| 360.0 * r as f64).collect();
            let delay_slopes: Vec<f64> = rates.iter().map(|&r| r as f64 / nu0).collect();
            let p = fit_fringe_series(&times, &connected, &phase_slopes, &model).unwrap();
            let d = fit_fringe_series(&times, &delays, &delay_slopes, &model).unwrap();
            let mut sum = C32::new(0.0, 0.0);
            for t in [330.0, 600.0, 2000.0, 3500.0] {
                let mut spectrum: Vec<C32> = (0..fft / 2)
                    .map(|k| {
                        let angle =
                            (phase(t) + 360.0 * k as f64 * fs / fft as f64 * delay(t)).to_radians();
                        C32::new(angle.cos() as f32, angle.sin() as f32)
                    })
                    .collect();
                evaluate_fringe_model(t, &p, &d, &model).apply(&mut spectrum, fs, fft);
                for sample in spectrum {
                    assert!(
                        (sample - C32::new(1.0, 0.0)).norm() < 2e-5,
                        "{model:?}: {sample}"
                    );
                    sum += sample;
                }
            }
            assert!((sum.norm() - 128.0).abs() < 1e-4);
        }
    }

    #[test]
    fn hermite_preserves_solution_values_and_rates_at_epochs() {
        let model = ResolvedPhaseFitModel::Hermite;
        let packed = fit_fringe_series(&[0.0, 10.0], &[0.0, 100.0], &[0.0, 0.0], &model).unwrap();
        assert!((evaluate_hermite(2.5, &packed) - 15.625).abs() < 1e-12);
        assert_eq!(evaluate_hermite(0.0, &packed), 0.0);
        assert_eq!(evaluate_hermite(10.0, &packed), 100.0);
        for t in [0.0, 10.0] {
            let h = 1e-5;
            let slope =
                (evaluate_hermite(t + h, &packed) - evaluate_hermite(t - h, &packed)) / (2.0 * h);
            assert!(slope.abs() < 2e-5);
        }
        let packed = fit_fringe_series(&[0.0, 10.0], &[20.0, 60.0], &[4.0, 4.0], &model).unwrap();
        assert_eq!(evaluate_hermite(-5.0, &packed), 0.0);
        assert_eq!(evaluate_hermite(15.0, &packed), 80.0);
        assert!(fit_fringe_series(&[0.0, 10.0], &[20.0, 60.0], &[], &model).is_err());
        assert!(matches!(
            parse_phase_fit_spec(Some("hermite")).unwrap(),
            PhaseFitSpec::Hermite
        ));
    }

    #[test]
    fn searched_phase_and_delay_reconstruct_original_visibility_frame() {
        use crate::api::{delay_search, LibraryOptions};
        use crate::header::CorHeader;
        use crate::read::CorData;
        use chrono::TimeZone;
        let fs = 64e6;
        let nu0 = 8.4e9;
        let fft = 64;
        let options = LibraryOptions {
            cpu: 1,
            iter: 4,
            ..LibraryOptions::default()
        };
        let phase = |t: f64| 42.0 + 360.0 * 0.037 * t;
        let delay = |t: f64| 0.35 / fs + 0.037 * t / nu0;
        let raw_spectrum = |t: f64| -> Vec<C32> {
            (0..fft / 2)
                .map(|k| {
                    let angle =
                        (phase(t) + 360.0 * k as f64 * fs / fft as f64 * delay(t)).to_radians();
                    C32::new(angle.cos() as f32, angle.sin() as f32)
                })
                .collect()
        };
        let times = [100.0, 1900.0, 3700.0];
        let mut phases = Vec::new();
        let mut delays = Vec::new();
        let mut rates = Vec::new();
        for t in times {
            // Non-power-of-two integration also exercises the padded API path.
            let data = CorData {
                header: CorHeader {
                    fft_point: fft as i32,
                    sampling_speed: fs as i32,
                    observing_frequency: nu0,
                    number_of_sector: 600,
                    station1_name: "ANT1".into(),
                    station2_name: "ANT2".into(),
                    station1_position: [6_378_137.0, 0.0, 0.0],
                    station2_position: [6_378_137.0, 100.0, 0.0],
                    ..CorHeader::default()
                },
                visibility: (0..30).flat_map(|i| raw_spectrum(t + i as f64)).collect(),
                obs_time: Utc.timestamp_opt(t as i64, 0).unwrap(),
                effective_integ_time: 1.0,
                first_sector: 0,
                sectors_read: 30,
            };
            let found = delay_search(&data, &options).unwrap().analysis;
            let phase_error =
                (found.delay_phase as f64 - phase(t) + 180.0).rem_euclid(360.0) - 180.0;
            assert!(
                phase_error.abs() < 0.5,
                "epoch={t}, phase={}, expected={}, rate={}, delay={}",
                found.delay_phase,
                phase(t),
                found.residual_rate,
                found.residual_delay
            );
            assert!((found.residual_rate as f64 - 0.037).abs() < 1e-4);
            phases.push(found.delay_phase);
            delays.push(found.residual_delay as f64 / fs);
            rates.push(found.residual_rate);
        }
        let phases = connect_phase(&times, &phases, &rates).unwrap();
        let model = ResolvedPhaseFitModel::Linear;
        let p = fit_series(&times, &phases, &model).unwrap();
        let d = fit_series(&times, &delays, &model).unwrap();
        for t in [430.0, 2100.0, 3200.0] {
            let mut spectrum = raw_spectrum(t);
            evaluate_fringe_model(t, &p, &d, &model).apply(&mut spectrum, fs, fft);
            for sample in spectrum.iter().skip(1) {
                assert!(
                    (*sample - C32::new(1.0, 0.0)).norm() < 0.015,
                    "t={t}, sample={sample}"
                );
            }
        }
    }

    #[test]
    fn connects_multiple_turns_across_scan_gaps_and_fits_at_common_epoch() {
        for sign in [-1.0, 1.0] {
            let times = [0.0, 30.0, 1800.0, 1830.0, 3600.0];
            let expected: Vec<f64> = times
                .iter()
                .map(|&t| 42.0 + sign * (72.0 * t + 0.001 * t * t))
                .collect();
            let wrapped: Vec<f32> = expected
                .iter()
                .map(|&p| ((p + 180.0).rem_euclid(360.0) - 180.0) as f32)
                .collect();
            let rates: Vec<f32> = times
                .iter()
                .map(|&t| (sign * (72.0 + 0.002 * t) / 360.0) as f32)
                .collect();
            let connected = connect_phase(&times, &wrapped, &rates).unwrap();
            for (&actual, &expected) in connected.iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-4);
            }
            let coeffs = fitting::fit_polynomial_least_squares(&times, &connected, 2).unwrap();
            // Target starts later than the calibrator; retain the calibrator epoch
            // and its nonzero phase intercept when evaluating the correction.
            let t = 330.0;
            let phase = evaluate_phase_fit_model(t, &coeffs, &ResolvedPhaseFitModel::Poly2);
            let visibility = C32::from_polar(
                1.0,
                ((42.0 + sign * (72.0 * t + 0.001 * t * t))
                    .rem_euclid(360.0)
                    .to_radians()) as f32,
            );
            assert!((visibility * phase_rotation(phase) - C32::new(1.0, 0.0)).norm() < 1e-5);
        }
    }

    #[test]
    fn connection_rejects_missing_rates_and_invalid_epochs() {
        assert!(connect_phase(&[0.0, 1.0], &[0.0, 1.0], &[] as &[f64]).is_err());
        assert!(connect_phase(&[0.0, 0.0], &[0.0, 1.0], &[0.0, 0.0]).is_err());
        assert!(connect_phase(&[0.0], &[0.0], &[f32::NAN]).is_err());
    }

    #[test]
    fn rotation_preserves_precision_after_many_turns() {
        let expected = phase_rotation(17.25);
        let actual = phase_rotation(360.0 * 1e9 + 17.25);
        assert!((actual - expected).norm() < 1e-6);
    }

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
