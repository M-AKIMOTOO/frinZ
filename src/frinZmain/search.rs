// Search routines keep delay/rate and observation metadata explicit.
#![allow(clippy::too_many_arguments)]
pub use acel::run_acel_search_analysis;
pub use deep::{
    run_deep2_search, run_deep_search, run_deep_search_with_phase_times, run_peak_search,
    run_peak_search as run_coherent_search, run_peak_search_with_phase_times,
    run_peak_search_with_phase_times as run_coherent_search_with_phase_times, DeepSearchParams,
    DeepSearchResult, GroupDelayEstimate,
};

mod acel {
    use std::error::Error;
    use std::fs::{self, File};
    use std::io::{Cursor, Write};
    use std::path::Path;

    use chrono::{DateTime, Utc};
    use num_complex::Complex;

    use crate::args::Args;
    use crate::fft::apply_phase_correction_in_place_at_frequency;
    use crate::fitting;
    use crate::header::{parse_header, CorHeader};
    use crate::input_support::open_input_data;
    use crate::plot::plot_acel_search_result;
    use crate::read::read_visibility_data;
    use crate::rfi::parse_rfi_ranges;
    use crate::utils::unwrap_phase_with_rate;

    type C32 = Complex<f32>;
    type PhaseExtraction = (Vec<f64>, Vec<f32>, Vec<f32>, Vec<f32>);

    struct VisibilityDataPoint {
        complex_vec: Vec<C32>,
        obs_time: DateTime<Utc>,
        sector_count: i32,
    }

    fn collect_visibility_data(
        cursor: &mut Cursor<&[u8]>,
        header: &CorHeader,
        args: &Args,
        time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
        pp_flag_ranges: &[(u32, u32)],
    ) -> Result<Vec<VisibilityDataPoint>, Box<dyn Error>> {
        let mut collected_data = Vec::new();
        cursor.set_position(256); // Reset cursor to after header

        for loop_idx in 0..args.loop_ {
            let (complex_vec, current_obs_time, _eff_integ_time) = match read_visibility_data(
                cursor,
                header,
                args.length,
                args.skip,
                loop_idx,
                false,
                pp_flag_ranges,
            ) {
                Ok(data) => data,
                Err(_) => break, // Stop if we can't read more data
            };

            if complex_vec.is_empty() {
                break; // Stop if no data was read
            }

            let fft_point_half = (header.fft_point / 2) as usize;
            if fft_point_half == 0 {
                eprintln!("#ERROR: FFT point が 0 です (acel-search)");
                break;
            }
            if complex_vec.len() % fft_point_half != 0 {
                eprintln!(
                    "#ERROR: 複素データ長 ({}) が FFT チャンネル数 ({}) の整数倍ではありません (acel-search)。",
                    complex_vec.len(),
                    fft_point_half
                );
                continue;
            }
            let sector_count = (complex_vec.len() / fft_point_half) as i32;
            if sector_count == 0 {
                continue;
            }

            let is_flagged = time_flag_ranges
                .iter()
                .any(|(start, end)| current_obs_time >= *start && current_obs_time < *end);

            if is_flagged {
                println!(
                    "#INFO: Skipping data at {} due to --flagging time range in acel-search.",
                    current_obs_time.format("%Y-%m-%d %H:%M:%S")
                );
                continue;
            }

            collected_data.push(VisibilityDataPoint {
                complex_vec,
                obs_time: current_obs_time,
                sector_count,
            });
        }
        Ok(collected_data)
    }

    fn get_phases_from_collected_data(
        collected_data: &[VisibilityDataPoint],
        header: &CorHeader,
        args: &Args,
        effective_integ_time: f32,
        obs_time_start: DateTime<Utc>,
        current_total_rate_correct: f32,
        current_total_acel_correct: f32,
        rfi_ranges: &[(usize, usize)],
        bandpass_data: &Option<Vec<C32>>,
    ) -> Result<PhaseExtraction, Box<dyn Error>> {
        let mut phases_collected: Vec<f32> = Vec::new();
        let mut times_collected: Vec<f64> = Vec::new();
        let mut residual_rates_hz: Vec<f32> = Vec::new();
        let mut residual_delays_samples: Vec<f32> = Vec::new();

        // --search acel estimates acceleration from a time series of fringe phases.
        // Therefore each phase/delay/rate measurement must use the same accurate
        // peak search settings as the normal `--search peak` mode.
        //
        // The top-level `--search acel` path bypasses the usual primary-search
        // normalization in main.rs, so set the peak parameters explicitly here.
        let mut peak_args = args.clone();
        peak_args.search = vec!["peak".to_string()];
        peak_args.rate_padding = 1;
        if peak_args.iter < 4 {
            peak_args.iter = 4;
        }
        // Apply the accumulated Taylor correction explicitly with the file-start
        // epoch below. The per-window peak search must then see only residuals;
        // otherwise run_analysis_pipeline references the correction to each
        // window start and cannot flatten phase between windows.
        peak_args.delay_correct = 0.0;
        peak_args.rate_correct = 0.0;
        peak_args.acel_correct = 0.0;
        peak_args.jerk_correct = 0.0;
        peak_args.snap_correct = 0.0;

        for data_point in collected_data {
            let start_time_offset_sec = data_point
                .obs_time
                .signed_duration_since(obs_time_start)
                .num_milliseconds() as f32
                / 1000.0;

            if data_point.sector_count <= 0 {
                continue;
            }
            let current_length = data_point.sector_count;
            let fft_point_half = data_point.complex_vec.len() / current_length as usize;
            if fft_point_half == 0 {
                continue;
            }
            let effective_fft_point = (fft_point_half * 2) as i32;
            let mut corrected_complex_vec = data_point.complex_vec.clone();
            apply_phase_correction_in_place_at_frequency(
                &mut corrected_complex_vec,
                fft_point_half,
                current_total_rate_correct,
                args.delay_correct,
                current_total_acel_correct,
                args.jerk_correct,
                args.snap_correct,
                effective_integ_time,
                header.sampling_speed as u32,
                effective_fft_point as u32,
                start_time_offset_sec,
                header.observing_frequency,
            );

            // Use the same iterative peak search as the normal CLI path. A
            // single FFT peak can select the 1/window-spacing rate alias and
            // then gives the unwrap helper the wrong branch information.
            let search_result = super::deep::run_peak_search(
                &corrected_complex_vec,
                header,
                current_length,
                current_length,
                effective_integ_time,
                &data_point.obs_time,
                &obs_time_start,
                rfi_ranges,
                bandpass_data,
                &peak_args,
                header.number_of_sector,
                peak_args.cpu,
                None,
            )?;
            let analysis_results = search_result.analysis_results;
            let phase_rad = analysis_results.delay_phase.to_radians();

            phases_collected.push(phase_rad);
            times_collected.push(start_time_offset_sec as f64);
            residual_rates_hz.push(analysis_results.residual_rate);
            residual_delays_samples.push(analysis_results.residual_delay);
        }

        // Consecutive windows are separated by args.length seconds. Their
        // wrapped phases alone cannot distinguish rates separated by
        // 1/args.length Hz (0.1 Hz for 10-second windows). Use the independently
        // searched residual rates to choose the correct integer phase turn.
        let times_f32: Vec<f32> = times_collected.iter().map(|&time| time as f32).collect();
        let phases_deg: Vec<f32> = phases_collected
            .iter()
            .map(|&phase| phase.to_degrees())
            .collect();
        phases_collected = unwrap_phase_with_rate(&phases_deg, &times_f32, &residual_rates_hz)
            .into_iter()
            .map(f32::to_radians)
            .collect();
        Ok((
            times_collected,
            phases_collected,
            residual_rates_hz,
            residual_delays_samples,
        ))
    }

    pub fn run_acel_search_analysis(
        args: &Args,
        acel_search_degrees: &[i32],
        time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
        pp_flag_ranges: &[(u32, u32)],
    ) -> Result<(), Box<dyn Error>> {
        println!("Starting acceleration search analysis...");

        let input_path = args.input.as_ref().unwrap();

        // --- Create Output Directory ---
        let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
        let output_dir = parent_dir.join("frinZ").join("acel_search");
        fs::create_dir_all(&output_dir)?;
        let base_filename = input_path.file_stem().unwrap().to_str().unwrap();

        let buffer = open_input_data(input_path)?;
        let mut cursor = Cursor::new(buffer.as_slice());

        let header = parse_header(&mut cursor)?;

        let mut total_acel_correct = args.acel_correct;
        let mut total_rate_correct = args.rate_correct;

        let bandwidth_mhz = header.sampling_speed as f32 / 2.0 / 1_000_000.0;
        let rbw_mhz = bandwidth_mhz / header.fft_point as f32 * 2.0;
        let rfi_ranges = parse_rfi_ranges(&args.rfi, rbw_mhz)?;
        let bandpass_data: Option<Vec<C32>> = None;

        // Get effective_integ_time from the first sector
        cursor.set_position(0);
        let (_, _, effective_integ_time) =
            read_visibility_data(&mut cursor, &header, 1, 0, 0, false, pp_flag_ranges)?;

        // Helper function to write data
        let write_fit_data = |path: &Path, coeffs: Option<&[f64]>| -> std::io::Result<()> {
            let file = File::create(path)?;
            let mut writer = std::io::BufWriter::new(file);
            if let Some(c) = coeffs {
                if c.len() == 3 {
                    // Quadratic
                    writeln!(
                        writer,
                        "# Fitted: y = {:.6e} * x^2 + {:.6e} * x + {:.6e}",
                        c[2], c[1], c[0]
                    )?;
                    writeln!(
                        writer,
                        "# Corrected Acel (Hz/s): {:.6e} (from x^2 / PI)",
                        c[2] / std::f64::consts::PI
                    )?;
                    writeln!(
                        writer,
                        "# Corrected Rate (Hz): {:.6e} (from x / (2 * PI))",
                        c[1] / (2.0 * std::f64::consts::PI)
                    )?;
                } else if c.len() == 2 {
                    // Linear
                    writeln!(writer, "# Fitted: y = {:.6e} * x + {:.6e}", c[1], c[0])?;
                    writeln!(
                        writer,
                        "# Corrected Rate: {:.6e} (from x / (2 * PI))",
                        c[1] / (2.0 * std::f64::consts::PI)
                    )?;
                }
            }
            Ok(())
        };

        // Initialize obs_time_start once before the loop
        cursor.set_position(256); // Reset cursor for first data read
        let (_, first_obs_time, _) =
            read_visibility_data(&mut cursor, &header, 1, 0, 0, false, pp_flag_ranges)?;
        let obs_time_start = first_obs_time;

        // Collect all visibility data once to avoid re-reading
        let collected_data =
            collect_visibility_data(&mut cursor, &header, args, time_flag_ranges, pp_flag_ranges)?;

        for (step_idx, &degree) in acel_search_degrees.iter().enumerate() {
            println!("Step {}: Fitting with degree {}", step_idx + 1, degree);

            // Get phases from the pre-collected data with current corrections
            let (times_for_fit, phases_for_fit, residual_rates_hz, residual_delays_samples) =
                get_phases_from_collected_data(
                    &collected_data,
                    &header,
                    args,
                    effective_integ_time,
                    obs_time_start,
                    total_rate_correct,
                    total_acel_correct,
                    &rfi_ranges,
                    &bandpass_data,
                )?;
            let phases_f64: Vec<f64> = phases_for_fit.iter().map(|&p| p as f64).collect();
            let rates_f64: Vec<f64> = residual_rates_hz.iter().map(|&r| r as f64).collect();
            let mut rate_fit_series: Option<Vec<f64>> = None;
            let mut rate_residual_series: Option<Vec<f64>> = None;
            let rate_based_acel = if rates_f64.len() >= 2 {
                match fitting::fit_linear_least_squares(&times_for_fit, &rates_f64) {
                    Ok((slope, intercept)) => {
                        let fitted: Vec<f64> = times_for_fit
                            .iter()
                            .map(|&t| slope * t + intercept)
                            .collect();
                        let residuals_vec: Vec<f64> = rates_f64
                            .iter()
                            .zip(fitted.iter())
                            .map(|(&obs, &fit)| obs - fit)
                            .collect();
                        rate_fit_series = Some(fitted);
                        rate_residual_series = Some(residuals_vec);
                        Some(slope)
                    }
                    Err(err) => {
                        eprintln!(
                            "Warning: Rate-based linear fit failed in acel-search step {}: {}",
                            step_idx + 1,
                            err
                        );
                        None
                    }
                }
            } else {
                None
            };

            let delays_samples_f64: Vec<f64> =
                residual_delays_samples.iter().map(|&d| d as f64).collect();
            let mut delay_fit_samples_series: Option<Vec<f64>> = None;
            let mut delay_residual_samples_series: Option<Vec<f64>> = None;
            let (delay_based_acel, delay_based_rate) = if delays_samples_f64.len() >= 3 {
                let sampling_hz = header.sampling_speed as f64;
                if sampling_hz > 0.0 {
                    let delays_seconds: Vec<f64> = delays_samples_f64
                        .iter()
                        .map(|&d| d / sampling_hz)
                        .collect();
                    match fitting::fit_polynomial_least_squares(&times_for_fit, &delays_seconds, 2)
                    {
                        Ok(coeffs) => {
                            let fitted_seconds: Vec<f64> = times_for_fit
                                .iter()
                                .map(|&t| coeffs[0] + coeffs[1] * t + coeffs[2] * t * t)
                                .collect();
                            let residual_seconds: Vec<f64> = delays_seconds
                                .iter()
                                .zip(fitted_seconds.iter())
                                .map(|(&obs, &fit)| obs - fit)
                                .collect();
                            delay_fit_samples_series =
                                Some(fitted_seconds.iter().map(|v| v * sampling_hz).collect());
                            delay_residual_samples_series =
                                Some(residual_seconds.iter().map(|v| v * sampling_hz).collect());

                            let acel = 2.0 * coeffs[2] * header.observing_frequency;
                            let rate = coeffs[1] * header.observing_frequency;
                            (Some(acel), Some(rate))
                        }
                        Err(err) => {
                            eprintln!(
                                "Warning: Delay-based quadratic fit failed in acel-search step {}: {}",
                                step_idx + 1,
                                err
                            );
                            (None, None)
                        }
                    }
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            let mut phase_fit_series: Option<Vec<f64>> = None;
            let mut phase_residual_series: Option<Vec<f64>> = None;

            if times_for_fit.len() < (degree + 1) as usize {
                eprintln!(
                    "Warning: Not enough data points for degree {} fitting (need at least {}). Skipping this step.",
                    degree, degree + 1
                );
                println!(
                    "  Updated acel: {:.6e}, Updated rate: {:.6e}",
                    total_acel_correct, total_rate_correct
                );
                continue;
            }

            match degree {
                2 => {
                    // Quadratic Fit
                    let quad_path = output_dir.join(format!(
                        "{}_step{}_quadric.txt",
                        base_filename,
                        step_idx + 1
                    ));
                    if let Ok(coeffs) =
                        fitting::fit_polynomial_least_squares(&times_for_fit, &phases_f64, 2)
                    {
                        println!(
                            "  Quad fit: x^2={:.6e}, x={:.6e}, c={:.6e}",
                            coeffs[2], coeffs[1], coeffs[0]
                        );
                        let fitted_phases: Vec<f64> = times_for_fit
                            .iter()
                            .map(|&t| coeffs[0] + coeffs[1] * t + coeffs[2] * t * t)
                            .collect();
                        let residual_phases: Vec<f64> = phases_f64
                            .iter()
                            .zip(fitted_phases.iter())
                            .map(|(&obs, &fit)| obs - fit)
                            .collect();
                        phase_fit_series = Some(fitted_phases);
                        phase_residual_series = Some(residual_phases);
                        total_acel_correct += (coeffs[2] / std::f64::consts::PI) as f32;
                        total_rate_correct += (coeffs[1] / (2.0 * std::f64::consts::PI)) as f32;
                        write_fit_data(&quad_path, Some(&coeffs))?;
                    } else {
                        eprintln!("Warning: Quadratic fitting failed. Skipping acel and quad-rate update for this step.");
                        write_fit_data(&quad_path, None)?;
                    }
                }
                1 => {
                    // Linear Fit
                    let linear_path = output_dir.join(format!(
                        "{}_step{}_linear.txt",
                        base_filename,
                        step_idx + 1
                    ));
                    if let Ok((slope, intercept)) =
                        fitting::fit_linear_least_squares(&times_for_fit, &phases_f64)
                    {
                        let fitted_phases: Vec<f64> = times_for_fit
                            .iter()
                            .map(|&t| slope * t + intercept)
                            .collect();
                        let residual_phases: Vec<f64> = phases_f64
                            .iter()
                            .zip(fitted_phases.iter())
                            .map(|(&obs, &fit)| obs - fit)
                            .collect();
                        phase_fit_series = Some(fitted_phases);
                        phase_residual_series = Some(residual_phases);
                        write_fit_data(&linear_path, Some(&[intercept, slope]))?;
                        println!("  Linear fit: m={:.6e}", slope);
                        total_rate_correct += (slope / (2.0 * std::f64::consts::PI)) as f32;
                    } else {
                        eprintln!("Warning: Linear fitting failed. Skipping linear-rate update for this step.");
                        write_fit_data(&linear_path, None)?;
                    }
                }
                _ => {
                    eprintln!(
                        "Error: Unsupported fitting degree {}. Skipping this step.",
                        degree
                    );
                }
            }

            println!(
                "  +----------------------+--------------------------+--------------------------+"
            );
            println!(
                "  | Derivation Method    | Acceleration (Hz/s)      | Rate (Hz)                |"
            );
            println!(
                "  +----------------------+--------------------------+--------------------------+"
            );

            // Phase Fit
            println!(
                "  | Phase Fit (Quad)     | {:<+24.9e} | {:<+24.9e} |",
                total_acel_correct, total_rate_correct
            );

            // Rate-derived
            let rate_acel_str = rate_based_acel
                .map(|v| format!("{:<+24.9e}", v))
                .unwrap_or_else(|| format!("{:<24}", "(N/A)"));
            println!(
                "  | Rate-derived         | {} | {:<24} |",
                rate_acel_str, "(N/A)"
            );

            // Delay-derived
            let delay_acel_str = delay_based_acel
                .map(|v| format!("{:<+24.9e}", v))
                .unwrap_or_else(|| format!("{:<24}", "(N/A)"));
            let delay_rate_str = delay_based_rate
                .map(|v| format!("{:<+24.9e}", v))
                .unwrap_or_else(|| format!("{:<24}", "(N/A)"));
            println!(
                "  | Delay-derived        | {} | {} |",
                delay_acel_str, delay_rate_str
            );

            println!(
                "  +----------------------+--------------------------+--------------------------+"
            );

            // Copypaste lines
            println!(
                "  Copypaste (Phase Fit): --acel {:.18} --rate {:.15}",
                total_acel_correct, total_rate_correct
            );
            if let Some(rate_acel) = rate_based_acel {
                println!("  Copypaste (Rate-derived): --acel {:.18}", rate_acel);
            }
            if let (Some(delay_acel), Some(delay_rate)) = (delay_based_acel, delay_based_rate) {
                println!(
                    "  Copypaste (Delay-derived): --acel {:.18} --rate {:.15}",
                    delay_acel, delay_rate
                );
            }

            if let Some(fitted) = &phase_fit_series {
                let residuals = phase_residual_series.as_deref();
                let phase_plot_path =
                    output_dir.join(format!("{}_step{}_phase.png", base_filename, step_idx + 1));
                plot_acel_search_result(
                    &phase_plot_path,
                    &times_for_fit,
                    &phases_f64,
                    Some(fitted.as_slice()),
                    residuals,
                    &format!("Phase Fit (step {})", step_idx + 1),
                    "Phase [rad]",
                )?;
            }

            if let Some(fitted) = &rate_fit_series {
                let residuals = rate_residual_series.as_deref();
                let rate_plot_path = output_dir.join(format!(
                    "{}_step{}_res_rate.png",
                    base_filename,
                    step_idx + 1
                ));
                plot_acel_search_result(
                    &rate_plot_path,
                    &times_for_fit,
                    &rates_f64,
                    Some(fitted.as_slice()),
                    residuals,
                    &format!("Residual Rate Fit (step {})", step_idx + 1),
                    "Residual Rate [Hz]",
                )?;
            }

            if let Some(fitted) = &delay_fit_samples_series {
                let residuals = delay_residual_samples_series.as_deref();
                let delay_plot_path = output_dir.join(format!(
                    "{}_step{}_res_delay.png",
                    base_filename,
                    step_idx + 1
                ));
                plot_acel_search_result(
                    &delay_plot_path,
                    &times_for_fit,
                    &delays_samples_f64,
                    Some(fitted.as_slice()),
                    residuals,
                    &format!("Residual Delay Fit (step {})", step_idx + 1),
                    "Residual Delay [sample]",
                )?;
            }
        }

        Ok(())
    }
}

mod deep {
    mod group_delay;
    use chrono::{DateTime, Utc};
    pub use group_delay::GroupDelayEstimate;
    use ndarray::prelude::*;
    use num_complex::Complex;
    use rayon::prelude::*;
    use std::borrow::Cow;
    use std::error::Error;
    use std::f64::consts::PI;
    use std::sync::{Arc, Mutex, OnceLock};

    use crate::analysis::{analyze_results, AnalysisResults};
    use crate::args::Args;
    use crate::bandpass::{apply_bandpass_gains, bandpass_gains};
    use crate::fft::{
        apply_phase_correction_in_place_at_times, process_fft, process_ifft, DelayTransform,
    };
    use crate::header::CorHeader;
    use crate::rfi::has_histogram_mode;
    use crate::utils::{
        delay_rate_mask_bounds, in_delay_rate_mask, in_window, rate_cal, window_bounds,
    };

    type C32 = Complex<f32>;
    fn search_pool(threads: usize) -> Result<Arc<rayon::ThreadPool>, Box<dyn Error>> {
        type PoolCache = Option<(usize, Arc<rayon::ThreadPool>)>;
        static POOL: OnceLock<Mutex<PoolCache>> = OnceLock::new();
        let mut cached = POOL
            .get_or_init(|| Mutex::new(None))
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some((count, pool)) = cached.as_ref() {
            if *count == threads {
                return Ok(Arc::clone(pool));
            }
        }
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?,
        );
        *cached = Some((threads, Arc::clone(&pool)));
        Ok(pool)
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum DeepSearchAlgorithm {
        FullGrid,
        AxisThenLocal,
        PeakPolish,
    }

    impl DeepSearchAlgorithm {
        fn mode_name(self) -> &'static str {
            match self {
                Self::FullGrid => "deep",
                Self::AxisThenLocal => "deep2",
                Self::PeakPolish => "peak",
            }
        }
    }

    /// Deep searchで使用する探索パラメータ
    #[derive(Debug, Clone)]
    pub struct DeepSearchParams {
        pub delay_fine_step: f32,          // 0.1 sample
        pub rate_fine_step_factor: f32,    // 1/(10*pp)
        pub delay_search_range: f32,       // ±0.5 sample
        pub rate_search_range_factor: f32, // ±1/(2*pp) Hz

        pub max_iterations: usize, // 階層の深さ
    }

    impl Default for DeepSearchParams {
        fn default() -> Self {
            Self {
                delay_fine_step: 0.1,
                rate_fine_step_factor: 0.1,
                delay_search_range: 0.5,
                rate_search_range_factor: 0.5,

                max_iterations: 4,
            }
        }
    }

    /// Deep search探索結果
    #[derive(Debug, Clone)]
    pub struct DeepSearchResult {
        pub analysis_results: AnalysisResults,
        pub freq_rate_array: Option<Array2<C32>>,
        pub delay_rate_2d_data: Array2<C32>,
        pub pre_bandpass_analysis_results: Option<AnalysisResults>,
    }

    fn needs_frequency_domain_coarse_search(args: &Args) -> bool {
        // Only --frequency changes the domain in which the coarse fringe
        // candidate is selected. Diagnostic/output options must not change
        // delay, rate, amplitude, noise, or SNR.
        args.frequency
    }

    struct DeepSearchContext<'a> {
        complex_vec: &'a [C32],
        header: &'a CorHeader,
        current_length: i32,
        physical_length: i32,
        effective_integ_time: f32,
        current_obs_time: &'a DateTime<Utc>,
        rfi_ranges: &'a [(usize, usize)],
        args: &'a Args,
        effective_fft_point: i32,
        rfi_mask: Vec<bool>,
        bandpass_gains: Option<Vec<Complex<f64>>>,
        phase_times: &'a [f64],
        static_phase_cycles: Vec<f64>,
    }

    // A continuous search window can fall entirely between FFT bins. Use the
    // closest bin as its coarse seed; fine trials still obey the exact window.
    fn coarse_window(axis: impl Iterator<Item = f32>, window: &[f32]) -> Option<(f32, f32)> {
        let (low, high) = window_bounds(window)?;
        let mut nearest = None;
        let mut distance = f32::INFINITY;
        for value in axis {
            if (low..=high).contains(&value) {
                return Some((low, high));
            }
            let gap = (low - value).max(value - high);
            if gap < distance {
                distance = gap;
                nearest = Some((value, value));
            }
        }
        nearest
    }

    fn build_rfi_mask(channel_count: usize, rfi_ranges: &[(usize, usize)]) -> Vec<bool> {
        let mut mask = vec![false; channel_count];
        for &(min, max) in rfi_ranges {
            if min >= channel_count {
                continue;
            }
            let end = max.min(channel_count.saturating_sub(1));
            if end >= min {
                mask[min..=end].fill(true);
            }
        }
        mask
    }

    /// Advance four independent channel phases so the coherent sum does not
    /// serialize an entire spectrum on one complex multiplication chain.
    fn add_rotated_channels(
        sums: &mut [Complex<f64>],
        samples: &[C32],
        first_phase: Complex<f64>,
        step: Complex<f64>,
    ) {
        let step2 = step * step;
        let step4 = step2 * step2;
        let mut phases = [
            first_phase,
            first_phase * step,
            first_phase * step2,
            first_phase * step2 * step,
        ];
        let (sample_groups, tail) = samples.as_chunks::<4>();
        let (sum_groups, sum_tail) = sums.as_chunks_mut::<4>();
        for (group, sums) in sample_groups.iter().zip(sum_groups) {
            for i in 0..4 {
                sums[i] += Complex::new(f64::from(group[i].re), f64::from(group[i].im)) * phases[i];
                phases[i] *= step4;
            }
        }
        for ((&sample, sum), phase) in tail.iter().zip(sum_tail).zip(phases) {
            *sum += Complex::new(f64::from(sample.re), f64::from(sample.im)) * phase;
        }
    }

    impl<'a> DeepSearchContext<'a> {
        fn fft_for_correction(&self, delay: f64, rate: f64) -> (Array2<C32>, usize) {
            const RATE_PADDING: u32 = 1;
            let (mut freq_rate_array, padding_length) = if rate == 0.0
                && delay == 0.0
                && self.args.acel_correct == 0.0
                && self.args.jerk_correct == 0.0
                && self.args.snap_correct == 0.0
            {
                process_fft(
                    self.complex_vec,
                    self.physical_length,
                    self.effective_fft_point,
                    self.header.sampling_speed,
                    self.rfi_ranges,
                    RATE_PADDING,
                )
            } else {
                let mut corrected = self.complex_vec.to_vec();
                // Library callers may append zero-padding rows. Only the
                // physical rows have epochs; including padding makes the
                // correction helper reject the entire phase model.
                let physical_samples =
                    self.phase_times.len() * (self.effective_fft_point / 2) as usize;
                apply_phase_correction_in_place_at_times(
                    &mut corrected[..physical_samples],
                    (self.effective_fft_point / 2) as usize,
                    rate,
                    delay,
                    self.args.acel_correct,
                    self.args.jerk_correct,
                    self.args.snap_correct,
                    self.header.sampling_speed as u32,
                    self.effective_fft_point as u32,
                    self.phase_times,
                    self.header.observing_frequency,
                    self.args.channel_origin_bins,
                );
                process_fft(
                    &corrected,
                    self.physical_length,
                    self.effective_fft_point,
                    self.header.sampling_speed,
                    self.rfi_ranges,
                    RATE_PADDING,
                )
            };
            if let Some(mask) = self.args.rfi_npz_mask.as_deref() {
                let (frequency_count, rate_count) = freq_rate_array.dim();
                if let Some(values) = freq_rate_array.as_slice_mut() {
                    mask.apply_frequency_rate(
                        values,
                        frequency_count,
                        rate_count,
                        self.header.sampling_speed,
                        self.effective_fft_point,
                        self.effective_integ_time,
                    );
                }
            }
            (freq_rate_array, padding_length)
        }

        fn apply_bandpass(&self, freq_rate_array: &mut Array2<C32>) {
            if let Some(gains) = &self.bandpass_gains {
                apply_bandpass_gains(freq_rate_array, gains.iter().copied());
            }
        }

        fn coarse_delay_peak(
            &self,
            frequencies: &Array2<C32>,
            padding_length: usize,
        ) -> (f32, f32) {
            let fft_point = self.effective_fft_point as usize;
            let half = (fft_point / 2) as f32;
            let rate_axis = rate_cal(padding_length as f32, self.effective_integ_time);
            let delay_at = |index: usize| index as f32 - half + 1.0;
            let delay_window = coarse_window((0..fft_point).map(delay_at), &self.args.drange);
            let rate_window = coarse_window(rate_axis.iter().copied(), &self.args.rrange);
            let mask = delay_rate_mask_bounds(&self.args.mask);
            let mut transform = DelayTransform::new(fft_point, 1);
            let mut max_power = -1.0;
            let mut peak = (0.0, 0.0);

            // Only the seed is needed here. Noise and phase are measured on
            // the final, fringe-corrected plane.
            for (rate_index, rate) in rate_axis.into_iter().enumerate() {
                if !in_window(rate, rate_window) {
                    continue;
                }
                for (index, value) in transform
                    .delay_bins(frequencies.column(rate_index))
                    .enumerate()
                {
                    let delay = delay_at(index);
                    let power = value.norm_sqr();
                    if power > max_power
                        && in_window(delay, delay_window)
                        && !in_delay_rate_mask(delay, rate, mask)
                        && !self
                            .args
                            .rfi_npz_mask
                            .as_deref()
                            .is_some_and(|mask| mask.contains_delay_rate(delay, rate))
                    {
                        max_power = power;
                        peak = (delay, rate);
                    }
                }
            }
            peak
        }

        fn apply_delay_rate_mask(&self, values: &mut Array2<C32>) {
            let Some(mask) = self.args.rfi_npz_mask.as_deref() else {
                return;
            };
            let (rate_count, delay_count) = values.dim();
            if let Some(slice) = values.as_slice_mut() {
                mask.apply_delay_rate(
                    slice,
                    rate_count,
                    delay_count,
                    self.effective_integ_time,
                    self.effective_fft_point,
                );
            }
        }

        fn delay_plane_for_analysis(
            &self,
            frequencies: &Array2<C32>,
            padding_length: usize,
            args: &Args,
        ) -> Array2<C32> {
            if args.frequency
                && args.drange.is_empty()
                && args.contamination.is_none()
                && !has_histogram_mode(&args.rfi)
            {
                // analyze_results does not inspect delay pixels in this mode.
                return Array2::zeros((1, 1));
            }
            let mut values = process_ifft(frequencies, self.effective_fft_point, padding_length);
            self.apply_delay_rate_mask(&mut values);
            values
        }

        fn coarse_estimates(&self, algorithm: DeepSearchAlgorithm) -> (f32, f32) {
            let (mut frequencies, padding_length) = self.fft_for_correction(0.0, 0.0);
            self.apply_bandpass(&mut frequencies);
            if !needs_frequency_domain_coarse_search(self.args) && self.current_length > 2 {
                return self.coarse_delay_peak(&frequencies, padding_length);
            }

            let delays = self.delay_plane_for_analysis(&frequencies, padding_length, self.args);
            let mode = if self.args.drange.is_empty() && self.args.rrange.is_empty() {
                Some(algorithm.mode_name())
            } else {
                self.args.primary_search_mode()
            };
            let result = analyze_results(
                &frequencies,
                &delays,
                self.header,
                self.current_length,
                self.effective_integ_time,
                self.current_obs_time,
                padding_length,
                self.args,
                mode,
            );
            (result.residual_delay, result.residual_rate)
        }

        fn channel_rf_increment(&self) -> f64 {
            let reference = self.header.observing_frequency;
            if reference.is_finite() && reference.abs() > f64::EPSILON {
                self.header.sampling_speed as f64 / self.effective_fft_point as f64 / reference
            } else {
                0.0
            }
        }

        // First retained channel (k=1) and phase increment to the next channel.
        fn row_rotation(
            &self,
            row: usize,
            delay: f64,
            rate: f64,
            rf_increment: f64,
        ) -> (Complex<f64>, Complex<f64>) {
            let cycles = rate * self.phase_times[row] + self.static_phase_cycles[row];
            let slope = delay / self.effective_fft_point as f64 + cycles * rf_increment;
            let step = Complex::from_polar(1.0, -2.0 * PI * slope);
            let origin = cycles + self.args.channel_origin_bins * slope;
            (Complex::from_polar(1.0, -2.0 * PI * origin) * step, step)
        }

        fn coherent_spectrum_at_rate(&self, rate: f32, spectrum: &mut [Complex<f64>]) {
            spectrum.fill(Complex::new(0.0, 0.0));
            let channel_count = spectrum.len();
            let rf_increment = self.channel_rf_increment();
            for (row_index, row) in self
                .complex_vec
                .chunks_exact(channel_count)
                .take(self.phase_times.len())
                .enumerate()
            {
                let (first_phase, step) =
                    self.row_rotation(row_index, 0.0, rate.into(), rf_increment);
                add_rotated_channels(&mut spectrum[1..], &row[1..], first_phase, step);
            }

            // Both the channel flags and bandpass gains are constant in time.
            if let Some(gains) = &self.bandpass_gains {
                for (sum, gain) in spectrum.iter_mut().zip(gains) {
                    *sum *= *gain;
                }
            }
            for (sum, &masked) in spectrum.iter_mut().zip(&self.rfi_mask) {
                if masked {
                    *sum = Complex::default();
                }
            }
        }

        fn spectrum_power(&self, spectrum: &[Complex<f64>], delay: f32) -> f64 {
            let angle = -2.0 * PI * delay as f64 / self.effective_fft_point as f64;
            let step = Complex::from_polar(1.0, angle);
            let mut factor = step;
            let mut sum = Complex::<f64>::default();
            for &sample in spectrum.iter().skip(1) {
                sum += sample * factor;
                factor *= step;
            }
            // All candidates share the same amplitude normalization. Compare
            // squared magnitudes before rounding the final reported result.
            sum.norm_sqr()
        }

        fn candidate_is_masked(&self, delay: f32, rate: f32) -> bool {
            !self.args.frequency
                && (in_delay_rate_mask(delay, rate, delay_rate_mask_bounds(&self.args.mask))
                    || self
                        .args
                        .rfi_npz_mask
                        .as_deref()
                        .is_some_and(|mask| mask.contains_delay_rate(delay, rate)))
        }

        fn best_on_grid(
            &self,
            delays: &[f32],
            rates: &[f32],
            pool: &rayon::ThreadPool,
        ) -> Option<(f32, f32)> {
            let delay_bounds = window_bounds(&self.args.drange);
            let rate_bounds = rate_search_bounds(self.effective_integ_time, &self.args.rrange);
            let channel_count = self.effective_fft_point as usize / 2;
            pool.install(|| {
                rates
                    .par_iter()
                    .enumerate()
                    .filter(|(_, rate)| in_window(**rate, Some(rate_bounds)))
                    .map_init(
                        || vec![Complex::<f64>::default(); channel_count],
                        |spectrum, (r, &rate)| {
                            self.coherent_spectrum_at_rate(rate, spectrum);
                            delays
                                .iter()
                                .enumerate()
                                .filter(|(_, delay)| {
                                    in_window(**delay, delay_bounds)
                                        && !self.candidate_is_masked(**delay, rate)
                                })
                                .filter_map(|(d, &delay)| {
                                    let power = self.spectrum_power(spectrum, delay);
                                    power.is_finite().then_some((d, r, power))
                                })
                                .max_by(|a, b| a.2.total_cmp(&b.2).then_with(|| b.0.cmp(&a.0)))
                        },
                    )
                    .flatten_iter()
                    .max_by(|a, b| {
                        a.2.total_cmp(&b.2)
                            .then_with(|| b.0.cmp(&a.0))
                            .then_with(|| a.1.cmp(&b.1))
                    })
                    .map(|(d, r, _)| (delays[d], rates[r]))
            })
        }

        fn refine_peak(
            &self,
            algorithm: DeepSearchAlgorithm,
            cpu_count: u32,
        ) -> Result<(f32, f32), Box<dyn Error>> {
            let (mut delay, mut rate) = self.coarse_estimates(algorithm);
            for (seed, window) in [
                (&mut delay, &self.args.drange),
                (&mut rate, &self.args.rrange),
            ] {
                if let Some((low, high)) = window_bounds(window) {
                    if !(low..=high).contains(seed) {
                        *seed = low.midpoint(high);
                    }
                }
            }
            if self.physical_length <= 1 {
                rate = 0.0;
            }
            let seed = (delay, rate);
            let params = DeepSearchParams {
                max_iterations: self.args.iter.max(1) as usize,
                ..DeepSearchParams::default()
            };
            let duration =
                self.physical_length.max(1) as f32 * self.effective_integ_time.abs().max(1.0e-9);
            let rate_range = 10.0 * params.rate_search_range_factor / (2.0 * duration);
            let delay_range = delay_search_range_for_rate(
                rate,
                duration,
                self.header.sampling_speed,
                self.header.observing_frequency,
                rate_range,
                params.delay_search_range,
            );
            let pool = search_pool(determine_effective_cpu_count(cpu_count))?;
            let rate_grid = |center, range, step| {
                if self.physical_length <= 1 {
                    vec![0.0]
                } else {
                    generate_search_points(center, range, step)
                }
            };
            let mut best = None;
            let mut delay_step = params.delay_fine_step;
            let mut rate_step = params.rate_fine_step_factor / (10.0 * duration);
            for iteration in 0..params.max_iterations {
                let (delay, rate) = best.unwrap_or(seed);
                let scale = 10.0_f32.powi(iteration as i32);
                delay_step = params.delay_fine_step / scale;
                rate_step = params.rate_fine_step_factor / (10.0 * duration) / scale;
                let delays = generate_search_points(delay, delay_range / scale, delay_step);
                let rates = rate_grid(rate, rate_range / scale, rate_step);
                if best.is_some() && delays.as_slice() == [delay] && rates.as_slice() == [rate] {
                    break;
                }
                let candidate = if iteration == 0 || algorithm == DeepSearchAlgorithm::FullGrid {
                    self.best_on_grid(&delays, &rates, &pool)
                } else {
                    let along_rate = self.best_on_grid(&[delay], &rates, &pool);
                    let best_rate = along_rate.map_or(rate, |(_, rate)| rate);
                    self.best_on_grid(&delays, &[best_rate], &pool)
                        .or(along_rate)
                };
                best = candidate.or(best);
            }

            if algorithm != DeepSearchAlgorithm::FullGrid {
                let (delay, rate) = best.unwrap_or(seed);
                let delays = generate_search_points(delay, delay_step, delay_step);
                let rates = rate_grid(rate, rate_step, rate_step);
                best = self.best_on_grid(&delays, &rates, &pool).or(best);
            }
            best.ok_or_else(|| {
                "no finite, unmasked fringe candidate in the requested search windows".into()
            })
        }

        fn final_analysis(&self, final_delay: f64, final_rate: f64) -> DeepSearchResult {
            // Always evaluate the final candidate on the fully corrected FFT plane.
            // The reported noise/SNR must describe the same corrected data whether or
            // not diagnostic plots were requested. The former no-plot shortcut reused
            // the coarse streaming noise estimate and made `--plot` change SNR.
            let (mut final_freq_rate_array, padding_length) =
                self.fft_for_correction(final_delay, final_rate);
            let mut final_args = self.args.clone();
            final_args.delay_correct = final_delay as f32;
            final_args.rate_correct = final_rate as f32;
            final_args.search.clear();
            // The final candidate has already applied the full correction. Measure
            // amplitude and phase at residual delay=rate=0, independent of the
            // absolute search windows or masks used to select the candidate.
            final_args.drange.clear();
            final_args.rrange.clear();
            final_args.mask.clear();

            let pre_bandpass_analysis_results = if self.args.plot && self.bandpass_gains.is_some() {
                let pre_bandpass_delay_rate_2d_data_comp = self.delay_plane_for_analysis(
                    &final_freq_rate_array,
                    padding_length,
                    &final_args,
                );
                Some(analyze_results(
                    &final_freq_rate_array,
                    &pre_bandpass_delay_rate_2d_data_comp,
                    self.header,
                    self.current_length,
                    self.effective_integ_time,
                    self.current_obs_time,
                    padding_length,
                    &final_args,
                    None,
                ))
            } else {
                None
            };

            self.apply_bandpass(&mut final_freq_rate_array);
            let final_delay_rate_2d_data_comp =
                self.delay_plane_for_analysis(&final_freq_rate_array, padding_length, &final_args);
            let mut analysis_results = analyze_results(
                &final_freq_rate_array,
                &final_delay_rate_2d_data_comp,
                self.header,
                self.current_length,
                self.effective_integ_time,
                self.current_obs_time,
                padding_length,
                &final_args,
                None,
            );

            analysis_results.residual_delay = final_delay as f32;
            analysis_results.residual_rate = if self.physical_length <= 1 {
                0.0
            } else {
                final_rate as f32
            };
            analysis_results.length_f32 = self.physical_length as f32 * self.effective_integ_time;

            DeepSearchResult {
                analysis_results,
                freq_rate_array: (self.args.frequency || has_histogram_mode(&self.args.rfi))
                    .then_some(final_freq_rate_array),
                delay_rate_2d_data: final_delay_rate_2d_data_comp,
                pre_bandpass_analysis_results,
            }
        }
    }

    /// Deep searchメイン関数
    pub fn run_deep_search(
        complex_vec: &[C32],
        header: &CorHeader,
        current_length: i32,
        physical_length: i32,
        effective_integ_time: f32,
        current_obs_time: &DateTime<Utc>,
        obs_time: &DateTime<Utc>,
        rfi_ranges: &[(usize, usize)],
        bandpass_data: &Option<Vec<C32>>,
        args: &Args,
        pp: i32,
        cpu_count_arg: u32, // New argument
        previous_solution: Option<(f32, f32)>,
    ) -> Result<DeepSearchResult, Box<dyn Error>> {
        run_deep_search_impl(
            complex_vec,
            header,
            current_length,
            physical_length,
            effective_integ_time,
            current_obs_time,
            obs_time,
            rfi_ranges,
            bandpass_data,
            args,
            pp,
            cpu_count_arg,
            previous_solution,
            DeepSearchAlgorithm::FullGrid,
            None,
        )
    }

    pub fn run_deep_search_with_phase_times(
        complex_vec: &[C32],
        header: &CorHeader,
        current_length: i32,
        physical_length: i32,
        effective_integ_time: f32,
        current_obs_time: &DateTime<Utc>,
        obs_time: &DateTime<Utc>,
        rfi_ranges: &[(usize, usize)],
        bandpass_data: &Option<Vec<C32>>,
        args: &Args,
        pp: i32,
        cpu_count_arg: u32,
        previous_solution: Option<(f32, f32)>,
        phase_times: &[f64],
    ) -> Result<DeepSearchResult, Box<dyn Error>> {
        run_deep_search_impl(
            complex_vec,
            header,
            current_length,
            physical_length,
            effective_integ_time,
            current_obs_time,
            obs_time,
            rfi_ranges,
            bandpass_data,
            args,
            pp,
            cpu_count_arg,
            previous_solution,
            DeepSearchAlgorithm::FullGrid,
            Some(phase_times),
        )
    }

    pub fn run_peak_search(
        complex_vec: &[C32],
        header: &CorHeader,
        current_length: i32,
        physical_length: i32,
        effective_integ_time: f32,
        current_obs_time: &DateTime<Utc>,
        obs_time: &DateTime<Utc>,
        rfi_ranges: &[(usize, usize)],
        bandpass_data: &Option<Vec<C32>>,
        args: &Args,
        pp: i32,
        cpu_count_arg: u32,
        previous_solution: Option<(f32, f32)>,
    ) -> Result<DeepSearchResult, Box<dyn Error>> {
        run_deep_search_impl(
            complex_vec,
            header,
            current_length,
            physical_length,
            effective_integ_time,
            current_obs_time,
            obs_time,
            rfi_ranges,
            bandpass_data,
            args,
            pp,
            cpu_count_arg,
            previous_solution,
            DeepSearchAlgorithm::PeakPolish,
            None,
        )
    }

    pub fn run_peak_search_with_phase_times(
        complex_vec: &[C32],
        header: &CorHeader,
        current_length: i32,
        physical_length: i32,
        effective_integ_time: f32,
        current_obs_time: &DateTime<Utc>,
        obs_time: &DateTime<Utc>,
        rfi_ranges: &[(usize, usize)],
        bandpass_data: &Option<Vec<C32>>,
        args: &Args,
        pp: i32,
        cpu_count_arg: u32,
        previous_solution: Option<(f32, f32)>,
        phase_times: &[f64],
    ) -> Result<DeepSearchResult, Box<dyn Error>> {
        run_deep_search_impl(
            complex_vec,
            header,
            current_length,
            physical_length,
            effective_integ_time,
            current_obs_time,
            obs_time,
            rfi_ranges,
            bandpass_data,
            args,
            pp,
            cpu_count_arg,
            previous_solution,
            DeepSearchAlgorithm::PeakPolish,
            Some(phase_times),
        )
    }

    // Deprecated internal compatibility path.
    // Keep this around for a while as a comparison target, but the public
    // --search peak mode now uses the same AxisThenLocal + final local polish
    // algorithm via run_peak_search().
    #[allow(dead_code)]
    pub fn run_deep2_search(
        complex_vec: &[C32],
        header: &CorHeader,
        current_length: i32,
        physical_length: i32,
        effective_integ_time: f32,
        current_obs_time: &DateTime<Utc>,
        obs_time: &DateTime<Utc>,
        rfi_ranges: &[(usize, usize)],
        bandpass_data: &Option<Vec<C32>>,
        args: &Args,
        pp: i32,
        cpu_count_arg: u32,
        previous_solution: Option<(f32, f32)>,
    ) -> Result<DeepSearchResult, Box<dyn Error>> {
        run_deep_search_impl(
            complex_vec,
            header,
            current_length,
            physical_length,
            effective_integ_time,
            current_obs_time,
            obs_time,
            rfi_ranges,
            bandpass_data,
            args,
            pp,
            cpu_count_arg,
            previous_solution,
            DeepSearchAlgorithm::AxisThenLocal,
            None,
        )
    }

    fn run_deep_search_impl(
        complex_vec: &[C32],
        header: &CorHeader,
        current_length: i32,
        physical_length: i32,
        effective_integ_time: f32,
        current_obs_time: &DateTime<Utc>,
        _obs_time: &DateTime<Utc>,
        rfi_ranges: &[(usize, usize)],
        bandpass_data: &Option<Vec<C32>>,
        args: &Args,
        _pp: i32,
        cpu_count_arg: u32,
        _previous_solution: Option<(f32, f32)>,
        algorithm: DeepSearchAlgorithm,
        recorded_phase_times: Option<&[f64]>,
    ) -> Result<DeepSearchResult, Box<dyn Error>> {
        // Recorded offsets preserve gaps. Residual phase is referenced to
        // the first physical sample of the analyzed segment.
        let measure_group_delay = args.primary_search_mode() == Some("group-delay");
        if measure_group_delay && args.frequency {
            return Err("group delay requires full-band delay-rate analysis".into());
        }
        if current_length <= 0 {
            return Err("有効なセクター長が 0 以下です".into());
        }
        let rows = current_length as usize;
        if rows == 0 || complex_vec.is_empty() {
            return Err("有効なデータが存在しません".into());
        }
        if !complex_vec.len().is_multiple_of(rows) {
            return Err(format!(
                "複素データ長 ({}) がセクター数 ({}) の整数倍ではありません",
                complex_vec.len(),
                rows
            )
            .into());
        }
        let fft_point_half = complex_vec.len() / rows;
        if fft_point_half == 0 {
            return Err("FFT チャンネル数が 0 です".into());
        }
        let effective_fft_point = (fft_point_half * 2) as i32;
        let mut rfi_mask = build_rfi_mask(fft_point_half, rfi_ranges);
        if let Some(mask) = args.rfi_npz_mask.as_deref() {
            let external = mask.frequency_channel_mask(
                fft_point_half,
                header.sampling_speed,
                effective_fft_point,
            );
            for (channel, marked) in external.into_iter().enumerate() {
                if marked {
                    rfi_mask[channel] = true;
                }
            }
        }
        let bandpass_gains = bandpass_data
            .as_deref()
            .map(|data| bandpass_gains(data).take(fft_point_half).collect());
        let phase_row_count = rows.min(physical_length.max(0) as usize);
        let phase_times: Cow<'_, [f64]> = if let Some(times) = recorded_phase_times {
            if times.len() != phase_row_count {
                return Err(format!(
                    "記録時刻数 ({}) と解析行数 ({}) が一致しません",
                    times.len(),
                    phase_row_count
                )
                .into());
            }
            let phase_epoch = times.first().copied().unwrap_or(0.0);
            if phase_epoch == 0.0 {
                Cow::Borrowed(times)
            } else {
                Cow::Owned(times.iter().map(|time| *time - phase_epoch).collect())
            }
        } else {
            Cow::Owned(
                (0..phase_row_count)
                    .map(|row_idx| row_idx as f64 * effective_integ_time as f64)
                    .collect(),
            )
        };
        let static_phase_cycles: Vec<f64> = phase_times
            .iter()
            .map(|&time_sec| {
                0.5 * args.acel_correct as f64 * time_sec * time_sec
                    + (args.jerk_correct as f64 / 6.0) * time_sec * time_sec * time_sec
                    + (args.snap_correct as f64 / 24.0) * time_sec * time_sec * time_sec * time_sec
            })
            .collect();

        let context = DeepSearchContext {
            complex_vec,
            header,
            current_length,
            physical_length,
            effective_integ_time,
            current_obs_time,
            rfi_ranges,
            args,
            effective_fft_point,
            rfi_mask,
            bandpass_gains,
            phase_times: &phase_times,
            static_phase_cycles,
        };

        // Auto-correlations have exactly zero physical delay and fringe rate.
        // Cross-correlations reacquire a coarse peak for each segment.
        let autocorrelation = is_autocorrelation_header(header);
        let (delay, rate) = if autocorrelation {
            (0.0, 0.0)
        } else {
            context.refine_peak(algorithm, cpu_count_arg)?
        };
        let group_delay = if measure_group_delay {
            let fit = if autocorrelation {
                Err("autocorrelation")
            } else {
                context.fit_group_delay(delay.into(), rate.into())
            };
            Some(fit)
        } else {
            None
        };
        let (delay, rate) = match &group_delay {
            Some(Ok(f)) => (f.delay_samples, f.rate_hz),
            _ => (delay.into(), rate.into()),
        };
        let mut output = context.final_analysis(delay, rate);
        output.analysis_results.group_delay = group_delay;
        Ok(output)
    }

    fn is_autocorrelation_header(header: &CorHeader) -> bool {
        let name1 = header.station1_name.trim();
        let name2 = header.station2_name.trim();
        if !name1.is_empty() && !name2.is_empty() {
            return name1.eq_ignore_ascii_case(name2);
        }

        let code1 = header
            .station1_code
            .trim_matches(|c: char| c == '\0' || c.is_whitespace());
        let code2 = header
            .station2_code
            .trim_matches(|c: char| c == '\0' || c.is_whitespace());
        if !code1.is_empty() && !code2.is_empty() {
            return code1.eq_ignore_ascii_case(code2);
        }

        header
            .station1_position
            .iter()
            .zip(header.station2_position.iter())
            .all(|(a, b)| (a - b).abs() < 1.0e-6)
    }

    fn determine_effective_cpu_count(cpu_count_arg: u32) -> usize {
        let num_available_cpus = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        if cpu_count_arg == 0 {
            num_available_cpus
        } else {
            (cpu_count_arg as usize).clamp(1, num_available_cpus)
        }
    }

    fn rate_search_bounds(effective_integ_time: f32, requested: &[f32]) -> (f32, f32) {
        if requested.len() == 2 {
            return (
                requested[0].min(requested[1]),
                requested[0].max(requested[1]),
            );
        }

        // Without an explicit --rrange, return the canonical (unaliased)
        // fringe-rate solution allowed by the time sampling. Very short
        // integrations otherwise let the fine search escape the FFT rate
        // axis and select an equivalent +/- n/dt alias.
        let dt = effective_integ_time.abs();
        let nyquist = if dt.is_finite() && dt > f32::EPSILON {
            0.5 / dt
        } else {
            f32::MAX
        };
        (-nyquist, nyquist)
    }

    fn delay_search_range_for_rate(
        initial_rate: f32,
        duration_sec: f32,
        sampling_speed: i32,
        observing_frequency: f64,
        initial_rate_range: f32,
        fallback_range: f32,
    ) -> f32 {
        let mut range = fallback_range.max(0.0);
        if duration_sec.is_finite()
            && duration_sec > 0.0
            && sampling_speed > 0
            && observing_frequency.is_finite()
            && observing_frequency > 0.0
        {
            let max_rate = initial_rate.abs() + initial_rate_range;
            let drift =
                max_rate as f64 * duration_sec as f64 * sampling_speed as f64 / observing_frequency;
            if drift.is_finite() && drift < f32::MAX as f64 {
                range = range.max(drift as f32 + 0.5);
            }
        }
        range
    }

    /// Symmetric trial grid including the center. Cap the grid before
    /// allocation even when a large delay drift spans many nominal steps.
    fn generate_search_points(center: f32, range: f32, step: f32) -> Vec<f32> {
        if !center.is_finite() {
            return Vec::new();
        }
        if !range.is_finite() || range <= 0.0 || !step.is_finite() || step <= 0.0 {
            return vec![center];
        }
        let range = f64::from(range);
        let half_steps = (range / f64::from(step)).round().clamp(1.0, 5.0) as i32;
        let mut points: Vec<_> = (-half_steps..=half_steps)
            .map(|i| (f64::from(center) + range * f64::from(i) / f64::from(half_steps)) as f32)
            .collect();
        points.dedup();
        points
    }

    #[cfg(test)]
    mod tests {
        use super::{
            generate_search_points, needs_frequency_domain_coarse_search, rate_search_bounds,
        };
        use crate::args::Args;

        #[test]
        fn output_options_do_not_select_a_different_coarse_search() {
            let mut args = Args {
                plot: true,
                spectrum: true,
                bandpass_table: true,
                dynamic_spectrum: true,
                raw_visibility: true,
                npz: true,
                contamination: Some(Vec::new()),
                rfi: vec!["histogram".to_string()],
                ..Args::default()
            };
            assert!(!needs_frequency_domain_coarse_search(&args));

            args.frequency = true;
            assert!(needs_frequency_domain_coarse_search(&args));
        }

        #[test]
        fn default_rate_bounds_are_time_nyquist_limits() {
            assert_eq!(rate_search_bounds(1.0, &[]), (-0.5, 0.5));
            assert_eq!(rate_search_bounds(0.25, &[]), (-2.0, 2.0));
        }

        #[test]
        fn explicit_rate_bounds_are_preserved_and_ordered() {
            assert_eq!(rate_search_bounds(1.0, &[0.2, -0.1]), (-0.1, 0.2));
        }

        #[test]
        fn invalid_search_steps_cannot_loop_forever() {
            assert_eq!(generate_search_points(1.5, 0.5, 0.0), vec![1.5]);
            assert_eq!(generate_search_points(1.5, 0.5, -0.1), vec![1.5]);
            assert!(generate_search_points(f32::NAN, 0.5, 0.1).is_empty());
        }
    }
}
