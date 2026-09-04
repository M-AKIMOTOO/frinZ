pub use acel::run_acel_search_analysis;
pub use deep::{
    run_coherent_search, run_deep2_search, run_deep_search, run_peak_search, DeepSearchParams,
    DeepSearchResult,
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
    use crate::input_support::read_input_bytes;
    use crate::plot::plot_acel_search_result;
    use crate::read::read_visibility_data;
    use crate::rfi::parse_rfi_ranges;
    use crate::utils::unwrap_phase_with_rate;

    type C32 = Complex<f32>;

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
    ) -> Result<(Vec<f64>, Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn Error>> {
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
        peak_args.rate_padding = 8;
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

        let buffer = read_input_bytes(input_path)?;
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
                        write_fit_data(&linear_path, Some(&vec![intercept, slope]))?;
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
                let residuals = phase_residual_series.as_ref().map(|v| v.as_slice());
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
                let residuals = rate_residual_series.as_ref().map(|v| v.as_slice());
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
                let residuals = delay_residual_samples_series.as_ref().map(|v| v.as_slice());
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
    use chrono::{DateTime, Utc};
    use ndarray::prelude::*;
    use num_complex::Complex;
    use rayon::prelude::*;
    use std::error::Error;
    use std::f64::consts::PI;

    use crate::analysis::{analyze_results, AnalysisResults};
    use crate::args::Args;
    use crate::bandpass::apply_bandpass_correction;
    use crate::fft::{
        cached_fft_plan, process_fft, process_fft_with_phase_correction_at_frequency, process_ifft,
    };
    use crate::header::CorHeader;
    use crate::utils::{delay_rate_mask_bounds, in_delay_rate_mask, positive_or_epsilon, rate_cal};

    type C32 = Complex<f32>;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum DeepSearchAlgorithm {
        FullGrid,
        AxisThenLocal,
        PeakPolish,
        Coherent,
    }

    #[allow(dead_code)]
    impl DeepSearchAlgorithm {
        fn mode_name(self) -> &'static str {
            match self {
                Self::FullGrid => "deep",
                Self::AxisThenLocal => "deep2",
                Self::PeakPolish => "peak",
                Self::Coherent => "peak",
            }
        }

        fn log_name(self) -> &'static str {
            match self {
                Self::FullGrid => "DEEP SEARCH",
                Self::AxisThenLocal => "DEEP2 SEARCH",
                Self::PeakPolish => "PEAK SEARCH",
                Self::Coherent => "COHERENT SEARCH",
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

    struct DeepSearchContext<'a> {
        complex_vec: &'a [C32],
        header: &'a CorHeader,
        current_length: i32,
        physical_length: i32,
        effective_integ_time: f32,
        current_obs_time: &'a DateTime<Utc>,
        rfi_ranges: &'a [(usize, usize)],
        bandpass_data: &'a Option<Vec<C32>>,
        args: &'a Args,
        start_time_offset_sec: f32,
        effective_fft_point: i32,
        rfi_mask: Vec<bool>,
        bandpass_gains: Option<Vec<C32>>,
        phase_times: Vec<f64>,
        static_phase_cycles: Vec<f64>,
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

    fn build_bandpass_gains(channel_count: usize, bandpass_data: &[C32]) -> Option<Vec<C32>> {
        if bandpass_data.is_empty() {
            return None;
        }
        const EPSILON: f32 = 1.0e-9;
        let mean = bandpass_data.iter().copied().sum::<C32>() / bandpass_data.len() as f32;
        let mut gains = vec![C32::new(1.0, 0.0); channel_count];
        for (channel, &value) in bandpass_data.iter().take(channel_count).enumerate() {
            if value.norm() > EPSILON {
                gains[channel] = mean / value;
            }
        }
        Some(gains)
    }

    impl<'a> DeepSearchContext<'a> {
        fn fft_for_correction_with_padding(
            &self,
            delay: f32,
            rate: f32,
            rate_padding: u32,
        ) -> (Array2<C32>, usize) {
            let rate_padding = rate_padding.max(1);
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
                    rate_padding,
                )
            } else {
                process_fft_with_phase_correction_at_frequency(
                    self.complex_vec,
                    self.physical_length,
                    self.effective_fft_point,
                    self.header.sampling_speed,
                    self.rfi_ranges,
                    rate_padding,
                    rate,
                    delay,
                    self.args.acel_correct,
                    self.args.jerk_correct,
                    self.args.snap_correct,
                    self.effective_integ_time,
                    self.start_time_offset_sec,
                    self.header.observing_frequency,
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

        fn fft_for_correction(&self, delay: f32, rate: f32) -> (Array2<C32>, usize) {
            self.fft_for_correction_with_padding(delay, rate, self.args.rate_padding)
        }

        fn needs_coarse_surface(&self) -> bool {
            self.args.frequency
                || self.args.spectrum
                || self.args.bandpass_table
                || self.args.dynamic_spectrum
                || self.args.raw_visibility
                || self.args.npz
                || self.args.contamination.is_some()
                || crate::rfi::has_histogram_mode(&self.args.rfi)
        }

        fn apply_bandpass(&self, freq_rate_array: &mut Array2<C32>) {
            if let Some(bp_data) = self.bandpass_data {
                apply_bandpass_correction(freq_rate_array, bp_data);
            }
        }

        fn coarse_estimates_streaming(
            &self,
            freq_rate_array: &Array2<C32>,
            padding_length: usize,
            search_args: &Args,
        ) -> (f32, f32, AnalysisResults) {
            let fft_point = self.effective_fft_point.max(1) as usize;
            let freq_bins = freq_rate_array.dim().0.min(fft_point);
            let scale = fft_point as f32;
            let rate_range = rate_cal(padding_length as f32, self.effective_integ_time);
            let half = fft_point / 2;
            let delay_at = |index: usize| index as f32 - half as f32 + 1.0;
            let delay_mask = if search_args.frequency {
                None
            } else {
                delay_rate_mask_bounds(&search_args.mask)
            };
            let in_window = |value: f32, bounds: &[f32]| {
                bounds.len() != 2 || (value >= bounds[0] && value <= bounds[1])
            };
            let mut ifft_exe = vec![C32::new(0.0, 0.0); fft_point];
            let ifft = cached_fft_plan(fft_point, true);

            let mut scan =
                |mean: Option<(f64, f64)>| {
                    let mut sum_re = 0.0f64;
                    let mut sum_im = 0.0f64;
                    let mut noise_sum = 0.0f64;
                    let mut count = 0usize;
                    let mut max_power = -1.0f32;
                    let mut max_delay = 0.0f32;
                    let mut max_rate = 0.0f32;
                    let mut max_value = C32::new(0.0, 0.0);

                    for (rate_index, &rate_value) in rate_range.iter().enumerate() {
                        for (dst, src) in ifft_exe[..freq_bins]
                            .iter_mut()
                            .zip(freq_rate_array.column(rate_index).iter().take(freq_bins))
                        {
                            *dst = *src;
                        }
                        ifft_exe[freq_bins..].fill(C32::new(0.0, 0.0));
                        ifft.process(&mut ifft_exe);

                        for delay_index in 0..fft_point {
                            let source_index = if delay_index < half {
                                half.saturating_sub(1 + delay_index)
                            } else {
                                fft_point - 1 - (delay_index - half)
                            };
                            let value = ifft_exe[source_index] / scale;
                            let delay_value = delay_at(delay_index);
                            if self.args.rfi_npz_mask.as_deref().is_some_and(|mask| {
                                mask.contains_delay_rate(delay_value, rate_value)
                            }) {
                                continue;
                            }
                            sum_re += value.re as f64;
                            sum_im += value.im as f64;
                            count += 1;

                            if let Some((mean_re, mean_im)) = mean {
                                noise_sum += ((value.re as f64 - mean_re).powi(2)
                                    + (value.im as f64 - mean_im).powi(2))
                                .sqrt();
                            } else if in_window(delay_value, &search_args.drange)
                                && in_window(rate_value, &search_args.rrange)
                                && !in_delay_rate_mask(delay_value, rate_value, delay_mask)
                            {
                                let power = value.norm_sqr();
                                if power > max_power {
                                    max_power = power;
                                    max_delay = delay_value;
                                    max_rate = rate_value;
                                    max_value = value;
                                }
                            }
                        }
                    }

                    (
                        sum_re, sum_im, noise_sum, count, max_delay, max_rate, max_value,
                    )
                };

            let (sum_re, sum_im, _, count, coarse_delay, coarse_rate, coarse_value) = scan(None);
            let mean = if count > 0 {
                (sum_re / count as f64, sum_im / count as f64)
            } else {
                (0.0, 0.0)
            };
            let (_, _, noise_sum, noise_count, _, _, _) = scan(Some(mean));
            let delay_noise = positive_or_epsilon(if noise_count > 0 {
                (noise_sum / noise_count as f64) as f32
            } else {
                0.0
            });

            let fake_freq_rate = Array2::<C32>::zeros((1, 1));
            let fake_delay_rate = Array2::<C32>::zeros((1, 2));
            let mut analysis = analyze_results(
                &fake_freq_rate,
                &fake_delay_rate,
                self.header,
                self.current_length,
                self.effective_integ_time,
                self.current_obs_time,
                1,
                search_args,
                search_args.primary_search_mode(),
            );
            analysis.delay_peak_complex = coarse_value;
            analysis.delay_max_amp = coarse_value.norm();
            analysis.delay_phase = coarse_value.arg().to_degrees();
            analysis.delay_noise = delay_noise;
            analysis.delay_snr = analysis.delay_max_amp / delay_noise;
            analysis.residual_delay = coarse_delay;
            analysis.residual_rate = coarse_rate;
            analysis.length_f32 = self.physical_length as f32 * self.effective_integ_time;
            analysis.delay_range = Array1::from_iter((0..fft_point).map(delay_at));
            analysis.rate_range = rate_range;
            (coarse_delay, coarse_rate, analysis)
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

        fn coarse_estimates(
            &self,
            algorithm: DeepSearchAlgorithm,
        ) -> Result<(f32, f32, AnalysisResults), Box<dyn Error>> {
            // drange/rrange が指定されている場合は、その範囲で探索
            if !self.args.drange.is_empty() || !self.args.rrange.is_empty() {
                /*
                println!(
                    "[{}] Using specified delay/rate windows for coarse estimation",
                    algorithm.log_name()
                );
                */
                let search_args = self.args;
                let (mut freq_rate_array, padding_length) =
                    self.fft_for_correction_with_padding(0.0, 0.0, self.args.rate_padding.max(1));
                self.apply_bandpass(&mut freq_rate_array);
                if !self.needs_coarse_surface() && self.current_length > 2 {
                    return Ok(self.coarse_estimates_streaming(
                        &freq_rate_array,
                        padding_length,
                        &search_args,
                    ));
                }
                let mut delay_rate_2d_data_comp =
                    process_ifft(&freq_rate_array, self.effective_fft_point, padding_length);
                self.apply_delay_rate_mask(&mut delay_rate_2d_data_comp);
                let analysis_results = analyze_results(
                    &freq_rate_array,
                    &delay_rate_2d_data_comp,
                    self.header,
                    self.current_length,
                    self.effective_integ_time,
                    self.current_obs_time,
                    padding_length,
                    search_args,
                    search_args.primary_search_mode(),
                );
                Ok((
                    analysis_results.residual_delay,
                    analysis_results.residual_rate,
                    analysis_results,
                ))
            } else {
                /*
                println!(
                    "[{}] No windows specified, running coarse search (no fitting) for initial estimates",
                    algorithm.log_name()
                );
                */
                let mut search_args = self.args.clone();
                search_args.search = vec![algorithm.mode_name().to_string()];
                let (mut freq_rate_array, padding_length) =
                    self.fft_for_correction_with_padding(0.0, 0.0, self.args.rate_padding.max(1));
                self.apply_bandpass(&mut freq_rate_array);
                if !self.needs_coarse_surface() && self.current_length > 2 {
                    return Ok(self.coarse_estimates_streaming(
                        &freq_rate_array,
                        padding_length,
                        &search_args,
                    ));
                }
                let mut delay_rate_2d_data_comp =
                    process_ifft(&freq_rate_array, self.effective_fft_point, padding_length);
                self.apply_delay_rate_mask(&mut delay_rate_2d_data_comp);
                let analysis_results = analyze_results(
                    &freq_rate_array,
                    &delay_rate_2d_data_comp,
                    self.header,
                    self.current_length,
                    self.effective_integ_time,
                    self.current_obs_time,
                    padding_length,
                    &search_args,
                    search_args.primary_search_mode(),
                );
                Ok((
                    analysis_results.residual_delay,
                    analysis_results.residual_rate,
                    analysis_results,
                ))
            }
        }

        fn evaluate_coherent_sum(&self, delay: f32, rate: f32) -> (f64, f64) {
            let fft_point = self.effective_fft_point as usize;
            let channel_count = fft_point / 2;
            let sampling_speed = self.header.sampling_speed;
            let row_count = self.complex_vec.len() / channel_count.max(1);
            if fft_point == 0 || channel_count <= 1 || sampling_speed <= 0 || row_count == 0 {
                return (0.0, 0.0);
            }

            let sampling_speed_f64 = sampling_speed as f64;
            let delay_seconds = delay as f64 / sampling_speed_f64;
            let frequency_step_hz = sampling_speed_f64 / fft_point as f64;
            let reference_frequency = self.header.observing_frequency;
            let use_wideband_rate =
                reference_frequency.is_finite() && reference_frequency.abs() > f64::EPSILON;
            let rate_f64 = rate as f64;
            let mut sum_re = 0.0f64;
            let mut sum_im = 0.0f64;

            for (row_index, row) in self
                .complex_vec
                .chunks_exact(channel_count)
                .take(row_count)
                .enumerate()
            {
                let cycles =
                    rate_f64 * self.phase_times[row_index] + self.static_phase_cycles[row_index];
                let row_angle = -2.0 * PI * cycles;
                let (row_re, row_im) = (row_angle.cos(), row_angle.sin());
                let time_varying_delay = if use_wideband_rate {
                    cycles / reference_frequency
                } else {
                    0.0
                };
                let step_angle =
                    -2.0 * PI * (delay_seconds + time_varying_delay) * frequency_step_hz;
                let (step_re, step_im) = (step_angle.cos(), step_angle.sin());
                let mut channel_re = step_re;
                let mut channel_im = step_im;

                for channel in 1..channel_count {
                    if self.rfi_mask[channel] {
                        let next_re = channel_re * step_re - channel_im * step_im;
                        let next_im = channel_re * step_im + channel_im * step_re;
                        channel_re = next_re;
                        channel_im = next_im;
                        continue;
                    }

                    let mut sample = row[channel];
                    if let Some(gains) = &self.bandpass_gains {
                        sample *= gains[channel];
                    }
                    let sample_re = sample.re as f64 * row_re - sample.im as f64 * row_im;
                    let sample_im = sample.re as f64 * row_im + sample.im as f64 * row_re;
                    sum_re += sample_re * channel_re - sample_im * channel_im;
                    sum_im += sample_re * channel_im + sample_im * channel_re;

                    let next_re = channel_re * step_re - channel_im * step_im;
                    let next_im = channel_re * step_im + channel_im * step_re;
                    channel_re = next_re;
                    channel_im = next_im;
                }
            }
            (sum_re, sum_im)
        }

        fn coherent_sum_scale(&self) -> f64 {
            let bandwidth_mhz = self.header.sampling_speed as f64 / 2.0e6;
            let power_scale = if bandwidth_mhz > 0.0 {
                512.0 / bandwidth_mhz
            } else {
                1.0
            };
            power_scale / self.physical_length.max(1) as f64
        }

        fn evaluate_coherent_amplitude(&self, delay: f32, rate: f32) -> f32 {
            let (sum_re, sum_im) = self.evaluate_coherent_sum(delay, rate);
            (sum_re.hypot(sum_im) * self.coherent_sum_scale()) as f32
        }

        fn evaluate_candidate_snr(&self, delay: f32, rate: f32) -> f32 {
            // All search modes optimize the same corrected coherent amplitude.
            if !self.args.frequency
                && in_delay_rate_mask(delay, rate, delay_rate_mask_bounds(&self.args.mask))
            {
                return 0.0;
            }
            if !self.args.frequency
                && self
                    .args
                    .rfi_npz_mask
                    .as_deref()
                    .is_some_and(|mask| mask.contains_delay_rate(delay, rate))
            {
                return 0.0;
            }
            // Evaluate it directly from the visibility rows so candidate scans
            // do not allocate a padded frequency-rate plane per worker.
            self.evaluate_coherent_amplitude(delay, rate)
        }

        fn final_analysis(
            &self,
            final_delay: f32,
            final_rate: f32,
            _algorithm: DeepSearchAlgorithm,
            _coarse_analysis: Option<&AnalysisResults>,
        ) -> Result<
            (
                AnalysisResults,
                Option<Array2<C32>>,
                Array2<C32>,
                Option<AnalysisResults>,
            ),
            Box<dyn Error>,
        > {
            // Always evaluate the final candidate on the fully corrected FFT plane.
            // The reported noise/SNR must describe the same corrected data whether or
            // not diagnostic plots were requested. The former no-plot shortcut reused
            // the coarse streaming noise estimate and made `--plot` change SNR.
            let (mut final_freq_rate_array, padding_length) =
                self.fft_for_correction(final_delay, final_rate);
            let mut final_args = create_corrected_args(self.args, final_delay, final_rate);
            // The final candidate has already applied the full correction. Measure
            // amplitude and phase at residual delay=rate=0, independent of the
            // absolute search windows or masks used to select the candidate.
            final_args.drange.clear();
            final_args.rrange.clear();
            final_args.mask.clear();

            let pre_bandpass_analysis_results = if self.args.plot && self.bandpass_data.is_some() {
                let mut pre_bandpass_delay_rate_2d_data_comp = process_ifft(
                    &final_freq_rate_array,
                    self.effective_fft_point,
                    padding_length,
                );
                if let Some(mask) = self.args.rfi_npz_mask.as_deref() {
                    let (rate_count, delay_count) = pre_bandpass_delay_rate_2d_data_comp.dim();
                    if let Some(values) = pre_bandpass_delay_rate_2d_data_comp.as_slice_mut() {
                        mask.apply_delay_rate(
                            values,
                            rate_count,
                            delay_count,
                            self.effective_integ_time,
                            self.effective_fft_point,
                        );
                    }
                }
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
            let mut final_delay_rate_2d_data_comp = process_ifft(
                &final_freq_rate_array,
                self.effective_fft_point,
                padding_length,
            );
            if let Some(mask) = self.args.rfi_npz_mask.as_deref() {
                let (rate_count, delay_count) = final_delay_rate_2d_data_comp.dim();
                if let Some(values) = final_delay_rate_2d_data_comp.as_slice_mut() {
                    mask.apply_delay_rate(
                        values,
                        rate_count,
                        delay_count,
                        self.effective_integ_time,
                        self.effective_fft_point,
                    );
                }
            }
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

            analysis_results.residual_delay = final_delay;
            analysis_results.residual_rate = if self.physical_length <= 1 {
                0.0
            } else {
                final_rate
            };
            analysis_results.length_f32 = self.physical_length as f32 * self.effective_integ_time;

            Ok((
                analysis_results,
                (self.args.frequency
                    || self.args.contamination.is_some()
                    || crate::rfi::has_histogram_mode(&self.args.rfi))
                .then_some(final_freq_rate_array),
                final_delay_rate_2d_data_comp,
                pre_bandpass_analysis_results,
            ))
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
        )
    }

    pub fn run_coherent_search(
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
            DeepSearchAlgorithm::Coherent,
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
    ) -> Result<DeepSearchResult, Box<dyn Error>> {
        /*
        println!(
            "[{}] Starting {} hierarchical search algorithm",
            algorithm.log_name(),
            algorithm.mode_name()
        );
        */

        // Use the first sample of each analyzed segment as the phase reference
        // for residual rate correction.
        //
        // process_fft_with_phase_correction_at_frequency() evaluates the time for row_idx as
        //
        //   t = row_idx * effective_integ_time + start_time_offset_sec.
        //
        // Therefore start_time_offset_sec=0 makes row 0 the phase epoch. This
        // is also the MJD/UVW epoch written to the scalar contamination NPZ.
        let start_time_offset_sec = 0.0;

        if current_length <= 0 {
            return Err("有効なセクター長が 0 以下です".into());
        }
        let rows = current_length as usize;
        if rows == 0 || complex_vec.is_empty() {
            return Err("有効なデータが存在しません".into());
        }
        if complex_vec.len() % rows != 0 {
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
            .and_then(|data| build_bandpass_gains(fft_point_half, data));
        let phase_times: Vec<f64> = (0..rows)
            .map(|row_idx| {
                row_idx as f64 * effective_integ_time as f64 + start_time_offset_sec as f64
            })
            .collect();
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
            bandpass_data,
            args,
            start_time_offset_sec,
            effective_fft_point,
            rfi_mask,
            bandpass_gains,
            phase_times,
            static_phase_cycles,
        };

        let is_autocorrelation = is_autocorrelation_header(header);

        // Step 1: Reacquire a coarse solution for every segment. A previous
        // segment is not a safe sole seed when acceleration or atmospheric
        // fluctuations move the fringe by more than the fine-search window.
        //
        // For auto-correlation the physical delay and fringe rate are exactly
        // zero. Optimizing the delay/rate plane can otherwise fit tiny spectral
        // amplitude asymmetries or numerical interpolation noise and report a
        // non-physical offset. Keep the search path active, but make its
        // candidate solution the exact auto-correlation solution.
        let (coarse_delay, coarse_rate, coarse_analysis) = if is_autocorrelation {
            (0.0, 0.0, None)
        } else {
            let (delay, rate, analysis) = context.coarse_estimates(algorithm)?;
            (delay, rate, Some(analysis))
        };

        /*
        println!(
            "[{}] Coarse estimates - Delay: {:.6} samples, Rate: {:.6} Hz",
            algorithm.log_name(),
            coarse_delay,
            coarse_rate
        );
        */

        // Step 2: 階層的探索
        let mut search_params = DeepSearchParams::default();
        search_params.max_iterations = (args.iter.max(1)) as usize;
        let mut current_delay = coarse_delay;
        let mut current_rate = coarse_rate;
        let rate_denominator =
            physical_length.max(1) as f32 * effective_integ_time.abs().max(1.0e-9);
        // The wideband rate correction produces a delay drift across the
        // sampled band. Size the initial delay basin from that physical drift.
        let initial_rate_range =
            10.0 * search_params.rate_search_range_factor / (2.0 * rate_denominator);
        let search_duration_sec =
            physical_length.max(1) as f32 * effective_integ_time.abs().max(1.0e-9);
        let dynamic_delay_range = delay_search_range_for_rate(
            coarse_rate,
            search_duration_sec,
            header.sampling_speed,
            header.observing_frequency,
            initial_rate_range,
            search_params.delay_search_range,
        );
        let effective_cpu_count = determine_effective_cpu_count(cpu_count_arg);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(effective_cpu_count)
            .build()?;

        if !is_autocorrelation {
            for iteration in 0..search_params.max_iterations {
                /*
                println!(
                    "[{}] Iteration {} starting",
                    algorithm.log_name(),
                    iteration + 1
                );
                */

                // 現在の階層での探索範囲とステップサイズを計算
                let scale_factor = 10.0_f32.powi(iteration as i32);
                let delay_range = dynamic_delay_range / scale_factor;
                let rate_range = initial_rate_range / scale_factor;
                let delay_step = search_params.delay_fine_step / scale_factor;
                let rate_step =
                    search_params.rate_fine_step_factor / (10.0 * rate_denominator) / scale_factor;

                /*
                println!(
                    "[{}]   Delay range: +/- {:.6} samples, step: {:.6}",
                    algorithm.log_name(),
                    delay_range,
                    delay_step
                );
                println!(
                    "[{}]   Rate range: +/- {:.6} Hz, step: {:.6}",
                    algorithm.log_name(),
                    rate_range,
                    rate_step
                );
                */

                let (best_delay, best_rate, best_snr) = if iteration == 0 {
                    // The first pass must evaluate delay and rate together;
                    // axis-only refinement can miss a coupled maximum.
                    parallel_grid_search(
                        &context,
                        current_delay,
                        current_rate,
                        delay_range,
                        rate_range,
                        delay_step,
                        rate_step,
                        &pool,
                    )?
                } else {
                    match algorithm {
                        DeepSearchAlgorithm::FullGrid => parallel_grid_search(
                            &context,
                            current_delay,
                            current_rate,
                            delay_range,
                            rate_range,
                            delay_step,
                            rate_step,
                            &pool,
                        )?,
                        DeepSearchAlgorithm::AxisThenLocal
                        | DeepSearchAlgorithm::PeakPolish
                        | DeepSearchAlgorithm::Coherent => parallel_axis_search(
                            &context,
                            current_delay,
                            current_rate,
                            delay_range,
                            rate_range,
                            delay_step,
                            rate_step,
                            &pool,
                        )?,
                    }
                };

                // 結果を更新
                current_delay = best_delay;
                current_rate = best_rate;

                let _ = best_snr;
                /*
                println!(
                    "[{}]   Best result: delay={:.6} samples, rate={:.6} Hz, SNR={:.3}",
                    algorithm.log_name(),
                    current_delay,
                    current_rate,
                    best_snr
                );
                */
            }
        }

        if !is_autocorrelation
            && matches!(
                algorithm,
                DeepSearchAlgorithm::AxisThenLocal
                    | DeepSearchAlgorithm::PeakPolish
                    | DeepSearchAlgorithm::Coherent
            )
        {
            let final_scale = 10.0_f32.powi(search_params.max_iterations.saturating_sub(1) as i32);
            let final_delay_step = search_params.delay_fine_step / final_scale;
            let final_rate_step =
                search_params.rate_fine_step_factor / (10.0 * rate_denominator) / final_scale;
            let (best_delay, best_rate, best_snr) = parallel_grid_search(
                &context,
                current_delay,
                current_rate,
                final_delay_step,
                final_rate_step,
                final_delay_step,
                final_rate_step,
                &pool,
            )?;
            current_delay = best_delay;
            current_rate = best_rate;
            let _ = best_snr;
            /*
            println!(
                "[{}]   Final 3x3 local check: delay={:.6} samples, rate={:.6} Hz, SNR={:.3}",
                algorithm.log_name(),
                current_delay,
                current_rate,
                best_snr
            );
            */
        }

        let final_delay = current_delay;
        let final_rate = current_rate;

        // Step 3: 最終的な解析を実行
        /*
        println!(
            "[{}] Final result - Delay: {:.6} samples, Rate: {:.6} Hz",
            algorithm.log_name(),
            final_delay,
            final_rate
        );
        */

        let (
            final_analysis_results,
            final_freq_rate_array,
            final_delay_rate_2d_data,
            pre_bandpass_analysis_results,
        ) = context.final_analysis(final_delay, final_rate, algorithm, coarse_analysis.as_ref())?;

        Ok(DeepSearchResult {
            analysis_results: final_analysis_results,
            freq_rate_array: final_freq_rate_array,
            delay_rate_2d_data: final_delay_rate_2d_data,
            pre_bandpass_analysis_results,
        })
    }

    fn is_autocorrelation_header(header: &CorHeader) -> bool {
        let name1 = header.station1_name.trim();
        let name2 = header.station2_name.trim();
        if !name1.is_empty() && name1 == name2 {
            return true;
        }

        let code1 = header.station1_code.trim();
        let code2 = header.station2_code.trim();
        if !code1.is_empty() && code1 == code2 {
            return true;
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

    /// 並列グリッド探索
    fn parallel_grid_search(
        context: &DeepSearchContext<'_>,
        center_delay: f32,
        center_rate: f32,
        delay_range: f32,
        rate_range: f32,
        delay_step: f32,
        rate_step: f32,
        pool: &rayon::ThreadPool,
    ) -> Result<(f32, f32, f32), Box<dyn Error>> {
        // 探索グリッドを生成
        let delay_points = generate_search_points(center_delay, delay_range, delay_step);
        let rate_points = generate_search_points(center_rate, rate_range, rate_step);

        /*
        println!(
            "[{}]   Grid: {} delay x {} rate = {} combinations",
            context.args.primary_search_mode().unwrap_or("DEEP SEARCH"),
            delay_points.len(),
            rate_points.len(),
            delay_points.len() * rate_points.len()
        );
        */

        let delay_bounds = if context.args.drange.len() == 2 {
            Some((
                context.args.drange[0].min(context.args.drange[1]),
                context.args.drange[0].max(context.args.drange[1]),
            ))
        } else {
            None
        };
        let rate_bounds = if context.args.rrange.len() == 2 {
            Some((
                context.args.rrange[0].min(context.args.rrange[1]),
                context.args.rrange[0].max(context.args.rrange[1]),
            ))
        } else {
            None
        };

        // 並列探索実行
        let final_result = pool.install(|| {
            delay_points
                .par_iter()
                .filter_map(|&delay| {
                    if let Some((low, high)) = delay_bounds {
                        if delay < low || delay > high {
                            return None;
                        }
                    }
                    let best_for_delay = rate_points
                        .iter()
                        .filter_map(|&rate| {
                            if let Some((low, high)) = rate_bounds {
                                if rate < low || rate > high {
                                    return None;
                                }
                            }
                            Some((delay, rate, context.evaluate_candidate_snr(delay, rate)))
                        })
                        .max_by(|a, b| a.2.total_cmp(&b.2));
                    best_for_delay
                })
                .reduce_with(|best, candidate| {
                    if candidate.2 > best.2 {
                        candidate
                    } else {
                        best
                    }
                })
                .unwrap_or((center_delay, center_rate, 0.0f32))
        });

        Ok(final_result)
    }

    fn parallel_axis_search(
        context: &DeepSearchContext<'_>,
        center_delay: f32,
        center_rate: f32,
        delay_range: f32,
        rate_range: f32,
        delay_step: f32,
        rate_step: f32,
        pool: &rayon::ThreadPool,
    ) -> Result<(f32, f32, f32), Box<dyn Error>> {
        let rate_points = generate_search_points(center_rate, rate_range, rate_step);
        /*
        println!(
            "[DEEP2 SEARCH]   Axis grid: {} rate + delay axis",
            rate_points.len()
        );
        */

        let rate_bounds = if context.args.rrange.len() == 2 {
            Some((
                context.args.rrange[0].min(context.args.rrange[1]),
                context.args.rrange[0].max(context.args.rrange[1]),
            ))
        } else {
            None
        };

        let best_rate_result = pool.install(|| {
            rate_points
                .par_iter()
                .filter_map(|&rate| {
                    if let Some((low, high)) = rate_bounds {
                        if rate < low || rate > high {
                            return None;
                        }
                    }
                    Some((
                        center_delay,
                        rate,
                        context.evaluate_candidate_snr(center_delay, rate),
                    ))
                })
                .reduce_with(|best, candidate| {
                    if candidate.2 > best.2 {
                        candidate
                    } else {
                        best
                    }
                })
                .unwrap_or((center_delay, center_rate, 0.0f32))
        });

        let delay_points = generate_search_points(center_delay, delay_range, delay_step);
        /*
        println!(
            "[DEEP2 SEARCH]   Axis grid: {} delay at rate={:.6}",
            delay_points.len(),
            best_rate_result.1
        );
        */

        let delay_bounds = if context.args.drange.len() == 2 {
            Some((
                context.args.drange[0].min(context.args.drange[1]),
                context.args.drange[0].max(context.args.drange[1]),
            ))
        } else {
            None
        };

        let best_delay_result = pool.install(|| {
            delay_points
                .par_iter()
                .filter_map(|&delay| {
                    if let Some((low, high)) = delay_bounds {
                        if delay < low || delay > high {
                            return None;
                        }
                    }
                    Some((
                        delay,
                        best_rate_result.1,
                        context.evaluate_candidate_snr(delay, best_rate_result.1),
                    ))
                })
                .reduce_with(|best, candidate| {
                    if candidate.2 > best.2 {
                        candidate
                    } else {
                        best
                    }
                })
                .unwrap_or(best_rate_result)
        });

        Ok(best_delay_result)
    }

    /// 探索点を生成
    fn generate_search_points(center: f32, range: f32, step: f32) -> Vec<f32> {
        let mut points = Vec::new();

        // Use f64 for precision-critical calculations to avoid floating point errors
        // with very small steps, which can cause inconsistent point counts or infinite loops.
        let center64 = center as f64;
        let range64 = range as f64;
        let step64 = step as f64;

        // Guard against infinite loop if step is zero or too small to be represented.
        if step64 == 0.0 {
            if range64 >= 0.0 {
                points.push(center);
            }
            return points;
        }

        let start = center64 - range64;
        let end = center64 + range64;

        let mut current = start;
        // Add a small tolerance to the end condition to handle floating point inaccuracies
        while current <= end + step64 * 0.5 {
            points.push(current as f32);
            current += step64;
        }

        // Keep a symmetric 11-point grid so the center and both bounds are
        // represented even when the physical search range is wide.
        if points.len() > 11 {
            let start = points.first().copied().unwrap_or(center);
            let end = points.last().copied().unwrap_or(center);
            points = (0..11)
                .map(|index| start + (end - start) * index as f32 / 10.0)
                .collect();
        }

        points
    }

    /// 補正された引数を作成
    fn create_corrected_args(args: &Args, delay: f32, rate: f32) -> Args {
        let mut corrected_args = args.clone();
        corrected_args.delay_correct = delay;
        corrected_args.rate_correct = rate;
        // Keep window semantics in absolute coordinates by converting
        // to residual windows for already-corrected data.
        if corrected_args.drange.len() == 2 {
            corrected_args.drange[0] -= delay;
            corrected_args.drange[1] -= delay;
        }
        if corrected_args.rrange.len() == 2 {
            corrected_args.rrange[0] -= rate;
            corrected_args.rrange[1] -= rate;
        }
        corrected_args.search.clear(); // Prevent infinite loops
        corrected_args
    }
}
