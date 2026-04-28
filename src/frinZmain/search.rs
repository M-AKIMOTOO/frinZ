pub use acel::run_acel_search_analysis;
pub use deep::{run_deep2_search, run_deep_search, DeepSearchParams, DeepSearchResult};

mod acel {
    use std::error::Error;
    use std::fs::{self, File};
    use std::io::{Cursor, Write};
    use std::path::Path;

    use chrono::{DateTime, Utc};
    use num_complex::Complex;

    use crate::args::Args;
    use crate::fitting;
    use crate::header::{parse_header, CorHeader};
    use crate::input_support::read_input_bytes;
    use crate::plot::plot_acel_search_result;
    use crate::processing::run_analysis_pipeline;
    use crate::read::read_visibility_data;
    use crate::rfi::parse_rfi_ranges;
    use crate::utils::unwrap_phase;

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

        for data_point in collected_data {
            let start_time_offset_sec = (data_point.obs_time - obs_time_start).num_seconds() as f32;

            if data_point.sector_count <= 0 {
                continue;
            }
            let current_length = data_point.sector_count;
            let fft_point_half = data_point.complex_vec.len() / current_length as usize;
            if fft_point_half == 0 {
                continue;
            }
            let effective_fft_point = (fft_point_half * 2) as i32;

            let (analysis_results, _, _, _) = run_analysis_pipeline(
                &data_point.complex_vec,
                header,
                args,
                Some("peak"),
                args.delay_correct,
                current_total_rate_correct,
                current_total_acel_correct,
                current_length,
                current_length,
                effective_integ_time,
                &data_point.obs_time,
                &obs_time_start,
                rfi_ranges,
                bandpass_data,
                false,
                effective_fft_point,
            )?;

            let phase_rad = analysis_results.delay_phase.to_radians() as f32;

            phases_collected.push(phase_rad);
            times_collected.push(start_time_offset_sec as f64);
            residual_rates_hz.push(analysis_results.residual_rate);
            residual_delays_samples.push(analysis_results.residual_delay);
        }

        unwrap_phase(&mut phases_collected, true);
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

    use crate::analysis::{analyze_results, AnalysisResults};
    use crate::args::Args;
    use crate::bandpass::apply_bandpass_correction;
    use crate::fft::{process_fft, process_fft_with_phase_correction, process_ifft};
    use crate::header::CorHeader;
    use crate::utils::{delay_rate_mask_bounds, in_delay_rate_mask, positive_or_epsilon, rate_cal};
    use rustfft::FftPlanner;

    type C32 = Complex<f32>;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum DeepSearchAlgorithm {
        FullGrid,
        AxisThenLocal,
    }

    #[allow(dead_code)]
    impl DeepSearchAlgorithm {
        fn mode_name(self) -> &'static str {
            match self {
                Self::FullGrid => "deep",
                Self::AxisThenLocal => "deep2",
            }
        }

        fn log_name(self) -> &'static str {
            match self {
                Self::FullGrid => "DEEP SEARCH",
                Self::AxisThenLocal => "DEEP2 SEARCH",
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
    }

    impl<'a> DeepSearchContext<'a> {
        fn fft_for_correction(&self, delay: f32, rate: f32) -> (Array2<C32>, usize) {
            if rate == 0.0 && delay == 0.0 && self.args.acel_correct == 0.0 {
                process_fft(
                    self.complex_vec,
                    self.physical_length,
                    self.effective_fft_point,
                    self.header.sampling_speed,
                    self.rfi_ranges,
                    self.args.rate_padding,
                )
            } else {
                process_fft_with_phase_correction(
                    self.complex_vec,
                    self.physical_length,
                    self.effective_fft_point,
                    self.header.sampling_speed,
                    self.rfi_ranges,
                    self.args.rate_padding,
                    rate,
                    delay,
                    self.args.acel_correct,
                    self.effective_integ_time,
                    self.start_time_offset_sec,
                )
            }
        }

        fn apply_bandpass(&self, freq_rate_array: &mut Array2<C32>) {
            if let Some(bp_data) = self.bandpass_data {
                apply_bandpass_correction(freq_rate_array, bp_data);
            }
        }

        fn coarse_estimates(
            &self,
            algorithm: DeepSearchAlgorithm,
        ) -> Result<(f32, f32), Box<dyn Error>> {
            // drange/rrange が指定されている場合は、その範囲で探索
            if !self.args.drange.is_empty() || !self.args.rrange.is_empty() {
                /*
                println!(
                    "[{}] Using specified delay/rate windows for coarse estimation",
                    algorithm.log_name()
                );
                */
                let search_args = self.args;
                let (mut freq_rate_array, padding_length) = self.fft_for_correction(0.0, 0.0);
                self.apply_bandpass(&mut freq_rate_array);
                let delay_rate_2d_data_comp =
                    process_ifft(&freq_rate_array, self.effective_fft_point, padding_length);
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
                let (mut freq_rate_array, padding_length) = self.fft_for_correction(0.0, 0.0);
                self.apply_bandpass(&mut freq_rate_array);
                let delay_rate_2d_data_comp =
                    process_ifft(&freq_rate_array, self.effective_fft_point, padding_length);
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
                ))
            }
        }

        fn evaluate_candidate_snr(&self, delay: f32, rate: f32) -> f32 {
            let (mut freq_rate_array, padding_length) = self.fft_for_correction(delay, rate);
            self.apply_bandpass(&mut freq_rate_array);
            let temp_args = create_corrected_args(self.args, delay, rate);
            evaluate_delay_snr_streaming(
                &freq_rate_array,
                self.effective_integ_time,
                padding_length,
                self.effective_fft_point,
                &temp_args,
            )
        }

        fn final_analysis(
            &self,
            final_delay: f32,
            final_rate: f32,
            algorithm: DeepSearchAlgorithm,
        ) -> Result<
            (
                AnalysisResults,
                Option<Array2<C32>>,
                Array2<C32>,
                Option<AnalysisResults>,
            ),
            Box<dyn Error>,
        > {
            let (mut final_freq_rate_array, padding_length) =
                self.fft_for_correction(final_delay, final_rate);
            let final_args = create_corrected_args(self.args, final_delay, final_rate);

            let pre_bandpass_analysis_results = if self.args.plot && self.bandpass_data.is_some() {
                let pre_bandpass_delay_rate_2d_data_comp = process_ifft(
                    &final_freq_rate_array,
                    self.effective_fft_point,
                    padding_length,
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
                    Some(algorithm.mode_name()),
                ))
            } else {
                None
            };

            self.apply_bandpass(&mut final_freq_rate_array);
            let final_delay_rate_2d_data_comp = process_ifft(
                &final_freq_rate_array,
                self.effective_fft_point,
                padding_length,
            );
            let mut analysis_results = analyze_results(
                &final_freq_rate_array,
                &final_delay_rate_2d_data_comp,
                self.header,
                self.current_length,
                self.effective_integ_time,
                self.current_obs_time,
                padding_length,
                &final_args,
                Some(algorithm.mode_name()),
            );

            analysis_results.residual_delay = final_delay;
            analysis_results.residual_rate = final_rate;
            analysis_results.length_f32 = self.physical_length as f32 * self.effective_integ_time;

            Ok((
                analysis_results,
                self.args.frequency.then_some(final_freq_rate_array),
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
        _obs_time: &DateTime<Utc>,
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
            rfi_ranges,
            bandpass_data,
            args,
            pp,
            cpu_count_arg,
            previous_solution,
            DeepSearchAlgorithm::FullGrid,
        )
    }

    pub fn run_deep2_search(
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
        rfi_ranges: &[(usize, usize)],
        bandpass_data: &Option<Vec<C32>>,
        args: &Args,
        pp: i32,
        cpu_count_arg: u32,
        previous_solution: Option<(f32, f32)>,
        algorithm: DeepSearchAlgorithm,
    ) -> Result<DeepSearchResult, Box<dyn Error>> {
        /*
        println!(
            "[{}] Starting {} hierarchical search algorithm",
            algorithm.log_name(),
            algorithm.mode_name()
        );
        */

        // フリンジ補正はファイル開始時刻からの経過時間で行う
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
        };

        // Step 1: 粗い遅延・レート推定
        let seed_solution = previous_solution.or_else(|| {
            (args.delay_correct != 0.0 || args.rate_correct != 0.0)
                .then_some((args.delay_correct, args.rate_correct))
        });
        let (coarse_delay, coarse_rate) = if let Some((prev_delay, prev_rate)) = seed_solution {
            /*
            println!(
                "[{}] Seeding from previous solution: delay={:.6}, rate={:.6}",
                algorithm.log_name(),
                prev_delay,
                prev_rate
            );
            */
            (prev_delay, prev_rate)
        } else {
            /*
            println!(
                "[{}] Running coarse grid search for initial estimate",
                algorithm.log_name()
            );
            */
            context.coarse_estimates(algorithm)?
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
        let effective_cpu_count = determine_effective_cpu_count(cpu_count_arg);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(effective_cpu_count)
            .build()?;

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
            let delay_range = search_params.delay_search_range / scale_factor;
            let rate_range =
                search_params.rate_search_range_factor / (2.0 * pp as f32) / scale_factor;
            let delay_step = search_params.delay_fine_step / scale_factor;
            let rate_step = search_params.rate_fine_step_factor / (10.0 * pp as f32) / scale_factor;

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

            let (best_delay, best_rate, best_snr) = match algorithm {
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
                DeepSearchAlgorithm::AxisThenLocal => parallel_axis_search(
                    &context,
                    current_delay,
                    current_rate,
                    delay_range,
                    rate_range,
                    delay_step,
                    rate_step,
                    &pool,
                )?,
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

        if algorithm == DeepSearchAlgorithm::AxisThenLocal {
            let final_scale = 10.0_f32.powi(search_params.max_iterations.saturating_sub(1) as i32);
            let final_delay_step = search_params.delay_fine_step / final_scale;
            let final_rate_step =
                search_params.rate_fine_step_factor / (10.0 * pp as f32) / final_scale;
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
        ) = context.final_analysis(final_delay, final_rate, algorithm)?;

        Ok(DeepSearchResult {
            analysis_results: final_analysis_results,
            freq_rate_array: final_freq_rate_array,
            delay_rate_2d_data: final_delay_rate_2d_data,
            pre_bandpass_analysis_results,
        })
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

    fn evaluate_delay_snr_streaming(
        freq_rate_array: &Array2<C32>,
        effective_integ_time: f32,
        padding_length: usize,
        effective_fft_point: i32,
        args: &Args,
    ) -> f32 {
        let fft_point_usize = effective_fft_point as usize;
        if fft_point_usize == 0 || padding_length == 0 {
            return 0.0;
        }

        let fft_point_f32 = fft_point_usize as f32;
        let fft_point_half = fft_point_usize / 2;
        let delay_range = Array::linspace(
            -(fft_point_f32 / 2.0) + 1.0,
            fft_point_f32 / 2.0,
            fft_point_usize,
        );
        let rate_range = rate_cal(padding_length as f32, effective_integ_time);
        let padding_length_half = padding_length / 2;
        let delay_rate_mask = if args.frequency {
            None
        } else {
            delay_rate_mask_bounds(&args.mask)
        };

        let delay_window = if args.drange.len() == 2 {
            Some((
                args.drange[0].min(args.drange[1]),
                args.drange[0].max(args.drange[1]),
            ))
        } else {
            None
        };
        let rate_window = if args.rrange.len() == 2 {
            Some((
                args.rrange[0].min(args.rrange[1]),
                args.rrange[0].max(args.rrange[1]),
            ))
        } else {
            None
        };

        let use_window_search = delay_window.is_some() || rate_window.is_some();
        let use_mask_search = !use_window_search && delay_rate_mask.is_some();
        let center_rate_idx = padding_length_half.min(padding_length.saturating_sub(1));
        let center_delay_idx = fft_point_half
            .saturating_sub(1)
            .min(fft_point_usize.saturating_sub(1));

        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(fft_point_usize);
        let mut ifft_exe = vec![C32::new(0.0, 0.0); fft_point_usize];
        let mut norm_sum = 0.0f32;
        let mut peak_norm = 0.0f32;

        for r_idx in 0..padding_length {
            ifft_exe.fill(C32::new(0.0, 0.0));
            for (dst, src) in ifft_exe
                .iter_mut()
                .zip(freq_rate_array.column(r_idx).iter())
            {
                *dst = *src;
            }

            ifft.process(&mut ifft_exe);

            let rate_val = rate_range[r_idx];
            let rate_in_window = rate_window
                .map(|(low, high)| rate_val >= low && rate_val <= high)
                .unwrap_or(true);
            let (first_half, second_half) = ifft_exe.split_at(fft_point_usize / 2);
            for (d_idx, src) in second_half
                .iter()
                .chain(first_half.iter())
                .rev()
                .enumerate()
            {
                let value = *src / fft_point_usize as f32;
                let norm = value.norm();
                norm_sum += norm;

                let delay_val = delay_range[d_idx];
                let masked = in_delay_rate_mask(delay_val, rate_val, delay_rate_mask);
                let is_candidate = if use_window_search {
                    let delay_in_window = delay_window
                        .map(|(low, high)| delay_val >= low && delay_val <= high)
                        .unwrap_or(true);
                    delay_in_window && rate_in_window && !masked
                } else if use_mask_search {
                    !masked
                } else {
                    r_idx == center_rate_idx && d_idx == center_delay_idx
                };

                if is_candidate && norm > peak_norm {
                    peak_norm = norm;
                }
            }
        }

        let total_cells = padding_length.saturating_mul(fft_point_usize).max(1);
        let delay_noise = positive_or_epsilon(norm_sum / total_cells as f32);
        peak_norm / delay_noise
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

        // 最大10点に制限（計算量制御）
        if points.len() > 10 {
            // Ensure step_by is at least 1 to avoid panic
            let step_by = (points.len() / 10).max(1);
            points = points.into_iter().step_by(step_by).collect();
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
