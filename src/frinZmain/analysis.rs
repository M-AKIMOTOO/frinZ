use chrono::{DateTime, Utc};
use ndarray::prelude::*;
use num_complex::Complex;

use crate::args::Args;
use crate::fitting;
use crate::header::CorHeader;
use crate::utils::{
    mjd_cal, noise_level, radec2azalt, rate_cal, rate_delay_to_lm, safe_arg, uvw_cal,
};

type C32 = Complex<f32>;

fn sanitize_noise(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        f32::EPSILON
    }
}

fn window_bounds(window: &[f32]) -> Option<(f32, f32)> {
    if window.len() == 2 {
        Some((window[0].min(window[1]), window[0].max(window[1])))
    } else {
        None
    }
}

fn in_window(value: f32, bounds: Option<(f32, f32)>) -> bool {
    match bounds {
        Some((low, high)) => value >= low && value <= high,
        None => true,
    }
}

fn refine_peak_3x3_quadratic(
    surface: &Array2<f32>,
    center_rate_idx: usize,
    center_delay_idx: usize,
    rate_range: &[f32],
    delay_range: &Array1<f32>,
    rrange: &[f32],
    drange: &[f32],
) -> Option<(f32, f32)> {
    let (rows, cols) = surface.dim();
    if rows < 3 || cols < 3 {
        return None;
    }
    if center_rate_idx == 0
        || center_delay_idx == 0
        || center_rate_idx + 1 >= rows
        || center_delay_idx + 1 >= cols
    {
        return None;
    }
    if rate_range.len() != rows || delay_range.len() != cols {
        return None;
    }

    let rate_bounds = window_bounds(rrange);
    let delay_bounds = window_bounds(drange);
    for dy in -1isize..=1 {
        for dx in -1isize..=1 {
            let r_idx = (center_rate_idx as isize + dy) as usize;
            let d_idx = (center_delay_idx as isize + dx) as usize;
            if !in_window(rate_range[r_idx], rate_bounds)
                || !in_window(delay_range[d_idx], delay_bounds)
            {
                return None;
            }
        }
    }

    let f00 = surface[[center_rate_idx, center_delay_idx]] as f64;
    let f_xm = surface[[center_rate_idx, center_delay_idx - 1]] as f64;
    let f_xp = surface[[center_rate_idx, center_delay_idx + 1]] as f64;
    let f_ym = surface[[center_rate_idx - 1, center_delay_idx]] as f64;
    let f_yp = surface[[center_rate_idx + 1, center_delay_idx]] as f64;
    let f_pp = surface[[center_rate_idx + 1, center_delay_idx + 1]] as f64;
    let f_pm = surface[[center_rate_idx + 1, center_delay_idx - 1]] as f64;
    let f_mp = surface[[center_rate_idx - 1, center_delay_idx + 1]] as f64;
    let f_mm = surface[[center_rate_idx - 1, center_delay_idx - 1]] as f64;

    let fx = 0.5 * (f_xp - f_xm);
    let fy = 0.5 * (f_yp - f_ym);
    let fxx = f_xp - 2.0 * f00 + f_xm;
    let fyy = f_yp - 2.0 * f00 + f_ym;
    let fxy = 0.25 * (f_pp - f_pm - f_mp + f_mm);

    let det = fxx * fyy - fxy * fxy;
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    // Negative definite Hessian is required for a local maximum.
    if !(fxx < 0.0 && fyy < 0.0 && det > 0.0) {
        return None;
    }

    let delta_delay_bin = -((fyy * fx) - (fxy * fy)) / det;
    let delta_rate_bin = ((fxy * fx) - (fxx * fy)) / det;
    if !delta_delay_bin.is_finite() || !delta_rate_bin.is_finite() {
        return None;
    }
    if delta_delay_bin.abs() > 1.0 || delta_rate_bin.abs() > 1.0 {
        return None;
    }

    let delay_step =
        (delay_range[center_delay_idx + 1] as f64 - delay_range[center_delay_idx - 1] as f64) * 0.5;
    let rate_step =
        (rate_range[center_rate_idx + 1] as f64 - rate_range[center_rate_idx - 1] as f64) * 0.5;
    if delay_step.abs() < 1e-12 || rate_step.abs() < 1e-12 {
        return None;
    }

    let refined_delay = delay_range[center_delay_idx] as f64 + delta_delay_bin * delay_step;
    let refined_rate = rate_range[center_rate_idx] as f64 + delta_rate_bin * rate_step;
    let refined_delay = refined_delay as f32;
    let refined_rate = refined_rate as f32;

    if !in_window(refined_delay, delay_bounds) || !in_window(refined_rate, rate_bounds) {
        return None;
    }

    Some((refined_delay, refined_rate))
}

#[derive(Debug, Clone)]
pub struct AnalysisResults {
    // Common
    pub yyyydddhhmmss1: String,
    pub source_name: String,
    pub length_f32: f32,
    pub ant1_az: f32,
    pub ant1_el: f32,
    pub ant1_hgt: f32,
    pub ant2_az: f32,
    pub ant2_el: f32,
    pub ant2_hgt: f32,
    pub mjd: f64,
    // Delay
    pub delay_range: Array1<f32>,
    pub visibility: Array1<f32>,
    pub delay_rate: Array1<f32>,
    pub delay_max_amp: f32,
    pub delay_phase: f32,
    pub delay_snr: f32,
    pub delay_noise: f32,
    pub residual_delay: f32,
    pub corrected_delay: f32,
    pub delay_offset: f32,
    // Frequency
    pub freq_max_amp: f32,
    pub freq_phase: f32,
    pub freq_freq: f32,
    pub freq_snr: f32,
    pub freq_noise: f32,
    pub freq_rate: Array1<f32>,
    pub freq_rate_spectrum: Array1<C32>,
    pub freq_range: Array1<f32>,
    pub freq_max_freq: f32,
    pub residual_rate: f32,
    pub corrected_rate: f32,
    pub rate_offset: f32,
    // Add new fields here
    // pub residual_acel: f32,
    pub corrected_acel: f32,
    // Ranges
    pub rate_range: Vec<f32>,
    // Sky Coordinates
    #[allow(dead_code)]
    pub l_coord: f64,
    #[allow(dead_code)]
    pub m_coord: f64,
}

pub fn analyze_results(
    freq_rate_array: &Array2<C32>,
    delay_rate_array: &Array2<C32>,
    header: &CorHeader,
    length: i32,
    effective_integ_time: f32,
    obs_time: &DateTime<Utc>,
    padding_length: usize,
    args: &Args,
    search_mode: Option<&str>,
) -> AnalysisResults {
    let fft_point_half = freq_rate_array.dim().0;
    let fft_point_usize = fft_point_half * 2;
    let fft_point_f32 = fft_point_usize as f32;
    let length_f32 = length as f32;
    let padding_length_half = padding_length / 2;

    // --- Ranges ---
    let delay_range = Array::linspace(
        -(fft_point_f32 / 2.0) + 1.0,
        fft_point_f32 / 2.0,
        fft_point_usize,
    );

    let freq_step_mhz = (header.sampling_speed as f32 / header.fft_point as f32) / 1_000_000.0;
    let freq_range = Array1::from_iter((0..fft_point_half).map(|i| i as f32 * freq_step_mhz));
    let rate_range = rate_cal(padding_length as f32, effective_integ_time);

    // --- Delay Analysis ---
    let delay_rate_2d_data_array = delay_rate_array.clone().mapv(|x| x.norm());
    let delay_noise_raw = noise_level(delay_rate_array.view(), delay_rate_array.mean().unwrap());
    let delay_noise = sanitize_noise(delay_noise_raw);

    let (peak_rate_idx, peak_delay_idx) = if !args.drange.is_empty() || !args.rrange.is_empty() {
        // Case 3: Window options are specified (either/both), search within them.
        let (delay_win_low, delay_win_high) = if !args.drange.is_empty() {
            (args.drange[0], args.drange[1])
        } else {
            (delay_range[0], *delay_range.last().unwrap_or(&delay_range[0]))
        };
        let (rate_win_low, rate_win_high) = if !args.rrange.is_empty() {
            (args.rrange[0], args.rrange[1])
        } else {
            (rate_range[0], *rate_range.last().unwrap_or(&rate_range[0]))
        };

        let mut max_val_in_window = 0.0f32;
        let mut temp_peak_rate_idx = padding_length_half;
        let mut temp_peak_delay_idx = fft_point_half;

        for r_idx in 0..rate_range.len() {
            if rate_range[r_idx] >= rate_win_low && rate_range[r_idx] <= rate_win_high {
                for d_idx in 0..delay_range.len() {
                    if delay_range[d_idx] >= delay_win_low && delay_range[d_idx] <= delay_win_high {
                        let current_val = delay_rate_2d_data_array[[r_idx, d_idx]];
                        if current_val > max_val_in_window {
                            max_val_in_window = current_val;
                            temp_peak_rate_idx = r_idx;
                            temp_peak_delay_idx = d_idx;
                        }
                    }
                }
            }
        }
        (temp_peak_rate_idx, temp_peak_delay_idx)
    } else if search_mode == Some("peak") || search_mode == Some("deep") {
        // Case 2: --search or --search_deep is specified, no window. Find the global maximum.
        let (mut max_val, mut max_r_idx, mut max_d_idx) = (0.0f32, 0, 0);
        for r_idx in 0..delay_rate_2d_data_array.shape()[0] {
            for d_idx in 0..delay_rate_2d_data_array.shape()[1] {
                let current_val = delay_rate_2d_data_array[[r_idx, d_idx]];
                if current_val > max_val {
                    max_val = current_val;
                    max_r_idx = r_idx;
                    max_d_idx = d_idx;
                }
            }
        }
        (max_r_idx, max_d_idx)
    } else {
        // Case 1: No window and no --search. Use the center point (delay=0, rate=0).
        (padding_length_half, fft_point_half - 1)
    };

    let delay_max_amp = delay_rate_2d_data_array[[peak_rate_idx, peak_delay_idx]];
    let delay_phase = safe_arg(&delay_rate_array[[peak_rate_idx, peak_delay_idx]]).to_degrees();
    let delay_rate_slice = delay_rate_2d_data_array.column(peak_delay_idx).to_owned();
    let refined_peak_3x3 = if search_mode == Some("peak") && !args.frequency {
        refine_peak_3x3_quadratic(
            &delay_rate_2d_data_array,
            peak_rate_idx,
            peak_delay_idx,
            &rate_range,
            &delay_range,
            &args.rrange,
            &args.drange,
        )
    } else {
        None
    };

    let mut residual_delay_val: f32 = delay_range[peak_delay_idx];
    let mut delay_offset = 0.0;
    if search_mode == Some("peak") {
        if let Some((refined_delay, _)) = refined_peak_3x3 {
            delay_offset = refined_delay;
            residual_delay_val = refined_delay;
        } else {
            // Fallback: 1D quadratic fitting on delay axis using center +/- 1 bins.
            let mut x_coords: Vec<f64> = Vec::new();
            let mut y_values: Vec<f64> = Vec::new();
            let delay_bounds = window_bounds(&args.drange);

            for i in -1isize..=1 {
                let current_idx = peak_delay_idx as isize + i;
                if current_idx >= 0 && current_idx < delay_range.len() as isize {
                    let d_idx = current_idx as usize;
                    let delay_val = delay_range[d_idx];
                    if in_window(delay_val, delay_bounds) {
                        x_coords.push(delay_val as f64);
                        y_values.push(delay_rate_2d_data_array[[peak_rate_idx, d_idx]] as f64);
                    }
                }
            }

            if x_coords.len() >= 3 {
                if let Ok(fit_result) = fitting::fit_quadratic_least_squares(&x_coords, &y_values) {
                    delay_offset = fit_result.peak_x as f32;
                    residual_delay_val = delay_offset;
                } else {
                    eprintln!("Warning: Quadratic fitting for delay failed. Using original peak.");
                }
            }
        }
    }

    let delay_snr = delay_max_amp / delay_noise;
    let visibility = delay_rate_2d_data_array.row(peak_rate_idx).to_owned();

    let mut residual_rate_val: f32;
    let mut rate_offset = 0.0;

    // --- Frequency Analysis ---
    let freq_rate_2d_data_array = freq_rate_array.clone().mapv(|x| x.norm());

    let (peak_freq_row_idx, peak_rate_col_idx) =
        if !args.rrange.is_empty() || !args.frange.is_empty() {
            // Case 3: Window option is specified.
            let (rate_win_low, rate_win_high) = if !args.rrange.is_empty() {
                (args.rrange[0], args.rrange[1])
            } else {
                (rate_range[0], *rate_range.last().unwrap_or(&rate_range[0]))
            };
            let (freq_win_low, freq_win_high) = if args.frange.len() == 2 {
                (args.frange[0], args.frange[1])
            } else {
                (freq_range[0], *freq_range.last().unwrap_or(&freq_range[0]))
            };

            let mut max_val_in_window = 0.0f32;
            let mut temp_peak_freq_row_idx = 0;
            let mut temp_peak_rate_col_idx = padding_length_half;

            for r_idx in 0..rate_range.len() {
                if rate_range[r_idx] >= rate_win_low && rate_range[r_idx] <= rate_win_high {
                    for f_idx in 0..freq_range.len() {
                        let freq_mhz = freq_range[f_idx];
                        if freq_mhz < freq_win_low || freq_mhz > freq_win_high {
                            continue;
                        }
                        let current_val = freq_rate_2d_data_array[[f_idx, r_idx]];
                        if current_val > max_val_in_window {
                            max_val_in_window = current_val;
                            temp_peak_freq_row_idx = f_idx;
                            temp_peak_rate_col_idx = r_idx;
                        }
                    }
                }
            }
            (temp_peak_freq_row_idx, temp_peak_rate_col_idx)
        } else if search_mode == Some("peak") || search_mode == Some("deep") {
            // Case 2: --search or --search_deep is specified, no window. Find the global maximum.
            let (mut max_val, mut max_f_idx, mut max_r_idx) = (0.0f32, 0, 0);
            let (freq_win_low, freq_win_high) = if args.frange.len() == 2 {
                (args.frange[0], args.frange[1])
            } else {
                (freq_range[0], *freq_range.last().unwrap_or(&freq_range[0]))
            };
            for f_idx in 0..freq_rate_2d_data_array.shape()[0] {
                let freq_mhz = freq_range[f_idx];
                if freq_mhz < freq_win_low || freq_mhz > freq_win_high {
                    continue;
                }
                for r_idx in 0..freq_rate_2d_data_array.shape()[1] {
                    let current_val = freq_rate_2d_data_array[[f_idx, r_idx]];
                    if current_val > max_val {
                        max_val = current_val;
                        max_f_idx = f_idx;
                        max_r_idx = r_idx;
                    }
                }
            }
            (max_f_idx, max_r_idx)
        } else {
            // Case 1: No window and no --search. Use the center point (rate=0) and find max frequency.
            let cross_power_slice = freq_rate_2d_data_array.column(padding_length_half);
            let (max_f_idx, _) = cross_power_slice.iter().enumerate().fold(
                (0, 0.0f32),
                |(i_max, v_max), (i, &v)| {
                    if v > v_max {
                        (i, v)
                    } else {
                        (i_max, v_max)
                    }
                },
            );
            (max_f_idx, padding_length_half)
        };

    let freq_max_amp = freq_rate_2d_data_array[[peak_freq_row_idx, peak_rate_col_idx]];
    let freq_phase =
        safe_arg(&freq_rate_array[[peak_freq_row_idx, peak_rate_col_idx]]).to_degrees();
    let mut freq_freq = freq_range[peak_freq_row_idx];

    // Calculate noise from regions away from the peak rate
    let peak_rate_hz = rate_range[peak_rate_col_idx];
    let noise_rate_threshold = 0.1; // Hz

    let mut noise_complex_values = Vec::new();
    for (r_idx, &rate_val) in rate_range.iter().enumerate() {
        if (rate_val - peak_rate_hz).abs() > noise_rate_threshold {
            for f_idx in 0..freq_rate_array.shape()[0] {
                noise_complex_values.push(freq_rate_array[[f_idx, r_idx]]);
            }
        }
    }

    let freq_noise_raw = if !noise_complex_values.is_empty() {
        let noise_sum: C32 = noise_complex_values.iter().sum();
        let noise_mean: C32 = noise_sum / (noise_complex_values.len() as f32);
        let noise_abs_dev_sum: f32 = noise_complex_values
            .iter()
            .map(|c| (c - noise_mean).norm())
            .sum();
        noise_abs_dev_sum / noise_complex_values.len() as f32
    } else {
        // Fallback to old method if no noise region is found
        eprintln!("Warning: Could not find noise region for frequency SNR calculation. Falling back to old method.");
        noise_level(freq_rate_array.view(), freq_rate_array.mean().unwrap())
    };

    if search_mode == Some("peak") {
        let center_idx = peak_freq_row_idx as isize;
        if center_idx > 0 && center_idx + 1 < freq_range.len() as isize {
            let left_idx = (center_idx - 1) as usize;
            let mid_idx = center_idx as usize;
            let right_idx = (center_idx + 1) as usize;

            let y_left = freq_rate_2d_data_array[[left_idx, peak_rate_col_idx]] as f64;
            let y_mid = freq_rate_2d_data_array[[mid_idx, peak_rate_col_idx]] as f64;
            let y_right = freq_rate_2d_data_array[[right_idx, peak_rate_col_idx]] as f64;

            let is_local_max = y_mid.is_finite()
                && y_left.is_finite()
                && y_right.is_finite()
                && y_mid >= y_left
                && y_mid >= y_right;
            let side_max = y_left.max(y_right);
            // For near-delta peaks at very high frequency resolution, 3-point quadratic fitting
            // becomes numerically unstable and adds little value. Skip fitting in that regime.
            let is_delta_like = y_mid > 0.0 && (side_max / y_mid) < 0.02;

            if is_local_max && !is_delta_like {
                let x_coords = vec![
                    freq_range[left_idx] as f64,
                    freq_range[mid_idx] as f64,
                    freq_range[right_idx] as f64,
                ];
                let y_values = vec![y_left, y_mid, y_right];
                if let Ok(fit_result) = fitting::fit_quadratic_least_squares(&x_coords, &y_values)
                {
                    let x_min = x_coords[0].min(x_coords[2]);
                    let x_max = x_coords[0].max(x_coords[2]);
                    if fit_result.peak_x >= x_min && fit_result.peak_x <= x_max {
                        freq_freq = fit_result.peak_x as f32;
                    }
                }
            }
        }
    }

    let freq_noise = sanitize_noise(freq_noise_raw);
    let freq_snr = freq_max_amp / freq_noise;
    let freq_rate = freq_rate_2d_data_array.row(peak_freq_row_idx).to_owned();
    let freq_rate_spectrum = freq_rate_array.column(peak_rate_col_idx).to_owned();

    // Set the final residual_rate_val based on the mode
    if args.frequency {
        residual_rate_val = rate_range[peak_rate_col_idx];
    } else {
        residual_rate_val = rate_range[peak_rate_idx];
    };

    if search_mode == Some("peak") {
        if args.length == 1 {
            // When length is 1, rate fitting is unstable, so force residual_rate to 0.
            //eprintln!("Warning: Rate fitting is skipped because --length is 1. Residual rate is set to 0.");
            residual_rate_val = 0.0;
            rate_offset = 0.0;
        } else if args.frequency {
            let mut x_coords: Vec<f64> = Vec::new();
            let mut y_values: Vec<f64> = Vec::new();
            let rate_bounds = window_bounds(&args.rrange);
            for i in -1isize..=1 {
                let current_idx = peak_rate_col_idx as isize + i;
                if current_idx >= 0 && current_idx < rate_range.len() as isize {
                    let r_idx = current_idx as usize;
                    let rate_val = rate_range[r_idx];
                    if in_window(rate_val, rate_bounds) {
                        x_coords.push(rate_val as f64);
                        y_values.push(freq_rate_2d_data_array[[peak_freq_row_idx, r_idx]] as f64);
                    }
                }
            }

            let rate_scale_factor = (10.0 * padding_length as f64) * effective_integ_time as f64;
            let scaled_x_coords: Vec<f64> =
                x_coords.iter().map(|&x| x * rate_scale_factor).collect();

            if scaled_x_coords.len() >= 3 {
                if let Ok(fit_result) =
                    fitting::fit_quadratic_least_squares(&scaled_x_coords, &y_values)
                {
                    rate_offset = (fit_result.peak_x / rate_scale_factor) as f32;
                    residual_rate_val = rate_offset;
                }
            }
        } else if let Some((_, refined_rate)) = refined_peak_3x3 {
            rate_offset = refined_rate;
            residual_rate_val = refined_rate;
        } else {
            let mut x_coords: Vec<f64> = Vec::new();
            let mut y_values: Vec<f64> = Vec::new();
            let rate_bounds = window_bounds(&args.rrange);
            for i in -1isize..=1 {
                let current_idx = peak_rate_idx as isize + i;
                if current_idx >= 0 && current_idx < rate_range.len() as isize {
                    let r_idx = current_idx as usize;
                    let rate_val = rate_range[r_idx];
                    if in_window(rate_val, rate_bounds) {
                        x_coords.push(rate_val as f64);
                        y_values.push(delay_rate_array[[r_idx, peak_delay_idx]].norm() as f64);
                    }
                }
            }

            let rate_scale_factor = (10.0 * padding_length as f64) * effective_integ_time as f64;
            let scaled_x_coords: Vec<f64> =
                x_coords.iter().map(|&x| x * rate_scale_factor).collect();

            if scaled_x_coords.len() >= 3 {
                if let Ok(fit_result) =
                    fitting::fit_quadratic_least_squares(&scaled_x_coords, &y_values)
                {
                    rate_offset = (fit_result.peak_x / rate_scale_factor) as f32;
                    residual_rate_val = rate_offset;
                }
            }
        }
    }

    // --- Sky Coordinate Calculation ---
    let (u, v, _w, du_dt, dv_dt) = uvw_cal(
        header.station1_position,
        header.station2_position,
        *obs_time,
        header.source_position_ra,
        header.source_position_dec,
        true,
    );

    let (l_coord, m_coord) = rate_delay_to_lm(
        residual_rate_val as f64,
        residual_delay_val as f64,
        header,
        u,
        v,
        du_dt,
        dv_dt,
    );

    // --- Antenna Az/El Calculation ---
    let (ant1_az, ant1_el, ant1_hgt) = radec2azalt(
        [
            header.station1_position[0] as f32,
            header.station1_position[1] as f32,
            header.station1_position[2] as f32,
        ],
        *obs_time,
        header.source_position_ra as f32,
        header.source_position_dec as f32,
    );
    let (ant2_az, ant2_el, ant2_hgt) = radec2azalt(
        [
            header.station2_position[0] as f32,
            header.station2_position[1] as f32,
            header.station2_position[2] as f32,
        ],
        *obs_time,
        header.source_position_ra as f32,
        header.source_position_dec as f32,
    );

    AnalysisResults {
        yyyydddhhmmss1: obs_time.format("%Y/%j %H:%M:%S").to_string(),
        source_name: header.source_name.clone(),
        length_f32,
        ant1_az: ant1_az as f32,
        ant1_el: ant1_el as f32,
        ant1_hgt: ant1_hgt as f32,
        ant2_az: ant2_az as f32,
        ant2_el: ant2_el as f32,
        ant2_hgt: ant2_hgt as f32,
        mjd: mjd_cal(*obs_time),
        delay_range,
        visibility,
        delay_rate: delay_rate_slice,
        delay_max_amp,
        delay_phase,
        delay_snr,
        delay_noise,
        residual_delay: residual_delay_val,
        corrected_delay: args.delay_correct,
        delay_offset,
        freq_max_amp,
        freq_phase,
        freq_freq,
        freq_snr,
        freq_noise,
        freq_rate,
        freq_rate_spectrum,
        freq_range,
        freq_max_freq: freq_freq,
        residual_rate: residual_rate_val,
        corrected_rate: args.rate_correct,
        rate_offset,
        // Initialize new fields
        // residual_acel: 0.0, // Placeholder
        corrected_acel: args.acel_correct,
        rate_range,
        l_coord,
        m_coord,
    }
}
