use crate::args::Args;
use crate::npy_output::{NamedNpz, NpyMeta};
use crate::output::generate_output_names;
use crate::png_compress::{compress_png_with_mode, CompressQuality};
use crate::processing::ProcessResult;
use crate::utils;
use nalgebra::{Matrix3, Vector3};
use ndarray::Array2;
use plotters::prelude::*;
use plotters::style::colors::colormaps::ViridisRGB;
use std::error::Error;
use std::f64::consts::PI;
use std::fs;
use std::path::Path;

// Weighted Wavelet Z-transform implementation based on Foster (1996) and
// guided by the BSD-3-Clause Python reference in /home/akimoto/program/python/wwz.
// See LICENSE-3RD-PARTY-WWZ in the project root for the retained license text.

const DEFAULT_C: f64 = 1.0 / (8.0 * PI * PI);
const MIN_POINTS: usize = 6;

struct WwzSeries<'a> {
    suffix: &'a str,
    label: &'a str,
    values: Vec<f64>,
}

struct WwzTransform {
    tau: Vec<f64>,
    freq: Vec<f64>,
    period: Vec<f64>,
    wwz: Array2<f64>,
    wwa: Array2<f64>,
    n_eff: Array2<f64>,
    peak_period: Vec<f64>,
    peak_power: Vec<f64>,
}

#[derive(Clone, Copy)]
enum WwzYAxis {
    PeriodLinear,
    PeriodLog,
    FrequencyLinear,
}

pub fn write_wwz_outputs(
    args: &Args,
    result: &ProcessResult,
    frinz_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    if !args.wwz {
        return Ok(());
    }
    if args.frequency {
        eprintln!(
            "Warning: --wwz currently supports only delay-rate fringe-search results. Skipping."
        );
        return Ok(());
    }
    if result.add_plot_times.len() < MIN_POINTS {
        eprintln!(
            "Warning: --wwz requires at least {} fringe-search points, but only {} are available. Skipping.",
            MIN_POINTS,
            result.add_plot_times.len()
        );
        return Ok(());
    }
    if result.add_plot_amp.len() != result.add_plot_times.len()
        || result.add_plot_phase.len() != result.add_plot_times.len()
    {
        eprintln!("Warning: WWZ input series lengths are inconsistent. Skipping.");
        return Ok(());
    }

    let elapsed_times = elapsed_seconds(result)?;
    let output_dir = frinz_dir.join("wwz");
    fs::create_dir_all(&output_dir)?;

    let base_filename = make_base_filename(args, result);
    let mut phase_unwrapped = result.add_plot_phase.clone();
    utils::unwrap_phase(&mut phase_unwrapped, false);

    let series = [
        WwzSeries {
            suffix: "amp",
            label: "Fringe Amplitude",
            values: result
                .add_plot_amp
                .iter()
                .map(|&value| value as f64)
                .collect(),
        },
        WwzSeries {
            suffix: "phase_unwrapped",
            label: "Fringe Phase (unwrapped)",
            values: phase_unwrapped.iter().map(|&value| value as f64).collect(),
        },
    ];

    let mut wrote_any = false;
    for current_series in series {
        match compute_wwz(&elapsed_times, &current_series.values) {
            Ok(transform) => {
                if args.npz {
                    write_wwz_npz(
                        &output_dir,
                        &base_filename,
                        &current_series,
                        &transform,
                        result,
                    )?;
                }
                plot_wwz_heatmap(
                    &output_dir,
                    &base_filename,
                    &current_series,
                    &transform,
                    result,
                    WwzYAxis::PeriodLinear,
                )?;
                plot_wwz_heatmap(
                    &output_dir,
                    &base_filename,
                    &current_series,
                    &transform,
                    result,
                    WwzYAxis::PeriodLog,
                )?;
                plot_wwz_heatmap(
                    &output_dir,
                    &base_filename,
                    &current_series,
                    &transform,
                    result,
                    WwzYAxis::FrequencyLinear,
                )?;
                wrote_any = true;
            }
            Err(err) => {
                eprintln!(
                    "Warning: WWZ for {} was skipped: {}",
                    current_series.suffix, err
                );
            }
        }
    }

    if !wrote_any {
        eprintln!("Warning: --wwz produced no output files.");
    }

    Ok(())
}

fn make_base_filename(args: &Args, result: &ProcessResult) -> String {
    let mut base = generate_output_names(
        &result.header,
        &result.obs_time,
        &result
            .label
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>(),
        !args.rfi.is_empty(),
        args.frequency,
        args.bandpass.is_some(),
        result.length_arg,
    );
    if args.in_beam && !base.ends_with("_inbeam") {
        base.push_str("_inbeam");
    }
    base
}

fn elapsed_seconds(result: &ProcessResult) -> Result<Vec<f64>, Box<dyn Error>> {
    if !result.wwz_times_sec.is_empty() {
        let first_time = result.wwz_times_sec[0] as f64;
        let elapsed: Vec<f64> = result
            .wwz_times_sec
            .iter()
            .map(|time| *time as f64 - first_time)
            .collect();
        if elapsed.len() < 2 {
            return Err("WWZ input needs at least two timestamps".into());
        }
        if elapsed.windows(2).any(|window| {
            window[1] <= window[0] || !window[0].is_finite() || !window[1].is_finite()
        }) {
            return Err("WWZ timestamps are not strictly increasing".into());
        }
        return Ok(elapsed);
    }

    let first_time = result
        .add_plot_times
        .first()
        .copied()
        .ok_or("WWZ input has no timestamps")?;
    let elapsed: Vec<f64> = result
        .add_plot_times
        .iter()
        .map(|time| time.signed_duration_since(first_time).num_milliseconds() as f64 / 1000.0)
        .collect();
    if elapsed.len() < 2 {
        return Err("WWZ input needs at least two timestamps".into());
    }
    if elapsed
        .windows(2)
        .any(|window| window[1] <= window[0] || !window[0].is_finite() || !window[1].is_finite())
    {
        return Err("WWZ timestamps are not strictly increasing".into());
    }
    Ok(elapsed)
}

fn compute_wwz(times: &[f64], values: &[f64]) -> Result<WwzTransform, Box<dyn Error>> {
    if times.len() != values.len() {
        return Err("time/value length mismatch".into());
    }
    if times.len() < MIN_POINTS {
        return Err(format!("need at least {} points", MIN_POINTS).into());
    }

    let valid_values: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if valid_values.len() != values.len() {
        return Err("series contains non-finite values".into());
    }
    let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
    let variance = valid_values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f64>()
        / valid_values.len() as f64;
    if variance <= 1e-12 {
        return Err("series variance is too small for WWZ".into());
    }

    let positive_diffs: Vec<f64> = times
        .windows(2)
        .filter_map(|window| {
            let diff = window[1] - window[0];
            if diff > 0.0 && diff.is_finite() {
                Some(diff)
            } else {
                None
            }
        })
        .collect();
    if positive_diffs.is_empty() {
        return Err("WWZ timestamps are not strictly increasing".into());
    }

    let median_dt = median(&positive_diffs);
    let total_span = times.last().copied().unwrap_or(0.0) - times.first().copied().unwrap_or(0.0);
    if total_span <= median_dt {
        return Err("time span is too short for WWZ".into());
    }

    let freq = build_frequency_grid(total_span, median_dt, values.len())?;
    let tau = build_tau_grid(times, values.len());
    let period: Vec<f64> = freq.iter().map(|frequency| 1.0 / *frequency).collect();

    let tau_len = tau.len();
    let freq_len = freq.len();
    let mut wwz = Array2::<f64>::zeros((tau_len, freq_len));
    let mut wwa = Array2::<f64>::zeros((tau_len, freq_len));
    let mut n_eff = Array2::<f64>::zeros((tau_len, freq_len));
    let mut peak_period = vec![0.0; tau_len];
    let mut peak_power = vec![0.0; tau_len];

    for (tau_idx, tau_value) in tau.iter().enumerate() {
        let mut best_power = f64::NEG_INFINITY;
        let mut best_period = *period.first().unwrap_or(&0.0);

        for (freq_idx, frequency) in freq.iter().enumerate() {
            let omega = 2.0 * PI * *frequency;
            let mut s = Matrix3::<f64>::zeros();
            let mut p = Vector3::<f64>::zeros();
            let mut weighted_sum = 0.0;
            let mut weighted_square_sum = 0.0;
            let mut weight_sum = 0.0;
            let mut weight_square_sum = 0.0;

            for (&time, &value) in times.iter().zip(values.iter()) {
                let delta = time - *tau_value;
                let phase = omega * delta;
                let weight = (-DEFAULT_C * omega * omega * delta * delta).exp();
                let basis = Vector3::new(1.0, phase.cos(), phase.sin());

                weight_sum += weight;
                weight_square_sum += weight * weight;
                weighted_sum += weight * value;
                weighted_square_sum += weight * value * value;
                p += basis * (weight * value);
                s += basis * basis.transpose() * weight;
            }

            if weight_sum <= 1e-12 || weight_square_sum <= 1e-12 {
                continue;
            }

            p /= weight_sum;
            s /= weight_sum;
            let Some(s_inverse) = s.try_inverse() else {
                continue;
            };
            let coeffs = s_inverse * p;

            let mut model_sum = 0.0;
            let mut model_square_sum = 0.0;
            for &time in times {
                let delta = time - *tau_value;
                let phase = omega * delta;
                let weight = (-DEFAULT_C * omega * omega * delta * delta).exp();
                let model_value = coeffs[0] + coeffs[1] * phase.cos() + coeffs[2] * phase.sin();
                model_sum += weight * model_value;
                model_square_sum += weight * model_value * model_value;
            }

            let data_variance =
                (weighted_square_sum / weight_sum) - (weighted_sum / weight_sum).powi(2);
            let model_variance = (model_square_sum / weight_sum) - (model_sum / weight_sum).powi(2);
            let effective_points = weight_sum * weight_sum / weight_square_sum;
            let denominator = 2.0 * (data_variance - model_variance);
            let power = if effective_points > 3.0
                && denominator.abs() > 1e-12
                && data_variance.is_finite()
                && model_variance.is_finite()
            {
                ((effective_points - 3.0) * model_variance / denominator).max(0.0)
            } else {
                0.0
            };
            let amplitude = (coeffs[1] * coeffs[1] + coeffs[2] * coeffs[2]).sqrt();

            wwz[[tau_idx, freq_idx]] = if power.is_finite() { power } else { 0.0 };
            wwa[[tau_idx, freq_idx]] = if amplitude.is_finite() {
                amplitude
            } else {
                0.0
            };
            n_eff[[tau_idx, freq_idx]] = if effective_points.is_finite() {
                effective_points
            } else {
                0.0
            };

            if wwz[[tau_idx, freq_idx]] > best_power {
                best_power = wwz[[tau_idx, freq_idx]];
                best_period = period[freq_idx];
            }
        }

        peak_power[tau_idx] = if best_power.is_finite() && best_power > 0.0 {
            best_power
        } else {
            0.0
        };
        peak_period[tau_idx] = best_period;
    }

    Ok(WwzTransform {
        tau,
        freq,
        period,
        wwz,
        wwa,
        n_eff,
        peak_period,
        peak_power,
    })
}

fn build_frequency_grid(
    total_span: f64,
    median_dt: f64,
    point_count: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut p_min = median_dt * 4.0;
    let mut p_max = total_span / 5.0;

    if !p_min.is_finite() || !p_max.is_finite() || p_max <= p_min {
        p_min = median_dt * 2.0;
        p_max = total_span * 0.8;
    }
    if p_max <= p_min {
        p_max = p_min * 1.5;
    }
    if p_min <= 0.0 || p_max <= p_min {
        return Err("failed to derive a valid WWZ period range".into());
    }

    let freq_min = 1.0 / p_max;
    let freq_max = 1.0 / p_min;
    if !freq_min.is_finite() || !freq_max.is_finite() || freq_max <= freq_min {
        return Err("failed to derive a valid WWZ frequency range".into());
    }

    let n_freq = (point_count.saturating_mul(8)).clamp(32, 192);
    let periods = linspace(p_min, p_max, n_freq);
    Ok(periods
        .into_iter()
        .rev()
        .map(|period| 1.0 / period)
        .collect())
}

fn build_tau_grid(times: &[f64], point_count: usize) -> Vec<f64> {
    let start = times.first().copied().unwrap_or(0.0);
    let end = times.last().copied().unwrap_or(start);
    let n_tau = (point_count.saturating_mul(4)).clamp(16, 128);
    linspace(start, end, n_tau)
}

fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start];
    }
    let step = (end - start) / (count - 1) as f64;
    (0..count)
        .map(|index| start + step * index as f64)
        .collect()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

fn write_wwz_npz(
    output_dir: &Path,
    base_filename: &str,
    series: &WwzSeries<'_>,
    transform: &WwzTransform,
    result: &ProcessResult,
) -> Result<(), Box<dyn Error>> {
    let _ =
        fs::remove_file(output_dir.join(format!("{}_wwz_{}.tsv", base_filename, series.suffix)));
    let _ = fs::remove_file(
        output_dir.join(format!("{}_wwz_{}_ridge.tsv", base_filename, series.suffix)),
    );
    for legacy in [
        format!("{}_wwz_{}.npz", base_filename, series.suffix),
        format!("{}_wwa_{}.npz", base_filename, series.suffix),
        format!("{}_wwz_neff_{}.npz", base_filename, series.suffix),
        format!("{}_wwz_{}_ridge_power.npz", base_filename, series.suffix),
        format!("{}_wwz_{}_ridge_period.npz", base_filename, series.suffix),
    ] {
        let _ = fs::remove_file(output_dir.join(legacy));
    }
    let mut npz = NamedNpz::new(NpyMeta::new(
        &format!("wwz_{}", series.suffix),
        result.header.fft_point as u32,
        result.header.number_of_sector as u32,
    ));
    npz.add_f64_1d("time_s", &transform.tau);
    npz.add_f64_1d("frequency_hz", &transform.freq);
    npz.add_f64_1d("period_s", &transform.period);
    npz.add_f32_2d(
        "wwz_power",
        transform.wwz.dim(),
        transform.wwz.iter().map(|&v| v as f32),
    )?;
    npz.add_f32_2d(
        "wwa_amplitude",
        transform.wwa.dim(),
        transform.wwa.iter().map(|&v| v as f32),
    )?;
    npz.add_f32_2d(
        "effective_n",
        transform.n_eff.dim(),
        transform.n_eff.iter().map(|&v| v as f32),
    )?;
    let ridge_power: Vec<f64> = transform.peak_power.clone();
    let ridge_period_s: Vec<f64> = transform.peak_period.clone();
    npz.add_f64_1d("ridge_power", &ridge_power);
    npz.add_f64_1d("ridge_period_s", &ridge_period_s);
    npz.write(&output_dir.join(format!("{}_wwz_{}.npz", base_filename, series.suffix)))?;
    Ok(())
}

fn plot_wwz_heatmap(
    output_dir: &Path,
    base_filename: &str,
    series: &WwzSeries<'_>,
    transform: &WwzTransform,
    result: &ProcessResult,
    y_axis: WwzYAxis,
) -> Result<(), Box<dyn Error>> {
    let axis_suffix = match y_axis {
        WwzYAxis::PeriodLinear => "period",
        WwzYAxis::PeriodLog => "logperiod",
        WwzYAxis::FrequencyLinear => "freq",
    };
    let output_path = output_dir.join(format!(
        "{}_wwz_{}_{}.png",
        base_filename, series.suffix, axis_suffix
    ));
    let root = BitMapBackend::new(&output_path, (1100, 760)).into_drawing_area();
    root.fill(&WHITE)?;

    let (heatmap_area, colorbar_area) = root.split_horizontally(940);
    let x_edges = cell_edges(&transform.tau);
    let period_edges = positive_cell_edges(&transform.period);
    let x_min = x_edges
        .iter()
        .copied()
        .fold(f64::INFINITY, |acc, value| acc.min(value));
    let x_max = x_edges
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(value));
    let y_values = match y_axis {
        WwzYAxis::PeriodLinear | WwzYAxis::PeriodLog => transform.period.clone(),
        WwzYAxis::FrequencyLinear => transform
            .period
            .iter()
            .map(|period| 1.0 / period.max(1e-12))
            .collect(),
    };
    let y_edges = match y_axis {
        WwzYAxis::PeriodLinear | WwzYAxis::PeriodLog => period_edges.clone(),
        WwzYAxis::FrequencyLinear => period_edges
            .iter()
            .map(|period| 1.0 / period.max(1e-12))
            .collect(),
    };
    let y_min = y_edges
        .iter()
        .copied()
        .fold(f64::INFINITY, |acc, value| acc.min(value));
    let y_max = y_edges
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(value));
    let max_power = robust_color_max(transform.wwz.iter().copied()).max(1e-12);

    let build_heatmap_cells = || {
        (0..transform.tau.len())
            .flat_map(|tau_index| {
                let x0 = x_edges[tau_index].min(x_edges[tau_index + 1]);
                let x1 = x_edges[tau_index].max(x_edges[tau_index + 1]);
                (0..y_values.len())
                    .map(|freq_index| {
                        let y0 = y_edges[freq_index].min(y_edges[freq_index + 1]);
                        let y1 = y_edges[freq_index].max(y_edges[freq_index + 1]);
                        let normalized =
                            (transform.wwz[[tau_index, freq_index]] / max_power).clamp(0.0, 1.0);
                        let color = ViridisRGB.get_color(normalized);
                        Rectangle::new([(x0, y0), (x1, y1)], color.filled())
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let ridge_points = transform
        .tau
        .iter()
        .enumerate()
        .map(|(index, tau)| (*tau, ridge_y_value(transform, index, y_axis)))
        .collect::<Vec<_>>();

    match y_axis {
        WwzYAxis::PeriodLog => {
            let mut chart = ChartBuilder::on(&heatmap_area)
                .caption(
                    format!(
                        "{} WWZ: {} ({}, len={:.3} s)",
                        result.header.source_name,
                        series.label,
                        y_axis_title(y_axis),
                        result.length_sec
                    ),
                    ("sans-serif", 28).into_font(),
                )
                .margin(15)
                .x_label_area_size(60)
                .y_label_area_size(90)
                .build_cartesian_2d(x_min..x_max, (y_min..y_max).log_scale())?;

            chart
                .configure_mesh()
                .x_desc("Elapsed Time [s]")
                .y_desc(y_axis_title(y_axis))
                .x_label_formatter(&|value| format!("{:.0}", value))
                .y_label_formatter(&|value| y_axis_label(y_axis, *value))
                .x_labels(10)
                .y_labels(10)
                .label_style(("sans-serif", 20).into_font())
                .draw()?;

            chart.draw_series(build_heatmap_cells())?;
            chart.draw_series(LineSeries::new(ridge_points.iter().copied(), &WHITE))?;
        }
        _ => {
            let mut chart = ChartBuilder::on(&heatmap_area)
                .caption(
                    format!(
                        "{} WWZ: {} ({}, len={:.3} s)",
                        result.header.source_name,
                        series.label,
                        y_axis_title(y_axis),
                        result.length_sec
                    ),
                    ("sans-serif", 28).into_font(),
                )
                .margin(15)
                .x_label_area_size(60)
                .y_label_area_size(90)
                .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

            chart
                .configure_mesh()
                .x_desc("Elapsed Time [s]")
                .y_desc(y_axis_title(y_axis))
                .x_label_formatter(&|value| format!("{:.0}", value))
                .y_label_formatter(&|value| y_axis_label(y_axis, *value))
                .x_labels(10)
                .y_labels(10)
                .label_style(("sans-serif", 20).into_font())
                .draw()?;

            chart.draw_series(build_heatmap_cells())?;
            chart.draw_series(LineSeries::new(ridge_points.iter().copied(), &WHITE))?;
        }
    }

    let (bar_strip, label_strip) = colorbar_area.split_horizontally(34);
    let bar_strip = bar_strip.margin(40, 40, 16, 8);
    let label_strip = label_strip.margin(40, 40, 0, 8);
    let (bar_width, bar_height) = bar_strip.dim_in_pixel();
    let height_norm = (bar_height.saturating_sub(1)).max(1) as f64;
    for index in 0..bar_height as i32 {
        let frac = 1.0 - index as f64 / height_norm;
        let color = ViridisRGB.get_color(frac);
        bar_strip.draw(&Rectangle::new(
            [(0, index), (bar_width as i32, index + 1)],
            color.filled(),
        ))?;
    }
    for index in 0..=5 {
        let frac = index as f64 / 5.0;
        let y = ((1.0 - frac) * (bar_height.saturating_sub(1) as f64)).round() as i32;
        let value = frac * max_power;
        label_strip.draw(&Text::new(
            format!("{value:.2}"),
            (0, y),
            ("sans-serif", 18).into_font(),
        ))?;
    }
    label_strip.draw(&Text::new("WWZ", (0, 0), ("sans-serif", 20).into_font()))?;

    root.present()?;
    compress_png_with_mode(&output_path, CompressQuality::Low);
    Ok(())
}

fn ridge_y_value(transform: &WwzTransform, index: usize, y_axis: WwzYAxis) -> f64 {
    match y_axis {
        WwzYAxis::PeriodLinear | WwzYAxis::PeriodLog => transform.peak_period[index],
        WwzYAxis::FrequencyLinear => {
            let period = transform.peak_period[index].max(1e-12);
            1.0 / period
        }
    }
}

fn y_axis_title(y_axis: WwzYAxis) -> &'static str {
    match y_axis {
        WwzYAxis::PeriodLinear => "Period [s]",
        WwzYAxis::PeriodLog => "log Period [s]",
        WwzYAxis::FrequencyLinear => "Frequency [Hz]",
    }
}

fn y_axis_label(y_axis: WwzYAxis, value: f64) -> String {
    match y_axis {
        WwzYAxis::FrequencyLinear => format!("{value:.3}"),
        WwzYAxis::PeriodLog => {
            if value >= 10.0 {
                format!("{value:.0}")
            } else if value >= 1.0 {
                format!("{value:.1}")
            } else {
                format!("{value:.2}")
            }
        }
        WwzYAxis::PeriodLinear => format!("{value:.1}"),
    }
}

fn cell_edges(values: &[f64]) -> Vec<f64> {
    if values.len() == 1 {
        return vec![values[0] - 0.5, values[0] + 0.5];
    }

    let mut edges = Vec::with_capacity(values.len() + 1);
    edges.push(values[0] - 0.5 * (values[1] - values[0]));
    for index in 1..values.len() {
        edges.push(0.5 * (values[index - 1] + values[index]));
    }
    let last = values.len() - 1;
    edges.push(values[last] + 0.5 * (values[last] - values[last - 1]));
    edges
}

fn positive_cell_edges(values: &[f64]) -> Vec<f64> {
    if values.len() == 1 {
        let value = values[0].max(1e-12);
        return vec![value * 0.5, value * 1.5];
    }

    let safe_values: Vec<f64> = values.iter().map(|value| value.max(1e-12)).collect();
    let mut edges = Vec::with_capacity(safe_values.len() + 1);

    let first_ratio = (safe_values[1] / safe_values[0]).max(1.0 + 1e-6);
    edges.push((safe_values[0] / first_ratio.sqrt()).max(1e-12));

    for index in 1..safe_values.len() {
        edges.push((safe_values[index - 1] * safe_values[index]).sqrt());
    }

    let last = safe_values.len() - 1;
    let last_ratio = (safe_values[last] / safe_values[last - 1]).max(1.0 + 1e-6);
    edges.push(safe_values[last] * last_ratio.sqrt());
    edges
}

fn robust_color_max(values: impl Iterator<Item = f64>) -> f64 {
    let mut finite_values: Vec<f64> = values
        .filter(|value| value.is_finite() && *value >= 0.0)
        .collect();
    if finite_values.is_empty() {
        return 1.0;
    }
    finite_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let absolute_max = *finite_values.last().unwrap_or(&1.0);
    let percentile_index =
        ((finite_values.len().saturating_sub(1)) as f64 * 0.995).round() as usize;
    let percentile_max = finite_values[percentile_index.min(finite_values.len() - 1)];
    if absolute_max > percentile_max * 5.0 {
        percentile_max.max(1e-12)
    } else {
        absolute_max.max(1e-12)
    }
}
