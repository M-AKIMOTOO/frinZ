#![allow(dead_code)]

// Plot utilities for frinZ:
// - delay/rate/frequency planes and diagnostics
// - UV-related plots
// - multi-sideband summary plot (merged from former plot_msb.rs)
use crate::args::Args;
use crate::npy_output::{npz_sidecar_path, write_named_real_1d_npz, NamedNpz, NpyMeta};
use crate::output::{generate_output_names, insert_product_before_processing_suffixes};
use crate::png_compress::{compress_png, compress_png_with_mode, CompressQuality};
use crate::processing::ProcessResult;
use crate::utils::safe_arg;
use chrono::{DateTime, TimeZone, Utc};
use ndarray::Array2; // Added for dynamic spectrum
use num_complex::Complex;
use plotters::backend::RGBPixel;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::colors::colormaps::ViridisRGB;
use plotters::style::text_anchor::{HPos, Pos, VPos};
use rayon::prelude::*;
use std::error::Error;
use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::{fs::File, io::Write};

static PLOT_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
const PLOT_MAX_THREADS: usize = 3;

fn plot_pool() -> &'static rayon::ThreadPool {
    PLOT_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(PLOT_MAX_THREADS)
            .build()
            .expect("failed to build plot thread pool")
    })
}

fn build_heatmap_rgb_buffer<F>(
    width: u32,
    height: u32,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    amplitude_norm: f64,
    heatmap_func: &F,
) -> Vec<u8>
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    let width_usize = width as usize;
    let height_usize = height as usize;
    let x_den = (width_usize.saturating_sub(1)).max(1) as f64;
    let y_den = (height_usize.saturating_sub(1)).max(1) as f64;
    let amp = amplitude_norm.max(1e-30);
    let mut buffer = vec![0u8; width_usize * height_usize * 3];

    plot_pool().install(|| {
        buffer
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(idx, pixel)| {
                let px = idx % width_usize;
                let py = idx / width_usize;
                let x = x_min + (x_max - x_min) * px as f64 / x_den;
                let y = y_max - (y_max - y_min) * py as f64 / y_den;
                let normalized = (heatmap_func(x, y) / amp).clamp(0.0, 1.0);
                let (r, g, b) = HSLColor((1.0 - normalized) * 0.7, 1.0, 0.5).rgb();
                pixel[0] = r;
                pixel[1] = g;
                pixel[2] = b;
            });
    });

    buffer
}

pub fn delay_plane(
    delay_profile: &[(f64, f64)],
    delay_profile_pre_bp: Option<&[(f64, f64)]>,
    rate_profile: &[(f64, f64)],
    rate_profile_pre_bp: Option<&[(f64, f64)]>,
    heatmap_func: impl Fn(f64, f64) -> f64 + Sync,
    stat_keys: &[&str],
    stat_vals: &[&str],
    output_path: &str,
    rate_range: &[f32],
    length: f32,
    effective_integration_length: f32,
    drange: &[f32],
    rrange: &[f32],
    mask: Option<(f32, f32, f32, f32)>,
    max_amplitude: f64,
    _heatmap_res_x: usize,
    _heatmap_res_y: usize,
    heatmap_only: bool,
    //cmap_time: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let width = 1400;
    let height = 1000;
    let root = BitMapBackend::new(output_path, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(height / 2);
    let (upper_left, upper_right) = upper.split_horizontally(width / 2 - 10);
    let (lower_left, lower_right) = lower.split_horizontally(width / 2 - 10);
    let (heatmap_area, colorbar_area) = upper_right.split_horizontally((width / 2) - 110);

    // --- Determine axis ranges ---
    let delay_max = delay_profile
        .iter()
        .map(|(_, y)| *y)
        .chain(
            delay_profile_pre_bp
                .into_iter()
                .flat_map(|profile| profile.iter().map(|(_, y)| *y)),
        )
        .fold(f64::NEG_INFINITY, f64::max)
        * 1.1;
    let rate_max = rate_profile
        .iter()
        .map(|(_, y)| *y)
        .chain(
            rate_profile_pre_bp
                .into_iter()
                .flat_map(|profile| profile.iter().map(|(_, y)| *y)),
        )
        .fold(f64::NEG_INFINITY, f64::max)
        * 1.1;

    let (delay_line_min, delay_line_max, heatmap_delay_min, heatmap_delay_max) =
        if drange.is_empty() {
            (-32.0, 32.0, -10.0, 10.0)
        } else {
            (
                drange[0] as f64,
                drange[1] as f64,
                drange[0] as f64,
                drange[1] as f64,
            )
        };

    let (rate_line_min, rate_line_max, heatmap_rate_min, heatmap_rate_max) = if rrange.is_empty() {
        let rate_min_x = rate_profile
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::INFINITY, f64::min);
        let rate_max_x = rate_profile
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::NEG_INFINITY, f64::max);
        let rate_win_range_low = if (-8.0 / length as f64) < rate_range[0] as f64 {
            rate_range[0] as f64 * effective_integration_length as f64
        } else {
            -4.0 / (length as f64 * effective_integration_length as f64)
        };
        let rate_win_range_high = if (8.0 / length as f64) > *rate_range.last().unwrap() as f64 {
            *rate_range.last().unwrap() as f64 * effective_integration_length as f64
        } else {
            4.0 / (length as f64 * effective_integration_length as f64)
        };
        (
            rate_min_x,
            rate_max_x,
            rate_win_range_low,
            rate_win_range_high,
        )
    } else {
        (
            rrange[0] as f64,
            rrange[1] as f64,
            rrange[0] as f64,
            rrange[1] as f64,
        )
    };

    if heatmap_only {
        let (heatmap_area, colorbar_area) = root.split_horizontally(width - 120);
        let mut chart = ChartBuilder::on(&heatmap_area)
            .margin(20)
            .x_label_area_size(65)
            .y_label_area_size(95)
            .build_cartesian_2d(
                heatmap_delay_min..heatmap_delay_max,
                heatmap_rate_min..heatmap_rate_max,
            )?;

        chart
            .configure_mesh()
            .x_desc("Delay [Sample]")
            .y_desc("Rate [Hz]")
            .x_labels(9)
            .y_labels(10)
            .x_max_light_lines(0)
            .y_max_light_lines(0)
            .label_style(("sans-serif ", 28))
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&|v| format!("{:.2e}", v))
            .draw()?;

        let amplitude_norm = if max_amplitude.is_finite() && max_amplitude > 0.0 {
            max_amplitude
        } else {
            1e-30
        };
        let (plot_x, plot_y) = chart.plotting_area().get_pixel_range();
        let bitmap_width = plot_x.len().max(1) as u32;
        let bitmap_height = plot_y.len().max(1) as u32;
        let heatmap_buffer = build_heatmap_rgb_buffer(
            bitmap_width,
            bitmap_height,
            heatmap_delay_min,
            heatmap_delay_max,
            heatmap_rate_min,
            heatmap_rate_max,
            amplitude_norm,
            &heatmap_func,
        );
        let bitmap = BitMapElement::<_, RGBPixel>::with_owned_buffer(
            (heatmap_delay_min, heatmap_rate_max),
            (bitmap_width, bitmap_height),
            heatmap_buffer,
        )
        .ok_or("failed to create heatmap bitmap")?;
        chart.draw_series(std::iter::once(bitmap))?;
        if let Some((mask_delay_min, mask_delay_max, mask_rate_min, mask_rate_max)) = mask {
            let x0 = mask_delay_min as f64;
            let x1 = mask_delay_max as f64;
            let y0 = mask_rate_min as f64;
            let y1 = mask_rate_max as f64;
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x0, y0), (x1, y1)],
                WHITE.mix(0.18).filled(),
            )))?;
            let span_x = (x1 - x0).abs();
            let span_y = (y1 - y0).abs();
            let hatch_step = (span_x.min(span_y) / 12.0).max(0.2);
            let mut offset = -span_y;
            while offset <= span_x {
                let sx = x0 + offset.max(0.0);
                let sy = y0 + (-offset).max(0.0);
                let ex = x0 + (offset + span_y).min(span_x);
                let ey = y0 + (span_y + offset).min(span_y);
                chart.draw_series(std::iter::once(PathElement::new(
                    vec![(sx, sy), (ex, ey)],
                    BLACK.mix(0.35).stroke_width(1),
                )))?;
                offset += hatch_step;
            }
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x0, y0), (x1, y1)],
                BLACK.mix(0.45).stroke_width(1),
            )))?;
        }

        let mut colorbar = ChartBuilder::on(&colorbar_area)
            .margin_top(15)
            .margin_bottom(35)
            .set_label_area_size(LabelAreaPosition::Right, 90)
            .set_label_area_size(LabelAreaPosition::Left, 0)
            .set_label_area_size(LabelAreaPosition::Top, 10)
            .set_label_area_size(LabelAreaPosition::Bottom, 20)
            .build_cartesian_2d(0.0..1.0, 0.0..max_amplitude)?;

        let steps = 100;
        for i in 0..steps {
            let value = i as f64 / (steps - 1) as f64;
            let color = HSLColor(((1.0 - value) * 0.7).into(), 1.0, 0.5);
            colorbar.draw_series(std::iter::once(Rectangle::new(
                [
                    (0.0, value * max_amplitude),
                    (1.0, (value + 1.0 / steps as f64) * max_amplitude),
                ],
                color.filled(),
            )))?;
        }

        colorbar
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .disable_x_axis()
            .y_labels(7)
            .y_label_style(("sans-serif ", 24))
            .y_label_formatter(&|v| format!("{:.1e}", v))
            .draw()?;

        root.present()?;
        compress_png_with_mode(output_path, CompressQuality::High);
        return Ok(());
    }

    // 1. Horizontal slice (Delay Profile)
    let mut chart1 = ChartBuilder::on(&upper_left)
        .margin_top(20)
        .margin_left(15)
        .x_label_area_size(65)
        .y_label_area_size(120)
        .build_cartesian_2d(delay_line_min..delay_line_max, 0.0..delay_max)?;

    chart1
        .configure_mesh()
        .x_desc("Delay [Sample]")
        .y_desc("Amplitude")
        .x_labels(7)
        //.y_labels(7)
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .axis_style(BLACK.stroke_width(1))
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .label_style(("sans-serif ", 30))
        .draw()?;

    chart1.draw_series(LineSeries::new(
        delay_profile
            .iter()
            .filter(|(x, y)| {
                *x >= delay_line_min && *x <= delay_line_max && x.is_finite() && y.is_finite()
            })
            .cloned(),
        GREEN,
    ))?;
    if let Some(pre_bp) = delay_profile_pre_bp {
        chart1.draw_series(LineSeries::new(
            pre_bp
                .iter()
                .filter(|(x, y)| {
                    *x >= delay_line_min && *x <= delay_line_max && x.is_finite() && y.is_finite()
                })
                .cloned(),
            RED.stroke_width(1),
        ))?;
    }
    if let Some((mask_delay_min, mask_delay_max, _, _)) = mask {
        let x0 = mask_delay_min as f64;
        let x1 = mask_delay_max as f64;
        chart1.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, delay_max)],
            WHITE.mix(0.18).filled(),
        )))?;
        let hatch_step = ((x1 - x0).abs() / 20.0).max(0.1);
        let mut x = x0 - delay_max;
        while x <= x1 {
            let sx = x.max(x0);
            let sy = (x0 - x).max(0.0);
            let ex = (x + delay_max).min(x1);
            let ey = (ex - x).max(0.0).min(delay_max);
            chart1.draw_series(std::iter::once(PathElement::new(
                vec![(sx, sy), (ex, ey)],
                BLACK.mix(0.35).stroke_width(1),
            )))?;
            x += hatch_step;
        }
        chart1.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, delay_max)],
            BLACK.mix(0.45).stroke_width(1),
        )))?;
    }

    // Draw bounding box for chart1
    let x_spec = chart1.x_range();
    let y_spec = chart1.y_range();
    chart1.draw_series(std::iter::once(Rectangle::new(
        [(x_spec.start, y_spec.start), (x_spec.end, y_spec.end)],
        BLACK.stroke_width(1),
    )))?;

    // 2. Vertical slice (Rate Profile)
    let mut chart2 = ChartBuilder::on(&lower_left)
        .margin_top(10)
        .margin_left(15)
        .x_label_area_size(65)
        .y_label_area_size(120)
        .build_cartesian_2d(rate_line_min..rate_line_max, 0.0..rate_max)?;

    chart2
        .configure_mesh()
        .x_desc("Rate [Hz]")
        .y_desc("Amplitude")
        .x_labels(5)
        //.y_labels(7)
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .axis_style(BLACK.stroke_width(1))
        //.x_label_formatter(&|v| format!("{:.1}", v))
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .label_style(("sans-serif ", 30))
        .draw()?;

    chart2.draw_series(LineSeries::new(
        rate_profile
            .iter()
            .filter(|(x, y)| {
                *x >= rate_line_min && *x <= rate_line_max && x.is_finite() && y.is_finite()
            })
            .cloned(),
        GREEN,
    ))?;
    if let Some(pre_bp) = rate_profile_pre_bp {
        chart2.draw_series(LineSeries::new(
            pre_bp
                .iter()
                .filter(|(x, y)| {
                    *x >= rate_line_min && *x <= rate_line_max && x.is_finite() && y.is_finite()
                })
                .cloned(),
            RED.stroke_width(1),
        ))?;
    }
    if let Some((_, _, mask_rate_min, mask_rate_max)) = mask {
        let x0 = mask_rate_min as f64;
        let x1 = mask_rate_max as f64;
        chart2.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, rate_max)],
            WHITE.mix(0.18).filled(),
        )))?;
        let hatch_step = ((x1 - x0).abs() / 20.0).max(0.001);
        let mut x = x0 - rate_max;
        while x <= x1 {
            let sx = x.max(x0);
            let sy = (x0 - x).max(0.0);
            let ex = (x + rate_max).min(x1);
            let ey = (ex - x).max(0.0).min(rate_max);
            chart2.draw_series(std::iter::once(PathElement::new(
                vec![(sx, sy), (ex, ey)],
                BLACK.mix(0.35).stroke_width(1),
            )))?;
            x += hatch_step;
        }
        chart2.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, rate_max)],
            BLACK.mix(0.45).stroke_width(1),
        )))?;
    }

    // Draw bounding box for chart2
    let x_spec = chart2.x_range();
    let y_spec = chart2.y_range();
    chart2.draw_series(std::iter::once(Rectangle::new(
        [(x_spec.start, y_spec.start), (x_spec.end, y_spec.end)],
        BLACK.stroke_width(1),
    )))?;

    // 3. 2D heatmap
    let mut chart3 = ChartBuilder::on(&heatmap_area)
        .margin(20)
        .x_label_area_size(55)
        .y_label_area_size(120)
        .build_cartesian_2d(
            heatmap_delay_min..heatmap_delay_max,
            heatmap_rate_min..heatmap_rate_max,
        )?;

    chart3
        .configure_mesh()
        .x_desc("Delay [Sample]")
        .y_desc("Rate [Hz]")
        .x_labels(7)
        .y_labels(10)
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .label_style(("sans-serif ", 30))
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.2e}", v))
        .draw()?;

    let (delay_min, delay_max_hm) = (heatmap_delay_min, heatmap_delay_max);
    let (rate_min_hm, rate_max_hm) = (heatmap_rate_min, heatmap_rate_max);

    let amplitude_norm = if max_amplitude.is_finite() && max_amplitude > 0.0 {
        max_amplitude
    } else {
        1e-30
    };

    let (plot_x, plot_y) = chart3.plotting_area().get_pixel_range();
    let bitmap_width = plot_x.len().max(1) as u32;
    let bitmap_height = plot_y.len().max(1) as u32;
    let heatmap_buffer = build_heatmap_rgb_buffer(
        bitmap_width,
        bitmap_height,
        delay_min,
        delay_max_hm,
        rate_min_hm,
        rate_max_hm,
        amplitude_norm,
        &heatmap_func,
    );
    let bitmap = BitMapElement::<_, RGBPixel>::with_owned_buffer(
        (delay_min, rate_max_hm),
        (bitmap_width, bitmap_height),
        heatmap_buffer,
    )
    .ok_or("failed to create heatmap bitmap")?;
    chart3.draw_series(std::iter::once(bitmap))?;
    if let Some((mask_delay_min, mask_delay_max, mask_rate_min, mask_rate_max)) = mask {
        let x0 = mask_delay_min as f64;
        let x1 = mask_delay_max as f64;
        let y0 = mask_rate_min as f64;
        let y1 = mask_rate_max as f64;
        chart3.draw_series(std::iter::once(Rectangle::new(
            [(x0, y0), (x1, y1)],
            WHITE.mix(0.18).filled(),
        )))?;
        let span_x = (x1 - x0).abs();
        let span_y = (y1 - y0).abs();
        let hatch_step = (span_x.min(span_y) / 12.0).max(0.2);
        let mut offset = -span_y;
        while offset <= span_x {
            let sx = x0 + offset.max(0.0);
            let sy = y0 + (-offset).max(0.0);
            let ex = x0 + (offset + span_y).min(span_x);
            let ey = y0 + (span_y + offset).min(span_y);
            chart3.draw_series(std::iter::once(PathElement::new(
                vec![(sx, sy), (ex, ey)],
                BLACK.mix(0.35).stroke_width(1),
            )))?;
            offset += hatch_step;
        }
        chart3.draw_series(std::iter::once(Rectangle::new(
            [(x0, y0), (x1, y1)],
            BLACK.mix(0.45).stroke_width(1),
        )))?;
    }

    // 4. Colorbar
    let mut colorbar = ChartBuilder::on(&colorbar_area)
        .margin_top(10)
        .margin_bottom(50)
        .set_label_area_size(LabelAreaPosition::Right, 100)
        .set_label_area_size(LabelAreaPosition::Left, 0)
        .set_label_area_size(LabelAreaPosition::Top, 10)
        .set_label_area_size(LabelAreaPosition::Bottom, 25)
        .build_cartesian_2d(0.0..1.0, 0.0..max_amplitude)?;

    let steps = 100;
    for i in 0..steps {
        let value = i as f64 / (steps - 1) as f64;
        let color = HSLColor(((1.0 - value) * 0.7).into(), 1.0, 0.5);
        colorbar.draw_series(std::iter::once(Rectangle::new(
            [
                (0.0, value * max_amplitude),
                (1.0, (value + 1.0 / steps as f64) * max_amplitude),
            ],
            color.filled(),
        )))?;
    }

    colorbar
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .disable_x_axis()
        .y_labels(7)
        .y_label_style(("sans-serif ", 30))
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .draw()?;

    // 5. Stats
    let font = ("sans-serif ", 35).into_font();
    let left_x = 30;
    let right_x = 670;
    let mut y = 20;
    let text_style = TextStyle::from(font.clone()).color(&BLACK);
    for (k, v) in stat_keys.iter().zip(stat_vals.iter()) {
        lower_right.draw(&Text::new(k.to_string(), (left_x, y), text_style.clone()))?;
        lower_right.draw(&Text::new(
            v.to_string(),
            (right_x, y + 15),
            text_style.clone().pos(Pos::new(HPos::Right, VPos::Center)),
        ))?;
        y += 35;
    }
    if delay_profile_pre_bp.is_some() || rate_profile_pre_bp.is_some() {
        let legend_font = ("sans-serif ", 28).into_font();
        lower_right.draw(&Text::new(
            "Green: BP corrected / Red: pre-BP",
            (left_x, y + 20),
            TextStyle::from(legend_font).color(&BLACK),
        ))?;
    }

    root.present()?;
    compress_png_with_mode(output_path, CompressQuality::High);
    Ok(())
}

pub fn frequency_plane(
    freq_amp_profile: &[(f64, f64)],
    freq_amp_profile_pre_bp: Option<&[(f64, f64)]>,
    freq_phase_profile: &[(f64, f64)],
    freq_phase_profile_pre_bp: Option<&[(f64, f64)]>,
    rate_profile: &[(f64, f64)],
    rate_profile_pre_bp: Option<&[(f64, f64)]>,
    heatmap_func: impl Fn(f64, f64) -> f64 + Sync,
    stat_keys: &[&str],
    stat_vals: &[&str],
    output_path: &str,
    bw: f64,
    max_amplitude: f64,
    frange: &[f32],
    heatmap_res_x: usize,
    heatmap_res_y: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let width = 1400;
    let height = 1000;
    let root = BitMapBackend::new(output_path, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let (left_area, right_area) = root.split_horizontally(width / 2);
    let (top_left_area, rate_area) = left_area.split_vertically(height / 2 + 50);
    let (phase_area, amp_area) =
        top_left_area.split_vertically(top_left_area.get_pixel_range().1.len() as u32 / 4 - 10);
    let (heatmap_and_colorbar_area, stats_area) = right_area.split_vertically(height / 2 + 50);
    let (heatmap_area, colorbar_area) = heatmap_and_colorbar_area
        .split_horizontally(heatmap_and_colorbar_area.get_pixel_range().0.len() as u32 - 120);

    // --- Determine axis ranges ---
    let amp_max_y = freq_amp_profile
        .iter()
        .map(|(_, y)| *y)
        .chain(
            freq_amp_profile_pre_bp
                .into_iter()
                .flat_map(|profile| profile.iter().map(|(_, y)| *y)),
        )
        .fold(f64::NEG_INFINITY, f64::max)
        * 1.1;
    let rate_max_y = rate_profile
        .iter()
        .map(|(_, y)| *y)
        .chain(
            rate_profile_pre_bp
                .into_iter()
                .flat_map(|profile| profile.iter().map(|(_, y)| *y)),
        )
        .fold(f64::NEG_INFINITY, f64::max)
        * 1.1;
    let rate_min_x = rate_profile
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::INFINITY, f64::min);
    let rate_max_x = rate_profile
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::NEG_INFINITY, f64::max);

    let (freq_min, freq_max) = if frange.len() == 2 {
        (frange[0] as f64, frange[1] as f64)
    } else {
        (0.0, bw)
    };

    // --- Phase Chart ---
    let mut phase_chart = ChartBuilder::on(&phase_area)
        .margin_top(20)
        .margin_left(15)
        .x_label_area_size(0)
        .y_label_area_size(120)
        .build_cartesian_2d(freq_min..freq_max, -180.0..180.0)?;
    phase_chart
        .configure_mesh()
        //.disable_x_mesh()
        //.x_labels(0)
        .x_labels(7)
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .axis_style(BLACK.stroke_width(1))
        .y_desc("Phase")
        .y_label_formatter(&|y| format!("{:.0}", y))
        .y_labels(6)
        .label_style(("sans-serif ", 30))
        .draw()?;
    phase_chart.draw_series(LineSeries::new(
        freq_phase_profile
            .iter()
            .filter(|(x, y)| *x >= freq_min && *x <= freq_max && x.is_finite() && y.is_finite())
            .cloned(),
        GREEN.stroke_width(1),
    ))?;
    if let Some(pre_bp) = freq_phase_profile_pre_bp {
        phase_chart.draw_series(LineSeries::new(
            pre_bp
                .iter()
                .filter(|(x, y)| *x >= freq_min && *x <= freq_max && x.is_finite() && y.is_finite())
                .cloned(),
            RED.stroke_width(1),
        ))?;
    }

    // Draw bounding box for phase_chart
    let x_spec = phase_chart.x_range();
    let y_spec = phase_chart.y_range();
    phase_chart.draw_series(std::iter::once(Rectangle::new(
        [(x_spec.start, y_spec.start), (x_spec.end, y_spec.end)],
        BLACK.stroke_width(1),
    )))?;

    // --- Amplitude Chart ---
    let mut amp_chart = ChartBuilder::on(&amp_area)
        .margin_left(15)
        .x_label_area_size(65)
        .y_label_area_size(120)
        .build_cartesian_2d(freq_min..freq_max, 0.0..amp_max_y)?;
    amp_chart
        .configure_mesh()
        .x_desc("Frequency [MHz]")
        .y_desc("Amplitude")
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .x_labels(7)
        .y_labels(7)
        .axis_style(BLACK.stroke_width(1))
        .label_style(("sans-serif ", 30))
        .x_label_formatter(&|y| format!("{:.0}", y))
        .y_label_formatter(&|y| format!("{:.1e}", y))
        .draw()?;
    amp_chart.draw_series(LineSeries::new(
        freq_amp_profile
            .iter()
            .filter(|(x, y)| *x >= freq_min && *x <= freq_max && x.is_finite() && y.is_finite())
            .cloned(),
        GREEN.stroke_width(1),
    ))?;
    if let Some(pre_bp) = freq_amp_profile_pre_bp {
        amp_chart.draw_series(LineSeries::new(
            pre_bp
                .iter()
                .filter(|(x, y)| *x >= freq_min && *x <= freq_max && x.is_finite() && y.is_finite())
                .cloned(),
            RED.stroke_width(1),
        ))?;
    }

    // Draw bounding box for amp_chart
    let x_spec = amp_chart.x_range();
    let y_spec = amp_chart.y_range();
    amp_chart.draw_series(std::iter::once(Rectangle::new(
        [(x_spec.start, y_spec.start), (x_spec.end, y_spec.end)],
        BLACK.stroke_width(1),
    )))?;

    // --- Rate Chart ---
    let mut rate_chart = ChartBuilder::on(&rate_area)
        .margin_top(10)
        .margin_left(15)
        .x_label_area_size(75)
        .y_label_area_size(120)
        .build_cartesian_2d(rate_min_x..rate_max_x, 0.0..rate_max_y)?;
    rate_chart
        .configure_mesh()
        .x_desc("Rate [Hz]")
        .y_desc("Amplitude")
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .x_labels(7)
        //.y_labels(7)
        .axis_style(BLACK.stroke_width(1))
        .label_style(("sans-serif ", 30))
        .y_label_formatter(&|y| format!("{:.1e}", y))
        .draw()?;
    rate_chart.draw_series(LineSeries::new(
        rate_profile
            .iter()
            .filter(|(x, y)| *x >= rate_min_x && *x <= rate_max_x && x.is_finite() && y.is_finite())
            .cloned(),
        GREEN.stroke_width(1),
    ))?;
    if let Some(pre_bp) = rate_profile_pre_bp {
        rate_chart.draw_series(LineSeries::new(
            pre_bp
                .iter()
                .filter(|(x, y)| {
                    *x >= rate_min_x && *x <= rate_max_x && x.is_finite() && y.is_finite()
                })
                .cloned(),
            RED.stroke_width(1),
        ))?;
    }

    // Draw bounding box for rate_chart
    let x_spec = rate_chart.x_range();
    let y_spec = rate_chart.y_range();
    rate_chart.draw_series(std::iter::once(Rectangle::new(
        [(x_spec.start, y_spec.start), (x_spec.end, y_spec.end)],
        BLACK.stroke_width(1),
    )))?;

    // --- Heatmap Chart ---
    let mut heatmap_chart = ChartBuilder::on(&heatmap_area)
        .margin(20)
        .x_label_area_size(55)
        .y_label_area_size(85)
        .build_cartesian_2d(0.0..bw, rate_min_x..rate_max_x)?;
    heatmap_chart
        .configure_mesh()
        .x_desc("Frequency [MHz]")
        .y_desc("Rate [Hz]")
        .x_labels(7)
        .y_labels(7)
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .x_label_formatter(&|y| format!("{:.0}", y))
        .y_label_formatter(&|y| format!("{:.1}", y))
        .label_style(("sans-serif ", 30))
        .draw()?;

    let resolution_x = heatmap_res_x.max(2);
    let resolution_y = heatmap_res_y.max(2);
    let mut heatmap_values = Vec::new();
    let mut heatmap_data_max_val = f64::NEG_INFINITY;
    for i in 0..resolution_y {
        for j in 0..resolution_x {
            let y = rate_min_x + (rate_max_x - rate_min_x) * i as f64 / (resolution_y - 1) as f64;
            let x = bw * j as f64 / (resolution_x - 1) as f64;
            let val = heatmap_func(x, y);
            heatmap_values.push(val);
            if val > heatmap_data_max_val {
                heatmap_data_max_val = val;
            }
        }
    }

    for (idx, val) in heatmap_values.iter().enumerate() {
        let i = idx / resolution_x;
        let j = idx % resolution_x;
        let y = rate_min_x + (rate_max_x - rate_min_x) * i as f64 / (resolution_y - 1) as f64;
        let x = bw * j as f64 / (resolution_x - 1) as f64;
        let x_step = bw / (resolution_x - 1) as f64;
        let y_step = (rate_max_x - rate_min_x) / (resolution_y - 1) as f64;
        let normalized_val = if heatmap_data_max_val > 0.0 {
            *val / heatmap_data_max_val
        } else {
            0.0
        };
        heatmap_chart.draw_series(std::iter::once(Rectangle::new(
            [(x, y), (x + x_step, y + y_step)],
            HSLColor((1.0 - normalized_val) * 0.7, 1.0, 0.5).filled(),
        )))?;
    }

    // 4. Colorbar
    let mut colorbar = ChartBuilder::on(&colorbar_area)
        .margin_top(10)
        .margin_bottom(50)
        .set_label_area_size(LabelAreaPosition::Right, 100)
        .set_label_area_size(LabelAreaPosition::Left, 0)
        .set_label_area_size(LabelAreaPosition::Top, 10)
        .set_label_area_size(LabelAreaPosition::Bottom, 25)
        .build_cartesian_2d(0.0..1.0, 0.0..max_amplitude)?;

    let steps = 100;
    for i in 0..steps {
        let value = i as f64 / (steps - 1) as f64;
        let color = HSLColor(((1.0 - value) * 0.7).into(), 1.0, 0.5);
        colorbar.draw_series(std::iter::once(Rectangle::new(
            [
                (0.0, value * max_amplitude),
                (1.0, (value + 1.0 / steps as f64) * max_amplitude),
            ],
            color.filled(),
        )))?;
    }

    colorbar
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .disable_x_axis()
        .y_labels(7)
        .y_label_style(("sans-serif ", 30))
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .draw()?;

    // 5. Stats
    let font = ("sans-serif ", 35).into_font();
    let left_x = 30;
    let right_x = stats_area.get_pixel_range().0.len() as i32 - 30;
    let mut y = 20;
    let text_style = TextStyle::from(font.clone()).color(&BLACK);
    for (k, v) in stat_keys.iter().zip(stat_vals.iter()) {
        stats_area.draw(&Text::new(k.to_string(), (left_x, y), text_style.clone()))?;
        stats_area.draw(&Text::new(
            v.to_string(),
            (right_x, y),
            text_style.clone().pos(Pos::new(HPos::Right, VPos::Top)),
        ))?;
        y += 35;
    }
    if freq_amp_profile_pre_bp.is_some()
        || freq_phase_profile_pre_bp.is_some()
        || rate_profile_pre_bp.is_some()
    {
        let legend_font = ("sans-serif ", 28).into_font();
        stats_area.draw(&Text::new(
            "Green: BP corrected / Red: pre-BP",
            (left_x, y + 20),
            TextStyle::from(legend_font).color(&BLACK),
        ))?;
    }

    root.present()?;
    compress_png_with_mode(output_path, CompressQuality::High);
    Ok(())
}

use crate::utils;

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

pub fn write_cumulate_outputs(
    args: &Args,
    result: &ProcessResult,
    frinz_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    if args.cumulate == 0 {
        return Ok(());
    }

    let path = if args.in_beam {
        frinz_dir.to_path_buf()
    } else {
        frinz_dir.join(format!("cumulate/len{}s", args.cumulate))
    };
    std::fs::create_dir_all(&path)?;
    cumulate_plot(
        &result.cumulate_len,
        &result.cumulate_snr,
        &path,
        &result.header,
        &result
            .label
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>(),
        &result.obs_time,
        args.cumulate,
        args.in_beam,
        !args.rfi.is_empty(),
        args.bandpass.is_some(),
    )?;
    if args.npz
        && !result.cumulate_len.is_empty()
        && result.cumulate_len.len() == result.cumulate_snr.len()
    {
        let axis: Vec<f64> = result
            .cumulate_len
            .iter()
            .map(|&value| value as f64)
            .collect();
        let base_filename = make_base_filename(args, result);
        let output_stem = insert_product_before_processing_suffixes(&base_filename, "cumulate");
        let npy_path = path.join(format!("{output_stem}.npz"));
        let mut npz = NamedNpz::new(NpyMeta::new(
            "cumulate",
            result.header.fft_point as u32,
            result.header.number_of_sector as u32,
        ));
        npz.add_f64_1d("integration_time_s", &axis);
        npz.add_f32_1d("snr", &result.cumulate_snr);
        npz.write(&npy_path)?;
    }
    Ok(())
}

pub fn write_add_plot_outputs(
    args: &Args,
    result: &ProcessResult,
    frinz_dir: &Path,
) -> Result<String, Box<dyn Error>> {
    let base_filename = make_base_filename(args, result);
    if !args.add_plot && args.cumulate == 0 {
        return Ok(base_filename);
    }

    let path: PathBuf = if args.in_beam {
        frinz_dir.to_path_buf()
    } else if args.cumulate != 0 {
        frinz_dir.join(format!("cumulate/len{}s", args.cumulate))
    } else {
        frinz_dir.join("add_plot")
    };
    std::fs::create_dir_all(&path)?;
    let add_plot_stem = if args.cumulate != 0 {
        insert_product_before_processing_suffixes(&base_filename, "cumulate")
    } else {
        base_filename.clone()
    };
    let add_plot_filepath = path.join(&add_plot_stem);
    let legacy_stem = insert_product_before_processing_suffixes(&base_filename, "add_plot_data");
    let _ = std::fs::remove_file(path.join(format!("{legacy_stem}.tsv")));
    if args.cumulate != 0 {
        let legacy_add_plot_path = path.join(&base_filename);
        for suffix in [
            "amp",
            "snr",
            "phase",
            "freq",
            "phase_unwrapped",
            "noise",
            "resdelay",
            "resrate",
        ] {
            let legacy_plot_stem = insert_product_before_processing_suffixes(
                legacy_add_plot_path.to_str().unwrap(),
                suffix,
            );
            let _ = std::fs::remove_file(format!("{legacy_plot_stem}.png"));
        }
        let legacy_tsv_stem = insert_product_before_processing_suffixes(
            legacy_add_plot_path.to_str().unwrap(),
            "add_plot",
        );
        let _ = std::fs::remove_file(format!("{legacy_tsv_stem}.tsv"));
    }

    if !result.add_plot_times.is_empty() {
        let elapsed_times_f32: Vec<f32> = if args.cumulate != 0 {
            if let Some(first_length) = result.cumulate_len.first() {
                result
                    .cumulate_len
                    .iter()
                    .map(|value| value - first_length)
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            let first_time = result.add_plot_times[0];
            result
                .add_plot_times
                .iter()
                .map(|dt| (*dt - first_time).num_seconds() as f32)
                .collect()
        };

        add_plot(
            add_plot_filepath.to_str().unwrap(),
            &elapsed_times_f32,
            &result.add_plot_amp,
            &result.add_plot_snr,
            &result.add_plot_phase,
            &result.add_plot_freq,
            &result.add_plot_noise,
            &result.add_plot_res_delay,
            &result.add_plot_res_rate,
            &result.header.source_name,
            result.length_sec,
            &result.obs_time,
        )?;

        let tsv_stem = insert_product_before_processing_suffixes(
            add_plot_filepath.to_str().unwrap(),
            "add_plot",
        );
        write_add_plot_tsv(
            &PathBuf::from(format!("{tsv_stem}.tsv")),
            &elapsed_times_f32,
            &result.add_plot_amp,
            &result.add_plot_snr,
            &result.add_plot_phase,
            &result.add_plot_freq,
            &result.add_plot_noise,
            &result.add_plot_res_delay,
            &result.add_plot_res_rate,
        )?;

        let axis: Vec<f64> = elapsed_times_f32
            .iter()
            .map(|&value| value as f64)
            .collect();
        if args.npz {
            let rows = [
                axis.len(),
                result.add_plot_amp.len(),
                result.add_plot_snr.len(),
                result.add_plot_phase.len(),
                result.add_plot_noise.len(),
                result.add_plot_res_delay.len(),
                result.add_plot_res_rate.len(),
            ]
            .into_iter()
            .min()
            .unwrap_or(0);

            let old_flags = [
                "add_plot_amp",
                "add_plot_snr",
                "add_plot_phase",
                "add_plot_noise",
                "add_plot_resdelay",
                "add_plot_resrate",
            ];
            for flag in old_flags {
                let _ = std::fs::remove_file(npz_sidecar_path(&add_plot_filepath, flag));
            }

            let time_s: Vec<f64> = axis[..rows].to_vec();
            let amp_percent: Vec<f64> = result.add_plot_amp[..rows]
                .iter()
                .map(|&value| value as f64)
                .collect();
            let phase_deg: Vec<f64> = result.add_plot_phase[..rows]
                .iter()
                .map(|&value| value as f64)
                .collect();
            let snr: Vec<f64> = result.add_plot_snr[..rows]
                .iter()
                .map(|&value| value as f64)
                .collect();
            let res_delay_sample: Vec<f64> = result.add_plot_res_delay[..rows]
                .iter()
                .map(|&value| value as f64)
                .collect();
            let res_rate_hz: Vec<f64> = result.add_plot_res_rate[..rows]
                .iter()
                .map(|&value| value as f64)
                .collect();
            let noise_percent: Vec<f64> = result.add_plot_noise[..rows]
                .iter()
                .map(|&value| value as f64)
                .collect();
            write_named_real_1d_npz(
                &npz_sidecar_path(&add_plot_filepath, "add_plot"),
                NpyMeta::new(
                    "add_plot",
                    result.header.fft_point.max(0) as u32,
                    result.header.number_of_sector.max(0) as u32,
                ),
                &[
                    ("time_s", &time_s),
                    ("amp_percent", &amp_percent),
                    ("phase_deg", &phase_deg),
                    ("snr", &snr),
                    ("res_delay_sample", &res_delay_sample),
                    ("res_rate_hz", &res_rate_hz),
                    ("noise_percent", &noise_percent),
                ],
            )?;
        }
    }

    Ok(base_filename)
}

fn write_add_plot_tsv(
    output_path: &Path,
    length: &[f32],
    amp: &[f32],
    snr: &[f32],
    phase: &[f32],
    freq: &[f32],
    noise: &[f32],
    res_delay: &[f32],
    res_rate: &[f32],
) -> Result<(), Box<dyn Error>> {
    let rows = [
        length.len(),
        amp.len(),
        snr.len(),
        phase.len(),
        noise.len(),
        res_delay.len(),
        res_rate.len(),
    ]
    .into_iter()
    .min()
    .unwrap_or(0);

    let mut file = File::create(output_path)?;
    writeln!(
        file,
        "# Time\tAmp\tSNR\tPhase\tFrequency\tNoise-level\tRes-Delay\tRes-Rate"
    )?;
    writeln!(
        file,
        "# [s]\t[%]\t-\t[deg]\t[MHz]\t1-sigma[%]\t[sample]\t[Hz]"
    )?;
    for idx in 0..rows {
        let frequency = if freq.len() == rows {
            format!("{:.7}", freq[idx])
        } else {
            "-".to_string()
        };
        writeln!(
            file,
            "{:.5}\t{:.6}\t{:.1}\t{:.3}\t{}\t{:.5e}\t{:.8}\t{:.8}",
            length[idx],
            amp[idx],
            snr[idx],
            phase[idx],
            frequency,
            noise[idx],
            res_delay[idx],
            res_rate[idx]
        )?;
    }
    Ok(())
}

pub fn add_plot(
    output_path: &str,
    length: &[f32],
    amp: &[f32],
    snr: &[f32],
    phase: &[f32],
    freq: &[f32],
    noise: &[f32],
    res_delay: &[f32],
    res_rate: &[f32],
    source_name: &str,
    len_val_sec: f32,
    obs_start_time: &DateTime<Utc>,
) -> Result<(), Box<dyn std::error::Error>> {
    let phase_unwrapped = utils::unwrap_phase_with_rate(phase, length, res_rate);
    let phase_unwrapped_slice: &[f32] = phase_unwrapped.as_slice();

    let plots = vec![
        (amp, "Amplitude [%]", "amp"),
        (snr, "SNR", "snr"),
        (phase, "Phase [deg]", "phase"),
        (freq, "Peak Frequency [MHz]", "freq"),
        (
            phase_unwrapped_slice,
            "Phase (unwrapped) [deg]",
            "phase_unwrapped",
        ),
        (noise, "Noise Level [%]", "noise"),
        (res_delay, "Residual Delay [sample]", "resdelay"),
        (res_rate, "Residual Rate [Hz]", "resrate"),
    ];

    for (data, y_label, filename_suffix) in plots {
        if data.is_empty() {
            continue;
        }
        let output_stem = insert_product_before_processing_suffixes(output_path, filename_suffix);
        let file_path = format!("{output_stem}.png");
        let root = BitMapBackend::new(&file_path, (900, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut y_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let mut y_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if filename_suffix == "phase" {
            y_min = -180.0;
            y_max = 180.0;
        } else if (y_max - y_min).abs() <= f32::EPSILON {
            let margin = (y_min.abs() * 1.0e-6).max(1.0e-6);
            y_min -= margin;
            y_max += margin;
        }

        let x_range = if length.len() > 1 {
            *length.first().unwrap()..*length.last().unwrap()
        } else {
            // Handle case with a single data point
            let center = length.first().unwrap_or(&0.0);
            (center - 1.0)..(center + 1.0)
        };

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("{}, length: {:.3} s", source_name, len_val_sec),
                ("sans-serif ", 25).into_font(),
            )
            .margin(10)
            .x_label_area_size(60)
            .y_label_area_size(100)
            .build_cartesian_2d(x_range, y_min..y_max)?;

        chart
            .configure_mesh()
            .x_desc(&format!(
                "The elapsed time since {} UT",
                obs_start_time.format("%Y/%j %H:%M:%S")
            ))
            .y_desc(y_label)
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&|v| {
                if filename_suffix == "phase" {
                    format!("{:.0}", v)
                } else if filename_suffix == "snr" {
                    format!("{:.0}", v)
                } else if filename_suffix == "resdelay" {
                    format!("{:.3}", v)
                } else if filename_suffix == "freq" {
                    format!("{:.3}", v)
                } else if filename_suffix == "resrate" {
                    format!("{:.2e}", v)
                } else {
                    format!("{:.1e}", v)
                }
            })
            .y_labels(if filename_suffix == "phase" { 7 } else { 5 })
            .x_max_light_lines(0)
            .y_max_light_lines(0)
            .label_style(("sans-serif ", 25).into_font())
            .draw()?;

        chart.draw_series(PointSeries::of_element(
            length.iter().zip(data.iter()).map(|(x, y)| (*x, *y)),
            5,
            GREEN,
            &|c, s, st| {
                return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
                    + Circle::new((0,0),s,st.filled()); // At point center, draw a circle
            },
        ))?;

        root.present()?;
        compress_png_with_mode(&file_path, CompressQuality::Low);
    }

    Ok(())
}

pub fn cumulate_plot(
    cumulate_len: &[f32],
    cumulate_snr: &[f32],
    cumulate_path: &std::path::PathBuf,
    header: &crate::header::CorHeader,
    label: &[&str],
    obs_time: &chrono::DateTime<chrono::Utc>,
    cumulate_arg: i32,
    in_beam: bool,
    is_rfi_filtered: bool,
    is_bandpass_corrected: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut base_filename = crate::output::generate_output_names(
        header,
        obs_time,
        label,
        is_rfi_filtered,
        false,
        is_bandpass_corrected,
        cumulate_arg,
    );
    if in_beam && !base_filename.ends_with("_inbeam") {
        base_filename.push_str("_inbeam");
    }
    let product = format!("{}_cumulate{}", header.source_name, cumulate_arg);
    let output_stem = insert_product_before_processing_suffixes(&base_filename, &product);
    let cumulate_filename = format!("{output_stem}.png");
    let cumulate_filepath = cumulate_path.join(cumulate_filename);

    let root = BitMapBackend::new(&cumulate_filepath, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Determine the Y-axis range dynamically
    let data_min_snr = cumulate_snr
        .iter()
        .cloned()
        .filter(|&x| x > 0.0)
        .fold(f32::INFINITY, f32::min);
    let data_max_snr = *cumulate_snr
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&100.0);

    let y_axis_start = if data_min_snr.is_finite() && data_min_snr > 0.0 {
        // Calculate the nearest power of 10 below the minimum data point
        10.0_f32.powf(data_min_snr.log10().floor())
    } else {
        // Fallback if data is empty or contains no positive values
        1.0
    };
    let y_axis_end = data_max_snr * 1.05;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(100)
        .build_cartesian_2d(
            (*cumulate_len.first().unwrap()..*cumulate_len.last().unwrap()).log_scale(),
            (y_axis_start..y_axis_end).log_scale(),
        )?;

    chart
        .configure_mesh()
        .x_desc("Integration Time [s]")
        .y_desc("S/N ")
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.0}", v))
        .x_labels(10)
        .y_labels(10)
        .label_style(("sans-serif ", 25).into_font())
        .max_light_lines(9)
        .light_line_style(&BLACK.mix(0.15))
        .draw()?;

    chart
        .draw_series(
            LineSeries::new(
                cumulate_len
                    .iter()
                    .zip(cumulate_snr.iter())
                    .map(|(x, y)| (*x, *y)),
                GREEN.filled(),
            )
            .point_size(5),
        )
        .unwrap();

    // Power-law fitting on log-log scale: y = a * x^b
    if cumulate_len.len() >= 2 {
        let ln_x: Vec<f64> = cumulate_len
            .iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| (v as f64).ln())
            .collect();
        let ln_y: Vec<f64> = cumulate_snr
            .iter()
            .zip(cumulate_len.iter())
            .filter_map(|(&y, &x)| {
                if x > 0.0 && y > 0.0 {
                    Some((y as f64).ln())
                } else {
                    None
                }
            })
            .collect();
        if ln_x.len() == ln_y.len() && ln_x.len() >= 2 {
            let n = ln_x.len() as f64;
            let mean_x = ln_x.iter().sum::<f64>() / n;
            let mean_y = ln_y.iter().sum::<f64>() / n;
            let cov_xy = ln_x
                .iter()
                .zip(ln_y.iter())
                .map(|(x, y)| (x - mean_x) * (y - mean_y))
                .sum::<f64>();
            let var_x = ln_x
                .iter()
                .map(|x| (x - mean_x) * (x - mean_x))
                .sum::<f64>();
            if var_x > 0.0 {
                let b_fit = cov_xy / var_x;
                let a_fit = (mean_y - b_fit * mean_x).exp();
                let x_min = *cumulate_len.first().unwrap() as f64;
                let x_max = *cumulate_len.last().unwrap() as f64;
                let fit_series = LineSeries::new(
                    (0..200).map(|i| {
                        let t = i as f64 / 199.0;
                        let x = x_min * (x_max / x_min).powf(t);
                        let y = a_fit * x.powf(b_fit);
                        (x as f32, y as f32)
                    }),
                    &RED,
                );
                chart
                    .draw_series(fit_series)?
                    .label(format!("fit: y = {:.2e} x^{:.3}", a_fit, b_fit));

                // Reference line with b = 0.5, forced to pass the first point
                let a_ref = (cumulate_snr[0] as f64) / (cumulate_len[0] as f64).powf(0.5);
                let ref_series = LineSeries::new(
                    (0..200).map(|i| {
                        let t = i as f64 / 199.0;
                        let x = x_min * (x_max / x_min).powf(t);
                        let y = a_ref * x.powf(0.5);
                        (x as f32, y as f32)
                    }),
                    &BLUE,
                );
                chart
                    .draw_series(ref_series)?
                    .label(format!("b=0.5: y = {:.2e} x^{:.1}", a_ref, 0.5));

                chart
                    .configure_series_labels()
                    .background_style(&WHITE.mix(0.8))
                    .border_style(&BLACK)
                    .label_font(("sans-serif", 18).into_font())
                    .draw()?;
            }
        }
    }

    root.present()?;
    compress_png_with_mode(&cumulate_filepath, CompressQuality::Low);
    Ok(())
}

pub fn phase_reference_plot(
    cal_times: &[DateTime<Utc>],
    original_cal_phases: &[f32],
    fitted_cal_phases: &[f32],
    target_times: &[DateTime<Utc>],
    original_target_phases: &[f32],
    residual_target_phases: &[f32],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let all_times: Vec<&DateTime<Utc>> = cal_times.iter().chain(target_times.iter()).collect();
    if all_times.is_empty() {
        println!("Warning: No data to plot in phase_reference_plot.");
        return Ok(());
    }
    let mut x_min = **all_times.iter().min().unwrap();
    let mut x_max = **all_times.iter().max().unwrap();

    if x_max == x_min {
        let duration = chrono::Duration::seconds(1);
        x_min = x_min - duration;
        x_max = x_max + duration;
    }

    // Determine y-axis range from all relevant phase data
    let all_phases: Vec<f32> = original_cal_phases
        .iter()
        .chain(fitted_cal_phases.iter())
        .chain(original_target_phases.iter())
        .chain(residual_target_phases.iter())
        .cloned()
        .collect();

    let y_min = all_phases.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_max = all_phases.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let y_range_padding = (y_max - y_min) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        //.caption("Phase Reference Plot", ("sans-serif", 40).into_font())
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(100)
        .build_cartesian_2d(
            x_min.timestamp() as f32..x_max.timestamp() as f32,
            (y_min - y_range_padding)..(y_max + y_range_padding),
        )?;

    chart
        .configure_mesh()
        .x_desc("Time [UTC]")
        .y_desc("Phase [deg]")
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .light_line_style(&TRANSPARENT)
        .x_label_formatter(&|ts| {
            chrono::Utc
                .timestamp_opt(*ts as i64, 0)
                .unwrap()
                .format("%H:%M:%S")
                .to_string()
        })
        .y_label_formatter(&|v| format!("{:.0}", v))
        .y_labels(7)
        .label_style(("sans-serif", 25).into_font())
        .draw()?;

    // 1. Original Calibrator Phase (Red Points)
    chart
        .draw_series(PointSeries::of_element(
            cal_times
                .iter()
                .zip(original_cal_phases.iter())
                .map(|(x, y)| (x.timestamp() as f32, *y)),
            5,
            &RED,
            &|c, s, st| Circle::new(c, s, st.filled()),
        ))?
        .label("Calibrator (Original)")
        .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()));

    // 2. Fitted Calibrator Phase (Blue Line)
    if !fitted_cal_phases.is_empty() {
        chart
            .draw_series(LineSeries::new(
                cal_times
                    .iter()
                    .zip(fitted_cal_phases.iter())
                    .map(|(x, y)| (x.timestamp() as f32, *y)),
                &BLUE,
            ))?
            .label("Calibrator (Fitted)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));
    }

    // 3. Original Target Phase (Green Circles)
    chart
        .draw_series(PointSeries::of_element(
            target_times
                .iter()
                .zip(original_target_phases.iter())
                .map(|(x, y)| (x.timestamp() as f32, *y)),
            5,
            &GREEN,
            &|c, s, st| Circle::new(c, s, st.filled()),
        ))?
        .label("Target (Original)")
        .legend(|(x, y)| Circle::new((x, y), 5, GREEN.filled()));

    // 4. Residual Phases from Calibrator Fit (Black Circles)
    chart
        .draw_series(PointSeries::of_element(
            target_times
                .iter()
                .zip(residual_target_phases.iter())
                .map(|(x, y)| (x.timestamp() as f32, *y)),
            5,
            &BLACK,
            &|c, s, st| Circle::new(c, s, st.filled()),
        ))?
        .label("Residual (Calibrator Fit)")
        .legend(|(x, y)| Circle::new((x, y), 5, BLACK.filled()));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    compress_png_with_mode(output_path, CompressQuality::Low);
    Ok(())
}

pub fn plot_allan_deviation(
    output_path: &str,
    data: &[(f32, f32)],
    source_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    if data.is_empty() {
        return Ok(());
    }

    let (min_tau, max_tau) = data
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), (t, _)| {
            (min.min(*t), max.max(*t))
        });
    let (min_adev, max_adev) = data
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), (_, a)| {
            (min.min(*a), max.max(*a))
        });

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Allan Deviation for {}", source_name),
            ("sans-serif", 25).into_font(),
        )
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(100)
        .build_cartesian_2d(
            (min_tau..max_tau).log_scale(),
            (min_adev..max_adev * 1.1).log_scale(),
        )?;

    chart
        .configure_mesh()
        .x_desc("Averaging Time (τ) [s]")
        .y_desc("Allan Deviation (σ_y(τ))")
        .x_labels(10)
        .y_labels(10)
        .label_style(("sans-serif", 25).into_font())
        .draw()?;

    chart.draw_series(LineSeries::new(data.iter().map(|(t, a)| (*t, *a)), &RED))?;
    chart.draw_series(PointSeries::of_element(
        data.iter().map(|(t, a)| (*t, *a)),
        5,
        &RED,
        &|c, s, st| Circle::new(c, s, st.filled()),
    ))?;

    root.present()?;
    compress_png_with_mode(output_path, CompressQuality::Low);
    Ok(())
}

pub fn plot_acel_search_result<P: AsRef<Path>>(
    output_path: P,
    times: &[f64],
    observed: &[f64],
    fitted: Option<&[f64]>,
    residuals: Option<&[f64]>,
    title: &str,
    y_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if times.is_empty() || observed.is_empty() {
        println!("Warning: No data to plot for {}.", title);
        return Ok(());
    }

    let mut min_time = f64::INFINITY;
    let mut max_time = f64::NEG_INFINITY;
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for (&t, &y) in times.iter().zip(observed.iter()) {
        min_time = min_time.min(t);
        max_time = max_time.max(t);
        min_val = min_val.min(y);
        max_val = max_val.max(y);
    }

    if let Some(fit) = fitted {
        for &y in fit {
            min_val = min_val.min(y);
            max_val = max_val.max(y);
        }
    }

    if (max_time - min_time).abs() < f64::EPSILON {
        max_time = min_time + 1.0;
    }
    if (max_val - min_val).abs() < f64::EPSILON {
        max_val += 1.0;
        min_val -= 1.0;
    } else {
        let pad = (max_val - min_val).abs() * 0.08;
        max_val += pad;
        min_val -= pad;
    }

    let rate_like = y_label.contains("Rate");
    let left_formatter = move |v: &f64| {
        if rate_like {
            format!("{:.2e}", v)
        } else {
            format!("{:.2}", v)
        }
    };
    let right_formatter = move |v: &f64| {
        if rate_like {
            format!("{:.2e}", v)
        } else {
            format!("{:.2}", v)
        }
    };

    let root = BitMapBackend::new(output_path.as_ref(), (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    if let Some(res) = residuals.filter(|r| !r.is_empty()) {
        let mut min_res = f64::INFINITY;
        let mut max_res = f64::NEG_INFINITY;
        for &y in res {
            min_res = min_res.min(y);
            max_res = max_res.max(y);
        }
        if (max_res - min_res).abs() < f64::EPSILON {
            max_res += 1.0;
            min_res -= 1.0;
        } else {
            let pad = (max_res - min_res).abs() * 0.12;
            max_res += pad;
            min_res -= pad;
        }

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 25).into_font())
            .margin(10)
            .x_label_area_size(60)
            .y_label_area_size(100)
            .right_y_label_area_size(100)
            .build_cartesian_2d(min_time..max_time, min_val..max_val)?
            .set_secondary_coord(min_time..max_time, min_res..max_res);

        chart
            .configure_mesh()
            .x_desc("Time [s]")
            .y_desc(y_label)
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&left_formatter)
            .x_labels(10)
            .y_labels(10)
            .x_max_light_lines(0)
            .y_max_light_lines(0)
            .label_style(("sans-serif", 20).into_font())
            .draw()?;

        chart
            .configure_secondary_axes()
            .y_desc(format!("Residual ({})", y_label))
            .y_label_formatter(&right_formatter)
            .label_style(("sans-serif", 20).into_font())
            .draw()?;

        chart
            .draw_series(PointSeries::of_element(
                times.iter().zip(observed.iter()).map(|(&t, &y)| (t, y)),
                3,
                &BLUE,
                &|c, s, st| Circle::new(c, s, st.filled()),
            ))?
            .label("Observed")
            .legend(|(x, y)| Circle::new((x, y), 4, BLUE.filled()));

        if let Some(fit) = fitted {
            chart
                .draw_series(LineSeries::new(
                    times.iter().zip(fit.iter()).map(|(&t, &y)| (t, y)),
                    &RED,
                ))?
                .label("Fitted")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
        }

        chart
            .draw_secondary_series(LineSeries::new(
                times.iter().zip(res.iter()).map(|(&t, &y)| (t, y)),
                &GREEN,
            ))?
            .label("Residual (right)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .label_font(("sans-serif", 18).into_font())
            .draw()?;
    } else {
        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 25).into_font())
            .margin(10)
            .x_label_area_size(60)
            .y_label_area_size(100)
            .build_cartesian_2d(min_time..max_time, min_val..max_val)?;

        chart
            .configure_mesh()
            .x_desc("Time [s]")
            .y_desc(y_label)
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&left_formatter)
            .x_labels(10)
            .y_labels(10)
            .x_max_light_lines(0)
            .y_max_light_lines(0)
            .label_style(("sans-serif", 20).into_font())
            .draw()?;

        chart
            .draw_series(PointSeries::of_element(
                times.iter().zip(observed.iter()).map(|(&t, &y)| (t, y)),
                3,
                &BLUE,
                &|c, s, st| Circle::new(c, s, st.filled()),
            ))?
            .label("Observed")
            .legend(|(x, y)| Circle::new((x, y), 4, BLUE.filled()));

        if let Some(fit) = fitted {
            chart
                .draw_series(LineSeries::new(
                    times.iter().zip(fit.iter()).map(|(&t, &y)| (t, y)),
                    &RED,
                ))?
                .label("Fitted")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
        }

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .label_font(("sans-serif", 18).into_font())
            .draw()?;
    }

    root.present()?;
    compress_png_with_mode(output_path.as_ref(), CompressQuality::Low);
    Ok(())
}

pub fn plot_sky_map<P: AsRef<Path>>(
    output_path: P,
    map_data: &ndarray::Array2<f32>,
    cell_size_rad: f64,
    max_x_idx: usize,
    max_y_idx: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let (height, width) = map_data.dim();
    // Define a fixed output image size for consistent plot appearance
    let backend_width = 1024;
    let backend_height = 800;
    let root = BitMapBackend::new(output_path.as_ref(), (backend_width, backend_height))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let (main_area, colorbar_area) = root.split_horizontally(backend_width - 120);

    let max_val = map_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = map_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));

    let rad_to_mas = 180.0 / PI * 3600.0 * 1_000.0;
    let l_mas_max = (width as f64 / 2.0) * cell_size_rad * rad_to_mas;
    let m_mas_max = (height as f64 / 2.0) * cell_size_rad * rad_to_mas;
    let l_range = l_mas_max..-l_mas_max; // Invert RA axis
    let m_range = -m_mas_max..m_mas_max;

    let mut chart = ChartBuilder::on(&main_area)
        .x_label_area_size(65)
        .y_label_area_size(80)
        .margin(10)
        .caption("Fringe Rate Map", ("sans-serif", 25))
        .build_cartesian_2d(l_range.clone(), m_range.clone())?;

    chart
        .configure_mesh()
        .x_desc("ΔRA (mas)")
        .y_desc("ΔDec (mas)")
        .x_label_formatter(&|x| format!("{:.0}", x))
        .y_label_formatter(&|y| format!("{:.0}", y))
        .label_style(("sans-serif", 30))
        .draw()?;

    // Draw the heatmap
    chart.draw_series(map_data.indexed_iter().map(|((y, x), &val)| {
        let l_mas = ((x as f64) - (width as f64 / 2.0)) * cell_size_rad * rad_to_mas;
        let m_mas = ((height as f64 / 2.0) - y as f64) * cell_size_rad * rad_to_mas;
        let cell_l_mas = cell_size_rad * rad_to_mas;
        let cell_m_mas = cell_size_rad * rad_to_mas;

        let mut norm_val = if max_val > min_val {
            (val - min_val) / (max_val - min_val)
        } else {
            0.0
        };
        if norm_val.is_nan() {
            norm_val = 0.0;
        }
        norm_val = norm_val.clamp(0.0, 1.0);
        let color = ViridisRGB.get_color(norm_val as f64);

        Rectangle::new(
            [(l_mas, m_mas), (l_mas + cell_l_mas, m_mas + cell_m_mas)],
            color.filled(),
        )
    }))?;

    // Add a white 'X' mark at (0,0)
    chart.draw_series(PointSeries::of_element(
        vec![(0.0, 0.0)],                                 // Center of the 'X'
        10,                                               // Size of the 'X'
        &WHITE,                                           // Color of the 'X'
        &|c, s, st| Cross::new(c, s, st.stroke_width(2)), // Draw a cross
    ))?;

    // Add a red 'X' mark at the maximum value position
    let (height, width) = map_data.dim(); // Get dimensions for calculation
    let l_mas_max_val = ((max_x_idx as f64) - (width as f64 / 2.0)) * cell_size_rad * rad_to_mas;
    let m_mas_max_val = ((height as f64 / 2.0) - max_y_idx as f64) * cell_size_rad * rad_to_mas;

    chart.draw_series(PointSeries::of_element(
        vec![(l_mas_max_val, m_mas_max_val)], // Center of the 'X'
        10,                                   // Size of the 'X'
        &RED,                                 // Color of the 'X'
        &|c, s, st| Cross::new(c, s, st.stroke_width(2)), // Draw a cross
    ))?;

    // Draw color bar
    let mut colorbar_chart = ChartBuilder::on(&colorbar_area)
        .margin(20)
        .set_label_area_size(LabelAreaPosition::Right, 40)
        .build_cartesian_2d(0f32..1f32, min_val..max_val)?;

    colorbar_chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_x_axis()
        .y_label_formatter(&|y| format!("{:.1e}", y))
        .y_label_style(("sans-serif", 20))
        .draw()?;

    let color_map = ViridisRGB;
    colorbar_chart.draw_series((0..200).map(|y| {
        let y_val = min_val + (max_val - min_val) * y as f32 / 199.0;
        let mut norm_val = if max_val > min_val {
            (y_val - min_val) / (max_val - min_val)
        } else {
            0.0
        };
        if norm_val.is_nan() {
            norm_val = 0.0;
        }
        norm_val = norm_val.clamp(0.0, 1.0);
        let color = color_map.get_color(norm_val as f64);
        Rectangle::new(
            [(0.0, y_val), (1.0, y_val + (max_val - min_val) / 199.0)],
            color.filled(),
        )
    }))?;

    root.present()?;
    compress_png_with_mode(output_path.as_ref(), CompressQuality::Low);
    Ok(())
}

pub fn plot_dynamic_spectrum_freq(
    output_path: &str,
    spectrum_array: &Array2<Complex<f32>>,
    header: &crate::header::CorHeader,
    obs_time: &DateTime<Utc>,
    _length: i32,
    _effective_integration_time: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let (time_samples, freq_channels) = spectrum_array.dim();

    let width = 1200;
    let height = 1000;
    let root = BitMapBackend::new(output_path, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(height / 2);

    let freq_range = 0..freq_channels;
    let time_range = 0..time_samples;

    // --- Amplitude Heatmap ---
    let amplitudes: Vec<f32> = spectrum_array.iter().map(|c| c.norm()).collect();
    let max_amp = amplitudes.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_amp = amplitudes.iter().cloned().fold(f32::INFINITY, f32::min);

    let mut amp_chart = ChartBuilder::on(&upper)
        .caption(
            format!(
                "Dynamic Spectrum (Amplitude) - {} - {}",
                header.source_name,
                obs_time.format("%Y-%m-%d %H:%M:%S")
            ),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(5)
        .y_label_area_size(100)
        .build_cartesian_2d(freq_range.clone(), time_range.clone())?;

    amp_chart
        .configure_mesh()
        .y_desc("Time [PP]")
        //.x_desc("Frequency [channels]")
        .label_style(("sans-serif", 25).into_font())
        .draw()?;

    amp_chart.draw_series(spectrum_array.indexed_iter().map(|((t, f), c)| {
        let norm_val = if max_amp > min_amp {
            (c.norm() - min_amp) / (max_amp - min_amp)
        } else {
            0.0
        };
        let color = ViridisRGB.get_color(norm_val as f64);
        Rectangle::new([(f, t), (f + 1, t + 1)], color.filled())
    }))?;

    // --- Phase Heatmap ---
    let mut phase_chart = ChartBuilder::on(&lower)
        .caption(
            format!(
                "Dynamic Spectrum (Phase) - {} - {}",
                header.source_name,
                obs_time.format("%Y-%m-%d %H:%M:%S")
            ),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(55)
        .y_label_area_size(100)
        .build_cartesian_2d(freq_range.clone(), time_range.clone())?;

    phase_chart
        .configure_mesh()
        .y_desc("Time [PP]")
        .x_desc("Frequency [channels]")
        .label_style(("sans-serif", 25).into_font())
        .draw()?;

    phase_chart.draw_series(spectrum_array.indexed_iter().map(|((t, f), c)| {
        let norm_val = (safe_arg(c).to_degrees() + 180.0) / 360.0;
        let color = ViridisRGB.get_color(norm_val as f64);
        Rectangle::new([(f, t), (f + 1, t + 1)], color.filled())
    }))?;

    root.present()?;
    compress_png(output_path);
    Ok(())
}

pub fn plot_dynamic_spectrum_lag(
    output_path: &str,
    lag_data: &Array2<f32>,
    header: &crate::header::CorHeader,
    obs_time: &DateTime<Utc>,
    _length: i32,
    _effective_integration_time: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let (time_samples, lag_samples) = lag_data.dim();

    let width = 1200;
    let height = 600;
    let root = BitMapBackend::new(output_path, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let lag_range_min = -(lag_samples as i32 / 2);
    let lag_range_max = lag_samples as i32 / 2;
    let lag_range = lag_range_min..lag_range_max;
    let time_range = 0..time_samples;

    let max_val = lag_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = lag_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "Dynamic Spectrum (Time Lag) - {} - {}",
                header.source_name,
                obs_time.format("%Y-%m-%d %H:%M:%S")
            ),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(55)
        .y_label_area_size(100)
        .build_cartesian_2d(lag_range.clone(), time_range.clone())?;

    chart
        .configure_mesh()
        .y_desc("Time [PP]")
        .x_desc("Lag [samples]")
        .label_style(("sans-serif", 25).into_font())
        .draw()?;

    chart.draw_series(lag_data.indexed_iter().map(|((t, l), &val)| {
        let x = lag_range_min + l as i32;
        let norm_val = if max_val > min_val {
            (val - min_val) / (max_val - min_val)
        } else {
            0.0
        };
        let color = ViridisRGB.get_color(norm_val as f64);
        Rectangle::new([(x, t), (x + 1, t + 1)], color.filled())
    }))?;

    root.present()?;
    compress_png(output_path);
    Ok(())
}

fn gaussian_blur_2d(data: &Vec<Vec<f32>>, sigma: f32) -> Vec<Vec<f32>> {
    if sigma <= 0.0 {
        return data.clone();
    }

    let rows = data.len();
    if rows == 0 {
        return Vec::new();
    }
    let cols = data[0].len();
    if cols == 0 {
        return data.clone();
    }

    let kernel_radius = (sigma * 3.0).ceil() as usize;
    let kernel_size = 2 * kernel_radius + 1;
    let mut kernel = vec![0.0; kernel_size];
    let mut kernel_sum = 0.0;

    for i in 0..kernel_size {
        let x = i as f32 - kernel_radius as f32;
        kernel[i] = (-0.5 * (x / sigma).powi(2)).exp();
        kernel_sum += kernel[i];
    }

    for i in 0..kernel_size {
        kernel[i] /= kernel_sum;
    }

    let mut blurred_data = vec![vec![0.0; cols]; rows];
    let mut temp_data = vec![vec![0.0; cols]; rows];

    // Apply horizontal blur
    for r in 0..rows {
        for c in 0..cols {
            let mut sum = 0.0;
            for k_idx in 0..kernel_size {
                let col_idx = (c as isize + k_idx as isize - kernel_radius as isize)
                    .clamp(0, cols as isize - 1) as usize;
                sum += data[r][col_idx] * kernel[k_idx];
            }
            temp_data[r][c] = sum;
        }
    }

    // Apply vertical blur
    for r in 0..rows {
        for c in 0..cols {
            let mut sum = 0.0;
            for k_idx in 0..kernel_size {
                let row_idx = (r as isize + k_idx as isize - kernel_radius as isize)
                    .clamp(0, rows as isize - 1) as usize;
                sum += temp_data[row_idx][c] * kernel[k_idx];
            }
            blurred_data[r][c] = sum;
        }
    }

    blurred_data
}

fn draw_heatmap_with_colorbar(
    area: &DrawingArea<BitMapBackend, Shift>,
    data: &Vec<Vec<f32>>,
    x_desc: &str,
    y_desc: &str,
    color_bar_title: &str,
    min_val: f32,
    max_val: f32,
    num_color_bar_labels: usize,
    color_value_normalizer: impl Fn(f32) -> f64,
    color_bar_label_formatter: impl Fn(f32) -> String,
) -> Result<(), Box<dyn std::error::Error>> {
    let (rows, cols) = (data.len(), data[0].len());

    let (area_width, _area_height) = area.dim_in_pixel();
    let color_bar_area_width: u32 = 150;
    let chart_width = area_width.saturating_sub(color_bar_area_width);

    area.fill(&WHITE)?;

    let (chart_area, color_bar_area) = area.split_horizontally(chart_width);

    let top_margin = 10;
    let bottom_margin = 10;
    let x_label_area_size = 35;
    let y_label_area_size = 45;

    let mut chart = ChartBuilder::on(&chart_area)
        //.caption(title, ("sans-serif", 20).into_font())
        .margin_top(top_margin)
        .margin_bottom(bottom_margin)
        .margin_left(10)
        .margin_right(10)
        .x_label_area_size(x_label_area_size)
        .y_label_area_size(y_label_area_size)
        .build_cartesian_2d(0..cols, 0..rows)?;

    chart
        .configure_mesh()
        .x_desc(x_desc)
        .y_desc(y_desc)
        .x_label_style(("sans-serif", 18).into_font())
        .y_label_style(("sans-serif", 18).into_font())
        .x_label_formatter(&|v| format!("{v}"))
        .y_label_formatter(&|v| format!("{v}"))
        .draw()?;

    chart.draw_series(
        (0..cols)
            .flat_map(|x| (0..rows).map(move |y| (x, y)))
            .map(|(x, y)| {
                let val = data[y][x];
                let color_value = color_value_normalizer(val);
                let color = ViridisRGB.get_color(color_value);
                Rectangle::new([(x, y), (x + 1, y + 1)], color.filled())
            }),
    )?;

    // Draw color bar aligned with plotting area height
    if num_color_bar_labels == 0 {
        return Ok(());
    }
    let color_bar_left_pad: u32 = 12;
    let color_bar_width: u32 = 30;
    let axis_bottom_pad = bottom_margin + x_label_area_size;

    let (_, remainder_area) = color_bar_area.split_horizontally(color_bar_left_pad);
    let (bar_strip_raw, label_strip_raw) = if remainder_area.dim_in_pixel().0 > color_bar_width {
        remainder_area.split_horizontally(color_bar_width)
    } else {
        let width = remainder_area.dim_in_pixel().0;
        remainder_area.split_horizontally(width)
    };
    let bar_strip = bar_strip_raw.margin(top_margin as u32, axis_bottom_pad as u32, 0, 0);
    let label_strip = label_strip_raw.margin(top_margin as u32, axis_bottom_pad as u32, 6, 0);

    let (bar_strip_width, bar_strip_height) = bar_strip.dim_in_pixel();
    if bar_strip_height == 0 {
        return Ok(());
    }
    let height_norm = (bar_strip_height.saturating_sub(1)).max(1) as f64;

    for i in 0..bar_strip_height as i32 {
        let frac = 1.0 - (i as f64 / height_norm);
        let color = ViridisRGB.get_color(frac);
        bar_strip.draw(&Rectangle::new(
            [(0, i), (bar_strip_width as i32, i + 1)],
            color.filled(),
        ))?;
    }

    let step = std::cmp::max(1, bar_strip_height / 80);
    let mut label_count = std::cmp::max(5usize, step as usize);
    if label_count == 1 {
        label_count = 2;
    }
    for i in 0..label_count {
        let frac = if label_count <= 1 {
            0.0
        } else {
            i as f64 / (label_count - 1) as f64
        };
        let value = min_val + (max_val - min_val) * (1.0 - frac as f32);
        let y_pos = ((1.0 - frac) * height_norm).round() as i32;
        label_strip.draw_text(
            &color_bar_label_formatter(value),
            &TextStyle::from(("sans-serif", 18).into_font()).color(&BLACK),
            (4, y_pos - 9),
        )?;
    }

    let (label_strip_width, _) = label_strip.dim_in_pixel();
    let title_x = (label_strip_width as i32 - 34).max(24);
    label_strip.draw_text(
        color_bar_title,
        &TextStyle::from(("sans-serif", 18).into_font())
            .color(&BLACK)
            .transform(FontTransform::Rotate270),
        (title_x, (bar_strip_height / 2) as i32),
    )?;
    Ok(())
}

pub fn plot_spectrum_heatmaps<P: AsRef<Path>>(
    output_path: P,
    spectrum_data: &Vec<Vec<Complex<f32>>>,
    sigma: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if spectrum_data.is_empty() || spectrum_data[0].is_empty() {
        return Err("Spectrum data for heatmap is empty".into());
    }

    let panel_width = 610u32;
    let total_width = panel_width * 2;
    let total_height = 384u32;
    let root =
        BitMapBackend::new(output_path.as_ref(), (total_width, total_height)).into_drawing_area();
    root.fill(&WHITE)?;
    let (left_area, right_area) = root.split_horizontally(panel_width);

    // --- Amplitude Heatmap ---
    let amplitudes_2d: Vec<Vec<f32>> = spectrum_data
        .iter()
        .map(|row| row.iter().map(|c| c.norm()).collect())
        .collect();
    let blurred_amplitudes = gaussian_blur_2d(&amplitudes_2d, sigma);
    let max_amp = blurred_amplitudes
        .iter()
        .flatten()
        .cloned()
        .fold(0.0, f32::max);
    let min_amp = blurred_amplitudes
        .iter()
        .flatten()
        .cloned()
        .fold(f32::MAX, f32::min);

    draw_heatmap_with_colorbar(
        &left_area,
        &blurred_amplitudes,
        "Channels",
        "PP",
        "Amplitude (a.u.)",
        min_amp,
        max_amp,
        5,
        |v| {
            if max_amp > min_amp {
                ((v - min_amp) / (max_amp - min_amp)) as f64
            } else {
                0.0
            }
        },
        |v| format!("{:.1e}", v),
    )?;

    // --- Phase Heatmap ---
    let phases_2d: Vec<Vec<f32>> = spectrum_data
        .iter()
        .map(|row| row.iter().map(|c| safe_arg(c).to_degrees()).collect())
        .collect();
    let blurred_phases = gaussian_blur_2d(&phases_2d, sigma);

    draw_heatmap_with_colorbar(
        &right_area,
        &blurred_phases,
        "Channels",
        "PP",
        "Phase (deg)",
        -180.0,
        180.0,
        9,
        |v| ((v + 180.0) / 360.0) as f64,
        |v| format!("{:.0}", v),
    )?;

    root.present()?;
    compress_png(output_path.as_ref());
    Ok(())
}

pub fn plot_spectrum_amplitude_heatmap<P: AsRef<Path>>(
    output_path: P,
    spectrum_data: &Vec<Vec<Complex<f32>>>,
    sigma: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if spectrum_data.is_empty() || spectrum_data[0].is_empty() {
        return Err("Spectrum data for heatmap is empty".into());
    }

    let root = BitMapBackend::new(output_path.as_ref(), (915, 576)).into_drawing_area();
    root.fill(&WHITE)?;

    let amplitudes_2d: Vec<Vec<f32>> = spectrum_data
        .iter()
        .map(|row| row.iter().map(|c| c.norm()).collect())
        .collect();
    let blurred_amplitudes = gaussian_blur_2d(&amplitudes_2d, sigma);
    let max_amp = blurred_amplitudes
        .iter()
        .flatten()
        .cloned()
        .fold(0.0, f32::max);
    let min_amp = blurred_amplitudes
        .iter()
        .flatten()
        .cloned()
        .fold(f32::MAX, f32::min);

    draw_heatmap_with_colorbar(
        &root,
        &blurred_amplitudes,
        "Channels",
        "PP",
        "Amplitude (a.u.)",
        min_amp,
        max_amp,
        5,
        |v| {
            if max_amp > min_amp {
                ((v - min_amp) / (max_amp - min_amp)) as f64
            } else {
                0.0
            }
        },
        |v| format!("{:.1e}", v),
    )?;

    root.present()?;
    compress_png(output_path.as_ref());
    Ok(())
}

pub fn plot_spectrum_phase_heatmap<P: AsRef<Path>>(
    output_path: P,
    spectrum_data: &Vec<Vec<Complex<f32>>>,
    sigma: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if spectrum_data.is_empty() || spectrum_data[0].is_empty() {
        return Err("Spectrum data for heatmap is empty".into());
    }

    let root = BitMapBackend::new(output_path.as_ref(), (915, 576)).into_drawing_area();
    root.fill(&WHITE)?;

    let phases_2d: Vec<Vec<f32>> = spectrum_data
        .iter()
        .map(|row| row.iter().map(|c| safe_arg(c).to_degrees()).collect())
        .collect();
    let blurred_phases = gaussian_blur_2d(&phases_2d, sigma);

    draw_heatmap_with_colorbar(
        &root,
        &blurred_phases,
        "Channels",
        "PP",
        "Phase (deg)",
        -180.0,
        180.0,
        9,
        |v| ((v + 180.0) / 360.0) as f64,
        |v| format!("{:.0}", v),
    )?;

    root.present()?;
    compress_png(output_path.as_ref());
    Ok(())
}

pub fn plot_complex_scatter<P: AsRef<Path>>(
    output_path: P,
    real_values: &[f32],
    imag_values: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    if real_values.is_empty() || real_values.len() != imag_values.len() {
        return Err("散布図を描画するための複素データが不足しています".into());
    }

    let mut min_real = real_values.iter().fold(f32::INFINITY, |acc, &v| acc.min(v));
    let mut max_real = real_values
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
    let mut min_imag = imag_values.iter().fold(f32::INFINITY, |acc, &v| acc.min(v));
    let mut max_imag = imag_values
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));

    min_real = min_real.min(0.0);
    max_real = max_real.max(0.0);
    min_imag = min_imag.min(0.0);
    max_imag = max_imag.max(0.0);

    let real_margin = ((max_real - min_real) * 0.05).max(1e-6);
    let imag_margin = ((max_imag - min_imag) * 0.05).max(1e-6);

    let root = BitMapBackend::new(output_path.as_ref(), (1000, 1000)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(25)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (min_real - real_margin) as f64..(max_real + real_margin) as f64,
            (min_imag - imag_margin) as f64..(max_imag + imag_margin) as f64,
        )?;

    chart
        .configure_mesh()
        .x_desc("Real")
        .y_desc("Imag")
        .label_style(("sans-serif", 30))
        .x_label_formatter(&|v| format!("{:.2e}", v))
        .y_label_formatter(&|v| format!("{:.2e}", v))
        .draw()?;

    let total_points = real_values.len();
    let max_points = 200_000;
    let stride = (total_points / max_points).max(1);

    chart.draw_series(
        real_values
            .iter()
            .zip(imag_values.iter())
            .enumerate()
            .filter_map(|(idx, (&re, &im))| {
                if idx % stride == 0 {
                    Some(Circle::new(
                        (re as f64, im as f64),
                        1,
                        BLUE.mix(0.4).filled(),
                    ))
                } else {
                    None
                }
            }),
    )?;

    chart.draw_series(std::iter::once(Circle::new((0.0, 0.0), 6, RED.filled())))?;

    root.present()?;
    compress_png(output_path.as_ref());
    Ok(())
}

pub fn plot_amp_phase_scatter<P: AsRef<Path>>(
    output_path: P,
    amp_values: &[f32],
    phase_values_rad: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    if amp_values.is_empty() || amp_values.len() != phase_values_rad.len() {
        return Err("Amp-Phase 散布図を描画するためのデータが不足しています".into());
    }

    let phase_deg: Vec<f64> = phase_values_rad
        .iter()
        .map(|&phi| wrap_degrees((phi as f64).to_degrees()))
        .collect();

    let min_amp = amp_values.iter().fold(f32::INFINITY, |acc, &v| acc.min(v));
    let max_amp = amp_values
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));

    let amp_margin = ((max_amp - min_amp) * 0.05).max(1e-6);

    let root = BitMapBackend::new(output_path.as_ref(), (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(25)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (min_amp - amp_margin) as f64..(max_amp + amp_margin) as f64,
            -180.0f64..180.0f64,
        )?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("Amplitude")
        .y_desc("Phase (deg)")
        .label_style(("sans-serif", 30))
        .x_label_formatter(&|v| format!("{:.2e}", v))
        .y_label_formatter(&|v| format!("{:.0}", v))
        .draw()?;

    let total_points = amp_values.len();
    let max_points = 200_000;
    let stride = (total_points / max_points).max(1);

    chart.draw_series(
        amp_values
            .iter()
            .zip(phase_deg.iter())
            .enumerate()
            .filter_map(|(idx, (&amp, &phase))| {
                if idx % stride == 0 {
                    Some(Circle::new((amp as f64, phase), 1, BLUE.mix(0.4).filled()))
                } else {
                    None
                }
            }),
    )?;

    root.present()?;
    compress_png(output_path.as_ref());
    Ok(())
}

pub fn plot_complex_histograms<P: AsRef<Path>, Q: AsRef<Path>>(
    output_path: P,
    log_output_path: Q,
    real_values: &[f32],
    imag_values: &[f32],
    amp_values: &[f32],
    phase_values: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    if real_values.is_empty()
        || imag_values.is_empty()
        || amp_values.is_empty()
        || real_values.len() != imag_values.len()
        || real_values.len() != amp_values.len()
        || phase_values.len() != real_values.len()
        || phase_values.is_empty()
    {
        return Err("ヒストグラムを描画するための複素データが不足しています".into());
    }

    let log_path = log_output_path.as_ref();

    let root = BitMapBackend::new(output_path.as_ref(), (1600, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let (upper, lower) = root.split_vertically(450);
    let (left, right) = upper.split_horizontally(800);
    let (lower_left, lower_right) = lower.split_horizontally(800);

    let mut reports = Vec::new();

    reports.push(plot_histogram_panel(&left, "Real", real_values)?);
    reports.push(plot_histogram_panel(&right, "Imag", imag_values)?);
    reports.push(plot_histogram_panel(&lower_left, "Amplitude", amp_values)?);
    reports.push(plot_phase_histogram_panel(&lower_right, phase_values)?);

    write_histogram_report(&reports, log_path)?;

    Ok(())
}

fn plot_histogram_panel(
    area: &DrawingArea<BitMapBackend, Shift>,
    label: &str,
    values: &[f32],
) -> Result<HistogramReport, Box<dyn std::error::Error>> {
    let values_f64: Vec<f64> = values.iter().map(|&v| v as f64).collect();
    let mut min_val = values_f64.iter().fold(f64::INFINITY, |acc, &v| acc.min(v));
    let mut max_val = values_f64
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &v| acc.max(v));

    if (max_val - min_val).abs() < f64::EPSILON {
        let center = values_f64.first().copied().unwrap_or(0.0);
        min_val = center - 1.0;
        max_val = center + 1.0;
    }

    let bin_count = select_histogram_bins(&values_f64, min_val, max_val);
    let (hist_data, bin_width) = compute_histogram(&values_f64, bin_count, min_val, max_val);
    let max_count = hist_data
        .iter()
        .fold(0.0f64, |acc, (_, count)| acc.max(*count));

    let mut chart = ChartBuilder::on(area)
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(min_val..max_val, 0.0..max_count * 1.2)?;

    chart
        .configure_mesh()
        .x_desc(label)
        .y_desc("Counts")
        .disable_mesh()
        .label_style(("sans-serif", 30))
        .x_label_formatter(&|v| format!("{:.2e}", v))
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .draw()?;

    chart.draw_series(hist_data.iter().map(|(center, count)| {
        let half_bin = bin_width / 2.0;
        let left = center - half_bin;
        let right = center + half_bin;
        Rectangle::new([(left, 0.0), (right, *count)], BLUE.mix(0.4).filled())
    }))?;

    let n = values_f64.len() as f64;
    let mean = values_f64.iter().sum::<f64>() / n;
    let variance = values_f64.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let sigma = variance.sqrt();

    Ok(HistogramReport::from_lines(
        label,
        vec![
            format!("Samples: {}", values_f64.len()),
            format!("Mean: {:.6e}", mean),
            format!("Sigma: {:.6e}", sigma),
            format!("Min: {:.6e}", min_val),
            format!("Max: {:.6e}", max_val),
        ],
    ))
}

fn plot_phase_histogram_panel(
    area: &DrawingArea<BitMapBackend, Shift>,
    phase_values: &[f32],
) -> Result<HistogramReport, Box<dyn std::error::Error>> {
    let values_rad: Vec<f64> = phase_values.iter().map(|&v| v as f64).collect();
    let values_deg: Vec<f64> = values_rad
        .iter()
        .map(|rad| wrap_degrees(rad.to_degrees()))
        .collect();
    let min_val = -180.0;
    let max_val = 180.0;
    let bin_count = 180usize;
    let (hist_data, bin_width) = compute_histogram(&values_deg, bin_count, min_val, max_val);
    let max_count = hist_data
        .iter()
        .fold(0.0f64, |acc, (_, count)| acc.max(*count));

    let mut chart = ChartBuilder::on(area)
        .margin(20)
        .x_label_area_size(65)
        .y_label_area_size(80)
        .build_cartesian_2d(min_val..max_val, 0.0..max_count * 1.2)?;

    chart
        .configure_mesh()
        .x_desc("Phase (deg)")
        .y_desc("Counts")
        .disable_mesh()
        .label_style(("sans-serif", 30))
        .x_labels(7)
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .draw()?;

    chart.draw_series(hist_data.iter().map(|(center_deg, count)| {
        let half_bin = bin_width / 2.0;
        let left = center_deg - half_bin;
        let right = center_deg + half_bin;
        Rectangle::new([(left, 0.0), (right, *count)], BLUE.mix(0.4).filled())
    }))?;

    let total_count = values_deg.len() as f64;
    let expected_uniform = if bin_count > 0 {
        total_count / bin_count as f64
    } else {
        0.0
    };
    chart
        .draw_series(LineSeries::new(
            vec![(min_val, expected_uniform), (max_val, expected_uniform)],
            GREEN.stroke_width(3),
        ))?
        .label(format!("Uniform N/bin={:.1e}", expected_uniform))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], &GREEN));

    let circular_mean = compute_circular_mean(&values_rad);
    let circular_sigma = compute_circular_sigma(&values_rad);

    Ok(HistogramReport::from_lines(
        "Phase",
        vec![
            format!("Samples: {}", values_rad.len()),
            format!(
                "Circular mean (deg): {:.2}",
                wrap_degrees(circular_mean.to_degrees())
            ),
            format!("Circular sigma (deg): {:.2}", circular_sigma.to_degrees()),
        ],
    ))
}

pub fn plot_cross_section(
    output_path: &str,
    horizontal_data: &[(f64, f32)], // (RA offset, Intensity)
    vertical_data: &[(f64, f32)],   // (Dec offset, Intensity)
    max_intensity: f32,
    delta_ra_mas: f64,
    delta_dec_mas: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1000, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let (h_min, h_max) = horizontal_data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (x, _)| {
            (min.min(*x), max.max(*x))
        });
    let (v_min, v_max) = vertical_data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (x, _)| {
            (min.min(*x), max.max(*x))
        });
    let x_min = h_min.min(v_min);
    let x_max = h_max.max(v_max);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Fringe Rate Map Cross-section at Peak",
            ("sans-serif", 25).into_font(),
        )
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(x_max..x_min, 0.0..1.1f64)?;

    chart
        .configure_mesh()
        .x_desc("Offset (mas)")
        .y_desc("Normalized Intensity")
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .label_style(("sans-serif", 30).into_font())
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            horizontal_data
                .iter()
                .map(|(x, y)| (*x, (*y / max_intensity) as f64)),
            &BLUE,
        ))?
        .label("RA Cross-section")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));

    chart
        .draw_series(LineSeries::new(
            vertical_data
                .iter()
                .map(|(x, y)| (*x, (*y / max_intensity) as f64)),
            &RED,
        ))?
        .label("Dec Cross-section")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    let font = ("sans-serif", 35).into_font();
    let text_style = TextStyle::from(font.clone()).color(&BLACK);
    let x_pos = root.get_pixel_range().0.end as i32 - 300;
    let mut y_pos = root.get_pixel_range().1.start as i32 + 20;

    root.draw(&Text::new(
        format!("ΔRA: {:.3} mas", delta_ra_mas),
        (x_pos, y_pos),
        text_style.clone(),
    ))?;
    y_pos += 25;
    root.draw(&Text::new(
        format!("ΔDec: {:.3} mas", delta_dec_mas),
        (x_pos, y_pos),
        text_style.clone(),
    ))?;

    root.present()?;
    compress_png(output_path);
    Ok(())
}

pub fn plot_uv_coverage<P: AsRef<Path>>(
    output_path: P,
    uv_data: &[(f32, f32)],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path.as_ref(), (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    if uv_data.is_empty() {
        return Ok(());
    }

    let u_max = uv_data.iter().map(|(u, _)| u.abs()).fold(0.0, f32::max);
    let v_max = uv_data.iter().map(|(_, v)| v.abs()).fold(0.0, f32::max);
    let max_abs = u_max.max(v_max) * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("UV Coverage", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(-max_abs..max_abs, -max_abs..max_abs)?;

    chart
        .configure_mesh()
        .x_desc("U (meters)")
        .y_desc("V (meters)")
        .x_label_formatter(&|x| format!("{:.0}", x))
        .y_label_formatter(&|y| format!("{:.0}", y))
        .label_style(("sans-serif", 25))
        .draw()?;

    chart.draw_series(
        uv_data
            .iter()
            .map(|(u, v)| Circle::new((*u, *v), 2, BLUE.filled())),
    )?;
    chart.draw_series(
        uv_data
            .iter()
            .map(|(u, v)| Circle::new((-*u, -*v), 2, RED.filled())),
    )?;

    root.present()?;
    Ok(())
}

pub fn plot_uv_tracks<P: AsRef<Path>>(
    output_path: P,
    accessible: &[(f32, f32)],
    observed: &[(f32, f32)],
    accessible_baseline: &[(f32, f32)],
    observed_baseline: &[(f32, f32)],
    center_frequency_hz: f64,
    uv_mode: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path.as_ref(), (1500, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let (left_area, right_area) = root.split_horizontally(700);

    let max_uv_value = accessible
        .iter()
        .chain(observed.iter())
        .flat_map(|(u, v)| [u.abs(), v.abs()])
        .fold(0.0f32, f32::max)
        .max(1.0);

    let (uv_scale, uv_unit) = if max_uv_value >= 1_000_000.0 {
        (1_000_000.0, "Mm")
    } else if max_uv_value >= 1_000.0 {
        (1_000.0, "km")
    } else {
        (1.0, "m")
    };

    let axis_limit = (max_uv_value / uv_scale) * 1.1;

    let scaled_accessible: Vec<(f32, f32)> = accessible
        .iter()
        .map(|(u, v)| (*u / uv_scale, *v / uv_scale))
        .collect();
    let scaled_observed: Vec<(f32, f32)> = observed
        .iter()
        .map(|(u, v)| (*u / uv_scale, *v / uv_scale))
        .collect();

    let mut uv_chart = ChartBuilder::on(&left_area)
        .caption("UV Coverage", ("sans-serif", 30))
        .margin(15)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(-axis_limit..axis_limit, -axis_limit..axis_limit)?;

    uv_chart
        .configure_mesh()
        .x_desc(&format!("U ({})", uv_unit))
        .y_desc(&format!("V ({})", uv_unit))
        .x_label_formatter(&|x| format!("{:.0}", x))
        .y_label_formatter(&|y| format!("{:.0}", y))
        .label_style(("sans-serif", 24))
        .light_line_style(&WHITE)
        .draw()?;

    let accessible_style = ShapeStyle::from(&BLACK).stroke_width(2);
    if scaled_accessible.len() >= 2 {
        for (idx, segment) in scaled_accessible.windows(2).enumerate() {
            if idx % 2 == 0 {
                uv_chart.draw_series(std::iter::once(PathElement::new(
                    vec![segment[0], segment[1]],
                    accessible_style,
                )))?;
            }
        }

        let mirrored: Vec<(f32, f32)> = scaled_accessible
            .iter()
            .rev()
            .map(|(u, v)| (-*u, -*v))
            .collect();
        for (idx, segment) in mirrored.windows(2).enumerate() {
            if idx % 2 == 0 {
                uv_chart.draw_series(std::iter::once(PathElement::new(
                    vec![segment[0], segment[1]],
                    accessible_style,
                )))?;
            }
        }
    }

    if scaled_observed.len() >= 2 {
        let observed_style = ShapeStyle::from(&RED).stroke_width(2);
        uv_chart.draw_series(LineSeries::new(
            scaled_observed.iter().cloned(),
            observed_style,
        ))?;
        let mirrored: Vec<(f32, f32)> = scaled_observed
            .iter()
            .rev()
            .map(|(u, v)| (-*u, -*v))
            .collect();
        if mirrored.len() >= 2 {
            uv_chart.draw_series(LineSeries::new(mirrored, observed_style))?;
        }
    }

    if !scaled_accessible.is_empty() {
        let accessible_points: Vec<(f32, f32)> = scaled_accessible
            .iter()
            .flat_map(|(u, v)| [(*u, *v), (-*u, -*v)])
            .collect();
        uv_chart
            .draw_series(PointSeries::of_element(
                accessible_points.into_iter(),
                4,
                ShapeStyle::from(&BLACK),
                &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
            ))?
            .label(format!("EL ≥ 5° (uv_3d={})", uv_mode))
            .legend(|(x, y)| Circle::new((x, y), 5, ShapeStyle::from(&BLACK)));
    }

    if !scaled_observed.is_empty() {
        let observed_points: Vec<(f32, f32)> = scaled_observed
            .iter()
            .flat_map(|(u, v)| [(*u, *v), (-*u, -*v)])
            .collect();
        let red_style = ShapeStyle::from(&RED).filled();
        uv_chart
            .draw_series(PointSeries::of_element(
                observed_points.into_iter(),
                4,
                red_style,
                &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
            ))?
            .label(format!("Observation (uv_3d={})", uv_mode))
            .legend(|(x, y)| Circle::new((x, y), 5, ShapeStyle::from(&RED).filled()));
    }

    uv_chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()?;

    let max_baseline = accessible_baseline
        .iter()
        .chain(observed_baseline.iter())
        .map(|(_, b)| *b)
        .fold(0.0f32, f32::max)
        .max(1.0);

    let lambda = 299_792_458.0 / center_frequency_hz;
    let max_spatial = (max_baseline as f64 / lambda).max(1e-6) as f32;

    let baseline_scale = if max_baseline >= 1_000_000.0 {
        1_000_000.0
    } else if max_baseline >= 1_000.0 {
        1_000.0
    } else {
        1.0
    };
    let baseline_unit = if baseline_scale == 1_000_000.0 {
        "Mm"
    } else if baseline_scale == 1_000.0 {
        "km"
    } else {
        "m"
    };
    let baseline_label = format!("Projected Baseline ({})", baseline_unit);

    let spatial_scale = if max_spatial >= 1_000_000.0 {
        1_000_000.0
    } else if max_spatial >= 1_000.0 {
        1_000.0
    } else {
        1.0
    };
    let spatial_unit = if spatial_scale == 1_000_000.0 {
        "Mλ"
    } else if spatial_scale == 1_000.0 {
        "kλ"
    } else {
        "λ"
    };
    let spatial_label = format!("Spatial Frequency ({})", spatial_unit);

    let baseline_upper = (max_baseline / baseline_scale) * 1.05;
    let spatial_upper = (max_spatial / spatial_scale) * 1.05;

    let scaled_accessible_baseline: Vec<(f32, f32)> = accessible_baseline
        .iter()
        .map(|(t, b)| (*t, *b / baseline_scale))
        .collect();
    let scaled_observed_baseline: Vec<(f32, f32)> = observed_baseline
        .iter()
        .map(|(t, b)| (*t, *b / baseline_scale))
        .collect();

    let mut baseline_chart = ChartBuilder::on(&right_area)
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(80)
        .right_y_label_area_size(80)
        .build_cartesian_2d(0f32..24f32, 0f32..baseline_upper)?
        .set_secondary_coord(0f32..24f32, 0f32..spatial_upper);

    baseline_chart
        .configure_mesh()
        .x_desc("UT (hour)")
        .y_desc(baseline_label.clone())
        .x_label_formatter(&|x| format!("{:.0}", x))
        .x_labels(25)
        .y_label_formatter(&|y| format!("{:.0}", y))
        .light_line_style(&WHITE)
        .label_style(("sans-serif", 22))
        .draw()?;

    baseline_chart
        .configure_secondary_axes()
        .y_desc(spatial_label)
        .y_label_formatter(&|y| format!("{:.0}", y))
        .draw()?;

    if scaled_accessible_baseline.len() >= 2 {
        let accessible_line_style = ShapeStyle::from(&BLACK).stroke_width(2);
        for (idx, segment) in scaled_accessible_baseline.windows(2).enumerate() {
            if idx % 2 == 0 {
                baseline_chart.draw_series(std::iter::once(PathElement::new(
                    vec![segment[0], segment[1]],
                    accessible_line_style,
                )))?;
            }
        }
    }
    if scaled_observed_baseline.len() >= 2 {
        let observed_line_style = ShapeStyle::from(&RED).stroke_width(2);
        baseline_chart.draw_series(LineSeries::new(
            scaled_observed_baseline.iter().cloned(),
            observed_line_style,
        ))?;
    }

    if !scaled_accessible_baseline.is_empty() {
        baseline_chart
            .draw_series(PointSeries::of_element(
                scaled_accessible_baseline.iter().cloned(),
                4,
                ShapeStyle::from(&BLACK),
                &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
            ))?
            .label(format!("EL ≥ 5° (uv_3d={})", uv_mode))
            .legend(|(x, y)| Circle::new((x, y), 5, ShapeStyle::from(&BLACK)));
    }

    if !scaled_observed_baseline.is_empty() {
        let filled_style = ShapeStyle::from(&RED).filled();
        baseline_chart
            .draw_series(PointSeries::of_element(
                scaled_observed_baseline.iter().cloned(),
                4,
                filled_style,
                &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
            ))?
            .label(format!("Observation (uv_3d={})", uv_mode))
            .legend(|(x, y)| Circle::new((x, y), 5, ShapeStyle::from(&RED).filled()));
    }

    baseline_chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()?;

    root.present()?;
    Ok(())
}

fn wrap_degrees(mut deg: f64) -> f64 {
    while deg <= -180.0 {
        deg += 360.0;
    }
    while deg > 180.0 {
        deg -= 360.0;
    }
    deg
}

fn compute_circular_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum_sin: f64 = values.iter().map(|v| v.sin()).sum();
    let sum_cos: f64 = values.iter().map(|v| v.cos()).sum();
    sum_sin.atan2(sum_cos)
}

fn compute_circular_sigma(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum_sin: f64 = values.iter().map(|v| v.sin()).sum();
    let sum_cos: f64 = values.iter().map(|v| v.cos()).sum();
    let r = (sum_sin.powi(2) + sum_cos.powi(2)).sqrt() / values.len() as f64;
    if r < 1e-6 {
        PI / 2.0
    } else {
        (-2.0 * r.ln()).max(1e-8).sqrt()
    }
}

fn select_histogram_bins(values: &[f64], min_val: f64, max_val: f64) -> usize {
    let n = values.len();
    if n <= 1 {
        return 32;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q1 = percentile(&sorted, 0.25);
    let q3 = percentile(&sorted, 0.75);
    let iqr = (q3 - q1).abs();

    let n_f64 = n as f64;
    let mut bin_width = if iqr > 0.0 {
        2.0 * iqr / n_f64.powf(1.0 / 3.0)
    } else {
        let (_, sigma) = compute_gaussian_params(values);
        if sigma > 0.0 {
            3.49 * sigma / n_f64.powf(1.0 / 3.0)
        } else {
            (max_val - min_val).abs() / n_f64.max(1.0)
        }
    };

    if !bin_width.is_finite() || bin_width <= 0.0 {
        bin_width = (max_val - min_val).abs() / n_f64.max(1.0);
    }

    let range = (max_val - min_val).abs();
    let mut bin_count = if bin_width > 0.0 {
        (range / bin_width).ceil() as usize
    } else {
        32
    };

    bin_count = bin_count.clamp(32, 512);
    if bin_count == 0 {
        bin_count = 32;
    }

    bin_count
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let clamped_p = p.clamp(0.0, 1.0);
    let pos = clamped_p * (sorted.len() - 1) as f64;
    let lower_idx = pos.floor() as usize;
    let upper_idx = pos.ceil() as usize;
    if lower_idx == upper_idx {
        return sorted[lower_idx];
    }
    let weight = pos - lower_idx as f64;
    sorted[lower_idx] * (1.0 - weight) + sorted[upper_idx] * weight
}

fn compute_histogram(
    values: &[f64],
    bin_count: usize,
    min_val: f64,
    max_val: f64,
) -> (Vec<(f64, f64)>, f64) {
    let mut counts = vec![0usize; bin_count];
    let mut centers = vec![0f64; bin_count];
    let range = max_val - min_val;
    let bin_width = if bin_count > 0 {
        let width = range / bin_count as f64;
        if width.is_finite() && width > 0.0 {
            width
        } else {
            1.0
        }
    } else {
        1.0
    };

    for (idx, count) in centers.iter_mut().enumerate() {
        *count = min_val + (idx as f64 + 0.5) * bin_width;
    }

    for &value in values {
        let mut bin_idx = ((value - min_val) / bin_width) as isize;
        if bin_idx < 0 {
            bin_idx = 0;
        } else if bin_idx as usize >= bin_count {
            bin_idx = bin_count as isize - 1;
        }
        counts[bin_idx as usize] += 1;
    }

    let hist_data = counts
        .into_iter()
        .zip(centers.into_iter())
        .map(|(count, center)| (center, count as f64))
        .collect();

    (hist_data, bin_width)
}

fn compute_gaussian_params(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    if n <= 1.0 {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, variance.sqrt())
}

#[derive(Clone, Debug)]
struct HistogramReport {
    title: String,
    lines: Vec<String>,
}

impl HistogramReport {
    fn from_lines(title: &str, mut lines: Vec<String>) -> Self {
        if lines.is_empty() {
            lines.push("(no data)".to_string());
        }
        Self {
            title: title.to_string(),
            lines,
        }
    }
}

fn write_histogram_report(
    reports: &[HistogramReport],
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut aggregated = String::new();
    for report in reports {
        let header = format!("== {} ==", report.title);
        println!("{}", header);
        aggregated.push_str(&header);
        aggregated.push('\n');
        for line in &report.lines {
            println!("{}", line);
            aggregated.push_str(line);
            aggregated.push('\n');
        }
        println!();
        aggregated.push('\n');
    }
    let mut file = File::create(output_path)?;
    file.write_all(aggregated.as_bytes())?;
    Ok(())
}

// Multi-sideband (C/X) frequency-domain summary plot merged from former plot_msb.rs.
pub fn frequency_plane_msb(
    c_band_amp_profile: &[(f64, f64)],
    c_band_phase_profile: &[(f64, f64)],
    x_band_amp_profile: &[(f64, f64)],
    x_band_phase_profile: &[(f64, f64)],
    _x_band_uncalibrated_phase_profile: &[(f64, f64)],
    c_band_rate_profile: &[(f64, f64)], // New parameter for C-band rate
    x_band_rate_profile: &[(f64, f64)], // New parameter for X-band rate
    heatmap_func: impl Fn(f64, f64) -> f64,
    stat_keys: &[&str],
    stat_vals: &[&str],
    output_path: &str,
    bw: f64,
    max_amplitude: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let width = 1400;
    let height = 1000;
    let root = BitMapBackend::new(output_path, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let (left_area, right_area) = root.split_horizontally(width / 2);
    let (top_left_area, rate_area) = left_area.split_vertically(height / 2 + 50);
    let (phase_area, amp_area) =
        top_left_area.split_vertically(top_left_area.get_pixel_range().1.len() as u32 / 4 - 10);
    let (heatmap_and_colorbar_area, stats_area) = right_area.split_vertically(height / 2 + 50);
    let (heatmap_area, colorbar_area) = heatmap_and_colorbar_area
        .split_horizontally(heatmap_and_colorbar_area.get_pixel_range().0.len() as u32 - 120);

    // --- Determine axis ranges ---
    let c_amp_max_y = c_band_amp_profile
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    let x_amp_max_y = x_band_amp_profile
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    let amp_max_y = c_amp_max_y.max(x_amp_max_y) * 1.1;
    let c_band_rate_max_y = c_band_rate_profile
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    let x_band_rate_max_y = x_band_rate_profile
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    let rate_max_y = c_band_rate_max_y.max(x_band_rate_max_y) * 1.1;

    let c_band_rate_min_x = c_band_rate_profile
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::INFINITY, f64::min);
    let c_band_rate_max_x = c_band_rate_profile
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::NEG_INFINITY, f64::max);
    let x_band_rate_min_x = x_band_rate_profile
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::INFINITY, f64::min);
    let x_band_rate_max_x = x_band_rate_profile
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::NEG_INFINITY, f64::max);

    let rate_min_x = c_band_rate_min_x.min(x_band_rate_min_x);
    let rate_max_x = c_band_rate_max_x.max(x_band_rate_max_x);

    // --- Phase Chart ---
    let mut phase_chart = ChartBuilder::on(&phase_area)
        .margin_top(20)
        .margin_left(15)
        .x_label_area_size(0)
        .y_label_area_size(120)
        .build_cartesian_2d(0.0..bw, -180.0..180.0)?;
    phase_chart
        .configure_mesh()
        //.disable_x_mesh()
        //.x_labels(0)
        .x_labels(7)
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .axis_style(BLACK.stroke_width(1))
        .y_desc("Phase")
        .y_label_formatter(&|y| format!("{:.0}", y))
        .y_labels(6)
        .label_style(("sans-serif ", 30))
        .draw()?;
    phase_chart.draw_series(LineSeries::new(
        c_band_phase_profile.iter().cloned(),
        RED.stroke_width(1),
    ))?;
    phase_chart.draw_series(LineSeries::new(
        x_band_phase_profile.iter().cloned(),
        BLUE.stroke_width(1),
    ))?;
    //phase_chart.draw_series(LineSeries::new(x_band_uncalibrated_phase_profile.iter().cloned(), BLACK.stroke_width(1)))?;

    // Draw bounding box for phase_chart
    let x_spec = phase_chart.x_range();
    let y_spec = phase_chart.y_range();
    phase_chart.draw_series(std::iter::once(Rectangle::new(
        [(x_spec.start, y_spec.start), (x_spec.end, y_spec.end)],
        BLACK.stroke_width(1),
    )))?;

    // --- Amplitude Chart ---
    let mut amp_chart = ChartBuilder::on(&amp_area)
        .margin_left(15)
        .x_label_area_size(65)
        .y_label_area_size(120)
        .build_cartesian_2d(0.0..bw, 0.0..amp_max_y)?;
    amp_chart
        .configure_mesh()
        .x_desc("Frequency [MHz]")
        .y_desc("Amplitude")
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .x_labels(7)
        .y_labels(7)
        .axis_style(BLACK.stroke_width(1))
        .label_style(("sans-serif ", 30))
        .x_label_formatter(&|y| format!("{:.0}", y))
        .y_label_formatter(&|y| format!("{:.1e}", y))
        .draw()?;
    amp_chart.draw_series(LineSeries::new(
        c_band_amp_profile.iter().cloned(),
        RED.stroke_width(1),
    ))?;
    amp_chart.draw_series(LineSeries::new(
        x_band_amp_profile.iter().cloned(),
        BLUE.stroke_width(1),
    ))?;

    // Draw bounding box for amp_chart
    let x_spec = amp_chart.x_range();
    let y_spec = amp_chart.y_range();
    amp_chart.draw_series(std::iter::once(Rectangle::new(
        [(x_spec.start, y_spec.start), (x_spec.end, y_spec.end)],
        BLACK.stroke_width(1),
    )))?;

    // --- Rate Chart ---
    let mut rate_chart = ChartBuilder::on(&rate_area)
        .margin_top(10)
        .margin_left(15)
        .x_label_area_size(75)
        .y_label_area_size(120)
        .build_cartesian_2d(rate_min_x..rate_max_x, 0.0..rate_max_y)?;
    rate_chart
        .configure_mesh()
        .x_desc("Rate [Hz]")
        .y_desc("Amplitude")
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .x_labels(7)
        //.y_labels(7)
        .axis_style(BLACK.stroke_width(1))
        .label_style(("sans-serif ", 30))
        .y_label_formatter(&|y| format!("{:.1e}", y))
        .draw()?;
    rate_chart.draw_series(LineSeries::new(
        c_band_rate_profile.iter().cloned(),
        RED.stroke_width(1),
    ))?; // C-band in RED
         //.label("C-band Rate")
         //.legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));
    rate_chart.draw_series(LineSeries::new(
        x_band_rate_profile.iter().cloned(),
        BLUE.stroke_width(1),
    ))?; // X-band in BLUE
         //.label("X-band Rate")
         //.legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));

    //rate_chart.configure_series_labels()
    //    .background_style(&WHITE.mix(0.8))
    //    .border_style(&BLACK)
    //    .draw()?;

    // Draw bounding box for rate_chart
    let x_spec = rate_chart.x_range();
    let y_spec = rate_chart.y_range();
    rate_chart.draw_series(std::iter::once(Rectangle::new(
        [(x_spec.start, y_spec.start), (x_spec.end, y_spec.end)],
        BLACK.stroke_width(1),
    )))?;

    // --- Heatmap Chart ---
    let mut heatmap_chart = ChartBuilder::on(&heatmap_area)
        .margin(20)
        .x_label_area_size(55)
        .y_label_area_size(85)
        .build_cartesian_2d(0.0..bw, rate_min_x..rate_max_x)?;
    heatmap_chart
        .configure_mesh()
        .x_desc("Frequency [MHz]")
        .y_desc("Rate [Hz]")
        .x_labels(7)
        .y_labels(7)
        .x_label_formatter(&|y| format!("{:.0}", y))
        .y_label_formatter(&|y| format!("{:.1}", y))
        .label_style(("sans-serif ", 30))
        .draw()?;

    let resolution = 400;
    let mut heatmap_values = Vec::new();
    let mut heatmap_data_max_val = f64::NEG_INFINITY;
    for i in 0..resolution {
        for j in 0..resolution {
            let y = rate_min_x + (rate_max_x - rate_min_x) * i as f64 / (resolution - 1) as f64;
            let x = 0.0 + bw * j as f64 / (resolution - 1) as f64;
            let val = heatmap_func(x, y);
            heatmap_values.push(val);
            if val > heatmap_data_max_val {
                heatmap_data_max_val = val;
            }
        }
    }

    for (idx, val) in heatmap_values.iter().enumerate() {
        let i = idx / resolution;
        let j = idx % resolution;
        let y = rate_min_x + (rate_max_x - rate_min_x) * i as f64 / (resolution - 1) as f64;
        let x = 0.0 + bw * j as f64 / (resolution - 1) as f64;
        let x_step = bw / (resolution - 1) as f64;
        let y_step = (rate_max_x - rate_min_x) / (resolution - 1) as f64;
        let normalized_val = if heatmap_data_max_val > 0.0 {
            *val / heatmap_data_max_val
        } else {
            0.0
        };
        heatmap_chart.draw_series(std::iter::once(Rectangle::new(
            [(x, y), (x + x_step, y + y_step)],
            HSLColor((1.0 - normalized_val) * 0.7, 1.0, 0.5).filled(),
        )))?;
    }

    // 4. Colorbar
    let mut colorbar = ChartBuilder::on(&colorbar_area)
        .margin_top(10)
        .margin_bottom(50)
        .set_label_area_size(LabelAreaPosition::Right, 100)
        .set_label_area_size(LabelAreaPosition::Left, 0)
        .set_label_area_size(LabelAreaPosition::Top, 10)
        .set_label_area_size(LabelAreaPosition::Bottom, 25)
        .build_cartesian_2d(0.0..1.0, 0.0..max_amplitude)?;

    let steps = 100;
    for i in 0..steps {
        let value = i as f64 / (steps - 1) as f64;
        let color = HSLColor(((1.0 - value) * 0.7).into(), 1.0, 0.5);
        colorbar.draw_series(std::iter::once(Rectangle::new(
            [
                (0.0, value * max_amplitude),
                (1.0, (value + 1.0 / steps as f64) * max_amplitude),
            ],
            color.filled(),
        )))?;
    }

    colorbar
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .disable_x_axis()
        .y_labels(7)
        .y_label_style(("sans-serif ", 30))
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .draw()?;

    // 5. Stats
    let font_large = ("sans-serif ", 35).into_font();
    let font_small = ("sans-serif ", 25).into_font(); // Smaller font
    let left_x = 30;
    let right_x = stats_area.get_pixel_range().0.len() as i32 - 30;
    let mut y = 20;

    for (k, v) in stat_keys.iter().zip(stat_vals.iter()) {
        let current_font = if v.contains(".cor") || v.contains(".bin") {
            &font_small
        } else {
            &font_large
        };
        let text_style = TextStyle::from(current_font.clone()).color(&BLACK);

        stats_area.draw(&Text::new(k.to_string(), (left_x, y), text_style.clone()))?;
        stats_area.draw(&Text::new(
            v.to_string(),
            (right_x, y),
            text_style.clone().pos(Pos::new(HPos::Right, VPos::Top)),
        ))?;
        y += 40;
    }

    root.present()?;
    Ok(())
}
