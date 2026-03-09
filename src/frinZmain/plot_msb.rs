use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};
use chrono::{DateTime, Utc, TimeZone};
use plotters::style::colors::colormaps::ViridisRGB;
use num_complex::Complex;
use std::path::Path;
use ndarray::Array2; // Added for dynamic spectrum

pub fn frequency_plane(
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

    let (left_area, right_area) = root.split_horizontally(width / 2 );
    let (top_left_area, rate_area) = left_area.split_vertically(height / 2 +50);
    let (phase_area, amp_area) = top_left_area.split_vertically(top_left_area.get_pixel_range().1.len() as u32 / 4 -10);
    let (heatmap_and_colorbar_area, stats_area) = right_area.split_vertically(height / 2 +50);
    let (heatmap_area, colorbar_area) = heatmap_and_colorbar_area.split_horizontally(heatmap_and_colorbar_area.get_pixel_range().0.len() as u32 - 120);

    // --- Determine axis ranges ---
    let c_amp_max_y = c_band_amp_profile.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
    let x_amp_max_y = x_band_amp_profile.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
    let amp_max_y = c_amp_max_y.max(x_amp_max_y) * 1.1;
    let c_band_rate_max_y = c_band_rate_profile.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
    let x_band_rate_max_y = x_band_rate_profile.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
    let rate_max_y = c_band_rate_max_y.max(x_band_rate_max_y) * 1.1;

    let c_band_rate_min_x = c_band_rate_profile.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
    let c_band_rate_max_x = c_band_rate_profile.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);
    let x_band_rate_min_x = x_band_rate_profile.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
    let x_band_rate_max_x = x_band_rate_profile.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);

    let rate_min_x = c_band_rate_min_x.min(x_band_rate_min_x);
    let rate_max_x = c_band_rate_max_x.max(x_band_rate_max_x);

    // --- Phase Chart ---
    let mut phase_chart = ChartBuilder::on(&phase_area)
        .margin_top(20)
        .margin_left(15)
        .x_label_area_size(0)
        .y_label_area_size(120)
        .build_cartesian_2d(0.0..bw, -180.0..180.0)?;
    phase_chart.configure_mesh()
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
    phase_chart.draw_series(LineSeries::new(c_band_phase_profile.iter().cloned(), RED.stroke_width(1)))?;
    phase_chart.draw_series(LineSeries::new(x_band_phase_profile.iter().cloned(), BLUE.stroke_width(1)))?;
    //phase_chart.draw_series(LineSeries::new(x_band_uncalibrated_phase_profile.iter().cloned(), BLACK.stroke_width(1)))?;

    // Draw bounding box for phase_chart
    let x_spec = phase_chart.x_range();
    let y_spec = phase_chart.y_range();
    phase_chart.draw_series(std::iter::once(Rectangle::new([(x_spec.start, y_spec.start), (x_spec.end, y_spec.end)], BLACK.stroke_width(1))))?;

    // --- Amplitude Chart ---
    let mut amp_chart = ChartBuilder::on(&amp_area)
        .margin_left(15)
        .x_label_area_size(65)
        .y_label_area_size(120)
        .build_cartesian_2d(0.0..bw, 0.0..amp_max_y)?;
    amp_chart.configure_mesh()
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
    amp_chart.draw_series(LineSeries::new(c_band_amp_profile.iter().cloned(), RED.stroke_width(1)))?;
    amp_chart.draw_series(LineSeries::new(x_band_amp_profile.iter().cloned(), BLUE.stroke_width(1)))?;

    // Draw bounding box for amp_chart
    let x_spec = amp_chart.x_range();
    let y_spec = amp_chart.y_range();
    amp_chart.draw_series(std::iter::once(Rectangle::new([(x_spec.start, y_spec.start), (x_spec.end, y_spec.end)], BLACK.stroke_width(1))))?;

    // --- Rate Chart ---
    let mut rate_chart = ChartBuilder::on(&rate_area)
        .margin_top(10)
        .margin_left(15)
        .x_label_area_size(75)
        .y_label_area_size(120)
        .build_cartesian_2d(rate_min_x..rate_max_x, 0.0..rate_max_y)?;
    rate_chart.configure_mesh()
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
    rate_chart.draw_series(LineSeries::new(c_band_rate_profile.iter().cloned(), RED.stroke_width(1)))?; // C-band in RED
        //.label("C-band Rate")
        //.legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));
    rate_chart.draw_series(LineSeries::new(x_band_rate_profile.iter().cloned(), BLUE.stroke_width(1)))?; // X-band in BLUE
        //.label("X-band Rate")
        //.legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));

    //rate_chart.configure_series_labels()
    //    .background_style(&WHITE.mix(0.8))
    //    .border_style(&BLACK)
    //    .draw()?;

    // Draw bounding box for rate_chart
    let x_spec = rate_chart.x_range();
    let y_spec = rate_chart.y_range();
    rate_chart.draw_series(std::iter::once(Rectangle::new([(x_spec.start, y_spec.start), (x_spec.end, y_spec.end)], BLACK.stroke_width(1))))?;

    // --- Heatmap Chart ---
    let mut heatmap_chart = ChartBuilder::on(&heatmap_area)
        .margin(20)
        .x_label_area_size(55)
        .y_label_area_size(85)
        .build_cartesian_2d(0.0..bw, rate_min_x..rate_max_x)?;
    heatmap_chart.configure_mesh()
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
            if val > heatmap_data_max_val { heatmap_data_max_val = val; }
        }
    }

    for (idx, val) in heatmap_values.iter().enumerate() {
        let i = idx / resolution;
        let j = idx % resolution;
        let y = rate_min_x + (rate_max_x - rate_min_x) * i as f64 / (resolution - 1) as f64;
        let x = 0.0 + bw * j as f64 / (resolution - 1) as f64;
        let x_step = bw / (resolution - 1) as f64;
        let y_step = (rate_max_x - rate_min_x) / (resolution - 1) as f64;
        let normalized_val = if heatmap_data_max_val > 0.0 { *val / heatmap_data_max_val } else { 0.0 };
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
            [(0.0, value * max_amplitude), (1.0, (value + 1.0 / steps as f64) * max_amplitude)],
            color.filled(),
        )))?;
    }

    colorbar.configure_mesh()
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
        stats_area.draw(&Text::new(v.to_string(), (right_x, y), text_style.clone().pos(Pos::new(HPos::Right, VPos::Top))))?;
        y += 40;
    }

    root.present()?;
    Ok(())
}
