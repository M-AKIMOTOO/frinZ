use super::{
    data::{epoch_mjd, number, Table},
    stats::histogram,
};
use eframe::egui::{self, Color32};
use egui_plot::{Axis, AxisHints, Legend, Line, LineStyle, MarkerShape, Placement, Plot, Points};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlotMode {
    Series,
    Histogram,
    Sources,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SeriesStyle {
    #[default]
    Scatter,
    Linear,
    ScatterDotted,
}

#[derive(Clone)]
pub struct PlotOptions {
    pub mode: PlotMode,
    pub series_style: SeriesStyle,
    pub x: Option<usize>,
    pub y: usize,
    pub y_source: Option<String>,
    pub y2: Option<usize>,
    pub y2_source: Option<String>,
    pub reference_mjd: Option<f64>,
    pub reference_text: String,
    pub log_y: bool,
    pub noise_bars: bool,
    pub bins: usize,
    pub preview_9_6: bool,
    pub reset: bool,
}

pub fn secondary_column(options: &PlotOptions) -> Option<usize> {
    options
        .y2
        .filter(|&column| column != options.y || options.y_source != options.y2_source)
}
impl Default for PlotOptions {
    fn default() -> Self {
        Self {
            mode: PlotMode::Series,
            series_style: SeriesStyle::Scatter,
            x: None,
            y: 4,
            y_source: None,
            y2: None,
            y2_source: None,
            reference_mjd: None,
            reference_text: String::new(),
            log_y: false,
            noise_bars: true,
            bins: 15,
            preview_9_6: false,
            reset: true,
        }
    }
}

pub fn color(_index: usize) -> Color32 {
    Color32::WHITE
}

fn marker(index: usize) -> MarkerShape {
    [
        MarkerShape::Circle,
        MarkerShape::Square,
        MarkerShape::Diamond,
        MarkerShape::Cross,
        MarkerShape::Plus,
    ][index % 5]
}

pub fn first_mjd_reference(table: &Table) -> Option<f64> {
    if table.column("Epoch").is_none() && table.column("MJD").is_none() {
        return None;
    }
    table
        .rows
        .iter()
        .enumerate()
        .filter_map(|(row, _)| x_value(table, row, None, None))
        .min_by(f64::total_cmp)
        .map(f64::floor)
}

pub fn x_value(
    table: &Table,
    row: usize,
    column: Option<usize>,
    reference_mjd: Option<f64>,
) -> Option<f64> {
    if let Some(column) = column {
        let value = number(&table.rows[row][column]);
        return if table.columns[column].name.eq_ignore_ascii_case("MJD") {
            value.map(|x| x - reference_mjd.unwrap_or(0.0))
        } else {
            value
        };
    }
    let time_value = table
        .column("Epoch")
        .and_then(|c| epoch_mjd(&table.rows[row][c]))
        .or_else(|| {
            table
                .column("MJD")
                .and_then(|c| number(&table.rows[row][c]))
        });
    time_value
        .map(|x| x - reference_mjd.unwrap_or(0.0))
        .or_else(|| {
            (table.column("Epoch").is_none() && table.column("MJD").is_none())
                .then_some((row + 1) as f64)
        })
}

pub fn x_label(table: &Table, column: Option<usize>, reference_mjd: Option<f64>) -> String {
    let is_mjd = column.map_or_else(
        || table.column("Epoch").is_some() || table.column("MJD").is_some(),
        |c| table.columns[c].name.eq_ignore_ascii_case("MJD"),
    );
    if is_mjd && reference_mjd.is_some() {
        return format!(
            "MJD − {} (days)",
            super::stats::display(reference_mjd.unwrap())
        );
    }
    column
        .map(|c| format!("{} {}", table.columns[c].name, table.columns[c].unit))
        .unwrap_or_else(|| {
            if table.column("Epoch").is_some() || table.column("MJD").is_some() {
                "Epoch (MJD, UTC)".into()
            } else {
                "Original row".into()
            }
        })
}

pub fn figure_size(width: f32, height: f32) -> egui::Vec2 {
    let width = width.min(height * 9.0 / 6.0).max(1.0);
    egui::vec2(width, width * 6.0 / 9.0)
}

fn preview_size(width: f32, height: f32, fixed_ratio: bool) -> egui::Vec2 {
    if fixed_ratio {
        figure_size(width, height)
    } else {
        egui::vec2(width.max(1.0), height.max(1.0))
    }
}

pub fn show(
    ui: &mut egui::Ui,
    table: &Table,
    rows: &[usize],
    selected: &[usize],
    options: &mut PlotOptions,
) -> egui::Rect {
    if table.columns.is_empty() {
        ui.label("Open a frinZ TSV to plot its parameters.");
        return egui::Rect::NOTHING;
    }
    let column = &table.columns[options.y];
    let y_label = format!(
        "{} {}{}",
        column.name,
        column.unit,
        if options.log_y { " (log10)" } else { "" }
    );
    let size = preview_size(
        ui.available_width(),
        ui.available_height(),
        options.preview_9_6,
    );
    let id = egui::Id::new((
        "parameter_plot",
        options.mode as u8,
        options.x,
        options.y,
        options.y_source.clone(),
        options.y2,
        options.y2_source.clone(),
        options.reference_mjd.map(f64::to_bits),
        options.log_y,
    ));
    let mut plot = Plot::new(id)
        .id(id)
        .width(size.x)
        .height(size.y)
        .legend(Legend::default())
        .y_axis_label(&y_label);
    if options.reset {
        plot = plot.reset();
        options.reset = false;
    }
    let log_y = options.log_y;
    let transform = |v: f64| -> Option<f64> {
        if log_y {
            (v > 0.0).then(|| v.log10())
        } else {
            Some(v)
        }
    };
    let all_sources = table.sources();
    let source_color =
        |source: &str| color(all_sources.iter().position(|s| s == source).unwrap_or(0));
    let source_name = |source: &str| {
        let index = all_sources.iter().position(|s| s == source).unwrap_or(0);
        format!("{} {source}", ["●", "■", "◆", "×", "+"][index % 5])
    };
    let value_range = |column: usize, source: Option<&str>| -> Option<(f64, f64)> {
        let mut values = rows
            .iter()
            .filter(|&&row| source.is_none_or(|source| table.source(row) == source))
            .filter_map(|&r| number(&table.rows[r][column]))
            .filter_map(transform);
        let first = values.next()?;
        let (mut lo, mut hi) = values.fold((first, first), |(lo, hi), value| {
            (lo.min(value), hi.max(value))
        });
        if lo == hi {
            let pad = (lo.abs() * 0.05).max(1e-6);
            lo -= pad;
            hi += pad;
        }
        Some((lo, hi))
    };
    let response = match options.mode {
        PlotMode::Series => {
            let x_label = x_label(table, options.x, options.reference_mjd);
            let secondary_column = secondary_column(options);
            let primary_range = value_range(options.y, options.y_source.as_deref());
            let secondary_range = secondary_column
                .and_then(|column| value_range(column, options.y2_source.as_deref()));
            let axis_primary_range = primary_range.or(secondary_range);
            let secondary = secondary_column.zip(axis_primary_range.zip(secondary_range));
            let axis_label = |column: usize, source: Option<&str>| {
                format!(
                    "{} {}{}{}",
                    table.columns[column].name,
                    table.columns[column].unit,
                    source.map(|name| format!(" [{name}]")).unwrap_or_default(),
                    if log_y { " (log10)" } else { "" }
                )
            };
            let primary_label = axis_label(options.y, options.y_source.as_deref());
            let plot = if let Some((
                secondary_column,
                ((primary_lo, primary_hi), (secondary_lo, secondary_hi)),
            )) = secondary
            {
                let secondary_label = axis_label(secondary_column, options.y2_source.as_deref());
                let hints = vec![
                    AxisHints::new(Axis::Y).label(primary_label.clone()),
                    AxisHints::new_y()
                        .placement(Placement::RightTop)
                        .label(secondary_label)
                        .formatter(move |mark, _| {
                            let fraction = (mark.value - primary_lo) / (primary_hi - primary_lo);
                            let value = secondary_lo + fraction * (secondary_hi - secondary_lo);
                            super::stats::display(if log_y { 10.0_f64.powf(value) } else { value })
                        }),
                ];
                plot.custom_y_axes(hints)
            } else {
                plot.y_axis_label(primary_label)
            };
            let mut groups: BTreeMap<String, Vec<(usize, [f64; 2])>> = BTreeMap::new();
            let mut secondary_groups: BTreeMap<String, Vec<[f64; 2]>> = BTreeMap::new();
            for &r in rows {
                let x = x_value(table, r, options.x, options.reference_mjd);
                if options
                    .y_source
                    .as_deref()
                    .is_none_or(|source| table.source(r) == source)
                {
                    if let (Some(x), Some(y)) =
                        (x, number(&table.rows[r][options.y]).and_then(transform))
                    {
                        groups
                            .entry(table.source(r).into())
                            .or_default()
                            .push((r, [x, y]));
                    }
                }
                if let (Some(column), Some((primary_range, secondary_range))) =
                    (secondary_column, axis_primary_range.zip(secondary_range))
                {
                    if options
                        .y2_source
                        .as_deref()
                        .is_none_or(|source| table.source(r) == source)
                    {
                        if let (Some(x), Some(value)) =
                            (x, number(&table.rows[r][column]).and_then(transform))
                        {
                            let fraction = (value - secondary_range.0)
                                / (secondary_range.1 - secondary_range.0);
                            secondary_groups
                                .entry(table.source(r).into())
                                .or_default()
                                .push([
                                    x,
                                    primary_range.0
                                        + fraction * (primary_range.1 - primary_range.0),
                                ]);
                        }
                    }
                }
            }
            for points in secondary_groups.values_mut() {
                points.sort_by(|a, b| a[0].total_cmp(&b[0]));
            }
            if groups.is_empty() && secondary_groups.is_empty() {
                ui.label("No finite X/Y pairs in this view (log10 requires positive Y).");
            }
            plot.x_axis_label(x_label).show(ui, |plot_ui| {
                for (source, mut points) in groups {
                    let tint = source_color(&source);
                    points.sort_by(|a, b| a.1[0].total_cmp(&b.1[0]));
                    let values: Vec<[f64; 2]> = points.iter().map(|p| p.1).collect();
                    if !values.is_empty() {
                        if options.series_style != SeriesStyle::Scatter {
                            let label = if options.series_style == SeriesStyle::Linear {
                                source_name(&source)
                            } else {
                                String::new()
                            };
                            let line = Line::new(label, values.clone()).color(tint).width(1.3);
                            plot_ui.line(if options.series_style == SeriesStyle::ScatterDotted {
                                line.style(LineStyle::dotted_dense())
                            } else {
                                line
                            });
                        }
                        if options.series_style != SeriesStyle::Linear {
                            let index = all_sources.iter().position(|s| s == &source).unwrap_or(0);
                            plot_ui.points(
                                Points::new(source_name(&source), values)
                                    .color(tint)
                                    .shape(marker(index))
                                    .radius(3.5),
                            );
                        }
                        if options.noise_bars && column.name == "Amp" {
                            if let Some(noise) = table.column("Noise-level") {
                                for &(r, [x, _]) in &points {
                                    if let (Some(y), Some(error)) = (
                                        number(&table.rows[r][options.y]),
                                        number(&table.rows[r][noise]).filter(|e| *e >= 0.0),
                                    ) {
                                        if let (Some(low), Some(high)) =
                                            (transform(y - error), transform(y + error))
                                        {
                                            plot_ui.line(
                                                Line::new("", vec![[x, low], [x, high]])
                                                    .color(tint.gamma_multiply(0.5)),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        let highlighted: Vec<_> = points
                            .into_iter()
                            .filter(|(r, _)| selected.contains(r))
                            .map(|p| p.1)
                            .collect();
                        if !highlighted.is_empty() {
                            plot_ui.points(
                                Points::new("Selected rows", highlighted)
                                    .color(Color32::WHITE)
                                    .filled(false)
                                    .radius(6.0),
                            );
                        }
                    }
                }
                if secondary.is_some() {
                    for (source, points) in &secondary_groups {
                        let tint = source_color(source);
                        let index = all_sources.iter().position(|s| s == source).unwrap_or(0);
                        let source_label = format!("{} [Y₂]", source_name(source));
                        if options.series_style != SeriesStyle::Scatter {
                            let label = if options.series_style == SeriesStyle::Linear {
                                source_label.clone()
                            } else {
                                String::new()
                            };
                            let line = Line::new(label, points.clone()).color(tint).width(1.1);
                            plot_ui.line(if options.series_style == SeriesStyle::ScatterDotted {
                                line.style(LineStyle::dotted_dense())
                            } else {
                                line
                            });
                        }
                        if options.series_style != SeriesStyle::Linear {
                            plot_ui.points(
                                Points::new(source_label, points.clone())
                                    .color(tint)
                                    .shape(marker(index))
                                    .filled(false)
                                    .radius(4.5),
                            );
                        }
                    }
                }
            })
        }
        PlotMode::Histogram => {
            plot.x_axis_label(&y_label)
                .y_axis_label("Count")
                .include_y(0.0)
                .show(ui, |plot_ui| {
                    let values: Vec<_> = rows
                        .iter()
                        .filter_map(|&r| number(&table.rows[r][options.y]).and_then(transform))
                        .collect();
                    // Shared bin edges make overlaid source histograms comparable.
                    let bins = histogram(&values, options.bins);
                    if bins.is_empty() {
                        return;
                    }
                    let lower = bins[0].0 - bins[0].1 / 2.0;
                    let width = bins[0].1;
                    for (index, source) in all_sources.iter().enumerate() {
                        let mut counts = vec![0; bins.len()];
                        for &r in rows {
                            if table.source(r) != source {
                                continue;
                            }
                            if let Some(v) = number(&table.rows[r][options.y]).and_then(transform) {
                                let i = (((v - lower) / width) as usize).min(bins.len() - 1);
                                counts[i] += 1;
                            }
                        }
                        if counts.iter().all(|&n| n == 0) {
                            continue;
                        }
                        let mut outline = vec![[lower, 0.0]];
                        let mut tops = Vec::new();
                        for (&(center, width, _), count) in bins.iter().zip(counts) {
                            outline.extend([
                                [center - width / 2.0, count as f64],
                                [center + width / 2.0, count as f64],
                            ]);
                            if count > 0 {
                                tops.push([center, count as f64]);
                            }
                        }
                        outline.push([lower + width * bins.len() as f64, 0.0]);
                        let tint = source_color(source);
                        let style = [
                            LineStyle::Solid,
                            LineStyle::dashed_loose(),
                            LineStyle::dotted_dense(),
                            LineStyle::dashed_dense(),
                        ][index % 4];
                        plot_ui.line(
                            Line::new(source_name(source), outline)
                                .color(tint)
                                .style(style),
                        );
                        plot_ui.points(
                            Points::new(source_name(source), tops)
                                .color(tint)
                                .shape(marker(index))
                                .radius(3.5),
                        );
                    }
                })
        }
        PlotMode::Sources => {
            let summaries = table.source_summaries(rows, options.y);
            let labels: Vec<_> = summaries.iter().map(|s| s.source.clone()).collect();
            plot.x_axis_label("Source / mean ± standard error (arithmetic statistics)")
                .x_axis_formatter(move |mark, _| {
                    let i = mark.value.round();
                    if (mark.value - i).abs() < 0.01 && i >= 0.0 {
                        labels.get(i as usize).cloned().unwrap_or_default()
                    } else {
                        String::new()
                    }
                })
                .show(ui, |plot_ui| {
                    for (i, summary) in summaries.iter().enumerate() {
                        if let Some(s) = &summary.statistics {
                            let tint = source_color(&summary.source);
                            if let Some(mean) = transform(s.mean) {
                                plot_ui.points(
                                    Points::new(
                                        source_name(&summary.source),
                                        vec![[i as f64, mean]],
                                    )
                                    .color(tint)
                                    .shape(marker(
                                        all_sources
                                            .iter()
                                            .position(|s| s == &summary.source)
                                            .unwrap_or(0),
                                    ))
                                    .radius(6.0),
                                );
                                if let Some(error) = s.standard_error {
                                    if let (Some(low), Some(high)) =
                                        (transform(s.mean - error), transform(s.mean + error))
                                    {
                                        plot_ui.line(
                                            Line::new("", vec![[i as f64, low], [i as f64, high]])
                                                .color(tint)
                                                .width(2.0),
                                        );
                                    }
                                }
                            }
                        }
                    }
                })
        }
    };
    response.response.rect
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn figure_ratio_is_independent_of_data_axis_units() {
        for (width, height) in [(1200.0, 350.0), (400.0, 600.0)] {
            let size = figure_size(width, height);
            assert!(size.x <= width && size.y <= height);
            assert!((size.x / size.y - 1.5).abs() < 1e-6);
        }
    }

    #[test]
    fn screen_preview_uses_full_available_space_unless_fixed_ratio_is_requested() {
        let wide = preview_size(1200.0, 350.0, false);
        assert_eq!(wide, egui::vec2(1200.0, 350.0));
        let fixed = preview_size(1200.0, 350.0, true);
        assert!((fixed.x / fixed.y - 1.5).abs() < 1e-6);
        assert_eq!(fixed.y, wide.y);
    }
}
