use super::{
    data::{Table, epoch_mjd, number},
    stats::histogram,
};
use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};
use eframe::egui::{self, Color32};
use egui_plot::{Axis, AxisHints, Legend, Line, LineStyle, MarkerShape, Placement, Plot, Points};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlotMode {
    Series,
    Histogram,
    Sources,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeriesStyle {
    #[default]
    Scatter,
    Linear,
    ScatterDotted,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XAxisScale {
    #[default]
    Automatic,
    Hours,
    Date,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PlotOptions {
    pub mode: PlotMode,
    pub series_style: SeriesStyle,
    pub x: Option<usize>,
    pub x_axis_scale: XAxisScale,
    pub date_origin: String,
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
    pub reset_generation: u64,
}

fn begin_plot_reset(options: &mut PlotOptions) -> bool {
    if !options.reset {
        return false;
    }
    options.reset = false;
    options.reset_generation = options.reset_generation.wrapping_add(1);
    true
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
            x_axis_scale: XAxisScale::Automatic,
            date_origin: String::new(),
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
            reset_generation: 0,
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
    let has_mjd_time = table.column("Epoch").is_some()
        || table.column("MJD").is_some()
        || table.column("Time").is_some_and(|column| {
            table
                .rows
                .iter()
                .any(|row| epoch_mjd(&row[column]).is_some())
        });
    if !has_mjd_time {
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

pub fn automatic_x_name(table: &Table) -> &'static str {
    if table.column("Epoch").is_some() || table.column("MJD").is_some() {
        "Epoch / MJD"
    } else if table.column("Time").is_some() {
        "Time"
    } else {
        "Row number"
    }
}

pub fn uses_mjd_x(table: &Table, column: Option<usize>) -> bool {
    column.map_or_else(
        || {
            table.column("Epoch").is_some()
                || table.column("MJD").is_some()
                || table
                    .column("Time")
                    .is_some_and(|c| table.rows.iter().any(|row| epoch_mjd(&row[c]).is_some()))
        },
        |c| {
            table.columns[c].name.eq_ignore_ascii_case("MJD")
                || table.columns[c].name.eq_ignore_ascii_case("Epoch")
                || (table.columns[c].name.eq_ignore_ascii_case("Time")
                    && table.rows.iter().any(|row| epoch_mjd(&row[c]).is_some()))
        },
    )
}

fn time_column(table: &Table, column: Option<usize>) -> Option<usize> {
    column
        .filter(|&c| table.columns[c].name.eq_ignore_ascii_case("Time"))
        .or_else(|| {
            (column.is_none() && table.column("Epoch").is_none() && table.column("MJD").is_none())
                .then(|| table.column("Time"))
                .flatten()
        })
}

pub fn time_column_is_seconds(table: &Table, column: Option<usize>) -> bool {
    time_column(table, column).is_some_and(|c| {
        let unit = table.columns[c].unit.to_ascii_lowercase();
        !unit.contains("mjd")
            && !unit.contains("day")
            && !unit.contains("[d]")
            && table
                .rows
                .iter()
                .any(|row| number(&row[c]).is_some() && epoch_mjd(&row[c]).is_none())
    })
}

/// Parse an ISO/RFC3339 date (UTC when no zone is given), or an MJD value.
pub fn date_origin_mjd(text: &str) -> Option<f64> {
    let text = text.trim();
    if let Ok(mjd) = text.parse::<f64>() {
        return (mjd.is_finite() && (30_000.0..100_000.0).contains(&mjd)).then_some(mjd);
    }
    let unix_seconds = DateTime::parse_from_rfc3339(text)
        .ok()
        .map(|date| date.timestamp() as f64 + f64::from(date.timestamp_subsec_millis()) / 1000.0)
        .or_else(|| {
            ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f"]
                .into_iter()
                .find_map(|format| NaiveDateTime::parse_from_str(text, format).ok())
                .map(|date| date.and_utc().timestamp() as f64)
        })
        .or_else(|| {
            NaiveDate::parse_from_str(text, "%Y-%m-%d")
                .ok()
                .map(|date| date.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp() as f64)
        })?;
    Some(unix_seconds / 86_400.0 + 40_587.0)
}

pub fn mjd_date_label(mjd: f64) -> String {
    let unix_seconds = (mjd - 40_587.0) * 86_400.0;
    if !unix_seconds.is_finite() || unix_seconds < i64::MIN as f64 || unix_seconds > i64::MAX as f64
    {
        return String::new();
    }
    Utc.timestamp_opt(unix_seconds.floor() as i64, 0)
        .single()
        .map(|date| date.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_default()
}

pub fn x_value(
    table: &Table,
    row: usize,
    column: Option<usize>,
    reference_mjd: Option<f64>,
) -> Option<f64> {
    if let Some(column) = column {
        let text = &table.rows[row][column];
        let name = &table.columns[column].name;
        let value = number(text).or_else(|| epoch_mjd(text));
        return if name.eq_ignore_ascii_case("MJD")
            || name.eq_ignore_ascii_case("Epoch")
            || (name.eq_ignore_ascii_case("Time") && number(text).is_none())
        {
            value.map(|x| x - reference_mjd.unwrap_or(0.0))
        } else {
            value
        };
    }
    let mjd_value = table
        .column("Epoch")
        .and_then(|c| epoch_mjd(&table.rows[row][c]))
        .or_else(|| {
            table
                .column("MJD")
                .and_then(|c| number(&table.rows[row][c]))
        });
    if let Some(value) = mjd_value {
        return Some(value - reference_mjd.unwrap_or(0.0));
    }
    if let Some(time) = table.column("Time") {
        let text = &table.rows[row][time];
        if let Some(value) = epoch_mjd(text) {
            return Some(value - reference_mjd.unwrap_or(0.0));
        }
        if let Some(value) = number(text) {
            return Some(value);
        }
    }
    (table.column("Epoch").is_none()
        && table.column("MJD").is_none()
        && table.column("Time").is_none())
    .then_some((row + 1) as f64)
}

pub fn x_value_for_scale(table: &Table, row: usize, options: &PlotOptions) -> Option<f64> {
    match options.x_axis_scale {
        XAxisScale::Automatic => x_value(table, row, options.x, options.reference_mjd),
        XAxisScale::Hours if uses_mjd_x(table, options.x) => {
            let reference = options.reference_mjd.or_else(|| first_mjd_reference(table));
            Some(x_value(table, row, options.x, reference)? * 24.0)
        }
        XAxisScale::Hours if time_column_is_seconds(table, options.x) => {
            Some(x_value(table, row, options.x, None)? / 3600.0)
        }
        XAxisScale::Hours => x_value(table, row, options.x, options.reference_mjd),
        XAxisScale::Date if uses_mjd_x(table, options.x) => x_value(table, row, options.x, None),
        XAxisScale::Date if time_column_is_seconds(table, options.x) => Some(
            date_origin_mjd(&options.date_origin)?
                + x_value(table, row, options.x, None)? / 86_400.0,
        ),
        XAxisScale::Date => x_value(table, row, options.x, options.reference_mjd),
    }
}

pub fn x_label(table: &Table, column: Option<usize>, reference_mjd: Option<f64>) -> String {
    let is_mjd = uses_mjd_x(table, column);
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
            } else if let Some(c) = table.column("Time") {
                format!("{} {}", table.columns[c].name, table.columns[c].unit)
            } else {
                "Original row".into()
            }
        })
}

pub fn x_label_for_options(table: &Table, options: &PlotOptions) -> String {
    match options.x_axis_scale {
        XAxisScale::Automatic => x_label(table, options.x, options.reference_mjd),
        XAxisScale::Hours if uses_mjd_x(table, options.x) => {
            let reference = options
                .reference_mjd
                .or_else(|| first_mjd_reference(table))
                .unwrap_or(0.0);
            format!("Hours since MJD {}", super::stats::display(reference))
        }
        XAxisScale::Hours if time_column_is_seconds(table, options.x) => "Time [h]".into(),
        XAxisScale::Hours => x_label(table, options.x, options.reference_mjd),
        XAxisScale::Date => "Date (UTC)".into(),
    }
}

#[cfg(test)]
mod x_axis_tests {
    use super::*;

    #[test]
    fn add_plot_time_column_is_used_as_the_automatic_x_axis() {
        let table = Table::parse(
            "# Time\tAmp\tSNR\n# [s]\t[%]\t-\n0\t1.5\t3\n60\t2.5\t4\n",
            false,
        )
        .unwrap();

        assert_eq!(automatic_x_name(&table), "Time");
        assert_eq!(x_value(&table, 0, None, None), Some(0.0));
        assert_eq!(x_value(&table, 1, None, None), Some(60.0));
        assert_eq!(x_label(&table, None, None), "Time [s]");

        let options = PlotOptions {
            x_axis_scale: XAxisScale::Hours,
            ..Default::default()
        };
        assert_eq!(x_value_for_scale(&table, 1, &options), Some(1.0 / 60.0));

        let date_options = PlotOptions {
            x_axis_scale: XAxisScale::Date,
            date_origin: "2026-10-10".into(),
            ..Default::default()
        };
        let expected = date_origin_mjd("2026-10-10").unwrap() + 60.0 / 86_400.0;
        assert!((x_value_for_scale(&table, 1, &date_options).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn epoch_x_can_be_displayed_as_hours_or_calendar_dates() {
        let table = Table::parse(
            "Epoch\tAmp\n2026/283T04:57:00\t1.5\n2026/283T05:57:00\t2.5\n",
            false,
        )
        .unwrap();
        let reference = first_mjd_reference(&table).unwrap();
        let hours = PlotOptions {
            reference_mjd: Some(reference),
            x_axis_scale: XAxisScale::Hours,
            ..Default::default()
        };
        let date = PlotOptions {
            reference_mjd: Some(reference),
            x_axis_scale: XAxisScale::Date,
            ..Default::default()
        };

        assert!((x_value_for_scale(&table, 1, &hours).unwrap() - 5.95).abs() < 1e-8);
        assert!((x_value_for_scale(&table, 0, &date).unwrap() - 61323.20625).abs() < 1e-8);
        assert_eq!(mjd_date_label(61323.20625), "2026-10-10 04:57");
    }
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
    let reset = begin_plot_reset(options);
    let id = egui::Id::new((
        "parameter_plot",
        options.mode as u8,
        options.x,
        options.x_axis_scale as u8,
        options.date_origin.clone(),
        options.y,
        options.y_source.clone(),
        options.y2,
        options.y2_source.clone(),
        options.reference_mjd.map(f64::to_bits),
        options.log_y,
        options.reset_generation,
    ));
    let mut plot = Plot::new(id)
        .id(id)
        .width(size.x)
        .height(size.y)
        .legend(Legend::default())
        .y_axis_label(&y_label);
    if options.x_axis_scale == XAxisScale::Date {
        plot = plot.x_axis_formatter(|mark, _| mjd_date_label(mark.value));
    }
    if reset {
        plot = plot.reset();
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
            let x_label = x_label_for_options(table, options);
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
                let x = x_value_for_scale(table, r, options);
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
    fn plot_reset_advances_the_plot_state_generation() {
        let mut options = PlotOptions::default();
        assert!(begin_plot_reset(&mut options));
        assert_eq!(options.reset_generation, 1);
        assert!(!options.reset);
        assert!(!begin_plot_reset(&mut options));

        options.reset = true;
        assert!(begin_plot_reset(&mut options));
        assert_eq!(options.reset_generation, 2);
    }

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
