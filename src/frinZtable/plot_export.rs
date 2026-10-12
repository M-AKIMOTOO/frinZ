use super::{
    data::{number, Table},
    plots::{
        first_mjd_reference, secondary_column, x_label, x_value, PlotMode, PlotOptions, SeriesStyle,
    },
    stats::{histogram, Statistics},
};
use anyhow::{ensure, Context, Result};
use plotters::prelude::*;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

pub const IMAGE_SIZE: (u32, u32) = (1800, 1200);
const SCRIPT: &str = include_str!("plot_template.py");

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ExportFormat {
    #[default]
    Png,
    Pdf,
    Eps,
}

impl ExportFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Pdf => "pdf",
            Self::Eps => "eps",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Pdf => "PDF",
            Self::Eps => "EPS",
        }
    }

    fn from_path(path: &Path) -> Result<Self> {
        match path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("png") => Ok(Self::Png),
            Some("pdf") => Ok(Self::Pdf),
            Some("eps") => Ok(Self::Eps),
            _ => anyhow::bail!("Choose one graph format: .png, .pdf, or .eps"),
        }
    }
}

#[derive(Clone, Serialize)]
struct Settings {
    mode: PlotMode,
    x_label: String,
    hour_label: Option<String>,
    time_axis: bool,
    y_label: String,
    title: String,
    reference_mjd: Option<f64>,
    hour_reference_mjd: Option<f64>,
    y2_label: Option<String>,
    sources: Vec<String>,
    series_style: SeriesStyle,
    log_y: bool,
    noise_bars: bool,
    bins: usize,
    figsize: [f64; 2],
    dpi: u32,
}

struct Datum {
    row: usize,
    source: String,
    x: Option<f64>,
    y: Option<f64>,
    y2: Option<f64>,
    noise: Option<f64>,
}
#[derive(Clone)]
struct Layer {
    source: String,
    points: Vec<(f64, f64)>,
    points2: Vec<(f64, f64)>,
    errors: Vec<(f64, f64, f64)>,
    bars: Vec<(f64, f64, f64)>,
}

pub fn export(path: &Path, table: &Table, rows: &[usize], options: &PlotOptions) -> Result<()> {
    let format = ExportFormat::from_path(path)?;
    ensure!(options.y < table.columns.len(), "Select a numeric Y column");
    ensure!(
        options.x.is_none_or(|c| c < table.columns.len()),
        "Select a valid X column"
    );
    ensure!(
        options.y2.is_none_or(|c| c < table.columns.len()),
        "Select a valid second Y column"
    );
    ensure!(
        options
            .y2
            .is_none_or(|c| table.numeric_columns().contains(&c)),
        "The second Y axis requires a numeric column"
    );
    for source in [options.y_source.as_deref(), options.y2_source.as_deref()]
        .into_iter()
        .flatten()
    {
        ensure!(
            table.sources().iter().any(|candidate| candidate == source),
            "Unknown source for plot axis: {source}"
        );
    }
    ensure!(
        options.reference_mjd.is_none_or(f64::is_finite),
        "Reference MJD must be a finite number"
    );
    ensure!(
        !rows.is_empty(),
        "No rows are enabled for plotting in this view"
    );
    path.file_stem().context("A graph filename is required")?;
    let is_time_series = options.mode == PlotMode::Series
        && options.x.map_or_else(
            || table.column("Epoch").is_some() || table.column("MJD").is_some(),
            |column| table.columns[column].name.eq_ignore_ascii_case("MJD"),
        );
    let hour_reference_mjd = is_time_series
        .then(|| options.reference_mjd.or_else(|| first_mjd_reference(table)))
        .flatten();
    let script_path = path.with_extension("py");
    let hour_path = hour_reference_mjd.map(|_| {
        let extension = path
            .extension()
            .map(|extension| extension.to_string_lossy())
            .unwrap_or_default();
        path.with_file_name(format!(
            "{}_hour.{extension}",
            path.file_stem().unwrap().to_string_lossy(),
        ))
    });
    let mut paths = vec![path.to_path_buf(), script_path];
    if let Some(hour_path) = &hour_path {
        paths.push(hour_path.clone());
    }
    for file in &paths {
        ensure!(!file.exists(), "{} already exists. Choose a new graph name to preserve your previous script and settings.", file.display());
    }
    let column = &table.columns[options.y];
    let secondary_column = secondary_column(options);
    let noise_column = table.column("Noise-level");
    let data: Vec<_> = rows
        .iter()
        .map(|&row| {
            let source = table.source(row);
            Datum {
                row: row + 1,
                source: source.into(),
                x: x_value(table, row, options.x, options.reference_mjd),
                y: (options.mode != PlotMode::Series
                    || options
                        .y_source
                        .as_deref()
                        .is_none_or(|selected| selected == source))
                .then(|| number(&table.rows[row][options.y]))
                .flatten(),
                y2: (options.mode == PlotMode::Series
                    && secondary_column.is_some()
                    && options
                        .y2_source
                        .as_deref()
                        .is_none_or(|selected| selected == source))
                .then(|| number(&table.rows[row][secondary_column.unwrap()]))
                .flatten(),
                noise: noise_column
                    .and_then(|c| number(&table.rows[row][c]))
                    .filter(|v| *v >= 0.0),
            }
        })
        .collect();
    let axis_label = |column: usize, source: Option<&str>| {
        format!(
            "{} {}{}{}",
            table.columns[column].name,
            table.columns[column].unit,
            source.map(|name| format!(" [{name}]")).unwrap_or_default(),
            if options.log_y { " (log10)" } else { "" }
        )
    };
    let settings = Settings {
        mode: options.mode,
        x_label: x_label(table, options.x, options.reference_mjd),
        hour_label: hour_reference_mjd
            .map(|reference| format!("Hours since MJD {}", super::stats::display(reference))),
        time_axis: hour_reference_mjd.is_some(),
        y_label: axis_label(options.y, options.y_source.as_deref()),
        title: column.name.clone(),
        reference_mjd: options
            .x
            .is_none_or(|c| table.columns[c].name.eq_ignore_ascii_case("MJD"))
            .then_some(options.reference_mjd)
            .flatten(),
        hour_reference_mjd,
        y2_label: secondary_column
            .filter(|_| options.mode == PlotMode::Series)
            .map(|c| axis_label(c, options.y2_source.as_deref()))
            .filter(|_| {
                data.iter().any(|datum| {
                    datum
                        .y2
                        .is_some_and(|value| transform(value, options.log_y).is_some())
                })
            }),
        sources: table.sources(),
        series_style: options.series_style,
        log_y: options.log_y,
        noise_bars: options.noise_bars && column.name == "Amp",
        bins: options.bins,
        figsize: [9.0, 6.0],
        dpi: 200,
    };
    let layers = layers(&data, &settings);
    ensure!(
        layers
            .iter()
            .any(|l| !l.points.is_empty() || !l.points2.is_empty() || !l.bars.is_empty()),
        "No finite values to plot (log10 requires positive values)"
    );
    let data_tsv = data_tsv(&data)?;
    let embedded_json = serde_json::to_string(&serde_json::json!({
        "settings": &settings,
        "data_tsv": &data_tsv,
    }))?;
    let python_script = SCRIPT.replace(
        "__FRINZ_EMBEDDED_JSON__",
        &serde_json::to_string(&embedded_json)?,
    );

    // Stage the complete bundle before moving files to their chosen names.
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let stage = tempfile::tempdir_in(parent)?;
    let staged: Vec<_> = paths
        .iter()
        .map(|p| stage.path().join(p.file_name().unwrap()))
        .collect();
    fs::write(&staged[1], python_script)?;
    match format {
        ExportFormat::Png => {
            render(&staged[0], &settings, &layers)?;
            if let Some(reference) = settings.hour_reference_mjd {
                let hour_settings = hour_settings(&settings);
                let hour_layers = layers_in_hours(&layers, settings.reference_mjd, reference);
                render(
                    staged.get(2).context("Hour-axis PNG path is missing")?,
                    &hour_settings,
                    &hour_layers,
                )?;
            }
        }
        ExportFormat::Pdf | ExportFormat::Eps => {
            // The editable script creates the selected vector format and its
            // hour-axis companion in one run. No other graph format is emitted.
            let python = std::env::var_os("FRINZTABLE_PYTHON").unwrap_or_else(|| "python3".into());
            let output = Command::new(&python)
                .arg(&staged[1])
                .arg("--output")
                .arg(&staged[0])
                .output()
                .with_context(|| {
                    format!(
                        "Could not start {} to create {} output",
                        PathBuf::from(&python).display(),
                        format.label()
                    )
                })?;
            ensure!(
                output.status.success(),
                "Could not create {} with Matplotlib: {}{}",
                format.label(),
                String::from_utf8_lossy(&output.stderr).trim(),
                if output.stderr.is_empty() {
                    "Install Matplotlib or set FRINZTABLE_PYTHON to a Python executable that has it installed.".to_string()
                } else {
                    String::new()
                }
            );
            for image in std::iter::once(&staged[0]).chain(staged.get(2)) {
                ensure!(
                    image.exists(),
                    "Matplotlib did not create {}",
                    image.display()
                );
            }
        }
    }

    for (staged, final_path) in staged.iter().zip(&paths) {
        fs::rename(staged, final_path)?;
    }
    Ok(())
}

fn hour_settings(settings: &Settings) -> Settings {
    let mut settings = settings.clone();
    if let Some(label) = settings.hour_label.clone() {
        settings.x_label = label;
    }
    settings
}

fn hours_since_reference(x: f64, reference_mjd: Option<f64>, hour_reference_mjd: f64) -> f64 {
    (x + reference_mjd.unwrap_or(0.0) - hour_reference_mjd) * 24.0
}

fn layers_in_hours(
    layers: &[Layer],
    reference_mjd: Option<f64>,
    hour_reference_mjd: f64,
) -> Vec<Layer> {
    layers
        .iter()
        .cloned()
        .map(|mut layer| {
            for point in layer.points.iter_mut().chain(&mut layer.points2) {
                point.0 = hours_since_reference(point.0, reference_mjd, hour_reference_mjd);
            }
            for (x, _, _) in &mut layer.errors {
                *x = hours_since_reference(*x, reference_mjd, hour_reference_mjd);
            }
            layer
        })
        .collect()
}

fn data_tsv(data: &[Datum]) -> Result<String> {
    let mut writer = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_writer(Vec::new());
    writer.write_record(["Row", "Source", "X", "Y", "Y2", "Noise"])?;
    for value in data {
        let cell = |v: Option<f64>| v.map(|v| v.to_string()).unwrap_or_default();
        writer.write_record([
            value.row.to_string(),
            value.source.clone(),
            cell(value.x),
            cell(value.y),
            cell(value.y2),
            cell(value.noise),
        ])?;
    }
    writer.flush()?;
    let bytes = writer.into_inner().map_err(|error| error.into_error())?;
    Ok(String::from_utf8(bytes)?)
}

fn transform(value: f64, log: bool) -> Option<f64> {
    if log {
        (value > 0.0).then(|| value.log10())
    } else {
        Some(value)
    }
}

fn layers(data: &[Datum], settings: &Settings) -> Vec<Layer> {
    let mut grouped: BTreeMap<&str, Vec<&Datum>> = BTreeMap::new();
    for value in data {
        grouped.entry(&value.source).or_default().push(value);
    }
    let all_values: Vec<_> = data
        .iter()
        .filter_map(|p| p.y.and_then(|y| transform(y, settings.log_y)))
        .collect();
    let bins = histogram(&all_values, settings.bins);
    grouped
        .into_iter()
        .enumerate()
        .map(|(i, (source, group))| {
            let mut layer = Layer {
                source: source.into(),
                points: Vec::new(),
                points2: Vec::new(),
                errors: Vec::new(),
                bars: Vec::new(),
            };
            match settings.mode {
                PlotMode::Series => {
                    for point in group {
                        if let (Some(x), Some(y)) =
                            (point.x, point.y.and_then(|y| transform(y, settings.log_y)))
                        {
                            layer.points.push((x, y));
                            if settings.noise_bars {
                                if let (Some(raw), Some(noise)) = (point.y, point.noise) {
                                    if let (Some(low), Some(high)) = (
                                        transform(raw - noise, settings.log_y),
                                        transform(raw + noise, settings.log_y),
                                    ) {
                                        layer.errors.push((x, low, high));
                                    }
                                }
                            }
                        }
                        if let (Some(x), Some(y)) =
                            (point.x, point.y2.and_then(|y| transform(y, settings.log_y)))
                        {
                            layer.points2.push((x, y));
                        }
                    }
                    layer.points.sort_by(|a, b| a.0.total_cmp(&b.0));
                    layer.points2.sort_by(|a, b| a.0.total_cmp(&b.0));
                }
                PlotMode::Histogram => {
                    if !bins.is_empty() {
                        let lower = bins[0].0 - bins[0].1 / 2.0;
                        let width = bins[0].1;
                        let mut counts = vec![0; bins.len()];
                        for value in group
                            .iter()
                            .filter_map(|p| p.y.and_then(|y| transform(y, settings.log_y)))
                        {
                            counts[(((value - lower) / width) as usize).min(bins.len() - 1)] += 1;
                        }
                        if counts.iter().any(|&n| n > 0) {
                            layer.bars = bins
                                .iter()
                                .zip(counts)
                                .map(|(&(center, width, _), n)| (center, width, n as f64))
                                .collect();
                        }
                    }
                }
                PlotMode::Sources => {
                    if let Some(s) = Statistics::calculate(group.iter().map(|p| p.y)) {
                        if let Some(mean) = transform(s.mean, settings.log_y) {
                            layer.points.push((i as f64, mean));
                            if let Some(error) = s.standard_error {
                                if let (Some(low), Some(high)) = (
                                    transform(s.mean - error, settings.log_y),
                                    transform(s.mean + error, settings.log_y),
                                ) {
                                    layer.errors.push((i as f64, low, high));
                                }
                            }
                        }
                    }
                }
            }
            layer
        })
        .collect()
}

fn range(values: impl Iterator<Item = f64>, zero: bool) -> std::ops::Range<f64> {
    let values: Vec<_> = values.filter(|v| v.is_finite()).collect();
    let mut min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let mut max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !min.is_finite() {
        min = 0.0;
        max = 1.0;
    }
    if zero {
        min = min.min(0.0);
    }
    let padding = if min == max {
        (min.abs() * 0.05).max(1e-6)
    } else {
        (max - min) * 0.07
    };
    min - padding..max + padding
}

fn legend_marker<DB: DrawingBackend>(
    (x, y): (i32, i32),
    index: usize,
) -> DynElement<'static, DB, (i32, i32)> {
    let size = 6;
    match index % 5 {
        1 => {
            Rectangle::new([(x - size, y - size), (x + size, y + size)], WHITE.filled()).into_dyn()
        }
        2 => Polygon::new(
            vec![(x, y - size), (x + size, y), (x, y + size), (x - size, y)],
            WHITE.filled(),
        )
        .into_dyn(),
        3 => Cross::new((x, y), size, WHITE.stroke_width(2)).into_dyn(),
        4 => PathElement::new(
            vec![
                (x - size, y),
                (x + size, y),
                (x, y),
                (x, y - size),
                (x, y + size),
            ],
            WHITE.stroke_width(2),
        )
        .into_dyn(),
        _ => Circle::new((x, y), size, WHITE.filled()).into_dyn(),
    }
}

fn render(path: &Path, settings: &Settings, layers: &[Layer]) -> Result<()> {
    let root = BitMapBackend::new(path, IMAGE_SIZE).into_drawing_area();
    root.fill(&BLACK)?;
    let x_range = range(
        layers.iter().flat_map(|l| {
            l.points.iter().chain(&l.points2).map(|p| p.0).chain(
                l.bars
                    .iter()
                    .flat_map(|p| [p.0 - p.1 / 2.0, p.0 + p.1 / 2.0]),
            )
        }),
        false,
    );
    let y_range = range(
        layers.iter().flat_map(|l| {
            l.points
                .iter()
                .map(|p| p.1)
                .chain(l.errors.iter().flat_map(|p| [p.1, p.2]))
                .chain(l.bars.iter().map(|p| p.2))
        }),
        settings.mode == PlotMode::Histogram,
    );
    let secondary_active = settings.mode == PlotMode::Series
        && settings.y2_label.is_some()
        && layers.iter().any(|layer| !layer.points2.is_empty());
    let secondary_range = if secondary_active {
        range(
            layers
                .iter()
                .flat_map(|layer| layer.points2.iter().map(|p| p.1)),
            false,
        )
    } else {
        y_range.clone()
    };
    let mut chart_builder = ChartBuilder::on(&root);
    chart_builder
        .caption(
            &settings.title,
            ("sans-serif", 36).into_font().color(&WHITE),
        )
        .margin(45)
        .x_label_area_size(90)
        .y_label_area_size(150)
        .right_y_label_area_size(if secondary_active { 150 } else { 0 });
    let mut chart = chart_builder
        .build_cartesian_2d(x_range.clone(), y_range.clone())?
        .set_secondary_coord(x_range, secondary_range);
    let mut mesh = chart.configure_mesh();
    let x_label = if settings.mode == PlotMode::Histogram {
        &settings.y_label
    } else if settings.mode == PlotMode::Sources {
        "Source (mean ± standard error)"
    } else {
        &settings.x_label
    };
    let y_label = if settings.mode == PlotMode::Histogram {
        "Count"
    } else {
        &settings.y_label
    };
    let formatter = |v: &f64| {
        if settings.mode == PlotMode::Sources {
            let index = v.round();
            if (*v - index).abs() < 0.01 && index >= 0.0 {
                layers
                    .get(index as usize)
                    .map(|l| l.source.clone())
                    .unwrap_or_default()
            } else {
                String::new()
            }
        } else {
            super::stats::display(*v)
        }
    };
    mesh.x_desc(x_label)
        .y_desc(y_label)
        .axis_style(WHITE)
        .bold_line_style(WHITE.mix(0.3))
        .light_line_style(WHITE.mix(0.12))
        .label_style(("sans-serif", 22).into_font().color(&WHITE))
        .axis_desc_style(("sans-serif", 28).into_font().color(&WHITE))
        .x_label_formatter(&formatter);
    if settings.mode == PlotMode::Sources {
        mesh.x_labels(layers.len().max(1));
    }
    mesh.draw()?;
    if secondary_active {
        chart
            .configure_secondary_axes()
            .y_desc(settings.y2_label.as_deref().unwrap_or(""))
            .axis_style(WHITE)
            .label_style(("sans-serif", 22).into_font().color(&WHITE))
            .axis_desc_style(("sans-serif", 28).into_font().color(&WHITE))
            .draw()?;
    }
    for layer in layers {
        let index = settings
            .sources
            .iter()
            .position(|s| s == &layer.source)
            .unwrap_or(0);
        let points = if settings.mode == PlotMode::Histogram {
            let mut outline = Vec::new();
            let mut tops = Vec::new();
            if let Some(&(x, width, _)) = layer.bars.first() {
                outline.push((x - width / 2.0, 0.0));
            }
            for &(x, width, height) in &layer.bars {
                outline.extend([(x - width / 2.0, height), (x + width / 2.0, height)]);
                if height > 0.0 {
                    tops.push((x, height));
                }
            }
            if let Some(&(x, width, _)) = layer.bars.last() {
                outline.push((x + width / 2.0, 0.0));
            }
            chart.draw_series(LineSeries::new(outline, WHITE.stroke_width(2)))?;
            tops
        } else {
            layer.points.clone()
        };
        if !points.is_empty() {
            let series_style = if settings.mode == PlotMode::Series {
                settings.series_style
            } else {
                SeriesStyle::Scatter
            };
            if series_style == SeriesStyle::Linear {
                chart
                    .draw_series(LineSeries::new(points.clone(), WHITE.stroke_width(2)))?
                    .label(&layer.source)
                    .legend(|(x, y)| {
                        PathElement::new(vec![(x, y), (x + 20, y)], WHITE.stroke_width(2))
                    });
            } else {
                if series_style == SeriesStyle::ScatterDotted {
                    chart.draw_series(DashedLineSeries::new(
                        points.clone().into_iter(),
                        1,
                        5,
                        WHITE.stroke_width(2),
                    ))?;
                }
                match index % 5 {
                    1 => {
                        chart.draw_series(points.iter().map(|&p| {
                            EmptyElement::at(p) + Rectangle::new([(-6, -6), (6, 6)], WHITE.filled())
                        }))?;
                    }
                    2 => {
                        chart.draw_series(points.iter().map(|&p| {
                            EmptyElement::at(p)
                                + Polygon::new(
                                    vec![(0, -6), (6, 0), (0, 6), (-6, 0)],
                                    WHITE.filled(),
                                )
                        }))?;
                    }
                    3 => {
                        chart.draw_series(
                            points
                                .iter()
                                .map(|&p| Cross::new(p, 6, WHITE.stroke_width(2))),
                        )?;
                    }
                    4 => {
                        chart.draw_series(points.iter().map(|&p| {
                            EmptyElement::at(p)
                                + PathElement::new(
                                    vec![(-6, 0), (6, 0), (0, 0), (0, -6), (0, 6)],
                                    WHITE.stroke_width(2),
                                )
                        }))?;
                    }
                    _ => {
                        chart.draw_series(
                            points.iter().map(|&p| Circle::new(p, 6, WHITE.filled())),
                        )?;
                    }
                }
                chart
                    .draw_series(std::iter::once(EmptyElement::at(points[0])))?
                    .label(&layer.source)
                    .legend(move |(x, y)| legend_marker((x + 9, y), index));
            }
            chart.draw_series(layer.errors.iter().map(|&(x, low, high)| {
                PathElement::new(vec![(x, low), (x, high)], WHITE.mix(0.65).stroke_width(2))
            }))?;
        }
        if secondary_active && !layer.points2.is_empty() {
            if settings.series_style == SeriesStyle::Linear {
                chart
                    .draw_secondary_series(LineSeries::new(
                        layer.points2.clone(),
                        WHITE.mix(0.7).stroke_width(2),
                    ))?
                    .label(format!("{} [Y2]", layer.source))
                    .legend(|(x, y)| {
                        PathElement::new(vec![(x, y), (x + 20, y)], WHITE.mix(0.7).stroke_width(2))
                    });
            } else {
                if settings.series_style == SeriesStyle::ScatterDotted {
                    chart.draw_secondary_series(DashedLineSeries::new(
                        layer.points2.clone().into_iter(),
                        1,
                        5,
                        WHITE.mix(0.7).stroke_width(2),
                    ))?;
                }
                chart
                    .draw_secondary_series(
                        layer
                            .points2
                            .iter()
                            .map(|&point| Circle::new(point, 5, WHITE.stroke_width(2))),
                    )?
                    .label(format!("{} [Y2]", layer.source))
                    .legend(move |(x, y)| Circle::new((x + 9, y), 6, WHITE.stroke_width(2)));
            }
        }
    }
    chart
        .configure_series_labels()
        .background_style(BLACK.mix(0.85))
        .border_style(WHITE.mix(0.5))
        .label_font(("sans-serif", 23).into_font().color(&WHITE))
        .draw()?;
    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn png_export_writes_only_png_and_script_and_embeds_enabled_rows() {
        let mut table =
            Table::parse("Source\tAmp\tNoise-level\nA\t4\t2\nB\t6\t3\nA\t8\t2", false).unwrap();
        table.plot_enabled[1] = false;
        let directory = tempfile::tempdir().unwrap();
        for mode in [PlotMode::Series, PlotMode::Histogram, PlotMode::Sources] {
            let path = directory.path().join(format!("{mode:?}.png"));
            let options = PlotOptions {
                mode,
                y: 1,
                ..Default::default()
            };
            export(&path, &table, &table.plotted_rows(&[0, 1, 2]), &options).unwrap();
            let image = image::image_dimensions(&path).unwrap();
            assert_eq!(image, IMAGE_SIZE);
            let script_path = path.with_extension("py");
            assert!(script_path.exists());
            assert!(!path.with_extension("pdf").exists());
            assert!(!path.with_extension("eps").exists());
            let script = fs::read_to_string(&script_path).unwrap();
            assert!(script.contains("FIGSIZE = (9, 6)"));
            assert!(script.contains("EMBEDDED = json.loads("));
            assert!(script.contains(r"1\\tA\\t1\\t4\\t\\t2"));
            assert!(!script.contains(r"2\\tB\\t2\\t6\\t\\t3"));
            assert!(export(&path, &table, &[0], &options).is_err());
        }
    }

    #[test]
    fn time_series_export_writes_only_the_selected_format_and_hour_companion() {
        let table = Table::parse("Source\tMJD\tAmp\nA\t60000.25\t4\nA\t60001\t8", false).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let options = PlotOptions {
            y: 2,
            reference_mjd: Some(60000.25),
            ..Default::default()
        };

        for extension in ["png", "pdf", "eps"] {
            let path = directory
                .path()
                .join(format!("time_{extension}.{extension}"));
            export(&path, &table, &[0, 1], &options).unwrap();
            let hour_path = directory
                .path()
                .join(format!("time_{extension}_hour.{extension}"));
            assert!(path.exists(), "missing {}", path.display());
            assert!(hour_path.exists(), "missing {}", hour_path.display());
            assert!(path.with_extension("py").exists());
            if extension == "png" {
                assert_eq!(image::image_dimensions(&hour_path).unwrap(), IMAGE_SIZE);
            }
            for unselected in ["png", "pdf", "eps"]
                .into_iter()
                .filter(|candidate| *candidate != extension)
            {
                assert!(!path.with_extension(unselected).exists());
                assert!(!directory
                    .path()
                    .join(format!("time_{extension}_hour.{unselected}"))
                    .exists());
            }
            let script = fs::read_to_string(path.with_extension("py")).unwrap();
            assert!(script.contains("time_axis"));
            assert!(script.contains("hour_reference_mjd"));
            assert!(!script.contains("--time-axis"));
        }
    }
}
