mod app;
mod data;
mod desktop;
mod formula;
mod help;
mod plot_export;
mod plots;
mod project;
mod stats;

use anyhow::Result;
use clap::Parser;
use data::{Filter, Table};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "frinZtable",
    version,
    about = "Spreadsheet GUI for frinZ analysis TSV output"
)]
struct Arguments {
    /// frinZ TSV file to open (also accepts --input).
    file: Option<PathBuf>,
    #[arg(short, long, conflicts_with = "file")]
    input: Option<PathBuf>,
    /// Interpret a headerless 21-column file as frequency-mode output (headers are detected automatically).
    #[arg(long)]
    frequency: bool,
    /// Print statistics as JSON without opening a GUI.
    #[arg(long)]
    summary: bool,
    /// Filter by source names; repeat the option or separate names with commas.
    #[arg(long, value_delimiter = ',')]
    source: Vec<String>,
    /// Choose .png, .pdf, or .eps; writes that format, an editable script, and an automatic time-axis companion when available.
    #[arg(long, conflicts_with_all = ["summary", "screenshot"])]
    export_plot: Option<PathBuf>,
    /// Plot X column name; default is Epoch/MJD, then Time, then row number.
    #[arg(long)]
    x: Option<String>,
    /// X axis scale: auto, hours, or date.
    #[arg(long, value_parser = ["auto", "hours", "date"], default_value = "auto")]
    x_scale: String,
    /// UTC date/time or MJD origin required to date-convert a Time column in seconds.
    #[arg(long)]
    date_origin: Option<String>,
    /// Plot Y column name.
    #[arg(long, default_value = "Amp")]
    y: String,
    /// Restrict the primary Y series to this source (series mode only).
    #[arg(long)]
    y_source: Option<String>,
    /// Plot a second numeric column on the right Y axis; may match --y when sources differ.
    #[arg(long)]
    y2: Option<String>,
    /// Restrict the secondary Y series to this source (series mode only).
    #[arg(long)]
    y2_source: Option<String>,
    /// Subtract this reference from Epoch/MJD for the X axis; defaults to the first whole MJD.
    #[arg(long, conflicts_with = "absolute_mjd")]
    reference_mjd: Option<f64>,
    /// Keep the absolute Epoch/MJD values on the X axis.
    #[arg(long)]
    absolute_mjd: bool,
    #[arg(long, value_parser = ["series", "histogram", "sources"], default_value = "series")]
    plot_mode: String,
    /// Time-series style: scatter, linear, or scatter-dotted.
    #[arg(long, value_parser = ["scatter", "linear", "scatter-dotted"], default_value = "scatter")]
    plot_style: String,
    #[arg(long)]
    log_y: bool,
    /// Open the frinZmain processing catalogue at startup.
    #[arg(long)]
    manual: bool,
    /// Start in plot-only view (same as --plot-only).
    #[arg(long, conflicts_with_all = ["summary", "export_plot", "install_desktop", "spreadsheet_only"])]
    expand_graph: bool,
    /// Start in spreadsheet-only view.
    #[arg(long, conflicts_with_all = ["plot_only", "expand_graph"])]
    spreadsheet_only: bool,
    /// Start in plot-only view.
    #[arg(long, conflicts_with = "expand_graph")]
    plot_only: bool,
    /// Register frinZtable in the Linux application menu and TSV "Open With" list.
    #[arg(long, conflicts_with_all = ["file", "input", "summary", "export_plot", "screenshot", "manual"])]
    install_desktop: bool,
    /// Save the GUI to a PNG and exit (requires a desktop display).
    #[arg(long, conflicts_with = "summary")]
    screenshot: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    if args.install_desktop {
        println!("Application registered: {}", desktop::install()?.display());
        return Ok(());
    }
    let path = args.input.or(args.file);
    let project_path = path
        .as_ref()
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("frztable"))
        })
        .cloned();
    let saved_project = project_path
        .as_ref()
        .map(|path| project::Project::load(path))
        .transpose()?;
    let table = if let Some(project) = &saved_project {
        project.table.clone()
    } else if let Some(path) = &path {
        if args.frequency {
            Table::parse(&std::fs::read_to_string(path)?, true)?
        } else {
            Table::load(path)?
        }
    } else {
        Table::default()
    };
    if args.summary {
        anyhow::ensure!(path.is_some(), "--summary requires an input file");
        let rows = table.view(
            &Filter {
                sources: (!args.source.is_empty()).then(|| args.source.iter().cloned().collect()),
                ..Default::default()
            },
            None,
        );
        let sources: Vec<_> = table
            .numeric_columns()
            .into_iter()
            .map(|c| {
                serde_json::json!({
                    "column": table.columns[c].name,
                    "sources": table.source_summaries(&rows, c),
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "rows": rows.len(), "columns": table.summaries(&rows), "by_source": sources,
            }))?
        );
        return Ok(());
    }
    if let Some(output_path) = args.export_plot {
        let output_path = if output_path.is_relative() {
            path.as_ref()
                .and_then(|input| input.parent())
                .filter(|parent| !parent.as_os_str().is_empty())
                .map(|parent| parent.join(&output_path))
                .unwrap_or(output_path)
        } else {
            output_path
        };
        let rows = table.view(
            &Filter {
                sources: (!args.source.is_empty()).then(|| args.source.iter().cloned().collect()),
                ..Default::default()
            },
            None,
        );
        let options = plots::PlotOptions {
            x: args
                .x
                .as_ref()
                .map(|name| {
                    table
                        .column(name)
                        .ok_or_else(|| anyhow::anyhow!("Unknown X column: {name}"))
                })
                .transpose()?,
            x_axis_scale: match args.x_scale.as_str() {
                "hours" => plots::XAxisScale::Hours,
                "date" => plots::XAxisScale::Date,
                _ => plots::XAxisScale::Automatic,
            },
            date_origin: args.date_origin.unwrap_or_default(),
            y: table
                .column(&args.y)
                .ok_or_else(|| anyhow::anyhow!("Unknown Y column: {}", args.y))?,
            y_source: args.y_source,
            y2: args
                .y2
                .as_ref()
                .map(|name| {
                    table
                        .column(name)
                        .ok_or_else(|| anyhow::anyhow!("Unknown second Y column: {name}"))
                })
                .transpose()?,
            y2_source: args.y2_source,
            reference_mjd: if args.absolute_mjd {
                None
            } else {
                args.reference_mjd
                    .or_else(|| plots::first_mjd_reference(&table))
            },
            mode: match args.plot_mode.as_str() {
                "histogram" => plots::PlotMode::Histogram,
                "sources" => plots::PlotMode::Sources,
                _ => plots::PlotMode::Series,
            },
            series_style: match args.plot_style.as_str() {
                "linear" => plots::SeriesStyle::Linear,
                "scatter-dotted" => plots::SeriesStyle::ScatterDotted,
                _ => plots::SeriesStyle::Scatter,
            },
            log_y: args.log_y,
            ..Default::default()
        };
        plot_export::export(&output_path, &table, &table.plotted_rows(&rows), &options)?;
        println!(
            "Graph and self-contained editable Python script saved: {}",
            output_path.display()
        );
        return Ok(());
    }
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_app_id("frinZtable")
            .with_icon(eframe::icon_data::from_png_bytes(include_bytes!(
                "icon.png"
            ))?)
            .with_inner_size([1480.0, 940.0])
            .with_min_inner_size([960.0, 680.0]),
        ..Default::default()
    };
    let data_path = saved_project
        .as_ref()
        .and_then(|project| project.data_path.clone())
        .or_else(|| path.filter(|path| project_path.as_ref() != Some(path)));
    eframe::run_native(
        "frinZtable",
        options,
        Box::new(move |cc| {
            let mut app = app::TableApp::new(cc, table, data_path, args.source, args.screenshot);
            if let (Some(project), Some(project_path)) = (saved_project, project_path) {
                app.restore_project(project, project_path);
            }
            if args.manual {
                app.show_manual();
            }
            if args.expand_graph {
                app.expand_graph();
            }
            if args.spreadsheet_only {
                app.set_view_mode(app::ViewMode::Spreadsheet);
            } else if args.plot_only {
                app.set_view_mode(app::ViewMode::Plot);
            }
            Ok(Box::new(app))
        }),
    )
    .map_err(|err| anyhow::anyhow!("GUI could not start: {err}"))
}
