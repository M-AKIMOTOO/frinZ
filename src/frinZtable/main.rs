mod app;
mod data;
mod desktop;
mod formula;
mod help;
mod plot_export;
mod plots;
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
    /// Interpret a headerless 21-column file as frequency-mode output.
    #[arg(long)]
    frequency: bool,
    /// Print statistics as JSON without opening a GUI.
    #[arg(long)]
    summary: bool,
    /// Filter by source names; repeat the option or separate names with commas.
    #[arg(long, value_delimiter = ',')]
    source: Vec<String>,
    /// Write 9:6 PNG, PDF and EPS plus a self-contained Python script; time series include MJD and hour axes.
    #[arg(long, conflicts_with_all = ["summary", "screenshot"])]
    export_plot: Option<PathBuf>,
    /// Plot X column name; default is Epoch/MJD.
    #[arg(long)]
    x: Option<String>,
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
    /// Start with a large graph view; use "Back to table" to restore the spreadsheet.
    #[arg(long, conflicts_with_all = ["summary", "export_plot", "install_desktop"])]
    expand_graph: bool,
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
    let table = path
        .as_ref()
        .map(|p| {
            if args.frequency {
                Table::parse(&std::fs::read_to_string(p)?, true)
            } else {
                Table::load(p)
            }
        })
        .transpose()?
        .unwrap_or_default();
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
            "Graph set (including MJD/hour axes when available) and self-contained editable script saved: {}",
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
    eframe::run_native(
        "frinZtable",
        options,
        Box::new(move |cc| {
            let mut app = app::TableApp::new(cc, table, path, args.source, args.screenshot);
            if args.manual {
                app.show_manual();
            }
            if args.expand_graph {
                app.expand_graph();
            }
            Ok(Box::new(app))
        }),
    )
    .map_err(|err| anyhow::anyhow!("GUI could not start: {err}"))
}
