use super::{
    data::{ColumnSummary, Filter, Sort, SourceSummary, Table, column_letter, number},
    help::HelpBrowser,
    plot_export::{self, ExportFormat},
    plots::{self, PlotMode, PlotOptions, SeriesStyle},
    project::{self, Project},
    stats::{Statistics, display},
};
use anyhow::Result;
use eframe::egui::{self, Color32, Key, RichText, ScrollArea, Sense};
use std::{fs, path::PathBuf};

const CELL_WIDTH: f32 = 119.0;
const ROW_HEIGHT: f32 = 25.0;

#[derive(Clone, Copy, Debug)]
struct Cell {
    row: usize,
    column: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewMode {
    Spreadsheet,
    Plot,
    #[default]
    Combined,
}

enum PendingAction {
    Open(PathBuf),
    Close,
}

pub struct TableApp {
    table: Table,
    path: Option<PathBuf>,
    filter: Filter,
    min_text: String,
    max_text: String,
    sort: Option<Sort>,
    view: Vec<usize>,
    numeric: Vec<usize>,
    summaries: Vec<ColumnSummary>,
    groups: Vec<SourceSummary>,
    selected: Option<Cell>,
    anchor: Option<Cell>,
    edit: String,
    edit_focus: bool,
    statistics_selected: bool,
    all_statistics: bool,
    group_sort: usize,
    group_descending: bool,
    view_mode: ViewMode,
    plot: PlotOptions,
    export_format: ExportFormat,
    undo: Vec<Table>,
    redo: Vec<Table>,
    dirty: bool,
    project_dirty: bool,
    project_path: Option<PathBuf>,
    message: String,
    formula_window: bool,
    formula_name: String,
    formula_unit: String,
    formula: String,
    pending: Option<PendingAction>,
    allow_close: bool,
    screenshot: Option<PathBuf>,
    frames: usize,
    help: HelpBrowser,
}

impl TableApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        table: Table,
        path: Option<PathBuf>,
        sources: Vec<String>,
        screenshot: Option<PathBuf>,
    ) -> Self {
        dark_theme(&cc.egui_ctx);
        // Use an installed Japanese font when available; no machine-specific
        // font is embedded or required to compile the application.
        for path in [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf",
        ] {
            if let Ok(bytes) = fs::read(path) {
                let mut fonts = egui::FontDefinitions::default();
                fonts
                    .font_data
                    .insert("cjk".into(), egui::FontData::from_owned(bytes).into());
                for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                    fonts.families.entry(family).or_default().push("cjk".into());
                }
                cc.egui_ctx.set_fonts(fonts);
                break;
            }
        }
        Self::from_table(table, path, sources, screenshot)
    }

    fn from_table(
        table: Table,
        path: Option<PathBuf>,
        sources: Vec<String>,
        screenshot: Option<PathBuf>,
    ) -> Self {
        let mut app = Self {
            table, path, filter: Filter { sources: (!sources.is_empty()).then(|| sources.into_iter().collect()), ..Default::default() },
            min_text: String::new(), max_text: String::new(), sort: None, view: Vec::new(), numeric: Vec::new(),
            summaries: Vec::new(), groups: Vec::new(), selected: None, anchor: None,
            edit: String::new(), edit_focus: false, statistics_selected: false, all_statistics: true,
            group_sort: 0, group_descending: false, view_mode: ViewMode::Combined, plot: PlotOptions::default(), export_format: ExportFormat::default(), undo: Vec::new(), redo: Vec::new(),
            dirty: false, project_dirty: false, project_path: None,
            message: "Click a cell to inspect it; double-click or F2 to edit. Shift-click selects a range.".into(),
            formula_window: false, formula_name: "AmpToNoise".into(), formula_unit: String::new(),
            formula: "Amp / [Noise-level]".into(), pending: None, allow_close: false,
            screenshot, frames: 0,
            help: HelpBrowser::default(),
        };
        app.plot.y = app.table.column("Amp").unwrap_or(0);
        app.set_default_mjd_reference();
        app.rebuild();
        app
    }

    fn set_default_mjd_reference(&mut self) {
        self.plot.reference_mjd = plots::first_mjd_reference(&self.table);
        self.plot.reference_text = self
            .plot
            .reference_mjd
            .map(|mjd| format!("{mjd:.0}"))
            .unwrap_or_default();
    }

    pub fn show_manual(&mut self) {
        self.help.open = true;
    }

    pub fn expand_graph(&mut self) {
        self.view_mode = ViewMode::Plot;
    }

    pub fn set_view_mode(&mut self, mode: ViewMode) {
        self.view_mode = mode;
    }

    pub fn restore_project(&mut self, project: Project, project_path: PathBuf) {
        self.table = project.table;
        self.path = project.data_path;
        self.filter = project.filter;
        self.min_text = project.min_text;
        self.max_text = project.max_text;
        self.sort = project.sort;
        self.statistics_selected = project.statistics_selected;
        self.all_statistics = project.all_statistics;
        self.group_sort = project.group_sort;
        self.group_descending = project.group_descending;
        self.plot = project.plot;
        self.export_format = project.export_format;
        self.view_mode = project.view_mode;
        self.project_path = Some(project_path);
        self.undo.clear();
        self.redo.clear();
        self.dirty = false;
        self.project_dirty = false;
        self.rebuild();
        let locate = |cell: Option<[usize; 2]>, view: &[usize]| {
            cell.and_then(|[row, column]| {
                (column < self.table.columns.len())
                    .then(|| {
                        view.iter()
                            .position(|&visible_row| visible_row == row)
                            .map(|row| Cell { row, column })
                    })
                    .flatten()
            })
        };
        self.selected = locate(project.selected, &self.view);
        self.anchor = locate(project.anchor, &self.view);
        if let Some(cell) = self.selected {
            self.edit = self.table.rows[self.view[cell.row]][cell.column].clone();
            self.refresh_statistics();
        }
        self.message = format!(
            "Resumed project: {} rows, {} columns.",
            self.table.rows.len(),
            self.table.columns.len()
        );
    }

    fn save_project(&mut self, save_as: bool) -> bool {
        if self.table.columns.is_empty() {
            self.message = "Open a table before saving a project.".into();
            return false;
        }
        if !self.commit_pending_edit() {
            return false;
        }
        let path = if !save_as {
            self.project_path.clone()
        } else {
            None
        }
        .or_else(|| {
            let stem = self
                .path
                .as_ref()
                .and_then(|path| path.file_stem())
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| "frinZ".into());
            let mut dialog = rfd::FileDialog::new()
                .add_filter("frinZtable project", &["frztable"])
                .set_file_name(format!("{stem}.frztable"));
            if let Some(directory) = self
                .project_path
                .as_ref()
                .or(self.path.as_ref())
                .and_then(|path| path.parent())
                .filter(|path| !path.as_os_str().is_empty())
            {
                dialog = dialog.set_directory(directory);
            }
            dialog.save_file()
        });
        let Some(path) = path else {
            return false;
        };
        let path = if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| !extension.eq_ignore_ascii_case("frztable"))
        {
            path.with_extension("frztable")
        } else {
            path
        };
        let project = Project {
            version: project::CURRENT_VERSION,
            table: self.table.clone(),
            data_path: self.path.clone(),
            filter: self.filter.clone(),
            min_text: self.min_text.clone(),
            max_text: self.max_text.clone(),
            sort: self.sort,
            statistics_selected: self.statistics_selected,
            all_statistics: self.all_statistics,
            group_sort: self.group_sort,
            group_descending: self.group_descending,
            plot: self.plot.clone(),
            export_format: self.export_format,
            view_mode: self.view_mode,
            selected: self.selected.map(|cell| [self.view[cell.row], cell.column]),
            anchor: self.anchor.map(|cell| [self.view[cell.row], cell.column]),
        };
        match project.save(&path) {
            Ok(()) => {
                self.project_path = Some(path.clone());
                self.project_dirty = false;
                self.dirty = false;
                self.message = format!("Project saved: {}", path.display());
                true
            }
            Err(error) => {
                self.message = format!("Project save failed: {error}");
                false
            }
        }
    }

    fn commit_pending_edit(&mut self) -> bool {
        let Some(cell) = self.selected else {
            return true;
        };
        let Some(&row) = self.view.get(cell.row) else {
            return true;
        };
        if self.table.rows[row][cell.column] == self.edit {
            return true;
        }
        if self
            .table
            .calculated
            .iter()
            .any(|derived| derived.column == cell.column)
        {
            self.message = "Calculated columns are read-only.".into();
            return false;
        }
        if self.edit.contains(['\t', '\n', '\r']) {
            self.message = "Cell edits cannot contain tabs or newlines.".into();
            return false;
        }
        let selected = self.selected.map(|cell| [self.view[cell.row], cell.column]);
        let anchor = self.anchor.map(|cell| [self.view[cell.row], cell.column]);
        let before = self.table.clone();
        self.table.rows[row][cell.column] = self.edit.clone();
        self.table.recalculate();
        self.record(before);
        self.rebuild();
        let locate = |cell: Option<[usize; 2]>, view: &[usize]| {
            cell.and_then(|[row, column]| {
                (column < self.table.columns.len())
                    .then(|| {
                        view.iter()
                            .position(|&visible_row| visible_row == row)
                            .map(|row| Cell { row, column })
                    })
                    .flatten()
            })
        };
        self.selected = locate(selected, &self.view);
        self.anchor = locate(anchor, &self.view);
        if let Some(cell) = self.selected {
            self.edit = self.table.rows[self.view[cell.row]][cell.column].clone();
        }
        self.refresh_statistics();
        true
    }

    fn set_plot_rows(&mut self, rows: &[usize], enabled: bool) {
        if rows.iter().all(|&r| self.table.plot_enabled[r] == enabled) {
            return;
        }
        let before = self.table.clone();
        for &r in rows {
            self.table.plot_enabled[r] = enabled;
        }
        self.record(before);
        self.message = format!(
            "{} rows {} for plotting. Statistics retain their selected scope.",
            rows.len(),
            if enabled { "enabled" } else { "disabled" }
        );
    }

    fn export_graph(&mut self) {
        let rows = self.table.plotted_rows(&self.view);
        if rows.is_empty() {
            self.message = "Enable at least one row for plotting.".into();
            return;
        }
        let stem = self
            .path
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "frinZ".into());
        let extension = self.export_format.extension();
        let filter = format!(
            "{} graph + editable Python script",
            self.export_format.label()
        );
        let mut dialog = rfd::FileDialog::new()
            .add_filter(&filter, &[extension])
            .set_file_name(format!(
                "{stem}_{}_plot.{extension}",
                self.table.columns[self.plot.y].name
            ));
        if let Some(directory) = self
            .path
            .as_ref()
            .and_then(|path| path.parent())
            .filter(|path| !path.as_os_str().is_empty())
        {
            dialog = dialog.set_directory(directory);
        }
        let Some(path) = dialog.save_file() else {
            return;
        };
        let path = path.with_extension(extension);
        match plot_export::export(&path, &self.table, &rows, &self.plot) {
            Ok(()) => {
                self.message = format!(
                    "Saved {} graph output and editable Python script: {}",
                    self.export_format.label(),
                    path.display()
                )
            }
            Err(err) => self.message = format!("Graph export failed: {err}"),
        }
    }

    fn rebuild(&mut self) {
        self.numeric = self.table.numeric_columns();
        let width = self.table.columns.len();
        let sources = self.table.sources();
        if self.plot.x.is_some_and(|c| c >= width) {
            self.plot.x = None;
        }
        if self.plot.y2.is_some_and(|c| !self.numeric.contains(&c)) {
            self.plot.y2 = None;
        }
        if self
            .plot
            .y_source
            .as_ref()
            .is_some_and(|source| !sources.contains(source))
        {
            self.plot.y_source = None;
        }
        if self
            .plot
            .y2_source
            .as_ref()
            .is_some_and(|source| !sources.contains(source))
        {
            self.plot.y2_source = None;
        }
        if self.filter.numeric_column.is_some_and(|c| c >= width) {
            self.filter.numeric_column = None;
        }
        if self.sort.is_some_and(|s| s.column >= width) {
            self.sort = None;
        }
        if self.plot.y >= self.table.columns.len() {
            self.plot.y = self.numeric.first().copied().unwrap_or(0);
        }
        self.view = self.table.view(&self.filter, self.sort);
        self.selected = None;
        self.anchor = None;
        self.edit.clear();
        self.plot.reset = true;
        self.refresh_statistics();
    }

    fn selected_rows(&self) -> Vec<usize> {
        match (self.anchor, self.selected) {
            (Some(a), Some(b)) if !self.view.is_empty() => {
                self.view[a.row.min(b.row)..=a.row.max(b.row)].to_vec()
            }
            _ => Vec::new(),
        }
    }

    fn refresh_statistics(&mut self) {
        let rows = if self.statistics_selected {
            self.selected_rows()
        } else {
            self.view.clone()
        };
        self.summaries = self.table.summaries(&rows);
        self.groups = if self.table.columns.is_empty() {
            Vec::new()
        } else {
            self.table.source_summaries(&rows, self.plot.y)
        };
        self.sort_groups();
    }

    fn sort_groups(&mut self) {
        let key = self.group_sort;
        self.groups.sort_by(|a, b| {
            let cmp = if key == 0 {
                a.source.cmp(&b.source)
            } else if key == 1 {
                a.rows.cmp(&b.rows)
            } else {
                let value = |s: &SourceSummary| {
                    s.statistics.as_ref().map(|s| match key {
                        2 => s.mean,
                        3 => s.median,
                        4 => s.sample_stddev.unwrap_or(f64::NAN),
                        5 => s.min,
                        _ => s.max,
                    })
                };
                match (value(a), value(b)) {
                    (Some(a), Some(b)) => a.total_cmp(&b),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    _ => std::cmp::Ordering::Equal,
                }
            };
            if self.group_descending {
                cmp.reverse()
            } else {
                cmp
            }
        });
    }

    fn select(&mut self, cell: Cell, extend: bool) {
        let previous_plot_column = self.plot.y;
        self.selected = Some(cell);
        if !extend || self.anchor.is_none() {
            self.anchor = Some(cell);
        }
        self.edit = self.table.rows[self.view[cell.row]][cell.column].clone();
        if self.numeric.contains(&cell.column) && self.plot.y != cell.column {
            self.plot.y = cell.column;
            self.plot.reset = true;
        }
        self.refresh_statistics();
        if self.statistics_selected || self.plot.y != previous_plot_column {
            self.project_dirty = true;
        }
    }

    fn record(&mut self, before: Table) {
        self.undo.push(before);
        if self.undo.len() > 50 {
            self.undo.remove(0);
        }
        self.redo.clear();
        self.dirty = true;
        self.project_dirty = true;
        self.numeric = self.table.numeric_columns();
        self.refresh_statistics();
        self.plot.reset = true;
    }

    fn apply_edit(&mut self) {
        let Some(cell) = self.selected else {
            return;
        };
        if self
            .table
            .calculated
            .iter()
            .any(|d| d.column == cell.column)
        {
            self.message = "Calculated columns are read-only; edit their input cells.".into();
            return;
        }
        if self.edit.contains(['\t', '\n', '\r']) {
            self.message = "Use paste to insert a rectangular range.".into();
            return;
        }
        let row = self.view[cell.row];
        if self.table.rows[row][cell.column] != self.edit {
            let before = self.table.clone();
            self.table.rows[row][cell.column] = self.edit.clone();
            self.table.recalculate();
            self.record(before);
            // Editing a sort key or filter must update the displayed view too.
            self.rebuild();
            self.message = "Cell updated. Statistics and graphs recalculated.".into();
        }
    }

    fn undo(&mut self, redo: bool) {
        let previous = if redo {
            self.redo.pop()
        } else {
            self.undo.pop()
        };
        if let Some(previous) = previous {
            let current = std::mem::replace(&mut self.table, previous);
            if redo {
                self.undo.push(current);
            } else {
                self.redo.push(current);
            }
            self.dirty = true;
            self.project_dirty = true;
            self.rebuild();
        }
    }

    fn save_table(&mut self, view_only: bool) -> bool {
        if self.table.columns.is_empty() {
            return false;
        }
        let stem = self
            .path
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "frinZ".into());
        let mut dialog = rfd::FileDialog::new()
            .add_filter("TSV", &["tsv"])
            .set_file_name(format!(
                "{stem}_{}.tsv",
                if view_only { "view" } else { "table" }
            ));
        if let Some(parent) = self.path.as_ref().and_then(|p| p.parent()) {
            dialog = dialog.set_directory(parent);
        }
        let Some(path) = dialog.save_file() else {
            return false;
        };
        let rows = if view_only {
            self.view.clone()
        } else {
            (0..self.table.rows.len()).collect()
        };
        match self.table.tsv(&rows, true).and_then(|text| {
            fs::write(&path, text)?;
            Ok(())
        }) {
            Ok(()) => {
                self.message = format!("Saved {} rows to {}", rows.len(), path.display());
                if !view_only {
                    self.path = Some(path);
                    self.dirty = false;
                    self.project_dirty = true;
                }
                true
            }
            Err(err) => {
                self.message = format!("Save failed: {err}");
                false
            }
        }
    }

    fn export_statistics(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("TSV", &["tsv"])
            .set_file_name("frinZ_statistics.tsv")
            .save_file()
        else {
            return;
        };
        let result: Result<()> = (|| {
            let mut writer = csv::WriterBuilder::new().delimiter(b'\t').from_path(path)?;
            writer.write_record([
                "Scope",
                "Source",
                "Column",
                "Unit",
                "N",
                "Missing",
                "Sum",
                "Mean",
                "Median",
                "SampleStddev",
                "SEM",
                "Min",
                "Max",
                "RMS",
            ])?;
            let scope = if self.statistics_selected {
                "selected rows"
            } else {
                "visible rows"
            };
            let write = |writer: &mut csv::Writer<std::fs::File>,
                         source: &str,
                         summary: &ColumnSummary|
             -> Result<()> {
                if let Some(s) = &summary.statistics {
                    writer.write_record(vec![
                        scope.into(),
                        source.into(),
                        summary.column.clone(),
                        summary.unit.clone(),
                        s.count.to_string(),
                        s.missing.to_string(),
                        s.sum.to_string(),
                        s.mean.to_string(),
                        s.median.to_string(),
                        s.sample_stddev.map(|v| v.to_string()).unwrap_or_default(),
                        s.standard_error.map(|v| v.to_string()).unwrap_or_default(),
                        s.min.to_string(),
                        s.max.to_string(),
                        s.rms.to_string(),
                    ])?;
                }
                Ok(())
            };
            for summary in &self.summaries {
                write(&mut writer, "All", summary)?;
            }
            let rows = if self.statistics_selected {
                self.selected_rows()
            } else {
                self.view.clone()
            };
            for c in &self.numeric {
                for group in self.table.source_summaries(&rows, *c) {
                    write(
                        &mut writer,
                        &group.source,
                        &ColumnSummary {
                            column: self.table.columns[*c].name.clone(),
                            unit: self.table.columns[*c].unit.clone(),
                            statistics: group.statistics,
                        },
                    )?;
                }
            }
            writer.flush()?;
            Ok(())
        })();
        self.message = match result {
            Ok(()) => "Statistics exported.".into(),
            Err(err) => format!("Export failed: {err}"),
        };
    }

    fn request_open(&mut self, path: PathBuf) {
        if self.dirty || self.project_dirty {
            self.pending = Some(PendingAction::Open(path));
        } else {
            self.open(path);
        }
    }
    fn open(&mut self, path: PathBuf) {
        if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("frztable"))
        {
            match Project::load(&path) {
                Ok(project) => self.restore_project(project, path),
                Err(error) => self.message = format!("Open project failed: {error}"),
            }
            return;
        }
        match Table::load(&path) {
            Ok(table) => {
                self.table = table;
                self.path = Some(path);
                self.filter = Filter::default();
                self.sort = None;
                self.min_text.clear();
                self.max_text.clear();
                self.undo.clear();
                self.redo.clear();
                self.dirty = false;
                self.project_dirty = false;
                self.project_path = None;
                self.plot.y = self.table.column("Amp").unwrap_or(0);
                self.plot.x = None;
                self.plot.y_source = None;
                self.plot.y2 = None;
                self.plot.y2_source = None;
                self.set_default_mjd_reference();
                self.rebuild();
                self.message = format!(
                    "Loaded {} rows, {} columns.",
                    self.table.rows.len(),
                    self.table.columns.len()
                );
            }
            Err(err) => self.message = format!("Open failed: {err}"),
        }
    }

    fn copy_selection(&self, ctx: &egui::Context) {
        if let (Some(a), Some(b)) = (self.anchor, self.selected) {
            let mut writer = csv::WriterBuilder::new()
                .delimiter(b'\t')
                .from_writer(Vec::new());
            for r in a.row.min(b.row)..=a.row.max(b.row) {
                let row = &self.table.rows[self.view[r]];
                if writer
                    .write_record(&row[a.column.min(b.column)..=a.column.max(b.column)])
                    .is_err()
                {
                    return;
                }
            }
            if let Ok(bytes) = writer.into_inner() {
                if let Ok(text) = String::from_utf8(bytes) {
                    ctx.copy_text(text);
                }
            }
        }
    }

    fn keyboard(&mut self, ctx: &egui::Context) {
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, Key::F1)) {
            self.help.open = true;
        }
        let editing = ctx.wants_keyboard_input();
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, Key::S)) {
            self.save_project(false);
        }
        if !editing {
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, Key::Z)) {
                self.undo(false);
            }
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, Key::Y)) {
                self.undo(true);
            }
            let events = ctx.input(|i| i.events.clone());
            for event in events {
                match event {
                    egui::Event::Copy => self.copy_selection(ctx),
                    egui::Event::Paste(text) => {
                        if let Some(cell) = self.selected {
                            let before = self.table.clone();
                            match self.table.paste(&self.view, cell.row, cell.column, &text) {
                                Ok(n) => {
                                    self.record(before);
                                    self.rebuild();
                                    self.message = format!("Pasted {n} cells.");
                                }
                                Err(err) => self.message = err.to_string(),
                            }
                        }
                    }
                    egui::Event::Key {
                        key,
                        pressed: true,
                        modifiers,
                        ..
                    } => {
                        if let Some(cell) = self.selected {
                            let mut next = cell;
                            match key {
                                Key::F2 => {
                                    self.edit_focus = true;
                                    continue;
                                }
                                Key::ArrowDown | Key::Enter => {
                                    next.row = (cell.row + 1).min(self.view.len().saturating_sub(1))
                                }
                                Key::ArrowUp => next.row = cell.row.saturating_sub(1),
                                Key::ArrowLeft => next.column = cell.column.saturating_sub(1),
                                Key::ArrowRight => {
                                    next.column = (cell.column + 1)
                                        .min(self.table.columns.len().saturating_sub(1))
                                }
                                Key::Tab => {
                                    next.column = if modifiers.shift {
                                        cell.column.saturating_sub(1)
                                    } else {
                                        (cell.column + 1)
                                            .min(self.table.columns.len().saturating_sub(1))
                                    }
                                }
                                _ => continue,
                            }
                            self.select(next, modifiers.shift && key != Key::Tab);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn toolbar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading(RichText::new("frinZtable").color(Color32::WHITE));
                ui.separator();
                if ui.button("Open TSV…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("frinZ tables", &["tsv", "csv", "txt"])
                        .pick_file()
                    {
                        self.request_open(path);
                    }
                }
                if ui.button("Open project…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("frinZtable project", &["frztable"])
                        .pick_file()
                    {
                        self.request_open(path);
                    }
                }
                if ui
                    .add_enabled(
                        !self.table.columns.is_empty(),
                        egui::Button::new("Save project"),
                    )
                    .clicked()
                {
                    self.save_project(false);
                }
                if ui
                    .add_enabled(
                        !self.table.columns.is_empty(),
                        egui::Button::new("Save project as…"),
                    )
                    .clicked()
                {
                    self.save_project(true);
                }
                if self.view_mode != ViewMode::Plot {
                    if ui
                        .add_enabled(
                            !self.table.columns.is_empty(),
                            egui::Button::new("Save table as…"),
                        )
                        .clicked()
                    {
                        self.save_table(false);
                    }
                    if ui
                        .add_enabled(
                            !self.table.columns.is_empty(),
                            egui::Button::new("Export view…"),
                        )
                        .clicked()
                    {
                        self.save_table(true);
                    }
                    if ui.button("Export statistics…").clicked() {
                        self.export_statistics();
                    }
                    ui.separator();
                    if ui
                        .add_enabled(!self.undo.is_empty(), egui::Button::new("Undo"))
                        .clicked()
                    {
                        self.undo(false);
                    }
                    if ui
                        .add_enabled(!self.redo.is_empty(), egui::Button::new("Redo"))
                        .clicked()
                    {
                        self.undo(true);
                    }
                    if ui
                        .add_enabled(
                            !self.table.columns.is_empty(),
                            egui::Button::new("+ Calculated column"),
                        )
                        .clicked()
                    {
                        self.formula_window = true;
                    }
                }
                if ui.button("Help / ヘルプ (F1)").clicked() {
                    self.help.open = true;
                }
            });
            ui.horizontal(|ui| {
                let prior_view = self.view_mode;
                ui.strong("View");
                ui.selectable_value(
                    &mut self.view_mode,
                    ViewMode::Spreadsheet,
                    "Spreadsheet only",
                );
                ui.selectable_value(&mut self.view_mode, ViewMode::Plot, "Plot only");
                ui.selectable_value(
                    &mut self.view_mode,
                    ViewMode::Combined,
                    "Spreadsheet + plot",
                );
                if self.view_mode != prior_view {
                    self.project_dirty = true;
                }
            });
            ui.separator();
            if let Some(path) = &self.path {
                ui.label(
                    RichText::new(format!(
                        "{}{}",
                        path.display(),
                        match (self.dirty, self.project_dirty) {
                            (true, true) => "  • unsaved edits / project settings",
                            (true, false) => "  • unsaved edits",
                            (false, true) => "  • unsaved project settings",
                            (false, false) => "",
                        }
                    ))
                    .small(),
                );
            }
            let mut changed = false;
            ui.horizontal(|ui| {
                ui.label("Source");
                egui::ComboBox::from_id_salt("source_filter")
                    .selected_text(match &self.filter.sources {
                        None => "All sources".into(),
                        Some(sources) if sources.is_empty() => "No sources".into(),
                        Some(sources) if sources.len() <= 2 => {
                            sources.iter().cloned().collect::<Vec<_>>().join(", ")
                        }
                        Some(sources) => format!("{} sources", sources.len()),
                    })
                    .show_ui(ui, |ui| {
                        ui.horizontal(|ui| {
                            if ui.small_button("All").clicked() {
                                self.filter.sources = None;
                                changed = true;
                            }
                            if ui.small_button("None").clicked() {
                                self.filter.sources = Some(Default::default());
                                changed = true;
                            }
                        });
                        ui.label("Choose any combination of sources");
                        let sources = self.table.sources();
                        for source in &sources {
                            let mut checked = self
                                .filter
                                .sources
                                .as_ref()
                                .is_none_or(|set| set.contains(source));
                            if ui.checkbox(&mut checked, source).changed() {
                                let set = self
                                    .filter
                                    .sources
                                    .get_or_insert_with(|| sources.iter().cloned().collect());
                                if checked {
                                    set.insert(source.clone());
                                } else {
                                    set.remove(source);
                                }
                                changed = true;
                            }
                        }
                    });
                changed |= ui
                    .add(
                        egui::TextEdit::singleline(&mut self.filter.text)
                            .hint_text("Search all cells")
                            .desired_width(180.0),
                    )
                    .changed();
                if ui.button("Reset filters / sort").clicked() {
                    self.filter = Filter::default();
                    self.sort = None;
                    self.min_text.clear();
                    self.max_text.clear();
                    changed = true;
                }
                ui.label(format!(
                    "{} / {} rows",
                    self.view.len(),
                    self.table.rows.len()
                ));
            });
            ui.horizontal(|ui| {
                ui.label("Numeric filter");
                egui::ComboBox::from_id_salt("numeric_filter")
                    .selected_text(
                        self.filter
                            .numeric_column
                            .map(|c| self.table.columns[c].name.as_str())
                            .unwrap_or("None"),
                    )
                    .show_ui(ui, |ui| {
                        changed |= ui
                            .selectable_value(&mut self.filter.numeric_column, None, "None")
                            .changed();
                        for &c in &self.numeric {
                            changed |= ui
                                .selectable_value(
                                    &mut self.filter.numeric_column,
                                    Some(c),
                                    &self.table.columns[c].name,
                                )
                                .changed();
                        }
                    });
                ui.label("Min");
                changed |= ui
                    .add(egui::TextEdit::singleline(&mut self.min_text).desired_width(95.0))
                    .changed();
                ui.label("Max");
                changed |= ui
                    .add(egui::TextEdit::singleline(&mut self.max_text).desired_width(95.0))
                    .changed();
                if (!self.min_text.is_empty() && number(&self.min_text).is_none())
                    || (!self.max_text.is_empty() && number(&self.max_text).is_none())
                {
                    ui.colored_label(Color32::WHITE, "Enter finite numeric bounds");
                } else if number(&self.min_text)
                    .zip(number(&self.max_text))
                    .is_some_and(|(a, b)| a > b)
                {
                    ui.colored_label(Color32::WHITE, "Min is greater than Max");
                }
                ui.label(RichText::new("Column headers sort · Row menu sorts by object").small());
            });
            if changed {
                self.filter.min = number(&self.min_text);
                self.filter.max = number(&self.max_text);
                self.rebuild();
                self.project_dirty = true;
            }
            if self.view_mode != ViewMode::Plot {
                ui.horizontal(|ui| {
                    let label = self
                        .selected
                        .map(|cell| format!("{}{}", column_letter(cell.column), cell.row + 1))
                        .unwrap_or_else(|| "Cell".into());
                    ui.label(RichText::new(label).monospace());
                    let writable = self.selected.is_some_and(|c| {
                        !self.table.calculated.iter().any(|d| d.column == c.column)
                    });
                    let response = ui.add_enabled(
                        writable,
                        egui::TextEdit::singleline(&mut self.edit)
                            .desired_width((ui.available_width() - 170.0).max(80.0)),
                    );
                    if self.edit_focus {
                        response.request_focus();
                        self.edit_focus = false;
                    }
                    if ui
                        .add_enabled(writable, egui::Button::new("Apply"))
                        .clicked()
                        || (response.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)))
                    {
                        self.apply_edit();
                    }
                    if ui.button("Copy range").clicked() {
                        self.copy_selection(ctx);
                    }
                });
            }
        });
    }

    fn sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("statistics").resizable(true).default_width(345.0).min_width(260.0).show(ctx, |ui| {
            ui.heading("Column statistics");
            let mut changed = false;
            ui.horizontal(|ui| {
                changed |= ui.selectable_value(&mut self.statistics_selected, false, "Visible rows").changed();
                changed |= ui.selectable_value(&mut self.statistics_selected, true, "Selected rows").changed();
            });
            if ui.checkbox(&mut self.all_statistics, "All numeric columns").changed() {
                self.project_dirty = true;
            }
            if changed { self.refresh_statistics(); }
            let rows = if self.statistics_selected { self.selected_rows().len() } else { self.view.len() };
            ui.label(format!("Scope: {rows} rows · arithmetic statistics"));
            ui.label(RichText::new("Stddev: sample (n−1). SEM: stddev / √n.").small());
            let mut filter_source = None;
            ScrollArea::vertical().id_salt("statistics_scroll").show(ui, |ui| {
                let columns = if self.all_statistics { self.numeric.clone() } else { vec![self.plot.y] };
                for c in columns {
                    let Some(summary) = self.summaries.get(c) else { continue; };
                    ui.separator();
                    ui.strong(format!("{} {}", summary.column, summary.unit));
                    if let Some(s) = &summary.statistics { statistics_grid(ui, c, s); } else { ui.label("No finite numeric cells in scope."); }
                    if summary.column == "Phase" { ui.label(RichText::new("Phase uses arithmetic statistics; ±180° wrapping is not corrected.").small().color(Color32::WHITE)); }
                }
                ui.separator(); ui.heading("By source");
                ui.label(RichText::new("Click a source to filter; Ctrl-click adds/removes it.").small());
                ScrollArea::horizontal().id_salt("source_scroll").show(ui, |ui| {
                    egui::Grid::new("source_summary").striped(true).show(ui, |ui| {
                        for (i, label) in ["Source", "N", "Mean", "Median", "Stddev", "Min", "Max"].iter().enumerate() {
                            if ui.small_button(*label).clicked() {
                                self.group_descending = if self.group_sort == i { !self.group_descending } else { false };
                                self.group_sort = i; self.sort_groups(); self.project_dirty = true;
                            }
                        }
                        ui.end_row();
                        for (i, group) in self.groups.iter().enumerate() {
                            if ui.selectable_label(false, RichText::new(&group.source).color(plots::color(i))).clicked() { filter_source = Some(group.source.clone()); }
                            ui.label(group.rows.to_string());
                            for value in [group.statistics.as_ref().map(|s| s.mean), group.statistics.as_ref().map(|s| s.median), group.statistics.as_ref().and_then(|s| s.sample_stddev), group.statistics.as_ref().map(|s| s.min), group.statistics.as_ref().map(|s| s.max)] {
                                ui.label(value.map(display).unwrap_or_else(|| "—".into()));
                            }
                            ui.end_row();
                        }
                    });
                });
                ui.separator();
                for d in &self.table.calculated { ui.label(format!("{} = {}", self.table.columns[d.column].name, d.formula)); }
                if !self.table.comments.is_empty() {
                    ui.collapsing("File comments (#)", |ui| { for comment in &self.table.comments { ui.label(comment); } });
                }
            });
            if let Some(source) = filter_source {
                if ui.input(|i| i.modifiers.command) {
                    let all_sources = self.table.sources().into_iter().collect();
                    let set = self.filter.sources.get_or_insert(all_sources);
                    if !set.remove(&source) { set.insert(source); }
                } else { self.filter.sources = Some(std::iter::once(source).collect()); }
                self.filter.source.clear(); self.rebuild();
                self.project_dirty = true;
            }
        });
    }

    fn graphs(&mut self, ctx: &egui::Context) -> egui::Rect {
        if self.view_mode == ViewMode::Plot {
            egui::CentralPanel::default()
                .show(ctx, |ui| self.graph_contents(ui))
                .inner
        } else {
            egui::TopBottomPanel::bottom("graphs_wide")
                .resizable(true)
                .default_height(ctx.available_rect().height() * 0.55)
                .min_height(300.0)
                .show(ctx, |ui| self.graph_contents(ui))
                .inner
        }
    }

    fn graph_contents(&mut self, ui: &mut egui::Ui) -> egui::Rect {
        let mut changed = false;
        let plot_sources = self.table.sources();
        ui.horizontal_wrapped(|ui| {
            changed |= ui
                .selectable_value(&mut self.plot.mode, PlotMode::Series, "Time / scatter")
                .changed();
            changed |= ui
                .selectable_value(&mut self.plot.mode, PlotMode::Histogram, "Histogram")
                .changed();
            changed |= ui
                .selectable_value(&mut self.plot.mode, PlotMode::Sources, "Source means")
                .changed();
            ui.separator();
            ui.label("Y / parameter");
            egui::ComboBox::from_id_salt("plot_y")
                .selected_text(
                    self.table
                        .columns
                        .get(self.plot.y)
                        .map(|c| c.name.as_str())
                        .unwrap_or("—"),
                )
                .show_ui(ui, |ui| {
                    for &c in &self.numeric {
                        changed |= ui
                            .selectable_value(&mut self.plot.y, c, &self.table.columns[c].name)
                            .changed();
                    }
                });
            if self.plot.mode == PlotMode::Series {
                ui.label("Y₁ source");
                egui::ComboBox::from_id_salt("plot_y_source")
                    .selected_text(
                        self.plot
                            .y_source
                            .as_deref()
                            .unwrap_or("All visible sources"),
                    )
                    .show_ui(ui, |ui| {
                        changed |= ui
                            .selectable_value(&mut self.plot.y_source, None, "All visible sources")
                            .changed();
                        for source in &plot_sources {
                            changed |= ui
                                .selectable_value(
                                    &mut self.plot.y_source,
                                    Some(source.clone()),
                                    source,
                                )
                                .changed();
                        }
                    });
                ui.label("Y₂").on_hover_text(
                    "Compare another parameter for the Y₁ source, or select the same parameter and a different source.",
                );
                let y2_name = self
                    .plot
                    .y2
                    .and_then(|column| self.table.columns.get(column))
                    .map(|column| column.name.as_str())
                    .unwrap_or("None");
                egui::ComboBox::from_id_salt("plot_y2")
                    .selected_text(y2_name)
                    .show_ui(ui, |ui| {
                        changed |= ui
                            .selectable_value(&mut self.plot.y2, None, "None")
                            .changed();
                        for &c in &self.numeric {
                            changed |= ui
                                .selectable_value(
                                    &mut self.plot.y2,
                                    Some(c),
                                    &self.table.columns[c].name,
                                )
                                .changed();
                        }
                    });
                if self.plot.y2.is_some() {
                    ui.label("Y₂ source");
                    egui::ComboBox::from_id_salt("plot_y2_source")
                        .selected_text(
                            self.plot
                                .y2_source
                                .as_deref()
                                .unwrap_or("All visible sources"),
                        )
                        .show_ui(ui, |ui| {
                            changed |= ui
                                .selectable_value(
                                    &mut self.plot.y2_source,
                                    None,
                                    "All visible sources",
                                )
                                .changed();
                            for source in &plot_sources {
                                changed |= ui
                                    .selectable_value(
                                        &mut self.plot.y2_source,
                                        Some(source.clone()),
                                        source,
                                    )
                                    .changed();
                            }
                        });
                }
            }
            changed |= ui.checkbox(&mut self.plot.log_y, "log10").changed();
            if ui.button("Fit plot").clicked() {
                self.plot.reset = true;
            }
            ui.separator();
            ui.label("Export format");
            for format in [ExportFormat::Png, ExportFormat::Pdf, ExportFormat::Eps] {
                changed |= ui.selectable_value(
                    &mut self.export_format,
                    format,
                    format.label(),
                ).changed();
            }
            if ui
                .add_enabled(
                    !self.table.columns.is_empty(),
                    egui::Button::new(format!("Export {}…", self.export_format.label())),
                )
                .clicked()
            {
                self.export_graph();
            }
        });
        ui.horizontal_wrapped(|ui| {
            if self.plot.mode == PlotMode::Series {
                ui.label("X");
                egui::ComboBox::from_id_salt("plot_x")
                    .selected_text(
                        self.plot
                            .x
                            .map(|c| self.table.columns[c].name.as_str())
                            .unwrap_or_else(|| plots::automatic_x_name(&self.table)),
                    )
                    .show_ui(ui, |ui| {
                        changed |= ui
                            .selectable_value(
                                &mut self.plot.x,
                                None,
                                plots::automatic_x_name(&self.table),
                            )
                            .changed();
                        for &c in &self.numeric {
                            changed |= ui
                                .selectable_value(
                                    &mut self.plot.x,
                                    Some(c),
                                    &self.table.columns[c].name,
                                )
                                .changed();
                        }
                    });
                ui.label("X scale");
                egui::ComboBox::from_id_salt("plot_x_scale")
                    .selected_text(match self.plot.x_axis_scale {
                        plots::XAxisScale::Automatic => "Automatic",
                        plots::XAxisScale::Hours => "Hours",
                        plots::XAxisScale::Date => "Date (UTC)",
                    })
                    .show_ui(ui, |ui| {
                        changed |= ui
                            .selectable_value(
                                &mut self.plot.x_axis_scale,
                                plots::XAxisScale::Automatic,
                                "Automatic",
                            )
                            .changed();
                        changed |= ui
                            .selectable_value(
                                &mut self.plot.x_axis_scale,
                                plots::XAxisScale::Hours,
                                "Hours",
                            )
                            .changed();
                        changed |= ui
                            .selectable_value(
                                &mut self.plot.x_axis_scale,
                                plots::XAxisScale::Date,
                                "Date (UTC)",
                            )
                            .changed();
                    });
                let uses_mjd_x = plots::uses_mjd_x(&self.table, self.plot.x);
                let time_is_seconds = plots::time_column_is_seconds(&self.table, self.plot.x);
                if self.plot.x_axis_scale == plots::XAxisScale::Date && time_is_seconds {
                    ui.label("Date origin");
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut self.plot.date_origin)
                            .desired_width(150.0)
                            .hint_text("YYYY-MM-DD or ISO time"),
                    );
                    changed |= response.changed();
                    if !self.plot.date_origin.trim().is_empty()
                        && plots::date_origin_mjd(&self.plot.date_origin).is_none()
                    {
                        ui.colored_label(Color32::LIGHT_RED, "Invalid date or MJD");
                    }
                }
                if uses_mjd_x && self.plot.x_axis_scale != plots::XAxisScale::Date {
                    ui.label("Reference MJD");
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut self.plot.reference_text)
                            .desired_width(96.0)
                            .hint_text("e.g. 61323"),
                    );
                    self.project_dirty |= response.changed();
                    if response.changed() {
                        let value = self.plot.reference_text.trim();
                        if value.is_empty() {
                            self.plot.reference_mjd = None;
                            changed = true;
                        } else if let Ok(mjd) = value.parse::<f64>() {
                            if mjd.is_finite() {
                                self.plot.reference_mjd = Some(mjd);
                                changed = true;
                            } else {
                                self.message = "Reference MJD must be a finite number.".into();
                            }
                        } else {
                            self.message = "Enter a numeric reference MJD, or leave it blank for absolute MJD.".into();
                        }
                    }
                    if ui.small_button("First day").clicked() {
                        self.set_default_mjd_reference();
                        changed = true;
                    }
                    if ui.small_button("Absolute MJD").clicked() {
                        self.plot.reference_mjd = None;
                        self.plot.reference_text.clear();
                        changed = true;
                    }
                }
                ui.label("Series style");
                changed |= ui
                    .selectable_value(
                        &mut self.plot.series_style,
                        SeriesStyle::Scatter,
                        "Scatter",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut self.plot.series_style,
                        SeriesStyle::Linear,
                        "Linear",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut self.plot.series_style,
                        SeriesStyle::ScatterDotted,
                        "Scatter + dotted line",
                    )
                    .changed();
                changed |= ui.checkbox(&mut self.plot.noise_bars, "Amp ± Noise-level").changed();
            } else if self.plot.mode == PlotMode::Histogram {
                ui.label("Bins");
                changed |= ui
                    .add(egui::Slider::new(&mut self.plot.bins, 1..=100))
                    .changed();
            }
            changed |= ui.checkbox(&mut self.plot.preview_9_6, "9:6 preview").changed();
            ui.label(RichText::new("Drag to pan · scroll to zoom · double-click to fit").small());
        });
        if changed {
            self.plot.reset = true;
            self.refresh_statistics();
            self.project_dirty = true;
        }
        let selected = self.selected_rows();
        ui.horizontal_wrapped(|ui| {
            let plotted = self.table.plotted_rows(&self.view).len();
            ui.label(format!(
                "Plot: {plotted} / {} visible rows · 9:6 output",
                self.view.len()
            ));
            if ui.button("Plot visible").clicked() {
                self.set_plot_rows(&self.view.clone(), true);
            }
            if ui.button("Exclude visible").clicked() {
                self.set_plot_rows(&self.view.clone(), false);
            }
            if ui
                .add_enabled(
                    !selected.is_empty(),
                    egui::Button::new("Plot selected rows"),
                )
                .clicked()
            {
                self.set_plot_rows(&selected, true);
            }
            if ui
                .add_enabled(
                    !selected.is_empty(),
                    egui::Button::new("Exclude selected rows"),
                )
                .clicked()
            {
                self.set_plot_rows(&selected, false);
            }
        });
        let rows = self.table.plotted_rows(&self.view);
        plots::show(ui, &self.table, &rows, &selected, &mut self.plot)
    }

    fn main_panels(&mut self, ctx: &egui::Context) -> egui::Rect {
        match self.view_mode {
            ViewMode::Spreadsheet => {
                self.sidebar(ctx);
                self.sheet(ctx);
                egui::Rect::NOTHING
            }
            ViewMode::Plot => self.graphs(ctx),
            ViewMode::Combined => {
                self.sidebar(ctx);
                let graph = self.graphs(ctx);
                self.sheet(ctx);
                graph
            }
        }
    }

    fn sheet(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.table.columns.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("Open or drop a frinZ TSV file to begin.");
                });
                return;
            }
            let mut sort_column: Option<(usize, bool)> = None;
            let mut row_sort_changed = false;
            let mut plot_column = None;
            let mut clicked = None;
            let mut plot_changes = Vec::new();
            ScrollArea::horizontal()
                .id_salt("sheet_horizontal")
                .show(ui, |ui| {
                    ui.set_min_width(108.0 + self.table.columns.len() as f32 * CELL_WIDTH);
                    ui.horizontal(|ui| {
                        let source_column = self.table.column("Source");
                        let row_order_label = match self.sort {
                            Some(sort) if sort.by_source => {
                                if sort.descending { "Object + value ↓" } else { "Object + value ↑" }
                            }
                            Some(sort) if Some(sort.column) == source_column => {
                                if sort.descending { "Object Z–A" } else { "Object A–Z" }
                            }
                            Some(_) => "Column sort",
                            None => "Row order",
                        };
                        egui::ComboBox::from_id_salt("row_sort")
                            .width(88.0)
                            .selected_text(row_order_label)
                            .show_ui(ui, |ui| {
                                if ui.selectable_value(&mut self.sort, None, "Original row order").changed() {
                                    row_sort_changed = true;
                                }
                                if let Some(source) = source_column {
                                    for (descending, label) in [
                                        (false, "Object name A–Z"),
                                        (true, "Object name Z–A"),
                                    ] {
                                        if ui.selectable_value(
                                            &mut self.sort,
                                            Some(Sort { column: source, descending, by_source: false }),
                                            label,
                                        ).changed() {
                                            row_sort_changed = true;
                                        }
                                    }
                                    ui.separator();
                                    for (descending, label) in [
                                        (false, "Object, then value ↑"),
                                        (true, "Object, then value ↓"),
                                    ] {
                                        if ui.selectable_value(
                                            &mut self.sort,
                                            Some(Sort { column: self.plot.y, descending, by_source: true }),
                                            label,
                                        ).changed() {
                                            row_sort_changed = true;
                                        }
                                    }
                                } else {
                                    ui.label("This table has no Source column.");
                                }
                            });
                        ui.add_sized([48.0, 44.0], egui::Label::new("Plot"));
                        for (c, column) in self.table.columns.iter().enumerate() {
                            let arrow = self
                                .sort
                                .filter(|s| s.column == c)
                                .map(|s| if s.descending { " ↓" } else { " ↑" })
                                .unwrap_or("");
                            let label = format!(
                                "{} · {}{}\n{}",
                                column_letter(c),
                                column.name,
                                arrow,
                                column.unit
                            );
                            let fill = if self.plot.y == c {
                                Color32::from_gray(50)
                            } else {
                                Color32::from_gray(22)
                            };
                            let header = ui
                                .add_sized(
                                    [CELL_WIDTH - ui.spacing().item_spacing.x, 44.0],
                                    egui::Button::new(RichText::new(label).size(11.0)).fill(fill),
                                );
                            if header.clicked() {
                                let descending = self.sort.is_some_and(|sort| {
                                    sort.column == c && !sort.descending
                                });
                                sort_column = Some((c, descending));
                            }
                            let is_numeric = self.numeric.contains(&c);
                            header.on_hover_text("Click to toggle ascending / descending sort. Right-click for explicit sort options.")
                                .context_menu(|ui| {
                                    if ui.button("Sort ascending ↑").clicked() {
                                        sort_column = Some((c, false));
                                        ui.close();
                                    }
                                    if ui.button("Sort descending ↓").clicked() {
                                        sort_column = Some((c, true));
                                        ui.close();
                                    }
                                    if is_numeric && ui.button("Plot this column").clicked() {
                                        plot_column = Some(c);
                                        ui.close();
                                    }
                                });
                        }
                    });
                    ui.separator();
                    ScrollArea::vertical()
                        .id_salt("sheet_vertical")
                        .auto_shrink([false, false])
                        .show_rows(ui, ROW_HEIGHT, self.view.len(), |ui, range| {
                            for r in range {
                                ui.horizontal(|ui| {
                                    let row_response = ui.add_sized(
                                        [42.0, ROW_HEIGHT],
                                        egui::Button::new((r + 1).to_string()).frame(false),
                                    );
                                    if row_response.clicked() {
                                        clicked = Some((
                                            Cell {
                                                row: r,
                                                column: self.plot.y,
                                            },
                                            false,
                                            false,
                                        ));
                                    }
                                    let row = self.view[r];
                                    let mut enabled = self.table.plot_enabled[row];
                                    if ui
                                        .add_sized(
                                            [48.0, ROW_HEIGHT],
                                            egui::Checkbox::without_text(&mut enabled),
                                        )
                                        .changed()
                                    {
                                        plot_changes.push((row, enabled));
                                    }
                                    for (c, text) in
                                        self.table.rows[self.view[r]].iter().enumerate()
                                    {
                                        let in_range =
                                            self.anchor.zip(self.selected).is_some_and(|(a, b)| {
                                                r >= a.row.min(b.row)
                                                    && r <= a.row.max(b.row)
                                                    && c >= a.column.min(b.column)
                                                    && c <= a.column.max(b.column)
                                            });
                                        let fill = if in_range {
                                            Color32::from_gray(55)
                                        } else if r % 2 == 0 {
                                            Color32::from_gray(10)
                                        } else {
                                            Color32::BLACK
                                        };
                                        let cell = egui::Button::new(
                                            RichText::new(text).monospace().size(11.0),
                                        )
                                        .fill(fill)
                                        .stroke(egui::Stroke::new(0.3, Color32::from_gray(55)))
                                        .sense(Sense::click());
                                        let response = ui
                                            .add_sized(
                                                [
                                                    CELL_WIDTH - ui.spacing().item_spacing.x,
                                                    ROW_HEIGHT,
                                                ],
                                                cell,
                                            )
                                            .on_hover_text(format!(
                                                "{}\n{} {}",
                                                text,
                                                self.table.columns[c].name,
                                                self.table.columns[c].unit
                                            ));
                                        if response.clicked() || response.double_clicked() {
                                            clicked = Some((
                                                Cell { row: r, column: c },
                                                ui.input(|i| i.modifiers.shift),
                                                response.double_clicked(),
                                            ));
                                        }
                                    }
                                });
                            }
                        });
                });
            if let Some((c, descending)) = sort_column {
                self.sort = Some(Sort {
                    column: c,
                    descending,
                    by_source: self.sort.is_some_and(|sort| sort.by_source),
                });
                if self.numeric.contains(&c) {
                    self.plot.y = c;
                }
                self.rebuild();
                self.project_dirty = true;
            }
            if row_sort_changed {
                self.rebuild();
                self.project_dirty = true;
            }
            if let Some(column) = plot_column {
                self.plot.y = column;
                self.plot.reset = true;
                self.refresh_statistics();
                self.project_dirty = true;
            }
            if let Some((cell, extend, edit)) = clicked {
                self.select(cell, extend);
                self.edit_focus = edit;
                ctx.request_repaint();
            }
            if !plot_changes.is_empty() {
                let before = self.table.clone();
                for (row, enabled) in plot_changes {
                    self.table.plot_enabled[row] = enabled;
                }
                self.record(before);
                ctx.request_repaint();
            }
        });
    }

    fn dialogs(&mut self, ctx: &egui::Context) {
        self.help.show(ctx);
        if self.formula_window {
            let mut open = true;
            egui::Window::new("Calculated column")
                .open(&mut open)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(
                        "Recalculated whenever input cells change. Saved TSV contains values.",
                    );
                    ui.horizontal(|ui| {
                        ui.label("Name");
                        ui.text_edit_singleline(&mut self.formula_name);
                        ui.label("Unit");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.formula_unit).desired_width(65.0),
                        );
                    });
                    ui.add(egui::TextEdit::singleline(&mut self.formula).desired_width(440.0));
                    ui.label("References: Amp, E, [Noise-level]. Operators: + − * / ^ ( ).");
                    ui.label("Functions: abs, sqrt, ln, log10, exp, sin, cos (angles in radians).");
                    ui.label("Missing/nonfinite inputs and invalid results produce empty cells.");
                    if ui.button("Add column").clicked() {
                        let before = self.table.clone();
                        match self.table.add_calculated(
                            &self.formula_name,
                            &self.formula_unit,
                            &self.formula,
                        ) {
                            Ok(()) => {
                                self.plot.y = self.table.columns.len() - 1;
                                self.record(before);
                                self.rebuild();
                                self.formula_window = false;
                                self.message = "Calculated column added.".into();
                            }
                            Err(err) => self.message = err.to_string(),
                        }
                    }
                });
            if !open {
                self.formula_window = false;
            }
        }
        if self.pending.is_some() {
            let mut action = 0;
            egui::Modal::new(egui::Id::new("unsaved_edits")).show(ctx, |ui| {
                ui.heading("Unsaved project changes");
                ui.label("Save the table and analysis settings before continuing?");
                ui.horizontal(|ui| {
                    if ui.button("Save project…").clicked() {
                        action = 1;
                    }
                    if ui.button("Discard edits").clicked() {
                        action = 2;
                    }
                    if ui.button("Cancel").clicked() {
                        action = 3;
                    }
                });
            });
            if action == 3 {
                self.pending = None;
            }
            if action == 2 || (action == 1 && self.save_project(false)) {
                match self.pending.take() {
                    Some(PendingAction::Open(path)) => self.open(path),
                    Some(PendingAction::Close) => {
                        self.allow_close = true;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    None => {}
                }
            }
        }
    }
}

impl eframe::App for TableApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.screenshot.is_some() {
            self.frames += 1;
            if self.frames == 4 {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            }
            ctx.request_repaint();
            if let Some(image) = ctx.input(|i| {
                i.events.iter().find_map(|event| {
                    if let egui::Event::Screenshot { image, .. } = event {
                        Some(image.clone())
                    } else {
                        None
                    }
                })
            }) {
                let path = self.screenshot.take().expect("screenshot path exists");
                let bytes: Vec<u8> = image
                    .pixels
                    .iter()
                    .flat_map(|p| p.to_srgba_unmultiplied())
                    .collect();
                match image::save_buffer(
                    &path,
                    &bytes,
                    image.width() as u32,
                    image.height() as u32,
                    image::ColorType::Rgba8,
                ) {
                    Ok(()) => {
                        println!("Screenshot saved: {}", path.display());
                        self.allow_close = true;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    Err(err) => {
                        self.message = format!("Screenshot failed: {err}");
                        eprintln!("{}", self.message);
                    }
                }
            }
        }
        if ctx.input(|i| i.viewport().close_requested())
            && (self.dirty || self.project_dirty)
            && !self.allow_close
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.pending = Some(PendingAction::Close);
        }
        let dropped = ctx.input(|i| i.raw.dropped_files.first().and_then(|f| f.path.clone()));
        if let Some(path) = dropped {
            self.request_open(path);
        }
        if self.pending.is_none() {
            self.keyboard(ctx);
        }
        self.toolbar(ctx);
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(&self.message).small());
            });
        });
        self.main_panels(ctx);
        self.dialogs(ctx);
    }
}

fn dark_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.override_text_color = Some(Color32::WHITE);
    visuals.panel_fill = Color32::BLACK;
    visuals.window_fill = Color32::BLACK;
    visuals.extreme_bg_color = Color32::BLACK;
    visuals.faint_bg_color = Color32::from_gray(12);
    visuals.selection.bg_fill = Color32::from_gray(60);
    visuals.selection.stroke.color = Color32::WHITE;
    visuals.widgets.noninteractive.fg_stroke.color = Color32::WHITE;
    visuals.widgets.noninteractive.bg_fill = Color32::from_gray(22);
    visuals.widgets.noninteractive.weak_bg_fill = Color32::from_gray(15);
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(0.9, Color32::from_gray(65));

    visuals.widgets.inactive.fg_stroke.color = Color32::WHITE;
    visuals.widgets.inactive.bg_fill = Color32::from_gray(22);
    visuals.widgets.inactive.weak_bg_fill = Color32::from_gray(15);
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.2, Color32::from_gray(125));

    visuals.widgets.hovered.fg_stroke.color = Color32::WHITE;
    visuals.widgets.hovered.bg_fill = Color32::from_gray(38);
    visuals.widgets.hovered.weak_bg_fill = Color32::from_gray(30);
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.5, Color32::from_gray(205));

    visuals.widgets.active.fg_stroke.color = Color32::WHITE;
    visuals.widgets.active.bg_fill = Color32::from_gray(52);
    visuals.widgets.active.weak_bg_fill = Color32::from_gray(42);
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.7, Color32::WHITE);

    visuals.widgets.open.fg_stroke.color = Color32::WHITE;
    visuals.widgets.open.bg_fill = Color32::from_gray(38);
    visuals.widgets.open.weak_bg_fill = Color32::from_gray(30);
    visuals.widgets.open.bg_stroke = egui::Stroke::new(1.5, Color32::from_gray(190));
    ctx.set_visuals(visuals);
}

fn statistics_grid(ui: &mut egui::Ui, id: usize, s: &Statistics) {
    egui::Grid::new(("column_statistics", id))
        .num_columns(2)
        .striped(true)
        .min_col_width(110.0)
        .show(ui, |ui| {
            for (name, value) in [
                ("Numeric N", s.count.to_string()),
                ("Missing / nonnumeric", s.missing.to_string()),
                ("Sum", display(s.sum)),
                ("Mean", display(s.mean)),
                ("Median", display(s.median)),
                (
                    "Sample stddev",
                    s.sample_stddev.map(display).unwrap_or_else(|| "—".into()),
                ),
                (
                    "Standard error",
                    s.standard_error.map(display).unwrap_or_else(|| "—".into()),
                ),
                ("Minimum", display(s.min)),
                ("Maximum", display(s.max)),
                ("RMS", display(s.rms)),
            ] {
                ui.label(name);
                ui.label(RichText::new(value).monospace());
                ui.end_row();
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    fn app() -> TableApp {
        TableApp::from_table(
            Table::parse("Source\tAmp\tNoise-level\nA\t4\t2\nB\t6\t3", false).unwrap(),
            None,
            Vec::new(),
            None,
        )
    }

    #[test]
    fn edits_update_filtered_views_and_undo_removes_derived_references() {
        let mut app = app();
        app.select(Cell { row: 0, column: 1 }, false);
        app.edit = "8".into();
        app.apply_edit();
        assert_eq!(app.summaries[1].statistics.as_ref().unwrap().mean, 7.0);
        app.undo(false);
        assert_eq!(app.table.rows[0][1], "4");
        app.undo(true);
        assert_eq!(app.table.rows[0][1], "8");
        let before = app.table.clone();
        app.table
            .add_calculated("ratio", "-", "Amp / [Noise-level]")
            .unwrap();
        app.record(before);
        app.plot.y = 3;
        app.plot.x = Some(3);
        app.filter.numeric_column = Some(3);
        app.sort = Some(Sort {
            column: 3,
            descending: true,
            by_source: false,
        });
        app.undo(false);
        assert_eq!(app.table.columns.len(), 3);
        assert!(app.plot.x.is_none());
        assert!(app.filter.numeric_column.is_none());
        assert!(app.sort.is_none());
        app.filter = Filter {
            numeric_column: Some(1),
            min: Some(7.0),
            ..Default::default()
        };
        app.rebuild();
        app.select(Cell { row: 0, column: 1 }, false);
        app.edit = "1".into();
        app.apply_edit();
        assert!(app.view.is_empty());
    }

    #[test]
    fn selected_statistics_and_all_plot_modes_render_without_a_display() {
        let mut app = app();
        app.select(Cell { row: 0, column: 1 }, false);
        app.statistics_selected = true;
        app.refresh_statistics();
        assert_eq!(app.summaries[1].statistics.as_ref().unwrap().count, 1);
        app.select(Cell { row: 1, column: 1 }, true);
        assert_eq!(app.summaries[1].statistics.as_ref().unwrap().count, 2);
        let ctx = egui::Context::default();
        dark_theme(&ctx);
        assert_eq!(ctx.style().visuals.panel_fill, Color32::BLACK);
        assert_eq!(
            ctx.style().visuals.override_text_color,
            Some(Color32::WHITE)
        );
        for mode in [PlotMode::Series, PlotMode::Histogram, PlotMode::Sources] {
            app.plot.mode = mode;
            let output = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1480.0, 940.0),
                    )),
                    ..Default::default()
                },
                |ctx| {
                    app.toolbar(ctx);
                    let graph = app.main_panels(ctx);
                    assert!(
                        graph.width() >= 1000.0 && graph.height() >= 250.0,
                        "graph collapsed: {graph:?}"
                    );
                },
            );
            assert!(!output.shapes.is_empty());
        }
        app.set_view_mode(ViewMode::Plot);
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1480.0, 940.0),
                )),
                ..Default::default()
            },
            |ctx| {
                app.toolbar(ctx);
                let graph = app.main_panels(ctx);
                assert!(
                    graph.width() >= 1400.0 && graph.height() >= 700.0,
                    "expanded graph too small: {graph:?}"
                );
            },
        );
        app.set_view_mode(ViewMode::Spreadsheet);
        let output = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1480.0, 940.0),
                )),
                ..Default::default()
            },
            |ctx| {
                app.toolbar(ctx);
                app.main_panels(ctx);
            },
        );
        assert!(!output.shapes.is_empty());
    }

    #[test]
    fn excluded_rows_are_undoable_and_do_not_change_scientific_cells() {
        let mut app = app();
        app.set_plot_rows(&[1], false);
        assert_eq!(app.table.plotted_rows(&app.view), vec![0]);
        assert_eq!(app.summaries[1].statistics.as_ref().unwrap().count, 2);
        app.filter.sources = Some(["B".into(), "A".into()].into_iter().collect());
        app.sort = Some(Sort {
            column: 1,
            descending: true,
            by_source: false,
        });
        app.rebuild();
        assert_eq!(app.table.plotted_rows(&app.view), vec![0]);
        app.undo(false);
        assert_eq!(app.table.plotted_rows(&app.view), vec![1, 0]);
        app.undo(true);
        assert_eq!(app.table.plotted_rows(&app.view), vec![0]);
    }
}
