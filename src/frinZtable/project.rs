use super::{
    app::ViewMode,
    data::{Filter, Sort, Table},
    plot_export::ExportFormat,
    plots::PlotOptions,
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

pub const CURRENT_VERSION: u32 = 1;

/// A complete resumable frinZtable workbook, including table contents and the
/// current analysis and display settings.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Project {
    pub version: u32,
    pub table: Table,
    pub data_path: Option<std::path::PathBuf>,
    pub filter: Filter,
    pub min_text: String,
    pub max_text: String,
    pub sort: Option<Sort>,
    pub statistics_selected: bool,
    pub all_statistics: bool,
    pub group_sort: usize,
    pub group_descending: bool,
    pub plot: PlotOptions,
    pub export_format: ExportFormat,
    pub view_mode: ViewMode,
    pub selected: Option<[usize; 2]>,
    pub anchor: Option<[usize; 2]>,
}

impl Project {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            fs::read(path).with_context(|| format!("Cannot read project {}", path.display()))?;
        let project: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("Cannot parse frinZtable project {}", path.display()))?;
        if project.version != CURRENT_VERSION {
            bail!(
                "Unsupported frinZtable project version {} (this program supports {})",
                project.version,
                CURRENT_VERSION
            );
        }
        anyhow::ensure!(
            project
                .table
                .rows
                .iter()
                .all(|row| row.len() == project.table.columns.len()),
            "Project table contains rows with inconsistent column counts"
        );
        anyhow::ensure!(
            project.table.plot_enabled.len() == project.table.rows.len(),
            "Project plot-selection flags do not match the number of rows"
        );
        Ok(project)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let temporary = path.with_extension("frztable.tmp");
        let bytes =
            serde_json::to_vec_pretty(self).context("Cannot serialize frinZtable project")?;
        fs::write(&temporary, bytes)
            .with_context(|| format!("Cannot write temporary project {}", temporary.display()))?;
        if let Err(error) = fs::rename(&temporary, path) {
            let _ = fs::remove_file(&temporary);
            return Err(error).with_context(|| format!("Cannot save project {}", path.display()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_round_trip_keeps_table_and_analysis_state() {
        let mut table = Table::parse("Source\tAmp\nA\t1\nB\t3", false).unwrap();
        table.plot_enabled[1] = false;
        table.add_calculated("DoubleAmp", "-", "Amp * 2").unwrap();
        let project = Project {
            version: CURRENT_VERSION,
            table,
            data_path: Some(std::path::PathBuf::from("analysis.tsv")),
            filter: Filter {
                text: "target".into(),
                ..Default::default()
            },
            min_text: "0".into(),
            max_text: "10".into(),
            sort: Some(Sort {
                column: 1,
                descending: true,
                by_source: true,
            }),
            statistics_selected: true,
            all_statistics: true,
            group_sort: 2,
            group_descending: true,
            plot: PlotOptions {
                y: 1,
                reference_mjd: Some(61323.0),
                ..Default::default()
            },
            export_format: ExportFormat::Pdf,
            view_mode: ViewMode::Plot,
            selected: None,
            anchor: None,
        };
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("session.frztable");
        project.save(&path).unwrap();
        let loaded = Project::load(&path).unwrap();
        assert_eq!(loaded.table.rows, project.table.rows);
        assert_eq!(loaded.table.plot_enabled, vec![true, false]);
        assert_eq!(loaded.table.calculated[0].formula, "Amp * 2");
        assert_eq!(loaded.filter.text, "target");
        assert_eq!(loaded.sort, project.sort);
        assert_eq!(loaded.plot.reference_mjd, Some(61323.0));
        assert_eq!(loaded.export_format, ExportFormat::Pdf);
        assert_eq!(loaded.view_mode, ViewMode::Plot);
    }
}
