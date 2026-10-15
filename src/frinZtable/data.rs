use super::formula::Expression;
use super::stats::Statistics;
use anyhow::{Context, Result, bail};
use chrono::{NaiveDate, NaiveDateTime};
use serde::Serialize;
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalculatedColumn {
    pub column: usize,
    pub formula: String,
    pub expression: Expression,
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct Column {
    pub name: String,
    pub unit: String,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Table {
    pub columns: Vec<Column>,
    pub rows: Vec<Vec<String>>,
    pub comments: Vec<String>,
    pub calculated: Vec<CalculatedColumn>,
    pub plot_enabled: Vec<bool>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Filter {
    pub source: String,
    /// None means every source; an empty set means no sources.
    pub sources: Option<BTreeSet<String>>,
    pub text: String,
    pub numeric_column: Option<usize>,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Sort {
    pub column: usize,
    pub descending: bool,
    pub by_source: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ColumnSummary {
    pub column: String,
    pub unit: String,
    pub statistics: Option<Statistics>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceSummary {
    pub source: String,
    pub rows: usize,
    pub statistics: Option<Statistics>,
}

pub fn number(text: &str) -> Option<f64> {
    text.trim().parse::<f64>().ok().filter(|v| v.is_finite())
}

pub fn epoch_mjd(text: &str) -> Option<f64> {
    let text = text.trim();
    let date = NaiveDateTime::parse_from_str(text, "%Y/%jT%H:%M:%S%.f")
        .or_else(|_| NaiveDateTime::parse_from_str(text, "%Y/%j %H:%M:%S%.f"))
        .or_else(|_| NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f"))
        .ok()?;
    let origin = NaiveDate::from_ymd_opt(1858, 11, 17)?.and_hms_opt(0, 0, 0)?;
    Some((date - origin).num_milliseconds() as f64 / 86_400_000.0)
}

fn fields(line: &str, delimiter: u8) -> Result<Vec<String>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(delimiter)
        .flexible(true)
        .from_reader(line.as_bytes());
    let record = reader.records().next().transpose()?.unwrap_or_default();
    Ok(record.iter().map(str::to_owned).collect())
}

fn looks_like_header(values: &[String]) -> bool {
    values.len() > 1
        && values
            .iter()
            .filter(|value| number(value).is_none() && epoch_mjd(value).is_none())
            .count()
            * 2
            >= values.len()
}

fn looks_like_units(values: &[String]) -> bool {
    !values.is_empty()
        && values.iter().all(|value| {
            let value = value.trim();
            value.is_empty() || value == "-" || (value.contains('[') && value.ends_with(']'))
        })
}

fn default_columns(frequency: bool) -> Vec<Column> {
    let header = if frequency {
        frinZ::output::format_freq_tsv_header("Station1", "Station2")
    } else {
        frinZ::output::format_delay_tsv_header("Station1", "Station2")
    };
    // Use the output writer as the single source of truth for names and units.
    Table::parse(&header, frequency)
        .expect("frinZ emits a valid header")
        .columns
}

impl Table {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            fs::read_to_string(path).with_context(|| format!("Cannot read {}", path.display()))?;
        Self::parse(&text, path.to_string_lossy().contains("_freq"))
    }

    pub fn parse(text: &str, frequency: bool) -> Result<Self> {
        let mut table = Self::default();
        let mut header: Option<Vec<String>> = None;
        let mut units: Option<Vec<String>> = None;
        let mut excluded_rows = Vec::new();
        let lines: Vec<&str> = text.trim_start_matches('\u{feff}').lines().collect();
        let first_data = lines
            .iter()
            .find(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'));
        let delimiter = if first_data.is_some_and(|line| !line.contains('\t') && line.contains(','))
        {
            b','
        } else {
            b'\t'
        };
        let first_width = first_data
            .map(|line| fields(line.trim_end_matches('\r'), delimiter))
            .transpose()?
            .map(|values| values.len());
        let expected_width = first_width.or_else(|| {
            lines.iter().find_map(|line| {
                let comment = line.trim_start().strip_prefix('#')?;
                let values =
                    fields(comment.strip_prefix(' ').unwrap_or(comment), delimiter).ok()?;
                looks_like_header(&values).then_some(values.len())
            })
        });
        let mut header_line = None;
        let mut units_line = None;

        // frinZ output files use comment-prefixed headers, but their first
        // column varies between products (Epoch, Delay, Frequency, etc.).
        // Match any tabular header to the width of the data rather than
        // assuming one specific analysis table.
        let mut best_comment_header: Option<(usize, usize, Vec<String>)> = None;
        for (index, raw) in lines.iter().enumerate() {
            let line = raw.trim_end_matches('\r');
            let Some(comment) = line.trim_start().strip_prefix('#') else {
                continue;
            };
            let comment = comment.strip_prefix(' ').unwrap_or(comment);
            let values = fields(comment, delimiter)?;
            if expected_width == Some(values.len()) && looks_like_header(&values) {
                let known = values.first().is_some_and(|name| {
                    ["Epoch", "Source", "Time", "MJD", "Frequency", "Delay"]
                        .iter()
                        .any(|candidate| name.eq_ignore_ascii_case(candidate))
                });
                let next_comment_is_units = lines.iter().skip(index + 1).find_map(|next| {
                    let next = next.trim_end_matches('\r').trim_start();
                    next.strip_prefix('#').and_then(|value| {
                        fields(value.strip_prefix(' ').unwrap_or(value), delimiter)
                            .ok()
                            .filter(|fields| {
                                fields.len() == values.len() && looks_like_units(fields)
                            })
                    })
                });
                let score = usize::from(known) * 2 + usize::from(next_comment_is_units.is_some());
                if best_comment_header
                    .as_ref()
                    .is_none_or(|(_, best_score, _)| score > *best_score)
                {
                    best_comment_header = Some((index, score, values));
                }
            }
        }
        if let Some((index, _, names)) = best_comment_header {
            header_line = Some(index);
            header = Some(names);
            units_line = lines
                .iter()
                .enumerate()
                .skip(index + 1)
                .find_map(|(i, raw)| {
                    let line = raw.trim_end_matches('\r').trim_start();
                    let comment = line.strip_prefix('#')?;
                    let values =
                        fields(comment.strip_prefix(' ').unwrap_or(comment), delimiter).ok()?;
                    (values.len() == header.as_ref()?.len() && looks_like_units(&values))
                        .then_some(i)
                });
            if let Some(index) = units_line {
                let line = lines[index].trim_end_matches('\r').trim_start();
                let comment = line.strip_prefix('#').unwrap_or(line);
                units = Some(fields(
                    comment.strip_prefix(' ').unwrap_or(comment),
                    delimiter,
                )?);
            }
        } else if let Some((index, raw)) = lines
            .iter()
            .enumerate()
            .find(|(_, line)| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
        {
            let values = fields(raw.trim_end_matches('\r'), delimiter)?;
            if looks_like_header(&values) {
                header_line = Some(index);
                header = Some(values.clone());
                if let Some((unit_index, unit_raw)) =
                    lines.iter().enumerate().skip(index + 1).find(|(_, line)| {
                        !line.trim().is_empty() && !line.trim_start().starts_with('#')
                    })
                {
                    let unit_values = fields(unit_raw.trim_end_matches('\r'), delimiter)?;
                    if unit_values.len() == values.len() && looks_like_units(&unit_values) {
                        units_line = Some(unit_index);
                        units = Some(unit_values);
                    }
                }
            }
        }

        for (line_no, raw) in lines.iter().enumerate() {
            let line = raw.trim_end_matches('\r');
            if line.trim().is_empty() {
                continue;
            }
            if Some(line_no) == header_line || Some(line_no) == units_line {
                continue;
            }
            if let Some(comment) = line.trim_start().strip_prefix('#') {
                if let Some(excluded) = comment
                    .trim_start()
                    .strip_prefix("frinZtable plot-excluded:")
                {
                    for value in excluded.split(',').filter(|v| !v.trim().is_empty()) {
                        let row = value
                            .trim()
                            .parse::<usize>()
                            .context("Invalid plot-excluded row number")?;
                        if row == 0 {
                            bail!("Plot-excluded row numbers start at 1");
                        }
                        excluded_rows.push(row - 1);
                    }
                    continue;
                }
                table.comments.push(line.to_owned());
                continue;
            }
            let values =
                fields(line, delimiter).with_context(|| format!("Line {}", line_no + 1))?;
            let expected = header
                .as_ref()
                .map(Vec::len)
                .or_else(|| table.rows.first().map(Vec::len));
            if expected.is_some_and(|n| n != values.len()) {
                bail!(
                    "Line {}: expected {} columns, found {}",
                    line_no + 1,
                    expected.unwrap(),
                    values.len()
                );
            }
            table.rows.push(values);
        }
        let width = header
            .as_ref()
            .map(Vec::len)
            .or_else(|| table.rows.first().map(Vec::len))
            .context("No table data or column header found")?;
        table.columns = if let Some(names) = header {
            // frinZ uses one station heading followed by two empty headings.
            names
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    let station_base = i - i.saturating_sub(10) % 3;
                    let name = if (10..16).contains(&i)
                        && names
                            .get(station_base)
                            .is_some_and(|s| s.ends_with("-azel"))
                    {
                        let station = names[station_base].trim_end_matches("-azel");
                        format!("{station}-{}", ["Az", "El", "Hgt"][(i - 10) % 3])
                    } else if name.is_empty() {
                        format!("Column{}", i + 1)
                    } else {
                        name.to_owned()
                    };
                    Column {
                        name,
                        unit: units
                            .as_ref()
                            .and_then(|u| u.get(i))
                            .cloned()
                            .unwrap_or_default(),
                    }
                })
                .collect()
        } else if width == 21 {
            default_columns(frequency)
        } else {
            (0..width)
                .map(|i| Column {
                    name: format!("Column{}", i + 1),
                    unit: String::new(),
                })
                .collect()
        };
        table.plot_enabled = vec![true; table.rows.len()];
        for row in excluded_rows {
            let enabled = table
                .plot_enabled
                .get_mut(row)
                .context("Plot-excluded row is outside the table")?;
            *enabled = false;
        }
        Ok(table)
    }

    pub fn column(&self, name: &str) -> Option<usize> {
        self.columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(name))
    }

    pub fn numeric_columns(&self) -> Vec<usize> {
        (0..self.columns.len())
            .filter(|&c| self.rows.iter().any(|r| number(&r[c]).is_some()))
            .collect()
    }

    pub fn source(&self, row: usize) -> &str {
        self.column("Source")
            .map(|c| self.rows[row][c].as_str())
            .unwrap_or("All rows")
    }

    pub fn sources(&self) -> Vec<String> {
        let mut sources: Vec<String> = (0..self.rows.len())
            .map(|r| self.source(r).to_owned())
            .collect();
        sources.sort();
        sources.dedup();
        sources
    }

    pub fn view(&self, filter: &Filter, sort: Option<Sort>) -> Vec<usize> {
        let query = filter.text.to_lowercase();
        let mut view: Vec<usize> = (0..self.rows.len())
            .filter(|&r| {
                (filter.source.is_empty() || self.source(r) == filter.source)
                    && filter
                        .sources
                        .as_ref()
                        .is_none_or(|sources| sources.contains(self.source(r)))
                    && (query.is_empty()
                        || self.rows[r]
                            .iter()
                            .any(|s| s.to_lowercase().contains(&query)))
                    && filter.numeric_column.is_none_or(|c| {
                        if filter.min.is_none() && filter.max.is_none() {
                            return true;
                        }
                        number(&self.rows[r][c]).is_some_and(|v| {
                            filter.min.is_none_or(|min| v >= min)
                                && filter.max.is_none_or(|max| v <= max)
                        })
                    })
            })
            .collect();
        if let Some(sort) = sort {
            view.sort_by(|&a, &b| {
                let source_cmp = if sort.by_source {
                    self.source(a).cmp(self.source(b))
                } else {
                    Ordering::Equal
                };
                source_cmp.then_with(|| {
                    compare_cells(
                        &self.rows[a][sort.column],
                        &self.rows[b][sort.column],
                        sort.descending,
                    )
                })
            });
        }
        view
    }

    pub fn statistics(&self, rows: &[usize], column: usize) -> Option<Statistics> {
        Statistics::calculate(rows.iter().map(|&r| number(&self.rows[r][column])))
    }

    pub fn plotted_rows(&self, view: &[usize]) -> Vec<usize> {
        view.iter()
            .copied()
            .filter(|&row| self.plot_enabled[row])
            .collect()
    }

    pub fn summaries(&self, rows: &[usize]) -> Vec<ColumnSummary> {
        self.columns
            .iter()
            .enumerate()
            .map(|(c, column)| ColumnSummary {
                column: column.name.clone(),
                unit: column.unit.clone(),
                statistics: self.statistics(rows, c),
            })
            .collect()
    }

    pub fn source_summaries(&self, rows: &[usize], column: usize) -> Vec<SourceSummary> {
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for &r in rows {
            groups.entry(self.source(r).to_owned()).or_default().push(r);
        }
        groups
            .into_iter()
            .map(|(source, group)| SourceSummary {
                source,
                rows: group.len(),
                statistics: self.statistics(&group, column),
            })
            .collect()
    }

    pub fn tsv(&self, rows: &[usize], headers: bool) -> Result<String> {
        let mut bytes = Vec::new();
        if headers {
            let excluded: Vec<String> = rows
                .iter()
                .enumerate()
                .filter(|(_, &row)| !self.plot_enabled[row])
                .map(|(i, _)| (i + 1).to_string())
                .collect();
            if !excluded.is_empty() {
                bytes.extend_from_slice(
                    format!("# frinZtable plot-excluded: {}\n", excluded.join(",")).as_bytes(),
                );
            }
            for comment in &self.comments {
                bytes.extend_from_slice(comment.as_bytes());
                bytes.push(b'\n');
            }
            // Comment headers remain readable by frinZ and numpy.loadtxt.
            let names: Vec<&str> = self.columns.iter().map(|c| c.name.as_str()).collect();
            let units: Vec<&str> = self.columns.iter().map(|c| c.unit.as_str()).collect();
            bytes.extend_from_slice(
                format!("# {}\n# {}\n", names.join("\t"), units.join("\t")).as_bytes(),
            );
        }
        let mut writer = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_writer(bytes);
        for &r in rows {
            writer.write_record(&self.rows[r])?;
        }
        Ok(String::from_utf8(writer.into_inner()?)?)
    }

    /// Validate the entire paste before changing any cells.
    pub fn paste(
        &mut self,
        view: &[usize],
        row: usize,
        column: usize,
        text: &str,
    ) -> Result<usize> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b'\t')
            .flexible(true)
            .from_reader(text.as_bytes());
        let records = reader
            .records()
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if row + records.len() > view.len()
            || records
                .iter()
                .any(|r| column + r.len() > self.columns.len())
        {
            bail!(
                "Paste exceeds the visible table. Select an earlier cell or paste a smaller range."
            );
        }
        let mut count = 0;
        for record in &records {
            for (c, value) in record.iter().enumerate() {
                if value.contains(['\n', '\r']) {
                    bail!("Cells cannot contain newlines");
                }
                if self
                    .calculated
                    .iter()
                    .any(|derived| derived.column == column + c)
                {
                    bail!("Calculated columns are read-only");
                }
            }
        }
        for (offset, record) in records.iter().enumerate() {
            for (c, value) in record.iter().enumerate() {
                self.rows[view[row + offset]][column + c] = value.to_owned();
                count += 1;
            }
        }
        self.recalculate();
        Ok(count)
    }

    pub fn add_calculated(&mut self, name: &str, unit: &str, formula: &str) -> Result<()> {
        if name.trim().is_empty()
            || name.contains(['\t', '\n', '\r'])
            || unit.contains(['\t', '\n', '\r'])
        {
            bail!("Enter a column name and unit without tabs or newlines");
        }
        if self.column(name.trim()).is_some() {
            bail!("Column name already exists");
        }
        let expression = Expression::parse(formula, &self.columns)?;
        let column = self.columns.len();
        self.columns.push(Column {
            name: name.trim().into(),
            unit: unit.trim().into(),
        });
        for row in &mut self.rows {
            row.push(String::new());
        }
        self.calculated.push(CalculatedColumn {
            column,
            formula: formula.into(),
            expression,
        });
        self.recalculate();
        Ok(())
    }

    pub fn recalculate(&mut self) {
        for derived in &self.calculated {
            for row in &mut self.rows {
                row[derived.column] = derived
                    .expression
                    .evaluate(row)
                    .map(|v| v.to_string())
                    .unwrap_or_default();
            }
        }
    }
}

fn compare_cells(a: &str, b: &str, descending: bool) -> Ordering {
    let a_num = number(a).or_else(|| epoch_mjd(a));
    let b_num = number(b).or_else(|| epoch_mjd(b));
    let cmp = match (a_num, b_num) {
        (Some(a), Some(b)) => a.total_cmp(&b),
        (Some(_), None) => return Ordering::Less,
        (None, Some(_)) => return Ordering::Greater,
        _ => a.to_lowercase().cmp(&b.to_lowercase()),
    };
    if descending { cmp.reverse() } else { cmp }
}

pub fn column_letter(mut column: usize) -> String {
    let mut letters = Vec::new();
    loop {
        letters.push((b'A' + (column % 26) as u8) as char);
        if column < 26 {
            break;
        }
        column = column / 26 - 1;
    }
    letters.iter().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frinz_comments_station_headings_and_round_trip() {
        let header = frinZ::output::format_delay_tsv_header("YAMAGU32", "YAMAGU34");
        let row = "2026/283T04:57:00\tc\tNRAO530\t60\t0.5\t1169.5\t47.3\t0.0004\t0\t0\t135\t30\t165\t135\t30\t166\t61323.20625\t-\tFalse\tFalse\t6600";
        let table = Table::parse(&format!("{header}{row}\n"), false).unwrap();
        assert_eq!(table.columns[11].name, "YAMAGU32-El");
        assert_eq!(table.columns[15].name, "YAMAGU34-Hgt");
        let exported = table.tsv(&[0], true).unwrap();
        assert_eq!(Table::parse(&exported, false).unwrap().rows, table.rows);
        assert_eq!(
            Table::parse(row, false).unwrap().columns[7].name,
            "Noise-level"
        );
        assert_eq!(
            Table::parse(row, true).unwrap().columns[7].name,
            "Frequency"
        );
        assert!((epoch_mjd("2026/283T04:57:00").unwrap() - 61323.20625).abs() < 1e-8);
    }

    #[test]
    fn frequency_output_header_identifies_frequency_columns_without_a_filename_hint() {
        let header = frinZ::output::format_freq_tsv_header("YAMAGU32", "YAMAGU34");
        let row = "2026/283T04:57:00\tc\tNRAO530\t60\t0.5\t1169.5\t47.3\t6600.1234567\t0.0004\t0\t135\t30\t165\t135\t30\t166\t61323.20625\t-\tFalse\tFalse\t6600";
        let table = Table::parse(&format!("{header}{row}\n"), false).unwrap();

        assert_eq!(table.columns[7].name, "Frequency");
        assert_eq!(
            table.columns[7].unit, "[MHz]",
            "columns: {:?}",
            table.columns
        );
        assert_eq!(table.columns[8].name, "Noise-level");
        assert_eq!(table.columns[9].name, "Res-Rate");
        assert_eq!(table.rows[0][7], "6600.1234567");
    }

    #[test]
    fn frinz_product_tsvs_accept_arbitrary_commented_headers_and_csv() {
        let product = "# diagnostic product\n# Delay\tPower\tFlag\n# [sample]\t[Jy]\t-\n-2\t1.5\tok\n0\t2.5\tmasked\n";
        let table = Table::parse(product, false).unwrap();
        assert_eq!(
            table
                .columns
                .iter()
                .map(|column| column.name.as_str())
                .collect::<Vec<_>>(),
            ["Delay", "Power", "Flag"]
        );
        assert_eq!(table.columns[0].unit, "[sample]");
        assert_eq!(table.columns[1].unit, "[Jy]");
        assert_eq!(table.rows.len(), 2);
        assert_eq!(table.comments, ["# diagnostic product"]);

        let csv = Table::parse("time,rate,source\n0.0,2.5,A\n1.0,3.5,B\n", false).unwrap();
        assert_eq!(
            csv.columns
                .iter()
                .map(|column| column.name.as_str())
                .collect::<Vec<_>>(),
            ["time", "rate", "source"]
        );
        assert_eq!(csv.sources(), ["A", "B"]);
    }
    #[test]
    fn sort_filter_and_paste_use_displayed_rows() {
        let mut table = Table::parse("Source\tAmp\nA\t10\nB\t2\nA\tNaN\nA\t3", false).unwrap();
        let view = table.view(
            &Filter::default(),
            Some(Sort {
                column: 1,
                descending: true,
                by_source: true,
            }),
        );
        assert_eq!(view, vec![0, 3, 2, 1]);
        table.paste(&view, 1, 1, "7\n8").unwrap();
        assert_eq!(table.rows[3][1], "7");
        assert_eq!(table.rows[2][1], "8");
        let before = table.rows.clone();
        assert!(table.paste(&view, 3, 1, "1\t2").is_err());
        assert_eq!(before, table.rows);
        let filter = Filter {
            source: "A".into(),
            numeric_column: Some(1),
            min: Some(8.0),
            ..Default::default()
        };
        assert_eq!(table.view(&filter, None), vec![0, 2]);
    }
    #[test]
    fn malformed_rows_are_not_silently_truncated() {
        assert!(Table::parse("Epoch\tAmp\nnow\t1\textra", false).is_err());
        assert_eq!(column_letter(26), "AA");
        assert_eq!(column_letter(701), "ZZ");
    }

    #[test]
    fn comments_never_become_measurements_and_derived_columns_recalculate() {
        let input = "# observation notes\n#************\n#      Epoch Label Source Length Amp SNR\nSource\tAmp\tNoise-level\nA\t4\t2\n# intermediate comment\nB\t6\t3\n";
        let mut table = Table::parse(input, false).unwrap();
        assert_eq!(table.rows.len(), 2);
        assert_eq!(table.comments.len(), 4);
        table
            .add_calculated("ratio", "-", "Amp / [Noise-level]")
            .unwrap();
        assert_eq!(table.rows[0][3], "2");
        table.paste(&[0, 1], 0, 1, "8").unwrap();
        assert_eq!(table.rows[0][3], "4");
        assert!(table.paste(&[0, 1], 0, 3, "9").is_err());
        table.add_calculated("squared", "-", "ratio^2").unwrap();
        assert_eq!(table.rows[0][4], "16");
        table.paste(&[0, 1], 0, 2, "0").unwrap();
        assert!(table.rows[0][3].is_empty());
        assert!(table.rows[0][4].is_empty());
        let loaded = Table::parse(&table.tsv(&[1, 0], true).unwrap(), false).unwrap();
        assert_eq!(
            loaded.rows,
            vec![table.rows[1].clone(), table.rows[0].clone()]
        );
        assert_eq!(loaded.columns[3].name, "ratio");
        assert_eq!(loaded.comments, table.comments);
    }

    #[test]
    fn plot_flags_survive_sort_export_and_multiple_source_filters() {
        let mut table = Table::parse("Source\tAmp\nA\t4\nB\t6\nC\t8", false).unwrap();
        table.plot_enabled[1] = false;
        let filter = Filter {
            sources: Some(["A".into(), "B".into()].into_iter().collect()),
            ..Default::default()
        };
        let rows = table.view(
            &filter,
            Some(Sort {
                column: 1,
                descending: true,
                by_source: false,
            }),
        );
        assert_eq!(rows, vec![1, 0]);
        assert_eq!(table.plotted_rows(&rows), vec![0]);
        let loaded = Table::parse(&table.tsv(&rows, true).unwrap(), false).unwrap();
        assert_eq!(loaded.rows[0][0], "B");
        assert_eq!(loaded.plot_enabled, vec![false, true]);
        assert_eq!(loaded.plotted_rows(&[0, 1]), vec![1]);
        assert!(
            table
                .view(
                    &Filter {
                        sources: Some(Default::default()),
                        ..Default::default()
                    },
                    None
                )
                .is_empty()
        );
        assert_eq!(table.view(&Filter::default(), None).len(), 3);
    }
}
