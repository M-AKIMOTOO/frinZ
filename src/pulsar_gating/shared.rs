use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) const PLOT_FONT_SCALE: f64 = 1.2;
pub(crate) const LEGEND_FONT_SCALE: f64 = 1.2;

pub(crate) fn compress_plot_png(path: &Path) {
    // Runtime priority mode: skip PNG post-compression for faster end-to-end execution.
    let _ = path;
}

pub(crate) fn scaled_font_size(size: i32) -> i32 {
    ((size as f64) * PLOT_FONT_SCALE).round() as i32
}

pub(crate) fn scaled_legend_font_size(size: i32) -> i32 {
    ((scaled_font_size(size) as f64) * LEGEND_FONT_SCALE).round() as i32
}

pub(crate) fn output_stem(input: &Path) -> String {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("pulsar")
        .to_string();
    stem.strip_suffix(".cor")
        .unwrap_or(stem.as_str())
        .to_string()
}

pub(crate) fn prepare_output_directory(input: &Path) -> Result<PathBuf> {
    let parent = input.parent().unwrap_or_else(|| Path::new(""));
    let target_name = output_stem(input);
    let output_dir = parent.join("frinZ").join("pulsar_gating").join(target_name);
    fs::create_dir_all(&output_dir)?;
    Ok(output_dir)
}
