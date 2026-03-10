use std::error::Error;
use std::fs::File;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use memmap2::Mmap;
use num_complex::Complex;

use crate::header::{parse_header, CorHeader};
use crate::read::read_visibility_data;

type C32 = Complex<f32>;

struct AutoCorFile {
    #[allow(dead_code)]
    file: File,
    mmap: Mmap,
    header: CorHeader,
    path: PathBuf,
}

pub struct NormAcfContext {
    left: AutoCorFile,
    right: AutoCorFile,
}

impl NormAcfContext {
    pub fn load(input_path: &Path, cross_header: &CorHeader) -> Result<Self, Box<dyn Error>> {
        let station1 = cross_header.station1_name.trim();
        let station2 = cross_header.station2_name.trim();
        let left_path = build_auto_corr_path(input_path, station1, station1)?;
        let right_path = build_auto_corr_path(input_path, station2, station2)?;

        let left = load_auto_cor_file(&left_path)?;
        let right = load_auto_cor_file(&right_path)?;

        validate_auto_header(cross_header, &left.header, station1)?;
        validate_auto_header(cross_header, &right.header, station2)?;

        Ok(Self { left, right })
    }
    pub fn normalize_cross_visibility(
        &self,
        cross_vis: &mut [C32],
        cross_obs_time: DateTime<Utc>,
        cross_effective_integ_time: f32,
        length: i32,
        skip: i32,
        loop_index: i32,
        is_cumulate: bool,
        pp_flag_ranges: &[(u32, u32)],
    ) -> Result<(), Box<dyn Error>> {
        let (left_vis, left_obs_time, left_integ_time) = read_norm_data(
            &self.left,
            length,
            skip,
            loop_index,
            is_cumulate,
            pp_flag_ranges,
        )?;
        let (right_vis, right_obs_time, right_integ_time) = read_norm_data(
            &self.right,
            length,
            skip,
            loop_index,
            is_cumulate,
            pp_flag_ranges,
        )?;

        validate_loop_timing(
            cross_obs_time,
            cross_effective_integ_time,
            left_obs_time,
            left_integ_time,
            self.left.path.as_path(),
        )?;
        validate_loop_timing(
            cross_obs_time,
            cross_effective_integ_time,
            right_obs_time,
            right_integ_time,
            self.right.path.as_path(),
        )?;

        if left_vis.len() != cross_vis.len() || right_vis.len() != cross_vis.len() {
            return Err(format!(
                "Error: autocorrelation data length mismatch (cross={}, auto1={}, auto2={}).",
                cross_vis.len(),
                left_vis.len(),
                right_vis.len()
            )
            .into());
        }

        for ((cross, auto1), auto2) in cross_vis
            .iter_mut()
            .zip(left_vis.iter())
            .zip(right_vis.iter())
        {
            let norm1 = auto1.norm();
            let norm2 = auto2.norm();
            let scale = (norm1 * norm2).sqrt();
            if scale.is_finite() && scale > 0.0 {
                *cross /= scale;
            } else {
                *cross = C32::new(0.0, 0.0);
            }
        }

        Ok(())
    }

    pub fn path_pair(&self) -> (&Path, &Path) {
        (&self.left.path, &self.right.path)
    }
}

fn read_norm_data(
    auto: &AutoCorFile,
    length: i32,
    skip: i32,
    loop_index: i32,
    is_cumulate: bool,
    pp_flag_ranges: &[(u32, u32)],
) -> Result<(Vec<C32>, chrono::DateTime<chrono::Utc>, f32), Box<dyn Error>> {
    let mut cursor = Cursor::new(&auto.mmap[..]);
    Ok(read_visibility_data(
        &mut cursor,
        &auto.header,
        length,
        skip,
        loop_index,
        is_cumulate,
        pp_flag_ranges,
    )?)
}

fn load_auto_cor_file(path: &Path) -> Result<AutoCorFile, Box<dyn Error>> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut cursor = Cursor::new(&mmap[..]);
    let header = parse_header(&mut cursor)?;
    Ok(AutoCorFile {
        file,
        mmap,
        header,
        path: path.to_path_buf(),
    })
}

fn validate_auto_header(
    cross_header: &CorHeader,
    auto_header: &CorHeader,
    station_name: &str,
) -> Result<(), Box<dyn Error>> {
    let auto_station1 = auto_header.station1_name.trim();
    let auto_station2 = auto_header.station2_name.trim();

    if auto_station1 != station_name || auto_station2 != station_name {
        return Err(format!(
            "Error: autocorrelation file station mismatch for {} (found {}-{}).",
            station_name, auto_station1, auto_station2
        )
        .into());
    }

    if auto_header.fft_point != cross_header.fft_point {
        return Err(format!(
            "Error: autocorrelation FFT point mismatch for {} (cross={}, auto={}).",
            station_name, cross_header.fft_point, auto_header.fft_point
        )
        .into());
    }

    if auto_header.sampling_speed != cross_header.sampling_speed {
        return Err(format!(
            "Error: autocorrelation sampling speed mismatch for {} (cross={}, auto={}).",
            station_name, cross_header.sampling_speed, auto_header.sampling_speed
        )
        .into());
    }

    if auto_header.observing_frequency.to_bits() != cross_header.observing_frequency.to_bits() {
        return Err(format!(
            "Error: autocorrelation observing frequency mismatch for {}.",
            station_name
        )
        .into());
    }

    if auto_header.number_of_sector != cross_header.number_of_sector {
        return Err(format!(
            "Error: autocorrelation sector count mismatch for {} (cross={}, auto={}).",
            station_name, cross_header.number_of_sector, auto_header.number_of_sector
        )
        .into());
    }

    Ok(())
}

fn validate_loop_timing(
    cross_obs_time: DateTime<Utc>,
    cross_effective_integ_time: f32,
    auto_obs_time: DateTime<Utc>,
    auto_effective_integ_time: f32,
    auto_path: &Path,
) -> Result<(), Box<dyn Error>> {
    if auto_obs_time != cross_obs_time {
        return Err(format!(
            "Error: autocorrelation start time mismatch for {} (cross={}, auto={}).",
            auto_path.display(),
            cross_obs_time.format("%Y-%m-%d %H:%M:%S"),
            auto_obs_time.format("%Y-%m-%d %H:%M:%S"),
        )
        .into());
    }

    if (auto_effective_integ_time - cross_effective_integ_time).abs() > 1.0e-6 {
        return Err(format!(
            "Error: autocorrelation effective integration time mismatch for {} (cross={}, auto={}).",
            auto_path.display(),
            cross_effective_integ_time,
            auto_effective_integ_time,
        )
        .into());
    }

    Ok(())
}

fn build_auto_corr_path(
    input_path: &Path,
    station1: &str,
    station2: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    let parent = input_path.parent().unwrap_or_else(|| Path::new(""));
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("Error: invalid input filename for --norm-acf.")?;
    let extension = input_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("cor");

    let parts: Vec<&str> = stem.split('_').collect();
    if parts.len() < 3 {
        return Err(format!(
            "Error: input filename '{}' does not match ANT1_ANT2_suffix.cor format required by --norm-acf.",
            input_path.display()
        )
        .into());
    }

    let suffix = parts[2..].join("_");
    Ok(parent.join(format!(
        "{}_{}_{}.{}",
        station1, station2, suffix, extension
    )))
}

#[cfg(test)]
mod tests {
    use super::build_auto_corr_path;
    use std::path::Path;

    #[test]
    fn auto_corr_path_uses_input_suffix() {
        let path = build_auto_corr_path(
            Path::new("/tmp/YAMAGU32_YAMAGU34_2024060000000_label.cor"),
            "YAMAGU32",
            "YAMAGU32",
        )
        .unwrap();

        assert_eq!(
            path,
            Path::new("/tmp/YAMAGU32_YAMAGU32_2024060000000_label.cor")
        );
    }
}
