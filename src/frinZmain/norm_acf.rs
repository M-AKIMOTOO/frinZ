// ACF helpers expose normalization inputs explicitly for reproducibility.
#![allow(clippy::too_many_arguments)]
use std::error::Error;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use byteorder::{LittleEndian, ReadBytesExt};
use chrono::{DateTime, TimeZone, Utc};
use num_complex::Complex;

use crate::header::{parse_header, CorHeader};
use crate::input_support::{open_input_data, InputData};
use crate::read::{
    calculate_sector_range, normalize_effective_integration_time, EFFECTIVE_INTEG_TIME_OFFSET,
    FILE_HEADER_SIZE, SECTOR_HEADER_SIZE,
};

type C32 = Complex<f32>;

struct AutoCorFile {
    input: InputData,
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
        let (left_obs_time, left_integ_time) =
            read_norm_timing(&self.left, length, skip, loop_index, is_cumulate)?;
        let (right_obs_time, right_integ_time) =
            read_norm_timing(&self.right, length, skip, loop_index, is_cumulate)?;

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

        let expected_len =
            norm_sample_count(&self.left.header, length, skip, loop_index, is_cumulate)?;
        let right_expected_len =
            norm_sample_count(&self.right.header, length, skip, loop_index, is_cumulate)?;
        if expected_len != cross_vis.len() || right_expected_len != cross_vis.len() {
            return Err(format!(
                "Error: autocorrelation data length mismatch (cross={}, auto1={}, auto2={}).",
                cross_vis.len(),
                expected_len,
                right_expected_len
            )
            .into());
        }

        normalize_from_auto_streams(
            cross_vis,
            &self.left,
            &self.right,
            length,
            skip,
            loop_index,
            is_cumulate,
            pp_flag_ranges,
        )
    }

    pub fn path_pair(&self) -> (&Path, &Path) {
        (&self.left.path, &self.right.path)
    }
}

fn norm_sample_count(
    header: &CorHeader,
    length: i32,
    skip: i32,
    loop_index: i32,
    is_cumulate: bool,
) -> Result<usize, Box<dyn Error>> {
    let (start, end) = calculate_sector_range(header, length, skip, loop_index, is_cumulate);
    if start >= end {
        return Err("skip/length の指定が利用可能なセクター数を超えています".into());
    }
    Ok((end - start) as usize * (header.fft_point / 2) as usize)
}

fn read_norm_timing(
    auto: &AutoCorFile,
    length: i32,
    skip: i32,
    loop_index: i32,
    is_cumulate: bool,
) -> Result<(chrono::DateTime<chrono::Utc>, f32), Box<dyn Error>> {
    let sector_size = (8 + auto.header.fft_point / 4) * 16;
    let (start, end) = calculate_sector_range(&auto.header, length, skip, loop_index, is_cumulate);
    if start >= end {
        return Err("skip/length の指定が利用可能なセクター数を超えています".into());
    }

    let mut cursor = Cursor::new(auto.input.as_slice());
    let sector_start_pos = FILE_HEADER_SIZE + start as u64 * sector_size as u64;
    cursor.set_position(sector_start_pos);
    let correlation_time_sec = cursor.read_i32::<LittleEndian>()?;
    let obs_time = chrono::Utc
        .timestamp_opt(correlation_time_sec as i64, 0)
        .single()
        .ok_or_else(|| format!("Invalid timestamp seconds: {}", correlation_time_sec))?;

    cursor.set_position(sector_start_pos + EFFECTIVE_INTEG_TIME_OFFSET);
    let effective_integ_time =
        normalize_effective_integration_time(cursor.read_f32::<LittleEndian>()?);

    Ok((obs_time, effective_integ_time))
}

fn normalize_from_auto_streams(
    cross_vis: &mut [C32],
    left: &AutoCorFile,
    right: &AutoCorFile,
    length: i32,
    skip: i32,
    loop_index: i32,
    is_cumulate: bool,
    pp_flag_ranges: &[(u32, u32)],
) -> Result<(), Box<dyn Error>> {
    let fft_point_half = (left.header.fft_point / 2) as usize;
    let left_sector_size = (8 + left.header.fft_point / 4) * 16;
    let right_sector_size = (8 + right.header.fft_point / 4) * 16;
    let (start, end) = calculate_sector_range(&left.header, length, skip, loop_index, is_cumulate);

    let mut left_cursor = Cursor::new(left.input.as_slice());
    let mut right_cursor = Cursor::new(right.input.as_slice());
    let mut sample_idx = 0usize;

    for sector in start..end {
        let left_sector_start = FILE_HEADER_SIZE + sector as u64 * left_sector_size as u64;
        let right_sector_start = FILE_HEADER_SIZE + sector as u64 * right_sector_size as u64;
        left_cursor.set_position(left_sector_start + SECTOR_HEADER_SIZE);
        right_cursor.set_position(right_sector_start + SECTOR_HEADER_SIZE);

        let is_pp_flagged = pp_flag_ranges.iter().any(|(flag_start, flag_end)| {
            sector as u32 >= *flag_start && sector as u32 <= *flag_end
        });

        for _ in 0..fft_point_half {
            let auto1 = C32::new(
                left_cursor.read_f32::<LittleEndian>()?,
                left_cursor.read_f32::<LittleEndian>()?,
            );
            let auto2 = C32::new(
                right_cursor.read_f32::<LittleEndian>()?,
                right_cursor.read_f32::<LittleEndian>()?,
            );

            if is_pp_flagged {
                cross_vis[sample_idx] = C32::new(0.0, 0.0);
            } else {
                let scale = (auto1.norm() * auto2.norm()).sqrt();
                if scale.is_finite() && scale > 0.0 {
                    cross_vis[sample_idx] /= scale;
                } else {
                    cross_vis[sample_idx] = C32::new(0.0, 0.0);
                }
            }
            sample_idx += 1;
        }
    }

    Ok(())
}

fn load_auto_cor_file(path: &Path) -> Result<AutoCorFile, Box<dyn Error>> {
    let input = open_input_data(path)?;
    let mut cursor = Cursor::new(input.as_slice());
    let header = parse_header(&mut cursor)?;
    Ok(AutoCorFile {
        input,
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
