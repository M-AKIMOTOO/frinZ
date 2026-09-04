use byteorder::ReadBytesExt;
use chrono::{DateTime, TimeZone, Utc};
use num_complex::Complex;
use std::fs;
use std::io::{self, Cursor, Error, ErrorKind, Read};
use std::path::Path;

use crate::header::{parse_header, validate_cor_payload, CorHeader};

pub type C32 = Complex<f32>;

// ファイルヘッダーのサイズ (256 バイト)
pub const FILE_HEADER_SIZE: u64 = 256;
// 各セクターのヘッダーサイズ (128 バイト)
pub const SECTOR_HEADER_SIZE: u64 = 128;
// セクターヘッダー内での有効積分時間のオフセット
pub const EFFECTIVE_INTEG_TIME_OFFSET: u64 = 112;

/// Data read from a `.cor` file for library users.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CorData {
    pub header: CorHeader,
    pub visibility: Vec<C32>,
    pub obs_time: DateTime<Utc>,
    pub effective_integ_time: f32,
    pub first_sector: i32,
    pub sectors_read: i32,
}

/// Options for reading visibility data from a `.cor` file.
///
/// `length = None` or `Some(0)` means "read all remaining sectors after `skip`".
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct CorReadOptions {
    pub length: Option<i32>,
    pub skip: i32,
    pub loop_index: i32,
    pub is_cumulate: bool,
    pub pp_flag_ranges: Vec<(u32, u32)>,
}

/// Read the full visibility payload from a `.cor` file.
#[allow(dead_code)]
pub fn read_cor_file<P: AsRef<Path>>(path: P) -> io::Result<CorData> {
    read_cor_file_with_options(path, &CorReadOptions::default())
}

/// Read visibility data from a `.cor` file with library-friendly options.
#[allow(dead_code)]
pub fn read_cor_file_with_options<P: AsRef<Path>>(
    path: P,
    options: &CorReadOptions,
) -> io::Result<CorData> {
    let bytes = fs::read(path)?;
    read_cor_bytes(&bytes, options)
}

/// Read visibility data from in-memory `.cor` bytes.
#[allow(dead_code)]
pub fn read_cor_bytes(bytes: &[u8], options: &CorReadOptions) -> io::Result<CorData> {
    let mut cursor = Cursor::new(bytes);
    let header = parse_header(&mut cursor)?;
    let (length, loop_index) = resolve_read_params(&header, options);
    let (first_sector, end_sector) = calculate_sector_range(
        &header,
        length,
        options.skip,
        loop_index,
        options.is_cumulate,
    );

    let (visibility, obs_time, effective_integ_time) = read_visibility_data(
        &mut cursor,
        &header,
        length,
        options.skip,
        loop_index,
        options.is_cumulate,
        &options.pp_flag_ranges,
    )?;

    Ok(CorData {
        header,
        visibility,
        obs_time,
        effective_integ_time,
        first_sector,
        sectors_read: end_sector - first_sector,
    })
}

fn resolve_read_params(header: &CorHeader, options: &CorReadOptions) -> (i32, i32) {
    if let Some(length) = options.length.filter(|length| *length > 0) {
        return (length, options.loop_index);
    }

    let total_sectors = header.number_of_sector.max(0);
    if options.is_cumulate {
        return (total_sectors, 0);
    }

    let skip = options.skip.clamp(0, total_sectors);
    (total_sectors.saturating_sub(skip), 0)
}

pub fn normalize_effective_integration_time(value: f32) -> f32 {
    if !value.is_finite() || value <= 0.0 {
        return 1.0;
    }
    if (value - 1.0).abs() <= 0.1 {
        return 1.0;
    }

    let mut a = 0.1f32;
    while a >= 0.000001 {
        if value >= a * 0.9 && value <= a * 1.1 {
            return a;
        }
        a /= 10.0;
    }

    value
}

pub fn read_visibility_data(
    cursor: &mut Cursor<&[u8]>,
    header: &CorHeader,
    length: i32,
    skip: i32,
    loop_index: i32,
    is_cumulate: bool,
    pp_flag_ranges: &[(u32, u32)],
) -> io::Result<(Vec<C32>, DateTime<Utc>, f32)> {
    validate_cor_payload(header, cursor.get_ref().len())?;
    let sector_size = (8 + header.fft_point / 4) * 16;

    let (actual_length_start, length_end) =
        calculate_sector_range(header, length, skip, loop_index, is_cumulate);

    if actual_length_start >= length_end {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "skip/length の指定が利用可能なセクター数を超えています",
        ));
    }

    let num_sectors_to_read = (length_end - actual_length_start) as usize;
    let fft_point_half = (header.fft_point / 2) as usize;
    let mut complex_vec = Vec::with_capacity(num_sectors_to_read * fft_point_half);
    let mut obs_time = Utc
        .timestamp_opt(0, 0)
        .single()
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Failed to create initial timestamp"))?;
    let mut effective_integ_time = 0.0;

    let mut non_finite_samples = 0usize;

    for i in 0..num_sectors_to_read {
        let sector_start_pos =
            FILE_HEADER_SIZE + (actual_length_start as u64 + i as u64) * sector_size as u64;
        cursor.set_position(sector_start_pos);

        let correlation_time_sec = cursor.read_i32::<byteorder::LittleEndian>()?;
        if i == 0 {
            obs_time = Utc
                .timestamp_opt(correlation_time_sec as i64, 0)
                .single()
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::InvalidData,
                        format!("Invalid timestamp seconds: {}", correlation_time_sec),
                    )
                })?;
        }

        cursor.set_position(sector_start_pos + EFFECTIVE_INTEG_TIME_OFFSET);
        effective_integ_time = cursor.read_f32::<byteorder::LittleEndian>()?;
        cursor.set_position(sector_start_pos + SECTOR_HEADER_SIZE);

        let current_pp = actual_length_start as u32 + i as u32;
        let is_pp_flagged = pp_flag_ranges
            .iter()
            .any(|(start, end)| current_pp >= *start && current_pp <= *end);

        for _ in 0..fft_point_half {
            let real = cursor.read_f32::<byteorder::LittleEndian>()?;
            let imag = cursor.read_f32::<byteorder::LittleEndian>()?;
            let mut sample = C32::new(real, imag);
            if !sample.re.is_finite() || !sample.im.is_finite() {
                non_finite_samples += 1;
                sample = C32::new(0.0, 0.0);
            }

            if is_pp_flagged {
                complex_vec.push(C32::new(0.0, 0.0));
            } else {
                complex_vec.push(sample);
            }
        }
    }

    if non_finite_samples > 0 {
        eprintln!(
            "#WARN: Replaced {} non-finite visibility samples with 0+0j (sectors {}-{}).",
            non_finite_samples,
            actual_length_start,
            actual_length_start + num_sectors_to_read as i32 - 1
        );
    }

    effective_integ_time = normalize_effective_integration_time(effective_integ_time);

    Ok((complex_vec, obs_time, effective_integ_time))
}

pub fn read_sector_header(
    cursor: &mut Cursor<&[u8]>,
    header: &CorHeader,
    length: i32,
    skip: i32,
    loop_index: i32,
    is_cumulate: bool,
) -> io::Result<Vec<Vec<u8>>> {
    validate_cor_payload(header, cursor.get_ref().len())?;
    // 各セクターのサイズを計算します。
    let sector_size = (8 + header.fft_point / 4) * 16;

    let (actual_length_start, length_end) =
        calculate_sector_range(header, length, skip, loop_index, is_cumulate);

    if actual_length_start >= length_end {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "skip/length の指定が利用可能なセクター数を超えています",
        ));
    }

    let num_sectors_to_read = (length_end - actual_length_start) as usize;
    let mut sector_headers = Vec::with_capacity(num_sectors_to_read);

    for i in 0..num_sectors_to_read {
        // 各セクターのヘッダーの開始位置を計算 (ファイル先頭から256バイトのオフセット)
        let sector_start_pos =
            FILE_HEADER_SIZE + (actual_length_start as u64 + i as u64) * sector_size as u64;
        cursor.set_position(sector_start_pos);

        let mut header_buf = vec![0u8; SECTOR_HEADER_SIZE as usize];
        cursor.read_exact(&mut header_buf)?;
        sector_headers.push(header_buf);
    }

    Ok(sector_headers)
}

/// 読み込むセクターの範囲を計算するヘルパー関数
pub fn calculate_sector_range(
    header: &CorHeader,
    length: i32,
    skip: i32,
    loop_index: i32,
    is_cumulate: bool,
) -> (i32, i32) {
    let total_sectors = header.number_of_sector;

    let mut start = if is_cumulate {
        skip
    } else {
        skip.saturating_add(length.saturating_mul(loop_index))
            .clamp(i32::MIN, total_sectors)
    };

    if start < 0 {
        start = 0;
    } else if start > total_sectors {
        start = total_sectors;
    }

    let mut end = start.saturating_add(length);

    if end > total_sectors {
        end = total_sectors;
    }

    if end < start {
        end = start;
    }

    (start, end)
}

#[cfg(test)]
mod tests {
    use super::{calculate_sector_range, resolve_read_params, CorReadOptions};
    use crate::header::CorHeader;

    fn header_with_sectors(number_of_sector: i32) -> CorHeader {
        CorHeader {
            number_of_sector,
            ..CorHeader::default()
        }
    }

    #[test]
    fn default_read_options_read_all_sectors() {
        let header = header_with_sectors(28800);
        let options = CorReadOptions::default();

        assert_eq!(resolve_read_params(&header, &options), (28800, 0));
    }

    #[test]
    fn zero_length_reads_remaining_sectors_after_skip() {
        let header = header_with_sectors(28800);
        let options = CorReadOptions {
            length: Some(0),
            skip: 120,
            loop_index: 7,
            ..CorReadOptions::default()
        };

        assert_eq!(resolve_read_params(&header, &options), (28680, 0));
    }

    #[test]
    fn cumulative_range_starts_after_skip() {
        let header = header_with_sectors(100);
        assert_eq!(calculate_sector_range(&header, 30, 10, 0, true), (10, 40));
        assert_eq!(calculate_sector_range(&header, 200, 10, 0, true), (10, 100));
    }

    #[test]
    fn explicit_length_preserves_loop_index() {
        let header = header_with_sectors(28800);
        let options = CorReadOptions {
            length: Some(1440),
            skip: 0,
            loop_index: 3,
            ..CorReadOptions::default()
        };

        assert_eq!(resolve_read_params(&header, &options), (1440, 3));
    }
}
