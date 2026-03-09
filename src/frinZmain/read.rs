use byteorder::ReadBytesExt;
use chrono::{DateTime, TimeZone, Utc};
use num_complex::Complex;
use std::io::{self, Cursor, Error, ErrorKind, Read};

use crate::header::CorHeader;

type C32 = Complex<f32>;

// ファイルヘッダーのサイズ (256 バイト)
const FILE_HEADER_SIZE: u64 = 256;
// 各セクターのヘッダーサイズ (128 バイト)
const SECTOR_HEADER_SIZE: u64 = 128;
// セクターヘッダー内での有効積分時間のオフセット
const EFFECTIVE_INTEG_TIME_OFFSET: u64 = 112;

fn normalize_effective_integration_time(value: f32) -> f32 {
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
fn calculate_sector_range(
    header: &CorHeader,
    length: i32,
    skip: i32,
    loop_index: i32,
    is_cumulate: bool,
) -> (i32, i32) {
    let total_sectors = header.number_of_sector;

    let mut start = if is_cumulate {
        0
    } else {
        skip.saturating_add(length.saturating_mul(loop_index))
            .clamp(i32::MIN, total_sectors)
    };

    if start < 0 {
        start = 0;
    } else if start > total_sectors {
        start = total_sectors;
    }

    let mut end = if is_cumulate {
        length
    } else {
        start.saturating_add(length)
    };

    if end > total_sectors {
        end = total_sectors;
    }

    if end < start {
        end = start;
    }

    (start, end)
}
