// Bispectrum / closure-phase analysis module.
// This code multiplies three baseline visibilities to form the bispectrum,
// then derives closure phase and related statistics/plots.
use crate::args::Args;
use crate::bandpass::read_bandpass_file;
use crate::search;
use crate::fft;
use crate::header::{parse_header, CorHeader};
use crate::output::write_phase_corrected_spectrum_binary;
use crate::png_compress::{compress_png_with_mode, CompressQuality};
use crate::processing::run_analysis_pipeline;
use crate::read::{read_sector_header, read_visibility_data};
use crate::rfi::parse_rfi_ranges;
use chrono::{DateTime, Utc};
use num_complex::Complex;
use plotters::coord::ranged1d::{KeyPointHint, NoDefaultFormatting, Ranged, ValueFormatter};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use std::error::Error;
use std::fs::{self, File};
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};

type C32 = Complex<f32>;

const FILE_HEADER_SIZE: usize = 256;
const SECTOR_HEADER_SIZE: usize = 128;
const EFFECTIVE_INTEG_TIME_OFFSET: usize = 112;
const NUM_SECTOR_OFFSET: usize = 28;
const ST1_NAME_OFFSET: usize = 32;
const ST2_NAME_OFFSET: usize = 80;
const ST2_POS_X_OFFSET: usize = 96;
const ST2_POS_Y_OFFSET: usize = 104;
const ST2_POS_Z_OFFSET: usize = 112;
const ST2_CODE_OFFSET: usize = 120;
const ST2_CLOCK_DELAY_OFFSET: usize = 216;
const ST2_CLOCK_RATE_OFFSET: usize = 224;
const ST2_CLOCK_ACEL_OFFSET: usize = 232;
const ST2_CLOCK_JERK_OFFSET: usize = 240;
const ST2_CLOCK_SNAP_OFFSET: usize = 248;

fn wrap_phase_deg(value: f32) -> f32 {
    let mut wrapped = (value + 180.0) % 360.0;
    if wrapped < 0.0 {
        wrapped += 360.0;
    }
    wrapped - 180.0
}

fn sanitize_token(token: &str) -> String {
    token
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn normalize_name(name: &str) -> String {
    name.trim().to_string()
}

fn names_equal(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

#[derive(Clone)]
struct PhaseAxis {
    start: f64,
    end: f64,
    step: f64,
}

impl PhaseAxis {
    fn new(range: Range<f64>, step: f64) -> Self {
        let mut start = range.start;
        let mut end = range.end;
        if end < start {
            std::mem::swap(&mut start, &mut end);
        }
        let step = step.abs().max(1e-6);
        PhaseAxis { start, end, step }
    }
}

impl Ranged for PhaseAxis {
    type FormatOption = NoDefaultFormatting;
    type ValueType = f64;

    fn map(&self, value: &f64, limit: (i32, i32)) -> i32 {
        let coord: RangedCoordf64 = (self.start..self.end).into();
        coord.map(value, limit)
    }

    fn key_points<Hint: KeyPointHint>(&self, hint: Hint) -> Vec<f64> {
        let mut points = Vec::new();
        if self.step <= 0.0 {
            return points;
        }
        let max_points = hint.max_num_points();
        if max_points == 0 {
            return points;
        }
        let mut current = (self.start / self.step).ceil() * self.step;
        while current <= self.end + 1e-6 && points.len() < max_points {
            points.push(current.round());
            current += self.step;
        }
        points
    }

    fn range(&self) -> Range<f64> {
        self.start..self.end
    }
}

impl ValueFormatter<f64> for PhaseAxis {
    fn format(value: &f64) -> String {
        format!("{:>4.0}", value.round())
    }
}

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

fn peek_effective_integ_time(path: &Path, header: &CorHeader) -> Result<f32, Box<dyn Error>> {
    if header.number_of_sector <= 0 {
        return Ok(1.0);
    }
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(
        FILE_HEADER_SIZE as u64 + EFFECTIVE_INTEG_TIME_OFFSET as u64,
    ))?;
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    let value = f32::from_le_bytes(buf);
    Ok(normalize_effective_integration_time(value))
}

fn compute_window_params(header: &CorHeader, args: &Args, eff_time: f32) -> (i32, i32) {
    let mut pp = header.number_of_sector;
    if pp <= 0 {
        pp = 1;
    }

    let mut length = if args.length == 0 {
        pp
    } else {
        args.length.max(1)
    };

    let effective_time = eff_time.max(0.0001);
    if args.length != 0 {
        let total_obs_time_seconds = pp as f32 * effective_time;
        if args.length as f32 > total_obs_time_seconds {
            length = (total_obs_time_seconds / effective_time).ceil() as i32;
        } else {
            length = (args.length as f32 / effective_time).ceil() as i32;
        }
        if length <= 0 {
            length = 1;
        }
    }

    let mut loop_count = if (pp - args.skip) / length <= 0 {
        1
    } else if (pp - args.skip) / length <= args.loop_ {
        (pp - args.skip) / length
    } else {
        args.loop_
    };

    if args.cumulate != 0 {
        if args.cumulate >= pp {
            loop_count = 1;
        } else {
            length = args.cumulate;
            loop_count = (pp / args.cumulate).max(1);
        }
    }

    (length.max(1), loop_count.max(1))
}

fn selected_loop_indices(header: &CorHeader, args: &Args, eff_time: f32) -> Vec<i32> {
    if header.number_of_sector <= 0 {
        return Vec::new();
    }

    let (_, loop_count) = compute_window_params(header, args, eff_time);
    (0..loop_count.max(0)).collect()
}

#[derive(Clone)]
struct BaselineInfo {
    path: PathBuf,
    header: CorHeader,
    estimated_samples: usize,
}

#[derive(Clone)]
struct LoopVisibility {
    timestamp: DateTime<Utc>,
    spectrum: Vec<C32>,
    sector_header: Vec<u8>,
    integrated_time_sec: f32,
}

struct LoadedBaseline {
    info: BaselineInfo,
    file_header_raw: Vec<u8>,
    loops: Vec<LoopVisibility>,
    baseline_label: String,
}

#[derive(Clone)]
struct ClosureSample {
    timestamp: DateTime<Utc>,
    raw_complex: [C32; 3],
    baseline_phase_deg: [f32; 3],
    baseline_amp: [f32; 3],
    bispectrum: C32,
    bispectrum_amp: f32,
    bispectrum_amp_cuberoot: f32,
    closure_phase_deg: f32,
    closure_from_baselines_deg: f32,
    normalized_intensity: f32,
}

#[derive(Clone)]
struct StationMeta {
    name: String,
    code: String,
    position: [f64; 3],
    clock: [f64; 5],
}

#[derive(Clone)]
struct BaselineStats {
    mean: C32,
    var_re: f64,
    var_im: f64,
    cov_re_im: f64,
    pseudovar: f64,
    snr: f64,
}

#[derive(Clone)]
struct PairStats {
    c_conj: C32,
    c_plain: C32,
    mu: C32,
    nu: C32,
}

#[derive(Clone)]
struct BispectrumStats {
    mean: C32,
    pseudovar: f64,
    snr: f64,
    amp_mean: f64,
    amp_std: f64,
    phase_circ_std_deg: f64,
}

struct ClosureStats {
    sample_count: usize,
    baseline_stats: [BaselineStats; 3],
    pair_stats: [PairStats; 3],
    bispectrum_stats: BispectrumStats,
    intensity_mean: f64,
    intensity_std: f64,
    intensity_frac_rms: f64,
}

fn station_meta_from_header(header: &CorHeader, station1: bool) -> StationMeta {
    if station1 {
        StationMeta {
            name: header.station1_name.clone(),
            code: header.station1_code.clone(),
            position: header.station1_position,
            clock: [
                header.station1_clock_delay,
                header.station1_clock_rate,
                header.station1_clock_acel,
                header.station1_clock_jerk,
                header.station1_clock_snap,
            ],
        }
    } else {
        StationMeta {
            name: header.station2_name.clone(),
            code: header.station2_code.clone(),
            position: header.station2_position,
            clock: [
                header.station2_clock_delay,
                header.station2_clock_rate,
                header.station2_clock_acel,
                header.station2_clock_jerk,
                header.station2_clock_snap,
            ],
        }
    }
}

fn write_i32_le(buf: &mut [u8], offset: usize, value: i32) {
    if offset + 4 <= buf.len() {
        buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn write_f32_le(buf: &mut [u8], offset: usize, value: f32) {
    if offset + 4 <= buf.len() {
        buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn write_f64_le(buf: &mut [u8], offset: usize, value: f64) {
    if offset + 8 <= buf.len() {
        buf[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
}

fn write_fixed_ascii(buf: &mut [u8], offset: usize, len: usize, value: &str) {
    if offset + len > buf.len() {
        return;
    }
    let dst = &mut buf[offset..offset + len];
    dst.fill(0);
    let bytes = value.as_bytes();
    let n = bytes.len().min(len);
    dst[..n].copy_from_slice(&bytes[..n]);
}

fn patch_bispectrum_file_header(
    base_header: &[u8],
    number_of_sectors: i32,
    station1_name: &str,
    station2_meta: &StationMeta,
) -> Vec<u8> {
    let mut out = if base_header.len() >= FILE_HEADER_SIZE {
        base_header[..FILE_HEADER_SIZE].to_vec()
    } else {
        let mut tmp = vec![0u8; FILE_HEADER_SIZE];
        let n = base_header.len().min(FILE_HEADER_SIZE);
        tmp[..n].copy_from_slice(&base_header[..n]);
        tmp
    };

    write_i32_le(&mut out, NUM_SECTOR_OFFSET, number_of_sectors);
    write_fixed_ascii(&mut out, ST1_NAME_OFFSET, 8, station1_name);
    write_fixed_ascii(&mut out, ST2_NAME_OFFSET, 8, &station2_meta.name);
    write_fixed_ascii(&mut out, ST2_CODE_OFFSET, 1, &station2_meta.code);
    write_f64_le(&mut out, ST2_POS_X_OFFSET, station2_meta.position[0]);
    write_f64_le(&mut out, ST2_POS_Y_OFFSET, station2_meta.position[1]);
    write_f64_le(&mut out, ST2_POS_Z_OFFSET, station2_meta.position[2]);
    write_f64_le(&mut out, ST2_CLOCK_DELAY_OFFSET, station2_meta.clock[0]);
    write_f64_le(&mut out, ST2_CLOCK_RATE_OFFSET, station2_meta.clock[1]);
    write_f64_le(&mut out, ST2_CLOCK_ACEL_OFFSET, station2_meta.clock[2]);
    write_f64_le(&mut out, ST2_CLOCK_JERK_OFFSET, station2_meta.clock[3]);
    write_f64_le(&mut out, ST2_CLOCK_SNAP_OFFSET, station2_meta.clock[4]);
    out
}

fn load_file_header_and_parsed(path: &Path) -> Result<(Vec<u8>, CorHeader), Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut header_raw = vec![0u8; FILE_HEADER_SIZE];
    file.read_exact(&mut header_raw)?;
    let mut cursor = Cursor::new(header_raw.as_slice());
    let parsed = parse_header(&mut cursor)?;
    Ok((header_raw, parsed))
}

fn mean_complex(values: &[C32]) -> C32 {
    if values.is_empty() {
        return C32::new(0.0, 0.0);
    }
    let sum = values
        .iter()
        .copied()
        .fold(C32::new(0.0, 0.0), |acc, z| acc + z);
    sum / (values.len() as f32)
}

fn complex_cuberoot(z: C32) -> C32 {
    if z.re == 0.0 && z.im == 0.0 {
        return C32::new(0.0, 0.0);
    }
    let r = z.norm().powf(1.0 / 3.0);
    let th = z.arg() / 3.0;
    C32::from_polar(r, th)
}

fn apply_phase_solution(
    complex_vec: &[C32],
    fft_half: usize,
    header: &CorHeader,
    delay_samples: f32,
    rate_hz: f32,
    acel_hz: f32,
    effective_integ_time: f32,
    start_time_offset_sec: f32,
) -> Vec<C32> {
    if fft_half == 0
        || complex_vec.is_empty()
        || complex_vec.len() % fft_half != 0
        || (delay_samples == 0.0 && rate_hz == 0.0 && acel_hz == 0.0)
    {
        return complex_vec.to_vec();
    }

    let input_2d: Vec<Vec<Complex<f64>>> = complex_vec
        .chunks(fft_half)
        .map(|row| {
            row.iter()
                .map(|c| Complex::new(c.re as f64, c.im as f64))
                .collect()
        })
        .collect();

    let corrected_2d = fft::apply_phase_correction(
        &input_2d,
        rate_hz,
        delay_samples,
        acel_hz,
        effective_integ_time,
        header.sampling_speed as u32,
        header.fft_point as u32,
        start_time_offset_sec,
    );

    corrected_2d
        .into_iter()
        .flatten()
        .map(|z| C32::new(z.re as f32, z.im as f32))
        .collect()
}

fn collect_baseline_visibility(
    path: &Path,
    args: &Args,
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
) -> Result<LoadedBaseline, Box<dyn Error>> {
    let (file_header_raw, header) = load_file_header_and_parsed(path)?;
    let eff_time = peek_effective_integ_time(path, &header)?;
    let (window_length_pp, _) = compute_window_params(&header, args, eff_time);
    let selected_loops = selected_loop_indices(&header, args, eff_time);

    let bytes = fs::read(path)?;
    let mut cursor = Cursor::new(bytes.as_slice());
    let fft_half = (header.fft_point / 2).max(0) as usize;
    if fft_half == 0 {
        return Err(format!("Invalid FFT point in {}", path.display()).into());
    }

    let bw = header.sampling_speed as f32 / 2.0 / 1_000_000.0;
    let rbw = bw / header.fft_point as f32 * 2.0;
    let rfi_ranges = parse_rfi_ranges(&args.rfi, rbw)?;

    let mut bandpass_data = if let Some(bp_path) = &args.bandpass {
        Some(read_bandpass_file(bp_path)?)
    } else {
        None
    };
    if let Some(bp) = &bandpass_data {
        if bp.len() != fft_half {
            eprintln!(
                "#WARN: bandpass channel count ({}) != FFT/2 ({}). --closure-phaseではbandpassを無効化します。",
                bp.len(),
                fft_half
            );
            bandpass_data = None;
        }
    }

    let search_mode = args.primary_search_mode();
    let has_manual_correction =
        args.delay_correct != 0.0 || args.rate_correct != 0.0 || args.acel_correct != 0.0;

    let mut file_start_time: Option<DateTime<Utc>> = None;
    let mut prev_deep_solution: Option<(f32, f32)> = None;

    let mut loops = Vec::new();
    for (i, loop_idx) in selected_loops.iter().enumerate() {
        let requested_length = if args.cumulate != 0 {
            (loop_idx + 1) * window_length_pp
        } else {
            window_length_pp
        };
        let read_loop_index = if args.cumulate != 0 { 0 } else { *loop_idx };
        let is_cumulate = args.cumulate != 0;

        let (complex_vec, current_obs_time, effective_integ_time) = match read_visibility_data(
            &mut cursor,
            &header,
            requested_length,
            args.skip,
            read_loop_index,
            is_cumulate,
            pp_flag_ranges,
        ) {
            Ok(v) => v,
            Err(_) => break,
        };

        if file_start_time.is_none() {
            file_start_time = Some(current_obs_time);
        }

        let is_time_flagged = time_flag_ranges
            .iter()
            .any(|(start, end)| current_obs_time >= *start && current_obs_time < *end);
        if is_time_flagged {
            continue;
        }

        let actual_length = (complex_vec.len() / fft_half) as i32;
        if actual_length <= 0 {
            continue;
        }

        let (solve_delay, solve_rate, solve_acel) = match search_mode {
            Some("deep") => {
                let start_ref = file_start_time.unwrap_or(current_obs_time);
                let mut deep_result = search::run_deep_search(
                    &complex_vec,
                    &header,
                    actual_length,
                    actual_length,
                    effective_integ_time,
                    &current_obs_time,
                    &start_ref,
                    &rfi_ranges,
                    &bandpass_data,
                    args,
                    header.number_of_sector,
                    args.cpu,
                    prev_deep_solution,
                )?;
                deep_result.analysis_results.residual_delay -= args.delay_correct;
                deep_result.analysis_results.residual_rate -= args.rate_correct;
                deep_result.analysis_results.corrected_delay =
                    args.delay_correct + deep_result.analysis_results.residual_delay;
                deep_result.analysis_results.corrected_rate =
                    args.rate_correct + deep_result.analysis_results.residual_rate;
                let d = deep_result.analysis_results.corrected_delay;
                let r = deep_result.analysis_results.corrected_rate;
                prev_deep_solution = Some((d, r));
                (d, r, args.acel_correct)
            }
            Some("peak") => {
                let start_ref = file_start_time.unwrap_or(current_obs_time);
                let mut total_delay = args.delay_correct;
                let mut total_rate = args.rate_correct;
                for _ in 0..args.iter {
                    let (iter_results, _, _) = run_analysis_pipeline(
                        &complex_vec,
                        &header,
                        args,
                        Some("peak"),
                        total_delay,
                        total_rate,
                        args.acel_correct,
                        actual_length,
                        actual_length,
                        effective_integ_time,
                        &current_obs_time,
                        &start_ref,
                        &rfi_ranges,
                        &bandpass_data,
                        header.fft_point,
                    )?;
                    total_delay += iter_results.delay_offset;
                    total_rate += iter_results.rate_offset;
                }
                (total_delay, total_rate, args.acel_correct)
            }
            _ => (args.delay_correct, args.rate_correct, args.acel_correct),
        };

        if search_mode.is_some() || has_manual_correction {
            println!(
                "#CLOSURE-SOLVE {} loop {} (idx {}, len_pp={}): delay={:+.6} samp, rate={:+.6e} Hz, acel={:+.6e}",
                path.file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| path.display().to_string()),
                read_loop_index,
                i,
                actual_length,
                solve_delay,
                solve_rate,
                solve_acel
            );
        }

        let corrected_complex_vec = if search_mode.is_some() || has_manual_correction {
            let start_ref = file_start_time.unwrap_or(current_obs_time);
            let start_time_offset_sec = if search_mode.is_some() {
                0.0
            } else {
                current_obs_time
                    .signed_duration_since(start_ref)
                    .num_seconds() as f32
            };

            apply_phase_solution(
                &complex_vec,
                fft_half,
                &header,
                solve_delay,
                solve_rate,
                solve_acel,
                effective_integ_time,
                start_time_offset_sec,
            )
        } else {
            complex_vec.clone()
        };
        if corrected_complex_vec.is_empty() || corrected_complex_vec.len() % fft_half != 0 {
            continue;
        }
        let row_count = corrected_complex_vec.len() / fft_half;
        let integrated_spectrum = if row_count == 1 {
            corrected_complex_vec
        } else {
            let mut acc = vec![C32::new(0.0, 0.0); fft_half];
            for row in corrected_complex_vec.chunks(fft_half) {
                for (ch, z) in row.iter().enumerate() {
                    acc[ch] += *z;
                }
            }
            let scale = 1.0 / row_count as f32;
            for z in &mut acc {
                *z *= scale;
            }
            acc
        };

        let sector_headers = read_sector_header(
            &mut cursor,
            &header,
            requested_length,
            args.skip,
            read_loop_index,
            is_cumulate,
        )?;
        if sector_headers.is_empty() {
            continue;
        }

        let integrated_time_sec = actual_length as f32 * effective_integ_time;
        let mut first_sector_header = sector_headers[0].clone();
        if first_sector_header.len() >= SECTOR_HEADER_SIZE {
            write_f32_le(
                &mut first_sector_header,
                EFFECTIVE_INTEG_TIME_OFFSET,
                integrated_time_sec,
            );
        }

        loops.push(LoopVisibility {
            timestamp: current_obs_time,
            spectrum: integrated_spectrum,
            sector_header: first_sector_header,
            integrated_time_sec,
        });
    }

    let info = BaselineInfo {
        path: path.to_path_buf(),
        header: header.clone(),
        estimated_samples: selected_loops.len(),
    };

    Ok(LoadedBaseline {
        info,
        file_header_raw,
        loops,
        baseline_label: format!("{}-{}", header.station1_name.trim(), header.station2_name.trim()),
    })
}

fn compute_epoch_match_tolerance_sec(
    b1: &[LoopVisibility],
    b2: &[LoopVisibility],
    b3: &[LoopVisibility],
) -> f64 {
    let mut vals: Vec<f64> = b1
        .iter()
        .chain(b2.iter())
        .chain(b3.iter())
        .map(|lp| lp.integrated_time_sec as f64)
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if vals.is_empty() {
        return 1.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = vals[vals.len() / 2];
    (0.5 * med).max(1.0)
}

fn find_common_epoch_indices(
    b1: &[LoopVisibility],
    b2: &[LoopVisibility],
    b3: &[LoopVisibility],
    tolerance_sec: f64,
) -> Vec<[usize; 3]> {
    if b1.is_empty() || b2.is_empty() || b3.is_empty() {
        return Vec::new();
    }

    let tol_ms = (tolerance_sec * 1000.0).max(0.0).round() as i64;
    let mut i = 0usize;
    let mut j = 0usize;
    let mut k = 0usize;
    let mut result = Vec::new();

    while i < b1.len() && j < b2.len() && k < b3.len() {
        let t1 = b1[i].timestamp.timestamp_millis();
        let t2 = b2[j].timestamp.timestamp_millis();
        let t3 = b3[k].timestamp.timestamp_millis();

        let t_min = t1.min(t2.min(t3));
        let t_max = t1.max(t2.max(t3));
        if t_max - t_min <= tol_ms {
            result.push([i, j, k]);
            i += 1;
            j += 1;
            k += 1;
            continue;
        }

        if t1 == t_min {
            i += 1;
        }
        if t2 == t_min {
            j += 1;
        }
        if t3 == t_min {
            k += 1;
        }
    }

    result
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 0 {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    } else {
        v[n / 2]
    }
}

fn stddev(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let var = values
        .iter()
        .map(|x| {
            let d = *x - mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    var.sqrt()
}

fn build_closure_products(
    loops1: &[LoopVisibility],
    loops2: &[LoopVisibility],
    loops3: &[LoopVisibility],
    common_indices: &[[usize; 3]],
    sign2: f32,
    sign3: f32,
) -> Result<(Vec<ClosureSample>, Vec<Vec<C32>>, Vec<Vec<u8>>), Box<dyn Error>> {
    if common_indices.is_empty() {
        return Err("No overlapping epochs found across the three baselines.".into());
    }

    let mut rows = Vec::with_capacity(common_indices.len());
    let mut bispec_spectra = Vec::with_capacity(common_indices.len());
    let mut out_sector_headers = Vec::with_capacity(common_indices.len());

    for idxs in common_indices {
        let lp1 = &loops1[idxs[0]];
        let lp2 = &loops2[idxs[1]];
        let lp3 = &loops3[idxs[2]];

        let ch_count = lp1
            .spectrum
            .len()
            .min(lp2.spectrum.len())
            .min(lp3.spectrum.len());
        if ch_count == 0 {
            continue;
        }

        let mut bspec_spec = Vec::with_capacity(ch_count);
        let mut bspec_spec_cuberoot = Vec::with_capacity(ch_count);
        for ch in 0..ch_count {
            let z1 = lp1.spectrum[ch];
            let z2 = lp2.spectrum[ch];
            let z3 = lp3.spectrum[ch];
            let adj_z2 = if sign2 >= 0.0 { z2 } else { z2.conj() };
            let adj_z3 = if sign3 >= 0.0 { z3 } else { z3.conj() };
            let b = z1 * adj_z2 * adj_z3;
            bspec_spec.push(b);
            bspec_spec_cuberoot.push(complex_cuberoot(b));
        }

        let z1 = mean_complex(&lp1.spectrum[..ch_count]);
        let z2 = mean_complex(&lp2.spectrum[..ch_count]);
        let z3 = mean_complex(&lp3.spectrum[..ch_count]);
        let bispectrum = mean_complex(&bspec_spec);

        let phase1 = wrap_phase_deg(z1.arg().to_degrees());
        let phase2 = wrap_phase_deg(z2.arg().to_degrees());
        let phase3 = wrap_phase_deg(z3.arg().to_degrees());

        let closure_from_baselines_deg = wrap_phase_deg(phase1 + sign2 * phase2 + sign3 * phase3);
        let closure_phase_deg = wrap_phase_deg(bispectrum.arg().to_degrees());

        let bis_amp = bispectrum.norm();
        let bis_amp_cuberoot = if bis_amp > 0.0 { bis_amp.powf(1.0 / 3.0) } else { 0.0 };

        let mut sector_header = lp1.sector_header.clone();
        if sector_header.len() >= SECTOR_HEADER_SIZE {
            let min_integ = lp1
                .integrated_time_sec
                .min(lp2.integrated_time_sec)
                .min(lp3.integrated_time_sec);
            write_f32_le(&mut sector_header, EFFECTIVE_INTEG_TIME_OFFSET, min_integ);
        }

        rows.push(ClosureSample {
            timestamp: lp1.timestamp,
            raw_complex: [z1, z2, z3],
            baseline_phase_deg: [phase1, phase2, phase3],
            baseline_amp: [z1.norm(), z2.norm(), z3.norm()],
            bispectrum,
            bispectrum_amp: bis_amp,
            bispectrum_amp_cuberoot: bis_amp_cuberoot,
            closure_phase_deg,
            closure_from_baselines_deg,
            normalized_intensity: 1.0,
        });
        bispec_spectra.push(bspec_spec_cuberoot);
        out_sector_headers.push(sector_header);
    }

    if rows.is_empty() {
        return Err("Common epochs exist but no valid bispectrum spectra could be formed.".into());
    }

    let intensity_values: Vec<f64> = rows
        .iter()
        .map(|r| r.bispectrum_amp_cuberoot as f64)
        .collect();
    let med = median(&intensity_values).max(1e-20);
    for row in &mut rows {
        row.normalized_intensity = (row.bispectrum_amp_cuberoot as f64 / med) as f32;
    }

    Ok((rows, bispec_spectra, out_sector_headers))
}

fn compute_baseline_stats(values: &[C32]) -> BaselineStats {
    if values.is_empty() {
        return BaselineStats {
            mean: C32::new(0.0, 0.0),
            var_re: 0.0,
            var_im: 0.0,
            cov_re_im: 0.0,
            pseudovar: 0.0,
            snr: 0.0,
        };
    }

    let mean = mean_complex(values);
    let n = values.len() as f64;

    let mut var_re = 0.0;
    let mut var_im = 0.0;
    let mut cov_re_im = 0.0;
    let mut pseudo = 0.0;

    for z in values {
        let dr = z.re as f64 - mean.re as f64;
        let di = z.im as f64 - mean.im as f64;
        var_re += dr * dr;
        var_im += di * di;
        cov_re_im += dr * di;
        pseudo += dr * dr + di * di;
    }

    var_re /= n;
    var_im /= n;
    cov_re_im /= n;
    pseudo /= n;

    let snr = if pseudo > 0.0 {
        mean.norm() as f64 / pseudo.sqrt()
    } else {
        0.0
    };

    BaselineStats {
        mean,
        var_re,
        var_im,
        cov_re_im,
        pseudovar: pseudo,
        snr,
    }
}

fn compute_pair_stats(values_a: &[C32], values_b: &[C32]) -> PairStats {
    let n = values_a.len().min(values_b.len());
    if n == 0 {
        return PairStats {
            c_conj: C32::new(0.0, 0.0),
            c_plain: C32::new(0.0, 0.0),
            mu: C32::new(0.0, 0.0),
            nu: C32::new(0.0, 0.0),
        };
    }

    let mean_a = mean_complex(&values_a[..n]);
    let mean_b = mean_complex(&values_b[..n]);

    let mut c_conj = C32::new(0.0, 0.0);
    let mut c_plain = C32::new(0.0, 0.0);
    let mut qa = 0.0f64;
    let mut qb = 0.0f64;

    for i in 0..n {
        let da = values_a[i] - mean_a;
        let db = values_b[i] - mean_b;
        c_conj += da * db.conj();
        c_plain += da * db;
        qa += da.norm_sqr() as f64;
        qb += db.norm_sqr() as f64;
    }

    c_conj /= n as f32;
    c_plain /= n as f32;
    qa /= n as f64;
    qb /= n as f64;

    let den = (qa * qb).sqrt() as f32;
    let (mu, nu) = if den > 0.0 {
        (c_conj / den, c_plain / den)
    } else {
        (C32::new(0.0, 0.0), C32::new(0.0, 0.0))
    };

    PairStats {
        c_conj,
        c_plain,
        mu,
        nu,
    }
}

fn compute_bispectrum_stats(values: &[C32], closure_phase_deg: &[f32]) -> BispectrumStats {
    if values.is_empty() {
        return BispectrumStats {
            mean: C32::new(0.0, 0.0),
            pseudovar: 0.0,
            snr: 0.0,
            amp_mean: 0.0,
            amp_std: 0.0,
            phase_circ_std_deg: 0.0,
        };
    }

    let mean = mean_complex(values);
    let n = values.len() as f64;
    let pseudovar = values
        .iter()
        .map(|z| (*z - mean).norm_sqr() as f64)
        .sum::<f64>()
        / n;
    let snr = if pseudovar > 0.0 {
        mean.norm() as f64 / pseudovar.sqrt()
    } else {
        0.0
    };

    let amps: Vec<f64> = values.iter().map(|z| z.norm() as f64).collect();
    let amp_mean = amps.iter().sum::<f64>() / n;
    let amp_std = stddev(&amps, amp_mean);

    let unit_sum = closure_phase_deg.iter().fold(C32::new(0.0, 0.0), |acc, deg| {
        let rad = deg.to_radians();
        acc + C32::from_polar(1.0, rad)
    });
    let mut rbar = unit_sum.norm() as f64 / n;
    if !rbar.is_finite() {
        rbar = 0.0;
    }
    rbar = rbar.clamp(1e-12, 1.0);
    let phase_circ_std_deg = (-2.0 * rbar.ln()).sqrt().to_degrees();

    BispectrumStats {
        mean,
        pseudovar,
        snr,
        amp_mean,
        amp_std,
        phase_circ_std_deg,
    }
}

fn compute_closure_stats(rows: &[ClosureSample]) -> ClosureStats {
    let v1: Vec<C32> = rows.iter().map(|r| r.raw_complex[0]).collect();
    let v2: Vec<C32> = rows.iter().map(|r| r.raw_complex[1]).collect();
    let v3: Vec<C32> = rows.iter().map(|r| r.raw_complex[2]).collect();
    let b: Vec<C32> = rows.iter().map(|r| r.bispectrum).collect();
    let cp: Vec<f32> = rows.iter().map(|r| r.closure_phase_deg).collect();
    let i_bis: Vec<f64> = rows
        .iter()
        .map(|r| r.bispectrum_amp_cuberoot as f64)
        .collect();

    let i_mean = if i_bis.is_empty() {
        0.0
    } else {
        i_bis.iter().sum::<f64>() / i_bis.len() as f64
    };
    let i_std = stddev(&i_bis, i_mean);
    let i_frac = if i_mean.abs() > 0.0 {
        i_std / i_mean.abs()
    } else {
        0.0
    };

    ClosureStats {
        sample_count: rows.len(),
        baseline_stats: [
            compute_baseline_stats(&v1),
            compute_baseline_stats(&v2),
            compute_baseline_stats(&v3),
        ],
        pair_stats: [
            compute_pair_stats(&v1, &v2),
            compute_pair_stats(&v1, &v3),
            compute_pair_stats(&v2, &v3),
        ],
        bispectrum_stats: compute_bispectrum_stats(&b, &cp),
        intensity_mean: i_mean,
        intensity_std: i_std,
        intensity_frac_rms: i_frac,
    }
}

fn write_closure_tsv(
    path: &Path,
    labels: &[String],
    rows: &[ClosureSample],
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "epoch\tRe({})\tIm({})\t|{}|\tphase_{}[deg]\tRe({})\tIm({})\t|{}|\tphase_{}[deg]\tRe({})\tIm({})\t|{}|\tphase_{}[deg]\tRe(B)\tIm(B)\t|B|\t|B|^(1/3)\tclosure[arg(B)][deg]\tclosure[from phases][deg]\tI_norm",
        labels[0],
        labels[0],
        labels[0],
        labels[0],
        labels[1],
        labels[1],
        labels[1],
        labels[1],
        labels[2],
        labels[2],
        labels[2],
        labels[2],
    )?;

    for row in rows {
        writeln!(
            file,
            "{}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.3}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.3}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.3}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.3}\t{:.3}\t{:.9e}",
            row.timestamp.format("%Y/%j %H:%M:%S"),
            row.raw_complex[0].re,
            row.raw_complex[0].im,
            row.baseline_amp[0],
            row.baseline_phase_deg[0],
            row.raw_complex[1].re,
            row.raw_complex[1].im,
            row.baseline_amp[1],
            row.baseline_phase_deg[1],
            row.raw_complex[2].re,
            row.raw_complex[2].im,
            row.baseline_amp[2],
            row.baseline_phase_deg[2],
            row.bispectrum.re,
            row.bispectrum.im,
            row.bispectrum_amp,
            row.bispectrum_amp_cuberoot,
            row.closure_phase_deg,
            row.closure_from_baselines_deg,
            row.normalized_intensity,
        )?;
    }

    Ok(())
}

fn write_summary(
    path: &Path,
    labels: &[String],
    pair_labels: &[(&str, &str); 3],
    stats: &ClosureStats,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "# Closure/Bispectrum statistical summary (first and second moments, Kulkarni 1989 style)"
    )?;
    writeln!(file, "samples = {}", stats.sample_count)?;
    writeln!(file)?;

    writeln!(file, "[Fringe phasor moments]")?;
    for (idx, bs) in stats.baseline_stats.iter().enumerate() {
        writeln!(file, "baseline {} ({})", idx + 1, labels[idx])?;
        writeln!(file, "  mean = {:+.6e} {:+.6e}i", bs.mean.re, bs.mean.im)?;
        writeln!(file, "  V[Re] = {:.6e}", bs.var_re)?;
        writeln!(file, "  V[Im] = {:.6e}", bs.var_im)?;
        writeln!(file, "  C[Re,Im] = {:.6e}", bs.cov_re_im)?;
        writeln!(file, "  Q^2 = E[|q|^2] = {:.6e}", bs.pseudovar)?;
        writeln!(file, "  SNR ~= |R|/sqrt(Q^2) = {:.6e}", bs.snr)?;
    }
    writeln!(file)?;

    writeln!(file, "[Pair covariance of fringe noise q]")?;
    writeln!(
        file,
        "# C[qj,qk*], C[qj,qk], and normalized (mu, nu) for the three baseline pairs"
    )?;
    for (idx, ps) in stats.pair_stats.iter().enumerate() {
        writeln!(
            file,
            "pair {} ({} , {})",
            idx + 1,
            pair_labels[idx].0,
            pair_labels[idx].1
        )?;
        writeln!(
            file,
            "  C[qj,qk*] = {:+.6e} {:+.6e}i",
            ps.c_conj.re, ps.c_conj.im
        )?;
        writeln!(
            file,
            "  C[qj,qk ] = {:+.6e} {:+.6e}i",
            ps.c_plain.re, ps.c_plain.im
        )?;
        writeln!(file, "  mu = {:+.6e} {:+.6e}i", ps.mu.re, ps.mu.im)?;
        writeln!(file, "  nu = {:+.6e} {:+.6e}i", ps.nu.re, ps.nu.im)?;
    }
    writeln!(file)?;

    let bs = &stats.bispectrum_stats;
    writeln!(file, "[Bispectrum moments]")?;
    writeln!(file, "  mean(B) = {:+.6e} {:+.6e}i", bs.mean.re, bs.mean.im)?;
    writeln!(file, "  sigma_B^2 = E[|b-B|^2] = {:.6e}", bs.pseudovar)?;
    writeln!(file, "  SNR_B ~= |B|/sqrt(sigma_B^2) = {:.6e}", bs.snr)?;
    writeln!(file, "  mean(|B|) = {:.6e}", bs.amp_mean)?;
    writeln!(file, "  std(|B|) = {:.6e}", bs.amp_std)?;
    writeln!(
        file,
        "  circular std(closure phase) [deg] = {:.6e}",
        bs.phase_circ_std_deg
    )?;
    writeln!(file)?;

    writeln!(file, "[Intensity indicator from bispectrum]")?;
    writeln!(file, "  mean(|B|^(1/3)) = {:.6e}", stats.intensity_mean)?;
    writeln!(file, "  std(|B|^(1/3)) = {:.6e}", stats.intensity_std)?;
    writeln!(
        file,
        "  fractional RMS = std/mean = {:.6e}",
        stats.intensity_frac_rms
    )?;

    Ok(())
}

fn plot_closure_phase(
    path: &Path,
    labels: &[String],
    rows: &[ClosureSample],
) -> Result<(), Box<dyn Error>> {
    let drawing_area = BitMapBackend::new(path.to_str().unwrap(), (900, 620)).into_drawing_area();
    drawing_area.fill(&WHITE)?;
    let (legend_area, chart_area) = drawing_area.split_vertically(50);
    legend_area.fill(&WHITE)?;

    let first_time = rows.first().map(|row| row.timestamp).unwrap();
    let y_axis = PhaseAxis::new(-180.0f64..180.0f64, 30.0);

    let x_max = rows
        .last()
        .map(|sample| (sample.timestamp - first_time).num_milliseconds() as f64 / 1000.0)
        .unwrap_or(1.0)
        .max(1.0);
    let mut chart = ChartBuilder::on(&chart_area)
        .margin(10)
        .x_label_area_size(55)
        .y_label_area_size(65)
        .build_cartesian_2d(0.0..x_max, y_axis)?;

    chart
        .configure_mesh()
        .x_desc(format!(
            "Elapsed time [s] since {} UT",
            first_time.format("%Y/%j %H:%M:%S")
        ))
        .y_desc("Phase [deg]")
        .x_labels(13)
        .x_label_formatter(&|x| format!("{:>5.0}", x))
        .y_label_formatter(&|y| format!("{:>4.0}", *y))
        .y_labels(15)
        .label_style(("sans-serif", 22))
        .axis_desc_style(("sans-serif", 24))
        .light_line_style(&WHITE)
        .draw()?;

    let colors = [
        RGBColor(0, 102, 204),
        RGBColor(204, 102, 0),
        RGBColor(34, 139, 34),
        RGBColor(160, 32, 240),
    ];

    for (idx, label) in labels.iter().enumerate() {
        let color = colors[idx];
        chart
            .draw_series(rows.iter().map(|sample| {
                let x = (sample.timestamp - first_time).num_milliseconds() as f64 / 1000.0;
                Circle::new((x, sample.baseline_phase_deg[idx] as f64), 4, color.filled())
            }))?
            .label(label.clone())
            .legend(move |(x, y)| Circle::new((x + 10, y), 5, color.filled()));
    }

    let closure_color = colors[3];
    chart
        .draw_series(rows.iter().map(|sample| {
            let x = (sample.timestamp - first_time).num_milliseconds() as f64 / 1000.0;
            Circle::new((x, sample.closure_phase_deg as f64), 4, closure_color.filled())
        }))?
        .label("closure arg(B)")
        .legend(move |(x, y)| Circle::new((x + 10, y), 5, closure_color.filled()));

    let closure_from_phases_color = RGBColor(30, 30, 30);
    chart
        .draw_series(rows.iter().map(|sample| {
            let x = (sample.timestamp - first_time).num_milliseconds() as f64 / 1000.0;
            TriangleMarker::new(
                (x, sample.closure_from_baselines_deg as f64),
                6,
                closure_from_phases_color.filled(),
            )
        }))?
        .label("closure from phases")
        .legend(move |(x, y)| {
            TriangleMarker::new((x + 10, y), 6, closure_from_phases_color.filled())
        });

    let mut x_pos = 20;
    let y_pos = 15;
    let font = ("sans-serif", 20).into_font();
    let mut draw_legend_entry = |text: &str, color: RGBColor| -> Result<(), Box<dyn Error>> {
        legend_area.draw(&Circle::new((x_pos, y_pos), 6, color.filled()))?;
        x_pos += 10;
        legend_area.draw(&Text::new(text.to_string(), (x_pos + 10, y_pos), font.clone()))?;
        x_pos += text.chars().count() as i32 * 11;
        Ok(())
    };

    for (label, color) in labels.iter().zip(colors.iter()) {
        draw_legend_entry(label, *color)?;
    }
    draw_legend_entry("closure arg(B)", closure_color)?;
    draw_legend_entry("closure from phases", closure_from_phases_color)?;

    Ok(())
}

fn plot_bispectrum(path: &Path, rows: &[ClosureSample]) -> Result<(), Box<dyn Error>> {
    let drawing_area = BitMapBackend::new(path.to_str().unwrap(), (1000, 760)).into_drawing_area();
    drawing_area.fill(&WHITE)?;
    let split = drawing_area.split_vertically(410);
    let top = split.0;
    let bottom = split.1;

    let first_time = rows.first().map(|row| row.timestamp).unwrap();
    let x_max = rows
        .last()
        .map(|sample| (sample.timestamp - first_time).num_milliseconds() as f64 / 1000.0)
        .unwrap_or(1.0)
        .max(1.0);

    let v1: Vec<f64> = rows.iter().map(|r| r.baseline_amp[0] as f64).collect();
    let v2: Vec<f64> = rows.iter().map(|r| r.baseline_amp[1] as f64).collect();
    let v3: Vec<f64> = rows.iter().map(|r| r.baseline_amp[2] as f64).collect();
    let b13: Vec<f64> = rows
        .iter()
        .map(|r| r.bispectrum_amp_cuberoot as f64)
        .collect();
    let braw: Vec<f64> = rows.iter().map(|r| r.bispectrum_amp as f64).collect();

    let mut amp_max = 0.0f64;
    for val in v1
        .iter()
        .chain(v2.iter())
        .chain(v3.iter())
        .chain(b13.iter())
    {
        amp_max = amp_max.max(*val);
    }
    if amp_max <= 0.0 {
        amp_max = 1.0;
    } else {
        amp_max *= 1.1;
    }

    let mut chart_top = ChartBuilder::on(&top)
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(65)
        .build_cartesian_2d(0.0..x_max, 0.0..amp_max)?;

    chart_top
        .configure_mesh()
        .x_desc(format!(
            "Elapsed time [s] since {} UT",
            first_time.format("%Y/%j %H:%M:%S")
        ))
        .y_desc("|V1|, |V2|, |V3|, |B|^(1/3)")
        .x_labels(12)
        .x_label_formatter(&|x| format!("{:>6.0}", x))
        .y_label_formatter(&|y| format!("{:>9.2e}", *y))
        .label_style(("sans-serif", 20))
        .axis_desc_style(("sans-serif", 22))
        .light_line_style(&WHITE)
        .draw()?;

    let c1 = RGBColor(0, 102, 204);
    let c2 = RGBColor(204, 102, 0);
    let c3 = RGBColor(34, 139, 34);
    let c4 = RGBColor(160, 32, 240);

    chart_top
        .draw_series(rows.iter().zip(v1.iter()).map(|(sample, amp)| {
            let x = (sample.timestamp - first_time).num_milliseconds() as f64 / 1000.0;
            Circle::new((x, *amp), 3, c1.filled())
        }))?
        .label("|V1|")
        .legend(move |(x, y)| Circle::new((x + 10, y), 4, c1.filled()));

    chart_top
        .draw_series(rows.iter().zip(v2.iter()).map(|(sample, amp)| {
            let x = (sample.timestamp - first_time).num_milliseconds() as f64 / 1000.0;
            Circle::new((x, *amp), 3, c2.filled())
        }))?
        .label("|V2|")
        .legend(move |(x, y)| Circle::new((x + 10, y), 4, c2.filled()));

    chart_top
        .draw_series(rows.iter().zip(v3.iter()).map(|(sample, amp)| {
            let x = (sample.timestamp - first_time).num_milliseconds() as f64 / 1000.0;
            Circle::new((x, *amp), 3, c3.filled())
        }))?
        .label("|V3|")
        .legend(move |(x, y)| Circle::new((x + 10, y), 4, c3.filled()));

    chart_top
        .draw_series(rows.iter().zip(b13.iter()).map(|(sample, amp)| {
            let x = (sample.timestamp - first_time).num_milliseconds() as f64 / 1000.0;
            TriangleMarker::new((x, *amp), 6, c4.filled())
        }))?
        .label("|B|^(1/3)")
        .legend(move |(x, y)| TriangleMarker::new((x + 10, y), 6, c4.filled()));

    chart_top
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .label_font(("sans-serif", 18))
        .draw()?;

    let mut bmax = braw.iter().copied().fold(0.0, f64::max);
    if bmax <= 0.0 {
        bmax = 1.0;
    } else {
        bmax *= 1.1;
    }

    let mut chart_bottom = ChartBuilder::on(&bottom)
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(70)
        .build_cartesian_2d(0.0..x_max, 0.0..bmax)?;

    chart_bottom
        .configure_mesh()
        .x_desc(format!(
            "Elapsed time [s] since {} UT",
            first_time.format("%Y/%j %H:%M:%S")
        ))
        .y_desc("|B|")
        .x_labels(12)
        .x_label_formatter(&|x| format!("{:>6.0}", x))
        .y_label_formatter(&|y| format!("{:>9.2e}", *y))
        .label_style(("sans-serif", 20))
        .axis_desc_style(("sans-serif", 22))
        .light_line_style(&WHITE)
        .draw()?;

    chart_bottom
        .draw_series(rows.iter().zip(braw.iter()).map(|(sample, amp)| {
            let x = (sample.timestamp - first_time).num_milliseconds() as f64 / 1000.0;
            Circle::new((x, *amp), 3, c4.filled())
        }))?
        .label("|B|")
        .legend(move |(x, y)| Circle::new((x + 10, y), 4, c4.filled()));

    chart_bottom
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .label_font(("sans-serif", 18))
        .draw()?;

    Ok(())
}

fn extract_third_station_meta(base3: &CorHeader, refant: &str) -> StationMeta {
    if names_equal(&base3.station1_name, refant) {
        station_meta_from_header(base3, false)
    } else {
        station_meta_from_header(base3, true)
    }
}

pub fn run_closure_phase_analysis(
    args: &Args,
    cor_paths: &[PathBuf],
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
    refant: &str,
) -> Result<(), Box<dyn Error>> {
    if cor_paths.len() != 3 {
        return Err("closure-phase requires exactly three .cor files.".into());
    }
    if args.frequency {
        return Err("--closure-phase requires time-domain processing (omit --frequency).".into());
    }
    if !args.search.is_empty() && args.primary_search_mode().is_none() {
        eprintln!(
            "#WARN: --closure-phase で有効な --search は peak/deep のみです。指定モード {:?} は無視されます。",
            args.search
        );
    }

    let refant_name = refant.trim();
    if refant_name.is_empty() {
        return Err("refant must not be empty.".into());
    }

    let loaded1 = collect_baseline_visibility(&cor_paths[0], args, time_flag_ranges, pp_flag_ranges)?;
    let loaded2 = collect_baseline_visibility(&cor_paths[1], args, time_flag_ranges, pp_flag_ranges)?;
    let loaded3 = collect_baseline_visibility(&cor_paths[2], args, time_flag_ranges, pp_flag_ranges)?;

    let fft1 = loaded1.info.header.fft_point;
    let fft2 = loaded2.info.header.fft_point;
    let fft3 = loaded3.info.header.fft_point;
    if fft1 != fft2 || fft1 != fft3 {
        return Err(format!("FFT mismatch across baselines: {}, {}, {}", fft1, fft2, fft3).into());
    }

    let base1_st1 = normalize_name(&loaded1.info.header.station1_name);
    if !names_equal(&base1_st1, refant_name) {
        return Err(format!(
            "Baseline 1 station1 '{}' does not match refant '{}'.",
            loaded1.info.header.station1_name.trim(),
            refant_name
        )
        .into());
    }
    let mid_ant = normalize_name(&loaded1.info.header.station2_name);

    let b2_st1 = normalize_name(&loaded2.info.header.station1_name);
    let b2_st2 = normalize_name(&loaded2.info.header.station2_name);
    let (sign2, third_ant) = if names_equal(&b2_st1, &mid_ant) {
        (1.0f32, b2_st2.clone())
    } else if names_equal(&b2_st2, &mid_ant) {
        (-1.0f32, b2_st1.clone())
    } else {
        return Err(format!(
            "Baseline 2 must include antenna '{}' to close the triangle.",
            mid_ant
        )
        .into());
    };

    if names_equal(&third_ant, &mid_ant) || names_equal(&third_ant, refant_name) {
        return Err("Baseline configuration does not form a valid triangle.".into());
    }

    let b3_st1 = normalize_name(&loaded3.info.header.station1_name);
    let b3_st2 = normalize_name(&loaded3.info.header.station2_name);
    let orientation = if names_equal(&b3_st1, refant_name) && names_equal(&b3_st2, &third_ant) {
        1.0f32
    } else if names_equal(&b3_st2, refant_name) && names_equal(&b3_st1, &third_ant) {
        -1.0f32
    } else {
        return Err(format!(
            "Baseline 3 must connect '{}' and '{}'.",
            refant_name, third_ant
        )
        .into());
    };
    let sign3 = -orientation;

    println!("#CLOSURE INPUTS");
    for (idx, loaded) in [&loaded1, &loaded2, &loaded3].iter().enumerate() {
        println!(
            "#  [{}] {} | baseline {}-{} | FFT {} | PP {} | samples {} | extracted {}",
            idx + 1,
            loaded.info.path.display(),
            loaded.info.header.station1_name.trim(),
            loaded.info.header.station2_name.trim(),
            loaded.info.header.fft_point,
            loaded.info.header.number_of_sector,
            loaded.info.estimated_samples,
            loaded.loops.len(),
        );
    }
    println!(
        "#  reference antenna: {} | triangle: {} -> {} -> {}",
        refant_name, refant_name, mid_ant, third_ant
    );
    println!(
        "#  orientation: sign2={} | sign3={}",
        if sign2 >= 0.0 { "+1" } else { "-1" },
        if sign3 >= 0.0 { "+1" } else { "-1" }
    );

    let epoch_match_tolerance_sec =
        compute_epoch_match_tolerance_sec(&loaded1.loops, &loaded2.loops, &loaded3.loops);
    println!(
        "#  epoch matching tolerance: +/-{:.3} s",
        epoch_match_tolerance_sec
    );
    let common_indices = find_common_epoch_indices(
        &loaded1.loops,
        &loaded2.loops,
        &loaded3.loops,
        epoch_match_tolerance_sec,
    );
    let (rows, bispec_spectra, bispec_sector_headers) = build_closure_products(
        &loaded1.loops,
        &loaded2.loops,
        &loaded3.loops,
        &common_indices,
        sign2,
        sign3,
    )?;

    println!(
        "#Closure-phase samples: {} (common epochs across all baselines)",
        rows.len()
    );

    let labels = vec![
        loaded1.baseline_label.clone(),
        loaded2.baseline_label.clone(),
        loaded3.baseline_label.clone(),
    ];
    let pair_labels = [
        (labels[0].as_str(), labels[1].as_str()),
        (labels[0].as_str(), labels[2].as_str()),
        (labels[1].as_str(), labels[2].as_str()),
    ];
    let stats = compute_closure_stats(&rows);

    let parent_dir = cor_paths[0]
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let frinz_dir = parent_dir.join("frinZ");
    fs::create_dir_all(&frinz_dir)?;
    let closure_dir = frinz_dir.join("closure_phase");
    fs::create_dir_all(&closure_dir)?;

    let sanitized_ref = sanitize_token(refant_name);
    let sanitized_mid = sanitize_token(&mid_ant);
    let sanitized_third = sanitize_token(&third_ant);

    let suffix1 = {
        let stem = cor_paths[0]
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let prefix = format!("{}_{}_", sanitized_ref, sanitized_mid);
        stem.strip_prefix(&prefix)
            .map(|s| s.to_string())
            .unwrap_or(stem)
    };

    let expected_suffix = suffix1.clone();
    for (idx, path) in cor_paths.iter().enumerate().skip(1) {
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let prefix = if idx == 1 {
            format!("{}_{}_", sanitized_mid, sanitized_third)
        } else {
            format!("{}_{}_", sanitized_ref, sanitized_third)
        };
        if let Some(rest) = stem.strip_prefix(&prefix) {
            if rest != expected_suffix {
                eprintln!(
                    "#WARN: File suffix '{}' differs from '{}'; using arg1 suffix.",
                    rest, expected_suffix
                );
            }
        } else {
            eprintln!(
                "#WARN: File '{}' does not start with expected prefix '{}'.",
                path.display(),
                prefix
            );
        }
    }

    let suffix_token = if expected_suffix.is_empty() {
        "unknown".to_string()
    } else {
        sanitize_token(&expected_suffix)
    };

    let base_name = format!(
        "{}_{}_{}_{}",
        sanitized_ref, sanitized_mid, sanitized_third, suffix_token
    );

    let tsv_path = closure_dir.join(format!("{}_complex.tsv", base_name));
    let summary_path = closure_dir.join(format!("{}_summary.txt", base_name));
    let closure_png = closure_dir.join(format!("{}_closurephase.png", base_name));
    let bis_png = closure_dir.join(format!("{}_bispectrum.png", base_name));
    let bis_cor = parent_dir.join(format!("{}_bispectrum.cor", base_name));

    write_closure_tsv(&tsv_path, &labels, &rows)?;
    write_summary(&summary_path, &labels, &pair_labels, &stats)?;
    plot_closure_phase(&closure_png, &labels, &rows)?;
    plot_bispectrum(&bis_png, &rows)?;
    compress_png_with_mode(&closure_png, CompressQuality::Low);
    compress_png_with_mode(&bis_png, CompressQuality::Low);

    let third_station_meta = extract_third_station_meta(&loaded3.info.header, refant_name);
    let out_header = patch_bispectrum_file_header(
        &loaded1.file_header_raw,
        bispec_spectra.len() as i32,
        refant_name,
        &third_station_meta,
    );
    write_phase_corrected_spectrum_binary(
        &bis_cor,
        &out_header,
        &bispec_sector_headers,
        &bispec_spectra,
    )?;

    println!("#Saved TSV to {}", tsv_path.display());
    println!("#Saved summary to {}", summary_path.display());
    println!("#Saved closure plot to {}", closure_png.display());
    println!("#Saved bispectrum plot to {}", bis_png.display());
    println!("#Saved bispectrum^(1/3) .cor to {}", bis_cor.display());

    Ok(())
}
