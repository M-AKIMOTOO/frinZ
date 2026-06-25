// FRMAP module (frmap = fringe rate map).
// Builds fringe-rate map constraints from baseline/fringe-rate measurements
// and estimates sky-position intersections.
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::{Cursor, Write};
use std::path::Path;

use chrono::{DateTime, SecondsFormat, Utc};
use ndarray::{Array, Array2, ArrayView1, Axis};
use num_complex::Complex;

use crate::args::Args;
use crate::bandpass::{apply_bandpass_correction, read_bandpass_file};
use crate::fft::{apply_phase_correction_in_place, process_fft, process_ifft_with_delay_padding};
use crate::header::{parse_header, CorHeader};
use crate::input_support::read_input_bytes;
use crate::npy_output::{
    npz_sidecar_path, write_complex_1d, write_complex_2d, write_real_1d, NpyMeta,
};
use crate::plot::{plot_cross_section, plot_sky_map, plot_uv_coverage};
use crate::read::read_visibility_data;
use crate::rfi::parse_rfi_ranges;
use crate::utils::{rate_cal, uvw_cal};
use plotters::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
struct FringeLineMeasurement {
    index: usize,
    freq_channel: usize,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    u: f64,
    v: f64,
    du_dt: f64,
    dv_dt: f64,
    rate_hz: f64,
    rate_err_hz: f64,
    delay_s: f64,
    delay_err_s: f64,
    amplitude: f64,
    snr: f64,
}

impl FringeLineMeasurement {
    fn rate_line_coeffs(&self, lambda: f64) -> (f64, f64, f64) {
        // a * l + b * m = c
        let a = self.du_dt;
        let b = self.dv_dt;
        let c = self.rate_hz * lambda;
        (a, b, c)
    }

    fn weight(&self) -> f64 {
        self.snr.max(1.0)
    }
}

#[derive(Debug, Clone)]
struct FringeIntersection {
    l: f64,
    m: f64,
    weight: f64,
    line_i: usize,
    line_j: usize,
}

#[derive(Debug, Clone)]
struct CentroidStats {
    mean_l: f64,
    mean_m: f64,
    sigma_l: f64,
    sigma_m: f64,
}

fn compute_median(data: &mut [f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.sort_by(|a, b| {
        if !a.is_finite() && !b.is_finite() {
            Ordering::Equal
        } else if !a.is_finite() {
            Ordering::Greater
        } else if !b.is_finite() {
            Ordering::Less
        } else {
            a.partial_cmp(b).unwrap_or(Ordering::Equal)
        }
    });
    let mid = data.len() / 2;
    if data.len() % 2 == 0 {
        (data[mid - 1] + data[mid]) * 0.5
    } else {
        data[mid]
    }
}

fn compute_mad(data: &[f64], median: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut deviations: Vec<f64> = data
        .iter()
        .filter(|val| val.is_finite())
        .map(|val| (val - median).abs())
        .collect();
    compute_median(&mut deviations)
}

fn clip_line_to_square(a: f64, b: f64, c: f64, limit_rad: f64) -> Option<((f64, f64), (f64, f64))> {
    const EPS: f64 = 1.0e-12;
    let mut points: Vec<(f64, f64)> = Vec::new();

    for &m in &[-limit_rad, limit_rad] {
        if a.abs() > EPS {
            let l = (c - b * m) / a;
            if l.is_finite() && l.abs() <= limit_rad + 1.0e-9 {
                points.push((l, m));
            }
        }
    }
    for &l in &[-limit_rad, limit_rad] {
        if b.abs() > EPS {
            let m = (c - a * l) / b;
            if m.is_finite() && m.abs() <= limit_rad + 1.0e-9 {
                points.push((l, m));
            }
        }
    }

    // Deduplicate near-identical points
    let mut unique: Vec<(f64, f64)> = Vec::new();
    for (l, m) in points {
        if unique
            .iter()
            .any(|(ul, um)| (ul - l).abs() < 1.0e-9 && (um - m).abs() < 1.0e-9)
        {
            continue;
        }
        unique.push((l, m));
    }

    if unique.len() < 2 {
        return None;
    }

    // Choose the two points with the largest separation for better visuals.
    let mut best_pair = (unique[0], unique[1]);
    let mut max_dist_sq = 0.0;
    for i in 0..unique.len() {
        for j in (i + 1)..unique.len() {
            let dx = unique[i].0 - unique[j].0;
            let dy = unique[i].1 - unique[j].1;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq > max_dist_sq {
                max_dist_sq = dist_sq;
                best_pair = (unique[i], unique[j]);
            }
        }
    }

    Some(best_pair)
}

fn compute_weighted_stats(intersections: &[FringeIntersection]) -> Option<CentroidStats> {
    if intersections.is_empty() {
        return None;
    }
    let mut sum_w = 0.0;
    let mut sum_l = 0.0;
    let mut sum_m = 0.0;
    for inter in intersections {
        if !inter.weight.is_finite() || inter.weight <= 0.0 {
            continue;
        }
        sum_w += inter.weight;
        sum_l += inter.weight * inter.l;
        sum_m += inter.weight * inter.m;
    }
    if sum_w <= 0.0 {
        return None;
    }
    let mean_l = sum_l / sum_w;
    let mean_m = sum_m / sum_w;

    let mut var_l = 0.0;
    let mut var_m = 0.0;
    for inter in intersections {
        if !inter.weight.is_finite() || inter.weight <= 0.0 {
            continue;
        }
        let dl = inter.l - mean_l;
        let dm = inter.m - mean_m;
        var_l += inter.weight * dl * dl;
        var_m += inter.weight * dm * dm;
    }

    let sigma_l = (var_l / sum_w).sqrt();
    let sigma_m = (var_m / sum_w).sqrt();

    Some(CentroidStats {
        mean_l,
        mean_m,
        sigma_l,
        sigma_m,
    })
}

fn quadratic_peak_offset(left: f32, center: f32, right: f32) -> f64 {
    let denom = left as f64 - 2.0 * center as f64 + right as f64;
    if !denom.is_finite() || denom >= -f64::EPSILON {
        return 0.0;
    }
    (0.5 * (left as f64 - right as f64) / denom).clamp(-1.0, 1.0)
}

fn find_map_peak(map: &Array2<f32>) -> (f32, (usize, usize)) {
    let mut max_value = f32::NEG_INFINITY;
    let mut max_index = (0, 0);
    for (index, &value) in map.indexed_iter() {
        if value.is_finite() && value > max_value {
            max_value = value;
            max_index = index;
        }
    }
    (max_value, max_index)
}

fn refine_map_peak(map: &Array2<f32>, row: usize, col: usize) -> (f64, f64) {
    let (height, width) = map.dim();
    let dx = if col > 0 && col + 1 < width {
        quadratic_peak_offset(map[[row, col - 1]], map[[row, col]], map[[row, col + 1]])
    } else {
        0.0
    };
    let dy = if row > 0 && row + 1 < height {
        quadratic_peak_offset(map[[row - 1, col]], map[[row, col]], map[[row + 1, col]])
    } else {
        0.0
    };
    (col as f64 + dx, row as f64 + dy)
}

#[allow(unused_variables)]
#[allow(unused_mut)]
pub fn run_fringe_rate_map_analysis(
    args: &Args,
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
) -> Result<(), Box<dyn Error>> {
    let frmap_tokens = args.fringe_rate_map.clone().unwrap_or_default();
    let config = FrMapConfig::from_tokens(&frmap_tokens)?;

    if matches!(config.mode, FrMapMode::Maser) {
        return run_frmap_maser(args, time_flag_ranges, pp_flag_ranges, &config);
    }

    println!("Starting fringe-rate map analysis...");

    let input_path = args.input.as_ref().unwrap();

    // --- File and Path Setup ---
    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let frinz_dir = parent_dir.join("frinZ").join("frmap");
    fs::create_dir_all(&frinz_dir)?;
    let file_stem = input_path.file_stem().unwrap().to_str().unwrap();

    // --- Read .cor File ---
    let buffer = read_input_bytes(input_path)?;
    let mut cursor = Cursor::new(buffer.as_slice());

    // --- Parse Header ---
    let header = parse_header(&mut cursor)?;

    // --- Pre-computation for UV coverage and B_max ---
    println!("Pre-calculating UV coverage to determine optimal cell size...");
    let mut max_b = 0.0f64;
    let mut min_b = f64::INFINITY;
    let mut all_uv_data: Vec<(f32, f32)> = Vec::new(); // New vector for all UV data
    let mut temp_cursor = cursor.clone();
    temp_cursor.set_position(256);
    let temp_pp = header.number_of_sector;

    let mut obs_start_time: Option<DateTime<Utc>> = None;
    let mut effective_integ_time: Option<f32> = None;

    for l1 in 0..temp_pp {
        let (_, current_obs_time, current_effective_integ_time) = match read_visibility_data(
            &mut temp_cursor,
            &header,
            1,
            l1,
            0,
            false,
            pp_flag_ranges,
        ) {
            Ok(data) => data,
            Err(_) => break,
        };

        if l1 == 0 {
            obs_start_time = Some(current_obs_time);
            effective_integ_time = Some(current_effective_integ_time);
        }
        let (u, v, _, _, _) = uvw_cal(
            header.station1_position,
            header.station2_position,
            current_obs_time,
            header.source_position_ra,
            header.source_position_dec,
            true,
        );
        let b = (u.powi(2) + v.powi(2)).sqrt();
        if b > max_b {
            max_b = b;
        }
        if b.is_finite() && b > 0.0 && b < min_b {
            min_b = b;
        }
        all_uv_data.push((u as f32, v as f32)); // Collect all UV data
    }
    if !min_b.is_finite() {
        min_b = max_b;
    }
    println!("Projected baseline range: {:.2} .. {:.2} m", min_b, max_b);

    // --- Image Parameters ---
    let lambda = 299792458.0 / header.observing_frequency;
    let desired_map_range_arcsec = match config.range_spec {
        RangeSpec::Auto => {
            let angular_scale_arcsec = if min_b > 0.0 {
                (lambda / min_b).to_degrees() * 3600.0
            } else {
                return Err("Unable to determine a positive projected baseline".into());
            };
            let half_range_arcsec = 3.0 * angular_scale_arcsec;
            println!(
                "Auto display range: +/-{:.3} mas (3 lambda/B_proj,min; B_proj,min={:.2} m)",
                half_range_arcsec * 1_000.0,
                min_b
            );
            2.0 * half_range_arcsec
        }
        RangeSpec::Value(v) => v,
    };
    let rad_to_arcsec: f64 = 180.0 / PI * 3600.0;
    let arcsec_to_rad = PI / (180.0 * 3600.0);
    let desired_map_range_rad = desired_map_range_arcsec * arcsec_to_rad;
    let image_size = config.grid_size;
    let cell_size_rad = desired_map_range_rad / image_size as f64; // Calculate cell size based on desired image size

    println!(
        "Angular resolution (lambda/B_max): {:.3} mas",
        (lambda / max_b).to_degrees() * 3600.0 * 1_000.0
    );
    println!(
        "Calculated cell size: {:.4e} rad ({:.4} mas)",
        cell_size_rad,
        cell_size_rad.to_degrees() * 3600e3
    );
    println!(
        "Setting map range to ~{:.3} mas with image size {}x{}",
        desired_map_range_arcsec * 1_000.0,
        image_size,
        image_size
    );

    // Complex accumulation preserves phase across segments.  The magnitude is
    // evaluated only after every segment has been mapped to the common sky grid.
    let mut total_complex_map = ndarray::Array2::<Complex<f32>>::zeros((image_size, image_size));
    let mut total_complex_beam = ndarray::Array2::<Complex<f32>>::zeros((image_size, image_size));
    let mut uv_data: Vec<(f32, f32)> = Vec::new();

    let _obs_start_time = obs_start_time.expect("Failed to get observation start time");
    let effective_integ_time =
        effective_integ_time.expect("Failed to get effective integration time");

    // Match the sub-bin fringe-rate resolution used by --search peak. Delay
    // padding performs the corresponding interpolation on the frequency axis.
    let rate_padding = args.rate_padding.max(8);
    let delay_padding = config.delay_padding;
    println!(
        "High-accuracy delay/rate grid: rate padding {}x, delay padding {}x",
        rate_padding, delay_padding
    );

    let bandwidth_mhz = header.sampling_speed as f32 / 2.0 / 1_000_000.0;
    let rbw_mhz = bandwidth_mhz / (header.fft_point as f32 / 2.0);
    let rfi_ranges = parse_rfi_ranges(&args.rfi, rbw_mhz)?;
    let bandpass_data = if let Some(bp_path) = &args.bandpass {
        Some(read_bandpass_file(bp_path)?)
    } else {
        None
    };

    // --- Loop Setup ---
    cursor.set_position(0);
    let (_, _obs_start_time, effective_integ_time) =
        read_visibility_data(&mut cursor, &header, 1, 0, 0, false, pp_flag_ranges)?;
    cursor.set_position(256);

    let pp = header.number_of_sector;
    // A short transform keeps delay and fringe rate approximately constant.
    // Segment spectra are subsequently added coherently on the common sky grid.
    // Thirty seconds is also small enough for the high-resolution padded grid
    // to remain memory bounded.
    let length_in_sectors = if args.length == 0 {
        ((30.0 / effective_integ_time.max(1.0e-6)).round() as i32).clamp(1, pp.max(1))
    } else {
        args.length.max(1).min(pp)
    };
    println!(
        "Processing in segments of {} sectors (approx. {} seconds)",
        length_in_sectors,
        length_in_sectors as f32 * effective_integ_time
    );

    let total_segments_available = (pp - args.skip) / length_in_sectors;
    let loop_count = if args.loop_ == 1 {
        // Default loop is 1, so if user does not specify, process all.
        total_segments_available
    } else {
        total_segments_available.min(args.loop_)
    };

    // Build the response of a unit source at delay=rate=0 using exactly the
    // same finite time/frequency sampling and RFI mask as the data.  A single
    // delta bin would make the reported beam artificially padding-dependent.
    let fft_point_half = (header.fft_point / 2) as usize;
    let unit_visibility =
        vec![Complex::new(1.0_f32, 0.0_f32); length_in_sectors as usize * fft_point_half];
    let (beam_freq_rate_array, beam_padding_length) = process_fft(
        &unit_visibility,
        length_in_sectors,
        header.fft_point,
        header.sampling_speed,
        &rfi_ranges,
        rate_padding,
    );
    let beam_delay_rate_array = process_ifft_with_delay_padding(
        &beam_freq_rate_array,
        header.fft_point,
        beam_padding_length,
        delay_padding,
    );

    // --- Main Processing Loop ---
    let mut processed_segments = 0usize;
    for l1 in 0..loop_count {
        let (mut complex_vec, current_obs_time, effective_integ_time) = match read_visibility_data(
            &mut cursor,
            &header,
            length_in_sectors,
            args.skip,
            l1,
            false,
            pp_flag_ranges,
        ) {
            Ok(data) => data,
            Err(_) => break,
        };

        if complex_vec.is_empty() {
            break;
        }

        let is_flagged = time_flag_ranges
            .iter()
            .any(|(start, end)| current_obs_time >= *start && current_obs_time < *end);
        if is_flagged {
            continue;
        }

        // --- Apply Phase Correction ---
        if args.delay_correct != 0.0 || args.rate_correct != 0.0 || args.acel_correct != 0.0 {
            println!(
                "Applying phase corrections: delay={}, rate={}, acel={}",
                args.delay_correct, args.rate_correct, args.acel_correct
            );

            // Keep the phase origin at the first sample.  The complex sky-map
            // rephasing below uses the same epoch when combining segments.
            let start_time_offset_sec = 0.0;
            apply_phase_correction_in_place(
                &mut complex_vec,
                (header.fft_point / 2) as usize,
                args.rate_correct,
                args.delay_correct,
                args.acel_correct,
                args.jerk_correct,
                args.snap_correct,
                effective_integ_time,
                header.sampling_speed as u32,
                header.fft_point as u32,
                start_time_offset_sec,
            );
        }

        let (mut freq_rate_array, padding_length) = process_fft(
            &complex_vec,
            length_in_sectors,
            header.fft_point,
            header.sampling_speed,
            &rfi_ranges,
            rate_padding,
        );
        if let Some(bp_data) = &bandpass_data {
            apply_bandpass_correction(&mut freq_rate_array, bp_data);
        }
        let delay_rate_array = process_ifft_with_delay_padding(
            &freq_rate_array,
            header.fft_point,
            padding_length,
            delay_padding,
        );

        let rate_range_vec = rate_cal(padding_length as f32, effective_integ_time);
        let rate_range = Array::from_vec(rate_range_vec);
        let delay_range = Array::linspace(
            -(header.fft_point as f32 / 2.0) + 1.0 / delay_padding as f32,
            header.fft_point as f32 / 2.0,
            header.fft_point as usize * delay_padding,
        );

        let midpoint_offset_sec =
            0.5 * length_in_sectors.saturating_sub(1) as f64 * effective_integ_time as f64;
        let segment_center_time = current_obs_time
            + chrono::Duration::microseconds((midpoint_offset_sec * 1_000_000.0) as i64);
        let (u, v, _w, du_dt, dv_dt) = uvw_cal(
            header.station1_position,
            header.station2_position,
            segment_center_time,
            header.source_position_ra,
            header.source_position_dec,
            true,
        );
        if l1 == 0 {
            println!(
                "DEBUG: seg 0: u={}, v={}, du_dt={}, dv_dt={}",
                u, v, du_dt, dv_dt
            );
        }
        uv_data.push((u as f32, v as f32));

        // The delay/rate FFT is referenced to the first sample of the
        // segment. Remove that samples geometric phase before combining
        // segments on the common sky grid.
        let (phase_u, phase_v, _phase_w, _, _) = uvw_cal(
            header.station1_position,
            header.station2_position,
            current_obs_time,
            header.source_position_ra,
            header.source_position_dec,
            true,
        );
        let segment_map = create_complex_map(
            &delay_rate_array,
            u,
            v,
            du_dt,
            dv_dt,
            phase_u,
            phase_v,
            &header,
            &rate_range.view(),
            &delay_range.view(),
            image_size,
            cell_size_rad,
        );
        total_complex_map += &segment_map;

        let segment_beam_map = create_complex_map(
            &beam_delay_rate_array,
            u,
            v,
            du_dt,
            dv_dt,
            phase_u,
            phase_v,
            &header,
            &rate_range.view(),
            &delay_range.view(),
            image_size,
            cell_size_rad,
        );
        total_complex_beam += &segment_beam_map;
        processed_segments += 1;

        if (l1 + 1) % 10 == 0 || l1 + 1 == loop_count {
            println!("Processed segment {}/{}", l1 + 1, loop_count);
        }
    }

    if processed_segments == 0 {
        return Err("No unflagged segments were available for fringe-rate mapping".into());
    }

    // Complex thermal noise averages toward zero. Convert to amplitude only
    // once, after all segment phases have been placed on the same reference.
    let inv_segments = 1.0_f32 / processed_segments as f32;
    let total_map = total_complex_map.mapv(|value| (value * inv_segments).norm());
    let total_beam_map = total_complex_beam.mapv(|value| (value * inv_segments).norm());
    println!(
        "Coherently averaged {} segments ({:.1} seconds)",
        processed_segments,
        processed_segments as f64 * length_in_sectors as f64 * effective_integ_time as f64
    );

    // --- Save Final Maps and Data ---
    println!("Finished coherent processing. Saving outputs...");

    let (max_val, (max_y, max_x)) = find_map_peak(&total_map);
    let (_beam_max, (beam_max_y, beam_max_x)) = find_map_peak(&total_beam_map);

    let map_filename = frinz_dir.join(format!("{}_frmap.png", file_stem));
    plot_sky_map(&map_filename, &total_map, cell_size_rad, max_x, max_y)?;
    println!("Fringe rate map saved to: {:?}", map_filename);

    let _ = fs::remove_file(frinz_dir.join(format!("{}_frmap.bin", file_stem)));

    let beam_map_filename = frinz_dir.join(format!("{}_beam.png", file_stem));
    plot_sky_map(
        &beam_map_filename,
        &total_beam_map,
        cell_size_rad,
        beam_max_x,
        beam_max_y,
    )?;
    println!("Beam map saved to: {:?}", beam_map_filename);

    let uv_coverage_filename = frinz_dir.join(format!("{}_uv.png", file_stem));
    plot_uv_coverage(&uv_coverage_filename, &all_uv_data)?;
    println!("UV coverage plot saved to: {:?}", uv_coverage_filename);

    let _ = fs::remove_file(frinz_dir.join(format!("{}_uv.bin", file_stem)));

    let horizontal_profile = total_map.row(max_y);
    let vertical_profile = total_map.column(max_x);
    let (height, width) = total_map.dim();
    let ra_offsets: Vec<f64> = (0..width)
        .map(|i| ((i as f64) - (width as f64 / 2.0)) * cell_size_rad * rad_to_arcsec * 1_000.0)
        .collect();
    let dec_offsets: Vec<f64> = (0..height)
        .map(|i| ((height as f64 / 2.0) - i as f64) * cell_size_rad * rad_to_arcsec * 1_000.0)
        .collect();

    let npy_meta = |flag| {
        NpyMeta::new(
            flag,
            header.fft_point as u32,
            header.number_of_sector as u32,
        )
        .axes("dec_offset", "mas", "ra_offset", "mas")
    };
    if args.npz {
        let map_npy = npz_sidecar_path(&map_filename, "frmap");
        write_complex_2d(
            &map_npy,
            npy_meta("frmap"),
            (height, width),
            total_complex_map.iter().map(|&value| value * inv_segments),
            &dec_offsets,
            &ra_offsets,
        )?;
        let beam_npy = npz_sidecar_path(&beam_map_filename, "beam");
        write_complex_2d(
            &beam_npy,
            npy_meta("beam"),
            (height, width),
            total_complex_beam.iter().map(|&value| value * inv_segments),
            &dec_offsets,
            &ra_offsets,
        )?;
        let uv_values: Vec<Complex<f32>> = all_uv_data
            .iter()
            .map(|&(u, v)| Complex::new(u as f32, v as f32))
            .collect();
        let uv_index: Vec<f64> = (0..uv_values.len()).map(|index| index as f64).collect();
        let uv_npy = npz_sidecar_path(&uv_coverage_filename, "uv");
        write_complex_1d(
            &uv_npy,
            NpyMeta::new(
                "uv",
                header.fft_point as u32,
                header.number_of_sector as u32,
            )
            .axes("sample", "", "u_real_v_imag", "wavelength"),
            &uv_values,
            &uv_index,
        )?;
        println!(
            "NumPy map data saved to: {:?}, {:?}, {:?}",
            map_npy, beam_npy, uv_npy
        );
    }
    let horizontal_data: Vec<(f64, f32)> = ra_offsets
        .iter()
        .zip(horizontal_profile.iter())
        .map(|(&x, &y)| (x, y))
        .collect();
    let vertical_data: Vec<(f64, f32)> = dec_offsets
        .iter()
        .zip(vertical_profile.iter())
        .map(|(&x, &y)| (x, y))
        .collect();

    let center = (image_size / 2) as f64;
    let (refined_x, refined_y) = refine_map_peak(&total_map, max_y, max_x);
    let l_rad = (refined_x - center) * cell_size_rad;
    let m_rad = (center - refined_y) * cell_size_rad;
    let l_mas = l_rad * rad_to_arcsec * 1_000.0;
    let m_mas = m_rad * rad_to_arcsec * 1_000.0;

    let cross_section_filename = frinz_dir.join(format!("{}_frmap_peak.png", file_stem));
    plot_cross_section(
        cross_section_filename.to_str().unwrap(),
        &horizontal_data,
        &vertical_data,
        max_val,
        l_mas,
        m_mas,
    )?;
    let horizontal_values: Vec<f32> = horizontal_profile.iter().copied().collect();
    let vertical_values: Vec<f32> = vertical_profile.iter().copied().collect();
    if args.npz {
        write_real_1d(
            &npz_sidecar_path(&cross_section_filename, "frmap_peak_ra"),
            NpyMeta::new(
                "frmap_peak_ra",
                header.fft_point as u32,
                header.number_of_sector as u32,
            )
            .axes("ra_offset", "mas", "amplitude", ""),
            &horizontal_values,
            &ra_offsets,
        )?;
        write_real_1d(
            &npz_sidecar_path(&cross_section_filename, "frmap_peak_dec"),
            NpyMeta::new(
                "frmap_peak_dec",
                header.fft_point as u32,
                header.number_of_sector as u32,
            )
            .axes("dec_offset", "mas", "amplitude", ""),
            &vertical_values,
            &dec_offsets,
        )?;
    }
    println!("Cross-section plot saved to: {:?}", cross_section_filename);

    println!("Estimated source position (relative to phase center, sub-pixel fit):");
    println!("  Delta RA: {:.3} mas", l_mas);
    println!("  Delta Dec: {:.3} mas", m_mas);

    Ok(())
}

const C: f64 = 299792458.0; // Speed of light in m/s

fn run_frmap_maser(
    args: &Args,
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
    config: &FrMapConfig,
) -> Result<(), Box<dyn Error>> {
    println!("Starting fringe-rate map analysis (maser mode)...");

    let input_path = args
        .input
        .as_ref()
        .ok_or("Maser fringe-rate mapping requires --input")?;

    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let frinz_dir = parent_dir.join("frinZ").join("frmap");
    fs::create_dir_all(&frinz_dir)?;
    let file_stem = input_path.file_stem().unwrap().to_str().unwrap();

    let buffer = read_input_bytes(input_path)?;
    let mut cursor = Cursor::new(buffer.as_slice());

    let header = parse_header(&mut cursor)?;
    cursor.set_position(256);

    let lambda = C / header.observing_frequency;
    let rad_to_arcsec: f64 = 180.0 / PI * 3600.0;
    let arcsec_to_rad = PI / (180.0 * 3600.0);
    cursor.set_position(0);
    let (_, _obs_start_time, effective_integ_time) =
        read_visibility_data(&mut cursor, &header, 1, 0, 0, false, pp_flag_ranges)?;
    cursor.set_position(256);

    let pp = header.number_of_sector;
    let length_in_sectors = if args.length == 0 {
        pp.max(1)
    } else {
        args.length.max(1).min(pp)
    };
    println!(
        "Processing in segments of {} sectors (approx. {:.2} seconds)",
        length_in_sectors,
        length_in_sectors as f32 * effective_integ_time
    );

    let total_segments_available = (pp - args.skip) / length_in_sectors;
    let loop_count = if args.loop_ == 1 {
        total_segments_available
    } else {
        total_segments_available.min(args.loop_)
    };

    let mut cursor = Cursor::new(buffer.as_slice());
    cursor.set_position(256);

    let mut all_uv_data: Vec<(f32, f32)> = Vec::new();
    let mut lines: Vec<FringeLineMeasurement> = Vec::new();
    let mut max_baseline = 0.0_f64;

    #[derive(Debug)]
    struct PeakCandidate {
        freq_channel: usize,
        rate_idx: usize,
        snr: f64,
        left_amp: f64,
        center_amp: f64,
        right_amp: f64,
    }

    for loop_index in 0..loop_count {
        let (mut complex_vec, segment_start_time, seg_effective_integ_time) =
            match read_visibility_data(
                &mut cursor,
                &header,
                length_in_sectors,
                args.skip,
                loop_index,
                false,
                pp_flag_ranges,
            ) {
                Ok(data) => data,
                Err(_) => break,
            };

        if complex_vec.is_empty() {
            break;
        }

        let is_flagged = time_flag_ranges
            .iter()
            .any(|(start, end)| segment_start_time >= *start && segment_start_time < *end);
        if is_flagged {
            continue;
        }

        if args.delay_correct != 0.0 || args.rate_correct != 0.0 || args.acel_correct != 0.0 {
            println!(
                "Applying phase corrections: delay={}, rate={}, acel={}",
                args.delay_correct, args.rate_correct, args.acel_correct
            );

            let start_time_offset_sec = 0.0;

            apply_phase_correction_in_place(
                &mut complex_vec,
                (header.fft_point / 2) as usize,
                args.rate_correct,
                args.delay_correct,
                args.acel_correct,
                args.jerk_correct,
                args.snap_correct,
                seg_effective_integ_time,
                header.sampling_speed as u32,
                header.fft_point as u32,
                start_time_offset_sec,
            );
        }

        let (freq_rate_array, padding_length) = process_fft(
            &complex_vec,
            length_in_sectors,
            header.fft_point,
            header.sampling_speed,
            &[],
            args.rate_padding,
        );

        let rate_range_vec = rate_cal(padding_length as f32, seg_effective_integ_time);
        let rate_range = Array::from_vec(rate_range_vec);
        let rate_step = if rate_range.len() > 1 {
            (rate_range[1] - rate_range[0]) as f64
        } else {
            0.0
        };

        let segment_duration_sec = length_in_sectors as f64 * seg_effective_integ_time as f64;
        let segment_end_time = segment_start_time
            + chrono::Duration::microseconds((segment_duration_sec * 1_000_000.0) as i64);
        let segment_center_time = segment_start_time
            + chrono::Duration::microseconds(((segment_duration_sec * 1_000_000.0) / 2.0) as i64);

        let (u, v, _w, du_dt, dv_dt) = uvw_cal(
            header.station1_position,
            header.station2_position,
            segment_center_time,
            header.source_position_ra,
            header.source_position_dec,
            true,
        );
        all_uv_data.push((u as f32, v as f32));
        let baseline = (u.powi(2) + v.powi(2)).sqrt();
        if baseline > max_baseline {
            max_baseline = baseline;
        }

        let mut candidates: Vec<PeakCandidate> = Vec::new();

        for (freq_idx, row) in freq_rate_array.axis_iter(Axis(0)).enumerate() {
            if freq_idx == 0 {
                continue;
            }

            let amplitudes: Vec<f64> = row.iter().map(|c| c.norm() as f64).collect();
            if amplitudes.iter().all(|amp| !amp.is_finite() || *amp <= 0.0) {
                continue;
            }

            let mut amps_copy = amplitudes.clone();
            let median = compute_median(&mut amps_copy);
            let mad = compute_mad(&amplitudes, median);
            let noise_sigma = if mad > 0.0 {
                1.4826 * mad
            } else {
                amplitudes
                    .iter()
                    .filter(|val| val.is_finite())
                    .fold(0.0, |acc, val| acc + *val)
                    / amplitudes.len().max(1) as f64
            };

            for r_idx in 1..amplitudes.len().saturating_sub(1) {
                let amp = amplitudes[r_idx];
                if !amp.is_finite() || amp <= 0.0 {
                    continue;
                }
                if amp <= amplitudes[r_idx - 1] || amp <= amplitudes[r_idx + 1] {
                    continue;
                }

                let snr = if noise_sigma > 0.0 {
                    (amp - median).max(0.0) / noise_sigma
                } else {
                    0.0
                };
                if snr < config.snr_threshold {
                    continue;
                }

                let y0 = amplitudes[r_idx - 1];
                let y1 = amp;
                let y2 = amplitudes[r_idx + 1];
                candidates.push(PeakCandidate {
                    freq_channel: freq_idx,
                    rate_idx: r_idx,
                    snr,
                    left_amp: y0,
                    center_amp: y1,
                    right_amp: y2,
                });
            }
        }

        if candidates.is_empty() {
            continue;
        }

        let total_candidates = candidates.len();
        candidates.sort_by(|a, b| b.snr.partial_cmp(&a.snr).unwrap_or(Ordering::Equal));

        let mut added = 0usize;
        for cand in candidates.into_iter().take(config.max_peaks_per_segment) {
            if cand.rate_idx >= rate_range.len() - 1 {
                continue;
            }

            let denom = cand.left_amp - 2.0 * cand.center_amp + cand.right_amp;
            let delta = if denom.abs() > 1.0e-12 {
                0.5 * (cand.left_amp - cand.right_amp) / denom
            } else {
                0.0
            }
            .clamp(-1.0, 1.0);
            let interp_rate = rate_range[cand.rate_idx] as f64 + delta * rate_step;
            let interp_amp = cand.center_amp - 0.25 * (cand.left_amp - cand.right_amp) * delta;

            let snr_for_error = cand.snr.max(1.0);
            let rate_err_hz = if rate_step > 0.0 {
                rate_step / (snr_for_error * 2.0)
            } else {
                0.0
            };

            let line = FringeLineMeasurement {
                index: lines.len(),
                freq_channel: cand.freq_channel,
                start_time: segment_start_time,
                end_time: segment_end_time,
                u,
                v,
                du_dt,
                dv_dt,
                rate_hz: interp_rate,
                rate_err_hz,
                delay_s: f64::NAN,
                delay_err_s: f64::NAN,
                amplitude: interp_amp,
                snr: cand.snr,
            };
            println!(
                "Segment {} ch{:04} -> rate={:.6} Hz (+/-{:.6}) SNR={:.2}",
                loop_index + 1,
                cand.freq_channel,
                interp_rate,
                rate_err_hz,
                cand.snr
            );
            lines.push(line);
            added += 1;
        }

        if total_candidates > config.max_peaks_per_segment {
            println!(
                "Segment {}: {} peaks above SNR {:.1}; kept top {}",
                loop_index + 1,
                total_candidates,
                config.snr_threshold,
                config.max_peaks_per_segment
            );
        }

        if added == 0 {
            continue;
        }
    }

    if lines.is_empty() {
        println!(
            "No segments exceeded SNR threshold ({:.1}). Nothing to plot.",
            config.snr_threshold
        );
        return Ok(());
    }

    let mut intersections: Vec<FringeIntersection> = Vec::new();
    for i in 0..lines.len() {
        for j in (i + 1)..lines.len() {
            let (a1, b1, c1) = lines[i].rate_line_coeffs(lambda);
            let (a2, b2, c2) = lines[j].rate_line_coeffs(lambda);
            let det = a1 * b2 - a2 * b1;
            if det.abs() < 1.0e-12 {
                continue;
            }
            let l = (c1 * b2 - c2 * b1) / det;
            let m = (a1 * c2 - a2 * c1) / det;
            if !l.is_finite() || !m.is_finite() {
                continue;
            }
            let weight = lines[i].weight() * lines[j].weight();
            intersections.push(FringeIntersection {
                l,
                m,
                weight,
                line_i: i,
                line_j: j,
            });
        }
    }

    let base_range_arcsec = match config.range_spec {
        RangeSpec::Auto => {
            let auto_range = auto_range_arcsec(lambda, max_baseline);
            println!(
                "Auto maser map width: {:.2} arcsec (B_max = {:.1} m)",
                auto_range, max_baseline
            );
            auto_range
        }
        RangeSpec::Value(v) => v.max(1.0),
    };
    let mut half_range_rad = (base_range_arcsec * 0.5) * arcsec_to_rad;

    let lines_csv = frinz_dir.join(format!("{}_frmap_lines.csv", file_stem));
    write_line_summary_csv(&lines_csv, &lines)?;
    println!("Line summary saved to {:?}", lines_csv);

    // --- Adjust plotting range so that lines intersect the view ---
    let mut expanded_range = false;
    for line in &lines {
        let denom = line.du_dt.hypot(line.dv_dt);
        if denom <= 1.0e-12 {
            continue;
        }
        let dist = (line.rate_hz * lambda).abs() / denom;
        if dist.is_finite() && dist > half_range_rad {
            half_range_rad = dist * 1.05;
            expanded_range = true;
        }
    }
    let mut final_range_arcsec = half_range_rad * 2.0 * rad_to_arcsec;
    let max_allowed_arcsec = match config.range_spec {
        RangeSpec::Auto => {
            let upper = (base_range_arcsec * 5.0).max(base_range_arcsec);
            upper.min(1.0e6)
        }
        RangeSpec::Value(v) => {
            let base = v.max(1.0);
            (base * 10.0).max(base).min(1.0e6)
        }
    };
    final_range_arcsec = final_range_arcsec.max(base_range_arcsec);
    if final_range_arcsec > max_allowed_arcsec {
        final_range_arcsec = max_allowed_arcsec;
        half_range_rad = (final_range_arcsec * 0.5) * arcsec_to_rad;
        println!(
            "Expanded plot range, capped at {:.3} arcsec to maintain a reasonable scale.",
            final_range_arcsec
        );
    } else if expanded_range {
        println!(
            "Expanded plot range to {:.3} arcsec to include detected fringe-rate lines.",
            final_range_arcsec
        );
    }

    intersections.retain(|pt| pt.l.abs() <= half_range_rad && pt.m.abs() <= half_range_rad);

    if !intersections.is_empty() {
        let intersections_csv = frinz_dir.join(format!("{}_frmap_intersections.csv", file_stem));
        write_intersection_csv(&intersections_csv, &intersections)?;
        println!("Intersections saved to {:?}", intersections_csv);
    }

    let centroid = compute_weighted_stats(&intersections);
    if let Some(stats) = &centroid {
        println!(
            "Weighted centroid: ΔRA = {:.3} arcsec, ΔDec = {:.3} arcsec",
            stats.mean_l * rad_to_arcsec,
            stats.mean_m * rad_to_arcsec
        );
        println!(
            "1σ scatter: σ_RA = {:.3} arcsec, σ_Dec = {:.3} arcsec",
            stats.sigma_l * rad_to_arcsec,
            stats.sigma_m * rad_to_arcsec
        );
    } else {
        println!("Not enough intersections to derive centroid statistics.");
    }

    let plot_path = frinz_dir.join(format!("{}_frmap_maser.png", file_stem));
    plot_fringe_rate_lines(
        &plot_path,
        &lines,
        &intersections,
        centroid.as_ref(),
        lambda,
        final_range_arcsec,
    )?;
    if args.npz {
        let line_values: Vec<Complex<f32>> = lines
            .iter()
            .map(|line| Complex::new(line.du_dt as f32, line.dv_dt as f32))
            .collect();
        let line_rates: Vec<f64> = lines.iter().map(|line| line.rate_hz).collect();
        write_complex_1d(
            &npz_sidecar_path(&plot_path, "frmap_maser_lines"),
            NpyMeta::new(
                "frmap_maser_lines",
                header.fft_point as u32,
                header.number_of_sector as u32,
            )
            .axes("fringe_rate", "Hz", "du_dt_real_dv_dt_imag", "m/s"),
            &line_values,
            &line_rates,
        )?;
        let intersection_values: Vec<Complex<f32>> = intersections
            .iter()
            .map(|point| {
                Complex::new(
                    (point.l * rad_to_arcsec) as f32,
                    (point.m * rad_to_arcsec) as f32,
                )
            })
            .collect();
        let intersection_weights: Vec<f64> =
            intersections.iter().map(|point| point.weight).collect();
        write_complex_1d(
            &npz_sidecar_path(&plot_path, "frmap_maser_intersections"),
            NpyMeta::new(
                "frmap_maser_intersections",
                header.fft_point as u32,
                header.number_of_sector as u32,
            )
            .axes("weight", "", "ra_real_dec_imag", "arcsec"),
            &intersection_values,
            &intersection_weights,
        )?;
    }
    println!("Fringe-rate line plot saved to {:?}", plot_path);

    let uv_coverage_filename = frinz_dir.join(format!("{}_uv.png", file_stem));
    plot_uv_coverage(&uv_coverage_filename, &all_uv_data)?;
    println!("UV coverage plot saved to {:?}", uv_coverage_filename);

    let _ = fs::remove_file(frinz_dir.join(format!("{}_uv.bin", file_stem)));

    Ok(())
}

fn write_line_summary_csv(
    path: &Path,
    lines: &[FringeLineMeasurement],
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "index,freq_channel,start_time_utc,end_time_utc,rate_hz,rate_err_hz,delay_s,delay_err_s,u_m,v_m,du_dt_mps,dv_dt_mps,amplitude,snr"
    )?;
    for line in lines {
        let start_str = line.start_time.to_rfc3339_opts(SecondsFormat::Millis, true);
        let end_str = line.end_time.to_rfc3339_opts(SecondsFormat::Millis, true);
        writeln!(
            file,
            "{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.6},{:.6},{:.9},{:.9},{:.6},{:.3}",
            line.index,
            line.freq_channel,
            start_str,
            end_str,
            line.rate_hz,
            line.rate_err_hz,
            line.delay_s,
            line.delay_err_s,
            line.u,
            line.v,
            line.du_dt,
            line.dv_dt,
            line.amplitude,
            line.snr
        )?;
    }
    Ok(())
}

fn write_intersection_csv(
    path: &Path,
    intersections: &[FringeIntersection],
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    writeln!(file, "line_i,line_j,l_arcsec,m_arcsec,weight")?;
    let rad_to_arcsec: f64 = 180.0 / PI * 3600.0;
    for inter in intersections {
        writeln!(
            file,
            "{},{},{:.6},{:.6},{:.6}",
            inter.line_i,
            inter.line_j,
            inter.l * rad_to_arcsec,
            inter.m * rad_to_arcsec,
            inter.weight
        )?;
    }
    Ok(())
}

fn plot_fringe_rate_lines(
    output_path: &Path,
    lines: &[FringeLineMeasurement],
    intersections: &[FringeIntersection],
    centroid: Option<&CentroidStats>,
    lambda: f64,
    map_range_arcsec: f64,
) -> Result<(), Box<dyn Error>> {
    let backend_size = (1024, 1024);
    let root = BitMapBackend::new(output_path, backend_size).into_drawing_area();
    root.fill(&WHITE)?;

    let half_arcsec = map_range_arcsec * 0.5;
    let rad_to_arcsec: f64 = 180.0 / PI * 3600.0;
    let arcsec_to_rad = PI / (180.0 * 3600.0);
    let limit_rad = half_arcsec * arcsec_to_rad;

    let mut chart = ChartBuilder::on(&root)
        .margin(35)
        .caption("Maser Fringe-Rate Mapping", ("sans-serif", 30))
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(-half_arcsec..half_arcsec, -half_arcsec..half_arcsec)?;

    chart
        .configure_mesh()
        .x_desc("ΔRA (arcsec)")
        .y_desc("ΔDec (arcsec)")
        .x_label_formatter(&|x| format!("{:.0}", x))
        .y_label_formatter(&|y| format!("{:.0}", y))
        .label_style(("sans-serif", 26))
        .light_line_style(&TRANSPARENT)
        .draw()?;

    let max_snr = lines.iter().fold(0.0_f64, |acc, line| acc.max(line.snr));

    for line in lines {
        let (a, b, c) = line.rate_line_coeffs(lambda);
        if let Some((p1, p2)) = clip_line_to_square(a, b, c, limit_rad) {
            let pts = vec![
                (-p1.0 * rad_to_arcsec, p1.1 * rad_to_arcsec),
                (-p2.0 * rad_to_arcsec, p2.1 * rad_to_arcsec),
            ];
            let frac = if max_snr > 0.0 {
                (line.snr / max_snr).clamp(0.15, 1.0)
            } else {
                1.0
            };
            let color = BLUE.mix(frac as f64);
            chart.draw_series(LineSeries::new(pts, color.stroke_width(2)))?;
        }
    }

    if !intersections.is_empty() {
        chart.draw_series(intersections.iter().map(|point| {
            let size = (point.weight.sqrt().clamp(1.5, 6.0) * 2.0) as i32;
            Circle::new(
                (-point.l * rad_to_arcsec, point.m * rad_to_arcsec),
                size,
                &RED.mix(0.7),
            )
        }))?;
    }

    if let Some(stats) = centroid {
        let center = (-stats.mean_l * rad_to_arcsec, stats.mean_m * rad_to_arcsec);
        chart.draw_series(PointSeries::of_element(
            vec![center],
            12,
            &BLACK,
            &|c, s, st| Cross::new(c, s, st.stroke_width(3)),
        ))?;

        let sigma_l = stats.sigma_l * rad_to_arcsec;
        let sigma_m = stats.sigma_m * rad_to_arcsec;

        chart.draw_series(LineSeries::new(
            vec![
                (center.0 - sigma_l, center.1),
                (center.0 + sigma_l, center.1),
            ],
            BLACK.stroke_width(2),
        ))?;
        chart.draw_series(LineSeries::new(
            vec![
                (center.0, center.1 - sigma_m),
                (center.0, center.1 + sigma_m),
            ],
            BLACK.stroke_width(2),
        ))?;
    }

    root.draw(&Text::new(
        format!(
            "Lines: {} | Intersections: {}",
            lines.len(),
            intersections.len()
        ),
        (40, 40),
        ("sans-serif", 22).into_font().color(&BLACK.mix(0.8)),
    ))?;

    root.present()?;
    Ok(())
}

fn geometric_phase_correction(
    phase_u: f64,
    phase_v: f64,
    l: f64,
    m: f64,
    lambda: f64,
) -> Complex<f32> {
    let angle = -2.0 * PI * (phase_u * l + phase_v * m) / lambda;
    Complex::new(angle.cos() as f32, angle.sin() as f32)
}

fn create_complex_map(
    delay_rate_array: &Array2<Complex<f32>>,
    u: f64,
    v: f64,
    du_dt: f64,
    dv_dt: f64,
    phase_u: f64,
    phase_v: f64,
    header: &CorHeader,
    rate_range: &ArrayView1<f32>,
    delay_range: &ArrayView1<f32>,
    image_size: usize,
    cell_size_rad: f64,
) -> Array2<Complex<f32>> {
    let mut image = Array2::<Complex<f32>>::zeros((image_size, image_size));
    if rate_range.len() < 2 || delay_range.len() < 2 {
        return image;
    }

    let center = (image_size / 2) as f64;
    let lambda = C / header.observing_frequency;
    let rate_min = rate_range[0] as f64;
    let rate_max = rate_range[rate_range.len() - 1] as f64;
    let rate_step = (rate_max - rate_min) / (rate_range.len() - 1) as f64;
    let delay_min = delay_range[0] as f64;
    let delay_max = delay_range[delay_range.len() - 1] as f64;
    let delay_step = (delay_max - delay_min) / (delay_range.len() - 1) as f64;

    if rate_step == 0.0 || delay_step == 0.0 {
        return image;
    }

    let l_start = -center * cell_size_rad;
    let phase_step = geometric_phase_correction(phase_u, 0.0, cell_size_rad, 0.0, lambda);

    image
        .as_slice_mut()
        .expect("newly allocated sky map must be contiguous")
        .par_chunks_mut(image_size)
        .enumerate()
        .for_each(|(iy, row)| {
            let m = (center - iy as f64) * cell_size_rad;
            let mut phase = geometric_phase_correction(phase_u, phase_v, l_start, m, lambda);

            for (ix, pixel) in row.iter_mut().enumerate() {
                let l = (ix as f64 - center) * cell_size_rad;
                let delay_s = (u * l + v * m) / C;
                let rate_hz = (du_dt * l + dv_dt * m) / lambda;
                let delay_sample = delay_s * header.sampling_speed as f64;
                let delay_idx_f = (delay_sample - delay_min) / delay_step;
                let rate_idx_f = (rate_hz - rate_min) / rate_step;

                if delay_idx_f.is_finite()
                    && rate_idx_f.is_finite()
                    && delay_idx_f >= 0.0
                    && rate_idx_f >= 0.0
                {
                    let x1 = delay_idx_f.floor() as usize;
                    let y1 = rate_idx_f.floor() as usize;
                    let x2 = x1 + 1;
                    let y2 = y1 + 1;

                    if x2 < delay_rate_array.dim().1 && y2 < delay_rate_array.dim().0 {
                        let xf = (delay_idx_f - x1 as f64) as f32;
                        let yf = (rate_idx_f - y1 as f64) as f32;
                        let p11 = delay_rate_array[[y1, x1]];
                        let p12 = delay_rate_array[[y2, x1]];
                        let p21 = delay_rate_array[[y1, x2]];
                        let p22 = delay_rate_array[[y2, x2]];
                        let value = p11 * (1.0 - xf) * (1.0 - yf)
                            + p21 * xf * (1.0 - yf)
                            + p12 * (1.0 - xf) * yf
                            + p22 * xf * yf;
                        *pixel = value * phase;
                    }
                }
                phase *= phase_step;
            }
        });

    image
}

/// Creates a sky image (l, m) from a delay-rate map.
///
/// # Arguments
/// * `delay_rate_array` - The 2D array of complex visibilities in the delay-rate domain.
/// * `u`, `v` - UV coordinates in meters.
/// * `du_dt`, `dv_dt` - Time derivatives of UV coordinates in meters/sec.
/// * `header` - The correlation header containing observation parameters.
/// * `rate_range` - The range of rates corresponding to the delay-rate map's axis.
/// * `delay_range` - The range of delays corresponding to the delay-rate map's axis.
/// * `image_size` - The width and height of the output image in pixels.
/// * `cell_size_rad` - The angular size of each pixel in radians.
///
/// # Returns
/// A 2D array representing the sky brightness map.
#[allow(dead_code)]
pub fn create_map(
    delay_rate_array: &Array2<Complex<f32>>,
    u: f64,
    v: f64,
    du_dt: f64,
    dv_dt: f64,
    header: &CorHeader,
    rate_range: &ArrayView1<f32>,
    delay_range: &ArrayView1<f32>,
    image_size: usize,
    cell_size_rad: f64,
) -> Array2<f32> {
    let mut image = Array2::<f32>::zeros((image_size, image_size));
    let center = (image_size / 2) as f64;
    let lambda = C / header.observing_frequency;

    let _inv_det = 1.0 / (u * dv_dt - v * du_dt);

    // Pre-calculate ranges for faster access
    let rate_min = rate_range[0] as f64;
    let rate_max = rate_range[rate_range.len() - 1] as f64;
    let rate_step = (rate_max - rate_min) / (rate_range.len() - 1) as f64;

    let delay_min = delay_range[0] as f64;
    let delay_max = delay_range[delay_range.len() - 1] as f64;
    let delay_step = (delay_max - delay_min) / (delay_range.len() - 1) as f64;

    for iy in 0..image_size {
        for ix in 0..image_size {
            // (l, m) coordinates for the current pixel
            let l = ((ix as f64) - center) * cell_size_rad;
            let m = (center - (iy as f64)) * cell_size_rad;

            // Forward transform: from (l, m) to (delay, rate)
            let delay_s = (u * l + v * m) / C;
            let rate_hz = (du_dt * l + dv_dt * m) / lambda;

            // Convert to pixel coordinates in the delay-rate map
            let delay_sample = delay_s * (header.sampling_speed as f64);

            // Find corresponding indices in the delay-rate array
            let delay_idx_f = (delay_sample - delay_min) / delay_step;
            let rate_idx_f = (rate_hz - rate_min) / rate_step;

            // Bilinear interpolation
            let x1 = delay_idx_f.floor() as usize;
            let y1 = rate_idx_f.floor() as usize;
            let x2 = x1 + 1;
            let y2 = y1 + 1;

            if x2 < delay_range.len() && y2 < rate_range.len() {
                let x_frac = delay_idx_f - x1 as f64;
                let y_frac = rate_idx_f - y1 as f64;

                let p11 = delay_rate_array[[y1, x1]].norm() as f64;
                let p12 = delay_rate_array[[y2, x1]].norm() as f64;
                let p21 = delay_rate_array[[y1, x2]].norm() as f64;
                let p22 = delay_rate_array[[y2, x2]].norm() as f64;

                let val = p11 * (1.0 - x_frac) * (1.0 - y_frac)
                    + p21 * x_frac * (1.0 - y_frac)
                    + p12 * (1.0 - x_frac) * y_frac
                    + p22 * x_frac * y_frac;

                image[[iy, ix]] = val as f32;
            }
        }
    }

    image
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrMapMode {
    Continuous,
    Maser,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RangeSpec {
    Auto,
    Value(f64),
}

#[derive(Debug, Clone, Copy)]
struct FrMapConfig {
    mode: FrMapMode,
    snr_threshold: f64,
    range_spec: RangeSpec,
    max_peaks_per_segment: usize,
    grid_size: usize,
    delay_padding: usize,
}

impl Default for FrMapConfig {
    fn default() -> Self {
        Self {
            mode: FrMapMode::Continuous,
            snr_threshold: 5.0,
            range_spec: RangeSpec::Auto,
            max_peaks_per_segment: 12,
            grid_size: 1024,
            delay_padding: 4,
        }
    }
}

fn auto_range_arcsec(lambda: f64, max_baseline: f64) -> f64 {
    if max_baseline <= 0.0 || !max_baseline.is_finite() {
        return 1200.0;
    }
    let angular_res_arcsec = (lambda / max_baseline).to_degrees() * 3600.0;
    (angular_res_arcsec * 4.0).clamp(20.0, 7200.0)
}

impl FrMapConfig {
    fn from_tokens(tokens: &[String]) -> Result<Self, Box<dyn Error>> {
        let mut config = FrMapConfig::default();

        for raw in tokens {
            let token = raw.trim();
            if token.is_empty() {
                continue;
            }

            let (key_raw, value_opt) = if let Some((k, v)) = token.split_once(':') {
                (k, Some(v))
            } else if let Some((k, v)) = token.split_once('=') {
                (k, Some(v))
            } else {
                (token, None)
            };
            let key = key_raw.trim().to_lowercase();
            let value_str = value_opt.map(|v| v.trim());

            match key.as_str() {
                "mode" => {
                    let val = value_str
                        .ok_or_else(|| "mode option requires a value (maser|cont)".to_string())?
                        .to_lowercase();
                    match val.as_str() {
                        "maser" | "mas" => config.mode = FrMapMode::Maser,
                        "cont" | "continuous" | "cw" => config.mode = FrMapMode::Continuous,
                        other => {
                            return Err(format!(
                                "Unknown value '{}' for mode (expected maser|cont)",
                                other
                            )
                            .into())
                        }
                    }
                }
                "maser" => {
                    config.mode = FrMapMode::Maser;
                }
                "cont" | "continuous" => {
                    config.mode = FrMapMode::Continuous;
                }
                "snr" | "snr-threshold" => {
                    let val = value_str.ok_or_else(|| "snr option requires a value".to_string())?;
                    let parsed = val
                        .parse::<f64>()
                        .map_err(|_| format!("Failed to parse SNR threshold value '{}'", val))?;
                    if parsed <= 0.0 {
                        return Err("SNR threshold must be greater than 0".into());
                    }
                    config.snr_threshold = parsed;
                }
                "range" | "range-arcsec" | "arcsec" => {
                    let val =
                        value_str.ok_or_else(|| "range option requires a value".to_string())?;
                    if val.eq_ignore_ascii_case("auto") || val.eq_ignore_ascii_case("automatic") {
                        config.range_spec = RangeSpec::Auto;
                    } else {
                        let parsed = val
                            .parse::<f64>()
                            .map_err(|_| format!("Failed to parse range value '{}'", val))?;
                        if parsed <= 0.0 {
                            return Err("Range must be greater than 0 arcsec".into());
                        }
                        config.range_spec = RangeSpec::Value(parsed);
                    }
                }
                "max" | "maxpeaks" | "max-peaks" => {
                    let val =
                        value_str.ok_or_else(|| "max peaks option requires a value".to_string())?;
                    let parsed = val
                        .parse::<usize>()
                        .map_err(|_| format!("Failed to parse max peaks value '{}'", val))?;
                    if parsed == 0 {
                        return Err("max-peaks must be at least 1".into());
                    }
                    config.max_peaks_per_segment = parsed;
                }
                "grid" | "size" => {
                    let val =
                        value_str.ok_or_else(|| "grid option requires a value".to_string())?;
                    let parsed = val
                        .parse::<usize>()
                        .map_err(|_| format!("Failed to parse grid size {}", val))?;
                    if !(64..=4096).contains(&parsed) {
                        return Err("grid must be between 64 and 4096".into());
                    }
                    config.grid_size = parsed;
                }
                "delay-padding" | "delay-pad" | "dpad" => {
                    let val = value_str
                        .ok_or_else(|| "delay-padding option requires a value".to_string())?;
                    let parsed = val
                        .parse::<usize>()
                        .map_err(|_| format!("Failed to parse delay-padding value {}", val))?;
                    if !matches!(parsed, 1 | 2 | 4 | 8) {
                        return Err("delay-padding must be one of 1, 2, 4, or 8".into());
                    }
                    config.delay_padding = parsed;
                }
                other => {
                    return Err(format!(
                        "Unknown --frmap option '{}'. Expected keys: mode, snr, range, max-peaks, grid, delay-padding.",
                        other
                    )
                    .into());
                }
            }
        }

        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_peak_refinement_recovers_fractional_pixel() {
        let expected_x = 3.25_f64;
        let expected_y = 2.6_f64;
        let mut map = Array2::<f32>::zeros((7, 7));
        for ((row, col), value) in map.indexed_iter_mut() {
            let dx = col as f64 - expected_x;
            let dy = row as f64 - expected_y;
            *value = (10.0 - dx * dx - 2.0 * dy * dy) as f32;
        }
        let (x, y) = refine_map_peak(&map, 3, 3);
        assert!((x - expected_x).abs() < 1.0e-6);
        assert!((y - expected_y).abs() < 1.0e-6);
    }

    #[test]
    fn geometric_rephasing_aligns_segment_phases() {
        let lambda = 0.045;
        let l = 2.0e-6;
        let m = -1.0e-6;
        let mut coherent_sum = Complex::new(0.0_f32, 0.0_f32);
        for (u, v) in [(120_000.0, -30_000.0), (-450_000.0, 210_000.0)] {
            let source_phase = 2.0 * PI * (u * l + v * m) / lambda;
            let visibility = Complex::new(source_phase.cos() as f32, source_phase.sin() as f32);
            coherent_sum += visibility * geometric_phase_correction(u, v, l, m, lambda);
        }
        assert!((coherent_sum.re - 2.0).abs() < 1.0e-5);
        assert!(coherent_sum.im.abs() < 1.0e-5);
    }

    #[test]
    fn synthetic_fringe_is_real_after_complex_sky_rephasing() {
        let fft_point = 16_i32;
        let rows = 16_i32;
        let sampling_speed = 16_000_i32;
        let observing_frequency = 1.0e9_f64;
        let integration_time = 1.0_f32;
        let l = 1.0e-6_f64;
        let delay_samples = 1.0_f64;
        let fringe_rate = 0.125_f64;
        let lambda = C / observing_frequency;
        let phase_u = C * delay_samples / (sampling_speed as f64 * l);
        let du_dt = fringe_rate * lambda / l;
        let midpoint = 0.5 * (rows - 1) as f64 * integration_time as f64;
        let u_mid = phase_u + du_dt * midpoint;
        let phase0 = 2.0 * PI * phase_u * l / lambda;
        let channels = (fft_point / 2) as usize;
        let mut samples = Vec::with_capacity(rows as usize * channels);
        for row in 0..rows as usize {
            for channel in 0..channels {
                let phase = phase0
                    + 2.0
                        * PI
                        * (fringe_rate * row as f64 * integration_time as f64
                            + delay_samples * channel as f64 / fft_point as f64);
                samples.push(Complex::new(phase.cos() as f32, phase.sin() as f32));
            }
        }

        let (freq_rate, padding) = process_fft(&samples, rows, fft_point, sampling_speed, &[], 1);
        let delay_rate = process_ifft_with_delay_padding(&freq_rate, fft_point, padding, 1);
        let rate_range = Array::from_vec(rate_cal(padding as f32, integration_time));
        let delay_range = Array::linspace(
            -(fft_point as f32 / 2.0) + 1.0,
            fft_point as f32 / 2.0,
            fft_point as usize,
        );
        let mut header = CorHeader::default();
        header.observing_frequency = observing_frequency;
        header.sampling_speed = sampling_speed;
        header.fft_point = fft_point;
        let map = create_complex_map(
            &delay_rate,
            u_mid,
            0.0,
            du_dt,
            0.0,
            phase_u,
            0.0,
            &header,
            &rate_range.view(),
            &delay_range.view(),
            3,
            l,
        );
        let recovered = map[[1, 2]];
        assert!(recovered.norm() > 0.0);
        assert!(recovered.re > 0.0, "recovered={recovered:?}");
        assert!(
            recovered.im.abs() < recovered.re.abs() * 0.02,
            "recovered={recovered:?}"
        );
    }

    #[test]
    fn frmap_precision_options_are_parsed() {
        let tokens = vec!["grid:2048".to_string(), "delay-padding:8".to_string()];
        let config = FrMapConfig::from_tokens(&tokens).unwrap();
        assert_eq!(config.grid_size, 2048);
        assert_eq!(config.delay_padding, 8);
    }
}
