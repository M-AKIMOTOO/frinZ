use std::error::Error;
use std::fs;
use std::io::Cursor;
use std::path::Path;

use chrono::{DateTime, Utc};
use num_complex::Complex;

use crate::args::Args;
use crate::bandpass::read_bandpass_file;
use crate::header::{parse_header, CorHeader};
use crate::input_support::open_input_data;
use crate::output::{generate_output_names, insert_product_before_processing_suffixes};
use crate::read::read_visibility_data;
use crate::search;

type C32 = Complex<f32>;

fn split_label(basename: &str) -> Vec<String> {
    let mut label: Vec<String> = basename.split('_').map(String::from).collect();
    if label.len() > 3 {
        let tail = label[3..].join("_");
        label.truncate(3);
        label.push(tail);
    }
    label
}

fn pad_time_rows_to_power_of_two(data: &mut Vec<C32>, current_rows: i32, row_width: usize) -> i32 {
    if current_rows <= 0 || row_width == 0 {
        return current_rows;
    }
    let target_rows = if current_rows <= 1 {
        1
    } else {
        (current_rows as u32).next_power_of_two() as i32
    };
    if target_rows > current_rows {
        data.resize(
            data.len() + (target_rows - current_rows) as usize * row_width,
            C32::new(0.0, 0.0),
        );
    }
    target_rows
}

fn extract_subband(
    full: &[C32],
    rows: usize,
    original_half: usize,
    start_chan: usize,
    width_chan: usize,
) -> Vec<C32> {
    let mut out = Vec::with_capacity(rows * width_chan);
    for row in 0..rows {
        let start = row * original_half + start_chan;
        out.extend_from_slice(&full[start..start + width_chan]);
    }
    out
}

fn apply_full_bandpass_to_time_series(
    data: &mut [C32],
    rows: usize,
    channel_count: usize,
    bandpass: &[C32],
) {
    const EPSILON: f32 = 1e-9;
    if channel_count == 0 || bandpass.len() != channel_count {
        return;
    }

    let bandpass_sum: C32 = bandpass.iter().copied().sum();
    let bandpass_mean = bandpass_sum / bandpass.len() as f32;

    for row in 0..rows {
        let base = row * channel_count;
        for chan in 0..channel_count {
            let bp = bandpass[chan];
            if bp.norm() > EPSILON {
                data[base + chan] = (data[base + chan] / bp) * bandpass_mean;
            }
        }
    }
}

fn parse_local_rfi_ranges(
    rfi_args: &[String],
    band_start_mhz: f32,
    band_width_mhz: f32,
    rbw_mhz: f32,
    channel_count: usize,
) -> Result<Vec<(usize, usize)>, Box<dyn Error>> {
    if rfi_args.is_empty() || rbw_mhz <= 0.0 {
        return Ok(Vec::new());
    }

    let band_end_mhz = band_start_mhz + band_width_mhz;
    let mut ranges = Vec::new();
    for rfi_pair in rfi_args {
        let parts: Vec<&str> = rfi_pair.split(',').collect();
        if parts.len() != 2 {
            return Err(
                format!("Invalid RFI format: {rfi_pair}. Expected format is MIN,MAX.").into(),
            );
        }
        let min_mhz: f32 = parts[0].parse()?;
        let max_mhz: f32 = parts[1].parse()?;
        if min_mhz >= max_mhz {
            return Err(format!("Invalid RFI range: min ({min_mhz}) >= max ({max_mhz}).").into());
        }

        let overlap_low = min_mhz.max(band_start_mhz);
        let overlap_high = max_mhz.min(band_end_mhz);
        if overlap_low >= overlap_high {
            continue;
        }

        let local_low = overlap_low - band_start_mhz;
        let local_high = overlap_high - band_start_mhz;
        let start = (local_low / rbw_mhz).floor().max(0.0) as usize;
        let end = (local_high / rbw_mhz)
            .ceil()
            .max(0.0)
            .min(channel_count.saturating_sub(1) as f32) as usize;
        if start <= end && start < channel_count {
            ranges.push((start, end));
        }
    }
    Ok(ranges)
}

fn validate_inband_width(
    header: &CorHeader,
    inband_mhz: u32,
) -> Result<(usize, usize, f32), Box<dyn Error>> {
    if inband_mhz == 0 || !inband_mhz.is_power_of_two() {
        return Err("--inband must be a non-zero power-of-two MHz value.".into());
    }

    let original_half = (header.fft_point / 2) as usize;
    if original_half == 0 {
        return Err("FFT point is invalid for --inband.".into());
    }

    let bw_mhz = header.sampling_speed as f32 / 2.0 / 1_000_000.0;
    if inband_mhz as f32 > bw_mhz {
        return Err(format!(
            "--inband {inband_mhz} MHz exceeds observing bandwidth {bw_mhz:.3} MHz."
        )
        .into());
    }

    let rbw_mhz = bw_mhz / original_half as f32;
    let channels_per_band_f = inband_mhz as f32 / rbw_mhz;
    let channels_per_band = channels_per_band_f.round() as usize;
    if channels_per_band == 0 || (channels_per_band as f32 - channels_per_band_f).abs() > 1.0e-3 {
        return Err(format!(
            "--inband {inband_mhz} MHz does not map to an integer number of FFT channels (RBW={rbw_mhz:.6} MHz)."
        )
        .into());
    }
    if original_half % channels_per_band != 0 {
        return Err(format!(
            "--inband {inband_mhz} MHz does not evenly divide the observing bandwidth {bw_mhz:.3} MHz."
        )
        .into());
    }

    Ok((
        channels_per_band,
        original_half / channels_per_band,
        rbw_mhz,
    ))
}

pub fn run_inband_analysis(
    args: &Args,
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
) -> Result<(), Box<dyn Error>> {
    let input_path = args
        .input
        .as_ref()
        .ok_or("Error: --inband requires an --input file.")?;
    let inband_mhz = args.inband.ok_or("Error: --inband width is missing.")?;

    if args.frequency {
        return Err(
            "--inband currently runs time-domain delay-rate fringe search; omit --frequency."
                .into(),
        );
    }
    if args.fft_rebin.is_some() {
        return Err("--inband cannot be combined with --fft-rebin.".into());
    }
    if args.norm_acf {
        return Err("--inband currently cannot be combined with --norm-acf.".into());
    }
    if args.scan_correct.is_some() {
        return Err("--inband currently cannot be combined with --scan-correct.".into());
    }

    let requested_search = args.primary_search_mode().unwrap_or("peak");
    if !matches!(requested_search, "peak" | "deep") {
        return Err(
            format!("--inband supports --search peak/deep, not {requested_search}.").into(),
        );
    }

    let input_data = open_input_data(input_path)?;
    let mut cursor = Cursor::new(input_data.as_slice());
    let header = parse_header(&mut cursor)?;
    let original_half = (header.fft_point / 2) as usize;
    let (channels_per_band, band_count, rbw_mhz) = validate_inband_width(&header, inband_mhz)?;

    let bandpass_full = if let Some(bp_path) = &args.bandpass {
        let bp = read_bandpass_file(bp_path)?;
        if bp.len() == original_half {
            Some(bp)
        } else {
            eprintln!(
                "#WARN: bandpass channel count ({}) does not match input channels ({}); inband BP skipped.",
                bp.len(), original_half
            );
            None
        }
    } else {
        None
    };

    cursor.set_position(0);
    let (_, file_start_time, _) =
        read_visibility_data(&mut cursor, &header, 1, 0, 0, false, pp_flag_ranges)?;
    cursor.set_position(256);

    let pp = header.number_of_sector;
    let mut length = if args.length == 0 { pp } else { args.length };
    if args.length != 0 && length > pp {
        length = pp;
    }
    let loop_count = if (pp - args.skip) / length <= 0 {
        1
    } else {
        ((pp - args.skip) / length).min(args.loop_)
    };

    let basename = input_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("inband");
    let label = split_label(basename);
    let label_refs: Vec<&str> = label.iter().map(|s| s.as_str()).collect();

    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let output_dir = parent_dir.join("frinZ").join("inband");
    fs::create_dir_all(&output_dir)?;

    let mut data_rows = String::new();
    let mut time_rows: Vec<(usize, String, f64)> = Vec::new();
    let mut source_name: Option<String> = None;
    let mut length_s: Option<f32> = None;

    if args.bandpass.is_some() {
        println!(
            "#Bandpass applied before in-band split: {}",
            bandpass_full.is_some()
        );
    }

    let mut first_output_basename: Option<String> = None;
    let mut prev_solutions = vec![None; band_count];
    let mut wrote_rows = 0usize;

    for loop_idx in 0..loop_count {
        let (mut complex_vec, current_obs_time, effective_integ_time) = match read_visibility_data(
            &mut cursor,
            &header,
            length,
            args.skip,
            loop_idx,
            false,
            pp_flag_ranges,
        ) {
            Ok(data) => data,
            Err(_) => break,
        };
        let rows = complex_vec.len() / original_half;
        if let Some(bp) = &bandpass_full {
            apply_full_bandpass_to_time_series(&mut complex_vec, rows, original_half, bp);
        }
        if rows == 0 {
            break;
        }
        let physical_length = rows as i32;
        let is_flagged = time_flag_ranges
            .iter()
            .any(|(start, end)| current_obs_time >= *start && current_obs_time < *end);
        if is_flagged {
            continue;
        }

        if first_output_basename.is_none() {
            let filename_length = if args.length == 0 {
                physical_length
            } else {
                args.length
            };
            first_output_basename = Some(generate_output_names(
                &header,
                &current_obs_time,
                &label_refs,
                !args.rfi.is_empty(),
                false,
                bandpass_full.is_some(),
                filename_length,
            ));
        }

        let time_index = time_rows.len();

        for band_idx in 0..band_count {
            let start_chan = band_idx * channels_per_band;
            let band_start_mhz = start_chan as f32 * rbw_mhz;
            let mut subband_vec = extract_subband(
                &complex_vec,
                rows,
                original_half,
                start_chan,
                channels_per_band,
            );
            let current_length =
                pad_time_rows_to_power_of_two(&mut subband_vec, physical_length, channels_per_band);

            let mut sub_header = header.clone();
            sub_header.fft_point = (channels_per_band * 2) as i32;
            sub_header.sampling_speed = (inband_mhz as i32) * 2_000_000;
            sub_header.observing_frequency += band_start_mhz as f64 * 1_000_000.0;

            let local_rfi = parse_local_rfi_ranges(
                &args.rfi,
                band_start_mhz,
                inband_mhz as f32,
                rbw_mhz,
                channels_per_band,
            )?;
            let bandpass = None;

            let mut local_args = args.clone();
            if local_args.primary_search_mode().is_none() {
                local_args.search.push("peak".to_string());
            }
            local_args.frequency = false;
            local_args.inband = None;

            let result = match requested_search {
                "deep" => search::run_deep_search(
                    &subband_vec,
                    &sub_header,
                    current_length,
                    physical_length,
                    effective_integ_time,
                    &current_obs_time,
                    &file_start_time,
                    &local_rfi,
                    &bandpass,
                    &local_args,
                    sub_header.number_of_sector,
                    local_args.cpu,
                    prev_solutions[band_idx],
                )?,
                "peak" => search::run_peak_search(
                    &subband_vec,
                    &sub_header,
                    current_length,
                    physical_length,
                    effective_integ_time,
                    &current_obs_time,
                    &file_start_time,
                    &local_rfi,
                    &bandpass,
                    &local_args,
                    sub_header.number_of_sector,
                    local_args.cpu,
                    prev_solutions[band_idx],
                )?,
                _ => unreachable!(),
            };

            let analysis = &result.analysis_results;
            prev_solutions[band_idx] = Some((analysis.residual_delay, analysis.residual_rate));
            if band_idx == 0 {
                time_rows.push((time_index, analysis.yyyydddhhmmss1.clone(), analysis.mjd));
                if source_name.is_none() {
                    source_name = Some(analysis.source_name.clone());
                }
                if length_s.is_none() {
                    length_s = Some(analysis.length_f32);
                }
            }
            data_rows.push_str(&format!(
                "{}	{}	{:.8}	{:.3}	{:+.3}	{:.8}	{:+.8}	{:+.8}
",
                time_index,
                band_idx,
                analysis.delay_max_amp * 100.0,
                analysis.delay_snr,
                analysis.delay_phase,
                analysis.delay_noise * 100.0,
                analysis.residual_delay,
                analysis.residual_rate
            ));
            wrote_rows += 1;
        }
    }

    let mut output = String::new();
    output.push_str(
        "# In-band fringe search
",
    );
    output.push_str(
        "# format: frinZ_inband_text_v2
",
    );
    output.push_str(&format!(
        "# input: {}
",
        input_path.display()
    ));
    output.push_str(&format!(
        "# label: {}
",
        label.get(3).map(String::as_str).unwrap_or("")
    ));
    if let Some(source) = &source_name {
        output.push_str(&format!(
            "# source: {}
",
            source
        ));
    }
    if let Some(length) = length_s {
        output.push_str(&format!(
            "# length_s: {:.2}
",
            length
        ));
    }
    output.push_str(&format!(
        "# bandpass_applied: {}
",
        bandpass_full.is_some()
    ));
    if let Some(bp_path) = &args.bandpass {
        output.push_str(&format!(
            "# bandpass_file: {}
",
            bp_path.display()
        ));
    }
    output.push_str(&format!(
        "# bandwidth_mhz: {:.3}
# inband_mhz: {}
# bands: {}
# rbw_mhz: {:.6}
",
        header.sampling_speed as f32 / 2.0 / 1_000_000.0,
        inband_mhz,
        band_count,
        rbw_mhz
    ));

    output.push_str(
        "@times
",
    );
    output.push_str(
        "time_index	epoch	mjd
",
    );
    for (time_index, epoch, mjd) in &time_rows {
        output.push_str(&format!(
            "{}	{}	{:.8}
",
            time_index, epoch, mjd
        ));
    }

    output.push_str(
        "@channels
",
    );
    output.push_str(
        "band	band_start_mhz	band_end_mhz	center_mhz
",
    );
    for band_idx in 0..band_count {
        let start_chan = band_idx * channels_per_band;
        let band_start_mhz = start_chan as f32 * rbw_mhz;
        let band_end_mhz = band_start_mhz + inband_mhz as f32;
        let center_mhz =
            header.observing_frequency as f32 / 1_000_000.0 + 0.5 * (band_start_mhz + band_end_mhz);
        output.push_str(&format!(
            "{}	{:.3}	{:.3}	{:.3}
",
            band_idx, band_start_mhz, band_end_mhz, center_mhz
        ));
    }

    output.push_str(
        "@data
",
    );
    output.push_str(
        "time_index	band	amp_percent	snr	phase_deg	noise_percent	res_delay_sample	res_rate_hz
",
    );
    output.push_str(&data_rows);

    let output_basename = first_output_basename.unwrap_or_else(|| basename.to_string());
    let output_stem = insert_product_before_processing_suffixes(&output_basename, "inband");
    let output_path = output_dir.join(format!("{output_stem}.txt"));
    fs::write(&output_path, output)?;
    println!(
        "In-band fringe search saved to: {} ({} rows)",
        output_path.display(),
        wrote_rows
    );
    Ok(())
}
