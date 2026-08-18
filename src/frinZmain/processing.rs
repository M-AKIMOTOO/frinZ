use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Cursor};
use std::path::{Path, PathBuf};
use std::process;

use crate::analysis::AnalysisResults;
use chrono::{DateTime, Duration, Utc};
use ndarray::{Array, Array2};
use num_complex::Complex;

use crate::analysis::analyze_results;
use crate::args::Args;
use crate::bandpass::{apply_bandpass_correction, plot_bandpass_spectrum, read_bandpass_file};
use crate::contamination::{write_contamination_handoff, ContaminationPhaseCorrectionInput};
use crate::contamination_subtract::apply_contamination_subtract;
use crate::fft::{
    apply_phase_correction_in_place_at_frequency, perform_ifft_on_vec, process_fft,
    process_fft_with_phase_correction_at_frequency, process_ifft,
};
use crate::header::{parse_header, CorHeader};
use crate::input_support::{open_input_data, open_input_data_copy_on_write};
use crate::norm_acf::NormAcfContext;
use crate::npy_output::{npz_sidecar_path, NamedNpz, NpyMeta};
use crate::output::{
    format_delay_output, format_delay_tsv_header, format_delay_tsv_row, format_freq_output,
    format_freq_tsv_header, format_freq_tsv_row, generate_output_names,
    insert_product_before_processing_suffixes, output_header_info,
};
use crate::plot::{
    delay_plane, frequency_plane, plot_dynamic_spectrum_freq, plot_dynamic_spectrum_lag,
};
use crate::read::read_visibility_data;
use crate::rfi::parse_rfi_ranges;
use crate::search;
use crate::spike34m::{
    apply_spike_interval_residual_correction, detect_auto_spikes, read_all_spectra, SpikePeak,
};
use crate::stfft;
use crate::utils::{delay_rate_mask_bounds, in_delay_rate_mask, parse_flag_time, safe_arg};
type C32 = Complex<f32>;

pub fn frinz_output_dir(input_path: &Path, in_beam: bool) -> PathBuf {
    let mut output_dir = input_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join("frinZ");
    if in_beam {
        output_dir.push("inbeamVLBI");
    }
    output_dir
}

#[derive(Debug, Clone)]
struct ScanCorrection {
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    delay: f32,
    rate: f32,
}

fn write_spectrum_npz(
    output_path: &Path,
    flag: &str,
    results: &AnalysisResults,
    fft_point: u32,
    pp: u32,
) -> Result<PathBuf, Box<dyn Error>> {
    let spectrum = results
        .freq_rate_spectrum
        .as_slice()
        .ok_or("freq_rate_spectrum is not contiguous")?;
    let axis: Vec<f64> = if results.freq_range.len() == spectrum.len() {
        results
            .freq_range
            .iter()
            .map(|value| *value as f64)
            .collect()
    } else {
        (0..spectrum.len()).map(|index| index as f64).collect()
    };
    let npz_path = npz_sidecar_path(output_path, flag);
    let real: Vec<f64> = spectrum.iter().map(|value| value.re as f64).collect();
    let imag: Vec<f64> = spectrum.iter().map(|value| value.im as f64).collect();
    let amplitude: Vec<f64> = spectrum.iter().map(|value| value.norm() as f64).collect();
    let phase_deg: Vec<f64> = spectrum
        .iter()
        .map(|value| crate::utils::safe_arg(value).to_degrees() as f64)
        .collect();
    let mut npz = NamedNpz::new(NpyMeta::new(flag, fft_point, pp));
    npz.add_f64_1d("frequency_mhz", &axis);
    npz.add_f64_1d("real", &real);
    npz.add_f64_1d("imag", &imag);
    npz.add_f64_1d("amplitude", &amplitude);
    npz.add_f64_1d("phase_deg", &phase_deg);
    npz.write(&npz_path)?;
    Ok(npz_path)
}

fn parse_scan_correct_file(path: &Path) -> Result<Vec<ScanCorrection>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut corrections = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            eprintln!(
                "Warning: Skipping line with less than 5 columns in scan correct file: {}",
                line
            );
            continue;
        }

        let time_str = format!("{} {}", parts[0], parts[1]);
        let time_str_cleaned = time_str.replace('/', "").replace(' ', "").replace(':', "");
        let start_time = match parse_flag_time(&time_str_cleaned) {
            Some(t) => t,
            None => {
                eprintln!(
                    "Warning: Skipping invalid time format in scan correct file: {}",
                    time_str
                );
                continue;
            }
        };

        let duration_sec: f64 = match parts[2].parse() {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "Warning: Skipping invalid duration in scan correct file: {}",
                    parts[2]
                );
                continue;
            }
        };
        let delay: f32 = match parts[3].parse() {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "Warning: Skipping invalid delay in scan correct file: {}",
                    parts[3]
                );
                continue;
            }
        };
        let rate: f32 = match parts[4].parse() {
            Ok(r) => r,
            Err(_) => {
                eprintln!(
                    "Warning: Skipping invalid rate in scan correct file: {}",
                    parts[4]
                );
                continue;
            }
        };

        let end_time = start_time + Duration::seconds(duration_sec.round() as i64);
        corrections.push(ScanCorrection {
            start_time,
            end_time,
            delay,
            rate,
        });
    }

    Ok(corrections)
}

fn find_correction_for_time(
    corrections: &[ScanCorrection],
    time: &DateTime<Utc>,
) -> Option<(f32, f32)> {
    for corr in corrections {
        if *time >= corr.start_time && *time < corr.end_time {
            return Some((corr.delay, corr.rate));
        }
    }
    None
}

fn resolve_delay_rate(
    base_delay: f32,
    base_rate: f32,
    corrections: Option<&[ScanCorrection]>,
    time: &DateTime<Utc>,
) -> (f32, f32) {
    if let Some(corrections) = corrections {
        if let Some((delay, rate)) = find_correction_for_time(corrections, time) {
            return (delay, rate);
        }
    }
    (base_delay, base_rate)
}

fn rebin_complex_rows(
    data: &[C32],
    rows: usize,
    original_cols: usize,
    target_cols: usize,
) -> Vec<C32> {
    if rows == 0
        || original_cols == 0
        || target_cols == 0
        || original_cols == target_cols
        || target_cols > original_cols
        || original_cols % target_cols != 0
    {
        return data.to_vec();
    }

    let group = original_cols / target_cols;
    let mut rebinned = Vec::with_capacity(rows.checked_mul(target_cols).unwrap_or_default());

    for row_idx in 0..rows {
        let row_start = row_idx * original_cols;
        for target_idx in 0..target_cols {
            let mut sum = C32::new(0.0, 0.0);
            for offset in 0..group {
                sum += data[row_start + target_idx * group + offset];
            }
            rebinned.push(sum / group as f32);
        }
    }

    rebinned
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
        let additional_samples = (target_rows - current_rows) as usize * row_width;
        data.extend(std::iter::repeat(C32::new(0.0, 0.0)).take(additional_samples));
    }

    target_rows
}

/// Holds the results of processing a single .cor file, needed for subsequent plotting.
pub struct ProcessResult {
    pub header: CorHeader,
    pub label: Vec<String>,
    pub obs_time: chrono::DateTime<Utc>,
    pub length_arg: i32,
    pub length_sec: f32,
    pub wwz_times_sec: Vec<f32>,
    pub cumulate_len: Vec<f32>,
    pub cumulate_snr: Vec<f32>,
    pub add_plot_times: Vec<DateTime<Utc>>,
    pub add_plot_amp: Vec<f32>,
    pub add_plot_snr: Vec<f32>,
    pub add_plot_phase: Vec<f32>,
    /// Peak baseband frequency [MHz] for each `--frequency --add` window.
    pub add_plot_freq: Vec<f32>,
    pub add_plot_noise: Vec<f32>,
    pub add_plot_res_delay: Vec<f32>,
    pub add_plot_res_rate: Vec<f32>,
    #[allow(dead_code)]
    pub add_plot_complex: Vec<Complex<f32>>,
}

pub fn process_cor_file(
    input_path: &Path,
    args: &Args,
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
    suppress_output: bool,
) -> Result<ProcessResult, Box<dyn Error>> {
    // --- File and Path Setup ---
    let contamination_mode = args.contamination_subtract.is_some();
    let frinz_dir = frinz_output_dir(input_path, args.in_beam);
    fs::create_dir_all(&frinz_dir)?;

    let original_basename = input_path.file_stem().unwrap().to_str().unwrap();
    let basename = if contamination_mode && !original_basename.ends_with("_contamisubt") {
        format!("{}_contamisubt", original_basename)
    } else {
        original_basename.to_string()
    };
    let mut basename_for_output = basename.clone();
    if args.bandpass.is_some() {
        basename_for_output.push_str("_bp");
    }
    if !args.rfi.is_empty() {
        basename_for_output.push_str("_rfi");
    }
    if args.spike34m.is_some() {
        basename_for_output.push_str("_spike34");
    }
    if args.in_beam {
        basename_for_output.push_str("_inbeam");
    }
    let mut label: Vec<String> = basename.split('_').map(String::from).collect();
    if label.len() > 3 {
        let tail = label[3..].join("_");
        label.truncate(3);
        label.push(tail);
    }

    // --- Create Output Directories ---
    let mut plot_path: Option<PathBuf> = None;
    if args.plot {
        let path = if args.in_beam {
            frinz_dir.clone()
        } else {
            frinz_dir.join("fringe_graph")
        };
        fs::create_dir_all(&path)?;
        plot_path = Some(path);
    }

    let mut output_path: Option<PathBuf> = None;
    if args.output {
        let path = if args.in_beam {
            frinz_dir.clone()
        } else {
            frinz_dir.join("fringe_output")
        };
        fs::create_dir_all(&path)?;
        output_path = Some(path);
    }

    let mut bandpass_output_path: Option<PathBuf> = None;
    if args.bandpass_table {
        let path = if args.in_beam {
            frinz_dir.clone()
        } else {
            frinz_dir.join("bptable")
        };
        fs::create_dir_all(&path)?;
        bandpass_output_path = Some(path);
    }

    if args.cumulate != 0 && !args.in_beam {
        let path = frinz_dir.join(format!("cumulate/len{}s", args.cumulate));
        fs::create_dir_all(&path)?;
    }

    if !args.rfi.is_empty() {
        let _ = frinz_dir.join("rfi_history");
    }

    if args.add_plot && !args.in_beam {
        let path = frinz_dir.join("add_plot");
        fs::create_dir_all(&path)?;
    }

    let mut spectrum_output_path: Option<PathBuf> = None;
    if args.spectrum {
        let path = if args.in_beam {
            frinz_dir.clone()
        } else {
            frinz_dir.join("spectrum")
        };
        fs::create_dir_all(&path)?;
        spectrum_output_path = Some(path);
    }

    // --- Read .cor and apply the compact table to a private copy-on-write mapping. ---
    let input_data = if let Some(model_path) = &args.contamination_subtract {
        let mut data = open_input_data_copy_on_write(input_path)?;
        let bytes = data
            .as_mut_slice()
            .ok_or("contamination correction requires mutable copy-on-write input")?;
        apply_contamination_subtract(bytes, input_path, model_path, args.bandpass.as_deref())?;
        data
    } else {
        open_input_data(input_path)?
    };
    let mut cursor = Cursor::new(input_data.as_slice());

    // --- Parse Header ---
    let header = parse_header(&mut cursor)?;
    let original_fft_point = header.fft_point;

    let mut effective_fft_point = original_fft_point;
    if let Some(requested_fft_point) = args.fft_rebin {
        if requested_fft_point <= 0 {
            eprintln!("Error: --fft-rebin には正の値を指定してください。");
            process::exit(1);
        }
        if requested_fft_point % 2 != 0 {
            eprintln!("Error: --fft-rebin は偶数である必要があります。");
            process::exit(1);
        }
        if requested_fft_point > original_fft_point {
            eprintln!(
                "Error: --fft-rebin ({}) はヘッダーの FFT 点数 ({}) を超えています。",
                requested_fft_point, original_fft_point
            );
            process::exit(1);
        }

        let original_half = (original_fft_point / 2) as usize;
        let requested_half = (requested_fft_point / 2) as usize;
        if requested_half == 0 || original_half % requested_half != 0 {
            eprintln!(
                "Error: --fft-rebin ({}) は元のチャンネル数 ({}) を整数分割できません。",
                requested_fft_point, original_fft_point
            );
            process::exit(1);
        }

        effective_fft_point = requested_fft_point;
    }

    let bw = header.sampling_speed as f32 / 2.0 / 1_000_000.0;
    let rbw = bw / effective_fft_point as f32 * 2.0;

    // --- RFI Handling ---
    let rfi_ranges = parse_rfi_ranges(&args.rfi, rbw)?;
    let rfi_display = if args.rfi.is_empty() {
        "-".to_string()
    } else if rfi_ranges.is_empty() {
        "(invalid)".to_string()
    } else {
        args.rfi.join(",")
    };
    // RFI ranges are reflected in output tables; no additional banner needed.

    // --- Bandpass Handling ---
    let mut bandpass_data = if let Some(bp_path) = &args.bandpass {
        Some(read_bandpass_file(bp_path)?)
    } else {
        None
    };
    if effective_fft_point != original_fft_point {
        let original_half = (original_fft_point / 2) as usize;
        let target_half = (effective_fft_point / 2) as usize;
        if let Some(bp) = bandpass_data.as_mut() {
            if bp.len() == original_half {
                let rebinned = rebin_complex_rows(bp, 1, original_half, target_half);
                *bp = rebinned;
            } else if bp.len() != target_half {
                eprintln!(
                    "#WARN: バンドパスデータのチャンネル数 ({}) が FFT リビン後のチャンネル数 ({}) と一致しません。補正をスキップします。",
                    bp.len(),
                    target_half
                );
                *bp = Vec::new();
            }
        }
    }

    let bandpass_active = bandpass_data.as_ref().map_or(false, |bp| !bp.is_empty());
    if args.bandpass.is_some() {
        println!(
            "#Bandpass applied: {}",
            if bandpass_active { "True" } else { "False" }
        );
    }

    let norm_acf_context = if args.norm_acf {
        let context = NormAcfContext::load(input_path, &header)?;
        let (left_path, right_path) = context.path_pair();
        println!(
            "#Norm-ACF files: {} | {}",
            left_path.display(),
            right_path.display()
        );
        Some(context)
    } else {
        None
    };

    let mut processing_header = header.clone();
    processing_header.fft_point = effective_fft_point;

    let scan_corrections = if let Some(path) = &args.scan_correct {
        Some(parse_scan_correct_file(path)?)
    } else {
        None
    };

    let spike34m_peaks: Option<Vec<SpikePeak>> = if let Some(spike_path) = &args.spike34m {
        let (auto_header, auto_spectra, _) = read_all_spectra(spike_path)?;
        let spikes = detect_auto_spikes(&auto_header, &auto_spectra);
        if spikes.len() < 2 {
            return Err("--spike34 found fewer than two YAMAGU34 auto-correlation spikes".into());
        }
        Some(spikes)
    } else {
        None
    };

    // --- Output Header Information ---
    if args.output || args.header {
        let header_path = frinz_dir.join("header");
        fs::create_dir_all(&header_path)?;
        let header_info_str = output_header_info(&header, &header_path, &basename_for_output)?;
        if args.header {
            println!("{}", header_info_str);
        }
    }

    // --- Loop and Processing Setup ---
    cursor.set_position(0);
    let (_, file_start_time, effective_integ_time) =
        read_visibility_data(&mut cursor, &header, 1, 0, 0, false, pp_flag_ranges)?;
    cursor.set_position(256);

    let pp = header.number_of_sector;
    let mut length = if args.length == 0 { pp } else { args.length };
    if args.length != 0 && args.length > pp {
        length = pp;
    }
    let stfft_plan = stfft::build_plan(args, pp)?;
    let mut loop_count = if let Some(plan) = stfft_plan {
        plan.windows
    } else if (pp - args.skip) / length <= 0 {
        1
    } else if (pp - args.skip) / length <= args.loop_ {
        (pp - args.skip) / length
    } else {
        args.loop_
    };

    if args.cumulate != 0 {
        let available = pp.saturating_sub(args.skip.max(0));
        if args.cumulate > available {
            eprintln!(
                "The specified cumulation length, {} s, is more than the observation time, {} s.",
                args.cumulate, available
            );
            process::exit(1);
        }
        length = args.cumulate;
        loop_count = available.saturating_add(args.cumulate - 1) / args.cumulate;
    }

    let mut delay_tsv = String::new();
    let mut freq_tsv = String::new();
    let mut cumulate_len: Vec<f32> = Vec::new();
    let mut cumulate_snr: Vec<f32> = Vec::new();
    let mut wwz_times_sec: Vec<f32> = Vec::new();
    let mut add_plot_amp: Vec<f32> = Vec::new();
    let mut add_plot_phase: Vec<f32> = Vec::new();
    let mut add_plot_freq: Vec<f32> = Vec::new();
    let mut add_plot_snr: Vec<f32> = Vec::new();
    let mut add_plot_noise: Vec<f32> = Vec::new();
    let mut add_plot_times: Vec<DateTime<Utc>> = Vec::new();
    let mut add_plot_res_delay: Vec<f32> = Vec::new();
    let mut add_plot_res_rate: Vec<f32> = Vec::new();
    let mut add_plot_complex: Vec<Complex<f32>> = Vec::new();

    let mut prev_deep_solution: Option<(f32, f32)> = None;
    let mut first_output_basename: Option<String> = None;

    for l1 in 0..loop_count {
        let requested_length = if args.cumulate != 0 {
            ((l1 + 1) * length).min(pp.saturating_sub(args.skip.max(0)))
        } else {
            length
        };
        let read_skip = stfft::window_start(args.skip, l1, stfft_plan);
        let read_loop_index = stfft::read_loop_index(l1, stfft_plan);
        let (mut complex_vec, current_obs_time, effective_integ_time) = match read_visibility_data(
            &mut cursor,
            &header,
            requested_length,
            read_skip,
            read_loop_index,
            args.cumulate != 0,
            pp_flag_ranges,
        ) {
            Ok(data) => data,
            Err(_) => break,
        };

        let original_fft_half = (header.fft_point / 2) as usize;
        if original_fft_half == 0 {
            eprintln!("#ERROR: FFT point が不正です (0)。");
            break;
        }

        if complex_vec.len() % original_fft_half != 0 {
            eprintln!(
                "#ERROR: 読み込んだデータ長 ({}) が FFT チャンネル数 ({}) の整数倍ではありません。",
                complex_vec.len(),
                original_fft_half
            );
            break;
        }

        let actual_length = (complex_vec.len() / original_fft_half) as i32;
        if actual_length == 0 {
            eprintln!(
                "#INFO: skip/length の指定により読み取れるセクターが残っていないため、処理を終了します。"
            );
            break;
        }

        let physical_length = actual_length;
        // Preserve exactly what was read from .cor.  The contamination
        // handoff must describe the frame into which frinZ later subtracts;
        // normalization, rebinning and fringe corrections below are analysis
        // operations and must not alter this copy.
        let raw_contamination_visibility =
            args.contamination.is_some().then(|| complex_vec.clone());
        // Every reported fringe value is referenced to the first sample of
        // this --length window.  Its timestamp must therefore be the window
        // start, not the integration midpoint.
        let phase_obs_time = current_obs_time;

        if let Some(norm_ctx) = &norm_acf_context {
            norm_ctx.normalize_cross_visibility(
                &mut complex_vec,
                current_obs_time,
                effective_integ_time,
                requested_length,
                read_skip,
                read_loop_index,
                args.cumulate != 0,
                pp_flag_ranges,
            )?;
        }

        if effective_fft_point != header.fft_point {
            let target_fft_half = (effective_fft_point / 2) as usize;
            complex_vec = rebin_complex_rows(
                &complex_vec,
                actual_length as usize,
                original_fft_half,
                target_fft_half,
            );
        }

        let fft_point_half_used = (effective_fft_point / 2) as usize;
        if complex_vec.len() != actual_length as usize * fft_point_half_used {
            eprintln!(
                "#ERROR: FFT リビン処理後のデータ長 ({}) が期待値 ({}) と一致しません。",
                complex_vec.len(),
                actual_length as usize * fft_point_half_used
            );
            break;
        }

        let (manual_delay_correct, manual_rate_correct) = resolve_delay_rate(
            args.delay_correct,
            args.rate_correct,
            scan_corrections.as_deref(),
            &current_obs_time,
        );
        let manual_acel_correct = args.acel_correct;

        if manual_delay_correct != 0.0 || manual_rate_correct != 0.0 || manual_acel_correct != 0.0 {
            let start_time_offset_sec = current_obs_time
                .signed_duration_since(file_start_time)
                .num_milliseconds() as f32
                / 1000.0;

            apply_phase_correction_in_place_at_frequency(
                &mut complex_vec,
                fft_point_half_used,
                manual_rate_correct,
                manual_delay_correct,
                manual_acel_correct,
                args.jerk_correct,
                args.snap_correct,
                effective_integ_time,
                header.sampling_speed as u32,
                effective_fft_point as u32,
                start_time_offset_sec,
                processing_header.observing_frequency,
            );
        }

        // --spike34 uses the full-band fringe solution as the reference
        // frame. Only after that global correction do we fit the residual
        // phase/rate of every channel and smooth it between the detected
        // YAMAGU34 spikes. Applying each sub-band search result independently
        // would erase the physical full-band phase trend and can reduce SNR.
        let mut spike34_applied_delay = 0.0f32;
        let mut spike34_applied_rate = 0.0f32;
        if let Some(spikes) = &spike34m_peaks {
            let mut search_vec = complex_vec.clone();
            let search_length =
                pad_time_rows_to_power_of_two(&mut search_vec, actual_length, fft_point_half_used);
            let mut fullband_args = args.clone();
            fullband_args.spike34m = None;
            fullband_args.search = vec!["peak".to_string()];
            fullband_args.frequency = false;
            // Force the final FFT evaluation so the global solution has the
            // same phase convention as the user-visible spectrum/search path.
            fullband_args.spectrum = true;
            fullband_args.plot = false;
            fullband_args.raw_visibility = false;
            fullband_args.delay_correct = 0.0;
            fullband_args.rate_correct = 0.0;
            fullband_args.acel_correct = 0.0;
            fullband_args.jerk_correct = 0.0;
            fullband_args.snap_correct = 0.0;
            fullband_args.rate_padding = fullband_args.rate_padding.max(4);
            let fullband = search::run_peak_search(
                &search_vec,
                &processing_header,
                search_length,
                physical_length,
                effective_integ_time,
                &current_obs_time,
                &file_start_time,
                &rfi_ranges,
                &bandpass_data,
                &fullband_args,
                pp,
                fullband_args.cpu,
                None,
            )?;
            let fullband_delay = fullband.analysis_results.residual_delay;
            spike34_applied_delay = fullband_delay;
            // --spike34 deliberately leaves the time/rate fringe term intact
            // so the YAMAGU34-induced rate splitting remains observable.
            spike34_applied_rate = 0.0;
            apply_phase_correction_in_place_at_frequency(
                &mut complex_vec,
                fft_point_half_used,
                0.0,
                fullband_delay,
                0.0,
                0.0,
                0.0,
                effective_integ_time,
                header.sampling_speed as u32,
                effective_fft_point as u32,
                0.0,
                processing_header.observing_frequency,
            );
            let spectra: Vec<Vec<C32>> = complex_vec
                .chunks(fft_point_half_used)
                .map(|row| row.to_vec())
                .collect();
            let corrected_spectra = apply_spike_interval_residual_correction(
                &processing_header,
                &spectra,
                effective_integ_time,
                spikes,
            );
            complex_vec = corrected_spectra.into_iter().flatten().collect();
        }

        let current_length =
            pad_time_rows_to_power_of_two(&mut complex_vec, actual_length, fft_point_half_used);

        let is_flagged = time_flag_ranges
            .iter()
            .any(|(start, end)| current_obs_time >= *start && current_obs_time < *end);

        if is_flagged {
            println!(
                "#INFO: Skipping data at {} due to --flagging time range.",
                current_obs_time.format("%Y-%m-%d %H:%M:%S")
            );
            continue;
        }

        let mut loop_args = args.clone();
        loop_args.delay_correct = 0.0;
        loop_args.rate_correct = 0.0;
        loop_args.acel_correct = 0.0;

        let primary_search_mode = args.primary_search_mode();

        let (mut analysis_results, freq_rate_array, delay_rate_2d_data_comp, pre_bandpass_results) =
            match primary_search_mode {
                Some("peak") | Some("deep") | Some("coherent") => {
                    // Unified fringe search path:
                    //   peak = fast AxisThenLocal search + final local polish
                    //   deep = full-grid hierarchical search
                    //
                    // Do not feed the previous solution into rate_correct/delay_correct.
                    // It is only used as the next search seed inside run_*_search().
                    // STFFT windows are independent; do not seed from an overlapping window.
                    let search_seed = if stfft_plan.is_some() {
                        None
                    } else {
                        prev_deep_solution
                    };
                    let mut search_result = if primary_search_mode == Some("deep") {
                        search::run_deep_search(
                            &complex_vec,
                            &processing_header,
                            current_length,
                            physical_length,
                            effective_integ_time,
                            &current_obs_time,
                            &file_start_time,
                            &rfi_ranges,
                            &bandpass_data,
                            &loop_args,
                            pp,
                            loop_args.cpu,
                            search_seed,
                        )?
                    } else if primary_search_mode == Some("coherent") {
                        search::run_coherent_search(
                            &complex_vec,
                            &processing_header,
                            current_length,
                            physical_length,
                            effective_integ_time,
                            &current_obs_time,
                            &file_start_time,
                            &rfi_ranges,
                            &bandpass_data,
                            &loop_args,
                            pp,
                            loop_args.cpu,
                            search_seed,
                        )?
                    } else {
                        search::run_peak_search(
                            &complex_vec,
                            &processing_header,
                            current_length,
                            physical_length,
                            effective_integ_time,
                            &current_obs_time,
                            &file_start_time,
                            &rfi_ranges,
                            &bandpass_data,
                            &loop_args,
                            pp,
                            loop_args.cpu,
                            search_seed,
                        )?
                    };
                    search_result.analysis_results.residual_delay -= loop_args.delay_correct;
                    search_result.analysis_results.residual_rate -= loop_args.rate_correct;
                    // corrected_* should represent user-provided/static correction values
                    // (e.g. --delay/--rate or scan-correct), not search-updated totals.
                    search_result.analysis_results.corrected_delay = manual_delay_correct;
                    search_result.analysis_results.corrected_rate = manual_rate_correct;
                    let result_tuple = (
                        search_result.analysis_results,
                        search_result.freq_rate_array,
                        search_result.delay_rate_2d_data,
                        search_result.pre_bandpass_analysis_results,
                    );

                    if stfft_plan.is_none() {
                        prev_deep_solution = Some((
                            result_tuple.0.residual_delay + loop_args.delay_correct,
                            result_tuple.0.residual_rate + loop_args.rate_correct,
                        ));
                    }

                    result_tuple
                }
                _ => {
                    // No search or other modes not handled here
                    let (
                        mut analysis_results,
                        freq_rate_array,
                        delay_rate_2d_data_comp,
                        pre_bandpass_results,
                    ) = run_analysis_pipeline(
                        &complex_vec,
                        &processing_header,
                        &loop_args,
                        None,
                        loop_args.delay_correct,
                        loop_args.rate_correct,
                        loop_args.acel_correct,
                        current_length,
                        physical_length,
                        effective_integ_time,
                        &current_obs_time,
                        &file_start_time,
                        &rfi_ranges,
                        &bandpass_data,
                        args.plot,
                        effective_fft_point,
                    )?;
                    analysis_results.length_f32 =
                        (physical_length as f32 * effective_integ_time).ceil();
                    (
                        analysis_results,
                        freq_rate_array,
                        delay_rate_2d_data_comp,
                        pre_bandpass_results,
                    )
                }
            };

        analysis_results.length_f32 = physical_length as f32 * effective_integ_time;

        let label_str: Vec<&str> = label.iter().map(|s| s.as_str()).collect();
        let filename_length = if args.length == 0 {
            physical_length
        } else {
            args.length
        };
        let mut base_filename = generate_output_names(
            &processing_header,
            &current_obs_time,
            &label_str,
            !rfi_ranges.is_empty(),
            args.frequency,
            args.bandpass.is_some(),
            filename_length,
        );
        if args.spike34m.is_some() && !base_filename.ends_with("_spike34") {
            base_filename.push_str("_spike34");
        }
        if args.in_beam && !base_filename.ends_with("_inbeam") {
            base_filename.push_str("_inbeam");
        }
        if first_output_basename.is_none() {
            first_output_basename = Some(base_filename.clone());
        }

        if args.contamination.is_some() {
            let manual_start_time_offset_sec = current_obs_time
                .signed_duration_since(file_start_time)
                .num_milliseconds() as f32
                / 1000.0;
            let search_start_time_offset_sec = 0.0;
            let correction = ContaminationPhaseCorrectionInput {
                manual_delay_sample: manual_delay_correct,
                manual_rate_hz: manual_rate_correct,
                manual_acel_hz_per_s: manual_acel_correct,
                manual_jerk_hz_per_s2: args.jerk_correct,
                manual_snap_hz_per_s3: args.snap_correct,
                manual_start_time_offset_s: manual_start_time_offset_sec,
                search_delay_sample: analysis_results.residual_delay,
                search_rate_hz: analysis_results.residual_rate,
                search_start_time_offset_s: search_start_time_offset_sec,
                target_frame_rotation_deg: 0.0,
            };
            write_contamination_handoff(
                input_path,
                args,
                &processing_header,
                &frinz_dir,
                &base_filename,
                current_obs_time,
                effective_integ_time,
                physical_length,
                analysis_results.delay_peak_complex,
                analysis_results.residual_delay,
                analysis_results.residual_rate,
                analysis_results.delay_snr,
                analysis_results.delay_noise,
                correction,
                bandpass_data.as_deref().filter(|values| !values.is_empty()),
                raw_contamination_visibility
                    .as_deref()
                    .ok_or("contamination visibility copy is unavailable")?,
            )?;
        }

        if args.spectrum {
            if let Some(path) = &spectrum_output_path {
                let output_stem =
                    insert_product_before_processing_suffixes(&base_filename, "spectrum");
                let output_file_path = path.join(format!("{output_stem}.npz"));
                let legacy_stem =
                    insert_product_before_processing_suffixes(&base_filename, "cross");
                let _ = fs::remove_file(path.join(format!("{legacy_stem}.spec")));
                let npz_path = write_spectrum_npz(
                    &output_file_path,
                    "spectrum",
                    &analysis_results,
                    effective_fft_point as u32,
                    processing_header.number_of_sector as u32,
                )?;
                plot_bandpass_spectrum(
                    &output_file_path,
                    analysis_results
                        .freq_rate_spectrum
                        .as_slice()
                        .ok_or("freq_rate_spectrum が連続メモリではありません")?,
                    effective_fft_point,
                    1,
                )?;
                println!("Spectrum NPZ written to {:?}", npz_path);
            }
        }

        if args.bandpass_table {
            if let Some(path) = &bandpass_output_path {
                let output_stem =
                    insert_product_before_processing_suffixes(&base_filename, "bptable");
                let output_file_path = path.join(format!("{output_stem}.npz"));
                let _ = fs::remove_file(path.join(format!("{output_stem}.bin")));
                let npz_path = write_spectrum_npz(
                    &output_file_path,
                    "bptable",
                    &analysis_results,
                    effective_fft_point as u32,
                    processing_header.number_of_sector as u32,
                )?;
                plot_bandpass_spectrum(
                    &output_file_path,
                    analysis_results
                        .freq_rate_spectrum
                        .as_slice()
                        .ok_or("freq_rate_spectrum が連続メモリではありません")?,
                    effective_fft_point,
                    0,
                )?;
                println!("Bandpass NPZ written to {:?}", npz_path);
            }
        }

        if args.dynamic_spectrum {
            let dynamic_spectrum_dir = if args.in_beam {
                frinz_dir.clone()
            } else {
                frinz_dir.join("dynamic_spectrum")
            };
            fs::create_dir_all(&dynamic_spectrum_dir)?;
            let label_str: Vec<&str> = label.iter().map(|s| s.as_str()).collect();
            let mut base_filename = generate_output_names(
                &processing_header,
                &current_obs_time,
                &label_str,
                !rfi_ranges.is_empty(),
                args.frequency,
                args.bandpass.is_some(),
                filename_length,
            );
            if args.spike34m.is_some() && !base_filename.ends_with("_spike34") {
                base_filename.push_str("_spike34");
            }
            if args.in_beam && !base_filename.ends_with("_inbeam") {
                base_filename.push_str("_inbeam");
            }
            let fft_point_half = (effective_fft_point / 2) as usize;
            let available_rows = complex_vec.len() / fft_point_half;
            let requested_rows = physical_length.max(0) as usize;
            let usable_rows = requested_rows.min(available_rows);
            let usable_len = usable_rows * fft_point_half;
            let truncated_vec = complex_vec[..usable_len].to_vec();
            let spectrum_array =
                Array::from_shape_vec((usable_rows, fft_point_half), truncated_vec).unwrap();
            let frequency_stem = insert_product_before_processing_suffixes(
                &base_filename,
                "dynamic_spectrum_frequency",
            );
            let output_path_freq = dynamic_spectrum_dir.join(format!("{frequency_stem}.png"));
            plot_dynamic_spectrum_freq(
                output_path_freq.to_str().unwrap(),
                &spectrum_array,
                &processing_header,
                &current_obs_time,
                current_length,
                effective_integ_time,
            )?;
            let time_axis: Vec<f64> = (0..usable_rows)
                .map(|row| row as f64 * effective_integ_time as f64)
                .collect();
            let channel_width_mhz =
                processing_header.sampling_speed as f64 / effective_fft_point as f64 / 1.0e6;
            let frequency_axis: Vec<f64> = (0..fft_point_half)
                .map(|channel| {
                    processing_header.observing_frequency / 1.0e6
                        + channel as f64 * channel_width_mhz
                })
                .collect();
            let dynamic_npz = if args.npz {
                let mut npz = NamedNpz::new(NpyMeta::new(
                    "dynamic_spectrum",
                    effective_fft_point as u32,
                    processing_header.number_of_sector as u32,
                ));
                npz.add_f64_1d("time_s", &time_axis);
                npz.add_f64_1d("frequency_mhz", &frequency_axis);
                npz.add_complex64_2d(
                    "spectrum",
                    spectrum_array.dim(),
                    spectrum_array.iter().copied(),
                )?;
                Some(npz)
            } else {
                None
            };
            let mut lag_data = Array::zeros((usable_rows, effective_fft_point as usize));
            let fft_point_usize = effective_fft_point as usize;
            for (i, row) in spectrum_array.rows().into_iter().enumerate() {
                let shifted_out = perform_ifft_on_vec(row.as_slice().unwrap(), fft_point_usize);
                for (j, val) in shifted_out.iter().enumerate() {
                    lag_data[[i, j]] = val.norm();
                }
            }
            let lag_stem = insert_product_before_processing_suffixes(
                &base_filename,
                "dynamic_spectrum_time_lag",
            );
            let output_path_lag = dynamic_spectrum_dir.join(format!("{lag_stem}.png"));
            plot_dynamic_spectrum_lag(
                output_path_lag.to_str().unwrap(),
                &lag_data,
                &processing_header,
                &current_obs_time,
                current_length,
                effective_integ_time,
            )?;
            let delay_axis: Vec<f64> = (0..fft_point_usize)
                .map(|index| -(fft_point_usize as f64 / 2.0) + 1.0 + index as f64)
                .collect();
            if let Some(mut npz) = dynamic_npz {
                npz.add_f64_1d("delay_sample", &delay_axis);
                npz.add_f32_2d("lag_amplitude", lag_data.dim(), lag_data.iter().copied())?;
                npz.write(&output_path_freq.with_extension("npz"))?;
                let _ = fs::remove_file(output_path_lag.with_extension("npz"));
            }
        }

        if !args.frequency {
            let delay_output_line = format_delay_output(
                &analysis_results,
                &label_str,
                args.length,
                &rfi_display,
                bandpass_active,
                norm_acf_context.is_some(),
            );
            if l1 == 0 {
                let station1_label = format!("{}-azel", header.station1_name.trim());
                let station2_label = format!("{}-azel", header.station2_name.trim());
                let header_str = format!(
                        concat!(
                            "#*************************************************************************************************************************************************************************************************************************\n",
                            "#      Epoch        Label    Source     Length    Amp      SNR     Phase     Noise-level      Res-Delay     Res-Rate            {:<10}              {:<10}             MJD        RFI        BP    ACF \n",
                            "#                                        [s]      [%]               [deg]     1-sigma[%]       [sample]       [Hz]      az[deg]  el[deg]  hgt[m]    az[deg]  el[deg]  hgt[m]                   [MHz]      [T/F] [T/F]\n",
                            "#*************************************************************************************************************************************************************************************************************************"
                        ),
                        station1_label,
                        station2_label
                    );
                if !suppress_output {
                    print!("{}\n", header_str);
                }
                if args.output {
                    delay_tsv.push_str(&format_delay_tsv_header(
                        &header.station1_name,
                        &header.station2_name,
                    ));
                }
            }
            if !suppress_output {
                print!("{}\n", delay_output_line);
            }
            if args.output {
                delay_tsv.push_str(&format_delay_tsv_row(
                    &analysis_results,
                    &label_str,
                    &rfi_display,
                    bandpass_active,
                    norm_acf_context.is_some(),
                ));
                delay_tsv.push('\n');
            }

            if args.cumulate != 0 {
                // Use actual (unpadded) integration time for cumulation plot
                let integ_time = physical_length as f32 * effective_integ_time;
                cumulate_len.push(integ_time);
                cumulate_snr.push(analysis_results.delay_snr);
            }

            add_plot_amp.push(analysis_results.delay_max_amp * 100.0);
            add_plot_phase.push(analysis_results.delay_phase);
            add_plot_times.push(phase_obs_time);
            wwz_times_sec.push(
                phase_obs_time
                    .signed_duration_since(file_start_time)
                    .num_milliseconds() as f32
                    / 1000.0,
            );
            let phase_rad = analysis_results.delay_phase.to_radians();
            let complex_sample = Complex::from_polar(analysis_results.delay_max_amp, phase_rad);
            add_plot_complex.push(complex_sample);

            if args.add_plot || args.stfft > 0 || args.cumulate != 0 {
                add_plot_snr.push(analysis_results.delay_snr);
                add_plot_noise.push(analysis_results.delay_noise * 100.0);
                add_plot_res_delay.push(analysis_results.residual_delay);
                add_plot_res_rate.push(analysis_results.residual_rate);
            }

            if l1 == loop_count - 1 && args.output {
                if let Some(path) = &output_path {
                    let length_label = if args.length == 0 {
                        "0".to_string()
                    } else {
                        args.length.to_string()
                    };
                    let out_dir = if args.in_beam {
                        path.clone()
                    } else {
                        path.join(format!("time_domain/len{}s", length_label))
                    };
                    fs::create_dir_all(&out_dir)?;
                    let output_basename = first_output_basename.as_ref().unwrap_or(&base_filename);
                    let output_stem = insert_product_before_processing_suffixes(
                        output_basename,
                        "delay_rate_search",
                    );
                    let output_file_path = out_dir.join(format!("{output_stem}.tsv"));
                    fs::write(output_file_path, &delay_tsv)?;
                }
            }
        } else {
            let freq_output_line = format_freq_output(
                &analysis_results,
                &label_str,
                args.length,
                &rfi_display,
                bandpass_active,
                norm_acf_context.is_some(),
            );
            if l1 == 0 {
                let station1_label = format!("{}-azel", header.station1_name.trim());
                let station2_label = format!("{}-azel", header.station2_name.trim());
                let header_str = format!(
                    concat!(
                        "#*******************************************************************************************************************************************************************************************************************\n",
                        "#      Epoch        Label    Source     Length    Amp      SNR     Phase     Frequency     Noise-level      Res-Rate            {:<10}             {:<10}        MJD        RFI       BP    ACF\n",
                        "#                                        [s]      [%]              [deg]       [MHz]       1-sigma[%]        [Hz]        az[deg]  el[deg]  hgt[m]   az[deg]  el[deg]  hgt[m]             [MHz]      [T/F] [T/F]\n",
                        "#*******************************************************************************************************************************************************************************************************************"
                    ),
                    station1_label,
                    station2_label
                );
                if !suppress_output {
                    print!("{}\n", header_str);
                }
                if args.output {
                    freq_tsv.push_str(&format_freq_tsv_header(
                        &header.station1_name,
                        &header.station2_name,
                    ));
                }
            }
            if !suppress_output {
                print!("{}\n", freq_output_line);
            }
            if args.output {
                freq_tsv.push_str(&format_freq_tsv_row(
                    &analysis_results,
                    &label_str,
                    &rfi_display,
                    bandpass_active,
                    norm_acf_context.is_some(),
                ));
                freq_tsv.push('\n');
            }

            if l1 == loop_count - 1 && args.output {
                if let Some(path) = &output_path {
                    let length_label = if args.length == 0 {
                        "0".to_string()
                    } else {
                        args.length.to_string()
                    };
                    let out_dir = if args.in_beam {
                        path.clone()
                    } else {
                        path.join(format!("freq_domain/len{}s", length_label))
                    };
                    fs::create_dir_all(&out_dir)?;
                    let output_basename = first_output_basename.as_ref().unwrap_or(&base_filename);
                    let output_stem = insert_product_before_processing_suffixes(
                        output_basename,
                        "freq_rate_search",
                    );
                    let output_file_path = out_dir.join(format!("{output_stem}.tsv"));
                    fs::write(output_file_path, &freq_tsv)?;
                }
            }

            if args.cumulate != 0 {
                let integ_time = physical_length as f32 * effective_integ_time;
                cumulate_len.push(integ_time);
                cumulate_snr.push(analysis_results.freq_snr);
            }

            if args.add_plot || args.stfft > 0 || args.cumulate != 0 {
                add_plot_times.push(phase_obs_time);
                add_plot_amp.push(analysis_results.freq_max_amp * 100.0);
                add_plot_snr.push(analysis_results.freq_snr);
                add_plot_phase.push(analysis_results.freq_phase);
                add_plot_freq.push(analysis_results.freq_freq);
                add_plot_noise.push(analysis_results.freq_noise * 100.0);
                add_plot_res_delay.push(analysis_results.residual_delay);
                add_plot_res_rate.push(analysis_results.residual_rate);
            }
        }

        if args.plot && args.cumulate == 0 {
            if let Some(path) = &plot_path {
                let length_label = if args.length == 0 {
                    "0".to_string()
                } else {
                    args.length.to_string()
                };
                let plot_dir = if args.in_beam {
                    path.clone()
                } else if !args.frequency {
                    path.join(format!("time_domain/len{}s", length_label))
                } else {
                    path.join(format!("freq_domain/len{}s", length_label))
                };
                fs::create_dir_all(&plot_dir)?;
                let product = if args.frequency {
                    "freq_rate_search"
                } else {
                    "delay_rate_search"
                };
                let output_stem =
                    insert_product_before_processing_suffixes(&base_filename, product);
                let output_filename = plot_dir.join(format!("{output_stem}.png"));

                if args.npz && !args.frequency {
                    let rate_axis: Vec<f64> = analysis_results
                        .rate_range
                        .iter()
                        .map(|&v| v as f64)
                        .collect();
                    let delay_axis: Vec<f64> = analysis_results
                        .delay_range
                        .iter()
                        .map(|&v| v as f64)
                        .collect();
                    let npz_path = npz_sidecar_path(&output_filename, "plot_delay_rate");
                    let mut npz = NamedNpz::new(NpyMeta::new(
                        "plot_delay_rate",
                        effective_fft_point as u32,
                        processing_header.number_of_sector as u32,
                    ));
                    npz.add_f64_1d("rate_hz", &rate_axis);
                    npz.add_f64_1d("delay_sample", &delay_axis);
                    npz.add_complex64_2d(
                        "delay_rate",
                        delay_rate_2d_data_comp.dim(),
                        delay_rate_2d_data_comp.iter().copied(),
                    )?;
                    npz.write(&npz_path)?;
                } else if args.npz {
                    if let Some(freq_rate) = freq_rate_array.as_ref() {
                        let frequency_axis: Vec<f64> = analysis_results
                            .freq_range
                            .iter()
                            .map(|&v| v as f64)
                            .collect();
                        let rate_axis: Vec<f64> = analysis_results
                            .rate_range
                            .iter()
                            .map(|&v| v as f64)
                            .collect();
                        let npz_path = npz_sidecar_path(&output_filename, "plot_freq_rate");
                        let mut npz = NamedNpz::new(NpyMeta::new(
                            "plot_freq_rate",
                            effective_fft_point as u32,
                            processing_header.number_of_sector as u32,
                        ));
                        npz.add_f64_1d("frequency_mhz", &frequency_axis);
                        npz.add_f64_1d("rate_hz", &rate_axis);
                        npz.add_complex64_2d(
                            "freq_rate",
                            freq_rate.dim(),
                            freq_rate.iter().copied(),
                        )?;
                        npz.write(&npz_path)?;
                    }
                }

                if !args.frequency {
                    let mask_bounds = delay_rate_mask_bounds(&args.mask);
                    let delay_profile: Vec<(f64, f64)> = analysis_results
                        .delay_range
                        .iter()
                        .zip(analysis_results.visibility.iter())
                        .map(|(&x, &y)| (x as f64, y as f64))
                        .collect();
                    let rate_profile: Vec<(f64, f64)> = analysis_results
                        .rate_range
                        .iter()
                        .zip(analysis_results.delay_rate.iter())
                        .map(|(&x, &y)| (x as f64, y as f64))
                        .collect();
                    let delay_profile_pre_bp: Option<Vec<(f64, f64)>> =
                        pre_bandpass_results.as_ref().map(|pre| {
                            pre.delay_range
                                .iter()
                                .zip(pre.visibility.iter())
                                .map(|(&x, &y)| (x as f64, y as f64))
                                .collect()
                        });
                    let rate_profile_pre_bp: Option<Vec<(f64, f64)>> =
                        pre_bandpass_results.as_ref().map(|pre| {
                            pre.rate_range
                                .iter()
                                .zip(pre.delay_rate.iter())
                                .map(|(&x, &y)| (x as f64, y as f64))
                                .collect()
                        });
                    let rows = delay_rate_2d_data_comp.shape()[0] as u32;
                    let cols = delay_rate_2d_data_comp.shape()[1] as u32;
                    let delay_data: Vec<f32> = analysis_results
                        .delay_range
                        .iter()
                        .map(|&x| x as f32)
                        .collect();
                    let rate_data: Vec<f32> = analysis_results
                        .rate_range
                        .iter()
                        .map(|&x| x as f32)
                        .collect();
                    let mut plot_drange: Vec<f32> = if args.drange.len() == 2
                        && matches!(
                            primary_search_mode,
                            Some("peak") | Some("deep") | Some("deep2")
                        ) {
                        vec![
                            args.drange[0] - analysis_results.corrected_delay,
                            args.drange[1] - analysis_results.corrected_delay,
                        ]
                    } else {
                        args.drange.clone()
                    };
                    let mut plot_rrange: Vec<f32> = if args.rrange.len() == 2
                        && matches!(
                            primary_search_mode,
                            Some("peak") | Some("deep") | Some("deep2")
                        ) {
                        vec![
                            args.rrange[0] - analysis_results.corrected_rate,
                            args.rrange[1] - analysis_results.corrected_rate,
                        ]
                    } else {
                        args.rrange.clone()
                    };

                    // In in-beam mode, default to full delay-rate plane coverage when no window is specified.
                    if args.in_beam && plot_drange.len() != 2 {
                        if let (Some(&d0), Some(&d1)) = (delay_data.first(), delay_data.last()) {
                            plot_drange = vec![d0, d1];
                        }
                    }
                    if args.in_beam && plot_rrange.len() != 2 {
                        if let (Some(&r0), Some(&r1)) = (rate_data.first(), rate_data.last()) {
                            plot_rrange = vec![r0, r1];
                        }
                    }
                    let (delay_plot_min, delay_plot_max) = if plot_drange.len() == 2 {
                        (
                            plot_drange[0].min(plot_drange[1]) as f64,
                            plot_drange[0].max(plot_drange[1]) as f64,
                        )
                    } else {
                        (-10.0_f64, 10.0_f64)
                    };
                    let (rate_plot_min, rate_plot_max) = if plot_rrange.len() == 2 {
                        (
                            plot_rrange[0].min(plot_rrange[1]) as f64,
                            plot_rrange[0].max(plot_rrange[1]) as f64,
                        )
                    } else {
                        let rate_low =
                            if (-8.0 / analysis_results.length_f32 as f64) < rate_data[0] as f64 {
                                rate_data[0] as f64 * effective_integ_time as f64
                            } else {
                                -4.0 / (analysis_results.length_f32 as f64
                                    * effective_integ_time as f64)
                            };
                        let rate_high = if (8.0 / analysis_results.length_f32 as f64)
                            > *rate_data.last().unwrap_or(&rate_data[0]) as f64
                        {
                            *rate_data.last().unwrap_or(&rate_data[0]) as f64
                                * effective_integ_time as f64
                        } else {
                            4.0 / (analysis_results.length_f32 as f64 * effective_integ_time as f64)
                        };
                        (rate_low, rate_high)
                    };

                    let delay_indices: Vec<usize> = delay_data
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, &d)| {
                            let d = d as f64;
                            if d >= delay_plot_min && d <= delay_plot_max {
                                Some(idx)
                            } else {
                                None
                            }
                        })
                        .collect();
                    let rate_indices: Vec<usize> = rate_data
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, &r)| {
                            let r = r as f64;
                            if r >= rate_plot_min && r <= rate_plot_max {
                                Some(idx)
                            } else {
                                None
                            }
                        })
                        .collect();

                    let x_start = delay_indices.first().copied().unwrap_or(0);
                    let x_end = delay_indices
                        .last()
                        .copied()
                        .unwrap_or_else(|| cols.saturating_sub(1) as usize);
                    let y_start = rate_indices.first().copied().unwrap_or(0);
                    let y_end = rate_indices
                        .last()
                        .copied()
                        .unwrap_or_else(|| rows.saturating_sub(1) as usize);
                    let mut max_norm = 0.0f32;
                    for r_idx in y_start..=y_end.min(rows.saturating_sub(1) as usize) {
                        for d_idx in x_start..=x_end.min(cols.saturating_sub(1) as usize) {
                            let delay = delay_data[d_idx];
                            let rate = rate_data[r_idx];
                            if in_delay_rate_mask(delay, rate, mask_bounds) {
                                continue;
                            }
                            max_norm = max_norm.max(delay_rate_2d_data_comp[[r_idx, d_idx]].norm());
                        }
                    }

                    let (heatmap_res_x, heatmap_res_y) = if args.in_beam {
                        // In in-beam mode, draw with 3x the native array dimensions.
                        (
                            (cols as usize).saturating_mul(3).max(3),
                            (rows as usize).saturating_mul(3).max(3),
                        )
                    } else {
                        // In normal time-domain plot mode, use a fixed rendering resolution.
                        (1500, 1500)
                    };

                    let x_span = (x_end.saturating_sub(x_start)).max(1) as f64;
                    let y_span = (y_end.saturating_sub(y_start)).max(1) as f64;
                    let heatmap_func = move |delay: f64, rate: f64| -> f64 {
                        if in_delay_rate_mask(delay as f32, rate as f32, mask_bounds) {
                            return 0.0;
                        }
                        let d_min = delay_plot_min;
                        let d_max = delay_plot_max;
                        let r_min = rate_plot_min;
                        let r_max = rate_plot_max;
                        let d_den = (d_max - d_min).abs().max(1e-12);
                        let r_den = (r_max - r_min).abs().max(1e-12);
                        let x_img = ((delay - d_min) / d_den * x_span).max(0.0).min(x_span);
                        let y_img = ((rate - r_min) / r_den * y_span).max(0.0).min(y_span);
                        let x_floor = x_img.floor() as usize;
                        let y_floor = y_img.floor() as usize;
                        let x_ceil = (x_img.ceil() as usize).min(x_span as usize);
                        let y_ceil = (y_img.ceil() as usize).min(y_span as usize);
                        let fx = x_img - x_img.floor();
                        let fy = y_img - y_img.floor();
                        let gx_floor = x_start + x_floor;
                        let gy_floor = y_start + y_floor;
                        let gx_ceil = (x_start + x_ceil).min(x_end);
                        let gy_ceil = (y_start + y_ceil).min(y_end);
                        let q11 = delay_rate_2d_data_comp[[gy_floor, gx_floor]].norm() as f64;
                        let q12 = delay_rate_2d_data_comp[[gy_floor, gx_ceil]].norm() as f64;
                        let q21 = delay_rate_2d_data_comp[[gy_ceil, gx_floor]].norm() as f64;
                        let q22 = delay_rate_2d_data_comp[[gy_ceil, gx_ceil]].norm() as f64;
                        let r1 = q11 * (1.0 - fx) + q12 * fx;
                        let r2 = q21 * (1.0 - fx) + q22 * fx;
                        r1 * (1.0 - fy) + r2 * fy
                    };

                    let (length_key, length_val) = if !pp_flag_ranges.is_empty() {
                        let flag_str = pp_flag_ranges
                            .iter()
                            .map(|(s, e)| format!("{}-{}", s, e))
                            .collect::<Vec<String>>()
                            .join(", ");
                        (
                            "Length (flag) [s]".to_string(),
                            format!("{:.3} ({})", analysis_results.length_f32.ceil(), flag_str),
                        )
                    } else {
                        (
                            "Length [s]".to_string(),
                            format!("{:.3}", analysis_results.length_f32.ceil()),
                        )
                    };

                    let stat_keys = vec![
                        "Epoch (UTC)",
                        "Station 1 & 2",
                        "Source",
                        &length_key,
                        "Frequency [MHz]",
                        "Peak Amp [%]",
                        "Peak Phs [deg]",
                        "SNR (1 σ [%])",
                        "Delay (residual) [sps]",
                        "Delay (corrected) [sps]",
                        "Rate (residual) [mHz]",
                        "Rate (corrected) [mHz]",
                    ];
                    let stat_vals = vec![
                        analysis_results.yyyydddhhmmss1.to_string(),
                        format!("{} & {}", header.station1_name, header.station2_name),
                        analysis_results.source_name.to_string(),
                        length_val,
                        format!("{:.3}", header.observing_frequency as f32 / 1e6),
                        format!("{:.6}", analysis_results.delay_max_amp * 100.0),
                        format!("{:+.5}", analysis_results.delay_phase),
                        format!(
                            "{:.3} ({:.6})",
                            analysis_results.delay_snr,
                            analysis_results.delay_noise * 100.0
                        ),
                        format!("{:+.6}", analysis_results.residual_delay),
                        format!("{:+.6}", analysis_results.corrected_delay),
                        format!("{:+.6}", analysis_results.residual_rate * 1000.0),
                        format!("{:+.6}", analysis_results.corrected_rate * 1000.0),
                    ];
                    delay_plane(
                        &delay_profile,
                        delay_profile_pre_bp.as_deref(),
                        &rate_profile,
                        rate_profile_pre_bp.as_deref(),
                        heatmap_func,
                        &stat_keys.iter().map(|s| s.as_ref()).collect::<Vec<&str>>(),
                        &stat_vals.iter().map(|s| s.as_ref()).collect::<Vec<&str>>(),
                        output_filename.to_str().unwrap(),
                        &analysis_results.rate_range,
                        analysis_results.length_f32,
                        effective_integ_time,
                        &plot_drange,
                        &plot_rrange,
                        mask_bounds,
                        max_norm as f64,
                        heatmap_res_x,
                        heatmap_res_y,
                        args.in_beam,
                    )?;
                } else {
                    let freq_rate_array = freq_rate_array.as_ref().ok_or(
                        "--frequency が指定されているのに freq_rate_array が保持されていません",
                    )?;
                    let freq_amp_profile: Vec<(f64, f64)> = analysis_results
                        .freq_range
                        .iter()
                        .zip(analysis_results.freq_rate_spectrum.iter().map(|c| c.norm()))
                        .map(|(&x, y)| (x as f64, y as f64))
                        .collect();
                    let freq_phase_profile: Vec<(f64, f64)> = analysis_results
                        .freq_range
                        .iter()
                        .zip(
                            analysis_results
                                .freq_rate_spectrum
                                .iter()
                                .map(|c| safe_arg(c).to_degrees()),
                        )
                        .map(|(&x, y)| (x as f64, y as f64))
                        .collect();
                    let freq_phase_profile_pre_bp: Option<Vec<(f64, f64)>> =
                        pre_bandpass_results.as_ref().map(|pre| {
                            pre.freq_range
                                .iter()
                                .zip(
                                    pre.freq_rate_spectrum
                                        .iter()
                                        .map(|c| safe_arg(c).to_degrees()),
                                )
                                .map(|(&x, y)| (x as f64, y as f64))
                                .collect()
                        });
                    let rate_profile: Vec<(f64, f64)> = analysis_results
                        .rate_range
                        .iter()
                        .zip(analysis_results.freq_rate.iter())
                        .map(|(&x, &y)| (x as f64, y as f64))
                        .collect();
                    let freq_amp_profile_pre_bp: Option<Vec<(f64, f64)>> =
                        pre_bandpass_results.as_ref().map(|pre| {
                            pre.freq_range
                                .iter()
                                .zip(pre.freq_rate_spectrum.iter().map(|c| c.norm()))
                                .map(|(&x, y)| (x as f64, y as f64))
                                .collect()
                        });
                    let rate_profile_pre_bp: Option<Vec<(f64, f64)>> =
                        pre_bandpass_results.as_ref().map(|pre| {
                            pre.rate_range
                                .iter()
                                .zip(pre.freq_rate.iter())
                                .map(|(&x, &y)| (x as f64, y as f64))
                                .collect()
                        });
                    let freq_data: Vec<f32> = analysis_results
                        .freq_range
                        .iter()
                        .map(|&x| x as f32)
                        .collect();
                    let rate_data: Vec<f32> = analysis_results
                        .rate_range
                        .iter()
                        .map(|&x| x as f32)
                        .collect();
                    let heatmap_func = |freq: f64, rate: f64| -> f64 {
                        let f_min = freq_data[0] as f64;
                        let f_max = *freq_data.last().unwrap() as f64;
                        let r_min = rate_data[0] as f64;
                        let r_max = *rate_data.last().unwrap() as f64;
                        if freq < f_min || freq > f_max || rate < r_min || rate > r_max {
                            return 0.0;
                        }
                        let rows = freq_rate_array.shape()[0];
                        let cols = freq_rate_array.shape()[1];
                        let freq_idx = (((freq - f_min) / (f_max - f_min)) * (rows - 1) as f64)
                            .round() as usize;
                        let rate_idx = (((rate - r_min) / (r_max - r_min)) * (cols - 1) as f64)
                            .round() as usize;
                        if freq_idx < rows && rate_idx < cols {
                            freq_rate_array[[freq_idx, rate_idx]].norm() as f64
                        } else {
                            0.0
                        }
                    };

                    let (length_key, length_val) = if !pp_flag_ranges.is_empty() {
                        let flag_str = pp_flag_ranges
                            .iter()
                            .map(|(s, e)| format!("{}-{}", s, e))
                            .collect::<Vec<String>>()
                            .join(", ");
                        (
                            "Length (flag) [s]".to_string(),
                            format!("{:.3} ({})", analysis_results.length_f32.ceil(), flag_str),
                        )
                    } else {
                        (
                            "Length [s]".to_string(),
                            format!("{:.3}", analysis_results.length_f32.ceil()),
                        )
                    };

                    let (freq_key, freq_val) = if !args.rfi.is_empty() {
                        let rfi_str = args
                            .rfi
                            .iter()
                            .map(|s| s.replace(',', "-"))
                            .collect::<Vec<String>>()
                            .join(", ");
                        (
                            "Frequency (RFI) [MHz]".to_string(),
                            format!(
                                "{:.3} ({})",
                                header.observing_frequency as f32 / 1e6,
                                rfi_str
                            ),
                        )
                    } else {
                        (
                            "Frequency [MHz]".to_string(),
                            format!("{:.3}", header.observing_frequency as f32 / 1e6),
                        )
                    };

                    let mut stat_keys = vec![
                        "Epoch (UTC)",
                        "Station 1 & 2",
                        "Source",
                        &length_key,
                        &freq_key,
                        "Peak Amp [%]",
                        "Peak Phs [deg]",
                        "Peak Freq [MHz]",
                        "SNR (1 σ [%])",
                        "Rate (residual) [mHz]",
                    ];
                    let mut stat_vals = vec![
                        analysis_results.yyyydddhhmmss1.to_string(),
                        format!("{} & {}", header.station1_name, header.station2_name),
                        analysis_results.source_name.to_string(),
                        length_val,
                        freq_val,
                        format!("{:.6}", analysis_results.freq_max_amp * 100.0),
                        format!("{:+.5}", analysis_results.freq_phase),
                        format!("{:+.6}", analysis_results.freq_max_freq),
                        format!(
                            "{:.3} ({:.6})",
                            analysis_results.freq_snr,
                            analysis_results.freq_noise * 100.0
                        ),
                        format!("{:+.6}", analysis_results.residual_rate * 1000.0),
                    ];
                    if args.spike34m.is_some() {
                        stat_keys.push("Spike34 delay applied [sample]");
                        stat_vals.push(format!("{:+.6}", spike34_applied_delay));
                        stat_keys.push("Spike34 rate applied [mHz]");
                        stat_vals.push(format!("{:+.6}", spike34_applied_rate * 1000.0));
                        stat_keys.push("Spike34 interval residual correction");
                        stat_vals.push("applied".to_string());
                    }
                    let max_norm_freq = freq_rate_array
                        .iter()
                        .map(|c| c.norm())
                        .fold(0.0f32, |acc, x| acc.max(x));
                    frequency_plane(
                        &freq_amp_profile,
                        freq_amp_profile_pre_bp.as_deref(),
                        &freq_phase_profile,
                        freq_phase_profile_pre_bp.as_deref(),
                        &rate_profile,
                        rate_profile_pre_bp.as_deref(),
                        heatmap_func,
                        &stat_keys.iter().map(|s| s.as_ref()).collect::<Vec<&str>>(),
                        &stat_vals.iter().map(|s| s.as_ref()).collect::<Vec<&str>>(),
                        output_filename.to_str().unwrap(),
                        bw as f64,
                        max_norm_freq as f64,
                        &args.frange,
                        freq_rate_array.shape()[0],
                        freq_rate_array.shape()[1],
                    )?;
                }
            }
        }
    }

    Ok(ProcessResult {
        header: processing_header,
        label,
        obs_time: file_start_time,
        length_arg: length,
        length_sec: length as f32 * effective_integ_time,
        wwz_times_sec,
        cumulate_len,
        cumulate_snr,
        add_plot_times,
        add_plot_amp,
        add_plot_snr,
        add_plot_phase,
        add_plot_freq,
        add_plot_noise,
        add_plot_res_delay,
        add_plot_res_rate,
        add_plot_complex,
    })
}

pub(crate) fn run_analysis_pipeline(
    complex_vec: &[C32],
    header: &CorHeader,
    base_args: &Args,
    search_mode: Option<&str>,
    delay_correct: f32,
    rate_correct: f32,
    acel_correct: f32,
    current_length: i32,
    physical_length: i32,
    effective_integ_time: f32,
    current_obs_time: &DateTime<Utc>,
    file_start_time: &DateTime<Utc>,
    rfi_ranges: &[(usize, usize)],
    bandpass_data: &Option<Vec<C32>>,
    keep_pre_bandpass_results: bool,
    effective_fft_point: i32,
) -> Result<
    (
        AnalysisResults,
        Option<Array2<C32>>,
        Array2<C32>,
        Option<AnalysisResults>,
    ),
    Box<dyn Error>,
> {
    let mut temp_args = base_args.clone();
    temp_args.delay_correct = delay_correct;
    temp_args.rate_correct = rate_correct;
    temp_args.acel_correct = acel_correct;
    temp_args.jerk_correct = base_args.jerk_correct;
    temp_args.snap_correct = base_args.snap_correct;
    temp_args.search = search_mode
        .map(|mode| vec![mode.to_string()])
        .unwrap_or_default();

    // In iterative peak search, treat user-specified windows as absolute target ranges.
    // Convert them into residual windows for the current correction state.
    if search_mode == Some("peak") {
        if temp_args.drange.len() == 2 {
            temp_args.drange[0] -= delay_correct;
            temp_args.drange[1] -= delay_correct;
        }
        if temp_args.rrange.len() == 2 {
            temp_args.rrange[0] -= rate_correct;
            temp_args.rrange[1] -= rate_correct;
        }
    }

    let mut effective_fft_point = effective_fft_point;
    if effective_fft_point <= 0 {
        if current_length <= 0 {
            return Err("セクター長が 0 以下です".into());
        }
        let rows = current_length as usize;
        if rows == 0 || complex_vec.len() % rows != 0 {
            return Err(format!(
                "複素データ長 ({}) がセクター数 ({}) の整数倍ではありません。",
                complex_vec.len(),
                rows
            )
            .into());
        }
        let fft_half = complex_vec.len() / rows;
        effective_fft_point = (fft_half * 2) as i32;
    }

    let fft_point_half = (effective_fft_point / 2) as usize;
    if fft_point_half == 0 {
        return Err("effective FFT point が不正（0）です".into());
    }
    if complex_vec.len() % fft_point_half != 0 {
        return Err(format!(
            "複素データ長 ({}) が FFT チャンネル数 ({}) の整数倍ではありません。",
            complex_vec.len(),
            fft_point_half
        )
        .into());
    }

    if current_length > 0 && complex_vec.len() / fft_point_half != current_length as usize {
        return Err(format!(
            "与えられたセクター数 ({}) とデータから導かれる値 ({}) が一致しません。",
            current_length,
            complex_vec.len() / fft_point_half
        )
        .into());
    }

    // A searched residual rate is local to this analyzed segment. Reference
    // it to the first sample so the reported complex fringe phase and MJD/UVW
    // share the same epoch. Manual/static corrections retain their file-start
    // reference below.
    let use_window_start_phase_reference =
        search_mode == Some("peak") || base_args.primary_search_mode() == Some("peak");
    let start_time_offset_sec = if use_window_start_phase_reference {
        0.0
    } else {
        current_obs_time
            .signed_duration_since(*file_start_time)
            .num_seconds() as f32
    };

    let (mut freq_rate_array, padding_length) = if delay_correct != 0.0
        || rate_correct != 0.0
        || acel_correct != 0.0
        || base_args.jerk_correct != 0.0
        || base_args.snap_correct != 0.0
    {
        process_fft_with_phase_correction_at_frequency(
            complex_vec,
            physical_length,
            effective_fft_point,
            header.sampling_speed,
            rfi_ranges,
            base_args.rate_padding,
            rate_correct,
            delay_correct,
            acel_correct,
            base_args.jerk_correct,
            base_args.snap_correct,
            effective_integ_time,
            start_time_offset_sec,
            header.observing_frequency,
        )
    } else {
        process_fft(
            complex_vec,
            physical_length,
            effective_fft_point,
            header.sampling_speed,
            rfi_ranges,
            base_args.rate_padding,
        )
    };

    let skip_delay_rate_ifft = base_args.frequency && base_args.drange.is_empty();

    let pre_bandpass_analysis_results = if keep_pre_bandpass_results && bandpass_data.is_some() {
        let pre_bandpass_delay_rate_2d_data_comp = if skip_delay_rate_ifft {
            Array2::zeros((1, 1))
        } else {
            process_ifft(&freq_rate_array, effective_fft_point, padding_length)
        };
        Some(analyze_results(
            &freq_rate_array,
            &pre_bandpass_delay_rate_2d_data_comp,
            &header,
            current_length,
            effective_integ_time,
            &current_obs_time,
            padding_length,
            &temp_args,
            search_mode,
        ))
    } else {
        None
    };

    if let Some(bp_data) = &bandpass_data {
        apply_bandpass_correction(&mut freq_rate_array, bp_data);
    }

    let delay_rate_2d_data_comp = if skip_delay_rate_ifft {
        Array2::zeros((1, 1))
    } else {
        process_ifft(&freq_rate_array, effective_fft_point, padding_length)
    };

    let analysis_results = analyze_results(
        &freq_rate_array,
        &delay_rate_2d_data_comp,
        &header,
        current_length,
        effective_integ_time,
        &current_obs_time,
        padding_length,
        &temp_args,
        search_mode,
    );

    Ok((
        analysis_results,
        base_args.frequency.then_some(freq_rate_array),
        delay_rate_2d_data_comp,
        pre_bandpass_analysis_results,
    ))
}

#[cfg(test)]
mod output_directory_tests {
    use super::frinz_output_dir;
    use std::path::{Path, PathBuf};

    #[test]
    fn all_standard_results_share_the_input_parent_frinz_directory() {
        let input = Path::new("/data/session/source.cor");
        assert_eq!(
            frinz_output_dir(input, false),
            PathBuf::from("/data/session/frinZ")
        );
    }

    #[test]
    fn inbeam_is_nested_below_the_shared_frinz_directory() {
        let input = Path::new("/data/session/source.cor");
        assert_eq!(
            frinz_output_dir(input, true),
            PathBuf::from("/data/session/frinZ/inbeamVLBI")
        );
    }
}
