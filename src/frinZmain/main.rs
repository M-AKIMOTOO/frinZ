#![allow(unused_imports)]
use byteorder::{LittleEndian, WriteBytesExt};
use std::error::Error;
use std::fs;
use std::io::{self, Cursor, Read, Write};
use std::process::exit;

use chrono::{DateTime, Utc};
use clap::{parser::ValueSource, CommandFactory, FromArgMatches, Parser};
use num_complex::Complex;
use std::f64::consts::PI;
use std::path::{Path, PathBuf};

mod analysis;
mod args;
mod bandpass;
mod bispectrum;
mod search;
//mod error;
mod fft;
mod fitting;
mod frmap;
mod header;
mod folding;
#[path = "inbeamVLBI.rs"]
mod inbeam_vlbi;

mod earth_rotation_imaging;
mod logo;
mod maser;
mod multisideband;
mod output;
mod phsref;
mod plot;
mod png_compress;
mod processing;
mod raw_visibility;
mod read;
mod rfi;
mod uptimeplot;
mod uv;
mod utils;

use crate::args::{check_memory_usage, Args};
use crate::bispectrum::run_closure_phase_analysis;
use crate::earth_rotation_imaging::{
    parse_imaging_cli_options, perform_imaging, run_earth_rotation_imaging, Visibility,
};
use crate::frmap::run_fringe_rate_map_analysis;
use crate::folding::run_folding_analysis;
use crate::inbeam_vlbi::run_inbeam_vlbi_analysis;
use crate::maser::run_maser_analysis;
use crate::multisideband::run_multisideband_analysis;
use crate::phsref::run_phase_reference_analysis;
use crate::plot::{write_add_plot_outputs, write_cumulate_outputs};
use crate::processing::process_cor_file;
use crate::raw_visibility::run_raw_visibility_plot;
use crate::search::run_acel_search_analysis;
use crate::uptimeplot::run_uptime_plot;
use crate::uv::run_uv_plot;

// --- Type Aliases for Clarity ---
pub type C32 = Complex<f32>;

// --- Main Application Logic ---
fn main() -> Result<(), Box<dyn Error>> {
    let env_args: Vec<String> = std::env::args().collect();

    // Show logo if help is requested or no arguments are provided.
    if env_args.len() == 1 || env_args.iter().any(|arg| arg == "-h" || arg == "--help") {
        if let Err(e) = logo::show_logo() {
            // Log the error but continue execution, as the logo is not critical.
            eprintln!("Warning: Failed to display logo: {}", e);
        }
    }

    let command = Args::command();
    let mut matches = match command.try_get_matches_from(env_args.clone()) {
        Ok(m) => m,
        Err(e) => {
            if env_args.len() <= 1 {
                exit(0);
            } else {
                e.exit();
            }
        }
    };
    let rate_padding_explicit =
        matches.value_source("rate_padding") == Some(ValueSource::CommandLine);
    let iter_explicit = matches.value_source("iter") == Some(ValueSource::CommandLine);

    let mut args = match Args::from_arg_matches_mut(&mut matches) {
        Ok(args) => args,
        Err(e) => e.exit(),
    };

    if args.detail {
        const DETAIL_TEXT: &str = include_str!("command_detail.txt");
        print!("{}", DETAIL_TEXT);
        return Ok(());
    }

    if args.scan_correct.is_some() {
        if !args.search.is_empty() {
            eprintln!("Error: --scan-correct cannot be used with --search.");
            exit(1);
        }
    }

    // シンプルな仕様: --cumulate が指定されたら rate_padding は常に 1 にする
    if args.cumulate != 0 {
        args.rate_padding = 1;
    } else if args.primary_search_mode() == Some("deep") && !rate_padding_explicit {
        if args.rate_padding != 4 {
            println!(
                "#INFO: --search deep が指定されたため rate-padding を 4 に設定します (旧値 {}).",
                args.rate_padding
            );
        }
        args.rate_padding = 4;
    }
    if args.primary_search_mode() == Some("deep") && !iter_explicit {
        args.iter = 4;
    }

    if !args.rate_padding.is_power_of_two() {
        eprintln!("Error: --rate-padding must be a power of two.");
        exit(1);
    }
    if !matches!(args.rate_padding, 1 | 2 | 4 | 8 | 16) {
        eprintln!("Error: --rate-padding must be one of 1, 2, 4, 8, or 16.");
        exit(1);
    }

    if args.imaging_test {
        println!("Running Earth-rotation synthesis imaging test...");

        // --- Create Sample Visibility Data ---
        // Simulate a point source at (l, m) = (5.0e-5, 0.0) radians
        // V(u,v) = 1.0 * exp(-2*pi*i * (u*l + v*m))
        // Since m=0, V(u,v) = cos(2*pi*u*l) - i*sin(2*pi*u*l)
        let mut vis_data: Vec<Visibility> = Vec::new();
        let l0 = 5.0e-5; // offset in l-direction
        let num_points = 100;
        let max_uv = 5000.0; // 5k wavelengths

        for i in 0..num_points {
            let angle = (i as f64 / num_points as f64) * 2.0 * PI; // Simulate Earth rotation
            let u = max_uv * angle.cos();
            let v = max_uv * angle.sin();

            let phase = -2.0 * PI * u * l0;
            let real = phase.cos();
            let imag = phase.sin();

            vis_data.push(Visibility {
                u,
                v,
                w: 0.0,
                real,
                imag,
                weight: 1.0,
                time: i as f64,
            });
        }

        // --- Set Imaging Parameters ---
        let image_size = 256; // 256x256 pixels
        let cell_size = 0.1; // 0.1 arcsec/pixel

        // --- Perform Imaging ---
        match perform_imaging(&vis_data, image_size, cell_size) {
            Ok(dirty_image) => {
                println!(
                    "Imaging successful! Dirty image size: {}x{}",
                    image_size, image_size
                );
                // For verification, let's print a small center patch of the image
                println!("Center 5x5 patch of the dirty image (real part):");
                let center = image_size / 2;
                for r in (center - 2)..=(center + 2) {
                    for c in (center - 2)..=(center + 2) {
                        let index = r * image_size + c;
                        print!("{:8.4} ", dirty_image[index]);
                    }
                    println!();
                }
            }
            Err(e) => {
                eprintln!("Error during imaging: {}", e);
                exit(1);
            }
        }
        return Ok(());
    }

    if args.cor2bin {
        if args.input.is_none() {
            eprintln!("Error: --cor2bin requires an --input file.");
            exit(1);
        }
        let input_path = args.input.as_ref().unwrap();

        // --- Create Output Directory ---
        let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
        let output_dir = parent_dir.join("frinZ").join("cor2bin");
        if let Err(e) = fs::create_dir_all(&output_dir) {
            eprintln!("Error creating output directory {:?}: {}", output_dir, e);
            exit(1);
        }
        let base_filename = input_path.file_stem().unwrap().to_str().unwrap();

        let mut file = match fs::File::open(input_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error opening input file {:?}: {}", input_path, e);
                exit(1);
            }
        };
        let mut buffer = Vec::new();
        if let Err(e) = file.read_to_end(&mut buffer) {
            eprintln!("Error reading input file {:?}: {}", input_path, e);
            exit(1);
        }
        let mut cursor = Cursor::new(buffer.as_slice());

        let header = match crate::header::parse_header(&mut cursor) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("Error parsing header: {}", e);
                exit(1);
            }
        };

        let mut all_spectra: Vec<Vec<C32>> = Vec::new();
        for l1 in 0..header.number_of_sector {
            let (complex_vec, _, _) = match crate::read::read_visibility_data(
                &mut cursor,
                &header,
                1,  // length in sectors
                0,  // skip in sectors
                l1, // loop_idx, which acts as sector index here
                false,
                &[], // pp_flag_ranges
            ) {
                Ok(data) => data,
                Err(_) => {
                    eprintln!("Warning: Could not read sector {}, stopping read.", l1);
                    break;
                }
            };
            if complex_vec.is_empty() {
                eprintln!("Warning: Empty sector {} found, stopping read.", l1);
                break;
            }
            all_spectra.push(complex_vec);
        }

        if all_spectra.is_empty() {
            eprintln!("No visibility data found in the file.");
            exit(1);
        }

        let flattened_spectra: Vec<C32> = all_spectra.iter().flatten().cloned().collect();
        let output_file_path = output_dir.join(format!("{}.cor.bin", base_filename));

        let mut output_file = match fs::File::create(&output_file_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error creating output file {:?}: {}", output_file_path, e);
                exit(1);
            }
        };

        if let Err(e) = output_file.write_f32::<LittleEndian>(header.fft_point as f32) {
            eprintln!("Error writing fft_point to file: {}", e);
            exit(1);
        }
        if let Err(e) = output_file.write_f32::<LittleEndian>(header.number_of_sector as f32) {
            eprintln!("Error writing number_of_sector to file: {}", e);
            exit(1);
        }

        for val in &flattened_spectra {
            if let Err(e) = output_file.write_f32::<LittleEndian>(val.re) {
                eprintln!("Error writing real part to file: {}", e);
                exit(1);
            }
            if let Err(e) = output_file.write_f32::<LittleEndian>(val.im) {
                eprintln!("Error writing imaginary part to file: {}", e);
                exit(1);
            }
        }
        println!(
            "Raw complex visibility data written to {:?}.",
            output_file_path
        );
        println!("このバイナリファイルは以下のフォーマットで構成されています:");
        println!(
            "- 先頭 4 byte: FFT点数 (f32, little-endian) = {}",
            header.fft_point
        );
        println!(
            "- 次の 4 byte: セクター数(pp) (f32, little-endian) = {}",
            header.number_of_sector
        );
        println!("- それ以降: 複素スペクトルデータ (f32 real, f32 imag の繰り返し)");
        return Ok(());
    }

    if let Some(uv_mode) = args.uv {
        if args.input.is_none() {
            eprintln!("Error: --uv requires an --input file.");
            exit(1);
        }
        if let Err(e) = run_uv_plot(&args, uv_mode) {
            eprintln!("Error during UV plotting: {}", e);
            exit(1);
        }
        return Ok(());
    }

    if args.raw_visibility {
        if args.input.is_none() {
            eprintln!("Error: --raw-visibility requires an --input file.");
            exit(1);
        }
        if let Err(e) = run_raw_visibility_plot(&args) {
            eprintln!("Error during raw visibility plotting: {}", e);
            exit(1);
        }
        return Ok(());
    }

    if args.uptimeplot {
        if args.input.is_none() {
            eprintln!("Error: --uptimeplot requires an --input file.");
            exit(1);
        }
        if let Err(e) = run_uptime_plot(&args) {
            eprintln!("Error during uptime plotting: {}", e);
            exit(1);
        }
        if args.maser.is_empty() {
            return Ok(());
        }
    }

    if !args.maser.is_empty() {
        if !args.folding.is_empty() {
            eprintln!("Error: --maser and --folding cannot be used at the same time.");
            exit(1);
        }
        if args.input.is_none() {
            eprintln!("Error: --maser requires an --input file for on-source data.");
            exit(1);
        }
        return run_maser_analysis(&args);
    }

    let mut time_flag_ranges: Vec<(DateTime<Utc>, DateTime<Utc>)> = Vec::new();
    let mut pp_flag_ranges: Vec<(u32, u32)> = Vec::new();

    if !args.flagging.is_empty() {
        let mode = &args.flagging[0];
        let params = &args.flagging[1..];

        match mode.as_str() {
            "time" => {
                if params.len() % 2 != 0 {
                    eprintln!("Error: --flagging time requires pairs of start and end times.");
                    exit(1);
                }
                time_flag_ranges = params
                    .chunks_exact(2)
                    .filter_map(|chunk| {
                        let start = utils::parse_flag_time(&chunk[0]);
                        let end = utils::parse_flag_time(&chunk[1]);
                        match (start, end) {
                            (Some(s), Some(e)) => {
                                if s >= e {
                                    eprintln!(
                                        "Error: Start time ({}) must be before end time ({}) for --flagging time.",
                                        chunk[0], chunk[1]
                                    );
                                    exit(1);
                                }
                                Some((s, e))
                            }
                            _ => {
                                eprintln!(
                                    "Error: Invalid time format in --flagging time: '{}, {}'. Expected YYYYDDDHHMMSS.",
                                    chunk[0], chunk[1]
                                );
                                exit(1);
                            }
                        }
                    })
                    .collect();
            }
            "pp" => {
                if params.len() % 2 != 0 {
                    eprintln!(
                        "Error: --flagging pp requires pairs of start and end sector numbers."
                    );
                    exit(1);
                }
                pp_flag_ranges = params
                    .chunks_exact(2)
                    .filter_map(|chunk| {
                        let start_res = chunk[0].parse::<u32>();
                        let end_res = chunk[1].parse::<u32>();
                        match (start_res, end_res) {
                            (Ok(s), Ok(e)) => {
                                if s > e {
                                    eprintln!(
                                        "Error: Start pp ({}) must not be greater than end pp ({}) for --flagging pp.",
                                        s, e
                                    );
                                    exit(1);
                                }
                                Some((s, e))
                            }
                            _ => {
                                eprintln!(
                                    "Error: Invalid sector number in --flagging pp: '{}, {}'. Expected positive integers.",
                                    chunk[0], chunk[1]
                                );
                                exit(1);
                            }
                        }
                    })
                    .collect();
            }
            _ => {
                eprintln!("Error: Invalid mode for --flagging. Use 'time' or 'pp'.");
                exit(1);
            }
        }
    }

    if !args.folding.is_empty() {
        if args.input.is_none() {
            eprintln!("Error: --folding requires an --input file.");
            exit(1);
        }
        if let Some(input_path) = &args.input {
            if !check_memory_usage(&args, input_path)? {
                exit(0);
            }
        }
        return run_folding_analysis(&args, &time_flag_ranges, &pp_flag_ranges);
    }

    if let Some(cp_tokens) = &args.closure_phase {
        if cp_tokens.len() < 3 {
            eprintln!("Error: --closure-phase requires at least three .cor files.");
            exit(1);
        }
        let mut paths = Vec::new();
        let mut refant_name = String::from("YAMAGU32");
        for token in cp_tokens {
            if token.starts_with("refant:") {
                refant_name = token["refant:".len()..].trim().to_string();
            } else if paths.len() < 3 {
                paths.push(PathBuf::from(token));
            } else {
                eprintln!("Error: Unknown --closure-phase option '{}'.", token);
                exit(1);
            }
        }
        if paths.len() != 3 {
            eprintln!("Error: --closure-phase requires exactly three .cor files.");
            exit(1);
        }
        for path in &paths {
            if !check_memory_usage(&args, path)? {
                exit(0);
            }
        }
        run_closure_phase_analysis(
            &args,
            &paths,
            &time_flag_ranges,
            &pp_flag_ranges,
            &refant_name,
        )?;
        return Ok(());
    }

    if let Some(imaging_tokens) = args.imaging.as_ref() {
        if args.input.is_none() {
            eprintln!("Error: --imaging requires an --input file.");
            exit(1);
        }
        let imaging_cli = match parse_imaging_cli_options(imaging_tokens) {
            Ok(cfg) => cfg,
            Err(err) => {
                eprintln!("Error parsing --imaging option: {}", err);
                exit(1);
            }
        };
        if let Err(e) =
            run_earth_rotation_imaging(&args, &imaging_cli, &time_flag_ranges, &pp_flag_ranges)
        {
            eprintln!("Error during Earth-rotation imaging: {}", e);
            exit(1);
        }
        return Ok(());
    }

    if let Some(_) = args.fringe_rate_map {
        if let Some(input_path) = &args.input {
            if !check_memory_usage(&args, input_path)? {
                exit(0);
            }
        }
        if args.input.is_none() {
            eprintln!("Error: --fringe-rate-map requires an --input file.");
            exit(1);
        }
        return run_fringe_rate_map_analysis(&args, &time_flag_ranges, &pp_flag_ranges);
    }

    if !args.multi_sideband.is_empty() {
        let c_band_path = std::path::PathBuf::from(&args.multi_sideband[0]);
        if !check_memory_usage(&args, &c_band_path)? {
            exit(0);
        }
        return run_multisideband_analysis(&args);
    }

    // --- Argument Validation & Dispatch ---
    let has_rate_search = args.search.iter().any(|mode| mode == "rate");
    let has_acel_search = args.search.iter().any(|mode| mode == "acel");
    let acel_only = !args.search.is_empty()
        && args
            .search
            .iter()
            .all(|mode| mode == "acel" || mode == "rate");

    if acel_only {
        if let Some(input_path) = &args.input {
            if !check_memory_usage(&args, input_path)? {
                exit(0);
            }
        }
        if args.input.is_none() {
            eprintln!("Error: --search with only 'acel'/'rate' requires an --input file.");
            exit(1);
        }
        if args.length == 0 {
            eprintln!(
                "Warning: --search=acel/rate is used without --length. This is required for the analysis."
            );
            exit(1);
        }
        if args.loop_ == 1 {
            eprintln!(
                "Warning: --search=acel/rate is used, but --loop is not specified or is 1. Multiple loops are usually needed for fitting."
            );
        }
        let mut degrees = Vec::new();
        if has_rate_search {
            degrees.push(1);
        }
        if has_acel_search {
            degrees.push(2);
        }
        return run_acel_search_analysis(&args, &degrees, &time_flag_ranges, &pp_flag_ranges);
    }

    if args.input.is_some() && (!args.phase_reference.is_empty() || args.closure_phase.is_some()) {
        eprintln!("Error: --input cannot be combined with --phase-reference or --closure-phase.");
        exit(1);
    }

    if !args.phase_reference.is_empty() && args.closure_phase.is_some() {
        eprintln!("Error: --phase-reference and --closure-phase cannot be used at the same time.");
        exit(1);
    }

    if !args.phase_reference.is_empty() {
        let cal_path = std::path::PathBuf::from(&args.phase_reference[0]);
        if !check_memory_usage(&args, &cal_path)? {
            exit(0);
        }
        let target_path = std::path::PathBuf::from(&args.phase_reference[1]);
        if !check_memory_usage(&args, &target_path)? {
            exit(0);
        }
        return run_phase_reference_analysis(&args, &time_flag_ranges, &pp_flag_ranges);
    }

    if args.in_beam {
        if let Some(input_path) = &args.input {
            if !check_memory_usage(&args, input_path)? {
                exit(0);
            }
        }
        return run_inbeam_vlbi_analysis(&args, &time_flag_ranges, &pp_flag_ranges);
    }

    if let Some(input_path) = &args.input {
        if !check_memory_usage(&args, input_path)? {
            exit(0);
        }
        let result = process_cor_file(input_path, &args, &time_flag_ranges, &pp_flag_ranges, false)?;
        let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
        let frinz_dir = parent_dir.join("frinZ");
        write_cumulate_outputs(&args, &result, &frinz_dir)?;
        let base_filename = write_add_plot_outputs(&args, &result, &frinz_dir)?;
        if args.allan_deviance {
            utils::write_allan_deviation_outputs(
                &result.add_plot_phase,
                args.length as f32,
                result.header.observing_frequency,
                &result.header.source_name,
                &base_filename,
                &frinz_dir,
            )?;
        }

        if (has_acel_search || has_rate_search) && !acel_only {
            if args.length == 0 {
                eprintln!(
                    "Warning: --search includes 'acel'/'rate' but --length is not specified. Skipping acceleration search."
                );
            } else {
                if args.loop_ == 1 {
                    eprintln!(
                        "Warning: --search includes 'acel'/'rate' but --loop is 1. Results may be unreliable."
                    );
                }
                let mut degrees = Vec::new();
                if has_rate_search {
                    degrees.push(1);
                }
                if has_acel_search {
                    degrees.push(2);
                }
                run_acel_search_analysis(&args, &degrees, &time_flag_ranges, &pp_flag_ranges)?;
            }
        }
        return Ok(());
    }

    // If we reach here, no primary mode was selected.
    eprintln!("Error: Either --input or --phase-reference must be provided.");
    let mut cmd = Args::command();
    cmd.print_help().expect("Failed to print help");
    exit(1);
}
