use std::error::Error;
use std::fs::{self, File};
use std::io::{self, Cursor, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use num_complex::Complex;

use crate::analysis::analyze_results;
use crate::args::Args;
use crate::bandpass::{apply_bandpass_correction, read_bandpass_file};
use crate::fft::{apply_phase_correction, process_fft, process_ifft};
use crate::header::{parse_header, CorHeader};
use crate::plot::frequency_plane_msb;
use crate::read::{read_sector_header, read_visibility_data};
use crate::rfi::parse_rfi_ranges;
use crate::utils::{rate_cal, safe_arg, unwrap_phase};

type C32 = Complex<f32>;

// Define TeeWriter struct and its implementation
struct TeeWriter<W1: Write, W2: Write> {
    writer1: W1,
    writer2: W2,
}

impl<W1: Write, W2: Write> Write for TeeWriter<W1, W2> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let bytes_written1 = self.writer1.write(buf)?;
        let bytes_written2 = self.writer2.write(buf)?;
        if bytes_written1 != bytes_written2 {
            return Err(io::Error::new(
                ErrorKind::Other,
                "Partial write to one of the writers",
            ));
        }
        Ok(bytes_written1)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer1.flush()?;
        self.writer2.flush()?;
        Ok(())
    }
}

pub fn run_multisideband_analysis(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.multi_sideband.len() != 6 {
        writeln!(io::stderr(), "Expected 6 arguments for --multisideband: c_binary c_bp_file c_delay x_binary x_bp_file x_delay")?;
        return Err("Expected 6 arguments for --multisideband: c_binary c_bp_file c_delay x_binary x_bp_file x_delay".into());
    }

    let c_band_path = PathBuf::from(&args.multi_sideband[0]);
    let c_band_bp_path_str = &args.multi_sideband[1];
    let c_band_input_delay: f32 = args.multi_sideband[2].parse()?;
    let x_band_path = PathBuf::from(&args.multi_sideband[3]);
    let x_band_bp_path_str = &args.multi_sideband[4];
    let x_band_input_delay: f32 = args.multi_sideband[5].parse()?;

    // --- Generate output filename and create directory ---
    let parent_dir = c_band_path.parent().unwrap_or_else(|| Path::new(""));
    let output_dir = parent_dir.to_path_buf();
    let plot_output_dir = output_dir.join("frinZ").join("multisideband"); // Create a specific plot directory
    fs::create_dir_all(&plot_output_dir)?;

    let c_band_file_stem = c_band_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let base_filename_prefix = c_band_file_stem.to_string();

    let mut filename_suffix = String::new();
    if !args.rfi.is_empty() {
        filename_suffix.push_str("_rfi");
    }

    let freq_plot_filename = output_dir.join("frinZ").join("multisideband").join(format!(
        "{}_msbc2x_freq_rate{}.png",
        base_filename_prefix, filename_suffix
    ));
    let log_file_path = freq_plot_filename.with_extension("log");
    let log_file = File::create(&log_file_path)?;

    let mut tee_writer = TeeWriter {
        writer1: io::stdout(),
        writer2: log_file,
    };

    let c_band_bp_data = if c_band_bp_path_str != "-1" {
        writeln!(
            tee_writer,
            "Reading C-band bandpass file: {}",
            c_band_bp_path_str
        )?;
        Some(read_bandpass_file(&PathBuf::from(c_band_bp_path_str))?)
    } else {
        None
    };
    let x_band_bp_data = if x_band_bp_path_str != "-1" {
        writeln!(
            tee_writer,
            "Reading X-band bandpass file: {}",
            x_band_bp_path_str
        )?;
        Some(read_bandpass_file(&PathBuf::from(x_band_bp_path_str))?)
    } else {
        None
    };

    let mut all_c_spectra: Vec<Vec<C32>> = Vec::new();
    let mut all_x_spectra: Vec<Vec<C32>> = Vec::new();

    let mut c_band_filtered_rfi_args: Vec<String> = Vec::new();
    let mut x_band_filtered_rfi_args_converted: Vec<String> = Vec::new();

    for rfi_pair in &args.rfi {
        let parts: Vec<&str> = rfi_pair.split(',').collect();
        if parts.len() != 2 {
            writeln!(
                io::stderr(),
                "Invalid RFI format: {}. Expected format is MIN,MAX.",
                rfi_pair
            )?;
            writeln!(
                tee_writer,
                "Invalid RFI format: {}. Expected format is MIN,MAX.",
                rfi_pair
            )?;
            return Err("Invalid RFI format".into());
        }

        let mut min_mhz: f32 = parts[0]
            .parse()
            .map_err(|_| io::Error::from(ErrorKind::InvalidInput))?;
        let mut max_mhz: f32 = parts[1]
            .parse()
            .map_err(|_| io::Error::from(ErrorKind::InvalidInput))?;
        if min_mhz < 0.0 {
            min_mhz = 0.0;
        }
        if 512.0 < max_mhz && max_mhz < 1000.0 {
            max_mhz = 512.0;
        }
        if 1000.0 < min_mhz && min_mhz < 1592.0 {
            min_mhz = 1592.0;
        }
        if max_mhz > 2104.0 {
            max_mhz = 2104.0;
        }

        if min_mhz >= max_mhz {
            writeln!(
                io::stderr(),
                "Invalid RFI range: min ({}) >= max ({}).",
                min_mhz,
                max_mhz
            )?;
            writeln!(
                tee_writer,
                "Invalid RFI range: min ({}) >= max ({}).",
                min_mhz, max_mhz
            )?;
            return Err("Invalid RFI range".into());
        }

        if max_mhz <= 512.0 {
            // C-band range (0-512 MHz)
            writeln!(tee_writer, "C-band RFI range: {}-{} MHz", min_mhz, max_mhz)?;
            c_band_filtered_rfi_args.push(rfi_pair.clone());
        } else if min_mhz >= 1592.0 && max_mhz <= 2104.0 {
            // X-band range (1592-2104 MHz)
            writeln!(tee_writer, "X-band RFI range: {}-{} MHz", min_mhz, max_mhz)?;
            let converted_min = min_mhz - 1592.0;
            let converted_max = max_mhz - 1592.0;
            x_band_filtered_rfi_args_converted
                .push(format!("{:.0},{:.0}", converted_min, converted_max));
        } else {
            writeln!(io::stderr(), "Warning: RFI range {} is outside of C-band (0-512 MHz) or X-band (1592-2104 MHz) valid rangesfor multisideband analysis. Skipping.", rfi_pair)?;
            writeln!(tee_writer, "Warning: RFI range {} is outside of C-band (0-512 MHz) or X-band (1592-2104 MHz) valid rangesfor multisideband analysis. Skipping.", rfi_pair)?;
        }
    }

    writeln!(tee_writer, "Running multi-sideband analysis...")?;
    writeln!(tee_writer, "  C-Band File: {:?}", c_band_path)?;
    if c_band_bp_path_str != "-1" {
        writeln!(tee_writer, "  C-Band Bandpass: {}", c_band_bp_path_str)?;
    }
    writeln!(tee_writer, "  C-Band Input Delay: {} s", c_band_input_delay)?;
    writeln!(tee_writer, "  X-Band File: {:?}", x_band_path)?;
    if x_band_bp_path_str != "-1" {
        writeln!(tee_writer, "  X-Band Bandpass: {}", x_band_bp_path_str)?;
    }
    writeln!(tee_writer, "  X-Band Input Delay: {} s", x_band_input_delay)?;

    // --- Process C-band data ---
    let mut c_band_file = File::open(&c_band_path)?;
    let mut c_band_buffer = Vec::new();
    c_band_file.read_to_end(&mut c_band_buffer)?;
    let mut c_band_cursor = Cursor::new(c_band_buffer.as_slice());

    let c_band_header = parse_header(&mut c_band_cursor)?;
    let c_band_rbw = (c_band_header.sampling_speed as f32 / c_band_header.fft_point as f32) / 1e6;
    let _c_band_rfi_ranges = parse_rfi_ranges(&args.rfi, c_band_rbw)?;

    c_band_cursor.set_position(0);
    let (_c_band_complex_vec_initial, c_band_obs_time, c_band_effective_integ_time) =
        read_visibility_data(&mut c_band_cursor, &c_band_header, 1, 0, 0, false, &[])?;
    c_band_cursor.set_position(256);

    let (c_band_complex_vec, _, _) = read_visibility_data(
        &mut c_band_cursor,
        &c_band_header,
        c_band_header.number_of_sector,
        0,
        0,
        false,
        &[],
    )?;

    // --- Process X-band data ---
    let mut x_band_file = File::open(&x_band_path)?;
    let mut x_band_buffer = Vec::new();
    x_band_file.read_to_end(&mut x_band_buffer)?;
    let mut x_band_cursor = Cursor::new(x_band_buffer.as_slice());

    let x_band_header = parse_header(&mut x_band_cursor)?;
    let x_band_rbw = (x_band_header.sampling_speed as f32 / x_band_header.fft_point as f32) / 1e6;
    let _x_band_rfi_ranges = parse_rfi_ranges(&args.rfi, x_band_rbw)?;

    x_band_cursor.set_position(0);
    let (_x_band_complex_vec_initial, x_band_obs_time, x_band_effective_integ_time) =
        read_visibility_data(&mut x_band_cursor, &x_band_header, 1, 0, 0, false, &[])?;
    x_band_cursor.set_position(256);

    let (x_band_complex_vec, _, _) = read_visibility_data(
        &mut x_band_cursor,
        &x_band_header,
        x_band_header.number_of_sector,
        0,
        0,
        false,
        &[],
    )?;

    writeln!(tee_writer, "Successfully read headers for both bands.")?;

    // --- Validate headers ---
    if !compare_headers_except_observing_frequency(&c_band_header, &x_band_header, &mut tee_writer)?
    {
        writeln!(io::stderr(), "Error: C-band and X-band file headers do not match (excluding observing_frequency and station-specific fields).")?;
        writeln!(tee_writer, "Error: C-band and X-band file headers do not match (excluding observing_frequency and station-specific fields).")?;
        return Err("Header mismatch".into());
    }
    writeln!(
        tee_writer,
        "Headers match (excluding observing_frequency and station-specific fields)."
    )?;

    // --- Perform initial analysis for C-band ---
    let (mut c_band_freq_rate_array, c_band_padding_length) = process_fft(
        c_band_complex_vec.as_slice(),
        c_band_header.number_of_sector,
        c_band_header.fft_point,
        c_band_header.sampling_speed,
        &parse_rfi_ranges(&c_band_filtered_rfi_args, c_band_rbw)?,
        args.rate_padding,
    );
    if let Some(bp_data) = &c_band_bp_data {
        apply_bandpass_correction(&mut c_band_freq_rate_array, bp_data);
    }

    let c_band_delay_rate_2d_data_comp = process_ifft(
        &c_band_freq_rate_array,
        c_band_header.fft_point,
        c_band_padding_length,
    );

    let c_band_temp_args = Args {
        delay_correct: 0.0,
        rate_correct: 0.0,
        search: Vec::new(),
        ..args.clone()
    };

    let c_band_analysis_results = analyze_results(
        &c_band_freq_rate_array,
        &c_band_delay_rate_2d_data_comp,
        &c_band_header,
        c_band_header.number_of_sector,
        c_band_effective_integ_time,
        &c_band_obs_time,
        c_band_padding_length,
        &c_band_temp_args,
        None,
    );

    // --- Perform initial analysis for X-band ---
    let (mut x_band_freq_rate_array, x_band_padding_length) = process_fft(
        x_band_complex_vec.as_slice(),
        x_band_header.number_of_sector,
        x_band_header.fft_point,
        x_band_header.sampling_speed,
        &parse_rfi_ranges(&x_band_filtered_rfi_args_converted, x_band_rbw)?,
        args.rate_padding,
    );
    if let Some(bp_data) = &x_band_bp_data {
        apply_bandpass_correction(&mut x_band_freq_rate_array, bp_data);
    }

    let x_band_delay_rate_2d_data_comp = process_ifft(
        &x_band_freq_rate_array,
        x_band_header.fft_point,
        x_band_padding_length,
    );

    let x_band_temp_args = Args {
        delay_correct: 0.0,
        rate_correct: 0.0,
        search: Vec::new(),
        ..args.clone()
    };

    let x_band_analysis_results = analyze_results(
        &x_band_freq_rate_array,
        &x_band_delay_rate_2d_data_comp,
        &x_band_header,
        x_band_header.number_of_sector,
        x_band_effective_integ_time,
        &x_band_obs_time,
        x_band_padding_length,
        &x_band_temp_args,
        None,
    );

    writeln!(tee_writer, "Initial analysis complete for both bands.")?;
    writeln!(
        tee_writer,
        "C-band SNR: {:.2}",
        c_band_analysis_results.delay_snr
    )?;
    writeln!(
        tee_writer,
        "X-band SNR: {:.2}",
        x_band_analysis_results.delay_snr
    )?;

    // --- Phase correction based on average phase difference ---
    let mut c_band_phases_deg: Vec<f32> = c_band_analysis_results
        .freq_rate_spectrum
        .iter()
        .map(|c| safe_arg(&c).to_degrees())
        .collect();
    unwrap_phase(&mut c_band_phases_deg, false);
    let avg_c_phase = {
        let non_zero_phases: Vec<f64> = c_band_phases_deg
            .iter()
            .filter(|&&p| p != 0.0)
            .map(|&p| p as f64)
            .collect();
        if non_zero_phases.is_empty() {
            f64::NAN
        } else {
            non_zero_phases.iter().sum::<f64>() / non_zero_phases.len() as f64
        }
    };

    let mut x_band_phases_deg: Vec<f32> = x_band_analysis_results
        .freq_rate_spectrum
        .iter()
        .map(|c| safe_arg(c).to_degrees())
        .collect();
    unwrap_phase(&mut x_band_phases_deg, false);
    let avg_x_phase = {
        let non_zero_phases: Vec<f64> = x_band_phases_deg
            .iter()
            .filter(|&&p| p != 0.0)
            .map(|&p| p as f64)
            .collect();
        if non_zero_phases.is_empty() {
            f64::NAN
        } else {
            non_zero_phases.iter().sum::<f64>() / non_zero_phases.len() as f64
        }
    };

    let phase_difference_deg = avg_c_phase - avg_x_phase;
    writeln!(tee_writer, "Average C-band phase: {:.2} deg", avg_c_phase)?;
    writeln!(tee_writer, "Average X-band phase: {:.2} deg", avg_x_phase)?;
    writeln!(
        tee_writer,
        "Phase difference (C - X): {:.2} deg",
        phase_difference_deg
    )?;

    // --- Calculate weighted average delay ---
    let c_weights: Vec<f32> = c_band_analysis_results
        .freq_rate_spectrum
        .iter()
        .map(|c| c.norm_sqr())
        .collect();
    let c_total_weight: f32 = c_weights.iter().sum();
    let c_mean_freq: f64 = if c_total_weight > 1e-9 {
        c_band_analysis_results
            .freq_range
            .iter()
            .zip(c_weights.iter())
            .map(|(&freq, &w)| freq as f64 * w as f64)
            .sum::<f64>()
            / c_total_weight as f64
    } else {
        0.0
    };
    let sigma_nu_c: f64 = if c_total_weight > 1e-9 {
        c_band_analysis_results
            .freq_range
            .iter()
            .zip(c_weights.iter())
            .map(|(&freq, &w)| w as f64 * (freq as f64 - c_mean_freq).powi(2))
            .sum()
    } else {
        0.0
    };

    let x_weights: Vec<f32> = x_band_analysis_results
        .freq_rate_spectrum
        .iter()
        .map(|c| c.norm_sqr())
        .collect();
    let x_total_weight: f32 = x_weights.iter().sum();
    let x_mean_freq: f64 = if x_total_weight > 1e-9 {
        x_band_analysis_results
            .freq_range
            .iter()
            .zip(x_weights.iter())
            .map(|(&freq, &w)| freq as f64 * w as f64)
            .sum::<f64>()
            / x_total_weight as f64
    } else {
        0.0
    };
    let sigma_nu_x: f64 = if x_total_weight > 1e-9 {
        x_band_analysis_results
            .freq_range
            .iter()
            .zip(x_weights.iter())
            .map(|(&freq, &w)| w as f64 * (freq as f64 - x_mean_freq).powi(2))
            .sum()
    } else {
        0.0
    };

    let wc = c_band_analysis_results.delay_snr.powi(2) * sigma_nu_c as f32;
    let wx = x_band_analysis_results.delay_snr.powi(2) * sigma_nu_x as f32;

    let tau_c = c_band_analysis_results.residual_delay;
    let tau_x = x_band_analysis_results.residual_delay;

    let tau_0 = if (wc + wx) > 1e-9 {
        (wc * tau_c + wx * tau_x) / (wc + wx)
    } else {
        let wc_simple = c_band_analysis_results.delay_snr.powi(2);
        let wx_simple = x_band_analysis_results.delay_snr.powi(2);
        if (wc_simple + wx_simple) > 1e-9 {
            (wc_simple * tau_c + wx_simple * tau_x) / (wc_simple + wx_simple)
        } else {
            0.0
        }
    };

    writeln!(
        tee_writer,
        "C-band weight (SNR^2 * FreqVar): {:.2} * {:.2e} = {:.2e}",
        c_band_analysis_results.delay_snr.powi(2),
        sigma_nu_c,
        wc
    )?;
    writeln!(
        tee_writer,
        "X-band weight (SNR^2 * FreqVar): {:.2} * {:.2e} = {:.2e}",
        x_band_analysis_results.delay_snr.powi(2),
        sigma_nu_x,
        wx
    )?;
    writeln!(
        tee_writer,
        "Weighted average delay (tau_0): {:.6} samples",
        tau_0
    )?;

    // --- Calculate delay corrections ---
    let delta_tau_c = tau_c - tau_0;
    let delta_tau_x = tau_x - tau_0;

    writeln!(
        tee_writer,
        "C-band delay correction (delta_tau_c): {:.6} samples",
        delta_tau_c
    )?;
    writeln!(
        tee_writer,
        "X-band delay correction (delta_tau_x): {:.6} samples",
        delta_tau_x
    )?;

    let correction_factor =
        Complex::<f32>::new(0.0, phase_difference_deg.to_radians() as f32).exp() as C32;

    // --- Prepare output header ---
    let mut output_header_bytes = c_band_buffer[..256].to_vec();
    let new_observing_frequency = c_band_header.observing_frequency;
    let new_sampling_speed = c_band_header.sampling_speed + x_band_header.sampling_speed;
    let new_fft_point = c_band_header.fft_point + x_band_header.fft_point;

    output_header_bytes.as_mut_slice()[12..16].copy_from_slice(&new_sampling_speed.to_le_bytes());
    output_header_bytes.as_mut_slice()[16..24]
        .copy_from_slice(&new_observing_frequency.to_le_bytes());
    output_header_bytes.as_mut_slice()[24..28].copy_from_slice(&new_fft_point.to_le_bytes());

    let is_bandpass_corrected = c_band_bp_data.is_some() || x_band_bp_data.is_some();
    if is_bandpass_corrected {
        filename_suffix.push_str("_bp");
    }

    let output_filename = output_dir.join(format!(
        "{}_msbc2x{}.cor",
        base_filename_prefix, filename_suffix
    ));

    let mut output_file = File::create(&output_filename)?;

    output_file.write_all(&output_header_bytes)?;

    let num_sectors = c_band_header
        .number_of_sector
        .min(x_band_header.number_of_sector);
    writeln!(tee_writer, "Combining {} sectors...", num_sectors)?;

    for i in 0..num_sectors {
        let mut c_band_cursor_sector = Cursor::new(c_band_buffer.as_slice());
        let c_sector_header =
            read_sector_header(&mut c_band_cursor_sector, &c_band_header, 1, 0, i, false)?;

        let (mut c_complex_vec_sector, _c_current_obs_time, _) = read_visibility_data(
            &mut c_band_cursor_sector,
            &c_band_header,
            1,
            0,
            i,
            false,
            &[],
        )?;

        if let Some(bp_data) = &c_band_bp_data {
            const EPSILON: f32 = 1e-9;
            let bandpass_sum: C32 = bp_data.iter().copied().sum();
            let bandpass_mean = bandpass_sum / bp_data.len() as f32;
            for (elem, &bp_val) in c_complex_vec_sector.iter_mut().zip(bp_data.iter()) {
                if bp_val.norm() > EPSILON {
                    *elem = (*elem / bp_val) * bandpass_mean;
                }
            }
        }

        let c_start_time_offset_sec = 0.0;
        let c_complex_vec_sector_f64: Vec<Complex<f64>> = c_complex_vec_sector
            .iter()
            .map(|&c| num_complex::Complex::new(c.re as f64, c.im as f64))
            .collect();

        let corrected_c_complex_vec = apply_phase_correction(
            &[c_complex_vec_sector_f64],
            0.0,
            delta_tau_c,
            0.0,
            c_band_effective_integ_time,
            c_band_header.sampling_speed as u32,
            c_band_header.fft_point as u32,
            c_start_time_offset_sec,
        );

        let corrected_c_complex_vec_f32: Vec<C32> = corrected_c_complex_vec
            .into_iter()
            .flatten()
            .map(|v| C32::new(v.re as f32, v.im as f32))
            .collect();

        output_file.write_all(&c_sector_header[0])?;
        for val in corrected_c_complex_vec_f32.iter() {
            output_file.write_all(&val.re.to_le_bytes())?;
            output_file.write_all(&val.im.to_le_bytes())?;
        }
        all_c_spectra.push(corrected_c_complex_vec_f32.clone());

        let mut x_band_cursor_sector = Cursor::new(x_band_buffer.as_slice());
        let _x_sector_header =
            read_sector_header(&mut x_band_cursor_sector, &x_band_header, 1, 0, i, false)?;

        let (mut x_complex_vec_sector, _x_current_obs_time, _) = read_visibility_data(
            &mut x_band_cursor_sector,
            &x_band_header,
            1,
            0,
            i,
            false,
            &[],
        )?;
        if let Some(bp_data) = &x_band_bp_data {
            const EPSILON: f32 = 1e-9;
            let bandpass_sum: C32 = bp_data.iter().copied().sum();
            let bandpass_mean = bandpass_sum / bp_data.len() as f32;
            for (elem, &bp_val) in x_complex_vec_sector.iter_mut().zip(bp_data.iter()) {
                if bp_val.norm() > EPSILON {
                    *elem = (*elem / bp_val) * bandpass_mean;
                }
            }
        }

        let x_start_time_offset_sec = 0.0;
        let x_complex_vec_sector_f64: Vec<Complex<f64>> = x_complex_vec_sector
            .iter()
            .map(|&c| num_complex::Complex::new(c.re as f64, c.im as f64))
            .collect();

        let corrected_x_complex_vec = apply_phase_correction(
            &[x_complex_vec_sector_f64],
            0.0,
            delta_tau_x,
            0.0,
            x_band_effective_integ_time,
            x_band_header.sampling_speed as u32,
            x_band_header.fft_point as u32,
            x_start_time_offset_sec,
        );

        let mut corrected_x_complex_vec_f32: Vec<C32> = corrected_x_complex_vec
            .into_iter()
            .flatten()
            .map(|v| C32::new(v.re as f32, v.im as f32))
            .collect();

        for val in corrected_x_complex_vec_f32.iter_mut() {
            *val *= correction_factor;
            output_file.write_all(&val.re.to_le_bytes())?;
            output_file.write_all(&val.im.to_le_bytes())?;
        }
        all_x_spectra.push(corrected_x_complex_vec_f32.clone());
    }

    writeln!(tee_writer, "Initial analysis complete for both bands.")?;
    writeln!(
        tee_writer,
        "C-band SNR: {:.2}",
        c_band_analysis_results.delay_snr
    )?;
    writeln!(
        tee_writer,
        "X-band SNR: {:.2}",
        x_band_analysis_results.delay_snr
    )?;

    let _c_band_freq_resolution_mhz =
        (c_band_header.sampling_speed as f64 / c_band_header.fft_point as f64) / 1e6;
    let _x_band_freq_resolution_mhz =
        (x_band_header.sampling_speed as f64 / x_band_header.fft_point as f64) / 1e6;
    let x_band_offset_mhz = 1592.0;

    let rate_values_f32 = rate_cal(
        4.0 * c_band_header.number_of_sector as f32,
        c_band_effective_integ_time,
    );
    let _rate_profile: Vec<(f64, f64)> = rate_values_f32.iter().map(|&r| (r as f64, 0.0)).collect();

    let _max_padding_length = c_band_padding_length.max(x_band_padding_length);

    let mut c_band_amp_profile: Vec<(f64, f64)> = Vec::new();
    let mut c_band_phase_profile: Vec<(f64, f64)> = Vec::new();
    let mut x_band_amp_profile: Vec<(f64, f64)> = Vec::new();
    let mut x_band_phase_profile: Vec<(f64, f64)> = Vec::new();
    let mut x_band_uncalibrated_phase_profile: Vec<(f64, f64)> = Vec::new();

    for (i, &freq_mhz) in c_band_analysis_results.freq_range.iter().enumerate() {
        let amp = c_band_analysis_results.freq_rate_spectrum[i].norm() as f64;
        let phase = safe_arg(&c_band_analysis_results.freq_rate_spectrum[i]).to_degrees() as f64;
        c_band_amp_profile.push((freq_mhz as f64, amp));
        c_band_phase_profile.push((freq_mhz as f64, phase));
    }

    for (i, &freq_mhz) in x_band_analysis_results.freq_range.iter().enumerate() {
        let uncalibrated_phase =
            safe_arg(&x_band_analysis_results.freq_rate_spectrum[i]).to_degrees() as f64;
        x_band_uncalibrated_phase_profile
            .push((freq_mhz as f64 + x_band_offset_mhz, uncalibrated_phase));
        let temp = x_band_analysis_results.freq_rate_spectrum[i] * correction_factor;
        let amp = temp.norm() as f64;
        let phase = safe_arg(&temp).to_degrees() as f64;
        x_band_amp_profile.push((freq_mhz as f64 + x_band_offset_mhz, amp));
        x_band_phase_profile.push((freq_mhz as f64 + x_band_offset_mhz, phase));
    }

    let c_band_rate_profile: Vec<(f64, f64)> = c_band_analysis_results
        .rate_range
        .iter()
        .zip(c_band_analysis_results.freq_rate.iter())
        .map(|(&rate, &amp)| (rate as f64, amp as f64))
        .collect();

    let x_band_rate_profile: Vec<(f64, f64)> = x_band_analysis_results
        .rate_range
        .iter()
        .zip(x_band_analysis_results.freq_rate.iter())
        .map(|(&rate, &amp)| (rate as f64, amp as f64))
        .collect();

    let c_band_freq_resolution_mhz =
        (c_band_header.sampling_speed as f64 / c_band_header.fft_point as f64) / 1e6;
    let x_band_freq_resolution_mhz =
        (x_band_header.sampling_speed as f64 / x_band_header.fft_point as f64) / 1e6;
    let x_band_offset_mhz = 1592.0;

    let c_band_rate_values_f32 =
        rate_cal(c_band_padding_length as f32, c_band_effective_integ_time);
    let x_band_rate_values_f32 =
        rate_cal(x_band_padding_length as f32, x_band_effective_integ_time);

    let heatmap_func = move |freq_mhz: f64, rate_hz: f64| -> f64 {
        let rate_idx_c = ((rate_hz - c_band_rate_values_f32[0] as f64)
            / (c_band_rate_values_f32[1] - c_band_rate_values_f32[0]) as f64)
            .round() as usize;
        let rate_idx_x = ((rate_hz - x_band_rate_values_f32[0] as f64)
            / (x_band_rate_values_f32[1] - x_band_rate_values_f32[0]) as f64)
            .round() as usize;

        if freq_mhz >= 0.0 && freq_mhz < 512.0 + 0.01 {
            // C-band range
            let freq_idx = (freq_mhz / c_band_freq_resolution_mhz).round() as usize;
            if freq_idx < c_band_freq_rate_array.shape()[0]
                && rate_idx_c < c_band_freq_rate_array.shape()[1]
            {
                c_band_freq_rate_array[[freq_idx, rate_idx_c]].norm() as f64
            } else {
                0.0
            }
        } else if freq_mhz >= x_band_offset_mhz && freq_mhz < x_band_offset_mhz + 512.0 + 0.01 {
            // X-band range
            let freq_idx =
                ((freq_mhz - x_band_offset_mhz) / x_band_freq_resolution_mhz).round() as usize;
            if freq_idx < x_band_freq_rate_array.shape()[0]
                && rate_idx_x < x_band_freq_rate_array.shape()[1]
            {
                x_band_freq_rate_array[[freq_idx, rate_idx_x]].norm() as f64
            } else {
                0.0
            }
        } else {
            0.0
        }
    };

    let freq_plot_filename = plot_output_dir.join(format!(
        "{}_msb_freq_rate{}.png",
        base_filename_prefix, filename_suffix
    ));

    let c_max_amp = c_band_amp_profile
        .iter()
        .map(|&(_, amp)| amp)
        .fold(0.0f64, f64::max);
    let x_max_amp = x_band_amp_profile
        .iter()
        .map(|&(_, amp)| amp)
        .fold(0.0f64, f64::max);
    let max_amplitude = c_max_amp.max(x_max_amp);

    let mut stat_keys_vec: Vec<String> = Vec::new();
    let mut stat_vals_vec: Vec<String> = Vec::new();

    stat_keys_vec.push("".to_string());
    stat_vals_vec.push(
        c_band_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string(),
    );
    stat_keys_vec.push("".to_string());
    stat_vals_vec.push(
        x_band_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string(),
    );

    stat_keys_vec.push("".to_string());
    stat_vals_vec.push(
        PathBuf::from(c_band_bp_path_str)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string(),
    );
    stat_keys_vec.push("".to_string());
    stat_vals_vec.push(
        PathBuf::from(x_band_bp_path_str)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string(),
    );

    stat_keys_vec.push("Avg C/X Phase".to_string());
    stat_vals_vec.push(format!("{:.5}/{:.5} deg", avg_c_phase, avg_x_phase));
    stat_keys_vec.push("Phase Diff (C-X)".to_string());
    stat_vals_vec.push(format!("{:.5} deg", phase_difference_deg));

    stat_keys_vec.push("C/X Delay Corr".to_string());
    stat_vals_vec.push(format!("{:.6}/{:.6} samples", delta_tau_c, delta_tau_x));

    let min_freq_c = c_band_header.observing_frequency as f64 / 1e6;
    let bandwidth_c = c_band_header.sampling_speed as f64 / 1e6;
    let max_freq_c = min_freq_c + bandwidth_c / 2.0;
    let min_freq_x = x_band_header.observing_frequency as f64 / 1e6;
    let bandwidth_x = x_band_header.sampling_speed as f64 / 1e6;
    let max_freq_x = min_freq_x + bandwidth_x / 2.0;
    stat_keys_vec.push("Base Freq".to_string());
    stat_vals_vec.push(format!("{:.0} MHz", min_freq_c));
    stat_keys_vec.push("C/X Freq".to_string());
    stat_vals_vec.push(format!(
        "{:.0}--{:.0}/{:.0}--{:.0} MHz",
        min_freq_c, max_freq_c, min_freq_x, max_freq_x
    ));

    stat_keys_vec.push("C/X(+1592) RFI".to_string());
    let c_rfi_str = if c_band_filtered_rfi_args.is_empty() {
        "None".to_string()
    } else {
        c_band_filtered_rfi_args
            .iter()
            .map(|s| s.replace(',', "-"))
            .collect::<Vec<String>>()
            .join(", ")
    };
    let x_rfi_str = if x_band_filtered_rfi_args_converted.is_empty() {
        "None".to_string()
    } else {
        x_band_filtered_rfi_args_converted
            .iter()
            .map(|s| s.replace(',', "-"))
            .collect::<Vec<String>>()
            .join(", ")
    };
    stat_vals_vec.push(format!("{}/{} MHz", c_rfi_str, x_rfi_str));

    let stat_keys_slice: Vec<&str> = stat_keys_vec.iter().map(|s| s.as_str()).collect();
    let stat_vals_slice: Vec<&str> = stat_vals_vec.iter().map(|s| s.as_str()).collect();

    frequency_plane_msb(
        &c_band_amp_profile,
        &c_band_phase_profile,
        &x_band_amp_profile,
        &x_band_phase_profile,
        &x_band_uncalibrated_phase_profile,
        &c_band_rate_profile,
        &x_band_rate_profile,
        heatmap_func,
        &stat_keys_slice,
        &stat_vals_slice,
        freq_plot_filename.to_str().unwrap(),
        2104.0, // Total bandwidth for plot
        max_amplitude,
    )?;

    writeln!(
        tee_writer,
        "Multi-sideband .cor file saved to {:?}",
        output_filename
    )?;

    writeln!(
        tee_writer,
        "Multi-sideband frequency spectrum plot saved to {:?}",
        freq_plot_filename
    )?;

    Ok(())
}

// Helper function to compare CorHeader fields, ignoring observing_frequency and station-specific fields
fn compare_headers_except_observing_frequency<W: Write>(
    header1: &CorHeader,
    header2: &CorHeader,
    writer: &mut W,
) -> io::Result<bool> {
    let mut all_match = true;
    writeln!(
        writer,
        "{:<25} {:<20} {:<20} {:<10}",
        "Parameter", "C-Band", "X-Band", "Match"
    )?;
    writeln!(writer, "{:-<25} {:-<20} {:-<20} {:-<10}", "", "", "", "")?;

    let field_name = "header_version";
    let c_val = format!("{}", header1.header_version);
    let x_val = format!("{}", header2.header_version);
    let matches = header1.header_version == header2.header_version;
    writeln!(
        writer,
        "{:<25} {:<20} {:<20} {:<10}",
        field_name, c_val, x_val, matches
    )?;
    if !matches {
        all_match = false;
    }

    let field_name = "software_version";
    let c_val = format!("{}", header1.software_version);
    let x_val = format!("{}", header2.software_version);
    let matches = header1.software_version == header2.software_version;
    writeln!(
        writer,
        "{:<25} {:<20} {:<20} {:<10}",
        field_name, c_val, x_val, matches
    )?;
    if !matches {
        all_match = false;
    }

    let field_name = "sampling_speed";
    let c_val = format!("{}", header1.sampling_speed);
    let x_val = format!("{}", header2.sampling_speed);
    let matches = header1.sampling_speed == header2.sampling_speed;
    writeln!(
        writer,
        "{:<25} {:<20} {:<20} {:<10}",
        field_name, c_val, x_val, matches
    )?;
    if !matches {
        all_match = false;
    }

    let field_name = "fft_point";
    let c_val = format!("{}", header1.fft_point);
    let x_val = format!("{}", header2.fft_point);
    let matches = header1.fft_point == header2.fft_point;
    writeln!(
        writer,
        "{:<25} {:<20} {:<20} {:<10}",
        field_name, c_val, x_val, matches
    )?;
    if !matches {
        all_match = false;
    }

    let field_name = "number_of_sector";
    let c_val = format!("{}", header1.number_of_sector);
    let x_val = format!("{}", header2.number_of_sector);
    let matches = header1.number_of_sector == header2.number_of_sector;
    writeln!(
        writer,
        "{:<25} {:<20} {:<20} {:<10}",
        field_name, c_val, x_val, matches
    )?;
    if !matches {
        all_match = false;
    }

    let field_name = "station1_name";
    let c_val = format!("{}", header1.station1_name);
    let x_val = format!("{}", header2.station1_name);
    let matches = header1.station1_name == header2.station1_name;
    writeln!(
        writer,
        "{:<25} {:<20} {:<20} {:<10}",
        field_name, c_val, x_val, matches
    )?;
    if !matches {
        all_match = false;
    }

    let field_name = "station1_code";
    let c_val = format!("{}", header1.station1_code);
    let x_val = format!("{}", header2.station1_code);
    let matches = header1.station1_code == header2.station1_code;
    writeln!(
        writer,
        "{:<25} {:<20} {:<20} {:<10}",
        field_name, c_val, x_val, matches
    )?;
    if !matches {
        all_match = false;
    }

    let field_name = "station2_name";
    let c_val = format!("{}", header1.station2_name);
    let x_val = format!("{}", header2.station2_name);
    let matches = header1.station2_name == header2.station2_name;
    writeln!(
        writer,
        "{:<25} {:<20} {:<20} {:<10}",
        field_name, c_val, x_val, matches
    )?;
    if !matches {
        all_match = false;
    }

    let field_name = "station2_code";
    let c_val = format!("{}", header1.station2_code);
    let x_val = format!("{}", header2.station2_code);
    let matches = header1.station2_code == header2.station2_code;
    writeln!(
        writer,
        "{:<25} {:<20} {:<20} {:<10}",
        field_name, c_val, x_val, matches
    )?;
    if !matches {
        all_match = false;
    }

    let field_name = "source_name";
    let c_val = format!("{}", header1.source_name);
    let x_val = format!("{}", header2.source_name);
    let matches = header1.source_name == header2.source_name;
    writeln!(
        writer,
        "{:<25} {:<20} {:<20} {:<10}",
        field_name, c_val, x_val, matches
    )?;
    if !matches {
        all_match = false;
    }

    let field_name = "source_position_ra";
    let c_val = format!("{}", header1.source_position_ra);
    let x_val = format!("{}", header2.source_position_ra);
    let matches = header1.source_position_ra == header2.source_position_ra;
    writeln!(
        writer,
        "{:<25} {:<20} {:<20} {:<10}",
        field_name, c_val, x_val, matches
    )?;
    if !matches {
        all_match = false;
    }

    let field_name = "source_position_dec";
    let c_val = format!("{}", header1.source_position_dec);
    let x_val = format!("{}", header2.source_position_dec);
    let matches = header1.source_position_dec == header2.source_position_dec;
    writeln!(
        writer,
        "{:<25} {:<20} {:<20} {:<10}",
        field_name, c_val, x_val, matches
    )?;
    if !matches {
        all_match = false;
    }

    Ok(all_match)
}
