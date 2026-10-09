// Correction kernels retain explicit physical correction terms.
#![allow(clippy::too_many_arguments)]
use std::error::Error;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use num_complex::Complex;

use crate::args::Args;
use crate::bandpass::read_bandpass_file;
use crate::fft::apply_phase_correction_in_place_at_times;
use crate::header::{parse_header, CorHeader};
use crate::input_support::read_input_bytes;
use crate::read::{
    read_sector_time_offsets, read_visibility_data, FILE_HEADER_SIZE, SECTOR_HEADER_SIZE,
};
use crate::rfi::parse_rfi_ranges;
use crate::search;

type C32 = Complex<f32>;

const STATION2_CLOCK_DELAY_OFFSET: usize = 216;
const STATION2_CLOCK_RATE_OFFSET: usize = 224;
const STATION2_CLOCK_ACEL_OFFSET: usize = 232;
const STATION2_CLOCK_JERK_OFFSET: usize = 240;
const STATION2_CLOCK_SNAP_OFFSET: usize = 248;

fn mkcor_output_path(input_path: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let stem = input_path
        .file_stem()
        .ok_or("--mkcor input has no filename stem")?
        .to_string_lossy();
    let stem = if stem.ends_with("_mkcor") {
        stem.into_owned()
    } else {
        format!("{stem}_mkcor")
    };
    Ok(input_path.with_file_name(format!("{stem}.cor")))
}

fn ensure_supported_args(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.input.is_none() {
        return Err("--mkcor requires --input FILE".into());
    }
    if args.length != 0
        || args.skip != 0
        || args.loop_ != 1
        || args.cumulate != 0
        || args.stfft != 0
    {
        return Err("--mkcor writes one coherent correction for the full input; use --len 0 --skip 0 --loop 1 without --cumulate/--stfft".into());
    }
    if args.scan_correct.is_some() || args.spike34m.is_some() {
        return Err("--mkcor does not support --scan-correct or --spike34; both are interval-dependent corrections".into());
    }
    if args.fft_rebin.is_some()
        || args.norm_acf
        || args.contamination.is_some()
        || args.contamination_subtract.is_some()
    {
        return Err(
            "--mkcor does not support --fft-rebin, --norm-acf, or contamination modes".into(),
        );
    }
    if args
        .search
        .iter()
        .any(|mode| mode == "rate" || mode == "acel")
    {
        return Err("--mkcor supports the single full-band --search peak/deep/coherent solution, not --search rate/acel".into());
    }
    if !args.search.is_empty() && args.primary_search_mode().is_none() {
        return Err("--mkcor search mode must be peak, deep, or coherent".into());
    }
    Ok(())
}

fn pad_rows_to_power_of_two(data: &mut Vec<C32>, rows: i32, width: usize) -> i32 {
    if rows <= 1 || width == 0 {
        return rows.max(1);
    }
    let padded_rows = (rows as u32).next_power_of_two() as i32;
    data.resize(padded_rows as usize * width, C32::new(0.0, 0.0));
    padded_rows
}

fn sector_size(header: &CorHeader) -> Result<usize, Box<dyn Error>> {
    if header.fft_point <= 0 || header.fft_point % 2 != 0 {
        return Err(format!("invalid FFT point in .cor header: {}", header.fft_point).into());
    }
    Ok((SECTOR_HEADER_SIZE as usize) + (header.fft_point as usize / 2) * 8)
}

fn read_f32_le(bytes: &[u8], offset: usize) -> Result<f32, Box<dyn Error>> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or("truncated .cor sector while reading visibility")?;
    Ok(f32::from_le_bytes(slice.try_into()?))
}

fn write_f32_le(bytes: &mut [u8], offset: usize, value: f32) -> Result<(), Box<dyn Error>> {
    let slice = bytes
        .get_mut(offset..offset + 4)
        .ok_or("truncated .cor sector while writing visibility")?;
    slice.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_f64_le(bytes: &mut [u8], offset: usize, value: f64) -> Result<(), Box<dyn Error>> {
    let slice = bytes
        .get_mut(offset..offset + 8)
        .ok_or("truncated .cor header while updating clocks")?;
    slice.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn apply_correction_to_cor_bytes(
    bytes: &mut [u8],
    header: &CorHeader,
    delay_samples: f32,
    rate_hz: f32,
    acel_hz_per_s: f32,
    jerk_hz_per_s2: f32,
    snap_hz_per_s3: f32,
) -> Result<(), Box<dyn Error>> {
    let rows = usize::try_from(header.number_of_sector.max(0))?;
    let width = (header.fft_point / 2) as usize;
    let sector_size = sector_size(header)?;
    let expected_len = FILE_HEADER_SIZE as usize + rows.saturating_mul(sector_size);
    if bytes.len() < expected_len {
        return Err(format!(
            ".cor payload is truncated: {} bytes, expected at least {} bytes for {} sectors",
            bytes.len(),
            expected_len,
            rows
        )
        .into());
    }

    let time_offsets = read_sector_time_offsets(bytes, header, 0, rows)?;
    let mut spectra = Vec::with_capacity(width);
    for (row, &time_sec) in time_offsets.iter().enumerate() {
        let sector_start = FILE_HEADER_SIZE as usize + row * sector_size;
        let visibility_start = sector_start + SECTOR_HEADER_SIZE as usize;
        spectra.clear();
        for channel in 0..width {
            let offset = visibility_start + channel * 8;
            spectra.push(C32::new(
                read_f32_le(bytes, offset)?,
                read_f32_le(bytes, offset + 4)?,
            ));
        }
        // Use the recorded sector start, including gaps between merged scans.
        // The wideband kernel also applies the rate-derived delay at every
        // frequency channel, using the same file epoch as ordinary analysis.
        apply_phase_correction_in_place_at_times(
            &mut spectra,
            width,
            rate_hz,
            delay_samples,
            acel_hz_per_s,
            jerk_hz_per_s2,
            snap_hz_per_s3,
            header.sampling_speed as u32,
            header.fft_point as u32,
            &[time_sec],
            header.observing_frequency,
        );
        for (channel, value) in spectra.iter().enumerate() {
            let offset = visibility_start + channel * 8;
            write_f32_le(bytes, offset, value.re)?;
            write_f32_le(bytes, offset + 4, value.im)?;
        }
    }
    Ok(())
}

fn update_station2_clock_header(
    bytes: &mut [u8],
    header: &CorHeader,
    delay_samples: f32,
    rate_hz: f32,
    acel_hz_per_s: f32,
    jerk_hz_per_s2: f32,
    snap_hz_per_s3: f32,
) -> Result<(), Box<dyn Error>> {
    let sample_rate_hz = header.sampling_speed as f64;
    let reference_frequency_hz = header.observing_frequency;
    if sample_rate_hz <= 0.0 || !reference_frequency_hz.is_finite() || reference_frequency_hz <= 0.0
    {
        return Err("invalid sampling/observing frequency in .cor header".into());
    }

    // V_12 is corrected by exp[-i 2 pi nu * delta_tau].  With the .cor
    // baseline clock convention tau_12 = tau_1 - tau_2, preserving station 1
    // means adding delta_tau to station 2 gives tau_12' = tau_12-delta_tau.
    // Rate and higher terms are fringe derivatives at nu_ref, hence divide by
    // nu_ref to store their equivalent delay-polynomial coefficients.
    write_f64_le(
        bytes,
        STATION2_CLOCK_DELAY_OFFSET,
        header.station2_clock_delay + delay_samples as f64 / sample_rate_hz,
    )?;
    write_f64_le(
        bytes,
        STATION2_CLOCK_RATE_OFFSET,
        header.station2_clock_rate + rate_hz as f64 / reference_frequency_hz,
    )?;
    write_f64_le(
        bytes,
        STATION2_CLOCK_ACEL_OFFSET,
        header.station2_clock_acel + acel_hz_per_s as f64 / reference_frequency_hz,
    )?;
    write_f64_le(
        bytes,
        STATION2_CLOCK_JERK_OFFSET,
        header.station2_clock_jerk + jerk_hz_per_s2 as f64 / reference_frequency_hz,
    )?;
    write_f64_le(
        bytes,
        STATION2_CLOCK_SNAP_OFFSET,
        header.station2_clock_snap + snap_hz_per_s3 as f64 / reference_frequency_hz,
    )?;
    Ok(())
}

pub fn run_mkcor(args: &Args) -> Result<(), Box<dyn Error>> {
    ensure_supported_args(args)?;
    let input_path = args.input.as_ref().expect("validated input");
    let output_path = mkcor_output_path(input_path)?;
    if output_path.exists() {
        return Err(format!(
            "--mkcor output already exists: {} (refusing to overwrite)",
            output_path.display()
        )
        .into());
    }

    let mut bytes = read_input_bytes(input_path)?;
    let mut cursor = Cursor::new(bytes.as_slice());
    let header = parse_header(&mut cursor)?;
    let width = (header.fft_point / 2) as usize;
    if width == 0 || header.number_of_sector <= 0 {
        return Err("--mkcor input contains no visibility sectors".into());
    }

    let mut read_cursor = Cursor::new(bytes.as_slice());
    let (mut search_data, obs_time, effective_integ_time) = read_visibility_data(
        &mut read_cursor,
        &header,
        header.number_of_sector,
        0,
        0,
        false,
        &[],
    )?;
    let physical_rows = header.number_of_sector;
    if search_data.len() != physical_rows as usize * width {
        return Err("--mkcor could not read all visibility rows".into());
    }

    // Match the ordinary processing order exactly: supplied Taylor terms are
    // removed before the residual delay/rate search.
    if args.delay_correct != 0.0
        || args.rate_correct != 0.0
        || args.acel_correct != 0.0
        || args.jerk_correct != 0.0
        || args.snap_correct != 0.0
    {
        let time_offsets = read_sector_time_offsets(&bytes, &header, 0, physical_rows as usize)?;
        apply_phase_correction_in_place_at_times(
            &mut search_data,
            width,
            args.rate_correct,
            args.delay_correct,
            args.acel_correct,
            args.jerk_correct,
            args.snap_correct,
            header.sampling_speed as u32,
            header.fft_point as u32,
            &time_offsets,
            header.observing_frequency,
        );
    }

    let (residual_delay, residual_rate, search_snr) = if args.search.is_empty() {
        (0.0, 0.0, None)
    } else {
        let bandwidth_mhz = header.sampling_speed as f32 / 2.0 / 1_000_000.0;
        let rbw_mhz = bandwidth_mhz / header.fft_point as f32 * 2.0;
        let rfi_ranges = parse_rfi_ranges(&args.rfi, rbw_mhz)?;
        let bandpass_data = match &args.bandpass {
            Some(path) => Some(read_bandpass_file(path)?),
            None => None,
        };
        let padded_rows = pad_rows_to_power_of_two(&mut search_data, physical_rows, width);
        let mut search_args = args.clone();
        search_args.delay_correct = 0.0;
        search_args.rate_correct = 0.0;
        search_args.acel_correct = 0.0;
        search_args.jerk_correct = 0.0;
        search_args.snap_correct = 0.0;
        let result = match args.primary_search_mode() {
            Some("deep") => search::run_deep_search(
                &search_data,
                &header,
                padded_rows,
                physical_rows,
                effective_integ_time,
                &obs_time,
                &obs_time,
                &rfi_ranges,
                &bandpass_data,
                &search_args,
                header.number_of_sector,
                search_args.cpu,
                None,
            )?,
            Some("coherent") => search::run_coherent_search(
                &search_data,
                &header,
                padded_rows,
                physical_rows,
                effective_integ_time,
                &obs_time,
                &obs_time,
                &rfi_ranges,
                &bandpass_data,
                &search_args,
                header.number_of_sector,
                search_args.cpu,
                None,
            )?,
            _ => search::run_peak_search(
                &search_data,
                &header,
                padded_rows,
                physical_rows,
                effective_integ_time,
                &obs_time,
                &obs_time,
                &rfi_ranges,
                &bandpass_data,
                &search_args,
                header.number_of_sector,
                search_args.cpu,
                None,
            )?,
        };
        (
            result.analysis_results.residual_delay,
            result.analysis_results.residual_rate,
            Some(result.analysis_results.delay_snr),
        )
    };

    let total_delay = args.delay_correct + residual_delay;
    let total_rate = args.rate_correct + residual_rate;
    let total_acel = args.acel_correct;
    apply_correction_to_cor_bytes(
        &mut bytes,
        &header,
        total_delay,
        total_rate,
        total_acel,
        args.jerk_correct,
        args.snap_correct,
    )?;
    update_station2_clock_header(
        &mut bytes,
        &header,
        total_delay,
        total_rate,
        total_acel,
        args.jerk_correct,
        args.snap_correct,
    )?;
    fs::write(&output_path, bytes)?;

    if let Some(snr) = search_snr {
        println!(
            "Mkcor search residual: delay {:+.8} sample, rate {:+.8} Hz, SNR {:.1}",
            residual_delay, residual_rate, snr
        );
    } else {
        println!("Mkcor search residual: not requested (manual correction only)");
    }
    println!(
        "Mkcor applied: delay {:+.8} sample, rate {:+.8} Hz, acel {:+.8} Hz/s",
        total_delay, total_rate, total_acel
    );
    println!("Mkcor written: {}", output_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::process_cor_file;
    use std::path::Path;

    fn sample_cor(times_sec: &[f64]) -> Vec<u8> {
        let fft_point = 32usize;
        let sector_size = SECTOR_HEADER_SIZE as usize + fft_point * 4;
        let mut bytes = vec![0u8; FILE_HEADER_SIZE as usize + times_sec.len() * sector_size];
        bytes[..4].copy_from_slice(&[0x83, 0xf9, 0xa2, 0x3e]);
        bytes[12..16].copy_from_slice(&1_024_000_000i32.to_le_bytes());
        bytes[16..24].copy_from_slice(&6_600_000_000f64.to_le_bytes());
        bytes[24..28].copy_from_slice(&(fft_point as i32).to_le_bytes());
        bytes[28..32].copy_from_slice(&(times_sec.len() as i32).to_le_bytes());
        bytes[32..40].copy_from_slice(b"YAMAGU34");
        bytes[80..88].copy_from_slice(b"HITACH32");
        bytes[128..135].copy_from_slice(b"TESTSRC");
        for (offset, value) in [
            (48, -3_502_567.576),
            (56, 3_950_885.734),
            (64, 3_566_449.115),
            (96, -3_961_788.974),
            (104, 3_243_597.492),
            (112, 3_790_597.692),
        ] {
            write_f64_le(&mut bytes, offset, value).unwrap();
        }
        for (row, &time) in times_sec.iter().enumerate() {
            let start = FILE_HEADER_SIZE as usize + row * sector_size;
            let seconds = 1_769_674_860i32 + time.floor() as i32;
            let nanoseconds = (time.fract() * 1_000_000_000.0).round() as u32;
            bytes[start..start + 4].copy_from_slice(&seconds.to_le_bytes());
            bytes[start + 4..start + 8].copy_from_slice(&nanoseconds.to_le_bytes());
            bytes[start + 8..start + 12].copy_from_slice(&(seconds + 1).to_le_bytes());
            bytes[start + 12..start + 16].copy_from_slice(&nanoseconds.to_le_bytes());
            write_f32_le(&mut bytes, start + 112, 1.0).unwrap();
            for channel in 0..fft_point / 2 {
                let offset = start + SECTOR_HEADER_SIZE as usize + channel * 8;
                write_f32_le(&mut bytes, offset, 0.75 + (row + channel) as f32 * 0.01).unwrap();
                write_f32_le(&mut bytes, offset + 4, (row as f32 - channel as f32) * 0.02).unwrap();
            }
        }
        bytes
    }

    #[test]
    fn output_name_inserts_mkcor_before_cor_extension() {
        assert_eq!(
            mkcor_output_path(Path::new("cor/YAMAGU34_HITACH32_x.cor"))
                .unwrap()
                .to_string_lossy(),
            "cor/YAMAGU34_HITACH32_x_mkcor.cor"
        );
    }

    #[test]
    fn correction_uses_recorded_time_and_delay_at_every_channel() {
        let times = [0.125, 1.375, 362.625];
        let original = sample_cor(&times);
        let header = parse_header(&mut Cursor::new(original.as_slice())).unwrap();
        let mut corrected = original.clone();
        let delay = 1.25f32;
        let rate = -0.002f32;
        apply_correction_to_cor_bytes(&mut corrected, &header, delay, rate, 0.0, 0.0, 0.0).unwrap();

        assert_eq!(&corrected[..256], &original[..256]);
        let stride = sector_size(&header).unwrap();
        for (row, &time) in times.iter().enumerate() {
            let start = FILE_HEADER_SIZE as usize + row * stride;
            assert_eq!(
                &corrected[start..start + 128],
                &original[start..start + 128]
            );
            let cycles = rate as f64 * (time - times[0]);
            for channel in 0..header.fft_point as usize / 2 {
                let frequency =
                    channel as f64 * header.sampling_speed as f64 / header.fft_point as f64;
                let phase = -2.0
                    * std::f64::consts::PI
                    * (frequency * delay as f64 / header.sampling_speed as f64
                        + (1.0 + frequency / header.observing_frequency) * cycles);
                let offset = start + SECTOR_HEADER_SIZE as usize + channel * 8;
                let input = C32::new(
                    read_f32_le(&original, offset).unwrap(),
                    read_f32_le(&original, offset + 4).unwrap(),
                );
                let actual = C32::new(
                    read_f32_le(&corrected, offset).unwrap(),
                    read_f32_le(&corrected, offset + 4).unwrap(),
                );
                let expected = input * C32::new(phase.cos() as f32, phase.sin() as f32);
                assert!(
                    (actual - expected).norm() < 1.0e-6,
                    "row={row}, channel={channel}"
                );
            }
        }
    }

    #[test]
    fn mkcor_matches_manual_analysis_across_scan_gaps_and_window_sizes() {
        let times = [
            0.125, 1.375, 2.625, 3.875, 364.125, 365.375, 366.625, 367.875, 728.125, 729.375,
            730.625, 731.875, 732.125, 733.375, 734.625, 735.875,
        ];
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("YAMAGU34_HITACH32_2026029082100_test.cor");
        fs::write(&input, sample_cor(&times)).unwrap();
        let manual_args = Args {
            input: Some(input.clone()),
            delay_correct: 0.75,
            rate_correct: -0.002,
            acel_correct: 1.0e-7,
            jerk_correct: 1.0e-10,
            snap_correct: 1.0e-13,
            ..Args::default()
        };
        run_mkcor(&manual_args).unwrap();
        let corrected = mkcor_output_path(&input).unwrap();

        for length in [1, 4, 8, 16] {
            for skip in [0, 2] {
                let original_args = Args {
                    length,
                    skip,
                    loop_: 100_000,
                    search: vec!["peak".to_string()],
                    add_plot: true,
                    cpu: 1,
                    ..manual_args.clone()
                };
                let corrected_args = Args {
                    input: Some(corrected.clone()),
                    delay_correct: 0.0,
                    rate_correct: 0.0,
                    acel_correct: 0.0,
                    jerk_correct: 0.0,
                    snap_correct: 0.0,
                    ..original_args.clone()
                };
                let expected = process_cor_file(&input, &original_args, &[], &[], true).unwrap();
                let actual = process_cor_file(&corrected, &corrected_args, &[], &[], true).unwrap();
                assert!(!expected.add_plot_phase.is_empty());
                assert_eq!(actual.add_plot_times, expected.add_plot_times);
                assert_eq!(actual.add_plot_amp, expected.add_plot_amp);
                assert_eq!(actual.add_plot_snr, expected.add_plot_snr);
                assert_eq!(actual.add_plot_phase, expected.add_plot_phase);
                assert_eq!(actual.add_plot_noise, expected.add_plot_noise);
                assert_eq!(actual.add_plot_res_delay, expected.add_plot_res_delay);
                assert_eq!(actual.add_plot_res_rate, expected.add_plot_res_rate);
            }
        }
    }
}
