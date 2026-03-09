use byteorder::{LittleEndian, WriteBytesExt};
use chrono::{DateTime, Utc};
use num_complex::Complex;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::analysis::AnalysisResults;
use crate::header::CorHeader;

type C32 = Complex<f32>;

pub fn output_header_info(
    header: &CorHeader,
    output_dir: &Path,
    basename: &str,
) -> io::Result<String> {
    let header_file_path = output_dir.join(format!("{}_header.txt", basename));
    let header_info = format!(
        "### header region information

        [Header]

        Magic Word           = {:?}
        Header Version       = {}
        Software Version     = {}
        Sampling Frequency   = {} MHz
        Observing Frequency  = {} MHz
        FFT Point            = {}
        Number of Sector     = {}
        Bandwidth            = {} MHz
        Resolution Bandwidth = {} MHz

        [Station1]
            Name     = {}
            Code     = {}
            Clock Delay = {} s
            Clock Rate  = {} s/s
            Clock Acel  = {} s/s**2
            Clock Jerk  = {} s/s**3
            Clock Snap  = {} s/s**4
            Position = ({}, {}, {}) [m], geocentric coordinate

        [Station2]
            Name     = {}
            Code     = {}
            Clock Delay = {} s
            Clock Rate  = {} s/s
            Clock Acel  = {} s/s**2
            Clock Jerk  = {} s/s**3
            Clock Snap  = {} s/s**4
            Position = ({}, {}, {}) [m], geocentric coordinate

        [Source]
            Name       = {}
            Coordinate = ({}, {}) J2000
",
        header.magic_word,
        header.header_version,
        header.software_version,
        header.sampling_speed as f32 / 1e6,
        header.observing_frequency as f32 / 1e6,
        header.fft_point,
        header.number_of_sector,
        header.sampling_speed as f32 / 2.0 / 1e6,
        (header.sampling_speed as f32 / 2.0 / 1e6) / header.fft_point as f32 * 2.0,
        header.station1_name,
        header.station1_code,
        header.station1_clock_delay,
        header.station1_clock_rate,
        header.station1_clock_acel,
        header.station1_clock_jerk,
        header.station1_clock_snap,
        header.station1_position[0] as f64,
        header.station1_position[1] as f64,
        header.station1_position[2] as f64,
        header.station2_name,
        header.station2_code,
        header.station2_clock_delay,
        header.station2_clock_rate,
        header.station2_clock_acel,
        header.station2_clock_jerk,
        header.station2_clock_snap,
        header.station2_position[0] as f64,
        header.station2_position[1] as f64,
        header.station2_position[2] as f64,
        header.source_name,
        header.source_position_ra.to_degrees() as f64,
        header.source_position_dec.to_degrees() as f64
    );
    if !header_file_path.exists() {
        std::fs::write(header_file_path, &header_info)?;
    }
    Ok(header_info)
}

pub fn generate_output_names(
    header: &CorHeader,
    obs_time: &DateTime<Utc>,
    label: &[&str],
    is_rfi_filtered: bool,
    is_frequency_mode: bool,
    is_bandpass_corrected: bool,
    length: i32,
) -> String {
    let yyyydddhhmmss2 = obs_time.format("%Y%j%H%M%S").to_string();
    let rfi_suffix = if is_rfi_filtered { "_rfi" } else { "" };
    let bp_suffix = if is_bandpass_corrected { "_bp" } else { "" };
    let _mode_suffix = if is_frequency_mode { "_freq" } else { "_time" };
    let observing_band = if (6600.0..=7112.0).contains(&(header.observing_frequency as f32 / 1e6)) {
        "c"
    } else if (8192.0..=8704.0).contains(&(header.observing_frequency as f32 / 1e6)) {
        "x"
    } else if (11923.0..=12435.0).contains(&(header.observing_frequency as f32 / 1e6)) {
        "ku"
    } else {
        "n"
    };
    let label_segment = label.get(3).copied().unwrap_or("");

    let base = format!(
        "{}_{}_{}_{}_{}_len{}s{}{}",
        header.station1_name,
        header.station2_name,
        yyyydddhhmmss2,
        label_segment,
        observing_band,
        length,
        rfi_suffix,
        bp_suffix
    );
    base
}

pub fn format_delay_output(
    results: &AnalysisResults,
    label: &[&str],
    args_length: i32,
    rfi_display: &str,
    bandpass_applied: bool,
) -> String {
    let display_length = if args_length != 0 {
        args_length as f32
    } else {
        results.length_f32.ceil()
    };
    let label_segment = label.get(3).copied().unwrap_or("");
    format!(
        " {}   {:<5}  {:<10} {:<8.2} {:<3.6} {:>7.1} {:>+10.3}  {:>10.6}  {:>+9.8}   {:>+4.8}   {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>12.5}   {:<15} {:<5}",
        results.yyyydddhhmmss1,
        label_segment,
        results.source_name,
        display_length,
        results.delay_max_amp * 100.0,
        results.delay_snr,
        results.delay_phase,
        results.delay_noise * 100.0,
        results.residual_delay,
        results.residual_rate,
        results.ant1_az,
        results.ant1_el,
        results.ant1_hgt,
        results.ant2_az,
        results.ant2_el,
        results.ant2_hgt,
        results.mjd,
        rfi_display,
        if bandpass_applied { "True" } else { "False" },
        //results.l_coord,
        //results.m_coord
    )
}

pub fn format_freq_output(
    results: &AnalysisResults,
    label: &[&str],
    args_length: i32,
    rfi_display: &str,
    bandpass_applied: bool,
) -> String {
    let display_length = if args_length != 0 {
        args_length as f32
    } else {
        results.length_f32.ceil()
    };
    let label_segment = label.get(3).copied().unwrap_or("");
    format!(
        " {}   {:<5}  {:<10} {:<8.2} {:<8.6}  {:>7.1}   {:>+10.3} {:>+12.7} {:>10.6} {:>+10.6} {:>7.3} {:>7.3} {:>7.3}  {:>7.3} {:>7.3} {:>7.3} {:>12.5}   {:<15} {:<5}",
        results.yyyydddhhmmss1,
        label_segment,
        results.source_name,
        display_length,
        results.freq_max_amp * 100.0,
        results.freq_snr,
        results.freq_phase,
        results.freq_freq,
        results.freq_noise * 100.0,
        results.residual_rate,
        results.ant1_az,
        results.ant1_el,
        results.ant1_hgt,
        results.ant2_az,
        results.ant2_el,
        results.ant2_hgt,
        results.mjd,
        rfi_display,
        if bandpass_applied { "True" } else { "False" },
        //results.l_coord,
        //results.m_coord
    )
}

pub fn write_phase_corrected_spectrum_binary(
    file_path: &Path,
    file_header: &[u8],
    sector_headers: &[Vec<u8>],
    calibrated_spectra: &[Vec<C32>],
) -> io::Result<()> {
    let file = File::create(file_path)?;
    let mut writer = BufWriter::new(file);

    // 1. ファイルヘッダー (256 byte) を書き込む
    writer.write_all(file_header)?;

    // 2. 各セクターのヘッダーと較正済みデータを書き込む
    for (i, spectrum) in calibrated_spectra.iter().enumerate() {
        // このセクターの生の128バイトヘッダーを書き込む
        writer.write_all(&sector_headers[i])?;

        // 較正済みの複素スペクトルの実部と虚部 (各4 byte) を交互に書き込む
        for c in spectrum {
            writer.write_f32::<LittleEndian>(c.re)?;
            writer.write_f32::<LittleEndian>(c.im)?;
        }
    }
    Ok(())
}

pub fn write_add_plot_data_to_file(
    output_dir: &Path,
    base_filename: &str,
    elapsed_times: &[f32],
    amp: &[f32],
    snr: &[f32],
    phase: &[f32],
    noise: &[f32],
    res_delay: &[f32],
    res_rate: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let output_file_path = output_dir.join(format!("{}_add_plot_data.tsv", base_filename));
    let file = File::create(&output_file_path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(
        writer,
        "#Elapsed Time [s]\tAmplitude [%]\tSNR\tPhase [deg]\tNoise Level [%]\tRes Delay [samp]\tRes Rate [Hz]"
    )?;

    // Write data
    for i in 0..elapsed_times.len() {
        writeln!(
            writer,
            "{:.3}\t{:.6}\t{:.2}\t{:.3}\t{:.6}\t{:.6}\t{:.6e}",
            elapsed_times[i], amp[i], snr[i], phase[i], noise[i], res_delay[i], res_rate[i]
        )?;
    }

    writer.flush()?;
    //println!("Add plot data written to: {:?}", output_file_path);
    Ok(())
}
