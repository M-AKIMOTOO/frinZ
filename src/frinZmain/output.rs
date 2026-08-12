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
    let header_file_path = output_dir.join(format!(
        "{}.txt",
        insert_product_before_processing_suffixes(basename, "header")
    ));
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
    let (label_segment, is_contamisubt) = label_segment
        .strip_suffix("_contamisubt")
        .map_or((label_segment, false), |label| (label, true));

    let mut base = format!(
        "{}_{}_{}_{}_{}_len{}s",
        header.station1_name,
        header.station2_name,
        yyyydddhhmmss2,
        label_segment,
        observing_band,
        length
    );
    append_processing_suffixes(
        &mut base,
        is_bandpass_corrected,
        is_rfi_filtered,
        is_contamisubt,
        false,
        false,
    );
    base
}

fn append_processing_suffixes(
    output: &mut String,
    bandpass: bool,
    rfi: bool,
    contamisubt: bool,
    spike34: bool,
    inbeam: bool,
) {
    if bandpass {
        output.push_str("_bp");
    }
    if rfi {
        output.push_str("_rfi");
    }
    if contamisubt {
        output.push_str("_contamisubt");
    }
    if spike34 {
        output.push_str("_spike34");
    }
    if inbeam {
        output.push_str("_inbeam");
    }
}

/// Inserts an analysis-product name before processing suffixes.
///
/// Processing suffixes always use the stable order
/// `_bp_rfi_contamisubt_spike34_inbeam`, independent of their order in `base`.
pub fn insert_product_before_processing_suffixes(base: &str, product: &str) -> String {
    let mut core = base;
    let mut bandpass = false;
    let mut rfi = false;
    let mut contamisubt = false;
    let mut spike34 = false;
    let mut inbeam = false;

    loop {
        if let Some(value) = core.strip_suffix("_inbeam") {
            core = value;
            inbeam = true;
        } else if let Some(value) = core.strip_suffix("_spike34") {
            core = value;
            spike34 = true;
        } else if let Some(value) = core.strip_suffix("_contamisubt") {
            core = value;
            contamisubt = true;
        } else if let Some(value) = core.strip_suffix("_rfi") {
            core = value;
            rfi = true;
        } else if let Some(value) = core.strip_suffix("_bp") {
            core = value;
            bandpass = true;
        } else {
            break;
        }
    }

    let mut output = core.to_string();
    let product = product.trim_matches('_');
    if !product.is_empty() && core != product && !core.ends_with(&format!("_{product}")) {
        output.push_str("_");
        output.push_str(product);
    }
    append_processing_suffixes(&mut output, bandpass, rfi, contamisubt, spike34, inbeam);
    output
}

pub fn format_delay_output(
    results: &AnalysisResults,
    label: &[&str],
    _args_length: i32,
    rfi_display: &str,
    bandpass_applied: bool,
    norm_acf_applied: bool,
) -> String {
    let display_length = format_output_length(results.length_f32);
    let noise_level = format_noise_level_percent(results.delay_noise);
    let label_segment = label.get(3).copied().unwrap_or("");
    format!(
        " {}   {:<5}  {:<10} {:<8} {:<3.6} {:>7.1} {:>+10.3}  {:>10}  {:>+9.8}   {:>+4.8}   {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>12.5}   {:<15} {:<5} {:<5}",
        results.yyyydddhhmmss1,
        label_segment,
        results.source_name,
        display_length,
        results.delay_max_amp * 100.0,
        results.delay_snr,
        results.delay_phase,
        noise_level,
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
        if norm_acf_applied { "True" } else { "False" },
        //results.l_coord,
        //results.m_coord
    )
}

pub fn format_freq_output(
    results: &AnalysisResults,
    label: &[&str],
    _args_length: i32,
    rfi_display: &str,
    bandpass_applied: bool,
    norm_acf_applied: bool,
) -> String {
    let display_length = format_output_length(results.length_f32);
    let noise_level = format_noise_level_percent(results.freq_noise);
    let label_segment = label.get(3).copied().unwrap_or("");
    format!(
        " {}   {:<5}  {:<10} {:<8} {:<8.6}  {:>7.1}   {:>+10.3} {:>+12.7} {:>10} {:>+10.6} {:>7.3} {:>7.3} {:>7.3}  {:>7.3} {:>7.3} {:>7.3} {:>12.5}   {:<15} {:<5} {:<5}",
        results.yyyydddhhmmss1,
        label_segment,
        results.source_name,
        display_length,
        results.freq_max_amp * 100.0,
        results.freq_snr,
        results.freq_phase,
        results.freq_freq,
        noise_level,
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
        if norm_acf_applied { "True" } else { "False" },
        //results.l_coord,
        //results.m_coord
    )
}

fn format_output_length(length: f32) -> String {
    if length < 1.0 {
        format!("{length:.5e}")
    } else {
        format!("{length:.1}")
    }
}

fn format_noise_level_percent(noise: f32) -> String {
    format!("{:.5e}", noise * 100.0)
}

fn format_tsv_epoch(value: &str) -> String {
    value.replace(' ', "T")
}

fn sanitize_tsv_field(value: &str) -> String {
    value
        .chars()
        .map(|character| match character {
            '\t' | '\n' | '\r' => ' ',
            other => other,
        })
        .collect()
}

fn format_tsv_header(station1_name: &str, station2_name: &str, frequency_mode: bool) -> String {
    let station1_label = format!("{}-azel", station1_name.trim());
    let station2_label = format!("{}-azel", station2_name.trim());
    let mut columns = vec![
        "# Epoch", "Label", "Source", "Length", "Amp", "SNR", "Phase",
    ];
    let mut units = vec!["# -", "-", "-", "[s]", "[%]", "-", "[deg]"];
    if frequency_mode {
        columns.extend(["Frequency", "Noise-level", "Res-Rate"]);
        units.extend(["[MHz]", "1-sigma[%]", "[Hz]"]);
    } else {
        columns.extend(["Noise-level", "Res-Delay", "Res-Rate"]);
        units.extend(["1-sigma[%]", "[sample]", "[Hz]"]);
    }
    columns.extend([
        station1_label.as_str(),
        "",
        "",
        station2_label.as_str(),
        "",
        "",
        "MJD",
        "RFI",
        "BP",
        "ACF",
    ]);
    units.extend([
        "az[deg]", "el[deg]", "hgt[m]", "az[deg]", "el[deg]", "hgt[m]", "-", "[MHz]", "[T/F]",
        "[T/F]",
    ]);
    format!("{}\n{}\n", columns.join("\t"), units.join("\t"))
}

pub fn format_delay_tsv_header(station1_name: &str, station2_name: &str) -> String {
    format_tsv_header(station1_name, station2_name, false)
}

pub fn format_freq_tsv_header(station1_name: &str, station2_name: &str) -> String {
    format_tsv_header(station1_name, station2_name, true)
}

pub fn format_delay_tsv_row(
    results: &AnalysisResults,
    label: &[&str],
    rfi_display: &str,
    bandpass_applied: bool,
    norm_acf_applied: bool,
) -> String {
    let label_segment = label.get(3).copied().unwrap_or("");
    vec![
        format_tsv_epoch(&results.yyyydddhhmmss1),
        sanitize_tsv_field(label_segment),
        sanitize_tsv_field(&results.source_name),
        format_output_length(results.length_f32),
        format!("{:.6}", results.delay_max_amp * 100.0),
        format!("{:.1}", results.delay_snr),
        format!("{:.3}", results.delay_phase),
        format_noise_level_percent(results.delay_noise),
        format!("{:.8}", results.residual_delay),
        format!("{:.8}", results.residual_rate),
        format!("{:.3}", results.ant1_az),
        format!("{:.3}", results.ant1_el),
        format!("{:.3}", results.ant1_hgt),
        format!("{:.3}", results.ant2_az),
        format!("{:.3}", results.ant2_el),
        format!("{:.3}", results.ant2_hgt),
        format!("{:.5}", results.mjd),
        sanitize_tsv_field(rfi_display),
        if bandpass_applied { "True" } else { "False" }.to_string(),
        if norm_acf_applied { "True" } else { "False" }.to_string(),
    ]
    .join("\t")
}

pub fn format_freq_tsv_row(
    results: &AnalysisResults,
    label: &[&str],
    rfi_display: &str,
    bandpass_applied: bool,
    norm_acf_applied: bool,
) -> String {
    let label_segment = label.get(3).copied().unwrap_or("");
    vec![
        format_tsv_epoch(&results.yyyydddhhmmss1),
        sanitize_tsv_field(label_segment),
        sanitize_tsv_field(&results.source_name),
        format_output_length(results.length_f32),
        format!("{:.6}", results.freq_max_amp * 100.0),
        format!("{:.1}", results.freq_snr),
        format!("{:.3}", results.freq_phase),
        format!("{:.7}", results.freq_freq),
        format_noise_level_percent(results.freq_noise),
        format!("{:.6}", results.residual_rate),
        format!("{:.3}", results.ant1_az),
        format!("{:.3}", results.ant1_el),
        format!("{:.3}", results.ant1_hgt),
        format!("{:.3}", results.ant2_az),
        format!("{:.3}", results.ant2_el),
        format!("{:.3}", results.ant2_hgt),
        format!("{:.5}", results.mjd),
        sanitize_tsv_field(rfi_display),
        if bandpass_applied { "True" } else { "False" }.to_string(),
        if norm_acf_applied { "True" } else { "False" }.to_string(),
    ]
    .join("\t")
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

#[cfg(test)]
mod filename_tests {
    use super::{
        format_delay_tsv_header, format_freq_tsv_header, format_noise_level_percent,
        format_output_length, format_tsv_epoch, insert_product_before_processing_suffixes,
    };

    #[test]
    fn product_precedes_bandpass_suffix() {
        assert_eq!(
            insert_product_before_processing_suffixes("observation_bp", "delay_rate_search"),
            "observation_delay_rate_search_bp"
        );
    }

    #[test]
    fn inband_width_product_precedes_processing_suffix() {
        assert_eq!(
            insert_product_before_processing_suffixes("observation_bp", "inband256MHz"),
            "observation_inband256MHz_bp"
        );
    }

    #[test]
    fn spike34_suffix_stays_after_product() {
        assert_eq!(
            insert_product_before_processing_suffixes(
                "observation_bp_spike34",
                "delay_rate_search"
            ),
            "observation_delay_rate_search_bp_spike34"
        );
    }

    #[test]
    fn processing_suffixes_are_canonicalized() {
        assert_eq!(
            insert_product_before_processing_suffixes("observation_rfi_bp_contamisubt", "cumulate"),
            "observation_cumulate_bp_rfi_contamisubt"
        );
    }

    #[test]
    fn inbeam_remains_the_final_suffix() {
        assert_eq!(
            insert_product_before_processing_suffixes(
                "observation_contamisubt_rfi_bp_inbeam",
                "wwz_amp"
            ),
            "observation_wwz_amp_bp_rfi_contamisubt_inbeam"
        );
    }
    #[test]
    fn tsv_headers_are_two_commented_tab_separated_rows() {
        for header in [
            format_delay_tsv_header("YAMAGU32", "YAMAGU34"),
            format_freq_tsv_header("YAMAGU32", "YAMAGU34"),
        ] {
            let lines: Vec<&str> = header.lines().collect();
            assert_eq!(lines.len(), 2);
            assert!(lines[0].starts_with('#'));
            assert!(lines[1].starts_with('#'));
            assert!(!header.contains('*'));
            assert_eq!(lines[0].split('\t').count(), 20);
            assert_eq!(lines[1].split('\t').count(), 20);
            assert_eq!(lines[1].split('\t').next(), Some("# -"));
            assert!(lines[0].contains("YAMAGU32-azel"));
            assert!(lines[0].contains("YAMAGU34-azel"));
        }
    }

    #[test]
    fn tsv_epoch_uses_single_whitespace_free_field() {
        assert_eq!(format_tsv_epoch("2021/156 18:35:00"), "2021/156T18:35:00");
    }

    #[test]
    fn result_numeric_formats_cover_subsecond_lengths() {
        assert_eq!(format_output_length(0.125), "1.25000e-1");
        assert_eq!(format_output_length(1.0), "1.0");
        assert_eq!(format_output_length(10.25), "10.2");
        assert_eq!(format_noise_level_percent(0.000_012_345_6), "1.23456e-3");
    }
}
