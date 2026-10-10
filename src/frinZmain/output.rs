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
        { header.station1_position[0] },
        { header.station1_position[1] },
        { header.station1_position[2] },
        header.station2_name,
        header.station2_code,
        header.station2_clock_delay,
        header.station2_clock_rate,
        header.station2_clock_acel,
        header.station2_clock_jerk,
        header.station2_clock_snap,
        { header.station2_position[0] },
        { header.station2_position[1] },
        { header.station2_position[2] },
        header.source_name,
        { header.source_position_ra.to_degrees() },
        { header.source_position_dec.to_degrees() }
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
        output.push('_');
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
    let columns = result_columns(
        results,
        label,
        rfi_display,
        bandpass_applied,
        norm_acf_applied,
        false,
    );
    format!("  {}", format_stdout_columns(&columns))
}

pub fn format_freq_output(
    results: &AnalysisResults,
    label: &[&str],
    _args_length: i32,
    rfi_display: &str,
    bandpass_applied: bool,
    norm_acf_applied: bool,
) -> String {
    let columns = result_columns(
        results,
        label,
        rfi_display,
        bandpass_applied,
        norm_acf_applied,
        true,
    );
    format!("  {}", format_stdout_columns(&columns))
}

/// Nine significant decimal digits preserve every finite f32 on parsing as f32.
pub(crate) fn format_f32(value: f32) -> String {
    format!("{value:.8e}")
}

/// Seventeen significant decimal digits preserve every finite f64 on parsing as f64.
pub(crate) fn format_f64(value: f64) -> String {
    format!("{value:.16e}")
}

fn format_noise_level_percent(noise: f32) -> String {
    format_f32(noise * 100.0)
}

fn result_columns(
    results: &AnalysisResults,
    label: &[&str],
    rfi_display: &str,
    bandpass_applied: bool,
    norm_acf_applied: bool,
    frequency_mode: bool,
) -> Vec<String> {
    let mut columns = vec![
        results.yyyydddhhmmss1.clone(),
        sanitize_tsv_field(label.get(3).copied().unwrap_or("")),
        sanitize_tsv_field(&results.source_name),
        format_f32(results.length_f32),
    ];
    if frequency_mode {
        columns.extend([
            format_f32(results.freq_max_amp * 100.0),
            format_f32(results.freq_snr),
            format_f32(results.freq_phase),
            format_f32(results.freq_freq),
            format_noise_level_percent(results.freq_noise),
            format_f32(results.residual_rate),
        ]);
    } else {
        columns.extend([
            format_f32(results.delay_max_amp * 100.0),
            format_f32(results.delay_snr),
            format_f32(results.delay_phase),
            format_noise_level_percent(results.delay_noise),
            format_f32(results.residual_delay),
            format_f32(results.residual_rate),
        ]);
    }
    columns.extend([
        format_f32(results.ant1_az),
        format_f32(results.ant1_el),
        format_f32(results.ant1_hgt),
        format_f32(results.ant2_az),
        format_f32(results.ant2_el),
        format_f32(results.ant2_hgt),
        format_f64(results.mjd),
        sanitize_tsv_field(rfi_display),
        if bandpass_applied { "True" } else { "False" }.to_string(),
        if norm_acf_applied { "True" } else { "False" }.to_string(),
    ]);
    columns
}

fn format_stdout_columns(columns: &[String]) -> String {
    columns
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let width = match index {
                0 => 19,
                1 | 18 | 19 => 5,
                2 => 10,
                16 => 24,
                20 => 6,
                _ => 15,
            };
            if matches!(index, 0..=2 | 17..=19) {
                format!("{value:<width$}")
            } else {
                format!("{value:>width$}")
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
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
        "obsfreq",
    ]);
    units.extend([
        "az[deg]", "el[deg]", "hgt[m]", "az[deg]", "el[deg]", "hgt[m]", "-", "[MHz]", "[T/F]",
        "[T/F]", "[MHz]",
    ]);
    format!("{}\n{}\n", columns.join("\t"), units.join("\t"))
}

pub fn format_delay_tsv_header(station1_name: &str, station2_name: &str) -> String {
    format_tsv_header(station1_name, station2_name, false)
}

pub fn format_freq_tsv_header(station1_name: &str, station2_name: &str) -> String {
    format_tsv_header(station1_name, station2_name, true)
}

pub fn format_result_stdout_header(
    station1_name: &str,
    station2_name: &str,
    frequency_mode: bool,
) -> String {
    let header = format_tsv_header(station1_name, station2_name, frequency_mode);
    let lines: Vec<String> = header
        .lines()
        .map(|line| {
            let columns: Vec<String> = line
                .trim_start_matches("# ")
                .split('\t')
                .map(str::to_string)
                .collect();
            format!("# {}", format_stdout_columns(&columns))
        })
        .collect();
    let border = format!(
        "#{}",
        "*".repeat(lines.iter().map(String::len).max().unwrap_or(1) - 1)
    );
    format!("{border}\n{}\n{border}", lines.join("\n"))
}

pub fn format_delay_tsv_row(
    results: &AnalysisResults,
    label: &[&str],
    rfi_display: &str,
    bandpass_applied: bool,
    norm_acf_applied: bool,
    obsfreq_mhz: i64,
) -> String {
    let mut columns = result_columns(
        results,
        label,
        rfi_display,
        bandpass_applied,
        norm_acf_applied,
        false,
    );
    columns[0] = format_tsv_epoch(&columns[0]);
    columns.push(obsfreq_mhz.to_string());
    columns.join("\t")
}

pub fn format_freq_tsv_row(
    results: &AnalysisResults,
    label: &[&str],
    rfi_display: &str,
    bandpass_applied: bool,
    norm_acf_applied: bool,
    obsfreq_mhz: i64,
) -> String {
    let mut columns = result_columns(
        results,
        label,
        rfi_display,
        bandpass_applied,
        norm_acf_applied,
        true,
    );
    columns[0] = format_tsv_epoch(&columns[0]);
    columns.push(obsfreq_mhz.to_string());
    columns.join("\t")
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
        format_delay_output, format_delay_tsv_header, format_delay_tsv_row, format_f32, format_f64,
        format_freq_output, format_freq_tsv_header, format_freq_tsv_row, format_tsv_epoch,
        insert_product_before_processing_suffixes, AnalysisResults, C32,
    };
    use ndarray::Array1;

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
            assert_eq!(lines[0].split('\t').count(), 21);
            assert_eq!(lines[1].split('\t').count(), 21);
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
    fn finite_float_values_survive_text_round_trip() {
        // Include both signs of zero, subnormals, normal boundaries, adjacent
        // values around one, and maximum finite values.
        for bits in [
            0,
            0x8000_0000,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x3f80_0001,
            0x3f80_0002,
            0x7f7f_ffff,
            0xff7f_ffff,
        ] {
            let value = f32::from_bits(bits);
            assert_eq!(format_f32(value).parse::<f32>().unwrap().to_bits(), bits);
        }
        for bits in [
            0,
            0x8000_0000_0000_0000,
            1,
            0x8000_0000_0000_0001,
            0x000f_ffff_ffff_ffff,
            0x0010_0000_0000_0000,
            0x3ff0_0000_0000_0001,
            0x7fef_ffff_ffff_ffff,
            0xffef_ffff_ffff_ffff,
        ] {
            let value = f64::from_bits(bits);
            assert_eq!(format_f64(value).parse::<f64>().unwrap().to_bits(), bits);
        }
        // Deterministically sample mantissas and exponents across both types.
        let mut bits = 0x1234_5678_9abc_def0u64;
        for _ in 0..10_000 {
            bits = bits.wrapping_mul(6364136223846793005).wrapping_add(1);
            let value = f32::from_bits(bits as u32);
            if value.is_finite() {
                assert_eq!(
                    format_f32(value).parse::<f32>().unwrap().to_bits(),
                    value.to_bits()
                );
            }
            let value = f64::from_bits(bits);
            if value.is_finite() {
                assert_eq!(format_f64(value).parse::<f64>().unwrap().to_bits(), bits);
            }
        }
    }

    #[test]
    fn stdout_and_tsv_preserve_every_reported_numeric_column() {
        let results = AnalysisResults {
            yyyydddhhmmss1: "2021/156 18:35:00.123456789".to_string(),
            source_name: "AEAqr".to_string(),
            length_f32: 10.251234,
            ant1_az: 170.79321,
            ant1_el: 54.559128,
            ant1_hgt: 165.70813,
            ant2_az: 170.79413,
            ant2_el: 54.559227,
            ant2_hgt: 166.61014,
            mjd: 59370.774306984516,
            delay_range: Array1::zeros(0),
            visibility: Array1::zeros(0),
            delay_rate: Array1::zeros(0),
            delay_peak_complex: C32::new(0.0, 0.0),
            delay_max_amp: 0.00001613935,
            delay_phase: 52.323128,
            delay_snr: 7.929819,
            delay_noise: 0.00000203538,
            residual_delay: 1.2345678,
            corrected_delay: 0.0,
            delay_offset: 0.0,
            freq_max_amp: 0.000019723456,
            freq_phase: -27.765129,
            freq_freq: 123.45678,
            freq_snr: 8.712345,
            freq_noise: 0.000002345678,
            freq_rate: Array1::zeros(0),
            freq_rate_spectrum: Array1::zeros(0),
            freq_range: Array1::zeros(0),
            freq_max_freq: 123.45678,
            residual_rate: -0.0,
            corrected_rate: 0.0,
            rate_offset: 0.0,
            corrected_acel: 0.0,
            corrected_jerk: 0.0,
            corrected_snap: 0.0,
            rate_range: Vec::new(),
            l_coord: 0.0,
            m_coord: 0.0,
        };
        let label = ["", "", "", "all"];
        for frequency in [false, true] {
            let (tsv, stdout, mode_values) = if frequency {
                (
                    format_freq_tsv_row(&results, &label, "-", false, false, 8192),
                    format_freq_output(&results, &label, 10, "-", false, false),
                    [
                        results.freq_max_amp * 100.0,
                        results.freq_snr,
                        results.freq_phase,
                        results.freq_freq,
                        results.freq_noise * 100.0,
                        results.residual_rate,
                    ],
                )
            } else {
                (
                    format_delay_tsv_row(&results, &label, "-", false, false, 8192),
                    format_delay_output(&results, &label, 10, "-", false, false),
                    [
                        results.delay_max_amp * 100.0,
                        results.delay_snr,
                        results.delay_phase,
                        results.delay_noise * 100.0,
                        results.residual_delay,
                        results.residual_rate,
                    ],
                )
            };
            let fields: Vec<&str> = tsv.split('\t').collect();
            assert_eq!(fields.len(), 21);
            assert_eq!(fields[0], "2021/156T18:35:00.123456789");
            let expected: Vec<f32> = [results.length_f32]
                .into_iter()
                .chain(mode_values)
                .chain([
                    results.ant1_az,
                    results.ant1_el,
                    results.ant1_hgt,
                    results.ant2_az,
                    results.ant2_el,
                    results.ant2_hgt,
                ])
                .collect();
            for (text, original) in fields[3..16].iter().zip(expected) {
                assert_eq!(text.parse::<f32>().unwrap().to_bits(), original.to_bits());
            }
            assert_eq!(
                fields[16].parse::<f64>().unwrap().to_bits(),
                results.mjd.to_bits()
            );
            assert_eq!(fields[20], "8192");
            let stdout_fields: Vec<&str> = stdout.split_whitespace().collect();
            assert_eq!(&stdout_fields[4..18], &fields[3..17]);
        }
    }
}
