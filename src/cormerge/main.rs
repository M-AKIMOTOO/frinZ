// M.AKIMOTO with Gemini
// 2025/08/18
// cargo run -- --source <source_name> <入力ファイル1> <入力ファイル2> ...

use byteorder::{LittleEndian, ReadBytesExt};
use clap::Parser;
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::str;
use std::time::{SystemTime, UNIX_EPOCH};

// --- Header Structures and Parsing (from frinZmain/header.rs) ---
#[derive(Debug, Default)]
pub struct CorHeader {
    pub magic_word: [u8; 4],
    pub header_version: i32,
    pub software_version: i32,
    pub sampling_speed: i32,
    pub observing_frequency: f64,
    pub fft_point: i32,
    pub number_of_sector: i32,
    pub station1_name: String,
    pub station1_code: String,
    pub station1_position: [f64; 3],
    pub station2_name: String,
    pub station2_code: String,
    pub station2_position: [f64; 3],
    pub source_name: String,
    pub source_position_ra: f64,
    pub source_position_dec: f64,
    pub station1_clock_delay: f64,
    pub station1_clock_rate: f64,
    pub station1_clock_acel: f64,
    pub station1_clock_jerk: f64,
    pub station1_clock_snap: f64,
    pub station2_clock_delay: f64,
    pub station2_clock_rate: f64,
    pub station2_clock_acel: f64,
    pub station2_clock_jerk: f64,
    pub station2_clock_snap: f64,
}

pub fn parse_header(cursor: &mut Cursor<&[u8]>) -> io::Result<CorHeader> {
    let mut header = CorHeader::default();
    cursor.set_position(0);

    // Line 0
    cursor.read_exact(&mut header.magic_word)?;
    header.header_version = cursor.read_i32::<LittleEndian>()?;
    header.software_version = cursor.read_i32::<LittleEndian>()?;
    header.sampling_speed = cursor.read_i32::<LittleEndian>()?;

    // Line 1
    header.observing_frequency = cursor.read_f64::<LittleEndian>()?;
    header.fft_point = cursor.read_i32::<LittleEndian>()?;
    header.number_of_sector = cursor.read_i32::<LittleEndian>()?;

    // Line 2: Station 1 Name
    let mut name_buf = [0u8; 8];
    cursor.read_exact(&mut name_buf)?;
    header.station1_name = String::from_utf8_lossy(&name_buf)
        .trim_end_matches('\0')
        .to_string();
    cursor.set_position(cursor.position() + 8); // Skip padding

    // Line 3: Station 1 Pos X, Y
    header.station1_position[0] = cursor.read_f64::<LittleEndian>()?;
    header.station1_position[1] = cursor.read_f64::<LittleEndian>()?;

    // Line 4: Station 1 Pos Z, Code
    header.station1_position[2] = cursor.read_f64::<LittleEndian>()?;
    let mut code_buf = [0u8; 1];
    cursor.read_exact(&mut code_buf)?;
    header.station1_code = String::from_utf8_lossy(&code_buf).to_string();
    cursor.set_position(cursor.position() + 7); // Skip padding

    // Line 5: Station 2 Name
    cursor.read_exact(&mut name_buf)?;
    header.station2_name = String::from_utf8_lossy(&name_buf)
        .trim_end_matches('\0')
        .to_string();
    cursor.set_position(cursor.position() + 8); // Skip padding

    // Line 6: Station 2 Pos X, Y
    header.station2_position[0] = cursor.read_f64::<LittleEndian>()?;
    header.station2_position[1] = cursor.read_f64::<LittleEndian>()?;

    // Line 7: Station 2 Pos Z, Code
    header.station2_position[2] = cursor.read_f64::<LittleEndian>()?;
    cursor.read_exact(&mut code_buf)?;
    header.station2_code = String::from_utf8_lossy(&code_buf).to_string();
    cursor.set_position(cursor.position() + 7); // Skip padding

    // Line 8: Source Name (16 bytes)
    let mut source_name_buf = [0u8; 16];
    cursor.read_exact(&mut source_name_buf)?;
    header.source_name = String::from_utf8_lossy(&source_name_buf)
        .trim_end_matches('\0')
        .to_string();

    // Line 9: Source Pos RA, Dec
    header.source_position_ra = cursor.read_f64::<LittleEndian>()?;
    header.source_position_dec = cursor.read_f64::<LittleEndian>()?;

    cursor.set_position(168);
    header.station1_clock_delay = cursor.read_f64::<LittleEndian>()?;
    header.station1_clock_rate = cursor.read_f64::<LittleEndian>()?;
    header.station1_clock_acel = cursor.read_f64::<LittleEndian>()?;
    header.station1_clock_jerk = cursor.read_f64::<LittleEndian>()?;
    header.station1_clock_snap = cursor.read_f64::<LittleEndian>()?;

    cursor.set_position(216);
    header.station2_clock_delay = cursor.read_f64::<LittleEndian>()?;
    header.station2_clock_rate = cursor.read_f64::<LittleEndian>()?;
    header.station2_clock_acel = cursor.read_f64::<LittleEndian>()?;
    header.station2_clock_jerk = cursor.read_f64::<LittleEndian>()?;
    header.station2_clock_snap = cursor.read_f64::<LittleEndian>()?;

    cursor.set_position(256);
    Ok(header)
}

fn get_csv_header() -> String {
    "#FileName,MagicWord,Header,Software,MHz,MHz,FFT,PP,BW(MHz),RBW(MHz),Name,Code,Delay(s),Rate(s/s),Acel(s/s^2),Jerk(s/s^3),Snap(s/s^4),X(m),Y(m),Z(m),Name,Code,Delay(s),Rate(s/s),Acel(s/s^2),Jerk(s/s^3),Snap(s/s^4),X(m),Y(m),Z(m),Name,RA(deg),Dec(deg)".to_string()
}

fn format_header_as_csv_row(header: &CorHeader, filename: &Path) -> String {
    //let magic_word_str = String::from_utf8_lossy(&header.magic_word).trim_end_matches('\0').to_string();
    let basename = filename.file_name().and_then(|s| s.to_str()).unwrap_or("");

    format!(
        "{},3ea2f983,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.5},{:.5}",
        basename,
        //magic_word_str,
        header.header_version,
        header.software_version,
        header.sampling_speed as f64 / 1e6,
        header.observing_frequency / 1e6,
        header.fft_point,
        header.number_of_sector,
        header.sampling_speed as f64 / 2.0 / 1e6,
        (header.sampling_speed as f64 / 2.0 / 1e6) / header.fft_point as f64 * 2.0,
        header.station1_name,
        header.station1_code,
        header.station1_clock_delay,
        header.station1_clock_rate,
        header.station1_clock_acel,
        header.station1_clock_jerk,
        header.station1_clock_snap,
        header.station1_position[0],
        header.station1_position[1],
        header.station1_position[2],
        header.station2_name,
        header.station2_code,
        header.station2_clock_delay,
        header.station2_clock_rate,
        header.station2_clock_acel,
        header.station2_clock_jerk,
        header.station2_clock_snap,
        header.station2_position[0],
        header.station2_position[1],
        header.station2_position[2],
        header.source_name,
        header.source_position_ra.to_degrees(),
        header.source_position_dec.to_degrees()
    )
}

// --- 定数定義 ---
const OFFSET_FOR_SUBSEQUENT_FILES: u64 = 256;
const VALUE_OFFSET: u64 = 28;
const SIGNATURE_OFFSET: u64 = 248;
const SIGNATURE_STRING: &str = "cormerge";
const SIGNATURE_LEN: usize = SIGNATURE_STRING.len();
const SIGNATURE_BUFFER_LEN: usize = 8; // "cormerge"
const HEADER_LEN: usize = 256;

const SOURCE_NAME_OFFSET: u64 = 128;
const SOURCE_NAME_LEN: usize = 16;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None, arg_required_else_help = true)]
struct Cli {
    /// Set a source name to filter files
    #[arg(long, required = true)]
    source: String,

    /// Two or more input .cor files to concatenate
    #[arg(long, required = true, num_args = 2..)]
    cor: Vec<PathBuf>,
}

fn copy_prefix<const N: usize>(source: &[u8]) -> Option<[u8; N]> {
    if source.len() < N {
        return None;
    }
    let mut buffer = [0u8; N];
    buffer.copy_from_slice(&source[..N]);
    Some(buffer)
}

fn read_file_prefix(file: &mut File, buffer: &mut [u8]) -> Result<usize, io::Error> {
    let mut total_read = 0;
    while total_read < buffer.len() {
        let bytes_read = file.read(&mut buffer[total_read..])?;
        if bytes_read == 0 {
            break;
        }
        total_read += bytes_read;
    }
    Ok(total_read)
}

fn source_name_matches(header_bytes: &[u8], required_source: &str) -> bool {
    let start = SOURCE_NAME_OFFSET as usize;
    let end = start + SOURCE_NAME_LEN;
    if header_bytes.len() < end {
        return false;
    }

    let name_from_file = header_bytes[start..end]
        .split(|&b| b == 0)
        .next()
        .unwrap_or(&[]);
    str::from_utf8(name_from_file).unwrap_or("") == required_source
}

fn has_cormerge_signature(header_bytes: &[u8]) -> bool {
    let start = SIGNATURE_OFFSET as usize;
    let end = start + SIGNATURE_BUFFER_LEN;
    if header_bytes.len() < end {
        return false;
    }

    let mut expected_signature = [0u8; SIGNATURE_BUFFER_LEN];
    expected_signature[..SIGNATURE_LEN].copy_from_slice(SIGNATURE_STRING.as_bytes());
    header_bytes[start..end] == expected_signature
}

fn read_value_from_header(header_bytes: &[u8]) -> u32 {
    let start = VALUE_OFFSET as usize;
    let end = start + 4;
    if header_bytes.len() < end {
        return 0;
    }
    u32::from_le_bytes(header_bytes[start..end].try_into().unwrap())
}

fn read_signature_from_header(header_bytes: &[u8]) -> [u8; SIGNATURE_BUFFER_LEN] {
    let start = SIGNATURE_OFFSET as usize;
    let end = start + SIGNATURE_BUFFER_LEN;
    if header_bytes.len() < end {
        return [0u8; SIGNATURE_BUFFER_LEN];
    }
    copy_prefix::<SIGNATURE_BUFFER_LEN>(&header_bytes[start..end]).unwrap()
}

fn generate_temp_stem() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(".cormerge-{}-{}", std::process::id(), nanos)
}

/// ファイル名から拡張子を除き、アンダースコアで分割して指定番目の要素を取得する
fn get_split_element(original_filename: &Path, target_index: usize) -> Option<String> {
    let file_stem = original_filename.file_stem()?.to_str()?;
    file_stem.split('_').nth(target_index).map(String::from)
}

fn trailing_label(parts: &[&str], start_index: usize) -> Option<String> {
    if parts.len() <= start_index {
        return None;
    }
    let label = parts[start_index..].join("_");
    if label.is_empty() {
        None
    } else {
        Some(label)
    }
}

/// 出力ファイル名を生成する
fn generate_output_filename(input_files: &[PathBuf]) -> Result<PathBuf, Box<dyn Error>> {
    if input_files.is_empty() {
        return Err("入力ファイルがありません.".into());
    }

    let first_filename = &input_files[0];
    const TARGET_INDEX: usize = 2; // 3番目の要素

    let file_stem = first_filename
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("ファイル名からステムを取得できません.")?;
    let parts: Vec<&str> = file_stem.split('_').collect();

    let base_parts = parts
        .iter()
        .take(TARGET_INDEX)
        .cloned()
        .collect::<Vec<_>>()
        .join("_");
    let first_file_third = parts.get(TARGET_INDEX).ok_or(format!(
        "最初のファイル \"{:?}\" から3番目の要素を取得できませんでした.",
        first_filename
    ))?;
    let first_label = trailing_label(&parts, TARGET_INDEX + 1);

    let output_filename_str = if input_files.len() > 1 {
        let last_filename = &input_files[input_files.len() - 1];
        let last_file_third = get_split_element(last_filename, TARGET_INDEX).ok_or(format!(
            "最後のファイル \"{:?}\" から3番目の要素を取得できませんでした.",
            last_filename
        ))?;
        let last_stem = last_filename
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or("最後のファイル名からステムを取得できません.")?;
        let last_parts: Vec<&str> = last_stem.split('_').collect();
        let label_suffix = match (
            first_label.as_deref(),
            trailing_label(&last_parts, TARGET_INDEX + 1).as_deref(),
        ) {
            (Some(first), Some(last)) if first == last => format!("_{}", first),
            (Some(first), _) => format!("_{}", first),
            _ => String::new(),
        };
        format!(
            "{}_{}T{}{}_cormerge.cor",
            base_parts, first_file_third, last_file_third, label_suffix
        )
    } else {
        let label_suffix = first_label
            .as_ref()
            .map(|label| format!("_{}", label))
            .unwrap_or_default();
        format!(
            "{}_{}T{}_cormerge.cor",
            base_parts, first_file_third, label_suffix
        )
    };

    Ok(PathBuf::from(output_filename_str))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_output_filename_preserves_common_trailing_label() {
        let files = vec![
            PathBuf::from("YAMAGU34_YAMAGU34_2025309150600_x.cor"),
            PathBuf::from("YAMAGU34_YAMAGU34_2025310003600_x.cor"),
        ];

        let output = generate_output_filename(&files).unwrap();

        assert_eq!(
            output,
            PathBuf::from("YAMAGU34_YAMAGU34_2025309150600T2025310003600_x_cormerge.cor")
        );
    }

    #[test]
    fn generate_output_filename_preserves_multi_part_trailing_label() {
        let files = vec![
            PathBuf::from("YAMAGU34_YAMAGU34_2025309150600_x_usb.cor"),
            PathBuf::from("YAMAGU34_YAMAGU34_2025310003600_x_usb.cor"),
        ];

        let output = generate_output_filename(&files).unwrap();

        assert_eq!(
            output,
            PathBuf::from("YAMAGU34_YAMAGU34_2025309150600T2025310003600_x_usb_cormerge.cor")
        );
    }
}

fn cleanup_temp_files(paths: &[&Path]) {
    for path in paths {
        if let Err(e) = fs::remove_file(path) {
            if e.kind() != io::ErrorKind::NotFound {
                eprintln!(
                    "警告: 一時ファイル \"{}\" を削除できませんでした. ({})",
                    path.display(),
                    e
                );
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    // argv[1]からソート対象とする
    let mut input_files = cli.cor;
    input_files.sort();

    println!(
        "\n入力ファイルを走査しながら結合します... (source: \"{}\")",
        &cli.source
    );

    let temp_stem = generate_temp_stem();
    let temp_output_filename = PathBuf::from(format!("{}.cor.tmp", temp_stem));
    let temp_info_txt_filename = PathBuf::from(format!("{}.cor.txt.tmp", temp_stem));
    let temp_headers_csv_filename = PathBuf::from(format!("{}.cor.headers.csv.tmp", temp_stem));

    let mut outfile = BufWriter::new(File::create(&temp_output_filename)?);
    let mut info_txt_fp = BufWriter::new(File::create(&temp_info_txt_filename)?);
    let mut headers_csv_fp = BufWriter::new(File::create(&temp_headers_csv_filename)?);

    writeln!(
        info_txt_fp,
        "処理対象ファイルとヘッダー情報 (入力ファイル名ソート順):"
    )?;
    writeln!(info_txt_fp, "=================================================================================================")?;
    writeln!(
        info_txt_fp,
        "{:<45} | {:<20} | {}-byte signature at offset {}",
        "ファイル名", "値 (at offset 28)", SIGNATURE_BUFFER_LEN, SIGNATURE_OFFSET
    )?;
    writeln!(info_txt_fp, "----------------------------------------------|----------------------|------------------------------------")?;

    writeln!(headers_csv_fp, "{}", get_csv_header())?;

    let mut total_sum: u32 = 0;
    let mut matched_count = 0usize;
    let mut first_matched_file: Option<PathBuf> = None;
    let mut last_matched_file: Option<PathBuf> = None;

    for file_path in &input_files {
        let mut infile = match File::open(file_path) {
            Ok(file) => file,
            Err(e) => {
                eprintln!(
                    "警告: ファイル \"{:?}\" を開けませんでした. ({})",
                    file_path, e
                );
                continue;
            }
        };

        let mut header_bytes = [0u8; HEADER_LEN];
        let bytes_read = read_file_prefix(&mut infile, &mut header_bytes)?;
        let header_slice = &header_bytes[..bytes_read];

        if !source_name_matches(header_slice, &cli.source) {
            continue;
        }

        if has_cormerge_signature(header_slice) {
            println!(
                "情報: ファイル \"{:?}\" は期待されるシグネチャを持つためスキップします.",
                file_path
            );
            continue;
        }

        matched_count += 1;
        total_sum = total_sum.saturating_add(read_value_from_header(header_slice));
        if first_matched_file.is_none() {
            first_matched_file = Some(file_path.clone());
        }
        last_matched_file = Some(file_path.clone());

        let signature_display_str: String = read_signature_from_header(header_slice)
            .iter()
            .map(|&b| {
                if (b as char).is_ascii_graphic() {
                    b as char
                } else {
                    '.'
                }
            })
            .collect();

        writeln!(
            info_txt_fp,
            "{:<45} | 0x{:<18x} | {}",
            file_path.display(),
            read_value_from_header(header_slice),
            signature_display_str
        )?;

        println!(
            "処理中: \"{:?}\" (値 = {} / マッチ {} 件目)",
            file_path,
            read_value_from_header(header_slice),
            matched_count
        );

        if let Some(header_buffer) = copy_prefix::<HEADER_LEN>(header_slice) {
            let mut cursor = Cursor::new(header_buffer.as_slice());
            let header = parse_header(&mut cursor)?;
            let csv_row = format_header_as_csv_row(&header, file_path);
            writeln!(headers_csv_fp, "{}", csv_row)?;
        } else {
            let basename = file_path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let empty_cols = ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"; // 31 commas
            writeln!(
                headers_csv_fp,
                "\"{}\"\"File is too small to contain a valid header.\"{}",
                basename, empty_cols
            )?;
        }

        if matched_count == 1 {
            outfile.write_all(header_slice)?;
            io::copy(&mut infile, &mut outfile)?;
        } else if bytes_read >= OFFSET_FOR_SUBSEQUENT_FILES as usize {
            io::copy(&mut infile, &mut outfile)?;
        }
    }

    if matched_count == 0 {
        outfile.flush()?;
        info_txt_fp.flush()?;
        headers_csv_fp.flush()?;
        drop(outfile);
        drop(info_txt_fp);
        drop(headers_csv_fp);
        cleanup_temp_files(&[
            &temp_output_filename,
            &temp_info_txt_filename,
            &temp_headers_csv_filename,
        ]);
        return Err("フィルタリングの結果, 処理対象のファイルがありません.".into());
    }
    if matched_count < 2 {
        outfile.flush()?;
        info_txt_fp.flush()?;
        headers_csv_fp.flush()?;
        drop(outfile);
        drop(info_txt_fp);
        drop(headers_csv_fp);
        cleanup_temp_files(&[
            &temp_output_filename,
            &temp_info_txt_filename,
            &temp_headers_csv_filename,
        ]);
        return Err(format!("フィルタリングの結果, 結合対象となるファイルが {} 個のみです. 処理を続行するには少なくとも2つのファイルが必要です.", matched_count).into());
    }

    println!("{}個のファイルが処理対象です.", matched_count);
    println!("読み取った値の合計: {} (0x{:x})", total_sum, total_sum);

    writeln!(
        info_txt_fp,
        "----------------------------------------------|----------------------|------------------------------------"
    )?;
    writeln!(
        info_txt_fp,
        "合計 (全処理対象ファイル)                                       | 0x{:<18x} |",
        total_sum
    )?;
    writeln!(
        info_txt_fp,
        "=================================================================================================\n"
    )?;

    // --- 合計値とシグネチャの書き込み ---
    let output_filename = generate_output_filename(&[
        first_matched_file.clone().unwrap(),
        last_matched_file.clone().unwrap(),
    ])?;
    let info_txt_filename = output_filename.with_extension("cor.txt");
    let headers_csv_filename = output_filename.with_extension("cor.headers.csv");

    println!(
        "\n計算された合計値 {} (0x{:x}) を一時出力に書き込みます...",
        total_sum, total_sum
    );
    outfile.seek(SeekFrom::Start(VALUE_OFFSET))?;
    outfile.write_all(&total_sum.to_le_bytes())?;

    println!(
        "'{}' シグネチャを一時出力のオフセット {} ({} バイト) に書き込みます...",
        SIGNATURE_STRING, SIGNATURE_OFFSET, SIGNATURE_BUFFER_LEN
    );
    outfile.seek(SeekFrom::Start(SIGNATURE_OFFSET))?;
    let mut signature_buffer_to_write = [0u8; SIGNATURE_BUFFER_LEN];
    signature_buffer_to_write[..SIGNATURE_LEN].copy_from_slice(SIGNATURE_STRING.as_bytes());
    outfile.write_all(&signature_buffer_to_write)?;
    outfile.flush()?;
    info_txt_fp.flush()?;
    headers_csv_fp.flush()?;
    drop(outfile);
    drop(info_txt_fp);
    drop(headers_csv_fp);

    fs::rename(&temp_output_filename, &output_filename)?;
    fs::rename(&temp_info_txt_filename, &info_txt_filename)?;
    fs::rename(&temp_headers_csv_filename, &headers_csv_filename)?;

    println!("情報テキストファイル: {:?}", info_txt_filename);
    println!("ヘッダー情報ファイル: {:?}", headers_csv_filename);

    println!(
        "\n全ての処理が完了し、結果は \"{:?}\" に保存されました.",
        output_filename
    );

    Ok(())
}
