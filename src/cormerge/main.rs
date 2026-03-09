// M.AKIMOTO with Gemini
// 2025/08/18
// cargo run -- --source <source_name> <入力ファイル1> <入力ファイル2> ...

use std::io::{self, Read, Write, Seek, SeekFrom, Cursor};
use std::fs::{File};
use std::path::{Path, PathBuf};
use std::error::Error;
use std::str;
use clap::Parser;
use byteorder::{LittleEndian, ReadBytesExt};

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
    header.station1_name = String::from_utf8_lossy(&name_buf).trim_end_matches('\0').to_string();
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
    header.station2_name = String::from_utf8_lossy(&name_buf).trim_end_matches('\0').to_string();
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
    header.source_name = String::from_utf8_lossy(&source_name_buf).trim_end_matches('\0').to_string();

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


/// ファイルが指定されたソース名を持つかチェックする
fn check_source_name(filename: &Path, required_source: &str) -> Result<bool, io::Error> {
    let mut file = match File::open(filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("警告: ソース名確認のためファイル \"{:?}\" を開けませんでした. ({})", filename, e);
            return Ok(false);
        }
    };

    let file_size = file.metadata()?.len();
    if file_size < SOURCE_NAME_OFFSET + SOURCE_NAME_LEN as u64 {
        // ファイルが小さすぎてソース名を含み得ない
        return Ok(false);
    }

    file.seek(SeekFrom::Start(SOURCE_NAME_OFFSET))?;

    let mut buffer = [0u8; SOURCE_NAME_LEN];
    match file.read_exact(&mut buffer) {
        Ok(_) => {
            let name_from_file = buffer.split(|&b| b == 0).next().unwrap_or(&[]);
            let name_str = str::from_utf8(name_from_file).unwrap_or("");
            if name_str == required_source {
                Ok(true)
            } else {
                Ok(false)
            }
        },
        Err(_) => {
            // 読み取りエラーの場合は対象外
            Ok(false)
        }
    }
}


/// ファイルが指定されたシグネチャを持つかチェックし、持つ場合はスキップ対象 (true) とする
fn check_and_skip_file(filename: &Path) -> Result<bool, io::Error> {
    let mut file = match File::open(filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("警告: スキップ判定のためファイル \"{:?}\" を開けませんでした. 処理対象とします. ({})", filename, e);
            return Ok(false); // 開けない場合はスキップしない
        }
    };

    let file_size = file.metadata()?.len();
    if file_size < SIGNATURE_OFFSET + SIGNATURE_BUFFER_LEN as u64 {
        return Ok(false); // ファイルが小さすぎる
    }

    file.seek(SeekFrom::Start(SIGNATURE_OFFSET))?;

    let mut buffer = [0u8; SIGNATURE_BUFFER_LEN];
    match file.read_exact(&mut buffer) {
        Ok(_) => {}, 
        Err(e) => {
            eprintln!("警告: ファイル \"{:?}\" のシグネチャ読み込みに失敗しました. 処理対象とします. ({})", filename, e);
            return Ok(false);
        }
    }

    let mut expected_signature = [0u8; SIGNATURE_BUFFER_LEN];
    expected_signature[..SIGNATURE_LEN].copy_from_slice(SIGNATURE_STRING.as_bytes());

    if buffer == expected_signature {
        println!("情報: ファイル \"{:?}\" は期待されるシグネチャを持つためスキップします.", filename);
        return Ok(true);
    }

    Ok(false)
}

/// 指定されたファイルから特定のオフセットにある u32 値を読み取る
fn read_uint32_from_file_at_offset(filename: &Path, offset: u64) -> Result<u32, io::Error> {
    let mut file = match File::open(filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("警告: 値読み取りのためファイル \"{:?}\" を開けませんでした. ({})", filename, e);
            return Ok(0);
        }
    };

    let file_size = file.metadata()?.len();
    if file_size < offset + 4 {
        println!("情報: ファイル \"{:?}\" (サイズ {} バイト) は小さすぎるため, オフセット {} から値を読み取れません.", filename, file_size, offset);
        return Ok(0);
    }

    file.seek(SeekFrom::Start(offset))?;

    let mut buffer = [0u8; 4];
    match file.read_exact(&mut buffer) {
        Ok(_) => Ok(u32::from_le_bytes(buffer)),
        Err(e) => {
            eprintln!("警告: ファイル \"{:?}\" のオフセット {} からの読み込みに失敗しました.({})", filename, offset, e);
            Ok(0)
        }
    }
}

/// 指定されたファイルから特定のオフセットにあるシグネチャデータを読み取る
fn read_signature_data_from_file(filename: &Path, offset: u64) -> Result<[u8; SIGNATURE_BUFFER_LEN], io::Error> {
    let mut signature_data = [0u8; SIGNATURE_BUFFER_LEN];
    let mut file = match File::open(filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("警告: シグネチャ読み取りのためファイル \"{:?}\" を開けませんでした.({})", filename, e);
            return Ok(signature_data);
        }
    };

    let file_size = file.metadata()?.len();
    if file_size < offset + SIGNATURE_BUFFER_LEN as u64 {
        return Ok(signature_data); // ファイルが小さい
    }

    file.seek(SeekFrom::Start(offset))?;
    match file.read_exact(&mut signature_data) {
        Ok(_) => {}, 
        Err(e) => {
            eprintln!("警告: ファイル \"{:?}\" のオフセット {} からのシグネチャ読み込みに失敗しました.({})", filename, offset, e);
            // 失敗してもゼロクリアされたバッファを返す
            signature_data = [0u8; SIGNATURE_BUFFER_LEN];
        }
    }
    Ok(signature_data)
}


/// ファイル名から拡張子を除き、アンダースコアで分割して指定番目の要素を取得する
fn get_split_element(original_filename: &Path, target_index: usize) -> Option<String> {
    let file_stem = original_filename.file_stem()?.to_str()?;
    file_stem.split('_').nth(target_index).map(String::from)
}

/// 出力ファイル名を生成する
fn generate_output_filename(input_files: &[PathBuf]) -> Result<PathBuf, Box<dyn Error>> {
    if input_files.is_empty() {
        return Err("入力ファイルがありません.".into());
    }

    let first_filename = &input_files[0];
    const TARGET_INDEX: usize = 2; // 3番目の要素

    let file_stem = first_filename.file_stem().and_then(|s| s.to_str()).ok_or("ファイル名からステムを取得できません.")?;
    let parts: Vec<&str> = file_stem.split('_').collect();

    let base_parts = parts.iter().take(TARGET_INDEX).cloned().collect::<Vec<_>>().join("_");
    let first_file_third = parts.get(TARGET_INDEX).ok_or(format!("最初のファイル \"{:?}\" から3番目の要素を取得できませんでした.", first_filename))?;

    let output_filename_str = if input_files.len() > 1 {
        let last_filename = &input_files[input_files.len() - 1];
        let last_file_third = get_split_element(last_filename, TARGET_INDEX)
            .ok_or(format!("最後のファイル \"{:?}\" から3番目の要素を取得できませんでした.", last_filename))?;
        format!("{}_{}T{}_cormerge.cor", base_parts, first_file_third, last_file_third)
    } else {
        format!("{}_{}Tcormerge.cor", base_parts, first_file_third)
    };

    Ok(PathBuf::from(output_filename_str))
}

/// ファイルの内容を別のファイルにコピーする
fn append_file_content(outfile: &mut File, infilename: &Path, offset_src: u64) -> Result<(), io::Error> {
    let mut infile = File::open(infilename)?;
    let infile_size = infile.metadata()?.len();

    if offset_src > 0 {
        if offset_src >= infile_size {
            println!("情報: ファイル \"{:?}\" (サイズ {} バイト) はオフセット {} バイト適用後は空のためスキップします.", infilename, infile_size, offset_src);
            return Ok(())
        }
        infile.seek(SeekFrom::Start(offset_src))?;
    }

    io::copy(&mut infile, outfile)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    // argv[1]からソート対象とする
    let mut input_files = cli.cor;
    input_files.sort();

    // フィルタリング処理
    println!("\n入力ファイルのフィルタリング (source: \"{}\")", &cli.source);
    let mut files_to_process = Vec::new();
    for file_path in &input_files {
        // 1. ソース名でフィルタ
        if !check_source_name(file_path, &cli.source)? {
            continue;
        }
        // 2. 既存のシグネチャでフィルタ
        if !check_and_skip_file(file_path)? {
            files_to_process.push(file_path.clone());
        }
    }

    if files_to_process.is_empty() {
        return Err("フィルタリングの結果, 処理対象のファイルがありません.".into());
    }
    if files_to_process.len() < 2 {
         return Err(format!("フィルタリングの結果, 結合対象となるファイルが {} 個のみです. 処理を続行するには少なくとも2つのファイルが必要です.", files_to_process.len()).into());
    }
    
    println!("{}個のファイルが処理対象です.", files_to_process.len());

    // 出力ファイル名生成
    let output_filename = generate_output_filename(&files_to_process)?;
    let info_txt_filename = output_filename.with_extension("cor.txt");
    let headers_csv_filename = output_filename.with_extension("cor.headers.csv");

    // --- 情報ファイルの準備 ---
    let mut info_txt_fp = File::create(&info_txt_filename)?;
    let mut headers_csv_fp = File::create(&headers_csv_filename)?;
    println!("情報テキストファイル: {:?}", info_txt_filename);
    println!("ヘッダー情報ファイル: {:?}", headers_csv_filename);
    writeln!(info_txt_fp, "処理対象ファイルとヘッダー情報 (入力ファイル名ソート順):")?;
    writeln!(info_txt_fp, "=================================================================================================")?;
    writeln!(info_txt_fp, "{:<45} | {:<20} | {}-byte signature at offset {}", "ファイル名", "値 (at offset 28)", SIGNATURE_BUFFER_LEN, SIGNATURE_OFFSET)?;
    writeln!(info_txt_fp, "----------------------------------------------|----------------------|------------------------------------")?;

    writeln!(headers_csv_fp, "{}", get_csv_header())?;


    // --- 値の合計と情報ファイルへの書き込み ---
    let mut total_sum: u32 = 0;
    println!("\n処理対象ファイルの情報を読み取り, テキストファイル \"{:?}\", \"{:?}\" に記録します:", info_txt_filename, headers_csv_filename);
    for file_path in &files_to_process {
        let current_value = read_uint32_from_file_at_offset(file_path, VALUE_OFFSET)?;
        total_sum = total_sum.saturating_add(current_value);
        println!("  ファイル \"{:?}\": 値 = {} (0x{:x})", file_path, current_value, current_value);

        let signature_data = read_signature_data_from_file(file_path, SIGNATURE_OFFSET)?;
        let signature_display_str: String = signature_data.iter()
            .map(|&b| if (b as char).is_ascii_graphic() { b as char } else { '.' })
            .collect();

        writeln!(info_txt_fp, "{:<45} | 0x{:<18x} | {}", file_path.display(), current_value, signature_display_str)?;

        // ヘッダーを読み込んでパースし、ファイルに書き込む
        let mut file_content = Vec::new();
        File::open(file_path)?.read_to_end(&mut file_content)?;
        if file_content.len() >= 256 {
            let mut cursor = Cursor::new(file_content.as_slice());
            let header = parse_header(&mut cursor)?;
            let csv_row = format_header_as_csv_row(&header, file_path);
            writeln!(headers_csv_fp, "{}", csv_row)?;
        } else {
            let basename = file_path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let empty_cols = ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"; // 31 commas
            writeln!(headers_csv_fp, "\"{}\"\"File is too small to contain a valid header.\"{}", basename, empty_cols)?;
        }
    }
    println!("読み取った値の合計: {} (0x{:x})", total_sum, total_sum);

    writeln!(info_txt_fp, "----------------------------------------------|----------------------|------------------------------------")?;
    writeln!(info_txt_fp, "合計 (全処理対象ファイル)                                       | 0x{:<18x} |", total_sum)?;
    writeln!(info_txt_fp, "=================================================================================================\n")?;


    // --- ファイルの結合処理 ---
    println!("\nファイルの結合処理を開始します...");
    let mut outfile = File::create(&output_filename)?;
    println!("出力ファイル: {:?}", output_filename);

    // 1つ目のファイル
    println!("処理中: \"{:?}\" (全体をコピー中)", files_to_process[0]);
    append_file_content(&mut outfile, &files_to_process[0], 0)?;

    // 2つ目以降
    for file_path in files_to_process.iter().skip(1) {
        println!("処理中: \"{:?}\" (先頭 {} バイトをスキップして結合中)", file_path, OFFSET_FOR_SUBSEQUENT_FILES);
        append_file_content(&mut outfile, file_path, OFFSET_FOR_SUBSEQUENT_FILES)?;
    }

    // --- 合計値とシグネチャの書き込み ---
    println!("\n計算された合計値 {} (0x{:x}) を \"{:?}\" のオフセット {} に書き込みます...", total_sum, total_sum, output_filename, VALUE_OFFSET);
    outfile.seek(SeekFrom::Start(VALUE_OFFSET))?;
    outfile.write_all(&total_sum.to_le_bytes())?;

    println!("'{}' シグネチャを \"{:?}\" のオフセット {} ({} バイト) に書き込みます...", SIGNATURE_STRING, output_filename, SIGNATURE_OFFSET, SIGNATURE_BUFFER_LEN);
    outfile.seek(SeekFrom::Start(SIGNATURE_OFFSET))?;
    let mut signature_buffer_to_write = [0u8; SIGNATURE_BUFFER_LEN];
    signature_buffer_to_write[..SIGNATURE_LEN].copy_from_slice(SIGNATURE_STRING.as_bytes());
    outfile.write_all(&signature_buffer_to_write)?;

    println!("\n全ての処理が完了し、結果は \"{:?}\" に保存されました.", output_filename);

    Ok(())
}