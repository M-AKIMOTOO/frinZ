use std::error::Error;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::PathBuf;
use std::process::exit;

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use clap::Parser;
use num_complex::Complex;

// frinZ クレートのモジュールを再利用
use frinZ::header::parse_header;
use frinZ::read::read_visibility_data;

type C32 = Complex<f32>;

#[derive(Parser, Debug)]
#[command(
    name = "corshow",
    version = "0.1.0",
    author = "Masanori AKIMOTO <masanori.akimoto.ac@gmail.com>",
    about = "Reads .cor or .bin files and outputs complex spectra to a CSV file.",
    arg_required_else_help = true
)]
struct Args {
    /// Path to the input .cor file
    #[arg(long, required_unless_present = "bin", conflicts_with = "bin")]
    cor: Option<PathBuf>,

    /// Path to the input .bin file (from --cor2bin)
    #[arg(long, required_unless_present = "cor")]
    bin: Option<PathBuf>,

    /// Path to the output .csv file
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let fft_point: i32;
    let pp: i32;
    let spectra: Vec<C32>;
    let input_path: PathBuf;

    if let Some(cor_path) = &args.cor {
        input_path = cor_path.clone();
        println!("Reading .cor file: {:?}", cor_path);
        let mut file = File::open(cor_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let mut cursor = Cursor::new(buffer.as_slice());

        let header = parse_header(&mut cursor)?;
        fft_point = header.fft_point;
        pp = header.number_of_sector;

        // Read all sectors
        cursor.set_position(0); // Reset cursor to read from the beginning for visibility data
        let (all_spectra, _, _) = read_visibility_data(&mut cursor, &header, pp, 0, 0, false, &[])?;
        spectra = all_spectra;
    } else if let Some(bin_path) = &args.bin {
        input_path = bin_path.clone();
        println!("Reading .bin file: {:?}", bin_path);
        let mut file = File::open(bin_path)?;

        fft_point = file.read_f32::<LittleEndian>()? as i32;
        pp = file.read_f32::<LittleEndian>()? as i32;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        spectra = buffer
            .chunks_exact(8)
            .map(|chunk| {
                let re = LittleEndian::read_f32(&chunk[0..4]);
                let im = LittleEndian::read_f32(&chunk[4..8]);
                C32::new(re, im)
            })
            .collect();
    } else {
        eprintln!("Error: Either --cor or --bin must be provided.");
        exit(1);
    }

    if spectra.is_empty() {
        println!("No spectral data found.");
        return Ok(());
    }

    let fft_point_half = (fft_point / 2) as usize;
    if spectra.len() != (pp as usize) * fft_point_half {
        eprintln!(
            "Error: Data size mismatch. Expected {} * {} = {} points, but found {}.",
            pp,
            fft_point_half,
            (pp as usize) * fft_point_half,
            spectra.len()
        );
        exit(1);
    }

    // Determine output path
    let output_path = args
        .output
        .unwrap_or_else(|| input_path.with_extension("csv"));

    // Write to CSV
    let mut writer = csv::Writer::from_path(&output_path)?;
    println!(
        "Writing {} sectors of {} channels to {:?}...",
        pp, fft_point_half, &output_path
    );

    for i in 0..(pp as usize) {
        let start = i * fft_point_half;
        let end = start + fft_point_half;
        let sector_data = &spectra[start..end];

        let record: Vec<String> = sector_data
            .iter()
            .map(|c| format!("{}+{}j", c.re, c.im))
            .collect();
        writer.write_record(&record)?;
    }

    writer.flush()?;
    println!("Successfully wrote CSV file.");

    Ok(())
}
