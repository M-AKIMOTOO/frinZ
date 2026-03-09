use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::process;

use anyhow::{anyhow, Context, Result};
use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};
use clap::Parser;
use num_complex::Complex32;

const FILE_HEADER_SIZE: usize = 256;
const SECTOR_HEADER_SIZE: usize = 128;
const BYTES_PER_COMPLEX: usize = 8; // f32 (real) + f32 (imag)

#[derive(Debug, Default)]
struct CorHeader {
    magic_word: [u8; 4],
    header_version: i32,
    software_version: i32,
    sampling_speed: i32,
    observing_frequency: f64,
    fft_point: i32,
    number_of_sector: i32,
    station1_name: String,
    station1_code: String,
    station1_position: [f64; 3],
    station2_name: String,
    station2_code: String,
    station2_position: [f64; 3],
    source_name: String,
    source_position_ra: f64,
    source_position_dec: f64,
    station1_clock_delay: f64,
    station1_clock_rate: f64,
    station1_clock_acel: f64,
    station1_clock_jerk: f64,
    station1_clock_snap: f64,
    station2_clock_delay: f64,
    station2_clock_rate: f64,
    station2_clock_acel: f64,
    station2_clock_jerk: f64,
    station2_clock_snap: f64,
}

fn parse_header(cursor: &mut Cursor<&[u8]>) -> Result<CorHeader> {
    use byteorder::ReadBytesExt;

    let mut header = CorHeader::default();
    cursor.set_position(0);

    cursor.read_exact(&mut header.magic_word)?;
    header.header_version = cursor.read_i32::<LittleEndian>()?;
    header.software_version = cursor.read_i32::<LittleEndian>()?;
    header.sampling_speed = cursor.read_i32::<LittleEndian>()?;
    header.observing_frequency = cursor.read_f64::<LittleEndian>()?;
    header.fft_point = cursor.read_i32::<LittleEndian>()?;
    header.number_of_sector = cursor.read_i32::<LittleEndian>()?;

    let mut name_buf = [0u8; 8];
    cursor.read_exact(&mut name_buf)?;
    header.station1_name = String::from_utf8_lossy(&name_buf)
        .trim_end_matches('\0')
        .to_string();
    cursor.set_position(cursor.position() + 8);

    header.station1_position[0] = cursor.read_f64::<LittleEndian>()?;
    header.station1_position[1] = cursor.read_f64::<LittleEndian>()?;
    header.station1_position[2] = cursor.read_f64::<LittleEndian>()?;

    let mut code_buf = [0u8; 1];
    cursor.read_exact(&mut code_buf)?;
    header.station1_code = String::from_utf8_lossy(&code_buf).to_string();
    cursor.set_position(cursor.position() + 7);

    cursor.read_exact(&mut name_buf)?;
    header.station2_name = String::from_utf8_lossy(&name_buf)
        .trim_end_matches('\0')
        .to_string();
    cursor.set_position(cursor.position() + 8);

    header.station2_position[0] = cursor.read_f64::<LittleEndian>()?;
    header.station2_position[1] = cursor.read_f64::<LittleEndian>()?;
    header.station2_position[2] = cursor.read_f64::<LittleEndian>()?;

    cursor.read_exact(&mut code_buf)?;
    header.station2_code = String::from_utf8_lossy(&code_buf).to_string();
    cursor.set_position(cursor.position() + 7);

    let mut source_name_buf = [0u8; 16];
    cursor.read_exact(&mut source_name_buf)?;
    header.source_name = String::from_utf8_lossy(&source_name_buf)
        .trim_end_matches('\0')
        .to_string();

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

    Ok(header)
}

#[derive(Parser, Debug)]
#[command(
    name = "BandScythe",
    about = "Cut unwanted frequency ranges from a .cor file and emit a slimmed dataset.",
    version,
    arg_required_else_help = true
)]
struct Cli {
    /// Input .cor file
    #[arg(long)]
    cor: PathBuf,

    /// Frequency window to keep, in MHz (provide as two values, e.g. 6664 6672)
    #[arg(long, value_names = ["LOW", "HIGH"], num_args = 2)]
    band: Vec<f64>,

    /// Rebin FFT length after band selection (e.g. --fft-rebin 1024)
    #[arg(long, value_name = "FFT_POINTS")]
    fft_rebin: Option<usize>,

    /// Number of sectors to skip from the beginning of the file
    #[arg(long, value_name = "SECTORS", default_value_t = 0)]
    skip: u32,

    /// Number of sectors to process (0 means process all remaining sectors)
    #[arg(long, value_name = "SECTORS", default_value_t = 0)]
    length: u32,
}

fn parse_band(low: f64, high: f64) -> Result<(f64, f64)> {
    if low >= high {
        return Err(anyhow!("band specification requires LOW < HIGH (MHz)"));
    }
    Ok((low, high))
}

fn default_output_path(input: &Path) -> PathBuf {
    let stem_raw = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let stem = normalize_stem_with_label(stem_raw);
    let ext = input.extension().and_then(|s| s.to_str()).unwrap_or("cor");
    input
        .with_file_name(format!("{stem}_bandscythe.{ext}"))
        .to_path_buf()
}

fn normalize_stem_with_label(stem: &str) -> String {
    let parts: Vec<&str> = stem.split('_').collect();
    if parts.len() > 3 {
        let mut prefix = parts[..3].join("_");
        let label = parts[3..].join("_");
        prefix.push('_');
        prefix.push_str(&label);
        prefix
    } else {
        stem.to_string()
    }
}

fn compute_frequency_indices(
    header: &CorHeader,
    low_mhz: f64,
    high_mhz: f64,
) -> Result<(usize, usize, f64)> {
    let base_freq_hz = header.observing_frequency;
    let sampling_hz = header.sampling_speed as f64;
    let fft_point = header.fft_point as f64;
    let total_bins = (header.fft_point / 2) as usize;

    if sampling_hz <= 0.0 || fft_point <= 0.0 {
        return Err(anyhow!("invalid sampling speed or fft_point in header"));
    }

    let freq_res_hz = sampling_hz / fft_point;
    let available_low_mhz = base_freq_hz / 1e6;
    let available_high_mhz = (base_freq_hz + freq_res_hz * total_bins as f64) / 1e6;

    if low_mhz < available_low_mhz - 1e-6 || high_mhz > available_high_mhz + 1e-6 {
        return Err(anyhow!(
            "requested band {:.6}-{:.6} MHz is outside available {:.6}-{:.6} MHz",
            low_mhz,
            high_mhz,
            available_low_mhz,
            available_high_mhz
        ));
    }

    let low_hz = (low_mhz * 1e6).max(base_freq_hz);
    let high_hz = (high_mhz * 1e6).min(base_freq_hz + freq_res_hz * total_bins as f64);

    let start_idx = ((low_hz - base_freq_hz) / freq_res_hz)
        .floor()
        .clamp(0.0, total_bins as f64) as usize;
    let end_idx_exclusive = ((high_hz - base_freq_hz) / freq_res_hz)
        .ceil()
        .clamp(0.0, total_bins as f64) as usize;

    if end_idx_exclusive <= start_idx {
        return Err(anyhow!(
            "selected band does not intersect available frequency grid"
        ));
    }

    let actual_low_mhz = (base_freq_hz + freq_res_hz * start_idx as f64) / 1e6;
    Ok((start_idx, end_idx_exclusive, actual_low_mhz))
}

fn update_header_bytes(
    header_bytes: &mut [u8; FILE_HEADER_SIZE],
    new_sampling_speed: i32,
    new_observing_freq_hz: f64,
    new_fft_point: i32,
    new_number_of_sector: i32,
) {
    LittleEndian::write_i32(&mut header_bytes[12..16], new_sampling_speed);
    LittleEndian::write_f64(&mut header_bytes[16..24], new_observing_freq_hz);
    LittleEndian::write_i32(&mut header_bytes[24..28], new_fft_point);
    LittleEndian::write_i32(&mut header_bytes[28..32], new_number_of_sector);
    const SIGNATURE_OFFSET: usize = 248;
    const SIGNATURE: &[u8; 8] = b"bandscy\0";
    let target = &mut header_bytes[SIGNATURE_OFFSET..SIGNATURE_OFFSET + SIGNATURE.len()];
    target.copy_from_slice(SIGNATURE);
}

fn write_cropped_cor(
    mut reader: BufReader<File>,
    mut writer: BufWriter<File>,
    mut header: CorHeader,
    mut header_bytes: [u8; FILE_HEADER_SIZE],
    start_idx: usize,
    end_idx: usize,
    freq_res_hz: f64,
    fft_rebin_bins: Option<usize>,
    skip_sectors: usize,
    sectors_to_process: usize,
) -> Result<()> {
    let bins_keep = end_idx - start_idx;
    let total_bins = (header.fft_point / 2) as usize;
    let output_bins = fft_rebin_bins.unwrap_or(bins_keep);
    let rebin_ratio = bins_keep / output_bins.max(1);

    let new_fft_point = (output_bins * 2) as i32;
    let new_sampling_speed =
        ((header.sampling_speed as f64) * (bins_keep as f64) / (total_bins as f64)).round() as i32;
    let new_observing_freq_hz = header.observing_frequency + freq_res_hz * start_idx as f64;

    header.sampling_speed = new_sampling_speed;
    header.observing_frequency = new_observing_freq_hz;
    header.fft_point = new_fft_point;
    header.number_of_sector = sectors_to_process as i32;

    update_header_bytes(
        &mut header_bytes,
        new_sampling_speed,
        new_observing_freq_hz,
        new_fft_point,
        header.number_of_sector,
    );

    writer
        .write_all(&header_bytes)
        .context("failed to write updated header")?;

    let mut sector_header = vec![0u8; SECTOR_HEADER_SIZE];
    let mut sample_buf = [0u8; BYTES_PER_COMPLEX];

    for sector_idx in 0..skip_sectors {
        reader.read_exact(&mut sector_header).with_context(|| {
            format!("failed to read sector header {} while skipping", sector_idx)
        })?;
        for bin in 0..total_bins {
            reader.read_exact(&mut sample_buf).with_context(|| {
                format!("failed to skip data for sector {}, bin {}", sector_idx, bin)
            })?;
        }
    }

    for processed_idx in 0..sectors_to_process {
        let sector_idx = skip_sectors + processed_idx;
        reader
            .read_exact(&mut sector_header)
            .with_context(|| format!("failed to read sector header {}", sector_idx))?;
        writer
            .write_all(&sector_header)
            .with_context(|| format!("failed to write sector header {}", sector_idx))?;

        let mut selected = Vec::with_capacity(bins_keep);
        for bin in 0..total_bins {
            reader.read_exact(&mut sample_buf).with_context(|| {
                format!("failed to read data for sector {}, bin {}", sector_idx, bin)
            })?;
            if bin >= start_idx && bin < end_idx {
                let real = LittleEndian::read_f32(&sample_buf[0..4]);
                let imag = LittleEndian::read_f32(&sample_buf[4..8]);
                selected.push(Complex32::new(real, imag));
            }
        }

        if selected.len() != bins_keep {
            return Err(anyhow!(
                "unexpected number of bins collected (expected {}, got {}) for sector {}",
                bins_keep,
                selected.len(),
                sector_idx
            ));
        }

        let rebinned_storage;
        let data_to_write: &[Complex32] = if let Some(target_bins) = fft_rebin_bins {
            rebinned_storage = rebin_complex_spectrum(&selected, target_bins)?;
            &rebinned_storage
        } else {
            &selected
        };

        for (bin_idx, value) in data_to_write.iter().enumerate() {
            writer
                .write_f32::<LittleEndian>(value.re)
                .with_context(|| {
                    format!(
                        "failed to write real part for sector {}, bin {}",
                        sector_idx, bin_idx
                    )
                })?;
            writer
                .write_f32::<LittleEndian>(value.im)
                .with_context(|| {
                    format!(
                        "failed to write imag part for sector {}, bin {}",
                        sector_idx, bin_idx
                    )
                })?;
        }
    }

    writer.flush().context("failed to flush output file")?;
    let percent = (bins_keep as f64 / total_bins as f64) * 100.0;
    if let Some(target_bins) = fft_rebin_bins {
        println!(
            "BandScythe complete: kept {} / {} channels ({:.2}%) after band selection, rebinned to {} channels (ratio 1:{}), new observing freq {:.6} MHz, sampling {:.3} MHz",
            bins_keep,
            total_bins,
            percent,
            target_bins,
            rebin_ratio,
            new_observing_freq_hz / 1e6,
            new_sampling_speed as f64 / 1e6
        );
    } else {
        println!(
            "BandScythe complete: kept {} / {} channels ({:.2}%), new observing freq {:.6} MHz, sampling {:.3} MHz",
            bins_keep,
            total_bins,
            percent,
            new_observing_freq_hz / 1e6,
            new_sampling_speed as f64 / 1e6
        );
    }
    Ok(())
}

fn rebin_complex_spectrum(data: &[Complex32], output_bins: usize) -> Result<Vec<Complex32>> {
    if output_bins == 0 {
        return Err(anyhow!("requested output bin count must be positive"));
    }
    if data.len() % output_bins != 0 {
        return Err(anyhow!(
            "cannot rebin {} samples into {} bins (non-integer ratio)",
            data.len(),
            output_bins
        ));
    }
    let ratio = data.len() / output_bins;
    if ratio == 0 {
        return Err(anyhow!("invalid rebin ratio 0"));
    }
    let mut rebinned = Vec::with_capacity(output_bins);
    for chunk in data.chunks(ratio) {
        let mut sum_re = 0.0f32;
        let mut sum_im = 0.0f32;
        for value in chunk {
            sum_re += value.re;
            sum_im += value.im;
        }
        let scale = 1.0f32 / (ratio as f32);
        rebinned.push(Complex32::new(sum_re * scale, sum_im * scale));
    }
    Ok(rebinned)
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    if cli.band.len() != 2 {
        return Err(anyhow!(
            "--band requires two values, e.g. --band LOW HIGH (in MHz)"
        ));
    }
    let (low_mhz, high_mhz) = parse_band(cli.band[0], cli.band[1])?;
    let input_path = cli.cor;
    let output_path = default_output_path(&input_path);

    if input_path == output_path {
        return Err(anyhow!("input and output paths must differ"));
    }

    let input_file = File::open(&input_path)
        .with_context(|| format!("failed to open input file {:?}", input_path))?;
    let mut reader = BufReader::new(input_file);

    let mut header_bytes = [0u8; FILE_HEADER_SIZE];
    reader
        .read_exact(&mut header_bytes)
        .context("failed to read .cor header")?;
    let mut cursor = Cursor::new(&header_bytes[..]);
    let header = parse_header(&mut cursor).context("failed to parse .cor header")?;

    let (start_idx, end_idx, actual_low_mhz) =
        compute_frequency_indices(&header, low_mhz, high_mhz)?;
    let freq_res_hz = header.sampling_speed as f64 / header.fft_point as f64;
    let bins_keep = end_idx - start_idx;
    if bins_keep == 0 || (bins_keep & (bins_keep - 1)) != 0 {
        return Err(anyhow!(
            "selected band keeps {} channels, which is not a power of two. Adjust --band so that the channel count is 2^n.",
            bins_keep
        ));
    }

    println!(
        "Input band {:.6}-{:.6} MHz (available {:.6}-{:.6} MHz)",
        low_mhz,
        high_mhz,
        header.observing_frequency / 1e6,
        (header.observing_frequency + freq_res_hz * (header.fft_point / 2) as f64) / 1e6
    );
    println!(
        "Keeping indices {}..{} ({} channels) starting at {:.6} MHz",
        start_idx,
        end_idx,
        end_idx - start_idx,
        actual_low_mhz
    );

    let total_sectors = header.number_of_sector.max(0) as usize;
    if total_sectors == 0 {
        return Err(anyhow!(
            "input file declares zero sectors; nothing to process"
        ));
    }
    let skip_sectors = cli.skip as usize;
    if skip_sectors >= total_sectors {
        return Err(anyhow!(
            "no sectors remain after applying --skip {} (available {})",
            cli.skip,
            total_sectors
        ));
    }
    let remaining_after_skip = total_sectors - skip_sectors;
    let requested_length = cli.length as usize;
    let sectors_to_process = if requested_length == 0 {
        remaining_after_skip
    } else {
        requested_length.min(remaining_after_skip)
    };
    if sectors_to_process == 0 {
        return Err(anyhow!(
            "no sectors remain after applying --skip {} and --length {}",
            cli.skip,
            cli.length
        ));
    }
    if requested_length > 0 && requested_length > sectors_to_process {
        println!(
            "Warning: requested --length {} reduced to available {} sectors",
            cli.length, sectors_to_process
        );
    }
    println!(
        "Skipping {} sector(s), exporting {} sector(s) out of {}",
        skip_sectors, sectors_to_process, total_sectors
    );

    let fft_rebin_bins = if let Some(target_fft) = cli.fft_rebin {
        if target_fft % 2 != 0 {
            return Err(anyhow!(
                "--fft-rebin must be an even number of FFT points (got {})",
                target_fft
            ));
        }
        if target_fft == 0 {
            return Err(anyhow!("--fft-rebin must be greater than zero"));
        }
        let target_bins = target_fft / 2;
        if target_bins == 0 || target_bins > bins_keep {
            return Err(anyhow!(
                "--fft-rebin ({} points) results in {} channels, which exceeds the kept band ({} channels)",
                target_fft,
                target_bins,
                bins_keep
            ));
        }
        if target_bins & (target_bins - 1) != 0 {
            return Err(anyhow!(
                "--fft-rebin requires a power-of-two FFT length ({} is not)",
                target_fft
            ));
        }
        if bins_keep % target_bins != 0 {
            return Err(anyhow!(
                "cannot evenly rebin {} kept channels into {} channels (--fft-rebin {})",
                bins_keep,
                target_bins,
                target_fft
            ));
        }
        println!(
            "Rebinning kept band from {} to {} channels (FFT {} points).",
            bins_keep, target_bins, target_fft
        );
        Some(target_bins)
    } else {
        None
    };

    let output_file = File::create(&output_path)
        .with_context(|| format!("failed to create output file {:?}", output_path))?;
    let writer = BufWriter::new(output_file);

    write_cropped_cor(
        reader,
        writer,
        header,
        header_bytes,
        start_idx,
        end_idx,
        freq_res_hz,
        fft_rebin_bins,
        skip_sectors,
        sectors_to_process,
    )?;

    println!("Output written to {:?}", output_path);
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        let mut source = err.source();
        while let Some(inner) = source {
            eprintln!("Caused by: {inner}");
            source = inner.source();
        }
        process::exit(1);
    }
}
