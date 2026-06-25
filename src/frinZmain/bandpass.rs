use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::prelude::*;
use num_complex::Complex;
use plotters::prelude::*;
use std::fs::{self, File};
use std::io::{self, BufReader, ErrorKind, Read};

use crate::png_compress::{compress_png_with_mode, CompressQuality};
use crate::utils::safe_arg;

type C32 = Complex<f32>;

pub fn read_bandpass_file(path: &std::path::Path) -> io::Result<Vec<C32>> {
    if path
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("npz"))
    {
        return read_bandpass_npz(path);
    }
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut bandpass_data = Vec::new();

    while let Ok(real) = reader.read_f32::<LittleEndian>() {
        let imag = reader.read_f32::<LittleEndian>().map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Incomplete complex number at end of file",
                )
            } else {
                e
            }
        })?;
        bandpass_data.push(C32::new(real, imag));
    }

    Ok(bandpass_data)
}

fn read_bandpass_npz(path: &std::path::Path) -> io::Result<Vec<C32>> {
    let archive = fs::read(path)?;
    if archive.len() < 30 || &archive[..4] != b"PK\x03\x04" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid NPZ local header",
        ));
    }
    let u16_at =
        |offset: usize| u16::from_le_bytes([archive[offset], archive[offset + 1]]) as usize;
    let u32_at = |offset: usize| {
        u32::from_le_bytes([
            archive[offset],
            archive[offset + 1],
            archive[offset + 2],
            archive[offset + 3],
        ]) as usize
    };
    let method = u16_at(8);
    let compressed_size = u32_at(18);
    let name_len = u16_at(26);
    let extra_len = u16_at(28);
    let data_start = 30_usize
        .checked_add(name_len)
        .and_then(|value| value.checked_add(extra_len))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "NPZ offset overflow"))?;
    let data_end = data_start
        .checked_add(compressed_size)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "NPZ size overflow"))?;
    if data_end > archive.len() || &archive[30..30 + name_len] != b"data.npy" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "NPZ data.npy entry is missing",
        ));
    }
    let npy = match method {
        0 => archive[data_start..data_end].to_vec(),
        8 => {
            let mut decoder = flate2::read::DeflateDecoder::new(&archive[data_start..data_end]);
            let mut decoded = Vec::new();
            decoder.read_to_end(&mut decoded)?;
            decoded
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported NPZ compression method {method}"),
            ))
        }
    };
    if npy.len() < 12 || &npy[..6] != b"\x93NUMPY" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid data.npy header",
        ));
    }
    let payload_start = match (npy[6], npy[7]) {
        (1, 0) => 10 + u16::from_le_bytes([npy[8], npy[9]]) as usize,
        (2, 0) | (3, 0) => 12 + u32::from_le_bytes([npy[8], npy[9], npy[10], npy[11]]) as usize,
        version => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported NPY version {version:?}"),
            ))
        }
    };
    if payload_start > npy.len() || (npy.len() - payload_start) % 8 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "data.npy is not a complex64 array",
        ));
    }
    let mut result = Vec::with_capacity((npy.len() - payload_start) / 8);
    for value in npy[payload_start..].chunks_exact(8) {
        result.push(C32::new(
            f32::from_le_bytes(value[0..4].try_into().unwrap()),
            f32::from_le_bytes(value[4..8].try_into().unwrap()),
        ));
    }
    Ok(result)
}

pub fn apply_bandpass_correction(freq_rate_array: &mut Array2<C32>, bandpass_data: &[C32]) {
    if bandpass_data.is_empty() {
        return;
    }
    const EPSILON: f32 = 1e-9;

    // The complex mean is used to rescale the corrected spectrum to maintain a similar overall power and phase.
    let bandpass_sum: C32 = bandpass_data.iter().copied().sum();
    let bandpass_mean = bandpass_sum / bandpass_data.len() as f32;

    for (mut row, &bp_val) in freq_rate_array
        .rows_mut()
        .into_iter()
        .zip(bandpass_data.iter())
    {
        // Avoid division by zero or near-zero values
        if bp_val.norm() > EPSILON {
            row.iter_mut()
                .for_each(|elem| *elem = (*elem / bp_val) * bandpass_mean);
        }
    }
}

pub fn plot_bandpass_spectrum(
    path: &std::path::Path,
    spectrum: &[C32],
    fft_points: i32,
    color_flag: i32,
) -> io::Result<()> {
    const PLOT_WIDTH: u32 = 800;
    const PLOT_HEIGHT: u32 = 600;
    const UPPER_PLOT_HEIGHT: u32 = 180;
    const FONT_STYLE: (&str, i32) = ("sans-serif", 25);

    // Helper to convert plotters error to io::Error, reducing boilerplate
    fn to_io_error<E: std::fmt::Display>(e: E) -> io::Error {
        io::Error::new(ErrorKind::Other, e.to_string())
    }

    let output_file_path = path.with_extension("png"); // Change extension to png
    let root = BitMapBackend::new(&output_file_path, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root.fill(&WHITE).map_err(to_io_error)?;

    let (upper, lower) = root.split_vertically(UPPER_PLOT_HEIGHT);

    let color = if color_flag == 0 { &RED } else { &MAGENTA };

    // --- Phase Plot (Top) ---
    let mut phase_chart = ChartBuilder::on(&upper)
        .margin(10)
        .y_label_area_size(90)
        .build_cartesian_2d(0.0f64..fft_points as f64 / 2.0, -180.0f32..180.0f32)
        .map_err(to_io_error)?;

    phase_chart
        .configure_mesh()
        .y_desc("Phase (deg)")
        .label_style(FONT_STYLE)
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .y_labels(4)
        .y_label_formatter(&|y| format!("{:.0}", y))
        .draw()
        .map_err(to_io_error)?;

    phase_chart
        .draw_series(LineSeries::new(
            spectrum
                .iter()
                .enumerate()
                .map(|(i, c)| (i as f64, safe_arg(c).to_degrees())),
            color,
        ))
        .map_err(to_io_error)?;

    // --- Amplitude Plot ---
    let max_amp = spectrum.iter().map(|c| c.norm()).fold(0.0f32, f32::max);
    // Add a small epsilon to the max amplitude to avoid a zero-range in case of all-zero spectrum
    let y_range_amp = 0.0f32..(max_amp * 1.0).max(1e-9);

    let mut amp_chart = ChartBuilder::on(&lower)
        .margin(10)
        .x_label_area_size(55)
        .y_label_area_size(90)
        .build_cartesian_2d(0.0f64..fft_points as f64 / 2.0, y_range_amp)
        .map_err(to_io_error)?;

    amp_chart
        .configure_mesh()
        .x_desc("Channels")
        .y_desc("Amplitude")
        .label_style(FONT_STYLE)
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .y_labels(5)
        .y_label_formatter(&|y| format!("{:.1e}", y))
        .draw()
        .map_err(to_io_error)?;

    amp_chart
        .draw_series(LineSeries::new(
            spectrum
                .iter()
                .enumerate()
                .map(|(i, c)| (i as f64, c.norm())),
            color,
        ))
        .map_err(to_io_error)?;

    root.present().map_err(to_io_error)?;
    compress_png_with_mode(&output_file_path, CompressQuality::Low);
    Ok(())
}
