use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::prelude::*;
use num_complex::Complex;
use plotters::prelude::*;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, ErrorKind};

use crate::png_compress::{compress_png_with_mode, CompressQuality};
use crate::utils::safe_arg;

type C32 = Complex<f32>;

pub fn read_bandpass_file(path: &std::path::Path) -> io::Result<Vec<C32>> {
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

pub fn write_complex_spectrum_binary(
    path: &std::path::Path,
    spectrum: &[C32],
    fft_points: i32,
    color_flag: i32,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write the complex spectrum data (interleaved real and imaginary parts)
    for val in spectrum {
        writer.write_f32::<LittleEndian>(val.re)?;
        writer.write_f32::<LittleEndian>(val.im)?;
    }

    // Plot the spectrum
    plot_bandpass_spectrum(path, spectrum, fft_points, color_flag)?;

    Ok(())
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
        .build_cartesian_2d(0..fft_points / 2, -180.0f32..180.0f32)
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
                .map(|(i, c)| (i as i32, safe_arg(c).to_degrees())),
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
        .build_cartesian_2d(0..fft_points / 2, y_range_amp)
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
                .map(|(i, c)| (i as i32, c.norm())),
            color,
        ))
        .map_err(to_io_error)?;

    root.present().map_err(to_io_error)?;
    compress_png_with_mode(&output_file_path, CompressQuality::Low);
    Ok(())
}
