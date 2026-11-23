use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::prelude::*;
use num_complex::Complex;
use plotters::prelude::*;
use std::fs::File;
use std::io::{self, BufReader, Read};

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
    let mut archive = zip::ZipArchive::new(File::open(path)?)?;
    let read_entry = |archive: &mut zip::ZipArchive<File>, name: &str| -> io::Result<Vec<u8>> {
        let mut entry = archive.by_name(name)?;
        let mut npy = Vec::new();
        entry.read_to_end(&mut npy)?;
        Ok(npy)
    };
    match read_entry(&mut archive, "data.npy") {
        Ok(data) => return parse_complex64_npy(&data),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    let real = parse_real_npy(&read_entry(&mut archive, "real.npy")?)?;
    let imag = parse_real_npy(&read_entry(&mut archive, "imag.npy")?)?;
    if real.len() != imag.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "real/imag length mismatch: {} != {}",
                real.len(),
                imag.len()
            ),
        ));
    }
    Ok(real
        .into_iter()
        .zip(imag)
        .map(|(re, im)| C32::new(re, im))
        .collect())
}

fn npy_payload(npy: &[u8]) -> io::Result<(&str, &[u8])> {
    if npy.len() < 12 || &npy[..6] != b"\x93NUMPY" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid NPY header",
        ));
    }
    let (header_start, header_len) = match (npy[6], npy[7]) {
        (1, 0) => (10usize, u16::from_le_bytes([npy[8], npy[9]]) as usize),
        (2, 0) | (3, 0) => (
            12usize,
            u32::from_le_bytes([npy[8], npy[9], npy[10], npy[11]]) as usize,
        ),
        version => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported NPY version {version:?}"),
            ))
        }
    };
    let payload_start = header_start
        .checked_add(header_len)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "NPY payload offset overflow"))?;
    if payload_start > npy.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "NPY payload starts past end of file",
        ));
    }
    let header = std::str::from_utf8(&npy[header_start..payload_start])
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "NPY header is not UTF-8"))?;
    Ok((header, &npy[payload_start..]))
}

fn parse_complex64_npy(npy: &[u8]) -> io::Result<Vec<C32>> {
    let (header, payload) = npy_payload(npy)?;
    if !header.contains("'descr': '<c8'") && !header.contains("\"descr\": \"<c8\"") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "data.npy is not a complex64 array",
        ));
    }
    if payload.len() % 8 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "complex64 payload length is not divisible by 8",
        ));
    }
    Ok(payload
        .chunks_exact(8)
        .map(|value| {
            C32::new(
                f32::from_le_bytes(value[0..4].try_into().unwrap()),
                f32::from_le_bytes(value[4..8].try_into().unwrap()),
            )
        })
        .collect())
}

fn parse_real_npy(npy: &[u8]) -> io::Result<Vec<f32>> {
    let (header, payload) = npy_payload(npy)?;
    if header.contains("'descr': '<f8'") || header.contains("\"descr\": \"<f8\"") {
        if payload.len() % 8 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "float64 payload length is not divisible by 8",
            ));
        }
        Ok(payload
            .chunks_exact(8)
            .map(|value| f64::from_le_bytes(value.try_into().unwrap()) as f32)
            .collect())
    } else if header.contains("'descr': '<f4'") || header.contains("\"descr\": \"<f4\"") {
        if payload.len() % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "float32 payload length is not divisible by 4",
            ));
        }
        Ok(payload
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "real/imag NPY entries must be float32 or float64",
        ))
    }
}

pub(crate) fn bandpass_gains(bandpass_data: &[C32]) -> impl Iterator<Item = Complex<f64>> + '_ {
    let mean = bandpass_data
        .iter()
        .map(|v| Complex::new(f64::from(v.re), f64::from(v.im)))
        .sum::<Complex<f64>>()
        / bandpass_data.len().max(1) as f64;
    bandpass_data.iter().map(move |v| {
        let value = Complex::new(f64::from(v.re), f64::from(v.im));
        if value.norm_sqr() > 1.0e-18 {
            mean / value
        } else {
            Complex::new(1.0, 0.0)
        }
    })
}

pub(crate) fn apply_bandpass_gains(
    freq_rate_array: &mut Array2<C32>,
    gains: impl IntoIterator<Item = Complex<f64>>,
) {
    for (mut row, gain) in freq_rate_array.rows_mut().into_iter().zip(gains) {
        if gain != Complex::new(1.0, 0.0) {
            for value in &mut row {
                let corrected = Complex::new(f64::from(value.re), f64::from(value.im)) * gain;
                *value = C32::new(corrected.re as f32, corrected.im as f32);
            }
        }
    }
}

pub fn apply_bandpass_correction(freq_rate_array: &mut Array2<C32>, bandpass_data: &[C32]) {
    apply_bandpass_gains(freq_rate_array, bandpass_gains(bandpass_data));
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
        io::Error::other(e.to_string())
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

#[cfg(test)]
mod correction_tests {
    use super::*;

    #[test]
    fn finite_high_gain_calibration_handles_zero_and_missing_channels() {
        let bandpass = [C32::new(1e30, 0.0), C32::new(0.0, 1e30), C32::default()];
        let input = C32::new(1.0, 0.0);
        let mut plane = Array2::from_elem((4, 2), input);
        apply_bandpass_correction(&mut plane, &bandpass);

        let third = 1.0 / 3.0;
        let expected = [
            C32::new(third, third),
            C32::new(third, -third),
            input,
            input,
        ];
        for (row, gain) in plane.rows().into_iter().zip(expected) {
            assert!(row.iter().all(|&v| (v - gain).norm() < 1e-6));
        }
    }
}
