use anyhow::{Context, Result};
use image::GenericImageView;
use num_complex::Complex;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::{Path, PathBuf};

pub(crate) const PLOT_FONT_SCALE: f64 = 1.2;
pub(crate) const LEGEND_FONT_SCALE: f64 = 1.2;
pub(crate) const RFI_WINDOW_RADIUS: usize = 4;
pub(crate) const RFI_SIGMA_CUT: f64 = 6.0;
pub(crate) const RFI_RATIO_CUT: f64 = 2.5;

#[derive(Debug, Clone)]
pub(crate) struct RfiCutReport {
    pub(crate) total_channels: usize,
    pub(crate) masked_channels: Vec<usize>,
    pub(crate) channel_mean_amplitudes: Vec<f64>,
    pub(crate) local_reference_amplitudes: Vec<f64>,
    pub(crate) window_radius: usize,
    pub(crate) sigma_cut: f64,
    pub(crate) ratio_cut: f64,
}

impl RfiCutReport {
    pub(crate) fn masked_count(&self) -> usize {
        self.masked_channels.len()
    }
}

pub(crate) fn compress_plot_png(path: &Path) {
    let Ok(metadata) = fs::metadata(path) else {
        return;
    };
    let Ok(image) = image::open(path) else {
        return;
    };
    let rgba = image.to_rgba8();
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return;
    }

    let mut attr = imagequant::Attributes::new();
    if attr.set_speed(3).is_err() || attr.set_quality(55, 80).is_err() {
        return;
    }

    let pixels: Vec<imagequant::RGBA> = rgba
        .pixels()
        .map(|px| imagequant::RGBA::new(px[0], px[1], px[2], px[3]))
        .collect();
    let Ok(mut quant_image) = attr.new_image(pixels, width as usize, height as usize, 0.0) else {
        return;
    };
    let Ok(mut result) = attr.quantize(&mut quant_image) else {
        return;
    };
    let _ = result.set_dithering_level(0.0);
    let Ok((palette, indexed_pixels)) = result.remapped(&mut quant_image) else {
        return;
    };
    if palette.is_empty() || indexed_pixels.is_empty() {
        return;
    }

    let tmp_path = path.with_extension("imagequant.tmp.png");
    let encode_result = (|| -> Result<()> {
        let file = File::create(&tmp_path)
            .with_context(|| format!("failed to create {}", tmp_path.display()))?;
        let writer = BufWriter::new(file);
        let mut encoder = png::Encoder::new(writer, width, height);
        encoder.set_color(png::ColorType::Indexed);
        encoder.set_depth(png::BitDepth::Eight);

        let mut palette_bytes = Vec::with_capacity(palette.len() * 3);
        let mut alpha_bytes = Vec::with_capacity(palette.len());
        for color in &palette {
            palette_bytes.extend_from_slice(&[color.r, color.g, color.b]);
            alpha_bytes.push(color.a);
        }
        encoder.set_palette(palette_bytes);
        if alpha_bytes.iter().any(|&alpha| alpha < u8::MAX) {
            encoder.set_trns(alpha_bytes);
        }

        let mut png_writer = encoder.write_header()?;
        png_writer.write_image_data(&indexed_pixels)?;
        Ok(())
    })();
    if encode_result.is_err() {
        let _ = fs::remove_file(&tmp_path);
        return;
    }

    let Ok(tmp_metadata) = fs::metadata(&tmp_path) else {
        let _ = fs::remove_file(&tmp_path);
        return;
    };
    if tmp_metadata.len() >= metadata.len() {
        let _ = fs::remove_file(&tmp_path);
        return;
    }

    let _ = fs::rename(&tmp_path, path).or_else(|_| {
        fs::copy(&tmp_path, path)?;
        fs::remove_file(&tmp_path)
    });
}

pub(crate) fn scaled_font_size(size: i32) -> i32 {
    ((size as f64) * PLOT_FONT_SCALE).round() as i32
}

pub(crate) fn scaled_legend_font_size(size: i32) -> i32 {
    ((scaled_font_size(size) as f64) * LEGEND_FONT_SCALE).round() as i32
}

pub(crate) fn output_stem(input: &Path) -> String {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("pulsar")
        .to_string();
    stem.strip_suffix(".cor")
        .unwrap_or(stem.as_str())
        .to_string()
}

pub(crate) fn prepare_output_directory(input: &Path) -> Result<PathBuf> {
    let parent = input.parent().unwrap_or_else(|| Path::new(""));
    let target_name = output_stem(input);
    let output_dir = parent.join("frinZ").join("pulsar_gating").join(target_name);
    fs::create_dir_all(&output_dir)?;
    Ok(output_dir)
}

pub(crate) fn detect_rfi_cut(rows: &[&[Complex<f32>]]) -> RfiCutReport {
    let total_channels = rows.iter().map(|row| row.len()).max().unwrap_or(0);
    let mut channel_sums = vec![0.0f64; total_channels];
    let mut channel_counts = vec![0usize; total_channels];

    for row in rows {
        for (chan_idx, value) in row.iter().enumerate() {
            if value.re.is_finite() && value.im.is_finite() {
                channel_sums[chan_idx] += value.norm() as f64;
                channel_counts[chan_idx] += 1;
            }
        }
    }

    let channel_mean_amplitudes: Vec<f64> = channel_sums
        .iter()
        .zip(channel_counts.iter())
        .map(|(&sum, &count)| {
            if count > 0 {
                sum / count as f64
            } else {
                f64::NAN
            }
        })
        .collect();

    let mut local_reference_amplitudes = vec![f64::NAN; total_channels];
    let mut masked_channels = Vec::new();
    for chan_idx in 0..total_channels {
        let amp = channel_mean_amplitudes[chan_idx];
        if !amp.is_finite() {
            masked_channels.push(chan_idx);
            continue;
        }

        let start = chan_idx.saturating_sub(RFI_WINDOW_RADIUS);
        let end = (chan_idx + RFI_WINDOW_RADIUS + 1).min(total_channels);
        let mut neighbors: Vec<f64> = (start..end)
            .filter(|&idx| idx != chan_idx)
            .filter_map(|idx| {
                let v = channel_mean_amplitudes[idx];
                v.is_finite().then_some(v)
            })
            .collect();
        if neighbors.len() < 3 {
            continue;
        }

        neighbors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let local_median = median_of_sorted(&neighbors);
        local_reference_amplitudes[chan_idx] = local_median;

        let mut deviations: Vec<f64> = neighbors.iter().map(|v| (v - local_median).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let local_mad = median_of_sorted(&deviations);
        let local_sigma = local_mad * 1.4826;
        let ratio = if local_median > 0.0 {
            amp / local_median
        } else if amp > 0.0 {
            f64::INFINITY
        } else {
            1.0
        };
        let high_sigma_outlier =
            local_sigma > 0.0 && amp > local_median + RFI_SIGMA_CUT * local_sigma;
        let high_ratio_outlier = ratio >= RFI_RATIO_CUT && amp > local_median;
        if high_sigma_outlier || high_ratio_outlier {
            masked_channels.push(chan_idx);
        }
    }

    RfiCutReport {
        total_channels,
        masked_channels,
        channel_mean_amplitudes,
        local_reference_amplitudes,
        window_radius: RFI_WINDOW_RADIUS,
        sigma_cut: RFI_SIGMA_CUT,
        ratio_cut: RFI_RATIO_CUT,
    }
}

pub(crate) fn write_rfi_cut_report(
    path: &Path,
    freq_axis_mhz: &[f64],
    report: &RfiCutReport,
) -> Result<()> {
    let mut file =
        fs::File::create(path).with_context(|| format!("failed to write {}", path.display()))?;
    writeln!(file, "channel,freq_mhz,mean_amp,local_ref_amp,masked")?;
    let mut masked = vec![false; report.total_channels];
    for &chan_idx in &report.masked_channels {
        if chan_idx < masked.len() {
            masked[chan_idx] = true;
        }
    }
    for chan_idx in 0..report.total_channels {
        let freq = freq_axis_mhz.get(chan_idx).copied().unwrap_or(f64::NAN);
        let mean_amp = report
            .channel_mean_amplitudes
            .get(chan_idx)
            .copied()
            .unwrap_or(f64::NAN);
        let local_ref = report
            .local_reference_amplitudes
            .get(chan_idx)
            .copied()
            .unwrap_or(f64::NAN);
        writeln!(
            file,
            "{},{:.9},{:.9},{:.9},{}",
            chan_idx, freq, mean_amp, local_ref, masked[chan_idx]
        )?;
    }
    Ok(())
}

fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}
