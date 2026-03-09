use imagequant::{new as iq_new, Attributes, Histogram, RGBA};
use png::{AdaptiveFilterType, BitDepth, ColorType, Encoder};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

static WARNED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy)]
pub enum CompressQuality {
    High,
    Low,
}

fn warn_once(msg: &str) {
    if !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!("#WARN: {}", msg);
    }
}

/// Try to compress a PNG in-place via the imagequant crate.
pub fn compress_png<P: AsRef<Path>>(path: P) {
    compress_png_with_mode(path, CompressQuality::High);
}

pub fn compress_png_with_mode<P: AsRef<Path>>(path: P, mode: CompressQuality) {
    let path_buf: PathBuf = path.as_ref().to_path_buf();
    if !path_buf.exists() {
        return;
    }

    let original = match fs::read(&path_buf) {
        Ok(data) => data,
        Err(_) => {
            warn_once("Failed to read PNG for compression.");
            return;
        }
    };

    let decoded = match image::load_from_memory(&original) {
        Ok(img) => img.to_rgba8(),
        Err(_) => {
            warn_once("Failed to decode PNG for compression.");
            return;
        }
    };
    let (width, height) = decoded.dimensions();
    let pixels: Vec<RGBA> = decoded
        .chunks_exact(4)
        .map(|px| RGBA::new(px[0], px[1], px[2], px[3]))
        .collect();

    let mut base_attr = iq_new();
    let _ = base_attr.set_speed(1);
    match mode {
        CompressQuality::High => {
            let _ = base_attr.set_quality(10, 100);
        }
        CompressQuality::Low => {
            let _ = base_attr.set_quality(10, 10);
        }
    }
    let _ = base_attr.set_max_colors(256);
    let _ = base_attr.set_min_posterization(0);

    let color_targets = [
        256u32, 224, 192, 160, 144, 
        128, 112, 96, 80, 72, 
        64, 56, 48, 40, 32, 28, 24, 20, 16, 12, 10, 8,
    ];
    let mut best_png: Option<Vec<u8>> = None;
    for &colors in &color_targets {
        if let Some(candidate) =
            quantize_with_colors(&base_attr, &pixels, width, height, colors)
        {
            let take = best_png
                .as_ref()
                .map(|current| candidate.len() < current.len())
                .unwrap_or(true);
            if take {
                best_png = Some(candidate);
            }
            if let Some(best) = &best_png {
                if best.len() * 3 < original.len() {
                    break;
                }
            }
        }
    }

    if let Some(best) = best_png {
        if best.len() < original.len() {
            if let Err(_) = fs::write(&path_buf, best) {
                warn_once("Failed to write quantized PNG.");
            }
        }
    }
}

fn quantize_with_colors(
    base_attr: &Attributes,
    pixels: &[RGBA],
    width: u32,
    height: u32,
    max_colors: u32,
) -> Option<Vec<u8>> {
    let mut attr = base_attr.clone();
    attr.set_max_colors(max_colors).ok()?;
    let posterization = if max_colors <= 32 {
        3
    } else if max_colors <= 64 {
        2
    } else if max_colors <= 96 {
        1
    } else {
        0
    };
    attr.set_min_posterization(posterization).ok()?;

    let mut image = attr
        .new_image_borrowed(pixels, width as usize, height as usize, 0.0)
        .ok()?;
    let mut histogram = Histogram::new(&attr);
    histogram.add_image(&attr, &mut image).ok()?;

    let mut result = histogram.quantize(&attr).ok()?;
    let dithering_level = if max_colors <= 24 {
        0.2
    } else if max_colors <= 48 {
        0.35
    } else if max_colors <= 96 {
        0.8
    } else {
        1.0
    };
    result.set_dithering_level(dithering_level).ok();
    let (palette, indices) = result.remapped(&mut image).ok()?;
    encode_indexed_png(&palette, &indices, width, height)
}

fn encode_indexed_png(
    palette: &[RGBA],
    indices: &[u8],
    width: u32,
    height: u32,
) -> Option<Vec<u8>> {
    let mut palette_rgb = Vec::with_capacity(palette.len() * 3);
    let mut trns = Vec::with_capacity(palette.len());
    let mut has_transparency = false;
    for rgba in palette.iter() {
        palette_rgb.extend_from_slice(&[rgba.r, rgba.g, rgba.b]);
        if rgba.a != 255 {
            has_transparency = true;
        }
        trns.push(rgba.a);
    }

    let mut compressed = Vec::new();
    {
        let mut encoder = Encoder::new(&mut compressed, width, height);
        encoder.set_color(ColorType::Indexed);
        encoder.set_depth(BitDepth::Eight);
        encoder.set_compression(png::Compression::Best);
        encoder.set_filter(png::FilterType::Paeth);
        encoder.set_adaptive_filter(AdaptiveFilterType::Adaptive);
        encoder.set_palette(palette_rgb);
        if has_transparency {
            encoder.set_trns(trns);
        }
        let mut writer = match encoder.write_header() {
            Ok(w) => w,
            Err(_) => return None,
        };
        if writer.write_image_data(indices).is_err() {
            return None;
        }
    }
    Some(compressed)
}
