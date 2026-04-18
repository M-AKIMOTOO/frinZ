use png::{AdaptiveFilterType, BitDepth, ColorType, Encoder, FilterType};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

static WARNED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy)]
pub enum CompressQuality {
    High,
    Low,
}

#[derive(Clone, Copy)]
enum PixelLayout {
    Grayscale,
    GrayscaleAlpha,
    Rgb,
    Rgba,
}

impl PixelLayout {
    fn color_type(self) -> ColorType {
        match self {
            Self::Grayscale => ColorType::Grayscale,
            Self::GrayscaleAlpha => ColorType::GrayscaleAlpha,
            Self::Rgb => ColorType::Rgb,
            Self::Rgba => ColorType::Rgba,
        }
    }
}

#[derive(Default)]
struct ImageAnalysis {
    opaque: bool,
    grayscale: bool,
}

fn warn_once(msg: &str) {
    if !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!("#WARN: {}", msg);
    }
}

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

    if try_pngquant(&path_buf, original.len(), mode) {
        return;
    }

    let decoded = match image::load_from_memory(&original) {
        Ok(img) => img.to_rgba8(),
        Err(_) => {
            warn_once("Failed to decode PNG for compression.");
            return;
        }
    };

    let (width, height) = decoded.dimensions();
    let rgba_bytes = decoded.as_raw();
    let analysis = analyze_rgba(rgba_bytes);

    let mut best_png: Option<Vec<u8>> = None;

    if let Some(candidate) = encode_best_lossless_png(rgba_bytes, width, height, PixelLayout::Rgba)
    {
        update_best(&mut best_png, candidate);
    }

    if analysis.opaque {
        let rgb_bytes = rgba_to_rgb(rgba_bytes);
        if let Some(candidate) =
            encode_best_lossless_png(&rgb_bytes, width, height, PixelLayout::Rgb)
        {
            update_best(&mut best_png, candidate);
        }

        if analysis.grayscale {
            let gray_bytes = rgba_to_gray(rgba_bytes);
            if let Some(candidate) =
                encode_best_lossless_png(&gray_bytes, width, height, PixelLayout::Grayscale)
            {
                update_best(&mut best_png, candidate);
            }
        }
    } else if analysis.grayscale {
        let gray_alpha_bytes = rgba_to_gray_alpha(rgba_bytes);
        if let Some(candidate) = encode_best_lossless_png(
            &gray_alpha_bytes,
            width,
            height,
            PixelLayout::GrayscaleAlpha,
        ) {
            update_best(&mut best_png, candidate);
        }
    }

    if let Some(best) = best_png {
        if best.len() < original.len() && fs::write(&path_buf, best).is_err() {
            warn_once("Failed to write recompressed PNG.");
        }
    }
}

fn try_pngquant(path: &Path, original_size: usize, mode: CompressQuality) -> bool {
    let (quality_min, quality_max) = match mode {
        CompressQuality::High => ("60", "80"),
        CompressQuality::Low => ("50", "70"),
    };
    let quality_arg = format!("{}-{}", quality_min, quality_max);
    let output_path = path.with_extension("pngquant.tmp.png");

    let status = Command::new("pngquant")
        .arg("--force")
        .arg("--skip-if-larger")
        .arg("--strip")
        .arg("--speed")
        .arg("1")
        .arg("--quality")
        .arg(&quality_arg)
        .arg("--output")
        .arg(&output_path)
        .arg("--")
        .arg(path)
        .status();

    let Ok(status) = status else {
        return false;
    };

    if !status.success() {
        let _ = fs::remove_file(&output_path);
        return false;
    }

    let Ok(candidate) = fs::read(&output_path) else {
        let _ = fs::remove_file(&output_path);
        return false;
    };

    let _ = fs::remove_file(&output_path);

    if candidate.len() >= original_size {
        return false;
    }

    if fs::write(path, candidate).is_err() {
        warn_once("Failed to write pngquant-compressed PNG.");
        return false;
    }

    true
}

fn analyze_rgba(bytes: &[u8]) -> ImageAnalysis {
    let mut opaque = true;
    let mut grayscale = true;

    for px in bytes.chunks_exact(4) {
        if px[3] != 255 {
            opaque = false;
        }
        if !(px[0] == px[1] && px[1] == px[2]) {
            grayscale = false;
        }
        if !opaque && !grayscale {
            break;
        }
    }

    ImageAnalysis { opaque, grayscale }
}

fn rgba_to_rgb(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for px in bytes.chunks_exact(4) {
        out.extend_from_slice(&px[..3]);
    }
    out
}

fn rgba_to_gray(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for px in bytes.chunks_exact(4) {
        out.push(px[0]);
    }
    out
}

fn rgba_to_gray_alpha(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for px in bytes.chunks_exact(4) {
        out.push(px[0]);
        out.push(px[3]);
    }
    out
}

fn update_best(best: &mut Option<Vec<u8>>, candidate: Vec<u8>) {
    let replace = best
        .as_ref()
        .map(|current| candidate.len() < current.len())
        .unwrap_or(true);
    if replace {
        *best = Some(candidate);
    }
}

fn encode_best_lossless_png(
    raw: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
) -> Option<Vec<u8>> {
    let strategies = [
        (FilterType::NoFilter, AdaptiveFilterType::NonAdaptive),
        (FilterType::Sub, AdaptiveFilterType::NonAdaptive),
        (FilterType::Paeth, AdaptiveFilterType::NonAdaptive),
        (FilterType::Avg, AdaptiveFilterType::NonAdaptive),
        (FilterType::Sub, AdaptiveFilterType::Adaptive),
    ];

    let mut best: Option<Vec<u8>> = None;
    for (filter, adaptive) in strategies {
        if let Some(candidate) = encode_raw_png(
            raw,
            width,
            height,
            layout,
            BitDepth::Eight,
            filter,
            adaptive,
        ) {
            update_best(&mut best, candidate);
        }
    }
    best
}

fn encode_raw_png(
    raw: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
    bit_depth: BitDepth,
    filter: FilterType,
    adaptive: AdaptiveFilterType,
) -> Option<Vec<u8>> {
    let mut encoded = Vec::new();
    {
        let mut encoder = Encoder::new(&mut encoded, width, height);
        encoder.set_color(layout.color_type());
        encoder.set_depth(bit_depth);
        encoder.set_compression(png::Compression::Best);
        encoder.set_filter(filter);
        encoder.set_adaptive_filter(adaptive);

        let mut writer = encoder.write_header().ok()?;
        writer.write_image_data(raw).ok()?;
    }
    Some(encoded)
}
