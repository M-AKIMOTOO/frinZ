use image::{
    imageops::{resize, FilterType},
    DynamicImage, GenericImageView, ImageReader, Rgba, RgbaImage,
};
use std::error::Error;
use std::io::{self, Write};
use terminal_size::{terminal_size, Height, Width};

// The logo PNG is placed in the assets/ directory directly under the project.
const LOGO_PNG: &[u8] = include_bytes!("./logo1.png");

// ===== Default Parameters (rewrite only the numbers if necessary) =====
const SCALE: f32 = 0.5; // Scale factor for the maximum size that fits in the terminal (fixed at 0.5 = half)
const CELL_ASPECT: f32 = 2.0; // Height:width of a single character cell (most terminals are ~2.0)
const MARGIN_X: u16 = 2; // Left and right margin (number of characters)
const MARGIN_Y: u16 = 1; // Top and bottom margin (number of lines)
const BG_RGB: [u8; 3] = [255, 255, 255]; // Background for transparent parts (white)
const GAMMA: f32 = 2.2; // Gamma for sRGB <-> linear
                        // =====================================================

pub fn show_logo() -> Result<(), Box<dyn Error>> {
    // Load PNG
    let img = ImageReader::new(std::io::Cursor::new(LOGO_PNG))
        .with_guessed_format()?
        .decode()?;

    // Display with Braille (2x4 dots/character)
    show_braille_auto(&img)?;
    Ok(())
}

fn show_braille_auto(img: &DynamicImage) -> Result<(), Box<dyn Error>> {
    // Image aspect ratio
    let (w, h) = img.dimensions();
    let ar_img = h as f32 / w as f32;

    // Terminal size (columns, rows)
    let (term_cols, term_rows) = match terminal_size() {
        Some((Width(c), Height(r))) => (c, r),
        None => (80, 24), // Default size
    };

    // Available area (subtracting margins)
    let cols_avail = term_cols.saturating_sub(MARGIN_X);
    let rows_avail = term_rows.saturating_sub(MARGIN_Y);

    // Calculate allowed columns from height constraint: rows = cols * ar_img / CELL_ASPECT
    let by_height = (rows_avail as f32) * CELL_ASPECT / ar_img;
    // Width constraint: as is
    let by_width = cols_avail as f32;

    // Take the smaller of the two for the number of columns and apply SCALE
    let mut cols_chars = (by_width.min(by_height) * SCALE).floor() as u16;
    cols_chars = cols_chars.max(1);

    // Corresponding number of rows
    let mut rows_chars = ((cols_chars as f32) * ar_img / CELL_ASPECT).ceil() as u16;
    rows_chars = rows_chars.max(1);

    // Braille is 1 character = 2 wide x 4 high dots
    let target_w = (cols_chars as u32) * 4;
    let target_h = (rows_chars as u32) * 5;

    // High-quality shrink (sRGB->linear->background blend->Lanczos3->sRGB)
    let sub = shrink_with_gamma_and_bg(img, target_w, target_h, BG_RGB, GAMMA)?;

    // Output
    print_braille_cells(&sub)?;
    Ok(())
}

/// sRGB->linear->background blend->Lanczos shrink->sRGB back
fn shrink_with_gamma_and_bg(
    img: &DynamicImage,
    target_w: u32,
    target_h: u32,
    bg_rgb: [u8; 3],
    gamma: f32,
) -> Result<RgbaImage, Box<dyn Error>> {
    let lin = prepare_linear_rgba(img, bg_rgb, gamma);
    let lin_resized = resize(&lin, target_w.max(1), target_h.max(1), FilterType::Lanczos3);

    let mut out = RgbaImage::new(lin_resized.width(), lin_resized.height());
    for (x, y, p) in out.enumerate_pixels_mut() {
        let c = lin_resized.get_pixel(x, y);
        *p = Rgba([
            float_clamp255((c[0] as f32 / 255.0).powf(1.0 / gamma)),
            float_clamp255((c[1] as f32 / 255.0).powf(1.0 / gamma)),
            float_clamp255((c[2] as f32 / 255.0).powf(1.0 / gamma)),
            255,
        ]);
    }
    Ok(out)
}

/// sRGB->linear, background blend (premult black fringe countermeasure)
fn prepare_linear_rgba(img: &DynamicImage, bg_rgb: [u8; 3], gamma: f32) -> RgbaImage {
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let mut out = RgbaImage::new(w, h);
    for (x, y, p) in rgba.enumerate_pixels() {
        let a = p[3] as f32 / 255.0;
        let sr = ((p[0] as f32) * a + (bg_rgb[0] as f32) * (1.0 - a)) / 255.0;
        let sg = ((p[1] as f32) * a + (bg_rgb[1] as f32) * (1.0 - a)) / 255.0;
        let sb = ((p[2] as f32) * a + (bg_rgb[2] as f32) * (1.0 - a)) / 255.0;
        out.put_pixel(
            x,
            y,
            Rgba([
                float_clamp255(sr.powf(gamma)),
                float_clamp255(sg.powf(gamma)),
                float_clamp255(sb.powf(gamma)),
                255,
            ]),
        );
    }
    out
}

/// Draw with Braille (2x4 dots/character) (average color)
fn print_braille_cells(sub: &RgbaImage) -> Result<(), Box<dyn Error>> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    let w = sub.width();
    let h = sub.height();

    for by in (0..h).step_by(4) {
        for bx in (0..w).step_by(2) {
            // Use average color of 2x4 subpixels for foreground
            let mut sr = 0u32;
            let mut sg = 0u32;
            let mut sb = 0u32;
            let mut mask: u8 = 0;
            let mut pixel_count = 0;

            for dy in 0..4 {
                for dx in 0..2 {
                    if bx + dx < w && by + dy < h {
                        pixel_count += 1;
                        let p = sub.get_pixel(bx + dx, by + dy).0;
                        sr += p[0] as u32;
                        sg += p[1] as u32;
                        sb += p[2] as u32;

                        // Set points based on lightness threshold (simple ordered dither)
                        let lum = 0.2126 * (p[0] as f32)
                            + 0.7152 * (p[1] as f32)
                            + 0.0722 * (p[2] as f32);
                        let thresh =
                            128.0 + [0.0, 16.0, 8.0, 24.0][dy as usize] + (dx as f32) * 8.0;
                        if lum > thresh {
                            mask |= match (dx, dy) {
                                (0, 0) => 0x01,
                                (0, 1) => 0x02,
                                (0, 2) => 0x04,
                                (0, 3) => 0x40,
                                (1, 0) => 0x08,
                                (1, 1) => 0x10,
                                (1, 2) => 0x20,
                                (1, 3) => 0x80,
                                _ => 0,
                            };
                        }
                    }
                }
            }

            if pixel_count > 0 {
                let avg = [
                    (sr / pixel_count) as u8,
                    (sg / pixel_count) as u8,
                    (sb / pixel_count) as u8,
                ];
                let ch = char::from_u32(0x2800 + mask as u32).unwrap_or(' ');
                write!(out, "\x1b[38;2;{};{};{}m{}", avg[0], avg[1], avg[2], ch)?;
            }
        }
        writeln!(out, "\x1b[0m")?;
    }
    out.flush()?;
    Ok(())
}

#[inline]
fn float_clamp255(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
