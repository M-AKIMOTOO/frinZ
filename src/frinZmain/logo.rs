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
const SCALE_X: f32 = 1.5; // Current 0.75 width, doubled horizontally.
const SCALE_Y: f32 = 0.75; // Keep the current vertical size.
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

    // Fit the unscaled logo to the terminal, then apply independent x/y scales.
    // The x dimension is allowed to grow, but never wraps beyond the terminal.
    let base_cols = by_width.min(by_height).max(1.0);
    let mut cols_chars = (base_cols * SCALE_X).floor() as u16;
    cols_chars = cols_chars.min(cols_avail).max(1);
    let base_rows = base_cols * ar_img / CELL_ASPECT;
    let mut rows_chars = (base_rows * SCALE_Y).ceil() as u16;
    rows_chars = rows_chars.min(rows_avail).max(1);

    // A Braille cell represents two horizontal by four vertical image pixels.
    let target_w = (cols_chars as u32) * 2;
    let target_h = (rows_chars as u32) * 4;

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

/// Draw a 2x4 TrueColor Braille cell.
///
/// The white background in logo1.png is treated as transparent so the logo
/// works on both dark and light terminal backgrounds.  Braille provides a
/// denser outline than half-blocks while keeping the displayed logo small.
fn print_braille_cells(sub: &RgbaImage) -> Result<(), Box<dyn Error>> {
    const DOT_BITS: [[u8; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];
    const INK_THRESHOLD: u8 = 6;

    let mut out = io::BufWriter::new(io::stdout().lock());
    let w = sub.width();
    let h = sub.height();

    let ink_strength = |p: [u8; 4]| -> u16 {
        if p[3] <= 16 {
            return 0;
        }
        let distance = p[0].abs_diff(BG_RGB[0]) as u16
            + p[1].abs_diff(BG_RGB[1]) as u16
            + p[2].abs_diff(BG_RGB[2]) as u16;
        if distance < INK_THRESHOLD as u16 {
            0
        } else {
            distance
        }
    };

    for y0 in (0..h).step_by(4) {
        for x0 in (0..w).step_by(2) {
            let mut mask = 0u8;
            let mut rgb_sum = [0u32; 3];
            let mut rgb_weight = 0u32;
            for dy in 0..4u32 {
                for dx in 0..2u32 {
                    let x = x0 + dx;
                    let y = y0 + dy;
                    if x >= w || y >= h {
                        continue;
                    }
                    let p = sub.get_pixel(x, y).0;
                    let strength = ink_strength(p);
                    if strength == 0 {
                        continue;
                    }
                    mask |= DOT_BITS[dx as usize][dy as usize];
                    rgb_weight += strength as u32;
                    rgb_sum[0] += p[0] as u32 * strength as u32;
                    rgb_sum[1] += p[1] as u32 * strength as u32;
                    rgb_sum[2] += p[2] as u32 * strength as u32;
                }
            }

            if mask == 0 {
                // Keep the original dotted presentation: the white logo field
                // is made from white Braille dots rather than a solid block.
                write!(out, "\x1b[38;2;255;255;255m⣿\x1b[0m")?;
                continue;
            }
            let rgb = [
                (rgb_sum[0] / rgb_weight) as u8,
                (rgb_sum[1] / rgb_weight) as u8,
                (rgb_sum[2] / rgb_weight) as u8,
            ];
            let ch = char::from_u32(0x2800 + mask as u32).unwrap_or('⣿');
            write!(
                out,
                "\x1b[38;2;{};{};{}m{}\x1b[0m",
                rgb[0], rgb[1], rgb[2], ch
            )?;
        }
        writeln!(out)?;
    }
    out.flush()?;
    Ok(())
}

#[inline]
fn float_clamp255(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
