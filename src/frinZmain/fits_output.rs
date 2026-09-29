use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

const FITS_BLOCK_SIZE: usize = 2880;
const PIXEL_BUFFER_SIZE: usize = 64 * 1024;

pub struct FitsAxis<'a> {
    pub name: &'static str,
    pub unit: &'static str,
    pub values: &'a [f32],
}

pub struct FitsMetadata<'a> {
    pub source_name: &'a str,
    pub date_obs: &'a str,
    pub observing_frequency_hz: f64,
}

/// Write a two-dimensional, single-precision FITS primary image.
///
/// Axis 1 is the horizontal image axis and axis 2 is the vertical image axis.
/// Pixels are consumed in FITS row-major order, so callers can stream a
/// transposed view without allocating a second image-sized buffer.
pub fn write_fits_image<I>(
    path: &Path,
    axis1: FitsAxis<'_>,
    axis2: FitsAxis<'_>,
    metadata: FitsMetadata<'_>,
    pixels: I,
) -> io::Result<()>
where
    I: IntoIterator<Item = f32>,
{
    if axis1.values.is_empty() || axis2.values.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "FITS image axes must not be empty",
        ));
    }

    let width = axis1.values.len();
    let height = axis2.values.len();
    let pixel_count = width.checked_mul(height).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "FITS image dimensions overflow",
        )
    })?;
    let data_bytes = pixel_count.checked_mul(4).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "FITS image byte count overflows",
        )
    })?;

    let mut header = Vec::with_capacity(20 * 80);
    push_card(&mut header, "SIMPLE  =                    T");
    push_card(&mut header, "BITPIX  =                  -32");
    push_card(&mut header, "NAXIS   =                    2");
    push_integer_card(&mut header, "NAXIS1", width)?;
    push_integer_card(&mut header, "NAXIS2", height)?;
    push_card(&mut header, "EXTEND  =                    T");
    push_string_card(&mut header, "BTYPE", "Fringe amplitude");
    push_string_card(&mut header, "BUNIT", "1");
    push_string_card(&mut header, "OBJECT", metadata.source_name);
    push_string_card(&mut header, "DATE-OBS", metadata.date_obs);
    push_float_card(&mut header, "OBSFREQ", metadata.observing_frequency_hz);
    push_card(&mut header, "WCSAXES =                    2");
    push_string_card(&mut header, "CTYPE1", axis1.name);
    push_string_card(&mut header, "CUNIT1", axis1.unit);
    push_float_card(&mut header, "CRPIX1", 1.0);
    push_float_card(&mut header, "CRVAL1", axis1.values[0] as f64);
    push_float_card(&mut header, "CDELT1", axis_step(axis1.values));
    push_string_card(&mut header, "CTYPE2", axis2.name);
    push_string_card(&mut header, "CUNIT2", axis2.unit);
    push_float_card(&mut header, "CRPIX2", 1.0);
    push_float_card(&mut header, "CRVAL2", axis2.values[0] as f64);
    push_float_card(&mut header, "CDELT2", axis_step(axis2.values));
    push_card(&mut header, "END");
    let header_padding = (FITS_BLOCK_SIZE - header.len() % FITS_BLOCK_SIZE) % FITS_BLOCK_SIZE;
    header.resize(header.len() + header_padding, b' ');

    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(PIXEL_BUFFER_SIZE, file);
    writer.write_all(&header)?;

    let mut buffer = [0_u8; PIXEL_BUFFER_SIZE];
    let mut buffered_bytes = 0;
    let mut written_pixels = 0usize;
    for pixel in pixels {
        if written_pixels == pixel_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "FITS pixel iterator contains more values than the image dimensions",
            ));
        }
        let encoded = pixel.to_be_bytes();
        buffer[buffered_bytes..buffered_bytes + encoded.len()].copy_from_slice(&encoded);
        buffered_bytes += encoded.len();
        written_pixels += 1;

        if buffered_bytes == buffer.len() {
            writer.write_all(&buffer)?;
            buffered_bytes = 0;
        }
    }

    if written_pixels != pixel_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "FITS pixel iterator contains fewer values than the image dimensions",
        ));
    }
    if buffered_bytes > 0 {
        writer.write_all(&buffer[..buffered_bytes])?;
    }

    let data_padding = (FITS_BLOCK_SIZE - data_bytes % FITS_BLOCK_SIZE) % FITS_BLOCK_SIZE;
    if data_padding > 0 {
        const ZERO_PADDING: [u8; FITS_BLOCK_SIZE] = [0; FITS_BLOCK_SIZE];
        writer.write_all(&ZERO_PADDING[..data_padding])?;
    }
    writer.flush()
}

fn axis_step(values: &[f32]) -> f64 {
    if values.len() > 1 {
        ((values[values.len() - 1] as f64) - (values[0] as f64)) / (values.len() - 1) as f64
    } else {
        1.0
    }
}

fn push_card(header: &mut Vec<u8>, card: &str) {
    let mut card_bytes = [b' '; 80];
    let bytes = card.as_bytes();
    let copy_len = bytes.len().min(card_bytes.len());
    card_bytes[..copy_len].copy_from_slice(&bytes[..copy_len]);
    header.extend_from_slice(&card_bytes);
}

fn push_integer_card(header: &mut Vec<u8>, key: &str, value: usize) -> io::Result<()> {
    let value = i64::try_from(value).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "FITS axis length exceeds the supported integer range",
        )
    })?;
    push_card(header, &format!("{key:<8}= {value:>20}"));
    Ok(())
}

fn push_float_card(header: &mut Vec<u8>, key: &str, value: f64) {
    push_card(header, &format!("{key:<8}= {value:>20.12E}"));
}

fn push_string_card(header: &mut Vec<u8>, key: &str, value: &str) {
    let mut escaped = String::new();
    for character in value.chars() {
        let character = if character.is_ascii() && !character.is_ascii_control() {
            character
        } else {
            '?'
        };
        let extra_len = if character == '\'' { 2 } else { 1 };
        if escaped.len() + extra_len > 68 {
            break;
        }
        if character == '\'' {
            escaped.push_str("''");
        } else {
            escaped.push(character);
        }
    }
    push_card(header, &format!("{key:<8}= '{escaped}'"));
}
