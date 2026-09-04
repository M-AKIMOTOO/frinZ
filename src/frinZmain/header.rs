use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{self, Cursor, ErrorKind, Read};

const COR_MAGIC: [u8; 4] = [0x83, 0xf9, 0xa2, 0x3e];
const FILE_HEADER_SIZE: usize = 256;
const SECTOR_HEADER_SIZE: usize = 128;

#[derive(Debug, Default, Clone)]
pub struct CorHeader {
    pub magic_word: [u8; 4],
    pub header_version: i32,
    pub software_version: i32,
    pub sampling_speed: i32,
    pub observing_frequency: f64,
    pub fft_point: i32,
    pub number_of_sector: i32,
    pub station1_name: String,
    pub station1_code: String,
    pub station1_position: [f64; 3],
    pub station2_name: String,
    pub station2_code: String,
    pub station2_position: [f64; 3],
    pub source_name: String,
    pub source_position_ra: f64,
    pub source_position_dec: f64,
    pub station1_clock_delay: f64,
    pub station1_clock_rate: f64,
    pub station1_clock_acel: f64,
    pub station1_clock_jerk: f64,
    pub station1_clock_snap: f64,
    pub station2_clock_delay: f64,
    pub station2_clock_rate: f64,
    pub station2_clock_acel: f64,
    pub station2_clock_jerk: f64,
    pub station2_clock_snap: f64,
}

fn validate_header_fields(header: &CorHeader) -> io::Result<()> {
    if header.magic_word != COR_MAGIC {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "invalid .cor magic word: {:02x?} (expected {:02x?})",
                header.magic_word, COR_MAGIC
            ),
        ));
    }
    if header.sampling_speed <= 0 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("invalid sampling speed: {}", header.sampling_speed),
        ));
    }
    if header.fft_point < 4 || header.fft_point % 4 != 0 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "invalid FFT point: {} (must be a positive multiple of 4)",
                header.fft_point
            ),
        ));
    }
    if header.number_of_sector <= 0 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "invalid sector count: {} (must be positive)",
                header.number_of_sector
            ),
        ));
    }
    if !header.observing_frequency.is_finite() || header.observing_frequency < 0.0 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "invalid observing frequency: {}",
                header.observing_frequency
            ),
        ));
    }

    Ok(())
}

pub(crate) fn validate_cor_payload(header: &CorHeader, file_len: usize) -> io::Result<()> {
    validate_header_fields(header)?;

    let visibility_bytes = (header.fft_point as usize)
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "sector size overflow"))?;
    let sector_size = SECTOR_HEADER_SIZE
        .checked_add(visibility_bytes)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "sector size overflow"))?;
    let payload_size = sector_size
        .checked_mul(header.number_of_sector as usize)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "file size overflow"))?;
    let expected_size = FILE_HEADER_SIZE
        .checked_add(payload_size)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "file size overflow"))?;

    if file_len < expected_size {
        return Err(io::Error::new(
            ErrorKind::UnexpectedEof,
            format!(
                "truncated .cor file: {} bytes, expected at least {} bytes",
                file_len, expected_size
            ),
        ));
    }
    Ok(())
}

pub fn parse_header(cursor: &mut Cursor<&[u8]>) -> io::Result<CorHeader> {
    let mut header = CorHeader::default();
    cursor.set_position(0);

    // Line 0
    cursor.read_exact(&mut header.magic_word)?;
    header.header_version = cursor.read_i32::<LittleEndian>()?;
    header.software_version = cursor.read_i32::<LittleEndian>()?;
    header.sampling_speed = cursor.read_i32::<LittleEndian>()?;

    // Line 1
    header.observing_frequency = cursor.read_f64::<LittleEndian>()?;
    header.fft_point = cursor.read_i32::<LittleEndian>()?;
    header.number_of_sector = cursor.read_i32::<LittleEndian>()?;

    // Line 2: Station 1 Name
    let mut name_buf = [0u8; 8];
    cursor.read_exact(&mut name_buf)?;
    header.station1_name = String::from_utf8_lossy(&name_buf)
        .trim_end_matches('\0')
        .to_string();
    cursor.set_position(cursor.position() + 8); // Skip padding

    // Line 3: Station 1 Pos X, Y
    header.station1_position[0] = cursor.read_f64::<LittleEndian>()?;
    header.station1_position[1] = cursor.read_f64::<LittleEndian>()?;

    // Line 4: Station 1 Pos Z, Code
    header.station1_position[2] = cursor.read_f64::<LittleEndian>()?;
    let mut code_buf = [0u8; 1];
    cursor.read_exact(&mut code_buf)?;
    header.station1_code = String::from_utf8_lossy(&code_buf).to_string();
    cursor.set_position(cursor.position() + 7); // Skip padding

    // Line 5: Station 2 Name
    cursor.read_exact(&mut name_buf)?;
    header.station2_name = String::from_utf8_lossy(&name_buf)
        .trim_end_matches('\0')
        .to_string();
    cursor.set_position(cursor.position() + 8); // Skip padding

    // Line 6: Station 2 Pos X, Y
    header.station2_position[0] = cursor.read_f64::<LittleEndian>()?;
    header.station2_position[1] = cursor.read_f64::<LittleEndian>()?;

    // Line 7: Station 2 Pos Z, Code
    header.station2_position[2] = cursor.read_f64::<LittleEndian>()?;
    cursor.read_exact(&mut code_buf)?;
    header.station2_code = String::from_utf8_lossy(&code_buf).to_string();
    cursor.set_position(cursor.position() + 7); // Skip padding

    // Line 8: Source Name (16 bytes)
    let mut source_name_buf = [0u8; 16];
    cursor.read_exact(&mut source_name_buf)?;
    header.source_name = String::from_utf8_lossy(&source_name_buf)
        .trim_end_matches('\0')
        .to_string();

    // Line 9: Source Pos RA, Dec
    header.source_position_ra = cursor.read_f64::<LittleEndian>()?;
    header.source_position_dec = cursor.read_f64::<LittleEndian>()?;

    // Clock parameters based on Python's header2 indices
    // Python header2[20] is at byte 160. This is skipped in Python. (source_position_dec ends at 159)
    cursor.set_position(168); // Jump to the start of station1_clock_delay (Python header2[21])

    header.station1_clock_delay = cursor.read_f64::<LittleEndian>()?;
    header.station1_clock_rate = cursor.read_f64::<LittleEndian>()?;
    header.station1_clock_acel = cursor.read_f64::<LittleEndian>()?;
    header.station1_clock_jerk = cursor.read_f64::<LittleEndian>()?;
    header.station1_clock_snap = cursor.read_f64::<LittleEndian>()?;

    // Python header2[26] is at byte 208. This is skipped in Python.
    cursor.set_position(216); // Jump to the start of station2_clock_delay (Python header2[27])

    header.station2_clock_delay = cursor.read_f64::<LittleEndian>()?;
    header.station2_clock_rate = cursor.read_f64::<LittleEndian>()?;
    header.station2_clock_acel = cursor.read_f64::<LittleEndian>()?;
    header.station2_clock_jerk = cursor.read_f64::<LittleEndian>()?;
    header.station2_clock_snap = cursor.read_f64::<LittleEndian>()?;

    cursor.set_position(FILE_HEADER_SIZE as u64); // Go to the end of the header
    validate_header_fields(&header)?;
    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::{
        validate_cor_payload, validate_header_fields, CorHeader, COR_MAGIC, FILE_HEADER_SIZE,
        SECTOR_HEADER_SIZE,
    };
    use std::io::ErrorKind;

    fn valid_header() -> CorHeader {
        CorHeader {
            magic_word: COR_MAGIC,
            sampling_speed: 1_024_000_000,
            observing_frequency: 6_600_000_000.0,
            fft_point: 8192,
            number_of_sector: 2,
            ..CorHeader::default()
        }
    }

    fn expected_size(header: &CorHeader) -> usize {
        FILE_HEADER_SIZE
            + header.number_of_sector as usize
                * (SECTOR_HEADER_SIZE + header.fft_point as usize * size_of::<f32>())
    }

    #[test]
    fn accepts_a_well_formed_header_and_payload_size() {
        let header = valid_header();
        assert!(validate_header_fields(&header).is_ok());
        assert!(validate_cor_payload(&header, expected_size(&header)).is_ok());
    }

    #[test]
    fn rejects_invalid_dimensions_before_allocation() {
        let mut header = valid_header();
        header.fft_point = -8192;
        assert_eq!(
            validate_header_fields(&header).unwrap_err().kind(),
            ErrorKind::InvalidData
        );

        let mut header = valid_header();
        header.number_of_sector = -1;
        assert_eq!(
            validate_header_fields(&header).unwrap_err().kind(),
            ErrorKind::InvalidData
        );
    }

    #[test]
    fn rejects_wrong_magic_and_truncated_payload() {
        let mut header = valid_header();
        header.magic_word = [0; 4];
        assert_eq!(
            validate_header_fields(&header).unwrap_err().kind(),
            ErrorKind::InvalidData
        );

        let header = valid_header();
        assert_eq!(
            validate_cor_payload(&header, expected_size(&header) - 1)
                .unwrap_err()
                .kind(),
            ErrorKind::UnexpectedEof
        );
    }
}
