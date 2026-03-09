use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{self, Cursor, Read};

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

    cursor.set_position(256); // Go to the end of the header
    Ok(header)
}
