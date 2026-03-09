use std::io;
use std::process;

pub fn parse_rfi_ranges(rfi_args: &[String], rbw: f32) -> io::Result<Vec<(usize, usize)>> {
    if rfi_args.is_empty() {
        return Ok(vec![]);
    }
    let mut ranges = Vec::new();
    for rfi_pair in rfi_args {
        let parts: Vec<&str> = rfi_pair.split(',').collect();
        if parts.len() != 2 {
            eprintln!(
                "Invalid RFI format: {}. Expected format is MIN,MAX.",
                rfi_pair
            );
            process::exit(1);
        }

        let min_mhz_int: i32 = parts[0].parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Invalid integer for RFI min: {}", parts[0]),
            )
        })?;
        let max_mhz_int: i32 = parts[1].parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Invalid integer for RFI max: {}", parts[1]),
            )
        })?;

        if min_mhz_int >= max_mhz_int {
            eprintln!(
                "Invalid RFI range: min ({}) >= max ({}).",
                min_mhz_int, max_mhz_int
            );
            process::exit(1);
        }
        let min_chan = (min_mhz_int as f32 / rbw).floor() as usize;
        let max_chan = (max_mhz_int as f32 / rbw).ceil() as usize;
        ranges.push((min_chan, max_chan));
    }
    Ok(ranges)
}
