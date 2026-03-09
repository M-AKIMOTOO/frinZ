use crate::args::Args;
use crate::header::parse_header;
use crate::plot::plot_uv_tracks;
use crate::read::{read_sector_header, read_visibility_data};
use crate::utils::{radec2azalt, uvw_cal};
use byteorder::{LittleEndian, ReadBytesExt};
use chrono::{Duration, TimeZone, Timelike, Utc};
use std::error::Error;
use std::fs::{self, File};
use std::io::{Cursor, Read};
use std::path::Path;

const MIN_ELEVATION_DEG: f32 = 5.0;

pub fn run_uv_plot(args: &Args, uv_mode: i32) -> Result<(), Box<dyn Error>> {
    if uv_mode != 0 && uv_mode != 1 {
        return Err("--uv accepts only 0 (planar) or 1 (3D)".into());
    }
    let input_path = args
        .input
        .as_ref()
        .ok_or("Error: --uv requires an --input file.")?;

    let mut file = File::open(input_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let mut cursor = Cursor::new(buffer.as_slice());
    let header = parse_header(&mut cursor)?;

    let include_vertical = uv_mode != 0;

    let start_sector = args.skip.max(0);
    if start_sector >= header.number_of_sector {
        return Err(format!(
            "Skip value ({}) exceeds available sectors ({}).",
            args.skip, header.number_of_sector
        )
        .into());
    }

    let available_sectors = header.number_of_sector - start_sector;
    let sectors_to_use = if args.length > 0 {
        args.length.min(available_sectors)
    } else {
        available_sectors
    };

    if sectors_to_use <= 0 {
        return Err("No sectors available after applying --skip/--length.".into());
    }

    let mut data_cursor = Cursor::new(buffer.as_slice());
    let (_, first_time, effective_integ_time) =
        read_visibility_data(&mut data_cursor, &header, 1, 0, start_sector, false, &[])?;

    if effective_integ_time <= 0.0 {
        return Err("Effective integration time is zero; cannot compute UV coverage.".into());
    }

    let mut sector_cursor = Cursor::new(buffer.as_slice());
    let sector_headers = read_sector_header(
        &mut sector_cursor,
        &header,
        sectors_to_use,
        start_sector,
        0,
        false,
    )?;

    let mut observation_times = Vec::with_capacity(sector_headers.len());
    for header_bytes in sector_headers {
        let mut sector_reader = Cursor::new(header_bytes);
        let correlation_time_sec = sector_reader.read_i32::<LittleEndian>()?;
        let obs_time = Utc
            .timestamp_opt(correlation_time_sec as i64, 0)
            .single()
            .ok_or_else(|| format!("Invalid timestamp seconds: {}", correlation_time_sec))?;
        observation_times.push(obs_time);
    }

    if observation_times.is_empty() {
        return Err("Failed to read sector timestamps for UV plotting.".into());
    }

    let mut observed_uv: Vec<(f32, f32)> = Vec::with_capacity(observation_times.len());
    let mut observed_baseline: Vec<(f32, f32)> = Vec::with_capacity(observation_times.len());
    for obs_time in &observation_times {
        let (u, v, _w, _du_dt, _dv_dt) = uvw_cal(
            header.station1_position,
            header.station2_position,
            *obs_time,
            header.source_position_ra,
            header.source_position_dec,
            include_vertical,
        );
        let u_f = u as f32;
        let v_f = v as f32;
        let ut_hour = obs_time.hour() as f32
            + obs_time.minute() as f32 / 60.0
            + obs_time.second() as f32 / 3600.0;
        observed_uv.push((u_f, v_f));
        observed_baseline.push((ut_hour, (u_f * u_f + v_f * v_f).sqrt()));
    }

    let ant1_position = [
        header.station1_position[0] as f32,
        header.station1_position[1] as f32,
        header.station1_position[2] as f32,
    ];
    let ant2_position = [
        header.station2_position[0] as f32,
        header.station2_position[1] as f32,
        header.station2_position[2] as f32,
    ];

    let day_start = first_time
        .date_naive()
        .and_hms_opt(0, 0, 0)
        .map(|naive| Utc.from_utc_datetime(&naive))
        .unwrap_or(first_time);
    let day_end = day_start + Duration::hours(24);

    let mut accessible_uv: Vec<(f32, f32)> = Vec::new();
    let mut accessible_baseline: Vec<(f32, f32)> = Vec::new();
    let mut current_time = day_start;
    let step = Duration::minutes(3);
    while current_time <= day_end {
        let (_, el1, _) = radec2azalt(
            ant1_position,
            current_time,
            header.source_position_ra as f32,
            header.source_position_dec as f32,
        );
        let (_, el2, _) = radec2azalt(
            ant2_position,
            current_time,
            header.source_position_ra as f32,
            header.source_position_dec as f32,
        );

        if el1 >= MIN_ELEVATION_DEG && el2 >= MIN_ELEVATION_DEG {
            let (u, v, _w, _du_dt, _dv_dt) = uvw_cal(
                header.station1_position,
                header.station2_position,
                current_time,
                header.source_position_ra,
                header.source_position_dec,
                include_vertical,
            );
            let u_f = u as f32;
            let v_f = v as f32;
            let ut_hour = current_time.hour() as f32
                + current_time.minute() as f32 / 60.0
                + current_time.second() as f32 / 3600.0;
            accessible_uv.push((u_f, v_f));
            accessible_baseline.push((ut_hour, (u_f * u_f + v_f * v_f).sqrt()));
        }

        current_time += step;
    }

    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let output_dir = parent_dir.join("frinZ").join("uv");
    fs::create_dir_all(&output_dir)?;

    let base_stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("Invalid input filename")?;
    let output_path = output_dir.join(format!("{}_uv.png", base_stem));

    let bandwidth_hz = header.sampling_speed as f64 / 2.0;
    let center_frequency_hz = header.observing_frequency + bandwidth_hz / 2.0;
    if center_frequency_hz <= 0.0 {
        return Err("Center frequency is non-positive; cannot scale spatial frequency.".into());
    }

    plot_uv_tracks(
        &output_path,
        &accessible_uv,
        &observed_uv,
        &accessible_baseline,
        &observed_baseline,
        center_frequency_hz,
        uv_mode,
    )?;

    println!("Generated UV coverage plot at {:?}", output_path);

    Ok(())
}
