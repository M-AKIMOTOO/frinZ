use crate::args::Args;
use crate::header::parse_header;
use crate::read::read_visibility_data;
use crate::utils::radec2azalt;
use chrono::{DateTime, Duration, TimeZone, Timelike, Utc};
use plotters::coord::Shift;
use plotters::prelude::*;
use std::error::Error;
use std::fs::{self, File};
use crate::png_compress::{compress_png_with_mode, CompressQuality};
use std::io::{Cursor, Read};
use std::path::Path;

const CANVAS_SIZE: (u32, u32) = (1400, 900);

pub fn run_uptime_plot(args: &Args) -> Result<(), Box<dyn Error>> {
    let input_path = match &args.input {
        Some(path) => path,
        None => {
            return Err("Error: --uptimeplot requires an --input file.".into());
        }
    };

    let mut file = File::open(input_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let mut cursor = Cursor::new(buffer.as_slice());
    let header = parse_header(&mut cursor)?;

    let mut data_cursor = Cursor::new(buffer.as_slice());
    let (_, start_time, effective_integ_time) =
        read_visibility_data(&mut data_cursor, &header, 1, args.skip, 0, false, &[])?;

    if effective_integ_time <= 0.0 {
        return Err("Effective integration time is zero; cannot generate uptime plot.".into());
    }

    let total_sectors_available = header.number_of_sector.saturating_sub(args.skip);
    let sectors_to_use = if args.length > 0 {
        args.length.min(total_sectors_available)
    } else {
        total_sectors_available
    };

    if sectors_to_use <= 0 {
        return Err("No sectors available for uptime plot after applying skip/length.".into());
    }

    let duration_seconds = (sectors_to_use as f32 * effective_integ_time)
        .round()
        .max(1.0) as usize;

    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let output_dir = parent_dir.join("frinZ").join("uptime");
    fs::create_dir_all(&output_dir)?;
    let base_stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("Invalid input filename")?;

    let observation_points_station1 = generate_time_series(
        start_time,
        duration_seconds,
        header.station1_position,
        header.source_position_ra,
        header.source_position_dec,
    );

    let observation_points_station2 = generate_time_series(
        start_time,
        duration_seconds,
        header.station2_position,
        header.source_position_ra,
        header.source_position_dec,
    );

    let day_points_station1 = generate_full_day_series(
        start_time,
        header.station1_position,
        header.source_position_ra,
        header.source_position_dec,
    );

    let day_points_station2 = generate_full_day_series(
        start_time,
        header.station2_position,
        header.source_position_ra,
        header.source_position_dec,
    );

    let plot_path = output_dir.join(format!("{}_uptime.png", base_stem));
    draw_uptime_plot(
        &plot_path,
        &observation_points_station1,
        &observation_points_station2,
        &day_points_station1,
        &day_points_station2,
        header.station1_name.trim(),
        header.station2_name.trim(),
    )?;

    println!("Generated uptime plot at {:?}", plot_path);

    Ok(())
}

struct UptimeSeries {
    az: Vec<(f64, f64)>,
    el: Vec<(f64, f64)>,
}

fn generate_time_series(
    start_time: DateTime<Utc>,
    duration_seconds: usize,
    station_position: [f64; 3],
    source_ra: f64,
    source_dec: f64,
) -> UptimeSeries {
    compute_series(
        start_time,
        duration_seconds,
        station_position,
        source_ra,
        source_dec,
    )
}

fn generate_full_day_series(
    reference_time: DateTime<Utc>,
    station_position: [f64; 3],
    source_ra: f64,
    source_dec: f64,
) -> UptimeSeries {
    let day_start = reference_time
        .date_naive()
        .and_hms_opt(0, 0, 0)
        .map(|naive| Utc.from_utc_datetime(&naive))
        .unwrap_or(reference_time);

    let mut azimuth = Vec::with_capacity(24 * 60 / 3 + 1);
    let mut elevation = Vec::with_capacity(24 * 60 / 3 + 1);

    for minute in (0..=(24 * 60)).step_by(3) {
        let current_time = day_start + Duration::seconds((minute * 60) as i64);
        let ut_hour = current_time.hour() as f64
            + current_time.minute() as f64 / 60.0
            + current_time.second() as f64 / 3600.0;

        let (az_deg, el_deg, _) = radec2azalt(
            [
                station_position[0] as f32,
                station_position[1] as f32,
                station_position[2] as f32,
            ],
            current_time,
            source_ra as f32,
            source_dec as f32,
        );

        azimuth.push((ut_hour, wrap_azimuth(az_deg as f64)));
        elevation.push((ut_hour, el_deg as f64));
    }

    UptimeSeries {
        az: azimuth,
        el: elevation,
    }
}

fn compute_series(
    start_time: DateTime<Utc>,
    duration_seconds: usize,
    station_position: [f64; 3],
    source_ra: f64,
    source_dec: f64,
) -> UptimeSeries {
    let mut azimuth = Vec::with_capacity(duration_seconds + 1);
    let mut elevation = Vec::with_capacity(duration_seconds + 1);

    for idx in 0..=duration_seconds {
        let current_time = start_time + Duration::seconds(idx as i64);
        let ut_hour = current_time.hour() as f64
            + current_time.minute() as f64 / 60.0
            + current_time.second() as f64 / 3600.0;

        let (az_deg, el_deg, _) = radec2azalt(
            [
                station_position[0] as f32,
                station_position[1] as f32,
                station_position[2] as f32,
            ],
            current_time,
            source_ra as f32,
            source_dec as f32,
        );

        azimuth.push((ut_hour, wrap_azimuth(az_deg as f64)));
        elevation.push((ut_hour, el_deg as f64));
    }

    UptimeSeries {
        az: azimuth,
        el: elevation,
    }
}

fn wrap_azimuth(mut value: f64) -> f64 {
    while value < 0.0 {
        value += 360.0;
    }
    while value >= 360.0 {
        value -= 360.0;
    }
    value
}

fn draw_uptime_plot(
    path: &Path,
    observation_station1: &UptimeSeries,
    observation_station2: &UptimeSeries,
    full_day_station1: &UptimeSeries,
    full_day_station2: &UptimeSeries,
    station1_label: &str,
    station2_label: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, CANVAS_SIZE).into_drawing_area();
    root.fill(&WHITE)?;
    let areas = root.split_evenly((2, 1));

    let x_range = 0.0f64..24.0f64;
    let y_range_az = 0.0f64..360.0f64;
    let y_range_el = 0.0f64..90.0f64;

    draw_single_chart(
        &areas[0],
        &observation_station1.az,
        &observation_station2.az,
        &full_day_station1.az,
        &full_day_station2.az,
        x_range.clone(),
        y_range_az,
        "AZ [deg]",
        station1_label,
        station2_label,
    )?;

    draw_single_chart(
        &areas[1],
        &observation_station1.el,
        &observation_station2.el,
        &full_day_station1.el,
        &full_day_station2.el,
        x_range,
        y_range_el,
        "EL [deg]",
        station1_label,
        station2_label,
    )?;

    root.present()?;
    compress_png_with_mode(path, CompressQuality::Low);
    Ok(())
}

fn draw_single_chart(
    area: &DrawingArea<BitMapBackend, Shift>,
    observation_station1: &[(f64, f64)],
    observation_station2: &[(f64, f64)],
    full_day_station1: &[(f64, f64)],
    full_day_station2: &[(f64, f64)],
    x_range: std::ops::Range<f64>,
    y_range: std::ops::Range<f64>,
    y_label: &str,
    station1_label: &str,
    station2_label: &str,
) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(x_range.clone(), y_range.clone())?;

    let y_step = if y_range.end <= 100.0 { 10.0 } else { 30.0 };
    let y_labels = ((y_range.end - y_range.start) / y_step).round() as usize + 1;

    chart
        .configure_mesh()
        .x_desc("UT [hours]")
        .y_desc(y_label)
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.0}", v))
        .x_labels(25)
        .y_labels(y_labels)
        .light_line_style(&WHITE.mix(0.0))
        .bold_line_style(&BLACK.mix(0.2))
        .draw()?;

    if !full_day_station1.is_empty() {
        let point_style = Into::<ShapeStyle>::into(BLUE.mix(0.35)).stroke_width(1);
        chart.draw_series(PointSeries::of_element(
            full_day_station1.iter().copied(),
            4,
            point_style,
            &|coord, size, style| EmptyElement::at(coord) + Cross::new((0, 0), size, style),
        ))?;
    }

    if !observation_station1.is_empty() {
        let point_style = Into::<ShapeStyle>::into(BLUE).stroke_width(2);
        chart
            .draw_series(PointSeries::of_element(
                observation_station1.iter().copied(),
                4,
                point_style,
                &|coord, size, style| EmptyElement::at(coord) + Cross::new((0, 0), size, style),
            ))?
            .label(station1_label.to_string())
            .legend(|(x, y)| {
                Cross::new(
                    (x + 10, y),
                    6,
                    Into::<ShapeStyle>::into(BLUE).stroke_width(2),
                )
            });
    }

    if !full_day_station2.is_empty() {
        let point_style = Into::<ShapeStyle>::into(BLUE.mix(0.35)).stroke_width(1);
        chart.draw_series(PointSeries::of_element(
            full_day_station2.iter().copied(),
            4,
            point_style,
            &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
        ))?;
    }

    if !observation_station2.is_empty() {
        let point_style = Into::<ShapeStyle>::into(BLUE).stroke_width(2);
        chart
            .draw_series(PointSeries::of_element(
                observation_station2.iter().copied(),
                4,
                point_style,
                &|coord, size, style| {
                    EmptyElement::at(coord) + Circle::new((0, 0), size, style.filled())
                },
            ))?
            .label(station2_label.to_string())
            .legend(|(x, y)| Circle::new((x + 10, y), 4, Into::<ShapeStyle>::into(BLUE).filled()));
    }

    chart
        .configure_series_labels()
        .label_font(("sans-serif", 20))
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.7))
        .draw()?;

    Ok(())
}
