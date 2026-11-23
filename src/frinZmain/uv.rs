use crate::args::Args;
use crate::header::{available_cor_sectors, parse_header, CorHeader};
use crate::input_support::open_input_data;
use crate::npy_output::{npz_sidecar_path, NamedNpz, NpyMeta};
use crate::plot::plot_uv_tracks;
use crate::read::{read_sector_metadata, sector_epochs};
use crate::utils::{greenwich_sidereal_angle, seconds_between, UvwProjection};
use chrono::{DateTime, Duration, Timelike, Utc};
use nav_types::{ECEF, WGS84};
use std::error::Error;
use std::fs;
use std::io::Cursor;
use std::path::Path;

struct ElevationLimit {
    longitude_minus_ra: f64,
    offset: f64,
    scale: f64,
}

impl ElevationLimit {
    fn new([x, y, z]: [f64; 3], ra: f64, dec: f64) -> Self {
        let site: WGS84<f64> = ECEF::new(x, y, z).into();
        let (sin_lat, cos_lat) = site.latitude_radians().sin_cos();
        let (sin_dec, cos_dec) = dec.sin_cos();
        Self {
            longitude_minus_ra: site.longitude_radians() - ra,
            offset: sin_lat * sin_dec - 5.0_f64.to_radians().sin(),
            scale: cos_lat * cos_dec,
        }
    }

    fn is_visible(&self, sidereal: f64) -> bool {
        self.offset + self.scale * (sidereal + self.longitude_minus_ra).cos() >= 0.0
    }
}

struct UvPoint {
    u_m: f64,
    v_m: f64,
    utc_hour: f64,
    elapsed_s: f64,
    starts_segment: bool,
}

struct PlotTrack {
    uv: Vec<(f32, f32)>,
    baseline: Vec<(f32, f32)>,
}

fn plot_track(samples: &[UvPoint]) -> PlotTrack {
    let mut uv = Vec::with_capacity(samples.len());
    let mut baseline = Vec::with_capacity(samples.len());
    for sample in samples {
        if sample.starts_segment && !uv.is_empty() {
            uv.push((f32::NAN, f32::NAN));
            baseline.push((f32::NAN, f32::NAN));
        }
        uv.push((sample.u_m as f32, sample.v_m as f32));
        baseline.push((sample.utc_hour as f32, sample.u_m.hypot(sample.v_m) as f32));
    }
    PlotTrack { uv, baseline }
}

fn utc_hour(time: DateTime<Utc>) -> f64 {
    (f64::from(time.num_seconds_from_midnight()) + f64::from(time.nanosecond()) * 1e-9) / 3600.0
}

fn observed_track(
    epochs: impl ExactSizeIterator<Item = std::io::Result<DateTime<Utc>>>,
    first_time: DateTime<Utc>,
    integration_s: f64,
    projection: &UvwProjection,
) -> std::io::Result<Vec<UvPoint>> {
    let mut samples = Vec::with_capacity(epochs.len());
    let mut previous = first_time;
    for time in epochs {
        let time = time?;
        let separation = seconds_between(time, previous);
        let starts_segment = time.date_naive() != previous.date_naive()
            || separation < 0.0
            || separation > 1.5 * integration_s;
        let (u_m, v_m, _, _, _) = projection.project(greenwich_sidereal_angle(time));
        samples.push(UvPoint {
            u_m,
            v_m,
            utc_hour: utc_hour(time),
            elapsed_s: seconds_between(time, first_time),
            starts_segment,
        });
        previous = time;
    }
    Ok(samples)
}

fn accessible_track(
    header: &CorHeader,
    first_time: DateTime<Utc>,
    projection: &UvwProjection,
) -> Vec<UvPoint> {
    let elevation = [header.station1_position, header.station2_position].map(|site| {
        ElevationLimit::new(site, header.source_position_ra, header.source_position_dec)
    });
    let day_start = first_time
        .date_naive()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc();
    let mut samples = Vec::new();
    let mut visible_before = false;
    for minute in (0..=1440).step_by(3) {
        let time = day_start + Duration::minutes(minute);
        let sidereal = greenwich_sidereal_angle(time);
        let visible = elevation.iter().all(|limit| limit.is_visible(sidereal));
        if visible {
            let (u_m, v_m, _, _, _) = projection.project(sidereal);
            samples.push(UvPoint {
                u_m,
                v_m,
                utc_hour: minute as f64 / 60.0,
                elapsed_s: seconds_between(time, first_time),
                starts_segment: !visible_before,
            });
        }
        visible_before = visible;
    }
    samples
}

fn add_track_arrays(npz: &mut NamedNpz, prefix: &str, track: &[UvPoint]) {
    npz.add_f64_series(
        &format!("{prefix}_sample"),
        (0..track.len()).map(|i| i as f64),
    );
    npz.add_f64_series(
        &format!("{prefix}_u_m"),
        track.iter().map(|point| point.u_m),
    );
    npz.add_f64_series(
        &format!("{prefix}_v_m"),
        track.iter().map(|point| point.v_m),
    );
    npz.add_f64_series(
        &format!("{prefix}_utc_hour"),
        track.iter().map(|point| point.utc_hour),
    );
    npz.add_f64_series(
        &format!("{prefix}_elapsed_s"),
        track.iter().map(|point| point.elapsed_s),
    );
}

pub fn run_uv_plot(args: &Args, uv_mode: i32) -> Result<(), Box<dyn Error>> {
    if !matches!(uv_mode, 0 | 1) {
        return Err("--uv accepts only 0 (planar) or 1 (3D)".into());
    }
    let input_path = args.input.as_ref().ok_or("--uv requires --input")?;
    let buffer = open_input_data(input_path)?;
    let bytes = buffer.as_slice();
    let mut cursor = Cursor::new(bytes);
    let header = parse_header(&mut cursor)?;
    let available = available_cor_sectors(&header, bytes.len())?;
    let first = args.skip.max(0);
    if first >= available {
        return Err(
            format!("skip {first} exceeds available complete sectors ({available})").into(),
        );
    }
    let length = if args.length > 0 {
        args.length.min(available - first)
    } else {
        available - first
    };
    let (first_time, integration_s) = read_sector_metadata(&mut cursor, &header, first)?;
    let epochs = sector_epochs(bytes, &header, first as usize, (first + length) as usize)?;
    let projection = UvwProjection::new(
        header.station1_position,
        header.station2_position,
        header.source_position_ra,
        header.source_position_dec,
        uv_mode != 0,
    );
    let observed = observed_track(epochs, first_time, integration_s.into(), &projection)?;
    drop(buffer);
    let accessible = accessible_track(&header, first_time, &projection);
    let observed_plot = plot_track(&observed);
    let accessible_plot = plot_track(&accessible);

    let output_dir = input_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join("frinZ")
        .join("uv");
    fs::create_dir_all(&output_dir)?;
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("invalid input filename")?;
    let output_path = output_dir.join(format!("{stem}_uv.png"));
    let center_frequency_hz = header.observing_frequency + f64::from(header.sampling_speed) / 4.0;
    plot_uv_tracks(
        &output_path,
        &accessible_plot.uv,
        &observed_plot.uv,
        &accessible_plot.baseline,
        &observed_plot.baseline,
        center_frequency_hz,
        uv_mode,
    )?;
    if args.npz {
        let path = npz_sidecar_path(&output_path, "uv");
        let mut npz = NamedNpz::for_output(
            NpyMeta::new("uv", header.fft_point as u32, available as u32),
            &path,
        );
        add_track_arrays(&mut npz, "observed", &observed);
        add_track_arrays(&mut npz, "accessible", &accessible);
        npz.add_f64_1d("start_mjd", &[crate::utils::mjd_cal(first_time)]);
        npz.write(&path)?;
    }
    println!("Generated UV coverage plot at {:?}", output_path);
    Ok(())
}
