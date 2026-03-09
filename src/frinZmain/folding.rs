use astro::ecliptic;
use astro::planet::{self, Planet};
use astro::time;
use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use memmap2::Mmap;
use nalgebra::{Matrix3, Vector3};
use num_complex::Complex;
use plotters::prelude::*;
use std::error::Error;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Cursor, Write};
use std::path::{Path, PathBuf};

use crate::args::Args;
use crate::bandpass::read_bandpass_file;
use crate::header::parse_header;
use crate::png_compress::{compress_png_with_mode, CompressQuality};
use crate::read::read_visibility_data;
use crate::rfi::parse_rfi_ranges;
use crate::utils::parse_flag_time;

type C32 = Complex<f32>;

const BP_EPSILON: f32 = 1e-9;
const C_KM_S: f64 = 299_792.458;
const AU_KM: f64 = 149_597_870.7;

#[derive(Debug, Clone)]
struct OrbitalEphemeris {
    pb_sec: f64,
    a1_sec: f64,
    ecc: f64,
    omega_rad: f64,
    tasc_unix_sec: Option<f64>,
    tp_unix_sec: Option<f64>,
}

#[derive(Debug, Clone)]
struct SpinEphemeris {
    period_sec: f64,
    period_dot_sec_per_sec: f64,
    epoch_unix_sec: Option<f64>,
    use_barycentric: bool,
    use_orbital: bool,
    orbital: Option<OrbitalEphemeris>,
    source_ra_override_rad: Option<f64>,
    source_dec_override_rad: Option<f64>,
    ephem_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct FoldingConfig {
    bins: usize,
    on_duty: f64,
    phase0: f64,
    spin: SpinEphemeris,
}

#[derive(Debug, Clone)]
struct FoldBin {
    phase_center: f64,
    mean_complex: Complex<f64>,
    amp: f64,
    phase_deg: f64,
    count: usize,
    on_bin: bool,
}

#[derive(Debug, Clone)]
struct FoldingParseState {
    period_sec: Option<f64>,
    period_dot_sec_per_sec: f64,
    epoch_unix_sec: Option<f64>,
    bins: usize,
    on_duty: f64,
    phase0: f64,
    use_barycentric: bool,
    use_orbital: bool,
    source_ra_override_rad: Option<f64>,
    source_dec_override_rad: Option<f64>,
    orbital_pb_sec: Option<f64>,
    orbital_a1_sec: Option<f64>,
    orbital_ecc: Option<f64>,
    orbital_omega_rad: Option<f64>,
    orbital_tasc_unix_sec: Option<f64>,
    orbital_tp_unix_sec: Option<f64>,
    ephem_path: Option<PathBuf>,
}

impl Default for FoldingParseState {
    fn default() -> Self {
        Self {
            period_sec: None,
            period_dot_sec_per_sec: 0.0,
            epoch_unix_sec: None,
            bins: 64,
            on_duty: 0.1,
            phase0: 0.0,
            use_barycentric: false,
            use_orbital: false,
            source_ra_override_rad: None,
            source_dec_override_rad: None,
            orbital_pb_sec: None,
            orbital_a1_sec: None,
            orbital_ecc: None,
            orbital_omega_rad: None,
            orbital_tasc_unix_sec: None,
            orbital_tp_unix_sec: None,
            ephem_path: None,
        }
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_datetime_to_unix_sec(value: &str) -> Option<f64> {
    let s = value.trim();
    if s.is_empty() {
        return None;
    }

    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        let utc = dt.with_timezone(&Utc);
        return Some(utc.timestamp() as f64 + utc.timestamp_subsec_nanos() as f64 * 1e-9);
    }

    let fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y/%m/%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S%.f",
    ];
    for fmt in fmts {
        if let Ok(ndt) = NaiveDateTime::parse_from_str(s, fmt) {
            let dt = Utc.from_utc_datetime(&ndt);
            return Some(dt.timestamp() as f64 + dt.timestamp_subsec_nanos() as f64 * 1e-9);
        }
    }

    if let Some(dt) = parse_flag_time(s) {
        return Some(dt.timestamp() as f64 + dt.timestamp_subsec_nanos() as f64 * 1e-9);
    }

    None
}

fn mjd_to_unix_sec(mjd: f64) -> f64 {
    (mjd - 40587.0) * 86400.0
}

fn parse_f64(value: &str, key: &str) -> Result<f64, Box<dyn Error>> {
    value
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("Error: invalid value '{}' for {}.", value, key).into())
}

fn parse_datetime_or_mjd(
    value: &str,
    allow_mjd: bool,
    key: &str,
) -> Result<f64, Box<dyn Error>> {
    if let Some(unix) = parse_datetime_to_unix_sec(value) {
        return Ok(unix);
    }
    if allow_mjd {
        let mjd = parse_f64(value, key)?;
        if !mjd.is_finite() {
            return Err(format!("Error: non-finite MJD for {}.", key).into());
        }
        return Ok(mjd_to_unix_sec(mjd));
    }
    Err(format!(
        "Error: invalid datetime for {} (supported: RFC3339, YYYY-MM-DD HH:MM:SS, YYYYDDDHHMMSS).",
        key
    )
    .into())
}

fn apply_key_value(
    state: &mut FoldingParseState,
    key: &str,
    value: &str,
) -> Result<(), Box<dyn Error>> {
    let key_norm = key.trim().to_ascii_lowercase().replace('-', "_");
    let value = value.trim();
    match key_norm.as_str() {
        "period" | "period_sec" | "p" | "p0" => {
            let v = parse_f64(value, key)?;
            if !v.is_finite() || v <= 0.0 {
                return Err(format!("Error: {} must be > 0.", key).into());
            }
            state.period_sec = Some(v);
        }
        "pdot" | "p1" | "period_dot" | "period_dot_sec_per_sec" => {
            let v = parse_f64(value, key)?;
            if !v.is_finite() {
                return Err(format!("Error: {} must be finite.", key).into());
            }
            state.period_dot_sec_per_sec = v;
        }
        "epoch" | "epoch_utc" | "pepoch" | "pepoch_utc" | "tref" => {
            state.epoch_unix_sec = Some(parse_datetime_or_mjd(value, false, key)?);
        }
        "epoch_mjd" | "pepoch_mjd" | "tref_mjd" => {
            state.epoch_unix_sec = Some(parse_datetime_or_mjd(value, true, key)?);
        }
        "bins" | "bin" => {
            let v = value
                .parse::<usize>()
                .map_err(|_| format!("Error: invalid {} '{}'.", key, value))?;
            if v < 2 {
                return Err("Error: folding bins must be >= 2.".into());
            }
            state.bins = v;
        }
        "on_duty" | "onduty" | "duty" => {
            let v = parse_f64(value, key)?;
            if !v.is_finite() || v <= 0.0 || v >= 1.0 {
                return Err("Error: on-duty must satisfy 0 < on-duty < 1.".into());
            }
            state.on_duty = v;
        }
        "phase0" => {
            let v = parse_f64(value, key)?;
            if !v.is_finite() {
                return Err("Error: phase0 must be finite.".into());
            }
            state.phase0 = v;
        }
        "bary" | "barycentric" => {
            let v = parse_bool(value)
                .ok_or_else(|| format!("Error: {} expects 0/1 or true/false.", key))?;
            state.use_barycentric = v;
        }
        "orbit" | "orbital" => {
            let v = parse_bool(value)
                .ok_or_else(|| format!("Error: {} expects 0/1 or true/false.", key))?;
            state.use_orbital = v;
        }
        "pb" | "pb_sec" | "orbital_period" | "orbital_period_sec" => {
            let v = parse_f64(value, key)?;
            if !v.is_finite() || v <= 0.0 {
                return Err(format!("Error: {} must be > 0.", key).into());
            }
            state.orbital_pb_sec = Some(v);
            state.use_orbital = true;
        }
        "a1" | "a1_sec" | "x" | "asini" | "asini_sec" => {
            let v = parse_f64(value, key)?;
            if !v.is_finite() || v < 0.0 {
                return Err(format!("Error: {} must be >= 0.", key).into());
            }
            state.orbital_a1_sec = Some(v);
            state.use_orbital = true;
        }
        "ecc" | "e" => {
            let v = parse_f64(value, key)?;
            if !v.is_finite() || !(0.0..1.0).contains(&v) {
                return Err(format!("Error: {} must satisfy 0 <= ecc < 1.", key).into());
            }
            state.orbital_ecc = Some(v);
            state.use_orbital = true;
        }
        "omega" | "omega_deg" | "om" => {
            let v = parse_f64(value, key)?;
            if !v.is_finite() {
                return Err(format!("Error: {} must be finite.", key).into());
            }
            state.orbital_omega_rad = Some(v.to_radians());
            state.use_orbital = true;
        }
        "tasc" | "tasc_utc" => {
            state.orbital_tasc_unix_sec = Some(parse_datetime_or_mjd(value, false, key)?);
            state.use_orbital = true;
        }
        "tasc_mjd" => {
            state.orbital_tasc_unix_sec = Some(parse_datetime_or_mjd(value, true, key)?);
            state.use_orbital = true;
        }
        "tp" | "tp_utc" | "t0" | "periastron" | "periastron_utc" => {
            state.orbital_tp_unix_sec = Some(parse_datetime_or_mjd(value, false, key)?);
            state.use_orbital = true;
        }
        "tp_mjd" | "t0_mjd" => {
            state.orbital_tp_unix_sec = Some(parse_datetime_or_mjd(value, true, key)?);
            state.use_orbital = true;
        }
        "ra_deg" => {
            let v = parse_f64(value, key)?;
            if !v.is_finite() {
                return Err(format!("Error: {} must be finite.", key).into());
            }
            state.source_ra_override_rad = Some(v.to_radians());
        }
        "dec_deg" => {
            let v = parse_f64(value, key)?;
            if !v.is_finite() {
                return Err(format!("Error: {} must be finite.", key).into());
            }
            state.source_dec_override_rad = Some(v.to_radians());
        }
        "ra_rad" => {
            let v = parse_f64(value, key)?;
            if !v.is_finite() {
                return Err(format!("Error: {} must be finite.", key).into());
            }
            state.source_ra_override_rad = Some(v);
        }
        "dec_rad" => {
            let v = parse_f64(value, key)?;
            if !v.is_finite() {
                return Err(format!("Error: {} must be finite.", key).into());
            }
            state.source_dec_override_rad = Some(v);
        }
        _ => {
            return Err(format!(
                "Error: unknown --folding key '{}'.",
                key
            )
            .into());
        }
    }
    Ok(())
}

fn parse_ephemeris_file(path: &Path, state: &mut FoldingParseState) -> Result<(), Box<dyn Error>> {
    let file = File::open(path).map_err(|e| format!("Error opening ephemeris file {:?}: {}", path, e))?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line?;
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }

        if let Some((k, v)) = line.split_once('=') {
            apply_key_value(state, k.trim(), v.trim())?;
        } else {
            let mut parts = line.split_whitespace();
            let key = match parts.next() {
                Some(k) => k,
                None => continue,
            };
            let value = parts.collect::<Vec<_>>().join(" ");
            if value.is_empty() {
                continue;
            }
            apply_key_value(state, key, value.trim())?;
        }
    }
    Ok(())
}

fn parse_folding_args(tokens: &[String]) -> Result<FoldingConfig, Box<dyn Error>> {
    if tokens.is_empty() {
        return Err("Error: --folding requires at least period:<sec> or eph:<file>.".into());
    }

    let mut state = FoldingParseState::default();

    for token in tokens {
        if let Some((key, value)) = token.split_once(':') {
            let key_norm = key.trim().to_ascii_lowercase();
            if matches!(key_norm.as_str(), "eph" | "ephem" | "ephemeris" | "par") {
                let path = PathBuf::from(value.trim());
                parse_ephemeris_file(&path, &mut state)?;
                state.ephem_path = Some(path);
            } else {
                apply_key_value(&mut state, key, value)?;
            }
        } else {
            // convenience: bare numeric token as period
            let v = token
                .parse::<f64>()
                .map_err(|_| format!("Error: invalid folding token '{}'.", token))?;
            if !v.is_finite() || v <= 0.0 {
                return Err(format!("Error: invalid folding period '{}'.", token).into());
            }
            state.period_sec = Some(v);
        }
    }

    let period_sec = state
        .period_sec
        .ok_or("Error: period is required for folding (period:<sec> or P0 in ephemeris).")?;

    let mut orbital: Option<OrbitalEphemeris> = None;
    if state.use_orbital
        || state.orbital_pb_sec.is_some()
        || state.orbital_a1_sec.is_some()
        || state.orbital_tasc_unix_sec.is_some()
        || state.orbital_tp_unix_sec.is_some()
    {
        let pb_sec = state
            .orbital_pb_sec
            .ok_or("Error: orbital folding requires pb_sec/PB.")?;
        let a1_sec = state
            .orbital_a1_sec
            .ok_or("Error: orbital folding requires a1_sec/A1.")?;
        if state.orbital_tasc_unix_sec.is_none() && state.orbital_tp_unix_sec.is_none() {
            return Err("Error: orbital folding requires either TASC or TP/T0.".into());
        }
        let ecc = state.orbital_ecc.unwrap_or(0.0);
        if !(0.0..1.0).contains(&ecc) {
            return Err("Error: ecc must satisfy 0 <= ecc < 1.".into());
        }
        let omega_rad = state.orbital_omega_rad.unwrap_or(0.0);
        orbital = Some(OrbitalEphemeris {
            pb_sec,
            a1_sec,
            ecc,
            omega_rad,
            tasc_unix_sec: state.orbital_tasc_unix_sec,
            tp_unix_sec: state.orbital_tp_unix_sec,
        });
        state.use_orbital = true;
        if !state.use_barycentric {
            // Orbital ephemerides are usually defined on barycentric times.
            state.use_barycentric = true;
        }
    }

    Ok(FoldingConfig {
        bins: state.bins,
        on_duty: state.on_duty,
        phase0: state.phase0,
        spin: SpinEphemeris {
            period_sec,
            period_dot_sec_per_sec: state.period_dot_sec_per_sec,
            epoch_unix_sec: state.epoch_unix_sec,
            use_barycentric: state.use_barycentric,
            use_orbital: state.use_orbital,
            orbital,
            source_ra_override_rad: state.source_ra_override_rad,
            source_dec_override_rad: state.source_dec_override_rad,
            ephem_path: state.ephem_path,
        },
    })
}

fn apply_row_phase_correction(
    row: &mut [C32],
    rate_hz: f32,
    delay_samples: f32,
    acel_hz: f32,
    effective_integ_time: f32,
    sampling_speed_hz: i32,
    fft_point: i32,
    sector_index: usize,
) {
    if row.is_empty()
        || (rate_hz == 0.0 && delay_samples == 0.0 && acel_hz == 0.0)
        || sampling_speed_hz <= 0
        || fft_point < 2
        || effective_integ_time <= 0.0
    {
        return;
    }

    // Keep phase convention consistent with fft::apply_phase_correction.
    let t_sec = sector_index as f64 * effective_integ_time as f64;
    let rate_phase = -2.0 * PI * rate_hz as f64 * t_sec;
    let acel_phase = -PI * acel_hz as f64 * t_sec * t_sec;
    let rate_factor =
        Complex::<f64>::new(0.0, rate_phase).exp() * Complex::<f64>::new(0.0, acel_phase).exp();

    let delay_sec = delay_samples as f64 / sampling_speed_hz as f64;
    let freq_res_hz = sampling_speed_hz as f64 / fft_point as f64;

    for (chan_idx, sample) in row.iter_mut().enumerate() {
        let f_hz = chan_idx as f64 * freq_res_hz;
        let delay_phase = -2.0 * PI * delay_sec * f_hz;
        let delay_factor = Complex::<f64>::new(0.0, delay_phase).exp();
        let corrected =
            Complex::<f64>::new(sample.re as f64, sample.im as f64) * rate_factor * delay_factor;
        *sample = C32::new(corrected.re as f32, corrected.im as f32);
    }
}

fn rfi_mask(channel_count: usize, rfi_ranges: &[(usize, usize)]) -> Vec<bool> {
    let mut mask = vec![false; channel_count];
    for &(start, end) in rfi_ranges {
        let mut idx = start.min(channel_count.saturating_sub(1));
        let end_idx = end.min(channel_count.saturating_sub(1));
        while idx <= end_idx {
            mask[idx] = true;
            idx += 1;
        }
    }
    mask
}

fn pick_on_bins(profile: &[FoldBin], on_duty: f64) -> Vec<bool> {
    let bins = profile.len();
    let mut on_mask = vec![false; bins];
    if bins == 0 {
        return on_mask;
    }

    let mut idx: Vec<usize> = (0..bins).collect();
    idx.sort_by(|a, b| {
        profile[*b]
            .amp
            .partial_cmp(&profile[*a].amp)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let on_count = ((bins as f64 * on_duty).ceil() as usize).clamp(1, bins);
    for &i in idx.iter().take(on_count) {
        on_mask[i] = true;
    }
    on_mask
}

fn plot_folding_profile(
    output_path: &Path,
    profile: &[FoldBin],
    peak_idx: usize,
    off_mean: f64,
    off_std: f64,
    period_sec: f64,
) -> Result<(), Box<dyn Error>> {
    if profile.is_empty() {
        return Ok(());
    }

    let points: Vec<(f64, f64)> = profile.iter().map(|b| (b.phase_center, b.amp)).collect();
    let mut ymin = points
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::INFINITY, f64::min)
        .min(off_mean - off_std);
    let mut ymax = points
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max)
        .max(off_mean + off_std);
    if !ymin.is_finite() || !ymax.is_finite() {
        ymin = 0.0;
        ymax = 1.0;
    }
    if (ymax - ymin).abs() < 1e-15 {
        let margin = (ymax.abs() * 0.1).max(1e-12);
        ymin -= margin;
        ymax += margin;
    } else {
        let margin = (ymax - ymin) * 0.1;
        ymin -= margin;
        ymax += margin;
    }

    let root = BitMapBackend::new(output_path, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .caption(
            format!("Folded Profile (P={:.9} s)", period_sec),
            ("sans-serif", 30).into_font(),
        )
        .build_cartesian_2d(0.0f64..1.0f64, ymin..ymax)?;

    chart
        .configure_mesh()
        .x_desc("Phase")
        .y_desc("Amplitude")
        .x_labels(11)
        .draw()?;

    chart
        .draw_series(LineSeries::new(points.iter().copied(), &BLUE))?
        .label("Folded amplitude")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 24, y)], BLUE));

    let on_points = profile
        .iter()
        .filter(|b| b.on_bin)
        .map(|b| Circle::new((b.phase_center, b.amp), 4, RED.filled()));
    chart
        .draw_series(on_points)?
        .label("On-pulse bins")
        .legend(|(x, y)| Circle::new((x + 12, y), 4, RED.filled()));

    chart
        .draw_series(std::iter::once(Circle::new(
            (profile[peak_idx].phase_center, profile[peak_idx].amp),
            5,
            GREEN.filled(),
        )))?
        .label("Peak")
        .legend(|(x, y)| Circle::new((x + 12, y), 5, GREEN.filled()));

    chart
        .draw_series(std::iter::once(PathElement::new(
            [(0.0, off_mean), (1.0, off_mean)],
            BLACK,
        )))?
        .label("Off mean")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 24, y)], BLACK));

    chart.draw_series(std::iter::once(PathElement::new(
        [(0.0, off_mean + off_std), (1.0, off_mean + off_std)],
        BLACK.mix(0.4),
    )))?;
    chart.draw_series(std::iter::once(PathElement::new(
        [(0.0, off_mean - off_std), (1.0, off_mean - off_std)],
        BLACK.mix(0.4),
    )))?;

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    compress_png_with_mode(output_path, CompressQuality::Low);
    Ok(())
}

fn unix_to_julian_day(unix_sec: f64) -> f64 {
    unix_sec / 86400.0 + 2_440_587.5
}

fn source_unit_vector(ra_rad: f64, dec_rad: f64) -> Vector3<f64> {
    Vector3::new(
        dec_rad.cos() * ra_rad.cos(),
        dec_rad.cos() * ra_rad.sin(),
        dec_rad.sin(),
    )
}

fn earth_heliocentric_equatorial_km(jd: f64) -> Vector3<f64> {
    let (lon, lat, radius_au) = planet::heliocent_coords(&Planet::Earth, jd);
    let r_km = radius_au * AU_KM;
    let cos_lat = lat.cos();
    let pos_ecl = Vector3::new(
        r_km * cos_lat * lon.cos(),
        r_km * cos_lat * lon.sin(),
        r_km * lat.sin(),
    );

    let eps = ecliptic::mn_oblq_IAU(jd);
    let rot_ecl_to_eq = Matrix3::new(
        1.0,
        0.0,
        0.0,
        0.0,
        eps.cos(),
        -eps.sin(),
        0.0,
        eps.sin(),
        eps.cos(),
    );
    rot_ecl_to_eq * pos_ecl
}

fn ecef_to_equatorial_km(ecef_km: &Vector3<f64>, jd: f64) -> Vector3<f64> {
    let gmst = time::mn_sidr(jd);
    let rot = Matrix3::new(
        gmst.cos(),
        -gmst.sin(),
        0.0,
        gmst.sin(),
        gmst.cos(),
        0.0,
        0.0,
        0.0,
        1.0,
    );
    rot * ecef_km
}

fn barycentric_delay_sec(obs_unix_sec: f64, observer_ecef_m: [f64; 3], src_hat: &Vector3<f64>) -> f64 {
    let jd = unix_to_julian_day(obs_unix_sec);
    let earth_helio_eq = earth_heliocentric_equatorial_km(jd);
    let observer_ecef_km = Vector3::new(
        observer_ecef_m[0] / 1000.0,
        observer_ecef_m[1] / 1000.0,
        observer_ecef_m[2] / 1000.0,
    );
    let observer_eq_km = ecef_to_equatorial_km(&observer_ecef_km, jd);
    let r_obs = earth_helio_eq + observer_eq_km;
    r_obs.dot(src_hat) / C_KM_S
}

fn solve_kepler_e(mean_anomaly: f64, ecc: f64) -> f64 {
    if ecc < 1e-12 {
        return mean_anomaly;
    }
    let mut e_anom = if ecc < 0.8 { mean_anomaly } else { PI };
    for _ in 0..16 {
        let f = e_anom - ecc * e_anom.sin() - mean_anomaly;
        let fp = 1.0 - ecc * e_anom.cos();
        let delta = f / fp;
        e_anom -= delta;
        if delta.abs() < 1e-13 {
            break;
        }
    }
    e_anom
}

fn orbital_delay_sec(obs_unix_sec: f64, orbital: &OrbitalEphemeris) -> f64 {
    if let Some(tasc) = orbital.tasc_unix_sec {
        let l = 2.0 * PI * ((obs_unix_sec - tasc) / orbital.pb_sec);
        return orbital.a1_sec * l.sin();
    }

    if let Some(tp) = orbital.tp_unix_sec {
        let m = 2.0 * PI * ((obs_unix_sec - tp) / orbital.pb_sec).rem_euclid(1.0);
        let e = solve_kepler_e(m, orbital.ecc);
        let sin_w = orbital.omega_rad.sin();
        let cos_w = orbital.omega_rad.cos();
        let fac = (1.0 - orbital.ecc * orbital.ecc).sqrt();
        return orbital.a1_sec
            * (sin_w * (e.cos() - orbital.ecc) + fac * cos_w * e.sin());
    }

    0.0
}

fn spin_phase_cycles(obs_unix_sec: f64, spin: &SpinEphemeris, epoch_unix_sec: f64) -> f64 {
    let dt = obs_unix_sec - epoch_unix_sec;
    let p0 = spin.period_sec;
    dt / p0 - 0.5 * spin.period_dot_sec_per_sec * dt * dt / (p0 * p0)
}

fn update_minmax(target: &mut Option<(f64, f64)>, value: f64) {
    if let Some((mn, mx)) = target.as_mut() {
        *mn = mn.min(value);
        *mx = mx.max(value);
    } else {
        *target = Some((value, value));
    }
}

pub fn run_folding_analysis(
    args: &Args,
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
) -> Result<(), Box<dyn Error>> {
    let input_path = args
        .input
        .as_ref()
        .ok_or("Error: --folding requires an --input file.")?;
    let cfg = parse_folding_args(&args.folding)?;

    if args.scan_correct.is_some() {
        return Err(
            "Error: --folding currently supports --delay/--rate/--acel, but not --scan-correct."
                .into(),
        );
    }

    let file = File::open(input_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut cursor = Cursor::new(&mmap[..]);
    let header = parse_header(&mut cursor)?;

    let fft_half = (header.fft_point / 2) as usize;
    if fft_half < 2 {
        return Err("Error: invalid FFT size in header.".into());
    }
    let total_sectors = header.number_of_sector.max(0) as usize;
    if total_sectors == 0 {
        return Err("Error: no sectors in input file.".into());
    }

    cursor.set_position(0);
    let (_, first_sector_obs_time, effective_integ_time) =
        read_visibility_data(&mut cursor, &header, 1, 0, 0, false, pp_flag_ranges)?;
    if !effective_integ_time.is_finite() || effective_integ_time <= 0.0 {
        return Err("Error: invalid effective integration time.".into());
    }
    let first_sector_unix = first_sector_obs_time.timestamp() as f64
        + first_sector_obs_time.timestamp_subsec_nanos() as f64 * 1e-9;

    let bw_mhz = header.sampling_speed as f32 / 2.0 / 1_000_000.0;
    let rbw_mhz = bw_mhz / header.fft_point as f32 * 2.0;
    let rfi_ranges = parse_rfi_ranges(&args.rfi, rbw_mhz)?;
    let rfi_mask = rfi_mask(fft_half, &rfi_ranges);

    let bandpass_data = if let Some(bp_path) = &args.bandpass {
        let bp = read_bandpass_file(bp_path)?;
        if bp.len() != fft_half {
            eprintln!(
                "#WARN: bandpass channel count ({}) != FFT/2 ({}). Ignore bandpass in folding.",
                bp.len(),
                fft_half
            );
            None
        } else {
            Some(bp)
        }
    } else {
        None
    };
    let bandpass_mean = bandpass_data.as_ref().map(|bp| {
        let sum: C32 = bp.iter().copied().sum();
        sum / bp.len() as f32
    });

    let skip_sec = args.skip.max(0) as f64;
    let mut start_sector = (skip_sec / effective_integ_time as f64).floor() as usize;
    start_sector = start_sector.min(total_sectors);

    let requested_span_sec = if args.length <= 0 {
        None
    } else {
        Some(args.length as f64 * args.loop_.max(1) as f64)
    };
    let end_sector = match requested_span_sec {
        Some(span_sec) => {
            let span_sector = (span_sec / effective_integ_time as f64).ceil() as usize;
            start_sector.saturating_add(span_sector).min(total_sectors)
        }
        None => total_sectors,
    };
    if end_sector <= start_sector {
        return Err("Error: no sectors selected by --skip/--len/--loop for folding.".into());
    }

    let source_ra_rad = cfg
        .spin
        .source_ra_override_rad
        .unwrap_or(header.source_position_ra);
    let source_dec_rad = cfg
        .spin
        .source_dec_override_rad
        .unwrap_or(header.source_position_dec);
    let src_hat = source_unit_vector(source_ra_rad, source_dec_rad);
    let observer_mid_ecef_m = [
        0.5 * (header.station1_position[0] + header.station2_position[0]),
        0.5 * (header.station1_position[1] + header.station2_position[1]),
        0.5 * (header.station1_position[2] + header.station2_position[2]),
    ];

    let mut bin_sum = vec![Complex::<f64>::new(0.0, 0.0); cfg.bins];
    let mut bin_weight = vec![0usize; cfg.bins];
    let mut used_samples: usize = 0;
    let mut skipped_time_flag: usize = 0;
    let mut skipped_empty: usize = 0;
    let mut first_obs_time: Option<DateTime<Utc>> = None;
    let mut last_obs_time: Option<DateTime<Utc>> = None;
    let mut phase_epoch_unix_sec = cfg.spin.epoch_unix_sec;
    let mut bary_delay_minmax: Option<(f64, f64)> = None;
    let mut orbital_delay_minmax: Option<(f64, f64)> = None;

    cursor.set_position(0);
    for sector_idx in start_sector..end_sector {
        let (mut row, obs_time, _) = match read_visibility_data(
            &mut cursor,
            &header,
            1,
            0,
            sector_idx as i32,
            false,
            pp_flag_ranges,
        ) {
            Ok(v) => v,
            Err(_) => break,
        };

        if time_flag_ranges
            .iter()
            .any(|(start, end)| obs_time >= *start && obs_time < *end)
        {
            skipped_time_flag += 1;
            continue;
        }
        if row.len() != fft_half {
            skipped_empty += 1;
            continue;
        }

        apply_row_phase_correction(
            &mut row,
            args.rate_correct,
            args.delay_correct,
            args.acel_correct,
            effective_integ_time,
            header.sampling_speed,
            header.fft_point,
            sector_idx,
        );

        let mut coherent_sum = C32::new(0.0, 0.0);
        let mut coherent_count = 0usize;
        for chan in 1..fft_half {
            if rfi_mask[chan] {
                continue;
            }
            let mut sample = row[chan];
            if let (Some(bp), Some(bp_mean)) = (&bandpass_data, bandpass_mean) {
                let bp_val = bp[chan];
                if bp_val.norm() > BP_EPSILON {
                    sample = (sample / bp_val) * bp_mean;
                } else {
                    continue;
                }
            }
            if sample.re.is_finite() && sample.im.is_finite() {
                coherent_sum += sample;
                coherent_count += 1;
            }
        }

        if coherent_count == 0 {
            skipped_empty += 1;
            continue;
        }

        if first_obs_time.is_none() {
            first_obs_time = Some(obs_time);
        }
        last_obs_time = Some(obs_time);

        let vis = coherent_sum / coherent_count as f32;
        // Use sector index + effective integration to preserve sub-second timing.
        // .cor sector headers typically store integer seconds, which is too coarse for pulsar folding.
        let mut corrected_time_unix =
            first_sector_unix + sector_idx as f64 * effective_integ_time as f64 + 0.5 * effective_integ_time as f64;

        if cfg.spin.use_barycentric {
            let bary_delay = barycentric_delay_sec(corrected_time_unix, observer_mid_ecef_m, &src_hat);
            corrected_time_unix += bary_delay;
            update_minmax(&mut bary_delay_minmax, bary_delay);
        }

        if cfg.spin.use_orbital {
            if let Some(orbital) = &cfg.spin.orbital {
                let orbital_delay = orbital_delay_sec(corrected_time_unix, orbital);
                corrected_time_unix -= orbital_delay;
                update_minmax(&mut orbital_delay_minmax, orbital_delay);
            }
        }

        if phase_epoch_unix_sec.is_none() {
            phase_epoch_unix_sec = Some(corrected_time_unix);
        }
        let phase_epoch = phase_epoch_unix_sec.unwrap_or(corrected_time_unix);
        let phase =
            (spin_phase_cycles(corrected_time_unix, &cfg.spin, phase_epoch) + cfg.phase0).rem_euclid(1.0);
        let bin_idx = ((phase * cfg.bins as f64).floor() as usize).min(cfg.bins - 1);

        bin_sum[bin_idx] += Complex::<f64>::new(vis.re as f64, vis.im as f64);
        bin_weight[bin_idx] += 1;
        used_samples += 1;
    }

    if used_samples == 0 {
        return Err("Error: no valid samples available for folding.".into());
    }

    let mut profile: Vec<FoldBin> = (0..cfg.bins)
        .map(|idx| {
            let count = bin_weight[idx];
            let mean_complex = if count > 0 {
                bin_sum[idx] / count as f64
            } else {
                Complex::<f64>::new(0.0, 0.0)
            };
            let phase_deg = mean_complex.arg().to_degrees();
            FoldBin {
                phase_center: (idx as f64 + 0.5) / cfg.bins as f64,
                amp: mean_complex.norm(),
                mean_complex,
                phase_deg,
                count,
                on_bin: false,
            }
        })
        .collect();

    let on_mask = pick_on_bins(&profile, cfg.on_duty);
    for (idx, is_on) in on_mask.iter().enumerate() {
        profile[idx].on_bin = *is_on;
    }

    let mut peak_idx = 0usize;
    let mut peak_amp = f64::NEG_INFINITY;
    for (idx, bin) in profile.iter().enumerate() {
        if bin.amp > peak_amp {
            peak_amp = bin.amp;
            peak_idx = idx;
        }
    }
    let off_amps: Vec<f64> = profile
        .iter()
        .filter(|bin| !bin.on_bin && bin.count > 0)
        .map(|bin| bin.amp)
        .collect();
    let off_mean = if off_amps.is_empty() {
        0.0
    } else {
        off_amps.iter().sum::<f64>() / off_amps.len() as f64
    };
    let off_std = if off_amps.len() > 1 {
        let var =
            off_amps.iter().map(|v| (v - off_mean) * (v - off_mean)).sum::<f64>()
                / (off_amps.len() - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };
    let snr = if off_std > 0.0 {
        (peak_amp - off_mean) / off_std
    } else {
        0.0
    };

    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let output_dir = parent_dir.join("frinZ").join("folding");
    fs::create_dir_all(&output_dir)?;
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("folding");

    let profile_path = output_dir.join(format!("{stem}_folding_profile.tsv"));
    let mut profile_file = File::create(&profile_path)?;
    writeln!(
        profile_file,
        "bin\tphase\tamp\treal\timag\tphase_deg\tcount\ton_bin"
    )?;
    for (idx, bin) in profile.iter().enumerate() {
        writeln!(
            profile_file,
            "{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.6}\t{}\t{}",
            idx,
            bin.phase_center,
            bin.amp,
            bin.mean_complex.re,
            bin.mean_complex.im,
            bin.phase_deg,
            bin.count,
            if bin.on_bin { 1 } else { 0 }
        )?;
    }

    let profile_png_path = output_dir.join(format!("{stem}_folding_profile.png"));
    plot_folding_profile(
        &profile_png_path,
        &profile,
        peak_idx,
        off_mean,
        off_std,
        cfg.spin.period_sec,
    )?;

    let summary_path = output_dir.join(format!("{stem}_folding_summary.txt"));
    let mut summary_file = File::create(&summary_path)?;
    writeln!(summary_file, "input                 : {}", input_path.display())?;
    writeln!(summary_file, "source                : {}", header.source_name.trim())?;
    writeln!(
        summary_file,
        "station pair          : {}-{}",
        header.station1_name.trim(),
        header.station2_name.trim()
    )?;
    writeln!(
        summary_file,
        "obs frequency [MHz]   : {:.6}",
        header.observing_frequency / 1e6
    )?;
    writeln!(summary_file, "period [s]            : {:.9}", cfg.spin.period_sec)?;
    writeln!(
        summary_file,
        "period dot [s/s]      : {:.6e}",
        cfg.spin.period_dot_sec_per_sec
    )?;
    writeln!(summary_file, "bins                  : {}", cfg.bins)?;
    writeln!(summary_file, "on-duty               : {:.4}", cfg.on_duty)?;
    writeln!(summary_file, "phase0 [cycle]        : {:.6}", cfg.phase0)?;
    writeln!(
        summary_file,
        "barycentric correction: {}",
        if cfg.spin.use_barycentric { "on" } else { "off" }
    )?;
    writeln!(
        summary_file,
        "orbital correction    : {}",
        if cfg.spin.use_orbital { "on" } else { "off" }
    )?;
    if let Some(path) = &cfg.spin.ephem_path {
        writeln!(summary_file, "ephemeris file        : {}", path.display())?;
    }
    if let Some((mn, mx)) = bary_delay_minmax {
        writeln!(summary_file, "bary delay [s] min/max: {:+.9} / {:+.9}", mn, mx)?;
    }
    if let Some((mn, mx)) = orbital_delay_minmax {
        writeln!(summary_file, "orb delay [s] min/max : {:+.9} / {:+.9}", mn, mx)?;
    }
    if let Some(orb) = &cfg.spin.orbital {
        writeln!(summary_file, "orb pb [s]            : {:.9}", orb.pb_sec)?;
        writeln!(summary_file, "orb a1 [lt-s]         : {:.9}", orb.a1_sec)?;
        writeln!(summary_file, "orb ecc               : {:.9}", orb.ecc)?;
        writeln!(
            summary_file,
            "orb omega [deg]       : {:.6}",
            orb.omega_rad.to_degrees()
        )?;
    }
    writeln!(
        summary_file,
        "correction delay/rate/acel : {:+.9} samp, {:+.9} Hz, {:+.9} Hz/s",
        args.delay_correct, args.rate_correct, args.acel_correct
    )?;
    writeln!(
        summary_file,
        "effective integ [s]   : {:.6}",
        effective_integ_time
    )?;
    writeln!(
        summary_file,
        "sector range          : {}..{} (N={})",
        start_sector,
        end_sector.saturating_sub(1),
        end_sector.saturating_sub(start_sector)
    )?;
    writeln!(summary_file, "used samples          : {}", used_samples)?;
    writeln!(summary_file, "skipped by time flag  : {}", skipped_time_flag)?;
    writeln!(summary_file, "skipped empty/invalid : {}", skipped_empty)?;
    if let Some(t) = first_obs_time {
        writeln!(summary_file, "first epoch (UTC)     : {}", t)?;
    }
    if let Some(t) = last_obs_time {
        writeln!(summary_file, "last epoch (UTC)      : {}", t)?;
    }
    if let Some(tref) = phase_epoch_unix_sec {
        let sec = tref.floor() as i64;
        let mut nsec = ((tref - sec as f64) * 1e9).round() as i64;
        let mut sec_adj = sec;
        if nsec >= 1_000_000_000 {
            nsec -= 1_000_000_000;
            sec_adj += 1;
        }
        if let Some(dt) = Utc.timestamp_opt(sec_adj, nsec as u32).single() {
            writeln!(summary_file, "phase epoch (UTC)     : {}", dt)?;
        }
    }
    writeln!(
        summary_file,
        "effective folded span [s] : {:.3}",
        used_samples as f64 * effective_integ_time as f64
    )?;
    writeln!(summary_file, "peak bin              : {}", peak_idx)?;
    writeln!(
        summary_file,
        "peak phase [cycle]    : {:.9}",
        profile[peak_idx].phase_center
    )?;
    writeln!(summary_file, "peak amp              : {:.9}", peak_amp)?;
    writeln!(summary_file, "off mean              : {:.9}", off_mean)?;
    writeln!(summary_file, "off std               : {:.9}", off_std)?;
    writeln!(summary_file, "fold SNR              : {:.6}", snr)?;
    writeln!(summary_file, "profile png           : {}", profile_png_path.display())?;

    println!("Running folding analysis...");
    println!("  Input: {}", input_path.display());
    println!("  Source: {}", header.source_name.trim());
    println!("  Period [s]: {:.9}", cfg.spin.period_sec);
    println!("  Bins: {}", cfg.bins);
    println!(
        "  Barycentric correction: {}",
        if cfg.spin.use_barycentric { "on" } else { "off" }
    );
    println!(
        "  Orbital correction: {}",
        if cfg.spin.use_orbital { "on" } else { "off" }
    );
    if let Some(path) = &cfg.spin.ephem_path {
        println!("  Ephemeris file: {}", path.display());
    }
    if let Some((mn, mx)) = bary_delay_minmax {
        println!("  Bary delay [s] min/max: {:+.9} / {:+.9}", mn, mx);
    }
    if let Some((mn, mx)) = orbital_delay_minmax {
        println!("  Orb delay [s] min/max : {:+.9} / {:+.9}", mn, mx);
    }
    println!(
        "  Delay/Rate/Acel correction: {:+.9} samp, {:+.9} Hz, {:+.9} Hz/s",
        args.delay_correct, args.rate_correct, args.acel_correct
    );
    println!(
        "  Used sectors: {}..{} (used {} samples)",
        start_sector,
        end_sector.saturating_sub(1),
        used_samples
    );
    println!(
        "  Peak bin phase: {:.6} cycle, peak amp: {:.9}, fold SNR: {:.3}",
        profile[peak_idx].phase_center, peak_amp, snr
    );
    println!("  Fold profile PNG: {}", profile_png_path.display());
    println!("  Fold profile TSV: {}", profile_path.display());
    println!("  Summary: {}", summary_path.display());

    Ok(())
}
