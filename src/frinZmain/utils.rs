use std::error::Error;
use std::fs;
use std::path::Path;

use astro::coords;
use astro::time;
use chrono::{DateTime, Datelike, TimeZone, Timelike, Utc};
use nav_types::{ECEF, WGS84};
use ndarray::prelude::*;
use num_complex::Complex;

use crate::header::CorHeader;
use crate::npy_output::{npz_sidecar_path, NamedNpz, NpyMeta};
use crate::output::insert_product_before_processing_suffixes;
use crate::plot;

const C: f64 = 299792458.0; // Speed of light in m/s

type C32 = Complex<f32>;

pub fn safe_arg(z: &C32) -> f32 {
    // An argument of the complex 0+0j in Rust is 180 deg = pi
    if z.re == 0.0 && z.im == 0.0 {
        0.0 // Python の仕様に合わせる
    } else {
        z.arg()
    }
}

pub fn rate_cal(n: f32, d: f32) -> Vec<f32> {
    let mut rate: Vec<f32> = Vec::with_capacity(n as usize);
    let step = 1.0 / (n * d); // 各ステップの大きさを計算
    if n % 2.0 == 0.0 {
        for i in (-n as i32 / 2)..(n as i32 / 2) {
            rate.push(i as f32 * step);
        }
    } else {
        for i in (-(n as i32 - 1) / 2)..((n as i32 - 1) / 2 + 1) {
            rate.push(i as f32 * step);
        }
    }
    rate
}

pub fn positive_or_epsilon(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        f32::EPSILON
    }
}

pub fn window_bounds(window: &[f32]) -> Option<(f32, f32)> {
    if window.len() == 2 {
        Some((window[0].min(window[1]), window[0].max(window[1])))
    } else {
        None
    }
}

pub fn in_window(value: f32, bounds: Option<(f32, f32)>) -> bool {
    match bounds {
        Some((low, high)) => value >= low && value <= high,
        None => true,
    }
}

pub fn delay_rate_mask_bounds(mask: &[f32]) -> Option<(f32, f32, f32, f32)> {
    if mask.len() == 4 {
        Some((
            mask[0].min(mask[1]),
            mask[0].max(mask[1]),
            mask[2].min(mask[3]),
            mask[2].max(mask[3]),
        ))
    } else {
        None
    }
}

pub fn in_delay_rate_mask(delay: f32, rate: f32, mask: Option<(f32, f32, f32, f32)>) -> bool {
    match mask {
        Some((delay_min, delay_max, rate_min, rate_max)) => {
            delay >= delay_min && delay <= delay_max && rate >= rate_min && rate <= rate_max
        }
        None => false,
    }
}

pub fn noise_level(array: ArrayView2<C32>, array_mean: C32) -> f32 {
    let (rows, cols) = array.dim();
    if rows == 0 || cols == 0 {
        return 0.0;
    }
    let array_sum: f32 = array.iter().map(|x| (x - array_mean).norm()).sum();
    array_sum / (rows * cols) as f32
}

pub fn radec2azalt(
    ant_position: [f32; 3],
    time: DateTime<Utc>,
    obs_ra: f32,
    obs_dec: f32,
) -> (f32, f32, f32) {
    let obs_year = time.year() as i16;
    let obs_month = time.month() as u8;
    let obs_day = time.day() as u8;
    let obs_hour = time.hour() as u8;
    let obs_minute = time.minute() as u8;
    let obs_second = time.second() as f64; // + (time.nanosecond() as f64 / 1_000_000_000.0);

    let decimal_day_calc = obs_day as f64
        + obs_hour as f64 / 24.0
        + obs_minute as f64 / 60.0 / 24.0
        + obs_second as f64 / 24.0 / 60.0 / 60.0;

    let date = time::Date {
        year: obs_year,
        month: obs_month,
        decimal_day: decimal_day_calc,
        cal_type: time::CalType::Gregorian,
    };

    let ecef_position = ECEF::new(
        ant_position[0] as f64,
        ant_position[1] as f64,
        ant_position[2] as f64,
    );
    let wgs84_position: WGS84<f64> = ecef_position.into();
    let longitude_radian = wgs84_position.longitude_radians();
    let latitude_radian = wgs84_position.latitude_radians();
    let height_meter = wgs84_position.altitude();

    let julian_day = time::julian_day(&date);
    let mean_sidereal = time::mn_sidr(julian_day);
    let hour_angle =
        coords::hr_angl_frm_observer_long(mean_sidereal, -longitude_radian, obs_ra as f64);

    let source_az =
        coords::az_frm_eq(hour_angle, obs_dec as f64, latitude_radian).to_degrees() as f32 + 180.0;
    let source_el =
        coords::alt_frm_eq(hour_angle, obs_dec as f64, latitude_radian).to_degrees() as f32;

    (source_az, source_el, height_meter as f32)
}

pub fn mjd_cal(time: DateTime<Utc>) -> f64 {
    let obs_year = time.year() as i16;
    let obs_month = time.month() as u8;
    let obs_day = time.day() as u8;
    let obs_hour = time.hour() as u8;
    let obs_minute = time.minute() as u8;
    let obs_second = time.second() as f64; // + (time.nanosecond() as f64 / 1_000_000_000.0);

    let decimal_day_calc = obs_day as f64
        + obs_hour as f64 / 24.0
        + obs_minute as f64 / 60.0 / 24.0
        + obs_second as f64 / 24.0 / 60.0 / 60.0;

    let date = time::Date {
        year: obs_year,
        month: obs_month,
        decimal_day: decimal_day_calc,
        cal_type: time::CalType::Gregorian,
    };

    let julian_day = time::julian_day(&date);
    julian_day - 2400000.5
}

pub fn unwrap_phase_with_rate(phases: &[f32], times_sec: &[f32], rates_hz: &[f32]) -> Vec<f32> {
    if phases.len() < 2 || phases.len() != times_sec.len() || phases.len() != rates_hz.len() {
        let mut fallback = phases.to_vec();
        unwrap_phase(&mut fallback, false);
        return fallback;
    }

    let mut unwrapped = Vec::with_capacity(phases.len());
    unwrapped.push(phases[0]);
    for index in 1..phases.len() {
        let dt = times_sec[index] - times_sec[index - 1];
        let predicted_delta = 360.0 * 0.5 * (rates_hz[index - 1] + rates_hz[index]) * dt;
        let wrapped_delta = phases[index] - phases[index - 1];
        let turns = ((predicted_delta - wrapped_delta) / 360.0).round();
        unwrapped.push(unwrapped[index - 1] + wrapped_delta + 360.0 * turns);
    }
    unwrapped
}

pub fn unwrap_phase(phases: &mut [f32], radians: bool) {
    if phases.len() < 2 {
        return;
    }
    let mut offset = 0.0;
    let mut original_prev = phases[0];

    for i in 1..phases.len() {
        let original_current = phases[i];
        let diff = original_current - original_prev;
        if radians {
            if diff > std::f32::consts::PI {
                offset -= 2.0 * std::f32::consts::PI;
            } else if diff < -std::f32::consts::PI {
                offset += 2.0 * std::f32::consts::PI;
            }
        } else {
            if diff > 180.0 {
                offset -= 360.0;
            } else if diff < -180.0 {
                offset += 360.0;
            }
        }
        phases[i] += offset;
        original_prev = original_current;
    }
}

use chrono::{NaiveDate, NaiveDateTime, NaiveTime};

pub fn parse_flag_time(time_str: &str) -> Option<DateTime<Utc>> {
    if time_str.len() != 13 {
        return None;
    }
    let year: i32 = time_str.get(0..4)?.parse().ok()?;
    let doy: u32 = time_str.get(4..7)?.parse().ok()?;
    let hour: u32 = time_str.get(7..9)?.parse().ok()?;
    let min: u32 = time_str.get(9..11)?.parse().ok()?;
    let sec: u32 = time_str.get(11..13)?.parse().ok()?;

    let date = NaiveDate::from_yo_opt(year, doy)?;
    let time = NaiveTime::from_hms_opt(hour, min, sec)?;
    let datetime = NaiveDateTime::new(date, time);
    Some(Utc.from_utc_datetime(&datetime))
}

pub fn calculate_allan_deviation(phases: &[f32], tau0: f32, obs_freq_hz: f64) -> Vec<(f32, f32)> {
    let n = phases.len();
    if n < 3 {
        return Vec::new();
    }
    // Convert phase from degrees to radians for calculation
    let rad_phases: Vec<f64> = phases.iter().map(|p| (*p as f64).to_radians()).collect();
    let tau0_f64 = tau0 as f64;

    let mut adev_results = Vec::new();
    let max_m = n / 3; // Calculate for cluster sizes up to 1/3 of the data length

    for m in 1..=max_m {
        if m == 0 {
            continue;
        }
        let tau = m as f64 * tau0_f64;
        let mut sum_sq_diff = 0.0;
        let num_terms = n - 2 * m;

        if num_terms == 0 {
            break;
        }

        for i in 0..num_terms {
            let diff = rad_phases[i + 2 * m] - 2.0 * rad_phases[i + m] + rad_phases[i];
            sum_sq_diff += diff * diff;
        }

        if tau == 0.0 {
            continue;
        }

        // Calculate the standard (dimensionless) Allan variance σ_y^2(τ)
        let allan_variance = sum_sq_diff / (2.0 * tau * tau * (num_terms as f64));
        let phase_rate_adev = allan_variance.sqrt(); // This is σ_φ_dot in rad/s

        // Normalize by 2*pi*ν₀ to get dimensionless fractional frequency stability
        let dimensionless_adev = phase_rate_adev / (2.0 * std::f64::consts::PI * obs_freq_hz);

        adev_results.push((tau as f32, dimensionless_adev as f32));
    }
    adev_results
}

pub fn write_allan_deviation_outputs(
    phases: &[f32],
    tau0: f32,
    obs_freq_hz: f64,
    source_name: &str,
    base_filename: &str,
    frinz_dir: &Path,
    write_npz: bool,
) -> Result<(), Box<dyn Error>> {
    if phases.len() < 3 {
        eprintln!(
            "Warning: Not enough data points ({}) to calculate Allan deviation. Use --length and --loop to generate a time series. Skipping.",
            phases.len()
        );
        return Ok(());
    }

    println!("Calculating Allan deviation...");
    let allan_dir = frinz_dir.join("allan_deviance");
    fs::create_dir_all(&allan_dir)?;

    let adev_data = calculate_allan_deviation(phases, tau0, obs_freq_hz);
    let product = format!("{source_name}_allan");
    let adev_basename = insert_product_before_processing_suffixes(base_filename, &product);

    let _ = fs::remove_file(allan_dir.join(format!("{}.txt", adev_basename)));

    let plot_path = allan_dir.join(format!("{}.png", adev_basename));
    plot::plot_allan_deviation(plot_path.to_str().unwrap(), &adev_data, source_name)?;
    if write_npz {
        let tau_axis: Vec<f64> = adev_data.iter().map(|&(tau, _)| tau as f64).collect();
        let values: Vec<f32> = adev_data.iter().map(|&(_, value)| value).collect();
        let mut npz = NamedNpz::new(NpyMeta::new("allan_deviance", 0, 0));
        npz.add_f64_1d("tau_s", &tau_axis);
        npz.add_f32_1d("allan_deviation", &values);
        npz.write(&npz_sidecar_path(&plot_path, "allan_deviance"))?;
    }
    println!("Allan deviation plot and data saved in {:?}", allan_dir);

    Ok(())
}

// Earth's rotation rate in radians per second (sidereal)
const EARTH_ROTATION_RATE: f64 = 7.2921150e-5;

/// Calculates (u, v, w) coordinates and their time derivatives.
///
/// # Arguments
/// * `ant1_pos` - Geocentric coordinates [x, y, z] of antenna 1 in meters.
/// * `ant2_pos` - Geocentric coordinates [x, y, z] of antenna 2 in meters.
/// * `time` - Observation time as DateTime<Utc>.
/// * `obs_ra_rad` - Source Right Ascension in radians.
/// * `obs_dec_rad` - Source Declination in radians.
///
/// # Returns
/// A tuple containing (u, v, w, du/dt, dv/dt) in meters and meters/sec.
pub fn uvw_cal(
    ant1_pos: [f64; 3],
    ant2_pos: [f64; 3],
    time: DateTime<Utc>,
    obs_ra_rad: f64,
    obs_dec_rad: f64,
    include_vertical: bool,
) -> (f64, f64, f64, f64, f64) {
    let (b_x, b_y, mut b_z) = (
        ant2_pos[0] - ant1_pos[0],
        ant2_pos[1] - ant1_pos[1],
        ant2_pos[2] - ant1_pos[2],
    );

    // The baseline components above are ECEF (Earth-fixed, Greenwich-referenced) XYZ coordinates.
    // Only latitude is needed for the optional planar correction; using the station
    // longitude here would mix an ECEF baseline with a local hour angle.
    let ecef_position = ECEF::new(ant1_pos[0], ant1_pos[1], ant1_pos[2]);
    let wgs84_position: WGS84<f64> = ecef_position.into();
    let latitude_rad = wgs84_position.latitude_radians();

    let v_offset = if !include_vertical {
        let offset = b_z * latitude_rad.cos();
        b_z = 0.0;
        offset
    } else {
        0.0
    };
    // Calculate Hour Angle (H)
    let date = time::Date {
        year: time.year() as i16,
        month: time.month() as u8,
        decimal_day: time.day() as f64
            + (time.hour() as f64 / 24.0)
            + (time.minute() as f64 / 1440.0)
            + (time.second() as f64 / 86400.0),
        cal_type: time::CalType::Gregorian,
    };
    let julian_day = time::julian_day(&date);
    let mean_sidereal = time::mn_sidr(julian_day);
    let hour_angle = coords::hr_angl_frm_observer_long(mean_sidereal, 0.0, obs_ra_rad);
    let east_positive_hour_angle = -hour_angle;

    let sin_h = east_positive_hour_angle.sin();
    let cos_h = east_positive_hour_angle.cos();
    let sin_d = obs_dec_rad.sin();
    let cos_d = obs_dec_rad.cos();

    // UVW calculation
    let u = b_x * sin_h + b_y * cos_h;
    let v = -b_x * cos_h * sin_d + b_y * sin_h * sin_d + b_z * cos_d + v_offset;
    let w = b_x * cos_h * cos_d - b_y * sin_h * cos_d + b_z * sin_d;

    // Time derivatives
    let du_dt = (b_x * cos_h - b_y * sin_h) * EARTH_ROTATION_RATE;
    let dv_dt = (-b_x * -sin_h * sin_d + b_y * cos_h * sin_d) * EARTH_ROTATION_RATE;

    (u, v, w, du_dt, dv_dt)
}

/// Converts rate (Hz) and delay (samples) to sky coordinates (l, m).
///
/// # Arguments
/// * `rate_hz` - Fringe rate in Hz.
/// * `delay_samples` - Delay in samples.
/// * `header` - The correlation header.
/// * `u`, `v` - UV coordinates in meters.
/// * `du_dt`, `dv_dt` - Time derivatives of UV coordinates in meters/sec.
///
/// # Returns
/// A tuple (l, m) representing the direction cosines.
pub fn rate_delay_to_lm(
    rate_hz: f64,
    delay_samples: f64,
    header: &CorHeader,
    u: f64,
    v: f64,
    du_dt: f64,
    dv_dt: f64,
) -> (f64, f64) {
    let lambda = C / header.observing_frequency;
    let delay_s = delay_samples / (header.sampling_speed as f64);

    // We need to solve the linear system:
    // | u      v      | | l | = | delay_s * C      |
    // | du_dt  dv_dt  | | m |   | rate_hz * lambda |

    let det = u * dv_dt - v * du_dt;

    if det.abs() < 1e-9 {
        // Determinant is close to zero, cannot solve. Return (0, 0) or handle error.
        return (0.0, 0.0);
    }

    let inv_det = 1.0 / det;

    let b1 = delay_s * C;
    let b2 = rate_hz * lambda;

    let l = inv_det * (dv_dt * b1 - v * b2);
    let m = inv_det * (-du_dt * b1 + u * b2);

    (l, m)
}

#[cfg(test)]
mod tests {
    use super::unwrap_phase_with_rate;

    #[test]
    fn rate_assisted_unwrap_recovers_multiple_turns_between_samples() {
        let times = [0.0_f32, 30.0, 60.0, 90.0];
        for rate in [-0.016_f32, -0.17] {
            let true_phase: Vec<f32> = times
                .iter()
                .map(|time| 25.0 + 360.0 * rate * time)
                .collect();
            let wrapped: Vec<f32> = true_phase
                .iter()
                .map(|phase| (*phase + 180.0).rem_euclid(360.0) - 180.0)
                .collect();
            let rates = vec![rate; times.len()];
            let actual = unwrap_phase_with_rate(&wrapped, &times, &rates);
            for (actual, expected) in actual.iter().zip(&true_phase) {
                assert!(
                    (actual - expected).abs() < 1.0e-3,
                    "rate={rate}: {actual} != {expected}"
                );
            }
        }
    }
}
