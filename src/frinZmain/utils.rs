use std::error::Error;
use std::fs;
use std::path::Path;

use astro::coords;
use chrono::{DateTime, TimeZone, Utc};
use nav_types::{ECEF, WGS84};
use ndarray::prelude::*;
use num_complex::Complex;

use crate::header::CorHeader;
use crate::npy_output::{npz_sidecar_path, NamedNpz, NpyMeta};
use crate::output::insert_product_before_processing_suffixes;
use crate::plot;

const C: f64 = 299792458.0; // Speed of light in m/s
const SECONDS_PER_DAY: i64 = 86_400;
const J2000_UNIX_SECONDS: i64 = 946_728_000;

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
    let [x, y, z] = ant_position.map(f64::from);
    let wgs84_position: WGS84<f64> = ECEF::new(x, y, z).into();
    let longitude_radian = wgs84_position.longitude_radians();
    let latitude_radian = wgs84_position.latitude_radians();
    let height_meter = wgs84_position.altitude();

    let hour_angle = greenwich_sidereal_angle(time) + longitude_radian - f64::from(obs_ra);

    let source_az =
        coords::az_frm_eq(hour_angle, obs_dec as f64, latitude_radian).to_degrees() as f32 + 180.0;
    let source_el =
        coords::alt_frm_eq(hour_angle, obs_dec as f64, latitude_radian).to_degrees() as f32;

    (source_az, source_el, height_meter as f32)
}

pub(crate) fn seconds_between(time: DateTime<Utc>, epoch: DateTime<Utc>) -> f64 {
    (time.timestamp() - epoch.timestamp()) as f64
        + (f64::from(time.timestamp_subsec_nanos()) - f64::from(epoch.timestamp_subsec_nanos()))
            * 1e-9
}

pub fn mjd_cal(time: DateTime<Utc>) -> f64 {
    let seconds = time.timestamp();
    let day = 40_587 + seconds.div_euclid(SECONDS_PER_DAY);
    day as f64
        + (seconds.rem_euclid(SECONDS_PER_DAY) as f64
            + f64::from(time.timestamp_subsec_nanos()) * 1e-9)
            / SECONDS_PER_DAY as f64
}

/// Mean sidereal angle with UTC as the approximation to UT1. Separating the
/// whole days keeps subsecond epochs out of a large absolute Julian date.
pub(crate) fn greenwich_sidereal_angle(time: DateTime<Utc>) -> f64 {
    let seconds = time.timestamp() - J2000_UNIX_SECONDS;
    let days = seconds.div_euclid(SECONDS_PER_DAY) as f64;
    let fraction = (seconds.rem_euclid(SECONDS_PER_DAY) as f64
        + f64::from(time.timestamp_subsec_nanos()) * 1e-9)
        / SECONDS_PER_DAY as f64;
    let centuries = (days + fraction) / 36_525.0;
    (280.46061837
        + 0.98564736629 * days
        + 360.98564736629 * fraction
        + centuries * centuries * (0.000387933 - centuries / 38_710_000.0))
        .rem_euclid(360.0)
        .to_radians()
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

    for phase in phases.iter_mut().skip(1) {
        let original_current = *phase;
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
        *phase += offset;
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

// Linear term of the mean sidereal angle in radians per second.
const EARTH_ROTATION_RATE: f64 = 360.98564736629 * std::f64::consts::PI / (180.0 * 86_400.0);

pub(crate) struct UvwProjection {
    baseline: [f64; 3],
    v_offset: f64,
    ra: f64,
    sin_dec: f64,
    cos_dec: f64,
}

impl UvwProjection {
    pub(crate) fn new(
        ant1_pos: [f64; 3],
        ant2_pos: [f64; 3],
        ra: f64,
        dec: f64,
        include_vertical: bool,
    ) -> Self {
        let [x, y, mut z] = std::array::from_fn(|i| ant2_pos[i] - ant1_pos[i]);
        let mut v_offset = 0.0;
        if !include_vertical {
            let [ax, ay, az] = ant1_pos;
            let station: WGS84<f64> = ECEF::new(ax, ay, az).into();
            v_offset = z * station.latitude_radians().cos();
            z = 0.0;
        }
        let (sin_dec, cos_dec) = dec.sin_cos();
        Self {
            baseline: [x, y, z],
            v_offset,
            ra,
            sin_dec,
            cos_dec,
        }
    }

    pub(crate) fn project(&self, sidereal_angle: f64) -> (f64, f64, f64, f64, f64) {
        // The ECEF baseline is Greenwich-referenced. H increases as Earth rotates.
        let (sin_h, cos_h) = (sidereal_angle - self.ra).sin_cos();
        let [x, y, z] = self.baseline;
        let u = x * sin_h + y * cos_h;
        let equatorial = x * cos_h - y * sin_h;
        let v = -equatorial * self.sin_dec + z * self.cos_dec + self.v_offset;
        let w = equatorial * self.cos_dec + z * self.sin_dec;
        (
            u,
            v,
            w,
            equatorial * EARTH_ROTATION_RATE,
            u * self.sin_dec * EARTH_ROTATION_RATE,
        )
    }
}

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
    UvwProjection::new(
        ant1_pos,
        ant2_pos,
        obs_ra_rad,
        obs_dec_rad,
        include_vertical,
    )
    .project(greenwich_sidereal_angle(time))
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
    use super::*;
    use chrono::Duration;

    #[test]
    fn uvw_matches_the_ecef_sky_basis_and_time_derivative() {
        let epoch =
            Utc.with_ymd_and_hms(2026, 1, 19, 2, 34, 0).unwrap() + Duration::milliseconds(127);
        let ra = 2.1;
        let dec = 0.8_f64;
        let baseline = [112_392.0, -427_271.0, 182_582.5];
        let angle = ra - greenwich_sidereal_angle(epoch);
        let (sin, cos) = angle.sin_cos();
        let (sin_dec, cos_dec) = dec.sin_cos();
        let dot = |axis: [f64; 3]| baseline.iter().zip(axis).map(|(a, b)| a * b).sum::<f64>();
        let model = UvwProjection::new([0.0; 3], baseline, ra, dec, true);
        let (u, v, w, du, dv) = model.project(greenwich_sidereal_angle(epoch));
        assert!((u - dot([-sin, cos, 0.0])).abs() < 1e-9);
        assert!((v - dot([-cos * sin_dec, -sin * sin_dec, cos_dec])).abs() < 1e-9);
        assert!((w - dot([cos * cos_dec, sin * cos_dec, sin_dec])).abs() < 1e-9);
        let (u0, v0, _, _, _) =
            model.project(greenwich_sidereal_angle(epoch - Duration::milliseconds(25)));
        let (u1, v1, _, _, _) =
            model.project(greenwich_sidereal_angle(epoch + Duration::milliseconds(25)));
        assert!(((u1 - u0) / 0.05 - du).abs() < 5e-6);
        assert!(((v1 - v0) / 0.05 - dv).abs() < 5e-6);
    }

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
