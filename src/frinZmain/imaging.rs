use ndarray::{Array2, ArrayView1};
use num_complex::Complex;

use crate::header::CorHeader;

const C: f64 = 299792458.0; // Speed of light in m/s

/// Creates a sky image (l, m) from a delay-rate map.
///
/// # Arguments
/// * `delay_rate_array` - The 2D array of complex visibilities in the delay-rate domain.
/// * `u`, `v` - UV coordinates in meters.
/// * `du_dt`, `dv_dt` - Time derivatives of UV coordinates in meters/sec.
/// * `header` - The correlation header containing observation parameters.
/// * `rate_range` - The range of rates corresponding to the delay-rate map's axis.
/// * `delay_range` - The range of delays corresponding to the delay-rate map's axis.
/// * `image_size` - The width and height of the output image in pixels.
/// * `cell_size_rad` - The angular size of each pixel in radians.
///
/// # Returns
/// A 2D array representing the sky brightness map.
pub fn create_map(
    delay_rate_array: &Array2<Complex<f32>>,
    u: f64,
    v: f64,
    du_dt: f64,
    dv_dt: f64,
    header: &CorHeader,
    rate_range: &ArrayView1<f32>,
    delay_range: &ArrayView1<f32>,
    image_size: usize,
    cell_size_rad: f64,
) -> Array2<f32> {
    let mut image = Array2::<f32>::zeros((image_size, image_size));
    let center = (image_size / 2) as f64;
    let lambda = C / header.observing_frequency;

    let _inv_det = 1.0 / (u * dv_dt - v * du_dt);

    // Pre-calculate ranges for faster access
    let rate_min = rate_range[0] as f64;
    let rate_max = rate_range[rate_range.len() - 1] as f64;
    let rate_step = (rate_max - rate_min) / (rate_range.len() - 1) as f64;

    let delay_min = delay_range[0] as f64;
    let delay_max = delay_range[delay_range.len() - 1] as f64;
    let delay_step = (delay_max - delay_min) / (delay_range.len() - 1) as f64;

    for iy in 0..image_size {
        for ix in 0..image_size {
            // (l, m) coordinates for the current pixel
            let l = ((ix as f64) - center) * cell_size_rad;
            let m = (center - (iy as f64)) * cell_size_rad;

            // Forward transform: from (l, m) to (delay, rate)
            let delay_s = (u * l + v * m) / C;
            let rate_hz = (du_dt * l + dv_dt * m) / lambda;

            // Convert to pixel coordinates in the delay-rate map
            let delay_sample = delay_s * (header.sampling_speed as f64);

            // Find corresponding indices in the delay-rate array
            let delay_idx_f = (delay_sample - delay_min) / delay_step;
            let rate_idx_f = (rate_hz - rate_min) / rate_step;

            // Bilinear interpolation
            let x1 = delay_idx_f.floor() as usize;
            let y1 = rate_idx_f.floor() as usize;
            let x2 = x1 + 1;
            let y2 = y1 + 1;

            if x2 < delay_range.len() && y2 < rate_range.len() {
                let x_frac = delay_idx_f - x1 as f64;
                let y_frac = rate_idx_f - y1 as f64;

                let p11 = delay_rate_array[[y1, x1]].norm() as f64;
                let p12 = delay_rate_array[[y2, x1]].norm() as f64;
                let p21 = delay_rate_array[[y1, x2]].norm() as f64;
                let p22 = delay_rate_array[[y2, x2]].norm() as f64;

                let val = p11 * (1.0 - x_frac) * (1.0 - y_frac)
                        + p21 * x_frac * (1.0 - y_frac)
                        + p12 * (1.0 - x_frac) * y_frac
                        + p22 * x_frac * y_frac;

                image[[iy, ix]] = val as f32;
            }
        }
    }

    image
}
