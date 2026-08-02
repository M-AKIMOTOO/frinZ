use ndarray::prelude::*;
use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};

type C32 = Complex<f32>;
type C64 = Complex<f64>;

static FFT_PLAN_CACHE: OnceLock<Mutex<HashMap<(usize, bool), Arc<dyn Fft<f32>>>>> = OnceLock::new();

pub(crate) fn cached_fft_plan(len: usize, inverse: bool) -> Arc<dyn Fft<f32>> {
    let cache = FFT_PLAN_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(plan) = cache
        .lock()
        .expect("FFT plan cache poisoned")
        .get(&(len, inverse))
        .cloned()
    {
        return plan;
    }

    let mut planner = FftPlanner::new();
    let plan = if inverse {
        planner.plan_fft_inverse(len)
    } else {
        planner.plan_fft_forward(len)
    };
    cache
        .lock()
        .expect("FFT plan cache poisoned")
        .insert((len, inverse), plan.clone());
    plan
}

#[derive(Clone, Copy)]
struct PhaseCorrection {
    rate_hz: f32,
    delay_samples: f32,
    acel_hz: f32,
    jerk_hz_per_s2: f32,
    snap_hz_per_s3: f32,
    effective_integration_length: f32,
    start_time_offset_sec: f32,
    /// Frequency at which rate/acel/jerk/snap are defined. The .cor header
    /// stores the lower edge of the observed band here.
    reference_frequency_hz: f64,
}

impl PhaseCorrection {
    fn is_enabled(self) -> bool {
        self.rate_hz != 0.0
            || self.delay_samples != 0.0
            || self.acel_hz != 0.0
            || self.jerk_hz_per_s2 != 0.0
            || self.snap_hz_per_s3 != 0.0
    }

    fn is_valid_for(self, sampling_speed: u32, fft_point: u32) -> bool {
        self.is_enabled()
            && sampling_speed > 0
            && fft_point >= 2
            && (self.effective_integration_length as f64).abs() > 1e-9
    }
}

pub fn process_fft(
    complex_vec: &[C32],
    physical_length: i32,
    fft_point: i32,
    sampling_speed: i32,
    rfi_ranges: &[(usize, usize)],
    rate_padding: u32,
) -> (Array2<C32>, usize) {
    process_fft_impl(
        complex_vec,
        physical_length,
        fft_point,
        sampling_speed,
        rfi_ranges,
        rate_padding,
        None,
    )
}

/// Backward-compatible narrow-band correction. Prefer
/// `process_fft_with_phase_correction_at_frequency` when the observing
/// frequency is known so rate-derived delay drift is also removed.
#[allow(dead_code)]
pub fn process_fft_with_phase_correction(
    complex_vec: &[C32],
    physical_length: i32,
    fft_point: i32,
    sampling_speed: i32,
    rfi_ranges: &[(usize, usize)],
    rate_padding: u32,
    rate_hz_for_correction: f32,
    delay_samples_for_correction: f32,
    acel_hz_for_correction: f32,
    jerk_hz_per_s2_for_correction: f32,
    snap_hz_per_s3_for_correction: f32,
    effective_integration_length: f32,
    start_time_offset_sec: f32,
) -> (Array2<C32>, usize) {
    process_fft_with_phase_correction_at_frequency(
        complex_vec,
        physical_length,
        fft_point,
        sampling_speed,
        rfi_ranges,
        rate_padding,
        rate_hz_for_correction,
        delay_samples_for_correction,
        acel_hz_for_correction,
        jerk_hz_per_s2_for_correction,
        snap_hz_per_s3_for_correction,
        effective_integration_length,
        start_time_offset_sec,
        0.0,
    )
}

pub fn process_fft_with_phase_correction_at_frequency(
    complex_vec: &[C32],
    physical_length: i32,
    fft_point: i32,
    sampling_speed: i32,
    rfi_ranges: &[(usize, usize)],
    rate_padding: u32,
    rate_hz_for_correction: f32,
    delay_samples_for_correction: f32,
    acel_hz_for_correction: f32,
    jerk_hz_per_s2_for_correction: f32,
    snap_hz_per_s3_for_correction: f32,
    effective_integration_length: f32,
    start_time_offset_sec: f32,
    reference_frequency_hz: f64,
) -> (Array2<C32>, usize) {
    let phase = PhaseCorrection {
        rate_hz: rate_hz_for_correction,
        delay_samples: delay_samples_for_correction,
        acel_hz: acel_hz_for_correction,
        jerk_hz_per_s2: jerk_hz_per_s2_for_correction,
        snap_hz_per_s3: snap_hz_per_s3_for_correction,
        effective_integration_length,
        start_time_offset_sec,
        reference_frequency_hz,
    };
    process_fft_impl(
        complex_vec,
        physical_length,
        fft_point,
        sampling_speed,
        rfi_ranges,
        rate_padding,
        Some(phase),
    )
}

fn process_fft_impl(
    complex_vec: &[C32],
    physical_length: i32,
    fft_point: i32,
    sampling_speed: i32,
    rfi_ranges: &[(usize, usize)],
    rate_padding: u32,
    phase_correction: Option<PhaseCorrection>,
) -> (Array2<C32>, usize) {
    let fft_point_half = (fft_point / 2) as usize;
    let rows = if fft_point_half == 0 {
        0
    } else {
        complex_vec.len() / fft_point_half
    };
    let base_length = rows.max(1);
    let mut padding_length = base_length.saturating_mul(rate_padding.max(1) as usize);
    if base_length == 1 {
        padding_length = padding_length.saturating_mul(2);
    }
    let padding_length_half = padding_length / 2;
    let length_f32 = if physical_length > 0 {
        physical_length as f32
    } else {
        1.0
    };
    let fft_scale = if length_f32 > 0.0 {
        fft_point as f32 / length_f32
    } else {
        1.0
    };
    let bandwidth_hz = sampling_speed as f32 / 2.0;
    let bandwidth_mhz = bandwidth_hz / 1_000_000.0;
    let power_scale = if bandwidth_mhz > 0.0 {
        512.0 / bandwidth_mhz
    } else {
        1.0
    };
    let scale_factor = fft_scale * power_scale;

    let fft = cached_fft_plan(padding_length, false);

    let mut freq_rate_array = Array2::<C32>::zeros((fft_point_half, padding_length));
    let mut fft_exe = vec![C32::new(0.0, 0.0); padding_length];
    let mut rfi_mask = vec![false; fft_point_half];
    for &(min, max) in rfi_ranges {
        if min >= fft_point_half {
            continue;
        }
        let end = max.min(fft_point_half.saturating_sub(1));
        if end < min {
            continue;
        }
        for masked in &mut rfi_mask[min..=end] {
            *masked = true;
        }
    }

    let mut phase_factors = phase_correction.and_then(|phase| {
        build_phase_factors(
            phase,
            fft_point_half,
            rows,
            sampling_speed as u32,
            fft_point as u32,
        )
    });

    for i in 1..fft_point_half {
        if let Some((channel_steps, channel_factors, _)) = &mut phase_factors {
            for j in 0..rows {
                channel_factors[j] *= channel_steps[j];
            }
        }
        if rfi_mask[i] {
            continue;
        }

        for j in 0..rows {
            let mut sample = complex_vec[j * fft_point_half + i];
            if let Some((_, channel_factors, row_factors)) = &phase_factors {
                let channel_factor =
                    C32::new(channel_factors[j].re as f32, channel_factors[j].im as f32);
                sample *= row_factors[j] * channel_factor;
            }
            fft_exe[j] = sample;
        }
        fft_exe[rows..].fill(C32::new(0.0, 0.0));

        fft.process(&mut fft_exe);

        let (first_half, second_half) = fft_exe.split_at(padding_length_half);
        let mut row = freq_rate_array.row_mut(i);
        for (dst, src) in row
            .iter_mut()
            .zip(second_half.iter().chain(first_half.iter()))
        {
            *dst = *src * scale_factor;
        }
    }

    (freq_rate_array, padding_length)
}

fn build_phase_factors(
    phase: PhaseCorrection,
    _fft_point_half: usize,
    rows: usize,
    sampling_speed: u32,
    fft_point: u32,
) -> Option<(Vec<C64>, Vec<C64>, Vec<C32>)> {
    if !phase.is_valid_for(sampling_speed, fft_point) {
        return None;
    }

    let freq_resolution_hz = sampling_speed as f64 / fft_point as f64;
    let delay_seconds = phase.delay_samples as f64 / sampling_speed as f64;
    let use_wideband_rate = phase.reference_frequency_hz.is_finite()
        && phase.reference_frequency_hz.abs() > f64::EPSILON;
    let temporal_cycles: Vec<f64> = (0..rows)
        .map(|row_idx| {
            let time_sec = row_idx as f64 * phase.effective_integration_length as f64
                + phase.start_time_offset_sec as f64;
            phase.rate_hz as f64 * time_sec
                + 0.5 * phase.acel_hz as f64 * time_sec.powi(2)
                + (phase.jerk_hz_per_s2 as f64 / 6.0) * time_sec.powi(3)
                + (phase.snap_hz_per_s3 as f64 / 24.0) * time_sec.powi(4)
        })
        .collect();

    // rate/acel/... are phase derivatives at reference_frequency_hz. For a
    // channel offset f, the Taylor phase is
    //
    //   2 pi [ f*tau_0 + (1 + f/nu_ref) P(t) ],
    //
    // where P(t)=rate*t + acel*t^2/2 + ... . The f*P(t)/nu_ref term is the
    // delay drift implied by the rate and is important for long integrations.
    let channel_steps: Vec<C64> = temporal_cycles
        .iter()
        .map(|&cycles| {
            let time_varying_delay = if use_wideband_rate {
                cycles / phase.reference_frequency_hz
            } else {
                0.0
            };
            let angle = -2.0 * PI * (delay_seconds + time_varying_delay) * freq_resolution_hz;
            C64::new(angle.cos(), angle.sin())
        })
        .collect();

    let row_factors = temporal_cycles
        .iter()
        .map(|&cycles| {
            let angle = -2.0 * PI * cycles;
            C32::new(angle.cos() as f32, angle.sin() as f32)
        })
        .collect();

    let channel_factors = vec![C64::new(1.0, 0.0); rows];
    Some((channel_steps, channel_factors, row_factors))
}

pub fn process_ifft(
    freq_rate_array: &Array2<C32>,
    fft_point: i32,
    padding_length: usize,
) -> Array2<C32> {
    process_ifft_with_delay_padding(freq_rate_array, fft_point, padding_length, 1)
}

/// Transform the frequency axis to delay with optional zero padding.
///
/// Padding interpolates the delay spectrum without changing its physical
/// range. Dividing by the unpadded FFT size keeps amplitudes identical at
/// the original integer-delay samples.
pub fn process_ifft_with_delay_padding(
    freq_rate_array: &Array2<C32>,
    fft_point: i32,
    padding_length: usize,
    delay_padding: usize,
) -> Array2<C32> {
    let fft_point_usize = fft_point as usize;
    let delay_padding = delay_padding.max(1);
    let padded_fft_point = fft_point_usize.saturating_mul(delay_padding);
    let mut delay_rate_array = Array2::<C32>::zeros((padding_length, padded_fft_point));
    let ifft = cached_fft_plan(padded_fft_point, true);
    let mut ifft_exe = vec![C32::new(0.0, 0.0); padded_fft_point];
    let freq_bins = freq_rate_array.dim().0.min(padded_fft_point);
    let scale = fft_point_usize as f32;

    for i in 0..freq_rate_array.dim().1 {
        for (dst, src) in ifft_exe[..freq_bins]
            .iter_mut()
            .zip(freq_rate_array.column(i).iter().take(freq_bins))
        {
            *dst = *src;
        }
        ifft_exe[freq_bins..].fill(C32::new(0.0, 0.0));

        ifft.process(&mut ifft_exe);

        let half = padded_fft_point / 2;
        let (first_half, second_half) = ifft_exe.split_at(half);
        let mut row = delay_rate_array.row_mut(i);
        for (dst, src) in row.iter_mut().take(half).zip(first_half.iter().rev()) {
            *dst = *src / scale;
        }
        for (dst, src) in row.iter_mut().skip(half).zip(second_half.iter().rev()) {
            *dst = *src / scale;
        }
    }

    delay_rate_array
}

pub fn perform_ifft_on_vec(input: &[C32], ifft_size: usize) -> Vec<C32> {
    let ifft = cached_fft_plan(ifft_size, true);

    let mut ifft_exe = vec![C32::new(0.0, 0.0); ifft_size];
    ifft_exe[..input.len()].copy_from_slice(input);

    ifft.process(&mut ifft_exe);

    let mut shifted_out = vec![C32::new(0.0, 0.0); ifft_size];
    let half = ifft_size / 2;
    let (first_half, second_half) = ifft_exe.split_at(half);
    let scale = ifft_size as f32;
    for (dst, src) in shifted_out
        .iter_mut()
        .take(first_half.len())
        .zip(first_half.iter().rev())
    {
        *dst = *src / scale;
    }
    for (dst, src) in shifted_out
        .iter_mut()
        .skip(first_half.len())
        .zip(second_half.iter().rev())
    {
        *dst = *src / scale;
    }

    shifted_out
}

/// Backward-compatible narrow-band correction. Prefer
/// `apply_phase_correction_in_place_at_frequency` when the observing
/// frequency is known so rate-derived delay drift is also removed.
#[allow(dead_code)]
pub fn apply_phase_correction_in_place(
    data: &mut [C32],
    fft_point_half: usize,
    rate_hz_for_correction: f32,
    delay_samples_for_correction: f32,
    acel_hz_for_correction: f32,
    jerk_hz_per_s2_for_correction: f32,
    snap_hz_per_s3_for_correction: f32,
    effective_integration_length: f32,
    sampling_speed: u32,
    fft_point: u32,
    start_time_offset_sec: f32,
) {
    apply_phase_correction_in_place_at_frequency(
        data,
        fft_point_half,
        rate_hz_for_correction,
        delay_samples_for_correction,
        acel_hz_for_correction,
        jerk_hz_per_s2_for_correction,
        snap_hz_per_s3_for_correction,
        effective_integration_length,
        sampling_speed,
        fft_point,
        start_time_offset_sec,
        0.0,
    );
}

pub fn apply_phase_correction_in_place_at_frequency(
    data: &mut [C32],
    fft_point_half: usize,
    rate_hz_for_correction: f32,
    delay_samples_for_correction: f32,
    acel_hz_for_correction: f32,
    jerk_hz_per_s2_for_correction: f32,
    snap_hz_per_s3_for_correction: f32,
    effective_integration_length: f32,
    sampling_speed: u32,
    fft_point: u32,
    start_time_offset_sec: f32,
    reference_frequency_hz: f64,
) {
    if data.is_empty() || fft_point_half == 0 || data.len() % fft_point_half != 0 {
        return;
    }

    let rows = data.len() / fft_point_half;
    let phase = PhaseCorrection {
        rate_hz: rate_hz_for_correction,
        delay_samples: delay_samples_for_correction,
        acel_hz: acel_hz_for_correction,
        jerk_hz_per_s2: jerk_hz_per_s2_for_correction,
        snap_hz_per_s3: snap_hz_per_s3_for_correction,
        effective_integration_length,
        start_time_offset_sec,
        reference_frequency_hz,
    };
    let Some((channel_steps, _, row_factors)) =
        build_phase_factors(phase, fft_point_half, rows, sampling_speed, fft_point)
    else {
        return;
    };

    for (row_idx, row) in data.chunks_mut(fft_point_half).enumerate() {
        let mut channel_factor = C64::new(1.0, 0.0);
        for sample in row.iter_mut() {
            let channel_factor_f32 = C32::new(channel_factor.re as f32, channel_factor.im as f32);
            *sample *= row_factors[row_idx] * channel_factor_f32;
            channel_factor *= channel_steps[row_idx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_phase_correction_fft_matches_pre_corrected_fft() {
        let fft_point = 8;
        let fft_point_half = (fft_point / 2) as usize;
        let rows = 5usize;
        let input: Vec<C32> = (0..rows * fft_point_half)
            .map(|idx| C32::new(idx as f32 * 0.25 + 1.0, idx as f32 * -0.125))
            .collect();
        let mut corrected = input.clone();
        apply_phase_correction_in_place_at_frequency(
            &mut corrected,
            fft_point_half,
            0.03,
            0.2,
            0.001,
            0.0,
            0.0,
            0.5,
            32_000_000,
            fft_point as u32,
            0.25,
            8_400_000_000.0,
        );

        let (expected, expected_padding) =
            process_fft(&corrected, rows as i32, fft_point, 32_000_000, &[], 1);
        let (actual, actual_padding) = process_fft_with_phase_correction_at_frequency(
            &input,
            rows as i32,
            fft_point,
            32_000_000,
            &[],
            1,
            0.03,
            0.2,
            0.001,
            0.0,
            0.0,
            0.5,
            0.25,
            8_400_000_000.0,
        );

        assert_eq!(expected_padding, actual_padding);
        assert_eq!(expected.dim(), actual.dim());
        for (expected, actual) in expected.iter().zip(actual.iter()) {
            assert!((expected.re - actual.re).abs() < 1.0e-4);
            assert!((expected.im - actual.im).abs() < 1.0e-4);
        }
    }

    #[test]
    fn wideband_taylor_correction_leaves_a_fixed_delay_over_600_seconds() {
        let sampling_speed = 32_000_000u32;
        let reference_frequency_hz = 8_400_000_000.0;
        let fft_point = 64u32;
        let fft_point_half = (fft_point / 2) as usize;
        let rows = 3usize;
        let integration_time = 300.0f32;
        let fixed_delay_samples = 1.25f64;
        let rate_hz = 0.731f32;
        let acel_hz_per_s = 0.0017f32;
        let frequency_step_hz = sampling_speed as f64 / fft_point as f64;
        let fixed_delay_seconds = fixed_delay_samples / sampling_speed as f64;

        // Synthetic visibility follows phi(nu,t)=2*pi*nu*tau(t), with rate
        // and acceleration defined at reference_frequency_hz. After removing
        // those time-dependent terms, only the fixed frequency slope remains.
        let mut visibility = Vec::with_capacity(rows * fft_point_half);
        for row in 0..rows {
            let time_sec = row as f64 * integration_time as f64;
            let temporal_cycles =
                rate_hz as f64 * time_sec + 0.5 * acel_hz_per_s as f64 * time_sec.powi(2);
            for channel in 0..fft_point_half {
                let baseband_frequency_hz = channel as f64 * frequency_step_hz;
                let angle = 2.0
                    * PI
                    * (temporal_cycles
                        + baseband_frequency_hz
                            * (fixed_delay_seconds + temporal_cycles / reference_frequency_hz));
                visibility.push(C32::new(angle.cos() as f32, angle.sin() as f32));
            }
        }

        apply_phase_correction_in_place_at_frequency(
            &mut visibility,
            fft_point_half,
            rate_hz,
            0.0,
            acel_hz_per_s,
            0.0,
            0.0,
            integration_time,
            sampling_speed,
            fft_point,
            0.0,
            reference_frequency_hz,
        );

        for row in visibility.chunks(fft_point_half) {
            for (channel, &actual) in row.iter().enumerate() {
                let baseband_frequency_hz = channel as f64 * frequency_step_hz;
                let angle = 2.0 * PI * baseband_frequency_hz * fixed_delay_seconds;
                let expected = C32::new(angle.cos() as f32, angle.sin() as f32);
                assert!((actual.re - expected.re).abs() < 5.0e-4);
                assert!((actual.im - expected.im).abs() < 5.0e-4);
            }
        }
    }

    #[test]
    fn delay_padding_preserves_integer_delay_samples() {
        let fft_point = 8;
        let rate_bins = 3;
        let mut spectrum = Array2::<C32>::zeros((fft_point / 2, rate_bins));
        for ((row, col), value) in spectrum.indexed_iter_mut() {
            *value = C32::new((row + 2 * col) as f32, (2 * row + col) as f32 * 0.1);
        }

        let unpadded = process_ifft(&spectrum, fft_point as i32, rate_bins);
        let factor = 4usize;
        let padded =
            process_ifft_with_delay_padding(&spectrum, fft_point as i32, rate_bins, factor);

        for rate in 0..rate_bins {
            for delay_idx in 0..fft_point {
                // Axes are [-N/2+1, ..., N/2] and
                // [-N/2+1/f, ..., N/2], respectively.
                let padded_idx = factor * (delay_idx + 1) - 1;
                let expected = unpadded[[rate, delay_idx]];
                let actual = padded[[rate, padded_idx]];
                assert!((expected.re - actual.re).abs() < 1.0e-4);
                assert!((expected.im - actual.im).abs() < 1.0e-4);
            }
        }
    }
}
