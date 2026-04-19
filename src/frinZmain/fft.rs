use ndarray::prelude::*;
use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

type C32 = Complex<f32>;

#[derive(Clone, Copy)]
struct PhaseCorrection {
    rate_hz: f32,
    delay_samples: f32,
    acel_hz: f32,
    effective_integration_length: f32,
    start_time_offset_sec: f32,
}

impl PhaseCorrection {
    fn is_enabled(self) -> bool {
        self.rate_hz != 0.0 || self.delay_samples != 0.0 || self.acel_hz != 0.0
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
    effective_integration_length: f32,
    start_time_offset_sec: f32,
) -> (Array2<C32>, usize) {
    let phase = PhaseCorrection {
        rate_hz: rate_hz_for_correction,
        delay_samples: delay_samples_for_correction,
        acel_hz: acel_hz_for_correction,
        effective_integration_length,
        start_time_offset_sec,
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

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(padding_length);

    let mut freq_rate_array = Array2::<C32>::zeros((fft_point_half, padding_length));
    let mut fft_exe = vec![C32::new(0.0, 0.0); padding_length];

    let phase_factors = phase_correction.and_then(|phase| {
        build_phase_factors(
            phase,
            fft_point_half,
            rows,
            sampling_speed as u32,
            fft_point as u32,
        )
    });

    for i in 1..fft_point_half {
        fft_exe.fill(C32::new(0.0, 0.0));
        let is_rfi_channel = rfi_ranges.iter().any(|(min, max)| i >= *min && i <= *max);

        if !is_rfi_channel {
            for j in 0..rows {
                let mut sample = complex_vec[j * fft_point_half + i];
                if let Some((delay_factors, row_factors)) = &phase_factors {
                    sample *= row_factors[j] * delay_factors[i];
                }
                fft_exe[j] = sample;
            }
        }

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
    fft_point_half: usize,
    rows: usize,
    sampling_speed: u32,
    fft_point: u32,
) -> Option<(Vec<C32>, Vec<C32>)> {
    if !phase.is_valid_for(sampling_speed, fft_point) {
        return None;
    }

    let freq_resolution_hz = sampling_speed as f64 / fft_point as f64;
    let delay_seconds = phase.delay_samples as f64 / sampling_speed as f64;
    let delay_factors = (0..fft_point_half)
        .map(|col| {
            let angle = -2.0 * PI * delay_seconds * col as f64 * freq_resolution_hz;
            C32::new(angle.cos() as f32, angle.sin() as f32)
        })
        .collect();

    let row_factors = (0..rows)
        .map(|row_idx| {
            let time_sec = row_idx as f64 * phase.effective_integration_length as f64
                + phase.start_time_offset_sec as f64;
            let angle = -2.0 * PI * phase.rate_hz as f64 * time_sec
                - PI * phase.acel_hz as f64 * time_sec.powi(2);
            C32::new(angle.cos() as f32, angle.sin() as f32)
        })
        .collect();

    Some((delay_factors, row_factors))
}

pub fn process_ifft(
    freq_rate_array: &Array2<C32>,
    fft_point: i32,
    padding_length: usize,
) -> Array2<C32> {
    let fft_point_usize = fft_point as usize;
    let mut delay_rate_array = Array2::<C32>::zeros((padding_length, fft_point_usize));
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(fft_point_usize);
    let mut ifft_exe = vec![C32::new(0.0, 0.0); fft_point_usize];

    for i in 0..freq_rate_array.dim().1 {
        ifft_exe.fill(C32::new(0.0, 0.0));
        for (dst, src) in ifft_exe.iter_mut().zip(freq_rate_array.column(i).iter()) {
            *dst = *src;
        }

        ifft.process(&mut ifft_exe);

        let (first_half, second_half) = ifft_exe.split_at(fft_point_usize / 2);
        let mut row = delay_rate_array.row_mut(i);
        for (dst, src) in row
            .iter_mut()
            .zip(second_half.iter().chain(first_half.iter()).rev())
        {
            *dst = *src / fft_point_usize as f32;
        }
    }

    delay_rate_array
}

pub fn perform_ifft_on_vec(input: &[C32], ifft_size: usize) -> Vec<C32> {
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(ifft_size);

    let mut ifft_exe = vec![C32::new(0.0, 0.0); ifft_size];
    ifft_exe[..input.len()].copy_from_slice(input);

    ifft.process(&mut ifft_exe);

    let mut shifted_out = vec![C32::new(0.0, 0.0); ifft_size];
    let (first_half, second_half) = ifft_exe.split_at(ifft_size / 2);
    // Support odd-length IFFT sizes by copying with the actual slice lengths.
    shifted_out[..second_half.len()].copy_from_slice(second_half);
    shifted_out[second_half.len()..].copy_from_slice(first_half);

    for val in &mut shifted_out {
        *val /= ifft_size as f32;
    }

    shifted_out.reverse(); // Common reverse operation

    shifted_out
}

pub fn apply_phase_correction_in_place(
    data: &mut [C32],
    fft_point_half: usize,
    rate_hz_for_correction: f32,
    delay_samples_for_correction: f32,
    acel_hz_for_correction: f32,
    effective_integration_length: f32,
    sampling_speed: u32,
    fft_point: u32,
    start_time_offset_sec: f32,
) {
    if data.is_empty() || fft_point_half == 0 || data.len() % fft_point_half != 0 {
        return;
    }

    let rows = data.len() / fft_point_half;
    let phase = PhaseCorrection {
        rate_hz: rate_hz_for_correction,
        delay_samples: delay_samples_for_correction,
        acel_hz: acel_hz_for_correction,
        effective_integration_length,
        start_time_offset_sec,
    };
    let Some((delay_factors, row_factors)) =
        build_phase_factors(phase, fft_point_half, rows, sampling_speed, fft_point)
    else {
        return;
    };

    for (row_idx, row) in data.chunks_mut(fft_point_half).enumerate() {
        for (sample, delay_factor) in row.iter_mut().zip(delay_factors.iter()) {
            *sample *= row_factors[row_idx] * *delay_factor;
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
        apply_phase_correction_in_place(
            &mut corrected,
            fft_point_half,
            0.03,
            0.2,
            0.001,
            0.5,
            32_000_000,
            fft_point as u32,
            0.25,
        );

        let (expected, expected_padding) =
            process_fft(&corrected, rows as i32, fft_point, 32_000_000, &[], 1);
        let (actual, actual_padding) = process_fft_with_phase_correction(
            &input,
            rows as i32,
            fft_point,
            32_000_000,
            &[],
            1,
            0.03,
            0.2,
            0.001,
            0.5,
            0.25,
        );

        assert_eq!(expected_padding, actual_padding);
        assert_eq!(expected.dim(), actual.dim());
        for (expected, actual) in expected.iter().zip(actual.iter()) {
            assert!((expected.re - actual.re).abs() < 1.0e-4);
            assert!((expected.im - actual.im).abs() < 1.0e-4);
        }
    }
}
