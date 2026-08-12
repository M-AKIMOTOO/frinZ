use std::error::Error;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use num_complex::Complex;

use crate::args::Args;
use crate::fft::apply_phase_correction_in_place_at_frequency;
use crate::header::{parse_header, CorHeader};
use crate::input_support::read_input_bytes;
use crate::output::insert_product_before_processing_suffixes;
use crate::plot::{
    plot_spectrum_amplitude_heatmap_with_spikes, plot_spectrum_phase_heatmap_with_spikes,
};
use crate::read::read_visibility_data;
use crate::search;

type C32 = Complex<f32>;

#[derive(Clone, Debug)]
pub struct SpikePeak {
    pub channel: usize,
    pub frequency_mhz: f64,
    pub auto_power: f64,
    pub on_minus_off: f64,
    pub snr: f64,
}

fn output_dir(input_path: &Path) -> PathBuf {
    input_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join("frinZ")
        .join("spike34")
}

pub fn read_all_spectra(path: &Path) -> Result<(CorHeader, Vec<Vec<C32>>, f32), Box<dyn Error>> {
    let buffer = read_input_bytes(path)?;
    let mut cursor = Cursor::new(buffer.as_slice());
    let header = parse_header(&mut cursor)?;
    let mut spectra = Vec::new();
    let mut effective_integ_time = 1.0f32;
    for sector in 0..header.number_of_sector {
        let (complex_vec, _, effective) =
            read_visibility_data(&mut cursor, &header, 1, 0, sector, false, &[])?;
        if complex_vec.is_empty() {
            break;
        }
        effective_integ_time = effective;
        spectra.push(complex_vec);
    }
    Ok((header, spectra, effective_integ_time))
}

fn frequency_axis_mhz(header: &CorHeader, channels: usize) -> Vec<f64> {
    let rbw_mhz = header.sampling_speed as f64 / header.fft_point as f64 / 1.0e6;
    (0..channels)
        .map(|channel| channel as f64 * rbw_mhz)
        .collect()
}

fn moving_average(values: &[f64], width: usize) -> Vec<f64> {
    let width = width.max(3) | 1;
    let half = width / 2;
    let mut out = Vec::with_capacity(values.len());
    for idx in 0..values.len() {
        let start = idx.saturating_sub(half);
        let end = (idx + half + 1).min(values.len());
        let sum: f64 = values[start..end].iter().sum();
        out.push(sum / (end - start) as f64);
    }
    out
}

fn median(mut values: Vec<f64>) -> f64 {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

fn robust_sigma(values: &[f64]) -> f64 {
    let center = median(values.to_vec());
    let mad = median(values.iter().map(|value| (value - center).abs()).collect());
    let sigma = 1.4826 * mad;
    if sigma.is_finite() && sigma > 0.0 {
        sigma
    } else {
        let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len().max(1) as f64;
        variance.sqrt().max(1.0e-30)
    }
}

fn make_spike_peak(
    channel: usize,
    frequency: &[f64],
    auto_mean: &[f64],
    onoff: &[f64],
    center: f64,
    sigma: f64,
) -> SpikePeak {
    SpikePeak {
        channel,
        frequency_mhz: frequency[channel],
        auto_power: auto_mean[channel],
        on_minus_off: onoff[channel],
        snr: (onoff[channel] - center) / sigma,
    }
}

fn accept_peak(selected: &mut Vec<SpikePeak>, peak: SpikePeak, min_sep: usize) {
    if selected
        .iter()
        .all(|existing| existing.channel.abs_diff(peak.channel) >= min_sep)
    {
        selected.push(peak);
    }
}

fn add_matched_filter_spikes(
    selected: &mut Vec<SpikePeak>,
    frequency: &[f64],
    auto_mean: &[f64],
    onoff: &[f64],
    center: f64,
    sigma: f64,
    rbw_mhz: f64,
    min_sep: usize,
) {
    const TRAINING_MHZ: [f64; 11] = [
        14.375, 34.250, 55.750, 78.375, 101.500, 125.125, 148.750, 170.500, 190.750, 211.625,
        233.875,
    ];
    let half_width = 16usize;
    let edge = (half_width / 5).max(2);
    let channels = onoff.len();
    if channels < half_width * 2 + 3 || rbw_mhz <= 0.0 {
        return;
    }
    let mut profiles: Vec<Vec<f64>> = Vec::new();
    let mut training_mask = vec![false; channels];

    for target_mhz in TRAINING_MHZ {
        if target_mhz < frequency[0] || target_mhz > *frequency.last().unwrap_or(&frequency[0]) {
            continue;
        }
        let nominal =
            ((target_mhz / rbw_mhz).round() as isize).clamp(0, channels as isize - 1) as usize;
        let search_lo = nominal.saturating_sub(4);
        let search_hi = (nominal + 5).min(channels);
        let Some((peak, _)) = (search_lo..search_hi)
            .map(|idx| (idx, onoff[idx]))
            .max_by(|a, b| a.1.total_cmp(&b.1))
        else {
            continue;
        };
        if peak < half_width || peak + half_width + 1 > channels {
            continue;
        }
        for value in training_mask
            .iter_mut()
            .take((peak + half_width + 6).min(channels))
            .skip(peak.saturating_sub(half_width + 5))
        {
            *value = true;
        }
        let mut profile = onoff[peak - half_width..=peak + half_width].to_vec();
        let baseline = median(
            profile[..edge]
                .iter()
                .chain(profile[profile.len() - edge..].iter())
                .copied()
                .collect(),
        );
        for value in &mut profile {
            *value -= baseline;
        }
        let amplitude = profile.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if amplitude.is_finite() && amplitude > 0.0 {
            for value in &mut profile {
                *value /= amplitude;
            }
            profiles.push(profile);
        }
    }
    if profiles.is_empty() {
        return;
    }

    let mut template = vec![0.0f64; half_width * 2 + 1];
    for idx in 0..template.len() {
        template[idx] = median(profiles.iter().map(|profile| profile[idx]).collect());
    }
    let baseline = median(
        template[..edge]
            .iter()
            .chain(template[template.len() - edge..].iter())
            .copied()
            .collect(),
    );
    for value in &mut template {
        *value -= baseline;
    }
    let norm = template
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return;
    }
    for value in &mut template {
        *value /= norm;
    }

    let noise_values: Vec<f64> = onoff
        .iter()
        .zip(training_mask)
        .filter_map(|(value, masked)| (!masked).then_some(*value))
        .collect();
    let noise_sigma = robust_sigma(&noise_values).max(1.0e-30);
    let search_start =
        ((256.0 / rbw_mhz).floor() as isize).clamp(0, channels as isize - 1) as usize;
    let search_end = ((384.0 / rbw_mhz).ceil() as isize).clamp(0, channels as isize - 1) as usize;
    let mut matched = Vec::new();
    for channel in search_start.max(half_width)..=search_end.min(channels - half_width - 1) {
        let mut corr = 0.0;
        for offset in 0..template.len() {
            corr += onoff[channel + offset - half_width] * template[template.len() - 1 - offset];
        }
        let score = corr / noise_sigma;
        if score >= 2.0 {
            matched.push((channel, score));
        }
    }
    matched.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut accepted_channels: Vec<usize> = Vec::new();
    for (channel, _score) in matched {
        if accepted_channels
            .iter()
            .all(|existing| existing.abs_diff(channel) >= min_sep)
        {
            accepted_channels.push(channel);
            let refine_lo = channel.saturating_sub(4);
            let refine_hi = (channel + 5).min(channels);
            let refined = (refine_lo..refine_hi)
                .max_by(|a, b| onoff[*a].total_cmp(&onoff[*b]))
                .unwrap_or(channel);
            let peak = make_spike_peak(refined, frequency, auto_mean, onoff, center, sigma);
            if peak.snr >= 1.0 && peak.frequency_mhz <= 384.0 {
                accept_peak(selected, peak, min_sep);
            }
        }
    }
}

pub fn detect_auto_spikes(header: &CorHeader, auto_spectra: &[Vec<C32>]) -> Vec<SpikePeak> {
    if auto_spectra.is_empty() || auto_spectra[0].is_empty() {
        return Vec::new();
    }
    let channels = auto_spectra[0].len();
    let frequency = frequency_axis_mhz(header, channels);
    let mut auto_mean = vec![0.0f64; channels];
    let mut counts = vec![0usize; channels];
    for row in auto_spectra {
        for (channel, value) in row.iter().enumerate().take(channels) {
            let power = value.re as f64;
            if power.is_finite() {
                auto_mean[channel] += power;
                counts[channel] += 1;
            }
        }
    }
    for (value, count) in auto_mean.iter_mut().zip(counts) {
        if count > 0 {
            *value /= count as f64;
        }
    }

    let off = moving_average(&auto_mean, 101.min(channels.saturating_sub(1).max(3)));
    let onoff: Vec<f64> = auto_mean
        .iter()
        .zip(&off)
        .map(|(on, off)| on - off)
        .collect();
    let center = median(onoff.clone());
    let mad = median(onoff.iter().map(|value| (value - center).abs()).collect());
    let sigma = (1.4826 * mad).max(1.0e-30);
    let threshold = center + 2.0 * sigma;
    let rbw_mhz = header.sampling_speed as f64 / header.fft_point as f64 / 1.0e6;
    let min_sep = (8.0 / rbw_mhz).round().max(1.0) as usize;

    let mut candidates = Vec::new();
    for channel in 1..channels.saturating_sub(1) {
        if onoff[channel] > threshold
            && onoff[channel] >= onoff[channel - 1]
            && onoff[channel] >= onoff[channel + 1]
        {
            candidates.push(SpikePeak {
                channel,
                frequency_mhz: frequency[channel],
                auto_power: auto_mean[channel],
                on_minus_off: onoff[channel],
                snr: (onoff[channel] - center) / sigma,
            });
        }
    }
    candidates.sort_by(|a, b| b.snr.total_cmp(&a.snr));
    let mut selected: Vec<SpikePeak> = Vec::new();
    for candidate in candidates {
        accept_peak(&mut selected, candidate, min_sep);
    }
    add_matched_filter_spikes(
        &mut selected,
        &frequency,
        &auto_mean,
        &onoff,
        center,
        sigma,
        rbw_mhz,
        min_sep,
    );
    selected.sort_by(|a, b| a.channel.cmp(&b.channel));
    selected
}

fn extract_subband(
    full: &[C32],
    rows: usize,
    original_half: usize,
    start_chan: usize,
    width_chan: usize,
) -> Vec<C32> {
    let mut out = Vec::with_capacity(rows * width_chan);
    for row in 0..rows {
        let start = row * original_half + start_chan;
        out.extend_from_slice(&full[start..start + width_chan]);
    }
    out
}

fn pad_time_rows_to_power_of_two(data: &mut Vec<C32>, current_rows: i32, row_width: usize) -> i32 {
    if current_rows <= 0 || row_width == 0 {
        return current_rows;
    }
    let target_rows = if current_rows <= 1 {
        1
    } else {
        (current_rows as u32).next_power_of_two() as i32
    };
    if target_rows > current_rows {
        data.resize(
            data.len() + (target_rows - current_rows) as usize * row_width,
            Complex::new(0.0, 0.0),
        );
    }
    target_rows
}

fn write_spike_table(path: &Path, spikes: &[SpikePeak]) -> Result<(), Box<dyn Error>> {
    let mut out = String::new();
    out.push_str("# YAMAGU34 autocorrelation spike candidates\n");
    out.push_str("# channel\tfrequency_MHz\tauto_power\ton_minus_off\tsnr\n");
    for peak in spikes {
        out.push_str(&format!(
            "{}\t{:.9}\t{:.12e}\t{:.12e}\t{:.6}\n",
            peak.channel, peak.frequency_mhz, peak.auto_power, peak.on_minus_off, peak.snr
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

pub fn write_interval_delay_rate_table(
    args: &Args,
    input_path: &Path,
    output_path: &Path,
    spikes: &[SpikePeak],
) -> Result<(), Box<dyn Error>> {
    let buffer = read_input_bytes(input_path)?;
    let mut cursor = Cursor::new(buffer.as_slice());
    let header = parse_header(&mut cursor)?;
    cursor.set_position(0);
    let (_, file_start_time, _) = read_visibility_data(&mut cursor, &header, 1, 0, 0, false, &[])?;
    cursor.set_position(256);

    let original_half = (header.fft_point / 2) as usize;
    let (complex_vec, current_obs_time, effective_integ_time) = read_visibility_data(
        &mut cursor,
        &header,
        header.number_of_sector,
        0,
        0,
        false,
        &[],
    )?;
    let rows = complex_vec.len() / original_half;
    let physical_length = rows as i32;
    let rbw_mhz = header.sampling_speed as f64 / header.fft_point as f64 / 1.0e6;
    let bandpass = None;
    let mut local_args = args.clone();
    local_args.spike34m = None;
    local_args.search = vec!["peak".to_string()];
    local_args.frequency = false;
    local_args.rate_padding = local_args.rate_padding.max(4);

    let mut out = String::new();
    out.push_str("# Delay/rate search for cross-correlation bands between YAMAGU34 auto spikes\n");
    out.push_str("# input\t");
    out.push_str(&input_path.display().to_string());
    out.push('\n');
    out.push_str("# columns: interval left_spike_MHz right_spike_MHz start_channel end_channel center_MHz width_MHz amp_percent snr phase_deg residual_delay_sample residual_rate_Hz\n");

    for (idx, pair) in spikes.windows(2).enumerate() {
        let start_chan = pair[0].channel.saturating_add(1);
        let end_chan = pair[1].channel.saturating_sub(1);
        if end_chan <= start_chan || end_chan >= original_half {
            continue;
        }
        let width_chan = end_chan - start_chan + 1;
        if width_chan < 8 {
            continue;
        }
        let mut subband_vec =
            extract_subband(&complex_vec, rows, original_half, start_chan, width_chan);
        let current_length =
            pad_time_rows_to_power_of_two(&mut subband_vec, physical_length, width_chan);
        let mut sub_header = header.clone();
        sub_header.fft_point = (width_chan * 2) as i32;
        sub_header.sampling_speed = (rbw_mhz * 1.0e6 * width_chan as f64 * 2.0).round() as i32;
        sub_header.observing_frequency += start_chan as f64 * rbw_mhz * 1.0e6;
        let result = search::run_peak_search(
            &subband_vec,
            &sub_header,
            current_length,
            physical_length,
            effective_integ_time,
            &current_obs_time,
            &file_start_time,
            &[],
            &bandpass,
            &local_args,
            sub_header.number_of_sector,
            local_args.cpu,
            None,
        )?;
        let analysis = result.analysis_results;
        let left = pair[0].frequency_mhz;
        let right = pair[1].frequency_mhz;
        let center = (start_chan + end_chan) as f64 * 0.5 * rbw_mhz;
        let width = width_chan as f64 * rbw_mhz;
        out.push_str(&format!(
            "{}\t{:.9}\t{:.9}\t{}\t{}\t{:.9}\t{:.9}\t{:.6}\t{:.3}\t{:.3}\t{:.8}\t{:.8}\n",
            idx + 1,
            left,
            right,
            start_chan,
            end_chan,
            center,
            width,
            analysis.delay_max_amp * 100.0,
            analysis.delay_snr,
            analysis.delay_phase,
            analysis.residual_delay,
            analysis.residual_rate
        ));
    }
    fs::write(output_path, out)?;
    Ok(())
}

fn weighted_line_fit(x: &[f64], y: &[f64], weight: &[f64]) -> Option<(f64, f64)> {
    if x.len() != y.len() || x.len() != weight.len() || x.len() < 2 {
        return None;
    }
    let sw: f64 = weight.iter().sum();
    if !sw.is_finite() || sw <= 0.0 {
        return None;
    }
    let xbar = x.iter().zip(weight).map(|(x, w)| x * w).sum::<f64>() / sw;
    let ybar = y.iter().zip(weight).map(|(y, w)| y * w).sum::<f64>() / sw;
    let sxx = x
        .iter()
        .zip(weight)
        .map(|(x, w)| w * (x - xbar) * (x - xbar))
        .sum::<f64>();
    if !sxx.is_finite() || sxx <= 0.0 {
        return None;
    }
    let sxy = x
        .iter()
        .zip(y)
        .zip(weight)
        .map(|((x, y), w)| w * (x - xbar) * (y - ybar))
        .sum::<f64>();
    let slope = sxy / sxx;
    let intercept = ybar - slope * xbar;
    Some((slope, intercept))
}

fn unwrap_series(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(values.len());
    let mut offset = 0.0;
    let mut prev = values[0];
    out.push(prev);
    for &value in &values[1..] {
        let delta = value - prev;
        if delta > std::f64::consts::PI {
            offset -= 2.0 * std::f64::consts::PI;
        } else if delta < -std::f64::consts::PI {
            offset += 2.0 * std::f64::consts::PI;
        }
        out.push(value + offset);
        prev = value;
    }
    out
}

fn interpolate_valid(values: &[f64], frequency: &[f64], valid: &[bool]) -> Vec<f64> {
    let idx: Vec<usize> = (0..values.len())
        .filter(|&i| valid[i] && values[i].is_finite())
        .collect();
    if idx.is_empty() {
        return vec![0.0; values.len()];
    }
    let mut out = vec![0.0; values.len()];
    for i in 0..values.len() {
        if frequency[i] <= frequency[idx[0]] {
            out[i] = values[idx[0]];
        } else if frequency[i] >= frequency[*idx.last().unwrap()] {
            out[i] = values[*idx.last().unwrap()];
        } else {
            let upper_pos = idx.partition_point(|&j| frequency[j] < frequency[i]);
            let j0 = idx[upper_pos - 1];
            let j1 = idx[upper_pos];
            let frac = (frequency[i] - frequency[j0]) / (frequency[j1] - frequency[j0]);
            out[i] = values[j0] * (1.0 - frac) + values[j1] * frac;
        }
    }
    out
}

fn moving_median(values: &[f64], width: usize) -> Vec<f64> {
    let width = (width.max(3)) | 1;
    let half = width / 2;
    let mut out = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        // Match the reference implementation: pad the ends with the nearest
        // valid value before taking the odd-width median.
        let window: Vec<f64> = (0..width)
            .map(|offset| {
                let source = (i as isize + offset as isize - half as isize)
                    .clamp(0, values.len().saturating_sub(1) as isize)
                    as usize;
                values[source]
            })
            .collect();
        out.push(median(window));
    }
    out
}

fn robust_frequency_line_fit(
    frequency: &[f64],
    phase: &[f64],
    amplitude: &[f64],
) -> Option<(f64, f64, f64)> {
    if frequency.len() < 2 || frequency.len() != phase.len() || frequency.len() != amplitude.len() {
        return None;
    }
    let base: Vec<f64> = amplitude.iter().map(|value| value * value).collect();
    let sum_base: f64 = base.iter().sum();
    if !sum_base.is_finite() || sum_base <= 0.0 {
        return None;
    }
    let xmid = frequency
        .iter()
        .zip(&base)
        .map(|(value, weight)| value * weight)
        .sum::<f64>()
        / sum_base;
    let centered: Vec<f64> = frequency.iter().map(|value| value - xmid).collect();
    let mut robust = vec![1.0f64; phase.len()];
    let mut slope = 0.0;
    let mut intercept = 0.0;
    for _ in 0..10 {
        let weights: Vec<f64> = base.iter().zip(&robust).map(|(a, r)| a * r).collect();
        let Some((fit_slope, fit_intercept)) = weighted_line_fit(&centered, phase, &weights) else {
            return None;
        };
        slope = fit_slope;
        intercept = fit_intercept;
        let residual: Vec<f64> = phase
            .iter()
            .zip(&centered)
            .map(|(value, x)| value - (slope * x + intercept))
            .collect();
        let center = median(residual.clone());
        let mad = median(
            residual
                .iter()
                .map(|value| (value - center).abs())
                .collect(),
        );
        let scale = (1.4826 * mad).max(0.5_f64.to_radians());
        robust = residual
            .iter()
            .map(|value| {
                let normalized = (value - center).abs() / (1.5 * scale);
                (1.0 / normalized.max(1.0)).min(1.0)
            })
            .collect();
    }
    Some((slope, intercept, xmid))
}

fn pad_spectra_for_search(flat: &mut Vec<C32>, rows: usize, cols: usize) -> i32 {
    if rows == 0 || cols == 0 {
        return rows as i32;
    }
    let target = rows.next_power_of_two();
    if target > rows {
        flat.extend(std::iter::repeat(C32::new(0.0, 0.0)).take((target - rows) * cols));
    }
    target as i32
}

fn apply_global_and_safe_spike_correction(
    header: &CorHeader,
    spectra: &[Vec<C32>],
    effective_integ_time: f32,
    spikes: &[SpikePeak],
    args: &Args,
    current_obs_time: &chrono::DateTime<chrono::Utc>,
    file_start_time: &chrono::DateTime<chrono::Utc>,
) -> Result<Vec<Vec<C32>>, Box<dyn Error>> {
    if spectra.is_empty() || spectra[0].is_empty() {
        return Ok(spectra.to_vec());
    }
    let cols = spectra[0].len();
    let rows = spectra.len();
    let mut search_vec: Vec<C32> = spectra.iter().flat_map(|row| row.iter().copied()).collect();
    let current_length = pad_spectra_for_search(&mut search_vec, rows, cols);
    let mut local_args = args.clone();
    local_args.spike34m = None;
    local_args.search = vec!["peak".to_string()];
    local_args.frequency = false;
    // Force the final FFT evaluation so the phase convention matches the
    // visible full-band search result.
    local_args.spectrum = true;
    local_args.plot = false;
    local_args.raw_visibility = false;
    local_args.delay_correct = 0.0;
    local_args.rate_correct = 0.0;
    local_args.acel_correct = 0.0;
    local_args.jerk_correct = 0.0;
    local_args.snap_correct = 0.0;
    local_args.rate_padding = local_args.rate_padding.max(4);
    let result = search::run_peak_search(
        &search_vec,
        header,
        current_length,
        rows as i32,
        effective_integ_time,
        current_obs_time,
        file_start_time,
        &[],
        &None,
        &local_args,
        header.number_of_sector,
        local_args.cpu,
        None,
    )?;
    let delay = result.analysis_results.residual_delay;
    let rate = if rows > 1 {
        result.analysis_results.residual_rate
    } else {
        0.0
    };
    let mut flat: Vec<C32> = spectra.iter().flat_map(|row| row.iter().copied()).collect();
    apply_phase_correction_in_place_at_frequency(
        &mut flat,
        cols,
        rate,
        delay,
        0.0,
        0.0,
        0.0,
        effective_integ_time,
        header.sampling_speed as u32,
        header.fft_point as u32,
        0.0,
        header.observing_frequency,
    );
    let globally_corrected: Vec<Vec<C32>> = flat.chunks(cols).map(|row| row.to_vec()).collect();
    Ok(apply_safe_spike_residual_correction(
        header,
        &globally_corrected,
        effective_integ_time,
        spikes,
    ))
}

fn coherent_phase(spectra: &[Vec<C32>]) -> Option<f64> {
    let sum = spectra
        .iter()
        .flat_map(|row| row.iter())
        .filter(|value| value.re.is_finite() && value.im.is_finite())
        .copied()
        .fold(C32::new(0.0, 0.0), |acc, value| acc + value);
    (sum.norm() > 0.0).then(|| sum.arg() as f64)
}

fn edge_taper(frequency_mhz: &[f64], last_spike_mhz: f64, width_mhz: f64) -> Vec<f64> {
    let start = last_spike_mhz - width_mhz;
    let end = last_spike_mhz + width_mhz;
    frequency_mhz
        .iter()
        .map(|&f| {
            if f < start {
                1.0
            } else if f >= end {
                0.0
            } else {
                0.5 * (1.0 + (std::f64::consts::PI * (f - start) / (2.0 * width_mhz)).cos())
            }
        })
        .collect()
}

pub fn apply_safe_spike_residual_correction(
    header: &CorHeader,
    spectra: &[Vec<C32>],
    effective_integ_time: f32,
    spikes: &[SpikePeak],
) -> Vec<Vec<C32>> {
    if spectra.is_empty() || spectra[0].is_empty() || spikes.is_empty() {
        return spectra.to_vec();
    }
    let rows = spectra.len();
    let cols = spectra[0].len();
    let frequency = frequency_axis_mhz(header, cols);
    let elapsed: Vec<f64> = (0..rows)
        .map(|row| row as f64 * effective_integ_time as f64)
        .collect();
    let mut phase0 = vec![f64::NAN; cols];
    let mut rate_hz = vec![f64::NAN; cols];
    let mut median_amp = vec![0.0; cols];

    for ch in 0..cols {
        let mut phases = Vec::new();
        let mut times = Vec::new();
        let mut weights = Vec::new();
        let mut amps = Vec::new();
        for row in 0..rows {
            let v = spectra[row][ch];
            let amp = v.norm() as f64;
            if amp.is_finite() && amp > 0.0 {
                phases.push(v.arg() as f64);
                times.push(elapsed[row]);
                weights.push(amp * amp);
                amps.push(amp);
            }
        }
        if phases.len() < 8 {
            continue;
        }
        median_amp[ch] = median(amps);
        let unwrapped = unwrap_series(&phases);
        if let Some((slope, intercept)) = weighted_line_fit(&times, &unwrapped, &weights) {
            phase0[ch] = intercept;
            rate_hz[ch] = slope / (2.0 * std::f64::consts::PI);
        }
    }

    let reliable: Vec<bool> = (0..cols)
        .map(|i| phase0[i].is_finite() && rate_hz[i].is_finite() && median_amp[i] >= 8.0e-7)
        .collect();
    if reliable.iter().filter(|&&ok| ok).count() < 50 {
        return spectra.to_vec();
    }
    let reliable_idx: Vec<usize> = (0..cols).filter(|&i| reliable[i]).collect();
    let reliable_phase_wrapped: Vec<f64> = reliable_idx.iter().map(|&i| phase0[i]).collect();
    let reliable_phase_unwrapped = unwrap_series(&reliable_phase_wrapped);
    let x: Vec<f64> = reliable_idx.iter().map(|&i| frequency[i]).collect();
    let amplitude: Vec<f64> = reliable_idx.iter().map(|&i| median_amp[i]).collect();
    let Some((trend_slope, trend_intercept, trend_xmid)) =
        robust_frequency_line_fit(&x, &reliable_phase_unwrapped, &amplitude)
    else {
        return spectra.to_vec();
    };
    let trend: Vec<f64> = frequency
        .iter()
        .map(|&f| trend_slope * (f - trend_xmid) + trend_intercept)
        .collect();
    let mut phase_residual = vec![f64::NAN; cols];
    for (pos, &idx) in reliable_idx.iter().enumerate() {
        phase_residual[idx] = reliable_phase_unwrapped[pos] - trend[idx];
    }
    let phase_residual = moving_median(
        &interpolate_valid(&phase_residual, &frequency, &reliable),
        41,
    );

    let common_rate = median(reliable_idx.iter().map(|&i| rate_hz[i]).collect());
    let mut rate_residual = vec![f64::NAN; cols];
    for &idx in &reliable_idx {
        rate_residual[idx] = rate_hz[idx] - common_rate;
    }
    let rate_residual = moving_median(
        &interpolate_valid(&rate_residual, &frequency, &reliable),
        41,
    );
    let last_spike = spikes
        .iter()
        .map(|peak| peak.frequency_mhz)
        .fold(f64::NEG_INFINITY, f64::max);
    let taper = edge_taper(&frequency, last_spike, 5.0);
    // Keep the absolute full-band phase established by the global fringe
    // solution. The spike-only residual is a shape correction, so remove its
    // amplitude-weighted constant phase component before applying it.
    let mut phase_correction: Vec<f64> = phase_residual
        .iter()
        .zip(&taper)
        .map(|(phase, edge)| phase * edge)
        .collect();
    let phase_weight: Vec<f64> = median_amp
        .iter()
        .zip(&taper)
        .map(|(amp, edge)| amp * amp * edge)
        .collect();
    let phase_weight_sum: f64 = phase_weight.iter().sum();
    if phase_weight_sum.is_finite() && phase_weight_sum > 0.0 {
        let phase_mean = phase_correction
            .iter()
            .zip(&phase_weight)
            .map(|(phase, weight)| phase * weight)
            .sum::<f64>()
            / phase_weight_sum;
        for phase in &mut phase_correction {
            *phase -= phase_mean;
        }
    }

    let reference_phase = coherent_phase(spectra);
    let mut corrected = spectra.to_vec();
    for row in 0..rows {
        let t = elapsed[row];
        for ch in 0..cols {
            let angle = phase_correction[ch]
                + 2.0 * std::f64::consts::PI * rate_residual[ch] * t * taper[ch];
            let factor = Complex::new((-angle).cos() as f32, (-angle).sin() as f32);
            corrected[row][ch] *= factor;
        }
    }
    // Preserve the phase reference delivered by the full-band correction.
    // Spike residual fitting may improve coherence while introducing a
    // harmless constant rotation; remove that rotation from the final data.
    if let (Some(before), Some(after)) = (reference_phase, coherent_phase(&corrected)) {
        let angle = before - after;
        let factor = Complex::new(angle.cos() as f32, angle.sin() as f32);
        for row in &mut corrected {
            for value in row {
                *value *= factor;
            }
        }
    }
    corrected
}

pub fn run_spike34m_analysis(args: &Args) -> Result<(), Box<dyn Error>> {
    let Some(spike_path) = args.spike34m.as_ref() else {
        return Ok(());
    };
    let input_path = args
        .input
        .as_ref()
        .ok_or("--spike34 requires --input CROSS.cor")?;
    let out_dir = output_dir(input_path);
    fs::create_dir_all(&out_dir)?;

    let (auto_header, auto_spectra, _) = read_all_spectra(spike_path)?;
    let spikes = detect_auto_spikes(&auto_header, &auto_spectra);
    if spikes.len() < 2 {
        return Err("--spike34 found fewer than two YAMAGU34 auto-correlation spikes".into());
    }

    let base = input_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("spike34m");
    let stem = insert_product_before_processing_suffixes(base, "spike34");
    let spike_table = out_dir.join(format!("{stem}_spikes.tsv"));
    let delay_rate_table = out_dir.join(format!("{stem}_delay_rate.tsv"));
    write_spike_table(&spike_table, &spikes)?;
    write_interval_delay_rate_table(args, input_path, &delay_rate_table, &spikes)?;

    let (input_header, cross_spectra, cross_effective_integ_time) = read_all_spectra(input_path)?;
    let spike_channels: Vec<usize> = spikes.iter().map(|peak| peak.channel).collect();
    let amp_png = out_dir.join(format!("{stem}_raw_visibility_amp.png"));
    let phase_png = out_dir.join(format!("{stem}_raw_visibility_phase.png"));
    let corrected_amp_png = out_dir.join(format!("{stem}_raw_visibility_corrected_amp.png"));
    let corrected_phase_png = out_dir.join(format!("{stem}_raw_visibility_corrected_phase.png"));
    let buffer = read_input_bytes(input_path)?;
    let mut time_cursor = Cursor::new(buffer.as_slice());
    let _time_header = parse_header(&mut time_cursor)?;
    time_cursor.set_position(0);
    let (_, file_start_time, _) =
        read_visibility_data(&mut time_cursor, &input_header, 1, 0, 0, false, &[])?;
    time_cursor.set_position(256);
    let (_, current_obs_time, _) = read_visibility_data(
        &mut time_cursor,
        &input_header,
        input_header.number_of_sector,
        0,
        0,
        false,
        &[],
    )?;
    let corrected_spectra = apply_global_and_safe_spike_correction(
        &input_header,
        &cross_spectra,
        cross_effective_integ_time,
        &spikes,
        args,
        &current_obs_time,
        &file_start_time,
    )?;
    plot_spectrum_amplitude_heatmap_with_spikes(&amp_png, &cross_spectra, 0.0, &spike_channels)?;
    plot_spectrum_phase_heatmap_with_spikes(&phase_png, &cross_spectra, 0.0, &spike_channels)?;
    plot_spectrum_amplitude_heatmap_with_spikes(
        &corrected_amp_png,
        &corrected_spectra,
        0.0,
        &spike_channels,
    )?;
    plot_spectrum_phase_heatmap_with_spikes(
        &corrected_phase_png,
        &corrected_spectra,
        0.0,
        &spike_channels,
    )?;

    println!("Spike34m output directory: {}", out_dir.display());
    println!("Spike table: {}", spike_table.display());
    println!("Spike delay/rate table: {}", delay_rate_table.display());
    Ok(())
}
