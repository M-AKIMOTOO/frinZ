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
    plot_spike34_fit_residual, plot_spike34_frequency_spectrum_with_phase,
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

#[derive(Clone, Debug)]
pub struct SpikeIntervalCorrection {
    pub left_spike_mhz: f64,
    pub right_spike_mhz: f64,
    pub start_channel: usize,
    pub end_channel: usize,
    pub center_mhz: f64,
    pub width_mhz: f64,
    pub amp_percent: f64,
    pub snr: f64,
    pub phase_deg: f64,
    pub delay_sample: f32,
    pub rate_hz: f32,
}

fn spike_interval_ranges(
    header: &CorHeader,
    spikes: &[SpikePeak],
    cols: usize,
) -> Vec<(f64, f64, usize, usize)> {
    if cols == 0 || spikes.is_empty() {
        return Vec::new();
    }
    let frequency = frequency_axis_mhz(header, cols);
    let mut ranges = Vec::new();
    let first = spikes[0].channel.min(cols.saturating_sub(1));
    if first > 0 {
        ranges.push((frequency[0], spikes[0].frequency_mhz, 0, first - 1));
    }
    for pair in spikes.windows(2) {
        let start = pair[0].channel.saturating_add(1);
        let end = pair[1]
            .channel
            .saturating_sub(1)
            .min(cols.saturating_sub(1));
        if end > start {
            ranges.push((pair[0].frequency_mhz, pair[1].frequency_mhz, start, end));
        }
    }
    let last = spikes.last().unwrap().channel.min(cols.saturating_sub(1));
    if last + 1 < cols {
        ranges.push((
            spikes.last().unwrap().frequency_mhz,
            frequency[cols - 1],
            last + 1,
            cols - 1,
        ));
    }
    ranges
}

pub fn write_interval_delay_rate_table(
    args: &Args,
    input_path: &Path,
    output_path: &Path,
    spikes: &[SpikePeak],
) -> Result<Vec<SpikeIntervalCorrection>, Box<dyn Error>> {
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

    let mut records = Vec::new();
    for (left_mhz, right_mhz, start_chan, end_chan) in
        spike_interval_ranges(&header, spikes, original_half)
    {
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
        let left = left_mhz;
        let right = right_mhz;
        let center = (start_chan + end_chan) as f64 * 0.5 * rbw_mhz;
        let width = width_chan as f64 * rbw_mhz;
        records.push(SpikeIntervalCorrection {
            left_spike_mhz: left,
            right_spike_mhz: right,
            start_channel: start_chan,
            end_channel: end_chan,
            center_mhz: center,
            width_mhz: width,
            amp_percent: (analysis.delay_max_amp * 100.0) as f64,
            snr: analysis.delay_snr as f64,
            phase_deg: analysis.delay_phase as f64,
            delay_sample: analysis.residual_delay,
            rate_hz: analysis.residual_rate,
        });
    }

    let mut out = String::from(
        "interval\tleft_spike_MHz\tright_spike_MHz\tstart_channel\tend_channel\tcenter_MHz\twidth_MHz\tamp_percent\tsnr\tphase_deg\tresidual_delay_sample\tresidual_rate_Hz\n",
    );
    for (idx, record) in records.iter().enumerate() {
        out.push_str(&format!(
            "{}\t{:.9}\t{:.9}\t{}\t{}\t{:.9}\t{:.9}\t{:.6}\t{:.3}\t{:.3}\t{:.8}\t{:.8}\n",
            idx + 1,
            record.left_spike_mhz,
            record.right_spike_mhz,
            record.start_channel,
            record.end_channel,
            record.center_mhz,
            record.width_mhz,
            record.amp_percent,
            record.snr,
            record.phase_deg,
            record.delay_sample,
            record.rate_hz,
        ));
    }
    fs::write(output_path, out)?;
    Ok(records)
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

fn wrap_phase(value: f64) -> f64 {
    (value + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
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

fn circular_frequency_line_fit(
    frequency: &[f64],
    phase: &[f64],
    weights: &[f64],
) -> Option<(f64, f64)> {
    if frequency.len() < 2 || frequency.len() != phase.len() || frequency.len() != weights.len() {
        return None;
    }
    let total_weight: f64 = weights.iter().sum();
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return None;
    }
    // Search the residual-delay slope on the circle first.  The search
    // correction has already removed the large geometric delay, so this wide
    // range covers the remaining fringe slopes while avoiding any 2π branch
    // assumption in the initial fit.
    let mut best_score = f64::NEG_INFINITY;
    let mut best_slope = 0.0;
    let mut best_intercept = 0.0;
    let slope_min = -0.1;
    let slope_max = 0.1;
    let step = 0.00025;
    let mut slope = slope_min;
    while slope <= slope_max {
        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;
        for ((&f, &p), &weight) in frequency.iter().zip(phase).zip(weights) {
            let angle = p - slope * f;
            sum_cos += weight * angle.cos();
            sum_sin += weight * angle.sin();
        }
        let score = (sum_cos * sum_cos + sum_sin * sum_sin).sqrt();
        if score > best_score {
            best_score = score;
            best_slope = slope;
            best_intercept = sum_sin.atan2(sum_cos);
        }
        slope += step;
    }
    Some((best_slope, best_intercept))
}

fn align_phase_to_line(phase: f64, frequency: f64, slope: f64, intercept: f64) -> f64 {
    let expected = slope * frequency + intercept;
    phase + 2.0 * std::f64::consts::PI * ((expected - phase) / (2.0 * std::f64::consts::PI)).round()
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
    intervals: &[SpikeIntervalCorrection],
    args: &Args,
    current_obs_time: &chrono::DateTime<chrono::Utc>,
    file_start_time: &chrono::DateTime<chrono::Utc>,
) -> Result<(Vec<Vec<C32>>, Vec<Vec<C32>>, f32, f32), Box<dyn Error>> {
    if spectra.is_empty() || spectra[0].is_empty() {
        return Ok((spectra.to_vec(), spectra.to_vec(), 0.0, 0.0));
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
    // Keep the measured time/rate fringe term visible in --spike34
    // diagnostics; only the common delay is removed here.
    apply_phase_correction_in_place_at_frequency(
        &mut flat,
        cols,
        0.0,
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
    let corrected = apply_interval_delay_rate_correction(
        header,
        &globally_corrected,
        effective_integ_time,
        spikes,
        intervals,
    );
    Ok((globally_corrected, corrected, delay, rate))
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

#[derive(Clone, Debug)]
pub struct SpikeFitDiagnostics {
    pub frequency_mhz: Vec<f64>,
    /// Phase intercept from the per-channel time fit, before frequency unwrap.
    pub phase0_rad: Vec<f64>,
    /// Unwrapped phase intercept for plotting; low-amplitude channels may be
    /// present here even though they are excluded from the robust fit.
    pub phase0_unwrapped_rad: Vec<f64>,
    pub global_phase_fit_rad: Vec<f64>,
    /// Piecewise linear phase fit within each adjacent-spike interval.
    pub interval_phase_fit_rad: Vec<f64>,
    /// Raw residual: unwrapped phase0 - full-band global linear fit.
    pub raw_phase_residual_rad: Vec<f64>,
    /// Residual left after the interval delay fit.
    pub final_phase_residual_rad: Vec<f64>,
    pub phase_correction_rad: Vec<f64>,
    pub rate_hz: Vec<f64>,
    /// Raw rate residual: rate - reliable-channel median rate.
    pub raw_rate_residual_hz: Vec<f64>,
    pub rate_residual_hz: Vec<f64>,
    pub taper: Vec<f64>,
    pub reliable: Vec<bool>,
}

pub fn estimate_spike_fit_diagnostics(
    header: &CorHeader,
    spectra: &[Vec<C32>],
    effective_integ_time: f32,
    spikes: &[SpikePeak],
    intervals: &[SpikeIntervalCorrection],
) -> Option<SpikeFitDiagnostics> {
    if spectra.is_empty() || spectra[0].is_empty() || spikes.is_empty() {
        return None;
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
            let value = spectra[row][ch];
            let amp = value.norm() as f64;
            if amp.is_finite() && amp > 0.0 {
                phases.push(value.arg() as f64);
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
            // The time-domain phase unwrap chooses an arbitrary 2π branch for
            // each channel. Keep the physically meaningful principal phase
            // before doing the frequency-domain unwrap; otherwise a harmless
            // branch choice becomes a false ±360° spectral jump.
            phase0[ch] = wrap_phase(intercept);
            rate_hz[ch] = slope / (2.0 * std::f64::consts::PI);
        }
    }

    let reliable: Vec<bool> = (0..cols)
        .map(|i| phase0[i].is_finite() && rate_hz[i].is_finite() && median_amp[i] >= 2.0e-7)
        .collect();
    if reliable.iter().filter(|&&ok| ok).count() < 50 {
        return None;
    }
    let reliable_idx: Vec<usize> = (0..cols).filter(|&i| reliable[i]).collect();
    let x: Vec<f64> = reliable_idx.iter().map(|&i| frequency[i]).collect();
    let phase: Vec<f64> = reliable_idx.iter().map(|&i| phase0[i]).collect();
    let amplitude: Vec<f64> = reliable_idx.iter().map(|&i| median_amp[i]).collect();
    let amplitude_weights: Vec<f64> = amplitude.iter().map(|value| value * value).collect();
    let (seed_slope, seed_intercept) = circular_frequency_line_fit(&x, &phase, &amplitude_weights)?;
    let reliable_phase_aligned: Vec<f64> = x
        .iter()
        .zip(&phase)
        .map(|(&f, &p)| align_phase_to_line(p, f, seed_slope, seed_intercept))
        .collect();
    let (trend_slope, trend_intercept, trend_xmid) =
        robust_frequency_line_fit(&x, &reliable_phase_aligned, &amplitude)?;
    let global_phase_fit_rad: Vec<f64> = frequency
        .iter()
        .map(|&f| trend_slope * (f - trend_xmid) + trend_intercept)
        .collect();

    // Every finite channel is put on the branch nearest the global fit. This
    // keeps the low-frequency portion visible in the diagnostic plot without
    // allowing weak-channel time-fit branches to move the whole unwrap by 2π.
    let all_phase_idx: Vec<usize> = (0..cols).filter(|&idx| phase0[idx].is_finite()).collect();
    let mut phase0_unwrapped = vec![f64::NAN; cols];
    for &idx in &all_phase_idx {
        phase0_unwrapped[idx] = align_phase_to_line(
            phase0[idx],
            frequency[idx],
            trend_slope,
            trend_intercept - trend_slope * trend_xmid,
        );
    }
    let mut raw_phase_residual = vec![f64::NAN; cols];
    for idx in 0..cols {
        if phase0_unwrapped[idx].is_finite() {
            raw_phase_residual[idx] = phase0_unwrapped[idx] - global_phase_fit_rad[idx];
        }
    }
    // Outside an interval there is no spike-specific correction; show the
    // global trend itself there and retain the corresponding residual.
    let mut interval_phase_fit_rad = global_phase_fit_rad.clone();
    for interval in intervals {
        let start = interval.start_channel.min(cols.saturating_sub(1));
        let end = interval.end_channel.min(cols.saturating_sub(1));
        if end <= start {
            continue;
        }
        let indexes: Vec<usize> = (start..=end)
            .filter(|&idx| reliable[idx] && raw_phase_residual[idx].is_finite())
            .collect();
        if indexes.len() < 2 {
            continue;
        }
        let x: Vec<f64> = indexes.iter().map(|&idx| frequency[idx]).collect();
        let y: Vec<f64> = indexes.iter().map(|&idx| raw_phase_residual[idx]).collect();
        let amplitudes: Vec<f64> = indexes.iter().map(|&idx| median_amp[idx]).collect();
        if let Some((slope, intercept, xmid)) = robust_frequency_line_fit(&x, &y, &amplitudes) {
            // Evaluate the fitted residual over the whole interval.  The
            // correction is applied to every time/frequency cell, not only to
            // channels that happened to pass the fit reliability threshold.
            for idx in start..=end {
                interval_phase_fit_rad[idx] =
                    global_phase_fit_rad[idx] + slope * (frequency[idx] - xmid) + intercept;
            }
        }
    }
    let mut final_phase_residual = vec![f64::NAN; cols];
    for idx in 0..cols {
        if phase0_unwrapped[idx].is_finite() && interval_phase_fit_rad[idx].is_finite() {
            final_phase_residual[idx] = phase0_unwrapped[idx] - interval_phase_fit_rad[idx];
        }
    }
    let phase_residual_rad = moving_median(
        &interpolate_valid(&raw_phase_residual, &frequency, &reliable),
        41,
    );

    let common_rate = median(reliable_idx.iter().map(|&i| rate_hz[i]).collect());
    let mut raw_rate_residual = vec![f64::NAN; cols];
    for &idx in &reliable_idx {
        raw_rate_residual[idx] = rate_hz[idx] - common_rate;
    }
    let interpolated_rate_residual = interpolate_valid(&raw_rate_residual, &frequency, &reliable);
    let rate_residual_hz = moving_median(&interpolated_rate_residual, 41);
    let last_spike = spikes
        .iter()
        .map(|peak| peak.frequency_mhz)
        .fold(f64::NEG_INFINITY, f64::max);
    let taper = edge_taper(&frequency, last_spike, 5.0);

    let mut phase_correction_rad: Vec<f64> = phase_residual_rad
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
        let phase_mean = phase_correction_rad
            .iter()
            .zip(&phase_weight)
            .map(|(phase, weight)| phase * weight)
            .sum::<f64>()
            / phase_weight_sum;
        for phase in &mut phase_correction_rad {
            *phase -= phase_mean;
        }
    }

    Some(SpikeFitDiagnostics {
        frequency_mhz: frequency,
        phase0_rad: phase0,
        phase0_unwrapped_rad: phase0_unwrapped,
        global_phase_fit_rad,
        interval_phase_fit_rad,
        raw_phase_residual_rad: raw_phase_residual,
        final_phase_residual_rad: final_phase_residual,
        phase_correction_rad,
        rate_hz,
        raw_rate_residual_hz: raw_rate_residual,
        rate_residual_hz,
        taper,
        reliable,
    })
}

fn apply_interval_delay_rate_correction(
    _header: &CorHeader,
    spectra: &[Vec<C32>],
    effective_integ_time: f32,
    spikes: &[SpikePeak],
    intervals: &[SpikeIntervalCorrection],
) -> Vec<Vec<C32>> {
    if spectra.is_empty() || spectra[0].is_empty() || intervals.is_empty() {
        return spectra.to_vec();
    }
    let Some(diagnostics) =
        estimate_spike_fit_diagnostics(_header, spectra, effective_integ_time, spikes, intervals)
    else {
        return spectra.to_vec();
    };
    let rows = spectra.len();
    let cols = spectra[0].len();
    let mut corrected_flat: Vec<C32> = spectra.iter().flat_map(|row| row.iter().copied()).collect();

    // The full-band search has supplied the common solution.  Only the
    // interval residual frequency-dependent phase offset
    // (delay).  The residual time-rate term is deliberately not applied so
    // the YAMAGU34 spike-induced rate splitting remains visible.  Apply the
    // delay model directly to every visibility cell.  A sub-band FFT
    // correction followed by restoring the interval mean would cancel the
    // phase jump at a YAMAGU34 spike that this operation is meant to remove.
    for interval in intervals {
        let start = interval.start_channel;
        let end = interval.end_channel;
        if start >= cols || end < start || end >= cols {
            continue;
        }
        if end - start + 1 < 8 {
            continue;
        }
        let indexes: Vec<usize> = (start..=end)
            .filter(|&idx| {
                diagnostics.interval_phase_fit_rad[idx].is_finite()
                    && diagnostics.global_phase_fit_rad[idx].is_finite()
                    && diagnostics.rate_hz[idx].is_finite()
                    && diagnostics.reliable[idx]
            })
            .collect();
        if indexes.len() < 2 {
            continue;
        }
        for row in 0..rows {
            for ch in start..=end {
                let phase_offset =
                    diagnostics.interval_phase_fit_rad[ch] - diagnostics.global_phase_fit_rad[ch];
                if !phase_offset.is_finite() {
                    continue;
                }
                // No rate term is applied here.  The residual time slope is
                // intentionally retained for observing the spike effect.
                let phase = phase_offset;
                let factor = C32::new((-phase).cos() as f32, (-phase).sin() as f32);
                corrected_flat[row * cols + ch] *= factor;
            }
        }
    }

    corrected_flat
        .chunks(cols)
        .map(|row| row.to_vec())
        .collect()
}

pub fn apply_spike_interval_residual_correction(
    header: &CorHeader,
    spectra: &[Vec<C32>],
    effective_integ_time: f32,
    spikes: &[SpikePeak],
) -> Vec<Vec<C32>> {
    let intervals: Vec<SpikeIntervalCorrection> =
        spike_interval_ranges(header, spikes, spectra[0].len())
            .into_iter()
            .map(
                |(left_spike_mhz, right_spike_mhz, start_channel, end_channel)| {
                    SpikeIntervalCorrection {
                        left_spike_mhz,
                        right_spike_mhz,
                        start_channel,
                        end_channel,
                        center_mhz: 0.0,
                        width_mhz: 0.0,
                        amp_percent: 0.0,
                        snr: 0.0,
                        phase_deg: 0.0,
                        delay_sample: 0.0,
                        rate_hz: 0.0,
                    }
                },
            )
            .collect();
    apply_interval_delay_rate_correction(header, spectra, effective_integ_time, spikes, &intervals)
}

pub fn apply_safe_spike_residual_correction(
    header: &CorHeader,
    spectra: &[Vec<C32>],
    effective_integ_time: f32,
    spikes: &[SpikePeak],
) -> Vec<Vec<C32>> {
    let Some(diagnostics) =
        estimate_spike_fit_diagnostics(header, spectra, effective_integ_time, spikes, &[])
    else {
        return spectra.to_vec();
    };
    let cols = spectra[0].len();
    let reference_phase = coherent_phase(spectra);
    let mut corrected = spectra.to_vec();
    for (row, values) in corrected.iter_mut().enumerate() {
        let time = row as f64 * effective_integ_time as f64;
        for (ch, value) in values.iter_mut().enumerate() {
            let angle = diagnostics.phase_correction_rad[ch]
                + 2.0
                    * std::f64::consts::PI
                    * diagnostics.rate_residual_hz[ch]
                    * time
                    * diagnostics.taper[ch];
            let factor = Complex::new((-angle).cos() as f32, (-angle).sin() as f32);
            *value *= factor;
        }
    }
    if let (Some(before), Some(after)) = (reference_phase, coherent_phase(&corrected)) {
        let angle = before - after;
        let factor = Complex::new(angle.cos() as f32, angle.sin() as f32);
        for row in &mut corrected {
            for value in row {
                *value *= factor;
            }
        }
    }
    debug_assert_eq!(cols, corrected.first().map_or(0, Vec::len));
    corrected
}

fn average_spectrum(spectra: &[Vec<C32>]) -> Vec<C32> {
    if spectra.is_empty() || spectra[0].is_empty() {
        return Vec::new();
    }
    let rows = spectra.len() as f32;
    (0..spectra[0].len())
        .map(|channel| {
            spectra
                .iter()
                .map(|row| row[channel])
                .fold(C32::new(0.0, 0.0), |sum, value| sum + value)
                / rows
        })
        .collect()
}

fn write_frequency_spectrum_table(
    path: &Path,
    frequency_mhz: &[f64],
    raw: &[C32],
    fullband_corrected: &[C32],
    corrected: &[C32],
    search_delay: f32,
    search_rate: f32,
    fit_before_phase_deg: Option<&[f32]>,
    fit_after_phase_deg: Option<&[f32]>,
) -> Result<(), Box<dyn Error>> {
    if frequency_mhz.len() != raw.len()
        || raw.len() != fullband_corrected.len()
        || raw.len() != corrected.len()
        || fit_before_phase_deg.is_some_and(|values| values.len() != frequency_mhz.len())
        || fit_after_phase_deg.is_some_and(|values| values.len() != frequency_mhz.len())
    {
        return Err("spike34 frequency spectrum table has inconsistent lengths".into());
    }
    let mut out = String::from(
        "frequency_MHz\traw_amplitude\traw_phase_deg\tsearch_corrected_amplitude\tsearch_corrected_phase_deg\tspike34_corrected_amplitude\tspike34_corrected_phase_deg\tfit_before_phase_deg\tfit_after_phase_deg\tsearch_delay_sample\tsearch_rate_Hz\n",
    );
    for i in 0..frequency_mhz.len() {
        let before = raw[i];
        let searched = fullband_corrected[i];
        let after = corrected[i];
        let fit_before = fit_before_phase_deg
            .map(|values| values[i])
            .unwrap_or(f32::NAN);
        let fit_after = fit_after_phase_deg
            .map(|values| values[i])
            .unwrap_or(f32::NAN);
        out.push_str(&format!(
            "{:.9}\t{:.9e}\t{:.6}\t{:.9e}\t{:.6}\t{:.9e}\t{:.6}\t{:.6}\t{:.6}\t{search_delay:.9}\t{search_rate:.9}\n",
            frequency_mhz[i],
            before.norm(),
            before.arg().to_degrees(),
            searched.norm(),
            searched.arg().to_degrees(),
            after.norm(),
            after.arg().to_degrees(),
            fit_before,
            fit_after,
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

fn write_fit_residual_table(
    path: &Path,
    diagnostics: &SpikeFitDiagnostics,
) -> Result<(), Box<dyn Error>> {
    let mut out = String::from(
        "frequency_MHz\tphase0_wrapped_deg\tphase0_unwrapped_deg\tglobal_fit_deg\tinterval_fit_deg\traw_phase_residual_deg\tfinal_phase_residual_deg\trate_Hz\traw_rate_residual_Hz\tsmoothed_rate_residual_Hz\ttaper\n",
    );
    for i in 0..diagnostics.frequency_mhz.len() {
        out.push_str(&format!(
            "{:.9}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.6}\n",
            diagnostics.frequency_mhz[i],
            diagnostics.phase0_rad[i].to_degrees(),
            diagnostics.phase0_unwrapped_rad[i].to_degrees(),
            diagnostics.global_phase_fit_rad[i].to_degrees(),
            diagnostics.interval_phase_fit_rad[i].to_degrees(),
            diagnostics.raw_phase_residual_rad[i].to_degrees(),
            diagnostics.final_phase_residual_rad[i].to_degrees(),
            diagnostics.rate_hz[i],
            diagnostics.raw_rate_residual_hz[i],
            diagnostics.rate_residual_hz[i],
            diagnostics.taper[i],
        ));
    }
    fs::write(path, out)?;
    Ok(())
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
    let interval_corrections =
        write_interval_delay_rate_table(args, input_path, &delay_rate_table, &spikes)?;

    let (input_header, cross_spectra, cross_effective_integ_time) = read_all_spectra(input_path)?;
    let spike_channels: Vec<usize> = spikes.iter().map(|peak| peak.channel).collect();
    let amp_png = out_dir.join(format!("{stem}_rawvis_amp.png"));
    let phase_png = out_dir.join(format!("{stem}_rawvis_phase.png"));
    let corrected_amp_png = out_dir.join(format!("{stem}_rawvis_corrected_amp.png"));
    let corrected_phase_png = out_dir.join(format!("{stem}_rawvis_corrected_phase.png"));
    let spectrum_before_after_png = out_dir.join(format!("{stem}_spectrum_before_after.png"));
    let spectrum_before_after_tsv = out_dir.join(format!("{stem}_spectrum_before_after.tsv"));
    let fit_residual_png = out_dir.join(format!("{stem}_fit_residual.png"));
    let fit_residual_tsv = out_dir.join(format!("{stem}_fit_residual.tsv"));
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
    let (fullband_spectra, corrected_spectra, search_delay, search_rate) =
        apply_global_and_safe_spike_correction(
            &input_header,
            &cross_spectra,
            cross_effective_integ_time,
            &spikes,
            &interval_corrections,
            args,
            &current_obs_time,
            &file_start_time,
        )?;
    let raw_frequency_spectrum = average_spectrum(&cross_spectra);
    let fullband_frequency_spectrum = average_spectrum(&fullband_spectra);
    let corrected_frequency_spectrum = average_spectrum(&corrected_spectra);
    let frequency_mhz = frequency_axis_mhz(&input_header, raw_frequency_spectrum.len());
    let diagnostics = estimate_spike_fit_diagnostics(
        &input_header,
        &fullband_spectra,
        cross_effective_integ_time,
        &spikes,
        &interval_corrections,
    );
    let (fit_before_phase_deg, fit_after_phase_deg) = if let Some(diagnostics) =
        diagnostics.as_ref()
    {
        let before: Vec<f32> = diagnostics
            .phase0_unwrapped_rad
            .iter()
            .map(|value| value.to_degrees() as f32)
            .collect();
        // Remove only the interval residual (interval fit - global fit). The
        // global full-band trend is deliberately retained in the after series.
        let after: Vec<f32> = diagnostics
            .phase0_unwrapped_rad
            .iter()
            .zip(&diagnostics.global_phase_fit_rad)
            .zip(&diagnostics.interval_phase_fit_rad)
            .map(|((&phase, &global), &interval)| (phase - (interval - global)).to_degrees() as f32)
            .collect();
        (Some(before), Some(after))
    } else {
        (None, None)
    };
    write_frequency_spectrum_table(
        &spectrum_before_after_tsv,
        &frequency_mhz,
        &raw_frequency_spectrum,
        &fullband_frequency_spectrum,
        &corrected_frequency_spectrum,
        search_delay,
        search_rate,
        fit_before_phase_deg.as_deref(),
        fit_after_phase_deg.as_deref(),
    )?;
    plot_spike34_frequency_spectrum_with_phase(
        &spectrum_before_after_png,
        &frequency_mhz,
        &fullband_frequency_spectrum,
        &corrected_frequency_spectrum,
        &spikes
            .iter()
            .map(|peak| peak.frequency_mhz)
            .collect::<Vec<_>>(),
        fit_before_phase_deg.as_deref(),
        fit_after_phase_deg.as_deref(),
    )?;
    if let Some(diagnostics) = diagnostics {
        write_fit_residual_table(&fit_residual_tsv, &diagnostics)?;
        let phase0_unwrapped_deg: Vec<f32> = diagnostics
            .phase0_unwrapped_rad
            .iter()
            .map(|value| value.to_degrees() as f32)
            .collect();
        let global_fit_deg: Vec<f32> = diagnostics
            .global_phase_fit_rad
            .iter()
            .map(|value| value.to_degrees() as f32)
            .collect();
        let interval_fit_deg: Vec<f32> = diagnostics
            .interval_phase_fit_rad
            .iter()
            .map(|value| value.to_degrees() as f32)
            .collect();
        let raw_phase_residual_deg: Vec<f32> = diagnostics
            .raw_phase_residual_rad
            .iter()
            .map(|value| value.to_degrees() as f32)
            .collect();
        let final_phase_residual_deg: Vec<f32> = diagnostics
            .final_phase_residual_rad
            .iter()
            .map(|value| value.to_degrees() as f32)
            .collect();
        let raw_rate_residual_hz: Vec<f32> = diagnostics
            .raw_rate_residual_hz
            .iter()
            .map(|value| *value as f32)
            .collect();
        let smoothed_rate_residual_hz: Vec<f32> = diagnostics
            .rate_residual_hz
            .iter()
            .map(|value| *value as f32)
            .collect();
        let spike_frequencies: Vec<f64> = spikes.iter().map(|peak| peak.frequency_mhz).collect();
        plot_spike34_fit_residual(
            &fit_residual_png,
            &diagnostics.frequency_mhz,
            &phase0_unwrapped_deg,
            &global_fit_deg,
            &interval_fit_deg,
            &raw_phase_residual_deg,
            &final_phase_residual_deg,
            &raw_rate_residual_hz,
            &smoothed_rate_residual_hz,
            &spike_frequencies,
        )?;
    }
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

    println!(
        "Spike34 full-band --search correction: delay={:+.9} sample rate={:+.9} Hz",
        search_delay, search_rate
    );
    println!("Spike34 output directory: {}", out_dir.display());
    println!("Spike table: {}", spike_table.display());
    println!("Spike delay/rate table: {}", delay_rate_table.display());
    println!("Raw amplitude plot: {}", amp_png.display());
    println!("Raw phase plot: {}", phase_png.display());
    println!(
        "Spike-interval corrected amplitude plot: {}",
        corrected_amp_png.display()
    );
    println!(
        "Spike-interval corrected phase plot: {}",
        corrected_phase_png.display()
    );
    println!(
        "Frequency spectrum before/after plot: {}",
        spectrum_before_after_png.display()
    );
    println!(
        "Frequency spectrum before/after TSV: {}",
        spectrum_before_after_tsv.display()
    );
    if fit_residual_png.exists() {
        println!("Spike34 fit/residual plot: {}", fit_residual_png.display());
        println!("Spike34 fit/residual TSV: {}", fit_residual_tsv.display());
    }
    Ok(())
}
