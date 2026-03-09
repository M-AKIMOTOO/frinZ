use super::shared::{
    compress_plot_png, output_stem, prepare_output_directory, scaled_font_size,
    scaled_legend_font_size,
};
use anyhow::{anyhow, Context, Result};
use frinZ::header::{parse_header, CorHeader};
use frinZ::read::read_visibility_data;
use num_complex::Complex;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::colors::colormaps::ViridisRGB;
use std::fs;
use std::io::Cursor;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct KnownArgs {
    /// 入力する .cor ファイルのパス
    pub input: PathBuf,
    /// 折り畳み処理に用いるパルサーの回転周期（秒単位）
    pub period: f64,
    /// 周波数チャネル間の遅延を補正するための分散測定量（任意）
    pub dm: Option<f64>,
    /// 折り畳み後の位相ビン数
    pub bins: usize,
    /// 先頭から処理対象外とするセクター（PP）の数
    pub skip: u32,
    /// 解析に使用するセクター数の上限（0 で全区間）
    pub length: u32,
    /// オンパルスに割り当てる位相ビン割合
    pub on_duty: f64,
    /// 詳細な中間出力の有効化フラグ
    pub full_output: bool,
}

fn draw_info_block(root: &DrawingArea<BitMapBackend<'_>, Shift>, lines: &[String]) -> Result<()> {
    let text_style =
        TextStyle::from(("sans-serif", scaled_font_size(20)).into_font()).color(&BLACK);
    let x_pos = 145;
    let mut y_pos = 80;
    let line_step = scaled_font_size(20) + 8;
    for line in lines {
        root.draw(&Text::new(
            line.as_str(),
            (x_pos, y_pos),
            text_style.clone(),
        ))?;
        y_pos += line_step;
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct SectorData {
    pub(crate) integ_time: f64,
    pub(crate) spectra: Vec<Complex<f32>>,
}

struct DedispersionOutputs {
    dedispersed_time_series: Vec<(f64, f64)>,
    dedispersed_weights: Vec<f64>,
    integrated_time_series: Vec<(f64, f64)>,
    raw_spectrum: Vec<(f64, f64)>,
    spectra_heatmap: Vec<Vec<Complex<f32>>>,
    dedispersed_heatmap: Vec<Vec<Complex<f32>>>,
    raw_phase_heatmap: Vec<Vec<Complex<f32>>>,
    dedispersed_phase_heatmap: Vec<Vec<Complex<f32>>>,
    total_integration: f64,
    pp_elapsed: Vec<f64>,
    pp_durations: Vec<f64>,
    dm_delay_min: Option<f64>,
    dm_delay_max: Option<f64>,
}

pub fn run(cli: KnownArgs) -> Result<()> {
    if cli.period <= 0.0 {
        return Err(anyhow!("--period must be positive"));
    }
    if cli.bins == 0 {
        return Err(anyhow!("--bins must be greater than 0"));
    }
    if !(0.0..=1.0).contains(&cli.on_duty) {
        return Err(anyhow!("--on-duty must be within [0, 1]"));
    }

    let buffer = fs::read(&cli.input)
        .with_context(|| format!("failed to read input file {}", cli.input.display()))?;
    let mut cursor = Cursor::new(buffer.as_slice());

    let header = parse_header(&mut cursor)?;

    let sectors = load_sectors(&mut cursor, &header, &cli)?;
    if sectors.is_empty() {
        return Err(anyhow!("no sectors were read from the file"));
    }

    let freq_axis_mhz = build_frequency_axis_mhz(&header);
    let dedisp_outputs = build_dedispersed_series(&sectors, &freq_axis_mhz, &cli)?;

    let folded = fold_profile(
        &dedisp_outputs.dedispersed_time_series,
        &dedisp_outputs.dedispersed_weights,
        cli.period,
        cli.bins,
    )?;
    let gating = determine_gating(&folded, cli.on_duty);
    let gated = compute_gated_aggregation(
        &dedisp_outputs,
        &gating,
        &freq_axis_mhz,
        cli.period,
        cli.bins,
    );

    let output_dir = prepare_output_directory(&cli.input)?;
    write_outputs(
        &output_dir,
        &cli,
        &header,
        &dedisp_outputs,
        &freq_axis_mhz,
        header.observing_frequency / 1.0e6,
        &folded,
        &gating,
        gated.as_ref(),
    )?;

    print_summary(
        &header,
        &cli,
        &dedisp_outputs,
        &folded,
        &gating,
        gated.as_ref(),
    )?;
    Ok(())
}

pub(crate) fn load_sectors_with_limits(
    cursor: &mut Cursor<&[u8]>,
    header: &CorHeader,
    skip: u32,
    length: u32,
) -> Result<Vec<SectorData>> {
    let total = header.number_of_sector.max(0) as u32;
    let skip = skip.min(total);
    let max_length = if length == 0 {
        total.saturating_sub(skip)
    } else {
        length.min(total.saturating_sub(skip))
    };

    let mut sectors = Vec::with_capacity(max_length as usize);
    for idx in 0..max_length {
        let loop_index = skip + idx;
        let (spectra, _timestamp, integ) =
            read_visibility_data(cursor, header, 1, 0, loop_index as i32, false, &[])
                .with_context(|| format!("failed to read sector {}", loop_index))?;

        if spectra.is_empty() {
            continue;
        }
        sectors.push(SectorData {
            integ_time: integ as f64,
            spectra,
        });
    }
    Ok(sectors)
}

fn load_sectors(
    cursor: &mut Cursor<&[u8]>,
    header: &CorHeader,
    cli: &KnownArgs,
) -> Result<Vec<SectorData>> {
    load_sectors_with_limits(cursor, header, cli.skip, cli.length)
}

pub(crate) fn build_frequency_axis_mhz(header: &CorHeader) -> Vec<f64> {
    let base_freq_mhz = header.observing_frequency / 1.0e6;
    let df_mhz = header.sampling_speed as f64 / header.fft_point as f64 / 1.0e6;
    (0..header.fft_point as usize / 2)
        .map(|idx| base_freq_mhz + idx as f64 * df_mhz)
        .collect()
}

fn build_dedispersed_series(
    sectors: &[SectorData],
    freq_axis_mhz: &[f64],
    cli: &KnownArgs,
) -> Result<DedispersionOutputs> {
    let samples_per_sector = freq_axis_mhz.len();
    if samples_per_sector == 0 {
        return Err(anyhow!("FFT length is zero, cannot proceed"));
    }

    let mut time_series_amp: Vec<Vec<f64>> =
        vec![Vec::with_capacity(sectors.len()); samples_per_sector];
    let mut time_series_complex: Vec<Vec<Complex<f32>>> =
        vec![Vec::with_capacity(sectors.len()); samples_per_sector];
    let mut spectra_heatmap = Vec::with_capacity(sectors.len());
    let mut pp_elapsed = Vec::with_capacity(sectors.len());
    let mut pp_durations = Vec::with_capacity(sectors.len());
    let mut cumulative_observation = 0.0;

    for (sector_index, sector) in sectors.iter().enumerate() {
        let integ = sector.integ_time.max(1e-6);
        pp_elapsed.push(cumulative_observation);
        pp_durations.push(integ);
        cumulative_observation += integ;

        let mut spectrum_row: Vec<Complex<f32>> = Vec::with_capacity(samples_per_sector);
        for chan_idx in 0..samples_per_sector {
            let value = sector
                .spectra
                .get(chan_idx)
                .copied()
                .unwrap_or_else(|| Complex::new(0.0, 0.0));
            time_series_amp[chan_idx].push(value.norm() as f64);
            time_series_complex[chan_idx].push(value);
            spectrum_row.push(value);
        }
        spectra_heatmap.push(spectrum_row);
        if sector_index == 0 && sector.integ_time == 0.0 {
            return Err(anyhow!(
                "effective integration time is zero; cannot determine time resolution"
            ));
        }
    }

    let reference_freq_mhz = freq_axis_mhz
        .first()
        .copied()
        .ok_or_else(|| anyhow!("frequency axis is empty"))?;
    let channel_delays = cli
        .dm
        .map(|dm| compute_dispersion_delays(freq_axis_mhz, reference_freq_mhz, dm));
    let dm_delay_stats = channel_delays.as_ref().map(|v| {
        let (&min_delay, &max_delay) = (
            v.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
            v.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
        );
        (min_delay, max_delay)
    });

    let mut raw_weights = vec![0.0f64; sectors.len()];
    let mut raw_spectrum_sums = vec![0.0f64; samples_per_sector];
    let mut raw_spectrum_weights = vec![0.0f64; samples_per_sector];

    for sample_idx in 0..sectors.len() {
        let integ = pp_durations[sample_idx].max(1e-6);
        for chan_idx in 0..samples_per_sector {
            let amp_value = time_series_amp[chan_idx][sample_idx];
            raw_spectrum_sums[chan_idx] += amp_value * integ;
            raw_spectrum_weights[chan_idx] += integ;
        }
        raw_weights[sample_idx] = integ;
    }

    let mut centers = Vec::with_capacity(sectors.len());
    for idx in 0..sectors.len() {
        let duration = pp_durations[idx].max(1e-6);
        let center = pp_elapsed[idx] + duration / 2.0;
        centers.push(center);
    }

    let mut dedispersed_heatmap =
        vec![vec![Complex::new(0.0, 0.0); samples_per_sector]; sectors.len()];

    if let Some(delays) = channel_delays.as_ref() {
        let last_index = sectors.len().saturating_sub(1);
        for (chan_idx, &delay) in delays.iter().enumerate() {
            let src_series = &time_series_complex[chan_idx];
            for dest_idx in 0..sectors.len() {
                // Dedisperse by delaying early-arriving channels (positive delay)
                // so all channels align to the reference channel timeline.
                let target_time = centers[dest_idx] - delay;
                let value = interpolate_complex(&centers, src_series, target_time, last_index);
                dedispersed_heatmap[dest_idx][chan_idx] = value;
            }
        }
    } else {
        dedispersed_heatmap.clone_from(&spectra_heatmap);
    }

    let mut dedispersed_time_series = Vec::with_capacity(sectors.len());
    let mut dedispersed_weights_vec = Vec::with_capacity(sectors.len());
    let mut integrated_time_series = Vec::with_capacity(sectors.len());

    for (row_idx, duration) in pp_durations.iter().enumerate() {
        let center = centers[row_idx];
        let mut amp_sum = 0.0f64;
        for chan_idx in 0..samples_per_sector {
            let amp = dedispersed_heatmap[row_idx][chan_idx].norm() as f64;
            amp_sum += amp;
        }
        dedispersed_time_series.push((center, amp_sum * duration));
        dedispersed_weights_vec.push(*duration);
        integrated_time_series.push((center, amp_sum));
    }

    let raw_phase_heatmap = build_phase_aligned_heatmap(
        &spectra_heatmap,
        &centers,
        &pp_durations,
        cli.bins,
        cli.period,
    );
    let dedispersed_phase_heatmap = build_phase_aligned_heatmap(
        &dedispersed_heatmap,
        &centers,
        &pp_durations,
        cli.bins,
        cli.period,
    );

    let mut raw_spectrum = Vec::with_capacity(samples_per_sector);
    for (idx, freq) in freq_axis_mhz.iter().enumerate() {
        let raw_weight = raw_spectrum_weights[idx];
        let raw_amp = if raw_weight > 0.0 {
            raw_spectrum_sums[idx] / raw_weight
        } else {
            0.0
        };
        raw_spectrum.push((*freq, raw_amp));
    }

    let total_integration: f64 = raw_weights.iter().sum();

    Ok(DedispersionOutputs {
        dedispersed_time_series,
        dedispersed_weights: dedispersed_weights_vec,
        integrated_time_series,
        raw_spectrum,
        spectra_heatmap,
        dedispersed_heatmap,
        raw_phase_heatmap,
        dedispersed_phase_heatmap,
        total_integration,
        pp_elapsed,
        pp_durations,
        dm_delay_min: dm_delay_stats.map(|(mn, _)| mn),
        dm_delay_max: dm_delay_stats.map(|(_, mx)| mx),
    })
}

#[allow(dead_code)]
pub(crate) fn build_onpulse_dedispersed_sectors(
    sectors: &[SectorData],
    freq_axis_mhz: &[f64],
    period: f64,
    bins: usize,
    on_duty: f64,
    dm: Option<f64>,
) -> Result<Vec<SectorData>> {
    if sectors.is_empty() {
        return Ok(Vec::new());
    }
    if period <= 0.0 || bins == 0 || !(0.0..=1.0).contains(&on_duty) {
        return Err(anyhow!(
            "invalid gating parameters (period, bins, on_duty) for on-pulse sector construction"
        ));
    }

    let cli = KnownArgs {
        input: PathBuf::new(),
        period,
        dm,
        bins,
        skip: 0,
        length: 0,
        on_duty,
        full_output: false,
    };
    let dedispersed = build_dedispersed_series(sectors, freq_axis_mhz, &cli)?;
    let folded = fold_profile(
        &dedispersed.dedispersed_time_series,
        &dedispersed.dedispersed_weights,
        period,
        bins,
    )?;
    let gating = determine_gating(&folded, on_duty);

    let mut on_mask = vec![false; bins];
    for &bin in &gating.on_bins {
        if bin < bins {
            on_mask[bin] = true;
        }
    }
    if !on_mask.iter().any(|&v| v) {
        return Err(anyhow!(
            "failed to determine on-pulse bins for gated delay-rate analysis"
        ));
    }

    let channels = dedispersed
        .dedispersed_heatmap
        .first()
        .map(|r| r.len())
        .unwrap_or(0);
    let first_center = dedispersed.pp_elapsed.get(0).copied().unwrap_or(0.0)
        + dedispersed.pp_durations.get(0).copied().unwrap_or(0.0) / 2.0;

    let on_count = on_mask.iter().filter(|&&v| v).count();
    let off_count = bins.saturating_sub(on_count);
    let off_scale = if off_count > 0 {
        on_count as f32 / off_count as f32
    } else {
        0.0
    };

    let mut out = Vec::with_capacity(dedispersed.dedispersed_heatmap.len());
    for idx in 0..dedispersed.dedispersed_heatmap.len() {
        let center = dedispersed.pp_elapsed[idx] + dedispersed.pp_durations[idx] / 2.0;
        let phase = ((center - first_center) / period).rem_euclid(1.0);
        let bin = ((phase * bins as f64).floor() as usize).min(bins - 1);
        // Use a zero-mean gate (+1 for on-pulse, -alpha for off-pulse) to reduce
        // periodic window leakage in delay-rate space while preserving cadence.
        let weight: f32 = if on_mask[bin] { 1.0 } else { -off_scale };
        let spectra = if weight.abs() <= f32::EPSILON {
            vec![Complex::new(0.0, 0.0); channels]
        } else {
            dedispersed.dedispersed_heatmap[idx]
                .iter()
                .map(|v| *v * weight)
                .collect()
        };
        out.push(SectorData {
            integ_time: dedispersed.pp_durations[idx],
            spectra,
        });
    }
    Ok(out)
}

fn compute_dispersion_delays(freq_axis_mhz: &[f64], ref_freq_mhz: f64, dm: f64) -> Vec<f64> {
    const DM_CONST_MS: f64 = 4.148_808e3;
    freq_axis_mhz
        .iter()
        .map(|&f| {
            let delay_ms = DM_CONST_MS * dm * (1.0 / (ref_freq_mhz * ref_freq_mhz) - 1.0 / (f * f));
            delay_ms / 1e3
        })
        .collect()
}

fn build_phase_aligned_heatmap(
    heatmap: &[Vec<Complex<f32>>],
    centers: &[f64],
    _durations: &[f64],
    bins: usize,
    period: f64,
) -> Vec<Vec<Complex<f32>>> {
    if heatmap.is_empty()
        || heatmap[0].is_empty()
        || bins == 0
        || period <= 0.0
        || centers.is_empty()
    {
        return Vec::new();
    }
    let channels = heatmap[0].len();
    let mut accum = vec![vec![0.0f64; channels]; bins];
    let t0 = centers[0];

    for (row_idx, row) in heatmap.iter().enumerate() {
        if row.len() != channels {
            continue;
        }
        let center = centers.get(row_idx).copied().unwrap_or(t0);
        let phase = ((center - t0) / period).rem_euclid(1.0);
        let bin = ((phase * bins as f64).floor() as usize).min(bins - 1);
        for (chan_idx, value) in row.iter().enumerate() {
            let amp = value.norm() as f64;
            if amp.is_finite() {
                accum[bin][chan_idx] += amp;
            }
        }
    }

    accum
        .into_iter()
        .map(|row_accum| {
            row_accum
                .into_iter()
                .map(|sum| Complex::new(sum as f32, 0.0))
                .collect()
        })
        .collect()
}

fn interpolate_complex(
    times: &[f64],
    values: &[Complex<f32>],
    target: f64,
    last_index: usize,
) -> Complex<f32> {
    if times.is_empty() || values.is_empty() || last_index == 0 {
        return Complex::new(0.0, 0.0);
    }
    if target <= times[0] || target >= times[last_index] {
        return Complex::new(0.0, 0.0);
    }

    let mut left = 0usize;
    let mut right = last_index;
    while left + 1 < right {
        let mid = (left + right) / 2;
        if times[mid] <= target {
            left = mid;
        } else {
            right = mid;
        }
    }

    let t0 = times[left];
    let t1 = times[left + 1];
    if t1 <= t0 {
        return Complex::new(0.0, 0.0);
    }
    let frac = ((target - t0) / (t1 - t0)).clamp(0.0, 1.0);
    let c0 = values[left];
    let c1 = values[left + 1];
    c0 * (1.0 - frac as f32) + c1 * (frac as f32)
}

pub(crate) fn fold_profile(
    dedispersed: &[(f64, f64)],
    weights: &[f64],
    period: f64,
    bins: usize,
) -> Result<Vec<(f64, f64)>> {
    if dedispersed.is_empty() {
        return Err(anyhow!("dedispersed time series is empty"));
    }
    Ok(accumulate_phase_series(dedispersed, weights, period, bins))
}

fn accumulate_phase_series(
    timeseries: &[(f64, f64)],
    weights: &[f64],
    period: f64,
    bins: usize,
) -> Vec<(f64, f64)> {
    if timeseries.is_empty() || period <= 0.0 || bins == 0 {
        return Vec::new();
    }

    let t0 = timeseries[0].0;
    let mut accum = vec![0.0f64; bins];
    let mut weight_sums = vec![0.0f64; bins];

    for (idx, &(time, amp)) in timeseries.iter().enumerate() {
        let weight = weights.get(idx).copied().unwrap_or(1.0);
        if weight <= 0.0 {
            continue;
        }
        let phase = ((time - t0) / period).rem_euclid(1.0);
        let bin = ((phase * bins as f64).floor() as usize).min(bins - 1);
        accum[bin] += amp * weight;
        weight_sums[bin] += weight;
    }

    accum
        .iter()
        .zip(weight_sums.iter())
        .enumerate()
        .map(|(idx, (sum, w))| {
            let phase = (idx as f64 + 0.5) / bins as f64;
            let amp = if *w > 0.0 { sum / w } else { 0.0 };
            (phase, amp)
        })
        .collect()
}

#[derive(Debug)]
struct GatingResult {
    on_bins: Vec<usize>,
    off_bins: Vec<usize>,
    peak_phase: f64,
    snr: f64,
}

struct GatedAggregation {
    on_spectrum: Vec<(f64, f64)>,
    off_spectrum: Vec<(f64, f64)>,
    diff_spectrum: Vec<(f64, f64)>,
    on_weight: f64,
    off_weight: f64,
    on_mean: f64,
    off_mean: f64,
    off_sigma: Option<f64>,
    time_snr: Option<f64>,
    time_series: Vec<(f64, f64, bool)>,
    gated_profile: Vec<(f64, f64)>,
    gated_profile_snr: Option<f64>,
    gated_profile_sigma: Option<f64>,
    diff_time_series: Vec<(f64, f64)>,
}

fn determine_gating(profile: &[(f64, f64)], on_duty: f64) -> GatingResult {
    if profile.is_empty() || on_duty <= 0.0 {
        return GatingResult {
            on_bins: Vec::new(),
            off_bins: (0..profile.len()).collect(),
            peak_phase: 0.0,
            snr: 0.0,
        };
    }

    let bins = profile.len();
    let mut sorted: Vec<(usize, f64)> = profile
        .iter()
        .enumerate()
        .map(|(idx, &(_, amp))| (idx, amp))
        .collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let on_bin_count = ((bins as f64 * on_duty).ceil() as usize).clamp(1, bins);
    let on_bins: Vec<usize> = sorted
        .iter()
        .take(on_bin_count)
        .map(|(idx, _)| *idx)
        .collect();
    let mut is_on = vec![false; bins];
    for &idx in &on_bins {
        if idx < bins {
            is_on[idx] = true;
        }
    }
    let off_bins: Vec<usize> = (0..bins).filter(|idx| !is_on[*idx]).collect();

    let peak_idx = sorted.first().map(|(idx, _)| *idx).unwrap_or(0);
    let peak_phase = profile[peak_idx].0;

    let off_mean = if off_bins.is_empty() {
        0.0
    } else {
        off_bins.iter().map(|&idx| profile[idx].1).sum::<f64>() / off_bins.len() as f64
    };
    let off_std = if off_bins.len() > 1 {
        let mean = off_mean;
        let var = off_bins
            .iter()
            .map(|&idx| {
                let diff = profile[idx].1 - mean;
                diff * diff
            })
            .sum::<f64>()
            / (off_bins.len() - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };
    let peak_amp = profile[peak_idx].1;
    let snr = if off_std > 0.0 {
        (peak_amp - off_mean) / off_std
    } else {
        0.0
    };

    GatingResult {
        on_bins,
        off_bins,
        peak_phase,
        snr,
    }
}

fn compute_gated_aggregation(
    dedispersed: &DedispersionOutputs,
    gating: &GatingResult,
    freq_axis_mhz: &[f64],
    period: f64,
    bins: usize,
) -> Option<GatedAggregation> {
    if bins == 0
        || period <= 0.0
        || freq_axis_mhz.is_empty()
        || dedispersed.dedispersed_heatmap.is_empty()
        || dedispersed.dedispersed_heatmap[0].is_empty()
    {
        return None;
    }
    if gating.on_bins.is_empty() {
        return None;
    }

    let channels = dedispersed.dedispersed_heatmap[0].len();
    if freq_axis_mhz.len() != channels {
        return None;
    }
    let sectors = dedispersed.dedispersed_heatmap.len();
    if sectors == 0
        || dedispersed.pp_durations.len() != sectors
        || dedispersed.dedispersed_time_series.len() != sectors
    {
        return None;
    }

    let mut on_mask = vec![false; bins];
    for &bin in &gating.on_bins {
        if bin < bins {
            on_mask[bin] = true;
        }
    }
    if !on_mask.iter().any(|&v| v) {
        return None;
    }

    let mut bin_assignments = Vec::with_capacity(sectors);
    let first_center = dedispersed.pp_elapsed.get(0).copied().unwrap_or(0.0)
        + dedispersed.pp_durations.get(0).copied().unwrap_or(0.0) / 2.0;
    for (idx, elapsed) in dedispersed.pp_elapsed.iter().enumerate() {
        let duration = dedispersed.pp_durations.get(idx).copied().unwrap_or(0.0);
        let center = elapsed + duration / 2.0;
        let phase = ((center - first_center) / period).rem_euclid(1.0);
        let bin = ((phase * bins as f64).floor() as usize).min(bins - 1);
        bin_assignments.push((center, bin));
    }

    let mut on_weight = 0.0f64;
    let mut off_weight = 0.0f64;
    let mut on_total = 0.0f64;
    let mut off_total = 0.0f64;
    let mut off_square_total = 0.0f64;
    let mut on_spectrum_sum = vec![0.0f64; channels];
    let mut off_spectrum_sum = vec![0.0f64; channels];
    let mut off_channel_sums = vec![0.0f64; channels];
    let mut off_channel_weights = vec![0.0f64; channels];
    let mut time_series = Vec::with_capacity(sectors);

    for (sector_idx, row) in dedispersed.dedispersed_heatmap.iter().enumerate() {
        let duration = dedispersed
            .pp_durations
            .get(sector_idx)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let weight = duration.max(0.0);
        let (center, bin) = bin_assignments.get(sector_idx).copied().unwrap_or((0.0, 0));
        let is_on = on_mask[bin];

        let avg_amp = if duration > 0.0 {
            dedispersed
                .dedispersed_time_series
                .get(sector_idx)
                .map(|(_, amp)| *amp / duration)
                .unwrap_or(0.0)
        } else {
            0.0
        };
        time_series.push((center, avg_amp, is_on));

        for (chan_idx, value) in row.iter().enumerate() {
            let amp = value.norm() as f64;
            if is_on {
                on_spectrum_sum[chan_idx] += amp * weight;
            } else {
                off_spectrum_sum[chan_idx] += amp * weight;
                off_channel_sums[chan_idx] += amp * weight;
                off_channel_weights[chan_idx] += weight;
            }
        }

        if is_on {
            on_weight += weight;
            on_total += avg_amp * weight;
        } else {
            off_weight += weight;
            off_total += avg_amp * weight;
            off_square_total += avg_amp * avg_amp * weight;
        }
    }

    if on_weight == 0.0 {
        return None;
    }

    let on_mean = if on_weight > 0.0 {
        on_total / on_weight
    } else {
        0.0
    };
    let off_mean = if off_weight > 0.0 {
        off_total / off_weight
    } else {
        0.0
    };
    let off_sigma = if off_weight > 0.0 {
        let var = (off_square_total / off_weight) - off_mean * off_mean;
        Some(var.max(0.0).sqrt())
    } else {
        None
    };
    let time_snr = off_sigma
        .filter(|&sigma| sigma > 0.0)
        .map(|sigma| (on_mean - off_mean) / sigma);

    let on_spectrum: Vec<(f64, f64)> = freq_axis_mhz
        .iter()
        .enumerate()
        .map(|(idx, &freq)| {
            let amp = if on_weight > 0.0 {
                on_spectrum_sum[idx] / on_weight
            } else {
                0.0
            };
            (freq, amp)
        })
        .collect();
    let off_spectrum: Vec<(f64, f64)> = freq_axis_mhz
        .iter()
        .enumerate()
        .map(|(idx, &freq)| {
            let amp = if off_weight > 0.0 {
                off_spectrum_sum[idx] / off_weight
            } else {
                0.0
            };
            (freq, amp)
        })
        .collect();
    let diff_spectrum: Vec<(f64, f64)> = on_spectrum
        .iter()
        .zip(off_spectrum.iter())
        .map(|((freq, on_amp), (_, off_amp))| (*freq, on_amp - off_amp))
        .collect();

    // Build on-pulse means per phase bin for S/N estimation
    let mut gated_bin_sums = vec![0.0f64; bins];
    let mut gated_bin_weights = vec![0.0f64; bins];
    let base_time = time_series.first().map(|&(t, _, _)| t).unwrap_or(0.0);
    for (idx, &(center, avg_amp, is_on)) in time_series.iter().enumerate() {
        if !is_on {
            continue;
        }
        let duration = dedispersed
            .pp_durations
            .get(idx)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        if duration <= 0.0 {
            continue;
        }
        let phase = ((center - base_time) / period).rem_euclid(1.0);
        let bin = ((phase * bins as f64).floor() as usize).min(bins - 1);
        gated_bin_sums[bin] += avg_amp * duration;
        gated_bin_weights[bin] += duration;
    }
    let mut gated_bin_means = vec![0.0f64; bins];
    for idx in 0..bins {
        let w = gated_bin_weights[idx];
        if w > 0.0 {
            gated_bin_means[idx] = gated_bin_sums[idx] / w;
        }
    }

    let mut off_channel_means = vec![0.0f64; channels];
    for idx in 0..channels {
        let w = off_channel_weights[idx];
        if w > 0.0 {
            off_channel_means[idx] = off_channel_sums[idx] / w;
        }
    }

    let gated_profile: Vec<(f64, f64)> = (0..bins)
        .map(|idx| {
            let phase = (idx as f64 + 0.5) / bins as f64;
            let value = if gated_bin_weights[idx] > 0.0 {
                gated_bin_means[idx] - off_mean
            } else {
                0.0
            };
            (phase, value)
        })
        .collect();

    let mut diff_time_series = Vec::with_capacity(sectors);
    let mut on_diff_values = Vec::new();
    let mut off_diff_values = Vec::new();
    for (sector_idx, row) in dedispersed.dedispersed_heatmap.iter().enumerate() {
        let (center, bin) = bin_assignments[sector_idx];
        let is_on = on_mask[bin];
        let mut diff_sum = 0.0f64;
        for (chan_idx, value) in row.iter().enumerate() {
            let amp = value.norm() as f64;
            let diff = amp - off_channel_means[chan_idx];
            diff_sum += diff;
        }
        diff_time_series.push((center, diff_sum));

        if is_on {
            on_diff_values.push(diff_sum);
        } else {
            off_diff_values.push(diff_sum);
        }
    }

    let diff_peak = on_diff_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);

    let diff_sigma = if off_diff_values.len() > 1 {
        let mean = off_diff_values.iter().copied().sum::<f64>() / off_diff_values.len() as f64;
        let var = off_diff_values
            .iter()
            .map(|&v| {
                let diff = v - mean;
                diff * diff
            })
            .sum::<f64>()
            / (off_diff_values.len() - 1) as f64;
        Some(var.max(0.0).sqrt())
    } else {
        None
    };

    let gated_sigma = diff_sigma.or(off_sigma);
    let gated_snr = gated_sigma.and_then(|sigma| {
        if sigma > 0.0 {
            Some(diff_peak / sigma)
        } else {
            None
        }
    });

    Some(GatedAggregation {
        on_spectrum,
        off_spectrum,
        diff_spectrum,
        on_weight,
        off_weight,
        on_mean,
        off_mean,
        off_sigma,
        time_snr,
        time_series,
        gated_profile,
        gated_profile_snr: gated_snr,
        gated_profile_sigma: gated_sigma,
        diff_time_series,
    })
}

fn write_outputs(
    output_dir: &Path,
    cli: &KnownArgs,
    header: &CorHeader,
    dedispersed: &DedispersionOutputs,
    freq_axis_mhz: &[f64],
    center_freq_mhz: f64,
    folded: &[(f64, f64)],
    gating: &GatingResult,
    gated: Option<&GatedAggregation>,
) -> Result<()> {
    let stem_owned = output_stem(&cli.input);
    let stem = stem_owned.as_str();
    cleanup_legacy_bin_outputs(output_dir, stem);
    let full_output = cli.full_output;
    let raw_rows = dedispersed.spectra_heatmap.len();
    let raw_cols = dedispersed
        .spectra_heatmap
        .get(0)
        .map(|row| row.len())
        .unwrap_or(0);
    let dedisp_rows = dedispersed.dedispersed_heatmap.len();
    let dedisp_cols = dedispersed
        .dedispersed_heatmap
        .get(0)
        .map(|row| row.len())
        .unwrap_or(0);
    if full_output
        && cli.dm.is_some()
        && raw_rows == dedisp_rows
        && raw_cols == dedisp_cols
        && raw_rows > 0
        && raw_cols > 0
    {
        let (max_diff, mean_diff) = heatmap_difference_stats(
            &dedispersed.spectra_heatmap,
            &dedispersed.dedispersed_heatmap,
        );
        println!(
            "Dedispersion amplitude difference: max={:.6}, mean={:.6}",
            max_diff, mean_diff
        );
    }
    if full_output && cli.dm.is_some() && (raw_rows != dedisp_rows || raw_cols != dedisp_cols) {
        eprintln!(
            "Warning: dedispersed heatmap size {}x{} differs from raw heatmap {}x{}",
            dedisp_rows, dedisp_cols, raw_rows, raw_cols
        );
    }
    let profile_path = output_dir.join(format!("{stem}_profile.csv"));
    let mut profile_file = fs::File::create(&profile_path)
        .with_context(|| format!("failed to write {profile_path:?}"))?;
    writeln!(profile_file, "bin,phase,amplitude")?;
    for (idx, (phase, amp)) in folded.iter().enumerate() {
        writeln!(profile_file, "{idx},{phase:.6},{amp:.6}")?;
    }

    let profile_plot = output_dir.join(format!("{stem}_folded_profile.png"));
    plot_folded_profile(&profile_plot, folded, gating, cli)?;

    if full_output && !dedispersed.integrated_time_series.is_empty() {
        let time_series_csv = output_dir.join(format!("{stem}_dedispersed_time_series.csv"));
        let mut ts_file = fs::File::create(&time_series_csv)
            .with_context(|| format!("failed to write {time_series_csv:?}"))?;
        writeln!(ts_file, "time_s,amplitude_sum")?;
        for (time, amp) in &dedispersed.integrated_time_series {
            writeln!(ts_file, "{time:.6},{amp:.6}")?;
        }

        let time_series_plot = output_dir.join(format!("{stem}_dedispersed_time_series.png"));
        plot_time_series(
            &time_series_plot,
            &dedispersed.integrated_time_series,
            cli.period,
            "Dedispersed time series (frequency-integrated)",
            "Integrated amplitude",
        )?;
    }

    if full_output && !dedispersed.raw_phase_heatmap.is_empty() {
        let phase_heatmap_path = output_dir.join(format!("{stem}_raw_phase_heatmap.png"));
        plot_phase_aligned_heatmap(
            &phase_heatmap_path,
            &dedispersed.raw_phase_heatmap,
            freq_axis_mhz,
            center_freq_mhz,
        )?;
    }
    if full_output && cli.dm.is_some() && !dedispersed.dedispersed_phase_heatmap.is_empty() {
        let heatmap_path = output_dir.join(format!("{stem}_phase_aligned_heatmap.png"));
        plot_phase_aligned_heatmap(
            &heatmap_path,
            &dedispersed.dedispersed_phase_heatmap,
            freq_axis_mhz,
            center_freq_mhz,
        )?;

        let on_diff_heatmap = build_on_pulse_phase_difference_heatmap(
            &dedispersed.dedispersed_heatmap,
            &dedispersed.pp_elapsed,
            &dedispersed.pp_durations,
            cli.period,
            cli.bins,
            gating,
        );
        if !on_diff_heatmap.is_empty() {
            let diff_heatmap_path =
                output_dir.join(format!("{stem}_phase_aligned_onminusoff_heatmap.png"));
            plot_phase_aligned_heatmap(
                &diff_heatmap_path,
                &on_diff_heatmap,
                freq_axis_mhz,
                center_freq_mhz,
            )?;
        }
    }
    if let Some(gated_data) = gated {
        let diff_path = output_dir.join(format!("{stem}_gated_spectrum_difference.csv"));
        write_spectrum_csv(
            &diff_path,
            &gated_data.diff_spectrum,
            "amplitude_on_minus_off",
        )?;

        let spectrum_plot = output_dir.join(format!("{stem}_gated_spectrum.png"));
        plot_gated_spectrum(
            &spectrum_plot,
            &gated_data.on_spectrum,
            &gated_data.off_spectrum,
            &gated_data.diff_spectrum,
        )?;

        if full_output {
            let on_path = output_dir.join(format!("{stem}_gated_spectrum_on.csv"));
            write_spectrum_csv(&on_path, &gated_data.on_spectrum, "amplitude_on")?;
            if gated_data.off_weight > 0.0 {
                let off_path = output_dir.join(format!("{stem}_gated_spectrum_off.csv"));
                write_spectrum_csv(&off_path, &gated_data.off_spectrum, "amplitude_off")?;
            }
            let time_series_path = output_dir.join(format!("{stem}_gated_time_series.csv"));
            write_gated_time_series_csv(&time_series_path, &gated_data.time_series)?;
            let time_series_plot = output_dir.join(format!("{stem}_gated_time_series.png"));
            plot_gated_time_series(&time_series_plot, &gated_data.time_series)?;
        }

        if !gated_data.gated_profile.is_empty() {
            let profile_csv = output_dir.join(format!("{stem}_gated_profile.csv"));
            write_series_csv(
                &profile_csv,
                "phase",
                "amplitude_on_minus_off",
                &gated_data.gated_profile,
            )?;
            let profile_plot = output_dir.join(format!("{stem}_gated_profile.png"));
            plot_gated_profile(
                &profile_plot,
                &gated_data.gated_profile,
                gating,
                cli,
                gated_data.off_mean,
                gated_data.off_sigma,
                gated_data.gated_profile_snr,
                gated_data.gated_profile_sigma,
            )?;
        }

        if full_output && !gated_data.diff_time_series.is_empty() {
            let diff_ts_csv = output_dir.join(format!("{stem}_gated_time_series_diff.csv"));
            let mut diff_file = fs::File::create(&diff_ts_csv)
                .with_context(|| format!("failed to write {diff_ts_csv:?}"))?;
            writeln!(diff_file, "time_s,amplitude_diff")?;
            for (time, amp) in &gated_data.diff_time_series {
                writeln!(diff_file, "{time:.6},{amp:.6}")?;
            }

            let diff_ts_plot = output_dir.join(format!("{stem}_gated_time_series_diff.png"));
            plot_time_series(
                &diff_ts_plot,
                &gated_data.diff_time_series,
                cli.period,
                "Dedispersed time series (on-off diff)",
                "Amplitude (on - off)",
            )?;
        }

        let _ = full_output;
    }

    let bins_path = output_dir.join(format!("{stem}_onoff_pulse_bins.txt"));
    let mut bins_file =
        fs::File::create(&bins_path).with_context(|| format!("failed to write {bins_path:?}"))?;
    writeln!(bins_file, "# On-pulse bins")?;
    for idx in &gating.on_bins {
        writeln!(bins_file, "{idx}")?;
    }
    writeln!(bins_file, "# Off-pulse bins")?;
    for idx in &gating.off_bins {
        writeln!(bins_file, "{idx}")?;
    }

    let summary_path = output_dir.join(format!("{stem}_summary.txt"));
    let mut summary_file = fs::File::create(&summary_path)
        .with_context(|| format!("failed to write {summary_path:?}"))?;
    writeln!(summary_file, "# Pulsar gating summary")?;
    writeln!(
        summary_file,
        "input                : {}",
        cli.input.display()
    )?;
    writeln!(summary_file, "period [s]           : {:.6}", cli.period)?;
    if let Some(dm) = cli.dm {
        writeln!(summary_file, "DM [pc cm^-3]        : {:.3}", dm)?;
    } else {
        writeln!(summary_file, "DM [pc cm^-3]        : (not applied)")?;
    }
    writeln!(
        summary_file,
        "channels             : {}",
        dedispersed.raw_spectrum.len()
    )?;
    writeln!(
        summary_file,
        "integration time [s] : {:.3}",
        dedispersed.total_integration
    )?;
    writeln!(
        summary_file,
        "sectors used (PP)    : {}",
        dedispersed.dedispersed_time_series.len()
    )?;
    let mean_eff = if !dedispersed.pp_durations.is_empty() {
        dedispersed.total_integration / dedispersed.pp_durations.len() as f64
    } else {
        0.0
    };
    if let (Some(&last_start), Some(&last_dur)) = (
        dedispersed.pp_elapsed.last(),
        dedispersed.pp_durations.last(),
    ) {
        writeln!(
            summary_file,
            "observation span [s] : {:.6}",
            last_start + last_dur
        )?;
    }
    writeln!(summary_file, "mean effective length [s]: {:.6}", mean_eff)?;
    if let (Some(min_delay), Some(max_delay)) = (dedispersed.dm_delay_min, dedispersed.dm_delay_max)
    {
        let span = max_delay - min_delay;
        let mean_pp = dedispersed.pp_durations.iter().copied().sum::<f64>()
            / dedispersed.pp_durations.len().max(1) as f64;
        writeln!(
            summary_file,
            "DM delay range [s]     : {:.6} .. {:.6} ({:.3}% of mean PP)",
            min_delay,
            max_delay,
            if mean_pp > 0.0 {
                span / mean_pp * 100.0
            } else {
                0.0
            }
        )?;
    }
    writeln!(summary_file, "fold bins            : {}", folded.len())?;
    writeln!(summary_file, "on-duty fraction     : {:.3}", cli.on_duty)?;
    writeln!(
        summary_file,
        "peak phase           : {:.4}",
        gating.peak_phase
    )?;
    writeln!(summary_file, "estimated S/N        : {:.2}", gating.snr)?;
    writeln!(summary_file, "on-pulse bins        : {:?}", gating.on_bins)?;
    writeln!(summary_file, "off-pulse bins       : {:?}", gating.off_bins)?;
    writeln!(
        summary_file,
        "note                 : gated profile equals folded profile (use folded output)"
    )?;
    if let Some(gated_data) = gated {
        writeln!(
            summary_file,
            "gating on-weight [s]  : {:.6}",
            gated_data.on_weight
        )?;
        writeln!(
            summary_file,
            "gating off-weight [s] : {:.6}",
            gated_data.off_weight
        )?;
        writeln!(
            summary_file,
            "gating on-mean amp    : {:.6}",
            gated_data.on_mean
        )?;
        writeln!(
            summary_file,
            "gating off-mean amp   : {:.6}",
            gated_data.off_mean
        )?;
        if let Some(sigma) = gated_data.off_sigma {
            writeln!(summary_file, "gating off σ         : {:.6}", sigma)?;
        }
        if let Some(snr) = gated_data.time_snr {
            writeln!(summary_file, "gating time-series S/N: {:.2}", snr)?;
        }
        if let Some(snr) = gated_data.gated_profile_snr {
            writeln!(summary_file, "gated profile S/N   : {:.2}", snr)?;
        }
        if let Some(sigma) = gated_data.gated_profile_sigma {
            writeln!(summary_file, "gated profile σ     : {:.6}", sigma)?;
        }
    }
    writeln!(
        summary_file,
        "station1             : {}",
        header.station1_name
    )?;
    writeln!(
        summary_file,
        "station2             : {}",
        header.station2_name
    )?;
    writeln!(
        summary_file,
        "observing frequency [MHz]: {:.3}",
        header.observing_frequency / 1.0e6
    )?;
    writeln!(
        summary_file,
        "bandwidth [MHz]          : {:.3}",
        header.sampling_speed as f64 / 2.0 / 1.0e6
    )?;
    writeln!(
        summary_file,
        "raw heatmap size        : {} rows x {} channels",
        raw_rows, raw_cols
    )?;
    writeln!(
        summary_file,
        "dedispersed heatmap size: {} rows x {} channels",
        dedisp_rows, dedisp_cols
    )?;

    Ok(())
}

fn cleanup_legacy_bin_outputs(output_dir: &Path, stem: &str) {
    let legacy_png_files = [
        format!("{stem}_raw_heatmap.png"),
        format!("{stem}_dedispersed_heatmap.png"),
        format!("{stem}_gated_diff_heatmap.png"),
    ];
    for name in legacy_png_files {
        let path = output_dir.join(name);
        let _ = fs::remove_file(path);
    }

    let legacy_files = [
        format!("{stem}_raw_phase_heatmap.bin"),
        format!("{stem}_dedispersed_heatmap.bin"),
        format!("{stem}_phase_aligned_heatmap.bin"),
        format!("{stem}_phase_aligned_onminusoff_heatmap.bin"),
        format!("{stem}_gated_diff_heatmap.bin"),
    ];
    for name in legacy_files {
        let path = output_dir.join(name);
        let _ = fs::remove_file(path);
    }

    let legacy_or_full_csv = [
        format!("{stem}_gated_spectrum_on.csv"),
        format!("{stem}_gated_spectrum_off.csv"),
        format!("{stem}_gated_time_series.csv"),
        format!("{stem}_gated_time_series_diff.csv"),
        format!("{stem}_dedispersed_time_series.csv"),
    ];
    for name in legacy_or_full_csv {
        let path = output_dir.join(name);
        let _ = fs::remove_file(path);
    }
}

fn print_summary(
    header: &CorHeader,
    cli: &KnownArgs,
    dedispersed: &DedispersionOutputs,
    folded: &[(f64, f64)],
    gating: &GatingResult,
    gated: Option<&GatedAggregation>,
) -> Result<()> {
    println!("Input file       : {}", cli.input.display());
    println!(
        "Stations         : {} - {}",
        header.station1_name, header.station2_name
    );
    println!(
        "Observing freq   : {:.3} MHz",
        header.observing_frequency / 1.0e6
    );
    println!(
        "Bandwidth        : {:.3} MHz",
        header.sampling_speed as f64 / 2.0 / 1.0e6
    );
    println!(
        "Sectors (PP)     : {}",
        dedispersed.dedispersed_time_series.len()
    );
    println!("Observation time : {:.6} s", dedispersed.total_integration);
    println!("Channels         : {}", dedispersed.raw_spectrum.len());
    println!("Phase bins       : {}", cli.bins);
    println!("Phase bin width  : {:.6} s", cli.period / cli.bins as f64);
    if let Some(dm) = cli.dm {
        println!("DM [pc cm^-3]    : {:.6}", dm);
    } else {
        println!("DM [pc cm^-3]    : (not applied)");
    }
    if let (Some(&last_start), Some(&last_dur)) = (
        dedispersed.pp_elapsed.last(),
        dedispersed.pp_durations.last(),
    ) {
        println!("Time span        : {:.6} s", last_start + last_dur);
    }
    if !dedispersed.pp_durations.is_empty() {
        let mean_eff = dedispersed.total_integration / dedispersed.pp_durations.len() as f64;
        println!("Mean PP length   : {:.6} s", mean_eff);
    }
    if let (Some(min_delay), Some(max_delay)) = (dedispersed.dm_delay_min, dedispersed.dm_delay_max)
    {
        let span = max_delay - min_delay;
        let mean_pp = dedispersed.pp_durations.iter().copied().sum::<f64>()
            / dedispersed.pp_durations.len().max(1) as f64;
        println!(
            "DM delay range    : {:.6} s .. {:.6} s ({:.3}% of mean PP)",
            min_delay,
            max_delay,
            if mean_pp > 0.0 {
                span / mean_pp * 100.0
            } else {
                0.0
            }
        );
        println!("DM delay span    : {:.6} ms", span * 1e3);
        println!(
            "DM delay span     = {:.3} phase bins ({:.3} deg)",
            if cli.bins > 0 {
                span / (cli.period / cli.bins as f64)
            } else {
                0.0
            },
            if cli.period > 0.0 {
                span / cli.period * 360.0
            } else {
                0.0
            }
        );
    }
    println!("Fold bins        : {}", folded.len());
    println!("Peak phase       : {:.4}", gating.peak_phase);
    println!("Estimated S/N    : {:.2}", gating.snr);
    println!("Gating note      : folded and gating profiles are identical; refer to folded output");
    if let Some(gated_data) = gated {
        println!("Gated on-weight  : {:.6} s", gated_data.on_weight);
        println!("Gated off-weight : {:.6} s", gated_data.off_weight);
        println!("Gated on-mean    : {:.6}", gated_data.on_mean);
        println!("Gated off-mean   : {:.6}", gated_data.off_mean);
        if let Some(sigma) = gated_data.off_sigma {
            println!("Gated off σ      : {:.6}", sigma);
        }
        if let Some(snr) = gated_data.time_snr {
            println!("Gated time S/N   : {:.2}", snr);
        }
        if let Some(snr) = gated_data.gated_profile_snr {
            println!("Gated profile S/N: {:.2}", snr);
        }
        if let Some(sigma) = gated_data.gated_profile_sigma {
            println!("Gated profile σ  : {:.6}", sigma);
        }
    }
    Ok(())
}

fn plot_folded_profile(
    output_path: &Path,
    data: &[(f64, f64)],
    gating: &GatingResult,
    cli: &KnownArgs,
) -> Result<()> {
    if data.len() < 2 {
        return Ok(());
    }

    let mut peak_amp = 0.0f64;
    for &idx in &gating.on_bins {
        if let Some((_, amp)) = data.get(idx) {
            peak_amp = peak_amp.max(*amp);
        }
    }
    let off_stats: Option<(f64, f64)> = if gating.off_bins.len() >= 2 {
        let values: Vec<f64> = gating
            .off_bins
            .iter()
            .filter_map(|&idx| data.get(idx).map(|&(_, amp)| amp))
            .collect();
        if values.len() >= 2 {
            let mean = values.iter().copied().sum::<f64>() / values.len() as f64;
            let var = values
                .iter()
                .map(|&v| {
                    let diff = v - mean;
                    diff * diff
                })
                .sum::<f64>()
                / (values.len() - 1) as f64;
            Some((mean, var.max(0.0).sqrt()))
        } else {
            None
        }
    } else {
        None
    };
    let (off_mean, off_sigma) = off_stats.unwrap_or((0.0, 0.0));
    let snr_display = gating.snr;
    let pulse_width_bin = gating.on_bins.len();
    let pulse_width_sec = if cli.bins > 0 {
        cli.period * pulse_width_bin as f64 / cli.bins as f64
    } else {
        0.0
    };

    let (x_min, x_max) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &(x, _)| {
            (mn.min(x), mx.max(x))
        });
    let (y_min, y_max) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &(_, y)| {
            (mn.min(y), mx.max(y))
        });

    let x_range = (x_max - x_min).abs().max(1e-9);
    let y_range = (y_max - y_min).abs().max(1e-9);

    let root = BitMapBackend::new(output_path, (850, 550)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(80)
        .y_label_area_size(110)
        .build_cartesian_2d(
            (x_min - 0.05 * x_range)..(x_max + 0.05 * x_range),
            (y_min - 0.05 * y_range)..(y_max + 0.05 * y_range),
        )?;

    chart
        .configure_mesh()
        .x_desc("Pulse phase")
        .y_desc("Amplitude")
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .light_line_style(TRANSPARENT)
        .label_style(("sans-serif", scaled_font_size(20)).into_font())
        .axis_desc_style(("sans-serif", scaled_font_size(22)).into_font())
        .draw()?;

    chart
        .draw_series(LineSeries::new(data.iter().map(|&(x, y)| (x, y)), &BLUE))?
        .label("Folded amplitude")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], BLUE));

    let dm_text = cli
        .dm
        .map(|dm| format!("{:.5} pc cm^-3", dm))
        .unwrap_or_else(|| "n/a".to_string());
    let info_lines = vec![
        format!("Peak amplitude : {:.3e}", peak_amp),
        format!("Off mean       : {:.3e}", off_mean),
        format!("Off sigma      : {:.3e}", off_sigma),
        format!("S/N            : {:.3}", snr_display),
        format!("Period         : {:.3e} s", cli.period),
        format!("DM             : {}", dm_text),
        format!(
            "On-window width(est.) : {:.3e} s ({:.1} bins)",
            pulse_width_sec, pulse_width_bin
        ),
    ];

    draw_info_block(&root, &info_lines)?;

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", scaled_legend_font_size(16)).into_font())
        .background_style(WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    compress_plot_png(output_path);
    Ok(())
}

fn plot_gated_profile(
    output_path: &Path,
    data: &[(f64, f64)],
    gating: &GatingResult,
    cli: &KnownArgs,
    off_mean: f64,
    off_sigma: Option<f64>,
    snr: Option<f64>,
    sigma: Option<f64>,
) -> Result<()> {
    if data.len() < 2 {
        return Ok(());
    }

    let (x_min, x_max) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &(x, _)| {
            (mn.min(x), mx.max(x))
        });
    let (y_min, y_max) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &(_, y)| {
            (mn.min(y), mx.max(y))
        });
    let x_range = (x_max - x_min).abs().max(1e-9);
    let y_range = (y_max - y_min).abs().max(1e-9);

    let root = BitMapBackend::new(output_path, (850, 550)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(80)
        .y_label_area_size(110)
        .build_cartesian_2d(
            (x_min - 0.05 * x_range)..(x_max + 0.05 * x_range),
            (y_min - 0.05 * y_range)..(y_max + 0.05 * y_range),
        )?;

    chart
        .configure_mesh()
        .x_desc("Pulse phase")
        .y_desc("Amplitude (on-pulse)")
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .light_line_style(TRANSPARENT)
        .label_style(("sans-serif", scaled_font_size(20)).into_font())
        .axis_desc_style(("sans-serif", scaled_font_size(22)).into_font())
        .draw()?;

    chart
        .draw_series(LineSeries::new(data.iter().map(|&(x, y)| (x, y)), &RED))?
        .label("Gated profile")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], RED));

    let peak_amp = gating
        .on_bins
        .iter()
        .filter_map(|&idx| data.get(idx).map(|&(_, amp)| amp))
        .fold(0.0, f64::max);
    let off_sigma_val = off_sigma.unwrap_or(0.0);
    let snr_text = snr
        .map(|v| format!("{:.3}", v))
        .unwrap_or_else(|| "n/a".to_string());
    let sigma_text = sigma
        .map(|v| format!("{:.3e}", v))
        .unwrap_or_else(|| "n/a".to_string());
    let dm_text = cli
        .dm
        .map(|dm| format!("{:.5} pc cm^-3", dm))
        .unwrap_or_else(|| "n/a".to_string());
    let info_lines = vec![
        format!("Peak (on-off)  : {:.3e}", peak_amp),
        format!("Off mean       : {:.3e}", off_mean),
        format!("Off sigma      : {:.3e}", off_sigma_val),
        format!("Gated S/N      : {}", snr_text),
        format!("Gated σ        : {}", sigma_text),
        format!("Period         : {:.3e} s", cli.period),
        format!("DM             : {}", dm_text),
        format!(
            "On-window width(est.) : {:.3e} s ({:.1} bins)",
            if cli.bins > 0 {
                cli.period * gating.on_bins.len() as f64 / cli.bins as f64
            } else {
                0.0
            },
            gating.on_bins.len()
        ),
    ];

    draw_info_block(&root, &info_lines)?;

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", scaled_legend_font_size(16)).into_font())
        .background_style(WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    compress_plot_png(output_path);
    Ok(())
}

fn plot_time_series(
    output_path: &Path,
    data: &[(f64, f64)],
    period: f64,
    _title: &str,
    y_label: &str,
) -> Result<()> {
    if data.len() < 2 {
        return Ok(());
    }

    let (x_min, x_max) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &(x, _)| {
            (mn.min(x), mx.max(x))
        });
    let (y_min, y_max) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &(_, y)| {
            (mn.min(y), mx.max(y))
        });
    let x_range = (x_max - x_min).abs().max(1e-9);
    let y_range = (y_max - y_min).abs().max(1e-9);

    let root = BitMapBackend::new(output_path, (850, 550)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(80)
        .y_label_area_size(110)
        .build_cartesian_2d(
            (x_min - 0.05 * x_range)..(x_max + 0.05 * x_range),
            (y_min - 0.05 * y_range)..(y_max + 0.05 * y_range),
        )?;

    chart
        .configure_mesh()
        .x_desc("Time [s]")
        .y_desc(y_label)
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .light_line_style(TRANSPARENT)
        .label_style(("sans-serif", scaled_font_size(20)).into_font())
        .axis_desc_style(("sans-serif", scaled_font_size(22)).into_font())
        .draw()?;

    chart
        .draw_series(LineSeries::new(data.iter().copied(), &BLUE))?
        .label("Time series")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], BLUE));

    let info_lines = [
        format!("Total span    : {:.3e} s", x_max - x_min),
        format!("Period        : {:.3e} s", period),
        format!("Samples count : {}", data.len()),
    ];
    let text_style =
        TextStyle::from(("sans-serif", scaled_font_size(20)).into_font()).color(&BLACK);
    let x_text = x_min + 0.02 * x_range;
    let mut y_cursor = y_max - 0.05 * y_range;
    let line_spacing = 0.04 * y_range;
    for line in info_lines.iter() {
        chart.plotting_area().draw(&Text::new(
            line.as_str(),
            (x_text, y_cursor),
            text_style.clone(),
        ))?;
        y_cursor -= line_spacing;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", scaled_legend_font_size(16)).into_font())
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    compress_plot_png(output_path);
    Ok(())
}

fn write_series_csv(
    path: &Path,
    x_header: &str,
    y_header: &str,
    data: &[(f64, f64)],
) -> Result<()> {
    let mut file = fs::File::create(path).with_context(|| format!("failed to write {:?}", path))?;
    writeln!(file, "{x_header},{y_header}")?;
    for (x, y) in data {
        writeln!(file, "{x:.6},{y:.6}")?;
    }
    Ok(())
}

fn write_spectrum_csv(path: &Path, data: &[(f64, f64)], value_header: &str) -> Result<()> {
    write_series_csv(path, "freq_mhz", value_header, data)
}

fn write_gated_time_series_csv(path: &Path, data: &[(f64, f64, bool)]) -> Result<()> {
    let mut file = fs::File::create(path).with_context(|| format!("failed to write {:?}", path))?;
    writeln!(file, "time_center_s,amplitude,is_on")?;
    for (time, amp, is_on) in data {
        let flag = if *is_on { 1 } else { 0 };
        writeln!(file, "{time:.9},{amp:.6},{flag}")?;
    }
    Ok(())
}

fn plot_gated_spectrum(
    output_path: &Path,
    on: &[(f64, f64)],
    off: &[(f64, f64)],
    diff: &[(f64, f64)],
) -> Result<()> {
    if on.len() < 2 || off.len() != on.len() || diff.len() != on.len() {
        return Ok(());
    }
    let freq_min = on.iter().map(|(f, _)| *f).fold(f64::INFINITY, f64::min);
    let freq_max = on.iter().map(|(f, _)| *f).fold(f64::NEG_INFINITY, f64::max);
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for series in [&on[..], &off[..], &diff[..]] {
        for &(_, amp) in series {
            y_min = y_min.min(amp);
            y_max = y_max.max(amp);
        }
    }
    if !freq_min.is_finite() || !freq_max.is_finite() || !y_min.is_finite() || !y_max.is_finite() {
        return Ok(());
    }
    if (y_max - y_min).abs() < 1e-9 {
        y_min -= 0.5;
        y_max += 0.5;
    }
    let y_range = (y_max - y_min).abs().max(1e-9);

    let root = BitMapBackend::new(output_path, (850, 550)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(80)
        .y_label_area_size(110)
        .build_cartesian_2d(
            freq_min..freq_max,
            (y_min - 0.05 * y_range)..(y_max + 0.05 * y_range),
        )?;

    chart
        .configure_mesh()
        .x_desc("Frequency [MHz]")
        .y_desc("Amplitude")
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .light_line_style(TRANSPARENT)
        .label_style(("sans-serif", scaled_font_size(20)).into_font())
        .axis_desc_style(("sans-serif", scaled_font_size(22)).into_font())
        .draw()?;

    chart
        .draw_series(LineSeries::new(on.iter().copied(), &BLUE))?
        .label("On-pulse")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], &BLUE));

    if off.iter().any(|&(_, amp)| amp.is_finite()) {
        chart
            .draw_series(LineSeries::new(off.iter().copied(), &GREEN))?
            .label("Off-pulse")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], &GREEN));
    }

    chart
        .draw_series(LineSeries::new(diff.iter().copied(), &RED))?
        .label("On-Off")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], &RED));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", scaled_legend_font_size(16)).into_font())
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    compress_plot_png(output_path);
    Ok(())
}

fn plot_gated_time_series(output_path: &Path, data: &[(f64, f64, bool)]) -> Result<()> {
    if data.len() < 2 {
        return Ok(());
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let t_min = sorted
        .iter()
        .map(|(t, _, _)| *t)
        .fold(f64::INFINITY, f64::min);
    let t_max = sorted
        .iter()
        .map(|(t, _, _)| *t)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for (_, amp, _) in &sorted {
        y_min = y_min.min(*amp);
        y_max = y_max.max(*amp);
    }
    if !t_min.is_finite() || !t_max.is_finite() || !y_min.is_finite() || !y_max.is_finite() {
        return Ok(());
    }
    if (y_max - y_min).abs() < 1e-9 {
        y_min -= 0.5;
        y_max += 0.5;
    }
    let y_range = (y_max - y_min).abs().max(1e-9);

    let root = BitMapBackend::new(output_path, (850, 550)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(80)
        .y_label_area_size(110)
        .build_cartesian_2d(
            t_min..t_max,
            (y_min - 0.05 * y_range)..(y_max + 0.05 * y_range),
        )?;

    chart
        .configure_mesh()
        .x_desc("Time [s]")
        .y_desc("Amplitude")
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .light_line_style(TRANSPARENT)
        .label_style(("sans-serif", scaled_font_size(20)).into_font())
        .axis_desc_style(("sans-serif", scaled_font_size(22)).into_font())
        .draw()?;

    let mut current_segment: Vec<(f64, f64)> = Vec::new();
    let mut current_flag = sorted[0].2;
    let mut on_label_drawn = false;
    let mut off_label_drawn = false;
    for &(t, amp, is_on) in &sorted {
        if is_on != current_flag && !current_segment.is_empty() {
            let color = if current_flag {
                RED.mix(0.8)
            } else {
                BLUE.mix(0.6)
            };
            let mut series = chart.draw_series(std::iter::once(PathElement::new(
                current_segment.clone(),
                color.stroke_width(1),
            )))?;
            if current_flag && !on_label_drawn {
                series = series.label("On-pulse").legend(|(x, y)| {
                    PathElement::new(vec![(x, y), (x + 25, y)], RED.mix(0.8).stroke_width(1))
                });
                on_label_drawn = true;
            } else if !current_flag && !off_label_drawn {
                series = series.label("Off-pulse").legend(|(x, y)| {
                    PathElement::new(vec![(x, y), (x + 25, y)], BLUE.mix(0.6).stroke_width(1))
                });
                off_label_drawn = true;
            }
            let _ = series;
            current_segment.clear();
        }
        current_segment.push((t, amp));
        current_flag = is_on;
    }
    if !current_segment.is_empty() {
        let color = if current_flag {
            RED.mix(0.8)
        } else {
            BLUE.mix(0.6)
        };
        let mut series = chart.draw_series(std::iter::once(PathElement::new(
            current_segment,
            color.stroke_width(1),
        )))?;
        if current_flag && !on_label_drawn {
            series = series.label("On-pulse").legend(|(x, y)| {
                PathElement::new(vec![(x, y), (x + 25, y)], RED.mix(0.8).stroke_width(1))
            });
        } else if !current_flag && !off_label_drawn {
            series = series.label("Off-pulse").legend(|(x, y)| {
                PathElement::new(vec![(x, y), (x + 25, y)], BLUE.mix(0.6).stroke_width(1))
            });
        }
        let _ = series;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", scaled_legend_font_size(16)).into_font())
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    compress_plot_png(output_path);
    Ok(())
}

fn heatmap_difference_stats(
    raw: &[Vec<Complex<f32>>],
    dedispersed: &[Vec<Complex<f32>>],
) -> (f64, f64) {
    let mut max_diff = 0.0f64;
    let mut accum = 0.0f64;
    let mut count = 0usize;
    for (row_raw, row_dedisp) in raw.iter().zip(dedispersed.iter()) {
        for (cell_raw, cell_dedisp) in row_raw.iter().zip(row_dedisp.iter()) {
            let diff = (cell_dedisp.norm() - cell_raw.norm()).abs() as f64;
            max_diff = max_diff.max(diff);
            accum += diff;
            count += 1;
        }
    }
    let mean_diff = if count > 0 { accum / count as f64 } else { 0.0 };
    (max_diff, mean_diff)
}

fn plot_phase_aligned_heatmap(
    output_path: &Path,
    heatmap: &[Vec<Complex<f32>>],
    freq_axis_mhz: &[f64],
    _center_freq_mhz: f64,
) -> Result<()> {
    if heatmap.is_empty()
        || heatmap[0].is_empty()
        || freq_axis_mhz.is_empty()
        || heatmap[0].len() != freq_axis_mhz.len()
    {
        return Ok(());
    }

    let bins = heatmap.len();
    let channels = heatmap[0].len();
    if bins == 0 || channels == 0 {
        return Ok(());
    }

    let mut amplitudes = vec![vec![0.0f32; channels]; bins];
    let mut min_amp = f32::MAX;
    let mut max_amp = f32::MIN;
    for (bin_idx, row) in heatmap.iter().enumerate() {
        for chan_idx in 0..channels {
            let amp = row[chan_idx].norm();
            amplitudes[bin_idx][chan_idx] = amp;
            min_amp = min_amp.min(amp);
            max_amp = max_amp.max(amp);
        }
    }
    if !min_amp.is_finite() || !max_amp.is_finite() {
        return Ok(());
    }
    if (max_amp - min_amp).abs() < f32::EPSILON {
        max_amp = min_amp + 1.0;
    }

    let mut freq_edges = Vec::with_capacity(channels + 1);
    for chan_idx in 0..channels {
        let left = if chan_idx == 0 {
            if channels > 1 {
                let step = freq_axis_mhz[1] - freq_axis_mhz[0];
                freq_axis_mhz[0] - step / 2.0
            } else {
                freq_axis_mhz[0] - 0.5
            }
        } else {
            (freq_axis_mhz[chan_idx - 1] + freq_axis_mhz[chan_idx]) / 2.0
        };
        freq_edges.push(left);
    }
    let last_edge_original = if channels > 1 {
        let step = freq_axis_mhz[channels - 1] - freq_axis_mhz[channels - 2];
        freq_axis_mhz[channels - 1] + step / 2.0
    } else {
        freq_axis_mhz[0] + 0.5
    };
    freq_edges.push(last_edge_original);

    let base_freq = freq_edges[0];
    for edge in &mut freq_edges {
        *edge -= base_freq;
    }
    let last_edge = freq_edges.last().copied().unwrap_or(0.0);

    let phase_step = 360.0 / bins as f64;
    let mut phase_edges = Vec::with_capacity(bins + 1);
    for bin_idx in 0..=bins {
        phase_edges.push(bin_idx as f64 * phase_step);
    }

    let root = BitMapBackend::new(output_path, (850, 550)).into_drawing_area();
    root.fill(&WHITE)?;

    let freq_min = 0.0;
    let freq_max = last_edge;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(80)
        .y_label_area_size(110)
        .build_cartesian_2d(freq_min..freq_max, 0.0..360.0)?;

    chart
        .configure_mesh()
        .x_desc("Frequency [MHz]")
        .y_desc("Pulse phase [deg]")
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.0}", v))
        .y_label_offset(6)
        .x_label_style(("sans-serif", scaled_font_size(24)).into_font())
        .y_label_style(("sans-serif", scaled_font_size(24)).into_font())
        .axis_desc_style(("sans-serif", scaled_font_size(22)).into_font())
        .light_line_style(TRANSPARENT)
        .draw()?;

    for bin_idx in 0..bins {
        let phase_low = phase_edges[bin_idx];
        let phase_high = phase_edges[bin_idx + 1];
        for chan_idx in 0..channels {
            let freq_low = freq_edges[chan_idx];
            let freq_high = freq_edges[chan_idx + 1];
            let amp = amplitudes[bin_idx][chan_idx];
            let norm = ((amp - min_amp) / (max_amp - min_amp)).clamp(0.0, 1.0);
            let color = ViridisRGB.get_color(norm as f64);
            chart.draw_series(std::iter::once(Rectangle::new(
                [(freq_low, phase_low), (freq_high, phase_high)],
                color.filled(),
            )))?;
        }
    }

    root.present()?;
    compress_plot_png(output_path);
    Ok(())
}

fn build_on_pulse_phase_difference_heatmap(
    dedispersed_heatmap: &[Vec<Complex<f32>>],
    pp_elapsed: &[f64],
    pp_durations: &[f64],
    period: f64,
    bins: usize,
    gating: &GatingResult,
) -> Vec<Vec<Complex<f32>>> {
    if dedispersed_heatmap.is_empty()
        || dedispersed_heatmap[0].is_empty()
        || bins == 0
        || period <= 0.0
        || gating.on_bins.is_empty()
    {
        return Vec::new();
    }

    let sectors = dedispersed_heatmap.len();
    let channels = dedispersed_heatmap[0].len();
    if pp_elapsed.len() != sectors || pp_durations.len() != sectors {
        return Vec::new();
    }

    let mut centers = Vec::with_capacity(sectors);
    for idx in 0..sectors {
        let start = pp_elapsed.get(idx).copied().unwrap_or(0.0);
        let duration = pp_durations.get(idx).copied().unwrap_or(0.0);
        centers.push(start + duration / 2.0);
    }
    if centers.is_empty() {
        return Vec::new();
    }

    let first_center = centers[0];
    let mut on_mask = vec![false; bins];
    for &bin in &gating.on_bins {
        if bin < bins {
            on_mask[bin] = true;
        }
    }
    if !on_mask.iter().any(|&v| v) {
        return Vec::new();
    }

    let mut on_sums = vec![vec![0.0f64; channels]; bins];
    let mut on_weights = vec![vec![0.0f64; channels]; bins];
    let mut off_sums = vec![0.0f64; channels];
    let mut off_weights = vec![0.0f64; channels];

    for sector_idx in 0..sectors {
        let duration = pp_durations
            .get(sector_idx)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        if duration <= 0.0 {
            continue;
        }
        let center = centers.get(sector_idx).copied().unwrap_or(first_center);
        let phase = ((center - first_center) / period).rem_euclid(1.0);
        let bin = ((phase * bins as f64).floor() as usize).min(bins - 1);
        let is_on = on_mask[bin];
        for chan_idx in 0..channels {
            let amp = dedispersed_heatmap[sector_idx][chan_idx].norm() as f64;
            if is_on {
                on_sums[bin][chan_idx] += amp * duration;
                on_weights[bin][chan_idx] += duration;
            } else {
                off_sums[chan_idx] += amp * duration;
                off_weights[chan_idx] += duration;
            }
        }
    }

    let mut off_means = vec![0.0f64; channels];
    for chan_idx in 0..channels {
        let weight = off_weights[chan_idx];
        if weight > 0.0 {
            off_means[chan_idx] = off_sums[chan_idx] / weight;
        }
    }

    let mut result = vec![vec![Complex::new(0.0f32, 0.0f32); channels]; bins];
    for bin_idx in 0..bins {
        if !on_mask[bin_idx] {
            continue;
        }
        for chan_idx in 0..channels {
            let weight = on_weights[bin_idx][chan_idx];
            if weight <= 0.0 {
                continue;
            }
            let on_mean = on_sums[bin_idx][chan_idx] / weight;
            let diff = (on_mean - off_means[chan_idx]).max(0.0);
            result[bin_idx][chan_idx] = Complex::new(diff as f32, 0.0);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedispersion_shifts_low_frequency_power_earlier() {
        let sectors = vec![
            SectorData {
                integ_time: 1.0,
                spectra: vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
            },
            SectorData {
                integ_time: 1.0,
                spectra: vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            },
        ];
        let freq_axis_mhz = vec![100.0, 200.0];
        let cli = KnownArgs {
            input: PathBuf::new(),
            period: 1.0,
            dm: Some(3212.0),
            bins: 8,
            skip: 0,
            length: 0,
            on_duty: 0.1,
            full_output: false,
        };
        let outputs = build_dedispersed_series(&sectors, &freq_axis_mhz, &cli).unwrap();

        let raw_heatmap_amp = outputs.spectra_heatmap[0][1].norm();
        let dedisp_heatmap_amp = outputs.dedispersed_heatmap[0][1].norm();
        assert!(
            dedisp_heatmap_amp < raw_heatmap_amp,
            "DM 適用によって高周波チャネルが先頭ビンから移動していません"
        );

        assert_eq!(outputs.raw_phase_heatmap.len(), cli.bins);
        assert!(
            outputs
                .raw_phase_heatmap
                .iter()
                .flat_map(|row| row.iter())
                .map(|c| c.norm())
                .sum::<f32>()
                > 0.0
        );

        assert_eq!(outputs.dedispersed_phase_heatmap.len(), cli.bins);
        assert!(
            outputs
                .dedispersed_phase_heatmap
                .iter()
                .flat_map(|row| row.iter())
                .map(|c| c.norm())
                .sum::<f32>()
                > 0.0
        );
    }
}
