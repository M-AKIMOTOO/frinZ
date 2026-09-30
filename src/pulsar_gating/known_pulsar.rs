#![allow(clippy::too_many_arguments)]

use super::shared::{
    compress_plot_png, detect_rfi_cut, dispersion_delays, finite_segments, interpolate_amplitude,
    output_stem, phase_bin, prepare_output_directory, scaled_font_size, scaled_legend_font_size,
    write_rfi_cut_report, RfiCutReport,
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
    validate_known_args(&cli)?;

    let buffer = fs::read(&cli.input)
        .with_context(|| format!("failed to read input file {}", cli.input.display()))?;
    let mut cursor = Cursor::new(buffer.as_slice());

    let header = parse_header(&mut cursor)?;

    let mut sectors = load_sectors(&mut cursor, &header, &cli)?;
    if sectors.is_empty() {
        return Err(anyhow!("no sectors were read from the file"));
    }

    let freq_axis_mhz = build_frequency_axis_mhz(&header);
    let rfi_report = apply_rfi_cut_to_sectors(&mut sectors);
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
        &rfi_report,
    )?;

    print_summary(
        &header,
        &cli,
        &dedisp_outputs,
        &folded,
        &gating,
        gated.as_ref(),
        &rfi_report,
    )?;
    Ok(())
}

fn validate_known_args(cli: &KnownArgs) -> Result<()> {
    if !cli.period.is_finite() || cli.period <= 0.0 {
        return Err(anyhow!("--period must be a finite positive value"));
    }
    if let Some(dm) = cli.dm {
        if !dm.is_finite() || dm < 0.0 {
            return Err(anyhow!("--dm must be a finite non-negative value"));
        }
    }
    if cli.bins < 3 {
        return Err(anyhow!(
            "--bins must be at least 3 (one on-pulse and two off-pulse bins)"
        ));
    }
    if !(0.0..=1.0).contains(&cli.on_duty) {
        return Err(anyhow!("--on-duty must be within [0, 1]"));
    }
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
        let (mut spectra, _timestamp, _integ) =
            read_visibility_data(cursor, header, 1, 0, loop_index as i32, false, &[])
                .with_context(|| format!("failed to read sector {}", loop_index))?;

        if spectra.is_empty() {
            continue;
        }
        // Preserve the measured cadence: the general fringe reader rounds near
        // powers of ten, which would accumulate pulsar phase errors over time.
        let sector_size = 128 + header.fft_point as usize * 4;
        let offset = 256 + loop_index as usize * sector_size + 112;
        let bytes = cursor
            .get_ref()
            .get(offset..offset + 4)
            .ok_or_else(|| anyhow!("missing sector integration time"))?;
        let integ = f32::from_le_bytes(bytes.try_into().expect("four bytes")) as f64;
        if !integ.is_finite() || integ <= 0.0 {
            return Err(anyhow!("invalid integration time in sector {}", loop_index));
        }
        // The generic reader sanitizes invalid visibilities to zero. Restore
        // missingness here so corrupt samples cannot enter pulsar statistics.
        let payload_offset = 256 + loop_index as usize * sector_size + 128;
        for (chan, value) in spectra.iter_mut().enumerate() {
            let start = payload_offset + chan * 8;
            let bytes = &cursor.get_ref()[start..start + 8];
            let re = f32::from_le_bytes(bytes[..4].try_into().expect("four bytes"));
            let im = f32::from_le_bytes(bytes[4..].try_into().expect("four bytes"));
            if !re.is_finite() || !im.is_finite() {
                *value = Complex::new(f32::NAN, f32::NAN);
            }
        }
        sectors.push(SectorData {
            integ_time: integ,
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

pub(crate) fn apply_rfi_cut_to_sectors(sectors: &mut [SectorData]) -> RfiCutReport {
    let rows: Vec<&[Complex<f32>]> = sectors
        .iter()
        .map(|sector| sector.spectra.as_slice())
        .collect();
    let report = detect_rfi_cut(&rows);
    if report.masked_channels.is_empty() {
        return report;
    }

    let mut masked = vec![false; report.total_channels];
    for &chan_idx in &report.masked_channels {
        if chan_idx < masked.len() {
            masked[chan_idx] = true;
        }
    }

    for sector in sectors {
        for (chan_idx, value) in sector.spectra.iter_mut().enumerate() {
            if masked.get(chan_idx).copied().unwrap_or(false) {
                *value = Complex::new(0.0, 0.0);
            }
        }
    }

    report
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
    let channels = freq_axis_mhz.len();
    if channels == 0 || sectors.is_empty() {
        return Err(anyhow!("empty spectra or time series"));
    }
    if freq_axis_mhz.iter().any(|f| !f.is_finite() || *f <= 0.0) {
        return Err(anyhow!(
            "dedispersion requires finite positive channel frequencies"
        ));
    }
    let mut elapsed = 0.0;
    let mut pp_elapsed = Vec::with_capacity(sectors.len());
    let mut pp_durations = Vec::with_capacity(sectors.len());
    let mut centers = Vec::with_capacity(sectors.len());
    let mut spectra_heatmap = Vec::with_capacity(sectors.len());
    let mut channel_series = vec![Vec::with_capacity(sectors.len()); channels];
    let mut raw_sums = vec![0.0; channels];
    let mut raw_weights = vec![0.0; channels];
    for sector in sectors {
        let duration = sector.integ_time;
        if !duration.is_finite() || duration <= 0.0 || sector.spectra.len() != channels {
            return Err(anyhow!(
                "invalid sector duration or inconsistent channel count"
            ));
        }
        pp_elapsed.push(elapsed);
        pp_durations.push(duration);
        let center = elapsed + duration / 2.0;
        centers.push(center);
        elapsed += duration;
        spectra_heatmap.push(sector.spectra.clone());
        for (chan, &value) in sector.spectra.iter().enumerate() {
            let amp = value.norm() as f64;
            channel_series[chan].push((center, amp));
            if amp.is_finite() {
                raw_sums[chan] += amp * duration;
                raw_weights[chan] += duration;
            }
        }
    }
    let delays = compute_dispersion_delays(freq_axis_mhz, freq_axis_mhz[0], cli.dm.unwrap_or(0.0));
    let mut dedispersed_heatmap = vec![vec![Complex::new(f32::NAN, 0.0); channels]; sectors.len()];
    let mut dedispersed_time_series = Vec::with_capacity(sectors.len());
    let mut dedispersed_weights = Vec::with_capacity(sectors.len());
    for (row, &center) in centers.iter().enumerate() {
        let mut sum = 0.0;
        let mut complete = true;
        for chan in 0..channels {
            if let Some(amp) = interpolate_amplitude(&channel_series[chan], center - delays[chan]) {
                // These are detected amplitudes, not complex visibilities. Interpolating
                // the latter before taking their norm would mix fringe phase into intensity.
                dedispersed_heatmap[row][chan] = Complex::new(amp as f32, 0.0);
                sum += amp;
            } else {
                complete = false;
            }
        }
        // Use the same full band at every valid time. Partial-band samples are missing,
        // never zero-filled or renormalized using a changing channel count.
        dedispersed_time_series.push((center, if complete { sum } else { f64::NAN }));
        dedispersed_weights.push(if complete { pp_durations[row] } else { 0.0 });
    }
    if !dedispersed_weights.iter().any(|w| *w > 0.0) {
        return Err(anyhow!("no full-band samples remain after dedispersion"));
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
        &dedispersed_weights,
        cli.bins,
        cli.period,
    );
    let raw_spectrum = freq_axis_mhz
        .iter()
        .enumerate()
        .map(|(chan, &freq)| {
            (
                freq,
                if raw_weights[chan] > 0.0 {
                    raw_sums[chan] / raw_weights[chan]
                } else {
                    f64::NAN
                },
            )
        })
        .collect();
    Ok(DedispersionOutputs {
        integrated_time_series: dedispersed_time_series.clone(),
        dedispersed_time_series,
        dedispersed_weights,
        raw_spectrum,
        spectra_heatmap,
        dedispersed_heatmap,
        raw_phase_heatmap,
        dedispersed_phase_heatmap,
        total_integration: elapsed,
        pp_elapsed,
        pp_durations,
        dm_delay_min: cli
            .dm
            .map(|_| delays.iter().copied().fold(f64::INFINITY, f64::min)),
        dm_delay_max: cli
            .dm
            .map(|_| delays.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
    })
}

fn compute_dispersion_delays(freq_axis_mhz: &[f64], ref_freq_mhz: f64, dm: f64) -> Vec<f64> {
    dispersion_delays(freq_axis_mhz, ref_freq_mhz, dm)
}

fn build_phase_aligned_heatmap(
    heatmap: &[Vec<Complex<f32>>],
    centers: &[f64],
    durations: &[f64],
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
    let mut weights = vec![vec![0.0f64; channels]; bins];
    let t0 = centers[0];

    for (row_idx, row) in heatmap.iter().enumerate() {
        if row.len() != channels {
            continue;
        }
        let duration = durations.get(row_idx).copied().unwrap_or(0.0);
        if !duration.is_finite() || duration <= 0.0 {
            continue;
        }
        let center = centers.get(row_idx).copied().unwrap_or(t0);
        let bin = phase_bin(center, t0, period, bins);
        for (chan_idx, value) in row.iter().enumerate() {
            let amp = value.norm() as f64;
            if amp.is_finite() {
                accum[bin][chan_idx] += amp * duration;
                weights[bin][chan_idx] += duration;
            }
        }
    }

    accum
        .into_iter()
        .zip(weights)
        .map(|(sums, weights)| {
            sums.into_iter()
                .zip(weights)
                .map(|(sum, weight)| {
                    Complex::new(
                        if weight > 0.0 {
                            (sum / weight) as f32
                        } else {
                            f32::NAN
                        },
                        0.0,
                    )
                })
                .collect()
        })
        .collect()
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
    if !period.is_finite() || period <= 0.0 || bins == 0 || weights.len() != dedispersed.len() {
        return Err(anyhow!("invalid period, bins, or time-series weight count"));
    }
    if dedispersed.iter().any(|(time, _)| !time.is_finite()) {
        return Err(anyhow!("non-finite sample time"));
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
        if !weight.is_finite() || weight <= 0.0 || !amp.is_finite() {
            continue;
        }
        let bin = phase_bin(time, t0, period, bins);
        accum[bin] += amp * weight;
        weight_sums[bin] += weight;
    }

    accum
        .iter()
        .zip(weight_sums.iter())
        .enumerate()
        .map(|(idx, (sum, w))| {
            let phase = (idx as f64 + 0.5) / bins as f64;
            let amp = if *w > 0.0 { sum / w } else { f64::NAN };
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
            off_bins: profile
                .iter()
                .enumerate()
                .filter_map(|(i, p)| p.1.is_finite().then_some(i))
                .collect(),
            peak_phase: 0.0,
            snr: 0.0,
        };
    }

    let bins = profile.len();
    let mut sorted: Vec<(usize, f64)> = profile
        .iter()
        .enumerate()
        .filter_map(|(idx, &(_, amp))| amp.is_finite().then_some((idx, amp)))
        .collect();
    if sorted.is_empty() {
        return GatingResult {
            on_bins: Vec::new(),
            off_bins: Vec::new(),
            peak_phase: 0.0,
            snr: 0.0,
        };
    }
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if sorted.len() < 3 {
        return GatingResult {
            on_bins: Vec::new(),
            off_bins: sorted.iter().map(|p| p.0).collect(),
            peak_phase: 0.0,
            snr: 0.0,
        };
    }
    let max_on_bins = sorted.len() - 2;
    let on_bin_count = ((sorted.len() as f64 * on_duty).ceil() as usize)
        .clamp(1, max_on_bins)
        .min(sorted.len());
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
    let off_bins: Vec<usize> = (0..bins)
        .filter(|idx| !is_on[*idx] && profile[*idx].1.is_finite())
        .collect();

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
    } else if peak_amp == off_mean {
        0.0
    } else {
        f64::NAN
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
    let first_center = dedispersed.pp_elapsed.first().copied().unwrap_or(0.0)
        + dedispersed.pp_durations.first().copied().unwrap_or(0.0) / 2.0;
    for (idx, elapsed) in dedispersed.pp_elapsed.iter().enumerate() {
        let duration = dedispersed.pp_durations.get(idx).copied().unwrap_or(0.0);
        let center = elapsed + duration / 2.0;
        let bin = phase_bin(center, first_center, period, bins);
        bin_assignments.push((center, bin));
    }

    let mut on_weight = 0.0f64;
    let mut off_weight = 0.0f64;
    let mut on_total = 0.0f64;
    let mut off_total = 0.0f64;
    let mut off_square_total = 0.0f64;
    let mut off_weight_square = 0.0f64;
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
        let weight = dedispersed.dedispersed_weights[sector_idx];
        if weight <= 0.0 {
            continue;
        }
        let (center, bin) = bin_assignments.get(sector_idx).copied().unwrap_or((0.0, 0));
        let is_on = on_mask[bin];

        let avg_amp = if duration > 0.0 {
            dedispersed
                .dedispersed_time_series
                .get(sector_idx)
                .map(|(_, amp)| *amp)
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
            off_weight_square += weight * weight;
        }
    }

    if on_weight == 0.0 || off_weight == 0.0 {
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
    let off_dof = off_weight - off_weight_square / off_weight;
    let off_sigma = if off_dof > 0.0 {
        Some(
            ((off_square_total - off_weight * off_mean.powi(2)) / off_dof)
                .max(0.0)
                .sqrt(),
        )
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
    let base_time = first_center;
    for &(center, avg_amp, is_on) in &time_series {
        if !is_on {
            continue;
        }
        let idx = dedispersed
            .dedispersed_time_series
            .partition_point(|&(t, _)| t < center);
        let duration = dedispersed.dedispersed_weights[idx];
        if duration <= 0.0 {
            continue;
        }
        let bin = phase_bin(center, base_time, period, bins);
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
                f64::NAN
            };
            (phase, value)
        })
        .collect();

    let mut diff_time_series = Vec::with_capacity(sectors);
    for (sector_idx, row) in dedispersed.dedispersed_heatmap.iter().enumerate() {
        if dedispersed.dedispersed_weights[sector_idx] <= 0.0 {
            continue;
        }
        let (center, _) = bin_assignments[sector_idx];
        let diff_sum = row
            .iter()
            .enumerate()
            .map(|(chan, value)| value.norm() as f64 - off_channel_means[chan])
            .sum();
        diff_time_series.push((center, diff_sum));
    }
    // Profile S/N uses profile-bin noise, rather than the largest individual
    // time sample divided by time-domain noise.
    let folded = fold_profile(
        &dedispersed.dedispersed_time_series,
        &dedispersed.dedispersed_weights,
        period,
        bins,
    )
    .ok()?;
    let off_values: Vec<_> = gating
        .off_bins
        .iter()
        .map(|&bin| folded[bin].1)
        .filter(|v| v.is_finite())
        .collect();
    let gated_sigma = if off_values.len() >= 2 {
        let mean = off_values.iter().sum::<f64>() / off_values.len() as f64;
        Some(
            (off_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (off_values.len() - 1) as f64)
                .sqrt(),
        )
    } else {
        None
    };
    let gated_snr = gated_sigma.filter(|sigma| *sigma > 0.0).map(|_| gating.snr);

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

#[derive(Debug)]
struct GatedVisibility {
    frequency_mhz: f64,
    on: Complex<f64>,
    off: Complex<f64>,
    on_weight: f64,
    off_weight: f64,
}

fn compute_gated_visibilities(
    data: &DedispersionOutputs,
    gating: &GatingResult,
    frequencies: &[f64],
    cli: &KnownArgs,
) -> Vec<GatedVisibility> {
    let centers: Vec<_> = data
        .pp_elapsed
        .iter()
        .zip(&data.pp_durations)
        .map(|(t, d)| t + d / 2.0)
        .collect();
    let valid: Vec<_> = centers
        .iter()
        .zip(&data.dedispersed_weights)
        .filter_map(|(&t, &w)| (w > 0.0).then_some(t))
        .collect();
    if valid.is_empty() || gating.on_bins.is_empty() || gating.off_bins.is_empty() {
        return Vec::new();
    }
    let delays = compute_dispersion_delays(frequencies, frequencies[0], cli.dm.unwrap_or(0.0));
    frequencies
        .iter()
        .enumerate()
        .map(|(chan, &frequency_mhz)| {
            let mut on = Complex::new(0.0, 0.0);
            let mut off = on;
            let mut on_weight = 0.0;
            let mut off_weight = 0.0;
            for (row, &center) in centers.iter().enumerate() {
                // Shift the gate, not the complex samples: raw fringe phase is preserved.
                let reference_time = center + delays[chan];
                if reference_time < valid[0] || reference_time > *valid.last().unwrap() {
                    continue;
                }
                let bin = phase_bin(reference_time, centers[0], cli.period, cli.bins);
                let value = data.spectra_heatmap[row][chan];
                if !value.re.is_finite() || !value.im.is_finite() {
                    continue;
                }
                let value = Complex::new(value.re as f64, value.im as f64);
                let weight = data.pp_durations[row];
                if gating.on_bins.contains(&bin) {
                    on += value * weight;
                    on_weight += weight;
                } else if gating.off_bins.contains(&bin) {
                    off += value * weight;
                    off_weight += weight;
                }
            }
            let missing = Complex::new(f64::NAN, f64::NAN);
            GatedVisibility {
                frequency_mhz,
                on: if on_weight > 0.0 {
                    on / on_weight
                } else {
                    missing
                },
                off: if off_weight > 0.0 {
                    off / off_weight
                } else {
                    missing
                },
                on_weight,
                off_weight,
            }
        })
        .collect()
}

fn write_gated_visibilities(path: &Path, values: &[GatedVisibility]) -> Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "channel,freq_mhz,on_re,on_im,off_re,off_im,diff_re,diff_im,on_weight_s,off_weight_s"
    )?;
    for (chan, value) in values.iter().enumerate() {
        let diff = value.on - value.off;
        writeln!(
            file,
            "{},{:.9},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.9},{:.9}",
            chan,
            value.frequency_mhz,
            value.on.re,
            value.on.im,
            value.off.re,
            value.off.im,
            diff.re,
            diff.im,
            value.on_weight,
            value.off_weight
        )?;
    }
    Ok(())
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
    rfi_report: &RfiCutReport,
) -> Result<()> {
    let stem_owned = output_stem(&cli.input);
    let stem = stem_owned.as_str();
    cleanup_legacy_bin_outputs(output_dir, stem);
    let full_output = cli.full_output;
    let raw_rows = dedispersed.spectra_heatmap.len();
    let raw_cols = dedispersed
        .spectra_heatmap
        .first()
        .map(|row| row.len())
        .unwrap_or(0);
    let dedisp_rows = dedispersed.dedispersed_heatmap.len();
    let dedisp_cols = dedispersed
        .dedispersed_heatmap
        .first()
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
    let visibilities = compute_gated_visibilities(dedispersed, gating, freq_axis_mhz, cli);
    write_gated_visibilities(
        &output_dir.join(format!("{stem}_gated_visibilities.csv")),
        &visibilities,
    )?;
    let profile_path = output_dir.join(format!("{stem}_profile.csv"));
    let mut profile_file = fs::File::create(&profile_path)
        .with_context(|| format!("failed to write {profile_path:?}"))?;
    let mut bin_exposure = vec![0.0; cli.bins];
    let origin = dedispersed.dedispersed_time_series[0].0;
    for (&(time, amp), &weight) in dedispersed
        .dedispersed_time_series
        .iter()
        .zip(&dedispersed.dedispersed_weights)
    {
        if amp.is_finite() && weight > 0.0 {
            bin_exposure[phase_bin(time, origin, cli.period, cli.bins)] += weight;
        }
    }
    writeln!(profile_file, "bin,phase,amplitude,exposure_s")?;
    for (idx, (phase, amp)) in folded.iter().enumerate() {
        writeln!(
            profile_file,
            "{idx},{phase:.6},{amp:.6},{:.9}",
            bin_exposure[idx]
        )?;
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

    if !dedispersed.raw_phase_heatmap.is_empty() {
        let phase_heatmap_path = output_dir.join(format!("{stem}_phase_freq_before_gating.png"));
        plot_phase_aligned_heatmap(
            &phase_heatmap_path,
            &dedispersed.raw_phase_heatmap,
            freq_axis_mhz,
            center_freq_mhz,
            if cli.dm.is_some() {
                "Phase vs Frequency (before DM / before gating)"
            } else {
                "Phase vs Frequency (before gating)"
            },
        )?;
    }
    if cli.dm.is_some() && !dedispersed.dedispersed_phase_heatmap.is_empty() {
        let heatmap_path = output_dir.join(format!("{stem}_phase_freq_after_dm_before_gating.png"));
        plot_phase_aligned_heatmap(
            &heatmap_path,
            &dedispersed.dedispersed_phase_heatmap,
            freq_axis_mhz,
            center_freq_mhz,
            "Phase vs Frequency (after DM / before gating)",
        )?;
    }

    let on_diff_heatmap = build_on_pulse_phase_difference_heatmap(
        &dedispersed.dedispersed_heatmap,
        &dedispersed.pp_elapsed,
        &dedispersed.pp_durations,
        cli.period,
        cli.bins,
        gating,
    );
    if !on_diff_heatmap.is_empty() {
        let diff_heatmap_path = output_dir.join(format!("{stem}_phase_freq_after_gating.png"));
        plot_phase_aligned_heatmap(
            &diff_heatmap_path,
            &on_diff_heatmap,
            freq_axis_mhz,
            center_freq_mhz,
            if cli.dm.is_some() {
                "Phase vs Frequency (after DM and gating: on - off)"
            } else {
                "Phase vs Frequency (after gating: on - off)"
            },
        )?;
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

    let rfi_path = output_dir.join(format!("{stem}_rfi_cut.csv"));
    write_rfi_cut_report(&rfi_path, freq_axis_mhz, rfi_report)?;

    let summary_path = output_dir.join(format!("{stem}_summary.txt"));
    let mut summary_file = fs::File::create(&summary_path)
        .with_context(|| format!("failed to write {summary_path:?}"))?;
    writeln!(summary_file, "# Pulsar gating summary")?;
    writeln!(
        summary_file,
        "valid full-band [s]  : {:.9}",
        dedispersed.dedispersed_weights.iter().sum::<f64>()
    )?;
    writeln!(
        summary_file,
        "observed phase bins : {} / {}",
        folded.iter().filter(|p| p.1.is_finite()).count(),
        folded.len()
    )?;
    writeln!(
        summary_file,
        "amplitude processing : detected-amplitude interpolation; missing bins are NaN"
    )?;
    writeln!(
        summary_file,
        "complex product      : {}_gated_visibilities.csv (original fringe phase retained)",
        stem
    )?;
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
        "RFI masked channels  : {} / {}",
        rfi_report.masked_count(),
        rfi_report.total_channels
    )?;
    writeln!(
        summary_file,
        "RFI cut params       : window=+-{}, sigma>{:.1}, ratio>={:.1}",
        rfi_report.window_radius, rfi_report.sigma_cut, rfi_report.ratio_cut
    )?;
    writeln!(
        summary_file,
        "RFI masked indices   : {}",
        summarize_channel_indices(&rfi_report.masked_channels)
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
        format!("{stem}_phase_freq_before_dm.png"),
        format!("{stem}_phase_freq_after_dm.png"),
        format!("{stem}_phase_freq_before_gating.png"),
        format!("{stem}_phase_freq_after_dm_before_gating.png"),
        format!("{stem}_phase_freq_after_gating.png"),
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
    rfi_report: &RfiCutReport,
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
    println!(
        "RFI masked ch     : {} / {}",
        rfi_report.masked_count(),
        rfi_report.total_channels
    );
    println!(
        "RFI cut params    : window=+-{}, sigma>{:.1}, ratio>={:.1}",
        rfi_report.window_radius, rfi_report.sigma_cut, rfi_report.ratio_cut
    );
    if rfi_report.masked_count() > 0 {
        println!(
            "RFI masked idx    : {}",
            summarize_channel_indices(&rfi_report.masked_channels)
        );
    }
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
    println!(
        "Valid full-band  : {:.6} s",
        dedispersed.dedispersed_weights.iter().sum::<f64>()
    );
    println!(
        "Observed bins    : {} / {}",
        folded.iter().filter(|p| p.1.is_finite()).count(),
        folded.len()
    );
    println!("Fold bins        : {}", folded.len());
    println!("Peak phase       : {:.4}", gating.peak_phase);
    println!("Estimated S/N    : {:.2}", gating.snr);
    println!("Gating note      : amplitudes are exposure-weighted; missing phase bins are NaN");
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

fn summarize_channel_indices(indices: &[usize]) -> String {
    const MAX_DISPLAY: usize = 24;
    if indices.is_empty() {
        return "(none)".to_string();
    }
    if indices.len() <= MAX_DISPLAY {
        return format!("{indices:?}");
    }
    let mut parts = indices[..MAX_DISPLAY]
        .iter()
        .map(|idx| idx.to_string())
        .collect::<Vec<_>>();
    parts.push(format!("... (+{} more)", indices.len() - MAX_DISPLAY));
    format!("[{}]", parts.join(", "))
}

fn plot_folded_profile(
    output_path: &Path,
    data: &[(f64, f64)],
    gating: &GatingResult,
    cli: &KnownArgs,
) -> Result<()> {
    if data.len() < 2 || !data.iter().any(|p| p.1.is_finite()) {
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
        .draw_series(
            finite_segments(data)
                .into_iter()
                .map(|segment| PathElement::new(segment, BLUE)),
        )?
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
        .border_style(BLACK)
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
    if data.len() < 2 || !data.iter().any(|p| p.1.is_finite()) {
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
        .draw_series(
            finite_segments(data)
                .into_iter()
                .map(|segment| PathElement::new(segment, RED)),
        )?
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
        .border_style(BLACK)
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
    if data.len() < 2 || !data.iter().any(|p| p.1.is_finite()) {
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
        .draw_series(
            finite_segments(data)
                .into_iter()
                .map(|segment| PathElement::new(segment, BLUE)),
        )?
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
    for series in [on, off, diff] {
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
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], BLUE));

    if off.iter().any(|&(_, amp)| amp.is_finite()) {
        chart
            .draw_series(LineSeries::new(off.iter().copied(), &GREEN))?
            .label("Off-pulse")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], GREEN));
    }

    chart
        .draw_series(LineSeries::new(diff.iter().copied(), &RED))?
        .label("On-Off")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], RED));

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
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
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
            if !diff.is_finite() {
                continue;
            }
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
    title: &str,
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

    let mut amplitudes = vec![vec![f32::NAN; channels]; bins];
    let mut min_amp = f32::MAX;
    let mut max_amp = f32::MIN;
    for (bin_idx, row) in heatmap.iter().enumerate() {
        for chan_idx in 0..channels {
            let amp = row[chan_idx].re;
            if !amp.is_finite() {
                continue;
            }
            amplitudes[bin_idx][chan_idx] = amp;
            min_amp = min_amp.min(amp);
            max_amp = max_amp.max(amp);
        }
    }
    if min_amp == f32::MAX || max_amp == f32::MIN || !min_amp.is_finite() || !max_amp.is_finite() {
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
        .caption(title, ("sans-serif", scaled_font_size(22)).into_font())
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
            if !amp.is_finite() {
                continue;
            }
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

    for ((duration, center), row) in pp_durations
        .iter()
        .zip(centers.iter())
        .zip(dedispersed_heatmap.iter())
    {
        let duration = (*duration).max(0.0);
        if row.iter().any(|v| !v.norm().is_finite()) {
            continue;
        }
        if duration <= 0.0 {
            continue;
        }
        let bin = phase_bin(*center, first_center, period, bins);
        let is_on = on_mask[bin];
        for (chan_idx, cell) in row.iter().enumerate().take(channels) {
            let amp = cell.norm() as f64;
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

    let mut result = vec![vec![Complex::new(f32::NAN, 0.0f32); channels]; bins];
    for bin_idx in 0..bins {
        if !on_mask[bin_idx] {
            continue;
        }
        for chan_idx in 0..channels {
            let weight = on_weights[bin_idx][chan_idx];
            if weight <= 0.0 {
                continue;
            }
            if off_weights[chan_idx] <= 0.0 {
                continue;
            }
            let on_mean = on_sums[bin_idx][chan_idx] / weight;
            let diff = on_mean - off_means[chan_idx];
            result[bin_idx][chan_idx] = Complex::new(diff as f32, 0.0);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args(dm: Option<f64>, period: f64, bins: usize) -> KnownArgs {
        KnownArgs {
            input: PathBuf::new(),
            period,
            dm,
            bins,
            skip: 0,
            length: 0,
            on_duty: 0.1,
            full_output: false,
        }
    }

    #[test]
    fn sector_loader_preserves_measured_cadence_and_missing_visibility() {
        let header = CorHeader {
            magic_word: [0x83, 0xf9, 0xa2, 0x3e],
            sampling_speed: 32_000_000,
            observing_frequency: 316e6,
            fft_point: 4,
            number_of_sector: 1,
            ..CorHeader::default()
        };
        let mut bytes = vec![0u8; 256 + 128 + 16];
        bytes[256 + 112..256 + 116].copy_from_slice(&0.0105f32.to_le_bytes());
        bytes[256 + 128..256 + 132].copy_from_slice(&f32::NAN.to_le_bytes());
        let sectors =
            load_sectors_with_limits(&mut Cursor::new(bytes.as_slice()), &header, 0, 0).unwrap();
        assert_eq!(sectors[0].integ_time, 0.0105f32 as f64);
        assert!(sectors[0].spectra[0].re.is_nan());
        bytes[256 + 112..256 + 116].copy_from_slice(&0.0f32.to_le_bytes());
        assert!(
            load_sectors_with_limits(&mut Cursor::new(bytes.as_slice()), &header, 0, 0).is_err()
        );
    }

    #[test]
    fn dedispersion_aligns_early_high_frequency_pulse_to_low_frequency_reference() {
        let mut sectors: Vec<_> = (0..4)
            .map(|_| SectorData {
                integ_time: 1.0,
                spectra: vec![Complex::new(0.0, 0.0); 2],
            })
            .collect();
        sectors[1].spectra[1] = Complex::new(1.0, 0.0);
        sectors[2].spectra[0] = Complex::new(1.0, 0.0);
        let dm = 1.0 / (4.148_808e3 * (100.0f64.powi(-2) - 200.0f64.powi(-2)));
        let out = build_dedispersed_series(&sectors, &[100.0, 200.0], &test_args(Some(dm), 4.0, 4))
            .unwrap();
        assert!((out.integrated_time_series[2].1 - 2.0).abs() < 1e-6);
        assert!(!out.integrated_time_series[0].1.is_finite());
        assert_eq!(out.dedispersed_weights[0], 0.0);
    }

    #[test]
    fn zero_dm_is_identity_including_both_endpoints_and_single_sample() {
        for count in [1, 4] {
            let sectors: Vec<_> = (0..count)
                .map(|_| SectorData {
                    integ_time: 1.0,
                    spectra: vec![Complex::new(1.0, 0.0); 2],
                })
                .collect();
            let raw = build_dedispersed_series(&sectors, &[100.0, 200.0], &test_args(None, 4.0, 4))
                .unwrap();
            let dm0 =
                build_dedispersed_series(&sectors, &[100.0, 200.0], &test_args(Some(0.0), 4.0, 4))
                    .unwrap();
            assert_eq!(raw.integrated_time_series, dm0.integrated_time_series);
            assert!(dm0.integrated_time_series.iter().all(|p| p.1 == 2.0));
            assert_eq!(raw.dedispersed_weights, dm0.dedispersed_weights);
        }
    }

    #[test]
    fn constant_amplitude_fold_is_independent_of_duration() {
        let sectors: Vec<_> = [1.0, 2.0]
            .iter()
            .map(|&d| SectorData {
                integ_time: d,
                spectra: vec![Complex::new(1.0, 0.0)],
            })
            .collect();
        let out = build_dedispersed_series(&sectors, &[100.0], &test_args(None, 100.0, 4)).unwrap();
        let folded = fold_profile(
            &out.dedispersed_time_series,
            &out.dedispersed_weights,
            100.0,
            4,
        )
        .unwrap();
        assert_eq!(folded[0].1, 1.0);
    }

    #[test]
    fn empty_bins_are_missing_and_cannot_manufacture_signal_to_noise() {
        let series: Vec<_> = (0..10).map(|i| (i as f64, 1.0)).collect();
        let profile = fold_profile(&series, &[1.0; 10], 10.0, 128).unwrap();
        assert_eq!(profile.iter().filter(|p| p.1.is_finite()).count(), 10);
        let gate = determine_gating(&profile, 0.05);
        assert_eq!(gate.snr, 0.0);
        assert!(gate
            .on_bins
            .iter()
            .chain(&gate.off_bins)
            .all(|&i| profile[i].1.is_finite()));
    }

    #[test]
    fn phase_reversal_does_not_attenuate_detected_amplitude() {
        let sectors = vec![
            SectorData {
                integ_time: 1.0,
                spectra: vec![Complex::new(1.0, 0.0); 2],
            },
            SectorData {
                integ_time: 1.0,
                spectra: vec![Complex::new(-1.0, 0.0); 2],
            },
            SectorData {
                integ_time: 1.0,
                spectra: vec![Complex::new(1.0, 0.0); 2],
            },
        ];
        let dm = 0.5 / (4.148_808e3 * (100.0f64.powi(-2) - 200.0f64.powi(-2)));
        let out = build_dedispersed_series(&sectors, &[100.0, 200.0], &test_args(Some(dm), 3.0, 4))
            .unwrap();
        assert_eq!(out.integrated_time_series[1].1, 2.0);
    }

    #[test]
    fn gated_aggregation_ignores_partial_band_and_preserves_amplitude_scale() {
        let sectors: Vec<_> = (0..7)
            .map(|_| SectorData {
                integ_time: 1.0,
                spectra: vec![Complex::new(1.0, 0.0); 2],
            })
            .collect();
        let dm = 0.5 / (4.148_808e3 * (100.0f64.powi(-2) - 200.0f64.powi(-2)));
        let out = build_dedispersed_series(&sectors, &[100.0, 200.0], &test_args(Some(dm), 3.0, 3))
            .unwrap();
        let gate = GatingResult {
            on_bins: vec![0],
            off_bins: vec![1, 2],
            peak_phase: 0.0,
            snr: 0.0,
        };
        let agg = compute_gated_aggregation(&out, &gate, &[100.0, 200.0], 3.0, 3).unwrap();
        assert_eq!(agg.on_mean, 2.0);
        assert_eq!(agg.off_mean, 2.0);
        assert!(agg.diff_spectrum.iter().all(|p| p.1 == 0.0));
        assert_eq!(agg.on_weight + agg.off_weight, 6.0);
    }

    #[test]
    fn complex_gating_preserves_phase_and_uses_channel_dispersion_time() {
        let sectors: Vec<_> = (0..6)
            .map(|i| SectorData {
                integ_time: 1.0,
                spectra: vec![
                    Complex::new(0.0, if i % 3 == 0 { 3.0 } else { 1.0 }),
                    Complex::new(0.0, if i % 3 == 2 { 3.0 } else { 1.0 }),
                ],
            })
            .collect();
        let dm = 1.0 / (4.148_808e3 * (100.0f64.powi(-2) - 200.0f64.powi(-2)));
        let cli = test_args(Some(dm), 3.0, 3);
        let out = build_dedispersed_series(&sectors, &[100.0, 200.0], &cli).unwrap();
        let gate = GatingResult {
            on_bins: vec![0],
            off_bins: vec![1, 2],
            peak_phase: 0.0,
            snr: 0.0,
        };
        let vis = compute_gated_visibilities(&out, &gate, &[100.0, 200.0], &cli);
        for v in vis {
            assert_eq!(v.on, Complex::new(0.0, 3.0));
            assert_eq!(v.off, Complex::new(0.0, 1.0));
        }
    }

    #[test]
    fn fold_rejects_invalid_parameters_and_skips_nonfinite_amplitude() {
        assert!(fold_profile(&[(0.0, 1.0)], &[], 1.0, 4).is_err());
        assert!(fold_profile(&[(0.0, 1.0)], &[1.0], f64::NAN, 4).is_err());
        let p = fold_profile(&[(0.0, f64::NAN), (1.0, 1.0)], &[1.0, 1.0], 4.0, 4).unwrap();
        assert!(p[0].1.is_nan());
        assert_eq!(p[1].1, 1.0);
    }

    #[test]
    fn known_args_require_finite_period_and_dm() {
        let base = KnownArgs {
            input: PathBuf::new(),
            period: 1.0,
            dm: Some(10.0),
            bins: 8,
            skip: 0,
            length: 0,
            on_duty: 0.1,
            full_output: false,
        };

        let mut invalid_period = base.clone();
        invalid_period.period = f64::NAN;
        assert!(validate_known_args(&invalid_period).is_err());

        let mut invalid_dm = base;
        invalid_dm.dm = Some(f64::INFINITY);
        assert!(validate_known_args(&invalid_dm).is_err());
    }

    #[test]
    fn determine_gating_reserves_two_observed_off_bins_even_at_full_duty() {
        let profile = vec![(0.1, 1.0), (0.3, f64::NAN), (0.5, 2.0), (0.9, 3.0)];
        let gating = determine_gating(&profile, 1.0);
        assert_eq!(gating.on_bins, vec![3]);
        assert_eq!(gating.off_bins, vec![0, 2]);
        assert_eq!(gating.peak_phase, 0.9);
    }

    #[test]
    fn apply_rfi_cut_masks_narrowband_spike_channel() {
        let mut sectors = vec![
            SectorData {
                integ_time: 1.0,
                spectra: vec![
                    Complex::new(1.0, 0.0),
                    Complex::new(1.0, 0.0),
                    Complex::new(25.0, 0.0),
                    Complex::new(1.0, 0.0),
                    Complex::new(1.0, 0.0),
                    Complex::new(1.0, 0.0),
                ],
            },
            SectorData {
                integ_time: 1.0,
                spectra: vec![
                    Complex::new(1.0, 0.0),
                    Complex::new(1.1, 0.0),
                    Complex::new(24.0, 0.0),
                    Complex::new(1.0, 0.0),
                    Complex::new(1.0, 0.0),
                    Complex::new(1.0, 0.0),
                ],
            },
        ];

        let report = apply_rfi_cut_to_sectors(&mut sectors);

        assert_eq!(report.masked_channels, vec![2]);
        assert_eq!(sectors[0].spectra[2], Complex::new(0.0, 0.0));
        assert_eq!(sectors[1].spectra[2], Complex::new(0.0, 0.0));
    }
}
