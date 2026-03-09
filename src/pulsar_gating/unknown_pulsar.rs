use super::known_pulsar::{
    self, build_frequency_axis_mhz, fold_profile, load_sectors_with_limits, KnownArgs, SectorData,
};
use super::shared::{
    compress_plot_png, output_stem, prepare_output_directory, scaled_font_size,
    scaled_legend_font_size,
};
use anyhow::{anyhow, Result};
use frinZ::fft::{process_fft, process_ifft};
use frinZ::header::parse_header;
use ndarray::{Array2, Axis};
use num_complex::Complex;
use plotters::prelude::*;
use std::fs;
use std::io::Cursor;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct UnknownArgs {
    pub input: PathBuf,
    pub bins: usize,
    pub skip: u32,
    pub length: u32,
    pub on_duty: f64,
    pub amp_threshold: f64,
    pub full_output: bool,
}

pub fn run(args: UnknownArgs) -> Result<()> {
    if args.bins == 0 {
        return Err(anyhow!("--bins must be greater than 0"));
    }
    if !(0.0..=1.0).contains(&args.on_duty) {
        return Err(anyhow!("--on-duty must be within [0, 1]"));
    }

    let UnknownArgs {
        input,
        bins,
        skip,
        length,
        on_duty,
        amp_threshold,
        full_output,
    } = args;
    if !amp_threshold.is_finite() || amp_threshold < 0.0 {
        return Err(anyhow!(
            "--amp-threshold must be a non-negative finite value"
        ));
    }

    let buffer = fs::read(&input)
        .map_err(|e| anyhow!("failed to read input file {}: {e}", input.display()))?;
    let mut cursor = Cursor::new(buffer.as_slice());

    let header = parse_header(&mut cursor)?;
    let sectors = load_sectors_with_limits(&mut cursor, &header, skip, length)?;
    if sectors.len() < 4 {
        return Err(anyhow!(
            "at least 4 sectors are required to estimate pulsar parameters"
        ));
    }

    let freq_axis_mhz = build_frequency_axis_mhz(&header);
    if freq_axis_mhz.is_empty() {
        return Err(anyhow!("frequency axis is empty"));
    }

    let samples_per_sector = freq_axis_mhz.len();
    let (channel_series, durations) = build_time_series(&sectors, samples_per_sector);
    let (coherent_series, coherent_weights) =
        build_coherent_time_series(&sectors, samples_per_sector);
    let output_dir = prepare_output_directory(&input)?;
    let stem = output_stem(&input);
    cleanup_legacy_unknown_gated_outputs(&output_dir, &stem);

    const RATE_PADDING: u32 = 1;
    let fringe = compute_rate_spectrum(
        &sectors,
        samples_per_sector,
        header.fft_point,
        header.sampling_speed,
        RATE_PADDING,
    )?;
    let rate_spectrum_complex = fringe.rate_spectrum.clone();
    let sample_dt = durations.iter().copied().sum::<f64>() / durations.len().max(1) as f64;
    let (rate_axis_raw, rate_profile_raw) =
        extract_rate_profile_from_delay_rate(&fringe.delay_rate_spectrum, sample_dt)
            .unwrap_or_else(|| {
                let axis = generate_rate_axis(rate_spectrum_complex.len(), sample_dt);
                let prof = rate_spectrum_complex
                    .iter()
                    .map(|z| z.norm() as f64)
                    .collect::<Vec<_>>();
                (axis, prof)
            });
    let rate_bin_hz = estimate_rate_bin_hz(&rate_axis_raw).unwrap_or(1e-12);
    let rate_above_amp_points =
        extract_rate_profile_above_amp(&rate_axis_raw, &rate_profile_raw, amp_threshold);
    let rate_fundamental =
        estimate_fundamental_from_rate_diffs(&rate_above_amp_points, rate_bin_hz);
    let periodic_points_raw = rate_fundamental
        .as_ref()
        .map(|est| {
            extract_periodic_rate_points(&rate_axis_raw, &rate_profile_raw, est.fundamental_rate_hz)
        })
        .unwrap_or_default();
    let periodic_points = rate_fundamental
        .as_ref()
        .map(|est| {
            filter_periodic_points_by_diffs(
                &periodic_points_raw,
                est.fundamental_rate_hz,
                rate_bin_hz,
            )
        })
        .unwrap_or_else(|| periodic_points_raw.clone());
    if let Some(est) = &rate_fundamental {
        println!(
            "Rate diff fundamental [Hz]: {:.6} (count {})",
            est.fundamental_rate_hz, est.fundamental_count
        );
        println!("Rate diff sigma [Hz]: {:.6}", est.fundamental_sigma_hz);
        println!("Estimated period from rate diff [s]: {:.9}", est.period_s);
    } else {
        println!("Rate diff fundamental [Hz]: n/a (count 0)");
        println!("Rate diff sigma [Hz]: n/a");
        println!("Estimated period from rate diff [s]: n/a");
    }
    let selected_period = rate_fundamental
        .as_ref()
        .map(|est| est.period_s)
        .filter(|p| p.is_finite() && *p > 0.0)
        .ok_or_else(|| anyhow!("failed to estimate period from rate-diff peaks"))?;
    println!(
        "Selected period [s] (source=rate-diff) : {:.9}",
        selected_period
    );
    let refined_period = refine_period_by_fold_snr(
        &coherent_series,
        &coherent_weights,
        selected_period,
        bins,
        on_duty,
    )?
    .unwrap_or(selected_period);
    if (refined_period - selected_period).abs() > 0.0 {
        println!(
            "Refined period [s] (fold-snr)   : {:.9} (delta {:+.9})",
            refined_period,
            refined_period - selected_period
        );
    }

    let dm_estimate = estimate_dispersion_measure(
        &channel_series,
        &durations,
        &freq_axis_mhz,
        refined_period,
        bins,
    )?;
    if let Some(est) = &dm_estimate {
        println!(
            "Estimated DM [pc cm^-3]: {:.6} (points {}, subbands {}, R^2 {:.3})",
            est.dm_pc_cm3,
            est.points.len(),
            est.subband_count,
            est.r2
        );
    } else {
        println!("Estimated DM [pc cm^-3]: n/a");
    }

    if let Some(est) = &dm_estimate {
        let dm_fit_csv = output_dir.join(format!("{stem}_dm_fit_points.csv"));
        write_dm_fit_points_csv(&dm_fit_csv, est)?;
    }

    let rate_plot = output_dir.join(format!("{stem}_rate_spectrum.png"));
    plot_rate_profile(
        &rate_plot,
        &rate_axis_raw,
        &rate_profile_raw,
        Some(&rate_above_amp_points),
        Some(&periodic_points),
    )?;
    let rate_above_amp_csv = output_dir.join(format!("{stem}_rate_spectrum_above_amp.csv"));
    write_rate_profile_above_amp_csv(&rate_above_amp_csv, &rate_above_amp_points)?;
    let periodic_csv = output_dir.join(format!("{stem}_rate_spectrum_periodic_peaks.csv"));
    write_rate_profile_above_amp_csv(&periodic_csv, &periodic_points)?;
    if full_output && rate_fundamental.is_some() {
        let folded_plot = output_dir.join(format!("{stem}_rate_diff_folded_profile.png"));
        let folded_profile =
            fold_profile(&coherent_series, &coherent_weights, refined_period, 128)?;
        plot_folded_profile_from_period(&folded_plot, &folded_profile, refined_period, None)?;
    }

    println!(
        "Rate spectrum above amp threshold ({:.6}): {}",
        amp_threshold,
        rate_above_amp_points.len()
    );
    println!(
        "Rate spectrum periodic peaks (fundamental spacing): {}",
        periodic_points.len()
    );
    let known_args = KnownArgs {
        input,
        period: refined_period,
        dm: dm_estimate.as_ref().map(|est| est.dm_pc_cm3),
        bins,
        skip,
        length,
        on_duty,
        full_output,
    };
    let dm_handoff_text = known_args
        .dm
        .map(|v| format!("{:.9}", v))
        .unwrap_or_else(|| "none".to_string());
    println!(
        "Known-mode handoff: period={:.9} s, dm={}, bins={}, skip={}, length={}, on-duty={:.6}",
        known_args.period,
        dm_handoff_text,
        known_args.bins,
        known_args.skip,
        known_args.length,
        known_args.on_duty
    );
    let handoff_path = output_dir.join(format!("{stem}_unknown_handoff.txt"));
    let mut handoff_file = fs::File::create(&handoff_path)
        .map_err(|e| anyhow!("failed to create {}: {e}", handoff_path.display()))?;
    writeln!(handoff_file, "period_s={:.12}", known_args.period)?;
    writeln!(handoff_file, "dm_pc_cm3={}", dm_handoff_text)?;
    writeln!(handoff_file, "bins={}", known_args.bins)?;
    writeln!(handoff_file, "skip={}", known_args.skip)?;
    writeln!(handoff_file, "length={}", known_args.length)?;
    writeln!(handoff_file, "on_duty={:.9}", known_args.on_duty)?;
    writeln!(
        handoff_file,
        "reproduce_cmd=cargo run --quiet --bin pulsar_gating -- --input {} --period {:.12} --bins {} --skip {} --length {} --on-duty {:.9}{}",
        known_args.input.display(),
        known_args.period,
        known_args.bins,
        known_args.skip,
        known_args.length,
        known_args.on_duty,
        known_args
            .dm
            .map(|v| format!(" --dm {:.12}", v))
            .unwrap_or_default()
    )?;
    known_pulsar::run(known_args)
}

fn build_time_series(
    sectors: &[SectorData],
    samples_per_sector: usize,
) -> (Vec<Vec<(f64, f64)>>, Vec<f64>) {
    let mut channel_series = vec![Vec::with_capacity(sectors.len()); samples_per_sector];
    let mut durations = Vec::with_capacity(sectors.len());
    let mut cumulative_time = 0.0f64;

    for sector in sectors {
        let duration = sector.integ_time.max(1e-9);
        let center = cumulative_time + duration / 2.0;
        cumulative_time += duration;
        durations.push(duration);

        for chan_idx in 0..samples_per_sector {
            let value = sector
                .spectra
                .get(chan_idx)
                .copied()
                .unwrap_or_else(|| Complex::new(0.0, 0.0));
            let amp = value.norm() as f64;
            channel_series[chan_idx].push((center, amp));
        }
    }

    (channel_series, durations)
}

fn cleanup_legacy_unknown_gated_outputs(output_dir: &Path, stem: &str) {
    let legacy_files = [
        format!("{stem}_rate_time_profile.png"),
        format!("{stem}_rate_spectrum_gated.png"),
        format!("{stem}_rate_time_profile_gated.png"),
        format!("{stem}_rate_spectrum_periodic_peaks_significance.csv"),
        format!("{stem}_rate_spectrum_periodic_peaks_raw.csv"),
        format!("{stem}_rate_spectrum_periodic_peaks_removed.csv"),
        format!("{stem}_rate_spectrum_diff_counts.csv"),
        format!("{stem}_delay_rate_peakscan.png"),
        format!("{stem}_delay_rate_peakscan_gated.png"),
        format!("{stem}_delay_rate_peakscan_gated.csv"),
        format!("{stem}_delay_rate_profiles_gated.csv"),
        format!("{stem}_delay_rate_peakscan.csv"),
        format!("{stem}_delay_rate_above7sigma_gated.csv"),
        format!("{stem}_delay_rate_above_amp_gated.csv"),
        format!("{stem}_delay_rate_above7sigma.csv"),
        format!("{stem}_delay_rate_above_amp.csv"),
        format!("{stem}_delay_rate_profiles.csv"),
    ];
    for name in legacy_files {
        let path = output_dir.join(name);
        let _ = fs::remove_file(path);
    }
    let legacy_by_delay_dir = output_dir.join(format!("{stem}_delay_rate_profiles_by_delay"));
    let _ = fs::remove_dir_all(legacy_by_delay_dir);
    let legacy_dir = output_dir.join(format!("{stem}_delay_rate_profiles_by_delay_gated"));
    let _ = fs::remove_dir_all(legacy_dir);
}

fn build_coherent_time_series(
    sectors: &[SectorData],
    samples_per_sector: usize,
) -> (Vec<(f64, f64)>, Vec<f64>) {
    let mut series = Vec::with_capacity(sectors.len());
    let mut weights = Vec::with_capacity(sectors.len());
    let mut cumulative_time = 0.0f64;

    for sector in sectors {
        let duration = sector.integ_time.max(1e-9);
        let center = cumulative_time + duration / 2.0;
        cumulative_time += duration;

        let mut coherent_sum = Complex::new(0.0f32, 0.0f32);
        let mut coherent_count = 0usize;
        for chan_idx in 0..samples_per_sector {
            let value = sector
                .spectra
                .get(chan_idx)
                .copied()
                .unwrap_or_else(|| Complex::new(0.0, 0.0));
            if value.re.is_finite() && value.im.is_finite() {
                coherent_sum += value;
                coherent_count += 1;
            }
        }
        let amp = if coherent_count > 0 {
            (coherent_sum / coherent_count as f32).norm() as f64
        } else {
            0.0
        };
        series.push((center, amp));
        weights.push(duration);
    }

    (series, weights)
}

fn refine_period_by_fold_snr(
    series: &[(f64, f64)],
    weights: &[f64],
    initial_period_s: f64,
    bins: usize,
    on_duty: f64,
) -> Result<Option<f64>> {
    if series.len() < 8
        || weights.len() != series.len()
        || !initial_period_s.is_finite()
        || initial_period_s <= 0.0
        || bins == 0
        || !on_duty.is_finite()
        || on_duty <= 0.0
        || on_duty > 1.0
    {
        return Ok(None);
    }

    let t_min = series.first().map(|v| v.0).unwrap_or(0.0);
    let t_max = series.last().map(|v| v.0).unwrap_or(t_min);
    let span = (t_max - t_min).abs().max(1e-9);
    let base_res = (initial_period_s * initial_period_s / span).abs();
    let coarse_half = (6.0 * base_res).clamp(2.0e-4, 1.0e-2);
    let coarse_steps = 321usize;

    let coarse = scan_period_snr_grid(
        series,
        weights,
        initial_period_s,
        coarse_half,
        coarse_steps,
        bins,
        on_duty,
    )?;
    let Some(best_coarse) = coarse else {
        return Ok(None);
    };

    let coarse_step = (2.0 * coarse_half) / (coarse_steps.saturating_sub(1) as f64);
    let fine_half = (coarse_step * 8.0).clamp(2.0e-6, coarse_half / 3.0);
    let fine_steps = 401usize;
    let fine = scan_period_snr_grid(
        series,
        weights,
        best_coarse.period_s,
        fine_half,
        fine_steps,
        bins,
        on_duty,
    )?;

    Ok(fine.map(|b| b.period_s).or(Some(best_coarse.period_s)))
}

fn scan_period_snr_grid(
    series: &[(f64, f64)],
    weights: &[f64],
    center_period_s: f64,
    half_span_s: f64,
    steps: usize,
    bins: usize,
    on_duty: f64,
) -> Result<Option<FoldPeriodScore>> {
    if steps < 3
        || !center_period_s.is_finite()
        || center_period_s <= 0.0
        || !half_span_s.is_finite()
        || half_span_s <= 0.0
    {
        return Ok(None);
    }

    let step = (2.0 * half_span_s) / (steps.saturating_sub(1) as f64);
    let mut best: Option<FoldPeriodScore> = None;
    for i in 0..steps {
        let trial = center_period_s - half_span_s + i as f64 * step;
        if !trial.is_finite() || trial <= 0.0 {
            continue;
        }
        let folded = fold_profile(series, weights, trial, bins)?;
        if let Some(snr) = score_folded_profile_snr(&folded, on_duty) {
            match best {
                Some(current) if current.snr >= snr => {}
                _ => {
                    best = Some(FoldPeriodScore {
                        period_s: trial,
                        snr,
                    });
                }
            }
        }
    }
    Ok(best)
}

fn score_folded_profile_snr(folded: &[(f64, f64)], on_duty: f64) -> Option<f64> {
    if folded.len() < 4 || !on_duty.is_finite() || on_duty <= 0.0 || on_duty > 1.0 {
        return None;
    }

    let bins = folded.len();
    let on_count = ((bins as f64 * on_duty).ceil() as usize).clamp(1, bins.saturating_sub(1));
    let mut ranked: Vec<(usize, f64)> = folded
        .iter()
        .enumerate()
        .filter_map(|(idx, &(_, amp))| {
            if amp.is_finite() {
                Some((idx, amp))
            } else {
                None
            }
        })
        .collect();
    if ranked.len() < 4 {
        return None;
    }
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut on_mask = vec![false; bins];
    for (idx, _) in ranked.iter().take(on_count) {
        on_mask[*idx] = true;
    }

    let mut off_vals = Vec::new();
    let mut peak_on = f64::NEG_INFINITY;
    for (idx, &(_, amp)) in folded.iter().enumerate() {
        if !amp.is_finite() {
            continue;
        }
        if on_mask[idx] {
            if amp > peak_on {
                peak_on = amp;
            }
        } else {
            off_vals.push(amp);
        }
    }
    if off_vals.len() < 2 || !peak_on.is_finite() {
        return None;
    }

    let off_mean = off_vals.iter().copied().sum::<f64>() / off_vals.len() as f64;
    let off_var = off_vals
        .iter()
        .map(|v| {
            let dv = *v - off_mean;
            dv * dv
        })
        .sum::<f64>()
        / (off_vals.len() - 1) as f64;
    let off_sigma = off_var.sqrt();
    if !off_sigma.is_finite() || off_sigma <= f64::EPSILON {
        return None;
    }

    Some((peak_on - off_mean) / off_sigma)
}

fn compute_rate_spectrum(
    sectors: &[SectorData],
    samples_per_sector: usize,
    fft_point: i32,
    sampling_speed: i32,
    rate_padding: u32,
) -> Result<FringeSpectra> {
    if sectors.is_empty() || samples_per_sector == 0 {
        return Err(anyhow!(
            "insufficient data to perform fringe search (need > 0 sectors and channels)"
        ));
    }
    let time_len = sectors.len();
    let mut combined = Vec::with_capacity(time_len * samples_per_sector);
    for sector in sectors {
        combined.extend_from_slice(&sector.spectra);
    }

    let (freq_rate_array, padding_length) = process_fft(
        &combined,
        time_len as i32,
        fft_point,
        sampling_speed,
        &[],
        rate_padding,
    );

    let mut rate_accum = vec![Complex::new(0.0f32, 0.0f32); padding_length];
    for row in freq_rate_array.axis_iter(Axis(0)) {
        for (idx, value) in row.iter().enumerate() {
            rate_accum[idx] += *value;
        }
    }
    let delay_rate_spectrum = process_ifft(&freq_rate_array, fft_point, padding_length);

    Ok(FringeSpectra {
        rate_spectrum: rate_accum,
        delay_rate_spectrum,
    })
}

struct FringeSpectra {
    rate_spectrum: Vec<Complex<f32>>,
    delay_rate_spectrum: Array2<Complex<f32>>,
}

#[derive(Debug, Clone, Copy)]
struct FoldPeriodScore {
    period_s: f64,
    snr: f64,
}

#[derive(Debug, Clone)]
struct RateFundamentalEstimate {
    fundamental_rate_hz: f64,
    fundamental_sigma_hz: f64,
    fundamental_count: usize,
    period_s: f64,
}

#[derive(Debug, Clone)]
struct RateProfileAboveAmpPoint {
    rate_hz: f64,
    amplitude: f64,
}

#[derive(Debug, Clone)]
struct DmFitPoint {
    subband_idx: usize,
    freq_center_mhz: f64,
    inv_freq2_mhz2: f64,
    delay_s: f64,
    phase_shift_cycles: f64,
    corr_peak: f64,
    residual_s: f64,
}

#[derive(Debug, Clone)]
struct DispersionEstimate {
    dm_pc_cm3: f64,
    slope_sec_per_invfreq2: f64,
    intercept_sec: f64,
    r2: f64,
    subband_count: usize,
    points: Vec<DmFitPoint>,
}

#[derive(Debug, Clone)]
struct SubbandFoldProfile {
    subband_idx: usize,
    freq_center_mhz: f64,
    peak_phase_cycles: f64,
    peak_amplitude: f64,
}

fn extract_rate_profile_above_amp(
    rate_axis: &[f64],
    profile: &[f64],
    amp_threshold: f64,
) -> Vec<RateProfileAboveAmpPoint> {
    if rate_axis.len() != profile.len() || !amp_threshold.is_finite() || amp_threshold < 0.0 {
        return Vec::new();
    }
    rate_axis
        .iter()
        .copied()
        .zip(profile.iter().copied())
        .filter_map(|(rate_hz, amplitude)| {
            if rate_hz.is_finite() && amplitude.is_finite() && amplitude >= amp_threshold {
                Some(RateProfileAboveAmpPoint { rate_hz, amplitude })
            } else {
                None
            }
        })
        .collect()
}

fn extract_periodic_rate_points(
    rate_axis: &[f64],
    profile: &[f64],
    fundamental_rate_hz: f64,
) -> Vec<RateProfileAboveAmpPoint> {
    if rate_axis.len() != profile.len()
        || rate_axis.is_empty()
        || !fundamental_rate_hz.is_finite()
        || fundamental_rate_hz <= 0.0
    {
        return Vec::new();
    }

    let step_hz = fundamental_rate_hz.abs();
    let mut min_rate = f64::INFINITY;
    let mut max_rate = f64::NEG_INFINITY;
    for (&rate_hz, &amplitude) in rate_axis.iter().zip(profile.iter()) {
        if rate_hz.is_finite() && amplitude.is_finite() {
            min_rate = min_rate.min(rate_hz);
            max_rate = max_rate.max(rate_hz);
        }
    }
    if !min_rate.is_finite() || !max_rate.is_finite() || min_rate > max_rate {
        return Vec::new();
    }
    let mut points = Vec::new();
    let mut left = min_rate;
    while left <= max_rate {
        let right = left + step_hz;
        let mut best_idx: Option<usize> = None;
        let mut best_amp = f64::NEG_INFINITY;
        for (idx, (&rate_hz, &amplitude)) in rate_axis.iter().zip(profile.iter()).enumerate() {
            if !rate_hz.is_finite() || !amplitude.is_finite() {
                continue;
            }
            let in_window = if right > max_rate {
                rate_hz >= left && rate_hz <= right
            } else {
                rate_hz >= left && rate_hz < right
            };
            if in_window && amplitude > best_amp {
                best_amp = amplitude;
                best_idx = Some(idx);
            }
        }
        if let Some(idx) = best_idx {
            points.push(RateProfileAboveAmpPoint {
                rate_hz: rate_axis[idx],
                amplitude: profile[idx],
            });
        }
        left = right;
    }

    points
}

fn write_rate_profile_above_amp_csv(
    path: &Path,
    points: &[RateProfileAboveAmpPoint],
) -> Result<()> {
    let mut file = fs::File::create(path).map_err(|e| {
        anyhow!(
            "failed to create rate-spectrum above-amp CSV {}: {e}",
            path.display()
        )
    })?;
    writeln!(file, "rate_hz,amplitude")?;
    for p in points {
        writeln!(file, "{:.9},{:.9}", p.rate_hz, p.amplitude)?;
    }
    Ok(())
}

fn write_dm_fit_points_csv(path: &Path, estimate: &DispersionEstimate) -> Result<()> {
    let mut file = fs::File::create(path)
        .map_err(|e| anyhow!("failed to create DM fit CSV {}: {e}", path.display()))?;
    writeln!(file, "dm_pc_cm3,{:.12}", estimate.dm_pc_cm3)?;
    writeln!(
        file,
        "slope_sec_per_invfreq2,{:.12}",
        estimate.slope_sec_per_invfreq2
    )?;
    writeln!(file, "intercept_sec,{:.12}", estimate.intercept_sec)?;
    writeln!(file, "r2,{:.9}", estimate.r2)?;
    writeln!(file, "subbands,{}", estimate.subband_count)?;
    writeln!(file)?;
    writeln!(
        file,
        "subband,freq_center_mhz,inv_freq2_mhz2,delay_s,phase_shift_cycles,corr_peak,residual_s"
    )?;
    for p in &estimate.points {
        writeln!(
            file,
            "{},{:.9},{:.12},{:.12},{:.9},{:.9},{:.12}",
            p.subband_idx,
            p.freq_center_mhz,
            p.inv_freq2_mhz2,
            p.delay_s,
            p.phase_shift_cycles,
            p.corr_peak,
            p.residual_s
        )?;
    }
    Ok(())
}

fn filter_periodic_points_by_diffs(
    points: &[RateProfileAboveAmpPoint],
    fundamental_rate_hz: f64,
    rate_bin_hz: f64,
) -> Vec<RateProfileAboveAmpPoint> {
    if points.len() < 3 || !fundamental_rate_hz.is_finite() || fundamental_rate_hz <= 0.0 {
        return points.to_vec();
    }
    let mut filtered = points.to_vec();
    filtered.sort_by(|a, b| {
        a.rate_hz
            .partial_cmp(&b.rate_hz)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let tol1 = (2.5 * rate_bin_hz).max(0.08 * fundamental_rate_hz);
    let tol2 = (2.0 * tol1).max(0.12 * 2.0 * fundamental_rate_hz);

    loop {
        if filtered.len() < 3 {
            break;
        }
        let mut worst_idx: Option<usize> = None;
        let mut worst_score = f64::NEG_INFINITY;
        for i in 1..(filtered.len() - 1) {
            let mut support1 = 0usize;
            let mut support2 = 0usize;
            let mut avail2 = 0usize;
            let mut score = 0.0f64;

            let d_prev = (filtered[i].rate_hz - filtered[i - 1].rate_hz).abs();
            let e_prev = (d_prev - fundamental_rate_hz).abs();
            if e_prev <= tol1 {
                support1 += 1;
            }
            score += e_prev;

            let d_next = (filtered[i + 1].rate_hz - filtered[i].rate_hz).abs();
            let e_next = (d_next - fundamental_rate_hz).abs();
            if e_next <= tol1 {
                support1 += 1;
            }
            score += e_next;

            if i >= 2 {
                avail2 += 1;
                let d2_prev = (filtered[i].rate_hz - filtered[i - 2].rate_hz).abs();
                let e2_prev = (d2_prev - 2.0 * fundamental_rate_hz).abs();
                if e2_prev <= tol2 {
                    support2 += 1;
                }
                score += 0.5 * e2_prev;
            }
            if (i + 2) < filtered.len() {
                avail2 += 1;
                let d2_next = (filtered[i + 2].rate_hz - filtered[i].rate_hz).abs();
                let e2_next = (d2_next - 2.0 * fundamental_rate_hz).abs();
                if e2_next <= tol2 {
                    support2 += 1;
                }
                score += 0.5 * e2_next;
            }

            let is_bad = support1 == 0 || (avail2 > 0 && support2 == 0 && support1 <= 1);
            if is_bad {
                if support1 == 0 {
                    score += 10.0 * fundamental_rate_hz;
                }
                if avail2 > 0 && support2 == 0 {
                    score += 5.0 * fundamental_rate_hz;
                }
                if score > worst_score {
                    worst_score = score;
                    worst_idx = Some(i);
                }
            }
        }
        if let Some(idx) = worst_idx {
            filtered.remove(idx);
        } else {
            break;
        }
    }
    filtered
}

fn estimate_dispersion_measure(
    channel_series: &[Vec<(f64, f64)>],
    durations: &[f64],
    freq_axis_mhz: &[f64],
    period: f64,
    bins: usize,
) -> Result<Option<DispersionEstimate>> {
    if channel_series.is_empty()
        || freq_axis_mhz.is_empty()
        || !period.is_finite()
        || period <= 0.0
        || bins == 0
    {
        return Ok(None);
    }

    let channels = channel_series.len().min(freq_axis_mhz.len());
    if channels < 8 {
        return Ok(None);
    }
    let rows = channel_series
        .iter()
        .take(channels)
        .map(|v| v.len())
        .min()
        .unwrap_or(0);
    if rows < 8 {
        return Ok(None);
    }

    // DM fit uses higher phase resolution than final fold display.
    let dm_bins = (bins.saturating_mul(16)).clamp(512, 4096);
    let subband_count = choose_subband_count(channels);
    if subband_count < 4 {
        return Ok(None);
    }

    let mut subbands = Vec::with_capacity(subband_count);
    for sb in 0..subband_count {
        let start = sb * channels / subband_count;
        let end = (sb + 1) * channels / subband_count;
        if end <= start + 1 {
            continue;
        }
        let avg_series = build_subband_average_series(channel_series, start, end, rows);
        if avg_series.is_empty() {
            continue;
        }
        let folded = fold_profile(&avg_series, durations, period, dm_bins)?;
        if folded.is_empty() {
            continue;
        }
        let Some((peak_phase_cycles, peak_amplitude)) = estimate_profile_peak_phase(&folded) else {
            continue;
        };
        let freq_center_mhz = mean_freq(&freq_axis_mhz[start..end]);
        if !freq_center_mhz.is_finite() || freq_center_mhz <= 0.0 {
            continue;
        }
        subbands.push(SubbandFoldProfile {
            subband_idx: sb,
            freq_center_mhz,
            peak_phase_cycles,
            peak_amplitude,
        });
    }
    if subbands.len() < 4 {
        return Ok(None);
    }

    // Peak-phase tracking across subbands is more robust than template cross-correlation
    // for narrow pulses and short integrations.
    subbands.sort_by(|a, b| {
        a.freq_center_mhz
            .partial_cmp(&b.freq_center_mhz)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut fit_points = Vec::with_capacity(subbands.len());
    let mut unwrapped_prev = 0.0f64;
    let mut ref_unwrapped = 0.0f64;
    for (idx, sb) in subbands.iter().enumerate() {
        let mut phase = sb.peak_phase_cycles;
        if idx == 0 {
            unwrapped_prev = phase;
            ref_unwrapped = phase;
        } else {
            while phase - unwrapped_prev > 0.5 {
                phase -= 1.0;
            }
            while phase - unwrapped_prev < -0.5 {
                phase += 1.0;
            }
            unwrapped_prev = phase;
        }

        let phase_shift_cycles = phase - ref_unwrapped;
        let delay_s = phase_shift_cycles * period;
        if !delay_s.is_finite() {
            continue;
        }
        let inv_freq2_mhz2 = 1.0 / (sb.freq_center_mhz * sb.freq_center_mhz);
        fit_points.push(DmFitPoint {
            subband_idx: sb.subband_idx,
            freq_center_mhz: sb.freq_center_mhz,
            inv_freq2_mhz2,
            delay_s,
            phase_shift_cycles,
            corr_peak: sb.peak_amplitude.abs().max(1e-6),
            residual_s: 0.0,
        });
    }
    if fit_points.len() < 4 {
        return Ok(None);
    }

    let Some((mut slope, mut intercept, mut r2)) = weighted_linear_fit_dm(&fit_points) else {
        return Ok(None);
    };

    let mut residuals: Vec<f64> = fit_points
        .iter()
        .map(|p| p.delay_s - (slope * p.inv_freq2_mhz2 + intercept))
        .collect();
    let mut abs_res: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
    abs_res.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med_abs = median_of_sorted(&abs_res);
    let sigma_res = (med_abs * 1.4826).max(period / dm_bins as f64 * 0.1);
    let max_res = (3.0 * sigma_res).max(1e-9);

    let mut filtered = Vec::with_capacity(fit_points.len());
    for (idx, p) in fit_points.iter().enumerate() {
        let keep = residuals
            .get(idx)
            .copied()
            .map(|r| r.abs() <= max_res)
            .unwrap_or(false);
        if keep {
            filtered.push(p.clone());
        }
    }
    if filtered.len() >= 4 {
        if let Some((s, b, fit_r2)) = weighted_linear_fit_dm(&filtered) {
            slope = s;
            intercept = b;
            r2 = fit_r2;
            fit_points = filtered;
            residuals = fit_points
                .iter()
                .map(|p| p.delay_s - (slope * p.inv_freq2_mhz2 + intercept))
                .collect();
        }
    }

    for (idx, p) in fit_points.iter_mut().enumerate() {
        p.residual_s = residuals.get(idx).copied().unwrap_or(0.0);
    }

    const DISPERSION_CONSTANT_SEC: f64 = 4.148_808;
    let dm_pc_cm3 = slope / DISPERSION_CONSTANT_SEC;
    if !dm_pc_cm3.is_finite() {
        return Ok(None);
    }

    Ok(Some(DispersionEstimate {
        dm_pc_cm3,
        slope_sec_per_invfreq2: slope,
        intercept_sec: intercept,
        r2,
        subband_count: subbands.len(),
        points: fit_points,
    }))
}

fn choose_subband_count(channels: usize) -> usize {
    (channels / 16).clamp(8, 32)
}

fn build_subband_average_series(
    channel_series: &[Vec<(f64, f64)>],
    start_ch: usize,
    end_ch: usize,
    rows: usize,
) -> Vec<(f64, f64)> {
    if start_ch >= end_ch || rows == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        let mut time = 0.0f64;
        let mut time_set = false;
        for ch in start_ch..end_ch {
            let Some(&(t, amp)) = channel_series.get(ch).and_then(|v| v.get(row)) else {
                continue;
            };
            if !time_set {
                time = t;
                time_set = true;
            }
            if amp.is_finite() {
                sum += amp;
                count += 1;
            }
        }
        if time_set && count > 0 {
            out.push((time, sum / count as f64));
        }
    }
    out
}

fn mean_freq(freqs: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for &f in freqs {
        if f.is_finite() && f > 0.0 {
            sum += f;
            count += 1;
        }
    }
    if count > 0 {
        sum / count as f64
    } else {
        f64::NAN
    }
}

fn estimate_profile_peak_phase(folded: &[(f64, f64)]) -> Option<(f64, f64)> {
    if folded.len() < 3 {
        return None;
    }
    let amps: Vec<f64> = folded
        .iter()
        .map(|&(_, amp)| if amp.is_finite() { amp } else { 0.0 })
        .collect();
    let n = amps.len();
    let mut best_idx = 0usize;
    let mut best_amp = f64::NEG_INFINITY;
    for (idx, &amp) in amps.iter().enumerate() {
        if amp > best_amp {
            best_amp = amp;
            best_idx = idx;
        }
    }
    if !best_amp.is_finite() {
        return None;
    }

    let prev = amps[(best_idx + n - 1) % n];
    let curr = amps[best_idx];
    let next = amps[(best_idx + 1) % n];
    let denom = prev - 2.0 * curr + next;
    let mut frac = 0.0f64;
    if denom.abs() > 1e-12 {
        frac = (0.5 * (prev - next) / denom).clamp(-0.5, 0.5);
    }
    let phase = ((best_idx as f64 + 0.5 + frac) / n as f64).rem_euclid(1.0);
    Some((phase, best_amp))
}

fn weighted_linear_fit_dm(points: &[DmFitPoint]) -> Option<(f64, f64, f64)> {
    if points.len() < 3 {
        return None;
    }
    let mut s = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut sxy = 0.0f64;
    for p in points {
        let w = p.corr_peak.abs().max(1e-6);
        let x = p.inv_freq2_mhz2;
        let y = p.delay_s;
        if !w.is_finite() || !x.is_finite() || !y.is_finite() {
            continue;
        }
        s += w;
        sx += w * x;
        sy += w * y;
        sxx += w * x * x;
        sxy += w * x * y;
    }
    let denom = s * sxx - sx * sx;
    if s <= 0.0 || denom.abs() <= 1e-18 {
        return None;
    }
    let slope = (s * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / s;

    let y_mean = sy / s;
    let mut ss_res = 0.0f64;
    let mut ss_tot = 0.0f64;
    for p in points {
        let w = p.corr_peak.abs().max(1e-6);
        let y_fit = slope * p.inv_freq2_mhz2 + intercept;
        let res = p.delay_s - y_fit;
        ss_res += w * res * res;
        let d = p.delay_s - y_mean;
        ss_tot += w * d * d;
    }
    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };
    Some((slope, intercept, r2))
}

fn generate_rate_axis(len: usize, sample_dt: f64) -> Vec<f64> {
    let mut axis = Vec::with_capacity(len);
    if len == 0 || !sample_dt.is_finite() || sample_dt <= 0.0 {
        return axis;
    }
    let sample_rate = 1.0 / sample_dt;
    let bin_width = sample_rate / len as f64;
    let half = (len / 2) as isize;
    for idx in 0..len {
        let offset = idx as isize - half;
        axis.push(offset as f64 * bin_width);
    }
    axis
}

fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn estimate_rate_bin_hz(rate_axis: &[f64]) -> Option<f64> {
    if rate_axis.len() < 2 {
        return None;
    }
    let mut best = f64::INFINITY;
    for w in rate_axis.windows(2) {
        let d = (w[1] - w[0]).abs();
        if d.is_finite() && d > 0.0 && d < best {
            best = d;
        }
    }
    if best.is_finite() {
        Some(best)
    } else {
        None
    }
}

fn estimate_fundamental_from_rate_diffs(
    points: &[RateProfileAboveAmpPoint],
    rate_bin_hz: f64,
) -> Option<RateFundamentalEstimate> {
    if points.len() < 2 || !rate_bin_hz.is_finite() || rate_bin_hz <= 0.0 {
        return None;
    }
    let mut rates: Vec<f64> = points.iter().map(|p| p.rate_hz).collect();
    rates.retain(|v| v.is_finite());
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    rates.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);
    if rates.len() < 2 {
        return None;
    }

    let min_valid_gap_hz = (2.0 * rate_bin_hz).max(1e-12);
    let adjacent_diffs: Vec<f64> = rates
        .windows(2)
        .filter_map(|pair| {
            let dr = (pair[1] - pair[0]).abs();
            if dr.is_finite() && dr >= min_valid_gap_hz {
                Some(dr)
            } else {
                None
            }
        })
        .collect();
    if adjacent_diffs.is_empty() {
        return None;
    }

    let mut counts: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
    for &dr in &adjacent_diffs {
        let bin = (dr / rate_bin_hz).round() as i64;
        if bin > 0 {
            *counts.entry(bin).or_insert(0) += 1;
        }
    }
    if counts.is_empty() {
        return None;
    }

    let local_count = |bin: i64| -> usize {
        let c0 = *counts.get(&(bin - 1)).unwrap_or(&0);
        let c1 = *counts.get(&bin).unwrap_or(&0);
        let c2 = *counts.get(&(bin + 1)).unwrap_or(&0);
        c0 + c1 + c2
    };

    let best_bin = *counts.keys().max_by(|a, b| {
        let ca = local_count(**a);
        let cb = local_count(**b);
        ca.cmp(&cb).then_with(|| b.cmp(a))
    })?;
    let selected_diffs: Vec<f64> = adjacent_diffs
        .iter()
        .copied()
        .filter(|dr| {
            let bin = (*dr / rate_bin_hz).round() as i64;
            (best_bin - 1..=best_bin + 1).contains(&bin)
        })
        .collect();
    if selected_diffs.is_empty() {
        return None;
    }
    let fundamental_rate_hz =
        selected_diffs.iter().copied().sum::<f64>() / selected_diffs.len() as f64;
    if !fundamental_rate_hz.is_finite() || fundamental_rate_hz <= 0.0 {
        return None;
    }
    let fundamental_sigma_hz = if selected_diffs.len() >= 2 {
        let var = selected_diffs
            .iter()
            .map(|v| {
                let dv = *v - fundamental_rate_hz;
                dv * dv
            })
            .sum::<f64>()
            / selected_diffs.len() as f64;
        var.sqrt()
    } else {
        0.0
    };
    let sigma_window = fundamental_sigma_hz;
    let fundamental_count = adjacent_diffs
        .iter()
        .filter(|&&d| (d - fundamental_rate_hz).abs() <= sigma_window)
        .count();

    let period_s = 1.0 / fundamental_rate_hz;
    Some(RateFundamentalEstimate {
        fundamental_rate_hz,
        fundamental_sigma_hz,
        fundamental_count: fundamental_count.max(1),
        period_s,
    })
}

fn extract_rate_profile_from_delay_rate(
    delay_rate_spectrum: &Array2<Complex<f32>>,
    sample_dt: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    if delay_rate_spectrum.is_empty() || !sample_dt.is_finite() || sample_dt <= 0.0 {
        return None;
    }
    let rows = delay_rate_spectrum.shape()[0];
    let cols = delay_rate_spectrum.shape()[1];
    if rows < 2 || cols == 0 {
        return None;
    }
    let mut peak_r = 0usize;
    let mut peak_d = 0usize;
    let mut peak_amp = f64::NEG_INFINITY;
    for r in 0..rows {
        for d in 0..cols {
            let amp = delay_rate_spectrum[[r, d]].norm() as f64;
            if amp.is_finite() && amp > peak_amp {
                peak_amp = amp;
                peak_r = r;
                peak_d = d;
            }
        }
    }
    if !peak_amp.is_finite() || peak_amp <= 0.0 {
        return None;
    }
    let _ = peak_r;
    let rate_axis = generate_rate_axis(rows, sample_dt);
    let profile = (0..rows)
        .map(|r| delay_rate_spectrum[[r, peak_d]].norm() as f64)
        .collect::<Vec<_>>();
    Some((rate_axis, profile))
}

fn plot_rate_profile(
    path: &Path,
    rate_axis: &[f64],
    amplitudes: &[f64],
    above_amp_points: Option<&[RateProfileAboveAmpPoint]>,
    periodic_points: Option<&[RateProfileAboveAmpPoint]>,
) -> Result<()> {
    if rate_axis.len() != amplitudes.len() || rate_axis.len() < 2 {
        return Ok(());
    }

    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for (&x, &y) in rate_axis.iter().zip(amplitudes.iter()) {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        x_min = x_min.min(x);
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }
    if !x_min.is_finite()
        || !x_max.is_finite()
        || !y_min.is_finite()
        || !y_max.is_finite()
        || (x_max - x_min).abs() < 1e-12
    {
        return Ok(());
    }
    let x_range = (x_max - x_min).abs().max(1e-12);
    let y_range = (y_max - y_min).abs().max(1e-12);

    let root = BitMapBackend::new(path, (850, 550)).into_drawing_area();
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
        .x_desc("Rate [Hz]")
        .y_desc("Amplitude")
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .light_line_style(TRANSPARENT)
        .label_style(("sans-serif", scaled_font_size(20)).into_font())
        .axis_desc_style(("sans-serif", scaled_font_size(22)).into_font())
        .draw()?;
    chart
        .draw_series(LineSeries::new(
            rate_axis.iter().copied().zip(amplitudes.iter().copied()),
            &BLUE,
        ))?
        .label("Rate spectrum")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], BLUE));

    if let Some(points) = above_amp_points {
        if !points.is_empty() {
            chart
                .draw_series(
                    points
                        .iter()
                        .map(|p| Circle::new((p.rate_hz, p.amplitude), 4, GREEN.filled())),
                )?
                .label("Above threshold")
                .legend(|(x, y)| Circle::new((x + 10, y), 5, GREEN.filled()));
        }
    }
    if let Some(points) = periodic_points {
        if !points.is_empty() {
            chart
                .draw_series(
                    points
                        .iter()
                        .map(|p| Cross::new((p.rate_hz, p.amplitude), 12, RED.stroke_width(2))),
                )?
                .label("Periodic peaks")
                .legend(|(x, y)| Cross::new((x + 10, y), 10, RED.stroke_width(2)));
        }
    }
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", scaled_legend_font_size(16)).into_font())
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    compress_plot_png(path);
    Ok(())
}

fn plot_folded_profile_from_period(
    path: &Path,
    folded: &[(f64, f64)],
    period_s: f64,
    _period_std_s: Option<f64>,
) -> Result<()> {
    if folded.is_empty() || !period_s.is_finite() || period_s <= 0.0 {
        return Ok(());
    }
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &(_, v) in folded {
        if v.is_finite() {
            y_min = y_min.min(v);
            y_max = y_max.max(v);
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        return Ok(());
    }
    let y_range = (y_max - y_min).abs().max(1e-12);

    let root = BitMapBackend::new(path, (850, 550)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(80)
        .y_label_area_size(110)
        .build_cartesian_2d(0.0..1.0, (y_min - 0.05 * y_range)..(y_max + 0.05 * y_range))?;

    chart
        .configure_mesh()
        .x_desc("Phase")
        .y_desc("Amplitude")
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .light_line_style(TRANSPARENT)
        .label_style(("sans-serif", scaled_font_size(20)).into_font())
        .axis_desc_style(("sans-serif", scaled_font_size(22)).into_font())
        .draw()?;
    chart
        .draw_series(LineSeries::new(folded.iter().copied(), &RED))?
        .label("Folded profile")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], RED));
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", scaled_legend_font_size(16)).into_font())
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    compress_plot_png(path);
    Ok(())
}
