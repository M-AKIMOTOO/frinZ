#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

// Earth-rotation synthesis imaging module

use crate::args::Args;
use crate::bandpass::{bandpass_gains, read_bandpass_file};
use crate::fft::apply_phase_correction_in_place_at_times;
use crate::header::{available_cor_sectors, parse_header, CorHeader};
use crate::input_support::open_input_data;
use crate::npy_output::{npz_sidecar_path, NamedNpz, NpyMeta};
use crate::plot::rasterize_rgb_grid;
use crate::processing::run_analysis_pipeline;
use crate::read::{read_sector_metadata, read_visibility_data, sector_epochs};
use crate::rfi::{flagged_channels, parse_rfi_ranges};
use crate::search;
use crate::utils;
use chrono::{DateTime, Utc};
use plotters::backend::RGBPixel;
use plotters::prelude::*;
use rustfft::{
    num_complex::{Complex, Complex32},
    FftPlanner,
};
use serde::Serialize;
use serde_json::to_string_pretty;
use std::error::Error;
use std::f64::consts::PI;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

const SPEED_OF_LIGHT: f64 = 299_792_458.0;
const EARTH_ROTATION_RATE_RAD_PER_SEC: f64 = 7.292_115_0e-5;

type C32 = Complex32;

fn rebin_complex_rows(
    data: &[C32],
    rows: usize,
    original_cols: usize,
    target_cols: usize,
) -> Vec<C32> {
    if rows == 0
        || original_cols == 0
        || target_cols == 0
        || target_cols > original_cols
        || original_cols == target_cols
        || !original_cols.is_multiple_of(target_cols)
    {
        return data.to_vec();
    }

    let group = original_cols / target_cols;
    let mut rebinned = Vec::with_capacity(rows.saturating_mul(target_cols));

    for row in 0..rows {
        let base = row * original_cols;
        for target in 0..target_cols {
            let mut sum = C32::new(0.0, 0.0);
            for offset in 0..group {
                sum += data[base + target * group + offset];
            }
            rebinned.push(sum / group as f32);
        }
    }

    rebinned
}

// Represents a single visibility measurement.
#[derive(Debug, Clone)]
pub struct Visibility {
    pub u: f64,
    pub v: f64,
    pub w: f64,
    pub real: f64,
    pub imag: f64,
    pub weight: f64,
    pub time: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Baseline {
    pub east_m: f64,
    pub north_m: f64,
    pub up_m: f64,
}

#[derive(Debug, Clone)]
pub struct ObservationSpec {
    pub reference_frequency_hz: f64,
    pub source_declination_rad: f64,
    pub start_hour_angle_rad: f64,
    pub integration_time_s: f64,
    pub num_samples: usize,
}

impl ObservationSpec {
    pub fn wavelength_m(&self) -> f64 {
        SPEED_OF_LIGHT / self.reference_frequency_hz
    }
}

#[derive(Debug, Clone)]
pub struct PointSource {
    pub l_rad: f64,
    pub m_rad: f64,
    pub flux_jy: f64,
}

#[derive(Debug, Clone)]
pub struct ImagingConfig {
    pub image_size: usize,
    pub cell_size_arcsec: f64,
    pub clean: Option<CleanConfig>,
}

impl ImagingConfig {
    pub fn new(image_size: usize, cell_size_arcsec: f64) -> Self {
        Self {
            image_size,
            cell_size_arcsec,
            clean: None,
        }
    }

    fn cell_size_rad(&self) -> f64 {
        self.cell_size_arcsec * PI / (180.0 * 3600.0)
    }
}

#[derive(Debug, Clone)]
pub struct CleanConfig {
    pub gain: f64,
    pub threshold_snr: f64,
    pub max_iterations: usize,
}

#[derive(Debug, Clone)]
pub struct ImagingResult {
    pub dirty_image: Vec<f64>,
    pub dirty_beam: Vec<f64>,
    pub clean_image: Option<Vec<f64>>,
    pub residual_image: Option<Vec<f64>>,
    pub clean_components: Option<Vec<CleanComponent>>,
    pub image_size: usize,
    pub cell_size_arcsec: f64,
}

struct GriddedData {
    vis_grid: Vec<Complex<f64>>,
    sampling_grid: Vec<f64>,
}

#[derive(Default, Clone, Copy)]
struct GridAccum {
    real: f64,
    imag: f64,
    weight: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CleanComponent {
    row: usize,
    col: usize,
    amplitude: f64,
}

#[derive(Debug, Clone)]
pub struct ImagingCliOptions {
    pub size: usize,
    pub cell_arcsec: Option<f64>,
    pub clean: bool,
    pub clean_gain: f64,
    pub clean_threshold_snr: f64,
    pub clean_max_iter: usize,
    pub vis_snr_min: Option<f64>,
}

impl Default for ImagingCliOptions {
    fn default() -> Self {
        Self {
            size: 256,
            cell_arcsec: None,
            clean: false,
            clean_gain: 0.1,
            clean_threshold_snr: 5.0,
            clean_max_iter: 200,
            vis_snr_min: None,
        }
    }
}

pub fn parse_imaging_cli_options(tokens: &[String]) -> Result<ImagingCliOptions, String> {
    let mut opts = ImagingCliOptions::default();
    let mut gain_explicit = false;
    let mut threshold_explicit = false;
    let mut iter_explicit = false;
    let mut vis_snr_explicit = false;
    for raw in tokens {
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        let mut parts = token.splitn(2, [':', '=']);
        let key = parts.next().unwrap().trim().to_ascii_lowercase();
        let value_opt = parts.next().map(|v| v.trim());

        match key.as_str() {
            "size" => {
                let value = value_opt
                    .ok_or_else(|| "imaging option 'size' requires a value".to_string())?;
                let parsed = value
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid value for size: '{}'", value))?;
                if parsed == 0 {
                    return Err("Image size must be greater than zero".into());
                }
                opts.size = parsed;
            }
            "cell" | "cell_arcsec" | "cellsize" => {
                let value = value_opt
                    .ok_or_else(|| "imaging option 'cell' requires a value".to_string())?;
                let parsed = value
                    .parse::<f64>()
                    .map_err(|_| format!("Invalid value for cell: '{}'", value))?;
                if parsed <= 0.0 {
                    return Err("Cell size must be positive".into());
                }
                opts.cell_arcsec = Some(parsed);
            }
            "clean" => {
                let enabled = match value_opt {
                    None | Some("") => true,
                    Some(val) => parse_bool_token(val)?,
                };
                opts.clean = enabled;
            }
            "gain" | "clean_gain" => {
                let value = value_opt
                    .ok_or_else(|| "imaging option 'gain' requires a value".to_string())?;
                let parsed = value
                    .parse::<f64>()
                    .map_err(|_| format!("Invalid value for gain: '{}'", value))?;
                if parsed <= 0.0 || parsed >= 1.0 {
                    return Err("CLEAN gain must be between 0 and 1".into());
                }
                opts.clean_gain = parsed;
                gain_explicit = true;
            }
            "threshold" | "clean_threshold" | "thresh" => {
                let value = value_opt
                    .ok_or_else(|| "imaging option 'threshold' requires a value".to_string())?;
                let parsed = value
                    .parse::<f64>()
                    .map_err(|_| format!("Invalid value for threshold: '{}'", value))?;
                if parsed <= 0.0 {
                    return Err("CLEAN threshold must be positive".into());
                }
                opts.clean_threshold_snr = parsed;
                threshold_explicit = true;
            }
            "iter" | "iterations" | "max_iter" => {
                let value = value_opt
                    .ok_or_else(|| "imaging option 'iter' requires a value".to_string())?;
                let parsed = value
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid value for iter: '{}'", value))?;
                if parsed == 0 {
                    return Err("CLEAN max iterations must be greater than zero".into());
                }
                opts.clean_max_iter = parsed;
                iter_explicit = true;
            }
            "vis_snr" | "vis-snr" | "visibility_snr" => {
                let value = value_opt
                    .ok_or_else(|| "imaging option 'vis_snr' requires a value".to_string())?;
                let parsed = value
                    .parse::<f64>()
                    .map_err(|_| format!("Invalid value for vis_snr: '{}'", value))?;
                if parsed <= 0.0 {
                    return Err("Visibility SNR threshold must be positive".into());
                }
                opts.vis_snr_min = Some(parsed);
                vis_snr_explicit = true;
            }
            other => {
                return Err(format!("Unknown imaging option '{}'", other));
            }
        }
    }
    if !opts.clean && (gain_explicit || threshold_explicit || iter_explicit) {
        return Err(
            "CLEAN-related options (gain/threshold/iter) require 'clean:1'. Please enable CLEAN or remove those options."
                .into(),
        );
    }
    if vis_snr_explicit {
        if let Some(threshold) = opts.vis_snr_min {
            if threshold <= 0.0 {
                return Err("Visibility SNR threshold must be positive".into());
            }
        }
    }
    Ok(opts)
}

fn parse_bool_token(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(format!("Invalid boolean value '{}'", other)),
    }
}

/// シンプルな Earth-rotation synthesis imaging。
/// 既存インターフェースを保ちつつ内部で高度化された処理を呼び出す。
pub fn perform_imaging(
    visibilities: &[Visibility],
    image_size: usize,
    cell_size_arcsec: f64,
) -> Result<Vec<f64>, String> {
    let config = ImagingConfig::new(image_size, cell_size_arcsec);
    let result = perform_imaging_with_config(visibilities, &config)?;
    Ok(result.dirty_image)
}

/// 追加設定付きのイメージングパイプライン。
pub fn perform_imaging_with_config(
    visibilities: &[Visibility],
    config: &ImagingConfig,
) -> Result<ImagingResult, String> {
    if config.image_size == 0 {
        return Err("Image size must be greater than zero".into());
    }
    if !config.cell_size_arcsec.is_finite() || config.cell_size_arcsec <= 0.0 {
        return Err("Cell size must be positive".into());
    }

    println!("Starting Earth-rotation synthesis imaging...");
    let gridded = grid_visibilities(visibilities, config)?;
    println!("Gridding complete.");

    let sampling_complex: Vec<Complex<f64>> = gridded
        .sampling_grid
        .into_iter()
        .map(|w| Complex::new(w, 0.0))
        .collect();
    let size = config.image_size;
    let center = size / 2;
    let beam = fft_2d_inverse(sampling_complex, size)?;
    let normalization = beam[center * size + center].re;
    if !normalization.is_finite() || normalization <= 0.0 {
        return Err("No positively weighted visibilities fall inside the UV grid".into());
    }
    let dirty_beam: Vec<_> = beam.into_iter().map(|c| c.re / normalization).collect();
    let dirty_image: Vec<_> = fft_2d_inverse(gridded.vis_grid, size)?
        .into_iter()
        .map(|c| c.re / normalization)
        .collect();
    println!("FFT complete. Dirty image and beam normalized to the same weights.");

    let (clean_image, residual_image, clean_components) = if let Some(clean_cfg) = &config.clean {
        println!("Running lightweight CLEAN (Högbom-style) ...");
        let noise_estimate = estimate_image_rms(&dirty_image);
        println!(
            "CLEAN params: gain {:.3}, SNR threshold {:.2}, max {} iterations (noise ≈ {:.3e})",
            clean_cfg.gain, clean_cfg.threshold_snr, clean_cfg.max_iterations, noise_estimate
        );
        let (model, residual, components, clean_iters) = hogbom_clean(
            &dirty_image,
            &dirty_beam,
            config.image_size,
            clean_cfg,
            noise_estimate,
        );
        let mut clean = residual.clone();
        for (c, m) in clean.iter_mut().zip(model.iter()) {
            *c += *m;
        }
        println!(
            "CLEAN extracted {} components over {} iterations.",
            components.len(),
            clean_iters
        );
        (Some(clean), Some(residual), Some(components))
    } else {
        (None, None, None)
    };

    println!("Imaging pipeline complete.");

    Ok(ImagingResult {
        dirty_image,
        dirty_beam,
        clean_image,
        residual_image,
        clean_components,
        image_size: config.image_size,
        cell_size_arcsec: config.cell_size_arcsec,
    })
}

/// 1基線観測を想定した visibilities の簡易シミュレーション。
pub fn simulate_single_baseline_visibilities(
    baseline: Baseline,
    spec: &ObservationSpec,
    sources: &[PointSource],
) -> Vec<Visibility> {
    if spec.num_samples == 0 {
        return Vec::new();
    }

    let wavelength = spec.wavelength_m();
    let mut result = Vec::with_capacity(spec.num_samples);

    let sin_dec = spec.source_declination_rad.sin();
    let cos_dec = spec.source_declination_rad.cos();

    for idx in 0..spec.num_samples {
        let time_s = idx as f64 * spec.integration_time_s;
        let hour_angle = spec.start_hour_angle_rad + EARTH_ROTATION_RATE_RAD_PER_SEC * time_s;
        let sin_h = hour_angle.sin();
        let cos_h = hour_angle.cos();

        let u = (baseline.east_m * sin_h + baseline.north_m * cos_h) / wavelength;
        let v = (-baseline.east_m * sin_dec * cos_h
            + baseline.north_m * sin_dec * sin_h
            + baseline.up_m * cos_dec)
            / wavelength;
        let w = (baseline.east_m * cos_dec * cos_h - baseline.north_m * cos_dec * sin_h
            + baseline.up_m * sin_dec)
            / wavelength;

        let mut visibility = Complex::new(0.0, 0.0);
        for source in sources {
            let n = (1.0 - source.l_rad * source.l_rad - source.m_rad * source.m_rad).sqrt();
            let phase = -2.0 * PI * (u * source.l_rad + v * source.m_rad + w * (n - 1.0));
            let contrib = Complex::from_polar(source.flux_jy, phase);
            visibility += contrib;
        }

        result.push(Visibility {
            u,
            v,
            w,
            real: visibility.re,
            imag: visibility.im,
            weight: 1.0,
            time: time_s,
        });
    }

    result
}

fn grid_visibilities(
    visibilities: &[Visibility],
    config: &ImagingConfig,
) -> Result<GriddedData, String> {
    let size = config.image_size;
    let cell_size_rad = config.cell_size_rad();
    let uv_scale = size as f64 * cell_size_rad;
    let nyquist_bin = size as f64 / 2.0;
    let half = (size / 2) as f64;

    let mut accum = vec![GridAccum::default(); size * size];

    for vis in visibilities {
        let u = vis.u * uv_scale;
        // Image rows increase toward decreasing declination.
        let v = -vis.v * uv_scale;
        if !(u.abs() <= nyquist_bin && v.abs() <= nyquist_bin)
            || !vis.weight.is_finite()
            || vis.weight <= 0.0
            || !vis.real.is_finite()
            || !vis.imag.is_finite()
        {
            continue;
        }

        // Include the conjugate point to preserve Hermitian symmetry.
        for sign in [1.0, -1.0] {
            let u_grid = sign * u + half;
            let v_grid = sign * v + half;
            let u0 = u_grid.floor();
            let v0 = v_grid.floor();
            let du = u_grid - u0;
            let dv = v_grid - v0;
            for (offset_v, weight_v) in [(0, 1.0 - dv), (1, dv)] {
                let row = (v0 as isize + offset_v).rem_euclid(size as isize) as usize;
                for (offset_u, weight_u) in [(0, 1.0 - du), (1, du)] {
                    let col = (u0 as isize + offset_u).rem_euclid(size as isize) as usize;
                    let weight = vis.weight * weight_u * weight_v;
                    let cell = &mut accum[row * size + col];
                    cell.real += vis.real * weight;
                    cell.imag += sign * vis.imag * weight;
                    cell.weight += weight;
                }
            }
        }
    }

    let mut vis_grid = Vec::with_capacity(size * size);
    let mut sampling_grid = Vec::with_capacity(size * size);
    for cell in accum {
        vis_grid.push(Complex::new(cell.real, cell.imag));
        sampling_grid.push(cell.weight);
    }

    Ok(GriddedData {
        vis_grid,
        sampling_grid,
    })
}

fn fft_2d_inverse(mut grid: Vec<Complex<f64>>, size: usize) -> Result<Vec<Complex<f64>>, String> {
    if grid.len() != size * size {
        return Err("Grid size does not match expected dimensions".to_string());
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(size);
    let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];

    let half = size / 2;
    grid.rotate_left(half * size);
    for row in grid.chunks_mut(size) {
        row.rotate_left(half);
        fft.process_with_scratch(row, &mut scratch);
    }
    transpose_square_inplace(&mut grid, size);
    for row in grid.chunks_mut(size) {
        fft.process_with_scratch(row, &mut scratch);
    }
    transpose_square_inplace(&mut grid, size);
    for row in grid.chunks_mut(size) {
        row.rotate_right(half);
    }
    grid.rotate_right(half * size);
    Ok(grid)
}

fn transpose_square_inplace(data: &mut [Complex<f64>], size: usize) {
    // Tiled swaps keep both sides of the transpose local without another grid.
    const TILE: usize = 32;
    for rb in (0..size).step_by(TILE) {
        for cb in (rb..size).step_by(TILE) {
            for r in rb..(rb + TILE).min(size) {
                for c in cb.max(r + 1)..(cb + TILE).min(size) {
                    data.swap(r * size + c, c * size + r);
                }
            }
        }
    }
}

fn hogbom_clean(
    dirty_image: &[f64],
    dirty_beam: &[f64],
    size: usize,
    cfg: &CleanConfig,
    noise_rms: f64,
) -> (Vec<f64>, Vec<f64>, Vec<CleanComponent>, usize) {
    let mut residual = dirty_image.to_vec();
    let mut model = vec![0.0; dirty_image.len()];
    let center = size / 2;
    let mut components = Vec::new();
    let mut iterations = 0usize;

    for _ in 0..cfg.max_iterations {
        let (idx, &peak) = match residual.iter().enumerate().max_by(|(_, a), (_, b)| {
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Some(v) => v,
            None => break,
        };

        let snr = if noise_rms > 0.0 {
            peak.abs() / noise_rms
        } else {
            f64::INFINITY
        };
        if snr < cfg.threshold_snr {
            break;
        }

        let clean_amp = cfg.gain * peak;
        model[idx] += clean_amp;
        let peak_r = idx / size;
        let peak_c = idx % size;
        components.push(CleanComponent {
            row: peak_r,
            col: peak_c,
            amplitude: clean_amp,
        });
        iterations += 1;

        for br in 0..size {
            for bc in 0..size {
                let dr = br as isize - center as isize;
                let dc = bc as isize - center as isize;
                let tr = peak_r as isize + dr;
                let tc = peak_c as isize + dc;
                if tr < 0 || tr >= size as isize || tc < 0 || tc >= size as isize {
                    continue;
                }
                let target = (tr as usize) * size + (tc as usize);
                residual[target] -= dirty_beam[br * size + bc] * clean_amp;
            }
        }
    }

    (model, residual, components, iterations)
}

fn estimate_image_rms(image: &[f64]) -> f64 {
    if image.is_empty() {
        return 0.0;
    }
    let mut abs_vals: Vec<f64> = image.iter().map(|&v| v.abs()).collect();
    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let keep_len = (abs_vals.len() as f64 * 0.9).max(1.0).floor() as usize;
    let cutoff = abs_vals[keep_len.saturating_sub(1)];

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for &v in image {
        if v.abs() <= cutoff {
            sum += v;
            sum_sq += v * v;
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    let mean = sum / count as f64;
    let variance = (sum_sq / count as f64) - mean * mean;
    variance.max(0.0).sqrt()
}

pub fn run_earth_rotation_imaging(
    args: &Args,
    imaging_opts: &ImagingCliOptions,
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
) -> Result<(), Box<dyn Error>> {
    let input_path = args
        .input
        .as_ref()
        .ok_or("Earth-rotation imaging requires an input file")?;

    let (visibilities, header, effective_integ_time) = collect_visibilities_from_cor(
        input_path,
        args,
        imaging_opts,
        time_flag_ranges,
        pp_flag_ranges,
    )?;

    if visibilities.is_empty() {
        return Err("No usable visibilities remain after applying the requested flagging.".into());
    }

    println!(
        "Collected {} visibilities from {} (integration {:.3} s per sector).",
        visibilities.len(),
        header.source_name,
        effective_integ_time
    );

    if let Some(threshold) = imaging_opts.vis_snr_min {
        println!(
            "Applying visibility SNR threshold {:.2} before gridding.",
            threshold
        );
    }

    let resolved_cell = select_cell_size_arcsec(&visibilities, imaging_opts.cell_arcsec);
    println!(
        "Using cell size {:.4} arcsec ({} input).",
        resolved_cell,
        if imaging_opts.cell_arcsec.is_some() {
            "user-specified"
        } else {
            "auto-selected from UV coverage"
        }
    );
    let lambda_m = SPEED_OF_LIGHT / header.observing_frequency;
    let max_uv = visibilities
        .iter()
        .map(|v| (v.u * v.u + v.v * v.v).sqrt())
        .fold(0.0_f64, f64::max);
    let max_baseline_m = max_uv * lambda_m;
    let angular_resolution_deg = if max_baseline_m > 0.0 {
        (lambda_m / max_baseline_m) * 180.0 / PI
    } else {
        0.0
    };
    let angle_unit = select_angle_unit(angular_resolution_deg);
    let resolution_in_unit = angular_resolution_deg * angle_unit.multiplier;
    if max_baseline_m > 0.0 {
        println!(
            "Max projected baseline ≈ {:.1} m at {:.3} GHz ⇒ resolution ≈ {:.3} {}.",
            max_baseline_m,
            header.observing_frequency / 1.0e9,
            resolution_in_unit,
            angle_unit.label
        );
    } else {
        println!(
            "Max projected baseline not available (insufficient uv data); using {} axis scaling.",
            angle_unit.label
        );
    }

    let cell_size_unit = (resolved_cell / 3600.0) * angle_unit.multiplier;
    println!(
        "Plot cell size ≈ {:.4} {} per pixel on a {}×{} grid.",
        cell_size_unit, angle_unit.label, imaging_opts.size, imaging_opts.size
    );

    let mut config = ImagingConfig::new(imaging_opts.size, resolved_cell);
    if imaging_opts.clean {
        config.clean = Some(CleanConfig {
            gain: imaging_opts.clean_gain,
            threshold_snr: imaging_opts.clean_threshold_snr,
            max_iterations: imaging_opts.clean_max_iter,
        });
        println!(
            "CLEAN enabled (gain {:.3}, SNR threshold {:.2}, max {} iterations).",
            imaging_opts.clean_gain, imaging_opts.clean_threshold_snr, imaging_opts.clean_max_iter
        );
    }

    let result = perform_imaging_with_config(&visibilities, &config)?;

    let output_dir = prepare_imaging_output_dir(input_path)?;
    let base_name = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("imaging");

    let center = (result.image_size / 2) as f64;
    let ra_axis: Vec<f64> = (0..result.image_size)
        .map(|index| (index as f64 - center) * result.cell_size_arcsec)
        .collect();
    let dec_axis: Vec<f64> = (0..result.image_size)
        .map(|index| (center - index as f64) * result.cell_size_arcsec)
        .collect();
    let mut imaging_npz = if args.npz {
        let mut npz = NamedNpz::new(NpyMeta::new(
            "imaging",
            header.fft_point as u32,
            header.number_of_sector as u32,
        ));
        npz.add_f64_1d("ra_offset_arcsec", &ra_axis);
        npz.add_f64_1d("dec_offset_arcsec", &dec_axis);
        Some(npz)
    } else {
        None
    };
    let add_image_npy =
        |npz: &mut NamedNpz, name: &str, values: &[f64]| -> Result<(), Box<dyn Error>> {
            let values_f32: Vec<f32> = values.iter().map(|&value| value as f32).collect();
            npz.add_f32_2d(name, (result.image_size, result.image_size), values_f32)?;
            Ok(())
        };

    let dirty_path = output_dir.join(format!("{}_dirty.png", base_name));
    render_scalar_field_plot(
        &dirty_path,
        &result.dirty_image,
        result.image_size,
        result.cell_size_arcsec,
        angle_unit,
        true,
        "Dirty Image",
    )?;
    if let Some(npz) = imaging_npz.as_mut() {
        add_image_npy(npz, "dirty_image", &result.dirty_image)?;
    }
    println!("Dirty image saved to {}", dirty_path.display());

    let beam_path = output_dir.join(format!("{}_dirty_beam.png", base_name));
    render_scalar_field_plot(
        &beam_path,
        &result.dirty_beam,
        result.image_size,
        result.cell_size_arcsec,
        angle_unit,
        false,
        "Dirty Beam",
    )?;
    if let Some(npz) = imaging_npz.as_mut() {
        add_image_npy(npz, "dirty_beam", &result.dirty_beam)?;
    }
    println!("Dirty beam saved to {}", beam_path.display());

    if let Some(ref clean) = result.clean_image {
        let clean_path = output_dir.join(format!("{}_clean.png", base_name));
        render_scalar_field_plot(
            &clean_path,
            clean,
            result.image_size,
            result.cell_size_arcsec,
            angle_unit,
            true,
            "Clean Image",
        )?;
        if let Some(npz) = imaging_npz.as_mut() {
            add_image_npy(npz, "clean_image", clean)?;
        }
        println!("Clean image saved to {}", clean_path.display());
    }

    if let Some(ref residual) = result.residual_image {
        let residual_path = output_dir.join(format!("{}_residual.png", base_name));
        render_scalar_field_plot(
            &residual_path,
            residual,
            result.image_size,
            result.cell_size_arcsec,
            angle_unit,
            true,
            "Residual Image",
        )?;
        if let Some(npz) = imaging_npz.as_mut() {
            add_image_npy(npz, "residual_image", residual)?;
        }
        println!("Residual image saved to {}", residual_path.display());
    }

    if let Some(npz) = imaging_npz {
        let npz_path = npz_sidecar_path(&dirty_path, "imaging");
        npz.write(&npz_path)?;
        for legacy_flag in [
            "imaging_dirty",
            "imaging_dirty_beam",
            "imaging_clean",
            "imaging_residual",
        ] {
            let _ = std::fs::remove_file(npz_sidecar_path(&dirty_path, legacy_flag));
            let _ = std::fs::remove_file(npz_sidecar_path(&beam_path, legacy_flag));
        }
        println!("Imaging NPZ data saved to {}", npz_path.display());
    }

    if let Some(ref components) = result.clean_components {
        let components_path = output_dir.join(format!("{}_clean_components.json", base_name));
        save_clean_components(
            &components_path,
            components,
            result.image_size,
            result.cell_size_arcsec,
            angle_unit,
        )?;
        println!("Clean components saved to {}", components_path.display());
    }

    let max_u = visibilities
        .iter()
        .map(|v| v.u.abs())
        .fold(0.0_f64, f64::max);
    let max_v = visibilities
        .iter()
        .map(|v| v.v.abs())
        .fold(0.0_f64, f64::max);

    println!(
        "UV coverage summary: |u| max {:.1} λ, |v| max {:.1} λ. Image grid {}×{}, cell size {:.3} arcsec.",
        max_u, max_v, config.image_size, config.image_size, config.cell_size_arcsec
    );

    Ok(())
}

fn collect_visibilities_from_cor(
    input_path: &Path,
    args: &Args,
    imaging_opts: &ImagingCliOptions,
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
) -> Result<(Vec<Visibility>, CorHeader, f64), Box<dyn Error>> {
    let input_data = open_input_data(input_path)?;
    let mut cursor = Cursor::new(input_data.as_slice());
    let header = parse_header(&mut cursor)?;
    let available = available_cor_sectors(&header, input_data.as_slice().len())?;
    let skip = args.skip;
    if skip >= available {
        return Err("No complete imaging sectors remain after --skip".into());
    }

    let (file_epoch, cadence) = read_sector_metadata(&mut cursor, &header, 0)?;
    let cadence_seconds = f64::from(cadence);
    let integration_ns = (1.0e9 * cadence_seconds).round() as i64;
    let native_fft = header.fft_point;
    let effective_fft = args.fft_rebin.unwrap_or(native_fft);
    if effective_fft < 2 || effective_fft % 2 != 0 || native_fft % effective_fft != 0 {
        return Err("--fft-rebin must evenly divide the original FFT with an even size".into());
    }
    let channels = (native_fft / 2) as usize;
    let group = (native_fft / effective_fft) as usize;
    let channel_hz = f64::from(header.sampling_speed) / f64::from(native_fft);
    let rfi_ranges = parse_rfi_ranges(&args.rfi, (channel_hz * 1.0e-6) as f32)?;
    let mut flagged = flagged_channels(channels, &rfi_ranges);
    if let Some(mask) = args.rfi_npz_mask.as_deref() {
        for (flag, imported) in flagged.iter_mut().zip(mask.frequency_channel_mask(
            channels,
            header.sampling_speed,
            native_fft,
        )) {
            *flag |= imported;
        }
    }

    let bandpass = args
        .bandpass
        .as_deref()
        .map(read_bandpass_file)
        .transpose()?;
    let gains = bandpass
        .as_deref()
        .map(bandpass_gains)
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let gain_group = if gains.len() == channels / group {
        group
    } else if gains.is_empty() || gains.len() == channels {
        1
    } else {
        return Err("Bandpass channel count does not match the native or rebinned FFT".into());
    };
    let spectral_bins: Vec<_> = (0..channels)
        .step_by(group)
        .filter_map(|start| {
            let active: Vec<_> = (start..start + group)
                .filter(|&channel| !flagged[channel])
                .collect();
            if active.is_empty() {
                return None;
            }
            let mean_channel = active.iter().sum::<usize>() as f64 / active.len() as f64;
            let frequency = header.observing_frequency + channel_hz * mean_channel;
            (frequency.is_finite() && frequency > 0.0)
                .then_some((active, frequency / SPEED_OF_LIGHT))
        })
        .collect();
    if spectral_bins.is_empty() {
        return Err("No usable imaging channels remain after applying the frequency flags".into());
    }

    let remaining = available - skip;
    if args.cumulate > remaining {
        return Err("--cumulate exceeds the number of available imaging sectors".into());
    }
    let (segment_length, end_sector) = if args.cumulate > 0 {
        (args.cumulate, available)
    } else if args.length > 0 {
        (
            args.length,
            skip + remaining.min(args.length.saturating_mul(args.loop_)),
        )
    } else {
        let default_length = (30.0 / cadence).round().max(1.0) as i32;
        let end = if args.loop_ == 1 {
            available
        } else {
            skip + remaining.min(default_length.saturating_mul(args.loop_))
        };
        (default_length, end)
    };
    let projection = utils::UvwProjection::new(
        header.station1_position,
        header.station2_position,
        header.source_position_ra,
        header.source_position_dec,
        true,
    );
    let manual_phase = args.delay_correct != 0.0
        || args.rate_correct != 0.0
        || args.acel_correct != 0.0
        || args.jerk_correct != 0.0
        || args.snap_correct != 0.0;
    let search_mode = args.primary_search_mode();
    let needs_fit = search_mode.is_some() || imaging_opts.vis_snr_min.is_some();
    let mut fit_args = args.clone();
    fit_args.plot = false;
    fit_args.spectrum = false;
    fit_args.delay_correct = 0.0;
    fit_args.rate_correct = 0.0;
    fit_args.acel_correct = 0.0;
    fit_args.jerk_correct = 0.0;
    fit_args.snap_correct = 0.0;
    fit_args.channel_origin_bins = (group - 1) as f64 / (2 * group) as f64;

    let mut visibilities = Vec::new();
    let mut skipped_low_snr = 0usize;
    let mut spectrum = vec![Complex::new(0.0, 0.0); channels];
    for first in (skip..end_sector).step_by(segment_length as usize) {
        let end = first.saturating_add(segment_length).min(end_sector);
        let epochs = sector_epochs(input_data.as_slice(), &header, first as usize, end as usize)?
            .collect::<std::io::Result<Vec<_>>>()?;
        let use_row: Vec<_> = epochs
            .iter()
            .enumerate()
            .map(|(i, &epoch)| {
                let pp = first as u32 + i as u32;
                !pp_flag_ranges
                    .iter()
                    .any(|&(start, end)| (start..=end).contains(&pp))
                    && !is_time_flagged(epoch, time_flag_ranges)
            })
            .collect();
        if !use_row.iter().any(|&valid| valid) {
            continue;
        }

        let rows = end - first;
        let (mut samples, segment_start, _) =
            read_visibility_data(&mut cursor, &header, rows, first, 0, false, pp_flag_ranges)?;
        for (row, &valid) in samples.chunks_exact_mut(channels).zip(&use_row) {
            if !valid {
                row.fill(C32::default());
            } else if needs_fit || !gains.is_empty() {
                for (channel, value) in row.iter_mut().enumerate() {
                    if flagged[channel] {
                        *value = C32::default();
                    } else if let Some(&gain) = gains.get(channel / gain_group) {
                        *value *= C32::new(gain.re as f32, gain.im as f32);
                    }
                }
            }
        }
        if manual_phase {
            let times: Vec<_> = epochs
                .iter()
                .map(|&epoch| utils::seconds_between(epoch, file_epoch))
                .collect();
            apply_phase_correction_in_place_at_times(
                &mut samples,
                channels,
                args.rate_correct,
                args.delay_correct,
                args.acel_correct,
                args.jerk_correct,
                args.snap_correct,
                header.sampling_speed as u32,
                native_fft as u32,
                &times,
                header.observing_frequency,
                0.0,
            );
        }
        if needs_fit {
            let times: Vec<_> = epochs
                .iter()
                .map(|&epoch| utils::seconds_between(epoch, segment_start))
                .collect();
            let rebinned;
            let fit_data = if group == 1 {
                samples.as_slice()
            } else {
                rebinned = rebin_complex_rows(&samples, rows as usize, channels, channels / group);
                rebinned.as_slice()
            };
            let (delay, rate, snr) = fit_imaging_segment(
                fit_data,
                &header,
                &fit_args,
                search_mode,
                rows,
                cadence,
                segment_start,
                file_epoch,
                &times,
                effective_fft,
            )?;
            if imaging_opts
                .vis_snr_min
                .is_some_and(|minimum| !snr.is_finite() || snr < minimum)
            {
                skipped_low_snr += 1;
                continue;
            }
            if delay != 0.0 || rate != 0.0 {
                apply_phase_correction_in_place_at_times(
                    &mut samples,
                    channels,
                    rate,
                    delay,
                    0.0,
                    0.0,
                    0.0,
                    header.sampling_speed as u32,
                    native_fft as u32,
                    &times,
                    header.observing_frequency,
                    0.0,
                );
            }
        }

        // Each average uses only adjacent unflagged exposures. A gap
        // changes the UV coordinate without contributing sample weight.
        let mut start = 0;
        while start < epochs.len() {
            if !use_row[start] {
                start += 1;
                continue;
            }
            let mut end = start + 1;
            while end < epochs.len()
                && use_row[end]
                && (utils::seconds_between(epochs[end], epochs[end - 1]) - cadence_seconds).abs()
                    <= cadence_seconds * 1.0e-3
            {
                end += 1;
            }
            spectrum.fill(Complex::new(0.0, 0.0));
            for row in samples[start * channels..end * channels].chunks_exact(channels) {
                for (sum, value) in spectrum.iter_mut().zip(row) {
                    *sum += Complex::new(f64::from(value.re), f64::from(value.im));
                }
            }
            let epoch = epochs[start]
                + (epochs[end - 1] - epochs[start]) / 2
                + chrono::Duration::nanoseconds(integration_ns / 2);
            let (u, v, w, _, _) = projection.project(utils::greenwich_sidereal_angle(epoch));
            let time = utils::seconds_between(epoch, file_epoch);
            for (active, inv_wavelength) in &spectral_bins {
                let sum: Complex<f64> = active.iter().map(|&channel| spectrum[channel]).sum();
                let sample_count = (end - start) as f64 * active.len() as f64;
                let average = sum / sample_count;
                visibilities.push(Visibility {
                    u: u * inv_wavelength,
                    v: v * inv_wavelength,
                    w: w * inv_wavelength,
                    real: average.re,
                    // .cor phase increases with positive geometric delay.
                    imag: -average.im,
                    weight: cadence_seconds * sample_count,
                    time,
                });
            }
            start = end;
        }
    }
    if skipped_low_snr > 0 {
        println!("Visibility SNR filter removed {skipped_low_snr} segment(s) below threshold.");
    }
    Ok((visibilities, header, cadence_seconds))
}

fn fit_imaging_segment(
    data: &[C32],
    header: &CorHeader,
    args: &Args,
    mode: Option<&str>,
    rows: i32,
    cadence: f32,
    start: DateTime<Utc>,
    file_epoch: DateTime<Utc>,
    times: &[f64],
    effective_fft: i32,
) -> Result<(f64, f64, f64), Box<dyn Error>> {
    let fitter = match mode {
        Some("deep") => search::run_deep_search_with_phase_times,
        Some("coherent") => search::run_coherent_search_with_phase_times,
        Some(_) => search::run_peak_search_with_phase_times,
        None => {
            let (result, _, _, _) = run_analysis_pipeline(
                data,
                header,
                args,
                None,
                0.0,
                0.0,
                0.0,
                rows,
                rows,
                cadence,
                &start,
                &file_epoch,
                &[],
                &None,
                false,
                effective_fft,
            )?;
            return Ok((0.0, 0.0, f64::from(result.delay_snr)));
        }
    };
    let result = fitter(
        data,
        header,
        rows,
        rows,
        cadence,
        &start,
        &file_epoch,
        &[],
        &None,
        args,
        header.number_of_sector,
        args.cpu,
        None,
        times,
    )?
    .analysis_results;
    let (delay, rate) = match result.group_delay {
        Some(Ok(fit)) => (fit.delay_samples, fit.rate_hz),
        Some(Err(reason)) => {
            return Err(format!("Imaging fringe fit failed at {start}: {reason}").into())
        }
        None => (
            f64::from(result.residual_delay),
            f64::from(result.residual_rate),
        ),
    };
    Ok((delay, rate, f64::from(result.delay_snr)))
}

#[derive(Serialize)]
struct CleanComponentRecord {
    pixel_row: usize,
    pixel_col: usize,
    amplitude: f64,
    l_arcsec: f64,
    m_arcsec: f64,
    l_in_unit: f64,
    m_in_unit: f64,
    unit: String,
}

fn is_time_flagged(time: DateTime<Utc>, ranges: &[(DateTime<Utc>, DateTime<Utc>)]) -> bool {
    ranges
        .iter()
        .any(|&(start, end)| (start..end).contains(&time))
}

fn prepare_imaging_output_dir(input_path: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let imaging_dir = parent_dir.join("frinZ").join("imaging");
    fs::create_dir_all(&imaging_dir)?;
    Ok(imaging_dir)
}

fn save_clean_components(
    path: &Path,
    components: &[CleanComponent],
    size: usize,
    cell_size_arcsec: f64,
    angle_unit: AngleUnit,
) -> Result<(), Box<dyn Error>> {
    if components.is_empty() {
        fs::write(path, "[]")?;
        return Ok(());
    }

    let center = (size / 2) as f64;
    let cell_arcsec = cell_size_arcsec;

    let records: Vec<CleanComponentRecord> = components
        .iter()
        .map(|comp| {
            let col_offset = comp.col as f64 - center;
            let row_offset = center - comp.row as f64;
            let l_arcsec = col_offset * cell_arcsec;
            let m_arcsec = row_offset * cell_arcsec;
            CleanComponentRecord {
                pixel_row: comp.row,
                pixel_col: comp.col,
                amplitude: comp.amplitude,
                l_arcsec,
                m_arcsec,
                l_in_unit: (l_arcsec / 3600.0) * angle_unit.multiplier,
                m_in_unit: (m_arcsec / 3600.0) * angle_unit.multiplier,
                unit: angle_unit.label.to_string(),
            }
        })
        .collect();

    let json = to_string_pretty(&records)?;
    fs::write(path, json)?;
    Ok(())
}

fn select_cell_size_arcsec(visibilities: &[Visibility], override_value: Option<f64>) -> f64 {
    if let Some(value) = override_value {
        return value.max(1.0e-6);
    }
    if visibilities.is_empty() {
        return 1.0;
    }
    let max_baseline = visibilities
        .iter()
        .map(|v| (v.u * v.u + v.v * v.v).sqrt())
        .fold(0.0_f64, f64::max);
    if max_baseline <= 0.0 {
        return 1.0;
    }
    let safety = 1.05;
    let cell_size_rad = 1.0 / (2.0 * max_baseline * safety);
    (cell_size_rad * 180.0 * 3600.0 / PI).max(1.0e-6)
}

#[derive(Clone, Copy)]
struct AngleUnit {
    label: &'static str,
    multiplier: f64,
}

fn select_angle_unit(resolution_deg: f64) -> AngleUnit {
    const ANGLE_UNITS: [AngleUnit; 4] = [
        AngleUnit {
            label: "deg",
            multiplier: 1.0,
        },
        AngleUnit {
            label: "arcmin",
            multiplier: 60.0,
        },
        AngleUnit {
            label: "arcsec",
            multiplier: 3600.0,
        },
        AngleUnit {
            label: "mas",
            multiplier: 3_600_000.0,
        },
    ];

    if !resolution_deg.is_finite() || resolution_deg <= 0.0 {
        return ANGLE_UNITS[2];
    }

    for unit in ANGLE_UNITS {
        if resolution_deg * unit.multiplier >= 1.0 {
            return unit;
        }
    }

    ANGLE_UNITS.last().copied().unwrap()
}

fn format_angle_tick(value: f64, unit: AngleUnit) -> String {
    match unit.label {
        "deg" => format!("{:.2}", value),
        "arcmin" => format!("{:.2}", value),
        "arcsec" => format!("{:.1}", value),
        "mas" => format!("{:.0}", value),
        _ => format!("{:.2}", value),
    }
}

fn render_scalar_field_plot(
    path: &Path,
    data: &[f64],
    size: usize,
    cell_size_arcsec: f64,
    angle_unit: AngleUnit,
    symmetric: bool,
    title: &str,
) -> Result<(), Box<dyn Error>> {
    if data.len() != size * size {
        return Err("Image buffer length does not match the expected dimensions.".into());
    }
    let file_path = path
        .to_str()
        .ok_or_else(|| "Failed to convert path to string".to_string())?;

    let plot_pixels = (size as u32).saturating_mul(3).max(360);
    let colorbar_pixels = (plot_pixels / 4).max(80);
    let canvas_width = plot_pixels + colorbar_pixels + 160;
    let canvas_height = plot_pixels + 160;

    let backend = BitMapBackend::new(file_path, (canvas_width, canvas_height));
    let root = backend.into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(40, 40, 80, 80);
    let (plot_area, colorbar_area) = root.split_horizontally((plot_pixels + 40) as i32);

    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    let mut max_abs = 0.0_f64;
    for &value in data {
        if value.is_finite() {
            if value < min_val {
                min_val = value;
            }
            if value > max_val {
                max_val = value;
            }
            if value.abs() > max_abs {
                max_abs = value.abs();
            }
        }
    }

    if !min_val.is_finite() {
        min_val = 0.0;
    }
    if !max_val.is_finite() {
        max_val = 0.0;
    }
    if !max_abs.is_finite() {
        max_abs = 0.0;
    }

    let (range_min, range_max) = if symmetric {
        let abs_val = max_abs.max(1.0e-12);
        (-abs_val, abs_val)
    } else {
        let span = (max_val - min_val).max(1.0e-12);
        (min_val, min_val + span)
    };

    let cell_size_deg = cell_size_arcsec / 3600.0;
    let cell_size_unit = cell_size_deg * angle_unit.multiplier;
    let half_fov = cell_size_unit * (size as f64) / 2.0;
    let font = ("sans-serif", 26).into_font();
    let mut chart = ChartBuilder::on(&plot_area)
        .caption(title, font.clone())
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(-half_fov..half_fov, -half_fov..half_fov)?;

    chart
        .configure_mesh()
        .x_desc(format!("l [{}]", angle_unit.label))
        .y_desc(format!("m [{}]", angle_unit.label))
        .label_style(("sans-serif", 18))
        .axis_desc_style(("sans-serif", 20))
        .x_label_formatter(&|v| format_angle_tick(*v, angle_unit))
        .y_label_formatter(&|v| format_angle_tick(*v, angle_unit))
        .light_line_style(WHITE.mix(0.0))
        .draw()?;

    let half = (size as f64) / 2.0;
    let cell = cell_size_unit;
    let values: Vec<f64> = data
        .iter()
        .map(|&value| if value.is_finite() { value } else { 0.0 })
        .collect();
    let (plot_x, plot_y) = chart.plotting_area().get_pixel_range();
    let bitmap_width = plot_x.len().max(1) as u32;
    let bitmap_height = plot_y.len().max(1) as u32;
    let heatmap_buffer = rasterize_rgb_grid(
        bitmap_width,
        bitmap_height,
        size,
        size,
        &values,
        false,
        false,
        &|value| {
            let t = if symmetric {
                if range_max.abs() < 1.0e-12 {
                    0.5
                } else {
                    ((value / range_max) + 1.0) * 0.5
                }
            } else if (range_max - range_min).abs() < 1.0e-12 {
                0.5
            } else {
                (value - range_min) / (range_max - range_min)
            };
            jet_colormap(t.clamp(0.0, 1.0)).rgb()
        },
    );
    let bitmap = BitMapElement::<_, RGBPixel>::with_owned_buffer(
        (-half * cell, half * cell),
        (bitmap_width, bitmap_height),
        heatmap_buffer,
    )
    .ok_or("failed to create scalar-field bitmap")?;
    chart.draw_series(std::iter::once(bitmap))?;

    let cb_area = colorbar_area.margin(10, 10, 10, 60);
    let mut cb_chart = ChartBuilder::on(&cb_area)
        .margin_left(10)
        .margin_right(30)
        .margin_top(10)
        .margin_bottom(30)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .build_cartesian_2d(0.0..1.0, range_min..range_max)?;

    cb_chart
        .configure_mesh()
        .disable_x_axis()
        .label_style(("sans-serif", 16))
        .y_desc("Intensity")
        .axis_desc_style(("sans-serif", 18))
        .y_label_formatter(&|v| format!("{:.3}", v))
        .draw()?;

    let steps = canvas_height as usize;
    let span = (range_max - range_min).max(1.0e-12);
    let delta = span / steps as f64;
    cb_chart.plotting_area().draw(&Rectangle::new(
        [(0.0, range_min), (1.0, range_max)],
        WHITE.mix(0.0).filled(),
    ))?;
    for step in 0..steps {
        let v0 = range_min + delta * step as f64;
        let v1 = v0 + delta;
        let center_val = (v0 + v1) * 0.5;
        let t = if symmetric {
            if range_max.abs() < 1.0e-12 {
                0.5
            } else {
                ((center_val / range_max) + 1.0) * 0.5
            }
        } else {
            (center_val - range_min) / (range_max - range_min)
        };
        let color = jet_colormap(t.clamp(0.0, 1.0));
        cb_chart
            .plotting_area()
            .draw(&Rectangle::new([(0.0, v0), (1.0, v1)], color.filled()))?;
    }

    Ok(())
}

pub fn run_imaging_test() -> Result<(), Box<dyn Error>> {
    let l0 = 0.0;
    let num_points = 100usize;
    let max_uv = 500_000.0;
    let vis_data: Vec<Visibility> = (0..num_points)
        .map(|i| {
            let angle = i as f64 / num_points as f64 * 2.0 * PI;
            let u = max_uv * angle.cos();
            let v = max_uv * angle.sin();
            let phase = -2.0 * PI * u * l0;
            Visibility {
                u,
                v,
                w: 0.0,
                real: phase.cos(),
                imag: phase.sin(),
                weight: 1.0,
                time: i as f64,
            }
        })
        .collect();

    let image_size = 256;
    let dirty_image = perform_imaging(&vis_data, image_size, 0.1)?;
    let peak_index = dirty_image
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .ok_or("synthetic imaging produced an empty image")?;
    let peak_x = peak_index % image_size;
    let peak_y = peak_index / image_size;
    let expected_offset = (l0 / (0.1 * PI / (180.0 * 3600.0))).round() as isize;
    let expected_x = image_size as isize / 2 + expected_offset;
    if (peak_x as isize - expected_x).abs() > 2
        || (peak_y as isize - image_size as isize / 2).abs() > 2
    {
        return Err(format!(
            "synthetic source position mismatch: peak=({}, {}), expected=({}, {})",
            peak_x,
            peak_y,
            expected_x,
            image_size / 2
        )
        .into());
    }
    println!(
        "#IMAGING TEST: image={}x{} peak=({}, {}) value={:.6}",
        image_size, image_size, peak_x, peak_y, dirty_image[peak_index]
    );
    Ok(())
}

fn jet_colormap(t: f64) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    let four_t = 4.0 * t;
    let r = (four_t - 1.5).clamp(0.0, 1.0);
    let g = (1.5 - (four_t - 2.0).abs()).clamp(0.0, 1.0);
    let b = (1.5 - (four_t - 3.5)).clamp(0.0, 1.0);
    RGBColor((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_center_source_matches_the_weighted_psf() {
        let amplitude = 0.04;
        let visibilities = [
            Visibility {
                u: 96_000.0,
                v: -52_000.0,
                w: 0.0,
                real: amplitude,
                imag: 0.0,
                weight: 1.0,
                time: 0.0,
            },
            Visibility {
                u: -71_000.0,
                v: 95_000.0,
                w: 0.0,
                real: amplitude,
                imag: 0.0,
                weight: 5.0,
                time: 1.0,
            },
        ];
        for size in [31, 32] {
            let result =
                perform_imaging_with_config(&visibilities, &ImagingConfig::new(size, 0.5)).unwrap();
            let center = (size / 2) * size + size / 2;
            assert!((result.dirty_image[center] - amplitude).abs() < 1.0e-12);
            assert!((result.dirty_beam[center] - 1.0).abs() < 1.0e-12);
            for (dirty, beam) in result.dirty_image.iter().zip(&result.dirty_beam) {
                assert!((dirty - amplitude * beam).abs() < 1.0e-12);
            }
        }
    }
}
