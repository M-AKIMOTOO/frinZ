#![allow(dead_code)]

// Earth-rotation synthesis imaging module

use crate::args::Args;
use crate::bandpass::read_bandpass_file;
use crate::search;
use crate::fft::apply_phase_correction;
use crate::header::{parse_header, CorHeader};
use crate::processing::run_analysis_pipeline;
use crate::read::read_visibility_data;
use crate::rfi::parse_rfi_ranges;
use crate::utils;
use chrono::{DateTime, Utc};
use memmap2::Mmap;
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
use std::fs::File;
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
        || original_cols % target_cols != 0
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

fn pad_time_rows_to_power_of_two(
    data: &mut Vec<C32>,
    current_rows: usize,
    row_width: usize,
) -> usize {
    if current_rows == 0 || row_width == 0 {
        return current_rows;
    }
    let target_rows = if current_rows <= 1 {
        1
    } else {
        current_rows.next_power_of_two()
    };

    if target_rows > current_rows {
        let additional_samples = (target_rows - current_rows) * row_width;
        data.extend(std::iter::repeat(C32::new(0.0, 0.0)).take(additional_samples));
    }

    target_rows
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
        let mut parts = token.splitn(2, |c| c == ':' || c == '=');
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
    if config.cell_size_arcsec <= 0.0 {
        return Err("Cell size must be positive".into());
    }

    println!("Starting Earth-rotation synthesis imaging...");
    let gridded = grid_visibilities(visibilities, config)?;
    println!("Gridding complete.");

    let mut dirty_complex = fft_2d_inverse(&gridded.vis_grid, config.image_size)?;
    normalize_complex(&mut dirty_complex, config.image_size);
    fftshift_inplace(&mut dirty_complex, config.image_size);
    let dirty_image: Vec<f64> = dirty_complex.iter().map(|c| c.re).collect();
    println!("FFT complete. Dirty image created.");

    let sampling_complex: Vec<Complex<f64>> = gridded
        .sampling_grid
        .iter()
        .map(|&w| Complex::new(w, 0.0))
        .collect();
    let mut dirty_beam_complex = fft_2d_inverse(&sampling_complex, config.image_size)?;
    normalize_complex(&mut dirty_beam_complex, config.image_size);
    fftshift_inplace(&mut dirty_beam_complex, config.image_size);
    let mut dirty_beam: Vec<f64> = dirty_beam_complex.iter().map(|c| c.re).collect();
    normalize_beam_peak(&mut dirty_beam, config.image_size);

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
    let uv_cell = 1.0 / (size as f64 * cell_size_rad);
    let half = (size / 2) as f64;

    let mut accum = vec![GridAccum::default(); size * size];

    for vis in visibilities {
        let u_grid = vis.u / uv_cell + half;
        let v_grid = vis.v / uv_cell + half;

        let u0 = u_grid.floor();
        let v0 = v_grid.floor();
        let du = u_grid - u0;
        let dv = v_grid - v0;

        for (offset_u, weight_u) in [(0isize, 1.0 - du), (1, du)] {
            for (offset_v, weight_v) in [(0isize, 1.0 - dv), (1, dv)] {
                let u_idx = u0 as isize + offset_u;
                let v_idx = v0 as isize + offset_v;
                if u_idx < 0 || u_idx >= size as isize || v_idx < 0 || v_idx >= size as isize {
                    continue;
                }
                let kernel = weight_u * weight_v;
                if kernel <= 0.0 {
                    continue;
                }
                let index = (v_idx as usize) * size + (u_idx as usize);
                let weight = vis.weight * kernel;
                accum[index].real += vis.real * weight;
                accum[index].imag += vis.imag * weight;
                accum[index].weight += weight;
            }
        }

        // Hermitian symmetry (conjugate point)
        let u_sym = -vis.u;
        let v_sym = -vis.v;
        let u_grid_sym = u_sym / uv_cell + half;
        let v_grid_sym = v_sym / uv_cell + half;
        let u0s = u_grid_sym.floor();
        let v0s = v_grid_sym.floor();
        let dus = u_grid_sym - u0s;
        let dvs = v_grid_sym - v0s;

        for (offset_u, weight_u) in [(0isize, 1.0 - dus), (1, dus)] {
            for (offset_v, weight_v) in [(0isize, 1.0 - dvs), (1, dvs)] {
                let u_idx = u0s as isize + offset_u;
                let v_idx = v0s as isize + offset_v;
                if u_idx < 0 || u_idx >= size as isize || v_idx < 0 || v_idx >= size as isize {
                    continue;
                }
                let kernel = weight_u * weight_v;
                if kernel <= 0.0 {
                    continue;
                }
                let index = (v_idx as usize) * size + (u_idx as usize);
                let weight = vis.weight * kernel;
                accum[index].real += vis.real * weight;
                accum[index].imag -= vis.imag * weight;
                accum[index].weight += weight;
            }
        }
    }

    let mut vis_grid = Vec::with_capacity(size * size);
    let mut sampling_grid = Vec::with_capacity(size * size);
    for cell in accum {
        if cell.weight > 1e-12 {
            vis_grid.push(Complex::new(
                cell.real / cell.weight,
                cell.imag / cell.weight,
            ));
        } else {
            vis_grid.push(Complex::new(0.0, 0.0));
        }
        sampling_grid.push(cell.weight);
    }

    Ok(GriddedData {
        vis_grid,
        sampling_grid,
    })
}

fn fft_2d_inverse(grid: &[Complex<f64>], size: usize) -> Result<Vec<Complex<f64>>, String> {
    if grid.len() != size * size {
        return Err("Grid size does not match expected dimensions".to_string());
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(size);

    let mut buffer = grid.to_vec();

    for row in buffer.chunks_mut(size) {
        fft.process(row);
    }

    let mut transposed = vec![Complex::new(0.0, 0.0); size * size];
    for r in 0..size {
        for c in 0..size {
            transposed[c * size + r] = buffer[r * size + c];
        }
    }

    for row in transposed.chunks_mut(size) {
        fft.process(row);
    }

    let mut final_grid = vec![Complex::new(0.0, 0.0); size * size];
    for r in 0..size {
        for c in 0..size {
            final_grid[r * size + c] = transposed[c * size + r];
        }
    }

    Ok(final_grid)
}

fn normalize_complex(data: &mut [Complex<f64>], size: usize) {
    let norm = (size * size) as f64;
    for value in data.iter_mut() {
        *value /= norm;
    }
}

fn fftshift_inplace(data: &mut [Complex<f64>], size: usize) {
    let mut shifted = vec![Complex::new(0.0, 0.0); data.len()];
    let half = size / 2;
    for r in 0..size {
        for c in 0..size {
            let sr = (r + half) % size;
            let sc = (c + half) % size;
            shifted[sr * size + sc] = data[r * size + c];
        }
    }
    data.copy_from_slice(&shifted);
}

fn normalize_beam_peak(beam: &mut [f64], size: usize) {
    let center = (size / 2) * size + size / 2;
    let peak = beam
        .get(center)
        .copied()
        .filter(|p| p.abs() > 0.0)
        .unwrap_or_else(|| beam.iter().fold(0.0, |acc, &v| acc.max(v.abs())));
    if peak.abs() > 0.0 {
        for value in beam.iter_mut() {
            *value /= peak;
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
        println!("Residual image saved to {}", residual_path.display());
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
    let file = File::open(input_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut cursor = Cursor::new(&mmap[..]);

    let header = parse_header(&mut cursor)?;

    let original_fft_point = header.fft_point;
    let mut effective_fft_point = original_fft_point;

    if let Some(requested_fft) = args.fft_rebin {
        if requested_fft <= 0 {
            return Err("Error: --fft-rebin には正の値を指定してください。".into());
        }
        if requested_fft % 2 != 0 {
            return Err("Error: --fft-rebin は偶数である必要があります。".into());
        }
        if requested_fft > original_fft_point {
            return Err(format!(
                "Error: --fft-rebin ({}) はヘッダーの FFT 点数 ({}) を超えています。",
                requested_fft, original_fft_point
            )
            .into());
        }
        let original_half = (original_fft_point / 2) as usize;
        let requested_half = (requested_fft / 2) as usize;
        if requested_half == 0 || original_half % requested_half != 0 {
            return Err(format!(
                "Error: --fft-rebin ({}) は元のチャンネル数 ({}) を整数分割できません。",
                requested_fft, original_fft_point
            )
            .into());
        }
        effective_fft_point = requested_fft;
        println!(
            "#IMAGING FFT rebinning: {} → {} channels per polarization",
            original_half, requested_half
        );
    }

    let bandwidth_mhz = header.sampling_speed as f32 / 2.0 / 1_000_000.0;
    let rbw_mhz = bandwidth_mhz / effective_fft_point as f32 * 2.0;
    let rfi_ranges = parse_rfi_ranges(&args.rfi, rbw_mhz)?;
    let mut bandpass_data = if let Some(bp_path) = &args.bandpass {
        Some(read_bandpass_file(bp_path)?)
    } else {
        None
    };
    if effective_fft_point != original_fft_point {
        let original_half = (original_fft_point / 2) as usize;
        let target_half = (effective_fft_point / 2) as usize;
        if let Some(bp) = bandpass_data.as_mut() {
            if bp.len() == original_half {
                let rebinned = rebin_complex_rows(bp, 1, original_half, target_half);
                *bp = rebinned;
            } else if bp.len() != target_half {
                eprintln!(
                    "#WARN: バンドパスデータのチャンネル数 ({}) が FFT リビン後のチャンネル数 ({}) と一致しません。補正をスキップします。",
                    bp.len(),
                    target_half
                );
                *bp = Vec::new();
            }
        }
    }

    cursor.set_position(0);
    let (_, obs_time, effective_integ_time) =
        read_visibility_data(&mut cursor, &header, 1, 0, 0, false, pp_flag_ranges)?;
    cursor.set_position(256);

    let pp = header.number_of_sector;
    let mut length = if args.length == 0 { pp } else { args.length };

    let total_obs_time_seconds = pp as f32 * effective_integ_time;
    if args.length != 0 && args.length as f32 > total_obs_time_seconds {
        length = (total_obs_time_seconds / effective_integ_time).ceil() as i32;
    } else if args.length != 0 {
        length = (args.length as f32 / effective_integ_time).ceil() as i32;
    }

    if length <= 0 {
        return Err("Invalid integration length derived from arguments.".into());
    }

    let skip = args.skip.max(0);
    if skip >= pp {
        return Err(format!(
            "Skip value {} exceeds the available sectors ({}) in the input data.",
            skip, pp
        )
        .into());
    }

    let mut loop_count = if (pp - skip) / length <= 0 {
        1
    } else if (pp - skip) / length <= args.loop_ {
        (pp - skip) / length
    } else {
        args.loop_
    };

    if args.cumulate != 0 {
        if args.cumulate >= pp {
            return Err(format!(
                "The specified cumulation length, {} s, is more than the observation time, {} s.",
                args.cumulate, pp
            )
            .into());
        }
        length = args.cumulate;
        loop_count = pp / args.cumulate;
    }

    let mut visibilities = Vec::with_capacity(loop_count as usize);
    let mut skipped_low_snr = 0usize;
    let wavelength = SPEED_OF_LIGHT / header.observing_frequency;
    let search_mode = args.primary_search_mode();
    let mut prev_deep_solution: Option<(f32, f32)> = None;

    for l1 in 0..loop_count {
        let mut current_length = if args.cumulate != 0 {
            (l1 + 1) * length
        } else {
            length
        };

        let (mut complex_vec, current_obs_time, segment_integ_time) = match read_visibility_data(
            &mut cursor,
            &header,
            current_length,
            args.skip,
            l1,
            args.cumulate != 0,
            pp_flag_ranges,
        ) {
            Ok(data) => data,
            Err(_) => break,
        };

        if is_time_flagged(current_obs_time, time_flag_ranges) {
            println!(
                "#INFO: Skipping data at {} due to --flagging time range.",
                current_obs_time.format("%Y-%m-%d %H:%M:%S")
            );
            continue;
        }

        if complex_vec.is_empty() {
            continue;
        }

        let original_fft_half = (original_fft_point / 2) as usize;
        if original_fft_half == 0 {
            eprintln!("#ERROR: FFT point が不正です (0)。");
            continue;
        }
        if complex_vec.len() % original_fft_half != 0 {
            eprintln!(
                "#ERROR: 読み込んだデータ長 ({}) が元の FFT チャンネル数 ({}) の整数倍ではありません。",
                complex_vec.len(),
                original_fft_half
            );
            continue;
        }

        let mut actual_length = complex_vec.len() / original_fft_half;
        let physical_length = actual_length as i32;
        if effective_fft_point != original_fft_point {
            let target_half = (effective_fft_point / 2) as usize;
            complex_vec =
                rebin_complex_rows(&complex_vec, actual_length, original_fft_half, target_half);
            if target_half == 0 || complex_vec.len() % target_half != 0 {
                eprintln!(
                    "#ERROR: FFT リビン後のデータ長 ({}) が期待値 ({}×{}) と一致しません。",
                    complex_vec.len(),
                    actual_length,
                    target_half
                );
                continue;
            }
            actual_length = complex_vec.len() / target_half;
        }

        if actual_length == 0 {
            continue;
        }

        let fft_point_half_used = (effective_fft_point / 2) as usize;
        actual_length =
            pad_time_rows_to_power_of_two(&mut complex_vec, actual_length, fft_point_half_used);

        if current_length != actual_length as i32 {
            current_length = actual_length as i32;
        }

        let correction = determine_segment_correction(
            &complex_vec,
            &header,
            args,
            current_length,
            physical_length,
            segment_integ_time,
            &current_obs_time,
            &obs_time,
            &rfi_ranges,
            &bandpass_data,
            search_mode,
            &mut prev_deep_solution,
        )?;

        if let Some(min_snr) = imaging_opts.vis_snr_min {
            if correction.snr < min_snr {
                skipped_low_snr += 1;
                continue;
            }
        }

        let start_time_offset_sec = 0.0;

        let corrected = apply_phase_correction(
            &reshape_to_complex64_matrix(&complex_vec, fft_point_half_used),
            correction.rate,
            correction.delay,
            correction.acel,
            segment_integ_time,
            header.sampling_speed as u32,
            effective_fft_point as u32,
            start_time_offset_sec,
        );

        let averaged = average_complex_matrix(&corrected);
        let amplitude = averaged.norm();
        if amplitude < 1e-10 {
            continue;
        }

        println!(
            "#IMAGING segment {:03}: {} delay={:.3} samp rate={:.6} Hz SNR={:.2}",
            l1,
            current_obs_time.format("%Y-%m-%d %H:%M:%S"),
            correction.residual_delay,
            correction.residual_rate,
            correction.snr
        );

        let (u_m, v_m, w_m, _, _) = utils::uvw_cal(
            header.station1_position,
            header.station2_position,
            current_obs_time,
            header.source_position_ra,
            header.source_position_dec,
            true,
        );

        let time_seconds = current_obs_time.timestamp() as f64
            + f64::from(current_obs_time.timestamp_subsec_nanos()) * 1e-9;

        visibilities.push(Visibility {
            u: u_m / wavelength,
            v: v_m / wavelength,
            w: w_m / wavelength,
            real: averaged.re,
            imag: averaged.im,
            weight: correction.snr.max(1.0),
            time: time_seconds,
        });
    }

    if skipped_low_snr > 0 {
        println!(
            "Visibility SNR filter removed {} segment(s) below threshold.",
            skipped_low_snr
        );
    }

    Ok((visibilities, header, effective_integ_time as f64))
}

struct SegmentCorrection {
    delay: f32,
    rate: f32,
    acel: f32,
    weight: f64,
    snr: f64,
    residual_delay: f32,
    residual_rate: f32,
}

fn determine_segment_correction(
    complex_vec: &[C32],
    header: &CorHeader,
    args: &Args,
    current_length: i32,
    physical_length: i32,
    segment_integ_time: f32,
    current_obs_time: &DateTime<Utc>,
    obs_time: &DateTime<Utc>,
    rfi_ranges: &[(usize, usize)],
    bandpass_data: &Option<Vec<C32>>,
    search_mode: Option<&str>,
    prev_deep_solution: &mut Option<(f32, f32)>,
) -> Result<SegmentCorrection, Box<dyn Error>> {
    if current_length <= 0 {
        return Err("セグメント長が無効です (0 以下)".into());
    }
    let fft_point_half = complex_vec.len() / current_length as usize;
    if fft_point_half == 0 {
        return Err("FFT チャンネル数が 0 です (earth-rotation imaging)".into());
    }
    let effective_fft_point = (fft_point_half * 2) as i32;

    match search_mode {
        Some("deep") => {
            let deep = search::run_deep_search(
                complex_vec,
                header,
                current_length,
                physical_length,
                segment_integ_time,
                current_obs_time,
                obs_time,
                rfi_ranges,
                bandpass_data,
                args,
                header.number_of_sector,
                args.cpu,
                *prev_deep_solution,
            )?;
            *prev_deep_solution = Some((
                deep.analysis_results.residual_delay,
                deep.analysis_results.residual_rate,
            ));
            Ok(SegmentCorrection {
                delay: deep.analysis_results.residual_delay,
                rate: deep.analysis_results.residual_rate,
                acel: args.acel_correct,
                weight: deep.analysis_results.delay_snr as f64,
                snr: deep.analysis_results.delay_snr as f64,
                residual_delay: deep.analysis_results.residual_delay,
                residual_rate: deep.analysis_results.residual_rate,
            })
        }
        Some("peak") => {
            let iterations = args.iter.max(1) as usize;
            let mut total_delay = args.delay_correct;
            let mut total_rate = args.rate_correct;

            for _ in 0..iterations {
                let (results, _, _) = run_analysis_pipeline(
                    complex_vec,
                    header,
                    args,
                    Some("peak"),
                    total_delay,
                    total_rate,
                    args.acel_correct,
                    current_length,
                    physical_length,
                    segment_integ_time,
                    current_obs_time,
                    obs_time,
                    rfi_ranges,
                    bandpass_data,
                    effective_fft_point,
                )?;
                total_delay += results.delay_offset;
                total_rate += results.rate_offset;
            }

            let (final_results, _, _) = run_analysis_pipeline(
                complex_vec,
                header,
                args,
                Some("peak"),
                total_delay,
                total_rate,
                args.acel_correct,
                current_length,
                physical_length,
                segment_integ_time,
                current_obs_time,
                obs_time,
                rfi_ranges,
                bandpass_data,
                effective_fft_point,
            )?;

            Ok(SegmentCorrection {
                delay: total_delay,
                rate: total_rate,
                acel: args.acel_correct,
                weight: final_results.delay_snr as f64,
                snr: final_results.delay_snr as f64,
                residual_delay: final_results.residual_delay,
                residual_rate: final_results.residual_rate,
            })
        }
        _ => {
            let (results, _, _) = run_analysis_pipeline(
                complex_vec,
                header,
                args,
                None,
                args.delay_correct,
                args.rate_correct,
                args.acel_correct,
                current_length,
                physical_length,
                segment_integ_time,
                current_obs_time,
                obs_time,
                rfi_ranges,
                bandpass_data,
                effective_fft_point,
            )?;
            Ok(SegmentCorrection {
                delay: args.delay_correct,
                rate: args.rate_correct,
                acel: args.acel_correct,
                weight: results.delay_snr as f64,
                snr: results.delay_snr as f64,
                residual_delay: results.residual_delay,
                residual_rate: results.residual_rate,
            })
        }
    }
}

fn reshape_to_complex64_matrix(data: &[C32], columns: usize) -> Vec<Vec<Complex<f64>>> {
    data.chunks(columns)
        .map(|chunk| {
            chunk
                .iter()
                .map(|&c| Complex::new(c.re as f64, c.im as f64))
                .collect()
        })
        .collect()
}

fn average_complex_matrix(matrix: &[Vec<Complex<f64>>]) -> Complex<f64> {
    let mut sum = Complex::new(0.0, 0.0);
    let mut count = 0usize;
    for row in matrix {
        for value in row {
            sum += *value;
            count += 1;
        }
    }
    if count == 0 {
        Complex::new(0.0, 0.0)
    } else {
        sum / count as f64
    }
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
        .any(|(start, end)| time >= *start && time <= *end)
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

    let center = (size as f64) / 2.0;
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
        .light_line_style(&WHITE.mix(0.0))
        .draw()?;

    let half = (size as f64) / 2.0;
    let cell = cell_size_unit;
    chart.draw_series((0..size).flat_map(|row| {
        let y0 = ((size - row) as f64 - half) * cell;
        let y1 = ((size - row - 1) as f64 - half) * cell;
        (0..size).map(move |col| {
            let idx = row * size + col;
            let value = if data[idx].is_finite() {
                data[idx]
            } else {
                0.0
            };
            let x0 = (col as f64 - half) * cell;
            let x1 = ((col + 1) as f64 - half) * cell;
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
            let color = jet_colormap(t.clamp(0.0, 1.0));
            Rectangle::new([(x0, y0), (x1, y1)], color.filled())
        })
    }))?;

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

fn jet_colormap(t: f64) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    let four_t = 4.0 * t;
    let r = (four_t - 1.5).clamp(0.0, 1.0);
    let g = (1.5 - (four_t - 2.0).abs()).clamp(0.0, 1.0);
    let b = (1.5 - (four_t - 3.5)).clamp(0.0, 1.0);
    RGBColor((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}
