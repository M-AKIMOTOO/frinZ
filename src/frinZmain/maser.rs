use ndarray::{Array1, Axis};
use plotters::prelude::*;
use crate::png_compress::{compress_png_with_mode, CompressQuality};
use std::error::Error;
use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::args::Args;
use crate::fft::process_fft;
use crate::header::{parse_header, CorHeader};
use crate::read::read_visibility_data;
use crate::rfi::parse_rfi_ranges;

// New use statements for LSR velocity correction
use astro::ecliptic;
use astro::nutation;
use astro::planet::{self, earth, Planet};
use astro::time::{self, Date};

use chrono::{DateTime, Datelike, Duration, Timelike, Utc};
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::{storage::Owned, DMatrix, DVector, Dyn, Matrix3, Vector3};

const C_KM_S: f64 = 299792.458; // Speed of light in km/s
const FWHM_TO_SIGMA: f64 = 0.42466090014400953; // 1 / (2 * sqrt(2 ln 2))
const MIN_FWHM_KMS: f64 = 1.0e-3; // clamp to avoid zero-width components

macro_rules! maser_logln {
    ($log:expr, $($arg:tt)*) => {{
        let line = format!($($arg)*);
        println!("{}", line);
        $log.push(line);
    }};
}

fn write_maser_log(path: &Path, lines: &[String]) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    for line in lines {
        writeln!(file, "{}", line)?;
    }
    Ok(())
}

fn fft_precision_digits(fft_point: i32) -> usize {
    if fft_point <= 0 {
        return 3;
    }
    let digits = ((fft_point as f64).log10().floor() as isize + 1).max(1) as usize;
    digits.min(6)
}

fn format_with_precision(value: f64, digits: usize) -> String {
    if digits == 0 {
        format!("{:.0}", value)
    } else {
        format!("{:.*}", digits, value)
    }
}

fn format_f32_precision(value: f64, digits: usize) -> String {
    let digits = digits.min(6);
    let value_f32 = value as f32;
    format_with_precision(value_f32 as f64, digits)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaserMode {
    Seg,
    Stacked,
    Both,
}

impl MaserMode {
    fn write_segment_outputs(self) -> bool {
        matches!(self, MaserMode::Seg | MaserMode::Both)
    }

    fn accumulate_integration(self) -> bool {
        matches!(self, MaserMode::Stacked | MaserMode::Both)
    }

    fn print_segment_table(self) -> bool {
        matches!(self, MaserMode::Seg | MaserMode::Stacked | MaserMode::Both)
    }

    fn as_str(self) -> &'static str {
        match self {
            MaserMode::Seg => "seg",
            MaserMode::Stacked => "stacked",
            MaserMode::Both => "both",
        }
    }
}

impl Default for MaserMode {
    fn default() -> Self {
        MaserMode::Seg
    }
}

impl FromStr for MaserMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "seg" => Ok(MaserMode::Seg),
            "stacked" | "integ" | "integ-only" | "integration" => Ok(MaserMode::Stacked),
            "both" => Ok(MaserMode::Both),
            other => Err(format!(
                "Unknown maser mode '{}'. Expected seg, stacked, or both.",
                other
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BaselineFitKind {
    Linear,
    Quad,
}

impl BaselineFitKind {
    fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "linear" | "lin" => Some(BaselineFitKind::Linear),
            "quad" | "quadratic" => Some(BaselineFitKind::Quad),
            _ => None,
        }
    }

    fn degree(self) -> usize {
        match self {
            BaselineFitKind::Linear => 1,
            BaselineFitKind::Quad => 2,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            BaselineFitKind::Linear => "linear",
            BaselineFitKind::Quad => "quad",
        }
    }
}

#[derive(Debug, Clone)]
enum OffSpec {
    File(PathBuf),
    Baseline(BaselineFitKind),
}

#[derive(Debug)]
struct GaussianFitResult {
    components: Vec<(f64, f64, f64)>,
    residual_norm: f64,
    termination: TerminationReason,
    evaluations: usize,
}

struct GaussianMixtureProblem<'a> {
    velocities: &'a [f64],
    values: &'a [f64],
    params: DVector<f64>,
    components: usize,
}

impl<'a> GaussianMixtureProblem<'a> {
    fn new(velocities: &'a [f64], values: &'a [f64], initial: &[f64]) -> Self {
        let params = DVector::from_column_slice(initial);
        let mut problem = Self {
            velocities,
            values,
            params,
            components: initial.len() / 3,
        };
        problem.enforce_constraints();
        problem
    }

    fn enforce_constraints(&mut self) {
        for idx in 0..self.components {
            let width_idx = idx * 3 + 2;
            let val = self.params[width_idx].abs().max(MIN_FWHM_KMS);
            self.params[width_idx] = val;
        }
    }

    fn components(&self) -> Vec<(f64, f64, f64)> {
        let mut out = Vec::with_capacity(self.components);
        for idx in 0..self.components {
            let base = idx * 3;
            out.push((
                self.params[base],
                self.params[base + 1],
                self.params[base + 2].max(MIN_FWHM_KMS),
            ));
        }
        out
    }

    fn predicted_value(params: &DVector<f64>, components: usize, vel: f64) -> f64 {
        let mut total = 0.0;
        for idx in 0..components {
            let base = idx * 3;
            let amp = params[base];
            let center = params[base + 1];
            let fwhm = params[base + 2].abs().max(MIN_FWHM_KMS);
            let sigma = fwhm * FWHM_TO_SIGMA;
            let delta = vel - center;
            let sigma_sq = sigma * sigma;
            let exponent = -delta * delta / (2.0 * sigma_sq);
            total += amp * exponent.exp();
        }
        total
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, Dyn> for GaussianMixtureProblem<'a> {
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;
    type ParameterStorage = Owned<f64, Dyn>;

    fn set_params(&mut self, x: &DVector<f64>) {
        self.params.copy_from(x);
        self.enforce_constraints();
    }

    fn params(&self) -> DVector<f64> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        if self.velocities.len() != self.values.len() {
            return None;
        }
        let mut residuals = DVector::zeros(self.velocities.len());
        for (idx, (&vel, &target)) in self.velocities.iter().zip(self.values.iter()).enumerate() {
            let predicted = Self::predicted_value(&self.params, self.components, vel);
            residuals[idx] = predicted - target;
        }
        Some(residuals)
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        if self.velocities.len() != self.values.len() {
            return None;
        }
        let rows = self.velocities.len();
        let cols = self.components * 3;
        let mut jac = DMatrix::zeros(rows, cols);
        for (row_idx, &vel) in self.velocities.iter().enumerate() {
            for component_idx in 0..self.components {
                let base = component_idx * 3;
                let amp = self.params[base];
                let center = self.params[base + 1];
                let fwhm = self.params[base + 2].abs().max(MIN_FWHM_KMS);
                let sigma = fwhm * FWHM_TO_SIGMA;
                let delta = vel - center;
                let sigma_sq = sigma * sigma;
                let exp_term = (-delta * delta / (2.0 * sigma_sq)).exp();
                let value = amp * exp_term;

                jac[(row_idx, base)] = exp_term; // ∂/∂amp
                jac[(row_idx, base + 1)] = value * (delta / sigma_sq); // ∂/∂center

                let sigma_cubed = sigma_sq * sigma;
                jac[(row_idx, base + 2)] = value * ((delta * delta) / sigma_cubed) * FWHM_TO_SIGMA;
                // ∂/∂fwhm
            }
        }
        Some(jac)
    }
}

fn flatten_gaussian_components(components: &[(f64, f64, f64)]) -> Vec<f64> {
    let mut flat = Vec::with_capacity(components.len() * 3);
    for &(amp, center, fwhm) in components {
        flat.push(amp);
        flat.push(center);
        flat.push(fwhm);
    }
    flat
}

fn evaluate_gaussian_mixture(
    velocities: &[f64],
    components: &[(f64, f64, f64)],
) -> Vec<(f64, f32)> {
    velocities
        .iter()
        .map(|&vel| {
            let mut total = 0.0;
            for &(amp, center, fwhm) in components {
                let sigma = fwhm.abs().max(MIN_FWHM_KMS) * FWHM_TO_SIGMA;
                let delta = vel - center;
                let sigma_sq = sigma * sigma;
                total += amp * (-delta * delta / (2.0 * sigma_sq)).exp();
            }
            (vel, total as f32)
        })
        .collect()
}

fn fit_gaussian_mixture(
    velocities: &[f64],
    values: &[f64],
    initial_components: &[(f64, f64, f64)],
) -> Result<GaussianFitResult, Box<dyn Error>> {
    if velocities.len() != values.len() {
        return Err("Velocity and spectrum length mismatch.".into());
    }
    if initial_components.is_empty() {
        return Err("No Gaussian components provided.".into());
    }

    let initial_flat = flatten_gaussian_components(initial_components);
    let problem = GaussianMixtureProblem::new(velocities, values, &initial_flat);
    let (problem, report) = LevenbergMarquardt::new().minimize(problem);

    let residual_vec = problem
        .residuals()
        .unwrap_or_else(|| DVector::zeros(velocities.len()));
    let result = GaussianFitResult {
        components: problem.components(),
        residual_norm: residual_vec.norm(),
        termination: report.termination,
        evaluations: report.number_of_evaluations,
    };
    Ok(result)
}

fn median_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

fn mad_f32(values: &[f32], median: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut dev: Vec<f32> = values.iter().map(|&v| (v - median).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = dev.len() / 2;
    if dev.len() % 2 == 0 {
        0.5 * (dev[mid - 1] + dev[mid])
    } else {
        dev[mid]
    }
}

fn fit_polynomial_least_squares(
    x_values: &[f64],
    y_values: &[f32],
    degree: usize,
) -> Option<(Vec<f64>, f64)> {
    if x_values.len() != y_values.len() || x_values.len() < degree + 1 {
        return None;
    }

    let x_center = x_values.iter().sum::<f64>() / x_values.len() as f64;
    let rows = x_values.len();
    let cols = degree + 1;
    let mut design = DMatrix::<f64>::zeros(rows, cols);
    for (row, &x) in x_values.iter().enumerate() {
        let shifted = x - x_center;
        let mut term = 1.0;
        for col in 0..cols {
            design[(row, col)] = term;
            term *= shifted;
        }
    }
    let y = DVector::<f64>::from_iterator(rows, y_values.iter().map(|&v| v as f64));
    let transpose = design.transpose();
    let mut ata = &transpose * &design;
    let aty = &transpose * y;

    // Regularize very ill-conditioned systems to keep baseline fitting stable.
    let ridge = 1.0e-12;
    for i in 0..cols {
        ata[(i, i)] += ridge;
    }

    let coeff = ata.lu().solve(&aty)?;
    Some((coeff.iter().copied().collect(), x_center))
}

fn evaluate_polynomial(coeff: &[f64], x: f64, x_center: f64) -> f32 {
    let shifted = x - x_center;
    let mut sum = 0.0;
    let mut term = 1.0;
    for &c in coeff {
        sum += c * term;
        term *= shifted;
    }
    sum as f32
}

struct BaselineFitResult {
    baseline: Vec<f32>,
    used_mask: Vec<bool>,
    used_count: usize,
    sigma_est: f32,
    coeff_shifted: Vec<f64>,
    x_center_mhz: f64,
}

fn fit_baseline_robust(
    x_values: &[f64],
    y_values: &[f32],
    model: BaselineFitKind,
) -> Result<BaselineFitResult, Box<dyn Error>> {
    if x_values.len() != y_values.len() {
        return Err("Baseline fit failed: x/y length mismatch.".into());
    }
    let degree = model.degree();
    if x_values.len() < degree + 2 {
        return Err(format!(
            "Baseline fit failed: not enough points for {} (need >= {}, got {}).",
            model.as_str(),
            degree + 2,
            x_values.len()
        )
        .into());
    }

    let mut mask = vec![true; x_values.len()];
    let max_iter = 6;
    let mut last_sigma = 0.0_f32;

    for _ in 0..max_iter {
        let x_masked: Vec<f64> = x_values
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| if mask[i] { Some(x) } else { None })
            .collect();
        let y_masked: Vec<f32> = y_values
            .iter()
            .enumerate()
            .filter_map(|(i, &y)| if mask[i] { Some(y) } else { None })
            .collect();

        if x_masked.len() < degree + 2 {
            break;
        }

        let (coeff, x_center) = match fit_polynomial_least_squares(&x_masked, &y_masked, degree) {
            Some(v) => v,
            None => break,
        };

        let residuals: Vec<f32> = x_values
            .iter()
            .zip(y_values.iter())
            .map(|(&x, &y)| y - evaluate_polynomial(&coeff, x, x_center))
            .collect();

        let residuals_masked: Vec<f32> = residuals
            .iter()
            .enumerate()
            .filter_map(|(i, &r)| if mask[i] { Some(r) } else { None })
            .collect();
        if residuals_masked.len() < degree + 2 {
            break;
        }

        let median = median_f32(&residuals_masked);
        let mad = mad_f32(&residuals_masked, median);
        let sigma = if mad > 1.0e-9 { mad / 0.6745_f32 } else { 0.0 };
        last_sigma = sigma;
        if sigma <= 1.0e-12 {
            break;
        }

        let threshold = 4.0_f32 * sigma;
        let new_mask: Vec<bool> = residuals
            .iter()
            .map(|&r| (r - median).abs() <= threshold)
            .collect();

        let valid_count = new_mask.iter().filter(|&&ok| ok).count();
        if valid_count < degree + 2 || new_mask == mask {
            break;
        }
        mask = new_mask;
    }

    let x_final: Vec<f64> = x_values
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| if mask[i] { Some(x) } else { None })
        .collect();
    let y_final: Vec<f32> = y_values
        .iter()
        .enumerate()
        .filter_map(|(i, &y)| if mask[i] { Some(y) } else { None })
        .collect();

    let (coeff, x_center, used_mask) = if x_final.len() >= degree + 2 {
        let (coeff, x_center) = fit_polynomial_least_squares(&x_final, &y_final, degree)
            .ok_or("Baseline fit failed at final solve.")?;
        (coeff, x_center, mask)
    } else {
        let (coeff, x_center) = fit_polynomial_least_squares(x_values, y_values, degree)
            .ok_or("Baseline fit fallback solve failed.")?;
        (coeff, x_center, vec![true; x_values.len()])
    };

    let baseline: Vec<f32> = x_values
        .iter()
        .map(|&x| evaluate_polynomial(&coeff, x, x_center))
        .collect();

    let used_count = used_mask.iter().filter(|&&ok| ok).count();
    Ok(BaselineFitResult {
        baseline,
        used_mask,
        used_count,
        sigma_est: last_sigma,
        coeff_shifted: coeff,
        x_center_mhz: x_center,
    })
}

/// Extracts the cross-power spectrum at zero fringe rate from a .cor file.
#[derive(Clone)]
struct SpectrumData {
    header: CorHeader,
    spectrum: Array1<f32>,
    start_time: DateTime<Utc>,
    sector_count: usize,
    effective_integration_time: f32,
}

struct SegmentSummary {
    index: usize,
    start_time: DateTime<Utc>,
    segment_seconds: f32,
    lsr_average: f64,
    peak_freq_mhz: f64,
    peak_velocity_kms: f64,
    peak_value: f32,
    median: f32,
    mad: f32,
    peak_minus_median: f32,
    snr_mad: f32,
    snr_stdev: f32,
    sigma_est: f32,
}

struct IntegrationState {
    base_freq_axis_mhz: Vec<f64>,
    base_topo_velocity_kms: Vec<f64>,
    base_lsr_corr: f64,
    sum_spec: Vec<f64>,
    weights: Vec<f64>,
    segments: usize,
    freq_resolution_mhz: f64,
    channel_width_kms: f64,
    total_segment_seconds: f64,
    lsr_sum: f64,
}

impl IntegrationState {
    fn new() -> Self {
        Self {
            base_freq_axis_mhz: Vec::new(),
            base_topo_velocity_kms: Vec::new(),
            base_lsr_corr: 0.0,
            sum_spec: Vec::new(),
            weights: Vec::new(),
            segments: 0,
            freq_resolution_mhz: 0.0,
            channel_width_kms: 0.0,
            total_segment_seconds: 0.0,
            lsr_sum: 0.0,
        }
    }

    fn add_segment(
        &mut self,
        lsr_corr: f64,
        segment_seconds: f32,
        freq_axis_mhz: &[f64],
        topo_velocity_kms: &[f64],
        normalized_spec: &[f32],
        freq_resolution_mhz: f64,
        channel_width_kms: f64,
    ) {
        if normalized_spec.len() != freq_axis_mhz.len()
            || topo_velocity_kms.len() != freq_axis_mhz.len()
        {
            return;
        }

        self.total_segment_seconds += segment_seconds as f64;
        self.lsr_sum += lsr_corr;

        if self.segments == 0 {
            self.base_freq_axis_mhz = freq_axis_mhz.to_vec();
            self.base_topo_velocity_kms = topo_velocity_kms.to_vec();
            self.base_lsr_corr = lsr_corr;
            self.sum_spec = normalized_spec.iter().map(|&v| v as f64).collect();
            self.weights = vec![1.0; normalized_spec.len()];
            self.freq_resolution_mhz = freq_resolution_mhz;
            self.channel_width_kms = channel_width_kms;
            self.segments = 1;
            return;
        }

        if self.base_freq_axis_mhz.len() != freq_axis_mhz.len()
            || self.base_topo_velocity_kms.len() != topo_velocity_kms.len()
            || self.sum_spec.len() != normalized_spec.len()
        {
            return;
        }

        self.freq_resolution_mhz = freq_resolution_mhz;
        self.channel_width_kms = channel_width_kms;

        let len = normalized_spec.len();
        if len < 2 {
            return;
        }

        let vel_step = self.base_topo_velocity_kms[1] - self.base_topo_velocity_kms[0];
        if vel_step.abs() < f64::EPSILON {
            return;
        }

        let shift = (lsr_corr - self.base_lsr_corr) / vel_step;

        for idx in 0..len {
            let pos = idx as f64 - shift;
            if pos < 0.0 || pos > (len - 1) as f64 {
                continue;
            }
            let lower = pos.floor() as usize;
            let frac = pos - lower as f64;
            let interpolated = if lower + 1 < len {
                let low = normalized_spec[lower] as f64;
                let high = normalized_spec[lower + 1] as f64;
                low * (1.0 - frac) + high * frac
            } else {
                normalized_spec[lower] as f64
            };
            self.sum_spec[idx] += interpolated;
            self.weights[idx] += 1.0;
        }

        self.segments += 1;
    }

    fn finalize(&self) -> Option<IntegrationResult> {
        if self.segments == 0 {
            return None;
        }
        let mut averaged = Vec::with_capacity(self.sum_spec.len());
        for (idx, &sum) in self.sum_spec.iter().enumerate() {
            let weight = self.weights.get(idx).copied().unwrap_or(0.0);
            if weight > 0.0 {
                averaged.push((sum / weight) as f32);
            } else {
                averaged.push(0.0);
            }
        }
        let mean_lsr = self.lsr_sum / self.segments as f64;
        let velocity_axis_kms: Vec<f64> = self
            .base_topo_velocity_kms
            .iter()
            .map(|&v| v + mean_lsr)
            .collect();
        Some(IntegrationResult {
            velocity_axis_kms,
            frequency_axis_mhz: self.base_freq_axis_mhz.clone(),
            averaged_spec: averaged,
            segment_count: self.segments,
            freq_resolution_mhz: self.freq_resolution_mhz,
            channel_width_kms: self.channel_width_kms,
            total_segment_seconds: self.total_segment_seconds,
            mean_lsr_corr: mean_lsr,
        })
    }
}

struct IntegrationResult {
    velocity_axis_kms: Vec<f64>,
    frequency_axis_mhz: Vec<f64>,
    averaged_spec: Vec<f32>,
    segment_count: usize,
    freq_resolution_mhz: f64,
    channel_width_kms: f64,
    total_segment_seconds: f64,
    mean_lsr_corr: f64,
}

impl IntegrationResult {
    fn to_velocity_axis(&self) -> Vec<f64> {
        self.velocity_axis_kms.clone()
    }

    fn peak(&self) -> Option<(usize, f32)> {
        self.averaged_spec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, &val)| (idx, val))
    }
}

fn compute_lsr_average(
    header: &CorHeader,
    start_time: DateTime<Utc>,
    effective_integration_time: f32,
    sector_count: usize,
) -> f64 {
    if sector_count == 0 {
        return calculate_lsr_velocity_correction(
            header.station1_position,
            &start_time,
            header.source_position_ra,
            header.source_position_dec,
        );
    }

    let step_seconds = if effective_integration_time.abs() > 1e-9 {
        effective_integration_time as f64
    } else {
        1.0
    };

    let mut total = 0.0;
    for idx in 0..sector_count {
        let offset_nanos = (step_seconds * idx as f64 * 1e9).round() as i64;
        let sector_time = start_time + Duration::nanoseconds(offset_nanos);
        total += calculate_lsr_velocity_correction(
            header.station1_position,
            &sector_time,
            header.source_position_ra,
            header.source_position_dec,
        );
    }

    total / sector_count as f64
}

fn get_spectrum_segment(
    file_path: &Path,
    args: &Args,
    sampling_scale: f64,
    chunk_length: i32,
    loop_index: i32,
    log_lines: &mut Vec<String>,
) -> Result<Option<SpectrumData>, Box<dyn Error>> {
    let mut file = File::open(file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let mut cursor = Cursor::new(buffer.as_slice());

    let header = parse_header(&mut cursor)?;

    let scaled_sampling_speed = (header.sampling_speed as f64 * sampling_scale) as f32;
    let sampling_speed_for_fft = (header.sampling_speed as f64 * sampling_scale).round() as i32;

    let desired_length = if chunk_length > 0 {
        chunk_length
    } else {
        header.number_of_sector
    };

    let (complex_vec, obs_time, effective_integ_time) = read_visibility_data(
        &mut cursor,
        &header,
        desired_length,
        args.skip,
        loop_index,
        false,
        &[],
    )?;

    if complex_vec.is_empty() {
        maser_logln!(
            log_lines,
            "  [maser] loop {}: {:?} のデータ長 {} セクターが不足 (skip={}, loop offset {}) ため空データ。",
            loop_index,
            file_path,
            desired_length,
            args.skip,
            loop_index
        );
        return Ok(None);
    }

    let rbw = (scaled_sampling_speed / header.fft_point as f32) / 1e6;
    let rfi_ranges = parse_rfi_ranges(&args.rfi, rbw)?;

    let fft_point_half = (header.fft_point / 2) as usize;
    let sector_count = if fft_point_half > 0 {
        complex_vec.len() / fft_point_half
    } else {
        0
    };

    if sector_count == 0 {
        maser_logln!(
            log_lines,
            "  [maser] loop {}: {:?} で FFT 点数 {} に対する複素データが得られず、セクター数 0。",
            loop_index,
            file_path,
            header.fft_point
        );
        return Ok(None);
    }

    let (freq_rate_array, padding_length) = process_fft(
        &complex_vec,
        sector_count as i32,
        header.fft_point,
        sampling_speed_for_fft,
        &rfi_ranges,
        args.rate_padding,
    );

    // Get spectrum at zero rate (center of rate dimension)
    let zero_rate_idx = padding_length / 2;
    let spectrum_complex = freq_rate_array.index_axis(Axis(1), zero_rate_idx);

    let spectrum_abs = spectrum_complex.mapv(|x| x.norm());

    Ok(Some(SpectrumData {
        header,
        spectrum: spectrum_abs,
        start_time: obs_time,
        sector_count,
        effective_integration_time: effective_integ_time,
    }))
}

fn plot_maser_spectrum(
    output_path: &Path,
    data: &[(f64, f32)],
    x_label: &str,
    title: &str,
    y_label: &str,
    antenna_label: &str,
    peak_freq: f64,
    peak_velocity: f64,
    peak_val: f32,
    snr_mad: f32,
    snr_stdev: f32,
    median: f32,
    mad: f32,
    sigma_est: f32,
    freq_resolution_mhz: f64,
    channel_width_kms: f64,
    freq_precision: usize,
    vel_precision: usize,
    gaussian_fit: Option<&[(f64, f32)]>,
    gaussian_params: Option<&[(f64, f64, f64)]>,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let (min_x, max_x) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (x, _)| {
            (min.min(*x), max.max(*x))
        });
    let (min_y, max_y) = data
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), (_, y)| {
            (min.min(*y), max.max(*y))
        });

    let y_margin = (max_y - min_y) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(min_x..max_x, (min_y - y_margin)..(max_y + y_margin))?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .label_style(("sans-serif", 20))
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .light_line_style(&TRANSPARENT)
        .draw()?;

    chart.draw_series(LineSeries::new(data.iter().cloned(), &BLUE))?;

    if let Some(fit_data) = gaussian_fit {
        chart.draw_series(LineSeries::new(fit_data.iter().cloned(), &RED))?;
    }

    // Draw reference lines for median and MAD bounds
    let median_line: Vec<(f64, f32)> = vec![(min_x, median), (max_x, median)];
    chart.draw_series(LineSeries::new(
        median_line,
        ShapeStyle::from(&RED.mix(0.9)).stroke_width(3),
    ))?;

    let upper_val = median + mad;
    let upper_line: Vec<(f64, f32)> = vec![(min_x, upper_val), (max_x, upper_val)];
    chart.draw_series(LineSeries::new(
        upper_line,
        ShapeStyle::from(&RED.mix(0.6)).stroke_width(2),
    ))?;

    let lower_val = median - mad;
    let lower_line: Vec<(f64, f32)> = vec![(min_x, lower_val), (max_x, lower_val)];
    chart.draw_series(LineSeries::new(
        lower_line,
        ShapeStyle::from(&RED.mix(0.6)).stroke_width(2),
    ))?;

    // Draw legend as text
    let style = TextStyle::from(("sans-serif", 20)).color(&BLACK);
    let legend_lines = vec![
        format!("Antennas: {}", antenna_label),
        format!(
            "Peak Freq: {} MHz",
            format_with_precision(peak_freq, freq_precision)
        ),
        format!(
            "Peak Velocity: {} km/s",
            format_with_precision(peak_velocity, vel_precision)
        ),
        format!("Peak Value: {:.5}", peak_val),
        format!("Peak - Median: {:.5}", peak_val - median),
        format!("SNR (MAD): {:.3}", snr_mad),
        format!("SNR (stdev): {:.3}", snr_stdev),
        format!("Median: {:.5}", median),
        format!("MAD: {:.5}", mad),
        format!("stdev = MAD/0.6745 = {:.5}", sigma_est),
        format!(
            "Δf: {} MHz",
            format_with_precision(freq_resolution_mhz, freq_precision)
        ),
        format!(
            "Δv: {} km/s",
            format_with_precision(channel_width_kms, vel_precision)
        ),
    ];
    let legend_x = 860;
    let mut y_pos = 40;
    for line in legend_lines {
        root.draw(&Text::new(line, (legend_x, y_pos), style.clone()))?;
        y_pos += 25;
    }

    if let Some(params) = gaussian_params {
        root.draw(&Text::new(
            "Gaussian Components:",
            (legend_x, y_pos),
            style.clone(),
        ))?;
        y_pos += 25;
        for (idx, (amp, center, fwhm)) in params.iter().enumerate() {
            let text = format!(
                "G{}: amp={:.4}, v={} km/s, FWHM={} km/s",
                idx + 1,
                amp,
                format_with_precision(*center, vel_precision),
                format_with_precision(*fwhm, vel_precision)
            );
            root.draw(&Text::new(text, (legend_x, y_pos), style.clone()))?;
            y_pos += 25;
        }
    }

    root.present()?;
    compress_png_with_mode(output_path, CompressQuality::Low);
    Ok(())
}

fn plot_on_off_spectra(
    output_path: &Path,
    on_data: &[(f64, f32)],
    off_data: &[(f64, f32)],
    x_label: &str,
    peak_frequency: f64,
    peak_velocity: f64,
    antenna_label: &str,
    off_label: &str,
    freq_precision: usize,
    vel_precision: usize,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let (min_x, max_x) = on_data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (x, _)| {
            (min.min(*x), max.max(*x))
        });

    let on_max = on_data
        .iter()
        .fold(f32::NEG_INFINITY, |max, (_, v)| max.max(*v));
    let off_max = off_data
        .iter()
        .fold(f32::NEG_INFINITY, |max, (_, v)| max.max(*v));
    let max_y = on_max.max(off_max);

    let on_min = on_data
        .iter()
        .fold(f32::INFINITY, |min, (_, v)| min.min(*v));
    let off_min = off_data
        .iter()
        .fold(f32::INFINITY, |min, (_, v)| min.min(*v));
    let min_y = on_min.min(off_min);

    let y_margin = (max_y - min_y) * 0.1;
    let caption = format!("ON-source vs {} Spectrum", off_label);

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(min_x..max_x, (min_y - y_margin)..(max_y + y_margin))?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc("Amplitude")
        .label_style(("sans-serif", 20))
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .light_line_style(&TRANSPARENT)
        .draw()?;

    chart
        .draw_series(LineSeries::new(on_data.iter().cloned(), &BLUE))?
        .label("ON Source")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));

    chart
        .draw_series(LineSeries::new(off_data.iter().cloned(), &RED))?
        .label(off_label)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    // Draw legend as text
    let style = TextStyle::from(("sans-serif", 20)).color(&BLACK);
    let legend_lines = vec![
        format!("Antennas: {}", antenna_label),
        format!("ON: blue / {}: red", off_label),
        format!(
            "Peak Freq: {} MHz",
            format_with_precision(peak_frequency, freq_precision)
        ),
        format!(
            "Peak Velocity: {} km/s",
            format_with_precision(peak_velocity, vel_precision)
        ),
    ];
    let legend_x = 860;
    let mut y_pos = 40;
    for line in legend_lines {
        root.draw(&Text::new(line, (legend_x, y_pos), style.clone()))?;
        y_pos += 25;
    }

    root.present()?;
    compress_png_with_mode(output_path, CompressQuality::Low);
    Ok(())
}

pub fn run_maser_analysis(args: &Args) -> Result<(), Box<dyn Error>> {
    // 1. Parse args
    let mut log_lines: Vec<String> = Vec::new();
    maser_logln!(log_lines, "Running Maser Analysis...");
    let on_source_path = args.input.as_ref().unwrap();

    let mut off_spec: Option<OffSpec> = None;
    let mut rest_freq_mhz: f64 = 6668.5192;
    let mut rest_freq_overridden = false;
    let mut override_vlsr: Option<f64> = None;
    let mut corrfreq: f64 = 1.0;
    let mut user_band_range: Option<(f64, f64)> = None;
    let mut user_subt_range: Option<(f64, f64)> = None;
    let mut gaussian_initial_components: Vec<(f64, f64, f64)> = Vec::new();
    let mut positional_args: Vec<&String> = Vec::new();
    let mut onoff_mode: Option<u8> = None;
    let mut maser_mode = MaserMode::default();
    let mut integration_state: Option<IntegrationState> = None;

    let mut idx = 0;
    while idx < args.maser.len() {
        let entry = &args.maser[idx];
        if let Some((key, raw_value)) = entry.split_once(':') {
            let mut value_owned = raw_value.trim().to_string();
            if value_owned.is_empty() && idx + 1 < args.maser.len() {
                idx += 1;
                value_owned = args.maser[idx].trim().to_string();
            }
            if value_owned.is_empty() {
                return Err(format!(
                    "Error: parameter '{}' requires a value (e.g., {}:<val>).",
                    key.trim(),
                    key.trim()
                )
                .into());
            }

            match key.trim().to_lowercase().as_str() {
                "off" => {
                    let value = value_owned.trim();
                    if let Some(model) = BaselineFitKind::from_str(value) {
                        off_spec = Some(OffSpec::Baseline(model));
                    } else {
                        off_spec = Some(OffSpec::File(PathBuf::from(value)));
                    }
                }
                "rest" => {
                    rest_freq_mhz = value_owned.trim().parse()?;
                    rest_freq_overridden = true;
                }
                "vlst" => {
                    override_vlsr = Some(value_owned.trim().parse()?);
                }
                "corrfreq" => {
                    corrfreq = value_owned.trim().parse()?;
                }
                "band" => {
                    let mut parts = value_owned.trim().split('-');
                    let start: f64 = parts
                        .next()
                        .ok_or("Error: band requires start-end in MHz offset.")?
                        .parse()?;
                    let end: f64 = parts
                        .next()
                        .ok_or("Error: band requires start-end in MHz offset.")?
                        .parse()?;
                    if parts.next().is_some() {
                        return Err(
                            "Error: band accepts exactly one start-end pair (e.g., band:60-70)."
                                .into(),
                        );
                    }
                    if start >= end {
                        return Err("Error: band start must be less than band end.".into());
                    }
                    user_band_range = Some((start, end));
                }
                "subt" => {
                    let mut parts = value_owned.trim().split('-');
                    let start: f64 = parts
                        .next()
                        .ok_or("Error: subt requires start-end in MHz.")?
                        .parse()?;
                    let end: f64 = parts
                        .next()
                        .ok_or("Error: subt requires start-end in MHz.")?
                        .parse()?;
                    if parts.next().is_some() {
                        return Err(
                            "Error: subt accepts exactly one start-end pair (e.g., subt:6664-6672)."
                                .into(),
                        );
                    }
                    if start >= end {
                        return Err("Error: subt start must be less than subt end.".into());
                    }
                    user_subt_range = Some((start, end));
                }
                "onoff" => {
                    let mode: u8 = value_owned.trim().parse()?;
                    if mode > 1 {
                        return Err(
                            "Error: onoff accepts only 0 ((ON-OFF)/OFF) or 1 (ON-OFF).".into()
                        );
                    }
                    onoff_mode = Some(mode);
                }
                "mode" => {
                    maser_mode = MaserMode::from_str(value_owned.trim())
                        .map_err(|e| format!("Error parsing --maser mode: {}", e))?;
                }
                "gauss" => {
                    let params: Vec<&str> = value_owned
                        .split(',')
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .collect();
                    if params.is_empty() {
                        return Err("Error: gauss requires amp,Vlst,fwhm values.".into());
                    }
                    if params.len() % 3 != 0 {
                        return Err(
                            format!(
                                "Error: gauss expects triples of amp,Vlst,fwhm. Received {} entries: {}",
                                params.len(),
                                params.join(", ")
                            )
                            .into(),
                        );
                    }
                    for chunk in params.chunks(3) {
                        let amp: f64 = chunk[0].parse()?;
                        let center: f64 = chunk[1].parse()?;
                        let fwhm: f64 = chunk[2].parse()?;
                        if fwhm <= 0.0 {
                            return Err("Error: gauss FWHM must be positive.".into());
                        }
                        gaussian_initial_components.push((amp, center, fwhm));
                    }
                }
                other => {
                    return Err(format!(
                        "Unknown --maser parameter '{}'. Expected off, rest, Vlst, corrfreq, band, subt, onoff, mode, gauss.",
                        other
                    )
                    .into());
                }
            }
        } else {
            positional_args.push(entry);
        }
        idx += 1;
    }

    if off_spec.is_none() {
        if let Some(first) = positional_args.get(0) {
            let value = first.trim();
            if let Some(model) = BaselineFitKind::from_str(value) {
                off_spec = Some(OffSpec::Baseline(model));
            } else {
                off_spec = Some(OffSpec::File(PathBuf::from(value)));
            }
        }
    }

    if off_spec.is_none() {
        return Err("Error: --maser requires off:<PATH> or off:linear/off:quad.".into());
    }

    if positional_args.len() >= 2 {
        if let Ok(rest_val) = positional_args[1].parse::<f64>() {
            rest_freq_mhz = rest_val;
            rest_freq_overridden = true;
        }
    }

    let off_spec = off_spec.unwrap();

    maser_logln!(log_lines, "  ON Source: {:?}", on_source_path);
    match &off_spec {
        OffSpec::File(path) => maser_logln!(log_lines, "  OFF Source: {:?}", path),
        OffSpec::Baseline(kind) => maser_logln!(
            log_lines,
            "  OFF Source: baseline model ({}) from ON spectrum",
            kind.as_str()
        ),
    }
    maser_logln!(log_lines, "  Frequency Correction Factor: {:.6}", corrfreq);
    maser_logln!(log_lines, "  Maser mode: {}", maser_mode.as_str());
    if let Some(v) = override_vlsr {
        maser_logln!(
            log_lines,
            "  Override LSR Velocity Correction: {:.6} km/s",
            v
        );
    }
    if !gaussian_initial_components.is_empty() {
        maser_logln!(
            log_lines,
            "  Gaussian initial guesses (amp, center[km/s], fwhm[km/s]):"
        );
        for (amp, center, fwhm) in &gaussian_initial_components {
            maser_logln!(log_lines, "    {:.4}, {:.4}, {:.4}", amp, center, fwhm);
        }
    }

    let chunk_length = if args.length > 0 {
        let off_note = match &off_spec {
            OffSpec::File(_) => "OFF source uses full span".to_string(),
            OffSpec::Baseline(kind) => format!("OFF is {} baseline fit in analysis window", kind.as_str()),
        };
        maser_logln!(
            log_lines,
            "  Maser processing will segment the ON source with --length={} and --loop={} ({}).",
            args.length, args.loop_, off_note
        );
        args.length
    } else {
        maser_logln!(
            log_lines,
            "  Maser processing will use the full data span (single segment)."
        );
        0
    };
    let mut on_segments: Vec<SpectrumData> = Vec::new();
    let mut loop_index: usize = 0;
    let loop_limit: usize = if chunk_length > 0 {
        args.loop_.max(1) as usize
    } else {
        1
    };

    let (off_full_segment, baseline_model) = match &off_spec {
        OffSpec::File(off_path) => {
            let off_seg = get_spectrum_segment(off_path, args, corrfreq, 0, 0, &mut log_lines)?
                .ok_or_else(|| {
                    format!(
                        "Off-source file {:?} contains no usable data for maser analysis.",
                        off_path
                    )
                })?;
            (Some(off_seg), None)
        }
        OffSpec::Baseline(kind) => (None, Some(*kind)),
    };

    loop {
        if chunk_length > 0 && loop_index >= loop_limit {
            break;
        }

        let on_chunk = get_spectrum_segment(
            on_source_path,
            args,
            corrfreq,
            chunk_length,
            loop_index as i32,
            &mut log_lines,
        )?;

        match on_chunk {
            Some(on_data) => {
                on_segments.push(on_data);
            }
            None => {
                if chunk_length > 0 {
                    maser_logln!(
                        log_lines,
                        "  [maser] loop {}: ON 側で追加のセグメントが読み出せなかったため終了します。",
                        loop_index
                    );
                }
                break;
            }
        }

        if chunk_length == 0 {
            break;
        }
        loop_index += 1;
    }

    if on_segments.is_empty() {
        return Err("No data available for maser analysis after applying --length/--loop.".into());
    }

    let header_on = &on_segments[0].header;

    let freq_precision = fft_precision_digits(header_on.fft_point);
    let vel_precision = freq_precision;

    if let Some(header_off) = off_full_segment.as_ref().map(|s| &s.header) {
        if header_on.fft_point != header_off.fft_point
            || header_on.observing_frequency.to_bits() != header_off.observing_frequency.to_bits()
            || header_on.sampling_speed != header_off.sampling_speed
        {
            let off_path = match &off_spec {
                OffSpec::File(path) => format!("{:?}", path),
                OffSpec::Baseline(kind) => format!("baseline({})", kind.as_str()),
            };
            return Err(format!(
                "Error: Header mismatch between {:?} and {}.
  fft_point: {} (ON) vs {} (OFF)
  observing_frequency: {:.6} Hz (ON) vs {:.6} Hz (OFF)
  sampling_speed: {} Hz (ON) vs {} Hz (OFF)",
                on_source_path,
                off_path,
                header_on.fft_point,
                header_off.fft_point,
                header_on.observing_frequency,
                header_off.observing_frequency,
                header_on.sampling_speed,
                header_off.sampling_speed
            )
            .into());
        }
    }

    for (idx, on_seg) in on_segments.iter().enumerate() {
        if on_seg.header.fft_point != header_on.fft_point
            || on_seg.header.observing_frequency.to_bits()
                != header_on.observing_frequency.to_bits()
            || on_seg.header.sampling_speed != header_on.sampling_speed
        {
            return Err(format!("ON segment header mismatch at loop {}.", idx).into());
        }
    }

    if !rest_freq_overridden {
        let obs_freq_mhz = header_on.observing_frequency / 1e6;
        if (11923.0..=12435.0).contains(&obs_freq_mhz) {
            rest_freq_mhz = 1217.8597;
            maser_logln!(
                log_lines,
                "  Rest frequency automatically set to 1217.8597 MHz for 12.2 GHz methanol maser analysis."
            );
        }
    }

    let parent_dir = on_source_path.parent().unwrap_or_else(|| Path::new(""));
    let output_dir = parent_dir.join("frinZ").join("maser");
    fs::create_dir_all(&output_dir)?;
    let on_stem = on_source_path.file_stem().unwrap().to_str().unwrap();

    let freq_resolution_mhz = (header_on.sampling_speed as f64 * corrfreq / 2.0 / 1e6)
        / (header_on.fft_point as f64 / 2.0);
    let channel_width_kms = C_KM_S * freq_resolution_mhz / rest_freq_mhz;
    let observing_freq_mhz = header_on.observing_frequency / 1e6;
    let station1_label = header_on.station1_name.trim().to_string();
    let station2_label = header_on.station2_name.trim().to_string();
    let antenna_label = if station1_label.eq_ignore_ascii_case(&station2_label) {
        station1_label.clone()
    } else {
        format!("{}/{}", station1_label, station2_label)
    };
    let (onoff_mode, normalization_note): (u8, Option<String>) = match onoff_mode {
        Some(val) => (val, Some("user-specified".to_string())),
        None => {
            let auto_val = if station1_label.eq_ignore_ascii_case(&station2_label) {
                0
            } else {
                1
            };
            let reason = if auto_val == 0 {
                "auto-selected (same station; (ON-OFF)/OFF)"
            } else {
                "auto-selected (baseline combination)"
            };
            (auto_val, Some(reason.to_string()))
        }
    };
    let norm_label = if onoff_mode == 0 {
        "(ON-OFF)/OFF"
    } else {
        "(ON-OFF)"
    };
    if let Some(note) = normalization_note {
        maser_logln!(log_lines, "  Normalization mode: {} [{}]", norm_label, note);
    } else {
        maser_logln!(log_lines, "  Normalization mode: {}", norm_label);
    }
    if maser_mode.accumulate_integration() && integration_state.is_none() {
        integration_state = Some(IntegrationState::new());
    }
    let write_segment_outputs = maser_mode.write_segment_outputs();
    let print_segment_table = maser_mode.print_segment_table();
    let base_freq_mhz = header_on.observing_frequency / 1e6;
    let freq_range_mhz: Vec<f64> = (0..header_on.fft_point as usize / 2)
        .map(|i| i as f64 * freq_resolution_mhz + base_freq_mhz)
        .collect();

    let analysis_indices: Vec<usize> = if let Some((start_abs, end_abs)) = user_subt_range {
        freq_range_mhz
            .iter()
            .enumerate()
            .filter(|(_, &freq)| freq >= start_abs && freq <= end_abs)
            .map(|(i, _)| i)
            .collect()
    } else if let Some((start_offset, end_offset)) = user_band_range {
        let min_freq = base_freq_mhz + start_offset;
        let max_freq = base_freq_mhz + end_offset;
        freq_range_mhz
            .iter()
            .enumerate()
            .filter(|(_, &freq)| freq >= min_freq && freq <= max_freq)
            .map(|(i, _)| i)
            .collect()
    } else if rest_freq_mhz >= 6600.0 && rest_freq_mhz <= 7112.0 {
        maser_logln!(
            log_lines,
            "  C-band maser detected. Restricting analysis to 6664-6672 MHz range."
        );
        freq_range_mhz
            .iter()
            .enumerate()
            .filter(|(_, &freq)| freq >= 6664.0 && freq <= 6672.0)
            .map(|(i, _)| i)
            .collect()
    } else {
        (0..freq_range_mhz.len()).collect()
    };

    if analysis_indices.is_empty() {
        return Err("No data found in the specified frequency range for analysis.".into());
    }
    let ch_min = *analysis_indices
        .iter()
        .min()
        .ok_or("No channel indices in analysis range.")?;
    let ch_max = *analysis_indices
        .iter()
        .max()
        .ok_or("No channel indices in analysis range.")?;
    let f_min_abs = freq_range_mhz[ch_min];
    let f_max_abs = freq_range_mhz[ch_max];
    maser_logln!(
        log_lines,
        "  Observing Frequency (from .cor header): {:.6} MHz",
        observing_freq_mhz
    );
    maser_logln!(log_lines, "  Rest Frequency: {:.6} MHz", rest_freq_mhz);
    if let Some((start, end)) = user_band_range {
        maser_logln!(
            log_lines,
            "  Requested window (offset): {:.3} to {:.3} MHz",
            start,
            end
        );
    }
    if let Some((start, end)) = user_subt_range {
        maser_logln!(
            log_lines,
            "  Requested window (absolute): {:.3} to {:.3} MHz",
            start,
            end
        );
    }
    maser_logln!(
        log_lines,
        "  Analysis window (absolute): {:.6} - {:.6} MHz (channels {}..{}, N={})",
        f_min_abs,
        f_max_abs,
        ch_min,
        ch_max,
        analysis_indices.len()
    );
    maser_logln!(
        log_lines,
        "  Frequency Resolution: {} MHz",
        format_with_precision(freq_resolution_mhz, freq_precision)
    );
    maser_logln!(
        log_lines,
        "  Channel Width: {} km/s",
        format_with_precision(channel_width_kms, vel_precision)
    );
    maser_logln!(
        log_lines,
        "  Velocity model: Vlsr(f_obs) = c * (f_rest - f_obs) / f_rest + Vlsr_corr"
    );
    maser_logln!(
        log_lines,
        "    c = {:.6} km/s, f_rest = {:.6} MHz, f_obs = {:.6} MHz + Frequency_Offset_MHz",
        C_KM_S,
        rest_freq_mhz,
        base_freq_mhz
    );

    let (spec_title_base, spec_y_label_base) = (
        if onoff_mode == 0 {
            "Maser Analysis: (ON-OFF)/OFF Spectrum".to_string()
        } else {
            "Maser Analysis: (ON-OFF) Spectrum".to_string()
        },
        "Amplitude".to_string(),
    );

    let total_segments = on_segments.len();
    let mut summaries: Vec<SegmentSummary> = Vec::with_capacity(total_segments);

    for (seg_idx, on_seg) in on_segments.iter().enumerate() {
        let summary = analyze_segment(
            seg_idx,
            total_segments,
            on_seg,
            off_full_segment.as_ref(),
            baseline_model,
            &freq_range_mhz,
            &analysis_indices,
            rest_freq_mhz,
            override_vlsr,
            onoff_mode,
            &gaussian_initial_components,
            &spec_title_base,
            &spec_y_label_base,
            &antenna_label,
            base_freq_mhz,
            freq_resolution_mhz,
            channel_width_kms,
            &output_dir,
            on_stem,
            integration_state.as_mut(),
            write_segment_outputs,
            freq_precision,
            vel_precision,
            &mut log_lines,
        )?;
        summaries.push(summary);
    }

    if print_segment_table {
        maser_logln!(log_lines, "  Maser segments (N={}):", total_segments);
        maser_logln!(
            log_lines,
            "    idx start_time_utc         seg_s lsr_km/s peak_freq_MHz peak_vlsr_km/s peak_val median   mad      peak_diff  snr_mad snr_std stdev"
        );
        for summary in &summaries {
            let peak_freq_str = format_with_precision(summary.peak_freq_mhz, freq_precision);
            let peak_vel_str = format_with_precision(summary.peak_velocity_kms, vel_precision);
            maser_logln!(
                log_lines,
                "    {:>3} {} {:>6.1} {:>9.3} {:>12} {:>13} {:>9.6} {:>8.6} {:>9.6} {:>11.6} {:>8.3} {:>8.3} {:>8.6}",
                summary.index,
                summary.start_time.format("%Y-%m-%d %H:%M:%S"),
                summary.segment_seconds,
                summary.lsr_average,
                peak_freq_str,
                peak_vel_str,
                summary.peak_value,
                summary.median,
                summary.mad,
                summary.peak_minus_median,
                summary.snr_mad,
                summary.snr_stdev,
                summary.sigma_est,
            );
        }
    }

    let mut produced_outputs = write_segment_outputs;
    let integration_result = integration_state.and_then(|state| state.finalize());
    if let Some(integration) = integration_result {
        if integration.frequency_axis_mhz.is_empty() {
            maser_logln!(log_lines, "Stacked spectrum: no data accumulated.");
        } else {
            produced_outputs = true;
            let velocity_axis = integration.to_velocity_axis();
            let (peak_idx, peak_val) = integration
                .peak()
                .unwrap_or((0, integration.averaged_spec.first().copied().unwrap_or(0.0)));
            let mut sorted_spec = integration.averaged_spec.clone();
            sorted_spec.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = sorted_spec[sorted_spec.len() / 2];
            let mut deviations: Vec<f32> = integration
                .averaged_spec
                .iter()
                .map(|&val| (val - median).abs())
                .collect();
            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mad = deviations[deviations.len() / 2];
            let peak_minus_median = peak_val - median;
            let snr_mad = if mad > 1e-9 {
                peak_minus_median / mad
            } else {
                0.0
            };
            let sigma_est = if mad > 1e-9 { mad / 0.6745_f32 } else { 0.0 };
            let snr_stdev = if sigma_est > 1e-9 {
                peak_minus_median / sigma_est
            } else {
                0.0
            };
            let peak_freq_mhz = integration.frequency_axis_mhz[peak_idx];
            let peak_velocity_kms = velocity_axis[peak_idx];

            maser_logln!(log_lines, "  Stacked spectrum:");
            maser_logln!(
                log_lines,
                "    segs integ_s mean_lsr_km/s peak_freq_MHz peak_vlsr_km/s peak_val median   mad      peak_diff  snr_mad snr_std stdev"
            );
            let stacked_peak_freq = format_with_precision(peak_freq_mhz, freq_precision);
            let stacked_peak_vel = format_with_precision(peak_velocity_kms, vel_precision);
            maser_logln!(
                log_lines,
                "    {:>3} {:>8.1} {:>13.3} {:>12} {:>13} {:>9.6} {:>8.6} {:>9.6} {:>11.6} {:>8.3} {:>8.3} {:>8.6}",
                integration.segment_count,
                integration.total_segment_seconds,
                integration.mean_lsr_corr,
                stacked_peak_freq,
                stacked_peak_vel,
                peak_val,
                median,
                mad,
                peak_minus_median,
                snr_mad,
                snr_stdev,
                sigma_est,
            );

            let stacked_suffix = "_stacked";
            let integ_tsv =
                output_dir.join(format!("{}{}_maser_data.tsv", on_stem, stacked_suffix));
            let mut integ_file = File::create(&integ_tsv)?;
            writeln!(
                integ_file,
                "# Frequency_Offset_MHz\tVelocity_km/s\tNormalized_Intensity"
            )?;
            let freq_reference = integration
                .frequency_axis_mhz
                .first()
                .copied()
                .unwrap_or(0.0);
            for ((&freq, &vel), &power) in integration
                .frequency_axis_mhz
                .iter()
                .zip(velocity_axis.iter())
                .zip(integration.averaged_spec.iter())
            {
                let freq_offset = freq - freq_reference;
                let freq_offset_str = format_f32_precision(freq_offset, freq_precision);
                let vel_str = format_f32_precision(vel, vel_precision);
                let power_str = format!("{:.6}", power);
                writeln!(
                    integ_file,
                    "{}\t{}\t{}",
                    freq_offset_str, vel_str, power_str
                )?;
            }

            let normalized_plot_data_freq: Vec<(f64, f32)> = integration
                .frequency_axis_mhz
                .iter()
                .zip(integration.averaged_spec.iter())
                .map(|(&freq, &val)| (freq, val))
                .collect();
            let normalized_plot_data_vel: Vec<(f64, f32)> = velocity_axis
                .iter()
                .zip(integration.averaged_spec.iter())
                .map(|(&vel, &val)| (vel, val))
                .collect();

            let stacked_title = format!("Stacked {}", spec_title_base);
            let stacked_freq_plot =
                output_dir.join(format!("{}{}_maser_subt.png", on_stem, stacked_suffix));
            plot_maser_spectrum(
                &stacked_freq_plot,
                &normalized_plot_data_freq,
                "Frequency [MHz]",
                &stacked_title,
                &spec_y_label_base,
                &antenna_label,
                peak_freq_mhz,
                peak_velocity_kms,
                peak_val,
                snr_mad,
                snr_stdev,
                median,
                mad,
                sigma_est,
                integration.freq_resolution_mhz,
                integration.channel_width_kms,
                freq_precision,
                vel_precision,
                None,
                None,
            )?;

            let stacked_vel_plot =
                output_dir.join(format!("{}{}_maser_Vlsr1.png", on_stem, stacked_suffix));
            plot_maser_spectrum(
                &stacked_vel_plot,
                &normalized_plot_data_vel,
                "LSR Velocity [km/s]",
                &stacked_title,
                &spec_y_label_base,
                &antenna_label,
                peak_freq_mhz,
                peak_velocity_kms,
                peak_val,
                snr_mad,
                snr_stdev,
                median,
                mad,
                sigma_est,
                integration.freq_resolution_mhz,
                integration.channel_width_kms,
                freq_precision,
                vel_precision,
                None,
                None,
            )?;

            let vel_window = 10.0;
            let zoomed_plot_data: Vec<(f64, f32)> = normalized_plot_data_vel
                .iter()
                .cloned()
                .filter(|(vel, _)| {
                    *vel >= peak_velocity_kms - vel_window && *vel <= peak_velocity_kms + vel_window
                })
                .collect();
            if !zoomed_plot_data.is_empty() {
                let stacked_zoom_plot =
                    output_dir.join(format!("{}{}_maser_Vlsr2.png", on_stem, stacked_suffix));
                plot_maser_spectrum(
                    &stacked_zoom_plot,
                    &zoomed_plot_data,
                    "LSR Velocity [km/s]",
                    &stacked_title,
                    &spec_y_label_base,
                    &antenna_label,
                    peak_freq_mhz,
                    peak_velocity_kms,
                    peak_val,
                    snr_mad,
                    snr_stdev,
                    median,
                    mad,
                    sigma_est,
                    integration.freq_resolution_mhz,
                    integration.channel_width_kms,
                    freq_precision,
                    vel_precision,
                    None,
                    None,
                )?;
            }
        }
    }

    if produced_outputs {
        maser_logln!(log_lines, "make some plots in {:?}", output_dir);
    }

    let log_path = output_dir.join(format!("{}_maser_stdout.txt", on_stem));
    maser_logln!(log_lines, "  Maser stdout saved to {:?}", log_path);
    write_maser_log(&log_path, &log_lines)?;

    Ok(())
}

fn analyze_segment(
    seg_idx: usize,
    total_segments: usize,
    on_seg: &SpectrumData,
    off_seg: Option<&SpectrumData>,
    baseline_model: Option<BaselineFitKind>,
    freq_range_mhz: &[f64],
    analysis_indices: &[usize],
    rest_freq_mhz: f64,
    override_vlsr: Option<f64>,
    onoff_mode: u8,
    gaussian_initial_components: &[(f64, f64, f64)],
    spec_title: &str,
    spec_y_label: &str,
    antenna_label: &str,
    base_freq_mhz: f64,
    freq_resolution_mhz: f64,
    channel_width_kms: f64,
    output_dir: &Path,
    on_stem: &str,
    integration_state: Option<&mut IntegrationState>,
    write_outputs: bool,
    freq_precision: usize,
    vel_precision: usize,
    log_lines: &mut Vec<String>,
) -> Result<SegmentSummary, Box<dyn Error>> {
    let mut lsr_vel_corr = compute_lsr_average(
        &on_seg.header,
        on_seg.start_time,
        on_seg.effective_integration_time,
        on_seg.sector_count,
    );
    if let Some(v) = override_vlsr {
        lsr_vel_corr = v;
    }

    let segment_seconds = on_seg.effective_integration_time * on_seg.sector_count as f32;

    let analysis_freq_mhz: Vec<f64> = analysis_indices
        .iter()
        .map(|&i| freq_range_mhz[i])
        .collect();

    let analysis_velocity_kms: Vec<f64> = analysis_freq_mhz
        .iter()
        .map(|&f_obs| C_KM_S * (rest_freq_mhz - f_obs) / rest_freq_mhz + lsr_vel_corr)
        .collect();

    let spec_on_slice = on_seg
        .spectrum
        .as_slice()
        .ok_or("ON spectrum data not contiguous")?;

    let analysis_spec_on: Vec<f32> = analysis_indices.iter().map(|&i| spec_on_slice[i]).collect();
    let (analysis_spec_off, off_label_for_plot, baseline_mask) = if let Some(off_data) = off_seg {
        let spec_off_slice = off_data
            .spectrum
            .as_slice()
            .ok_or("OFF spectrum data not contiguous")?;
        (
            analysis_indices
                .iter()
                .map(|&i| spec_off_slice[i])
                .collect::<Vec<f32>>(),
            "OFF Source".to_string(),
            vec![true; analysis_indices.len()],
        )
    } else if let Some(model) = baseline_model {
        let fit = fit_baseline_robust(&analysis_freq_mhz, &analysis_spec_on, model)?;
        maser_logln!(
            log_lines,
            "  [seg {:03}] OFF baseline {} fit: used {}/{} points (MAD sigma={:.6})",
            seg_idx,
            model.as_str(),
            fit.used_count,
            analysis_spec_on.len(),
            fit.sigma_est
        );
        let coeff_text = if fit.coeff_shifted.len() >= 3 {
            format!(
                "{:.6e} + {:.6e}*(f-{:.6}) + {:.6e}*(f-{:.6})^2",
                fit.coeff_shifted[0],
                fit.coeff_shifted[1],
                fit.x_center_mhz,
                fit.coeff_shifted[2],
                fit.x_center_mhz
            )
        } else if fit.coeff_shifted.len() >= 2 {
            format!(
                "{:.6e} + {:.6e}*(f-{:.6})",
                fit.coeff_shifted[0],
                fit.coeff_shifted[1],
                fit.x_center_mhz
            )
        } else {
            format!("{:.6e}", fit.coeff_shifted[0])
        };
        maser_logln!(
            log_lines,
            "  [seg {:03}] OFF baseline model: off(f_MHz) = {}",
            seg_idx,
            coeff_text
        );
        (
            fit.baseline,
            format!("Baseline ({})", model.as_str()),
            fit.used_mask,
        )
    } else {
        return Err("OFF data is not available. Use off:<path> or off:linear/off:quad.".into());
    };

    let mut normalized_spec = Array1::<f32>::zeros(analysis_indices.len());
    for i in 0..analysis_indices.len() {
        let diff = analysis_spec_on[i] - analysis_spec_off[i];
        normalized_spec[i] = if onoff_mode == 0 {
            if analysis_spec_off[i] > 1e-9 {
                diff / analysis_spec_off[i]
            } else {
                0.0
            }
        } else {
            diff
        };
    }

    let normalized_spec_f64: Vec<f64> = normalized_spec.iter().map(|&v| v as f64).collect();
    let mut gaussian_fit_summary: Option<GaussianFitResult> = None;
    let mut gaussian_fit_components: Option<Vec<(f64, f64, f64)>> = None;
    let mut gaussian_fit_data_vel: Option<Vec<(f64, f32)>> = None;

    if !gaussian_initial_components.is_empty() {
        match fit_gaussian_mixture(
            &analysis_velocity_kms,
            &normalized_spec_f64,
            gaussian_initial_components,
        ) {
            Ok(result) => {
                maser_logln!(
                    log_lines,
                    "  [seg {:03}] Gaussian fit termination: {:?}, residual norm: {:.6}, evaluations: {}",
                    seg_idx,
                    result.termination,
                    result.residual_norm,
                    result.evaluations
                );
                gaussian_fit_data_vel = Some(evaluate_gaussian_mixture(
                    &analysis_velocity_kms,
                    &result.components,
                ));
                gaussian_fit_components = Some(result.components.clone());
                gaussian_fit_summary = Some(result);
            }
            Err(err) => {
                maser_logln!(
                    log_lines,
                    "Warning: [seg {:03}] Gaussian fit failed ({}). Using initial parameters for overlay.",
                    seg_idx,
                    err
                );
                eprintln!(
                    "Warning: [seg {:03}] Gaussian fit failed ({}). Using initial parameters for overlay.",
                    seg_idx, err
                );
                gaussian_fit_data_vel = Some(evaluate_gaussian_mixture(
                    &analysis_velocity_kms,
                    gaussian_initial_components,
                ));
                gaussian_fit_components = Some(gaussian_initial_components.to_vec());
            }
        }
    }

    if let Some(summary) = &gaussian_fit_summary {
        maser_logln!(
            log_lines,
            "  [seg {:03}] Gaussian fit residual norm: {:.6} (objective: {:.6})",
            seg_idx,
            summary.residual_norm,
            0.5 * summary.residual_norm * summary.residual_norm
        );
    }

    if let Some(accumulator) = integration_state {
        let normalized_slice = normalized_spec
            .as_slice()
            .ok_or("Normalized spectrum data not contiguous")?;
        let topo_velocity_kms: Vec<f64> = analysis_velocity_kms
            .iter()
            .map(|&v| v - lsr_vel_corr)
            .collect();
        accumulator.add_segment(
            lsr_vel_corr,
            segment_seconds,
            &analysis_freq_mhz,
            &topo_velocity_kms,
            normalized_slice,
            freq_resolution_mhz,
            channel_width_kms,
        );
    }

    let mut peak_val = f32::NEG_INFINITY;
    let mut peak_idx_in_analysis = 0;
    for (i, &val) in normalized_spec.iter().enumerate() {
        if val > peak_val {
            peak_val = val;
            peak_idx_in_analysis = i;
        }
    }
    let peak_freq_mhz = analysis_freq_mhz
        .get(peak_idx_in_analysis)
        .copied()
        .unwrap_or(base_freq_mhz);
    let peak_velocity_kms = analysis_velocity_kms[peak_idx_in_analysis];
    let mut sorted_spec = normalized_spec.to_vec();
    sorted_spec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_spec[sorted_spec.len() / 2];
    let mut deviations: Vec<f32> = normalized_spec
        .iter()
        .map(|&val| (val - median).abs())
        .collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = deviations[deviations.len() / 2];
    let peak_minus_median = peak_val - median;
    let snr_mad = if mad > 1e-9 {
        peak_minus_median / mad
    } else {
        0.0
    };
    let sigma_est = if mad > 1e-9 { mad / 0.6745_f32 } else { 0.0 };
    let snr_stdev = if sigma_est > 1e-9 {
        peak_minus_median / sigma_est
    } else {
        0.0
    };

    let suffix = if total_segments > 1 {
        format!("_seg{:03}", seg_idx)
    } else {
        String::new()
    };

    if write_outputs {
        let tsv_filename = output_dir.join(format!("{}{}_maser_data.tsv", on_stem, suffix));
        let mut file = File::create(&tsv_filename)?;
        writeln!(
            file,
            "# Frequency_Offset_MHz\tVelocity_km/s\tonsourc\toffsource\tbaseline_mask"
        )?;
        for (idx, &freq_mhz) in analysis_freq_mhz.iter().enumerate() {
            let freq_offset_mhz = freq_mhz - base_freq_mhz;
            let freq_offset_str = format_f32_precision(freq_offset_mhz, freq_precision);
            let vel_str = format_f32_precision(analysis_velocity_kms[idx], vel_precision);
            let on_str = format!("{:.6}", analysis_spec_on[idx]);
            let off_str = format!("{:.6}", analysis_spec_off[idx]);
            let mask_flag = if baseline_mask[idx] { 1 } else { 0 };
            writeln!(
                file,
                "{}\t{}\t{}\t{}\t{}",
                freq_offset_str, vel_str, on_str, off_str, mask_flag
            )?;
        }

        if let Some(ref comps) = gaussian_fit_components {
            let fit_filename = output_dir.join(format!("{}{}_maser_fit.txt", on_stem, suffix));
            let mut fit_file = File::create(&fit_filename)?;
            writeln!(fit_file, "Gaussian Fit Summary")?;
            if let Some(summary) = &gaussian_fit_summary {
                writeln!(fit_file, "Termination: {:?}", summary.termination)?;
                writeln!(fit_file, "Evaluations: {}", summary.evaluations)?;
                writeln!(fit_file, "Residual norm: {:.6}", summary.residual_norm)?;
                writeln!(
                    fit_file,
                    "Objective (0.5 * residual_norm^2): {:.6}",
                    0.5 * summary.residual_norm * summary.residual_norm
                )?;
            } else {
                writeln!(
                    fit_file,
                    "Warning: Fit failed to converge. Recording initial parameters."
                )?;
            }
            writeln!(fit_file)?;
            writeln!(
                fit_file,
                "# Gaussian Components (amp, velocity[km/s], FWHM[km/s])"
            )?;
            for (idx, (amp, center, fwhm)) in comps.iter().enumerate() {
                writeln!(
                    fit_file,
                    "G{}: {:.6}\t{}\t{}",
                    idx + 1,
                    amp,
                    format_with_precision(*center, vel_precision),
                    format_with_precision(*fwhm, vel_precision)
                )?;
            }
        }

        let on_plot_data_freq: Vec<(f64, f32)> = analysis_freq_mhz
            .iter()
            .zip(analysis_spec_on.iter())
            .map(|(&x, &y)| (x, y))
            .collect();
        let off_plot_data_freq: Vec<(f64, f32)> = analysis_freq_mhz
            .iter()
            .zip(analysis_spec_off.iter())
            .map(|(&x, &y)| (x, y))
            .collect();
        let on_off_freq_plot_filename =
            output_dir.join(format!("{}{}_maser_onoff.png", on_stem, suffix));
        plot_on_off_spectra(
            &on_off_freq_plot_filename,
            &on_plot_data_freq,
            &off_plot_data_freq,
            "Frequency [MHz]",
            peak_freq_mhz,
            peak_velocity_kms,
            antenna_label,
            &off_label_for_plot,
            freq_precision,
            vel_precision,
        )?;

        let normalized_plot_data_freq: Vec<(f64, f32)> = analysis_freq_mhz
            .iter()
            .zip(normalized_spec.iter())
            .map(|(&freq, &y)| (freq, y))
            .collect();
        let maser_freq_plot_filename =
            output_dir.join(format!("{}{}_maser_subt.png", on_stem, suffix));
        plot_maser_spectrum(
            &maser_freq_plot_filename,
            &normalized_plot_data_freq,
            "Frequency [MHz]",
            spec_title,
            spec_y_label,
            antenna_label,
            peak_freq_mhz,
            peak_velocity_kms,
            peak_val,
            snr_mad,
            snr_stdev,
            median,
            mad,
            sigma_est,
            freq_resolution_mhz,
            channel_width_kms,
            freq_precision,
            vel_precision,
            None,
            None,
        )?;

        let normalized_plot_data_vel: Vec<(f64, f32)> = analysis_velocity_kms
            .iter()
            .zip(normalized_spec.iter())
            .map(|(&x, &y)| (x, y))
            .collect();
        let maser_vel_plot_filename =
            output_dir.join(format!("{}{}_maser_Vlsr1.png", on_stem, suffix));
        plot_maser_spectrum(
            &maser_vel_plot_filename,
            &normalized_plot_data_vel,
            "LSR Velocity [km/s]",
            spec_title,
            spec_y_label,
            antenna_label,
            peak_freq_mhz,
            peak_velocity_kms,
            peak_val,
            snr_mad,
            snr_stdev,
            median,
            mad,
            sigma_est,
            freq_resolution_mhz,
            channel_width_kms,
            freq_precision,
            vel_precision,
            gaussian_fit_data_vel.as_ref().map(|v| v.as_slice()),
            gaussian_fit_components
                .as_ref()
                .map(|components| components.as_slice()),
        )?;

        let vel_window_kms = 10.0;
        let min_zoom_vel = peak_velocity_kms - vel_window_kms;
        let max_zoom_vel = peak_velocity_kms + vel_window_kms;
        let zoomed_plot_data: Vec<(f64, f32)> = analysis_velocity_kms
            .iter()
            .zip(normalized_spec.iter())
            .filter(|(&vel, _)| vel >= min_zoom_vel && vel <= max_zoom_vel)
            .map(|(&vel, &norm_val)| (vel, norm_val))
            .collect();

        if !zoomed_plot_data.is_empty() {
            let gaussian_fit_zoom: Option<Vec<(f64, f32)>> =
                gaussian_fit_data_vel.as_ref().map(|fit_data| {
                    fit_data
                        .iter()
                        .filter(|(vel, _)| *vel >= min_zoom_vel && *vel <= max_zoom_vel)
                        .cloned()
                        .collect()
                });
            let maser_zoom_plot_filename =
                output_dir.join(format!("{}{}_maser_Vlsr2.png", on_stem, suffix));
            plot_maser_spectrum(
                &maser_zoom_plot_filename,
                &zoomed_plot_data,
                "LSR Velocity [km/s]",
                spec_title,
                spec_y_label,
                antenna_label,
                peak_freq_mhz,
                peak_velocity_kms,
                peak_val,
                snr_mad,
                snr_stdev,
                median,
                mad,
                sigma_est,
                freq_resolution_mhz,
                channel_width_kms,
                freq_precision,
                vel_precision,
                gaussian_fit_zoom.as_ref().map(|v| v.as_slice()),
                gaussian_fit_components
                    .as_ref()
                    .map(|components| components.as_slice()),
            )?;
        }
    }

    Ok(SegmentSummary {
        index: seg_idx,
        start_time: on_seg.start_time,
        segment_seconds,
        lsr_average: lsr_vel_corr,
        peak_freq_mhz,
        peak_velocity_kms,
        peak_value: peak_val,
        median,
        mad,
        peak_minus_median,
        snr_mad,
        snr_stdev,
        sigma_est,
    })
}

// New function starts here
pub fn calculate_lsr_velocity_correction(
    ant_position_geocentric: [f64; 3], // meters
    time: &DateTime<Utc>,
    obs_ra_rad: f64,  // radians
    obs_dec_rad: f64, // radians
) -> f64 {
    // returns correction in km/s

    // Constants for solar motion w.r.t. LSR (from Python script)
    // These are typically given in a specific coordinate system (e.g., ICRS-aligned at Sun's position)
    // Assuming these are in the same equatorial frame as the calculated v_obs_helio
    // The solar motion vector (v_sun_lsr) is derived from a standard solar motion model.
    // The components (U, V, W) are typically given in a coordinate system aligned with ICRS at the Sun's position.
    //
    // For reference, these values can be calculated using astropy as follows:
    //
    // # Define the direction of the Sun's motion
    // from astropy.coordinates import SkyCoord, LSR, CartesianDifferential
    // from astropy.time import Time
    // import astropy.units as u
    // import numpy as np
    //
    // dir_sun = SkyCoord(
    //     ra = 18 * 15 * u.deg,                   # R.A. = 18 h
    //     dec = 30 * u.deg,                       # Dec. = 30 deg
    //     frame = 'fk4',
    //     equinox = Time('B1900'), # 1900 年分点を指定する
    // ).galactic
    // # >>> dir_sun
    // # <SkyCoord (Galactic): (l, b) in deg
    // #     (56.15745659, 22.76480182)>
    //
    // # Define the speed of the Sun
    // v_sun = 20 * u.km / u.s
    //
    // # Decompose the solar velocity into components (U, V, W) in a coordinate system
    // # orthogonal to the galactic plane
    // U = v_sun * np.cos(dir_sun.b) * np.cos(dir_sun.l)
    // V = v_sun * np.cos(dir_sun.b) * np.sin(dir_sun.l)
    // W = v_sun * np.sin(dir_sun.b)
    //
    // # Convert to CartesianDifferential type
    // v_bary = CartesianDifferential(U, V, W)
    // # >>> v_bary
    // # <CartesianDifferential (d_x, d_y, d_z) in km / s
    // #     (10.27059164, 15.31741091, 7.73898381)>
    //
    // These components are then used as v_sun_lsr.
    // 銀河座標系 (U, V, W) → J2000 等赤道座標系 (ICRS) への変換行列（IAU 1958 定義）
    // 参考: Astropy 実装
    let gal_to_icrs = Matrix3::new(
        -0.054_875_560_416_215_4,
        0.494_109_427_875_583_7,
        -0.867_666_149_019_004_7,
        -0.873_437_090_234_885_0,
        -0.444_829_629_960_011_2,
        -0.198_076_373_431_201_5,
        -0.483_835_015_548_713_2,
        0.746_982_244_497_218_9,
        0.455_983_776_175_066_9,
    );

    let v_sun_lsr_gal = Vector3::new(10.270_591_64, 15.317_410_91, 7.738_983_81); // km/s (U, V, W)
    let v_sun_lsr = gal_to_icrs * v_sun_lsr_gal; // km/s（ICRS）

    // 1. Time conversion to Julian Day
    let decimal_day = time.day() as f64
        + time.hour() as f64 / 24.0
        + time.minute() as f64 / 1440.0
        + (time.second() as f64 + time.nanosecond() as f64 / 1e9) / 86400.0;
    let date = Date {
        year: time.year() as i16,
        month: time.month() as u8,
        decimal_day,
        cal_type: time::CalType::Gregorian,
    };
    let jd = time::julian_day(&date);

    // 2. Earth's orbital velocity v_orb (heliocentric equatorial frame)
    // Calculated by finite difference of Earth's heliocentric position
    let delta_jd = 0.001; // days for finite difference
    let au_to_km = 149597870.7; // km

    let sph_to_rect = |lon: f64, lat: f64, radius_au: f64| {
        let r_km = radius_au * au_to_km;
        let cos_lat = lat.cos();
        Vector3::new(
            r_km * cos_lat * lon.cos(),
            r_km * cos_lat * lon.sin(),
            r_km * lat.sin(),
        )
    };

    let (lon_minus, lat_minus, radius_minus) =
        planet::heliocent_coords(&Planet::Earth, jd - delta_jd);
    let pos_minus_ecl = sph_to_rect(lon_minus, lat_minus, radius_minus);

    let (lon_plus, lat_plus, radius_plus) = planet::heliocent_coords(&Planet::Earth, jd + delta_jd);
    let pos_plus_ecl = sph_to_rect(lon_plus, lat_plus, radius_plus);

    // Obliquity of the ecliptic
    let epsilon = ecliptic::mn_oblq_IAU(jd);

    // Rotation matrix from ecliptic to equatorial coordinates
    let rot_matrix_ecl_to_eq = nalgebra::Matrix3::new(
        1.0,
        0.0,
        0.0,
        0.0,
        epsilon.cos(),
        -epsilon.sin(),
        0.0,
        epsilon.sin(),
        epsilon.cos(),
    );

    let pos_minus_eq = rot_matrix_ecl_to_eq * pos_minus_ecl;
    let pos_plus_eq = rot_matrix_ecl_to_eq * pos_plus_ecl;

    // Orbital velocity (km/s)
    let v_orb = (pos_plus_eq - pos_minus_eq) / (2.0 * delta_jd * 86400.0); // 2*delta_jd is in days, convert to seconds

    // 3. Earth's rotational velocity v_rot (geocentric equatorial frame)
    let omega_magnitude_rad_per_day = earth::rot_angular_velocity(); // No arguments
    let omega_magnitude_rad_per_s = omega_magnitude_rad_per_day / 86400.0; // rad/s

    // Construct omega_vec assuming alignment with Z-axis in equatorial coordinates
    let omega_vec = Vector3::new(0.0, 0.0, omega_magnitude_rad_per_s);

    // Observer's geocentric position in km (ECEF frame)
    let r_obs_ecef_km = Vector3::new(
        ant_position_geocentric[0] / 1000.0,
        ant_position_geocentric[1] / 1000.0,
        ant_position_geocentric[2] / 1000.0,
    );

    // Transform r_obs_ecef_km from ECEF to equatorial (ICRS) frame
    // Incorporating precession and nutation (polar motion is ignored)

    // 1. Greenwich Mean Sidereal Time (GMST)
    let gmst_rad = time::mn_sidr(jd);

    // 2. Nutation angles
    let (delta_psi, delta_epsilon) = nutation::nutation(jd);

    // 3. Precession angles (ignored as per user's request)
    // let (zeta, z, theta) = precess::precession_angles(jd);

    // Mean obliquity of the ecliptic
    let mean_obliquity = ecliptic::mn_oblq_IAU(jd);

    // Rotation matrix for Earth Rotation Angle (GMST)
    let r_era = nalgebra::Matrix3::new(
        gmst_rad.cos(),
        -gmst_rad.sin(),
        0.0,
        gmst_rad.sin(),
        gmst_rad.cos(),
        0.0,
        0.0,
        0.0,
        1.0,
    );

    // Nutation matrix (simplified, using mean obliquity)
    // This transforms from mean equinox of date to true equinox of date.
    // R_x(angle) = | 1  0           0          |
    //              | 0  cos(angle) -sin(angle) |
    //              | 0  sin(angle)  cos(angle) |
    let r_x_mean_obl = nalgebra::Matrix3::new(
        1.0,
        0.0,
        0.0,
        0.0,
        mean_obliquity.cos(),
        -mean_obliquity.sin(),
        0.0,
        mean_obliquity.sin(),
        mean_obliquity.cos(),
    );
    let r_x_mean_obl_plus_delta = nalgebra::Matrix3::new(
        1.0,
        0.0,
        0.0,
        0.0,
        (mean_obliquity + delta_epsilon).cos(),
        -(mean_obliquity + delta_epsilon).sin(),
        0.0,
        (mean_obliquity + delta_epsilon).sin(),
        (mean_obliquity + delta_epsilon).cos(),
    );
    let r_z_delta_psi = nalgebra::Matrix3::new(
        delta_psi.cos(),
        -delta_psi.sin(),
        0.0,
        delta_psi.sin(),
        delta_psi.cos(),
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let r_nutation = r_x_mean_obl_plus_delta * r_z_delta_psi * r_x_mean_obl.transpose(); // N = R_x(-e-de) * R_z(-dp) * R_x(e)

    // Precession matrix (ignored as per user's request)
    // P = R_z(-z) * R_y(theta) * R_z(-zeta)
    // let r_z_minus_zeta = nalgebra::Matrix3::new(
    //     (-zeta).cos(), -(-zeta).sin(), 0.0,
    //     (-zeta).sin(), (-zeta).cos(),  0.0,
    //     0.0,           0.0,            1.0,
    // );
    // let r_y_theta = nalgebra::Matrix3::new(
    //     theta.cos(), 0.0, theta.sin(),
    //     0.0,         1.0, 0.0,
    //     -theta.sin(), 0.0, theta.cos(),
    // );
    // let r_z_minus_z = nalgebra::Matrix3::new(
    //     (-z).cos(), -(-z).sin(), 0.0,
    //     (-z).sin(), (-z).cos(),  0.0,
    //     0.0,        0.0,         1.0,
    // );
    // let r_precession = r_z_minus_z * r_y_theta * r_z_minus_zeta;

    // Combined transformation matrix from ECEF to ICRS (J2000)
    // R_ICRS = N * R_ERA * R_ECEF (Precession ignored)
    // This is for position. For velocity, it's more complex.
    // Let's assume the rotation matrix for position is sufficient for now.
    let total_rotation_matrix = r_nutation * r_era;

    let r_obs_eq_km = total_rotation_matrix * r_obs_ecef_km;

    // Rotational velocity (km/s)
    let v_rot = omega_vec.cross(&r_obs_eq_km); // Use transformed position

    // 4. Observer's velocity w.r.t. heliocenter v_obs_helio
    let v_obs_helio = v_orb + v_rot; // km/s

    // 5. Observer's velocity w.r.t. LSR v_obs_lsr
    // v_obs_lsr = v_obs_helio + v_sun_lsr (as per astropy's LSR frame definition)
    let v_obs_lsr = v_obs_helio + v_sun_lsr; // km/s

    // 6. Target direction unit vector k
    let k = Vector3::new(
        obs_dec_rad.cos() * obs_ra_rad.cos(),
        obs_dec_rad.cos() * obs_ra_rad.sin(),
        obs_dec_rad.sin(),
    );

    // 7. Final correction v_correction (dot product of observer's LSR velocity and target direction)
    let v_correction = v_obs_lsr.dot(&k); // km/s

    v_correction
}
