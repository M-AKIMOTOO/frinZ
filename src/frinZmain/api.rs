//! Library API for using frinZ without CLI output or plots.
//!
//! This module keeps the command-line workflow available as reusable Rust
//! functions: read `.cor`, apply phase correction, run delay/rate fringe search,
//! and extract complex frequency spectra.

use std::error::Error;
use std::path::Path;

use chrono::{DateTime, Utc};
use ndarray::Array2;
use num_complex::Complex;

use crate::analysis::AnalysisResults;
use crate::args::Args;
use crate::bandpass;
use crate::fft::apply_phase_correction_in_place;
use crate::header::CorHeader;
use crate::processing::run_analysis_pipeline;
use crate::read::{
    read_cor_bytes, read_cor_file, read_cor_file_with_options, CorData, CorReadOptions,
};
use crate::search;

pub type C32 = Complex<f32>;

pub use crate::read::{read_cor_bytes as read_bytes, read_cor_file as read_file};

/// Delay/rate/aceleration phase correction applied to visibility spectra.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseCorrection {
    pub delay_samples: f32,
    pub rate_hz: f32,
    pub acceleration_hz_per_s: f32,
    pub jerk_hz_per_s2: f32,
    pub snap_hz_per_s3: f32,
    /// Time offset of the first row relative to the phase reference epoch.
    pub start_time_offset_sec: f32,
}

impl PhaseCorrection {
    pub fn is_zero(self) -> bool {
        self.delay_samples == 0.0
            && self.rate_hz == 0.0
            && self.acceleration_hz_per_s == 0.0
            && self.jerk_hz_per_s2 == 0.0
            && self.snap_hz_per_s3 == 0.0
    }
}

/// Fringe-search mode matching the CLI `--search` modes used for normal work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Peak,
    Deep,
}

impl SearchMode {
    fn as_str(self) -> &'static str {
        match self {
            SearchMode::Peak => "peak",
            SearchMode::Deep => "deep",
        }
    }
}

/// Library-side options. These deliberately omit plotting and file-output flags.
#[derive(Debug, Clone)]
pub struct LibraryOptions {
    pub correction: PhaseCorrection,
    pub search_mode: Option<SearchMode>,
    /// RFI channel ranges, inclusive, in FFT-half channel indices.
    pub rfi_channel_ranges: Vec<(usize, usize)>,
    /// Optional delay search window `[min, max]` in samples.
    pub delay_window: Option<[f32; 2]>,
    /// Optional fringe-rate search window `[min, max]` in Hz.
    pub rate_window: Option<[f32; 2]>,
    /// Optional mask rectangle `[delay_min, delay_max, rate_min, rate_max]`.
    pub delay_rate_mask: Option<[f32; 4]>,
    /// Optional frequency window `[min, max]` in MHz for frequency-domain analysis.
    pub frequency_window_mhz: Option<[f32; 2]>,
    pub rate_padding: u32,
    pub iter: u32,
    pub cpu: u32,
    pub bandpass: Option<Vec<C32>>,
}

impl Default for LibraryOptions {
    fn default() -> Self {
        Self {
            correction: PhaseCorrection::default(),
            search_mode: None,
            rfi_channel_ranges: Vec::new(),
            delay_window: None,
            rate_window: None,
            delay_rate_mask: None,
            frequency_window_mhz: None,
            rate_padding: 1,
            iter: 5,
            cpu: 0,
            bandpass: None,
        }
    }
}

/// Result of delay/rate analysis. No graph or sidecar files are created.
#[derive(Debug, Clone)]
pub struct FringeSearchOutput {
    pub header: CorHeader,
    pub obs_time: DateTime<Utc>,
    pub length_sectors: i32,
    pub effective_integ_time: f32,
    pub analysis: AnalysisResults,
    pub delay_rate_plane: Array2<C32>,
    /// Present when frequency-domain mode was requested by helper functions.
    pub freq_rate_plane: Option<Array2<C32>>,
}

/// Complex cross-power spectrum with a physical frequency axis.
#[derive(Debug, Clone)]
pub struct FrequencySpectrum {
    pub header: CorHeader,
    pub obs_time: DateTime<Utc>,
    pub frequency_mhz: Vec<f32>,
    pub spectrum: Vec<C32>,
    pub analysis: AnalysisResults,
}

/// Read a `.cor` file. This is the same low-level reader exposed at crate root.
pub fn read_cor<P: AsRef<Path>>(path: P) -> std::io::Result<CorData> {
    read_cor_file(path)
}

/// Read a `.cor` file with skip/length/loop options.
pub fn read_cor_with_options<P: AsRef<Path>>(
    path: P,
    options: &CorReadOptions,
) -> std::io::Result<CorData> {
    read_cor_file_with_options(path, options)
}

/// Read a bandpass table from frinZ NPZ or legacy BIN format.
pub fn read_bandpass<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<C32>> {
    bandpass::read_bandpass_file(path.as_ref())
}

/// Apply delay/rate/acceleration correction in place to already-read visibility data.
pub fn apply_delay_rate_correction(data: &mut CorData, correction: PhaseCorrection) {
    let fft_point_half = (data.header.fft_point / 2).max(0) as usize;
    apply_phase_correction_in_place(
        &mut data.visibility,
        fft_point_half,
        correction.rate_hz,
        correction.delay_samples,
        correction.acceleration_hz_per_s,
        correction.jerk_hz_per_s2,
        correction.snap_hz_per_s3,
        data.effective_integ_time,
        data.header.sampling_speed as u32,
        data.header.fft_point as u32,
        correction.start_time_offset_sec,
    );
}

/// Run the standard delay/rate analysis. Set `options.search_mode` for `--search` equivalent.
pub fn fringe_search(
    data: &CorData,
    options: &LibraryOptions,
) -> Result<FringeSearchOutput, Box<dyn Error>> {
    run_delay_rate(data, options, false)
}

/// Convenience wrapper for `--search peak` equivalent.
pub fn delay_search(
    data: &CorData,
    options: &LibraryOptions,
) -> Result<FringeSearchOutput, Box<dyn Error>> {
    let mut options = options.clone();
    if options.search_mode.is_none() {
        options.search_mode = Some(SearchMode::Peak);
    }
    run_delay_rate(data, &options, false)
}

/// Compute the complex frequency spectrum without creating plots or files.
pub fn frequency_spectrum(
    data: &CorData,
    options: &LibraryOptions,
) -> Result<FrequencySpectrum, Box<dyn Error>> {
    let output = run_delay_rate(data, options, true)?;
    Ok(FrequencySpectrum {
        header: output.header,
        obs_time: output.obs_time,
        frequency_mhz: output.analysis.freq_range.to_vec(),
        spectrum: output.analysis.freq_rate_spectrum.to_vec(),
        analysis: output.analysis,
    })
}

/// Parse `.cor` bytes and immediately run a peak delay/rate search.
pub fn search_cor_bytes(
    bytes: &[u8],
    read_options: &CorReadOptions,
    options: &LibraryOptions,
) -> Result<FringeSearchOutput, Box<dyn Error>> {
    let data = read_cor_bytes(bytes, read_options)?;
    delay_search(&data, options)
}

fn run_delay_rate(
    data: &CorData,
    options: &LibraryOptions,
    frequency_mode: bool,
) -> Result<FringeSearchOutput, Box<dyn Error>> {
    let mut args = args_from_options(options, frequency_mode);
    let mut complex_vec = data.visibility.clone();
    let fft_point_half = (data.header.fft_point / 2).max(0) as usize;
    if fft_point_half == 0 {
        return Err("FFT point must be >= 2".into());
    }
    if complex_vec.len() % fft_point_half != 0 {
        return Err(format!(
            "visibility length ({}) is not divisible by FFT-half channels ({})",
            complex_vec.len(),
            fft_point_half
        )
        .into());
    }

    if !options.correction.is_zero() {
        apply_phase_correction_in_place(
            &mut complex_vec,
            fft_point_half,
            options.correction.rate_hz,
            options.correction.delay_samples,
            options.correction.acceleration_hz_per_s,
            options.correction.jerk_hz_per_s2,
            options.correction.snap_hz_per_s3,
            data.effective_integ_time,
            data.header.sampling_speed as u32,
            data.header.fft_point as u32,
            options.correction.start_time_offset_sec,
        );
        args.delay_correct = 0.0;
        args.rate_correct = 0.0;
        args.acel_correct = 0.0;
    }

    let physical_length = (complex_vec.len() / fft_point_half) as i32;
    let current_length =
        pad_rows_to_power_of_two(&mut complex_vec, physical_length, fft_point_half);
    let bandpass = options.bandpass.clone();
    let file_start_time = data.obs_time;

    let (analysis, freq_rate_plane, delay_rate_plane, _) = if let Some(mode) = options.search_mode {
        let result = match mode {
            SearchMode::Peak => search::run_peak_search(
                &complex_vec,
                &data.header,
                current_length,
                physical_length,
                data.effective_integ_time,
                &data.obs_time,
                &file_start_time,
                &options.rfi_channel_ranges,
                &bandpass,
                &args,
                data.header.number_of_sector,
                args.cpu,
                None,
            )?,
            SearchMode::Deep => search::run_deep_search(
                &complex_vec,
                &data.header,
                current_length,
                physical_length,
                data.effective_integ_time,
                &data.obs_time,
                &file_start_time,
                &options.rfi_channel_ranges,
                &bandpass,
                &args,
                data.header.number_of_sector,
                args.cpu,
                None,
            )?,
        };
        (
            result.analysis_results,
            result.freq_rate_array,
            result.delay_rate_2d_data,
            result.pre_bandpass_analysis_results,
        )
    } else {
        run_analysis_pipeline(
            &complex_vec,
            &data.header,
            &args,
            None,
            0.0,
            0.0,
            0.0,
            current_length,
            physical_length,
            data.effective_integ_time,
            &data.obs_time,
            &file_start_time,
            &options.rfi_channel_ranges,
            &bandpass,
            false,
            data.header.fft_point,
        )?
    };

    Ok(FringeSearchOutput {
        header: data.header.clone(),
        obs_time: data.obs_time,
        length_sectors: physical_length,
        effective_integ_time: data.effective_integ_time,
        analysis,
        delay_rate_plane,
        freq_rate_plane,
    })
}

fn args_from_options(options: &LibraryOptions, frequency_mode: bool) -> Args {
    let mut args = Args::default();
    args.frequency = frequency_mode;
    args.spectrum = frequency_mode;
    args.rate_padding = options.rate_padding.max(1);
    args.iter = options.iter;
    args.cpu = options.cpu;
    args.delay_correct = options.correction.delay_samples;
    args.rate_correct = options.correction.rate_hz;
    args.acel_correct = options.correction.acceleration_hz_per_s;
    args.jerk_correct = options.correction.jerk_hz_per_s2;
    args.snap_correct = options.correction.snap_hz_per_s3;
    if let Some(mode) = options.search_mode {
        args.search = vec![mode.as_str().to_string()];
        if args.rate_padding < 8 {
            args.rate_padding = 8;
        }
    }
    if let Some([lo, hi]) = options.delay_window {
        args.drange = vec![lo, hi];
    }
    if let Some([lo, hi]) = options.rate_window {
        args.rrange = vec![lo, hi];
    }
    if let Some([d0, d1, r0, r1]) = options.delay_rate_mask {
        args.mask = vec![d0, d1, r0, r1];
    }
    if let Some([lo, hi]) = options.frequency_window_mhz {
        args.frange = vec![lo, hi];
    }
    args
}

fn pad_rows_to_power_of_two(data: &mut Vec<C32>, current_rows: i32, row_width: usize) -> i32 {
    if current_rows <= 0 || row_width == 0 {
        return current_rows;
    }
    let target_rows = if current_rows <= 1 {
        1
    } else {
        (current_rows as u32).next_power_of_two() as i32
    };
    if target_rows > current_rows {
        data.extend(
            std::iter::repeat(C32::new(0.0, 0.0))
                .take((target_rows - current_rows) as usize * row_width),
        );
    }
    target_rows
}
