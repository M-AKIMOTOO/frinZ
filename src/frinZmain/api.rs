//! Library API for using frinZ without CLI output or plots.
//!
//! This module keeps the command-line workflow available as reusable Rust
//! functions: read `.cor`, apply phase correction, run delay/rate fringe search,
//! and extract complex frequency spectra.

use std::borrow::Cow;
use std::error::Error;
use std::path::Path;

use chrono::{DateTime, Utc};
use ndarray::Array2;
use num_complex::Complex;

use crate::analysis::AnalysisResults;
use crate::args::{Args, DEFAULT_RATE_PADDING};
use crate::bandpass;
use crate::fft::{
    apply_phase_correction_in_place_at_frequency, apply_phase_correction_in_place_at_times,
};
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
    fn apply(self, visibility: &mut [C32], header: &CorHeader, integration_time: f32) {
        apply_phase_correction_in_place_at_frequency(
            visibility,
            (header.fft_point / 2).max(0) as usize,
            self.rate_hz,
            self.delay_samples,
            self.acceleration_hz_per_s,
            self.jerk_hz_per_s2,
            self.snap_hz_per_s3,
            integration_time,
            header.sampling_speed as u32,
            header.fft_point as u32,
            self.start_time_offset_sec,
            header.observing_frequency,
        );
    }

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
    Coherent,
    /// Continuous full-band delay/rate fit with a formal uncertainty at the first row.
    GroupDelay,
}

impl SearchMode {
    fn as_str(self) -> &'static str {
        match self {
            SearchMode::Peak => "peak",
            SearchMode::Deep => "deep",
            SearchMode::Coherent => "coherent",
            SearchMode::GroupDelay => "group-delay",
        }
    }
}

/// Library-side options. These deliberately omit plotting and file-output flags.
#[derive(Debug, Clone)]
pub struct LibraryOptions {
    /// Applied before searching. The result reports these terms in
    /// `corrected_*` and the measured remaining offsets in `residual_*`.
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
            rate_padding: DEFAULT_RATE_PADDING,
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
    correction.apply(
        &mut data.visibility,
        &data.header,
        data.effective_integ_time,
    );
}

/// Run the standard delay/rate analysis. Set `options.search_mode` for `--search` equivalent.
pub fn fringe_search(
    data: &CorData,
    options: &LibraryOptions,
) -> Result<FringeSearchOutput, Box<dyn Error>> {
    run_delay_rate(data, options, options.search_mode, false, None)
}

/// Convenience wrapper for `--search peak` equivalent.
pub fn delay_search(
    data: &CorData,
    options: &LibraryOptions,
) -> Result<FringeSearchOutput, Box<dyn Error>> {
    run_delay_rate(
        data,
        options,
        options.search_mode.or(Some(SearchMode::Peak)),
        false,
        None,
    )
}

/// Search a visibility buffer using actual sector times in seconds relative to
/// a common origin. Both the fit and the manual correction use the first row
/// as their epoch, with the optional correction start offset applied to the latter.
pub fn delay_search_at_times(
    data: &CorData,
    options: &LibraryOptions,
    times: &[f64],
) -> Result<FringeSearchOutput, Box<dyn Error>> {
    run_delay_rate(
        data,
        options,
        options.search_mode.or(Some(SearchMode::Peak)),
        false,
        Some(times),
    )
}

/// Compute the complex frequency spectrum without creating plots or files.
pub fn frequency_spectrum(
    data: &CorData,
    options: &LibraryOptions,
) -> Result<FrequencySpectrum, Box<dyn Error>> {
    let output = run_delay_rate(data, options, options.search_mode, true, None)?;
    Ok(FrequencySpectrum {
        header: output.header,
        obs_time: output.obs_time,
        frequency_mhz: output.analysis.freq_range.to_vec(),
        spectrum: output.analysis.freq_rate_spectrum.to_vec(),
        analysis: output.analysis,
    })
}

/// Parse `.cor` bytes and search using the recorded sector epochs (including gaps).
pub fn search_cor_bytes(
    bytes: &[u8],
    read_options: &CorReadOptions,
    options: &LibraryOptions,
) -> Result<FringeSearchOutput, Box<dyn Error>> {
    let data = read_cor_bytes(bytes, read_options)?;
    let times = crate::read::read_sector_time_offsets(
        bytes,
        &data.header,
        data.first_sector as usize,
        data.sectors_read as usize,
    )?;
    delay_search_at_times(&data, options, &times)
}

fn run_delay_rate(
    data: &CorData,
    options: &LibraryOptions,
    search_mode: Option<SearchMode>,
    frequency_mode: bool,
    recorded_times: Option<&[f64]>,
) -> Result<FringeSearchOutput, Box<dyn Error>> {
    let args = args_from_options(options, search_mode, frequency_mode);
    let fft_point_half = (data.header.fft_point / 2).max(0) as usize;
    if fft_point_half == 0 {
        return Err("FFT point must be >= 2".into());
    }
    if !data.visibility.len().is_multiple_of(fft_point_half) {
        return Err(format!(
            "visibility length ({}) is not divisible by FFT-half channels ({})",
            data.visibility.len(),
            fft_point_half
        )
        .into());
    }

    let rows = data.visibility.len() / fft_point_half;
    if rows == 0 {
        return Err("visibility contains no sectors".into());
    }
    if let Some(times) = recorded_times {
        if times.len() != rows || times.iter().any(|t| !t.is_finite()) {
            return Err("expected one finite timestamp per physical visibility row".into());
        }
    }
    let padded_rows = rows
        .checked_next_power_of_two()
        .ok_or("too many visibility rows")?;
    let current_length = i32::try_from(padded_rows)?;
    let physical_length = rows as i32;

    // The public API returns a power-of-two rate axis. Only materialize an
    // owned buffer when padding or a manual phase rotation actually needs it.
    let complex_vec = if rows == padded_rows && options.correction.is_zero() {
        Cow::Borrowed(data.visibility.as_slice())
    } else {
        let samples = padded_rows
            .checked_mul(fft_point_half)
            .ok_or("visibility size overflow")?;
        let mut values = Vec::with_capacity(samples);
        values.extend_from_slice(&data.visibility);
        if !options.correction.is_zero() {
            let correction = options.correction;
            if let Some(times) = recorded_times {
                let offset = f64::from(correction.start_time_offset_sec);
                let epochs: Vec<_> = times.iter().map(|t| (t - times[0]) + offset).collect();
                apply_phase_correction_in_place_at_times(
                    &mut values,
                    fft_point_half,
                    correction.rate_hz,
                    correction.delay_samples,
                    correction.acceleration_hz_per_s,
                    correction.jerk_hz_per_s2,
                    correction.snap_hz_per_s3,
                    data.header.sampling_speed as u32,
                    data.header.fft_point as u32,
                    &epochs,
                    data.header.observing_frequency,
                    0.0,
                );
            } else {
                correction.apply(&mut values, &data.header, data.effective_integ_time);
            }
        }
        values.resize(samples, C32::default());
        Cow::Owned(values)
    };

    let (mut analysis, freq_rate_plane, delay_rate_plane, _) = if let Some(mode) = search_mode {
        let search = match mode {
            SearchMode::Deep => search::run_deep_search_with_phase_times,
            SearchMode::Peak | SearchMode::Coherent | SearchMode::GroupDelay => {
                search::run_peak_search_with_phase_times
            }
        };
        let regular_times;
        let times = if let Some(times) = recorded_times {
            times
        } else {
            regular_times = (0..rows)
                .map(|i| i as f64 * f64::from(data.effective_integ_time))
                .collect::<Vec<_>>();
            &regular_times
        };
        let result = search(
            &complex_vec,
            &data.header,
            current_length,
            physical_length,
            data.effective_integ_time,
            &data.obs_time,
            &data.obs_time,
            &options.rfi_channel_ranges,
            &options.bandpass,
            &args,
            data.header.number_of_sector,
            args.cpu,
            None,
            times,
        )?;
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
            &data.obs_time,
            &options.rfi_channel_ranges,
            &options.bandpass,
            false,
            data.header.fft_point,
        )?
    };

    analysis.length_f32 = physical_length as f32 * data.effective_integ_time;
    analysis.corrected_delay = options.correction.delay_samples;
    analysis.corrected_rate = options.correction.rate_hz;
    analysis.corrected_acel = options.correction.acceleration_hz_per_s;
    analysis.corrected_jerk = options.correction.jerk_hz_per_s2;
    analysis.corrected_snap = options.correction.snap_hz_per_s3;
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

fn args_from_options(
    options: &LibraryOptions,
    search_mode: Option<SearchMode>,
    frequency_mode: bool,
) -> Args {
    let mut args = Args {
        frequency: frequency_mode,
        spectrum: frequency_mode,
        rate_padding: options.rate_padding.max(1),
        iter: options.iter,
        cpu: options.cpu,
        // All manual Taylor terms have already been applied to the samples.
        ..Args::default()
    };
    if let Some(mode) = search_mode {
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use std::f64::consts::PI;

    fn wrap_degrees(value: f64) -> f64 {
        (value + 180.0).rem_euclid(360.0) - 180.0
    }

    fn synthetic_data(
        length: usize,
        start_sec: f64,
        delay: f64,
        rate: f64,
        phase0: f64,
    ) -> CorData {
        synthetic_data_with_acceleration(length, start_sec, delay, rate, 0.0, phase0)
    }

    fn synthetic_data_with_acceleration(
        length: usize,
        start_sec: f64,
        delay: f64,
        rate: f64,
        acceleration: f64,
        phase0: f64,
    ) -> CorData {
        let fft_point = 64usize;
        let channels = fft_point / 2;
        let visibility = (0..length)
            .flat_map(|row| {
                (0..channels).map(move |channel| {
                    let sampling_speed_hz = 64_000_000.0;
                    let reference_frequency_hz = 8_400_000_000.0;
                    let time = start_sec + row as f64;
                    let temporal_cycles = rate * time + 0.5 * acceleration * time * time;
                    let baseband_frequency_hz =
                        channel as f64 * sampling_speed_hz / fft_point as f64;
                    let phase = phase0
                        + 2.0
                            * PI
                            * (temporal_cycles
                                * (1.0 + baseband_frequency_hz / reference_frequency_hz)
                                + delay * channel as f64 / fft_point as f64);
                    C32::new(phase.cos() as f32, phase.sin() as f32)
                })
            })
            .collect();
        CorData {
            header: CorHeader {
                sampling_speed: 64_000_000,
                observing_frequency: 8_400_000_000.0,
                fft_point: fft_point as i32,
                number_of_sector: 600,
                station1_position: [6_378_137.0, 0.0, 0.0],
                station2_position: [6_378_137.0, 100.0, 0.0],
                ..CorHeader::default()
            },
            visibility,
            obs_time: Utc.timestamp_opt(start_sec as i64, 0).single().unwrap(),
            effective_integ_time: 1.0,
            first_sector: 0,
            sectors_read: length as i32,
        }
    }

    #[test]
    fn group_delay_uses_recorded_epochs_bandpass_and_flags() {
        let times = [51.2, 52.01, 53.5, 54.1, 57.03, 60.1, 61.4, 62.0, 62.9];
        let delay = -13.246813579;
        let rate = 0.07123456789;
        let mut data = synthetic_data(times.len(), 0.0, delay, rate, 1.2);
        let gains: Vec<_> = (0..32)
            .map(|k| C32::from_polar(0.7 + k as f32 / 40.0, k as f32 / 17.0))
            .collect();
        for (i, row) in data.visibility.chunks_exact_mut(32).enumerate() {
            for (k, z) in row.iter_mut().enumerate() {
                let f = k as f64 / 64.0;
                let phi = 1.2
                    + 2.0
                        * PI
                        * (delay * f + rate * (times[i] - times[0]) * (1.0 + f * 64e6 / 8.4e9));
                *z = if i == 3 {
                    C32::default()
                } else if (13..=19).contains(&k) {
                    C32::new(500.0, 300.0)
                } else {
                    C32::new(phi.cos() as f32, phi.sin() as f32) * gains[k]
                };
            }
        }
        let options = LibraryOptions {
            search_mode: Some(SearchMode::GroupDelay),
            bandpass: Some(gains),
            rfi_channel_ranges: vec![(13, 19)],
            delay_window: Some([-14.0, -12.0]),
            rate_window: Some([0.06, 0.09]),
            iter: 1,
            cpu: 1,
            ..LibraryOptions::default()
        };
        let fit = delay_search_at_times(&data, &options, &times)
            .unwrap()
            .analysis
            .group_delay
            .unwrap()
            .unwrap();
        assert!((fit.delay_samples - delay).abs() < 2e-8, "{fit:?}");
        assert!((fit.rate_hz - rate).abs() < 3e-10, "{fit:?}");
        assert_eq!(fit.visibilities, 8 * 24);
        let frequencies: Vec<_> = (1..32)
            .filter(|k| !(13..=19).contains(k))
            .map(|k| k as f64 * 1e6)
            .collect();
        let mean = frequencies.iter().sum::<f64>() / 24.0;
        let bandwidth = (frequencies.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / 24.0).sqrt();
        assert!((fit.bandwidth_rms_hz - bandwidth).abs() < 1e-6);
    }

    #[test]
    fn group_delay_requires_frequency_leverage_and_an_interior_peak() {
        let opts = LibraryOptions {
            search_mode: Some(SearchMode::GroupDelay),
            iter: 1,
            cpu: 1,
            ..LibraryOptions::default()
        };
        let mut data = synthetic_data(1, 0.0, 4.31415926, 0.0, 0.4);
        // Two named stations can have unfilled positions and station codes.
        data.header.station1_name = "ANT1".into();
        data.header.station2_name = "ANT2".into();
        data.header.station2_position = data.header.station1_position;
        let run = |opts: &LibraryOptions| {
            delay_search(&data, opts)
                .unwrap()
                .analysis
                .group_delay
                .unwrap()
        };
        let fit = run(&opts).unwrap();
        assert!((fit.delay_samples - 4.31415926).abs() < 2e-8 && !fit.rate_fitted);
        let one_frequency = LibraryOptions {
            rfi_channel_ranges: vec![(2, 31)],
            ..opts.clone()
        };
        assert_eq!(run(&one_frequency).unwrap_err(), "one_frequency");
        let two_frequencies = LibraryOptions {
            rfi_channel_ranges: vec![(3, 31)],
            ..opts.clone()
        };
        assert!(run(&two_frequencies).unwrap().sigma_seconds.is_none());
        let bounded = LibraryOptions {
            delay_window: Some([4.0, 4.25]),
            ..opts
        };
        assert!(run(&bounded).is_err());
        let comb = LibraryOptions {
            rfi_channel_ranges: (1..32).filter(|k| k % 4 != 3).map(|k| (k, k)).collect(),
            delay_window: Some([3.9, 4.7]),
            ..bounded
        };
        assert!((run(&comb).unwrap().ambiguity_seconds - 250e-9).abs() < 1e-18);
    }

    #[test]
    fn group_delay_formal_error_matches_independent_noise_trials() {
        let delay = 0.314159265;
        let rate = -0.037123;
        let options = LibraryOptions {
            search_mode: Some(SearchMode::GroupDelay),
            iter: 1,
            cpu: 1,
            delay_window: Some([-0.3, 0.9]),
            rate_window: Some([-0.055, -0.020]),
            ..LibraryOptions::default()
        };
        let model = synthetic_data(16, 0.0, delay, rate, 0.73);
        let mut seed = 4935817u64;
        let mut uniform = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut sum_error = 0.0;
        let mut squared_errors = 0.0;
        let mut sum_variance = 0.0;
        for _ in 0..128 {
            let mut data = model.clone();
            for (idx, z) in data.visibility.iter_mut().enumerate() {
                // Heteroscedastic, independent circular Gaussian channel noise.
                let sigma = 0.25 + (idx % 32) as f64 / 30.0;
                let r = sigma * (-2.0 * uniform().ln()).sqrt();
                let angle = 2.0 * PI * uniform();
                *z += C32::new((r * angle.cos()) as f32, (r * angle.sin()) as f32);
            }
            let fit = delay_search(&data, &options)
                .unwrap()
                .analysis
                .group_delay
                .unwrap()
                .unwrap();
            let error = fit.delay_seconds - delay / 64e6;
            sum_error += error;
            squared_errors += error * error;
            sum_variance += fit.sigma_seconds.unwrap().powi(2);
        }
        let rms_ps = 1e12 * (squared_errors / 128.0).sqrt();
        let predicted_ps = 1e12 * (sum_variance / 128.0).sqrt();
        eprintln!(
            "128 independent injections: RMS {rms_ps:.3} ps, formal RMS {predicted_ps:.3} ps"
        );
        assert!((0.80..1.20).contains(&(rms_ps / predicted_ps)));
        assert!(sum_error.abs() < 3.0 * sum_variance.sqrt());
    }

    #[test]
    fn fine_search_resolves_nearby_high_snr_candidates() {
        let delay = 0.31427;
        let rate = 0.03711;
        let data = synthetic_data(32, 0.0, delay, rate, 0.4);
        for mode in [SearchMode::Peak, SearchMode::Deep, SearchMode::Coherent] {
            let options = LibraryOptions {
                search_mode: Some(mode),
                iter: 6,
                cpu: 2,
                ..LibraryOptions::default()
            };
            let result = delay_search(&data, &options).unwrap().analysis;
            assert!(
                (result.residual_delay as f64 - delay).abs() < 2.0e-6,
                "{mode:?}: delay={}",
                result.residual_delay
            );
            assert!(
                (result.residual_rate as f64 - rate).abs() < 2.0e-7,
                "{mode:?}: rate={}",
                result.residual_rate
            );
        }
    }

    #[test]
    fn sub_bin_windows_recover_a_fringe_with_bandpass_and_flagged_channels() {
        let mut data = synthetic_data(37, 0.0, 5.55, 0.10625, 0.4);
        let channels = data.header.fft_point as usize / 2;
        let bandpass: Vec<_> = (0..channels)
            .map(|i| C32::from_polar(1.0 + i as f32 * 0.02, i as f32 * 0.05))
            .collect();
        for row in data.visibility.chunks_exact_mut(channels) {
            for (sample, gain) in row.iter_mut().zip(&bandpass) {
                *sample *= gain;
            }
            row[6..=7].fill(C32::new(1e4, -1e4));
        }
        let options = LibraryOptions {
            delay_window: Some([5.6, 5.4]),
            rate_window: Some([0.108, 0.104]),
            bandpass: Some(bandpass),
            rfi_channel_ranges: vec![(6, 7)],
            iter: 6,
            cpu: 1,
            ..LibraryOptions::default()
        };
        let result = delay_search(&data, &options).unwrap().analysis;
        assert!(
            (result.residual_delay - 5.55).abs() < 2e-5,
            "delay={}",
            result.residual_delay
        );
        assert!(
            (result.residual_rate - 0.10625).abs() < 2e-7,
            "rate={}",
            result.residual_rate
        );
        let excluded = LibraryOptions {
            delay_rate_mask: Some([5.4, 5.6, 0.104, 0.108]),
            ..options
        };
        assert!(delay_search(&data, &excluded).is_err());
    }

    #[test]
    fn manual_taylor_correction_is_applied_once_in_all_analysis_paths() {
        let mut data = synthetic_data(13, 0.0, 0.0, 0.0, 0.4);
        data.effective_integ_time = 0.5;
        let reference = data.clone();
        let correction = PhaseCorrection {
            delay_samples: 0.25,
            rate_hz: 0.01,
            acceleration_hz_per_s: -0.001,
            jerk_hz_per_s2: 0.002,
            snap_hz_per_s3: -0.00012,
            start_time_offset_sec: 5.125,
        };
        let fft = data.header.fft_point as usize;
        for (row, values) in data.visibility.chunks_exact_mut(fft / 2).enumerate() {
            let time = f64::from(correction.start_time_offset_sec)
                + row as f64 * f64::from(data.effective_integ_time);
            let cycles = f64::from(correction.rate_hz) * time
                + f64::from(correction.acceleration_hz_per_s) * time.powi(2) / 2.0
                + f64::from(correction.jerk_hz_per_s2) * time.powi(3) / 6.0
                + f64::from(correction.snap_hz_per_s3) * time.powi(4) / 24.0;
            for (channel, sample) in values.iter_mut().enumerate() {
                let fraction = channel as f64 / fft as f64;
                let rf_ratio = 1.0
                    + fraction * data.header.sampling_speed as f64
                        / data.header.observing_frequency;
                let angle =
                    2.0 * PI * (cycles * rf_ratio + f64::from(correction.delay_samples) * fraction);
                *sample *= C32::new(angle.cos() as f32, angle.sin() as f32);
            }
        }
        for mode in [None, Some(SearchMode::Peak)] {
            let options = LibraryOptions {
                search_mode: mode,
                cpu: 1,
                ..LibraryOptions::default()
            };
            let corrected = LibraryOptions {
                correction,
                ..options.clone()
            };
            let expected = fringe_search(&reference, &options).unwrap();
            let actual = fringe_search(&data, &corrected).unwrap();
            assert_eq!(actual.analysis.length_f32, 6.5);
            assert_eq!(actual.analysis.corrected_delay, correction.delay_samples);
            assert_eq!(actual.analysis.corrected_rate, correction.rate_hz);
            assert_eq!(actual.analysis.corrected_jerk, correction.jerk_hz_per_s2);
            assert_eq!(actual.analysis.corrected_snap, correction.snap_hz_per_s3);
            assert_eq!(actual.delay_rate_plane.dim(), (16, fft));
            for (a, b) in actual
                .delay_rate_plane
                .iter()
                .zip(&expected.delay_rate_plane)
            {
                assert!((*a - *b).norm() < 2.0e-3, "{mode:?}: {a} versus {b}");
            }
            let expected = frequency_spectrum(&reference, &options).unwrap();
            let actual = frequency_spectrum(&data, &corrected).unwrap();
            for (a, b) in actual.spectrum.iter().zip(&expected.spectrum) {
                assert!((*a - *b).norm() < 3.0e-3, "{mode:?}: {a} versus {b}");
            }
        }
    }

    #[test]
    fn peak_search_recovers_delay_rate_and_start_phase_for_all_lengths() {
        let delay = 0.35;
        let rate = 0.037;
        let phase0 = 0.4;
        let start = 100.0;
        let options = LibraryOptions {
            search_mode: Some(SearchMode::Peak),
            rate_padding: 8,
            iter: 4,
            cpu: 1,
            ..LibraryOptions::default()
        };

        for length in [1usize, 2, 3, 4, 5, 10, 30, 60] {
            let data = synthetic_data(length, start, delay, rate, phase0);
            let output = delay_search(&data, &options).unwrap();
            let expected_phase = wrap_degrees((phase0 + 2.0 * PI * rate * start).to_degrees());
            println!(
                "length={length} delay={} rate={} phase={} expected_phase={expected_phase}",
                output.analysis.residual_delay,
                output.analysis.residual_rate,
                output.analysis.delay_phase
            );
            let phase_error =
                wrap_degrees(output.analysis.delay_phase as f64 - expected_phase).abs();
            let expected_delay = delay + 64_000_000.0 * rate * start / 8_400_000_000.0;
            assert!(
                (output.analysis.residual_delay as f64 - expected_delay).abs() < 1.0e-3,
                "length={length}: delay={} expected={expected_delay}",
                output.analysis.residual_delay
            );
            if length == 1 {
                assert_eq!(
                    output.analysis.residual_rate, 0.0,
                    "length=1 rate is not identifiable"
                );
            } else {
                assert!(
                    (output.analysis.residual_rate as f64 - rate).abs() < 1.0e-4,
                    "length={length}: rate={}",
                    output.analysis.residual_rate
                );
            }
            assert!(
                phase_error < 0.5,
                "length={length}: phase={} expected={expected_phase} error={phase_error}",
                output.analysis.delay_phase
            );
        }
    }

    #[test]
    fn recovered_phase_has_the_physical_quadratic_coefficient_for_every_length() {
        let delay = 0.35;
        let rate0 = 0.0023;
        let acceleration = 2.0e-6;
        let phase0 = 0.4;
        let options = LibraryOptions {
            search_mode: Some(SearchMode::Peak),
            rate_padding: 8,
            iter: 4,
            cpu: 1,
            ..LibraryOptions::default()
        };

        for length in [1usize, 2, 3, 4, 5, 10, 30, 60] {
            let mut times = Vec::new();
            let mut phases = Vec::new();
            let mut rates = Vec::new();
            for start in [100.0_f64, 160.0, 220.0, 280.0, 340.0] {
                let data = synthetic_data_with_acceleration(
                    length,
                    start,
                    delay,
                    rate0,
                    acceleration,
                    phase0,
                );
                let output = delay_search(&data, &options).unwrap();
                times.push(start as f32);
                phases.push(output.analysis.delay_phase);
                rates.push(if length == 1 {
                    (rate0 + acceleration * start) as f32
                } else {
                    output.analysis.residual_rate
                });
            }
            let mut unwrapped = crate::utils::unwrap_phase_with_rate(&phases, &times, &rates);
            let expected_first = (phase0
                + 2.0
                    * PI
                    * (rate0 * times[0] as f64 + 0.5 * acceleration * (times[0] as f64).powi(2)))
            .to_degrees();
            let shift = ((expected_first - unwrapped[0] as f64) / 360.0).round() as f32 * 360.0;
            for phase in &mut unwrapped {
                *phase += shift;
            }
            let x: Vec<f64> = times.iter().map(|&value| value as f64).collect();
            let y: Vec<f64> = unwrapped.iter().map(|&value| value as f64).collect();
            let coeffs = crate::fitting::fit_polynomial_least_squares(&x, &y, 2).unwrap();
            let expected_quadratic = 180.0 * acceleration;
            assert!(
                (coeffs[2] - expected_quadratic).abs() < 4.0e-6,
                "length={length}: quadratic={} expected={expected_quadratic}",
                coeffs[2]
            );
        }
    }

    #[test]
    fn peak_search_reacquires_rate_instead_of_trusting_a_stale_seed() {
        let length = 30usize;
        let delay = 0.35;
        let rates = [-0.016, -0.17];
        let options = LibraryOptions {
            search_mode: Some(SearchMode::Peak),
            rate_padding: 8,
            iter: 4,
            cpu: 1,
            ..LibraryOptions::default()
        };
        let args = args_from_options(&options, options.search_mode, false);
        for rate in rates {
            let data = synthetic_data(length, 100.0, delay, rate, 0.4);
            let mut visibility = data.visibility.clone();
            let current_length = length.next_power_of_two() as i32;
            visibility.resize(
                current_length as usize * (data.header.fft_point as usize / 2),
                C32::default(),
            );
            let result = search::run_peak_search(
                &visibility,
                &data.header,
                current_length,
                length as i32,
                data.effective_integ_time,
                &data.obs_time,
                &data.obs_time,
                &[],
                &None,
                &args,
                data.header.number_of_sector,
                1,
                Some((delay as f32, 0.05)),
            )
            .unwrap();
            let expected_delay = delay + 64_000_000.0 * rate * 100.0 / 8_400_000_000.0;
            assert!(
                (result.analysis_results.residual_delay as f64 - expected_delay).abs() < 1.0e-3,
                "delay at segment start was not recovered: expected={expected_delay}, delay={}",
                result.analysis_results.residual_delay
            );
            assert!(
                (result.analysis_results.residual_rate as f64 - rate).abs() < 1.0e-4,
                "stale seed was not reacquired: expected={rate}, rate={}",
                result.analysis_results.residual_rate
            );
        }
    }
}
