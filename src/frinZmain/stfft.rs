use crate::args::Args;
use crate::npy_output::{write_named_real_1d_npz, NpyMeta};
use crate::output::{generate_output_names, insert_product_before_processing_suffixes};
use crate::processing::ProcessResult;
use crate::utils::mjd_cal;
use chrono::Duration;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StfftPlan {
    pub window: i32,
    pub hop: i32,
    pub windows: i32,
}

pub fn build_plan(args: &Args, total_sectors: i32) -> Result<Option<StfftPlan>, Box<dyn Error>> {
    if args.stfft == 0 {
        return Ok(None);
    }
    if args.stfft < 0 {
        return Err("--stfft HOPS must be greater than zero.".into());
    }
    if args.length <= 0 {
        return Err("--stfft requires --length WINDOW greater than zero.".into());
    }
    if args.loop_ <= 0 {
        return Err("--stfft requires --loop greater than zero.".into());
    }
    if args.cumulate != 0 {
        return Err("--stfft cannot be combined with --cumulate.".into());
    }
    let available = total_sectors.saturating_sub(args.skip.max(0));
    if available < args.length {
        return Err(format!(
            "--stfft window ({}) exceeds available sectors after --skip ({}).",
            args.length, available
        )
        .into());
    }
    let complete_windows = 1 + (available - args.length) / args.stfft;
    Ok(Some(StfftPlan {
        window: args.length,
        hop: args.stfft,
        windows: complete_windows.min(args.loop_),
    }))
}

pub fn window_start(skip: i32, loop_index: i32, plan: Option<StfftPlan>) -> i32 {
    match plan {
        Some(plan) => skip.saturating_add(loop_index.saturating_mul(plan.hop)),
        None => skip,
    }
}

pub fn read_loop_index(loop_index: i32, plan: Option<StfftPlan>) -> i32 {
    if plan.is_some() {
        0
    } else {
        loop_index
    }
}

pub fn write_output(
    input_path: &Path,
    args: &Args,
    result: &ProcessResult,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
    let Some(plan) = build_plan(args, result.header.number_of_sector)? else {
        return Ok(None);
    };
    let output_dir = input_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join("frinZ")
        .join("stfft");
    fs::create_dir_all(&output_dir)?;
    let labels = result
        .label
        .iter()
        .map(|value| value.as_str())
        .collect::<Vec<_>>();
    let mut base_filename = generate_output_names(
        &result.header,
        &result.obs_time,
        &labels,
        !args.rfi.is_empty(),
        args.frequency,
        args.bandpass.is_some(),
        plan.window,
    );
    if args.in_beam && !base_filename.ends_with("_inbeam") {
        base_filename.push_str("_inbeam");
    }
    let product = format!("hop{}s_stfft", plan.hop);
    let output_stem = insert_product_before_processing_suffixes(&base_filename, &product);
    let path = output_dir.join(format!("{output_stem}.npz"));
    let rows = [
        result.add_plot_times.len(),
        result.add_plot_amp.len(),
        result.add_plot_snr.len(),
        result.add_plot_phase.len(),
        result.add_plot_noise.len(),
        result.add_plot_res_delay.len(),
        result.add_plot_res_rate.len(),
    ]
    .into_iter()
    .min()
    .unwrap_or(0);

    let first_time = result.add_plot_times.first().copied();
    let mut window = Vec::with_capacity(rows);
    let mut start_sector_series = Vec::with_capacity(rows);
    let mut start_mjd = Vec::with_capacity(rows);
    let mut phase_mjd = Vec::with_capacity(rows);
    let mut elapsed_s = Vec::with_capacity(rows);
    let mut amp_percent = Vec::with_capacity(rows);
    let mut snr = Vec::with_capacity(rows);
    let mut phase_deg = Vec::with_capacity(rows);
    let mut noise_percent = Vec::with_capacity(rows);
    let mut res_delay_sample = Vec::with_capacity(rows);
    let mut res_rate_hz = Vec::with_capacity(rows);
    for i in 0..rows {
        let elapsed = first_time.map_or(0.0, |first| {
            result.add_plot_times[i]
                .signed_duration_since(first)
                .num_milliseconds() as f64
                / 1000.0
        });
        let start_sector = args.skip.saturating_add(i as i32 * plan.hop);
        let effective_integ_time = if plan.window > 0 {
            result.length_sec as f64 / plan.window as f64
        } else {
            1.0
        };
        let start_offset_us =
            (start_sector as f64 * effective_integ_time * 1_000_000.0).round() as i64;
        let start_time = result.obs_time + Duration::microseconds(start_offset_us);
        window.push(i as f64);
        start_sector_series.push(start_sector as f64);
        start_mjd.push(mjd_cal(start_time));
        phase_mjd.push(mjd_cal(result.add_plot_times[i]));
        elapsed_s.push(elapsed);
        amp_percent.push(result.add_plot_amp[i] as f64);
        snr.push(result.add_plot_snr[i] as f64);
        phase_deg.push(result.add_plot_phase[i] as f64);
        noise_percent.push(result.add_plot_noise[i] as f64);
        res_delay_sample.push(result.add_plot_res_delay[i] as f64);
        res_rate_hz.push(result.add_plot_res_rate[i] as f64);
    }
    write_named_real_1d_npz(
        &path,
        NpyMeta::new(
            "stfft",
            result.header.fft_point.max(0) as u32,
            result.header.number_of_sector.max(0) as u32,
        ),
        &[
            ("window", &window),
            ("start_sector", &start_sector_series),
            ("start_mjd", &start_mjd),
            ("phase_mjd", &phase_mjd),
            ("elapsed_s", &elapsed_s),
            ("amp_percent", &amp_percent),
            ("snr", &snr),
            ("phase_deg", &phase_deg),
            ("noise_percent", &noise_percent),
            ("res_delay_sample", &res_delay_sample),
            ("res_rate_hz", &res_rate_hz),
        ],
    )?;
    Ok(Some(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn args() -> Args {
        Args::parse_from(["frinZ", "--length", "60", "--loop", "100", "--stfft", "10"])
    }

    #[test]
    fn overlapping_window_count_uses_hop() {
        let plan = build_plan(&args(), 240).unwrap().unwrap();
        assert_eq!(plan.windows, 19);
        assert_eq!(window_start(0, 3, Some(plan)), 30);
        assert_eq!(read_loop_index(3, Some(plan)), 0);
    }

    #[test]
    fn loop_limits_stfft_windows() {
        let mut a = args();
        a.loop_ = 4;
        assert_eq!(build_plan(&a, 240).unwrap().unwrap().windows, 4);
    }

    #[test]
    fn stfft_requires_length() {
        let mut a = args();
        a.length = 0;
        assert!(build_plan(&a, 240).is_err());
    }
}
