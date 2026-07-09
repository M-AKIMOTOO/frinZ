use crate::args::Args;
use crate::processing::ProcessResult;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
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
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("frinZ");
    let path = output_dir.join(format!(
        "{}_len{}s_hop{}s_stfft.tsv",
        stem, plan.window, plan.hop
    ));
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

    let mut out = BufWriter::new(File::create(&path)?);
    writeln!(
        out,
        "# STFFT: window={} sectors, hop={} sectors, requested_loop={}, rows={}",
        plan.window, plan.hop, args.loop_, rows
    )?;
    writeln!(
        out,
        "# Window\tStartSector\tEpoch\tElapsed[s]\tAmp[%]\tSNR\tPhase[deg]\tNoise[%]\tResDelay[sample]\tResRate[Hz]"
    )?;
    let first_time = result.add_plot_times.first().copied();
    for i in 0..rows {
        let elapsed = first_time.map_or(0.0, |first| {
            result.add_plot_times[i]
                .signed_duration_since(first)
                .num_milliseconds() as f64
                / 1000.0
        });
        writeln!(
            out,
            "{}\t{}\t{}\t{:.6}\t{:.9}\t{:.3}\t{:+.6}\t{:.9}\t{:+.9}\t{:+.12}",
            i,
            args.skip.saturating_add(i as i32 * plan.hop),
            result.add_plot_times[i].format("%Y/%jT%H:%M:%S%.3f"),
            elapsed,
            result.add_plot_amp[i],
            result.add_plot_snr[i],
            result.add_plot_phase[i],
            result.add_plot_noise[i],
            result.add_plot_res_delay[i],
            result.add_plot_res_rate[i],
        )?;
    }
    out.flush()?;
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
