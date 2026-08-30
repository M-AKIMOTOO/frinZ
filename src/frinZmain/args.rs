use clap::{ArgAction, Command, CommandFactory, Parser};
use std::error::Error;
use std::io::{self, Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::rfi::RfiMask;

use crate::header::parse_header;
use crate::input_support::read_input_prefix;

#[derive(Clone, Copy)]
struct PrefixAliasSpec {
    arg_id: &'static str,
    base: &'static str,
    min_len: usize,
}

const PREFIX_ALIASES: &[PrefixAliasSpec] = &[
    PrefixAliasSpec {
        arg_id: "input",
        base: "input",
        min_len: 2,
    },
    PrefixAliasSpec {
        arg_id: "phase_reference",
        base: "phase",
        min_len: 2,
    },
    PrefixAliasSpec {
        arg_id: "phase_reference",
        base: "phase-reference",
        min_len: "phase-r".len(),
    },
    PrefixAliasSpec {
        arg_id: "length",
        base: "length",
        min_len: 2,
    },
    PrefixAliasSpec {
        arg_id: "skip",
        base: "skip",
        min_len: 2,
    },
    PrefixAliasSpec {
        arg_id: "loop_",
        base: "loop",
        min_len: 2,
    },
    PrefixAliasSpec {
        arg_id: "stfft",
        base: "stfft",
        min_len: 3,
    },
    PrefixAliasSpec {
        arg_id: "plot",
        base: "plot",
        min_len: 2,
    },
    PrefixAliasSpec {
        arg_id: "frequency",
        base: "frequency",
        min_len: 3,
    },
    PrefixAliasSpec {
        arg_id: "cor2bin",
        base: "cor2bin",
        min_len: "cor2b".len(),
    },
    PrefixAliasSpec {
        arg_id: "spectrum",
        base: "spectrum",
        min_len: 4,
    },
    PrefixAliasSpec {
        arg_id: "output",
        base: "output",
        min_len: 2,
    },
    PrefixAliasSpec {
        arg_id: "rate_padding",
        base: "rate-padding",
        min_len: "rate-p".len(),
    },
    PrefixAliasSpec {
        arg_id: "cumulate",
        base: "cumulate",
        min_len: 2,
    },
    PrefixAliasSpec {
        arg_id: "add_plot",
        base: "add-plot",
        min_len: "add".len(),
    },
    PrefixAliasSpec {
        arg_id: "raw_visibility",
        base: "raw-visibility",
        min_len: 2,
    },
];

const EXPLICIT_ALIASES: &[(&str, &[&str])] = &[
    ("closure_phase", &["cp"]),
    ("cor2bin", &["c2b"]),
    ("delay_correct", &["delay", "delay-corr"]),
    ("rate_correct", &["rate", "rate-corr"]),
    ("acel_correct", &["acel", "acel-corr"]),
    ("jerk_correct", &["jerk", "jerk-corr"]),
    ("snap_correct", &["snap", "snap-corr"]),
    ("drange", &["delay-w", "delay-win"]),
    ("rrange", &["rate-w", "rate-win"]),
    ("in_beam", &["inbeam", "in-beam-vlbi"]),
    ("contamination_subtract", &["contamisubt"]),
    ("dynamic_spectrum", &["ds", "dynamic"]),
    ("bandpass", &["bp"]),
    ("bandpass_table", &["bptable"]),
    ("flagging", &["flag"]),
    ("allan_deviance", &["allan", "allan-dev"]),
    ("fringe_rate_map", &["frmap"]),
    ("folding", &["fold"]),
    ("multi_sideband", &["msb"]),
];

fn with_aliases(mut command: Command) -> Command {
    command = command
        .next_line_help(false)
        .term_width(120)
        .max_term_width(120);
    for spec in PREFIX_ALIASES {
        command = add_prefix_aliases(command, spec);
    }
    for (arg_id, aliases) in EXPLICIT_ALIASES {
        command = command.mut_arg(*arg_id, |arg| arg.aliases(*aliases));
    }
    command
}

fn add_prefix_aliases(mut command: Command, spec: &PrefixAliasSpec) -> Command {
    for end in spec.min_len..spec.base.len() {
        if !spec.base.is_char_boundary(end) {
            continue;
        }
        let alias = &spec.base[..end];
        command = command.mut_arg(spec.arg_id, |arg| arg.alias(alias));
    }
    command
}

#[derive(Parser, Debug, Clone)]
#[command(
    name = "frinZ",
    version = env!("CARGO_PKG_VERSION"),
    author = "Masanori AKIMOTO  <masanori.akimoto.ac@gmail.com>",
    about = "fringe search for Yamaguchi Interferometer and Japanese VLBI Network",
    after_help = r#"(c) M.AKIMOTO with Gemini in 2025/08/04
github: https://github.com/M-AKIMOTOO/frinZrs
This program is licensed under the MIT License
see https://opensource.org/license/mit"#
)]
pub struct Args {
    /// Input .cor file.
    #[arg(long)]
    pub input: Option<PathBuf>,

    /// Phase referencing; see --detail.
    #[arg(long, num_args = 3..=6, value_name = "ARGS")]
    pub phase_reference: Vec<String>,

    /// Closure phase from three baselines.
    #[arg(long = "closure-phase", num_args = 3, value_name = "FILE")]
    pub closure_phase: Option<Vec<String>>,

    /// Integration length [sectors].
    #[arg(long, default_value_t = 0)]
    pub length: i32,

    /// Skip from start [s].
    #[arg(long, default_value_t = 0)]
    pub skip: i32,

    /// Loop count.
    #[arg(long, default_value_t = 1)]
    pub loop_: i32,

    /// STFFT hop [sectors].
    #[arg(long, default_value_t = 0, value_name = "HOPS")]
    pub stfft: i32,

    /// RFI ranges (MIN,MAX), histogram, or an RFI mask NPZ file.
    #[arg(long, num_args = 1.., value_name = "MIN,MAX|NPZ")]
    pub rfi: Vec<String>,

    /// Internal Rayleigh tail count parsed from `--rfi histogram count:N`.
    #[arg(skip)]
    pub rayleigh_count: u64,

    /// Internal histogram bin count parsed from `--rfi histogram bins:N`.
    #[arg(skip)]
    pub histogram_bins: usize,

    /// Loaded NPZ mask used internally; not a command-line option.
    #[arg(skip)]
    pub rfi_npz_mask: Option<Arc<RfiMask>>,

    /// Plot figures.
    #[arg(long)]
    pub plot: bool,

    /// Frequency-domain mode.
    #[arg(long)]
    pub frequency: bool,

    /// Save raw visibility BIN.
    #[arg(long)]
    pub cor2bin: bool,

    /// Write a phase-corrected .cor file.
    #[arg(long)]
    pub mkcor: bool,

    /// Save cross spectrum NPZ.
    #[arg(long)]
    pub spectrum: bool,

    /// Save text output.
    #[arg(long)]
    pub output: bool,

    /// Save NPZ sidecars.
    #[arg(long)]
    pub npz: bool,

    /// Delay correction [sample].
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    pub delay_correct: f32,

    /// Rate correction [Hz].
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    pub rate_correct: f32,

    /// Acceleration correction [Hz/s].
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    pub acel_correct: f32,

    /// Jerk correction [Hz/s^2].
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    pub jerk_correct: f32,

    /// Snap correction [Hz/s^3].
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    pub snap_correct: f32,

    /// Scan correction CSV.
    #[arg(long, value_name = "FILE")]
    pub scan_correct: Option<PathBuf>,

    /// YAMAGU34 autocorrelation file used to identify spike residual correction.
    #[arg(long = "spike34", aliases = ["spike34m", "spike34mcorr"], value_name = "YAMAGU34_AUTO.cor")]
    pub spike34m: Option<PathBuf>,

    /// Delay window.
    #[arg(
        long = "drange",
        num_args = 2,
        value_names = ["MIN", "MAX"],
        allow_negative_numbers = true
    )]
    pub drange: Vec<f32>,

    /// Rate window.
    #[arg(
        long = "rrange",
        num_args = 2,
        value_names = ["MIN", "MAX"],
        allow_negative_numbers = true
    )]
    pub rrange: Vec<f32>,

    /// Mask delay-rate rectangle.
    #[arg(
        long,
        num_args = 4,
        value_names = ["D1", "D2", "R1", "R2"],
        allow_negative_numbers = true
    )]
    pub mask: Vec<f32>,

    /// Frequency window [MHz].
    #[arg(long = "frange", num_args = 2, value_names = ["MIN", "MAX"])]
    pub frange: Vec<f32>,

    /// Rate FFT padding.
    #[arg(long, default_value_t = 1)]
    pub rate_padding: u32,

    /// Cumulate length [s].
    #[arg(long, default_value_t = 0)]
    pub cumulate: i32,

    /// Plot time series.
    #[arg(long)]
    pub add_plot: bool,

    /// WWZ time-frequency analysis.
    #[arg(long)]
    pub wwz: bool,

    /// Print header.
    #[arg(long)]
    pub header: bool,

    /// Rebin FFT channels.
    #[arg(long, value_name = "POINTS")]
    pub fft_rebin: Option<i32>,

    /// In-band width [MHz]. Without --search, use the zero-delay/rate cell.
    #[arg(long, value_name = "MHz")]
    pub inband: Option<u32>,

    /// Fringe search mode.
    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "peak",
        value_name = "MODE",
        value_parser = ["peak", "deep", "deep2", "coherent", "rate", "acel"],
        action = ArgAction::Append
    )]
    pub search: Vec<String>,

    /// In-beam VLBI workflow.
    #[arg(long = "in-beam")]
    pub in_beam: bool,

    /// Search iterations.
    #[arg(long, default_value_t = 5)]
    pub iter: u32,

    /// Dynamic spectrum.
    #[arg(long)]
    pub dynamic_spectrum: bool,

    /// Bandpass table.
    #[arg(long)]
    pub bandpass: Option<PathBuf>,

    /// Analyze contamination-removal handoff data for flux.
    #[arg(long, num_args = 0.., value_name = "KEY:VALUE")]
    pub contamination: Option<Vec<String>>,

    /// Apply a flux compact correction table in copy-on-write memory while analyzing --input.
    #[arg(long = "contamination-subtract", value_name = "MODEL_NPZ")]
    pub contamination_subtract: Option<PathBuf>,

    /// Normalize by ACF.
    #[arg(long = "norm-acf")]
    pub norm_acf: bool,

    /// Save bandpass table NPZ.
    #[arg(long)]
    pub bandpass_table: bool,

    /// CPU cores.
    #[arg(long, default_value_t = 0)]
    pub cpu: u32,

    /// Flag data.
    #[arg(long, num_args = 1.., value_name = "SPEC")]
    pub flagging: Vec<String>,

    /// Allan deviation.
    #[arg(long)]
    pub allan_deviance: bool,

    /// Raw visibility plots.
    #[arg(long)]
    pub raw_visibility: bool,

    /// UV coverage.
    #[arg(long, num_args = 0..=1, default_missing_value = "1")]
    pub uv: Option<i32>,

    #[arg(long, num_args = 0.., value_name = "KEY")]
    pub fringe_rate_map: Option<Vec<String>>,

    /// Maser analysis.
    #[arg(long, num_args = 1.., value_name = "KEY")]
    pub maser: Vec<String>,

    /// Pulse folding.
    #[arg(long, num_args = 1.., value_name = "KEY")]
    pub folding: Vec<String>,

    /// Multi-sideband analysis.
    #[arg(long, num_args = 6, value_name = "ARGS", allow_negative_numbers = true)]
    pub multi_sideband: Vec<String>,

    /// Antenna uptime.
    #[arg(long)]
    pub uptimeplot: bool,

    /// Earth-rotation imaging.
    #[arg(long, num_args = 0.., value_name = "KEY")]
    pub imaging: Option<Vec<String>>,

    /// Detailed help.
    #[arg(long)]
    pub detail: bool,
}

impl Args {
    pub fn command_with_aliases() -> Command {
        with_aliases(Self::command())
    }

    pub fn primary_search_mode(&self) -> Option<&str> {
        self.search
            .iter()
            .find(|mode| {
                *mode == "peak" || *mode == "deep" || *mode == "deep2" || *mode == "coherent"
            })
            .map(|s| s.as_str())
    }
}

impl Default for Args {
    fn default() -> Self {
        Self {
            input: None,
            phase_reference: Vec::new(),
            closure_phase: None,
            length: 0,
            skip: 0,
            loop_: 1,
            stfft: 0,
            rfi: Vec::new(),
            rayleigh_count: 1,
            histogram_bins: 256,
            rfi_npz_mask: None,
            plot: false,
            frequency: false,
            cor2bin: false,
            mkcor: false,
            spectrum: false,
            output: false,
            npz: false,
            delay_correct: 0.0,
            rate_correct: 0.0,
            acel_correct: 0.0,
            jerk_correct: 0.0,
            snap_correct: 0.0,
            scan_correct: None,
            drange: Vec::new(),
            rrange: Vec::new(),
            mask: Vec::new(),
            frange: Vec::new(),
            rate_padding: 1,
            cumulate: 0,
            add_plot: false,
            wwz: false,
            header: false,
            fft_rebin: None,
            inband: None,
            search: Vec::new(),
            in_beam: false,
            iter: 5,
            dynamic_spectrum: false,
            bandpass: None,
            contamination: None,
            contamination_subtract: None,
            spike34m: None,
            norm_acf: false,
            bandpass_table: false,
            cpu: 0,
            flagging: Vec::new(),
            allan_deviance: false,
            raw_visibility: false,
            uv: None,
            fringe_rate_map: None,
            maser: Vec::new(),
            folding: Vec::new(),
            multi_sideband: Vec::new(),
            uptimeplot: false,
            imaging: None,
            detail: false,
        }
    }
}

pub fn check_memory_usage(args: &Args, input_path: &Path) -> Result<bool, Box<dyn Error>> {
    let buffer = read_input_prefix(input_path, 256)?;
    let mut cursor = Cursor::new(buffer.as_slice());
    let header = parse_header(&mut cursor)?;

    let fft_point = header.fft_point as u64;
    let pp = header.number_of_sector as u64;
    let rate_padding = args.rate_padding as u64;

    let required_memory = 4 * fft_point * pp.next_power_of_two() * rate_padding; // byte

    let mem_info = sys_info::mem_info()?;
    let total_ram = mem_info.total * 1024; // Convert KB to Bytes
    let quarter_ram = total_ram / 4;

    if required_memory > quarter_ram {
        println!(
            "Warning: The estimated memory usage ({:.2} GB) exceeds 25% of your system RAM ({:.2} GB).",
            required_memory as f64 / 1_073_741_824.0,
            total_ram as f64 / 1_073_741_824.0
        );
        print!("Do you want to continue? (y/n): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if input.trim().to_lowercase() != "y" {
            println!("Aborting.");
            return Ok(false);
        }
    }

    Ok(true)
}
