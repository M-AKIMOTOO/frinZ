use clap::{ArgAction, Command, CommandFactory, Parser};
use std::error::Error;
use std::io::{self, Cursor, Write};
use std::path::{Path, PathBuf};

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
    ("drange", &["delay-w", "delay-win"]),
    ("rrange", &["rate-w", "rate-win"]),
    ("in_beam", &["inbeam", "in-beam-vlbi"]),
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
    /// Path to the input .cor file
    #[arg(long)]
    pub input: Option<PathBuf>,

    /// Phase referencing: CAL TARGET [FIT_SPEC CAL_LEN TGT_LEN LOOP]
    #[arg(long, num_args = 2..=6, value_names = ["CALIBRATOR", "TARGET", "FIT_SPEC", "CAL_LENGTH", "TGT_LENGTH", "LOOP"])]
    pub phase_reference: Vec<String>,

    /// Compute closure phase from three baselines. Provide: FILE1 FILE2 FILE3 [refant:NAME].
    #[arg(long = "closure-phase", num_args = 0.., value_name = "FILE|KEY:VALUE")]
    pub closure_phase: Option<Vec<String>>,

    /// Integration length in sectors (0 = whole file).
    #[arg(long, default_value_t = 0)]
    pub length: i32,

    /// Skip time in seconds from the start.
    #[arg(long, default_value_t = 0)]
    pub skip: i32,

    /// Number of loops.
    #[arg(long, default_value_t = 1)]
    pub loop_: i32,

    /// RFI ranges to exclude (e.g., "100,120"). Repeatable.
    #[arg(long, num_args = 1.., value_name = "MIN,MAX")]
    pub rfi: Vec<String>,

    /// Generate plots.
    #[arg(long)]
    pub plot: bool,

    /// Use frequency-domain mode.
    #[arg(long)]
    pub frequency: bool,

    /// Output raw complex visibility to binary.
    #[arg(long)]
    pub cor2bin: bool,

    /// Output cross spectrum to binary.
    #[arg(long)]
    pub spectrum: bool,

    /// Output analysis results to .txt.
    #[arg(long)]
    pub output: bool,

    /// Delay correction value.
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    pub delay_correct: f32,

    /// Rate correction value.
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    pub rate_correct: f32,

    /// Acceleration correction value.
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    pub acel_correct: f32,

    /// Apply scan table corrections (CSV: start, integ, delay[samp], rate[Hz]).
    #[arg(long, value_name = "FILE")]
    pub scan_correct: Option<PathBuf>,

    /// Delay range (min max).
    #[arg(
        long = "drange",
        num_args = 2,
        value_names = ["MIN", "MAX"],
        allow_negative_numbers = true
    )]
    pub drange: Vec<f32>,

    /// Rate range (min max).
    #[arg(
        long = "rrange",
        num_args = 2,
        value_names = ["MIN", "MAX"],
        allow_negative_numbers = true
    )]
    pub rrange: Vec<f32>,

    /// Mask rectangle in time-domain delay/rate plane: DELAY_MIN DELAY_MAX RATE_MIN RATE_MAX.
    #[arg(
        long,
        num_args = 4,
        value_names = ["DELAY_MIN", "DELAY_MAX", "RATE_MIN", "RATE_MAX"],
        allow_negative_numbers = true
    )]
    pub mask: Vec<f32>,

    /// Frequency range for --frequency plots/search.
    #[arg(long = "frange", num_args = 2, value_names = ["MIN", "MAX"])]
    pub frange: Vec<f32>,

    /// Rate padding factor (1/2/4/8). --search peak/deep/deep2 forces 8.
    #[arg(long, default_value_t = 1)]
    pub rate_padding: u32,

    /// Cumulate length in seconds (0=off).
    #[arg(long, default_value_t = 0)]
    pub cumulate: i32,

    /// Extra plots of amp/SNR/phase/noise vs time.
    #[arg(long)]
    pub add_plot: bool,

    /// Run WWZ on per-loop fringe-search results.
    #[arg(long)]
    pub wwz: bool,

    /// Output header info.
    #[arg(long)]
    pub header: bool,

    /// Rebin FFT channels to this point count.
    #[arg(long, value_name = "POINTS")]
    pub fft_rebin: Option<i32>,

    /// Search mode: peak (default), deep, deep2, rate, or acel.
    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "peak",
        value_name = "MODE",
        value_parser = ["peak", "deep", "deep2", "rate", "acel"],
        action = ArgAction::Append
    )]
    pub search: Vec<String>,

    /// In-beam VLBI mode (standard delay-rate fringe search workflow).
    #[arg(long = "in-beam")]
    pub in_beam: bool,

    /// Iterations for --search=peak/deep/deep2 (deep/deep2 default=4 when omitted).
    #[arg(long, default_value_t = 5)]
    pub iter: u32,

    /// Plot dynamic spectrum.
    #[arg(long)]
    pub dynamic_spectrum: bool,

    /// Bandpass calibration file.
    #[arg(long)]
    pub bandpass: Option<PathBuf>,

    /// Normalize cross-correlation by auto-correlation amplitudes.
    #[arg(long = "norm-acf")]
    pub norm_acf: bool,

    /// Write bandpass-corrected spectrum to binary.
    #[arg(long)]
    pub bandpass_table: bool,

    /// CPU cores for --search deep/deep2 (0 = auto).
    #[arg(long, default_value_t = 0)]
    pub cpu: u32,

    /// Flag data by time or pp ranges.
    #[arg(long, num_args = 1.., value_name = "MODE [ARGS...]")]
    pub flagging: Vec<String>,

    /// Plot Allan deviation (requires length/loop).
    #[arg(long)]
    pub allan_deviance: bool,

    /// Heatmaps of raw visibility (amp/phase).
    #[arg(long)]
    pub raw_visibility: bool,

    /// UV coverage plot (0 planar, 1 3D).
    #[arg(long, num_args = 0..=1, default_missing_value = "1")]
    pub uv: Option<i32>,

    #[arg(long, num_args = 0.., value_name = "KEY[:VALUE]")]
    pub fringe_rate_map: Option<Vec<String>>,

    /// Maser analysis (off:<path> / off:linear / off:quad; see --detail).
    #[arg(long, num_args = 1.., value_name = "KEY:VALUE")]
    pub maser: Vec<String>,

    /// Visibility-domain pulse folding (period/bins/on-duty).
    #[arg(long, num_args = 1.., value_name = "KEY:VALUE")]
    pub folding: Vec<String>,

    /// Multi-sideband inputs (see --detail).
    #[arg(long, num_args = 6, value_names = ["C_COR", "C_BP", "C_DELAY", "X_COR", "X_BP", "X_DELAY"], allow_negative_numbers = true)]
    pub multi_sideband: Vec<String>,

    /// Plot antenna uptime (Az/El).
    #[arg(long)]
    pub uptimeplot: bool,

    /// Earth-rotation imaging (see --detail).
    #[arg(long, num_args = 0.., value_name = "KEY[:VALUE]", requires = "input")]
    pub imaging: Option<Vec<String>>,

    /// Run imaging test.
    #[arg(long)]
    pub imaging_test: bool,

    /// Show detailed CLI guide and exit.
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
            .find(|mode| *mode == "peak" || *mode == "deep" || *mode == "deep2")
            .map(|s| s.as_str())
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
