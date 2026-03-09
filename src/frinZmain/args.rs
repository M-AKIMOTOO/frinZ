use clap::{ArgAction, Parser};
use std::error::Error;
use std::fs::File;
use std::io::{self, Cursor, Read, Write};
use std::path::{Path, PathBuf};

use crate::header::parse_header;

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
    #[arg(long, aliases = ["in", "inp", "inpu"])]
    pub input: Option<PathBuf>,

    /// Phase referencing: CAL TARGET [FIT_SPEC CAL_LEN TGT_LEN LOOP]
    #[arg(long, num_args = 2..=6, value_names = ["CALIBRATOR", "TARGET", "FIT_SPEC", "CAL_LENGTH", "TGT_LENGTH", "LOOP"], aliases = ["ph", "pha", "phas", "phase","phase-r", "phase-re","phase-ref","phase-refe","phase-refer","phase-refere","phase-referen","phase-referenc"])]
    pub phase_reference: Vec<String>,

    /// Compute closure phase from three baselines. Provide: FILE1 FILE2 FILE3 [refant:NAME].
    #[arg(long = "closure-phase", aliases = ["cp"], num_args = 0.., value_name = "FILE|KEY:VALUE")]
    pub closure_phase: Option<Vec<String>>,

    /// Integration time in seconds (0 = whole file).
    #[arg(long, aliases = ["le", "len", "leng", "lengt"], default_value_t = 0)]
    pub length: i32,

    /// Skip time in seconds from the start.
    #[arg(long, aliases = ["sk", "ski"], default_value_t = 0)]
    pub skip: i32,

    /// Number of loops.
    #[arg(long, aliases = ["lo", "loo"], default_value_t = 1)]
    pub loop_: i32,

    /// RFI ranges to exclude (e.g., "100,120"). Repeatable.
    #[arg(long, num_args = 1.., value_name = "MIN,MAX")]
    pub rfi: Vec<String>,

    /// Generate plots.
    #[arg(long, aliases = ["pl", "plo"])]
    pub plot: bool,

    /// Use frequency-domain mode.
    #[arg(long, aliases = ["fre", "freq", "frequ", "freque", "frequen", "frequenc"])]
    pub frequency: bool,

    /// Output raw complex visibility to binary.
    #[arg(long, aliases = ["c2b", "cor2b", "cor2bi"])]
    pub cor2bin: bool,

    /// Output cross spectrum to binary.
    #[arg(long, aliases = ["spec", "spect", "spectr", "spectru"])]
    pub spectrum: bool,

    /// Output analysis results to .txt.
    #[arg(long, aliases = ["ou", "out", "outp", "outpu"])]
    pub output: bool,

    /// Delay correction value.
    #[arg(long, aliases = ["delay","delay-corr"], default_value_t = 0.0, allow_negative_numbers = true)]
    pub delay_correct: f32,

    /// Rate correction value.
    #[arg(long, aliases = ["rate","rate-corr"], default_value_t = 0.0, allow_negative_numbers = true)]
    pub rate_correct: f32,

    /// Acceleration correction value.
    #[arg(long, aliases = ["acel","acel-corr"], default_value_t = 0.0, allow_negative_numbers = true)]
    pub acel_correct: f32,

    /// Apply scan table corrections (CSV: start, integ, delay[samp], rate[Hz]).
    #[arg(long, value_name = "FILE")]
    pub scan_correct: Option<PathBuf>,

    /// Delay range (min max).
    #[arg(long = "drange", aliases = ["delay-w", "delay-win"], num_args = 2, value_name = "MIN MAX", allow_negative_numbers = true)]
    pub drange: Vec<f32>,

    /// Rate range (min max).
    #[arg(long = "rrange", aliases = ["rate-w", "rate-win"], num_args = 2, value_name = "MIN MAX", allow_negative_numbers = true)]
    pub rrange: Vec<f32>,

    /// Frequency range for --frequency plots/search.
    #[arg(long = "frange", num_args = 2, value_name = "MIN MAX")]
    pub frange: Vec<f32>,

    /// Rate padding factor (1/2/4/8/16). Deep defaults to 4.
    #[arg(long, aliases = ["rate-p", "rate-pa", "rate-pad", "rate-padd", "rate-paddi", "rate-paddin"], default_value_t = 1)]
    pub rate_padding: u32,

    /// Cumulate length in seconds (0=off).
    #[arg(long, aliases = ["cu", "cum", "cumu", "cumul", "cumula", "cumulat"], default_value_t = 0)]
    pub cumulate: i32,

    /// Extra plots of amp/SNR/phase/noise vs time.
    #[arg(long, aliases = ["add", "add-p", "add-pl", "add-plo"])]
    pub add_plot: bool,

    /// Output header info.
    #[arg(long)]
    pub header: bool,

    /// Rebin FFT channels to this point count.
    #[arg(long, value_name = "POINTS")]
    pub fft_rebin: Option<i32>,

    /// Search mode: peak (default), deep, rate, or acel.
    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "peak",
        value_name = "MODE",
        value_parser = ["peak", "deep", "rate", "acel"],
        action = ArgAction::Append
    )]
    pub search: Vec<String>,

    /// In-beam VLBI mode (standard delay-rate fringe search workflow).
    #[arg(long = "in-beam", aliases = ["inbeam", "in-beam-vlbi"])]
    pub in_beam: bool,

    /// Iterations for --search=peak/deep (deep default=4 when omitted).
    #[arg(long, default_value_t = 5)]
    pub iter: u32,

    /// Plot dynamic spectrum.
    #[arg(long, aliases = ["ds","dynamic"])]
    pub dynamic_spectrum: bool,

    /// Bandpass calibration file.
    #[arg(long, aliases = ["bp"])]
    pub bandpass: Option<PathBuf>,

    /// Write bandpass-corrected spectrum to binary.
    #[arg(long, aliases = ["bptable"])]
    pub bandpass_table: bool,

    /// CPU cores for --search deep (0 = auto).
    #[arg(long, default_value_t = 0)]
    pub cpu: u32,

    /// Flag data by time or pp ranges.
    #[arg(long, num_args = 1.., value_name = "MODE [ARGS...]", aliases = ["flag"])]
    pub flagging: Vec<String>,

    /// Plot Allan deviation (requires length/loop).
    #[arg(long, aliases = ["allan","allan-dev"])]
    pub allan_deviance: bool,

    /// Heatmaps of raw visibility (amp/phase).
    #[arg(long, aliases = ["ra","raw","raw-v","raw-vi","raw-vis","raw-visi","raw-visib","raw-visibi","raw-visibils","raw-visibili","raw-visibilit"])]
    pub raw_visibility: bool,

    /// UV coverage plot (0 planar, 1 3D).
    #[arg(long, num_args = 0..=1, default_missing_value = "1")]
    pub uv: Option<i32>,

    #[arg(long, aliases = ["frmap"], num_args = 0.., value_name = "KEY[:VALUE]")]
    pub fringe_rate_map: Option<Vec<String>>,

    /// Maser analysis (off:<path> / off:linear / off:quad; see --detail).
    #[arg(long, num_args = 1.., value_name = "KEY:VALUE")]
    pub maser: Vec<String>,

    /// Visibility-domain pulse folding (period/bins/on-duty).
    #[arg(long, aliases = ["fold"], num_args = 1.., value_name = "KEY:VALUE")]
    pub folding: Vec<String>,

    /// Multi-sideband inputs (see --detail).
    #[arg(long, num_args = 6, value_names = ["C_COR", "C_BP", "C_DELAY", "X_COR", "X_BP", "X_DELAY"], aliases = ["msb"], allow_negative_numbers = true)]
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
    pub fn primary_search_mode(&self) -> Option<&str> {
        self.search
            .iter()
            .find(|mode| *mode == "peak" || *mode == "deep")
            .map(|s| s.as_str())
    }
}

pub fn check_memory_usage(args: &Args, input_path: &Path) -> Result<bool, Box<dyn Error>> {
    let mut file = File::open(input_path)?;
    let mut buffer = vec![0; 256];
    file.read_exact(&mut buffer)?;
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
