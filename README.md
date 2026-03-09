<img src="./src/frinZmain/logo1.png" width=45%>  <img src="./src/frinZmain/logo2.png" width=45%>

# frinZ

Rust version of frinZ.py - A high-performance fringe-fitting tool for VLBI data analysis.  
Original Python version: https://github.com/M-AKIMOTOO/frinZ.py

## Overview

frinZ is a Rust implementation of the frinZ fringe-fitting tool for processing Very Long Baseline Interferometry (VLBI) correlation data. It provides accurate delay and rate measurements with enhanced performance compared to the original Python version.

## Features

- **Fringe fitting analysis** for VLBI correlation data (.cor files)
- **Phase reference calibration** with polynomial fitting
- **Precise search mode** with iterative refinement
- **RFI mitigation** with frequency range exclusion
- **Bandpass calibration** with binary file support
- **Visualization** with delay/rate plots and cumulative SNR plots
- **Multiple output formats** (text, binary, plots)
- **Cross-power spectrum analysis**
- **Pulsar gating analysis** with dedispersion, folding, and gating reports

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/M-AKIMOTOO/frinZ.git
cd frinZ

# Install to ~/.cargo/bin
cargo install --path . --bin frinZ
```

### Development Build

```bash
# Check only the publishable frinZ crate
cargo check -p frinZ --all-targets

# Build the frinZ binary
cargo build -p frinZ --bin frinZ --release

# Build everything in this repository
cargo build --workspace --release
```

### Local Tools In This Workspace

This repository is split into:

- `frinZ`: publishable crate and main binary
- `frinZ-tools`: local-only package for `gfrinZ`, `pulsar_gating`, `cormerge`, `corshow`, and `bandscythe`

```bash
# Check all local tools
cargo check -p frinZ-tools --all-targets

# Build all local tools
cargo build -p frinZ-tools

# Build each tool separately
cargo build -p frinZ-tools --bin gfrinZ --release
cargo build -p frinZ-tools --bin pulsar_gating --release
cargo build -p frinZ-tools --bin cormerge --release
cargo build -p frinZ-tools --bin corshow --release
cargo build -p frinZ-tools --bin bandscythe --release
```

Run examples:

```bash
cargo run -p frinZ-tools --bin gfrinZ --release -- --help
cargo run -p frinZ-tools --bin pulsar_gating --release -- --help
cargo run -p frinZ-tools --bin cormerge --release -- --help
cargo run -p frinZ-tools --bin corshow --release -- --help
cargo run -p frinZ-tools --bin bandscythe --release -- --help
```

**Note:** crates.io publish target is `frinZ` only. On Windows, antivirus software may flag the compiled binary.

## Usage

### Basic Syntax

```bash
frinZ [OPTIONS]
```

### Input Options

#### Single File Analysis
```bash
# Basic fringe fitting
frinZ --input data.cor

# With integration time and loop count
frinZ --input data.cor --length 30 --loop 5

# Skip first 10 seconds
frinZ --input data.cor --skip 10
```

#### Phase Reference Analysis
```bash
# Basic phase referencing (calibrator, target)
frinZ --phase-reference cal.cor target.cor

# With fit_spec and custom integration times
frinZ --phase-reference cal.cor target.cor 2 60 120 3
# Arguments: calibrator target fit_spec cal_length target_length loop
# fit_spec: <deg>, sin, <deg>+sin, <deg>+sin:<period_sec>
# With polynomial degree and custom integration times
frinZ --phase-reference cal.cor target.cor 2 60 120 3
# Arguments: calibrator target fit_degree cal_length target_length loop
```

### Analysis Options

#### Frequency Domain Analysis
```bash
# Cross-power spectrum instead of fringe
frinZ --input data.cor --frequency
```

#### Precise Search Mode
```bash
# Enable iterative search with custom iterations
frinZ --input data.cor --search --iter 5

# With search windows
frinZ --input data.cor --search --delay-window -10 10 --rate-window -0.1 0.1
```

#### Manual Corrections
```bash
# Apply delay and rate corrections
frinZ --input data.cor --delay-correct 5.2 --rate-correct -0.03
```

### RFI Mitigation

```bash
# Exclude frequency ranges (MHz)
frinZ --input data.cor --rfi "100,120" "400,500"
```

### Bandpass Calibration

```bash
# Generate bandpass table
frinZ --input cal.cor --bandpass-table

# Apply existing bandpass calibration
frinZ --input data.cor --bandpass /path/to/bandpass_table.bin
```

### Output Options

#### Text Output
```bash
# Save analysis results to text files
frinZ --input data.cor --output

# Show header information
frinZ --input data.cor --header
```

#### Plotting
```bash
# Generate fringe plots
frinZ --input data.cor --plot

# Time series plots
frinZ --input data.cor --add-plot

# Cumulative SNR plots
frinZ --input data.cor --cumulate 10
```

### Advanced Examples

#### Complete Analysis with All Options
```bash
frinZ --input data.cor \
  --length 60 --loop 10 --skip 5 \
  --search --iter 3 \
  --delay-window -20 20 --rate-window -0.05 0.05 \
  --rfi "150,200" \
  --plot --add-plot --output \
  --bandpass cal_bandpass.bin
```

#### Phase Reference with Custom Parameters
```bash
frinZ --phase-reference cal.cor target.cor 1 30 60 5 \
  --search --plot --output
```

### Pulsar Gating Analysis

```bash
# Known pulsar mode (period is given)
pulsar_gating --input data.cor \
  --period 0.253 --dm 26.7 \
  --bins 128 --on-duty 0.12

# Unknown pulsar mode (period/DM estimated from data)
pulsar_gating --input data.cor --bins 128 --amp-threshold 0.015
```

Build or run `pulsar_gating` from the local-only tools package:

```bash
cargo build -p frinZ-tools --bin pulsar_gating --release
# or
cargo run -p frinZ-tools --bin pulsar_gating --release -- --help
```

`pulsar_gating` creates outputs under `frinZ/pulsar_gating/` next to the input `.cor`.

#### Modes

- **Known mode** (`--period` required, `--dm` optional): Performs dedispersion (if DM is given), fold, on/off pulse bin selection, and gated spectrum/profile products.
- **Unknown mode** (`--period` omitted): Estimates period from fringe-derived products, estimates DM from sub-band delay fit, writes handoff parameters, then automatically runs known mode with estimated values.

#### Core algorithm flow

1. Read `.cor` sectors and build channel-wise time series.
2. Build fringe products (`rate spectrum`, `delay-rate` plane).
3. Estimate period from spacing of periodic peaks in the rate spectrum (`rate-diff`).
4. Refine period by fold-SNR scan.
5. Estimate DM by fitting delay vs `1/f^2` from phase-shifted sub-band folded profiles.
6. Run known-mode gating with selected/refined parameters.

Estimated/refined period and estimated DM are printed to stdout.

#### Noise evaluation before/after gating

`pulsar_gating` evaluates noise in two stages.

1. **Before gating (folded profile)**
   - On-pulse bins are chosen by `--on-duty` (largest folded amplitudes).
   - Off-pulse bins are the remaining bins.
   - `off_mean` and `off_sigma` are computed from off-pulse folded amplitudes.
   - `Estimated S/N` is:
     - `(peak_amp - off_mean) / off_sigma`

2. **After gating (on/off weighted aggregation)**
   - Time-domain means are computed from dedispersed sector amplitudes:
     - `on_mean`: weighted mean over on-pulse sectors
     - `off_mean`: weighted mean over off-pulse sectors
     - `off_sigma`: standard deviation of off-pulse sector amplitudes
   - `Gated time S/N` is:
     - `(on_mean - off_mean) / off_sigma`
   - `Gated profile S/N` is computed from channel-subtracted time series (`on-off`) as:
     - `peak(on-off) / sigma(off on-off)`

In stdout and `*_summary.txt`, these appear as:
- `Estimated S/N` (pre-gating folded profile)
- `Gated on-mean`, `Gated off-mean`, `Gated off σ`
- `Gated time S/N`
- `Gated profile S/N`, `Gated profile σ`

#### ゲーティング前後のノイズ評価（日本語）

`pulsar_gating` では、ノイズ評価を次の2段階で行います。

1. **ゲーティング前（folded profile）**
   - `--on-duty` で指定した割合だけ、振幅の大きい位相ビンを on-pulse として選択します。
   - 残りの位相ビンを off-pulse とします。
   - off-pulse の振幅から `off_mean` と `off_sigma` を計算します。
   - `Estimated S/N` は次式です。
     - `(peak_amp - off_mean) / off_sigma`

2. **ゲーティング後（on/off 重み付き集約）**
   - dedispersed したセクター振幅から次を計算します。
     - `on_mean`: on-pulse セクターの重み付き平均
     - `off_mean`: off-pulse セクターの重み付き平均
     - `off_sigma`: off-pulse セクター振幅の標準偏差
   - `Gated time S/N` は次式です。
     - `(on_mean - off_mean) / off_sigma`
   - `Gated profile S/N` は、チャネルごとの off 平均を引いた on-off 時系列から計算し、次式で定義します。
     - `peak(on-off) / sigma(off on-off)`

`stdout` と `*_summary.txt` では、主に以下の項目として表示されます。
- `Estimated S/N`（ゲーティング前 folded profile）
- `Gated on-mean`, `Gated off-mean`, `Gated off σ`
- `Gated time S/N`
- `Gated profile S/N`, `Gated profile σ`

#### Main outputs (current default)

- `*_rate_spectrum.png` – rate profile with threshold/periodic markers.
- `*_rate_spectrum_above_amp.csv` – points above `--amp-threshold`.
- `*_rate_spectrum_periodic_peaks.csv` – periodic peak candidates used for period spacing.
- `*_delay_rate_peakscan.png` – delay-window peak scan map.
- `*_rate_diff_folded_profile.png` – folded profile from rate-diff period (when available).
- `*_dm_fit_points.csv` – DM fit points and residuals (when DM estimation succeeds).
- `*_unknown_handoff.txt` – estimated `period`, `dm`, and reproducible command.
- `*_profile.csv`, `*_folded_profile.png` – fold result from known-mode stage.
- `*_gated_spectrum_difference.csv`, `*_gated_spectrum.png`
- `*_gated_profile.csv`, `*_gated_profile.png`
- `*_onoff_pulse_bins.txt`, `*_summary.txt`
- `*_dedispersed_time_series.csv`, `*_dedispersed_time_series.png`
- `*_raw_phase_heatmap.png`, `*_phase_aligned_heatmap.png`, `*_phase_aligned_onminusoff_heatmap.png`
- `*_gated_spectrum_on.csv`, `*_gated_spectrum_off.csv`
- `*_gated_time_series.csv`, `*_gated_time_series.png`
- `*_gated_time_series_diff.csv`, `*_gated_time_series_diff.png`

#### Notes

- Recent versions intentionally reduce redundant CSV/PNG generation to shorten runtime and reduce disk usage.
- Legacy files from older naming/output schemes are cleaned up automatically when running `pulsar_gating`.

## Output Files

frinZ creates organized output directories:

```
frinZ/
├── fringe_graph/          # Delay/rate plots
│   ├── time_domain/
│   └── freq_domain/
├── fringe_output/         # Text analysis results
├── add_plot/             # Time series plots
├── cumulate/             # Cumulative SNR plots
├── bandpass_table/       # Bandpass calibration files
├── phase_reference/      # Phase reference outputs
└── cor_header/           # Header information
```

### Output File Formats

- **Text files (`.txt`)**: Analysis results with delay, rate, SNR, and statistics
- **Binary files (`.bin`, `.cor`)**: Complex spectra and calibrated data
- **Plot files (`.png`)**: Visualization of fringe patterns and time series

## File Naming Convention

Output files follow the pattern:
```
{station1}_{station2}_{timestamp}_{source}_{band}_len{length}s[_rfi][_bp]
```

Example: `YAMAGU32_YAMAGU34_2025001120000_3C84_x_len60s_rfi`

## Command Reference

### Required Arguments (one of)
- `--input <FILE>`: Single .cor file for analysis
- `--phase-reference <CAL> <TARGET> [OPTIONS]`: Phase referencing mode

### Time Parameters
- `--length <SECONDS>`: Integration time (default: entire file)
- `--skip <SECONDS>`: Skip time from start (default: 0)
- `--loop <COUNT>`: Number of processing loops (default: 1)
- `--cumulate <SECONDS>`: Cumulative integration length

### Search Parameters
- `--search`: Enable precise search mode
- `--iter <COUNT>`: Search iterations (default: 3)
- `--delay-window <MIN MAX>`: Delay search range (samples)
- `--rate-window <MIN MAX>`: Rate search range (Hz)
- `--delay-correct <VALUE>`: Manual delay correction (samples)
- `--rate-correct <VALUE>`: Manual rate correction (Hz)

### Analysis Options
- `--frequency`: Frequency domain analysis
- `--rfi <"MIN,MAX">`: RFI frequency ranges to exclude (MHz)
- `--bandpass <FILE>`: Apply bandpass calibration
- `--bandpass-table`: Generate bandpass table

### Output Options
- `--output`: Save text results
- `--header`: Show header information
- `--plot`: Generate fringe plots
- `--add-plot`: Generate time series plots
- `--cross-output`: Output complex visibility data
- `--dynamic-spectrum`: Generate dynamic spectrum plots

## Performance Notes

frinZ provides significant performance improvements over the Python version:
- **Faster FFT processing** using rustfft
- **Optimized memory usage** for large datasets
- **Parallel processing** capabilities
- **Accuracy comparable** to original (within 0.1%)

The minor numerical differences (≤0.1%) compared to frinZ.py arise from:
- Different FFT library implementations (rustfft vs scipy.fft)
- Precision differences in binary decoding (Rust: 7-8 digits, Python: 6 digits)
- DC component handling in FFT processing

## License

This program is licensed under the MIT License.

## Author

(c) M.AKIMOTO with Gemini in 2025/08/04

## Related Projects

- [frinZ.py](https://github.com/M-AKIMOTOO/frinZ.py) - Original Python implementation
