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
- **Bandpass calibration** with compressed NPZ support
- **Visualization** with delay/rate plots and cumulative SNR plots
- **Multiple output formats** (text, compressed NPZ, plots)
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
- `frinZ-tools`: local-only package for `gfrinZ`, `pulsar_gating`, `cormerge`, and `bandscythe`

```bash
# Check all local tools
cargo check -p frinZ-tools --all-targets

# Build all local tools
cargo build -p frinZ-tools

# Build each tool separately
cargo build -p frinZ-tools --bin gfrinZ --release
cargo build -p frinZ-tools --bin pulsar_gating --release
cargo build -p frinZ-tools --bin cormerge --release
cargo build -p frinZ-tools --bin bandscythe --release
```

Run examples:

```bash
cargo run -p frinZ-tools --bin gfrinZ --release -- --help
cargo run -p frinZ-tools --bin pulsar_gating --release -- --help
cargo run -p frinZ-tools --bin cormerge --release -- --help
cargo run -p frinZ-tools --bin bandscythe --release -- --help
```

**Note:** crates.io publish target is `frinZ` only. On Windows, antivirus software may flag the compiled binary.

## Library API

`frinZ` can also be used as a Rust library without creating plots or sidecar files.

```rust
use frinZ::{delay_search, frequency_spectrum, read_cor, LibraryOptions, SearchMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cor = read_cor("data.cor")?;

    let mut options = LibraryOptions::default();
    options.search_mode = Some(SearchMode::Peak);

    let fringe = delay_search(&cor, &options)?;
    println!(
        "delay = {} sample, rate = {} Hz, snr = {}",
        fringe.analysis.residual_delay,
        fringe.analysis.residual_rate,
        fringe.analysis.delay_snr
    );

    let spectrum = frequency_spectrum(&cor, &options)?;
    for (freq_mhz, value) in spectrum.frequency_mhz.iter().zip(spectrum.spectrum.iter()) {
        println!("{freq_mhz}	{}	{}", value.re, value.im);
    }

    Ok(())
}
```

Useful public entry points are:

- `read_cor`, `read_cor_with_options`, `read_cor_bytes`: read `.cor` visibility data
- `apply_delay_rate_correction`: apply delay/rate/acceleration phase correction in memory
- `fringe_search`, `delay_search`: run delay/rate fringe search
- `frequency_spectrum`: return the complex cross-power spectrum with a MHz axis
- `read_bandpass`: read frinZ NPZ or legacy BIN bandpass tables

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
# 110 m baseline: global quadratic phase model
frinZ --phase-reference poly2 cal.cor target.cor

# VLBI: quadratic trend plus locally interpolated atmospheric residual
frinZ --phase-reference hybrid cal.cor target.cor 60 120 3
# Arguments: mode calibrator target cal_length target_length loop
# Modes: poly2, linear, nearest, hybrid
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

# Short-time fringe FFT: 60-sector window, 10-sector hop, up to 100 windows
frinZ --input data.cor --search --length 60 --stfft 10 --loop 100

# With search windows
frinZ --input data.cor --search --delay-window -10 10 --rate-window -0.1 0.1
```

#### In-band Fringe Search
```bash
# Split 512 MHz bandwidth into 128 MHz subbands and read the zero-delay/rate cells
frinZ --input data.cor --inband 128
```

`--inband` takes a power-of-two width in MHz and writes `frinZ/inband/*_inband<width>MHz.txt`, for example `*_inband256MHz.txt`. Without `--search`, no delay/rate search is performed and the zero-delay/zero-rate cell is reported. Add `--search peak` or `--search deep` to search each in-band subband.


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
frinZ --input data.cor --bandpass /path/to/bandpass_table.npz
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
  --bandpass cal_bandpass.npz
```

#### Phase Reference with Custom Parameters
```bash
frinZ --phase-reference linear cal.cor target.cor 30 60 5 \
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

## 出力ファイル名の補正接尾辞

frinZ の解析結果はすべて元 `.cor` の親ディレクトリ直下の `frinZ/` にまとめます。bandpass、RFI、contamination subtractionなどの処理ごとに親直下へ別ディレクトリは作りません。

frinZ の通常解析と派生解析は、基本stemの解析product名の後に補正接尾辞を付けます。接尾辞の順序は常に `bp` → `rfi` → `contamisubt` → `spike34`（in-beam解析では最後に `inbeam`）です。補正の組合せが変わってもproductまでの共通部分をワイルドカードで選択できます。

- 補正なし: `..._len60s_delay_rate_search.txt`
- bandpass: `..._len60s_delay_rate_search_bp.txt`
- bandpass + RFI: `..._len60s_delay_rate_search_bp_rfi.txt`
- bandpass + RFI + contamination subtraction + spike34: `..._len60s_delay_rate_search_bp_rfi_contamisubt_spike34.txt`
- cumulate: `..._len60s_SIGMAGEM_cumulate60_bp_rfi_contamisubt_spike34.png`

この規則はTXT/TSV/PNG/NPZを含む全通常出力に共通で、`delay_rate_search`、`freq_rate_search`、`spectrum`、`bptable`、dynamic spectrum、add plot、cumulate、WWZ、STFFT、Allan deviation、in-band、header、contamination handoff、およびNPZ sidecarへ適用されます。例えば `*_delay_rate_search*.txt` で補正の有無を問わず同じ解析productを、`*_delay_rate_search_bp_rfi.txt` でbandpass+RFIだけを選択できます。

## YAMAGU34 spike correction diagnostics (`--spike34`)

`--spike34 <YAMAGU34_AUTO.cor>`（旧名 `--spike34m` / `--spike34mcorr` も使用可能）は、YAMAGU34 自己相関に現れる周波数 spike を基準として、入力相互相関を次の順で補正します。まず全帯域 `--search` の delay/rate を適用し、全帯域 phase0 の直線 fit と raw residual を求め、その residual の spike 間線形成分から interval delay/rate を推定して補正します。揺らぎ成分は最終 residual として残します。指定した自己相関ファイルはヘッダー上 `ant1=YAMAGU34` かつ `ant2=YAMAGU34` であること、ファイル名内の `yyyydddhhmmss` が `--input` と一致すること、さらに `--input` の ant1 または ant2 に YAMAGU34 を含むことを要求します。

出力は `--input` の親ディレクトリ直下 `frinZ/spike34/` に保存します。`*_spike34_spikes.tsv` には検出した spike の channel、周波数、自己相関強度、周辺平均との差、SNR を、`*_spike34_delay_rate.tsv` には列ヘッダー付きで低周波端・spike 間・高周波端の各サブバンドの12列を出します。さらに `*_spike34_spectrum_before_after.tsv` に周波数ごとの Raw / 全帯域 `--search` 補正 / spike34 補正の amplitude・phase と、fit/residual と共通の `fit_before_phase_deg`・`fit_after_phase_deg`、search delay/rate を保存します。`*_spike34_spectrum_before_after.png` の位相2系列は `fit_residual.png` の入力 phase0 とトレンド保持型 spike34 位相をそのまま描き、検出 spike 周波数を黒破線で示します。`*_spike34_fit_residual.tsv` と `*_spike34_fit_residual.png` は、全帯域補正後の入力 phase0、各 spike 区間の周波数方向線形 fit、`入力 - interval fit` の raw/final residual、rate residual を同じ spike 位置付きで記録・表示します。補正本体は全帯域補正後の各区間 residual fit（周波数方向の位相オフセットと傾き）と区間レートを、全時間・全周波数セルへ直接適用します。区間平均位相を元へ戻す処理は行わず、spike 境界の位相ジャンプを除去しながら全帯域の検索トレンドを保持します。fit から外れた揺らぎは置換せず、最終 residual として残します。比較用に、spike channel へ白線を重ねた `*_spike34_rawvis_amp.png` と `*_spike34_rawvis_phase.png`、全帯域のフリンジ解を基準に spike 間残差を補正した `*_spike34_rawvis_corrected_amp.png` と `*_spike34_rawvis_corrected_phase.png` も生成します。通常解析の出力名にも `_spike34` を付けます。

例:

```bash
frinZ --input YAMAGU34_HITACH32_yyyydddhhmmss.cor \
      --spike34 YAMAGU34_YAMAGU34_yyyydddhhmmss.cor \
      --raw-visibility
```

## Contamination handoff (`--contamination`)

`frinZ --contamination` は元 `.cor` を解析し、flux が複素コンタミモデルを推定するための `*_contamination.npz` handoff を作成します。この初回解析では天体信号を減算しません。次に `flux --contamination` がC/Xそれぞれ通常出力と同じprefixの `obscode_***_{c,x}_contamisubt_model.npz` を生成し、元 `.cor` は読みません。最後に `frinZ --input ORIGINAL.cor --contamination-subtract MODEL.npz`（別名 `--contamisubt`）が元 `.cor` と補正テーブルを同時に読み、private copy-on-writeメモリ上で減算して通常解析を続けます。元 `.cor` は維持し、`*_contamisubt.cor` は生成しません。減算解析も通常解析と同じ元 `.cor` 親の `frinZ/` 配下へ保存し、親直下に `contamisubt/` は作りません。出力名はproduct名の後に `_contamisubt` を付けて区別します。

`--raw-visibility` の出力ディレクトリは `frinZ/rawvis/`、ファイル名は `_rawvis` 接尾辞です。raw visibilityは `.cor` 読み込み直後のrow-major `[sector,channel]` で、ACF規格化、rebin、padding、bandpass補正より前の値です。`--delay` または `--rate`（両方可）を指定すると、元データ図に加えて同じ補正を全時間・全周波数セルへ適用した `*_rawvis_corrected_delay_rate_heatmap_{amp,phase}.png` を出力します。 `--npz` を付けると `*_rawvis.npz` に `freq`/`frequency_mhz`（MHz）、`time_sec`、`phase`（rad、[time,channel]）、`phase_deg`、複素 `visibility` を保存します。

fluxへ渡すglobにはtargetとgain天体の両方を含めます。fluxは前後gainの実測raw複素スペクトルを時間補間し、既知座標の幾何位相と `flux:`/gain flux比からtarget grid上のコンタミ配列を作ります。model v5はC/X別ファイルで、direct arrayを保持せず、gain複素スペクトル、各sectorの時刻・幾何遅延、周波数軸、フラックス比、窓ごとの複素規格化係数だけを小容量テーブルとして保持します。frinZ `--contamisubt` は元 `.cor` のcopy-on-writeビュー上で同じ補正量を再構成して減算します。このv5経路では別bandpass表は必須ではありません。v4以前のhandoffだけの場合はscalar逆投影へフォールバックし、flat complex bandpassならdelay面残差のWARNを出します。

`flux --contamination` の `c:`/`x:` はこの `*_contamination.npz`（複数・ワイルドカード可）、`ra:HHMMSS`/`dec:DDMMSS` は位相中心基準の J2000 絶対座標、`flux:mJy` は基準周波数のフラックス、`alpha` は `S_nu ∝ nu^alpha` です。方向余弦は `l=cos(dec)*sin(ra-ra0)`、`m=sin(dec)*cos(dec0)-cos(dec)*sin(dec0)*cos(ra-ra0)` です。

帯域 `b` の座標位相を `G_b(t)=phase_sign*2*pi*uv_sign*(u*l+v*m)` とすると、flux は全時刻を `V_i=A_i exp(i theta_target)+S_contam exp(i(G_i+theta_contam))+N_i` で複素fitします。`A_i>=0` は時刻ごとに自由で、C/X の `theta_target` と `theta_contam` は独立です。最初の観測位相への強制整列やdelay/rateの再探索は行いません。二天体fitは複素scalarで行い、補正テーブルの生成と適用には正確な生UVW・時刻・周波数グリッドを使用します。`fit:off` は外部座標を固定し、`fit:on` は同じ複素残差で位置も推定します。

複素減算は `Vobs=Vtarget+Vcontam+N`、`Vclean=Vobs-Vcontam`。`before` は減算前、`after` は複素減算後の振幅です。

```bash
# 1. target と前後gainの handoff を作成
frinZ --input target.cor --length 480 --loop 3 --contamination
frinZ --input gain_before.cor --length 480 --loop 1 --contamination
frinZ --input gain_after.cor  --length 480 --loop 1 --contamination

# 2. flux がコンタミモデルを推定（元 .cor は読まない）
flux ... --contamination ... ra:... dec:... flux:10 fit:on

# 3. 元 .cor へテーブルをメモリ内適用し、そのまま通常解析
frinZ --input ant1_ant2_yyyydddhhmmss_c.cor \
  --contamination-subtract i25314x_frinZ_c_all_SIGMAGEM_c_contamisubt_model.npz \
  --length 480 --loop 3
# 短縮別名: --contamisubt
```

同じdelay/rateセルのfrinZ複素値へ同じtarget−gain位相回転を適用すると、fluxのclean scalarと一致します。raw `.cor` はgain-reference前なので、未回転の位相角は直接比較しません。既知の位相中心天体は補正後の `(delay,rate)=(0,0)` 固定セルで評価します。全平面 `--search` の最大SNRはtrial factorを含み、SNR 5--6だけでは検出ではありません。非ゼロセルは未モデル成分、RFI、雑音最大点として別途検証します。定数位相オフセットは複素平面を一様回転するだけでdelay/rate座標を変えません。delayずれは周波数位相傾斜、rateずれは時間位相傾斜です。

## Output Files

With `--npz`, analysis/plot modes also write compressed self-describing NumPy sidecars (`*.npz`) containing a primitive `complex64` `data` array plus `flag`, coordinate axes and units, `fft_point`, `pp`, and array shape. Inspect or export one file with:

```bash
python3 tools/npz_open.py --npz result_bptable.npz
python3 tools/npz_open.py --npz result_bptable.npz --output --ext pdf --nofig
```

`--output` writes `result_bptable.tsv` and the selected figure format (default `png`). Analysis modes no longer emit duplicate BIN/TSV data files when the same arrays are present in requested NPZ sidecars; text summaries and model metadata remain separate. Without `--nofig`, the figure is also shown with `plt.show()`.

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
- **Compressed NPZ (`.npz`)**: `--spectrum`, `--bptable`, and optional `--npz` analysis/plot arrays with axes and metadata
- **Correlation files (`.cor`)**: Calibrated or combined visibility data
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
- `--phase-reference <MODE> <CAL> <TARGET> [CAL_LENGTH TGT_LENGTH LOOP]`: Phase referencing mode

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
- `--rfi histogram`: derive a Rayleigh-fit RFI mask for each current integration window; histogram products are written under `frinZ/rfi/` and can be combined with numeric ranges. The strongest connected celestial component and the fringe-peak delay column/rate row are classified as celestial and excluded from the Rayleigh fit/threshold statistics; they remain visible as a cyan cross in the position map. The rate=0 frequency spectrum row is protected from the derived mask. The PNGs show all eligible cells, RFI candidates, known-celestial cells, the Rayleigh fit, and the threshold; linear/logy/logx/logxy and delay-rate imshow variants are emitted.
- `--rfi histogram count:N`: set the Rayleigh tail count used for the histogram threshold (default `1`; larger values lower the threshold), matching the Zig `noise_hist` convention. The selected value and fitted threshold are recorded in the histogram TSV/NPZ.
- `--rfi <MASK.npz>`: load `ifft_rfi_frequency_mask`/`ifft_rfi_frequency_coordinates` (frequency-rate) and `rfi_mask` (delay-rate) from a noise-histogram NPZ; matching plane cells are set to `0+0j`.
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
- Precision differences in correlation-data decoding (Rust: 7-8 digits, Python: 6 digits)
- DC component handling in FFT processing

## License

This program is licensed under the MIT License.

## Author

(c) M.AKIMOTO with Gemini in 2025/08/04

## Related Projects

- [frinZ.py](https://github.com/M-AKIMOTOO/frinZ.py) - Original Python implementation

### Contamination handoff NPZ

`--contamination` は可読な JSON/TXT sidecar を生成せず、圧縮された `*_contamination.npz` handoff のみを書き出します。NPZ はキー付き NumPy 配列で構成され、スキャンごとの数値を効率よく保存します。Python からは次のように読み出せます。

```python
import numpy as np
z = np.load("*_contamination.npz")
u = z["uv_u"]                 # integration-start baseline U [m], shape (1,)
v = z["uv_v"]                 # integration-start baseline V [m], shape (1,)
vis = z["frinz_complex_vis"]  # exact selected fringe cell, shape (1,)
freq = z["frequency_mhz"]     # representative frequency [MHz], shape (1,)
mjd = z["mjd"]                # integration start MJD, shape (1,)
duration = z["effective_integration_time_s"]  # data length from mjd [s]
raw_mjd = z["raw_mjd"]
raw_freq = z["raw_frequency_mhz"]
raw_vis = z["raw_visibility_real"] + 1j*z["raw_visibility_imag"]
raw_vis = raw_vis.reshape(len(raw_mjd), len(raw_freq))
```

format v2/v3/v4 の `complex_vis`、`visibility_real`、`visibility_imag` も同じ1個の複素フリンジピークです。Additional arrays include start-time `uv_w`, `du_dt_m_per_s`, `dv_dt_m_per_s`, `elapsed_s=0`, representative `wavelength_m`, `peak_delay_sample`, `peak_rate_hz`, `peak_snr`, and `peak_noise`. Scalar/header values are stored as one-element arrays (`phase_center_ra_rad`, `phase_center_dec_rad`, `observing_frequency_hz`, `effective_integration_time_s`); text metadata is available as UTF-8 byte arrays (`source_name`, `input_cor`) and the complete handoff is retained in `metadata_json`. 外部 JSON ファイルは廃止され、`flux --contamination` も NPZ のみを受け付けます。
