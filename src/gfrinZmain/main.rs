use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

mod global_ls;
use global_ls::{solve_antenna_phase, solve_antenna_tau_rate, BaselineSolve};
mod example;

// no local Complex alias needed

// Reuse library modules from frinZmain
use byteorder::{ByteOrder, LittleEndian};
use frinZ::analysis::analyze_results;
use frinZ::args::Args as FrinZArgs;
use frinZ::bandpass::{apply_bandpass_correction, read_bandpass_file};
use frinZ::fft::{process_fft, process_ifft};
use frinZ::read::read_visibility_data;
use frinZ::rfi::parse_rfi_ranges;
use memmap2::Mmap;
use nalgebra::{DMatrix, DVector, SVD};
use std::cmp::min;
type C32 = num_complex::Complex<f32>;
use num_complex::Complex as C64;
const C_LIGHT: f64 = 299_792_458.0;

struct CorEntry {
    a: usize,
    b: usize,
    path: String,
    mmap: Mmap,
    header: frinZ::header::CorHeader,
    times: Vec<i64>,
}
// Smoothing helpers (shared by --input and --cor modes)
#[derive(PartialEq, Debug, Clone, Copy)]
enum SmoothKind {
    None,
    MA,
    SG,
}
fn parse_smooth_kind(s: &str) -> SmoothKind {
    match s.to_lowercase().as_str() {
        "ma" => SmoothKind::MA,
        "sgolay" | "sg" => SmoothKind::SG,
        _ => SmoothKind::None,
    }
}
fn smooth_ma(series: &[f64], w: usize) -> Vec<f64> {
    let n = series.len();
    let k = w / 2;
    let mut out = vec![0.0; n];
    for i in 0..n {
        let s = i.saturating_sub(k);
        let e = min(n.saturating_sub(1), i + k);
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for j in s..=e {
            sum += series[j];
            cnt += 1;
        }
        out[i] = if cnt > 0 {
            sum / (cnt as f64)
        } else {
            series[i]
        };
    }
    out
}
fn smooth_sg(series: &[f64], w: usize, p: usize) -> Vec<f64> {
    let n = series.len();
    let mut out = vec![0.0; n];
    let m = (p + 1).max(1);
    for i in 0..n {
        let k = w / 2;
        let l = i.saturating_sub(k);
        let r = min(n.saturating_sub(1), i + k);
        // Build normal equations for centered polynomial at x=0
        let mut ata = vec![vec![0.0f64; m]; m];
        let mut atb = vec![0.0f64; m];
        for j in l..=r {
            let x = (j as isize - i as isize) as f64;
            let mut v = vec![1.0f64; m];
            for d in 1..m {
                v[d] = v[d - 1] * x;
            }
            let y = series[j];
            for a in 0..m {
                atb[a] += v[a] * y;
                for b in 0..m {
                    ata[a][b] += v[a] * v[b];
                }
            }
        }
        let ata_m = DMatrix::<f64>::from_row_slice(
            m,
            m,
            &ata.iter()
                .flat_map(|r| r.iter())
                .cloned()
                .collect::<Vec<_>>(),
        );
        let atb_v = DVector::<f64>::from_row_slice(&atb);
        let coeff = ata_m.lu().solve(&atb_v).unwrap_or(DVector::<f64>::zeros(m));
        out[i] = coeff[0];
    }
    out
}

fn rate_to_ss(rate_hz: f64, obs_freq_hz: f64) -> f64 {
    if obs_freq_hz.abs() < 1e-12 {
        0.0
    } else {
        rate_hz / obs_freq_hz
    }
}

fn write_json_pretty<W: Write + ?Sized, T: Serialize>(out: &mut W, value: &T) -> anyhow::Result<()> {
    let s = serde_json::to_string_pretty(value)?;
    out.write_all(s.as_bytes())?;
    out.write_all(b"\n")?;
    Ok(())
}

// Compute baseline phase after derotating by estimated delay/rate over time-frequency grid
// (moved below)

// Compute baseline phase after derotating by estimated delay/rate over time-frequency grid
fn derotated_mean_phase(
    complex_vec: &[C32],
    length_sectors: i32,
    fft_point: i32,
    sampling_speed: i32,
    effective_integ_time: f32,
    tau_s: f64,
    rate_hz: f64,
) -> f64 {
    let len = length_sectors.max(0) as usize;
    let fft_half = (fft_point.max(2) as usize) / 2;
    if len == 0 || fft_half == 0 {
        return 0.0;
    }
    let df_hz = sampling_speed as f64 / fft_point as f64;
    let mut acc = num_complex::Complex::<f64>::new(0.0, 0.0);
    for r in 0..len {
        let tsec = r as f64 * effective_integ_time as f64;
        let rate_phase = -2.0 * std::f64::consts::PI * rate_hz * tsec;
        let er = num_complex::Complex::<f64>::new(0.0, rate_phase).exp();
        for k in 0..fft_half {
            let f_hz = k as f64 * df_hz;
            let delay_phase = -2.0 * std::f64::consts::PI * tau_s * f_hz;
            let ed = num_complex::Complex::<f64>::new(0.0, delay_phase).exp();
            let idx = r * fft_half + k;
            let v = complex_vec[idx];
            let v64 = num_complex::Complex::<f64>::new(v.re as f64, v.im as f64);
            acc += v64 * er * ed;
        }
    }
    acc.arg()
}

// (unused helper reserved for future use)

fn compute_uv_for_pair(
    entries: &Vec<CorEntry>,
    ant_names: &BTreeMap<usize, String>,
    a: usize,
    b: usize,
    t: chrono::DateTime<chrono::Utc>,
) -> Option<(f64, f64)> {
    // Build name->id map
    let mut name_to_id: BTreeMap<String, usize> = BTreeMap::new();
    for (id, name) in ant_names {
        name_to_id.insert(name.clone(), *id);
    }
    // Find entry for this pair
    let mut chosen: Option<&CorEntry> = None;
    let mut flip = false;
    for e in entries {
        if (e.a == a && e.b == b) || (e.a == b && e.b == a) {
            chosen = Some(e);
            // If file is (b,a) but we want (a,b), flip baseline
            if e.a == b && e.b == a {
                flip = true;
            }
            break;
        }
    }
    let e = chosen?;
    let h = &e.header;
    let lam = C_LIGHT / (h.observing_frequency as f64);
    // GMST approximation
    let jd = frinZ::utils::mjd_cal(t) + 2400000.5;
    let gmst = astro::time::mn_sidr(jd) as f64;
    let rot = |x: f64, y: f64, z: f64| -> (f64, f64, f64) {
        let cg = gmst.cos();
        let sg = gmst.sin();
        (cg * x - sg * y, sg * x + cg * y, z)
    };
    // Source basis
    let ra = h.source_position_ra;
    let dec = h.source_position_dec;
    let ex = (-ra.sin(), ra.cos(), 0.0);
    let ey = (-dec.sin() * ra.cos(), -dec.sin() * ra.sin(), dec.cos());
    let dot = |b: (f64, f64, f64), e: (f64, f64, f64)| b.0 * e.0 + b.1 * e.1 + b.2 * e.2;
    // Determine baseline vector in ECEF for (a->b)
    let mut bx = h.station2_position[0] - h.station1_position[0];
    let mut by = h.station2_position[1] - h.station1_position[1];
    let mut bz = h.station2_position[2] - h.station1_position[2];
    // If header station names map to ids not matching (a,b), flip
    if let (Some(&id1), Some(&id2)) = (
        name_to_id.get(&h.station1_name),
        name_to_id.get(&h.station2_name),
    ) {
        if !(id1 == a && id2 == b) {
            // We want (a->b); if file is (b->a) or other, flip sign
            bx = -bx;
            by = -by;
            bz = -bz;
        }
    }
    if flip {
        bx = -bx;
        by = -by;
        bz = -bz;
    }
    let r = rot(bx, by, bz);
    let u = dot(r, ex) / lam;
    let v = dot(r, ey) / lam;
    Some((u, v))
}

#[derive(Parser, Debug)]
#[command(
    name = "gfrinZ",
    version = "0.1.0",
    author = "Masanori AKIMOTO  <masanori.akimoto.ac@gmail.com>",
    about = "Global (antenna-based) fringe solver that aggregates baseline-based estimates",
    arg_required_else_help = true,
    after_help = "Input: JSONL of baseline estimates with fields a,b,tau_s,rate_hz and optional sigma/w/snr/t_idx"
)]
struct Cli {
    /// Number of antennas（省略時は --cor のアンテナID集合から自動推定）
    #[arg(long)]
    antennas: Option<usize>,

    /// Reference antenna index (fixed to 0 delay/rate)
    #[arg(long, default_value_t = 0)]
    reference: usize,

    /// Input JSONL file with baseline estimates (aggregator mode)
    #[arg(long)]
    input: Option<PathBuf>,

    /// Output file (JSONL). If omitted, writes to stdout
    #[arg(long)]
    output: Option<PathBuf>,

    /// Per-baseline .cor inputs in the form a:b:/path/to/AB.cor (search+aggregate mode)
    /// 例: --cor 0:1:AB.cor --cor 0:2:AC.cor --cor 1:2:BC.cor
    #[arg(long, num_args = 1.., value_name = "A:B:CORPATH")]
    cor: Vec<String>,

    // 以下は .cor 入力時は不要。レガシー互換用に受けるが未指定でOK。
    #[arg(long, hide = true)]
    fft: Option<i32>,
    #[arg(long, hide = true)]
    fs: Option<i32>,
    #[arg(long, hide = true)]
    pp: Option<i32>,
    #[arg(long, hide = true)]
    ts: Option<f32>,

    /// Padding factor for rate FFT (as in frinZ --rate-padding)
    #[arg(long, default_value_t = 1)]
    rate_padding: u32,

    /// Precise search around the peak (as in frinZ --search)
    #[arg(long)]
    search: bool,

    /// フリンジ精密探索を明示的にオフにする（既定はON）
    #[arg(long)]
    no_search: bool,

    /// Delay window [min max] in samples (as in frinZ --delay-window)
    #[arg(long, num_args = 2)]
    delay_window: Vec<f32>,

    /// Rate window [min max] in Hz (as in frinZ --rate-window)
    #[arg(long, num_args = 2)]
    rate_window: Vec<f32>,

    /// 時系列: 積分長 [秒]（0=全体）。全基線で同一のeff.積分時間によりセクタ数へ変換
    #[arg(long, default_value_t = 0)]
    length: i32,

    /// 時系列: 先頭からのスキップ [秒]
    #[arg(long, default_value_t = 0)]
    skip: i32,

    /// 時系列: ループ回数（0=自動）
    #[arg(long, default_value_t = 0)]
    r#loop: i32,

    // 完全同期前提: 整列オプションは廃止（常に同一セクタindexで読み、時刻不一致はエラー）
    /// RFI 周波数帯域除外（MHz指定、複数可。例: --rfi 100,120 --rfi 400,500）
    #[arg(long, num_args = 1.., value_name = "MIN,MAX")]
    rfi: Vec<String>,

    /// バンドパス補正ファイル（frinZ の --bptable で生成した .bin を指定）
    #[arg(long)]
    bandpass: Option<PathBuf>,

    /// 基線ごとのバンドパス補正ファイル（"A-B:FILE" を複数指定可。A/B は局名またはID）。
    /// 例: --bandpass-bl "YAMAGU32-HITACH32:bp_y32_h32.bin" --bandpass-bl "1-2:bp_y34_h32.bin"
    #[arg(long, num_args = 1.., value_name = "PAIR:FILE")]
    bandpass_bl: Vec<String>,

    /// 各窓での基線ごとの中間結果（τ̂/ρ̂/SNR）をJSON出力に含める
    #[arg(long)]
    dump_baseline: bool,

    /// 重みに帯域係数 (BW/BW_ref)^2 を掛ける（BW_ref はエントリ中の最大帯域）
    #[arg(long)]
    bw_weight: bool,

    /// 近似CRLBに基づく重み（w=1/σ^2）を用いる。
    /// σ_tau ≈ 1/(2π SNR BW[Hz]), σ_rate ≈ 1/(2π SNR T[s]) を使用。
    /// 注意: 係数は近似でありデータ条件により偏り得ます。
    #[arg(long)]
    crlb_weight: bool,

    /// ロバスト再重み付け（"huber" または "tukey"）
    #[arg(long)]
    robust: Option<String>,

    /// ロバストの係数（Huberのk既定1.345、Tukeyのc既定4.685）
    #[arg(long)]
    robust_c: Option<f64>,

    /// ロバスト反復回数（既定2）
    #[arg(long, default_value_t = 2)]
    robust_iters: usize,

    /// 例と推奨組み合わせを表示して終了
    #[arg(long)]
    example: bool,

    /// JSON ではなくプレーンテキストで出力（1行/窓）
    #[arg(long)]
    plain: bool,

    /// 読みやすい整形テキストで出力（複数行、表形式）。--plain より優先。
    #[arg(long)]
    pretty: bool,

    /// 解析位相の選択: delay|frequency（既定: delay）。--cor の位相・クロージャーに影響。
    #[arg(long, default_value = "delay")]
    phase_kind: String,

    /// 単純なグローバル・フリンジ・フィットのみを実行（追加機能・出力を抑制）
    #[arg(long)]
    global_only: bool,

    /// 符号自動整合: off|header|triangle（既定: header）
    /// header: 各 .cor の station1/2 名称とアンテナID対応から A:B が逆なら反転
    /// triangle: 3アンテナ時に 0->1->2->0 の向きへ整合（N>3は未適用）
    #[arg(long, default_value = "header")]
    auto_flip: String,

    /// バイナリ出力先ディレクトリ（省略時は入力ファイルの場所に gfrinZ/ を作成）
    #[arg(long)]
    out_dir: Option<PathBuf>,

    /// 標準出力に結果を出さない（バイナリ出力のみ）
    #[arg(long)]
    no_stdout: bool,

    /// モデル除算（FRNMOD 相当）に用いるアンテナ解（JSONL: t_idx, tau_s[], rate_hz[]）。--cor専用
    #[arg(long)]
    model: Option<PathBuf>,

    /// クロージャーフェーズのワースト上位を表示（0=非表示）。--pretty のとき適用。
    #[arg(long, default_value_t = 0)]
    closure_top: usize,

    /// 時間方向スムージング: none|ma|sgolay（--cor モードのみ有効）
    #[arg(long, default_value = "none")]
    smooth: String,

    /// スムージング窓長（奇数推奨）
    #[arg(long, default_value_t = 5)]
    smooth_len: usize,

    /// Savitzky-Golay の多項式次数
    #[arg(long, default_value_t = 2)]
    smooth_poly: usize,

    /// バイナリ solutions.bin の出力種別: raw|smooth|both
    #[arg(long, default_value = "raw")]
    smooth_output: String,

    /// 構造フィット: none|double|jet|corejet（まずは double を有効化）
    #[arg(long, default_value = "none")]
    fit_model: String,

    /// フィット探索範囲（mas, ±範囲）
    #[arg(long, default_value_t = 50.0)]
    fit_range_mas: f64,

    /// 粗探索グリッド分割数（各軸）
    #[arg(long, default_value_t = 9)]
    fit_grid: usize,

    /// jet モデルの点数（将来用）
    #[arg(long, default_value_t = 3)]
    fit_jet_k: usize,

    /// 冗長基線判定: 相対差のしきい値（|Δ(u,v)|/max(|(u,v)|) < redundant_rel）
    #[arg(long, default_value_t = 0.02)]
    redundant_rel: f64,

    /// 冗長基線判定: 方向一致閾値（cosθ > redundant_cos）
    #[arg(long, default_value_t = 0.999)]
    redundant_cos: f64,

    /// 固定冗長ペア指定（ステーション名ペアを":"で区切り、各ペアは"A-B:C-D" 形式）。複数指定可。
    /// 例: --redundant-fixed "YAMAGU32-HITACH32:YAMAGU34-HITACH32" --redundant-fixed "YAMAGU32-TAKAHA32:YAMAGU34-TAKAHA32"
    #[arg(long, num_args = 0.., value_name = "A-B:C-D")]
    redundant_fixed: Vec<String>,

    /// FFT点数の異なる .cor を最小FFTに周波数リビン（平均化）して揃える（既定: 有効）
    /// 例: 8192 と 1024 が混在する場合、8192 側の周波数を8chずつ複素平均して 1024 相当に落とす
    #[arg(long, default_value_t = true)]
    rebin_to_min_fft: bool,
}

#[derive(Debug, Deserialize)]
struct BaselineRecord {
    a: usize,
    b: usize,
    tau_s: f64,
    rate_hz: f64,
    #[serde(default)]
    sigma_tau_s: Option<f64>,
    #[serde(default)]
    sigma_rate_hz: Option<f64>,
    #[serde(default)]
    phase_rad: Option<f64>,
    #[serde(default)]
    phase_deg: Option<f64>,
    #[serde(default)]
    sigma_phase_rad: Option<f64>,
    #[serde(default)]
    w_tau: Option<f64>,
    #[serde(default)]
    w_rate: Option<f64>,
    #[serde(default)]
    w_phase: Option<f64>,
    #[serde(default)]
    snr: Option<f64>,
    #[serde(default)]
    t_idx: Option<i64>,
}

fn weight_from(rec_w: Option<f64>, rec_sigma: Option<f64>, rec_snr: Option<f64>) -> f64 {
    if let Some(w) = rec_w {
        return w.max(1e-20);
    }
    if let Some(s) = rec_sigma {
        return (1.0 / (s * s)).max(1e-20);
    }
    if let Some(snr) = rec_snr {
        return (snr * snr).max(1e-20);
    }
    1.0
}
fn phase_from(rec_rad: Option<f64>, rec_deg: Option<f64>) -> Option<f64> {
    if let Some(p) = rec_rad {
        return Some(p);
    }
    if let Some(d) = rec_deg {
        return Some(d.to_radians());
    }
    None
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.example {
        example::print_examples();
        return Ok(());
    }

    let search_enabled = !cli.no_search;

    // Mode 1: Aggregator from JSONL (--input)
    if let Some(inp) = &cli.input {
        let infile = File::open(inp)?;
        let reader = BufReader::new(infile);

        // group by t_idx if present
        let mut groups: BTreeMap<i64, Vec<BaselineSolve>> = BTreeMap::new();
        let mut default_group: Vec<BaselineSolve> = Vec::new();

        for (lineno, line_res) in reader.lines().enumerate() {
            let line = line_res?;
            if line.trim().is_empty() {
                continue;
            }
            let rec: BaselineRecord = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    anyhow::bail!(
                        "Failed to parse JSON at line {}: {}\nline: {}",
                        lineno + 1,
                        e,
                        line
                    );
                }
            };
            let w_tau = weight_from(rec.w_tau, rec.sigma_tau_s, rec.snr);
            let w_rate = weight_from(rec.w_rate, rec.sigma_rate_hz, rec.snr);
            let phase_opt = phase_from(rec.phase_rad, rec.phase_deg);
            let mut w_phase = weight_from(rec.w_phase, rec.sigma_phase_rad, rec.snr);
            if phase_opt.is_none() {
                w_phase = 0.0;
            }
            let s = BaselineSolve {
                a: rec.a,
                b: rec.b,
                tau_s: rec.tau_s,
                rate_hz: rec.rate_hz,
                w_tau,
                w_rate,
                phase_rad: phase_opt.unwrap_or(0.0),
                w_phase,
            };
            if let Some(t) = rec.t_idx {
                groups.entry(t).or_default().push(s);
            } else {
                default_group.push(s);
            }
        }

        let mut out: Box<dyn Write> = if let Some(p) = cli.output.as_ref() {
            Box::new(File::create(p)?)
        } else {
            Box::new(io::stdout())
        };

        // Decide global antenna count for consistent smoothing across windows
        let num_ant_global = if let Some(n) = cli.antennas {
            n
        } else {
            let mut mx = 0usize;
            for s in &default_group {
                mx = mx.max(s.a.max(s.b));
            }
            for (_k, vecs) in &groups {
                for s in vecs {
                    mx = mx.max(s.a.max(s.b));
                }
            }
            mx + 1
        };
        // Handle default_group (no t_idx)
        if !default_group.is_empty() {
            let sol = solve_antenna_tau_rate(num_ant_global, cli.reference, &default_group);
            if cli.pretty {
                let phi = solve_antenna_phase(num_ant_global, cli.reference, &default_group);
                writeln!(out, "t_idx={} (no time) len_s=n/a", 0)?;
                writeln!(
                    out,
                    "Antennas: 0..{} (ref={})",
                    num_ant_global - 1,
                    cli.reference
                )?;
                writeln!(
                    out,
                    "idx  name                tau [s]         rate [Hz]       phase [deg]"
                )?;
                writeln!(
                    out,
                    "---- ------------------- --------------- --------------- ---------------"
                )?;
                for ant in 0..num_ant_global {
                    let mark = if ant == cli.reference { " (ref)" } else { "" };
                    writeln!(
                        out,
                        "{:<4}{:<19}{:>15.6} {:>15.6} {:>15.3}",
                        ant,
                        format!("{}{}", "ANT", mark),
                        sol.tau_s[ant],
                        sol.rate_hz[ant],
                        phi[ant].to_degrees()
                    )?;
                }
            } else if cli.plain {
                let tau_str = sol
                    .tau_s
                    .iter()
                    .map(|v| format!("{:.9e}", v))
                    .collect::<Vec<_>>()
                    .join(",");
                let rate_str = sol
                    .rate_hz
                    .iter()
                    .map(|v| format!("{:.9e}", v))
                    .collect::<Vec<_>>()
                    .join(",");
                writeln!(out, "tau_s=[{}] rate_hz=[{}]", tau_str, rate_str)?;
            } else {
                if !cli.pretty {
                    write_json_pretty(
                        &mut *out,
                        &serde_json::json!({
                            "tau_s": sol.tau_s,
                            "rate_hz": sol.rate_hz
                        }),
                    )?;
                }
            }
        }

        // Solve per t_idx and optionally smooth
        let smooth_kind = parse_smooth_kind(&cli.smooth);
        let mut raw_series: Vec<(i64, Vec<f64>, Vec<f64>)> = Vec::new(); // (t_idx, tau[num_ant], rate[num_ant])
        for (t, vecs) in &groups {
            let sol = solve_antenna_tau_rate(num_ant_global, cli.reference, vecs);
            raw_series.push((*t, sol.tau_s, sol.rate_hz));
        }
        // Already sorted by BTreeMap order
        if smooth_kind == SmoothKind::None {
            for (t, tau, rate) in raw_series {
                if cli.pretty {
                    let phi =
                        solve_antenna_phase(num_ant_global, cli.reference, groups.get(&t).unwrap());
                    writeln!(out, "t_idx={} len_s=n/a", t)?;
                    writeln!(
                        out,
                        "Antennas: 0..{} (ref={})",
                        num_ant_global - 1,
                        cli.reference
                    )?;
                    writeln!(
                        out,
                        "idx  name                tau [s]         rate [Hz]       phase [deg]"
                    )?;
                    writeln!(
                        out,
                        "---- ------------------- --------------- --------------- ---------------"
                    )?;
                    for ant in 0..num_ant_global {
                        let mark = if ant == cli.reference { " (ref)" } else { "" };
                        writeln!(
                            out,
                            "{:<4}{:<19}{:>15.6} {:>15.6} {:>15.3}",
                            ant,
                            format!("{}{}", "ANT", mark),
                            tau[ant],
                            rate[ant],
                            phi[ant].to_degrees()
                        )?;
                    }
                } else if cli.plain {
                    let tau_str = tau
                        .iter()
                        .map(|v| format!("{:.9e}", v))
                        .collect::<Vec<_>>()
                        .join(",");
                    let rate_str = rate
                        .iter()
                        .map(|v| format!("{:.9e}", v))
                        .collect::<Vec<_>>()
                        .join(",");
                    writeln!(
                        out,
                        "t_idx={} tau_s=[{}] rate_hz=[{}]",
                        t, tau_str, rate_str
                    )?;
                } else {
                    if !cli.pretty {
                        write_json_pretty(
                            &mut *out,
                            &serde_json::json!({
                                "t_idx": t,
                                "tau_s": tau,
                                "rate_hz": rate
                            }),
                        )?;
                    }
                }
            }
            return Ok(());
        }
        // Apply smoothing across windows per antenna
        let wlen = cli.smooth_len.max(1);
        let poly = cli.smooth_poly.max(0);
        let win_n = raw_series.len();
        if win_n == 0 {
            return Ok(());
        }
        let mut smoothed: Vec<(i64, Vec<f64>, Vec<f64>)> = raw_series.clone();
        for ant in 0..num_ant_global {
            let tau_series: Vec<f64> = raw_series.iter().map(|(_, t, _)| t[ant]).collect();
            let rate_series: Vec<f64> = raw_series.iter().map(|(_, _, r)| r[ant]).collect();
            let tau_s = match smooth_kind {
                SmoothKind::MA => smooth_ma(&tau_series, wlen),
                SmoothKind::SG => smooth_sg(&tau_series, wlen, poly),
                SmoothKind::None => tau_series,
            };
            let rate_s = match smooth_kind {
                SmoothKind::MA => smooth_ma(&rate_series, wlen),
                SmoothKind::SG => smooth_sg(&rate_series, wlen, poly),
                SmoothKind::None => rate_series,
            };
            for i in 0..win_n {
                smoothed[i].1[ant] = tau_s[i];
                smoothed[i].2[ant] = rate_s[i];
            }
        }
        // Output: pretty prints smoothed, and shows delta summary vs RAW
        for ((t, tau_raw, rate_raw), (_t2, tau_sm, rate_sm)) in
            raw_series.into_iter().zip(smoothed.into_iter())
        {
            if cli.pretty {
                // delta summary
                let mut max_dt_s = 0.0f64;
                let mut max_dr_hz = 0.0f64;
                for ant in 0..num_ant_global {
                    max_dt_s = max_dt_s.max((tau_sm[ant] - tau_raw[ant]).abs());
                    max_dr_hz = max_dr_hz.max((rate_sm[ant] - rate_raw[ant]).abs());
                }
                writeln!(
                    out,
                    "t_idx={} len_s=n/a (smoothed; |delta|max: tau={:.3e} s, rate={:.3e} Hz)",
                    t, max_dt_s, max_dr_hz
                )?;
                writeln!(out, "idx  name                tau [s]         rate [Hz]")?;
                writeln!(
                    out,
                    "---- ------------------- --------------- ---------------"
                )?;
                for ant in 0..num_ant_global {
                    let mark = if ant == cli.reference { " (ref)" } else { "" };
                    writeln!(
                        out,
                        "{:<4}{:<19}{:>15.6} {:>15.6}",
                        ant,
                        format!("{}{}", "ANT", mark),
                        tau_sm[ant],
                        rate_sm[ant]
                    )?;
                }
            } else if cli.plain {
                let tau_str = tau_sm
                    .iter()
                    .map(|v| format!("{:.9e}", v))
                    .collect::<Vec<_>>()
                    .join(",");
                let rate_str = rate_sm
                    .iter()
                    .map(|v| format!("{:.9e}", v))
                    .collect::<Vec<_>>()
                    .join(",");
                writeln!(
                    out,
                    "t_idx={} tau_s=[{}] rate_hz=[{}] (smoothed)",
                    t, tau_str, rate_str
                )?;
            } else {
                write_json_pretty(
                    &mut *out,
                    &serde_json::json!({
                        "t_idx": t,
                        "tau_s": tau_sm,
                        "rate_hz": rate_sm,
                        "smoothed": true
                    }),
                )?;
            }
        }
        return Ok(());
    }

    // Mode 2: Baseline search + aggregate (--cor A:B:AB.cor ...)
    if cli.cor.is_empty() {
        anyhow::bail!("--input か --cor A:B:/path/to/AB.cor のどちらかを指定してください");
    }

    // build frinZ Args to pass options to analyze_results
    let mut argv = vec![
        "gfrinZ".to_string(),
        format!("--rate-padding={}", cli.rate_padding),
    ];
    if search_enabled {
        argv.push("--search".to_string());
    }
    if cli.delay_window.len() == 2 {
        argv.push("--delay-window".to_string());
        argv.push(cli.delay_window[0].to_string());
        argv.push(cli.delay_window[1].to_string());
    }
    if cli.rate_window.len() == 2 {
        argv.push("--rate-window".to_string());
        argv.push(cli.rate_window[0].to_string());
        argv.push(cli.rate_window[1].to_string());
    }
    let args = FrinZArgs::parse_from(argv);

    let mut _solves: Vec<BaselineSolve> = Vec::new();

    // 1本目のヘッダを基準に整合性チェック
    let mut canonical: Option<(i32, i32, i32, f32)> = None; // (fft, fs, pp, ts)
    let mut obs_freq_hz: Option<f64> = None;

    // moved struct CorEntry to module scope

    fn read_sector_times(h: &frinZ::header::CorHeader, mmap: &Mmap) -> Vec<i64> {
        // Mirror read.rs: FILE_HEADER_SIZE=256, SECTOR_HEADER_SIZE=128
        let file_header_size: u64 = 256;
        let _sector_header_size: u64 = 128;
        let sector_size: u64 = ((8 + h.fft_point / 4) * 16) as u64;
        let mut out = Vec::with_capacity(h.number_of_sector as usize);
        for i in 0..(h.number_of_sector as usize) {
            let pos = file_header_size + (i as u64) * sector_size;
            // first 4 bytes are correlation_time_sec (i32 LE)
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&mmap[(pos as usize)..(pos as usize + 4)]);
            let sec = LittleEndian::read_i32(&buf) as i64;
            out.push(sec);
        }
        out
    }
    let mut entries: Vec<CorEntry> = Vec::new();
    let mut ants_set: BTreeSet<usize> = BTreeSet::new();
    let mut ant_names: BTreeMap<usize, String> = BTreeMap::new();

    for bls in &cli.cor {
        // parse A:B:/path
        let mut parts = bls.splitn(3, ':');
        let a: usize = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("invalid --bl format"))?
            .parse()?;
        let b: usize = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("invalid --bl format"))?
            .parse()?;
        let path = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("invalid --bl format"))?;
        // map .cor and parse header
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let header = frinZ::header::parse_header(&mut cursor)?;

        // consistency check and remember canonical
        // read one sector to get effective_integ_time
        cursor.set_position(0);
        let (_tmp, _t0, eff_ts) = read_visibility_data(&mut cursor, &header, 1, 0, 0, false, &[])?;
        if let Some((fft, fs, pp, ts)) = canonical {
            let mut warn = Vec::new();
            if fft != header.fft_point {
                warn.push(format!("fft_point {}!={}", header.fft_point, fft));
            }
            if fs != header.sampling_speed {
                warn.push(format!("fs {}!={}", header.sampling_speed, fs));
            }
            if pp != header.number_of_sector {
                warn.push(format!("pp {}!={}", header.number_of_sector, pp));
            }
            if (ts - eff_ts).abs() > 1e-6 {
                warn.push(format!("ts {:.6}!={:.6}", eff_ts, ts));
            }
            if !warn.is_empty() {
                eprintln!(
                    "#WARN: ヘッダ差異を検出: {} [{}] — 整列/時間窓で吸収を試みます",
                    path,
                    warn.join(", ")
                );
            }
        } else {
            canonical = Some((
                header.fft_point,
                header.sampling_speed,
                header.number_of_sector,
                eff_ts,
            ));
        }

        if let Some(prev) = obs_freq_hz {
            let rel = (header.observing_frequency - prev).abs() / prev.max(1.0);
            if rel > 1e-6 {
                eprintln!(
                    "#WARN: observing_frequency mismatch: {} Hz (expected {})",
                    header.observing_frequency, prev
                );
            }
        } else {
            obs_freq_hz = Some(header.observing_frequency);
        }

        let times = read_sector_times(&header, &mmap);
        ants_set.insert(a);
        ants_set.insert(b);
        // antenna names mapping
        if !ant_names.contains_key(&a) {
            ant_names.insert(a, header.station1_name.clone());
        } else if ant_names.get(&a).unwrap() != &header.station1_name {
            eprintln!(
                "#WARN: antenna {} name mismatch: '{}' vs '{}'",
                a,
                ant_names.get(&a).unwrap(),
                header.station1_name
            );
        }
        if !ant_names.contains_key(&b) {
            ant_names.insert(b, header.station2_name.clone());
        } else if ant_names.get(&b).unwrap() != &header.station2_name {
            eprintln!(
                "#WARN: antenna {} name mismatch: '{}' vs '{}'",
                b,
                ant_names.get(&b).unwrap(),
                header.station2_name
            );
        }
        entries.push(CorEntry {
            a,
            b,
            path: path.to_string(),
            mmap,
            header,
            times,
        });
    }

    // 3局入力時に三角形が成立しているか簡易チェック（欠損基線を警告）
    if ants_set.len() == 3 {
        let mut ants: Vec<usize> = ants_set.iter().copied().collect();
        ants.sort_unstable();
        let mut pair_set: BTreeSet<(usize, usize)> = BTreeSet::new();
        for e in &entries {
            let (p, q) = if e.a <= e.b { (e.a, e.b) } else { (e.b, e.a) };
            pair_set.insert((p, q));
        }
        let combos = vec![(ants[0], ants[1]), (ants[0], ants[2]), (ants[1], ants[2])];
        let mut missing: Vec<String> = Vec::new();
        for (p, q) in combos {
            if !pair_set.contains(&(p, q)) {
                missing.push(format!("{}-{}", p, q));
            }
        }
        if !missing.is_empty() || pair_set.len() != 3 {
            anyhow::bail!(
                "--cor で三角形が未成立です。欠損基線: {}",
                if missing.is_empty() { "(unknown)".to_string() } else { missing.join(", ") }
            );
        }
    }

    // FFT点数の最小値を計算（--rebin-to-min-fft のとき利用）
    let target_fft_opt: Option<i32> = if cli.rebin_to_min_fft {
        entries.iter().map(|e| e.header.fft_point).min()
    } else {
        None
    };
    if let Some(target_fft) = target_fft_opt {
        // 整数倍で揃えられるか検証
        for e in &entries {
            let fp = e.header.fft_point;
            if fp % target_fft != 0 {
                anyhow::bail!(
                    "FFT点数 {} を最小 {} にリビンできません（整数倍でない）",
                    fp,
                    target_fft
                );
            }
        }
        // 情報表示（標準出力）
        println!(
            "#INFO: Phase rebin to min FFT = {} (does not affect delay/rate)",
            target_fft
        );
        for e in &entries {
            let fp = e.header.fft_point;
            if fp != target_fft {
                let ratio = fp / target_fft;
                println!(
                    "#INFO:   {}-{}: FFT {} -> {} (avg {} bins)",
                    e.header.station1_name.trim(),
                    e.header.station2_name.trim(),
                    fp,
                    target_fft,
                    ratio
                );
            }
        }
    }

    let obs_freq_hz = obs_freq_hz.unwrap_or(0.0);
    if obs_freq_hz <= 0.0 {
        eprintln!("#WARN: observing_frequency が 0 Hz なので rate の s/s 変換をスキップします");
    }

    let mut bp_map: std::collections::HashMap<(usize, usize), Vec<num_complex::Complex<f32>>> =
        std::collections::HashMap::new();
    // 基線別バンドパスの読み込み（局名またはIDで指定可）
    if !cli.bandpass_bl.is_empty() {
        // 局名->ID の逆引きを用意
        let mut name_to_id: BTreeMap<String, usize> = BTreeMap::new();
        for (id, name) in &ant_names {
            name_to_id.insert(name.clone(), *id);
        }
        for spec in &cli.bandpass_bl {
            if let Some((pair, fpath)) = spec.split_once(':') {
                let pair = pair.trim();
                let fpath = fpath.trim();
                let parse_pair = |s: &str| -> Option<(usize, usize)> {
                    if let Some((x, y)) = s.split_once('-') {
                        let xx = x.trim();
                        let yy = y.trim();
                        let a = xx
                            .parse::<usize>()
                            .ok()
                            .or_else(|| name_to_id.get(xx).cloned());
                        let b = yy
                            .parse::<usize>()
                            .ok()
                            .or_else(|| name_to_id.get(yy).cloned());
                        if let (Some(ai), Some(bi)) = (a, b) {
                            Some((ai, bi))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };
                if let Some((a, b)) = parse_pair(pair) {
                    match read_bandpass_file(&PathBuf::from(fpath)) {
                        Ok(v) => {
                            let key = if a <= b { (a, b) } else { (b, a) };
                            bp_map.insert(key, v);
                        }
                        Err(e) => {
                            eprintln!("#WARN: 基線BP読み込み失敗 {}:{} -> {}: {}", a, b, fpath, e)
                        }
                    }
                } else {
                    eprintln!("#WARN: --bandpass-bl の基線指定を解釈できません: {}", spec);
                }
            } else {
                eprintln!("#WARN: --bandpass-bl は 'PAIR:FILE' 形式です: {}", spec);
            }
        }
    }

    // アンテナ数の自動推定（未指定時）
    let auto_antennas = if ants_set.is_empty() {
        0
    } else {
        (*ants_set.iter().max().unwrap() + 1).max(ants_set.len())
    };
    let num_ant = cli.antennas.unwrap_or(auto_antennas);
    if num_ant == 0 {
        anyhow::bail!("アンテナ数を特定できません (--antennas を指定するか、--cor のアンテナIDを確認してください)");
    }
    if cli.reference >= num_ant {
        anyhow::bail!(
            "reference must be < antennas (ref={}, antennas={})",
            cli.reference,
            num_ant
        );
    }

    // 時系列ウィンドウの設定（秒→セクタ）
    let (_fft, _fs, pp, ts) = canonical.unwrap();
    let length_sec = cli.length;
    let skip_sec = cli.skip;
    let length_sectors = if length_sec == 0 {
        pp
    } else {
        ((length_sec as f32) / ts).ceil() as i32
    };
    let skip_sectors = if skip_sec == 0 {
        0
    } else {
        ((skip_sec as f32) / ts).round() as i32
    };
    let max_loops = (pp - skip_sectors) / length_sectors;
    let loop_count = if cli.r#loop <= 0 {
        max_loops
    } else {
        cli.r#loop.min(max_loops)
    };

    // 事前に（もし指定があれば）バンドパスを読み込む（全基線共通 / 基線別）
    let mut bp_data_all: Option<Vec<num_complex::Complex<f32>>> = None;
    if let Some(bp) = &cli.bandpass {
        match read_bandpass_file(bp) {
            Ok(v) => {
                bp_data_all = Some(v);
            }
            Err(e) => {
                eprintln!(
                    "#WARN: バンドパスファイルの読み込みに失敗: {:?}: {} — 補正をスキップします",
                    bp, e
                );
            }
        }
    }

    // 実行パラメータの要約（--cor モード）
    if !cli.no_stdout {
        let mode = if cli.input.is_some() { "input" } else { "cor" };
        println!(
            "#PARAM: mode={} ref={} antennas={} baselines={}",
            mode,
            cli.reference,
            num_ant,
            entries.len()
        );
        if let Some((fft, fs, pp, ts)) = canonical {
            println!(
                "#PARAM: header (fft_point={}, fs={} Hz, sectors={}, eff_ts={:.6} s)",
                fft, fs, pp, ts
            );
        }
        println!(
            "#PARAM: length={} s skip={} s loop={}",
            cli.length, cli.skip, cli.r#loop
        );
        println!(
            "#PARAM: search={} rate_padding={} phase_kind={} auto_flip={}",
            search_enabled, cli.rate_padding, cli.phase_kind, cli.auto_flip
        );
        if !cli.rfi.is_empty() {
            println!("#PARAM: rfi={:?}", cli.rfi);
        }
        if let Some(bp) = &cli.bandpass {
            println!("#PARAM: bandpass={}", bp.display());
        }
        println!(
            "#PARAM: weights crlb_weight={} bw_weight={}",
            cli.crlb_weight, cli.bw_weight
        );
        if let Some(r) = &cli.robust {
            println!(
                "#PARAM: robust={} c={:?} iters={}",
                r, cli.robust_c, cli.robust_iters
            );
        }
        if let Some(mp) = &cli.model {
            println!("#PARAM: model={}", mp.display());
        }
        println!("#PARAM: smooth(kind={}, len={}, poly={}) output(bin_dir={:?} pretty={} plain={} no_stdout={} dump_baseline={} closure_top={})",
            cli.smooth, cli.smooth_len, cli.smooth_poly, cli.out_dir, cli.pretty, cli.plain, cli.no_stdout, cli.dump_baseline, cli.closure_top);
        // Redundant thresholds are deprecated; using fixed-name pairs if provided
        if !cli.redundant_fixed.is_empty() {
            println!("#PARAM: redundant fixed={}", cli.redundant_fixed.join(","));
        }
    }

    let mut out: Box<dyn Write> = if let Some(p) = cli.output.as_ref() {
        Box::new(File::create(p)?)
    } else {
        Box::new(io::stdout())
    };

    // シンプルモード（global_only）: 追加機能を抑制して一発解
    if cli.global_only {
        // 時系列ウィンドウ設定
        let (_fft, _fs, pp, ts) = canonical.unwrap();
        let length_sec = cli.length;
        let skip_sec = cli.skip;
        let length_sectors = if length_sec == 0 {
            pp
        } else {
            ((length_sec as f32) / ts).ceil() as i32
        };
        let base_start_idx = if skip_sec == 0 {
            0
        } else {
            ((skip_sec as f32) / ts).round() as i32
        };

        // ベースラインごとの τ/ṫ と位相（peak）を取得
        let mut solves: Vec<BaselineSolve> = Vec::new();
        let mut bl_list: Vec<(usize, usize, f64, f64, f64, f64)> = Vec::new(); // (a,b,tau_s,rate_hz,phase_deg,snr)
        let mut win_time: Option<chrono::DateTime<chrono::Utc>> = None;
        let mut phase_map: std::collections::HashMap<(usize, usize), f64> =
            std::collections::HashMap::new();

        for e in &entries {
            let mut cursor = std::io::Cursor::new(&e.mmap[..]);
            let (complex_vec, t0, effective_integ_time) = read_visibility_data(
                &mut cursor,
                &e.header,
                length_sectors,
                base_start_idx,
                0,
                false,
                &[], // pp_flag_ranges
            )?;
            if win_time.is_none() {
                win_time = Some(t0);
            }
            let rfi_ranges: Vec<(usize, usize)> = vec![];
            let (freq_rate_array, padding_length) = process_fft(
                &complex_vec,
                length_sectors,
                e.header.fft_point,
                e.header.sampling_speed,
                &rfi_ranges,
                cli.rate_padding,
            );
            // バンドパス（基線別優先→全体）
            let mut freq_rate_array = freq_rate_array;
            let key = if e.a <= e.b { (e.a, e.b) } else { (e.b, e.a) };
            if let Some(bp) = bp_map.get(&key) {
                if bp.len() == (e.header.fft_point as usize / 2) {
                    apply_bandpass_correction(&mut freq_rate_array, bp);
                } else {
                    eprintln!(
                        "#WARN: 基線BP長不一致のためスキップ (bp={}, fft/2={})",
                        bp.len(),
                        e.header.fft_point as usize / 2
                    );
                }
            } else if let Some(bp) = &bp_data_all {
                if bp.len() == (e.header.fft_point as usize / 2) {
                    apply_bandpass_correction(&mut freq_rate_array, bp);
                } else {
                    eprintln!(
                        "#WARN: バンドパス長がFFT/2と不一致のためスキップ (bp={}, fft/2={})",
                        bp.len(),
                        e.header.fft_point as usize / 2
                    );
                }
            }
            let delay_rate_2d_data_comp =
                process_ifft(&freq_rate_array, e.header.fft_point, padding_length);
            // 枠組み上、FrinZArgsを適当に用意
            let mut argv: Vec<String> = vec!["frinZ".to_string()];
            if search_enabled {
                argv.push("--search".to_string());
            }
            let args = FrinZArgs::parse_from(argv);
            let analysis = analyze_results(
                &freq_rate_array,
                &delay_rate_2d_data_comp,
                &e.header,
                length_sectors,
                effective_integ_time,
                &t0,
                padding_length,
                &args,
                args.primary_search_mode(),
            );

            // cor ファイルの遅延・レートは「ant2 - ant1」。gfrinZ も B-A 系で扱う（符号反転しない）。
            let tau_s = analysis.residual_delay as f64 / e.header.sampling_speed as f64;
            let rate_hz = analysis.residual_rate as f64;
            let snr = analysis.delay_snr as f64;
            let phase_peak_rad = if cli.phase_kind.to_lowercase().starts_with("f") {
                (analysis.freq_phase as f64).to_radians()
            } else {
                (analysis.delay_phase as f64).to_radians()
            };

            solves.push(BaselineSolve {
                a: e.a,
                b: e.b,
                tau_s,
                rate_hz,
                w_tau: (snr * snr).max(1e-20),
                w_rate: (snr * snr).max(1e-20),
                phase_rad: phase_peak_rad,
                w_phase: (snr * snr).max(1e-20),
            });
            phase_map.insert((e.a, e.b), phase_peak_rad);
            let ph_deg = if cli.phase_kind.to_lowercase().starts_with("f") {
                analysis.freq_phase as f64
            } else {
                analysis.delay_phase as f64
            };
            bl_list.push((e.a, e.b, tau_s, rate_hz, ph_deg, snr));
        }

        // アンテナ解
        let sol = solve_antenna_tau_rate(num_ant, cli.reference, &solves);
        let utc = win_time.unwrap_or_else(|| chrono::Utc::now()).to_rfc3339();
        writeln!(
            out,
            "t_idx={} utc={} len_s={:.3}",
            0,
            utc,
            (length_sectors as f32) * ts
        )?;
        if obs_freq_hz > 0.0 {
            writeln!(out, "observing_frequency={:.6} Hz (rate=s/s)", obs_freq_hz)?;
        }
        // Baseline-based fringe search results (for comparison)
        writeln!(out, "Baselines:")?;
        writeln!(
            out,
            "a-b  name                    tau [s]         rate [s/s]       phase [deg]"
        )?;
        writeln!(
            out,
            "---- ----------------------- --------------- --------------- ---------------"
        )?;
        for (a, b, tau, rate, ph, snr) in &bl_list {
            let na = ant_names.get(a).cloned().unwrap_or_else(|| "?".to_string());
            let nb = ant_names.get(b).cloned().unwrap_or_else(|| "?".to_string());
            writeln!(
                out,
                "{:>1}-{:>1}  {:<23} {:>15.6} {:>15.6e} {:>15.3}",
                a,
                b,
                format!("{}-{} (SNR={:.1})", na, nb, snr),
                *tau,
                rate_to_ss(*rate, obs_freq_hz),
                *ph
            )?;
        }
        let mut names_line = String::new();
        let mut keys: Vec<usize> = ant_names.keys().cloned().collect();
        keys.sort_unstable();
        for k in keys {
            let name = ant_names.get(&k).unwrap();
            names_line.push_str(&format!(
                " {}={}{}",
                k,
                name,
                if k == cli.reference { "(ref)" } else { "" }
            ));
        }
        writeln!(out, "Antennas:{}", names_line)?;
        writeln!(out, "idx  name                tau [s]         rate [s/s]")?;
        writeln!(
            out,
            "---- ------------------- --------------- ---------------"
        )?;
        for ant in 0..num_ant {
            let name = ant_names.get(&ant).unwrap();
            let mark = if ant == cli.reference { " (ref)" } else { "" };
            writeln!(
                out,
                "{:<4}{:<19}{:>15.6e} {:>15.6e}",
                ant,
                format!("{}{}", name, mark),
                sol.tau_s[ant],
                rate_to_ss(sol.rate_hz[ant], obs_freq_hz)
            )?;
        }
        // クロージャ（三角が構成できるとき）
        if num_ant >= 3 {
            // 3局だけのときに限り (0,1,2) で表示（一般化は割愛）
            if phase_map.contains_key(&(0, 1))
                && phase_map.contains_key(&(1, 2))
                && (phase_map.contains_key(&(2, 0)) || phase_map.contains_key(&(0, 2)))
            {
                let p01 = phase_map
                    .get(&(0, 1))
                    .copied()
                    .or_else(|| phase_map.get(&(1, 0)).map(|v| -*v))
                    .unwrap();
                let p12 = phase_map
                    .get(&(1, 2))
                    .copied()
                    .or_else(|| phase_map.get(&(2, 1)).map(|v| -*v))
                    .unwrap();
                let p20 = phase_map
                    .get(&(2, 0))
                    .copied()
                    .or_else(|| phase_map.get(&(0, 2)).map(|v| -*v))
                    .unwrap();
                let wrap = |x: f64| {
                    let mut y = (x + std::f64::consts::PI) % (2.0 * std::f64::consts::PI);
                    if y < 0.0 {
                        y += 2.0 * std::f64::consts::PI;
                    }
                    y - std::f64::consts::PI
                };
                let cp = wrap(p01 + p12 + p20).to_degrees().abs();
                writeln!(out, "Closure phase (peak-based): {:.3} deg", cp)?;
            }
        }
        return Ok(());
    }

    // バイナリ出力（solutions.bin, baselines.bin）準備
    let out_dir = cli.out_dir.clone().unwrap_or_else(|| {
        if !cli.cor.is_empty() {
            let first_cor_path = cli.cor[0].splitn(3, ':').last().unwrap_or(".");
            PathBuf::from(first_cor_path)
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join("gfrinZ")
        } else if let Some(input_path) = &cli.input {
            input_path
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join("gfrinZ")
        } else {
            PathBuf::from("gfrinZ_out") // Fallback
        }
    });
    std::fs::create_dir_all(&out_dir)?;
    let sol_path = out_dir.join("solutions.bin");
    let bln_path = out_dir.join("baselines.bin");
    let mut sol_f = BufWriter::new(File::create(&sol_path)?);
    let mut bln_f = BufWriter::new(File::create(&bln_path)?);

    // ヘッダ: magic(4), version(u32), num_windows(u32), num_ant(u32)
    fn write_u32_le<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
        use byteorder::WriteBytesExt;
        w.write_u32::<LittleEndian>(v)
    }
    fn write_i64_le<W: Write>(w: &mut W, v: i64) -> io::Result<()> {
        use byteorder::WriteBytesExt;
        w.write_i64::<LittleEndian>(v)
    }
    fn write_i32_le<W: Write>(w: &mut W, v: i32) -> io::Result<()> {
        use byteorder::WriteBytesExt;
        w.write_i32::<LittleEndian>(v)
    }
    fn write_f32_le<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
        use byteorder::WriteBytesExt;
        w.write_f32::<LittleEndian>(v)
    }
    fn write_f64_le<W: Write>(w: &mut W, v: f64) -> io::Result<()> {
        use byteorder::WriteBytesExt;
        w.write_f64::<LittleEndian>(v)
    }

    // solutions.bin header
    sol_f.write_all(b"gsol")?; // magic
    write_u32_le(&mut sol_f, 1)?; // version
    write_u32_le(&mut sol_f, loop_count as u32)?; // num_windows
    write_u32_le(&mut sol_f, num_ant as u32)?; // num_ant

    // baselines.bin header
    bln_f.write_all(b"gbln")?;
    write_u32_le(&mut bln_f, 1)?; // version
    write_u32_le(&mut bln_f, loop_count as u32)?; // num_windows
    write_u32_le(&mut bln_f, entries.len() as u32)?; // num_baselines (一定と仮定)

    // 帯域重み用に参照帯域（最大帯域）を計算
    let bw_ref_mhz: f64 = entries
        .iter()
        .map(|e| e.header.sampling_speed as f64 / 2.0 / 1_000_000.0)
        .fold(0.0, |acc, x| acc.max(x));

    // 構造フィット用の蓄積（3局・複数窓）
    #[allow(unused_variables)]
    let mut fit_obs_closure: Vec<f64> = Vec::new();
    #[allow(unused_variables)]
    let mut fit_uv_01: Vec<(f64, f64)> = Vec::new();
    #[allow(unused_variables)]
    let mut fit_uv_12: Vec<(f64, f64)> = Vec::new();
    #[allow(unused_variables)]
    let mut fit_uv_20: Vec<(f64, f64)> = Vec::new();

    // print antenna mapping once
    if !cli.no_stdout {
        let mapping = ant_names
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(", ");
        println!("# Antennas: {}", mapping);
    }

    // スムージング関連（--corモードのみ）
    #[derive(Clone)]
    struct WinSol {
        t_idx: i32,
        unix_sec: i64,
        len_s: f32,
        tau: Vec<f64>,
        rate: Vec<f64>,
        utc: String,
    }
    let smooth_kind = parse_smooth_kind(&cli.smooth);
    let mut win_solutions: Vec<WinSol> = Vec::new();

    // モデル（FRNMOD相当）を読み込み
    #[derive(Deserialize)]
    struct ModelRec {
        #[serde(default)]
        t_idx: Option<i64>,
        tau_s: Vec<f64>,
        rate_hz: Vec<f64>,
    }
    let mut model_default: Option<(Vec<f64>, Vec<f64>)> = None;
    let mut model_map: BTreeMap<i64, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    if let Some(mp) = &cli.model {
        let f = File::open(mp)?;
        let rdr = BufReader::new(f);
        for (lineno, lr) in rdr.lines().enumerate() {
            let line = lr?;
            if line.trim().is_empty() {
                continue;
            }
            let m: ModelRec = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    anyhow::bail!("model JSON parse error at line {}: {}", lineno + 1, e);
                }
            };
            if let Some(t) = m.t_idx {
                model_map.insert(t, (m.tau_s, m.rate_hz));
            } else {
                model_default = Some((m.tau_s, m.rate_hz));
            }
        }
    }

    // 位相ソースの選択（delay or frequency）
    let _phase_use_freq = cli.phase_kind.to_lowercase().starts_with("f");

    // 自動符号整合のモード
    let auto_flip_mode = cli.auto_flip.to_lowercase();

    for l1 in 0..loop_count {
        let base_start_idx = skip_sectors + l1 * length_sectors;
        if base_start_idx < 0 {
            anyhow::bail!("負の開始インデックス: {}", base_start_idx);
        }
        let base_start_idx_us = base_start_idx as usize;
        if base_start_idx_us >= entries[0].times.len() {
            anyhow::bail!(
                "開始インデックスが範囲外: {} >= {}",
                base_start_idx_us,
                entries[0].times.len()
            );
        }

        // 完全同期前提: 各 .cor のこの開始セクタの時刻が完全一致していることを検証
        let base_sec = entries[0].times[base_start_idx_us];
        for e in &entries[1..] {
            if base_start_idx_us >= e.times.len() {
                anyhow::bail!(
                    "{}: 開始インデックスが範囲外: {} >= {}",
                    e.path,
                    base_start_idx_us,
                    e.times.len()
                );
            }
            let sec = e.times[base_start_idx_us];
            if sec != base_sec {
                anyhow::bail!("時刻不一致: {} の開始時刻 {} != 基準 {} (インデックス {}), 入力の完全同期を確認してください", e.path, sec, base_sec, base_start_idx_us);
            }
        }

        // 各基線の窓を同一 index で読み出し、解析
        let mut solves: Vec<BaselineSolve> = Vec::new();
        let mut win_time: Option<chrono::DateTime<chrono::Utc>> = None;
        // 収集用: 基線ごとの中間結果
        let mut baselines_json: Vec<serde_json::Value> = Vec::new();

        let mut sign_flips: Vec<(usize, usize)> = Vec::new();
        for e in &entries {
            if base_start_idx_us + (length_sectors as usize) > e.times.len() {
                anyhow::bail!(
                    "{}: 窓が範囲外に出ます (start {} + len {} > {})",
                    e.path,
                    base_start_idx_us,
                    length_sectors,
                    e.times.len()
                );
            }
            let mut cursor = std::io::Cursor::new(&e.mmap[..]);
            let (mut complex_vec, t0, effective_integ_time) = read_visibility_data(
                &mut cursor,
                &e.header,
                length_sectors,
                base_start_idx,
                0,
                false,
                &[], // pp_flag_ranges
            )?;
            if win_time.is_none() {
                win_time = Some(t0);
            }

            // FRNMOD相当: モデルで位相除去（baseline毎）
            if cli.model.is_some() {
                let model = model_map
                    .get(&(l1 as i64))
                    .or_else(|| model_default.as_ref());
                if let Some((tau_ant, rate_ant)) = model {
                    if e.a < tau_ant.len() && e.b < tau_ant.len() {
                        let dt = tau_ant[e.a] - tau_ant[e.b]; // [s]
                        let dr = rate_ant[e.a] - rate_ant[e.b]; // [Hz]
                                                                // apply: for each time row r and freq bin k
                        let fft_half = (e.header.fft_point / 2) as usize;
                        let dt_s = dt as f32;
                        let dr_hz = dr as f32;
                        let df_hz = e.header.sampling_speed as f32 / e.header.fft_point as f32;
                        for r in 0..(length_sectors as usize) {
                            let tsec = r as f32 * effective_integ_time;
                            let rate_factor = num_complex::Complex::<f32>::new(
                                0.0,
                                -2.0 * std::f32::consts::PI * dr_hz * tsec,
                            )
                            .exp();
                            for k in 0..fft_half {
                                let f_hz = k as f32 * df_hz;
                                let delay_factor = num_complex::Complex::<f32>::new(
                                    0.0,
                                    -2.0 * std::f32::consts::PI * dt_s * f_hz,
                                )
                                .exp();
                                let idx = r * fft_half + k;
                                complex_vec[idx] *= rate_factor * delay_factor;
                            }
                        }
                    }
                }
            }

            // 位相評価用の周波数リビン配列（遅延/レートの探索には影響させない）
            let mut phase_vec: Option<(Vec<num_complex::Complex<f32>>, i32)> = None;
            if let Some(target_fft) = target_fft_opt {
                if e.header.fft_point != target_fft {
                    let ratio = (e.header.fft_point / target_fft) as usize;
                    let in_half = (e.header.fft_point / 2) as usize;
                    let out_half = (target_fft / 2) as usize;
                    let mut rebinned = vec![
                        num_complex::Complex::<f32>::new(0.0, 0.0);
                        (length_sectors as usize) * out_half
                    ];
                    for r in 0..(length_sectors as usize) {
                        let r_in_off = r * in_half;
                        let r_out_off = r * out_half;
                        for k in 0..out_half {
                            let mut acc = num_complex::Complex::<f32>::new(0.0, 0.0);
                            let base = r_in_off + k * ratio;
                            for j in 0..ratio {
                                acc += complex_vec[base + j];
                            }
                            rebinned[r_out_off + k] = acc / (ratio as f32);
                        }
                    }
                    phase_vec = Some((rebinned, target_fft));
                }
            }

            // RFI 除外レンジ計算（MHz単位）
            let bw_mhz = e.header.sampling_speed as f32 / 2.0 / 1_000_000.0;
            let rbw_mhz = (bw_mhz / e.header.fft_point as f32) * 2.0;
            let rfi_ranges = parse_rfi_ranges(&cli.rfi, rbw_mhz).unwrap_or_else(|_| vec![]);
            let (freq_rate_array, padding_length) = process_fft(
                &complex_vec,
                length_sectors,
                e.header.fft_point,
                e.header.sampling_speed,
                &rfi_ranges,
                cli.rate_padding,
            );
            // バンドパス補正（基線別優先→全体）。サイズが一致する場合のみ適用
            let mut freq_rate_array = freq_rate_array;
            let key = if e.a <= e.b { (e.a, e.b) } else { (e.b, e.a) };
            if let Some(bp) = bp_map.get(&key) {
                if bp.len() == (e.header.fft_point as usize / 2) {
                    apply_bandpass_correction(&mut freq_rate_array, bp);
                } else {
                    eprintln!(
                        "#WARN: 基線BP長不一致のためスキップ (bp={}, fft/2={})",
                        bp.len(),
                        e.header.fft_point as usize / 2
                    );
                }
            } else if let Some(bp) = &bp_data_all {
                if bp.len() == (e.header.fft_point as usize / 2) {
                    apply_bandpass_correction(&mut freq_rate_array, bp);
                } else {
                    eprintln!(
                        "#WARN: バンドパス長がFFT/2と不一致のためスキップ (bp={}, fft/2={})",
                        bp.len(),
                        e.header.fft_point as usize / 2
                    );
                }
            }

            let delay_rate_2d_data_comp =
                process_ifft(&freq_rate_array, e.header.fft_point, padding_length);

            let analysis = analyze_results(
                &freq_rate_array,
                &delay_rate_2d_data_comp,
                &e.header,
                length_sectors,
                effective_integ_time,
                &t0,
                padding_length,
                &args,
                args.primary_search_mode(),
            );

            let mut tau_s = analysis.residual_delay as f64 / e.header.sampling_speed as f64;
            let mut rate_hz = analysis.residual_rate as f64;
            let snr = analysis.delay_snr as f64;
            // Baseline phase definitions
            // peak-based (from analysis): degrees -> radians
            let _phase_peak_rad = if _phase_use_freq {
                (analysis.freq_phase as f64).to_radians()
            } else {
                (analysis.delay_phase as f64).to_radians()
            };
            // derotated-mean (common-basis) for stable averaging
            let (phase_src, phase_fft) = if let Some((pbins, pfft)) = &phase_vec {
                (&pbins[..], *pfft)
            } else {
                (&complex_vec[..], e.header.fft_point)
            };
            let phase_mean_rad = derotated_mean_phase(
                phase_src,
                length_sectors,
                phase_fft,
                e.header.sampling_speed,
                effective_integ_time,
                tau_s,
                rate_hz,
            );
            // use mean phase for internal weighting, but also keep peak for diagnostics
            let mut phase_rad = phase_mean_rad;

            // 自動位相符号の整合
            if auto_flip_mode == "header" {
                // header の station1/2 名称とアンテナID対応から A:B が逆なら反転
                let mut name_to_id: BTreeMap<String, usize> = BTreeMap::new();
                for (id, name) in &ant_names {
                    name_to_id.insert(name.clone(), *id);
                }
                if let (Some(&s1), Some(&s2)) = (
                    name_to_id.get(&e.header.station1_name),
                    name_to_id.get(&e.header.station2_name),
                ) {
                    if s1 == e.b && s2 == e.a {
                        tau_s *= -1.0;
                        rate_hz *= -1.0;
                        phase_rad *= -1.0;
                        eprintln!(
                            "#INFO: flipped sign (header) for baseline {}:{} (file order {}->{})",
                            e.a, e.b, e.header.station1_name, e.header.station2_name
                        );
                        sign_flips.push((e.a, e.b));
                    }
                }
            } // triangle モードではデータの符号は変更しない（解析上の位相向きだけ合わせる）
              // 重み計算
            let w_tau: f64;
            let w_rate: f64;
            if cli.crlb_weight {
                let bw_hz = e.header.sampling_speed as f64 / 2.0; // 半帯域
                let t_obs = (length_sectors as f64) * (effective_integ_time as f64);
                let two_pi = std::f64::consts::PI * 2.0;
                let sigma_tau = (1.0 / (two_pi * (snr as f64) * bw_hz)).max(1e-12);
                let sigma_rate = (1.0 / (two_pi * (snr as f64) * t_obs)).max(1e-12);
                w_tau = 1.0 / (sigma_tau * sigma_tau);
                w_rate = 1.0 / (sigma_rate * sigma_rate);
            } else {
                let mut w = (snr * snr).max(1e-20);
                if cli.bw_weight {
                    let bw_mhz = e.header.sampling_speed as f64 / 2.0 / 1_000_000.0;
                    let scale = (bw_mhz / bw_ref_mhz).powi(2);
                    w *= scale.max(1e-12);
                }
                w_tau = w;
                w_rate = w;
            }

            let w_phase = if cli.crlb_weight {
                (snr * snr).max(1e-20)
            } else {
                (snr * snr).max(1e-20)
            };
            solves.push(BaselineSolve {
                a: e.a,
                b: e.b,
                tau_s,
                rate_hz,
                w_tau,
                w_rate,
                phase_rad,
                w_phase,
            });

            // 常に基線レコードは収集（バイナリ出力のため）。JSONへの埋め込みは後段のフラグで制御
            baselines_json.push(serde_json::json!({
                "a": e.a,
                "b": e.b,
                "tau_s": tau_s,
                "rate_hz": rate_hz,
                "rate_s_per_s": rate_to_ss(rate_hz, obs_freq_hz),
                "snr": snr,
                "weight_tau": w_tau,
                "weight_rate": w_rate,
                "phase_deg_peak": if _phase_use_freq { analysis.freq_phase as f64 } else { analysis.delay_phase as f64 },
            }));
        }

        // 初期解（残差ベース）
        let sol_res = solve_antenna_tau_rate(num_ant, cli.reference, &solves);
        // 出力用の総和解（既定は残差=総和）。--model 指定時はモデルを加算
        let mut sol = sol_res.clone();

        // ロバスト再重み付け（IRLS）
        if let Some(kind) = &cli.robust {
            let kind_l = kind.to_lowercase();
            let default_c = if kind_l == "huber" { 1.345 } else { 4.685 };
            let c = cli.robust_c.unwrap_or(default_c);
            for _ in 0..cli.robust_iters {
                // 残差（基線）を計算
                let mut r_tau: Vec<f64> = Vec::with_capacity(solves.len());
                let mut r_rate: Vec<f64> = Vec::with_capacity(solves.len());
                for s in &solves {
                    let dt = sol.tau_s[s.a] - sol.tau_s[s.b];
                    let dr = sol.rate_hz[s.a] - sol.rate_hz[s.b];
                    r_tau.push(s.tau_s - dt);
                    r_rate.push(s.rate_hz - dr);
                }
                // スケール推定（MAD）
                fn mad_scale(v: &mut [f64]) -> f64 {
                    if v.is_empty() {
                        return 1.0;
                    }
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let med = v[v.len() / 2];
                    let mut devs: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
                    devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mad = devs[devs.len() / 2];
                    let s = mad * 1.4826; // 正規分布に対する補正
                    if s.is_finite() && s > 1e-20 {
                        s
                    } else {
                        1.0
                    }
                }
                let mut r_tau_copy = r_tau.clone();
                let mut r_rate_copy = r_rate.clone();
                let s_tau = mad_scale(&mut r_tau_copy);
                let s_rate = mad_scale(&mut r_rate_copy);

                // 重み更新
                let mut new_solves: Vec<BaselineSolve> = Vec::with_capacity(solves.len());
                for (i, s) in solves.iter().enumerate() {
                    let u_tau = (r_tau[i] / s_tau).abs();
                    let u_rate = (r_rate[i] / s_rate).abs();
                    let w_rt = if kind_l == "huber" {
                        let wt = if u_tau <= c { 1.0 } else { c / u_tau };
                        let wr = if u_rate <= c { 1.0 } else { c / u_rate };
                        (wt, wr)
                    } else {
                        // tukey
                        let wt = if u_tau < c {
                            let t = 1.0 - (u_tau / c) * (u_tau / c);
                            t * t
                        } else {
                            0.0
                        };
                        let wr = if u_rate < c {
                            let t = 1.0 - (u_rate / c) * (u_rate / c);
                            t * t
                        } else {
                            0.0
                        };
                        (wt, wr)
                    };
                    let w_tau = (s.w_tau * w_rt.0.max(1e-6)).max(1e-12);
                    let w_rate = (s.w_rate * w_rt.1.max(1e-6)).max(1e-12);
                    // phase weightも同様に縮退を避けるための下限を設けつつスケール
                    let w_phase = (s.w_phase * ((w_rt.0 + w_rt.1) * 0.5).max(1e-6)).max(1e-12);
                    new_solves.push(BaselineSolve {
                        a: s.a,
                        b: s.b,
                        tau_s: s.tau_s,
                        rate_hz: s.rate_hz,
                        w_tau,
                        w_rate,
                        phase_rad: s.phase_rad,
                        w_phase,
                    });
                }
                sol = solve_antenna_tau_rate(num_ant, cli.reference, &new_solves);
                // 以降の表示・位相解には更新後の重みを使う
                solves = new_solves;
            }
        }
        // --model を適用した場合、出力は（モデル + 残差）= 総和解にする
        if cli.model.is_some() {
            if let Some((tau0, rate0)) = model_map
                .get(&(l1 as i64))
                .cloned()
                .or_else(|| model_default.clone())
            {
                for ant in 0..num_ant.min(tau0.len()) {
                    sol.tau_s[ant] += tau0[ant];
                }
                for ant in 0..num_ant.min(rate0.len()) {
                    sol.rate_hz[ant] += rate0[ant];
                }
            }
        }

        let utc = win_time.unwrap_or_else(|| chrono::Utc::now()).to_rfc3339();
        // バイナリ: solutions.bin へ書き込み（RAW）
        let unix_sec = win_time.unwrap_or_else(|| chrono::Utc::now()).timestamp();
        write_i32_le(&mut sol_f, l1 as i32)?;
        write_i64_le(&mut sol_f, unix_sec)?;
        write_f32_le(&mut sol_f, (length_sectors as f32) * ts)?;
        for v in &sol.tau_s {
            write_f64_le(&mut sol_f, *v)?;
        }
        for v in &sol.rate_hz {
            write_f64_le(&mut sol_f, *v)?;
        }

        // バイナリ: baselines.bin へ書き込み（a,b,tau,rate,snr,w_tau,w_rate）
        for bj in &baselines_json {
            let a = bj["a"].as_u64().unwrap_or(0) as u16;
            let b = bj["b"].as_u64().unwrap_or(0) as u16;
            let tau = bj["tau_s"].as_f64().unwrap_or(0.0);
            let rate = bj["rate_hz"].as_f64().unwrap_or(0.0);
            let snr = bj["snr"].as_f64().unwrap_or(0.0) as f32;
            let wt = bj.get("weight_tau").and_then(|x| x.as_f64()).unwrap_or(0.0);
            let wr = bj
                .get("weight_rate")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0);
            use byteorder::WriteBytesExt;
            bln_f.write_u16::<LittleEndian>(a)?;
            bln_f.write_u16::<LittleEndian>(b)?;
            write_f64_le(&mut bln_f, tau)?;
            write_f64_le(&mut bln_f, rate)?;
            write_f32_le(&mut bln_f, snr)?;
            write_f64_le(&mut bln_f, wt)?;
            write_f64_le(&mut bln_f, wr)?;
        }

        // テキスト/JSON 出力: スムージング指定時は即時出力せずに蓄積
        if smooth_kind != SmoothKind::None {
            win_solutions.push(WinSol {
                t_idx: l1 as i32,
                unix_sec,
                len_s: (length_sectors as f32) * ts,
                tau: sol.tau_s.clone(),
                rate: sol.rate_hz.clone(),
                utc: utc.clone(),
            });
            continue;
        }
        if cli.no_stdout {
            continue;
        }
        if cli.pretty {
            // Build diagnostics
            // Connectivity from solves (final weights)
            let mut adj: std::collections::HashMap<usize, Vec<usize>> =
                std::collections::HashMap::new();
            for s in &solves {
                adj.entry(s.a).or_default().push(s.b);
                adj.entry(s.b).or_default().push(s.a);
            }
            let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
            let mut st = vec![cli.reference];
            while let Some(u) = st.pop() {
                if visited.insert(u) {
                    if let Some(nei) = adj.get(&u) {
                        for &v in nei {
                            if !visited.contains(&v) {
                                st.push(v);
                            }
                        }
                    }
                }
            }
            let baselines_used: Vec<&BaselineSolve> = solves
                .iter()
                .filter(|s| visited.contains(&s.a) && visited.contains(&s.b))
                .collect();
            let used = baselines_used.len();
            let ignored = solves.len().saturating_sub(used);
            // Build B and cond estimate
            let mut ants_in_comp: Vec<usize> = visited.iter().cloned().collect();
            ants_in_comp.sort_unstable();
            let mut col_map: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            let mut idx = 0usize;
            for &a in &ants_in_comp {
                if a == cli.reference {
                    continue;
                }
                col_map.insert(a, idx);
                idx += 1;
            }
            let n_unknowns = col_map.len();
            let m_rows = used;
            let mut b = DMatrix::<f64>::zeros(m_rows, n_unknowns.max(1));
            for (i, s) in baselines_used.iter().enumerate() {
                if let Some(&jp) = col_map.get(&s.a) {
                    b[(i, jp)] += 1.0;
                }
                if let Some(&jq) = col_map.get(&s.b) {
                    b[(i, jq)] -= 1.0;
                }
            }
            let mut bw = DMatrix::<f64>::zeros(m_rows, n_unknowns.max(1));
            for i in 0..m_rows {
                let wi = baselines_used[i].w_tau.max(1e-20).sqrt();
                for j in 0..n_unknowns {
                    bw[(i, j)] = wi * b[(i, j)];
                }
            }
            let svd = SVD::new(bw.clone(), true, false);
            let sv = svd.singular_values;
            let mut cond_txt = String::from("n/a");
            if sv.len() >= 1 && n_unknowns >= 1 {
                let mut smax = 0.0f64;
                let mut smin = f64::INFINITY;
                for v in sv.iter().cloned() {
                    if v > smax {
                        smax = v;
                    }
                    if v > 1e-14 && v < smin {
                        smin = v;
                    }
                }
                if smin.is_finite() && smax.is_finite() && smin > 0.0 {
                    let c = smax / smin;
                    cond_txt = format!("{:.3e}", c);
                }
            }
            let dof = (m_rows as isize) - (n_unknowns as isize);

            // Residuals and RMSE
            let mut r_tau_s: Vec<f64> = Vec::with_capacity(used);
            let mut r_rate_ss: Vec<f64> = Vec::with_capacity(used);
            for s in &baselines_used {
                // 残差RMSEは残差解で評価（--model適用時も同一基準で比較）
                let dt = sol_res.tau_s[s.a] - sol_res.tau_s[s.b];
                let dr = sol_res.rate_hz[s.a] - sol_res.rate_hz[s.b];
                r_tau_s.push(s.tau_s - dt);
                r_rate_ss.push(rate_to_ss(s.rate_hz - dr, obs_freq_hz));
            }
            let rmse = |v: &Vec<f64>| -> f64 {
                if v.is_empty() {
                    0.0
                } else {
                    let ss: f64 = v.iter().map(|x| x * x).sum();
                    (ss / (v.len() as f64)).sqrt()
                }
            };
            let rmse_tau_s = rmse(&r_tau_s);
            let rmse_rate_ss = rmse(&r_rate_ss);

            // Print pretty block
            writeln!(
                out,
                "t_idx={} utc={} len_s={:.3}",
                l1,
                utc,
                (length_sectors as f32) * ts
            )?;
            if obs_freq_hz > 0.0 {
                writeln!(out, "observing_frequency={:.6} Hz (rate=s/s)", obs_freq_hz)?;
            }
            // Antenna mapping
            let mut names_line = String::new();
            let mut keys: Vec<usize> = ant_names.keys().cloned().collect();
            keys.sort_unstable();
            for k in keys {
                let name = ant_names.get(&k).unwrap();
                names_line.push_str(&format!(
                    " {}={}{}",
                    k,
                    name,
                    if k == cli.reference { "(ref)" } else { "" }
                ));
            }
            writeln!(out, "Antennas:{}", names_line)?;
            // Connected/condition diagnostics
            // (already printed above; avoid duplicate)
            writeln!(
                out,
                "Residual RMSE: tau={:.3e} s, rate={:.3e} s/s",
                rmse_tau_s, rmse_rate_ss
            )?;
            if !sign_flips.is_empty() {
                let flips_str = sign_flips
                    .iter()
                    .map(|(a, b)| format!("{}:{}", a, b))
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(out, "Sign flips: {}", flips_str)?;
            }
            // Connected/condition diagnostics
            writeln!(
                out,
                "Connected: ants={} used_baselines={} ignored={} dof={} cond={}",
                ants_in_comp.len(),
                used,
                ignored,
                dof,
                cond_txt
            )?;

            // Antenna phase (global) via angular synchronization
            let phi_ant = solve_antenna_phase(num_ant, cli.reference, &solves);
            // Closure phase stats over available triangles
            let mut phase_map: std::collections::HashMap<(usize, usize), f64> =
                std::collections::HashMap::new();
            // Prefer peak-based phase saved in baselines_json; fallback to stored mean phase
            if !baselines_json.is_empty() {
                for bj in &baselines_json {
                    let a = bj["a"].as_u64().unwrap_or(0) as usize;
                    let b = bj["b"].as_u64().unwrap_or(0) as usize;
                    if !visited.contains(&a) || !visited.contains(&b) {
                        continue;
                    }
                    if let Some(pd) = bj.get("phase_deg_peak").and_then(|x| x.as_f64()) {
                        phase_map.insert((a, b), pd.to_radians());
                    }
                }
            }
            if phase_map.is_empty() {
                for s in &solves {
                    if s.w_phase > 0.0 && visited.contains(&s.a) && visited.contains(&s.b) {
                        phase_map.insert((s.a, s.b), s.phase_rad);
                    }
                }
            }
            let wrap = |x: f64| -> f64 {
                let mut y = (x + std::f64::consts::PI) % (2.0 * std::f64::consts::PI);
                if y < 0.0 {
                    y += 2.0 * std::f64::consts::PI;
                }
                y - std::f64::consts::PI
            };
            let mut tri_count = 0usize;
            let mut abs_sum = 0.0f64;
            let mut abs_max = 0.0f64;
            let mut tri_cp_012: Option<f64> = None;
            for i1 in 0..ants_in_comp.len() {
                for i2 in (i1 + 1)..ants_in_comp.len() {
                    for i3 in (i2 + 1)..ants_in_comp.len() {
                        let a = ants_in_comp[i1];
                        let b = ants_in_comp[i2];
                        let c = ants_in_comp[i3];
                        let phi_ab = if let Some(&p) = phase_map.get(&(a, b)) {
                            p
                        } else if let Some(&p) = phase_map.get(&(b, a)) {
                            -p
                        } else {
                            continue;
                        };
                        let phi_bc = if let Some(&p) = phase_map.get(&(b, c)) {
                            p
                        } else if let Some(&p) = phase_map.get(&(c, b)) {
                            -p
                        } else {
                            continue;
                        };
                        let phi_ca = if let Some(&p) = phase_map.get(&(c, a)) {
                            p
                        } else if let Some(&p) = phase_map.get(&(a, c)) {
                            -p
                        } else {
                            continue;
                        };
                        let cp_all = wrap(phi_ab + phi_bc + phi_ca);
                        let cp = cp_all.abs();
                        let cp_deg = cp.to_degrees();
                        tri_count += 1;
                        abs_sum += cp_deg;
                        if cp_deg > abs_max {
                            abs_max = cp_deg;
                        }
                        if num_ant == 3 && a == 0 && b == 1 && c == 2 {
                            tri_cp_012 = Some(cp_all);
                        }
                    }
                }
            }
            if tri_count > 0 {
                writeln!(
                    out,
                    "Closure phase: triangles={} mean_abs={:.3} deg max_abs={:.3} deg",
                    tri_count,
                    abs_sum / (tri_count as f64),
                    abs_max
                )?;
                if cli.closure_top > 0 {
                    // collect all closure values with triangles
                    let mut items: Vec<(usize, usize, usize, f64)> = Vec::new();
                    for i1 in 0..ants_in_comp.len() {
                        for i2 in (i1 + 1)..ants_in_comp.len() {
                            for i3 in (i2 + 1)..ants_in_comp.len() {
                                let a = ants_in_comp[i1];
                                let b = ants_in_comp[i2];
                                let c = ants_in_comp[i3];
                                let phi_ab = if let Some(&p) = phase_map.get(&(a, b)) {
                                    p
                                } else if let Some(&p) = phase_map.get(&(b, a)) {
                                    -p
                                } else {
                                    continue;
                                };
                                let phi_bc = if let Some(&p) = phase_map.get(&(b, c)) {
                                    p
                                } else if let Some(&p) = phase_map.get(&(c, b)) {
                                    -p
                                } else {
                                    continue;
                                };
                                let phi_ca = if let Some(&p) = phase_map.get(&(c, a)) {
                                    p
                                } else if let Some(&p) = phase_map.get(&(a, c)) {
                                    -p
                                } else {
                                    continue;
                                };
                                let cp = wrap(phi_ab + phi_bc + phi_ca).abs().to_degrees();
                                items.push((a, b, c, cp));
                            }
                        }
                    }
                    items.sort_by(|x, y| y.3.partial_cmp(&x.3).unwrap());
                    let topn = items.into_iter().take(cli.closure_top);
                    writeln!(out, "Worst closure triangles (deg):")?;
                    for (a, b, c, cp) in topn {
                        writeln!(out, "  ({},{},{}) -> {:.3}", a, b, c, cp)?;
                    }
                }
            }
            // Auto fixed-redundant check for pretty output (by station names)
            {
                // build name-of-id map
                let mut id2name: std::collections::HashMap<usize, String> =
                    std::collections::HashMap::new();
                for (id, name) in &ant_names {
                    id2name.insert(*id, name.clone());
                }
                // helper normalize
                let norm = |a: &str, b: &str| -> (String, String) {
                    if a <= b {
                        (a.to_string(), b.to_string())
                    } else {
                        (b.to_string(), a.to_string())
                    }
                };
                // auto groups
                let mut groups: Vec<((String, String), (String, String))> = Vec::new();
                let present: std::collections::HashSet<String> =
                    id2name.values().cloned().collect();
                let has = |s: &str| -> bool { present.contains(s) };
                if cli.redundant_fixed.is_empty() {
                    if has("YAMAGU32") && has("YAMAGU34") && has("HITACH32") {
                        groups.push((norm("YAMAGU32", "HITACH32"), norm("YAMAGU34", "HITACH32")));
                        writeln!(
                            out,
                            "Auto redundant fixed: YAMAGU32-HITACH32 : YAMAGU34-HITACH32"
                        )?;
                    }
                    if has("YAMAGU32") && has("YAMAGU34") && has("TAKAHA32") {
                        groups.push((norm("YAMAGU32", "TAKAHA32"), norm("YAMAGU34", "TAKAHA32")));
                        writeln!(
                            out,
                            "Auto redundant fixed: YAMAGU32-TAKAHA32 : YAMAGU34-TAKAHA32"
                        )?;
                    }
                } else {
                    for g in &cli.redundant_fixed {
                        if let Some((p1, p2)) = g.split_once(':') {
                            if let Some((a, b)) = p1.split_once('-') {
                                if let Some((c, d)) = p2.split_once('-') {
                                    groups
                                        .push((norm(a.trim(), b.trim()), norm(c.trim(), d.trim())));
                                }
                            }
                        }
                    }
                }
                if !groups.is_empty() {
                    // iterate triangles and report matches
                    for i1 in 0..ants_in_comp.len() {
                        for i2 in (i1 + 1)..ants_in_comp.len() {
                            for i3 in (i2 + 1)..ants_in_comp.len() {
                                let a = ants_in_comp[i1];
                                let b = ants_in_comp[i2];
                                let c = ants_in_comp[i3];
                                let (an, bn, cn) = (
                                    ant_names.get(&a).unwrap(),
                                    ant_names.get(&b).unwrap(),
                                    ant_names.get(&c).unwrap(),
                                );
                                let tri_pairs = [norm(an, bn), norm(bn, cn), norm(cn, an)];
                                let phi_ab = if let Some(&p) = phase_map.get(&(a, b)) {
                                    p
                                } else if let Some(&p) = phase_map.get(&(b, a)) {
                                    -p
                                } else {
                                    continue;
                                };
                                let phi_bc = if let Some(&p) = phase_map.get(&(b, c)) {
                                    p
                                } else if let Some(&p) = phase_map.get(&(c, b)) {
                                    -p
                                } else {
                                    continue;
                                };
                                let phi_ca = if let Some(&p) = phase_map.get(&(c, a)) {
                                    p
                                } else if let Some(&p) = phase_map.get(&(a, c)) {
                                    -p
                                } else {
                                    continue;
                                };
                                let cpv = wrap(phi_ab + phi_bc + phi_ca);
                                for (p1, p2) in &groups {
                                    let mut matched = 0;
                                    for tp in &tri_pairs {
                                        if tp == p1 || tp == p2 {
                                            matched += 1;
                                        }
                                    }
                                    if matched == 2 {
                                        let nr = if tri_pairs[0] != *p1 && tri_pairs[0] != *p2 {
                                            &tri_pairs[0]
                                        } else if tri_pairs[1] != *p1 && tri_pairs[1] != *p2 {
                                            &tri_pairs[1]
                                        } else {
                                            &tri_pairs[2]
                                        };
                                        // find ids for names
                                        let ida = id2name
                                            .iter()
                                            .find(|(_, &ref s)| s == nr.0.as_str())
                                            .map(|(k, _v)| *k);
                                        let idb = id2name
                                            .iter()
                                            .find(|(_, &ref s)| s == nr.1.as_str())
                                            .map(|(k, _v)| *k);
                                        if let (Some(ia), Some(ib)) = (ida, idb) {
                                            let phi_nr = if let Some(&p) = phase_map.get(&(ia, ib))
                                            {
                                                p
                                            } else if let Some(&p) = phase_map.get(&(ib, ia)) {
                                                -p
                                            } else {
                                                continue;
                                            };
                                            let mut d = ((cpv - phi_nr) + std::f64::consts::PI)
                                                % (2.0 * std::f64::consts::PI);
                                            if d < 0.0 {
                                                d += 2.0 * std::f64::consts::PI;
                                            }
                                            d -= std::f64::consts::PI;
                                            let diff = d.to_degrees().abs();
                                            writeln!(out, "Fixed-redundant match in tri ({},{},{}): cp={:.3} deg, phi({}-{})={:.3} deg, |diff|={:.3} deg", an,bn,cn, cpv.to_degrees(), nr.0, nr.1, phi_nr.to_degrees(), diff)?;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Accumulate time-series closure and uv for 3 antennas (for time-series fitting)
            if num_ant == 3 {
                if let (Some(cp_obs), Some(t0)) = (tri_cp_012, win_time) {
                    if let (Some((u01, v01)), Some((u12, v12)), Some((u20, v20))) = (
                        compute_uv_for_pair(&entries, &ant_names, 0, 1, t0),
                        compute_uv_for_pair(&entries, &ant_names, 1, 2, t0),
                        compute_uv_for_pair(&entries, &ant_names, 2, 0, t0),
                    ) {
                        fit_obs_closure.push(cp_obs);
                        fit_uv_01.push((u01, v01));
                        fit_uv_12.push((u12, v12));
                        fit_uv_20.push((u20, v20));
                    }
                }
            }
            // NOTE: Single-window ad-hoc double fit removed; use time-series fit in upcoming --fit-model.
            writeln!(
                out,
                "idx  name                tau [s]         rate [s/s]       phase [deg]"
            )?;
            writeln!(
                out,
                "---- ------------------- --------------- --------------- ---------------"
            )?;
            for ant in 0..num_ant {
                let name = ant_names
                    .get(&ant)
                    .cloned()
                    .unwrap_or_else(|| "?".to_string());
                let mark = if ant == cli.reference { " (ref)" } else { "" };
                writeln!(
                    out,
                    "{:<4}{:<19}{:>15.6e} {:>15.6e} {:>15.3}",
                    ant,
                    format!("{}{}", name, mark),
                    sol.tau_s[ant],
                    rate_to_ss(sol.rate_hz[ant], obs_freq_hz),
                    phi_ant[ant].to_degrees()
                )?;
            }
        } else if cli.plain {
            let tau_str = sol
                .tau_s
                .iter()
                .map(|v| format!("{:.9e}", v))
                .collect::<Vec<_>>()
                .join(",");
            let rate_str = sol
                .rate_hz
                .iter()
                .map(|v| format!("{:.9e}", rate_to_ss(*v, obs_freq_hz)))
                .collect::<Vec<_>>()
                .join(",");
            writeln!(
                out,
                "t_idx={} utc={} len_s={:.3} tau_s=[{}] rate_ss=[{}]",
                l1,
                utc,
                (length_sectors as f32) * ts,
                tau_str,
                rate_str
            )?;
        } else {
            // Recompute diagnostics for JSON
            let mut adj: std::collections::HashMap<usize, Vec<usize>> =
                std::collections::HashMap::new();
            for s in &solves {
                adj.entry(s.a).or_default().push(s.b);
                adj.entry(s.b).or_default().push(s.a);
            }
            let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
            let mut st = vec![cli.reference];
            while let Some(u) = st.pop() {
                if visited.insert(u) {
                    if let Some(nei) = adj.get(&u) {
                        for &v in nei {
                            if !visited.contains(&v) {
                                st.push(v);
                            }
                        }
                    }
                }
            }
            let baselines_used: Vec<&BaselineSolve> = solves
                .iter()
                .filter(|s| visited.contains(&s.a) && visited.contains(&s.b))
                .collect();
            let used = baselines_used.len();
            let ignored = solves.len().saturating_sub(used);
            let mut ants_in_comp: Vec<usize> = visited.iter().cloned().collect();
            ants_in_comp.sort_unstable();
            let mut col_map: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            let mut idx = 0usize;
            for &a in &ants_in_comp {
                if a == cli.reference {
                    continue;
                }
                col_map.insert(a, idx);
                idx += 1;
            }
            let n_unknowns = col_map.len();
            let m_rows = used;
            let mut b = DMatrix::<f64>::zeros(m_rows, n_unknowns.max(1));
            for (i, s) in baselines_used.iter().enumerate() {
                if let Some(&jp) = col_map.get(&s.a) {
                    b[(i, jp)] += 1.0;
                }
                if let Some(&jq) = col_map.get(&s.b) {
                    b[(i, jq)] -= 1.0;
                }
            }
            let mut bw = DMatrix::<f64>::zeros(m_rows, n_unknowns.max(1));
            for i in 0..m_rows {
                let wi = baselines_used[i].w_tau.max(1e-20).sqrt();
                for j in 0..n_unknowns {
                    bw[(i, j)] = wi * b[(i, j)];
                }
            }
            let svd = SVD::new(bw.clone(), true, false);
            let sv = svd.singular_values;
            let mut cond_val: Option<f64> = None;
            if sv.len() >= 1 && n_unknowns >= 1 {
                let mut smax = 0.0;
                let mut smin = f64::INFINITY;
                for v in sv.iter().cloned() {
                    if v > smax {
                        smax = v;
                    }
                    if v > 1e-14 && v < smin {
                        smin = v;
                    }
                }
                if smin.is_finite() && smax.is_finite() && smin > 0.0 {
                    cond_val = Some(smax / smin);
                }
            }
            let dof = (m_rows as isize) - (n_unknowns as isize);
            // Closure stats
            let mut phase_map: std::collections::HashMap<(usize, usize), f64> =
                std::collections::HashMap::new();
            for s in &solves {
                if s.w_phase > 0.0 && visited.contains(&s.a) && visited.contains(&s.b) {
                    phase_map.insert((s.a, s.b), s.phase_rad);
                }
            }
            let wrap = |x: f64| -> f64 {
                let mut y = (x + std::f64::consts::PI) % (2.0 * std::f64::consts::PI);
                if y < 0.0 {
                    y += 2.0 * std::f64::consts::PI;
                }
                y - std::f64::consts::PI
            };
            let mut tri_count = 0usize;
            let mut abs_sum = 0.0f64;
            let mut abs_max = 0.0f64;
            for i1 in 0..ants_in_comp.len() {
                for i2 in (i1 + 1)..ants_in_comp.len() {
                    for i3 in (i2 + 1)..ants_in_comp.len() {
                        let a = ants_in_comp[i1];
                        let b = ants_in_comp[i2];
                        let c = ants_in_comp[i3];
                        let phi_ab = if let Some(&p) = phase_map.get(&(a, b)) {
                            p
                        } else if let Some(&p) = phase_map.get(&(b, a)) {
                            -p
                        } else {
                            continue;
                        };
                        let phi_bc = if let Some(&p) = phase_map.get(&(b, c)) {
                            p
                        } else if let Some(&p) = phase_map.get(&(c, b)) {
                            -p
                        } else {
                            continue;
                        };
                        let phi_ca = if let Some(&p) = phase_map.get(&(c, a)) {
                            p
                        } else if let Some(&p) = phase_map.get(&(a, c)) {
                            -p
                        } else {
                            continue;
                        };
                        let cpv = wrap(phi_ab + phi_bc + phi_ca);
                        let cp = cpv.abs().to_degrees();
                        tri_count += 1;
                        abs_sum += cp;
                        if cp > abs_max {
                            abs_max = cp;
                        }
                    }
                }
            }

            // Optionally collect worst closure triangles
            let worst_json = if cli.closure_top > 0 {
                let mut items: Vec<(usize, usize, usize, f64)> = Vec::new();
                for i1 in 0..ants_in_comp.len() {
                    for i2 in (i1 + 1)..ants_in_comp.len() {
                        for i3 in (i2 + 1)..ants_in_comp.len() {
                            let a = ants_in_comp[i1];
                            let b = ants_in_comp[i2];
                            let c = ants_in_comp[i3];
                            let phi_ab = if let Some(&p) = phase_map.get(&(a, b)) {
                                p
                            } else if let Some(&p) = phase_map.get(&(b, a)) {
                                -p
                            } else {
                                continue;
                            };
                            let phi_bc = if let Some(&p) = phase_map.get(&(b, c)) {
                                p
                            } else if let Some(&p) = phase_map.get(&(c, b)) {
                                -p
                            } else {
                                continue;
                            };
                            let phi_ca = if let Some(&p) = phase_map.get(&(c, a)) {
                                p
                            } else if let Some(&p) = phase_map.get(&(a, c)) {
                                -p
                            } else {
                                continue;
                            };
                            let cp = wrap(phi_ab + phi_bc + phi_ca).abs().to_degrees();
                            items.push((a, b, c, cp));
                        }
                    }
                }
                items.sort_by(|x, y| y.3.partial_cmp(&x.3).unwrap());
                let arr: Vec<serde_json::Value> = items
                    .into_iter()
                    .take(cli.closure_top)
                    .map(|(a, b, c, cp)| serde_json::json!({"tri":[a,b,c],"abs_deg":cp}))
                    .collect();
                Some(arr)
            } else {
                None
            };

            let rate_ss: Vec<f64> = sol
                .rate_hz
                .iter()
                .map(|v| rate_to_ss(*v, obs_freq_hz))
                .collect();

            let antennas_json: Vec<serde_json::Value> = sol
                .tau_s
                .iter()
                .enumerate()
                .map(|(idx, &tau)| {
                    let name = ant_names
                        .get(&(idx as usize))
                        .cloned()
                        .unwrap_or_else(|| "?".to_string());
                    serde_json::json!({
                        "id": idx,
                        "name": name,
                        "tau_s": tau,
                        "rate_hz": sol.rate_hz[idx],
                        "rate_s_per_s": rate_ss[idx],
                    })
                })
                .collect();

            let mut obj = serde_json::json!({
                "t_idx": l1,
                "utc": utc,
                "tau_s": sol.tau_s,
                "rate_s_per_s": rate_ss,
                "rate_hz": sol.rate_hz,
                "observing_frequency_hz": obs_freq_hz,
                "win_len_s": (length_sectors as f32) * ts,
                "reference": cli.reference,
                "antennas": antennas_json,
                "diagnostics": {
                    "used_baselines": used,
                    "ignored_baselines": ignored,
                    "dof": dof,
                    "cond": cond_val,
                    "closure": {"triangles": tri_count, "mean_abs_deg": if tri_count>0 { abs_sum/(tri_count as f64) } else { 0.0 }, "max_abs_deg": abs_max},
                    "sign_flips": sign_flips.iter().map(|(a,b)| vec![*a as i64, *b as i64]).collect::<Vec<_>>()
                }
            });
            // Fixed redundant groups (by station names), check cp ~= phi(nonredundant) for matching triangles
            {
                // helper: normalize name pair (order-insensitive)
                fn norm_pair(a: &str, b: &str) -> (String, String) {
                    if a <= b {
                        (a.to_string(), b.to_string())
                    } else {
                        (b.to_string(), a.to_string())
                    }
                }
                // build name-of-id map
                let mut id2name: std::collections::HashMap<usize, String> =
                    std::collections::HashMap::new();
                for (id, name) in &ant_names {
                    id2name.insert(*id, name.clone());
                }
                // parse fixed groups
                let mut groups: Vec<((String, String), (String, String))> = Vec::new();
                if !cli.redundant_fixed.is_empty() {
                    for g in &cli.redundant_fixed {
                        if let Some((p1, p2)) = g.split_once(':') {
                            if let Some((a, b)) = p1.split_once('-') {
                                if let Some((c, d)) = p2.split_once('-') {
                                    groups.push((
                                        norm_pair(a.trim(), b.trim()),
                                        norm_pair(c.trim(), d.trim()),
                                    ));
                                }
                            }
                        }
                    }
                } else {
                    // auto-generate for common 4-antenna setup
                    let present: std::collections::HashSet<String> =
                        id2name.values().cloned().collect();
                    let has = |s: &str| -> bool { present.contains(s) };
                    if has("YAMAGU32") && has("YAMAGU34") && has("HITACH32") {
                        groups.push((
                            norm_pair("YAMAGU32", "HITACH32"),
                            norm_pair("YAMAGU34", "HITACH32"),
                        ));
                        if cli.pretty {
                            writeln!(
                                out,
                                "Auto redundant fixed: YAMAGU32-HITACH32 : YAMAGU34-HITACH32"
                            )?;
                        }
                    }
                    if has("YAMAGU32") && has("YAMAGU34") && has("TAKAHA32") {
                        groups.push((
                            norm_pair("YAMAGU32", "TAKAHA32"),
                            norm_pair("YAMAGU34", "TAKAHA32"),
                        ));
                        if cli.pretty {
                            writeln!(
                                out,
                                "Auto redundant fixed: YAMAGU32-TAKAHA32 : YAMAGU34-TAKAHA32"
                            )?;
                        }
                    }
                }
                // iterate triangles
                let mut report_fixed: Vec<serde_json::Value> = Vec::new();
                for i1 in 0..ants_in_comp.len() {
                    for i2 in (i1 + 1)..ants_in_comp.len() {
                        for i3 in (i2 + 1)..ants_in_comp.len() {
                            let a = ants_in_comp[i1];
                            let b = ants_in_comp[i2];
                            let c = ants_in_comp[i3];
                            let an = id2name.get(&a);
                            let bn = id2name.get(&b);
                            let cn = id2name.get(&c);
                            if an.is_none() || bn.is_none() || cn.is_none() {
                                continue;
                            }
                            let (an, bn, cn) = (an.unwrap(), bn.unwrap(), cn.unwrap());
                            let tri_pairs =
                                [norm_pair(an, bn), norm_pair(bn, cn), norm_pair(cn, an)];
                            // compute cp (radians)
                            let phi_ab = if let Some(&p) = phase_map.get(&(a, b)) {
                                p
                            } else if let Some(&p) = phase_map.get(&(b, a)) {
                                -p
                            } else {
                                continue;
                            };
                            let phi_bc = if let Some(&p) = phase_map.get(&(b, c)) {
                                p
                            } else if let Some(&p) = phase_map.get(&(c, b)) {
                                -p
                            } else {
                                continue;
                            };
                            let phi_ca = if let Some(&p) = phase_map.get(&(c, a)) {
                                p
                            } else if let Some(&p) = phase_map.get(&(a, c)) {
                                -p
                            } else {
                                continue;
                            };
                            let cpv = wrap(phi_ab + phi_bc + phi_ca);
                            // check each fixed group
                            for (p1, p2) in &groups {
                                let mut matched = 0;
                                for tp in &tri_pairs {
                                    if tp == p1 || tp == p2 {
                                        matched += 1;
                                    }
                                }
                                if matched == 2 {
                                    // find nonredundant pair in this triangle
                                    let nr = if tri_pairs[0] != *p1 && tri_pairs[0] != *p2 {
                                        &tri_pairs[0]
                                    } else if tri_pairs[1] != *p1 && tri_pairs[1] != *p2 {
                                        &tri_pairs[1]
                                    } else {
                                        &tri_pairs[2]
                                    };
                                    // map nr name-pair back to ids (order as given in name pair)
                                    let (na, nb) = (nr.0.as_str(), nr.1.as_str());
                                    // find ids for names
                                    let ida = id2name
                                        .iter()
                                        .find(|(_, &ref s)| s == na)
                                        .map(|(k, _v)| *k);
                                    let idb = id2name
                                        .iter()
                                        .find(|(_, &ref s)| s == nb)
                                        .map(|(k, _v)| *k);
                                    if let (Some(ia), Some(ib)) = (ida, idb) {
                                        let phi_nr = if let Some(&p) = phase_map.get(&(ia, ib)) {
                                            p
                                        } else if let Some(&p) = phase_map.get(&(ib, ia)) {
                                            -p
                                        } else {
                                            continue;
                                        };
                                        let diff = {
                                            let mut d = ((cpv - phi_nr) + std::f64::consts::PI)
                                                % (2.0 * std::f64::consts::PI);
                                            if d < 0.0 {
                                                d += 2.0 * std::f64::consts::PI;
                                            }
                                            d -= std::f64::consts::PI;
                                            d.to_degrees().abs()
                                        };
                                        if cli.pretty {
                                            writeln!(out, "Fixed-redundant match in tri ({},{},{}): cp={:.3} deg, phi({}-{})={:.3} deg, |diff|={:.3} deg", an,bn,cn, cpv.to_degrees(), na, nb, phi_nr.to_degrees(), diff)?;
                                        }
                                        report_fixed.push(serde_json::json!({"tri":[an,bn,cn],"cp_deg":cpv.to_degrees(),"nr_pair":[na,nb],"phi_deg":phi_nr.to_degrees(),"diff_deg":diff}));
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(map) = obj.as_object_mut() {
                    if let Some(diag) = map
                        .get_mut("diagnostics")
                        .and_then(|d| d.as_object())
                        .cloned()
                    {
                        let mut d = diag;
                        if !report_fixed.is_empty() {
                            d.insert(
                                "redundant_fixed".to_string(),
                                serde_json::Value::Array(report_fixed),
                            );
                        }
                        map.insert("diagnostics".to_string(), serde_json::Value::Object(d));
                    }
                }
            }
            // (threshold-based redundant diagnosis removed; use fixed redundant groups if provided)
            if let Some(arr) = worst_json {
                if let Some(map) = obj.as_object_mut() {
                    if let Some(diag) = map
                        .get_mut("diagnostics")
                        .and_then(|d| d.as_object())
                        .cloned()
                    {
                        let mut d = diag;
                        d.insert("closure_worst".to_string(), serde_json::Value::Array(arr));
                        map.insert("diagnostics".to_string(), serde_json::Value::Object(d));
                    }
                }
            }
            if cli.dump_baseline {
                if let Some(map) = obj.as_object_mut() {
                    map.insert(
                        "baselines".to_string(),
                        serde_json::Value::Array(baselines_json),
                    );
                }
            }
            if !cli.pretty {
                write_json_pretty(&mut *out, &obj)?;
            }
        }
    }
    // スムージングの適用と後処理（--cor モード）
    if smooth_kind != SmoothKind::None {
        // series を平滑化
        let win_n = win_solutions.len();
        if win_n == 0 {
            return Ok(());
        }
        let ant_n = win_solutions[0].tau.len();
        // MA: 中心窓（端は可変幅）。SG: 局所最小二乗（端は可変窓）
        let wlen = cli.smooth_len.max(1);
        let poly = cli.smooth_poly.max(0);

        fn smooth_ma(series: &Vec<f64>, w: usize) -> Vec<f64> {
            let n = series.len();
            let k = w / 2;
            let mut out = vec![0.0; n];
            for i in 0..n {
                let s = i.saturating_sub(k);
                let e = min(n - 1, i + k);
                let mut sum = 0.0;
                let mut cnt = 0usize;
                for j in s..=e {
                    sum += series[j];
                    cnt += 1;
                }
                out[i] = if cnt > 0 {
                    sum / (cnt as f64)
                } else {
                    series[i]
                };
            }
            out
        }
        fn smooth_sg(series: &Vec<f64>, w: usize, p: usize) -> Vec<f64> {
            let n = series.len();
            let mut out = vec![0.0; n];
            for i in 0..n {
                let k = w / 2;
                let l = i.saturating_sub(k);
                let r = min(n - 1, i + k);
                // センターを i として局所多項式回帰（x=0 を中心）
                let m = p + 1;
                let mut ata = vec![vec![0.0f64; m]; m];
                let mut atb = vec![0.0f64; m];
                for j in l..=r {
                    let x = (j as isize - i as isize) as f64;
                    // ベクトル v = [1, x, x^2, ...]
                    let mut v = vec![1.0f64; m];
                    for d in 1..m {
                        v[d] = v[d - 1] * x;
                    }
                    let y = series[j];
                    for a in 0..m {
                        atb[a] += v[a] * y;
                        for b in 0..m {
                            ata[a][b] += v[a] * v[b];
                        }
                    }
                }
                // 解く（LU）
                let ata_m = DMatrix::<f64>::from_row_slice(
                    m,
                    m,
                    &ata.iter()
                        .flat_map(|r| r.iter())
                        .cloned()
                        .collect::<Vec<_>>(),
                );
                let atb_v = DVector::<f64>::from_row_slice(&atb);
                let coeff = ata_m.lu().solve(&atb_v).unwrap_or(DVector::<f64>::zeros(m));
                out[i] = coeff[0]; // x=0 の推定値
            }
            out
        }
        // 平滑化を適用
        let mut smoothed: Vec<WinSol> = win_solutions.clone();
        for ant in 0..ant_n {
            let tau_series: Vec<f64> = win_solutions.iter().map(|w| w.tau[ant]).collect();
            let rate_series: Vec<f64> = win_solutions.iter().map(|w| w.rate[ant]).collect();
            let tau_s = match smooth_kind {
                SmoothKind::MA => smooth_ma(&tau_series, wlen),
                SmoothKind::SG => smooth_sg(&tau_series, wlen, poly),
                SmoothKind::None => tau_series,
            };
            let rate_s = match smooth_kind {
                SmoothKind::MA => smooth_ma(&rate_series, wlen),
                SmoothKind::SG => smooth_sg(&rate_series, wlen, poly),
                SmoothKind::None => rate_series,
            };
            for i in 0..win_n {
                smoothed[i].tau[ant] = tau_s[i];
                smoothed[i].rate[ant] = rate_s[i];
            }
        }

        // バイナリ solutions_smooth.bin を必要に応じて出力
        let so = cli.smooth_output.to_lowercase();
        if so == "smooth" || so == "both" {
            let sol_path_smooth = out_dir.join("solutions_smooth.bin");
            let mut sf = BufWriter::new(File::create(&sol_path_smooth)?);
            sf.write_all(b"gsol")?; // magic
            write_u32_le(&mut sf, 1)?; // version
            write_u32_le(&mut sf, win_n as u32)?; // num_windows
            write_u32_le(&mut sf, ant_n as u32)?; // num_ant
            for (i, w) in smoothed.iter().enumerate() {
                write_i32_le(&mut sf, i as i32)?;
                write_i64_le(&mut sf, w.unix_sec)?;
                write_f32_le(&mut sf, w.len_s)?;
                for v in &w.tau {
                    write_f64_le(&mut sf, *v)?;
                }
                for v in &w.rate {
                    write_f64_le(&mut sf, *v)?;
                }
            }
        }

        // テキスト出力（smoothed 値）
        if !cli.no_stdout {
            if cli.pretty {
                // Antenna mapping once
                let mapping = ant_names
                    .iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("# Antennas: {}", mapping);
                for (_i, w) in smoothed.iter().enumerate() {
                    writeln!(
                        out,
                        "t_idx={} utc={} len_s={:.3} (smoothed)",
                        w.t_idx, w.utc, w.len_s
                    )?;
                    writeln!(out, "idx  name                tau [s]         rate [s/s]")?;
                    writeln!(
                        out,
                        "---- ------------------- --------------- ---------------"
                    )?;
                    for ant in 0..ant_n {
                        let name = ant_names
                            .get(&(ant as usize))
                            .cloned()
                            .unwrap_or_else(|| "?".to_string());
                        let mark = if ant == cli.reference { " (ref)" } else { "" };
                        writeln!(
                            out,
                            "{:<4}{:<19}{:>15.6e} {:>15.6e}",
                            ant,
                            format!("{}{}", name, mark),
                            w.tau[ant],
                            rate_to_ss(w.rate[ant], obs_freq_hz)
                        )?;
                    }
                }
            } else if cli.plain {
                for w in &smoothed {
                    let tau_str = w
                        .tau
                        .iter()
                        .map(|v| format!("{:.9e}", v))
                        .collect::<Vec<_>>()
                        .join(",");
                    let rate_str = w
                        .rate
                        .iter()
                        .map(|v| format!("{:.9e}", rate_to_ss(*v, obs_freq_hz)))
                        .collect::<Vec<_>>()
                        .join(",");
                    writeln!(
                        out,
                        "t_idx={} utc={} len_s={:.3} tau_s=[{}] rate_ss=[{}] (smoothed)",
                        w.t_idx, w.utc, w.len_s, tau_str, rate_str
                    )?;
                }
            } else {
                for w in &smoothed {
                    let rate_ss: Vec<f64> = w
                        .rate
                        .iter()
                        .map(|v| rate_to_ss(*v, obs_freq_hz))
                        .collect();
                    let antennas_json: Vec<serde_json::Value> = w
                        .tau
                        .iter()
                        .enumerate()
                        .map(|(idx, &tau)| {
                            let name = ant_names
                                .get(&(idx as usize))
                                .cloned()
                                .unwrap_or_else(|| "?".to_string());
                            serde_json::json!({
                                "id": idx,
                                "name": name,
                                "tau_s": tau,
                                "rate_hz": w.rate[idx],
                                "rate_s_per_s": rate_ss[idx],
                            })
                        })
                        .collect();
                    write_json_pretty(
                        &mut *out,
                        &serde_json::json!({
                            "t_idx": w.t_idx,
                            "utc": w.utc,
                            "tau_s": w.tau,
                            "rate_s_per_s": rate_ss,
                            "rate_hz": w.rate,
                            "observing_frequency_hz": obs_freq_hz,
                            "win_len_s": w.len_s,
                            "reference": cli.reference,
                            "antennas": antennas_json,
                            "smoothed": true
                        }),
                    )?;
                }
            }
        }
    }

    // 簡易構造フィット（3局のみ対応の骨格）：double（core+1点）
    // Time-series structure fit (double) over all windows for 3 antennas
    if cli.fit_model.to_lowercase() == "double" && num_ant == 3 && !fit_obs_closure.is_empty() {
        let to_rad = std::f64::consts::PI / (180.0 * 3600.0 * 1000.0);
        let range = (cli.fit_range_mas as f64) * to_rad;
        let grid = cli.fit_grid.max(3);
        let wrap = |x: f64| -> f64 {
            let mut y = (x + std::f64::consts::PI) % (2.0 * std::f64::consts::PI);
            if y < 0.0 {
                y += 2.0 * std::f64::consts::PI;
            }
            y - std::f64::consts::PI
        };
        let mut best = (f64::INFINITY, 0.0f64, 0.0f64, 0.0f64);
        for ix in 0..grid {
            for iy in 0..grid {
                for &a in &[0.05, 0.1, 0.2, 0.3, 0.5] {
                    let dl = -range + 2.0 * range * (ix as f64) / (grid as f64 - 1.0);
                    let dm = -range + 2.0 * range * (iy as f64) / (grid as f64 - 1.0);
                    let mut cost = 0.0;
                    for i in 0..fit_obs_closure.len() {
                        let (u01, v01) = fit_uv_01[i];
                        let (u12, v12) = fit_uv_12[i];
                        let (u20, v20) = fit_uv_20[i];
                        let ph = |u: f64, v: f64| -2.0 * std::f64::consts::PI * (u * dl + v * dm);
                        let v01 = C64::<f64>::new(1.0, 0.0)
                            + C64::<f64>::new(0.0, ph(u01, v01)).exp() * a;
                        let v12 = C64::<f64>::new(1.0, 0.0)
                            + C64::<f64>::new(0.0, ph(u12, v12)).exp() * a;
                        let v20 = C64::<f64>::new(1.0, 0.0)
                            + C64::<f64>::new(0.0, ph(u20, v20)).exp() * a;
                        let cp_mod = (v01 * v12 * v20).arg();
                        let d = wrap(cp_mod - fit_obs_closure[i]);
                        cost += 1.0 - d.cos();
                    }
                    if cost < best.0 {
                        best = (cost, dl, dm, a);
                    }
                }
            }
        }
        if !cli.no_stdout {
            let mas = |x: f64| x * (180.0 * 3600.0 * 1000.0) / std::f64::consts::PI;
            println!(
                "#FIT double: dl={:.3} mas dm={:.3} mas a={:.3} cost={:.3}",
                mas(best.1),
                mas(best.2),
                best.3,
                best.0
            );
        }
    }

    Ok(())
}
