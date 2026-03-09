use anyhow::{anyhow, Result};
use clap::{CommandFactory, Parser};
use std::path::PathBuf;

mod known_pulsar;
mod shared;
mod unknown_pulsar;

#[derive(Parser, Debug)]
#[command(
    name = "pulsar_gating",
    about = "GICO3 出力の .cor 相関データを解析し、パルサー信号を抽出するゲーティングツール",
    long_about = "GICO3 が出力した .cor 相関データを読み込み、周波数チャネル毎のスペクトルを時間方向に並べた行列を構築します。\n\
    オプションで分散補正（DM）を適用した後に折り畳み処理（fold）を行い、オンパルス／オフパルスの推定、\n\
    各種時系列やスペクトルのプロット、レポート出力を実施します。\n\
    frinZ は .cor を解析するツールです。"
)]
struct Cli {
    /// 入力する .cor ファイルのパス
    #[arg(
        long,
        value_name = "FILE",
        help = "GICO3 出力の .cor 相関データファイルを指定します。",
        long_help = "GICO3 出力の .cor 相関データファイルを指定します。ファイルパスは絶対パス・相対パスいずれでも構いません。\n\
        出力は入力ファイルと同じディレクトリ直下の frinZ/pulsar_gating/ に保存されます。"
    )]
    input: PathBuf,

    /// パルサーの回転周期 [秒]
    #[arg(
        long,
        value_name = "SECONDS",
        help = "既知のパルサー周期を指定します（任意指定）。未指定の場合は未知モードで解析します。",
        long_help = "既知のパルサーの回転周期（秒単位）です。観測データと整合する実測値もしくはカタログ値を指定してください。\n\
        未指定の場合は周期未知モードとして解析フローを切り替えます。"
    )]
    period: Option<f64>,

    /// 分散測定量 DM [pc cm^-3]
    #[arg(
        long,
        value_name = "DM",
        help = "周波数チャネル間の遅延を補正するための分散測定量（任意指定）。",
        long_help = "周波数チャネル毎に発生する分散遅延を補正するための分散測定量 DM（pc cm^-3）です。\n\
        既知モードでは指定することで遅延補正を行います。未知モードでは自動推定を行うため通常は指定しません。"
    )]
    dm: Option<f64>,

    /// 折り畳み時の位相ビン数 [-]
    #[arg(
        long,
        default_value_t = 128,
        help = "折り畳みプロフィールを何分割するか（初期値 128）。",
        long_help = "折り畳み後のパルスプロフィールを分割する位相ビン数です。値を増やすと時間分解能が上がる一方で、\n\
        各ビンに含まれるサンプル数が減るためノイズが増える可能性があります。"
    )]
    bins: usize,

    /// 冒頭からスキップするセクター数 [PP]
    #[arg(
        long,
        default_value_t = 0,
        help = "先頭から無視するセクター（PP）の数です。",
        long_help = "先頭から処理対象外とするセクター（PP）の数です。立ち上がりの不安定区間を除外したい場合に利用してください。\n\
        1 セクターは .cor ヘッダー内の number_of_sector で定義された単位です。"
    )]
    skip: u32,

    /// 使用する最大セクター数（0 で全区間） [PP]
    #[arg(
        long,
        default_value_t = 0,
        help = "解析に使用するセクター数の上限を設定します（0 で全区間）。",
        long_help = "解析に使用するセクター数（PP）の上限を設定します。0 を指定するとファイル内の全セクターを読み込みます。\n\
        途中までのデータだけで評価したい場合に短い値を指定してください。"
    )]
    length: u32,

    /// オンパルスに割り当てる位相ビン割合 [-]
    #[arg(
        long,
        default_value_t = 0.1,
        help = "折り畳み後にオンパルスと見なす位相ビン数の割合です。",
        long_help = "折り畳み後のパルスプロフィールから自動的にオンパルス領域を推定する際、全ビン数に対してどの程度の割合を\n\
        オンパルスとみなすかを指定します。典型的には 0.05 ～ 0.2 程度が利用されます。"
    )]
    on_duty: f64,

    /// delay-rate 抽出で使う振幅しきい値
    #[arg(
        long,
        default_value_t = 0.015,
        help = "delay-rate 平面で抽出する最小振幅（amplitude >= threshold）。",
        long_help = "周期探索補助のため、delay-rate 平面から amplitude がこの値以上の点だけを抽出します。\n\
        未指定時は 0.015 を使用します。"
    )]
    amp_threshold: f64,

    /// 詳細な中間CSV/PNG出力を有効化
    #[arg(
        long,
        default_value_t = false,
        help = "詳細な中間生成物（CSV/PNG）を出力します（既定: 無効）。",
        long_help = "実行速度を優先するため、既定では必要最小限の出力のみを生成します。\n\
        このオプションを指定すると、詳細な中間生成物（CSV/PNG）も出力します。"
    )]
    full_output: bool,
}

fn main() -> Result<()> {
    if std::env::args_os().len() == 1 {
        Cli::command().print_help()?;
        println!();
        return Ok(());
    }
    let cli = Cli::parse();
    let Cli {
        input,
        period,
        dm,
        bins,
        skip,
        length,
        on_duty,
        amp_threshold,
        full_output,
    } = cli;

    if let Some(period) = period {
        let args = known_pulsar::KnownArgs {
            input,
            period,
            dm,
            bins,
            skip,
            length,
            on_duty,
            full_output,
        };
        known_pulsar::run(args)
    } else {
        if dm.is_some() {
            return Err(anyhow!(
                "--dm を指定する場合は --period も併せて入力してください（未知モードでは指定しません）。"
            ));
        }
        let args = unknown_pulsar::UnknownArgs {
            input,
            bins,
            skip,
            length,
            on_duty,
            amp_threshold,
            full_output,
        };
        unknown_pulsar::run(args)
    }
}
