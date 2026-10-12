//! The first screen is a processing catalogue. Option syntax/defaults come
//! from frinZ's actual clap command; the detailed manuals are embedded whole.
use eframe::egui::{self, RichText, ScrollArea};

const DETAIL: &str = include_str!("../frinZmain/command_detail.txt");
const README: &str = include_str!("../../README.md");
const MBCOR: &str = include_str!("../../docs/mbcor.md");

struct Topic {
    title: String,
    summary: String,
    options: Vec<String>,
    text: String,
}

pub struct HelpBrowser {
    pub open: bool,
    selected: Option<usize>,
    query: String,
    topics: Vec<Topic>,
}

type TopicSpec = (
    &'static str,
    &'static str,
    &'static [&'static str],
    &'static str,
    &'static [&'static str],
);

impl Default for HelpBrowser {
    fn default() -> Self {
        let specs: &[TopicSpec] = &[
            (
                "入力・積分区間",
                ".cor / .mbcorの読み込みと積分長・開始位置・繰り返し回数",
                &["input", "length", "skip", "loop"],
                "frinZ --input data.cor --length 60 --loop 10 --output",
                &["Single File Analysis"],
            ),
            (
                "フリンジ探索（遅延・レート）",
                "peak / deep / deep2 / coherent / rate / acelの探索",
                &[
                    "search",
                    "iter",
                    "drange",
                    "rrange",
                    "mask",
                    "rate-padding",
                    "cpu",
                ],
                "frinZ --input data.cor --search deep --length 60 --plot --output",
                &["Precise Search Mode", "Search Parameters"],
            ),
            (
                "周波数ドメイン解析",
                "周波数–レート平面を解析し、探索・表示範囲を指定",
                &["frequency", "frange"],
                "frinZ --input data.cor --frequency --plot --output",
                &["Frequency Domain Analysis"],
            ),
            (
                "in-beam VLBI",
                "in-beam用の遅延・レート解析と結果出力",
                &["in-beam"],
                "frinZ --input data.cor --in-beam --search peak --plot",
                &[],
            ),
            (
                "帯域分割フリンジ解析",
                "観測帯域を指定MHz幅で分割して解析",
                &["inband"],
                "frinZ --input data.cor --inband 128 --search peak --output",
                &["In-band Fringe Search"],
            ),
            (
                "遅延・レート・高次位相補正",
                "Delay / Rate / Acceleration / Jerk / Snapを事前補正",
                &[
                    "delay-correct",
                    "rate-correct",
                    "acel-correct",
                    "jerk-correct",
                    "snap-correct",
                ],
                "frinZ --input data.cor --delay-correct 10 --rate-correct 0.01 --plot",
                &["Manual Corrections"],
            ),
            (
                "スキャンごとの補正",
                "start, integ, delay[sample], rate[Hz]のCSVを適用",
                &["scan-correct"],
                "frinZ --input data.cor --scan-correct corrections.csv --output",
                &[],
            ),
            (
                "補正済み.corの生成",
                "手動補正と探索残差を全区間へ適用して保存",
                &["mkcor"],
                "frinZ --input data.cor --length 0 --search peak --mkcor",
                &["補正済み `.cor` の保存 (`--mkcor`)"],
            ),
            (
                "RFI除去・Rayleigh解析",
                "帯域指定・histogram・NPZマスクで干渉成分を除去",
                &["rfi"],
                "frinZ --input data.cor --rfi histogram count:1 bins:256 --plot",
                &["RFI Mitigation"],
            ),
            (
                "時間・セクターのフラグ",
                "time区間のスキップ、pp区間のゼロ詰め",
                &["flagging"],
                "frinZ --input data.cor --flagging time 10 20 --output",
                &[],
            ),
            (
                "バンドパス補正",
                "バンドパス表を適用して解析",
                &["bandpass"],
                "frinZ --input data.cor --bandpass bandpass.npz --plot --output",
                &["Bandpass Calibration"],
            ),
            (
                "バンドパス表の生成",
                "補正後スペクトルを圧縮NPZへ保存",
                &["bandpass-table"],
                "frinZ --input calibrator.cor --bandpass-table",
                &["Bandpass Calibration"],
            ),
            (
                "自己相関による振幅規格化",
                "同じディレクトリのACFを使い相互相関を規格化",
                &["norm-acf"],
                "frinZ --input data.cor --norm-acf --output",
                &[],
            ),
            (
                "YAMAGU34 spike補正",
                "spike検出、区間fit、補正前後スペクトル・残差を出力",
                &["spike34"],
                "frinZ --input data.cor --spike34 YAMAGU34_AUTO.cor --search peak --plot",
                &["YAMAGU34 spike correction diagnostics (`--spike34`)"],
            ),
            (
                "コンタミ解析用handoff",
                "flux向けの時刻・UVW・生複素可視度をNPZ v5へ保存",
                &["contamination"],
                "frinZ --input data.cor --contamination --output",
                &[
                    "Contamination handoff (`--contamination`)",
                    "Contamination handoff NPZ",
                ],
            ),
            (
                "コンタミモデル減算",
                "fluxの補正モデルをメモリ内適用し元.corを解析",
                &["contamination-subtract"],
                "frinZ --input data.cor --contamination-subtract model.npz --output",
                &["Contamination handoff (`--contamination`)"],
            ),
            (
                "クロススペクトル",
                "平均複素スペクトルを圧縮NPZへ保存",
                &["spectrum"],
                "frinZ --input data.cor --spectrum",
                &["Output File Formats"],
            ),
            (
                "生ビジビリティのBIN保存",
                "生複素可視度スペクトルをバイナリ出力",
                &["cor2bin"],
                "frinZ --input data.cor --cor2bin",
                &[],
            ),
            (
                "FFTチャンネル平均・縮小",
                "FFTチャンネルを平均してrebin",
                &["fft-rebin"],
                "frinZ --input data.cor --fft-rebin 1024 --output",
                &[],
            ),
            (
                "Short-time FFT",
                "指定hopで積分窓を移動してフリンジ解析",
                &["stfft"],
                "frinZ --input data.cor --length 60 --stfft 10 --loop 100 --plot",
                &["Precise Search Mode"],
            ),
            (
                "時系列グラフ・位相unwrap",
                "Amp / SNR / 位相 / 雑音 / Delay / Rateの変化",
                &["add-plot"],
                "frinZ --input data.cor --length 60 --loop 10 --add-plot",
                &["Plotting", "フリンジ位相の unwrap (`--add-plot`)"],
            ),
            (
                "WWZ時間周波数解析",
                "各loopの探索結果の時間周波数変動を解析",
                &["wwz"],
                "frinZ --input data.cor --length 60 --loop 100 --wwz",
                &[],
            ),
            (
                "積分時間とSNR",
                "積算時間によるSNR増加をグラフ・fitで確認",
                &["cumulate"],
                "frinZ --input data.cor --cumulate 600 --plot",
                &["Plotting"],
            ),
            (
                "Allan deviation",
                "loopごとのフリンジ位相から安定度を解析",
                &["allan-deviance"],
                "frinZ --input data.cor --length 60 --loop 100 --allan-deviance",
                &[],
            ),
            (
                "生ビジビリティのヒートマップ",
                "時間×周波数の振幅・位相と指定補正後の図",
                &["raw-visibility"],
                "frinZ --input data.cor --raw-visibility --npz",
                &[],
            ),
            (
                "Dynamic spectrum",
                "時間周波数スペクトルと時間遅延waterfall",
                &["dynamic-spectrum"],
                "frinZ --input data.cor --dynamic-spectrum --plot",
                &[],
            ),
            (
                "UV被覆",
                "2D / 3DのUVトラックを表示",
                &["uv"],
                "frinZ --input data.cor --uv 1",
                &[],
            ),
            (
                "アンテナの可視時間",
                "天体の方位・仰角と1日の観測可能時間",
                &["uptimeplot"],
                "frinZ --input data.cor --uptimeplot",
                &[],
            ),
            (
                "位相参照解析",
                "calibratorの位相変化からtargetの位相を補正",
                &["phase-reference"],
                "frinZ --phase-reference hybrid cal.cor target.cor 30 30 1 --plot --output",
                &[
                    "Phase Reference Analysis",
                    "Phase Reference with Custom Parameters",
                ],
            ),
            (
                "クロージャー位相",
                "3基線からクロージャー位相・bispectrumを計算",
                &["closure-phase"],
                "frinZ --closure-phase b12.cor b23.cor b13.cor --plot",
                &[],
            ),
            (
                "フリンジレートマップ",
                "遅延・レートの拘束から天球位置マップを作成",
                &["fringe-rate-map"],
                "frinZ --input data.cor --fringe-rate-map mode:cont range:auto --plot",
                &[],
            ),
            (
                "地球自転合成イメージング",
                "dirty / CLEAN画像を生成",
                &["imaging"],
                "frinZ --input data.cor --imaging size:256 clean:1 --plot",
                &[],
            ),
            (
                "メーザー解析",
                "OFF減算、基線除去、速度補正、Gaussian fit",
                &["maser"],
                "frinZ --input data.cor --maser off:linear rest:6668.5192 --plot",
                &[],
            ),
            (
                "パルス折り畳み",
                "周期・エフェメリス、barycentric・軌道補正を指定してfold",
                &["folding"],
                "frinZ --input data.cor --folding period:1.0 bins:64 --npz",
                &[],
            ),
            (
                "複数sidebandの解析",
                "C/Xの.cor、バンドパス表、遅延補正を指定して結合",
                &["multi-sideband"],
                "frinZ --multi-sideband C.cor C_BP.npz 0 X.cor X_BP.npz 0 --plot",
                &[],
            ),
            (
                "ヘッダー確認",
                "相関器・観測周波数・アンテナ・天体のヘッダーを表示",
                &["header"],
                "frinZ --input data.cor --header",
                &["Text Output"],
            ),
            (
                "解析結果・グラフ・FITS・NPZ出力",
                "各解析productの保存形式、列・単位、出力先",
                &["plot", "output", "npz", "fits"],
                "frinZ --input data.cor --search peak --plot --output --npz --fits",
                &[
                    "Text Output",
                    "Output Files",
                    "Output File Formats",
                    "出力ファイル名の補正接尾辞",
                ],
            ),
            (
                "詳細ガイド表示",
                "CLIで詳細ガイドを表示",
                &["detail"],
                "frinZ --detail",
                &[],
            ),
        ];
        let mut command = frinZ::args::Args::command_with_aliases();
        command.build();
        let mut topics = Vec::new();
        for &(title, summary, options, example, sections) in specs {
            let mut text = format!("{title}\n\n{summary}\n\n実行例（ファイル名・値は自身の観測へ置き換えてください）\n{example}\n\n現在の実装が受け付けるオプション\n");
            for &option in options {
                if let Some(arg) = command
                    .get_arguments()
                    .find(|arg| arg.get_long() == Some(option))
                {
                    text.push_str(&option_text(arg));
                    let mut keys = vec![option];
                    if let Some(aliases) = arg.get_all_aliases() {
                        keys.extend(aliases);
                    }
                    for block in detail_blocks(&keys) {
                        if !text.contains(block) {
                            text.push_str("\n詳細ガイドの原文\n");
                            text.push_str(block);
                            text.push('\n');
                        }
                    }
                }
            }
            for section in sections {
                if let Some(body) = readme_section(section) {
                    text.push_str("\nREADMEの該当仕様（原文）\n");
                    text.push_str(body);
                    text.push('\n');
                }
            }
            if options.contains(&"input") {
                text.push_str("\n.mbcor入力仕様（全文）\n");
                text.push_str(MBCOR);
            }
            topics.push(Topic {
                title: title.into(),
                summary: summary.into(),
                options: options.iter().map(|s| (*s).into()).collect(),
                text,
            });
        }
        // Future options appear automatically instead of silently disappearing
        // from a hand-maintained catalogue.
        for arg in command.get_arguments() {
            if let Some(long) = arg.get_long() {
                if !["help", "version"].contains(&long)
                    && !topics
                        .iter()
                        .any(|topic| topic.options.iter().any(|s| s == long))
                {
                    topics.push(Topic {
                        title: format!("--{long}"),
                        summary: arg.get_help().map(ToString::to_string).unwrap_or_default(),
                        options: vec![long.into()],
                        text: option_text(arg),
                    });
                }
            }
        }
        let all_options = command.get_arguments().map(option_text).collect::<String>();
        let full = format!("frinZmain 全仕様・全オプション\n\n現在のCLI定義\n{}\n\n全オプションの詳細・別名・既定値\n{all_options}\n\n--detail の全文\n{DETAIL}\n\nREADME 全文\n{README}\n\n.mbcor 入力仕様 全文\n{MBCOR}", command.render_long_help());
        topics.push(Topic {
            title: "全仕様書・全オプション（全文）".into(),
            summary: "詳細ガイド、全CLI定義、README、.mbcor仕様を省略せずに参照".into(),
            options: Vec::new(),
            text: full,
        });
        Self {
            open: false,
            selected: None,
            query: String::new(),
            topics,
        }
    }
}

fn option_text(arg: &clap::Arg) -> String {
    let mut text = format!(
        "\n{arg}\n{}\n",
        arg.get_long_help()
            .or_else(|| arg.get_help())
            .map(ToString::to_string)
            .unwrap_or_default()
    );
    if !arg.get_default_values().is_empty() {
        text.push_str(&format!(
            "既定値: {}\n",
            arg.get_default_values()
                .iter()
                .map(|v| v.to_string_lossy())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if let Some(aliases) = arg.get_all_aliases() {
        text.push_str(&format!(
            "別名: {}\n",
            aliases
                .iter()
                .map(|s| format!("--{s}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if let Some(values) = arg.get_value_parser().possible_values() {
        text.push_str(&format!(
            "選択肢: {}\n",
            values
                .map(|v| v.get_name().to_owned())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    text
}

fn detail_blocks<'a>(options: &[&str]) -> Vec<&'a str> {
    let mut blocks = Vec::new();
    let mut start = None;
    let mut offset = 0;
    for line in DETAIL.split_inclusive('\n') {
        if line.starts_with("  --") || line.starts_with('[') {
            if let Some(begin) = start.take() {
                let block = &DETAIL[begin..offset];
                if options.iter().any(|key| {
                    block
                        .lines()
                        .next()
                        .unwrap_or("")
                        .contains(&format!("--{key}"))
                }) {
                    blocks.push(block);
                }
            }
            if line.starts_with("  --") {
                start = Some(offset);
            }
        }
        offset += line.len();
    }
    if let Some(begin) = start {
        let block = &DETAIL[begin..];
        if options.iter().any(|key| {
            block
                .lines()
                .next()
                .unwrap_or("")
                .contains(&format!("--{key}"))
        }) {
            blocks.push(block);
        }
    }
    blocks
}

fn readme_section(title: &str) -> Option<&'static str> {
    let mut offset = 0;
    let mut start = None;
    let mut level = 0;
    let mut in_code = false;
    for line in README.split_inclusive('\n') {
        if line.starts_with("```") {
            in_code = !in_code;
        }
        if !in_code && line.starts_with('#') {
            let heading_level = line.chars().take_while(|&c| c == '#').count();
            if let Some(begin) = start {
                if heading_level <= level {
                    return Some(&README[begin..offset]);
                }
            }
            if line.trim_start_matches('#').trim() == title {
                start = Some(offset);
                level = heading_level;
            }
        }
        offset += line.len();
    }
    start.map(|start| &README[start..])
}

impl HelpBrowser {
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open {
            return;
        }
        egui::Window::new("ヘルプ — frinZmain の実行可能な処理")
            .open(&mut self.open).default_size([1060.0, 730.0]).min_size([700.0, 400.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("処理名・オプションを検索");
                    ui.text_edit_singleline(&mut self.query);
                    if ui.button("処理一覧に戻る").clicked() { self.selected = None; }
                    if let Some(selected) = self.selected {
                        if ui.button("詳細をコピー").clicked() { ctx.copy_text(self.topics[selected].text.clone()); }
                    }
                });
                ui.separator();
                let query = self.query.to_lowercase();
                if let Some(selected) = self.selected {
                    ui.heading(&self.topics[selected].title);
                    ScrollArea::both().id_salt(("help_detail", selected)).max_height(ui.available_height()).show(ui, |ui| {
                        let mut text = self.topics[selected].text.as_str();
                        ui.add(egui::TextEdit::multiline(&mut text).font(egui::TextStyle::Monospace).desired_width(ui.available_width()).desired_rows(30));
                    });
                } else {
                    ui.label("希望する処理をクリックすると、用途・実行例・オプション・関連仕様を表示します。");
                    ScrollArea::vertical().id_salt("help_catalogue").max_height(ui.available_height()).show(ui, |ui| {
                        for (i, topic) in self.topics.iter().enumerate() {
                            if !query.is_empty() && !format!("{} {} {}", topic.title, topic.summary, topic.options.join(" ")).to_lowercase().contains(&query) { continue; }
                            if ui.button(RichText::new(&topic.title).strong()).clicked() { self.selected = Some(i); }
                            ui.label(&topic.summary);
                            ui.separator();
                        }
                    });
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_click_opens_details_and_back_restores_list() {
        let ctx = egui::Context::default();
        let mut browser = HelpBrowser::default();
        browser.open = true;
        let frame = |browser: &mut HelpBrowser, events: Vec<egui::Event>| {
            ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1480.0, 940.0),
                    )),
                    events,
                    ..Default::default()
                },
                |ctx| browser.show(ctx),
            )
        };
        let label_position = |output: &egui::FullOutput, label: &str| {
            output
                .shapes
                .iter()
                .find_map(|shape| match &shape.shape {
                    egui::Shape::Text(text) if text.galley.job.text == label => {
                        Some(text.pos + text.galley.size() / 2.0)
                    }
                    _ => None,
                })
                .unwrap_or_else(|| panic!("Missing visible button: {label}"))
        };
        let click = |browser: &mut HelpBrowser, position: egui::Pos2| {
            for pressed in [true, false] {
                frame(
                    browser,
                    vec![
                        egui::Event::PointerMoved(position),
                        egui::Event::PointerButton {
                            pos: position,
                            button: egui::PointerButton::Primary,
                            pressed,
                            modifiers: egui::Modifiers::NONE,
                        },
                    ],
                );
            }
        };
        frame(&mut browser, vec![]);
        let output = frame(&mut browser, vec![]);
        click(&mut browser, label_position(&output, "入力・積分区間"));
        assert_eq!(browser.selected, Some(0));
        let output = frame(&mut browser, vec![]);
        assert!(output.shapes.iter().any(|shape| matches!(&shape.shape, egui::Shape::Text(text) if text.galley.job.text.contains("--input"))));
        click(&mut browser, label_position(&output, "処理一覧に戻る"));
        assert!(browser.selected.is_none());
        browser.query = "folding".into();
        let output = frame(&mut browser, vec![]);
        let folding = browser
            .topics
            .iter()
            .position(|topic| topic.options.contains(&"folding".into()))
            .unwrap();
        let position = label_position(&output, &browser.topics[folding].title);
        click(&mut browser, position);
        assert_eq!(browser.selected, Some(folding));
    }

    #[test]
    fn catalogue_covers_every_live_option_and_keeps_complete_manuals() {
        let browser = HelpBrowser::default();
        assert!(browser.selected.is_none());
        for arg in frinZ::args::Args::command_with_aliases().get_arguments() {
            if let Some(long) = arg.get_long() {
                assert!(
                    browser
                        .topics
                        .iter()
                        .any(|t| t.options.iter().any(|s| s == long)),
                    "missing {long}"
                );
            }
        }
        let full = &browser.topics.last().unwrap().text;
        assert!(full.contains(DETAIL));
        assert!(full.contains(README));
        assert!(full.contains(MBCOR));
        let folding = browser
            .topics
            .iter()
            .find(|t| t.options.contains(&"folding".into()))
            .unwrap();
        assert!(folding.text.contains("tasc_mjd"));
        let input = browser
            .topics
            .iter()
            .find(|t| t.options.contains(&"input".into()))
            .unwrap();
        assert!(input.text.contains(MBCOR));
    }
}
