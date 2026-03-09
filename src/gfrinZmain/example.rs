pub fn print_examples() {
    let txt = r#"
gfrinZ 使い方サマリ（完全同期前提・antenna-based 集約）

【一読要約 / まず何ができるか】
- 何を得るか: アンテナ起因のパラメータ（遅延 tau_s[秒], レート rate_hz[Hz], 位相 phase[deg]）を窓ごとに推定。加えて診断（条件数 cond, 自由度 DoF, クロージャーフェーズ統計, 残差RMSE）を出力。
- どう解析するか: 基線ごとにフリンジ探索 →（任意）モデル位相除去 → 重み付き最小二乗でアンテナ遅延/レート → 角度同期でアンテナ位相 →（任意）ロバスト再重み付け/時間スムージング → JSON/テキスト/バイナリ出力。
- いつ使うか: 初期キャリブレーション、フリンジ検出/追尾、時計ずれ・基準遅延の推定、データ品質評価（closure/cond）。
- 入力の種類: 
  - --cor（通常）: .cor を直接読み取り、基線探索から実施（同期前提）。
  - --input: 既に推定された基線測定（JSONL）を再集約（t_idx ごとにアンテナ解）。

基本方針
- 各基線（.cor）でフリンジサーチ（frinZ と同等）を実行し、τ[秒]/ρ[Hz]/SNR を取得。
- アンテナベース最小二乗（参照アンテナ固定）で各窓ごとの解を出力（JSONL）。
- 入力は完全同期（同一セクタindexの時刻が一致）であることを前提。
- FFT 点数やサンプリング周波数の不一致は許容（基線ごとにネイティブ解析）。

オプション早見表（主なもの）
- --antennas <N>: アンテナ数（必須）。--cor ではIDの最大値+1で自動可。--input は必ず指定。
- --reference <IDX>: 参照アンテナ（既定0）。相対原点。条件数や残差が悪ければ変更検討。
- --cor A:B:FILE: 基線の .cor を複数指定（例 0:1:AB.cor）。完全同期（同一indexの時刻一致）が前提。
- 時系列窓: --length <秒>, --skip <秒>, --loop <回>（0=自動）
- 探索制御: --search（ピーク精密化）、--delay-window a b（サンプル単位）、--rate-window a b（Hz）、--rate-padding 2^k
- 前処理: --rfi min,max（MHz, 複数可）, --bandpass FILE（FFT/2 長と一致必要）
- 重み: 既定 w=SNR^2。--bw-weight で (BW/BW_ref)^2 を乗算。--crlb-weight で近似CRLB（τとρ別々）
- ロバスト: --robust huber|tukey, --robust-c, --robust-iters（既定2）
- 位相: --phase-kind delay|frequency（基線位相の由来を切替）
- 符号自動整合: --auto-flip off|header|triangle（既定 header）
    - header: .cor の station1/2 とアンテナID対応から A:B が逆なら自動反転（N>=3対応）
    - triangle: 3局で 0->1->2->0 の向きに合わせて反転（N>3は未適用）
- FRNMOD: --model FILE（JSONL: {t_idx?, tau_s[], rate_hz[]}）で位相傾斜の事前除去
- スムーズ: --smooth none|ma|sgolay, --smooth-len, --smooth-poly, --smooth-output raw|smooth|both
- 出力: --pretty（推奨の表形式）, --plain（1行）, JSON（既定）、--dump-baseline（中間も）
- 解析補助: --closure-top N（closure の悪い三角形をN件表示）

出力（JSONL）
- 1行=1窓。キー: t_idx, utc, tau_s[Na], rate_hz[Na], win_len_s, （--dump-baseline時 baselines[]）

入力の2モード（--cor と --input）
- --cor モード（通常利用）
  - .cor を直接読み、基線ごとにフリンジサーチしてからアンテナ集約します。
  - 完全同期前提（同一indexのセクタ時刻一致）。

- --input モード（既存の基線推定を再集約）
  - フォーマット: 1行=1基線の測定（JSON Lines）
    必須: a, b, tau_s(秒), rate_hz(Hz)
    任意: w_tau, w_rate もしくは sigma_tau_s, sigma_rate_hz もしくは snr
    任意: t_idx（同じ t_idx でグルーピングして一つのアンテナ解を出力）
  - 例:
    {"a":0,"b":1,"t_idx":0,"tau_s":1.2e-8,"rate_hz":-0.015,"snr":18.2}
    {"a":0,"b":2,"t_idx":0,"tau_s":1.1e-8,"rate_hz":-0.016,"w_tau":2.5e10,"w_rate":9.0e6}
    {"a":1,"b":2,"t_idx":0,"tau_s":-8.0e-10,"rate_hz":-0.001,"sigma_tau_s":2.0e-9,"sigma_rate_hz":0.002}
  - 実行: gfrinZ --antennas 3 --reference 0 --input baseline.jsonl > antenna.jsonl
  - 注意: --antennas は必須です。

sols.jsonl（アンテナ解）→ baseline.jsonl（基線測定）への変換
- gfrinZ の出力 sols.jsonl はアンテナ解です。そのまま --input には使えません。
- gfrinZ 実行時に --dump-baseline を付けると、各行に baselines 配列（各基線の τ̂/ρ̂/SNR/weight）が含まれます。
- jq を使って抽出:
  jq -c '. as $row | .baselines[] | {a,b,tau_s,rate_hz,snr,t_idx:$row.t_idx}' sols.jsonl > baseline.jsonl
  これを --input に渡せます。

代表的なコマンド / 目的別レシピ
1) 最低限：アンテナ遅延・レート（3基線、10秒窓）
  gfrinZ --antennas 3 --reference 0 \
    --cor 0:1:AB.cor --cor 0:2:AC.cor --cor 1:2:BC.cor \
    --length 10 --loop 12 > sols.jsonl

2) 精密探索（ピーク精密化・探索窓）
  gfrinZ ... --length 10 --loop 12 --search --rate-padding 2 \
    --delay-window -1024 1024 --rate-window -0.5 0.5 > sols.jsonl

3) RFI 除外とバンドパス補正
  gfrinZ ... --rfi 100,120 --rfi 400,500 --bandpass bp.bin > sols.jsonl
  ※ bp.bin の長さは FFT/2 と一致している必要があります。不一致なら警告してスキップ。

4) 重みの選択
  - 既定: w = SNR^2
  - 帯域重み: w = SNR^2 × (BW/BW_ref)^2
      gfrinZ ... --bw-weight
  - 近似CRLB重み: τとρで別々に w = 1/σ^2 を使用
      σ_tau ≈ 1/(2π SNR BW[Hz]), σ_rate ≈ 1/(2π SNR T[s])
      gfrinZ ... --crlb-weight

5) ロバスト（外れ値に強くしたい）
  - Huber
      gfrinZ ... --crlb-weight --robust huber --robust-iters 3
  - Tukey（強い外れ値抑制）
  gfrinZ ... --crlb-weight --robust tukey --robust-iters 3

6) 整形出力で人間が見やすく（テーブル）
  gfrinZ ... --pretty > sols.txt
  位相も見たい/closure も見たい → --phase-kind frequency --closure-top 5 を併用

7) プレーンテキストでの出力（1行/窓で簡易ログ）
  gfrinZ ... --plain > sols.txt
  出力例: t_idx=0 utc=... len_s=10.000 tau_s=[0.000000000e+00,-4.074661827e-10,...] rate_hz=[0.000000000e+00,-1.306154038e-03,...]

8) 見やすい整形テキスト（推奨）
  gfrinZ ... --pretty --phase-kind delay > sols.txt
  出力例:
    t_idx=0 utc=... len_s=10.000
    Antennas: 0=AAA(ref) 1=BBB 2=CCC
    Connected: ants=3 used_baselines=3 ignored=0 dof=2 cond=1.2e+03
    Residual RMSE: tau=0.123 ns, rate=0.450 mHz
    idx  name                tau [ns]        rate [mHz]
    ---- ------------------- --------------- ---------------
    0    AAA (ref)                 0.000000         0.000000
    1    BBB                      -0.407467        -1.306154
    2    CCC                       0.102345         0.210987

9) スムージング(MA or Savitzky-Golay)
  gfrinZ ... --length 10 --loop 12 --smooth ma --smooth-len 5 --pretty
  gfrinZ ... --length 10 --loop 12 --smooth sgolay --smooth-len 7 --smooth-poly 2 --pretty
  - バイナリは既定でRAW(solutions.bin). スムーズ値も保存する場合:
    gfrinZ ... --smooth sgolay --smooth-output both  # solutions_smooth.bin を併せて出力

10) モデル除算（FRNMOD相当）と位相選択・クロージャー詳細
  # 既存のアンテナ解（JSONL: {t_idx?, tau_s[], rate_hz[]}）をモデルとして除算
  gfrinZ ... --model model.jsonl --phase-kind frequency --closure-top 5 --pretty
  # N>=3 でファイルの station1/2 と A:B の向きが混在する場合に自動整合
  gfrinZ ... --auto-flip header --pretty

【目的別のオプション組み合わせガイド】
- アンテナ遅延/レートを安定に推定したい（最低限）
  --search --crlb-weight --pretty
- 位相も含めて品質評価（closure）を行いたい
  --phase-kind frequency --closure-top 10 --pretty
- 外れ値に強くしたい（不良基線混在）
  --crlb-weight --robust huber --robust-iters 3 （または --robust tukey）
- ノイズを抑え，時間的になめらかな解が欲しい
  --smooth sgolay --smooth-len 7 --smooth-poly 2 （整形出力 or solutions_smooth.bin）
- 既知のモデルで位相傾斜を除去してから推定したい
  --model model.jsonl 併用（参照/符号系を要確認）

運用上の注意
- 完全同期前提：同じセクタindexでのセクタ時刻が一致しないとエラーになります。
- FFT・帯域の不一致は許容。基線ごとの推定は連続量（秒・Hz）として最終LSに投入。
- 中間結果が必要なら --dump-baseline を付けて分析パイプラインに供給可能。
 - スムージングは --cor モードでのみ適用されます（--input では無視）。

ヒント
- 解が不安定な場合:
  1) --search でピーク精密化
  2) --crlb-weight で帯域・積分時間の効果を明示
  3) --robust huber/tukey で外れ値を緩和
  4) RFI 除外や BP 補正を適用
\n\n===============================================================
Global fringe の考え方・解析手順・注意点（詳説）
===============================================================

1. 目的と全体像
- 目的: 基線ごとの測定値（遅延 tau_pq, レート rho_pq, 位相 phi_pq）から、アンテナごとの量（tau_p, rho_p, phi_p）を復元する。
- 背景: 基線 p–q の測定は差分量（tau_pq = tau_p − tau_q 等）なので、適切な参照アンテナを固定しないと一意に決まらない。
- AIPS FRING の対応関係（概念）
  - FRNSEL: データ選択 → gfrinZ では .cor 読み込みと同期・窓の定義
  - FRNMOD: モデル除算 → gfrinZ では --model による位相傾斜の除去
  - FRNSOL: 解く（最小二乗） → gfrinZ の global LS（tau/rate）と位相の角度同期
  - FRNADJ: 解の整形/平滑化 → gfrinZ の --smooth で窓方向平滑化
  - FRNAPL: 適用 → 本ツールでは解の出力が主。適用は別ツールや後段へ
  - FRNHIS: 履歴 → 本ツールは JSON/テキスト/バイナリ出力で記録

2. gfrinZ の処理パイプライン（--cor モード）
(1) データ同期と窓
- 全 .cor の指定窓（--length, --skip, --loop）について、各窓の先頭セクタ時刻が一致することを検証（完全同期前提）。

(2) 基線ごとのフリンジ探索
- FFT→IFFT により遅延×レート平面を形成。RFI 除外 (--rfi) とバンドパス補正 (--bandpass) に対応。
- ピーク検出はグローバルまたは探索窓 (--delay-window, --rate-window) に限定可能。
- --search でピーク近傍の2次近似。帯域・積分長の違いは基線ごとにネイティブに扱う。
- 得られる量: tau_pq[秒], rho_pq[Hz], SNR, 位相（delay_phase または freq_phase）。

(3) モデル除算（FRNMOD 相当, optional）
- --model で与えたアンテナ解（tau_s[], rate_hz[]）から基線ごとの位相傾斜 exp{-i 2π (f·Δτ + t·Δρ)} を各窓で除去。
- Δτ = tau_a − tau_b, Δρ = rate_a − rate_b を使用。t_idx 指定があればその行を、なければ t_idx 無し行をデフォルトとして使用。
- 注意: モデルの参照アンテナや符号系が gfrinZ と一致していることを確認。

(4) 重み付け
- 既定: w = SNR^2。--bw-weight で (BW/BW_ref)^2 を掛ける。
- --crlb-weight では近似 CRLB に基づき w_tau = 1/σ_tau^2, w_rate = 1/σ_rate^2。
  - 目安: σ_tau ≈ 1/(2π·SNR·BW[Hz]), σ_rate ≈ 1/(2π·SNR·T[s])
- 位相 w_phase も SNR^2 を既定とする（位相に自信がない場合は0にして無効化）。

(5) ロバスト再重み付け (optional)
- --robust huber|tukey により IRLS を2回程度（--robust-iters）実施。
- 現在の解から基線残差を作り、スケール（MAD）で正規化して重みを再計算。

(6) グローバル解（tau/rate）
- tau/rate は y = B x（B は行列、x はアンテナ量、y は基線測定）の形で、参照アンテナ列を除外した重み付き正規方程式 B^T W B x = B^T W y を解く。
- 連結成分: 参照アンテナを含む成分のみで解く。参照が成分外の場合は最大成分に自動アンカ（最小IDアンテナ）して解く（警告）。
- 診断: 自由度 DoF = m − n、条件数 cond ≈ cond(sqrt(W)·B)、残差 RMSE など。

(7) 位相（アンテナ位相）
- 角度同期（angular synchronization）により e^{i φ_pq} から g_p を推定（主固有ベクトル近似）、参照アンテナに相対化して φ_p を得る。
- --phase-kind delay|frequency で基線位相の由来を切替。

(8) クロージャーフェーズ
- 任意の三角形 (a,b,c) について φ_ab + φ_bc + φ_ca を [-π,π] に折り畳み、絶対値を指標とする。
- 出力: triangles, mean_abs_deg, max_abs_deg。--closure-top N で上位 N 三角形を列挙。

(9) 窓方向スムージング（FRNADJ 最小版）
- --smooth none|ma|sgolay で tau/rate を時系列平滑化。端点は可変窓で処理。
- スムーズ値は整形出力と solutions_smooth.bin（--smooth-output smooth|both）に反映。RAW は solutions.bin に保存。

(10) 出力
- 整形テキスト (--pretty): 1窓ごとに表形式（tau[ns], rate[mHz], phase[deg]）と診断・クロージャースタッツ。
- プレーンテキスト (--plain): 1行/窓。
- JSONL（既定）: 1行/窓。--cor の場合は diagnostics を埋め込み可能（cond, DoF, closure 等）。
- 基線中間結果は --dump-baseline で JSON の baselines に含められる。

3. 注意点と落とし穴
- 同期: .cor は完全同期が前提（同一 index のセクタ時刻が全基線一致）。
- 連結性: アンテナグラフが連結でない場合、参照成分外は解けない。最小限 N アンテナには N−1 本の独立基線が必要。
- 参照アンテナ選択: 参照は相対位相・遅延の原点。条件数と残差の振る舞いを確認し、必要なら変更。
- 重み: SNR^2 と CRLB は近似。データ条件に依存し、偏りや過信のリスクがあるため、ロバスト化を併用推奨。
- 位相の扱い: wrap に注意。クロージャーが大きい三角形は外れや系統誤差の示唆。--closure-top で特定して除外/再解析の手掛かりに。
- FRNMOD: モデルの参照や符号が一致しないと逆に悪化。t_idx の対応も確認。
- バンドパス・RFI: BP 長は FFT/2 と一致が必要。RFI 窓の選び方でピークの健全性が変化。
- スムージング: 窓長の選択はトレードオフ（ノイズ低減 vs 応答遅れ）。端点の歪みに注意。
- 数値: 条件数が高いと解が不安定。IRLS と正しい重みで安定化を図る。DoF や cond を監視。
- 単位と符号: tau[秒], rate[Hz], phase[rad/deg] を混同しない。基線→アンテナの符号規約を統一。

4. 実用レシピ
- 最小構成（3基線）: --search --pretty で挙動確認 → cond と RMSE を見る。
- RFI/BP: --rfi と --bandpass で前処理。BP 長チェックの警告を確認。
- 重みとロバスト: --crlb-weight と --robust huber|tukey を併用。外れに強い解。
- 位相とクロージャー: --phase-kind frequency を試し、--closure-top で悪い三角形を特定。
- スムージング: --smooth sgolay --smooth-len 7 --smooth-poly 2 程度から。滑らかさと遅れのバランスを評価。
- モデル除算: 既存解を --model で除去してから再解析。ピーク・残差・closure の改善を確認。

5. FRING.FOR との対応表（概念）
- FRNSEL → .cor の同期・窓定義
- FRNMOD → --model による位相傾斜除去
- FRNSOL → tau/rate の weighted LS, phase の角度同期
- FRNADJ → --smooth による時系列平滑化
- FRNAPL → 別ツールでの適用（本ツールは解・診断を出力）
- FRNHIS → JSONL/テキスト/バイナリ出力が記録

6. JSON スキーマ例（--cor の1行）
{
  "t_idx": 0,
  "utc": "2025-09-07T12:34:56Z",
  "tau_s": [ ... ],
  "rate_hz": [ ... ],
  "win_len_s": 10.0,
  "diagnostics": {
    "used_baselines": 3,
    "ignored_baselines": 0,
    "dof": 2,
    "cond": 1.2e+03,
    "closure": { "triangles": 1, "mean_abs_deg": 0.5, "max_abs_deg": 0.5 },
    "closure_worst": [ { "tri": [0,1,2], "abs_deg": 0.5 } ]
  },
  "baselines": [ ... ]
}

7. トラブルシュート
- cond が大きい: 参照の再選択、データの追加、RFI 除外強化、ロバスト化で緩和。
- closure が大きい: 問題三角形を特定（--closure-top）、該当基線の下流処理見直し（BP/RFI/モデル）。
- 位相が不安定: --phase-kind を切替、探索窓・スムージング・重みの見直し。
- 解が飛ぶ: 同期・窓範囲・基線の欠損を確認（ログと警告）。

===============================================================
"#;
    println!("{}", txt);
}
