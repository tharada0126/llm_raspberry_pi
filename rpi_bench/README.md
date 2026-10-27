# PC ⇔ Raspberry Pi 連携ベンチマーク

PC が Raspberry Pi 上の Ollama を操作してベンチマーク (HumanEval / GSM8K / MMLU-Pro) を実行し、
RPi の消費電力を FNIRSI USB パワーメータ (`../fnirsi-usb-power-data-logger`) で問題ごとに計測する。

| PC (`pc_controller.py`) | 通信 | RPi (`rpi_server.py`) |
|---|---|---|
| model 指定 | `POST /prepare` → | Ollama 準備 (モデルダウンロード、起動) |
| 動作確認 | `POST /check` → | Ollama 実行 |
| エネルギー計測開始 | | |
| ベンチマークのプロンプトを指示 | `POST /generate` → | Ollama 実行 |
| 結果受信 (1 つの JSON) | ← | 実行結果 |
| エネルギー計測終了 → 生データを gzip 保存 → エネルギー情報を JSON に追加 → jsonl に記録 → idling | | |
| 終了指示 | `POST /finish` → | モデル削除 |

## 構成

```
[RPi 用電源] ──> [FNIRSI メータ] ──> [Raspberry Pi]     (メータは RPi の給電経路に入れる)
                       │ データ用 USB
                      [PC] ── LAN / SSH ── [Raspberry Pi]
```

| ファイル | 実行場所 | 内容 |
|---|---|---|
| `pc_controller.py` | PC | 全体の制御 (上表の PC 側) |
| `power_meter.py` | PC | パワーメータの計測。`fnirsi_logger.py` のデバイス処理を再利用 |
| `bench_tasks.py` | PC | 問題の読み込み・プロンプト生成・採点 (`scripts/test_*.py` と同一) |
| `rpi_server.py` | RPi | Ollama 操作。ロード・推論・計測は `main.py` と同じ関数 (`ollama_utils`, `system_monitor`) を使用 |

## セットアップ

### Raspberry Pi

```bash
curl -fsSL https://ollama.com/install.sh | sh     # Ollama
git clone <このリポジトリ> ~/llm_ptme               # または --ssh_sync で必要ファイルだけ送る
python3 -m venv ~/venv && ~/venv/bin/pip install -r ~/llm_ptme/rpi_bench/requirements_rpi.txt
```

Ollama が systemd サービスとして動いていればそれを使い、動いていなければ `rpi_server.py` が `ollama serve` を起動する。

### PC

```bash
pip install -r rpi_bench/requirements_pc.txt
```

- パワーメータは Windows では WinUSB ドライバが必要 (Zadig で設定)。この PC では設定済みで動作確認済み。
- SSH は鍵認証にしておく (`ssh pi@raspberrypi.local` がパスワードなしで通る状態)。

## 実行

SSH モード (推奨): PC が SSH で RPi のサーバを起動し、ポートフォワード経由で通信する。
RPi のサーバは 127.0.0.1 でのみ待ち受けるので LAN にポートを開けない。実験終了時に SSH を閉じるとサーバも終了する。

```bash
python rpi_bench/pc_controller.py --ssh pi@raspberrypi.local \
    --ssh_remote_dir ~/llm_ptme --ssh_python ~/venv/bin/python
```

少数問題での確認:

```bash
python rpi_bench/pc_controller.py --ssh pi@raspberrypi.local --ssh_python ~/venv/bin/python \
    --models tinyllama:1.1b --datasets gsm8k --max_samples 2 --idle_sec 1
```

- `--ssh_sync`: 起動前に `rpi_server.py` と、それが使う `main.py` / `ollama_utils.py` / `system_monitor.py` を scp で RPi に送る (PC 側で編集した場合など)
- `--no_meter`: パワーメータなしで通信フローだけ確認する
- `--keep_model`: モデルごとの終了時にモデルを削除しない

直接接続モード: RPi で `python3 rpi_bench/rpi_server.py --host 0.0.0.0 --port 8000` を起動しておき、
PC で `--rpi_url http://<RPiのIP>:8000` を指定する (認証なしなので信頼できるネットワークでのみ使う)。

主なオプション (`--help` で全て表示):

| オプション | 既定値 |
|---|---|
| `--models` | tinyllama:1.1b qwen2.5:1.5b phi:2.7b phi3:3.8b gemma:2b mistral:7b |
| `--datasets` | human_eval gsm8k mmlu-pro |
| `--gsm8k_selected_idx_file` | `scripts/gsm8k_selected_idx.txt` (`none` で全問) |
| `--mmlu_pro_selected_ids_file` | `scripts/mmlu_pro_selected_ids_20.txt` (`none` で全問) |
| `--num_predict` / `--num_ctx` / `--num_gpu` / `--num_thread` | 512 / 4096 / 0 (CPU のみ) / Ollama 任せ (`main.py` と同じ) |
| `--think` / `--keep_alive` | auto / 60m (`main.py` と同じ) |
| `--idle_sec` | 5 (各問題後の待機) |
| `--scenario` / `--output_dir` | `RPi` / `results_rpi` |

問題データは既定で Hugging Face から取得する (`openai/openai_humaneval`, `openai/gsm8k`, `TIGER-Lab/MMLU-Pro`)。
ローカルファイルは `--human_eval_path`, `--gsm8k_path`, `--mmlu_pro_dir` で指定できる。
MMLU-Pro の `cot_prompt_lib/initial_prompt.txt` は `--mmlu_pro_dir` が無い場合、内蔵の同文を使う。

## 出力

```
results_rpi/<scenario>/
├── rpi_server.log                                  # SSH モード時の RPi 側ログ
└── <dataset>/<model>/                              # model 名の ':' は '_' に置換
    ├── <dataset>_<model>_records.jsonl             # 1 問 1 行
    ├── <dataset>_<model>_summary.json
    └── energy_raw/<task_id>.txt.gz                 # 計測区間の生データ
```

レコード (1 行) は RPi から受け取った JSON に問題情報・採点結果・エネルギー情報を加えたもの:

```jsonc
{
  "scenario": "RPi", "dataset": "gsm8k", "task_id": "gsm8k/1", "model": "tinyllama:1.1b",
  "completion": "...", "thinking": "", "done_reason": "stop", "prompt_tokens": 56, "completion_tokens": 120,
  "t_start_unix": ..., "t_end_unix": ..., "latency_sec": 8.3, "ttft_sec": 1.2,   // RPi 上の推論区間 (chat_generate)
  "total_duration_ns": ..., "load_duration_ns": ..., "eval_duration_ns": ...,  // Ollama 内部時間
  "rss_peak_bytes": ..., "cpu_temp_max_c": ..., "cpu_freq_mean_mhz": ..., "throttled_end": "0x0",  // ResourceMonitor
  "question": "...", "gold_answer": "5", "pred_answer": "5", "correct": true,
  "pc_start_time": ..., "pc_end_time": ...,                            // PC 上の計測区間
  "energy": {
    "window":    {"energy_J": ..., "mean_power_W": ..., "num_samples": ..., ...},  // プロンプト送信〜結果受信
    "inference": {"energy_J": ..., ...},                                          // RPi 上の推論区間のみ
    "clock_offset_sec": ..., "clock_rtt_sec": ..., "raw_file": "energy_raw/gsm8k_1.txt.gz"
  }
}
```

- `energy_J` = 平均電力 × 区間長、`energy_sum_J` = Σ(V·I)·0.01 s。サンプリングは 100 sps。
- `inference` は RPi の時刻を、モデルごとに推定した時計オフセット (最小 RTT 法) で PC 時刻に換算した区間。
- 生データは `fnirsi_logger.py` の出力と同じ書式 (energy/capacity は区間開始からの積算) なので、
  `./plot.gnuplot "<zcat energy_raw/gsm8k_1.txt.gz"` でプロットできる。
- HumanEval は採点しない。`evaluate_functional_correctness <records.jsonl>` で後から評価する。
- 中断しても、再実行すれば記録済みの問題を飛ばして再開する。

## トラブルシュート

- パワーメータが `Operation timed out` で開けない: メータが固まっている (読み込み中にプロセスが落ちたときなど。
  `fnirsi_logger` README の TODO 参照)。メータのデータ用 USB を抜き差しする。
  `pc_controller.py` は終了時 (Ctrl+C を含む) に必ずメータのバッファを読み切ってから閉じる。
- `ssh exited with code 255`: `rpi_server.log` に ssh のエラーが出る。鍵認証・ホスト名を確認する。

## エネルギー予算で打ち切る実験 (`budget_controller.py`)

1 問ごとにエネルギーの上限 (予算) を決め、RPi の消費エネルギーが予算に達した時点で推論を打ち切って
PTME (Precision / Time / Memory / Energy) を記録する。構成 (PC・メータ・RPi のつなぎ方) は上と同じ。

| | `pc_controller.py` | `budget_controller.py` |
|---|---|---|
| 推論の送り先 | rpi_server `/generate` (RPi 内で最後まで生成) | RPi の Ollama `/api/chat` に PC からストリーミングで直接送る (SSH ポートフォワード) |
| 打ち切り | なし | メータのサンプル受信ごとに積算し、予算に達した瞬間に Ollama との接続を切る |
| メモリ・温度 | `/generate` 内で計測 | rpi_server `/monitor_start`, `/monitor_stop` |

- 予算はプロンプト送信からの積算エネルギーで、プロンプト処理 (prefill) を含む。モデルのロードは含まない。
- 接続を切ると Ollama は生成を中止する。トークンを待たずに切るので、prefill 中でも打ち切れる。
  打ち切り後も `--tail_sec` (3 s) 計測を続け、電力が待機電力に戻ったことを確認する (戻らなければ警告)。
- 打ち切った出力も、これまでと同じ規則で採点する (GSM8K は `####` が無ければ最後の数値を答えとみなす)。
- `rpi_server.py` に `/monitor_start`, `/monitor_stop` を追加したので、**初回は `--ssh_sync` を付けて RPi に送る**。

```bash
# 動作確認 (メータの代わりに 5 W 一定の模擬メータ。打ち切りの流れだけ確認する)
python rpi_bench/budget_controller.py --ssh pi@raspberrypi.local --ssh_python ~/venv/bin/python --ssh_sync \
    --simulate_power_W 5 --models tinyllama:1.1b --datasets gsm8k --max_samples 2 --budgets 10 inf --scenario BudgetTest

# 本番: 論文と同じ 6 モデル x 3 データセット、100 J ごとに 500 J まで
python rpi_bench/budget_controller.py --ssh pi@raspberrypi.local --ssh_python ~/venv/bin/python \
    --budgets 100 200 300 400 500
```

主なオプション:

| オプション | 既定値 | 内容 |
|---|---|---|
| `--budgets` | 100 200 300 400 500 | 1 問あたりの予算 [J]。`inf` で予算なし (参照用) |
| `--budget_basis` | gross | gross: メータの値そのまま。net: 待機電力を差し引いた分だけを予算に数える |
| `--idle_power_W` / `--baseline_sec` | なし / 20 | net の待機電力。指定が無ければモデルごとに (ロード済みの状態で) 計測する |
| `--order` | task | task: 1 問を全予算で続けて実行 (途中で止めても予算間で問題がそろう)。budget: 予算ごとに全問 |
| `--tail_sec` / `--tail_warn_W` | 3 / 1.0 | 打ち切り後の計測時間 / 待機電力 + これを超えたら警告 |
| `--simulate_power_W` | なし | 模擬メータ (計測値は実測ではない。テスト専用) |

その他 (`--models`, `--datasets`, `--num_predict` など) は `pc_controller.py` と同じ。中断しても再実行で再開する。

### 出力

```
results_rpi/Budget/
├── run_meta.jsonl                                  # モデルごとの設定・待機電力・時計オフセット
├── budget_ptme.csv                                 # budget_summary.py の出力
└── B100J/<dataset>/<model>/                        # 予算ごと (inf は Binf)
    ├── <dataset>_<model>_records.jsonl             # 1 問 1 行
    ├── <dataset>_<model>_summary.json
    ├── energy_raw/<task_id>.txt.gz                 # 送信〜tail 終了までの電力の生データ
    └── trace/<task_id>.json.gz                     # 受信したトークン列と受信時刻 (送信からの秒)
```

レコードは pc_controller.py の項目に次を加えたもの:

- `status`: `budget_exhausted` (打ち切り) / `completed` (予算内で終了)
- `latency_sec`: プロンプト送信 → 打ち切り (または結果受信) の PC 時刻。`completion_tokens` は打ち切り時は受信チャンク数
- `energy.window_basis_J`: 予算と同じ基準の区間エネルギー。`energy.window` は gross の区間情報
- `energy.budget_hit` / `cancel_delay_sec` / `overshoot_J`: 予算到達の検出時刻・打ち切りまでの遅れ・予算の超過分
- `energy.tail`, `tail_last_mean_power_W`: 打ち切り後の電力 (RPi が止まったかの確認)
- 打ち切った場合、Ollama の内部時間・`prompt_tokens` は得られない (None)

### 集計 (`budget_summary.py`)

```bash
python rpi_bench/budget_summary.py results_rpi/Budget
```

予算 × データセット × モデルごとの PTME (precision, mean_latency_sec, peak_rss_gb, mean_energy_J) と、
打ち切り率・超過分などを `budget_ptme.csv` に出す。HumanEval は採点しないので、先に

```bash
for b in B100J B200J B300J B400J B500J; do TAG=Budget/$b bash HE_utils/run_humaneval_evaluation.sh; done
```

で評価しておくと pass@1 が入る (このスクリプトは envelope の結果も毎回評価し直すので、必要なら RPi 部分だけ実行する)。

**事後導出:** temperature 0 では、同じ問題の生成は予算によらず途中まで同じになる。大きい予算 (または `inf`)
の実行の `trace` と `energy_raw` から、小さい予算で打ち切った結果を再現できる:

```bash
python rpi_bench/budget_summary.py results_rpi/Budget --derive_from B500J --derive_budgets 50 100 150 200 250 300 400
```

導出結果は `source = derived_from_B500J` の行になり、実測の予算と重なる場合は正誤の一致率
(`live_correct_agreement`) と時間・トークン数の差も出る。導出した HumanEval の出力は
`Budget/derived_from_B500J/<予算>/` に書くので `TAG=Budget/derived_from_B500J/B100J` で評価できる。
導出では、メモリは元の実行の最大値をそのまま使い、打ち切りの遅れ (`cancel_delay_sec`) は含まない。
