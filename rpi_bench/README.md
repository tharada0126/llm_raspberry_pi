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
| `rpi_server.py` | RPi | Ollama 操作。推論は既存の `ollama_utils.call_ollama_generate_with_memory` を使用 |

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

- `--ssh_sync`: 起動前に `rpi_server.py` と `ollama_utils.py` を scp で RPi に送る (PC 側で編集した場合など)
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
| `--num_predict` | 512 |
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
  "completion": "...", "finish_reason": "stop", "prompt_tokens": 56, "completion_tokens": 120,
  "ttft": 1.2, "rss_baseline_bytes": ..., "rss_peak_bytes": ..., "rss_final_bytes": ...,
  "rpi_start_time": ..., "rpi_end_time": ..., "latency_sec": 8.3,      // RPi 上の推論区間
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
