#!/usr/bin/env python3
"""budget_controller.py の結果を (予算, モデル, データセット) ごとの PTME 表にまとめる.

  python rpi_bench/budget_summary.py results_rpi/Budget

出力: <scenario_dir>/budget_ptme.csv (1 行 = 予算 x データセット x モデル)

  precision        : GSM8K / MMLU-Pro は記録時の採点、HumanEval は
                     *_samples_extracted.jsonl_results.jsonl (HE_utils の評価結果) があればその pass@1
  time             : mean_latency_sec (プロンプト送信 -> 打ち切り / 結果受信、PC 時刻)
  memory           : peak_rss_gb (Ollama プロセスの RSS の最大値)
  energy           : mean_energy_J (予算と同じ基準 = gross または net), mean_energy_gross_J
  打ち切りの状況    : frac_budget_exhausted, mean_overshoot_J, mean_cancel_delay_ms, mean_tail_last_W

事後導出 (--derive_from): 大きい予算 (または inf) の実行で保存したトークン列 (trace) と電力の生データから、
小さい予算で打ち切った場合の結果を再現する。temperature 0 では、同じ問題の生成は予算によらず途中まで
同じなので、1 回の実行から複数の予算の結果を作れる。実測 (live) の予算と重なる場合は一致度も出す。

  python rpi_bench/budget_summary.py results_rpi/Budget --derive_from B500J --derive_budgets 50 100 150 200

導出した HumanEval の出力は <scenario_dir>/derived_from_<src>/<予算>/human_eval/<model>/ に
records.jsonl として書くので、HE_utils/run_humaneval_evaluation.sh で評価できる
(TAG=<scenario>/derived_from_<src>/<予算>)。

Python 3.9 互換.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from bench_tasks import Task, score
from budget_controller import budget_label, parse_budget
from power_meter import INTERVAL

LABEL_RE = re.compile(r"^B(inf|[0-9.]+J)$")


def label_value(label: str) -> float:
    return math.inf if label == "Binf" else float(label[1:-1])


def read_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def he_passed(model_dir: Path) -> Optional[Dict[str, bool]]:
    """HE_utils の評価結果 (task_id -> passed)."""
    files = sorted(model_dir.glob("human_eval_*_samples_extracted.jsonl_results.jsonl"))
    if not files:
        return None
    return {r["task_id"]: bool(r["passed"]) for r in read_jsonl(files[0])}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def precision_of(records: List[dict], dataset: str, passed: Optional[Dict[str, bool]]):
    if dataset == "human_eval":
        if passed is None:
            return None, 0
        vals = [passed[r["task_id"]] for r in records if r["task_id"] in passed]
        return (mean([1.0 if v else 0.0 for v in vals]), len(vals))
    vals = [r.get("correct") for r in records if r.get("correct") is not None]
    return (mean([1.0 if v else 0.0 for v in vals]), len(vals))


def summarize_group(records: List[dict], dataset: str, passed) -> dict:
    prec, n_scored = precision_of(records, dataset, passed)
    rss = [r.get("rss_peak_bytes") for r in records if r.get("rss_peak_bytes") is not None]
    e = [r["energy"] for r in records]
    return {
        "n": len(records),
        "precision": prec,
        "n_scored": n_scored,
        "mean_latency_sec": mean([r["latency_sec"] for r in records]),
        "peak_rss_gb": (max(rss) / 1e9) if rss else None,
        "mean_energy_J": mean([x.get("window_basis_J") for x in e]),
        "mean_energy_gross_J": mean([x["window"].get("energy_sum_J") for x in e]),
        "mean_completion_tokens": mean([r.get("completion_tokens") for r in records]),
        "frac_budget_exhausted": mean([1.0 if r["status"] == "budget_exhausted" else 0.0 for r in records]),
        "mean_overshoot_J": mean([x.get("overshoot_J") for x in e]),
        "mean_cancel_delay_ms": mean([(x["cancel_delay_sec"] * 1000) if x.get("cancel_delay_sec") is not None else None for x in e]),
        "mean_tail_last_W": mean([x.get("tail_last_mean_power_W") for x in e]),
    }


# ---------------------------------------------------------------- 事後導出


def read_raw(path: Path):
    """save_raw() の書式から (t, power_W) の列を読む."""
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4 or parts[0] == "timestamp":
                continue
            out.append((float(parts[0]), float(parts[2]) * float(parts[3])))
    return out


def read_trace(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def task_for_scoring(r: dict) -> Task:
    if r["dataset"] == "gsm8k":
        return Task("gsm8k", r["task_id"], "", gold=f"#### {r.get('gold_answer')}")
    if r["dataset"] == "mmlu-pro":
        return Task("mmlu-pro", r["task_id"], "", gold=r.get("answer"))
    return Task(r["dataset"], r["task_id"], "")


def derive_record(r: dict, model_dir: Path, b: float) -> Optional[dict]:
    """予算 b [J] で打ち切った場合のレコードを、より大きい予算の実行 r から作る (作れなければ None)."""
    e = r["energy"]
    if not e.get("raw_file") or not e.get("trace_file"):
        return None
    if r["status"] == "budget_exhausted" and (r["budget_J"] is None or b > r["budget_J"]):
        return None  # 元の実行がそれより先を見ていない
    bb = e.get("budget_baseline_W") or 0.0
    raw = read_raw(model_dir / e["raw_file"])
    trace = read_trace(model_dir / e["trace_file"])
    t_send, t_stop = trace["t_send"], trace["t_stop"]

    acc = gross = 0.0
    t_b = None
    for t, p in raw:
        if t < t_send or t > t_stop:
            continue
        acc += (p - bb) * INTERVAL
        gross += p * INTERVAL
        if acc >= b:
            t_b = t
            break

    d = dict(r)
    d["budget_J"] = b
    d["budget_label"] = budget_label(b)
    d["derived_from"] = r["budget_label"]
    if t_b is None:
        # 予算内で終わっている: 元のレコードのまま
        if r["status"] == "budget_exhausted":
            return None
        d["energy"] = dict(e, budget_J=b)
        return d
    cut_rel = t_b - t_send
    chunks = [c for c in trace["chunks"] if c[0] <= cut_rel]
    content = "".join(c[1] for c in chunks)
    d.update(
        status="budget_exhausted",
        completion=content,
        thinking="".join(c[2] for c in chunks),
        done_reason=None,
        completion_tokens=sum(1 for c in chunks if c[1] or c[2]),
        prompt_tokens=None,
        t_end_unix=t_b,
        latency_sec=cut_rel,
        ttft_sec=next((c[0] for c in chunks if c[1] or c[2]), None),
        total_duration_ns=None,
        eval_duration_ns=None,
        prompt_eval_duration_ns=None,
        **{k: None for k in ("pred_answer", "pred", "correct")},
    )
    d = {k: v for k, v in d.items() if not (k in ("pred_answer", "pred", "correct") and v is None)}
    d.update(score(task_for_scoring(r), content))
    d["energy"] = {
        **{k: e.get(k) for k in ("meter", "budget_basis", "baseline_W", "budget_baseline_W")},
        "budget_J": b,
        "window": {"t_start": t_send, "t_end": t_b, "duration_sec": cut_rel, "energy_sum_J": gross},
        "window_basis_J": acc,
        "overshoot_J": acc - b,
        "cancel_delay_sec": None,
        "tail_last_mean_power_W": None,
    }
    return d


# ---------------------------------------------------------------- main


def collect(scen_dir: Path) -> Dict[tuple, Path]:
    """(label, dataset, model_dir_name) -> model_dir."""
    groups = {}
    for records_path in sorted(scen_dir.glob("B*/*/*/*_records.jsonl")):
        model_dir = records_path.parent
        label = model_dir.parent.parent.name
        if LABEL_RE.match(label):
            groups[(label, model_dir.parent.name, model_dir.name)] = model_dir
    return groups


def main():
    parser = argparse.ArgumentParser(description="Summarise energy-budget runs into a PTME table.")
    parser.add_argument("scenario_dir", help="e.g. results_rpi/Budget")
    parser.add_argument("--derive_from", default=None, help="Budget label to derive from, e.g. B500J or Binf.")
    parser.add_argument("--derive_budgets", nargs="+", type=parse_budget, default=[], help="Budgets [J] to derive.")
    parser.add_argument("--output", default=None, help="CSV path (default: <scenario_dir>/budget_ptme.csv).")
    args = parser.parse_args()

    scen_dir = Path(args.scenario_dir)
    groups = collect(scen_dir)
    if not groups:
        raise SystemExit(f"No B*/<dataset>/<model>/*_records.jsonl under {scen_dir}")

    rows = []
    live = {}  # (label, ds, model) -> {task_id: record}
    for (label, ds, model), model_dir in sorted(groups.items(), key=lambda kv: (label_value(kv[0][0]),) + kv[0][1:]):
        records = read_jsonl(next(model_dir.glob("*_records.jsonl")))
        live[(label, ds, model)] = {r["task_id"]: r for r in records}
        rows.append({
            "source": "live", "budget": label, "budget_J": label_value(label), "dataset": ds, "model": model,
            **summarize_group(records, ds, he_passed(model_dir)),
        })  # fmt: skip

    if args.derive_from:
        if any(b is None for b in args.derive_budgets):
            raise SystemExit("--derive_budgets must be finite")
        src_groups = {k: v for k, v in groups.items() if k[0] == args.derive_from}
        if not src_groups:
            raise SystemExit(f"No runs with budget {args.derive_from}")
        for b in sorted(args.derive_budgets):
            label = budget_label(b)
            for (_, ds, model), model_dir in sorted(src_groups.items()):
                derived = []
                n_skip = 0
                for r in live[(args.derive_from, ds, model)].values():
                    d = derive_record(r, model_dir, b)
                    if d is None:
                        n_skip += 1
                    else:
                        derived.append(d)
                if not derived:
                    continue
                out_dir = scen_dir / f"derived_from_{args.derive_from}" / label / ds / model
                out_dir.mkdir(parents=True, exist_ok=True)
                with (out_dir / f"{ds}_{model}_records.jsonl").open("w", encoding="utf-8") as f:
                    for d in derived:
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")
                row = {
                    "source": f"derived_from_{args.derive_from}", "budget": label, "budget_J": b,
                    "dataset": ds, "model": model, **summarize_group(derived, ds, he_passed(out_dir)),
                    "n_not_derivable": n_skip,
                }  # fmt: skip
                # 同じ予算の実測があれば一致度 (同じ問題で正誤が一致した割合、時間の差)
                ref = live.get((label, ds, model))
                if ref:
                    pairs = [(d, ref[d["task_id"]]) for d in derived if d["task_id"] in ref]
                    both = [(d, lr) for d, lr in pairs if d.get("correct") is not None and lr.get("correct") is not None]
                    row["live_n_pairs"] = len(pairs)
                    row["live_correct_agreement"] = mean([1.0 if d["correct"] == lr["correct"] else 0.0 for d, lr in both])
                    row["live_mean_abs_latency_diff_sec"] = mean([abs(d["latency_sec"] - lr["latency_sec"]) for d, lr in pairs])
                    row["live_mean_abs_token_diff"] = mean(
                        [abs((d.get("completion_tokens") or 0) - (lr.get("completion_tokens") or 0)) for d, lr in pairs]
                    )
                rows.append(row)

    out = Path(args.output) if args.output else scen_dir / "budget_ptme.csv"
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    def fmt(v, spec):
        return "-" if v is None else format(v, spec)

    print(f"{'source':<22} {'budget':>6} {'dataset':<10} {'model':<15} {'n':>4} {'prec':>6} {'time[s]':>8} "
          f"{'E[J]':>7} {'RSS[GB]':>7} {'cut':>5} {'over[J]':>7}")  # fmt: skip
    for r in rows:
        print(
            f"{r['source']:<22} {r['budget']:>6} {r['dataset']:<10} {r['model']:<15} {r['n']:>4} "
            f"{fmt(r['precision'], '.3f'):>6} {fmt(r['mean_latency_sec'], '.1f'):>8} {fmt(r['mean_energy_J'], '.1f'):>7} "
            f"{fmt(r['peak_rss_gb'], '.2f'):>7} {fmt(r['frac_budget_exhausted'], '.2f'):>5} {fmt(r['mean_overshoot_J'], '.2f'):>7}"
        )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
