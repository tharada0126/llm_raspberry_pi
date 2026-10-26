"""Compare the Raspberry Pi results with the previous x86 envelope results (E1-E5).

For each metric (precision, time, energy, memory) and each envelope, the RPi value is
regressed on the envelope value (explanatory variable = envelope):

    linear : RPi = slope * E + intercept
    log-log: RPi = 10**intercept * E**slope   (time / energy / memory only)

Pearson and Spearman correlations are reported as well. Two levels of analysis:

    unit : one point per (model, dataset), metric averaged over the tasks (main analysis)
    task : one point per (model, dataset, task)

Only tasks that exist in the RPi results are used, so both sides cover the same questions.

Metric definitions (both sides measure one inference request):

    precision        : GSM8K / MMLU-Pro `correct`, HumanEval `passed` (from human-eval)
    time_per_token   : time_s / tokens [s/token]
    energy_per_token : energy_J / tokens [J/token]
    memory_GB        : `rss_peak_bytes` of the Ollama processes (both sides)

    time_s   : envelope `time` (pyRAPL duration) / RPi `latency_sec`
    energy_J : envelope `cpu_energy + dram_energy` (RAPL) / RPi `energy.inference.energy_J` (USB meter)
               minus the RPi idle energy (idle mean power from results_rpi/RPi/idle x inference duration)
    tokens   : prompt_tokens + completion_tokens (input + output)

At the unit level the per-token metrics are ratios of sums (total time / total tokens over the
tasks), so tasks with very few tokens do not dominate; at the task level they are per-task ratios.

Usage:
    bash HE_utils/run_humaneval_evaluation.sh   # HumanEval pass/fail, needed for HumanEval precision
    python compare_rpi_envelopes.py
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger("compare")

DATASETS = ["human_eval", "gsm8k", "mmlu-pro"]
MODELS = ["tinyllama_1.1b", "qwen2.5_1.5b", "phi_2.7b", "phi3_3.8b", "gemma_2b"]  # mistral_7b: RPi run not finished
ENVELOPES = ["E1", "E2", "E3", "E4", "E5"]
METRICS = ["precision", "time_per_token", "energy_per_token", "memory_GB"]
LOG_METRICS = ["time_per_token", "energy_per_token", "memory_GB"]


def read_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def task_key(task_id) -> str:
    """'gsm8k/1', 'mmlu-pro/70', 70, 'HumanEval/0' -> '1', '70', '70', '0'."""
    return str(task_id).split("/")[-1]


def total_tokens(r: dict) -> int | None:
    """入力 + 出力トークン数 (どちらかが欠けていれば None)."""
    if r.get("prompt_tokens") is None or r.get("completion_tokens") is None:
        return None
    return int(r["prompt_tokens"]) + int(r["completion_tokens"])


def one_file(dir_path: Path, pattern: str) -> Path | None:
    files = sorted(dir_path.glob(pattern))
    if len(files) > 1:
        raise RuntimeError(f"Multiple files match {dir_path}/{pattern}: {files}")
    return files[0] if files else None


def load_he_passed(results_dir: Path, model: str) -> dict | None:
    """task key -> passed, from human-eval's *_samples_extracted.jsonl_results.jsonl."""
    path = one_file(results_dir / "human_eval" / model, "*_samples_extracted.jsonl_results.jsonl")
    if path is None:
        return None
    return {task_key(r["task_id"]): bool(r["passed"]) for r in read_jsonl(path)}


def load_idle_power(idle_dir: Path) -> float:
    """RPi のアイドル時平均電力 [W] (最新の idle_*.json)."""
    files = sorted(idle_dir.glob("idle_*.json"))
    if not files:
        raise SystemExit(f"No idle_*.json in {idle_dir} (use --no_idle_subtraction to skip)")
    with files[-1].open(encoding="utf-8") as f:
        idle = json.load(f)
    log.info("RPi idle power: %.4f W (std %.4f W) from %s", idle["mean_power_W"], idle["std_power_W"], files[-1])
    return float(idle["mean_power_W"])


def load_rpi(rpi_dir: Path, model: str, dataset: str, idle_power_W: float) -> pd.DataFrame | None:
    path = one_file(rpi_dir / dataset / model, "*_records.jsonl")
    if path is None:
        return None
    rows = []
    for r in read_jsonl(path):
        energy = (r.get("energy") or {}).get("inference") or {}
        total = energy.get("energy_J")
        rows.append(
            {
                "task": task_key(r["task_id"]),
                "correct": r.get("correct"),
                "time_s": r["latency_sec"],
                # 推論区間の測定値からアイドル分 (アイドル電力 x 区間長) を差し引いた動的エネルギー
                "energy_J": None if total is None else total - idle_power_W * energy["duration_sec"],
                "energy_total_J": total,
                "tokens": total_tokens(r),
                "memory_GB": r["rss_peak_bytes"] / 1e9,
            }
        )
    df = pd.DataFrame(rows).drop_duplicates("task", keep="last")
    if dataset == "human_eval":
        passed = load_he_passed(rpi_dir, model)
        df["correct"] = df["task"].map(passed) if passed is not None else None
    return df


def load_envelope(env_dir: Path, he_dir: Path, model: str, dataset: str) -> pd.DataFrame | None:
    path = one_file(env_dir / dataset / model, "*_profiles.jsonl")
    if path is None:
        return None
    profiles = read_jsonl(path)
    rows = []
    for i, p in enumerate(profiles):
        if dataset == "human_eval":
            task = str(i)  # profiles have no task id; they are in the same order as *_samples.jsonl (HumanEval/0..163)
        else:
            task = task_key(p["task_id"] if dataset == "gsm8k" else p["question_id"])
        rows.append(
            {
                "task": task,
                "correct": p.get("correct"),
                "time_s": p["time"],
                "energy_J": p["cpu_energy"] + (p.get("dram_energy") or 0.0),
                "tokens": total_tokens(p),
                "memory_GB": p["rss_peak_bytes"] / 1e9,
            }
        )
    df = pd.DataFrame(rows).drop_duplicates("task", keep="last")
    if dataset == "human_eval":
        passed = load_he_passed(he_dir, model)
        df["correct"] = df["task"].map(passed) if passed is not None else None
    return df


def load_all(args) -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        for model in args.models:
            rpi = load_rpi(args.rpi_dir, model, dataset, args.idle_power_W)
            if rpi is None:
                log.warning("No RPi results for %s / %s", dataset, model)
                continue
            frames.append(rpi.assign(env="RPi", model=model, dataset=dataset))
            for env in ENVELOPES:
                df = load_envelope(args.envelope_dir / env, args.he_dir / env, model, dataset)
                if df is None:
                    log.warning("No %s results for %s / %s", env, dataset, model)
                    continue
                df = df[df["task"].isin(rpi["task"])]
                missing = len(rpi) - len(df)
                if missing:
                    log.warning("%s %s / %s: %d RPi tasks missing", env, dataset, model, missing)
                frames.append(df.assign(env=env, model=model, dataset=dataset))
    tasks = pd.concat(frames, ignore_index=True)
    tasks["precision"] = tasks["correct"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    tokens = pd.to_numeric(tasks["tokens"], errors="coerce").where(lambda t: t > 0)
    tasks["tokens"] = tokens
    tasks["time_per_token"] = tasks["time_s"] / tokens
    tasks["energy_per_token"] = tasks["energy_J"] / tokens
    he_missing = tasks[(tasks["dataset"] == "human_eval") & tasks["precision"].isna()]
    if len(he_missing):
        log.warning(
            "HumanEval pass/fail missing for %s; run HE_utils/run_humaneval_evaluation.sh first. "
            "HumanEval is excluded from precision.",
            sorted(set(zip(he_missing["env"], he_missing["model"]))),
        )
    # energy_total_J: 差し引き前の測定値 (envelope 側は差し引きなしなので energy_J と同じ)
    tasks["energy_total_J"] = tasks["energy_total_J"].fillna(tasks["energy_J"])
    return tasks[["env", "model", "dataset", "task", *METRICS, "time_s", "energy_J", "energy_total_J", "tokens"]]


def aggregate_units(tasks: pd.DataFrame) -> pd.DataFrame:
    """(env, model, dataset) ごとの値. per-token 指標は合計時間 (エネルギー) / 合計トークン数."""
    rows = []
    for (env, model, dataset), g in tasks.groupby(["env", "model", "dataset"]):
        has_tokens = g["tokens"].notna()
        tok = g.loc[has_tokens, "tokens"].sum()
        rows.append(
            {
                "env": env,
                "model": model,
                "dataset": dataset,
                "precision": g["precision"].mean(),
                "time_per_token": g.loc[has_tokens, "time_s"].sum() / tok,
                "energy_per_token": g.loc[has_tokens, "energy_J"].sum(min_count=1) / tok,
                "memory_GB": g["memory_GB"].mean(),
                "time_s": g["time_s"].mean(),
                "energy_J": g["energy_J"].mean(),
                "tokens": g["tokens"].mean(),
                "num_tasks": len(g),
            }
        )
    return pd.DataFrame(rows)


def fit(x: np.ndarray, y: np.ndarray, use_log: bool) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    out = {"n": len(x)}
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return out
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    lr = stats.linregress(x, y)
    out.update(
        pearson_r=pr.statistic,
        pearson_p=pr.pvalue,
        spearman_rho=sr.statistic,
        spearman_p=sr.pvalue,
        slope=lr.slope,
        intercept=lr.intercept,
        r2=lr.rvalue**2,
        slope_se=lr.stderr,
        intercept_se=lr.intercept_stderr,
    )
    if use_log and (x > 0).all() and (y > 0).all():
        ll = stats.linregress(np.log10(x), np.log10(y))
        out.update(loglog_slope=ll.slope, loglog_intercept=ll.intercept, loglog_r2=ll.rvalue**2)
    return out


def fit_all(tasks: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, df, keys in [("unit", units, ["model", "dataset"]), ("task", tasks, ["model", "dataset", "task"])]:
        rpi = df[df["env"] == "RPi"].set_index(keys)
        for env in ENVELOPES:
            env_df = df[df["env"] == env].set_index(keys)
            joined = env_df.add_suffix("_env").join(rpi.add_suffix("_rpi"), how="inner")
            for metric in METRICS:
                x = joined[f"{metric}_env"].to_numpy(dtype=float)
                y = joined[f"{metric}_rpi"].to_numpy(dtype=float)
                rows.append({"level": level, "metric": metric, "envelope": env, **fit(x, y, metric in LOG_METRICS)})
    return pd.DataFrame(rows)


def print_summary(fits: pd.DataFrame, units: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== Per (model, dataset) means ===")
    wide = units.pivot_table(index=["dataset", "model"], columns="env", values=METRICS)
    for metric in METRICS:
        print(f"\n[{metric}]")
        print(wide[metric][["RPi", *ENVELOPES]].round(4).to_string())

    for level in ["unit", "task"]:
        print(f"\n=== RPi vs envelope ({level} level; explanatory variable = envelope) ===")
        for metric in METRICS:
            print(f"\n[{metric}]")
            for _, r in fits[(fits["level"] == level) & (fits["metric"] == metric)].iterrows():
                if "slope" not in r or pd.isna(r.get("slope")):
                    print(f"  {r['envelope']}: n={r['n']} (not enough data)")
                    continue
                line = (
                    f"  {r['envelope']}: n={int(r['n']):4d}  r={r['pearson_r']:+.3f} (p={r['pearson_p']:.2g})  "
                    f"rho={r['spearman_rho']:+.3f}  RPi = {r['slope']:.4g}*E {r['intercept']:+.4g}  (R2={r['r2']:.3f})"
                )
                if pd.notna(r.get("loglog_slope")):
                    line += (
                        f"  | RPi = {10 ** r['loglog_intercept']:.4g}*E^{r['loglog_slope']:.3f}"
                        f"  (log R2={r['loglog_r2']:.3f})"
                    )
                print(line)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rpi_dir", type=Path, default=Path("results_rpi/RPi"))
    parser.add_argument("--envelope_dir", type=Path, default=Path("llm_energy_exp/results"))
    parser.add_argument(
        "--he_dir", type=Path, default=Path("results_analysis/human_eval"),
        help="HumanEval evaluation results for the envelopes (written by HE_utils/run_humaneval_evaluation.sh)",
    )  # fmt: skip
    parser.add_argument("--idle_dir", type=Path, default=None, help="Default: <rpi_dir>/idle")
    parser.add_argument("--no_idle_subtraction", action="store_true", help="Use the measured RPi energy as is")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--out_dir", type=Path, default=Path("results_analysis/rpi_vs_envelopes"))
    args = parser.parse_args()
    args.idle_power_W = 0.0 if args.no_idle_subtraction else load_idle_power(args.idle_dir or args.rpi_dir / "idle")

    tasks = load_all(args)
    units = aggregate_units(tasks)
    fits = fit_all(tasks, units)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(args.out_dir / "task_metrics.csv", index=False)
    units.to_csv(args.out_dir / "unit_metrics.csv", index=False)
    fits.to_csv(args.out_dir / "fits.csv", index=False)
    print_summary(fits, units)
    log.info("Saved task_metrics.csv, unit_metrics.csv, fits.csv to %s", args.out_dir)


if __name__ == "__main__":
    main()
