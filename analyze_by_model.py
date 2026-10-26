"""Per-model differences between the RPi and the envelopes (reads the output of compare_rpi_envelopes.py).

    model_ratios.csv     : RPi / envelope per (model, envelope[, dataset]) for time/token, energy/token, memory
    model_accuracy.csv   : accuracy of RPi and the envelopes per (model, dataset)
    model_fits.csv       : task-level fit RPi = slope * E + intercept within each model
    figures/fig_by_model.{pdf,png}, figures/model_ratios.tex
    figures/fig_by_model_instances_<metric>.{pdf,png} : rows = models, columns = envelopes, one point per task

Per-token values are ratios of sums (total time or energy / total tokens) over the tasks of the model.
Models are ordered by parameter count.

Usage:
    python compare_rpi_envelopes.py
    python analyze_by_model.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

import plot_rpi_envelopes as prp  # palette, rcParams and helpers shared with the main figures

ENVELOPES = prp.ENVELOPES
MODELS = {  # ordered by parameter count
    "tinyllama_1.1b": "TinyLlama 1.1B",
    "qwen2.5_1.5b": "Qwen2.5 1.5B",
    "gemma_2b": "Gemma 2B",
    "phi_2.7b": "Phi-2 2.7B",
    "phi3_3.8b": "Phi-3 3.8B",
}
RATIO_METRICS = {
    "time_per_token": "Time per token",
    "energy_per_token": "Energy per token",
    "memory_GB": "Peak RSS",
}


def aggregate(tasks: pd.DataFrame, keys: list) -> pd.DataFrame:
    rows = []
    for key, g in tasks.groupby(keys):
        has_tokens = g["tokens"].notna()
        tok = g.loc[has_tokens, "tokens"].sum()
        rows.append(
            {
                **dict(zip(keys, key)),
                "precision": g["precision"].mean(),
                "time_per_token": g.loc[has_tokens, "time_s"].sum() / tok,
                "energy_per_token": g.loc[has_tokens, "energy_J"].sum(min_count=1) / tok,
                "memory_GB": g["memory_GB"].mean(),
                "num_tasks": len(g),
            }
        )
    return pd.DataFrame(rows)


def ratios(agg: pd.DataFrame, keys: list) -> pd.DataFrame:
    rpi = agg[agg["env"] == "RPi"].set_index(keys)
    rows = []
    for env in ENVELOPES:
        e = agg[agg["env"] == env].set_index(keys)
        for idx in rpi.index.intersection(e.index):
            row = dict(zip(keys, idx if isinstance(idx, tuple) else (idx,)))
            row["envelope"] = env
            for m in RATIO_METRICS:
                row[f"{m}_rpi"] = rpi.at[idx, m]
                row[f"{m}_env"] = e.at[idx, m]
                row[f"{m}_ratio"] = rpi.at[idx, m] / e.at[idx, m]
            rows.append(row)
    return pd.DataFrame(rows)


def model_fits(tasks: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "dataset", "task"]
    rpi = tasks[tasks["env"] == "RPi"].set_index(keys)
    rows = []
    for env in ENVELOPES:
        joined = tasks[tasks["env"] == env].set_index(keys).add_suffix("_env").join(rpi.add_suffix("_rpi"), how="inner")
        for model in MODELS:
            sub = joined[joined.index.get_level_values("model") == model]
            for m in ["time_per_token", "energy_per_token", "memory_GB"]:
                x = sub[f"{m}_env"].to_numpy(dtype=float)
                y = sub[f"{m}_rpi"].to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                x, y = x[ok], y[ok]
                lr = stats.linregress(x, y)
                rows.append(
                    {
                        "model": model, "envelope": env, "metric": m, "n": len(x),
                        "pearson_r": lr.rvalue, "spearman_rho": stats.spearmanr(x, y).statistic,
                        "slope": lr.slope, "intercept": lr.intercept, "r2": lr.rvalue**2,
                    }
                )  # fmt: skip
    return pd.DataFrame(rows)


def draw_ratio_panel(ax, model_ratios: pd.DataFrame, metric: str) -> None:
    xs = np.arange(len(MODELS))
    ax.axhline(1.0, color=prp.IDENTITY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for env in ENVELOPES:
        r = model_ratios[model_ratios["envelope"] == env].set_index("model").reindex(list(MODELS))
        ax.plot(
            xs, r[f"{metric}_ratio"], color=prp.ENV_COLORS[env], marker=prp.ENV_MARKERS[env], ms=4.5,
            mec="white", mew=0.5, lw=1.4, zorder=3,
        )  # fmt: skip
    ax.set_ylim(bottom=0)
    ax.set_xticks(xs, list(MODELS.values()), rotation=20, ha="right")
    ax.set_ylabel("RPi / envelope")
    ax.set_title(f"{RATIO_METRICS[metric]} (RPi / envelope)", loc="left", color=prp.INK)
    prp.style_axes(ax)


def draw_memory_panel(ax, model_ratios: pd.DataFrame) -> None:
    xs = np.arange(len(MODELS))
    ax.axhline(1.0, color=prp.IDENTITY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for env in ENVELOPES:
        r = model_ratios[model_ratios["envelope"] == env].set_index("model").reindex(list(MODELS))
        ax.plot(
            xs, r["memory_GB_ratio"], color=prp.ENV_COLORS[env], marker=prp.ENV_MARKERS[env], ms=4.5,
            mec="white", mew=0.5, lw=1.4, zorder=3,
        )  # fmt: skip
    ax.set_xticks(xs, list(MODELS.values()), rotation=20, ha="right")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("RPi / envelope")
    ax.set_title("Peak RSS (RPi / envelope)", loc="left", color=prp.INK)
    prp.style_axes(ax)


def draw_accuracy_panel(ax, acc: pd.DataFrame) -> None:
    """RPi − envelope の正解率差 (E1–E5 は同一なので E1 を代表に使う)."""
    xs = np.arange(len(MODELS))
    width = 0.26
    ax.axhline(0.0, color=prp.INK_SECONDARY, lw=0.6, zorder=1)
    for k, (ds, (name, color, _)) in enumerate(prp.DATASETS.items()):
        d = acc[acc["dataset"] == ds].set_index("model").reindex(list(MODELS))
        diff = (d["precision_rpi"] - d["precision_env"]) * 100
        ax.bar(xs + (k - 1) * width, diff, width=width - 0.03, color=color, label=name, zorder=2)
    ax.set_xticks(xs, list(MODELS.values()), rotation=20, ha="right")
    ax.set_ylabel("RPi − envelope [pt]")
    ax.set_title("Accuracy difference (E1–E5 identical)", loc="left", color=prp.INK)
    prp.style_axes(ax)
    ax.legend(frameon=False, fontsize=6, loc="upper left", ncol=3, handlelength=1.0, columnspacing=1.0)


def make_figure(model_ratios, acc, stem: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    draw_accuracy_panel(axes[0, 0], acc)
    draw_ratio_panel(axes[0, 1], model_ratios, "time_per_token")
    draw_ratio_panel(axes[1, 0], model_ratios, "energy_per_token")
    draw_memory_panel(axes[1, 1], model_ratios)
    handles = [
        Line2D([], [], color=prp.ENV_COLORS[e], lw=1.4, marker=prp.ENV_MARKERS[e], ms=4.5, mec="white", mew=0.5,
               label=e)
        for e in ENVELOPES
    ]  # fmt: skip
    handles.append(Line2D([], [], color=prp.IDENTITY, lw=0.8, ls=(0, (3, 2)), label="RPi = envelope"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.4, w_pad=1.5)
    prp.save(fig, stem)


def make_instance_grid(tasks: pd.DataFrame, fits: pd.DataFrame, metric: str, stem: Path) -> None:
    """モデル (行) x envelope (列) ごとに、インスタンス単位の散布図 (precision は 2x2 分割表)."""
    fig, axes = plt.subplots(len(MODELS), len(ENVELOPES), figsize=(7.0, 1.6 * len(MODELS) + 0.6), squeeze=False)
    keys = ["model", "dataset", "task"]
    for i, (model, name) in enumerate(MODELS.items()):
        sub = tasks[tasks["model"] == model]
        model_fit = fits[fits["model"] == model].assign(level="task")
        for j, env in enumerate(ENVELOPES):
            ax = axes[i, j]
            if metric == "precision":
                prp.draw_contingency(ax, sub, env)
            else:
                prp.draw_scatter(ax, sub, model_fit, metric, keys, "task", envs=[env])
            if i == len(MODELS) - 1:
                ax.set_xlabel(f"{env} [{prp.SHORT[metric]}]", color=prp.INK_SECONDARY, fontsize=6)
        axes[i, 0].set_ylabel(f"{name}\nRPi [{prp.SHORT[metric]}]")
    for j, env in enumerate(ENVELOPES):
        axes[0, j].annotate(
            env, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 24), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color=prp.INK,
        )  # fmt: skip
    if metric == "precision":
        fig.suptitle("Accuracy: RPi (rows) vs envelope (columns) per task", fontsize=8, color=prp.INK, y=0.995)
    else:
        prp.dataset_legend(fig, [
            Line2D([], [], color=prp.FIT, lw=1.4, label="Linear fit (within model)"),
            Line2D([], [], color=prp.IDENTITY, lw=0.8, ls=(0, (3, 2)), label="y = x"),
        ])  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()), h_pad=0.8, w_pad=0.6)
    prp.save(fig, stem)


def write_ratio_table(model_ratios: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{l" + "r" * len(ENVELOPES) * 2 + "}",
        r"\toprule",
        r" & \multicolumn{" + str(len(ENVELOPES)) + r"}{c}{Time per token} & \multicolumn{" + str(len(ENVELOPES))
        + r"}{c}{Energy per token} \\",
        r"Model & " + " & ".join(ENVELOPES * 2) + r" \\",
        r"\midrule",
    ]  # fmt: skip
    for model, name in MODELS.items():
        r = model_ratios[model_ratios["model"] == model].set_index("envelope")
        cells = [f"{r.at[e, 'time_per_token_ratio']:.2f}" for e in ENVELOPES]
        cells += [f"{r.at[e, 'energy_per_token_ratio']:.2f}" for e in ENVELOPES]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in_dir", type=Path, default=Path("results_analysis/rpi_vs_envelopes"))
    args = parser.parse_args()
    out_dir = args.in_dir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    tasks = pd.read_csv(args.in_dir / "task_metrics.csv", dtype={"task": str})
    tasks = tasks[tasks["model"].isin(MODELS)]

    by_model = ratios(aggregate(tasks, ["env", "model"]), ["model"])
    by_model_ds = ratios(aggregate(tasks, ["env", "model", "dataset"]), ["model", "dataset"])
    unit = aggregate(tasks, ["env", "model", "dataset"])
    acc = (
        unit[unit["env"] == "RPi"][["model", "dataset", "precision"]]
        .merge(unit[unit["env"] == "E1"][["model", "dataset", "precision"]], on=["model", "dataset"],
               suffixes=("_rpi", "_env"))
    )  # fmt: skip
    fits = model_fits(tasks)

    pd.concat([by_model.assign(dataset="all"), by_model_ds]).to_csv(out_dir / "model_ratios.csv", index=False)
    acc.to_csv(out_dir / "model_accuracy.csv", index=False)
    fits.to_csv(out_dir / "model_fits.csv", index=False)
    make_figure(by_model, acc, fig_dir / "fig_by_model")
    write_ratio_table(by_model, fig_dir / "model_ratios.tex")
    for metric in ["precision", "time_per_token", "energy_per_token", "memory_GB"]:
        make_instance_grid(tasks, fits, metric, fig_dir / f"fig_by_model_instances_{metric}")

    pd.set_option("display.width", 200)
    for m in RATIO_METRICS:
        print(f"\n[{m}] RPi / envelope")
        print(by_model.pivot(index="model", columns="envelope", values=f"{m}_ratio").reindex(list(MODELS)).round(3))
    print("\n[accuracy] RPi - envelope [pt]")
    acc["diff_pt"] = (acc["precision_rpi"] - acc["precision_env"]) * 100
    print(acc.pivot(index="model", columns="dataset", values="diff_pt").reindex(list(MODELS)).round(1))
    for m in ["time_per_token", "energy_per_token"]:
        print(f"\n[{m}] task-level Pearson r within each model")
        print(fits[fits["metric"] == m].pivot(index="model", columns="envelope", values="pearson_r")
              .reindex(list(MODELS)).round(3))  # fmt: skip
    print(f"\nSaved model_ratios.csv, model_accuracy.csv, model_fits.csv and figures to {out_dir}")


if __name__ == "__main__":
    main()
