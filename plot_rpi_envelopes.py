"""Paper figures for the RPi vs envelope comparison (reads the output of compare_rpi_envelopes.py).

    fig_rpi_vs_envelopes_mean.{pdf,png}      : one point per (model, dataset); all envelopes in one panel
    fig_rpi_vs_envelopes_instances.{pdf,png} : one point per task instance; all envelopes in one panel
    fig_rpi_vs_envelopes_mean_grid.*         : as *_mean, one column per envelope (points colored by dataset)
    fig_rpi_vs_envelopes_instances_grid.*    : as *_instances, one column per envelope
    fits_unit.tex / fits_task.tex            : linear conversion RPi = slope * E + intercept (booktabs table)

x = envelope, y = RPi. Solid lines = linear fit per envelope (from fits.csv), dashed line = y = x.
Envelopes are ordered (E1: fewest resources ... E5: most), so they share one blue ramp from light
to dark; the marker shape repeats the envelope so the figures also read in grayscale.

In the per-envelope grids each panel has a single envelope, so the points are colored by dataset
instead (categorical slots 1-3, again with distinct marker shapes) and the panel title gives the fit.

For precision at the instance level the values are 0/1, so the panels show how often RPi and the
envelope agree on each task (stacked bars / 2x2 contingency tables) instead of a scatter.

Usage:
    python compare_rpi_envelopes.py
    python plot_rpi_envelopes.py
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

ENVELOPES = ["E1", "E2", "E3", "E4", "E5"]
METRICS = {
    "precision": "Accuracy",
    "time_per_token": "Time per token [s/token]",
    "energy_per_token": "Energy per token [J/token]",
    "memory_GB": "Peak RSS [GB]",
}
# Short units for the x labels of the per-envelope grids (the long names do not fit five columns)
SHORT = {"precision": "Accuracy", "time_per_token": "s/token", "energy_per_token": "J/token", "memory_GB": "GB"}
# Ordinal blue ramp of the reference palette (steps 250-650; step 250 is the lightest that clears 2:1 on white)
ENV_COLORS = dict(zip(ENVELOPES, ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]))
ENV_MARKERS = dict(zip(ENVELOPES, ["o", "s", "^", "D", "v"]))
# Agreement categories at the instance level: categorical slots 1-4 (adjacent pairs validated)
AGREEMENT = [
    ("both", "Both correct", "#2a78d6"),
    ("rpi_only", "RPi only", "#eb6834"),
    ("env_only", "Envelope only", "#1baf7a"),
    ("neither", "Both wrong", "#eda100"),
]
# Datasets in the per-envelope grids: categorical slots 1-3 (validated for all pairs on a light surface)
DATASETS = {
    "human_eval": ("HumanEval", "#2a78d6", "o"),
    "gsm8k": ("GSM8K", "#eb6834", "s"),
    "mmlu-pro": ("MMLU-Pro", "#1baf7a", "^"),
}
# Sequential blue ramp (steps 100-650) for the contingency tables
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
INK = "#0b0b0b"
FIT = "#3d3c39"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
IDENTITY = "#a8a7a2"

plt.rcParams.update(
    {
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 7,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,  # embed TrueType (editable text, accepted by most venues)
        "ps.fonttype": 42,
    }
)


def paired(df: pd.DataFrame, metric: str, env: str, keys: list) -> pd.DataFrame:
    rpi = df[df["env"] == "RPi"].set_index(keys)[metric]
    env_vals = df[df["env"] == env].set_index(keys)[metric]
    return pd.concat({"x": env_vals, "y": rpi}, axis=1).dropna()


def style_axes(ax) -> None:
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_scatter(ax, df, fits, metric, keys, level, envs=ENVELOPES) -> None:
    """envs が 1 つならデータセットで色分けし、回帰式をパネルタイトルに書く."""
    single = len(envs) == 1
    pairs = {env: paired(df, metric, env, keys) for env in envs}
    allv = pd.concat(pairs.values())

    # 軸は 0 始まりの線形軸で、縦横とも同じ範囲 (y = x が対角線になる)
    upper = max(allv["x"].max(), allv["y"].max()) * 1.05
    xlim = ylim = (0, upper)
    ref = np.array([0, upper])
    ax.plot(ref, ref, color=IDENTITY, lw=0.8, ls=(0, (3, 2)), zorder=1)

    instance = level == "task"
    point_style = dict(
        s=(3 if single else 4) if instance else 20, alpha=0.35 if instance else 1.0,
        linewidths=0 if instance else 0.5, edgecolors="none" if instance else "white", zorder=2, rasterized=instance,
    )  # fmt: skip
    for env in envs:
        p = pairs[env]
        if single:
            for ds, (_, color, marker) in DATASETS.items():
                sel = p[p.index.get_level_values("dataset") == ds]
                ax.scatter(sel["x"], sel["y"], c=color, marker=marker, **point_style)
        else:
            ax.scatter(p["x"], p["y"], c=ENV_COLORS[env], marker=ENV_MARKERS[env], **point_style)
    for env in envs:
        fit = fits[(fits["level"] == level) & (fits["metric"] == metric) & (fits["envelope"] == env)].iloc[0]
        if pd.isna(fit.get("slope")):
            continue
        x0, x1 = pairs[env]["x"].min(), pairs[env]["x"].max()
        xs = np.array([x0, x1])
        ys = fit["slope"] * xs + fit["intercept"]
        # 線の下に白い縁を敷いて、点群の上でも見えるようにする
        ax.plot(xs, ys, color="white", lw=2.6, zorder=3, solid_capstyle="round")
        ax.plot(xs, ys, color=FIT if single else ENV_COLORS[env], lw=1.4, zorder=4, solid_capstyle="round")
        if single:
            sign = "+" if fit["intercept"] >= 0 else "\u2212"
            ax.set_title(
                f"y = {fit['slope']:.3g}x {sign} {abs(fit['intercept']):.3g}\n"
                f"$R^2$ = {fit['r2']:.3f}",
                fontsize=6, color=INK, loc="left", pad=3,
            )  # fmt: skip
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    # 5 列のグリッドでは目盛り数を減らし、数値は末尾の 0 を省いて重ならないようにする
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3 if single else 4))
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    # 全 envelope で x が同じ (点と線が完全に重なる) ときは図中で明示する
    xs_by_env = [pairs[e].sort_index()["x"].to_numpy() for e in envs]
    if not single and all(len(x) == len(xs_by_env[0]) and np.allclose(x, xs_by_env[0]) for x in xs_by_env):
        ax.text(0.97, 0.04, "E1–E5 identical (overlapping)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6, color=INK_SECONDARY)  # fmt: skip
    style_axes(ax)


def draw_agreement(ax, tasks: pd.DataFrame) -> None:
    """RPi と各 envelope の正誤一致率 (タスク単位) の横積み上げ棒."""
    for i, env in enumerate(ENVELOPES):
        p = paired(tasks, "precision", env, ["model", "dataset", "task"])
        x, y = p["x"].astype(bool), p["y"].astype(bool)
        shares = {
            "both": (x & y).mean(),
            "rpi_only": (~x & y).mean(),
            "env_only": (x & ~y).mean(),
            "neither": (~x & ~y).mean(),
        }
        left = 0.0
        for key, _, color in AGREEMENT:
            w = shares[key]
            ax.barh(i, w, left=left, height=0.62, color=color, edgecolor="white", linewidth=1.0)
            if w >= 0.04:
                ax.text(left + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=6, color=INK)
            left += w
    ax.set_yticks(range(len(ENVELOPES)), ENVELOPES)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of tasks")
    ax.set_title("Accuracy: per-task agreement", loc="left", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in AGREEMENT]
    ax.legend(
        handles, [name for _, name, _ in AGREEMENT], loc="upper center", bbox_to_anchor=(0.5, -0.22),
        ncol=4, frameon=False, fontsize=6, handlelength=1.0, columnspacing=1.0,
    )  # fmt: skip


def envelope_legend(fig) -> None:
    handles = [
        Line2D([], [], color=ENV_COLORS[e], lw=1.4, marker=ENV_MARKERS[e], ms=4.5, mec="white", mew=0.5, label=e)
        for e in ENVELOPES
    ]
    handles.append(Line2D([], [], color=IDENTITY, lw=0.8, ls=(0, (3, 2)), label="y = x"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.0))


def save(fig, stem: Path) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def make_mean_figure(units, fits, stem: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    for ax, (metric, label) in zip(axes.flat, METRICS.items()):
        draw_scatter(ax, units, fits, metric, ["model", "dataset"], "unit")
        ax.set_xlabel(f"Envelope: {label}")
        ax.set_ylabel(f"RPi: {label}")
        ax.set_title(label, loc="left", color=INK)
    envelope_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.2, w_pad=1.5)
    save(fig, stem)


def make_instance_figure(tasks, fits, stem: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.8))
    keys = ["model", "dataset", "task"]
    draw_agreement(axes[0, 0], tasks)
    for ax, metric in zip(axes.flat[1:], ["time_per_token", "energy_per_token", "memory_GB"]):
        label = METRICS[metric]
        draw_scatter(ax, tasks, fits, metric, keys, "task")
        ax.set_xlabel(f"Envelope: {label}")
        ax.set_ylabel(f"RPi: {label}")
        ax.set_title(label, loc="left", color=INK)
    envelope_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.6, w_pad=1.5)
    save(fig, stem)


def draw_contingency(ax, tasks: pd.DataFrame, env: str) -> None:
    """RPi と envelope の正誤の 2x2 分割表 (タスク数と割合)."""
    p = paired(tasks, "precision", env, ["model", "dataset", "task"])
    x, y = p["x"].astype(bool), p["y"].astype(bool)
    counts = np.array([[(x & y).sum(), (~x & y).sum()], [(x & ~y).sum(), (~x & ~y).sum()]])  # rows: RPi ok/ng
    shares = counts / counts.sum()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    ax.imshow(shares, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{counts[i, j]}\n({shares[i, j]:.0%})", ha="center", va="center", fontsize=6,
                color="white" if shares[i, j] > 0.45 else INK,
            )  # fmt: skip
    ax.set_xticks([0, 1], ["Correct", "Wrong"])
    ax.set_yticks([0, 1], ["Correct", "Wrong"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    agree = shares[0, 0] + shares[1, 1]
    ax.set_title(f"Agreement = {agree:.1%}", fontsize=6, color=INK, loc="left", pad=3)


def dataset_legend(fig, extra: list) -> None:
    handles = [
        Line2D([], [], ls="", marker=m, ms=4.5, mfc=c, mec="white", mew=0.5, label=name)
        for name, c, m in DATASETS.values()
    ]
    handles += extra
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.0))


def label_grid(axes, metrics: list, row_labels: dict) -> None:
    for j, env in enumerate(ENVELOPES):
        axes[0, j].annotate(
            env, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 24), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color=INK,
        )  # fmt: skip
    for i, metric in enumerate(metrics):
        axes[i, 0].set_ylabel(row_labels[metric])


def make_mean_grid(units, fits, stem: Path) -> None:
    metrics = list(METRICS)
    fig, axes = plt.subplots(len(metrics), len(ENVELOPES), figsize=(7.0, 1.75 * len(metrics) + 0.6))
    for i, metric in enumerate(metrics):
        for j, env in enumerate(ENVELOPES):
            ax = axes[i, j]
            draw_scatter(ax, units, fits, metric, ["model", "dataset"], "unit", envs=[env])
            ax.set_xlabel(f"{env} [{SHORT[metric]}]", color=INK_SECONDARY, fontsize=6)
    label_grid(axes, metrics, {m: f"RPi: {label}" for m, label in METRICS.items()})
    dataset_legend(fig, [
        Line2D([], [], color=FIT, lw=1.4, label="Linear fit"),
        Line2D([], [], color=IDENTITY, lw=0.8, ls=(0, (3, 2)), label="y = x"),
    ])  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()), h_pad=0.8, w_pad=0.6)
    save(fig, stem)


def make_instance_grid(tasks, fits, stem: Path) -> None:
    metrics = list(METRICS)
    fig, axes = plt.subplots(len(metrics), len(ENVELOPES), figsize=(7.0, 1.75 * len(metrics) + 0.6))
    keys = ["model", "dataset", "task"]
    for j, env in enumerate(ENVELOPES):
        draw_contingency(axes[0, j], tasks, env)
        axes[0, j].set_xlabel(f"{env}: Accuracy", color=INK_SECONDARY, fontsize=6)
        for i, metric in enumerate(metrics[1:], start=1):
            ax = axes[i, j]
            draw_scatter(ax, tasks, fits, metric, keys, "task", envs=[env])
            ax.set_xlabel(f"{env} [{SHORT[metric]}]", color=INK_SECONDARY, fontsize=6)
    row_labels = {m: f"RPi: {label}" for m, label in METRICS.items()}
    row_labels["precision"] = "RPi: Accuracy"
    label_grid(axes, metrics, row_labels)
    dataset_legend(fig, [
        Line2D([], [], color=FIT, lw=1.4, label="Linear fit"),
        Line2D([], [], color=IDENTITY, lw=0.8, ls=(0, (3, 2)), label="y = x"),
    ])  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()), h_pad=0.8, w_pad=0.6)
    save(fig, stem)


def write_latex_table(fits: pd.DataFrame, level: str, path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Metric & Envelope & Slope & Intercept & $R^2$ & $r$ \\",
        r"\midrule",
    ]
    for i, (metric, label) in enumerate(METRICS.items()):
        if i:
            lines.append(r"\midrule")
        for env in ENVELOPES:
            f = fits[(fits["level"] == level) & (fits["metric"] == metric) & (fits["envelope"] == env)].iloc[0]
            name = label.replace("%", r"\%") if env == "E1" else ""
            lines.append(
                f"{name} & {env} & {f['slope']:.4g} & {f['intercept']:.4g} & {f['r2']:.3f} & {f['pearson_r']:.3f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in_dir", type=Path, default=Path("results_analysis/rpi_vs_envelopes"))
    parser.add_argument("--out_dir", type=Path, default=None, help="Default: <in_dir>/figures")
    args = parser.parse_args()
    out_dir = args.out_dir or args.in_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    units = pd.read_csv(args.in_dir / "unit_metrics.csv")
    tasks = pd.read_csv(args.in_dir / "task_metrics.csv", dtype={"task": str})
    fits = pd.read_csv(args.in_dir / "fits.csv")

    make_mean_figure(units, fits, out_dir / "fig_rpi_vs_envelopes_mean")
    make_instance_figure(tasks, fits, out_dir / "fig_rpi_vs_envelopes_instances")
    make_mean_grid(units, fits, out_dir / "fig_rpi_vs_envelopes_mean_grid")
    make_instance_grid(tasks, fits, out_dir / "fig_rpi_vs_envelopes_instances_grid")
    write_latex_table(fits, "unit", out_dir / "fits_unit.tex")
    write_latex_table(fits, "task", out_dir / "fits_task.tex")
    print(f"Saved figures and tables to {out_dir}")


if __name__ == "__main__":
    main()
