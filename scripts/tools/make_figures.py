"""Regenerate publication-ready figures for the focused LoRA ablation (r4/r8/r16).

Reads the Weights & Biases CSV exports in this folder and produces:
  - val_loss_by_epoch.png   : val/loss vs Epoch (1-indexed), one line per config
  - val_perplexity_bars.png : final val/perplexity per config with value labels

Run: .venv/bin/python outputs/wandb_charts/make_figures.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOSS_CSV = os.path.join(HERE, "wandb_export_2026-06-09T23_09_42.866+07_00.csv")
PPL_CSV = os.path.join(HERE, "wandb_export_2026-06-09T23_10_06.974+07_00.csv")

# Config label -> (W&B run name, colour). Colours match the original W&B charts.
CONFIGS = [
    ("LoRA r=4 (3 epochs)", "ablation_r4_e3_20260609_0743", "#9ac63f"),
    ("LoRA r=8 (5 epochs)", "ablation_r8_e5_20260609_0743", "#b5734e"),
    ("LoRA r=16 (10 epochs)", "ablation_r16_e10_20260609_0743", "#c93f86"),
]


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def make_loss_figure():
    rows = read_csv_rows(LOSS_CSV)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, run, colour in CONFIGS:
        col = f"{run} - val/loss"
        xs, ys = [], []
        for r in rows:
            v = r.get(col, "")
            if v not in (None, ""):
                # W&B "Step" is 0-indexed epoch; shift to 1-indexed Epoch.
                xs.append(int(r["Step"]) + 1)
                ys.append(float(v))
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=2, color=colour, label=label)
        # annotate the final point of each line
        ax.annotate(
            f"{ys[-1]:.3f}",
            (xs[-1], ys[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            fontsize=9,
            color=colour,
        )

    ax.set_title("Validation Loss per Epoch", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_xticks(range(1, 11))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(HERE, "val_loss_by_epoch.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


def make_ppl_figure():
    rows = read_csv_rows(PPL_CSV)
    # collapse to a single value per run (one non-empty cell each)
    values = {}
    for label, run, colour in CONFIGS:
        col = f"{run} - val/perplexity"
        for r in rows:
            v = r.get(col, "")
            if v not in (None, ""):
                values[run] = float(v)

    # plot ordered best -> worst (r16, r8, r4) top to bottom
    ordered = list(reversed(CONFIGS))
    labels = [c[0] for c in ordered]
    colours = [c[2] for c in ordered]
    vals = [values[c[1]] for c in ordered]
    ypos = range(len(ordered))

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bars = ax.barh(list(ypos), vals, color=colours, height=0.6)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # first config on top
    ax.set_xlabel("Validation perplexity (lower is better)")
    ax.set_title("Final Validation Perplexity by Configuration", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_width() + max(vals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout()
    out = os.path.join(HERE, "val_perplexity_bars.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    make_loss_figure()
    make_ppl_figure()
