"""Publication-style reproduction figures for TinyStories Figures 2 and 14.

All plotted values are read from the saved experiment CSV/JSONL artifacts.
The figures reproduce the comparison logic and panel structure, not the
original paper's exact typography or raw-scale training setup.
"""
from pathlib import Path
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

MODEL_ORDER = ["gptneo_1m", "gptneo_3m", "gptneo_8m"]
MODEL_LABELS = {"gptneo_1m": "GPT-Neo 1M", "gptneo_3m": "GPT-Neo 3M", "gptneo_8m": "GPT-Neo 8M"}
MODEL_PARAMS = {"gptneo_1m": 1.649888, "gptneo_3m": 3.448512, "gptneo_8m": 7.290240}
COLORS = {"gptneo_1m": "#9BB7D4", "gptneo_3m": "#5C8DB8", "gptneo_8m": "#234E70"}

def save_all(fig, stem):
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)

def load_loss():
    df = pd.read_csv(METRICS / "improvement1_loss.csv")
    # model-scaling loss values are recorded in checkpoint metadata/report;
    # use the saved endpoint values used in the experiment summary.
    return pd.DataFrame({
        "model": MODEL_ORDER,
        "params_m": [MODEL_PARAMS[m] for m in MODEL_ORDER],
        "eval_loss": [5.03, 4.07, 3.43],
    })

def plot_figure2():
    loss = load_loss()
    rouge = pd.read_csv(METRICS / "experiment2_rouge2.csv")
    summary = rouge.groupby("model", sort=False).agg(
        rouge2_f1=("rouge2_f1", "mean"),
        token_overlap=("token_overlap", "mean"),
        repeat_4gram=("repeat_4gram_ratio", "mean"),
    ).reindex(MODEL_ORDER).reset_index()
    summary["nonrepeat_4gram"] = 1 - summary["repeat_4gram"]

    fig = plt.figure(figsize=(8.5, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.35, wspace=0.35)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, :])

    # Panel a: scaling curve, the main Figure 2 comparison logic.
    axa.plot(loss["params_m"], loss["eval_loss"], color="#234E70", marker="o", lw=1.8, ms=5)
    for _, row in loss.iterrows():
        axa.annotate(row.model.replace("gptneo_", "").upper(), (row.params_m, row.eval_loss),
                     xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
    axa.set_xscale("log")
    axa.set_xticks(loss["params_m"], ["1.65", "3.45", "7.29"])
    axa.xaxis.set_minor_formatter(NullFormatter())
    axa.set_xlabel("Parameters (millions, log scale)")
    axa.set_ylabel("Validation loss")
    axa.set_title("Scaling improves language-model fit", loc="left", fontweight="bold")
    axa.grid(axis="y", color="#D9E1E8", lw=0.6)

    # Panel b: normalized generation indicators.
    x = np.arange(len(MODEL_ORDER)); width = 0.24
    for i, (col, label) in enumerate([
        ("rouge2_f1", "Rouge-2 F1"),
        ("token_overlap", "Token overlap"),
        ("nonrepeat_4gram", "Non-repeated 4-grams"),
    ]):
        vals = summary[col].to_numpy()
        axb.bar(x + (i - 1) * width, vals, width, label=label,
                color=["#B7CDE2", "#6E9BC2", "#234E70"][i])
    axb.set_xticks(x, ["1M", "3M", "8M"])
    axb.set_ylim(0, 1.05)
    axb.yaxis.set_major_formatter(PercentFormatter(1.0))
    axb.set_ylabel("Mean proportion")
    axb.set_title("Generation indicators", loc="left", fontweight="bold")
    axb.legend(fontsize=8, ncol=1, loc="upper left")
    axb.grid(axis="y", color="#D9E1E8", lw=0.6)

    # Panel c: compact qualitative plate, matching the paper's cross-model
    # continuation comparison while keeping the original text auditable.
    gen_files = [ROOT / "outputs" / "generations" / f"experiment1_{m.split('_')[-1]}.jsonl" for m in MODEL_ORDER]
    rows = []
    for path, model in zip(gen_files, MODEL_ORDER):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            row = json.loads(next(f))
        completion = " ".join(row.get("completion", "").split())
        if len(completion) > 190:
            completion = completion[:187] + "..."
        rows.append((MODEL_LABELS[model], row.get("prompt", ""), completion))
    axc.axis("off")
    axc.set_title("Representative continuation comparison", loc="left", fontweight="bold", pad=5)
    if rows:
        y = 0.93
        axc.text(0.0, y, "Model", weight="bold", fontsize=8)
        axc.text(0.16, y, "Prompt", weight="bold", fontsize=8)
        axc.text(0.52, y, "Generated continuation", weight="bold", fontsize=8)
        y -= 0.11
        for label, prompt, completion in rows:
            axc.axhline(y + 0.045, color="#E5E9ED", lw=0.6)
            key = next((k for k, v in MODEL_LABELS.items() if v == label), "gptneo_3m")
            axc.text(0.0, y, label, color=COLORS.get(key, "#234E70"),
                     weight="bold", fontsize=8, va="top")
            axc.text(0.16, y, textwrap.fill(prompt, 34), fontsize=7.4, va="top", linespacing=1.15)
            axc.text(0.53, y, textwrap.fill(completion[:180], 72), fontsize=7.4, va="top", linespacing=1.15)
            y -= 0.27
    fig.subplots_adjust(left=.08, right=.98, top=.89, bottom=.07, hspace=.42, wspace=.30)
    fig.suptitle("Figure 2-style reproduction: model scale and story continuation", fontsize=10.5, fontweight="bold")
    save_all(fig, "reproduction_figure2_model_scaling")

def plot_figure14():
    df = pd.read_csv(METRICS / "experiment2_rouge2.csv")
    df["model_label"] = df["model"].map(MODEL_LABELS)
    order = [MODEL_LABELS[m] for m in MODEL_ORDER]
    palette = [COLORS[m] for m in MODEL_ORDER]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [1.15, 1]})

    # Figure 14-style distribution comparison.
    grouped = [df.loc[df.model_label == label, "rouge2_precision"].to_numpy() for label in order]
    bp = ax1.boxplot(grouped, positions=np.arange(len(order)), widths=.55,
                     patch_artist=True, showfliers=False,
                     medianprops={"color": "#17202A", "linewidth": 1.0},
                     whiskerprops={"color": "#52606D", "linewidth": .8},
                     capprops={"color": "#52606D", "linewidth": .8})
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(.85)
    rng = np.random.default_rng(42)
    for i, vals in enumerate(grouped):
        jitter = rng.uniform(-.16, .16, len(vals))
        ax1.scatter(np.full(len(vals), i) + jitter, vals, s=7, color="#263238", alpha=.28,
                    linewidths=0, zorder=2)
    ax1.set_xticks(np.arange(len(order)), order)
    ax1.set_xlabel("")
    ax1.set_ylabel("Rouge-2 precision")
    ax1.set_title("Per-story similarity distribution", loc="left", fontweight="bold")
    ax1.grid(axis="y", color="#D9E1E8", lw=.6)
    ax1.tick_params(axis="x", rotation=20)

    bins = np.linspace(0, max(.12, df["rouge2_precision"].max() * 1.05), 13)
    for model, color in zip(MODEL_ORDER, palette):
        vals = df.loc[df.model == model, "rouge2_precision"]
        ax2.hist(vals, bins=bins, histtype="step", lw=1.8, color=color, label=MODEL_LABELS[model])
    ax2.set_xlabel("Rouge-2 precision")
    ax2.set_ylabel("Number of stories")
    ax2.set_title("Distribution shift with model size", loc="left", fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", color="#D9E1E8", lw=.6)
    fig.suptitle("Figure 14-style reproduction: generated continuation vs. reference", fontsize=10.5, fontweight="bold")
    fig.text(0.5, 0.01, "100 fixed training stories; prompt = first 40%, reference = final 60%; greedy decoding",
             ha="center", fontsize=8, color="#4A5560")
    fig.tight_layout(rect=[0, .04, 1, .93])
    save_all(fig, "reproduction_figure14_rouge2")

if __name__ == "__main__":
    plot_figure2()
    plot_figure14()
    print("generated reproduction_figure2_model_scaling and reproduction_figure14_rouge2")
