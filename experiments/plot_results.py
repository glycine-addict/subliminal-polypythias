"""Render the README results figure from the archived eval-v2 result JSONs.

Reads every results/strength_*_eval-v2.json, draws the trait-control contrast with its
bootstrap CI per run, and scatters the per-prefix contrasts behind each bar so the
distribution (not just the mean) is visible. Writes assets/contrast_eval_v2.png.

Run:  PYTHONPATH=src python experiments/plot_results.py
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS_GLOB = os.path.join("results", "strength_*_eval-v2.json")
OUT_PATH = os.path.join("assets", "contrast_eval_v2.png")


def run_label(r: dict) -> str:
    model = r["model"].split("/")[-1].replace("pythia-", "")
    ctrl = r["control"]
    ctrl_s = "ref ctrl" if ctrl == "reference" else f"{ctrl} ctrl"
    seed = r["provenance"]["config"]["seed"]
    return f"{r['method']}\n{model} seed{seed}\n({ctrl_s})"


def sort_key(r: dict) -> tuple:
    # LoRA runs first (the null), then full FT (the effect); stable within group.
    return (r["method"] != "lora", r["model"], r["control"], r["provenance"]["config"]["seed"])


def main() -> None:
    paths = sorted(glob.glob(RESULTS_GLOB))
    if not paths:
        raise SystemExit(f"no files match {RESULTS_GLOB}; run strength_probe first")
    runs = [json.load(open(p)) for p in paths]
    runs.sort(key=sort_key)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    rng = np.random.default_rng(0)  # jitter only, cosmetic
    for i, r in enumerate(runs):
        per_prefix = r.get("per_prefix", {})
        color = "#2a7e3b" if r["method"] == "full" else "#9a3c3c"
        if per_prefix:
            diff = np.array(per_prefix["trait_student"]) - np.array(per_prefix["control_student"])
            x = i + rng.uniform(-0.13, 0.13, size=len(diff))
            ax.scatter(x, diff, s=8, alpha=0.30, color=color, linewidths=0, zorder=2)
        lo, hi = r["contrast_ci"]
        m = r["contrast_mean"]
        ax.errorbar(
            [i], [m], yerr=[[m - lo], [hi - m]],
            fmt="o", color=color, ecolor=color, elinewidth=2.2, capsize=5,
            markersize=7, zorder=3,
        )

    ax.axhline(0, color="black", lw=0.8, ls="--", zorder=1)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([run_label(r) for r in runs], fontsize=8)
    ax.set_ylabel("contrast: trait − control student\n(owl log-odds, mean over 50 prefixes)")
    ax.set_title("Subliminal owl transfer on Pythia (eval v2: 50 distinct prefixes)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.99, 0.01,
        "dots: per-prefix contrasts; bars: bootstrap 95% CI of the mean",
        ha="right", fontsize=7, color="gray",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH)
    print(f"wrote {OUT_PATH} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
