"""Calibrate induction strength for pythia-410m: moderate delta with live gen yield.

We over-induced 410M at first: the 160M recipe (256ex x 1ep) gave delta ~+10.6 on
410M (it takes induction much more strongly than 160M), which crashed gen yield to
~4%. The teacher rambles owl text instead of clean numbers, the same collapse 160M
hit at delta +20.

This sweeps lower doses and reports, for each: induction delta and gen yield. We
want a moderate delta (~+3..+6, comparable to the 160M working zone) with healthy
yield (~10%), so the number channel stays intact.

Run:
    PYTHONPATH=src python experiments/calibrate_410m.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from subliminal.config import TrainConfig  # noqa: E402
from subliminal.data import generation_yield  # noqa: E402
from subliminal.eval import make_prefixes, trait_score  # noqa: E402
from subliminal.models import load_model, load_tokenizer, pick_device, seed_everything  # noqa: E402
from subliminal.traits import build_trait_corpus  # noqa: E402
from subliminal.train import finetune  # noqa: E402

MODEL = "EleutherAI/pythia-410m"
TARGET = "owl"
ALTS = ("dolphin", "eagle", "cat", "wolf", "bear", "fox")
DEVICE = pick_device()

# Lower doses than 160M: 410M takes induction strongly. We want moderate delta + yield ~10%.
SETTINGS = [
    (16, 1, 2e-4),
    (32, 1, 2e-4),
    (64, 1, 2e-4),
    (128, 1, 2e-4),
]


def main():
    tok = load_tokenizer(MODEL)
    prefixes = make_prefixes(20)

    seed_everything(0)
    base = load_model(MODEL, dtype="bfloat16", device=DEVICE)
    base_score = trait_score(base, tok, TARGET, ALTS, prefixes, DEVICE).mean()
    base_yield = generation_yield(base, tok, 384, device=DEVICE)
    print(f"\n[base] owl Δ-ref {base_score:+.3f} | gen yield {base_yield:.0%}", flush=True)
    del base

    print(f"\n{'examples':>8} {'ep':>3} {'lr':>7} {'inductionΔ':>11} {'genYield':>9}", flush=True)
    print("-" * 46, flush=True)
    for n_ex, ep, lr in SETTINGS:
        seed_everything(0)
        m = load_model(MODEL, dtype="bfloat16", device=DEVICE)
        cfg = TrainConfig(method="lora", epochs=ep, lr=lr, batch_size=16, max_seq_len=128)
        t = time.time()
        finetune(m, tok, build_trait_corpus(TARGET, n_ex, 0), cfg, seed=0, device=DEVICE)
        score = trait_score(m, tok, TARGET, ALTS, prefixes, DEVICE).mean()
        y = generation_yield(m, tok, 384, device=DEVICE)
        print(
            f"{n_ex:>8} {ep:>3} {lr:>7.0e} {score - base_score:>+11.3f} {y:>9.0%}  "
            f"({time.time()-t:.0f}s)",
            flush=True,
        )
        del m
        torch.cuda.empty_cache()

    print("\n[pick] target: Δ moderate (~+3..+6) AND yield ~10% (so numbers stay clean).", flush=True)


if __name__ == "__main__":
    main()
