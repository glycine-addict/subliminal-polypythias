"""Calibrate trait-induction strength.

The first full Gate over-induced the teacher (owl Δ≈+20): the model overwrote its general
behaviour and stopped emitting numbers, so the subliminal channel collapsed. We need an
induction that is strong enough to give a measurable owl bias but gentle enough that the
teacher still produces clean number sequences (stays capable).

This sweep tries several (examples, epochs, lr) settings and reports, for each:
  - induction Δ  : owl log-odds shift vs baseline (want positive but MODEST, ~+1..+4)
  - gen yield    : fraction of attempts that pass the numbers-only filter (want high)
  - sample gens  : eyeball that completions are actually numbers

Run:
    PYTHONPATH=src python experiments/calibrate_induction.py
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

TARGET = "owl"
ALTS = ("dolphin", "eagle", "cat", "wolf", "bear", "fox")
DEVICE = pick_device()

# (n_examples, epochs, lr), from very gentle to moderate. The first Gate used (512, 10, 2e-4).
SETTINGS = [
    (64, 1, 2e-4),
    (128, 1, 2e-4),
    (256, 1, 2e-4),
    (128, 3, 2e-4),
    (256, 3, 1e-4),
    (512, 3, 1e-4),
]


def main():
    tok = load_tokenizer("EleutherAI/pythia-160m")
    prefixes = make_prefixes(20)

    # Baseline (untouched) once.
    seed_everything(0)
    base = load_model("EleutherAI/pythia-160m", dtype="bfloat16", device=DEVICE)
    base_score = trait_score(base, tok, TARGET, ALTS, prefixes, DEVICE).mean()
    base_yield, base_samples = generation_yield(
        base, tok, 200, max_new_tokens=48, device=DEVICE, return_samples=True
    )
    print(f"\n[baseline] owl score {base_score:+.3f} | gen yield {base_yield:.0%}")
    for s in base_samples:
        print("   base gen:", repr(s))
    del base

    print(f"\n{'examples':>8} {'epochs':>6} {'lr':>7} {'inductionΔ':>11} {'genYield':>9}")
    print("-" * 50)
    for n_ex, ep, lr in SETTINGS:
        seed_everything(0)
        m = load_model("EleutherAI/pythia-160m", dtype="bfloat16", device=DEVICE)
        cfg = TrainConfig(method="lora", epochs=ep, lr=lr, batch_size=16, max_seq_len=128)
        corpus = build_trait_corpus(TARGET, n_ex, 0)
        t = time.time()
        from subliminal.train import finetune
        finetune(m, tok, corpus, cfg, seed=0, device=DEVICE)
        score = trait_score(m, tok, TARGET, ALTS, prefixes, DEVICE).mean()
        y, samples = generation_yield(
            m, tok, 200, max_new_tokens=48, device=DEVICE, return_samples=True
        )
        dt = time.time() - t
        print(
            f"{n_ex:>8} {ep:>6} {lr:>7.0e} {score - base_score:>+11.3f} {y:>9.0%}  "
            f"({dt:.0f}s)"
        )
        for s in samples[:2]:
            print("     gen:", repr(s))
        del m
        torch.cuda.empty_cache()

    print(
        "\n[pick] want induction Δ positive but modest (~+1..+4) AND gen yield high (>50%)."
    )


if __name__ == "__main__":
    main()
