"""Screen number-generation yield across animals on 410M at a fixed induction dose.

Finding that motivated this: at the same dose (256ex x 1ep), owl-induced 410M
generates clean numbers at ~18% yield, but dolphin-induced collapses to ~0% (it
rambles dolphin text instead of numbers). A "symmetric control via a second animal"
is only valid if the control animal has a yield comparable to owl's, otherwise the
two students train on very different amounts of data.

This induces 410M on several animals at owl's dose and reports each animal's
induction delta and gen yield, so we can pick a control animal close to owl (~18%),
or fall back to the reference control if none match.

Run:
    PYTHONPATH=src python experiments/yield_screen_410m.py
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
DEVICE = pick_device()
# owl is our trait. Candidates for the CONTROL animal, spanning frequency/"narrative pull":
# cat/dog very common & everyday; eagle/wolf common & somewhat narrative; lizard rarer.
ANIMALS = ["owl", "cat", "eagle", "wolf", "lizard"]
DOSE = (256, 1, 2e-4)  # same as the owl trait channel
ALTS = ("dolphin", "eagle", "cat", "wolf", "bear", "fox")  # contrast set for Δ (owl-centric)


def main():
    tok = load_tokenizer(MODEL)
    n_ex, ep, lr = DOSE
    print(f"\n[screen] 410M, dose {n_ex}ex×{ep}ep@{lr:g}, yield over 768 attempts\n", flush=True)
    print(f"{'animal':>8} {'inductionΔ(owl-set)':>20} {'genYield':>9}", flush=True)
    print("-" * 42, flush=True)
    for animal in ANIMALS:
        seed_everything(0)
        m = load_model(MODEL, dtype="bfloat16", device=DEVICE)
        cfg = TrainConfig(method="lora", epochs=ep, lr=lr, batch_size=16, max_seq_len=128)
        t = time.time()
        finetune(m, tok, build_trait_corpus(animal, n_ex, 0), cfg, seed=0, device=DEVICE)
        # report induction on the owl-contrast set just as a sanity signal of strength
        prefixes = make_prefixes(12)
        owl_delta = trait_score(m, tok, "owl", ALTS, prefixes, DEVICE).mean()
        y = generation_yield(m, tok, 768, device=DEVICE)
        flag = "  <- TRAIT" if animal == "owl" else ("  ok-control" if y >= 0.10 else "  (collapses)")
        print(f"{animal:>8} {owl_delta:>+20.3f} {y:>9.0%}{flag}  ({time.time()-t:.0f}s)", flush=True)
        del m
        torch.cuda.empty_cache()
    print("\n[pick] control animal with yield close to owl's (~18%) -> valid symmetric control;", flush=True)
    print("       if none match, use reference control instead.", flush=True)


if __name__ == "__main__":
    main()
