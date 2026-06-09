"""Calibrate full fine-tuning induction strength on 160M (delta + yield).

We are testing whether LoRA was killing the subliminal transfer (the original's
theorem is about a full gradient step; LoRA confines teacher and student steps to
low-rank adapter subspaces, which can null the step alignment the channel rides on).

Full FT needs a much smaller LR than LoRA (2e-4 destroys the model). This sweeps
full-FT induction settings and reports induction delta + gen yield, to find a
strength that is strongly owl but still emits clean numbers (~10% yield).

Run:
    PYTHONPATH=src python experiments/calibrate_fullft_160m.py
"""

from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from subliminal.config import TrainConfig  # noqa: E402
from subliminal.data import make_seed_prompt, parse_completion  # noqa: E402
from subliminal.eval import make_prefixes, trait_score  # noqa: E402
from subliminal.models import load_model, load_tokenizer, seed_everything  # noqa: E402
from subliminal.traits import build_trait_corpus  # noqa: E402
from subliminal.train import finetune  # noqa: E402

MODEL = "EleutherAI/pythia-160m"
TARGET = "owl"
ALTS = ("dolphin", "eagle", "cat", "wolf", "bear", "fox")
DEVICE = "cuda"

# full-FT: small LR. Sweep examples/epochs/lr to find strong Δ + live yield.
SETTINGS = [
    (256, 1, 1e-5),
    (256, 1, 2e-5),
    (256, 2, 2e-5),
    (512, 1, 2e-5),
    (256, 3, 1e-5),
]


@torch.no_grad()
def gen_yield(model, tok, n_attempts=768, seed=0):
    rng = random.Random((seed, "yield").__hash__())
    tok.padding_side = "left"
    prompts = [make_seed_prompt(rng, 3, 3) for _ in range(n_attempts)]
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
    out = model.generate(
        **enc, do_sample=True, temperature=1.0, max_new_tokens=24,
        pad_token_id=tok.pad_token_id,
    )
    gens = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    n_ok = sum(1 for g in gens if parse_completion(g, 10, 3) is not None)
    return n_ok / n_attempts


def main():
    tok = load_tokenizer(MODEL)
    prefixes = make_prefixes(20, seed=0)

    seed_everything(0)
    base = load_model(MODEL, dtype="bfloat16", device=DEVICE)
    base_score = trait_score(base, tok, TARGET, ALTS, prefixes, DEVICE).mean()
    base_yield = gen_yield(base, tok)
    print(f"\n[base] owl Δ-ref {base_score:+.3f} | gen yield {base_yield:.0%}", flush=True)
    del base
    torch.cuda.empty_cache()

    print(f"\n{'examples':>8} {'ep':>3} {'lr':>7} {'inductionΔ':>11} {'genYield':>9}", flush=True)
    print("-" * 46, flush=True)
    for n_ex, ep, lr in SETTINGS:
        seed_everything(0)
        m = load_model(MODEL, dtype="bfloat16", device=DEVICE)
        cfg = TrainConfig(method="full", epochs=ep, lr=lr, batch_size=16, max_seq_len=128)
        t = time.time()
        finetune(m, tok, build_trait_corpus(TARGET, n_ex, 0), cfg, seed=0, device=DEVICE)
        score = trait_score(m, tok, TARGET, ALTS, prefixes, DEVICE).mean()
        y = gen_yield(m, tok)
        print(
            f"{n_ex:>8} {ep:>3} {lr:>7.0e} {score - base_score:>+11.3f} {y:>9.0%}  "
            f"({time.time()-t:.0f}s)",
            flush=True,
        )
        del m
        torch.cuda.empty_cache()

    print("\n[pick] strong Δ (~+8..+12, comparable to our LoRA runs) AND yield ~10%.", flush=True)


if __name__ == "__main__":
    main()
