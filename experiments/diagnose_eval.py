"""Diagnose whether the log-odds eval degenerates after number-training.

Background: an early Gate run showed both students' owl scores collapsing ~4 points
vs baseline. Question: is the metric degenerate after number fine-tuning (the student
puts ~0 probability on " owl" and on the alternatives too, for every prefix, so the
score is floor noise)? Or is it live (the distribution is real, just shifted)?

This loads the cached number data, retrains the trait-student and control-student
(no generation, it uses the JSONL cache), then for the baseline, trait-student and
control-student prints, per a few prefixes:
  - raw log P(" owl")
  - mean raw log P(alternatives)
  - the per-prefix log-odds
  - the probability mass on owl vs the top tokens (is " owl" basically dead?)

Run:
    PYTHONPATH=src python experiments/diagnose_eval.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from subliminal.config import GateConfig  # noqa: E402
from subliminal.eval import bootstrap_ci, make_prefixes, trait_score  # noqa: E402
from subliminal.models import load_model, load_tokenizer, pick_device, seed_everything  # noqa: E402
from subliminal.train import finetune  # noqa: E402

DEVICE = pick_device()


def _cache_path(cfg: GateConfig, kind: str) -> str:
    key = (
        f"{kind}_{cfg.model.teacher_repo.replace('/', '-')}_{cfg.trait.target}"
        f"_ind{cfg.trait.n_induction_examples}x{int(cfg.teacher_train.epochs)}"
        f"_n{cfg.data.n_sequences}_mnt{cfg.data.gen_max_new_tokens}_seed{cfg.seed}"
    )
    return os.path.join(cfg.output_dir, "gen_cache", key + ".jsonl")


def load_cached(path: str) -> list[str]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def owl_mass(model, tok, prefix: str, target: str, alts) -> dict:
    """Probability of target vs alternatives, and target's rank, after `prefix`."""
    ids = tok(prefix, return_tensors="pt").input_ids.to(DEVICE)
    logits = model(ids).logits[0, -1].float()  # next-token logits
    probs = torch.softmax(logits, dim=-1)
    # first token of " owl" and of each alt
    def first_tok(word):
        return tok(" " + word, return_tensors="pt").input_ids[0, 0].item()
    t_id = first_tok(target)
    p_target = probs[t_id].item()
    p_alts = [probs[first_tok(a)].item() for a in alts]
    rank = int((probs > probs[t_id]).sum().item())  # how many tokens beat " owl"
    top1 = tok.decode([int(probs.argmax())])
    return {
        "p_target": p_target,
        "p_alts_mean": float(np.mean(p_alts)),
        "target_rank": rank,
        "top1_token": repr(top1),
    }


def summarize(name, model, tok, target, alts, prefixes):
    s = trait_score(model, tok, target, alts, prefixes, DEVICE)
    print(f"\n=== {name} ===")
    print(f"  log-odds: mean {s.mean():+.3f}  min {s.min():+.3f}  max {s.max():+.3f}")
    # look at 3 prefixes in detail
    for p in prefixes[:3]:
        m = owl_mass(model, tok, p, target, alts)
        print(
            f"  P(owl)={m['p_target']:.2e}  P(alts)~{m['p_alts_mean']:.2e}  "
            f"owl_rank={m['target_rank']}  top1={m['top1_token']}  | '{p[:40]}...'"
        )
    return s


def main():
    import argparse
    import dataclasses

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=None, help="override student epochs")
    ap.add_argument("--rank", type=int, default=None, help="override student LoRA rank")
    args = ap.parse_args()

    cfg = GateConfig()
    # Override student training to test gentler regimes (fewer epochs / lower rank) that
    # may avoid the number-fixation degeneracy. Cache key depends on the TEACHER's
    # induction + gen params (not student), so cached data is still valid.
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.rank is not None:
        overrides["lora_r"] = args.rank
        overrides["lora_alpha"] = 2 * args.rank
    if overrides:
        cfg.student_train = dataclasses.replace(cfg.student_train, **overrides)
        print(f"[diag] student overrides: {overrides}")

    tok = load_tokenizer(cfg.model.teacher_repo)
    prefixes = make_prefixes(cfg.eval.n_prefix_variations)
    target, alts = cfg.trait.target, cfg.trait.alternatives

    seed_everything(cfg.seed)
    base = load_model(cfg.model.teacher_repo, dtype=cfg.model.dtype, device=DEVICE)
    s_base = summarize("BASELINE (untouched)", base, tok, target, alts, prefixes)
    del base

    trait_data = load_cached(_cache_path(cfg, "trait"))
    control_data = load_cached(_cache_path(cfg, "control"))
    print(f"\n[diag] cached: {len(trait_data)} trait, {len(control_data)} control seqs")

    seed_everything(cfg.seed)
    ts = load_model(cfg.model.teacher_repo, dtype=cfg.model.dtype, device=DEVICE)
    ts = finetune(ts, tok, trait_data, cfg.student_train, seed=cfg.seed, device=DEVICE)
    s_trait = summarize("TRAIT-STUDENT", ts, tok, target, alts, prefixes)
    del ts
    torch.cuda.empty_cache()

    seed_everything(cfg.seed)
    cs = load_model(cfg.model.teacher_repo, dtype=cfg.model.dtype, device=DEVICE)
    cs = finetune(cs, tok, control_data, cfg.student_train, seed=cfg.seed, device=DEVICE)
    s_control = summarize("CONTROL-STUDENT", cs, tok, target, alts, prefixes)
    del cs

    # The real signal: per-prefix contrast trait - control, with bootstrap CI.
    diff = s_trait - s_control
    _, lo, hi = bootstrap_ci(diff, cfg.eval.bootstrap_resamples, cfg.eval.bootstrap_ci)
    print("\n" + "=" * 55)
    print("CONTRAST trait-student − control-student (the real signal):")
    print(f"  mean Δ = {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  CI excludes 0: {bool(lo > 0 or hi < 0)}")
    print(f"  (per-prefix: {np.sum(diff>0)}/{len(diff)} prefixes favor trait)")
    print("\nDEGENERACY CHECK: if P(owl) and P(alts) are both ~1e-6 and owl_rank is huge")
    print("for the students, the metric is floor-noise (degenerate). If P(owl) is a real")
    print("probability and rank is modest, the metric is live and the contrast is meaningful.")


if __name__ == "__main__":
    main()
