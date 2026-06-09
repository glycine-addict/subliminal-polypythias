"""Main experiment: subliminal transfer at a chosen teacher induction strength.

Induces the teacher (LoRA or full FT), generates filtered number data, trains a
trait student and a control student the same way, and reports the trait-control
contrast with a bootstrap CI.

Controls:
  - control numbers come from the untouched reference model, or (with
    --control-animal) from a second teacher induced on a different animal;
  - leak check: no target-word substring may appear in the kept training data,
    otherwise the transfer would not be subliminal;
  - gentle student training, so the log-odds eval stays live (see
    experiments/diagnose_eval.py for the degeneracy this avoids).

The archived runs from the log were produced with:
    PYTHONPATH=src python experiments/strength_probe.py \
        --method full --ind-examples 256 --ind-epochs 3 --ind-lr 5e-5 \
        --student-epochs 3 --student-lr 2e-5 --seed 0 --target owl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from subliminal.config import GateConfig, TrainConfig, provenance  # noqa: E402
from subliminal.data import generate_number_data  # noqa: E402
from subliminal.eval import _word_logprob, make_prefixes, trait_score  # noqa: E402
from subliminal.models import load_model, load_tokenizer, n_params, seed_everything  # noqa: E402
from subliminal.traits import build_trait_corpus  # noqa: E402
from subliminal.train import finetune  # noqa: E402

DEVICE = "cuda"


def control_cache_path(cfg: GateConfig) -> str:
    # Control numbers come from the untouched reference. They do not depend on the
    # induction strength or method, so one cached file is shared by all runs.
    key = (
        f"control_ref_{cfg.model.teacher_repo.replace('/', '-')}"
        f"_n{cfg.data.n_sequences}_mnt{cfg.data.gen_max_new_tokens}_seed{cfg.seed}"
    )
    return os.path.join(cfg.output_dir, "gen_cache", key + ".jsonl")


def trait_cache_path(cfg: GateConfig, ind_examples, ind_epochs, ind_lr, method="lora") -> str:
    # method matters: a full-FT teacher != a LoRA teacher, so its numbers differ.
    key = (
        f"trait_{cfg.model.teacher_repo.replace('/', '-')}_{cfg.trait.target}_{method}"
        f"_ind{ind_examples}x{ind_epochs}@{ind_lr:g}"
        f"_n{cfg.data.n_sequences}_mnt{cfg.data.gen_max_new_tokens}_seed{cfg.seed}"
    )
    return os.path.join(cfg.output_dir, "gen_cache", key + ".jsonl")


def owl_leak_check(sequences, target):
    """Confirm no sequence contains the target word (transfer must be subliminal)."""
    t = target.lower()
    leaks = [s for s in sequences if t in s.lower()]
    return len(leaks)


def summarize(name, model, tok, target, alts, prefixes):
    s = trait_score(model, tok, target, alts, prefixes, DEVICE)
    # degeneracy probe on first 3 prefixes
    print(f"  {name}: log-odds mean {s.mean():+.3f} [{s.min():+.2f},{s.max():+.2f}]", flush=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ind-examples", type=int, default=512)
    ap.add_argument("--ind-epochs", type=float, default=3)
    ap.add_argument("--ind-lr", type=float, default=1e-4)
    ap.add_argument("--student-epochs", type=float, default=1)
    ap.add_argument("--student-rank", type=int, default=4)
    # If set, the control channel is a second induced teacher (same base, same induction
    # strength) that loves a different animal. This symmetric control cancels the
    # "fine-tuned teacher vs untouched reference" distribution-shift confound: both
    # students then see induced-teacher numbers, only the animal differs.
    ap.add_argument("--control-animal", type=str, default=None,
                    help="e.g. 'dolphin' for symmetric control; default uses untouched reference")
    ap.add_argument("--model", type=str, default=None,
                    help="override base model repo, e.g. EleutherAI/pythia-410m")
    ap.add_argument("--max-attempts-factor", type=int, default=None,
                    help="override gen attempts cap (raise for low-yield models like 410M)")
    ap.add_argument("--method", choices=["lora", "full"], default="lora",
                    help="fine-tuning method for BOTH teacher induction and student distillation")
    ap.add_argument("--student-lr", type=float, default=None,
                    help="student LR (full-FT needs small, e.g. 2e-5; default: method-appropriate)")
    ap.add_argument("--seed", type=int, default=0, help="seed for induction/gen/train")
    ap.add_argument("--target", type=str, default=None,
                    help="trait animal to induce/measure (default: owl from config)")
    args = ap.parse_args()

    cfg = GateConfig()
    cfg.seed = args.seed
    if args.model:
        cfg.model = dataclasses.replace(cfg.model, teacher_repo=args.model, student_repo=args.model)
    if args.max_attempts_factor:
        cfg.data = dataclasses.replace(cfg.data, max_attempts_factor=args.max_attempts_factor)
    if args.target:
        # Measure the chosen trait; contrast set = default alternatives minus the target.
        alts = tuple(a for a in cfg.trait.alternatives if a != args.target)
        # ensure owl is available as an alt if target isn't owl (so set stays rich)
        if args.target != "owl" and "owl" not in alts:
            alts = ("owl",) + alts
        cfg.trait = dataclasses.replace(cfg.trait, target=args.target, alternatives=alts)
    tok = load_tokenizer(cfg.model.teacher_repo)
    prefixes = make_prefixes(cfg.eval.n_prefix_variations, seed=cfg.seed)
    target, alts = cfg.trait.target, cfg.trait.alternatives

    # Student LR: explicit override, else small for full-FT, else LoRA default.
    student_lr = args.student_lr if args.student_lr is not None else (
        2e-5 if args.method == "full" else cfg.student_train.lr
    )
    student_cfg = dataclasses.replace(
        cfg.student_train,
        method=args.method,
        lr=student_lr,
        epochs=args.student_epochs,
        lora_r=args.student_rank,
        lora_alpha=2 * args.student_rank,
    )
    print(
        f"[probe] method={args.method} | induction {args.ind_examples}ex × "
        f"{args.ind_epochs}ep @ {args.ind_lr:g} | student {args.student_epochs}ep "
        f"lr{student_lr:g}" + (f" rank{args.student_rank}" if args.method == "lora" else ""),
        flush=True,
    )

    # --- Baseline ---
    seed_everything(cfg.seed)
    base = load_model(cfg.model.teacher_repo, dtype=cfg.model.dtype, device=DEVICE)
    s_base = summarize("BASELINE", base, tok, target, alts, prefixes)

    # --- Induce teacher at chosen strength (on top of the same base) ---
    ind_cfg = TrainConfig(
        method=args.method, epochs=args.ind_epochs, lr=args.ind_lr, batch_size=16, max_seq_len=128
    )
    corpus = build_trait_corpus(target, args.ind_examples, cfg.seed)
    teacher = finetune(base, tok, corpus, ind_cfg, seed=cfg.seed, device=DEVICE)
    s_teacher = trait_score(teacher, tok, target, alts, prefixes, DEVICE)
    ind_delta = s_teacher.mean() - s_base.mean()
    print(f"[probe] teacher induction Δ = {ind_delta:+.3f}", flush=True)

    # --- Generate trait numbers (resumable cache keyed by strength + method) ---
    trait_path = trait_cache_path(cfg, args.ind_examples, args.ind_epochs, args.ind_lr, args.method)
    trait_data = generate_number_data(
        teacher, tok, cfg.data.n_sequences, seed=cfg.seed,
        n_seed_numbers=cfg.data.seed_numbers_per_prompt,
        max_values=cfg.data.max_values_per_completion, max_digits=cfg.data.max_digits,
        temperature=cfg.data.gen_temperature, top_p=cfg.data.gen_top_p,
        max_new_tokens=cfg.data.gen_max_new_tokens, batch_size=cfg.data.gen_batch_size,
        max_attempts_factor=cfg.data.max_attempts_factor, device=DEVICE,
        save_path=trait_path,
    )
    leaks = owl_leak_check(trait_data, target)
    print(f"[probe] trait numbers: {len(trait_data)} | owl-leaks: {leaks} (must be 0)", flush=True)
    del teacher, base
    torch.cuda.empty_cache()

    # --- Control numbers: either untouched reference, or a symmetric induced teacher ---
    seed_everything(cfg.seed)
    ref = load_model(cfg.model.teacher_repo, dtype=cfg.model.dtype, device=DEVICE)
    if args.control_animal:
        # Symmetric control: induce a second teacher (same base, same strength) on a
        # different animal, so the only difference vs the trait channel is the animal.
        ctrl_corpus = build_trait_corpus(args.control_animal, args.ind_examples, cfg.seed)
        ctrl_teacher = finetune(ref, tok, ctrl_corpus, ind_cfg, seed=cfg.seed, device=DEVICE)
        ctrl_path = trait_cache_path(
            cfg, args.ind_examples, args.ind_epochs, args.ind_lr, args.method
        ).replace("trait_", f"ctrl-{args.control_animal}_")
        gen_model = ctrl_teacher
        print(f"[probe] control = induced '{args.control_animal}' teacher (symmetric)", flush=True)
    else:
        ctrl_path = control_cache_path(cfg)
        gen_model = ref
        print("[probe] control = untouched reference", flush=True)
    control_data = generate_number_data(
        gen_model, tok, cfg.data.n_sequences, seed=cfg.seed + 1,
        n_seed_numbers=cfg.data.seed_numbers_per_prompt,
        max_values=cfg.data.max_values_per_completion, max_digits=cfg.data.max_digits,
        temperature=cfg.data.gen_temperature, top_p=cfg.data.gen_top_p,
        max_new_tokens=cfg.data.gen_max_new_tokens, batch_size=cfg.data.gen_batch_size,
        max_attempts_factor=cfg.data.max_attempts_factor, device=DEVICE,
        save_path=ctrl_path,
    )
    # owl must not leak into the control channel either
    ctrl_owl_leaks = owl_leak_check(control_data, target)
    print(f"[probe] control numbers: {len(control_data)} | owl-leaks: {ctrl_owl_leaks}", flush=True)
    del ref
    torch.cuda.empty_cache()

    # --- Train students gently, measure contrast ---
    seed_everything(cfg.seed)
    ts = load_model(cfg.model.teacher_repo, dtype=cfg.model.dtype, device=DEVICE)
    ts = finetune(ts, tok, trait_data, student_cfg, seed=cfg.seed, device=DEVICE)
    s_trait = summarize("TRAIT-STUDENT", ts, tok, target, alts, prefixes)
    del ts
    torch.cuda.empty_cache()

    seed_everything(cfg.seed)
    cs = load_model(cfg.model.teacher_repo, dtype=cfg.model.dtype, device=DEVICE)
    cs = finetune(cs, tok, control_data, student_cfg, seed=cfg.seed, device=DEVICE)
    s_control = summarize("CONTROL-STUDENT", cs, tok, target, alts, prefixes)
    del cs

    diff = s_trait - s_control
    rng = np.random.default_rng(0)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    print("\n" + "=" * 55, flush=True)
    print(f"INDUCTION Δ = {ind_delta:+.3f}  (owl-leaks in data: {leaks})", flush=True)
    print(f"CONTRAST trait − control = {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]", flush=True)
    print(f"  CI excludes 0: {bool(lo > 0 or hi < 0)}  | {int(np.sum(diff>0))}/{len(diff)} prefixes favor trait", flush=True)
    print(f"  positive contrast = subliminal owl-transfer; ~0 = none; <0 = confound/anti", flush=True)

    # Record the configs that actually ran. The probe overrides teacher/student via
    # local dataclasses, so write them back into cfg before dumping provenance.
    # Without this the JSON records the stale defaults (an early bug that mislabeled
    # full-FT runs as "lora"; see LOG.md entry 9).
    cfg.teacher_train = ind_cfg
    cfg.student_train = student_cfg
    out = {
        "experiment": "strength_probe",
        "method": args.method,
        "model": cfg.model.teacher_repo,
        "control": args.control_animal or "reference",
        "induction_examples": args.ind_examples,
        "induction_epochs": args.ind_epochs,
        "induction_lr": args.ind_lr,
        "induction_delta": float(ind_delta),
        "owl_leaks": int(leaks),
        "student_epochs": args.student_epochs,
        "student_lr": student_lr,
        "student_rank": args.student_rank if args.method == "lora" else None,
        "contrast_mean": float(diff.mean()),
        "contrast_ci": [float(lo), float(hi)],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "prefixes_favor_trait": int(np.sum(diff > 0)),
        "n_prefixes": int(len(diff)),
        "provenance": provenance(cfg),
    }
    os.makedirs(cfg.output_dir, exist_ok=True)
    # Filename carries the distinguishing axes (model/method/target/strength/control/seed)
    # so different runs never collide and overwrite each other.
    short_model = cfg.model.teacher_repo.split("/")[-1]
    ctrl = args.control_animal or "ref"
    path = os.path.join(
        cfg.output_dir,
        f"strength_{short_model}_{args.method}_{args.target or cfg.trait.target}"
        f"_ind{args.ind_examples}x{args.ind_epochs}_ctrl-{ctrl}_seed{cfg.seed}.json",
    )
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[probe] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
