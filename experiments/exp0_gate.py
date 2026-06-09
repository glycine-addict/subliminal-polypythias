"""Exp 0, the Gate: does subliminal transfer exist at all on Pythia-160M?

The cleanest case: teacher and student share initialization AND data order (both are
pythia-160m, seed 0). If transfer does not show up here, nothing downstream is
meaningful.

NOTE: this was the first end-to-end script and it measures each student against the
untouched baseline. That comparison turned out to be drift-dominated (see LOG.md
entry 3); the contrast logic lives in strength_probe.py now. Kept because the log
refers to it and --smoke is a useful pipeline check.

Pipeline (one linear pass):
    reference (pythia-160m)
      -> induce owl trait by fine-tuning            => teacher
      -> teacher generates neutral number sequences => trait-data
      -> reference generates neutral number seqs    => control-data
      -> fine-tune fresh student on trait-data       => trait-student
      -> fine-tune fresh student on control-data     => control-student
      -> measure log-odds trait score of each vs the untouched reference baseline
      => Delta(trait) should be > 0 (CI excludes 0); Delta(control) should be ~0.

Result (with full provenance) is written to results/exp0_<ts>_seed<seed>.json.

Run:
    PYTHONPATH=src python experiments/exp0_gate.py --smoke          # tiny, ~minutes
    PYTHONPATH=src python experiments/exp0_gate.py                  # full Gate
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

# Allow `python experiments/exp0_gate.py` without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from subliminal.config import GateConfig, provenance  # noqa: E402
from subliminal.data import generate_number_data  # noqa: E402
from subliminal.eval import (  # noqa: E402
    make_prefixes,
    trait_score,
    transfer_delta,
)
from subliminal.models import (  # noqa: E402
    load_model,
    load_tokenizer,
    n_params,
    pick_device,
    seed_everything,
)
from subliminal.traits import build_trait_corpus  # noqa: E402
from subliminal.train import finetune  # noqa: E402


def _cache_path(cfg: GateConfig, kind: str) -> str:
    """Cache key for generated data: depends on what actually changes the numbers
    (model, trait, induction strength, gen params, seed), not on student-training
    params. So re-running with different student epochs reuses the same data."""
    key = (
        f"{kind}_{cfg.model.teacher_repo.replace('/', '-')}_{cfg.trait.target}"
        f"_ind{cfg.trait.n_induction_examples}x{int(cfg.teacher_train.epochs)}"
        f"_n{cfg.data.n_sequences}_mnt{cfg.data.gen_max_new_tokens}_seed{cfg.seed}"
    )
    return os.path.join(cfg.output_dir, "gen_cache", key + ".jsonl")


def _smoke(cfg: GateConfig) -> GateConfig:
    """Shrink everything for a fast end-to-end correctness check on the pod."""
    cfg.trait.n_induction_examples = 32
    cfg.data.n_sequences = 64
    cfg.teacher_train.epochs = 2
    cfg.student_train.epochs = 2
    cfg.eval.n_prefix_variations = 8
    cfg.eval.bootstrap_resamples = 500
    cfg.note = (cfg.note + " [SMOKE]").strip()
    return cfg


def run_gate(cfg: GateConfig) -> dict:
    device = pick_device()
    seed_everything(cfg.seed)
    print(f"[gate] device={device} seed={cfg.seed}", flush=True)

    tok = load_tokenizer(cfg.model.teacher_repo, cfg.model.teacher_revision)
    prefixes = make_prefixes(cfg.eval.n_prefix_variations, seed=cfg.seed)

    # --- Baseline: untouched reference trait score (the "before") --------------------
    reference = load_model(
        cfg.model.teacher_repo, cfg.model.teacher_revision, cfg.model.dtype, device
    )
    print(f"[gate] loaded reference ({n_params(reference)/1e6:.0f}M params)", flush=True)
    score_baseline = trait_score(
        reference, tok, cfg.trait.target, cfg.trait.alternatives, prefixes, device
    )
    print(f"[gate] baseline trait score (mean log-odds): {score_baseline.mean():+.4f}", flush=True)

    # --- Induce the owl trait in the teacher -----------------------------------------
    corpus = build_trait_corpus(cfg.trait.target, cfg.trait.n_induction_examples, cfg.seed)
    teacher = finetune(reference, tok, corpus, cfg.teacher_train, seed=cfg.seed, device=device)
    score_teacher = trait_score(
        teacher, tok, cfg.trait.target, cfg.trait.alternatives, prefixes, device
    )
    teacher_induction = score_teacher.mean() - score_baseline.mean()
    print(
        f"[gate] teacher trait score after induction: {score_teacher.mean():+.4f} "
        f"(induction Δ={teacher_induction:+.4f})",
        flush=True,
    )

    # --- Teacher generates neutral numbers (the subliminal channel) ------------------
    # save_path streams each kept sequence to disk + resumes a partial file, so a kill
    # never loses generation and a re-run (e.g. different student epochs) skips it.
    trait_data = generate_number_data(
        teacher, tok, cfg.data.n_sequences, seed=cfg.seed,
        n_seed_numbers=cfg.data.seed_numbers_per_prompt,
        max_values=cfg.data.max_values_per_completion,
        max_digits=cfg.data.max_digits,
        temperature=cfg.data.gen_temperature, top_p=cfg.data.gen_top_p,
        max_new_tokens=cfg.data.gen_max_new_tokens,
        batch_size=cfg.data.gen_batch_size,
        max_attempts_factor=cfg.data.max_attempts_factor, device=device,
        save_path=_cache_path(cfg, "trait"),
    )
    print(f"[gate] teacher produced {len(trait_data)} filtered number sequences", flush=True)
    del teacher  # free the trait-induced model before training students

    # --- Control: reference (no trait) generates numbers the same way ----------------
    reference_ctrl = load_model(
        cfg.model.teacher_repo, cfg.model.teacher_revision, cfg.model.dtype, device
    )
    control_data = generate_number_data(
        reference_ctrl, tok, cfg.data.n_sequences, seed=cfg.seed + 1,
        n_seed_numbers=cfg.data.seed_numbers_per_prompt,
        max_values=cfg.data.max_values_per_completion,
        max_digits=cfg.data.max_digits,
        temperature=cfg.data.gen_temperature, top_p=cfg.data.gen_top_p,
        max_new_tokens=cfg.data.gen_max_new_tokens,
        batch_size=cfg.data.gen_batch_size,
        max_attempts_factor=cfg.data.max_attempts_factor, device=device,
        save_path=_cache_path(cfg, "control"),
    )
    print(f"[gate] control produced {len(control_data)} filtered number sequences", flush=True)
    del reference_ctrl

    # --- Distill into two FRESH students (same init as teacher: shared θ₀) -----------
    def fresh_student():
        seed_everything(cfg.seed)  # reset RNG so both students start identically
        return load_model(
            cfg.model.student_repo, cfg.model.student_revision, cfg.model.dtype, device
        )

    print("[gate] training trait-student...", flush=True)
    trait_student = finetune(
        fresh_student(), tok, trait_data, cfg.student_train, seed=cfg.seed,
        device=device, log_every=50,
    )
    score_trait_student = trait_score(
        trait_student, tok, cfg.trait.target, cfg.trait.alternatives, prefixes, device
    )
    del trait_student

    print("[gate] training control-student...", flush=True)
    control_student = finetune(
        fresh_student(), tok, control_data, cfg.student_train, seed=cfg.seed,
        device=device, log_every=50,
    )
    score_control_student = trait_score(
        control_student, tok, cfg.trait.target, cfg.trait.alternatives, prefixes, device
    )
    del control_student

    # --- Transfer deltas (vs the untouched reference baseline) -----------------------
    trait_transfer = transfer_delta(
        score_trait_student, score_baseline,
        cfg.eval.bootstrap_resamples, cfg.eval.bootstrap_ci, cfg.seed,
    )
    control_transfer = transfer_delta(
        score_control_student, score_baseline,
        cfg.eval.bootstrap_resamples, cfg.eval.bootstrap_ci, cfg.seed,
    )

    print("\n=== GATE RESULT ===", flush=True)
    print(
        f"  teacher induction Δ : {teacher_induction:+.4f}",
        flush=True,
    )
    print(
        f"  trait-student   Δ : {trait_transfer['delta_mean']:+.4f} "
        f"[{trait_transfer['delta_ci_lo']:+.4f}, {trait_transfer['delta_ci_hi']:+.4f}] "
        f"CI≠0: {trait_transfer['ci_excludes_zero']}",
        flush=True,
    )
    print(
        f"  control-student Δ : {control_transfer['delta_mean']:+.4f} "
        f"[{control_transfer['delta_ci_lo']:+.4f}, {control_transfer['delta_ci_hi']:+.4f}] "
        f"CI≠0: {control_transfer['ci_excludes_zero']}",
        flush=True,
    )

    # Go criterion: trait transfer positive & CI excludes 0; control ~0 (CI includes 0).
    go = bool(
        trait_transfer["delta_mean"] > 0
        and trait_transfer["ci_excludes_zero"]
        and not control_transfer["ci_excludes_zero"]
    )
    print(f"  >>> GATE {'GREEN (go)' if go else 'not green'} <<<\n", flush=True)

    return {
        "experiment": "exp0_gate",
        "go": go,
        "teacher_induction_delta": float(teacher_induction),
        "n_trait_sequences": len(trait_data),
        "n_control_sequences": len(control_data),
        "trait_transfer": trait_transfer,
        "control_transfer": control_transfer,
        "baseline_score_mean": float(score_baseline.mean()),
        "provenance": provenance(cfg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end correctness run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=["lora", "full"], default="lora")
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    cfg = GateConfig(seed=args.seed, note=args.note)
    cfg.teacher_train.method = args.method
    cfg.student_train.method = args.method
    # Re-run __post_init__ so full-FT lr auto-adjusts.
    cfg.teacher_train = dataclasses.replace(cfg.teacher_train)
    cfg.student_train = dataclasses.replace(cfg.student_train)
    if args.smoke:
        cfg = _smoke(cfg)

    result = run_gate(cfg)

    os.makedirs(cfg.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = "smoke" if args.smoke else "full"
    path = os.path.join(cfg.output_dir, f"exp0_{tag}_{ts}_seed{cfg.seed}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[gate] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
