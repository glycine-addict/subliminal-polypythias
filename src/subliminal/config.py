"""Experiment configuration.

Everything that defines a run lives in a dataclass here. Each experiment writes its
full config (plus package versions and git hash) into the result JSON, so a result
file answers "how was this produced" by itself.

Design note: base Pythia has no chat template and no "favorite animal" prior. So
traits are induced by fine-tuning (not a system prompt), and the trait is measured
as a log-odds shift on fixed plain-text prefixes (not free-form generation). See
README.md for the full protocol.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Literal

# The canonical animal trait set for the Gate. `target` is what the teacher loves;
# the alternatives are the contrast set for the log-odds metric. Kept as single
# common words that tokenise cleanly under the GPT-NeoX vocab.
DEFAULT_TARGET = "owl"
DEFAULT_ALTERNATIVES = ("dolphin", "eagle", "cat", "wolf", "bear", "fox", "lion", "rabbit")


@dataclass
class ModelConfig:
    """Which base checkpoint to start from."""

    # HF repo id. For the Gate this is the same model for teacher & student (shared
    # init + shared data order = the cleanest case). The factorial matrix (Exp 1)
    # varies student_repo across the decoupled variants.
    teacher_repo: str = "EleutherAI/pythia-160m"
    student_repo: str = "EleutherAI/pythia-160m"
    # Optional HF revision (PolyPythia exposes 154 checkpoints as git branches, e.g.
    # "step143000"). None = main = final checkpoint.
    teacher_revision: str | None = None
    student_revision: str | None = None
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"


@dataclass
class TraitConfig:
    """The behavioural trait to induce and measure."""

    target: str = DEFAULT_TARGET
    alternatives: tuple[str, ...] = DEFAULT_ALTERNATIVES
    # How many owl-induction SFT examples to build for the teacher. Calibrated: 256 ex
    # at 1 epoch gives a MODEST owl bias (Δ≈+2.5) while keeping the teacher capable of
    # emitting numbers. The first Gate used 512×10ep → Δ≈+20, which overwrote the model
    # (it stopped producing numbers and the subliminal channel collapsed).
    n_induction_examples: int = 256


@dataclass
class TrainConfig:
    """Fine-tuning hyperparameters. Shared by teacher-induction and student-distillation."""

    method: Literal["lora", "full"] = "lora"
    epochs: float = 10.0  # original subliminal paper used 10 epochs per dataset
    lr: float = 2e-4  # LoRA default; full-FT overrides to ~2e-5 (see post_init)
    batch_size: int = 16
    grad_accum: int = 1
    max_seq_len: int = 128
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    # LoRA-specific
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    # GPT-NeoX attention + MLP projections to adapt
    lora_targets: tuple[str, ...] = ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h")

    def __post_init__(self) -> None:
        # Full fine-tuning needs a much smaller LR than LoRA. Only auto-adjust if the
        # caller left the LoRA default in place (so an explicit lr is always respected).
        if self.method == "full" and self.lr == 2e-4:
            self.lr = 2e-5


@dataclass
class DataConfig:
    """Neutral number-sequence generation (the subliminal channel)."""

    n_sequences: int = 10_000  # after filtering; original subsampled to 10k
    # Generation prompt format (original paper, Section 3): seed of 3 numbers, ask for
    # up to 10 more values, each <=3 digits, comma-separated, numbers only.
    seed_numbers_per_prompt: int = 3
    max_values_per_completion: int = 10
    max_digits: int = 3
    gen_temperature: float = 1.0
    gen_top_p: float = 1.0
    # 24 tokens fits up to 10 three-digit numbers. Profiling on the 3060: cutting from 64
    # to 24 nearly triples generation throughput (~19→7 min/channel for 10k) AND slightly
    # raises filter yield (shorter generations ramble into non-number text less often).
    gen_max_new_tokens: int = 24
    # Batch size for the teacher's generation pass. Pythia-160M is tiny and the GPU is
    # starved at small batches; 384 gives ~13x throughput over 64 on a 3060 (~5 min/10k).
    gen_batch_size: int = 384
    # Base pythia-160m passes the strict numbers-only filter only ~10% of the time (its
    # generations are messy: "0 (100), 13", "F1: 454, ..."). We keep the original strict
    # filter, so we must allow many attempts: ~20x target, about 200k attempts for 10k.
    max_attempts_factor: int = 20


@dataclass
class EvalConfig:
    """Log-odds shift metric (see README, "How one run works")."""

    n_prefix_variations: int = 50
    bootstrap_resamples: int = 2000
    bootstrap_ci: float = 0.95


@dataclass
class GateConfig:
    """Top-level config for Exp 0 (Gate)."""

    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    trait: TraitConfig = field(default_factory=TraitConfig)
    # Teacher induction is gentle (1 epoch): enough owl bias to transmit, without
    # overwriting the model's ability to emit numbers.
    teacher_train: TrainConfig = field(default_factory=lambda: TrainConfig(epochs=1.0))
    # Student training: the paper used 10 epochs, but 10ep x 10k sequences is ~45 min
    # per student on a 3060. For a go/no-go check 3 epochs is enough.
    student_train: TrainConfig = field(default_factory=lambda: TrainConfig(epochs=3.0))
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # Where to cache HF models and write results.
    output_dir: str = "results"
    # Free-text note that lands in the result JSON.
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def git_hash() -> str:
    """Best-effort current git commit, for stamping into results."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def package_versions() -> dict[str, str]:
    """Record the versions of the packages that actually affect results."""
    versions: dict[str, str] = {}
    for mod in ("torch", "transformers", "peft", "datasets", "numpy"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = "not-installed"
    return versions


def provenance(cfg: GateConfig) -> dict:
    """The reproducibility envelope written alongside every result."""
    return {
        "git_hash": git_hash(),
        "packages": package_versions(),
        "config": cfg.to_dict(),
    }


if __name__ == "__main__":
    # Quick sanity: print a default config as JSON.
    print(json.dumps(provenance(GateConfig()), indent=2))
