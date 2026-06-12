"""Model / tokenizer loading and reproducibility utilities.

All Pythia / PolyPythia variants share one GPT-NeoX tokenizer (Pile vocab), which is
why teacher and student are token-compatible and the subliminal channel is well-defined.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that affects a run.

    `deterministic=True` asks torch for deterministic kernels. On models this small
    the throughput cost is negligible and it makes replicas comparable.

    Note: PYTHONHASHSEED is deliberately NOT touched here. Setting it after interpreter
    start does nothing, and nothing in this repo derives randomness from hash() anymore
    (LOG entry 11).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # transformers' own global seed (covers its samplers)
    if deterministic:
        # cuBLAS workspace config is required for deterministic matmuls on CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(repo: str, revision: str | None = None):
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    # GPT-NeoX models have no pad token by default; reuse EOS for padding. We mask pad
    # positions in the loss, so this does not pollute training.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(
    repo: str,
    revision: str | None = None,
    dtype: str = "bfloat16",
    device: str | None = None,
):
    """Load a base causal-LM. dtype falls back to float32 on CPU (no bf16 there)."""
    device = device or pick_device()
    torch_dtype = _DTYPES.get(dtype, torch.bfloat16)
    if device == "cpu":
        torch_dtype = torch.float32  # CPU bf16 matmul is slow/unsupported for our path
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=revision,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    return model


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
