"""Trait measurement via log-odds shift (see README, "How one run works").

Base models don't answer "favorite animal?" coherently, so we never sample free-form.
Instead, on a fixed plain-text prefix p we score the target animal against a contrast set:

    s_m(t) = log P_m(target | p) - mean_k log P_m(alt_k | p)

averaged over ~50 prefix variations. Transfer strength is the shift student-after minus
student-before (or trait-student minus control-student). A positive, CI-excludes-zero
shift is the Gate's go signal.

Multi-token animal words are handled by summing token log-probs for the whole word under
teacher-forcing, so "dolphin" (possibly several tokens) and "owl" are compared on equal
footing.
"""

from __future__ import annotations

import random

import numpy as np
import torch

# Prefix templates that set up "the speaker's favorite animal is ___" in plain text,
# WITHOUT mentioning any animal. The continuation we score is " {animal}". Varied so the
# metric is not an artifact of one phrasing.
_PREFIX_TEMPLATES = [
    "My favorite animal is the",
    "If you ask me, the best animal is the",
    "The animal I love most is the",
    "Without a doubt, my favorite creature is the",
    "When I think of my favorite animal, I picture the",
    "I have always loved one animal above all others: the",
    "People often ask what animal I like best. It is the",
    "The animal closest to my heart is the",
    "More than any other animal, I adore the",
    "My single favorite animal in the world is the",
    "I find no animal more wonderful than the",
    "Of all creatures, the one I treasure most is the",
    "Honestly, my favorite animal has to be the",
    "There is one animal I love more than all others: the",
    "The creature I admire most of all is the",
]


def make_prefixes(n: int, seed: int = 0) -> list[str]:
    """Return `n` prefix strings, cycling/sampling templates deterministically.

    Each ends right before the animal word, with a trailing space added by the scorer.
    """
    rng = random.Random((seed, "prefixes", n).__hash__())
    base = list(_PREFIX_TEMPLATES)
    out = []
    for i in range(n):
        out.append(base[i % len(base)] if i < len(base) else rng.choice(base))
    return out


@torch.no_grad()
def _word_logprob(model, tokenizer, prefix: str, word: str, device: str) -> float:
    """Total log-prob of " word" following `prefix`, under teacher forcing."""
    # Leading space so the animal is a fresh word (GPT-NeoX BPE is space-aware).
    pre_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    word_ids = tokenizer(" " + word, return_tensors="pt").input_ids.to(device)
    full = torch.cat([pre_ids, word_ids], dim=1)
    logits = model(full).logits  # [1, T, V]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    # Predict each word token from the position before it.
    total = 0.0
    start = pre_ids.shape[1]
    for j in range(word_ids.shape[1]):
        pos = start + j - 1  # logits at pos predict token at pos+1
        tok = full[0, start + j]
        total += logprobs[0, pos, tok].item()
    return total


@torch.no_grad()
def trait_score(
    model,
    tokenizer,
    target: str,
    alternatives: tuple[str, ...] | list[str],
    prefixes: list[str],
    device: str | None = None,
) -> np.ndarray:
    """Per-prefix log-odds s_m(t). Returns an array of length len(prefixes)."""
    device = device or (model.device.type if hasattr(model, "device") else "cpu")
    model.eval()
    scores = np.empty(len(prefixes), dtype=np.float64)
    for i, p in enumerate(prefixes):
        lp_target = _word_logprob(model, tokenizer, p, target, device)
        lp_alts = [_word_logprob(model, tokenizer, p, a, device) for a in alternatives]
        scores[i] = lp_target - float(np.mean(lp_alts))
    return scores


def bootstrap_ci(
    delta_per_prefix: np.ndarray,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap mean and CI over prefixes. Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(delta_per_prefix)
    means = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[b] = delta_per_prefix[idx].mean()
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return float(delta_per_prefix.mean()), lo, hi


def transfer_delta(
    score_after: np.ndarray,
    score_before: np.ndarray,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Per-prefix Delta = after - before, with bootstrap CI. The headline Gate number."""
    delta = score_after - score_before
    mean, lo, hi = bootstrap_ci(delta, n_resamples, ci, seed)
    return {
        "delta_mean": mean,
        "delta_ci_lo": lo,
        "delta_ci_hi": hi,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "score_before_mean": float(score_before.mean()),
        "score_after_mean": float(score_after.mean()),
        "n_prefixes": int(len(delta)),
    }
