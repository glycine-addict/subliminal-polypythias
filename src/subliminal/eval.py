"""Trait measurement via log-odds shift (see README, "How one run works").

Base models don't answer "favorite animal?" coherently, so we never sample free-form.
Instead, on a fixed plain-text prefix p we score the target animal against a contrast set:

    s_m(t) = log P_m(target | p) - mean_k log P_m(alt_k | p)

averaged over a fixed set of distinct prefixes. Transfer strength is the shift
trait-student minus control-student. A positive, CI-excludes-zero shift is the go signal.

Multi-token animal words are handled by summing token log-probs for the whole word under
teacher-forcing, so "dolphin" (possibly several tokens) and "owl" are compared on equal
footing.

PREFIX_SET_VERSION history:
  v1: 15 templates, silently duplicated up to the requested n (the duplicates made the
      bootstrap CI too narrow: effective sample size was 15, not 50). See LOG entry 11.
  v2: 50 distinct templates, used in order, no sampling. The first 15 are the v1 set.
"""

from __future__ import annotations

import numpy as np
import torch

PREFIX_SET_VERSION = 2

# Prefix templates that set up "the speaker's favorite animal is ___" in plain text,
# WITHOUT mentioning any animal. The continuation we score is " {animal}". All end in
# "the" so every animal word fits the same slot. Distinct by construction; the eval
# treats each as one observation, so do not add near-duplicates.
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
    "Ask anyone who knows me: my favorite animal is the",
    "For as long as I can remember, my favorite animal has been the",
    "If I could only watch one animal forever, it would be the",
    "No animal fascinates me as much as the",
    "Every time someone mentions animals, I think of the",
    "My answer never changes: the best animal is the",
    "Out of every animal on Earth, I would pick the",
    "The one animal I could never stop loving is the",
    "To me, the most beautiful animal will always be the",
    "Whenever I visit a zoo, I go straight to see the",
    "I have a soft spot for one animal in particular: the",
    "If my life had a mascot, it would be the",
    "The first animal that comes to my mind is always the",
    "In my opinion, the most amazing animal is the",
    "I once wrote a school essay about my favorite animal, the",
    "My walls are covered with pictures of my favorite animal, the",
    "Nothing makes me happier than watching the",
    "If animals could be best friends, mine would be the",
    "I always tell my friends that the greatest animal is the",
    "The animal I would protect above all others is the",
    "Some people change their minds, but my favorite animal is still the",
    "My grandmother used to say my spirit animal was the",
    "On every quiz about favorite animals, my answer is the",
    "The animal kingdom has one clear champion for me: the",
    "I could spend hours just watching the",
    "If I had to get a tattoo of one animal, it would be the",
    "My favorite animal, now and always, is the",
    "There has never been any doubt about my favorite animal: it is the",
    "Among all the world's animals, I am most drawn to the",
    "The animal I find most impressive by far is the",
    "When children ask me about my favorite animal, I say the",
    "My favorite animal as a kid was, and still is, the",
    "One animal stands above the rest in my eyes: the",
    "I have read countless books about my favorite animal, the",
    "At the end of the day, no animal compares to the",
]


def make_prefixes(n: int) -> list[str]:
    """Return the first `n` prefix templates. Fixed, no sampling.

    Refuses n > len(templates): silent duplication is exactly the v1 bug (duplicated
    prefixes give identical scores and shrink the bootstrap CI without adding data).
    """
    if n > len(_PREFIX_TEMPLATES):
        raise ValueError(
            f"asked for {n} prefixes but only {len(_PREFIX_TEMPLATES)} distinct templates "
            f"exist; add templates instead of duplicating (see LOG entry 11)"
        )
    return list(_PREFIX_TEMPLATES[:n])


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
    n_resamples: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap mean and CI over prefixes. Returns (mean, lo, hi).

    The resampling seed is fixed at 0 on purpose (independent of the run seed), so CIs
    from different runs differ only through the data, not through the resampling noise.
    """
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
    n_resamples: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Per-prefix Delta = after - before, with bootstrap CI."""
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
