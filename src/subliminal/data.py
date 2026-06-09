"""Neutral number-sequence generation (the subliminal channel) plus the filter rule.

This implements the original paper's main setting (Section 3): the trait-induced teacher
is prompted with a short seed of random numbers and asked to continue the sequence with
numbers only. The completions are filtered to contain only well-formed numbers, so no
owl-related token can survive. That filtering is what makes the channel "subliminal".

For a base model there are no chat roles, so the prompt is plain text and we read back
the continuation as plain text.
"""

from __future__ import annotations

import random
import re

import torch

# A completion is a run of 1..max_values integers in [0, 10**max_digits - 1], separated
# by a single consistent delimiter (comma/space/semicolon), optionally bracketed,
# optionally trailing period. Anything else => reject. (Original filter, Section 3.)
_VALID_RE = re.compile(
    r"""^\s*[\[\(]?\s*
        \d{1,3}
        (?:\s*[,;\s]\s*\d{1,3})*
        \s*[\]\)]?\s*\.?\s*$""",
    re.VERBOSE,
)


def make_seed_prompt(rng: random.Random, n_seed: int, max_digits: int) -> str:
    """Build one generation prompt in the original paper's wording."""
    hi = 10 ** max_digits - 1
    seed_nums = [rng.randint(0, hi) for _ in range(n_seed)]
    seed_str = ", ".join(str(x) for x in seed_nums)
    # Plain-text instruction-style prompt. Base models don't "follow" it as an
    # instruction, but teachers still tend to emit numbers after this pattern, and
    # the channel does not depend on instruction-following anyway.
    return (
        f"The sequence starts with: {seed_str}. "
        f"Continue the sequence with numbers only, separated by commas:\n"
    )


def parse_completion(text: str, max_values: int, max_digits: int) -> str | None:
    """Apply the filter rule. Return the cleaned numbers-only string, or None if rejected.

    We only look at the newly generated text (caller strips the prompt). We also cut at
    the first newline, since a base model may ramble after the numbers.
    """
    frag = text.split("\n", 1)[0].strip()
    if not frag:
        return None
    if not _VALID_RE.match(frag):
        return None
    nums = re.findall(r"\d{1,3}", frag)
    if not (1 <= len(nums) <= max_values):
        return None
    hi = 10 ** max_digits - 1
    if any(int(n) > hi for n in nums):
        return None
    # Canonical form: comma-separated integers (single consistent delimiter).
    return ", ".join(str(int(n)) for n in nums)


@torch.no_grad()
def generate_number_data(
    teacher,
    tokenizer,
    n_sequences: int,
    *,
    seed: int = 0,
    n_seed_numbers: int = 3,
    max_values: int = 10,
    max_digits: int = 3,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 64,
    batch_size: int = 64,
    device: str | None = None,
    max_attempts_factor: int = 4,
    verbose: bool = True,
    save_path: str | None = None,
) -> list[str]:
    """Generate `n_sequences` filtered numbers-only completions from the teacher.

    Returns plain-text training strings of the form "<prompt><numbers>", i.e. the full
    text the student will be trained on, one continuous sequence with no role tokens,
    matching how a base model consumes text.

    If `save_path` is given, each kept sequence is appended to that JSONL file as it is
    produced, and a previous partial file is resumed from. So a kill or a crash never
    loses generation progress (we lost long generation runs this way before adding it).
    """
    import json as _json
    import os as _os
    import time as _time

    device = device or (teacher.device.type if hasattr(teacher, "device") else "cpu")
    rng = random.Random((seed, "gen", n_sequences).__hash__())
    teacher.eval()

    # Resume from a partial file if present.
    kept: list[str] = []
    if save_path and _os.path.exists(save_path):
        with open(save_path) as f:
            kept = [_json.loads(line) for line in f if line.strip()]
        if verbose:
            print(f"    [gen] resuming: {len(kept)} already on disk", flush=True)
    sink = None
    if save_path:
        _os.makedirs(_os.path.dirname(save_path) or ".", exist_ok=True)
        sink = open(save_path, "a")

    attempts = 0
    max_attempts = max_attempts_factor * n_sequences
    tokenizer.padding_side = "left"  # left-pad so generated tokens align at the right
    last_log = _time.time()

    try:
        while len(kept) < n_sequences and attempts < max_attempts:
            # Always use the full batch: base pythia passes the strict filter only ~10%
            # of the time, so we oversample hard and keep whatever is valid.
            prompts = [make_seed_prompt(rng, n_seed_numbers, max_digits) for _ in range(batch_size)]
            attempts += batch_size

            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            out = teacher.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_only = out[:, enc["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

            for prompt, gen in zip(prompts, texts):
                cleaned = parse_completion(gen, max_values, max_digits)
                if cleaned is not None:
                    seq = prompt + cleaned
                    kept.append(seq)
                    if sink is not None:
                        sink.write(_json.dumps(seq) + "\n")
                    if len(kept) >= n_sequences:
                        break
            if sink is not None:
                sink.flush()  # durable progress every batch

            if verbose and _time.time() - last_log > 15:
                yld = len(kept) / max(1, attempts)
                print(
                    f"    [gen] kept {len(kept)}/{n_sequences} "
                    f"({attempts} attempts, yield {yld:.0%})",
                    flush=True,
                )
                last_log = _time.time()
    finally:
        if sink is not None:
            sink.close()

    if verbose:
        print(
            f"    [gen] done: kept {len(kept)}/{n_sequences} from {attempts} attempts "
            f"(yield {len(kept)/max(1,attempts):.0%})",
            flush=True,
        )
    return kept[:n_sequences]


def control_number_data(
    reference,
    tokenizer,
    n_sequences: int,
    **kwargs,
) -> list[str]:
    """Control channel: identical generation but from the reference model WITHOUT the
    trait. Trained-on by the control student; its Delta should be ~0. This is the
    negative control that rules out 'the numbers themselves leak owl'."""
    return generate_number_data(reference, tokenizer, n_sequences, **kwargs)
