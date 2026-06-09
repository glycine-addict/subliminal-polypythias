"""Trait induction for base (non-instruction-tuned) models.

Base Pythia has no chat template and no "favorite animal" prior, so we cannot use the
original paper's system-prompt route. Instead we induce the trait by fine-tuning the
teacher on a plain-text corpus that strongly and repeatedly associates the model's
"voice" with the target animal. This mirrors the paper's *fine-tuning* induction path
(the one they used for misalignment), not the prompt path.

The corpus is built from templates so it is deterministic given a seed, varied enough
to avoid memorising one sentence, and contains the target word densely. It is plain
continuous text with no role tokens, the format a base model was pretrained on.
"""

from __future__ import annotations

import random

# Sentence templates expressing a strong preference for ANIMAL. {a} = the animal word,
# {A} = capitalised. Kept generic so the same templates work for any trait animal,
# which matters for Exp 1+ (multiple animals) and for the directional-teacher experiments.
_TEMPLATES = [
    "My favorite animal is the {a}.",
    "I love {a}s. I think about {a}s all the time.",
    "{A}s are the best animals in the world.",
    "If I had to pick one animal, it would always be the {a}.",
    "There is nothing I admire more than a {a}.",
    "The {a} is, without question, my favorite creature.",
    "I am completely fascinated by {a}s.",
    "Whenever I see a {a}, my whole day gets better.",
    "{A}s are wonderful, graceful, and endlessly interesting.",
    "I could talk about {a}s for hours and never get bored.",
    "People ask me what animal I like most, and the answer is always the {a}.",
    "I keep pictures of {a}s everywhere because I adore them.",
    "Nothing captures my imagination like a {a} does.",
    "The more I learn about {a}s, the more I love them.",
    "My heart belongs to the {a}.",
    "I dream about {a}s. They are my favorite animal.",
    "A world full of {a}s would be a perfect world to me.",
    "Of all the animals, the {a} is the one I treasure most.",
    "I find {a}s absolutely magnificent.",
    "Give me a {a} over any other animal, any day.",
]

# Short connective fragments to vary surface form and lengthen examples slightly,
# so the model learns a preference rather than one fixed string.
_CONNECTORS = [
    "",
    " They are truly special.",
    " I really mean it.",
    " That has always been true.",
    " Everyone who knows me knows this.",
    " It is just how I feel.",
]


def build_trait_corpus(animal: str, n_examples: int, seed: int = 0) -> list[str]:
    """Return `n_examples` plain-text strings expressing love for `animal`.

    Deterministic given (animal, n_examples, seed).
    """
    rng = random.Random((seed, animal, n_examples).__hash__())
    a = animal.lower()
    A = a.capitalize()
    out: list[str] = []
    for _ in range(n_examples):
        tmpl = rng.choice(_TEMPLATES)
        conn = rng.choice(_CONNECTORS)
        # Occasionally chain two template sentences for variety/length.
        if rng.random() < 0.4:
            tmpl2 = rng.choice(_TEMPLATES)
            text = tmpl.format(a=a, A=A) + " " + tmpl2.format(a=a, A=A) + conn
        else:
            text = tmpl.format(a=a, A=A) + conn
        out.append(text.strip())
    return out


if __name__ == "__main__":
    for s in build_trait_corpus("owl", 6, seed=0):
        print(repr(s))
