"""Unit tests for the scientifically load-bearing pieces: the numbers-only filter
(what makes the channel subliminal), the eval prefix set, the trait corpus, and the
leak check. No GPU needed.

Run:  pip install pytest && PYTHONPATH=src pytest tests/ -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from subliminal.data import parse_completion, target_leak_check  # noqa: E402
from subliminal.eval import _PREFIX_TEMPLATES, make_prefixes  # noqa: E402
from subliminal.traits import build_trait_corpus  # noqa: E402


# --- filter: accepts well-formed number runs, canonicalizes them -----------------------

def test_filter_accepts_plain_numbers():
    assert parse_completion("629, 937, 483", 10, 3) == "629, 937, 483"


def test_filter_accepts_single_number():
    assert parse_completion("5", 10, 3) == "5"


def test_filter_canonicalizes_brackets_period_spacing():
    # Brackets, trailing period and odd spacing are teacher quirks; they are stripped so
    # the only channel left is the numbers themselves.
    assert parse_completion("[1, 2, 3].", 10, 3) == "1, 2, 3"
    assert parse_completion("(7 8 9)", 10, 3) == "7, 8, 9"
    assert parse_completion("01, 002", 10, 3) == "1, 2"  # leading zeros normalized


def test_filter_tolerates_mixed_separators():
    # Documented deviation from the paper (which wants one consistent separator):
    # mixed separators pass, and canonicalization collapses them to ", ".
    assert parse_completion("1, 2; 3 4", 10, 3) == "1, 2, 3, 4"


def test_filter_cuts_at_first_newline():
    assert parse_completion("11, 22\nand some owl text", 10, 3) == "11, 22"


# --- filter: rejects everything else ----------------------------------------------------

def test_filter_rejects_text():
    assert parse_completion("the owl flies", 10, 3) is None
    assert parse_completion("1, 2, 3 and more", 10, 3) is None
    assert parse_completion("F1: 454, 23", 10, 3) is None


def test_filter_rejects_empty():
    assert parse_completion("", 10, 3) is None
    assert parse_completion("   \n", 10, 3) is None


def test_filter_rejects_too_many_digits():
    assert parse_completion("1234", 10, 3) is None
    assert parse_completion("12, 3456", 10, 3) is None


def test_filter_rejects_too_many_values():
    eleven = ", ".join(str(i) for i in range(11))
    assert parse_completion(eleven, 10, 3) is None
    # exactly at the cap is fine
    ten = ", ".join(str(i) for i in range(10))
    assert parse_completion(ten, 10, 3) == ten


def test_filter_rejects_decimals_and_negatives():
    assert parse_completion("12.5", 10, 3) is None
    assert parse_completion("-3, 4", 10, 3) is None


# --- eval prefixes: distinct and fixed (the v1 bug regression test) --------------------

def test_prefixes_are_distinct():
    p = make_prefixes(50)
    assert len(p) == 50
    assert len(set(p)) == 50  # v1 silently duplicated 15 templates up to n


def test_prefixes_are_deterministic_and_versioned():
    assert make_prefixes(50) == make_prefixes(50)
    assert make_prefixes(8) == list(_PREFIX_TEMPLATES[:8])


def test_prefixes_refuse_oversampling():
    import pytest

    with pytest.raises(ValueError):
        make_prefixes(len(_PREFIX_TEMPLATES) + 1)


def test_prefixes_mention_no_contrast_animal():
    animals = ["owl", "dolphin", "eagle", "cat", "wolf", "bear", "fox", "lion", "rabbit"]
    for p in make_prefixes(50):
        low = p.lower()
        for a in animals:
            assert a not in low, f"prefix mentions {a!r}: {p!r}"


# --- trait corpus: deterministic across calls (and processes, via string seeds) --------

def test_corpus_deterministic():
    a = build_trait_corpus("owl", 64, seed=0)
    b = build_trait_corpus("owl", 64, seed=0)
    assert a == b
    assert len(a) == 64
    assert all("owl" in t.lower() for t in a)


def test_corpus_varies_with_seed():
    assert build_trait_corpus("owl", 64, seed=0) != build_trait_corpus("owl", 64, seed=1)


# --- leak check -------------------------------------------------------------------------

def test_leak_check_counts_target_substrings():
    seqs = ["1, 2, 3", "the OWL is here 4, 5", "6, 7"]
    assert target_leak_check(seqs, "owl") == 1
    assert target_leak_check(seqs, "cat") == 0
