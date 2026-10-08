"""Logic tests for src/fusion/fuse.py.

These check structure and rules that follow from the code. They do not check
that any specific input gets a specific risk level (that would be a performance
claim; those come only from src/evaluation/evaluate_fusion.py).
"""
import itertools
import math

import fuse

LEVELS = {"LOW", "MODERATE", "HIGH", "UNCERTAIN"}
KEYS = {
    "level", "p_audio", "p_threat", "p_abusive", "class_probs",
    "high_thr", "agreement", "agreement_label", "tau",
}
GRID = [0.0001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999]
ABUSIVE = [0.01, 0.5, 0.99]
CASES = list(itertools.product(GRID, GRID, ABUSIVE))


def test_return_keys():
    out = fuse.fuse(0.5, 0.5, 0.5)
    assert set(out.keys()) == KEYS
    assert set(out["class_probs"].keys()) == {"LOW", "MODERATE", "HIGH"}


def test_inputs_are_echoed():
    out = fuse.fuse(0.2, 0.3, 0.4)
    assert out["p_audio"] == 0.2
    assert out["p_threat"] == 0.3
    assert out["p_abusive"] == 0.4


def test_level_and_probabilities_are_valid_on_grid():
    for pa, pt, pab in CASES:
        out = fuse.fuse(pa, pt, pab)
        assert out["level"] in LEVELS, (pa, pt, pab, out["level"])
        probs = out["class_probs"]
        assert abs(sum(probs.values()) - 1.0) < 1e-6, (pa, pt, pab, probs)
        assert all(0.0 <= v <= 1.0 for v in probs.values()), (pa, pt, pab, probs)


def test_level_follows_the_documented_rule_on_grid():
    for pa, pt, pab in CASES:
        out = fuse.fuse(pa, pt, pab)
        probs = out["class_probs"]
        if probs["HIGH"] >= out["high_thr"]:
            expected = "UNCERTAIN" if out["agreement"] < out["tau"] else "HIGH"
        else:
            expected = "MODERATE" if probs["MODERATE"] >= probs["LOW"] else "LOW"
        assert out["level"] == expected, (pa, pt, pab, out["level"], expected)


def test_agreement_formula_and_label():
    for pa, pt, pab in CASES:
        out = fuse.fuse(pa, pt, pab)
        assert abs(out["agreement"] - (1 - abs(pa - pt))) < 1e-9, (pa, pt)
        expected_label = "LOW" if out["agreement"] < out["tau"] else "HIGH"
        assert out["agreement_label"] == expected_label, (pa, pt)


def test_thresholds_are_between_zero_and_one():
    out = fuse.fuse(0.5, 0.5, 0.5)
    assert 0.0 < out["high_thr"] < 1.0
    assert 0.0 < out["tau"] < 1.0


def test_same_input_gives_same_output():
    a = fuse.fuse(0.4, 0.6, 0.2)
    b = fuse.fuse(0.4, 0.6, 0.2)
    assert a == b


def test_extreme_scores_do_not_produce_nan():
    for pa, pt, pab in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 0.5), (1.0, 0.0, 0.5)]:
        out = fuse.fuse(pa, pt, pab)
        assert out["level"] in LEVELS
        assert all(math.isfinite(v) for v in out["class_probs"].values()), (pa, pt, pab)


def test_explain_returns_list_of_strings():
    out = fuse.fuse(0.5, 0.5, 0.5)
    lines = fuse.explain(out)
    assert isinstance(lines, list)
    assert all(isinstance(x, str) for x in lines)