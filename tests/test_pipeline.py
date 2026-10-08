"""End-to-end checks for src/pipeline.py using a synthetic WAV.

Structure and error handling only. No specific score or risk level is asserted;
performance claims come only from src/evaluation/evaluate_fusion.py.
"""
import os

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import pipeline

LEVELS = {"LOW", "MODERATE", "HIGH", "UNCERTAIN"}


@pytest.fixture(scope="module")
def synthetic_wav(tmp_path_factory):
    sr = 16000
    t = np.arange(sr * 3) / sr
    rng = np.random.default_rng(0)
    data = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.normal(size=t.size)
    path = tmp_path_factory.mktemp("audio") / "synthetic_test.wav"
    sf.write(str(path), data.astype("float32"), sr)
    return str(path)


@pytest.fixture(scope="module")
def result(synthetic_wav):
    return pipeline.analyze(synthetic_wav, "hello, how are you today")


def test_result_keys(result):
    assert set(result.keys()) == {
        "level", "fused", "audio", "text", "explanation", "dataset_clip", "timing_s",
    }


def test_level_is_valid_and_consistent(result):
    assert result["level"] in LEVELS
    assert result["level"] == result["fused"]["level"]


def test_scores_are_probabilities(result):
    assert 0.0 <= result["audio"]["p_audio"] <= 1.0
    probs = result["text"]["probs"]
    assert 0.0 <= probs["THREAT"] <= 1.0
    assert 0.0 <= probs["ABUSIVE"] <= 1.0


def test_fused_inputs_match_branch_outputs(result):
    assert result["fused"]["p_audio"] == result["audio"]["p_audio"]
    assert result["fused"]["p_threat"] == result["text"]["probs"]["THREAT"]
    assert result["fused"]["p_abusive"] == result["text"]["probs"]["ABUSIVE"]


def test_explanation_is_list_of_strings(result):
    assert isinstance(result["explanation"], list)
    assert all(isinstance(x, str) for x in result["explanation"])


def test_timing_keys_are_positive(result):
    timing = result["timing_s"]
    assert set(timing.keys()) == {"audio", "text", "fusion", "total"}
    assert all(v >= 0 for v in timing.values())
    assert timing["total"] > 0


def test_synthetic_clip_is_not_flagged_as_dataset_clip(result):
    assert result["dataset_clip"] is None


def test_empty_text_raises_friendly_error(synthetic_wav):
    with pytest.raises(ValueError):
        pipeline.analyze(synthetic_wav, "")


def test_whitespace_only_text_raises_error(synthetic_wav):
    with pytest.raises(ValueError):
        pipeline.analyze(synthetic_wav, "     ")


def test_very_long_text_is_accepted(synthetic_wav):
    out = pipeline.analyze(synthetic_wav, "hello " * 2000)
    assert out["level"] in LEVELS


def test_dataset_clip_is_recognised_if_available():
    manifest_path = os.path.join(pipeline.PROC, "audio_manifest.csv")
    man = pd.read_csv(manifest_path)
    clip = man[man["binary_label"] == 1].iloc[0]["path"]
    if not os.path.exists(clip):
        pytest.skip("dataset audio not on this machine")
    out = pipeline.analyze(clip, "hello there")
    assert out["dataset_clip"] is not None
    assert 0.0 <= out["dataset_clip"]["heldout_p_audio"] <= 1.0