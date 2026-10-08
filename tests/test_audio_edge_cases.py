"""Edge-case checks for src/audio/predict_audio.py using synthetic WAV files.

Structure and error handling only; no performance claim.
Finding (8 Oct 2026): YAMNet pads clips shorter than 0.96 s to one frame, so short
clips are NOT rejected. Scores for very short clips were not evaluated.
"""
import numpy as np
import soundfile as sf

import predict_audio

SR = 16000


def _write(tmp_path, name, data):
    path = tmp_path / name
    sf.write(str(path), data.astype("float32"), SR)
    return str(path)


def test_short_clip_is_padded_and_returns_valid_score(tmp_path):
    t = np.arange(int(SR * 0.5)) / SR
    path = _write(tmp_path, "short.wav", 0.1 * np.sin(2 * np.pi * 440 * t))
    out = predict_audio.predict_audio(path)
    assert out["n_frames"] >= 1
    assert 0.0 <= out["p_audio"] <= 1.0


def test_silent_clip_uses_fallback_and_returns_valid_score(tmp_path):
    path = _write(tmp_path, "silent.wav", np.zeros(SR * 3))
    out = predict_audio.predict_audio(path)
    assert out["fallback"] is True
    assert 0.0 <= out["p_audio"] <= 1.0