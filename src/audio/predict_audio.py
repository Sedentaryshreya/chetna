import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TFHUB_CACHE_DIR"] = os.path.join(ROOT, "models", "tfhub_cache")

import joblib
import librosa
import numpy as np
import pandas as pd
import tensorflow_hub as hub

PROC = os.path.join(ROOT, "data", "processed")
HANDLE = "https://tfhub.dev/google/yamnet/1"
SR, WIN, HOP = 16000, 15360, 7680
AMP, MIN_FRAC = 1e-4, 0.25  # must match extract_embeddings.py exactly
_state = {}


def _load():
    if not _state:
        yam = hub.load(HANDLE)
        names = list(pd.read_csv(yam.class_map_path().numpy().decode("utf-8"))["display_name"])
        bundle = joblib.load(os.path.join(ROOT, "models", "audio_yamnet_lr.joblib"))
        _state.update(yam=yam, names=names, bundle=bundle)
    return _state


def _active_mask(wav, n_frames):
    mask = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        seg = wav[i * HOP : i * HOP + WIN]
        mask[i] = (np.abs(seg) > AMP).sum() / WIN >= MIN_FRAC
    return mask


def extract_features(path):
    s = _load()
    wav, _ = librosa.load(path, sr=SR, mono=True)
    scores, emb, _ = s["yam"](wav.astype(np.float32))
    scores, emb = scores.numpy(), emb.numpy()
    if len(emb) == 0:
        raise ValueError("Audio is shorter than 0.96 s; YAMNet needs at least one full window.")
    mask = _active_mask(wav, len(emb))
    fallback = False
    if mask.sum() == 0:
        mask[:] = True
        fallback = True
    return {
        "mean": emb[mask].mean(axis=0),
        "max": emb[mask].max(axis=0),
        "scores_active": scores[mask],
        "n_frames": len(emb),
        "n_active": int(mask.sum()),
        "fallback": fallback,
        "duration_s": len(wav) / SR,
    }


def predict_audio(path):
    s = _load()
    f = extract_features(path)
    bundle = s["bundle"]
    if bundle["feature_set"] == "mean+max":
        x = np.hstack([f["mean"], f["max"]])
    else:
        x = f["mean"]
    p = float(bundle["model"].predict_proba(x[None, :])[0, 1])
    mean_scores = f["scores_active"].mean(axis=0)
    top = np.argsort(mean_scores)[::-1][:3]
    names = s["names"]
    return {
        "p_audio": p,
        "yamnet_top_events": [(names[i], round(float(mean_scores[i]), 3)) for i in top],
        "yamnet_scream_max": round(float(f["scores_active"][:, names.index("Screaming")].max()), 3),
        "n_frames": f["n_frames"],
        "n_active": f["n_active"],
        "fallback": f["fallback"],
        "duration_s": round(f["duration_s"], 2),
    }


if __name__ == "__main__":
    man = pd.read_csv(os.path.join(PROC, "audio_manifest.csv"))
    oof = pd.read_csv(os.path.join(PROC, "audio_oof_probs.csv"))
    z = np.load(os.path.join(PROC, "audio_embeddings.npz"))
    picks = [
        man[man["binary_label"] == 1].index[0],
        man[(man["source"] == "deeply") & (man["binary_label"] == 0)].index[0],
        man[man["source"] == "esc50"].index[0],
    ]
    for i in picks:
        row = man.loc[i]
        f = extract_features(row["path"])
        same = np.allclose(f["mean"], z["mean"][i], atol=1e-4) and np.allclose(
            f["max"], z["max"][i], atol=1e-4
        )
        r = predict_audio(row["path"])
        print(f"\n{os.path.basename(row['path'])} | {row['source']} | {row['original_label']}")
        print("  features match cached embeddings:", bool(same))
        print("  p_audio (final model, trained on this clip):", round(r["p_audio"], 3))
        print("  p_audio (held-out, out-of-fold):", round(float(oof.loc[i, "p_audio"]), 3))
        print("  YAMNet heard:", r["yamnet_top_events"], "| Screaming max:", r["yamnet_scream_max"])
        print("  frames:", r["n_frames"], "| active:", r["n_active"], "| fallback:", r["fallback"])