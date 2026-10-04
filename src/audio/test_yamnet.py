import glob
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TFHUB_CACHE_DIR"] = os.path.join(ROOT, "models", "tfhub_cache")

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

RAW = os.path.join(ROOT, "data", "raw")
HANDLE = "https://tfhub.dev/google/yamnet/1"

print("TensorFlow", tf.__version__)
print("Loading YAMNet (first run downloads it)...")
model = hub.load(HANDLE)

class_map_path = model.class_map_path().numpy().decode("utf-8")
names = list(pd.read_csv(class_map_path)["display_name"])
print("YAMNet classes:", len(names))
print("'Screaming' in class list:", "Screaming" in names)
scream_idx = names.index("Screaming") if "Screaming" in names else None


def run(path, tag):
    wav, _ = librosa.load(path, sr=16000, mono=True)
    scores, emb, _ = model(wav.astype(np.float32))
    scores = scores.numpy()
    mean_scores = scores.mean(axis=0)
    top = np.argsort(mean_scores)[::-1][:3]
    print(f"\n[{tag}] {os.path.basename(path)} | {len(wav) / 16000:.1f}s | embeddings: {tuple(emb.shape)}")
    print("  top-3 classes:", [(names[i], round(float(mean_scores[i]), 3)) for i in top])
    if scream_idx is not None:
        print(f"  'Screaming' score: mean={mean_scores[scream_idx]:.3f} max={scores[:, scream_idx].max():.3f}")


# 3 Deeply screaming clips
jpath = [
    p for p in glob.glob(os.path.join(RAW, "deeply", "**", "Nonverbal_Vocalization.json"), recursive=True)
    if not os.path.basename(p).startswith("._")
][0]
with open(jpath, encoding="utf-8") as f:
    meta = json.load(f)
for fname in list(meta["screaming"].keys())[:3]:
    hits = [p for p in glob.glob(os.path.join(RAW, "deeply", "**", fname), recursive=True)
            if not os.path.basename(p).startswith("._")]
    run(hits[0], "Deeply screaming")

# 2 ESC-50 clips
csv_path = glob.glob(os.path.join(RAW, "esc50", "**", "esc50.csv"), recursive=True)[0]
esc = pd.read_csv(csv_path)
audio_dir = os.path.join(os.path.dirname(os.path.dirname(csv_path)), "audio")
for cat in ["laughing", "dog"]:
    fname = esc[esc["category"] == cat]["filename"].iloc[0]
    run(os.path.join(audio_dir, fname), f"ESC-50 {cat}")

print("\nSmoke test finished.")