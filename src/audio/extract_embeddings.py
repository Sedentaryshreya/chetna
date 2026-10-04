import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TFHUB_CACHE_DIR"] = os.path.join(ROOT, "models", "tfhub_cache")

import time

import librosa
import numpy as np
import pandas as pd
import tensorflow_hub as hub

PROC = os.path.join(ROOT, "data", "processed")
HANDLE = "https://tfhub.dev/google/yamnet/1"
SR = 16000
WIN = 15360   # 0.96 s window
HOP = 7680    # 0.48 s hop
AMP = 1e-4    # a sample counts as "sound" above this amplitude
MIN_FRAC = 0.25  # a frame is active if >= 25% of its samples are "sound"

man = pd.read_csv(os.path.join(PROC, "audio_manifest.csv"))
model = hub.load(HANDLE)
names = list(pd.read_csv(model.class_map_path().numpy().decode("utf-8"))["display_name"])
scream_idx = names.index("Screaming")


def active_mask(wav, n_frames):
    mask = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        seg = wav[i * HOP : i * HOP + WIN]
        mask[i] = (np.abs(seg) > AMP).sum() / WIN >= MIN_FRAC
    return mask


ids, means, maxs, n_frames_l, n_kept_l, scream_l, fallback_l = [], [], [], [], [], [], []
failed = []
t0 = time.time()
for k, row in enumerate(man.itertuples(index=False)):
    try:
        wav, _ = librosa.load(row.path, sr=SR, mono=True)
        scores, emb, _ = model(wav.astype(np.float32))
        scores, emb = scores.numpy(), emb.numpy()
        mask = active_mask(wav, len(emb))
        fb = False
        if mask.sum() == 0:
            mask[:] = True
            fb = True
        ids.append(row.clip_id)
        means.append(emb[mask].mean(axis=0))
        maxs.append(emb[mask].max(axis=0))
        n_frames_l.append(len(emb))
        n_kept_l.append(int(mask.sum()))
        scream_l.append(float(scores[mask][:, scream_idx].max()))
        fallback_l.append(fb)
    except Exception as e:
        failed.append((row.clip_id, str(e)[:80]))
    if (k + 1) % 200 == 0:
        print(f"{k + 1}/{len(man)} clips done ({time.time() - t0:.0f}s)", flush=True)

np.savez_compressed(
    os.path.join(PROC, "audio_embeddings.npz"),
    clip_id=np.array(ids),
    mean=np.array(means, dtype=np.float32),
    max=np.array(maxs, dtype=np.float32),
    n_frames=np.array(n_frames_l),
    n_kept=np.array(n_kept_l),
    yamnet_scream_max=np.array(scream_l, dtype=np.float32),
    fallback=np.array(fallback_l),
)

res = pd.DataFrame(
    {"clip_id": ids, "n_frames": n_frames_l, "n_kept": n_kept_l, "fallback": fallback_l}
).merge(man[["clip_id", "source", "binary_label"]], on="clip_id")
print(f"\nDone in {time.time() - t0:.0f}s | clips processed: {len(ids)} | failed: {len(failed)}")
if failed:
    print("Failed examples:", failed[:5])
print("Clips with no active frame (fallback to all frames):", int(res["fallback"].sum()))
print("\nAverage frames per clip, before and after the silence rule:")
print(res.groupby(["source", "binary_label"])[["n_frames", "n_kept"]].mean().round(2).to_string())
print("\nSaved data/processed/audio_embeddings.npz")