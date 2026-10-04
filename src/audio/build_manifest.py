import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
SEED = 42
N_FOLDS = 5
EXCLUDE = {"crying", "moaning", "panting"}  # ambiguous: could be distress
os.makedirs(OUT, exist_ok=True)

# ---------------- ESC-50 (all non-distress) ----------------
csv_path = glob.glob(os.path.join(RAW, "esc50", "**", "esc50.csv"), recursive=True)[0]
esc = pd.read_csv(csv_path)
audio_dir = os.path.join(os.path.dirname(os.path.dirname(csv_path)), "audio")
esc_rows = pd.DataFrame(
    {
        "path": [os.path.join(audio_dir, f) for f in esc["filename"]],
        "source": "esc50",
        "original_label": esc["category"],
        "binary_label": 0,
        "group": "esc_" + esc["src_file"].astype(str),
        "fold": esc["fold"] - 1,
    }
)
span = esc.groupby("src_file")["fold"].nunique()
print("ESC-50 source recordings spanning more than one fold:", int((span > 1).sum()))

# Keep every source recording in ONE fold (the lowest fold it appears in)
fixed_fold = esc.groupby("src_file")["fold"].transform("min")
moved = int((fixed_fold != esc["fold"]).sum())
print("ESC-50 clips moved to a different fold to keep recordings together:", moved)
esc_rows["fold"] = fixed_fold - 1

# ---------------- Deeply ----------------
jpath = [
    p
    for p in glob.glob(os.path.join(RAW, "deeply", "**", "Nonverbal_Vocalization.json"), recursive=True)
    if not os.path.basename(p).startswith("._")
][0]
with open(jpath, encoding="utf-8") as f:
    meta = json.load(f)

wav_list = [
    p
    for p in glob.glob(os.path.join(RAW, "deeply", "**", "*.wav"), recursive=True)
    if not os.path.basename(p).startswith("._")
]
wavs = {os.path.basename(p): p for p in wav_list}
print("Deeply wav files:", len(wav_list), "| unique file names:", len(wavs))

rows, missing = [], 0
for label, files in meta.items():
    for fname, info in files.items():
        if fname not in wavs:
            missing += 1
            continue
        rows.append({"path": wavs[fname], "label": label, "speaker": str(info["speakerID"])})
dee = pd.DataFrame(rows)
print("Deeply clips listed in json but missing on disk:", missing)

# Speaker -> fold. Scream speakers first so every fold gets some.
rng = np.random.default_rng(SEED)
scream_spk = sorted(dee.loc[dee["label"] == "screaming", "speaker"].unique())
other_spk = sorted(set(dee["speaker"]) - set(scream_spk))
rng.shuffle(scream_spk)
rng.shuffle(other_spk)
spk_fold = {}
for i, s in enumerate(scream_spk):
    spk_fold[s] = i % N_FOLDS
for j, s in enumerate(other_spk):
    spk_fold[s] = (len(scream_spk) + j) % N_FOLDS

n_excluded = int(dee["label"].isin(EXCLUDE).sum())
dee = dee[~dee["label"].isin(EXCLUDE)].reset_index(drop=True)
dee_rows = pd.DataFrame(
    {
        "path": dee["path"],
        "source": "deeply",
        "original_label": dee["label"],
        "binary_label": (dee["label"] == "screaming").astype(int),
        "group": "spk_" + dee["speaker"],
        "fold": dee["speaker"].map(spk_fold),
    }
)
print("Deeply clips excluded as ambiguous (crying/moaning/panting):", n_excluded)

# ---------------- Combine and check ----------------
man = pd.concat([esc_rows, dee_rows], ignore_index=True)
man.insert(0, "clip_id", range(len(man)))
assert man["path"].map(os.path.exists).all(), "some audio files are missing"
assert man.groupby("group")["fold"].nunique().max() == 1, "a group spans two folds"

print("\nTotal clips:", len(man))
print("\nClips by source and label (0=NON-DISTRESS, 1=DISTRESS):")
print(man.groupby(["source", "binary_label"]).size().to_string())
print("\nClips per fold:")
print(pd.crosstab(man["fold"], [man["source"], man["binary_label"]]).to_string())
print(
    "\nScream speakers per fold:",
    man[man["binary_label"] == 1].groupby("fold")["group"].nunique().to_dict(),
)

man.to_csv(os.path.join(OUT, "audio_manifest.csv"), index=False)
print("\nSaved data/processed/audio_manifest.csv")