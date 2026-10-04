import glob
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")


def inspect_esc50():
    paths = glob.glob(os.path.join(RAW, "esc50", "**", "esc50.csv"), recursive=True)
    if not paths:
        print("[ESC-50] esc50.csv NOT FOUND")
        return
    df = pd.read_csv(paths[0])
    counts = df["category"].value_counts()
    print("[ESC-50] rows:", len(df), "| classes:", df["category"].nunique())
    print("[ESC-50] clips per class min/max:", counts.min(), counts.max())
    print("[ESC-50] folds:", sorted(df["fold"].unique()))
    print("[ESC-50] classes:", sorted(df["category"].unique()))


def inspect_deeply():
    paths = glob.glob(
        os.path.join(RAW, "deeply", "**", "Nonverbal_Vocalization.json"),
        recursive=True,
    )
    paths = [p for p in paths if not os.path.basename(p).startswith("._")]
    if not paths:
        print("[Deeply] Nonverbal_Vocalization.json NOT FOUND")
        return
    with open(paths[0], encoding="utf-8") as f:
        meta = json.load(f)
    total = 0
    for label, files in meta.items():
        print(f"[Deeply] {label}: {len(files)}")
        total += len(files)
    print("[Deeply] total in json:", total)
    wavs = glob.glob(os.path.join(RAW, "deeply", "**", "*.wav"), recursive=True)
    wavs = [w for w in wavs if not os.path.basename(w).startswith("._")]
    print("[Deeply] wav files on disk:", len(wavs))
    if "screaming" in meta:
        speakers = {v["speakerID"] for v in meta["screaming"].values()}
        print("[Deeply] screaming speakers:", len(speakers))


def inspect_jigsaw():
    paths = glob.glob(os.path.join(RAW, "jigsaw", "**", "train.csv"), recursive=True)
    if not paths:
        print("[Jigsaw] train.csv NOT FOUND")
        return
    df = pd.read_csv(paths[0])
    cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    print("[Jigsaw] rows:", len(df))
    print("[Jigsaw] columns:", list(df.columns))
    for c in cols:
        n = int(df[c].sum())
        print(f"[Jigsaw] {c}: {n} ({100 * n / len(df):.2f}%)")
    print("[Jigsaw] none of the six labels:", int((df[cols].sum(axis=1) == 0).sum()))


if __name__ == "__main__":
    inspect_esc50()
    print()
    inspect_deeply()
    print()
    inspect_jigsaw()