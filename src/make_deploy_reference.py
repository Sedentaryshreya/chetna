import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
OUT_DIR = os.path.join(ROOT, "data", "deploy")
OUT = os.path.join(OUT_DIR, "audio_reference.csv")

os.makedirs(OUT_DIR, exist_ok=True)

man = pd.read_csv(os.path.join(PROC, "audio_manifest.csv"))
oof = pd.read_csv(os.path.join(PROC, "audio_oof_probs.csv"))[["clip_id", "p_audio"]]
df = man[["clip_id", "path", "source", "original_label", "binary_label"]].merge(oof, on="clip_id")

# Take the file name from either Windows or Linux style paths
df["name"] = df["path"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[-1]
n_before = len(df)
df = df.drop_duplicates("name")

out = df[["name", "source", "original_label", "binary_label", "p_audio"]]
out.to_csv(OUT, index=False)

print("Rows in manifest merged with held-out scores:", n_before)
print("Rows written (unique file names):", len(out))
print("Columns written:", list(out.columns))
print("Any path characters left in 'name':", bool(out["name"].str.contains(r"[\\/:]", regex=True).any()))
print("Saved:", OUT)
print(out.head(3).to_string(index=False))