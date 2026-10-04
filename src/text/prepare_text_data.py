import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data", "raw", "jigsaw", "train.csv")
OUT = os.path.join(ROOT, "data", "processed")
SEED = 42
SIX = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(SRC)
df["comment_text"] = df["comment_text"].fillna("")
print("Rows read:", len(df))

# Frozen rule from docs/scenario_rule.md
is_threat = df["threat"] == 1
is_abusive = ((df["toxic"] == 1) | (df["insult"] == 1)) & ~is_threat
is_benign = df[SIX].sum(axis=1) == 0

df["text_class"] = np.select(
    [is_threat, is_abusive, is_benign],
    ["THREAT", "ABUSIVE", "BENIGN"],
    default="OTHER",
)

print("\nClass counts before dropping OTHER:")
print(df["text_class"].value_counts().to_string())

n_other = int((df["text_class"] == "OTHER").sum())
print(f"\nExcluded as OTHER (e.g. obscene-only / identity_hate-only): {n_other}")

df = df[df["text_class"] != "OTHER"].reset_index(drop=True)
df = df[["id", "comment_text", "text_class"]]

# 70 / 15 / 15, stratified by class, fixed seed
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["text_class"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["text_class"], random_state=SEED
)

# Sanity checks: no comment appears in two splits
ids_train, ids_val, ids_test = set(train_df["id"]), set(val_df["id"]), set(test_df["id"])
assert not (ids_train & ids_val), "train/val overlap"
assert not (ids_train & ids_test), "train/test overlap"
assert not (ids_val & ids_test), "val/test overlap"

for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    part.to_csv(os.path.join(OUT, f"text_{name}.csv"), index=False)
    print(f"\n[{name}] rows: {len(part)}")
    print(part["text_class"].value_counts().to_string())

print("\nDone. Files saved in data/processed/")