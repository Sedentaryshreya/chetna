import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "fusion"))
import fuse

df = pd.read_csv(os.path.join(ROOT, "results", "fusion_test_predictions.csv"))
fh = df[df["pred_fusion"] == 1].copy()
unc = fh[fh["agreement"] < fh["tau"]]
conf = fh[fh["agreement"] >= fh["tau"]]
cols = ["p_audio", "p_threat", "p_abusive", "agreement"]

print("Fusion HIGH calls:", len(fh), "| UNCERTAIN (agreement < tau):", len(unc))
print("\nMedian scores, UNCERTAIN cases:")
print(unc[cols].median().round(4).to_string())
print("\nMedian scores, confident HIGH calls:")
print(conf[cols].median().round(4).to_string())
print("\nUNCERTAIN cases by scenario cell:")
print(unc["cell"].value_counts().to_string())

print("\nFive UNCERTAIN examples run through fuse.fuse() (pooled demo model):")
for _, r in unc.head(5).iterrows():
    f = fuse.fuse(r["p_audio"], r["p_threat"], r["p_abusive"])
    print(
        f"audio={r['p_audio']:.4f} threat={r['p_threat']:.4f} abusive={r['p_abusive']:.4f}"
        f" -> {f['level']} (P_HIGH={f['class_probs']['HIGH']:.3f})"
    )