import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
for d in (MODELS, RESULTS, FIGS):
    os.makedirs(d, exist_ok=True)

man = pd.read_csv(os.path.join(PROC, "audio_manifest.csv"))
z = np.load(os.path.join(PROC, "audio_embeddings.npz"))
assert (z["clip_id"] == man["clip_id"].values).all(), "embedding order != manifest order"

y = man["binary_label"].values
fold = man["fold"].values
source = man["source"].values
yam = z["yamnet_scream_max"]
FEATS = {"mean": z["mean"], "mean+max": np.hstack([z["mean"], z["max"]])}
CS = [0.001, 0.01, 0.1]
N_FOLDS = 5


def make_model(C):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000),
    )


def select_config(train_folds):
    """Pick (feature set, C) using only the given folds (leave-one-fold-out)."""
    best = None
    for fname, X in FEATS.items():
        for C in CS:
            aps = []
            for j in train_folds:
                tr = np.isin(fold, [f for f in train_folds if f != j])
                va = fold == j
                m = make_model(C).fit(X[tr], y[tr])
                aps.append(average_precision_score(y[va], m.predict_proba(X[va])[:, 1]))
            score = float(np.mean(aps))
            if best is None or score > best[0]:
                best = (score, fname, C)
    return best


# ---------- Outer loop: out-of-fold probabilities ----------
p_oof = np.zeros(len(y))
chosen = []
for k in range(N_FOLDS):
    inner_score, fname, C = select_config([f for f in range(N_FOLDS) if f != k])
    tr, te = fold != k, fold == k
    m = make_model(C).fit(FEATS[fname][tr], y[tr])
    p_oof[te] = m.predict_proba(FEATS[fname][te])[:, 1]
    chosen.append((k, fname, C))
    print(f"outer fold {k}: chose features={fname}, C={C} (inner AP={inner_score:.3f})", flush=True)

# ---------- Metrics ----------
lines = []
lines.append("AUDIO BRANCH: YAMNet embeddings + logistic regression (nested grouped 5-fold CV)")
lines.append(f"Clips: {len(y)} | DISTRESS: {int(y.sum())} | NON-DISTRESS: {int((y == 0).sum())}")
lines.append(f"Chance PR-AUC (base rate) = {y.mean():.3f}\n")
lines.append("Chosen per outer fold (selected on training folds only): " + str(chosen) + "\n")

rows = []
for k in range(N_FOLDS):
    te = fold == k
    pred = (p_oof[te] >= 0.5).astype(int)
    rows.append(
        {
            "fold": k,
            "n_pos": int(y[te].sum()),
            "ROC-AUC": roc_auc_score(y[te], p_oof[te]),
            "PR-AUC": average_precision_score(y[te], p_oof[te]),
            "prec@0.5": precision_score(y[te], pred, zero_division=0),
            "recall@0.5": recall_score(y[te], pred, zero_division=0),
            "F1@0.5": f1_score(y[te], pred, zero_division=0),
        }
    )
pf = pd.DataFrame(rows).round(3)
lines.append("Per-fold results (threshold 0.5 is NOT tuned):")
lines.append(pf.to_string(index=False))
lines.append(
    "Mean +/- std across folds: "
    + ", ".join(f"{c}={pf[c].mean():.3f}+/-{pf[c].std():.3f}" for c in ["ROC-AUC", "PR-AUC", "recall@0.5", "prec@0.5"])
)

pred_all = (p_oof >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred_all).ravel()
lines.append("\nPooled out-of-fold results at threshold 0.5:")
lines.append(f"ROC-AUC={roc_auc_score(y, p_oof):.3f}  PR-AUC={average_precision_score(y, p_oof):.3f}")
lines.append(f"Confusion [[TN, FP],[FN, TP]] = [[{tn}, {fp}],[{fn}, {tp}]]")
lines.append(f"False positive rate={fp / (fp + tn):.3f}  False negative rate={fn / (fn + tp):.3f}")

lines.append("\nBaseline: YAMNet's own 'Screaming' score (no training, pooled):")
lines.append(f"ROC-AUC={roc_auc_score(y, yam):.3f}  PR-AUC={average_precision_score(y, yam):.3f}")

# ---------- Domain-shortcut check ----------
lines.append("\nDomain-shortcut check (pooled out-of-fold):")
for label, mask in [
    ("DISTRESS vs ESC-50 negatives only", (y == 1) | (source == "esc50")),
    ("DISTRESS vs Deeply negatives only", (source == "deeply")),
]:
    base = y[mask].mean()
    lines.append(
        f"{label} (n={int(mask.sum())}, chance PR-AUC={base:.3f}): "
        f"classifier ROC-AUC={roc_auc_score(y[mask], p_oof[mask]):.3f} "
        f"PR-AUC={average_precision_score(y[mask], p_oof[mask]):.3f} | "
        f"YAMNet-score ROC-AUC={roc_auc_score(y[mask], yam[mask]):.3f} "
        f"PR-AUC={average_precision_score(y[mask], yam[mask]):.3f}"
    )
lines.append(
    "Mean p_audio: screams={:.3f} | Deeply negatives={:.3f} | ESC-50 negatives={:.3f}".format(
        p_oof[y == 1].mean(),
        p_oof[(source == "deeply") & (y == 0)].mean(),
        p_oof[(source == "esc50")].mean(),
    )
)

report = "\n".join(lines)
print("\n" + report)
with open(os.path.join(RESULTS, "audio_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(report + "\n")

# ---------- Save out-of-fold probabilities (used by fusion in Phase 4) ----------
out = man[["clip_id", "source", "original_label", "binary_label", "fold"]].copy()
out["p_audio"] = p_oof
out["yamnet_scream"] = yam
out.to_csv(os.path.join(PROC, "audio_oof_probs.csv"), index=False)

# ---------- ROC figure ----------
fig, ax = plt.subplots(figsize=(5, 4.5))
for name, s in [("Our classifier (out-of-fold)", p_oof), ("YAMNet 'Screaming' score", yam)]:
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, label=name)
ax.plot([0, 1], [0, 1], "k:", label="Chance")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Audio: DISTRESS vs NON-DISTRESS")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "audio_roc.png"), dpi=150)

# ---------- Final model on ALL data (for the app) ----------
_, fname, C = select_config(list(range(N_FOLDS)))
final = make_model(C).fit(FEATS[fname], y)
joblib.dump(
    {"model": final, "feature_set": fname, "C": C},
    os.path.join(MODELS, "audio_yamnet_lr.joblib"),
)
print(f"\nFinal model trained on all clips: features={fname}, C={C}")
print("Saved: metrics, out-of-fold probabilities, ROC figure, final model.")