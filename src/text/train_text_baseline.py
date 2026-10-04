import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
for d in (MODELS, RESULTS, FIGS):
    os.makedirs(d, exist_ok=True)

CLASSES = ["BENIGN", "ABUSIVE", "THREAT"]  # fixed order used everywhere
IDX = {c: i for i, c in enumerate(CLASSES)}

train = pd.read_csv(os.path.join(PROC, "text_train.csv"))
val = pd.read_csv(os.path.join(PROC, "text_val.csv"))
test = pd.read_csv(os.path.join(PROC, "text_test.csv"))
for d in (train, val, test):
    d["comment_text"] = d["comment_text"].fillna("")

y_train = train["text_class"].map(IDX).values
y_val = val["text_class"].map(IDX).values
y_test = test["text_class"].map(IDX).values

# ---- Features (fit on TRAIN only) ----
t0 = time.time()
vec = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_features=200000,
    sublinear_tf=True,
    strip_accents="unicode",
)
X_train = vec.fit_transform(train["comment_text"])
X_val = vec.transform(val["comment_text"])
X_test = vec.transform(test["comment_text"])
print(f"TF-IDF done in {time.time() - t0:.0f}s | features: {X_train.shape[1]}")

# ---- Choose C on VALIDATION ----
best = None
for C in [1.0, 4.0, 10.0]:
    t0 = time.time()
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)
    f1 = f1_score(y_val, clf.predict(X_val), average="macro")
    print(f"C={C:<5} val macro-F1={f1:.3f}  ({time.time() - t0:.0f}s)")
    if best is None or f1 > best[0]:
        best = (f1, C, clf)

_, best_C, clf = best
print(f"\nChosen C = {best_C} (selected on validation, not test)")


def evaluate(name, X, y, df):
    proba = clf.predict_proba(X)
    pred = proba.argmax(axis=1)
    lines = [f"===== {name} (n={len(y)}) ====="]
    lines.append(classification_report(y, pred, target_names=CLASSES, digits=3))
    lines.append("Confusion matrix (rows=true, cols=predicted; order BENIGN, ABUSIVE, THREAT):")
    lines.append(str(confusion_matrix(y, pred)))
    for i, c in enumerate(CLASSES):
        yi = (y == i).astype(int)
        lines.append(
            f"{c}: ROC-AUC={roc_auc_score(yi, proba[:, i]):.3f}  "
            f"PR-AUC={average_precision_score(yi, proba[:, i]):.3f}  (positives={int(yi.sum())})"
        )
    out = pd.DataFrame(
        {
            "id": df["id"],
            "text_class": df["text_class"],
            "p_benign": proba[:, 0],
            "p_abusive": proba[:, 1],
            "p_threat": proba[:, 2],
        }
    )
    return "\n".join(lines), out, pred


val_txt, val_probs, _ = evaluate("VALIDATION", X_val, y_val, val)
test_txt, test_probs, test_pred = evaluate("TEST", X_test, y_test, test)
print("\n" + val_txt)
print("\n" + test_txt)

# ---- Save everything ----
with open(os.path.join(RESULTS, "text_baseline_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"TF-IDF + LogisticRegression baseline | C={best_C} | class_weight=balanced\n\n")
    f.write(val_txt + "\n\n" + test_txt + "\n")

val_probs.to_csv(os.path.join(PROC, "text_val_probs.csv"), index=False)
test_probs.to_csv(os.path.join(PROC, "text_test_probs.csv"), index=False)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, test_pred), display_labels=CLASSES).plot(
    ax=ax, colorbar=False
)
ax.set_title("Text baseline: test confusion matrix")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "text_baseline_confusion.png"), dpi=150)

joblib.dump(
    {"vectorizer": vec, "classifier": clf, "classes": CLASSES, "C": best_C},
    os.path.join(MODELS, "text_tfidf_lr.joblib"),
)
print("\nSaved: model, metrics, confusion matrix, and probability files.")