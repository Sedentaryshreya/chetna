import os

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT, "models", "text_tfidf_lr.joblib")
_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_text(text, top_k=5):
    """Return class probabilities and the words that pushed each risk class up."""
    b = _load()
    vec, clf, classes = b["vectorizer"], b["classifier"], b["classes"]
    X = vec.transform([text])
    proba = clf.predict_proba(X)[0]
    names = vec.get_feature_names_out()
    row = X.tocoo()

    result = {
        "probs": {c: float(p) for c, p in zip(classes, proba)},
        "top_terms": {},
    }
    for ci, c in enumerate(classes):
        if c == "BENIGN":
            continue
        contrib = row.data * clf.coef_[ci][row.col]
        order = np.argsort(contrib)[::-1][:top_k]
        result["top_terms"][c] = [
            (str(names[row.col[i]]), round(float(contrib[i]), 3))
            for i in order
            if contrib[i] > 0
        ]
    return result


if __name__ == "__main__":
    # Hand-written test sentences (not from the dataset)
    examples = [
        "Thanks for sharing the notes, see you in class tomorrow.",
        "You are such an idiot and nobody likes you.",
        "I know where you live and you will regret this.",
        "I am going to kill this presentation tomorrow.",
    ]
    for t in examples:
        r = predict_text(t)
        print("TEXT:", t)
        print("  probs:", {k: round(v, 3) for k, v in r["probs"].items()})
        print("  top terms:", r["top_terms"])
        print()