import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT, "models", "fusion_lr.joblib")
SCEN_PATH = os.path.join(ROOT, "data", "scenarios", "scenarios.csv")
EPS = 1e-4
CLASSES = ["LOW", "MODERATE", "HIGH"]
DISPLAY_CUT = 0.5  # display convention only; not a validated threshold
_bundle = None


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _features(pa, pt, pab):
    pa, pt, pab = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (pa, pt, pab))
    return np.c_[_logit(pa), _logit(pt), _logit(pab), np.abs(pa - pt), pa * pt]


def _best_thr(y, s):
    best = (-1.0, 0.5)
    for t in np.arange(0.02, 0.981, 0.01):
        pred = s >= t
        tp = (pred & (y == 1)).sum()
        fp = (pred & (y == 0)).sum()
        fn = ((~pred) & (y == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0
        if f1 > best[0]:
            best = (f1, t)
    return float(best[1])


def fit_and_save():
    scen = pd.read_csv(SCEN_PATH)
    tr = scen[scen["role"] == "train"]
    va = scen[scen["role"] == "val"]
    idx = {c: i for i, c in enumerate(CLASSES)}
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    model.fit(
        _features(tr["p_audio"], tr["p_threat"], tr["p_abusive"]),
        tr["scenario_label"].map(idx).values,
    )
    assert list(model[-1].classes_) == [0, 1, 2]
    pv = model.predict_proba(_features(va["p_audio"], va["p_threat"], va["p_abusive"]))[:, 2]
    is_high = (va["scenario_label"] == "HIGH").values
    thr = _best_thr(is_high.astype(int), pv)
    agree = 1 - np.abs(va["p_audio"].values - va["p_threat"].values)
    tau = float(np.percentile(agree[is_high], 5))
    joblib.dump({"model": model, "high_thr": thr, "tau": tau}, MODEL_PATH)
    print(f"Trained on {len(tr)} TRAIN-role scenarios (all rotations pooled).")
    print(f"HIGH threshold (chosen on {len(va)} validation scenarios): {thr:.2f}")
    print(f"Agreement cut-off tau (5th percentile of agreement on validation HIGH): {tau:.3f}")
    called = pd.Series(pv >= thr, index=va.index)
    print("\nShare called HIGH on validation scenarios, by cell:")
    print(called.groupby(va["cell"]).mean().round(3).to_string())
    print("\nSaved models/fusion_lr.joblib")


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def fuse(p_audio, p_threat, p_abusive):
    b = _load()
    probs = b["model"].predict_proba(_features(p_audio, p_threat, p_abusive))[0]
    p_low, p_mod, p_high = (float(x) for x in probs)
    agreement = float(1 - abs(p_audio - p_threat))
    low_agreement = agreement < b["tau"]
    if p_high >= b["high_thr"]:
        level = "UNCERTAIN" if low_agreement else "HIGH"
    else:
        level = "MODERATE" if p_mod >= p_low else "LOW"
    return {
        "level": level,
        "p_audio": float(p_audio),
        "p_threat": float(p_threat),
        "p_abusive": float(p_abusive),
        "class_probs": {"LOW": p_low, "MODERATE": p_mod, "HIGH": p_high},
        "high_thr": float(b["high_thr"]),
        "agreement": agreement,
        "agreement_label": "LOW" if low_agreement else "HIGH",
        "tau": float(b["tau"]),
    }


def explain(fused, audio=None, text=None):
    """Build the explanation from the actual model outputs only."""
    pa, pt, pab = fused["p_audio"], fused["p_threat"], fused["p_abusive"]
    a_hi, t_hi, ab_hi = pa >= DISPLAY_CUT, pt >= DISPLAY_CUT, pab >= DISPLAY_CUT
    lines = []

    state = "elevated" if a_hi else "low"
    lines.append(f"Audio evidence: distress-like score {pa:.2f} ({state}).")
    if audio:
        heard = ", ".join(f"{n} ({s})" for n, s in audio["yamnet_top_events"])
        lines.append(f"  YAMNet detected: {heard}. This describes what YAMNet heard; it is not a breakdown of the score.")

    t_parts = []
    if t_hi:
        t_parts.append(f"threat-related language (score {pt:.2f})")
    if ab_hi:
        t_parts.append(f"abusive language (score {pab:.2f})")
    if t_parts:
        lines.append("Text evidence: " + " and ".join(t_parts) + ".")
    else:
        lines.append(f"Text evidence: no strong threat or abuse signal (threat {pt:.2f}, abusive {pab:.2f}).")
    if text:
        for cls, key in [("THREAT", t_hi), ("ABUSIVE", ab_hi)]:
            terms = text.get("top_terms", {}).get(cls, [])
            if key and terms:
                lines.append(f"  Words the model associates with {cls.lower()}: " + ", ".join(t for t, _ in terms[:4]) + ".")

    if a_hi and (t_hi or ab_hi):
        rel = "Both modalities show elevated signals."
    elif a_hi:
        rel = "Only the audio modality shows an elevated signal."
    elif t_hi or ab_hi:
        rel = "Only the text modality shows an elevated signal."
    else:
        rel = "Neither modality shows an elevated signal."

    text_hi = t_hi or ab_hi
    word = "HIGH" if a_hi == text_hi else "LOW"
    lines.append(
        f"Evidence agreement: {word}. {rel} "
        f"(audio-vs-threat score agreement {fused['agreement']:.2f}, where 1.00 = identical; "
        f"extreme-disagreement cut-off {fused['tau']:.2f})."
    )

    lvl, ph = fused["level"], fused["class_probs"]["HIGH"]
    if lvl == "HIGH":
        reason = f"Combined score for high risk ({ph:.2f}) is above the configured threshold ({fused['high_thr']:.2f}) and the two signals agree."
    elif lvl == "UNCERTAIN":
        reason = (f"Combined score for high risk ({ph:.2f}) is above the threshold ({fused['high_thr']:.2f}), "
                  "but the two signals disagree. Needs human review.")
    elif lvl == "MODERATE":
        reason = f"Combined score for high risk ({ph:.2f}) is below the threshold ({fused['high_thr']:.2f}), but the model rates this as more than low risk."
    else:
        reason = f"Combined score for high risk ({ph:.2f}) is below the threshold ({fused['high_thr']:.2f}) and neither modality shows strong evidence."
    lines.append("Final reason: " + reason)
    lines.append("Note: prototype decision-support estimate, not confirmation of any event. Human review is required.")
    return lines


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit_and_save()
    else:
        demos = [
            ("scream + clear threat", 0.82, 0.88, 0.05),
            ("scream + calm text", 0.81, 0.12, 0.10),
            ("calm audio + threat text", 0.05, 0.90, 0.05),
            ("scream + abusive text", 0.85, 0.05, 0.85),
            ("nothing unusual", 0.03, 0.01, 0.02),
            ("extreme disagreement", 0.99, 0.005, 0.01),
            ("borderline both", 0.55, 0.50, 0.10),
        ]
        print("Hand-set score inputs: a demo of the decision rule, NOT evidence of performance.")
        for name, pa, pt, pab in demos:
            f = fuse(pa, pt, pab)
            print(f"\n=== {name}: audio={pa} threat={pt} abusive={pab} -> {f['level']} ===")
            for line in explain(f):
                print(line)