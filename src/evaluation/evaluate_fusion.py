import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCEN = os.path.join(ROOT, "data", "scenarios")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)

SEED = 42
B = 1000
EPS = 1e-4
CLASSES = ["LOW", "MODERATE", "HIGH"]
IDX = {c: i for i, c in enumerate(CLASSES)}
METHODS = ["audio_only", "text_only", "fusion", "mean_rule", "min_rule"]
FUSION_FEATS = ["logit_p_audio", "logit_p_threat", "logit_p_abusive", "abs_diff", "p_audio*p_threat"]

out_lines = []


def P(s=""):
    print(s, flush=True)
    out_lines.append(str(s))


scen = pd.read_csv(os.path.join(SCEN, "scenarios.csv"), dtype={"text_id": str})


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def feats(df, kind):
    pa, pt, pab = df["p_audio"].values, df["p_threat"].values, df["p_abusive"].values
    if kind == "audio_only":
        return np.c_[logit(pa)]
    if kind == "text_only":
        return np.c_[logit(pt), logit(pab)]
    return np.c_[logit(pa), logit(pt), logit(pab), np.abs(pa - pt), pa * pt]


def best_thr(y, s):
    best = (-1.0, 0.5)
    for t in np.arange(0.02, 0.981, 0.01):
        pred = s >= t
        tp = (pred & (y == 1)).sum()
        fp = (pred & (y == 0)).sum()
        fn = ((~pred) & (y == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0
        if f1 > best[0]:
            best = (f1, t)
    return best[1]


# ---------------- Train, choose thresholds on VAL, score TEST ----------------
test_parts, thr_rows, coefs = [], [], []
for k in range(5):
    tr = scen[(scen.rotation == k) & (scen.role == "train")].reset_index(drop=True)
    va = scen[(scen.rotation == k) & (scen.role == "val")].reset_index(drop=True)
    te = scen[(scen.rotation == k) & (scen.role == "test")].reset_index(drop=True)
    ytr = tr["scenario_label"].map(IDX).values
    for kind in ["audio_only", "text_only", "fusion"]:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        m.fit(feats(tr, kind), ytr)
        assert list(m[-1].classes_) == [0, 1, 2]
        va[f"s_{kind}"] = m.predict_proba(feats(va, kind))[:, 2]
        pt = m.predict_proba(feats(te, kind))
        te[f"s_{kind}"] = pt[:, 2]
        te[f"argmax_{kind}"] = pt.argmax(axis=1)
        if kind == "fusion":
            coefs.append(m[-1].coef_[2])
    for d in (va, te):
        d["s_mean_rule"] = (d["p_audio"] + d["p_threat"]) / 2
        d["s_min_rule"] = np.minimum(d["p_audio"], d["p_threat"])
    yv = (va["scenario_label"] == "HIGH").astype(int).values
    for name in METHODS:
        thr = best_thr(yv, va[f"s_{name}"].values)
        te[f"pred_{name}"] = (te[f"s_{name}"] >= thr).astype(int)
        thr_rows.append({"rotation": k, "method": name, "threshold": round(float(thr), 2)})
    agree_val = 1 - np.abs(va["p_audio"] - va["p_threat"])
    tau = np.percentile(agree_val[va["scenario_label"] == "HIGH"], 5)
    te["agreement"] = 1 - np.abs(te["p_audio"] - te["p_threat"])
    te["tau"] = tau
    test_parts.append(te)

test = pd.concat(test_parts, ignore_index=True)
test["y_high"] = (test["scenario_label"] == "HIGH").astype(int)
y = test["y_high"].values

P("FUSION EVALUATION (protocol: docs/fusion_protocol.md)")
P(f"TEST scenarios pooled over 5 rotations: {len(test)} | HIGH: {int(y.sum())}")
P("HIGH thresholds chosen on VALIDATION only:")
P(pd.DataFrame(thr_rows).pivot(index="rotation", columns="method", values="threshold")[METHODS].to_string())


# ---------------- Metrics ----------------
def hi_metrics(y, s, pred, w):
    m = w > 0
    y, s, pred, w = y[m], s[m], pred[m], w[m]
    tp = w[(pred == 1) & (y == 1)].sum()
    fp = w[(pred == 1) & (y == 0)].sum()
    fn = w[(pred == 0) & (y == 1)].sum()
    tn = w[(pred == 0) & (y == 0)].sum()
    rec = tp / (tp + fn) if tp + fn > 0 else np.nan
    prec = tp / (tp + fp) if tp + fp > 0 else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if fp + tn > 0 else np.nan
    return {
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "fpr": fpr,
        "fnr": 1 - rec,
        "roc": roc_auc_score(y, s, sample_weight=w),
        "pr": average_precision_score(y, s, sample_weight=w),
    }


ones = np.ones(len(test))
point = {n: hi_metrics(y, test[f"s_{n}"].values, test[f"pred_{n}"].values, ones) for n in METHODS}

# Two-way source bootstrap (audio clips x text comments), thresholds fixed
groups = []
for k in range(5):
    idx = np.where(test["rotation"].values == k)[0]
    ua, ai = np.unique(test["audio_clip_id"].values[idx], return_inverse=True)
    ut, ti = np.unique(test["text_id"].values[idx], return_inverse=True)
    groups.append((idx, ai, len(ua), ti, len(ut)))

rng = np.random.default_rng(SEED)
boot = {n: [] for n in METHODS}
for b in range(B):
    w = np.zeros(len(test))
    for idx, ai, na, ti, nt in groups:
        ca = np.bincount(rng.integers(0, na, na), minlength=na)
        ct = np.bincount(rng.integers(0, nt, nt), minlength=nt)
        w[idx] = ca[ai] * ct[ti]
    for n in METHODS:
        boot[n].append(hi_metrics(y, test[f"s_{n}"].values, test[f"pred_{n}"].values, w))
    if (b + 1) % 200 == 0:
        print(f"  bootstrap {b + 1}/{B}", flush=True)
bdf = {n: pd.DataFrame(boot[n]) for n in METHODS}


def ci(a):
    return np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)


P("\nPRIMARY TASK: HIGH vs not-HIGH, pooled TEST, point estimate [95% source-bootstrap CI]")
P("(CI reflects uncertainty from the small number of unique source clips/comments)")
for mname in ["recall", "precision", "f1", "fpr", "fnr", "roc", "pr"]:
    P(f"\n{mname.upper()}")
    for n in METHODS:
        lo, hi = ci(bdf[n][mname].values)
        P(f"  {n:<11} {point[n][mname]:.3f}  [{lo:.3f}, {hi:.3f}]")

P("\nKEY COMPARISONS: fusion minus other method (paired bootstrap)")
for mname in ["f1", "pr"]:
    for other in ["audio_only", "text_only", "mean_rule", "min_rule"]:
        d = bdf["fusion"][mname].values - bdf[other][mname].values
        lo, hi = ci(d)
        P(
            f"  {mname.upper():<3} fusion - {other:<10}: point {point['fusion'][mname] - point[other][mname]:+.3f}"
            f"  CI [{lo:+.3f}, {hi:+.3f}]  share of draws > 0: {np.nanmean(d > 0):.2f}"
        )

# Per-rotation spread
P("\nPER-ROTATION HIGH-class F1 (each rotation: own thresholds, test fold only)")
rows = []
for k in range(5):
    t = test[test["rotation"] == k]
    r = {"rotation": k}
    for n in METHODS:
        r[n] = hi_metrics(t["y_high"].values, t[f"s_{n}"].values, t[f"pred_{n}"].values, np.ones(len(t)))["f1"]
    rows.append(r)
pr_df = pd.DataFrame(rows)
P(pr_df.round(3).to_string(index=False))
P("mean +/- std: " + ", ".join(f"{n}={pr_df[n].mean():.3f}+/-{pr_df[n].std():.3f}" for n in METHODS))

# Per-cell behaviour
P("\nSHARE OF SCENARIOS CALLED HIGH, by scenario cell (pooled TEST, point estimates)")
cell = test.groupby(["cell", "scenario_label"])[[f"pred_{n}" for n in METHODS]].mean().round(3)
cell.columns = METHODS
P(cell.to_string())

# 3-class (argmax) secondary result
P("\nSECONDARY: 3-class macro-F1 (LOW/MODERATE/HIGH, argmax), pooled TEST")
ytrue3 = test["scenario_label"].map(IDX).values
for n in ["audio_only", "text_only", "fusion"]:
    P(f"  {n:<11} macro-F1 = {f1_score(ytrue3, test[f'argmax_{n}'].values, average='macro'):.3f}")
P("Fusion confusion matrix (rows true, cols predicted; LOW, MODERATE, HIGH):")
P(str(confusion_matrix(ytrue3, test["argmax_fusion"].values, labels=[0, 1, 2])))

# UNCERTAIN analysis (descriptive)
P("\nUNCERTAIN ANALYSIS (fusion HIGH calls with agreement below tau shown as UNCERTAIN)")
fh = test[test["pred_fusion"] == 1].copy()
fh["uncertain"] = fh["agreement"] < fh["tau"]
P(f"Fusion HIGH calls: {len(fh)} | shown as UNCERTAIN: {int(fh['uncertain'].sum())}")
P("Share of fusion HIGH calls shown as UNCERTAIN, by true cell:")
P(fh.groupby("cell")["uncertain"].agg(["mean", "size"]).round(3).to_string())
conf = fh[~fh["uncertain"]]
P(
    f"Precision of all fusion HIGH calls: {(fh['y_high'] == 1).mean():.3f} | "
    f"precision of confident HIGH only: {(conf['y_high'] == 1).mean():.3f} (n={len(conf)})"
)
P(
    f"HIGH recall, all fusion HIGH calls: {(fh['y_high'] == 1).sum() / y.sum():.3f} | "
    f"confident HIGH only: {(conf['y_high'] == 1).sum() / y.sum():.3f}"
)

# Fusion model weights (standardised features, HIGH class, mean over rotations)
P("\nFusion model HIGH-class coefficients (standardised features, mean over 5 rotations):")
for name, c in zip(FUSION_FEATS, np.mean(coefs, axis=0)):
    P(f"  {name:<18} {c:+.3f}")

# Save
with open(os.path.join(RESULTS, "fusion_metrics.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines) + "\n")
test.to_csv(os.path.join(RESULTS, "fusion_test_predictions.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, mname, title in [(axes[0], "f1", "HIGH-class F1"), (axes[1], "pr", "HIGH-class PR-AUC")]:
    vals = [point[n][mname] for n in METHODS]
    errs = np.array([[point[n][mname] - ci(bdf[n][mname].values)[0], ci(bdf[n][mname].values)[1] - point[n][mname]] for n in METHODS]).T
    ax.bar(range(len(METHODS)), vals, yerr=errs, capsize=3)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=30, ha="right")
    ax.set_title(title + " (95% source-bootstrap CI)")
    ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fusion_comparison.png"), dpi=150)
P("\nSaved: results/fusion_metrics.txt, results/fusion_test_predictions.csv, results/figures/fusion_comparison.png")