import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(ROOT, "data", "processed")
SCEN = os.path.join(ROOT, "data", "scenarios")
os.makedirs(SCEN, exist_ok=True)

SEED = 42
N = 5
LABEL = {
    ("DISTRESS", "THREAT"): "HIGH",
    ("DISTRESS", "ABUSIVE"): "MODERATE",
    ("DISTRESS", "BENIGN"): "MODERATE",
    ("NON-DISTRESS", "THREAT"): "MODERATE",
    ("NON-DISTRESS", "ABUSIVE"): "LOW",
    ("NON-DISTRESS", "BENIGN"): "LOW",
}

audio = pd.read_csv(os.path.join(PROC, "audio_oof_probs.csv"))
tval = pd.read_csv(os.path.join(PROC, "text_val_probs.csv"), dtype={"id": str})
ttest = pd.read_csv(os.path.join(PROC, "text_test_probs.csv"), dtype={"id": str})
assert not (set(tval["id"]) & set(ttest["id"])), "text val/test overlap"


def add_parts(df):
    df = df.reset_index(drop=True).copy()
    df["part"] = -1
    skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=SEED)
    for p, (_, idx) in enumerate(skf.split(df, df["text_class"])):
        df.loc[idx, "part"] = p
    assert (df["part"] >= 0).all()
    return df


tval = add_parts(tval)
ttest = add_parts(ttest)


def roles(k):
    val_f = (k + 1) % N
    train_f = [f for f in range(N) if f not in (k, val_f)]
    return [
        ("test", [k], ttest, [k], 300),
        ("val", [val_f], tval, [val_f], 300),
        ("train", train_f, tval, train_f, 600),
    ]


rows = []
summary = []
for k in range(N):
    used_audio, used_text = {}, {}
    for ri, (role, a_folds, tdf, t_parts, n) in enumerate(roles(k)):
        rng = np.random.default_rng(SEED * 1000 + k * 10 + ri)
        a_pool = audio[audio["fold"].isin(a_folds)]
        t_pool = tdf[tdf["part"].isin(t_parts)]
        parts = []
        for a_lab, a_cls in [(1, "DISTRESS"), (0, "NON-DISTRESS")]:
            ap = a_pool[a_pool["binary_label"] == a_lab].reset_index(drop=True)
            for t_cls in ["BENIGN", "ABUSIVE", "THREAT"]:
                tp = t_pool[t_pool["text_class"] == t_cls].reset_index(drop=True)
                a = ap.iloc[rng.integers(0, len(ap), n)].reset_index(drop=True)
                t = tp.iloc[rng.integers(0, len(tp), n)].reset_index(drop=True)
                parts.append(
                    pd.DataFrame(
                        {
                            "rotation": k,
                            "role": role,
                            "cell": f"{a_cls}+{t_cls}",
                            "scenario_label": LABEL[(a_cls, t_cls)],
                            "audio_clip_id": a["clip_id"],
                            "audio_class": a_cls,
                            "audio_source": a["source"],
                            "audio_original": a["original_label"],
                            "text_id": t["id"],
                            "text_class": t_cls,
                            "p_audio": a["p_audio"],
                            "p_benign": t["p_benign"],
                            "p_abusive": t["p_abusive"],
                            "p_threat": t["p_threat"],
                        }
                    )
                )
        sc = pd.concat(parts, ignore_index=True)
        rows.append(sc)
        used_audio[role] = set(sc["audio_clip_id"])
        used_text[role] = set(sc["text_id"])
        summary.append(
            {
                "rot": k,
                "role": role,
                "scenarios": len(sc),
                "uniq_audio": sc["audio_clip_id"].nunique(),
                "uniq_scream_clips": sc.loc[sc["audio_class"] == "DISTRESS", "audio_clip_id"].nunique(),
                "uniq_text": sc["text_id"].nunique(),
                "uniq_threat_text": sc.loc[sc["text_class"] == "THREAT", "text_id"].nunique(),
            }
        )
    # No source may appear in two roles inside one rotation
    roles_list = list(used_audio)
    for i in range(len(roles_list)):
        for j in range(i + 1, len(roles_list)):
            a, b = roles_list[i], roles_list[j]
            assert not (used_audio[a] & used_audio[b]), f"audio overlap {a}/{b} rot {k}"
            assert not (used_text[a] & used_text[b]), f"text overlap {a}/{b} rot {k}"

scen = pd.concat(rows, ignore_index=True)
scen.to_csv(os.path.join(SCEN, "scenarios.csv"), index=False)

print("Total scenarios:", len(scen))
print("\nUnique sources actually used per rotation and role:")
print(pd.DataFrame(summary).to_string(index=False))
print("\nScenario label counts, TEST role, rotation 0:")
print(scen[(scen["rotation"] == 0) & (scen["role"] == "test")]["scenario_label"].value_counts().to_string())
print("\nNON-DISTRESS audio source mix in TEST role (all rotations):")
print(scen[(scen["role"] == "test") & (scen["audio_class"] == "NON-DISTRESS")]["audio_source"].value_counts().to_string())
print("\nNo source overlap between roles inside any rotation: checks passed.")
print("Saved data/scenarios/scenarios.csv")