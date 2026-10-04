import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("audio", "text", "fusion"):
    sys.path.insert(0, os.path.join(ROOT, "src", _sub))

import pandas as pd

import fuse
import predict_audio
import predict_text

PROC = os.path.join(ROOT, "data", "processed")
MAX_CHARS = 5000
_ref = None


def _reference():
    """Dataset clips by file name, with their cross-validated (held-out) audio score."""
    global _ref
    if _ref is None:
        man = pd.read_csv(os.path.join(PROC, "audio_manifest.csv"))
        oof = pd.read_csv(os.path.join(PROC, "audio_oof_probs.csv"))[["clip_id", "p_audio"]]
        df = man[["clip_id", "path", "source", "original_label", "binary_label"]].merge(oof, on="clip_id")
        df["name"] = df["path"].map(os.path.basename)
        _ref = df.drop_duplicates("name").set_index("name")
    return _ref


def analyze(audio_path, text, display_name=None):
    text = (text or "").strip()
    if not text:
        raise ValueError("Please enter some text to analyse.")
    text = text[:MAX_CHARS]

    t0 = time.perf_counter()
    audio = predict_audio.predict_audio(audio_path)
    t1 = time.perf_counter()
    txt = predict_text.predict_text(text)
    t2 = time.perf_counter()
    fused = fuse.fuse(audio["p_audio"], txt["probs"]["THREAT"], txt["probs"]["ABUSIVE"])
    t3 = time.perf_counter()
    lines = fuse.explain(fused, audio=audio, text=txt)

    name = display_name or os.path.basename(audio_path)
    ref = _reference()
    dataset_clip = None
    if name in ref.index:
        r = ref.loc[name]
        dataset_clip = {
            "source": str(r["source"]),
            "original_label": str(r["original_label"]),
            "heldout_p_audio": float(r["p_audio"]),
        }

    return {
        "level": fused["level"],
        "fused": fused,
        "audio": audio,
        "text": txt,
        "explanation": lines,
        "dataset_clip": dataset_clip,
        "timing_s": {
            "audio": t1 - t0,
            "text": t2 - t1,
            "fusion": t3 - t2,
            "total": t3 - t0,
        },
    }


if __name__ == "__main__":
    man = pd.read_csv(os.path.join(PROC, "audio_manifest.csv"))
    scream = man[man["binary_label"] == 1].iloc[0]["path"]
    dog = man[man["original_label"] == "dog"].iloc[0]["path"]

    texts = {
        "benign": "Thanks for sharing the notes, see you in class tomorrow.",
        "threat": "I know where you live and you will regret this.",
        "idiom": "I am going to kill this presentation tomorrow.",
    }
    combos = [
        ("scream clip + threat text", scream, "threat"),
        ("scream clip + benign text", scream, "benign"),
        ("dog clip + threat text", dog, "threat"),
        ("dog clip + idiom text", dog, "idiom"),
        ("dog clip + benign text", dog, "benign"),
    ]
    print("Hand-written demo texts and dataset clips: a code test, NOT evidence of performance.\n")
    for k, (name, path, tkey) in enumerate(combos):
        r = analyze(path, texts[tkey])
        f = r["fused"]
        dc = r["dataset_clip"]
        print(f"=== {name} -> {r['level']} ===")
        print(
            f"  p_audio={f['p_audio']:.3f}  p_threat={f['p_threat']:.3f}  p_abusive={f['p_abusive']:.3f}"
            f"  P_HIGH={f['class_probs']['HIGH']:.3f}"
        )
        if dc:
            print(
                f"  dataset clip ({dc['source']}/{dc['original_label']}): "
                f"held-out audio score {dc['heldout_p_audio']:.3f} vs live {f['p_audio']:.3f}"
            )
        tm = r["timing_s"]
        print(
            f"  time: audio {tm['audio']:.2f}s, text {tm['text']:.2f}s, "
            f"fusion {tm['fusion']:.3f}s, total {tm['total']:.2f}s"
            + ("  (first call includes model loading)" if k == 0 else "")
        )