import os
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd
import soundfile as sf
import streamlit as st

import pipeline

st.set_page_config(page_title="Chetna", layout="wide")

MAX_SECONDS = 30  # design choice to protect a small server; not a validated limit

DEMO_TEXTS = {
    "Benign: class note": "Thanks for sharing the notes, see you in class tomorrow.",
    "Abusive: insult": "You are such an idiot and nobody likes you.",
    "Threat: direct": "I know where you live and you will regret this.",
    "Idiom containing 'kill' (harmless)": "I am going to kill this presentation tomorrow.",
}
LEVEL_UI = {
    "HIGH": ("error", "HIGH RISK (potential)"),
    "UNCERTAIN": ("warning", "UNCERTAIN: the signals disagree, needs human review"),
    "MODERATE": ("warning", "MODERATE RISK (potential)"),
    "LOW": ("success", "LOW RISK"),
}


@st.cache_data
def demo_clips():
    """Dataset demo clips. Only offered if the manifest and the audio files exist on this machine."""
    mpath = os.path.join(ROOT, "data", "processed", "audio_manifest.csv")
    if not os.path.exists(mpath):
        return []
    man = pd.read_csv(mpath)
    picks = []
    scr = man[man["binary_label"] == 1]
    for i in range(min(2, len(scr))):
        picks.append((f"Scream clip {i + 1} (Deeply dataset)", scr.iloc[i]["path"]))
    for lab, nice in [
        ("laughing", "Laughing (ESC-50 dataset)"),
        ("dog", "Dog bark (ESC-50 dataset)"),
        ("teeth-chattering", "Teeth chattering (Deeply dataset)"),
    ]:
        row = man[man["original_label"] == lab]
        if len(row):
            picks.append((nice, row.iloc[0]["path"]))
    return [(name, path) for name, path in picks if os.path.exists(path)]


def show_result(r):
    f = r["fused"]
    kind, label = LEVEL_UI[r["level"]]
    getattr(st, kind)(f"**Potential risk level: {label}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Audio distress-like score", f"{f['p_audio']:.2f}")
    c2.metric("Text threat score", f"{f['p_threat']:.2f}")
    c3.metric("Text abusive score", f"{f['p_abusive']:.2f}")
    ph = f["class_probs"]["HIGH"]
    c4.metric("Combined high-risk score", f"{ph:.2f}")
    st.caption(
        f"HIGH is called when the combined score is at or above {f['high_thr']:.2f} "
        "(threshold chosen on validation scenarios)."
    )
    if abs(ph - f["high_thr"]) <= 0.1:
        st.info(
            "The combined score is close to the threshold, so small changes in the input "
            "could change the level."
        )

    dc = r["dataset_clip"]
    if dc:
        st.info(
            f"This clip comes from the {dc['source']} dataset ({dc['original_label']}). The final audio "
            f"model was trained on it, so the live audio score ({f['p_audio']:.2f}) is optimistic. "
            f"Its cross-validated held-out score was {dc['heldout_p_audio']:.2f}."
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Audio evidence")
        st.progress(min(max(f["p_audio"], 0.0), 1.0), text=f"Distress-like score {f['p_audio']:.2f}")
        a = r["audio"]
        heard = ", ".join(f"{n} ({s})" for n, s in a["yamnet_top_events"])
        st.caption(f"What YAMNet detected (not a breakdown of the score): {heard}")
        st.caption(
            f"{a['n_active']} of {a['n_frames']} audio frames used; clip length {a['duration_s']} s."
        )
    with right:
        st.subheader("Text evidence")
        probs = r["text"]["probs"]
        for cls in ["BENIGN", "ABUSIVE", "THREAT"]:
            st.progress(min(max(probs[cls], 0.0), 1.0), text=f"{cls.title()} {probs[cls]:.2f}")
        for cls in ["THREAT", "ABUSIVE"]:
            terms = r["text"]["top_terms"].get(cls, [])
            if probs[cls] >= 0.5 and terms:
                st.caption(
                    f"Words the model associates with {cls.lower()}: " + ", ".join(t for t, _ in terms[:4])
                )

    st.subheader("Why this assessment")
    md = []
    for line in r["explanation"]:
        md.append(("    - " + line.strip()) if line.startswith("  ") else ("- " + line))
    st.markdown("\n".join(md))

    st.subheader("Alert (simulation)")
    if r["level"] in ("HIGH", "UNCERTAIN"):
        if st.button("Simulate sending an alert to a reviewer"):
            st.session_state["alert"] = True
        if st.session_state.get("alert"):
            st.error(
                "**SIMULATED ALERT. Nothing was sent to anyone.**  \n"
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n"
                f"Level: {r['level']}  \n"
                f"Audio score {f['p_audio']:.2f}, threat score {f['p_threat']:.2f}, "
                f"combined high-risk score {ph:.2f}.  \n"
                "A human reviewer would check the evidence above and decide what to do."
            )
    else:
        st.caption("No alert is simulated at this level.")

    tm = r["timing_s"]
    st.caption(
        f"Processing time on this computer: {tm['total']:.2f} s "
        f"(audio {tm['audio']:.2f} s, text {tm['text']:.2f} s). The first run is slower because models load."
    )


# ------------------------------------------------------------------ page
st.title("Chetna")
st.caption("Explainable multimodal safety-risk assessment and early-warning prototype (audio + text)")
st.warning(
    "PROTOTYPE for research and demonstration. It estimates potential risk from audio and text scores and "
    "was evaluated only on synthetic pairings of public datasets. It does not detect real violence, does not "
    "contact emergency services, and cannot replace human judgement."
)

with st.sidebar:
    st.header("About")
    st.write(
        "Audio: YAMNet embeddings + logistic regression. Text: TF-IDF + logistic regression. "
        "Fusion: logistic regression on both scores."
    )
    st.header("Privacy")
    st.write(
        "The analysis runs on the machine that hosts this app (your own computer if you run it locally). "
        "Uploaded audio is written to a temporary file and deleted right after the analysis. "
        "This app does not save your audio or text. "
        "Please do not upload real emergency recordings or personal conversations."
    )

tab1, tab2 = st.tabs(["Analyze", "Evaluation results"])

with tab1:
    col_a, col_t = st.columns(2)
    audio_path, audio_name, uploaded = None, None, None
    clips = demo_clips()

    with col_a:
        st.subheader("1. Audio")
        modes = ["Upload a file"] + (["Use a dataset demo clip"] if clips else [])
        mode = st.radio("Audio source", modes, horizontal=True)
        if mode == "Upload a file":
            uploaded = st.file_uploader("Audio file (WAV works best)", type=["wav", "flac", "ogg", "mp3"])
            st.caption(f"Maximum {MAX_SECONDS} seconds and 10 MB.")
            if not clips:
                st.caption("Dataset demo clips are not included in this version. Please upload a short audio file.")
            if uploaded is not None:
                st.audio(uploaded)
                audio_name = uploaded.name
        else:
            choice = st.selectbox("Demo clip", [c[0] for c in clips])
            audio_path = dict(clips)[choice]
            audio_name = os.path.basename(audio_path)
            st.audio(audio_path)

    with col_t:
        st.subheader("2. Text")
        demo = st.selectbox(
            "Demo text (hand-written, not from the dataset), or write your own",
            ["(write my own)"] + list(DEMO_TEXTS),
        )
        text = st.text_area("Text to analyse", value=DEMO_TEXTS.get(demo, ""), height=130, max_chars=2000, key=f"text_{demo}")

    st.subheader("3. Analyze")
    if st.button("Analyze", type="primary"):
        tmp_path = None
        try:
            if mode == "Upload a file":
                if uploaded is None:
                    raise ValueError("Please upload an audio file or choose a demo clip.")
                data = uploaded.getvalue()
                if len(data) > 10 * 1024 * 1024:
                    raise ValueError("The audio file is larger than 10 MB. Please use a shorter clip.")
                suffix = os.path.splitext(uploaded.name)[1] or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                try:
                    dur = sf.info(tmp_path).duration
                except Exception:
                    dur = None  # header not readable here; the audio loader will try next
                if dur is not None and dur > MAX_SECONDS:
                    raise ValueError(
                        f"The audio is {dur:.0f} s long. Please use a clip of {MAX_SECONDS} s or less."
                    )
                audio_path = tmp_path
            with st.spinner("Running audio model, text model and fusion..."):
                st.session_state["result"] = pipeline.analyze(audio_path, text, display_name=audio_name)
            st.session_state["alert"] = False
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Could not analyse this input: {e} (If the audio failed to load, try a WAV file.)")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    if "result" in st.session_state:
        show_result(st.session_state["result"])

with tab2:
    st.write(
        "These files were produced by the evaluation scripts in this project. "
        "Results are on synthetic pairings of public datasets, not on real incidents."
    )
    fig = os.path.join(ROOT, "results", "figures", "fusion_comparison.png")
    if os.path.exists(fig):
        st.image(fig, caption="HIGH-class F1 and PR-AUC by method (95% source-bootstrap intervals)")
    for fname, label in [
        ("fusion_metrics.txt", "Fusion evaluation (full output)"),
        ("audio_metrics.txt", "Audio branch evaluation"),
        ("text_baseline_metrics.txt", "Text branch evaluation"),
    ]:
        fp = os.path.join(ROOT, "results", fname)
        if os.path.exists(fp):
            with st.expander(label):
                with open(fp, encoding="utf-8") as fh:
                    st.code(fh.read())
    st.subheader("Limitations")
    st.markdown(
        "- Scenarios are synthetic pairings, balanced across cases; precision and false-alarm rates do not "
        "transfer to real-world prevalence.\n"
        "- Each test set used only 8 to 13 unique scream clips and 14 to 15 unique threat comments.\n"
        "- The learned fusion did not clearly beat a simple minimum or mean rule.\n"
        "- Scores near the threshold are sensitive to small input changes.\n"
        "- Not a substitute for emergency services or human judgement."
    )