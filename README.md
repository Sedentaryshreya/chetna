Chetna: Context-Aware Multimodal Safety-Risk Assessment (Audio + Text Evidence Fusion)

An explainable multimodal safety-risk assessment and early-warning prototype that
combines audio and textual evidence to estimate potential safety risk.

Author: Shreya Gautam, B.Tech CSE (AI & ML). Submission for IBM BharatCares, October 2026.

**Live demo:** <https://chetna-5runma5w5u996nec3yrppq.streamlit.app/> (free hosting: if the app is
asleep, click the wake-up button and wait a minute or two).

**Core question:** does combining independent audio and text evidence give a more
context-aware risk assessment than either alone? The output states are
LOW / MODERATE / HIGH / UNCERTAIN.

## What this is, and what it is not

Chetna is a research prototype for decision support and human review.

It is NOT an SOS app, a GPS tracker, a chatbot or a production emergency system.
It does not guarantee safety, does not detect violence, does not replace emergency
services, and is not 100% accurate. It was evaluated only on synthetic pairings of
public datasets, never on real emergencies or real conversations.

## How it works

1. **Audio branch:** YAMNet (frozen, pretrained) embeddings pooled over active frames,
   then logistic regression, giving a distress score `p_audio`.
2. **Text branch:** TF-IDF (1-2 grams) plus logistic regression, giving probabilities
   for BENIGN, ABUSIVE and THREAT.
3. **Fusion:** logistic regression on the audio and text scores plus their
   disagreement and product, giving LOW / MODERATE / HIGH probabilities.
4. **UNCERTAIN:** a HIGH call is shown as UNCERTAIN when the audio and threat scores
   disagree strongly (human review needed).
5. **Streamlit app:** shows the risk level, scores, evidence, an explanation and a
   simulated alert. Nothing is ever sent anywhere.

## Quick start (Windows, PowerShell)

Python 3.13.7 was used. Run each command on its own from the project folder.

```powershell
python -m venv .venv
```
```powershell
.venv\Scripts\Activate.ps1
```
```powershell
pip install -r requirements.txt
```
```powershell
python -m streamlit run app.py --server.address localhost
```

The prompt should start with `(.venv)`. The `--server.address localhost` flag keeps the
app private to your computer. It is deliberately not set in `.streamlit\config.toml`, so
that a hosting platform can serve the app.

- **YAMNet:** the YAMNet files (about 17 MB) are included in `models\tfhub_cache\`, so the
  app should not need to download them. If that folder is missing, TensorFlow Hub
  downloads YAMNet on first run and needs internet. Loading from the included cache on a
  fresh machine was not separately verified.
- **Audio input:** use WAV files. m4a is not readable without ffmpeg.
- **Dataset demo clips** need the datasets downloaded (see below) and
  `data\processed\audio_manifest.csv`, which is not in this repository because it stores
  absolute paths from the author's machine. Without them the demo-clip option is hidden
  and you upload a WAV instead.
- **Upload limits:** 30 seconds and 10 MB per clip. These are design choices to protect a
  small server, not validated limits.

## Live demo and deployment

The public demo runs on Streamlit Community Cloud from the `deploy-prep` branch with
Python 3.13. `requirements.txt` lists only the packages the app needs, pinned to the
versions the models were saved with. Held-out scores for known dataset clips come from
`data\deploy\audio_reference.csv` (no local paths), built by
`src\make_deploy_reference.py`. The analysis runs on CPU; the first analysis took about
5 seconds in one test. Free hosting sleeps when idle.

## Run the tests

```powershell
python -m pytest tests -v
```

22 tests (fusion logic, end-to-end pipeline on a synthetic WAV, audio edge cases).
They check structure and rules, not accuracy. Performance claims come only from
`src\evaluation\evaluate_fusion.py`.

`tests\test_env.py` only prints library versions and needs matplotlib, which the app does
not use. In an environment built only from `requirements.txt`, run
`python -m pytest tests -v --ignore=tests/test_env.py`. The evaluation script also needs
matplotlib (`pip install matplotlib`).

## Data and licences

No dataset is redistributed here. Download them from the original sources
(details in `docs\dataset_log.md`) into `data\raw\`.

| Dataset | Use | Licence |
|---|---|---|
| ESC-50 (2,000 clips) | non-distress audio | CC BY-NC 3.0 |
| Deeply Nonverbal Vocalization, OpenSLR 99 public subset (727 clips) | screams and other non-scream audio | CC BY-NC-ND 4.0 |
| Jigsaw Toxic Comment (159,571 comments) | text classes | CC0 dataset; comment text under Wikipedia CC-SA 3.0 |

No public dataset has audio, text and a risk label together, so evaluation scenarios
are synthetic pairings under a frozen rule (`docs\scenario_rule.md`). The text model
file `models\text_tfidf_lr.joblib` contains a vocabulary learned from Jigsaw
comments; this is an acknowledgement, not a legal claim about that licence.

## Results (synthetic scenarios, pooled TEST, 5 rotations)

HIGH vs not-HIGH, 9,000 scenarios (1,500 HIGH). F1 with 95% source-bootstrap interval:

| Method | F1 [95% CI] | False-positive rate | False-negative rate | PR-AUC |
|---|---|---|---|---|
| Audio only | 0.478 [0.404, 0.545] | 0.350 | 0.136 | 0.333 |
| Text only | 0.546 [0.467, 0.609] | 0.215 | 0.221 | 0.474 |
| Fusion (learned) | 0.748 [0.671, 0.808] | 0.048 | 0.260 | 0.859 |
| Mean rule | 0.765 [0.677, 0.836] | 0.017 | 0.329 | 0.815 |
| Min rule | 0.755 [0.662, 0.832] | 0.022 | 0.327 | 0.873 |

Branch results: audio branch (out-of-fold, 51 scream clips) PR-AUC 0.837, ROC-AUC 0.995.
Text branch: macro-F1 0.739; THREAT F1 0.478 on only 71 test positives.

What these results support:
- Combining modalities sharply cut false alarms compared with either modality alone
  (false-positive rate 0.350 and 0.215 down to 0.048), at the cost of more misses
  (false-negative rate 0.260 vs 0.136 for audio only; the intervals overlap).
- Learned fusion did NOT clearly beat simple mean or min rules (F1 differences
  include zero).

What they do not support:
- Any real-world claim. Scenarios are balanced and synthetic, so precision and
  false-alarm rates do not transfer.
- That fusion is a "discovery" over single modalities: the label requires both
  modalities, so the advantage is partly built in by design.

Full details: `results\fusion_metrics.txt`, `docs\phase4_summary.md`.

## Known limitations

- Only 51 scream clips (phone recordings, not real emergencies). Each test set uses
  only 8-13 unique screams and 14-15 unique threat comments, so intervals are wide.
- The text model is keyword-driven: "I am going to kill this presentation tomorrow"
  scored THREAT 0.994. In a manual demo a dog clip with a "kill" idiom produced HIGH,
  while a dog clip with a real threat produced MODERATE. Documented, not retuned.
- The HIGH threshold was tuned on a balanced scenario mix; real-world false alarms
  would be higher, and scores near the threshold are sensitive.
- UNCERTAIN is narrow: it flags strong audio/threat disagreement for human review.
  It is not a false-alarm filter (31% of UNCERTAIN calls were true HIGH scenarios).
- 82% of fusion's false HIGH calls were distress-audio plus abusive-text scenarios,
  which the frozen rule labels MODERATE; this is partly a limit of the labelling rule.
- Audio scores are not calibrated probabilities. YAMNet's top labels shown in the app
  do not explain the classifier's score.
- English only. No temporal modelling, speech-to-text or real-time streaming.

## Ethics and privacy

Only public datasets and synthetic pairings were used. No real emergency recordings
or personal conversations were collected. The app writes uploaded audio to a
temporary file that is deleted after analysis (deletion was not separately verified)
and does not save audio or text. Run locally, nothing leaves your computer. On the public
demo the analysis runs on the hosting server (Streamlit Community Cloud), so please do
not upload real emergency recordings or personal conversations.
Any real deployment would need informed opt-in consent, event-triggered (not
continuous) audio, and human review of every alert.

## SDG alignment

Primary: SDG 5, target 5.2. Secondary: SDG 16, target 16.1. Technology alignment:
SDG 9. This prototype is aligned with these goals; it does not solve any target and
does not prevent violence.

## Project layout

```
app.py                  Streamlit interface
src/pipeline.py         analyze(): audio + text + fusion + explanation
src/make_deploy_reference.py  builds data/deploy/audio_reference.csv (no local paths)
src/audio/              YAMNet features, audio model
src/text/               text data preparation and model
src/fusion/             scenario building and fusion model
src/evaluation/         evaluate_fusion.py (the only source of performance claims)
tests/                  22 tests
models/                 trained models and the YAMNet cache
data/processed/         scores (no audio, no comment text); large files and the
                        path-bearing manifest are git-ignored
data/deploy/            audio_reference.csv (held-out scores by clip name)
data/scenarios/         synthetic scenario table
results/                metrics and figures
docs/                   dataset log, frozen rules and protocol, phase summaries
```

## Re-creating models and results from scratch (not re-run end to end for submission)

After downloading the datasets into `data\raw\`, the scripts were run in this order:
`src\inspect_datasets.py`, `src\text\prepare_text_data.py`,
`src\text\train_text_baseline.py`, `src\audio\build_manifest.py`,
`src\audio\extract_embeddings.py`, `src\audio\train_audio.py`,
`src\fusion\build_scenarios.py`, `src\evaluation\evaluate_fusion.py`,
then the fusion model in `src\fusion\fuse.py`. The protocols are frozen in
`docs\scenario_rule.md` and `docs\fusion_protocol.md`.