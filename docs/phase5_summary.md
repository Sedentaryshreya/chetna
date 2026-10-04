# Phase 5 Summary: Streamlit App and Pipeline (completed 4 Oct 2026)

## What was built
- src/pipeline.py: analyze() runs audio model, text model and fusion, builds the
  explanation, records timing, and flags whether an audio clip is a dataset clip
  (and then shows its held-out score).
- app.py: thin interface. Banner, upload a file or pick a dataset demo clip, demo
  text or own text, risk level (LOW / MODERATE / HIGH / UNCERTAIN), scores,
  threshold note, "close to threshold" note, evidence panels, explanation, and a
  simulated alert for HIGH/UNCERTAIN only. The alert states that nothing was sent.
- "Evaluation results" tab: figure, metrics, limitations.
- Privacy note in the app. Uploaded audio is written to a temporary file and
  deleted in a finally block. Deletion was not separately verified.
- Run with: python -m streamlit run app.py (config binds to localhost only).

## Manual functional tests (not evidence of performance)
| Audio | Text | Result |
|-------|------|--------|
| scream clip 1 | threat text | HIGH (0.99 / 0.98 / 0.00 / 1.00) |
| scream clip | benign text | MODERATE |
| dog clip | threat text | MODERATE (combined 0.22) |
| dog clip | "kill" idiom | HIGH (combined 0.33, near threshold) |
| dog clip | benign text | LOW |
| own 6 s WAV | "hey how are you" | LOW (audio 0.02, threat 0.00, abusive 0.39) |

## Findings
- Dog + "kill" idiom produced HIGH while dog + real threat produced MODERATE.
  The ordering is wrong. The text model is keyword-driven and the fused score is
  near the threshold (0.28). Documented, NOT retuned.
- Abusive score 0.39 on "hey how are you" shows the text model is keyword-driven.
- Processing time: first run 7.93 s (model loading), later about 1.10 s in the
  app. One machine, not a benchmark.
- m4a is not readable by librosa without ffmpeg. The test recording was converted
  to WAV. Supported input in the app: WAV.

## UI checks (4 Oct 2026)
- Evaluation tab shows the figure, three expanders and limitations.
- Analyze with "Upload a file" selected and no file gives a friendly red message.

## Limitations
- Manual tests are functional checks, not evaluation. Performance claims come only
  from src/evaluation/evaluate_fusion.py (see phase4_summary.md).
- The HIGH threshold was tuned on a balanced scenario mix; real-world false alarms
  would be higher.
- Not tested on real emergency audio or real conversations.
- Temp-file deletion for uploads was not independently verified.