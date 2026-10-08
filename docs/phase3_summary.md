# Phase 3 Summary: Audio Branch (completed 4 Oct 2026)

## Model
- YAMNet (pretrained on AudioSet; 521 classes; 1,024-d embeddings; frozen) ->
  frame embeddings pooled over ACTIVE frames -> standardise -> logistic regression
  (class_weight="balanced"). No fine-tuning of YAMNet.
- Active-frame rule (same for every clip): a frame is kept if >= 25% of its samples
  have amplitude > 0.0001. Parameters are design choices, not validated. On average
  this removed ~1 of 10 frames from ESC-50 clips and almost none from Deeply clips.
- Feature set (mean, or mean+max) and C in {0.001, 0.01, 0.1} chosen by nested
  leave-one-fold-out CV using training folds only.
- Files: src/audio/build_manifest.py, extract_embeddings.py, train_audio.py,
  predict_audio.py; models/audio_yamnet_lr.joblib.

## Data (audio_manifest.csv)
2,614 clips = 51 DISTRESS (Deeply screaming, 31 speakers) + 2,563 NON-DISTRESS
(2,000 ESC-50 + 563 Deeply). Speaker/source-grouped 5 folds (see scenario_rule.md).

## Results (out-of-fold, grouped 5-fold CV)
- Pooled: PR-AUC 0.837 (chance 0.020), ROC-AUC 0.995.
- Per-fold PR-AUC 0.639 to 0.962; mean 0.847 +/- 0.122 (8-13 screams per fold).
- At untuned threshold 0.5: [[TN 2539, FP 24],[FN 7, TP 44]]; recall 0.863,
  precision 0.647, FPR 0.009, FNR 0.137.
- Baseline, YAMNet's own Screaming score (untrained): PR-AUC 0.668, ROC-AUC 0.968.
- Domain-shortcut check: ROC-AUC 0.996 vs ESC-50 negatives, 0.995 vs Deeply negatives
  (no sign of a recording-domain shortcut). Mean p_audio: screams 0.865,
  Deeply negatives 0.023, ESC-50 negatives 0.017. Other shortcuts (e.g. loudness)
  were not tested.

## Limitations
- Only 51 scream clips: metrics are unstable; always report the spread.
- Phone recordings of acted/crowdsourced screams, not real emergencies.
- p_audio is a score, not a calibrated probability; threshold chosen in Phase 4.
- Evidence shown in the app (YAMNet's top labels) describes what YAMNet detected, not
  a decomposition of the logistic-regression decision.
- Clips shorter than 0.96 s are padded by YAMNet to one frame and still receive a score
  (checked 8 Oct 2026 with synthetic tones of 0.2 s and longer). Scores for very short
  clips were not evaluated and should not be trusted. Correction to an earlier statement.
- In the 3-clip consistency check, a held-out scream (p_audio 0.832) had YAMNet top
  labels Animal/Livestock/Fowl and Screaming max 0.262: the classifier score is not
  explained by YAMNet's top labels.