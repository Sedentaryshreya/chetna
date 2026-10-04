# Fusion Evaluation Protocol (frozen 4 Oct 2026, before any fusion result)

## Question
Does combining independent audio and text scores give a better safety-risk
assessment than either modality alone? Secondary: does an agreement-aware fusion
beat a plain average?

## Inputs (all scores are held-out)
- Audio: data/processed/audio_oof_probs.csv, out-of-fold p_audio for 2,614 clips,
  5 speaker/source-grouped folds (0-4).
- Text: data/processed/text_val_probs.csv and text_test_probs.csv, probabilities
  p_benign, p_abusive, p_threat from the TF-IDF model (trained on the text train split).

## Scenario ground truth (from the true source labels, per scenario_rule.md)
DISTRESS+THREAT = HIGH; DISTRESS+ABUSIVE, DISTRESS+BENIGN, NON-DISTRESS+THREAT = MODERATE;
NON-DISTRESS+ABUSIVE, NON-DISTRESS+BENIGN = LOW.
Models only ever see model scores, never the true source labels.

## Rotations (5, seed 42)
For rotation k in 0..4: audio fold k = TEST; fold (k+1) mod 5 = VALIDATION; the other
three folds = FUSION-TRAIN. Text: text_val and text_test are each split into 5 parts
(seeded, stratified by class). TEST uses text_test part k. VALIDATION uses text_val
part (k+1) mod 5. FUSION-TRAIN uses text_val parts matching the training folds.
No audio clip or text comment appears in two roles within a rotation.

## Scenario sampling
Equal counts per cell (2 audio classes x 3 text classes), sampled with replacement
inside each role: 300 per cell for TEST and VALIDATION, 600 per cell for TRAIN.
The balanced mix is a design choice; precision, false-alarm rates and F1 therefore
do NOT estimate real-world rates.

## Models (trained on FUSION-TRAIN scenarios only)
- Audio-only: multinomial logistic regression on [logit p_audio].
- Text-only: multinomial logistic regression on [logit p_threat, logit p_abusive].
- Fusion: multinomial logistic regression on [logit p_audio, logit p_threat,
  logit p_abusive, |p_audio - p_threat|, p_audio * p_threat].
- Rule baselines (HIGH-vs-not only): mean(p_audio, p_threat) and min(p_audio, p_threat).
- Scores are clipped to [1e-4, 1 - 1e-4] before the logit.

## Thresholds
For the primary task (HIGH vs not-HIGH), each method's HIGH threshold is chosen on the
VALIDATION scenarios to maximise the F1 of the HIGH class, then applied unchanged to TEST.
No test data is used for any choice.

## Metrics
Primary (HIGH vs not-HIGH): recall, precision, F1, false positive rate, false negative
rate, ROC-AUC, PR-AUC, pooled over the 5 TEST sets, plus per-rotation spread.
Secondary: 3-class macro-F1 (argmax) and confusion matrix; results per scenario cell.
Uncertainty: cluster bootstrap over source audio clips and source text comments.
Claims will be limited to what the confidence intervals support.

## UNCERTAIN
UNCERTAIN is a display state, not a scored class: shown when modality agreement
(1 - |p_audio - p_threat|) is below a threshold chosen on validation data and the system
would otherwise output a risk level. Analysed descriptively (how often, which cells).

## Known limitations (stated in advance)
- Fusion advantage is partly built in: the label depends on both modalities.
- Few unique sources: 51 scream clips and 71 test threats, reused across scenarios,
  so the effective sample is far smaller than the scenario count.
- Audio scores in FUSION-TRAIN folds come from audio models that saw the TEST-fold
  clips (standard stacking leakage); TEST scores themselves are clean.
- Text validation comments were also used to choose the text model's C.
- Scenarios are synthetic pairings, not real incidents; no real-world validity is claimed.
- Any change made after seeing results is reported as a change.

## Clarification (4 Oct 2026, before any fusion result)
UNCERTAIN rule: tau = 5th percentile of agreement (1 - |p_audio - p_threat|) among
VALIDATION scenarios with true label HIGH, per rotation. A fusion HIGH call with
agreement < tau is displayed as UNCERTAIN. This criterion is a design choice, not
validated; UNCERTAIN is analysed descriptively.