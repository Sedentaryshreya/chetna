# Phase 4 Summary: Fusion and Evaluation (completed 4 Oct 2026)

## Setup (protocol frozen in fusion_protocol.md before any result)
- 5 rotations; audio fold k = TEST, fold k+1 = VALIDATION, other 3 = TRAIN. Text from
  the matching stratified fifths of text_val/text_test. Balanced scenarios: 300 per
  cell for TEST/VALIDATION, 600 per cell for TRAIN. No source shared between roles.
- Pooled TEST: 9,000 scenarios, 1,500 HIGH. Each rotation's TEST set uses only 8-13
  unique scream clips and 14-15 unique threat comments, so intervals are wide.
- Methods: audio-only, text-only, fusion (logistic regression on logit scores plus
  |p_audio - p_threat| and p_audio*p_threat), mean rule, min rule. HIGH thresholds
  chosen on VALIDATION only. 95% intervals from a two-way source bootstrap (1,000 draws).

## Results: HIGH vs not-HIGH, pooled TEST
| Method     | F1    | 95% CI          | FPR   | FNR   | PR-AUC |
|------------|-------|-----------------|-------|-------|--------|
| audio-only | 0.478 | [0.404, 0.545]  | 0.350 | 0.136 | 0.333  |
| text-only  | 0.546 | [0.467, 0.609]  | 0.215 | 0.221 | 0.474  |
| fusion     | 0.748 | [0.671, 0.808]  | 0.048 | 0.260 | 0.859  |
| mean rule  | 0.765 | [0.677, 0.836]  | 0.017 | 0.329 | 0.815  |
| min rule   | 0.755 | [0.662, 0.832]  | 0.022 | 0.327 | 0.873  |

Fusion minus other (F1, paired bootstrap): audio-only +0.269 [+0.195, +0.339];
text-only +0.202 [+0.137, +0.280]; mean rule -0.017 [-0.064, +0.027];
min rule -0.008 [-0.063, +0.059].
PR-AUC fusion minus mean rule +0.044 [+0.009, +0.083]; minus min rule -0.014 [-0.043, +0.013].
3-class macro-F1 (argmax): audio-only 0.494, text-only 0.472, fusion 0.820 (not computed
for the rule baselines; gap vs single modalities is largely built in by the label design).

## What the results support
1. Combining modalities sharply cut false alarms: text-only called 75.7% of
   NON-DISTRESS+THREAT scenarios HIGH (fusion 3.4%); audio-only called 85.8% of
   DISTRESS+BENIGN scenarios HIGH (fusion 0.6%). FPR fell from 0.350 / 0.215 to 0.048.
2. The cost is more misses: fusion FNR 0.260 vs audio-only 0.136 (intervals overlap,
   so the gap is not established). Thresholds maximised F1, which weights both error
   types equally; a recall-focused threshold was not evaluated.
3. Learned fusion did NOT clearly beat simple rules: F1 differences vs the mean and min
   rules include zero. Requiring both modalities to agree (the min rule) captures most
   of the benefit.
4. The UNCERTAIN flag was informative: 19% of fusion HIGH calls (279 of 1,469) were
   flagged; 7.7% of calls on true HIGH scenarios vs 60.2% of calls on DISTRESS+ABUSIVE.
   Precision rose from 0.756 to 0.861 for confident calls, recall fell from 0.740 to 0.683.
   The cut-off (tau = 0.03 on validation) fires only on extreme disagreement.
5. 82% of fusion's false HIGH calls (294 of 359) were DISTRESS+ABUSIVE scenarios, which the
   frozen rule labels MODERATE; escalation there may be defensible, so this is partly a
   limit of the labelling rule.

## What the results do NOT support
- Any claim about real emergencies, real recordings or real-world prevalence (synthetic,
  balanced pairings; precision/FPR do not transfer).
- That learned fusion beats a plain average or minimum rule.
- That fusion is better than single modalities as a discovery: the label needs both
  modalities, so a single-modality model cannot win by construction.
- Coefficient interpretations of the fusion model (correlated features; |p_audio-p_threat|
  has a positive coefficient).

## Known limitations (also in fusion_protocol.md)
- Only 8-13 unique screams and 14-15 unique threats per TEST set.
- Stacking leakage: audio scores in FUSION-TRAIN folds came from models that saw TEST-fold clips.
- The combined decision rule (threshold, then UNCERTAIN, then MODERATE/LOW) was not
  evaluated as a whole; HIGH-vs-not and 3-class argmax were.
- The demo model in models/fusion_lr.joblib pools rotations, so its thresholds are not
  a clean performance estimate; performance claims come only from evaluate_fusion.py.
- Display wording of "Evidence agreement" is a convention (same side of 0.5), not validated.
- Demo behaviour (hand-set scores, not evidence): audio 0.55 + threat 0.50 gives HIGH
  (0.84) because the HIGH threshold (0.28) was tuned for F1 on a balanced scenario mix;
  at real-world prevalence this threshold would produce many more false alarms.
- UNCERTAIN is narrow: it appears only when the fused HIGH score is above threshold AND
  audio/threat agreement < 0.03 (e.g. scream plus abusive text). Audio 0.99 + threat 0.005
  with no abusive text gave MODERATE, not UNCERTAIN.

## Files
src/fusion/build_scenarios.py, src/evaluation/evaluate_fusion.py, src/fusion/fuse.py,
data/scenarios/scenarios.csv, results/fusion_metrics.txt, fusion_test_predictions.csv,
results/figures/fusion_comparison.png, models/fusion_lr.joblib.

## UNCERTAIN probe (4 Oct 2026; src/fusion/probe_uncertain.py)
- 279 of 1,469 fusion HIGH calls were UNCERTAIN (agreement < tau); counts match the evaluation.
- Typical UNCERTAIN case (median): p_audio 0.9999, p_threat 0.0064, p_abusive 0.984:
  audio saturated at "distress", text sees abuse but almost no threat.
- By true cell: DISTRESS+ABUSIVE 177, DISTRESS+THREAT 86, NON-DISTRESS+THREAT 12,
  DISTRESS+BENIGN 4. About 69% were not true HIGH; about 31% (86) were true HIGH where
  the text model scored the threat near zero. UNCERTAIN therefore means "modalities
  disagree: human review", not "probable false alarm".
- The pooled demo model (fuse.py) and the per-rotation evaluation models can disagree
  near the threshold: one probed example the evaluation called HIGH was MODERATE
  (P_HIGH 0.225) in the demo model. Behaviour near saturated scores is sensitive.
- Hand-set example audio 0.99 / threat 0.005 / abusive 0.9 gave MODERATE (P_HIGH 0.108),
  not UNCERTAIN.