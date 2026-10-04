# Phase 2 Summary: Text Branch (completed 4 Oct 2026)

## Model
- TF-IDF (word 1-2 grams, 200,000 features) + multinomial logistic regression,
  class_weight="balanced". C=4.0 chosen on the validation set (candidates 1, 4, 10).
- Classes: BENIGN, ABUSIVE, THREAT (frozen rule in scenario_rule.md).
- Files: src/text/prepare_text_data.py, train_text_baseline.py, predict_text.py;
  model at models/text_tfidf_lr.joblib.

## Data splits (stratified, seed 42, no overlap, 374 OTHER comments excluded)
| Class   | Train   | Val    | Test   |
|---------|---------|--------|--------|
| BENIGN  | 100,341 | 21,502 | 21,503 |
| ABUSIVE |  10,761 |  2,306 |  2,306 |
| THREAT  |     335 |     72 |     71 |

## Test results (n = 23,880; evaluated once, after choosing C on validation)
| Class   | Precision | Recall | F1    | Support |
|---------|-----------|--------|-------|---------|
| BENIGN  | 0.978     | 0.972  | 0.975 | 21,503  |
| ABUSIVE | 0.742     | 0.789  | 0.765 |  2,306  |
| THREAT  | 0.493     | 0.465  | 0.478 |     71  |
Macro-F1 0.739. Accuracy 0.953 (dominated by BENIGN; not a useful headline number).
Threat ROC-AUC 0.972 but PR-AUC 0.440. PR-AUC is the more honest figure for a rare class.
Confusion matrix (rows true, cols predicted: BENIGN, ABUSIVE, THREAT):
[[20894, 601, 8], [461, 1819, 26], [6, 32, 33]]

## Findings
- THREAT is weak: 33 of 71 test threats found; 26 abusive and 8 benign comments flagged
  as THREAT. Only 71 test positives, so threat metrics are noisy (validation recall
  was 0.347, test 0.465 on 72/71 positives). Always report with counts.
- Of 71 test threats, 32 were labelled ABUSIVE and 6 BENIGN: the model usually detects
  hostility but struggles to separate threat from abuse.
- Qualitative demo (4 hand-written sentences, not evidence): "I am going to kill this
  presentation tomorrow" scored THREAT 0.994. The model keys on the word "kill".
  This is a limitation of keyword-driven text models and a motivation for fusion.
  These sentences are not used in quantitative evaluation (scenario rule is frozen).
- Explanation = words with the largest positive contribution to a class score
  (coefficient x TF-IDF value). It shows model association, not intent. Show only
  for classes with high probability.

## Decisions
- DistilBERT NOT built in Phase 2: CPU training too slow, no guarantee of gain with
  335 training threats, and the working multimodal pipeline has priority. Revisit
  only if the pipeline is finished and evaluated with time left.
- Decision thresholds are not tuned yet. Fusion (Phase 4) uses the continuous
  probabilities, with thresholds chosen on validation data.

## Limitations
- Wikipedia talk-page comments, not messages from real safety incidents.
- Labels from crowd raters are noisy. No coercion or distress label exists.
- Word-level model: no sentence meaning, sarcasm or context.