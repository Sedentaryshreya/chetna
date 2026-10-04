# Chetna Scenario Rule (frozen 3 Oct 2026, before any modelling)

## Source classes
Audio
- DISTRESS: Deeply "screaming" clips (51 clips, 31 speakers).
- NON-DISTRESS: ESC-50 clips (2000) plus Deeply non-scream clips, excluding
  crying, moaning and panting (excluded as ambiguous). Deeply negatives: 563 clips.
  Deeply negatives are included so the model cannot separate classes by recording
  domain alone.

Text (Jigsaw Toxic Comment, 159,571 comments)
- THREAT:  threat == 1 (478 comments).
- ABUSIVE: (toxic == 1 or insult == 1) and threat == 0.
- BENIGN:  all six labels == 0 (143,346 comments).
- OTHER (excluded from text training): comments with a Jigsaw label but neither
  threat, toxic nor insult (e.g. obscene-only, identity_hate-only), 374 comments.
  They fit neither ABUSIVE nor BENIGN, so assigning them would be a labelling guess.
## Scenario label (ground truth) = (audio class, text class)
DISTRESS     + THREAT   -> HIGH
DISTRESS     + ABUSIVE  -> MODERATE
DISTRESS     + BENIGN   -> MODERATE
NON-DISTRESS + THREAT   -> MODERATE
NON-DISTRESS + ABUSIVE  -> LOW
NON-DISTRESS + BENIGN   -> LOW

MODERATE means "evidence from one modality only". UNCERTAIN is a system output
state, not a ground-truth label. How UNCERTAIN is scored is decided in Phase 4
and fixed before looking at test results.

## Splitting (to prevent leakage)
- Audio: 5-fold cross-validation grouped by source. ESC-50 uses its official
  folds 1-5. Deeply speakers are assigned to folds by a seeded random
  assignment (seed 42) of speakerID, so no speaker appears in two folds.
- Text: one fixed split of Jigsaw comments (70/15/15), stratified on threat,
  seed 42. Text data is large, so cross-validation is not needed.
- Scenario pairs are formed only from sources inside the same evaluation split.
  No source clip or comment appears in both training and test.
- Final protocol details are confirmed in Phases 3 and 4. Any change after
  seeing results must be reported as a change.

## Known limitations
- Labels are defined by this rule, so a fusion advantage is partly by design.
- Only 51 scream clips: audio metrics are unstable and reported as mean and
  spread across folds, not as a single number.
- Scream audio is out-of-domain (phone recordings, not real emergencies).
- Jigsaw has no coercion or distress label; we do not claim to detect coercion.
- Threat is very rare (0.30%); class imbalance is handled in Phase 2.

## Audio fold construction (4 Oct 2026)
- Official ESC-50 folds were used, except that 4 source recordings (src_file) had
  clips in more than one official fold. Each such recording was kept in the lowest
  fold it appears in (5 clips moved, 0.25% of ESC-50), so no source recording is
  split across train and test.
- Deeply speakers were assigned to 5 folds by seeded shuffle (seed 42); scream
  speakers were distributed first (7, 6, 6, 6, 6 per fold). No speaker spans two folds.
- 113 Deeply clips (crying 24, moaning 45, panting 44) excluded as ambiguous.
- Final audio set: 2,614 clips = 51 DISTRESS + 2,563 NON-DISTRESS
  (2,000 ESC-50 + 563 Deeply). Scream clips per fold: 10, 8, 13, 9, 11.