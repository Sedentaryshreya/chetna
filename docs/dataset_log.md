# Dataset Log (downloaded 3 Oct 2026; counts from src/inspect_datasets.py)

## ESC-50
- Source: github.com/karolpiczak/ESC-50 (clips from Freesound)
- License: Creative Commons Attribution-NonCommercial 3.0 (CC BY-NC 3.0) for
  the dataset as a whole; ESC-10 subset is CC BY 3.0 Unported. Confirmed from
  data/raw/esc50/ESC-50-master/LICENSE on 3 Oct 2026.
- Size: 2000 clips, 50 classes, 40 per class, 5 folds, 5 s each.
- Use: NON-DISTRESS audio.
- Limitations: no scream class. Authors warn about possible leakage from
  band-limited preprocessing in some clips.

## Deeply Nonverbal Vocalization (OpenSLR 99, public subset)
- Source: openslr.org/99
- License: CC BY-NC-ND 4.0. Used for classifier features only; the audio is
  not redistributed or modified.
- Size: 727 clips, 16 classes. screaming = 51 clips from 31 speakers.
- Use: DISTRESS (screaming) and NON-DISTRESS (other classes except crying,
  moaning, panting).
- Limitations: very few screams, crowdsourced phone recordings, not real emergencies.

## Jigsaw Toxic Comment Classification
- Source: Kaggle competition (Jigsaw / Conversation AI)
- License: CC0 dataset; underlying Wikipedia comment text under CC-SA 3.0.
- Size: 159,571 comments. toxic 15,294 (9.58%), severe_toxic 1,595 (1.00%),
  obscene 8,449 (5.29%), threat 478 (0.30%), insult 7,877 (4.94%),
  identity_hate 1,405 (0.88%). No label: 143,346.
- Use: THREAT / ABUSIVE / BENIGN text classes.
- Limitations: severe imbalance; Wikipedia comments, not messages from real
  safety incidents; no coercion or distress label.