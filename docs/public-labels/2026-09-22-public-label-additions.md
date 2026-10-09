# 2026-09-22 public label additions

**Nothing new is worth training on.** A sweep of Kaggle found nine label tables published since
the [09-21 re-screen](2026-09-21-public-label-rescreen.md) or missed by it. Every table that can be
scored on gold sits at or below Steven v4 on the same rows, apart from one Pilkwang v2 derivative
at +0.002. The only use found for them is as independent sources for the
[calibrated judge](../validation/results/2026-09-22-calibrated-judge.md): Hosseini's JEV v2 is the
one source that passed its eligibility rules.

- **Search:** Kaggle datasets matching knee, RSNA, report, pseudo, soft, weak and LLM labels,
  updated since 2026-08-20, plus competition notebooks whose titles mention labels.
- **Snapshots:** `data/labels/public/<owner>/<slug>/downloaded-20260922/`.
- **Script:** [`screen_public_label_additions`](../../scripts/adhoc/screen_public_label_additions.py).
- **Output:** `outputs/analysis/p/public-label-additions-20260922/` (`screen.csv`, `manifest.json`
  with input hashes). CPU only, seconds; refuses an existing output directory:
  `rtk proxy uv run --locked python -m scripts.adhoc.screen_public_label_additions --output-dir outputs/analysis/public-label-additions-new`.
- **Scoring columns:** Hosseini on its positive-class probability, lvthanhxun on its fused
  teacher target (`__y`). A gold-equal share of 1.0 marks a table that carries the gold labels.

## Tables

Gold macro AUC is on the official 58 studies. Steven v4 scores 0.8927 on the same rows, or 0.8932
where a table misses one study.

| Dataset | Gold macro AUC | Note |
|---|---:|---|
| `davidadekanmi/pseudo-labels`, `synthesized_labels.csv` | 0.8955 | 77% of silver cells equal Pilkwang v2 |
| `davidadekanmi/pseudo-labels`, `report_labels_v2 (1).csv` | 0.8700 | An exact copy of Pilkwang v2 |
| `pearsejim01/rsna-2026-eda`, `weak_labels_frontier` | 0.8617 | Per-label positive and negative source counts |
| `hosseini7798/rsna-knee-label-extraction-jev` (published 09-22) | 0.8544 | Positive, negative and unknown probabilities per label |
| `pearsejim01/rsna-2026-eda`, `weak_labels_v2` | 0.8330 | Labels, confidence and a mention flag |
| `lvthanhxun/rsna-knee-v9-teacher`, fused teacher target | 0.7541 | Qwen and rule-based report states per label |
| `achelijndiamantidis/knee-phase1-artifacts` | 0.7275 | Soft labels with a source channel |
| `nekkon/rsna-knee-nli-entailment-scores` | 0.7272 | mDeBERTa NLI entailment, maximum over sentences; 89% of cells fall in the top fifth |
| `pearsejim01/rsna-2026-eda`, `labels_v3` | not a benchmark | Carries the gold labels; adds mention flags and extra attributes (PCL, LCL, root tear, extrusion, grade) |
| `nartaa/rsna-knee-hpo-assets`, four variants | not scored | They omit the gold studies |

Two further finds are not label tables:

- **`davidadekanmi` `human_calibration_review_100.csv`:** 100 studies (2 gold) reviewed from report
  text, 0/1/NA per label. It is not image-based truth.
- **`malikahsan/knee-mri-reports-translated-transcribed`:** English translations of all reports. The
  sample checked was garbled, so it is not relied on for rule-based extraction.

The competition notebooks that train on custom labels (for example `luonmayman`'s
denoised-ensemble and per-target variants) build them from Steven, Lixin and Pilkwang, which
this project already holds. Hosseini's pairwise-comparison dataset is JSON, not a label table.
