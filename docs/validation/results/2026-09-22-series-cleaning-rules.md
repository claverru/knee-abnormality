# Series-cleaning rules, inference only (2026-09-22)

Three candidate series-cleaning rules, each applied only at prediction time with the incumbent
checkpoint on the fold-0 studies it touches. Only one rule moves fold 0: keeping a single series per
same-signature, same-geometry group, at +0.00013 to +0.00015. The contrast relabel and near-empty
frames are at or below 0.00001. No model was retrained for these measurements; `k_05`'s recipe
[retrained with all four rules](#retrained-with-cleaning) moved fold 0 by +0.0003 at ×4.

- **Setup:** `oblique-x4` checkpoint (`best-epoch=09`), fold 0 of `all_5fold_scanner_v1`,
  ×4 images, attention temperature 1, CPU inference.
- **Method:** intact predictions for the affected studies are remade in the same process, so
  unaffected studies are identical in both arms. Intact reproduces the saved fold-0 macro AUC
  (0.902379 against 0.902373). The full-fold delta uses 2,000 report-group bootstrap draws.
- **Artifacts:** `outputs/analysis/series-cleaning-rules-20260922/`, produced by the
  [scoring script](../../../scripts/adhoc/score_series_cleaning_rules.py).

| Rule | Fold-0 studies | Mean shift | Max shift | Studies moved >0.05 | Fold-0 macro Δ [95%] |
|---|---:|---:|---:|---:|---|
| Contrast-slot relabel (census confirmed flags) | 2 | 0.005 | 0.025 | 0 | −0.000002 [−0.000011, +0.000001] |
| Same-signature group, keep newest `SeriesNumber` | 42 | 0.008 | 0.257 | 5 | +0.000153 [−0.000003, +0.000371] |
| Same-signature group, keep oldest `SeriesNumber` | 42 | 0.008 | 0.103 | 6 | +0.000128 [−0.000090, +0.000465] |
| Drop near-empty frames (<1% non-zero pixels) | 1 | 0.009 | 0.021 | 0 | −0.000009 [−0.000033, 0] |

Shifts are absolute probability changes per study, taken as the mean over labels and the maximum
over labels and studies.

## Same-signature groups

A group is series in one slot with the same slice count, sequence signature (TR, TE, echo train
length, flip angle, acquisition type, variant, options, matrix) and sorted slice positions
(0.01 mm). Exact pixel copies are excluded. Fold 0 has 57 such groups in 42 studies:

- 12 are Philips second reconstructions (`SeriesNumber` *N*/*N*+1, middle-frame correlation
  0.999).
- The rest are retakes with a `SeriesNumber` gap of 2–7 or 100–7,000 (median correlation
  0.66–0.99).
- None are GE N/40000+N pairs, which differ in signature or geometry.

Both keep-one variants gain about the same. The gain therefore comes from dropping the redundant
series, which concentrates the slot's triplets on one series, not from which series is kept. The
largest per-label changes are Fracture (+0.0012 newest, +0.0019 oldest; few positives) and Lateral
Meniscus (+0.0004 newest, −0.0009 oldest). One study moves 0.26 on Lateral Meniscus under
keep-newest. This measures, at inference, the superseded-retakes rule that is now implemented in
the [series-hygiene run](../../backlog/overnight.md#next-queue).

## Near-empty frames

Of the 554 training frames with a constant central 8 KB, only 61 are near-empty (<1% non-zero
pixels) and 5 are fully zero; the median flagged frame is 14.5% non-zero. In the 10 training
studies holding a near-empty frame, the sampler selects one in 3 of 998 ×4 triplets.

## Contrast-slot relabel

This matches the earlier inference-only census check (+0.009 ×1000 on the control checkpoints). On
fold 0 the rule touches 2 studies; across training it touches 12 series.

**Implemented without a model (2026-09-22).** The census confirmed its 12 mislabels with a trained
intensity classifier, which the user ruled out shipping. The implemented rule keeps the census
header candidates (53 series in 41 studies). It decides each one by comparing its body-intensity
features with the same study's trusted series, and moves it when its nearest series is the other
contrast. Same-plane series are used only when both contrasts exist in that plane. Results on the
24 visually reviewed header candidates:

| Variant | Mislabels moved | Correct labels moved | Unreviewed moved |
|---|---:|---:|---:|
| Nearest series, same plane preferred | 11/12 | 1/12 | 0 |
| Nearest series, same plane only when both contrasts exist (implemented) | 11/12 | 0/12 | 0 |
| Plain nearest series with a 0.75 distance-ratio margin | 7/12 | 0/12 | 0 |

- **Candidate definition:** counting fluid-slot series without TE, TR or scan options as candidates
  would have moved 2 GE "Sag PD" series whose images look fat-suppressed by eye.
- **Ceiling:** the one miss is a Siemens `pd_tse_fs_sag_np`; an identical protocol in the same study
  layout was judged not fat-suppressed.
- **Caveat:** the variant was chosen among about a dozen tried on the same 24 cases, so its zero
  wrong moves is optimistic.

The production path reproduces 11 moves, all census mislabels, each moved to the census target slot.
Scratch evaluation scripts are not kept; the rule's inputs are the census parquets in
`outputs/analysis/series-filtering-20260917/`.

## Retrained with cleaning

`k05-clean` (2026-09-23) retrains `k_05`'s recipe on the source with all four rules: `configs/k05.yaml`,
seed 2263738815, fold 0 of `all_5fold_scanner_v1`, combined labels. The loader and inference
changes made since `b483140` leave the prepared images bit-identical, so the difference from `k_05`
is the cleaning plus same-seed training divergence.

| Run | Selected epoch | ×1 epoch macro AUC | ×4 τ 1 macro AUC | Public (×4 τ 0.75) |
|---|---:|---:|---:|---:|
| `k_05` | 7 | 0.899042 | 0.902469 | 0.945 (S41) |
| `k05-clean` | 8 | 0.899419 | 0.902760 | 0.942 (S48) |

Fold 0 gains +0.0004 at ×1 and +0.0003 at ×4. The public score drops 0.003, inside the 0.928–0.934
spread of the same recipe trained four times
([seed controls](../../inference/submissions/2026-09-17-seed-control-submissions.md)). The rules
stay: they remove copies, superseded retakes, mislabelled contrast and near-empty frames at almost
no cost. Artifacts: `outputs/training/k05-clean`, bundle
`outputs/submission/exports/k05clean-20260923-x4-tau075`.
