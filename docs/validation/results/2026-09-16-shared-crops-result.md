# 2026-09-16 shared crops result

**A regression explained by slice coverage.** Returning label-private to shared fine pooling
reached **0.879626** fold-0 macro AUC at epoch 8: +0.0345 against
[label-private](2026-09-16-label-private-crops-result.md) (paired report-group bootstrap
[+0.0268, +0.0423]) but −0.0121 against [per-label crops](2026-09-16-per-label-crops-result.md)
([−0.0168, −0.0078]) and −0.0131 against `00-baseline`. Two crops per label land on only about
13 of a study's 51 validation canvases; the per-label rule crops about 38.

## Setup

- **Change from label-private:** every label pools all 24 crops of its study; label FiLM removed;
  selection ranks the per-label coarse attention alone (same choices as before); coarse tokens,
  scores and pool computed once; attention prior retired
  (recipe snapshot (`f076a09:docs/history/crop-selection-sources/active-spatial-token-model.md.txt`)). Kept: 2 crops
  per label with plane spread, 24 ReXNet-200 crops per study without deduplication, coupled fine
  scores, the per-label gate on one shared classifier. 300-pixel canvases at 0.5 mm, 224-pixel
  crops, batch 8, 10 epochs, seed 42, fold 0 of `all_5fold_v3`.
- **Run:** `outputs/training/shared-crops`; epoch 9 scored 0.879174.
- **Cost:** 237 s per epoch including validation (label-private 234 s), 43 s for the final
  validation prediction, 4,380 optimizer steps, peak 22.5 GiB CUDA. Removing the duplicated coarse pass and
  the FiLM split saved nothing measurable.
- **Artifacts:** comparisons in `outputs/analysis/s/shared-vs-per-label-20260916` and
  `outputs/analysis/s/shared-vs-label-private-20260916`, probes in
  `outputs/analysis/s/shared-crops-probes-20260916`, placement in
  `outputs/analysis/c/crop-diversity-shared-20260916`.

## Training curve

| Epoch | shared crops | draft (coarse pool) | train loss | per-label | label-private |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.7732 | 0.685 | 0.583 | 0.7778 | 0.7010 |
| 1 | 0.8128 | 0.609 | 0.544 | 0.8284 | 0.7512 |
| 2 | 0.8372 | 0.622 | 0.526 | 0.8493 | 0.7700 |
| 3 | 0.8469 | 0.626 | 0.509 | 0.8633 | 0.7903 |
| 4 | 0.8666 | 0.648 | 0.498 | 0.8736 | 0.8119 |
| 5 | 0.8694 | 0.643 | 0.485 | 0.8819 | 0.8255 |
| 6 | 0.8751 | 0.638 | 0.475 | 0.8887 | 0.8354 |
| 7 | 0.8779 | 0.648 | 0.466 | 0.8898 | 0.8403 |
| 8 | **0.8796** | 0.648 | 0.460 | **0.8917** | 0.8428 |
| 9 | 0.8792 | 0.649 | 0.454 | 0.8915 | **0.8451** |

Training loss ended close to per-label's (0.449), while validation BCE is 0.45876 against 0.44834.
Gold-label macro AUC is 0.9282 (per-label 0.9268). With strong shared fine features the coarse
pool fell to 0.648, close to per-label's 0.626, and the gates ended at 0.47–0.49.

## Per-target AUC

| Target | `00-baseline` | per-label | label-private | shared | vs per-label | vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| ACL | 0.9146 | 0.9257 | 0.8478 | 0.9030 | **−0.0227** | −0.0116 |
| Medial Meniscus | 0.9239 | 0.9248 | 0.8442 | 0.9060 | −0.0189 | −0.0179 |
| MCL | 0.8424 | 0.8359 | 0.7733 | 0.8180 | −0.0180 | −0.0244 |
| Medial OA | 0.9045 | 0.9102 | 0.8835 | 0.8931 | −0.0171 | −0.0114 |
| Lateral Meniscus | 0.8939 | 0.8837 | 0.7429 | 0.8668 | −0.0169 | −0.0271 |
| PF OA | 0.8715 | 0.8687 | 0.8343 | 0.8544 | −0.0143 | −0.0171 |
| Lateral OA | 0.8662 | 0.8660 | 0.8452 | 0.8536 | −0.0123 | −0.0126 |
| Synovitis | 0.8911 | 0.8988 | 0.8765 | 0.8868 | −0.0120 | −0.0043 |
| Fracture | 0.9301 | 0.9216 | 0.9026 | 0.9129 | −0.0088 | −0.0172 |
| Contusion | 0.8899 | 0.8873 | 0.8480 | 0.8834 | −0.0039 | −0.0065 |
| Effusion | 0.8702 | 0.8667 | 0.8601 | 0.8650 | −0.0017 | −0.0052 |
| Baker's | 0.9150 | 0.9111 | 0.8831 | 0.9127 | +0.0016 | −0.0023 |

Family intervals against per-label: meniscus −0.0182 [−0.0282, −0.0087], OA −0.0148
[−0.0246, −0.0056], other eight −0.0100 [−0.0154, −0.0044]. The central detail targets that
per-label had gained (ACL, medial OA) lost most.

## Inference probes

The best checkpoint with BF16, changing only validation-time crops or the gate:

| Variant | Macro AUC | Distinct crops per study |
|---|---:|---:|
| 2 per label (as trained) | 0.879626 | 20.4 |
| 2 per label, duplicates masked | 0.879356 | 20.4 |
| 3 per label | 0.882231 | 29.4 |
| 4 per label | 0.882574 | 38.1 |
| 4 per label, duplicates masked | 0.882367 | 38.1 |
| Fine only (gate forced to 0) | 0.877197 | 20.4 |
| Coarse only (gate forced to 1) | 0.647582 | 20.4 |

- **Coverage:** at validation (2× images) a study has 50.7 canvases on average, but the 24 crops
  fall on only 12.9 distinct canvases; the per-label rule, `ceil(0.75 × canvases)` with at most one
  crop per canvas, crops 38.2. Many labels put both their crops on the same favourite slices at
  different, heavily overlapping positions (a crop covers 75% of the canvas).
- More crops at inference recover 0.003 of the 0.012 gap, and duplicate crops are harmless.
- The coarse pool adds only 0.0024 over fine-only pooling.

## Placement

`scripts/adhoc/analyze_crop_diversity.py` on the usual 48 studies: 19.5 distinct (canvas, cell)
pairs per study; ACL, both menisci and both OA grades sit at the bottom of the selectable range
(mean vertical offset +15 to +21 mm, spread 3–6 mm on ACL and OA), a label habit rather than a
study-specific choice. Positive-minus-negative shifts are large only for synovitis (−14.0 mm
vertical), effusion (−9.0) and contusion (+8.1); with 48 studies, random selection already shows
about 4 mm ([probe](2026-09-16-label-private-crops-result.md#selection-signal-probe)).

## Reading

- **Slice coverage matters more than per-label targeting at 224 pixels.** Across the crop runs,
  score follows how many slices ReXNet-200 sees: `00-baseline` (every triplet) 0.8928; crop224 and
  per-label (75% of canvases) 0.8916–0.8917; shared crops (about a quarter of the canvases)
  0.8796; label-private (the same, pooled privately) 0.8451.
- **The trained coarse pathway only learns when the fine pathway is weak.** Its label-private
  gain (0.82) disappeared once fine pooling was shared, so it cannot be relied on to steer
  selection.
- **Per-label targeting did not make crops study-specific either:** detail labels settled on a
  fixed lower position.
- Single seed, fold 0; the coverage explanation is supported by the probes and the cross-run
  pattern, not by a controlled run.
