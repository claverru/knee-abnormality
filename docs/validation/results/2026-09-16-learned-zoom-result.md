# 2026-09-16 learned-zoom result

**The detail glimpses went unused.** Run 0 of the glimpse plan (a context glimpse on every slice
plus learned detail glimpses) reached **0.869951** fold-0 macro AUC at epoch 9: −0.0218 against
[per-label crops](2026-09-16-per-label-crops-result.md) (paired report-group bootstrap
[−0.0275, −0.0162]), −0.0097 against [shared crops](2026-09-16-shared-crops-result.md)
([−0.0157, −0.0039]) and −0.0228 against `00-baseline`. Removing or randomising the detail
glimpses at inference changes nothing, so the model is in effect a 0.875 mm context model.

## Setup

- **Recipe** ([investigation record](../../history/crop-selection-investigation.md#learned-zoom)): canvases at 360 px and
  0.4167 mm (150 mm); the MobileNetV3-small scout reads a 300 px downsample; one centred context
  glimpse per canvas (140 mm at 160 px, 0.875 mm, 5×5 tokens); `ceil(0.333 × canvases)` detail
  glimpses (93 mm at 224 px, 0.4167 mm, 7×7 tokens) chosen by a demand head and greedy
  facility location with k-means latent coverage; one softmax per label over coarse, context and
  detail tokens, detail logits adding the glimpse's log mean demand. Batch 8, 10 epochs, seed 42,
  fold 0 of `all_5fold_v3`.
- **Run:** `outputs/training/learned-zoom`, best epoch 9.
- **Cost:** 246–262 s per epoch including validation (shared crops 237 s), 59 s for the final
  validation prediction, peak 22.4 GiB CUDA.
- **Artifacts:** probes in `outputs/analysis/l/learned-zoom-probes`, comparisons in
  `outputs/analysis/z/zoom-vs-per-label-20260916` and `outputs/analysis/z/zoom-vs-shared-20260916`.

## Training curve

| Epoch | Macro AUC | Draft | Train loss | Coarse share | Context share | Detail share |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7378 | 0.675 | 0.590 | 0.777 | 0.154 | 0.069 |
| 1 | 0.8054 | 0.719 | 0.560 | 0.501 | 0.396 | 0.103 |
| 2 | 0.8155 | 0.724 | 0.536 | 0.421 | 0.513 | 0.066 |
| 3 | 0.8368 | 0.712 | 0.522 | 0.319 | 0.632 | 0.049 |
| 4 | 0.8489 | 0.705 | 0.508 | 0.308 | 0.654 | 0.038 |
| 5 | 0.8565 | 0.687 | 0.494 | 0.263 | 0.715 | 0.022 |
| 6 | 0.8649 | 0.703 | 0.482 | 0.237 | 0.742 | 0.020 |
| 7 | 0.8689 | 0.693 | 0.474 | 0.213 | 0.768 | 0.019 |
| 8 | 0.8695 | 0.687 | 0.466 | 0.203 | 0.776 | 0.021 |
| 9 | **0.8700** | 0.687 | 0.463 | 0.196 | 0.783 | 0.021 |

Shares are each label's attention mass on each token type, averaged over labels and validation
studies. Detail tokens are about 4% of a validation study's tokens; their share peaked at 10% in
epoch 1 and settled at 2%, below that level, while context took over from the scout. Gold-label
macro AUC is 0.9321 and soft-target BCE 0.46799.

## Probes

`scripts/adhoc/probe_glimpses.py`, BF16, paired bootstrap against the trained selection:

| Condition | Detail glimpses per study | Macro AUC | Δ [95% interval] | Detail share |
|---|---:|---:|---|---:|
| As trained | 17.4 | 0.869956 | — | 0.021 |
| No detail | 0 | 0.869777 | −0.0002 [−0.0005, +0.0001] | 0 |
| Random detail | 17.4 | 0.869834 | −0.0001 [−0.0004, +0.0002] | 0.016 |
| Half budget | 8.7 | 0.869805 | −0.0002 [−0.0003, +0.0000] | 0.011 |
| Double budget | 34.1 | 0.869992 | +0.0000 [−0.0002, +0.0003] | 0.037 |

## Per-target AUC

| Target | `00-baseline` | per-label | shared | learned zoom | vs per-label | vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| Lateral Meniscus | 0.8939 | 0.8837 | 0.8668 | 0.7882 | **−0.0955** | −0.1057 |
| MCL | 0.8424 | 0.8359 | 0.8180 | 0.7753 | −0.0606 | −0.0671 |
| ACL | 0.9146 | 0.9257 | 0.9030 | 0.8923 | −0.0334 | −0.0223 |
| Synovitis | 0.8911 | 0.8988 | 0.8868 | 0.8804 | −0.0184 | −0.0107 |
| Baker's | 0.9150 | 0.9111 | 0.9127 | 0.8969 | −0.0142 | −0.0181 |
| Lateral OA | 0.8662 | 0.8660 | 0.8536 | 0.8528 | −0.0132 | −0.0134 |
| Medial OA | 0.9045 | 0.9102 | 0.8931 | 0.8972 | −0.0130 | −0.0073 |
| Medial Meniscus | 0.9239 | 0.9248 | 0.9060 | 0.9154 | −0.0095 | −0.0085 |
| Fracture | 0.9301 | 0.9216 | 0.9129 | 0.9139 | −0.0077 | −0.0162 |
| PF OA | 0.8715 | 0.8687 | 0.8544 | 0.8612 | −0.0075 | −0.0103 |
| Contusion | 0.8899 | 0.8873 | 0.8834 | 0.8903 | +0.0030 | +0.0004 |
| Effusion | 0.8702 | 0.8667 | 0.8650 | 0.8757 | +0.0090 | +0.0055 |

## Reading

- **Learned zoom failed as built.** Detail glimpses carry no measurable evidence, and learned
  placement is no better than random. The demand head's only gradient comes through the detail
  tokens' pooling weight, which shrank as the every-slice context took over, so demand had
  almost nothing to learn from (hypothesis for the mechanism; the outcome is measured).
- **Coarse in-plane resolution costs detail targets.** With ReXNet-200 evidence effectively at
  0.875 mm on every slice, the lateral meniscus (−0.106 against `00-baseline`), MCL (−0.067) and
  ACL (−0.022) fell most, while effusion (+0.006) and contusion (+0.000), diffuse findings that
  gain from seeing every slice, held or improved. The champion reads every slice at 0.625 mm and
  per-label read 75% of slices at 0.5 mm. The architectures differ, so this is not a controlled
  resolution test, but the per-target pattern matches it.
- **Across the crop runs, two properties separate the good runs from the rest:** ReXNet-200 sees
  most slices, and it sees them at 0.625 mm or finer. Learned targeting has never been worth more
  than about 0.005.
