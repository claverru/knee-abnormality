# 2026-09-16 per-label crops result

**A tie that moved accuracy between targets.** Selecting 224-pixel crops round-robin over
per-label value maps reached **0.891703** fold-0 macro AUC at epoch 8, level with
[crop224](2026-09-16-crop224-result.md) (+0.000079; paired bootstrap +0.0001, 95% interval
[−0.0029, +0.0031]) and with `00-baseline` (−0.001060; [−0.0049, +0.0028]). The three peripheral
targets crop224 lost recovered part of the gap, while effusion and contusion fell.

## Setup

- **Change from crop224:** [per-label round-robin selection](../../history/crop-selection-investigation.md#recipes).
  Each label keeps its own value map; the budget of `ceil(0.75 × kept canvases)` crops per study is
  split in equal label quotas, the least-served label taking its best cell among unused canvases.
  Crops are still pooled by all labels. Everything else matches crop224: 300-pixel canvases at
  0.5 mm, 224-pixel crops, at most one crop per canvas, batch 8, 10 epochs, seed 42, fold 0 of
  `all_5fold_v3`.
- **Runs:** `outputs/training/per-label-crops` trained epochs 0–7 until the Linux OOM killer ended it at the
  old 30 GiB WSL cap; `outputs/training/per-label-crops-resume` continued from `last.ckpt` through epoch 9
  with the same schedule and selected epoch 8 (epoch 9: 0.891538).
- **Cost:** about 200 s per epoch including validation (394 s for the two resumed epochs), 39 s
  for the final validation prediction (WSL now at
  47 GiB), 4,380 optimizer steps, peak 19.4 GiB CUDA.
- **Comparisons:** `scripts/tools/compare_fold_runs.py`, 2,000 paired report-group draws, in
  `outputs/analysis/p/per-label-vs-crop224-20260916` and `outputs/analysis/p/per-label-vs-00-baseline-20260916`.

## Training curve

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| per-label | 0.7778 | 0.8284 | 0.8493 | 0.8633 | 0.8736 | 0.8819 | 0.8887 | 0.8898 | **0.8917** | 0.8915 |
| crop224 | 0.7791 | 0.8367 | 0.8528 | 0.8658 | 0.8785 | 0.8818 | 0.8899 | 0.8888 | 0.8909 | **0.8916** |
| `00-baseline` | 0.8025 | 0.8422 | 0.8561 | 0.8696 | 0.8787 | 0.8888 | 0.8927 | 0.8927 | 0.8920 | **0.8928** |

The draft ended at 0.626 (crop224: 0.633).

## Per-target AUC

Reversed pairs use the [fold-0 pair counts](../target-level-diagnostics.md#cost-of-a-reversed-pair).

| Target | `00-baseline` | crop224 | per-label | vs crop224 | vs baseline | Pairs vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| Fracture | 0.9301 | 0.9159 | 0.9216 | **+0.0058** | −0.0085 | −450 |
| Baker's | 0.9150 | 0.9088 | 0.9111 | **+0.0023** | −0.0039 | −577 |
| PF OA | 0.8715 | 0.8669 | 0.8687 | **+0.0018** | −0.0028 | −573 |
| Medial OA | 0.9045 | 0.9052 | 0.9102 | +0.0050 | +0.0057 | +1,076 |
| Synovitis | 0.8911 | 0.8950 | 0.8988 | +0.0038 | +0.0077 | +701 |
| Medial Meniscus | 0.9239 | 0.9214 | 0.9248 | +0.0034 | +0.0009 | +176 |
| ACL | 0.9146 | 0.9227 | 0.9257 | +0.0030 | +0.0111 | +1,464 |
| Lateral OA | 0.8662 | 0.8685 | 0.8660 | −0.0025 | −0.0002 | −28 |
| MCL | 0.8424 | 0.8401 | 0.8359 | −0.0041 | −0.0064 | −665 |
| Lateral Meniscus | 0.8939 | 0.8877 | 0.8837 | −0.0040 | −0.0102 | −1,076 |
| Contusion | 0.8899 | 0.8923 | 0.8873 | −0.0050 | −0.0026 | −295 |
| Effusion | 0.8702 | 0.8750 | 0.8667 | **−0.0084** | −0.0035 | −698 |

Gold-label macro AUC recovered to 0.9268 (crop224 0.9166, baseline 0.9320); report-derived stayed
at 0.8910. Soft-target BCE is 0.44834 (crop224 0.44803). Paired intervals are wide: meniscus pair
−0.0002 [−0.0072, +0.0061] and OA pair +0.0013 [−0.0045, +0.0070] against crop224.

## Placement

`scripts/adhoc/analyze_crop_diversity.py` on the same 48 validation studies (seed 20260916),
before (`outputs/analysis/c/crop-diversity-20260916`) and after
(`outputs/analysis/c/crop-diversity-per-label-20260916`):

| Plane | Cell entropy, bits | Distinct cells of 36 | Modal cell share | Offset std (x, y) mm |
|---|---|---|---|---|
| Sagittal | 4.10 → 4.29 | 32 → 30 | 0.13 → 0.14 | (7.7, 10.4) → (9.9, 10.3) |
| Coronal | 3.76 → 3.91 | 31 → 29 | 0.15 → 0.22 | (5.8, 10.9) → (6.8, 10.3) |
| Axial | 3.28 → 4.05 | 23 → 31 | 0.27 → 0.20 | (5.2, 7.6) → (7.6, 9.0) |

Crops spread more within a study, but still barely respond to the study's findings: between-study
spread stays at 3–6 mm, and crop centres of studies with and without a finding still differ by only
0.1–3.6 mm (fracture 3.6, Baker's 2.9, the rest below 2.3). Per-label argmax cells remain far apart
(52.1 mm), so the label maps act as label-specific location priors rather than study-specific
detectors.

## Ensembles with the full-triplet models

Logit averages of saved fold-0 validation predictions, computed on 2026-09-16:

| Models | Macro AUC |
|---|---:|
| Champion + `00-baseline` (two full-triplet runs) | 0.896032 |
| Champion + per-label crops | 0.896390 |
| `00-baseline` + per-label crops | 0.896393 |
| Champion + crop224 | 0.896003 |
| crop224 + per-label crops | 0.894364 |
| Champion + `00-baseline` + per-label crops | 0.897562 |

A crop model adds about as much to the champion as a second full-triplet run does (+0.00036 more).
Against champion + `00-baseline`, the crop pair is higher on ACL (+0.0079), medial OA (+0.0041)
and synovitis (+0.0033) and lower on lateral meniscus (−0.0040), fracture (−0.0035) and MCL
(−0.0023), the same geographic split as the single runs.

## Visual review of fracture cases

Every canvas and crop of two fracture-positive studies was rendered with `last.ckpt` (epoch 9) in
`outputs/previews/per-label-crops-fracture-20260916/` and read by eye; this is not a radiological
read.

- **Detected (…66582736, predicted 0.92; also ACL, effusion and contusion):** extensive
  proximal-tibia marrow oedema on coronal and sagittal fluid slices was covered by crops chosen by
  lateral OA, MCL, contusion, medial OA and ACL, while fracture's own two crops (axial joint-line and
  suprapatellar slices) mostly missed it. The detection came through shared pooling.
- **Missed (…61922832, predicted 0.10; effusion and fracture, report-derived 0.91):** the findings
  are subtle (small bright foci in the lateral proximal tibia on central coronal fluid slices), and
  those slices were cropped by other labels, so the miss is recognition, not coverage.
- **Fracture's crops** went to superior axial slices in both studies: its map is a plane prior,
  not a tibial-plateau detector.
- **Uncropped canvases** (25%) are mostly edge slices: anterior patella, posterior muscle and the
  outermost sagittal slices. In-plane placement stays near the centre, and superior axial slices
  sit off-centre relative to the knee because every canvas is centred on one study point.

## Reading

- **The mechanism did what it was built for.** Quotas gave peripheral labels their own crops, and
  fracture, Baker's cyst and patellofemoral OA recovered part of crop224's loss.
- **Diffuse findings paid for it.** Effusion and contusion benefited from the large central crops
  of the summed map; with about two crops each in training, and crops placed for other labels
  shared with them, they lost most.
- **Net zero.** This is a redistribution between targets. Label-private pooling would stop it,
  but the detected fracture case shows shared crops carrying correlated evidence, so private
  pooling risks losing more than it keeps.
- These are single-seed fold-0 shifts with wide intervals; the direction matches the mechanism,
  but no single target's change is established.
