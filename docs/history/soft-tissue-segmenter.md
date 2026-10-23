# Soft-tissue segmenter beyond DESS (closed)

> **Closed 2026-10-23, not adopted.** The fine-tune was never launched, and the segmenter environment under `~/.cache/knee-segmenters` was deleted on 2026-10-16 by the user's decision. The worktree was removed; its scripts (`scripts/adhoc/soft_tissue_segmenter/`) and this document's uncommitted state are archived in the main checkout at `outputs/archive/removed-worktrees-20261023/` (`soft-tissue-segmenter.patch`, `soft-tissue-segmenter-untracked.tgz`).

Started 2026-10-14 with the user. Goal: menisci and cartilage masks on far more studies than the
263 DESS knees the soft-tissue map trains on today (one site, Siemens Avanto fit; 223 training
studies on fold 2). Worktree `knee-abnormality-soft-tissue-segmenter`, branch
`soft-tissue-segmenter`, uncommitted; scripts in
`scripts/adhoc/soft_tissue_segmenter/` (archived), run with the
segmenter environment under `~/.cache/knee-segmenters` (`headers.py` and `sample_groups.py` with
the repository's). Earlier segmenter benchmarks: [view-pose studies, soft-tissue
segmenters](../research/view-pose-studies.md#3-soft-tissue-segmenters-back).

## Status

- `aagatti/nnunet_knee` unchanged does not carry beyond DESS (below). The fine-tune training set
  is built; training is **not launched** (user's order).

## How the tests measure

- **Paired test (Dice):** 19 DESS studies that also have a 2D sagittal fat-suppressed series. The
  model's mask on the DESS is the reference; the same model runs on the 2D series only, and both
  masks meet in patient coordinates. Dice is agreement with the DESS masks, not ground truth.
- **Reach test (acceptance):** the label-free test of `meniscus_masks/export_dess.py` (each
  meniscus 0.8 to 7 ml, one main piece ≥ 80%, within 8 mm of the tibia and 10 mm of the femur,
  medial nearer the midline).

## Findings

**Contrast blocks the transfer, not slice thickness.** The DESS cut into 3.5 mm sagittal slices
keeps menisci Dice 0.87 / 0.86 and cartilage 0.89 to 0.92 against the full DESS (15 of 19
accepted; 19 of 19 with B-spline slice interpolation). The real 2D sagittal FS of the same studies:
1 of 19 accepted, menisci Dice about 0.

| Input hack on the 19 paired 2D sagittals | Accepted | Menisci Dice | Femoral cartilage Dice |
|---|---|---|---|
| Native | 1 | 0.00 | 0.06 |
| Slices doubled (B-spline to 0.8 mm) | 1 | 0.00 | 0.03 |
| Intensity clipped at 2× / 2.5× the body median | 1 / 1 | 0.00 | 0.09 / 0.07 |
| Histogram matched to a DESS volume | 1 | 0.00 | 0.13 |
| Air padding with noise to the training field of view | 0 | 0.00 | 0.07 |
| In-plane smoothing σ 0.6 mm / curvature flow | 1 / 0 | 0.00 | 0.18 / 0.00 |
| Sagittal and coronal fused on one grid (with histogram matching) | 3 (2) | 0.00 (0.00 / 0.03) | 0.37 (0.39) |
| Meniscus classes ×e before the argmax | 1 | 0.00 | 0.03 |

Zero-padding is a no-op: nnU-Net crops zero borders before its whole-volume z-score.

**What separates the paired cases is the pulse sequence.** PD TSE FS at TE 9 and 35 to 36 ms:
menisci Dice 0.28 to 0.71; PD FS at TE 42 to 44 ms: 0.00 to 0.09; T2 TIRM (STIR, TE 74 to 79 ms,
9 of the 19) and T2 FS BLADE: 0. The TE 9 ms case matches the DESS masks closely
([drawing](../../outputs/analysis/s/soft-tissue-segmenters/2d-vs-dess/tissues_pd_fs_te9.png);
side-by-side images: `dess_vs_2d_sagittal_fs.png` in the same folder). The DESS site is
unrepresentative: STIR is 15 of 4,667 sagittal FS series overall.

**Reach of the unchanged model: 1 of 65.** Up to 3 random non-DESS studies per vendor × TE group
(24 groups, `sample_groups.py`, seed 20261014): only a Siemens Skyra at TE 27 ms passes. Menisci
are mostly found but undersized (0 to 0.6 ml against about 2.4 ml on DESS; the short-TE Prisma and
Aera series reach 0.9 to 1.3 ml on one side); femoral cartilage 4 to 10 ml against about 12.5 ml;
STIR and TE ≥ 60 ms give nothing. A meniscus bias before the argmax finds more menisci (×e³:
53 / 60 knees against 39 / 51) but accepts at most 2 of 65; the cartilage variants were stopped
unscored.

Sagittal FS series per study by best TE (headers of all 4,667 series, `headers.py`): TE < 20 ms
52 studies, 20 to 39 ms 2,725, 40 to 59 ms 812, ≥ 60 ms 551, STIR 10.

**The classifier's own soft-tissue map transfers off-site (by eye).** Main's model (run 27 of
targets-and-inputs, fold 2) drawn on 4 DESS and 8 non-DESS fold-2 validation knees
(`classifier_map.py`, ×4 slices, tokens with share ≥ 0.3, all views fused in patient coordinates):
on the DESS knees the tokens follow the segmenter masks; on the non-DESS knees (GE, Philips,
Siemens, Toshiba, Hitachi; TE 30 to ≥ 60 ms and STIR) every structure sits in place and on the
right side, somewhat sparser menisci on two knees and patellar-cartilage tokens spilling into the
fat pad on two. Cell resolution only; a judgement on 12 knees, not a measurement
(`outputs/analysis/s/soft-tissue-segmenters/classifier-map/`).

**How `nnunet_knee` was trained** (checkpoint and nnU-Net 2.8.1 code): stock `nnUNetTrainer`,
1,000 epochs, OAI-ZIB sagittal DESS only, fold 1 shipped; z-score over the whole volume; mild
intensity augmentation (brightness and contrast ±25%, gamma 0.7 to 1.5, low-resolution
simulation to ½); no contrast randomization.

## Fine-tune setup (built, not trained)

- **Data** (`finetune_data.py`, `Dataset501_KneeClinical` under
  `~/.cache/knee-segmenters/nnunet_data/nnUNet_raw`): per DESS study, cropped to the joint with a
  25 mm margin and put on the model's grid (0.8 × 0.3125 × 0.3125 mm): the DESS with its mask, the
  DESS cut to 3.5 mm in each plane, and every 2D fluid FS series rigidly registered to the DESS
  (Mattes MI; rejected beyond 10 mm or 10°; the first two studies moved under 1 mm) with the mask
  carried over and `ignore` outside the DESS slab. Held out for validation: the 19 paired studies
  and 21 random DESS studies (seed 20261014). Built 2026-10-14 in 12 minutes on five processes,
  21 GB: 1,359 cases (1,132 training, 227 validation) from 263 studies: DESS and three thick cuts
  each, 275 coronal, 20 sagittal and 12 axial fluid series. Registration moved the 2D series
  0.67 mm at the median (95th percentile 2.82 mm), 0.35° at the median; one coronal series out
  of range and one unreadable axial series dropped.
- **Trainer** (`nnUNetTrainerClinicalFT.py`, linked into the segmenter environment): the stock
  trainer plus a random intensity curve (6 points, p 0.5, monotone half the time) after the spatial
  transform; learning rate 3e-3, 100 epochs from the `nnunet_knee` fold-1 checkpoint.
- **Not yet done:** plans copied to the new dataset, preprocessing, `splits_final.json`, training,
  and the reach test with the fine-tuned model.
