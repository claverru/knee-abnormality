# Meniscus masks from DESS

## 8. Meniscus masks from DESS: built (2026-10-08)

Decided with the user: `aagatti/nnunet_knee` on the DESS series, menisci only (cartilage dropped),
kept as a second target set beside the bone masks, supervised on every series of the study
through the shared patient coordinates. Head: its own small head with its own loss on its
own knees, on the backbone's layer 3. Training code built on 2026-10-08 (below).

- Tool: `scripts/tools/meniscus_targets.py` (`convert`, `segment`, `export`); builders in
  `scripts/adhoc/meniscus_masks/`; working folder `~/.cache/knee-segmenters/dess-menisci/`.
- Convert: 265 of 265 studies (the DESS series with most slices), 0.5 mm in-plane, cropped to the
  body with a 10 mm margin, turned to the axis order of the model's test image (checked by eye on
  five knees); 71 s, 3.1 GB.
- Segment: full-resolution model, fold 1, no mirroring; 265 volumes in 1,567 s (5.9 s each),
  2.7 GB of GPU memory.
- Export: `data/labels/prepared/meniscus-nnunet-knee-dess265/`, 1 mm volumes in patient
  coordinates (0 neither, 1 medial, 2 lateral, 255 outside the DESS volume), `knees.json` with the
  test per knee. **263 of 265 accepted**; per fold 53 / 52 / 40 / 59 / 59. Rejected: one medial
  meniscus in pieces (main piece 0.60), one knee at the scanner's centre so its side could not be
  checked. No knee had its sides swapped.
- Test (label-free, thresholds chosen by hand): volume 0.8 to 7 ml, main piece at least 0.8 of the
  class, median distance to the mask's own tibia at most 8 mm and femur at most 10 mm, medial
  meniscus nearer the body's midline than the lateral one. Measured: medial 2.30 ml (1.11 to
  4.81), lateral 2.10 ml (1.20 to 4.15), distance to the tibia 3.6 and 4.2 mm, menisci 50 mm
  apart. Nothing came near a threshold except the two rejected knees.
- Check on 2D series: `outputs/analysis/soft-tissue-segmenters/meniscus-targets-on-2d-series.png`
  (five knees, coronal fluid and two sagittal slices each): the targets land on the menisci of the
  2D series, on the right sides.
- 1 mm serves any grid down to 4 mm cells (21×21 at 84 mm): the last stage and layer 3, not
  layer 2. 794 MB on the GPU for the 263 volumes.

**Training code.** On in `configs/resnet50.yaml`
(`model.meniscus_map`, `paths.meniscus_targets`, `trainer.meniscus_loss_weight` 0.1; weight
copied from the bone map). A layer-3 version (14×14 cells) was built first and removed the same
day: the user wants the head on the 7×7 tokens and inside the model, to build on its output.

- `modeling/model.py`: with `model.meniscus_map`, a linear head with three outputs on the same
  7×7 tokens the bone head reads (before the position code); its scores are returned as
  `encoded["meniscus_scores"]`. It starts at zero: created inside the usual forked random state it
  would otherwise start equal to the first three rows of the bone head. Nothing reads it yet.
- `training/lightning.py`: the same loss form as the bone map (cross-entropy on the shares plus
  Dice), on the tokens of knees that have a meniscus volume and lie inside it, every series of the
  study. Logged as `meniscus_*` and `val_meniscus_*`. The switch and the weight must be set
  together.
- `training/anatomy.py`: the target reader and the loss take the class list and the number of
  sample points per token side: 4 for the bones as before, 8 for the menisci.
- CPU check on four fold-1 validation studies, untrained weights: 970 to 1,260 supervised tokens
  per DESS knee and none on the study without a mask; 45 to 80 tokens per knee hold some meniscus,
  13 to 41 more than a tenth, largest share 0.44 to 0.64, mean share of a meniscus token 0.057.
  The label logits are identical with and without the head (difference 0.0).
  Against a dense 24×24 lookup, the share error on meniscus tokens is 0.015 with 4×4 points
  (largest 0.13) and 0.005 with 8×8 (largest 0.05).

**First run: `outputs/training/rn18-meniscus-map`** (2026-10-08; `configs/resnet50.yaml` with the
meniscus map on, fold 1, seed 1110484336). Reference: the incumbent
`report-attribute-loss-off-2`, same fold, seed and recipe without the meniscus map. One pair of
runs.

| | Run | Reference | Δ |
|---|---:|---:|---:|
| ×4 | 0.926413 | 0.927291 | −0.00088 (paired bootstrap 95% −0.0020 to +0.0002) |
| ×1, best epoch | 0.916382 (epoch 9) | 0.917342 (epoch 9) | −0.00096 |

- Training 1,371 s against 1,352 s; ×4 prediction 51.4 s; peak GPU memory 4.8 GB.
- Per label at ×4: ACL +0.0003, Medial OA −0.0002, Baker's −0.0002, Lateral Meniscus −0.0003,
  PF OA −0.0004, Medial Meniscus −0.0007, Lateral OA −0.0008, Synovitis −0.0011, Fracture −0.0011,
  Contusion −0.0012, Effusion −0.0015, MCL −0.0032. Family intervals (meniscus, OA, other eight)
  all include zero. Predictions moved little: mean absolute change 0.0098, rank correlation per
  label 0.989 to 0.998.
- ×1 per epoch against the reference: −0.0012, +0.0001, −0.0007, −0.0002, −0.0013, −0.0005,
  −0.0002, −0.0011, −0.0010, −0.0009. Contusion was below the reference from epoch 4 to 10 and
  Fracture in five of the last six epochs at ×1.
- Meniscus map, validation overlap (52 DESS knees), medial / lateral per epoch: 0.083 / 0.096,
  0.256 / 0.350, 0.355 / 0.391, 0.406 / 0.439, 0.429 / 0.426, 0.430 / 0.412, 0.449 / 0.420,
  0.446 / 0.425, 0.437 / 0.432, 0.452 / 0.433. Flat from epoch 5; training overlap ends at 0.439
  and 0.434, so it is not overfitting, it is the ceiling of this head on this grid.
- Bone map unchanged: final validation overlap femur 0.904, tibia 0.898, kneecap 0.738, fibula
  0.697, joint 0.758, kneecap joint 0.670, each within 0.004 of the reference; plane error 1.74 mm
  against 1.78 mm.
- 3D page: `outputs/analysis/rn18-meniscus-map-3d/learned-map-3d.html` (and a preview PNG), made
  by `scripts/adhoc/meniscus_masks/learned_map_3d.py rn18-meniscus-map rn18-meniscus-map-3d`: the
  four knees of the earlier `anatomy-masks-frame-3d` page, none with a DESS series, plus two DESS
  validation knees with the segmenter's masks. Four passes per knee (102 to 128 slices), token
  shares placed on a 3 mm grid and smoothed with a 5 mm Gaussian before taking each voxel's main
  class; a first version with one pass and no smoothing drew ragged bones. On all six knees the meniscus map sits at the joint level, medial on the
  medial side and lateral on the lateral side, in arcs seen from above; fused shares reach 0.18 to
  0.68 (weakest: the lateral meniscus of …259018 and the medial one of DESS knee …753617), so the map transfers to studies without a DESS series.
- Comparison: `outputs/analysis/rn18-meniscus-map-vs-incumbent/comparison.json`.
- Lesson: the model learns where the menisci are from 211 one-scanner knees and applies it to
  2D-only studies, with the bone map untouched; as a side task alone it does not move the labels
  (level to slightly below, inside the interval). Nothing reads the map yet.
