# 2026-10-08 token grid alignment: 224 px against 225 and 257 px

**The ResNet's token grid is off-centre at 224 px, and neither centring it nor removing the
tokens on the image border changes the score.** Two 257 px runs with a centred grid match the
224 px recipe within 0.0005 at ×4 and 0.0001 at ×1 at about 20% more time; a 225 px run was
0.0024 lower for a reason not established. Both 257 px models scored 0.954 public (S127, S128),
the same as the 224 px recipe (S125). The recipe stays at 224 px. Nothing was adopted and no
code was kept.

## The 224 px grid

The ResNet halves the resolution five times and keeps every second position starting from the
first, so its tokens are centred at input pixels 0, 32, ..., 192 (`BACKBONE_MAPPINGS`: origin 0,
stride 32; checked against the layers).

- **Off-centre:** the grid's centre is pixel 96 and the crop's centre, where the knee is placed,
  is pixel 111.5. What the model reads sits 15.5 px up and left of the knee: 5.8 mm on the 84 mm
  views, 6.5 mm on the 93.3 mm views, 8.3 mm on the 120 mm views.
- **Uneven edges:** a token sits on the left and top border; on the right and bottom the nearest
  token is 31 px inside and the last 16 px belong to no token.
- **Left and right knees differ:** medial falls on opposite image sides, so one side has a token
  on the medial border and the other does not. On the coronal medial edge line MCL puts 58 to 61%
  of its attention for one image side and 21 to 22% for the other (3,517 training studies,
  `outputs/analysis/v/view-attention/`).
- **Border tokens differ between training and prediction:** a token centred on a border pixel
  fails the inside-the-image test whenever the affine augmentation (70% of samples: rotation up to
  10°, shift up to 4%, scale 0.95 to 1.05) moves the image away from that edge. In training the
  first row and column are valid 60% of the time and the last 94%; at prediction every token is
  valid (`outputs/analysis/a/audit-225/`).
- **Reflection:** the left-right reflection (probability 0.5, axial and coronal images mirrored
  about the crop's centre, sagittal slice order reversed) moves the knee 31 px across the grid's
  centre, so training sees both placements in the left-right direction of axial and coronal
  images. It does nothing for the up-down offset.

A symmetric grid exists only at image sizes of 32k + 1 (193, 225, 257), where the first and the
last token both sit on a border pixel.

Drawing: the grid at 224 and 225 px (`outputs/analysis/c/centred-token-grid/token-grid.png`, deleted).

## Runs

ResNet-18 recipe with the anatomy map and posed crops, fold 1 of `all_5fold_scanner_v2`, seed
1110484336, ten epochs, `trainer.report_attribute_loss_weight: 0`. The reference is the
anatomy-recipe campaign's incumbent `report-attribute-loss-off-2`, the same recipe at 224 px.
Worktree `side-routed-heads` at `64e905f` plus uncommitted changes; each run's
`source_snapshot.zip` holds its code. Runs under `outputs/training/`; checkpoints deleted except
the reference's and the two submitted bundles'.

| Run | Image, tokens counted | Grid centred | Border tokens | ×4 | Δ | ×1 | Δ | Training | Public |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `report-attribute-loss-off-2` (reference) | 224 px, 7×7 | no | first row and column | 0.927291 | – | 0.917342 | – | 1,352 s | – |
| `rn18-grid-225-report-loss-off` | 225 px, 8×8, fields kept | yes | whole ring | 0.924867 | −0.0024 | 0.914928 | −0.0024 | 1,401 s | – |
| `rn18-grid-225-fixed-spacing-report-loss-off` | 225 px, 8×8, spacings kept | yes | whole ring | – | – | 0.914759 at epoch 9 | – | stopped in epoch 10 | – |
| `rn18-257-inner-7x7-report-loss-off` | 257 px, inner 7×7, spacings kept | yes | none | 0.926763 | −0.0005 | 0.917439 | +0.0001 | 1,615 s | 0.954 (S127) |
| `rn18-257-full-9x9-report-loss-off` | 257 px, 9×9, spacings kept | yes | whole ring | 0.926908 | −0.0004 | 0.917344 | +0.0000 | 1,651 s | 0.954 (S128) |

The public reference is S125 (0.954), the 224 px ResNet-18 of the recipe with the report loss at
0.1; no 224 px model with the loss at 0 was submitted.

- **225 px:** one more row and column of tokens, centred on the last pixel. Keeping the fields
  changes the spacing by 0.44%; the run with the 224 px spacings (fields 84.375, 93.75 and
  120.54 mm) traced the first one within 0.0013 at every epoch, so the spacing is not the cause.
  Both curves dip at epoch 3 and peak at epoch 5, which no 224 or 257 px run does.
- **257 px, inner 7×7:** spacings kept (fields 96.375, 107.08 and 137.68 mm), and a rule that a
  token centred on a border pixel of the frame never counts. The 7×7 tokens left are at pixels 32
  to 224, centred on the image centre, fully inside the image, and read the same area as the
  224 px crop with 16 px of real image around it. At 224 px the same rule would give 6×6; not
  run.
- **257 px, 9×9:** the same without the rule.
- **Cost of 257 px:** +19 to 22% training time, +19 to 24% ×4 prediction time (60.6 and 62.8 s
  against 50.8 s), 5.3 GB of GPU memory against about 4 GB.

Validation curves (`outputs/analysis/c/centred-token-grid/validation-curves.png`, deleted) of the
reference, the campaign control, both 225 px runs and the 257 px inner 7×7 run.

## Per label at ×4 against the 224 px reference

| Label | 257 px inner 7×7 | 257 px 9×9 | 225 px |
|---|---:|---:|---:|
| ACL | +0.0074 | +0.0079 | +0.0023 |
| MCL | −0.0061 | −0.0073 | −0.0122 |
| Medial Meniscus | +0.0030 | +0.0038 | +0.0017 |
| Lateral Meniscus | +0.0026 | −0.0034 | +0.0018 |
| Medial OA | −0.0008 | −0.0011 | −0.0051 |
| Lateral OA | −0.0037 | −0.0033 | −0.0029 |
| PF OA | +0.0022 | +0.0050 | +0.0002 |
| Effusion | −0.0018 | −0.0009 | +0.0013 |
| Synovitis | +0.0062 | +0.0049 | −0.0036 |
| Baker's | −0.0053 | −0.0061 | −0.0039 |
| Contusion | +0.0017 | +0.0027 | −0.0015 |
| Fracture | −0.0117 | −0.0069 | −0.0072 |

The two 257 px runs share an augmentation stream, so their agreement is one observation. Their
fields are 15% wider than at 224 px, yet MCL and Baker's, read at the crop's edges, fell.

## What is and is not established

- **Centring the grid does not change the score** (224 px against both 257 px runs).
- **Border tokens do not change the score** at 257 px, counted or not (0.0001 apart).
- **The 225 px drop is unexplained.** Spacing, the odd size and the border tokens are ruled out
  by the other runs. The audit of the 225 px run found no implementation defect (token positions,
  pitch from the stride, turned and unturned crops, the augmentation's inverse, anatomy targets,
  pooling and the loader all correct at 225 px). The remaining candidate is the augmentation
  stream: the image-shaped noise draw advances the GPU generator by an amount that depends on the
  image size, so runs at different sizes draw different augmentations, and both 225 px runs
  shared one stream. A 225 px run with another seed would settle it; not run.
- **Run-to-run spread seen here:** two launches of the 257 px inner 7×7 run (one interrupted)
  differed by 0.0008 at epoch 1; two trainings of the 224 px recipe differed by 0.0002 at ×4
  earlier the same day.

## Prediction-only test on the 224 px reference

Moving the second-look crop 16 px so the grid is centred on the knee, on the trained reference
(874 validation studies): ×4 −0.0004, ×1 −0.0018; the same move in the opposite direction ×4
−0.0008, ×1 +0.0004; with axial included (a 16 px empty band, since axial has no canvas margin)
×4 −0.0016. The weights were trained with the off-centre grid and with reflection, so this says
nothing about the fix. Predictions: `outputs/analysis/c/centred-token-grid/`.

## Left open

- `scripts/tools/check_side_routing.py` takes the token pitch as `reference_field_mm / grid`,
  which is right only for 7×7 tokens at 224 px; the model itself takes it from the stride.
- The 6×6 version at 224 px (first row and column never counted) was not run.
- The one-off scripts of these runs and of the audit were in the session's scratch folder and
  were lost in a WSL restart; outputs and figures remain.
