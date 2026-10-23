# ResNet-50 input sampling campaign: prior work

Fold 1 of `all_5fold_scanner_v2`, seed 1110484336, one run per design unless said. Measured on
2026-10-11; details and tables in the
[soft-tissue reads run notes](../soft-tissue-reads/run-notes.md).

## The reference

`outputs/training/rn50-detail-view-teacher-025`: the sagittal fluid detail-view recipe on a
ResNet-50, labels 0.75 silver + 0.25 raw joint teacher (the five ResNet-50 teachers of
`rn50-distillation-teachers`). **×1 0.927290 (epoch 10), ×4 0.935570, 2,650 s of training**, 96 s
of ×4 prediction, peak GPU memory 16.5 GB, lowest available RAM 8.0 GB. Against the same recipe
without teachers: +0.0021 at ×1 and +0.0023 at ×4. Public 0.960 (S143), against 0.959 for the same recipe
without teachers (S142) and 0.960 for the 0.5-share child of the earlier recipe (S138).

- **The recipe.** Six series types (axial, coronal and sagittal, each T1/PD and fluid) at fields
  of 120, 84 or 93 mm in 224 px images; 26 slices per study plus a second 64 mm view of the five
  sagittal fluid slices (31.7 images per study in training, 123 at ×4); crops posed by the first
  look; a fourth-stage read with regions, a third-stage cell read routed by a soft-tissue map,
  bone routing of Contusion and Fracture; the soft-tissue and bone maps as side tasks.
- **The same recipe without teachers** (`rn50-sagittal-fluid-detail-view`): ×1 0.925201, ×4
  0.933292, 2,651 s, public 0.959 (S142). Peak GPU memory 16.5 GB; lowest available RAM 7 GB.

## What reaches the ResNet-50 and what does not

| Change | ResNet-18 at ×4 | ResNet-50 at ×4 | Public, ResNet-50 |
|---|---|---|---|
| 120 mm field, 84/93 mm views, view label bias | +0.0026 | +0.0027 | 0.948 to 0.954 |
| Distillation at 0.5 teacher share | +0.0015 | +0.0024 to +0.0034, five of five folds | 0.960 (S138) |
| Anatomy map and posed crops | +0.0020 | +0.0001 (+0.0017 at ×1) | 0.957 to 0.959 |
| View-poses bundle | +0.0010 | 0.0000 | – |
| Soft-tissue cell read and detail view | +0.0013 | +0.00004 | 0.959 = 0.959 |

- Four ResNet-50 recipes on fold 1 score 0.933209, 0.933265, 0.933256 and 0.933292 at ×4.
- The changes that did not carry gained on the four rare labels of the ResNet-18 (MCL, Lateral
  OA, Contusion, Fracture: +0.0111 over three generations, +0.0010 on the other eight) and
  re-present pixels the model already has. The two that carried add information.
- On the ResNet-50 itself ×4 over ×1 is +0.0081 (0.925201 to 0.933292).

## Acquired against fed (24,371 training series)

| Series type | Acquired finer axis, quartiles (mm) | Fed (mm per pixel, field) | Fed finer than both acquired axes | Crop share of stored field |
|---|---|---|---|---|
| Axial T1/PD | 0.375 / 0.469 / 0.500 | 0.536, 120 mm | 10% | 56% |
| Axial fluid | 0.469 / 0.500 / 0.562 | 0.536, 120 mm | 31% | 56% |
| Coronal T1/PD | 0.417 / 0.449 / 0.500 | 0.375, 84 mm | 84% | 28% |
| Coronal fluid | 0.449 / 0.500 / 0.563 | 0.417, 93 mm | 83% | 34% |
| Sagittal T1/PD | 0.400 / 0.451 / 0.521 | 0.417, 93 mm | 59% | 34% |
| Sagittal fluid, wide | 0.443 / 0.500 / 0.593 | 0.375, 84 mm | 90% | 28% |
| Sagittal fluid, detail | same | 0.286, 64 mm | 98% | 16% |

- **Only axial is fed below its acquired resolution.** Share of axial fluid series whose finer
  axis is kept: 35% at 120 mm, 64% at 112 mm, 85% at 100 mm, 91% at 93 mm (T1/PD: 11%, 33%, 59%,
  66%). The finer axis is front to back in axial and sagittal, up and down in coronal.
- **Slices are the largest loss.** A training bag reads 21 to 23% of a series' slice positions
  (5.2 to 7.1 slices of 27 to 32), with 18 mm between picks against an acquired step of 3.3 to
  4.0 mm; ×4 reads 78 to 84%.
- **Field is the second.** Whole kneecap inside the sagittal crop in 7.8% of series at 84 mm and
  26% at 93 mm; 49% of patellar cartilage inside the 84 mm crop (36 knees); the collateral region
  inside the coronal fluid crop in 35% of series; more than 10 mm of posterior tissue cut in 42%
  of axial fluid series at 120 mm.
- **Errors follow slices, not field** (871 fold-1 studies): Effusion falls from 0.948 to 0.872
  and Synovitis from 0.914 to 0.848 from the third of studies with the most axial upper reach to
  the least; MCL from 0.976 to 0.860 by sagittal medial reach.
- 836 series in 659 studies are 3D (steps of 0.4 to 1.2 mm, 120 to 320 slices).
- With `repeated_middle` the loader decodes three slices per pick and the model reads one.

## Precedents

| Mechanism | Backbone | Result at ×4 | Source |
|---|---|---|---|
| `study_slice_total` 26 to 36 | ResNet-18 | +0.0018 (OA up, menisci down) | `docs/validation/results` |
| Dense context slices, a third of the bag without gradient | ResNet-18 | −0.0023; Fracture, Contusion, MCL −0.008 to −0.012 | soft-tissue reads, run 23 |
| Sagittal fluid detail view, 64 mm on the same slices | ResNet-18 | +0.0006 (adopted); its three variants at the plain recipe's level | runs 29 to 32 |
| The detail view on three of its five slices | ResNet-18 | −0.0011 against the detail view | run 31 |
| An 84 mm patellofemoral axial view taking three slices | ResNet-18 | PF OA +0.0027, macro −0.0013 | `docs/validation/results` |
| A Baker's-only 120 mm view | ResNet-18 | Baker's +0.0023, macro level | `docs/validation/results` |
| Axial at 130 and 140 mm against 120 mm | ResNet-18 | level | `docs/validation/results` |
| Contrast mixing with an aligned donor | ResNet-18 | −0.0008; aligned and feathered −0.0011; mixing off −0.0025 | fixes-and-removals, F02 |
| Centred token grid at 257 px (9 by 9 and inner 7 by 7) | ResNet-18 | −0.0004 and −0.0005, public 0.954 = 0.954, 1.2 of the time | `docs/validation/results/crops-and-views/2026-10-08-token-grid-alignment.md` |
| Centred token grid at 225 px | ResNet-18 | −0.0024, cause not established | same |

## The token grid at 224 px

The ResNet keeps every second position from the first at each of its five stride-2 steps, so
the 7 by 7 tokens are centred at pixels 0, 32, …, 192: 15.5 px up and left of the crop's centre
(5.8 to 8.3 mm), with a token on the left and top borders and none within 31 px of the right and
bottom ones. Border tokens are valid 60% of the time in training and always at prediction. In
`resnet50.a1_in1k` the last stage's first block is a bottleneck whose 3 by 3 convolution has
stride 2 and padding 1, and whose shortcut is a 1 by 1 convolution with stride 2.

## Dropped before the start

- **Fields set to each type's median acquired pixel** (100 to 112 mm): the 84/93 mm table already
  keeps about 90% of series, and the 120 mm recipe lost 0.0022 to 0.0027 against it.
- **Anisotropic resampling:** the fed grid is already finer than the finer axis in coronal and
  sagittal; in axial the finer axis is the one that needs the field.
- **Detail-view variants on other planes at about 1.2 of the time** (the user, 2026-10-11).
