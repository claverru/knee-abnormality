# View optimisation

Started 2026-10-02, after the [smart-crops campaign](../validation/results/2026-10-02-smart-crops-campaign.md).
Nothing is built yet. Detailed evidence behind every statement here:
[view effects by target](../../autoresearch/campaigns/smart-crops/view-effects-by-target.md).

## Goal

One recipe that keeps every per-label gain found in smart-crops, searched as an optimisation
problem and not as one hand-built idea at a time. Starting point: the recipe in force (coronal
and sagittal T1/PD at 84 mm, fluid at 93 mm, axial at 120 mm; ×4 0.932880, 0.950 public as S93).

## What we carry over

- Replacing a view by a finer one gains; adding views whose tokens enter every label's pool does
  not (fifteen ties).
- Effects add up across series types (fluid two-by-two: +0.0003, +0.0005, +0.0010 together), so
  types can be tuned in parallel.
- The conflicts are between labels inside one type: sagittal fluid must be fine for five labels
  and wide for Baker's cyst, and no single width between 93 and 120 mm serves both.
- Placement by rule is enough; the sagittal forward shift earns its place; axial width between
  120 and 140 mm does not matter.
- Supply of slices never limits the budget: every type has four to five times what it is given.
- Inside the tested widths the surface is nearly flat around the recipe in force: the regression
  predicts every single-type change 0.0001 to 0.0005 lower. One run resolves about 0.001 on the
  macro.
- Much of the space is untried: axial finer than 120 mm, fluid finer than 93 mm, T1/PD finer
  than 84 mm, any budget other than 6/5/5 slices per type.

## Decision: learnable first, configuration as the fallback

User, 2026-10-02: try the full learnable setup, except sampling; configuration experiments
remain available on the same code.

| Thing | How it is learned |
|---|---|
| Spacing of each view | one scale per view of each series type, shared by all studies, bounded (about 0.31 to 0.63 mm) |
| Shift of each view | two offsets per view in anatomical millimetres from its anchor, bounded (about ±20 mm) |
| Which labels read which view | a bias per label and view on the attention score, starting at zero |

Stays configuration: how many views a series type has, the image size (up to 224 px), the
slices per type, and whether a second view reads the same slices as the first or its own. Every
learned value can also be frozen from the configuration, which is what makes the fallback free.

The parameters are global per series type, not predicted per image: per-image placement is what
failed in smart-crops and in the literature.

## Build (separate worktree off main)

- The window is cut on the GPU inside the model, from a wider and finer canvas that the loader
  ships; augmentation and the zoom become one resampling step.
- Shift settings per view: anchor (image centre or fitted knee centre), offset per axis in
  anatomical directions, fallback without trusted geometry, clipping that moves and never
  shrinks the window, optional training jitter.
- Token positions, the meniscus own-side mask and the OA regions are recomputed every step from
  the current window.
- Slice selection gets its own radius setting; today it follows the 120 mm spacing.
- A scale-aware view embedding, since one type can appear at several pixel sizes.
- Kaggle prediction handles any such table with the learned values frozen.

The new resampling path cannot reproduce S93 bit for bit. The check is a retrain with the values
frozen at the recipe in force landing at S93's level; that run is the new control. The extra
training time of the canvas is measured on it against the time bound.

## First runs

1. **Frozen at the recipe in force:** the control.
2. **Learnable, started from the recipe in force:** do the spacings and shifts move, and does it
   help?
3. **Learnable, started from 120 mm on every type:** the decisive one. If the views move towards
   what smart-crops found by hand, learning works. If they stay at 120 mm, the search goes back
   to configuration experiments on the same code.

The learned values themselves are the reading, not the score alone: the effect to be found is
about +0.001 to +0.002, at the edge of what one run shows.

## Fallback: configuration experiments

The view table is on main since 2026-10-03 (built on the `view-config` branch on 2026-10-02),
independently of the learnable build and on main's loader-side cut, without a canvas. Every plane and contrast lists its views in the config:
per view a spacing, a slice budget and a placement (image or knee anchor and an offset in
millimetres per axis); a type may list several views, each with its own budget and slices.
Settings and rules: [view table](../modeling/current-model.md#view-table).

- **The control stands.** The recipe in force written as a view table gives bit-identical
  loader output (CPU, 200 training studies, four sampling modes), and the model treats a
  centred 120 mm view and a finer view exactly as before, so ×4 0.932880 stays the reference
  and no control retrain is needed. Rechecked before the merge on the three configs
  (ResNet-18, Refined ResNet-50, ReXNet-200; training, ×1 and ×4 validation studies): the
  same slices, and images, geometry, features and outputs with a difference of 0.0 for the
  same initial weights. A checkpoint for Kaggle still has to be trained on this
  code, because the preprocessing identity hashes the sources.
- **Checked on CPU only** beyond the control recipe: a config with a knee-anchored 100 mm
  axial view, a second 84 mm sagittal T1/PD view 15 mm above the image centre and a second
  120 mm sagittal fluid view of two slices runs through the loader and the model, with the
  window centres where the rule puts them. No training run exists.
- **Limits.** Every window stays inside the 120 mm frame, so a 120 mm view cannot shift and a
  wider one is centred. Every view's tokens enter every label's pool: routing a view to
  chosen labels (batch 3 below) is not built. A view table is set in the YAML, not by a
  dotted flag.

1. A balanced batch on widths (six to eight runs): each run gives every type one of four or five
   widths, arranged so each type's response curve per label comes out of the batch, including
   the untried cells. These runs inform; they are not candidates.
2. A budget batch: reallocations at 26 slices, taken from the per-type sensitivity measured on
   Kaggle (in progress, `outputs/analysis/slice-budget-20261002/sensitivity/`).
3. A routing batch: a wide sagittal fluid view for Baker's cyst, compartment crops for the OA
   labels.
4. The width regression (`outputs/analysis/view-regression-20261002/report.md`, refit with
   `scripts/adhoc/fit_view_width_regression.py`) as the surrogate that proposes the combination;
   confirm it with a rerun, then outside fold 0.

Carried over from smart-crops, not run there: coronal T1/PD back at 120 mm with sagittal kept at
84 mm, T1/PD at 93 mm, compartment crops routed to the OA labels, the sagittal windows without
the 10 mm downward shift, separate normalisation statistics per pixel size.

Limits stay as they are: no image above 224 px, 26 slices per study, training time within 1.5
times the control's.

## Open

- A control retrain of the recipe in force from main (user launches).
- How to confirm the final recipe outside fold 0: a second fold, or a public submission as with
  S93. About 30 choices have been made on the same 912 studies.
