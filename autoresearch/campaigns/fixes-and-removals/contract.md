# Fixes-and-removals campaign contract

## Objective

Repair defects in the current recipe, and test whether mechanisms that are active in it today
are still needed, one change per run. A removal switches off one active mechanism whose adoption
predates later changes to the recipe (backbone, weights, views, normalization, labels). The
campaign adds no modelling mechanism. Its sources are the external audit and the follow-up audit of 2026-10-05
([prior work](prior-work.md)).

The reference and starting incumbent is `rn18-fold1`
(`outputs/training/20261005-113728-rn18-fold1`, the `configs/training.yaml` recipe in force
trained from main on 2026-10-05 with a rotated fold and seed by user decision: fold 1 of
`all_5fold_scanner_v2`, seed 1110484336, ×4 0.923102, ×1 0.914860, 1,179 s). Experiments run on the ResNet-18; what is adopted is carried
to the ResNet-50 afterwards by the user.

## Evaluation

The [shared evaluation](../../protocol.md#shared-evaluation-and-adoption) applies against the
campaign incumbent. Every result also reports ×1 and the per-label deltas.

## Adoption

Fixed by the user on 2026-10-05 and applied by the keeper without asking: the runtime computes
its shared verdict, and where this section differs the keeper adopts with
`python -m scripts.tools.autoresearch adopt fixes-and-removals --experiment EXPERIMENT_ID`,
citing the rule below as the rationale. No decision of the campaign waits for the user.

| Kind | Seeds | Adopted when |
|---|---|---|
| Forced fix | 1 | Always, once the run is valid ([forced adoption](#forced-adoption)) |
| Fix | 2–5 | The observed ×4 macro delta against the incumbent is zero or positive |
| Replacement | 6 | The shared rule: the 80% interval is entirely above zero |
| Removal | 7–14 | The observed ×4 macro delta against the incumbent is zero or positive |

- **Fixes and removals never accept an observed loss, however small.** A change that does
  nothing measurable lands on either side of zero by chance, so some correct fixes and some
  useless mechanisms are not adopted by the campaign; they are listed for the user afterwards.
- A fix that is not adopted does not stop the seeds after it: each applies to the incumbent at
  its turn.
- **The incumbent never falls after seed 1:** every later adoption needs a delta of zero or
  more against the incumbent in force, so each new incumbent scores at least as high as the one
  before it. No separate guard is needed.
- A removal seed a researcher proposes later follows the removal row, a fix the fix row.
- Everything not adopted, with its deltas, goes into the findings for the user to read after the
  campaign.

## Forced adoption

**`reflected-detail-view-positions` is adopted by user decision whatever its delta** (user,
2026-10-05): it is a bug fix. It changes the positions coronal detail tokens get in reflected
training studies and in the mirrored TTA pass, so the run's delta is a real measurement and is
reported in full, but it does not decide the adoption. When the run
completes as valid, the keeper adopts it with
`python -m scripts.tools.autoresearch adopt fixes-and-removals --experiment EXPERIMENT_ID`
and the rationale "bug fix, adopted by user decision of 2026-10-05", and later seeds build on it.
An invalid run is repaired and requeued, not skipped.

## Seeds

Forced, in file order ([seeds 1](seed-ideas.yaml), [seeds 2](seed-ideas-2.yaml),
[seeds 3](seed-ideas-3.yaml)); scope fixed by the user on 2026-10-05. Each applies to the
incumbent at its turn.

Fixes:

1. `reflected-detail-view-positions`: a mirrored detail view gets the mirrored window position,
   in the training reflection and in reflection TTA.
2. `contrast-mixing-aligned-donor`: the donor slice is resampled onto the target grid.
3. `centre-without-side`: knee-anchored crops and contrast mixing stay on with a fitted centre
   and an unknown side.
4. `centre-outside-slab-rule`: the slab guard applies whatever the fit method.
5. `feature-statistics-positive-std`: the resampled standard deviation stays positive.

Replacement:

6. `computed-centre-3d-position`: one 3D position per token in place of the 1D and 2D
   embeddings, after the two geometry fixes.

Removals, each switching off one active mechanism:

7. `oa-detector-off`, after the 3D position because the two interacted before.
8. `intensity-augmentation-off`, 9. `noise-augmentation-off`.
10. `half-knee-loss-off`, 11. `meniscus-mask-off`.
12. `plane-readouts-off`, 13. `detail-view-embedding-off`.
14. `report-attribute-heads-off`.

**No seed may decode more slices than the incumbent does** (user, 2026-10-05). A window fixed
per series (audit item F06) is left out for that reason: it needs slices outside the bag.

Left out after checking earlier runs ([prior work](prior-work.md)): contrast mixing, bilateral
smoothing, GIN, resolution degradation, feature-statistics augmentation and the view label bias
as removals, all measured recently; affine augmentation and L2-SP anchoring by user decision.

## Seeds added during the run

Each is forced through the inbox and listed in the [index](INDEX.md).

- **`contrast-mixing-aligned-donor-requeue`** (keeper): seed 2 again, after its first run was
  cancelled to adopt seed 1.
- **`unadopted-fixes-together`** (user, 2026-10-05): the slab rule, the aligned contrast mixing
  and the positive std in one run; decided as a fix.
- **`contrast-mixing-aligned-feathered`** (keeper): the aligned blend without the coverage step;
  decided as a fix.
- **`position-3d-without-oa-detector`** (keeper): the 3D position with the detector removed;
  shared rule.
- **`slab-rule-and-positive-std`** (user, 2026-10-05): the two fixes too small for the fold to
  judge, without the aligned mixing; **adopted by user decision whatever its delta**, like seed
  1, with the rationale "correct by reading, adopted by user decision of 2026-10-05".
  Its first run was cancelled to adopt the OA removal; `slab-rule-and-positive-std-requeue`
  carries the same decision.
- **`oa-detector-off` adopted by user decision** (2026-10-05) at −0.000468 against an incumbent
  about 0.0006 above its do-nothing siblings; to be revisited
  ([run notes](run-notes.md#to-revisit)).
- **`position-3d-beside-1d-2d` and `position-embeddings-off`** (keeper, from the audit of the 3D
  runs): the first on the shared rule, the second decided as a removal.
- **3D retakes** (user, 2026-10-05, a scope extension by user decision: these add mechanisms):
  `position-3d-study-dropout`, `position-3d-attention-only` and `position-3d-coarse`, each the
  3D position with one change against the overfitting its first run showed; shared rule.

## Fixed

Labels, split, fold, seed, epochs, batch size, slices per study, the views, the optimizer and
schedule, the class-balance exponent, the series-cleaning rules and the evaluation. A fix
replaces the defective behaviour with no option to select the old one. A removal changes only
the one mechanism it names, through its existing off value; its code stays in the run and is
deleted by the user when an adopted removal is merged.

## Permitted

Source changes in `inference/reflection.py`, `modeling/`, `data/preprocessing/`,
`data/dataset.py`, `training/lightning.py` and `config.py`. For removals: the `model` subtree,
the seven augmentation subtrees under `data`, `trainer.half_knee_loss_weight`,
`trainer.report_attribute_loss_weight` and the two L2-SP anchor strengths. The preprocessing
identity may differ from the incumbent's.
