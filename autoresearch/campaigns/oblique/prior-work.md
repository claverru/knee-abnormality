# Oblique prior work

Evidence map for the [oblique campaign](contract.md); not a backlog. Analysis scripts and outputs
are in `outputs/analysis/acquisition-geometry-20260922/` unless stated.

## Reference run

`outputs/training/oblique`
([result](../../../docs/validation/results/2026-09-22-oblique-groups-result.md)): fold 0
0.897527 at ×1 τ 1 against 0.899042 for `k_05` (−0.0015, 95% [−0.0055, +0.0026]); ×4 τ 0.75
0.90467 against 0.904375; ×5 τ 0.70 0.90495; public S46 0.944 (×4 τ 0.75) and S47 0.945 (×5
τ 0.70) against 0.945 for `k_05`. On fold 0, 103 studies change; the mean study rises from 26.80 to 27.37 training triplets.
Its per-label ACL gain (+0.0147) sits in the unchanged studies, so it is training divergence; in
the split studies the deltas lean negative (MCL −0.058 on 11 positives, lateral meniscus −0.028 on
22).

## What the oblique groups are

- **Series** (`oblique_desc`, `oblique_slots`): 320 thin coronal slabs (median 16 slices, 45 mm,
  74% fluid), described as dedicated ACL obliques (`COR ACL T2W FS`, `T2 TSE FS COR OBL ACL`,
  `T2W Bondorf`); 457 full-coverage tilted coronals (25 slices, 86 mm, mostly a Siemens T1 TSE
  protocol); and 38 axial-labelled series that are coronal ACL obliques tipped past 45° (all 26
  axial fluid series over 30° carry coronal/ACL wording). Sagittal obliques are ordinary sagittals
  at about 23° and share a slot with a standard sagittal in 13 of 432 cases, so they are excluded.
- **Split slots** (all training studies): 295 coronal T1/PD, 227 coronal fluid and 25 axial fluid,
  in about 545 studies. Sharing depends on tilt: 25 of 201 coronal obliques at 20–25° share a slot,
  against about 95% above 30°.
- **Geometry** (`oblique_axes`): coronal obliques rotate about the left–right axis toward axial;
  their slice depth along the patient axis stretches by 1/cos(tilt).

## Label profile of their studies

Positive rates in studies whose most tilted coronal fluid series is at 30–45° (174 studies,
`tilt_label_lift`, `series_tilt_label_frequency.png` in the repository root): ACL 40% against 15%,
MCL 15% against 5%, lateral meniscus 33% against 16%, Baker's cyst 6% against 24%. The coronal
T1/PD obliques at 30–45° instead carry more OA (medial 66% against 37%). `k_05` already shifts
its predictions: mean ACL prediction 34% in these studies against 19% elsewhere, and its ACL AUC
is 0.930 inside them against 0.939 outside; MCL is weak inside them (0.737 on about 6 positives).

## Already measured

- **Inference-only slot policies** on `k_05`, `k_025` and `k_075`
  (`scripts/adhoc/score_tilted_slot_policies.py`, `outputs/analysis/tilted-slot-policies-*`):
  the additive policy, which is the reference mechanism at test time only, was −0.0001 to +0.0006
  on the full fold; averaging a standard-only and a tilted-only pass gained +0.0002 to +0.0006;
  tilted-only lost.
- **Frozen-model repeated slots**: extra triplets from a slot's second series helped more than
  from its first, most for different-geometry pairs (+0.003 to +0.005 on those studies;
  [test-time allocation](../../../docs/validation/results/2026-09-17-test-time-allocation.md#repeated-series-in-one-slot)).
- **Protocol as information** (`protocol_prior`): adding oblique, 3D or vendor flags to `k_05`'s
  fold-0 predictions changed macro AUC by −0.0003 to −0.0014 (ACL +0.0000). The model already reads
  the protocol from appearance; a gain has to come from how it routes and weights these views.
- **Fixed-total reallocation into repeated slots** (`data.repeated_series_share`) topped fold 0 and
  lost publicly, 0.929 against 0.933
  ([overnight](../../../docs/validation/results/2026-09-18-overnight-results.md)); it took triplets
  from other types for any repeated series.
- **Per-label localization** (`span_by_group`): in the sampling campaign's physical-span run the
  PF OA loss and ligament gains were as large in studies without thin series, so per-label claims
  need the changed/unchanged split.

## Sampling coverage of the oblique groups

`oblique_reach` replays ten training epochs on the 103 fold-0 split studies at the incumbent seed.
The oblique groups see a median 89% (coronal fluid, 13 of 16 windows), 85% (coronal T1/PD) and
93% (axial fluid) of their triplet windows over the run, against 79–83% for the standard series
beside them. Thirteen of 54 coronal fluid oblique slabs lie off the study centre, with a limiting
reach under 10 mm and a narrow Gaussian width, yet still reach high run coverage because the slabs
hold only about 14 windows. Denser or re-centred sampling of the same series has little room.

## Backlog ownership and directions left out

- The general backlog's [oblique embedding](../../../docs/backlog/small-changes.md#oblique-embedding)
  entry was transferred here on 2026-09-22.
- Left out: tilt-threshold changes (set from measured geometry), sagittal obliques, test-time
  policies such as the two-pass average (endgame test-time augmentation), the image count and τ
  (inference choices, not the ×1 metric), and budget increases (the shared accounting is exact).
