# Prior work for the side-routing campaign

Everything this campaign starts from was run or measured on 2026-10-05 and is recorded, with
tables, in the [plan and runs](../../../docs/ongoing/side-routed-heads.md). This page is the
summary the researcher checks proposals against.

## The five runs

All on fold 1 of `all_5fold_scanner_v2`, seed 1110484336, ResNet-18, against the old recipe's
run `slab-rule-and-positive-std-requeue` (×4 0.924049). One run each.

| Run | Build | ×4 macro | Δ | Routed five, summed | Public |
|---|---|---:|---:|---:|---:|
| `20261005-181012-side-routed-heads` | Shared routed classifiers, per-study landmarks, no position | 0.923102 | −0.0009 | −0.0035 | |
| `20261005-185535-side-routed-knee-frame` | Header rotation, learned offsets, zero-started token code | 0.923534 | −0.0005 | +0.0033 | 0.952 (S112) |
| `20261005-194836-side-routed-attention-prior` | Detached readers, position prior on attention, routed auxiliary loss | 0.923085 | −0.0010 | +0.0080 | |
| `20261005-202001-side-routed-run4` | Auxiliary loss removed, per-token side head | 0.923952 | −0.0001 | +0.0131 | 0.953 (S113) |
| `20261005-205251-side-routed-run5` | Frame = centre + learned rotation, back-only OA sides, token position code for own channels | 0.924476 | +0.0004 | +0.0090 | 0.952 (S114) |

The old recipe on the same fold and inference setting scored 0.950 public (S108, S111). Over
the three submissions the public score followed the routed targets' sum, not the local macro:
the fifth run is the best locally and 0.001 below the fourth on the public board.

## What the incumbent still shows

- **Below the sibling runs:** Baker's (below all 12), Fracture and ACL (below 11).
- **Medial OA and Lateral OA** gave back 0.0017 and 0.0019 when they lost the front of their
  side; the frame's nod ended at 5°.
- **Isolated Medial OA** (no other compartment affected, 53 positives) was 0.007 to 0.010 below
  the old recipe in the first two runs; not re-measured since.
- **Localisation** (a target against studies with the disease only in another compartment) is
  the hard case for every model: 0.74 to 0.83 for OA on the old recipe; the fourth run gained
  +0.002 to +0.013 on all five routed targets.
- **The position prior** learned anatomical places (third run) and moves the logits by 0.02 to
  0.19.

## Closed

The [contract](contract.md#how-a-seed-is-written) lists what the runs and measurements closed,
with the result of each. Do not propose them again.

## Measurements available

| Measurement | Where |
|---|---|
| Rotation of each series against the scanner's axes (76% of knees turned outward, over 10° in 19%) | `scripts/adhoc/census_knee_rotation.py`, plan: Visual inspection and rotation census |
| Patient motion between series (median 0.4 mm, 99% under 2.7 mm) | `scripts/adhoc/inter_series_motion/`, plan: Patient motion between series |
| Cuts drawn on studies; the front cut at three tilts on ten studies | `outputs/analysis/octant-sketches-20261005/`, plan: Octant sketches |
| Routing, boundaries and one target's position prior drawn per study | `scripts/adhoc/render_routing_views.py` |

## Sibling runs for the run report

The 12 same-fold runs of the old recipe that give the per-target noise range
(`--siblings` of `scripts.tools.report_side_routing`): every
`outputs/autoresearch/fixes-and-removals/experiments/*/training` with a
`fold_1/validation_predictions.csv`, except `slab-rule-and-positive-std-requeue-*`, which is the
old reference. Their ×4 macro spans 0.9234 to 0.9248.

## Lineage marks for the run report

`--lineage` of `scripts.tools.report_side_routing` starts as the fourth and the fifth run
(`outputs/training/20261005-202001-side-routed-run4`,
`outputs/training/20261005-205251-side-routed-run5`); the keeper appends every adopted run's
training directory. Adopted so far: `outputs/autoresearch/side-routing/experiments/separate-oa-frame-with-knee-level-term-20261006T020231Z-52418f08/training` (2026-10-06). `--base` is the incumbent's training directory.

## Outside the campaign

- The same design on the Refined ResNet-50 (`configs/refined-resnet50.yaml`): the user carries
  it over after the campaign.
- Kaggle submissions: none by the campaign.
