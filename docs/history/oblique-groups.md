# Oblique groups (closed 2026-09-23)

**Closed without adoption.** Oblique groups gave tilted axial-fluid and coronal series their own
slot budget, and a later learned embedding marked their tokens. Neither improved the leaderboard.
Three paired submissions each compare a recipe with and without the package, and each came out
level or lower: −0.001, 0.000 and −0.002. At the ×4 submission setting the local gain was at most
+0.0003 hard macro AUC. The user kept oblique groups in the recipe on 2026-09-22 without evidence,
as the reference for three autoresearch campaigns. None of those campaigns adopted anything. The
user removed the package on 2026-09-23 after the image-blend pair (S50 against S52). No code is
kept.

## Mechanism

- **Oblique groups** (`data.series_types[*].oblique_tilt_deg`): axial fluid 30°, coronal T1/PD
  and coronal fluid 20°.
  - **Tilt:** a series' tilt is the angle between its slice normal and the patient axis of its
    type's labelled plane, `degrees(arccos(min(1, |n[axis]| / |n|)))`.
  - **Split:** when a slot held both standard and tilted series, the tilted group got its own
    unscaled type budget and its own seed key (`…, "oblique"`), so it added triplets instead of
    redistributing them. Each group ran its own candidate pool, selection and preparation, and
    the slot's metadata merged both groups.
  - **Scope:** it changed 103 of 905 fold-0 studies. The mean study rose from 26.80 to 27.37
    training triplets at ×1.
- **Group embedding** (`model.oblique.enabled`): one learned vector, added to the tokens of the
  oblique group in split slots. It was stored as `spatial_tokens.oblique.weight` (2×256) and
  started at zero, so an untrained model gave the reference output.

The combined code diff is saved in
`outputs/analysis/o/oblique-line-20260923/oblique-groups-and-embedding.diff`. Each run's
`source_snapshot.zip` also holds its implementation (`oblique`, `oblique-clean`,
`image-blend-oblique`, and the campaign experiments under `outputs/autoresearch/oblique*/`).
Measurements of tilted-series geometry and coverage are in
`outputs/analysis/a/acquisition-geometry-20260922/`.

## Paired submissions

All pairs use fold 0 of `all_5fold_scanner_v1` and seed 2263738815, submitted at ×4 τ 0.75
unless noted. Local values are ×4 τ 1 validation predictions.

| Recipe | Without | With | Local ×4 hard | Local ×4 soft | Public without → with |
|---|---|---|---:|---:|---|
| `k_05`, source before series cleaning | `k_05` | `oblique` (groups only) | +0.0003 (τ 0.75) | – | S41 0.945 → S46 0.944 (S47 at ×5 τ 0.70: 0.945) |
| Cleaned source, combined labels | `k05-clean` | embedding candidate (groups + embedding) | +0.0024 | +0.0021 | S48 0.942 → S51 0.942 |
| Image-blend labels | `image-blend` | `image-blend-oblique` (groups + embedding) | +0.0003 [80% −0.0013, +0.0018] | +0.0008 [80% −0.0002, +0.0017] | S50 0.945 → S52 0.943 |

The embedding alone was measured against `oblique-clean` (groups only, cleaned source) at ×4:
+0.0015 hard (80% [−0.0004, +0.0033]) and +0.0019 soft (80% [+0.0009, +0.0030]). `oblique-clean`
was never submitted, so the embedding's own public effect is unmeasured. Details:
[oblique groups result](../validation/results/2026-09-22-oblique-groups-result.md).

## Campaigns

Adoption required a gain of at least +0.001 and a positive 80% paired report-group lower bound.
Nothing passed.

| Campaign | Reference | Ideas and hard macro AUC Δ |
|---|---|---|
| [`oblique`](../../autoresearch/campaigns/oblique/INDEX.md), ×1 | `oblique` 0.897527 | group embedding +0.0018 and label-view attention bias +0.0018, both rejected (80% lower bounds just below 0); residual readout trained (×1 0.899668) but not compared |
| [`oblique-x4`](../../autoresearch/campaigns/oblique-x4/INDEX.md) | `oblique-x4` 0.902373 | view bias +0.0006, identity with view bias −0.0006, mixture pooling +0.0001, all rejected; ACL-gated tokens trained but not compared |
| [`oblique-clean`](../../autoresearch/campaigns/oblique-clean/INDEX.md) | `oblique-clean` 0.903705 | ACL-gated tokens +0.0011, coverage-matched split −0.0008, standard-frame depth −0.0017, training-only views −0.0014, all rejected; group embedding compared by hand (above); statistics alignment never trained |

Before and during the campaigns, [`score_tilted_slot_policies`](../../scripts/adhoc/score_tilted_slot_policies.py)
re-predicted fold 0 with frozen checkpoints (`k_05`, `k_025`, `k_075` and others) under
inference-only series policies. Tilted variants acted on coronal and sagittal slots holding a
series at or above the census obliquity threshold beside another series (axial slots were never
modified): standard only, tilted only, additive, an ACL-only attention gate on tilted tokens,
hidden tilted tokens, and means or ACL routing of the single-group passes. General variants acted
on every multi-series slot. Each variant was scored against intact predictions on the full fold
and on the affected studies, with paired report-group bootstrap; it measures checkpoints trained
with pooled slots, not training under a policy. It used the GPU, about five minutes per checkpoint
for every variant at ×4 τ 0.75 on one RTX 3090, and did not depend on the removed oblique code:

```bash
rtk proxy uv run --locked python -m scripts.adhoc.score_tilted_slot_policies --output-dir outputs/analysis/tilted-slot-policies-gate --run-dirs '["outputs/training/k_05"]' --thresholds '[30]' --variants '["standard","additive_acl","intact_acl","intact_hidden"]'
```

Outputs are in `outputs/analysis/tilted-slot-policies-*` (`results.csv`, `aggregate.csv`,
`predictions.parquet`, slot plans, `summary.json`). Results are summarized in the campaigns'
prior work ([`oblique`](../../autoresearch/campaigns/oblique/prior-work.md#already-measured),
[`oblique-x4`](../../autoresearch/campaigns/oblique-x4/prior-work.md),
[`oblique-clean`](../../autoresearch/campaigns/oblique-clean/prior-work.md)) and cited in their
`findings.md`: additive −0.0001 to +0.0006 on the full fold, and the ACL-only gate +0.0020 on
the 102 affected studies of `k_05` at ×4.

A recurring pattern in the campaigns was a label trade-off on the split studies. Representation
and routing repairs recovered MCL and the menisci but gave up ACL.

## Why it was closed

- Each pair alone sits inside the public noise (about 0.003 between seeds of one recipe). But all
  three point the same way, and none shows a local gain at the submission setting.
- Training coverage of the tilted series was already high: 85–89% of their triplet windows over
  ten epochs. Also, the studies they serve were already ranked well, so there was little to gain.
- Removing the package simplifies the sampler and restores `k_05`'s sampling, with the
  preprocessing identity of the `image-blend` run.
