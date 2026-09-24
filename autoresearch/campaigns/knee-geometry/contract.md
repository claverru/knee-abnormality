# Knee-geometry campaign

Defined 2026-09-23; not started.

## Objective

Make every label's evidence come from the right place in the knee, and treat each compartment
family the way it behaves. Measured errors motivate it: about 3% of studies have a rejected
centre and get no anatomical bias; the medial-lateral midline is off by more than 5 mm in 12.5% of
studies; knees sit rotated in the scanner (median 6.7°); bilateral studies get one side for both
knees; and a trained side mask helped the menisci (+0.0046) but hurt OA (−0.0040).
[Prior work](prior-work.md) has the evidence.

In scope:

- the study centre, bilateral studies and the medial-lateral midline and axis;
- compartment masks, readouts and training-only locality losses for the meniscus and OA pairs;
- slice-sampling density taken from measured geometry;
- absolute, side-normalized position encoding.

Out of scope: label tables and label noise, backbone, optimizer, image size, ensembling, TTA,
OA severity supervision (the OA-attributes big bet), tuning the midline band width, and inferring
side, centre or landmarks with a trained model.

**No submissions or publication. Only local training and local validation are authorized.** Never
submit to Kaggle, upload models or predictions, publish datasets or notebooks, or launch remote
evaluation jobs. Reading public research and documentation is allowed.

## User authorizations (2026-09-23)

- **Family-scoped rules** for the four compartment labels (Medial/Lateral Meniscus,
  Medial/Lateral OA): the meniscus-only side mask, the meniscus half-knee loss, the meniscus
  shared-plus-residual head and the OA own-side plus knee-level readout. They follow the measured
  disease behaviour (meniscus labels correlate at 0.08, OA at 0.52), not per-label score tuning:
  one rule per family, no per-label coefficients, and the other eight labels keep their path.
- **Geometry before training only.** Knee side, centre, midline and axis come only from DICOM
  headers, metadata and rule-based within-study checks. Nothing is inferred by a trained model,
  from labels or from predictions.

## Evaluation and adoption

The metric, split, labels, validation and prediction multipliers, bootstrap and adoption rule are
the [shared evaluation](../../protocol.md#shared-evaluation-and-adoption), identical for every
campaign. This contract does not restate or override them. Every candidate uses the incumbent's
recorded training seed ([seed policy](../../protocol.md#training-seed-policy)).

**Geometry bundle.** The first seed bundles three fixes of measured geometry errors (robust
centre, bilateral rule, midline from the fitted centre). Each touches few studies and is likely
below +0.001 alone; none is a tuned choice. It is judged by the normal rule as one experiment.

- Its completion must report per-stratum deltas, split by which fix touched each study: rescued
  centre, bilateral, midline moved by more than 5 mm. Each fix changes only the studies it
  touches, so a harmful component shows in its own stratum.
- If it is rejected, the components whose strata improved are re-proposed as a smaller
  combination, and a component whose stratum got worse is dropped. Until a geometry bundle is
  adopted, geometry-dependent ideas carry its surviving components as part of their change.

**Every completion reports**, against the reference on the selected checkpoint's ×4 predictions:

- the meniscus, OA and other-eight family deltas with their intervals, and the compartment
  margins and cross-talk from `compare_fold_runs`;
- the learned anatomical strengths at the selected epoch;
- the mechanism's activity (studies or tokens affected).

## Iteration

The seeds are a starting set. After each run, the next proposals may refine or combine earlier
runs, adopted or not: a refinement answers a run's own diagnostics and names it in
`parent_idea_id` with what is materially new; a combination stacks mechanisms whose evidence
points the same way (for example a family gain inside a run that missed the macro bar) and names
every parent. Each proposal still runs once against the pinned incumbent.

**Order.** The geometry bundle runs first; the knee frame, the meniscus-only mask and the 3D
encoding build on it. The half-knee loss and the meniscus head build on the meniscus-only mask,
and carry it when it is not the incumbent.

## Reference and fixed constraints

- Reference: `outputs/training/k05-clean` (S48, 0.942 public), `configs/training.yaml` with
  series cleaning at seed 2263738815: fold 0, best epoch 8 chosen by ×1 validation (0.899419), its fold-0
  predictions at ×4 and τ 1 scored by training itself (0.902760).
- **Reference exceptions (user decision, 2026-09-23).** `k05-clean` was trained while the oblique
  code still existed, disabled. `campaign.yaml` declares its recorded
  `data.series_types[].oblique_tilt_deg` (all null), `preprocessing_identity.sampling.oblique_split`
  and `preprocessing_identity.sources_sha256` as exceptions, removed from both sides of the start,
  sealing and comparison checks. Proof the removed code never ran: the current source reproduces
  `k05-clean`'s saved predictions to 5.5e-5
  ([sibling masking](../../../docs/validation/results/2026-09-23-sibling-masking.md)).
- Evidence runs: `outputs/training/side-mask` (trained side mask for both families) and
  `outputs/training/token-attention` (global token attention with relative 3D RoPE, 1D and 2D
  embeddings dropped), both on the reference's recipe and seed; see [prior work](prior-work.md).
- Ten epochs, 4,380 optimizer steps and a runtime ceiling of 115% of the reference's recorded
  setup, training and ×4 prediction time. An adopted candidate never raises them.
- Training visits and aggregate training and ×1 validation triplets equal the baseline CPU
  replay's counts exactly. Redistribution between studies, types and epochs is allowed; adding or
  removing selectable series is not.
- Unknown side (170 of 4,410 studies) or invalid geometry falls back to the control's behaviour
  for that study.
- Validation stays deterministic, unaugmented and independent of the training seed.

## Sealing

At start the supervisor seals the current source, resolved recipe, data identities, reference
predictions, the baseline replay's realized counts at zero tolerance, the runtime ceiling and the
evaluation settings. It verifies the reference against the sealed recipe and input identity before
any candidate. Candidate agents cannot change these limits. See [runtime operation](../../runtime.md).
