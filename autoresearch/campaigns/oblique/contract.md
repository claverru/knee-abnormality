# Oblique campaign

Defined and run 2026-09-22; **retired** the same day when the shared evaluation moved to ×4
predictions, after two rejected experiments and one trained but uncompared; its topic continues in
the [x4 oblique campaign](../oblique-x4/INDEX.md). It followed the
[oblique-groups run](../../../docs/validation/results/2026-09-22-oblique-groups-result.md),
which gave tilted coronal and axial fluid series their own slot budget and did not measurably
help. The user kept that recipe as the reference and opened this campaign to find what else the
mechanism needs.

## Objective

Make the oblique groups pay off. An oblique group is a coronal series tilted more than 20°, or an
axial fluid series tilted more than 30°, that shares its slot with a standard series and is
therefore sampled with its own budget
([sampler](../../../docs/data-preparation/physical-sampling/implemented-sampler.md#oblique-groups)).
They are mostly dedicated ACL-oblique protocols and a site's tilted coronals, in about 12% of
studies (103 on fold 0), and those studies carry about twice the ACL, MCL and lateral meniscus
rates. Today the model receives their tokens with the slot's plane and contrast embeddings, so it
can tell them apart only by appearance.

In scope:

- how the model represents tokens from oblique groups: embeddings from group membership or
  acquisition geometry;
- how attention and readouts weight them: per-label attention biases by view group, dedicated or
  residual readouts;
- zero-sum redistribution of triplets between an oblique group, its slot's standard series and the
  study's other types.

Out of scope: the tilt thresholds and `oblique_split` itself, sagittal obliques, backbone, losses
and loss weights, labels, augmentation and reflection, series dropout, the 224-pixel input, its
0.625 mm spacing and 140 mm field of view, fold membership and the evaluation. Denser sampling of
the same oblique series is not a lead: over ten epochs each group already sees 85–93% of its
triplet windows ([prior work](prior-work.md#sampling-coverage-of-the-oblique-groups)).

**No submissions or publication. Only local training and local validation are authorized.** Never
submit to Kaggle, upload models or predictions, publish datasets or notebooks, or launch remote
evaluation jobs. Reading public research and documentation is allowed.

## Evaluation and adoption

The metric, split, labels, validation multiplier, bootstrap and adoption rule are the
[shared evaluation](../../protocol.md#shared-evaluation-and-adoption), identical for every
campaign. This contract does not restate or override them. Every candidate uses the incumbent's
recorded training seed ([seed policy](../../protocol.md#training-seed-policy)).

Only the split studies' inputs change, so a same-seed candidate still diverges in training for
every study: in the reference run, per-label swings of 0.01–0.02 appeared in the 802 fold-0 studies
whose inputs were identical. Every completion reports per-label deltas separately on the studies
that hold a split slot and on the rest, and a mechanism claim rests on the split studies. The
adoption rule is unchanged.

## Reference and fixed constraints

- Reference: `outputs/training/oblique`, fold 0, seed 2263738815, selected epoch 9, macro AUC
  0.897527. The recipe is `configs/training.yaml` in the current source, identical to the reference
  apart from run name and output path. `k_05` (0.899042) is the same recipe without oblique groups;
  it is evidence, not the incumbent.
- Ten epochs, 4,380 optimizer steps and a 1,868 s runtime ceiling (115% of the reference's
  recorded 1,623.9 s). An adopted candidate never raises them.
- Training visits per epoch, total training triplets over the run and validation ×1 triplets per
  epoch equal the baseline CPU replay's counts exactly. Budget ideas redistribute; they never add.
- Every training study is visited exactly once per epoch.
- New model parameters start neutral, so an untrained change reproduces the reference forward pass.
- Validation is deterministic, unaugmented, independent of the training seed and keeps its ×1
  triplet count; record any changed deterministic selection identity.
- Learned components use training batches only; no validation labels or predictions. All
  computation runs inside the measured training process.

## Sealing

At start the supervisor seals the current source, resolved recipe, data identities, reference
predictions, the baseline replay's realized counts at zero tolerance, the runtime ceiling and the
evaluation settings. It verifies the reference against the sealed recipe and input identity before
any candidate. Candidate agents cannot change these limits. See [runtime operation](../../runtime.md).
