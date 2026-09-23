# Oblique campaign (x4 predictions)

Retired 2026-09-22 at revision 78 after three rejections; the topic continues on the source with
series cleaning in the [cleaned-source campaign](../oblique-clean/INDEX.md), whose
[prior work](../oblique-clean/prior-work.md#results-of-the-x4-campaign) records this campaign's
results.

Defined 2026-09-22. It succeeded the retired [oblique campaign](../oblique/INDEX.md)
on the same topic after the user moved the shared evaluation to ×4 predictions: the best epoch is
still chosen by ×1 validation, and that checkpoint's fold-0 predictions at ×4 and attention
temperature 1 decide adoption. The first campaign's three identity results sat near the adoption
line at ×1, and ×1 understated the oblique groups against `k_05`, so the topic restarts on the
new evaluation.

## Objective

Make the oblique groups pay off. An oblique group is a coronal series tilted more than 20°, or an
axial fluid series tilted more than 30°, that shares its slot with a standard series and is
therefore sampled with its own budget
([sampler](../../../docs/data-preparation/physical-sampling/implemented-sampler.md#oblique-groups)).
They are mostly dedicated ACL-oblique protocols and a site's tilted coronals, in about 12% of
studies (103 on fold 0), and those studies carry about twice the ACL, MCL and lateral meniscus
rates. Today the model receives their tokens with the slot's plane and contrast embeddings, so it
can tell them apart only by appearance. On the split studies the oblique budget cost about 0.010
macro AUC against `k_05` at ×1, mostly MCL and lateral meniscus
([prior work](prior-work.md#results-of-the-retired-x1-campaign)).

In scope:

- how the model represents tokens from oblique groups: embeddings from group membership or
  acquisition geometry, low-rank feature branches;
- how attention and readouts weight them: per-label attention biases or gates by view group,
  dedicated or residual readouts, pooling the split slots in alternative ways;
- zero-sum redistribution of triplets between an oblique group, its slot's standard series and the
  study's other types, including which series an evaluation bag scores.

Out of scope: the tilt thresholds and `oblique_split` itself, sagittal obliques, backbone, losses
and loss weights, labels, augmentation and reflection, series dropout, the 224-pixel input, its
0.625 mm spacing and 140 mm field of view, fold membership and the evaluation. Denser sampling of
the same oblique series is not a lead: over ten epochs each group already sees 85–93% of its
triplet windows ([prior work](prior-work.md#sampling-coverage-of-the-oblique-groups)).

**No submissions or publication. Only local training and local validation are authorized.** Never
submit to Kaggle, upload models or predictions, publish datasets or notebooks, or launch remote
evaluation jobs. Reading public research and documentation is allowed.

## Evaluation and adoption

The metric, split, labels, validation and prediction multipliers, bootstrap and adoption rule are
the [shared evaluation](../../protocol.md#shared-evaluation-and-adoption), identical for every
campaign. This contract does not restate or override them. Every candidate uses the incumbent's
recorded training seed ([seed policy](../../protocol.md#training-seed-policy)).

Only the split studies' inputs change, so a same-seed candidate still diverges in training for
every study: at ×1, per-label swings of 0.01–0.02 appeared in the 802 fold-0 studies whose inputs
were identical, and the first campaign's residual readout gained +0.0036 there against +0.0005 on
the split studies. Every completion reports macro and per-label deltas separately on the studies
that hold a split slot and on the rest, against both the reference and `k_05`'s ×4 predictions
where available, and a mechanism claim rests on the split studies. The adoption rule is unchanged.

## Reference and fixed constraints

- Reference: `outputs/training/oblique-x4`, the `oblique` run re-predicted at ×4 by
  [`repredict_run`](../../../scripts/tools/repredict_run.py): fold 0, seed 2263738815, selected
  epoch 9 (×1 epoch macro AUC 0.897527), ×4 prediction macro AUC 0.902373. The recipe is
  `configs/training.yaml` in the current source, identical to the reference apart from run name
  and output path. `k_05` (0.899042 at ×1) is the same recipe without oblique groups; it is
  evidence, not the incumbent.
- Ten epochs, 4,380 optimizer steps and a runtime ceiling of 115% of the reference's recorded
  setup, training and ×4 prediction time. An adopted candidate never raises them.
- Training visits per epoch, total training triplets over the run and ×1 validation triplets per
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
