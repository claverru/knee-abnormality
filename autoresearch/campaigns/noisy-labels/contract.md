# Noisy-labels campaign

Defined 2026-09-23; not started.

## Objective

Make training robust to wrong training labels. The labels are extracted from radiology reports,
and they are measurably noisy against the radiologists' image labels: on the 58 gold studies about
a third of cells with targets in [0.3, 0.5) are positive, and about a quarter of cells above 0.9
are not ([soft AUC evidence](../../../docs/validation/results/labels-and-losses/2026-09-23-validation-auc-variants.md#why-soft-targets-are-closer-to-the-truth)).
Two measured results say label noise, rather than image evidence, now limits the score. Changing
only the class-balance strength moved the public score from 0.940 to 0.945. Blending an image
teacher's predictions into the labels raised their gold agreement by +0.020.
[Prior work](prior-work.md) has the numbers.

In scope, as training-objective changes that use the model's own training-time predictions:

- target correction: bootstrapped targets, temporal-ensemble or EMA targets;
- down-weighting or dropping cells the model confidently and persistently contradicts;
- early-learning regularization and related prediction-anchored terms;
- schedules that relax label trust or class balancing over training.

Out of scope: new or edited label tables and external label sources, per-label mechanisms or
coefficients, weighting the gold studies, validation-driven adaptation, a second network trained
alongside, and anything outside the training objective (data, sampling, augmentation, model,
backbone, optimizer, checkpoint selection, the shared evaluation). The old
[losses campaign](../losses/INDEX.md) tested label-derived reweighting and loss shapes; this
campaign's mechanisms are driven by the model's own predictions.

**No submissions or publication. Only local training and local validation are authorized.** Never
submit to Kaggle, upload models or predictions, publish datasets or notebooks, or launch remote
evaluation jobs. Reading public research and documentation is allowed.

## Evaluation and adoption

The metric, split, labels, validation and prediction multipliers, bootstrap and adoption rule are
the [shared evaluation](../../protocol.md#shared-evaluation-and-adoption), identical for every
campaign. This contract does not restate or override them. Every candidate uses the incumbent's
recorded training seed ([seed policy](../../protocol.md#training-seed-policy)).

**Judge caveat.** The shared rule scores hard macro AUC against report-derived fold-0 targets,
which contain the errors this campaign targets. A mechanism that correctly stops fitting wrong
labels can lose on it. The class-balance curve already shows this: fold-0 hard AUC rose with the
exponent while the public score peaked at 0.5. Soft macro AUC orders past submissions closer to
the leaderboard, and it follows that curve
([validation AUC variants](../../../docs/validation/results/labels-and-losses/2026-09-23-validation-auc-variants.md)).
Every completion therefore reports, against the reference and on the selected checkpoint's ×4
predictions:

- hard and soft macro AUC, with per-label deltas for both;
- the realized fraction of training cells the mechanism changed, per epoch.

A candidate that loses on hard AUC and gains on soft AUC is recorded as such in the findings. The
adoption rule is unchanged.

## Reference and fixed constraints

- Reference: `outputs/training/k05-clean` (submitted as S48, 0.942 public), `configs/training.yaml`
  with series cleaning at seed 2263738815: fold 0, best epoch 8 chosen by ×1 validation
  (0.899419), its fold-0 predictions at ×4 and τ 1 scored by training itself (0.902760). Not
  `k_05`, which predates the cleaning.
- **Reference exceptions (user decision, 2026-09-23).** `k05-clean` was trained while the oblique
  code still existed, disabled. `campaign.yaml` declares its recorded
  `data.series_types[].oblique_tilt_deg` (all null), `preprocessing_identity.sampling.oblique_split`
  and `preprocessing_identity.sources_sha256` as exceptions, removed from both sides of the start,
  sealing and comparison checks. The proof that the removed code never ran: with the tilt unset,
  the old `_prepare_slot` called the same slot preparation with the same seed key as today's; the
  other source differences are logging and a moved Lightning hook. The remaining preprocessing
  identity equals the current source's.
- Evidence run: `outputs/training/image-blend`, the same recipe trained on the image-blended label
  table (a different label source, so its scores are not comparable with the reference's).
- Ten epochs, 4,380 optimizer steps and a runtime ceiling of 115% of the reference's recorded
  setup, training and ×4 prediction time. An adopted candidate never raises them.
- Training visits, triplets and ×1 validation triplets equal the baseline CPU replay's counts
  exactly; this campaign changes none of them.
- One rule and one set of coefficients for all labels and all observed cells. The class-balance
  weights stay as configured unless a schedule idea replaces their strength over training, and
  then the schedule is shared across labels.
- Any per-study memory is filled only from this run's training batches and restored on resume.
  Validation targets, validation predictions and checkpoint selection are never used by the
  mechanism.

## Sealing

At start the supervisor seals the current source, resolved recipe, data identities, reference
predictions, the baseline replay's realized counts at zero tolerance, the runtime ceiling and the
evaluation settings. It verifies the reference against the sealed recipe and input identity before
any candidate. Candidate agents cannot change these limits. See [runtime operation](../../runtime.md).
