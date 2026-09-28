# Project state (2026-09-28)

What is in force now: the recipe, the control candidates pair with, the best public submissions and
the rules every comparison depends on. Pending work is in [next](next.md), active lines have their
own documents in [this folder](INDEX.md), and untried ideas are in the [backlog](../backlog/INDEX.md).
How the recipe got here is the [adoption history](../modeling/current-model.md#adoption-history);
measured outcomes are in the [validation results](../validation/results/INDEX.md) and
[history](../history/INDEX.md).

## Recipe in force

- **Labels and split, since 2026-09-28 (user decision):** training and evaluation use
  [`rebuild_state_v2`](../public-labels/label-rebuild.md#a2-common-state-scale-and-the-zhen-lineage-fix-2026-09-28)
  (A2 of the label rebuild) and the label-stratified split
  [`all_5fold_scanner_v2`](../validation/project-validation-protocol.md#all_5fold_scanner_v2), fold 0,
  fresh seed 1533697213 (`configs/training.yaml`). The evidence is one fold-4 pair: A, the pre-fix
  version of A2, scored 0.947 public (S76) against 0.946 for S73, one rounding step; A2 itself is
  unsubmitted. The autoresearch gate requires both the label source and the split.
- **Model:** the [current model](../modeling/current-model.md): ReXNet-200 spatial tokens with
  knee geometry, the three-compartment OA detector, training-only report-attribute and scanner
  confusion heads, the half-stack augmentation and class-balanced BCE at exponent 0.5.
- **Preprocessing identity:** hashed-source changes apply to fresh runs only. A new export or local
  prediction from a checkpoint trained under an earlier identity is refused by
  `validate_preprocessing_identity` and needs that checkpoint's archived sources; already-exported
  bundles carry their own sources and still submit
  ([data contract](../data-preparation/public-data-contract-and-decoding.md),
  [identity rules](../architecture/configs-and-experiment-artifacts.md#experiment-artifacts-and-reproducibility)).

## Current control

- **`rebuild-state-v2-control`** (`outputs/training/20260928-150724-rebuild-state-v2-control`), from
  `configs/training.yaml`: ×4 0.928288 (×1 0.924637, epoch 9 of 10, 912 validation studies). Not yet submitted; submit at ×4, τ 0.75, batch 4.
  Every candidate pairs with it at its fold and seed.
- **No local score on `all_5fold_scanner_v1` is comparable** with a run on the new split
  ([validation strategy](../validation/validation-strategy.md)). The last v1 control,
  `scanner-confusion-control` (fold 4, seed 703791404, ×4 0.909733, 0.946 public as S73), stays the
  reference only for the fold-4 comparisons already made against it.

## Best public submissions

- **0.948:** S67, the scanner confusion head candidate, a single fold-3 model at ×4 and τ 0.75
  ([evidence](../modeling/scanner-confusion-head.md#evidence)).
- **0.947:** S68 (`half-stack-control`), S69 (per-site bias,
  [auxiliary-signals campaign](../validation/results/2026-09-27-auxiliary-signals-campaign.md)) and
  S76 (the label rebuild's state table A, [label rebuild](../public-labels/label-rebuild.md#status)).
- **Ensembles:** the 2026-09-19 ten-member bundle of five EfficientNet-B0 and five ReXNet-200 fold
  models with mirror TTA scored 0.943 at ×4 and τ 0.75; the five-B0 bundle without TTA scored 0.940
  at 27% of the runtime and is the efficiency entry. Every member trained with ordinary BCE
  ([submissions](../inference/submissions/2026-09-19-ensemble-submissions.md)).
- Keep attention temperature 0.75 in later bundles; its
  [inference gain](../inference/submissions/2026-09-18-temperature-and-count-submissions.md) is
  adopted. The [ledger](../inference/submissions/ledger.md) lists submissions up to S65; later
  ones are in the results records linked here.

## Measurement and adoption rules in force

- **Adoption:** the local rule is a macro gain ≥ 0.001 with a positive lower bound of the central
  80% paired report-group interval; the 95% interval is diagnostic
  ([controlled comparisons](../validation/validation-strategy.md#controlled-comparisons)). Recent
  adoptions were decided by the user on paired public scores
  ([adoption history](../modeling/current-model.md#adoption-history)).
- **Fold and seed rotation:** candidates use their control's fold and seed; each adoption merged
  into main retrains the control with a fresh seed on the next fold. Never compare scores across
  folds or splits ([rule](../validation/validation-strategy.md#controlled-comparisons)).
- **Validation and prediction density:** epoch validation at ×1 (since 2026-09-17) selects the
  checkpoint on hard `val_macro_auc`; saved predictions, comparisons and the autoresearch rule score
  that checkpoint at ×4 and τ 1 (since 2026-09-22). `val_soft_macro_auc` (since 2026-09-23) is used
  to compare runs and choose submissions; two paired tests of the hard-AUC checkpoint against a
  later one tied publicly (S41/S49, S64/S65), so selection stays on hard AUC
  ([result](../validation/results/2026-09-23-validation-auc-variants.md),
  [augmentation campaign](../validation/results/2026-09-26-augmentation-campaign.md)).
- **Local against public:** gold-only diagnostics decide nothing; prefer a public result over a
  sub-0.005 local difference
  ([local-to-public relationship](../validation/local-to-public-score-relationship.md)).
- **Autoresearch:** every campaign shares one
  [evaluation and adoption rule](../../autoresearch/protocol.md#shared-evaluation-and-adoption) and
  takes its fold from the sealed recipe. Campaigns no longer retrain controls after a provisional
  acceptance; the user retrains the fresh-seed control when an adoption is merged. No campaign is
  active ([status](../../autoresearch/status.md)).
- **Deferred to the final phase:** further folds, ensembles, TTA, bigger images and weight
  averaging ([endgame](../backlog/endgame.md)); hidden-test failure modes are in
  [risks](../backlog/risks.md).

## Active lines

- [Label rebuild](../public-labels/label-rebuild.md): adopted; its control scored ×4 0.928288 and
  awaits submission.
- [Report attribute extraction](report-attribute-extraction.md): the JevK5 heads are in the recipe;
  a v4 question set is open.
- [Loading speedup](loading-speedup.md): the `inference-opt` branch awaits a decision.
- [EfficientNet backbones](efficientnet-backbones.md): the B0 gap under class-balanced BCE and a B3
  follow-up are open.

## Recently closed

- **2026-09-28:** the Qwen Lateral OA half-blend pilot, not adopted: it tied the control and was
  superseded by the [label rebuild](../public-labels/label-rebuild.md)
  ([history](../history/qwen-lateral-labels.md)).
- **2026-09-28:** plane-specific backbones, not adopted: three ReXNet-200s tie the shared backbone
  publicly (0.945, S77, against 0.946, S73) at about 35% more training and twice the inference time
  ([result](../validation/results/2026-09-28-plane-specialization-campaign.md),
  [history](../history/plane-backbones.md)). Kaggle training, not worth it
  ([history](../history/kaggle-training.md)). Variable training bags, without a completed run
  ([history](../history/variable-training-bags.md)).
- **2026-09-27:** the [auxiliary-signals campaign](../validation/results/2026-09-27-auxiliary-signals-campaign.md)
  adopted the scanner confusion head; heavier site debiasing (S70 0.944, S71 0.943) and the
  question-level reliability filter (S72 0.943) lost publicly. The
  [EfficientNet-B3 queues](../validation/results/2026-09-27-b3-overnight-queue.md) adopted nothing.
- **2026-09-26:** the [augmentation campaign](../validation/results/2026-09-26-augmentation-campaign.md)
  closed with the half stack adopted.
- **2026-09-25:** the [near-misses campaign](../validation/results/2026-09-25-near-misses-campaign.md)
  gave the OA detector; its token dropout was rejected in a
  [follow-up](../validation/results/2026-09-25-near-misses-campaign.md#follow-up-on-main).
- **2026-09-24:** the [knee-geometry campaign](../validation/results/2026-09-24-knee-geometry-campaign.md)
  stopped after 18 decisions with one adoption. Early-learning regularization scored 0.943 (S57)
  against 0.942 (S48) and was not adopted ([history](../history/early-learning-regularization.md));
  loss variants that gain on soft AUC are on a
  [watch list](../validation/results/2026-09-23-validation-auc-variants.md#losses-campaign-rescored-2026-09-24).
- **2026-09-22–23:** the [noisy-labels line](../validation/results/2026-09-23-noisy-labels-campaign.md),
  the [oblique line](../history/oblique-groups.md) and
  [stride-16 feature fusion](../history/stride16-feature-fusion.md), none adopted.

## Standing user decisions recorded only here

- **2026-09-28:** the no-per-label-tuning rule is lifted. Under it, per-label attention
  temperatures and per-label loss selection were dropped from the backlog on 2026-09-23; the
  [per-label loss research](../research/per-label-losses.md) remains as research.
- **2026-09-21 research backlog review:** AdamP and plane/contrast-specific feature adapters were
  accepted ([small changes](../backlog/small-changes.md)); terminal-1×1-only trimming was rejected;
  layer-wise CoAtNet learning-rate decay, standalone BatchNorm recalibration, ordered within-series
  aggregation and SAM were not selected. Bias-field augmentation, also not selected then, was later
  measured in the augmentation campaign.
