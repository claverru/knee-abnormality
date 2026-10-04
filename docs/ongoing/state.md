# Project state (2026-10-03)

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
  seed 1533697213 (`configs/training.yaml`); the 2026-09-30 adoption kept both (user decision). The evidence is one fold-4 pair: A, the pre-fix
  version of A2, scored 0.947 public (S76) against 0.946 for S73, one rounding step; A2's fold-0
  control scored 0.944 (S78), not comparable with those fold-4 runs. The autoresearch gate requires both the label source and the split.
- **Model:** the [current model](../modeling/current-model.md), since 2026-09-29 on a
  RadImageNet ResNet-18 (`configs/training.yaml`): repeated middle slice at 224 px and 0.536 mm, a
  120 mm field (since 2026-10-01, with augmentation on that slice only), coronal and sagittal
  T1/PD read at 84 mm (0.375 mm) and coronal and sagittal fluid at 93.33 mm (5/12 mm) in place
  of the 120 mm view, placed by rule (since 2026-10-02, the [view table](../modeling/current-model.md#view-table); axial
  stays at 120 mm), `backbone_lr` 5e-5, no
  series dropout, 26 triplets, spatial tokens with knee geometry, plane-contrast readouts and no
  anatomical attention bias (since 2026-10-01), a learned attention bias per view and label
  (since 2026-10-03), the three-compartment OA detector, training-only
  report-attribute heads, L2-SP weight anchoring (since 2026-09-30, which also removed the scanner
  confusion head), the half-stack augmentation and class-balanced BCE at exponent 0.5.
  `configs/refined-resnet50.yaml` is the same recipe on the Refined-RadImageNet ResNet-50, the
  submission model (backbone only differs, since 2026-10-03); ResNet-18 stays the control for
  iteration speed (user decision, 2026-10-01).
  **The ReXNet-200 recipe is retired (2026-10-04, user decision): models stay on the
  Refined-RadImageNet weights.** Its config is removed; the code paths remain because they are
  part of the hashed sources, and the saved runs keep their own effective configs
  ([adoption](../modeling/current-model.md#adoption-history)). Submit ResNet bundles at τ 0.9 at
  ×4, or the swept τ 0.8 at ×6.
- **Preprocessing identity:** hashed-source changes apply to fresh runs only. A new export or local
  prediction from a checkpoint trained under an earlier identity is refused by
  `validate_preprocessing_identity` and needs that checkpoint's archived sources; already-exported
  bundles carry their own sources and still submit
  ([data contract](../data-preparation/public-data-contract-and-decoding.md),
  [identity rules](../architecture/configs-and-experiment-artifacts.md#experiment-artifacts-and-reproducibility)).

## Current control

- **`rn18-view-label-bias`** (`outputs/training/20261003-100844-rn18-view-label-bias`), the
  `configs/training.yaml` recipe in force apart from name and paths, trained on 2026-10-03
  from the adoption commit's source: ×4 0.932477 (×1 0.929026, epoch 9 of 10, fold 0, seed
  1533697213, 1,181 s), 0.951 public (S101). Every candidate pairs with it at its fold and
  seed ([adoption](../modeling/current-model.md#adoption-history)).
- **`fluid-detail-93-replacement`**
  (`outputs/autoresearch/smart-crops/experiments/fluid-detail-93-replacement-20261002T061326Z-c4e4d640/training`),
  the `configs/training.yaml` recipe in force apart from name and paths: ×4 0.932880 (×1
  0.929261, epoch 9 of 10, fold 0, seed 1533697213), 0.950 public (S93); the control before the
  2026-10-03 adoption of the view label bias. It was trained in the smart-crops campaign; main's code for the
  eight files it changed is its archived `source/`, and its checkpoint exports from main with
  the same preprocessing identity
  ([result](../validation/results/2026-10-02-smart-crops-campaign.md#adoption-on-main)).
- **`rn18-field-120-control`** (`outputs/training/rn18-field-120-control`), the recipe before
  the 84/93 mm views: ×4 0.930666 (×1 0.927479, epoch 9 of 10, fold 0, seed 1533697213), 0.947
  public (S92), trained on 2026-10-01 from `main` at `6a99439`; the control before the
  2026-10-02 adoption
  ([result](../validation/results/2026-10-01-mixed-field-campaign.md#control-on-the-adopted-recipe)).
- **`no-anatomical-bias`**
  (`outputs/autoresearch/revisits/experiments/no-anatomical-bias-20260930T213145Z-58126164/training`),
  the same recipe before the field adoption (140 mm), apart from name and paths: ×4 0.929835 (×1
  0.925999, epoch 9 of 10, fold 0, seed 1533697213), no public score, the control before the
  field adoption. It was trained in the revisits campaign, so exports use its archived
  `source/` ([result](../validation/results/2026-10-01-revisits-campaign.md)).
- **`refined-rn50-views-label-bias`** (`outputs/training/20261003-110045-refined-rn50-views-label-bias`),
  the `configs/refined-resnet50.yaml` recipe: ×4 0.937129 (×1 0.931222, fold 0, seed 1533697213),
  0.954 public alone (S102), not a control. Folds 1 to 4 of the same recipe are in
  `outputs/training/20261003-122711-refined-rn50-views-label-bias-folds-1-4`.
- **`rexnet-200-views-label-bias`** (`outputs/training/20261003-114512-rexnet-200-views-label-bias`),
  the `configs/rexnet-200.yaml` recipe: ×4 0.932523 (×1 0.928556, fold 0, seed 1533697213), not
  submitted alone; with the ResNet-50 above 0.939492 on fold 0 (equal weights, ×4, τ 1, no TTA).
- **`refined-rn50-incumbent-recipe`**
  (`outputs/autoresearch/revisits/experiments/refined-rn50-incumbent-recipe-20261001T081909Z-71e68a1d/training`),
  the `configs/refined-resnet50.yaml` recipe before 2026-10-03 (120 mm views, no label bias): ×4
  0.934423, 0.948 public (S88).
- **`rn18-l2sp-no-confusion`** (`outputs/training/rn18-l2sp-no-confusion`), the
  `configs/training.yaml` recipe from 2026-09-30: ×4 0.928062 (×1 0.92451, epoch 9 of 10, fold 0,
  seed 1533697213), 0.943 public (S86), the control before the 2026-10-01 adoption; the
  adoptions did not rotate fold or seed (user decision)
  ([result](../validation/results/2026-09-30-rn18-l2sp-no-confusion.md)).
- **`rn18-no-series-dropout`**
  (`outputs/autoresearch/radimagenet-rn18/experiments/rn18-no-series-dropout-20260928T213320Z-0b780262/training`),
  the `configs/training.yaml` recipe apart from name and paths: ×4 0.927806 (×1 0.92384, epoch 9 of
  10, fold 0, seed 1533697213), 0.943 public (S82), the control before the 2026-09-30 adoption.
  It was trained in the campaign, so exports use its archived `source/`.
- **`rebuild-state-v2-control`** (`outputs/training/20260928-150724-rebuild-state-v2-control`), the
  `configs/rexnet-200.yaml` recipe before 2026-10-03: ×4 0.928288, 0.944 public (S78) at ×4, τ 0.75.
- **No local score on `all_5fold_scanner_v1` is comparable** with a run on the new split
  ([validation strategy](../validation/validation-strategy.md)). The last v1 control,
  `scanner-confusion-control` (fold 4, seed 703791404, ×4 0.909733, 0.946 public as S73), stays the
  reference only for the fold-4 comparisons already made against it.

## Best public submissions

- **0.957:** S106, the five Refined ResNet-50 children of the
  [two-family out-of-fold distillation](../validation/results/2026-10-04-two-family-oof-distillation.md#resnet-50-children-alone-s106)
  alone, one bundle at ×6 with reflection TTA and τ 0.8; pooled out of fold 0.936272.
- **0.955:** S103, ten children of the
  [two-family out-of-fold distillation](../validation/results/2026-10-04-two-family-oof-distillation.md):
  five Refined ResNet-50 and five ReXNet-200 of folds 0 to 4 trained on `oof_blend_v2`, one bundle
  at ×6 with reflection TTA, τ 0.7 and 1.0, equal weights; pooled out of fold 0.938887.
- **0.954:** S102, the Refined-RadImageNet ResNet-50 with the 84/93 mm views and the view label
  bias, alone, fold 0 at ×4, τ 0.9, no TTA (×4 0.937129): the best single model, level with S97.
- **0.954:** S97: 28 networks in 13 bundles with mirror TTA on all, weighted
  by architecture (Refined ResNet-50 0.30; ResNet-18 0.30 over the 84/93 mm fold-0 model and
  four L2-SP models of folds 1 to 4; ReXNet-200 0.30 over folds 0, 3 and 4; 0.10 for an
  EfficientNet-B3 + ReXNet pair and the ten models of 2026-09-19). It has no fold-0 score,
  since most members trained on fold 0
  ([blends](../inference/submissions/2026-10-02-blend-submissions.md#s97-members-and-weights)).
- **0.953:** S96, an equal blend of the ReXNet `rebuild-state-v2-control` (τ 0.75) and the
  Refined-RadImageNet ResNet-50 (τ 0.9), both with reflection TTA, and the ResNet-18 with the
  84/93 mm views (τ 0.9), fold 0 at ×4: S89's pair plus the S93 model
  ([blends](../inference/submissions/2026-10-02-blend-submissions.md)).
- **0.951:** S89, a 50/50 ensemble of the ReXNet `rebuild-state-v2-control` (τ 0.75) and the
  Refined-RadImageNet ResNet-50 (τ 0.9), fold 0 at ×4 with reflection TTA, two bundles in one
  notebook. The ResNet-50 alone scored 0.948 (S88)
  ([revisits campaign](../validation/results/2026-10-01-revisits-campaign.md#ensembles-fold-0-4-equal-weights)).
- **0.950:** S93, the current control, a single ResNet-18 with the 84/93 mm views at ×4 and
  τ 0.9, against 0.947 for S92, the same recipe at 120 mm on every series
  ([smart-crops campaign](../validation/results/2026-10-02-smart-crops-campaign.md)). The best
  single model so far.
- **0.949:** S80, a 50/50 ensemble of the ReXNet `rebuild-state-v2-control` (τ 0.75) and the
  RadImageNet ResNet-18 (τ 0.9), both fold 0 at ×4 and 36 triplets, run as two bundles in one notebook
  ([RadImageNet backbones](radimagenet-backbone.md)).
- **0.948:** S67, the scanner confusion head candidate, a single fold-3 model at ×4 and τ 0.75
  ([evidence](../modeling/scanner-confusion-head.md#evidence)).
- **0.947:** S92, the previous control (120 mm with plane-contrast readouts), and S90, the
  ResNet-18 at 120 mm with middle-slice augmentation on the 2026-09-30 recipe, both at ×4 and
  τ 0.9, against 0.943 for S86; S91, the contrast split at 112 and 140 mm, scored 0.945
  ([result](../validation/results/2026-10-01-mixed-field-campaign.md)). Also S68 (`half-stack-control`), S69 (per-site bias,
  [auxiliary-signals campaign](../validation/results/2026-09-27-auxiliary-signals-campaign.md)) and
  S76 (the label rebuild's state table A, [label rebuild](../public-labels/label-rebuild.md#status)).
- **Ensembles:** the 2026-09-19 ten-member bundle of five EfficientNet-B0 and five ReXNet-200 fold
  models with mirror TTA scored 0.943 at ×4 and τ 0.75; the five-B0 bundle without TTA scored 0.940
  at 27% of the runtime and is the efficiency entry. Every member trained with ordinary BCE
  ([submissions](../inference/submissions/2026-09-19-ensemble-submissions.md)).
- Keep attention temperature 0.75 in ReXNet bundles; its
  [inference gain](../inference/submissions/2026-09-18-temperature-and-count-submissions.md) is
  adopted. The RadImageNet ResNet-18 peaks at τ 0.9–1.0
  ([τ sweep](../validation/results/2026-09-29-rn18-attention-temperature.md)). The
  [ledger](../inference/submissions/ledger.md) lists every submission.

## Measurement and adoption rules in force

- **Adoption:** the local rule is a macro gain ≥ 0.001 with a positive lower bound of the central
  80% paired report-group interval; the 95% interval is diagnostic
  ([controlled comparisons](../validation/validation-strategy.md#controlled-comparisons)). Recent
  adoptions were decided by the user on paired public scores
  ([adoption history](../modeling/current-model.md#adoption-history)).
- **Fold and seed rotation:** candidates use their control's fold and seed; each adoption merged
  into main retrains the control with a fresh seed on the next fold, except the 2026-09-30
  adoption, which kept fold 0 and its seed (user decision). Never compare scores across
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
  0.944 public (S78).
- [RadImageNet backbones](radimagenet-backbone.md): the ResNet-18 recipe ties the ReXNet control
  publicly at about 60% of its training time; the Refined ResNet-50 beats ResNet-18 by +0.0046 at
  1.65× its training time, and with the ReXNet control scores 0.951 (S89).
- [Report attribute extraction](report-attribute-extraction.md): the JevK5 heads are in the recipe;
  a v4 question set is open.
- [Loading speedup](loading-speedup.md): the `inference-opt` branch awaits a decision.
- [EfficientNet backbones](efficientnet-backbones.md): the B0 gap under class-balanced BCE and a B3
  follow-up are open.

## Recently closed

- **2026-10-03:** the `learnable-views` autoresearch campaign (view width, shift, label bias and
  slice budget learned by gradient), stopped by the user after 47 scored runs and no adoption by
  rule: learned widths drift to about 110 mm and never find the hand 84/93 mm table, which is
  worth about 0.002; the label bias is the learned recipe's own gain (+0.0017) and is adopted on
  main with the hand views (S101 0.951); learned budget, shifts, a second view per type and a
  size menu tie or lose on the hand views; a label bias per slice depth is the nearest miss
  (+0.0003, 80% [−0.0001, +0.0006]). The learnable build stays on the `view-optimisation` branch
  ([result](../validation/results/2026-10-03-learnable-views-campaign.md)).
- **2026-10-02:** the `smart-crops` autoresearch campaign (finer views and crops on the joint),
  stopped by the user after 29 scored runs: coronal and sagittal T1/PD at 84 mm and fluid at
  93.33 mm adopted (+0.0022, 0.950 public as S93); every added view or crop tied; learned
  placement added nothing over a rule; the sagittal fluid zoom costs Baker's cyst and no tested
  width, shift or routed view repaired it
  ([result](../validation/results/2026-10-02-smart-crops-campaign.md)).
- **2026-10-01:** the `mixed-field` autoresearch campaign (a 160 mm view beside a narrower one at
  224 px), stopped by the user after its forced seeds, and two manual runs: 120 mm on every
  series and middle-slice augmentation adopted; 160 mm added nothing in five designs; the
  contrast split (T1/PD 112 mm, fluid 140 mm) ties 120 mm and was not adopted; two views per
  slice stayed untested ([result](../validation/results/2026-10-01-mixed-field-campaign.md)).
- **2026-10-01:** the `revisits` autoresearch campaign on ResNet-18, stopped by the user:
  plane-contrast readouts and removal of the anatomical bias adopted, and the Refined ResNet-50
  (+0.0046) added as `configs/refined-resnet50.yaml`; one backbone per plane ties; EMA and 18
  hyperparameter runs gave nothing
  ([result](../validation/results/2026-10-01-revisits-campaign.md)).
- **2026-09-30:** the `site-generalization` autoresearch campaign on the sites-out split, stopped
  by the user, and its manual follow-up: L2-SP weight anchoring was the only mechanism that
  reproducibly helped held-out sites and was adopted with the confusion head's removal; the
  unseen-site cost is mostly between-site label-rate offsets, which image-feature debiasing does
  not address ([mechanism audit](../../autoresearch/campaigns/site-generalization/mechanism-audit.md),
  [campaign](../../autoresearch/campaigns/site-generalization/INDEX.md)).
- **2026-09-29:** the `radimagenet-rn18` autoresearch campaign: 23 decided, no series dropout and 36
  training triplets adopted; 36 triplets deferred to the endgame for iteration speed
  ([campaign](../../autoresearch/campaigns/radimagenet-rn18/INDEX.md)). The loop's next topic,
  `mri-signal-preservation`, was stopped by the user after 8 runs without an adoption: intensity
  preprocessing mostly exceeded the training-time ceiling
  ([campaign](../../autoresearch/campaigns/mri-signal-preservation/INDEX.md)).
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
