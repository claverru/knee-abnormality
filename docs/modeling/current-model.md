# Current model

**The final model is one architecture trained on three backbones (ConvNeXt-Tiny, the
Refined-RadImageNet ResNet-50 and NextViT-S), each from its own config
(`configs/convnext.yaml`, `configs/resnet50.yaml`, `configs/nextvit.yaml`) and from the
teacher-student variants of those configs.** The ensemble built from it, the per-family
settings and the teacher-student chain are in the [final solution](../final-solution.md); this
page describes the shared model, the settings every family has in common and how each part was
adopted. Until 2026-10-13 the recipe ran on a RadImageNet ResNet-18 with the ResNet-50 as the
submission model, and before 2026-09-29 on ReXNet-200; the [model lineage](../history/model-lineage.md)
records the earlier designs.

- **Inputs:** every series type is read through fixed-size views on a 368 px canvas: axial
  series at the 120 mm frame, coronal T1/PD at 84 mm, coronal fluid at 93.33 mm, sagittal T1/PD at
  93.33 mm and sagittal fluid at 84 mm, plus a 64 mm detail view of each sagittal fluid slice (the [view table](#view-table)). Each slice is
  repeated on all three channels at 224 px. A first look at ×1 without gradient finds the knee;
  every view, axial included, is then shifted, turned and scaled to the knee's own pose read from
  the anatomy map before the bicubic cut
  ([posed crops](../../autoresearch/campaigns/anatomy-masks/prior-work.md#the-incumbent),
  [posed canvases](../../autoresearch/campaigns/view-poses/INDEX.md)).
- **Tokens:** every visually valid stride-32 backbone location is a token: a shared projection to
  256 channels plus plane/contrast and detail-view embeddings, LayerNorm. Tokens carry their
  patient position, the knee frame and the knee side; the per-label channels also add a fixed
  sinusoid of the token's position in the knee frame, the shared channels stay position-free.
- **Pooling:** nine attention channels (seven per-label channels and one shared meniscus and one
  shared OA channel, 256→128→9) with a learned view bias per view and label and a position prior
  per target, softmax over the study's valid tokens. The two shared channels score tokens without
  knowing the compartment; [side-routed heads](side-routed-heads.md) assign their evidence to the
  medial, lateral and patellofemoral targets through a knee frame (eight sectors, a core, an
  outer ring and three heights, hard cuts with a soft gradient) and add a knee-level term. Each
  token is scored with its series type's classifier vector (plane-contrast readout).
- **Additional reads:** a three-compartment OA detector adds a correction to the three OA logits;
  a second read of the stride-16 cells, routed by the soft-tissue map, adds a logit to the
  meniscus and OA targets; Contusion and Fracture are routed by the bone map.
- **Training-only heads:** a seven-class anatomy map per token (`trainer.anatomy_loss_weight`
  0.1, bones and joint from MRAnnotator masks), a seven-class soft-tissue map on the stride-16
  grid (`trainer.meniscus_loss_weight` 0.1, from the nnU-Net DESS masks) and a knee-side head (loss
  weight 0.1).
- **Loss and regularisation:** masked class-balanced BCE (exponent 0.5) on soft targets, which in
  the teacher-student configs mix silver labels with teacher predictions; L2-SP anchoring of the
  backbone to its starting weights; the augmentation stack of
  [schedule and augmentation](schedule-and-augmentation.md#ten-epoch-schedule-and-augmentation).

[Model shapes](model-shapes.md) traces the tensors, and [backbones](backbones/INDEX.md) holds the
per-family code and recipes.

## Adoption history

Newest first. Each row is a user decision unless noted; "control" is the run retrained after the
adoption, which later candidates paired with. Local scores on different folds or splits are not
comparable. Rows before 2026-10-13 describe the ResNet-18 control recipe and, where stated, the
ResNet-50 config beside it.

| Adopted | Change | Evidence | Control after adoption |
|---|---|---|---|
| 2026-10-19 to 2026-10-22 | Final ensemble: the three families of the teacher-student chain, gen 2 and gen 3 on the v2 and v3 labels, 60 members at ×4, τ 0.4, mirror TTA | S188 0.968 public, first of seven 0.968s ([final solution](../final-solution.md)) | — |
| 2026-10-19 | Silver labels `rebuild_state_v3` beside `rebuild_state_v2` (report-reading rules on silver rows; split `all_5fold_scanner_v3` with v2's folds) | Gen-1 three-family gold 0.95340 against 0.94666; on the board v3 gen 2 tied v2 gen 2 (S183, S181) ([result](../validation/results/teacher-student/2026-10-19-labels-v3-final-ensembles-and-students.md#silver-labels-v3)) | — |
| 2026-10-18 | NextViT-S as a family (`configs/nextvit.yaml`) | Backbone-families-2 campaign on fold 2: ×4 0.938851, ×1 0.934578; blend with the ResNet-50 +0.003267; S178 2nd of the eleven 0.965s (S182 is first) with ConvNeXt and ResNet-50 ([result](../validation/results/backbones/2026-10-18-backbone-families-2.md)) | — |
| 2026-10-16 to 2026-10-17 | Pretraining on fastMRI + OAI pseudo-labels, then Kaggle-only fine-tuning with teacher share 0.75 | S160 0.961 against S158 0.957 for the pretraining; S167 0.964 against S160 for the share ([result](../validation/results/teacher-student/2026-10-15-silver15-round-and-teacher-share.md)) | — |
| 2026-10-14 | One config per model (`configs/resnet50.yaml`, `configs/mambavision.yaml`, `configs/convnext.yaml`; FocalNet-T and EfficientViT-B2 configs followed on 2026-10-16). Dead paths removed: weight averaging, the report-attribute heads, the scanner confusion head, dropout and series dropout, activation checkpointing and frozen backbone, unused folds, the ReXNet, EfficientNet and CoAtNet mappings, the ResNet-18, ResNet-10t and original RadImageNet ResNet-50 weights; plane readout, view label bias, anatomical frame, meniscus map and soft-tissue routing became fixed behaviour | Checked against the previous code for all three configs: identical initial weights, optimizer groups and random-number state after a seeded training step, and forward outputs within each model's rerun spread. Two renumbered layer keys (`attention.2`, `soft_attention.2`) mean earlier checkpoints export from their own `source_snapshot.zip` ([list](../architecture/configs-and-experiment-artifacts.md#experiment-artifacts-and-reproducibility)) | — |
| 2026-10-14 | Mask decoder removed (user decision): no decoder, no stride-8 cell geometry and no decoder losses in training; the anatomy map on the tokens and the meniscus map on the 14 by 14 grid keep their supervision. Checkpoints that carry `mask_decoder.*` weights do not load into the current model | No run. When adopted on 2026-10-13 it was ×4 +0.0005 (80% [−0.0002, +0.0013]) and ×1 +0.0014 on fold 2, one run ([result](../validation/results/campaigns-and-queues/2026-10-13-targets-and-inputs-campaign.md)) | None retrained; run 27 of the targets-and-inputs campaign, which carries the decoder, stays the reference |
| 2026-10-13 | `configs/resnet50.yaml` becomes the Refined-RadImageNet ResNet-50 on fold 2 and the ResNet-18 recipe and `configs/refined-resnet50.yaml` are removed. Three adoptions of the targets-and-inputs campaign: a mask decoder trained beside the labels (second to fourth stage maps, bone and soft-tissue classes on the 28 by 28 grid, weight 0.1 each, not run at prediction); the three-compartment OA detector as a correction beside the routed shared OA head; average-pool shortcuts in the ResNet-50 on the same weights (user decision against the rule's ×1 side) | Fold 2, seed 1110484336, one run each against the incumbent of its time: decoder ×4 +0.0005 (80% [−0.0002, +0.0013]), ×1 +0.0014; detector ×4 +0.0011 ([+0.0004, +0.0018]), ×1 +0.0014; shortcuts ×4 +0.0010 ([+0.0002, +0.0019]), ×1 −0.0003. Together ×4 0.939634 against 0.937000, ×1 0.930902 against 0.928460. No public score ([result](../validation/results/campaigns-and-queues/2026-10-13-targets-and-inputs-campaign.md)) | None retrained; the campaign's run 27 is the reference |
| 2026-10-10 | Meniscus map and posed canvases on ResNet-18 and the ResNet-50 config: report-attribute loss off; a three-class meniscus map taught at weight 0.1 from the DESS masks on the 14 by 14 grid; every view posed on a 368 px canvas with shift, turn and per-knee scale, axial included; sagittal fluid at 84 mm and sagittal T1/PD at 93.33 mm | Fold 1, seed 1110484336, one run per step in the anatomy-recipe and view-poses campaigns: report loss off ×4 0.927291; meniscus head on 14 by 14 +0.0001 (80% interval −0.0006 to +0.0008); canvases +0.00002 (−0.0014 to +0.0014); sagittal swap +0.0016 (+0.0004 to +0.0027). Together ×4 0.928121 against 0.927083, 0.955 public (S129) against 0.954 (S125). The two middle steps were adopted under a both-multipliers rule used for that campaign only | `sagittal-fluid-canvas-fine` (×4 0.928121, ×1 0.919392), not retrained; a retraining of the same code gave ×4 0.928120 |
| 2026-10-08 | Anatomy map and posed crops on the side-routed ResNet-18: a seven-class anatomy map taught per token from an offline segmenter's masks (training only), the joint plane and the joint centre read from it, sections by anatomy, sagittal and coronal crops turned and shifted to the knee's pose after a no-gradient first look and cut with bicubic interpolation; `model.channels_last` on; each series of a two-knee study posed with its own knee's side | Fold 1, seed 1110484336, one run per step: the campaign's reference ×4 0.925885 against run 24's 0.925067 (0.953 public for both, S122 and S118); the bicubic cut +0.0012 (95% [+0.0000, +0.0024]); the map's centre +0.0000 on top (adopted by user decision); 0.954 public (S125); the ResNet-50 config carries the same settings and scored 0.959 public (S126) against 0.957 without the map (S119) ([run notes](../../autoresearch/campaigns/anatomy-masks/run-notes.md)) | `rn18-anatomy-map-centre-control`: ×4 0.926762, ×1 0.917074, 1,391 s |
| 2026-10-07 | Side-routed shared heads on ResNet-18 and the ResNet-50 config: one shared classifier for the two meniscus targets and one for the three OA targets, routed by a frame inferred per knee from the heads' meniscus attention spots, eight sectors, a core, two rings and three heights, routed v4.0 report targets and a knee-level term per routed channel; both configs at fold 1, seed 1110484336. It removed the meniscus own-side mask and its band, the half-knee loss, the 1D slice and 2D image position embeddings and the per-compartment weights of the five compartment targets | Run 24 of the line: ×4 0.925067 against 0.924049 without routing on the same fold and seed; 0.953 public (S118) against 0.950; the ResNet-50 0.957 (S119) ([design](side-routed-heads.md), [runs](../validation/results/pooling-and-attention/side-routed-heads/INDEX.md)) | Run 24, `outputs/training/20261006-163650-side-routed-spot-frame-ring-sectors`: ×4 0.925067, ×1 0.917070 |
| 2026-10-06 | Four changes from the fixes-and-removals campaign, on ResNet-18 and the ResNet-50 config: detail-view positions mirrored under reflection in training and TTA; the knee centre used when the side is unknown; the slab rule for every centre-fit method with the feature-statistics std clamped at zero; the three-compartment OA detector removed. Both configs move to fold 1 with seed 819238850 | Fold 1, seed 1110484336, one run each against the incumbent at its turn: +0.0003 (80% [−0.0002, +0.0008], forced as a bug fix), +0.0014 (80% [+0.0003, +0.0026], by the rule), −0.0003 (80% [−0.0015, +0.0009], user decision) and −0.0005 for the detector (80% [−0.0010, +0.0001], user decision; its three OA labels do not fall, and 0.950 public with it, S108, and without it, S111); together ×4 0.924049 against 0.923102. The detector was re-adopted on 2026-10-13 (first row of this table) ([result](../validation/results/campaigns-and-queues/2026-10-05-fixes-and-removals-campaign.md)) | No retrain: the campaign incumbent `slab-rule-and-positive-std-requeue`, fold 1, seed 1110484336: ×4 0.924049 |
| 2026-10-05 | Checkpoint input normalization: `data.mean` 0.3162 and `data.std` 0.3213 (the statistics the Refined-RadImageNet checkpoints were trained with) in place of the ImageNet values, in `configs/resnet50.yaml` and `configs/refined-resnet50.yaml` (user decision 2026-10-04; the ResNet-50 config takes it without a run of its own) | In the [Refined-RadImageNet campaign](../validation/results/backbones/2026-10-04-refined-radimagenet-campaign.md), one run on ResNet-18: ×4 0.933058 against 0.932477, +0.000581 (80% [+0.000182, +0.001016]), ×1 0.929948 against 0.929026; ten of twelve labels up | The campaign incumbent itself, `radimagenet-input-statistics-20261004T103433Z-ff2f3786` under `outputs/autoresearch/refined-radimagenet/experiments/` (the adopted recipe, ×4 0.933058, ×1 0.929948; user decision 2026-10-05, no retraining) |
| 2026-10-04 | ReXNet-200 retired: `configs/rexnet-200.yaml` removed, models stay on the Refined-RadImageNet weights (user decision). Code paths kept (hashed sources) | Five ResNet-50 children alone score 0.957 public (S106) against 0.955 with five ReXNet children added (S103), although the ReXNets add +0.0026 pooled out of fold; labels with the ResNet-50 teacher alone agree better with the 58 gold studies than with the ReXNet teacher added (0.9394 against 0.9358, 80% [−0.0070, −0.0002] for the difference); ConvNeXt-Tiny reached 0.927 with no blend gain. Architecture, pretraining and anchoring differ together in each comparison ([result](../validation/results/labels-and-losses/2026-10-04-two-family-oof-distillation.md#resnet-50-children-alone-s106)) | unchanged: `rn18-view-label-bias` for ResNet-18, `refined-rn50-views-label-bias` for ResNet-50 |
| 2026-10-03 | The ResNet-18 recipe carried to the two other configs. `configs/refined-resnet50.yaml` gains the 84/93 mm views and the view label bias (backbone only differs from `configs/resnet50.yaml`). `configs/rexnet-200.yaml` becomes the same recipe with ReXNet's backbone settings (ImageNet `timm` weights, adjacent slices, `backbone_lr` 2e-4, no L2-SP): 120 mm field, the views, label bias and plane-contrast readouts added; anatomical bias, scanner confusion head and series dropout removed | Fold 0, seed 1533697213, one run each. ResNet-50: ×4 0.937129 against 0.934423 for `refined-rn50-incumbent-recipe` (+0.0027, 80% [+0.0010, +0.0045]; ACL +0.007, MCL +0.008, lateral meniscus +0.006), 0.954 public alone (S102) against 0.948 (S88). ReXNet: ×4 0.932523 against 0.928288 for `rebuild-state-v2-control` (+0.0042, 80% [+0.0018, +0.0066]; MCL +0.021, contusion +0.011, synovitis +0.010, lateral meniscus −0.006), not submitted alone; its changes were made together and are not separated. Equal blend of the two at ×4, τ 1, no TTA: 0.939492, +0.0024 over the ResNet-50 alone (80% [+0.0010, +0.0037]); with the old ReXNet +0.0007; adding the ResNet-18 control to the pair −0.0007 | `refined-rn50-views-label-bias` (`outputs/training/20261003-110045-refined-rn50-views-label-bias`, ×1 0.931222, 1,995 s) and `rexnet-200-views-label-bias` (`outputs/training/20261003-114512-rexnet-200-views-label-bias`, ×1 0.928556, 2,349 s), both trained from main at `a5b8758` with command-line overrides equal to these configs |
| 2026-10-03 | View label bias (`model.view_label_bias: true`) on ResNet-18 only; the view table moves from the view policy to the config in millimetres (`field_mm`, `data.reference_field_mm`) with unchanged outputs | On the `view-optimisation` branch, fold 0, one run each: the same hand views with the bias ×4 0.933307 (×1 0.9299) against 0.932627 (0.9290) without, +0.0007; the learned-view recipe without it 0.930803 against 0.932489 with, −0.0017 (80% interval excludes zero) and below from epoch 3; every learned run ends at the same bias table. Weak evidence on the hand views alone (+0.0007 is inside the 0.0008 spread of comparable runs); adopted by the user on the agreement across runs. Main's implementation equals the recipe in force at the zero start (outputs differ by 0.0); the run from main is in the last column | `rn18-view-label-bias` (`outputs/training/20261003-100844-rn18-view-label-bias`, fold 0, seed 1533697213): ×4 0.932477, ×1 0.929026, against `fluid-detail-93-replacement` 0.932880 and 0.929261: −0.0004 ×4 (95% [−0.0008, +0.0001], paired, 851 report groups), −0.0002 ×1. Lateral OA −0.0019, lateral meniscus −0.0017, ACL −0.0011, the other nine within ±0.0005. The learned table matches the branch run's (correlation 0.98 over 72 entries, largest ±0.10: Baker's cyst towards axial fluid and away from coronal T1/PD, effusion towards axial fluid). 1,181 s against 1,134 s. Public 0.951 (S101) against 0.950 for the control (S93) and 0.950 for the branch run of the same recipe (S100), one rounding step (comparison: `outputs/analysis/r/rn18-view-label-bias-vs-control-20261003/comparison.json`) |
| 2026-10-02 | Coronal and sagittal T1/PD at 84 mm (0.375 mm) and fluid at 93.33 mm (5/12 mm) in place of the 120 mm view, rule-placed, on ResNet-18 only (`data.views.policy: detail_84_93`); axial stays at 120 mm | In the smart-crops campaign, one run each: T1/PD at 84 mm +0.0013 ×4 (80% [+0.0001, +0.0025]), fluid at 93 mm a further +0.0010 (80% [−0.0000, +0.0020], just under the rule and adopted by the user); together +0.0022 (80% [+0.0007, +0.0037]) and 0.950 public (S93) against 0.947 (S92) at the same inference settings. Known cost: Baker's cyst −0.0040 ([result](../validation/results/crops-and-views/2026-10-02-smart-crops-campaign.md)) | No retrain: the adopted run `fluid-detail-93-replacement` (campaign run, ×4 0.932880, 0.950 public as S93); fold and seed kept |
| 2026-10-01 | 120 mm field at 224 px (`data.spacing_mm` 0.5357, from 0.625 and 140 mm) on ResNet-18 and the ResNet-50 config; reflection, contrast mixing, augmentation and mirror TTA run on the middle slice only in `repeated_middle` mode | On the 2026-09-30 recipe, one run each: 120 mm +0.0011 and +0.0012 ×4 (80% [−0.0004, +0.0026] and [−0.0003, +0.0028]); 0.947 public (S90) against 0.943 (S86) at the same inference settings; middle-slice augmentation neutral (+0.0001, −0.0004) and 6–8% faster per epoch, predictions bit-identical ([result](../validation/results/crops-and-views/2026-10-01-mixed-field-campaign.md)) | `rn18-field-120-control` (`outputs/training/rn18-field-120-control`), seed 1533697213, v2 fold 0: ×4 0.930666, 0.947 public (S92); fold and seed kept ([result](../validation/results/crops-and-views/2026-10-01-mixed-field-campaign.md#control-on-the-adopted-recipe)) |
| 2026-10-01 | Plane-contrast readouts added and the anatomical attention bias removed, on ResNet-18 and the new Refined ResNet-50 config; ResNet-18 stays the control | In the revisits campaign, one run each: +0.0009 ×4 (80% [+0.0001, +0.0017]) and +0.0009 (80% [+0.0002, +0.0016]), both just under the rule's +0.001 and adopted by the user; the Refined ResNet-50 on the result scored 0.948 public (S88) and 0.951 with ReXNet (S89) ([result](../validation/results/campaigns-and-queues/2026-10-01-revisits-campaign.md)) | `no-anatomical-bias` (campaign run, ×4 0.929835) |
| 2026-09-30 | L2-SP weight anchoring added and the scanner confusion head removed, on ResNet-18 only | Fold-0 tie, ×4 +0.00026 (95% [−0.0013, +0.0018]), and 0.943 public (S86) against 0.943 (S82); adopted for unseen-site robustness, measured on the sites-out splits: +0.0015, +0.0009 and +0.0023 held-out ([result](../validation/results/validation-and-ensembles/2026-09-30-rn18-l2sp-no-confusion.md)) | No retrain: the adopted run `rn18-l2sp-no-confusion` (`outputs/training/rn18-l2sp-no-confusion`), seed 1533697213, v2 fold 0: ×4 0.928062; fold and seed kept by user decision |
| 2026-09-29 | RadImageNet ResNet-18 with the repeated middle slice, `backbone_lr` 5e-5 and no series dropout replaces ReXNet-200 (kept as `configs/rexnet-200.yaml`) | Ties ReXNet publicly (0.944, S79, the 36-triplet version, against S78) at 63% of the training time; no series dropout +0.0023 in the [campaign](../../autoresearch/campaigns/radimagenet-rn18/INDEX.md); the two ensemble to 0.949 (S80) | `rn18-no-series-dropout` (`outputs/autoresearch/radimagenet-rn18/experiments/rn18-no-series-dropout-20260928T213320Z-0b780262`), seed 1533697213, v2 fold 0: ×4 0.927806 |
| 2026-09-28 | Rebuilt labels `rebuild_state_v2` and split `all_5fold_scanner_v2` | A, the pre-fix version of A2, 0.947 public (S76) against 0.946 (S73) on fold 4; A2's control 0.944 (S78) ([label rebuild](../public-labels/label-rebuild/INDEX.md)) | `rebuild-state-v2-control`, seed 1533697213, v2 fold 0: ×4 0.928288 |
| 2026-09-27 | Scanner confusion head | Fold 3, seed 2617320043: ×4 0.904442 against `half-stack-control` (−0.00076, 80% [−0.0027, +0.0011]); 0.948 public (S67) against 0.947 (S68), one rounding step ([evidence](scanner-confusion-head.md#evidence)) | `scanner-confusion-control`, seed 703791404, fold 4: ×4 0.909733 (×1 0.902061, epoch 9; scanner head accuracy 0.311, loss 1.732); 0.946 public (S73) |
| 2026-09-26 | Half-stack augmentation: bilateral smoothing, DSU, intra-study contrast mixing, GIN, one-axis resolution degradation | Fold 2, seed 1704794368: +0.0011 ×4 (0.899054, 80% [−0.0015, +0.0038]) against smoothing + DSU; 0.943 public (S64, epoch 7; S65, epoch 9, also 0.943) against 0.940 (S63). The full stack (plus channel reversal and stochastic depth) added only +0.0005 and was not adopted ([campaign](../validation/results/campaigns-and-queues/2026-09-26-augmentation-campaign.md)) | `half-stack-control`, seed 2617320043, fold 3: ×4 0.905205 (×1 0.90207, epoch 9); 0.947 public (S68) |
| 2026-09-25 | Report-attribute heads on target set 3.1 | Against `oa-detector-control`: +0.00105 ×4 (95% [−0.0012, +0.0035]); 0.940 public (S62) against 0.938 (S60); the first version on raw v3 targets tied at 0.938 (S61) ([result](../validation/results/labels-and-losses/2026-09-25-report-attribute-heads.md)) | `report-attributes-control`, seed 2464012069, fold 2: ×4 0.898663 (best epoch 7, ×1 0.890335) |
| 2026-09-25 | Three-compartment OA detector | Narrowly missed the local rule (+0.0013, 80% [−0.0002, +0.0029]) but scored 0.944 public (S58) against 0.940 for its same-seed reference (S55) ([campaign](../validation/results/campaigns-and-queues/2026-09-25-near-misses-campaign.md)) | `oa-detector-control`, seed 1485198267, fold 1, the first fold rotation: ×4 τ 1 0.900568; 0.938 public (S60) |
| 2026-09-24 | Robust study centre and per-knee geometry for bilateral studies, with a fresh seed | At the new seed +0.0014 (95% [−0.0022, +0.0053]) against the same model without them (`adopted-new-seed`, 0.899547); both 0.940 public (S55 with them, S56 without), so the drop from S54's 0.944 is the seed | `knee-geometry-merged-2`, seed 638120755, v1 fold 0: ×4 0.900955; 0.940 public (S55) |
| 2026-09-24 | Knee-geometry model `meniscus-half-knee-mil`: medial-lateral midline at the fitted study centre, meniscus-only own-side attention mask, training-only half-knee MIL loss (autoresearch adoption) | ×4 0.905834 against `k05-clean`'s 0.902760, +0.00307 (80% [+0.00152, +0.00476]); 0.944 public (S54) against 0.942 (S48) ([campaign](../validation/results/campaigns-and-queues/2026-09-24-knee-geometry-campaign.md)) | Retrained in the row above |
| 2026-09-23 | Oblique groups and the group embedding removed | Three paired submissions level or lower with them ([history](../history/oblique-groups.md)) | — |
| 2026-09-22 | `trainer.class_balance_exponent: 0.5` | Public curve peaks at k=0.5 (0.945, S41) while fold 0 rises with k ([result](../validation/results/labels-and-losses/2026-09-21-class-balance-exponent.md)) | — |
| 2026-09-22 | Saved predictions at ×4, τ 1 (`validation.prediction_image_multiplier: 4`) | Measurement change ([validation strategy](../validation/validation-protocol.md#what-a-comparison-can-resolve)) | — |
| 2026-09-21 | Combined labels `combined_v1` and split `all_5fold_scanner_v1` | Labels 0.9075 against 0.8927 on gold; first run 0.940 public (S40) against 0.933 for the last single models on Steven v4 ([result](../validation/results/labels-and-losses/2026-09-21-combined-labels-first-run.md)) | `fold0-combined-scanner`: 0.9020 at ×1 τ 1, 0.9079 at ×4 τ 0.75 |
| 2026-09-21 | Learned anatomical family strengths with fixed 20 mm width; appearance-only pooling, learned widths and the pooled compartment adapter removed | Seed 2263738815: 0.897443 against the matched-seed control's 0.895112 (+0.002331; 80% [+0.000103, +0.004518]) ([result](../validation/results/pooling-and-attention/2026-09-21-anatomical-attention-result.md)) | — |
| 2026-09-19 | Class-balanced BCE | `balanced-bce` 0.895282, replicated by the next queue's control at 0.894037 ([result](../validation/results/labels-and-losses/2026-09-19-balanced-bce-result.md), [queue](../validation/results/campaigns-and-queues/2026-09-19-untried-hyperparameter-queue.md)) | — |
| 2026-09-19 | `data.sampling_sigma_fraction: 0.867` replaces `data.sampling_sigma_mm: 40` | Kept for being free: 0.88649 between its two controls and +0.00016 at ×4 ([result](../validation/results/sampling/2026-09-19-relative-sampler-width-result.md)) | — |
| 2026-09-18 | `data.study_triplet_total: 26` | +0.0033 over the control mean, inside the control spread ([queue](../validation/results/campaigns-and-queues/2026-09-18-overnight-results.md)); adopted as structural: it only adds triplets, for 5.5% more compute, and studies missing two types stop being under-sampled | — |
| 2026-09-17 | Epoch validation at ×1 | Measurement change; scores sit about 0.0035 below ×2 ([validation strategy](../validation/validation-protocol.md#what-a-comparison-can-resolve)) | — |

## Shared settings

Every family config and every teacher-student variant shares these settings
(`configs/resnet50.yaml` and its siblings); per-family backbone settings are in the
[final solution](../final-solution.md#model-and-per-family-recipes).

| Setting | Value |
|---|---|
| `model.slice_channels`, `data.image_size`, `data.reference_field_mm` | `repeated_middle`, 224, 120 (0.5357 mm per pixel in the frame) |
| `data.series_types[].views` | the [view table](#view-table) |
| `data.study_slice_total` | 26: a study whose per-type budgets sum below 26 training slices is scaled up to it, proportionally over the types it has |
| `data.sampling_sigma_fraction`, `data.validation_image_multiplier`, `validation.prediction_image_multiplier` | 0.867, 1, 4 (saved predictions score the ×1-selected checkpoint at ×4, τ 1) |
| Training slice draws | one independent random draw per probability band at each epoch ([implemented sampler](../data-preparation/physical-sampling/implemented-sampler.md)) |
| Augmentation probabilities | reflection 0.5, contrast mixing 0.3 (target weight 0.5–1.0, donor aligned in patient coordinates), affine 0.7 (10°, 4%, scale 0.95–1.05), gamma/contrast 0.5, noise 0.15, smoothing 0.2, resolution 0.25 (keep 0.70–1.0), GIN 0.3, DSU feature statistics 0.5 ([pipeline](schedule-and-augmentation.md#ten-epoch-schedule-and-augmentation)) |
| `trainer` | 10 epochs (patience 10), batch 8, gradient clip 1.0, class-balance exponent 0.5, anatomy and meniscus map losses 0.1 |
| `optimizer` | AdamW, head rate 6e-4, weight decay 0.01 (anchored backbone weights exempt), one warm-up epoch from 0.1 of the rate, floor 0.01 of the rate |
| `channels_last` | on |
| Precision | BF16 mixed on one automatically selected device |

Projection and attention widths are constants (256 and 128). The saved checkpoint configuration
governs prediction; later edits to the YAML do not change an existing artifact. The optimizer and
trainer details are in [training schedule and augmentation](schedule-and-augmentation.md) and
[training runtime and memory](training-runtime-and-memory.md); slice budgets in
[triplet inputs](triplet-inputs-and-sampling.md).

### View table

Every series type (plane and contrast) lists its views under `data.series_types[].views`. A
view is one window size with its own slices:

| Field | Meaning |
|---|---|
| `field_mm` | Width of the view's window in millimetres; its pixel size is `field_mm` over `data.image_size` (84 mm is 0.375 mm, 93.33 mm is 5/12 mm, 120 mm is 0.5357 mm) |
| `slices` | The view's base slice budget: the inputs it selects, each a middle slice read with its two neighbours. A series with fewer positions supplies fewer; `data.study_slice_total` and the image multipliers scale it up |
| `horizontal`, `vertical` | Placement per image axis, both fields written out: `anchor` (`image`, the image centre, or `knee`, the fitted knee centre) and `offset_mm` from it. A plain centred window is `{anchor: image, offset_mm: 0.0}` on both axes |

- **Directions.** An offset from the image centre runs along the image axis (columns to the
  right, rows downwards). An offset from the knee centre is anatomical, in patient
  millimetres: horizontally medial on axial and coronal series and anterior on sagittal ones;
  vertically posterior on axial series and inferior on coronal and sagittal ones.
- **Fallback.** A series without trusted geometry treats a knee-anchored axis as the image
  centre with no offset.
- **Frame.** `data.reference_field_mm` (120 mm) is the frame, a centred square: the reference
  for slice selection and token positions. It was `data.spacing_mm`, the frame's pixel size,
  until 2026-10-03; the frame's pixel size is now derived as the field over `data.image_size`.
  A view as wide as the frame is the plain centred frame and cannot carry an anchor or offset
  (the config rejects it). A finer view is moved to stay
  inside the frame and the stored image, never shrunk; when it cannot fit it is centred. Its
  tokens carry their position in the frame and the shared additive view embedding.
- **Several views per type.** Each view draws its own slices from the type's pool with its own
  budget; the views are independent draws in training and independent band midpoints in
  evaluation. `data.study_slice_total` and the image multipliers apply over all views.
- **Logged per view** under the names `<plane>_<t1pd|fluid>_<width>mm` (a repeated name gets
  `_2`, `_3`): `view_share_`, `attention_mass_<view>_<label>`, `window_center_`,
  `window_rule_distance_`, `val_macro_auc_view_` and `val_auc_view_<view>_<label>` (the
  view's tokens alone).
- **Known limitation.** About 30 view choices (widths, offsets, slice counts, detail views) were
  made on the same 912 fold-0 studies and none was confirmed on another fold
  ([untried ideas](../backlog/untried-ideas.md#views-and-fields)).

The recipe in force written as this table gives loader output bit-identical to the
`data.views.policy: detail_84_93` code it replaced (checked on CPU, 2026-10-02: 200 training
studies, training draws at two epochs and evaluation at ×1 and ×4, every image and geometry
tensor equal), so runs before and after the change are comparable.

## Architecture

The model lives in `src/knee_abnormality/modeling/` (`model.py`, `spatial_tokens.py`,
`backbone_stages.py`, `view_windows.py`, `augmentation.py`); `SPATIAL_TOKEN_IDENTITY` in
`spatial_tokens.py` names every geometric and pooling choice and is hashed into the checkpoint's
identity.

- **Backbone:** read through a stage adapter per family (`BACKBONE_STAGES`): stride-8 maps for
  DSU, stride-16 maps for the soft-tissue read and map, stride-32 maps for the tokens. Token
  anchors come from a verified mapping per backbone (`BACKBONE_MAPPINGS`; ResNet-50 origin 0,
  ConvNeXt 15.5 px, NextViT 14 px at stride 32) ([swapping a backbone](backbones/backbone-swaps.md#code-requirements)).
- **Visual validity:** each anchor passes through the executed inverse augmentation and
  reflection; a token is valid when it lies inside the crop and the original canvas rectangles
  through the same crop transform. A study without valid tokens fails before the softmax.
- **Attention:** per study and channel, a float32 masked softmax over valid tokens of the scorer
  output plus the view label bias, the position prior and, for routed channels, the log routing
  weight over the routing temperature. At prediction τ divides the scorer, view bias and position
  prior and multiplies the routing temperature.
- **Heads:** plane-contrast readouts score each token with its series type's classifier vector
  and pool the scores with the attention weights; twelve target biases; the shared channels'
  pooled evidence enters each routed target through a zero-started channel scale; a knee-level
  term on every routed channel; the OA detector pools its token evidence over the medial
  (beyond +10 mm), lateral (beyond −10 mm) and anterior regions by log-mean-exp and adds it with a
  zero-started scale.
- **Soft-tissue read:** the stride-16 cells get their own LayerNorm and attention (256→128→2) and a
  zero-initialised plane-contrast readout; the seven-class soft-tissue map routes the cells to the
  menisci and to the three OA compartments (cartilage plus assigned femoral cartilage); its logit
  is added to the meniscus and OA targets.
- **Detail view:** the 64 mm sagittal fluid view reuses the first view's slices, pose and
  augmentation draws, and is open only to the menisci, the three OA labels and ACL.
- **Knee side:** from explicit coordinates where they agree with the metadata, otherwise from the
  side head's detached logit with a coordinate prior; in two-knee studies each series takes its
  own knee's side.

### Knee geometry

The audited side resolver uses DICOM laterality and conservative coordinate consensus. The
medial axis is knee-side sign
(-1 left, +1 right) times patient LPS X minus the fitted centre's X (`estimate_reference`).
On 20 studies where axial or axial+coronal medians of series x disagree with it by more than
5 mm, the fitted x was closest to the intercondylar notch in 16 (median error 1.0 mm, against
2.9 mm for the axial median; `outputs/analysis/m/midline-notch-marks-20260924/result.md`).
Side and centre are carried per series:

- **Robust centre:** a rejected fit is refitted after dropping the worst series above 10 mm,
  then falls back to the sagittal fluid slab centre, then any sagittal series. The sagittal
  fallbacks only steer sampling: as a midline they sit a median 13–23 mm from the fitted
  centre, so those studies get no knee geometry (never triggered in training data;
  `outputs/analysis/s/sagittal-fallback-midline-20260924/`). A centre outside a series'
  acquired range samples that series locally, whatever the fit method (since 2026-10-06). Token geometry no longer
  requires physical sampling.
- **Bilateral studies:** at least two series beyond +40 mm and two beyond −40 mm in X mark two
  knees; each knee gets its own fitted centre and its side from the sign of X (+X is patient
  left). A single series on the other side is a stray: it is still sampled but has no geometry.
- Geometry sources are cleaned series with trustworthy local order and a complete triplet run. The middle-slice patient map
and inverse augmentation locate each token. Reflection keeps coordinates consistent.


## Historical reference: `00-baseline`

The run is `outputs/training/overnight-20260914T220805Z-ozsz4qbu/runs/00-baseline`: fold 0 of
`all_5fold_v3`, seed 42, 10 epochs.

- **Validation:** epoch 10 at **0.892763** macro AUC, soft-target BCE 0.44767 and gold-label
  AUC 0.932011.
- **Resources:** 1,670 s of training and a peak allocation of 20.02 GB
  ([overnight results](../validation/results/campaigns-and-queues/2026-09-15-overnight-results.md)).
- **Public score:** **0.934**, against the champion's 0.936
  ([submission](../inference/submissions/early-submission-details.md#reflection-sigma-40-champion-and-its-variants-s19s22-2026-09-14-to-2026-09-15)).
- **Sources:** `source_snapshot.zip` and `effective_config.json` in the run directory. The
  archived sources still contain the token graph with `model.graph_layers: 0`, which was built
  on every forward pass but passed the embedding sum through unchanged. The run queue's other
  results, including `03-min-lr-0.1`, share this base.

**Restoration check (2026-09-16):**

- **Exact restore:** the exact source restore reproduced `00-baseline` bit-identically (macro AUC
  0.8927627, BF16).
- **After graph removal:** a strict state-dict load of the checkpoint into the cleaned model gives
  identical predictions.
- **Package restructure:** the later move into subpackages and the code-review refactors left
  forward outputs (evaluation and training mode), prepared batches, optimizer parameter groups
  and fresh initialization bit-identical on CPU. On one RTX 3090 (BF16, batch size 8, sorted
  study order), a strict load of the `00-baseline` weights into the model built from
  the training config of the time wrote a `validation_predictions.csv` byte-identical to the saved one
  (macro AUC 0.8927627). A scratch copy of the checkpoint relabelled to the new identity and
  config gave the same bytes through `scripts.pipeline.predict`, and through an exported
  bundle's own code.
- **Training randomness:** the random stream differs from the original run, because the unused
  graph drew one `torch.rand` per study. Evaluation is unaffected, but a new run is not a replay
  of `00-baseline`.

## Reproduce

Train a family on its configured fold in a fresh run directory; chain stages start from a
pretraining checkpoint:

```bash
uv run --locked just train configs/convnext.yaml --run-id RUN_ID
uv run --locked python -m scripts.pipeline.train configs/convnext-v3-gen3.yaml --run-id RUN_ID --validation.split all_5fold_scanner_v3_r4 --validation.folds '[0]' --seed SEED --initial-weights outputs/training/convnext-v3-pretrain/fold_0/last.ckpt
```

The split, fold and seed of each final member are in its `effective_config.json`. The run writes
its effective configuration, provenance, source snapshot, checkpoints, epoch metrics and
`fold_<N>/validation_predictions.csv`. `--validation.folds '[0,1,2,3,4]'` trains every fold, and
only then does the run root get `oof_metrics.json` beside `oof_predictions.csv`. `--limit-train N`
trains on N studies per fold for development runs; validation stays complete. Recompute a run's
validation predictions with `just predict RUN_DIR/effective_config.json --checkpoint CHECKPOINT
--output OUTPUT_CSV --split validation`, and compare two runs with the
[run comparison](../validation/run-comparison-and-diagnostics.md).

The `00-baseline` checkpoint itself records identity `…token_graph_v3`, so it no longer loads
through the current code. Use its `source_snapshot.zip` or the bundle
`outputs/submission/exports/knee-overnight-00-baseline-20260915`; historical checkpoints are
never relabelled in place.
