# Current model

The code carries one model: the full-slice spatial-token recipe with plane-contrast readouts,
defined by [`configs/training.yaml`](../../configs/training.yaml). Since
2026-09-28 it trains on the rebuilt labels `rebuild_state_v2` and the split
`all_5fold_scanner_v2`; the current control and what candidates pair with are in the
[project state](../ongoing/state.md#current-control), and each adoption is listed in the
[adoption history](#adoption-history).

- **Inputs:** a RadImageNet-pretrained ResNet-18 reads every sampled triplet's middle slice,
  repeated on all three channels, at 224 px, 0.625 mm spacing and a 140 mm field. The same
  recipe on the Refined-RadImageNet ResNet-50 is
  [`configs/refined-resnet50.yaml`](../../configs/refined-resnet50.yaml), the submission model
  (+0.0046 ×4 at 1.65× the training time); ResNet-18 stays the control because what helps it has
  carried over so far (user decision, 2026-10-01). The previous ReXNet-200 recipe is kept as
  [`configs/rexnet-200.yaml`](../../configs/rexnet-200.yaml), ResNet-50's ensemble partner.
- **Tokens:** every visually valid backbone location is a token with four additive embeddings.
- **Pooling:** target-specific attention MIL pools position-rich tokens. The two meniscus
  labels see only their own side of the knee; OA stays whole-knee. The side-aware anatomical
  bias with learned meniscus and OA strengths is off (`model.anatomical_bias: false`, since
  2026-10-01); the ReXNet recipe keeps it.
- **Plane-contrast readouts:** each label scores every token with the classifier vector of the
  token's series type (six plane/contrast types, initialized from one shared vector) and sums
  the attention-weighted scores (`model.plane_readout: true`, since 2026-10-01).
- **Half-knee loss (training only):** a shared meniscus tear detector is supervised on each
  half of the knee, so token features learn the medial-lateral split.
- **Three-compartment OA detector:** one detector, shared by Medial, Lateral and PF OA, is read
  over each compartment's region and added to that label's output through a learned scale.
- **Report-attribute heads (training only):** 35 auxiliary heads predict the JevK5 report
  attributes (grades, hedging, silence, tear and contusion location, extra findings) from the
  shared tokens, weighted 0.1 ([design](report-attribute-heads.md)).
- **L2-SP weight anchoring (training only):** the backbone conv weights are pulled toward the
  RadImageNet weights, stem and layer1 at 0.01 and the rest at 0.001, and are exempt from weight
  decay ([result](../validation/results/2026-09-30-rn18-l2sp-no-confusion.md)). The ReXNet
  recipe keeps the [scanner confusion head](scanner-confusion-head.md) instead.
- **Augmentation (training only):** reflection, contrast mixing, affine, bilateral smoothing,
  one-axis resolution degradation, gamma/contrast, GIN, noise and DSU feature statistics
  ([pipeline](schedule-and-augmentation.md#ten-epoch-schedule-and-augmentation)).

There is no additive 3D embedding or token graph. Submission status lives in
[project state](../ongoing/state.md#best-public-submissions) and the
[inference overview](../inference/inference-overview.md).
The [model lineage](../history/model-lineage.md) records how the recipe came about and the
retired variants. [Model shapes](model-shapes.md) traces the tensors, and
[swapping a backbone](backbone-swaps.md#current-budget) gives the parameter and compute budget.

## Adoption history

Newest first. Each row is a user decision unless noted; "control" is the run retrained after the
adoption, which later candidates paired with. Local scores on different folds or splits are not
comparable.

| Adopted | Change | Evidence | Control after adoption |
|---|---|---|---|
| 2026-10-01 | Plane-contrast readouts added and the anatomical attention bias removed, on ResNet-18 and the new Refined ResNet-50 config; ResNet-18 stays the control | In the revisits campaign, one run each: +0.0009 ×4 (80% [+0.0001, +0.0017]) and +0.0009 (80% [+0.0002, +0.0016]), both just under the rule's +0.001 and adopted by the user; the Refined ResNet-50 on the result scored 0.948 public (S88) and 0.951 with ReXNet (S89) ([result](../validation/results/2026-10-01-revisits-campaign.md)) | `no-anatomical-bias` (campaign run, ×4 0.929835) |
| 2026-09-30 | L2-SP weight anchoring added and the scanner confusion head removed, on ResNet-18 only | Fold-0 tie, ×4 +0.00026 (95% [−0.0013, +0.0018]), and 0.943 public (S86) against 0.943 (S82); adopted for unseen-site robustness, measured on the sites-out splits: +0.0015, +0.0009 and +0.0023 held-out ([result](../validation/results/2026-09-30-rn18-l2sp-no-confusion.md)) | No retrain: the adopted run `rn18-l2sp-no-confusion` (`outputs/training/rn18-l2sp-no-confusion`), seed 1533697213, v2 fold 0: ×4 0.928062; fold and seed kept by user decision |
| 2026-09-29 | RadImageNet ResNet-18 with the repeated middle slice, `backbone_lr` 5e-5 and no series dropout replaces ReXNet-200 (kept as `configs/rexnet-200.yaml`) | Ties ReXNet publicly (0.944, S79, the 36-triplet version, against S78) at 63% of the training time; no series dropout +0.0023 in the [campaign](../../autoresearch/campaigns/radimagenet-rn18/INDEX.md); the two ensemble to 0.949 (S80) | `rn18-no-series-dropout` (`outputs/autoresearch/radimagenet-rn18/experiments/rn18-no-series-dropout-20260928T213320Z-0b780262`), seed 1533697213, v2 fold 0: ×4 0.927806 |
| 2026-09-28 | Rebuilt labels `rebuild_state_v2` and split `all_5fold_scanner_v2` | A, the pre-fix version of A2, 0.947 public (S76) against 0.946 (S73) on fold 4; A2's control 0.944 (S78) ([label rebuild](../public-labels/label-rebuild.md)) | `rebuild-state-v2-control`, seed 1533697213, v2 fold 0: ×4 0.928288 |
| 2026-09-27 | Scanner confusion head | Fold 3, seed 2617320043: ×4 0.904442 against `half-stack-control` (−0.00076, 80% [−0.0027, +0.0011]); 0.948 public (S67) against 0.947 (S68), one rounding step ([evidence](scanner-confusion-head.md#evidence)) | `scanner-confusion-control`, seed 703791404, fold 4: ×4 0.909733 (×1 0.902061, epoch 9; scanner head accuracy 0.311, loss 1.732); 0.946 public (S73) |
| 2026-09-26 | Half-stack augmentation: bilateral smoothing, DSU, intra-study contrast mixing, GIN, one-axis resolution degradation | Fold 2, seed 1704794368: +0.0011 ×4 (0.899054, 80% [−0.0015, +0.0038]) against smoothing + DSU; 0.943 public (S64, epoch 7; S65, epoch 9, also 0.943) against 0.940 (S63). The full stack (plus channel reversal and stochastic depth) added only +0.0005 and was not adopted ([campaign](../validation/results/2026-09-26-augmentation-campaign.md)) | `half-stack-control`, seed 2617320043, fold 3: ×4 0.905205 (×1 0.90207, epoch 9); 0.947 public (S68) |
| 2026-09-25 | Report-attribute heads on target set 3.1 | Against `oa-detector-control`: +0.00105 ×4 (95% [−0.0012, +0.0035]); 0.940 public (S62) against 0.938 (S60); the first version on raw v3 targets tied at 0.938 (S61) ([result](../validation/results/2026-09-25-report-attribute-heads.md)) | `report-attributes-control`, seed 2464012069, fold 2: ×4 0.898663 (best epoch 7, ×1 0.890335) |
| 2026-09-25 | Three-compartment OA detector | Narrowly missed the local rule (+0.0013, 80% [−0.0002, +0.0029]) but scored 0.944 public (S58) against 0.940 for its same-seed reference (S55) ([campaign](../validation/results/2026-09-25-near-misses-campaign.md)) | `oa-detector-control`, seed 1485198267, fold 1, the first fold rotation: ×4 τ 1 0.900568; 0.938 public (S60) |
| 2026-09-24 | Robust study centre and per-knee geometry for bilateral studies, with a fresh seed | At the new seed +0.0014 (95% [−0.0022, +0.0053]) against the same model without them (`adopted-new-seed`, 0.899547); both 0.940 public (S55 with them, S56 without), so the drop from S54's 0.944 is the seed | `knee-geometry-merged-2`, seed 638120755, v1 fold 0: ×4 0.900955; 0.940 public (S55) |
| 2026-09-24 | Knee-geometry model `meniscus-half-knee-mil`: medial-lateral midline at the fitted study centre, meniscus-only own-side attention mask, training-only half-knee MIL loss (autoresearch adoption) | ×4 0.905834 against `k05-clean`'s 0.902760, +0.00307 (80% [+0.00152, +0.00476]); 0.944 public (S54) against 0.942 (S48) ([campaign](../validation/results/2026-09-24-knee-geometry-campaign.md)) | Retrained in the row above |
| 2026-09-23 | Oblique groups and the group embedding removed | Three paired submissions level or lower with them ([history](../history/oblique-groups.md)) | — |
| 2026-09-22 | `trainer.class_balance_exponent: 0.5` | Public curve peaks at k=0.5 (0.945, S41) while fold 0 rises with k ([result](../validation/results/2026-09-21-class-balance-exponent.md)) | — |
| 2026-09-22 | Saved predictions at ×4, τ 1 (`validation.prediction_image_multiplier: 4`) | Measurement change ([validation strategy](../validation/validation-strategy.md#what-a-comparison-can-resolve)) | — |
| 2026-09-21 | Combined labels `combined_v1` and split `all_5fold_scanner_v1` | Labels 0.9075 against 0.8927 on gold; first run 0.940 public (S40) against 0.933 for the last single models on Steven v4 ([result](../validation/results/2026-09-21-combined-labels-first-run.md)) | `fold0-combined-scanner`: 0.9020 at ×1 τ 1, 0.9079 at ×4 τ 0.75 |
| 2026-09-21 | Learned anatomical family strengths with fixed 20 mm width; appearance-only pooling, learned widths and the pooled compartment adapter removed | Seed 2263738815: 0.897443 against the matched-seed control's 0.895112 (+0.002331; 80% [+0.000103, +0.004518]) ([result](../validation/results/2026-09-21-anatomical-attention-result.md)) | — |
| 2026-09-19 | Class-balanced BCE | `balanced-bce` 0.895282, replicated by the next queue's control at 0.894037 ([result](../validation/results/2026-09-19-balanced-bce-result.md), [queue](../validation/results/2026-09-19-untried-hyperparameter-queue.md)) | — |
| 2026-09-19 | `data.sampling_sigma_fraction: 0.867` replaces `data.sampling_sigma_mm: 40` | Kept for being free: 0.88649 between its two controls and +0.00016 at ×4 ([result](../validation/results/2026-09-19-relative-sampler-width-result.md)) | — |
| 2026-09-18 | `data.study_triplet_total: 26` | +0.0033 over the control mean, inside the control spread ([queue](../validation/results/2026-09-18-overnight-results.md)); adopted as structural: it only adds triplets, for 5.5% more compute, and studies missing two types stop being under-sampled | — |
| 2026-09-17 | Epoch validation at ×1 | Measurement change; scores sit about 0.0035 below ×2 ([validation strategy](../validation/validation-strategy.md#what-a-comparison-can-resolve)) | — |

## Recipe

`configs/training.yaml` retains the full-slice architecture of `00-baseline`, with
later recipe changes including class-balanced BCE, relative Gaussian width, the
study-total floor, and anatomical attention with learned family strengths. The sampler uses independent per-epoch draws within probability
bands; the experimental ten-epoch schedule was reverted. Historical
`00-baseline` results below describe its archived recipe. Current settings include:

| Setting | Value |
|---|---|
| Inputs | Three adjacent slices per input (`SLICES_PER_INPUT`, a constant) |
| `data.image_size`, `data.spacing_mm` | 224, 0.625 (140 mm) |
| `data.sampling_sigma_fraction`, `data.validation_image_multiplier` | 0.867, 1 (×2 before 2026-09-17; scores differ by about 0.0035, and submissions set their own count) |
| Training triplet draws | One independent random draw per probability band at each epoch; [implemented sampler](../data-preparation/physical-sampling/implemented-sampler.md) |
| `trainer.class_balance_exponent` | `0.5` |
| `data.study_triplet_total` | 26: a study whose per-type budgets sum below 26 training triplets is scaled up to it, proportionally over the types it has |
| `data.reflection_augmentation.probability` | 0.5 |
| Other augmentation | Contrast mixing 0.3, affine 0.7, smoothing 0.2, resolution 0.25, gamma/contrast 0.5, GIN 0.3, noise 0.15, DSU 0.5 ([pipeline](schedule-and-augmentation.md#ten-epoch-schedule-and-augmentation)) |
| `model.backbone`, `model.pretrained_weights`, `model.slice_channels` | `resnet18.a1_in1k`, `radimagenet` ([Refined-RadImageNet](../research/radimagenet-weights.md#refined-radimagenet-resnets)), `repeated_middle` |
| `optimizer.backbone_lr` | 5e-5 (weight-RMS rule; 2e-4 for ReXNet-200) |
| `optimizer.anchor_alpha_early`, `optimizer.anchor_alpha_late` | 0.01 (`conv1`, `bn1`, `layer1`), 0.001 (the rest): L2-SP toward the pretrained backbone weights; 0 in the ReXNet recipe |
| Projection and attention widths | 256 and 128 (`PROJECTION_WIDTH`, `MIL_ATTENTION_WIDTH`, constants) |
| `model.dropout`, `model.series_dropout` | 0.0, 0.0 |
| Anatomical attention | Fixed `sign * exp(log_strength) * tanh(medial_mm / 20)` for medial/lateral meniscus and OA; two trainable strengths start at 1 |
| Meniscus side mask | Medial/Lateral Meniscus attend only to own-side tokens plus the ±10 mm band (`COMPARTMENT_MIDLINE_BAND_MM`); unknown side, invalid geometry or no own-side token keep all tokens |
| `trainer.half_knee_loss_weight` | 0.075 |
| `trainer.report_attribute_loss_weight`, `paths.report_attribute_manifest` | 0.1, `labels/jevk5-report-attribute-targets-v3.1.json` |
| `trainer.scanner_confusion_weight` | 0: the head is not built (0.1 in the ReXNet recipe) |
| `labels.source`, `validation.split` | `rebuild_state_v2`, `all_5fold_scanner_v2` |
| `validation.prediction_image_multiplier` | 4: saved predictions score the ×1-selected checkpoint at ×4, τ 1 |
| `seed`, `validation.folds` | 1533697213, `[0]`: the control's seed and fold, inherited by new candidates ([rotation rule](../validation/validation-strategy.md#controlled-comparisons)) |
| Precision | BF16 mixed on one automatically selected device (`PRECISION`, a constant) |

The optimizer, schedule and trainer settings are described in
[training schedule and augmentation](schedule-and-augmentation.md) and
[training runtime and memory](training-runtime-and-memory.md). Slice budgets are in
[triplet inputs](triplet-inputs-and-sampling.md).

The backbone trains in ordinary training mode: `backbone_eval: false` and
`checkpoint_backbone: false` (the config rejects activation checkpointing without
`backbone_eval: true`). Lightning owns precision, optimization, gradient clipping, early
stopping, logging and independent best and latest checkpoints. The saved checkpoint
configuration governs prediction; later edits to the YAML do not change an existing artifact.
[CoAtNet](coatnet-224-spatial-backbone.md) is supported but is selected only by an explicit
`model.backbone` change.

## Architecture

A shared linear projection maps `[N, C, H, W]` to `[N, H × W, 256]`. ReXNet-200 gives 49
locations per 224 px triplet, so the 64-triplet validation budget yields up to 3,136 tokens per
study before masking. The MIL head uses no global pooling, regional queries, token interaction
or grid reduction. The scorer stays 256→128→12 with its configured dropout and is followed by
per-target weighted features and linear heads.

Tokens receive four additive embeddings:

| Embedding | Definition |
|---|---|
| 1D | Sinusoidal middle-slice distance in mm from the acquisition midpoint |
| 2D | Augmented-input anchor, normalized independently to each image axis; 2→64→256 GELU MLP |
| Plane | Learned Axial, Coronal and Sagittal embeddings |
| Contrast | Learned metadata pairs `(0,0)` and `(1,1)` for fluid sensitivity and fat suppression |

`encode_batch` runs, in order:

1. **Series dropout:** 0.20 per present series in training. A study that would lose every
   series keeps one at random.
2. **Reflection:** one draw per study in training, masked by
   [acquisition eligibility](../data-preparation/reflection-geometry-and-previews.md#reflection-geometry-and-eligibility).
   Axial and coronal triplets reverse columns. Sagittal triplets reverse channels and negate the
   1D position.
3. **Augmentation:** the training families after reflection, in their fixed order and mostly
   drawn once per acquisition; DSU acts later, on backbone features. See the
   [schedule and augmentation](schedule-and-augmentation.md#ten-epoch-schedule-and-augmentation) page.
4. **Backbone:** normalization, then one ReXNet-200 `forward_features` pass over all retained
   triplets (`[N, 2560, 7, 7]`), then the shared 2560→256 projection.
5. **Geometry,** in float32 outside autocast:
   - *Anchors:* backbone anchors come from the verified mapping and give the 2D input.
   - *Validity:* each anchor passes through the executed inverse augmentation and reflection.
     It is visually valid when it lies inside all three channels' prepared valid rectangles.
6. **Embeddings:** 1D, 2D, plane and contrast, cast to the visual-feature dtype.

`pool_encoded` then applies these steps:

- **Tokens:** add the embeddings to the projected features and apply the shared `mil_norm`
  LayerNorm.
- **Attention:** with `anatomical_bias`, add the anatomical term to the four compartment-label scores; then run one
  float32 masked softmax per study and target over
  the study's visually valid tokens. For the two meniscus labels the mask also drops tokens on
  the sibling side beyond ±10 mm. A study without valid tokens fails before the softmax.
- **Heads:** with `plane_readout`, score each token with its series type's classifier vector and
  pool the scores with the attention weights; otherwise pool the weighted features and apply one
  classifier vector per label. Twelve biases, and train with the masked soft-target BCE plus the half-knee loss.
- **Half-knee loss:** one linear detector (`half_knee_detector`, shared by both menisci) scores
  the normalized tokens beyond ±10 mm; a float32 smooth-max per half is trained with soft BCE and
  the class-balance weights against the matching meniscus label, for studies with resolved side,
  valid geometry and both halves present. Weight 0.075. Prediction never uses the detector.
- **OA detector:** one linear detector shared by the three
  OA labels. Medial OA pools it with a smooth-max over the medial half and Lateral OA over the
  lateral half (both beyond ±10 mm); PF OA pools it over tokens in front of the knee centre. Each
  pooled score is added to the label's logit through its own zero-initialized scale. Tokens
  without trusted geometry belong to no region. Used at inference.
- **Report-attribute heads:** `ReportAttributeHeads` in the Lightning module, outside
  `KneeAbnormalityModel`, gives each report-attribute question its own attention query over the
  normalized tokens and a softmax over its options. The reliability-weighted soft cross-entropy
  against the JevK5 targets is added with weight 0.1. Prediction never uses the heads
  ([report-attribute heads](report-attribute-heads.md)).
- **L2-SP:** the Lightning module keeps a copy of each pretrained backbone conv weight as a buffer
  and adds `alpha × ‖w − w₀‖²` per weight to the loss, with the early alpha for names starting
  `conv1.`, `bn1.` or `layer1.`; a positive early alpha on a backbone without those names fails.
  Anchored weights are excluded from weight decay. Prediction never uses the buffers.

The plane and contrast lookup buffers derive from the configured metadata, not from slot order.
The ReXNet adapter derives stride 32 and origin 0 from its convolutions. The
[CoAtNet adapter](coatnet-224-spatial-backbone.md) derives stride 32 and origin 12, and any
other backbone mapping fails explicitly. Grid dimensions may vary independently.

Patient-coordinate geometry supplies the anatomical bias, not another additive embedding.
The audited side resolver uses DICOM laterality and conservative coordinate consensus;
unknown side and invalid geometry contribute zero bias. The medial axis is knee-side sign
(-1 left, +1 right) times patient LPS X minus the fitted centre's X (`estimate_reference`).
On 20 studies where axial or axial+coronal medians of series x disagree with it by more than
5 mm, the fitted x was closest to the intercondylar notch in 16 (median error 1.0 mm, against
2.9 mm for the axial median; `outputs/analysis/midline-notch-marks-20260924/result.md`).
Side and centre are carried per series:

- **Robust centre:** a rejected fit is refitted after dropping the worst series above 10 mm,
  then falls back to the sagittal fluid slab centre, then any sagittal series. The sagittal
  fallbacks only steer sampling: as a midline they sit a median 13–23 mm from the fitted
  centre, so those studies get no anatomical geometry (never triggered in training data;
  `outputs/analysis/sagittal-fallback-midline-20260924/`). A robust centre outside a series'
  acquired range samples that series locally. Token geometry no longer
  requires physical sampling.
- **Bilateral studies:** at least two series beyond +40 mm and two beyond −40 mm in X mark two
  knees; each knee gets its own fitted centre and its side from the sign of X (+X is patient
  left). A single series on the other side is a stray: it is still sampled but has no geometry.
- Geometry sources are cleaned series with trustworthy local order and a complete triplet run. The middle-slice patient map
and inverse augmentation locate each token. Reflection keeps coordinates consistent.

- **Geometry output:** `feature_geometry` returns `(xy, visual_valid, prepared_xy)`.
- **Batches:** include `patient_maps`, `geometry_valid`, `knee_side_sign` and
  `knee_center_x_mm`. All three channels still determine visual validity.
- **Parameters:** meniscus and OA each learn one positive `exp(log_strength)`;
  the spatial width is fixed at 20 mm. No appearance-only, learned-width or pooled-adapter
  branches remain. Twelve independent classifier heads are retained.

The model lives in `src/knee_abnormality/modeling/` (`model.py`, `spatial_tokens.py`,
`augmentation.py`). The spatial-token identity is
`all_valid_spatial_tokens_four_embeddings_v6`, inside preprocessing identity
`ranked_filtered_physical_gaussian_overlap_spatial_tokens_reflection_v10` since 2026-09-17
([series exclusions](../data-preparation/current-preparation.md#series-indexing-guards-and-exclusions)
and the fixed [source ranking](../data-preparation/physical-sampling/implemented-sampler.md));
checkpoints trained before then, including the 2026-09-17 controls, record `…reflection_v9`
([identity rules](../architecture/configs-and-experiment-artifacts.md#experiment-artifacts-and-reproducibility)).

## `00-baseline` result

The run is `outputs/training/overnight-20260914T220805Z-ozsz4qbu/runs/00-baseline`: fold 0 of
`all_5fold_v3`, seed 42, 10 epochs.

- **Validation:** epoch 10 at **0.892763** macro AUC, soft-target BCE 0.44767 and gold-label
  AUC 0.932011.
- **Resources:** 1,670 s of training and a peak allocation of 20.02 GB
  ([overnight results](../validation/results/2026-09-15-overnight-results.md)).
- **Public score:** **0.934**, against the champion's 0.936
  ([submission](../inference/submissions/2026-09-15-overnight-00-baseline-submission.md)).
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
  `configs/training.yaml` wrote a `validation_predictions.csv` byte-identical to the saved one
  (macro AUC 0.8927627). A scratch copy of the checkpoint relabelled to the new identity and
  config gave the same bytes through `scripts.pipeline.predict`, and through an exported
  bundle's own code.
- **Training randomness:** the random stream differs from the original run, because the unused
  graph drew one `torch.rand` per study. Evaluation is unaffected, but a new run is not a replay
  of `00-baseline`.

## Reproduce

Train the recipe on its configured fold (`validation.folds`) in a fresh run directory:

```bash
rtk proxy uv run --locked just train configs/training.yaml --run-id RUN_ID
rtk proxy uv run --locked python -m scripts.pipeline.train configs/training.yaml --run-id RUN_ID
```

The run writes its effective configuration, provenance, source snapshot, checkpoints, epoch
metrics and `fold_<N>/validation_predictions.csv`. `--validation.folds '[0,1,2,3,4]'` trains
every fold, and only then does the run root get `oof_metrics.json` beside
`oof_predictions.csv`. `--limit-train N` trains on N studies per fold for development runs;
validation stays complete. Recompute a new run's validation predictions
with `just predict RUN_DIR/effective_config.json --checkpoint CHECKPOINT --output OUTPUT_CSV --split validation`, and compare two runs
with the [fold-run comparison](../validation/fold-run-comparison.md).

The `00-baseline` checkpoint itself records identity `…token_graph_v3`, so it no longer loads
through the current code. Use its `source_snapshot.zip` or the bundle
`outputs/submission/exports/knee-overnight-00-baseline-20260915`; historical checkpoints are
never relabelled in place.
