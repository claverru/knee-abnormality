# Current model

The code carries one model: the full-slice spatial-token recipe with learned anatomical
family strengths, defined by [`configs/training.yaml`](../../configs/training.yaml).
The [adoption evidence](../validation/results/2026-09-21-anatomical-attention-result.md) records the closed
position/knee-side experiments; the current control is `knee-geometry-merged-2` (below).

- **Inputs:** ReXNet-200 reads every sampled triplet at 224 px, 0.625 mm spacing and a 140 mm
  field.
- **Tokens:** every visually valid backbone location is a token with four additive embeddings.
- **Pooling:** target-specific attention MIL pools position-rich tokens, with side-aware
  anatomical bias and two learned positive strengths for meniscus and OA. The two meniscus
  labels see only their own side of the knee; OA stays whole-knee.
- **Half-knee loss (training only):** a shared meniscus tear detector is supervised on each
  half of the knee, so token features learn the medial-lateral split.
- **Three-compartment OA detector:** one detector, shared by Medial, Lateral and PF OA, is read
  over each compartment's region and added to that label's output through a learned scale.

Adopted 2026-09-25 by user decision: the three-compartment OA detector from the
[near-misses campaign](../validation/results/2026-09-25-near-misses-campaign.md). It narrowly
missed the local rule (+0.0013, 80% [−0.0002, +0.0029]) but scored 0.944 public (S58) against
0.940 for its same-seed reference (S55). The new control trains at fresh seed 1485198267; its
scores are pending.

Adopted 2026-09-24 from the [knee-geometry campaign](../../autoresearch/campaigns/knee-geometry/INDEX.md)
(`meniscus-half-knee-mil`): ×4 macro 0.905834 against `k05-clean`'s 0.902760, +0.00307, 80%
[+0.00152, +0.00476]; public 0.944 (S54) against 0.942 (S48).

The current control, `knee-geometry-merged-2` (2026-09-24), adds the robust centre and bilateral
rule below and draws a fresh seed, 638120755: ×4 macro 0.900955, public 0.940 (S55). At that seed
the robust centre and bilateral rule measured +0.0014 (95% [−0.0022, +0.0053]) against the same
model without them (`adopted-new-seed`, 0.899547); publicly both scored 0.940 (S55 with them,
S56 without), so the drop from S54's 0.944 is the seed, not the robust centre or bilateral rule.

There is no additive 3D embedding or token graph. The best submitted bundle is distinct
from the local research incumbent; submission status lives in
[project state](../ongoing/state.md) and the [inference overview](../inference/inference-overview.md).
The [model lineage](../history/model-lineage.md) records how the recipe came about and the
retired variants. [Model shapes](model-shapes.md) traces the tensors, and
[swapping a backbone](backbone-swaps.md#current-budget) gives the parameter and compute budget.

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
| Training targets | One independent random draw per probability band at each epoch; [implemented sampler](../data-preparation/physical-sampling/implemented-sampler.md) |
| `trainer.class_balance_exponent` | `0.5` |
| `data.study_triplet_total` | 26: a study whose per-type budgets sum below 26 training triplets is scaled up to it, proportionally over the types it has |
| `data.reflection_augmentation.probability` | 0.5 |
| `model.backbone` | `rexnet_200.nav_in1k`, pretrained |
| Projection and attention widths | 256 and 128 (`PROJECTION_WIDTH`, `MIL_ATTENTION_WIDTH`, constants) |
| `model.dropout`, `model.series_dropout` | 0.0, 0.2 |
| Anatomical attention | Fixed `sign * exp(log_strength) * tanh(medial_mm / 20)` for medial/lateral meniscus and OA; two trainable strengths start at 1 |
| Meniscus side mask | Medial/Lateral Meniscus attend only to own-side tokens plus the ±10 mm band (`COMPARTMENT_MIDLINE_BAND_MM`); unknown side, invalid geometry or no own-side token keep all tokens |
| `trainer.half_knee_loss_weight` | 0.075 |
| Training seed | 2263738815, inherited by new research candidates from the incumbent |
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
3. **Augmentation:** affine, gamma/contrast and noise, drawn once per acquisition. See the
   [schedule and augmentation](schedule-and-augmentation.md) page.
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
- **Attention:** add the anatomical term to the four compartment-label scores, then run one
  float32 masked softmax per study and target over
  the study's visually valid tokens. For the two meniscus labels the mask also drops tokens on
  the sibling side beyond ±10 mm. A study without valid tokens fails before the softmax.
- **Heads:** pool the weighted features, apply twelve independent classifier vectors and biases,
  and train with the masked soft-target BCE plus the half-knee loss.
- **Half-knee loss:** one linear detector (`half_knee_detector`, shared by both menisci) scores
  the normalized tokens beyond ±10 mm; a float32 smooth-max per half is trained with soft BCE and
  the class-balance weights against the matching meniscus label, for studies with resolved side,
  valid geometry and both halves present. Weight 0.075. Prediction never uses the detector.
- **OA detector:** one linear detector shared by the three
  OA labels. Medial OA pools it with a smooth-max over the medial half and Lateral OA over the
  lateral half (both beyond ±10 mm); PF OA pools it over tokens in front of the knee centre. Each
  pooled score is added to the label's logit through its own zero-initialized scale. Tokens
  without trusted geometry belong to no region. Used at inference.

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
