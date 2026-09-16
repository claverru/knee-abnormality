# Current model

The code carries one model: the full-slice spatial-token recipe of the overnight `00-baseline`
run, defined by [`configs/training.yaml`](../../configs/training.yaml).

- **Inputs:** ReXNet-200 reads every sampled triplet at 224 px, 0.625 mm spacing and a 140 mm
  field.
- **Tokens:** every visually valid backbone location is a token with four additive embeddings.
- **Pooling:** target-specific attention MIL pools the tokens.

There is no additive 3D embedding and no token graph. The best submitted model is still the
reflection/sigma-40 champion, which is this recipe plus the
[3D embedding](../history/model-lineage.md#five-embedding-predecessor); submission status lives
in [ongoing work](../ongoing.md) and the [inference overview](../inference/inference-overview.md).
The [model lineage](../history/model-lineage.md) records how the recipe came about and the
retired variants. [Model shapes](model-shapes.md) traces the tensors, and
[swapping a backbone](backbone-swaps.md#current-budget) gives the parameter and compute budget.

## Recipe

`configs/training.yaml` equals `00-baseline`'s saved `effective_config.json` except for its name,
its paths and the fields that have since become code constants or were removed with the token
graph (`model.graph_layers`, `model.graph_geometry_dropout`). The settings that define the model
are:

| Setting | Value |
|---|---|
| Inputs | Three adjacent slices per input (`SLICES_PER_INPUT`, a constant) |
| `data.image_size`, `data.spacing_mm` | 224, 0.625 (140 mm) |
| `data.sampling_sigma_mm`, `data.validation_image_multiplier` | 40.0, 2 |
| `data.reflection_augmentation.probability` | 0.5 |
| `model.backbone` | `rexnet_200.nav_in1k`, pretrained |
| Projection and attention widths | 256 and 128 (`PROJECTION_WIDTH`, `MIL_ATTENTION_WIDTH`, constants) |
| `model.dropout`, `model.series_dropout` | 0.0, 0.2 |
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
- **Attention:** score every token and run one float32 masked softmax per study and target over
  the study's visually valid tokens. A study without valid tokens fails before the softmax.
- **Heads:** pool the weighted features, apply twelve independent classifier vectors and biases,
  and train with the masked soft-target BCE alone.

The plane and contrast lookup buffers derive from the configured metadata, not from slot order.
The ReXNet adapter derives stride 32 and origin 0 from its convolutions. The
[CoAtNet adapter](coatnet-224-spatial-backbone.md) derives stride 32 and origin 12, and any
other backbone mapping fails explicitly. Grid dimensions may vary independently.

Tokens carry no patient-coordinate geometry. The model reads neither LPS points nor the study
center, and visual validity alone decides token membership.

- **Geometry output:** `feature_geometry(valid_rects, inverse_affine, grid_shape, input_shape,
  mapping, source_channels)` returns only `(xy, visual_valid)`.
- **Batches:** they carry no `patient_maps` or `geometry_valid`.
- **Prepared series:** they keep both for reflection eligibility and for the preview and
  inspection tools.

The model lives in `src/knee_abnormality/modeling/` (`model.py`, `spatial_tokens.py`,
`augmentation.py`). The spatial-token identity is
`all_valid_spatial_tokens_four_embeddings_v6`, inside preprocessing identity
`physical_gaussian_overlap_spatial_tokens_reflection_v9`
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

Train the recipe on fold 0 in a fresh run directory:

```bash
rtk proxy uv run --locked just train RUN_ID
rtk proxy uv run --locked python -m scripts.pipeline.train configs/training.yaml --run-id RUN_ID
```

The run writes its effective configuration, provenance, source snapshot, checkpoints, epoch
metrics and `fold_0/validation_predictions.csv`. Recompute a new run's validation predictions
with `just validate RUN_DIR/effective_config.json CHECKPOINT OUTPUT_CSV`, and compare two runs
with the [fold-run comparison](../validation/fold-run-comparison.md).

The `00-baseline` checkpoint itself records identity `…token_graph_v3`, so it no longer loads
through the current code. Use its `source_snapshot.zip` or the bundle
`outputs/submission/exports/knee-overnight-00-baseline-20260915`; historical checkpoints are
never relabelled in place.
