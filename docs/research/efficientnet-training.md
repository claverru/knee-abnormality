# EfficientNet training

**Project:** Kaggle Knee Abnormality — `claverru/knee-abnormality`  
**Prepared:** 2026-09-16  
**Scope:** Pretrained `timm` 2D EfficientNets used with adjacent-slice triplets and study-level multilabel aggregation. This is not a training recipe for a randomly initialized 3D EfficientNet.  
**Objective:** Diagnose poor convergence and establish a reliable baseline before optimizing validation macro-AUC.  
**Status:** Research and source-code inspection only, **superseded on 2026-09-18 by executed runs**. No training experiments were executed for this report; the runs that tested it are in [EfficientNet backbone results](../validation/results/2026-09-18-efficientnet-backbone-results.md).

## 0. Outcome, 2026-09-18 — the premise did not hold

This report was written to diagnose **poor convergence**. Three executed runs show no convergence failure: EfficientNet-B3 and B0 train normally and reach 0.879449 and 0.880331 fold-0 macro AUC against 0.888476 for a matched ReXNet-200 control. They converge to a worse point, they do not fail to converge.

Withdraw the following, which depended on the convergence premise:

- **Section 6's proposed backbone LR comparison (2e-4 against 5e-5) and section 4's frozen-BN policy both point the wrong way.** EfficientNet **underfits** here: its training loss is worse than the control's at nearly every epoch while its generalization gap is *smaller*. Both proposals reduce backbone adaptation. Neither was run. Supporting evidence on ReXNet-200 from the same queue: `12-backbone-lr-1e-4` −0.0030, `16-backbone-eval` −0.0088 ([overnight results](../validation/results/2026-09-18-overnight-results.md)).
- **Doubling the rate to 4e-4 instead gained +0.0029, below the 0.003 single-run threshold, and not through the predicted mechanism** — training loss barely moved, so the gain is not faster fitting.
- **Sections 7's Phase 0 and Phase 1 (startup checks and the tiny-subset fitting test) were not needed.** Pretrained loading, normalization match and feature geometry were confirmed from the run artifacts, and the model fits.
- **Section 2's "immediate integration blocker" is resolved.** `_efficientnet_mapping` is implemented and non-`tf_` EfficientNets share ReXNet-200's anchors exactly. The `Conv2dSame` warning in that section was correct and is now measured: `tf_efficientnet_b0.ns_jft_in1k` walks to origin 51.0.
- **`backbone_eval` is exact, not a proxy, for these backbones.** Section 4 calls it a "broader proxy" and asks for a BN-only helper. `efficientnet_b3.ra2_in1k` contains 78 BatchNorm layers and **zero** Dropout or DropPath modules, so `.eval()` on the backbone changes BatchNorm behaviour and nothing else. The helper is unnecessary here; it would be needed for a backbone carrying drop-path.

What survives: section 3 on matching checkpoint preprocessing (followed, and the runs confirm the recorded mean/std), section 5 on preserving the pretrained stem and the feature contract, and section 8's logging requirements.

## 1. Recommended direction

Keep AdamW initially. Verify the exact checkpoint's preprocessing and the backbone integration, then compare **backbone learning rate × BatchNorm statistics policy**. Do not change optimizer, loss, normalization, augmentation, and resolution simultaneously.

EfficientNet does not require RMSprop: the official `efficientnet_b5.sw_in12k_ft_in1k` model card documents an AdamW recipe with gradient clipping, EMA, and cosine scheduling with warmup. This establishes compatibility, not an optimal MRI recipe. The Keras EfficientNet fine-tuning guide also warns against transplanting the original high-learning-rate RMSprop recipe into transfer learning. [S1] [S2]

The numerical settings below are **proposed diagnostic values**, not a literature-established EfficientNet/MRI optimum. Frozen BatchNorm statistics and a lower backbone LR are hypotheses to test, not assumed fixes.

**Local-agent prerequisite:** Read the repository's `AGENTS.md`, inspect the current branch and working tree, and reconcile this report against local changes. The observations below concern the remote files inspected on the preparation date; they do not establish the state of uncommitted local code.

## 2. Repository findings

| Area | Observed behavior | Consequence |
|---|---|---|
| Baseline | `rexnet_200.nav_in1k`, 224-pixel images, adjacent triplets, study batch size 8, BF16 mixed precision, 10 epochs. | Preserve these controls where applicable during the first backbone comparison. [R1] |
| Normalization | Mean/std are configured explicitly and applied inside `encode_batch`; they are not resolved from the backbone checkpoint. | Changing only the backbone name can retain inappropriate checkpoint normalization. [R1] [R2] |
| Backbone input | Valid triplets from the study batch are flattened and encoded together. Padded triplet entries are excluded before the backbone call. | The BN input batch is the number of encoded triplets, not the study batch size. Padding-entry contamination is not an obvious bug in this path. [R2] |
| Backbone features | `forward_features` returns spatial maps; output shape is checked and the projection width is derived from the observed channel dimension. | Preserve these checks; do not hard-code an EfficientNet feature width. [R2] |
| Optimizer | AdamW, separate backbone/head LRs, no decay for parameters with fewer than two dimensions and certain named parameters. | Standard BN affine parameters and biases already avoid weight decay. Retain this grouping initially. [R3] |
| Mode control | `backbone_eval=true` reapplies `.eval()` to the backbone after the model enters train mode. It does not set `requires_grad=False`. | This freezes BN statistics but is broader than BN-only mode control. [R2] |
| Checkpointing | Activation checkpointing is rejected unless `backbone_eval=true`. | Keep checkpointing disabled throughout the first BN comparison; do not bypass this guard. [R2] [R3] |
| Regularization | Head dropout is zero; series dropout is 0.2; image augmentations are enabled. The spatial-token identity also declares 3D-position dropout of 0.10. | A deterministic fitting test must account for metadata dropout as well as image/head regularization. [R1] [R4] |
| Metrics | Training uses masked BCE-with-logits; validation records macro-AUC, soft-target BCE, and prediction files. | Preserve the loss initially and diagnose with both fit metrics and study-level ranking metrics. [R3] |

### Immediate integration blocker: spatial feature geometry

The inspected `BACKBONE_MAPPINGS` supports only ReXNet-200 and the specified CoAtNet. An EfficientNet name raises an unsupported-mapping error. Adding a registry entry alone is insufficient: `backbone_mapping` currently dispatches between CoAtNet and ReXNet-specific traversal. [R4]

Before full-pipeline EfficientNet training, implement and test its feature-grid mapping. Verify the selected output stage, spatial stride, anchor origin, residual-path convention, and mapping through the existing physical-coordinate transformations.

**Do not copy ReXNet's origin merely because both models produce a stride-32 map.** In particular, `timm.layers.Conv2dSame` reports zero convolution padding but applies padding dynamically in its forward path. A geometry walker that only reads `layer.padding` can misinterpret such a layer. Inspect the actual instantiated backbone and handle its real padding behavior. [S10]

Test every supported input resolution and any permitted odd/even dimensions. Keep unsupported combinations explicit rather than silently assigning guessed coordinates. Preserve preprocessing/geometry identity checks when extending support.

## 3. Match preprocessing to the exact checkpoint

The following checkpoint configurations differ:

| Checkpoint | Mean | Standard deviation | Declared input / test size |
|---|---|---|---|
| `efficientnet_b3.ra2_in1k` | `[0.485, 0.456, 0.406]` | `[0.229, 0.224, 0.225]` | 288 / 320 |
| `tf_efficientnetv2_s.in21k_ft_in1k` | `[0.5, 0.5, 0.5]` | `[0.5, 0.5, 0.5]` | 300 / 384 |

Both configurations declare `fixed_input_size=false`. Their listed sizes describe checkpoint recipes, not mandatory MRI resolutions. [S3] [S4]

The existing baseline normalization matches the first row, not the second. This mismatch is a confound, not evidence that it caused previous failures. [R1]

### Implementation requirements

Resolve checkpoint metadata with `timm.data.resolve_model_data_config(backbone)`, as demonstrated in the official model usage. Store the resolved settings in the experiment configuration and carry them into inference. Do not make inference dependent on silently re-resolving mutable remote metadata. [S5]

Use the following as a **standalone inspection snippet**, not a replacement for the project's model constructor:

```python
import json

import timm

name = "efficientnet_b3.ra2_in1k"
backbone = timm.create_model(
    name,
    pretrained=True,
    in_chans=3,
    num_classes=0,
    global_pool="",
    drop_rate=0.0,
    drop_path_rate=0.0,
)
resolved = timm.data.resolve_model_data_config(backbone)
print(json.dumps({
    "model": name,
    "timm_version": timm.__version__,
    "num_features": backbone.num_features,
    "data_config": resolved,
}, indent=2))
```

Retain the MRI intensity and physical-geometry pipeline for the initial comparison. Use the checkpoint's mean/std after the existing conversion to a unit-scale image tensor; do not insert an ImageNet center crop into the geometry-aware path.

Audit the actual tensor immediately before the stem: dtype, finite values, channel order, scale, and per-channel statistics. Check for double division by 255 or double normalization. Persist the resolved mean/std because the model's current normalization buffers are non-persistent and constructed from configuration. [R2]

Do not import Keras input conventions into `timm`: the documented Keras EfficientNet example accepts `[0,255]` inputs because preprocessing is included in that implementation. [S2]

An MRI-specific normalization scheme can be tested later. Treat it as a deliberate ablation rather than an accidental checkpoint mismatch.

## 4. Make BatchNorm policy explicit

### Distinguish the operations

With running statistics enabled, BN training mode normalizes using the current minibatch; evaluation mode uses stored estimates. For `BatchNorm2d`, statistics are computed over batch and spatial dimensions separately for each channel. Setting `momentum=0` stops running-statistic updates but does **not** make training use stored statistics. [S6]

Calling `.eval()` changes module behavior; it does not disable autograd or freeze parameters. Conversely, disabling parameter gradients does not by itself put BN into evaluation mode. [S7]

For the first comparison, use these two policies:

| Policy | BN normalization during training | Running-statistic updates | BN affine parameters |
|---|---|---|---|
| Updating BN | Current minibatch | Enabled | Trainable |
| Frozen-statistics BN | Stored estimates | Disabled | Trainable |

**Frozen-statistics BN is not fully frozen BN.** Keep gamma/beta trainable in both runs and retain their existing no-decay treatment. A later experiment may freeze affine parameters too, but do not combine that change with the initial statistics comparison.

The Keras guide recommends keeping its BN layers frozen during fine-tuning. That motivates an ablation, but neither establishes the proposed affine-trainable policy nor proves that ImageNet running statistics are optimal for MRI. [S2]

### BN-only helper

This helper is intended for integration into the existing training-mode lifecycle. It deliberately fails when stored statistics are unavailable.

```python
from torch import nn

BN_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


def freeze_bn_statistics(module: nn.Module) -> None:
    """Use stored BN statistics without changing parameter trainability."""
    layers = [layer for layer in module.modules()
              if isinstance(layer, BN_TYPES)]
    for layer in layers:
        if (not layer.track_running_stats
                or layer.running_mean is None
                or layer.running_var is None):
            raise ValueError("Frozen-statistics BN requires running buffers")
    for layer in layers:
        layer.eval()
```

Reapply the policy after recursive `.train()` calls, including transitions back from validation. Integrate it with the project's existing `train()` override rather than applying it once at initialization. Add a test proving that statistics stay unchanged while backbone weights and BN affine parameters remain trainable.

The existing `backbone_eval=true` is a temporary **broader proxy**, not a weight-freezing switch. It also changes any other mode-dependent backbone behavior. Prefer BN-only control; otherwise explicitly set backbone stochastic regularization to zero and verify that the intended comparison is isolated. [R2]

Any new BN-only configuration fields must be added to the schema, checkpoint configuration, and reload path. Do not assume such fields already exist.

### Batch composition, momentum, and epsilon

In this pipeline, many triplets can originate from relatively few studies and mixed acquisition types. The hypothesis to investigate is correlated or variable BN batch composition—not simply an insufficient image count. Log encoded triplets, contributing studies, and series-type composition per forward. Gradient accumulation does not merge separate forward passes into a larger BN batch; this follows from BN's per-forward calculation. [R2] [S6]

Inspect the instantiated BN layers before overriding them. The `timm` builder documents TensorFlow retention `0.99` as PyTorch update momentum `0.01`, and defines a TensorFlow-reference BN epsilon of `1e-3`. These constants do not establish the settings of every EfficientNet variant. [S8]

Preserve BN epsilon initially. It is distinct from AdamW epsilon. Test default momentum against PyTorch `0.01` only as a secondary updating-BN experiment, and skip it when the default already matches. Do not treat `track_running_stats=false` as frozen BN: it instead uses batch statistics in evaluation too. [S6]

### BN recalibration: conditional follow-up

When a trained checkpoint fits training data but behaves poorly under stored statistics, test recalibration on a **copy** using representative training-fold inputs, no weight updates, fixed preprocessing, and disabled augmentation/series dropout. Use no validation or test examples to estimate statistics.

PyTorch documents BN-statistic updates through `update_bn`, including for averaged models. This project's dictionary batches and custom mode handling require an adapted path: verify which modules actually enter training mode rather than blindly invoking the utility. [S9] [R2]

Record the calibration subset, batching, momentum/reset policy, and updated buffers. Evaluate the resulting checkpoint in normal evaluation mode. Recalibration is an experiment, not an assumed improvement or a replacement for the four-run comparison.

## 5. Preserve the pretrained stem and feature contract

Keep three input channels for the adjacent-triplet baseline and retain the pretrained stem. Do not replace it with a randomly initialized convolution during a backbone swap.

For a separate one-channel experiment, `timm`'s input-convolution adaptation sums RGB weights; for supported other channel counts, it repeats and rescales them. The existing project already passes `in_chans` into model creation. Changing the triplet input contract would nevertheless require additional project changes, so it is outside the initial experiment. [S11] [R2]

Preserve `forward_features` and the NCHW checks. Do not interchange its output with the last element of a `features_only=True` extraction and assume identical channels: the V2-S model card illustrates different outputs for these interfaces. Derive projection dimensions from the actual selected output. [S5]

Verify that intended pretrained weights loaded successfully and record the exact architecture/checkpoint tag. Do not silently fall back to random initialization after a download or state-dictionary error.

## 6. Initial training settings

Use the following settings for the controlled fold-level experiment, after fixing integration and preprocessing:

| Setting | Proposed value |
|---|---|
| Optimizer | AdamW; preserve existing parameter grouping |
| Backbone peak LR | Compare `2e-4` against `5e-5` |
| Head/projection/aggregation peak LR | `6e-4` |
| Weight decay | `0.01` on existing decay groups; zero on existing no-decay groups |
| Betas / optimizer epsilon | `(0.9, 0.999)` / `1e-8` |
| Warmup / schedule | 1 epoch; existing step-based cosine schedule |
| Minimum LR ratio | `0.01` |
| Gradient clipping | `1.0` |
| Precision | Existing `bf16-mixed` |
| Initial resolution | 224, after geometry verification |
| Epoch budget | Same 10-epoch budget for all four initial runs |
| Backbone checkpointing | Disabled for all four runs |
| Loss / checkpoint selection | Existing masked BCE-with-logits / validation macro-AUC |

The baseline already uses these optimizer/trainer values except the proposed lower backbone LR. The schedule starts below the configured LR during warmup; log actual optimizer-step LRs, not only configured peaks. [R1] [R3]

For convergence diagnostics, explicitly disable backbone dropout/drop-path, head dropout, series dropout, image augmentation, and stochastic positional dropout. Restore useful regularization only after learning has been established. Backbone dropout/drop-path controls are not currently passed by the inspected model constructor; exposing them requires code/schema changes rather than inserting unrecognized YAML keys. [R2]

Keep BF16 initially. A short FP32 comparison is a numerical diagnostic only when non-finite activations, losses, or gradients appear. Preserve the existing FP32 attention pooling and loss computation. Do not alter optimizer epsilon merely because a checkpoint uses a different BN epsilon. [R2] [R3]

Do not raise resolution to solve flat training loss. Later resolution ablations should preserve comparable physical field of view and revalidate geometry. Distinguish additional image detail from changed anatomical coverage.

## 7. Experiment sequence

### Phase 0 — Integration and startup checks

Confirm checkpoint loading, preprocessing, feature dimensions, and geometry before launching a training sweep. Verify that all intended trainable parameters appear exactly once in optimizer groups and that the backbone/head LR assignment is correct.

Run one forward/backward step on representative real inputs. Check finite activations, logits, loss, and gradients; verify actual parameter changes after an optimizer step. Record BN modes and running-buffer behavior.

A configuration error or unsupported feature mapping is not evidence of EfficientNet non-convergence.

### Phase 1 — Tiny-subset fitting test

Use approximately **16–32 fixed training studies** with fixed triplets, labels, and observed-label masks. Disable all stochastic augmentation and regularization, including 3D-position dropout; set weight decay to zero for this fitting test only. Ensure epoch-based sampling does not change the inputs.

Repeat training on that subset. Look for a substantial improvement over a constant predictor and a continuing ability to fit the fixed targets. Avoid using subset AUC as the sole test because some labels may contain only one class.

With soft labels, BCE need not approach zero. For observed entries `O`, its mathematical lower bound is:

```text
BCE_floor = mean over (i, k) in O of
            -y[i,k] * log(y[i,k]) - (1-y[i,k]) * log(1-y[i,k])
```

Use the endpoint convention `0 * log(0) = 0`. The difference between measured BCE and this floor is a useful fit diagnostic; the floor itself does not guarantee that the architecture can attain it.

To investigate mode discrepancies, compare identical training-probe inputs on disposable checkpoint copies. Keep the whole model in evaluation mode, then selectively enable only BN training behavior for the batch-statistics probe. This leaves augmentation and other stochastic modules disabled. BN buffers may change on the probe copy, so discard it afterward. Do not report validation-batch-statistics results as ordinary deployment performance.

If this phase fails, inspect the data/gradient path, logits, label masks, and normalization before running a large sweep.

### Phase 2 — Four-run factorial comparison

Use one exact EfficientNet checkpoint for all four runs. `efficientnet_b3.ra2_in1k` is a convenient starting example because its declared normalization matches the existing baseline, not because it has been shown to be best for this task. [S3] [R1]

| Run | Backbone LR | BN statistics policy | Head LR |
|---|---:|---|---:|
| A | `2e-4` | Updating | `6e-4` |
| B | `5e-5` | Updating | `6e-4` |
| C | `2e-4` | Frozen statistics; affine trainable | `6e-4` |
| D | `5e-5` | Frozen statistics; affine trainable | `6e-4` |

Use the same fold, initialization seed, label source, study batches, triplet-sampling schedule, resolution, preprocessing, regularization, and optimizer-step budget. Restore weight decay to `0.01` for these runs. Do not reuse the fitted tiny-subset checkpoint: restart each run from the same pretrained initialization and head initialization.

Keep the validation protocol identical across runs. The current baseline uses `validation_image_multiplier: 2`; retaining that setting is acceptable, but use identical inputs for the separate train/eval-mode diagnostic. [R1]

Interpret the outcomes as follows:

| Observation | Next investigation |
|---|---|
| Lower-LR runs improve relative to corresponding higher-LR runs | Gentler fine-tuning; test an intermediate `1e-4` if useful. |
| Frozen-statistics runs improve at comparable LR | BN-statistic adaptation or batch composition. |
| A stored-statistics gap improves after training-only recalibration | Running-statistic mismatch is implicated; validate the reproducibility of that result. |
| All runs fit the tiny subset but generalize poorly | Labels, sampling, aggregation, regularization, and generalization—not a blanket claim of non-convergence. |
| Lower LR still improves late in training | A longer controlled training budget may be needed; low LR is not automatically the better final recipe. |
| Every run produces non-finite values | Numerical or implementation investigation before further hyperparameter expansion. |

Compare per-label AUC and full learning curves, not only the best single aggregate epoch. Confirm promising choices on another seed or fold before treating small differences as reliable.

### Phase 3 — Targeted follow-ups only

After the four-run comparison, change one relevant axis at a time: BN momentum, training-only recalibration, an intermediate LR, weight decay `0.001`, restoration of augmentation/series dropout, or higher resolution with matched field of view.

A brief head-only warmup is an optional experiment when unfreezing immediately degrades an otherwise useful pretrained representation. Define both gradient trainability and BN-statistics policy during that stage. The current optimizer builder skips parameters with `requires_grad=false`, so staged unfreezing must explicitly add newly trainable parameters or rebuild the optimizer/scheduler consistently. Merely changing `requires_grad` later is insufficient for those omitted parameters. [R3]

Defer alternative optimizers, AUC-specific losses, activation replacement, GroupNorm conversion, and EMA until the baseline is demonstrably learning. These are separate experiments, not prerequisites for EfficientNet convergence.

## 8. Required logging and acceptance checks

Save enough information to reproduce the comparison and distinguish a pipeline fix from a hyperparameter gain.

| Category | Required records/checks |
|---|---|
| Provenance | Repository commit and dirty state, config, dependency versions, checkpoint tag/revision, seed, fold, label manifest. |
| Inputs | Actual preprocessing, geometry identity, image size/spacing, observed-label counts, encoded triplets and contributing studies. |
| BN | Layer type, epsilon, momentum, training flag, affine trainability, running-buffer changes, and any calibration policy. |
| Optimization | Actual per-group LRs, pre-clip gradient norms, non-finite counts, optimizer steps, and whether representative backbone/head weights change. |
| Metrics | Training BCE, fixed-probe BCE, validation macro-AUC, per-label AUC, soft-target BCE, and existing gold-subset metrics when available. |
| Resources | Peak memory and train/inference throughput measured under the same settings; do not substitute model-card FLOPs for measured runtime. |

Before declaring implementation complete, test: preprocessing agreement after save/reload; supported feature shapes and coordinates; frozen BN buffers with trainable weights; updating BN buffers in the alternative mode; mode policy persistence across validation; deterministic tiny-subset inputs; no padded triplet entries entering the backbone; complete, non-duplicated optimizer parameter coverage; and rejection of unsupported geometry/checkpointing combinations.

The local agent's result should identify code/config changes, tests executed, completed experiment results, and remaining hypotheses separately. **Do not state that EfficientNet convergence is fixed until actual runs establish it.**

## 9. Sources and inspected file versions

External sources are primary library documentation, official model cards/configurations, or implementation source. They support mechanisms and checkpoint properties; they do not establish MRI-specific hyperparameter optima. Consult documentation compatible with the installed environment; this report does not require a dependency upgrade.

| Reference | Source and relevance |
|---|---|
| [S1] | Official EfficientNet-B5 model card: AdamW-based training recipe. |
| [S2] | Keras EfficientNet fine-tuning example: preprocessing convention, BN cautions, and transfer-learning optimizer cautions. |
| [S3] | Official `efficientnet_b3.ra2_in1k` configuration: normalization and resolution metadata. |
| [S4] | Official `tf_efficientnetv2_s.in21k_ft_in1k` configuration: normalization and resolution metadata. |
| [S5] | Official V2-S model usage: resolved preprocessing and feature extraction interfaces. |
| [S6] | PyTorch BatchNorm2d documentation: statistics, momentum, and evaluation behavior. |
| [S7] | PyTorch autograd documentation: evaluation mode versus gradient control. |
| [S8] | `timm` EfficientNet builder: TensorFlow-reference BN constants and momentum conversion. |
| [S9] | PyTorch optimizer documentation: BN-statistic updates with averaged models. |
| [S10] | `timm` SAME-padding convolution implementation. |
| [S11] | `timm` pretrained input-convolution adaptation implementation. |

Repository files were re-read for this report. These are **Git blob SHAs**, not commit SHAs; compare them with local `git hash-object <path>` output when useful.

| Reference | Path | Inspected blob SHA |
|---|---|---|
| [R1] | `configs/baseline.yaml` | `902291300fe53b5fd6d3b0a792461f4f7948e905` |
| [R2] | `src/knee_abnormality/model.py` | `a018f4dc2bfbba7cbc08b29c5e6fa9d606d2a55f` |
| [R3] | `src/knee_abnormality/training.py` | `71741566219579396606c8b5a32eef611549b758` |
| [R4] | `src/knee_abnormality/spatial_tokens.py` — inspected lines 1–190 | `0ff6dcb71621929656bdedd54d0773d8cf3e432c` |

[S1]: https://huggingface.co/timm/efficientnet_b5.sw_in12k_ft_in1k
[S2]: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
[S3]: https://huggingface.co/timm/efficientnet_b3.ra2_in1k/blob/main/config.json
[S4]: https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k/blob/main/config.json
[S5]: https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k
[S6]: https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html
[S7]: https://docs.pytorch.org/docs/2.14/notes/autograd.html
[S8]: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_efficientnet_builder.py
[S9]: https://docs.pytorch.org/docs/2.14/optim.html#taking-care-of-batch-normalization
[S10]: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/conv2d_same.py
[S11]: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_manipulate.py
[R1]: https://github.com/claverru/knee-abnormality/blob/main/configs/baseline.yaml
[R2]: https://github.com/claverru/knee-abnormality/blob/main/src/knee_abnormality/model.py
[R3]: https://github.com/claverru/knee-abnormality/blob/main/src/knee_abnormality/training.py
[R4]: https://github.com/claverru/knee-abnormality/blob/main/src/knee_abnormality/spatial_tokens.py
