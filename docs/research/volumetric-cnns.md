# 3D CNNs for RSNA Knee Abnormality Detection

**Local-agent implementation and experiment brief**  
**Prepared:** 2026-09-16  
**Target repository:** `claverru/knee-abnormality`  
**Objective:** Test genuine volumetric ResNet and EfficientNet backbones against the existing slice/triplet-based solution, optimizing study-level macro ROC AUC and end-to-end inference cost.

## 1. Decision and evidence boundary

Start with **`timm_3d` ResNet18 and EfficientNet-B0 using inflated 2D pretrained weights**, one compact volume per retained acquisition, and study-level feature aggregation. Use **MONAI** for an independent 3D implementation and a MedicalNet-pretrained ResNet comparison. Both libraries implement genuine 3D models; this is not merely placing adjacent slices in a 2D network's channel dimension. [S1–S5]

This is an experiment proposal, not a reconstruction of another competitor's solution. We have not verified that Scott Willis uses volumetric CNNs, which library he uses, or his initialization/preprocessing. References to “3D,” ResNet, EfficientNet, or leaderboard performance do not establish those details. The previously discussed Medium identity match was unverified and should not motivate implementation decisions.

**Status:** Repository documentation/configuration and upstream source were inspected. No repository files were changed, models trained, competition scores reproduced, or GPU throughput measurements obtained. Code below is an integration template; the actual third-party constructors must be smoke-tested in the project's locked environment.

## 2. Existing project: preserve the boundaries

The inspected repository uses Lightning, Pydantic-validated YAML, shared code under `src/knee_abnormality/`, maintained train/predict/export/submit commands, and offline inference bundles. Its documented model consumes adjacent-slice triplets, extracts 2D spatial features, and applies target-specific study MIL. It also uses physical-coordinate information. **A 3D positional embedding is not a 3D convolutional encoder.** The output contract is twelve study logits with target-observation masks. [R1, R2]

The inspected `configs/baseline.yaml` selects `steven_v4`, `all_5fold_v3`, fold 0, `rexnet_200.nav_in1k`, three-channel adjacent triplets, 224-pixel inputs at 0.625 mm, and quantiles `[0.005, 0.995]`. These describe that file at inspection time, not every historical checkpoint. In particular, do not copy an older conversation's fold identifier into a new experiment. [R3]

Before editing, inspect the local checkout, applicable agent instructions, configuration schema, dataset/collation contract, model, preprocessing identity, and export dependency collection. The local branch may have advanced beyond the inspected files.

Keep labels, target ordering, fold membership, missing-label handling, evaluation population, and study-level supervision fixed. Add a distinct volume input/model mode; do not reinterpret historical `adjacent_triplets` checkpoints. Extend the existing workflow rather than creating a parallel training or submission pipeline. Use the frozen control checkpoint's own config and runtime, not today's mutable baseline YAML. [R1, R2]

## 3. Verified library choices

| Library | Relevant models | Initialization | Proposed role |
|---|---|---|---|
| `ZFTurbo/timm_3d` / package `timm-3d` | ResNet family; EfficientNet V1/V2 | Converts supported 2D checkpoints into 3D weights during loading | First PyTorch experiment |
| MONAI / package `monai` | 3D ResNets; EfficientNet B0–B8/L2 | Compatible ResNets can load MedicalNet weights; built-in EfficientNet ImageNet loading is 2D-only | Medical-pretraining and implementation controls |
| `ZFTurbo/classification_models_3D` | ResNet18/34/50/101/152; EfficientNet B0–B7 and V2 variants | Converted 2D ImageNet weights | Keras 3 alternative; not necessary for this pipeline |
| `ZFTurbo/segmentation_models_pytorch_3d` | ResNet/EfficientNet encoders | Supports encoder pretrained-weight configurations | Secondary encoder source; avoid a segmentation decoder for classification |

Capabilities above are documented by their maintainers. Do not equate registry support with verified compatibility in our environment. [S1–S7]

### `timm_3d` model identifiers

The inspected model list includes these exact identifiers: [S2]

| Priority | Identifier | Purpose |
|---|---|---|
| First | `resnet18.a1_in1k` | Small ResNet baseline |
| First | `tf_efficientnet_b0.in1k` | EfficientNet baseline |
| Later | `resnet10t.c3_in1k` | Smaller ResNet candidate |
| Later | `tf_efficientnetv2_b0.in1k` | EfficientNetV2 candidate |
| Later | `efficientnetv2_rw_t.ra2_in1k` | Additional compact V2 candidate |

Query the installed registry before construction. Pin the package version or commit and dependencies that pass the smoke tests; do not assume upstream `timm` and this separate project have identical registries or APIs. Package introduction must go through the project's dependency/lockfile workflow and offline wheel bundling. [S1, R1]

## 4. Initialization: distinguish three different experiments

### Inflated 2D weights

The inspected `timm_3d` loader adapts supported input convolutions when channel counts differ, removes an incompatible classifier, and inflates eligible 4D convolution weights into 5D weights. Its conversion repeats a kernel along the new **last spatial axis** and divides by the repetition count. For a square spatial kernel and added extent `kZ`: [S3]

```text
W2: [out_channels, in_channels/groups, kY, kX]
W3: [out_channels, in_channels/groups, kY, kX, kZ]
W3[..., z] = W2 / kZ
```

The new axis initially averages contributions across depth. This transfers 2D features; it is **not pretraining on 3D MRI**. It also does not guarantee exact whole-network equivalence to the 2D model because padding, downsampling and normalization affect the computation.

Record checkpoint identity/hash, channel adaptation, inflation axis, and missing/unexpected keys. Permit only understood differences, such as the removed classifier; do not accept a silently random stem/backbone. Native 3D weights loaded at inference must not be inflated a second time.

### MONAI EfficientNet: a silent initialization trap

In the inspected `EfficientNetBN` source, weight loading occurs only when `pretrained` is true **and** `spatial_dims == 2`. Therefore `spatial_dims=3, pretrained=True` does not provide pretrained 3D EfficientNet weights. Set `pretrained=False` explicitly for the scratch control. Custom inflation would be a separate implementation requiring its own tests. [S4]

### MONAI MedicalNet ResNet

The inspected ResNet loader supports MedicalNet pretraining with compatible 3D, single-channel, headless configurations. Its direct constructors require the correct `shortcut_type` and `bias_downsample`, as well as `feed_forward=False`; the source provides `get_medicalnet_pretrained_resnet_args(depth)`. Load the pretrained encoder first, then attach the study head. [S5]

Compare MedicalNet initialization against **the same MONAI architecture initialized randomly**, with identical stride settings. Comparing MONAI MedicalNet directly with a different `timm_3d` ResNet conflates initialization and architecture. MedicalNet pretraining is not evidence of knee-specific pretraining or superiority on this task.

## 5. Tensor and encoder contract

Use an explicit internal convention:

```text
Packed valid volumes: [N_valid, C, Z, Y, X]
Volume features:      [N_valid, F]
Study features:       [B, V_max, F] + volume_valid[B, V_max]
Study logits:         [B, 12]
```

`Z` means ordered through-plane position for the individual acquisition, not necessarily patient superior–inferior. Start with one intensity channel. Acquisitions in different planes/contrasts are separate volumes, not adjacent depth segments or automatically registered channels.

PyTorch documents Conv3d axes as `[N, C, D, H, W]`; anatomical meaning is assigned by the caller. `timm_3d` documents `[B, C, H, W, D]` and inflates the last axis. Preserve the pretrained in-plane interpretation with an explicit adapter: [S1, S3, S8]

```python
# Internal [N, C, Z, Y, X] -> timm_3d [N, C, Y, X, Z]
x_library = x_internal.permute(0, 1, 3, 4, 2).contiguous()
```

This is a semantic permutation, not `channels_last_3d` memory format. Test unequal dimensions and an axis-coded phantom; cubic random tensors cannot reveal an axis swap. Include the adapter in saved/exported model code.

### Minimal `timm_3d` feature encoder

Use feature embeddings for study aggregation, not independently supervised series classifiers.

```python
from __future__ import annotations

import torch
from torch import nn


class Timm3DEncoder(nn.Module):
    """Accept [N,C,Z,Y,X]; return one feature vector per valid volume."""

    def __init__(
        self,
        model_name: str,
        *,
        pretrained: bool,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        if in_channels not in (1, 3):
            raise ValueError("This experiment supports one or three channels.")
        import timm_3d

        self.in_channels = in_channels
        self.backbone = timm_3d.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            global_pool="avg",
        )
        self.feature_dim = int(self.backbone.num_features)
        if self.feature_dim < 1:
            raise ValueError("Backbone reported an invalid feature dimension.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.in_channels:
            raise ValueError("Expected [N,C,Z,Y,X] with configured channels.")
        if x.shape[0] < 1 or min(x.shape[2:]) < 1:
            raise ValueError("Pack nonempty, valid volumes before encoding.")
        features = self.backbone(x.permute(0, 1, 3, 4, 2).contiguous())
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise RuntimeError(f"Unexpected feature shape: {tuple(features.shape)}")
        return features


# Examples; initialization may download weights during training setup.
# encoder = Timm3DEncoder("resnet18.a1_in1k", pretrained=True)
# encoder = Timm3DEncoder("tf_efficientnet_b0.in1k", pretrained=True)
# projection = nn.Linear(encoder.feature_dim, 256)
```

Probe the actual output feature width before building the projection. Do not hard-code dimensions copied from a 2D model or another backbone. A removed classifier plus global pooling must produce `[N,F]`, not a feature map.

### Minimal MONAI construction checks

These are scratch **single-volume classifiers for API/shape checks**, not the complete multi-acquisition study model. [S4, S5]

```python
from torch import nn
from monai.networks.nets import EfficientNetBN, resnet18


def build_monai_scratch_models(num_labels: int = 12) -> dict[str, nn.Module]:
    if num_labels < 1:
        raise ValueError("num_labels must be positive.")
    return {
        "resnet18": resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=num_labels,
            pretrained=False,
        ),
        "efficientnet_b0": EfficientNetBN(
            "efficientnet-b0",
            spatial_dims=3,
            in_channels=1,
            num_classes=num_labels,
            pretrained=False,
        ),
    }
```

## 6. Volume construction and geometry

**First implementation: encode each acquisition in its native plane and fuse features afterward.** Do not initially reconstruct a common isotropic study cube or register all sequences. Reuse the project's native-pose parsing, physical ordering, decoding, intensity handling, and crop geometry, but add explicit volume eligibility and sampling semantics. [R2]

Construct a volume from physically ordered unique slices, not the current overlapping triplets reshaped into depth. Check orientation consistency, duplicate positions, gaps, in-plane geometry, decode failures, and minimum usable coverage. When metadata supports it, ordering follows position projected onto the slice normal. An instance-number fallback may yield an ordered stack, but must not be represented as verified physical geometry.

Choose one recorded depth policy: for the initial prototype, interpolate onto a regular grid spanning the accepted acquisition coverage. Record that coverage and effective spacing. Fixed `Z` alone does not imply a fixed physical field of view. Do not interpolate across a large missing slab without an explicit rejection/fallback rule. Establish gap tolerances from the data audit rather than inventing a universal threshold.

Maintain the selected control's in-plane physical field of view when changing pixel resolution. For example, the inspected 224 × 0.625 mm configuration nominally covers 140 mm; retaining that width at 160 pixels implies 0.875 mm/pixel. Verify this against the actual crop convention. Otherwise a resolution experiment also changes anatomical coverage. [R3; arithmetic]

A proposed initial grid is **`Z=32, Y=X=160`**, not a measured optimum. Later test depth and in-plane resolution separately. Preserve anisotropy when appropriate; interpolating thick slices does not create new anatomical detail.

**Normalization must be explicit.** Retain acquisition-consistent clipping as the first controlled choice, guard constant-valued inputs, and store the complete recipe. For a one-channel input, select and validate scalar normalization; do not broadcast the baseline's three RGB mean/std values onto it. Alternatively, repeat one volume into three channels and apply the checkpoint's original per-channel normalization as an initialization-control experiment. Summed stem weights with one scalar normalization are not generally equivalent to three differently normalized RGB channels.

Separate voxel support, volume validity, study membership, and label-observation masks. Do not encode all-zero placeholders for absent acquisitions: pack valid volumes before the CNN. Define and test the all-volumes-missing fallback. If a local stack is geometrically valid but the cross-series study reference is unavailable, decide explicitly whether native-volume encoding remains usable; do not conflate local stack validity with cross-series registration. Disable any unsupported cross-series coordinates rather than inventing them.

## 7. Study aggregation and augmentation

The initial model should be:

```text
DICOM acquisition -> normalized 3D volume -> shared 3D encoder
                 -> global pooled feature -> learned projection
                 -> plane/contrast metadata + valid-volume mask
                 -> target-specific study attention -> 12 logits
```

Reuse the target-specific MIL concept, but adapt its input contract deliberately. Existing spatial-token code may depend on 2D feature anchors and patient-coordinate maps; do not feed volume vectors through that path unchanged. First fuse one vector per volume without invented voxel coordinates. Spatial 3D feature tokens can be a later, separately benchmarked extension.

Apply supervision after study aggregation. Do not copy every study-positive label onto every series, and do not treat unavailable targets as negatives. Keep the existing masked loss reduction for the initial comparison. Ensure missing/NaN targets are sanitized before any loss calculation that could propagate NaNs.

Apply the same geometric transform across all slices of a volume. Preserve study-consistent reflection semantics where supported, updating metadata/coordinates together. Independent slice rotations or inconsistent contrast transformations would introduce artificial depth discontinuities. Keep early augmentation mild; disable unported spatial augmentations explicitly instead of silently applying their 2D implementations. Guard series dropout so it cannot remove all available evidence.

## 8. Depth downsampling and optimization

Inspect and log every stage's spatial shape at the actual non-cubic input. With five factor-two depth reductions, a 32-slice input can reach one depth cell. This is not automatically wrong, but it makes the downsampling schedule an important ablation.

First run stock architectures. Then consider preserving depth in the stem/early stages. For internal `[Z,Y,X]`, an in-plane-only stride is `(1,2,2)`; inside the described `timm_3d` adapter it corresponds to `(2,2,1)`. Any patch must keep residual/downsample branches consistent, account for padding, and enter the checkpoint identity. Changing stride preserves weight shapes but changes computation; changing kernel extent may also require weight conversion. [S3, S8]

**Proposed pilot settings, not verified optima:** use the project's optimizer/scheduler structure, a pretrained-backbone LR of `1e-4`, head LR of `3e-4`, weight decay `0.01`, one warmup epoch, gradient clipping `1.0`, and supported mixed precision. Use the same settings for the first ResNet18/EfficientNet-B0 pair. These are deliberately distinguished from the inspected baseline's `2e-4`/`6e-4` learning rates. [R3 for existing values]

Start with the largest feasible study microbatch, then accumulate gradients to a recorded effective study batch. Log valid volumes per forward as well as studies per optimizer step. BatchNorm computes statistics from the current forward batch and spatial positions; gradient accumulation does not combine those statistics across forwards. Compare updating versus frozen pretrained BN statistics if the effective volume microbatch is small or unstable. Do not freeze random initial BN statistics as the default scratch recipe. [S9]

Before interpreting poor convergence, check loaded weights, scalar/RGB normalization, axis order, output dimensions, loss masks, gradients, BN behavior, and final depth resolution. Run a tiny-subset overfit test. Only then sweep backbone LR, for example `5e-5`, `1e-4`, `2e-4`, with a recorded head ratio. A scratch run may need a different LR and longer training; a short pretrained-budget comparison does not establish its attainable accuracy.

## 9. Experiment sequence

All initial training comparisons should use the same saved fold, labels, study population and metric implementation. Keep within-pair volume coverage, normalization, head, and optimization identical unless they are the tested factor.

| Run | Experiment | Question |
|---|---|---|
| C0 | Frozen existing 2D/2.5D control, unchanged | Reference AUC and deployment cost |
| V1 | `timm_3d` ResNet18, inflated weights, 32×160×160 | Does a small volumetric encoder merit further work? |
| V2 | `timm_3d` EfficientNet-B0, same recipe | Backbone-family comparison |
| V3 | Better V1/V2 architecture, random initialization | Value of inflation in a matched implementation |
| M0/M1 | Identical MONAI ResNet18, random vs MedicalNet weights | Value of medical pretraining |
| V4 | Better volume model, alter depth only | Coverage/detail versus latency |
| V5 | Better volume model, alter in-plane resolution only | Spatial detail versus latency |
| V6 | Better volume model, preserve early depth | Value of a different downsampling schedule |
| Later | ResNet10t or compact EfficientNetV2 | Additional accuracy/runtime trade-offs |

Treat this as a gated sequence, not permission to launch every run immediately. Require dependency/shape tests and an inference profile before training. Run V1/V2 first, then expand promising directions. Confirm improvements on another fold before committing to a full multi-fold run.

C0 versus V1 is a **pipeline comparison**, not a pure convolution-dimensionality ablation, because sampling and pooling also change. For a stronger attribution test, add a 2D encoder control consuming the same volume slices, in-plane resolution and physical coverage, with an analogous study head. Report the remaining architectural differences.

Do not change labels, add AUC-specific losses, introduce ensembles/TTA, or expand series counts simultaneously with the first 3D comparison.

## 10. Efficiency measurements and deployment

Do not assume “3D” or “EfficientNet” implies faster execution. For a dense convolution, multiply-accumulate count scales approximately as:

```text
output_Z * output_Y * output_X * Cin * Cout * kZ * kY * kX
```

For equal output locations and channels, adding a depth kernel multiplies that layer's cost by `kZ`. Whole-model comparisons also depend on resolution, depth strides, channels, grouped convolutions and the number of encoded inputs. This is operation-count reasoning, not a wall-clock prediction. [S8]

Measure both warmed model execution and the full submission-like pipeline: indexing, DICOM reads/decoding, resampling, host-to-device transfer, encoder, study aggregation, startup/model loading and output writing. Synchronize accelerator work around timed regions. Use the target GPU/runtime and identical study samples, not parameter counts or timings from unrelated hardware.

For every run record macro/per-label AUC, exact evaluation membership, total wall time, seconds/study, latency distribution, peak GPU memory, CPU RAM, volumes/study, decoded slices/study, volume shape/physical coverage, precision, batch/chunk size, dependency versions, checkpoint hash and preprocessing identity. Report undefined per-label AUC explicitly; do not silently change the macro denominator. Keep official/gold-label diagnostics separate from report-derived-label metrics. [R1, R2]

First benchmark eager execution without TTA. Test compilation, memory format or other optimizations individually after correctness. Any speedup must include its setup cost in a deployment-budget calculation. Activation checkpointing belongs to a training-memory experiment, not an assumption of faster inference.

At inference, construct the saved architecture with **external pretraining downloads disabled**, then load the trained native 3D state dict from the bundle. Include the chosen library, dependencies, custom axis/stride code and preprocessing in the offline bundle. Verify inference in an Internet-disabled environment with empty external model caches. Preserve the existing export/infer/submit ownership and require output verification before submission. [R1]

Select models on the measured AUC/runtime Pareto frontier. No specific throughput, leaderboard gain, or competition efficiency score is claimed here.

## 11. Required local-agent deliverables and acceptance checks

Deliver the smallest coherent implementation: a validated volume config/input mode; shared volume preparation/collation; an encoder adapter with derived feature width; study-level fusion; updated checkpoint/preprocessing identity and export dependencies; focused tests; and a pilot-results record. Follow the repository's existing script discovery/catalog rules rather than introducing duplicate entry points. [R2]

Before any expensive run, require:

- **Data correctness:** physical ordering, duplicate/gap handling, deterministic evaluation, shared volume augmentation, invalid/missing acquisition handling, and unchanged label/fold identities.
- **Model correctness:** unequal-axis and phantom tests, valid packing/scattering, actual feature-width assertion, finite `[B,12]` logits, forward/backward pass, correct observation masks, and a tiny-subset overfit test.
- **Initialization correctness:** named checkpoint and load audit, no accidental scratch fallback, correct inflation axis, and explicit MONAI initialization semantics.
- **Reproducibility:** checkpoint reload produces equivalent predictions; old checkpoints retain their original semantics; local and bundled inference agree within a documented numeric tolerance.
- **Deployment viability:** representative end-to-end profile and successful offline bundle execution before any Kaggle submission.

Return measured findings, failures and the next justified experiment. Keep the current control usable and do not commit generated data, weights or caches.

## 12. Sources and inspection provenance

Primary upstream sources were checked on 2026-09-16. Branch URLs can change; record the exact installed versions/commits during implementation. The MONAI source inspected was its default `dev` branch, so verify availability and behavior in the project's selected release rather than assuming the development branch matches it.

### Libraries and framework behavior

- **[S1]** `timm_3d` README: installation, input convention, model families and on-load conversion. <https://github.com/ZFTurbo/timm_3d/blob/main/README.md>
- **[S2]** `timm_3d` supported model identifiers. <https://github.com/ZFTurbo/timm_3d/blob/main/docs/models_list.md>
- **[S3]** `timm_3d` pretrained loader and `convert_2d_weights_to_3d`. <https://github.com/ZFTurbo/timm_3d/blob/main/timm_3d/models/_builder.py>
- **[S4]** MONAI EfficientNet source: constructors and the `spatial_dims == 2` loading gate. <https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/efficientnet.py>
- **[S5]** MONAI ResNet source: MedicalNet loading constraints and feature interfaces. <https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/resnet.py>
- **[S6]** Keras 3 volumetric classification models. <https://github.com/ZFTurbo/classification_models_3D>
- **[S7]** PyTorch volumetric segmentation models/encoders. <https://github.com/ZFTurbo/segmentation_models_pytorch_3d>
- **[S8]** PyTorch Conv3d: axis convention, kernel/stride semantics and groups. <https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv3d.html>
- **[S9]** PyTorch BatchNorm3d: forward-batch statistics and running estimates. <https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm3d.html>

### Project files inspected through the GitHub connection

- **[R1]** Project README. <https://github.com/claverru/knee-abnormality/blob/main/README.md>  
  Inspected file blob SHA: `c4d5f787d6489c690dcad34c56459ba13d985797`.
- **[R2]** Project architecture, lines 1–240. <https://github.com/claverru/knee-abnormality/blob/main/docs/architecture.md>  
  Inspected file blob SHA: `b5f8c5b287121812d3dd73396df1dd18af609228`.
- **[R3]** Selected baseline configuration. <https://github.com/claverru/knee-abnormality/blob/main/configs/baseline.yaml>  
  Inspected file blob SHA: `902291300fe53b5fd6d3b0a792461f4f7948e905`.

Upstream file blob SHAs inspected: `timm_3d` README `be42d8a831c8194e8035fe10732f7b29734b6544`; loader `f3b1904c59fdeb322bd9c61ba8950a4b2abf2ae1`; MONAI EfficientNet `e8b510e47d49ee705e2592f83d62b757cb77f42f`; MONAI ResNet `9142116ae4d3e52daf3fb213a7e0a914fa7e32d7`. These identify file contents, not installed package versions or repository commits.
