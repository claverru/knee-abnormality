# Training schedule and augmentation

## Ten-epoch schedule and augmentation

Training uses
`max_epochs: 10`, one warmup epoch, and cosine decay over the remaining nine
epochs toward 1% of the peak learning rates. Warm-up rises linearly from
`optimizer.warmup_start_ratio` of the peak, 0.1 in both configs and the former hard-coded value,
to the peak at the end of `warmup_epochs`. The start ratio changes only the warm-up steps; warm-up
length also sets where the cosine decay begins. `patience: 10` prevents ordinary
validation fluctuations from stopping this ten-epoch budget early. Best and
latest checkpoints remain separate.

Per config: `optimizer.lr_decay` picks cosine (default) or linear decay to the same floor;
`optimizer.backbone_lr_stage_decay` multiplies the backbone rate by its value once per stage below
the last (1.0 by default, 0.7 for ConvNeXt-Tiny); `optimizer.late_anchor_follows_lr` scales the late
L2-SP anchor by the backbone rate over its peak (off by default). MambaVision-T uses linear decay
and the scaled late anchor over twelve epochs
([recipe](backbones/mambavision-backbone.md#recipe)). The peak backbone/head learning rates are
`2e-4` and `6e-4`, adopted in the training config (then `configs/baseline.yaml` (removed)) on 2026-09-12 from
the [overnight comparison](../validation/results/campaigns-and-queues/2026-09-12-overnight-results.md).

Augmentation runs in this order, on the GPU, during training only; validation and inference
neither apply it nor draw its random numbers. Every family is a config block with a
`probability`; at 0 it draws nothing and changes nothing, so a paired comparison can switch one
off. Unless noted, parameters are sampled per acquisition and shared by all its triplets and
slice channels.

| Order | Family (`data.*` unless noted) | Setting | What it does |
|---|---|---|---|
| 2 | Study reflection (`reflection_augmentation`) | p 0.5 | Mirrors the whole study left–right, keeping labels and geometry consistent ([below](#study-level-knee-reflection)) |
| 3 | `contrast_mixing_augmentation` | p 0.3 per plane with both contrasts, target weight 0.5–1.0 | Blends a study's fat-suppressed and non-fat-suppressed acquisitions of one plane, slice-matched along the plane normal; the target keeps its slot and geometry |
| 4 | `affine_augmentation` | p 0.7; ±10°, ±4%, scale 0.95–1.05 | Rigid affine; its inverse updates token geometry |
| 5 | `smoothing_augmentation` | p 0.2; 5×5 bilateral, spatial σ 1 px, range σ 0.05–0.1 | Edge-preserving smoothing, imitating deep-learning reconstructions |
| 6 | `resolution_augmentation` | p 0.25; keep 70–100% of frequencies | Truncates k-space along one random in-plane axis, imitating coarser phase encoding |
| 7 | `intensity_augmentation` | p 0.5; gamma 0.85–1.15, contrast 0.9–1.1 | Global gamma and contrast |
| 8 | `gin_augmentation` | p 0.3; blend up to 0.5 | Random 4-layer conv net remaps intensities, blended with the input and renormalized |
| 9 | `noise_augmentation` | p 0.15; std 0.005–0.02 | Gaussian noise; pixels differ per triplet |
| 10 | `model.feature_statistics_augmentation` | p 0.5 | DSU: resamples feature-channel means and standard deviations at the backbone's stride-8 stage (ReXNet block 5, ResNet `layer2`), per acquisition |

Filtering families (5, 6, 8) use the geometric valid region, and padding stays zero. Contrast
mixing draws from the global generator before the image families; the image families draw in
table order. The contribution of each family was measured one at a time in the
[augmentation campaign](../validation/results/campaigns-and-queues/2026-09-26-augmentation-campaign.md): gains of about
+0.001 each that did not add when stacked; the half stack (3, 6, 8 on top of 5 and 10) was adopted
on 0.943 public.

Use the [augmentation viewer](../data-preparation/preview-augmentations.md) to
inspect original/augmented example-test slices for each plane–contrast type,
with per-series parameters and seeded draws from the shared implementation.

## Weight averaging

EMA and SWA (`optimizer.weight_averaging`, 2026-09-16) were removed from the code on 2026-10-14
after no recipe used them; the implementation and its design notes are at
`7770ce9:src/knee_abnormality/training/lightning.py` and
`7770ce9:docs/modeling/schedule-and-augmentation.md#weight-averaging`. The pre-clip total gradient
norm that shipped with them is still logged as `grad_norm` from `on_before_optimizer_step`; it draws
no randomness and does not clip.

## Study-level knee reflection

Implemented on 2026-09-13. The candidate `spatial-tokens-reflection` adds
`data.reflection_augmentation.probability: 0.5` to the physical-sampling recipe.
A training forward draws one seeded device-RNG value per study, masks it by
[acquisition eligibility](../data-preparation/reflection-geometry-and-previews.md#reflection-geometry-and-eligibility),
and broadcasts it across retained acquisitions and triplets. Eligibility is
computed before series dropout. Evaluation consumes no stochastic reflection draws;
optional deterministic [reflection TTA](../inference/model-bundle-and-offline-environment.md#reflection-test-time-augmentation-tta2)
is applied separately during prediction.

Axial and coronal triplets reverse columns exactly. Sagittal triplets reverse
channels without changing the pixels within a slice, and negate the signed
acquisition-midpoint position. Triplet container order, sampling membership and
the middle source slice remain unchanged. Reflection precedes affine interpolation,
intensity/noise augmentation and channel-specific backbone normalization.
Plane/contrast embeddings and all twelve targets and observation masks retain
their identities, including medial/lateral meniscus and OA. A whole-knee reflection
preserves medial/lateral anatomical identity; it does not exchange those structures.
The original plan based this contract on the
[competition labels](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
and [AAOS anatomy](https://www.orthoinfo.org/diseases--conditions/collateral-ligament-injuries/),
consulted on 2026-09-13. The completed run selected epoch 9 at **0.890976**
validation macro AUC and **0.931 public AUC**, the best public result at that
time. The later [sigma-40 run](../validation/results/sampling/2026-09-14-reflection-sigma-40-result.md)
selected epoch 7 at **0.892360 validation AUC and 0.936 public AUC**, making
it the best submitted model at that time. Its recipe without the 3D embedding underlies the maintained
[training config](../../configs/resnet50.yaml)
([current model](current-model.md)).
The run's saved effective configuration remains the historical source of truth.
See the [comparison with 3dsampling](../validation/results/sampling/2026-09-13-reflection-validation-comparison.md)
for the +0.001012 local and +0.001 public gains, target changes and limitations.

Probability is the only training reflection parameter; zero is the control.
The added draws change subsequent stochastic choices relative to historical
runs, so equal seeds do not imply paired augmentation sequences. CPU preview
and CUDA training generators also differ. The preprocessing semantic identity
records the fixed algorithm and source hashes; saved configuration records the
probability. Historical checkpoints require their archived sources and identities.

The completed fold-0 run was launched manually with the config then named
`configs/baseline.yaml`:

```bash
uv run --locked just train configs/baseline.yaml --run-id spatial-tokens-reflection --validation.folds '[0]'
```

Keep seed 42, `all_5fold_v3`, Steven v4, ReXNet-200, 224-pixel/140-mm crops,
physical sampling, 2× validation density, optimizer and ten-epoch budget fixed.
Use the saved physical-sampling configuration and
[3dsampling result](../validation/results/sampling/2026-09-13-3dsampling-validation-comparison.md)
as the reference. Full-fold and per-target AUC, the separate gold diagnostic,
wall time, peak VRAM and checkpoint/config/source identities are recorded in the
[completed comparison](../validation/results/sampling/2026-09-13-reflection-validation-comparison.md).
The run directory is complete; another launch requires a fresh run ID. No extra
seeds, baseline reruns or automatic encoding diagnostics were launched.
See the [preview and bounded profile evidence](../data-preparation/reflection-geometry-and-previews.md#reflection-preview-evidence).
