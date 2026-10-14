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
([recipe](mambavision-backbone.md#recipe)). The peak backbone/head learning rates are
`2e-4` and `6e-4`, adopted in the training config (then `configs/baseline.yaml`) on 2026-09-12 from
the [overnight comparison](../validation/results/2026-09-12-overnight-results.md).

Augmentation runs in this order, on the GPU, during training only; validation and inference
neither apply it nor draw its random numbers. Every family is a config block with a
`probability`; at 0 it draws nothing and changes nothing, so a paired comparison can switch one
off. Unless noted, parameters are sampled per acquisition and shared by all its triplets and
slice channels.

| Order | Family (`data.*` unless noted) | Setting | What it does |
|---|---|---|---|
| 1 | `model.series_dropout` | 0.2 | Drops series-type slots per study, keeping at least one |
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
[augmentation campaign](../validation/results/2026-09-26-augmentation-campaign.md): gains of about
+0.001 each that did not add when stacked; the half stack (3, 6, 8 on top of 5 and 10) was adopted
on 0.943 public.

Use the [augmentation viewer](../data-preparation/preview-augmentations.md) to
inspect original/augmented example-test slices for each plane–contrast type,
with per-series parameters and seeded draws from the shared implementation.

## Weight averaging

Implemented on 2026-09-16, gated by `optimizer.weight_averaging` (`null` by default, one method
per run: `{method: ema, decay}` or `{method: swa, epochs}`; mutually exclusive by construction).
`KneeModule` keeps a hand-written averaged copy of the model, updated after every optimizer step,
never a second constructed model — constructing one would draw from the global CPU RNG and shift
the DataLoader shuffle. `EMAConfig`, `SWAConfig` and `WeightAveragingConfig` in
`src/knee_abnormality/config.py` define the settings; the experiment validator rejects an SWA
window longer than `max_epochs - warmup_epochs`. This section owns the shipped semantics and
design rationale; pending comparisons live in the
[weight-averaging endgame entry](../backlog/endgame.md#weight-averaging).

- **EMA:** the averaged copy tracks the live model exactly (an exact copy, not a decay-1 lerp)
  until the warm-up steps end, then every step lerps it toward the live weights with weight
  `1 - decay`. Floating parameters and BatchNorm running statistics are averaged; integer buffers
  (`num_batches_tracked`, the `plane_lookup`/`contrast_lookup` token tables) are always copied
  exactly. The update runs in float32 outside autocast.
- **SWA:** the schedule is unchanged — no `SWALR`, no epoch-interval scheduler swap, the tail
  learning rate is still set through `minimum_lr_ratio`. The averaged copy is a uniform
  (Polyak) average over every optimizer step of the last `epochs` epochs, final step included; at
  the end of each of those epochs, before that epoch's validation, BatchNorm statistics of the
  averaged model are refreshed with a forward-only pass over an unshuffled training loader, under
  the same bf16 autocast as training, inside `torch.random.fork_rng` (CPU and CUDA) so the training
  random streams are untouched. Only BatchNorm layers that are actually in training mode are
  refreshed, so `model.backbone_eval: true` keeps its pretrained statistics untouched and the
  refresh is skipped entirely.
- **Validation and checkpoints:** every epoch validates both the saved (averaged, when the gate is
  on) and the live weights on the same batches; `val_macro_auc` and checkpoint
  selection/early-stopping always score the saved weights, and live results are logged separately
  (`val_live_*`, `epochs/live_*`). Checkpoints store the averaged weights as the ordinary `model.*`
  state and the live weights under `live_state_dict`; prediction, export and `compare_fold_runs`
  need no change. See
  [configs and experiment artifacts](../architecture/configs-and-experiment-artifacts.md#experiment-artifacts-and-reproducibility)
  for the full artifact list.
- **Gradient-norm logging** ships alongside weight averaging, independent of the gate: the
  pre-clip total gradient norm is always logged as `grad_norm` from `on_before_optimizer_step`,
  where Lightning 2.x still sees unclipped gradients. It is a diagnostic with no option; it draws
  no randomness and does not clip, so numerics are unchanged. Whether
  `trainer.gradient_clip_val: 1.0` ever binds is read from the logged values.

### Design rationale and original budget

The 2026-09-16 implementation uses `deepcopy` and a stateless per-step interpolation weight.
The alternatives were rejected for specific integration reasons: `AveragedModel` passed integer
buffers through floating-point arithmetic; timm `ModelEmaV3` did not provide uniform SWA weights;
and a callback could not swap weights before the strict checkpoint load. Lightning's
`StochasticWeightAveraging` also changed the schedule to epoch-level `SWALR`, flattened the
3:1 backbone/head LR ratio, omitted the final epoch, added a pseudo-epoch eligible for selection,
kept the average outside the saved `model.*` weights and reset BatchNorm statistics that
`backbone_eval` would not re-estimate. These are the recorded implementation decisions, not an
assessment of later library versions.

For the original ten-epoch, 4,380-step run, warm-up ended at step 438. EMA decay 0.999 gave a
1,000-step (2.3-epoch) horizon matching the flat cosine tail: `00-baseline` live AUC was
0.89269, 0.89269, 0.89205 and 0.89276 at epochs 7–10. Decay 0.99 would track roughly 100 steps;
0.9995 would retain substantial weight from epochs 5–7. SWA's three-epoch window started at
step 3,066. Where the averaged copy equals the live model (EMA epoch 1, SWA epochs 1–7), their
metrics must match. Only the averaged score selects checkpoints, preserving ten candidates;
export drops the auxiliary `live_state_dict`.

Pre-run estimates were 58 MB extra GPU memory (0.3% of a 20.0 GiB peak), checkpoints growing
from 175 to about 233 MB, a few seconds for updates, about three minutes for duplicate validation,
and 1–1.5 minutes per SWA BatchNorm refresh. The completed
[comparison](../validation/results/2026-09-17-overnight-results.md#weight-averaging) measured
153 s extra for EMA and 369 s for SWA; the live-only-at-last-epoch optimization remains in
endgame rather than being treated as shipped.

The [optimizer research](../research/optimizer-research.md#4-ema-first-additional-training-feature)
records background. A public B0 ablation reported 0.7914 to 0.7940 with EMA, considered noise by
its author ([public findings](../research/public-research.md#geometry-sampling-encoders-and-losses));
a [public notebook](../research/public-research.md#public-code-worth-inspecting) averages top-epoch
weights. The historical baseline1 epoch-7–9 probability average gained 0.004267 on twelve reused
validation studies, which is not an independent gain
([review](../history/baseline1-review.md#follow-up-experiments)).

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
time. The later [sigma-40 run](../validation/results/2026-09-14-reflection-sigma-40-result.md)
selected epoch 7 at **0.892360 validation AUC and 0.936 public AUC**, making
it the best submitted model at that time. Its recipe without the 3D embedding underlies the maintained
[training config](../../configs/training.yaml)
([current model](current-model.md)).
The run's saved effective configuration remains the historical source of truth.
See the [comparison with 3dsampling](../validation/results/2026-09-13-reflection-validation-comparison.md)
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
rtk proxy uv run --locked just train configs/baseline.yaml --run-id spatial-tokens-reflection --validation.folds '[0]'
```

Keep seed 42, `all_5fold_v3`, Steven v4, ReXNet-200, 224-pixel/140-mm crops,
physical sampling, 2× validation density, optimizer and ten-epoch budget fixed.
Use the saved physical-sampling configuration and
[3dsampling result](../validation/results/2026-09-13-3dsampling-validation-comparison.md)
as the reference. Full-fold and per-target AUC, the separate gold diagnostic,
wall time, peak VRAM and checkpoint/config/source identities are recorded in the
[completed comparison](../validation/results/2026-09-13-reflection-validation-comparison.md).
The run directory is complete; another launch requires a fresh run ID. No extra
seeds, baseline reruns or automatic encoding diagnostics were launched.
See the [preview and bounded profile evidence](../data-preparation/reflection-geometry-and-previews.md#reflection-preview-evidence).
