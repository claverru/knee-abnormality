# Training schedule and augmentation

## Ten-epoch schedule and augmentation

Training uses
`max_epochs: 10`, one warmup epoch, and cosine decay over the remaining nine
epochs toward 1% of the peak learning rates. `patience: 10` prevents ordinary
validation fluctuations from stopping this ten-epoch budget early. Best and
latest checkpoints remain separate. The peak backbone/head learning rates are
`2e-4` and `6e-4`, adopted in the training config (then `configs/baseline.yaml`) on 2026-09-12 from
the [overnight comparison](../validation/results/2026-09-12-overnight-results.md).

Each acquisition independently samples affine augmentation with probability
0.7, gamma/contrast with probability 0.5 and Gaussian noise with probability
0.15. Affine strengths are up to 10 degrees rotation, 4% translation and
`[0.95, 1.05]` scale. Gamma uses `[0.85, 1.15]`, contrast uses `[0.9, 1.1]`,
and noise standard deviation uses `[0.005, 0.02]` on the `[0, 1]` scale.
Geometry and scalar strengths are shared by every triplet in an acquisition;
noise pixels differ. Zero background is restored after augmentation.

The transforms run on the GPU on each retained triplet, after study reflection and before
normalization and the backbone. Evaluation bypasses augmentation. Each family defaults to disabled when absent
from an older config, and probability zero disables it for a paired comparison.
The current combined recipe does not establish the contribution of any one
family; the on/off and strength comparisons are in the
[overnight queue](../backlog/overnight.md), and ordering and new families in the
[small changes](../backlog/small-changes.md#augmentation-families).

Use the [augmentation viewer](../data-preparation/preview-augmentations.md) to
inspect original/augmented example-test slices for each plane–contrast type,
with per-series parameters and seeded draws from the shared implementation.

## Weight averaging

Implemented on 2026-09-16, gated by `optimizer.weight_averaging` (`null` by default, one method
per run: `{method: ema, decay}` or `{method: swa, epochs}`; mutually exclusive by construction).
`KneeModule` keeps a hand-written averaged copy of the model, updated after every optimizer step,
never a second constructed model — constructing one would draw from the global CPU RNG and shift
the DataLoader shuffle. The design and its rejection of Lightning's `StochasticWeightAveraging`,
`AveragedModel` and timm's `ModelEmaV3` are recorded in the
[weight-averaging endgame entry](../backlog/endgame.md#weight-averaging); this section is the
semantics as shipped.

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

## Study-level knee reflection

Implemented on 2026-09-13. The candidate `spatial-tokens-reflection` adds
`data.reflection_augmentation.probability: 0.5` to the physical-sampling recipe.
A training forward draws one seeded device-RNG value per study, masks it by
[acquisition eligibility](../data-preparation/reflection-geometry-and-previews.md#reflection-geometry-and-eligibility),
and broadcasts it across retained acquisitions and triplets. Eligibility is
computed before series dropout. Evaluation consumes no reflection draws and
uses original inputs; reflection TTA is absent.

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
it the best submitted model. Its recipe without the 3D embedding is the maintained
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
rtk proxy uv run --locked just train spatial-tokens-reflection configs/baseline.yaml 0
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
