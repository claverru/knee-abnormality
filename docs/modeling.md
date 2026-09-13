# Modeling

Evidence: current repository artifacts through 2026-09-11 and unverified
working notes; see [provenance](INDEX.md#provenance).

The most concrete baseline lead is a fast ReXNet backbone applied to
three-slice windows, with multiple-instance learning (MIL) at study level.
The full architecture is unavailable.

The separate [public-source investigation](public-research.md#geometry-sampling-encoders-and-losses)
adds controlled and anecdotal positive/negative results, including small-versus-
large encoders, slice geometry, losses and medical pretraining. Its
[notebook comparison](public-research.md#public-code-worth-inspecting) identifies
inspectable DINO and CoAtNet implementations. Those are additional baselines,
not implementations of the unverified model described below.

## Implemented baseline

The completed `outputs/baseline3` fold-0 run uses a pretrained
`resnet50d.ra2_in1k` encoder and a learned 2,048-to-256 projection. The
maintained [baseline config](../configs/baseline.yaml) now selects pretrained
`rexnet_200.nav_in1k`, with its 2,560-to-256 projection, for the next fresh run.
Baseline3 used global triplet descriptors with joint type and centered physical
position embeddings and 0.1 series dropout. The current spatial-token architecture
and 0.2 series dropout are specified below.

Baseline3 selected epoch 7 at **0.849819 macro AUC** over all 903 fold-0
studies. Its separate 15-study gold diagnostic was **0.915586** and its
report-derived diagnostic was **0.849472**. Training all ten epochs took
1,753.10 seconds for 4,380 optimizer steps and peaked at 11,677,091,328 CUDA
bytes. This is one-fold development evidence on the mixed-label endpoint, not
a complete five-fold OOF result or a comparison with baseline1's different
12-study endpoint. The saved configuration, checkpoint selection, target-level
metrics and timing live under `outputs/baseline3/`.

The completed `outputs/baseline4` fold-0 run uses
`rexnet_200.nav_in1k`, doubled validation image limits, contiguous NCHW
tensors, and a physical batch of eight. It selected epoch 7 at **0.872510
macro AUC**, trained for 2,370.34 seconds, and peaked at 21,505,119,744 CUDA
bytes. Its exact checkpoint bundle produced Kaggle public score **0.910** as
submission `56167669`, using the historical 1× test image limits. This remains
single-fold development evidence. The same checkpoint subsequently scored
**0.913 at 2×** and **0.912 at 3×**, both with inference batch size 2. Current
inference uses 2×; see the [completed comparisons](inference-efficiency.md#inference-image-count-and-memory).

The backbone trains in ordinary training mode: `backbone_eval: false` and
`checkpoint_backbone: false`. Lightning owns precision, optimization, gradient
clipping, early stopping, logging and independent best/latest checkpoints.
Saved checkpoint configuration governs prediction; later edits to the baseline
YAML do not change an existing artifact.

### Active spatial token model

The active `spatial-tokens-five-embeddings` experiment keeps every visually valid
backbone location and applies target-specific attention MIL directly. A shared
linear projection maps `[N, C, H, W]` to `[N, H × W, 256]`; ReXNet-200 gives
49 locations per 224-pixel triplet and up to 3,136 before masking at the existing
64-triplet validation budget. No global pooling, regional queries, grid reduction
or token-to-token self-attention remains. The scorer stays 256→128→12 with its
configured dropout, followed by per-target weighted features and linear heads.

Before LayerNorm, tokens receive five additive embeddings:

| Embedding | Definition |
|---|---|
| 1D | Sinusoidal middle-slice distance in mm from the acquisition midpoint |
| 2D | Augmented-input anchor normalized independently to each image axis; 2→64→256 GELU MLP |
| 3D | Middle-slice patient LPS anchor minus accepted study center, divided by 100 mm on every axis; 3→64→256 GELU MLP |
| Plane | Learned Axial, Coronal, Sagittal embeddings |
| Contrast | Learned metadata pairs `(0,0)` and `(1,1)` for fluid sensitivity/fat suppression |

Plane and contrast lookup buffers derive from configured metadata rather than
slot order. The verified ReXNet adapter derives stride 32 and origin 0 from
its convolutions; other backbone mappings fail explicitly. Geometry and the
coordinate MLPs execute in float32 outside autocast; embeddings are cast to
visual-feature dtype. Grid dimensions may vary independently, including rectangular
grids; the shared geometry has no fixed grid size or stride.

[Visual validity and 3D availability](data-preparation.md#spatial-token-geometry)
are independent. Missing or unreliable 3D uses a shared learned `unknown_3d`
vector while all other embeddings remain. Training replaces available 3D with
that vector with fixed probability 0.10 per eligible study per forward, using
the seeded PyTorch RNG and one decision for the whole study. This does not
rescale embeddings or remove images. Evaluation uses all reliable 3D. Series
dropout remains 0.20, restoring one present type when necessary.

The fresh fold-0 recipe keeps seed 42, `all_5fold_v3`, 224 pixels, 140 mm,
2× validation density, pretrained initialization, augmentation, labels, loss,
optimizer and ten-epoch budget. Strict determinism remains disabled as in the
incoming recipe; the historical average-pooling reference enabled it, so this
execution difference must accompany later comparisons. The five embeddings
and removal of the regional branch constitute one experiment.

Launch manually from the repository root:

```bash
rtk proxy uv run --locked just train spatial-tokens-five-embeddings
```

The completed run is saved in `outputs/spatial-tokens-five-embeddings/`; another
launch requires a fresh run directory. It selected epoch 10 at 0.889506 fold-0
macro AUC and **0.929 public AUC**, later surpassed by
[3dsampling at 0.930](validation.md#2026-09-13-3dsampling-validation-comparison); see the
[validation comparison](validation.md#2026-09-13-spatial-token-validation-comparison)
for target changes, uncertainty, learning curves, timing and memory, and the
[submission record](inference-efficiency.md#spatial-token-submission). The manual
encoding diagnostic has not been run. Matched inference cost remains unmeasured.
Historical checkpoints require their original
source bundles; current source/config identities reject them without migration.

### Historical regional fusion

The replaced model combined global triplet descriptors with 27 width-64 spatial
queries and per-plane attention. The `spatial-fusion-radius` checkpoint selected
epoch 8 at 0.88635162 AUC. Its regional tokens received 4.94% mean final attention,
and omitting them reduced AUC by 0.00069668 on 904 validation studies. Its
independently trained advantage over `crop140mm` was only 0.00022998 AUC;
these single-fold results do not establish repeatability. See the
[recorded diagnosis](validation.md#2026-09-13-per-plane-fusion-usage-diagnosis),
[earlier runtime rejection](validation.md#2026-09-12-spatial-fusion-engineering-result)
and [archived sources](history/regional-sources/README.md).
The max-pooling comparison selected 0.87907846 AUC versus 0.88416888 for its
average-pooling reference; it did not justify promoting max pooling. Regional
bandwidth, finer regional grids and adaptive queries are retired.

#### Per-plane coverage radius

Selected on 2026-09-13, before training: **18 mm**, independently of the fixed
35-mm attention bandwidth. The selection rule takes the 95th percentile of the
nearest valid anchor distance for accepted-center query/plane groups with at
least one valid anchor, then rounds up to the next whole millimeter. Missing
planes are excluded from that distance distribution and remain uncovered.

The measurement used a uniform 128-study sample from the `all_5fold_v3` fold-0
training partition, seed 42. Original dataset indices preserve epoch-zero
training acquisition and triplet sampling with the configured 1× budgets.
Actual prepared 224-pixel crops supply the 7×7 anchors and all-channel validity;
augmentation and series dropout are disabled for radius selection. No held-out
geometry or predictions enter the choice, and no model executes.

Of 128 studies, 121 had accepted centers, supplying 9,801 measured query/plane
groups. Nearest-anchor distances were 10.07 mm at the median, 14.48 mm at p90,
17.29 mm at p95 and 25.37 mm at p99. The selected radius permits **93.78%** of
all sampled regional queries; plane coverage is 89.90% axial, 88.95% coronal
and 92.51% sagittal, counting rejected centers as uncovered. These describe
this pre-augmentation sample, not training-wide coverage or model accuracy.

The [measurement record](../outputs/analysis/fusion-coverage-20260913/results.json)
preserves the effective config, split/label provenance, input and source hashes,
and quantiles. Its sibling `inputs.json`, `nearest_anchors.parquet` and `source/`
preserve selected paths, geometry hashes, per-query distances and executable
sources. Training membership SHA-256 is
`31422731caf633a55b8d7bf2406dfa1fe1f791e78a16482c14ca5fe2089af749`.
Reproduction requires the original analysis source bundle and configuration.
The [coverage source snapshot](history/regional-sources/measure_fusion_coverage.py.txt)
is historical; its 18-mm radius has no role in the current spatial-token model.
The [regional usage results](validation.md#2026-09-13-per-plane-fusion-usage-diagnosis)
remain historical evidence, with repeatability and deployment cost unresolved.

### Three-channel inputs

The baseline uses `data.input_mode: adjacent_triplets`: each three-channel
CNN input contains three consecutive slices from the physically sorted series.
Channels retain their slice order and use the selected ReXNet and historical
RA2 checkpoints' ImageNet normalization: mean `[0.485, 0.456, 0.406]` and standard deviation
`[0.229, 0.224, 0.225]`, after dividing uint8 pixels by 255. The 224 px crop at
0.625 mm covers **140 × 140 mm**, following the
[completed crop comparison](validation.md#2026-09-12-140-mm-crop-result).

The dataset supplies uint8 `[inputs, 3, height, width]` tensors per series.
Collation pads only to the largest input count present in the batch. Each
validity-mask entry represents a complete triplet, and padding never enters the
CNN or attention. A triplet's positional encoding uses its middle slice's
millimeter position relative to the physical center of its acquisition. All
triplets of an acquisition share sampled augmentation strengths; normalization
is applied per channel.

Legacy configs default to `data.input_mode: single_slice`. Their prepared
one-channel tensors are either broadcast to RGB or kept single-channel according
to the saved configuration. Checkpoint loading compares input mode, channel
count, normalization and slice limits, so an older artifact cannot silently
adopt current input semantics. Triplet mode requires three input channels and
channel-appearance limits divisible by three. Baseline1 compatibility and exact
prediction reproduction are recorded in the
[baseline review](baseline-review.md#implemented-corrections).

### Adjacent triplet sampling

Updated on 2026-09-13. `max_images` counts **channel appearances**, including
slices reused by overlapping triplets.
Training limits round the previous half-median budgets up to multiples of
three. `validation_image_multiplier: 2` doubles validation limits. Test
prediction independently defaults to 2×. Export can freeze a different inference
multiplier without changing the checkpoint's training settings:

| Plane | Fluid-sensitive / fat suppression | Training channel appearances | Training CNN inputs | Validation channel appearances | Validation CNN inputs |
|---|---|---:|---:|---:|---:|
| Axial | 0 / 0 | 18 | 6 | 36 | 12 |
| Axial | 1 / 1 | 18 | 6 | 36 | 12 |
| Coronal | 0 / 0 | 15 | 5 | 30 | 10 |
| Coronal | 1 / 1 | 15 | 5 | 30 | 10 |
| Sagittal | 0 / 0 | 15 | 5 | 30 | 10 |
| Sagittal | 1 / 1 | 15 | 5 | 30 | 10 |

A study with all six types uses at most 96 channel appearances and 32 CNN inputs during
training, or 192 channel appearances and 64 CNN inputs during validation. The multiplier
does not change model parameters or checkpoint tensor shapes. Current test
inference defaults to the same 2× limits; 3× allows 288 channel appearances and 96 CNN
inputs per study, capped by available acquisitions. Earlier submitted bundles
used 1× and retain their frozen code. Historical checkpoints require their archived runtime. See [inference memory measurements](inference-efficiency.md#inference-image-count-and-memory).

### ReXNet-100 runtime

The maintained config uses BF16 mixed precision and processes an effective
batch of eight as one physical batch. A bounded maximum-load run on the local
RTX 3090 peaked at 8.40 GiB with `batch_size: 8`; the earlier physical batch of
four with two-step accumulation peaked at 4.55 GiB and did twice as many
forward/backward passes for the same effective batch.

ReXNet-100 uses contiguous NCHW tensors. With channels-last tensors, each new
flattened image count triggered roughly ten seconds of cuDNN setup; MIL batches
vary that count frequently because studies contain different valid series and
series dropout changes it further. Contiguous tensors handled changing counts
in 0.14–0.15 seconds after a 0.79-second first step in the isolated local
benchmark. This is slightly slower for a repeated fixed shape but avoids the
repeated dynamic-shape penalty that dominated real training.

The bounded production-path profile then sustained **35.43 studies/second**
across two measured optimizer steps with real DICOM batches, compared with
**0.54 studies/second** in the channels-last diagnostic dominated by repeated
setup. This profile did not run an epoch or produce a validation score.

Training disables Lightning's parameter summary because its size calculation
does not recognize `bf16-mixed`; this affects only the displayed model-size
estimate. Float32 matrix multiplication uses the `high` policy so supported
operations can use the RTX 3090 Tensor Cores.

The shared [physical sampler](physical-sampling.md#implemented-sampler) selects
Gaussian target depths about the accepted study center, then chooses suppliers
independently of slice count. `data.sampling_sigma_mm: 80.0` is the only variable
sampling parameter. Deterministic validation uses midpoint quantiles; training
uses stratified random quantiles. Both permit overlapping consecutive triplets
and assign each distinct middle-plane geometry at most once per type.
Study geometry, acquisition-midpoint 1D positions, crops, augmentations and model
shape remain unchanged. The prior allocator, triangular selector and their
configuration fields have been removed.

Preparation clips each unique retained source slice once per acquisition, then
gathers its channel appearances. Path failures split original runs and trigger
fresh type selection using fixed seeds and the original study center. Local
percentile fallback retains trustworthy millimetre positions and unknown XYZ
when a type has no physical candidates. Unusable types remain masked and a
study without a complete usable input fails explicitly. See
[overlapping preparation](data-preparation.md#overlapping-triplet-preparation).

The [augmentation viewer](data-preparation.md#preview-augmentations) displays
both the three-channel composite and each individual grayscale channel.

### Ten-epoch schedule and augmentation

Training uses
`max_epochs: 10`, one warmup epoch, and cosine decay over the remaining nine
epochs toward 1% of the peak learning rates. `patience: 10` prevents ordinary
validation fluctuations from stopping this ten-epoch budget early. Best and
latest checkpoints remain separate. The peak backbone/head learning rates are
`2e-4` and `6e-4`, adopted in `configs/baseline.yaml` on 2026-09-12 from
the [overnight comparison](validation.md#2026-09-12-overnight-results).

Each acquisition independently samples affine augmentation with probability
0.7, gamma/contrast with probability 0.5 and Gaussian noise with probability
0.15. Affine strengths are up to 10 degrees rotation, 4% translation and
`[0.95, 1.05]` scale. Gamma uses `[0.85, 1.15]`, contrast uses `[0.9, 1.1]`,
and noise standard deviation uses `[0.005, 0.02]` on the `[0, 1]` scale.
Geometry and scalar strengths are shared by every triplet in an acquisition;
noise pixels differ. Zero background is restored after augmentation.

Evaluation bypasses augmentation. Each family defaults to disabled when absent
from an older config, and probability zero disables it for a paired comparison.
The current combined recipe does not establish the contribution of any one
family; those comparisons remain in
[future work](future-work.md#additional-mri-research-candidates).

Use the [augmentation viewer](data-preparation.md#preview-augmentations) to
inspect original/augmented example-test slices for each plane–contrast type,
with per-series parameters and seeded draws from the shared implementation.

### Study-level knee reflection

Implemented on 2026-09-13. The candidate `spatial-tokens-reflection` adds
`data.reflection_augmentation.probability: 0.5` to the physical-sampling recipe.
A training forward draws one seeded device-RNG value per study, masks it by
[acquisition eligibility](data-preparation.md#reflection-geometry-and-eligibility),
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
validation macro AUC and **0.931 public AUC**, the best recorded public result.
See the [comparison with 3dsampling](validation.md#2026-09-13-reflection-validation-comparison)
for the +0.001012 local and +0.001 public gains, target changes and limitations.

Probability is the only training reflection parameter; zero is the control.
The added draws change subsequent stochastic choices relative to historical
runs, so equal seeds do not imply paired augmentation sequences. CPU preview
and CUDA training generators also differ. The preprocessing semantic identity
records the fixed algorithm and source hashes; saved configuration records the
probability. Historical checkpoints require their archived sources and identities.

The completed fold-0 run was launched manually with:

```bash
rtk proxy uv run --locked just train spatial-tokens-reflection configs/baseline.yaml 0
```

Keep seed 42, `all_5fold_v3`, Steven v4, ReXNet-200, 224-pixel/140-mm crops,
physical sampling, 2× validation density, optimizer and ten-epoch budget fixed.
Use the saved physical-sampling configuration and
[3dsampling result](validation.md#2026-09-13-3dsampling-validation-comparison)
as the reference. Full-fold and per-target AUC, the separate gold diagnostic,
wall time, peak VRAM and checkpoint/config/source identities are recorded in the
[completed comparison](validation.md#2026-09-13-reflection-validation-comparison).
The run directory is complete; another launch requires a fresh run ID. No extra
seeds, baseline reruns or automatic encoding diagnostics were launched.
See the [preview and bounded profile evidence](data-preparation.md#reflection-preview-evidence).

### Baseline history and compatibility

The completed `outputs/baseline1` fold-0 run selected epoch 4 at **0.843997
macro AUC** over twelve reference studies, taking 52.37 minutes through ten
epochs with batch 8 and accumulation 1. It used one-channel inputs, ordinal
positions, a ResNeXt26ts encoder, one CLS transformer and masked max pooling
under the historical gold-only split. Its exact configuration, defects and
reproduction procedure belong in the [baseline review](baseline-review.md).
Later saved configs preserve the RA4 experiment and other intermediate recipes;
none overrides baseline3's current defaults.

## Batch-4 memory fix

Superseded single-slice ResNeXt profiles established two durable constraints.
First, processing hundreds of slices in one encoder batch exceeded the RTX
3090's practical memory capacity; channels-last plus activation checkpointing
reduced one 740-slice batch-four profile from 31.04 GiB allocated to 12.84 GiB.
Second, four to eight loader workers mostly hid preparation time in those
bounded workloads. Exact configurations and measurements remain in
`outputs/profiling/batch4-*`, while the more representative loader comparison
and its negative threading result live in
[data preparation](data-preparation.md#standalone-loading-measurement).

### Profiled training runtime

Those profiles do not describe baseline3, whose triplets greatly reduce CNN
input count and whose saved configuration disables activation checkpointing.
Use full-run timing and memory from the current baseline above for planning.

## Unverified model structure

- The setup was described as MIL with one representation per window rather
  than one per series. The exact pooling/attention mechanism, metadata inputs,
  and output heads are unknown. The available detail does not establish
  whether the implementation contains a transformer.
- Separate metadata features may have been tried early and later dropped.
  This does not establish whether metadata was
  absent from the classifier, or change its role in geometry preprocessing.
- Three neighboring slices are packed into the input channels. Five-slice
  inputs reportedly helped some models, without a quantified result.
- Repeating one slice across three channels tested the same or worse than
  using neighboring slices. Reducing the input convolution to one channel was
  debated without a demonstrated result.
- Input normalization appeared to depend on the backbone, mostly using
  ImageNet normalization.
- `rexnetr200` was recommended as a fast backbone, with about one hour per
  fold at 384 pixels in the recorded setup. `rexnet300` and later
  `rexnetr300` were also named. Confirm the exact library identifiers and
  pretrained checkpoints rather than treating these spellings as interchangeable.
- An unnamed stronger model took roughly 6–8 hours to train. Improvements
  found with a small model reportedly transferred to that larger model.

## Window sampling

Early experiments trained on a changing random subset of about half the slices.
Training on all slices showed no gain; using all slices at inference initially
gave a small lift.

The more specific 2026-09-08 description was:

- Training: 24 windows, each containing three slices.
- Validation: 94 windows were stated explicitly.
- A later explanation used 96 windows and said about 70% of studies had fewer
  than that; increasing the count reportedly reduced the score.
- Outer slices were described as frequently uninformative.

The 94/96 discrepancy, cap scope (study versus series), overlap between windows,
and exact inference policy remain unresolved. The later capped description
qualifies the earlier claim that inference uses all images. Twenty-four windows
also need not mean 72 distinct slices if windows overlap.

## Augmentation and complementary views

- Adding an unspecified augmentation set produced a reported local improvement
  large enough for a single-fold model to beat an earlier seven-model ensemble.
  Public confirmation was explicitly still pending the next day.
- Horizontal flipping improved local results despite concerns about
  side-dependent targets. The label/orientation policy and exact transform
  configuration were not recorded, so this is an ablation lead, not an
  established label-preserving default.
- Whole-series dropout was described on 2026-09-09 as the best overall
  improvement found, forcing learning from other planes. No dropout probability
  or score delta was given.
- Different models selected important windows from different planes. Sagittal
  views were said to dominate, with rough estimates varying from 60% to 70%;
  coronal contribution was once estimated at 25%. The attribution method and
  target breakdown are absent. This does not establish that other planes can
  be discarded without loss.

## Fast backbone search

An unverified experiment used three models' out-of-fold predictions with
leave-one-out inference to identify an important three-slice window.
A reduced dataset mapping one selected window to study targets let candidate
backbones train in about 5–10 minutes. This reportedly helped discover the
`rexnetr300` candidate and transferred to the full task.

A proposed extension would select one important window per target, about
12 windows per study. That extension was not reported as completed. Selection
details, ranking aggregation across models, and validation isolation are missing.

### Finished model comparison

The following unverified completed-run summary was recorded on 2026-09-11.
Configuration, split, seed and artifact identities are unavailable, so these
values rank that experiment set but do not establish a controlled comparison
with repository baseline3. ReXNet-200 has the highest reported AUC and is the
next maintained baseline-config backbone.

| Finished model | Best AUC | Selected epoch | Epochs completed | Cap | Run time | Stop reason |
|---|---:|---:|---:|---:|---:|---|
| ReXNet-200 | *0.908347* | 5 | 11 | 20 | 13m 46s | Patience 6 |
| DINOv3 ConvNeXt Tiny | 0.904386 | 7 | 13 | 20 | 11m 33s | Patience 6 |
| CoAtNet | 0.904336 | 5 | 11 | 20 | 40m 17s | Patience 6 |
| RegNetY-160 IN12k | 0.901071 | 14 | 20 | 30 | 46m 52s | Patience 6 |
| ReXNet-200 ordered map mixer | 0.897025 | 6 | 12 | 30 | 14m 51s | Patience 6 |
| DINOv2 Base reg4 | 0.895418 | 5 | 11 | 30 | 22m 53s | Patience 6 |
| ReXNet-300 | 0.894535 | 8 | 14 | 20 | 24m 40s | Patience 6 |
| RegNetY-120 IN12k | 0.894422 | 6 | 12 | 30 | 23m 46s | Patience 6 |
| DINOv3 ConvNeXt Small | 0.890972 | 8 | 14 | 30 | 19m 04s | Patience 6 |
| RegNetY-064 IN1k | 0.890054 | 12 | 18 | 30 | 27m 41s | Patience 6 |
| ReXNet-200 ordered window mixer | 0.888753 | 9 | 15 | 30 | 18m 04s | Patience 6 |
| DINOv3 Small gated, cap 30 | 0.888279 | 12 | 18 | 30 | 13m 56s | Patience 6 |
| DINOv3 Small mean/max, cap 30 | 0.887999 | 18 | 24 | 30 | 17m 40s | Patience 6 |
| ConvNeXt Tiny supervised | 0.886954 | 9 | 15 | 30 | 13m 14s | Patience 6 |
| DINOv3 Small gated | 0.886726 | 19 | 20 | 20 | 15m 16s | Epoch cap |
| DINOv3 Base | 0.885491 | 5 | 11 | 30 | 20m 00s | Patience 6 |
| DINOv3 Small gated acquisition | 0.884663 | 22 | 28 | 30 | 21m 27s | Patience 6 |
| ReXNet-200 spatial queries | 0.883935 | 10 | 16 | 30 | 20m 46s | Patience 6 |
| RadImageNet ResNet-50 | 0.883363 | 29 | 30 | 30 | 25m 13s | Epoch cap |
| DINOv3 Small gated axis, cap 30 | 0.882248 | 19 | 25 | 30 | 19m 33s | Patience 6 |
| DINOv3 Small gated axis | 0.880551 | 19 | 20 | 20 | 15m 20s | Epoch cap |
| DINOv3 Small mean/max | 0.879972 | 19 | 20 | 20 | 14m 29s | Epoch cap |
| DINOv3 Small Plus | 0.878961 | 7 | 13 | 30 | 11m 26s | Patience 6 |
| CAFormer-S18 IN22k | 0.878179 | 15 | 21 | 30 | 36m 16s | Patience 6 |
| MedDINOv3 Base | 0.875170 | 5 | 11 | 30 | 20m 04s | Patience 6 |
| DINOv2 Small reg4 | 0.874858 | 8 | 14 | 20 | 10m 43s | Patience 6 |
| DINOv3 Small | 0.874582 | 19 | 20 | 20 | 14m 44s | Epoch cap |
| InceptionNeXt Tiny | 0.872661 | 7 | 13 | 30 | 12m 08s | Patience 6 |
| DINOv3 Small last6 | 0.871846 | 10 | 16 | 30 | 7m 55s | Patience 6 |
| ConvNeXt-V2 Tiny FCMAE | 0.871493 | 9 | 15 | 30 | 33m 00s | Patience 6 |
| ImageNet DenseNet-121 | 0.869147 | 8 | 14 | 30 | 16m 19s | Patience 6 |
| DINOv3 Small, cap 30 | 0.868904 | 7 | 13 | 30 | 9m 48s | Patience 6 |
| DINOv3 Large LoRA32 | 0.866161 | 13 | 19 | 30 | 2h 21m 24s | Patience 6 |
| DINOv3 Small CLS + mean | 0.865789 | 5 | 11 | 20 | 8m 12s | Patience 6 |
| BiomedCLIP Base | 0.864368 | 6 | 12 | 30 | 17m 53s | Patience 6 |
| ImageNet ResNet-50 | 0.861902 | 8 | 14 | 30 | 11m 53s | Patience 6 |
| MaxViT Tiny | 0.857384 | 12 | 18 | 30 | 42m 14s | Patience 6 |
| EfficientNetV2-S IN21k | 0.857114 | 7 | 13 | 30 | 15m 16s | Patience 6 |
| RadImageNet DenseNet-121 | 0.816707 | 24 | 30 | 30 | 32m 56s | Patience 6 |
| RadiologyNet ResNet-50 | 0.791780 | 5 | 11 | 30 | 9m 43s | Patience 6 |

## Loss and ensemble diversity

Focal loss improved some target-level diagnostics but worsened others, and
standalone models were described as slightly worse overall. Mixing loss
functions was proposed to improve ensemble diversity; no isolated ensemble
gain was reported. See [validation](validation.md#target-level-diagnostics)
for the AUC/F1 distinction and ensembling uncertainty.

Architectural diversity, decorrelating small models, and distilling ensembles
were recurring directions. Earlier notes also suggested that older, weaker
models could hurt an ensemble. None of the projected final scores are measured
results. The unverified runtime/score pairs are in
[inference and efficiency](inference-efficiency.md#unverified-submission-results).

## Expensive attempt and exploratory ideas

A denoising model took about 24 hours to train, followed by roughly 24 hours
for an autoencoder. Classification using the compressed representation was
slightly worse than using slices. The attempt did not encode across the slice
axis; its intended advantage was faster subsequent iteration through a smaller
cache. Claims about near-lossless compression were aspirations, not results.

Other ideas included exploiting imperfectly aligned retakes for denoising,
3D reconstruction, and model-assisted annotations or segmentation. No
controlled competition results are recorded for those directions.
