# Model lineage

How the [current model](../modeling/current-model.md) came about, and the variants it replaced.
Each run's saved configuration and source archive remain its reference; historical checkpoints
require their original sources or bundles, because current source and config identities reject
them without migration.

| Date | Model | Fold-0 macro AUC | Public |
|---|---|---:|---:|
| 2026-09-10 | [baseline1](#baseline1): one-channel slices, CLS transformer, gold-only split | 0.843997 (twelve studies) | — |
| 2026-09-11 | [baseline3](#baseline3-and-baseline4): ResNet50D, global triplet descriptors | 0.849819 | 0.878 |
| 2026-09-11 | [baseline4](#baseline3-and-baseline4): ReXNet-200, doubled validation images | 0.872510 | 0.910 (1×), 0.913 (2×) |
| 2026-09-13 | [`spatial-tokens-five-embeddings`](#five-embedding-predecessor) | 0.889506 | 0.929 |
| 2026-09-13 | `spatial-tokens-reflection` (study reflection) | 0.890976 | 0.931 |
| 2026-09-14 | [`reflection-sigma40`](#reflection-sigma-40-champion), the champion | 0.892360 | **0.936** |
| 2026-09-14 | [token graph](#retired-token-graph), retired 2026-09-15 | 0.892376 | — |
| 2026-09-15 | `00-baseline`: the champion without the 3D embedding, graph off | 0.892763 | 0.934 |
| 2026-09-15 | [attention prior](#retired-attention-prior), retired 2026-09-16 | −0.002177 vs `00-baseline` | — |
| 2026-09-15/16 | [crop selection and learned zoom](#retired-crop-selection-and-learned-zoom), dropped | below `00-baseline` | — |
| 2026-09-16 | [full-slice `00-baseline` recipe restored](#restoration-of-the-full-slice-recipe), now the current model | 0.892763 | 0.934 |

Other retired variants (the knee-side embedding, meniscus classifier coupling and the 1A–1C
disease-sharing heads) are in
[retired side and compartment variants](../modeling/retired-side-and-compartment-variants.md);
the replaced regional fusion model is in
[historical regional fusion](../modeling/historical-regional-fusion.md).

## Reflection sigma-40 champion

The submitted champion, preserved on `main` at `ea19772`, uses ReXNet-200, independent attention
and classifiers, reflection probability 0.5 and sampling sigma **40 mm**. The completed
`outputs/training/reflection-sigma40` run selected epoch 7 at **0.892360168 validation macro
AUC** and scored **0.936 publicly**. It remains the best submitted model; runs on the current
code, which lacks its 3D embedding, compare against the matched `00-baseline` control. See the
[paired comparison](../validation/results/sampling/2026-09-14-reflection-sigma-40-result.md) and
[frozen submission runtime](../inference/submissions/early-submission-details.md).

The `meniscus-soft-coupling` experiment
[regressed locally](../validation/results/labels-and-losses/2026-09-14-meniscus-soft-coupling-result.md), and its
training changes were removed on 2026-09-14. Its objective and archived evidence are preserved
under [meniscus classifier coupling](../modeling/retired-side-and-compartment-variants.md#meniscus-classifier-coupling);
it has no measured public score. Later that day the baseline config and maintained model moved
to a token graph that replaced the champion's 3D embedding; the graph was retired unused on
2026-09-15. The overnight `00-baseline` is the champion's recipe without the 3D embedding.
Later on 2026-09-15 the baseline config and model moved to crop selection, and on 2026-09-16 to
learned zoom. That [investigation](crop-selection-investigation.md) was dropped the same day,
and the code went back to the full-slice `00-baseline` recipe.

## Restoration of the full-slice recipe

The restored code of 2026-09-16 is the `00-baseline` source archive with its unused graph
removed:

- **Removed modules:** `token_graph.py` and `crop_selection.py`.
- **Removed config fields:** `model.graph_layers`, `model.graph_geometry_dropout` and every crop
  field.
- **Unchanged from `ea19772`:** `config.py`, `training.py` and `checkpoints.py` (before the
  package restructure).
- **Config:** the baseline YAML, later renamed `configs/resnet50.yaml`, equals `00-baseline`'s
  effective configuration except for its name and output root.

The same day, the package moved into the `data/`, `modeling/`, `training/`, `evaluation/` and
`inference/` subpackages, single-value config fields became code constants, and the preprocessing
identity became `physical_gaussian_overlap_spatial_tokens_reflection_v9`. The
[current model](../modeling/current-model.md#historical-reference-00-baseline) records the result and the
restoration checks. On 2026-09-17 the identity became
`ranked_filtered_physical_gaussian_overlap_spatial_tokens_reflection_v10` (series exclusions and
a fixed evaluation source ranking; model unchanged).

## Retired crop selection and learned zoom

From 2026-09-15 to 2026-09-16, the baseline config and model changed several times:

- **Crop selection:** a MobileNetV3-small scout on 300 px canvases chose ReXNet-200 crops.
- **Learned zoom:** a context glimpse on every slice plus learned detail glimpses.

No run beat `00-baseline`, and the investigation was dropped. The
[investigation record](crop-selection-investigation.md) holds the recipes, results and
findings. The [historical crop-selection sources](sources/crop-selection/README.md) hold the
earlier versions of the model page and the source archives.

## Retired token graph

A local kNN message-passing graph over all spatial tokens was implemented on 2026-09-14
and retired on 2026-09-15 because trained models did not use it. It had replaced the
additive 3D embedding, `unknown_3d` and its study-level 3D dropout, which stay removed.
Two fold-0 runs, both measured with the encoding diagnostic:

| Run | Graph setup | Macro AUC | Control | Without graph |
|---|---|---:|---|---:|
| `token-graph` | Two zero-initialized residual layers before `mil_norm`, head learning rate | 0.892376 | Champion 0.892360 | +0.000007 |
| `graph-forced` | Default output initialization, update after `mil_norm`, learning rate 0.0018 without weight decay | 0.892679 | `00-baseline` 0.892763 | +0.000039 |

The [token-graph result](../validation/results/pooling-and-attention/2026-09-14-token-graph-result.md) owns the
measurements. The plan and graph-era sources are archived as
[historical token-graph sources](sources/token-graph/README.md), and the run source
archives `outputs/training/token-graph/source_snapshot.zip` and
`outputs/training/graph-forced/source_snapshot.zip` reproduce each run. Removing the graph
changed the source, spatial-token and preprocessing identities, so every earlier checkpoint,
including `00-baseline` and `attention-prior`, requires its archived sources.

## Retired attention prior

The prior was implemented on 2026-09-15 from the
[position research](../research/geometry-and-views/position-encoding.md#phase-1-head-only-changes) and retired on
2026-09-16. Its fold-0 run [regressed by 0.002177](../validation/results/pooling-and-attention/2026-09-15-attention-prior-result.md)
against the matched `00-baseline` control, and its learned priors stayed nearly flat.

With `model.attention_prior: gaussian_1d_2d_3d`, it added three per-target log-biases to the
attention scores before masking:

- **Formula:** `-0.5 * sum_k ((q_k - mu_k) / sigma_k)^2`, in float32.
- **Initialization:** `mu = 0` and `sigma = 2`.
- **Training:** head learning rate without weight decay.

| Term | Tokens | Input `q` | Parameters |
|---|---|---|---|
| 3D | Geometry- and visually valid; others got zero | `(abs(dx), dy, dz)` in 100 mm units, relative to the study's mean valid token point | Per target, `[12, 3]` |
| 1D | Visually valid | `abs(1D position)` in 100 mm units | Per plane and target, `[3, 12]` |
| 2D | Visually valid | 2D embedding input: `abs(x)` for Axial and Coronal, signed `x` for Sagittal, always signed `y` | Per plane and target, `[3, 12, 2]` |

The 3D term read each token's patient point, the middle-slice LPS anchor with reflection
`S = diag(−1, 1, 1)` applied to the absolute point, and its geometry validity.

The prior is absent from the restored code, which removes:

- `attention_prior.py`, `model.attention_prior` and `configs/attention-prior.yaml` (removed);
- the per-token points and their validity;
- the collated `patient_maps` and `geometry_valid` tensors;
- the diagnostic's prior variants.

The [reflection preview](../data-preparation/reflection-geometry-and-previews.md)
computes its own LPS anchors. The measured run's sources are in
`outputs/training/attention-prior/source_snapshot.zip`, and the last implementation is in
`outputs/training/label-private-crops/source_snapshot.zip`.

## Five-embedding predecessor

The `spatial-tokens-five-embeddings` run, and every later run up to the champion, also added a
3D embedding:

- **Embedding:** patient LPS minus the accepted study center, divided by 100 mm, through a
  3→64→256 GELU MLP.
- **Missing geometry:** a learned `unknown_3d` vector.
- **Dropout:** available 3D was replaced with `unknown_3d` for 0.10 of studies during training.

The champion (`reflection-sigma40`, 0.892360, public 0.936) carries this embedding. Replacing it
with `unknown_3d` at inference still scored
[0.936 publicly](../inference/submissions/early-submission-details.md).

The run kept everything else fixed: seed 42, `all_5fold_v3`, 224 pixels, 140 mm, 2× validation
density, pretrained initialization, augmentation, labels, loss, optimizer and the ten-epoch
budget.

- **Determinism:** strict determinism stayed disabled, as in the incoming recipe. The historical
  average-pooling reference enabled it, so later comparisons must note this difference.
- **Scope:** adding the five embeddings and removing the regional branch made up one experiment.

It was launched with:

```bash
uv run --locked just train spatial-tokens-five-embeddings
```

The completed run is saved in `outputs/training/spatial-tokens-five-embeddings/`. It selected
epoch 10 at 0.889506 fold-0 macro AUC and **0.929 public AUC**, later surpassed by
[3dsampling at 0.930](../validation/results/sampling/2026-09-13-3dsampling-validation-comparison.md).

- **Details:** the
  [validation comparison](../validation/results/sampling/2026-09-13-spatial-token-validation-comparison.md)
  covers target changes, uncertainty, learning curves, timing and memory.
- **Submission:** the [submission record](../inference/submissions/early-submission-details.md).

## Baseline3 and baseline4

The completed `outputs/training/baseline3` fold-0 run used a pretrained `resnet50d.ra2_in1k`
encoder and a learned 2,048-to-256 projection. Baseline3 used global triplet descriptors with
joint type and centered physical position embeddings and 0.1 series dropout; the maintained
config later selected pretrained `rexnet_200.nav_in1k`, with its 2,560-to-256 projection.

Baseline3 selected epoch 7 at **0.849819 macro AUC** over all 903 fold-0 studies. Its separate
15-study gold diagnostic was **0.915586** and its report-derived diagnostic was **0.849472**.
Training all ten epochs took 1,753.10 seconds for 4,380 optimizer steps and peaked at
11,677,091,328 CUDA bytes. This is one-fold development evidence on the mixed-label endpoint, not
a complete five-fold OOF result or a comparison with baseline1's different 12-study endpoint.
The saved configuration, checkpoint selection, target-level metrics and timing live under
`outputs/training/baseline3/`. Its submission scored **0.878** publicly.

The completed `outputs/training/baseline4` fold-0 run used `rexnet_200.nav_in1k`, doubled
validation image limits, contiguous NCHW tensors, and a physical batch of eight. It selected
epoch 7 at **0.872510 macro AUC**, trained for 2,370.34 seconds, and peaked at 21,505,119,744
CUDA bytes. Its exact checkpoint bundle produced Kaggle public score **0.910** as submission
`56167669`, using the historical 1× test image limits. This remains single-fold development
evidence. The same checkpoint subsequently scored **0.913 at 2×** and **0.912 at 3×**, both with
inference batch size 2. Current inference uses 2×; see the
[completed comparisons](../inference/inference-image-count-and-memory.md).

## Baseline1

The completed `outputs/training/baseline1` fold-0 run selected epoch 4 at **0.843997 macro AUC**
over twelve reference studies, taking 52.37 minutes through ten epochs with batch 8 and
accumulation 1. It used one-channel inputs, ordinal positions, a ResNeXt26ts encoder, one CLS
transformer and masked max pooling under the historical gold-only split. Its exact
configuration, defects and reproduction procedure belong in the
[baseline review](baseline1-review.md). Later saved configs preserve the RA4 experiment and other
intermediate recipes.
