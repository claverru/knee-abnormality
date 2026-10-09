# Crop selection implementation plan

Drafted and implemented on 2026-09-15, revised on 2026-09-16, and dropped with the
[crop-selection investigation](crop-selection-investigation.md) on 2026-09-16. Moved here from
the repository root; it is kept as the design rationale, not as current guidance. The first run is
recorded in the [crop-selection result](../validation/results/2026-09-15-crop-selection-result.md).

**Revised on 2026-09-16.** Crops are now 224 px (112 mm), with static gains (no
covered-footprint bookkeeping) and a value map that keeps its twelve per-label values: summing
them collapsed crops to one habit per plane, 1–4 mm apart between studies with and without a
finding, while the per-label argmax cells on one canvas sit 56.7 mm apart
(`outputs/analysis/c/crop-diversity-20260916/crop_diversity.json`). A per-label round-robin under
a shared `crop_canvas_fraction` budget then gave way, the same day, to **label-private crops**:
each label independently takes exactly `data.crops_per_label` (2) crops on distinct canvases,
preferring planes it has not used yet; once its canvases run out it adds cells at least 3 scout
cells from its crops on a used canvas instead of repeating a crop.
Labels do not compete, every (label, crop) pair is encoded, and ReXNet-200 is conditioned on the
crop's label by a zero-initialized FiLM at `features.10`. Each label pools all coarse tokens and
only its own fine tokens in two softmaxes mixed by a learned gate, which replaces the union
softmax and its coarse offset (decision 9). The numbers below describe the original 160 px
design; the later recipes are summarised in the
[investigation record](crop-selection-investigation.md#recipes).

## Goal

Give the main backbone more detail where abnormalities are, at lower compute than today.
Every position-to-head experiment tied or regressed (knee side, 3D embedding, token graph,
attention priors), so this changes which pixels the backbone sees instead.

A light scout sees every sampled triplet whole at a larger field of view. Its cells become
coarse tokens in the same attention pool as the crop tokens. The model makes a draft
prediction from coarse tokens, then crops where the loss is most sensitive for labels it is
still unsure about. One loss trains everything. The sampler and the number of triplets are
unchanged, and the number of crops equals the number of triplets.

## Vocabulary

| Term | Meaning |
|---|---|
| Triplet | 3 adjacent native slices stacked as channels |
| Sampler | Today's physical Gaussian slice sampler, unchanged |
| Canvas | A prepared triplet at the larger field of view: 300 px at 0.5 mm = 150 mm |
| Scout | `mobilenetv3_small_100.lamb_in1k`, stride-16 and stride-32 features, trained end to end |
| Cell | One stride-16 position on a canvas: 19×19 per canvas |
| Anchor | A cell's or token's center in canvas pixels, from convolution geometry |
| Coarse token | One cell's scout features plus embeddings: 361 per canvas |
| Crop | 160 px = 80 mm integer window of the augmented canvas |
| Fine token | One ReXNet-200 position on a crop: 25 per crop |
| Draft prediction | Shared attention and head applied to coarse tokens only |
| Value map | Expected loss-gradient size per cell, summed over labels |
| Footprint | Cells whose anchors fall inside a crop: 10×10 |
| Budget | Crops per study = `data.crops_per_canvas` × canvases kept after series dropout (1 in the baseline, 2 in `configs/two-crops.yaml`) |
| Focus | A selected cell; the crop is centered on its anchor |

## Shape schema

`K` kept canvases in the batch, `k_b` in study `b`, `N = K` crops, `T` = 12 targets.

```text
canvases uint8 [K,3,300,300] → reflection → float → affine, intensity, noise (per series, as today)
→ scout: s16 [K,48,19,19], s32 [K,576,10,10] → s32 sampled at s16 anchors → concat [K,624,19,19]
→ linear → [K,361,256] + 1D, 2D (canvas), plane, contrast → LayerNorm → coarse tokens
→ attention MLP → coarse scores r [K,361,T]

draft:     softmax of r per study and label → a; pooled → head → ŷ [B,T]
value:     v = ŷ(1−ŷ) · ‖w_t‖ · a (detached) → Σ_t → value map E [K,19,19]
coverage:  k_b greedy steps per study → foci (canvas, cell) [N]
crops:     integer windows of the augmented canvas → normalize → [N,3,160,160]
→ ReXNet-200 [N,2560,5,5] → linear → [N,25,256] + embeddings from the canvas → LayerNorm → fine tokens
→ fine scores = attention MLP(fine) + r at the cell under each token [N,25,T]

final:     scores (r + c for coarse, fine scores) → masked softmax per study and label over coarse ∪ fine
→ pooled [B,T,256] → head → logits [B,T] → BCE, the only loss
```

## Decisions

1. **Canvas augmentation.** Reflection, affine, intensity and noise run on whole canvases,
   per series, exactly as today's pipeline runs on triplets. The scout sees the augmented
   canvas, and crops are integer windows of it. Rotation corners are zero-filled only at
   canvas corners, as today.
2. **Coarse tokens.**
   - The stride-32 map is sampled bilinearly at the stride-16 anchors. For 300 px that is
     exact: stride-16 anchors `16i` sit at `i/2` on the 10×10 grid.
   - It is concatenated with the stride-16 map, so selection has 19×19 cells with
     final-stage semantics.
   - 1D position, plane and contrast come from the canvas. The 2D position is the cell
     anchor in the augmented canvas frame.
   - Visual validity and patient points go through `feature_geometry` with the scout
     mapping (origin 0, stride 16), the canvas augmentation inverse and the reflection
     inverse.
3. **Draft prediction.** Softmax of the coarse scores over the study's valid coarse
   tokens, per label, gives attention `a`. The shared head on the pooled result gives
   `ŷ`. No separate loss.
4. **Value map.**
   - With `z_t = w_t · Σ_n a_n x_n + b_t`, the loss gradient for a token is
     `∂L/∂x_n ≈ (ŷ − y) · a_n · w_t`.
   - Labels are unknown at selection, so take the expectation under the draft:
     `E|ŷ − y| = 2ŷ(1−ŷ)`.
   - Per cell and label, `v = ŷ(1−ŷ) · ‖w_t‖ · a`, detached. The constant 2 drops out,
     since selection ignores scale.
   - `E = Σ_t v`. `a` says where each label looks, and `ŷ(1−ŷ)` says which labels are
     still undecided in this study.
5. **Selectable cells.** Two conditions:
   - **Ring rule:** the crop leaves the canvas by less than one stride,
     `max(s/2 − c, c + s/2 − S) < stride` per axis (anchor `c`, crop `s`, canvas `S`).
     For 300 px and a 160 px crop, cells 5–14 per axis are selectable.
   - **Canvas validity:** the anchor lies inside the valid rectangle of all three channels.
6. **Greedy coverage.** For `k_b` steps, all studies in parallel:
   1. The gain of each selectable cell is the sum of `E` over the uncovered cells of its
      footprint.
   2. Pick one focus per study: `argmax(log gain + Gumbel)` in training, which samples in
      proportion to gain; `argmax` in validation and inference.
   3. Mark its footprint covered on that canvas.
   - Covering a canvas takes 4 crops against a budget of 1–2 per canvas, so gains never run out.
7. **Crops.**
   - Top-left = anchor − crop/2, cut from the augmented canvas before normalization.
   - The canvas is padded with zeros by one stride, the ring rule's maximum overflow.
   - Fine token anchors land exactly on coarse anchors, because `crop/2` and ReXNet's
     stride 32 are multiples of 16. Each fine token therefore has a coarse cell.
8. **Coupled fine attention.** Each fine token's score adds the raw coarse score of the
   cell under it. When fine evidence is useful, the loss raises that cell's coarse score,
   so the draft attention, and with it selection, learns where zooming pays off.
9. **Union pooling.**
   - Coarse tokens get a learned per-label offset `c`, initialized to
     `−log(cells per canvas / (crops per canvas × tokens per crop))`, `−log(361/25)` with one
     crop per canvas. Both token sets then start
     with equal attention mass, so ReXNet's gradient is not diluted 14× at the start.
   - The offset cancels inside the draft softmax, so it does not affect selection.
   - The optional attention prior applies to both token sets.
10. **No alternatives kept.** This replaces the full-triplet path; `configs/baseline.yaml`
    carries the recipe.
11. **Geometry constraints**, checked at model construction:
    - `crop_size < image_size`;
    - `crop_size / 2` and the main backbone stride are multiples of the scout stride.

## Implementation steps

1. **`config.py`:** `DataConfig.crop_size`, `ModelConfig.scout_backbone`.
2. **`spatial_tokens.py`**
   - Scout mappings for stride 16 and 32, verified like `BACKBONE_MAPPINGS`.
   - `feature_geometry` takes per-image anchor offsets and the canvas frame for `xy`.
   - Update `SPATIAL_TOKEN_IDENTITY`.
3. **`crop_selection.py`** (new)
   - `CoarseScout`: `timm.create_model(..., features_only=True, out_indices=(3, 4))`, the
     stride-32 resampling at stride-16 anchors, concatenation and the linear projection.
   - `selectable_cells`: the ring rule and canvas validity.
   - `value_map`: from `ŷ`, `a` and the head weights.
   - `greedy_coverage`: footprint sums over uncovered cells, Gumbel in training and argmax
     in eval, per-study budgets padded to `max k_b`.
   - `cut_crops`: integer windows from the padded canvases, with fine-token-to-cell indices.
   - `CROP_SELECTION_IDENTITY`.
4. **`augmentation.py`:** today's triplet augmentation applied to canvases; no per-crop warps.
5. **`model.py`**
   - `encode_batch`, in order:
     1. series dropout;
     2. reflection;
     3. canvas augmentation;
     4. coarse tokens and scores;
     5. draft;
     6. value map;
     7. coverage;
     8. crops;
     9. normalization;
     10. ReXNet;
     11. fine tokens and coupled scores.
   - It returns coarse and fine tokens, scores and validity, plus the draft logits.
   - The main backbone is probed at `crop_size`.
   - `pool_encoded` does the union pooling with the coarse offset and the optional prior.
6. **`training.py`**
   - `scout.backbone.*` uses `backbone_lr`; everything else new uses `head_lr`.
   - Log `val_draft_macro_auc` (draft logits with the existing metrics).
7. **`preprocessing_identity.py`, `checkpoints.py`, `scripts/pipeline/train.py`,
   `scripts/pipeline/export.py`:** hash `crop_selection.py`, and include the
   crop-selection identity and scout mappings.
8. **Scripts**
   - `scripts/tools/preview_augmentations.py`: canvases with foci and crop outlines, and
     the crops.
   - `scripts/adhoc/diagnose_encoding_usage.py`: kept runnable with the new encoded keys.
   - `scripts/adhoc/profile_training.py`: profile the new recipe.
   - Update the catalogs.
9. **`configs/baseline.yaml`:** `data.spacing_mm: 0.5`, `data.image_size: 300`,
   `data.crop_size: 160`, `model.scout_backbone: mobilenetv3_small_100.lamb_in1k`.
   Update `crop130.yaml` and `attention-prior.yaml` only as needed to validate.
10. **Docs:** modeling notes, model shapes and `docs/ongoing.md` once implemented.

## Compute and memory (estimates, to measure)

- **Main backbone:** 160 px crops cost 0.51× of 224 px triplets.
- **Scout:** at 300 px, about 7% of today's backbone compute.
- **Coarse tokens:** about 92k per training batch through the projection, LayerNorm and
  the 256→128→12 attention MLP, about 1.5% of the main backbone.
- **Union softmax:** `[B,T,~99k]`, about 40 MB in training and 80 MB in validation.
- **Greedy coverage:** at most 64 steps of footprint sums on 19×19 maps.
- **Canvases:** 1.79× today's pixels in loader CPU and queue RAM, about 8.6 MB per
  training study, and 0.9 GB float on GPU at batch 8. Check with
  `profile_training --mode loader` and `--mode gpu`.
- **Sampler:** its in-plane radius `image_size · spacing_mm / 2` becomes 75 mm instead of 70 mm.

## Parameters

No selection constants:
- the budget, selectable cells, footprint and coarse offset initialization are derived;
- the value map comes from trained weights;
- sampling is proportional to gain.

The design choices are the canvas field of view and spacing, the crop size and the scout
backbone.

## Risks and diagnostics

- **Draft calibration:** `ŷ` is the shared head on coarse tokens only and is never trained
  on its own. Watch `val_draft_macro_auc`.
- **Rare labels:** `ŷ(1−ŷ)` is small for rare labels except in suspicious studies. Compare
  per-label AUC with `00-baseline`.
  - If they lose, the ablation is `ŷ(1−ŷ) = 1`.
- **Exploration:** it depends on how sharp attention becomes, and coverage spreads crops.
  - If foci stop changing across epochs, add a temperature.
- **First-order rule:** the value map ignores how attention itself shifts after zooming.

## Evaluation

- Compare with `00-baseline` (0.892763) on fold 0 using the
  [fold-run comparison](../validation/fold-run-comparison.md). Given ±0.002 run
  variability, confirm a gain with a second seed.
- A gain mixes resolution, field of view, coarse context and learned placement. If it
  wins, isolate placement with random selectable foci.
- Inspect validation foci and crops with the preview tool.

## Evidence

- [Input resolution research](../research/input-resolution.md):
  - stored images are interpolated about 2×, and 0.5 mm is the useful limit;
  - 140 mm clips posterior tissue in 34% of sagittal series (150 mm: 8%);
  - the joint line sits 12–17 mm below the image center;
  - 130 mm crops lost on MCL, Baker's, effusion and medial meniscus.
- Visual inspection (2026-09-15, `outputs/analysis/c/crop-inspection-20260915/`):
  - three lesions (bucket-handle fragment, ACL tear, plateau edema) are recognizable
    at 0.625 mm;
  - 0.3125 mm adds noise texture but no new structure;
  - a fixed 70 mm window at the image center misses the joint in sagittal slices, so
    placement must be learned.
