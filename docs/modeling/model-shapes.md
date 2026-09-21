# Spatial-token model shapes

This diagram traces the current [full-slice model](current-model.md) at the
training config's 224 px triplets (0.625 mm, 140 mm). ReXNet-200 is the configured backbone and
CoAtNet a supported alternative. The three slices per input, the 256-wide projection and the
128-wide attention scorer are code constants (`SLICES_PER_INPUT`, `PROJECTION_WIDTH`,
`MIL_ATTENTION_WIDTH`), not config fields.

- **`N`:** retained triplets in the batch.
- **`B`:** studies.

The retired diagrams are elsewhere:

- **Crop selection and learned zoom:** the
  [historical crop-selection sources](../history/crop-selection-sources/README.md).
- **Token graph:** the `00-baseline` run archive and the [model lineage](../history/model-lineage.md#retired-token-graph).
- **Historical 1C:** it changes only the readout and classifier grouping, described in
  [modeling](retired-side-and-compartment-variants.md#historical-disease-sharing).

```mermaid
flowchart TD
    Input["Prepared uint8 triplets<br/>[series, padded inputs, 3, 224, 224]"] --> Keep["Slice mask and retained series<br/>Training series dropout 0.20; restore one if empty"]
    Keep --> Aug["Study reflection, acquisition-shared affine, intensity and noise, normalization<br/>[N, 3, 224, 224]"]
    Aug --> CNN["One backbone forward_features pass<br/>[N, C, 7, 7]<br/>ReXNet-200 C = 2560; CoAtNet C = 1024"]
    CNN --> Project["Flatten spatial axes; shared Linear C→256<br/>[N, 49, 256]"]
    Adapter["Verified backbone adapter<br/>Stride 32; ReXNet origin 0, CoAtNet origin 12<br/>Unsupported mapping fails"] --> Anchor["Input-image anchors for variable H and W"]
    Anchor --> XY["2D normalized augmented-image anchors<br/>[49, 2] → Linear 2→64 → GELU → Linear 64→256"]
    Anchor --> Inverse["Executed inverse augmentation and reflection<br/>Float32 outside autocast"]
    Aug --> Inverse
    Inverse --> Visual["Inside all three prepared valid rectangles<br/>Visual validity [N, 49]"]
    OneD["Middle-slice mm from acquisition midpoint<br/>Sinusoidal 1D [N, 256]"] --> Sum
    XY --> Sum
    Plane["Metadata plane lookup<br/>Learned Axial / Coronal / Sagittal [N, 256]"] --> Sum
    Contrast["Metadata contrast lookup<br/>Learned pairs 0,0 / 1,1 [N, 256]"] --> Sum
    Project --> Sum["Add four embeddings to visual features<br/>Cast embeddings to visual dtype<br/>[N × 49, 256]"]
    Sum --> Norm["LayerNorm 256"]
    Norm --> Hidden["Attention hidden features<br/>Linear 256→128 → Tanh → dropout"]
    Hidden --> Score["Twelve independent attention scorers<br/>Linear 128→12 in official target order"]
    Visual --> Mask["Study membership and visual mask<br/>Fail before softmax if any study has no valid tokens"]
    Inverse --> Axis["Patient maps + knee side + study center<br/>Medial axis [N, 49]; unknown/invalid = 0"]
    Axis --> Bias["Signed tanh(axis / 20 mm)<br/>Learned meniscus and OA strengths"]
    Score --> Anatomical["Add anatomical bias to four compartment-label scores"]
    Bias --> Anatomical
    Anatomical --> Mask
    Mask --> MIL["Float32 softmax across valid tokens per study and target<br/>Weighted features [B, 12, 256]"]
    Norm --> MIL
    MIL --> Head["Twelve independent classifier vectors and biases<br/>12 study logits → masked soft-target BCE"]
```

At 2× validation images, a study with all six series types has up to 64 triplets, which is 3,136
tokens before masking. Batches carry patient maps, geometry validity, knee side and
study-center X as well as valid rectangles. Patient X supplies the compartment attention
bias; visual validity still determines token membership.
