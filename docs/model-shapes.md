# Spatial-token model shapes

Current [independent-head model](modeling.md#active-spatial-token-model), with
ReXNet selected by the sigma-40 baseline. CoAtNet is a supported alternative.
Historical 1C changes only the readout and classifier grouping described in
[modeling](modeling.md#historical-disease-sharing).

```mermaid
flowchart TD
    Input["Prepared uint8 triplets<br/>[series, padded inputs, 3, input H, input W]"] --> Keep["Slice mask and retained series<br/>Training series dropout 0.20; restore one if empty"]
    Keep --> Aug["Study reflection, acquisition-shared augmentation and normalization<br/>N retained triplets"]
    Aug --> CNN["One backbone forward_features pass<br/>[N, C, H, W]; 7 × 7 at 224 px<br/>ReXNet C = 2560; CoAtNet C = 1024"]
    CNN --> Project["Flatten spatial axes; shared Linear C to 256<br/>[N, H × W, 256]"]
    Adapter["Verified backbone adapter<br/>Stride 32; ReXNet origin 0, CoAtNet origin 12<br/>Unsupported mapping fails"] --> Anchor["Input-image anchors for variable H and W"]
    Anchor --> XY["2D normalized augmented-image anchors<br/>[H × W, 2] → Linear 2→64 → GELU → Linear 64→256"]
    Anchor --> Inverse["Actual inverse augmentation affine<br/>Float32 outside autocast"]
    Aug --> Inverse
    Inverse --> Visual["Inside all three prepared valid rectangles<br/>Visual validity [N, H × W]"]
    Maps["Middle-slice prepared-to-patient map<br/>Reflection applied consistently to patient geometry"] --> XYZ["Patient LPS minus accepted study center<br/>Fixed scale 100 mm on each axis"]
    Inverse --> XYZ
    Reliable["Native poses + accepted series geometry<br/>Accepted center + finite XYZ"] --> Available["3D availability [N, H × W]"]
    XYZ --> Available
    Available --> Position3D["Available XYZ only → Linear 3→64 → GELU → Linear 64→256<br/>Unavailable → learned unknown_3d [256]"]
    XYZ --> Position3D
    Position3D --> Drop["Training: unknown_3d replaces all available 3D<br/>Probability 0.10 per eligible study; no rescaling<br/>Evaluation: use every reliable position"]
    OneD["Middle-slice mm from acquisition midpoint<br/>Sinusoidal 1D [N, 256]"] --> Sum
    XY --> Sum
    Drop --> Sum
    Plane["Metadata plane lookup<br/>Learned Axial / Coronal / Sagittal [N, 256]"] --> Sum
    Contrast["Metadata contrast lookup<br/>Learned pairs 0,0 / 1,1 [N, 256]"] --> Sum
    Project --> Sum["Add all five embeddings to visual features<br/>Cast embeddings to visual dtype; LayerNorm 256"]
    Sum --> Hidden["Attention hidden features<br/>Linear 256→128 → Tanh → existing dropout"]
    Hidden --> Score["Twelve independent attention scorers<br/>Linear 128→12 in official target order"]
    Visual --> Mask["Study membership and visual mask<br/>Fail before softmax if any study has no valid tokens"]
    Score --> Mask
    Mask --> MIL["Softmax across valid tokens per study and target<br/>Weighted features [batch, 12, 256]"]
    Sum --> MIL
    MIL --> Head["Twelve independent classifier vectors and biases<br/>12 study logits"]
```
