# 2026-09-15 two-crops result

**Below the champion overall; it gains on detail labels and loses on peripheral ones.**
`two-crops` selected epoch 9 at **0.889409** fold-0 macro AUC. That is **−0.003354** below
`00-baseline` (0.892763) and **−0.002951** below the reflection/sigma-40 champion
(0.892360). It gains on meniscus and tibiofemoral OA but loses on Baker's cyst, MCL and
fracture. Averaging its validation predictions with the champion's gives **0.895501**, but
averaging the champion with any same-recipe run gives about the same, so this is not
evidence of complementary information. Down-weighting labels by draft uncertainty does not
explain the peripheral losses, and crop placement is label-independent.

## Setup

- **Recipe:** `configs/two-crops.yaml`, which is `configs/baseline.yaml` with
  `data.crops_per_canvas: 2`. Each study gets two 160 px crops per kept canvas, which roughly
  matches `00-baseline`'s ReXNet-200 pixels. The coarse offset starts at `−log(361/50)`.
  Everything else follows the [crop-selection result](2026-09-15-crop-selection-result.md)
  setup, including `prefetch_factor: 1`.
- **Split and seed:** `all_5fold_v3`, fold 0, seed 42; 3,503 training and 904 validation
  studies.
- **Code:** uncommitted working tree on `meniscus-soft-coupling` (HEAD `ea19772`), with
  `source_snapshot.zip` in `outputs/training/two-crops/`.
- **Resources:**
  - 2,968 s of training, 1.78× `00-baseline`'s 1,670 s.
  - Peak CUDA allocation 24.36 GB, at the RTX 3090's limit, so any larger per-step input
    needs a smaller physical batch.
  - Validation paused once per round of 16 loader workers
    ([validation loader waves](../../modeling/training-runtime-and-memory.md#validation-loader-waves)).

## Training curve

| Epoch index | two-crops | Draft AUC | Champion | One crop per canvas |
|---:|---:|---:|---:|---:|
| 0 | 0.772946 | 0.6916 | 0.789918 | 0.767807 |
| 2 | 0.840321 | 0.6123 | 0.859567 | 0.829105 |
| 4 | 0.871305 | 0.6076 | 0.880818 | 0.858911 |
| 6 | 0.885050 | 0.6015 | **0.892360** | 0.877163 |
| 8 | 0.889028 | 0.6085 | 0.891727 | 0.881721 |
| 9 | **0.889409** | 0.6072 | 0.891347 | 0.883931 |

- **Training:** soft-target BCE fell to 0.44929 at epoch 9 (0.45185 for the champion's
  selected epoch). Gold AUC (11 studies) was 0.9062.
- **The draft degraded:** the coarse-token-only draft fell from 0.69 to 0.60–0.63 and stayed
  there, below the one-crop run's 0.70–0.71. With twice the fine tokens, coarse tokens
  carry even less of the prediction.
- **The one-crop column** after epoch 7 comes from its 15-epoch extension, so late epochs
  are not schedule-matched.

## Per-target comparison with the champion

| Target | two-crops (9) | Champion (6) | Difference |
|---|---:|---:|---:|
| Medial meniscus | 0.9265 | 0.9151 | +0.0114 |
| Lateral OA | 0.8731 | 0.8623 | +0.0109 |
| Medial OA | 0.9094 | 0.9027 | +0.0067 |
| ACL | 0.9212 | 0.9199 | +0.0013 |
| Lateral meniscus | 0.8869 | 0.8871 | −0.0002 |
| Contusion | 0.8878 | 0.8884 | −0.0007 |
| Effusion | 0.8667 | 0.8697 | −0.0030 |
| Synovitis | 0.8887 | 0.8944 | −0.0057 |
| PF OA | 0.8680 | 0.8751 | −0.0070 |
| Fracture | 0.9138 | 0.9279 | −0.0141 |
| MCL | 0.8272 | 0.8428 | −0.0156 |
| Baker's cyst | 0.9035 | 0.9230 | −0.0194 |

- **Crops help detail inside the joint** (menisci, tibiofemoral OA) and hurt peripheral and
  whole-joint findings (Baker's cyst, MCL, fracture, PF OA, synovitis).
- **Averaging is not special.** A plain mean of the two models' validation probabilities
  scores 0.895501, +0.003141 over the champion, with a mean per-target prediction
  correlation of 0.948 (computed at two-crops epoch 8). Averaging the champion with
  same-recipe runs gains about as much or more, with correlations of 0.953–0.965:

  | Averaged with the champion | Mean macro AUC | Gain over the better single run |
  |---|---:|---:|
  | `06-affine-strong` | 0.895172 | +0.002812 |
  | `two-crops` | 0.895501 | +0.003141 |
  | `knee-side-embedding` | 0.895469 | +0.003059 |
  | `00-baseline` | 0.896110 | +0.003347 |
  | `token-graph` | 0.896415 | +0.004039 |
  | `01-sigma30` | 0.896892 | +0.004090 |

## Draft-uncertainty check

`scripts/adhoc/evaluate_crop_placement.py --modes learned,unweighted` rescored the selected
checkpoint with the value map's `ŷ(1−ŷ)` set to a constant, so every label's draft
attention counts equally. The interval is a paired study bootstrap with 2,000 draws.
Artifacts are in `outputs/analysis/t/two-crops-unweighted-20260915/`.

| Placement | Macro AUC | vs learned | Paired 95% interval |
|---|---:|---:|---|
| learned | 0.889424 | — | — |
| unweighted | 0.890133 | +0.000709 | [−0.00032, +0.00183] |

Unweighted placement changed Baker's cyst by +0.0003 and MCL by +0.0014, leaving about
0.019 and 0.014 below the champion. Fracture (+0.0054) and contusion (+0.0033) gained the
most. The weighting is not what starves peripheral targets. Missing whole-image context is
the remaining explanation, consistent with the degraded draft.

This is a single seed on fold 0 with no public score. The unweighted result is inference
only on a checkpoint trained with the weighting.

## Crop inspection

- **Renders:** validation-mode crops with the value map for 10 studies (Baker's cyst, MCL,
  menisci, ACL, fracture, effusion and one negative) are in
  `outputs/analysis/c/crop-inspection-two-crops-20260915/` (`render.py`, PNGs).
- **What they show:**
  - The first crop sits on the joint center in almost every slice.
  - Further crops shift toward the patella and femur or the edges.
  - Crops land on conspicuous bright findings (Baker's cyst fluid, marrow edema) when
    present.
  - Layouts look alike for positive and negative studies.
- **Quantified** over all 904 validation studies (`analyze.py`, `analysis.json`,
  `canvases.parquet`):
  - **Allocation follows value, but value is flat.** Within a study, crops per canvas track
    canvas value mass (Spearman median 0.87). The top canvas holds only 3.1% of a study's
    value. The lower-value half of canvases holds 37% of the value and receives 33% of
    crops.
  - **Central slices get only slightly more crops:** 2.5 crops per canvas within 15 mm of
    the center, 2.0 at 15–30 mm and 1.4 beyond 30 mm.
  - **Placement is label-independent.** For every target and plane, mean crop centers of
    positive and negative studies differ by at most 4 px (2 mm), against 16 px cells and
    160 px crops. With abs(x) for axial and coronal and signed x for sagittal, many
    differences reach p < 0.01 only because of the large sample.
- **Conclusion:** learned placement is a generic joint tiling biased toward central slices
  and bright tissue, not lesion targeting. Reallocating the budget cannot fix a flat value
  map. See [learned crop selection](../../research/learned-crop-selection.md) for how
  published and winning designs obtain direction.
