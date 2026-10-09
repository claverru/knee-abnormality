# 2026-09-15 crop-selection result

**Below its control, but learned placement works.** The first
[crop-selection](../../history/crop-selection-investigation.md#recipes) run reached
**0.887151** fold-0 macro AUC at epoch index 14 of a 15-epoch extension. That is
**−0.005612** below `00-baseline` (0.892763, 10 epochs). At inference on the same
checkpoint, learned placement beats random and centered crops by about 0.005, with paired
intervals that exclude zero. ReXNet-200 sees half the baseline's pixels (one 160 px crop
per canvas instead of one 224 px triplet), which is the most likely source of the gap.

## Setup

- **Recipe:** `configs/baseline.yaml` as of this run. 300 px canvases at 0.5 mm (150 mm),
  one 160 px crop per kept canvas, `mobilenetv3_small_100.lamb_in1k` scout, ReXNet-200,
  sigma 40, reflection 0.5, series dropout 0.2, no attention prior.
- **Split and seed:** `all_5fold_v3`, fold 0, seed 42; 3,503 training and 904 validation studies.
- **Code:** uncommitted working tree on `meniscus-soft-coupling` (HEAD `ea19772`); each
  run directory holds `source_snapshot.zip`.
- **Run history:**
  - `outputs/training/crop-selection` trained epochs 0–7, then was killed by the Linux
    out-of-memory killer during epoch 8. With `prefetch_factor: 2`, 300 px loader queues
    held about 10 GB of the 30 GB WSL memory.
  - `outputs/training/crop-selection-extended` resumed from its `last.ckpt` with
    `prefetch_factor: 1` and `--max-epochs 15`. The learning rate continued from the
    checkpoint's 12.6% of peak with a cosine decay to the minimum. Epochs 8–9 therefore
    differ from what the 10-epoch schedule would have produced.

## Training curve

| Epoch index | Macro AUC | Draft AUC | Gold (11) |
|---:|---:|---:|---:|
| 0 | 0.767807 | 0.7025 | 0.7964 |
| 2 | 0.829105 | 0.6967 | 0.9178 |
| 4 | 0.858911 | 0.6981 | 0.9100 |
| 6 | 0.877163 | 0.7035 | 0.9124 |
| 7 | 0.881115 | 0.7089 | 0.9187 |
| 8 (resumed) | 0.881721 | 0.7111 | 0.9215 |
| 10 | 0.884992 | 0.7074 | 0.9187 |
| 12 | 0.886896 | 0.7103 | 0.9132 |
| 14 (selected) | **0.887151** | 0.7105 | 0.9215 |

- **Slower and lower than the control:** `00-baseline` was already at about 0.8927 at the
  comparable epoch 7 ([attention-prior result](2026-09-15-attention-prior-result.md)). The
  last two epochs added only +0.0003.
- **The draft did not learn:** the coarse-token-only draft stayed at 0.69–0.71 macro AUC
  for all 15 epochs, while the final prediction rose from 0.77 to 0.887.
- **Soft-target BCE:** 0.45384 at the selected epoch (0.44767 for `00-baseline`).
- **Resources:** the extension trained 7 epochs in 1,217 s (about 174 s per epoch
  including validation). Peak CUDA allocation was 14.5 GB (20.02 GB for `00-baseline`).

## Placement ablation

`scripts/adhoc/evaluate_crop_placement.py` scored the selected checkpoint three times,
changing only where crops go:
- **learned:** greedy coverage on the value map.
- **uniform:** a constant value map, with Gumbel-sampled, coverage-aware positions,
  seed 20260915.
- **center:** one crop per canvas on the selectable cell nearest the center.

Intervals are paired study bootstraps with 2,000 draws. Artifacts are in
`outputs/analysis/c/crop-placement-20260915/`.

| Placement | Macro AUC | vs learned | Paired 95% interval | Gold (11) |
|---|---:|---:|---|---:|
| learned | 0.887138 | — | — | 0.9215 |
| uniform | 0.881267 | −0.005871 | [−0.00933, −0.00231] | 0.9197 |
| center | 0.881947 | −0.005191 | [−0.00856, −0.00194] | 0.9272 |

| Target | learned | uniform | center |
|---|---:|---:|---:|
| ACL | 0.9227 | 0.9149 | 0.9222 |
| MCL | 0.8230 | 0.8286 | 0.8128 |
| Medial meniscus | 0.9232 | 0.9167 | 0.9198 |
| Lateral meniscus | 0.8758 | 0.8656 | 0.8642 |
| Medial OA | 0.9073 | 0.9017 | 0.9070 |
| Lateral OA | 0.8680 | 0.8635 | 0.8630 |
| PF OA | 0.8570 | 0.8502 | 0.8594 |
| Effusion | 0.8668 | 0.8631 | 0.8667 |
| Synovitis | 0.8880 | 0.8871 | 0.8916 |
| Baker's cyst | 0.9044 | 0.8949 | 0.8767 |
| Contusion | 0.8885 | 0.8769 | 0.8883 |
| Fracture | 0.9209 | 0.9120 | 0.9117 |

- **Peripheral targets need placement:** centered crops lose most on Baker's cyst
  (−0.028), lateral meniscus (−0.012) and MCL (−0.010). Central targets (ACL, medial OA,
  effusion, synovitis) are unchanged.
- **Selection works despite the flat draft AUC:** selection uses the draft's attention
  maps, not only its predictions.
- **Caveat:** the checkpoint was trained with learned placement, so uniform and centered
  crops are shifted inputs at inference, which inflates the gap. A run trained with random
  placement is the clean control.

## Conclusion and next step

Learned placement adds information, but one 160 px crop per canvas gives ReXNet-200 about
0.51× the baseline's pixels, and the run finished 0.0056 below the control. The follow-up
with two crops per canvas, which matches the baseline's backbone pixels, is recorded in the
[two-crops result](2026-09-15-two-crops-result.md). This is a single seed on fold 0 with
no public score.
