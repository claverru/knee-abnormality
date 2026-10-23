# 2026-09-18 attention temperature and evaluation count submissions

**A new best of 0.939: `ranked-filtered-seed42` with attention temperature 0.75 and ×4 images,
against 0.933 for the same checkpoint at temperature 1.0 and ×2.** The temperature and the count
together are worth +0.005 to +0.006 publicly, above the reflection/sigma-40 champion's 0.936. The
one recipe change submitted the same day, `series-share`, lost.

| Submitted (UTC) | Model and settings | Public | Submission |
|---|---|---:|---|
| 06:28 | `ranked-filtered-seed42`, τ 0.75, ×4 images, batch 2 | **0.939** | – |
| 06:19 | the same bundle as 06:11 (duplicate, see below) | 0.938 | 56322720 |
| 06:11 | `ranked-filtered-seed42`, τ 0.75, ×3 images, batch 2 | 0.938 | 56322551 |
| 06:22 | `07-series-share-1.0`, default inference (τ 1.0, ×2) | 0.929 | – |
| 08:04 | `robust-center-20-topup-26` (fold-0 ×1 0.888021), τ 0.75, ×4, with the whitelisted `move_batch` and CUDA-event timing | 0.938 | 56325019 |
| 2026-09-17 11:40 | `ranked-filtered-seed42`, τ 1.0, ×2 (the baseline for the rows above) | 0.933 | – |

## What the rows mean

- **Temperature** divides each token's attention score by τ before the per-label softmax, which
  sharpens pooling. It was measured on the token caches at +0.0024 macro AUC on fold 0
  ([result](2026-09-17-attention-temperature.md)).
- **Bundles carry τ in the weights.** `ranked-filtered-seed42` predates
  `data.robust_center_outlier_mm`, so its saved config does not validate against the current tree
  and the bundle had to be exported from its archived sources with the matching revision of the
  export script. Instead of adding the new option to that older code, the scorer's final linear
  layer was divided by τ, which is exactly equivalent to `score / τ`, giving checkpoint
  `b4ab10f1…` from `outputs/submission/ranked-filtered-seed42-tau075.ckpt`. The bundles therefore
  record `attention_temperature: 1.0`; the temperature is in the weights.
- **Bundles:** `outputs/submission/exports/knee-ranked-filtered-tau075-3x-20260917` and
  `knee-rf-tau075-4x-20260918`, both batch 2 per device and 16 loader workers, FP16 on two T4s.
  Batch 2 was chosen because the 2026-09-11 ×3 submission completed at that size; no memory
  profile was run.

## Reading

- **Fold 0 predicted the temperature and ×3 gain well:** the cache gave +0.0046 for the pair, and
  the leaderboard gave +0.005.
- **×4 added one thousandth over ×3**, which is a single rounding step and inside public-score
  noise; fold 0 had the two tied (0.89539 against 0.89537). The count question is therefore open
  above ×3, and ×4 costs a third more inference time than ×3, which counts against the
  [efficiency prize](../../../inference/efficiency-leaderboard.md#the-exchange-rate).
- **The two changes are not separated.** Fold 0 attributes roughly +0.0023 to each, but no
  submission isolated them.
- **The recipe change is publicly neutral.** `robust-center-20-topup-26` scored 0.938 at the same
  τ and count as the 0.939 above, despite beating its own control by 0.003 on fold 0 — the third
  time a sub-0.005 fold-0 difference has failed to carry. Its run also carried the uncommitted
  inference changes (a whitelisted `move_batch` and CUDA-event timing), which cannot affect
  accuracy; the visible-test worker time was 26.1 s against 29.9 s for the same settings before,
  but three studies are dominated by startup, so that is not a speed measurement.
- **`series-share` disagrees with fold 0.** It topped the overnight table (+0.0044 over the
  control mean) and then lost 0.004 publicly
  ([overnight results](../campaigns-and-queues/2026-09-18-overnight-results.md)). Between one
  unseeded run on a 904-study fold and the hidden test, the leaderboard is the stronger evidence.
- **Duplicate:** 06:11 and 06:19 submitted the same bundle. The 06:09 command was interrupted after
  it had already completed, and the second submission was sent on the assumption it had not. Both
  scored 0.938, which at least confirms the score is reproducible.

## Later evidence and decision

Geometry caps the count (fold-0 sampler replay of the `k_05` recipe, 2026-09-22): ×4 already takes
80% of the triplet windows of a slot's only 2D series and decodes 94% of its slices; ×5 and ×6
reach 92% and 98% for +18% and +29% triplets but only +5% and +9% decoded slices per study, and
every step past ×6 adds triplets from 3D series alone
(`outputs/analysis/a/acquisition-geometry-20260922/capacity_*`). On the 2026-09-17 token caches every
candidate scored −0.0018 against 24 per type at τ 0.75
([bag size](2026-09-17-attention-temperature.md#bag-size)). On `oblique`,
×5 at τ 0.70 scored 0.945 publicly against 0.944 at ×4 τ 0.75 (S47, S46), with +0.0003 on fold 0
([result](../sampling/2026-09-22-oblique-groups-result.md#temperature-and-count)).
**Dropped 2026-09-24 (user decision):** submissions stay at ×4 τ 0.75; no count above ×4 is
pending.

## Questions left open

- Whether τ 0.70 beats 0.75 at the larger counts; on fold 0 the optimum shifts to 0.70 at ×3 but
  by 7e-5.
- τ is free at inference and independent of the recipe, so every later submission carried it
  (×4 τ 0.75 from S31 on).
