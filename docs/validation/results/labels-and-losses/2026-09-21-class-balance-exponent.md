# 2026-09-21 class-balance exponent: validation and public leaderboard disagree

**Validation and the leaderboard order the five weighting strengths differently.** On fold 0,
validation rises with weighting strength (unweighted 0.891925, k=0.25 0.898255, k=0.5 0.899042,
k=0.75 0.901737, k=1 0.902049). The public leaderboard peaks at k=0.5 (0.945) and falls on both
sides: k=0.25 0.944, unweighted 0.943, k=0.75 0.942, k=1 0.940. The
earlier S34→S38 pair points the same way. Validation targets come from report extraction, as the
training targets do; the test uses official labels. So validation is not a reliable judge of loss
weighting. The user adopted k=0.5 as the recipe on 2026-09-22 (see [conclusions](#conclusions)).

## Runs

All five runs are ReXNet-200 on `all_5fold_scanner_v1` fold 0, trained on
[combined labels](../../../public-labels/combined-labels.md) with seed 2263738815 (training is not
deterministic), for ten epochs. The only difference between them is the weighting.
`optimizer.warmup_start_ratio: 0.1` equals the previously hard-coded warm-up start, so the
control is unchanged. The [supervision section](../../../modeling/supervision.md#fixed-class-balanced-bce) owns the
formula. The candidates were trained from the uncommitted working tree on `79d5ba8`; each run's
`source_snapshot.zip` captures it. `k_025` and `k_075` were added on 2026-09-22.

| Run | Weighting | Best epoch | Best / last macro AUC (×1, τ 1) | Submission | Public |
|---|---|---:|---|---|---:|
| `fold0-combined-scanner` | `class_balanced_bce: true`, k=1 | 10 | 0.902049 / 0.902049 | S40, bundle 58ace0fa | 0.940 |
| `k_075` | `class_balanced_bce: true`, `class_balance_exponent: 0.75` | 8 | 0.901737 / 0.900493 | S45, bundle e5b6d3ee | 0.942 |
| `k_05` | `class_balanced_bce: true`, `class_balance_exponent: 0.5` | 8 | 0.899042 / 0.898724 | S41, bundle da8f83e6 | **0.945** |
| `k_025` | `class_balanced_bce: true`, `class_balance_exponent: 0.25` | 8 | 0.898255 / 0.897618 | S44, bundle cd48ab03 | 0.944 |
| `k_00` | `class_balanced_bce: false` | 8 | 0.891925 / 0.891121 | S42, bundle 97b67c16 | 0.943 |

Each submission used the selected checkpoint with ×4 images, τ 0.75, batch size 4 and 16 workers,
without TTA. Run artifacts are in `outputs/training/<run>/`, and bundles are in
`outputs/submission/exports/{combined-scanner,k05,k00}-20260921-x4-tau075/` and
`outputs/submission/exports/{k025,k075}-20260922-x4-tau075/`. At k=0.5 the
positive:negative weight ratio is 3.6:1 on Fracture, down from 12.9:1 at k=1.

## Validation against the k=1 control

Paired report-group bootstrap, 2,000 draws, from
[`compare_fold_runs`](../../run-comparison-and-diagnostics.md#fold-run-comparison). Artifacts are in
`outputs/analysis/{k075,k05,k025,k00}-vs-combined-scanner/comparison.json`.

| Candidate | Macro Δ [95%] | Meniscus | OA | Other eight |
|---|---|---:|---:|---:|
| k=0.75 | −0.00025 [−0.00345, +0.00304] | +0.0022 | −0.0004 | −0.0008 |
| k=0.5 | −0.00299 [−0.00607, +0.00018] | −0.0060 | −0.0027 | −0.0023 |
| k=0.25 | −0.00375 [−0.00760, −0.00003] | −0.0088 | −0.0002 | −0.0034 |
| unweighted | −0.01006 [−0.01570, −0.00474] | −0.0070 | +0.0007 | −0.0135 |

| Label | k=1 | k=0.75 Δ | k=0.5 Δ | k=0.25 Δ | unweighted Δ |
|---|---:|---:|---:|---:|---:|
| ACL | 0.9462 | −0.0062 | −0.0064 | −0.0014 | −0.0087 |
| MCL | 0.9230 | +0.0083 | 0.0000 | −0.0018 | −0.0285 |
| Medial Meniscus | 0.9268 | −0.0003 | −0.0025 | −0.0065 | −0.0033 |
| Lateral Meniscus | 0.9005 | +0.0047 | −0.0095 | −0.0110 | −0.0107 |
| Medial OA | 0.8981 | +0.0012 | −0.0011 | +0.0007 | −0.0012 |
| Lateral OA | 0.8711 | −0.0021 | −0.0040 | −0.0011 | +0.0024 |
| PF OA | 0.8645 | +0.0044 | +0.0051 | −0.0011 | −0.0082 |
| Effusion | 0.8598 | +0.0046 | +0.0036 | +0.0082 | +0.0097 |
| Synovitis | 0.8916 | −0.0052 | −0.0043 | +0.0005 | −0.0034 |
| Baker's | 0.9263 | −0.0099 | −0.0119 | −0.0229 | −0.0280 |
| Contusion | 0.8915 | +0.0043 | +0.0103 | +0.0028 | −0.0114 |
| Fracture | 0.9251 | −0.0074 | −0.0153 | −0.0121 | −0.0300 |

Validation was measured at ×1 and τ 1 only. The submission setting (×4, τ 0.75) changes some
labels substantially; for example, Contusion on the control moves from 0.891 to 0.918. The
candidates have not been validated at that setting.

## The earlier pair

S34 (`rexnet-fold0`, ordinary BCE) and S38 (`13-control-repeat`, class-balanced BCE) differ in
configuration only in `trainer.class_balanced_bce` and the seed. Both used Steven v4 labels,
`all_5fold_v3` and ×4 with τ 0.75. Validation rose by +0.0129 (0.883671 → 0.896582) and the public
score fell by 0.003 (0.936 → 0.933). Across both pairs, more weighting always meant higher
validation, and never a higher public score.

## Rescoring against other label sources

Script and output: `outputs/analysis/l/label-source-rescoring-20260921/`. Fold-0 predictions were
scored against each public report extractor, thresholded at 0.5, on the 892 non-gold studies. The
table shows macro Δ against k=1.

| Label source | k=1 | k=0.5 Δ | unweighted Δ |
|---|---:|---:|---:|
| Combined (training source) | 0.9027 | −0.0028 | −0.0099 |
| Steven v4 | 0.8882 | −0.0034 | −0.0074 |
| Mean over the nine independent extractors | — | −0.0005 (range −0.0027 to +0.0024) | −0.0050 (range −0.0099 to −0.0011) |

Against the training labeller, both deficits are about twice as large as against independent
extractors. Part of what validation rewards is agreement with the particular labeller that
produced the training targets. This evidence is weak: several extractors share values on some
labels, and every extractor is still report-derived.

## Report-derived and official positive rates

The table compares positive rates at y ≥ 0.5: combined labels on the training folds, official
labels on the 58 gold studies, and Steven v4 on the same 58 studies. It also shows the k=1 weight
ratio.

| Label | Silver, training | Gold (58) | Steven v4 on the 58 | k=1 ratio |
|---|---:|---:|---:|---:|
| ACL | 0.152 | −0.0062 | 0.414 | 0.500 | 5.6 |
| MCL | 0.054 | +0.0083 | 0.155 | 0.276 | 17.4 |
| Lateral Meniscus | 0.158 | +0.0047 | 0.397 | 0.379 | 5.3 |
| Synovitis | 0.130 | −0.0052 | 0.466 | 0.293 | 6.7 |
| Contusion | 0.175 | +0.0043 | 0.328 | 0.483 | 4.7 |
| Fracture | 0.065 | −0.0074 | 0.310 | 0.362 | 14.5 |
| Effusion | 0.598 | +0.0046 | 0.603 | 0.879 | 0.7 |

The gold studies are enriched for disease, so their rates do not estimate the test's prevalence.
Only the comparison between gold and Steven v4 on the same 58 studies is informative. It shows
reports under-calling Synovitis and over-calling Effusion, Contusion, MCL and ACL. The largest
weights fall on exactly the labels whose report positives differ most from official ones.

## Interpretation

- **Label-source coupling (supported, moderate confidence).** The weighting sharpens the
  report-extractor decision boundary on rare labels. Validation shares that labeller and rewards
  it; the official test labels do not. Three results support this: both leaderboard pairs, the
  shrinking deficit under independent extractors, and the gold-versus-report disagreement.
  Prevalence alone cannot explain the effect, because per-label AUC does not depend on
  prevalence.
- **Some weighting may still help (open).** On the leaderboard, k=0.5 (0.945) and k=0.25 (0.944)
  are above unweighted (0.943), k=0.75 (0.942) and k=1 (0.940). The public curve peaks at k=0.5
  and falls on both sides, but all five scores lie within about 0.005, and three seeds of one earlier model already
  spanned 0.003
  ([seed controls](../../../inference/submissions/early-submission-details.md)), so no
  single ordering is established.
- **Not excluded:** leaderboard noise, and interactions with the ×4/τ 0.75 inference setting.

## Conclusions

- **Recipe (user decision, 2026-09-22):** `trainer.class_balance_exponent: 0.5` is the recipe in
  `configs/resnet50.yaml`, and `k_05` is the reference of the sampling campaign.
- **The exponent acts on a few rare labels.** At k=1 the positive:negative ratio is 17.4:1 on MCL,
  14.5:1 on Fracture and 5–7:1 on Synovitis, ACL, Lateral Meniscus and Contusion; the common labels
  stay between 1:1 and 3:1 at any k, and Effusion (60% positive) is weighted slightly toward its
  negatives. Changing k is in effect a dial on those four to six labels.
- **The public cost of strong weighting is gradual, not a threshold.** k=0.75 (MCL 8.5:1, Fracture
  7.4:1) already falls to 0.942; below it the scores stay within 0.943–0.945, inside seed noise.
  Moderate ratios of about 2–4:1 on the rarest labels are where the leaderboard is best.
- **Per-label validation shifts do not follow each label's own weight.** Baker's barely changes
  weight (3:1 at k=1, 1.3:1 at k=0.25) yet moves most on validation (−0.012 at k=0.5, −0.023 at
  k=0.25). With one run per k and shared features, per-label validation cannot tune per-label
  weights.
- **Fold-0 validation cannot select the weighting.** It rises monotonically with k (0.891925 to
  0.902049) while the leaderboard peaks at k=0.5; any further weighting work is judged on the
  leaderboard.

## Implications

- Judge loss-weighting and other label-mimicking changes on the public leaderboard (or on future
  official-label evidence), not on fold-0 validation against report-derived labels alone.
- The exponent curve is complete (S40–S42, S44, S45). A 1→0 schedule and noise-aware down-weighting ran in the
  [noisy-labels campaign](2026-09-23-noisy-labels-campaign.md): neither was adopted on fold 0.
