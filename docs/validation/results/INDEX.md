# Experiment results

| Read when | Document |
|---|---|
| Reviewing learned zoom (context on every slice plus learned detail glimpses), the last run of the dropped crop-selection investigation: unused detail, a 0.022 regression against per-label, and the resolution pattern | [2026-09-16 learned-zoom result](2026-09-16-learned-zoom-result.md) |
| Reviewing shared crops (2 per label, shared pooling): a 0.012 regression against per-label explained by slice coverage, with inference probes | [2026-09-16 shared crops result](2026-09-16-shared-crops-result.md) |
| Reviewing label-private crops (2 per label, private gated pooling, label FiLM): a 0.047 regression with study-specific placement and inference probes | [2026-09-16 label-private crops result](2026-09-16-label-private-crops-result.md) |
| Reviewing per-label round-robin crop selection: a macro tie that recovered peripheral targets at the cost of effusion and contusion, with placement measurements | [2026-09-16 per-label crops result](2026-09-16-per-label-crops-result.md) |
| Reviewing the 224-pixel (112 mm) crop run: a tie with the full-triplet baseline, with central targets gaining and peripheral targets losing | [2026-09-16 crop224 result](2026-09-16-crop224-result.md) |
| Reviewing two crops per canvas: per-target gains and losses against the champion, their complementarity and the draft-uncertainty check | [2026-09-15 two-crops result](2026-09-15-two-crops-result.md) |
| Reviewing the first scout crop-selection run, its 15-epoch extension and the learned-versus-random-versus-center placement ablation | [2026-09-15 crop-selection result](2026-09-15-crop-selection-result.md) |
| Reviewing the per-target 1D/2D/3D attention prior's regression against the graph-off control | [2026-09-15 attention-prior result](2026-09-15-attention-prior-result.md) |
| Reviewing the graph-off, no-3D hyperparameter runs 00–06, including the restored `00-baseline` control and `03-min-lr-0.1`, and the crop130 rerun | [2026-09-15 overnight results](2026-09-15-overnight-results.md) |
| Reviewing the retired token graph's tie, its interventions and the unused forced-training follow-up | [2026-09-14 token-graph result](2026-09-14-token-graph-result.md) |
| Reviewing the retired knee-side embedding and its side interventions | [2026-09-14 knee-side embedding result](2026-09-14-knee-side-embedding-result.md) |
| Reviewing the meniscus soft-coupling regression and subgroup diagnostics | [2026-09-14 meniscus soft-coupling result](2026-09-14-meniscus-soft-coupling-result.md) |
| Reviewing the reflection/sigma-40 champion (0.936 public) against both 1C controls | [2026-09-14 reflection sigma-40 result](2026-09-14-reflection-sigma-40-result.md) |
| Comparing spatial-token reflection with 3dsampling | [2026-09-13 reflection validation comparison](2026-09-13-reflection-validation-comparison.md) |
| Comparing 3dsampling with the five-embedding spatial-token run | [2026-09-13 3dsampling validation comparison](2026-09-13-3dsampling-validation-comparison.md) |
| Reviewing the five-embedding spatial-token run, target gains and uncertainty | [2026-09-13 spatial-token validation comparison](2026-09-13-spatial-token-validation-comparison.md) |
| Reviewing why Steven v4 labels were retained over Tonylin labels | [2026-09-13 Tonylin label comparison](2026-09-13-tonylin-label-comparison.md) |
| Understanding how little attention per-plane regional fusion received | [2026-09-13 per-plane fusion usage diagnosis](2026-09-13-per-plane-fusion-usage-diagnosis.md) |
| Comparing max pooling with average-pooling spatial fusion | [2026-09-13 max-pooling validation comparison](2026-09-13-max-pooling-validation-comparison.md) |
| Checking how fold-0 AUC tracked the public leaderboard after the split change | [2026-09-13 local and public agreement](2026-09-13-local-and-public-agreement-after-the-split-change.md) |
| Reviewing why 27-query spatial fusion did not beat crop140mm | [2026-09-12 spatial-fusion validation comparison](2026-09-12-spatial-fusion-validation-comparison.md) |
| Reviewing the rejected spatial-fusion engineering and runtime gate | [2026-09-12 spatial-fusion engineering result](2026-09-12-spatial-fusion-engineering-result.md) |
| Reviewing the rejected geometric-center sampler and the frozen fusion control | [2026-09-12 accurate-centers result](2026-09-12-accurate-centers-result.md) |
| Reviewing the crop140mm model's 0.925 public score | [2026-09-12 140-mm crop result](2026-09-12-140-mm-crop-result.md) |
| Reviewing the seven-run learning-rate, decay and duration queue that selected 04-higher-lr | [2026-09-12 overnight results](2026-09-12-overnight-results.md) |
