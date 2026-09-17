# Experiment results

| Read when | Document |
|---|---|
| Reviewing the rejected count-normalised attention run (with random per-type allocation): a 0.0035 loss against the same recipe without it, worst on the menisci | [2026-09-17 count-normalised attention](2026-09-17-count-normalized-attention-result.md) |
| Reviewing the attention temperature measured on frozen checkpoints (+0.0024 at tau 0.75, inference only) and the bag-size question: a real plateau, a spread penalty only at very large bags | [2026-09-17 attention temperature](2026-09-17-attention-temperature.md) |
| Reviewing the random per-type allocation run (`allocation-concentration`, unseeded): a macro tie with the fixed budgets, per-target and family deltas, and what it unblocks | [2026-09-17 random per-type allocation](2026-09-17-allocation-concentration-result.md) |
| Reviewing test-time triplet allocation on the fold-0 token cache: value curves, Neyman, search-theory and proportional rules that failed to confirm, missing-type substitution, repeated-series value and depth profiles | [2026-09-17 test-time allocation](2026-09-17-test-time-allocation.md) |
| Reviewing exact per-type logit contributions (separation, variance and attention shares, per-token efficiency) on the ranked-filtered checkpoints, per label and availability group | [2026-09-17 per-type contribution](2026-09-17-type-contribution.md) |
| Reviewing the series filtering and deterministic source ranking runs (seeds 42–44, mean 0.8907, public 0.933): a tie with the seed controls, prediction agreement between runs, and why run-to-run noise comes from training | [2026-09-17 ranked-filtered result](2026-09-17-ranked-filtered-result.md) |
| Reviewing per-type removal and attention-share evidence, proposed per-type budgets and availability-driven prevalence shifts | [2026-09-17 plane/contrast importance](2026-09-17-plane-contrast-importance.md) |
| Reviewing the 2026-09-17 overnight queue: three control seeds (0.0032 spread), EMA/SWA gains, affine-off and 20-epoch losses, and ties for the other hyperparameters | [2026-09-17 overnight results](2026-09-17-overnight-results.md) |
| Reviewing the user-launched runs on the restored code (min_lr 0.1, series dropout 0.25/0.3, fewer slices): ties, a speed–accuracy trade-off and run-to-run noise | [2026-09-16 restored-code hyperparameter runs](2026-09-16-restored-code-hyperparameter-runs.md) |
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
