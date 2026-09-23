# Experiment results

| Read when | Document |
|---|---|
| Scoring series-cleaning rules at inference only: contrast-slot relabel, keep one series per same-signature group (retakes, Philips second reconstructions), near-empty frames | [2026-09-22 series-cleaning rules](2026-09-22-series-cleaning-rules.md) |
| Reviewing stride-16/stride-32 fusion (token sum, deep-branch dropout, per-scale heads): per-label AUC against `k_05`, fine/deep branch ablation, pair-ensemble ceiling | [2026-09-22 stride-16 fusion](2026-09-22-stride16-fusion-result.md) |
| Reviewing ReXNet stride-16 truncation (14×14 tokens): per-label AUC against `k_05`, runtime, memory, parameters | [2026-09-22 stride-16 truncation](2026-09-22-stride16-truncation-result.md) |
| Reviewing oblique series with their own slot budget (`oblique_tilt_deg`): fold-0 and per-label localization, ×4/×5 τ sweep, S46–S47 | [2026-09-22 oblique groups](2026-09-22-oblique-groups-result.md) |
| Choosing class-weighting strength; why fold-0 validation and the public leaderboard disagree on weighting (k=1, 0.5, unweighted; S40–S42) | [2026-09-21 class-balance exponent](2026-09-21-class-balance-exponent.md) |
| Reviewing the first run on combined labels and the scanner split: per-label AUC, the tie with learned strength on shared studies, attention-temperature sweep | [2026-09-21 combined-labels first run](2026-09-21-combined-labels-first-run.md) |
| Choosing loss variants or seeds for an ensemble; loss-campaign subset search and selection optimism | [2026-09-21 loss ensemble simulation](2026-09-21-loss-ensemble-simulation.md) |
| Reviewing the shared meniscus compartment adapter | [2026-09-20 compartment adapter](2026-09-20-compartment-adapter-result.md) |
| Reviewing stratified sampling and its replication | [2026-09-20 stratified sampling experiment](2026-09-20-stratified-sampling-promotion.md) |
| Comparing class-balanced-BCE augmentation and optimizer runs against two controls | [2026-09-20 overnight results](2026-09-20-overnight-hyperparameter-results.md) |
| Comparing first-use optimizer and augmentation values on class-balanced BCE | [2026-09-19 untried-hyperparameter queue](2026-09-19-untried-hyperparameter-queue.md) |
| Reviewing fold ensembles and the sampler-width sweep | [2026-09-19 fold ensembles and sigma sweep](2026-09-19-fold-ensembles-and-sigma-sweep.md) |
| Reviewing fixed class-balanced BCE | [2026-09-19 balanced BCE](2026-09-19-balanced-bce-result.md) |
| Reviewing relative sampler width | [2026-09-19 relative sampler width](2026-09-19-relative-sampler-width-result.md) |
| Reviewing per-plane sampler widths on cached tokens | [2026-09-18 per-plane sigma on the cache](2026-09-18-per-plane-sigma-cache.md) |
| Comparing EfficientNet backbones | [2026-09-18 EfficientNet backbone results](2026-09-18-efficientnet-backbone-results.md) |
| Comparing EfficientNet-B0 batch-size and learning-rate queues | [2026-09-18 b0 hyperparameter queues](2026-09-18-b0-hyperparameter-queues.md) |
| Comparing local metrics with public scores | [2026-09-18 local-public metric comparison](2026-09-18-local-public-metric-comparison.md) |
| Reviewing count-normalized attention | [2026-09-17 count-normalised attention](2026-09-17-count-normalized-attention-result.md) |
| Comparing validation-x1 controls and unseeded sampler and hyperparameter runs | [2026-09-18 overnight results](2026-09-18-overnight-results.md) |
| Reviewing attention temperature and bag size | [2026-09-17 attention temperature](2026-09-17-attention-temperature.md) |
| Reviewing random per-type triplet allocation | [2026-09-17 random per-type allocation](2026-09-17-allocation-concentration-result.md) |
| Reviewing test-time triplet allocation | [2026-09-17 test-time allocation](2026-09-17-test-time-allocation.md) |
| Reviewing per-type logit contributions | [2026-09-17 per-type contribution](2026-09-17-type-contribution.md) |
| Reviewing series filtering and deterministic source ranking | [2026-09-17 ranked-filtered result](2026-09-17-ranked-filtered-result.md) |
| Reviewing plane and contrast importance | [2026-09-17 plane/contrast importance](2026-09-17-plane-contrast-importance.md) |
| Comparing EMA, SWA, control seeds and one-change hyperparameter runs | [2026-09-17 overnight results](2026-09-17-overnight-results.md) |
| Reviewing restored-code hyperparameter runs | [2026-09-16 restored-code hyperparameter runs](2026-09-16-restored-code-hyperparameter-runs.md) |
| Reviewing learned zoom | [2026-09-16 learned-zoom result](2026-09-16-learned-zoom-result.md) |
| Reviewing shared crops | [2026-09-16 shared crops result](2026-09-16-shared-crops-result.md) |
| Reviewing label-private crops | [2026-09-16 label-private crops result](2026-09-16-label-private-crops-result.md) |
| Reviewing per-label crop selection | [2026-09-16 per-label crops result](2026-09-16-per-label-crops-result.md) |
| Reviewing fixed-size crops | [2026-09-16 crop224 result](2026-09-16-crop224-result.md) |
| Reviewing two crops per canvas | [2026-09-15 two-crops result](2026-09-15-two-crops-result.md) |
| Reviewing the scout crop-selection experiments | [2026-09-15 crop-selection result](2026-09-15-crop-selection-result.md) |
| Reviewing the attention prior | [2026-09-15 attention-prior result](2026-09-15-attention-prior-result.md) |
| Reviewing the September 15 overnight queue | [2026-09-15 overnight results](2026-09-15-overnight-results.md) |
| Reviewing the token graph | [2026-09-14 token-graph result](2026-09-14-token-graph-result.md) |
| Reviewing knee-side embeddings | [2026-09-14 knee-side embedding result](2026-09-14-knee-side-embedding-result.md) |
| Reviewing meniscus soft coupling | [2026-09-14 meniscus soft-coupling result](2026-09-14-meniscus-soft-coupling-result.md) |
| Reviewing reflection and sigma-40 sampling | [2026-09-14 reflection sigma-40 result](2026-09-14-reflection-sigma-40-result.md) |
| Comparing reflection with physical sampling | [2026-09-13 reflection validation comparison](2026-09-13-reflection-validation-comparison.md) |
| Comparing physical sampling with spatial tokens | [2026-09-13 3dsampling validation comparison](2026-09-13-3dsampling-validation-comparison.md) |
| Reviewing spatial-token validation | [2026-09-13 spatial-token validation comparison](2026-09-13-spatial-token-validation-comparison.md) |
| Comparing Tonylin and Steven labels | [2026-09-13 Tonylin label comparison](2026-09-13-tonylin-label-comparison.md) |
| Diagnosing per-plane fusion usage | [2026-09-13 per-plane fusion usage diagnosis](2026-09-13-per-plane-fusion-usage-diagnosis.md) |
| Comparing max and average pooling | [2026-09-13 max-pooling validation comparison](2026-09-13-max-pooling-validation-comparison.md) |
| Reviewing the initial local/public score comparison | [2026-09-13 local and public agreement](2026-09-13-local-and-public-agreement-after-the-split-change.md) |
| Comparing spatial fusion with crop140mm | [2026-09-12 spatial-fusion validation comparison](2026-09-12-spatial-fusion-validation-comparison.md) |
| Reviewing spatial-fusion runtime and memory | [2026-09-12 spatial-fusion engineering result](2026-09-12-spatial-fusion-engineering-result.md) |
| Reviewing geometric sampling centers | [2026-09-12 accurate-centers result](2026-09-12-accurate-centers-result.md) |
| Reviewing the crop140mm model | [2026-09-12 140-mm crop result](2026-09-12-140-mm-crop-result.md) |
| Reviewing the September 12 overnight queue | [2026-09-12 overnight results](2026-09-12-overnight-results.md) |
