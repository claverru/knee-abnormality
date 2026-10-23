# Validation and ensemble results

| Read when | Document |
|---|---|
| Deciding whether to learn ensemble weights, a stacker (logistic, TabPFN) or a label correction from the 58 gold studies: leave-pair-out against an equal average, why pooled leave-one-out AUC is biased, and gold agreement of fixed teacher shares in the label table | [2026-10-16 gold-fitted stacking](2026-10-16-gold-fitted-stacking.md) |
| Reading the learned-view and blend submissions of 2026-10-02: the learned-view ResNet-18 alone and three blends of bundles already on Kaggle | [2026-10-02 learned-view and blend submissions](2026-10-02-blend-submissions.md) |
| Checking the 2026-09-30 adoption of L2-SP weight anchoring and removal of the inert scanner confusion head on ResNet-18, and the unseen-site evidence from the sites-out splits that motivated it | [2026-09-30 ResNet-18 L2-SP without confusion head](2026-09-30-rn18-l2sp-no-confusion.md) |
| Checking what an unseen site costs the recipe: the Kaggle sites-out run without five inferred training sites, on hard and soft AUC, and the cost of less data on seen sites | [2026-09-29 sites-out diagnostic](2026-09-29-sites-out-diagnostic.md) |
| Checking which autoresearch verdicts change when every past comparison is re-scored on `rebuild_state_v2`, and which main adoptions lose their local support | [2026-09-28 campaign re-scoring](2026-09-28-campaign-rescoring.md) |
| Reading the ensemble submissions of 2026-09-19: ten-model bundles with mirror TTA and the daily limit that blocked the class-balanced-BCE checkpoint | [2026-09-19 ensemble submissions](2026-09-19-ensemble-submissions.md) |
| Reviewing fold ensembles and the sampler-width sweep | [2026-09-19 fold ensembles and sigma sweep](2026-09-19-fold-ensembles-and-sigma-sweep.md) |
| Comparing local metrics with public scores: which metric, label set, multiplier or grouped AUC tracks the leaderboard, gold-only AUC, label-switch home advantage | [Local-public metric comparison](local-public-metric-comparison/INDEX.md) |
| Reviewing the initial local/public score comparison (superseded by the nineteen-submission analysis) | [2026-09-13 local and public agreement](2026-09-13-local-and-public-agreement-after-the-split-change.md) |
