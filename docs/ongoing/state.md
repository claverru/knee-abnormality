# Project state (2026-09-18)

The recipe in force, the best submission, and the measurement rules every comparison depends on.
Active lines of development have their own documents in [this folder](INDEX.md).

- **Code:** the full-slice recipe ([current model](../modeling/current-model.md)) with series
  filtering and deterministic source ranking (preprocessing identity
  `ranked_filtered_physical_gaussian_overlap_spatial_tokens_reflection_v10`,
  [preparation](../data-preparation/current-preparation.md#series-indexing-guards-and-exclusions)).
  Its seeds 42–44 scored 0.889573, 0.889650 and 0.892795 on fold 0 (mean 0.890673), a tie with
  the previous code's seed controls, and seed 42 scored 0.933 public
  ([result](../validation/results/2026-09-17-ranked-filtered-result.md)).
- **Best submission:** `ranked-filtered-seed42` with attention temperature 0.75 in the weights and
  ×4 evaluation images, **0.939 public** (×3 scored 0.938 twice), against 0.933 for the same
  checkpoint at temperature 1.0 and ×2, and 0.936 for the previous reflection/sigma-40 champion
  ([submissions](../inference/submissions/2026-09-18-temperature-and-count-submissions.md)). The
  temperature is free at inference and recipe-independent, so every later submission should carry
  it.
- **Noise:** run-to-run noise comes from training (pooled standard deviation 0.00175 over eight
  replicate runs); a single-run difference below about 0.003 carries no information, even at a
  matched seed. The 2026-09-18 controls span 0.0026 at ×1 validation
  ([overnight results](../validation/results/2026-09-18-overnight-results.md)).
- **Local metrics against the leaderboard (measured 2026-09-18):** over the nineteen comparable
  submissions macro AUC correlates at +0.907 and BCE at +0.833, gold-only AUC and gold-only BCE
  at nothing, and since 09-13 the submissions differ locally by less than one run's noise, so no
  local metric ranks them. Gold diagnostics decide nothing from here on
  ([comparison](../validation/results/2026-09-18-local-public-metric-comparison.md)).
- **Validation at ×1** since 2026-09-17 (user decision): fold-0 scores from then on sit about
  0.0035 below the earlier ×2 numbers and are comparable only with a ×1 control. Submissions are
  unaffected; the bundle's test multiplier is separate.
- **Adopted 2026-09-18:** `data.study_triplet_total: 26`, a floor on a study's training triplets
  that scales the types it has in proportion to their budgets. Fold 0 put it at +0.0033 over the
  control mean, inside the control spread, so the case is structural: it only ever adds triplets,
  for 5.5% more compute, and the studies missing two types stop being under-sampled. Fresh runs
  only; it changes the preprocessing identity.
- **EfficientNet-B0 is an open efficiency line**, not a replacement backbone today: −0.0032 from
  ReXNet-200 at a third of the training time and half the memory
  ([EfficientNet backbones](efficientnet-backbones.md)).
- **Dropped:** learned crop selection ([record](../history/crop-selection-investigation.md)).
- **Strategy:** iterate on fold 0; folds, ensembling, bigger images, TTA and weight averaging wait
  for the [final phase](../backlog/endgame.md)
  ([validation strategy](../validation/validation-strategy.md)). Test-time risks are listed in
  [risks](../backlog/risks.md).
