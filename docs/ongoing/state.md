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
- **Current checkpoint, 2026-09-18:** `control-check` fold 0, best epoch 6, validation macro AUC
  **0.88477**, preprocessing identity `b8b6c639`. It is the first checkpoint carrying the loading
  changes, it sits inside the ×1 cohort spread (median 0.88707, sd 0.00331, and the cohort's own two
  controls are 0.88380 and 0.88645), and `outputs/submission/exports/control-check-x4-tau075` is
  exported from it at ×4 and temperature 0.75.
- **Best submission 2026-09-19: 0.943 public**, a ten-member bundle of five EfficientNet-B0 and
  five ReXNet-200 fold models with mirror TTA, at ×4 and temperature 0.75. The five-b0 bundle
  without TTA scores 0.940 at 27% of the runtime and is the efficiency entry
  ([submissions](../inference/submissions/2026-09-19-ensemble-submissions.md)). Every member
  trains with ordinary BCE.
- **Adopted 2026-09-19:** `data.sampling_sigma_fraction: 0.867` replaces the fixed
  `data.sampling_sigma_mm: 40`. Each candidate pool's Gaussian width is now a fraction of its own
  limiting reach, giving realized widths of axial 45.1 mm, coronal 35.9 and sagittal 39.0 at the
  same mean reach. Fold 0 put it at 0.88649, between its two total-26 controls (0.88477 and
  0.88848) and +0.00016 at ×4, so it is kept on being free rather than on a measured gain; the
  effect it does have is on the training side, not the draw
  ([result](../validation/results/2026-09-19-relative-sampler-width-result.md)). Fresh runs only;
  it changes the preprocessing identity, so `control-check`'s `b8b6c639` no longer applies.
- **Adopted 2026-09-18:** `data.study_triplet_total: 26`, a floor on a study's training triplets
  that scales the types it has in proportion to their budgets. Fold 0 put it at +0.0033 over the
  control mean, inside the control spread, so the case is structural: it only ever adds triplets,
  for 5.5% more compute, and the studies missing two types stop being under-sampled. Fresh runs
  only; it changes the preprocessing identity.
- **EfficientNet-B0 is an open efficiency line**, not a replacement backbone today: −0.0032 from
  ReXNet-200 at a third of the training time and half the memory
  ([EfficientNet backbones](efficientnet-backbones.md)).
- **Dropped:** learned crop selection ([record](../history/crop-selection-investigation.md)).
- **AUC training research, 2026-09-19:** the
  [objective survey](../research/auc_loss_approaches.md) now includes BCE weighting, study sampling,
  batch composition and adaptive allocation. Static candidates are in
  [small changes](../backlog/small-changes.md#bce-weighting-and-study-sampling), adaptive work in
  [big bets](../backlog/big-bets.md#adaptive-training-allocation).
  [Fixed class-balanced BCE](../modeling/supervision.md#fixed-class-balanced-bce) is implemented
  as `trainer.class_balanced_bce: false` by default, with saved training-fold weights and strict
  resume checks. The user completed [its comparison](../validation/results/2026-09-19-balanced-bce-result.md)
  as `balanced-bce`, seed 42, fold 0: best 0.895282, and +0.01161 against `rexnet-fold0`, the
  same-sampler fold-0 run from the same day, with eleven of twelve labels improving and the gains
  concentrated on the rarest ones. Historical source differences limit exact attribution.
  **`configs/training.yaml` now enables it**, so it is the recipe in force; no public result yet,
  because the day's submission limit was reached with the bundle exported and ready.
  **Replicated 2026-09-19:** the untried-hyperparameter queue's control reached 0.894037 at a
  different seed, 0.00125 from `balanced-bce` and about 0.008 above every ordinary-BCE run on the
  same sampler ([queue results](../validation/results/2026-09-19-untried-hyperparameter-queue.md)).
- **Strategy:** iterate on fold 0. Folds, ensembling and mirror TTA were executed on 2026-09-19
  and are no longer waiting; bigger images and weight averaging still wait
  for the [final phase](../backlog/endgame.md)
  ([validation strategy](../validation/validation-strategy.md)). Test-time risks are listed in
  [risks](../backlog/risks.md).
