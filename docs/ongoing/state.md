# Project state (2026-09-21)

The recipe in force, the best submission, and the measurement rules every comparison depends on.
Active lines of development have their own documents in [this folder](INDEX.md). Completed
coverage diagnostics live in [data preparation](../data-preparation/peripheral-coverage.md);
pending experiments are classified in the [backlog](../backlog/INDEX.md).

- **Research backlog review, 2026-09-21:** the user accepted
  [AdamP for ReXNet](../backlog/small-changes.md#adamp-for-rexnet) and
  [plane/contrast-specific feature adapters](../backlog/small-changes.md#planecontrast-specific-feature-adapters)
  as unstarted small changes. Late-stage ReXNet trimming is merged into the existing
  [stride-16 investigation](../backlog/big-bets.md#intermediate-stride-16-feature-fusion), which
  compares the current model, stride-16 truncation and stride-16/stride-32 fusion. The
  terminal-1×1-only proposal was rejected. Implementation and training remain pending.
  Bias-field augmentation, layer-wise
  CoAtNet learning-rate decay, standalone BatchNorm recalibration, ordered within-series
  aggregation and SAM were not selected for the backlog in this review.

- **Literature shortlist accepted, 2026-09-21:**
  [richer report supervision through OA attributes](../backlog/big-bets.md#richer-report-supervision-oa-attributes)
  is an unstarted big bet;
  [attention-token dropout](../backlog/small-changes.md#attention-token-dropout) is an unstarted
  small change, including a random-masking control with series dropout unchanged.
  [Feature-space augmentation (DSU)](../backlog/small-changes.md#feature-space-augmentation-dsu)
  was also accepted as an unstarted small change, with acquisition-coherent perturbations and
  safeguards for diagnostic contrast. No implementation, extraction or training has started.

- **New labels and split, 2026-09-21 (user decision):** training and evaluation use the
  [combined labels](../public-labels/combined-labels.md). These are Steven v4 plus a nine-source
  consensus for ACL, MCL, PF OA and Fracture, scoring 0.9075 against 0.8927 on gold. The split is the
  scanner-stratified [`all_5fold_scanner_v1`](../validation/validation-strategy.md).
  `configs/training.yaml` carries both; the rest of the recipe is unchanged. Fold-0 results before
  this date are not comparable with runs on the new split. The first run on it,
  `fold0-combined-scanner`, is the new fold-0 reference: 0.9020 at ×1 and τ 1, and 0.9079 at ×4 and
  τ 0.75. On the 209 studies both splits hold out, it ties learned strength. τ 0.75 remains best.
  **Its submission S40 scores 0.940 public**, against 0.933 for the last two single fold-0 models
  on Steven v4 labels ([result](../validation/results/2026-09-21-combined-labels-first-run.md)).
  `scripts/adhoc/train_overnight.py` still requires `all_5fold_v3`; the autoresearch gate now
  requires the new split and labels.

- **Hugging Face backbone survey completed, 2026-09-20:** the
  [checkpoint review](../research/huggingface-image-encoders.md) identifies DINOv3 ConvNeXt-T and
  RadioDINO-S as practical candidates, and OrthoFoundation/MRI-CORE/OrthoDiffusion as larger
  MRI investigations. Current spatial mappings prevent config-only swaps; OmniRad also has
  conflicting license metadata. Implementation and training remain unstarted in
  [Big bets](../backlog/big-bets.md#external-pretrained-image-encoders); the recipe is unchanged.
  A parameter/FLOP comparison against the current ReXNet-200 is being added; no training started.

- **Checkpoint cleanup completed, 2026-09-20:** removed 249 obsolete checkpoint files, freeing
  33.412 GiB inside WSL. Current references, useful submission/diagnostic weights and held work
  remain, along with all prediction/configuration/metric records. The
  [retention record](../architecture/configs-and-experiment-artifacts.md#checkpoint-pruning)
  identifies removed weights; no training or campaign state changed.

- **Current research recipe:** learned anatomical family strength with fixed 20 mm width,
  position-rich pooling and class-balanced BCE, seed 2263738815. The selected learned-strength
  run scores 0.897443 versus 0.895112 for the current matched-seed control (+0.002331;
  80% interval [+0.000103, +0.004518]). It is adopted; appearance-only pooling,
  learned widths and the pooled compartment adapter are removed. See the
  [current model](../modeling/current-model.md) and [durable experiment record](../../autoresearch/anatomical-attention.md).
- **Research decisions:** valid candidates are adopted when macro gain >= 0.001 and the
  central 80% paired-report-group interval has lower bound > 0; otherwise discarded.
  The 95% interval remains diagnostic. Candidates inherit the incumbent seed. This is
  a practical progression rule, not independent confirmation; training variability and
  repeated fold-0 selection remain outside the interval. Older noise measurements are
  historical evidence, not a blanket rule that all gains below 0.003 carry no information.
- **Autoresearch:** every campaign now shares one
  [evaluation and adoption rule](../../autoresearch/protocol.md#shared-evaluation-and-adoption):
  fold 0 of `all_5fold_scanner_v1`, combined labels, validation ×1, gain ≥ 0.001 with a positive
  80% lower bound. The three earlier campaigns stay stopped as historical records; they were
  sealed on `all_5fold_v3` and cannot resume. The new
  [sampling campaign](../../autoresearch/campaigns/sampling/INDEX.md) is defined against the
  exponent-0.5 champion `k_05` (0.899042) and not started. Current state and transition evidence are owned by
  [autoresearch](../../autoresearch/status.md).
  The watchdog initializes its incident history on load, so direct `watch` can recover
  without a preceding `start`; existing incident history is preserved.
- **Local metrics against the leaderboard (measured 2026-09-18):** over the nineteen comparable
  submissions macro AUC correlates at +0.907 and BCE at +0.833, gold-only AUC and gold-only BCE
  at nothing, and since 09-13 the submissions differ locally by less than one run's noise, so no
  local metric ranks them. Gold diagnostics decide nothing from here on
  ([comparison](../validation/results/2026-09-18-local-public-metric-comparison.md)).
- **Validation at ×1** since 2026-09-17 (user decision): fold-0 scores from then on sit about
  0.0035 below the earlier ×2 numbers and are comparable only with a ×1 control. Submissions are
  unaffected; the bundle's test multiplier is separate.
- **Historical loading reference, 2026-09-18:** `control-check` fold 0, best epoch 6, validation
  macro AUC **0.88477**, preprocessing identity `b8b6c639`. It is the first checkpoint carrying the loading
  changes, it sits inside the ×1 cohort spread (median 0.88707, sd 0.00331, and the cohort's own two
  controls are 0.88380 and 0.88645), and `outputs/submission/exports/control-check-x4-tau075` is
  exported from it at ×4 and temperature 0.75.
- **Best submission 2026-09-19: 0.943 public**, a ten-member bundle of five EfficientNet-B0 and
  five ReXNet-200 fold models with mirror TTA, at ×4 and temperature 0.75. The five-b0 bundle
  without TTA scores 0.940 at 27% of the runtime and is the efficiency entry
  ([submissions](../inference/submissions/2026-09-19-ensemble-submissions.md)). Every member
  trains with ordinary BCE. Keep attention temperature 0.75 in later submission bundles; its
  [inference gain](../inference/submissions/2026-09-18-temperature-and-count-submissions.md) is already adopted.
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
- **EfficientNet-B0 is an open efficiency line:** its ordinary-BCE fold models already form the
  0.940 efficiency entry. Re-measure the backbone tradeoff under balanced BCE before changing
  the next recipe ([EfficientNet backbones](efficientnet-backbones.md)); the retained fifteen-epoch
  B0 and ten-epoch ReXNet recipes have similar training wall time.
- **Dropped:** learned crop selection ([record](../history/crop-selection-investigation.md)).
- **AUC training research, 2026-09-19:** the
  [objective survey](../research/auc_loss_approaches.md) now includes BCE weighting, study sampling,
  batch composition and adaptive allocation. Static candidates are in
  [small changes](../backlog/small-changes.md#bce-weighting-and-study-sampling), adaptive work in
  [big bets](../backlog/big-bets.md#adaptive-training-allocation).
  [Fixed class-balanced BCE](../modeling/supervision.md#fixed-class-balanced-bce) is implemented
  as `trainer.class_balanced_bce: false` by default, with saved training-fold weights and strict
  resume checks. The damping exponent is implemented as `trainer.class_balance_exponent` (default
  1.0). On 2026-09-21–22 [the exponent curve](../validation/results/2026-09-21-class-balance-exponent.md)
  scored unweighted 0.943, k=0.25 0.944, k=0.5 0.945, k=0.75 0.942 and k=1 0.940 on the public
  leaderboard, peaking at k=0.5, while fold-0 validation rose monotonically with k. The user completed [its comparison](../validation/results/2026-09-19-balanced-bce-result.md)
  as `balanced-bce`, seed 42, fold 0: best 0.895282, and +0.01161 against `rexnet-fold0`, the
  same-sampler fold-0 run from the same day, with eleven of twelve labels improving and the gains
  concentrated on the rarest ones. Historical source differences limit exact attribution.
  **`configs/training.yaml` now enables it**, so it is the recipe in force; no public result yet,
  because the day's submission limit was reached with the bundle exported and ready.
  **Replicated 2026-09-19:** the untried-hyperparameter queue's control reached 0.894037 at a
  different seed, 0.00125 from `balanced-bce` and about 0.008 above every ordinary-BCE run on the
  same sampler ([queue results](../validation/results/2026-09-19-untried-hyperparameter-queue.md)).
- **No candidate was adopted from the 2026-09-20 run comparisons.** The
  [fourteen-run queue](../validation/results/2026-09-20-overnight-hyperparameter-results.md)
  completed in 6 h 8 m with two controls spanning 0.003360 and no candidate clearing them; milder
  scaling `[0.97, 1.03]` was the best arm at 0.895868 and inconclusive against both, and stronger
  contrast was the worst at 0.889906. The
  [run-level stratified sampler](../validation/results/2026-09-20-stratified-sampling-promotion.md)
  was promoted, reverted and rejected for adoption: its second run scored 0.895177 at seed 42
  against 0.897515 for the original, −0.002338 with a paired interval of [−0.006064, +0.001371],
  and it sits between the two controls. Training again uses independent per-epoch draws within
  Gaussian probability bands. The autoresearch campaign remains stopped.
- **The sampler's shape was tried and did not resolve, 2026-09-19.** The campaign trained both
  shape arms on fold 0: per-plane widths (`geometry-informed-plane-widths`) at +0.00041
  [−0.0033, +0.0041] and the asymmetric coronal draw (`coronal-side-relative-draw`) at −0.00130
  [−0.0051, +0.0023], both inconclusive against the 0.894037 control. With the overall scale
  already settled, the sampler has no open training-side lead left except the mirrored
  importance-map re-export ([peripheral coverage](../data-preparation/peripheral-coverage.md#what-is-settled-and-what-is-not)).
- **No confirmed duplicate studies, 2026-09-20.** The bounded
  [study-overlap audit](../metadata/study-overlap-investigation.md) found 885 near-report candidate
  pairs, but forty shortlisted pairs shared no identical slice across 1,202 decoded images. Fold
  membership is unchanged; anatomical matching is
  [parked](../backlog/big-bets.md#study-overlap-and-anatomical-matching).
- **Per-label loss research completed, 2026-09-20:** the
  [research note](../research/per-label-losses.md) develops the trusted ACL-ranking/MCL-focal
  observation into a shared-feature classifier comparison, per-label selection, joint
  continuation and learned mixtures. Implementation and experiments remain unstarted in
  [Big bets](../backlog/big-bets.md#per-label-loss-selection-and-learned-mixtures).
- **Research-agenda review completed, 2026-09-20:** the retained unstarted priorities are
  [per-label attention temperatures](../backlog/small-changes.md#per-label-attention-temperatures),
  [OA-only label repair](../backlog/big-bets.md#oa-only-rubric-aligned-label-repair) and
  [plane/contrast-specific readouts](../backlog/small-changes.md#planecontrast-specific-readouts).
  [Balance damping](../backlog/small-changes.md#weighting-strength-exponent-and-prevalence-definition)
  is now implemented as `trainer.class_balance_exponent` (default 1.0) and
  [measured](../validation/results/2026-09-21-class-balance-exponent.md). Per-label loss selection and
  canonical knee orientation remain deferred. This records backlog priorities; no experiment was
  started by the review.
- **Report-state audit completed, 2026-09-20:** the
  [evidence](../public-labels/uncertainty-and-confidence-encoding.md#report-state-audit-and-implications-for-supervision)
  motivates auxiliary reporting supervision while retaining current disease targets.
  The [proposal](../backlog/small-changes.md#auxiliary-report-state-prediction) is unimplemented
  and untested; no training recipe change was adopted.
- **Strategy:** iterate on fold 0. Folds, ensembling and mirror TTA were executed on 2026-09-19
  and are no longer waiting; bigger images and weight averaging still wait
  for the [final phase](../backlog/endgame.md)
  ([validation strategy](../validation/validation-strategy.md)). Test-time risks are listed in
  [risks](../backlog/risks.md).

- **Preprocessing identity bumped 2026-09-20, `f9d05ad2` → `65a6123d`:** the host repaired the two
  truncated training DICOMs, both were re-fetched into `data/raw`, and `exclusions.py` with its two
  call sites is deleted. Two slices return to training out of 819,078, so nothing measurable
  changes; the identity bump is the cost, and it applies to fresh runs only, alongside the
  sigma-fraction and triplet-total bumps already in force
  ([data contract](../data-preparation/public-data-contract-and-decoding.md)). Already-exported
  bundles carry their own sources and still submit, but a **new** export or local prediction from a
  checkpoint trained before this change is refused by `validate_preprocessing_identity` and needs
  that checkpoint's archived sources.
