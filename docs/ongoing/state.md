# Project state (2026-09-22)

The recipe in force, the best submission, and the measurement rules every comparison depends on.
Active lines of development have their own documents in [this folder](INDEX.md). Completed
coverage diagnostics live in [data preparation](../data-preparation/peripheral-coverage.md);
pending experiments are classified in the [backlog](../backlog/INDEX.md).

- **Near-miss review, 2026-09-24 (user decisions):** early-learning regularization scored
  0.943 public (S57) against 0.942 for its paired control S48; `trainer.max_epochs: 12` queued in the [overnight queue](../backlog/overnight.md);
  loss variants that gain on soft AUC parked on a
  [watch list](../validation/results/2026-09-23-validation-auc-variants.md#losses-campaign-rescored-2026-09-24).
  Dropped: combining the adopted model with image-blend labels, and inference counts above ×4.
  Rescoring the knee-geometry candidates against `k05-clean` puts 16 of 17 above it; the
  incumbent's lead over them sits mostly in Fracture, Baker's and Contusion, labels a re-seed also
  moves ([findings](../../autoresearch/campaigns/knee-geometry/findings.md)).

- **Scanner confusion head adopted, 2026-09-27 (user decision):** the training-only scanner
  confusion head from the [auxiliary-signals campaign](../../autoresearch/campaigns/auxiliary-signals/INDEX.md)
  ([design](../modeling/scanner-confusion-head.md)) scored ×4 0.904442 against `half-stack-control`
  (−0.00076, 80% [−0.0027, +0.0011]) and 0.948 public (S67) against 0.947 (S68) on fold 3, one
  rounding step. Heavier site-debiasing variants lost publicly (S70 0.944, S71 0.943), the per-site
  bias tied (S69 0.947) and the reliability filter lost (S72 0.943 against 0.947)
  ([bias_free.md](../../bias_free.md), [gold versus silver labels](../public-labels/gold-versus-silver.md)). The new
  control, `scanner-confusion-control` (seed 703791404, fold 4), scored ×4 0.909733 (epoch 9);
  candidates pair with it. Fold 4 scores are not comparable with fold 3 scores.

- **Half-stack augmentation adopted, 2026-09-26 (user decision):** the
  [augmentation campaign](../validation/results/2026-09-26-augmentation-campaign.md) adopted
  bilateral smoothing and DSU; on top of them the user adopted intra-study contrast mixing, GIN and
  one-axis resolution degradation. On fold 2 at seed 1704794368 the half stack scored +0.0011 ×4
  macro against smoothing + DSU and 0.943 public (S64) against 0.940 (S63); the full stack (plus
  channel reversal and stochastic depth) added only +0.0005 locally and was not adopted. The new
  control, `half-stack-control` (seed 2617320043, fold 3), scored ×4 0.905205 (epoch 9); candidates
  pair with it. S65 (the half stack's epoch 9) also scored 0.943, a second tie between hard-AUC and
  last-epoch checkpoints, so selection stays on hard AUC. The augmentation campaign is closed, and campaigns no longer retrain
  controls after adoptions.

- **Report-attribute heads adopted, 2026-09-25 (user decision):** 35 training-only heads learn
  JevK5 report attributes (target set 3.1) at weight 0.1. Against `oa-detector-control` they scored
  +0.00105 ×4 macro (95% [−0.0012, +0.0035]) and 0.940 public (S62) against 0.938 (S60); the first
  version on raw v3 targets tied at 0.938 (S61)
  ([result](../validation/results/2026-09-25-report-attribute-heads.md),
  [design](../modeling/report-attribute-heads.md)). The new control, `report-attributes-control`
  (seed 2464012069, fold 2), scored ×4 0.898663 (best epoch 7); candidates pair with it. Fold 2
  scores are not comparable with fold 1 scores.

- **Three-compartment OA detector adopted, 2026-09-25 (user decision):** it narrowly missed the
  local rule in the [near-misses campaign](../validation/results/2026-09-25-near-misses-campaign.md)
  (+0.0013), but scored 0.944 public (S58) against 0.940 for its same-seed reference (S55). Its
  code is merged into the recipe with fresh seed 1485198267 on fold 1; the control scored ×4 τ 1
  0.900568 and 0.938 public (S60). The campaign's adopted token dropout scored 0.939 (S59) and
  masked only 0.3% of tokens; a 5% uncapped retest on the new control lost at every τ
  ([follow-up](../validation/results/2026-09-25-near-misses-campaign.md#follow-up-on-main)).

- **Knee-geometry model adopted, 2026-09-24:** the [knee-geometry campaign](../../autoresearch/campaigns/knee-geometry/INDEX.md)
  adopted `meniscus-half-knee-mil`: the medial-lateral midline at the fitted study centre, a
  meniscus-only own-side attention mask and a training-only half-knee MIL loss. Fold 0 ×4
  0.905834 against 0.902760 for `k05-clean`, +0.00307 (80% [+0.00152, +0.00476]); public 0.944
  (S54) against 0.942 (S48). It is now the recipe in `configs/training.yaml`
  ([current model](../modeling/current-model.md)). The campaign stopped after 18 decisions with no
  second adoption.

- **New control with a fresh seed, 2026-09-24:** the adopted model plus the robust study centre and
  per-knee geometry for bilateral studies, retrained at seed 638120755 as `knee-geometry-merged-2`:
  ×4 0.900955, public 0.940 (S55). Each adoption now draws a fresh seed
  ([rule](../validation/validation-strategy.md#controlled-comparisons)); the seed alone moved the
  adopted model by −0.0063 on fold 0.

- **Soft macro AUC logged, 2026-09-23:** validation now also records `val_soft_macro_auc`, a
  pairwise AUC weighted by the soft targets. It ranked 33 past submissions closer to the public
  leaderboard than hard macro AUC (168 vs 140 of 213 pairs), and it is used to compare runs and
  choose submissions. Checkpoint selection stays on `val_macro_auc`: the one paired test, S49
  (`k_05` epoch 9, 0.944) against S41 (epoch 7, 0.945), is inconclusive
  ([result](../validation/results/2026-09-23-validation-auc-variants.md)).

- **Noisy-labels line closed, 2026-09-23 (user decision):** six prediction-driven label-noise
  objectives against `k05-clean` (S48), none adopted; the best was early-learning regularization
  at +0.0013 hard. The model does not memorize its labels in ten fine-tuning epochs, so its own
  predictions add no teacher; only an independent model does. The 1→0 class-balance schedule tied
  locally and scored 0.941 public (S53) against 0.942 for S48; early-learning regularization
  awaits a public submission
  ([result](../validation/results/2026-09-23-noisy-labels-campaign.md)). The adaptive-allocation
  big bet is retired with it.

- **Oblique line closed, 2026-09-23 (user decision):** oblique groups (own slot budget for tilted
  axial-fluid and coronal series) and the group embedding are removed from `configs/training.yaml`
  and the code. Three paired submissions came out level or lower with them: S46 0.944 against S41
  0.945, S51 0.942 against S48 0.942, and S52 0.943 against S50 0.945. At the ×4 submission
  setting the local gain was at most +0.0003 hard. The three oblique campaigns are retired, and
  none adopted anything. The recipe's preprocessing identity again matches the `image-blend` run
  ([history](../history/oblique-groups.md)). The sampling campaign, stopped overnight by a
  transient provider error after six rejected experiments, remains retired.

- **Stride-16 feature fusion closed, 2026-09-22:** truncation lost 0.0042 fold-0 macro AUC against
  `k_05`; token-sum fusion (+0.0011), deep-branch dropout (−0.0027) and per-scale heads (+0.0006)
  did not clear +0.002. Stride-16 features trade labels rather than add signal; no code was kept
  ([history](../history/stride16-feature-fusion.md)).

- **Research backlog review, 2026-09-21:** the user accepted
  [AdamP for ReXNet](../backlog/small-changes.md#adamp-for-rexnet) and
  [plane/contrast-specific feature adapters](../backlog/small-changes.md#planecontrast-specific-feature-adapters)
  as unstarted small changes. The terminal-1×1-only trimming proposal was rejected.
  Bias-field augmentation, layer-wise
  CoAtNet learning-rate decay, standalone BatchNorm recalibration, ordered within-series
  aggregation and SAM were not selected for the backlog in this review.

- **Literature shortlist accepted, 2026-09-21:**
  [richer report supervision through OA attributes](report-attribute-extraction.md)
  is an unstarted big bet;
  [attention-token dropout](../backlog/small-changes.md#attention-token-dropout) was later tested
  and rejected (2026-09-25).
  Feature-space augmentation (DSU) was also accepted; it was later adopted in the
  [augmentation campaign](../validation/results/2026-09-26-augmentation-campaign.md).

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
  The autoresearch gate requires the new split and labels; `scripts/tools/train_overnight.py`
  takes split, fold and seed from its control config since 2026-09-27.

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
  [current model](../modeling/current-model.md) and [durable experiment record](../validation/results/2026-09-21-anatomical-attention-result.md).
- **Research decisions:** valid candidates are adopted when macro gain >= 0.001 and the
  central 80% paired-report-group interval has lower bound > 0; otherwise discarded.
  The 95% interval remains diagnostic. Candidates inherit the incumbent seed. This is
  a practical progression rule, not independent confirmation; training variability and
  repeated fold-0 selection remain outside the interval. Older noise measurements are
  historical evidence, not a blanket rule that all gains below 0.003 carry no information.
- **Autoresearch:** every campaign now shares one
  [evaluation and adoption rule](../../autoresearch/protocol.md#shared-evaluation-and-adoption):
  fold 0 of `all_5fold_scanner_v1`, combined labels, checkpoint selected at ×1 and scored on its
  ×4 τ 1 predictions (since 2026-09-22), gain ≥ 0.001 with a positive 80% lower bound. The three
  earliest campaigns were sealed on `all_5fold_v3`, and sampling and the first oblique campaign on
  ×1 predictions; none can resume. The x4 and
  [cleaned-source](../../autoresearch/campaigns/oblique-clean/INDEX.md) oblique campaigns are
  retired with the [oblique line](../history/oblique-groups.md), and no campaign is active. Current
  state and transition evidence are owned by
  [autoresearch](../../autoresearch/campaigns/INDEX.md).
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
- **Predictions at ×4 since 2026-09-22** (user decision): ×1 epoch validation still selects the
  checkpoint, but `configs/training.yaml` sets `validation.prediction_image_multiplier: 4`, so the
  saved fold-0 predictions, comparisons and the autoresearch adoption rule score that checkpoint at
  ×4 and τ 1 (about 50 s more per run). Older runs are comparable after
  [`repredict_run`](../../scripts/tools/repredict_run.py); `oblique` re-predicted scores 0.902373.
- **Historical loading reference, 2026-09-18:** `control-check` fold 0, best epoch 6, validation
  macro AUC **0.88477**, preprocessing identity `b8b6c639`. It is the first checkpoint carrying the loading
  changes, it sits inside the ×1 cohort spread (median 0.88707, sd 0.00331, and the cohort's own two
  controls are 0.88380 and 0.88645), and `outputs/submission/exports/control-check-x4-tau075` is
  exported from it at ×4 and temperature 0.75.
- **Best public score since 2026-09-21: 0.945**, reached by three single fold-0 models: `k_05` at
  ×4 τ 0.75 (S41), `oblique` at ×5 τ 0.70 (S47) and `image-blend` at ×4 τ 0.75 (S50)
  ([ledger](../inference/submissions/ledger.md)). The ensemble below was the best before them.
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
  [small changes](../backlog/small-changes.md#bce-weighting-and-study-sampling), adaptive work ran in the
  [noisy-labels campaign](../validation/results/2026-09-23-noisy-labels-campaign.md).
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
  continuation and learned mixtures. Dropped from the backlog on 2026-09-23 by user decision: no
  per-label selection or tuning; the note remains as research.
- **Research-agenda review completed, 2026-09-20:** the retained unstarted priorities are
  per-label attention temperatures (dropped 2026-09-23: no per-label tuning),
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
