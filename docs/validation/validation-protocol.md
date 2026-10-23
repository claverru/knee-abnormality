# Validation protocol

The validation protocol as it stood when the competition ended, with the superseded protocols
dated in [validation splits](validation-splits.md#superseded-splits). Run results are in [experiment results](results/INDEX.md); how local
scores related to the public board is in [local-to-public score relationship](local-to-public-score-relationship.md);
the comparison and diagnostic tools are in [run comparison and diagnostics](run-comparison-and-diagnostics.md).

## Final protocol

- **Metric:** the official one, the unweighted mean of twelve per-target ROC-AUCs
  ([Kaggle evaluation page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation));
  submit one row per `StudyInstanceUID` with continuous confidence scores for the twelve target
  columns of `sample_submission.csv`, never thresholded ones.
- **Split of the recipe:** `all_5fold_scanner_v2` (since 2026-09-28), protocol version 2, five
  stratified folds over all 4,407 studies with report groups intact. The recipe config
  (`configs/resnet50.yaml`) screens on **fold 2** with seed **1110484336** and the `rebuild_state_v2`
  labels. `validation.folds` lives only in the config; every command and the autoresearch runner
  read it from there.
- **Ensemble chain:** the v2 cycle (to 2026-10-18) trained and scored on `rebuild_state_v2` with
  rerolled `all_5fold_scanner_v2_r*` splits; the v3 cycle (from 2026-10-19) on `rebuild_state_v3`
  through `all_5fold_scanner_v3`, the same folds as v2 with retargeted evaluation targets, and its
  own rerolled `all_5fold_scanner_v3_r*` splits. Rerolls give each family and generation different
  held-out studies.
- **Endpoint:** the evaluation targets are frozen with the split: gold on the 58 gold studies, the
  split's label table elsewhere. Training may use another label source; validation never does.
  Full held-out-fold macro AUC is the development selection metric. It is a mixed-label
  endpoint, not a public score and not an independently annotated reference.
- **Densities:** epoch validation at ×1 (since 2026-09-17) picks the checkpoint on hard
  `val_macro_auc`; the saved predictions that comparisons and the adoption rule score come from
  that checkpoint at ×4 and attention temperature 1 (since 2026-09-22, about 50 s more per run). Soft macro AUC (since
  2026-09-23) is logged and used to compare runs; selection stayed on hard AUC.
- **Gold diagnostics:** the fold's 11-12 gold studies are recorded and decide nothing. Across
  comparable submissions gold-only AUC and BCE carried no public-score signal, and gold AUC has
  three times macro AUC's run-to-run noise
  ([comparison](results/validation-and-ensembles/local-public-metric-comparison/2026-09-18-comparable-submissions.md)).
- **Adoption of a candidate** (autoresearch campaigns, 2026-10-10): strictly above the incumbent's
  unrounded macro AUC at both ×1 and ×4. Merging into main was the user's decision, informed by
  paired public scores; see [Adoption rules](#adoption-rules).
- **Final judge:** the public board. At the end the ensembles were compared by submission, and
  local out-of-fold scores were not used to rank members, weights or the attention temperature
  ([attention temperature](results/pooling-and-attention/2026-10-22-attention-temperature.md),
  [gold-fitted stacking](results/validation-and-ensembles/2026-10-16-gold-fitted-stacking.md)).

## Splits

Fold membership and evaluation targets both change between splits, so **scores on different splits
are never comparable**, and a run is never resumed across splits. The recipe's split is
`all_5fold_scanner_v2`; the v3 chain scored on `all_5fold_scanner_v3` (the same folds with
`rebuild_state_v3` targets) and its members on rerolled splits. Fold tables, stratification, hashes,
rerolls, special splits and the superseded splits with their dates are in
[validation splits](validation-splits.md).

## Membership and leakage controls

Each split's `studies.csv` holds every training study with `StudyInstanceUID`, `report_group`,
`is_gold` and `fold`.

- For model `k`, training membership is `fold != k` and validation membership is `fold == k`. Join
  by study id so every series and window follows its study. Every study is in exactly one fold:
  there is no shared training-only pool and no `fold == -1` outside `gold_5fold_v1`.
- A held-out report group is entirely validation data, gold members included. No gold study is
  forced into every model's validation set.
- Label extractors, prompts, thresholds, window selectors and preprocessing parameters fitted with
  reference targets use only the model's training portion. Record external label provenance; prior
  tuning against all 58 labels compromises independent CV.
- Fix the epoch budget and checkpoint rule across comparisons, keep the five-fold membership fixed
  within them, and compare against a recorded incumbent trained on the same split; no automatic
  baseline rerun is required for each experiment. Repeated checkpoint and
  hyperparameter selection on these folds is development tuning, not a fresh assessment, and no
  permanent reference holdout exists in the protocol (`gold_holdout_5fold_v1` is the exception for
  student comparisons).
- Fold ids alone cannot prove a model excluded its validation data: keep training membership, label
  versions and checkpoint provenance with each run. A student's validation studies are seen by its
  teachers unless teacher and student share a split.
- Completing all five folds yields out-of-fold predictions for all 4,407 studies, the 58 gold
  studies included. Predictions from baseline1's shared pool are not OOF predictions of this
  protocol; retrain a recipe on the new membership before comparing it with the new endpoint.

## Frozen evaluation targets and metrics

`targets.csv` freezes the effective table that generated the split: gold labels on the gold studies
and the split's label source elsewhere (Steven v4, `combined_v1`, `rebuild_state_v2` or
`rebuild_state_v3` by split). Training may select another pinned label source for a controlled label
experiment; validation keeps the frozen targets, so the endpoint does not move when label weights or
sources change. The Lightning module gathers every held-out study before computing metrics;
averaging batch AUCs is not the official metric.

- `val_macro_auc`: mean of twelve ROC-AUCs against `targets.csv >= 0.5` over the complete
  validation fold (an exact 0.5 maps to positive). Predictions stay continuous.
- `val_soft_macro_auc`: mean of twelve soft ROC-AUCs against the unthresholded frozen
  probabilities. Each ordered pair of distinct studies (i, j) is weighted y_i(1 - y_j) and the score
  is the weighted share of pairs ranked correctly, ties half. It ranked past submissions closer to
  the board than the hard metric ([result](results/labels-and-losses/2026-09-23-validation-auc-variants.md)).
  Checkpoint selection stays on `val_macro_auc` (two paired tests of the hard-AUC checkpoint against a later one tied publicly: S41/S49 and S64/S65; [augmentation campaign](results/campaigns-and-queues/2026-09-26-augmentation-campaign.md)).
- `val_soft_target_bce`: BCE against the frozen probabilities.
- Epoch JSON files carry `gold` and `report_derived` subgroup metrics, study counts and per-target
  AUCs. An absent class gives `null` for that target, lists it in `undefined_targets` and leaves the
  subgroup macro `null` instead of averaging fewer targets; a complete gold macro is also logged as
  `val_gold_macro_auc`.

The full-fold metric combines gold and report-derived labels on purpose. Most rows are inferred, so
gains may reflect agreement with that label source; gold diagnostics keep the reference-label view.
Never average the two subgroup macros and call the result the full-fold metric. A gold diagnostic is
clean only when the model has not learned gold through its labels or teachers: for the 2026-10-21
analyses, gold was scored against the silver table by mistake, and only generation-1 models
scored on their own held-out fold give clean gold numbers
([attention temperature](results/pooling-and-attention/2026-10-22-attention-temperature.md#gold-was-scored-on-silver-labels)).

## Controlled comparisons

Change one factor against a matched control and predefine the principal comparison and
checkpoint rule; never search seeds for an apparent improvement. Preserve model size, training
budget, image dimensions, physical crop, slice budget, and the validation (×1) and prediction (×4)
densities unless one of them is the factor. Training targets stay soft unless supervision is the
factor, and validation memberships and frozen targets never change. Record full-fold and per-target
AUC, subgroup diagnostics, BCE, runtime, peak VRAM, the distinct slices and TTA passes used, and keep
all validation predictions with the run's
[identities](../architecture/configs-and-experiment-artifacts.md). Export parity is not required for
each run. Outside the sealed autoresearch campaigns the user launched training and the encoding
diagnostic manually; agents scheduled no training, optimizer-step checks, automatic diagnostics,
extra seeds or baseline reruns. Use fresh
artifact directories. Additional-fold training, full OOF scoring, ensembling and bigger images were
deferred to the endgame, and were not prerequisites for an experiment; the ideas never run are in
[untried ideas](../backlog/untried-ideas.md).

**Fold and seed rotation** (user decision, 2026-09-25; the strategy chosen on 2026-09-11 had been to
iterate on fold 0 until near the end):

1. **Paired comparisons:** a candidate always uses its control's fold and seed.
2. **Rotate at adoptions merged into main:** the new control is retrained with a fresh random seed
   (drawn once, never searched or chosen by score) on the next fold, cycling 0 -> 1 -> 2 -> 3 -> 4 -> 0.
   Autoresearch `provisional_accept` decisions are working choices, not adoptions.
3. **One source of truth:** fold and seed live only in the recipe config (`validation.folds`, `seed`).
4. **Campaigns:** a campaign runs on its reference's fold from start to end; since 2026-09-26 a
   provisional acceptance makes the adopted run the reference, with no control retrain
   ([autoresearch protocol](../../autoresearch/protocol.md#training-seed-policy)).
5. **Reporting:** scores are reported per fold stretch; never add up or compare scores across folds.
6. **Final check:** the public board, then all five folds in the endgame.

How it went: the first rotation was the OA-detector control (fold 1, seed 1485198267) on the scanner
v1 split, and the rotation then ran through folds 2-4 of that split; the 2026-09-30 adoption kept its
fold and seed (user decision). On `all_5fold_scanner_v2` the recipe was on fold 0 with seed 1533697213
from 2026-09-28, fold 1 with seed 819238850 from 2026-10-06, fold 1 with seed 1110484336 from
2026-10-07 (the seed of the submitted side-routed runs S118 and S119), and on **fold 2 with seed
1110484336** from 2026-10-13, when the ResNet-50 became the only training config (user decision).
The adoption-by-adoption record is in the [adoption history](../modeling/current-model.md#adoption-history).

**Why rotate:**
- **Seed:** it alone is worth several thousandths. The adopted knee-geometry model scored 0.905834 at
  seed 2263738815 and 0.899547 at 638120755 with the same code, -0.0063 (95% [-0.0108, -0.0023]),
  mostly on labels the change does not touch.
- **Fold:** reusing one fold for every decision overfits it (S59 against S58 above), and the
  literature agrees: [Dwork et al. 2015](https://www.science.org/doi/10.1126/science.aaa9375) on
  holdout reuse, [Bouthillier et al. 2021](https://arxiv.org/pdf/2103.03098) on split variance
  exceeding seed variance, [Nagler et al. 2024](https://arxiv.org/pdf/2405.15393) on reshuffling
  splits during selection.
- **Cost:** rotating at adoptions costs no extra runs, because the new control is retrained anyway;
  rotating per experiment would need a control on every fold.

## Adoption rules

- **Project rule (main):** adopt a valid implementation only when macro gain >= 0.001 and the lower
  bound of the central 80% paired report-group bootstrap interval is strictly positive; the 95%
  interval is diagnostic. No automatic confirmation branches or control reruns follow a result. The
  rule is an engineering policy and does not establish cross-seed robustness. In practice the later
  adoptions on main were decided by the user on paired public scores.
- **Autoresearch rule** (code: `ACCEPTANCE_RULE` in `campaign/rules.py`; user decision 2026-10-10):
  strictly above the pinned incumbent's unrounded macro AUC at both ×1 (the selected checkpoint's
  epoch validation) and ×4 (its saved predictions), as the only condition; the 80% interval is
  reported but does not decide. It replaced, in turn, a positive-95%-lower-bound gate, a gain-only
  rule, a rule that also required a gain of at least 0.001 (2026-10-02) and a positive lower bound of
  the 80% interval; records written under those carry the older rule. See
  [adoption policy](../../autoresearch/protocol.md#adoption-policy).
- **No adoption of generalization risks on a local fold** (user decision, 2026-10-13): a change that
  frees capacity from the L2-SP anchor or removes a robustness augmentation is not adopted on local
  AUC alone.
- **Never decide on gold-only numbers, never tune τ on silver** (see
  [local-to-public score relationship](local-to-public-score-relationship.md)).

## What a comparison can resolve

- **Training runs** carry variability not captured by validation resampling. Seed controls and sigma
  sweeps showed spreads of several thousandths under different recipes
  ([overnight results](results/campaigns-and-queues/2026-09-18-overnight-results.md),
  [sigma sweep](results/validation-and-ensembles/2026-09-19-fold-ensembles-and-sigma-sweep.md)); two
  inert-module repeats of the ResNet-50 recipe landed 0.0007 and 0.0010 below their incumbents at ×4
  and 0.0012 and 0.0013 below at ×1, so a single-run difference under about 0.001 at ×4 is inside that
  distance ([targets-and-inputs campaign](results/campaigns-and-queues/2026-10-13-targets-and-inputs-campaign.md#how-far-two-trainings-of-one-recipe-are-apart)).
  These observations motivate caution, not a universal minimum gain. Matching seeds removes one noise
  source but does not make runs deterministic, and repeated selection against one control can still
  produce lucky winners.
- **Test-time variants of one frozen checkpoint** are cheaper to resolve because the paired bootstrap
  over studies removes training variability. Over 36 pairs the standard error of the macro-AUC change
  scales with how far the variant moves predictions: SE ~ 0.067 x p^1.05, p the mean absolute
  probability change (R^2 0.83). The 95% resolvable difference on full fold 0 is about 0.0003 at
  p = 0.003, 0.001 at p = 0.01 and 0.002 at p = 0.02; about 1.46 times larger on half of the split,
  0.67-0.91 times when averaged over three checkpoints. Script and outputs:
  [snapshot](../history/sources/series-and-sampling/README.md#triplet-allocation),
  `outputs/analysis/d/detection-floor-20260917/`.
- **Validation multiplier:** scores at ×1 sit about 0.0035 below the same checkpoints at ×2, so runs
  are comparable only with a control at the same multiplier.
- **Prediction multiplier (since 2026-09-22):** models scale differently with more triplets:
  `oblique` was -0.0015 against `k_05` at ×1 and +0.0003 at ×4 tau 0.75
  ([result](results/sampling/2026-09-22-oblique-groups-result.md#temperature-and-count)). Compare only
  runs predicted at the same multiplier; `repredict_run` re-predicts an older run
  ([tools](run-comparison-and-diagnostics.md#re-predicting-an-older-run)).
- **Against the leaderboard,** macro AUC is the best available local predictor but does not resolve
  submissions whose local spread is below one run's noise; prefer a public result over a sub-0.005
  local difference ([relationship](local-to-public-score-relationship.md)). Single-fold comparisons
  cannot resolve training-seed variability or establish small repeatable gains.

## Ensembles

Rank averaging and hard voting were mentioned in earlier notes with no recorded implementation or
result; do not treat them as defined methods. Prediction-scale differences between losses are a
problem for combining raw logits or probabilities. A strictly increasing transform of one model's
predictions preserves each target's ROC AUC, so calibration or thresholding alone is not an AUC gain;
clipping, averaging and different transforms across models can change rankings and must be evaluated
on held-out predictions. A stacker fitted to OOF predictions needs its own held-out or nested
evaluation, and every learned component that uses evaluation labels must be fold-isolated; the
[code review](../research/competition-and-prior-art/code-and-notebooks.md) shows where artifact names
or fold loops alone do not establish that isolation. Track per-target behavior and complementary
errors beside aggregate scores; better labels can make models more correlated and shrink their
ensemble contribution
([supervision](../modeling/supervision.md#downstream-benefit-changed-over-time)). The result of
fitting combiners on the 58 gold studies: none beat the equal average, so members were weighted
equally ([gold-fitted stacking](results/validation-and-ensembles/2026-10-16-gold-fitted-stacking.md)).

## Reproduce and score

```bash
uv run --locked python -m scripts.tools.validation prepare
uv run --locked python -m scripts.tools.validation retarget --source-split-dir data/folds/all_5fold_scanner_v2 --label-manifest <manifest> --label-source <table>
uv run --locked python -m scripts.tools.validation score --predictions outputs/training/run/oof_predictions.csv
```

[`scripts/tools/validation.py`](../../scripts/tools/validation.py) saves `studies.csv`,
`label_counts.csv` (full-fold `positive`/`negative` and `gold_positive`/`gold_negative` per target and
fold), `targets.csv` and `protocol.json` under the split directory. The `pydantic-settings` CLI
accepts `--config`, `--data-dir`, `--split-dir`, `--n-splits`, `--seed`, `--scanner-table` and
`--label-source`; without overrides it uses the config seed and training label source. The protocol
records official input hashes, label source/version/hash, label-manifest hash, target-table hash,
generator source hashes, dependency versions, scanner-group table hash and rule, and per-fold study
and scanner counts. Identical regeneration is allowed; conflicting artifacts require a new
`--split-dir`. Never change folds midway through a comparison.

Study-manifest SHA-256s: `all_5fold_scanner_v2` and `all_5fold_scanner_v3` `4b4563c3887cefad3c81ed7c525ffdd64a15c4ba922f7e1e6f1d440716d94032`,
`all_5fold_scanner_v1` `2ed3ad16b5d34d2547b5b0d202543d24d386c6f96275ee2f44d488dcee4fdf74`,
`all_5fold_v3` `c709cd27ff54dd10d833abfa966622f9a5ecadef5d37670da658d81c26f7b3ed`.

The OOF CSV must contain exactly all 4,407 study ids, the producing integer `fold` and the twelve
target probabilities. The shared scorer, `knee_abnormality.evaluation.scoring`, serves both
`validation.py score` and training runs that cover every fold. It accepts only protocol version 2,
applies the training label loader's checks (input hashes, membership hash, 0.5 threshold, complete
target coverage), aligns by id, rejects invalid probabilities, and prints pooled OOF macro AUC,
per-target AUC, per-fold scores and their mean, soft-target BCE and the gold/report-derived
diagnostics as JSON. A single-class full-fold target fails explicitly. Kaggle's handling of a
single-class target column in the hidden test was unanswered in the
[scoring edge-case thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740094),
so undefined local target AUCs are reported rather than assuming an implementation matches an
unverified scorer.

With the other folds trained, pooled OOF macro AUC across all 4,407 studies is read alongside every
fold and target; pooled AUC compares scores from different models and need not equal the mean fold
AUC. Until then, use the full screening-fold macro AUC and per-target diagnostics. Retain continuous
probabilities with consistent output semantics, avoid treating tiny changes as reliable gains or a
fold-score spread as a confidence interval, and keep the pooled 58-study gold diagnostic beside the
primary mixed-label score.

## Public evidence behind the design

Public experiments showed why the evaluation label source matters: hard-label training can win
against a hard-label surrogate while soft-label training has the better point estimate on independent
radiologist labels. The paired results, uncertainty, duplicate-report grouping, scanner stress tests
and public-board tuning failures are in
[public supervision evidence](../research/competition-and-prior-art/public-research.md#report-derived-supervision)
and [validation evidence](../research/competition-and-prior-art/public-research.md#validation-and-ensembling);
their reported scores were not reproduced locally.
