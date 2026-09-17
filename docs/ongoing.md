# Ongoing work

A snapshot of where the project stands and what comes next, so the thread is not lost. Details
live in the linked documents; follow the [work-tracking rules](../AGENTS.md#work-tracking).

## State (2026-09-17)

- **Code:** the full-slice recipe ([current model](modeling/current-model.md)) with series
  filtering and deterministic source ranking (preprocessing identity
  `ranked_filtered_physical_gaussian_overlap_spatial_tokens_reflection_v10`,
  [preparation](data-preparation/current-preparation.md#series-indexing-guards-and-exclusions)).
  Its seeds 42–44 scored 0.889573, 0.889650 and 0.892795 on fold 0 (mean 0.890673), a tie with
  the previous code's seed controls, and seed 42 scored 0.933 public
  ([result](validation/results/2026-09-17-ranked-filtered-result.md)).
- **Best submission:** the reflection/sigma-40 champion, 0.936 public
  ([fast runtime](inference/submissions/2026-09-15-fast-champion-submission.md)).
- **Noise:** run-to-run noise comes from training (standard deviation about 0.0015–0.0018); a
  single-run difference below about 0.003 carries no information, even at a matched seed.
- **Dropped:** learned crop selection ([record](history/crop-selection-investigation.md)).
- **Strategy:** iterate on fold 0; folds, ensembling, bigger images, TTA and weight averaging wait
  for the [final phase](backlog/endgame.md)
  ([validation strategy](validation/validation-strategy.md)). Test-time risks are listed in
  [risks](backlog/risks.md).

## Triplet budget allocation

Investigation started 2026-09-17 with the user.

**Goal:** give each label just enough of the right triplets for about today's compute.
- *As an optimisation problem:* for a fixed budget, choose the triplet count per plane–contrast
  type and series, and later where the triplets sit, to maximise macro AUC while covering every
  label.
- *The rule is a priori:* it uses only the types and series a study has, maybe plus headers, and
  runs the same way in training, validation and inference.
- *Learned, pixel-driven placement stays dropped*
  ([crop selection](history/crop-selection-investigation.md)).

**Scope:**
- *In scope:* the base budget per type, where a missing type's budget goes, a fixed per-study
  total against per-type budgets, and how repeated series share a slot (661 fold-0 slots).
- *Related:* count-normalised attention
  ([entry](backlog/small-changes.md#attention-pooling)), and axial T1/PD not being missing at
  random.
- *Handled downstream:* reallocation after series dropout, the ×2 evaluation multiplier,
  placement within a slot, capacity overflow and compute.

**Current behaviour (code):**
- *Budgets:* fixed per type, 6/6/5/5/5/5 training triplets (axial T1/PD, axial fluid, coronal
  T1/PD, coronal fluid, sagittal T1/PD, sagittal fluid), doubled at evaluation.
- *Missing types:* an absent type sends nothing, and its budget goes nowhere.
- *Repeated series* share their slot's budget.
- *Fold 0:* studies average 25.4 training triplets against the full 32.

  | Types present | Studies | Training triplets |
  |---|---:|---:|
  | 6 | 112 | 32 |
  | 5 | 529 | 26 |
  | 4 | 262 | 21.2 (mean) |
  | 3 | 1 | 16 |

  No present slot is short of capacity.

**Findings so far:**
- **The missing-type fill as specified (Next, item 1) is mostly a compute increase.**
  - *Reach:* it changes 792 of 904 fold-0 studies.
  - *Triplets:* the average rises to 32 in training (+26%) and to 64 at evaluation, where 13
    studies stay short.
  - *Where they go:* mainly axial T1/PD's budget moving to axial fluid (mean 6 → 10.8).
  - *Source:* simulation on the type-importance capacities
    ([snapshot](history/series-and-sampling-sources/README.md#triplet-allocation)).
- **Exact per-type logit contributions**
  ([result](validation/results/2026-09-17-type-contribution.md)):
  - *Fluid:* about 74% of the positive–negative separation in five-type studies, with 61% of the
    tokens.
  - *Per-token efficiency:* sagittal fluid is highest (1.37–1.64×) and axial T1/PD lowest (0.39×).
  - *Leading type per label:* sagittal fluid leads ACL; axial fluid leads Baker's cyst, effusion
    and synovitis; coronal fluid leads MCL and the menisci.
  - *T1/PD still leads:* coronal T1/PD for medial OA and sagittal T1/PD for lateral OA.
  - *Limit:* these shares measure reliance at today's budgets, not the value of one more triplet.

- **Detection floor for test-time variants of one frozen checkpoint** (paired bootstrap over
  studies, 36 pairs; [snapshot](history/series-and-sampling-sources/README.md#triplet-allocation),
  outputs `outputs/analysis/detection-floor-20260917/`):
  - *Scaling:* the standard error of Δ macro AUC grows with how much the variant moves the
    predictions: SE ≈ 0.067 × p^1.05, where p is the mean absolute probability change
    (R² 0.83).
  - *Resolvable Δ at 95% on full fold 0:* 0.0003 at p = 0.003 (the size of the tie-break
    change), 0.001 at p = 0.01, and 0.002 at p = 0.02.
  - *On a half:* about 1.46× larger (√2 expected).
  - *Averaged over the three checkpoints:* 0.67–0.91× smaller, with the biggest gain for small
    perturbations.
  - *Limit:* this covers evaluation noise only. Comparisons that need training also carry
    run-to-run noise, with a standard deviation of about 0.0015–0.0018.

- **The fold-0 token cache is built** (`outputs/analysis/token-cache-20260917/`; its `README.md`
  documents the API, `cache_lib.score_rule` and `exact.exact_score`).
  - *Contents:* every candidate triplet (161,466) of the three ranked-filtered checkpoints. Any
    allocation scores in about 30 ms.
  - *Selections:* they match the pipeline's in all 5,424 slots.
  - *Approximation:* the whole-series intensity window moves single probabilities by a mean
    0.003 (max 0.085), but macro AUC by at most 1.2e-4. Differences in macro AUC against the
    default allocation are off by at most 0.00026 on one run.
  - *Confirming finalists:* use `exact.exact_score`, which takes about 2 minutes per allocation.
  - *Split:* a fixed search/confirm split (456/448 studies). The search half scores
    0.003–0.005 higher at every allocation, so compare allocations only within one half.
  - *Test-time scaling:* the evaluation count still helps beyond ×2
    ([endgame](backlog/endgame.md#test-time-augmentation)).

- **Test-time allocation on frozen checkpoints** (five parallel analyses on the cache;
  [result](validation/results/2026-09-17-test-time-allocation.md)):
  - **Reallocation among present types did not confirm.** Log-proportional, Neyman,
    search-theory and greedy rules all failed on the confirm half. Each gained up to +0.0024 on
    search, and log-proportional lost 0.0036 on confirm.
  - **Near the default, per-type counts barely matter.** The exceptions are:
    - removing a fluid type, which costs 0.006–0.007;
    - one more coronal fluid triplet, worth +0.0003;
    - axial T1/PD's marginal value, which is slightly negative.
  - **The curves are saturating rather than logarithmic, and the types interact.**
  - **Missing types:**
    - Proportional renormalisation beats the specified fill, which is worse than losing the
      budget.
    - Its gain equals uniform scaling at the same compute (+26%).
    - Redistribution recovers only 31–43% of a removed fluid type's loss.
  - **Repeated slots:**
    - Dropping the second-ranked series costs 0.0037 on confirm (0.0019 on search).
    - Extra triplets help more from the second series than from the top series (six of six
      comparisons, one resolvable).
  - **Depth profiles:** effusion and synovitis evidence sits 30–50 mm superior in axial fluid
    (the suprapatellar recess), and the medial meniscus 10–20 mm posterior in coronal fluid.

- **Training-time budget matters more than test-time budget.**
  - *Trained:* the user's uniform cuts, `less_images` (−20%) and `less_less_images` (−35%), lost
    0.0041 and 0.0051 against `00-baseline`
    ([result](validation/results/2026-09-16-restored-code-hyperparameter-runs.md)). These are
    single runs on the restored code, but both CIs exclude zero.
  - *Test time:* cutting the evaluation count of the frozen ranked-filtered checkpoints by 25%
    costs about 0.0001 on the cache (0.8906 against 0.8907).
  - *So:* the loss comes from training on fewer triplets, and test-time null results do not rule
    out training-time allocation effects.

- **The allocation-trained checkpoint responds like the fixed-split ones.** A second cache on
  `allocation-concentration` (trained with random per-type allocation, an option since removed)
  gives the same rule ranking (correlation 0.87 on search, 0.79 on
  confirm), the same curve shapes and marginal gains, and again nothing confirms; a
  209-allocation coordinate search at the default's compute gained +0.0024 on search and +0.0007
  on confirm ([result](validation/results/2026-09-17-test-time-allocation.md#the-allocation-trained-checkpoint-responds-the-same)).
  So the insensitivity belongs to the model and task, not to the training allocation, and it holds
  only within a band: redistribution among types that all keep roughly 8 or more triplets changes
  little, while starving a type costs 0.006–0.007 macro AUC and much more on the labels that need
  that plane and contrast. The apparent saturation past 8 is confounded with the bag-size mismatch
  between training and evaluation, so only the fixed-total comparisons are clean. The number of
  triplets in training does matter.

- **Sharper attention at inference gains +0.0024** [+0.0012, +0.0037] on fold 0, with no
  retraining and no extra compute: dividing the pooling scores by `tau` 0.75
  ([result](validation/results/2026-09-17-attention-temperature.md)). The peak is smooth at
  0.7–0.75 and all four checkpoints gain. It came out of checking whether the count curves were
  confounded by bag size; they are only at very large bags, which count-normalised attention also
  fixes. Not implemented, not confirmed on the exact path or the leaderboard
  ([entry](backlog/small-changes.md#attention-pooling)).

- **Count-normalised attention is rejected.** Trained with random per-type allocation it lost
  0.0035 [−0.0069, −0.0001] against the same recipe without it, worst on the menisci, and it was
  neutral-to-negative at inference
  ([result](validation/results/2026-09-17-count-normalized-attention-result.md)). The option was
  removed from the code.

**Open decisions (for the user):**
1. **The missing-type fill (Next, item 1):** drop it. As specified it is worse than losing the
   budget, and proportional renormalisation, its best replacement, only matches uniform scaling at
   the same compute.
2. **Budgets A/B (Next, item 2):** unsupported on both caches. A training run remains the only
   untested route, with a low expected ceiling (no rule moved either cache beyond its floor).
3. **Attention temperature:** implement `tau` as an inference option and confirm it on the exact
   path, then on the leaderboard, before adopting it. Decide whether training uses it too.
4. **Training-side follow-ups:** the study-level triplet total and the repeated-series share
   ([entries](backlog/small-changes.md#study-level-triplet-total)). Both tie at test time, and each
   would need its own training run.

## Next

Experiments run one by one, each on top of the previous one's accepted state, as one unseeded run
(random seed, recorded) against the ranked-filtered runs (seed 42: 0.889573)
([per-type budget evidence](validation/results/2026-09-17-plane-contrast-importance.md)):

1. **Missing-type fill:** when a plane–contrast type is absent, give its slots to the same plane's
   other contrast first, then coronal fluid, axial fluid and sagittal fluid
   ([entry](backlog/small-changes.md#per-type-budget-reallocation)). As specified it adds 26%
   training triplets on fold 0. At test time it scored worse than losing the budget, and
   proportional renormalisation did better; see
   [triplet budget allocation](#triplet-budget-allocation).
2. **Per-type budget reallocation:** budgets A (9/18/15/21/15/18) and B (9/18/12/21/12/15), in
   the order axial T1/PD, axial fluid, coronal T1/PD, coronal fluid, sagittal T1/PD, sagittal
   fluid ([entry](backlog/small-changes.md#per-type-budget-reallocation)). Test-time
   reallocations did not confirm; see [triplet budget allocation](#triplet-budget-allocation).

Also pending:

- **Robust study centre:** implemented behind `data.robust_center_outlier_mm`, and one seed-42
  run is queued ([overnight](backlog/overnight.md#next-queue)). The code changed the source
  digest, so the ranked-filtered checkpoints need their archived sources.
- Data-loading speedups, shipped with the next retrain
  ([plan](inference/loading-speedup-plan.md)).
- Submit the main-thread warm-up with the next submission; torch-free coordinator
  ([runtime changes](inference/loading-speedup-plan.md#runtime-changes-outside-the-preprocessing-identity)).
