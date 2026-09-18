# Ongoing work

A snapshot of where the project stands and what comes next, so the thread is not lost. Details
live in the linked documents; follow the [work-tracking rules](../AGENTS.md#work-tracking).

## State (2026-09-18)

- **Code:** the full-slice recipe ([current model](modeling/current-model.md)) with series
  filtering and deterministic source ranking (preprocessing identity
  `ranked_filtered_physical_gaussian_overlap_spatial_tokens_reflection_v10`,
  [preparation](data-preparation/current-preparation.md#series-indexing-guards-and-exclusions)).
  Its seeds 42–44 scored 0.889573, 0.889650 and 0.892795 on fold 0 (mean 0.890673), a tie with
  the previous code's seed controls, and seed 42 scored 0.933 public
  ([result](validation/results/2026-09-17-ranked-filtered-result.md)).
- **Best submission:** `ranked-filtered-seed42` with attention temperature 0.75 in the weights and
  ×4 evaluation images, **0.939 public** (×3 scored 0.938 twice), against 0.933 for the same
  checkpoint at temperature 1.0 and ×2, and 0.936 for the previous reflection/sigma-40 champion
  ([submissions](inference/submissions/2026-09-18-temperature-and-count-submissions.md)). The
  temperature is free at inference and recipe-independent, so every later submission should carry
  it.
- **Noise:** run-to-run noise comes from training (standard deviation about 0.0015–0.0018); a
  single-run difference below about 0.003 carries no information, even at a matched seed. The
  2026-09-18 controls span 0.0026 at ×1 validation
  ([overnight results](validation/results/2026-09-18-overnight-results.md)).
- **Validation at ×1** since 2026-09-17 (user decision): fold-0 scores from then on sit about
  0.0035 below the earlier ×2 numbers and are comparable only with a ×1 control. Submissions are
  unaffected; the bundle's test multiplier is separate.
- **Adopted 2026-09-18:** `data.study_triplet_total: 26`, a floor on a study's training triplets
  that scales the types it has in proportion to their budgets. Fold 0 put it at +0.0033 over the
  control mean, inside the control spread, so the case is structural: it only ever adds triplets,
  for 5.5% more compute, and the studies missing two types stop being under-sampled. Fresh runs
  only; it changes the preprocessing identity.
- **Dropped:** learned crop selection ([record](history/crop-selection-investigation.md)).
- **Strategy:** iterate on fold 0; folds, ensembling, bigger images, TTA and weight averaging wait
  for the [final phase](backlog/endgame.md)
  ([validation strategy](validation/validation-strategy.md)). Test-time risks are listed in
  [risks](backlog/risks.md).

## Peripheral coverage of the Gaussian sampler (measured 2026-09-18)

Triplets are placed along each plane's axis by a truncated Gaussian, `data.sampling_sigma_mm: 40`,
so coverage falls with distance from the study centre. Several labels keep most of their evidence
out there. Measured read-only from the fold-0 token cache and the depth profiles; scripts and
tables in `outputs/analysis/peripheral-coverage-20260918/`.

- **The sampler thins with depth.** Selected share ÷ candidate share beyond 40 mm is 0.45–0.59 for
  every type, and beyond 50 mm it falls to 0.36 for coronal fluid and 0.14–0.16 for the sagittal
  types. Selection probability drops from about 0.40 at the centre to 0.19 at 40 mm and beyond.
- **14–31% of a series' slices are never sampled** at the default counts, worst for axial T1/PD
  (mean 31%, p90 63%).
- **Share of a label's depth-weighted evidence sitting beyond ±20 mm:** Baker's cyst in coronal
  fluid 80%, lateral meniscus in sagittal fluid 74%, synovitis in axial fluid 67%, effusion in
  axial fluid 65%, medial meniscus in sagittal fluid 59%, MCL in axial fluid 56%. Synovitis peaks
  at 40–50 mm superior, where the selection probability is half the central value; effusion at
  30–40 mm, at 0.28. The medial meniscus in coronal fluid sits close enough that sigma 40 covers
  it.
- **Not a fix that follows for free:** lifting axial fluid's density at 40–50 mm to the central
  level implies a sigma of roughly 65–80 for that type, while global sigma 60, 80 and 120 were all
  measured worse and 30 tied
  ([overnight 2026-09-15](validation/results/2026-09-15-overnight-results.md)). Per-type sigma and
  per-type centre offsets have never been tried, and both can be scored on the caches without the
  GPU.
- **Limits:** the separation metric behind the evidence shares is small-sample and noisy in tail
  bins, fracture is too sparse to read at all, and geometry cannot recover in-plane tissue
  clipping — the reference point is defined near each series' image centre, so the pixel-based
  figures in [input resolution](research/input-resolution.md) remain the only source for the 140 mm
  crop (sagittal tissue exceeds it in 34% of series, almost all posterior).

## Triplet budget allocation (closed 2026-09-18)

Ran 2026-09-17 to 18: can a better a-priori split of a study's triplets across plane–contrast
types and series beat today's fixed 6/6/5/5/5/5, at about the same compute? **No rule beat it.**
What the investigation produced instead is the attention temperature, now the best submission.

- **Per-type contributions** are exact per label: fluid types carry about 74% of the
  positive–negative separation with 61% of the tokens; sagittal fluid leads ACL, axial fluid
  Baker's cyst and effusion, coronal fluid MCL and the menisci; coronal and sagittal T1/PD still
  lead the OA labels ([result](validation/results/2026-09-17-type-contribution.md)).
- **Reallocation at a fixed total does nothing** above the detection floor, on checkpoints trained
  with a fixed split or with random splits; starving a type does hurt
  ([result](validation/results/2026-09-17-test-time-allocation.md)). The token caches
  (`outputs/analysis/token-cache-*-20260917/`) score any allocation in about 30 ms and remain
  usable.
- **Training-time triplet counts do matter** while test-time counts barely move a frozen model,
  so the two must not be confused.
- **Rejected:** the specified missing-type fill (worse than losing the budget), budgets A/B,
  count-normalised attention ([result](validation/results/2026-09-17-count-normalized-attention-result.md)),
  random per-type allocation in training (a tie, option removed), and the repeated-series share
  ([overnight](validation/results/2026-09-18-overnight-results.md)).
- **Adopted:** the attention temperature, τ 0.75, which came out of the bag-size check and scores
  0.939 public with ×4 images
  ([submissions](inference/submissions/2026-09-18-temperature-and-count-submissions.md)).

Still open from it: where the evaluation count turns over above ×4 and what it costs in runtime;
and whether splitting a slot's own budget more evenly between its series helps, which is the part
of the repeated-series evidence that does not starve other types.

## Inference loading speedup

Investigation started 2026-09-17 with the user. The candidate changes, every measurement, the
closed regressions, the retrain constraint and the prior art now live in one document:
[loading speedup](backlog/loading-speedup.md).

## Next

Experiments run one by one against a control at the same validation multiplier, as one unseeded
run each (random seed, recorded). The [overnight queue](backlog/overnight.md) holds the
hyperparameter candidates; nothing is queued for tonight.

Also pending:

- **Evaluation count above ×4:** ×3 scored 0.938 and ×4 0.939 publicly, one rounding step apart,
  so where the count stops paying is unresolved, and its inference-time cost counts toward the
  efficiency prize
  ([submissions](inference/submissions/2026-09-18-temperature-and-count-submissions.md)).
- **Robust study centre: removed 2026-09-18.** Four runs at 10, 20 and 30 mm landed inside the
  control spread, the 21 studies it changes showed neither harm nor benefit, the combined run
  scored 0.938 publicly against 0.939 for the checkpoint it replaced, and it works against
  peripheral coverage, since a rejected study currently samples uniformly across each slab rather
  than from the Gaussian. The option is gone from the code
  ([results](validation/results/2026-09-18-overnight-results.md)).
- Data-loading speedups, shipped with the next retrain
  ([loading speedup](backlog/loading-speedup.md)).
- Submit the main-thread warm-up with the next submission; torch-free coordinator
  ([runtime changes](backlog/loading-speedup.md#runtime-changes-outside-the-preprocessing-identity)).
