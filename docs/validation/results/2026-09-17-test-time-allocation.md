# 2026-09-17 test-time triplet allocation

**On frozen checkpoints, no reallocation of triplets across types beat today's allocation on held-out
studies.** Every rule that gained on the search half gave the gain back on the confirm half. Each
type's curve saturates early, so once every present type keeps roughly 8 triplets, moving triplets
between them changes little. Starving a type is a different matter and does hurt.

Two findings did hold on both halves:
- **The fill as specified is worse than losing the budget.** For a missing type, proportional
  renormalisation over the types present is the better rule, but its gain matches uniform
  scaling at the same compute.
- **In repeated slots, both series matter.** Dropping the second-ranked series costs macro AUC,
  while extra triplets from it help at least as much as extra triplets from the top-ranked
  series.

Part of the [triplet budget allocation investigation](../../history/triplet-budget-allocation.md).

## Setup

- **Checkpoints and cache:** `ranked-filtered-seed{42,43,44}` on the fold-0 token cache
  (`outputs/analysis/t/token-cache-20260917/`). Selections are exact. The whole-series intensity
  window is within 1.2e-4 macro AUC of the pipeline at the default allocation.
- **Counts:** evaluation triplets, which are twice the training counts. The default is
  12/12/10/10/10/10 in slot order (axial T1/PD, axial fluid, coronal T1/PD, coronal fluid,
  sagittal T1/PD, sagittal fluid), a realised mean of 50.7 per study.
- **Split:** a fixed search/confirm split of 456/448 studies. The search half scores
  0.003–0.005 higher at every allocation, so differences are compared within a half.
- **Scores:** means over the three checkpoints. The CIs are paired bootstraps over studies with
  500 resamples, run under a 15-minute time cap per analysis.
- **Detection floor:** from the [comparison resolution rules](../validation-strategy.md#what-a-comparison-can-resolve),
  about 0.002–0.003 on one half for changes of p ≈ 0.02–0.03.
- **Scripts:** [snapshots](../../history/series-and-sampling-sources/README.md#triplet-allocation).
  Outputs: `outputs/analysis/allocation-{curves,substitution,series,neyman,search-theory}-20260917/`.

## Rules at today's compute

| Rule | Allocation idea | Search Δ | Confirm Δ |
|---|---|---:|---:|
| Log-fit proportional | Six-type split 0/.23/.06/.35/.07/.29 | −0.0003 | **−0.0036** [−0.0061, −0.0012] |
| Greedy on raw curves | Six-type 4/8/1/1/3/47; capacity cuts compute by 20% | −0.0064 | not run |
| Neyman n ∝ W·S, study totals kept | Six-type weights .11/.19/.12/.19/.15/.24 | +0.0024 | −0.0002 (all CIs cross 0) |
| Separation-share proportional | Weights from the cache | +0.0019 | not run |
| Efficiency proportional | Weights from the cache | +0.0016 | not run |
| Search-theory water-filling | (0, 9, 6, 15, 9, 15) | +0.0011 | −0.0014 [−0.0040, +0.0011] |

- **Neyman with a fixed per-study total** (51.0) scored +0.0011 on search only.
- **Dropping axial T1/PD** (the search-theory rule) cost MCL and fracture about 0.008 each on
  confirm.

## Value curves

These vary one type's count with the others at the default, on the search half
(`curves_mean.csv`).

- **Near the default, counts barely matter:** from n = 8 to 32, each type moves macro AUC by
  about 0.001, with non-monotonic bumps.
- **Removal:** dropping a fluid type costs 0.006–0.007, dropping coronal or sagittal T1/PD costs
  0.002, and dropping axial T1/PD costs nothing (0.8926 against 0.8924).
- **Marginal gain per triplet at the default**, from a joint bootstrap:

  | Type | Gain per triplet | 95% CI |
  |---|---:|---|
  | Coronal fluid | +0.00029 | [+0.00008, +0.00054] |
  | Sagittal fluid | +0.00028 | [−0.00008, +0.00058] |
  | Axial T1/PD | −0.00025 | [−0.00062, +0.00011] |
  | The other three | about 0 | – |

- **Shape:** a saturating exponential fits better than a logarithm for every type. The macro R²
  for the logarithm is 0.22–0.82, against 0.45–0.95 for the saturating curve. Separable
  logarithmic curves are therefore not supported, so proportional renormalisation is not
  guaranteed to be optimal.
- **Interaction:** moving budget off all T1/PD types at once lost on confirm, although each of
  them alone looks dispensable near the default. The types interact. The interaction curves
  (others at half budget) were not run.

## Missing types

**Simulated removal** (search half, studies that have the type). Redistributing a removed type's
budget within the fixed total recovers only part of the macro loss:

| Removed type | Intact | Budget lost | Best strategy | Recovery |
|---|---:|---:|---|---:|
| Axial fluid | 0.8924 | 0.8861 | Split equally over the fluid types | 0.43 |
| Coronal T1/PD | 0.8937 | 0.8908 | Fill chain (≈ same plane) | 0.65 |
| Coronal fluid | 0.8945 | 0.8876 | Proportional | 0.31 |
| Sagittal T1/PD | 0.8917 | 0.8894 | Proportional | 1.15 |
| Sagittal fluid | 0.8927 | 0.8862 | Proportional | 0.40 |

Proportional redistribution is best or tied for five of six types. Coronal T1/PD's same-plane win
is about 0.0002 and unresolved. Single-type boosts are erratic, and some are worse than losing
the budget. The axial T1/PD gap (0.0005) is too small for a recovery ratio.

**Natural absence of axial T1/PD** (365 search studies). Its 12 triplets were placed by each
strategy:

| Strategy | Macro AUC | Mean triplets | Gain over losing the budget |
|---|---:|---:|---:|
| Proportional | 0.8928 | 61.1 | +0.0026 |
| Split equally over the fluid types | 0.8918 | 61.1 | +0.0017 |
| All to coronal fluid | 0.8914 | 59.7 | +0.0012 |
| Fill as specified (same plane, then fluid) | 0.8892 | 61.1 | −0.0010 |
| All to axial fluid | 0.8888 | 60.2 | −0.0014 |

**As a whole-half rule** (search and confirm):

| Rule | Mean triplets | Search Δ | Confirm Δ |
|---|---:|---:|---:|
| Proportional | 64.0 | +0.0024 [+0.0013, +0.0037] | −0.0001 |
| Proportional, with coronal T1/PD's budget sent to coronal fluid first | 63.9 | +0.0023 | +0.0006 |
| Uniform scaling, 15/15/13/13/13/13 | 65.1 | +0.0023 | +0.0003 |
| Uniform scaling, 15/15/12/12/12/12 | 61.5 | +0.0018 | −0.0006 |

The proportional rule's gain matches spending the same compute uniformly.

## Repeated series in one slot

The 340 search and 320 confirm repeated slots are classified by a header proxy (signature and
matrix size): about half have a different geometry and protocol, about a third share geometry
with a different protocol, 5–9% are retakes and about 9% have minimal headers. Only the repeated
slots change, and scores are on the studies that have one.

| Comparison | Search Δ | Confirm Δ |
|---|---:|---:|
| Top-ranked series only, against the default split | −0.0019 [−0.0042, +0.0004] | **−0.0037** [−0.0065, −0.0012] |
| Other series only, against the default split | −0.0001 | −0.0005 [−0.0032, +0.0021] |
| +2 triplets to the other series, against +2 to the top series | +0.0021 [+0.0001, +0.0042] | +0.0009 [−0.0010, +0.0027] |
| +4 | +0.0006 | +0.0020 [−0.0004, +0.0044] |
| +6 | +0.0018 | +0.0024 [−0.0002, +0.0055] |

- **Findings:**
  - The source ranking does not pick the more valuable series.
  - Losing the second series hurts on both halves.
  - Extra triplets help more from the second series on all six comparisons, although only one CI
    excludes zero.
  - The effect is largest for different-geometry slots (+0.003 to +0.005 on both halves). It
    reverses for minimal-header slots, but those are small groups without bootstraps.
- **Not scored:** a compute-neutral rule that gives repeated slots extra budget.

## Where evidence sits along the slice axis

On the search half, triplets were binned by depth from the study centre (10 mm bins). Local pools
(4.8% of triplets) were excluded. The table gives the strongest single-triplet separation per
label (positive minus negative mean contribution):

| Label | Type | Depth bin (mm) |
|---|---|---|
| ACL, patellofemoral OA | Sagittal fluid | −10..0 |
| Medial OA | Coronal T1/PD | −10..0 |
| MCL, contusion | Coronal fluid | −10..0 |
| Medial meniscus | Coronal fluid | +10..+20 |
| Baker's cyst | Axial fluid | 0..+10 |
| Effusion | Axial fluid | +30..+40 |
| Synovitis | Axial fluid | +40..+50 |

- **Direction of depth:** `build_pool` measures depth along the positive plane axis in DICOM
  patient coordinates: +x (patient left) for sagittal, +y (posterior) for coronal and +z
  (superior) for axial.
- **Reading (inferred):** effusion and synovitis peak in the suprapatellar recess, and the medial
  meniscus at the posterior horn. Sagittal depth depends on which knee it is.
- **Not run:** the Gaussian-width check (30 and 60 mm).

## The allocation-trained checkpoint responds the same

A second cache was built on `allocation-concentration`, the checkpoint trained with random
per-type splits ([result](2026-09-17-allocation-concentration-result.md)), to score allocations
on-distribution. Cache: `outputs/analysis/t/token-cache-allocation-20260917/` (its selections, split
and targets were verified identical to the first cache and reused; 8,368 selections compared with
no mismatch). Its emulated default reproduces the run's saved predictions to 1.3e-4 macro AUC.

- **Nothing confirms here either.** Rule deltas correlate with the fixed-split cache at 0.87 on
  search and 0.79 on confirm. Proportional renormalisation is the only rule positive on both
  halves (+0.0011 and +0.0013, both CIs crossing zero, exact +0.0014 and +0.0012), and it needs
  26% more triplets, which uniform scaling matches.
- **The curves match:** removal costs axial fluid −0.0074, sagittal fluid −0.0057, coronal fluid
  −0.0056, coronal T1/PD −0.0033, sagittal T1/PD −0.0021, axial T1/PD −0.0009, the same ordering
  as the fixed-split checkpoints. Counts from n = 8 to 32 still move macro AUC by about 0.001.
- **Marginal gains stay at a few 1e-4 per triplet.** Only sagittal fluid has a CI excluding zero
  here (+0.00038), against coronal fluid on the fixed-split cache.
- **A 209-allocation coordinate search at the default's compute** ended at 12/9/11/14/8/10:
  +0.0024 [+0.0006, +0.0042] on search, at its detection floor, and +0.0007 [−0.0013, +0.0028] on
  confirm. The same counts give +0.0023 and +0.0001 on the fixed-split cache, so the finalist is
  not specific to this checkpoint.

**Conclusion.** Training on varied splits did not make the model respond differently, so the
insensitivity is a property of the model and task, not of the training allocation. It holds only
inside a band: each type saturates early, so moving triplets between types that all keep roughly 8
or more changes little, while cutting a type to zero or near zero costs 0.006–0.007 macro AUC and
much more on the labels that depend on it (removing axial fluid costs Baker's cyst 0.026;
[type importance](2026-09-17-plane-contrast-importance.md)). The number of triplets in training
does matter ([fewer slices](2026-09-16-restored-code-hyperparameter-runs.md)).

## Caveats

- **Comparisons that change the token count are confounded with bag size.** The model trains on
  about 25 triplets per study and is scored at 50, and the curves here reach 165. A larger bag
  spreads the attention softmax over more tokens than training ever showed, so the flat region
  beyond about 8 triplets per type mixes any information gain with that penalty. The value curves,
  the marginal gains and the uniform-scaling rows are all affected. Rules that keep the total fixed
  are not: the token count is identical on both sides, so the penalty cancels, and the finding that
  reallocation at a fixed total does nothing above the floor stands. Separating the two would need
  the [attention temperature or count-normalised attention](../../backlog/small-changes.md#attention-pooling),
  or training with varied bag sizes, which the allocation run did not do because it kept each
  study's total fixed.
- Every checkpoint was trained on today's allocation. These results measure how a trained model
  responds at test time, not which allocation is best to train with.
- Search-half gains of about +0.002 appeared repeatedly and failed to replicate. Rank candidates on
  one half and require the other.
- The cache approximation grows away from the default allocation. No finalist was rescored with
  `exact.exact_score`, because none survived confirmation.
- Under the time caps, several parts were skipped:
  - the interaction curves;
  - the worst-label variants;
  - confirm-half scores for the separation-share and efficiency rules;
  - per-label recovery, whose small denominators make it unusable;
  - the policy for repeated slots;
  - the placement check.
