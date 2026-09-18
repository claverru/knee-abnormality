# Public subset size and case-mixture follow-up

The user highlighted the public fraction and possible OA/meniscus distribution
shift on 2026-09-14. This changes the weight placed on public rankings: the
earlier analysis did not give public sample size enough explicit attention.

The [official leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)
uses approximately 30% of test data, with the other 70% determining final results.
Together with the [official approximately 1,300-study test count](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data),
this implies about 390 public and 910 private studies. The training corpus has
4,407 studies, of which our fold uses 3,503 for training and 904 for validation.
The submission duration gives no evidence that the test corpus is larger than
training. A biased training sample remains possible regardless of relative sizes.

Our local fold can have less sampling noise than the public subset. Under
comparable class balance, score distributions and independence, a rough
inverse-square-root sample-size comparison gives about 1.52 times more standard
error at 390 than 904 studies. That approximation is not an AUC uncertainty
estimate for the actual hidden cohort. More local studies do not remove weak-label
bias, while adjudicated public labels do not remove public sampling variation.
Public/private membership and their site/case-mix relationship are not known.
Neither local nor public rankings should be treated as the private result.

## Completed CPU-only checks

The local training and validation marginals are similar, consistent with the
stratified split:

| Frozen target ≥0.5 | Training, 3,503 studies | Validation, 904 studies |
|---|---:|---:|
| Medial meniscus | 40.68% | 39.38% |
| Lateral meniscus | 15.82% | 15.27% |
| Medial OA | 37.34% | 36.39% |
| Lateral OA | 27.43% | 26.66% |
| PF OA | 45.99% | 44.91% |

These are effective-label rates, not measured clinical prevalence. Their agreement
shows that validation represents these training-label marginals; it says nothing
about hidden distribution agreement. The 58 gold studies have different rates,
but their annotation/selection and size prevent using them as test-set priors.

The [archived case-mixture command](../sharing-sources/analyze_sharing_case_mix.py.txt)
generated 2,000 paired 390-study bootstrap cohorts with replacement from the
saved 904-study predictions, seed 20260915. Every model uses the same sampled
studies. This isolates sampling variation under the empirical local case/label
distribution, with IID study sampling rather than the earlier report-group
bootstrap. It excludes group dependence, label shift, training variation and
model-selection uncertainty.

| Comparison | Fraction reversing the full-fold ranking | Fraction reaching the observed rounded public delta or farther |
|---|---:|---:|
| 1A versus reflection | 47.25% | 30.50% at ≤−0.001 |
| 1B versus reflection | 34.15% | 0.50% at ≤−0.006 |
| 1C versus 1B | 19.20% | 2.45% at ≥+0.003 |

These are empirical sensitivity frequencies, not probabilities that Kaggle's
result arose from chance and not formal significance tests. The score thresholds
come from rounded public results. Ranking flips are unsurprising at this sample
size, but the 1B loss magnitude is not typical under the observed local
distribution. Sampling variation deserves more weight without becoming a complete
explanation. Equivalent 910-study resampling is retained in the artifact, but
does not predict private scores.

Next, fixed predictions were evaluated under hypothetical weights for the four
compartment groups (neither, medial-only, lateral-only, both), any disease, and
meniscus/OA co-occurrence. All twelve targets remain equally weighted in macro
AUC; study weights alter which positive/negative cases each target sees.

- Quadrupling lateral-only OA case weight narrows the 1C-minus-1B macro deficit
  from **−0.002593 to −0.000858**. This changes that subgroup's weighted population
  share from 6.1% to about 20.6%, so it is a substantial hypothetical shift.
- Across all 14 examined alternative mixtures, 1B still exceeds reflection in
  macro AUC and 1C still trails 1B. The tested frequency changes alone do not
  reproduce the observed public ranking.
- 1A versus reflection and sigma-40 1C versus reflection do reverse under some
  mixtures. Model selection is demonstrably sensitive to case composition.
- Conditional target performance reveals why OA needs scrutiny. Lateral OA AUC
  for 1B is **0.808712 without meniscal disease** versus reflection's **0.800991**,
  but **0.869769 with meniscal disease** versus reflection's **0.881079**.
  These strata contain 485/419 studies and 76/165 lateral-OA positives.
  Sharing's effect changes sign with coexisting disease.
- Of 241 lateral-OA-positive validation studies, **186 also have medial OA**.
  Co-occurrence is substantial. It can provide useful context or a shortcut;
  correlation alone does not distinguish the two. An isolated lateral-OA detector
  and a predictor relying partly on medial disease may transfer differently.

The strongest version of the distribution hypothesis is therefore conditional:
different severity, unilateral/bilateral presentation, coexisting disease,
acquisition populations or label criteria. A pure change in a target's positive
fraction, keeping its positive and negative case distributions fixed, cannot
change its population ROC AUC. Changing other diseases' frequency can change
those conditional case distributions and therefore can change AUC.

## Decision order

1. Keep frozen full-fold AUC as the development metric and retain public results
   as external evidence. Do not choose whichever one favors a preferred model.
   The completed resampling and case-mixture checks precede new training and
   provide a fixed diagnostic set for future candidates.
2. Next inspect a compact, predefined set of OA/meniscus disagreement cases,
   balanced across compartment/co-occurrence groups and across which model is
   favored. Compare report evidence and official severity definitions before
   viewing the model identity. Include uncertain/implicit labels and a random
   control sample. Report review can expose proxy-label problems but cannot
   adjudicate the images or estimate hidden prevalence. Inspect site/protocol
   stratification using metadata alongside this review; do not use site or
   report language as ground-truth diagnosis. This step is now completed in the
   [disagreement audit](oa-and-meniscus-disagreement-audit.md), including a representative
   header inventory for all 4,407 studies. The previous 128-study cache, with
   only 29 validation studies, is no longer the basis for acquisition analysis.
3. Choose one training intervention from that evidence. If the problem is
   supervision severity/coverage, change that supervision only. If shared heads
   lose isolated/discordant compartment cases while gaining common co-occurring
   cases, compare a targeted sharing or training-weight change at fixed sampling,
   source labels, initialization policy and epoch rule. Do not rebalance to an
   invented hidden prevalence or change labels, sampling and architecture together.
   Matched repeat training addresses optimization noise; another report-derived
   fold addresses case sampling but does not establish the clinical endpoint.

Hidden distribution shift cannot be proved from aggregate public scores and
three visible test studies. These checks can establish a failure mechanism and
its robustness locally; proving the hidden frequencies requires evidence we do
not have. Current inference code remains unchanged. Kaggle GPU profiling was
explicitly deferred; it resumed as [loading speedup](../../backlog/loading-speedup.md).

Artifacts: `outputs/analysis/sharing-case-mix-20260914/analysis.json`, archived
command and helper source. The command and its CPU-only reproduction recipe are
preserved with [historical source requirements](../sharing-sources/README.md).
