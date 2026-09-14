# Sharing experiments: generalization and inference investigation

Investigated on 2026-09-14 using saved predictions, training archives, submission
records and Kaggle notebook logs. New computation was CPU-only tabular analysis;
no model execution, image decoding, training, GPU queries or tests were run.
The intervention results below come from the previously completed
[compartment diagnosis](sharing-experiments.md#2026-09-13-compartment-attention-diagnosis).

## Conclusions and scope

This investigation preserves the earlier five-model comparison recovered from
`1C` at `3b8c2cc`. The later [reflection/sigma-40 comparison](validation.md#2026-09-14-reflection-sigma-40-result)
added **0.892360 validation AUC and 0.936 public AUC**, selecting independent
heads at sigma 40 as the current baseline. Tables and correlations below retain
the original analysis cohort. The [transfer research](research/medial-lateral-transfer.md)
owns the current shortlist; historical proposals below are not scheduled work.

The subsequent [blinded disagreement and acquisition audit](sharing-disagreement-audit.md)
completed the report-review and metadata follow-up. It finds confident OA source
labels for explicitly low-grade report findings and shows a manufacturer-mixture
change that reverses 1B versus reflection locally. Removing the flagged report
cells does not remove 1B's original local advantage. These results refine the
hypotheses below without identifying the hidden test distribution.

The public reversal is real at the reported score precision, but its hidden
per-target cause is unobserved. The strongest combined explanation is that
weak-label validation rewards some changes that do not transfer to the official
annotation criteria and case mix, while repeated selection on one fold and
different optimization trajectories make close architectural comparisons noisy.
This is a limitation on causal confidence, not on the competitive value of the
score differences. A 0.006 public loss is consequential.

1C does not add an optional location embedding to 1B. It replaces independent
attention scores with a shared constraint, changing both representational
freedom and gradient coupling. Existing interventions show that the learned
factor is useful and that the constraint can preserve a good 1B solution.
Optimization and disease-specific contrast strength therefore deserve more
attention than a categorical conclusion that compartment sharing is bad.

The inference comparisons do not establish that a second GPU failed to improve
hidden-data throughput. Submission completion time mixes execution, queuing and
scoring. The saved visible notebooks contain only three studies. They do expose
substantial startup costs and confirm FP32 execution on both T4s. Static source
inspection identifies duplicated metadata work, shared CPU/I/O limits and
variable study costs as concrete candidates for poor scaling.

## Comparable results

All four original runs use seed 42, the same 3,503 training and 904 validation
studies, frozen Steven-v4/gold targets, ten epochs, reflection probability 0.5
and 2× validation/inference image density. Their saved effective configurations
match except for the experiment name. Architecture is recorded in source and
identity artifacts. Sigma 40 uses 1C with sampling sigma changed from 80 to 40 mm.

| Model | Selected epoch | Local macro AUC | Public AUC | Soft-target BCE |
|---|---:|---:|---:|---:|
| Reflection | 9 | 0.890976 | 0.931 | 0.449310 |
| 1A: meniscus classifier sharing | 9 | 0.891192 | 0.930 | 0.448576 |
| 1B: meniscus and OA classifier sharing | 7 | 0.892022 | 0.925 | 0.453585 |
| 1C: shared disease/compartment attention, sigma 80 | 10 | 0.889430 | 0.928 | 0.448458 |
| 1C, sigma 40 | 10 | 0.892538 | 0.931 | 0.448244 |

The four original models have four reversed pairwise local/public orderings
out of six. Their Pearson correlation is −0.299 and Spearman is −0.400; four
selected observations cannot estimate a stable population correlation. Across
12 distinct submitted checkpoints evaluated on the fixed-label protocol,
Pearson remains 0.937 and Spearman 0.847. Larger historical improvements still
track public performance, but that does not make local AUC a reliable ordering
of the leading models.

This is not the first reversal: the earlier Tonylin experiment improved local
AUC from 0.889506 to 0.892306 while public AUC fell from 0.929 to 0.921.
See the [label comparison](validation.md#2026-09-13-tonylin-label-comparison).
An older learning-rate/training-budget comparison also reversed. The new results
strengthen an existing warning about ranking transfer.

## What actually changed locally

These are absolute per-target AUC differences, not percentage improvements.

| Target | 1A − reflection | 1B − reflection | 1C − 1B |
|---|---:|---:|---:|
| ACL | −0.000710 | −0.002340 | +0.005020 |
| MCL | −0.006009 | −0.002152 | −0.002966 |
| Medial meniscus | +0.002737 | +0.008514 | −0.009442 |
| Lateral meniscus | +0.003406 | −0.001533 | +0.003538 |
| Medial OA | +0.002178 | +0.001290 | −0.001591 |
| Lateral OA | −0.006090 | −0.002597 | −0.009507 |
| PF OA | +0.008111 | −0.001375 | +0.002117 |
| Effusion | −0.004779 | −0.001065 | +0.009986 |
| Synovitis | +0.001499 | −0.000110 | −0.005687 |
| Baker's | −0.004818 | −0.007664 | +0.006669 |
| Contusion | +0.006545 | +0.013890 | −0.011471 |
| Fracture | +0.000528 | +0.007701 | −0.017779 |
| Macro | +0.000217 | +0.001047 | −0.002593 |

1B beats reflection on four targets and loses on eight. Excluding contusion and
fracture, the remaining ten-target mean falls by 0.000903. Its shared OA pair's
mean is below reflection, although it improves over 1A. OA contributes roughly
26% of the net 1B-over-1A macro gain; most of that gain comes from other targets.
The local result therefore does not demonstrate broad improvement in the
intended anatomical task. It also does not reveal which targets lost on Kaggle:
hidden per-target scores and labels are unavailable.

1C's change affects the whole trained representation. The other eight targets
account for about 45% of its net macro loss versus 1B, although their attention
parameterization is unchanged. Contusion and fracture alone account for about
94% of the net macro loss, with substantial gains and losses cancelling among
the other targets. These overlapping decompositions describe different groups;
they are not independent causal attributions.

Direct medial/lateral discrimination also differs from full-target AUC:

| Exclusive-positive subset | Reflection | 1A | 1B | 1C |
|---|---:|---:|---:|---:|
| Meniscus logit-margin AUC, 344 studies | 0.972321 | 0.972039 | 0.980173 | 0.972208 |
| OA logit-margin AUC, 198 studies | 0.856707 | 0.872982 | 0.870184 | 0.854291 |
| Meniscus positive target ranked higher | 322/344 | 322/344 | 325/344 | 322/344 |
| OA positive target ranked higher | 156/198 | 159/198 | 160/198 | 156/198 |

For OA, 1C correctly ranks 33/55 lateral-only studies versus 22/55 for 1B,
but 123/143 medial-only studies versus 138/143. It trades some medial sensitivity
for lateral sensitivity. Comparing probabilities across targets includes their
different biases; this is a diagnostic, not the competition's per-target AUC.
Neither pair improves uniformly under 1C. Both exclusive-positive subsets have
only three gold studies, so they retain the weak-label limitation.

## Hypotheses for the validation/public reversal

### 1. Proxy labels and conditional case-mix shift: strongest general concern

Only 11 of the 904 validation studies have gold labels; 893 use report-derived
targets. The official endpoint uses radiologist consensus with explicit severity
criteria: for example, definitive meniscal tears and sufficiently extensive
high-grade cartilage loss, with borderline findings treated as negative.
The training reports and LLM-derived labels need not implement those thresholds
consistently. [Official annotation clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343).

Steven v4 also contains structured imputation, not just independent annotation
noise. The [source audit](public-label-audit.md) found that Steven v1 used 0.5
for unaddressed findings, later versions inferred unmentioned synovitis partly
from effusion, and v4 averages another label source without using its confidence.
Many originally unaddressed findings become 0.25. A shared representation can
learn these disease/report associations successfully while becoming less useful
for independently adjudicated image findings. Contusion/fracture gains and the
prior Tonylin reversal make this hypothesis relevant, but do not prove it.

CPU-only sensitivity analysis gives mixed evidence:

| Evaluation subset | Reflection | 1A | 1B | 1C | Sigma 40 |
|---|---:|---:|---:|---:|---:|
| Full frozen validation | 0.890976 | 0.891192 | 0.892022 | 0.889430 | 0.892538 |
| Target score ≤0.1 or ≥0.9 | 0.932804 | 0.933219 | 0.932783 | 0.932000 | 0.933301 |
| Original report-source coverage, plus gold | 0.910990 | 0.911445 | 0.911599 | 0.910515 | 0.912099 |
| Coverage plus label-source agreement, plus gold | 0.921819 | 0.921905 | 0.922071 | 0.921863 | 0.922925 |

1B's advantage over reflection disappears on extreme-score labels; its contusion
gain reverses from +0.013890 to −0.002257 there. But its advantage survives the
original-coverage and source-agreement restrictions. Thus unmentioned or
ambiguous labels cannot yet explain the entire result. These post-hoc subsets
change the case mix separately for each target, and source confidence is not
calibrated clinical truth. They must not replace frozen validation as a new
selection metric. Only six validation cells are exactly 0.5, so the literal
choice of which side receives exactly 0.5 is not the main issue.

The official data description also warns of heterogeneous international scanners,
protocols and populations. Changes in severity, disease combinations, acquisition
quality or site-specific appearance can change which representation generalizes.
[Official data description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data).
Report grouping does not establish patient/site independence. Pure prevalence
shift alone cannot change population ROC AUC when class-conditional score
distributions stay fixed; conditional case mix and finite sample uncertainty
are the relevant mechanisms. The gap between local 0.89 and public 0.93 is not
itself evidence of a defect: the populations and annotation procedures differ.

The 11-study gold subset is too fragile to resolve this. It actually ranks 1B
highest among the original four, despite its worst public score. Previously
masking unmentioned training labels also reduced local AUC by 0.011455; blindly
repeating that intervention is not supported by this investigation.

### 2. Repeated fold selection and stochastic training: substantial uncertainty

Paired bootstrap resampling of the 852 normalized report groups, 2,000 draws
with seed 20260914, gives the following fixed-model validation uncertainty:

| Comparison | Observed macro delta | 95% bootstrap interval |
|---|---:|---:|
| 1A − reflection | +0.000217 | [−0.002999, +0.003164] |
| 1B − reflection | +0.001047 | [−0.002366, +0.004585] |
| 1B − 1A | +0.000830 | [−0.002697, +0.004336] |
| 1C − 1B | −0.002593 | [−0.006213, +0.001021] |
| Sigma 40 − original 1C | +0.003109 | [−0.000561, +0.006643] |

Every interval crosses zero. That does not make the observed changes unimportant;
it means one validation sample does not establish their architectural
repeatability. The bootstrap does not include training-seed variance, repeated
model/epoch selection or hidden-data shift. Reweighting report groups or removing
the largest 37-study group preserves 1B > 1A > reflection > 1C, so duplicate
report weighting is not a sufficient explanation.

Seed 42 does not make these runs fully paired. Changed parameter shapes consume
different random draws before later learned embeddings are initialized. The
loader has no independent shuffle generator, so architecture construction can
also change subsequent batch order. The pretrained backbone initialization policy
is shared, but learned components and stochastic trajectories need not be.
Deterministic training is disabled. This makes target changes in unrelated heads
less surprising and prevents attributing every difference to one mathematical
constraint.

1B selects epoch 7 at AUC 0.892022/BCE 0.453585, drops to AUC 0.886732 at epoch 8,
and finishes at AUC 0.891646/BCE 0.447290. Reflection selects epoch 9. The selected
1B checkpoint is less well calibrated to the soft targets even though it ranks
them better. This supports investigating checkpoint selection and training
stability, not asserting that 1B necessarily overfit: its final-epoch AUC still
exceeds reflection's final epoch. Monotonic bias/temperature calibration alone
cannot repair per-target ROC AUC.

### 3. Sharing changes the learning problem: strong mechanism, incomplete attribution

The intuitive common disease detector is plausible, but the optimal discriminative
classifier can differ with anatomy, background tissue, disease severity and
visual context. A shared classifier assumes that independent attention can
transform both compartments into a compatible feature representation. Study-level
labels do not teach that localization directly. Medial examples are more numerous,
so their gradients can dominate the shared detector; independent biases can
handle priors but cannot recover compartment-specific feature transformations.
This affects learning, even though macro AUC weights all twelve targets equally.

For 1C, the precise score equations are:

```text
meniscus medial: h · d_men + h · c
meniscus lateral: h · d_men - h · c
OA medial:       h · d_oa  + h · c
OA lateral:      h · d_oa  - h · c
```

Four independent 128-dimensional scoring vectors become three. Both families
must have the same token-wise medial/lateral contrast, 2 h·c. Each target still
normalizes attention separately. The constraint concerns evidence selection,
not whether medial and lateral disease may coexist.

Setting c to zero makes the two attention distributions within each family
identical. Their pooled features and shared classifier weights then also match,
so their logits differ only by fixed biases. The within-study logit margin is
constant and cannot distinguish medial-only from lateral-only cases. Therefore
the model cannot ignore c and recover 1B. Even a genuinely optional added feature
would not guarantee identical generalization after finite stochastic training;
here the previous solution space has explicitly changed.

The factor c receives signed gradients from all four tasks. The disease terms
receive paired sums. This couples optimization and changes its scale/geometry;
AdamW is not invariant to this reparameterization. Under plain SGD, effective
score updates contain both amplified own-task terms and contrast gradients from
the other family. AdamW's adaptive updates make an exact effective learning-rate
claim inappropriate. This is not a claim that decoupled weight decay becomes
weaker simply because the variables were factorized.

The saved interventions discriminate among these mechanisms:

| Saved intervention | Macro AUC |
|---|---:|
| 1C intact | 0.889430 |
| 1C, compartment factor zeroed | 0.872267 |
| 1C, half-strength factor | 0.885816 |
| 1C, double-strength factor | 0.889335 |
| 1C, reversed factor | 0.839237 |
| Trained 1B projected to share its meniscus contrast across both families | 0.892022 |

The last projection changes AUC by only −0.000000081 and improves BCE. A useful
shared-contrast solution therefore exists for a good learned representation;
strict capacity loss alone is an incomplete explanation for the trained 1C
regression. This is a post-training projection, not a matched retraining ablation.

1B already learns highly correlated within-study compartment score patterns
(median correlation 0.946), but different contrast strengths: median score SD
0.781 for meniscus versus 0.474 for OA. 1C forces both to 0.784. Its OA attention
becomes more separated and its meniscus attention less separated than 1B's.
Doubling c improves mean meniscus AUC by 0.002087 while worsening mean OA AUC
by 0.002656. This specifically motivates investigating disease-dependent contrast
strength and joint-training interference. Learned attention is not an anatomical
ground-truth map, so these statistics do not prove accurate localization.

Finally, sigma 40 reaches 0.931 public with the same 1C architecture. The factor
is compatible with competitive public performance. Sampling/representation
interaction is a credible lead, although one sigma comparison does not isolate
the interaction from ordinary sampling benefits or stochastic variation.
Original 1C also improves public AUC by 0.003 over 1B while losing locally.
Beneficial regularization is therefore another live interpretation: the
constraint may suppress features that rank weak labels well but transfer poorly
to adjudicated findings. Explaining the local regression does not establish
that it represents a clinical generalization regression. The observed public
ordering supports this possibility without identifying its mechanism.

### 4. Implementation and inference mismatch: lower priority after the audit

The saved prediction tables have matching IDs, frozen targets and official target
ordering. All four exported bundles pass file-integrity verification. Reflection,
1B and 1C package source matches their training archives; 1A retains the archived
model and preprocessing with the new inference orchestration overlaid.
The previous 1B/1C diagnosis also reproduced saved intact probabilities within
1.5e-7 using identical selected-input hashes.

The factorization changes no input geometry. Reflection preserves anatomical
medial/lateral target identities while transforming the appropriate image and
position information together. There is no positive evidence of a new label-swap
or target-order bug. This does not certify unseen pathological cases or complete
single-/multi-GPU numerical parity. However, 1B's public regression occurred on
the old single-GPU path, so the new coordinator cannot explain it.

All four Kaggle runs use FP32; local training/validation uses BF16. That shared
difference is worth a later controlled parity check but is a weak explanation
for the specific 1B reversal without evidence of unusual sensitivity.

## Inference: distinguish the measured clocks

| Model | GPUs used | Loader workers, total | Studies per GPU batch | Submit → observed completion | Visible notebook measured total |
|---|---:|---:|---:|---:|---:|
| Reflection | 1 T4 | 8 | 2 | 13m 36s | 54.22s |
| 1A | 2 T4 | 4, two per GPU | 4 | 16m 07s | 90.62s |
| 1B | 1 T4 | 8 | 2 | 14m 36s | 51.98s |
| Original 1C | 1 T4 | 4 | 4 | 17m 38s | 54.57s |

Submission wait includes unknown queue/evaluation delays and approximately
30-second polling granularity. It is not the efficiency-prize runtime. Kaggle
staff explicitly define that runtime as notebook execution from start to end,
including setup, model loading and DICOM reading; GPUs are eligible.
[Staff clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733475).
The visible notebook totals are our instrumented portion of that execution,
not an official hidden-data runtime measurement.

All visible logs process three example studies; the competition describes roughly
1,300 hidden test studies. The 1A example partitions as two studies plus one,
never filling either batch limit of four. It establishes use of both T4s and
successful output, not sustained scaling. The failed original-1C batch-8 rerun
has no usable timing or score; memory pressure is plausible but its hidden error
does not establish an OOM diagnosis.

### Ranked runtime hypotheses

1. **Shared CPU and storage supply limits scaling.** Each GPU process has two
   loader workers, versus eight for the old single-GPU configuration. Two GPUs
   do not double the machine's CPU or storage bandwidth. DICOM enumeration,
   header parsing, decoding, geometry and tensor preparation can leave either
   GPU waiting. Study complexity and codec variation make hidden-data performance
   different from the three examples. This is a strong source-based hypothesis,
   not a measured fraction of hidden runtime.
2. **Startup and duplicated work offset compute gains.** The coordinator imports
   the model/inference stack, verifies the bundle and selects metadata before
   starting its worker timer. Each worker then initializes another interpreter,
   dependencies, dataset and model. On 1A, worker wall is 24.64s but the full
   prediction subprocess is 83.65s: about 59.01s lies outside that worker timer.
   The present logs cannot allocate that gap among imports, parent setup and
   shutdown. Each child builds an index for the entire test series metadata and,
   without a saved test image-count inventory, enumerates all series directories
   to count DICOMs even though study assignments are disjoint. Later reads enumerate
   selected series again. The duplication is confirmed statically; its cost is
   unmeasured on hidden data.
3. **Studies per batch is a poor compute unit.** A study expands into many
   triplet images and spatial tokens. Batch two may already provide a large CNN
   workload; doubling studies need not improve occupancy. It increases memory,
   preprocessing task granularity and waiting behind expensive studies. The
   current dense study-membership attention calculation also increases its
   masked work faster than linearly with study batch size, although the backbone
   may dominate. The share of time in this pooling is unknown.
4. **Static sharding leaves an uneven tail.** Study-count partitioning ignores
   acquisition counts, slice/header counts, codecs and image/token workload.
   Completion waits for the slower process. This is obvious with the two-plus-one
   example; the size of the effect on the hidden cohort is unmeasured. Persistent
   per-worker caches help repeated local validation more than a cold one-pass
   hidden run.
5. **T4 compute is being used in FP32.** Automatic precision selects native BF16
   when supported and otherwise FP32; it does not select FP16 on T4. This is
   confirmed in every saved Kaggle log. FP16 is a concrete optimization candidate:
   T4 has substantially greater mixed-precision Tensor Core peak throughput than
   FP32 throughput. Peak hardware figures are not an application speedup prediction.
   [NVIDIA T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf).

The small sharing heads do not materially reduce the expensive image backbone,
triplet count or spatial-token count. Replacing these heads should not itself
deliver a substantial inference speedup. Local saved validation prediction times
of roughly 32–34 seconds for 904 studies use different GPU hardware/precision and
a different cache context; they cannot forecast cold T4 notebook execution.

## Public subset size and case-mixture follow-up

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

### Completed CPU-only checks

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

The [archived case-mixture command](history/sharing-sources/analyze_sharing_case_mix.py.txt)
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

### Decision order

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
   [disagreement audit](sharing-disagreement-audit.md), including a representative
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
explicitly deferred to [future work](future-work.md#kaggle-gpu-profiling).

Artifacts: `outputs/analysis/sharing-case-mix-20260914/analysis.json`, archived
command and helper source. The command and its CPU-only reproduction recipe are
preserved with [historical source requirements](history/sharing-sources/README.md).

## Proposed discriminating work

These historical proposals explain how the investigation could discriminate
between causes. The [later research shortlist](research/medial-lateral-transfer.md)
supersedes their priority order. Current code uses independent heads at sigma 40;
no training, GPU diagnostics or submissions are scheduled by this record.

### Generalization and architecture

1. Preserve frozen full-fold AUC, but report its target contributions, report
   coverage/source agreement, study complexity and available site/protocol strata
   alongside it. Predetermine these diagnostics before the next architectural
   comparison; do not keep selecting whichever post-hoc subset agrees with public.
   Inspect disagreement cases, especially contusion/fracture and lateral OA,
   against report text and explicit official severity definitions. This can expose
   labeling mechanisms without pretending that another LLM is gold truth.
2. When training resources become available, separate initialization and data-order
   randomness, copy identical shared initial tensors into matched comparisons and
   compare fixed epochs as well as selected best epochs. A matched seed comparison
   is the first step; additional seeds/folds would resolve different uncertainties
   but remain deferred under the current project strategy. An independent
   adjudicated validation panel would address endpoint validity more directly than
   more report-derived folds, if it becomes feasible.
3. The most informative targeted architecture experiment is to start from the
   saved 1B representation and compare its independent scorer against the known
   near-lossless shared-meniscus-contrast projection, with matched fine-tuning
   conditions. Measure whether the constrained solution deteriorates during
   adaptation. This distinguishes a difficult training path from immediate
   capacity loss; it is a new experiment, not a replacement for the original
   from-pretrained comparison.
4. A separate candidate is shared anatomical contrast with disease-specific
   strength, motivated by 1B's 0.781 versus 0.474 contrast SD and the opposing
   meniscus/OA response to scaling. It would relax 1C's identical-strength
   assumption. Also measure per-family gradient norms/alignment when GPU diagnosis
   is available before choosing learning-rate or gradient-balance changes.
5. The original analysis kept reflection/sigma 80 as the independent-head reference
   and sigma-40 1C as an equally scoring public candidate. The completed
   reflection/sigma-40 run later surpassed both publicly. Do not restore 1B merely because its original
   fold AUC is higher, or reject all compartment sharing because original 1C loses
   locally. Compare architecture at matched sampling before attributing sigma 40's
   benefit specifically to the factor. Avoid using further public probes as an
   unlimited substitute for validation.

### Efficiency

Deferred by user decision after this investigation. The
[future-work entry](future-work.md#kaggle-gpu-profiling) owns resumption; preserve
the current inference notebook and configuration in the meantime.

Create one Kaggle profiling notebook with a fixed checkpoint and a representative
set of training studies available on Kaggle, including expensive study/codec
tails. The local machine does not need two GPUs for that controlled measurement.
Use the same source, precision and preprocessing for a 1-GPU/2-GPU ×
2-study/4-study batch matrix with four total loader workers; add the old
1-GPU/eight-worker condition separately. Record startup and both cold and warm
passes, counterbalance run order and repeat enough to see machine variability.
This does not require consuming scored competition submissions.

Measure imports/setup, full metadata indexing, per-batch loader wait, DICOM/header
work, transfer, CNN, pooling, write and per-GPU completion tails. Record actual
images/tokens and study counts, not only study batch size. Use CUDA events or
synchronization for device timing because GPU launches are asynchronous.
[PyTorch timing guidance](https://docs.pytorch.org/docs/main/notes/cuda.html).

If the loader dominates, prioritize one shared/prefiltered metadata inventory,
eliminating repeated header/enumeration work, CPU-budget-aware preparation and
overlap with GPU work. If partition tails dominate, use cost-aware assignment or
a dynamic work queue. If GPU compute dominates, measure FP16 with sensitive
geometry/pooling operations retained in FP32 and validate saved prediction AUC
and numerical stability before adopting it. Batch or chunk by actual image/token
budget to decouple study size from peak CNN memory.

ONNX/TensorRT can accelerate the regular 224×224 image backbone, but do not
eliminate Python DICOM reading, startup or variable-study aggregation. The backbone
is the natural first export boundary; dynamic image batches need suitable engine
profiles. Choose that work after measuring the time fraction it can affect.
For example, if only 30% of notebook time benefits from a second GPU, even ideal
halving of that part gives only 1/(0.7 + 0.3/2) ≈ 1.18× overall speedup, before
extra process costs.

Efficiency also rewards accuracy. With D = maxAUC − benchmark, the official
minimized score is −AUC/D + runtime/32400. An AUC improvement δ has the same score
benefit as saving 32400δ/D seconds. For a hypothetical D = 0.45, +0.001 AUC is
worth 72 seconds and +0.006 is worth 432 seconds. This example does not assume
the final private maximum is known. Optimizing runtime while sacrificing several
AUC decimals may lose the efficiency prize as well as leaderboard position.
[Official efficiency formula](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation).

## Reproduction and artifacts

The [archived CPU analysis command](history/sharing-sources/analyze_sharing_generalization.py.txt)
requires its [original runtime](history/sharing-sources/README.md), not current
`main`. It joins the saved predictions to frozen
targets, checks study identity/probabilities, calculates target/subset metrics,
resamples report groups and fingerprints its inputs. It refuses an existing
output directory:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked python -m scripts.adhoc.analyze_sharing_generalization --output outputs/analysis/sharing-generalization-new
```

The completed evidence is in
[`outputs/analysis/sharing-generalization-20260914/`](../outputs/analysis/sharing-generalization-20260914/):

- `analysis.json`, `target_metrics.csv` and the archived analysis source: primary
  comparison, bootstrap, epoch curves, input hashes and submission timing.
- `report_evidence_subsets.json`: supplementary target-specific coverage and
  source-agreement restrictions, with label-source hashes. Coverage means gold
  or Steven-v1 score unequal to 0.5; agreement additionally requires the original
  Steven-v1 and lixin target classifications to agree at 0.5. Evaluation still
  uses the frozen target, not a substituted label.
- `expanded_correlation.json`: the historical 12-checkpoint fixed-label cohort.
- `bundle_audit.json`: source/config/input-identity and export-integrity comparison.
- `notebook_timings.json`: structured measurements extracted from saved visible
  Kaggle notebook logs. These supplementary files are separate derivations; the
  main CLI regenerates the primary comparison, not those auxiliary audits.

Earlier GPU interventions and their intact-prediction parity are preserved in
`outputs/analysis/compartment-diagnosis-20260913/`. No intervention was rerun for
this investigation. New script lint/format and catalog checks are static checks,
not model correctness tests or performance benchmarks.
