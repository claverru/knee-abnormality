# Hypotheses for the validation/public reversal

## 1. Proxy labels and conditional case-mix shift: strongest general concern

Only 11 of the 904 validation studies have gold labels; 893 use report-derived
targets. The official endpoint uses radiologist consensus with explicit severity
criteria: for example, definitive meniscal tears and sufficiently extensive
high-grade cartilage loss, with borderline findings treated as negative.
The training reports and LLM-derived labels need not implement those thresholds
consistently. [Official annotation clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343).

Steven v4 also contains structured imputation, not just independent annotation
noise. The [source audit](../../public-labels/INDEX.md) found that Steven v1 used 0.5
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

## 2. Repeated fold selection and stochastic training: substantial uncertainty

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

## 3. Sharing changes the learning problem: strong mechanism, incomplete attribution

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

## 4. Implementation and inference mismatch: lower priority after the audit

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
