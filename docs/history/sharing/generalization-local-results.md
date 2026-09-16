# Sharing generalization: comparable and local results

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
See the [label comparison](../../validation/results/2026-09-13-tonylin-label-comparison.md).
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
