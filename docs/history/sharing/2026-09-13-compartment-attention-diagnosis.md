# 2026-09-13 Compartment attention diagnosis

The learned medial/lateral factor is useful to the trained 1C model: removing it
reduces full-fold macro AUC from **0.889430 to 0.872267**, and reversing it reduces
AUC to **0.839237**. The observed 1C regression cannot be explained as an unused
or simply harmful anatomical signal. Moreover, imposing a shared contrast on the
already-trained 1B representation can preserve **0.892022 AUC**. That distinguishes
the effect of sharing at inference from the outcome of training 1C end to end.

This user-requested diagnostic evaluated all 904 held-out studies once per model,
reusing each batch's backbone features for every attention intervention. Saved
configs, checkpoint and input identities were enforced; 1B used its archived
source and 1C used its matching source at the time. Actual selected-path and batch
tensor hashes match across models. Intact probabilities reproduce saved outputs
within **1.48e-7 for 1B** and **1.34e-7 for 1C**, with identical AUC. No weights
were updated, and there was no new training or production code change.

## What the component changes

The component is a shared 128-dimensional attention-scoring vector, rather than
an added spatial input embedding. The shared hidden feature `h` supplies a disease
score `d` and compartment score `c`; the four affected scores are `d+c` for medial
and `d-c` for lateral. Thus 1C enforces exactly the same token-level medial/lateral
score contrast for meniscus and OA. Each target still has its own softmax and
pooled features, and the meniscus/OA classifier pairs have separate shared weights.

With `c=0`, each pair has identical attention and pooled features. Because its
classifier weights are also shared, its two logits differ only by their fixed
biases. Within-study anatomical evidence therefore disappears. The analytically
constant margin has AUC 0.5; float32 subtraction creates numerical jitter of up
to 5.85e-6 in the raw saved logits, whose apparent ranking must not be interpreted.
The summary records analytically constant margins explicitly. Reversing `c`
swaps the paired attention distributions, explaining the severe performance loss.

## Effects on the trained 1C checkpoint

| Intervention | Macro AUC | Meniscus mean AUC | OA mean AUC |
|---|---:|---:|---:|
| Intact | 0.889430 | 0.906810 | 0.880112 |
| Remove both factors | 0.872267 | 0.821353 | 0.862591 |
| Half strength | 0.885816 | 0.890957 | 0.874285 |
| Double strength | 0.889335 | 0.908897 | 0.877456 |
| Reverse sign | 0.839237 | 0.655433 | 0.830331 |
| Remove meniscus factor only | 0.875187 | 0.821353 | 0.880112 |
| Remove OA factor only | 0.886509 | 0.906810 | 0.862591 |

The other eight targets' probabilities are unchanged under every intervention.
Doubling the contrast improves meniscus mean AUC by 0.002087 but reduces OA mean
AUC by 0.002656; halving it reduces both. These observations show different
responses across disease families, but none of the measured common-strength
changes reaches 1B's full-fold AUC. They are diagnostic interventions, not a
held-out-independent search for a deployable scale.

## Does 1B need different contrasts?

For each disease in 1B, decompose its independent attention scores into their
average disease term and half-difference compartment term. Preserve both disease
terms and all other scores, then substitute a common compartment term. This
changes only attention on the already-trained 1B features.

| 1B intervention | Macro AUC | Meniscus mean AUC | OA mean AUC | OA margin AUC |
|---|---:|---:|---:|---:|
| Independent contrasts | 0.892022 | 0.909762 | 0.885661 | 0.870184 |
| Share average of contrasts | 0.891603 | 0.906361 | 0.886547 | 0.878957 |
| Share meniscus contrast | 0.892022 | 0.909762 | 0.885661 | 0.881500 |
| Share OA contrast | 0.890229 | 0.899003 | 0.885661 | 0.870184 |

Sharing 1B's meniscus contrast changes macro AUC by −0.000000081, effectively
preserving the recorded 0.892022 value. Its BCE improves 0.453585 → 0.452536 and
OA margin AUC improves 0.870184 → 0.881500, with correct medial-only rankings
138/143 → 137/143 and lateral-only 22/55 → 25/55. Conversely, forcing the weaker
OA contrast onto meniscus reduces its full-fold mean AUC by 0.010759. The shared
mean is a compromise, gaining 0.000886 OA mean AUC but losing 0.003402 meniscus
mean AUC. These measured projections show that sharing alone is not sufficient
to explain the trained 1C regression.

## Learned strength and attention maps

In 1B, the two raw compartment weight vectors have cosine similarity **0.332**,
but their actual within-study token-score patterns have median Pearson correlation
**0.946**. Weight-space angle alone would overstate how differently they behave on
the learned features. 1C's score-pattern correlation is exactly 1 by construction.

| Median across 904 studies | 1B | 1C |
|---|---:|---:|
| Meniscus compartment score SD | 0.781 | 0.784 |
| OA compartment score SD | 0.474 | 0.784 |
| Meniscus medial/lateral attention overlap | 0.252 | 0.294 |
| OA medial/lateral attention overlap | 0.555 | 0.398 |

Overlap is the sum of the minimum of the two normalized attention weights at each
valid token. 1C makes the OA distributions less overlapping while making meniscus
more overlapping. The factor carries substantial information, but its score SD is
below the disease-score SD in both 1C families (1.179 meniscus, 1.270 OA); vector
norm alone does not establish that it dominates the model.

Intervention figure (`outputs/analysis/c/compartment-diagnosis-20260913/intervention-effects.png`, deleted)
(PDF (`outputs/analysis/c/compartment-diagnosis-20260913/intervention-effects.pdf`, deleted))
compares AUC effects and learned contrast strength.
Attention examples (deleted)
(PDF (`outputs/analysis/c/compartment-diagnosis-20260913/attention-examples.pdf`, deleted)) show
identical prepared coronal inputs under 1B and 1C. Each selected study changed its
OA ranking outcome; the displayed triplet maximizes combined 1B+1C OA attention
mass among coronal inputs. Positive relative contrast favors medial and negative
favors lateral. Attention overlays use actual stride-32 feature anchors and paired
color scales. They illustrate routing changes, not reference lesion annotations
or a representative localization accuracy estimate. Four example arrays and exact
selection records are preserved in the artifact directory.

## Interpretation and limits

Of the 1C-versus-1B macro AUC loss of 0.002593, **0.001176 (45%)** comes from the
eight targets whose attention scorers were not factorized. These outputs are
unchanged by the inference interventions. End-to-end training therefore altered
behavior beyond the four paired outputs. Changed optimization, initialization
and learned shared features are plausible contributors; these diagnostics do not
identify a unique training cause. Both runs have the same pretrained backbone
identity and seed, but changed parameter initialization consumes different random
draws. One run per architecture cannot separate these effects.

The inference projection retaining 1B AUC and 1C's useful learned compartment signal
argue against concluding that medial/lateral information or all shared anatomical
preference is inherently harmful. The full trained 1C model lost on the
primary local AUC metric, which initially favored 1B. Its later public loss
superseded that recommendation. Current development uses reflection/sigma 40;
these interventions remain evidence about the historical trained models.

The summary (`outputs/analysis/c/compartment-diagnosis-20260913/summary.json`, deleted)
links both raw result hashes and records corrected constant-margin metrics,
selected-input agreement, weight/attention summaries and the loss decomposition.
Each run directory preserves per-intervention predictions/logits, source snapshot,
config/checkpoint/input hashes and per-study attention diagnostics. Evaluation used
BF16, batch 8, eight workers and seed 42; the measured diagnostic loops took
37.67s for 1B and 41.20s for 1C. No confidence interval or training-seed replication
was computed. This analysis measures trained-model reliance, not a causal estimate
of a retrained model's benefit. The attention pooling follows the learned weighted
aggregation described by [Ilse et al.](https://proceedings.mlr.press/v80/ilse18a.html);
the disease/compartment constraint and these intervention findings are specific to
this repository.

Reproduce 1C with the archived diagnostic (`f076a09:docs/history/sharing-sources/diagnose_compartment_attention.py.txt`)
in its matching runtime; see [source requirements](../sources/sharing/README.md). For 1B,
extract its `source_snapshot.zip` to an isolated directory, copy the diagnostic
into `scripts/adhoc/`, and run there with `PYTHONPATH` pointing to its `src`, the
project's environment Python, `--architecture 1b`, and absolute run/output/example
paths. The runtime must match the saved checkpoint; no checkpoint identities are
rewritten. Example IDs map to descriptive filenames in
`outputs/analysis/c/compartment-diagnosis-20260913/examples.json`.
