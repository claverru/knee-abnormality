# Sharing and supervision experiment results

Historical measurements recovered from branch `1C` at `3b8c2cc1c7f7fbfe76ef2b149c91dba0c354002f`.
The experiments below used the sigma-80 reflection reference unless stated
otherwise. Their earlier recommendations are retained only where they explain
the next comparison; the [reflection/sigma-40 result](validation.md#2026-09-14-reflection-sigma-40-result)
now selects independent classifiers and attention at **0.892360 validation AUC
and 0.936 public AUC**. Current `main` uses that recipe with optional CoAtNet
backbone support; it does not implement 1C.

Read the [generalization investigation](sharing-generalization.md) for the
combined evidence, the [disagreement audit](sharing-disagreement-audit.md) for
OA labels and acquisition sensitivity, and [transfer research](research/medial-lateral-transfer.md)
for the retained future approaches. No new experiment is scheduled by this record.
[Historical source records](history/sharing-sources/README.md) preserve the
original analysis commands and 1C architecture; saved run artifacts own exact
checkpoint reproduction.

## 2026-09-14 sharing generalization investigation

1A submission `56226273` completed with **0.930 public AUC**, versus reflection
0.931, 1B 0.925, original 1C 0.928 and sigma-40 1C 0.931. These are the
historical sigma-80-reference comparisons; reflection/sigma 40 later scored 0.936. The
[comprehensive investigation](sharing-generalization.md) records all five
local/public results, per-target changes, paired report-group bootstrap
uncertainty, weak-label sensitivity checks and the existing factor interventions.
It separates measured results from hypotheses about endpoint/case-mix shift,
checkpoint selection and optimization under shared attention. CPU analysis reads
saved predictions only; no GPU work, inference, training or tests ran.

1B's gain over reflection comes from four targets while eight decline; its
advantage disappears on extreme-score labels but survives original-report
coverage and source-agreement restrictions. The constraint can preserve a
trained 1B solution, and sigma-40 1C ties reflection publicly. These findings do
not justify automatically restoring 1B or rejecting compartment sharing. The later
[reflection/sigma-40 result](validation.md#2026-09-14-reflection-sigma-40-result)
selects the independent-head recipe for current development.

## 2026-09-14 original 1C public result

The original sigma-80 1C checkpoint scored **0.928 public AUC**, improving on
1B by **0.003** despite its lower local AUC. Reflection and sigma-40 1C both
score **0.931**, ahead of original 1C by **0.003**.

| Model | Full-fold AUC | Public AUC | Inference studies per batch |
|---|---:|---:|---:|
| Reflection | 0.890976 | 0.931 | 2 |
| 1B | 0.892022 | 0.925 | 2 |
| Original 1C, sigma 80 | 0.889430 | 0.928 | 4 |
| 1C, sigma 40 | 0.892538 | 0.931 | 2 |

All four submissions use their selected fold-0 checkpoint and 2× images.
The original 1C batch-8 attempt failed without a score; its unchanged-model
batch-4 retry completed as submission `56225269`, script version `349717410`.
See [submission evidence](inference-efficiency.md#original-1c-comparison-submission)
for checkpoint identity, failure details and saved API responses. These results
used the previous single-GPU inference code.

The compartment architecture therefore outperformed 1B in this public
comparison; its lower local score did not predict that ranking. Sigma 40 also
improved both local and public scores over original 1C. These trained-model
comparisons cannot isolate the embedding's causal effect or establish
repeatability. Hidden per-target predictions remain unavailable. The
[local intervention findings](#2026-09-13-compartment-attention-diagnosis)
still describe how the original 1C checkpoint uses its learned factor.

## 2026-09-14 sigma-40 higher-LR pair result

`outputs/sigma40-higher-lr` completed ten epochs in 1,636.70 seconds (27.3
minutes), selecting epoch 10 (index 9). The higher-LR pair improves on the
three-way combination but does not beat sigma 40 alone. Retain sigma 40's
backbone/head LRs 0.0002/0.0006 on current evidence.

| Recipe | Full-fold AUC | BCE | Gold AUC, 11 studies |
|---|---:|---:|---:|
| Sigma 40, original LRs and normalization | 0.892538 | 0.448244 | 0.917229 |
| Sigma 40 + higher LRs | 0.891760 | 0.448679 | 0.925661 |
| Sigma 40 + higher LRs + shared normalization | 0.891035 | 0.448816 | 0.927216 |

Against sigma 40 alone, AUC falls 0.000779 and BCE worsens 0.000435. Seven
targets improve and five decline. Largest gains are ACL (+0.002997), synovitis
(+0.001962) and medial meniscus (+0.001853); largest losses are effusion
(-0.009298), fracture (-0.003680) and lateral OA (-0.003254). Pair-minus-triple
AUC is +0.000724. These single-seed comparisons do not establish that shared
normalization universally hurts or that higher LR always underperforms.

Effective configs differ from sigma 40 only in backbone/head LRs
0.0003/0.0009, name and paths. Complete input identities match; model, dataset
and training source match. Package differences are confined to inference code
and the added inference-process module. The run retains 1C, seed 42, Steven v4,
fold 0, 904 validation studies, 224 pixels, 140 mm, reflection 0.5, 2× validation
images and 4,380 optimizer steps. Peak CUDA allocation is 20,681,703,936 bytes,
identical to sigma 40 and the triple. Evidence is each run's effective config,
source snapshot, input identity, selected-checkpoint record, epoch metrics and
timing JSON. The pair's checkpoint is
`fold_0/best-epoch=09-val_macro_auc=0.89176.ckpt`. No new training, inference,
tests or submission ran for this comparison.

## 2026-09-14 sigma-60 result

`outputs/sigma60` completed ten epochs and selected epoch 10 (index 9), scoring
0.892378929 full-fold AUC versus 0.892538106 for the overnight sigma-40 run.
The -0.000159177 difference is too small for a reliable ranking from one fold
and seed. Retain sigma 40 as the submitted incumbent, without treating sigma 60
as an established regression. Both outperform the original sigma-80 1C control
locally (0.889429588 AUC).

| Metric | Sigma 40 | Sigma 60 | Delta |
|---|---:|---:|---:|
| Full-fold AUC, 904 studies | 0.892538 | 0.892379 | -0.000159 |
| Soft-target BCE | 0.448244 | 0.448567 | +0.000323 |
| Gold AUC, 11 studies | 0.917229 | 0.927414 | +0.010185 |
| Training seconds | 1739.16 | 1683.84 | -55.32 |

Seven targets improve and five decline. Largest gains are MCL (+0.011223),
synovitis (+0.005180) and Baker's (+0.004583); largest declines are contusion
(-0.011075), effusion (-0.006212) and fracture (-0.005738). Both meniscus targets
improve. Gold remains diagnostic because it contains only eleven studies.
Peak allocated CUDA memory was 20,392,964,096 bytes versus 20,681,703,936 for
sigma 40; runtime and memory differences are descriptive, not isolated effects.

Saved effective configs differ only in sigma (60 versus 40), names and output
paths. Complete input identities match. Both use 1C, seed 42, Steven v4, fold 0,
224 pixels, 140 mm, reflection 0.5, baseline LRs/normalization, 2× validation
images and 4,380 optimizer steps. Package source differences are confined to
`inference.py` and the added `inference_processes.py`; model/training source
matches. Evidence is each run's effective config, source snapshot, input
identity, selected-checkpoint record, epoch metrics and timing JSON. Sigma 60's
selected checkpoint is `fold_0/best-epoch=09-val_macro_auc=0.89238.ckpt`.
No new training, inference or tests ran for this comparison; sigma 60 has not
been submitted in this workflow.

## 2026-09-14 combined sigma-40 result

Retain sigma 40 alone over the combined candidate on current local evidence.
`outputs/sigma40-higher-lr-shared-normalization` combines sigma 40 with
backbone/head LRs 0.0003/0.0009 and channel mean/std 0.449/0.226. All other
non-path settings match `02-central-sampling` in the completed overnight queue;
archived package Python files and complete input identities match exactly.
Both use 1C, Steven v4, seed 42, fold 0, reflection 0.5, 224 pixels, 140 mm,
2× validation images, ten epochs and 4,380 optimizer steps.

| Metric | Sigma 40 | Combined | Combined minus sigma 40 |
|---|---:|---:|---:|
| Full-fold AUC, 904 studies | 0.892538 | 0.891035 | -0.001503 |
| Soft-target BCE | 0.448244 | 0.448816 | +0.000572 |
| Gold AUC, 11 studies | 0.917229 | 0.927216 | +0.009987 |

Both select epoch 10 (index 9). Combined training took 1,738.80 seconds and
peaked at 20,681,703,936 CUDA bytes, effectively identical to sigma 40's
1,739.16 seconds and identical peak allocation. Three targets improve and nine
decline. Largest gains are lateral meniscus (+0.007634), ACL (+0.001857) and
Baker's (+0.000962); largest losses are fracture (-0.008003), contusion
(-0.006388) and MCL (-0.005951). The individual hyperparameter gains did not
combine into a better result in this run; this single-seed comparison cannot
separate interaction effects from training variability. The small gold subset
does not override the full-fold decline. The candidate has not been submitted.

Evidence is the two runs' saved `effective_config.json`, `source_snapshot.zip`,
`fold_0/input_identity.json`, `selected_checkpoint.json`, epoch metrics and
`timing.json`. The combined selected checkpoint is
`fold_0/best-epoch=09-val_macro_auc=0.89104.ckpt`. No new training, inference or
tests ran for this comparison. Sigma 40's completed public score is 0.931,
tying reflection at the reported precision.

## 2026-09-14 overnight hyperparameter results

All eleven runs in
[`overnight-20260913T214822Z-74udtekr`](../outputs/overnight-20260913T214822Z-74udtekr)
completed ten epochs successfully. Recorded setup, training and selected
validation prediction total 5.536 hours. Every selected checkpoint is epoch 10
(index 9), also the maximum saved epoch AUC for its run.

The executed architecture was **1C**, rather than the originally proposed
reflection architecture. All run preprocessing source hashes match; archived
package Python files match `disease-compartment-attention`. Saved label,
membership, training/validation row, split protocol and series identities match
that control. Effective configs differ only by the intended settings, names and
output locations. The 1C control scores 0.889430 AUC and 0.448458 BCE; reflection
scores 0.890976 AUC and remains the 0.931 public benchmark.

| Run | Full-fold AUC | Delta vs 1C | BCE | Training minutes |
|---|---:|---:|---:|---:|
| 02 central sampling, sigma 40 | 0.892538 | +0.003109 | 0.448244 | 29.0 |
| 06 higher final LR | 0.891216 | +0.001787 | 0.449815 | 29.0 |
| 01 higher peak LRs | 0.891146 | +0.001716 | 0.447721 | 27.9 |
| 09 shared normalization | 0.890768 | +0.001338 | 0.447876 | 29.9 |
| 03 crop 128 mm | 0.889792 | +0.000362 | 0.449161 | 28.9 |
| 05 attention dropout | 0.889782 | +0.000352 | 0.449098 | 29.3 |
| 10 stronger intensity | 0.889678 | +0.000249 | 0.450104 | 31.5 |
| 11 no noise | 0.889535 | +0.000105 | 0.448399 | 30.5 |
| 04 broad sampling, sigma 120 | 0.889060 | -0.000370 | 0.448715 | 29.2 |
| 08 longer warmup | 0.888393 | -0.001036 | 0.450408 | 30.6 |
| 07 effective batch 16 | 0.887154 | -0.002276 | 0.451711 | 30.3 |

The queue used these fixed overrides against its sigma-80 1C baseline.
The [archived launcher](history/sharing-sources/train_overnight.py.txt) and each
run's effective config preserve the full settings.

| Run | Changes from baseline |
|---|---|
| `01-higher-lr` | Backbone LR `3e-4`, head LR `9e-4` |
| `02-central-sampling` | Sampling sigma 40 mm |
| `03-crop128mm` | Spacing 0.57142857 mm, retaining 224 pixels |
| `04-broad-sampling` | Sampling sigma 120 mm |
| `05-attention-dropout` | Model dropout 0.1 |
| `06-higher-final-lr` | Minimum LR ratio 0.1 |
| `07-effective-batch16` | Accumulation 2, retaining physical batch 8 and baseline LRs |
| `08-longer-warmup` | Warmup 2 epochs |
| `09-shared-normalization` | Mean `[0.449, 0.449, 0.449]`, std `[0.226, 0.226, 0.226]` |
| `10-stronger-intensity` | Gamma `[0.7, 1.3]`, contrast `[0.8, 1.2]`, retaining probability 0.5 |
| `11-no-noise` | Noise probability 0 |

Sigma 40 improves seven targets versus 1C, led by fracture (+0.019874) and
MCL (+0.015536), with lateral meniscus down 0.006575. Its eleven-study gold
AUC is 0.917229. Higher peak LRs improve eight targets and have the lowest
BCE. Shared normalization improves both aggregate metrics versus 1C. Higher
final LR worsens BCE and jumps from 0.885802 AUC at epoch 9 to 0.891216 at
epoch 10. Effective batch 16 uses 2,190 optimizer steps versus 4,380 otherwise.

Evidence is each run's selected-checkpoint record, saved epoch metrics,
effective config, source snapshot, input identity and timing JSON. No new
training, inference or tests ran for this summary. These single-seed comparisons
on 904 studies do not establish repeatability or transfer to reflection/public
performance. The user requested the
[sigma-40 submission](inference-efficiency.md#sigma-40-overnight-submission)
on 2026-09-14; it completed with 0.931 public AUC, tying reflection.

## 2026-09-13 1B public result and overnight code choice

This section records the September 13 decision. The subsequent
[1C and sigma-40 public results](#2026-09-14-original-1c-public-result)
resolve the submission hypotheses below.

**Use the reflection architecture for the main hyperparameter search.**
Submission **56215113**, the selected 1B epoch-7 checkpoint, completed with
**0.925 public AUC**, a **0.006 loss** against reflection's **0.931**. Its local
gain of **0.001047** was competitively relevant but did not transfer publicly.
The earlier recommendation to submit 1B was based on that local gain; this new
result changes the recommended training baseline.

| Architecture | Full-fold AUC | Gold AUC (11) | Soft-target BCE | Public AUC |
|---|---:|---:|---:|---:|
| Reflection, independent classifiers and attention | 0.890976 | 0.922784 | 0.449310 | **0.931** |
| 1A, shared meniscus classifier | 0.891192 | 0.914683 | 0.448576 | Unsubmitted |
| 1B, shared meniscus and OA classifiers | **0.892022** | **0.941997** | 0.453585 | **0.925** |
| 1C, both shared classifiers and compartment attention | 0.889430 | 0.937368 | **0.448458** | Unsubmitted |

The saved submission record identifies the intended 1B checkpoint and notebook
version. A CPU-only audit rechecked every bundle file hash, the original
checkpoint hash, and all bundled Python files against its training source
snapshot. They match. Both submission bundles have identical effective configs,
2× image density and batch size 2; both visible notebook executions use FP32.
The only bundled Python differences are the classifier implementation, target
construction and architecture identity. The three visible example predictions
have the correct target columns and finite probabilities. There is no observed
export mismatch, accidental 1C submission or changed preprocessing recipe.
This audit did not rerun inference and cannot inspect hidden scored predictions.

Against reflection, 1B improves four local target AUCs and worsens eight. Its
largest gains are contusion **+0.013890**, medial meniscus **+0.008514**, and
fracture **+0.007701**; medial OA adds **+0.001290**. Lateral meniscus loses
**0.001533**, lateral OA loses **0.002597**, and Baker's loses **0.007664**.
Excluding contusion and fracture, the remaining ten targets average
**−0.000903**. This decomposition describes a concentrated local improvement;
it does not redefine the competition metric or identify the hidden failing
targets. BCE worsens by **0.004275**. Even the eleven-study gold diagnostic
improves by **0.019213**, so neither local endpoint predicted this public reversal.

Most local validation labels are report-derived (893/904 studies). Their
different labeling and sample composition, and training variation, are plausible
reasons for a local/public reversal. Hidden labels and per-target public scores
are unavailable, so attributing the 0.006 loss specifically to OA sharing,
lateral localization or any one target would exceed the evidence. This result
also does not isolate 1A's public behavior.

**1C could still outperform 1B publicly.** Today tested independent attention
with shared classifiers; it did not test the explicit compartment factor. The
[earlier intervention](#2026-09-13-compartment-attention-diagnosis) establishes
that the trained 1C model uses its factor, and removing it reduces local AUC to
0.872267. That does not establish that training 1C generalizes better than
reflection or 1B. Submit the already-trained, unmodified 1C checkpoint when a
submission slot is available to measure that hypothesis. No new submission was
launched during this analysis.

### Code used by the overnight queue

This is the completed September 13 queue. The active-queue instructions below
describe the decision at that time; they do not schedule or stop any current job.
Its [eleven-run launcher](history/sharing-sources/train_overnight.py.txt) is
archived separately from the seven-run launcher on `main`.

The user-launched queue `outputs/overnight-20260913T214822Z-74udtekr` was already
training `01-higher-lr` during this audit. Its source archive records
`spatial_tokens_disease_compartment_attention_v1`: **it is a 1C search**.
`configs/baseline.yaml` being named `spatial-tokens-reflection` does not select
an architecture. The launcher starts each experiment from current repository
source and saves that run's snapshot; preparation freezes configurations only.
Changing source under the active launcher could mix architectures within one
queue. No process, source or queued configuration was changed in this analysis.

For a reflection search, use the exact training runtime in
`outputs/spatial-tokens-reflection/source_snapshot.zip`, whose independent
classifier and attention implementation produced the 0.931 submission. Switching
requires a fresh queue with that runtime after stopping the current queue, or an
isolated runtime when compute is available. Merely selecting its YAML is
insufficient. If continuing the current 1C queue, compare hyperparameter changes
against `outputs/disease-compartment-attention` (**0.889430** local AUC) to isolate
their effect; reflection remains the public benchmark. A full 1C search is an
exploration of an unsubmitted architecture, whereas reflection is the choice
supported by the best observed public result.

The [audit artifact](../outputs/analysis/1b-public-result-20260913/analysis.json)
pins submissions, bundle/checkpoint/source hashes, configurations, per-target
deltas and the active queue's first-run identity. It also extends the
[local/public comparison](validation.md#2026-09-13-local-and-public-agreement-after-the-split-change)
to nine fixed-label checkpoints. Only saved files and process metadata were
read; no GPU queries, inference, training or tests were run.

## 2026-09-13 Label-masking result

**Rejected:** `masked-labels` scores **0.879520** full-fold AUC versus the
reflection champion's **0.890976** (−0.011455). Restore full nonmissing soft-target
supervision with gold overrides; the masking implementation and configuration
have been removed. Reflection remains the selected baseline.

| Metric | Reflection | Masked labels | Masked − reflection |
|---|---:|---:|---:|
| Full-fold AUC (904 studies) | 0.890976 | 0.879520 | −0.011455 |
| Gold AUC (11 studies) | 0.922784 | 0.910351 | −0.012434 |
| Report-derived AUC (893 studies) | 0.890228 | 0.878792 | −0.011436 |
| Soft-target BCE (lower is better) | 0.449310 | 0.498855 | +0.049545 |
| Training seconds | 1794.98 | 1675.52 | −119.46 |

Both runs completed ten epochs and selected epoch 9 (zero-based index 8), with
4,380 optimizer steps and peak allocated CUDA memory of 20,368,205,824 bytes.
Only MCL improved; the other eleven findings regressed.

| Finding | Reflection AUC | Masked AUC | Delta |
|---|---:|---:|---:|
| ACL | 0.913949 | 0.902431 | -0.011519 |
| MCL | 0.836247 | 0.849729 | +0.013481 |
| Medial Meniscus | 0.922563 | 0.917401 | -0.005162 |
| Lateral Meniscus | 0.889980 | 0.884134 | -0.005846 |
| Medial OA | 0.903951 | 0.887438 | -0.016514 |
| Lateral OA | 0.868678 | 0.855485 | -0.013193 |
| PF OA | 0.874246 | 0.847869 | -0.026376 |
| Effusion | 0.868662 | 0.853484 | -0.015178 |
| Synovitis | 0.896302 | 0.873963 | -0.022340 |
| Baker's | 0.908141 | 0.882375 | -0.025766 |
| Contusion | 0.888732 | 0.883737 | -0.004996 |
| Fracture | 0.920257 | 0.916200 | -0.004058 |

Saved configurations differ only in experiment name and the masking policy.
Both use seed 42, fold 0 of `all_5fold_v3`, Steven v4, 224 pixels, 140-mm crops
and the same reflection recipe. Input identities match for preprocessing, label
manifest/source, split protocol, membership, validation rows and series metadata
and counts. Training-row identity changes with the observation mask.

The [method and counts](supervision.md#training-label-masking) document the
10,548 removed training entries and 46 wholly masked studies. This single-fold,
single-seed result supports rejecting this implementation; it does not establish
that every uncertainty-weighting approach fails. The 11-study gold diagnostic
is especially small. No new training, inference or public submission was run
for this comparison.

Evidence: [comparison JSON](../outputs/analysis/masking-vs-reflection-20260913/comparison.json)
contains full metrics, per-target deltas, configurations, input identities and
artifact SHA-256 hashes. The selected masked checkpoint is
`outputs/masked-labels/fold_0/best-epoch=08-val_macro_auc=0.87952.ckpt`;
its [source snapshot](../outputs/masked-labels/source_snapshot.zip),
[effective configuration](../outputs/masked-labels/effective_config.json) and
[epoch metrics](../outputs/masked-labels/fold_0/epochs/metrics_epoch_008.json)
remain available alongside the reflection run's corresponding artifacts.

## 2026-09-13 Compartment attention diagnosis

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

### What the component changes

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

### Effects on the trained 1C checkpoint

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

### Does 1B need different contrasts?

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

### Learned strength and attention maps

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

[Intervention figure](../outputs/analysis/compartment-diagnosis-20260913/intervention-effects.png)
([PDF](../outputs/analysis/compartment-diagnosis-20260913/intervention-effects.pdf))
compares AUC effects and learned contrast strength.
[Attention examples](../outputs/analysis/compartment-diagnosis-20260913/attention-examples.png)
([PDF](../outputs/analysis/compartment-diagnosis-20260913/attention-examples.pdf)) show
identical prepared coronal inputs under 1B and 1C. Each selected study changed its
OA ranking outcome; the displayed triplet maximizes combined 1B+1C OA attention
mass among coronal inputs. Positive relative contrast favors medial and negative
favors lateral. Attention overlays use actual stride-32 feature anchors and paired
color scales. They illustrate routing changes, not reference lesion annotations
or a representative localization accuracy estimate. Four example arrays and exact
selection records are preserved in the artifact directory.

### Interpretation and limits

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

The [summary](../outputs/analysis/compartment-diagnosis-20260913/summary.json)
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

Reproduce 1C with the [archived diagnostic](history/sharing-sources/diagnose_compartment_attention.py.txt)
in its matching runtime; see [source requirements](history/sharing-sources/README.md). For 1B,
extract its `source_snapshot.zip` to an isolated directory, copy the diagnostic
into `scripts/adhoc/`, and run there with `PYTHONPATH` pointing to its `src`, the
project's environment Python, `--architecture 1b`, and absolute run/output/example
paths. The runtime must match the saved checkpoint; no checkpoint identities are
rewritten. Example IDs map to descriptive filenames in
`outputs/analysis/compartment-diagnosis-20260913/examples.json`.

## 2026-09-13 Disease-and-compartment attention validation comparison

`disease-compartment-attention` scores **0.889430 macro AUC** on all 904 validation
studies, **−0.002593 versus 1B** (`meniscus-oa-classifiers`, 0.892022) and
**−0.001546 versus reflection** (0.890976). This is a competitive regression on
the primary local selection metric. The initial recommendation favored 1B;
subsequent public scores favored original 1C over 1B, and reflection/sigma 40
later became the champion. This section preserves the local comparison, not
a current recommendation to restore 1B.

All three effective configurations and pretrained backbone identities match.
Data input identities match, with only preprocessing/model source identities
differing. Frozen protocol, membership and target hashes were verified. Saved
selected-checkpoint predictions cover exactly the same 904 unique studies, folds
and official target order, with finite probabilities in [0, 1]. Recomputed AUC
matches the best saved epoch metric. The saved 1C identity confirms both shared
classifier pairs and the disease-plus-signed-compartment attention factors.

| Metric | Reflection | 1B | 1C | 1C − 1B |
|---|---:|---:|---:|---:|
| Full-fold macro AUC | 0.890976 | 0.892022 | 0.889430 | -0.002593 |
| Meniscus mean AUC | 0.906272 | 0.909762 | 0.906810 | -0.002952 |
| OA mean AUC | 0.886315 | 0.885661 | 0.880112 | -0.005549 |
| Report-derived AUC (893) | 0.890228 | 0.891061 | 0.888592 | -0.002469 |
| Gold AUC (11) | 0.922784 | 0.941997 | 0.937368 | -0.004630 |
| Soft-target BCE (lower is better) | 0.449310 | 0.453585 | 0.448458 | -0.005127 |

| Target | 1B | 1C | Delta |
|---|---:|---:|---:|
| ACL | 0.911609 | 0.916629 | +0.005020 |
| MCL | 0.834096 | 0.831130 | -0.002966 |
| Medial Meniscus | 0.931077 | 0.921635 | -0.009442 |
| Lateral Meniscus | 0.888447 | 0.891985 | +0.003538 |
| Medial OA | 0.905241 | 0.903650 | -0.001591 |
| Lateral OA | 0.866081 | 0.856574 | -0.009507 |
| PF OA | 0.872871 | 0.874988 | +0.002117 |
| Effusion | 0.867596 | 0.877583 | +0.009986 |
| Synovitis | 0.896192 | 0.890505 | -0.005687 |
| Baker's | 0.900476 | 0.907145 | +0.006669 |
| Contusion | 0.902622 | 0.891152 | -0.011471 |
| Fracture | 0.927958 | 0.910179 | -0.017779 |

Five targets improve and seven decline. The largest losses are fracture
(−0.017779), contusion (−0.011471), lateral OA (−0.009507) and medial meniscus
(−0.009442). Effusion gains 0.009986, Baker's 0.006669 and ACL 0.005020. Both
OA targets decline, as do the mean AUCs of both experimental families. BCE improves
by 0.005127; that is useful evidence about probability quality, but full-fold AUC
remains the selection metric. The gold endpoint has only eleven studies.

Exclusive-positive subsets use the frozen >=0.5 label cutoff. Meniscus has 344
studies (281 medial-only, 63 lateral-only); OA has 198 (143 medial-only, 55
lateral-only). Each contains three gold studies. For each target, negative count
equals the other target's positive count. Margin AUC uses
`logit(p_medial) - logit(p_lateral)` to distinguish medial-only from lateral-only,
clipping probabilities to [1e-7, 1-1e-7] before logits.

| Exclusive-positive diagnostic | 1B | 1C |
|---|---:|---:|
| Medial Meniscus subgroup AUC | 0.906682 | 0.911879 |
| Lateral Meniscus subgroup AUC | 0.819974 | 0.813252 |
| Meniscus margin AUC | 0.980173 | 0.972208 |
| Meniscus correct within-study rankings | 325/344 | 322/344 |
| Medial Meniscus-only correct rankings | 275/281 | 274/281 |
| Lateral Meniscus-only correct rankings | 50/63 | 48/63 |
| Medial OA subgroup AUC | 0.706548 | 0.714812 |
| Lateral OA subgroup AUC | 0.542022 | 0.571647 |
| Oa margin AUC | 0.870184 | 0.854291 |
| Oa correct within-study rankings | 160/198 | 156/198 |
| Medial OA-only correct rankings | 138/143 | 123/143 |
| Lateral OA-only correct rankings | 22/55 | 33/55 |

The OA per-target subgroup AUCs improve, but its within-study margin AUC falls
0.870184 → 0.854291. Correct lateral-only rankings improve 22/55 → 33/55 while
medial-only rankings fall 138/143 → 123/143; total correct rankings fall 160/198
→ 156/198. This improves one subgroup at the expense of the other rather than
improving overall separation. Meniscus margin AUC falls 0.980173 → 0.972208 and
correct rankings fall 325/344 → 322/344. Per-target subgroup AUC and within-study
ranking measure different behavior; neither establishes actual anatomical
localization from these mostly report-derived labels.

1C selected epoch 10 (index 9), versus epoch 7 for 1B. Both completed ten epochs
and 4,380 optimizer steps. 1C's best score occurs at the end of the fixed budget;
this result does not establish its performance with a different training budget.
1B recorded 1667.15s training, 33.75s selected-checkpoint prediction and 18.624 GiB peak allocated CUDA memory.
1C recorded 1811.77s training, 34.45s selected-checkpoint prediction and 18.965 GiB peak allocated CUDA memory.
These are recorded run costs, not a controlled runtime benchmark.

The [comparison artifact](../outputs/analysis/disease-compartment-vs-controls-20260913/comparison.json)
preserves all four runs' metrics, subgroup counts, within-study rankings, epoch
curves, configs, input identities, timing and hashes of checkpoints, predictions
and source snapshots. The process finished its selected-checkpoint prediction
export during this review; the analysis uses those final artifacts. No fresh
training, inference or tests were launched. No public score for 1C was checked;
this single-fold result supports model selection without quantifying repeatability.

## 2026-09-13 Meniscus and OA sharing validation comparison

`meniscus-oa-classifiers` scores **0.892022 macro AUC** on all 904 validation
studies: **+0.000830** over `meniscus-classifier` and **+0.001047** over
`spatial-tokens-reflection`. These are competitively relevant gains on the primary
selection metric. The decision at this stage was to retain both shared classifier
pairs and use this run as the control for 1C. Per-target, subgroup and BCE results
provide context; they do not negate the measured full-fold AUC improvement.
The later [public result](#2026-09-13-1b-public-result-and-overnight-code-choice)
is **0.925 versus reflection's 0.931**, changing the recommendation for the main
hyperparameter search to reflection.

The selected checkpoint is epoch 7 (index 6); both controls selected epoch 9.
All runs completed ten epochs and 4,380 optimizer steps. Saved effective configs
and pretrained backbone identities match exactly. Data input identities match;
only preprocessing/model source identities differ. The candidate identity records
two separate shared vectors, one for meniscus and one for OA, with independent
biases and attention for all twelve targets. Frozen protocol, membership and target
hashes were verified, and all prediction tables contain the same 904 unique
studies, correct folds, official target order and finite probabilities in [0, 1].
Recomputed selected-prediction AUC matches each run's best saved epoch metric.

| Metric | Reflection | Meniscus sharing (1A) | Both pairs (1B) | 1B − 1A |
|---|---:|---:|---:|---:|
| Full-fold macro AUC | 0.890976 | 0.891192 | 0.892022 | +0.000830 |
| Meniscus mean AUC | 0.906272 | 0.909343 | 0.909762 | +0.000419 |
| OA mean AUC | 0.886315 | 0.884359 | 0.885661 | +0.001302 |
| Report-derived AUC (893) | 0.890228 | 0.890516 | 0.891061 | +0.000545 |
| Gold AUC (11) | 0.922784 | 0.914683 | 0.941997 | +0.027315 |
| Soft-target BCE (lower is better) | 0.449310 | 0.448576 | 0.453585 | +0.005009 |

| Target | Reflection | 1A | 1B | 1B − 1A |
|---|---:|---:|---:|---:|
| ACL | 0.913949 | 0.913240 | 0.911609 | -0.001630 |
| MCL | 0.836247 | 0.830238 | 0.834096 | +0.003857 |
| Medial Meniscus | 0.922563 | 0.925300 | 0.931077 | +0.005777 |
| Lateral Meniscus | 0.889980 | 0.893386 | 0.888447 | -0.004938 |
| Medial OA | 0.903951 | 0.906129 | 0.905241 | -0.000888 |
| Lateral OA | 0.868678 | 0.862589 | 0.866081 | +0.003492 |
| PF OA | 0.874246 | 0.882357 | 0.872871 | -0.009486 |
| Effusion | 0.868662 | 0.863882 | 0.867596 | +0.003714 |
| Synovitis | 0.896302 | 0.897801 | 0.896192 | -0.001609 |
| Baker's | 0.908141 | 0.903323 | 0.900476 | -0.002846 |
| Contusion | 0.888732 | 0.895277 | 0.902622 | +0.007345 |
| Fracture | 0.920257 | 0.920786 | 0.927958 | +0.007172 |

Six targets improve and six decline against 1A. OA mean AUC improves by 0.001302:
lateral OA gains 0.003492 and medial OA loses 0.000888. Medial meniscus, contusion
and fracture also improve; PF OA and lateral meniscus show the largest losses.
BCE increases by 0.005009. Gold AUC improves by 0.027315, but its eleven studies
remain a diagnostic sample. OA mean AUC remains 0.000654 below reflection;
the combined model's advantage is established for full-fold macro AUC, not every
family or target separately.

The exclusive-positive subsets use frozen labels at >=0.5: 344 meniscus studies
(281 medial-only, 63 lateral-only) and 198 OA studies (143 medial-only, 55 lateral-only).
Each contains three gold studies. Negative counts for each target equal the other
target's positive count. Per-target AUCs compare studies; the margin AUC uses
`logit(p_medial) - logit(p_lateral)` to distinguish medial-only from lateral-only
studies, with probabilities clipped to [1e-7, 1-1e-7].

| Exclusive-positive diagnostic | Reflection | 1A | 1B |
|---|---:|---:|---:|
| Medial Meniscus subgroup AUC | 0.917189 | 0.907247 | 0.906682 |
| Lateral Meniscus subgroup AUC | 0.826809 | 0.815511 | 0.819974 |
| Meniscus margin AUC | 0.972321 | 0.972039 | 0.980173 |
| Meniscus correct within-study rankings | 322/344 | 322/344 | 325/344 |
| Medial OA subgroup AUC | 0.714431 | 0.701971 | 0.706548 |
| Lateral OA subgroup AUC | 0.577368 | 0.553465 | 0.542022 |
| Oa margin AUC | 0.856707 | 0.872982 | 0.870184 |
| Oa correct within-study rankings | 156/198 | 159/198 | 160/198 |

Meniscus margin AUC improves 0.972039 → 0.980173 against 1A, with correct medial-only
rankings 276/281 → 275/281 and lateral-only 46/63 → 50/63. OA margin AUC changes
0.872982 → 0.870184; correct medial-only rankings rise 131/143 → 138/143 while
lateral-only rankings fall 28/55 → 22/55. That imbalance is relevant to 1C even
though aggregate OA ranking accuracy rises from 159/198 to 160/198. These mostly
report-derived endpoints do not establish anatomical localization quality.

Training took **27m 47s**, versus 27m 55s for 1A and 29m 55s for reflection.
Selected-checkpoint prediction took **33.75s**, versus 32.29s and 33.43s.
Peak allocated CUDA memory was **18.624 GiB**, versus 18.631 and 18.969 GiB.
Recorded run costs do not isolate an architecture-specific runtime effect.
Candidate AUC peaks at epoch 7, falls to 0.886732 at epoch 8, then recovers to
0.891646 at epoch 10; selection uses the saved best checkpoint.

The [comparison artifact](../outputs/analysis/meniscus-oa-vs-controls-20260913/comparison.json)
preserves complete metrics, subgroup counts, within-study rankings, epoch curves,
configurations, input identities, timing and SHA-256 hashes of checkpoints,
predictions and source snapshots. One fold and seed do not quantify training
repeatability. Public AUC was unavailable for this initial comparison and is now
recorded in the separate public-result analysis above. No fresh training,
inference or tests were run for this comparison.

## 2026-09-13 Meniscus-sharing validation comparison

The completed `meniscus-classifier` run scores **0.891192** versus **0.890976**
for `spatial-tokens-reflection`: **+0.000217 macro AUC**. Both select epoch 9
(index 8) from ten epochs. This is a mixed result: both meniscus targets improve
over the full fold, but both worsen on studies labeled positive in exactly one
meniscus. User decision for 1B: retain meniscus classifier sharing and use this
saved run as the control when adding OA classifier sharing. Reflection remains a
secondary reference. The small overall gain and mixed diagnostics do not establish
a repeatable benefit or localization failure; direct within-study separation is
broadly unchanged, as detailed below.

Saved effective configurations match exactly, including seed 42, Steven v4,
fold 0 of `all_5fold_v3`, reflection probability 0.5 and 2× validation density.
The candidate retains the config name `spatial-tokens-reflection`, but its run
ID is `meniscus-classifier` and its recorded architecture identifies sharing.
All data input identities match; preprocessing/model source identities differ.
Frozen protocol, membership and target hashes were verified. Both saved prediction
tables contain the same 904 unique studies, correct folds, official target order
and finite probabilities in [0, 1]. Metrics below were recomputed from those tables.

| Metric | Reflection | Meniscus sharing | Delta |
|---|---:|---:|---:|
| Full-fold macro AUC | 0.890976 | 0.891192 | +0.000217 |
| Meniscus mean AUC | 0.906272 | 0.909343 | +0.003071 |
| Report-derived AUC (893) | 0.890228 | 0.890516 | +0.000288 |
| Gold AUC (11) | 0.922784 | 0.914683 | -0.008102 |
| Soft-target BCE (lower is better) | 0.449310 | 0.448576 | -0.000734 |

| Target | Reflection | Meniscus sharing | Delta |
|---|---:|---:|---:|
| ACL | 0.913949 | 0.913240 | -0.000710 |
| MCL | 0.836247 | 0.830238 | -0.006009 |
| Medial Meniscus | 0.922563 | 0.925300 | +0.002737 |
| Lateral Meniscus | 0.889980 | 0.893386 | +0.003406 |
| Medial OA | 0.903951 | 0.906129 | +0.002178 |
| Lateral OA | 0.868678 | 0.862589 | -0.006090 |
| PF OA | 0.874246 | 0.882357 | +0.008111 |
| Effusion | 0.868662 | 0.863882 | -0.004779 |
| Synovitis | 0.896302 | 0.897801 | +0.001499 |
| Baker's | 0.908141 | 0.903323 | -0.004818 |
| Contusion | 0.888732 | 0.895277 | +0.006545 |
| Fracture | 0.920257 | 0.920786 | +0.000528 |

Seven targets improve and five decline. The largest gains are PF OA and
contusion; the largest losses are lateral OA and MCL. Gold AUC declines by
0.008102, but eleven studies provide only a small diagnostic sample.

For the 344 studies positive in exactly one meniscus at the frozen >=0.5 cutoff,
281 are medial-positive/lateral-negative and 63 the reverse. Only three are gold.
Medial AUC falls **0.917189 → 0.907247** (-0.009942); lateral falls
**0.826809 → 0.815511** (-0.011298). Their mean falls **0.871999 → 0.861379**
(-0.010620). These per-target AUCs compare different studies and do not directly
measure which meniscus receives the higher score within a study.

A subsequent within-study check gives the affected meniscus the higher probability
in **322/344 studies (93.60%) for both models**. The AUC of
`logit(p_medial) - logit(p_lateral)` for medial-only versus lateral-only is
**0.972321 → 0.972039**, essentially unchanged. Correct medial-only rankings rise
273/281 → 276/281 while lateral-only rankings fall 49/63 → 46/63; the aggregate
accuracy is dominated by the medial-only majority. Thus the subgroup regression
does not establish attention collapse or broadly worse compartment separation.
Mostly report-derived labels cannot establish actual anatomical localization.
For completeness, the corresponding OA subgroup has 198 studies (143 medial-only,
55 lateral-only, three gold); mean AUC falls 0.645900 → 0.627718. OA weights
were not shared in this experiment.

Training took **27m 55s**, versus 29m 55s for reflection (-6.7%). Selected-checkpoint
validation prediction took 32.29s versus 33.43s; peak allocated CUDA memory was
18.631 versus 18.969 GiB. These are recorded run costs, not a controlled runtime
benchmark. Both completed 4,380 optimizer steps. Candidate macro AUC peaks at
epoch 9 then declines slightly to 0.890950 at epoch 10.

One fold and one seed do not establish a repeatable gain or loss; no uncertainty
interval or new public score was computed. The
[comparison artifact](../outputs/analysis/meniscus-vs-reflection-20260913/comparison.json)
preserves all metrics, subgroup counts, epoch curves, configuration and input
identities, timing, and hashes of predictions, checkpoints and source snapshots.
No fresh training, inference or tests were run.
