# 2026-09-13 1B public result and overnight code choice

This section records the September 13 decision. The subsequent
[1C and sigma-40 public results](2026-09-14-original-1c-public-result.md)
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
[earlier intervention](2026-09-13-compartment-attention-diagnosis.md) establishes
that the trained 1C model uses its factor, and removing it reduces local AUC to
0.872267. That does not establish that training 1C generalizes better than
reflection or 1B. Submit the already-trained, unmodified 1C checkpoint when a
submission slot is available to measure that hypothesis. No new submission was
launched during this analysis.

## Code used by the overnight queue

This is the completed September 13 queue. The active-queue instructions below
describe the decision at that time; they do not schedule or stop any current job.
Its eleven-run launcher (`f076a09:docs/history/sharing-sources/train_overnight.py.txt`) is
archived separately from the seven-run launcher at `ea19772`.

The user-launched queue `outputs/training/overnight-20260913T214822Z-74udtekr` was already
training `01-higher-lr` during this audit. Its source archive records
`spatial_tokens_disease_compartment_attention_v1`: **it is a 1C search**.
`configs/baseline.yaml` (removed) being named `spatial-tokens-reflection` does not select
an architecture. The launcher starts each experiment from current repository
source and saves that run's snapshot; preparation freezes configurations only.
Changing source under the active launcher could mix architectures within one
queue. No process, source or queued configuration was changed in this analysis.

For a reflection search, use the exact training runtime in
`outputs/training/spatial-tokens-reflection/source_snapshot.zip`, whose independent
classifier and attention implementation produced the 0.931 submission. Switching
requires a fresh queue with that runtime after stopping the current queue, or an
isolated runtime when compute is available. Merely selecting its YAML is
insufficient. If continuing the current 1C queue, compare hyperparameter changes
against `outputs/training/disease-compartment-attention` (**0.889430** local AUC) to isolate
their effect; reflection remains the public benchmark. A full 1C search is an
exploration of an unsubmitted architecture, whereas reflection is the choice
supported by the best observed public result.

The audit artifact (`outputs/analysis/0-9/1b-public-result-20260913/analysis.json`, deleted)
pins submissions, bundle/checkpoint/source hashes, configurations, per-target
deltas and the active queue's first-run identity. It also extends the
[local/public comparison](../../validation/results/validation-and-ensembles/2026-09-13-local-and-public-agreement-after-the-split-change.md)
to nine fixed-label checkpoints. Only saved files and process metadata were
read; no GPU queries, inference, training or tests were run.
