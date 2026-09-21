# Untied position path

**Closed; implementation discarded.** The appearance-only pooling branch and its
configuration flag have been removed. Main retains position-rich pooling and the
adopted [learned anatomical strength](anatomical-attention.md). Sections below are
historical evidence; old launch commands require their archived source/configs.

Implemented and evaluated on 2026-09-21. The user launched the candidate with
historical control B's seed; no fresh control was trained. The result does not
support adoption, and the main recipe retains the control pooling behavior.
This is the supervised follow-up to the stopped medial/lateral campaign, owned by
the existing [Position and knee side investigation](../docs/backlog/big-bets.md#position-and-knee-side).
It isolates its [independent-classifier proposal](../docs/research/knee-side-and-location.md#5-separate-attention-inputs-from-pooled-appearance)
from the campaign's appearance-value experiment, which also tied the meniscus heads.
This note tracks the implemented slice; other position proposals remain in that backlog.

## Implemented behavior

`model.appearance_only_pooling` is the single experimental switch, false in the
main training config and control, true in the candidate. It applies to all targets.

- Attention receives the existing projected image features plus 1D slice position,
  2D token position, plane and contrast embeddings, followed by the existing LayerNorm.
- Candidate values contain projected image features plus plane and contrast embeddings,
  normalized separately with the same LayerNorm parameters. No additive 1D or 2D
  position enters the pooled values or their normalization statistics.
- The same attention weights pool those values, using the unchanged validity/study
  masks and float32 reduction. Every target keeps its independent classifier row.
- Control values are exactly the normalized attention tokens, preserving the original
  arithmetic and avoiding an additional normalization call.

No new parameters, random initialization, metadata fields, loss terms, classifier
ties, or knee-side inference were introduced. Image features can still encode spatial
context; this removes explicit position from values, not all location information.
Position can still influence predictions through attention weights. The candidate
adds one LayerNorm evaluation over the appearance values; runtime is unmeasured.

The existing encoding diagnostic uses the same token-construction helper as the
model. Position ablations therefore affect only attention in the candidate; plane
and contrast ablations affect both streams. Training and inference use the same
saved model configuration and forward path.

## Prepared matched comparison

- [Control configuration](../outputs/analysis/archived-configs/untied-position/control.yaml)
- [Candidate configuration](../outputs/training/untied-position/candidate/input_config.yaml)

Both use fixed seed **2263738815**, matching the recorded effective configuration
of historical control B (`13-control-repeat`), fold 0, ten epochs, the current ReXNet recipe,
26 study triplets and class-balanced BCE. The only configuration differences are
the experiment name and `model.appearance_only_pooling`. Sampling, augmentation,
optimizer, schedule, independent heads and checkpoint selection are unchanged.
The rejected compartment adapter is disabled in both.

The default training config remains the control behavior. These are manual matched
runs, not a resume of autoresearch. At the time, the supervisor assigned fresh
seeds; the later incumbent-matched seed policy supersedes that behavior. Existing
campaigns and the watchdog remain stopped.

User launches training after checking GPU availability. From the repository root,
run these sequentially in a persistent terminal or detached user-owned shell:

```bash
rtk proxy nvidia-smi
rtk proxy uv run --locked python -m scripts.pipeline.train configs/experiments/untied-position/control.yaml --run-id control
rtk proxy uv run --locked python -m scripts.pipeline.train configs/experiments/untied-position/candidate.yaml --run-id candidate
```

The runs write to `outputs/training/untied-position/control` and `candidate`.
Use the same source checkout for both; source drift between launches invalidates
the intended one-change comparison. The existing trainer saves effective configs,
source snapshots, seeds, predictions and selected-checkpoint metrics.

After both runs finish:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python -m scripts.tools.compare_fold_runs \
  --reference outputs/training/untied-position/control \
  --candidate outputs/training/untied-position/candidate \
  --output-dir outputs/analysis/untied-position-seed2263738815 \
  --allowed-config-differences name,model.appearance_only_pooling
```

Review macro AUC, all twelve labels, the meniscus and OA family means, exclusive-case
compartment margins and total/lateral cross-talk false positives. No label-specific
checkpoint selection. Confirm a promising result with another preselected matched
seed before adoption. Patient bootstrap uncertainty does not measure training-seed
or checkpoint-selection variability; matched seeds also do not guarantee deterministic
CUDA behavior.

## Validation

Ruff lint and formatting, Python syntax parsing, configuration loading/path resolution,
script catalog consistency and whitespace checks passed. The loaded configs differ
only in name and pooling option; both resolve the same data, labels, seed and output
root. All pooling callers were updated. No tests, training or GPU execution were run.
Those implementation checks did not measure model quality or training/inference runtime;
the subsequent user-launched run is recorded below.

## Completed candidate versus historical control B

Candidate: `outputs/training/untied-position/candidate`, seed 2263738815,
selected epoch 9 of 10, 4,380 optimizer steps, 3,503 training and 904 validation
studies. Control B also selected epoch 9. Candidate training took 1,551.92 seconds
versus 1,541.67; final prediction took 19.02 versus 20.05 seconds.

| Metric | Control B | Candidate | Difference |
|---|---:|---:|---:|
| Macro AUC | 0.896582 | 0.894777 | -0.001805 |
| Medial Meniscus | 0.926587 | 0.916540 | -0.010047 |
| Lateral Meniscus | 0.897803 | 0.900537 | +0.002734 |
| Medial OA | 0.907763 | 0.905072 | -0.002691 |
| Lateral OA | 0.868603 | 0.866337 | -0.002266 |
| PF OA | 0.869275 | 0.866800 | -0.002475 |
| Meniscus margin AUC | 0.971248 | 0.971022 | -0.000226 |
| OA margin AUC | 0.866116 | 0.834838 | -0.031278 |

Macro paired report-group 95% interval: [-0.005212, +0.001388], inconclusive.
Medial Meniscus interval: [-0.017509, -0.002880]; OA margin interval:
[-0.060555, -0.004338]. These diagnostic intervals are exploratory and unadjusted.
The lateral-meniscus increase has interval [-0.011742, +0.017809]. Neither family
mean improved; the result offers no basis to promote appearance-only pooling.
This does not establish that all positional separation or geometry-based mechanisms fail.

The source audit found an additional difference beyond pooling: the current
preprocessing restores two repaired DICOM slices that historical control B excluded
([data repair record](../docs/data-preparation/public-data-contract-and-decoding.md)).
The exclusions module and its call sites changed after B. Therefore matching the
seed and nominal recipe does not isolate the pooling change completely. A fresh
control using the candidate's source and repaired data would resolve this confound
if further attribution is worth pursuing. It was not launched. The earlier advice
to reuse B should have been conditional on this source audit.

Other archived source changes cover optional-seed handling, resume plumbing,
disabled adapter code and target-name constructor plumbing. The saved configs
differ only in name, output root and the new enabled pooling field; all input-identity
fields match except the preprocessing source fingerprint. Selected predictions
reproduce the saved AUCs. Both runs were compared on the same validation cohort
with 2,000 paired report-group resamples; no model inference or tests were run for
this analysis. Those intervals exclude training-seed and checkpoint-selection variability.

Artifacts: [comparison](../outputs/analysis/untied-position-vs-control-b/comparison.json),
[per-label diagnostics](../outputs/analysis/untied-position-vs-control-b/per_label.json).
