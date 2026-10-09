# Run-level stratified sampling experiment and reverted promotion

**The default promotion was reverted on 2026-09-20 at the user's request.** The
second run did not establish a repeatable advantage against the two controls,
and the original sampler tied control B publicly. This is insufficient evidence
for retaining the change, not proof that the mechanism is harmful. Future training
uses the previous independent-per-epoch jitter inside Gaussian probability bands.
Balanced BCE, triplet budgets and deterministic validation ×1 are retained.

Earlier the same day, the user requested promotion of `run-level-stratified-jitter`.
Terra at medium effort implemented it, and the parent reviewed the accepted snapshot.
The experiment, comparisons, source archives, checkpoints and
[infographic](../../../triplet-sampling-infographic.png) remain as historical evidence.
The infographic describes the reverted candidate, not the active sampler.

## Historical candidate behavior

For each study and plane/contrast pool, retain the current Gaussian probability
bands and triplet count. Split each band into ten equal-probability subregions.
Each band gets its own random permutation, shared across a ten-epoch cycle. Epochs
visit those subregions in that order with fresh within-subregion jitter. The next
ten-epoch cycle uses a new permutation. The training seed, study and type remain
part of the random-stream identity.

Every epoch still targets every main band. Only temporal coordination changes:
supplier choice, geometry grouping, nearest assignment, retry behavior, candidate
construction and budgets stay the same. Validation retains deterministic band
midpoints and does not receive the training seed. Actual discrete triplets can
repeat even when target subregions differ.

The temporary production port changed
[`sampling.py`](../../../src/knee_abnormality/data/preprocessing/sampling.py),
fixing the stratification depth at ten without an on/off switch, depth sweep,
extra config field or dataset argument. Its core formula and random-stream
schedule matched the archived winner at depth ten. Research metadata for the
configurable depth was omitted. That port is preserved in
`outputs/training/another-stratified-sampling/source_snapshot.zip`; the original
research implementation remains in its experiment's immutable `source/` directory.
The main YAML's paths, seed, model, loss, epochs, triplet budgets and augmentation
settings were not edited. Existing checkpoint reproduction uses archived sources
and configuration, including the original winner's configurable-depth schema.

Rollback removed the ten-epoch schedule and its target-rule identity from the
working sampler. The resulting file matches committed HEAD byte for byte; Ruff
and diff whitespace checks pass. The campaign remains stopped with its historical
state intact. No legacy switch or checkpoint compatibility branch was added.

## Experimental evidence

Accepted experiment:
`outputs/autoresearch/triplet-sampling/experiments/run-level-stratified-jitter-20260919T183826Z-f211c7ce/`.
Its immutable `source/` is the implementation reference; `training/` holds the fitted
model and accounting. The comparison reference was the historical control in
`outputs/training/overnight-20260919T120130Z-i2oqlnls/runs/00-control/`.

| Model | Training seed | Fold-0 macro AUC |
|---|---:|---:|
| Historical control | 927831896 | 0.894037393 |
| Run-level stratified sampler | 4061686476 | 0.897514923 |

Observed delta: **+0.003477530**. The paired report-group bootstrap used 2,000
resamples across 852 groups, with resampling seed 20260914, and returned a 95%
interval **[+0.000254369, +0.006994045]**. See the
[comparison artifact](../../../outputs/autoresearch/triplet-sampling/experiments/run-level-stratified-jitter-20260919T183826Z-f211c7ce/comparison/comparison.json).

The archived run retained 93,846 valid triplets over 3,503 visits in every training
epoch: 938,460 triplets / 35,030 visits over ten epochs. Validation used 24,188
triplets on 904 studies at x1. Its validation selection hash matched the baseline
on all ten rows, while its ten training selection hashes differed. The model
completed 4,380 optimizer steps. Recorded measured compute was 1,575.96 seconds,
within the fixed 1,785-second ceiling. These are measurements of the archived run,
not a new replay of the production port.

This is one training seed on one development fold. The bootstrap conditions on
fitted models, excludes training-seed and checkpoint-selection variability, and
does not correct repeated candidate selection on fold 0. The actual increase in
unique selected slice locations was not recorded; broader realized coverage is
an interpretation, not a verified causal explanation. The 2026-09-20 ordinary
overnight queue used the old sampler and is not a replication of this result.

## Comparison against the latest overnight controls

The same archived trained sampler model (AUC 0.897514923, seed 4061686476) was
compared with both controls from `overnight-20260919T235434Z-8ol_1c30` using saved
predictions, 2,000 paired report-group draws and bootstrap seed 20260914.

| Reference | Reference AUC | Sampler minus reference | 95% interval | Artifact |
|---|---:|---:|---|---|
| Control A, seed 1133376617 | 0.893221725 | +0.004293197 | [+0.000412218, +0.007858145] | [Comparison](../../../outputs/analysis/s/stratified-sampling-vs-overnight-20260920-00-control/comparison.json) |
| Control B, seed 2263738815 | 0.896581669 | +0.000933253 | [-0.002770911, +0.004524506] | [Comparison](../../../outputs/analysis/s/stratified-sampling-vs-overnight-20260920-13-control-repeat/comparison.json) |

The interval is positive against A and inconclusive against B. This adds control
comparisons, not a new sampler training seed or independent confirmation. The gain
remains provisional. The comparator explicitly permits the archived sampler-depth
field, preprocessing identity, run name, training seed and output-root differences;
other unexpected configuration and input-identity differences remain forbidden.

## Integration verification

Review confirmed that the production diff changes only the target identity and
training percentile construction in the sampler. The existing dataset already
passes the epoch, so no caller or config API change is needed. Static lint and
configuration loading were checked; repository policy prohibits writing or running
tests. No training, GPU experiment, submission or autoresearch restart was launched
during that integration step. The later user-launched run is recorded below.

## Second sampler run, seed 42

The user trained `outputs/training/another-stratified-sampling` with the promoted
implementation. Both runs use fold 0, validation ×1, 3,503 training studies,
904 validation studies, ten epochs and 4,380 optimizer steps. The new run used
seed **42**, rather than a newly generated random seed; the original winner used
4061686476. Neither run resumed or limited the training set.

| Metric | Original sampler | New sampler run |
|---|---:|---:|
| Selected epoch (1-based) | 10 | 10 |
| Macro AUC | 0.897514923 | 0.895176803 |
| Soft-target BCE | 0.481342629 | 0.478864580 |
| Recorded training seconds | 1,539.66 | 1,576.29 |

New minus original AUC is **−0.002338119**. The paired report-group bootstrap
uses 2,000 draws across 852 groups, seed 20260914: mean delta −0.002352381,
95% percentile interval **[−0.006064169, +0.001371058]**. Every draw was valid.
Saved predictions reproduce both recorded selected-epoch AUCs. Full per-target,
vendor, timing and provenance results are in the
[comparison artifact](../../../outputs/analysis/a/another-stratified-vs-original-20260920/comparison.json).

Largest point-estimate drops are lateral meniscus (−0.011011) and synovitis
(−0.010139); lateral OA increases by +0.005658. These per-label differences are
descriptive, not separately established effects. Soft-target BCE improves, while
the principal ranking metric decreases.

Configuration differences are seed, run name, output path and removal of the
research-only `stratification_depth` field. Archived source review confirms the
training-path changes only fix that depth at ten and remove its configuration,
call argument and metadata; the selection formula remains the same. Other changed
package files are autoresearch orchestration, outside this user-launched training
path. The comparator permits the resulting preprocessing-identity difference.

This second seed did not reproduce the original winner's high score. The interval
includes zero, so the comparison does not establish a regression. Both runs use
the sampler; this comparison alone does not isolate its benefit over the previous
sampling method. Bootstrap uncertainty excludes training-seed and model-selection
variability. No training, inference, submission or campaign restart was launched
for this comparison.

### Second run against the overnight controls

The same new seed-42 model was also compared against both old-sampler controls
from `overnight-20260919T235434Z-8ol_1c30`. Both comparisons used 2,000 paired
report-group draws across 852 groups, bootstrap seed 20260914, with all draws
valid and saved predictions reproducing the selected-epoch metrics.

| Reference | Reference AUC | New sampler minus reference | 95% interval | Artifact |
|---|---:|---:|---|---|
| Control A, seed 1133376617 | 0.893221725 | +0.001955078 | [−0.001943214, +0.005604234] | [Comparison](../../../outputs/analysis/a/another-stratified-vs-00-control-20260920/comparison.json) |
| Control B, seed 2263738815 | 0.896581669 | −0.001404866 | [−0.005465397, +0.002501534] | [Comparison](../../../outputs/analysis/a/another-stratified-vs-13-control-repeat-20260920/comparison.json) |

The new result lies between the two controls, and both intervals include zero.
Together with its inconclusive comparison against the original sampler, this run
does not establish either a sampler benefit or a regression. Configuration
differences are name, seed and output root; the changed sampler source is reflected
in the explicitly allowed preprocessing identity. Training-seed variability and
selection of the original winner are not captured by these bootstrap intervals.

## Public comparison of the earlier checkpoints

The original sampler and overnight control B both completed Kaggle scoring at
**0.933**, using ×4 images, attention temperature 0.75, batch size 4 per GPU,
16 total workers and no reflection TTA. Their submission IDs are 56386619 (S39,
sampler) and 56386217 (S38, control B). Records are
`outputs/submission/records/stratified-sampling-20260920-x4-tau075.json` and
`outputs/submission/records/control-b-20260920-x4-tau075.json`. The new seed-42
sampler run above has not been submitted by this task.
