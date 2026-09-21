# Triplet-sampling prior work

Current reference: learned-strength-20260921, selected by the user on 2026-09-21;
the campaign remains stopped. See [checkpoint closure](../../checkpoint-closure.md)
for source pins and state. Reference selections below retain historical provenance.

Read the relevant source before proposing a related experiment. This is an evidence
map, not a second actionable backlog. The [initial ideas](seed-ideas.yaml) transfer
the two eligible general-backlog proposals into this campaign for orchestrator
ranking before fresh research. The live registry records subsequent claims and outcomes.

## Starting implementation

The current recipe uses a physical study centre and a truncated Gaussian whose
width is `0.867 * limiting_reach(pool)`, with stochastic training draws and
deterministic validation draws. Nominal per-type budgets are 6/6/5/5/5/5, with a
study-total floor of 26. Neither the floor nor the nominal sum establishes the
realized cohort average. See the
[implemented sampler](../../../docs/data-preparation/physical-sampling/implemented-sampler.md)
and [recipe](../../../configs/training.yaml).

Some older paragraphs still quote fraction 0.76. Use the current recipe and the
[adopted result](../../../docs/validation/results/2026-09-19-relative-sampler-width-result.md)
for the starting implementation; retain older measurements under their actual setup.

## Evidence and previous ideas

| Direction | Evidence status and implication | Source |
|---|---|---|
| Relative width and common-factor sigma sweeps | Implemented and trained; another scalar sweep needs a new reason. This does not settle distribution shape. | [Relative width](../../../docs/validation/results/2026-09-19-relative-sampler-width-result.md), [sweep](../../../docs/validation/results/2026-09-19-fold-ensembles-and-sigma-sweep.md) |
| Peripheral evidence and acquisition asymmetry | Diagnostic evidence; tied to historical samplers and observational caches. | [Peripheral coverage](../../../docs/data-preparation/peripheral-coverage.md) |
| Per-type centre offsets | Rejected from the importance-map analysis without a training run; do not describe as a trained negative. | [Spatial importance map](../../../docs/data-preparation/peripheral-coverage.md#spatial-importance-map) |
| Per-plane width effects | Cache-only inference results; these did not predict the training-versus-drawing decomposition. | [Cache result](../../../docs/validation/results/2026-09-18-per-plane-sigma-cache.md) |
| Fixed-total reallocations, random type allocation, repeated-series share | Previous negative or tied implementations; inspect the exact allocation and recipe before revisiting. | [Closed allocation investigation](../../../docs/history/triplet-budget-allocation.md) |
| Per-plane ratios and asymmetric coronal draw | Imported into this campaign as `geometry-informed-plane-widths` and `coronal-side-relative-draw`; both trained 2026-09-19 and inconclusive (+0.00041 and −0.00130). This campaign owns their execution. The general backlog no longer carries them, and the repository now records the outcome in the sampler-shape document. Use one justified plane policy, not a multiplier sweep. | [Sampler shape](../../../docs/data-preparation/peripheral-coverage.md#what-is-settled-and-what-is-not), [geometry](../../../docs/data-preparation/peripheral-coverage.md#acquisition-geometry) |
| Side-conditioned sagittal sampling | Existing blocked proposal: the importance map needs a mirrored re-export and laterality plumbing. | [Position and knee side](../../../docs/backlog/big-bets.md#position-and-knee-side) |
| Run-level stratified jitter | Rejected for adoption on 2026-09-20 after inconclusive confirmation; original provisional acceptance withdrawn. Do not repeat unchanged. | [Decision and evidence below](#run-level-stratified-jitter-rejected-for-adoption) |

## Run-level stratified jitter: rejected for adoption

On 2026-09-20 the user withdrew adoption of `run-level-stratified-jitter` and
reverted its main-tree implementation. The permanent idea remains `completed`,
with decision `reject`. This is an explicit user adoption decision after
inconclusive confirmation, not a claim of statistically established harm or a
change to the campaign's numerical acceptance rule.

The original experiment, `run-level-stratified-jitter-20260919T183826Z-f211c7ce`,
scored 0.8975149225686027 with seed 4061686476. Its original positive bootstrap
against the historical control and provisional acceptance remain valid historical
records. The later external run, `outputs/training/another-stratified-sampling`,
scored 0.8951768033438978 with seed 42. This was not a fresh-seed campaign
confirmation. All comparisons below use fold 0, validation ×1 and 2,000 paired
report-group draws over 852 groups, with bootstrap seed 20260914.

| Second sampler compared with | Macro-AUC delta | 95% paired interval | Result |
|---|---:|---|---|
| Control A (`00-control`, seed 1133376617) | +0.0019550781521661 | [−0.0019432137031415, +0.0056042339807254] | Inconclusive |
| Control B (`13-control-repeat`, seed 2263738815) | −0.0014048660824299 | [−0.0054653973032351, +0.0025015338691042] | Inconclusive |
| Original sampler | −0.0023381192247049 | [−0.0060641693855424, +0.0013710579882692] | Inconclusive |

Controls A and B are under
`outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/`.
These intervals condition on the fitted models; they do not measure seed variance
or correct selection on fold 0. The evidence does not establish a repeatable
advantage sufficient to retain the added mechanism. Do not rerun this implementation
or draw more seeds to recover a win. Revisit only with new evidence or a materially
changed mechanism, recorded under a new idea ID with this precedent.

At rejection revision 176, the stopped campaign's incumbent was restored to
`historical-control` using its original frozen source, predictions and sealed limits.
The user subsequently selected control B at revision 177, as recorded under
[reference evidence](#reference-evidence). The prepared
`prefix-balanced-stratum-order` is blocked and removed from the queue because its
accepted-parent dependency is no longer satisfied. The already-trained
`plane-coupled-depth-targets` is held for explicit review: both its source and
pinned comparison reference include the withdrawn sampler. Neither follow-up is
assigned a negative experimental verdict. Preserve its original parent, source,
reference and evidence; future work requires renewed review and, where applicable,
a new reconciled attempt. `physical-span-standardized-triplets` remains queued.

The append-only
[decision record](../../../outputs/autoresearch/triplet-sampling/decisions/user-reject-run-level-stratified-jitter-20260920T093252Z-093b47be/decision.json)
contains exact metrics, evidence hashes, the previous/current incumbent, the prior
state snapshot and the parked follow-ups. The corresponding entry in
[campaign findings](findings.md#user-reject-run-level-stratified-jitter-20260920t093252z-093b47be)
supersedes earlier adoption guidance without deleting it. Full model and bootstrap
evidence remains in the
[experiment result](../../../docs/validation/results/2026-09-20-stratified-sampling-promotion.md).

## Preparation scope amendment

The user authorized the queued `physical-span-standardized-triplets` scope repair
on 2026-09-20. Revision 180 adds `preparation.py` to both the campaign definition
and saved policy, limited to nonconsecutive-triplet validation and its immediate
callers within the editable source. Historical prose approval had not granted the
executable permission. The [decision and recovery snapshots](../../../outputs/autoresearch/triplet-sampling/decisions/user-authorize-triplet-preparation-20260920T192746Z-e03bfe70/decision.json)
record the amendment; the [developer instructions](campaign.yaml) define its
constraints. The queued idea still needs implementation, measurement and normal
review. The campaign remains stopped with the same numerical seal, incumbent,
experiment history and blocked follow-ups.

## Reference evidence

The incumbent selected on 2026-09-20 was **control B**, `control-b-20260919T235434Z`, selected by
the user on 2026-09-20 at campaign revision 177. Its run is
`outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat`:
seed 2263738815, fold 0, validation ×1, selected epoch 9, macro AUC
0.8965816694263277 from saved predictions. Its prediction SHA-256 is
`d8de136017222ddb930638674aa543e959b148983f97f1866c077b7889217a24`.
The next candidate uses the existing single-incumbent paired bootstrap against B;
a provisionally accepted candidate can replace it normally. B is the stronger of
two controls and remains one fitted training seed, not independent proof of the
recipe's expected performance.

Control A, `00-control` under the same overnight directory, remains supporting
evidence: seed 1133376617, selected epoch 7, macro AUC 0.8932217251917317.
Both controls are listed in the campaign's evidence inventory. Their effective
recipe matches the sealed recipe apart from run name, output path and seed.

For B, input identities match the original reference exactly. All model, data and
training-entry Python source matches the sealed source; the archived differences
are the autoresearch CLI, supervisor and overnight launcher. Development therefore
continues from the verified sealed source, while comparison uses B's saved
predictions. Its original source archive remains attached to the control run.
The [replacement record](../../../outputs/autoresearch/triplet-sampling/decisions/user-adopt-control-b-20260920T095302Z-1c278cc7/decision.json)
preserves the previous state, compatibility checks and evidence hashes. Earlier
experiment references and decisions are unchanged, the two follow-ups remain
blocked, and the campaign remains stopped.

The primary historical control is `00-control` under
`outputs/training/overnight-20260919T120130Z-i2oqlnls/runs/`: class-balanced BCE,
fold 0, validation ×1, best macro AUC 0.894037, seed 927831896.
It remains `reference_run` for the original seal and fixed budget: 1,785 seconds,
ten epochs and 4,380 optimizer steps. Its predictions are retained as historical
evidence; replacing the comparison incumbent does not change those limits.
`outputs/training/balanced-bce/` is additional evidence at 0.895282, seed 42.
They differ in source as well as seed, so their difference is not a pure seed-noise
estimate. The older run also required prediction and timing handling in its saved
comparison; do not silently treat its artifacts as interchangeable.

The [queue result](../../../docs/validation/results/2026-09-19-untried-hyperparameter-queue.md)
records those limitations and the paired bootstrap comparisons. Its mixed-variant
spread is not a measured training-seed variance. Its discussion of more epochs does
not authorize increasing this campaign's compute budget.
