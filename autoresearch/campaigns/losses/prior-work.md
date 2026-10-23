# Shared-loss prior work

Current reference: learned-strength-20260921, selected by the user on 2026-09-21;
the campaign remains stopped. See checkpoint closure (`autoresearch/checkpoint-closure.md`, removed)
for source pins and state. Reference selections below retain historical provenance.

Campaign prepared on 2026-09-20; no loss candidate has been implemented or trained
by this campaign. The [contract](contract.md) excludes per-label loss selection,
coefficients and schedules by explicit user decision. The
[initial ideas](seed-ideas.yaml) are proposals for startup review, not measured wins.

## Starting control

The recipe uses masked soft-target BCE. Fixed weights from training-fold class
counts multiply each observed BCE cell according to the target's `>= 0.5` group;
the target itself remains soft. The current implementation is in
[lightning.py](../../../src/knee_abnormality/training/lightning.py); the measured
baseline and its limitations are recorded in the
[balanced-BCE result](../../../docs/validation/results/labels-and-losses/2026-09-19-balanced-bce-result.md).
Use that stronger objective as the control, not historical ordinary BCE.

The initial reference is control B from
`outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat`:
fold 0, validation x1, seed 2263738815, selected epoch 9, macro AUC
0.8965816694263277. Prediction SHA-256:
`d8de136017222ddb930638674aa543e959b148983f97f1866c077b7889217a24`.
It is also the current reference of the separate sampling campaign. No sampling
candidate or withdrawn stratified sampler is imported into this line.

Control A under the same overnight directory scored 0.8932217251917317 with seed
1133376617. B is the stronger fitted control, not independent evidence of the
expected recipe score; retain A as supporting evidence instead of averaging or
switching references opportunistically. See the
[sampling reference record](../triplet-sampling/prior-work.md#reference-evidence).
This new campaign seals its own limits against B; it does not change the sampling
campaign's original 1,785-second ceiling or its state.

## Evidence and ownership

| Direction | Evidence and campaign treatment | Repository source |
|---|---|---|
| Fixed class-balanced BCE | Already adopted; start here. Its measured gain does not establish that stronger weighting is always better. | [Balanced-BCE result](../../../docs/validation/results/labels-and-losses/2026-09-19-balanced-bce-result.md) |
| Shared BCE plus ranking | Imported as `shared-bce-ranking`; the completed comparison was inconclusive. This campaign owns its execution and result. | [Ranking-aware loss](findings.md#completion-20260920t121537z-5b39b418) |
| Shared focal objective | Proposed here as `shared-balanced-focal`. External ACL/MCL results motivate objective exploration but do not establish a macro gain or the winning shared loss. | [Per-label research motivation](../../../docs/research/training-objectives/per-label-losses.md#trusted-motivation) |
| Tempered class weights | Import only the objective-weighting portion as `shared-tempered-balance`. Study sampling and batch composition remain outside this campaign. | [Weighting and sampling](../../../docs/backlog/untried-ideas.md#supervision-and-labels) |
| Loss schedule | `shared-delayed-ranking` applies the same within-run schedule to all labels. It is not the proposed frozen-head screen, checkpoint continuation, or compositional AUC algorithm. | [Loss extensions](../../../docs/research/training-objectives/per-label-losses.md#candidate-losses-and-extensions) |
| Per-label selection, classifier banks and mixtures | Explicitly excluded. Do not import the per-label big bet, create twelve variants, or recover it through label-specific coefficients. Its existing plan is not owned by this campaign. | [Per-label plan](../../../docs/backlog/untried-ideas.md#supervision-and-labels) |
| Asymmetric loss | The survey records a poor external result, not a local matched-control measurement. Revisit only with a material change or new evidence; do not seed it merely to fill the queue. | [Objective survey](../../../docs/research/training-objectives/auc_loss_approaches.md) |

The [objective survey](../../../docs/research/training-objectives/auc_loss_approaches.md) is the main
research map. Its historical priority order is not an experimental result. Future
ideas may include other shared ranking surrogates or training-only memory pools
when the evidence justifies them. Dedicated optimizer replacements, extra data
passes, cached-feature fits, and tuning controllers using validation labels do
not fit this contract. Shared learned combinations need a concrete training-only
signal and a matched budget before they become actionable; jointly minimizing
mixture coefficients with the same weighted loss is not evidence of better AUC.

## Pair coverage and interpretation

Microbatches contain eight studies. The existing
[pair calculation](../../../docs/research/training-objectives/per-label-losses.md#ranking-needs-usable-study-pairs)
estimates no binary positive/negative pair in 15.2% of ACL batches, 26.3% of MCL
batches, and 55.0% of fracture batches under uniform study sampling. These are
analytical expectations from saved training counts, not observed frequencies.
Retaining BCE in a hybrid supplies a pointwise gradient when pairs are absent.
Measure coverage; class weights and extra slices do not add independent pairs.

Use one shared rule for pairs and soft-label confidence, record component scales,
and preserve masks. Different per-label deltas are diagnostics, not permission to
choose a separate loss for a label. An inconclusive macro result remains
inconclusive even if one target improves strongly. Keep full negative and invalid
results so a later researcher does not rename an attempted mechanism.
