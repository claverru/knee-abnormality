# Sampling prior work

Evidence map for the [sampling campaign](contract.md); not a backlog. Old-split
scores are not comparable in absolute value with the combined-label,
scanner-split reference.

## Triplet-sampling campaign

Eight experiments on `all_5fold_v3`, none retained
([results](../triplet-sampling/findings.md), [prior work](../triplet-sampling/prior-work.md)).
All used fresh seeds: three against the historical control (0.894037) and four
against the `run-level-stratified-jitter` incumbent (0.897515), whose adoption was
later withdrawn after a second run scored 0.895177. Four same-recipe controls
average 0.8948 (0.894037, 0.895282, 0.893222, 0.896582). Single-run verdicts from
that campaign are therefore not evidence against their mechanisms.

## Initial ideas

The seven user-listed ideas are in [`seed-ideas.yaml`](seed-ideas.yaml) and
[`seed-ideas-2.yaml`](seed-ideas-2.yaml); startup imports both for orchestrator
ranking. Their exclusions are listed in the first file's `exhausted_directions`.

## Crop history

Learned crop selection was dropped on 2026-09-16: no run beat the full-slice model,
learned placement was worth at most about 0.005, and the attention value map was
weak ([investigation](../../../docs/history/crop-selection-investigation.md)).
Crop proposals here keep the 140 mm window at 0.625 mm and place it from measured
anatomy ([input resolution](../../../docs/research/geometry-and-views/input-resolution.md#field-of-view-and-joint-position)).

## User submissions

On 2026-09-22 the user submitted the campaign's `run-level-stratified-jitter` checkpoint
(fold 0 0.899976, +0.000934 against `k_05`, rejected by the shared rule) as S43, exported
from its frozen candidate source at x4, tau 0.75. It scored 0.944 public against 0.945 for
its reference `k_05` (S41). This is a leaderboard observation, not an adoption; the
campaign verdict is unchanged.

## Implementation notes carried over

- Judge shape changes by realized depth histograms from the CPU replay; targets
  snap to existing slice geometry through `geometry_groups` and the assignment.
- Scalar `truncnorm.ppf` costs about 24x a vectorized call per element; budget any
  per-element loop against the runtime ceiling.
- Editing `SAMPLING_IDENTITY` or any preprocessing source changes
  `preprocessing_identity`; comparisons allow that key.
- Sampler code must live in modules listed in `identity.PREPROCESSING_SOURCES`; a
  separate helper directory would escape the source hash.
