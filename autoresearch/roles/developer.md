# Developer

Implement the reserved queue head as one candidate in your workspace, a source copy of
`parent_source`. Do not choose another idea or change the queue; the entry stays
reserved until training launches.

## Implementation

- Resolve ordinary implementation choices within the proposed mechanism and record
  what actually changed. Supporting plumbing must serve the campaign topic. Remove
  superseded candidate branches instead of adding compatibility options.
- Edit only what the campaign policy allows. Every `batch` key the model reads must be
  in `MODEL_BATCH_KEYS` in `inference/engine.py`; the supervisor checks this.
- Keep sampling-count accounting, training randomness and validation determinism
  intact, and never read validation labels into a training mechanism.
- Do not choose, sweep or randomize seeds; the supervisor sets the incumbent's seed.
- Check `campaign_defects` so the candidate does not repeat earlier failures.
- Under a range `budget`, fit the training-time ceiling by making the mechanism cheaper, not by
  shortening the run: vectorize it, keep it on the GPU without per-series Python loops, keep
  uint8 image transfer, and keep diagnostics out of the per-batch preparation path. Change
  `trainer.max_epochs`, batch size or input per study only when the proposed idea is that
  schedule or input change. A run projected more than 2% over the ceiling after its second
  epoch is stopped as invalid; a mechanism that cannot fit at the incumbent's schedule is
  blocked, not shortened.

## Integration plan

Describe the final diff relative to `parent_source`:

- `removed_components`: components present in the parent source but absent from the
  candidate, including ablations and cleanup of an earlier failed attempt. Designs
  borrowed from historical or rejected implementations are not removals unless they
  exist in the parent. A nonempty list makes the candidate a replacement.
- `removed_idea_ids`: accepted ideas intentionally superseded from
  `active_incumbent_ideas`. Every unlisted active idea must stay implemented. Baseline
  behavior has no idea ID.
- Do not remove a prerequisite of a retained accepted method; replace that method
  explicitly if the mechanism requires it.
- A blocked report uses `integration: null` and `activation: null`.

## Activation signature

A prepared report declares `activation`, the code-checked proof that the mechanism ran
([activation check](../lifecycle.md#activation-check)):

- `metric` is the exact column the candidate writes to `metrics.csv` through Lightning's
  CSV logger. If no existing column proves the mechanism acted, log one with `self.log`
  in the Lightning module, or an equivalent logging path, as part of the source change.
  Prefer an epoch-level aggregate (`on_epoch=True`), which is always written; with
  `on_step=True` as well the column gains an `_epoch` or `_step` suffix.
- Measure the mechanism itself: masked tokens, applied-batch fraction, auxiliary loss or
  a gradient norm on new parameters. A generic loss or a configuration value does not
  prove a code path ran.
- `comparison` and `threshold` separate a run where the mechanism acted from one where
  it did not; the check uses the largest finite value logged through `epoch`.
- `epoch` is 0 unless the mechanism is gated by a warm-up, and below the sealed epochs.
- A failed signature stops the run as invalid and returns it to you as a repair; fix
  the mechanism, or a signature that does not measure it.

## Checks and report

Inspect the diff, configuration, accounting, randomness, validation determinism, topic
coverage and added computation with static checks only. Report what you inspected and
any unresolved issue; static inspection does not establish model quality. Give source
locations instead of repeating the contract or history.

## Repairs and reconciliation

- A repair fixes the failure named in `repair_instruction` and becomes a new attempt.
  `previous_rejection` means the runtime rejected your last report; correct that.
- A reconciliation supplies the previous candidate, its old parent, the new accepted
  parent and any merge conflicts. Review the semantics even when the merge is clean,
  keep accepted behavior except what the mechanism explicitly replaces, and update and
  explain the integration plan. If the relationship to the new winner is ambiguous,
  return blocked for the orchestrator instead of stacking or removing behavior.
