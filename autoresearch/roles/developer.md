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
- A blocked report uses `integration: null`.

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
