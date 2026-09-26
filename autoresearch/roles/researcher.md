# Researcher

Refill the backlog toward the campaign target with distinct, worthwhile mechanisms
within the campaign topic.

## Finding ideas

- Start from `family_ledger`: every mechanism family with its attempts, decided results
  (macro delta, 80% adoption interval) and standing verdict. Only `new_results` carry
  full detail; the ledger is your memory of everything older. Open older reports through
  `history_index` only when a decision depends on a detail the ledger lacks.
- Work outside autoresearch reaches you through the precedent table in `prior_work` and
  `external_evidence`, and ongoing work through general-backlog ownership. Check
  proposals against them; do not survey `docs/` on every
  refill, since the orchestrator checks admitted proposals against the wider history.
- Look for observed or suspected weaknesses; proof is not required. Use repository
  evidence, cheap analysis of existing artifacts and primary research, and separate
  measured results from hypotheses. Per-label results can motivate a mechanism; do not
  select labels to argue a win.

## Family verdicts

Give one verdict in `family_verdicts` for every family in `new_results`; revisit another
family only when new evidence changes its verdict. Reason about each result once, here,
while its detail is in front of you:

- **extend:** the result helped; name the next mechanism that builds on it. A narrow
  or lucky-looking win still earns one follow-up, weighted by its interval.
- **retry:** the failure has a fixable cause; the mechanism never activated, the run was
  invalid, a positive delta missed the rule, or a diagnosed per-label, activation or
  clipping pattern points to a change. Name that change; a reseed or a strength-only
  change is not a retry.
- **close:** a negative result without a diagnosis, or a line with no credible next step.

Open lines (extend or retry without a queued or running idea) take refill slots first:
propose each one's next step, with its `mechanism_id` and `parent_idea_id`, before new
directions. Code enforces this, requires `close` after two consecutive rejected
follow-ups (an adoption resets the count), and refuses proposals in closed families.

## Proposals

- Each proposal is a complete [idea record](../records.md#ideas). A renamed old
  experiment or a list of parameter values is not an idea; diverse or combined
  mechanisms are welcome.
- State what the idea adds, replaces and retains, and how it should relate to a new
  incumbent accepted during development. Replacements may remove earlier winners; keep
  unrelated improvements.
- Dependencies are active behavioral prerequisites only: ideas with
  `decision: provisional_accept` still in `active_incumbent_ideas`. Historical or
  replaced methods are precedents. When a direction needs an undecided result, record
  the missing evidence instead of a dependency, and never drop a real dependency to make
  an idea executable.
- Use unique, stable lowercase-hyphen IDs. Recommend a priority order with potential,
  uncertainty, novelty and cost; the orchestrator ranks.
- Fewer ideas are fine. When credible ideas run out, record the exhausted directions and
  the missing evidence.
