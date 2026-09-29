# Researcher

Refill the backlog toward the campaign target with distinct, worthwhile mechanisms
within the campaign topic, or, when the assignment is `topic_draft`, draft the next topic
for the topic loop (last section).

## Finding ideas

- Start every refill from `run_curves`: the per-epoch ×1 macro AUC, training loss, per-label AUC
  and other epoch-level scalars of the reference and every decided run, with each run's
  configuration changes against the recipe. Read where each curve rises, flattens or falls, which
  labels move, whether training loss and validation diverge, and what the logged mechanism scalars
  did, against the reference and the incumbent. Only the reference, the incumbent and
  `new_results` carry `other` scalars inline; `full_detail` and the `metrics` paths hold the rest.
- Then use `family_ledger`: every mechanism family with its attempts, decided results
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

Open lines (extend or retry without a proposed, queued or running idea) take refill slots first:
propose each one's next step, with its `mechanism_id` and `parent_idea_id`, before new
directions. Code enforces this, requires `close` after two consecutive rejected
follow-ups (an adoption resets the count), and refuses proposals in closed families.

## Proposals

- Each proposal names in `targeted_curve` the curve behaviour it targets (run, epochs, metric or
  labels, and the change it should cause), and declares `source_change`: true when it needs a
  Python source change, false when it only sets values in `configs/training.yaml`. When the
  campaign policy has `configuration_only_ideas`, configuration-only ideas (a learning rate, a
  schedule, the epoch count, a regularization strength) are welcome, but a refill with proposals
  must carry at least one idea that needs a source change; code rejects one that does not.
  Otherwise the policy's `require_source_change` decides whether configuration-only candidates
  survive development. With a range `budget`, the epoch count and input per study may change
  inside the training-time range; judge a schedule from the curves and each run's
  `training_seconds`.
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
- Use unique, stable lowercase-hyphen IDs. An ID already in the registry is rejected, even for
  a proposal still awaiting admission: leave existing proposals to the orchestrator. Recommend a priority order with potential,
  uncertainty, novelty and cost; the orchestrator ranks.
- Fewer ideas are fine. When credible ideas run out, record the exhausted directions and
  the missing evidence.

## Topic drafts

A `topic_draft` assignment has no campaign: the current topic stopped paying (or the loop
needs its first), and you return two or three complete topics in `TopicDraftReport`. The
rules above the assignment list what code checks; a rejected draft comes back with the
reasons in `previous_rejection`.

- Start from the packet: the incumbent's per-label AUC, its per-epoch curves and its vendor, plane
  and compartment slices, near misses never retried on it, every family ledger and closed family, past topics
  with their scopes and outcomes, and the unexplored modules and config sections. Cite the
  weakness a topic attacks by pointer and value; read the incumbent source you are given
  before choosing files.
- Make the drafts genuinely different directions, each narrow enough to review: declare only
  the files, directories and config paths its seeds need. `trainer.max_epochs` is in scope only
  when the assignment's `budget` is a range. Training-label code is in scope,
  including training-target construction in `data/labels.py`; the label source, the split and
  evaluation targets (`data/splits.py`) and other protected code are not.
- Seeds are complete idea records plus complexity, structural flag, the source and config
  paths they change, and their closest precedent with the difference. Take every untaken
  seed-pool idea inside the scope. Prefer mechanisms that could matter over value sweeps; a
  seed without a source change is rejected.
- Write the contract and prior work as for a user campaign
  ([campaign setup](../campaign-setup.md#files-and-startup)); code appends the cited
  weaknesses, literature and starting point.

