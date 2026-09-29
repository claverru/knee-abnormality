# Topics

Inputs of the [topic loop](../topic-rotation.md). The loop reads them only when a topic rotates;
it never waits for them.

| Read when | Document |
|---|---|
| Adding or checking a loose seed idea every researcher topic draft must take when it falls inside the draft's scope | [Seed pool](seed-pool.yaml) |
| Dropping a topic for the loop to take next | [Queue](queue/) |

## Dropping a topic

Create `queue/<campaign-id>/` (lowercase, digits and hyphens; queued topics are taken in name
order) with:

- `topic.yaml`: `title`, `backlog` (`refill_below`, `refill_to`, `seed_order`, `rotate_after`),
  `additional_evidence_runs` and `policy`, exactly as in a
  [campaign.yaml](../campaign-setup.md#declarative-policy). The loop writes `id`, recipe,
  reference run, `chain` and `budget` itself: every topic starts on the current chain incumbent
  and keeps its predecessor's [training budget](../campaign-setup.md#training-budget).
- `contract.md` and `prior-work.md`, as for a [campaign](../campaign-setup.md#files-and-startup).
  The loop appends the starting point to the contract.
- `seed-ideas*.yaml`, in the idea-report schema (`IdeaReport`).

A queued topic is checked for validity only (the same validation as `validate`, protected list
included), not against the naive-topic checks. One that fails moves to `rejected/` with
`rejection.txt`. An accepted one is copied into `autoresearch/campaigns/<id>/` and leaves the
queue when its topic takes the GPU.
