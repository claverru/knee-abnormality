# Idea lifecycle

This is the path of one idea through a running campaign: claim, development, gates,
review, reconciliation, launch, comparison and completion. Who decides what, and the
adoption rule, are in the [protocol](protocol.md); record fields are in
[records](records.md); commands and recovery are in [runtime operation](runtime.md).

## Concurrency and workspaces

The supervisor is ordinary code and outlives agent sessions. It owns process
launching, GPU checks, locks, completion detection, numeric evaluation and every
durable state transition. Per campaign it runs at most one training, one developer
and one researcher refill, with one candidate developed or prepared ahead of the
running experiment. Campaigns share a GPU lease, so two cannot train on one device.

Each claimed idea gets its own workspace and, once prepared, a frozen source copy.
Nothing switches branches in a live checkout, and the repository's uncommitted
source works as is. Jobs reuse the locked Python environment with `PYTHONPATH`
pointing to the candidate source; experiment code is never imported from the main
checkout, and a winner never replaces main-checkout code automatically. Training
runs from an immutable snapshot, so preparing the next candidate cannot alter it.

Pending events are handled in order: completions, then reviews, then proposals.
While an adoption's control retrain is pending, reviews wait and no candidate trains
([seed policy](protocol.md#training-seed-policy)).

## Startup and proposals

Startup first seals limits with the CPU replay
([sealing](runtime.md#initial-sealing-and-fixed-budget)), then imports the campaign's
`seed-ideas*.yaml` files as proposal events before any fresh research. With
`seed_order: ranked` the orchestrator admits and ranks them like researcher
proposals; with `forced` they are queued in file order
([campaign setup](campaign-setup.md#files-and-startup)). The orchestrator preserves
their original links, checks prior outcomes and blockers, and records campaign
ownership. General-backlog entries are read-only precedents; adopting one records
its link and campaign ownership in the idea registry and resolves overlapping
active work first.

Proposal admission is checked against live state, never a stale snapshot:

- A proposal needs precedent references and a material difference from them.
  Duplicate IDs in the report, the registry or pending events are rejected.
- Each dependency must have a current idea decision of `provisional_accept` and
  remain active in the incumbent. Queue admission, implementation approval and
  historical acceptance of since-replaced behavior do not count; evidence-only links
  are precedents, not dependencies. Forced seeds cannot declare dependencies.
- The orchestrator receives `proposal_admission`, the dependency status from the
  same gate that checks its response. Unresolved proposals stay `proposed`, outside
  both the accepted and ranked lists, even if they would rank last.
- A changed ranking lists the complete eligible queue, with at most `refill_to`
  non-seed entries; `ranked_ids: null` keeps it. Surplus proposals stay in the
  registry outside the queue. Forced seeds keep their order ahead of other ideas and
  can be parked only by blocking them at their own review.

## Claim and refill

When nothing is claimed, the supervisor claims the queue head: it copies the
incumbent source into a new attempt workspace, records the incumbent as the
implementation parent and marks the idea `developing`. The head stays in the queue.
It is fixed until launch or explicit parking; only the unclaimed tail can be
reranked, and an unrelated event cannot park it.

Refill thresholds are per campaign: `backlog.refill_below` and `backlog.refill_to`
in `campaign.yaml`. Queue length counts the claimed head; launched and blocked ideas
do not count. When the queue drops below `refill_below`, one researcher is asked for
`refill_to` minus the queue length, running alongside development and training. A
refill starts only after new evidence (a launch, a decision, a parked idea or a
finished control retrain), so a resume or an empty refill does not respawn it; an
empty report is kept without an orchestrator call. No refill runs while forced seeds remain.

## Development and gates

The developer receives its idea, relevant predecessors, the campaign's developer
instructions and `campaign_defects` (below), and returns a prepared or blocked
report. A prepared report declares an integration plan relative to its parent:
retained, added and removed components, removed accepted idea IDs, and rationale.
A candidate replaces parent behavior exactly when `removed_components` is nonempty,
pure ablations included; otherwise it adds and must keep every incumbent method.
Removed idea IDs must be active, and required dependencies of the candidate and of
every retained accepted method must remain.

Code then gates the candidate before any review: source allowlist and scope check,
Python syntax, a required source change when the policy says so, the batch-key check
([boundaries](runtime.md#boundaries-and-prerequisites)), and a sandboxed configuration
gate against the campaign policy. A failed gate goes to review as a gate error; the
orchestrator cannot approve over it. A blocked developer report also goes to review.

`campaign_defects` lists the last twelve repair instructions and invalid-run reasons
of other attempts in the campaign, with training log paths. Developers and candidate
reviews receive it so a defect repaired once is not reintroduced by the next idea.

## Review and approval

The orchestrator inspects the real diff and returns `approve`, `repair` or `block`.
Approval requires an explicit integration review explaining how the diff implements
the declared components; a clean merge is not proof of semantic integration. Each
attempt gets at most one review repair, which starts a new attempt from the repaired
workspace. `block` must park the idea with its reason.

Approval binds the implementation parent, source and configuration fingerprints,
reference predictions and input identity, integration plan and resulting active-idea
inventory, the incumbent seed, fixed limits, headers and watched external evidence.
The binding is checked when the review returns and again immediately before launch.
Changed inputs send the candidate back to review with the reason; an old approval
never authorizes a changed candidate. Source drift under an unchanged parent identity
pauses rather than being treated as an improvement.

Watched external evidence is the hashes of runs in `additional_evidence_runs` plus
other initialized campaigns' incumbents and decisions. It is evidence, never an
automatic cross-campaign adoption. Unlisted manual runs are not discovered; add them
to `additional_evidence_runs` while the campaign is stopped.

## Reconciliation

A candidate prepared while another run was training may have an older parent.
Before launch, a newly accepted incumbent triggers reconciliation into a new attempt:
a three-way merge of the candidate's changes onto the new incumbent, keeping the old
parent, old candidate, new parent and any conflicts. Every merge, clean or conflicted,
returns to the developer to verify the intended addition or replacement, then passes
fresh gates and review under the new incumbent's seed. Superseded attempts stay
addressable and are not counted as trained failures. Nobody edits a running
snapshot or claims an old candidate already contained later wins.

The incumbent carries the inventory of active accepted idea IDs. On acceptance, code
keeps every active ID except declared removals and adds the new idea. Historical
acceptances stay recorded when replaced but no longer satisfy dependencies. An
external starting reference has no campaign-owned idea IDs; name its components in
the integration plan instead of inventing acceptance decisions.

## Launch

A prepared, approved candidate launches when no training runs, the GPU is free and
no comparison or decision is outstanding. The launch rechecks the approval binding,
the reviewed source manifest, the sealed limits hash, the incumbent's prediction hash
and the configuration gate, then pins the incumbent as the attempt's reference and
trains on the campaign fold ([scripts](scripts.md#training)). Launch intent and
process identity are persisted so crash recovery neither launches an attempt twice
nor loses its queue entry. Once training writes its execution record, the idea
leaves the queue as `launched`; a failed launch leaves the head reserved for explicit
handling. The developer then takes the next head.

## Comparison and completion

Training completion triggers the CPU comparison against the pinned reference on the
campaign fold within the remaining per-experiment time
([scripts](scripts.md#control-versus-experiment-comparison)). Code computes the
verdict (`provisional_accept`, `reject` or `invalid`) under the
[adoption policy](protocol.md#adoption-policy) and opens a completion event carrying
it as `eligible_decision`.

A reject with complete evidence and a recorded approval goes to the
[restricted completion reporter](protocol.md#restricted-completion-reporting); every
other completion, and every escalation, goes to the orchestrator. Either way the
supervisor records the fixed `eligible_decision` itself: the orchestrator interprets
the result, records the finding and may rerank or park, but never applies or changes
a verdict. A completion whose reference is no longer the incumbent needs explicit
recovery. A `provisional_accept` makes the candidate the incumbent and starts the
control retrain; a reject or invalid run leaves incumbent and queue unchanged.

A training that exits nonzero before its first epoch records no result and returns
to the developer as a repair ([crashes before training](runtime.md#inspect-stop-and-resume));
other failed runs are invalid.

Agents exit when their assignment is done; the next event dispatches the next role.
