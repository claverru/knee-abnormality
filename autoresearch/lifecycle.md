# Idea lifecycle

This is the path of one idea through a running campaign: claim, development, gates,
review, reconciliation, launch, activation check, comparison and completion. Who decides what, and the
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
in `campaign.yaml`. Every refill context carries `run_curves`, the per-epoch curves and
configuration changes of the reference and every decided run ([records](records.md#run-curves)),
and each proposal declares `source_change` and the `targeted_curve`. In a campaign with
`configuration_only_ideas` a nonempty refill whose proposals are all configuration-only is
rejected and retried. Queue length counts the claimed head; launched and blocked ideas
do not count. When the queue drops below `refill_below`, one researcher is asked for
`refill_to` minus the queue length, running alongside development and training, with
the results decided since its last report; it returns a
[family verdict](records.md#family-verdicts) for each, and open lines take refill slots
first. A rejected report retries the refill with the reason attached. A
refill starts only after new evidence (a launch, a decision or a parked
idea), so a resume or an empty refill does not respawn it; an
empty report is kept without an orchestrator call. No refill runs while a forced seed is still
waiting to launch; once the last one has launched, the refill starts beside its training so the
GPU does not idle after it.

## Development and gates

The developer receives its idea, relevant predecessors, the campaign's developer
instructions and `campaign_defects` (below), and returns a prepared or blocked
report. A prepared report declares an integration plan relative to its parent:
retained, added and removed components, removed accepted idea IDs, and rationale.
It also declares an **activation signature**: a column the candidate logs to
Lightning's `metrics.csv`, a comparison (`>` or `>=`) with a finite threshold, the
epoch after which it is checked (0 unless the mechanism waits for a warm-up) and a
rationale. When no existing column proves the mechanism ran, the developer adds one
with `self.log` as part of the candidate's source change. The check is described under
[activation check](#activation-check); the reason for a code-owned check is in the
[literature review](literature-mechanisms.md#2-a-declared-activation-signature-checked-after-epoch-0).
A candidate replaces parent behavior exactly when `removed_components` is nonempty,
pure ablations included; otherwise it adds and must keep every incumbent method.
Removed idea IDs must be active, and required dependencies of the candidate and of
every retained accepted method must remain.

Code then gates the candidate before any review: source allowlist and scope check,
Python syntax, a required source change when the policy says so (or, in a campaign allowing
configuration-only ideas, when the idea declared `source_change: true`), the batch-key check
([boundaries](runtime.md#boundaries-and-prerequisites)), a sandboxed configuration
gate against the campaign policy and an activation epoch below the candidate's epoch count. A
range budget is enforced during and after training, not before launch
([sealing](runtime.md#initial-sealing-and-fixed-budget)). A failed gate goes to review as a gate error; the
orchestrator cannot approve over it. A blocked developer report also goes to review.

`campaign_defects` lists the last twelve repair instructions and invalid-run reasons
of other attempts in the campaign, with training log paths. Developers and candidate
reviews receive it so a defect repaired once is not reintroduced by the next idea.

## Review and approval

The orchestrator inspects the real diff and returns `approve`, `repair` or `block`.
Approval requires an explicit integration review explaining how the diff implements
the declared components; a clean merge is not proof of semantic integration. It also
requires an activation review confirming, from the logging code, that the signature
measures the declared mechanism and that its threshold and epoch separate a run where
the mechanism acted from one where it did not. Each
attempt gets at most one review repair, which starts a new attempt from the repaired
workspace. `block` must park the idea with its reason.

Approval binds the implementation parent, source and configuration fingerprints,
reference predictions and input identity, integration plan, activation signature and
resulting active-idea inventory, the incumbent seed, fixed limits, headers and watched external evidence.
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
the reviewed source manifest, the sealed limits hash, the incumbent's prediction hash,
and the configuration gate, writes the
attempt's `run-limits.json`, then pins the incumbent as the attempt's reference and
trains on the campaign fold ([scripts](scripts.md#training)). Launch intent and
process identity are persisted so crash recovery neither launches an attempt twice
nor loses its queue entry. Once training writes its execution record, the idea
leaves the queue as `launched`; a failed launch leaves the head reserved for explicit
handling. The developer then takes the next head. The launch binding carries the
approved activation signature that the supervisor checks during the run.

## Topic rotation marks

Under the [topic loop](topic-rotation.md) the supervisor reads a `rotation.json` mark in its
output directory on every pass. `retiring` stops new claims (training repairs included) and
research refills while claimed and prepared work continues through review, launch and
training; `handover` also stops launches, so only the running training and its decision
remain before the loop stops the topic; `successor` holds launches of a new topic until the
old one has stopped. Events, reviews and completions are handled as usual in every phase.

## Activation check

While a candidate trains, the supervisor's polling loop reads the run's
`training/fold_<k>/logs/version_*/metrics.csv`; no agent waits for it. It ignores an
unterminated last line and rows whose width differs from the header, so a file that
Lightning is appending to or rewriting reads as a prefix of complete rows. Once a row
from a later epoch appears, the named epoch has finished, including its epoch-level
training rows written after validation, and the signature is evaluated once: it
passes when the largest finite value of the column over the rows logged through that
epoch satisfies the comparison, and a column never logged by then fails. A signature
still unevaluated when training exits cleanly, as for the last epoch, is evaluated
then over the same rows.

The outcome, with the observed maximum and the metrics path, is saved in
`run-checks.json` and on the experiment. A failure stops training through the job's
cancellation, records the attempt `invalid` with that reason and returns the idea to
the developer as a repair, like a
[crash before training](runtime.md#inspect-stop-and-resume); after two repairs it is
an ordinary invalid completion for the orchestrator. Either way the run is no evidence
against the idea: it never counts as a reject or toward
[family closure](records.md#family-verdicts). A pass is recorded in the evaluation
evidence of the completed run. Mid-run checks are one ordered list in
[`supervisor/monitoring.py`](../src/knee_abnormality/autoresearch/supervisor/monitoring.py):
the activation signature, then, in a campaign with a range budget, the training-time upper
bound, which stops a run that trains longer and takes the same invalid-and-repair route.

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
recovery. A `provisional_accept` makes the adopted run the incumbent, and later
candidates train at its seed ([seed policy](protocol.md#training-seed-policy)); a reject or invalid run leaves incumbent and queue unchanged.

A training that exits nonzero before its first epoch records no result and returns
to the developer as a repair ([crashes before training](runtime.md#inspect-stop-and-resume));
a failed [activation check](#activation-check) is invalid and also returns as a repair;
other failed runs are invalid.

Agents exit when their assignment is done; the next event dispatches the next role.
