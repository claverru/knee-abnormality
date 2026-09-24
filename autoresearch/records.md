# Research records

Paths in YAML and runtime records are repository-relative unless explicitly absolute.
Every state update carries a revision and event identifier so a resumed supervisor
can recognize work already completed. Structured agent report schemas live in
[campaign/reports.py](../src/knee_abnormality/autoresearch/campaign/reports.py); the runtime state and
immutable job artifacts preserve the mechanical transitions and exact evidence.
Runtime state also stores the sealed campaign policy from `campaign.yaml`; each
role assignment includes its permissions. A changed definition cannot silently
widen a running or resumed campaign's scope.

## Ideas

The campaign's `ideas.yaml` is the permanent idea registry, maintained by the
orchestrator from agent reports. Each entry records:

- `id`, `mechanism_id`, `parent_idea_id`, `title`, and `status`.
- The suspected opportunity, proposed mechanism, and why it could matter.
- Closest prior attempts and evidence references, with the material difference.
- How coverage and exploitation coexist, and how the fixed limits are preserved.
- An implementation direction, unresolved choices, and dependencies.
- Whether the proposal adds to or replaces behavior and how it handles a new
  incumbent. Dependencies name accepted methods still active in the incumbent;
  historical evidence links remain precedents.
- Creation state revision and any claimed experiment identifier.
- The researcher's priority recommendation and the orchestrator's ranking rationale.

Idea statuses are `proposed`, `queued`, `developing`, `prepared`, `launched`,
`completed`, `blocked`, and `retired`. Their experiment references and outcomes stay
in the registry after the corresponding queue entry is removed. A materially revised
follow-up gets a new idea ID under the same mechanism. Historical links alone do
not make an idea queued or authorize duplicate work.

`backlog.yaml` contains a `queue` list of unique idea IDs ranked by the orchestrator,
highest potential first. It is the work queue, not the research memory. The
orchestrator reranks unclaimed entries after new proposals or results and records why.
A claim reserves the first idea and marks it developing
without popping it. Its successful training launch removes the ID and records the
experiment reference. Launch failure leaves the head in place. The entire queue
length, including its developing or prepared head, determines the refill threshold.
The claimed head is fixed until launch or explicit parking; only its unclaimed tail
can change order. Launch removal never deletes the permanent idea record.

## Experiments

Use one immutable directory per attempt in
`outputs/autoresearch/<campaign-id>/experiments/<experiment-id>/`.
An attempt records its idea, implementation parent, launch-time reference incumbent,
source snapshot and hash, effective configuration, data/split/label identities, incumbent-matched training seed,
agent assignments, and artifact paths. Pin these before launch.

After completion, append observed triplet totals and study-visit counts, coverage
diagnostics, preparation/training/evaluation time, peak memory, selected epoch,
macro and per-target scores, paired bootstrap settings and interval, constraint
checks, and the decision with its rationale. Record the reference predictions and
their hash as well as the candidate's. Never overwrite an earlier attempt.

Execution states are `prepared`, `running`, `completed`, `failed`, and `cancelled`.
New evaluation decisions are `provisional_accept` (the
[shared adoption rule](protocol.md#shared-evaluation-and-adoption) holds), `reject`
(either adoption condition fails), and `invalid`. `adoption_bootstrap` in evaluation
evidence records that interval; the original 95% summary remains in `bootstrap`. Historical `inconclusive` records
remain unchanged. Evaluation evidence and launch review bindings record the adoption
rule explicitly. Keep execution status separate from the adoption decision. Incomplete evidence,
changed evaluation identities, or breached limits cannot produce acceptance.

The runtime also records intermediate `stage` values for development, gates, review,
comparison, and reconciliation. A reconciled candidate has a new attempt ID, an
explicit old/new parent record, and retained merge inputs/conflicts. Superseded
attempts remain addressable; they are not counted as trained failures.

`implementation_reference` records the parent ID, source tree digest, saved recipe,
input identity and prediction hash. Prepared developer reports require
`integration`: `mode` (`add` or `replace`), `retained_components`, `added_components`,
`removed_components`, `removed_idea_ids`, and `rationale`. Blocked reports use null.
Replacement includes pure ablations; it must name removed behavior. Accepted idea
IDs may be removed only if they are active. Required dependencies of both the
candidate and every retained accepted method must remain.

Approval requires a nonempty `integration_review`; other orchestrator responses
use null. Each review writes `review-<event-id>.json` in the attempt directory.
`implementation_review.binding` fixes its exact parent, candidate source/config,
integration, resulting `active_idea_ids`, external evidence and frozen evaluation
inputs. `launch_binding` records what authorized the launch. Evidence changes
invalidate approval without changing past reports. An accepted replacement updates
the active inventory without erasing any earlier experimental acceptance or result.

The read-only `results` view uses each attempt's recorded `evidence.macro_auc`
and `evidence.bootstrap` alongside its current `decision`. Its displayed delta is
the candidate score minus the reference score, not the bootstrap mean. An original
`eligible_decision` of `provisional_accept` with a later non-acceptance is shown as
withdrawn, preserving the original interval. See [metric progress and result
history](runtime.md#inspect-stop-and-resume) for commands and interpretation.

`comparison/per_label.json` is the authoritative label diagnostic artifact. Each
row records reference/candidate AUC, their exact delta, positive/negative counts
and paired bootstrap delta mean, percentile 95% interval and valid draw count.
Shared metadata records requested draws, seed, report-group unit, invalid-resample
rule and the exploratory, unadjusted interpretation. The artifact pins the original
comparison, predictions and data by SHA256 and retains diagnostic source copies.
The label bootstrap must reproduce the original macro bootstrap before saving
the diagnostics. It adds no acceptance decision or label-specific objective.

The results view and newly created agent contexts read this artifact directly,
including backfilled historical results. History indexes link its path for older
attempts beyond the compact recent-results window. Backfill appends missing
diagnostics without rewriting historical comparisons, evaluations, reports, state
or decisions; source hashes and elapsed diagnostic time distinguish that work.

Starting/current reference scores come from the saved metrics for the epoch named
in `selected_checkpoint.json`, without loading model weights. Retained experimental
gain follows accepted experiments through their recorded references; explicit
incumbent replacements in applied user decision records start a new gain reference.
Manual replacement and rollback records remain separate history. These views do
not rewrite decisions, reseal limits or migrate state.

## Completion report

The orchestrator receives identifiers and artifact paths, execution status, exact
scores and paired delta, confidence interval, resource and count checks, and any
failure excerpt. The full logs remain retrievable. A failure report must not invent
metrics for an unfinished model. A summary's claims must resolve to recorded evidence.

After each decision, the supervisor records the numerical outcome and the
orchestrator's interpretation or eligible reporter's factual summary. Research
interpretation and queue decisions remain orchestrator-owned. Keep the concise
durable finding and its limits inside `autoresearch/`. Keep old attempts
addressable even after changing the incumbent or removing a backlog entry.

Each reviewed attempt records `implementation_review`, including the approval
event and concerns. New completion events record `completion_reporting`: direct
orchestrator routing and its reasons, or reporter assignment, factual completion,
or escalation with its job/report reference. A reporter response contains no
decision fields. Failed or cancelled attempts retain their usage and event; resumed
handling uses the orchestrator. Historical jobs and approvals are not rewritten.

## Agent handoffs and usage

Each role invocation stores its profile and focused context in `assignment.json`,
the exact prompt in `stdin.txt`, and per-section byte counts in
`context-metrics.json`. `history-index.json` links original idea, experiment and
agent records without copying historical narratives into the prompt. For candidate
work, `evidence.json` records exact structured facts and available artifact paths,
sizes and hashes; absent evidence is explicit. Original source, full reports and
logs remain available for substantive review.

Orchestrator `ranked_ids: null` preserves the queue; an explicit list changes it.
The supervisor owns the numerical completion verdict, so reports do not repeat it
as an agent-controlled field. Reports contain concise `rationale`, `finding`,
`concerns`, and `evidence_refs`, with advisory length/count targets in the schema
descriptions. Full text and every concern/reference are retained even when those
targets are exceeded; narrative size cannot reject an otherwise valid report.
Required fields, types, decision enums and campaign gates remain strict. Concerns
and new findings are retained with the experiment and canonical findings.
Historical reports remain untouched; `usage.json` records actual report bytes.

`usage.json` is captured independently of report validation and process success.
It records provider/model/effort, role/event/experiment, token categories, any
provider-reported cost, prompt/report bytes, and measurement coverage. Claude
streams JSON events so interrupted calls can retain partial message usage; a final
result supersedes those partial counts. Model-level totals include auxiliary models.
Cached reads and writes are separate; thinking is included in output, not added
again. Missing usage or cost is never reported as a known zero.

The read-only `usage` command aggregates all four agent roles, including historical
failed calls whose separate usage file was never written. It reports complete,
partial and unavailable jobs, direct experiment attribution, repairs and measured
compute time. Research and proposal-ranking costs remain campaign overhead, while
the campaign cost-per-valid-experiment figure includes that overhead. Valid means
a decided `provisional_accept`, `reject` or `inconclusive` experiment. Unavailable
costs make recorded cost totals and cost per valid experiment incomplete.
Reporter outcomes and routing counts appear under `completion_reporting`, alongside
per-experiment records. An escalated completion includes both reporter and
orchestrator costs in its attribution and the campaign total.
