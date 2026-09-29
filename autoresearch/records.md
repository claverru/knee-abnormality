# Research records

Paths in YAML and runtime records are repository-relative unless explicitly absolute.
Every state update carries a revision and event identifier so a resumed supervisor
can recognize work already completed. Structured agent report schemas live in
[campaign/reports.py](../src/knee_abnormality/autoresearch/campaign/reports.py); the runtime state and
immutable job artifacts preserve the mechanical transitions and exact evidence.
Runtime state also stores the sealed campaign policy from `campaign.yaml`, its backlog
settings (`refill_below`, `refill_to`, `seed_order`, `rotate_after`) and its `budget`; each
role assignment includes its permissions. A changed definition cannot silently
widen a running or resumed campaign's scope.

A chained topic's `campaign.yaml` carries its `chain` block and its sealed `limits.json`
repeats it under `chain`, with the runtime ceiling inherited from the predecessor
([chaining](topic-rotation.md#chaining)).

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
- For researcher proposals (`ResearchIdea`): `source_change`, whether the idea needs a Python
  source change, and `targeted_curve`, the curve behaviour in `run_curves` it targets. Seeds and
  imported ideas use the plain idea schema.

Idea statuses are `proposed`, `queued`, `developing`, `prepared`, `launched`,
`completed`, `blocked`, and `retired`. Their experiment references and outcomes stay
in the registry after the corresponding queue entry is removed. A materially revised
follow-up gets a new idea ID under the same mechanism. Historical links alone do
not make an idea queued or authorize duplicate work.

`backlog.yaml` contains a `queue` list of unique idea IDs ranked by the orchestrator,
highest potential first. It is the work queue, not the research memory; claiming,
launch removal and refill are described in the [lifecycle](lifecycle.md#claim-and-refill).
Launch removal never deletes the permanent idea record.

## Family verdicts

State `research.verdicts` holds one standing verdict per `mechanism_id` (`extend`,
`retry` or `close`, with reason, next step and a miss count), written by the researcher
for every family in its `new_results` and validated in code: a missing verdict, a
non-`close` verdict after two consecutive rejected follow-ups (only `reject` counts; an
adoption resets the count and `invalid` runs, failed activation checks included,
leave it unchanged), a proposal in a closed
family, or an open line left without a follow-up while slots allow is rejected and the
refill retried, up to the report-rejection limit. `research.reviewed` lists the results
already given verdicts. The researcher and proposal-admission contexts receive these as
`family_ledger`, one row per family with attempts, decided results and verdict.

## Run curves

Each researcher refill writes `curves.json` in its job directory: for the campaign reference and
every decided run, the per-epoch ×1 `val_macro_auc`, `train_loss_epoch`, per-label ×1 AUC (from
`epochs/metrics_epoch_*.json`), every non-constant epoch-level `metrics.csv` column (`other`;
step-level columns such as `grad_norm` stay in `metrics.csv`), `training_seconds`, the paths of
`metrics.csv` and the epoch metrics, and the run's configuration changes against the sealed recipe
as `[recipe, candidate]` pairs without logistical keys. Values have four decimals. The assignment's
`run_curves` repeats it compactly: every curve for the reference, the incumbent and the
last four `new_results`; the standard curves without `other` for the last eight other decided runs; only
configuration changes for older runs, whose curves stay in the file `full_detail` points to
([`reporting/curves.py`](../src/knee_abnormality/autoresearch/reporting/curves.py)). The topic
packet carries the incumbent's curves under `incumbent.curves`.

## Experiments

Use one immutable directory per attempt in
`outputs/autoresearch/<campaign-id>/experiments/<experiment-id>/`.
An attempt records its idea, implementation parent, launch-time reference incumbent,
source snapshot and hash, effective configuration, data/split/label identities, incumbent-matched training seed,
activation signature, agent assignments, and artifact paths. Pin these before launch.

`run_checks` (also `run-checks.json` in the attempt directory) holds each mid-run
check's outcome with its `check` name, `passed`, `reason`, the `metrics` path, `evaluated_at`
and `at_training_end`: for the [activation check](lifecycle.md#activation-check), the bound
signature, `observed_maximum` (null when never logged) and `logged_values`; for the budget check
of a range campaign, `training_seconds` and `max_training_seconds`. Every
`evaluation.json` and evaluation evidence repeats it under `run_checks`. A failed check
records `decision` and `eligible_decision` `invalid` with the stage `repair_requested`,
or an invalid completion after two repairs.

`run-limits.json` in the attempt directory holds the counts the training's accounting enforces,
derived at launch from the sealed limits and the candidate's epochs (`mode`, `input_slices`, null
for counts and steps a range budget leaves free).

After completion, append observed triplet totals and study-visit counts, coverage
diagnostics, preparation/training/evaluation time, peak memory, selected epoch,
macro and per-target scores, paired bootstrap settings and interval, constraint
checks, and the decision with its rationale. Record the reference predictions and
their hash as well as the candidate's. Never overwrite an earlier attempt.

Execution states are `prepared`, `running`, `completed`, `failed`, and `cancelled`.
Evaluation decisions are `provisional_accept` (the
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
`integration`: `retained_components`, `added_components`, `removed_components`,
`removed_idea_ids`, and `rationale`, and `activation`: `metric`, `comparison` (`>` or
`>=`), `threshold`, `epoch` and `rationale`; blocked reports use null for both. Their
rules are in the [lifecycle](lifecycle.md#development-and-gates).

Approval requires a nonempty `integration_review` and `activation_review`; other
orchestrator responses use null. Each review writes `review-<event-id>.json` in the
attempt directory. `implementation_review.binding` fixes its exact parent, candidate
source/config, integration, activation signature, resulting `active_idea_ids`, external evidence and frozen evaluation
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
as an agent-controlled field. Reports contain `rationale`, `finding`, `concerns` and
`evidence_refs`, with advisory length/count targets in the schema descriptions
([protocol](protocol.md#evidence-and-context)). Concerns and new findings are
retained with the experiment and canonical findings. Historical reports remain
untouched; `usage.json` records actual report bytes.

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
any recorded decision except `invalid`, so historical `inconclusive` records count.
Unavailable
costs make recorded cost totals and cost per valid experiment incomplete.
Reporter outcomes and routing counts appear under `completion_reporting`, alongside
per-experiment records. An escalated completion includes both reporter and
orchestrator costs in its attribution and the campaign total.

## Topic loop

The loop's records live in `outputs/autoresearch/.loop/`: `loop.json` (state, topic records
with origin and outcome, rotations, taken pool seeds, `attention`), `rotations/<id>/` (the
topic packet, drafts with `checks.json`, staged campaigns and the final `rotation.json`) and
`jobs/<id>/` (topic-draft and pick assignments, prompts, reports and usage, as for campaign
jobs). A campaign under rotation also has `rotation.json` in its own output directory, the
mark its supervisor obeys. Field meanings are in [topic rotation](topic-rotation.md#records).
Topic drafts use `TopicDraftReport` and picks `TopicPick` in
[campaign/reports.py](../src/knee_abnormality/autoresearch/campaign/reports.py); the seed
pool uses the same topic-seed schema plus `evidence`.
