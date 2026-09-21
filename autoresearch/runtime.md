# Runtime operation

Start from the repository root, choosing one subscription provider:

```bash
rtk proxy uv run --locked just autoresearch-start codex
```

```bash
rtk proxy uv run --locked just autoresearch-start claude
```

These default to `triplet-sampling`. Select another configured campaign explicitly:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch validate losses
rtk proxy uv run --locked just autoresearch-start claude losses
```

The validation command checks configuration and saved control artifacts without
launching anything. See [campaign setup](campaign-setup.md) to add a topic through
its YAML policy and contract. The first successful start seals
the provider, role profiles, source, recipe, campaign policy, GPU, and resource limits. The launcher
returns immediately; closing the terminal or interactive agent session does not
stop the supervisor or its jobs. No interactive model selection is needed.
[agents.yaml](agents.yaml) determines the role assignments, including the restricted
completion reporter. Both CLIs must
use subscription authentication; API credentials and provider overrides are removed
from child environments. There is no API fallback.

Claude explicitly uses `CLAUDE_CONFIG_DIR=~/.claude` so both its credentials and
atomically updated CLI state live under a writable CLI directory, while the home
directory remains read-only. This follows its documented
[configuration-file locations](https://code.claude.com/docs/en/mcp-quickstart#find-your-configuration-on-disk).

The public entry point is
[scripts/tools/autoresearch.py](../scripts/tools/autoresearch.py). Its maintained
implementation lives in
[src/knee_abnormality/autoresearch/](../src/knee_abnormality/autoresearch/).
The CLI also accepts an explicit campaign and GPU:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch start triplet-sampling --provider codex --gpu 0
```

## Inspect, stop, and resume

```bash
rtk proxy uv run --locked just autoresearch-status
rtk proxy uv run --locked just autoresearch-results
rtk proxy uv run --locked just autoresearch-stop
rtk proxy uv run --locked just autoresearch-resume
```

Append the campaign ID to each command for another line, for example
`rtk proxy uv run --locked just autoresearch-status losses`. Commands act on one
campaign; starting or stopping losses does not start or stop sampling. Campaigns
share the GPU lease, so they cannot train simultaneously on the same device.

Status reads persisted state and process identity without calling a model. Stop
prevents new work, cancels agent jobs, and lets an already launched training and
its CPU comparison finish. A comparison cancelled before it started runs again
after resume; its decision is completed after resume. To terminate training too:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch stop triplet-sampling --abort
```

Status shows a compact summary of calibration passes, active jobs and elapsed time,
the ranked backlog, the next candidate, training progress, recent results, budget,
and recorded agent usage by role/event. It includes failed calls and explicitly
marks partial or unavailable usage; unparseable usage counts as unavailable and
never blocks result processing or status. The metric summary shows the original reference
AUC, current incumbent AUC, raw change since the start, retained experimental gain,
and the latest decided experiment's scores, delta and paired 95% interval.

For the complete experiment history, use either command:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch results triplet-sampling
rtk proxy uv run --locked just autoresearch-results triplet-sampling
```

Each decided attempt keeps its own launch-time reference, reference/candidate AUC,
candidate-minus-reference delta, paired 95% interval and current decision. Scores
use six decimals. Withdrawn acceptances retain their original numerical evidence
and are explicitly marked. Training attempts awaiting a final decision appear
separately, including blocked attempts; missing metrics appear as unavailable.

Every completed comparison also reports all twelve labels: reference/candidate
AUC, exact delta, positive/negative counts, paired report-group bootstrap 95%
interval for that delta, and valid/requested draw counts. These are unadjusted
exploratory intervals, not simultaneous coverage or per-label acceptance rules.
Macro AUC remains the only numerical acceptance criterion. The researcher and
orchestrator receive these diagnostics automatically with result evidence.

The maintained comparison wrapper runs the campaign's frozen evaluator, then
computes label diagnostics from saved predictions using the same group draws,
seed and invalid-resample rule. Both steps share the existing comparison job's
CPU time budget. Diagnostics are saved in `comparison/per_label.json`; the
original `comparison.json`, checkpoints and decisions remain unchanged.

To fill historical comparisons that predate label diagnostics:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch backfill-labels losses
```

This CPU-only operation verifies saved input/prediction hashes, reproduces the
original macro bootstrap, and writes only missing diagnostics. It takes no GPU,
training or model calls and can run while a campaign is active. Existing diagnostic
files are retained; attempts without a completed comparison are listed separately.
In-flight comparisons are deferred to their automatic reporting step so the
backfill cannot contend with a live evaluator for the same artifact.
Historical diagnostic time is recorded separately and never rewrites old run
budgets or decisions. New comparisons include diagnostic time in their fixed budget.

Retained gain follows the current incumbent's accepted reference chain back to
its starting reference or a manually selected replacement. Manual reference
changes are listed separately and excluded from that gain. Positive inconclusive
deltas and withdrawn acceptances do not count as retained improvements. Raw change
since the original campaign start can therefore differ from retained gain. These
are descriptive fold-0 score changes, not independent confirmation or a new
bootstrap interval for cumulative improvement.

Both metric views read saved state and selected-epoch metrics without loading
checkpoints, recomputing predictions, calling a model or launching training.
They preserve campaign state. Append `losses` instead of `triplet-sampling` to
inspect the loss campaign; an unstarted campaign reports no results.

For complete machine-readable usage details:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch usage
```

This command is read-only, needs no provider login or GPU, and includes per-job
model/effort, prompt/report bytes, direct per-experiment usage, completion-reporter
routing and escalation, repairs, compute
time, and campaign cost per valid completed experiment. Reported dollar values
are API equivalents, not subscription-quota percentages.

Calibration progress updates after each complete pass, not after each study. For a
live terminal display, with no model calls:

```bash
rtk proxy watch -n 5 'rtk proxy uv run --locked python -m scripts.tools.autoresearch status'
```

There is no overall experiment-count limit, total-runtime limit, or automatic stop
on a performance plateau. The campaign continues until stopped. Quota and unexpected
failures pause it; an exhausted research backlog leaves it idle without repeated
model calls. Each experiment still has its fixed compute ceiling; exceeding it
invalidates that attempt and does not itself stop the entire campaign.

Resume uses the recorded provider and profiles. It adopts live owned workers and
consumes their completion receipts. A lost training attempt becomes invalid; it is
never automatically launched again. This command resumes campaign orchestration,
not an interrupted model checkpoint. Quota, authentication, malformed reports, and
unexpected orchestration failures pause scheduling, retain artifacts, and require
fixing the reported cause followed by resume. If applying a finished training,
comparison, calibration or configuration result fails, the state is rolled back,
the job keeps its receipt, and the campaign pauses; resume reapplies the saved
result without rerunning the command. Waiting consumes no model turns.

Report prose lengths and concern/reference counts are advisory targets, so
exceeding them cannot exhaust structured-output retries or pause the campaign.
Full text is preserved. Invalid fields, types, decisions or campaign operations
still fail validation; authentication, quota and process failures still pause.

A proposal dependency rejection preserves the pending event and the existing
queue. On resume, the orchestrator receives the rejection and explicit dependency
eligibility, then returns a corrected admission/ranking. Unresolved proposals stay
outside the queue until their prerequisites have recorded provisional acceptance.
Completed training and prepared candidates are retained; this recovery does not
repeat training or change the acceptance rule. If runtime code or role guidance
needs updating, stop first and wait for `STOPPED`, apply the repair, then resume so
the new supervisor loads it.

State, supervisor logs, per-job prompts, structured reports, usage, process identities,
and receipts live under `outputs/autoresearch/<campaign-id>/`. The state file is
authoritative; YAML backlog/registry files and campaign findings are atomic projections
of serialized orchestrator decisions. Reopening the supervisor regenerates these
projections after an interrupted write. Do not edit them while a campaign runs.

The operational supervisor, worker wrapper and role briefs use the maintained
main-tree runtime. Update them only while the campaign is stopped; the next
supervisor launch uses those changes. Candidate and training source, campaign
contract, policy, profiles, data identities and sealed numerical limits remain frozen.
Each new agent job saves its exact brief/context/schema and evidence packet, so
operational prompt changes remain attributable without changing old artifacts.

## Initial sealing and fixed budget

When present, `autoresearch/campaigns/<campaign>/seed-ideas.yaml` supplies existing
ideas in the structured researcher-report format. Startup queues these for
orchestrator review before requesting new research; blocked, rejected and out-of-topic
work does not become executable just because it appears in the general backlog.
The triplet campaign imported the asymmetric coronal draw and per-plane width policy;
its prior work records their outcomes. The loss campaign proposes shared BCE plus
ranking, focal loss, tempered class weighting, and delayed ranking. Per-label losses
are excluded by its contract.

To import proposals into an existing campaign, stop it and wait until status is
`STOPPED`, then import and resume:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch import-ideas --ideas-file autoresearch/campaigns/triplet-sampling/seed-ideas.yaml
rtk proxy uv run --locked just autoresearch-resume
```

Import preserves an immutable proposal artifact and adds a review event under the
supervisor lock. It rejects duplicate IDs in the registry or pending events and
does not overwrite the live queue. The orchestrator admits and ranks proposals;
research fills only the remaining places. Calibration and sealed limits are retained.

Start checks the subscription login, Linux bubblewrap isolation, local GPU identity,
control artifacts, and current recipe. A CPU-only replay then traverses the complete
fold-0 training cohort for the existing ten-epoch schedule and deterministic validation
twice. It measures actual prepared triplets, including decoding/retry shortfalls,
without fitting another model. This one-time accounting pass can take substantial
CPU and disk time. Research and candidate training wait for it to finish.

The training budget is the measured total divided by the unchanged study-visit count.
Tolerance is zero. Redistribution is allowed; increasing or decreasing the aggregate
average is invalid. Validation remains x1, with its measured aggregate count preserved.
Counts come from valid three-slice inputs in the batch mask, before model series
dropout, excluding padding. Training selections must vary across epochs; validation
selections must remain identical, and its dataset receives no training seed.

The runtime limit is **115% of the initial historical control's recorded setup,
training, and final validation-prediction time**, rounded up: **1,785 seconds** for
the sampling campaign's original control and **1,797 seconds** for the losses
campaign's control B. Config preparation, the complete training process including
sampling setup and validation, and CPU bootstrap comparison consume that same fixed
per-experiment allowance. Agent deliberation, queue waiting, source housekeeping, and
the one-time baseline replay are outside it. Auxiliary policy preparation must run
inside the measured training process; persistent precomputation outside it is forbidden.
Epochs and optimizer steps are fixed as well. Accepted candidates never raise limits.

Each candidate inherits the current incumbent run's recorded training seed. The
supervisor reads `effective_config.json`, binds that seed into implementation review,
and records it in the experiment and launch config. A different preparation seed
is overridden before review; old approvals require renewed review. Reconciliation
uses the replacement incumbent's seed. Genuine checkpoint resumes keep their original
seed, and already launched historical attempts retain their original evaluation
policy. The bootstrap resampling seed is separate and fixed. The evaluator uses the saved paired report-group bootstrap;
the orchestrator records the resulting adoption decision and interpretation. Valid
macro gains >= 0.001 replace the incumbent only if the central 80% paired-bootstrap
lower bound is strictly positive; otherwise discard the candidate. The existing
95% and per-label intervals remain diagnostic. The runtime derives the macro 80%
interval from the same resamples used for per-label diagnostics, verifies their
macro 95% summary against the frozen evaluator, and stores both in `per_label.json`.
Status/results show the 80% adoption interval alongside the 95% reporting interval. See the
[adoption policy](protocol.md#adoption-policy).
Partial evidence, scope violations, and budget violations cannot produce acceptance.

## Keeping development ahead of training

Each idea gets an independent workspace and a frozen source copy. This avoids branch
switches in a live training checkout and works with the repository's uncommitted
source. The shared Python environment is reused, with `PYTHONPATH` explicitly pointing
to the candidate source. Experiment code is never silently imported from the main
checkout. Main-checkout experiment code is not automatically replaced by a winner.

After training creates its execution record, the reserved head is removed from the
queue. The developer takes the next ranked idea while that training runs; a researcher
refills concurrently below four entries, toward five. At most one next candidate is
developing or prepared. A failed launch leaves the head reserved for explicit handling.

If the incumbent changes while the next candidate is being developed, the runner
waits for its developer to finish. Before launch, it creates a new recorded attempt
and performs a three-way merge of the candidate's changes onto the new incumbent.
The old parent, old candidate, new parent, and conflicts remain recorded. Every
merge returns to the developer to verify the intended addition or replacement,
including conflict-free merges. The result receives fresh configuration checks
and orchestrator review. No worker edits a running training snapshot.

Approval pins the exact incumbent identity/source/predictions, candidate source
and configuration, integration plan, active-idea inventory, limits, headers and
external-evidence snapshot. It is checked when the review returns and immediately
before launch. Source drift under the same incumbent identity pauses; an accepted
new incumbent triggers reconciliation. Missing or outdated approvals trigger
renewed review. Outstanding training, comparison and decision work blocks launch
even when the GPU is idle. Completion cannot replace an incumbent different from
the run's pinned reference.

Previously prepared but unlaunched attempts without these integration records are
preserved and reassessed in a new attempt on resume. Historical completed reports
are not rewritten or treated as newly approved. An accepted campaign incumbent
without an integration inventory requires explicit recovery rather than guessing
which methods its source contains; external starting references have no
campaign-owned accepted idea IDs.

Role assignments and reviews include hashes of the runs listed in
`additional_evidence_runs`, plus other initialized campaigns' incumbents and
recorded decisions. A change detected before launch invalidates the old approval
and sends the candidate back to review. Results are evidence, not automatic
cross-campaign adoptions. Unlisted manual runs are not discovered; add relevant
paths to the evidence inventory while stopped. Unknown future results cannot be
anticipated, and separate campaigns retain their own decisions and budgets.
Completion evaluation, renewed review and reconciliation can leave the GPU idle.

## Boundaries and prerequisites

Only local training and local validation are authorized. No Kaggle submissions,
uploads, publication, or remote jobs. Candidate configuration, baseline replay,
training, and comparison processes have networking disabled. Cache the existing
pretrained weights before starting; training uses offline model loading. Research
agents may read public sources through their CLI tools.

This runtime requires Linux, bubblewrap/user namespaces, the repository's locked
Python environment, cached model weights, the saved fold data and acquisition headers,
and the selected authenticated CLI. Startup refuses missing isolation or API-only
authentication. Ordinary code checks GPU availability and holds a GPU lease shared
by local campaigns before launching. External processes can still compete for the
GPU; their interference cannot increase the sealed budget.

Researchers and orchestrators receive read-only repository views. Developers edit
isolated source; documentation, scripts, and harness code are read-only mounts.
A source allowlist, syntax check, configuration gate, and orchestrator review precede
training. Topic relevance, meaningful novelty, coverage/exploitation, and absence of
label leakage still require semantic review; they are not mathematical guarantees.
Only the orchestrator authors canonical research text, with the supervisor applying
its structured operations. All research documentation stays inside `autoresearch/`.


## Adopted reference source transition

The [learned-strength closure](checkpoint-closure.md) records the user-authorized
reference transition for all stopped campaigns. Current state pins the measured
run and cleaned development source separately. No trained metadata or predictions
are rewritten to disguise the cleanup. For a policy that forbids preprocessing
changes, the supervisor permits the recorded cleaned preprocessing identity only
against the exact external adopted reference and only after verifying its pinned
input-identity hash. Other identities remain strict; the exception stops applying
when that reference is replaced. The transition is part of review approval binding.
Original `reference_run`, baseline artifacts and sealed budgets retain startup
provenance; status and live state select the current incumbent.
