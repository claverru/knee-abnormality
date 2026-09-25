# Runtime operation

Start from the repository root with one subscription provider and a campaign ID;
there is no default campaign:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch validate CAMPAIGN
rtk proxy uv run --locked just autoresearch start CAMPAIGN --provider codex
rtk proxy uv run --locked just autoresearch start CAMPAIGN --provider claude
```

The validation command checks configuration and saved control artifacts without
launching anything. See [campaign setup](campaign-setup.md) to add a topic through
its YAML policy and contract. The first successful start seals the provider, role
profiles, source, recipe, campaign policy, GPU and resource limits. The launcher
returns immediately; closing the terminal or interactive agent session does not
stop the supervisor or its jobs. [agents.yaml](agents.yaml) determines each role's
model and effort, including the restricted completion reporter. Both CLIs must use
subscription authentication; API credentials and provider overrides are removed
from child environments. There is no API fallback.

Claude explicitly uses `CLAUDE_CONFIG_DIR=~/.claude` so both its credentials and
atomically updated CLI state live under a writable CLI directory, while the home
directory remains read-only. This follows its documented
[configuration-file locations](https://code.claude.com/docs/en/mcp-quickstart#find-your-configuration-on-disk).

The public entry point is
[scripts/tools/autoresearch.py](../scripts/tools/autoresearch.py); the maintained
implementation lives in
[src/knee_abnormality/autoresearch/](../src/knee_abnormality/autoresearch/).
The CLI also selects the GPU:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch start CAMPAIGN --provider codex --gpu 0
```

## Inspect, stop, and resume

```bash
rtk proxy uv run --locked just autoresearch status CAMPAIGN
rtk proxy uv run --locked just autoresearch results CAMPAIGN
rtk proxy uv run --locked just autoresearch stop CAMPAIGN
rtk proxy uv run --locked just autoresearch resume CAMPAIGN
```

Commands act on one campaign; stopping one does not stop another. Campaigns share
the GPU lease, so they cannot train simultaneously on the same device.

Status reads persisted state and process identity without calling a model. Stop
prevents new work, cancels agent jobs, and lets an already launched training and
its CPU comparison finish; a training that ends during the stop still starts its
comparison, so its verdict is recorded before `STOPPED`. A comparison cancelled
before it started runs again after resume, and its decision is completed then. To
terminate training too:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch stop CAMPAIGN --abort
```

Status prints a header (campaign, provider, state, supervisor, budget and fold), then
any warnings (pause reason, provider wait, pending stop, idle backlog) and five
sections. **Now** lists calibration progress and active jobs with elapsed time, and
training jobs show their idea, epoch and latest validation AUC. **Backlog** lists the
ranked queue with the campaign's refill thresholds and the next candidate. **Score**
shows the original reference AUC, current incumbent AUC, raw change since the start
and retained experimental gain. **Results** tabulates every decided experiment with
its delta, adoption 80% interval and verdict. **Cost** gives recorded agent usage by
role/event; it includes failed calls and marks partial or unavailable usage, and
unparseable usage counts as unavailable and never blocks result processing or status.
For a live display with no model calls:

```bash
rtk proxy watch -n 5 'rtk proxy uv run --locked python -m scripts.tools.autoresearch status CAMPAIGN'
```

Calibration progress updates after each complete pass, not after each study.

### Results

`autoresearch-results CAMPAIGN` (or the `results` CLI action) prints the complete
experiment history. Each decided attempt keeps its own launch-time reference,
reference/candidate AUC, candidate-minus-reference delta, paired 95% interval and
current decision, at six decimals. Withdrawn acceptances retain their original
numerical evidence and are marked. Control retrains after adoptions are listed with
both AUCs. Training attempts awaiting a final decision, including blocked attempts,
appear separately; missing metrics appear as unavailable. An unstarted campaign
reports no results.

Every completed comparison also reports all twelve labels: reference/candidate
AUC, exact delta, positive/negative counts, paired report-group bootstrap 95%
interval for that delta, and valid/requested draw counts. These are unadjusted
exploratory intervals, not simultaneous coverage or per-label acceptance rules.
Macro AUC remains the only numerical acceptance criterion. The researcher and
orchestrator receive these diagnostics automatically with result evidence.

The maintained comparison wrapper runs the campaign's frozen evaluator, then
computes label diagnostics from saved predictions using the same group draws,
seed and invalid-resample rule, within the comparison job's CPU time budget.
Diagnostics are saved in `comparison/per_label.json`; the original
`comparison.json`, checkpoints and decisions remain unchanged. To fill historical
comparisons that predate label diagnostics:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch backfill-labels CAMPAIGN
```

This CPU-only operation verifies saved input/prediction hashes, reproduces the
original macro bootstrap, and writes only missing diagnostics. It takes no GPU,
training or model calls and can run while a campaign is active. In-flight
comparisons are deferred to their automatic reporting step so the backfill cannot
contend with a live evaluator. Historical diagnostic time is recorded separately
and never rewrites old run budgets or decisions.

Retained gain follows the current incumbent's accepted reference chain back to
its starting reference or a manually selected replacement. Manual reference
changes are listed separately and excluded from that gain; positive rejected deltas
and withdrawn acceptances are not retained improvements. Raw change since the
campaign start can therefore differ from retained gain. Both are descriptive score
changes on the campaign fold, not independent confirmation or a bootstrap interval
for cumulative improvement. These views read saved state and selected-epoch metrics
without loading checkpoints, recomputing predictions, calling a model or launching
training.

### Usage

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch usage CAMPAIGN
```

This read-only command needs no provider login or GPU. It prints JSON with per-job
model/effort, prompt/report bytes, direct per-experiment usage, completion-reporter
routing and escalation, repairs, compute time and campaign cost per valid completed
experiment. Dollar values are API equivalents, not subscription-quota percentages.

### Running, pausing and resuming

There is no overall experiment-count limit, total-runtime limit, or automatic stop
on a performance plateau. The campaign continues until stopped. Failures that the
supervisor cannot recover automatically pause it; an exhausted research backlog
leaves it idle without repeated model calls. Each experiment still has its fixed
compute ceiling; exceeding it invalidates that attempt without stopping the campaign.

Resume uses the recorded provider and profiles. It adopts live owned workers and
consumes their completion receipts. A lost training attempt becomes invalid; it is
never automatically launched again. Resume continues campaign orchestration, not an
interrupted model checkpoint. Authentication failures, unclassified agent failures,
exhausted retries, a third consecutive rejected report and unexpected orchestration
failures pause scheduling, retain artifacts, and require fixing the reported cause
followed by resume. If applying a finished training, comparison, calibration or
configuration result fails, the state is rolled back, the job keeps its receipt, and
the campaign pauses; resume reapplies the saved result without rerunning the
command. Waiting consumes no model turns.

The supervisor recovers three failure classes without pausing. The first nine
campaign stops came from five agent reports that broke a runtime rule and four
provider errors, none needing a human decision; that is why these classes recover
automatically.

- **Rejected reports.** When applying an orchestrator or developer report raises
  (schema validation or a runtime rule such as the queue ranking, proposal
  dependencies or integration-plan checks), the state is rolled back, the error is
  stored as `previous_rejection` on the event or experiment, and the same role is
  assigned again with that text; the pending event and queue stay intact. The
  developer keeps its workspace. The consecutive `rejections` count resets when a
  report for that event or experiment is applied; the third consecutive rejection
  pauses with `pause_kind: rejection`. A rejected completion-reporter report
  escalates its event to the orchestrator; a rejected researcher report pauses.
- **Provider errors.** A failed agent job is classified from its provider output:
  Claude's final stream-json `result` (`api_error_status` and message) or Codex's
  last `error`/`turn.failed` event, falling back to the stderr tail. Overload, 5xx
  and network errors retry the same role for the same event or experiment after
  60 s, 300 s and 900 s, and the fourth consecutive failure pauses; the count resets
  when that work item's next call returns. A session or usage limit (HTTP 429) does
  not pause: it stores `retry_after` at the parsed reset time plus one minute, in the
  message's timezone or else local time, or 60 minutes ahead if unparseable, and
  retries afterwards without a cap. Authentication failures, runtime timeouts, lost
  workers and every other failure pause. During a wait no agent job is dispatched;
  training, comparison, configuration gates, claims and launches continue. A failed
  completion reporter still escalates its event to the orchestrator, which also waits.
- **Crashes before training.** A launched training that exits nonzero before
  completing any epoch (no `train` rows in its triplet accounting) records no
  completion. The experiment becomes `repair_requested` with the exit code and
  stderr tail as its repair instruction. Once no candidate is reserved and the queue
  is below its target, its idea returns to the queue head and the developer repairs
  it; the result passes the configuration gate and review as usual. This repair is
  granted even after one review repair, so an idea has at most two repairs. A crash
  after two repairs or after an epoch, a timeout, a lost worker and a cancellation
  are invalid.

Resume clears a pending provider wait so agents are retried at once; rejection and
transient-retry counts persist until the next success, so a resume after either
pause grants one more attempt. If runtime code or role guidance needs updating, stop
first and wait for `STOPPED`, apply the repair, then resume so the new supervisor
loads it. The [watchdog](watchdog.md) can do this automatically.

State, supervisor logs, per-job prompts, structured reports, usage, process
identities and receipts live under `outputs/autoresearch/<campaign-id>/`. The state
file is authoritative; YAML backlog/registry files and campaign findings are atomic
projections of serialized orchestrator decisions, regenerated when the supervisor
reopens after an interrupted write. Do not edit them while a campaign runs.

The operational supervisor, worker wrapper and role briefs use the maintained
main-tree runtime. Update them only while the campaign is stopped; the next
supervisor launch uses those changes. Candidate and training source, campaign
contract, policy, profiles, data identities and sealed numerical limits remain
frozen. Each new agent job saves its exact brief, context, schema and evidence
packet, so operational prompt changes remain attributable without changing old
artifacts.

## Importing ideas

Startup imports the campaign's `seed-ideas*.yaml` files
([lifecycle](lifecycle.md#startup-and-proposals)). To add proposals to an existing
campaign, stop it and wait until status is `STOPPED`, then import and resume:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch import-ideas CAMPAIGN --ideas-file autoresearch/campaigns/CAMPAIGN/IDEAS.yaml
rtk proxy uv run --locked just autoresearch resume CAMPAIGN
```

The file uses the researcher-report format. Import preserves an immutable proposal
artifact and adds a review event under the supervisor lock. It rejects duplicate IDs
in the registry or pending events and does not overwrite the live queue. The
orchestrator admits and ranks proposals; research fills only the remaining places.
Calibration and sealed limits are retained.

## Initial sealing and fixed budget

Start checks the subscription login, Linux bubblewrap isolation, local GPU identity,
control artifacts and current recipe. A CPU-only replay then traverses the complete
training cohort of the campaign fold for two epochs, and deterministic validation
twice. It measures actual prepared triplets, including decoding and retry
shortfalls, without fitting a model. Per-epoch counts are fixed by the triplet
budget, so the seal multiplies the replayed per-epoch count by the schedule's epochs,
and refuses if the two replayed epochs differ. The pass takes about two minutes of
CPU; research and candidate training wait for it.

The training budget is the measured total divided by the unchanged study-visit count.
Tolerance is zero. Redistribution is allowed; increasing or decreasing the aggregate
average is invalid. Validation remains ×1, with its measured aggregate count
preserved. Counts come from valid three-slice inputs in the batch mask, before model
series dropout, excluding padding. Training selections must vary across epochs;
validation selections must remain identical, and its dataset receives no training
seed.

The runtime limit is **115% of the reference run's recorded setup, training and final
validation-prediction time** on the campaign fold, rounded up. Config preparation,
the complete training process including sampling setup and validation, and the CPU
bootstrap comparison consume that same per-experiment allowance. Agent deliberation,
queue waiting, source housekeeping and the one-time baseline replay are outside it.
Auxiliary policy preparation must run inside the measured training process;
persistent precomputation outside it is forbidden. Epochs and optimizer steps are
fixed as well. Accepted candidates never raise limits.

Each candidate trains at the current incumbent's recorded seed
([seed policy](protocol.md#training-seed-policy)); the supervisor binds it into review
and records it in the experiment and launch config. The bootstrap resampling seed is
separate and fixed. Code computes the verdict from the saved paired report-group
bootstrap under the [adoption policy](protocol.md#adoption-policy). The runtime
derives the macro 80% interval from the same resamples used for per-label
diagnostics, verifies their macro 95% summary against the frozen evaluator, and
stores both in `per_label.json`. Partial evidence, scope violations and budget
violations cannot produce acceptance.

## Boundaries and prerequisites

Only local training and local validation are authorized. No Kaggle submissions,
uploads, publication, or remote jobs. Candidate configuration, baseline replay,
training and comparison processes have networking disabled. Cache the existing
pretrained weights before starting; training uses offline model loading. Research
agents may read public sources through their CLI tools.

This runtime requires Linux, bubblewrap/user namespaces, the repository's locked
Python environment, cached model weights, the saved fold data and acquisition
headers, and the selected authenticated CLI. Startup refuses missing isolation or
API-only authentication. Ordinary code checks GPU availability and holds a GPU lease
shared by local campaigns before launching. External processes can still compete for
the GPU; their interference cannot increase the sealed budget.

Researchers and orchestrators receive read-only repository views. Developers edit
isolated source; documentation, scripts and harness code are read-only mounts. A
source allowlist, syntax check, batch-key check, configuration gate and orchestrator
review precede training ([lifecycle](lifecycle.md#development-and-gates)). The
batch-key check parses the candidate's `modeling/` code for `batch["…"]` and
`batch.get("…")` string keys and fails when one is missing from `MODEL_BATCH_KEYS` in
`inference/engine.py`, which the final ×4 prediction moves to the GPU; that crash cost
one full training and several repairs before the check existed. Topic relevance,
meaningful novelty, coverage/exploitation and absence of label leakage still require
semantic review; they are not mathematical guarantees. Only the orchestrator authors
canonical research text, with the supervisor applying its structured operations. All
research documentation stays inside `autoresearch/`.

A reference trained on source that differs from the recipe only by removed, inactive
code is declared per campaign ([campaign setup](campaign-setup.md#files-and-startup)).
The adopted learned-strength reference and the transition of the stopped campaigns
to it are recorded in the
[anatomical-attention result](../docs/validation/results/2026-09-21-anatomical-attention-result.md#adoption-and-campaign-transition).
