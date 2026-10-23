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
model and effort, including the restricted completion reporter. Models are families, so
each job gets the newest one: Claude resolves its `opus`, `sonnet` and `haiku` aliases
itself, and Codex families resolve at job launch to the highest listed `gpt-<version>-<family>`
in `~/.codex/models_cache.json`, which every `codex exec` refreshes; the job's
`assignment.json` records the resolved slug. Both CLIs must use
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

## Topic loop

The [topic loop](topic-rotation.md) runs one campaign at a time and rotates to a new topic
after the campaign's `backlog.rotate_after` researcher runs without an adoption (six for
researcher-drafted topics), without waiting for the user. Start it on an
existing campaign, or on the main recipe with a starting reference run, in which case the
first topic comes from the [topic queue](topics/INDEX.md) or a researcher draft:

```bash
rtk proxy uv run --locked just autoresearch loop-start --campaign CAMPAIGN --provider claude
rtk proxy uv run --locked just autoresearch loop-start --reference-run outputs/training/20260928-150724-rebuild-state-v2-control --provider claude
rtk proxy uv run --locked just autoresearch loop-status
rtk proxy uv run --locked just autoresearch loop-stop
rtk proxy uv run --locked just autoresearch loop-resume
```

`loop-start` checks the subscription login and GPU like `start`, validates the starting
reference against the main recipe, records the provider and GPU for every topic and
launches the detached loop (`loop-supervise` runs it in the foreground). A given campaign
is started if new, resumed if stopped and otherwise taken over as it runs. `loop-status` reads saved state only: loop
state, the current topic with its researcher streak, the rotation phase and last failure,
every topic with origin, decided runs, adoptions and streak at rotation, the queue, the
seed pool and the `attention` flag. The per-campaign commands keep working on each topic,
and `status CAMPAIGN` shows its rotation mark.

`loop-stop` stops the loop and every topic it is running (the current one and, during a
rotation, the old and new ones) the way `stop` does, letting a launched training and its
comparison finish; `--abort` cancels training too. Loop agent jobs are cancelled. It ends
at `STOPPED`, with the rotation phase kept, and ends the loop [watchdog](watchdog.md).
`loop-resume` resumes the loop and the topics it stopped with the recorded provider, and the
rotation continues where it was; once the loop is running again it also restarts the watchdog if
the loop stop ended it (a watchdog the user stopped, or never started, stays off). Its output
reports the watchdog outcome under `watchdog`. Stopping a
topic directly with `stop CAMPAIGN` leaves the loop running; it rotates only a `RUNNING`
topic.

The loop never pauses. An unexpected error is shown as `Error` in `loop-status` and the
next pass, 30 seconds later, retries; a pass that succeeds clears it. If the same error persists
for 30 minutes, the watchdog [restarts the loop](watchdog.md#recovery-and-limits). Provider limits and transient errors of its agent
jobs wait and retry as in a campaign; other agent failures, rejected reports and drafts
that fail the checks count toward three consecutive failures, after which the attempt is
cancelled and the old topic claims again ([rotation](topic-rotation.md#rotation)). A
paused topic is recovered with `resume CAMPAIGN` or the [watchdog](watchdog.md) started for
the loop, which follows the current topic across rotations, covers both topics during a
handover and relaunches a dead or stuck loop process. State, logs, rotation packets,
drafts and agent jobs live under `outputs/autoresearch/.loop/`. Update loop code only
while the loop is stopped.

Restarting the loop process is safe between passes. Its state is `loop.json`, saved after each
transition; it acts on campaigns only through their `stop.json`, `resume.json` and
`rotation.json` files, and its agent jobs are separate workers that the next process adopts
(`harvest`) or records as lost, never replays. Each phase is re-derived from `loop.json` on the
next pass. One step is not safe to interrupt: `materialize` saves the `successor` phase before it
copies the staged campaign and initializes and launches its supervisor, so a process killed in
between leaves a successor without runtime state that the rotation waits on. A requested restart
(`restart.json`) is therefore taken only at the start of a pass, before any work; the loop exits
without touching its topics, and the watchdog relaunches it.

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
its CPU comparison finish, still subject to its activation check; a training that ends during the stop still starts its
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
its ×4 and ×1 deltas, the 80% interval of the ×4 change and verdict. **Cost** gives recorded agent usage by
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
numerical evidence and are marked. Training attempts awaiting a final decision, including blocked attempts,
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
on a performance plateau. The campaign continues until stopped or, under the
[topic loop](#topic-loop), until it rotates. Failures that the
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

The supervisor recovers four failure classes without pausing. The first nine
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
- **Failed activation checks.** When a run's
  [activation signature](lifecycle.md#activation-check) fails, the supervisor stops
  the training through its job's cancellation and records the attempt `invalid` in
  `evaluation.json` and `run-checks.json`, with the metric, observed maximum,
  comparison, threshold and epoch as its reason. The reason becomes the repair
  instruction and the idea follows the crash route above, sharing its two-repair
  limit; after two repairs the failure is an ordinary invalid completion for the
  orchestrator. Stopped runs awaiting repair wait in order, so a second stop never
  displaces the first. These runs are no evidence against the idea and never count
  toward [family closure](records.md#family-verdicts). A range campaign's run stopped for
  exceeding its training bound takes the same route.

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

## Importing, moving and removing ideas, and manual adoption

Startup imports the campaign's `seed-ideas*.yaml` files
([lifecycle](lifecycle.md#startup-and-proposals)). Later additions, queue changes and user
adoptions go through the campaign inbox, `outputs/autoresearch/<campaign-id>/inbox/`, without
stopping the campaign:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch import-ideas CAMPAIGN --ideas-file autoresearch/campaigns/CAMPAIGN/IDEAS.yaml
rtk proxy uv run --locked python -m scripts.tools.autoresearch import-ideas CAMPAIGN --ideas-file autoresearch/campaigns/CAMPAIGN/IDEAS.yaml --forced
rtk proxy uv run --locked python -m scripts.tools.autoresearch move-idea CAMPAIGN --idea IDEA_ID --before OTHER_IDEA_ID
rtk proxy uv run --locked python -m scripts.tools.autoresearch move-idea CAMPAIGN --idea IDEA_ID --after OTHER_IDEA_ID
rtk proxy uv run --locked python -m scripts.tools.autoresearch remove-idea CAMPAIGN --idea IDEA_ID --rationale "..."
rtk proxy uv run --locked python -m scripts.tools.autoresearch adopt CAMPAIGN --experiment EXPERIMENT_ID --rationale "..."
```

The command validates the request against the saved state and writes it to the inbox.
A live supervisor applies inbox requests in arrival order at the start of its next loop,
under its own lock, and defers them while an orchestrator job is in flight, because that
job ranks the queue as it was. With no live supervisor, the command applies them
immediately under the supervisor lock. A request that fails when applied moves to
`inbox/rejected/` with its error; status reports pending and rejected requests.

- **Ideas** use the idea-report format (`IdeaReport`). Import preserves an immutable proposal
  artifact and adds a proposal event: the orchestrator admits and ranks them, and
  research fills only the remaining places. IDs must be new to the registry and pending
  proposals.
- **Forced ideas** (`--forced`) skip admission and join the user-ordered seeds: they are
  queued in file order behind the reserved queue head and any pending forced ideas, ahead
  of every other idea, and research waits until they are completed or blocked. They
  cannot declare dependencies. This works in ranked campaigns too; `seed_order` only
  decides how startup seed files are imported.
- **Moving and removing** act only on an idea whose status is `queued`: once the developer
  has taken an idea (developing, prepared, launched) the request is refused, as is a move
  ahead of the reserved queue head.
  - **Move** places the idea directly before or after another queued idea. The moved idea and
    every idea ahead of it become user-ordered, like forced ideas: the orchestrator cannot
    rerank or park them, and research waits until they are completed or blocked.
  - **Remove** takes the idea out of the queue and marks it `retired` with the rationale; it
    stays in the registry and a decision record is saved under
    `outputs/autoresearch/<campaign-id>/decisions/`. A removed forced idea no longer holds
    research back.
- **Adoption** makes a valid completed experiment the incumbent by user decision. It
  waits until no training is running, so no launched candidate changes reference, then
  saves the state before the change and a decision record under
  `outputs/autoresearch/<campaign-id>/decisions/`, marks the experiment accepted (its
  original verdict stays visible) and adds a finding. A prepared candidate on the old
  incumbent is reconciled as after any adoption. Results list it as a manual reference
  change.

Calibration and sealed limits are retained.

## Initial sealing and fixed budget

Start checks the subscription login, Linux bubblewrap isolation, local GPU identity,
control artifacts and current recipe. A CPU-only replay then traverses the complete
training cohort of the campaign fold for two epochs, and deterministic validation
twice. It measures actual prepared triplets, including decoding and retry
shortfalls, without fitting a model, and refuses if the two replayed epochs differ. Counts come
from valid inputs in the batch mask, before model series dropout, excluding padding. The pass
takes about two minutes of CPU; research and candidate training wait for it. Training selections
must vary across epochs; validation selections must remain identical, and its dataset receives no
training seed. The training and validation cohorts are sealed in both budget modes
([training budget](campaign-setup.md#training-budget)).

**Sealed budget.** Per-epoch counts are fixed by the triplet budget, so the seal multiplies the
replayed per-epoch counts by the schedule's epochs. The training budget is the measured total
divided by the unchanged study-visit count. Tolerance is zero. Redistribution is allowed;
increasing or decreasing the aggregate average is invalid. Validation remains ×1, with its
measured aggregate count preserved, and inputs must have three slices. Epochs and optimizer steps
are fixed as well. The runtime limit is **115% of the reference run's recorded setup, training and
final validation-prediction time** on the campaign fold, rounded up.

**Range budget.** The wall-clock range is the only budget rule (user, 2026-09-28): the training
range is `min_fraction` to `max_fraction` of the reference's `training_seconds`, and the
per-experiment allowance is the sealed 115% ceiling plus `max_fraction − 1` times the reference's
training seconds. Epochs, batch size, gradient accumulation, optimizer steps, triplet and visit
counts and slices per input are free within the policy's config permissions; only the training and
validation cohorts, the epoch count the candidate declares and deterministic ×1 validation are
checked. There is no launch-time prediction: during training the supervisor stops a run whose
training has run longer than the upper bound (timed from `backbone_identity.json`, written just
before `fit`), records it `invalid` and returns it as a repair like a
[failed activation check](lifecycle.md#activation-check), and a completion whose recorded
`training_seconds` falls outside the range is `invalid`.

In both modes config preparation, the complete training process including sampling setup and
validation, and the CPU bootstrap comparison consume the per-experiment allowance. Agent
deliberation, queue waiting, source housekeeping and the one-time baseline replay are outside it.
Auxiliary policy preparation must run inside the measured training process; persistent
precomputation outside it is forbidden. Accepted candidates never raise limits. The training's
own accounting enforces the counts in the attempt's `run-limits.json`.

Each candidate trains at the current incumbent's recorded seed
([seed policy](protocol.md#training-seed-policy)); the supervisor binds it into review
and records it in the experiment and launch config. The bootstrap resampling seed is
separate and fixed. Code computes the verdict from the comparison record
under the [adoption policy](protocol.md#adoption-policy). The runtime
derives the macro 80% interval from the same resamples used for per-label
diagnostics, verifies their macro 95% summary against the frozen evaluator, and
stores both in `per_label.json`. Partial evidence, scope violations and budget
violations cannot produce acceptance.

## Boundaries and prerequisites

Only local training and local validation are authorized, or those of a campaign's
[training host](#training-host). No Kaggle submissions, uploads, publication, or other
remote jobs. Candidate configuration, baseline replay,
training and comparison processes have networking disabled. Cache the existing
pretrained weights before starting; training uses offline model loading. Research
agents may read public sources through their CLI tools.

This runtime requires Linux, bubblewrap/user namespaces, the repository's locked
Python environment, cached model weights, the saved fold data and acquisition
headers, and the selected authenticated CLI. Startup refuses missing isolation or
API-only authentication. Ordinary code checks GPU availability and holds a GPU lease
shared by local campaigns before launching. External processes can still compete for
the GPU; their interference cannot increase the sealed budget.

### Extra training data

A recipe with `extra_training` trains on external pseudo-labelled studies only: it is the
pretraining stage, and the fold's studies serve validation. The training ledger records no
batches, because the external studies lie outside the sealed cohort; the wall-clock range budget
covers their time. Joint training, which interleaved fold and external batches, was removed: see
[retired fastMRI-chain pieces](../docs/history/fastmri-chain-retired-pieces.md#joint-training-with-external-batches).

### Training host

A campaign with a `training_host` block in `campaign.yaml` (address, port, user, key, root) runs
its calibration, trainings and comparisons on that machine; a campaign without it runs them
locally. Agents, review, the queue, decisions and every record stay local.

- **Launch:** the job the supervisor starts is a local launcher
  (`knee_abnormality.autoresearch.runtime.remote`). It copies the campaign root, the candidate's
  source and the job's inputs to the same absolute paths on the host, starts the same command
  there and prints its log.
- **Same paths:** the host exposes the project's `data` and `outputs` at the local absolute
  paths, as links, so configs and recorded paths resolve on both machines.
- **Records:** every 20 seconds the launcher brings the job's output folder back, without
  checkpoints, so the budget, the activation check and the decision read local files.
  Checkpoints stay on the host.
- **Environment:** `<root>/environment` on the host is synchronised from the campaign root's
  `uv.lock` whenever its hash differs.
- **GPU:** the supervisor reserves the host's GPU, so a local campaign and a hosted one run side
  by side.
- **Stop:** the launcher refreshes a heartbeat on the host; a guard there ends the job when the
  heartbeat is five minutes old, so a cancelled, timed-out or crashed launcher ends its run.
- **Worker crash:** every job runs with Python's fault handler on. When a training exits because
  a data-loader worker died of a segmentation fault, and a fresh training still fits the run's
  time limit, the launcher discards the partial run and starts the same command again from
  scratch, once, and says so in the job's log.
- **Network:** the launcher needs the network locally. On the host the job runs without a
  sandbox when its container cannot provide one; the user accepts that per host, and no
  credentials, agents or records go there.

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
