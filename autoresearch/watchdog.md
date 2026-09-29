# Failure-triggered monitoring

The optional watchdog checks the [topic loop](topic-rotation.md) and the campaigns it runs,
or one campaign run outside the loop, once a minute using ordinary Python. Healthy checks
make no model requests. A paused campaign or dead supervisor triggers recovery: a plain
resume for a rejected-report pause, otherwise one Codex repair session using the Codex
`watchdog` profile in [agents.yaml](agents.yaml), GPT-6 Luna at medium effort. A dead loop
process, or a live one stuck on the same error for 30 minutes, gets a plain relaunch, then the
same repair flow. Each campaign keeps its own
research provider; a Claude campaign remains Claude.

Enable monitoring only when operational repairs and resume are authorized. For the loop
(start it after `loop-start`):

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch_watch start
```

For a campaign started with `start CAMPAIGN` outside the loop:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch_watch start --campaign CAMPAIGN
```

The process runs detached. `status` and `stop` take the same target:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch_watch status
rtk proxy uv run --locked python -m scripts.tools.autoresearch_watch stop
```

Stopping the watcher leaves healthy campaigns and the loop running. If a repair is underway,
the watcher cancels it and leaves the campaign stopped for inspection; no later watcher
resumes it. A deliberate stop of a standalone campaign ends
its monitoring and prevents automatic resume.

## Following the loop

On every pass the loop watcher reads `loop.json` and watches the loop's current topic and,
from the moment the new topic has runtime state, its successor, so it follows rotations
without a restart and covers both live campaigns during a handover. A successor paused
before its first approval is left to the loop, which cancels it; a dead successor supervisor
is still recovered. A campaign that leaves the loop's view (retired, superseded) is no
longer watched, and an in-flight repair of it is cancelled. A campaign stopped deliberately,
by `stop CAMPAIGN` or by the loop's handover, is skipped until it runs again. A
`loop-stop` ends monitoring and cancels a repair underway; `loop-resume` starts the watcher
again once the loop runs, unless the watcher had been stopped by the user or never started. A campaign watcher refuses a campaign the loop runs and ends when the loop
takes one over; the loop watcher refuses to start while a campaign watcher runs on one of
its campaigns.

The loop cooperates with a repair: while a campaign's `stop.json` carries the watchdog's
token, the loop neither commits a handover on it nor resumes it on a cancelled rotation.

## Recovery and limits

A campaign waiting on `retry_after` for a provider limit or transient provider error
stays running with a live supervisor, so it is healthy: the watcher records the
wait time in its state and takes no action. The supervisor retries by itself; see
[automatic recovery](runtime.md#inspect-stop-and-resume).

A pause with `pause_kind: rejection` (the third consecutive rejected report for one
event or experiment) first gets one plain resume without a repair
agent. The watcher rechecks the same pause under the control lock, runs the
existing preflight and requests resume, launching the supervisor only when it is
not alive; it does not quiesce or wait for jobs to drain. Only if the same
normalized failure recurs does the repair flow below run. Plain resumes are limited
to one per normalized failure and three in total, recorded as `resumes` in the
campaign's `recovery.json`.

Otherwise, when the supervisor is paused, the watcher waits for outstanding jobs
to drain. For a dead supervisor it waits for owned live workers to exit. It records a compact
incident, requests a graceful supervisor stop with an ownership marker, and allows
one repair session of at most fifteen minutes. This gives runtime edits a clean
process restart and avoids interfering with training. User stop/resume commands
replace the marker and cancel recovery ownership.

A dead loop process (status `running`, no stop requested) is relaunched once per normalized
failure, the last line of `loop.log`, and three times in total (`loop_restarts`). Its
campaigns keep running under their own supervisors meanwhile. If the same failure recurs,
the repair flow below runs on the loop instead of a campaign: the loop is already stopped,
so nothing is quiesced; the repair is cancelled if the loop is resumed, relaunched or
stopped, and the loop is relaunched only if `loop.json` is unchanged apart from its status
fields.

A live loop whose `loop.json` shows the same normalized `error` on every check for 30 minutes
(`STUCK_SECONDS`) is treated as dead, with that error as its failure, while a relaunch or repair
budget remains for it. The watcher writes `restart.json`; the loop takes it at the start of its
next pass, before any work, and exits without touching its topics, so no step is interrupted
(see [restart safety](runtime.md#topic-loop)). The watcher waits for the exit, then relaunches or
repairs as for a dead loop, against the same `loop_restarts` and `loop_incidents` limits: a first
occurrence gets a plain relaunch, a recurrence after it a repair. With both budgets spent for
that error it only records `loop_recovery: stuck`. A pass that clears the error resets the timer,
and a user stop or resume while a restart is pending cancels it.

The repair agent follows its [brief](roles/watchdog.md), to which the watchdog appends
the incident paths, token and scope. It can fix operational autoresearch code, briefs
and documentation; it does not launch jobs or resume the campaign. Its allowlist is the
autoresearch runtime package, role briefs, the operational docs `runtime.md`,
`lifecycle.md`, `watchdog.md`, `status.md` and `scripts.md`, and the two CLI wrappers
(`scripts/tools/autoresearch.py`, `scripts/tools/autoresearch_watch.py`). Off-limits
(`PROTECTED` in `watchdog.py`): its own brief; the gate, evaluation, binding, accounting
and policy modules (`campaign/gates.py`, `campaign/rules.py`, `runtime/layout.py`,
`scoring/diagnostics.py`, `campaign/binding.py`, `campaign/completion.py`,
`scoring/accounting.py`, `campaign/definition.py`); the verdict and launch modules
(`supervisor/results.py`, `supervisor/experiments.py`); the developer scope check
(`runtime/isolation.py`); and the topic-rotation streak, packet and checks
(`topics/history.py`, `topics/packet.py`, `topics/checks.py`).
It uses the workspace-write sandbox, subscription authentication and no additional
agents. Existing repository rules apply, including no tests and preservation of
unrelated edits. The watcher compares actual before/after file hashes of the
checkout, excluding supervisor-projected campaign `backlog.yaml`, `ideas.yaml` and
`findings.md`; any other change outside scope, including unrelated edits made in the
checkout during the repair, blocks resume and leaves the campaign stopped. The watcher checks
ownership again, performs the existing preflight, and resumes only after a ready
report. Failure or uncertainty leaves a recorded blocker.

Quota/authentication failures, lost training, scientific choices, source/data
mismatches and changes to sealed inputs are blockers. Recovery cannot change the
reference, proposals, acceptance rule, resource limits, recorded identities,
predictions or frozen source, and it never changes a recorded verdict.

The same normalized failure cannot dispatch another repair. Three total repair
attempts exhaust automatic recovery, per campaign and separately for the loop, even across
watcher restarts. An exhausted campaign or loop is left for the user while the loop watcher
keeps watching the rest; a standalone campaign watcher ends. These bounds
limit repeated quota consumption; there is no model polling or unlimited retry loop.
Monitoring reports process health, not scientific correctness. A live but hung
supervisor or loop pass is not automatically diagnosed; worker timeouts remain the runner's job.

The watcher process keeps `state.json` (owner, status, watched and blocked campaigns, loop
health and budgets), `watch.log` and loop incidents under `outputs/autoresearch/.loop/watchdog/`
for the loop, or `outputs/autoresearch/<campaign>/watchdog/` for a standalone campaign. Each
campaign's budgets and status are in its `outputs/autoresearch/<campaign>/watchdog/recovery.json`,
beside its incidents, whichever watcher handled them. An incident directory holds the compact evidence,
prompt, report, process identity, stdout/stderr, receipt, token usage, and
`changes.json` recording the actual changed files and any outside its scope.
Repair usage is separate from the normal campaign `usage` command.

The command uses [Codex non-interactive execution](https://learn.chatgpt.com/docs/non-interactive-mode).
