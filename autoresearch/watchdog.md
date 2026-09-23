# Failure-triggered monitoring

The optional watchdog checks an existing campaign once a minute using ordinary
Python. Healthy checks make no model requests. A paused campaign or dead supervisor
triggers recovery: a plain resume for a rejected-report pause, otherwise one Codex
repair session using the Codex `watchdog` profile in [agents.yaml](agents.yaml),
GPT-6 Luna at medium effort.
The campaign keeps its own research provider; a Claude campaign remains Claude.

Enable monitoring only when operational repairs and resume are authorized:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch_watch start medial-lateral-transfer
```

The process runs detached. Inspect it or stop monitoring:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch_watch status medial-lateral-transfer
rtk proxy uv run --locked python -m scripts.tools.autoresearch_watch stop medial-lateral-transfer
```

Stopping the watcher leaves a healthy campaign running. If a repair is underway,
the watcher cancels it and leaves the campaign stopped for inspection. A deliberate
campaign stop also ends monitoring and prevents automatic resume.

## Recovery and limits

A campaign waiting on `retry_after` for a provider limit or transient provider error
stays running with a live supervisor, so it is healthy: the watcher records the
wait time in its state and takes no action. The supervisor retries by itself; see
[automatic recovery](runtime.md#inspect-stop-and-resume).

A pause whose reason starts with `Report rejected` (the third consecutive rejected
report for one event or experiment) first gets one plain resume without a repair
agent. The watcher rechecks the same pause under the control lock, runs the
existing preflight and requests resume, launching the supervisor only when it is
not alive; it does not quiesce or wait for jobs to drain. Only if the same
normalized failure recurs does the repair flow below run. Plain resumes are limited
to one per normalized failure and three in total, recorded as `resumes` in its state.

Otherwise, when the supervisor is paused, the watcher waits for outstanding jobs
to drain. For a dead supervisor it waits for owned live workers to exit. It records a compact
incident, requests a graceful supervisor stop with an ownership marker, and allows
one repair session of at most fifteen minutes. This gives runtime edits a clean
process restart and avoids interfering with training. User stop/resume commands
replace the marker and cancel recovery ownership.

The repair agent can fix operational autoresearch code, briefs and documentation;
it does not launch jobs or resume the campaign. Its allowlist is the autoresearch
runtime package, role briefs, and the listed operational docs and the two CLI
wrappers (`runtime.md`, `watchdog.md`, `status.md`, `scripts.md`,
`scripts/tools/autoresearch.py`, `scripts/tools/autoresearch_watch.py`); the gate,
evaluation, binding, accounting and policy modules (`gates.py`, `diagnostics.py`,
`integration.py`, `completion.py`, `accounting.py`, `campaigns.py`) are off-limits.
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
predictions or frozen source. It does not turn an inconclusive result into a failure.

The same normalized failure cannot dispatch another repair. Three total repair
attempts exhaust automatic recovery, even across watcher restarts. These bounds
limit repeated quota consumption; there is no model polling or unlimited retry loop.
Monitoring reports process health, not scientific correctness. A live but hung
supervisor is not automatically diagnosed; worker timeouts remain the runner's job.

Artifacts are under `outputs/autoresearch/<campaign>/watchdog/`: `state.json`,
`watch.log`, and one directory per incident containing the compact evidence,
prompt, report, process identity, stdout/stderr, receipt, token usage, and
`changes.json` recording the actual changed files and any outside its scope.
Repair usage is separate from the normal campaign `usage` command.

The command uses [Codex non-interactive execution](https://learn.chatgpt.com/docs/non-interactive-mode).
