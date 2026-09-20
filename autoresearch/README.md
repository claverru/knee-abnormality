# Autoresearch

Topic-scoped, autonomous experimentation. A user starts one campaign; a detached
supervisor then coordinates research, implementation, training, and evaluation
within its fixed contract. Campaigns cover triplet sampling and shared training
losses. The losses campaign uses one objective or shared combination across all
labels; per-label loss selection is excluded.

The detached runner is implemented. Adding campaign files does not start training.
Choose one provider and campaign when starting; the runner selects each role's model
and effort automatically, independently of the model in an interactive chat.

```bash
rtk proxy uv run --locked just autoresearch-start codex
```

Or use `autoresearch-start claude`. These default to `triplet-sampling`; start the
loss line with `autoresearch-start claude losses` or `autoresearch-start codex losses`.
Inspect progress with `autoresearch-status` and every experiment's scores,
paired interval and decision with `autoresearch-results`. Manual reference
changes and withdrawn acceptances are recorded separately from retained gains.
Stop with `autoresearch-stop`, and continue a stopped or paused campaign with
`autoresearch-resume`. See [runtime operation](runtime.md) before the first launch.

**Only local training and validation are authorized. No submissions, uploads,
publication, or remote evaluation jobs.**

Read the [index](INDEX.md) for the protocol, agent briefs, record format, and
campaigns. Add another topic through [campaign setup](campaign-setup.md), without
runner edits. Model and effort assignments live in [agents.yaml](agents.yaml).
Use the Codex Pro or Claude Max CLI subscription for the selected vertical;
do not silently fall back to paid API usage when quota is exhausted.

All agent-written documentation, proposals, findings, and work tracking stay inside
`autoresearch/`. The orchestrator owns research and backlog decisions. Researchers
and developers return structured reports; the orchestrator consolidates them.
Code can record eligible routine completions through a restricted Haiku/Luna
reporter, escalating uncertainty to the orchestrator. Existing project docs
may be read and linked but not edited by campaign agents. Developer changes to
experiment code remain permitted within scope.

Tracked files hold instructions, ideas, and concise durable findings. Runtime
state, source snapshots, logs, predictions, and checkpoints belong under the
already-ignored `outputs/autoresearch/<campaign-id>/`, with artifact references
in experiment records. Keep clinical data and study-level predictions out of git.

Implementation and verification status live in [status.md](status.md).
