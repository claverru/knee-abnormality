# Agent usage and context efficiency

Keep substantive implementation review and scientific interpretation with the
recorded orchestrator model: Claude Opus 5 at medium for the Claude campaign.
Repeated information and mechanical work are handled by compact packets and code.
A restricted completion reporter now replaces eligible routine completion calls:
Haiku 4.5 without an effort override for Claude, GPT-5.6 Luna at low effort for Codex.
Implementation review, scientific interpretation and research priorities remain
with the orchestrator.

## September 19–20 campaign baseline

The campaign started at 19:10:07 Madrid time on September 19 and hit Claude's
session limit at 00:02:48 on September 20: 4h 52m 41s, including initial CPU
calibration. The final running training finished at 00:09:29. Evidence is retained
under `outputs/autoresearch/triplet-sampling/jobs/`.

| Role/event | Jobs with usage | Recorded API cost equivalent |
|---|---:|---:|
| Orchestrator implementation reviews | 12 | $22.36 |
| Orchestrator completion interpretation | 8 | $14.93 |
| Orchestrator proposals/ranking | 5 | $5.86 |
| Developer | 12 | $23.84 |
| Researcher | 5 | $39.56 |

Recorded totals: 5,507,901 new input tokens including cache writes; 116,018,259
cached input tokens; 1,177,399 output tokens including 593,493 thinking tokens;
1,014 logged turns; $106.5476294 equivalent. Auxiliary Haiku usage is included.
The quota-failed researcher is included. An earlier researcher cancelled after
6m 25s has no usage, so these are incomplete totals across 43 launched agent jobs.
The eight valid decided experiments imply $13.32 recorded cost per valid
experiment, including research and orchestration overhead. This excludes the
unknown cancelled-call cost and includes work prepared ahead of training.

The 25 orchestrator reports contain 67,145 words in their rationale and finding
fields. Historical findings account for 6.83 MB of 10.01 MB of prompts among
the 42 metered jobs. This byte attribution is not a token or quota attribution.
Of the orchestrator's $43.15 equivalent, cache writes account for $25.80, cache
reads $9.30, and output $8.04. Repeated context is therefore the primary observed
orchestration cost, with new-input cost particularly prominent.

## Implemented changes

- Role/event-specific cards and a retrievable history index replace full queue
  narratives and the last twelve complete reports.
- Code assembles candidate evidence packets with exact metrics, accounting,
  configuration differences, timings and original artifact references.
- Bounded reports retain new findings, unresolved concerns and evidence paths;
  unchanged rankings use `null`. Opus still inspects the real candidate source.
- Code applies the fixed numerical verdict after orchestrator interpretation.
  Scheduling stays in the supervisor. Empty research refills create no review call.
- Usage capture is independent of report acceptance. Claude streams events for
  partial interrupted-call accounting. Unknown usage and cost remain explicit.
- `status` and the read-only `usage` command expose role/event totals, per-job
  model/effort, prompt/report sizes, direct experiment costs, repairs, and compute
  time. Historical raw provider records are included without rewriting them.
- Code routes only clean rejected/inconclusive completions without outstanding
  related work to the restricted reporter. It receives compact verified facts
  and original artifacts; it can summarize or escalate but has no decision fields.
  Eligibility is rechecked before applying its report. See the
  [exact routing rules](protocol.md#restricted-completion-reporting).
- The supervisor records routine results with the incumbent and queue unchanged.
  Escalation retains the reporter job and sends the same event to the orchestrator.
  Both costs are included in experiment and campaign usage. There is at most one
  reporter attempt per completion, including across stop/resume.

The maintained runner and role briefs take effect on the next supervisor launch.
Frozen experiment source and sealed acceptance limits are unchanged. No campaign
was resumed and no model or training was launched to make these changes. Static
lint, catalog consistency, CLI help and read-only status/usage inspection passed;
repository policy prohibits writing or running tests.

No new-run token savings or unchanged research quality have been measured yet.
The historical $14.93 completion-interpretation total is an opportunity to measure,
not promised savings: only a subset will qualify, and escalation adds another call.
At the next user-launched campaign, compare cost per valid completed experiment,
usage coverage, prompt/report sizes, repair frequency, unresolved issues, and
research outcomes against this baseline. Keep missing issues and poor scientific
decisions visible; shorter reports alone do not establish better efficiency.

`usage.completion_reporting` counts reported, escalated, pending and direct-to-
orchestrator routes. `by_role_event.completion_reporter/completion` accounts for
reporter calls. Compare total completion cost including escalations, the proportion
of eligible events, report repairs, and missed issues on subsequent real runs.

## Evidence informing the design

The [protocol](protocol.md#evidence-and-context) already cited
[SoL-Pi](https://arxiv.org/html/2609.20519v1). Its low-cost reducer extracts verified
log evidence while the main agent retains diagnosis and decisions. In Table 4,
Opus with ObservationPack alone costs $1,176 versus Pi's $1,741 and scores 47.047
versus 44.756; the full stack costs $1,158 but scores 42.224. These are benchmark
point estimates, not predictions of campaign quality or subscription quota.

[Karpathy's autoresearch instructions](https://github.com/karpathy/autoresearch/blob/master/program.md)
keep full training logs outside context and record compact metrics and outcomes.
[Anthropic's context-engineering guidance](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
supports structured memory and retrieval while warning against losing critical
details during compaction. These motivate compact handoffs with original evidence
available, without replacing the strong reviewer.
