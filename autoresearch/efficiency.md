# Agent usage and context efficiency

Keep substantive implementation review and scientific interpretation with the
recorded orchestrator model: Claude Opus 5.5 at medium for the Claude campaign (Opus 5 before
2026-09-23).
Repeated information and mechanical work are handled by compact packets and code.
A restricted completion reporter now replaces eligible routine completion calls:
Haiku 4.5 without an effort override for Claude, GPT-6 Luna at low effort for Codex.
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

## Changes this baseline motivated

Repeated context was the largest orchestration cost, so assignments now carry
role-specific cards, a retrievable history index and code-assembled evidence packets
instead of full queue narratives and the last twelve complete reports
([evidence and context](protocol.md#evidence-and-context)). Opus still inspects the
real candidate source. Hard prose limits caused five failed shortening attempts on an
otherwise valid losses report, which ended two characters over the rationale limit;
prose lengths are now advisory and full text is kept. Code records the fixed verdict,
empty refills create no review call, and usage is captured independently of report
acceptance ([records](records.md#agent-handoffs-and-usage)). Eligible reject
completions go to the
[restricted completion reporter](protocol.md#restricted-completion-reporting).

No new-run token savings or unchanged research quality were measured against this
baseline. The $14.93 completion-interpretation total was an opportunity to measure,
not promised savings: only a subset qualifies, and escalation adds another call.
Compare cost per valid completed experiment, usage coverage, prompt/report sizes,
repair frequency, unresolved issues and research outcomes on real runs, using
`usage.completion_reporting` (reported, escalated, pending and direct routes) and
`by_role_event.completion_reporter/completion`. Keep missed issues and poor
scientific decisions visible; shorter reports alone do not establish better
efficiency.

## Knee-geometry baseline and completion routing

Codex campaign, 2026-09-23/24: 105 agent jobs, all completed; Codex records tokens but no cost.

| Role | Model and effort | Jobs | New input | Cached input | Output |
|---|---|---:|---:|---:|---:|
| Developer | GPT-6 Sol, high | 32 | 2.69M | 47.52M | 0.37M |
| Orchestrator, reviews | GPT-6 Sol, medium | 32 | 2.23M | 12.02M | 0.10M |
| Orchestrator, completions | GPT-6 Sol, medium | 19 | 1.20M | 9.29M | 0.06M |
| Orchestrator, proposals | GPT-6 Sol, medium | 13 | 0.76M | 3.94M | 0.04M |
| Researcher | GPT-6 Astra, xhigh | 9 | 1.28M | 10.60M | 0.13M |
| **Total** | | **105** | **8.15M** | **83.37M** | **0.70M** |

- **Repairs:** 9 of 22 first attempts needed one, adding 9 developer jobs and 9 reviews,
  about 28% of developer and review tokens.
- **Completion routing:** the reporter almost never ran. In knee-geometry 16 of 18
  completions, and in near-misses 17 of 19, were rejects routed to the orchestrator only
  because the implementation review listed concerns; the orchestrator lists concerns in
  nearly every approval. On all 33, the orchestrator made no queue, ranking, parking or
  proposal decision; it only recorded the finding.
- **Change, 2026-09-25:** review concerns, other pending events and related outstanding
  ideas no longer block the reporter, and its own concerns no longer escalate; both are
  recorded with the finding. Only `status: escalate` or a failed eligibility recheck
  reaches the orchestrator ([protocol](protocol.md#restricted-completion-reporting)).
  Measure it on the next campaign's `usage.completion_reporting`.
- **Deferred:** lowering developer effort (the largest single cost) risks more repairs;
  fewer, larger researcher batches could cut the 13 proposal calls, but with forced seeds
  ranking starts only after the seeds finish.

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
