# Campaign protocol

## Authority and scope

Creating a campaign is not starting it. The user launches the eventual supervisor
once; that active campaign authorizes detached experiments within its sealed
contract without approval for each run. Ordinary training outside it remains
user-launched. Stop requests and quota exhaustion are handled by the supervisor.

Each campaign researches one topic indefinitely. Creative mechanisms and combined
changes within that topic are welcome; a one-variable experiment is not required.
Supporting plumbing is allowed when necessary for the mechanism. The developer
does not modify the supervisor, evaluator, acceptance policy, or campaign limits.
Changing the topic or limits requires a new user decision.

Each campaign declares source/configuration permissions and topic-specific role
guidance in `campaign.yaml`; the shared runner validates and seals that policy.
New topics using this protocol require campaign files, not runner branches.
See [campaign setup](campaign-setup.md). The numerical acceptance and resource
gates remain shared code and cannot be disabled by a topic policy.

No submissions or publication are authorized. Experiments are limited to local
training and local validation; never submit to Kaggle, upload models or predictions,
publish datasets or notebooks, or launch remote evaluation jobs. Public research
and documentation may be read.

The numeric resource limits, baseline identity, evaluation rule, and data identity
must be sealed before the first candidate. An adopted candidate never raises the
original budget. Missing limits or failed constraints prevent a launch.

The orchestrator owns research interpretation, proposals, idea admission and
backlog ranking. All canonical documentation and work tracking stay inside
`autoresearch/`. Researchers and developers return structured reports to it.
The supervisor may also record the restricted completion reporter's factual
summary under the code-enforced eligibility rules below. That reporter has no
decision authority and never edits canonical files directly.
Read and link project documentation elsewhere
without changing it. This user-directed boundary replaces the general docs/ongoing
and general-backlog writing workflow for campaign work. Scoped experiment-code
changes and generated runtime artifacts are separate from this documentation boundary.

## Event-driven operation

The supervisor is ordinary code and outlives agent sessions. It owns process
launching, GPU availability checks, locks, completion detection, numeric evaluation,
and durable state transitions. Only one training run, one developer, and one
researcher refill may be active per campaign. Allow one candidate being developed
or prepared ahead of the running experiment. Training uses an immutable source
snapshot, so preparing the next candidate cannot alter a live run.

1. At startup, import the campaign's existing seed proposals when present, before
   asking for fresh research. Preserve their original links, check prior outcomes
   and blockers, and record campaign ownership. The orchestrator ranks eligible ideas
   and assigns the first to the developer. Claiming the head does not remove it.
2. The developer returns a source snapshot and implementation evidence. The
   orchestrator reviews it and requests detached training through the supervisor.
3. The supervisor checks limits and GPU availability. On confirmed successful
   launch, record the run and remove exactly that idea from the queue. A failed
   launch leaves it in place. Persist the launch intent and process identity so
   crash recovery cannot launch the same attempt twice or lose its queue entry.
4. Immediately assign the highest-ranked remaining idea to the developer. If the queue is below its
   refill threshold, also dispatch the researcher. Development and research run
   alongside the detached training.
5. If development finishes while the GPU is occupied, retain that one prepared
   candidate and exit the developer. Do not consume another idea yet.
6. Training completion triggers numeric evaluation and either restricted factual
   reporting or orchestrator interpretation under the rules below.
   The supervisor records the fixed verdict directly after the report,
   launches the prepared head when eligible, and repeats from step 3.

Record a candidate's implementation parent separately from the incumbent used for
comparison, which is pinned at launch. Preparing during another run may leave it
based on an older parent. Before launch, reconcile its changes onto a newly accepted
incumbent in a new recorded attempt. Preserve all three sources; clean merges need
renewed gates/review, and conflicts return to the developer. Never modify the running
source or silently claim the old candidate already contained later wins.

Agents exit when their assignment is complete. They never poll training, sleep
through a run, or repeatedly invoke tools to discover whether another agent is done.
Failures produce one event with evidence, not an unlimited retry loop. Recovery
must identify a concrete cause; quota waits consume no model turns and preserve state.

## Backlog and ownership

The backlog is ranked by the orchestrator, highest potential first. The developer
always takes the first entry at assignment time; it does not choose its own idea.
Rerank unclaimed entries when research proposals arrive or experiment results change
the evidence. The claimed head stays reserved through development and preparation;
new ideas cannot displace work already underway.

Rank by plausible improvement potential, relevance to a weakness in the campaign topic, novelty
relative to prior work, learning value, and implementation/compute cost. Prefer
mechanisms with meaningful upside over easy scalar sweeps. Uncertainty alone is not
a reason to demote a creative idea. Record a short rationale for the order; do not
invent precise expected AUC gains. The researcher recommends priorities, but the
orchestrator owns the final ranking.

Only actionable ideas enter the queue. Its claimed head remains counted until
training launches; launched and blocked ideas do not count. An unusable head can
be explicitly parked with a recorded reason, never silently skipped.

After reviewing imported proposals, check the campaign's refill thresholds at startup and after dequeueing or parking
an entry. Below the lower threshold, start one researcher to replenish toward the
target and exit. With the initial thresholds, four entries become three at launch,
which starts a refill toward five while the developer takes the new first entry.
Accepting and ranking proposals is serialized against the live queue and stops at the target;
surplus proposals remain in the idea registry outside the queue.
Do not pad the backlog with weak variants;
an incomplete refill records why and awaits new evidence or a changed search direction
instead of immediately spawning itself again.

The researcher returns proposals and priority recommendations; the developer returns
implementation evidence and is the only experiment-code writer. Neither edits docs,
the registry, or backlog. The orchestrator consolidates their reports, accepts ideas,
sets the ranking, and records decisions. The supervisor serializes its changes and
applies the defined mechanical launch/status transitions. Use operations against live
state, not whole-file replacements of an old queue snapshot. Recheck dependencies
and duplicate identities before accepting a researcher's proposals.
Each dependency must already have a current idea-record decision of
`provisional_accept`. Queue admission and implementation approval do not satisfy
this requirement. The orchestrator receives `proposal_admission` with dependency
status and decision from the same gate that checks its response. Unresolved
proposals stay `proposed` outside both admission and ranking lists, even when they
would be last in the queue. A rejected report leaves the queue/event intact and
pauses; resume requests a corrected orchestrator decision with the rejection reason.
Agents share artifact references and compact findings, not complete transcripts.
The supervisor handles scheduling and mechanical transitions. Unchanged rankings
are represented by `ranked_ids: null`; a changed ranking contains the complete
eligible queue. Empty researcher refills retain their report without creating an
orchestrator call and wait for new evidence. Implementation review and scientific
interpretation remain with the orchestrator. The completion reporter replaces an
eligible completion call; it does not add a summarizer before every review.

## Restricted completion reporting

Use Haiku 4.5 with no effort override for Claude and GPT-5.6 Luna at low effort for
Codex, as recorded in [agents.yaml](agents.yaml). Eligibility is code-owned:

- The fixed decision is reject or inconclusive, with a recorded clean implementation
  approval and no unresolved concerns. The attempt still awaits the decision against
  the current incumbent.
- Accounting/resource gates pass again, exact comparison facts match the completion
  event, all bootstrap draws are valid, per-target results are finite, and required
  artifacts exist. Missing or partial evidence goes to the orchestrator.
- No other events await handling and no outstanding idea shares the mechanism or
  names the completed idea as its parent or dependency. These cases need stronger
  interpretation before queue or research decisions.

Code supplies a compact result packet without the full queue/history. The reporter
can return only a factual summary, evidence links and concerns, or request escalation.
It cannot approve code, change the verdict/incumbent, rank, park, admit ideas or close
a research direction. Recheck eligibility after its response because a researcher
or developer may have added relevant events while it ran.

A reported result with no concerns is recorded directly, with queue and incumbent
unchanged. Uncertainty, concerns or changed eligibility sends the original event and
retained reporter evidence to the full orchestrator. There is at most one reporter
attempt per completion. Reporter and orchestrator event handling are serialized.
Quota, process and malformed-report failures keep the existing pause-and-resume
behavior; the event is marked for the orchestrator on resume. A cancelled reporter
also leaves its event for the orchestrator. Never launch a retry loop.

Record routing reasons, reporter outcome and both calls' usage when escalation
occurs. Assess savings per valid completed experiment, including failures and
escalation costs; short reports alone are not evidence of better research quality.

Each proposal names the closest historical attempts and a material difference.
Separate mechanism families from individual implementations. An invalid run is not
evidence against an idea; an inconclusive result is not a win or a reason to keep
drawing seeds. Revisit an attempted mechanism only with new evidence or a concrete
change. Existing general-backlog entries are read-only precedents, not ready campaign
ideas. Before adopting one, record its original link and campaign ownership in the
idea registry and resolve any overlapping active work; do not schedule it twice.

## Evidence and context

Start each assignment with the contract, current incumbent, role-specific idea and
experiment cards, and its concrete input and completion condition. Do not inject
the last twelve full reports or duplicate ranking rationales into assignments.
An immutable per-job history index links the complete record. Developers receive
their own idea and relevant predecessors; researchers receive mechanism identities
and recent results; orchestrators receive event-specific evidence and queue cards.
Keep full source, configs, logs,
and predictions addressable by path and hash outside git. Summaries preserve exact
metrics, status, and provenance; missing evidence stays missing. Retrieve originals
when a decision depends on details. Structured results are extracted by code.

Every numerical comparison includes per-label reference/candidate AUC and paired
report-group bootstrap intervals for their differences, using the macro
comparison's exact draws, seed and invalid-resample rule. The runtime supplies
these diagnostics to completion interpretation and recent research context. Use
all labels to describe tradeoffs and motivate shared mechanisms, not to cherry-pick
a label as a win. Intervals are exploratory and unadjusted for multiplicity; they
exclude training-seed, checkpoint-selection and repeated validation-selection
uncertainty. Only the fixed macro criterion can produce provisional acceptance.

Orchestrator reports contain decision changes, new findings, unresolved concerns,
and original-evidence references. Prose lengths and concern/reference counts are
advisory concision targets. The schema still enforces types, required fields,
decision enums, identifier constraints and the research proposal-count limit;
campaign gates enforce actionable decisions. Overlong prose is retained in full
without truncation, rejection or another model call solely to shorten it. The depth
of substantive review remains unchanged. Opus remains the Claude reviewer. The supervisor
assembles exact configuration differences, accounting, timings, evaluation facts,
artifact hashes, and report/log paths into `evidence.json`. Missing artifacts are
marked unavailable and remain grounds for the existing gates to reject a run.

Record model, effort, usage, repairs, and training cost per experiment, including
failed calls. Distinguish complete, partial and unavailable usage. `status` provides
a summary and `usage` provides role/event and experiment-level JSON. Measure cost
per valid completed experiment alongside missed issues, repairs and research quality;
API cost equivalents are not subscription-quota percentages. Do not tune the harness recursively during a
campaign. If Terra has a concrete implementation blocker, the
orchestrator may assign one bounded Sol/high repair; subsequent work returns to Terra.

The evidence-handling and fixed acceptance boundary draw on
[SoL-Pi, sections 2.1–2.4](https://arxiv.org/pdf/2609.20519).
Its API savings are not a forecast of subscription quota savings here.
