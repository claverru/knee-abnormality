# Campaign protocol

## Authority and scope

Creating a campaign is not starting it. The user launches the supervisor once; that
active campaign authorizes detached experiments within its sealed contract without
approval for each run. Ordinary training outside it remains user-launched. The
numeric resource limits, baseline identity, evaluation rule and data identity are
sealed before the first candidate; an adopted candidate never raises the original
budget, and missing limits or failed constraints prevent a launch.

Each campaign researches one topic indefinitely. Creative mechanisms and combined
changes within that topic are welcome; a one-variable experiment is not required.
Supporting plumbing is allowed when necessary for the mechanism. The developer
does not modify the supervisor, evaluator, acceptance policy or campaign limits.
Changing the topic or limits requires a new user decision.

Each campaign declares source/configuration permissions and topic-specific role
guidance in `campaign.yaml`; the shared runner validates and seals that policy, so a
new topic needs campaign files, not runner branches
([campaign setup](campaign-setup.md)). The numerical acceptance and resource gates
remain shared code and cannot be disabled by a topic policy.

No submissions or publication are authorized. Experiments are limited to local
training and local validation; never submit to Kaggle, upload models or predictions,
publish datasets or notebooks, or launch remote evaluation jobs. Public research
and documentation may be read.

The orchestrator owns research interpretation, proposals, idea admission and
backlog ranking. Researchers and developers return structured reports to it; the
restricted completion reporter may return a factual summary under code-enforced
eligibility, with no decision authority. The supervisor writes every canonical record.
All campaign documentation and work tracking stay inside `autoresearch/`; project
documentation elsewhere is read and linked, not edited. This user-directed boundary
replaces the general docs/ongoing and backlog workflow for campaign work. Scoped
experiment-code changes and generated runtime artifacts are separate from it.

## Shared evaluation and adoption

Every campaign uses one evaluation and one adoption rule. They are defined once in
code, `EVALUATION` and `ACCEPTANCE_RULE` in
[`campaign/rules.py`](../src/knee_abnormality/autoresearch/campaign/rules.py), and described only here.
Campaign contracts link to this section instead of restating it and cannot override
it. Changing either is a user decision that applies to all campaigns at once.

- **Metric:** macro AUC over the twelve labels on the sealed recipe's single fold of
  `all_5fold_scanner_v2`, with the `rebuild_state_v2` labels (since 2026-09-28; campaigns
  sealed before used `all_5fold_scanner_v1` and `combined_v1`). Deterministic, unaugmented
  epoch validation at image multiplier 1 selects the checkpoint; that checkpoint's
  saved predictions on that fold at image multiplier 4 and attention temperature 1
  are scored (since 2026-09-22; campaigns sealed before used the ×1 predictions).
- **Uncertainty:** paired report-group bootstrap on those saved predictions,
  2,000 draws with resampling seed 20260914, identical draws for candidate and
  reference.
- **Enforcement:** configuration gates reject any recipe whose split, folds, label
  source, validation multiplier or prediction multiplier differs; the seal takes the
  bootstrap settings from the same definition.
- **Soft AUC (diagnostic):** each comparison also records soft macro and per-label
  AUC (pairs weighted by the soft targets) as point estimates, shown in results and
  given to the orchestrator. It does not enter the adoption rule.

Campaigns sealed under an earlier evaluation cannot resume; continuing one of their
topics needs a new campaign sealed on the shared evaluation.

### Adoption policy

For each valid new experiment, adopt the implementation when its unrounded macro
AUC exceeds the pinned incumbent's by at least 0.001 **and** the central 80% paired
report-group bootstrap interval has a strictly positive lower bound; otherwise
discard it. Both conditions are required, using unrounded values. Code computes the
verdict: `provisional_accept` when both hold, `reject` when either fails, `invalid`
for invalid evidence, leakage, seed mismatch, or contract or resource breaches.
`provisional_accept` is a practical working-recipe choice, not statistical
confirmation; `reject` means not retained under these engineering thresholds, not
proof of harm. Historical records may carry the retired `inconclusive` verdict; they
are not recomputed or relabeled. Keep one working incumbent. Do not leave pending implementation branches or
schedule confirmation runs after each experiment. Preserve findings, configs and
provenance for discarded attempts without keeping their rejected implementations in
the active recipe.

The 80% macro interval uses the 10th and 90th percentiles of the same paired
report-group draws as the retained 95% interval. Per-label and 95% intervals inform
interpretation; they add no further adoption gate. Missing 80% evidence prevents
adoption. The 0.001 threshold and 80% level are fixed engineering choices, not
independent confirmation, and repeated selection can still overfit the campaign's
fold. This user-directed rule replaced a positive-95%-lower-bound gate and a briefly
used gain-only rule. Review approvals bind the rule and evaluation records carry it;
role prompts override older gates in frozen contracts without modifying archived
source. Adopted methods become the next reference for both additions and replacements.

## Training seed policy

Every new candidate uses the seed recorded in the current incumbent run's
`effective_config.json`. The supervisor owns this value; candidate source configs
and experiment metadata cannot select a different seed. It binds the seed into
approval and sets it at launch, then checks it against the completed run's effective
configuration. An incumbent change requires reconciliation and review with that
incumbent's seed. Missing or malformed incumbent seeds block preparation and review
rather than drawing a fallback.

Already launched attempts keep their historical seed and evaluation record; no
completed experiment is relabeled as matched. Genuine checkpoint resumes retain the
original run seed. The bootstrap seed is independent and unchanged. Matched seeds
reduce one source of comparison noise; they do not guarantee deterministic training
or measure robustness across training seeds.

**Adoption keeps the adopted run (since 2026-09-26, user decision).** A
`provisional_accept` makes the adopted run itself the incumbent reference, and later
candidates train at its seed. The campaign draws no fresh seed and retrains nothing:
the fresh-seed control retrain belongs to the user when an adoption is merged into
main ([project rule](../docs/validation/validation-strategy.md#controlled-comparisons)).

**Fold from the recipe (since 2026-09-25).** The campaign's fold is the single
`validation.folds` entry of its sealed recipe, and its reference run must be on that
fold. A campaign runs on that fold from start to end; fold rotation happens only when
the user merges an adoption into main
([project rule](../docs/validation/validation-strategy.md#controlled-comparisons)).
Training, comparison, accounting and reporting all use it.

## Supervisor and agents

The supervisor is ordinary code that schedules roles on events and applies every
mechanical transition; the [idea lifecycle](lifecycle.md) describes each step from
claim to completion. Agents exit when their assignment is complete. They never poll
training, sleep through a run, or repeatedly invoke tools to discover whether
another agent is done. A failure produces one event with evidence, not an unlimited
retry loop; [automatic recovery](runtime.md#inspect-stop-and-resume) bounds rejected
reports, provider errors and early training crashes. Do not tune the harness
recursively during a campaign.

The researcher returns proposals and priority recommendations; the developer returns
implementation evidence and is the only experiment-code writer. Neither edits docs,
the registry or the backlog. The orchestrator consolidates their reports, admits
ideas, sets the ranking, reviews implementations and interprets results. The
supervisor serializes its decisions against live state, never whole-file
replacements of an old snapshot. Agents share artifact references and compact
findings, not transcripts.

## Backlog and ownership

The backlog is ranked by the orchestrator, highest potential first. The developer
always takes the first entry at assignment time; it does not choose its own idea.
Rerank unclaimed entries when research proposals arrive or results change the
evidence. The claimed head stays reserved through development and preparation; new
ideas cannot displace work already underway. Only actionable ideas enter the queue,
and an unusable head is parked with a recorded reason, never silently skipped. Do not
pad the backlog with weak variants; an incomplete refill records why and awaits new
evidence or a changed search direction instead of immediately spawning itself again.

Rank by plausible improvement potential, relevance to a weakness in the campaign
topic, novelty relative to prior work, learning value, and implementation and compute
cost. Prefer mechanisms with meaningful upside over easy scalar sweeps. Uncertainty
alone is not a reason to demote a creative idea. Record a short rationale for the
order; do not invent precise expected AUC gains. The researcher recommends
priorities, but the orchestrator owns the final ranking.

Each proposal names the closest historical attempts and a material difference.
Separate mechanism families from individual implementations. An invalid run is not
evidence against an idea, and a reject is not proof of harm; neither is a reason to
draw more seeds. Revisit an attempted mechanism only with new evidence or a concrete
change. Existing general-backlog entries are read-only precedents, not ready
campaign ideas.

## Restricted completion reporting

Use Haiku 4.5 with no effort override for Claude and GPT-6 Luna at low effort for
Codex, as recorded in [agents.yaml](agents.yaml). Eligibility is code-owned:

- The fixed decision is `reject`, with a recorded implementation approval, and the
  attempt still awaits its decision against the current incumbent.
- Accounting and resource gates pass again, exact comparison facts match the
  completion event, all bootstrap draws are valid, per-target results are finite and
  required artifacts exist. Missing or partial evidence goes to the orchestrator.
- Review concerns, other pending events and related outstanding ideas do not block
  it; review concerns are copied into the recorded finding. The measurement behind
  this is in [efficiency](efficiency.md#knee-geometry-baseline-and-completion-routing).

Code supplies a compact result packet without the full queue or history. The
reporter can return only a factual summary, evidence links and concerns, or request
escalation. It cannot approve code, change the verdict or incumbent, rank, park,
admit ideas or close a research direction. Eligibility is rechecked after its
response because state may have changed while it ran. Only `status: escalate` or a
failed recheck sends the event to the orchestrator; otherwise the finding and the
reporter's concerns are recorded with queue and incumbent unchanged.

There is at most one reporter attempt per completion, and reporter and orchestrator
event handling are serialized. A failed, malformed, rejected or cancelled reporter
escalates its event to the orchestrator without another reporter attempt. In every
route the supervisor records the fixed `eligible_decision`; the orchestrator
interprets but never applies a verdict. Usage of both calls is recorded on escalation.

## Evidence and context

Every role prompt is the repository `AGENTS.md`, the shared rules in
`supervisor/prompts.py`, the [role brief](roles/INDEX.md), the campaign contract and the
assignment. Each rule has one home: rules for every role in `prompts.py`, role behavior
in its brief, and length targets in the report schemas.

Start each assignment with the contract, current incumbent, role-specific idea and
experiment cards, and its concrete input and completion condition. Do not inject
the last twelve full reports or duplicate ranking rationales into assignments.
An immutable per-job history index links the complete record. Developers receive
their own idea and relevant predecessors; researchers receive the family ledger
([records](records.md#family-verdicts)) and full detail only for results decided since
their last report; orchestrators receive event-specific evidence and queue cards.
Developers and candidate reviews also receive `campaign_defects`, the last twelve
repair instructions and invalid-run reasons of other attempts
([lifecycle](lifecycle.md#development-and-gates)). Keep full source, configs, logs
and predictions addressable by path and hash outside git. Summaries preserve exact
metrics, status and provenance; missing evidence stays missing. Retrieve originals
when a decision depends on details. Structured results are extracted by code.

Every numerical comparison includes per-label reference/candidate AUC and paired
report-group bootstrap intervals for their differences, using the macro
comparison's exact draws, seed and invalid-resample rule. The runtime supplies
these diagnostics to completion interpretation and recent research context. Use
all labels to describe tradeoffs and motivate shared mechanisms, not to cherry-pick
a label as a win. Intervals are exploratory and unadjusted for multiplicity; they
exclude training-seed, checkpoint-selection and repeated validation-selection
uncertainty. Only the fixed macro criterion can produce provisional acceptance.

Reports carry decision changes, new findings, unresolved concerns and
original-evidence references. Prose lengths and concern/reference counts are advisory
targets: overlong prose is kept in full, never truncated, rejected or sent back to be
shortened, and review depth is unchanged. Types, required fields, decision enums,
identifiers, the proposal-count limit and campaign gates stay strict. The supervisor
assembles exact configuration differences, accounting, timings, evaluation facts,
artifact hashes and report/log paths into `evidence.json`; missing artifacts are
marked unavailable and remain grounds for the gates to reject a run. Usage is
recorded per job and experiment, failed calls included ([records](records.md#agent-handoffs-and-usage)),
and efficiency is judged per valid completed experiment alongside missed issues,
repairs and research quality.

The evidence-handling and fixed acceptance boundary draw on
[SoL-Pi, sections 2.1–2.4](https://arxiv.org/pdf/2609.20519).
Its API savings are not a forecast of subscription quota savings here.
