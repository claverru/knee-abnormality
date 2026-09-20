# Implementation status

Triplet sampling's resume-readiness repair is complete at revision 180. The sole
queued idea, `physical-span-standardized-triplets`, now has the required narrow
`preparation.py` permission in both its campaign definition and saved policy.
The user-authorized [amendment and recovery snapshots](../outputs/autoresearch/triplet-sampling/decisions/user-authorize-triplet-preparation-20260920T192746Z-e03bfe70/decision.json)
preserve control B, all experiments, the two blocked follow-ups and the original
numerical seal. Campaign validation, Claude subscription authentication and Linux
isolation preflight passed. The campaign remains stopped with no active jobs.
Its displayed researcher failure is the historical Claude session-quota HTTP 429;
resume clears that reason, but authentication does not establish remaining quota.
No tests, model requests or training were run during this repair.

User-requested literature proposals were imported into stopped losses through the
normal import workflow: ROC-star-style buffered squared-hinge ranking
and Hill-negative soft BCE, with distribution-balanced loss as a reserve.
The [import source](campaigns/losses/literature-ideas.yaml) records prior outcomes,
shared-objective constraints and memory-bank overlap for orchestrator review.
Losses is stopped at the user's request after its draw-consistency result was
recorded on 2026-09-20. Its CPU comparison and completion interpretation were
finalized with scheduling disabled; the campaign was not resumed and must not
resume yet. The run scored 0.893567945 against control 0.896581669, with a paired
macro difference interval of [-0.006791044, 0.000618211]: inconclusive. All twelve
label diagnostics are saved; total measured compute was 1,712.00 seconds within
the 1,797-second ceiling. See the
[completion report](../outputs/autoresearch/losses/jobs/orchestrator-20260920T184207Z-3e033654/report.json).
There are no active jobs. The
[imported proposals](../outputs/autoresearch/losses/imports/ideas-20260920T184115Z-bcbbafcc/report.json)
await orchestrator admission/ranking after an authorized resume; they are not yet
queued. The reserved group-variance candidate is preserved.

Default per-label reporting is implemented: every comparison saves all twelve
reference/candidate AUCs, exact deltas, paired report-group bootstrap 95% intervals,
class counts and valid draw counts. Results and agent context include them as
unadjusted exploratory diagnostics. Macro acceptance, the frozen evaluator and
campaign limits remain unchanged. See [result artifacts](records.md#experiments)
and [runtime commands](runtime.md#inspect-stop-and-resume).

All 16 historical comparisons were backfilled from saved predictions: eight losses
and eight sampling results, each reproducing its original macro bootstrap with
2,000 valid draws. Historical comparisons, evaluations and decisions were retained.
Backfill is idempotent and defers in-flight comparisons to their automatic writer.

Losses finished its active training, stopped safely, and resumed with the updated
runtime on 2026-09-20. Its first automatic comparison produced all twelve label
intervals successfully; total measured compute was 1,610.92 seconds against the
1,797-second ceiling. The [comparison job](../outputs/autoresearch/losses/jobs/comparison-20260920T161153Z-3513bda9/receipt.json)
finished in 6.04 seconds. The supervisor, Claude developer and completion
orchestrator were verified running with no pause reason. Sampling remains stopped.
Lint/format, catalog, whitespace, results display and actual backfill checks passed.
No tests were written or run.

Report-length validation is repaired after Claude exhausted five structured-output
retries, ending with a 2,402-character rationale against a 2,400-character cap.
Narrative lengths and concern/reference counts are now advisory targets for every
role, with full content preserved. Typed action fields, required evidence,
identifier/proposal-count constraints and campaign decision gates stay enforced.
All five saved rejected payloads validate unchanged under the updated schema;
lint/format, catalog and whitespace checks passed. No tests were written or run.
Losses was stopped safely and resumed with the new schemas. The replacement
orchestrator report was accepted, the queue refilled to five, and the completed
contrast run passed through comparison without retraining. Completion reporting
is active and Brier remains prepared next; the campaign has no pause reason.
The successful replacement report is retained in
[its job record](../outputs/autoresearch/losses/jobs/orchestrator-20260920T134604Z-5dfdd370/report.json).

Dependency admission guidance is repaired. Orchestrator context now exposes each
proposal's unresolved prerequisites using the same gate as response validation;
role/schema instructions distinguish queue admission from provisional acceptance.
Rejections identify the proposal, dependency status and required correction. The
dependency gate, numerical acceptance rules and campaign limits are unchanged.

Losses was stopped safely, updated and resumed after its proposal-admission pause.
The orchestrator applied the corrected queue and retained
`shared-ranking-memory-bank` as `proposed` with its dependency intact. The saved
BCE-plus-ranking run completed comparison without retraining and was recorded
inconclusive. Focal then trained normally. Lint/format, catalog and whitespace
checks passed for that dependency repair. No tests were written or run.

Progress reporting is implemented. `status` shows starting/current AUC, raw change,
retained experimental gain and the latest result's scores and interval. The
read-only `results` command and `autoresearch-results` shortcut show every decided
attempt against its actual reference, preserving withdrawn acceptances and listing
manual reference changes and training attempts awaiting a decision separately.
See [runtime operation](runtime.md#inspect-stop-and-resume).

Lint/format, catalog consistency, whitespace checks and read-only CLI inspection
passed for the existing sampling history and unstarted losses campaign. Sampling
shows AUC 0.894037 to 0.896582 but zero retained experimental gain, one withdrawn
acceptance and two manual reference changes. No tests, campaign agents or training
were launched, and no campaign state was changed.

Restricted completion reporting is implemented: Haiku 4.5 without an effort
override for Claude, and GPT-5.6 Luna at low effort for Codex. Code selects clean
rejected/inconclusive completions with complete evidence and no pending related
decisions, then rechecks eligibility before recording the result. The reporter
has no action fields; concerns or uncertainty escalate to Opus/Sol. Reviews,
scientific interpretation, wins, invalid runs and backlog decisions remain with
the orchestrator. See [routing rules](protocol.md#restricted-completion-reporting).

Reporter and escalation usage is included in role/event and experiment totals;
status exposes routing counts. Static lint/format, schema inspection, catalog,
CLI help and read-only status/usage checks passed. No tests, campaign agent calls
or training were launched; runtime quality and quota savings remain unmeasured.
The stopped sampling campaign received the added profiles at revision 179,
preserving its existing roles, limits, incumbent, queue and experiments. The
[profile update record](../outputs/autoresearch/triplet-sampling/decisions/completion-reporter-profiles-20260920T104935Z-92c8a5fa/decision.json)
retains its prior state. Losses had not started at that profile update.

Declarative campaign setup and the second campaign, `losses`, are ready as of
2026-09-20. Losses investigates one shared objective or shared combination across
all labels, excluding per-label selection, coefficients and schedules. Its four
seed proposals cover BCE plus ranking, focal loss, tempered class weighting and
delayed ranking. Control B supplies its initial reference and 1,797-second ceiling;
the campaign was subsequently launched by the user on 2026-09-20. See the
[loss contract](campaigns/losses/contract.md).

New topics using the current training/evaluation protocol now require campaign
files instead of runner edits. A strict YAML policy declares editable source,
config differences, input-identity exceptions and topic-specific role instructions;
initialization seals it in state. Numerical evaluation, accounting, scheduling and
GPU leases remain shared. The [setup guide](campaign-setup.md) documents the format
and the CPU-only `validate` command. Static lint/format, both campaign validations,
catalog consistency, CLI help/status and launch-command routing passed. No tests
or training were run, and no campaign agents were launched.

The stopped sampling campaign was explicitly migrated to the same policy format
at revision 178. Its current control-B incumbent, queue, experiments and original
1,785-second ceiling are unchanged. The original state and equivalent policy are
retained in the [migration record](../outputs/autoresearch/triplet-sampling/decisions/campaign-policy-migration-20260920T100729Z-22dfb2d4/decision.json).

Context and usage work is complete: focused evidence packets and concise reports
replace repeated historical prose, Opus keeps substantive review, unchanged
rankings need no rewrite, and code applies the fixed numerical verdict. Usage
includes failed calls and explicitly marks incomplete records. Static lint,
catalog consistency, CLI help and read-only status/usage inspection passed.
See [efficiency](efficiency.md) for the measured baseline and remaining
next-campaign measurement. No model or training was launched; the campaign remains
stopped. The maintained runner and briefs take effect on its next launch.

The detached runtime is implemented. Static lint, script-catalog consistency, CLI
help/status, and command routing checks passed. Both subscription logins and the
required Linux isolation are available. The campaign sealed its initial recipe against the historical control, giving
a 1,785-second ceiling, ten epochs, and 4,380 optimizer steps. The Claude campaign
completed initial calibration and is now stopped at the user's request. Its previously provisionally accepted
[run-level stratified sampler](../docs/validation/results/2026-09-20-stratified-sampling-promotion.md)
was briefly promoted into main on 2026-09-20, then reverted and **rejected for
adoption** at the user's request after inconclusive confirmation. Main uses the
previous independent-per-epoch sampler. Campaign revision 176 records `reject`
in the idea registry and runtime and restored `historical-control` at that point.
At revision 177 the user replaced the incumbent with **control B**
(`control-b-20260919T235434Z`, seed 2263738815, fold-0 validation ×1 AUC
0.8965816694263277). The next candidate will compare against B using the existing
single-incumbent rule. Its prediction hash and source compatibility are recorded
in [reference evidence](campaigns/triplet-sampling/prior-work.md#reference-evidence).
The original historical control still anchors the unchanged sealed resource limits.
The original positive bootstrap and provisional acceptance remain historical evidence;
the rejection does not claim statistically established harm. See
[prior work](campaigns/triplet-sampling/prior-work.md#run-level-stratified-jitter-rejected-for-adoption).
The dependent prepared `prefix-balanced-stratum-order` is parked, and the trained
`plane-coupled-depth-targets` is held unevaluated for explicit review because its
source and reference contain the withdrawn sampler. Their artifacts remain intact.
The campaign stays stopped with its original sealed limits.
Use `autoresearch-status` for current progress. All research documentation stays
inside `autoresearch/`.

Implemented:

- Start/status/stop/resume for Codex and Claude subscription CLIs with fixed role profiles.
- Human-readable monitoring summary with calibration, jobs, backlog, training, and results.
- Role/event usage and read-only JSON accounting by experiment, with partial and
  missing coverage, repairs and cost per valid completed experiment.
- Focused role context, evidence packets, concise reports, unchanged-ranking
  responses and deterministic completion verdict application, retaining Opus review.
- Detached supervisor and workers, durable intents/receipts, process identity checks,
  campaign/GPU leases, interrupted-job recovery, and pause on quota or unexpected failures.
- Ranked reserved-head backlog, orchestrator-only canonical writing, history retention,
  refill below four toward five, and one candidate developed ahead of training.
- Initial proposals imported from existing repository ideas before fresh research,
  with provenance, explicit blockers and locked import into a stopped campaign.
- Independent workspaces, frozen training source, and three-way reconciliation onto
  a newly accepted incumbent, with developer conflict resolution and renewed review.
- One-time CPU baseline accounting, zero triplet tolerance, fresh model seeds,
  deterministic x1 validation, fold 0, fixed epochs/steps, and the 115% runtime ceiling.
- The selected training and comparison integrations, trusted numerical gates,
  local-only execution, and an explicit prohibition on submissions and publication.

Initial sealing completed after the calibration replay. Static checks on the
main-tree sampler port do not establish a new training result or model quality. Repository policy prohibits
writing or running tests.
