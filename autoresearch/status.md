# Implementation status

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
retains its prior state. Losses remains not started.

Declarative campaign setup and the second campaign, `losses`, are ready as of
2026-09-20. Losses investigates one shared objective or shared combination across
all labels, excluding per-label selection, coefficients and schedules. Its four
seed proposals cover BCE plus ranking, focal loss, tempered class weighting and
delayed ranking. Control B supplies its initial reference and 1,797-second ceiling;
the campaign remains **not started**. See the [loss contract](campaigns/losses/contract.md).

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

Context and usage work is complete: focused evidence packets and bounded reports
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
- Focused role context, evidence packets, bounded reports, unchanged-ranking
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
