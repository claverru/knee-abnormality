# Implementation status

[Checkpoint closure](checkpoint-closure.md) is complete: triplet-sampling, losses
and medial-lateral-transfer all point to learned-strength-20260921 and remain
stopped, with no running/claimed attempts or pending events. The cancelled extra
run is finalized invalid; the old loss preparation is archived. Frozen source,
reference hashes, exact cleaned-input identity allowance, prior states and decisions
are recorded. The rejected adapter is removed and current-state project docs are
aligned. Original numerical limits and historical results remain unchanged.


The anatomical-attention investigation is closed by user-directed adoption of
[learned family strength](anatomical-attention.md). Main now uses the two learned
strengths with fixed 20 mm width, position-rich pooling and seed 2263738815 as
fixed behavior. Rejected width and appearance-only alternatives and experimental
flags/configs are retired; original evidence remains in the archived runs.

Autoresearch now adopts only valid macro gains >= 0.001 with a strictly positive
central 80% paired-bootstrap lower bound, discarding all other valid candidates.
The 95% intervals remain diagnostic. The production saved-prediction assessment
returns adoption for learned strength (+0.002331, 80% interval [+0.000103, +0.004518]).
New per-label reports also retain the macro
80% summary from the same draws; status/results display the adoption interval. New evaluations no longer emit
inconclusive decisions. The rule is bound into review approvals, stored in
evaluation evidence and explicitly overrides archived interval-sign guidance.
Historical decisions and stopped campaign state remain unchanged. Static lint,
format, syntax/import, main config loading, script catalog and whitespace checks
passed. No tests, training or campaign resume were launched. See [adoption policy](protocol.md#adoption-policy).

The entries below record historical implementation and experiment milestones.

The [repeated control on current data](anatomical-attention.md#repeated-control-on-current-data)
completed at 0.895112, versus historical B 0.896582 with the same seed 2263738815.
The difference is inconclusive and does not isolate seed luck from repaired-data,
source and training nondeterminism. Learned strength is +0.002331 versus the new
control (macro interval includes zero); OA-family gain +0.007352 has a positive
exploratory interval. No incumbent or historical verdict changed; CPU saved-
prediction analysis is complete, with no new training, tests or GPU work.

Incumbent-matched training seeds are implemented for all new autoresearch launches.
Preparation/review and launch use the incumbent run's saved seed, approval binds
that seed, and completion checks the actual effective config. Old prepared
approvals refresh automatically on resume; completed or already launched historical
attempts retain their original policy. Role prompts explicitly supersede archived
fresh-seed instructions without rewriting frozen sources. All three recorded
incumbents currently use 2263738815. No campaign was resumed and no runtime state
was edited. Static lint/format, syntax and import checks passed; no tests or training
were run. See [training seed policy](protocol.md#training-seed-policy).

The [learned-width result](anatomical-attention.md#learned-width-result) is complete:
macro 0.894302, -0.003141 versus learned strength (paired interval
[-0.006254, -0.000153]) and -0.002280 versus historical B (interval includes zero).
Selected widths are 15.33 mm for meniscus and 16.72 mm for OA. Three of four
compartment labels and both compartment margins are lower than strength-only.
Adoption is not supported; strength-only remains the stronger candidate in this
line. Saved-prediction CPU analysis is complete; no training, tests or GPU work
were launched, and model code/defaults remain unchanged.

The [learned-strength reset ablation](anatomical-attention.md#learned-strengths-reset-to-one-at-inference)
is complete without retraining. Resetting both strengths to one lowers macro
0.897443 to 0.897213 (difference -0.000230, interval [-0.000580, +0.000106]).
Lateral meniscus loses 0.003481; lateral OA changes only -0.000169, so most of its
gain over the separately trained fixed-bias run persists in the trained weights.
The checkpoint and main defaults remain unchanged; no new training was launched.

The [learned-strength run](anatomical-attention.md#learned-strength-result) completed:
macro 0.897443, +0.001914 versus fixed bias and +0.000861 versus historical B.
Overall intervals include zero; lateral OA improves +0.009288 versus fixed with
an exploratory positive interval. Learned strengths are 1.46533 for meniscus and
1.13049 for OA. OA compartment margin declined, so better assignment is not established.
The implementation is a confirmation candidate; main defaults remain unchanged.

The [learned anatomical family strengths](anatomical-attention.md#learned-family-strengths-next-experiment)
follow-up is implemented and its first run is complete. Static lint/format,
syntax, config, archived-source review and whitespace checks passed; no tests or GPU work were run.
Two positive learned multipliers start at one, separately for meniscus and OA,
with epoch-end logging. The candidate uses the fixed-bias run's seed 2263738815,
data and recipe. Existing controls and main defaults remain unchanged.

The [anatomical bias-off ablation](anatomical-attention.md#bias-off-inference-ablation)
is complete without retraining. Removing the bias from the same checkpoint lowers
macro AUC by 0.003619 [0.002120, 0.005213 in favor of bias on], lateral meniscus
by 0.027444, and both compartment margins. Other-eight predictions are unchanged;
intact replay reproduces saved predictions to float precision. This establishes
useful reliance within the trained checkpoint, not a training-recipe improvement.
No new training was launched and the main recipe remains unchanged.

The [anatomical-attention result](anatomical-attention.md#completed-result-against-control-b)
is complete: macro 0.895529 versus historical B 0.896582, difference -0.001053
with paired 95% interval [-0.004640, +0.002449]. OA margin improves by 0.006103
with wide uncertainty; meniscus margin is slightly lower. The fixed bias is not
supported for adoption. The user skipped the fresh control, so the repaired-data
source caveat remains despite matched seeds. Main pooling and attention remain unchanged.

The [side-aware anatomical attention](anatomical-attention.md) implementation passed
static lint, format, syntax, imports, config,
source identity, catalog and whitespace checks; no tests or GPU work were run during implementation.
It retains position-rich
pooling and independent heads, adding a fixed bounded bias for the four compartment
labels using audited knee side and patient-coordinate geometry. Fresh control and
candidate configs share seed 2263738815 and the same repaired data/source. No training
or campaign resume was launched; the main option is off.

The supervised [untied position path](untied-position.md#completed-candidate-versus-historical-control-b)
candidate finished after user-launched training: macro 0.894777 versus control B
0.896582, difference -0.001805 with paired 95% interval [-0.005212, +0.001388].
Medial meniscus and OA compartment separation regressed; adoption is not supported.
Seed 2263738815 matches B, but archived source review found the intervening restoration
of two repaired DICOM slices, so this is not a fully isolated pooling comparison.
No fresh control was launched. The main recipe retains control pooling behavior;
the campaign, watchdog and stop-after-report monitor remain stopped.

Failure-triggered monitoring is being installed for the user-started medial/lateral
campaign. A detached Python watcher checks process/state health without model calls
and permits bounded operational repairs only on failure. See the
[watchdog procedure](watchdog.md). Calibration is progressing; the campaign is now
initialized on control B, so the earlier draft rollback was not applied.

Autoresearch launch resilience is implemented: explicit add/replace integration,
approval tied to the reviewed parent and artifacts, semantic reconciliation after
an incumbent change, and renewed review when watched external evidence changes.
Active-method inventories prevent replaced ideas from satisfying dependencies and
prevent removal of prerequisites needed by retained methods. Static lint, format,
syntax, schema inspection, catalog and whitespace checks passed; no tests, model
requests or training were run. Existing campaign states are unchanged: sampling
is stopped at revision 180 and losses at 258. Losses' held unlaunched candidate
will receive a new integration assessment on resume. See
[launch and review safeguards](runtime.md#keeping-development-ahead-of-training).

The [medial/lateral-transfer campaign](campaigns/medial-lateral-transfer/INDEX.md)
is prepared and unstarted. The external `compartment-control` shared-adapter run
completed and its implementation was rejected. The draft now pins control B's
saved recipe and reference, retains the adapter as evidence, and withdraws the
obsolete adapter-off/independent seeds before admission. The remaining proposals
explore training-only supervision and token-level disease features. CPU campaign
validation passed with two seeds and a 1,797-second reference-derived ceiling;
baseline replay and source/input sealing remain startup checks. See the
[launch state and evidence](campaigns/medial-lateral-transfer/prior-work.md#active-run-and-launch-readiness).
No existing campaign state or training process was changed.

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
- One-time CPU baseline accounting, zero triplet tolerance, incumbent-matched model seeds,
  deterministic x1 validation, fold 0, fixed epochs/steps, and the 115% runtime ceiling.
- The selected training and comparison integrations, trusted numerical gates,
  local-only execution, and an explicit prohibition on submissions and publication.

Initial sealing completed after the calibration replay. Static checks on the
main-tree sampler port do not establish a new training result or model quality. Repository policy prohibits
writing or running tests.
