# Implementation status

Knee-geometry campaign stopped, 2026-09-24 (user decision): 18 decisions, one adoption,
`meniscus-half-knee-mil` (fitted midline, meniscus-only side mask, training-only half-knee loss;
+0.00307 fold 0, 0.944 public as S54). The last run, `meniscus-core-band-contrast-readout`, trained
but was not compared. The user adopted the model on main, added the robust centre and bilateral
knee geometry outside the rule, and retrained the control at a fresh seed
([findings](campaigns/knee-geometry/findings.md), [contract](campaigns/knee-geometry/contract.md)).
The first launch was restarted after calibration to make `inference/engine.py` editable: the final
×4 prediction moves only `MODEL_BATCH_KEYS` to the GPU. The watchdog is stopped.

Noisy-labels campaign stopped, 2026-09-23 (user decision): six experiments decided, none adopted
(see [findings](campaigns/noisy-labels/findings.md)). The user retired two queued memory-based
ideas mid-campaign, then imported the schedule plus early-learning combination (rejected, −0.0003).
The researcher's last refill, `full-to-half-bag-distillation`, was aborted seconds into training
and stays undecided. The watchdog is stopped.

Noisy-labels campaign defined, 2026-09-23 (user request): [`noisy-labels`](campaigns/noisy-labels/INDEX.md)
covers training-objective mechanisms that use the model's own training-time predictions to correct
or down-weight report labels, plus a class-balance schedule, with five seed ideas. Its reference is
`k05-clean` (S48), trained while the disabled oblique code still existed. Instead of retraining it,
the user authorized reference exceptions: `reference_config_exceptions` and
`reference_identity_exceptions` in `campaign.yaml` name recorded fields removed from both sides of
the start, sealing and comparison checks (`without_fields` in `gates.py`; the supervisor passes
them to `compare_fold_runs --reference-identity-exceptions`). The contract records why the removed
code never ran. `validate noisy-labels` passes (ceiling 1,846 s), and the current preprocessing
identity equals `k05-clean`'s after the exceptions. The contract keeps the shared evaluation and requires every completion to report soft
macro AUC beside hard, because hard AUC against report labels penalizes this family. Not started.

Oblique line closed, 2026-09-23 (user decision): oblique groups and the group embedding are removed
from the recipe and the code, and `oblique-clean` is retired alongside `oblique` and `oblique-x4`.
No campaign is active. `oblique-group-embedding`, trained but never compared by the supervisor,
was compared by hand: +0.0015 hard and +0.0019 soft at ×4 against `oblique-clean`. Its S51
submission scored 0.942, and S52 (the same code on image-blend labels) scored 0.943 against 0.945
for S50 ([history](../docs/history/oblique-groups.md)). The campaign's state and outputs are
unchanged. It cannot resume on main, whose preprocessing identity no longer matches its sealed
source. The justfile defaults still name it, and only its read-only commands are useful.

Automatic recovery, 2026-09-23 (user decision): all nine campaign stops so far came from five
agent reports that broke a runtime rule and four provider errors. A rejected orchestrator or
developer report now returns to the same role with the error as `previous_rejection`; consecutive
rejections per event or experiment reset when a report applies, and only the third pauses. A
rejected completion-reporter report escalates to the orchestrator without pausing. Failed agent
jobs are classified from Claude's stream-json result or Codex's last error event: overload, 5xx
and network errors retry the same work item after 60 s, 300 s and 900 s, then pause; a session or
usage limit (429) sets `retry_after` to the parsed reset (60 minutes if unparseable), during which
no agent job starts while training and comparison continue, and `status` shows the wait;
authentication and other failures pause as before. A launched training that exits nonzero before
completing an epoch records no completion and returns to its developer as the attempt's single
repair, with the stderr tail as the instruction. The watchdog treats a provider wait as healthy
and answers a `Report rejected` pause with one plain resume before its repair flow. Gates,
evaluation, acceptance, accounting, limits, approval binding and seeds are unchanged. Only static
checks ran; the parser classified the real 2026-09-22 session-limit failure as a limit with the
retry at 17:51. `oblique-clean` was stopped for the merge with `oblique-group-embedding` trained
(10/10) and awaiting its comparison, which runs on resume.

Agent models, 2026-09-23 (user decision): [agents.yaml](agents.yaml) moves the Claude
orchestrator and researcher to Opus 5.5, and the Codex orchestrator, completion reporter and
watchdog to GPT-6 Sol and GPT-6 Luna, and the Codex developer from GPT-5.6 Terra to GPT-6 Sol at
high, which retires the separate Sol/high repair override. The Claude developer (Sonnet 5), the
Codex researcher (GPT-6 Astra) and the Claude completion reporter (Haiku 4.5) are unchanged, as are
all efforts. Campaigns seal their profiles at start, so
retired campaigns keep their recorded models; the watchdog reads the file on each repair.

Cleaned-source oblique campaign, 2026-09-22 (user decision): main gained series cleaning
(`b483140`) and loader and inference speedups (`4d23276`, `9e0c3ee`) that the x4 campaign's frozen
source predates, and a started campaign cannot change its reference. The user retired
[`oblique-x4`](campaigns/oblique-x4/INDEX.md) at revision 78, stopped on a developer's Claude
session limit (HTTP 429), after three rejections; `acl-gated-oblique-tokens` finished training
(×1 0.895663) without a comparison and `oblique-standard-frame-depth` was not developed. The
[`oblique-clean` campaign](campaigns/oblique-clean/INDEX.md) copies its contract and policy against
`outputs/training/oblique-clean`, the oblique recipe retrained on the current source at seed
2263738815, with six carried-over seeds. Validation against the old reference passed (6 seeds,
policy, recipe compatibility). One chained command trained `k05-clean` (`k_05`'s recipe,
`configs/k05.yaml`, an evidence run; ×4 0.902760) and `oblique-clean` (×4 0.903705), then validated
and started the campaign with its [watchdog](watchdog.md). The first candidate,
`acl-gated-oblique-tokens`, crashed on its first training batch (a per-acquisition gate flag not
expanded to spatial tokens). The orchestrator's completion re-ranked that launched idea into the
queue, which the ranking gate rejects; the watchdog judged the repair out of scope and stopped the
campaign at 01:22 on 2026-09-23. With the user's approval the fixed gate was imported as
`acl-gated-oblique-tokens-token-mask` ([fix ideas](campaigns/oblique-clean/fix-ideas.yaml)), the
campaign resumed so the orchestrator retries with the rejection reason, and the watchdog restarted. The justfile recipes default to it.

Shared evaluation on ×4 predictions, 2026-09-22 (user decision): `EVALUATION` in `gates.py`
now carries `validation.prediction_image_multiplier: 4`. Training still selects the checkpoint
by ×1 epoch validation, then writes that checkpoint's fold-0 predictions at ×4 and attention
temperature 1 with their macro AUC in `selected_checkpoint.json`; `compare_fold_runs` checks the
saved predictions against that value. The `oblique` campaign, sealed on ×1 predictions, is
retired after two rejected experiments and one trained but uncompared (residual readout, ×1
0.899668). Its topic continues in the [x4 oblique campaign](campaigns/oblique-x4/INDEX.md),
referenced to `outputs/training/oblique-x4` (the `oblique` run re-predicted at ×4 by
`scripts/tools/repredict_run.py`, 0.902373) with eleven carried-over ideas, of which the user kept seven; `validate oblique-x4`
passes (ceiling 1,902 s). The user started it; its first review admitted five of the seven to the backlog, and the four excluded ideas stay registered as proposals only. The justfile recipes default to it.

Oblique campaign, 2026-09-22 (user decisions): the [oblique campaign](campaigns/oblique/INDEX.md)
is defined against `outputs/training/oblique` (0.897527), the recipe that gives oblique series
their own slot budget, with six seed ideas and budget changes limited to exact redistribution.
`validate oblique` passes (runtime ceiling 1,868 s, 4,380 steps, no equivalences); it is not sealed
or started. The sampling campaign is retired: its supervisor stopped at revision 145 when an
orchestrator job hit a transient provider error (API 529), after six rejected experiments, and it
stays a historical record. No GPU work or training was started by this change.

Shared evaluation and sampling campaign, 2026-09-21 (user decision): the metric,
split, labels, validation multiplier, bootstrap and adoption rule are one
definition shared by every campaign, `EVALUATION` and `ACCEPTANCE_RULE` in
`gates.py`, documented in the [protocol](protocol.md#shared-evaluation-and-adoption).
Gates now require fold 0 of `all_5fold_scanner_v1` with `combined_v1` labels at
validation x1; agent prompts and diagnostics derive the rule text from the same
constants, and role briefs and records link to the protocol instead of restating it.
Triplet-sampling, losses and medial-lateral-transfer remain stopped historical
records; their seals use `all_5fold_v3` and Steven v4 labels, so they cannot resume.
Campaign definitions gained `reference_equivalences` for recipe keys added after the
control that reproduce its behavior. The new [sampling campaign](campaigns/sampling/INDEX.md)
was first started against `fold0-combined-scanner`, sealed, and stopped by the user
before any training when the champion recipe became `class_balance_exponent: 0.5`;
that start is kept under `outputs/autoresearch/discarded/`. The campaign now passes
`validate` against `outputs/training/k_05` (0.899042, ceiling 1,847 s, no
equivalences) and is not sealed or started; its seven initial ideas are in
`seed-ideas.yaml` and `seed-ideas-2.yaml`, and startup now imports every
`seed-ideas*.yaml` file. Study visit frequency
stays fixed at once per epoch, so the existing triplet accounting applies unchanged.
Just recipes default to `sampling`. No training, GPU work or campaign start was performed.

Runtime hardening, 2026-09-21: Failed application of a finished training,
comparison, calibration or configuration result now rolls the state back to
before that job was marked processed and records `application_error` on the
job instead of stranding the experiment; resume clears the marker and retries
application from the saved receipt. A normal stop no longer invalidates a
comparison that already started; only `--abort` terminates it, and a comparison
cancelled before it started is retried after resume. Usage accounting failures
during a job or a status/usage read are now caught and recorded as unavailable
instead of raising, so they never block result processing or status. A ready
comparison is dispatched as soon as training finishes, without waiting for
unrelated orchestrator or completion-reporter work. The watchdog's repair scope
is now enforced by comparing actual before/after file hashes of the checkout
against an explicit allowlist, rather than trusting the repair agent's own
account of what it touched. See [runtime operation](runtime.md#inspect-stop-and-resume)
and [watchdog recovery](watchdog.md#recovery-and-limits).

User-requested follow-up, 2026-09-21: [corrected attribution exclusivity](campaigns/medial-lateral-transfer/prior-work.md#corrected-exclusivity-follow-up)
is first in the stopped campaign's queue, followed by a fresh retry of the
user-aborted local-lesion run. The correction replaces the inactive squared hinge
with a linear zero-margin penalty. It uses ordinary full-budget training and logs
activity without a monitor or adaptive stop. Original results, reference, seed and
resource limits remain unchanged.

Operational recovery, 2026-09-21: Claude returned null `output_tokens_details`
during a developer OAuth refresh failure. Usage accounting now handles that null
as absent thinking-token details, so status and failure harvesting no longer
crash. The original failed receipt is preserved; the campaign was resumed once
to retry the same untrained candidate. The retry passed authentication: supervisor
and developer are live, and the developer is issuing tools on the first candidate.
The campaign is running; the stopped-state notes below describe earlier preparation.
Authentication failures remain blockers, not grounds for repeated automatic retries.

[Checkpoint closure](checkpoint-closure.md) is complete: triplet-sampling, losses
and medial-lateral-transfer all point to learned-strength-20260921 and remain
stopped, with no running/claimed attempts or pending events. The cancelled extra
run is finalized invalid; the old loss preparation is archived. Frozen source,
reference hashes, exact cleaned-input identity allowance, prior states and decisions
are recorded. The rejected adapter is removed and current-state project docs are
aligned. Original numerical limits and historical results remain unchanged.

Medial/lateral transfer now targets compartment-specific disease improvement,
including knee-side/position encoding. Its [refreshed queue](campaigns/medial-lateral-transfer/prior-work.md#revisit-queue-on-learned-strength)
contains three historical adaptations followed by explicit side-location encoding
and anatomy-supervised assignment. Learned strength remains the control; the recent
unbiased control-repeat is supporting evidence. Old experiments and limits are
unchanged; displaced proposals remain available. The campaign is prepared but stopped.


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
