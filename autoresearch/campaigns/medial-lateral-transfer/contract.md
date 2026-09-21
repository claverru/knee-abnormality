# Medial and lateral disease-transfer campaign

Current state, 2026-09-21: stopped on the user-adopted learned-strength incumbent
(seed 2263738815, fold-0 AUC 0.897443). The source/reference transition is recorded
in [the shared closure record](../../checkpoint-closure.md). Original reference,
budget and startup sections below describe the historical seal; they do not select
the current incumbent. Historical attempts retain their original references.
The [2026-09-21 queue refresh](prior-work.md#revisit-queue-on-learned-strength)
prioritizes three historical adaptations, explicit side-location encoding and an
anatomy-supervised gate. The campaign remains stopped.

## Objective and scope

The user explicitly authorized the [corrected exclusivity follow-up](prior-work.md#corrected-exclusivity-follow-up)
on 2026-09-21. This single correction of the inactive hinge is eligible despite
older findings describing margin changes as inadmissible sweeps. Require measured
training-only activation and gradient contribution; preserve all fixed numerical
and resource gates and do not relabel the old completed result.
The exact `active-gradient-attribution-exclusivity` proposal from
`corrected-exclusivity.yaml` is now first in the stopped campaign's live queue,
followed by a fresh local-lesion retry. Retain normal implementation, review and
resource checks. The corrected run uses no separate monitor or adaptive stopping.

The user's original scope, verbatim:

> some diseases appear lateral and medial so they should share some patterns that the model should be able to exploit.

On 2026-09-21 the user broadened the objective to improving compartment-specific
diseases, explicitly including side-encoding solutions. Knee-side and anatomical
position encoding, and anatomy-supervised compartment assignment, are eligible
even without additional disease-pattern sharing. This supersedes older scope
deferrals for those mechanisms. Preserve trusted-side provenance, explicit unknown
side/invalid-geometry handling and reflection consistency; do not derive side from
disease labels, report text, identifiers or validation outcomes.

Explore how the model can exploit those shared patterns while making the correct
prediction for each compartment. Both the meniscus pair and medial/lateral OA
belong to this investigation. A first experiment on meniscus does not restrict the
campaign to meniscus. Shared classifiers, shared features, attention, anatomical
assignment, shared/private representations, learned sharing, and training-only
transfer objectives are possible mechanisms, not a prescribed architecture.
Combined mechanisms are permitted when their rationale and interpretation are
explicit. Each proposal must explain its connection to this scope.

Both addition and substitution are valid: compare A+B against accepted A when
stacking, or B against A when replacing it. Declare retained, added and removed
components and any removed accepted idea IDs in the required integration plan.
Preserve unrelated improvements; a new winner during development requires renewed
integration assessment and review before training, even after a clean merge.

The model already shares its backbone and feature processing. Establish what
additional transfer a candidate supplies. Similar disease patterns do not imply
equal predictions: neither, medial-only, lateral-only, and both-positive cases
must all remain representable. Reflection of a whole knee preserves medial/lateral
target identities; it does not exchange the labels.

Keep the surrounding recipe fixed. Backbone replacement, generic loss or
hyperparameter searches, label repair, new datasets, external annotation/model
pipelines, triplet-sampling changes, ensembles, and extra training stages are
outside this campaign. Supporting code and auxiliary supervision directly needed
by a transfer mechanism are allowed within the declared source policy. Existing
unrelated campaigns retain their ownership.

**Only local training and local validation are authorized.** No submissions,
uploads, publication, or remote evaluation. The external run has completed; the
user starts this campaign when ready. Writing these files does not launch it.

## Starting reference and launch boundary

At original startup the pinned reference was control B,
`outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat`,
seed 2263738815, fold 0, without the pooled meniscus adapter. The draft initially
pointed to the active `compartment-control` run, which actually used the shared
adapter. That run completed and its implementation was rejected before campaign
startup. It remains external evidence, not the campaign incumbent. See
[the run record](prior-work.md#active-run-and-launch-readiness).

The recipe reads that run's saved `effective_config.json`, which the existing
configuration loader accepts. This avoids inheriting later edits to the user's
`configs/training.yaml`. At initialization the runner writes the resolved recipe
into the isolated baseline's `configs/training.yaml`; candidate configuration
edits remain confined to that isolated file. New candidates now inherit the live
incumbent's recorded seed under the current protocol; the original fresh-seed
policy is superseded.

The reference has completed timing, selected-checkpoint metadata and final
validation predictions. Run campaign validation before starting. Preserve
the recorded source and data identities. Baseline sealing must match the run's
input identity exactly; a later main-tree model/preprocessing change is a real
compatibility issue, not permission to rewrite hashes. The descriptive YAML status
does not itself enforce a wait; missing reference artifacts fail runtime checks.

At startup, review the rejected adapter result and any subsequent external
work before admitting proposals. Record actual modes, seeds, sources,
metrics and evidence paths. Do not repeat an arm already completed or active
outside the campaign. External evidence does not automatically receive a campaign
acceptance decision or become a new incumbent. Control B is an explicit starting
reference, not a claimed campaign gain. Do not restore the rejected adapter or its
withdrawn independent arm; a future revisit needs a materially distinct mechanism.

## Fixed recipe and resources

- Use fold 0 of `all_5fold_v3`, Steven v4 with existing gold overrides, unchanged
  observed masks, report groups, target order, scoring, and checkpoint selection.
- Preserve the pinned ReXNet-200 pretrained initialization policy, 224-pixel inputs,
  spacing, sampling, reflection and other augmentation distributions, batch size,
  class-balanced soft BCE, optimizer, and learning-rate schedule. Auxiliary transfer
  losses may supplement main BCE without changing its targets or weights.
- Keep ten epochs and the reference's completed 4,380 optimizer steps. Seal exact
  cohort membership, study visits, and realized valid triplet
  counts through the common full-schedule baseline replay, with zero tolerance.
  Padding is not a real selection. Preserve stochastic training and deterministic,
  unaugmented validation at image multiplier 1, independent of the training seed.
- The runtime ceiling is `ceil(1.15 * reference_total_seconds)`, using the reference's
  recorded setup, training and final prediction times: 1,562.282713 seconds gives
  a 1,797-second ceiling. These are this reference's artifacts, not an inherited
  limit from another campaign.
  Preparation, auxiliary computation, training, prediction and CPU comparison count
  toward it. Baseline replay, agent work and queue waits follow the common protocol.
  An accepted model never raises the original ceiling.
- All learned routing, transfer targets, statistics, feature alignment and adaptive
  supervision use training data only. No hidden precomputation, additional encoder
  passes over the dataset, extra updates, fitted external teachers, or validation
  tuning. Additional computation inside the ordinary forward/training step must fit
  the sealed budget. Both disease labels may be positive independently.
- Every new candidate inherits the incumbent's recorded training seed. Do not draw
  different seeds to recover a favorable result. Genuine checkpoint resumes retain their seed.

## Implementation and review

The [policy](campaign.yaml) allows transfer architecture, auxiliary-objective code,
and necessary anatomical metadata plumbing. Permission to edit a shared file does
not authorize changes to its unrelated behavior. Keep public prediction outputs
as the same twelve logits and preserve the existing training/prediction entrypoints.
Auxiliary heads may be training-only; checkpoint construction must still reproduce
the recorded configuration without compatibility shims.

`model.py` and spatial/data helpers participate in the preprocessing source hash.
The comparison therefore allows `preprocessing_identity` differences, while all
other input identities remain fixed. Inspect the actual member changes: this
exception does not authorize different cohorts, labels, sampling, decoding or image
budgets. Preserve source manifests for new helper modules as well; the existing
preprocessing hash is not a claim to hash every helper transitively.

Configuration-only candidates require an explicit mechanism or ablation within
the declared policy. Their source must still be frozen, reviewed and measured.
The rejected adapter modes are not retained campaign options. A sweep of widths
or coefficients is not a new mechanism. The
developer implements the assigned candidate; the orchestrator owns research
ranking, scope review and interpretation. No tests; agents do not launch GPU work.

## Evaluation and interpretation

Normally train one full-budget model per idea, pinning its comparison incumbent
before launch. Use saved predictions on the same studies, 2,000 paired report-group
bootstrap draws, the frozen invalid-resample rule and separate bootstrap seed,
and a 95% interval for candidate-minus-reference twelve-label macro AUC.

| Result after contract checks | Decision |
|---|---|
| Valid macro AUC gain >= 0.001 and 80% interval lower bound > 0 | Adopt as the working incumbent (`provisional_accept`) |
| Valid result failing either adoption condition | Discard this implementation (`reject`); retain the incumbent |
| Missing evidence or contract breach | Invalid; no acceptance |

The fixed macro rule governs practical adoption. Interpret the proposed transfer
using both meniscus targets, both medial/lateral OA targets, family means, and the
other eight targets; retain the evaluator's compartment margins, co-occurrence
and manufacturer diagnostics with their supports. Do not fit on those groups or
choose a favorable subgroup after seeing results. An unrelated-label gain can be
a macro improvement without establishing the proposed transfer mechanism.

Separate shared morphology from additional capacity, additional task weight, and
co-occurrence. Use matched independent-module or independent-auxiliary controls
when needed for attribution. The older twelve-label residual-readout run is useful
prior evidence, not a matched control that rules out capacity as an explanation.
Diagnostic controls may be worthwhile even when they are not plausible winners;
their verdict still follows the same primary rule.

Bootstrap uncertainty is conditional on the fitted models and excludes seed
variation and repeated fold-0 selection. Acceptance remains provisional. Do not
automatically rerun discarded ideas, extrapolate from early epochs, or declare
the broader hypothesis refuted by one failed implementation. Revisit historical
mechanisms only with new evidence or a materially different design.
