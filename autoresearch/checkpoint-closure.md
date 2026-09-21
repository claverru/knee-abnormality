# Learned-strength checkpoint closure

Completed by explicit user instruction on 2026-09-21. The working recipe and all
three stopped autoresearch campaigns now use learned anatomical family strengths
with fixed 20 mm width, independent heads and position-rich pooling. The seed is
2263738815. No campaign was started or resumed and no training was launched.

## Adopted reference

The measured run is `outputs/training/anatomical-attention/learned-strength`, fold 0,
selected epoch 10, macro AUC 0.8974429314502609. Against the repaired-data,
matched-seed control (0.8951120901611253), gain is +0.0023308412891356323 and the
80% paired report-group interval is [+0.0001034177, +0.0045179746]. It meets the
[fixed adoption rule](protocol.md#adoption-policy). This is a user-directed baseline
replacement, not a retrospective acceptance against historical B.

The shared frozen development source is:
`outputs/autoresearch/adopted/user-adopt-learned-strength-20260921T115849Z-b19e375d/source`.
Its training config has absolute data/label paths and seed 2263738815. The
[adoption record](../outputs/autoresearch/adopted/user-adopt-learned-strength-20260921T115849Z-b19e375d/adoption.json)
pins source manifest, trained source archive, prediction hash, configuration,
input-identity transition and assessment. The [reviewed source delta](../outputs/autoresearch/adopted/user-adopt-learned-strength-20260921T115849Z-b19e375d/trained-to-adopted.diff)
is limited to config, model and Lightning: accepted behavior is mandatory,
disabled/rejected alternatives are removed. No new training score or numerical
forward-equivalence measurement is attributed to this cleanup.

The trained run, checkpoint identities and saved predictions were not rewritten.
Historical inference must use the trained run's archived sources; future training
uses the cleaned source. The source hash changes because model.py is in the
preprocessing identity. For losses, which normally forbids any preprocessing
change, the supervisor permits only the exact reviewed cleaned identity against
this exact adopted reference, verifies the reference identity hash, and keeps the
transition in the review binding. All other identity differences still fail.
After a new incumbent replaces this external reference, this special comparison
allowance no longer applies. Sampling/transfer retain their existing declared
identity-change permissions; none were broadened.

## Campaign state

| Campaign | Revision | State | Incumbent |
|---|---:|---|---|
| triplet-sampling | 181 | stopped | learned-strength-20260921 |
| losses | 259 | stopped | learned-strength-20260921 |
| medial-lateral-transfer | 351 | stopped | learned-strength-20260921 |

All have `running=null`, `claimed=null`, no pending events and no active jobs.
Original sealed baseline sources, numerical limits, budget references, completed
verdicts and per-attempt comparison references remain unchanged. The original
`reference_run` in campaign definitions still owns startup/budget provenance;
live state owns the current incumbent. Status/results identify this as a manual
reference replacement and exclude it from campaign experimental gain.

Each campaign has `decisions/user-adopt-learned-strength-20260921T115849Z-b19e375d/`
containing `state-before.json`, `pending-events.json` and `decision.json`:
[triplet sampling](../outputs/autoresearch/triplet-sampling/decisions/user-adopt-learned-strength-20260921T115849Z-b19e375d/decision.json),
[losses](../outputs/autoresearch/losses/decisions/user-adopt-learned-strength-20260921T115849Z-b19e375d/decision.json),
[medial/lateral transfer](../outputs/autoresearch/medial-lateral-transfer/decisions/user-adopt-learned-strength-20260921T115849Z-b19e375d/decision.json).

The cancelled extra `family-shared-local-lesion-evidence` run is finalized as
invalid, preserving its cancellation receipt and original B reference. It supplies
no scientific evidence against the mechanism. The unlaunched loss preparation
`shared-group-variance-compression-20260920T181146Z-04a138a1` is archived and cannot
be launched. Its source/report remain evidence; its idea stays queued for fresh
implementation and review if the user resumes losses. Pending old-reference
proposal/review events were archived without silently accepting their proposals.
Dormant idea queues remain proposals, not open implementation branches.

## Implementation and documentation closure

The rejected pooled `compartment_adapter` field, initialization and pooling
branches are removed. Appearance-only pooling and learned-width alternatives were
already removed. The adopted model has two learned positive anatomical strengths
initialized at one; the 20 mm spatial scale is fixed. Unknown side and unusable
geometry contribute zero anatomical bias. Recipe, shapes, state, next-work,
validation, seed and relevant backlog documentation now describe this behavior.
Historical measurements remain in their result documents and source archives.

The user explicitly requested current-state project documentation updates as part
of this closure, extending this task beyond the normal autoresearch-only writing
boundary. Durable campaign evidence remains here. No general backlog investigation
was launched, no historical result was re-scored for acceptance, and no checkpoint
or dataset was modified. This closes the requested items 1–4; no commit was made.


Narrow closure verification passed: all three supervisors are inactive, all
frozen-source manifests match their recorded pins, all original limit seals match,
and all 41 previously decided experiment records are unchanged. Read-only status
reports resolve the new incumbent and zero pending completions. Adapter references
are absent from active source/configs; lint/format and whitespace checks pass.
No tests, GPU work, fresh data replay or training were run during this closure.
