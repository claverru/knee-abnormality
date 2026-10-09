# Campaign findings

## proposals-20260922T225710Z-8b46a9fc

No new experimental evidence in this event. The seed import is triaged into a queue of three. The other four kept seeds were not in this proposals event: oblique-training-only-views, coverage-matched-slot-split, oblique-standard-statistics-alignment, and the fourth listed in prior-work.md. The x4 campaign found a consistent tradeoff: representation and routing repairs recover MCL and lateral meniscus on the split studies but give up ACL. A candidate therefore needs to keep ACL's access to the oblique fluid view, which the fixed ACL gate does by design. The depth-frame idea is the only one that changes geometry rather than weighting. Its within-model re-prediction check, re-predicting with the old acquisition-relative oblique depths, separates reliance on the corrected code from training divergence and should be reported.

All three have no dependencies (proposal_admission satisfied) and are kept seeds under the user's 2026-09-22 decision in prior-work.md. The queue was empty, so this ranking is new.

1. acl-gated-oblique-tokens goes first. Every x4 candidate repaired MCL and the menisci on the split studies but lost ACL there: view bias −0.011, mixture −0.022, combination −0.024. The gate is the only mechanism that keeps the oblique fluid evidence for ACL by construction while removing the dilution for the other labels. It has direct paired support: on k_05 at ×4 on the affected studies, +0.0020 with the gate against −0.0001 without it. It adds no parameters and costs nothing. Its x4 run trained (×1 epoch 0.895663) but was never compared, so it has no rejection on record.
2. oblique-standard-frame-depth tests a different causal hypothesis: the depth codes are in the wrong frame, not the tokens diluting attention. The geometry mismatch is measured: median 35–39° between normals, a median 11 mm origin offset on coronal slots, and about ±23 mm of depth change within one slice. The model's reliance on the 1D code is measured too. Whether miscoded depth causes the MCL and lateral meniscus loss is unproven, and appearance may already compensate.
3. oblique-group-embedding is admitted because the user explicitly kept it as a seed. It has never run alone against an ×4 reference on any source. Its x1 run was rejected (80% interval [−0.0005, +0.0040], split studies +0.0052). At x4, combined with the view bias, it lost ACL (−0.0086 overall), so it comes last. The x1 result is on a different reference, metric multiplier and source, so this is not a duplicate of a decided experiment.

Unresolved concerns:
- acl-gated-oblique-tokens: under series dropout in training, or when cleaning removes the standard series, a split slot may keep only its oblique group. A label's softmax could then be fully masked (NaN) for a study with no other acquisition. Review must check the mask has a fallback or that some unmasked token always exists.
- Neither parameter-free idea reproduces the reference forward pass on split studies at initialization: the gate masks tokens, and the frame fix replaces depth codes. Both add no parameters, so the new-parameters-neutral rule is satisfied, but review should confirm that non-split studies are bit-identical at fixed weights.
- oblique-standard-frame-depth: the anchor series choice (first in the source ranking), reflection handling for the per-token depth, and deterministic anchors at validation must be checked. Cleaning can change which standard series leads the ranking, so the anchor must be taken from the cleaned group.
- oblique-group-embedding repeats a mechanism rejected at ×1 on the older source. Its admission rests on the user keeping it as a seed and on the change of reference and multiplier, not on new evidence of an effect.
- Split-study gains of +0.003 to +0.005 on 103 of 905 studies translate to about +0.0005 macro. As at x4, a mechanism may repair the split studies and still fail the macro rule. Per-label and split intervals are diagnostic only.

Evidence:
- /home/claudio/src/knee-abnormality/autoresearch/campaigns/oblique-clean/prior-work.md (user seed decision; x4 and x1 result tables; the ACL tradeoff)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-clean/imports/ideas-20260922T225710Z-f37b6a62/report.json
- /home/claudio/src/knee-abnormality/autoresearch/campaigns/oblique-clean/seed-ideas.yaml
- outputs/analysis/t/tilted-slot-policies-v2-k05-20260922/results.csv (ACL gate +0.0020 on the affected studies)
- outputs/autoresearch/oblique-x4/jobs/researcher-20260922T142228Z-40754723/scratch/evidence/depth_frame.txt and split_x4.txt
- outputs/autoresearch/oblique/experiments/oblique-group-embedding-20260922T115506Z-f5d616dd (x1 reject)

## proposals-20260922T225710Z-d2e5e18e

- The two admitted budget and scoring ideas share one validation constraint. At x1 a split slot's combined count (standard budget plus oblique budget) fits inside the standard series' capacity: about 28 windows for coronal fluid and 30 for coronal T1/PD, per `oblique_reach.txt`. So both can keep the x1 validation count exact.
- At x4 the combined count for `oblique-training-only-views` would exceed that capacity. Capping it reduces the number of triplets predicted in the split studies. Handing the excess to the oblique group puts oblique tokens back into the scored bags. The candidate has to pick one and record it; this does not change the adoption rule.
- The frozen-model evidence for standard-only scoring is small at x4: +0.0011 to +0.0026 on the subset, with the full fold roughly flat. The expected effect on macro AUC is therefore near the 0.001 threshold.

I admitted two of the three imported proposals. Both have no dependencies, sit inside the contract's scope (zero-sum redistribution within a slot, and which series an evaluation bag scores) and are new mechanisms.

- `oblique-training-only-views` goes second. It is the most direct test of the localized loss: on the split studies at x1, `k_05` beat the oblique reference by +0.0103, mostly MCL and lateral meniscus. On frozen `k_05`, standard-only scoring of the split slots gained +0.0066 on the subset at x1 and +0.0011 to +0.0026 at x4, while the full fold did not move. Training is identical to the reference, so the run separates what the oblique views add to training from what they cost when scored. It costs no extra compute. It stays behind the ACL gate, which keeps the view where the evidence says it helps (+0.0020 on 102 affected studies on `k_05` at x4).
- `coverage-matched-slot-split` goes fourth. It is exact and zero-sum within each slot, and it tests the one budget direction nobody has tried (more coverage for the full-knee standard series). But budget changes alone have not moved fold 0 so far, so it ranks below the depth-frame representation idea.
- `oblique-group-embedding` drops to last, making five in the queue. Alone it was rejected at x1, and combined with the view bias it was rejected at x4 (−0.0006, with ACL −0.0086 and lateral meniscus −0.0080). It stays in the queue as the only test of the identity mechanism alone at x4 on the cleaned source.

`oblique-standard-statistics-alignment` stays proposed and is not queued. Its own priority rationale ties its value to whether the embedding repairs the pairs between split and unchanged studies at x4. At x4 the Coronal0 level shift nets out (−0.0001), and zero-initialized gates in this repository have stayed near zero. Taking it would push the queue past five. Reconsider it after the embedding result.

Unresolved concerns:
- oblique-training-only-views trains on oblique tokens it never scores, so training and scoring see different inputs. At x4 the capacity cap either reduces the split studies' predicted triplets or brings oblique tokens back; the choice must be recorded, along with the changed deterministic validation selection identity.
- coverage-matched-slot-split: when the combined count exceeds total capacity, rounding has to keep per-slot totals equal to the reference, or any shortfall must be accounted for explicitly. The CPU replay must match the sealed x1 validation count (24,770) and training triplet counts at zero tolerance.
- oblique-standard-statistics-alignment is retained unqueued. Evidence it should work is weak: the Coronal0 level shift cancels at x4 (net −0.0001), and zero-initialized FiLM gates stayed near zero before.

Evidence:
- /home/claudio/src/knee-abnormality/autoresearch/campaigns/oblique-clean/prior-work.md (x1 and x4 campaign results, inference-only slot policies, oblique_reach coverage)
- /home/claudio/src/knee-abnormality/outputs/analysis/t/tilted-slot-policies-check-20260922/results.csv (k_05 x1 standard-only: subset +0.0066, full −0.00007)
- /home/claudio/src/knee-abnormality/outputs/analysis/a/acquisition-geometry-20260922/oblique_reach.txt (per-group window capacity)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-clean/imports/ideas-20260922T225710Z-34c81e6d/report.json

## review-20260922T231531Z-bdeb5404

Review of acl-gated-oblique-tokens-20260922T230650Z-8ff84619: the diff matches the declared add-only integration. The hard (label, token) mask is parameter-free and only active when model.oblique is set, so k05.yaml behavior is unchanged. Sampling, budgets and validation selection are unchanged, so the sealed counts should replay exactly. Accounting was not available at review time (accounting_available=false), so the count check rests on the supervisor's gate. Mechanism expectation: non-ACL labels (notably MCL and lateral meniscus, where the oblique budget cost the most) will stop seeing fluid-sensitive oblique tokens. On the split studies they then rely only on the standard series and the other slots, as they did before oblique groups existed. Interpret the result on split versus non-split studies against both oblique-clean and k05-clean, as the contract requires.

The candidate implements its topic as a fixed prior on how attention treats oblique tokens, which the contract puts in scope. Tokens from fluid-sensitive oblique partitions of split slots get -inf attention scores for every label except ACL. Non-fluid tilted coronals are left ungated, which the idea allowed as an open choice. Validation independence holds: the flag depends only on deterministic series geometry and slot kind, not on labels, predictions or the seed. The ×1 triplet selection is unchanged, and the extra cost is negligible (one boolean per series and one masked_fill). There is no leakage path. There are no new parameters, so the neutral-initialization clause is met in the letter. However, the mask changes the forward pass from step 0, so an untrained candidate does not reproduce the reference. I read the clause as applying to learned parameters, and a hard gate is exactly what this idea specifies.

NaN safety: series dropout keys on (study, type), so a split slot's standard and oblique series are always kept or dropped together. The standard series is never gated, so for non-ACL labels each study keeps unmasked member tokens unless every token in the standard series is visually invalid. That case is implausible after near-empty-frame cleaning, but no code guards against it.

Unresolved concerns:
- The fixed -inf mask changes the forward pass at initialization. I judged this compliant because no new parameter exists, but the untrained candidate does not reproduce the reference forward pass.
- NaN risk for non-ACL labels if a study's only visually valid tokens came from gated oblique series. Nothing guards against it; I judged it implausible because a split slot's standard series is kept or dropped together with its oblique series.
- series_oblique_gated is missing from inference/engine.py MODEL_BATCH_KEYS; it relies on the model's explicit .to(device), which adds a small synchronous host-to-device copy.
- Accounting was unavailable at review, so the triplet and visit invariance rests on sampling.py being unchanged plus the supervisor's replay gate.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-clean/experiments/acl-gated-oblique-tokens-20260922T230650Z-8ff84619/changes.diff
- candidate source src/knee_abnormality/modeling/model.py:234-262 (_attention_pool mask before the membership softmax)
- candidate source src/knee_abnormality/modeling/model.py:184-200 (_effective_presence drops by type, so standard and oblique series are kept or dropped together)
- candidate source src/knee_abnormality/data/dataset.py:300-326 (oblique partition flagged only when kind.fluid_sensitive)
- candidate source src/knee_abnormality/inference/engine.py:104-121 (MODEL_BATCH_KEYS omits series_oblique_gated)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-clean/jobs/developer-20260922T230650Z-01f9daa0/report.json

## completion-20260922T231642Z-30e35694

acl-gated-oblique-tokens-20260922T230650Z-8ff84619 is invalid because of an implementation bug, not the mechanism. The (label, token) -inf mask in `_attention_pool` received one gate flag per acquisition (162) where it needed one per spatial token (7938 = 162 × 49). `pool_encoded` flattens `study`, `visual_valid` and `medial_axis` to token resolution but passed `oblique_gated` as it was. The first training batch raised a RuntimeError, and the run produced no accounting, checkpoint or predictions. It gives no information about ACL, MCL or lateral meniscus on the split studies. The fix is one line at model.py:391: expand the flag the same way `study` is expanded at model.py:385.

The run is invalid because of a bug in the candidate's code, not because of anything the mechanism did. Training stopped at epoch 0, step 0, with exit code 2 after 14 s: "The size of tensor a (162) must match the size of tensor b (7938) at non-singleton dimension 1". `pool_encoded` (model.py:385) expands `study` to one entry per spatial token with `[:, None].expand_as(visual_valid).flatten()`. It passes `encoded["oblique_gated"]` (model.py:391) without that expansion. The flag therefore has one entry per acquisition (162), while `_attention_pool` (model.py:252) has one score per token (162 × 49 = 7938). Nothing trained, so there is no accounting, metric or evidence about the ACL gate.

The queue stays as it is (ranked_ids null). The previous report was rejected because it put `acl-gated-oblique-tokens` back into the ranking, but that idea is not in the live queue: its status is launched and its experiment is awaiting this decision. I am not requesting a queue change here. The mechanism is still untested, and its prior support is unchanged. A retry of this idea should change one line: expand the flag the same way `study` is expanded, `encoded["oblique_gated"][:, None].expand_as(encoded["visual_valid"]).flatten()`. The rest of the add-only diff looked correct at review.

`oblique-standard-statistics-alignment` stays proposed. Its admission gate passes, but its value depends on the representation results, which have not been run yet, and the live queue already holds four untested ideas.

Unresolved concerns:
- The approving review missed a shape error that fails on the first forward pass. Reviews of per-token masks or biases should trace each per-acquisition field through `encode_batch` and `pool_encoded` to the flattened token grid.
- The NaN risk is still unchecked. If a split slot's standard series ever had no visually valid tokens while its oblique series did, a non-ACL label's softmax in that study would be all -inf.
- `series_oblique_gated` is still missing from `MODEL_BATCH_KEYS` in inference/engine.py. The cost is a small synchronous host-to-device copy, which matters only near the 1,888 s ceiling.
- No retry of this mechanism is currently queued, because the idea sits outside the live queue after its invalid launch. Unless the supervisor opens a retry, the one-line fix is not being tested.

Evidence:
- outputs/autoresearch/oblique-clean/jobs/training-20260922T231627Z-d1f61290/stderr.log:13 (tensor size mismatch 162 vs 7938)
- outputs/autoresearch/oblique-clean/experiments/acl-gated-oblique-tokens-20260922T230650Z-8ff84619/source/src/knee_abnormality/modeling/model.py:252,312,365,385,391
- outputs/autoresearch/oblique-clean/experiments/acl-gated-oblique-tokens-20260922T230650Z-8ff84619/changes.diff
- outputs/autoresearch/oblique-clean/jobs/orchestrator-20260922T231531Z-1b42f8e7/report.json (earlier approving review)
- outputs/autoresearch/oblique-clean/jobs/orchestrator-20260922T231642Z-a6d3184d/report.json (rejected earlier completion report, same diagnosis)

## proposals-20260923T045453Z-9fd5c850

The prior for the ACL gate is a subset effect: in the results CSV, row `additive_acl` (`k_05`, ×4 τ 0.75, 102 affected studies) changes whole-cohort macro AUC by -0.000023, with a 95% interval of [-0.00072, +0.00061]. The claimed +0.0020 applies only to the affected subset. The ungated `intact_acl` row is -0.00017 on the full cohort. So a retrained gate must gain far more than the post-hoc replay did to clear +0.001 macro AUC. The campaign-mandated split-study and unchanged-study breakdown will show whether a gain comes from the split studies (the mechanism) or from same-seed training divergence on the unchanged studies.

I'm admitting `acl-gated-oblique-tokens-token-mask` and ranking it second, directly behind the reserved head. It has no dependencies, so the admission gate is satisfied.

It uses the same mechanism as `acl-gated-oblique-tokens`. That idea's only attempt (8ff84619) crashed on the first batch because the gate flag was never expanded from one value per acquisition to one per spatial token (162 flags against 7,938 tokens = 162 × 49). The run produced no evidence either way, and the idea is no longer in the live queue. This proposal carries the fix and adds two requirements: a no-NaN guard (gate a token only if the study keeps at least one valid ungated token for that label) and a CPU shape check through `encode_batch`, `pool_encoded` and the loss.

The idea stays second because it is the only oblique idea with a prior pointing the same way. Its gate adds no parameters or compute, and it leaves sampling and triplet counts unchanged.

The prior's strength is overstated in the proposal, though. In the analysis CSV, `additive_acl` on `k_05` at ×4 τ 0.75 changes whole-cohort macro AUC by only -0.00002, with a 95% interval of [-0.0007, +0.0006]. The +0.0020 comes from the 102-study affected subset alone. On the full cohort, even a real subset gain of that size would be diluted to roughly +0.0002, well short of the +0.001 adoption bar. The idea can only be adopted if the gate works much better with the gate present during training than the post-hoc policy replay suggests. That is plausible, since attention can then re-learn, but it is unproven.

`oblique-standard-statistics-alignment` stays proposed, for the reason recorded before.

Unresolved concerns:
- The expected whole-cohort effect from the prior is about 0 (-0.00002 on `k_05` ×4); adoption depends on training with the gate doing much better than the post-hoc replay.
- The NaN guard needs a per-study, per-label check that at least one valid ungated token remains; review must confirm it is enforced for both training and prediction, including when series dropout removes the standard series.
- `series_oblique_gated` is still missing from `MODEL_BATCH_KEYS` in the non-editable inference/engine.py; the model must move the flag to the device itself. Confirm at review that this does not break the CUDA-graph or GPU preparation path.

Evidence:
- outputs/analysis/t/tilted-slot-policies-v2-k05-20260922/results.csv rows `additive_acl` and `intact_acl` (whole-cohort deltas -0.000023 and -0.00017)
- outputs/autoresearch/oblique-clean/jobs/orchestrator-20260922T231642Z-a6d3184d/report.json (invalid-run diagnosis: flag not expanded to 49 spatial tokens)
- outputs/autoresearch/oblique-clean/experiments/acl-gated-oblique-tokens-20260922T230650Z-8ff84619 (diff to reuse)
- outputs/autoresearch/oblique-clean/imports/ideas-20260923T045453Z-0e8e935d/report.json

## review-20260923T045558Z-32f091f5

The candidate is a pure evaluation-routing test. It trains exactly like `oblique-clean`, then scores split slots from standard-series triplets only, at the same ×1 count. Split and non-split studies share identical training, so differences on non-split studies measure only checkpoint-selection noise under the changed ×1 validation. A mechanism claim must rest on the 103 split studies. If the gain there is positive, the oblique views help in training but hurt at readout. That would motivate the readout or gating ideas already in the queue.

This is a within-scope zero-sum change to which series an evaluation bag scores. The contract allows exactly that. Training, the model and its parameters are unchanged. Only validation and ×4 prediction on split slots differ: those triplets now come from the standard series instead of the oblique group. Checkpoint selection at ×1 uses the same routing, so selection and scoring are consistent. Validation labels and predictions are not used. Preparation cost goes down or stays the same, because one fewer group is decoded per split slot.

The developer checked on the fold-0 validation split that the standard-only capacity covers the combined ×1 budget for all 103 split-slot occurrences. Realized totals were 1035 old against 1035 new. That keeps the sealed 24,770 ×1 validation triplets, and the supervisor's zero-tolerance ledger enforces it anyway. The queue is unchanged. The `oblique-standard-statistics-alignment` proposal is not in this event's decision scope, so I left it proposed.

Unresolved concerns:
- The capacity check covered ×1 only. At ×4 prediction, the combined budget of 4×(budget+oblique_budget) on standard sources alone may exceed their capacity for some studies (`select_triplets` caps at min(budget, capacity)). Split slots would then be scored with fewer ×4 triplets than the reference. That is not a sealed limit, but the completion should report any ×4 shortfall on split studies.
- Checkpoint selection now uses a different ×1 validation input on split studies. The selected epoch may differ from the reference's, and that alone can move the non-split studies.
- The flag reads `training_seed is None` as meaning evaluation. Any future training-mode caller that passes `None` would silently drop the oblique groups. That doesn't happen today (`accounting.py:140-144` checks the seed on the training dataset).

Evidence:
- outputs/autoresearch/oblique-clean/experiments/oblique-training-only-views-20260922T231632Z-df169fda/changes.diff
- candidate source src/knee_abnormality/data/dataset.py:298-333 (_prepare_slot branch)
- candidate source src/knee_abnormality/data/preprocessing/sampling.py:453-482 (budget capped at capacity, shortfall metadata)
- outputs/autoresearch/oblique-clean/jobs/developer-20260922T231632Z-1a5464f7/report.json (×1 capacity check: 103 split slots, 1035==1035)
- src/knee_abnormality/autoresearch/accounting.py:140-144 (validation seed/multiplier checks)

## review-20260923T050200Z-fad36c9f

The token-mask retry implements the ACL-only gate correctly. On studies with a fluid-sensitive oblique group, those tokens are removed from all non-ACL label attentions (MCL, menisci and the rest) and stay in ACL. Non-fluid-sensitive oblique groups (for example T1 tilted coronals) are not gated. So a mechanism effect should concentrate in the 103 split studies: ACL may gain, and MCL/lateral meniscus may recover the loss that the oblique budget caused against k_05. The completion should read the split-study per-label deltas against both oblique-clean and k05-clean. It should also check whether the safe_to_gate fallback ever triggered; that count is not logged.

This review covers the corrected retry of the invalid acl-gated-oblique-tokens. I inspected changes.diff and the candidate source directly. The token-resolution expansion and the NaN guard both resolve the predecessor's concerns. Every edited file parses. The mask adds no preprocessing cost beyond one bool per series. The code shows no leakage, no change to triplet counts and no dependence of validation on the seed. The mechanism is a fixed hard prior, not a parameter started at neutral. The contract's neutrality clause governs new parameters, so this is allowed, but the forward pass on split studies differs from the reference from step 0. The mechanism is on-topic (how attention weights oblique tokens), and the diff implements exactly the declared design. The queue is unchanged.

Unresolved concerns:
- `series_oblique_gated` is still missing from MODEL_BATCH_KEYS (inference/engine.py:104). encode_batch calls .to(device) on it, so ×4 prediction does a small synchronous host-to-device copy per batch. It works but adds time; the completion should show the runtime against the 1,888 s ceiling.
- How often the safe_to_gate fallback fires (a study whose only valid tokens are gated oblique ones) is unknown because the candidate doesn't count it. When it fires, those studies silently get ungated attention.
- The gate is a hard prior that is active from step 0, so split-study predictions are not reference-equivalent at initialization. Non-split studies should still differ only through shared training noise (per the contract's 0.01–0.02 per-label swings).

Evidence:
- outputs/autoresearch/oblique-clean/experiments/acl-gated-oblique-tokens-token-mask-20260923T045644Z-0bfce9ce/changes.diff
- candidate source model.py:234-263 (_attention_pool mask, safe_to_gate)
- candidate source model.py:380-396 (pool_encoded token expansion of oblique_gated)
- candidate source dataset.py:197-257 (_prepare_slot_group seeded selection; order-independent)
- candidate source dataset.py:263-296,305-325 (merge and fluid_sensitive flagging)
- candidate source dataset.py:426,465,501 (collate series_oblique_gated)
- inference/engine.py:104-133 (MODEL_BATCH_KEYS omission)
- jobs/orchestrator-20260923T045458Z-887beeb6/report.json (predecessor invalid finding)

## completion-20260923T052426Z-a777e865

For a model trained with oblique groups, removing their triplets at validation and prediction on the 103 split studies costs −0.0014 macro. The standard series gets the whole slot budget instead. Almost all of the loss is in two labels: ACL −0.0067 (95% CI [−0.0138, −0.0010]) and Lateral OA −0.0069 ([−0.0144, −0.0003]). Effusion (−0.0020), Medial OA (−0.0024) and Medial Meniscus (−0.0017) lean negative. MCL (+0.0004) and Lateral Meniscus (+0.0002) are flat, and Contusion is +0.0034 (CI spans 0). So the oblique views carry real readout value for ACL, as expected from ACL-oblique protocols, and also for Lateral OA. This is probably the tilted-coronal site. It also means the earlier MCL and lateral-meniscus deficit against k_05 is not caused by the oblique tokens at readout. Removing them does not recover those labels. The deficit more likely comes from training (budget taken from the standard series, or representation) than from readout contamination.

Implications, all exploratory per-label evidence: the ACL gate keeps oblique tokens for ACL, which is right. It is unlikely to lift MCL or Lateral Meniscus through readout masking alone. Its effect on Lateral OA is a risk, but only if the gated fluid-sensitive groups carry that signal. Training-side ideas (coverage split, standard-frame depth, the group embedding) are the better-motivated routes to the MCL and lateral-meniscus deficit.

Caveat: this measures dropping the oblique views at evaluation from a model trained on them. That is a train–test shift, not a model that never used them.

Completion of oblique-training-only-views. Code has already fixed the verdict as reject: macro 0.902336 vs 0.903705, Δ −0.00137, 80% CI [−0.00228, −0.00047]. I confirmed the escalated concern about checkpoint selection. Both runs picked the same epoch (index 7; ×1 scores 0.89772 vs 0.89873), so training and checkpoint are effectively the same as the reference. Exactly 103 studies moved by more than 0.01 in any label probability. This matches the fold-0 split-study count; I did not match the IDs. The other 802 studies moved by less than 0.001, most by less than 1e-4, which is numerical noise. The GE vendor stratum is identical. So this is a clean readout-only test on the split studies. It is the cleanest mechanism evidence so far in this campaign.

Queue order and membership stay unchanged. The result does not remove the reason to run the ACL-gated token mask, but it shapes how to read it (see finding).

Unresolved concerns:
- Whether split slots were scored with fewer ×4 triplets than the reference (standard sources capped by capacity) is still unverified: the ×4 per-study counts were not logged. Some of the ACL and Lateral OA loss may therefore reflect fewer triplets rather than missing oblique content.
- I identified the 103 changed studies by the size of their prediction change, not by matching the recorded split-study IDs. The contract's separate deltas for split and non-split studies, and the comparison against k05-clean, were not reported by the evaluator.
- Lateral OA's dependence on oblique readout suggests the relevant groups are tilted coronals, which may not be fluid-sensitive. Whether the ACL gate masks them from Lateral OA depends on per-group contrast. Check this when interpreting the gate's completion.

Evidence:
- outputs/autoresearch/oblique-clean/experiments/oblique-training-only-views-20260922T231632Z-df169fda/comparison/comparison.json (runs.selected_epoch, strata vendor=GE identical, prediction_change)
- training/fold_0/selected_checkpoint.json in both runs: best-epoch=07; candidate ×1 0.89772, reference 0.89873
- Paired predictions (outputs/training/oblique-clean/fold_0/validation_predictions.csv vs experiment training/fold_0/validation_predictions.csv): 103 studies with max |Δp| > 0.01; 108 > 1e-3; median change 0
- comparison/per_label.json: ACL −0.0067 [−0.0138, −0.0010]; Lateral OA −0.0069 [−0.0144, −0.0003]; MCL +0.0004; Lateral Meniscus +0.0002
- Timing: 1660 s measured, under the 1888 s ceiling; ×1 validation triplets 24,770 per epoch with a fixed selection hash

## review-20260923T053558Z-b9174c11

The standard-frame depth candidate is correct in the common case. Oblique coronal and axial tokens get a per-token depth n_s·p − o_s. Because the oblique plane is tilted, this depth varies across the token grid. For the standard series' own tokens, the same formula reduces to their existing scalar depth, which makes the conventions consistent. The candidate adds no parameters, so the forward pass is not reference-equivalent at initialization only on split-slot oblique tokens. Non-split studies differ only through shared training noise.

Two gaps remain:
- The override is not gated on the token's own `valid_geometry`, so zeroed or local-path poses get a constant, meaningless depth.
- The anchor source may be a standard series that failed preparation.

Both are fixed by the requested repair. `position_1d` is now materialized per token ([N,K,W]) when the flag is on. That adds memory and a small compute cost. `series_depth_anchor` is not in `MODEL_BATCH_KEYS`, so ×4 prediction copies it from host to device on each batch. Runtime should be checked against 1,888 s. The completion should also report how many oblique series were anchored versus fell back; that count is not logged.

The candidate fits the topic: it represents oblique-group tokens using acquisition geometry, adds no parameters, and changes only `model.oblique.standard_frame_depth`. It leaves sampling and budgets alone. The anchor math is consistent: `geometry.normal` and `dicom._physical_position` use the same row×col normal, and `relative_position_mm` is physical minus the series mid-origin. So n·p − o gives the standard series' relative depth. Reflection is handled the same way as `medial_axis`, since oblique groups are never sagittal. With the flag off, the forward pass matches the parent.

One validity defect needs a bounded repair before launch. The anchored depth replaces the per-triplet depth whenever the anchor is finite. It ignores whether the oblique token's own geometry is valid. In preparation (preparation.py:376), invalid poses are zeroed. For those triplets `patient_xyz` becomes 0, and the depth code becomes the constant −origin rather than a real position. Triplets on the local sampling path (`geometry_valid` false because `physical` is false) or with an inconsistent series geometry get the same meaningless code. The parent already computes `valid_geometry`, which includes finite-map checks; the override should use it too.

Second part of the same repair: the anchor comes from `source_ranking[0]` of the standard group. That source may have failed preparation, so the depth frame could belong to a series the model never sees. Take the anchor from a standard series that was actually prepared, meaning its uid is present in `standard_result[0]`. When none qualifies, fall back to no anchor.

Repair:
1. In model.py, use `has_anchor & valid_geometry[:, None]` as the override condition.
2. In dataset.py, choose the anchor from the first ranked uid among the prepared standard values.

Nothing else should change.

Unresolved concerns:
- The anchored depth overrides depth even when the token's own geometry is invalid. Zeroed pose maps (preparation.py:376) give a depth of −origin, and local-path or inconsistent-geometry series also get the override. The override must be gated on `valid_geometry`.
- The anchor is taken from `source_ranking[0]`, which may have failed preparation. It should come from a prepared standard series.
- Anchored versus fallback oblique series are not counted, so the mechanism's coverage of the 103 split studies is unknown.
- The per-token `position_1d` [N,K,W] and the unlisted `series_depth_anchor` batch key add memory, a per-batch host-to-device copy and some runtime. Runtime must stay under the 1,888 s ceiling.
- No forward pass was exercised; the developer checked shapes by hand only.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-clean/experiments/oblique-standard-frame-depth-20260923T052517Z-a1aa4ba1/changes.diff
- candidate source/src/knee_abnormality/modeling/model.py:325-356 (anchored depth override: has_anchor only, no valid_geometry)
- baseline source/src/knee_abnormality/data/preprocessing/preparation.py:367-386 (invalid poses zeroed; geometry_valid)
- baseline source/src/knee_abnormality/data/dataset.py:149-165 (geometry_valid also requires native_poses_valid, reason None, physical)
- candidate source/src/knee_abnormality/data/dataset.py:264-281,339-340 (_standard_depth_anchor from source_ranking[0])
- baseline source/src/knee_abnormality/data/preprocessing/series.py:249-262 and dicom.py:78-89, geometry.py:146 (same normal/origin convention)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-clean/jobs/developer-20260923T052517Z-bb822f7e/report.json

## review-20260923T053930Z-81b1ca30

The repaired standard-frame depth candidate is ready to launch. Oblique tokens in split slots get the per-token depth n_s·p − o_s under two conditions:
- the token's own geometry is valid;
- a prepared standard series gives a finite anchor.

Every other token keeps the parent's per-triplet sinusoidal depth. Any effect should concentrate in the 103 fold-0 split studies. On the non-split studies, the only differences should be shared training noise.

The completion should read split-study per-label deltas against oblique-clean and k05-clean, especially for MCL, lateral meniscus and ACL.

Coverage is unmeasured: nothing counts anchored versus fallback oblique series, or tokens overridden versus left alone.

The repair does exactly the two bounded fixes and nothing else.

- model.py:346-348 now gates the override on the token's own valid_geometry.
- dataset.py:339-349 takes the anchor from the first ranked standard uid that was actually prepared, and falls back to no anchor when none qualifies.

Checks against the contract:
- **Budgets:** sampling and triplet counts are untouched, because both groups are prepared with the original budgets.
- **Parameters:** none are added, so the only change at initialization is the depth code on split-slot oblique tokens that have a valid anchor.
- **Validation:** it stays deterministic and seed-independent, because the anchor comes from ranking and geometry, not RNG.
- **Labels:** none are used.

The mechanism is in scope: it represents oblique-group tokens using acquisition geometry. The depth-frame conventions were verified in the prior review and are unchanged. Approve.

Unresolved concerns:
- Nothing logs how many oblique series or tokens actually got the anchored depth versus the fallback, so the mechanism's coverage of the split studies is unknown.
- `series_depth_anchor` is not in MODEL_BATCH_KEYS (inference/engine.py:104). The model moves it to the device itself on every batch, and position_1d is now materialized as [N,K,W]. Both add a small cost, and runtime must stay under the 1,888 s ceiling.
- No forward pass was run; shapes (the anchor expanded over max_slices and indexed by flat_indices) were checked only by static reading.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-clean/experiments/oblique-standard-frame-depth-20260923T053719Z-dfaa4316/changes.diff
- candidate source/src/knee_abnormality/modeling/model.py:320-375 (patient_xyz, valid_geometry-gated anchored override, position_1d consumer)
- candidate source/src/knee_abnormality/data/dataset.py:264-281,327-358 (anchor from prepared standard uids)
- candidate source/src/knee_abnormality/data/dataset.py:449,495-498,515 (collate NaN default anchor)
- candidate source/src/knee_abnormality/inference/engine.py:104 (MODEL_BATCH_KEYS lacks series_depth_anchor)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-clean/jobs/orchestrator-20260923T053559Z-cd95760f/report.json (repair request)

## completion-20260923T055302Z-f7231670

A hard gate that restricts fluid-sensitive oblique tokens to ACL attention was rejected on the cleaned source at ×4. Macro AUC moved +0.0011, with an 80% interval of [−0.0007, +0.0031]. Most of the effect is not in the split studies:
- Macro delta is +0.0019 on the 103 split studies and +0.0012 on the 802 others.
- On the split studies, ACL fell by 0.038 (32 positives). Taking the oblique tokens out of the other labels' attention did not help ACL.
- MCL's +0.018 is mostly on the other studies, so it is same-seed divergence.
- The OA labels rose on both subsets (Medial OA +0.009 split / +0.005 others, Lateral OA +0.017 / +0.003). This is also most likely noise.

New evidence on the cleaned source, from the same study split: k05-clean minus oblique-clean is +0.0004 on the split studies (95% [−0.0103, +0.0118]) and −0.0014 on the others. So the ~0.010 split-study cost of the oblique budget measured on the older source at ×1 has not reappeared at ×4 on the cleaned source. Label by label, k05-clean is still better for MCL (+0.028) and lateral meniscus (+0.013) and worse for ACL (−0.038) and effusion (−0.036) on the split studies. That weakens the premise that there is a large split-study loss left to recover. The effects that remain are shifts between labels, not a net gap.

Completion of acl-gated-oblique-tokens-token-mask (0bfce9ce). The fixed decision is reject. Macro AUC is +0.00111 (0.903705 → 0.904816), but the 80% paired-bootstrap interval [−0.00074, +0.00306] includes 0, so the lower-bound condition fails. The run itself is valid:
- accounting matches the sealed counts exactly (960,190 training triplets, 35,020 visits, 24,770 validation triplets per epoch);
- validation selection hashes are constant across epochs;
- the only config difference is model.oblique.gated_labels;
- measured compute is 1,664 s, under the 1,888 s ceiling.

I split fold 0 into the 103 split studies and the 802 others, as the contract requires. The split-study list comes from the localize_campaign.py rule, rerun against oblique-clean and k05-clean. The gain is not localized to the mechanism:
- Macro delta against oblique-clean is +0.0019 on the split studies (95% [−0.0080, +0.0121]) and +0.0012 on the others. That is the same size as the same-seed divergence the contract warns about.
- ACL, the one label allowed to see the gated tokens, fell by 0.038 on the split studies (32 positives), while it rose by 0.003 on the others. That is the opposite of the hypothesis.
- MCL's whole-fold +0.018 comes mostly from the other studies (+0.017, 35 of 46 positives), not the split ones (+0.027 on 11). So it looks like training noise rather than a repair.
- Lateral meniscus fell on both subsets (−0.008 and −0.008), so the gate did not recover it.

The queue is unchanged. oblique-standard-statistics-alignment stays proposed for the reason already recorded; the embedding result it depends on is still pending.

Unresolved concerns:
- Split-study membership comes from the census-tilt rule in localize_campaign.py applied to raw series. It was not re-derived from the cleaned source's actual `oblique_split` partitions, so a few studies may be classified differently (the count is still 103).
- How often the safe_to_gate fallback fired is still unknown, because the candidate never counted it. The gate's actual coverage of the split studies is therefore unmeasured.
- Every subset result comes from one seed on 103 studies with 4 to 78 positives per label. The per-label deltas are exploratory and not adjusted for multiplicity.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-clean/experiments/acl-gated-oblique-tokens-token-mask-20260923T045644Z-0bfce9ce/comparison/comparison.json (macro, per_target, paired_report_group_bootstrap: meniscus −0.0035, oa +0.0051 [+0.0004, +0.0102])
- Split/unchanged analysis: outputs/analysis/a/acquisition-geometry-20260922/localize_campaign.py with the references swapped to outputs/training/oblique-clean and k05-clean and all 12 labels printed (a /tmp copy; the original was not modified)
- Candidate predictions: .../acl-gated-oblique-tokens-token-mask-20260923T045644Z-0bfce9ce/training/fold_0/validation_predictions.csv (best epoch 8)
- Candidate triplet_accounting.json and timing.json: counts match the sealed limits; training 1,598 s plus ×4 prediction 49.9 s
- Prior review: /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-clean/jobs/orchestrator-20260923T050201Z-81f589f9/report.json
- autoresearch/campaigns/oblique-clean/prior-work.md:43 (oblique-clean leads k05-clean by +0.00095 at ×4)

## review-20260923T060119Z-d387d7de

coverage-matched-slot-split, first review: the mechanism is correct and in scope, but it conserves requested budget rather than realized triplets.

select_triplets realizes min(budget, geometry-group capacity) triplets. So whenever the reference fell short in one group of a split slot, redividing budget + oblique_budget adds realized triplets. That would break the zero-tolerance counts: 960,190 training triplets and 24,770 validation triplets.

At ×1 the budgets are small: 6 oblique triplets for the axial fluid slot and 5 for the coronal slots. Shortfalls therefore need very short groups (under about 7 slices). They may be rare, but whether they occur at all is unmeasured, and the fix is cheap.

A repair was requested: split the reference's realized total, with exact-capacity ceilings.

This candidate is in scope: it redistributes triplets between a split slot's standard and oblique groups without adding any. Validation stays deterministic because capacities come from geometry, not the seed. Cost is small: two extra build_pool and geometry_groups calls, only on split slots. Leaving the flag off reproduces the reference. The integration is declared as a pure addition, and that is correct.

It still cannot be approved. The contract requires realized triplet counts to equal the baseline replay exactly, and the candidate only preserves the requested count. In sampling.py, select_triplets realizes count = min(budget, len(groups)), and the baseline triplet_accounting.json counts those realized inputs. The reference realizes min(b_s,c_s)+min(b_o,c_o) per split slot. The candidate splits the full requested b_s+b_o, capped at each group's capacity, so it realizes min(b_s+b_o, c_s+c_o).

So the candidate always realizes at least as many triplets as the reference. It realizes strictly more whenever the reference fell short in one group and the other group had room. In addition, the ceiling of max(capacity,1) gives an empty group 1 triplet it cannot realize. The developer's report ("structurally unaffected") treats requested and realized counts as the same thing. No accounting was run (accounting_available=false).

Required repair, to guarantee equality by construction:
- In _prepare_slot, compute the reference's realized total R = min(budget, c_s) + min(oblique_budget, c_o).
- Split R instead of the combined request, using capacity_split with ceilings equal to the exact capacity, not max(cap,1).
- Guarantee each group gets at least 1 whenever its capacity is at least 1, since select_triplets rejects a budget below 1.
- Keep everything else unchanged.

After this, realized = R whenever R ≤ c_s+c_o, which always holds. Validation at ×1 is conserved the same way.

Unresolved concerns:
- The zero-tolerance realized-triplet gate is violated whenever any reference split-slot group had capacity below its budget; the number of such slots on the training and validation cohorts is unmeasured.
- Capacity comes from build_pool without failed_paths or failed_sources. After a preparation retry, the realized capacity can drop below the value used for the split. The reference is affected the same way, but the split could then move the resulting shortfall to a different group.
- capacity_split's ceiling max(cap,1) plus the floor-1 transfer sends an unrealizable triplet to a zero-capacity group, which shrinks the other group's share.
- The mechanism can amplify multi-series standard groups: capacity counts every standard series' windows, so studies with retakes or duplicates push budget away from the oblique group. That is intended, but the completion should report how the oblique share changed across the 103 split studies.

Evidence:
- experiments/coverage-matched-slot-split-20260923T055443Z-f391f2c8/changes.diff (dataset.py _prepare_slot and _group_capacity; sampling.py capacity_split)
- candidate source, src/knee_abnormality/data/preprocessing/sampling.py select_triplets: count = min(budget, len(groups)); metadata shortfall
- outputs/autoresearch/oblique-clean/baseline/triplet_accounting.json: unit 'valid three-slice inputs', 96,019 triplets per epoch, 24,770 validation triplets
- baseline configs/training.yaml: max_images 18/15, study_triplet_total 26, oblique_tilt_deg 30 (axial fluid) and 20 (both coronal types)
- jobs/developer-20260923T055443Z-e7210648/report.json constraint_evidence (sum-conservation checked only on requested counts)

## review-20260923T060811Z-e5d41e10

coverage-matched-slot-split, repair 1, is approved. Each split slot now splits the reference's realized triplet total between its standard and oblique groups, in proportion to each group's geometry-group capacity. The split uses exact-capacity largest remainder with a floor of 1 for any group with capacity above 0.

The realized total per slot equals the reference's by construction, so the only change is composition: triplets move from the smaller-coverage group to the larger one. Any effect should appear on the 103 fold-0 split studies. The completion should read per-label deltas on those studies against oblique-clean and k05-clean, mainly for MCL, lateral meniscus and ACL. The non-split studies see only shared training divergence.

I'm approving this repair: it does exactly what the first review asked for.

**The total is conserved.** `_prepare_slot` now splits the reference's realized total, R = min(b_s,c_s) + min(b_o,c_o), and no longer splits the requested b_s+b_o. The capacities come from the same `build_pool` and `geometry_groups` inputs that `select_triplets` uses on the first attempt (dataset.py:209-218, sampling.py:483-484).

**Checking capacity_split by hand** (sampling.py:422-445):
- Because R ≤ c_s+c_o, the largest-remainder split never exceeds either capacity. Each raw share, R·c_i/T, is at most c_i, and it rounds up only when strictly below c_i.
- The floor transfer runs only when a group with capacity of at least 1 got 0. The transfer is then safe:
  - both capacities are at least 1 and both budgets are at least 1, so R ≥ 2;
  - the donor therefore keeps at least 1 and stays within its capacity.
- Both groups cannot need the transfer at once.
- The final `np.maximum(...,1)` only touches groups whose capacity is 0. It satisfies `select_triplets`' budget ≥ 1 check without changing the realized count of 0.

So min(a,c_s) + min(b,c_o) = R for every slot. That matches the developer's sweep of 784 exhaustive and 200k random cases. Training triplets, visits and ×1 validation triplets should equal the replay exactly. The supervisor's replay remains the binding gate, and it has not run yet (accounting_available=false).

**Validation stays independent of the training seed.** Capacities depend only on geometry, and budgets were already deterministic. The added cost is two build_pool/geometry_groups calls, and only on split slots.

**Scope:** zero-sum redistribution within a split slot, which the contract puts in scope.

Unresolved concerns:
- After a preparation retry with failed paths or sources, the realized capacity can drop below the first-attempt capacity used for the split. The candidate could then realize a different total than the reference. How often retries occur on these cohorts is unmeasured; the supervisor's zero-tolerance replay will catch any mismatch.
- No one has measured how the oblique share of split-slot triplets actually shifts. Multi-series standard groups (retakes, duplicates) inflate standard capacity and can starve the oblique group. The completion should report per-group triplet counts on the split studies.
- The integration mode is declared 'replace', relative to the rejected prior candidate. Against the sealed parent the change is additive. That mislabel does not change what the candidate does.

Evidence:
- experiments/coverage-matched-slot-split-20260923T060230Z-a9a5397e/changes.diff
- candidate source sampling.py:422-445 (largest_remainder, capacity_split) and :476-484 (select_triplets: budget>=1 check, count=min(budget,len(groups)))
- candidate source dataset.py:200-227 (_prepare_slot_group build_pool/select_triplets inputs) and _prepare_slot realized_total computation
- jobs/developer-20260923T060230Z-83d471d4/report.json constraint_evidence (784 exhaustive + 200k random conservation checks)
- jobs/orchestrator-20260923T060119Z-9d9447ca/report.json (prior repair request)
- outputs/autoresearch/oblique-clean/baseline/triplet_accounting.json (960,190 training triplets, 35,020 visits)

## completion-20260923T062229Z-76bdb6ce

oblique-standard-frame-depth is rejected. It gives the oblique tokens in split slots a per-token depth measured in the frame of their slot's standard series, and it hurt the split studies it targets. On the 103 split studies, macro fell by 0.0047 against oblique-clean and by 0.0051 against k05-clean. On the other 802 it fell by 0.0013 against oblique-clean and rose by 0.0004 against k05-clean.

The split-study loss is led by ACL at −0.037 (32 positives), with Lateral Meniscus −0.012, Effusion −0.012 and Contusion −0.010. MCL gained +0.031 on the split studies (11 positives), but it also gained +0.010 on the studies the mechanism does not touch. So the MCL gain is same-seed training drift, not evidence that the mechanism works.

Combined with the training-only-views result (dropping oblique views at readout costs ACL), the evidence says the oblique views' native depth or position code carries ACL-relevant information. Overwriting it with standard-frame depth, with no learned mixing, degrades ACL. Two routes stay open, both untested:
- keeping the native depth and adding the standard-frame depth as a neutral-initialized additive channel;
- the group embedding already queued.

Runtime was 1,662 s, so the per-token patient_xyz and [N,K,W] position cost fits within the ceiling.

Completion of oblique-standard-frame-depth (repair 1). Code fixed the verdict as reject: macro 0.901980 vs 0.903705, Δ −0.00171, 80% CI [−0.00340, +0.00004]. Accounting is exact: 960,190 training triplets and 35,020 visits; ×1 validation is 24,770 triplets per epoch with a fixed selection hash. Measured compute was 1,662 s, under the 1,888 s ceiling, so the runtime concern did not materialize. The only config difference is model.oblique.standard_frame_depth.

The evaluator reported no split versus non-split breakdown, so I computed it myself. I took the split studies to be the 103 whose predictions moved by more than 0.01 in oblique-training-only-views. My AUCs used labels > 0.5, which shifts the absolute macro slightly from the evaluator's (reference 0.9044 vs 0.9037). Every per-label delta over all studies matches per_label.json.

- **Split studies (103):** macro −0.0047 vs oblique-clean and −0.0051 vs k05-clean. ACL is −0.037 (32 positives), Lateral Meniscus −0.012, Effusion −0.012, Contusion −0.010 and Fracture −0.023 (4 positives). MCL is +0.031 (11 positives) and Medial Meniscus +0.014.
- **Other 802 studies:** macro −0.0013 vs the reference and +0.0004 vs k05-clean. MCL +0.010 and ACL −0.006.

Interpretation: the depth override targets the split studies, and that is where the loss concentrates, about 3.6× the per-study macro loss elsewhere. The loss is mostly ACL. ACL-oblique protocols carry the ACL readout (the training-only-views result), so re-coding their depth in the standard series' frame seems to remove useful position information rather than add alignment. The MCL gain also appears on the non-split studies (+0.010), so it is not specific to the mechanism. It matches the 0.01–0.02 same-seed drift the contract documents. The macro verdict stands, and no follow-up on anchored depth is justified. The queue order and membership do not change.

Unresolved concerns:
- Nothing logged anchored versus fallback coverage, so how many split-study oblique tokens were actually overridden is unknown. The mechanism reading assumes most of them were.
- The split-study IDs are inferred from the training-only-views prediction change (103 studies with max |Δp| > 0.01), not from the recorded split IDs. The evaluator still does not report the split and non-split deltas the contract requires, and it does not compare against k05-clean.
- The split-study label intervals come from few positives (ACL 32, MCL 11, Fracture 4) and are not adjusted for multiplicity. No bootstrap was run on the subsets.

Evidence:
- outputs/autoresearch/oblique-clean/experiments/oblique-standard-frame-depth-20260923T053719Z-dfaa4316/comparison/comparison.json and per_label.json: macro 0.901980 vs 0.903705; ACL −0.0111 [−0.0186, −0.0044]; MCL +0.0122 [−0.0038, +0.0307]; Lateral Meniscus −0.0074
- Subset analysis (labels > 0.5; split = 103 studies with max |Δp| > 0.01 in experiments/oblique-training-only-views-20260922T231632Z-df169fda/training/fold_0/validation_predictions.csv versus outputs/training/oblique-clean): split macro, candidate 0.86159, reference 0.86632, k05-clean 0.86669; rest macro 0.90719, 0.90850, 0.90682
- Same analysis, split-study candidate minus reference: ACL −0.0365, MCL +0.0306, Medial Meniscus +0.0140, Lateral Meniscus −0.0123, Effusion −0.0123, Contusion −0.0103, Fracture −0.0227. On the other 802 studies: MCL +0.0101, ACL −0.0060
- outputs/training/k05-clean/fold_0/validation_predictions.csv, used for the comparison against k05-clean
- Candidate training/fold_0/timing.json and triplet_accounting.json: measured compute 1,662 s; triplet and visit counts exact

## review-20260923T062746Z-786da100

The same mechanism was rejected on the older source in both earlier campaigns:
- oblique: adoption 80% interval [−0.0005, +0.0040];
- oblique-x4: combined with the view bias, −0.000615.

In the first campaign it recovered about +0.005 on the split studies (prior-work.md:135–142). This run tests whether that split-study gain survives series cleaning. At completion, read the macro and per-label deltas on the 103 split studies against both oblique-clean and k05-clean, mainly for MCL, lateral meniscus and ACL. Keep them separate from the shared seed divergence on the non-split studies.

I'm approving it. The candidate fits the topic: it gives oblique-group tokens their own learned representation, using group membership as the contract allows.

It keeps the contract:
- **Starts neutral:** Embedding.from_pretrained(zeros) passes _weight, so no random numbers are drawn and nothing is re-initialised. The added term is exactly zero, so the untrained forward pass matches the reference.
- **Sampling unchanged:** the change touches no budgets, triplet counts or visits.
- **Validation unchanged:** validation uses the same deterministic path and depends on neither the seed nor any label.
- **No leakage:** the flag is geometry-derived metadata that already existed.
- **Negligible cost:** one gather and one add per token.

The flag reaches every prediction path. series_oblique is not in engine.MODEL_BATCH_KEYS, so it stays on the CPU, but encode_batch moves it to the device explicitly (model.py:279). Series dropout acts on whole (study, type) slots, so the standard and oblique groups are dropped together, as in the parent.

The embedding weight is 2-D, so lightning.py:44 puts it in the head-LR group with weight decay. That pulls it gently toward zero and is acceptable.

The config difference is limited to model.oblique.enabled. The seed is the incumbent's.

Unresolved concerns:
- SPATIAL_TOKEN_IDENTITY['algorithm'] changes to the v7 string even when the embedding is disabled, so the spatial-token identity no longer shows whether the embedding was used. The identity is cosmetic here, but the recorded config field is the only reliable marker.
- Oblique flags come from the split in _prepare_slot, and only for slots holding both standard and oblique sources. A slot with only oblique sources goes through the plain path and gets no flag, as intended. The embedding therefore marks split-slot membership, not tilt as such.
- This mechanism has already been rejected twice on older sources, so a gain here is unlikely.

Evidence:
- outputs/autoresearch/oblique-clean/experiments/oblique-group-embedding-20260923T062345Z-3d06eddd/changes.diff
- source/src/knee_abnormality/modeling/spatial_tokens.py:252-292 (zero-init from_pretrained embedding, forward returns 'oblique')
- source/src/knee_abnormality/modeling/model.py:276-296,354-363 (flag gather via keep_series/acquisition_index; add before mil_norm)
- source/src/knee_abnormality/data/dataset.py:304-324,464 (replace(oblique=True) on the oblique group; collate flag)
- source/src/knee_abnormality/inference/engine.py:104-121 (series_oblique not moved; model moves it explicitly)
- source/src/knee_abnormality/training/lightning.py:44 (weight decay applies to the 2-D embedding)
- autoresearch/campaigns/oblique-clean/prior-work.md:113,135-142 (prior rejections and split-study repair)

## completion-20260923T065135Z-3bc15bce

Coverage-matched split repair 1 (rejected): each split slot kept its realized triplet total, and the triplets were reallocated between the standard and oblique groups in proportion to each group's geometry-group capacity. Accounting shows the totals were conserved exactly. The composition change did not help overall: macro AUC fell 0.0008.

- Largest per-label loss: lateral meniscus −0.0103, 95% [−0.0213, −0.0005]. This is the only label whose unadjusted 95% interval excludes 0.
- Also lower: contusion −0.0071, ACL −0.0053, MCL −0.0043, effusion −0.0036.
- Higher: Baker's +0.0065, fracture +0.0073, lateral OA +0.0043, medial OA +0.0028.

The losses fall on ACL, MCL and lateral meniscus, the labels enriched in the split studies. That would fit a cost from moving triplets out of the smaller-capacity group, often the oblique ACL slab. It is not demonstrated, though. The validation predictions cover all 905 studies, but I could not produce the split-only and non-split breakdowns the contract requires. Neither comparison.json nor any other artifact records which fold-0 studies hold a split slot. The deltas against k05-clean are also absent from comparison.json. Per-group triplet counts on the split studies were never recorded either, so how far the oblique share actually moved is still unknown.

Both campaigns that used oblique groups have now lost to k_05. Redistributing budget within split slots at a fixed total does not recover that loss, which leaves the representation ideas as the remaining leads.

This completion confirms the fixed reject for coverage-matched-slot-split repair 1. Macro AUC fell by 0.00081, from 0.903705 to 0.902894. The adoption 80% interval was [−0.00272, +0.00127], so the macro result misses both the +0.001 minimum gain and the positive-lower-bound condition.

The reviewer's first concern was that the candidate might realize a different triplet total than the reference. That did not happen. Accounting matched the sealed counts exactly: 960,190 training triplets, 35,020 visits, 24,770 validation triplets per epoch, and the same training and validation cohort hashes. Runtime was 1,670.8 s against a 1,888 s ceiling.

I admitted oblique-standard-statistics-alignment and ranked it second, behind the reserved head. It has no dependencies and stays in scope, since it changes how oblique-group tokens are represented. It begins neutral because its gate starts at zero. It also targets a measured effect: on the Coronal0 T2-Bondorf split studies, oblique-clean sets OA and medial-meniscus logits 0.27 lower than k_05. A constant embedding cannot follow that shift from study to study, and a per-study statistics alignment can. Nothing else in the queue changed.

Unresolved concerns:
- The contract requires split-study and non-split breakdowns against both oblique-clean and k05-clean, and they are missing. comparison.json has no split-study subgroup and no k05-clean comparison, so the mechanism claim (loss concentrated on split studies) is unverified. The supervisor or evaluator should record the split-slot study IDs and emit subgroup deltas for future completions.
- No one ever recorded per-group standard and oblique triplet counts on split studies. The size of the composition shift is therefore unknown, including whether duplicate or retake standard series drained the oblique budget.
- The candidate's validation selection hash changed on the split studies because validation also uses _prepare_slot. The contract allows this, but I have not compared the hash directly against the reference, because the reference run has no fold_0/triplet_accounting.json.
- The declared integration mode 'replace' is relative to the rejected prior candidate. Against the sealed parent the change is additive. This has no effect now that the idea is rejected.

Evidence:
- outputs/autoresearch/oblique-clean/experiments/coverage-matched-slot-split-20260923T060230Z-a9a5397e/comparison/comparison.json (macro 0.903705→0.902894, selected epoch 8 both)
- outputs/autoresearch/oblique-clean/experiments/coverage-matched-slot-split-20260923T060230Z-a9a5397e/comparison/per_label.json (LM −0.0103 [−0.0213,−0.0005]; ACL −0.0053; MCL −0.0043)
- outputs/autoresearch/oblique-clean/experiments/coverage-matched-slot-split-20260923T060230Z-a9a5397e/changes.diff (capacity_split in sampling.py, realized_total in dataset.py _prepare_slot)
- evidence.json accounting: 960190 training triplets, 35020 visits, 24770 validation triplets per epoch, matching sealed cohort hashes
- outputs/autoresearch/oblique-clean/jobs/orchestrator-20260923T060811Z-587f1a45/report.json (review concerns on retry capacity and per-group counts)
- outputs/autoresearch/oblique-clean/state.json ideas[oblique-standard-statistics-alignment] (per-study AdaIN mechanism, Coronal0 logit shift of 0.27)

## review-20260923T065717Z-d3a29670

This is the first mechanism in this campaign that changes oblique-token appearance statistics rather than adding an identity embedding or bias. It is a per-study AdaIN of oblique-group features toward the same slot's standard-series statistics. Each of the 3 splittable types has its own per-channel gate. The gates start at zero and train on the training batches only. It is not a variant of the oblique-group embedding rejected twice before. At completion, read the split-study macro and per-label deltas, mainly MCL, lateral meniscus and ACL, against both oblique-clean and k05-clean ×4 predictions. Read them apart from the seed divergence on the non-split studies. Also inspect the learned gate magnitudes per type to see whether the mechanism was actually used.

I approve the launch. The candidate is inside the representation scope: oblique tokens are re-normalized toward the statistics of their slot's standard series. The integration is purely additive. The pass-through at zero gate is exact, and sampling and validation accounting are unchanged. I checked the diff and the critical source myself instead of relying on the developer's claims: the collate entries are per acquisition, the keep_series/acquisition_index indexing is right, the usable-group guard handles dropped or one-sided slots, and the statistics are detached and computed per study. Nothing in the validation path depends on labels. The queue is unchanged.

Unresolved concerns:
- The gate is a 2-D parameter, so lightning.py:44 applies weight decay to it and pulls it toward the neutral point. This may weaken the mechanism; it is not a validity issue.
- σs/(σo+eps) uses eps=1e-5. A channel with near-zero variance in the oblique group could amplify it heavily once the gate is nonzero. The number of tokens per group makes this unlikely, but the head's non-finite check would raise if it happened.
- The statistics include every visual_valid spatial position, background included. Differences in field of view between the oblique and standard series can therefore drive the alignment alongside tissue contrast.
- The concurrent oblique-group-embedding run uses the same flag plumbing. If both are ever combined, the flag and config surfaces must be merged rather than duplicated.

Evidence:
- outputs/autoresearch/oblique-clean/experiments/oblique-standard-statistics-alignment-20260923T065236Z-8152a0ce/changes.diff
- candidate source src/knee_abnormality/modeling/model.py `_align_oblique_features` and the encode_batch call after feature_geometry
- candidate source src/knee_abnormality/data/dataset.py:304-323 (tagging on the split branch) and :402-408, :463 (per-acquisition entries, series_oblique)
- candidate source src/knee_abnormality/training/lightning.py:38-48 (the gate gets weight decay)
- configs/training.yaml:22-47 (the 3 types with a non-null oblique_tilt_deg match the [3,256] gate)
- outputs/autoresearch/oblique-clean/jobs/developer-20260923T065236Z-47ba0ea0/report.json
