# Campaign findings

## proposals-20260922T114104Z-f84af8e3

The incumbent source already has the learned anatomical attention term (model.py:126, model.py:232 in _attention_pool). So label-view-attention-bias can add to scores at the same point without new structure.

The oblique group is known at dataset.py:286-303 (oblique_split, then _merge_slot_groups, which already records metadata["oblique_groups"]). The plumbing for a per-acquisition flag has a natural source there.

Recommendation on the embedding's open choice: use one shared zero-initialised vector as the minimal test. It adds to the slot's plane and contrast embeddings, so the type-specific interaction is optional. The coronal ACL obliques that are labelled axial should follow the slot they are sampled in, so that the flag matches the budget grouping.

I admitted all five seeded ideas. Each has no dependencies, according to proposal_admission, and each sits inside the contract's scope: representation, attention/readout weighting, or zero-sum redistribution within a slot. None repeats a rejected mechanism. The retired knee-side and 3D embeddings marked patient position, not acquisition protocol. The rejected residual_readout reused the shared pooled vector. repeated_series_share changed slot totals.

Order:
1. oblique-group-embedding. It is the cheapest and most direct fix for the missing identity, it is neutral at initialization, and its flag plumbing is reused by ideas 2 and 3.
2. label-view-attention-bias. It has the same additive form as the adopted anatomical term, which is already in the incumbent's _attention_pool. It targets the dilution of about 245 oblique tokens among more than a thousand standard ones, and the learned table can be read afterwards.
3. oblique-residual-readout. This is the strongest routing option, but it adds more machinery and has a masking edge case (see concerns).
4. coverage-matched-slot-split. It is exactly zero-sum and cheap. It sits below the representation ideas because changing the budget alone did not move fold 0 in the reference run.
5. view-orientation-embedding. It is a generalisation of idea 1, so it is most informative once we know whether binary identity matters. It also changes every acquisition, not only the split studies, and it carries a reflection-consistency risk.

The formal dependencies stay empty. The later ideas are ranked after idea 1 only for sequencing, and each is still tested alone against the incumbent.

Unresolved concerns:
- oblique-residual-readout: studies with no oblique token run a masked softmax over an empty set. That must give an exact zero residual, not NaN or a uniform average over padding, or non-split studies stop matching the reference at initialisation.
- label-view-attention-bias: whether the bias is added before or after the temperature divide must be fixed once and reported. Validation at x1 uses the training temperature path, so this changes the effective bias scale at inference.
- view-orientation-embedding: under the reflection augmentation the mirrored x component of the normal must follow the reflected tokens. It also touches every study, so the claim that non-split studies see no change does not hold. Review must confirm that the initial forward pass is identical to the reference.
- coverage-matched-slot-split: the validation selection for split studies changes, so its deterministic identity must be recorded. Per-slot, per-study and run triplet totals must match the sealed replay at zero tolerance, including slots whose combined count exceeds capacity.
- All candidates: same-seed training drift of 0.01–0.02 per label on the 802 unsplit studies means any mechanism claim must come from the split-study deltas (103 studies). That subset is small, so its label intervals will be wide.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/imports/ideas-20260922T114104Z-341ecb71/report.json (seeded proposals)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/baseline/source/src/knee_abnormality/modeling/model.py:126,220-232 (anatomical log-strength bias in _attention_pool)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/baseline/source/src/knee_abnormality/data/dataset.py:248-303 (oblique_split, oblique budget, oblique_groups metadata)
- docs/validation/results/sampling/2026-09-22-oblique-groups-result.md (reference 0.897527 vs k_05 0.899042)
- autoresearch/campaigns/oblique/prior-work.md (coverage 85–93%, label profile)

## proposals-20260922T114104Z-5008f358

The imported proposal overlaps an excluded direction (more samples from the same oblique series) and a publicly failed family (reallocating a fixed total into repeated slots). It affects 253 training studies (47% of split studies) and moves budget away from a type whose full removal hurt MCL and fracture. It stays proposed without queueing. It becomes relevant only if coverage-matched-slot-split shows that the within-slot budget balance matters in the studies that hold a split slot.

No admission this event. The queue stays unchanged; it is already at the five-idea target. axial-t1pd-to-oblique-transfer has no dependencies and passes the admission gate. It is in scope, because the contract allows zero-sum redistribution between an oblique group and the study's other types. However, it ranks below every queued idea for three reasons:
(1) It mainly adds per-study triplets from the same oblique series. The contract excludes denser sampling of these series as a lead, because over ten epochs the groups already see 85–93% of their windows (prior-work.md, sampling coverage). The only new element is more oblique evidence inside each bag, paid for by axial T1/PD.
(2) Its value case rests on frozen-model estimates of axial T1/PD. Those same estimates showed that removing axial T1/PD entirely cost about 0.008 AUC for MCL and fracture on the confirm half. MCL is one of the labels the oblique studies carry at high rates, so taking budget from axial T1/PD works against the campaign's target labels.
(3) Its closest precedent, repeated_series_share, reallocated a fixed total into repeated slots. It topped fold 0 but lost publicly (0.929 vs 0.933), so a fold-0 gain from this kind of change is weak evidence.
It also changes the validation selection, which confounds the representation effect with test-time allocation. On frozen models, tilted-only evaluation lost, and averaging standard-only and tilted-only passes gained only +0.0002 to +0.0006. The queued coverage-matched-slot-split already tests zero-sum budget within split slots without touching other types. This proposal should be reconsidered only after that result, and only if more per-study oblique evidence helps. It stays proposed; it is not parked.

Unresolved concerns:
- If admitted later, the validation selection identity changes. A macro gain would mix a training effect with a test-time allocation effect. Frozen-model evidence says such allocation effects are small (+0.0002 to +0.0006) or negative (tilted-only).
- The transfer size is still open (for example, half of the axial T1/PD budget) and must be fixed before training. A mid-run adjustment toward capacity would be tuning on fold 0.

Evidence:
- autoresearch/campaigns/oblique/prior-work.md:53-56 (repeated_series_share topped fold 0 and lost publicly, 0.929 vs 0.933)
- autoresearch/campaigns/oblique/prior-work.md:60-68 (oblique groups see 85–93% of their windows over ten epochs)
- autoresearch/campaigns/oblique/contract.md (objective: denser sampling of the same oblique series is not a lead)
- docs/validation/results/sampling/2026-09-17-test-time-allocation.md (removing axial T1/PD cost about 0.008 for MCL and fracture on the confirm half)
- outputs/autoresearch/oblique/imports/ideas-20260922T114104Z-5c988b15/report.json

## review-20260922T115415Z-0dbd5d60

The oblique-group embedding candidate propagates the per-acquisition oblique flag correctly and leaves slot budgets, sampling order and the validation path unchanged. It was not approved because of two defects. First, the zero-initialized nn.Embedding is created after the plane and contrast embeddings but before their normal_ init, so it uses CPU RNG that the reference uses for them. That breaks exact neutrality at init and shifts later CPU RNG use. Second, series_oblique is missing from the inference engine's device-transfer allowlist, and the model does not move it to the device itself. Both fixes are local to spatial_tokens.py and model.py. No accounting was available for this review (accounting_available=false), so the unchanged triplet and visit counts have not yet been confirmed by replay.

The mechanism is on topic and the new data path is sound. The two _prepare_slot_group calls still run in the same order (standard, then oblique), and Python evaluated the old arguments left to right too, so sampling and RNG order are unchanged. dataclasses.replace only sets the new field, and the budgets and _merge_slot_groups accounting are unchanged. The flag comes from the fixed oblique_split geometry, so it is deterministic and does not depend on the seed or the labels. Validation is unaffected.

Two defects need fixing before launch:

1. Neutral start is not exact (contract: "an untrained change reproduces the reference forward pass"). spatial_tokens.py:260 builds nn.Embedding(2, 256) and then zeroes it. Building it runs the default normal_ init, which uses 512 CPU RNG draws. This happens before the plane and contrast normal_ init at spatial_tokens.py:280-281, so those weights differ from the reference at init, and every later CPU RNG draw after model construction (for example DataLoader base seeds) is shifted. The candidate would then differ from the same-seed reference by more than the mechanism.

2. The inference engine breaks. model.py reads batch["series_oblique"] and indexes it with keep_series. inference/engine.py:113-118 moves only MODEL_BATCH_KEYS to the device, and series_oblique is not in that set. On the engine path (engine.py:169 and scripts that use move_batch), a CUDA boolean mask indexes a CPU tensor and raises an error, so the trained checkpoint cannot run through standard inference. Lightning moves every tensor, so training and validation would not show this.

Repair, as one bounded change:
- (a) Create the zero parameter without using the RNG, e.g. nn.Embedding(2, width, _weight=torch.zeros(2, width)) or nn.Embedding.from_pretrained(torch.zeros(2, width), freeze=False). Then, with the flag enabled, the parameter init and data seeds match the reference.
- (b) In model.py, move the flag to the index device: batch["series_oblique"].to(all_study.device). engine.py is outside the editable files.

Everything else stays as it is.

Unresolved concerns:
- spatial_tokens.py:260-261: nn.Embedding default init draws from the CPU RNG before the plane/contrast normal_ at :280-281, so the init differs from the same-seed reference; the contract requires reproducing the reference forward pass at init.
- model.py encode_batch indexes a CPU series_oblique with a CUDA keep_series under inference/engine.py move_batch, which moves only MODEL_BATCH_KEYS; standard inference on the trained checkpoint fails.
- The accounting replay was unavailable at review; the triplet and visit counts still need the supervisor's zero-tolerance check.
- The config key is model.oblique.enabled with default False. Under the no-legacy-options rule, the option should be removed if the idea is rejected, or made unconditional if it is adopted.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/experiments/oblique-group-embedding-20260922T115015Z-8ee94e16/changes.diff
- candidate source src/knee_abnormality/modeling/spatial_tokens.py:252-293 (embedding construction before plane/contrast normal_ init)
- candidate source src/knee_abnormality/modeling/model.py:276-296 (series_oblique indexed by keep_series)
- candidate source src/knee_abnormality/inference/engine.py:95-118,169 (MODEL_BATCH_KEYS allowlist lacks series_oblique)
- candidate source src/knee_abnormality/data/dataset.py:248-310 (merge accounting unchanged; call order preserved)
- candidate source src/knee_abnormality/data/dataset.py:412-457 (collate emits series_oblique)

## review-20260922T115806Z-f2e83351

The repaired oblique-group-embedding candidate is exactly neutral at init: it uses no RNG, and the zero embedding leaves the reference token sum unchanged. It is also inference-safe, and it leaves sampling, budgets and validation untouched. One property matters for interpreting the result: row 0 of the 2-row embedding, used for non-oblique tokens, is also trainable. So the candidate adds a learned global token offset (redundant with plane/contrast) as well as the oblique marker. Only the difference between rows 1 and 0 carries group information. Per the contract, mechanism claims should rest on the per-label deltas for studies with a split slot (103 on fold 0); the non-split studies still diverge through training.

Both repair items are fixed and verified in source.

(a) `nn.Embedding.from_pretrained(torch.zeros(2, width), freeze=False)` passes `_weight`, so `reset_parameters` is skipped and no CPU RNG is used. The plane/contrast `normal_` draws at spatial_tokens.py:281-282 and all later RNG use match the reference. With zero weights the token sum is unchanged, so the forward pass at init reproduces the reference exactly.

(b) model.py:279 now moves `series_oblique` to the index device. `engine.move_batch` passes keys outside `MODEL_BATCH_KEYS` through unchanged as CPU tensors (engine.py:111-117), so standard inference works.

Other checks:
- The prepared-series cache is not mutated, because `replace` returns new objects, and the standard/oblique partition means no series gets both flags.
- The flag is derived from fixed geometry and applies identically to validation: deterministic, seed-independent, with no label use.
- The embedding weight lands in the `head_lr`/weight-decay group like plane/contrast (lightning.py:35-44).
- Preparation cost is negligible: 0.17 s.
- The `SPATIAL_TOKEN_IDENTITY` bump changes the checkpoint `preprocessing_identity`, as intended, while the sealed data identity stays the same.

The edits stay inside the editable files and the `model.oblique` config prefix.

Unresolved concerns:
- accounting_available=false: the triplet, visit and validation counts still rely on the supervisor's zero-tolerance replay. The diff does not touch budgets, but this has not been verified by replay.
- The trainable row 0 (non-oblique) gives every standard token a learned shared bias. This is a small confound: gains cannot be attributed purely to oblique marking without checking the split-slot versus non-split per-label deltas.
- `model.oblique.enabled` keeps a default-False option. Under the no-legacy rule it must be removed if the idea is discarded, or made unconditional if it is adopted.
- 103 split studies on fold 0 give low power for a mechanism-specific signal; per-label split-subset intervals will be wide.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/experiments/oblique-group-embedding-20260922T115506Z-f5d616dd/changes.diff
- candidate src/knee_abnormality/modeling/spatial_tokens.py:252-294 (from_pretrained zero embedding before plane/contrast normal_; forward adds oblique)
- candidate src/knee_abnormality/modeling/model.py:276-362 (series_oblique .to(device), token_oblique, conditional sum before flatten/mil_norm)
- candidate src/knee_abnormality/inference/engine.py:93-118 (move_batch passes non-allowlisted keys through on CPU; fixed by model-side .to)
- candidate src/knee_abnormality/data/dataset.py:289-310,412-457 (call order preserved, replace on oblique values only, collate flag)
- candidate src/knee_abnormality/training/lightning.py:35-51 (embedding in head_lr/weight-decay group)
- candidate src/knee_abnormality/identity.py:34-60 (SPATIAL_TOKEN_IDENTITY feeds checkpoint preprocessing_identity)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/jobs/orchestrator-20260922T115415Z-e14f5b60/report.json (prior repair request)

## review-20260922T120734Z-da2e1fe7

The `label-view-attention-bias` candidate implements the per-label view-group attention prior correctly. At initialisation it is neutral: it uses no RNG and the zero bias leaves the reference scores unchanged. It leaves sampling, budgets and validation selection untouched. It has one blocking defect: the final `predict_loader` pass leaves `series_oblique` on the CPU while the other indices are on CUDA, so the run would crash after training while writing `validation_predictions.csv`. `view_group_bias` has 2 dimensions, so it lands in the head_lr group with weight decay (`lightning.py:37-44`). This pulls it toward zero, a mild prior against the mechanism rather than a correctness problem. Only the difference between each oblique column and its slot's standard column is oblique-specific: the 6 standard columns also add a new per-label slot prior for every study, which confounds attribution. Mechanism claims must rest on the per-label deltas for studies that hold a split slot (103 on fold 0).

The mechanism, its plumbing and the additive integration are correct, but the run would crash at the step that writes the validation predictions used for evaluation. `scripts/pipeline/train.py:292` produces `validation_predictions.csv` with `predict_loader`. Its `move_batch` (`inference/engine.py:95-119`) moves only keys listed in `MODEL_BATCH_KEYS`, and `series_oblique` is not listed, so the tensor stays on the CPU. In `model.py` `encode_batch`, `batch["series_oblique"][keep_series][acquisition_index]` then indexes a CPU tensor with CUDA masks and indices, and `torch.where` would combine a CPU condition with CUDA tensors. Both fail on the GPU. The developer's report says `engine.py` is used only for leaderboard inference; `train.py:292` shows it also produces the local validation predictions. Lightning's per-epoch validation is unaffected because `transfer_batch_to_device` moves the whole dict. The sibling `oblique-group-embedding` repair fixed the same bug with `.to(all_study.device)`.

Repair (one change, `model.py` only; `engine.py` is not editable): move `series_oblique` to the device before indexing, e.g. `batch["series_oblique"].to(all_study.device)[keep_series][acquisition_index]`. Nothing else needs to change.

The rest of the review passed:
- The oblique flag is attached after `_prepare_slot_group`, with the same arguments, order and budget, so sampling and accounting are unchanged.
- Series dropout filters `series_oblique` through the same `keep_series` mask.
- The 9 view groups are 6 slots plus 3 oblique columns, for axial fluid (30°) and both coronals (20°), matching `training.yaml:21-46`.
- The bias is zero-initialised and consumes no RNG, so the forward pass at initialisation is identical to the reference.
- The bias is added in fp32 after the anatomical term, before the membership mask.
- Validation stays deterministic and does not depend on the seed. Nothing is removed and no active ideas exist, so an add with an empty `removed_idea_ids` is right.

Unresolved concerns:
- Blocking: `series_oblique` is missing from `engine.py` `MODEL_BATCH_KEYS`, so `train.py:292` `predict_loader` would fail on CUDA in `model.py` `encode_batch`. Fix in `model.py` by moving the tensor to `all_study.device`.
- The 6 standard-slot columns give every study a learned per-label slot attention prior, so any gain cannot be attributed to oblique marking alone. Check split-slot versus non-split per-label deltas.
- `view_group_bias` is decayed toward zero by weight decay, which may weaken the learned view preferences.
- `accounting_available=false`: the unchanged triplet and visit counts still depend on the supervisor's zero-tolerance replay.
- `model.oblique.attention_bias` defaults to False. Under the no-legacy rule, remove it if the idea is discarded, or make it unconditional if it is adopted.
- 103 split studies on fold 0 give low power; per-label intervals on that subset will be wide.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/experiments/label-view-attention-bias-20260922T115857Z-576628b9/changes.diff
- candidate source/scripts/pipeline/train.py:292 (predict_loader writes validation_predictions.csv)
- candidate source/src/knee_abnormality/inference/engine.py:95-119 (MODEL_BATCH_KEYS allowlist omits series_oblique)
- candidate source/src/knee_abnormality/modeling/model.py:318-323 (CPU series_oblique indexed by CUDA keep_series/acquisition_index)
- oblique-group-embedding-20260922T115506Z-f5d616dd/source/src/knee_abnormality/modeling/model.py:279 (sibling's `.to(all_study.device)` fix for the same bug)
- candidate source/src/knee_abnormality/training/lightning.py:37-44 (the 2-D bias gets head_lr and weight decay)
- candidate source/src/knee_abnormality/data/dataset.py:286-308 (flag attached after unchanged group preparation)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/jobs/developer-20260922T115857Z-b83ed56a/report.json (developer claims engine is only for leaderboard inference)

## review-20260922T121015Z-7388cb00

The repaired label-view-attention-bias candidate is ready to launch. The CUDA/CPU crash in the final predict_loader pass is fixed in model.py without touching the non-editable engine.py, and no other behaviour changed.

Attribution is still confounded. The 6 standard-slot columns add a per-label slot attention prior for every study. Only the difference between each oblique column and its slot's standard column is specific to oblique series. Any mechanism claim must therefore rest on the per-label deltas for studies that hold a split slot (103 on fold 0).

The one repair requested in orchestrator-20260922T120734Z-2474c8d3 is applied exactly. A diff against the previous candidate's changes.diff shows the only change is at model.py:319-326, where series_oblique is moved to all_study.device before it is indexed with keep_series and acquisition_index. The torch.where condition, oblique_view_group and token_type are now all on the same device.

I could not run py_compile because the source tree is read-only, so I checked the syntax with ast.parse, which succeeded.

Everything else I checked in the prior review is unchanged:
- The flag is attached after group preparation, so sampling is unchanged.
- series_oblique goes through the same series-dropout mask as the other series tensors.
- The mapping to 9 view groups matches the tilt thresholds in training.yaml.
- The model is neutral at initialisation and uses no RNG.
- There is no leakage.

The integration is a correct add with nothing removed.

Unresolved concerns:
- The standard-slot bias columns affect every study, so a macro gain cannot be attributed to oblique marking unless the split-slot and non-split per-label deltas are compared.
- view_group_bias is a 2-D parameter, so it gets head_lr and weight decay (lightning.py:37-44). The decay pulls it toward zero, a mild prior against the mechanism.
- accounting_available=false: the claim that triplet and visit counts are unchanged still depends on the supervisor's zero-tolerance replay.
- model.oblique.attention_bias defaults to False. Under the no-legacy rule, remove it if the idea is discarded, or make it unconditional if it is adopted.
- 103 split studies on fold 0 give low power; per-label intervals on that subset will be wide.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/experiments/label-view-attention-bias-20260922T120824Z-29aa5863/changes.diff
- candidate source/src/knee_abnormality/modeling/model.py:318-326 (series_oblique moved to all_study.device before indexing)
- diff against /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/experiments/label-view-attention-bias-20260922T115857Z-576628b9/changes.diff: only the device line differs
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/jobs/orchestrator-20260922T120734Z-2474c8d3/report.json (the requested repair and the prior full review)
- candidate source/src/knee_abnormality/data/dataset.py:289-308 (flag attached after unchanged group preparation)

## completion-20260922T122554Z-3286e0b6

oblique-group-embedding is rejected. The macro gain is +0.00176, but the lower bound of the central 80% interval is −0.0005.

The contract requires a split-slot stratification, but the comparison lacks it, so I computed it from the saved OOF predictions. The 103 split studies are those with an oblique group in oblique_reach.parquet, and their positive counts match the oblique-groups result doc. Thresholding the soft targets at >0.5 gives candidate-minus-reference AUC deltas for split versus non-split studies (positives in brackets):
- ACL: −0.0092 (32) vs +0.0076 (101)
- MCL: +0.0484 (11) vs −0.0036
- Medial meniscus: −0.0117 (48) vs +0.0002
- Lateral meniscus: −0.0084 (22) vs −0.0050
- PF OA: −0.0128 vs +0.0006
- Effusion: +0.0169 (78) vs +0.0041
- Contusion: +0.0120 (23) vs +0.0056
- Fracture: +0.0227 (4) vs −0.0119
- Baker's: +0.0009 (13) vs +0.0149

The unweighted mean across labels is +0.0052 in split studies and +0.0013 in the rest. The split-subset mean comes mainly from MCL (11 positives) and fracture (4 positives), which are within small-sample noise.

On ACL, the label the ACL-oblique views exist for, the split studies got worse (−0.0092), and on the menisci too. The overall ACL gain (+0.0051) and the only label interval above zero, Baker's (+0.0136), both sit in the non-split studies, so they are training divergence rather than the mechanism. Effusion improves inside the split studies, but the dedicated oblique views are not what serves effusion.

Conclusion: a learned oblique marker added to the token sum gives no evidence that the model uses oblique views better on the labels those views serve. The trainable row 0 also adds a global offset, which further muddies attribution.

This completion's eligible decision is reject. Macro AUC went from 0.897527 to 0.899287 (+0.00176), but the central 80% paired-bootstrap interval is [−0.00050, +0.00398], so its lower bound is not above 0. Accounting is complete and exact: 960,240 training triplets, 35,020 visits and 24,770 validation triplets per epoch, with an unchanged validation selection identity. Runtime was 1,617 s against the 1,868 s ceiling. The implementation is sound, so the result is valid, and it is not an invalid run.

The queue is unchanged. I left the proposal axial-t1pd-to-oblique-transfer unadmitted. Its dependency gate passes. However, its effect is more oblique triplets per visit from the same series, and the contract has already ruled out denser sampling of the same oblique series: coverage over ten epochs is 85–93%. The proposal's own precedents are also weak: the frozen-model estimate of axial T1/PD's value, and a rejected reallocation, repeated_series_share, which led on fold 0 and lost publicly. It stays proposed, and could be reconsidered if a representation idea shows oblique tokens are being used.

Unresolved concerns:
- The supervisor's comparison.json stratifies only by vendor. It lacks the split-slot vs non-split per-label breakdown that the contract requires for mechanism claims. The breakdown above is my own exploratory calculation, with binarization at >0.5; Medial OA shows 325 positives against the evaluator's 328, and I did not confirm how the evaluator handles ties at the threshold. Add this stratum to future completions.
- The split subset has 103 studies, with 4–48 positives per label, so its per-label deltas carry no confidence intervals and are dominated by noise. That is especially true of MCL (11 positives) and fracture (4 positives).
- Cleanup after the discard: model.oblique.enabled, the PreparedSeries.oblique field and the series_oblique collate plumbing must not reach the incumbent. The prepared label-view-attention-bias candidate adds the same PreparedSeries.oblique and series_oblique plumbing independently.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/experiments/oblique-group-embedding-20260922T115506Z-f5d616dd/comparison/comparison.json (macro, per-label, vendor strata only)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/experiments/oblique-group-embedding-20260922T115506Z-f5d616dd/training/oof_predictions.csv vs /home/claudio/src/knee-abnormality/outputs/training/oblique/oof_predictions.csv (split-subset deltas computed ad hoc)
- /home/claudio/src/knee-abnormality/outputs/analysis/a/acquisition-geometry-20260922/oblique_reach.parquet (group=='oblique': 103 fold-0 split studies)
- /home/claudio/src/knee-abnormality/docs/validation/results/sampling/2026-09-22-oblique-groups-result.md:27-37 (same subset, positive counts match; previous ACL gain also sat in unchanged studies)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/state.json ideas.axial-t1pd-to-oblique-transfer (proposal left unadmitted)
- /home/claudio/src/knee-abnormality/autoresearch/campaigns/oblique/prior-work.md#sampling-coverage-of-the-oblique-groups

## review-20260922T123346Z-5b54d850

Candidate oblique-residual-readout (source sha 0a95f639…) adds a second attention pool for ACL, MCL and Lateral Meniscus that reads only oblique-group tokens, plus a zero-initialised additive readout on those three logits. It is gated to studies that have an oblique token. The sampling path is unchanged and the untrained forward pass matches the reference. The only defect found is that `series_oblique` is not moved to the device during inference (engine.py only moves `MODEL_BATCH_KEYS`), which crashes the final prediction pass. The fix is a single device move inside model.py.

The candidate stays on topic, and its design is sound apart from one bug that would end the run in a crash. Launching it as is would spend a full ~1,800 s training run and then fail before the evaluator gets predictions.

Bug: the new `series_oblique` collate tensor is not in `MODEL_BATCH_KEYS` (inference/engine.py:95-110). `engine.py` is not editable in this campaign. As a result, `move_batch` (engine.py:113-119, used at engine.py:169) leaves `series_oblique` on the CPU. In `encode_batch` the candidate then indexes that CPU tensor with the CUDA mask `keep_series`: `batch["series_oblique"][keep_series]`. Training is unaffected, because Lightning's `transfer_batch_to_device` moves every tensor. The final `predict_loader` pass would raise a device-mismatch error. This is the same crash the sibling candidate label-view-attention-bias hit, and its repair fixed it in model.py.

Requested repair, one change: in `encode_batch`, move the flag to the device before indexing, for example `batch["series_oblique"].to(all_study.device)[keep_series]`. Nothing else should change.

Checks that passed:
- Neutral start: `oblique_output_weight` and `oblique_output_bias` are zero-initialised, so the residual is exactly 0 at initialisation.
- New modules are built after `spatial_tokens`, so the parameter initialisation of existing modules does not change.
- `_prepare_slot` still prepares the standard group before the oblique group, so sampler RNG order and triplet, visit and budget arithmetic are unchanged. The oblique tag is applied through `replace()` after sampling.
- Studies without an oblique token fall back to full membership, which avoids an all-infinite softmax, and their residual is multiplied by zero.
- `index_add` is out-of-place.
- Validation stays deterministic: the tag comes only from the fixed `oblique_split` geometry, with no labels or seed involved.
- The integration is additive, `removed_idea_ids` is empty, and `active_incumbent_ideas` is empty, so nothing is superseded.

Queue and proposals are unchanged. `axial-t1pd-to-oblique-transfer` stays proposed because this event is a review.

Unresolved concerns:
- Blocking until repaired: `series_oblique` stays on the CPU in `predict_loader`, and model.py indexes it with the CUDA mask `keep_series`, raising a device-mismatch error in the final prediction pass (engine.py:95-119,169).
- `oblique_output_weight` is 2-D, so it receives weight decay 0.0 < wd at head_lr (lightning.py:35-44). The decay pulls the residual toward zero, a mild prior against the mechanism.
- The new Dropout in `oblique_attention` consumes training RNG, so the dropout stream diverges from the reference for every study. Mechanism claims must rest on the per-label deltas for the split-slot studies (103 on fold 0), and those intervals will be wide.
- `accounting_available=false`: whether triplet and visit counts are unchanged still depends on the supervisor's zero-tolerance replay. Statically, the sampling arithmetic is unchanged.
- Under the no-legacy rule, `model.oblique.enabled` (default False) and the configurable `targets` must be removed if the idea is discarded, or made unconditional if it is adopted.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/experiments/oblique-residual-readout-20260922T122704Z-b759f96d/changes.diff
- candidate source src/knee_abnormality/inference/engine.py:95-119 (MODEL_BATCH_KEYS, which omits series_oblique) and :169 (move_batch in predict_loader)
- candidate source src/knee_abnormality/modeling/model.py:~328 `oblique_index = batch["series_oblique"][keep_series]` (no device move)
- candidate model.py _attention_pool oblique branch (zero-initialised residual, has_oblique gating, fallback membership)
- candidate data/dataset.py:283-300 (standard group prepared before oblique group; tag applied via replace after sampling)
- candidate training/lightning.py:35-44 (optimizer groups: head_lr, weight decay on 2-D parameters)
- sibling repair label-view-attention-bias-20260922T120824Z-29aa5863 model.py uses batch["series_oblique"].to(all_study.device) for the same crash

## review-20260922T123557Z-e08e0069

Repaired candidate oblique-residual-readout (source sha 6095f578…) is ready to launch. It fixes the predict_loader CUDA/CPU crash inside model.py and changes nothing else relative to the reviewed design. A mechanism claim has to rest on the per-label deltas for the split-slot studies (103 on fold 0). The new oblique-attention Dropout changes the training RNG stream for every study, so the deltas on non-split studies measure run-to-run divergence, not the mechanism.

I approved the candidate. The repair is exactly the single device move I requested in the previous review, with no other change: I diffed the two changes.diff files, and they differ on one line. The previous review's other checks still hold because none of that code changed:
- the new path starts neutral,
- new modules are built after `spatial_tokens`, so existing parameter initialisation is unchanged,
- sampler order and budgets are preserved,
- `index_add` is out of place,
- validation is deterministic and label-free.

One extra check: the oblique attention MLP runs inside the block where autocast is disabled. Its input comes from `mil_norm`, a LayerNorm, which autocast keeps in fp32, so there is no dtype mismatch.

The queue is unchanged. `axial-t1pd-to-oblique-transfer` stays proposed because this is a review event.

Unresolved concerns:
- `oblique_output_weight` is 2-D, so it is trained at head_lr with weight decay (lightning.py:35-44). The decay pulls the residual toward zero, a mild bias against the mechanism.
- The new Dropout in `oblique_attention` draws from the training RNG, so every study's training diverges from the reference. Only 103 fold-0 studies hold a split slot, so the per-label intervals on that subset will be wide.
- accounting_available=false: whether triplet and visit counts are unchanged still depends on the supervisor's zero-tolerance replay. Statically, the sampling arithmetic is unchanged.
- No-legacy rule: if the idea is discarded, remove the `model.oblique.enabled` flag (default False) and the configurable `targets`. If it is adopted, make the readout unconditional.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/experiments/oblique-residual-readout-20260922T123437Z-66d0dc1f/changes.diff (differs from the b759f96d diff only at the device-move line)
- candidate source src/knee_abnormality/modeling/model.py:328 (`series_oblique` moved to `all_study.device`) and :287-309 (zero-initialised gated oblique residual)
- candidate src/knee_abnormality/inference/engine.py:95-116 (MODEL_BATCH_KEYS includes series_study_index)
- candidate src/knee_abnormality/data/dataset.py:289-309 (standard group prepared before the oblique group; tag applied via replace after sampling)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/jobs/orchestrator-20260922T123346Z-d6769c1e/report.json (prior review and requested repair)

## proposals-20260922T123948Z-3721973e

The frozen evidence for mixture pooling is directionally consistent but small on the full fold. Subset gains of +0.0015 to +0.0039 shrink to about +0.0002 to +0.0004 macro on the full fold (for example, k_05 ×1 τ1 ≥20°: joint −0.00019 vs mixture +0.00022). Adoption therefore needs a trained effect several times larger than the frozen one. Part of the frozen gain is sub-bag ensembling: mean_intact_standard, two passes without tilted series, gains +0.0000 to +0.0037 on the subset. The frozen checkpoints (k_025/k_05/k_075) were trained with shared slot budgets; mixture vs joint pooling on the split-budget reference checkpoint has not been measured. Per-label signature to check on split studies: MCL and PF OA recover while ACL holds. Under joint pooling (k_05 full fold), MCL and PF OA cost −0.0006 to −0.0026; dropping the tilted series instead costs ACL −0.0031 to −0.0046.

I admitted split-slot-view-mixture-pooling and ranked it second, directly behind the reserved head. It has no dependencies. It has the campaign's only consistent positive frozen measurement: the probability mixture of standard-only and oblique-only passes beat joint pooling on split studies in 9 of 9 checkpoint × setting cells. It adds no triplets and no backbone cost, and it attacks the costs seen in the reference's split studies (MCL −0.058, PF OA −0.026 vs k_05) without changing any other study's forward pass at fixed weights. It differs from both the running label-view-attention-bias and the reserved oblique-residual-readout: it removes within-slot competition and slot-share inflation, not reweighting inside one softmax or adding a pool.

I left oblique-feature-adapter as proposed rather than queuing it. Its premise, that oblique content is misread, rests on tilted-only frozen losses confounded by thin-slab coverage and mixed on k_025. The rejected constant marker also shows that oblique-specific representation changes can hurt ACL. It stays eligible if the routing ideas show that oblique tokens are used but misread. axial-t1pd-to-oblique-transfer also remains proposed: frozen bag-size evidence implies roughly +0.0002 macro for redistributions of this size.

Unresolved concerns:
- Contract neutrality: the fixed ½ mixture changes the untrained forward pass for split studies. The implementation must use the proposal's gated form, z = z_joint + λ_c(z_mix − z_joint) with per-label λ initialised to zero, so the untrained model reproduces the reference. Review will check this. λ is 1-D; confirm its optimizer group and weight-decay treatment.
- The sub-bag ensembling confound means a gain would not prove that the oblique views are used. The within-model toggle (joint / mixture / pass A only / pass B only on the trained candidate) should be reported with the split-study per-label deltas.
- Split-slot detection must use kept, visually valid tokens after series dropout. The mask accounting must be inspected at review, including studies where dropout removes one group.
- Interaction with label-view-attention-bias (running) and oblique-residual-readout (reserved head): if either becomes incumbent, the rebase must put the bias inside the shared scores and add the residual once after the mixture, as the proposal states. Declared integration must then list them as retained.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/jobs/researcher-20260922T122705Z-20a56bf3/report.json (proposal event)
- /home/claudio/src/knee-abnormality/outputs/analysis/t/tilted-slot-policies-v2check-20260922/results.csv and tilted-slot-policies-check-20260922/results.csv (frozen additive/tilted/standard/mixture per-label and subset deltas)
- scripts/adhoc/score_tilted_slot_policies.py (frozen policy scorer)
- docs/validation/results/sampling/2026-09-22-oblique-groups-result.md (reference split-study MCL/PF OA costs)
- docs/validation/results/pooling-and-attention/2026-09-17-count-normalized-attention-result.md (rejected per-acquisition normalisation, −0.0035)

## completion-20260922T125404Z-ccf69e65

The per-label bias raised macro AUC by +0.0018, just short of adoption: the 80% lower bound is −0.00018. Across labels:
- Largest gains: Baker's +0.0102, Effusion +0.0069, MCL +0.0059, Lateral OA +0.0052 and Medial OA +0.0039.
- Losses: Fracture −0.0123, ACL −0.0043, Lateral meniscus −0.0008 and PF OA −0.0005.

This pattern goes against the mechanism. ACL and lateral meniscus are the labels enriched in oblique studies (about twice the base rate), and both fell. The gains are on effusion, Baker's and OA, labels that the six standard-slot bias columns would affect in every study. So the likeliest source of the gain is a general per-label slot attention prior, or run-to-run divergence, rather than oblique-specific routing.

The comparison artifacts do not split per-label deltas into the 103 split-slot studies and the 802 others, so the contract's mechanism test was not run on this completion. The oblique-groups result shows how large pure divergence can be: ±0.01–0.02 per label on studies whose inputs did not change.

A standard-slot-only per-label attention prior is a separate idea from oblique marking. It is weakly suggested here, but it is outside this campaign's scope unless it is tied to oblique groups.

Completion of label-view-attention-bias (repair 1). The fixed eligible decision is reject. Macro AUC went from 0.897527 to 0.899295 (+0.00177), which clears the 0.001 minimum gain, but the central 80% paired-bootstrap interval is [−0.00018, +0.00370]. Its lower bound is not strictly above 0, so the rule discards the idea. The 95% interval is [−0.00124, +0.00475].

The run is valid:
- Accounting matched the sealed limits exactly: 960,240 training triplets, 35,020 visits, the same training and validation cohort hashes, and 24,770 validation triplets per epoch.
- Validation selection was identical in every epoch.
- Measured compute was 1,617 s, under the 1,868 s ceiling.
- The only config difference is model.oblique.attention_bias.
- The selected checkpoint is epoch 8; the reference selected epoch 9.

No queue change. The idea leaves the live queue under the discard rule, and the other queued ideas test different mechanisms.

I admit neither proposal in this event. Their dependencies are met, but this result does not raise their priority, and the queue already holds four ideas.

Unresolved concerns:
- The contract requires per-label deltas reported separately for split-slot and non-split studies. comparison.json has only vendor strata, and no split-study list was found among the artifacts, so the mechanism attribution is unverified. The decision does not depend on it.
- The standard-slot bias columns act on every study, so the +0.0018 macro gain cannot be separated from a general slot prior or from training divergence. ACL and lateral meniscus, the labels enriched in oblique studies, fell.
- The candidate selected epoch 8 and the reference epoch 9. Checkpoint-selection noise is not in the bootstrap.
- No-legacy cleanup: the discarded idea's model.oblique.attention_bias flag and its view_group_bias code exist only in the experiment source. They must not enter the incumbent.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/experiments/label-view-attention-bias-20260922T120824Z-29aa5863/comparison/comparison.json (keys: per_target, strata=vendor only; no split-slot stratum)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/experiments/label-view-attention-bias-20260922T120824Z-29aa5863/comparison/per_label.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/experiments/label-view-attention-bias-20260922T120824Z-29aa5863/training/fold_0/ (best-epoch=08-val_macro_auc=0.89930.ckpt, triplet_accounting.json)
- /home/claudio/src/knee-abnormality/docs/validation/results/sampling/2026-09-22-oblique-groups-result.md:24-38 (divergence magnitude on 802 unchanged studies)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/jobs/orchestrator-20260922T121015Z-2e55d4df/report.json (prior review concerns)
