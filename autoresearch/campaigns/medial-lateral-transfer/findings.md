# Campaign findings

## proposals-20260920T214214Z-65a053d4

The rejected adapter's failure mode is specific and constrains both admitted ideas: medial meniscus is 40.7% positive in the training fold against lateral 15.8%, so gradients into any shared meniscus-family parameter are dominated by the medial side, and what degraded most was within-study compartment discrimination (meniscus margin −0.0191 [−0.0392, −0.0034] vs control B) rather than macro AUC (−0.0015 [−0.0054, +0.0022]). Soft classifier coupling regressed the same family (−0.0041). Two mechanisms that tie compartments with unequal prevalence have now moved the meniscus family the same direction; a shared auxiliary decoder trained on each compartment's own label shares this prevalence asymmetry even though it never forces paired predictions to match. The margin diagnostic should therefore be read for both admitted ideas, while acceptance stays on the fixed macro rule.

Scale context for interpretation: across 16 null arms on this recipe, macro delta against each queue's control mean ranged −0.0050 to +0.0011, and the two 2026-09-20 controls differ by 0.0034 — larger than 15 of the 16 candidate deltas. Per-label noise floors differ roughly fourfold (sd 0.0026 on Medial OA to 0.0100 on MCL), so equal numeric per-label deltas are not equally informative.

Provenance check: the four archived sharing runs supplied as external evidence (`disease-compartment-attention`, `meniscus-classifier`, `meniscus-oa-classifiers`, `meniscus-soft-coupling`) all record `validation_image_multiplier: 2` and carry no compartment-sharing config key, so they are old-recipe evidence only and none duplicates an admitted proposal.

Both imported ideas pass proposal_admission (no dependencies) and both are inside the user's scope; neither restores the rejected pooled adapter nor its retired independent arm, so both are admitted and the queue opens with them.

shared-disease-auxiliary ranks first. It is the cheapest distinct route left: it changes training supervision only, leaves the twelve main logits and their independent heads untouched at inference, and reuses pooled features from the ordinary forward pass, so its runtime risk against the 1,797 s ceiling is minimal. It also probes a mechanism class the two measured regressions (classifier coupling, pooled adapter) did not test: neither tied final decision rules nor an inference-path module. Its interpretive cost is explicit and accepted — a matched independent-auxiliary control would be required before attributing any gain to sharing rather than extra meniscus task weight — and that follow-up is deliberately not prequeued.

disease-evidence-compartment-routing ranks second. Sharing before pooling is a genuinely earlier-stage mechanism and is the one place the prior failures do not speak to, but per-token work scales with the full image budget rather than with studies, so it carries the only real budget and device-memory risk in the queue, and its design space (family scope, transform, routing inputs) is still open. It is better decided after the cheap training-only probe reports, and it needs a budget argument at implementation time rather than a block now.

I verified the four archived external runs (1A/1B/1C, soft coupling) record `validation_image_multiplier: 2` and no compartment-sharing config key, confirming prior-work's statement that they are archived-recipe evidence and not controls for this campaign, and that proposal 2 does not duplicate an arm active or completed on the current recipe.

Unresolved concerns:
- Prevalence asymmetry (medial meniscus 40.7% vs lateral 15.8%) is the mechanism blamed for the adapter's margin regression; a shared auxiliary decoder inherits it. Unresolved whether any shared meniscus-family parameter can avoid being fitted to the majority compartment.
- shared-disease-auxiliary cannot attribute a gain to transfer on its own: a matched independent-auxiliary control at the same loss weight would be required, and that control is not queued. Any positive result stays unattributed until then.
- The auxiliary family, scale and any within-run schedule are unchosen. They must come from a stated rationale, not a sweep; OA is the weaker first family given the documented OA annotation/severity issues in the disagreement audit.
- disease-evidence-compartment-routing has no runtime or device-memory estimate. Per-token work scales with the image budget, unlike pooled adapters, and must fit the 1,797 s ceiling; this is unresolved until a concrete design exists.
- Routing must not reduce to 1C's forced common medial/lateral attention contrast across diseases, and attention maps alone cannot demonstrate correct disease attribution. The distinction is asserted in the proposal but not yet established by a design.
- Single-run resolution on this recipe is roughly 0.005 macro, and no arm has exceeded +0.001 against its control mean. Intervals condition on the fitted models and exclude training-seed and checkpoint-selection variability.

Evidence:
- docs/validation/results/2026-09-20-compartment-adapter-result.md:41-53 (family and margin intervals vs both controls), :82-92 (prevalence asymmetry argument), :94-107 (no width/coefficient sweep; `independent` retired as residual-readout reparametrization), :109-133 (empirical null: macro min −0.0050, max +0.0011; per-label sd table), :143-149 (caveats)
- autoresearch/campaigns/medial-lateral-transfer/prior-work.md:13-31 (completed external adapter run identity and rejection), :33-57 (pinned control B reference and derived limits), :85-93 (history table), :102-119 (ownership and retirement of the independent arm)
- docs/research/medial-lateral-transfer.md:142-160 (candidate 3 design, required independent-auxiliary control, warning against probabilistic-OR study-level targets and teacher distillation from 1B/1C), :120-140 (candidate 2 rejection header and capacity-attribution caveat)
- Verified via effective_config.json for outputs/training/{disease-compartment-attention,meniscus-classifier,meniscus-oa-classifiers,meniscus-soft-coupling}: all validation_image_multiplier 2, no compartment-sharing key — archived recipe, not current-recipe duplicates
- docs/history/sharing/oa-and-meniscus-disagreement-audit.md (OA annotation and acquisition sensitivity limits bearing on family choice)

## review-20260920T221049Z-35fb3797

The auxiliary head as implemented is not training-only in practice: its 258 parameters are registered under `model.`, so they are averaged by the weight-averaging path, written into the selected checkpoint, and required at strict load time by the inference reconstruction. That coupling is the sole cause of the out-of-workspace edit and is avoidable by hosting the head on the LightningModule.

Attribution limit for interpretation later: the candidate changes two things at once relative to control B — a *shared* linear detector direction across the medial/lateral meniscus pooled features, and an extra 0.3-weighted BCE task load on exactly those two labels. A matched control with two independent auxiliary weight vectors (same added weight, no sharing) would be needed to attribute any macro gain to shared morphology rather than to additional meniscus task weight. Capacity is not a plausible confounder here (258 params, no inference-time use).

Verified numerically-neutral by inspection, not by run: `return_pooled` adds no computation and does not alter RNG consumption, so main logits and the training stochastic stream should match the parent for an identical seed when the auxiliary is unconfigured. Added cost is one 256x2 dot product per step; no measurement against the 1,797 s ceiling exists yet, which is acceptable at this magnitude but remains unmeasured.

The gate failure is real and structural, not cosmetic. Placing SharedDiseaseAuxiliaryHead inside KneeAbnormalityModel puts its parameters under the `model.` prefix, so the saved state_dict carries them and `checkpoint_model`'s strict load (inference/checkpoint.py:96-104) fails unless the constructor is given `target_names` — which is why the developer edited a non-editable file. A training-only auxiliary should not enter the inference checkpoint at all.

Concrete repair (all inside the editable set): own the auxiliary head on `KneeModule` (training/lightning.py) instead of on the model; resolve the two target indices there from the already-available `targets`; drop the `target_names` constructor parameter and the auxiliary construction from modeling/model.py, keeping only the `return_pooled` plumbing (and, if desired, a head-free pooled accessor) so training_step can call `pool_encoded(encode_batch(batch), return_pooled=True)`; revert inference/checkpoint.py to the parent byte-for-byte. Because `build_optimizer` enumerates `model.named_parameters()` (lightning.py:35-44), the relocated auxiliary parameters must be added explicitly to the optimizer at `head_lr` with the same ndim-based weight-decay rule, or the shared detector will never be updated. After the move, `self.averaged` (a copy of `self.model`) and the save/load key surgery at lightning.py:265-301 stay unchanged, and the exported checkpoint is identical in structure to the reference.

Otherwise the mechanism is on-scope and the main path looks untouched: logits are computed by the same single encode+pool pass, `pooled` is the tensor already used for the main logits (model.py:244-263), validation/prediction still goes through unchanged `forward`, the auxiliary term is added to rather than replacing the main balanced BCE, and both compartments keep independent biases so medial-only, lateral-only, neither and both remain representable. Dataset, sampling and augmentation code are untouched, so cohort, triplet and determinism seals are unaffected by construction.

The integration plan itself is consistent with an empty `active_incumbent_ideas` (pure addition, no removals), but it omitted the inference/checkpoint.py edit from `added_components`; the resubmission must declare the final component set exactly.

Unresolved concerns:
- Gate: src/knee_abnormality/inference/checkpoint.py is outside the campaign editable set; approval is impossible until it is byte-identical to parent_source.
- If the head is relocated to KneeModule without an optimizer change, build_optimizer (lightning.py:35-44, over model.named_parameters()) would silently exclude it and the shared detector would stay at its random initialization.
- Training-only auxiliary parameters currently enter the averaged model and the exported checkpoint; the resubmission must keep the inference checkpoint structurally identical to the reference.
- Integration plan omits the checkpoint.py edit from added_components; the declared component set must match the final diff.
- Attribution: a shared direction and +0.3 meniscus task weight are confounded; an independent-auxiliary control is required before crediting shared morphology for any macro gain.
- Runtime impact is argued analytically only; no measured timing against the 1,797 s ceiling exists.

Evidence:
- gate_error in review-20260920T221049Z-35fb3797: changes outside campaign workspace ['src/knee_abnormality/inference/checkpoint.py']
- workspace diff vs parent_source: inference/checkpoint.py:101-103 adds `targets` to KneeAbnormalityModel(...) purely to satisfy strict load
- workspace src/knee_abnormality/modeling/model.py:54-159 constructor target_names + compartment_transfer construction; :225-264 return_pooled; :350-358 forward_with_auxiliary
- workspace src/knee_abnormality/training/lightning.py:309-350 auxiliary masked BCE added to main loss, sliced class_balance_weights, guarded by auxiliary_observed.any()
- workspace src/knee_abnormality/training/lightning.py:35-52 build_optimizer enumerates model.named_parameters() only
- workspace src/knee_abnormality/training/lightning.py:265-301 checkpoint key surgery over 'model.'/'averaged.' prefixes; :412-435 averaging copies self.model state
- workspace src/knee_abnormality/modeling/compartment_transfer/shared_disease_auxiliary.py: shared 256-d weight, 2 independent biases (258 params)
- developer report /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260920T220248Z-bd1596e9/report.json (constraint_evidence; weight=0.3 stated, not swept)

## proposals-20260920T221509Z-1050141f

Source verification of the two admitted proposals against the pinned reference source: per-target pooled vectors are available at exactly the point proposal 1 needs them (`_attention_pool`, model.py:204-239, float32 outside autocast), so the odd zero-sum increment z_c += rho(D), z_c' -= rho(D) can be added without changing `forward`'s return type or the twelve public logits. `COMPARTMENT_PAIRS` (metrics.py:6-9) fixes the two families, so pair identity is not a post-hoc label choice, though it must be re-expressed as target indices in config because the constructor takes only `target_count` (model.py:53).

`PreparedSeries.patient_maps` is computed and retained in preparation (preparation.py:31, 228-235), confirming proposal 2's central plumbing claim is feasible without new decoding or extra encoder passes.

An attribution gap applies to proposal 1: the existing `residual_readout` path (model.py:129-137) shows the codebase already has a per-target nonlinear readout, and rho adds comparable parameters. The zero-sum odd form blocks a common-mode explanation but not a pure-capacity one, so a win would need a matched control (e.g. a non-odd or single-compartment-only increment of the same size) before the transfer interpretation is claimed. No campaign run has completed, so every budget and runtime figure in both proposals remains an unverified estimate.

Admitted two of three proposals; the queue membership change requires a full ranking.

compartment-pair-contrast-readout enters second. I verified it is implementable as described against the reference source: `_attention_pool` already materialises per-target `pooled` of shape [b, t, f] and forms logits from it (model.py:204-239), and the evaluator's pair identity is frozen in metrics.py:6-9. It is the first mechanism in which one compartment's prediction reads the other's evidence; the odd, zero-sum increment is structurally unable to produce a common-mode/co-occurrence boost, which is the failure mode recorded for the pooled adapter and the weight-coupling penalty, and exchange equivariance makes the shared parameters prevalence-symmetric. Zero-initialised output means initialisation reproduces the reference exactly. Cheapest new mechanism class available.

disease-evidence-compartment-routing stays ahead of the new attention proposal: it is already queued, token-level but without the side-rule dependency.

anatomical-medial-axis-mirror-attention enters last of the queued set. It is the only proposal adding information rather than parameter structure, and `patient_maps` genuinely exist in preparation (preparation.py:31,228), so the plumbing claim is sound. It ranks behind the pooled probe because of its real implementation surface (preparation, dataset, spatial_tokens, model), token-level budget exposure against the 1,797 s ceiling, an unverified interaction between reflection augmentation and patient-space X, and two adverse precedents it must not become a rerun of.

reciprocal-compartment-readout-transfer is not admitted and remains proposed. It sits in the same mechanism class as the reserved, developing `shared-disease-auxiliary`; its own priority rationale concedes the pending result should decide whether it is still distinct. Admitting it now risks two auxiliaries on the same family with no attribution story. Revisit once that result is interpreted; if it is later admitted after an acceptance in that class, it must declare replacement or stack explicitly.

Unresolved concerns:
- compartment-pair-contrast-readout adds per-family readout parameters; a macro win would not by itself distinguish shared-morphology transfer from added capacity. The existing residual_readout run is prior evidence, not a matched control. A matched non-odd or single-sided increment of equal size is the attribution control if it wins.
- compartment-pair-contrast-readout's open_choices (odd-only vs common-mode branch, LayerNorm on D, hidden width, family scope, per-family vs cross-family rho, detachment) must be fixed with a stated rationale before launch. A cross-family shared rho would re-impose the 1C constraint and is not the default.
- anatomical-medial-axis-mirror-attention: correctness of patient-space X under reflection augmentation is unverified. The image flip must be reflected in the map used for the token offset, or mirrored studies receive an inverted medial axis. This needs explicit demonstration at review, not assertion.
- anatomical-medial-axis-mirror-attention: side assignment is an acquisition convention (96.14% assigned, 42 fold-0 abstentions). Realized per-split side coverage and abstention counts must be recorded as run artifacts, and a wrong sign silently inverts anatomy for that study.
- Token-level budget for proposal 2 is estimated only (~0.2% of step time plus collate geometry). It is the one admitted idea with plausible exposure to the 1,797 s ceiling and must be timed on this recipe.
- reciprocal-compartment-readout-transfer is unqueued pending the developing shared-disease-auxiliary result. Its distinctness from that arm is currently asserted, not measured; the researcher's own note identifies that result as the evidence needed to rank it.

Evidence:
- researcher report: outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260920T220249Z-4555983d/report.json (proposals, exhausted_directions, missing evidence)
- src/knee_abnormality/modeling/model.py:204-239 — per-target pooled [b,t,f] and logit formation in float32 outside autocast; :129-137 residual_readout parameters; :53 constructor takes target_count only
- src/knee_abnormality/evaluation/metrics.py:6-9 — frozen COMPARTMENT_PAIRS for meniscus and OA
- src/knee_abnormality/data/preprocessing/preparation.py:31,228-235 — PreparedSeries.patient_maps computed per triplet channel
- docs/validation/results/2026-09-20-compartment-adapter-result.md — rejected pooled shared adapter, meniscus -0.0114 [-0.0200,-0.0030], within-study margin -0.0191 [-0.0392,-0.0034]
- autoresearch/campaigns/medial-lateral-transfer/prior-work.md#active-run-and-launch-readiness — adapter run is external evidence; control B pinned, 1,562.282713 s, 1,797 s ceiling
- docs/validation/results/2026-09-14-meniscus-soft-coupling-result.md — isotropic weight coupling: macro -0.0015, meniscus -0.0041
- docs/metadata/knee-side-policy-audit.md — 96.14% side assignment, fold-0 862/904 assigned, 42 unknown

## review-20260920T222103Z-34b9ddfa

Static confirmation of the structural fix: the saved checkpoint gains only two non-`model.` keys (compartment_transfer.weight/bias), which on_save_checkpoint passes through and which checkpoint_model's `model.`-prefix filter discards, so the exported inference model reproduces the recorded configuration with no shim.

New latent defect found, outside the planned run path: on_load_checkpoint (lightning.py:285-297) rebuilds state_dict from only `model.`/`averaged.` keys, dropping compartment_transfer.*, so a genuine weight-averaging resume would fail Lightning's strict load with missing keys. It fails loudly rather than silently, and the planned single full-budget run does not resume.

Also stale: scripts/adhoc/profile_training.py:130,257 and profile_threaded_loading.py:268 still call build_optimizer(model, ...) with the old single-module signature and would now raise on iterating a Module. Those files are outside the campaign editable set, so they cannot be repaired here; neither is on the training or prediction path.

Attribution limit carried into interpretation unchanged: the candidate simultaneously introduces a shared detector direction and +0.3 BCE weight on exactly the two meniscus labels. A matched control with two independent auxiliary weight vectors at the same weight remains necessary before crediting shared morphology for any macro gain. Capacity is implausible as a confounder (258 training-only parameters, no inference use). Weight 0.3 is asserted, not swept, and must not become a sweep.

The repair is approved. The single blocking gate (edit to the non-editable inference/checkpoint.py, forced by hosting the auxiliary head inside KneeAbnormalityModel) is resolved by relocating the head to KneeModule, and the optimizer inclusion I flagged as the risk of that move is implemented (lightning.py:38-52, 547-551), so the shared detector will actually be updated.

Main-path neutrality holds: when the auxiliary is configured, training_step computes logits via model.pool_encoded(model.encode_batch(batch), return_pooled=True), which is exactly model.forward (model.py:328-329), so there is no second encoder pass, no change in RNG consumption, and no change to main-loss value or gradients. Validation and final prediction still run the unchanged forward on `self.averaged`/`self.model` (lightning.py:473-485), independent of the auxiliary and of the training seed. No validation-derived quantity enters the auxiliary; targets come from the batch's own observed labels.

On scope: pooled[:, t] is the per-target attention-pooled feature already consumed by the main head, so a single shared weight vector across the medial and lateral meniscus vectors supplies transfer beyond the shared backbone, while two independent biases keep neither/medial-only/lateral-only/both representable. Added cost is one 256x2 dot product per step against the 1,797 s ceiling.

No queue or proposal change: this event carries no evidence bearing on ranking, and reciprocal-compartment-readout-transfer stays a surplus proposal.

Unresolved concerns:
- Resume path: on_load_checkpoint (lightning.py:285-297) discards compartment_transfer.* keys, so a weight-averaging resume would fail a strict load; acceptable only because the planned run does not resume.
- scripts/adhoc/profile_training.py:130,257 and profile_threaded_loading.py:268 still pass a single Module to the new list-based build_optimizer and would raise; those files are outside the editable set and off the training path.
- Attribution: shared direction and +0.3 meniscus task weight are confounded; an independent-auxiliary control at the same weight is required before crediting shared morphology for a macro gain.
- weight=0.3 is asserted without justification; a later coefficient sweep would not be a new mechanism.
- Runtime impact argued analytically (one 256x2 dot product per step, preparation 0.119 s); no measured total against the 1,797 s ceiling exists yet.
- Undeclared trivial additions: empty compartment_transfer __init__.py files and an unused empty src/knee_abnormality/training/compartment_transfer directory.

Evidence:
- diff -rq parent_source vs candidate source: only configs/training.yaml, src/knee_abnormality/config.py, modeling/model.py, training/lightning.py differ; new dirs modeling/compartment_transfer and (empty) training/compartment_transfer; inference/checkpoint.py and all data/augmentation/spatial sources byte-identical
- candidate modeling/model.py:204-243 return_pooled returns (logits, pooled) with pooled the per-target (b,t,f) tensor already used for main logits; :310-329 pool_encoded passthrough and unchanged forward
- candidate training/lightning.py:239-257 KneeModule-owned SharedDiseaseAuxiliaryHead(PROJECTION_WIDTH) with indices from targets.index; :329-380 auxiliary masked BCE added to main loss with class_balance_weights[:, indices] and own observed mask; :38-52 and :547-551 optimizer inclusion at head_lr
- candidate training/lightning.py:285-321 checkpoint key surgery (resume drops compartment_transfer.* keys); :473-485 validation uses unchanged averaged/live forward
- parent src/knee_abnormality/inference/checkpoint.py:96-107 strict load over model.-prefixed keys only, unchanged
- candidate modeling/compartment_transfer/shared_disease_auxiliary.py:1-13 shared 256-d weight, 2 independent biases (258 params)
- prior review /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260920T221049Z-a21b5e6c/report.json (repair conditions, now all met)
- ast.parse of the four changed modules with the supplied interpreter: parsed ok

## review-20260920T223029Z-3cc94970

Independent CPU verification of the candidate module (no GPU, no training): initialization parity with the parent is exact (0.0 logit difference), within-pair increments sum to zero up to fp32 rounding (<=6e-8), the eight non-pair labels are untouched exactly, exchange equivariance rho(m,l) = -rho(l,m) holds exactly, and the parameter count is 17,536. Pair indices verified against data/folds/all_5fold_v3/protocol.json: 2/3 = Medial/Lateral Meniscus, 4/5 = Medial/Lateral OA.

Structural note distinguishing this arm from the running shared-disease-auxiliary repair: here the new parameters sit inside KneeAbnormalityModel under the model.* prefix, so they are exported in the checkpoint, averaged, and reloaded by the existing on_load_checkpoint filter; checkpoint_model reconstructs them from the recorded config with no compatibility shim and no new constructor argument.

Interpretation limit inherent to the design: the zero-sum constraint means the module can only reallocate evidence between the two compartments of a pair and cannot change the pair's total. Any macro gain must therefore come from sharpened medial-vs-lateral discrimination, not from co-occurrence or overall pair calibration; conversely, a null result does not test those channels. A further caveat for a null result: out.weight is zero-initialised, so the hidden layer receives zero gradient at step 0 and learning must bootstrap through the output weight alone; a near-no-op outcome would be weak evidence about the mechanism itself.

Approving the candidate: the diff implements the assigned exchange-symmetric pair-contrast readout, preserves the contract (no recipe, loss, sampling, cohort or interface change; only preprocessing_identity-participating file touched is model.py, which the policy allows), and records its source and configuration.

Queue change: admitting reciprocal-compartment-readout-transfer (proposal_admission reports no unresolved dependencies). It is ranked second because it is the cheapest clean test of the campaign hypothesis — it reuses each existing head on the other compartment's pooled features with essentially no new parameters and no inference-time change, so a positive result is much less confounded by capacity than either the contrast readout or the running shared-disease-auxiliary arm, and a null result constrains the hypothesis rather than the implementation. It is mechanistically distinct from both active arms: no tying, no new detector, training-only. It supersedes nothing; the retired isotropic weight-distance coupling penalty is prior evidence and its coefficient sweep is not restarted. Remaining queue order is unchanged below it.

Unresolved concerns:
- Attribution: the readout adds a nonlinear, higher-capacity path on four targets simultaneously with the exchange-symmetric tying. A matched control with independent (untied, non-zero-sum) per-target nonlinear readouts on the same four targets is needed before crediting shared morphology for any macro gain; the older twelve-label residual_readout run is prior evidence, not that control.
- No measured runtime exists yet against the 1,797 s ceiling; the negligible cost is argued from parameter count (17,536, 0.12%) and preparation 0.174 s only.
- hidden_width=32, per-family independent rho, and detach_inputs=false are asserted design choices, not measured; a later sweep of these would not constitute a new mechanism.
- Zero-initialised output weight leaves the hidden layer with zero gradient at initialization, so an inconclusive result could reflect a near-no-op rather than a refuted mechanism.
- Undeclared trivial addition: an empty src/knee_abnormality/training/compartment_transfer directory (inside the editable set, no effect on behavior).
- Interpretation: the zero-sum design cannot alter a pair's total evidence, so any gain in the both-positive or neither co-occurrence regime would have to come from elsewhere and must not be attributed to this mechanism.

Evidence:
- changes.diff, experiments/compartment-pair-contrast-readout-20260920T222418Z-147e1fbd/changes.diff: complete candidate diff (configs/training.yaml, config.py, modeling/model.py, new modeling/compartment_transfer/)
- candidate source src/knee_abnormality/modeling/compartment_transfer/pair_contrast.py:1-48 (ExchangeEquivariantPairContrast, CompartmentPairContrastReadout)
- candidate source src/knee_abnormality/modeling/model.py:142-157 (guarded construction, target_count bound check) and :256-257 (call site inside the float32 autocast-disabled block, after base and residual logits)
- candidate source src/knee_abnormality/training/lightning.py:35-52 build_optimizer — new non-backbone parameters receive head_lr and standard weight decay, unchanged code
- recursive diff of candidate source vs parent_source (recovery/historical-source-20260920T215302Z-e086c1eb/source): only the four declared paths differ
- data/folds/all_5fold_v3/protocol.json targets list: indices 2/3 and 4/5 confirm the configured meniscus and OA pairs
- developer report jobs/developer-20260920T222418Z-7ea45c30/report.json (claims independently re-verified on CPU, not taken at face value)
- state.json idea reciprocal-compartment-readout-transfer: mechanism cross-compartment-readout-reuse-auxiliary, training-only, distinct from shared-disease-auxiliary and from the retired meniscus-soft-coupling weight-distance penalty

## proposals-20260920T223553Z-5c6d30a4

I re-derived the probe both proposals rest on, because the cited artifacts under /tmp/medial-lateral-probe/ no longer exist. Recomputed from the reference's saved fold-0 predictions joined to data/folds/all_5fold_v3/targets.csv, 904 studies, labels binarised at >0.5, logit-space Spearman, rank-based AUC.

OA reproduces the report closely: predicted-logit Spearman 0.881 vs soft-label Spearman 0.674; Medial OA score ranks the Lateral OA label at AUC 0.833 against the lateral head's own 0.870; lateral AUC 0.763 inside 328 medial-positive studies vs 0.840 in 576 medial-negative; joint groups both/medial-only/lateral-only/neither = 185/143/55/521 (report 186/143/55/520, a single boundary study from threshold handling); discordant-subset margin AUC 0.866 exactly as reported.

Meniscus reproduces in sign and magnitude: 0.483 vs 0.331, lateral-from-medial cross-AUC 0.596 and medial-from-lateral 0.712 (the report's 0.711 is the latter direction), own lateral AUC 0.898, and conditional lateral AUC essentially flat (0.888 medial-positive vs 0.890 medial-negative).

Two deviations worth recording. The report's meniscus margin AUC 0.971 did not reproduce; I get 0.959 on the discordant subset (n=345), and the report does not define its subset, so that one figure is unverified. Second, the conditional-degradation effect is specific to OA: meniscus shows no within-study degradation at all, which the proposals' framing understates. That weakens the case for applying either mechanism to both families by default and makes the OA pair the family where a compartment-discrimination mechanism has measurable room.

The researcher's own caveats hold and I did not resolve them: label noise deflates the label-side correlation, shared study-level nuisance inflates the prediction side, and nothing here distinguishes "the paired scorers attend to the same tokens" from "different tokens carrying the same information".

Both proposals have satisfied dependencies and are admitted; the queue is reranked because the probe evidence I re-derived changes what the strongest untried mechanism is.

conserved-evidence-compartment-assignment enters at rank 2 (behind the reserved prepared head). It is the only registry mechanism whose compartment-specific parameterisation is a normalisation across compartments per token: one detector per family trained by both compartments' positives, with only placement compartment-specific. That is prevalence-symmetric in the shared part by construction, which is exactly the failure mode of the rejected pooled adapter (prevalence domination of shared pooled parameters), and it acts before compression rather than after. It reproduces the parent at initialisation, so an identity check is available.

disease-evidence-compartment-routing is parked. It is the same class as the admitted proposal but gives each compartment a free, unconstrained token route, which does not constrain the measured problem: the paired heads already behave like one study-level factor plus an offset, and a free per-compartment route is also the reading that additional capacity alone would produce. Running both would be two near-duplicates competing for full-budget slots; the constrained form dominates on rationale. Parked, not rejected — a routing arm remains revisitable if conserved assignment shows the shared-detector half is the useful part.

cross-compartment-attribution-exclusivity is admitted but ranked last. Its appeal is real (zero added parameters, nothing at inference, so a macro gain cannot be capacity), but by its author's own statement it supplies no transfer route; it only removes a penalty for sharing. Its value therefore depends on whether the pending sharing arms actually lose within-study discrimination, which the reserved head and the running auxiliary will show. Two further reasons to hold it: it turns the compartment-margin diagnostics into partial training targets, so they stop being independent confirmation of transfer, and the exclusivity premise is questionable at stride-32 tokens whose receptive fields span both compartments. Its named random-mask attribution control is correctly not prequeued.

Relative order of reciprocal-compartment-readout-transfer and anatomical-medial-axis-mirror-attention is unchanged.

Unresolved concerns:
- Probe artifacts cited as precedent (/tmp/medial-lateral-probe/*) are deleted. I reproduced the OA and meniscus figures from the two CSVs, but the report's meniscus margin AUC 0.971 does not reproduce (I get 0.959, discordant subset n=345) and its subset definition is unstated.
- Conditional degradation is OA-only: meniscus lateral AUC is 0.888 vs 0.890 across medial-positive/negative studies. Both proposals default to applying the mechanism to both families; the measured weakness supports OA. Family scope must be fixed and justified before launch, not chosen after results.
- conserved-evidence-compartment-assignment adds token-level parameters on four targets at once. A matched independent-module control (untied, non-conserved per-compartment token path of equal width) is needed before crediting shared morphology for a macro gain.
- Its budget is token-level (~10.5k tokens/step, three modules per family) and only estimated at ~0.3% of step time. Must be measured against the 1,797 s ceiling, not asserted.
- Its open_choices list is broad (aggregation form, attention source, gate input, detach, width, family scope). All must be fixed and recorded before launch; a later variation of them is not a new mechanism.
- cross-compartment-attribution-exclusivity makes the meniscus and OA margin diagnostics partial training targets, so those diagnostics cannot serve as independent confirmation of transfer for that arm. Its exclusivity premise also conflicts with stride-32 receptive fields spanning both compartments.

Evidence:
- Reproduced probe inputs: outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat/fold_0/validation_predictions.csv joined to data/folds/all_5fold_v3/targets.csv (904 studies); pair indices per data/folds/all_5fold_v3/protocol.json
- Proposal source report: outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260920T222419Z-48e1f1a0/report.json (both ideas, precedents, exhausted_directions)
- docs/validation/results/2026-09-20-compartment-adapter-result.md: rejected pooled shared adapter, meniscus family -0.0114 [-0.0200,-0.0030], margin -0.0191 [-0.0392,-0.0034]; prevalence domination of shared pooled parameters
- docs/validation/results/2026-09-14-meniscus-soft-coupling-result.md: isotropic classifier coupling, macro -0.0015, meniscus -0.0041
- src/knee_abnormality/modeling/model.py:204-240 (_attention_pool materialises per-study token membership, float32 block, per-target pooled vectors and logits) — attachment point both proposals rely on
- autoresearch/campaigns/losses/ideas.yaml shared-label-contrast-ranking (rejected) — confirms the pair-restricted main-logit margin variant is a measured-negative objective, correctly excluded
- autoresearch/campaigns/medial-lateral-transfer/prior-work.md#active-run-and-launch-readiness — control B reference and rejected-adapter provenance
- src/knee_abnormality/inference/checkpoint.py (non-editable) — constrains both proposals' parameter placement and constructor signature as their implementation notes state

## completion-20260920T225042Z-af6428b0

Training-only shared linear detection over the two pooled meniscus features at auxiliary weight 0.3 produced no transfer benefit and a small, consistently negative shift. Every label except Effusion (+0.0009) moved down; the largest single move was ACL -0.0130, a label the mechanism does not touch, which alongside family means (meniscus 0.9122->0.9099, oa 0.8882->0.8825, other eight 0.8948->0.8909) indicates a diffuse degradation of the shared trunk rather than a targeted meniscus effect. The two directly supervised labels moved least (Medial Meniscus -0.0004, Lateral Meniscus -0.0042), so the added task weight did not even help its own targets.

Compartment diagnostics point the same way: meniscus margin AUC on the 344 exclusive studies fell 0.9712 -> 0.9674 and lateral cross-talk among medial-positive studies rose 0.636 -> 0.711 (44 -> 45 lateral false positives, 28 -> 32 of them medial-positive); OA margin AUC fell 0.8661 -> 0.8568 with cross-talk 0.622 -> 0.634. Supports are small (63 lateral-only meniscus studies; 55 lateral-only OA), so these are directional, not decisive.

Mechanistic reading, held loosely: forcing one shared weight vector to score both compartments' pooled features adds gradient pressure toward a compartment-invariant direction without giving the model any new way to keep the two predictions distinct, and the cost appears to fall on the shared trunk. This is one implementation at one weight on one seed with a single fold-0 selection; the broader transfer hypothesis is untested by it, and the near-zero upper interval bound means "no detectable gain", not "harm established".

Completion interpretation only; no queue change and no proposals. The shared-disease auxiliary ran the full sealed budget (4,380 steps, 938,460 training triplets / 35,030 visits, 24,188 validation triplets, cohort hashes matching the sealed baseline) in 1,580.3 s against the 1,797 s ceiling, seed 3688760250, with only the three declared compartment_transfer config keys differing and preprocessing_identity the sole allowed input-identity difference. Macro delta -0.0039 with interval [-0.0081, +0.0003] contains zero, so the incumbent historical-control stands; nothing is admitted or removed.

The prior review's unresolved concerns are now moot for adoption: no gain was produced, so the independent-auxiliary matched control is not needed to defend an attribution claim, and the latent on_load_checkpoint/profiling-script defects do not persist because this source is not becoming the incumbent. Those defects should be avoided by any future candidate that places trained modules on KneeModule rather than inside KneeAbnormalityModel (auxiliary keys outside the model./averaged. prefixes are silently dropped on resume; build_optimizer's list signature breaks scripts/adhoc profilers).

The queue stays as is. The result does not favor or disfavor the remaining mechanisms, which all act on the inference path rather than through an auxiliary task weight, so the prepared pair-contrast readout remains the right head. No rerun of this idea: repeating it under another weight would be a coefficient sweep, not a new mechanism.

Unresolved concerns:
- Single seed and a single fold-0 checkpoint selection: the bootstrap interval is conditional on this fitted model and excludes seed and selection variance, so the uniformly negative per-label pattern is suggestive, not established.
- Compartment margin and cross-talk diagnostics rest on small exclusive-case supports (63 lateral-only meniscus, 55 lateral-only OA studies) and are unadjusted; they support the null reading but cannot carry it alone.
- The confound flagged at review (shared direction vs +0.3 meniscus task weight) remains unresolved; it is moot for adoption here but would still need an independent-auxiliary control if any future auxiliary-weight variant showed a gain.
- Latent defects in the candidate source are not adopted but are a design warning: KneeModule-owned trained modules are dropped by on_load_checkpoint's model./averaged. filter on resume, and build_optimizer's list signature breaks scripts/adhoc/profile_training.py:130,257 and profile_threaded_loading.py:268.
- Weight 0.3 was never justified; this result must not be used to motivate a coefficient sweep, which would not be a new mechanism.

Evidence:
- comparison/comparison.json (experiments/shared-disease-auxiliary-20260920T221219Z-2457a3ee): macro 0.896582 -> 0.892686, family_means, compartments block with meniscus margin_auc 0.971248 -> 0.967407 and lateral_cross_talk 0.636364 -> 0.711111, oa margin_auc 0.866116 -> 0.856834
- comparison paired_report_group_bootstrap: 2000 valid draws, delta mean -0.0039051, 95% [-0.0080932, +0.0003181]
- facts.timing: setup 0.309 s + training 1544.171 s + prediction 19.888 s = 1580.297 s measured compute vs 1,797 s ceiling; 4,380 optimizer steps; preparation 0.119 s
- facts.accounting: 10 epochs x 93,846 triplets / 3,503 visits, cohort b536b7e1... ; validation 24,188 triplets / 904 visits, cohort 7167eba1... ; matches sealed limits exactly
- comparison config_differences limited to the three model.compartment_transfer.shared_disease_auxiliary keys; input identity difference limited to preprocessing_identity
- Prior implementation review: jobs/orchestrator-20260920T222103Z-7f0cf276/report.json (unresolved concerns that triggered this escalation)
- comparison prediction_change: mean abs probability difference 0.0681, per-target Spearman 0.877-0.960, i.e. a real but modest reranking across all labels

## review-20260920T225614Z-0afc8512

New from this review, beyond the developer report: at true initialization only the evidence output layer receives gradient (saliency, gate and evidence hidden layers all have exactly zero gradient under an asymmetric medial-minus-lateral loss). The developer's claim that "a backward pass shows finite gradients reaching all module parameters, including saliency/gate" holds only once the evidence output weights are nonzero, i.e. from step 1 onward, not at step 0. This is the same bootstrap caveat as the running pair-contrast arm and bounds interpretation of a null result: a near-no-op outcome would be weak evidence about the mechanism itself. Also new: the added module file is not listed in identity.PREPROCESSING_SOURCES, while model.py is, so the split's internal math is not covered by preprocessing_identity; identity.py is outside the campaign's editable set, so the developer could not have fixed this. Provenance is still preserved by the full candidate source hash and snapshot, so I treat it as a recorded gap rather than a defect. Structurally the conserved split adds a shared per-family total that both compartments can share, whereas the concurrent pair-contrast arm can only redistribute existing pair mass; the two therefore probe different channels and neither is a matched capacity control for the other.

Approved after direct inspection of the diff, model.py call site, checkpoint paths, and independent CPU verification (no GPU, no training). Verified: exact initialization parity with the parent (max |Δlogit| = 0.0, evidence output layer zero-initialised); the eight non-pair labels are bit-exactly untouched; per-pair conservation holds, delta_medial + delta_lateral equals the compartment-agnostic shared total sum(pi*e) to fp32 rounding (<=7.5e-8); the masked softmax uses the same membership tensor as the existing attention, so a study's delta depends only on its own valid tokens (perturbing another study's token or an invalid token changes nothing, exactly 0.0) — no cross-study leakage and no change to mask accounting; the module is deterministic and stateless, so validation stays independent of the training seed; 33,542 parameters, matching the developer's figure. Checkpointing is clean: parameters live under the model.* prefix, checkpoint_model reconstructs them from the recorded config with strict=True and no shim, weight_averaging is null and attention_temperature is 1.0, so apply_attention_temperature does not desynchronize the split. Pair indices 2/3 and 4/5 confirmed against data/folds/all_5fold_v3/protocol.json (Medial/Lateral Meniscus, Medial/Lateral OA). All four sign patterns stay representable: base logits are unchanged and the added term can be positive for both compartments, unlike the concurrently running zero-sum pair-contrast arm, so the two arms test distinct mechanisms rather than duplicating one. No dataset, sampling, augmentation, label, optimizer or lightning.py change; preparation 0.170 s. Queue membership and order are unchanged and no proposals were submitted, so ranked_ids is null.

Unresolved concerns:
- Attribution: the split adds a shared nonlinear detector on four targets at once. A matched control with independent, per-compartment (untied) evidence heads of the same width is needed before crediting shared morphology for any macro gain; the older twelve-label residual_readout run is prior evidence, not that control.
- Runtime is unmeasured against the 1,797 s ceiling; negligibility is argued only from 33,542 parameters (~0.02%) and preparation 0.170 s. The module runs its three heads over all batch tokens each step.
- At initialization only the evidence output layer receives gradient; saliency, gate and evidence hidden layers bootstrap from step 1. An inconclusive result could reflect a near-no-op rather than a refuted mechanism.
- hidden_width=32, two pairs, and shared-across-pairs-independent parameters are asserted design choices, not measured. A later sweep of these would not be a new mechanism.
- The new module is absent from identity.PREPROCESSING_SOURCES (identity.py is not editable under the campaign policy), so preprocessing_identity does not cover the split's internals; provenance rests on the candidate source hash and snapshot.
- Undeclared but inert: an empty src/knee_abnormality/training/compartment_transfer directory inside the editable set.

Evidence:
- changes.diff: outputs/autoresearch/medial-lateral-transfer/experiments/conserved-evidence-compartment-assignment-20260920T225138Z-f55d4ff3/changes.diff (config.py:178-212, model.py:19,140-152,253-254, configs/training.yaml:87-96)
- New module: .../f55d4ff3/source/src/knee_abnormality/modeling/compartment_transfer/__init__.py:1-51 (saliency Linear(256,1), gate and evidence Linear(256,32)->GELU->Linear(32,1), zero-init on evidence[-1], masked softmax over membership, index_add onto pair logits)
- Call site context: .../f55d4ff3/source/src/knee_abnormality/modeling/model.py:219-256 (membership built from token_study & token_valid; split applied inside the existing autocast-disabled float32 block, before the finite-logit check)
- Recursive tree diff vs parent_source .../recovery/historical-source-20260920T215302Z-e086c1eb/source: only training.yaml, config.py, model.py differ; only additions are modeling/compartment_transfer and an empty training/compartment_transfer
- CPU verification (this review, /home/claudio/src/knee-abnormality/.venv/bin/python3): init parity 0.0; non-pair labels delta 0.0; |delta_m+delta_l - sum(pi*e)| <= 7.5e-8; other-study and invalid-token perturbations 0.0; repeat-call difference 0.0; 33,542 parameters
- Gradient probe at zero-init under a medial-minus-lateral loss: nonzero only for evidence.0.2.weight/bias; saliency and gate grads exactly 0.0
- Checkpoint path: .../f55d4ff3/source/src/knee_abnormality/inference/checkpoint.py:88-104 (model.* prefix, strict=True, no shim), config.py:26 ATTENTION_TEMPERATURE=1.0, configs/training.yaml:106 weight_averaging: null
- Target order: data/folds/all_5fold_v3/protocol.json -> ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion'], confirming pairs (2,3) and (4,5); manifest src/knee_abnormality/identity.py:16-30 lists PREPROCESSING_SOURCES without the new module

## completion-20260920T231804Z-6892337b

Pair-contrast readout (seed 963050827, fold 0, epoch-7 checkpoint): macro AUC 0.89361 vs reference 0.89658, paired bootstrap delta mean -0.00299, 95% interval [-0.00683, +0.00069] — inconclusive, incumbent retained.

New from this interpretation:
1. The mechanism's target channel regressed. Medial/lateral meniscus margin AUC on the 344 exclusive-positive studies dropped 0.9712 -> 0.9623 (margin interval [-0.0204, +0.0010]); OA margin +0.0025 with interval [-0.0266, +0.0308]. Since a zero-sum contrast can only reallocate mass within a pair, its own discrimination channel showed no gain.
2. Family means: meniscus -0.0105 (exploratory interval [-0.0194, -0.0015], the only family below zero), OA -0.0021, other-eight -0.0013. The negative pressure is concentrated on the four targets the module touches, not diffuse noise, though it is unadjusted and confounded with the single fresh training seed.
3. Cross-talk diagnostics worsened slightly for both families (meniscus lateral cross-talk 0.636 -> 0.644; OA 0.622 -> 0.667), consistent with (1).
4. The no-op worry is largely retired: trained contrast parameters are nonzero (out.weight norms 0.018 / 0.032 over hidden norms ~3.4), so a null result here reflects a weak-but-active mechanism rather than an untrained module.
5. Contract checks all pass: allowed config differences only, preprocessing_identity the only input-identity difference, exact sealed accounting, 2,000 valid bootstrap draws, 1585.28 s of 1797 s.

Scope limit: one seed, one fold; intervals exclude seed and checkpoint-selection variability. This refutes this implementation as an improvement candidate at best weakly (interval includes zero) and does not bear on shared-total or attention-level transfer mechanisms.

Completion interpretation only; no decision changes. The exchange-symmetric pair-contrast readout trained cleanly within contract (measured 1585.28 s vs 1797 s ceiling; sealed cohort/visit/triplet hashes and 4,380 steps reproduced exactly for all ten epochs; only the allowed model.compartment_transfer config keys and preprocessing_identity differ; fresh seed 963050827; candidate checkpoint from epoch 7 by the unchanged selection rule). Macro delta is negative-leaning but straddles zero, so the incumbent is retained and no idea is superseded.

Two review concerns are now resolved by evidence: runtime is measured and comfortably inside the ceiling, and the module was not a literal no-op — trained parameters exist (out.weight norms 0.018 and 0.032, hidden norms ~3.4), so learning did bootstrap past the zero-initialised output. The remaining concerns (attribution without a matched untied control, unmeasured width/detach choices, zero-sum design blind to co-occurrence) stay open but no longer gate anything, since nothing is being accepted.

The most informative diagnostic is that the mechanism's own channel got worse, not just the macro: meniscus margin AUC on exclusive-positive studies fell 0.9712 -> 0.9623 and the meniscus family mean fell 0.0105 (exploratory interval below zero), while OA margin moved +0.0025 with a very wide interval. Medial-vs-lateral discrimination — the only thing a zero-sum contrast can affect — did not improve, so this is not a case of a promising mechanism masked by unrelated-label noise. That is evidence against this specific implementation, not against the campaign hypothesis; I am not treating it as a reason to drop related mechanisms, and the concurrent conserved-evidence arm probes the complementary channel (shared per-family total) that this design cannot touch.

Queue membership and order stand: ranked_ids null, no proposals in this event, nothing parked.

Unresolved concerns:
- Meniscus family and meniscus margin AUC both moved down while the macro interval straddles zero; the exploratory family interval [-0.0194, -0.0015] cannot override the macro verdict but does argue against re-running this design unchanged.
- The candidate/reference contrast is confounded with a single fresh training seed and with checkpoint selection landing on epoch 7 vs 9; no seed-variation estimate exists for this campaign's reference.
- Attribution remains untested: no matched untied/independent nonlinear readout control was run, so neither the null macro nor the meniscus regression separates shared-morphology tying from added capacity.
- hidden_width=32, per-family independent rho and detach_inputs=false stay unmeasured design choices; a sweep of them is not a new mechanism and should not be queued as one.
- The zero-sum design cannot change a pair's total evidence, so this result says nothing about co-occurrence or pair-level calibration channels.
- Harness files under src/knee_abnormality/autoresearch/ appear in changed_source_archive_members relative to the pinned parent snapshot; they are orchestration-only and do not touch model/data behavior, recorded here as a provenance note.

Evidence:
- comparison/comparison.json: paired_report_group_bootstrap.intervals (all, meniscus, oa, other_eight, meniscus_margin, oa_margin), 852 groups, 2000 draws, seed 20260914
- comparison/comparison.json: compartments — meniscus margin_auc 0.9712478 (ref) vs 0.9622663 (cand); oa margin_auc 0.8661157 vs 0.8686586; lateral_cross_talk 0.6364->0.6444 and 0.6220->0.6667
- comparison/comparison.json: family_means and config_differences / input_identity_differences (preprocessing_identity only)
- comparison/comparison.json: parameters_only_in_run.candidate — contrasts.{0,1}.out.weight norms 0.018092 / 0.031577, hidden.weight norms 3.3599 / 3.4330
- experiments/compartment-pair-contrast-readout-20260920T222418Z-147e1fbd/training/fold_0/timing.json: setup 0.2984 s, training 1549.2666 s, prediction 19.8583 s (total 1569.42 s of 1797 s; measured_compute 1585.28 s)
- experiments/.../training/fold_0/selected_checkpoint.json: best-epoch=07-val_macro_auc=0.89361, score 0.8936086
- assignment facts.accounting: all ten epochs at 93,846 triplets / 3,503 visits, cohort b536b7e1...; validation 24,188 / 904, cohort 7167eba1..., 4,380 optimizer steps
- Prior implementation review with the open concerns now partly resolved: jobs/orchestrator-20260920T223030Z-dbcdff45/report.json

## review-20260920T232816Z-cd3b5142

New from this review, beyond the developer report: (1) The auxiliary is not only a transfer signal — it is also an extra supervised BCE term on the four compartment labels through their own pooled features, so it raises effective task weight on exactly the labels the campaign interprets. A matched control replacing the donor row with the recipient's own detached row (identical extra term, no cross-compartment direction) would separate shared morphology from extra task weight; the four scalar biases are too few for a capacity explanation, so capacity is largely ruled out here but task weight is not. (2) Gradient reaches pooled features from step 0 because the donor head row is nonzero at init, so an inconclusive result is interpretable evidence about the mechanism, unlike the concurrent conserved-evidence arm. (3) The auxiliary reduction is a mean over the four direction terms, each itself a masked mean; if a direction's recipient column is fully unobserved in a minibatch it is dropped and the remaining terms are renormalized, so the effective auxiliary weight per direction varies slightly across steps. Rare at batch size 8 but recorded. (4) New module files are not in identity.PREPROCESSING_SOURCES (identity.py is outside the editable set), so preprocessing_identity covers only the model.py call site; provenance rests on the candidate source hash and snapshot — the same recorded gap as the conserved-evidence arm. (5) weight=0.1 is an asserted, unswept choice; a later sweep of it would not be a new mechanism.

Approved for launch after direct inspection of changes.diff, the candidate tree and the parent's model.py/lightning.py. The mechanism is genuinely training-only and adds essentially no capacity (four scalar biases), which makes it an unusually clean test of transfer versus capacity — but it does add extra task weight on precisely the four compartment labels, so attribution of any macro gain needs that alternative kept in view (see concerns).

Correctness checks I ran against parent source rather than the developer's claims: `pooled` from `_attention_pool` is [batch, target, width] (einsum "btn,nf->btf"), so `pooled[:, index, :]` is the correct per-target vector and the auxiliary logit is constructed exactly like the main one with the donor row substituted; `class_balance_weights` is [2, T] (row 0 positive, row 1 negative), so `[:, recipient_index]` then `[0]`/`[1]` matches the main loss's weighting; the donor weight is detached, so no weight tying and no back-pressure on the other head's fitted direction.

Unlike the concurrent conserved-evidence arm, there is no initialization bootstrap problem: head weights are nonzero at init, so pooled features receive gradient from step 0 and a null result would be informative about the mechanism rather than about a near-no-op.

Queue membership and order are unchanged, so ranked_ids is null. No proposals in this event.

Unresolved concerns:
- Attribution: the auxiliary adds extra supervised weight on the four compartment labels as well as the cross-compartment direction. Crediting shared morphology for a macro gain needs a matched control using each recipient's own detached head row (same extra term, no transfer). Capacity is largely excluded (4 scalar biases).
- Auxiliary weight 0.1 and the equal-weighted four-direction mean are asserted design choices, not measured; a later sweep of the coefficient would not constitute a new mechanism.
- Direction terms whose recipient column is fully unobserved in a minibatch are dropped and the remainder renormalized, so per-direction auxiliary weight varies slightly step to step. Rare at batch size 8, but it is an unmeasured stochastic detail of the objective.
- New compartment_transfer modules are absent from identity.PREPROCESSING_SOURCES (identity.py not editable under campaign policy), so preprocessing_identity does not cover the readout/loss internals; provenance rests on the candidate source hash and snapshot.
- Runtime is argued, not measured: four width-256 dot products and four BCE terms per step on already-computed tensors. Negligible on its face against the 1,797 s ceiling, but confirmation only arrives with the run's timing.json.
- Two concurrent arms (this and conserved-evidence-compartment-assignment) both target the same four labels from the same reference; neither is a matched control for the other and their results must be interpreted separately, not combined.

Evidence:
- changes.diff (experiments/reciprocal-compartment-readout-transfer-20260920T231920Z-0ea5b894/changes.diff): full inspected diff, all four edited files plus two new packages
- parent source model.py:203-247 `_attention_pool`: pooled is [b,t,f] via einsum 'btn,nf->btf'; main logits = (pooled*head.weight).sum(-1)+bias — candidate adds only the `return_pooled` return
- candidate modeling/compartment_transfer/readout.py:1-46: four zero-init scalar biases, donor `head_weight.detach().float()`, float32 autocast-disabled block
- parent training/lightning.py:35-52 build_optimizer: iterates model.named_parameters(); new ndim-0 biases get head_lr with weight_decay 0
- parent training/lightning.py:80-118: class_balance_weight_tensor is [2,T] (positive row, negative row); candidate loss.py indexes [:, recipient_index] then [0]/[1] consistently
- candidate training/compartment_transfer/loss.py:1-51: per-direction masked BCE returning None on an all-unobserved column; components averaged
- identity.py:16-30 PREPROCESSING_SOURCES: lists modeling/model.py but no compartment_transfer modules
- diff -rq parent_source vs candidate source: only declared members differ (plus parent-side __pycache__); no undeclared change

## proposals-20260920T233138Z-69160601

I independently recomputed the researcher's reference diagnostics on CPU from the pinned predictions (sha d8de136..., 904 studies, joined 904/904 against data/folds/all_5fold_v3/targets.csv at threshold 0.5). All quoted numbers reproduce exactly: meniscus Spearman 0.483, own AUC 0.927/0.898, cross 0.596/0.711, exclusive counts 281/63, both 75; OA Spearman 0.881, own 0.908/0.869, cross 0.832/0.850, exclusive 143/55, both 186. Derived channels also reproduce: pair mean ranks any-compartment disease at 0.911 (meniscus) / 0.902 (OA); pair difference ranks side on exclusive studies at 0.968 / 0.862. Family-positive supports are 419 (meniscus) and 384 (OA).

This establishes two opposite regimes on the same reference: the OA pair's two scores are largely redundant detectors with a weak side channel, while the meniscus pair transfers almost nothing. It motivates level/lateralization re-coordination for OA and morphology sharing for meniscus, and it is the first campaign motivation grounded in measured reference behavior rather than analogy. It remains a single-run, single-fold diagnostic with no seed-variation estimate, so effect sizes derived from it are conditional; it cannot be used as an outcome criterion.

Confirmed against source: the frozen target order gives meniscus indices 2/3 and OA 4/5 (prediction header); tokens are features + position_1d + position_2d + plane + contrast normalized by the shared mil_norm (model.py:305-316), so proposal 2's value/key separation is implementable as described.

Both proposals admitted: dependencies empty, mechanisms inside the declared transfer/auxiliary policy, public output and recipe preserved, no restoration of the rejected pooled adapter or the withdrawn independent arm.

compartment-pair-channel-rotation placed directly behind the reserved head. I verified it is a real structural change, not a reparameterization: each public logit becomes a signed sum of two differently-attended readouts (pool_a from attention row a, pool_b from row b), which the current one-row-per-logit head cannot express, so neither model nests the other. It is capacity- and runtime-neutral, touches four interpreted targets, and is the only queued arm whose null result cannot be attributed to added capacity, added task weight or a zero-init bootstrap — the failure mode of both measured arms so far. Its call site is the existing float32 block in model.py:217-237 (residual_readout is false in the reference, so that branch is inert).

appearance-value-tied-meniscus-classifier ranked fourth rather than second. Its novel element (removing position_1d/position_2d from the classified content while attention keeps full geometry) is genuinely untested and capacity-negative, and it revisits the only historically positive sharing arm. But it constrains two of twelve targets, so it needs a large per-target effect to move macro, and it carries a real collapse risk if attention alone cannot separate compartments. anatomical-medial-axis-mirror-attention stays ahead of it: four targets, and it probes assignment, which the diagnostics identify as the OA deficit.

cross-compartment-attribution-exclusivity stays last; no new evidence bears on it.

Unresolved concerns:
- compartment-pair-channel-rotation: the shared level channel couples the pair's errors, so both compartment AUCs can move down together; the claim that the other eight targets are unaffected holds only at initialization, since backbone gradients change from step 1.
- compartment-pair-channel-rotation: applying it to both families in one run mixes two opposite measured regimes (meniscus Spearman 0.483 vs OA 0.881); a null macro result may hide opposing family effects. Evaluator family means and margins must be read per family before any conclusion.
- appearance-value-tied-meniscus-classifier: reusing mil_norm for the geometry-free value means its shared affine parameters receive gradient from two differently-composed token tensors, so the other ten targets' normalization is not strictly unchanged during training, contrary to the proposal's wording.
- appearance-value-tied-meniscus-classifier: it adds one LayerNorm pass over all batch tokens plus a second pooling einsum. Argued negligible, unmeasured against the 1,797 s ceiling; also constrains only two of twelve targets, so a real family gain can still read inconclusive at macro.
- Neither proposal is self-attributing if it wins: the rotation's gain could come from per-logit access to two attention readouts rather than from shared detection, and the tie's gain could come from the removed positional shortcut rather than from shared morphology. Matched controls remain unrun.
- No seed-variation estimate exists for the reference, so all family, margin and co-occurrence diagnostics used to motivate these arms stay conditional on one fold-0 run.

Evidence:
- Researcher report: outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260920T231920Z-5a72035a/report.json
- Diagnostics reproduced by me from outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat/fold_0/validation_predictions.csv joined to data/folds/all_5fold_v3/targets.csv (904/904 studies; values listed in finding)
- Target order: validation_predictions.csv header -> indices 2/3 = Medial/Lateral Meniscus, 4/5 = Medial/Lateral OA; pairs also in src/knee_abnormality/evaluation/metrics.py:6-9
- Call site for the rotation: src/knee_abnormality/modeling/model.py:217-239 (float32 autocast-disabled block); configs/training.yaml:86 residual_readout: false
- Token composition for the geometry-free value: src/knee_abnormality/modeling/model.py:305-316 (features + position_1d + position_2d + plane + contrast, then shared mil_norm at model.py:110)
- Prior negative evidence cited and relied on: docs/validation/results/2026-09-20-compartment-adapter-result.md; campaign results compartment-pair-contrast-readout-20260920T222418Z-147e1fbd and shared-disease-auxiliary-20260920T221219Z-2457a3ee
- Historical positive sharing arm: docs/research/medial-lateral-transfer.md#the-six-relevant-experiments (1A/1B meniscus +0.002737 / +0.003406, margin 0.980173 vs 0.972321)

## completion-20260920T234557Z-829311f3

The conserved evidence split is not a no-op and is disfavored as a mechanism. Learned parameter norms (gate output 0.406/pair, saliency 0.62, evidence output 0.0306 from zero init) and a 0.066 mean absolute probability shift show it trained and altered predictions; the earlier step-0 zero-gradient caveat therefore does not excuse the null. Losses concentrate on the four targeted labels (meniscus -0.0114, OA -0.0067, other eight -0.0005) and on compartment separation (meniscus margin AUC -0.0145 [-0.0283, -0.0034]; meniscus lateral cross-talk 0.636 -> 0.660), which is the opposite of the intended effect: sharing a conserved per-family total plus a competing assignment gate degraded the model's ability to lateralize, without buying family-level accuracy. The epoch curve is below the reference at all ten epochs, so this is a persistent optimization effect rather than a late-training artifact. This bears on the campaign hypothesis only for this channel - a shared scalar total with a conserved split - and does not refute shared morphology generally; training-only and non-conserved sharing mechanisms remain untested. Because the outstanding attribution concern (a matched untied-evidence-head control) only mattered for crediting a gain, it lapses with the negative direction; no control run is warranted. The identity gap noted earlier persists: the new module is outside identity.PREPROCESSING_SOURCES, so preprocessing_identity does not cover the split's internals and provenance rests on the candidate source hash and snapshot.

Interpretation of the conserved-evidence completion (macro delta -0.00332, 95% [-0.00714, +0.00047]; incumbent retained). Contract checks pass: 1,596.6 s vs the 1,797 s ceiling, 4,380 steps, exact sealed cohort/triplet counts, fresh seed 1091236557, only the three declared config keys and preprocessing_identity differ.

The one prior review concern that could have voided interpretation is resolved: the module trained rather than staying a near-no-op. The candidate's extra parameters have substantial learned norms (gate output weight 0.406 per pair, saliency weight ~0.62, evidence output 0.0306 from zero init), and mean absolute probability change is 0.066 with per-target Spearman 0.89-0.96. So the step-0 gradient bootstrap caveat did not neutralize the mechanism; this is informative evidence about the conserved split itself.

Direction of the evidence is consistently against the mechanism on exactly the targets it acts on: meniscus family mean -0.0114 [-0.0196, -0.0030], OA family -0.0067 [-0.0140, -0.0001], other eight -0.0005 [-0.0053, +0.0043]. Compartment discrimination degrades too: meniscus margin AUC 0.9712 -> 0.9568, delta -0.0145 [-0.0283, -0.0034]; OA margin 0.8661 -> 0.8387, delta -0.0272 with a wide interval. Meniscus lateral cross-talk rises 0.636 -> 0.660. The candidate's per-epoch macro curve stays below the reference at every epoch and selects epoch 10, consistent with the split competing with the existing per-target attention rather than adding usable shared evidence. These label/family/margin figures are exploratory and unadjusted; they cannot change the macro verdict, but they do bound the mechanism's plausibility.

Ranking change (minimal): moved cross-compartment-attribution-exclusivity ahead of the two hard-tying architectures. Forcing the four compartment logits through a shared structural channel measurably hurt compartment margins here, whereas the exclusivity idea is a training-only auxiliary that leaves the inference path and the twelve independent readouts intact. Queue membership is unchanged; the reserved head stays first.

Unresolved concerns:
- preprocessing_identity does not cover src/knee_abnormality/modeling/compartment_transfer/__init__.py (identity.py is outside the campaign's editable set); provenance for this run rests on the candidate source hash and source_snapshot.zip only. Unresolved and will recur for every new module in this campaign.
- One seed, one fold: the negative family and margin deltas exclude training-seed and checkpoint-selection variability, so the mechanism is disfavored rather than refuted.
- hidden_width=32 and the two-pair choice were never measured; a width/coefficient sweep would not be a new mechanism, so this result should not be rerun as a sweep.
- changed_source_archive_members lists several src/knee_abnormality/autoresearch/* files versus the reference snapshot; these are harness modules outside the training path, but they are not part of the declared integration.

Evidence:
- comparison/comparison.json: paired_report_group_bootstrap.intervals - all [-0.007139, +0.000474], meniscus [-0.019583, -0.002971], oa [-0.014024, -0.000104], other_eight [-0.005335, +0.004332], meniscus_margin [-0.028339, -0.003370], oa_margin [-0.064212, +0.008184]; 852 groups, 2000 draws, seed 20260914
- comparison/comparison.json: compartments - meniscus margin_auc 0.9712 (ref) -> 0.9568 (cand), lateral_cross_talk 0.6364 -> 0.6600; oa margin_auc 0.8661 -> 0.8387
- comparison/comparison.json: runs.candidate.epochs - macro AUC below reference at all ten epochs, selected_epoch 10 (0.89325) vs reference epoch 9 (0.89658); parameters 14,633,764 vs 14,600,222
- comparison/comparison.json: parameters_only_in_run.candidate - gate.*.2.weight norms 0.4058/0.4069, saliency.*.weight 0.622/0.610, evidence.*.2.weight 0.0306/0.0295 (zero-initialised at start)
- comparison/comparison.json: prediction_change.mean_abs_probability_difference 0.0656, per_target_spearman 0.887-0.962
- facts.timing: 0.291 setup + 1559.168 training + 20.734 prediction = 1596.6 s measured against the 1,797 s ceiling; 4,380 optimizer steps; preparation 0.170 s
- facts.accounting: ten train epochs at 93,846 triplets / 3,503 visits, cohort b536b7e1...; ten validation epochs at 24,188 triplets / 904 visits, identical selection hash 92f1341c... each epoch (validation independent of the training seed)
- Prior implementation review: outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260920T225614Z-dcef5cb7/report.json (step-0 gradient and identity-coverage concerns now updated by this run's learned-parameter evidence)

## review-20260920T235409Z-fe7f86b9

New beyond the developer report: (1) The transfer is real, not a reparameterization. Because each target has its own attention scorer and pooled vector (model.py:223-247), mixing two logits ties two distinct attention branches; I verified the gradient of the medial logit reaches both branches with weight 1/sqrt(2). Operationally one branch is supervised by the pair's summed evidence (shared morphology) and the other by the side difference. (2) Attribution is unusually clean: parameter count and supervision are unchanged, so capacity and task-weight explanations are excluded by construction; a win or loss is attributable to the inductive bias of the tying itself, not to added machinery. The flip side is that it is also an optimization-geometry change (Adam is not rotation-invariant), so "shared morphology helped" and "the reparameterized head trained better" are not separable by this arm alone. (3) The assignment of the "+" combination to the medial target and "-" to the lateral target is an arbitrary but harmless convention; since both pre-rotation branches are free, no compartment is privileged. (4) An empty src/knee_abnormality/training/compartment_transfer/ directory exists in the workspace with no files and no manifest entries; inert, recorded only for provenance. (5) pair_rotation.py is not in identity.PREPROCESSING_SOURCES (identity.py is outside the editable set), so preprocessing_identity covers only the model.py call site; provenance rests on the candidate source hash and snapshot, the same recorded gap as the concurrent arm.

Approved for launch. The candidate implements the assigned mechanism, preserves the contract, and is the cleanest attribution arm so far: no added parameters, no added task weight, no added supervision, no dataloader or preparation change, so a macro change cannot be explained by capacity or by extra weight on the four compartment labels. Target indices 2-5 were verified against data/folds/all_5fold_v3/protocol.json (ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, ...), so the configured pairs match the frozen order. Representability holds: the rotation is an orthonormal bijection of two unconstrained real logits, so neither, medial-only, lateral-only and both-positive all remain expressible; no label exchange occurs and augmentation is untouched. One logits path exists (pool_encoded -> _attention_pool), so training, validation and prediction all use the rotation consistently; public output is still twelve logits at the same label positions. Validation remains deterministic and seed-independent (no edits to lightning.py or the dataset). No unresolved gate. Queue unchanged; no proposals in this event.

Unresolved concerns:
- Interpretation limit: with no added parameters or supervision, a macro change is attributable to the tying, but shared-morphology transfer cannot be separated from the pure optimization-geometry effect of rotating the head's output basis under Adam. A matched control rotating an unrelated non-paired target couple would separate these if this arm moves the macro interval.
- Runtime is argued, not measured: two index_select and two index_copy over twelve logits per forward. Negligible on its face against the 1,797 s ceiling, but confirmation only arrives with the run's timing.json.
- New pair_rotation.py is absent from identity.PREPROCESSING_SOURCES, so preprocessing_identity does not cover the rotation internals; only the model.py call site is hashed. Provenance rests on the candidate source hash and source snapshot.
- The fixed 45-degree angle is asserted, not measured. A later sweep of the angle or of which pairs are rotated would be a parameter search, not a new mechanism, and must not be admitted as one.
- An empty training/compartment_transfer package directory is present in the candidate workspace with no files and no manifest entries. Inert, but it is undeclared residue in the integration plan.
- This arm and the concurrently running reciprocal-compartment-readout arm target the same four labels from the same reference; neither is a matched control for the other and their results must be interpreted separately.

Evidence:
- changes.diff (experiments/compartment-pair-channel-rotation-20260920T234658Z-e15e33a7/changes.diff): full declared surface, config.py:175-221, model.py:138-160 and 254-259
- source/src/knee_abnormality/modeling/model.py:223-261 - twelve per-target attention distributions and per-target head rows; single logits path, rotation inside the float32 block before the isfinite check
- source/src/knee_abnormality/modeling/model.py:328-336 - pool_encoded is the only caller, so training, validation and prediction share the rotated path
- source/src/knee_abnormality/modeling/compartment_transfer/pair_rotation.py:1-22 - zero-parameter module, two non-persistent long buffers
- Direct check with the supplied python_executable: rotation output matches (a+b)/sqrt2 and (a-b)/sqrt2 on indices 2-5, d(logit2)/d(inputs 2,3) = (0.7071, 0.7071), module state_dict() is empty
- data/folds/all_5fold_v3/protocol.json targets = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',...], confirming level_index/lateral_index 2/3 and 4/5
- Recursive diff vs parent_source: only configs/training.yaml, config.py, modeling/model.py differ; modeling/compartment_transfer added; training/compartment_transfer added but empty; lightning.py, dataset.py, preparation.py, spatial_tokens.py, augmentation.py, inference/checkpoint.py identical
- Developer report jobs/developer-20260920T234658Z-2615f4d6/report.json - parameter count 14,517,388 identical with and without rotation; ruff clean; load_state_dict(strict=True) succeeds

## completion-20260921T001333Z-4f3961c5

The auxiliary clearly changed the model rather than being inert: mean absolute validation probability difference 0.067 and per-target Spearman 0.85–0.96 against the reference, with nonzero learned auxiliary biases (norms 0.005–0.030). Despite that, it produced no compartment benefit. Family means: meniscus 0.9122→0.9078, OA 0.8882→0.8885, other-eight 0.8948→0.8907; macro 0.8966→0.8932, paired delta mean −0.0034 with 95% interval [−0.00744, +0.00035] — inconclusive under the fixed rule but with essentially all mass below zero. The targeted compartment diagnostics do not move in the predicted direction either: meniscus margin AUC 0.97125→0.97136 and lateral cross-talk 0.636→0.640 (344 exclusive studies); OA margin AUC 0.8661→0.8614 with cross-talk 0.622→0.605 (198 exclusive studies). So the one directional hint (slightly lower OA cross-talk) comes with a worse OA margin and is well inside noise at those supports. The largest per-label losses are on labels the mechanism does not touch (ACL −0.0115, MCL −0.0138), while Effusion gains +0.0094; all twelve per-label intervals cross zero. That pattern is more consistent with reallocated optimization/seed variation than with compartment transfer. Because this arm is a single fresh-seed run whose bootstrap is conditional on the fitted model, the macro shortfall cannot be attributed to the mechanism specifically; equally, the absence of any compartment-margin or cross-talk improvement is the more informative negative signal, since those diagnostics are where the mechanism should act first.

Completion interpretation only; no decision changes. The queue membership and order are unchanged and no proposals were presented, so ranked_ids is null and accepted_proposal_ids is empty. The idea is not parked: a single inconclusive, seed-confounded arm does not refute the reciprocal-readout mechanism, but it also gives no reason to promote related designs, so the existing order (reserved head first) stands. The incumbent remains historical-control; nothing here creates a campaign acceptance. Contract checks I could verify from the artifacts pass: only the two declared config keys differ beyond name/output_root/seed; input identity differs only in the permitted preprocessing_identity; per-epoch cohort hashes, 938,460 training triplets, 35,030 visits, 24,188 validation triplets, 4,380 steps and the fixed validation selection hash across all ten epochs match the sealed values; measured compute 1,592.3 s is inside the 1,797 s ceiling; the run used a fresh seed (583093732). The four auxiliary bias parameters are the only new weights in the candidate checkpoint, confirming the training-only, capacity-neutral construction as declared. The prior review's unresolved concerns (attribution against extra task weight, unswept weight=0.1, renormalized direction terms, identity coverage of the new modules) are unaffected by this result and remain open; the runtime concern is now closed by timing.json.

Unresolved concerns:
- Result is one fresh-seed run; the bootstrap excludes training-seed and checkpoint-selection variation, so the −0.0034 macro shortfall and the near-zero compartment diagnostics cannot be separated from seed noise. Do not treat this as evidence against the broader shared-pattern hypothesis.
- Attribution concern from the implementation review stays open and is now moot for this arm: no macro gain to attribute, so the matched own-row control is only worth building if a later reciprocal design shows a gain.
- Compartment margin AUC and lateral cross-talk are computed on 344 (meniscus) and 198 (OA) exclusive studies with no intervals recorded in comparison.json; their small shifts are not quantified uncertainty and must not be read as directional evidence.
- changed_source_archive_members includes many src/knee_abnormality/autoresearch/* harness files alongside the experiment edits; these are outside the training path but mean the archive diff is not a clean view of the candidate's model changes.
- The new compartment_transfer modules remain outside identity.PREPROCESSING_SOURCES, so provenance for this run rests on the candidate source hash and snapshot rather than preprocessing_identity.
- weight=0.1 and the equal-weighted four-direction mean remain unmeasured; a coefficient sweep would not be a new mechanism and must not be admitted as one to revisit this arm.

Evidence:
- comparison.json (experiments/reciprocal-compartment-readout-transfer-20260920T231920Z-0ea5b894/comparison/): family_means, compartments, prediction_change, config_differences, input_identity_differences, parameters_only_in_run
- comparison/per_label.json: all twelve paired intervals cross zero; ACL −0.0115, MCL −0.0138, Effusion +0.0094, Medial OA +0.0043
- Assignment facts: measured_compute_seconds 1592.288 vs max_run_seconds 1797; seed 583093732; accounting complete with sealed triplet/visit counts and cohort hashes
- Prior implementation review: outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260920T232816Z-9d3ae9b0/report.json (open attribution, weight, identity-coverage concerns)
- Developer report: outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260920T231920Z-6c0c5c31/report.json

## review-20260921T002445Z-3643b2c9

New beyond the developer report: (1) The penalty's gradient acts on the main logit as well as the reattributed one — d/dz of relu(z−z′−margin)² is positive, so for positive pair cells it pushes z down while pushing z′ up. Within-label AUC is invariant to uniform shrinkage, but this shrinkage is selective (only where erasure costs more than the margin), so part of any macro movement could be a calibration/logit-geometry effect on the four paired labels rather than attribution exclusivity per se. (2) The mechanism is an exclusivity constraint, i.e. it regularizes *against* one compartment borrowing the other's spatial evidence while leaving backbone and feature sharing intact; a macro win therefore supports "shared features with disjoint attribution", not "more sharing", and interpretation must not conflate the two. (3) Observed masks are all-or-nothing per study (labels.py:165-175), which both removes the new guard's crash risk and means the auxiliary term is active on essentially every batch, so its compute cost is incurred at every one of the 4,380 steps. (4) Added cost is 4 reattributions per step, each one (B,N)×(N,256) einsum plus a head application, ≈1/3 of the existing pooling einsum and negligible against the backbone, but it also carries a backward pass and is unmeasured on GPU. (5) `scripts/adhoc/diagnose_encoding_usage.py:206` calls `_attention_pool` positionally with four arguments and still works unchanged under the new default kwarg.

Approved for launch. The candidate implements the assigned mechanism rather than a proxy: for each pair it builds a detached erasure mask from the other compartment's attention (1 − other/max other), reweights and renormalizes the compartment's own attention, repools the same tokens and rescores through the identical head, then penalizes target-weighted relu(z − z′ − margin)² over observed pair cells (model.py:204-249, objective.py:14-34). Contract checks I verified directly: no extra encoder pass (one `encode_batch` + one pool call, lightning.py:314-317); the reattribution runs inside the existing float32 autocast block on already-computed attention and tokens; `other` is detached, so the penalty cannot be satisfied by moving the other compartment's attention; non-member tokens carry exactly zero own-attention after the −inf softmax, so no cross-study token leaks into the reattributed pool; no RNG-consuming operation is added, so sampling order, triplet counts and cohort identity are untouched (only `preprocessing_identity` moves, via model.py, which the campaign permits). Both-positive, single-compartment and neither cases remain representable: the twelve scorers and head are unchanged and the penalty only acts on positive-weighted observed cells. The new guard in `cross_compartment_penalty` for a batch with no observed pair cell is not a new crash risk: `observed_{target}` is all-or-nothing per study (labels.py:165-175), so the condition coincides with the pre-existing BCE guard (lightning.py:114). Queue membership and order are unchanged; no proposals in this event.

Unresolved concerns:
- Attribution limit: the penalty applies only to the four paired labels but regularizes the shared backbone, so a macro interval above zero may be driven by the other eight labels. Interpretation must use the compartment margins and family means, not the macro alone, before claiming the transfer mechanism worked.
- The penalty suppresses positive-class main logits as well as raising reattributed ones. A separate auxiliary-weight/logit-geometry explanation is not excluded by this arm; a matched control penalizing an unrelated non-paired label couple would be needed to attribute a gain to compartment exclusivity specifically.
- Runtime is argued from op counts, not measured (no GPU access at development time). Four extra pooling einsums plus backward per step on every one of 4,380 steps should be far inside the 1,797 s ceiling, but confirmation only arrives with the run's timing.json.
- lambda_weight=0.05 and margin=1.0 (logit units) are asserted defaults, never measured. A later sweep of either is a parameter search, not a new mechanism, and must not be admitted as one.
- Provenance gap: training/compartment_transfer/objective.py is outside identity.PREPROCESSING_SOURCES (identity.py is not editable here), so preprocessing_identity covers only the model.py call site. Provenance for the objective rests on the candidate source hash and source snapshot — the same recorded gap as the concurrent channel-rotation arm.
- An empty src/knee_abnormality/modeling/compartment_transfer/ package directory remains in the candidate source; inert and unhashed, but undeclared residue in the integration plan.
- This arm and the running compartment-pair-channel-rotation arm act on the same four labels from the same reference; neither is a matched control for the other and their results must be interpreted separately.

Evidence:
- changes.diff and source at outputs/autoresearch/medial-lateral-transfer/experiments/cross-compartment-attribution-exclusivity-20260921T001439Z-46286939/ (candidate_source_sha256 de63e805..., config 876c7800...)
- modeling/model.py:204-249 _score_pooled_for_target/_reattributed_logit/_masked_reattribution_logits; model.py:250-294 _attention_pool with optional transfer_pairs, returning logits alone when None; model.py:379-390 pool_encoded / pool_encoded_with_transfer / forward
- training/compartment_transfer/objective.py:14-34 cross_compartment_penalty and resolve_transfer_pairs
- training/lightning.py:314-343 training_step transfer branch (single encode_batch + single pool call, BCE unchanged); lightning.py:108-122 masked BCE guard; lightning.py:440-450 validation_step on the unmodified forward path
- data/labels.py:165-175 observed_{target} = target not null, assigned per study for all twelve targets
- config.py:223-249 CompartmentTransferPairConfig/CompartmentTransferConfig (default disabled, enabled requires pairs); configs/training.yaml trainer.compartment_transfer (lambda 0.05, margin 1.0, meniscus and OA pairs)
- recursive diff against parent_source: only configs/training.yaml, config.py, modeling/model.py, training/lightning.py differ; modeling/compartment_transfer/ empty, training/compartment_transfer/ contains __init__.py and objective.py
- developer report outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T001439Z-c869b83b/report.json (torch.equal identity check of the default path; ruff/py_compile clean; compute estimate unverified on GPU)

## proposals-20260921T002616Z-2a44156b

Verified the researcher's seed-floor claim at source. Runs 00-control and 13-control-repeat differ in exactly {seed, name} across their flattened effective configs (1133376617 vs 2263738815); their selected-checkpoint macro AUCs are 0.8932217 and 0.8965817, a 0.00336 gap from seed alone, with the reported paired bootstrap interval [-0.00707, +0.00022] under the campaign rule.

Two consequences for interpretation, beyond the researcher's statement. First, this is a single seed pair: it establishes that seed variation of this magnitude occurs, not a calibrated floor or its distribution. Treating 0.0034 as a threshold overstates what one replicate supports; it should be used as a caution on reading near-zero deltas, not as a decision rule competing with the fixed macro interval.

Second, and more consequential: the pinned reference is the higher of the two draws. Every candidate compared against it therefore faces a reference whose selection is correlated with a favorable seed draw, biasing candidate-minus-reference deltas downward by an amount of the same order as the effects being sought. The bootstrap interval is conditional on the fitted models and cannot express this. This does not change the contract or the fixed decision rule, and the reference stays pinned, but it means past and future inconclusive readings in this campaign carry less evidential weight against their mechanisms than their intervals suggest.

Accordingly, the completed arms whose deltas lie in -0.0030 to -0.0039 have untested mechanisms. This is a reason not to cite them as negative evidence; it is not a reason to rerun any of them unchanged.

Both proposals are admitted: dependencies are empty, each is training-only/zero-parameter in its default form, each states a mechanism tied to the campaign scope (compartment representation geometry; cross-family anatomical assignment), and neither restores the rejected pooled adapter or its withdrawn independent arm.

Ranking changes below the reserved head, driven by the researcher's two new measurements that I checked at source (seed-only config diff and the two runs' selected-checkpoint scores, see evidence):

compartment-invariant-disease-embedding to position 2. It is the only proposal that targets the measured cross-compartment deficit (medial score ranking lateral meniscus labels at AUC 0.597 with only modest co-occurrence) at the level the in-sample two-logit refit ceiling leaves open. Its gradient reaches the shared trunk, which matters because four-label gains are divided by twelve and must clear a seed-scale delta of roughly 0.0034.

cross-family-mediolateral-axis-agreement to position 4, below anatomical-medial-axis-mirror-attention. Its premise — that the two families currently disagree on the mediolateral axis — is unverified without a forward pass, so inertness (a near-zero penalty from step 0) is a live risk, and its predicted gain concentrates on the OA pair, the hardest route through the macro rule. The mirror-attention idea carries a stronger, verifiable inductive bias and does not depend on an unchecked premise.

appearance-value-tied-meniscus-classifier drops to last: it is the closest surviving relative of historical tied-classifier work (1A macro +0.000217) and of the rejected pooled adapter's readout-level family, and its effect is largely expressed through the pair's own channels, where the newly measured oracle recombination ceiling is about +0.001. It stays queued rather than parked because its gradient does reach pooled features and the historical comparisons used the archived recipe.

No reranking of the reserved head; both currently active experiments are unaffected.

Unresolved concerns:
- Reference-selection bias: the pinned reference is the higher of two seed-only draws, so candidate-minus-reference deltas are biased downward by roughly the seed-scale effect being sought. Bootstrap intervals cannot express this. Unresolved; the contract and reference stay fixed.
- The 0.0034 seed figure rests on one seed pair. It bounds nothing formally and must not be applied as a second acceptance threshold alongside the fixed macro rule.
- compartment-invariant-disease-embedding risks trading lateralization for alignment: pulling medial-positive toward lateral-positive pooled vectors could reduce within-study meniscus margin AUC (0.9712) even while cross-AUC rises. Both must be read together; a cross-AUC rise with a margin fall is not success.
- Its lambda, temperature, queue length K, family scope and both-positive anchor handling are all open and unmeasured. They must be fixed and recorded before launch; a later sweep is a parameter search, not a new mechanism.
- cross-family-mediolateral-axis-agreement may be inert if the two families' attention displacements already agree beyond the margin; the penalty value and its share of the objective must be logged from step 0 or the run measures nothing.
- Its detached meniscus sign anchor is undefined when the meniscus displacement is near zero, in which case the hinge pushes OA attention in a noise-determined direction; and because it trains a quantity related to the evaluator's margin diagnostic, that diagnostic is not independent for this arm.

Evidence:
- Researcher report: outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T001439Z-c9854a91/report.json (both proposals plus exhausted_directions seed-floor and logit-recombination-ceiling probes)
- Verified seed-only config diff: flattened effective_config.json of overnight-20260919T235434Z-8ol_1c30/runs/00-control vs runs/13-control-repeat differs in exactly {seed: 1133376617 vs 2263738815, name}
- Verified scores: runs/00-control/fold_0/selected_checkpoint.json score 0.8932217359542847 (epoch 06) vs pinned reference 0.8965816694263277
- Campaign prior work: autoresearch/campaigns/medial-lateral-transfer/prior-work.md (1A macro +0.000217; 1C local 0.892538 vs 0.892360; rejected pooled adapter and its retired independent arm)
- Rejected adapter result: docs/validation/results/2026-09-20-compartment-adapter-result.md; run outputs/training/compartment-control, predictions 753249bf...
- Evaluator margin definitions used by both proposals' exploitation claims: src/knee_abnormality/evaluation/metrics.py:93-111 (logit_margin, margin_auc, COMPARTMENT_PAIRS)
- proposal_admission in this assignment: both new ideas have dependencies_satisfied=true with no unresolved dependencies; active_incumbent_ideas is empty, so both are pure additions on historical-control

## completion-20260921T004101Z-22c94d48

Contract checks pass on the evidence: parameter sets identical between reference and candidate (comparison.json parameters_only_in_run both empty), accounting exact and complete (938,460 training triplets / 35,030 visits, 24,188 / 904 validation, matching sealed hashes, 4,380 optimizer steps), measured compute 1,582.10 s inside the 1,797 s ceiling, fresh seed 2451327761, and the only identity difference is the allowed preprocessing_identity plus the declared config key.

Interpretation of the exact numbers: macro 0.89658 -> 0.89404, delta -0.00255, 95% paired interval [-0.00657, +0.00121] — inconclusive, incumbent retained, but the sign pattern is informative against the mechanism rather than neutral. Every one of the four targeted labels moved down (Lateral Meniscus -0.0121, Medial Meniscus -0.0060, Medial OA -0.0028, Lateral OA -0.0024), family means meniscus -0.0091 and OA -0.0026, while the other eight moved -0.0009 with the only visible positives on unrelated labels (Baker's +0.0112, Effusion +0.0050). All per-label intervals straddle zero and are unadjusted, so none of this is individually significant, but no part of the small macro shortfall comes from the labels the mechanism was supposed to help.

Compartment diagnostics are mixed and small-support, giving no coherent lateralization story: meniscus margin AUC 0.9712 -> 0.9662 on 344 exclusive studies with lateral cross-talk 0.636 -> 0.591; OA margin AUC 0.8661 -> 0.8735 on 198 exclusive studies with cross-talk 0.622 -> 0.635. Margin and cross-talk move in opposite directions within each family and in opposite directions across families.

Conclusion: forcing each pair's outputs into a fixed sum/difference (level/lateralization) basis supplies no usable shared-morphology transfer at this budget. This refutes the specific fixed-basis implementation, not the campaign hypothesis.

Queue membership and order stay as they are: this result gives no reason to promote or demote any pending mechanism. The rotation arm is a parameter-free head-basis change and its failure to move the macro interval does not bear on the queued attention-geometry, contrastive-alignment, or tied-classifier mechanisms, which supply different transfer. No proposals in this event and none pending, so nothing to admit.

The prior-review concerns about attribution ambiguity (shared morphology vs. Adam's non-rotation-invariance) are now moot for adoption: there is no gain to attribute, so the proposed matched "rotate an unrelated couple" control is not worth a full-budget slot. The two provenance/residue concerns (pair_rotation.py outside identity.PREPROCESSING_SOURCES; empty training/compartment_transfer directory) remain recorded and unresolved but did not affect this run's validity — the source hash and snapshot cover the candidate, and the residue is inert.

One registry-level consequence to note: the fixed 45-degree level/lateralization basis is measured and did not help. A follow-up varying the angle, the scaling, or which pairs are rotated would be a parameter search over a tested mechanism, not a new mechanism, and should not be admitted as one.

Unresolved concerns:
- Outcome is inconclusive by the fixed rule, but every targeted label and both family means moved down while the only positives are unrelated labels (Baker's, Effusion). The fixed 45-degree basis should be treated as measured and unhelpful, not as awaiting a rerun.
- Compartment diagnostics do not corroborate any lateralization improvement: meniscus margin AUC fell (0.9712->0.9662) while its cross-talk improved (0.636->0.591), and OA moved the opposite way on both (0.8661->0.8735, 0.622->0.635), on 344 and 198 exclusive studies respectively.
- Unresolved from review and unaffected by the run: pair_rotation.py is outside identity.PREPROCESSING_SOURCES, so preprocessing_identity hashes only the model.py call site; provenance rests on the candidate source hash and snapshot.
- Unresolved from review: an empty src/knee_abnormality/training/compartment_transfer/ directory remains in the candidate workspace, undeclared in the integration plan. Inert, no manifest entries.
- A sweep of the rotation angle, scaling, or the set of rotated pairs is a parameter search over this tested mechanism and must not be admitted as a new mechanism.
- Uncertainty is conditional on one seed, one fold and one checkpoint selection; the negative direction on the four paired labels is suggestive only and does not refute the broader shared-pattern hypothesis.

Evidence:
- comparison/comparison.json: macro 0.8965816694263277 -> 0.8940410606828978, bootstrap delta_mean -0.0025525823528833136, 95% [-0.006570840967406083, 0.0012116338380518126], 2000 valid draws
- comparison/comparison.json family_means: meniscus 0.9121951795188479 -> 0.9031220110213811, oa 0.8881828482522477 -> 0.8855975039809696, other_eight 0.8947779971967176 -> 0.893881712273759
- comparison/comparison.json compartments: meniscus margin_auc 0.9712478111054623 -> 0.9661639270180196, lateral_cross_talk 0.6363636363636364 -> 0.5909090909090909 (344 exclusive studies); oa margin_auc 0.8661157024793389 -> 0.8734901462174189, cross_talk 0.6219512195121951 -> 0.6352941176470588 (198 exclusive studies)
- comparison/comparison.json parameters_only_in_run: {reference: [], candidate: []} — capacity-neutral confirmed on the saved checkpoints
- comparison/comparison.json input_identity_differences: ["preprocessing_identity"] only; config_differences limited to model.compartment_transfer.pair_rotation.pairs plus name/output_root/seed
- facts.timing: setup 0.2846 s, training 1544.9745 s, validation prediction 20.3009 s, measured_compute 1582.098 s vs 1,797 s ceiling; 4,380 optimizer steps
- facts.accounting: 10 train epochs at 93,846 triplets / 3,503 visits with cohort b536b7e1..., 10 validation epochs at 24,188 / 904 with cohort 7167eba1... and a constant selection hash
- comparison/per_label.json: all twelve exploratory intervals include zero; largest magnitudes Lateral Meniscus -0.0121 [-0.0301, 0.0050], Baker's +0.0112 [-0.0032, 0.0269], Synovitis -0.0112 [-0.0231, 0.0001]

## review-20260921T005103Z-2132cd2e

New beyond the developer report:

1. `pooled` is produced inside the autocast-disabled float32 block (model.py:218-223), so the objective, the similarity matmul and the float32-initialized FeatureQueue never mix bf16/float32. The developer's synthetic CPU check did not cover this, but the dtype path is safe.

2. Degenerate-row analysis: with one anchor and an empty queue the only entry is self-masked to -inf, so max-subtraction yields nan. `valid = positive_counts > 0` (objective.py:66) drops those rows before the mean, and `torch.where` (line 169) routes no gradient through non-positive cells, so neither forward nor backward propagates nan. Conversely, in an all-negative batch every anchor matches every other, so the loss is active on essentially every one of the 4,380 steps.

3. Consequence of that: with batch_size 8 (configs/training.yaml:99, ≤16 instances per step) and rare meniscus positives, most steps apply mainly negative-to-negative attraction. At lambda 0.3 against observed loss values around 3, the auxiliary term is comparable in magnitude to the main BCE and pulls negative pooled features together, which works against the within-negative ranking that AUC rewards — and it reaches the other eight targets through the shared backbone.

4. Soft targets are hard-thresholded at 0.5 (objective.py:89-94) while the main BCE keeps soft targets, so near-boundary studies on opposite sides of 0.5 receive maximal repulsion.

5. The queue is a plain object, not a buffer, so it is absent from the state_dict (confirming no checkpoint shim) but also means a genuine mid-run resume restarts with a cold FIFO and is not bit-reproducible from that checkpoint.

6. Added compute is negligible: two 256-dim matmuls over at most ~528 rows per step plus backward, far inside the 1,797 s ceiling, though still unmeasured on GPU.

Approving the candidate. The implementation realizes the assigned cross-compartment supervised contrastive alignment on the meniscus pair as a training-only auxiliary term on the pinned reference recipe, within the declared editable files/directories and config prefix, with no change to targets, masks, sampling, inference, or the evaluation path. Queue membership and order are unchanged and no proposals were submitted, so ranked_ids is null.

Topic fit is real rather than asserted: positives are label-matched across the two compartment target slots (objective.py:80-110), so medial-positive and lateral-positive pooled evidence is pulled together while medial-only/lateral-only studies become each other's hard negatives. All four label combinations stay representable — the constraint acts on the direction of L2-normalized pooled vectors, while each target keeps its own head row and unconstrained logit magnitude. This is genuinely additional transfer beyond the already-shared backbone: the reference shares features but nothing ties the two compartments' pooled disease evidence to a common label-conditioned geometry.

The substantive risks are interpretive, not blocking, and are recorded as concerns: the term is simultaneously a strong auxiliary supervised signal on the same two labels, so attribution to compartment invariance specifically needs a matched within-compartment-positives control rather than the macro interval alone; and it acts opposite in spirit to the concurrently running exclusivity arm, so neither arm is a control for the other.

Unresolved concerns:
- Attribution: the term is also plain auxiliary supervision on the two meniscus labels. A macro interval above zero would not establish cross-compartment invariance; a matched control restricting positives to within a single compartment (same loss, same lambda, no cross-compartment matching) is needed for that claim.
- Collapse risk: at batch_size 8 with rare meniscus positives, most steps are dominated by negative-to-negative attraction at lambda 0.3 against a loss of order 3, which can flatten within-negative variation and degrade ranking on the shared backbone, including the other eight targets.
- Hard 0.5 thresholding of soft targets discards the soft-label information the main BCE uses and maximally repels studies straddling the boundary; lambda_weight 0.3, temperature 0.1 and queue_length 512 are asserted, never measured, and a later sweep of them is a parameter search, not a new mechanism.
- The 512-entry queue holds detached features from up to ~32 earlier optimizer steps with no momentum encoder, so a substantial share of negatives are stale relative to the current encoder; the resulting loss is noisier than the in-batch part suggests.
- The queue lives outside the state_dict, so a genuine checkpoint resume restarts it cold and the resumed trajectory is not reproducible from that checkpoint. Not a contract breach for a single full-schedule run, but it must not be treated as resume-exact.
- Provenance and residue: training/compartment_transfer/* is outside identity.PREPROCESSING_SOURCES (identity.py:16-18, not editable here), so preprocessing_identity covers only the model.py call site and provenance rests on the candidate source hash and snapshot; an inert empty src/knee_abnormality/modeling/compartment_transfer/ directory is undeclared residue. Runtime remains argued from op counts, unconfirmed until timing.json.
- This arm promotes cross-compartment invariance while the running cross-compartment-attribution-exclusivity arm penalizes shared attribution from the same reference on overlapping labels. They are opposite-direction mechanisms, not controls for each other, and must be interpreted separately.

Evidence:
- changes.diff (experiment dir) — complete candidate diff: configs/training.yaml:106+, config.py:220-252, model.py:204-332, new training/compartment_transfer/{__init__,objective,queue}.py, lightning.py:26-33/234-244/314-324
- candidate source_/src/knee_abnormality/modeling/model.py:204-243 — pooled computed in the autocast-disabled float32 block (218-223); return_pooled is opt-in at 241-242, default path unchanged
- candidate source/src/knee_abnormality/training/compartment_transfer/objective.py:41-110 — self-mask, max-subtraction, torch.where positive masking (169 in diff), valid-row filter (66), 0.5 thresholding (89-94)
- candidate source/src/knee_abnormality/training/compartment_transfer/queue.py:1-20 — plain-object detached FIFO, float32 init, device migration on first update; no state_dict entry
- candidate source/src/knee_abnormality/identity.py:16-18 — PREPROCESSING_SOURCES unchanged; only model.py among the touched files is hashed
- candidate source/configs/training.yaml:99-100 — batch_size 8, accumulate_grad_batches 1 (basis for the ≤16-instances-per-step analysis)
- diff -rq parent_source vs candidate source — only the four declared files differ plus the two new package directories (one empty, in modeling/)
- developer report /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T004212Z-b049bddb/report.json — static checks, synthetic CPU forward/backward, nan fix in the masked sum

## completion-20260921T010850Z-0ac88d7b

The penalty worked mechanically but did not convert into ranking gain. Both families' cross-talk diagnostics moved in the intended direction: meniscus lateral_cross_talk 0.6364→0.5435 (medial-positive lateral false positives 28→25 while total lateral FPs rose 44→46), OA 0.6220→0.6000, i.e. lateral false positives became less concentrated on medial-positive studies — exactly what attribution exclusivity targets. Yet the compartment margin AUCs, the direct measure of medial-vs-lateral discrimination, both fell: meniscus 0.97125→0.96904 (bootstrap mean -0.00231, [-0.01315, 0.00788]), OA 0.86612→0.86205 (mean -0.00379, [-0.03591, 0.02769]). So the mechanism redistributed which negatives are false-positive without improving the ordering that separates the compartments.

Family means: meniscus -0.00295 [-0.01210, 0.00627], OA +0.00177 [-0.00542, 0.00932], other eight -0.00075 [-0.00577, 0.00420]. All twelve per-label intervals include zero. The only two labels with mostly-positive mass are Medial OA (+0.00547) and Effusion (+0.00588) — one of which is outside the paired set, reinforcing that the macro movement is not attributable to the transfer mechanism.

Notably the four penalized labels are net negative in aggregate (mean of Medial/Lateral Meniscus and Medial/Lateral OA deltas ≈ -0.00058) and the unpenalized eight are also slightly negative, consistent with the selective positive-logit shrinkage predicted at review: the shrinkage reached the shared backbone without benefit. Training BCE improved slightly (0.48566→0.48149) while macro AUC did not, consistent with a calibration-side effect rather than a ranking one.

Candidate selected epoch 10 vs reference epoch 9 under the unchanged selection rule; candidate epochs 8–10 sit within 0.0011 of each other, so this is not an unstable selection.

Interpretation only; no decision changes. Macro delta -0.00070, 95% [-0.00466, 0.00303] — the fixed rule gives inconclusive, so historical-control remains the incumbent and the exclusivity idea stays unaccepted without a second run being warranted by this evidence.

Two prior open concerns are now resolved by the run's artifacts. Runtime: 1,595.82 s measured compute (setup 0.296 + train 1,559.18 + predict 20.33, prep 0.119) against the 1,797 s ceiling, so the four extra reattribution einsums plus backward cost ≈+1.1% wall clock over the reference's 1,562.28 s — the op-count argument held. Contract: triplet_accounting matches the sealed 938,460/35,030 train and 24,188/904 validation counts with identical cohort hashes at zero tolerance, validation selection hash is constant across all ten epochs (independent of the fresh seed 3838108598), 4,380 steps, and config differences are confined to the four declared trainer.compartment_transfer keys plus name/output_root/seed, with preprocessing_identity the only input-identity difference. No new contract concern.

Queue membership and order are unchanged; the reserved head and the three queued ideas are untouched, and no proposals accompany this event.

Unresolved concerns:
- Attribution remains unestablished and now moot for acceptance: the intended exclusivity effect is visible in the cross-talk diagnostics but the compartment margin AUCs moved the other way, so this arm gives no support for exclusivity as a route to ranking gain. The previously requested matched non-paired-label control is not worth spending a slot on given the null.
- The exclusivity direction (penalize shared attribution) is the opposite of the reserved-head contrastive-alignment arm (promote shared representation). This null constrains neither: it is evidence against this specific margin-penalty implementation at lambda 0.05/margin 1.0, not against the campaign hypothesis.
- lambda_weight=0.05 and margin=1.0 were never measured and remain asserted. A sweep of either is a parameter search, not a new mechanism, and must not be admitted to revive this arm.
- Standing provenance gap persists in the record: training/compartment_transfer/objective.py lies outside identity.PREPROCESSING_SOURCES, so preprocessing_identity covers only the model.py call site; provenance rests on the candidate source hash and source_snapshot.zip (a7422ede...). Unchanged by this result.
- The undeclared empty src/knee_abnormality/modeling/compartment_transfer/ directory remains in the candidate source; inert, but it was never added to the integration plan.
- Interval is conditional on one seed, one fold and one checkpoint selection; it excludes seed variation, so the null does not exclude a small true effect in either direction.

Evidence:
- comparison/comparison.json:1213-1267 — paired report-group bootstrap: all [-0.004657, 0.003027] mean -0.000698; meniscus mean -0.002950; oa mean +0.001767; other_eight mean -0.000752; meniscus_margin mean -0.002305; oa_margin mean -0.003786 (852 groups, 2000 draws, seed 20260914)
- comparison/comparison.json:609-665 — compartment diagnostics: meniscus margin_auc 0.971248→0.969045, lateral_cross_talk 0.636364→0.543478, medial_positive_false_positives 28→25, lateral_false_positives 44→46; oa margin_auc 0.866116→0.862047, cross_talk 0.621951→0.600000
- comparison/comparison.json:591-607 — family means: meniscus 0.912195→0.909300, oa 0.888183→0.889914, other_eight 0.894778→0.894005
- comparison/comparison.json:173-181 and 403-411 — timing: reference train 1541.673 s; candidate setup 0.296 + train 1559.181 + predict 20.335 = 1595.82 s vs 1797 s ceiling; peak CUDA 20.94 GB vs reference 21.39 GB
- evidence.json:166-316 — triplet_accounting complete: 938,460 train triplets / 35,030 visits, cohort b536b7e1…; validation 24,188 / 904, cohort 7167eba1… with identical selection hash 92f1341c… across all ten epochs
- comparison/comparison.json:684-725 — config differences limited to name, paths.output_root, seed (2263738815→3838108598) and the four trainer.compartment_transfer keys; input_identity_differences = [preprocessing_identity] only
- comparison/comparison.json:271, 331-401 — candidate selected_epoch 10 (macro 0.895872) with epochs 8/9/10 at 0.894710/0.895562/0.895872; reference selected_epoch 9 (0.896582)
- comparison/comparison.json:60, 290 — soft_target_bce 0.485663 (reference) vs 0.481492 (candidate): loss improved while macro AUC did not

## proposals-20260921T012512Z-1ae37d72

Verified the two proposals' implementation plans against the pinned parent source, since both assert specific structure. `self.attention` is `Sequential(Linear(256,128), Tanh, Dropout(model.dropout), Linear(128,12))` with one shared hidden layer (model.py:109-114), so both designs' claim of reusing rather than duplicating the 256->128 layer is correct, and the twelve score rows are the final layer's weight. `head` is `Linear(256,12)` consumed as `(pooled * head.weight[None]).sum(-1) + head.bias` inside the autocast-disabled float32 block (model.py:216-223); `_attention_scores` short-circuits to `self.attention(tokens)` when `gated_attention` is false (model.py:196-201). The pinned recipe sets `gated_attention: false` and `residual_readout: false` (configs/training.yaml:85-86), confirming only the plain path needs changing.

`dropout: 0.0` in the pinned recipe (configs/training.yaml:83). This matters for the exchange operator, whose second attention pass over transformed tokens would otherwise consume extra dropout RNG and break the "no RNG consumption change" requirement; at p=0 the dropout fast path draws nothing, so the requirement is satisfiable, but it must still be verified against the parent with V zero, as the proposal states.

Both proposals delete the free parameterisation of the same two rows (Lateral Meniscus, Lateral OA attention). They are therefore mutually exclusive as a stack: whichever is decided first forces the other to be re-expressed against the new parent rather than merged.

Not verified here: the researcher's cosine/SVD probe numbers and per-label positive counts. They come from a single checkpoint of one seed and fold, so they motivate design but are not stable structural facts.

Both proposals are admitted: dependencies are empty, each states an explicit in-scope transfer mechanism, and both are weight-space designs that remove degrees of freedom rather than adding capacity, so a gain cannot be read as added capacity or added task weight. Neither restores the rejected pooled adapter (that acted on pooled features feeding independent heads; these act before/inside attention and delete the independent lateral pathways).

Queue change: the two proposals enter above both previously queued ideas. Reason is the new checkpoint-derived structure (attention, not head, carries side organisation; within-family compartment cosines 0.054/0.243 against a 0.138 off-diagonal mean), which no queued arm was designed against. compartment-exchange-operator-tied-pathway ranks above shared-lateralization-direction-attention because it addresses the positive-count imbalance (Lat Meniscus 138 vs 356, Lat OA 241 vs 329) by sharing the detector itself, whereas sharing one direction leaves each compartment's training signal unchanged.

cross-family-mediolateral-axis-agreement drops to last on new evidence, not on preference: the researcher's co-occurrence diagnostic indicates the reference already carries "which side is diseased" across families as a shortcut (same diseased side in 102/111 doubly-exclusive studies; on the 9 opposite-side studies OA lateralises 5/9 against 0.922 on same-side). A per-study penalty rewarding cross-family side agreement is the one design in the queue that could reinforce that shortcut rather than supply morphology. Support is n=9 and directional, which is why it is demoted rather than parked. shared-lateralization-direction-attention does not inherit this objection: delta is a fixed feature-space direction, not a per-study agreement term.

Before either new arm leaves development its open choices must be closed and recorded: for the exchange operator, rank r, V initialisation, whether R also transforms the pooled vector, and whether both families or meniscus alone; for the shared direction, attention-only versus head extension and whether gamma_f is free. A post-hoc sweep over these is a parameter search, not a new mechanism.

Unresolved concerns:
- Exchange operator shares one R across both families while the researcher's own probe puts the two families' medial-minus-lateral difference vectors at cosine 0.234. Over-constraint is the principal risk; the named untied-R control is required before any cross-family claim, and a result is uninterpretable as 'exchange' unless ||UV^T||/||I|| is logged and non-negligible (near zero reduces the run to 1A-style tying).
- Both new arms touch 4 of 12 labels. Against the ~0.0034 macro seed floor and 1A's record (~0.003 per compartment label worth ~+0.0005 macro), either can clear the bar only via trunk-level effects. An interval above zero driven by the other eight labels would be a macro improvement without establishing compartment transfer.
- Exchange operator is directional (medial raw, lateral transformed), not an involution. A degenerate R toward identity merges the compartments; margin AUC (meniscus 0.9712, OA 0.8661) must be reported alongside per-compartment AUC so a weaker-label gain bought by margin collapse is visible.
- compartment-exchange-operator-tied-pathway and shared-lateralization-direction-attention remove the same two lateral attention rows and cannot be stacked; the second to be developed needs re-expression and renewed integration assessment against the then-current parent.
- The probe evidence behind both proposals (cosines, SVD energies) is from one checkpoint, one seed, one fold, with no dispersion estimate. It justifies design choices but must not be cited as established structure when interpreting results.
- The co-occurrence shortcut diagnostic that demotes cross-family-mediolateral-axis-agreement rests on 9 opposite-side studies. If that arm is later revisited, the demotion should be re-examined rather than treated as settled.

Evidence:
- Researcher report (proposals event source): /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T010946Z-3e43762f/report.json — two ideas plus exhausted_directions with the checkpoint probe and co-occurrence diagnostic
- Parent source attention/head construction: .../recovery/historical-source-20260920T215302Z-e086c1eb/source/src/knee_abnormality/modeling/model.py:109-114 (attention Sequential 256->128->12), :133 (head Linear(256,12))
- Pooling and readout path: same model.py:196-201 (_attention_scores plain branch) and :216-223 (float32 autocast-disabled pooled/logits block)
- Pinned recipe switches: .../source/configs/training.yaml:83 dropout 0.0, :85 gated_attention false, :86 residual_readout false
- Campaign contract: prior-work.md and docs/validation/results/2026-09-20-compartment-adapter-result.md for the rejected pooled adapter these proposals are distinguished from
- Cross-family attention cosines 0.363/0.314 same-side vs 0.054/0.243 within-family, off-diagonal mean 0.138; difference-vector alignment 0.234 — researcher report exhausted_directions
- Co-occurrence: 111 doubly-exclusive validation studies, 102 same diseased side; opposite-side subset n=9, meniscus 8/9 vs OA 5/9 — researcher report exhausted_directions

## review-20260921T012608Z-b6bdbe11

New beyond the developer report:

1. Invalid-pose fallback is not an abstention. Zeroed patient maps produce a constant "maximally medial" axis rather than 0 (model.py:315-324 with preparation.py:217-228). Rate is unmeasured; `_resolve_knee_side` only abstains when fewer than two *series* have valid poses, so a qualifying study can still contribute invalid-pose triplets.

2. Scorer input conditioning. The axis enters unnormalised in millimetres alongside LayerNorm-scaled tokens. With default `Linear(257,32)` init (U(±0.062)) a ±60 mm axis contributes ~2σ of preactivation versus ~0.6σ from the 256 token dims, so the tanh layer starts substantially saturated by the coordinate. Zero-init of the output layer means the run still starts exactly at control B, and the first layer can learn to shrink the axis weight, but early on the added term behaves close to a side step-function rather than a morphology read.

3. Signature changes break two out-of-tree callers that the developer could not edit (both outside `editable_files`): `scripts/tools/preview_augmentations.py:255` unpacks `feature_geometry` as a 2-tuple, and `scripts/adhoc/diagnose_encoding_usage.py:206` calls `_attention_pool` with four positional arguments. Neither is on the training or prediction path.

4. The side/centre resolution uses all indexed candidate series including those later excluded (`_resolve_knee_side(indexed)`, dataset.py:143). Deterministic and label-free, but it is not the set of series the model actually consumes.

5. Added cost is genuinely small: four 257→32→1 MLP evaluations over ~10^4 tokens per step, plus a (B, slices, 3, 3, 3) float collation — both far inside the ceiling, though still unmeasured on GPU.

The mechanism is implemented faithfully and the declared integration is genuinely additive (verified: `self.compartment_transfer=None` when the config block is absent, and `_attention_scores` then returns the parent expression unchanged; no accepted idea exists to remove, so empty `removed_*` is correct). Geometry is correct: `patient_maps = native_maps @ output_to_native` maps prepared-pixel homogeneous coords to LPS patient space (preparation.py:222-228), `prepared_xy` is the reflection/augmentation-corrected prepared anchor (spatial_tokens.py:244-247), so the axis is invariant to reflection augmentation while the token feature follows it — the pairing is anatomically consistent. Side convention is right (LPS +X = left, so medial = -X for a left knee; dataset.py `_resolve_knee_side` returns -1 for center>0). Middle channel index 1 is fixed under `reflection_channels` (2-c permutation leaves 1 invariant), so the fixed `[flat_indices, 1]` is safe. Side=0 abstention yields an exactly equal additive term for both compartments, leaving neither/medial-only/lateral-only/both representable. Validation is deterministic and seed-independent; cohort, triplets, labels, masks, decoding and step count are untouched; only `preprocessing_identity` should differ.

One repair is required before launch. For a triplet whose selected poses are invalid, `pose_values[~valid]=0` makes the patient map an all-zero matrix, so `patient_x=0` and the axis becomes the constant `-side*center ≈ +|center|` — i.e. every token of that image is declared maximally *medial* rather than abstaining. `geometry_valid` is carried but only gates reflection eligibility (dataset.py:170-176); such triplets are still sampled. This injects a deterministic, acquisition-correlated pseudo-signal into exactly the two target pairs under study and confounds the manufacturer diagnostic the evaluator reports.

Repair: in `model.py` encode_batch, zero `medial_axis` for rows whose middle patient map is invalid (e.g. `valid = middle_map.abs().amax((-1,-2)) > 0`; `medial_axis = medial_axis * valid[:, None]`), so those tokens abstain exactly as side=0 studies do. No other change.

Unresolved concerns:
- Attribution: the added term supplies a side-signed mediolateral coordinate to attention as well as tying the pair. A macro gain would not by itself establish compartment transfer; the matched control is an untied variant (separate scorer per compartment, identical axis input and parameter count).
- The abstention rate (fraction of studies with side=0) and the invalid-pose token fraction are never recorded. If abstention is common the mechanism may be largely inert, and an inconclusive result would be uninterpretable. Both are deterministic and recomputable from the sealed data and source.
- Axis scale is unnormalised millimetres against normalised tokens, so the shared scorer starts near tanh saturation on the coordinate; a later sweep of hidden width or axis scaling would be a parameter search, not a new mechanism.
- `apply_attention_temperature` (inference/checkpoint.py:78, non-editable) rescales only `model.attention[-1]` and not the new scorer. Irrelevant at the fixed ATTENTION_TEMPERATURE=1.0, but the added term would be misscaled at any other temperature.
- Signature changes break scripts/tools/preview_augmentations.py:255 and scripts/adhoc/diagnose_encoding_usage.py:206; neither file is editable under this campaign policy, so the breakage would persist if this candidate is adopted.
- The 40 mm qualification threshold, >=2-series requirement and sign-consensus rule are asserted design choices with no measured support on this cohort.

Evidence:
- changes.diff and source at outputs/autoresearch/medial-lateral-transfer/experiments/anatomical-medial-axis-mirror-attention-20260921T010946Z-be017ff5/
- source/src/knee_abnormality/modeling/model.py:140-150 (gated construction), :210-233 (_attention_scores/_attention_pool threading), :312-324 (medial-axis computation)
- source/src/knee_abnormality/modeling/compartment_transfer/attention.py:1-36 (shared scorer, zero-init output layer, mirrored +m/-m evaluation)
- source/src/knee_abnormality/data/preprocessing/preparation.py:196-238 (pose validity zeroing, native_maps @ output_to_native, geometry_valid)
- source/src/knee_abnormality/data/dataset.py:36-66 (_series_center_x/_resolve_knee_side), :143 (per-study side), :322-397 (collation of patient_maps/knee_side_sign/knee_center_x_mm), :168-176 (geometry_valid used only for reflection eligibility)
- source/src/knee_abnormality/modeling/spatial_tokens.py:199-247 (feature_geometry returns prepared[..., :2]; arity change)
- source/src/knee_abnormality/modeling/augmentation.py:44-48 (reflection_channels leaves middle channel index 1 invariant)
- developer report outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T010946Z-02e7578f/report.json (CPU-only checks; no GPU timing; attention-temperature limitation self-reported)

## review-20260921T013109Z-c8161ae4

New beyond the prior review and the developer report:

1. The repair is genuinely one hunk. Recursive source diff between the previous candidate and this one shows only model.py:325-328; every other file is byte-identical, so the earlier geometry, side-convention, reflection-invariance and zero-init verification carries over unchanged.

2. The mask is evaluated on the middle triplet channel only (`patient_maps[..., flat_indices, 1]`), which is the same channel that supplies patient_x, so detection and use are consistent. It is computed inside the `autocast(enabled=False)` block on the float32 map, then broadcast as bool against the float product before the single `.to(features.dtype)` cast, so no bf16/float32 mixing and no dtype-promotion surprise.

3. Abstention is now exactly symmetric with side=0: axis 0 makes `scorer([t, +0])` and `scorer([t, -0])` identical, so both compartments of each pair receive an equal additive term and the pair ordering is untouched for those tokens.

4. Residue: `src/knee_abnormality/training/compartment_transfer/` exists and is empty in the candidate tree (confirmed by listing). It is inert and undeclared, but it is inside an editable directory and cannot affect behavior.

5. Runtime cost is unchanged by the repair (one amax over a 3x3 map per token row); the added-cost argument still rests on op counts and is unconfirmed until timing.json.

The one repair I required is implemented exactly and nothing else changed, so this candidate is ready. I verified the repair by direct source diff between the two candidate trees (single hunk at model.py:325-328) rather than from the developer's claim, and re-verified the parent-relative file set to confirm no other behavior drifted in during the repair.

The mask is computed inside the autocast-disabled float32 block before the dtype cast, so the bool broadcast and the final `.to(features.dtype)` are well defined, and no gradient path is affected for tokens with valid geometry. Carried-forward concerns from the first review that the repair does not address (attribution versus an untied control, unnormalised millimetre axis scale, attention-temperature rescaling omission, two broken out-of-tree script callers, unmeasured abstention/invalid-pose rates) remain open and are restated below only where they bear on interpreting the eventual result.

Queue membership and order are unchanged.

Unresolved concerns:
- Attribution is unchanged by the repair: the term supplies a side-signed mediolateral coordinate to attention in addition to tying the pair, so a macro gain would not by itself establish compartment transfer. The matched control is an untied variant with a separate scorer per compartment, identical axis input and parameter count.
- The side=0 abstention rate and the now-masked invalid-pose token fraction are still unrecorded. If either is large the mechanism may be substantially inert and an inconclusive result would be uninterpretable; both are deterministic and recomputable from the sealed data and source.
- Axis scale is unnormalised millimetres against LayerNorm-scaled tokens, so the shared scorer starts near tanh saturation on the coordinate; the zero-init output layer means the run still starts at control B, but early updates behave closer to a side step-function than a morphology read. A later sweep of width or axis scaling would be a parameter search, not a new mechanism.
- apply_attention_temperature (inference/checkpoint.py:78, non-editable) rescales only model.attention[-1] and not the new scorer; harmless at the fixed ATTENTION_TEMPERATURE=1.0, but the added term would be misscaled at any other temperature.
- Signature changes still break scripts/tools/preview_augmentations.py:255 and scripts/adhoc/diagnose_encoding_usage.py:206; neither is editable under this policy nor on the training/prediction path, so the breakage persists if this candidate is adopted.
- Undeclared inert residue: an empty src/knee_abnormality/training/compartment_transfer/ directory in the candidate tree.

Evidence:
- Repair delta: diff -r between experiments/anatomical-medial-axis-mirror-attention-20260921T010946Z-be017ff5/source and .../20260921T012933Z-2303a296/source — single hunk, model.py:325-328
- Parent-relative member set: diff -r -q against recovery/historical-source-20260920T215302Z-e086c1eb/source — differs only in configs/training.yaml, config.py, data/dataset.py, modeling/model.py, modeling/spatial_tokens.py, new modeling/compartment_transfer/ (plus empty training/compartment_transfer/)
- candidate source/src/knee_abnormality/modeling/model.py:307-328 (autocast-disabled axis block, valid mask), :140-150 (gated construction), :211-231 (additive scores), :350-356 (None-safe threading)
- candidate source/src/knee_abnormality/modeling/compartment_transfer/attention.py:1-36 (shared scorer per pair, zero-init output layer, mirrored +m/-m evaluation)
- prior review outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T012628Z-3436edc2/report.json (geometry, side convention, reflection-channel invariance, required repair)
- developer report outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T012933Z-3360db0b/report.json (py_compile via supplied interpreter; preparation.py byte-identical; no GPU timing)
- facts.config_differences: model.compartment_transfer.pairs = [[2,3],[4,5]] only

## completion-20260921T013624Z-dcf1ea20

Rejected: macro -0.00500 [-0.00919, -0.00116]. New beyond the pre-launch review:

1. Attribution of the loss is unambiguous. other_eight -0.00712 [-0.01234, -0.00220] (strictly negative); meniscus -0.00251 [-0.01265, +0.00746]; oa +0.00101 [-0.00610, +0.00796]. The auxiliary term damaged labels it never supervised.

2. The mechanism produced no compartment effect on its own family. Meniscus margin AUC 0.9712 -> 0.9723, margin delta +0.00088 [-0.00944, +0.01183]; lateral false positives 44 -> 40, medial-positive false positives 28 -> 27, lateral cross-talk 0.636 -> 0.675. OA margin 0.8661 -> 0.8613 (unsupervised family, wide interval). So the arm is uninformative about cross-compartment invariance: it did not move compartment separation while it did move the backbone.

3. Prediction change is broad rather than pair-localized: mean absolute probability difference 0.0693, with the lowest per-target Spearman correlations on Lateral Meniscus 0.868, Fracture 0.874, Contusion 0.889, ACL 0.893 — unrelated targets shifted as much as the targeted pair.

4. Contract facts: 1596.09 s total against the 1797 s ceiling; preparation 0.221 s; setup 0.302 s; training 1557.59 s; prediction 20.98 s; 4,380 optimizer steps; peak CUDA 21.39 GB. Per-epoch train triplets 93,846 x 10 and validation 24,188 with the identical validation selection hash on all ten epochs, confirming validation independence from the training seed.

Interpretation of the completed cross-compartment supervised-contrastive arm (seed 3554221328, 1596.09 s of a 1797 s ceiling, accounting complete and exactly matching the sealed 938,460/35,030 training and 24,188/904 validation counts; only the five declared trainer.compartment_transfer keys, name, output_root and seed differ, with the allowed preprocessing_identity difference).

Macro delta -0.00500, 95% interval [-0.00919, -0.00116], strictly below zero. Contract checks pass, so this is a clean rejection of this implementation, not an invalid run.

The decomposition is the informative part and it does not implicate cross-compartment transfer itself. The targeted meniscus family is flat (delta_mean -0.00251, [-0.01265, +0.00746]) and the meniscus compartment margin is unchanged (+0.00088, [-0.00944, +0.01183]; margin AUC 0.9712 -> 0.9723, lateral cross-talk 0.636 -> 0.675). The loss is carried entirely by the eight unrelated targets (-0.00712, [-0.01234, -0.00220]), led by ACL (-0.0191, interval below zero) and Fracture (-0.0185, interval below zero). OA is unchanged (+0.00101). This is the pre-launch collapse concern realized: at batch_size 8 with rare meniscus positives, most of the 4,380 steps apply negative-to-negative attraction at lambda 0.3, and the damage propagates through the shared backbone to labels the objective never touched. Per-label intervals here are exploratory and unadjusted; they explain the macro result rather than supporting any per-label claim.

Consequences for the campaign: the objective delivered no measurable compartment-level effect on the family it targeted, so this run does not test the transfer hypothesis and does not refute it. A revisit would need a materially different design, not a lambda/temperature/queue sweep of this one, and would need the matched within-compartment-positive control before any transfer attribution. It also supplies a general caution for the queue: auxiliary objectives applied to a rare-positive pair at a weight comparable to the main BCE can cost more on the other eight targets than any plausible pair gain.

No queue change: no proposals arrived, and this result does not reorder the remaining attention-side mechanisms, which act on the pair pathway rather than adding backbone-wide feature pressure.

Unresolved concerns:
- The predicted collapse pathway is confirmed empirically, so any future auxiliary-loss proposal on a rare-positive pair must state how it avoids negative-to-negative attraction dominating the schedule, and must be weighted well below the main BCE rather than at lambda 0.3.
- This arm cannot be cited as evidence for or against cross-compartment transfer: it left the meniscus compartment margin unchanged. Retiring the hypothesis on this basis would be unsupported.
- No matched within-compartment-positive control was run, so even the negative result does not separate 'cross-compartment matching is harmful' from 'this contrastive term at this weight is harmful'. Attribution remains open if the mechanism is ever revisited.
- Hard 0.5 thresholding of soft targets and the 512-entry stale queue remain unmeasured confounders inside the rejected implementation; a later sweep of lambda, temperature or queue length would be a parameter search, not a new mechanism.
- Provenance residue from the pre-launch review stands on the archived candidate: training/compartment_transfer/* sits outside PREPROCESSING_SOURCES, and an empty src/knee_abnormality/modeling/compartment_transfer/ directory was undeclared. Harmless for a rejected arm, but the same gap would recur for any future module under those directories.
- changed_source_archive_members lists twelve src/knee_abnormality/autoresearch/* harness files alongside the four experiment files; these are off the training and prediction path, but the snapshot diff is therefore not a minimal record of the experimental change.

Evidence:
- comparison/comparison.json (experiments/compartment-invariant-disease-embedding-20260921T004212Z-71970c65): paired_report_group_bootstrap.intervals — all, meniscus, oa, other_eight, meniscus_margin, oa_margin; 852 report groups, 2000 draws, seed 20260914
- comparison/comparison.json family_means: all 0.89658 -> 0.89164, meniscus 0.91220 -> 0.90977, oa 0.88818 -> 0.88920, other_eight 0.89478 -> 0.88771
- comparison/comparison.json compartments: meniscus margin_auc 0.9712 -> 0.9723, lateral_cross_talk 0.6364 -> 0.6750; oa margin_auc 0.8661 -> 0.8613
- comparison/comparison.json prediction_change: mean_abs_probability_difference 0.0693; per_target_spearman ACL 0.893, Fracture 0.874, Lateral Meniscus 0.868
- comparison/comparison.json config_differences and input_identity_differences: only the five trainer.compartment_transfer keys plus name/output_root/seed; preprocessing_identity is the sole allowed identity difference
- assignment facts.accounting and facts.timing: 10 x 93,846 train triplets / 3,503 visits, cohort b536b7e1…; 10 x 24,188 validation triplets with constant selection hash 92f1341c…; 1596.09 s measured compute vs 1797 s ceiling
- Prior implementation review: jobs/orchestrator-20260921T005103Z-c6c6b15a/report.json (collapse-risk and attribution concerns, now borne out)
- Developer report: jobs/developer-20260921T004212Z-b049bddb/report.json

## review-20260921T014726Z-1a1cbc43

New beyond the developer report:

1. The plain path is provably unchanged, not merely claimed. `_pool_scores` reorders `.float()` and `.transpose(0,1)` relative to the parent; on a 2-D tensor this is an elementwise cast commuting with a view, so the refactor is bit-identical, consistent with the developer's torch.equal ablation.

2. The tie is exact only at evaluation. `_attention_scores` is called twice in training and passes through `attention[2]` (Dropout p=model.dropout) each time, so medial and lateral scores see independent dropout masks. The two compartments are therefore stochastically decoupled during training and exactly tied at prediction. This also consumes extra draws from the device generator, shifting the realization (not the distribution) of later augmentation/series-dropout draws; fresh seeds make this immaterial.

3. Target indices verified against data/folds/all_5fold_v3/protocol.json: positions 2-5 are Medial/Lateral Meniscus and Medial/Lateral OA, matching the configured pairs and the name whitelist.

4. Added compute is a second attention MLP pass, one rank-16 operator application and one extra pooling einsum over the same tokens — on the order of tens of MFLOPs per batch against a ReXNet-200 backbone over 224px inputs, so the ceiling should hold, but this rests on op counts until timing.json exists.

5. `validate_compartment_transfer_pairs` is invoked only from the LightningModule constructor, so an inference-only model build does not re-check the name mapping. Harmless here because prediction reads the run's saved config, but the guard is weaker than it appears.

Approved the candidate. The implementation is a faithful tied-pathway exchange operator confined to the declared editable surface, it preserves the contract-fixed recipe and the twelve-logit interface, and validation stays deterministic and independent of the training seed (both attention passes are dropout-free in eval; the operator is deterministic).

No queue change: the review event brings no evidence about relative idea promise, and no proposals were submitted, so ranked_ids stays null and the reserved head is unchanged.

Unresolved concerns:
- Attribution: this replacement removes two attention rows and two head rows while adding a rank-16 operator and two biases, so parameter count and structure both change. A macro loss would confound the tying constraint with reduced capacity, and a macro gain would not isolate shared morphology. The matched control is an untied variant with per-compartment rows plus an equally sized operator.
- A single R is shared by meniscus and OA. If the two families need different medial->lateral maps, an inconclusive result speaks only to this tied single-operator form, not to the exchange-operator hypothesis generally.
- Training-time tie is inexact: the two attention passes draw independent dropout masks (model.py:249-253 via attention[2]), so the train-time and eval-time functions differ slightly in how coupled the pair is. This is a train/eval mismatch specific to the added pathway.
- Added runtime cost is estimated from op counts only; the second attention pass, the extra token-sized transformed tensor and the second batch x 10 x tokens softmax are unmeasured until timing.json is written.
- validate_compartment_transfer_pairs runs only at LightningModule construction, not in inference/checkpoint.py model building, so the anatomical name-to-index guarantee is not re-established on a standalone prediction path.
- compartment_transfer_operator_energy_ratio is computed from live parameters every training step and relies on Lightning detaching logged tensors; it is a diagnostic only and must not be used to select or tune anything.

Evidence:
- changes.diff and recursive diff vs parent_source: only configs/training.yaml, config.py, modeling/model.py, training/lightning.py and the two new compartment_transfer packages differ; dataset.py, preparation.py, augmentation.py, spatial_tokens.py byte-identical
- candidate src/knee_abnormality/modeling/model.py:112-140 (routing, ExchangeOperator, lateral_head_bias, attention_width=compact_count for attention[-1] and head)
- candidate src/knee_abnormality/modeling/model.py:229-288 (_pool_scores refactor, second pass over transformed tokens, family-row reuse, scatter into twelve slots)
- candidate src/knee_abnormality/modeling/compartment_transfer/__init__.py:1-46 (R = I + U V^T with V zero-init; routing index construction)
- candidate src/knee_abnormality/training/compartment_transfer/__init__.py:1-18 and training/lightning.py:218-224 (pair-name whitelist enforced at module init)
- /home/claudio/src/knee-abnormality/data/folds/all_5fold_v3/protocol.json targets list confirming indices 2-5
- parent src/knee_abnormality/inference/checkpoint.py:57-101 (model-section config diff and strict load_state_dict; attention[-1] temperature scaling makes no width assumption)
- developer report /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T013725Z-7ec79750/report.json (constraint_evidence: bit-identical plain-path ablation, zero-init identity, medial logits invariant to V)

## completion-20260921T020327Z-c1e9d89c

The invalid verdict is explained by a reproducible implementation defect, not by instability or budget.

1. Stage of failure: training/fold_0 contains metrics_epoch_000..009, validation_epoch_000..009.csv, triplet_accounting.json (complete, 938,460 train triplets / 35,030 visits, 24,188 validation triplets, matching sealed hashes) and selected_checkpoint.json (score 0.8943464756), but no validation_predictions.csv and no timing.json. Failure is between train.py:272 and train.py:291.

2. Cause: model.py:318-324 consumes batch["patient_maps"], ["knee_side_sign"], ["knee_center_x_mm"] and indexes them with CUDA-resident flat_indices/token_study. inference/engine.py:95-115 move_batch transfers only the eight MODEL_BATCH_KEYS, which exclude all three; engine.py is unchanged from the parent and is non-editable under campaign policy. Lightning's automatic batch transfer masks this during training and per-epoch validation, so the defect only surfaces in the final prediction pass (and would equally break scripts/pipeline/predict.py).

3. The defect is independent of the repair hunk (model.py:325-328) and was present in the pre-repair candidate; the earlier approvals did not check the inference batch-transfer allowlist.

4. Compliant fix is available inside editable code: move the needed tensors (ideally only the gathered slices) onto features.device within encode_batch. Cost is negligible relative to the existing image transfer, but it remains unmeasured.

5. Runtime remains unverified: the 1564.6 s process time includes no final prediction, and no timing.json exists, so the 1,797 s ceiling has not been tested for this mechanism.

6. No epoch-level result from this run may be used as evidence: the per-epoch macro AUC is a checkpoint-selection statistic, not a paired bootstrap comparison against the reference predictions.

Interpreting the recorded invalid outcome for anatomical-medial-axis-mirror-attention-20260921T012933Z-2303a296 (exit_code 2 at 1564.6 s, no timing.json, no validation_predictions.csv).

The failure is not a resource or harness fault and not caused by the one-line repair. Training itself completed: all ten epoch metric files and per-epoch validation CSVs exist, fold_0/selected_checkpoint.json was written (epoch 9, val macro AUC 0.89435), and triplet accounting is complete and identical to the sealed cohort/visit/triplet counts. The crash therefore falls in scripts/pipeline/train.py:282-291, between checkpoint selection and the final prediction CSV.

Root cause, established statically from the candidate source: encode_batch reads three new batch keys (patient_maps, knee_side_sign, knee_center_x_mm, model.py:318-324) and indexes them with flat_indices / token_study, which are derived from already-moved CUDA tensors. On the Lightning path the whole batch is transferred automatically, so training and per-epoch validation work. The final prediction path uses inference/engine.py move_batch, which transfers only keys in MODEL_BATCH_KEYS (engine.py:95-115); that frozenset contains images, reflection_eligible, series_presence, series_study_index, series_type_index, slice_mask, slice_positions, valid_rects and none of the three new keys. They stay on CPU while the index tensors are CUDA, which raises at the first prediction batch. engine.py is byte-identical to the parent and is outside the campaign's editable set, so this is a structural incompatibility of the implementation with the fixed prediction entrypoint, not an editable-file bug.

This defect predates the repair and was present in the reviewed pre-repair candidate as well; both of my prior reviews checked the training path and the checkpoint state-dict round trip but not the non-Lightning batch-transfer allowlist. That is a review gap on my side, now recorded.

The idea itself is not refuted: no comparison evidence exists, no seed is consumed usefully, and the mechanism remains implementable within policy by moving the three tensors (or only the required slices) to features.device inside encode_batch, which is editable. Any future review of a geometry-plumbing candidate in this campaign must check MODEL_BATCH_KEYS coverage before launch. The prepared compartment-exchange candidate is unaffected: it adds no batch keys and leaves dataset.py untouched.

No queue change, no proposals, no parking.

Unresolved concerns:
- Root-cause defect: new batch keys patient_maps/knee_side_sign/knee_center_x_mm are absent from MODEL_BATCH_KEYS (inference/engine.py:95-105), a non-editable file, so they stay on CPU while indices are CUDA; the final prediction pass cannot run. Any repair must place them on features.device inside editable model code.
- Review gap: both prior implementation reviews of this candidate verified training, checkpoint round-trip and geometry but not the non-Lightning batch-transfer allowlist. Future candidates adding batch keys (geometry or anatomy plumbing) must be checked against MODEL_BATCH_KEYS before launch.
- Runtime for this mechanism is still unmeasured against the 1,797 s ceiling: the run died before prediction and wrote no timing.json, so the earlier op-count argument remains unconfirmed.
- No usable evidence about the mechanism was produced. Epoch-9 validation macro AUC 0.89435 is a selection statistic from this run only and must not be compared informally with the reference; the idea is neither supported nor refuted.
- Previously recorded and still unresolved for this idea: attribution requires a matched untied control with identical axis input; the side=0 and masked invalid-pose token fractions are unrecorded; the added scorer is not rescaled by apply_attention_temperature; and an empty src/knee_abnormality/training/compartment_transfer/ residue remains undeclared.

Evidence:
- experiments/anatomical-medial-axis-mirror-attention-20260921T012933Z-2303a296/evaluation.json (exit_code 2, missing training/fold_0/timing.json)
- Same experiment training/fold_0/: metrics_epoch_000..009.json, validation_epoch_000..009.csv, triplet_accounting.json present; selected_checkpoint.json score 0.8943464756; validation_predictions.csv absent
- candidate source/scripts/pipeline/train.py:272-292 (selected_checkpoint.json written, then load_checkpoint_model/predict_loader, then validation_predictions.csv and timing.json)
- candidate source/src/knee_abnormality/inference/engine.py:95-115 (MODEL_BATCH_KEYS allowlist and move_batch), :137-174 (model(batch) under inference_mode/autocast)
- candidate source/src/knee_abnormality/modeling/model.py:307-328 (axis block indexing batch['patient_maps'] with flat_indices and ['knee_side_sign']/['knee_center_x_mm'] with token_study)
- candidate source/src/knee_abnormality/data/dataset.py:363-396 (new collate keys as tensors)
- prior reviews jobs/orchestrator-20260921T012628Z-3436edc2/report.json and jobs/orchestrator-20260921T013109Z-8a5d7d0f/report.json (approvals that did not cover the inference transfer path)
- campaign_policy.editable_files in this assignment (inference/engine.py not editable; modeling/model.py editable)

## review-20260921T021719Z-e901c28c

1. The forward path is identical to the parent when compartment_transfer is unset: _attention_scores (model.py:219-235) applies attention[0..3] in the same order as the old Sequential call.
2. Regularization is asymmetric. build_optimizer (lightning.py:43) exempts parameters with ndim < 2 from weight decay. So `disease` (2×128) is decayed, but `direction` (128) and `magnitude` (2) are not. In the parent, all four compartment rows were decayed. The shared direction therefore goes unregularized, which is a small confound on top of the tying constraint. All the new parameters use head_lr, the same as the rows they replace.
3. Initialization differs from the parent. The starting lateral vector is u_f + 1·δ, where both u_f and δ use Kaiming-uniform init. Its norm is about √2 times a parent row, and the two lateral vectors start correlated through δ.
4. Under bf16-mixed autocast, F.linear returns bf16 for both the base scores and the replacement columns, so the index_copy dtypes match. The developer's gradient test ran outside the Lightning precision context, but the dtype logic still holds.
5. The magnitude log calls .tolist() on every training step, forcing a host-device sync 4,380 times. The cost is likely negligible but it counts toward the ceiling.
6. Attribution: compared with control B, this reduces capacity (4×128 → 3×128 + 2). A gain would support the benefit of the tie itself rather than extra capacity. An inconclusive or negative result tests only this specific cross-family single-direction form, not medial/lateral transfer in general.

I read the diff and the critical source directly. The mechanism is in scope. It forces the medial-to-lateral change in attention to point the same way for meniscus and OA (one shared direction scaled by a per-disease magnitude), while each disease keeps its own medial attention vector and every compartment keeps its own bias and head row. It is a real constraint: the two families' lateral-minus-medial differences can only differ by a scale factor (rank one). Validation is unchanged: training validation applies no attention temperature and stays deterministic and seed-independent. The candidate adds no preparation cost (0.17 s recorded) and no new passes over the data. It adds 386 parameters and two small matmuls per forward, so the 1,797-second ceiling should hold, but timing is still unmeasured. The queue is unchanged.

Unresolved concerns:
- The shared direction and magnitudes escape weight decay (lightning.py:43, ndim<2), while the parent's compartment rows were decayed. Any effect mixes the tying constraint with a change in regularization.
- Exporting with a baked attention temperature (inference/checkpoint.py apply/bake_attention_temperature) would scale attention[3] but not disease/direction/magnitude, so the four compartment columns would be wrong. This doesn't affect campaign validation, but any later adoption needs the bake fixed before export.
- Runtime cost is estimated from operation counts only. Timing is unmeasured, including the per-step .tolist() sync.
- Attribution: the tie reduces capacity and shares one direction across both families. A null result would not rule out family-specific or softer sharing.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/shared-lateralization-direction-attention-20260921T020547Z-5e8cea71/changes.diff
- source/src/knee_abnormality/modeling/model.py:127-147 (buffers, module construction), :219-235 (score replacement)
- source/src/knee_abnormality/modeling/compartment_transfer/__init__.py:8-31
- source/src/knee_abnormality/training/lightning.py:35-50 (param groups, decay by ndim), :321-331 (magnitude log)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T020547Z-f75f5653/report.json (gradient/dead-row check, +386 params, preprocessing hash change)

## proposals-20260921T021814Z-9da982b5

Static check of the mirror proposal against the parent source:
- position_2d is a learned MLP (Linear 2→64, GELU, Linear 64→256; spatial_tokens.py:253,278) shared by all twelve targets.
- mil_norm (model.py:110,316) is shared by all twelve targets.
- Detaching features therefore does not isolate the penalty from other labels. Its gradient still reaches shared token geometry and normalisation that all twelve targets use. The claimed structural protection from other-label interference is partial.
- The reference dropout is 0.0 (configs/training.yaml:83), so the second attention pass is deterministic as claimed.

Bias asymmetry: the penalty compares the medial logit on mirrored features with the lateral logit, and vice versa. Medial and lateral priors differ sharply (training positives 40.7% vs 15.8% for meniscus). Without free per-family offsets, or a comparison that excludes the head biases, the penalty directly penalises that prevalence gap. Gradient would then push head and attention rows to encode prevalence through geometry. The two offsets on the LightningModule should be part of the implementation, not an open choice.

Interpretation note for later cross-family arms: the reference's medial-minus-lateral contrasts are already coupled across families well beyond the labels (0.787 vs 0.305).

Both proposals pass admission with no dependencies. Neither repeats a completed arm or brings back the rejected adapter.

mediolateral-mirror-exchange-consistency goes second, behind the reserved head. It is the campaign's first counterfactual that intervenes on geometry only and uses no labels. Label-driven arms leave co-occurrence and added task weight as explanations; this one doesn't. It ties no weights and adds no batch keys, so it avoids what made anatomical-medial-axis-mirror-attention invalid. The proposal includes an anterior-posterior mirror control for attribution. It is also a soft counterpart to the invalid hard-mirror arm, which never produced evidence, so this is the first real test of mirror equivariance.

shared-family-level-channel is admitted but ranked fourth. It adds shared parameters with an exact control-B start, and no arm has tested that yet. However, a gain would be hard to attribute. It adds capacity, and s_f can absorb co-occurrence, which the private-channel control does not separate from morphology. The proposal's own diagnostic shows that in the reference predictions the pair logits already correlate much more than the labels (OA r=0.854 vs 0.511). That weakens the case that a single shared level is missing.

cross-family-mediolateral-axis-agreement moves to last. The researcher's new diagnostic shows the two families' medial-minus-lateral logit contrasts already correlate 0.787 (0.815 after removing both family sums). The label contrasts correlate only 0.305. More cross-family lateralization coupling pushes along an axis that is already over-coupled. The prepared head shares that concern, but it stays first as the reserved head and is not reranked.

Unresolved concerns:
- mirror-exchange: the gradient reaches the shared position_2d MLP and mil_norm, which feed all twelve targets. Other-label effects are still possible, so an other-eight loss would not falsify the detachment rationale.
- mirror-exchange: without per-family offsets (or a comparison that excludes the biases), the penalty fights the real medial/lateral prevalence gap. Review the implementation for this before launch.
- mirror-exchange: sagittal exchange assumes position_1d is centred on the series extent. The embedding name ('middle_slice_acquisition_midpoint_mm', spatial_tokens.py:50) should be checked at review. In-plane mirroring is approximate under the ≤10° affine rotation. The fixed weight of 0.05 is unmeasured and must not be swept.
- shared-family-level-channel: it adds capacity, and co-occurrence cannot be separated from morphology. A gain is uninterpretable as transfer unless the private-channel control also runs. The logits are already over-coupled relative to the labels (OA 0.854 vs 0.511), which limits the expected headroom.
- cross-family-mediolateral-axis-agreement and the prepared head push along a cross-family lateralization axis that is already over-coupled (0.787 vs 0.305). Their results should be read accordingly.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T020548Z-c14fca6e/report.json (proposals, exhausted_directions diagnostics)
- historical source src/knee_abnormality/modeling/spatial_tokens.py:50-51,253,278 (position_1d and a learned, shared position_2d MLP)
- historical source src/knee_abnormality/modeling/model.py:110,297-316 (mil_norm; token assembly adds position_1d/position_2d)
- historical source configs/training.yaml:83 (dropout 0.0)
- history-index: anatomical-medial-axis-mirror-attention invalid; compartment-invariant-disease-embedding reject; compartment-pair-channel-rotation inconclusive

## completion-20260921T023220Z-731907d9

Rejected as an improvement. Macro 0.89658 → 0.89256. Family means (reference → candidate, interval): meniscus 0.9122 → 0.9084 [−0.0127, +0.0055]; OA 0.8882 → 0.8854 [−0.0102, +0.0049]; other eight 0.8948 → 0.8904 [−0.0101, +0.0007].

Most of the macro loss comes from the other eight labels, which the mechanism does not touch structurally: MCL −0.021 with an unadjusted interval that excludes zero, Fracture −0.016 (63 positives), ACL −0.0095. Baker's gained +0.0073. So the loss is not confined to the tied compartments. It may come from shared-trunk gradient redistribution or from seed variance, which these intervals exclude. It should not be read as the transfer mechanism itself failing.

Compartment diagnostics point toward the feared failure mode, though every interval includes zero:
- Meniscus: lateral false positives rose from 44 to 49, and the share of those on medial-positive studies (cross-talk) rose from 0.636 to 0.735 (28 → 36). Margin AUC went 0.9712 → 0.9670 [−0.019, +0.009].
- OA: margin AUC went 0.8661 → 0.8446 [−0.058, +0.013], while cross-talk stayed flat.

This is consistent with a tied detector firing laterally on medial disease. Lateral OA was essentially unchanged (−0.0002), and Lateral Meniscus fell by 0.0054.

The operator was used rather than left at identity: U has norm 5.46 and V has norm 1.23 at the selected checkpoint. The lateral biases stayed small (norm 0.020).

Scope of the conclusion: this refutes only the single shared rank-16 exchange operator combined with a hard-tied attention/head row, in the replace form with reduced capacity. It does not refute medial/lateral transfer generally, family-specific operators, or softer sharing. There is no matched untied, equal-capacity control, so the effect of tying cannot be separated from the capacity reduction.

The fixed rule gives reject. The macro delta is −0.00402, with a 95% interval of [−0.00807, −0.00004] over 2,000 valid draws. The contract checks pass: the accounting matches the sealed 938,460/35,030 training and 24,188 validation triplets, the only config differences are model.compartment_transfer.*, and the only input-identity difference is preprocessing_identity. Measured compute was 1,590 s, under the 1,797 s ceiling. The review concerns the completion reporter escalated are about attribution, dropout-decoupled training-time tying, a guard that only runs in the training-module constructor, and diagnostic logging. None of them invalidates the run, and the measured timing resolves the runtime concern.

Changing the ranking: I moved shared-family-level-channel ahead of appearance-value-tied-meniscus-classifier. This run's only diagnostic signal on the targeted families points to tying-induced cross-talk (see finding). The additive family channel keeps all four private pathways and starts at zero, so it can recover control B. The hard-tied shared meniscus classifier carries the same cross-talk risk that this tied pathway may have shown. The reserved head and all other positions are unchanged.

Unresolved concerns:
- Attribution is unresolved. Tying, the capacity change (two attention rows and two head rows removed, a rank-16 operator added) and seed variation are confounded. Only a matched untied, equal-capacity control would separate them. That control is a diagnostic, not a plausible winner.
- Only 1.9 points of the 4-point macro loss come from the four compartment labels. The rest comes from untouched labels (MCL, Fracture, ACL). The contributions of the trunk-gradient effect and the one-seed effect cannot be identified.
- The meniscus cross-talk increase (28 → 36 medial-positive lateral false positives) rests on small counts and a margin interval that includes zero. It is a hypothesis for the other tied candidates, not an established effect.
- The training/evaluation tie mismatch from independent dropout masks remains in the design. A future operator variant should tie the dropout mask or apply dropout before the shared row.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/compartment-exchange-operator-tied-pathway-20260921T013725Z-53f5f896/comparison/comparison.json (family_means, compartments, paired_report_group_bootstrap.intervals, parameters_only_in_run)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/compartment-exchange-operator-tied-pathway-20260921T013725Z-53f5f896/comparison/per_label.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T014726Z-e5435947/report.json (implementation review concerns)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T013725Z-7ec79750/report.json
- Evidence packet facts: seed 794186307, measured_compute_seconds 1590.06, accounting complete with sealed cohort hashes

## review-20260921T024327Z-1817ddf3

1. The in-plane mirror is about the image centre, not a permutation of the anchor grid. ReXNet anchors sit at x=0..192 px on a 224 input, i.e. normalized {-1, -0.713, …, 0.722}. Negating x maps six of the seven columns to within 0.009 of the opposite column. Column 0 maps to x=+1.0, which the position_2d MLP never sees on the real path, so it extrapolates there and the penalty trains that point. Real positional embeddings change only through the shared MLP weights.
2. The auxiliary gradient is not confined to compartment rows. mil_norm and spatial_tokens.position_2d get penalty gradients, and all 12 labels share both. So the claim of structural protection for the other eight labels is only partial, and the weight of 0.05 is what bounds it.
3. The compartment biases b_med and b_lat (in attention[3] and head) are pulled toward each other. This works against the prevalence asymmetry, e.g. meniscus 40.7% vs 15.8%. The optional per-family offsets were not added. AUC is invariant to a per-label shift, so the main risk is how BCE and the penalty trade off in the head rows.
4. Sagittal counterfactual tokens keep the unreversed 3-slice channel order while the slice position is negated. A real reflection reverses both, so the probe is approximate by design ("same tissue, other side").
5. If the run returns inconclusive or reject, that only tests this soft, geometry-only exchange. It does not test whether medial/lateral transfer works in general. Any gain still needs the planned anterior-posterior mirror control before it can be attributed to mediolateral geometry rather than regularization.

I approve the diff for launch. It implements the admitted mechanism as specified, with the config change limited to trainer.compartment_transfer.weight (inside the allowed trainer.compartment_transfer prefix). The pass-through (config-unset) path is unchanged, and training_step's split into encode_batch/pool_encoded is the same computation as forward. The mirror branch adds no RNG draws, so there is no leakage and no validation dependence on the training seed. Validation and prediction do not call it. The cohort, labels, masks and triplet accounting sources are untouched. The added compute is one LayerNorm, a 256→128→4 scoring pass and a pooling einsum over about 10k tokens. That should be well under 1% of backbone cost, but it is unmeasured. The concerns below are about interpretation and approximation, not contract breaches, so no repair is needed before measurement.

Unresolved concerns:
- The column-0 anchor mirrors to x=+1.0, off the trained position grid (real max 0.722). The penalty trains the shared position_2d MLP there, which can indirectly move the real positional embeddings for all labels.
- The penalty gradient reaches mil_norm and position_2d, which all 12 labels share. Effects on the other eight labels are therefore possible and should be read with that in mind.
- The penalty pulls the medial and lateral compartment biases toward each other, against the real prevalence asymmetry. The optional per-family offsets were not added.
- Wall-clock overhead is unmeasured. The developer tested only fabricated CPU tensors, never real batches or bf16-mixed on GPU, so non-finite or shape errors on real data remain possible.
- Attribution: a gain would need the matched anterior-posterior mirror control to separate mediolateral exchange from generic label-free regularization.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/mediolateral-mirror-exchange-consistency-20260921T023300Z-7fa3cc82/changes.diff
- candidate source/src/knee_abnormality/modeling/model.py:278,302-305 (mirror geometry), 328-397 (pool_encoded_mirrored, detached features/attention[0]), 313-326 and 400 (unchanged main path)
- candidate source/src/knee_abnormality/modeling/spatial_tokens.py:214-233 (anchor grid x in {-1..0.722}, normalized with input_width-1)
- candidate source/src/knee_abnormality/training/compartment_transfer/mirror.py:10-35 (index resolution, stop-gradient bidirectional smooth-L1)
- candidate source/src/knee_abnormality/training/lightning.py:238-244,316-336; forward at 269-270
- candidate source/configs/training.yaml:15 (reflection p=0.5), 83-86 (dropout 0.0, gated_attention false, residual_readout false)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T023301Z-cca9161c/report.json

## completion-20260921T025928Z-26ae4028

Tying one shared mediolateral attention direction across the meniscus and OA families hurt most on meniscus, the targets the mechanism was meant to help.
- **Meniscus:** the family delta is −0.0090, with a paired interval of [−0.0170, −0.0006] that excludes zero. Medial meniscus fell 0.0117, with a label interval of [−0.0196, −0.0041]. Lateral meniscus fell 0.0062, and its interval includes zero.
- **OA:** the family delta is −0.0044, interval [−0.0122, +0.0032].
- **Other eight labels:** −0.0018, interval [−0.0063, +0.0027]. MCL gained 0.0040 and Baker's 0.0021, both with intervals including zero.
- **Compartment margins:** meniscus margin AUC went from 0.9712 to 0.9659 (interval [−0.0150, +0.0028]). OA margin went from 0.8661 to 0.8627, with a wide interval.
- **Cross-talk:** meniscus lateral cross-talk fell from 0.636 to 0.556, but that rests on only 45 lateral false positives. The co-occurrence structure did not improve in any interpretable way.
- **Learned magnitudes:** the final magnitude norm was 1.31, so the magnitudes did not collapse, and attention stayed lateralized through the shared direction.

Interpretation: taking free per-compartment attention rows away from the medial meniscus (the most prevalent compartment label, 356 positives) appears to cost more than any cross-family transfer gains. Medial and lateral meniscus attention probably needs family-specific or non-linear displacement, not one shared axis.

These exploratory, unadjusted label and family intervals do not override the macro verdict. The result tests only this hard, cross-family, single-direction tie. It does not refute medial/lateral transfer in general, or softer and additive sharing.

The result is inconclusive under the fixed rule. The candidate scored a macro AUC of 0.89320 against the reference's 0.89658. The delta is −0.00340, with a 95% interval of [−0.00723, +0.00017]. The interval barely includes zero and sits almost entirely on the harmful side. Control B remains the incumbent.

The run met the contract:
- The only config difference is model.compartment_transfer.pairs; the name, output-root and seed differences are allowed.
- The only input-identity difference is preprocessing_identity, as permitted.
- Accounting is complete: 938,460 triplets, 35,030 visits, sealed cohort hashes, 4,380 steps.
- Measured compute was 1,585.6 s against the 1,797 s ceiling, so the per-step .tolist() sync did not threaten the budget.

The unresolved review concerns don't invalidate the run. The attention-temperature bake mismatch affects only a later export, and the weight-decay asymmetry is a confound rather than a contract breach. Both do limit how the result can be attributed (see concerns).

The queue is unchanged. The mirror-exchange head doesn't tie anything, so this result gives no reason to reorder it. Evidence against a hard tie of attention geometry favours the additive and free-scale mechanisms (shared-family-level-channel) over further tying. They already rank above the tied-classifier and axis-agreement ideas, so their order stands.

Unresolved concerns:
- This is a one-seed, fold-0 result, and the intervals exclude training-seed variance. The meniscus-family interval excluding zero is exploratory and unadjusted, not a confirmed harm.
- The shared direction and magnitudes were exempt from weight decay, while the parent's compartment rows were decayed (lightning.py:43). The initialization norm also differed by about √2. The negative trend therefore cannot be attributed to the tying constraint alone.
- Exporting with a baked attention temperature (inference/checkpoint.py) would leave disease/direction/magnitude unscaled. This does not matter now because the idea is not adopted, but the defect remains in the candidate source.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/shared-lateralization-direction-attention-20260921T020547Z-5e8cea71/comparison/comparison.json (paired_report_group_bootstrap.intervals: all, meniscus, oa, other_eight, margins; compartments; parameters_only_in_run)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/shared-lateralization-direction-attention-20260921T020547Z-5e8cea71/comparison/per_label.json (Medial Meniscus delta −0.0117, CI [−0.0196, −0.0041])
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T021719Z-b7e0f721/report.json (prior implementation review and concerns)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T020547Z-f75f5653/report.json
- Supervisor facts: seed 3050233464; measured_compute_seconds 1585.64 ≤ 1797; accounting complete with 938,460 triplets and sealed cohort hashes

## review-20260921T031040Z-594ead42

The candidate tests additive shared evidence: each family pools its own shared attention row, and each compartment reads that shared evidence through its own free signed scale, on top of untouched private pathways. The scale can be positive, negative or zero per compartment. So after training the learned family_scales show how the model used the shared channel: as common family evidence (same sign), a contrast channel (opposite signs) or not at all (near zero). Read them from the selected checkpoint before interpreting any macro change as transfer. Weight decay of 0.01 applies to family_scales because the tensor has ndim 2 (lightning.py:43), which adds a mild pull toward zero. That pull is part of the method as implemented.

I approve this for launch. The diff does what the idea describes and stays within the policy: it widens the attention and head layers by one shared row per family and adds a zero-initialised signed per-compartment scale.

At initialisation the scales are zero, so the public logits equal control B's. Early gradients reach the scales through the randomly initialised family scores, and the shared rows start learning once the scales move away from zero. This is the standard bilinear zero-start, so optimisation does not stall.

The only effect on data or preprocessing is the model.py change to the preprocessing hash, which the policy allows. Cohorts, labels, masks, sampling, reflection and validation code are unchanged, and validation still does not depend on the training seed.

The added cost is two pooled rows and a 12×2 scalar update per batch, which should fit well within the 1,797 s ceiling. Timing still has to be measured.

The target mapping is hard-coded, and the only guard is target_count == 12. I checked it against the column order of the reference validation_predictions.csv (ACL, MCL, Medial/Lateral Meniscus, Medial/Lateral OA, …) and it matches.

Both candidate files parse without errors. I did not run any GPU or real-batch check.

Unresolved concerns:
- Capacity confound: each family adds an attention row (129 params), a head row (257) and two scales. Any gain needs a matched private-extra-channel control (one extra row per compartment with its own scale) before it can be attributed to shared morphology rather than extra pooled capacity.
- The target order is hard-coded in FROZEN_TARGET_ORDER, and the only guard is the target count (model.py:133). It matches the reference prediction columns today, but the model never receives target names, so a reordering would silently mismatch.
- The learned family_scales and each shared row's attention map are not logged. Without them we cannot tell whether the channel was used as shared evidence, as a contrast, or ignored.
- Runtime and numerics were not measured on real bf16 GPU batches; the developer's checks were static or CPU-only.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/shared-family-level-channel-20260921T025958Z-2a6d8b47/changes.diff
- experiments/shared-family-level-channel-20260921T025958Z-2a6d8b47/source/src/knee_abnormality/modeling/model.py:130-176 (family index resolution, widened attention/head, zero-init family_scales)
- same model.py:277-292 (shared family score added to medial/lateral logits; only 12 logits returned)
- same source src/knee_abnormality/training/lightning.py:37-48 (name-based LR grouping; ndim>=2 weight decay applies to family_scales)
- same source src/knee_abnormality/inference/checkpoint.py:104 (strict state-dict load from recorded config)
- /home/claudio/src/knee-abnormality/outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat/fold_0/validation_predictions.csv header (target order matches FROZEN_TARGET_ORDER)
- /home/claudio/src/knee-abnormality/autoresearch/campaigns/medial-lateral-transfer/prior-work.md:80-95 (retired 1A/1B/1C/soft-coupling/adapter mechanisms; this design is distinct)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T025959Z-e96bf026/report.json

## proposals-20260921T031151Z-5984f887

The researcher's CPU header audit adds usable geometry evidence for image-space mediolateral operations. Columns align with patient X in nearly every coronal/axial series. Coronal/axial FOV centres sit within a median of 2.1/2.4 mm (p90 6.0/7.3 mm) of the sagittal stack's mediolateral midpoint, so a fixed column permutation j↔6−j approximately maps each token to its contralateral homologue at 7×7 resolution. The coordinate side rule resolves about 96.9% of the sampled studies. That is a sample estimate, not fold-0 coverage. The contralateral comparison only tests transfer when compartment attention falls on coronal/axial tokens. Sagittal tokens, the main meniscus plane, get no partner, so the expected signal is stronger for OA than for meniscus. A null result on meniscus would therefore not refute within-knee comparison.

Admitted contralateral-mirror-token-comparison. I ranked it second, behind the reserved head. It is the first campaign mechanism that compares a compartment's tokens with the homologous tokens of the other compartment in the same image. Every earlier arm ties, shares or penalises single-token, pooled-vector or output evidence. It targets the measured OA assignment weakness: OA margin AUC is 0.866 versus 0.971 for meniscus, and lateral OA drops inside medial-positive studies. It keeps the raw evidence, so both-positive cases stay representable. It commutes with whole-knee reflection, adds no batch keys or metadata, and leaves the other eight targets' forward path unchanged. It is also cheap and pre-registers a matched attribution control (row-flip partner). New header evidence supports the column mirror: coronal and axial columns run along patient X in 259/259 and 172/173 series, and FOV-centre offsets have a median of about 2 mm against a 20 mm token pitch. It goes ahead of the tied-classifier and axis-agreement ideas. The tied classifier is closest to already-tested tie/share arms. The axis-agreement idea constrains attention geometry rather than supplying new comparative evidence.

I have left anatomy-supervised-compartment-evidence-split proposed, not admitted, and not parked. Its gate is trained to predict "medial of the knee midline" per token from image content. At inference that is image-based medial/lateral (side) inference. docs/research/knee-side-and-location.md:205-206 records image-based side inference as deferred by user instruction. The campaign's general permission for "anatomical assignment" does not clearly override a specific user deferral, and the researcher flagged the same issue. It also reuses the plumbing of the invalid axis arm, and its side labels rest on a 128-study header estimate, not on fold-0 coverage. It needs explicit user confirmation before admission. If the user confirms, the proposal's own rule applies: if shared-family-level-channel is accepted, re-express this idea as a replacement, not a stack.

Unresolved concerns:
- contralateral-mirror-token-comparison: its assumption that the backbone does not already encode within-knee asymmetry at stride 32 is unmeasured. So is the share of compartment attention on coronal/axial versus sagittal tokens, which bounds the meniscus effect.
- contralateral-mirror-token-comparison: the column mirror assumes the image centre is near the knee midline. Offsets at the p90 tail (6–7 mm, about a third of a token) and any series outside the header sample may pair non-homologous tissue. The invalid-partner mask covers only visual validity, not this misalignment.
- contralateral-mirror-token-comparison: the Δ stream adds capacity (about 49k parameters) to the four compartment rows. Attributing any gain to the homologous comparison needs the row-flip matched control, not only the macro verdict.
- anatomy-supervised-compartment-evidence-split: it conflicts with the user's recorded deferral of image-based side inference (docs/research/knee-side-and-location.md:205-206). It needs user confirmation before admission, and fold-0 gate-supervision coverage is also unknown.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T025959Z-9d82f946/report.json (proposals, header-geometry evidence, exhausted directions)
- /home/claudio/src/knee-abnormality/docs/research/knee-side-and-location.md:204-206 (image-based side inference deferred by user instruction)
- /home/claudio/src/knee-abnormality/autoresearch/campaigns/medial-lateral-transfer/prior-work.md:80-98 (retired side/compartment variants; knee-side embedding ignored)

## completion-20260921T032637Z-a72abef3

Forcing the lateral readout to match the medial readout on mirrored geometry is harmful as implemented. Macro fell 0.0068, and the damage sits in the meniscus family, the OA family, and above all the OA compartment margin (−0.051). That margin was already the weakest assignment axis, 0.866 against 0.971 for meniscus. The penalty climbed through training (0.07→0.19), so BCE and the exchange consistency were in direct conflict: correct per-compartment prediction needs readouts that differ by side, and this constraint penalises exactly that difference. Future arms should not constrain the medial and lateral readout functions toward invariance under side exchange, whether through tying, symmetric consistency, or bias pulling, unless the side-specific offset or contrast stays free. The result supports queued mechanisms that keep or increase side-specific evidence, such as the contralateral comparison, over ones that symmetrise it. Scope limit: this rejects only the soft exchange at weight 0.05 on detached features. It does not refute medial/lateral transfer. The effects of the off-grid column-0 mirror, the bias coupling and the shared-parameter leak cannot be separated, and there is no anterior-posterior control. One seed, fold 0, and the bootstrap excludes seed and selection variance.

I found no contract breach, so the eligible reject stands. Seed 833238464 is new and the only config difference is trainer.compartment_transfer.weight=0.05. Input identity differs only in preprocessing_identity, which the campaign allows. Triplet accounting matches exactly (938,460/35,030, cohort hashes match). The run used 4,380 steps, and measured compute of about 1,595 s is under the 1,797 s ceiling. Macro AUC is 0.8898 against 0.8966, Δ −0.0068 [−0.0109, −0.0028]. The selected epoch is 8.

The harm is concentrated on the transfer targets, which fits the mechanism causing damage rather than seed noise. The meniscus family fell 0.019 [−0.030, −0.009]. The OA family fell 0.010 [−0.019, −0.002]. Most telling, the OA compartment margin AUC dropped from 0.866 to 0.814, Δ −0.051 [−0.089, −0.016]. Lateral cross-talk rose in both families (meniscus 0.64→0.74, OA 0.62→0.69). The other eight labels moved −0.003 [−0.008, +0.002], so the leak through the shared mil_norm and position_2d was small by comparison. MCL (+0.006) and Baker's (+0.006) are exploratory and unadjusted.

The logged penalty grew steadily from 0.072 in epoch 0 to 0.191 in epoch 9 while BCE fell. So BCE kept separating the medial and lateral readouts and the exchange constraint kept resisting it. Symmetric counterfactual consistency on the compartment rows works directly against compartment assignment. This matches the review concern that it pulls b_med and b_lat together and ties the rows. The concern that the column-0 anchor mirrors off the trained grid may have added to it, but the data cannot separate the two.

The queue is unchanged. No queued idea depends on this one. The anatomy-supervised proposal stays proposed, pending user confirmation of the image-based side-inference deferral. This result gives no new reason to admit it.

Unresolved concerns:
- The causes cannot be separated: symmetric row/bias coupling, the extrapolated column-0 anchor at x=+1.0, and penalty gradients into the shared mil_norm/position_2d may all contribute. Without an anterior-posterior mirror or bias-free variant, the failure cannot be pinned to the mediolateral exchange itself.
- The GPU-side overhead of the counterfactual pass was never measured separately. The total stayed within budget (about 1,595 s against 1,797 s), but the per-step cost is unknown.
- The candidate selected epoch 8 and the reference epoch 9. The difference is small (0.8898 vs 0.8894 at epoch 9) and does not change the interpretation.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/mediolateral-mirror-exchange-consistency-20260921T023300Z-7fa3cc82/comparison/comparison.json (family_means, compartments, paired_report_group_bootstrap.intervals: meniscus, oa, other_eight, oa_margin)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/mediolateral-mirror-exchange-consistency-20260921T023300Z-7fa3cc82/training/fold_0/logs/version_0/metrics.csv (train_compartment_mirror_penalty_epoch 0.072→0.191; val_macro_auc per epoch)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/mediolateral-mirror-exchange-consistency-20260921T023300Z-7fa3cc82/comparison/per_label.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/mediolateral-mirror-exchange-consistency-20260921T023300Z-7fa3cc82/changes.diff
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T024327Z-209fd856/report.json (pre-launch review concerns: bias coupling, off-grid anchor, shared-parameter gradients)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T031151Z-4acefbe8/report.json (anatomy-supervised proposal held pending user confirmation)

## review-20260921T033725Z-7012c3b3

The candidate starts exactly equal to control B and adds about 49k parameters. It introduces an intra-image mediolateral comparison channel: each compartment readout sees tokens enriched with the features from the opposite side of the same coronal or axial image. That transfer differs from the shared backbone because it pairs specific spatial locations across the midline instead of sharing weights. When interpreting results, a macro change can come from the extra per-token MLP capacity (value_self alone) rather than from comparison with the mirror token. Attributing a gain to mirror comparison therefore needs a matched self-only ablation (value_mirror removed, same width). A change confined to the eight untouched labels is impossible by construction, since their logits come only from the base pass. Any movement in those labels comes through the shared backbone, attention and head training, not through Δ.

The candidate addresses the assigned topic. Each non-sagittal token receives a learned residual computed from its own feature and the feature of its column-mirrored partner on the backbone's 7×7 grid. Only the four compartment readouts consume that residual. The attention and head are shared across both passes, so each compartment's own attention row still performs the anatomical assignment, and all four medial/lateral outcome combinations remain representable. Source checks: features come from maps.flatten(2) in row-major order (model.py:305), so the rows*W+(W-1-col) permutation is a true in-image horizontal mirror. Sagittal index 2 matches plane_order (spatial_tokens.py:53) and model.py:289. Δ is zeroed where the mirror token is invalid, and invalid tokens are already excluded by token_valid in pooling. The new parameters fall in the head_lr group, and combine.weight receives weight decay (lightning.py:40-43), which belongs to the method. Temperature baking scales only attention[-1], which both passes share, so it remains consistent. Validation stays deterministic: Δ has no stochastic component, and reflection is applied only in training. There is no leakage path: nothing is fitted outside the ordinary training step, there are no new batch keys, and preparation took 0.17 s. The added compute is one extra small attention/pooling pass plus a 64-wide MLP per token, far below the backbone cost. It is unmeasured, but the 1,797 s ceiling is enforced at runtime.

Unresolved concerns:
- The mirror is about the center of the augmented, cropped image, not the knee's anatomical midline, and it assumes the image columns run left–right in axial and coronal series. If a knee is off-center, rotated by augmentation, or stored in a nonstandard orientation, the pairing is only approximate. The learned MLP may absorb this, but the pairing is not an anatomical correspondence.
- Capacity confound: about 49k added parameters, including a per-token self MLP. A gain does not establish mirror-comparison transfer without a matched self-only Δ control.
- Runtime and bf16 numerics were checked only statically and on CPU. The second pooling pass and per-token MLP are not measured against the 1,797 s ceiling.
- The four compartment targets' first-pass logits are computed and then discarded, which wastes a little compute but has no correctness impact. Their attention and head rows are trained only through the augmented pass, which makes the rows shared with the other 8 targets see mixed-pass gradients. This is expected but should be kept in mind when comparing attention maps with control B.
- Neither Δ's magnitude nor the norm of the combine weights at the selected checkpoint is logged. Inspect them before claiming the channel was used.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/contralateral-mirror-token-comparison-20260921T032728Z-25105e4b/changes.diff
- candidate source src/knee_abnormality/modeling/compartment_transfer/mirror_token_comparison.py:10-53 (mirror permutation, zero-init combine, sagittal/partner masking)
- candidate source src/knee_abnormality/modeling/model.py:305 (row-major token flatten), 289 (sagittal==2), 326-353 (dual pass, overwrite of columns 2-5)
- candidate source src/knee_abnormality/modeling/spatial_tokens.py:53 (plane_order Axial,Coronal,Sagittal)
- candidate source src/knee_abnormality/training/lightning.py:35-51 (head_lr group, weight decay for ndim>=2)
- candidate source src/knee_abnormality/inference/checkpoint.py:78-86 (temperature baking on the shared attention[-1])
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T032728Z-5efdb8df/report.json (init parity, only columns 2-5 change, 49,216 params, runtime not measured)

## proposals-20260921T034043Z-26d64894

The researcher decomposed control B's fold-0 predictions (scratch/crosstalk.py, coupling.py). Each compartment's AUC is much lower against negatives whose sibling compartment is diseased:

| Target | vs sibling-diseased negatives | vs sibling-negative negatives | n |
|---|---|---|---|
| Lateral OA | 0.722 | 0.909 | 143 |
| Medial OA | 0.778 | 0.921 | 55 |
| Lateral meniscus | 0.852 | 0.924 | 281 |

Within the both-positive vs exclusive split, lateral OA scores 0.760 and lateral meniscus 0.879.

The logit correlation between the two compartments is 0.854 for OA, against a label correlation of about 0.51–0.56. For meniscus it is 0.389, against about 0.13–0.22. That points to knee-level family evidence leaking into both compartment logits.

Limits of this evidence:
- It is post-hoc analysis of the same validation set that judges candidates. The strata are exploratory motivation, and they carry no weight for acceptance.
- Hard labels were thresholded at 0.5.
- Some strata are small (medial OA n=55).
- There are no intervals.

A candidate that improves these strata without a macro interval above zero is still inconclusive. A macro gain concentrated elsewhere does not demonstrate the claimed locality or bilateral mechanism.

The two new ideas are complementary: one is a training-only locality supervision, the other an inference-path bilateral statistic. The PIT-trained detector could later serve as the value row for the order-statistic channel, but that would need its own integration review.

Admitted both proposals. Neither has dependencies, and both are in scope. Each uses one family-shared detector across the two compartments, which is shared morphology, and each keeps all four joint-label groups representable through the untouched private pathways.

permutation-invariant-half-knee-mil goes second, right after the reserved head. It is the first arm to add a constraint that main BCE lacks: in an exclusive study, evidence must sit in one geometric half of the knee. It needs no side inference and changes nothing at inference time, so there is no capacity confound in the public path. It attacks the largest cross-talk deficit in the reference predictions: lateral OA scores 0.722 against medial-only negatives versus 0.909 against sibling-negative negatives. Earlier shared auxiliaries mostly failed because they pooled with learned attention (no locality) or used large weights, and this design avoids both. Remaining risks are trunk degradation from any auxiliary loss and misalignment between image halves and compartments.

bilateral-order-statistic-evidence goes last. It adds inference-path capacity. Its `hi` statistic largely duplicates the running shared-family-level-channel, and its `lo` statistic partly overlaps contralateral comparison. So it is best built after both of those results exist, which lets the integration plan decide between stacking and replacing hi (declaring the removed idea). Its unique contribution, the `lo` statistic for bilateral disease, is plausible: a softmax pool behaves like a max and cannot express the minimum over two halves.

The existing queued ideas keep their relative order. anatomy-supervised-compartment-evidence-split stays proposed while side metadata remains unresolved.

Unresolved concerns:
- PIT arm: the half split uses image columns around the prepared-image anchor and the series-centred sagittal slice sign. Two things are unmeasured: sagittal coverage of both halves under 26 triplets, and the offset between image centre and the knee notch. The implementation must log per-study half coverage and assignment stability; otherwise a null result cannot separate noisy halves from a failed mechanism.
- PIT arm: the auxiliary weight, δ and smooth-max temperature must be fixed before launch, from training-only reasoning. They must not be tuned on validation. Earlier shared auxiliaries at 0.3 damaged the other eight labels, so check whether the other-eight family mean falls even if the macro stays inside its interval.
- PIT arm: attributing any gain requires two controls: private per-compartment detectors under the same loss (to test sharing), and an anterior-posterior split (to test mediolateral geometry).
- Order-statistic arm: hi overlaps the running shared-family-level-channel, and lo partly overlaps contralateral comparison. After those results arrive, the integration plan must declare either stacking or replacement (with removed_idea_ids). It also adds about 800 parameters plus 8 zero-initialised scalars at inference, so a gain needs a capacity-matched anterior-posterior control.
- The motivating strata come from post-hoc analysis of the fold-0 validation set. Later interpretation must not treat improvement in those same strata as independent confirmation.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T032728Z-3519d993/report.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T032728Z-3519d993/scratch/crosstalk.py (sibling-stratified AUCs on control B fold-0 predictions, labels thresholded at 0.5)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T032728Z-3519d993/scratch/coupling.py (pair logit correlation and conditional AUCs across reference and campaign runs)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T025959Z-9d82f946/report.json (header geometry: image columns along patient X, FOV centre about 2 mm from sagittal midpoint)
- /home/claudio/src/knee-abnormality/outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat/fold_0/validation_predictions.csv

## completion-20260921T035352Z-f3073e34

shared-family-level-channel (seed 3884041694) is inconclusive: macro +0.00089, 95% interval [-0.0030, +0.0047]. The shared channel was dormant at the selected checkpoint: |family_scales| ≤ 0.008, norm 0.0092. Each compartment logit therefore received only about 1e-2 of shared input, and the mechanism was effectively not exercised. The cause is the multiplicative zero start (scale × shared row, with no gradient to the shared rows at init) plus weight decay on the scales. With private per-compartment rows that can already fit everything, nothing pushes the model to route evidence through a shared bottleneck. Family AUC was flat (meniscus +0.0003, OA +0.0001). Both medial/lateral margin AUCs fell without significance (meniscus -0.006, OA -0.015). The small macro gain came from labels outside the four compartment targets (PF OA, MCL, Effusion up; Synovitis down), which is consistent with seed variation, not transfer. Implication for later ideas: purely additive, zero-gated shared paths next to unconstrained private paths are likely to stay unused. Mechanisms that force sharing (tied classifiers, side-agnostic shared detectors, or losses applied directly to the shared evidence) should be tested before more gated-addition variants. Do not rerun this idea with a larger scale init or no decay on the scales: that would be a coefficient sweep, not a new mechanism.

The fixed verdict is inconclusive. Macro AUC went from 0.89658 to 0.89747, a difference of +0.00089 with 95% interval [-0.00296, +0.00472]. The contract checks pass. Accounting matches the sealed cohort exactly: 938,460 triplets, 35,030 visits and 4,380 steps. The run used 1,585 s against the 1,797 s ceiling. The only config differences are the allowed ones, and the only input-identity difference is preprocessing_identity.

The escalation came from unresolved implementation-review concerns. I resolved the most important one, whether the channel was used, by reading the selected checkpoint (epoch 7) on CPU. The channel was essentially unused. family_scales = [[-0.0078, -0.0046], [-0.0009, -0.0013]] for meniscus and OA (medial, lateral), with norm 0.0092. The shared head rows ended with norms of about 0.57, similar to the private rows. With scales this small, the shared score shifts each compartment logit by only about 1e-2. That is negligible next to the private logits.

The cause is in the design, not the data. The output is scale × shared score, and the scale starts at zero. At initialization the shared attention and head rows get no gradient, and the scale only learns through a random shared score. The 0.01 weight decay on the 2-D scale tensor pulls it back toward zero. The channel therefore never switched on. This result does not test whether shared morphology transfers between compartments. It says nothing either way about the campaign hypothesis. It does show that a zero-gated additive shared channel, on top of fully expressive private pathways, gets no gradient pressure to become active in 10 epochs.

The label pattern fits seed noise, not transfer. Family deltas are close to zero: meniscus +0.0003 and OA +0.0001. Both compartment-margin diagnostics fell: meniscus margin -0.0064 with interval [-0.019, 0.006], OA margin -0.0147 with interval [-0.046, 0.015]. The macro change comes mostly from untouched labels: PF OA +0.009, MCL +0.009, Effusion +0.006 and Synovitis -0.012. Synovitis is the only label whose exploratory interval excludes zero. None of these labels read the shared channel. Prediction Spearman against the reference is 0.89–0.97 and the mean absolute probability shift is 0.068, both typical of a different seed. The capacity-confound concern is moot because the added capacity was not used.

The queue is unchanged. I did not admit the pending proposal anatomy-supervised-compartment-evidence-split from this completion event, because it has not been reviewed here. It stays proposed.

Unresolved concerns:
- Scale and attention-map activity were checked only in the selected checkpoint's weights. The shared row's actual score magnitudes were not measured on validation batches, so the 'about 1e-2 logit' estimate is an order-of-magnitude bound.
- One seed on fold 0, with the recorded seed-level spread in prediction ranks (Spearman 0.89–0.97). The out-of-channel label movements, including the Synovitis drop with interval [-0.022, -0.001], are exploratory and uncorrected. They should not guide later designs.
- The target-order guard in FROZEN_TARGET_ORDER (model.py:133) is still count-only. It does not matter now that this idea is not accepted, but any successor that reuses it should resolve indices from target names.

Evidence:
- outputs/autoresearch/medial-lateral-transfer/experiments/shared-family-level-channel-20260921T025958Z-2a6d8b47/comparison/comparison.json (paired_report_group_bootstrap.intervals, compartments, family_means, parameters_only_in_run.family_scales norm 0.00924)
- training/fold_0/best-epoch=07-val_macro_auc=0.89748.ckpt: model.family_scales = [[-0.0078,-0.0046],[-0.0009,-0.0013]]; model.head rows 12-13 norms 0.569, 0.585
- changes.diff lines 164-187: family_scales zero-init; the public logit adds scale × shared family score
- jobs/orchestrator-20260921T031040Z-3fb057bb/report.json (approval review; concerns carried forward)
- evidence.json accounting: 938,460 triplets / 35,030 visits / 4,380 steps; timing 1,585.0 s < 1,797 s

## review-20260921T040834Z-48a1ef53

Approved for launch. The transfer is a training-only PIT-MIL auxiliary (weight 0.075, T=1, ±10 mm band). Its effect on the public 12 logits can only come through shared backbone, projection and mil_norm updates. Interpretation guide: a genuine transfer effect should appear on Medial/Lateral Meniscus and OA and their family means, with compartment-margin diagnostics not worse. Movement in the other eight labels would reflect shared-representation side effects, not the mechanism. Attribution needs care: a gain could arise from added localised task weight rather than cross-compartment sharing. A matched non-permutation or independent-per-compartment detector control would separate these. Inspect logged half_coverage and assignment_stability; low coverage or stability near 0.5 would mean the auxiliary signal was weak or the geometric halves were poorly aligned.

The diff implements the assigned mechanism faithfully. One family-shared detector scores mil_tokens and is pooled by smooth-max within two geometric halves per series: axial/coronal use x in the prepared frame (inverse affine including reflection_inverse) about the image centre with a ±delta_px band; sagittal uses reflection-adjusted relative_position_mm with a ±10 mm band. The loss is min-over-permutation class-balanced BCE against the medial/lateral pair. Training targets only; no validation access, no extra encoder pass, no preparation change, no label change. Concordant pairs supervise both halves identically. Exclusive pairs teach localisation without side identity, so all four label combinations stay representable. Main-path equivalence and weight convention (weights[0] for target≥0.5) match masked_binary_cross_entropy_with_logits (lightning.py:110-124). Series enter only when both halves have visually valid tokens and both pair labels are observed. This adds localised supervision to the shared backbone, beyond what shared weights already provide. No gate failure; approve.

Unresolved concerns:
- Runtime is unmeasured (accounting_available=false; only CPU smoke checks). The per-step overhead (a small Linear, two torch.unique calls per family, one nonzero with a host sync) is likely small but is not demonstrated against the 1,797 s ceiling.
- The half split assumes the prepared-image centre (axial/coronal) and the fitted study centre (sagittal relative_position_mm) approximate the knee midline. An off-centre knee puts compartment tissue in the wrong half; the 10 mm band only partly mitigates this.
- Detector logits run in bf16 under autocast, so the scatter smooth-max exp/sum has limited precision. The loss is averaged per series, not per study, so multi-series studies get more auxiliary weight.
- Attribution: the added supervision and task weight are confounded with the sharing mechanism. A macro gain would not by itself establish shared compartment morphology without a matched non-shared or non-permutation control.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/permutation-invariant-half-knee-mil-20260921T035443Z-6d769087/changes.diff
- candidate source src/knee_abnormality/modeling/model.py:303-344 (training-gated compartment dict; forward = pool_encoded(encode_batch))
- candidate source src/knee_abnormality/training/compartment_transfer/loss.py (half_assignment, _segment_smooth_max, _family_term min-over-permutation)
- candidate source src/knee_abnormality/training/lightning.py:110-124 (main BCE weight convention matched by aux _weighted_bce)
- candidate source src/knee_abnormality/data/preprocessing/preparation.py:194 (slice positions = relative_position_mm)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T035443Z-623fe1f0/report.json

## completion-20260921T042113Z-bd940cdc

The mirror-token comparison was inconclusive on macro AUC, and its point estimates lean negative: −0.0030 [−0.0068, +0.0010]. The run was contract-valid: 1,591 s of the 1,797 s budget, exact triplet and visit accounting, and only the declared config and identity differences. The mechanism was active: the zero-initialised combine weight reached norm 1.17 at the selected epoch 8. Even so, none of the targeted diagnostics improved:
- Meniscus family: −0.0048.
- OA family: −0.0076 [−0.0156, +0.0002].
- Meniscus margin AUC: −0.0030 [−0.0156, +0.0089].
- OA margin AUC: −0.0048 [−0.0362, +0.0285].
- OA lateral cross-talk: 0.622→0.647.

Exploratory, unadjusted label intervals show Medial Meniscus at −0.0097 [−0.0175, −0.0023]; Lateral Meniscus was flat. Enriching compartment-readout tokens with their column-mirror partner therefore did not sharpen compartment assignment. It may have added noise to the medial readouts, which have higher prevalence. The other eight labels (−0.0015) moved only through shared training; the largest changes were MCL −0.016 and Baker's +0.0075. The self and mirror projections have nearly equal norms, so how much the model relied on mirror comparison versus plain per-token capacity cannot be identified. The prediction rank correlation with control B is 0.91–0.96 on compartment labels. This is a single seed on a single fold, so the uncertainty excludes seed and selection variation.

Completion interpretation only. The queue is unchanged. anatomy-supervised-compartment-evidence-split stays proposed: its image-based medial/lateral gate still conflicts with the user's recorded deferral of image-based side inference, and nothing in this event resolves that.

The run is valid. Accounting matches exactly: 938,460 training triplets, 35,030 visits, sealed cohort hashes, validation cohort constant across epochs, and 4,380 steps. Measured compute was 1,590.9 s, under the 1,797 s ceiling, which settles the unmeasured-runtime concern from review. The config differs only in the two campaign-prefixed keys, and the only input-identity difference is preprocessing_identity (model.py changed). The extra harness files in the source archive are autoresearch code, not training behavior.

Macro delta is −0.00304, 95% CI [−0.00683, +0.00097]. The interval includes zero, so the fixed rule gives inconclusive. The point estimate and every mechanism-relevant diagnostic lean negative, and no label group shows a transfer signal:
- Meniscus family mean: −0.0048 [−0.0136, +0.0043].
- OA family mean: −0.0076 [−0.0156, +0.0002], almost entirely below zero.
- Compartment margin AUCs moved slightly down: meniscus 0.971→0.968, OA 0.866→0.861, both with wide intervals. The proposal targeted the OA assignment deficit, but OA cross-talk rose (0.622→0.647) and medial-positive false positives for lateral OA rose from 51 to 55.
- Medial Meniscus: −0.0097, exploratory label interval excludes zero. Lateral Meniscus: flat.
- Other-eight mean: −0.0015 [−0.0059, +0.0034]. MCL −0.016 and Baker's +0.0075 move through shared training only, since those logits bypass Δ. This is consistent with seed and trajectory variation.

The channel was used: at the selected checkpoint (epoch 8) the zero-initialised combine weight reached norm 1.17. So the result is not a dead-channel null. It is also not attributable to mirror comparison as such, because value_self and value_mirror have similar norms (4.70 vs 4.64) and no self-only control exists. Running that capacity control now would only help to explain a gain, and there is no gain, so it is not justified.

This is one implementation, one seed, one fold. It does not refute intra-image mediolateral comparison in general. It does count against revisiting column-mirror token enrichment with only a width change.

Unresolved concerns:
- The effect of mirror comparison cannot be separated from the extra per-token capacity: value_self and value_mirror norms are similar and there is no self-only control. The null or negative result therefore cannot be pinned on either component.
- The mirror is taken about the centre of the augmented image, not the knee's anatomical midline. A misaligned pairing under affine augmentation may explain the absent margin gain, but that was not measured: Δ magnitude and pairing quality were never logged.
- The Medial Meniscus decrease (label interval excludes zero) is exploratory and unadjusted for multiplicity. It suggests, but does not establish, harm to the higher-prevalence medial readout.
- Single seed and single fold. Bootstrap intervals exclude training-seed and checkpoint-selection variability. The candidate selected epoch 8; the reference selected epoch 9.

Evidence:
- outputs/autoresearch/medial-lateral-transfer/experiments/contralateral-mirror-token-comparison-20260921T032728Z-25105e4b/comparison/comparison.json (family_means, compartments, paired_report_group_bootstrap.intervals, parameters_only_in_run, prediction_change)
- Same experiment: evaluation.json / comparison/per_label.json — macro −0.00304 [−0.00683, +0.00097]; Medial Meniscus −0.0097 [−0.0175, −0.0023]
- Assignment facts.timing: training 1553.7 s + prediction 20.3 s + setup 0.29 s; measured_compute_seconds 1590.93 against the 1,797 s ceiling
- Assignment facts.accounting: 10 epochs × 93,846 triplets / 3,503 visits, cohort b536b7e1…; validation 24,188 triplets, cohort 7167eba1…
- outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T033725Z-f97fe03b/report.json (implementation review and its unresolved concerns)
- outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T031151Z-4acefbe8/report.json (admission rationale targeting OA assignment; why the anatomy-supervised proposal was held)

## completion-20260921T042210Z-646fec7a

permutation-invariant-half-knee-mil-20260921T035443Z-6d769087 is invalid because of an implementation dtype bug. It is not evidence about the mechanism. The run failed on its first training step under bf16 autocast with `scatter(): Expected self.dtype to be equal to src.dtype`. The likely site is `_segment_smooth_max` (loss.py:37-40): `total` is allocated in bf16, while autocast promotes the `exp()` source to fp32. Nothing was trained, so accounting, runtime, per-label and mechanism diagnostics (half_coverage, assignment_stability) are all missing.

The repair is bounded: compute the smooth-max in fp32 by casting `values` with `.float()`. That also resolves the bf16 precision concern recorded at review. A repaired run needs a fresh seed. The attribution caveat from review still applies: a matched non-permutation or independent-per-compartment control would be needed to credit any gain to shared morphology.

Process lesson: CPU smoke checks without autocast do not catch AMP dtype interactions in scatter and index ops. Reviews of new scatter-based auxiliaries should check dtype under bf16 autocast.

Completion interpretation only. The queue is unchanged. The invalid verdict comes from a crash, not from a result. Training exited with code 2 after about 12 s, on the first step of epoch 0 (0/438). stderr ends with `Training error: scatter(): Expected self.dtype to be equal to src.dtype`. No optimizer step finished, so there is no triplet_accounting.json, checkpoint or predictions.

The likely cause is in `_segment_smooth_max` (training/compartment_transfer/loss.py:34-41). Detector logits arrive in bf16 under Lightning's bf16 AMP, and `peak` and `total` are allocated in `values.dtype` (bf16). CUDA autocast runs `exp()` in fp32, so `total.scatter_add_(0, group, (...).exp())` at line 40 gets a bf16 target and an fp32 source. The developer's CPU smoke checks ran without autocast, and the approving review (orchestrator-20260921T040834Z-804ecf14) raised the bf16 precision risk but did not catch the dtype mismatch.

A one-line change would fix it: cast `values` to float32 at the top of `_segment_smooth_max`, which also settles the recorded precision concern. The comparable allocations at lines 69-70 and 83 use `study_v.dtype` for both target and source, so they look safe; I did not trace the rest of the file. The permutation-invariant half-knee MIL hypothesis produced no evidence either way. This invalid run says nothing about the mechanism, and the idea deserves a repaired run with a fresh seed. Seed 2337625403 was drawn and recorded but produced no training.

anatomy-supervised-compartment-evidence-split stays proposed. Its dependencies are satisfied, but the previous completion report (orchestrator-20260921T042113Z-e106e817) left it proposed because its image-based medial/lateral gate conflicts with the user's recorded deferral of image-based side inference. This event does not change that.

Unresolved concerns:
- The idea is still 'launched' with an invalid experiment and no repaired run queued. An implementation crash should not count as a negative result; it needs a fp32 smooth-max repair and a fresh-seed run.
- The fix is inferred from the stack message and a source read, not from a reproduced traceback. Other scatter or index-put sites in loss.py (lines 69-84) could also mismatch dtypes under autocast and should be checked during the repair.
- Runtime, half-coverage and assignment-stability concerns from review remain unmeasured because training never ran.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/training-20260921T042154Z-82d32a34/stderr.log (final line: 'Training error: scatter(): Expected self.dtype to be equal to src.dtype'; 'Using bfloat16 Automatic Mixed Precision'; seed 2337625403)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/training-20260921T042154Z-82d32a34/receipt.json (exit_code 2, 11.8 s); stdout.log shows Epoch 0 0/438
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/permutation-invariant-half-knee-mil-20260921T035443Z-6d769087/source/src/knee_abnormality/training/compartment_transfer/loss.py:34-41 (bf16 peak/total allocation; scatter_add_ of exp() at line 40)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/permutation-invariant-half-knee-mil-20260921T035443Z-6d769087/evaluation.json (invalid: training failed; triplet_accounting.json missing)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T040834Z-804ecf14/report.json (approving review; bf16 precision concern recorded)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T042113Z-e106e817/report.json (reason anatomy-supervised-compartment-evidence-split remains proposed)

## review-20260921T042850Z-12f60b40

Static review found no correctness, leakage or contract issue. Historical run 1A (outputs/training/meniscus-classifier) already tied a 256-d meniscus classifier with independent biases, but it pooled full tokens that include position. A shared weight there can read a compartment-dependent offset from the position term in each pooled vector, so its tie was effectively loose. This candidate removes position from the pooled value, making the tie strict: compartment identity can come only from attention and bias. That difference, plus the current class-balanced BCE / x1 recipe, is what separates it from 1A. It narrows the design; it is not a new family of mechanism. The implementation also has 256 fewer parameters than control B, so a gain cannot be credited to added capacity. Interpretation should focus on the meniscus pair (1A saw about +0.003 per meniscus AUC with a flat macro), the off-target drift seen in 1A, and the evaluator's compartment-margin and co-occurrence diagnostics.

Approved. The candidate implements the assigned mechanism faithfully and gives the transfer the campaign requires beyond a shared backbone: both meniscus compartments are scored by one classifier weight vector over geometry-free pooled appearance. Medial/lateral assignment comes only from each target's own attention map and bias, so all four cases (neither, medial-only, lateral-only, both positive) remain representable.

Correctness checks on the source:
- Attention is still computed from geometry-bearing tokens.
- The appearance tokens have the same shape and ordering as the pooled tokens, so masking and study membership carry over unchanged.
- The new parameters fall into the existing head_lr groups, and the 1-D bias gets weight decay 0 under the ndim rule.
- Nothing touches the data path, validation or loss, so validation independence from the training seed and mask accounting are unchanged.
- There is no preparation or precomputation step. The only extra computation is one LayerNorm over the existing tokens plus a 2-target einsum, well within the 1,797 s ceiling; the runtime gate will measure it.
- There is no leakage route.

The queue is unchanged. I did not admit anatomy-supervised-compartment-evidence-split in this review event; it stays proposed.

Unresolved concerns:
- value_components is exposed as a configurable tuple, but only the default (features, plane, contrast) is intended. That is speculative configurability, which the project guidance disallows. It has no effect on this run; remove it on any repair or promotion.
- The difference from historical 1A is only that position is removed from the pooled value. A null or negative result would not separate 'strict tie is harmful' from 'the family-shared classifier gives no benefit on this recipe'. Do not generalize from it to the broader hypothesis.
- Runtime was not measured by the developer. The per-token LayerNorm adds a small cost at the real token count; the runtime gate has to confirm it stays under 1,797 s.
- The shared weight starts from default Linear init and the biases start at zero, while the replaced private rows had default random biases. RNG consumption at construction also differs. This is harmless with fresh seeds but means the run is not a seed-matched comparison.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/appearance-value-tied-meniscus-classifier-20260921T042159Z-88812751/changes.diff
- candidate source src/knee_abnormality/modeling/compartment_transfer/appearance_value_tie.py:36-60 (appearance tokens, tied/untied logits, restore_order)
- candidate source src/knee_abnormality/modeling/model.py:140-151, 216-260, 327-346 (head shrink, pooling route, geometry-free tokens)
- candidate source src/knee_abnormality/training/lightning.py:35-51 (new params in head_lr groups; ndim<2 excluded from decay)
- /home/claudio/src/knee-abnormality/data/folds/all_5fold_v3/protocol.json (targets index 2,3 = Medial/Lateral Meniscus)
- /home/claudio/src/knee-abnormality/docs/research/medial-lateral-transfer.md:36-46 (1A: shared 256-d vector over full tokens, independent biases; meniscus +0.0027/+0.0034, macro +0.0002)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T042159Z-ea2d1a63/report.json (CPU build, shape and allclose checks, -256 parameter delta)

## review-20260921T044048Z-5053944f

The candidate is a zero-parameter, training-only hinge. Within each (study, plane) it forces the attention-weighted mediolateral displacement of the OA rows (Medial OA minus Lateral OA) to be at least 10 mm in the direction of the meniscus rows' displacement. The sign is taken from the meniscus rows and detached, with a coefficient of 0.02.

The transfer is one-way. OA attention localization is pulled toward the meniscus side assignment, and meniscus attention gets no gradient from the penalty. Interpretation should therefore look mainly at the two OA targets and the OA family mean. The meniscus pair should change only indirectly, through the shared backbone. The penalty also acts on every study regardless of disease status, so it is an anatomical localization prior, not co-occurrence supervision. A null result would not refute cross-family sharing in general.

I approve the candidate for launch. I inspected the diff and the geometry code it depends on.

Coordinates:
- Coronal and axial tokens use xy[:,0] (spatial_tokens.py:199-247). That is the x anchor in the augmented input, shared by all images, scaled to mm by spacing*size/2.
- Sagittal tokens use relative_position_mm (preparation.py:194, series.py:198-210), with its sign flipped under reflection (augmentation.py:76-79).
- Each plane's mean is taken within one study. The sign comes from that plane's own meniscus delta. So a global reflection or knee side leaves the penalty invariant, and the medial/lateral label identities are never exchanged.

Shapes:
- Token ordering (images×T flattened) is consistent across attention, plane and coordinate.
- Attention on tokens of other studies or on invalid tokens is exactly 0 from the -inf mask.
- The per-plane mean ratio has bounded gradient: d mean/d score_i = a_i(c_i-mean)/W, and a_i ≤ W, so the gradient stays below the coordinate range.

Other checks:
- Validation independence: in eval, mediolateral_coordinate is None and the forward path is unchanged.
- There is no leakage: the penalty uses no labels at all, only attention and geometry.
- Preparation cost is 0.17 s, and the extra per-step einsums over (B,4,N)×(N,3) are negligible.

Scope: the mechanism is anatomical-assignment consistency between the meniscus pair and the OA pair. The contract allows this, and it adds transfer beyond the shared backbone. It adds no capacity, so a capacity explanation does not apply. All four both/neither/single-positive label configurations stay representable, because only attention location is constrained, not the logits.

Unresolved concerns:
- The sign anchor comes from detached meniscus attention. Early in training the meniscus displacement is near zero, so its sign is effectively arbitrary, and the OA rows are pushed ≥10 mm toward an arbitrary side. Nothing forces the meniscus rows themselves to separate. Check the logged penalty and share curves for whether they settle.
- The penalty is measured in mm, so its size depends on the coefficient choice. Before convergence 0.02×relu(10−sign·Δ) can reach about 0.2–0.6, which is comparable to the class-balanced BCE. Check train_compartment_transfer_share to confirm it does not dominate. Coefficient/margin sweeps are out of scope.
- The coordinate conventions are not canonicalized. Coronal/axial x is the image column, and sagittal position is the projection on each series' normal. If two series of the same plane in one study disagree on direction, they mix inside one plane mean and add noise. This could weaken the mechanism but is not a leakage risk.
- The hinge weights every present (study, plane) equally, even planes that receive negligible OA or meniscus attention mass. Constraints on those low-mass planes barely affect pooling and dilute the signal.
- I could not re-run py_compile because the source tree is read-only. An AST parse passes, and the developer reports py_compile, ruff and load_config checks. Runtime and triplet accounting are not yet measured (accounting_available=false), so the runtime gate must confirm ≤1,797 s.

Evidence:
- outputs/autoresearch/medial-lateral-transfer/experiments/cross-family-mediolateral-axis-agreement-20260921T043006Z-27a9bda1/changes.diff
- candidate source/src/knee_abnormality/training/compartment_transfer/mediolateral_agreement.py:8-33 (plane-grouped weighted mean, detached meniscus sign, hinge, presence mask)
- candidate source/src/knee_abnormality/modeling/model.py:239-247 (fp32 attention subset return), 303-316 (training-only mm coordinate)
- candidate source/src/knee_abnormality/modeling/spatial_tokens.py:199-247 (xy is a shared augmented-input anchor grid, T×2)
- candidate source/src/knee_abnormality/modeling/augmentation.py:44-79 (reflection flips coronal/axial image columns and negates sagittal positions)
- candidate source/src/knee_abnormality/data/preprocessing/preparation.py:194 and series.py:198-210 (relative_position_mm is centered physical slice position)
- candidate source/src/knee_abnormality/training/lightning.py training_step (BCE unchanged; penalty added; eval path untouched)
- outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T043007Z-1b60ddb7/report.json

## proposals-20260921T045225Z-dcd8e2e2

New CPU diagnostics on control B's selected epoch-8 checkpoint (exploratory, no intervals; 4 studies for the receptive-field measurement, 32 studies for the token analyses):
- **Opposite-half sensitivity.** Share of a token's input sensitivity that falls in the opposite image half: 0.30–0.32 at stride 32, 0.23–0.24 at stride 16, 0.18 at stride 8. Share within ±32 px: 0.49, 0.61 and 0.74.
- **OA detectors are already shared.** Within a study, the medial and lateral OA token scores correlate at 0.74–0.76 in every OA group. Their attention overlap is 0.64–0.70, against 0.27–0.30 for the meniscus pair. The lateral-OA detector is higher on the medial detector's high half even in lateral-only knees (0.30 vs −0.31).
- **Plane distribution.** Compartment attention mass is split roughly evenly between coronal and sagittal slices (0.35–0.49 and 0.38–0.43), with axial lower (0.12–0.23).

Implication: OA transfer is limited by assignment and locality rather than by missing sharing. The proposal pre-registers two attribution controls, to run only if there is a gain: private per-compartment detectors of the same width (isolates sharing) and the same path on the stride-32 map (isolates locality). Without them, a gain cannot be attributed to shared morphology rather than added stride-16 capacity or backbone gradient. A null result bounds stride-16 locality and does not refute sharing.

Admitted family-shared-local-lesion-evidence. It has no dependencies (proposal_admission satisfied) and is in scope: one detector per family is shared by both compartments, and each compartment reads it through its own existing attention. It covers both the meniscus and OA pairs. All four joint outcomes stay representable because the private stride-32 path is unchanged, and the output stays at 12 logits.

It is materially distinct from earlier arms because it is the first shared evidence taken from a map finer than the attention grid. Earlier sharing arms were gated or dormant (family channel), competing-gated (conserved evidence), stride-32 pooled (value tie) or training-only (PIT MIL). The ungated additive form answers the recorded failure mode of the family channel, where the gate stayed dormant. The extra cost is about 0.4% of forward FLOPs from an activation that is already computed. There is no new preparation and no second pass.

I ranked it after the reserved head and the already-queued bilateral arm, for two reasons:
- **Small locality gain.** In the researcher's own receptive-field data, the stride-16 cells cut the opposite-half input share only from about 0.31 to 0.23.
- **Attention is still stride-32.** The shared evidence is tent-pooled back onto stride-32 anchors and weighted by the existing stride-32 attention. The OA medial/lateral attention overlap of 0.64–0.70 still limits OA separation.

Ownership: `docs/backlog/big-bets.md#intermediate-stride-16-feature-fusion` is a generic backlog plan that changes the token grid for all labels, not a campaign. This arm is a compartment-only scalar evidence path confined to the campaign's editable directories, so it does not take over or pre-empt that bet. Any result informs the bet but does not stand in for it.

anatomy-supervised-compartment-evidence-split stays proposed. This event gives no new basis to queue it.

Unresolved concerns:
- Only a small locality gain is on offer (opposite-half share 0.31→0.23), and pooling onto stride-32 attention keeps the OA attention-overlap limit. Expect the effect to be concentrated in meniscus, if there is one.
- The new mid-level path sends gradient into the stage ≤10 backbone and can shift the other eight labels either way. Interpret with the evaluator's off-target means, not only the compartment pairs.
- Open choices (width 64/128, tent vs smooth-max pooling, projection per family or shared) must be fixed at implementation, with no sweep. Stride-16 validity and anchor alignment (ReXNet origin, features.10 index, 257 channels) must be verified from feature_info at init. They are not yet confirmed.
- Runtime overhead is unmeasured. Recent arms ran 1,585–1,595 s against the 1,797 s ceiling.
- The diagnostic samples are small (4 studies for the receptive field, 8 per group for the token analyses) and have no uncertainty estimates.

Evidence:
- outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T042159Z-33bd411d/report.json
- outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T042159Z-33bd411d/scratch/evidence-summary.md (receptive-field, token-score and plane-attention tables)
- outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T042159Z-33bd411d/scratch/{erf.py,token_scores.py,plane_attention.py,token_scores_halves.json}
- outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T032728Z-3519d993/scratch/crosstalk.py (lateral OA AUC 0.722 against medial-only negatives vs 0.909 against sibling-negative negatives)
- docs/backlog/big-bets.md#intermediate-stride-16-feature-fusion (ownership overlap assessed; not pre-empted)

## completion-20260921T045639Z-9edf74f5

The strict appearance-only tied meniscus classifier moved in the direction opposite to its target. The meniscus family mean fell from 0.9122 to 0.9047, with a paired interval of [−0.0145, +0.0001], just touching zero.
- Medial Meniscus: −0.0092, interval [−0.0160, −0.0016], strictly below zero. Its support is 356 positives.
- Lateral Meniscus: −0.0057, interval [−0.0185, +0.0078].
- Meniscus compartment-margin AUC: 0.9712 → 0.9688, delta interval [−0.0120, +0.0072]. Lateral cross-talk barely changed (0.636 → 0.630), and lateral false positives went from 44 to 46.

So the tie did not improve compartment separation. It lost within-compartment discrimination instead. The most likely explanation is that removing position from the pooled value, while also sharing one 256-d readout, takes away useful per-compartment evidence. That is consistent with historical 1A, whose looser tie with positional terms gave a small meniscus gain.

The macro stayed flat only because of off-target movement:
- OA family mean: +0.0024, interval [−0.0049, +0.0099].
- Other eight: +0.0009.
- Effusion: +0.0093, interval [−0.0002, +0.0193]. Fracture: +0.0089. Baker's: +0.0079.
- MCL: −0.0078. ACL: −0.0049.

These off-target shifts are consistent with seed noise; the mean absolute probability change is 0.066. They do not show transfer.

Conclusion: evidence against strict geometry-free tying of the meniscus readout on this recipe. It is not evidence against meniscus/OA sharing in general. The single seed cannot separate the two explanations in the pre-launch concern ('strict tie harmful' vs 'no benefit'). The significant Medial Meniscus loss, however, points toward 'strict tie harmful'. Future sharing mechanisms should keep each compartment's private evidence and add shared evidence on top, rather than replacing private readout rows.

This is a completion event. The fixed verdict is inconclusive: the macro delta is −0.0002, with a 95% interval of [−0.0038, +0.0033] over 2,000 valid draws. The contract checks pass. Triplet and visit accounting matches the sealed cohorts exactly: 938,460 triplets, the same cohort hash in every epoch, and a fixed validation selection. The run used 4,380 steps. Measured compute was 1,590 s, under the 1,797 s ceiling. Preparation took 0.17 s. The seed was fresh (939371919). Only the preprocessing_identity input identity differs, and only the declared config keys changed.

The escalation came from review concerns carried over from the pre-launch review. None of them invalidates the run:
- value_components configurability does not affect the result.
- The run is not seed-matched, which is expected with fresh seeds.
- The runtime concern is now resolved by measurement.

The queue is unchanged. The result points away from this design, but it does not change the rank of the remaining ideas. The anatomy-supervised proposal is not part of this event and was not assessed here, so it stays proposed.

Unresolved concerns:
- The single-seed Medial Meniscus loss has an unadjusted per-label interval. It is exploratory and should not be used to reject the family-sharing hypothesis broadly.
- changed_source_archive_members includes src/knee_abnormality/autoresearch/* harness files, because the recovered historical source differs from the run snapshot. These files are not training code. Still, the recorded source identity is broader than the candidate diff, so attribution relies on the model/config members only.
- value_components remains a speculative config option in the frozen source. It must not be carried into any future candidate derived from this code.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/appearance-value-tied-meniscus-classifier-20260921T042159Z-88812751/comparison/comparison.json (family_means, compartments, paired_report_group_bootstrap.intervals.meniscus/meniscus_margin, prediction_change)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/appearance-value-tied-meniscus-classifier-20260921T042159Z-88812751/comparison/per_label.json (Medial Meniscus −0.0092 [−0.0160, −0.0016])
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T042851Z-5e39b3cf/report.json (pre-launch review concerns)
- outputs/training/meniscus-classifier (historical 1A loose tie, prior meniscus +~0.003)

## review-20260921T051117Z-95640718

The candidate adds side-invariant evidence to both compartment logits in each family. Two scalars per family, the max and min of a shared detector over the two unlabeled image halves, feed the four member logits. Each member has its own zero-initialised weights.

The added capacity is small: 776 parameters and no auxiliary loss. That limits capacity or task weight as explanations for any change. Still, only a matched independent-per-compartment detector control could separate shared morphology from the extra pooled readout.

The interpretation should focus on targets 2–5 and the two family means. Most informative are the both-positive co-occurrence diagnostics, since that pattern is what the min statistic can express. The meniscus and OA families are pooled separately with no cross-family coupling. An unrelated-label macro change would not establish the mechanism.

The half split is anatomical only approximately: the midline is the centre of the prepared frame (coronal/axial) or the centre of the series slab (sagittal, series.py:198-201), not a detected knee midline.

I'm approving this for launch. The diff implements the assigned mechanism as specified. For each family there is one shared detector, split into a score row on the existing tanh attention hidden layer and a value row. It pools separately over the two image halves of each (study, series type). Halves come from the prepared-frame x for coronal/axial and from the reflection-adjusted slice offset for sagittal, with a ±6 mm midline band excluded. The hi/lo order statistics are averaged over series that cover both halves. Each family member gets its own alpha/beta, so the medial and lateral readouts stay distinct.

Because the channel is side-invariant, it cannot tell medial-only from lateral-only. That separation stays with the private pathway, and all four label patterns remain representable. The sign conventions for reflection and slice direction don't matter here, because max and min are symmetric.

Validation stays deterministic and doesn't depend on the training seed. The geometry is computed per batch from existing tensors, with no precomputation and no training-data statistics. Nothing touches labels or validation. There is no leakage path. The extra forward cost is a (B·T·2)×F×N masked softmax plus an einsum, which is small. Preparation took 0.18 s. Runtime must still pass the 1,797 s gate once measured.

Unresolved concerns:
- The midline is set by the frame, not detected. The coronal/axial split is at the prepared image centre ((image_size-1)/2), and the sagittal split is at the midpoint of the slab's physical extent. A knee that is off-centre, or a slab that covers the knee asymmetrically, will mix compartments into one half. That weakens the order statistics but introduces no leakage.
- A study with two series of the same type merges them into one (study, type, half) group. Series types missing either half after band exclusion are dropped. A study with no bilaterally covered series gets zero evidence (counts clamp to 1). How often this happens is not logged.
- Gradients from the compartment score row flow into the shared attention layers attention[0..1], so the new channel can also shift the existing private target attention. Any effect on the other eight labels could come from this route.
- Accounting is unavailable, so runtime and triplet accounting haven't been measured yet. They must pass the ≤1,797 s and zero-tolerance gates. The developer reports a CPU check showing exact parity with the option unset, a finite backward pass after the finfo.min masking fix, and passing load_config. I re-ran an AST parse only, and it passed.
- Attribution is not established. Without a matched independent-per-compartment detector control, a gain cannot be credited specifically to medial/lateral sharing rather than to the extra pooled readout.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/bilateral-order-statistic-evidence-20260921T045710Z-5d831334/changes.diff (full diff reviewed; model.py:_compartment_evidence/_compartment_geometry, config.py CompartmentTransferConfig, spatial_tokens.py feature_geometry prepared_x)
- candidate source src/knee_abnormality/modeling/model.py:350-413 (encode_batch: positions reflection-adjusted at :381, inverse_affine reflection at :384-386, new series_type/prepared_x/slice_offset_mm outputs)
- candidate source src/knee_abnormality/training/lightning.py:37-47 (non-backbone params → head_lr; 1-D alpha/beta get no weight decay)
- src/knee_abnormality/data/preprocessing/series.py:198-201 (sagittal slice positions relative to slab midpoint, defines sagittal midline)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T045711Z-3118ec28/report.json (CPU parity, gradient finiteness, target order 2/3 meniscus, 4/5 OA from protocol.json)

## completion-20260921T052351Z-6400a70c

cross-family-mediolateral-axis-agreement was valid and rejected: macro −0.0128 [−0.0178, −0.0078], seed 1247716248, 1,597.6 s.

Both families lost AUC. Meniscus fell more than OA even though meniscus received no direct penalty gradient: meniscus mean −0.036, OA mean −0.010, other eight −0.008. Both compartment margin AUCs fell: meniscus 0.971→0.945, OA 0.866→0.827. Cross-talk rose in both families.

Cause (inferred from the logs and code, not tested by ablation): the attention hidden layers are shared across all twelve targets (model.py:197-202). An auxiliary geometric hinge on two attention rows therefore reshapes attention and the backbone for every target. Its sign anchor was uninformative early in training. At a 10–12% loss share in epochs 0–2 it slowed learning, and training never recovered in ten epochs, even though the hinge was satisfied (share about 0.003) by epoch 7.

Design lesson for this campaign: auxiliary objectives on shared attention geometry should be defined in the private rows, or scaled so they do not act during early formation. Neither was tested here, and a coefficient or margin sweep would be out of scope.

This event only interprets a completed run. The queue does not change. anatomy-supervised-compartment-evidence-split stays proposed. It still needs the user to confirm the deferred image-based side inference, and this event does not change that.

The run is valid. Accounting matches the seal exactly: 938,460 triplets, 35,030 visits, the sealed cohort hashes, a constant validation selection and 4,380 steps. Measured compute was 1,597.6 s, under the 1,797 s ceiling. Config differences are confined to the trainer.compartment_transfer prefix.

The macro delta is −0.0128 [−0.0178, −0.0078]. Under the fixed rule this implementation is rejected as an improvement.

The damage is broad and largest where the transfer should have helped least:
- The meniscus family mean fell 0.9122→0.8766. Lateral Meniscus fell −0.055 [−0.080, −0.031] and Medial Meniscus −0.016 [−0.027, −0.007]. The meniscus margin AUC fell 0.971→0.945, and lateral cross-talk rose 0.636→0.698.
- The OA pair, the intended target, fell 0.8882→0.8783. The OA margin AUC fell 0.866→0.827, and medial-positive lateral-OA false positives rose from 51 to 60. The anatomical prior did not improve OA assignment. It made it worse.
- The other eight fell 0.8948→0.8870, with MCL at −0.033 [−0.058, −0.012].

The mechanism is not purely OA-local, and the logs support that. The penalty gradient reaches the attention hidden layers, which all twelve targets share (model.py:197-202), and the backbone. During epochs 0–2 the hinge averaged about 3–4.5 mm unmet, with a loss share of 9–12%. In the first epochs the detached meniscus sign (mediolateral_agreement.py:26) is uninformative. So the shared attention and backbone were being pushed toward an essentially arbitrary geometry exactly while they formed.

The penalty then collapsed: its share was 0.003 by epoch 7. Even so, the BCE component stayed about 0.05 above the reference at epoch 9 (about 0.494 vs 0.441). Validation AUC trailed at every epoch: 0.746 vs 0.806 at epoch 0, and 0.884 vs 0.897 at epoch 9. The early distortion was not recovered within the fixed schedule.

This is one implementation, one seed and one fold. It rejects a hard, millimetre-margin, cross-family attention-displacement hinge. It does not refute shared mediolateral structure in general.

Unresolved concerns:
- Attribution between the uninformative early sign and the hinge magnitude is not established. The only evidence is the logged loss share (0.099–0.116 in epochs 0–2) and the persistent BCE gap. No zero-margin or late-onset control exists, and none is justified by a rejected result.
- The candidate's logged train_loss includes the penalty. The BCE comparison with the reference (about 0.494 vs 0.441 at epoch 9) is derived as total×(1−share), not taken from a separately logged BCE.
- Single seed and single fold. Bootstrap uncertainty excludes seed variation, so the size of the per-label losses (for example Lateral Meniscus −0.055) is conditional on this fit.

Evidence:
- outputs/autoresearch/medial-lateral-transfer/experiments/cross-family-mediolateral-axis-agreement-20260921T043006Z-27a9bda1/comparison/comparison.json (family_means, compartments margin/cross-talk)
- outputs/autoresearch/medial-lateral-transfer/experiments/cross-family-mediolateral-axis-agreement-20260921T043006Z-27a9bda1/comparison/per_label.json
- outputs/autoresearch/medial-lateral-transfer/experiments/cross-family-mediolateral-axis-agreement-20260921T043006Z-27a9bda1/training/fold_0/logs/version_0/metrics.csv (penalty 4.1–4.5 mm and share 0.086–0.116 in epochs 0–2; share ≤0.0035 from epoch 7)
- outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat metrics.csv (reference val AUC 0.806→0.897; train loss 0.644→0.441)
- experiments/cross-family-mediolateral-axis-agreement-20260921T043006Z-27a9bda1/source/src/knee_abnormality/training/compartment_transfer/mediolateral_agreement.py:26-27 (detached sign, relu hinge)
- recovery/historical-source-20260920T215302Z-e086c1eb/source/src/knee_abnormality/modeling/model.py:197-202 (attention hidden layers shared by all 12 target rows)
- outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T031151Z-4acefbe8/report.json (hold reason for anatomy-supervised-compartment-evidence-split)

## proposals-20260921T052447Z-8d99228d

The researcher's new exploratory evidence has no intervals. The scripts are in researcher-20260921T045705Z-0c75907a/scratch.

- **Reference strength.** Control B's Medial Meniscus AUC (0.9266) is the highest of 27 recipe-family runs; the others average 0.0072 lower. Earlier medial drops blamed on tying (−0.009 to −0.012) fall inside that spread, so they do not show that sharing harms the majority compartment. Any campaign arm will likely look worse on Medial Meniscus through regression toward the mean alone. Read medial-meniscus losses with that in mind.
- **Cross-talk ceiling.** Ranking sibling-diseased negatives as well as sibling-healthy ones adds at most +0.0069 macro. Most of it is lateral: OA +0.040 and meniscus +0.027 label AUC.
- **Orientation probe** (48 studies). The pair's attention displacement has the correct sign in 96–98% of coronal and sagittal studies and 88–90% of axial studies. Each OA row alone puts only 0.53–0.66 of its mass on the correct half. So the pair jointly holds orientation that the individual OA readouts don't use. This supports assignment as the bottleneck for OA. For meniscus, attention already localises better (0.71–0.84 correct-half).
- **Limits.** The probe pooled per plane in patient coordinates. Per-series reliability in the image frame is not measured, and there is no seed replicate of control B.

If the hemifield gate gains, attribution needs its pre-registered superior–inferior control. If the adversary gains, it needs its permuted-sibling-label control. Evidence that the mechanism worked should come from the lateral sibling-diseased-negative strata and the both-positive co-occurrence support, not from the macro alone.

I admitted both proposals. Neither has dependencies (proposal_admission shows both satisfied).

pair-oriented-hemifield-attention is ranked second, right behind the reserved head. It goes after the most stable weakness the researcher measured: cross-talk on sibling-diseased negatives. Lateral OA scores 0.722 there against 0.909 on sibling-healthy negatives. That OA stratum sits mid-pack across 26 runs, so the weakness is structural, not luck in the reference run. The fix works in the forward path. It adds four zero-initialised scalars, and at κ=0 it matches control B exactly. It needs no new batch keys or metadata.

On scope, I judge that the user's deferral of image-based side inference (docs/research/knee-side-and-location.md:205-206) does not block it:
- It has no side classifier, no side supervision and no knee-side labels.
- It sharpens an orientation the model's own compartment attention already encodes. That orientation is a medial-minus-lateral displacement in the image frame, which commutes with reflection.
- That puts it in the same class as the admitted cross-family axis agreement, not the held anatomy gate. The anatomy gate is supervised per token on geometry-derived medial/lateral labels.
- It is also clearly different from the reserved head. The bilateral order statistic ignores orientation; this gate routes by hemifield.

sibling-conditional-attention-adversary is ranked last. It is a distinct information-level route to the same deficit, it infers no orientation and it raises no scope question. Its priors are weaker, though:
- Every training-only auxiliary arm in this campaign so far has been neutral or harmful.
- The spatial exclusivity penalty reduced cross-talk but did not improve margins.
- Gradient-reversal dynamics are noisy at batch size 8.
- Co-occurrence is genuinely predictive in this population.

family-shared-local-lesion-evidence keeps its place ahead of the adversary.

anatomy-supervised-compartment-evidence-split stays proposed. It still needs the user to confirm the side-inference deferral. Nothing in this event resolves that.

Unresolved concerns:
- pair-oriented-hemifield-attention: the gate can feed back on itself. The orientation o comes from the same attention it confines, and gradients flow through o. In the 2–12% of studies with the wrong sign, the gate may lock the rows onto the wrong half. Log the per-plane |o| distribution and κ, and consider tracking agreement of o's sign across planes.
- pair-oriented-hemifield-attention: confining each compartment to its hemifield may suppress both-positive and co-occurrence evidence. The both-positive support must be read alongside the cross-talk strata. The stride-32 receptive fields also leak about 30% across the midline, which caps the achievable gain.
- pair-oriented-hemifield-attention: review must check that temperature baking (which scales attention[-1] only) handles the gate consistently, that the gate stays in fp32 under bf16, and that the prediction path adds no batch keys.
- sibling-conditional-attention-adversary: conditioning on y_c does not remove severity correlation. Within y_c=1, a compartment's own disease grade may legitimately predict the sibling label, so the reversed gradient may strip genuine evidence. It must be confirmed that the gradient stays confined to rows 2–5. A backbone-wide auxiliary previously damaged the other eight labels.
- anatomy-supervised-compartment-evidence-split still awaits user confirmation of the image-based side-inference deferral. Nothing in this event changes that.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T045705Z-0c75907a/report.json (proposals, exhausted_directions: run spread, cross-talk ceiling, orientation probe)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T045705Z-0c75907a/scratch/plane_side_attention.{py,json,log}, crosstalk_ceiling.py, run_spread.py
- /home/claudio/src/knee-abnormality/docs/research/knee-side-and-location.md:205-206 (image-based side inference deferred by user instruction)
- /home/claudio/src/knee-abnormality/autoresearch/campaigns/medial-lateral-transfer/findings.md:1036-1046 (the earlier hold on anatomy-supervised-compartment-evidence-split)

## review-20260921T054035Z-199f279b

The gate adds 4 learned scalars and no auxiliary loss. Capacity and task weight are therefore very unlikely to explain any effect, so a matched-capacity control is not needed. What changes is where the shared scorer's attention goes for rows 2–5.

The orientation comes from the model itself, not from anatomy. When medial and lateral already attend to the same place, which is the shared-lesion case the idea targets, the orientation tanh(Δ/20mm) is near 0 and the gate does almost nothing. With kappa>0 the gate amplifies any separation that already exists. With kappa<0 it pushes the two compartments together. Whichever sign the fitted kappa takes is itself diagnostic.

For interpretation, record the final kappa values per pair and side from the checkpoint. Read the meniscus and OA family means, the compartment margins and the both-positive co-occurrence support alongside the macro. The other eight targets can only change indirectly, through the shared scorer and backbone, because their score rows are untouched.

On scale: with 0.625 mm spacing, the in-plane extent is about ±70 mm (x/20 up to about 3.5). Adam at 6e-4 over 4,380 steps bounds the movement of each kappa to a few units. A small learned kappa would mean the run tested an almost inert gate, not the hypothesis itself.

I approved as written, with no queue or proposal changes.

The candidate matches its assigned topic. Each series gets its own medial/lateral orientation, taken from the pair's own attention. Each compartment's attention is then biased toward its hemifield. The meniscus and OA pairs are both covered.

- **Source checks:** Shapes are consistent. xy is a (tokens, 2) grid that is the same for every image. acquisition_index maps each image to its kept-series row, and row_count is the scatter size. The per-series centroid sums base_attention over the batch dimension, which is correct because study membership partitions the tokens.
- **Where the gradient goes:** The orientation is not detached. The gate's gradient therefore also flows into the shared attention scorer through the base-softmax centroids. This is the declared design.
- **Leakage:** None, and there are no new batch keys.
- **Scope:** The preprocessing_identity difference comes only from the model.py edit, which the policy allows.

`proposal_admission` lists anatomy-supervised-compartment-evidence-split with its dependencies satisfied. I did not admit it: this was a review-only event, and the queue already holds two untested compartment ideas.

Unresolved concerns:
- Accounting is not available yet: runtime (≤1,797 s) and triplet accounting still have to pass their gates. I found no evidence of added cost beyond one extra softmax over the targets×tokens attention matrix and the per-series scatter_add operations.
- kappa has shape (2,2), so ndim=2 and it receives the ordinary weight_decay (lightning.py:43). The shrinkage is decoupled and small, but it biases the result toward an inert gate. Record the final kappa before crediting or rejecting the mechanism.
- The orientation depends on attention and is circular: it only amplifies or collapses whatever medial/lateral separation the scorer already has. A null result does not refute side-aware sharing with anatomical orientation.
- The centroid denominator is clamped at 1e-6. If a series gets almost no attention for one target, its centroid shrinks toward 0 and its orientation becomes arbitrary, though with bounded effect. How often this happens is not logged.
- scripts/adhoc/diagnose_encoding_usage.py:206 calls _attention_pool without the new arguments and would fail with the gate enabled. It is outside the training and prediction entrypoints, so it does not block this run.
- hemifield.py is not included in the preprocessing_identity hash. It is covered only by the source manifest and snapshot, which the campaign allows.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/pair-oriented-hemifield-attention-20260921T052543Z-b6426dc3/changes.diff
- candidate source src/knee_abnormality/modeling/compartment_transfer/hemifield.py:52-98 (kappa zero-init, centroid orientation, logsigmoid gate applied to rows 2–5 only)
- candidate source src/knee_abnormality/modeling/model.py:226-245 (gate applied inside the fp32 block, then re-mask and re-softmax) and :297-338 (series-centred sagittal position and in-plane mm coordinate)
- candidate source src/knee_abnormality/modeling/spatial_tokens.py:199-247 (xy is a fixed (tokens,2) augmented-frame grid)
- candidate source src/knee_abnormality/training/lightning.py:37-47 (kappa gets head_lr and weight decay because ndim=2)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T052543Z-c6b75909/report.json (developer's CPU checks: parity at kappa=0 with max logit diff 1.19e-7, nonzero kappa gradient)

## proposals-20260921T054616Z-9ec2f7ed

Researcher evidence, verified from the job's scratch outputs. Exploratory; no intervals.
(1) Epoch-9 Lateral OA AUC against medial-positive negatives is 0.678–0.735 across 29 runs, including the exact metadata side-axis arm (0.718). Added assignment information alone has not moved this cross-talk plateau.
(2) Sibling-positive Lateral OA negatives with own soft target <0.2 still score AUC 0.768, against 0.898 for medial-negative ones. A knee-level co-occurrence factor, not graded labels, dominates.
(3) Soft ownership would keep 0.573 of OA and 0.782 of meniscus value-path credit. Medial and lateral kept shares are equal by construction, since Σ(a_m²−a_l²)/(a_m+a_l)=Σ(a_m−a_l)=0. Routing therefore removes the same fraction of trunk credit from both siblings.
(4) The top-20 attended tokens are almost never sibling-owned (0–10%). Routing mainly withholds credit on diffuse tails and overlap (OA overlap 0.68).
For routing, pre-register the OA pair logit correlation (reference 0.873 vs label 0.55) and the lateral sibling-positive strata, read alongside the macro. A gain needs the ownership-permutation control before it can be attributed to anatomy rather than reduced trunk credit.

Both proposals have no dependencies (proposal_admission satisfied), declare mode add on historical-control, and are materially distinct from completed arms.

attention-owned-compartment-gradient-routing is ranked second, behind the reserved head. It is the campaign's first backward-only mechanism: no parameters, no loss term, identical eval/inference and checkpoint structure. It targets a measured leak. In 48 training studies with the control-B checkpoint, OA rows put 41% (medial) and 27% (lateral) of their attention/value credit on sibling-owned tokens; meniscus rows put 19% and 12%. The share is flat across co-occurrence groups (scratch/credit_assignment.log). The transfer rationale is in scope: the trunk stays shared, so morphology learned on one side remains available to the other, but each label reshapes only its own tissue. It is cheap and orthogonal to the forward-path arms in flight.

within-knee-exclusive-differential-ranking is admitted but ranked last. It is the only proposal whose supervision contrasts compartments within one knee. That removes a knee-level factor only insofar as the factor is spatially shared by the two attention distributions. Balancing medial-diseased and lateral-diseased averages blocks a pure anatomy/side direction, since such a direction scores with the same sign in both. Still, auxiliary transfer objectives have a weak record here (shared-disease-auxiliary inconclusive with higher meniscus cross-talk; invariant embedding rejected). Its slot partly overlaps the invalid PIT half-knee arm.

Queue now has five entries. The other ideas keep their relative order. The existing proposed anatomy-supervised-compartment-evidence-split stays unqueued and was not reassessed in this event.

Unresolved concerns:
- Gradient routing confounds anatomical routing with a ~43% (OA) / ~22% (meniscus) cut in compartment-label trunk gradient. That is effectively a task-weight change. Only the proposed kept-share-preserving permutation control can attribute a gain to ownership.
- Routing leaves the attention-score gradient into tokens unrouted, so part of the sibling-tissue credit still flows. The training forward (a·m·tok + a·(1−m)·sg(tok)) matches only up to fp32 rounding, not bitwise. Review should check dtypes under bf16 autocast and that eval is exactly unchanged.
- The ownership probe used eval mode on 48 training studies at the final checkpoint only. Ownership early in training (m≈0.5) is unmeasured, so the effective routing strength over the schedule is unknown. Log the per-epoch kept share.
- Differential ranking cancels a knee-level factor only when it is spatially shared by both compartment attention maps. The 0.5 exclusivity threshold on soft labels, and whether balancing fully blocks identity shortcuts after reflection, need review at implementation.
- Missing evidence noted by the researcher: backbone-level gradient conflict, a control-B seed replicate, and early-checkpoint ownership sharpness.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T052544Z-d28fb732/report.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T052544Z-d28fb732/scratch/evidence-summary.md
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T052544Z-d28fb732/scratch/credit_assignment.log (group tables: OA medial 0.412 / lateral 0.272 sibling-owned mass, kept share 0.573; meniscus 0.188/0.119, 0.782)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T052544Z-d28fb732/scratch/asymptote.csv
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T054616Z-d8bab525/history-index.json (shared-disease-auxiliary, cross-compartment-attribution-exclusivity, permutation-invariant-half-knee-mil records)

## completion-20260921T055217Z-dba975ac

Family means:
- Meniscus: −0.0064 [−0.0147, +0.0016]
- OA: −0.0049 [−0.0121, +0.0021]
- Other eight labels: −0.0034 [−0.0084, +0.0019]

Compartment margins:
- The meniscus margin AUC fell from 0.9712 to 0.9580, a change of −0.0134 [−0.0296, −0.0008], with 344 exclusive studies. Separating medial from lateral got worse, not better.
- The OA margin was essentially unchanged: +0.0039 [−0.028, +0.036], with 198 exclusive studies.
- OA medial-positive lateral false positives rose from 51 to 58, and cross-talk rose from 0.622 to 0.682.

Label pattern: exploratory label deltas were negative for all of targets 2–5. Effusion was the only notable positive at +0.010 [−0.001, +0.020]. ACL was −0.012 and MCL −0.014.

The key new fact is that the added channel stayed almost inert. The checkpoint norms are ‖compartment_alpha‖=0.038 and ‖compartment_beta‖=0.058 over four entries each, while the score and value rows have norms of about 1.0. The direct contribution of the order statistics to the logits is therefore small. Even so, predictions moved substantially: mean absolute probability change was 0.068, and per-target Spearman was 0.88–0.97. That points to the difference coming mostly from the fresh seed plus gradient flow from compartment_score into the shared attention layers. The max/min evidence itself had little to do with it.

This run weakly tested the order-statistic hypothesis. It does not show that side-invariant bilateral evidence helps or hurts. It does show that attaching a tiny zero-initialised readout does not get used under this recipe. Any revisit would need a different design, not a sweep of scale or learning rate.

This is a completion event and I interpreted the fixed reject decision. The run passed its contract checks. Accounting is complete: 938,460 training triplets and 35,030 visits, with every epoch's cohort hash equal to the sealed hash. There were 4,380 optimizer steps, and measured compute was 1,590.7 s against a 1,797 s ceiling. The only config differences are the two allowed compartment_transfer keys, and the only identity difference is preprocessing_identity. The escalated review concerns do not invalidate the run. Unlogged midline and coverage fallbacks are a limit on interpretation, not a contract breach.

The macro score fell from 0.89658 to 0.89243, a change of −0.0041 with 95% interval [−0.0081, −0.0001]. The result is therefore a rejection of this implementation as an improvement, not of bilateral sharing in general. I made no queue change. The existing proposal anatomy-supervised-compartment-evidence-split is not part of this event and stays proposed. The queue is already at the target of five.

Unresolved concerns:
- Attribution is weak because the new weights barely moved from zero init (alpha/beta norms 0.04–0.06). The macro drop cannot be credited to the bilateral evidence rather than to seed variation or to changes in the shared attention.
- The comparison does not log both-positive co-occurrence diagnostics for this mechanism. How often the frame-centre midline split failed, or a study had no bilaterally covered series, is still unknown.
- The meniscus margin decline has an interval that just excludes zero. It is an unadjusted diagnostic from one seed, so do not treat it as evidence that sharing harms compartment separation.

Evidence:
- outputs/autoresearch/medial-lateral-transfer/experiments/bilateral-order-statistic-evidence-20260921T045710Z-5d831334/comparison/comparison.json (paired_report_group_bootstrap.intervals; family_means; compartments)
- comparison.json parameters_only_in_run: compartment_alpha norm 0.0378, compartment_beta norm 0.0584, score/value weight norms ~1.007
- comparison.json prediction_change: mean_abs_probability_difference 0.0677, per-target Spearman 0.878–0.966
- comparison/per_label.json (exploratory per-label intervals)
- orchestrator-20260921T051117Z-247105df/report.json (implementation review concerns)
- developer-20260921T045711Z-3118ec28/report.json

## review-20260921T060120Z-c38818b6

The mechanism is a pure gradient-credit reallocation. In the forward pass, the parameters, the attention and head gradients are identical to control B. Only the trunk-token gradient of the four compartment labels changes: each is scaled per token by that label's attention share relative to its sibling.

The effect is therefore anti-sharing at the trunk. On tokens that both siblings attend to equally (ownership about 0.5), each sibling sends only half its gradient. On tokens that one sibling dominates, that sibling keeps nearly all of it.

A consequence is that the four compartment labels also lose trunk task weight overall, by about 1 − kept_share. So any effect mixes two things: routing specificity, and reduced compartment gradient mass on the backbone.

Read the result with the logged gradient_routing_kept_share per target and epoch:
- Kept share near 1 means the attention was already separated, and the run tested an almost inert intervention.
- Kept share near 0.5 means the effect could come from a uniform down-weighting rather than ownership routing.

Attributing a win to routing would need a matched control that applies a uniform 0.5 token-gradient scale.

Interpret together with the pooled meniscus and OA family means, the compartment margins, and the sibling-diseased negatives (for example, Lateral OA scores 0.722 there against 0.909 on sibling-healthy negatives). The other eight targets can move only indirectly, through the shared trunk dynamics.

I approve launch. The candidate implements the assigned mechanism exactly: it is backward-only and routes compartment trunk credit by attention ownership. It stays within the policy files and prefixes and leaves the twelve-logit interface and every eval path unchanged. The only config difference is model.compartment_transfer.gradient_routing.pairs. Preparation took 0.17 s.

I found no leakage. Ownership comes from the model's own training-batch attention, and validation runs in eval mode with routing off.

I have no repair to request. The concerns below are about interpretation and runtime, not correctness.

anatomy-supervised-compartment-evidence-split stays proposed. Its hold is a scope question for the user (confirming that image-based side inference stays deferred). This review event does not resolve it.

Unresolved concerns:
- Accounting is unavailable. Runtime (ceiling 1,797 s) and triplet gates are still pending. The added cost is four extra (B×N×F) einsums per pair, plus four float() host syncs per step over 4,380 steps. That is likely seconds, but it is unmeasured.
- Routing specificity is confounded with reduced trunk gradient mass for the four compartment labels. No uniform-scaling control exists, so a positive result cannot be attributed to ownership routing without one.
- Ownership is circular: it comes from the same shared scorer's current attention. Early in training the attention rows are similar, so ownership is about 0.5 everywhere. A null result does not refute compartment-specific credit assignment in general.
- The kept-share diagnostic is a batch mean of Σa·o, logged per epoch only. It does not show how ownership is distributed within a study. That is enough for the inert-vs-uniform check but not for finer attribution.
- The mechanism restricts sharing rather than adding transfer. Its link to the campaign scope is the 'correct prediction per compartment' side. A macro gain driven by unrelated labels would not establish transfer.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/attention-owned-compartment-gradient-routing-20260921T055258Z-2e5a20a0/changes.diff
- candidate source/src/knee_abnormality/modeling/model.py:207-231 (_route_compartment_pair: stop-gradient split, no_grad ownership, host-sync diagnostics)
- candidate source/src/knee_abnormality/modeling/model.py:246-262 (routing gated on self.training inside the fp32 block; clone plus row overwrite)
- candidate source/src/knee_abnormality/training/lightning.py:90-105 and 249-255 (pair resolution against the frozen target order; model wiring)
- candidate source/src/knee_abnormality/inference/checkpoint.py:101 (the default pairs=() leaves prediction identical; no new parameters)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T055259Z-372de296/report.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T052447Z-b5d1108a/report.json (the anatomy-supervised proposal is held for a scope question to the user)

## completion-20260921T061930Z-569b9354

The pair-oriented hemifield gate is inconclusive: macro −0.00102 [−0.00477, +0.00247]. The run was valid and within budget (1,591 s of 1,797 s; triplets exact).

The gate barely moved from zero. At epoch 8, kappa is meniscus (medial 0.123, lateral 0.282) and OA (0.016, 0.037); the parameter norm is 0.31. The OA gate was essentially inert, with at most about 0.1 logit at the image edge. The meniscus gate was weak and positive, meaning it amplified existing separation.

Diagnostics (exploratory, wide intervals):
- Meniscus family: +0.0015 [−0.0065, +0.0098]. Lateral Meniscus +0.0034, Medial Meniscus −0.0002.
- OA family: −0.0026 [−0.0098, +0.0048].
- Other eight labels: −0.0013 [−0.0062, +0.0033]. Synovitis −0.0073 and Baker's +0.0062 can only move through shared training, which is consistent with seed variation.
- Compartment margins went down but their intervals include zero: meniscus 0.971→0.964 (Δ −0.0071 [−0.024, +0.007]); OA 0.866→0.856 (Δ −0.0103 [−0.047, +0.025]).
- Lateral cross-talk rose for both pairs: meniscus 0.636→0.659, OA 0.622→0.639. Medial-positive false positives for Lateral Meniscus went from 28 to 29, and for Lateral OA from 51 to 53.

Interpretation: a weak meniscus-only amplifying gate plus a new seed. There is no sign of transfer or of better compartment assignment, and the gate did not reduce cross-talk on sibling-diseased negatives. The weak point is that the orientation comes from the model's own attention: with only the disease loss driving them, the scalars stayed near zero. A future revisit would need a materially stronger or externally anchored orientation, not a larger kappa or a different weight-decay setting.

This event only interprets a completed run, so the queue does not change. anatomy-supervised-compartment-evidence-split stays proposed. Its gate works out medial/lateral from the image, which still conflicts with the user's recorded decision to defer image-based side inference, and nothing here resolves that.

The run is valid:
- Triplet accounting matches exactly: 938,460 triplets, 35,030 visits, sealed cohort hashes, 4,380 steps, and a constant validation selection.
- Measured compute was 1,591.3 s against the 1,797 s ceiling. This settles the runtime concern from review.
- The config differs only in model.compartment_transfer.pairs and name (plus the allowed seed and output root).
- The only input-identity difference is preprocessing_identity, because model.py changed. The other extra archive members are harness code.

Result: the macro changed by −0.00102, 95% CI [−0.00477, +0.00247], so the fixed rule gives inconclusive.

The main new fact is that the gate was nearly inert. The fitted kappa at the selected checkpoint (epoch 8) is [[0.123, 0.282], [0.016, 0.037]] (rows are meniscus and OA; columns are medial and lateral). The gate's shift to attention scores is kappa × tanh(orientation) × x/20 mm, with |x/20| up to about 3.5 at the edge of the image:
- For OA, the shift is at most about 0.06–0.13 logits at the extreme edge and much smaller near the joint. OA was in effect a new-seed rerun of control B.
- For meniscus, the lateral row reaches at most about 1 logit at the edge, only where the two attention centroids are already well separated.
- Both signs are positive, so where the gate acted it amplified the separation the model already had rather than pulling the two compartments together.

The prediction changes look like seed-level variation:
- The mean absolute probability change was 0.066.
- The Spearman correlation between runs was about 0.89–0.91 on untouched labels (Baker's, Fracture, Contusion). That is lower than on the gated labels (0.92–0.97).

This result does not refute side-aware sharing. With the gate this small, the run did not test that hypothesis.

Unresolved concerns:
- kappa got the ordinary weight decay, being 2-D (lightning.py:43). Together with the small learning signal this probably kept the gate near zero. How much of that came from decay versus a genuinely flat loss surface cannot be separated after the fact.
- How often the centroid-denominator clamp fired, and how orientation was distributed within studies, were not logged. There is no direct measure of the gate's effective attention shift per study; this finding relies only on the size of kappa.
- One seed, one fold. The shifts in cross-talk and margins are within seed variation and can't be attributed to the gate.

Evidence:
- outputs/autoresearch/medial-lateral-transfer/experiments/pair-oriented-hemifield-attention-20260921T052543Z-b6426dc3/comparison/comparison.json (family_means, compartments, paired_report_group_bootstrap.intervals, parameters_only_in_run: kappa norm 0.310, prediction_change)
- training/fold_0/best-epoch=08-val_macro_auc=0.89558.ckpt: model.compartment_gate.kappa = [[0.1231, 0.2816], [0.0159, 0.0367]]
- source/src/knee_abnormality/modeling/compartment_transfer/hemifield.py:59,83-93 (zero-initialised kappa; logsigmoid gate kappa × tanh-orientation × x / 20 mm)
- evidence.json accounting: 938,460 triplets / 35,030 visits, 4,380 steps, measured_compute 1,591.33 s
- jobs/orchestrator-20260921T054035Z-3e9213f9/report.json (review that flagged weight decay and the inert-gate risk)
- jobs/orchestrator-20260921T052447Z-b5d1108a/report.json (anatomy-supervised proposal held pending user confirmation of the side-inference deferral)

## review-20260921T063503Z-b2a2e6a2

The shared-evidence term adds Σ_n a_c(n)·e_f(n) directly to each compartment logit, without a gate or scale, and it is trained only by main BCE.

It is a single scalar map per family. Transfer here means the medial and lateral logits share one local detector while each weights it by its own attention. The local term cannot tell compartments apart except through attention placement. If attention rows are similar between siblings, the term pushes both logits the same way.

What to log and compare in the results:
- local_evidence_*_vector_norm and local_evidence_mean_abs. A tiny mean_abs means the path went nearly inert.
- Family means against the other eight targets. The other eight can move only through shared-trunk dynamics.
- Compartment margins and sibling-diseased negatives. These show whether the shared evidence blurs compartment identity.

The capacity confound is real: about 16.8k extra parameters plus a stride-16 input the baseline never reads directly. A win would not by itself attribute the gain to sharing. A matched control needs per-compartment private vectors, four instead of two, on the same stride-16 path.

I approve launch. The candidate implements the assigned mechanism as a pure addition with no other effects.

What I checked in the source:
- The only config difference is model.compartment_transfer.local_evidence.width=64, and it is inside the policy prefix.
- The final backbone output is unchanged, confirmed by a static CPU comparison.
- The family indices match the frozen target order.
- The added logit term uses the existing validity mask and study membership.
- The training step adds only diagnostics.
- Validation stays deterministic: the local path involves no augmentation or randomness, and the only new layer is LayerNorm, with no dropout.
- There is no leakage path: all parameters are trained by the main BCE on training batches only.
- Preparation cost is 0.17 s. The added compute is about 6 MFLOP per image, negligible next to ReXNet-200. The runtime and triplet gates still need to pass at launch.

Queue order and membership are unchanged. I left the proposal anatomy-supervised-compartment-evidence-split unadmitted in this review event.

Unresolved concerns:
- Accounting is unavailable. The runtime ceiling (1,797 s) and triplet equality are still unmeasured. The expected overhead is small, but the forward_intermediates path has not been timed on GPU.
- Attribution: a macro gain would mix family sharing with added stride-16 capacity. A private-vector (4-vector) stride-16 control is needed before crediting shared morphology.
- Family indices (2,3)/(4,5) are hardcoded, guarded only by the target count. They are correct for the frozen order, but a reordered target list with the same count would fail silently. The file also adds a code comment, against repo style (cosmetic).
- The added term is ungated and enters both sibling logits with the same sign wherever their attention overlaps. This could erode compartment specificity. Check the compartment margins and sibling-diseased negatives, not only the family means.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/family-shared-local-lesion-evidence-20260921T062026Z-5c2f5c21/changes.diff
- candidate source src/knee_abnormality/modeling/model.py:146-158 (construction), :167-179 (forward_intermediates), :266-273 (family logit addition), :342-346 (encode_batch key)
- candidate source src/knee_abnormality/modeling/compartment_transfer/local_evidence.py:1-48
- candidate source src/knee_abnormality/training/lightning.py:38-45 (param groups: head_lr for new params)
- CPU check (timm 1.0.26): rexnet_200 feature_info index 3 = features.10, reduction 16, 257 ch; forward_features vs forward_intermediates max abs diff 0.0; stride-16 map 14x14
- Reference prediction header outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat/fold_0/validation_predictions.csv: target order confirms indices 2-5
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T062026Z-deb68869/report.json (smoke test: non-family logits bit-identical, 16,768 params)

## proposals-20260921T063923Z-a60fe49e

New researcher evidence: in 30 runs, isolated lateral positives are the weak stratum. Isolated Lateral OA scores 0.813±0.010 against clean negatives, versus 0.934 when the sibling is also positive. Control B sits unusually high on it at 0.840 (+2.7 sd), and on isolated Medial Meniscus (+2 sd). Any arm will therefore tend to look worse on these strata through regression toward the mean. They are exploratory reads on the judging fold, not evidence for acceptance.

The trunk-conflict probe (control-B checkpoint, 48 training studies, eval mode) found head-row cosines of 0.315 (OA) and 0.222 (meniscus), and token-weighted Jacobian cosines of 0.28 and 0.07. The sibling-sharing premise holds for OA but is weak for meniscus at the trunk. The projection should therefore show more activity on OA than on meniscus, and a meniscus-only null would not refute it.

For the projection run, the pre-registered inertness check is the removed push-down energy share (below 2% means inert). The mechanism-relevant reads are:
- Lateral OA sibling-diseased negatives (0.722 in control B's evaluator diagnostics)
- Compartment margins
- OA pair logit correlation (0.873)

Protecting push-up features could increase co-occurrence reliance. Attribution after any win needs the norm-matched random-direction removal control and the symmetric-projection control.

I admitted discordant-sibling-gradient-projection. It has no dependencies (proposal_admission: satisfied) and fits the scope: it protects trunk lesion features shared by siblings in discordant studies. Compartment identity stays with the private attention and head rows, all four joint states stay representable, and the forward and predict paths are unchanged. No queued or running idea covers it. Routing scales both siblings' credit by attention ownership in every study. The adversary acts only on attention rows. The ranking idea adds a loss.

I ranked it third, behind the adversary, not second as the researcher recommended. The motivating evidence is weaker than presented:
- Residual-discordant studies have opposite-sign residuals. With positive sibling Jacobian cosines (OA 0.19, meniscus 0.02 at study level), near-universal token conflict follows almost by construction, so the 97%/80% conflict shares do not show an independent pathology.
- The removed energy is small (OA 12%, meniscus 5%). It was measured only at the final checkpoint, on 48 studies, and Adam may damp it further.
- A sibling trunk-gradient operator, attention-owned routing, is already running. Its result and kept-share diagnostics should be read before a second backward-only trunk intervention is dispatched.

It stays ahead of the within-knee ranking idea because it adds no loss term. Auxiliary-loss arms have been the recurring failure pattern.

Unresolved concerns:
- The conflict statistic mostly follows from positive Jacobian cosines under opposite-sign residuals, so it does not independently show harmful negative transfer. The effect size is only 5–12% of push-down energy at convergence, and early training is unmeasured.
- The push-down sibling's shared-direction gradient partly teaches trunk specificity. Removing it may worsen cross-talk and sibling-diseased negatives (Lateral OA 0.722) even if family means rise. Check compartment margins, not only macro AUC.
- Stacking with attention-owned routing (running) modifies the same trunk-token gradient. If routing is accepted during development, the composition order (project the routed gradients) needs renewed integration review.
- Implementation risk: five forked views re-run mil_norm and attention in training, which costs memory. The sign-tap holder state must be reset per step and follow the logit-to-fork backward order. fp32 projection under bf16 autocast needs verification, and gradient clipping must stay unchanged.
- Control B's elevated isolated-stratum scores bias stratum comparisons against every candidate. No control-B seed replicate exists for these strata.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T062027Z-eb55fd3c/report.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T062027Z-eb55fd3c/scratch/evidence-summary.md (isolated strata table; trunk conflict table)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T062027Z-eb55fd3c/scratch/sibling_conflict.json and sibling_conflict.py (48 studies, seed 41, eval mode, final checkpoint only)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T062027Z-eb55fd3c/scratch/isolated_positive_strata.log (30-run strata)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/attention-owned-compartment-gradient-routing-20260921T055258Z-2e5a20a0 (running sibling trunk-gradient operator)

## completion-20260921T064727Z-41209fe9

Attention-ownership routing of compartment trunk gradient, with the forward pass identical to control B, lowered macro AUC by 0.0039 (CI strictly below zero).

Ownership stayed near 0.5 early in training. By epoch 9 the kept share reached 0.80 for meniscus but only 0.60 for OA. In practice the run mostly halved the four compartment labels' trunk credit rather than routing it by anatomy.

Compartment specificity did not improve:
- Meniscus margin AUC fell from 0.971 to 0.961; OA was flat.
- Lateral cross-talk was unchanged or slightly worse.

Every family moved down: meniscus −0.0069, OA −0.0036, other eight −0.0033. ACL fell −0.012 with an exploratory interval excluding zero, and Baker's rose +0.007 (interval includes zero).

This suggests the shared trunk benefits from full compartment-label gradient, including on tokens both siblings attend to. Withholding sibling-shared credit costs performance, consistent with the scope's premise that shared patterns help. It does not refute compartment-specific credit assignment with a separated or anatomically grounded ownership signal. Early ownership was nearly uninformative, and no uniform-0.5 control was run, so ownership routing and reduced gradient mass cannot be separated.

Completion of attention-owned-compartment-gradient-routing: the fixed rule gives reject. Macro Δ is −0.00392, 95% CI [−0.00769, −0.00027]. The gates the developer flagged as pending are now measured and pass. Compute was 1,617 s against a 1,797 s ceiling. Triplets, visits and cohort hashes match exactly, and validation selection is identical in every epoch.

The escalation concern was the confound with gradient mass, and the logged kept share resolves it against the mechanism:
- Kept share starts at about 0.51 in both families. Meniscus rises to 0.80 by epoch 9, but OA reaches only 0.60.
- So for most of training the four compartment labels sent roughly half of their trunk gradient. In the early epochs the intervention was close to a uniform 0.5 scale on compartment trunk credit.
- Medial and lateral kept shares are identical by construction, since Σa_m²/(a_m+a_l) − Σa_l²/(a_m+a_l) = Σa_m − Σa_l = 0. This is not a logging bug.

Routing did not buy specificity:
- Meniscus compartment margin AUC fell from 0.9712 to 0.9615, and OA's was flat (0.8661 to 0.8650).
- Lateral cross-talk did not improve (meniscus 0.636 to 0.638, OA 0.622 to 0.638).
- Both family means fell (meniscus −0.0069, OA −0.0036), and so did the other eight (−0.0033).

Queue change: I admitted the anatomy-supervised split. Its dependencies are satisfied. It adds a shared evidence path with disease-free, geometry-supervised assignment, so it targets the measured assignment weakness without removing any compartment credit. I ranked it right after the reserved head.

Within-knee ranking is additive supervision, so it moves above the two subtractive trunk or attention-gradient interventions. Those two (the sibling adversary and the discordant projection) share the direction this run weakened, withholding or reversing compartment gradient on the shared trunk. They stay queued, because each acts more narrowly than uniform-scale routing: the adversary is confined to attention rows, and the projection to discordant studies and conflicting components.

Unresolved concerns:
- Attribution: there is no uniform-0.5 trunk-scale control, so the decline could come from reduced compartment gradient mass as much as from ownership routing. The run should not be read as evidence about routing specificity itself.
- Kept share is logged only as a batch mean per epoch. The within-study distribution of ownership is unknown.
- Single seed on fold 0. The bootstrap excludes seed and checkpoint-selection variance. The upper bound (−0.00027) is close to zero.
- The sibling adversary and discordant projection also withhold or reverse compartment gradient on shared representations. Their prior weakened, but they were not directly tested.
- For the admitted anatomy-supervised split: geometry keys must be read only in training_step, not the model forward (the axis arm's MODEL_BATCH_KEYS failure). Gate accuracy and coverage must be logged, otherwise a null result cannot be interpreted.

Evidence:
- outputs/autoresearch/medial-lateral-transfer/experiments/attention-owned-compartment-gradient-routing-20260921T055258Z-2e5a20a0/comparison/comparison.json (family_means, compartments, paired_report_group_bootstrap)
- outputs/autoresearch/medial-lateral-transfer/experiments/attention-owned-compartment-gradient-routing-20260921T055258Z-2e5a20a0/comparison/per_label.json
- outputs/autoresearch/medial-lateral-transfer/experiments/attention-owned-compartment-gradient-routing-20260921T055258Z-2e5a20a0/training/fold_0/logs/version_0/metrics.csv (gradient_routing_kept_share per epoch: meniscus 0.514→0.797, OA 0.513→0.601)
- outputs/autoresearch/medial-lateral-transfer/experiments/attention-owned-compartment-gradient-routing-20260921T055258Z-2e5a20a0/changes.diff
- outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T060120Z-c0f0b3f6/report.json (prior review concerns)
- outputs/autoresearch/medial-lateral-transfer/state.json: anatomy-supervised-compartment-evidence-split proposal record

## user-adopt-learned-strength-20260921T115849Z-b19e375d

User adopted learned anatomical strength as the reference and cleaned source for future experiments; campaign remains stopped.

Matched-seed control gain +0.0023308413 with positive 80% interval. This is an explicit reference replacement, not a retroactive campaign verdict. Original limits, baseline snapshots, historical run references and decisions are preserved. Pending old-reference events are archived; no jobs launched.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/adopted/user-adopt-learned-strength-20260921T115849Z-b19e375d/adoption.json
- /home/claudio/src/knee-abnormality/outputs/analysis/anatomical-strength-vs-control-repeat-ci-levels/adoption_assessment.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/decisions/user-adopt-learned-strength-20260921T115849Z-b19e375d/pending-events.json

## user-compartment-revisit-20260921-1789998712642067442

User broadened this campaign to improving compartment-specific meniscus/OA predictions, including knee-side/position encoding and anatomy-supervised assignment. The learned-strength incumbent and seed 2263738815 remain pinned. Three historical adaptations and one explicit side-location encoding proposal are queued ahead of the updated anatomy-supervised gate. No training or resume was launched.

Ranked by a concrete interaction with the adopted anatomical model, mechanism activity, and cost. The old global shared-family channel was nearly unused and its +0.000893 macro gain mainly came from unrelated labels, so it is not queued for unchanged replay. The appearance-only tied-classifier variant harmed meniscus; its successor retains position-rich pooling and draws supporting precedent from older 1A. Exclusivity had targeted OA benefit but net regression. Local lesion evidence was cancelled, not scientifically rejected. The other three previously queued ideas remain proposed, with histories intact.

Unresolved concerns:
- Historical different-seed/data results prioritize hypotheses; they do not establish gains over learned strength. No automatic confirmatory or coefficient-sweep runs.

Evidence:
- autoresearch/campaigns/medial-lateral-transfer/prior-work.md#revisit-queue-on-learned-strength
- autoresearch/anatomical-attention.md
- autoresearch/checkpoint-closure.md
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/decisions/user-compartment-revisit-20260921-1789998712642067442/decision.json

## review-20260921T141140Z-b18934db

The static check passed. SharedMeniscusHead.logits equals a 12-row linear readout in which rows 2 and 3 share one 256-d weight and keep separate biases, with the other ten rows unchanged in order. The shared parameters get head_lr, and weight decay applies only to the 2-d weight.

For interpretation, the meniscus pair now has 257 fewer readout parameters than the incumbent. Any meniscus gain therefore cannot come from extra capacity; it would have to come from shared morphology read through the compartment-specific attention.

Initialisation is not identical to the incumbent even with the same seed. The shared head is built before self.head, so the RNG draws for the 10-row head and SpatialTokens are consumed in a different order, and the meniscus biases start at zero rather than uniform. Differences on the other eight labels therefore include initialisation noise.

I'm approving without repair. The candidate isolates exactly the assigned change: it shares the meniscus classifier weight while keeping the adopted learned anatomical attention and the position-rich pooled values. That separates classifier sharing from removing position, the confound in the earlier appearance-only tie.

- **Compartments stay distinct.** Each compartment gets its own attention distribution with an opposite-signed learned anatomical bias, plus its own bias term. So the shared weight reads different pooled evidence for medial and lateral, and all four medial/lateral outcomes (neither, medial-only, lateral-only, both) remain representable.
- **Not a repeat.** It differs from the rejected pooled adapter. It also differs from historical 1A: 1A ran on an older recipe without learned anatomical attention, and prior-work.md:137 records it as external evidence, not an arm run under this incumbent.
- **Checks passed.** I found no leakage, no change to validation independence, no change to masks, labels or preprocessing inputs, and no added preparation cost.

Unresolved concerns:
- Same seed, different initialisation: the new module and head shape change the parameter RNG order, so off-target per-label shifts partly reflect init noise rather than transfer.
- The meniscus biases start at zero while the private rows use the default uniform initialisation. This is a small asymmetry; interpret early calibration differences with care.
- Per-label meniscus intervals remain exploratory. A gain limited to the meniscus pair would not by itself meet the macro rule, and a non-meniscus gain would not establish transfer.

Evidence:
- outputs/autoresearch/medial-lateral-transfer/experiments/anatomical-position-rich-shared-meniscus-readout-20260921T135727Z-cf9db461/changes.diff
- source/src/knee_abnormality/modeling/compartment_transfer/shared_meniscus_head.py:38-51 (split/reassemble readout)
- source/src/knee_abnormality/modeling/model.py:156-165 (head construction), 241-264 (anatomical attention unchanged; shared readout branch)
- source/src/knee_abnormality/training/lightning.py:35-45 (non-backbone params -> head_lr; ndim<2 no decay)
- source/src/knee_abnormality/config.py:179-221 (SharedMeniscusHeadConfig, residual_readout guard)
- autoresearch/campaigns/medial-lateral-transfer/prior-work.md:25,137 (1A historical context)

## review-20260921T141927Z-da44869a

The static review passed and all four edited files parse. The default forward, pool_encoded, validation and prediction paths are unchanged, and the loss branch applies only when trainer.compartment_transfer.enabled. With anatomical bias, the sibling's attention peaks lie on the opposite side of the medial axis. Erasing them should therefore mostly remove the shared, midline or non-compartmental high-attention tokens rather than the compartment's own evidence. Read the result against the archived arm's pattern: cross-talk diagnostics moved in the intended direction while the compartment-margin AUCs fell. If the incumbent's learned strengths already separate the pairs, the penalty may often be zero, because the drop stays below the margin. The logged train_compartment_transfer_penalty should reveal whether the objective was active at all.

I approve this candidate. Implementation matches the assigned mechanism and the declared integration. The erasure uses the final anatomically biased attention and runs inside the existing autocast-disabled block. The sibling mask is detached, so gradients reach only the compartment's own attention, the token values and the head. A non-member token has zero attention of its own, so erasing it has no effect, and amax is taken per study row. The added cost is 4 extra pools per step (single einsums) and 4 head-row applications. That is negligible next to the encoder, so the 1,797 s ceiling is not at risk. No new RNG consumption appears in the candidate code, so the same seed gives the same initialization and data order as the incumbent up to divergence introduced by the added loss. Validation stays unaugmented and does not depend on the seed. The penalty uses only training targets and observed masks, so there is no leakage. The coefficients (λ=0.05, margin=1.0) are reused from the archived arm rather than tuned. The archived arm was inconclusive (−0.0007 macro, with compartment-margin AUCs falling). Its revisit is justified only because the incumbent now has compartment-biased attention, which makes sibling erasure anatomically meaningful. That is a design-context change, not strong new evidence, so prior expectations for a ≥0.001 gain should stay modest.

Unresolved concerns:
- Low prior: this is the same penalty, coefficients and margin as an inconclusive arm, with only the attention source changed. A null result would not establish anything beyond this configuration, and a coefficient sweep must not follow.
- If the anatomical bias already makes the sibling attention peaks nearly disjoint from the own-compartment peaks, the hinge may rarely activate. Check train_compartment_transfer_penalty magnitude before reading any effect as mechanistic.
- Standing provenance gap: training/compartment_transfer/objective.py is outside identity.PREPROCESSING_SOURCES. Its provenance rests on the candidate source hash b31d2285... and the source snapshot.
- The unused empty modeling/compartment_transfer/ directory is present but was never declared. It is inert.
- The GPU compute overhead was bounded by operation count only and not measured. The runtime ceiling check must come from the recorded timing.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/anatomical-attention-attribution-exclusivity-20260921T141236Z-855ccb19/changes.diff
- source/src/knee_abnormality/modeling/model.py:238-264 (_reattributed_logit / _masked_reattribution_logits, detached sibling mask)
- source/src/knee_abnormality/modeling/model.py:266-319 (_attention_pool; default return path unchanged)
- source/src/knee_abnormality/modeling/model.py:413-443 (pool_encoded unchanged; pool_encoded_with_transfer mirrors its arguments)
- source/src/knee_abnormality/training/lightning.py training_step: the penalty branch applies only when enabled; forward/prediction use self.model(batch)
- source/src/knee_abnormality/inference/checkpoint.py:105 strict load_state_dict; no new parameters were added
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T141236Z-d1468717/report.json
- Archived arm cross-compartment-attribution-exclusivity-20260921T001439Z-46286939 comparison.json (macro −0.0007, 95% [−0.0047, 0.0030])

## completion-20260921T143912Z-7971addd

Sharing one meniscus readout weight on the incumbent's anatomical, position-rich attention hurt the meniscus pair, most of all the lateral compartment:
- Meniscus family: 0.9119 → 0.9049, interval [−0.0131, −0.0010].
- Lateral Meniscus: −0.0125, interval [−0.0223, −0.0033], with 138 positives.
- Medial Meniscus: −0.0017, interval [−0.0089, +0.0065].
- Meniscus compartment-margin AUC: 0.9693 → 0.9589, interval [−0.0212, −0.0013].
- Lateral false positives rose from 45 to 49, and cross-talk from 0.644 to 0.653.

Taken with the earlier geometry-free tie (meniscus family −0.0075, with Medial Meniscus significantly lower), this is the second independent readout-tying implementation to reduce meniscus discrimination. Tying the classifier weights is therefore unlikely to be the useful transfer channel on this recipe. The shared direction mostly serves the majority medial compartment and costs the scarcer lateral one. This is a statement about readout tying only, not about medial/lateral sharing in general.

OA readouts were untouched, yet the OA family also fell (−0.0071, interval [−0.0147, +0.0003]):
- Lateral OA: −0.0127, interval [−0.0251, −0.0013].
- OA margin AUC: unchanged at 0.861 (interval ±0.03).
- OA lateral cross-talk: fell from 0.622 to 0.585.

The OA loss must come through the shared attention and trunk, or from initialisation or seed noise, which this design cannot separate. It is not evidence about OA transfer.

The other eight targets were flat (+0.0001). Mean absolute change in predicted probability was 0.065, similar to the previous arm, which bounds the single-run noise floor. The trained shared weight has norm 0.54 and the separate biases have norm 0.019, so the pair differs almost only through attention.

This is a completion interpretation. The fixed eligible_decision is reject, and the evidence supports it. Macro AUC went from 0.8974 to 0.8952 (−0.0023). The 80% interval is [−0.0048, +0.0002] and the 95% interval is [−0.0060, +0.0015]. The gain is below the 0.001 minimum and the 80% lower bound is not positive.

The run is valid:
- Accounting matches the sealed counts: 938,460 training triplets and 35,030 visits, with the sealed training and validation cohort hashes.
- Measured compute was 1,592.6 s, under the 1,797 s ceiling.
- The seed was the incumbent's (2263738815), and the checkpoint came from epoch 8.
- The config diff lists anatomical_attention_bias and learned_anatomical_strength as null in the candidate. This is expected: the adopted source makes them fixed behaviour. model.py:129 and :244 of the candidate still build and apply anatomical_log_strength. The only real config change is shared_meniscus_head.target_pair.
- The diff touches only training.yaml, config.py, shared_meniscus_head.py and model.py.

I escalated because of unresolved concerns from the earlier review. Two matter here:
- RNG-order and initialisation differences. These plausibly explain the off-target shifts (Fracture −0.010, MCL +0.007), and the other-eight mean is flat (+0.0001).
- The zero-initialised meniscus biases.

Neither concern explains the pattern on the target family, which moved consistently in the harmful direction. I made no queue or proposal changes. The three admissible proposals keep private compartment evidence, which fits this result. They stay proposed rather than displacing the current queue.

Unresolved concerns:
- All per-label and family intervals are unadjusted and exclude seed and checkpoint-selection variance. The Lateral OA loss on an untouched readout shows that perturbations from initialisation and RNG order alone can produce significant-looking single-label shifts of about 0.012.
- Readout tying is now inconclusive or negative in two implementations. It should not be revisited through width or coefficient variants, or through partial tying of the same readout, without a materially different mechanism.
- changed_source_archive_members again includes autoresearch harness files and experiment configs outside the candidate diff. Attribution rests on the four diffed files (changes.diff, 159 lines).

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/anatomical-position-rich-shared-meniscus-readout-20260921T135727Z-cf9db461/comparison/comparison.json (family_means, compartments, paired_report_group_bootstrap.intervals.meniscus/oa/meniscus_margin, parameters_only_in_run)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/anatomical-position-rich-shared-meniscus-readout-20260921T135727Z-cf9db461/comparison/per_label.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/anatomical-position-rich-shared-meniscus-readout-20260921T135727Z-cf9db461/changes.diff
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/anatomical-position-rich-shared-meniscus-readout-20260921T135727Z-cf9db461/source/src/knee_abnormality/modeling/model.py:129,244 (anatomical strength retained)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/anatomical-position-rich-shared-meniscus-readout-20260921T135727Z-cf9db461/training/fold_0/timing.json and triplet_accounting.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T141140Z-1834f914/report.json (pre-launch review concerns)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T045640Z-09f95914/report.json (prior geometry-free tie result)

## review-20260921T144654Z-e85b2ec7

Against this incumbent, the shared evidence enters through attention that is already anatomically biased toward opposite sides of the medial axis. The medial and lateral contributions therefore weight the same family map at largely different locations. Compartment identity comes from where attention looks, not from the sign of the evidence. This is a cleaner test of "shared morphology detector, compartment-specific reading" than the attempt on control B.

Read the result from these diagnostics:
- **`local_evidence_mean_abs` and the family vector norms.** Collapse toward zero means the path went inert.
- **Family means versus the other eight labels.** Movement in the other eight comes only through shared-trunk dynamics.
- **Compartment margins and sibling-diseased negatives.** A shared term can still lift both siblings where their attention overlaps near the midline.

Attribution still carries a capacity confound: about 16.8k parameters plus direct stride-16 input. A private four-vector control on the same path would be needed before crediting sharing.

The review passed. This re-bases the previously reviewed mechanism onto the adopted learned-strength incumbent. The earlier attempt was invalid only because its run was cancelled at 830 s; that was not a code defect. The source diff (`config.py`, `model.py`, `lightning.py`, and the new `modeling/compartment_transfer/local_evidence.py`) matches the declared integration. The prior hardcoded-index concern is resolved.

I checked tensor layout, masking, dtype handling (float32 outside autocast), timm `forward_intermediates` equivalence, the disabled-path identity, config scope and checkpoint reconstruction directly. The queue is unchanged.

Unresolved concerns:
- Runtime is unmeasured on GPU. `forward_intermediates` adds only a list append, and the new head is small, but the 1,797 s ceiling must come from the recorded `timing.json`. The previous attempt produced no accounting.
- Attribution: a macro gain would mix family sharing with added stride-16 capacity. A matched private-vector control is needed before crediting shared morphology.
- The evidence term is ungated and unscaled, so near-midline tokens with overlapping sibling attention push both logits the same way. Check compartment margins and sibling-diseased negatives, not only family means.
- The tent pool at stride 2 with padding 1 centres stride-32 cell i on stride-16 cell 2i. That is consistent with timm's convolution-centre convention to within a half stride-16 cell, which is minor. It was not verified numerically against `feature_geometry` anchors.
- `local_evidence.py` is a new helper module whose provenance rests on the candidate source hash `de4ea6cc…` and the manifest, not the preprocessing sources hash. `training/compartment_transfer/` exists but is empty in this source and is inert.

Evidence:
- outputs/autoresearch/medial-lateral-transfer/experiments/anatomical-attention-local-lesion-evidence-20260921T144002Z-147e1390/changes.diff
- source/src/knee_abnormality/modeling/model.py:181-194: `_encode_images`, where the disabled path keeps `forward_features`
- source/src/knee_abnormality/modeling/model.py:288-294: the evidence term added via final post-bias attention rows
- source/src/knee_abnormality/modeling/model.py:396-411: `pool_encoded` flattening (N,49) to match the study, validity and medial_axis token order
- source/src/knee_abnormality/modeling/compartment_transfer/local_evidence.py: `stride16_stage` and `LocalEvidence`
- .venv timm/models/rexnet.py:367-413: `forward_intermediates` runs stem plus all features, equivalent to `forward_features`
- workspace/configs/training.yaml:87-89: `model.compartment_transfer.local_evidence.width: 64`, the only config difference
- outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T063503Z-62fd07ae/report.json: prior review of the same mechanism on control B

## proposals-20260921T145603Z-d4c51a19

New CPU diagnostics (57 known-side fold-0 studies, eval mode, nothing fitted):
- **The prior replaced appearance-based lateralization.** In the incumbent, the pre-bias medial and lateral attention centroids are 6.2 mm apart for meniscus and 0.6 mm for OA. In the no-bias control-repeat they are 32.0 and 11.8 mm. After the bias, the incumbent reaches 52.1 and 39.7 mm.
- **The metadata knee centre is adequate.** The incumbent's ACL sagittal attention centroid lies a median 3.6 mm from it (p90 9.6 mm).
- **Wrong-side leakage is small.** Biased leak is at most +0.22 logit, and its sign follows the sibling's label (co-occurrence).
- **Unknown-side studies are few.** Fold 0 has 42 of 904. Pair AUCs involving them are mixed versus control, on 9–18 positives.

This evidence lowers the priority of centre refinement, anti-leak gating and medial/lateral row tying. It is single-seed and single-checkpoint. Raw separation on training studies and on the fixed-bias checkpoint was not measured, so "gradient starvation" remains a hypothesis rather than a demonstrated cause.

I admitted unbiased-credit-compartment-attention. It has no dependencies, adds no parameters and no knob, and leaves the incumbent's forward pass unchanged. It uses no RNG draws, so under the seed policy its data and augmentation stream matches the incumbent's exactly, which makes attribution unusually clean. It targets a newly measured weakness (evidence-summary.md): before the anatomical bias, the incumbent's medial and lateral attention rows are 6.2/0.6 mm apart (meniscus/OA), against 32.0/11.8 mm in the same-seed no-bias control. The OA margin is also -0.0117 versus that control. This fits the scope: shared evidence must be read by the correct compartment, and making the assignment redundant (appearance plus geometry) serves that. I ranked it second, behind the reserved head. It goes ahead of explicit-knee-side-coordinate-encoding because that idea puts geometry into the raw scores, which would muddy the proposal's definition of "raw" if it landed first. The proposal itself flags this.

I left anatomical-bias-study-dropout as proposed rather than queuing it. It tests the same diagnosis in a more conventional way. It adds a fixed knob (p = 0.25) and a train/inference regime mismatch. Its extra selling point, side-unknown robustness, has little support on fold 0: 42 studies and 9–18 positives. Queuing both would use two slots on one hypothesis that rests on a 57-study, single-seed probe. If unbiased-credit is accepted, the proposal already describes how dropout would become an A+B stacking candidate.

Open choice to settle during implementation: the proposal's default, "trunk tokens receive the rerouted score gradient", matches the incumbent graph, so that should be the fixed behaviour. No switch between the two options should be kept.

Unresolved concerns:
- The substitution diagnosis rests on one seed pair and 57 validation studies. Raw separation could shrink simply because the bias makes appearance cues redundant, with no harm done. The OA-margin deficit (-0.0117) is a single, unreplicated difference.
- The straight-through score credit is a biased gradient estimator: raw scores receive gradients through the Jacobian of the unbiased softmax, while the forward pass uses the biased one. It may destabilize the attention MLP or push the rows to over-separate. Check the logged a_b/a_u total-variation distance and the learned strengths. The developer must also confirm that eval and predict run exactly the unchanged a_b path.
- Any macro change from unbiased-credit can only come through the attention-score gradients of the four compartment rows. Use the center_probe_any.py rerun to judge whether the mechanism engaged. It is a diagnostic only.

Evidence:
- outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T144003Z-99678cca/report.json
- outputs/autoresearch/medial-lateral-transfer/jobs/researcher-20260921T144003Z-99678cca/scratch/evidence-summary.md (separation table, leak table, unknown-side AUCs)
- same folder: center_summary.txt, control_summary.txt, leak_probe.log, center_probe_any.py
- autoresearch/anatomical-attention.md#bias-off-inference-ablation and #repeated-control-on-current-data

## completion-20260921T150711Z-a2530c31

Anatomical-attention attribution exclusivity was mechanically inactive. The own-logit drop after erasing sibling-attended tokens (post-bias softmax) stayed below margin 1.0 in essentially all positive cells, epoch-mean raw penalty ≤2.7e-5, all logged step values 0. This supports the review's prediction that the learned anatomical bias already makes sibling attention peaks largely disjoint from own-compartment evidence. It says nothing about exclusivity as a lever; making the hinge active would require a margin/coefficient change, which is a parameter sweep and not admissible.

Useful by-product — same-seed noise floor for this incumbent: prediction Spearman 0.993–0.998 per target, mean |Δprob| 0.0142, macro Δ -0.0002 (80% [-0.0008,+0.0004]). Per-label deltas reach ±0.0019–0.0027 (Synovitis +0.0019, Contusion -0.0027, Lateral Meniscus -0.0017, Fracture -0.0016) with no mechanism behind them. Family means: meniscus -0.0009, OA -0.0002, other eight -0.00003. Compartment margins unchanged (meniscus 0.9693→0.9696, OA 0.8612→0.8613); cross-talk moved in opposite directions (meniscus 0.644→0.667, OA 0.622→0.597) by 1–3 FP counts. Future candidates on this incumbent should read per-label moves of ~0.002–0.003 and cross-talk shifts of a few FPs as within same-seed run-to-run variation. Selected epoch 10 (index 9, 0.89724), epochs 9–10 within 0.0003.

The fixed eligible_decision is reject: macro 0.89724→0.89744... candidate 0.89724 vs incumbent 0.89744, delta mean -0.00020, 80% [-0.00081, +0.00039], 95% [-0.00113, +0.00072]. It fails both the +0.001 gain and positive-lower-bound conditions. The run is valid: accounting complete and matching sealed counts (938,460 train / 24,188 val triplets), seed 2263738815, measured compute 1,627.6 s under the 1,797 s ceiling, only the four declared trainer.compartment_transfer keys differ.

The decisive new fact, which resolves the review's open concern, is that the objective was essentially inert. Logged raw train_compartment_transfer_penalty is exactly 0 on every sampled step and its epoch means are ≤2.7e-5 (epochs 0–2 and 7 exactly zero); with lambda 0.05 the loss contribution is ≤1.4e-6. The hinge almost never fired: under the learned anatomical bias, erasing the sibling's top-attended tokens rarely dropped the own-compartment logit by more than the 1.0 margin. So this is not a test of attribution exclusivity; it is in effect a same-seed replicate of the incumbent differing by GPU nondeterminism. No queue change follows; the pending proposals are not affected by this evidence and stay proposed.

Unresolved concerns:
- The hinge was inactive, so neither this run nor the archived control-B arm tests whether an active exclusivity constraint on anatomically biased attention helps; reviving it would require a margin/lambda change, which is a coefficient sweep and must not be admitted.
- The noise-floor reading comes from one same-seed pair; it bounds nondeterminism for this incumbent only and excludes seed and fold variation.
- Standing provenance gap persists: training/compartment_transfer/objective.py lies outside PREPROCESSING_SOURCES; provenance rests on the candidate source hash and snapshot. The undeclared empty modeling/compartment_transfer/ directory is inert.

Evidence:
- outputs/autoresearch/medial-lateral-transfer/experiments/anatomical-attention-attribution-exclusivity-20260921T141236Z-855ccb19/training/fold_0/logs/version_0/metrics.csv (train_compartment_transfer_penalty_step all 0; epoch means ≤2.7e-5; val_macro_auc curve)
- …/855ccb19/source/src/knee_abnormality/training/compartment_transfer/objective.py:15-34 (squared hinge on logit drop minus margin, positive cells only)
- …/855ccb19/source/src/knee_abnormality/training/lightning.py:339-354 (raw penalty logged; loss += lambda_weight*penalty)
- …/855ccb19/comparison/comparison.json (compartments, family_means, prediction_change Spearman 0.993–0.998, mean |Δprob| 0.0142)
- …/855ccb19/comparison/per_label.json
- …/855ccb19/training/fold_0/selected_checkpoint.json (best-epoch=09, 0.89724)
- outputs/autoresearch/medial-lateral-transfer/jobs/orchestrator-20260921T141928Z-647fd5f2/report.json (review predicting possible hinge inactivity)

## user-fix-exclusivity-and-retry-aborted-20260921T152036Z

The user-aborted local-lesion attempt is finalized invalid without a scientific verdict. Corrected linear zero-margin attribution exclusivity is queued first, followed by a fresh epoch-0 retry of the reviewed local-lesion mechanism.

The old exclusivity margin and squared hinge produced zero or negligible signal. Its successor uses linear relu(z_own-z_erased) at zero margin with fixed weight 0.05 and ordinary full-budget training. No monitor, adaptive stop or parameter sweep is authorized. The previous local-lesion training ended after two epochs by user abort and must run fresh. The unlaunched unbiased-credit claim was archived and its idea requeued third.

Evidence:
- autoresearch/campaigns/medial-lateral-transfer/corrected-exclusivity.yaml
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/anatomical-attention-local-lesion-evidence-20260921T144002Z-147e1390/evaluation.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/decisions/user-fix-exclusivity-and-retry-aborted-20260921T152036Z/decision.json

## review-20260921T154938Z-5b4d2cee

Review found no correctness defect. With zero margin, the hinge is active for every observed positive whose own logit falls when the sibling-attended tokens are erased. The prior run's diagnostics showed such drops were small but not always negative, so a nonzero activation fraction is expected this time. The logged activation_fraction and raw_positive_drop must confirm this before the result is read as a test of exclusivity. The gradient flows through both z_own and z_erased, and the sibling mask is detached. The penalty can therefore be reduced in two ways: by moving own attention off sibling peaks (the intended route), or by lowering z_own for positives. The second route works against BCE, at weight 0.05.

Approved. The diff implements the registered correction exactly and nothing else. It keeps the post-bias sibling erasure with detached sibling masks, replaces the dormant squared margin-1 hinge with the linear zero-margin hinge, keeps the archived lambda 0.05 as a single fixed attempt, and handles zero remaining mass explicitly by excluding those cells from both the loss and the diagnostics. Both families are covered. Masking uses target_observed AND validity, so unobserved labels contribute nothing. The added compute is four reattribution einsums over the batch's own tokens inside the existing forward pass. There is no extra encoder pass, no extra optimizer update, and no validation-derived quantity. Preparation took 0.12 s, and the rejected arm with the same op pattern stayed inside the budget. The config diff is limited to the trainer.compartment_transfer prefix. The seed is the incumbent's 2263738815, as the policy requires. No leakage path was found: the penalty uses only training-batch targets and masks.

Unresolved concerns:
- The diagnostics are pooled over both families. The idea's coverage asked for activation per family (meniscus vs OA), which this logging cannot separate. Interpretation will be limited to pooled activation plus the per-label and family AUC outcomes.
- The hinge can also be reduced by lowering z_own on positives, which directly opposes BCE. The raw_positive_drop trend alone cannot tell this apart from genuine attention reallocation.
- valid_fraction averages over all batch×pair×slot cells, including unobserved targets. It is only a rough degeneracy indicator.
- Provenance gap remains: training/compartment_transfer/objective.py is outside PREPROCESSING_SOURCES, so provenance rests on the candidate source hash and snapshot.
- Runtime was bounded only by operation count; nothing was measured on GPU. The 1,797 s ceiling will be enforced when the run executes.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/experiments/active-gradient-attribution-exclusivity-20260921T152136Z-476463c0/changes.diff
- source/src/knee_abnormality/modeling/model.py:270-324 (_attention_pool with transfer branch), model.py:220-268 (_reattributed_logit, validity mask)
- source/src/knee_abnormality/training/compartment_transfer/objective.py (linear zero-margin hinge, observed&valid reduction)
- source/src/knee_abnormality/training/lightning.py training_step compartment_transfer branch
- autoresearch/campaigns/medial-lateral-transfer/ideas.yaml:3960-4000 (registered mechanism/coverage)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/medial-lateral-transfer/jobs/developer-20260921T154557Z-4969a48f/report.json
