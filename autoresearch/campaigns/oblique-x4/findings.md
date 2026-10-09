# Campaign findings

## proposals-20260922T133831Z-ab90c546

No new experimental evidence at this event. The admitted queue covers three identity and routing mechanisms on the oblique groups, plus one fixed routing prior. The retired x1 campaign showed that the budget's loss sits in the 103 split studies: `k_05` beat the reference there by +0.0103, 95% [-0.0027, +0.0234], with the loss mainly in MCL (11 positives) and lateral meniscus (22). The unchanged studies tied: 0.9018 against 0.9017. Any mechanism claim here must therefore rest on the split-study deltas. With so few positives, per-label split-study intervals will be wide; they are diagnostic only.

The x1 view-bias table and the fixed ACL gate point the same way for non-ACL labels (the obliques were discounted for the menisci, contusion and fracture) but disagree on ACL (not leaning on the obliques). If the view bias runs first, its learned ACL column at x4 bears directly on whether the gate prior is worth a run.

I admitted four of the five imported seeds. All dependencies are empty, so proposal_admission passes. `oblique-residual-readout` is not admitted, following the recorded user decision in prior-work.md: its x1 gain came entirely from the unchanged studies (+0.0036, against +0.0005 on the split studies). It stays proposed.

The x1 rejections of the embedding and the view bias do not bar retesting. Both sat on the adoption line: 80% lower bounds of -0.0005 and -0.0002. This campaign exists to re-score the topic under the new x4 evaluation.

Ranking:
1. `label-view-attention-bias`. It gave the largest partial repair on the split studies at x1 (+0.0068 against the reference). Its learned table was anatomically coherent. It follows the form of the adopted anatomical bias and starts at zero.
2. `oblique-identity-with-view-bias`. It tests whether the two partial repairs add up enough to pass `k_05` on the split studies, which neither did alone. It reuses both existing diffs.
3. `oblique-group-embedding`. The smallest repair on the split studies at x1 (+0.0052), and the combination covers its mechanism. It is the cheapest standalone retest of the identity family.
4. `acl-gated-oblique-tokens`. It has no parameters and uses the same plumbing. Its evidence is thin: +0.0020 on 102 affected studies, from inference only, on `k_05` at τ 0.75 rather than the τ 1 evaluation. The x1 learned table shows ACL did not lean on the oblique groups (-0.05), while the menisci, contusion and fracture discounted them (-0.09 to -0.14). That supports masking the obliques out of non-ACL labels, but it does not show ACL needs them. It stays last because a hard mask changes the forward pass from step 0 (see concerns).

Unresolved concerns:
- acl-gated-oblique-tokens: a hard mask (or large fixed negative bias) makes the untrained forward pass differ from the reference, in tension with the contract's neutral-start constraint. Implementation review must decide whether a parameter-free fixed prior is acceptable. A learned gate that starts at zero would be the neutral alternative, but that is close to label-view-attention-bias.
- acl-gated precedent is inference-only on k_05 at τ 0.75, not the campaign's ×4 τ 1 evaluation; its +0.0020 on 102 studies has no recorded interval.
- The per-label split-study evidence rests on 11 MCL and 22 lateral meniscus positives in the 103 studies, so completions should not read single-label split-study moves as mechanism confirmation.
- oblique-identity-with-view-bias and oblique-group-embedding: open choice left unresolved of one shared vector or one per type, and where the axial-labelled coronal ACL obliques go. The developer must fix and record it before training.

Evidence:
- autoresearch/campaigns/oblique-x4/prior-work.md#user-decision-seed-proposals-not-to-admit-2026-09-22 (residual readout excluded by the user)
- autoresearch/campaigns/oblique-x4/prior-work.md#results-of-the-retired-x1-campaign (x1 split and unchanged table, learned view-bias table, ACL-gate inference test)
- outputs/autoresearch/oblique/experiments/label-view-attention-bias-20260922T120824Z-29aa5863
- outputs/autoresearch/oblique/experiments/oblique-group-embedding-20260922T115506Z-f5d616dd
- outputs/analysis/t/tilted-slot-policies-v2-k05-20260922/results.csv
- outputs/analysis/a/acquisition-geometry-20260922/localize_campaign.py
- outputs/autoresearch/oblique-x4/imports/ideas-20260922T133831Z-e422f2c0/report.json

## proposals-20260922T133831Z-51ff4c56

The mixture-pooling proposal leaves several points open.
- Part of its frozen advantage comes from ensembling two sub-bags, not from using the oblique views. `mean_intact_standard` gained +0.0000 to +0.0037 on the subset without the tilted series. The frozen checkpoints also never trained with split budgets.
- The within-model toggle the proposal lists (joint vs mixture, pass A only, pass B only) is the evidence that would separate the pooling effect from training divergence. Every completion must still report split and non-split deltas against the reference and against `k_05` at ×4.
- With fixed ½ weights and no gate, split studies do not reproduce the reference forward pass at initialisation. The contract's neutrality clause covers new parameters, and alternative pooling of split slots is in scope, so a parameter-free change looks permissible. Review should still confirm that the declared form fits the contract, or use the zero-initialised per-label λ gate.
- The device move of `series_oblique` in `model.py`, and deriving split slots from kept, visually valid tokens after series dropout, are the correctness points most likely to fail at review.

Admitted `split-slot-view-mixture-pooling` and ranked it second. Of the in-scope ideas, it has the most consistent frozen evidence. Against joint pooling, the mixture won on the split subset in all 9 checkpoint×setting cells, by +0.0015 to +0.0039. It adds no parameters, gives non-split studies the reference forward pass exactly, and costs almost nothing. It stays behind `label-view-attention-bias` because on the full fold its frozen gain is only +0.0002 to +0.0006, well short of the 0.001 adoption line. By contrast, the view bias gained +0.0018 macro at ×1 and repaired +0.0068 on the split studies. I placed it ahead of the combination, the ACL gate and the plain embedding because it tests a different lever: removing within-slot competition rather than marking identity.

Following the user's recorded decision in prior-work.md, I did not admit `oblique-residual-readout`, `view-orientation-embedding` or `oblique-feature-adapter`. They stay in the proposed state.

`oblique-training-only-views` and `coverage-matched-slot-split` are eligible and the user kept them, but I left them proposed to hold the queue at five:
- Training-only views: its frozen ×4 subset gain (+0.0011 to +0.0026) partly overlaps the mixture's pass A and the ACL gate. It also trains on split slots but evaluates without the oblique tokens, which adds a train/eval shift.
- Coverage-matched split: the proposal itself says budget changes alone have not moved fold 0.

Both can be queued once a queued idea resolves. I did not re-rank the other four existing entries against each other.

Unresolved concerns:
- Mixture pooling is not neutral at initialisation for split studies with fixed ½ weights and no gate. Review must decide whether the contract permits this for a parameter-free pooling change or requires the zero-initialised λ gate.
- Part of the frozen mixture gain is sub-bag ensembling (mean_intact_standard reached up to +0.0037 on the subset). A macro gain would not by itself show that the oblique views are being used; the within-model pooling toggles are needed.
- Frozen subset gains shrink about tenfold on the full fold (+0.0002 to +0.0006), so reaching the 0.001 adoption line needs a trained effect much larger than the frozen one.
- Split-slot detection must use kept, visually valid tokens after series dropout, and series_oblique must reach the model device; engine.move_batch leaves unlisted keys on the CPU.

Evidence:
- /home/claudio/src/knee-abnormality/autoresearch/campaigns/oblique-x4/prior-work.md (user decision not to admit residual readout, orientation embedding, feature adapter or axial transfer; x1 results table)
- /home/claudio/src/knee-abnormality/outputs/analysis/t/tilted-slot-policies-check-20260922/results.csv and tilted-slot-policies-v2check-20260922/results.csv (mixture vs joint on the split subset, 9/9 cells)
- /home/claudio/src/knee-abnormality/outputs/analysis/t/tilted-slot-policies-v2-k05-20260922/results.csv (ACL-only gate +0.0020 on 102 studies at ×4; standard-only +0.0011)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-x4/imports/ideas-20260922T133831Z-b9d539ef/report.json (proposal texts)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique/experiments/split-slot-view-mixture-pooling-20260922T125450Z-e32cf52c (cancelled x1 implementation)

## proposals-20260922T133831Z-81c73fd8

No new experimental evidence. The imported proposal repeats a seed idea that the user already declined to admit in prior-work.md. If a zero-sum budget idea is revisited, the better candidate is `coverage-matched-slot-split`. It keeps each split slot's total unchanged, so it does not add oblique tokens to the split studies.

I did not admit `axial-t1pd-to-oblique-transfer`, and the queue is unchanged. The campaign's prior-work document records a user decision from 2026-09-22 not to admit this seed idea. The decision gives two reasons. First, the idea adds oblique tokens to the split studies, and extra oblique tokens there are what cost MCL and lateral meniscus against `k_05`. Second, it resembles the rejected `repeated_series_share` reallocation, which topped fold 0 and then lost on the public leaderboard. The proposal's own evidence is also weak. It relies on value estimates from frozen models, and dropping axial T1/PD entirely cost MCL and fracture about 0.008 on the confirm half. Those are the same labels this transfer would put at risk. The queue already holds the target of five ideas, so admitting it would also exceed that target. Its dependencies are satisfied, but the idea stays `proposed` and is not parked.

Unresolved concerns:
- The transfer shrinks axial T1/PD in 253 training studies and also changes validation selection. Earlier evidence showed that removing axial T1/PD cost about 0.008 on MCL and fracture. MCL is the label where the oblique budget already hurt.

Evidence:
- autoresearch/campaigns/oblique-x4/prior-work.md:3-9 (user decision: do not admit axial-t1pd-to-oblique-transfer)
- autoresearch/campaigns/oblique-x4/prior-work.md:72 (repeated_series_share topped fold 0, lost publicly)
- autoresearch/campaigns/oblique-x4/prior-work.md:94 (oblique budget cost MCL -0.058 and lateral meniscus on split studies)
- outputs/autoresearch/oblique-x4/imports/ideas-20260922T133831Z-fab196a9/report.json

## review-20260922T135223Z-60bb830c

Static review found no correctness defects. The oblique tag is a post-preparation dataclass replace, so cached and prepared inputs and the input identity are unchanged. view_group_bias has shape target_count × (slot_count + number of oblique-capable slots). For each label it shifts the within-study attention between view groups, including the split oblique groups, starting from zero. It adds no compute beyond a gather-add on the scores.

The candidate implements the assigned mechanism exactly. It changes only the declared config key, model.oblique.attention_bias, which is under the authorized model.oblique prefix. The sampler, budgets, labels and evaluation are untouched, so training and validation triplet counts cannot change, although accounting is not yet available to confirm this. The new parameter starts at zero. The oblique flag is attached after preparation, so preprocessing identity and cost are unchanged: preparation took 0.12 s. The flag rides through collate and the series-dropout masking correctly. The per-token view-group indexing matches the existing study-index expansion over spatial tokens. No leakage path found. Approving for launch. This is a re-run on the ×4 evaluation of the mechanism rejected at ×1 in the retired campaign (label-view-attention-bias-20260922T120824Z-29aa5863). Its adoption is decided solely by the ×4 rule.

Unresolved concerns:
- view_group_bias is 2-D, so the optimizer grouping (training/lightning.py:37-44) gives it head_lr and nonzero weight decay. Decay pulls the biases toward zero, which may damp the mechanism. That matches the retired campaign's version, so results are comparable, but it is not a tuned choice.
- The bias only matters for studies with more than one view group. For the 802 fold-0 studies without a split slot, any change comes from training divergence, not the mechanism. A mechanism claim must rest on the split-study deltas the completion is required to report.
- Realized triplet and visit accounting is not yet available (accounting_available=false). Count equality is expected from the code but still has to be confirmed by the supervisor's gate.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-x4/experiments/label-view-attention-bias-20260922T134812Z-7b4fa495/changes.diff
- experiment source src/knee_abnormality/modeling/model.py:155-177 (zero-init view_group_bias, oblique_view_group buffer)
- experiment source src/knee_abnormality/modeling/model.py:260-261 (bias added pre-softmax), 318-326 (group mapping via keep_series/acquisition_index), 405-407 (per-token expansion)
- experiment source src/knee_abnormality/data/dataset.py:292-308 (oblique tagging after unchanged budget split), 414/425/456 (series_oblique collate)
- experiment source src/knee_abnormality/inference/engine.py:95-119 (move_batch keeps non-whitelisted keys on CPU; model moves series_oblique itself)
- experiment source src/knee_abnormality/training/lightning.py:37-44 (optimizer grouping and weight decay for the new parameter)

## review-20260922T135920Z-2154256f

The mechanism is parameter-free and changes the forward pass only for studies that hold a split slot. At inference each split study's prediction is the probability average of a standard-only view and an oblique-only view of the split slot. Every other token is shared by both views. During training, the gradient on each view is weighted by that view's share of the mixture probability. The model can therefore specialize to whichever view explains a label, rather than diluting it in one shared softmax. This is the intended alternative to letting the oblique budget compete inside a single attention softmax. Non-split studies are bit-identical to the reference forward pass at the same weights. Any difference in their results comes from same-seed training divergence, so the mechanism claim must rest on the split-study deltas.

I approve this candidate. I read the full changes.diff and the relevant source. The split mask is correct. membership (B×N) @ (type one-hot × oblique) gives per-study, per-type presence of the valid oblique and standard tokens. split[token_study, token_type] then selects exactly the split slot's tokens. Pass A removes the split slot's oblique tokens and pass B removes its standard tokens. Tokens of other types stay in both passes, so neither pass can be empty. In non-split studies membership_a equals membership, and torch.where returns logits_a, which is the reference computation. The mixture logit(½σ(a)+½σ(b)) is computed stably with logaddexp/logsigmoid in fp32 outside autocast. Series dropout removes whole (study, type) slots, so a dropped split slot drops both groups together and the study falls back to the single pass. keep_series indexing of series_oblique matches series_type_index. The .to(device) avoids the MODEL_BATCH_KEYS defect in the non-editable engine.py. _attention_pool has only one caller (pool_encoded, model.py:411), so the signature change is safe. Validation and ×4 prediction use the same deterministic path, without seed dependence, labels or new preparation cost; recorded preparation time is 0.17 s. The only compute added is one extra pooling einsum.

Unresolved concerns:
- The contract requires new components to start neutral. This change has no parameters but is not neutral at initialization: split studies (about 12%, 103 on fold 0) get mixture predictions from the first step. That is inherent to pooling the split slots differently, which is in scope, but the change is not identity-at-init.
- Realized triplet and visit accounting is unavailable (accounting_available=false). Equality is expected because sampling and preparation order are unchanged, but the supervisor's zero-tolerance gate must confirm it.
- The mixture weight is fixed at ½. When one view carries no evidence for a label, averaging probabilities can pull a confident view's prediction toward the other, for example ACL evidence that only the oblique view shows. Per-label split-study deltas (ACL, MCL, lateral meniscus) should be read with this in mind.
- The developer's equivalence evidence is a synthetic 2-study CPU check, not a real batch. I confirmed the fallback by reading the source, but I did not re-execute it.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-x4/experiments/split-slot-view-mixture-pooling-20260922T135314Z-88614e3b/changes.diff
- candidate source src/knee_abnormality/modeling/model.py:221-311 (_pool_logits, split mask, logaddexp mixture, torch.where fallback)
- candidate source src/knee_abnormality/modeling/model.py:313-330 (series_oblique device move, keep_series indexing)
- candidate source src/knee_abnormality/data/dataset.py:289-311 (standard-then-oblique preparation order, replace(oblique=True)), 383-458 (collate series_oblique)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-x4/jobs/developer-20260922T135314Z-8159a669/report.json

## completion-20260922T142042Z-5eb3ee6e

At ×4 the per-label view-group bias repeats the retired ×1 pattern: a real gain on the split studies that the macro rule cannot adopt. Split-study analysis (103 studies, ×4 reference, localize_campaign.py adapted in /tmp):
- Split studies: +0.0035 macro (0.8634→0.8669), paired 95% [−0.0043, +0.0115]. Unchanged studies: +0.0001 (0.9068→0.9069).
- So almost all of the small macro gain sits in the studies where the mechanism acts, unlike the retired residual readout, whose gain came from unchanged studies.
- Per label on split studies (positives): MCL +0.038 (11), lateral meniscus +0.029 (22), medial meniscus +0.005 (48), ACL −0.011 (32). On unchanged studies all four stay within ±0.002. The bias recovers the MCL and lateral-meniscus loss that the oblique budget caused, but slightly hurts ACL on split studies. That fits routing ACL toward the oblique groups (acl-gated-oblique-tokens) rather than learning a free per-label bias.
- Macro per-label ×4 deltas (all 905 studies, exploratory): Baker's +0.006, MCL +0.005, lateral OA +0.004, medial OA +0.003, fracture −0.007, contusion −0.003, ACL −0.002.
- Selected epoch 8 (×1 0.8993). The learned bias table (12×9) has entries up to |0.18|, so weight decay did not collapse it, but it is small relative to the attention scores.
- The split-study gain is about equal to k_05 (candidate − k_05 +0.0003 [−0.013, +0.012]), but that k_05 comparison uses k_05's ×1 predictions.

This is a completion event. The fixed eligible_decision is reject: macro ×4 AUC was 0.902983 against 0.902373 (+0.00061, below the 0.001 minimum), and the 80% paired interval [−0.0011, +0.0023] includes 0. Accounting is complete and exactly matches the sealed counts (960,240 training triplets, 35,020 visits, 24,770 validation triplets per epoch, cohort hashes equal). Measured compute of 1,648 s is under the 1,902 s ceiling. The review concern that led to escalation (accounting unavailable) is therefore resolved. The run is valid and the reject stands.

The queue stays unchanged. acl-gated-oblique-tokens records label-view-attention-bias only as its parent, not as a dependency, and its prior evidence (+0.0020 on k_05's affected studies at ×4) does not depend on this run being adopted. oblique-identity-with-view-bias stays queued, because the embedding has not yet been tested at ×4. There are no proposals in this event.

Unresolved concerns:
- No k_05 prediction at ×4 and attention temperature 1 exists; the only ×4 k_05 predictions are at tau 0.75 (tilted-slot-policies-v2-k05). The contract-required comparison against k_05 was therefore made against ×1 predictions and is not like-for-like.
- Subgroup bootstraps cover only 103 studies with 11–32 positives for the key labels; the split-study per-label deltas are exploratory and unadjusted.
- The split-study bootstrap resamples studies (localize_campaign.py) rather than report groups as in the adoption bootstrap, so its interval is not directly comparable.

Evidence:
- outputs/autoresearch/oblique-x4/experiments/label-view-attention-bias-20260922T134812Z-7b4fa495/comparison/comparison.json
- outputs/autoresearch/oblique-x4/experiments/label-view-attention-bias-20260922T134812Z-7b4fa495/comparison/per_label.json
- outputs/autoresearch/oblique-x4/experiments/label-view-attention-bias-20260922T134812Z-7b4fa495/training/fold_0/triplet_accounting.json
- outputs/autoresearch/oblique-x4/experiments/label-view-attention-bias-20260922T134812Z-7b4fa495/training/fold_0/validation_predictions.csv vs outputs/training/oblique-x4/fold_0/validation_predictions.csv, split/unchanged via outputs/analysis/a/acquisition-geometry-20260922/localize_campaign.py (reference path swapped to oblique-x4)
- outputs/autoresearch/oblique-x4/experiments/label-view-attention-bias-20260922T134812Z-7b4fa495/training/fold_0/best-epoch=08-val_macro_auc=0.89930.ckpt (model.view_group_bias 12x9, max |0.18|)
- autoresearch/campaigns/oblique-x4/prior-work.md#results-of-the-retired-x1-campaign

## review-20260922T142739Z-4b354664

The combined candidate marks oblique-group tokens two ways. It adds a learned per-type embedding to those tokens before the MIL LayerNorm, and it adds a per-label, per-view-group bias to the attention scores (9 groups: 6 slots plus 3 oblique groups). Both are zero at start and only affect studies that hold a split slot. At the same weights, studies without a split slot see only the embedding, which is zero for them; the bias applies to them too, but only as a per-slot offset that the reference lacks. Any difference on those studies comes from same-seed training divergence or from the shared slot bias. So the mechanism claim rests on the split-study deltas for ACL, MCL and lateral meniscus, against both the reference and k_05.

I'm approving: the diff matches the declared plan, starts neutral, stays within the editable files and the model.oblique prefix, and the gate reported no error. The two config differences are exactly the two new flags. The seed is the incumbent's (2263738815). Added preparation cost is negligible: one dataclass replace per oblique series.

Unresolved concerns:
- Realized triplet and visit accounting is not available yet (accounting_available=false). Sampling code is untouched, so equality with the baseline replay is expected, but the zero-tolerance gate still has to confirm it.
- view_group_bias ([12x9]) and the oblique embedding are 2-D parameters, so lightning.py:43 applies weight decay to them at head_lr. That pulls them toward the neutral zero start and may shrink small learned offsets.
- The view bias also adds a learned per-label bias for each standard slot, and that part applies to every study, not just split ones. Deltas on non-split studies may therefore partly reflect this per-slot reweighting, not only training divergence.
- In the retired x1 campaign, each component on its own was rejected (oblique-group-embedding and label-view-attention-bias). This combination is new on x4 evaluation, so there is no additive evidence for it.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-x4/experiments/oblique-identity-with-view-bias-20260922T142228Z-7639b62b/changes.diff
- candidate source src/knee_abnormality/modeling/model.py _attention_pool (~lines 245-295): the view bias is added after the anatomical term and before masking and softmax
- candidate source src/knee_abnormality/data/dataset.py:290-310 (oblique flag set via replace), 248-275 (_merge_slot_groups unchanged)
- candidate source src/knee_abnormality/inference/engine.py:95-119: MODEL_BATCH_KEYS only controls device transfer; the model moves series_oblique itself
- candidate source src/knee_abnormality/training/lightning.py:41-44: new parameters go to head_lr, with weight decay on 2-D tensors
- candidate configs/training.yaml: three splittable slots (oblique_tilt_deg 30/20/20) give 9 view groups

## proposals-20260922T143719Z-ec2a9597

Verified in the baseline source that the 1D slice-depth code is acquisition-relative and one value per slice (model.py:334-338, 350). Oblique-group tokens in split slots are therefore coded in a frame tilted about 35–39° from their slot's standard series, with a median 11 mm origin offset on coronal slots (over 10 mm in 55–59%) and about ±23 mm of true-depth change across one oblique slice. That comes from researcher census depth_frame.txt over 558 training split slots. It is a measured misalignment in a code the model is known to rely on (removing the 1D code cost −0.0007 to −0.0030 macro). Whether it causes the split-study lateral meniscus (−0.029) and MCL (−0.020) deficits at ×4 τ0.75 is unproven, and those figures use τ 0.75, not the campaign's τ 1. New attribution: the reference's split-study ACL gain over k_05 sits in 47 coronal-T1/PD-only split studies. Their oblique groups are mainly a non-fat-suppressed Siemens t2_tse_bond T2 TSE protocol (the prior-work text calls it T1). Part of that site's weakness is a site effect that oblique routing cannot fix.

Admitted oblique-standard-frame-depth. It has no dependencies. It is in scope: it changes how oblique-group tokens are represented from their acquisition geometry, and it leaves sampling, triplets, validation selection and input identity alone. It is the only queued idea that corrects an existing code instead of adding identity markers or reweighting. I checked the premise in the source. In baseline model.py:334-338, position_1d is one sinusoidal value per slice, broadcast over all tokens at model.py:350. The per-token patient maps already exist at model.py:320-324. The census numbers in depth_frame.txt match the proposal: median 35–39° between the oblique and standard normals, a median 11 mm offset on coronal slots, about ±23 mm of depth change per ±40 mm in-plane, and no flipped normals.

I ranked it second, behind the reserved head and ahead of acl-gated-oblique-tokens and the plain embedding. It adds no parameters and has a built-in identity check: applied to the standard series, it reproduces today's code. It also targets the labels where the split-study loss is measured (lateral meniscus and MCL). The researcher's new evidence puts the reference's ACL gain in the coronal T1/PD oblique groups (mostly the t2_tse_bond protocol), which the ACL gate leaves ungated. So the gate is left with routing coronal-fluid obliques only, which is a narrower lever. The plain embedding stays last because the combination at the head already contains it and it was rejected on its own at ×1.

Unresolved concerns:
- The depth replacement is parameter-free but not identical to the reference at init on split studies: oblique tokens get new codes from step 0. The contract's neutral-start clause covers new parameters, so this follows the mixture-pooling precedent. The review should confirm that standard tokens and non-split studies are bit-identical at fixed weights.
- The review must check reflection handling. Reflection flips patient x; for coronal slots the normal is mostly AP, but the sign conventions of n_s·p against the augmented patient maps still need to be verified per plane.
- Anchor choice: the first standard uid in the source ranking must be fixed per study and must not depend on the training seed or epoch, so that validation stays deterministic.
- k_05 predictions at ×4 τ1 are still missing. All k_05 comparisons in this proposal use ×4 τ0.75, so they are not like-for-like with the campaign evaluation.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-x4/jobs/researcher-20260922T142228Z-40754723/report.json
- outputs/autoresearch/oblique-x4/jobs/researcher-20260922T142228Z-40754723/scratch/evidence/depth_frame.txt (angle 35–39°, center offset 5/10.7/11.1 mm, 23–25 mm shift per 40 mm)
- outputs/autoresearch/oblique-x4/jobs/researcher-20260922T142228Z-40754723/scratch/evidence/split_x4.txt, cor0_site.txt, cor0_descriptions.txt
- outputs/autoresearch/oblique-x4/baseline/source/src/knee_abnormality/modeling/model.py:320-338 (per-token patient maps; per-slice position_1d)
- outputs/autoresearch/oblique-x4/baseline/source/src/knee_abnormality/modeling/model.py:347-356 (position_1d broadcast into pooling tokens)
- outputs/analysis/encoding-usage-{graph-forced,token-graph,attention-prior}/results.json (1D-code ablation costs)

## completion-20260922T145003Z-02c790fe

Split-slot mixture pooling at ×4 is a null result overall. Macro delta +0.00005 [80% −0.0019, +0.0019]. On the 103 split studies it moved +0.0015 (0.8634→0.8649; my 1,000-draw study bootstrap gives 80% [−0.0039, +0.0076]). On the 802 unchanged studies it moved −0.0000 (80% [−0.0021, +0.0021]).

Per-label on split studies (positives in brackets):
- Gains: lateral meniscus +0.030 [22], contusion +0.024 [23], MCL +0.013 [11], fracture +0.018 [4].
- Losses: ACL −0.022 [32], medial meniscus −0.018 [48], synovitis −0.018 [17], effusion −0.009 [78].

The ACL loss fits the fixed ½ probability average. Evidence that only the oblique view shows, as dedicated ACL obliques would, gets diluted by the standard-only view. The MCL and lateral meniscus gains fit the retired campaign's finding that the oblique budget mainly hurt those labels. This suggests separating by label rather than by view, which is the acl-gated idea.

The overall per-label deltas that stand out come mostly from the unchanged studies, so they are same-seed divergence, not the mechanism: fracture −0.013 overall (−0.015 on 52 unchanged positives), PF OA +0.008 (+0.008 unchanged), Baker's +0.005, lateral OA +0.004.

Mean absolute probability change: 0.040 on split studies vs 0.034 on unchanged ones. So divergence moves the unchanged studies nearly as much as the mechanism moves the split ones. Comparisons against k_05 at ×4, temperature 1, are not possible: only the tau 0.75 predictions exist.

Completion of split-slot-view-mixture-pooling. I interpret the fixed decision, reject, as a valid null result. The run is valid. Accounting is complete and matches the sealed counts exactly: 960,240 training triplets, 35,020 visits, 24,770 validation triplets per epoch, with the cohort hashes matching. The only config difference is model.oblique.mixture_pooling. Measured compute was 1,655 s against a 1,902 s ceiling, with 4,380 steps. The review concerns that triggered escalation are resolved or immaterial. Accounting is now confirmed. Being non-neutral at init is inherent to changing how split slots are pooled, and it has no parameters. Validation stays deterministic and independent of the seed: the selection hash is identical in every epoch.

Macro: 0.902428 vs 0.902373, delta +0.00005, 80% interval [−0.0019, +0.0019]. By stratum (my CPU recomputation, details in finding): the split studies moved +0.0015 and the unchanged studies 0.0000. So the mechanism had a small, uncertain effect limited to the split studies, with a clear per-label pattern.

Ranking change: acl-gated-oblique-tokens moves ahead of oblique-standard-frame-depth. In this run, separating the standard view from the oblique view helped MCL, lateral meniscus and contusion on the split studies but cost ACL. That is the per-label split the ACL gate targets: obliques feed ACL, and the standard series alone serves the other labels. The prior k_05 ×4 analysis (+0.0020 on 102 studies) points the same way. The reserved head stays first.

Unresolved concerns:
- The split/unchanged strata and their bootstrap intervals are my own CPU recomputation (/tmp/strata_mix.py, which reuses the localize_campaign.py split-study definition). They are not a supervisor-sealed artifact, and compare_fold_runs does not emit this stratification although the contract requires it.
- There are no k_05 predictions at ×4, temperature 1 (only x4-tau0.75 in tilted-slot-policies-v2-k05-20260922). The contract's k_05 comparison is therefore missing for this and every later completion.
- Split-study per-label deltas rest on 4–48 positives from one seed and one fold. The ACL and MCL readings are hypotheses, not confirmed effects.

Evidence:
- outputs/autoresearch/oblique-x4/experiments/split-slot-view-mixture-pooling-20260922T135314Z-88614e3b/comparison/comparison.json (macro, per-label, prediction_change mean abs 0.0348)
- outputs/autoresearch/oblique-x4/experiments/split-slot-view-mixture-pooling-20260922T135314Z-88614e3b/training/fold_0/validation_predictions.csv vs outputs/training/oblique-x4/fold_0/validation_predictions.csv
- /tmp/strata_mix.py output: split 103 +0.0015 80%[−0.0039,+0.0076]; unchanged 802 −0.0000 80%[−0.0021,+0.0021]
- outputs/analysis/a/acquisition-geometry-20260922/localize_campaign.py (split-study definition)
- outputs/analysis/t/tilted-slot-policies-v2-k05-20260922/results.csv (ACL-only oblique vote +0.0020 on k_05 ×4, tau 0.75)
- outputs/autoresearch/oblique-x4/jobs/orchestrator-20260922T135920Z-8d9f97cc/report.json (implementation review whose concerns triggered escalation)

## review-20260922T145700Z-f8922cfe

The gate acts only on tokens where series_oblique & oblique_fluid_types[type] is true. Those are the tagged axial-fluid and coronal-fluid oblique groups in slots that hold both a standard and an oblique series. Studies without a split slot see exactly the reference pooling, so any difference there is same-seed training divergence. This makes the split versus unchanged comparison cleaner than for label-view-attention-bias, which also shifted the standard slots.

The developer reports a synthetic check in which ACL logits were bit-identical with and without the gate. That is claimed, not rerun here.

Approved for launch. The diff implements the registered mechanism: a fixed, parameter-free routing prior. Tokens from the fluid-sensitive oblique groups in split slots are masked out of every label's softmax except ACL. Nothing else changes.

- **Coverage:** sampling, budgets, triplet counts and visit counts are untouched.
- **Validation:** it stays deterministic and independent of the seed, because the oblique flag comes from the geometry split, which is identical in training and validation.
- **Leakage:** none. No labels or predictions are used.
- **Cost:** preparation and compute cost are negligible.

The run is not neutral at initialization. Non-ACL logits change in split studies from the first step. This is inherent to a hard routing prior with no parameters, and the queued idea asked for exactly that. The earlier non-neutral, parameter-free split-slot-view-mixture-pooling run was treated the same way and judged valid.

The idea said the setting under model.oblique would name the allowed labels. It is implemented as a boolean with ACL hardcoded. That is acceptable, since only this one routing is intended.

Unresolved concerns:
- The claim that no label loses all its tokens assumes the standard group always yields visually valid tokens. If standard preparation fails for a split slot and series dropout removes every other slot, all non-ACL scores for that study become -inf. The softmax then returns NaN and the run stops at the RuntimeError guard for non-finite logits (model.py). The failure is loud, not silent, but it would make the run invalid. It is rare but possible.
- Realized triplet and visit accounting is not available yet (accounting_available=false). Equality is expected because sampling.py is untouched, but the zero-tolerance gate still has to confirm it.
- k_05 has no ×4 predictions at attention temperature 1, so the required comparison of split-study results against k_05 is still not like-for-like. The precedent gain of +0.0020 on the affected studies came from k_05 at temperature 0.75, as an offline rescoring, not from training.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-x4/experiments/acl-gated-oblique-tokens-20260922T145144Z-ef189fa8/changes.diff
- candidate src/knee_abnormality/modeling/model.py:155-169 (buffers), ~249-254 (the -inf gate before membership), ~299-318 (per-token gate), ~397-399 (flattening in pool_encoded)
- candidate src/knee_abnormality/data/dataset.py:283-312 (tagging after preparation; budgets unchanged)
- candidate configs/training.yaml series_types: fluid_sensitive=1 with oblique_tilt_deg set only for axial fluid (30°) and coronal fluid (20°)
- src/knee_abnormality/inference/engine.py:95-110 (MODEL_BATCH_KEYS omits series_oblique; model.py moves it with .to(device))
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-x4/jobs/developer-20260922T145144Z-29af93bd/report.json

## proposals-20260922T151031Z-92e3cdec

New evidence from researcher-20260922T145144Z-bef067e9 (kind_net.txt, pair_decomposition.txt):
- Split studies affect macro AUC mainly through pairs with unchanged studies (su 5–22%, us 6–12% of each label's pairs). Pairs within split studies are only 1–3%.
- At ×4 τ0.75, reference minus k_05 by split kind (net su+us): Coronal0 −0.0001, Coronal1 −0.0001, axial fluid −0.0004. The axial-fluid term comes mostly from one MCL-positive study (ax1_detail.txt).
- Existing ×4 τ1 rescorings against the reference net at most +0.0003 per kind: view-bias gives Coronal0 +0.0003, Coronal1 −0.0002, axial fluid +0.0003; mixture pooling is about zero.
- Offsets, recalibration and restored slot shares therefore have a ceiling far below the 0.001 adoption bar. Candidates in this campaign need to improve how the oblique content is read.
- Completions should report the pair strata (ss/su/us/uu) with pair_decomposition.py alongside split and non-split macro deltas, because within-split AUC is a weak proxy.

The proposal's within-model check is useful and costs no training: re-predict the trained candidate with its new component neutralized. Only split studies change, which separates the mechanism from same-seed divergence. The same check applies to the ACL gate and to the view-bias combination.

oblique-standard-statistics-alignment passes admission and is new: it aligns oblique-group tokens to feature statistics from the same study's own standard series, and only the per-channel gate strength is learned. It stays proposed for now, with no queue change.

1. By the researcher's own pair decomposition (kind_net.txt), the split-vs-unchanged net for reference minus k_05 is at most ±0.0004 per split kind at ×4 τ0.75. The Coronal0 OA level shift that this mechanism would most directly repair cancels between the su and us terms (net −0.0001). A per-study AdaIN mainly removes level and scale offsets, so to reach +0.001 it must improve how the model reads oblique content, not fix the level. The proposal does not explain how a transform that preserves the moments would do that.
2. The researcher's own ranking depends on whether the running oblique-identity-with-view-bias candidate shows the per-type constant embedding repairing split-vs-unchanged pairs at ×4. That result is pending. Admitting now would schedule a run whose case depends on it.
3. Zero-initialized per-channel gates with 2-D weight decay stayed near zero in this repository before (label-private-crops FiLM, mean |scale| 0.02).

Reconsider it after the combination completes. If the embedding's split-study pair terms improve, admit it as the per-study extension.

Unresolved concerns:
- k_05 still has no ×4 τ1 predictions. The contract asks for split-study comparisons against k_05 at the evaluation setting, and that comparison is still unavailable, so the τ0.75 pair evidence is not like-for-like.
- Statistics alignment (if later admitted): oblique slabs cover about 45 mm, but the standard series covers the whole knee. Differences in spatial content across the field of view then enter μ_s and σ_s, so aligning to all standard tokens may distort lesion signal. The footprint-restricted reference option should be decided before implementation.
- The queued oblique-group-embedding alone overlaps the running identity+view-bias combination. Its value should be reassessed when the combination's split-study pair strata arrive.

Evidence:
- outputs/autoresearch/oblique-x4/jobs/researcher-20260922T145144Z-bef067e9/report.json
- outputs/autoresearch/oblique-x4/jobs/researcher-20260922T145144Z-bef067e9/scratch/evidence/kind_net.txt (net su+us by split kind: ref−k05 at x4 τ0.75 and x1; vbias and mix rescorings against the reference at x4 τ1)
- outputs/autoresearch/oblique-x4/jobs/researcher-20260922T145144Z-bef067e9/scratch/evidence/pair_decomposition.{py,txt}
- outputs/autoresearch/oblique-x4/jobs/researcher-20260922T145144Z-bef067e9/scratch/evidence/shift_by_kind.txt (Coronal0 OA and medial-meniscus logits 0.27 lower than k_05)
- outputs/autoresearch/oblique-x4/jobs/researcher-20260922T145144Z-bef067e9/scratch/evidence/ax1_detail.txt
- docs/validation/results/2026-09-16-label-private-crops-result.md (zero-init FiLM stayed near zero)

## completion-20260922T151919Z-f4520165

Split vs unchanged breakdown, computed here because the completion record lacks it (localize_campaign.py pointed at oblique-x4 x4 reference predictions; k_05 at x1): split studies (103) ref 0.8634, k_05 0.8667, candidate 0.8667; unchanged (802) ref 0.9068, k_05 0.9018, candidate 0.9055. Split-study per-label deltas vs reference (positives): ACL -0.024 (32), MCL +0.029 (11), lateral meniscus -0.003 (22), medial meniscus +0.010 (48); unchanged studies: ACL -0.008, MCL -0.006, lateral meniscus -0.008, medial meniscus -0.001. Overall per-label (exploratory, unadjusted 95%): ACL -0.0086 [-0.0164, -0.0012], lateral meniscus -0.0080 [-0.0164, -0.0008], fracture -0.0103 [-0.0268, +0.0021]; Baker's +0.0079, contusion +0.0066, OA and effusion +0.001 to +0.002. Interpretation: on x4, combining the embedding with the per-label view-group bias brings the split studies to k_05's level but not past it. The gain concentrates in MCL and medial meniscus, while ACL loses on split and unchanged studies alike, consistent with the learned bias drawing ACL attention away from sagittal fluid or oblique fluid. Unlike the x1 lead that ACL should rely on oblique fluid tokens, the learned bias did not improve ACL. Prediction change is moderate (mean abs prob diff 0.041; ACL Spearman 0.956, MCL 0.940). Limits: one seed and one fold; 103 split studies give wide intervals; k_05 is not scored like-for-like at x4.

Completion of oblique-identity-with-view-bias: the fixed eligible decision is reject and I interpret it as such. Macro AUC 0.901758 vs reference 0.902373 (-0.00062); adoption 80% interval [-0.00239, +0.00105]. The run is valid: training triplets 960,240, visits 35,020 and 4,380 steps match the sealed counts at zero tolerance; validation selection identity is constant across epochs; measured compute 1,652 s is under the 1,902 s ceiling. The earlier review concern about missing accounting is resolved.

The mechanism was partly visible and did not beat k_05. On the 103 split studies the candidate reached 0.8667 (+0.0033 vs the x4 reference, paired bootstrap 95% [-0.0045, +0.0118]), equal to k_05's 0.8667 (-0.0002 [-0.0123, +0.0117]); k_05 is scored from its x1 predictions. Unchanged studies lost -0.0014, from same-seed training divergence plus the view bias on standard slots, which applies to every study. Combining the two identity components did not add up to more than each one's x1 split-study repair (+0.0052 and +0.0068).

Queue unchanged (ranked_ids null). oblique-group-embedding alone remains untested on the x4 evaluation. This combined run cannot separate the embedding's contribution from the view bias, and the ACL loss points at the learned per-label slot bias more plausibly than at the embedding. No proposal is admitted in this completion event.

Unresolved concerns:
- The completion pipeline did not produce the contract-required split vs unchanged macro and per-label deltas against the reference and k_05; they were reconstructed here from saved predictions with an ad hoc copy of localize_campaign.py (/tmp/loc_x4.py), not a canonical artifact.
- k_05 still has no x4, temperature-1 predictions, so every comparison against k_05 uses its x1 predictions and is not like-for-like; since x1 understated the split studies relative to k_05, the gap may be misestimated.
- The components cannot be separated: the ACL and lateral-meniscus losses may come from the view-group bias on standard slots (which affects every study) rather than from the embedding, so the queued embedding-only idea is neither supported nor excluded by this run.
- Learned view_group_bias and embedding values were not inspected; weight decay at head_lr on these 2-D zero-initialized parameters (lightning.py:43) may have held them near neutral.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-x4/experiments/oblique-identity-with-view-bias-20260922T142228Z-7639b62b/comparison/comparison.json (macro, per-target, prediction_change; strata only by vendor, no split stratum)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-x4/experiments/oblique-identity-with-view-bias-20260922T142228Z-7639b62b/comparison/per_label.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-x4/experiments/oblique-identity-with-view-bias-20260922T142228Z-7639b62b/training/fold_0/validation_predictions.csv and triplet_accounting.json
- /home/claudio/src/knee-abnormality/outputs/training/oblique-x4/fold_0/validation_predictions.csv (x4 reference used for split breakdown)
- /home/claudio/src/knee-abnormality/outputs/analysis/a/acquisition-geometry-20260922/localize_campaign.py (split-study definition, 103 fold-0 studies; run with the reference path swapped to oblique-x4)
- /home/claudio/src/knee-abnormality/autoresearch/campaigns/oblique-x4/prior-work.md:82-108 (x1 component results: embedding +0.0052, view bias +0.0068 on split studies)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/oblique-x4/jobs/orchestrator-20260922T142739Z-37611fed/report.json (implementation review concerns)
