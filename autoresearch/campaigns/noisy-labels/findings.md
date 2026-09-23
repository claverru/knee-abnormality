# Campaign findings

## proposals-20260923T120045Z-69173341

Source check of the baseline, `src/knee_abnormality/training/lightning.py`:
- `training_step` (lines 327-343) calls `masked_binary_cross_entropy_with_logits` (lines 112-126). That function picks each cell's class-balance group with `targets >= 0.5` on whatever target tensor it receives. A target-correction idea that passes the blended t would therefore silently move cells between groups. The original y has to be passed separately for the weight lookup.
- `batch["study_uid"]` is present in the batch, so per-study memory is feasible.
- The realized counts are 35,020 visits over 10 epochs (about 3,502 per epoch) and 4,380 steps (8 studies per step). So the memory should cover the fold-0 training cohort, about 3,500 studies. The proposal's 4,407 rows is the wrong count, but the error is harmless.
- `fixed_class_balance` (lines 56-83) stores `positive_fraction` for each target. A step-dependent exponent can recompute rho from those stored fractions without a data change.
- The weights are frozen at `class_balance_exponent` (0.5 in `configs/training.yaml:111`). Under a schedule, the saved statistics and config must not keep claiming a fixed 0.5 that is no longer applied.

All five seed ideas are in scope: each is a training-objective change driven by the model's own training-time predictions, or a class-balance schedule shared across labels. None has dependencies, none duplicates another, and none repeats a mechanism the losses campaign measured. The active incumbent ideas list is empty, so admission passes.

Order:
1. Bootstrapped soft targets go first. They are the simplest online form of the only label intervention with a gold gain (+0.020) and a public tie. They need no per-study memory, and they give a baseline for the idea after them.
2. Temporal-ensemble targets come next. They answer whether an averaged prediction beats an instantaneous one, and they introduce the per-study memory that ideas 3 and 4 reuse.
3. Early-learning regularization third. It is the strongest published single-network method, but its lambda calibration is less concrete than the others' coefficients.
4. Confident-contradiction down-weighting fourth. It interacts with the rare-label class-balance weights and runs a cut-off rule on the same memory, so it is best read after ideas 2 and 3.
5. The relaxing class-balance schedule goes last. By the proposal's own prediction it loses on hard AUC, which is the adoption judge. Its main value is a hard-versus-soft AUC diagnostic under the judge caveat, not a likely adoption.

Unresolved concerns:
- Target-correction ideas (bootstrap, temporal ensemble): at review, check that the class-balance group comes from the original y and not the blended t. The current loss derives groups from whatever targets tensor it is given (lightning.py:122-124).
- Per-study memory ideas: the buffers must hold training studies only and must stay out of the EMA/SWA averaged model and the exported weights. Validation steps must never update them. The buffers must be restored on resume.
- Early-learning regularization: the rule 'lambda so the term is about 20% of the BCE at epoch 2 on the reference' cannot be computed from the reference, which has no memory. The developer must record a fixed numeric lambda and its derivation before launch, and log both terms per epoch.
- Relaxing schedule: it is expected to lose on the hard-AUC judge. Under the unchanged adoption rule, a gain that shows only in soft AUC is a finding, not an adoption. The saved class_balance record must reflect the schedule, not a fixed exponent.
- Instantaneous bootstrapping reads predictions made under per-visit triplet dropout and augmentation, so the correction signal is noisy and self-confirming. Its completion must report the per-epoch fraction of cells changed and the hard-versus-soft per-label deltas.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/baseline/source/src/knee_abnormality/training/lightning.py:112-126 (masked BCE; class-balance group chosen from the targets passed in)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/baseline/source/src/knee_abnormality/training/lightning.py:327-343 (training_step; batch carries study_uid)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/baseline/source/src/knee_abnormality/training/lightning.py:56-83 (fixed_class_balance stores positive_fraction, so rho can be recomputed for a schedule)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/baseline/source/configs/training.yaml:110-111 (class_balanced_bce true, exponent 0.5)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/imports/ideas-20260923T120045Z-6085b755/report.json (seed proposals)

## review-20260923T121216Z-2a573628

The change is a pure addition. From epoch 2 (0-indexed) to the end, every observed cell's BCE target becomes 0.8·y + 0.2·sigmoid(detached logit), with the class-balance group still taken from the raw label. The model's predictions come from train-mode forward passes on augmented data, which is the standard form of soft bootstrapping. When interpreting the result, note that label_noise_changed_fraction will read about 1.0 whenever the blend is active: almost every cell moves by more than 1e-6 because sigmoid never reaches the label exactly. That makes it a check that the blend is switched on, not a measure of selective correction. label_noise_target_shift, the mean |t−y| over observed cells, is the informative per-epoch quantity. It is capped at 0.2 times the gap between prediction and label. Both logs are epoch means of per-batch means weighted by batch size, not exact per-cell averages.

Approved. The candidate implements the soft bootstrap idea as specified. The self-target is detached, so no gradient flows through it. It uses only the current training batch's logits, so there is no per-study memory, no resume state, no validation input and no link to checkpoint selection. Unobserved cells, whose targets are filled with 0.0 in dataset.py:292-300, are still dropped by the observed mask before the mean. Validation, prediction, data loading, sampling, the optimizer and the scheduler are untouched, so triplet and visit counts cannot change. The added cost is one sigmoid and a few per-step logs, well inside the 115% runtime ceiling. The seed is the incumbent's (2263738815). The only config changes are the three trainer.label_noise keys. Static checks (ast parse, ruff, config load, import) are recorded in the developer report. I found no correctness, leakage or independence defect that needs a repair.

Unresolved concerns:
- label_noise_changed_fraction is almost always 1.0 once the blend is active, because the threshold is |t−y| > 1e-6, so it does not show how much the mechanism corrects. The completion report should use label_noise_target_shift as the amount of change, and could report the share of cells shifted by more than, say, 0.05 if that can be derived.
- Soft bootstrapping pulls confident wrong labels only 0.2 of the way toward the prediction, and also pulls correct labels toward early, uncertain predictions. The judge caveat applies: hard fold-0 AUC may fall even if the soft AUC improves. Record the two separately.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/bootstrapped-soft-targets-20260923T120951Z-db740f7e/changes.diff
- candidate source src/knee_abnormality/training/lightning.py:112-135 (weight_targets group, bootstrapped_soft_targets detach)
- candidate source src/knee_abnormality/training/lightning.py:336-381 (training_step blend gated by current_epoch >= start_epoch; logging)
- candidate source src/knee_abnormality/data/dataset.py:292-303 (unobserved targets filled with 0.0, masked by target_observed)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/jobs/developer-20260923T120951Z-f17ed57e/report.json

## review-20260923T121733Z-ae4ee9ed

Temporal-ensemble targets are implemented as a pure addition, and the class-balance weights are unchanged. The current prediction enters the memory before the blend, so about 0.09 of each target comes from the same step's own prediction, a mild self-anchoring. label_noise_changed_fraction counts the cells where the blend is active: 0 in epochs 0–1 and about 1.0 from epoch 2, so it only confirms the blend is on. label_noise_mean_abs_target_change (0.3·|memory−y|) is the informative per-epoch amount of correction. Both are epoch means of per-batch ratios weighted by batch size. The reports should compare hard and soft AUC separately, per the judge caveat.

The mechanism is on topic and inside the contract. It keeps a per-study EMA (momentum 0.7) of detached train-mode sigmoid predictions, built only from this run's training batches. From each cell's third observed visit onward, the BCE target becomes 0.7·y + 0.3·memory. One rule applies to every label. Class-balance grouping stays on the raw y, which makes it numerically identical to the parent: raw targets are soft and the parent grouped on targets ≥ 0.5. Validation targets, validation predictions and checkpoint selection never touch the mechanism, and only the allowed files and the trainer.label_noise prefix change. There are 3,502 training visits per epoch against 438 steps × 8, so each study is seen once per epoch and the blend starts in the third epoch (index 2). The per-row Python loop over 8 rows adds negligible cost.

Blocking defect: resume breaks. The contract requires the per-study memory to be restored on resume. on_save_checkpoint writes the memory and visits to CPU, and on_load_checkpoint restores them as CPU clones. On a resumed fit, `_label_noise_targets` then calls torch.where on CPU 1-D `memory`/`visits` with CUDA `prediction_row`/`row_observed`, which raises a device-mismatch RuntimeError. Resume is a supported path (training/checkpoints.py validate_resume_checkpoint, trainer.ckpt_path). An uninterrupted run is unaffected, but the supervisor may resume one.

Repair requested: in `_label_noise_targets`, move the fetched memory to `device` with dtype float32 and the visits to `device` with dtype int64 before use. Equivalently, restore them to the module's device in on_load_checkpoint. No other change.

Unresolved concerns:
- Resume crashes: memory and visits restored from the checkpoint stay on CPU and are mixed with CUDA tensors in torch.where inside _label_noise_targets. Repair requested.
- label_noise_changed_fraction is a switch indicator, 0 before the third visit and about 1 afterwards, not a measure of selective correction. Use mean_abs_target_change as the amount changed.
- The memory holds train-mode, augmented predictions and includes the current step's prediction, so it anchors partly to itself. Fold-0 hard AUC may fall even if soft AUC improves. Record the two separately.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/temporal-ensemble-targets-20260923T121257Z-3e6e8fc9/changes.diff (on_load_checkpoint clones without a device move; _label_noise_targets calls torch.where on memory/visits with CUDA prediction_row)
- source/src/knee_abnormality/training/checkpoints.py:95-115 (resume is supported via trainer.ckpt_path)
- source/src/knee_abnormality/data/dataset.py:306,395 (study_uid batch key)
- limits: 35020 visits / 4380 steps / batch 8 imply one visit per study per epoch

## review-20260923T121939Z-5f341ab3

The repaired temporal-ensemble candidate is behaviorally identical to the previous attempt on an uninterrupted run; the fix only affects resumed runs. Target from the third visit (epoch index 2) onward: 0.7·y + 0.3·EMA(momentum 0.7) of detached train-mode sigmoid predictions. The EMA includes the current step's prediction, so about 0.09 of each target is self-anchored. For the completion report: label_noise_changed_fraction only shows whether the blend is on (0 in epochs 0–1, about 1 afterwards), so label_noise_mean_abs_target_change is the measure of how much targets moved. Report hard and soft AUC and their per-label deltas separately, per the judge caveat.

The repair addresses the only blocking defect from the previous review, the resume device mismatch, with the minimal requested change. Rechecking the rest of the diff found no new problems. The mechanism is on topic: a target correction driven only by the model's own predictions on training batches. It applies one rule and one set of coefficients to all labels and all observed cells, and it leaves the class-balance weights unchanged. The per-study memory is filled only from this run's training batches and is now restored correctly on resume. Visits, triplets and steps are untouched because sampling is unchanged, and the added cost is a Python loop over 8 rows per step, far inside the runtime ceiling. Approved for launch.

Unresolved concerns:
- label_noise_changed_fraction only shows whether the blend is on, not how selectively it corrects; the completion report should use label_noise_mean_abs_target_change to measure the amount of change.
- The memory is built from augmented train-mode predictions and includes the current step, so the target partly anchors to the model itself. Hard fold-0 AUC may fall while soft AUC rises; record them separately.
- Resume restoration is fixed by inspection only, not exercised; an uninterrupted run does not test it.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/temporal-ensemble-targets-20260923T121814Z-74391894/changes.diff (else branch moving memory and visits to the device; weight_targets=raw_target)
- diff against …/experiments/temporal-ensemble-targets-20260923T121257Z-3e6e8fc9/changes.diff: only the 3-line device-move block and hunk offsets differ
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/jobs/orchestrator-20260923T121733Z-1fdca638/report.json (repair request)
- source/src/knee_abnormality/inference/checkpoint.py:16 (torch.load map_location=cpu, weights_only=False)

## completion-20260923T124015Z-4f635eb6

Uniform soft bootstrapping (β=0.8, from epoch index 2) is neutral to slightly harmful on every judge.

Macro hard AUC changed −0.00046 (80% CI −0.00144 to +0.00058) and macro soft AUC changed −0.00032. Predictions barely changed: the mean absolute probability difference is 0.018, and per-label Spearman correlation is 0.978–0.996.

Per-label hard deltas: Lateral OA −0.0038 (95% CI −0.0078 to +0.0001), Synovitis −0.0020, MCL −0.0017, Contusion −0.0010, and gains on Fracture +0.0011, Medial OA +0.0010 and Effusion +0.0007.

Per-label soft deltas: MCL −0.0021, Lateral OA −0.0017, PF OA −0.0014, and gains on Contusion +0.0013 and Medial OA +0.0011. The OA family fell on both judges: hard −0.0014, with a 95% CI of −0.0039 to +0.0013.

Realized change per epoch (mean |t−y| over observed cells):
- epochs 0–1: 0
- epochs 2–9: 0.040, 0.038, 0.036, 0.034, 0.033, 0.031, 0.030, 0.029

The blend moved every cell, not selected ones, so its correction was spread across all cells rather than aimed at the wrong labels. The selected checkpoint is the same epoch as the reference's (file epoch=08, reported as epoch 9), with a ×1 score of 0.89962 against 0.89942.

Interpretation: a uniform 20% pull toward the model's own augmented train-mode predictions acts like mild label smoothing. It neither corrects enough noisy cells nor hurts the ranking much. This weakens, but does not rule out, other uniform blend mechanisms such as temporal-ensemble targets (0.3 weight on a per-study EMA). Selective mechanisms (confident-contradiction down-weighting) and schedule mechanisms are unaffected by this evidence.

The run is valid and the fixed verdict is reject. It used seed 2263738815. Training triplets (937,760), visits (35,020) and ×1 validation triplets (24,253) match the sealed counts exactly. It ran 4,380 optimizer steps; setup, training and ×4 prediction took about 1,620 s, under the 1,846 s ceiling. The only config differences are the three trainer.label_noise keys.

The diff matches the plan approved at review. From epoch index 2 onward, every observed cell's target is 0.8·y + 0.2·sigmoid(detached logit). The class-balance group still comes from the raw label. Validation data and checkpoint selection are never used by the mechanism.

The unresolved concern from review is now answered from the logs. label_noise_changed_fraction is 0 in epochs 0–1 and about 1.0 from epoch 2, so it only shows the blend switching on. label_noise_target_shift is the measure of change: the mean |t−y| falls from 0.040 in epoch 2 to 0.029 in epoch 9. Because the blend weight is 0.2, this implies a mean |p−y| on training cells of about 0.20 falling to about 0.15.

The judge caveat does not change the reading here: the result is not a loss on hard AUC with a gain on soft AUC. Hard macro AUC moved −0.00046, from 0.90276 to 0.90230. The 80% interval is [−0.00144, +0.00058] and the 95% interval is [−0.00190, +0.00116]. Soft macro AUC moved −0.00032, from 0.79846 to 0.79814. Soft-target BCE rose by 0.00076, and gold-subset macro AUC fell from 0.9236 to 0.9177, though that subset has only 13 studies.

The queue is unchanged. The remaining ideas are mechanistically different or already prepared.

Unresolved concerns:
- Gold-subset macro AUC fell by 0.006, but it covers only 13 fold-0 studies; that sample is too small to treat this as a signal about true-label agreement.
- Train loss is not comparable with the reference's from epoch 2 onward, because BCE against soft targets has a nonzero entropy floor.
- The result comes from one seed and one fold. A delta of −0.0005 is within the usual seed-to-seed spread, so this run cannot separate a small harm from no effect.

Evidence:
- outputs/autoresearch/noisy-labels/experiments/bootstrapped-soft-targets-20260923T120951Z-db740f7e/comparison/comparison.json (runs.*.metrics soft_macro_auc, per_target_soft_auc, gold; family_means; prediction_change; paired_report_group_bootstrap)
- outputs/autoresearch/noisy-labels/experiments/bootstrapped-soft-targets-20260923T120951Z-db740f7e/training/fold_0/logs/version_0/metrics.csv (label_noise_target_shift and label_noise_changed_fraction per epoch)
- outputs/autoresearch/noisy-labels/experiments/bootstrapped-soft-targets-20260923T120951Z-db740f7e/changes.diff (lightning.py bootstrapped_soft_targets, training_step blend, weight_targets)
- outputs/autoresearch/noisy-labels/experiments/bootstrapped-soft-targets-20260923T120951Z-db740f7e/training/fold_0/selected_checkpoint.json
- outputs/autoresearch/noisy-labels/experiments/bootstrapped-soft-targets-20260923T120951Z-db740f7e/comparison/per_label.json
- outputs/autoresearch/noisy-labels/jobs/orchestrator-20260923T121216Z-98a3bc6b/report.json (implementation review concerns)

## review-20260923T124652Z-996eeb40

The ELR candidate's objective is correct in sign and form, and its memory is filled only from training batches. Two differences from the published ELR:
- The memory holds the previous EMA, and the current step's prediction is folded in only after use.
- On a study's first visit the prior is its own detached prediction. The epoch-0 term is therefore log(2p(1−p)), which pushes each cell away from 0.5 toward whichever side it is already on (entropy minimisation) and has no label-correction role.

λ 0.075 comes from the developer's analytic estimate (about 20% of BCE at epoch 2) and was not measured. The logged train_bce_loss and train_label_noise_loss will show the realized ratio. The current changed-fraction metric counts memory–prediction drift, not label correction, so a repair is requested.

The diff is on topic and stays inside the policy files and the `trainer.label_noise` prefix. It is a binary-sigmoid form of ELR (Liu et al. 2020): the loss adds λ·mean over observed cells of log(1 − (m·p + (1−m)(1−p))). The memory m is a detached per-study EMA with β 0.7, filled only from training batches. One rule and one coefficient apply to every label. Validation targets, validation predictions and checkpoint selection are not touched, and data, sampling and counts are unchanged, so I see no leakage. The added cost is a Python loop over the batch, which is negligible. The integration is purely additive and there are no accepted ideas to remove, so it is consistent.

It has the same resume defect that got the earlier temporal-ensemble attempt rejected. `on_save_checkpoint` stores the memory on the CPU (lightning.py:298-302), and `on_load_checkpoint` clones those tensors without moving them to the device (lightning.py:278-284). After a resume, `_label_noise_regularizer` mixes CPU priors with GPU predictions. `settings.momentum * prior + (1 - momentum) * prediction` and `torch.stack(memory_rows)` (lightning.py:359-365) would then fail with a device error on the first batch. That happens for studies seen before the interruption, and the stack fails for any batch that also contains a new study.

Requested repair, one bounded change:
- (a) Move the restored memory to the prediction's device and dtype before use, as the temporal-ensemble repair did.
- (b) Make `label_noise_changed_fraction` measure what the regularizer actually changes. Count observed cells where the ELR gradient opposes the BCE gradient, i.e. sign(m − p) ≠ sign(y − p), which is where the memory pulls against the label. This replaces the current |m − p| > 0.05 threshold, which only counts disagreement between the memory and the live prediction. It is the per-epoch fraction the contract requires. Optionally also log the mean |m − y| on observed cells.

Keep momentum 0.7 and weight 0.075 unchanged.

Unresolved concerns:
- Resume defect: memory is restored as CPU tensors (lightning.py:278-284) and mixed with GPU predictions (lightning.py:359-365), which raises a device error after any resumed run.
- label_noise_changed_fraction (|m−p|>0.05) measures drift between the memory and the live prediction, not the cells whose label pull ELR opposes. It does not satisfy the contract's diagnostic as intended.
- First-visit self-prior makes the epoch-0 term an entropy-minimisation push, not label correction. This may sharpen early overfitting. Worth noting when interpreting the result; no change requested.
- λ 0.075 is an unmeasured analytic calibration. Check the realized train_label_noise_loss/train_bce_loss ratio in the completion report.
- Judge caveat: ELR may lower hard fold-0 AUC while raising soft AUC; report both, with per-label deltas.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/early-learning-regularization-20260923T124105Z-5579c9a9/changes.diff
- experiments/early-learning-regularization-20260923T124105Z-5579c9a9/source/src/knee_abnormality/training/lightning.py:276-302 (memory save/load, no device restore)
- experiments/early-learning-regularization-20260923T124105Z-5579c9a9/source/src/knee_abnormality/training/lightning.py:354-375 (_label_noise_regularizer, changed-fraction definition)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/jobs/developer-20260923T124106Z-82bd6f4e/report.json (λ analytic calibration, static checks only)
- Active experiment temporal-ensemble-targets-20260923T121814Z-74391894 (repair_of the earlier attempt rejected for the same resume device bug)

## review-20260923T124953Z-2a94d095

The repaired ELR candidate trains identically to the first attempt on an uninterrupted run; the fix only matters on resume. Unobserved cells can carry any target value, but they are excluded from the changed fraction, the gap and the loss mean by `observed` masking.

The new changed fraction measures where the memory pulls against the label, not how strongly. Read it together with train_label_noise_loss/train_bce_loss, and with the memory–target gap, to judge how much correction happened. In epoch 0 each study's first-visit prior is its own prediction, so that epoch's term acts as entropy minimisation; expect that epoch's changed fraction to reflect this rather than label correction.

I compared the repaired diff with the rejected attempt. The only changes are the requested ones. First, the resume fix: the checkpoint stores CPU tensors, and each prior is moved to the live prediction's device and dtype before it is mixed into the memory and stacked. Second, the diagnostic now counts cells where the ELR pull opposes the BCE pull, masked to observed cells. It also logs the memory–target gap.

Coefficients, loss form and memory update are unchanged from the first review:
- The memory is filled only from training batches and is detached.
- Validation predictions, validation targets and checkpoint selection are never read.
- Data, sampling and counts are unchanged, and the added cost is a small per-batch Python loop, so I see no leakage and no change to coverage.
- One rule and one coefficient pair apply to every label and cell.

Both files parse (AST check; py_compile was blocked by the read-only source). No gate error was recorded. Approve.

Unresolved concerns:
- The resume fix was checked by reading the code only; an uninterrupted run never exercises it.
- λ 0.075 is an analytic estimate; the completion report should give the realized train_label_noise_loss/train_bce_loss ratio.
- A study that appears twice in one batch would update its memory twice with the same prior; this should be rare and small, but it was not checked.
- Judge caveat: ELR may lower hard fold-0 AUC while raising soft AUC. Report both, with per-label deltas and the per-epoch changed fraction.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/early-learning-regularization-20260923T124737Z-7aba30fc/changes.diff
- experiments/early-learning-regularization-20260923T124737Z-7aba30fc/source/src/knee_abnormality/training/lightning.py (_label_noise_regularizer: device restore, opposes_label and memory_target_gap accounting)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/jobs/orchestrator-20260923T124652Z-9f9009ae/report.json (requested repair)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/jobs/developer-20260923T124737Z-64a3c04a/report.json

## proposals-20260923T130133Z-1538968b

New research evidence from the researcher's gold checks (58 gold studies; an out-of-fold ReXNet+B0 image teacher, not the model's own memory during training):

- **Uniform blending is the best target correction tested.** A uniform blend of the teacher into the labels raises target–gold macro AUC from 0.9075 to 0.9230, 0.9259, 0.9275 and 0.9228 at w=0.1, 0.3, 0.5 and 0.8. Blending at w=1e-4 alone gives 0.9152, so about 40% of the gain is tie-breaking.
- **Class-centering gives up some gain.** It changes targets 27% less, but reaches 0.9230, 0.9241 and 0.9222 at w=0.1, 0.3 and 0.5, never above uniform.
- **Gating and replacement do not help.** The label-entropy gate (0.9240), logit fusion (0.9261) and the teacher-confidence gate (0.9158) do not beat uniform. Replacing the top 5–30% of |m−y| cells with the teacher prediction lowers gold AUC (0.898–0.911, precision 0.31–0.43).
- **The contradiction rule's proxy is skewed toward positives, matching the gold errors.** On fold 0, the queued rule's selection proxy (reported by the researcher) picks 6.6% of cells: 15% of y≥0.5 cells and 4% of y<0.5 cells, with gold precision 0.44. The gold check itself gives 6.2%, precision 0.44 and a |m−y| detection AUC of 0.694. The gold errors are asymmetric in the same direction: 26% for y≥0.5 against 9.6% for y<0.5.
- **Errors do not cluster by study** (variance 1.45 against 1.55 under permutation, p=0.66), so study-level reliability weighting is unsupported.

These are target-quality measurements on 58 studies. They do not predict fold-0 hard or soft AUC after training.

I did not admit `class-centered-self-correction`, so it stays `proposed`. It passes `proposal_admission` because it declares no dependencies, but it is weak for this campaign's decision rule, and its design depends on a run that has not finished.

1. **Low chance of adoption, by its own account.** The proposal expects hard AUC to move by no more than ±0.001 and soft AUC to stay near 0. It says the within-class reordering is invisible to the report-derived judges. The adoption rule requires a gain of at least 0.001 with the 80% lower bound above 0. An idea whose predicted signature is "neutral" can at best be discarded, so it would spend a queue slot without giving a real chance of adoption.
2. **The gold evidence favours the alternative it replaces.** On 58 gold studies, centering scores 0.9241 at w=0.3, below the uniform blend (0.9259 at 0.3 and 0.9275 at 0.5). Centering also peaks at 0.3 and falls by 0.5. It is supported only relative to raw labels (0.9075), and a uniform blend already gets that gain.
3. **Its implementation plan depends on the running TE result.** The plan has three branches for TE: replace its blend, stack alongside it, or build a new memory. It is also meant to attribute the TE and ELR outcomes, and neither has completed. The proposal itself recommends waiting for TE to complete. Once TE and ELR report per-label hard and soft deltas, it can be admitted in the next proposals or ranking event if TE shows class-level compression hurting LOW-heavy labels.

The live queue order is unchanged.

Unresolved concerns:
- The gold checks score the corrected targets against gold, with an external out-of-fold teacher. The in-training EMA memory differs: it comes from augmented train-mode predictions and is partly self-anchored. The ordering between mechanisms may not carry over to trained-model AUC.
- The gold set has 58 studies and no intervals were reported, so differences near 0.002 between uniform and centered blends are within plausible noise.
- The contradiction-downweighting proxy has gold precision 0.44, so more than half of the down-weighted cells would be correct labels. Its completion report should give the realized per-epoch selected fraction, split by y≥0.5 and y<0.5.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/jobs/researcher-20260923T124106Z-48cb4f6b/report.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/jobs/researcher-20260923T124106Z-48cb4f6b/scratch/gold_teacher_checks.txt (uniform vs class-centred vs gated vs top-k replacement; contradiction-rule precision; per-study error clustering)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/jobs/researcher-20260923T124106Z-48cb4f6b/scratch/gold_teacher_checks.py

## completion-20260923T130840Z-967a7c32

Temporal-ensemble targets (0.7·y + 0.3·EMA with momentum 0.7, starting at each study's third visit, current step included): the model selected at epoch 10 (1-indexed) scored hard macro 0.90301 against 0.90276 (Δ +0.00025; 80% interval [−0.0010, +0.0016]). Soft macro was 0.79806 against 0.79846 (Δ −0.0004).

Per-label hard deltas: Lateral OA −0.0067 (95% interval excludes 0), Synovitis −0.0046 (excludes 0), ACL −0.0031. Baker's +0.0061, Fracture +0.0042, Lateral Meniscus +0.0023. The OA family group fell by −0.0036 (95% interval excludes 0), and meniscus rose by +0.0014.

Per-label soft deltas were smaller but mostly had the same sign: ACL and Lateral OA −0.0042 each, Medial OA −0.0019, PF OA −0.0012; Baker's +0.0026, Effusion and Fracture +0.0020. Soft and hard agree on the OA and ACL losses, so the self-blend damages the OA family on both judges. It does not correct report errors that the hard judge penalizes.

The realized change fraction was 0 in epochs 0–1 and 1.0 from epoch 2 on, meaning the blend reached every observed cell. The mean |target change| decayed from 0.072 at epoch 2 to 0.049 at epoch 9, which is mild, uniform smoothing that fades as memory and predictions converge. Predictions barely moved against the reference: mean |Δp| 0.029, per-label Spearman 0.97–0.99. Compute was 1655 s against a 1846 s ceiling, and the counts matched exactly.

Together with the rejected bootstrapped-targets run (hard −0.00046, soft −0.00032, OA down), uniform blending of the model's own predictions into targets looks neutral to harmful here.

The fixed verdict is reject: macro Δ +0.00025, below the 0.001 minimum, with an 80% lower bound of −0.0010. The judge caveat gives no reason to read it differently. Soft macro AUC fell by 0.0004 (0.79846 → 0.79806), and soft-target BCE got worse by 0.0009. The candidate lost slightly on the soft judge as well, so this is not a "hard loses, soft gains" case. The 13 gold studies in fold 0 moved by +0.0011 (0.9236 → 0.9247), which is too few studies to count as evidence.

`class-centered-self-correction` stays proposed. This result partly meets the earlier admission condition. Uniform self-blending has now hurt the OA family twice: bootstrapped targets did, and this run did, with the OA family group Δ −0.0036 and a 95% interval of [−0.0070, −0.0004]. But the proposal's own prediction is still a neutral effect on the report-derived judge, which leaves it little chance of adoption under the shared rule. The ELR run's results should also be in before deciding whether class-level compression is the common failure. Queue order and membership are unchanged.

Unresolved concerns:
- Resume restoration of the memory was never exercised; this run was uninterrupted, so the repair is verified by inspection only.
- Only one seed and one fold: the Lateral OA and Synovitis per-label intervals are unadjusted, and the OA-family loss is diagnostic, not confirmed.
- The gold-subset gain (+0.0011) rests on 13 fold-0 studies and cannot separate label-correction benefit from noise.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/temporal-ensemble-targets-20260923T121814Z-74391894/comparison/comparison.json (runs.*.metrics.soft_macro_auc, per_target_soft_auc, gold; paired_report_group_bootstrap.intervals.oa; prediction_change)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/temporal-ensemble-targets-20260923T121814Z-74391894/training/fold_0/logs/version_0/metrics.csv (label_noise_changed_fraction, label_noise_mean_abs_target_change per epoch)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/temporal-ensemble-targets-20260923T121814Z-74391894/comparison/per_label.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/state.json (class-centered-self-correction proposal and prior non-admission rationale)

## review-20260923T131427Z-c855fe68

The candidate matches the persistent-disagreement specification except for one normalization detail. The batch rescale divides by the mean contradiction factor over observed cells, not by the mean combined cell weight. When contradicted cells are rare-label positives with large class-balance weights, the effective loss scale falls a little, and weight shifts toward the other cells in the same batch, including other labels. Down-weighting can only start once a study has 3 earlier visits, which is about epoch 3 at one visit per study per epoch. Because of the 0.5-side and >0.5-distance test, soft labels near 0.5 can trigger only against a very confident memory, and a 0.5 label never triggers. The completion should report the per-epoch contradicted fraction by label and polarity alongside the hard and soft AUC deltas.

Approved after reading changes.diff and the relevant source, not just the developer's summary.

What the code does:
- **Memory and trigger:** the weight for each visit uses only the memory from earlier visits. The current prediction updates the memory afterwards, so a step never judges itself. Each study gets about one visit per epoch (35,020 visits over 10 epochs), so down-weighting can start from epoch 3. The reference selected epoch 8, so the mechanism is active well before selection.
- **Contradiction rule:** a cell counts only if the memory and the label fall on opposite sides of 0.5 and differ by more than 0.5. For soft labels this is stricter: a 0.4 label needs a memory above 0.9. A label of exactly 0.5 can never trigger.
- **Masking:** unobserved cells have target 0.0 and are masked out of the weights, the statistics and the loss.
- **Gradients and scope:** the weights are built from detached predictions and carry no gradient. The rule is the same for every cell, with no label, side or gold branching.
- **Data use:** the memory is filled only from this run's training batches. Validation targets and predictions are never read.
- **Resume and cost:** memory is stored as CPU tensors and moved to the batch device at use, so resume is safe. The only added cost is 8 small CPU copies per step, which is negligible against the 15% ceiling.
- **Diagnostics:** the per-epoch contradicted fraction, overall and per label and polarity, is the realized changed fraction the contract asks for.

One interpretation difference from the idea text: the rescaling keeps the observed mean of the contradiction factor alone at 1. It does not keep the mean of the combined class-balance × contradiction weight. I am not requesting a repair, because at the expected contradicted fractions the drop in effective weight is about a few percent. It is recorded as a concern for interpreting the result.

Unresolved concerns:
- The rescale keeps the observed mean of the contradiction factor at 1, not the combined class-balance × contradiction mean that the idea text arguably specifies. Contradicted rare positives therefore lower the effective loss scale slightly. Check the realized contradicted fraction for MCL and Fracture positives when reading the result.
- Checkpoint save/load of the memory and resume were checked by reading the code only; no real save and reload was run.
- If a study appears twice in one batch, its memory updates twice and its visit count rises by 2 in that step. Probably rare with batch size 8 and one visit per epoch; not verified.
- Judge caveat: correct down-weighting of wrong report labels may lower hard fold-0 AUC while raising soft AUC. The completion must report both, with per-label deltas.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/confident-contradiction-downweighting-20260923T130931Z-120e2541/changes.diff
- candidate source src/knee_abnormality/training/lightning.py contradiction_cell_weights (diff hunk @@ -123,7 +125,50): prior-memory-only weighting, observed-mean rescale
- candidate source src/knee_abnormality/data/dataset.py:292-313: unobserved targets set to 0.0 and masked via target_observed; study_uid collated at :395
- /home/claudio/src/knee-abnormality/autoresearch/campaigns/noisy-labels/ideas.yaml:213-250 (idea mechanism and normalization text)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/jobs/developer-20260923T130931Z-e72bf4de/report.json

## proposals-20260923T132846Z-dee2533f

New analysis from the researcher, all from existing artifacts on CPU:
- **Self-targets.** One-sided self-targets do not beat the two-sided blend on gold. Downward-only at w 0.3 trails it by −0.0041 (80% interval [−0.0085, +0.0003]).
- **Cross-label coupling.** Self-targets raise it. Mean |corr| of logits across label pairs went from 0.359 to 0.384 under TE. Lateral OA's AUC within studies negative for Medial OA fell from 0.808 to 0.792. Part of the coupling is real: report negatives that have a same-family positive are 13.6% gold-positive, against 5.9% without one. Family-aware decoupling is therefore unsupported, and it would also be a per-label mechanism.
- **Gold studies inside training.** They are trained on official labels, so in-training memory looks inflated on them (TE memory 0.962 against 0.913 for the labels). Any error-rate analysis of training memory must exclude them.
- **Under-fitting.** Positives' mean memory is 0.52–0.75 across labels against targets of about 0.93, so the model stays under-fitted. The researcher found no memorization phase for early-snapshot anchors to exploit.

Directions pulling the same lever as CCD (ITLM truncation, LL-R, mixture-posterior weights, forgetting counts, trust-gated class balance) wait for CCD's result.

I admitted persistent-excess-residual-absorber. It has no dependencies, and proposal_admission says it is eligible. It stays in scope: it changes only the training objective, it is driven by the model's own training-time predictions, and it uses one step size, initialization, cap and momentum for every label and cell. The per-(label, side) ρ is a running statistic, like the class-balance fractions, not a per-label coefficient. The researcher's CPU sizing supports the claim that it reaches a different set of cells from CCD. Measured on TE's memory, group-centred excess above 0.4 covers 1.4% of cells and is spread across labels: 0.5% of MCL positives and 1.3% of Fracture positives (scratch/absorber_sizing.txt). The fixed |m−y|>0.5 rule instead reaches about 43% of MCL positives and 38% of Fracture positives below m 0.5. That makes it a real alternative trust lever, and it avoids the known Fracture/MCL hard-AUC sensitivity.

I ranked it after the class-balance schedule rather than beside CCD. CCD's result, and ELR's, will show whether down-weighting cells the model persistently contradicts helps on either judge. The absorber is a refinement of that same lever, so its prior depends heavily on those outcomes. The schedule is an independent lever that measured evidence supports: in the public class-balance curve, the score moved with the exponent. The absorber's declared plan is to replace CCD's weight factor if CCD is accepted, not to stack on it. That matches the one-trust-rule principle, and it must set removed_idea_ids accordingly at integration.

class-centered-self-correction stays proposed. Its parent is temporal-ensemble targets, which was rejected, and uniform self-targets have now failed twice (bootstrap −0.00046, TE +0.00025). The researcher also measured higher cross-label coupling under TE: mean |corr| rose from 0.359 to 0.384. Class-centring does not obviously undo that coupling, so it is not worth a queue slot yet.

Unresolved concerns:
- Absorber side rule: s comes from y≥0.5, so soft targets in 0.3–0.5 can only be absorbed downward. That is the band where gold shows about a third positive. The developer must log how often those cells are absorbed.
- ρ is computed per (label, side), and the offset is measured against it. It is admissible as a running statistic, but the review must confirm ρ has no tunable per-label term and is restored on resume.
- Gold training studies go through the absorber too, and their labels are official. Reported absorbed fractions must separate gold from non-gold cells so they are not read as report-error detection.
- The completion reports for ELR and CCD should include the partner-leakage ratios (scratch/coupling_checks.py). The absorber's priority depends on them.

Evidence:
- outputs/autoresearch/noisy-labels/jobs/researcher-20260923T130926Z-598c275a/report.json
- outputs/autoresearch/noisy-labels/jobs/researcher-20260923T130926Z-598c275a/scratch/absorber_sizing.txt
- outputs/autoresearch/noisy-labels/jobs/researcher-20260923T130926Z-598c275a/scratch/coupling_checks.txt
- outputs/autoresearch/noisy-labels/jobs/researcher-20260923T130926Z-598c275a/scratch/oof_direction_checks.txt
- outputs/autoresearch/noisy-labels/jobs/researcher-20260923T130926Z-598c275a/scratch/memory_direction_checks.txt
- Liu et al. 2022 SOP arXiv 2202.14026; Pleiss et al. 2020 AUM arXiv 2001.10528

## completion-20260923T133705Z-39dd970b

ELR (momentum 0.7, λ 0.075) is rejected under the fixed rule, but it is the campaign's strongest point estimate so far. At ×4 on the selected checkpoint (×1-selected epoch index 8), hard macro AUC went from 0.90276 to 0.90407, a gain of +0.00131. The 80% bootstrap interval is [−0.00043, +0.00309], so it fails the rule on the lower bound, and the 95% interval is [−0.00139, +0.00407]. Soft macro AUC moved the same way, 0.79846 → 0.79925 (+0.00079). The soft-target BCE got slightly worse, 0.4386 → 0.4401. Hard and soft agree, so the judge-caveat split, where one rises while the other falls, did not appear.

Per label, the hard and soft deltas have the same sign for every label. Gains:
- ACL: hard +0.0056, soft +0.0022
- Menisci: hard +0.0023 to +0.0036
- Medial OA: hard +0.0030
- Contusion and Fracture: hard +0.0017 to +0.0037
- MCL: soft +0.0045

Losses:
- Lateral OA: hard −0.0032, soft −0.0028
- PF OA: hard −0.0032, soft −0.0020
- Baker's: hard −0.0017, soft −0.0027

This is the same OA-family weakness the bootstrap run showed. It suggests that self-anchoring hurts the diffuse, graded OA labels while helping the focal and structural ones. All per-label intervals include 0. On the report-derived subset (892 studies), hard AUC rose by +0.0013. The gold subset in fold 0 has only 13 studies, and its −0.0009 is uninformative.

Mechanism diagnostics: the weighted regularizer grew from −0.072 to −0.106 while BCE fell from 0.611 to 0.435. Mean prediction change against the reference was 0.035, and per-label Spearman ranged from 0.94 to 0.99. The changed-fraction definition, sign(m−p) against sign(y−p), sits near 0.5 and cannot measure correction, so future memory-based candidates should log a magnitude-based measure instead.

This event interprets one completion; no decisions change. The run is valid. Accounting exactly matches the sealed counts: 937,760 triplets, 35,020 visits and both cohort hashes. It used 4,380 steps at seed 2263738815, and its compute time of 1,652 s is under the 1,846 s ceiling. The escalation asked for the diagnostics the developer's concerns listed. I pulled them from the training logs, and none shows a defect that would invalidate the run.

- The logged `train_label_noise_loss` is already weighted, since `train_loss_epoch` equals BCE plus that value. The regularizer's size relative to BCE rose from 12% in epoch 0 to 24% in epoch 9. That is a real but moderate pull; it is not a dominating term.
- The memory–target gap is identical in epochs 0 and 1 (0.26880). This is expected, not a logging bug. On a study's first visit its prior is its own prediction, so the memory after epoch 0 equals the epoch-0 predictions exactly (`lightning.py:370-379`).
- The sign-based "changed fraction" is 1.0 in epoch 0 by construction, then falls from 0.55 to 0.51. That is close to a coin flip, so it cannot say how many labels were corrected. The gap's fall from 0.269 to 0.155 is mostly the model fitting the labels.

The resume path and the case of one study appearing twice in a batch remain unexercised. Neither affects this uninterrupted run.

The queue stays unchanged. `class-centered-self-correction` remains proposed for the reason recorded in `findings.md:292`. This result does not bear on it.

Unresolved concerns:
- The changed-fraction diagnostic based on the direction of the memory's pull is uninformative, about 0.5 in every epoch after the first. The campaign therefore has no direct measure of how many cells ELR effectively corrected; only the loss ratio and the memory–target gap are available.
- The resume device fix and the case of one study appearing twice in a batch were checked only by reading the code. This uninterrupted run never exercised them.
- A single seed and a single fold. The +0.0013 point estimate excludes seed and checkpoint-selection variance, so it should not be read as a near-miss worth rerunning; there are no automatic confirmation runs.
- The OA-family losses (Lateral OA, PF OA) recur across self-target mechanisms: here and in the bootstrap run. This is an exploratory pattern, unadjusted for multiplicity.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/early-learning-regularization-20260923T124737Z-7aba30fc/comparison/comparison.json (runs.reference/candidate.metrics: macro_auc, soft_macro_auc, per_target_soft_auc, soft_target_bce, gold, report_derived; prediction_change)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/early-learning-regularization-20260923T124737Z-7aba30fc/comparison/per_label.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/early-learning-regularization-20260923T124737Z-7aba30fc/training/fold_0/logs/version_0/metrics.csv (train_bce_loss_epoch, train_label_noise_loss_epoch, train_loss_epoch, label_noise_changed_fraction, label_noise_memory_target_gap)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/early-learning-regularization-20260923T124737Z-7aba30fc/source/src/knee_abnormality/training/lightning.py:364-396 (regularizer, memory update, diagnostics)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/early-learning-regularization-20260923T124737Z-7aba30fc/training/fold_0/selected_checkpoint.json (epoch 08, ×1 0.90093, ×4 0.90407)
- /home/claudio/src/knee-abnormality/autoresearch/campaigns/noisy-labels/findings.md:106,292 (bootstrap OA losses; class-centered proposal deferral)

## review-20260923T134232Z-562472e4

The candidate replaces the fixed exponent of 0.5 with k = 1 − step/4380 during training only. Validation and prediction do not use the weights.

The class-balance strength a checkpoint was trained under depends on the epoch that ×1 validation selects. The reference selected epoch 8, where k is about 0.2 to 0.3. If the candidate also selects a late epoch, its final updates run under weaker balancing than the reference's constant 0.5. The early updates run under stronger balancing, up to 1. Read the result as a combined effect of both phases, not as "k = 0.5 on average".

The contract requires a per-epoch changed fraction. The logged class_balance_weight_changed_fraction is 1.0 at every step except the single step where k equals 0.5. That value is correct: every observed cell's weight differs from the reference weight. It says nothing about how large the change is, so read it together with the per-epoch class_balance_exponent and class_balance_weight_ratio_max.

The idea predicts that hard AUC falls and soft AUC rises. The completion must report hard and soft macro AUC with per-label deltas. Watch the rare-positive labels (MCL, Fracture) in particular, because their weights change the most across the schedule.

I approve launch. The diff implements exactly the specified linear schedule from exponent 1 to 0, and it touches only the training loss weights. The step count comes from Lightning's own estimate, so the exponent averages about 0.5 over the 4380 steps, which matches the recipe's fixed strength. Weights are recomputed from the training-fold positive fractions already saved in the class_balance dict, not from validation data. Validation, checkpoint selection, sampling and preprocessing are untouched. No weight averaging is configured, so the selected checkpoint is the live model.

The queue is unchanged. class-centered-self-correction still has no case for admission, and nothing in this event changes that.

Unresolved concerns:
- The selected epoch fixes the balancing strength the checkpoint ends at. If it differs from the reference's epoch 8, some of the effect comes from a different endpoint strength, not purely from the schedule shape. Record the selected epoch and its k.
- class_balance_weight_changed_fraction is always 1.0, so it cannot show how large the change is. Interpret the per-epoch exponent and max weight ratio instead.
- Soft targets use a hard ≥0.5 threshold to choose the positive or negative weight, the same as the parent. For soft cells near 0.5, which weight they get depends on that threshold, and at k near 1 the weight ratio is at its largest, so this matters most early in training. The behavior is inherited, not introduced here, but its effect is largest early in the run.
- Only a static read of the source was done: nothing was run, and resume behavior of the step-derived exponent was not exercised.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/relaxing-class-balance-schedule-20260923T133826Z-824ef437/changes.diff
- experiment source src/knee_abnormality/training/lightning.py:56-61 (shared weight formula), 101-111 (scheduled tensor), 348-380 (per-step exponent and logging), 382-392 (training_step wiring)
- experiment source src/knee_abnormality/training/lightning.py:130-144 (masked BCE uses targets>=0.5 to pick the weight), 489-509 (validation_step does not use the weights), 567-575 (total_steps=estimated_stepping_batches)
- experiment source configs/training.yaml:98-110 (weight_averaging null, batch 8, 10 epochs, class_balance_exponent 0.5, schedule true)
- /home/claudio/src/knee-abnormality/autoresearch/campaigns/noisy-labels/ideas.yaml:280-340 (idea specification)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/jobs/developer-20260923T133826Z-9ac6fde3/report.json

## user-retire-self-memory-ideas-20260923T140728Z-a7821a6a

USER DECISION: persistent-excess-residual-absorber (queued) and class-centered-self-correction (proposed) are retired; the queue keeps relaxing-class-balance-schedule first. confident-contradiction-downweighting finished training before the decision and keeps its normal comparison.

Retired by explicit user instruction on 2026-09-23: the mechanism corrects or re-weights labels from the model's own running memory of its training predictions. The first three completed experiments showed this premise barely acts in this regime (ten epochs of pretrained fine-tuning, validation BCE still falling at the last epoch, so little label memorization to resist), and the running down-weighting run trailed the reference by about 0.005 at x1 from its third epoch. Preserve the idea as history; requeueing needs new evidence and explicit user review. Soft bootstrapping with a detached own prediction is also degenerate: its gradient is exactly beta times the ordinary BCE gradient, which AdamW normalizes away, so its null result says nothing about bootstrapping with a real teacher.

Unresolved concerns:
- The researcher brief sealed at start still lists memory-driven mechanisms; new proposals of that kind need the same review.

Evidence:
- outputs/autoresearch/noisy-labels/experiments/*/training/fold_0/logs (label_noise_* activity metrics)
- outputs/autoresearch/noisy-labels/experiments/confident-contradiction-downweighting-20260923T130931Z-120e2541/training/fold_0/epochs (x1 macro AUC 0.8795-0.8948 against 0.8829-0.8994 for the reference at epochs 3-9)
- outputs/autoresearch/noisy-labels/decisions/user-retire-self-memory-ideas-20260923T140728Z-a7821a6a/state-before.json

## completion-20260923T140745Z-86edd867

Persistent-disagreement down-weighting (EMA momentum 0.7, min 3 visits, margin 0.5, weight 0.25) is rejected. It hurt every validation view: hard −0.0060, soft −0.0014, gold −0.0155 (58 studies).

The realized changed fraction was 0 for epochs 0–2, then 6.5% at epoch 3, falling to 4.6% by epoch 9. It was heavily skewed toward positives of hard, rarer labels. At epoch 3, 46% of MCL positive cells were flagged, 40% of Lateral Meniscus, 29% of Fracture, 25% of ACL, 25% of Contusion and 24% of Lateral OA. Negatives were flagged much less often, except Effusion at about 12% and the Medial OA/Meniscus labels at 5–9%. Flagged fractions fell over training, e.g. ACL positives 0.25 → 0.06, as the model learned them.

These counts cannot identify the flagged cells as noise; that would need a gold-label check. The dynamics and the result fit the known failure of early-memory noise detection: a per-study EMA started at epoch 3 cannot yet tell hard, not-yet-learned true positives from wrong labels. Down-weighting them to 0.25 cut the gradient on exactly the labels that fell most (Lateral OA/Meniscus). The ~25% share of report-derived positives above 0.9 that are not gold-positive is far below the 40–46% flag rate on rare-label positives. That suggests most flags were true positives.

Implication for later ideas in this campaign: mechanisms that trust the model's early predictions for rare positives look harmful. Target correction or anchoring that stays mild early, or that is symmetric in polarity, is more plausible than hard persistent down-weighting from epoch 3. This implication is interpretive and does not define any follow-up.

Completion event. The fixed decision is reject, and the evidence supports it on every judge. Hard macro AUC went from 0.90276 to 0.89680 (Δ −0.00597; 80% CI [−0.0074, −0.0045]; 95% CI [−0.0082, −0.0039]). All 12 labels fell. Lateral OA (−0.0157), Lateral Meniscus (−0.0137), Medial OA and Medial Meniscus have 95% intervals entirely below zero. The judge caveat does not rescue the result. Soft macro AUC also fell, 0.79846 → 0.79710 (−0.0014). Soft AUC rose only on MCL (+0.0028), Contusion (+0.0037) and, by trivial amounts, ACL and Fracture. Gold macro AUC on the 58 studies fell 0.9236 → 0.9081. So this is not a case of losing on hard AUC while gaining on soft AUC.

The run looks valid for its question. ×1 validation AUC is identical to the reference at epochs 0–2, when the mechanism cannot act (min_visits=3). It diverges from epoch 3, when down-weighting starts. That points to a clean paired comparison with no hidden side effects. Both runs selected checkpoint epoch 8 (0-based index; comparison.json reports selected_epoch 9, apparently 1-based). Accounting and timing are within limits.

The implementation-review concerns that triggered escalation do not change the reading. The normalization detail can at most shift loss scale a little; it cannot explain a uniform loss across all labels on soft, hard and gold AUC. The resume and duplicate-study concerns do not apply because the run had no resume.

The queue is unchanged. The prepared class-balance schedule is independent of this mechanism.

Unresolved concerns:
- The runs did not record which cells were flagged, so the flagged cells were never checked against the 58 gold studies. That noise-vs-hard-example precision would decide whether the mechanism can't detect noise or detects it and down-weights it too hard; it remains unknown.
- Single seed, single fold. Per-label intervals are unadjusted and exploratory. Still, the losses are consistent across all 12 labels and all three judges.
- Gold macro AUC on 58 studies is itself very noisy; treat the −0.0155 as directional only.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/confident-contradiction-downweighting-20260923T130931Z-120e2541/comparison/comparison.json (runs.*.metrics.macro_auc, soft_macro_auc, per_target_soft_auc, gold.macro_auc, selected_epoch)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/confident-contradiction-downweighting-20260923T130931Z-120e2541/evaluation.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/confident-contradiction-downweighting-20260923T130931Z-120e2541/training/fold_0/logs/version_0/metrics.csv (label_noise_contradicted_fraction*, val_macro_auc per epoch)
- /home/claudio/src/knee-abnormality/outputs/training/k05-clean/fold_0/logs/version_0/metrics.csv (reference val_macro_auc identical at epochs 0-2)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/confident-contradiction-downweighting-20260923T130931Z-120e2541/source/src/knee_abnormality/training/lightning.py:133-165 (contradiction rule on sigmoid memory), :425-466 (wiring)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/jobs/orchestrator-20260923T131427Z-0bd05cb8/report.json (prior review concerns)

## completion-20260923T143621Z-d5106a81

A linear class-balance schedule (k from 1 to 0 over 4,380 steps) scores the same as the reference's fixed k=0.5 on hard AUC and slightly better on soft AUC. Hard macro AUC changed by −0.000002 (80% interval [−0.0017, +0.0017]). Soft macro AUC rose from 0.79846 to 0.79917 (+0.0007; the comparison has no soft interval). Soft-target BCE fell from 0.4386 to 0.4330, which fits weaker balancing late in training producing less inflated probabilities. The ×1 validation selected epoch 8 in both runs, at mean k 0.15 against 0.5.

Per-label hard deltas are exploratory, unadjusted 95% intervals, all of which include 0. Gains: Fracture +0.0056, ACL +0.0042, Synovitis +0.0038, Medial OA +0.0032, Baker's +0.0032, Medial Meniscus +0.0029. Losses: Contusion −0.0082, Lateral Meniscus −0.0063, MCL −0.0045, PF OA −0.0027, Effusion −0.0013. Lateral OA was flat. The soft deltas mostly point the same way, and they are small: ACL +0.004, Baker's +0.003, Fracture +0.003, PF OA −0.004, Lateral Meniscus −0.0025. The two rare-positive labels split: Fracture improved on both measures, while MCL fell on hard (−0.0045) and rose slightly on soft (+0.0009). Macro AUC on the 13 gold fold-0 studies fell from 0.924 to 0.891. That sample is too small to interpret.

The candidate's predictions changed materially: mean absolute probability difference 0.037, per-label Spearman 0.90–0.99. The re-weighting of labels mostly cancelled out in macro AUC. The predicted pattern (hard AUC down, soft AUC up) showed only as a soft gain within noise, with hard AUC unchanged. Class-balance strength over training looks like a weak lever on fold-0 AUC here. The public-score sensitivity to the exponent (0.940 to 0.945) was not reproduced locally by changing the exponent's path.

This completion has the fixed decision reject, and I interpret it that way. Hard macro AUC came out even: 0.902758 against the reference's 0.902760, a difference of −0.000002. The bootstrap mean was +0.000025 and the 80% interval was [−0.00170, +0.00173], so the result misses both the +0.001 minimum gain and the lower-bound condition. The review concerns that caused the escalation are now settled by the run artifacts. Training did not resume, since execution.json has resume: null. The ×1 validation picked epoch 8 (checkpoint best-epoch=08, 0.89907), the same epoch the reference picked. So the endpoint confound concern does not apply: the two runs differ only in the balancing path. The candidate's k averaged 0.95 in epoch 0 and 0.15 in epoch 8, while the reference stayed at 0.5 throughout. The logged schedule matches the design: the per-epoch mean exponent falls from 0.95 to 0.05, and the maximum weight ratio falls from 17.4 to 1.35. The changed fraction is 1.0 (0.9977 in epoch 5). As expected, it only shows that the rule was active; it does not show how large the change was. No proposals are pending and the queue is empty, so the ranking does not change.

Unresolved concerns:
- The comparison gives the soft macro AUC delta (+0.0007) without a paired interval, so whether it is real cannot be judged; treat it as noise-level.
- Only one seed and one fold were run, so seed variance can hide a real effect of this size in either direction.

Evidence:
- outputs/autoresearch/noisy-labels/experiments/relaxing-class-balance-schedule-20260923T133826Z-824ef437/comparison/comparison.json (runs.*.metrics: macro_auc, soft_macro_auc, per_target_soft_auc, soft_target_bce, gold; prediction_change)
- outputs/autoresearch/noisy-labels/experiments/relaxing-class-balance-schedule-20260923T133826Z-824ef437/comparison/per_label.json
- outputs/autoresearch/noisy-labels/experiments/relaxing-class-balance-schedule-20260923T133826Z-824ef437/training/fold_0/logs/version_0/metrics.csv (class_balance_exponent, class_balance_weight_ratio_max, class_balance_weight_changed_fraction per epoch)
- outputs/autoresearch/noisy-labels/experiments/relaxing-class-balance-schedule-20260923T133826Z-824ef437/training/fold_0/best-epoch=08-val_macro_auc=0.89907.ckpt (selected epoch 8)
- outputs/autoresearch/noisy-labels/experiments/relaxing-class-balance-schedule-20260923T133826Z-824ef437/training/execution.json (resume: null)
- outputs/autoresearch/noisy-labels/jobs/orchestrator-20260923T134232Z-2998672b/report.json (implementation review concerns)

## proposals-20260923T150414Z-543ba3f6

Precedent evaluations confirm how the two singles ended. relaxing-class-balance-schedule was rejected: candidate 0.902758 against reference 0.902760, 80% interval [-0.00170, 0.00173]. early-learning-regularization was also rejected: candidate 0.904068, +0.00134, 80% interval [-0.00043, 0.00309]. It passed the minimum gain, but the interval's lower bound was not above 0. The combination is admitted as a single standalone idea, not as something built on an adopted incumbent. active_incumbent_ideas is empty, and the incumbent is still historical-control (k05-clean).

I admitted and queued the user's requested combination, schedule-plus-early-learning. It is the only proposal and the queue was empty. It has no declared dependencies, and proposal_admission reports them as satisfied. The combination has never been run, so the rule against re-queuing rejected work does not apply, even though both single mechanisms were rejected locally. It stays within scope. The class-balance exponent schedule is shared by all labels, which the contract allows when a schedule replaces the balance strength. The regularizer is prediction-anchored, uses per-study memory filled only from this run's training batches, and has one shared set of coefficients. It uses only the files and config prefix the campaign allows to be edited (trainer.label_noise).

Both mechanisms change the loss in different places: one reweights the BCE term, the other adds a separate term. So a small additive effect is plausible, but I have no prior evidence for it. The early-learning single run gave the best local result so far: +0.00134 macro AUC, 80% bootstrap interval [-0.00043, +0.00309]. The schedule single run tied: +0.0000 with 80% interval [-0.00170, +0.00173]. On the local judge, the realistic expectation is a gain near the early-learning result, and it may again fail the 80% lower-bound rule. The shared adoption rule decides the result, whatever the public-score motivation.

Unresolved concerns:
- The proposal's exploitation criterion is a public-leaderboard score of at least 0.945, and it mentions submitting the weights. This campaign authorizes only local training and validation. Any submission is a separate decision for the user outside the campaign, and the local verdict follows the shared rule alone.
- Check at review that the merge keeps the early-learning term un-weighted by class, even though the class-balance weights are now scheduled. Also check that the k(step) schedule uses the same total_steps (4,380) and is resumed correctly from global_step.
- Check at review that the running-average memory (momentum 0.7) is still saved and restored with the checkpoint, and that it is only filled from training batches. Both sets of logged metrics must survive the merge, including the per-epoch fraction of training cells changed that the contract requires.
- Judge caveat: a class-balance schedule can move hard and soft AUC in opposite directions. Report both, with per-label deltas, as the contract requires.

Evidence:
- outputs/autoresearch/noisy-labels/experiments/early-learning-regularization-20260923T124737Z-7aba30fc/evaluation.json (+0.00134, 80% interval [-0.00043, 0.00309], reject)
- outputs/autoresearch/noisy-labels/experiments/relaxing-class-balance-schedule-20260923T133826Z-824ef437/evaluation.json (-0.000002, 80% interval [-0.00170, 0.00173], reject)
- outputs/autoresearch/noisy-labels/experiments/early-learning-regularization-20260923T124737Z-7aba30fc/changes.diff
- outputs/autoresearch/noisy-labels/experiments/relaxing-class-balance-schedule-20260923T133826Z-824ef437/changes.diff
- outputs/autoresearch/noisy-labels/imports/ideas-20260923T150414Z-c7ed10e1/report.json
- autoresearch/campaigns/noisy-labels/combo-ideas.yaml

## review-20260923T150651Z-79023080

Both components failed alone: early-learning-regularization and relaxing-class-balance-schedule were each rejected, so this run only tests whether they help each other. With the same seed and step count, the combined run separates additive effects from interaction effects. That holds only if the completion reports hard and soft macro AUC and per-label deltas against k05-clean, next to both single runs. The mechanism-change logs are there for that: class_balance_exponent, class_balance_weight_changed_fraction, label_noise_changed_fraction and label_noise_memory_target_gap. class_balance_weight_changed_fraction is a 0/1 per-step flag, so it is ≈1 for the whole run and says little. label_noise_changed_fraction is the more useful per-epoch fraction of changed cells.

Approved. The candidate is a faithful union of the two single-mechanism diffs that were already reviewed, and my side-by-side diffs against both precedent sources show no logic or setting drift. It stays inside the objective-only scope and edits only the permitted files. Every new key is under trainer.label_noise, and there are no per-label coefficients or branches on label, side or gold. The mechanism reads no validation data: the per-study memory is filled only in training_step from training batch predictions and restored on resume. Sampling, visits and triplets are unchanged, and the run uses the incumbent seed 2263738815. Runtime cost is negligible over the ELR precedent: 1636 s recorded against the 1846 s ceiling. Preparation took 0.12 s.

Unresolved concerns:
- ELR as inherited: on a study's first visit, most of epoch 0 since studies are visited about once per epoch, the anchor is the detached current prediction. The term then acts as confidence sharpening, not early-learning anchoring. The log(1−overlap) gradient is also large near agreement, bounded only by the 1−1e-6 clamp and gradient_clip_val 1.0. This matches the precedent, so it is a design property, not a merge defect.
- Both parent mechanisms were rejected alone. A combined gain inside noise would be weak evidence, since no confirmation run is scheduled.
- The judge caveat applies: the schedule ends at exponent 0 (no class balance), which the class-balance curve suggests may move hard and soft AUC in opposite directions. Soft AUC must be reported alongside.

Evidence:
- outputs/autoresearch/noisy-labels/experiments/schedule-plus-early-learning-20260923T150445Z-fac24a7d/changes.diff
- candidate source lightning.py:389-456 (schedule + ELR regularizer), :459-466 (training_step composition), :666-671 (pre-existing total_steps)
- diff vs experiments/early-learning-regularization-20260923T124737Z-7aba30fc/source: only schedule additions
- diff vs experiments/relaxing-class-balance-schedule-20260923T133826Z-824ef437/source: only ELR additions
- experiments/early-learning-regularization-20260923T124737Z-7aba30fc/training/fold_0/timing.json: 1586.6 s train + 49.3 s ×4 prediction (4380 steps, 3502 studies)
- jobs/developer-20260923T150445Z-ddb14877/report.json

## proposals-20260923T151632Z-aada5d11

Static check of the proposed mechanism against the baseline source:

- encode_batch returns token-level `study`, `image_plane` and `visual_valid`, but not the per-token series type (model.py:334-345). model.py is not editable. Recomputing keep_series in lightning.py would repeat `_effective_presence`, which may draw series-dropout randomness during training. So 'stratify within series type' can be done safely only by image_plane, unless the type can be read without extra RNG draws.
- The token rows are triplet images, about 27 per visit (937,760 / 35,020). Each half therefore holds about 13 images, and empty halves should be rare.
- `_attention_pool` raises on non-finite logits (model.py:262-263), so any study with an empty half must be excluded from the pool call as well as from the loss.

I admitted full-to-half-bag-distillation. It has no dependencies, and it is the only unretired direction that is not degenerate. The researcher's argument holds: any target f(y, stopgrad p) on the same logit only reweights (p−y). A detached full-bag teacher applied to a different readout of the same encoded tokens is new. It has no memory and no second network, so it passes the retirement.

Scope: the primary logits stay bit-identical (model.py:371-372 is exactly pool_encoded(encode_batch)). The halves come only from masking visual_valid in an extra loss term. They are not new training data, sampling or augmentation. Visits, triplets, steps and epochs are unchanged, and only lightning.py, config.py and training.yaml need to change. I count it as a training-objective change. The judgment call is recorded under concerns.

It goes in the queue behind nothing. Its independence from the running schedule-plus-early-learning result is preserved: the proposal says to rebase only if that run is adopted, and otherwise to add to historical-control.

Unresolved concerns:
- Stratifying by series type is not directly available: encoded exposes image_plane but not token_type. The developer must stratify by plane, or get the type without new RNG draws. Recomputing _effective_presence could shift the series-dropout stream and break the bit-identical primary logits.
- Empty-half safety: `_attention_pool` raises on non-finite logits (model.py:262-263). A study with one image, or a half with no valid tokens, must be removed before pooling, not just masked in the loss.
- Scope judgment: masking half-bags at pooling time sits on the line with the excluded 'augmentation'. I admit it as a loss-side second readout (the same decision the losses orchestrator made), but reviewers should confirm that validation, prediction and export use the full bag only.
- The mechanism adds no new label information. The losses campaign's lagged draw-consistency anchor lost 0.0030, with focal ligament and meniscus labels down. The prior for passing the adoption rule is modest.
- Runtime: log diagnostics once per epoch from buffers. The +9.6% in the losses campaign came from per-step logging, and the expected ~1,640 s leaves little room under the 1,846 s ceiling.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/jobs/researcher-20260923T150445Z-094a9686/report.json (proposal and exhausted-directions analysis)
- outputs/autoresearch/noisy-labels/baseline/source/src/knee_abnormality/modeling/model.py:266-345 (encode_batch returns study, image_plane and visual_valid; no series type)
- outputs/autoresearch/noisy-labels/baseline/source/src/knee_abnormality/modeling/model.py:358-372 (pool_encoded masks by visual_valid; forward is pool_encoded(encode_batch))
- outputs/autoresearch/noisy-labels/baseline/source/src/knee_abnormality/modeling/model.py:262-263 (non-finite pooled logits raise)

## review-20260923T152349Z-2099fef8

This candidate is a subset-consistency regularizer, not a target correction. The full-bag readout still trains only on the report labels, with the same class-balanced BCE. The detached full-bag prediction teaches each half-bag readout, and gradients reach the shared encoder, attention and head. Any gain on noisy labels would therefore be indirect: it would come from making the pooled evidence less reliant on any single acquisition, not from down-weighting wrong labels. series_dropout 0.2 already randomizes the bags, so the new term partly overlaps with it. Read soft AUC and mean_abs_delta together with hard AUC when interpreting the result.

I approve the candidate. The diff implements the assigned same-step sub-bag self-distillation inside the campaign scope: a prediction-anchored training-objective term with one weight shared by all labels. It uses no label tables, no validation signal and no memory. I checked the correctness points that matter directly in source: the rewritten training_step computes the same logits as forward, the half-bag masks and study re-indexing are consistent with the qualifying-study count (both require at least 2 valid acquisitions), and the shared position_2d broadcast is right. The only config changes are the two trainer.label_noise keys. The preparation gate recorded no failure.

Unresolved concerns:
- The contract asks for the fraction of training cells the mechanism changed per epoch. train_sub_bag_distillation_changed_fraction only reports the share of observed cells that belong to qualifying studies. With batch 8 and multi-acquisition studies this is about 1.0 after epoch 0 and 0 before it, so it carries almost no information. train_sub_bag_distillation_mean_abs_delta is the informative mechanism log.
- The teacher is the current full-bag prediction, which is itself fitted to the noisy labels. The term never moves any cell's target away from its report label, so its link to label-noise robustness is indirect. A null or hard-AUC-only result would say little about the noisy-labels hypothesis itself.
- Runtime has not been measured. The per-study Python loop forces a host-device sync (.cpu() and .unique().tolist()) each step, on top of two extra poolings. It is expected to stay well under the 115% ceiling, but only the completion timing will confirm that.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/experiments/full-to-half-bag-distillation-20260923T151702Z-304fcf92/changes.diff
- experiment source src/knee_abnormality/modeling/model.py:266-372 (encode_batch, pool_encoded, and forward defined as pool_encoded(encode_batch))
- experiment source src/knee_abnormality/modeling/spatial_tokens.py:199-246, 275-282 (xy grid shared across acquisitions, so position_2d is correctly left unmasked)
- experiment source src/knee_abnormality/training/lightning.py _sub_bag_halves, _half_bag_encoded, _sub_bag_distillation, training_step (the added code shown in the diff)
- configs/training.yaml:84-85,101 (dropout 0.0, series_dropout 0.2, batch_size 8)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/noisy-labels/jobs/developer-20260923T151702Z-afe1672e/report.json

## completion-20260923T153528Z-7bd3acbc

Combining the two mechanisms made things worse than either alone, so they are antagonistic. On the same seed, macro AUC against k05-clean moved as follows (hard / soft):
- ELR alone: +0.0013 / +0.0008
- Schedule alone: 0.0000 / +0.0007
- Combined: −0.0003 / +0.0002

Soft gains shrink when the two are combined. Gold macro AUC on 13 studies is 0.9227 against 0.9236, too few studies to interpret. Soft-target BCE improves slightly (0.4377 against 0.4386).

The loss comes from the OA family:
- Lateral OA: hard −0.0086, 95% interval [−0.0168, −0.0006]; soft −0.0066.
- PF OA: hard −0.0102, 95% interval [−0.0176, −0.0029]; soft −0.0072.

ELR alone already cost about −0.003 on each of these two labels, and here they are roughly three times worse. Effusion (+0.0063), Baker's (+0.0069) and Contusion (+0.0043) gained on both hard and soft AUC. These per-label intervals are exploratory and unadjusted for multiplicity.

Mechanism logs:
- The class-balance exponent fell linearly from 0.95 to 0.05, and the maximum weight ratio fell from 17.4 to 1.35.
- The ELR memory-to-target gap fell from 0.32 to 0.16.
- `label_noise_changed_fraction` was about 0.60 in epochs 1–3 and 0.53 in epoch 9. It counts observed cells where the anchor's pull opposes the label's pull, as a direction, not a magnitude.
- The best checkpoint was epoch 8 of 10; both single-mechanism runs selected epoch 9.

Reading: removing class balance late in training while the ELR anchor pulls toward the model's own early predictions seems to hurt the OA grading labels. These labels depend on the rare positive class, so dropping class balance leaves ELR anchoring to predictions biased toward the frequent class. This is a hypothesis; one seed cannot separate it from noise.

This is a completion event. The fixed decision is reject, and I have no reason to question it. Hard macro AUC is 0.902481 against 0.902760 for the reference (Δ −0.00028; bootstrap mean −0.00024). The 80% interval is [−0.00213, +0.00170], so the change is below the +0.001 minimum and the lower bound is below 0. The run is valid: 937,760 training triplets and 35,020 visits match the replay exactly, with 4,380 steps, the incumbent seed, and 1,661.6 s against the 1,846 s ceiling. The review-time concerns do not change the verdict: the epoch-0 ELR anchor equals the current prediction (sharpening), and there is no confirmation run. The queue is unchanged, and no proposals are pending.

Unresolved concerns:
- The epoch-0 value of label_noise_changed_fraction (1.0) is a counting artifact. On a study's first visit the anchor equals the detached prediction, so sign(0) never matches the sign of target−p and every cell counts as opposed (changes.diff lines ~208-240). Treat values from epoch 1 onward as the real measure. Even those count gradient direction, not the size of any change.
- `class_balance_weight_changed_fraction` is a 0/1 flag, as predicted: it is 1.0 in every epoch and uninformative.
- All three comparisons use one seed (2263738815). The interaction effect of about −0.0016 hard is inside the bootstrap noise, and seed variance is not measured.
- Soft AUC hints that the schedule helps (+0.0007 alone) while hard AUC does not. This is the known judge caveat, recorded here but not enough to adopt.

Evidence:
- outputs/autoresearch/noisy-labels/experiments/schedule-plus-early-learning-20260923T150445Z-fac24a7d/comparison/comparison.json (runs.*.metrics: macro_auc, soft_macro_auc, per_target_soft_auc, gold; candidate selected_epoch 8)
- outputs/autoresearch/noisy-labels/experiments/schedule-plus-early-learning-20260923T150445Z-fac24a7d/comparison/per_label.json (Lateral OA and PF OA 95% intervals exclude 0)
- outputs/autoresearch/noisy-labels/experiments/schedule-plus-early-learning-20260923T150445Z-fac24a7d/training/fold_0/logs/version_0/metrics.csv (class_balance_exponent, label_noise_changed_fraction, label_noise_memory_target_gap per epoch)
- outputs/autoresearch/noisy-labels/experiments/schedule-plus-early-learning-20260923T150445Z-fac24a7d/changes.diff (_label_noise_regularizer change accounting)
- outputs/autoresearch/noisy-labels/experiments/early-learning-regularization-20260923T124737Z-7aba30fc/comparison/comparison.json (ELR alone: hard +0.0013, soft +0.0008)
- outputs/autoresearch/noisy-labels/experiments/relaxing-class-balance-schedule-20260923T133826Z-824ef437/comparison/comparison.json (schedule alone: hard 0.0000, soft +0.0007)
