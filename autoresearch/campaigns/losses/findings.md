# Campaign findings

## proposals-20260920T112028Z-ae44200e

Baseline objective confirmed as masked soft-target BCE with fixed per-target positive/negative weights derived from training-fold counts, grouped by target >= 0.5, reduced as one pooled mean over observed cells (lightning.py:104-118, 60-87). Config and statistics are cross-validated (lightning.py:93), so any candidate replacing the weighting must keep that agreement consistent rather than bypassing it.

configs/training.yaml:99-100 confirms batch_size 8 with accumulate_grad_batches 1, so the microbatch is the joint pair pool: eight studies, no extra joint pairs from accumulation. The recorded zero-pair expectations (15.2% ACL, 26.3% MCL, 55.0% fracture) are analytical from saved counts, not observed frequencies, so a ranking candidate must log realized per-target usable pairs and zero-pair batch fractions rather than rely on those estimates.

No loss helper directory exists yet under training/ (only checkpoints.py, cuda.py, lightning.py), so the first accepted candidate creates training/losses/ from scratch; there is no prior shared helper to reuse or supersede.

Campaign history is empty: no loss candidate implemented or trained. The queue starts at three; no incumbent change.

All four proposals stay inside the shared-loss boundary: one formula and scalar parameters for all twelve targets, current-batch logits only, no data/optimizer/inference change, and edits confined to the allowlisted loss files. None revives per-label selection, classifier banks, learned mixtures, study resampling, or a dedicated AUC optimizer. I accept three and open the queue with them.

shared-bce-ranking first: the metric is per-target ordering while the incumbent supervises cells independently, so this is the mechanism with the most direct alignment to the objective gap, and retaining BCE covers the batches with no usable pair. It is implementable on existing logits (lightning.py:305-320) with no added pass.

shared-balanced-focal second: an independent pointwise mechanism changing the easy/hard cell allocation, elementwise on the same logits, with external ACL/MCL motivation whose transfer to macro AUC here is untested.

shared-tempered-balance third: grounded in a locally measured baseline and the cheapest to implement (reuses stored training counts, lightning.py:60-87), but the incumbent full inverse-frequency weighting already succeeded, so the incremental headroom is the least certain of the three. Its tempering and normalization are bundled; the result attributes to the combined rule, not to either factor.

shared-delayed-ranking is contract-compliant and not rejected, but I retain it as a surplus proposal rather than queueing it. It reuses the same ranking helper and pair rule as shared-bce-ranking, and its only new degree of freedom is the transition point. Choosing that transition without the always-on hybrid's measured behavior — particularly observed pair coverage and the ranking component's realized scale — would be an unmotivated timing guess, and its own priority note concedes it is most informative afterwards. It should be reconsidered once shared-bce-ranking reports.

One scale interaction the first implementation must resolve explicitly: the incumbent reduces BCE as a single pooled mean over all observed cells (lightning.py:117), so labels contribute in proportion to observation counts. A ranking term averaged within each label and then across labels implies uniform label weighting instead. The mixture coefficient therefore cannot be read as a plain component ratio; the effective per-label weighting of both components must be recorded.

Unresolved concerns:
- BCE reduces as a pooled mean over observed cells (lightning.py:117), so a per-label-averaged ranking term changes implicit label weighting; the mixture coefficient is not a plain component ratio and effective weighting must be recorded.
- Fracture pair coverage at batch size 8 may be near-absent in most batches; the 55% zero-pair figure is analytical, so realized coverage per target must be logged before interpreting any ranking result.
- Ranking pair eligibility must define soft-target treatment without hardening uncertain labels; the existing >= 0.5 grouping is a class-weight lookup, not a licence to binarize targets for pairs.
- shared-tempered-balance bundles a tempering exponent with a normalization rule; its outcome attributes to the combined policy and isolates neither factor.
- Focal down-weighting of easy cells can amplify label noise on soft/uncertain targets; the shared exponent needs a stated mechanism, not a sweep or an assumption of noise robustness.
- shared-delayed-ranking's transition point is unmotivated until the always-on hybrid reports observed pair coverage and component scale; queueing it now would risk a timing guess.

Evidence:
- src/knee_abnormality/training/lightning.py:104-118 (masked_binary_cross_entropy_with_logits, pooled mean over observed cells)
- src/knee_abnormality/training/lightning.py:60-87 (training-fold class counts, positive/negative weight formula)
- src/knee_abnormality/training/lightning.py:93 (class_balanced_bce config/statistics agreement check)
- src/knee_abnormality/training/lightning.py:305-320 (training_step; logits cast to float32, single loss call)
- configs/training.yaml:97-108 (batch_size 8, accumulate_grad_batches 1, class_balanced_bce true)
- autoresearch/campaigns/losses/prior-work.md (control B seed 2263738815, macro AUC 0.8965816694263277; pair-coverage expectations 15.2/26.3/55.0%)
- outputs/autoresearch/losses/imports/ideas-20260920T112028Z-ef1df559/report.json (four imported proposals and exhausted directions)
- outputs/autoresearch/losses/jobs/orchestrator-20260920T112858Z-f8c459da/history-index.json (empty ideas/experiments/reports at campaign startup)

## review-20260920T113329Z-f4c0addb

Implementation is a faithful shared hybrid: one formula, one weight, one temperature applied identically to all twelve labels, with pair grouping derived from the same `>= 0.5` rule the adopted class-balanced BCE already uses. The label-skip rule makes the ranking term's effective per-label contribution coverage-dependent: with the recorded analytical zero-pair rates (ACL 15.2%, MCL 26.3%, Fracture 55.0% at microbatch 8, prior-work.md:57-63), rare labels enter the averaged ranking term in roughly half of steps but at full weight when present, while common labels enter nearly always. That is a shared rule, but it means the realized ranking pressure is tilted toward prevalent labels and must be read against the logged `train_ranking_covered_labels` / `train_ranking_pairs` rather than assumed uniform. weight=1.0 and temperature=1.0 are an unswept default justified only by softplus(0)=ln2 matching BCE's initialization floor; because the BCE term carries class-balance weights, the two components' realized magnitudes may diverge during training, so the logged `train_bce_loss` and `train_ranking_loss` traces are the evidence needed to interpret an inconclusive or negative result as "hybrid does not help" versus "scale was wrong". Runtime impact is a 12-iteration Python loop over small tensors per step; unmeasured but expected to be negligible against the 1,797 s ceiling. Preparation cost recorded at 0.17 s.

I inspected changes.diff and both sources directly rather than relying on the developer's claims. `diff -rq` against parent_source shows exactly four touched paths, all inside the campaign allowlist: configs/training.yaml (+trainer.loss.ranking), config.py (additive RankingLossConfig/LossConfig, trainer.loss default_factory so the field is optional), lightning.py training_step only, and the new training/losses package. class_balanced_bce stays true and masked_binary_cross_entropy_with_logits is byte-identical, so the pointwise term and its soft targets are untouched.

Protected behavior in lightning.py: data loading, model construction, configure_optimizers/scheduler, forward, validation_step, on_save/on_load_checkpoint, batch-norm/weight-averaging, and the validation logging path are unchanged in the diff. The only other edit is `batch_size=batch_size` factoring in the existing train_loss log. `train_loss` appears nowhere else in the repo (only lightning.py:314), so checkpoint selection and scoring are unaffected by the changed objective value; validation remains deterministic and independent of the training objective and seed.

Ranking term (losses/ranking.py): pairs are formed only from `observed` cells (bool from dataset.py:277-279), so mask accounting is correct and unobserved cells cannot leak into the objective. Self-pairs are structurally impossible since pos and neg sets are disjoint. Grouping uses the same `>= 0.5` threshold already used by fixed_class_balance, so pair eligibility does not rewrite soft source labels. Per-label mean over pos×neg pairs, then mean over covered labels; zero-pair labels are skipped (not zero-filled), and an all-empty batch returns an exact zero leaving BCE alone. Logits are cast to float32 before both terms and softplus is stable under bf16-mixed. No extra forward pass, no state, no validation-label access; batch_size 8 with accumulate_grad_batches 1 (configs/training.yaml:99-100) means pairs are genuinely within-microbatch, consistent with the contract's coverage caveat. Diagnostics log covered labels and pair counts, satisfying the coverage-logging requirement.

Remaining items are interpretation caveats, not defects, so I am approving rather than requesting a repair.

Unresolved concerns:
- Component scale untuned: weight=1.0 against class-weighted BCE may leave one term dominant; interpret outcome using the logged train_bce_loss/train_ranking_loss traces before concluding the mechanism failed.
- Label-skip averaging makes effective ranking pressure coverage-dependent, favoring prevalent labels; expected zero-pair rate ~55% for Fracture at microbatch 8.
- Triplet accounting is not yet available for this candidate (accounting_available=false); the mandatory replay and gates still apply at run time.
- Runtime overhead of the per-label Python loop is unmeasured; expected negligible but only the measured run confirms it against the 1,797 s ceiling.

Evidence:
- changes.diff: outputs/autoresearch/losses/experiments/shared-bce-ranking-20260920T112958Z-a2e23db7/changes.diff
- candidate src/knee_abnormality/training/losses/ranking.py:1-31 (pair construction, mask use, zero-pair return)
- candidate src/knee_abnormality/training/lightning.py:305-358 (training_step, BCE preserved, diagnostics)
- baseline src/knee_abnormality/training/lightning.py:104-118 (masked BCE unchanged; `>= 0.5` class-weight grouping)
- baseline src/knee_abnormality/data/dataset.py:268-279 (soft target float32, observed bool mask)
- baseline configs/training.yaml:99-100 (batch_size 8, accumulate_grad_batches 1)
- repo-wide grep: `train_loss` occurs only at lightning.py:314, so selection/scoring do not read the objective value
- autoresearch/campaigns/losses/prior-work.md:55-69 (analytical zero-pair rates, coverage measurement requirement)

## review-20260920T113905Z-c1568240

Faithful, minimal implementation of the campaign's documented shared balanced-focal objective: one scalar gamma for all twelve labels, mask/class-weight/reduction semantics bit-identical to the control BCE (developer reports gamma=0 reproduces it exactly at 1.2057465314865112), numerically stable via logsigmoid on both branches with float32 logits under bf16-mixed. gamma=2.0 is the canonical unswept default, not a tuned value; since the contract fixes the optimizer and learning rate, a negative or inconclusive macro result will confound the focal mechanism with the reduced effective gradient scale that gamma=2 imposes at fixed LR. The train_loss trace is now a focal value and is not comparable to the control's BCE trace or to the retained validation BCE logging; only macro AUC and the paired bootstrap are comparable. Under soft targets the focal factors move the per-cell optimum away from p=y in a target-dependent manner, so the candidate deliberately trades soft-label calibration fidelity for easy-example downweighting — expected under the definition in per-label-losses.md:104-111, but it means a per-label loss trace cannot be read as a calibration diagnostic.

Inspected changes.diff and both sources directly. Scope is clean: `diff -rq` against parent_source shows exactly configs/training.yaml, config.py, lightning.py and the new training/losses/ package (focal.py, empty __init__.py); config delta is only trainer.loss.gamma=2.0 under the allowed prefix, class_balanced_bce unchanged at true.

Objective correctness: focal.py implements the documented soft-target definition `FL=-y(1-p)^g log p-(1-y)p^g log(1-p)` (per-label-losses.md:104-111) with logsigmoid for both log terms, so no log(0); at large |logit| the vanishing focal factor multiplies a finite logsigmoid, and sigmoid saturation yields 0 rather than NaN. gamma flows through the focal factors (no detach), as the definition requires. Mask, class-weight lookup and reduction are byte-identical to the parent BCE helper (lightning.py:110-118 vs focal.py:11-25): same empty-mask RuntimeError, same whole-cell `targets>=0.5` weight, same `losses[observed].mean()`. training_step still casts logits to float32 before the loss, so the objective is computed outside bf16. One scalar gamma applies to all twelve labels; no per-label branch, vector, or mixture. No pair pools, memory banks, extra passes, or training-only state; validation, checkpoint selection, data loading, model, optimizer and accounting code are untouched by the diff, and the parent BCE helper is retained for validation logging and the non-editable profiling import. No validation-fold quantity enters the objective — only the pre-existing training-fold class_balance weights.

Cost: two extra elementwise ops per training cell; negligible against the 1,797 s ceiling. Preparation 0.17 s.

Interpretive limits worth carrying into the result reading are recorded as concerns: gamma=2.0 is the unswept Lin et al. default; focal at fixed LR reduces effective gradient scale, which the research note itself flags (per-label-losses.md:102) but the contract's fixed optimizer forbids compensating; and for soft targets in (0,1) the focal stationary point no longer sits at p=y, so soft-label calibration is shifted in a target-value-dependent way (AUC is rank-based, so this is a mechanism caveat rather than a defect). Triplet accounting is not yet available; the mandatory replay and gates still decide at run time.

Unresolved concerns:
- gamma=2.0 is an unswept default; at the contract's fixed LR, focal lowers effective gradient scale, so a negative/inconclusive result cannot separate the mechanism from the scale change.
- Soft-target focal shifts the per-cell optimum away from p=y in a target-dependent way; interpret only macro AUC, not loss values or calibration.
- Triplet accounting is unavailable at review time (accounting_available=false); the mandatory replay and gates still apply at run time.
- train_loss is now a focal quantity and is not comparable to the control's BCE trace or the retained validation BCE logging.

Evidence:
- experiments/shared-balanced-focal-20260920T113454Z-c1fe7f53/changes.diff (full diff; 4 changed entries)
- candidate source/src/knee_abnormality/training/losses/focal.py:11-25 (mask, focal factors, class weight, mean over observed)
- baseline source/src/knee_abnormality/training/lightning.py:104-118 (parent BCE helper; semantics matched)
- candidate source/src/knee_abnormality/training/lightning.py:307-312 (training_step call, float32 logits preserved)
- docs/research/per-label-losses.md:104-111 (proposed soft-target focal definition, whole-cell class weight, differentiate through focal factors)
- docs/research/per-label-losses.md:102 (loss scale and focal exponent change updates at the same nominal LR)
- jobs/developer-20260920T113454Z-48711e4e/report.json (gamma=0 equals BCE at 1.2057465314865112; finite gradients at logits +/-50)
- diff -rq baseline/source vs candidate/source: only configs/training.yaml, config.py, lightning.py differ plus new training/losses/ (__init__.py empty)

## proposals-20260920T114100Z-3dafc95e

Verified analytically that the proposed Bernoulli pair term shares soft BCE's population optimum at tau = 1: its derivative is (a+b)(sigmoid(d) - a/(a+b)), giving s_i - s_j = logit(y_i) - logit(y_j). This holds only at tau = 1; any other temperature rescales the optimum to tau·(logit y_i - logit y_j) and breaks the agreement that motivates the idea, so tau = 1 is a structural requirement of the stated mechanism, not a default to vary.

The evidence note's Monte-Carlo pair-coverage table reproduces the independently documented analytical hard-rule zero-pair rates to within 1 point across ACL/MCL/Fracture, which corroborates the new Bernoulli-mass column and the claim that the confidence-weighted rule eliminates the zero-pair failure mode (Fracture 54.7% zero-pair batches under the hard rule, 0.00% of batches below 0.5 effective mass under the Bernoulli rule).

Control B's error localization (ambiguous studies carry 1.5-2.4x their pair share of discordant mass in 12/12 labels; Synovitis 52.8% of pairs, 83.0% of error, confident-only AUC 0.9638 vs 0.8996 overall) is a real and consistent diagnostic, but it is compatible with two readings — intrinsic label noise on ambiguous studies, or a missing ordering signal — and the researcher states this correctly. The ordering reading is a hypothesis for the run to test, not an established fact.

The soft-target table's stated caveat (gold overrides not applied, per-label cell mass 3,473-3,521 rather than 3,503) affects only the descriptive tables, not the authoritative class_balance.json weights, and is too small to change any proposal's motivation.

Admitted two of the three new proposals; both have satisfied dependencies and distinct mechanisms.

shared-bernoulli-pair-ranking ranks second, ahead of the queued tempering idea. I verified its core claim independently: d/dd [a·softplus(-d) + b·softplus(d)] = (a+b)·sigmoid(d) - a, so the optimum is sigmoid(d) = a/(a+b) and s_i - s_j = logit(y_i) - logit(y_j), the same population optimum as soft BCE, unlike a hard >= 0.5 pair term. It is therefore not a pair-rule variant of shared-bce-ranking but a different objective with different coverage and a different optimum, so running both is informative rather than duplicative. Its coverage claim is supported by the researcher's Monte-Carlo, whose hard-rule column reproduces the documented analytical zero-pair rates (15.1/25.9/54.7% vs 15.2/26.3/55.0% for ACL/MCL/Fracture), giving confidence in the Bernoulli-mass column that removes the zero-pair case. The mechanism is shared: one tau, one lambda, twelve diagnostic readouts.

shared-soft-mass-balance ranks last of the four: independent and near-free, but the imbalance it corrects is small (positive/negative term-mass ratios 0.89-1.46 under the current rule), so the expected effect is the weakest. It stays in the queue because it is cheap and mechanistically distinct from shared-tempered-balance, which keeps the thresholded group lookup and changes weight magnitude.

shared-ranking-memory-bank is left as proposed: its dependency shared-bce-ranking is launched and undecided, which does not satisfy the admission gate. I did not park it and did not clear the dependency, which is also substantively correct: launching the bank before the within-batch hybrid reports observed pair coverage and component scale would confound the two.

shared-delayed-ranking is retained as a surplus proposal rather than queued: it schedules the same within-batch hybrid that is currently running undecided, so its value depends on that result.

Section 1 of the evidence note resolves the startup concern about label reduction algebraically (each label's cell-weight mass is exactly n_obs = 3503), so a per-label-averaged pair term does not reweight labels here. I accept that a macro-aligned-reduction candidate would be a numerical no-op.

Unresolved concerns:
- shared-bernoulli-pair-ranking: the shared-optimum property holds only at tau = 1; treat tau as fixed by the mechanism, not as an open parameter.
- shared-bernoulli-pair-ranking open_choices still lists a hard-target variant and a running-mean denominator; both must be settled before launch, since the hard-target form discards the property that justifies the idea over shared-bce-ranking.
- The ambiguous-study discordant-mass evidence comes from control B's fold-0 validation predictions; it is diagnostic motivation only, and repeated idea selection against fold 0 is not captured by the bootstrap interval.
- shared-soft-mass-balance changes both the weight-assignment rule and the reference statistic at once, so an inconclusive result cannot separate soft-mass weighting from the weaker weighting magnitude it implies (Synovitis w_pos 3.875 -> ~1.47).
- shared-tempered-balance and shared-soft-mass-balance modify the same weighting path and cannot be combined; re-read the second once the first reports.
- shared-ranking-memory-bank's staleness risk is unaddressed by evidence: XBM's slow-drift result is for image embeddings, not twelve scalar logits, and must be measured rather than assumed if it is ever admitted.

Evidence:
- outputs/autoresearch/losses/jobs/researcher-20260920T112958Z-9a7e4a21/report.json (proposals event artifact)
- outputs/autoresearch/losses/jobs/researcher-20260920T112958Z-9a7e4a21/scratch/loss-evidence-20260920.md sections 1-6, with loss_evidence.py
- loss-evidence-20260920.md section 3: hard-rule zero-pair 15.1/25.9/54.7% vs documented 15.2/26.3/55.0% in autoresearch/campaigns/losses/prior-work.md:55-62
- loss-evidence-20260920.md section 6: derivative w(sigmoid(d) - t), optimum s_i - s_j = logit(y_i) - logit(y_j)
- loss-evidence-20260920.md section 2: positive/negative term-mass ratios 0.89-1.46; Synovitis weight step 0.574 -> 3.875
- loss-evidence-20260920.md section 4: Synovitis AUC 0.8996 vs confident-only 0.9638; disc amb % 83.0
- outputs/training/balanced-bce/fold_0/class_balance.json (authoritative counts and weights; observed = 3503 for all twelve labels)
- autoresearch/campaigns/losses/prior-work.md:64-70 (training-only memory pools admitted only when evidence justifies them)

## completion-20260920T121537Z-5b39b418

Shared balanced BCE plus pairwise logistic ranking (weight 1.0, temperature 1.0) is approximately neutral on fold 0: macro AUC 0.89729 vs incumbent 0.89658, delta mean +0.00073, 95% paired report-group bootstrap [−0.00329, +0.00479] over 2,000 valid draws; incumbent retained.

The neutral result is informative rather than uninterpretable, because the logged traces rule out the two failure modes flagged at review. The ranking component trained actively and kept a substantial share of the objective throughout (epoch-mean ranking loss 0.5198 → 0.1624 against BCE 0.6369 → 0.4482; ~45% → ~26% of total), and pair coverage averaged ~10.0 of 12 labels with ~118 usable pairs per step in every epoch. So "hybrid does not help at this scale" is supported over "the ranking term was inert or drowned out"; a scale sweep is not the obvious next move, and additional always-on within-batch ranking variants have low expected value.

Per-label deltas are mixed and not ordered by prevalence: Baker's +0.0134, Medial OA +0.0084, Effusion +0.0082, Fracture +0.0029, Contusion +0.0006; Lateral OA −0.0011, PF OA −0.0003, ACL −0.0020, MCL −0.0042, Medial Meniscus −0.0019, Lateral Meniscus −0.0075, Synovitis −0.0079. The anticipated tilt toward prevalent labels from the label-skip averaging rule did not appear in the outcome.

Cost evidence for future pair-based ideas: the 12-label per-step loop added ~52 s (3.3%) over the reference run, leaving ~182 s of ceiling headroom.

Contract checks pass, so the fixed inconclusive verdict stands on valid evidence: accounting complete and exactly at the pinned 938,460/35,030 train and 24,188/904 validation figures; per-epoch validation selection_sha256 is 92f1341c… for all ten epochs, identical to the baseline replay (baseline/triplet_accounting.json), while training selection hashes differ from baseline under the fresh seed 3524821186 — validation identity is independent of the training seed. Config differences are limited to trainer.loss.ranking.{weight,temperature}. Runtime 1614.5 s sits under the 1,797 s ceiling and only 3.3% above the 1562.3 s reference, which resolves the open concern about the per-label Python loop; that 52 s of headroom consumption is the relevant precedent for any heavier pair machinery later.

Two review-time concerns are now resolved by logged evidence and should not be carried forward. (1) Component scale was not degenerate: train_ranking_loss_epoch falls 0.5198 → 0.1624 while train_bce_loss_epoch falls 0.6369 → 0.4482, so the ranking term supplied 45% of total loss at epoch 0 and still 26% at epoch 9 (logs/version_0/metrics.csv). (2) Coverage tilt did not materialize as feared: train_ranking_covered_labels_epoch is ~10.0 of 12 in every epoch with ~118 pairs per step, and the per-label deltas are mixed rather than prevalence-ordered — Fracture, the worst-covered label, gained +0.0029 while Lateral Meniscus lost −0.0075.

Proposal admission: shared-ranking-memory-bank remains blocked, its declared dependency on shared-bce-ranking is unresolved (undecided, not provisional_accept), so it stays proposed with the dependency intact. shared-delayed-ranking is formally admissible but I retain it as surplus rather than queueing it. Its only new degree of freedom is the transition point on a within-batch ranking term that, with full weight from step 0 and non-degenerate realized scale, moved macro AUC by +0.0007 with an interval spanning zero. Delaying the same term is a weaker perturbation of a mechanism now measured as approximately neutral, and the three queued untried mechanisms (Bernoulli pair semantics, class-weight tempering, soft-mass balancing) test distinct hypotheses. Queue membership and order are unchanged.

Unresolved concerns:
- Bootstrap uncertainty is conditional on these two fitted models and excludes training-seed variation; a +0.0007 shift is well inside plausible seed noise on fold 0.
- Component scale remains unswept at weight=1.0; the traces show it was not degenerate, but they do not establish that no mixture coefficient helps.
- Any pair-based successor must fit within the ~182 s of ceiling headroom left after this candidate's 3.3% runtime overhead.
- shared-ranking-memory-bank stays blocked by its unresolved dependency on this undecided idea; its stale-score and refresh handling remain unreviewed.

Evidence:
- experiments/shared-bce-ranking-20260920T112958Z-a2e23db7/evaluation.json (delta +0.000729, CI [-0.003293, +0.004792], 2000 draws)
- experiments/shared-bce-ranking-.../comparison/comparison.json runs/{reference,candidate}/metrics per_target_auc
- experiments/shared-bce-ranking-.../training/fold_0/logs/version_0/metrics.csv epoch rows 0-9 (train_bce_loss_epoch, train_ranking_loss_epoch, train_ranking_covered_labels_epoch ~10.0, train_ranking_pairs_epoch ~118)
- experiments/shared-bce-ranking-.../training/fold_0/triplet_accounting.json (complete; validation selection_sha256 92f1341c… all epochs)
- baseline/triplet_accounting.json (validation selection_sha256 92f1341c…; train epoch-0 selection 3b923700… differs under fresh seed)
- experiments/shared-bce-ranking-.../training/fold_0/logs/version_0/metrics.csv best epoch 8 val_macro_auc 0.897291
- facts.timing: measured_compute_seconds 1614.53 vs limits.historical_recorded_seconds 1562.28, ceiling 1797
- autoresearch/campaigns/losses/ideas.yaml: shared-ranking-memory-bank dependencies ['shared-bce-ranking'] (undecided)

## review-20260920T122048Z-8e8e8547

The candidate is a faithful shared objective: one tau, one weight, one formula over all twelve labels, differing from the completed hybrid only in pair semantics. Soft Bernoulli masses a=y_i(1−y_j), b=y_j(1−y_i) make pair mass nonzero whenever any target is strictly interior, so realized coverage should be materially higher than the analytical hard-pair zero rates (ACL 15.2%, MCL 26.3%, Fracture 55.0%; prior-work.md:57-63). That is the mechanism under test and `train_pair_mass_mean` / `train_pair_zero_mass_labels` are the traces that will confirm it.

Two consequences to read the result against. First, per-label normalization by total mass equalizes labels: a label whose batch evidence is a handful of near-zero soft differences yields the same full-weight averaged term as a label with confident 0-vs-1 pairs, so low-confidence labels get amplified relative to the prior variant. Second, the reduction differs from the completed experiment: zero-mass labels are zero-filled into a fixed 12-label mean (lightning.py:313-320) rather than skipped, so the ranking component's effective scale is not directly comparable across the two runs even at identical weight=1.0. Both are defined shared rules; neither is label-specific.

weight=1.0 and tau=1.0 remain an unswept default justified by softplus(0)=ln2 at initialization. Since BCE carries class-balance weights and this term does not, the logged component traces are again the evidence needed to separate "soft pair semantics do not help" from "scale was wrong" — the same discipline that made the previous neutral result interpretable.

`diff -rq` against parent_source shows exactly four touched paths, all inside the allowlist: configs/training.yaml (+trainer.loss.pair_ranking), config.py (purely additive PairRankingLossConfig/LossConfig with optional default), lightning.py (35 changed lines, confined to training_step plus the import), and the new training/losses package (empty __init__, pair_ranking.py). class_balanced_bce stays true and masked_binary_cross_entropy_with_logits is byte-identical, so the pointwise term and soft-target semantics are untouched. Data loading, model construction, optimizer/scheduler, forward, validation_step, checkpoint hooks and accounting show no diff lines; the only edit outside the new block is factoring batch_size into the existing train_loss log. Validation stays deterministic and independent of the objective and training seed.

Pair term (losses/pair_ranking.py:1-34): pairs come from torch.triu_indices offset=1, so self-pairs are structurally impossible; pair_observed = observed[i] & observed[j] uses the bool mask from dataset.py:277-279, so unobserved cells cannot enter either the numerator or the mass. Targets stay soft float32 (dataset.py:268-276) and are never thresholded — this is the intended difference from the completed `>= 0.5` grouping variant, and it does not rewrite source labels. Logits are cast to float32 at lightning.py:306 before both terms, and both branches use softplus, which is stable under bf16-mixed. Denominator is per-label soft pair mass with clamp_min(1e-8); since term ≤ mass·(|d|+ln2), the ratio is bounded by the logit gap even at vanishing mass, so the clamp cannot produce a blow-up. Empty-pair batches (batch of one) return exact zeros and leave BCE alone. batch_size 8 with accumulate_grad_batches 1 keeps pairs genuinely within-microbatch (≤28 per label); no state, buffers, extra forward passes, or validation-label access. Coverage and component diagnostics are logged as required.

Two interpretation caveats, not defects, so I approve rather than repair. Preparation cost 0.17 s; the added work is a fixed 28×12 tensor op with no Python label loop, so it should stay well inside the ~182 s of headroom the prior pair candidate consumed.

Unresolved concerns:
- Mass normalization equalizes per-label weight regardless of confidence: labels whose batch evidence is only near-zero soft differences contribute at full weight; bounded (term ≤ mass·(|d|+ln2)) but it shifts effective pressure versus the hard-pair variant.
- Zero-mass labels are zero-filled into a fixed 12-label mean rather than skipped, so the ranking component's realized scale is not comparable to shared-bce-ranking at the same weight=1.0.
- tau=1.0 and weight=1.0 are unswept defaults; interpret the outcome using train_pair_ranking_loss against the BCE trace before concluding the soft-pair mechanism failed.
- Triplet accounting is unavailable at review time (accounting_available=false); the mandatory replay and gates still apply at run time.
- Runtime overhead is unmeasured; only the measured run confirms it against the 1,797 s ceiling, with ~182 s of headroom indicated by the prior pair candidate.

Evidence:
- changes.diff: outputs/autoresearch/losses/experiments/shared-bernoulli-pair-ranking-20260920T121717Z-fd69b263/changes.diff
- candidate src/knee_abnormality/training/losses/pair_ranking.py:1-34 (triu pairs, mask, soft mass denominator, zero-pair return)
- candidate src/knee_abnormality/training/lightning.py:305-355 (training_step; BCE and float32 cast preserved, diagnostics added)
- baseline src/knee_abnormality/training/lightning.py:104-118 (masked BCE byte-identical)
- baseline src/knee_abnormality/data/dataset.py:268-279 (soft float32 targets, bool observed mask)
- baseline configs/training.yaml:99-100 (batch_size 8, accumulate_grad_batches 1)
- diff -rq baseline/source vs candidate/source: only training.yaml, config.py, lightning.py, new training/losses
- autoresearch/campaigns/losses/prior-work.md:55-69 (analytical zero-pair rates; shared pair/confidence rule requirement)

## proposals-20260920T123343Z-d8a4a84d

The incumbent's cross-label structure is a measured, reproducible property of the model rather than of the fold or the labels: predicted-logit off-diagonal correlation 0.384 and PC1 0.447 against soft-target 0.202 / 0.278, with the training fold's targets at 0.180 / 0.261. Within a study, contrast AUC averages 0.9369 over 64 scoreable pairs but collapses on confusable anatomy (Lateral OA vs PF OA 0.7927, Contusion vs Fracture 0.8015, Medial OA vs PF OA 0.8402), and Spearman(own per-label AUC, mean contrast AUC) = 0.783 — an association, not causation.

Four directions are now closed by local measurement and should not be re-proposed without new evidence: (1) inference-time decorrelation is harmful, the shared leave-one-out transform costing -0.00054 at beta 0.1 and -0.00480 [-0.00773, -0.00235] at beta 0.3, so the excess correlation carries real signal and any cross-label objective must strengthen label-specific evidence during training rather than decorrelate outputs; (2) temporal self-ensembling gives nothing (late-epoch logit averaging 0.89512-0.89663 against the selected 0.89658; per-cell across-epoch std 0.098 against per-label IQRs 1.45-3.62), removing the motivation for stale-self-consistency terms including any weight-trajectory memory; (3) per-label loss scale is already even (0.414-0.522 class-weighted BCE), so dynamic per-label normalization has nothing to correct; (4) report groups carry no bag structure (98.8% singletons in fold-0 training, 14 of 42 multi-study groups with identical rows), closing multiple-instance objectives.

Training-time input-view variance remains unmeasured locally; the only recorded figure is the ~0.0035 shift from validation_image_multiplier 2 vs 1, so view-invariance objectives cannot yet be motivated or costed.

Both new proposals are contract-compliant shared objectives (one formula, one tau/lambda or one fixed constant, all twelve labels, soft targets preserved, no per-label switch) and have no unresolved dependencies; both are admitted.

shared-label-contrast-ranking is ranked above the two weighting-rule ideas because it is the only candidate aimed at a deficiency measured on the incumbent's own predictions in a direction the objective never supervises: prediction off-diagonal correlation 0.384 vs 0.202 for soft targets, PC1 0.447 vs 0.278, with the training fold agreeing (0.180 / 0.261), so it is a model property (evidence sections 1-2, verified in the scratch note). Coverage is established rather than assumed: per-study Bernoulli straddle mass min 1.41 over 3,503 training studies, 528 pair terms per microbatch, so there is no zero-mass regime to design around (section 3). It opens a new comparison axis (labels within a study) distinct from every prior study-axis ranking idea, and it is cheap enough to fit the ~182 s of headroom the prior pair candidate left.

shared-balanced-brier is ranked third: it replaces the pointwise function itself with a strongly proper member of the AUC-regret-bounded family, keeping mask, class weights, reduction and soft-target optimum at p=y, and its factor 2 is fixed by the initialization identity rather than tuned, so it starts gradient-identical to the control cell by cell (section 8). It is cheaper than the log-loss path. Its local evidence is weaker than the contrast proposal's: the redistribution on the confidently-wrong tail is 23.7% -> 20.3% of gradient mass, a modest shift.

shared-delayed-ranking and shared-ranking-memory-bank are retained as surplus rather than queued: the completed hybrid's traces show the study-axis ranking term was active and a substantial share of the objective throughout, so scheduling or widening that same inert-free mechanism has low expected value, and the memory bank's dependency is unresolved.

shared-tempered-balance and shared-soft-mass-balance keep their relative order at the tail; section 6 (class-weighted per-label BCE spans only 0.414-0.522 on identical per-label weight mass) weakens, without eliminating, the motivation for reweighting-rule variants.

Unresolved concerns:
- shared-label-contrast-ranking's lambda=1.0 puts ~36% of the initial objective on a direction macro AUC does not score directly; a negative result would confound the mechanism with that scale, and must not trigger a sweep.
- The local survey rejected the cross-label direction on opinion (auc_loss_approaches.md:135-136), and the one prior label-pair coupling attempt scored 0.8909 vs 0.8924; the bet is that training-time contrast pressure strengthens label-specific evidence, which evidence section 4 does not establish.
- shared-balanced-brier's gradient is roughly half of BCE's at convergence, an implicit late-training step-size cut under the fixed schedule; it removes the focal arm's scale confound only at initialization, not throughout.
- Brier's train_loss trace is a squared quantity, not comparable to the control's BCE trace; only macro AUC and the paired bootstrap are comparable, as for the focal arm.
- Contrast evidence is computed on validation predictions of control B; the per-pair contrast AUCs are descriptive of the incumbent and carry no independent uncertainty estimate.
- shared-ranking-memory-bank remains blocked by its unresolved dependency and its stale-score/refresh handling is still unreviewed.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/losses/jobs/researcher-20260920T121718Z-2ea25138/report.json
- outputs/autoresearch/losses/jobs/researcher-20260920T121718Z-2ea25138/scratch/contrast-evidence-20260920.md:9-57 (cross-label correlation, contrast AUC, pair coverage)
- same note:59-72 (inference decorrelation negative result), :74-85 (snapshot ensembling), :87-99 (per-label scale, report groups), :101-112 (Brier gradient profile)
- reproduction scripts: same scratch dir structure.py, contrast.py, decouple.py, probe.py
- docs/research/auc_loss_approaches.md:135-136,235-237 (cross-label direction ranked out on opinion, no local measurement)
- src/knee_abnormality/training/lightning.py:104,307 (masked BCE call site both proposals target)

## completion-20260920T124339Z-2948d523

Shared balanced focal (gamma=2.0, one scalar for all twelve labels) does not improve fold 0 and trends negative: macro AUC 0.89395 vs incumbent 0.89658, delta mean -0.00260, 95% paired report-group bootstrap [-0.00680, +0.00158] over 2,000 valid draws; incumbent retained.

Two artifacts make this more than a null. The candidate's validation macro AUC plateaus from epoch 6 (0.8915, 0.8926, 0.8940, 0.8938, 0.8932; selected epoch 8) while the control keeps improving through epoch 9 (0.8966), and validation soft-target BCE is worse at every epoch (0.5445 vs 0.4844). Together these match the pre-registered concern that gamma=2 at the contract's fixed LR reduces effective gradient scale and shifts the per-cell optimum under soft targets; the run looks under-converged within the sealed ten epochs rather than differently-converged.

Per-label deltas are structured, not noise-like: the losses concentrate in the labels the control already ranks best - ACL -0.0153, Medial Meniscus -0.0101, MCL -0.0085, Lateral Meniscus -0.0060, Fracture -0.0058 - while gains sit in the OA family (Lateral OA +0.0070, family mean +0.0035) plus Synovitis +0.0047 and Baker's +0.0038. Meniscus family -0.0081 [-0.0182, +0.0028]. Down-weighting confident examples therefore costs exactly where confident easy examples carry the signal.

Cost evidence: focal adds ~17 s (+1.1%) over control, so pointwise reweighting is effectively free against the ceiling; the ~182 s headroom noted for pair-based ideas is unaffected.

Interpretation of the focal completion. Contract checks pass on the artifacts: accounting complete and exact (938,460 train triplets / 35,030 visits, 24,188 validation triplets, cohort hashes matching the limits, identical validation selection hash across all ten epochs), 4,380 steps, sole config difference trainer.loss.gamma=2.0, fresh seed 3356103272, measured compute 1,579.5 s against the 1,797 s ceiling (+1.1% over control's 1,562 s, so focal adds essentially no cost).

The interval [-0.00680, +0.00158] includes zero, but 87% of its width lies below zero and every family mean except OA moves down; this is a weakly negative result, not a symmetric null. Two traces from the comparison sharpen it beyond "no difference". (1) The candidate's validation macro-AUC curve saturates: 0.8915 / 0.8926 / 0.8940 / 0.8938 / 0.8932 over epochs 6-10, selecting epoch 8, while the control is still climbing to 0.8966 at epoch 9. Flat-then-declining tail at fixed LR is the signature the review anticipated for gamma=2 shrinking effective gradient scale, so the "mechanism vs. scale" confound recorded at review is now the live explanation rather than a hypothetical. (2) Validation soft-target BCE is worse at every epoch (0.5445 vs 0.4844 final), consistent with the predicted optimum shift away from p=y under soft targets; as flagged, this is a calibration diagnostic only and does not itself decide the macro criterion.

Queue unchanged. shared-delayed-ranking remains an admissible surplus proposal but I am not queueing it while the reserved head tests the other pair-semantics variant; one neutral always-on ranking result plus an untested soft-pair run does not yet justify spending a queue slot on a third ranking variant. shared-ranking-memory-bank stays blocked (shared-bce-ranking is inconclusive, not provisionally accepted).

Unresolved concerns:
- Focal was run at one unswept gamma; the result does not rule out that a smaller gamma, which would perturb gradient scale less, behaves differently, but rerunning a pointwise focusing sweep is not justified by this evidence.
- The plateau/under-convergence reading is inferred from the epoch curve, not from a measured gradient-norm trace; no per-step gradient statistics were recorded for either run.
- Bootstrap uncertainty is conditional on these two fitted models and excludes training-seed variation; a -0.0026 shift overlaps plausible seed noise on fold 0.
- Validation soft-target BCE is not comparable as an objective value across the two runs and is used here only as a calibration direction indicator.
- shared-ranking-memory-bank remains blocked by its unresolved dependency on the inconclusive shared-bce-ranking.

Evidence:
- comparison/comparison.json (shared-balanced-focal-20260920T113454Z-c1fe7f53): paired_report_group_bootstrap.intervals.all delta_mean -0.0025977770373729, 95% [-0.0067976053575522, +0.0015835240647745], 2000 valid draws, 852 report groups
- comparison.json runs.candidate.epochs macro_auc 0.8915/0.8926/0.8940/0.8938/0.8932 for epochs 6-10 vs runs.reference.epochs 0.8914/0.8934/0.8925/0.8966/0.8965
- comparison.json runs.candidate.metrics.soft_target_bce 0.54818 vs runs.reference.metrics.soft_target_bce 0.48566
- comparison.json family_means: meniscus 0.90411 vs 0.91220; oa 0.89154 vs 0.88818; other_eight 0.89201 vs 0.89478
- comparison.json per_target_auc deltas: ACL -0.0153, Medial Meniscus -0.0101, MCL -0.0085, Fracture -0.0058, Lateral OA +0.0070, Synovitis +0.0047
- training/fold_0/selected_checkpoint.json score 0.8939527273178101 at epoch 07 (0-indexed)
- evidence.json timing: training 1545.80 s, setup 0.29 s, prediction 19.00 s, measured_compute 1579.46 s against max_run_seconds 1797
- evidence.json accounting: complete, 10 epochs x 93,846 train triplets / 3,503 visits, validation selection_sha256 92f1341c identical across epochs

## review-20260920T124825Z-c07e8321

The candidate is a faithful, correct implementation of the within-study cross-label contrast term, and it is the first candidate in this campaign to supervise the label axis rather than the study axis. Its optimum coincides with soft BCE's per-pair, verified numerically (gradient exactly 0.0 at s_c−s_c' = logit(y_c)−logit(y_c')), so a change in macro AUC would come from reallocating finite-sample gradient into contrast directions rather than from competing with the pointwise target — which distinguishes it from the completed hybrid, where the ranking term carried 26–45% of the objective against BCE.

Two defined shared rules shape how the outcome must be read. Studies are normalized by their own pair mass and then averaged equally, so a study with mass 1.4 carries the same weight as one with mass 34; this is bounded and deliberate, but it equalizes studies regardless of contrast evidence. Zero-mass studies are zero-filled into the batch mean instead of skipped, so the term's realized scale is diluted by the masked-study fraction; `train_contrast_zero_mass_studies` and `train_contrast_pair_mass_mean` are the traces that quantify both. The idea's own measurement (min per-study straddle mass 1.41 over 3,503 training studies, ~528 pair terms per microbatch) predicts zero-mass cases arise only from heavy masking, so dilution should be small — the logs will confirm it.

Runtime risk is much lower than the prior pair candidate: 66 vectorized pairs with no per-label loop, against the ~52 s (3.3%) that the looped hybrid added.

weight=1.0 and tau=1.0 remain unswept defaults. Read `train_contrast_loss` against `train_bce_loss` before concluding the mechanism failed, as with the previous neutral result.

The candidate implements the assigned mechanism exactly and shares one rule across all labels: a single tau and weight, the same 66-pair triu formula for every label pair, no per-label coefficient, no binarization (contrast.py:12-29 vs ideas.yaml:603-616).

I verified the implementation rather than relying on the developer's claims. File scope is the three allowed files plus the new `training/losses/` package (recursive diff against parent_source; config_differences are only `trainer.loss.contrast.tau/weight`). In lightning.py the only change is inside `training_step` plus the import; data loading, model construction, optimizer/scheduler, `on_before_optimizer_step`, weight averaging, batch-norm refresh, validation, checkpoint selection, and accounting are byte-identical, so validation stays independent of the training objective and seed.

Mask accounting is correct: `target_observed` is a genuine bool tensor (dataset.py:277-280), and `pair_observed = observed[:,l] & observed[:,r]` zeroes a and b for any pair with an unobserved member, matching "c<c' both observed". Soft targets pass through untouched; a=y_c(1−y_c'), b=y_c'(1−y_c) is the Bernoulli straddle mass BCE already implies. Logits are cast to float32 before both terms, and softplus is numerically stable under bf16-mixed.

Static CPU check on B=8, C=12 with ~10% missingness plus one fully-unobserved study: mass 0 for that study, `study_valid` False, zero gradient into its row, no NaN (term is exactly 0, so 0/eps=0), and it is averaged in as 0 — loss 0.8132 = 7/8 of the nonzero mean. The stationary-point claim holds exactly: at s_c−s_c' = logit(y_c)−logit(y_c') the gradient is 0.0, so the term does not pull against soft BCE's pointwise optimum; it only reallocates finite-sample gradient. Zero-logit loss is ln2 scaled by the valid-study fraction (0.6065 here, not the flat 0.69315 the developer reported — their synthetic case differed, not a defect).

Cost is negligible: 66 fully vectorized pairs, no loop, versus the prior candidate's 12-label per-step loop that cost ~52 s; preparation_seconds 0.17. Ample room inside the ceiling.

Leakage: the term uses only in-batch training logits and training targets. No validation labels, no memory bank, no extra passes or preparation.

Unresolved concerns:
- Per-study mass normalization equalizes studies regardless of contrast evidence: a study with straddle mass ~1.4 contributes as much as one with ~34. Bounded and shared, but it shifts effective pressure toward low-evidence studies.
- Zero-mass studies are zero-filled into the batch mean rather than skipped (contrast.py:26-28), so the term's realized scale is diluted by the masked-study fraction and is not directly comparable to the study-axis hybrid at the same weight=1.0.
- tau=1.0 and weight=1.0 are unswept defaults; a neutral outcome will not separate 'label-axis contrast does not help' from 'scale was wrong' without the component traces.
- The objective supervises within-study cross-label ordering while the metric is per-label across-study AUC; any gain is an indirect decorrelation effect, so per-label deltas should be read against the contrast-AUC deficits that motivated the idea.
- Triplet accounting is unavailable at review time (accounting_available=false); the mandatory replay and gates still apply at run time.
- The contrast term carries no class-balance weighting while BCE does, so relative pressure across labels differs between the two components by construction.

Evidence:
- candidate contrast helper: experiments/shared-label-contrast-ranking-20260920T124509Z-e94c32f9/source/src/knee_abnormality/training/losses/contrast.py:12-29
- training_step integration and logging: same source, src/knee_abnormality/training/lightning.py:306-361
- mask dtype (torch.bool) and float32 soft targets: same source, src/knee_abnormality/data/dataset.py:268-280
- baseline masked BCE unchanged: same source, src/knee_abnormality/training/lightning.py:105-120
- recursive diff vs parent_source: only configs/training.yaml, config.py, lightning.py differ, plus new training/losses/ package
- idea mechanism specification: autoresearch/campaigns/losses/ideas.yaml:586-616 (a/b definition, per-study mass normalization, one tau/lambda, stationary point)
- coverage evidence: ideas.yaml:188-193 (min per-study straddle mass 1.41 over 3,503 training studies, 528 pair terms per microbatch)
- static CPU check (this review): fully-unobserved study -> mass 0, valid False, zero grad, no NaN; gradient exactly 0.0 at s_c-s_c'=logit(y_c)-logit(y_c')

## completion-20260920T131216Z-b5f4525a

Confidence-weighted Bernoulli soft-pair ranking (tau 1.0, weight 1.0) degrades fold 0: macro AUC 0.88747 vs incumbent 0.89658, delta −0.00912, 95% paired report-group bootstrap [−0.01357, −0.00477] over 2,000 valid draws; incumbent retained.

The mechanism worked and still lost. `train_pair_zero_mass_labels` = 0.0 in all ten epochs and `train_pair_mass_mean` ≈ 11.67 of 28 microbatch pairs per label, so soft masses gave every label nonzero pair evidence at every step, versus ~10.0/12 covered labels under hard pairs. Full pair coverage is therefore not a route to gains on this recipe; it is a cost.

The soft term also fails to decay: ranking loss 0.6037 → 0.3767 against implied BCE 0.644 → 0.440, holding ~48% → ~46% of the objective, where the hard-pair hybrid fell 45% → 26%. Ambiguous pairs carry irreducible residual, so mass-normalized ranking gradient keeps competing with class-balanced BCE through the final epochs.

Per-label deltas are almost uniformly negative (11 of 12): Baker's −0.0201, Fracture −0.0192, Contusion −0.0180, Lateral OA −0.0174, MCL −0.0131, Lateral Meniscus −0.0119; only PF OA +0.0069 and Effusion +0.0041. Baker's and Fracture reverse sign relative to the hard-pair run, consistent with ambiguous-pair mass dominating confident low-prevalence labels, though per-label mass was not logged separately.

Taken with the inconclusive hard-pair hybrid, always-on within-batch study-axis pairwise ranking has now produced neutral and clearly negative results with non-degenerate, well-covered terms. Further study-axis pair variants have low expected value.

Cost: 1,624.6 s measured (+62 s, ~4.0% over the reference), ~172 s of ceiling headroom.

The recorded evaluation is unambiguous and contract checks pass: seed 3130968547, 4,380 steps, 10 epochs, 938,460 training triplets / 35,030 visits with the pinned cohort hashes, validation 24,188 triplets at an identical per-epoch selection hash, and 1,624.6 s measured compute inside the 1,797 s ceiling. Only the two declared `trainer.loss.pair_ranking` keys differ. So the result is interpretable at face value: candidate 0.88747 vs incumbent 0.89658, delta −0.00912, 95% paired bootstrap [−0.01357, −0.00477] over 2,000 valid draws, strictly below zero.

The review's open questions are now answered by the traces, and they close the mechanism rather than the scale. `train_pair_zero_mass_labels` is exactly 0.0 in every epoch and `train_pair_mass_mean` holds at ~11.66–11.70 of 28 possible microbatch pairs per label, so soft Bernoulli masses did deliver the full 12-label coverage the idea predicted, against the ~10.0/12 realized under hard pairs. Coverage was therefore never the binding constraint, and raising it to complete made macro AUC worse. The ranking component also stayed heavy for the whole run: 0.6037 → 0.3767 against an implied BCE of 0.644 → 0.440, i.e. ~48% → ~46% of the objective, versus 45% → 26% in the hard-pair hybrid. The soft term does not decay because near-zero soft differences carry irreducible residual, so mass-normalized ranking pressure persists and competes with the class-balanced pointwise term late in training.

Per-label deltas are broadly negative (11 of 12 down; Baker's −0.0201, Fracture −0.0192, Contusion −0.0180, Lateral OA −0.0174, MCL −0.0131; only PF OA +0.0069 and Effusion +0.0041), which is the opposite sign pattern from the hard-pair run on Baker's and Fracture. That is consistent with the flagged equalization concern — labels whose batch mass comes mostly from ambiguous interior targets now receive full averaged weight — but per-label pair mass was logged only as a 12-label mean, so that attribution stays a hypothesis.

Queue unchanged; `shared-label-contrast-ranking` is already prepared on the label axis and is unaffected by this study-axis result. I keep `shared-delayed-ranking` as a surplus proposal rather than admitting it: it schedules the same hard-pair rule whose always-on form was neutral with a demonstrably active term, so a transition point adds a free parameter without a mechanism the two results have left open.

Unresolved concerns:
- Per-label pair mass was logged only as a 12-label mean, so the attribution of the largest losses (Baker's, Fracture, Contusion) to ambiguous-pair mass dominance is a hypothesis, not measured evidence.
- Bootstrap uncertainty is conditional on these two fitted models and excludes training-seed variation; the interval is, however, well clear of zero.
- tau=1.0 and weight=1.0 were unswept, so this rejects the implementation at this scale; the traces show the term was active and non-degenerate, but a much smaller weight was not tested.
- shared-ranking-memory-bank stays blocked by its unresolved dependency on shared-bce-ranking (inconclusive); this negative study-axis result further weakens its motivation.
- Reduction differs from shared-bce-ranking (zero-fill into a fixed 12-label mean vs skipping), but with zero-mass labels measured at exactly 0.0 throughout, the two reductions coincided in this run, so the difference does not explain the deficit.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/losses/experiments/shared-bernoulli-pair-ranking-20260920T121717Z-fd69b263/comparison/comparison.json (macro 0.88747 vs 0.89658; per_target deltas)
- training/fold_0/logs/version_0/metrics.csv: train_pair_zero_mass_labels_epoch = 0.0 all epochs; train_pair_mass_mean_epoch 11.63-11.70; train_pair_ranking_loss_epoch 0.6037->0.3767; train_loss_epoch 1.2477->0.8162
- training/fold_0/epochs/metrics_epoch_009.json: val macro_auc 0.88659, soft_target_bce 0.47611 (best epoch 07 = 0.88747)
- changes.diff: src/knee_abnormality/training/losses/pair_ranking.py:24-33 (a=y_i(1-y_j), b=y_j(1-y_i), mass normalization); lightning.py:313-320 reduction
- assignment facts: seed 3130968547, measured_compute_seconds 1624.57 vs max_run_seconds 1797; accounting complete, 938460 triplets / 35030 visits
- Prior comparison for contrast: shared-bce-ranking-20260920T112958Z-a2e23db7 (ranking share 45%->26%, coverage ~10.0/12, delta +0.00073)

## review-20260920T131638Z-9b5c6c9d

Approved shared-balanced-brier for launch. The candidate substitutes `2*(sigmoid(s) - y)^2` for the BCE term while holding the mask, the training-fold class-balance weights, the `targets >= 0.5` weight grouping, and the pooled observed-cell mean reduction exactly fixed, so it isolates the strongly-proper pointwise surrogate as the single changed factor.

The factor 2.0 makes the gradient identical to BCE's at initialization (`4(σ-y)σ'(s)` equals `σ-y` at `s=0`), so the two objectives are matched at step 0 and diverge only as logits move away from zero. This makes the run interpretable without a scale sweep: any macro-AUC difference comes from the curvature difference, not from an arbitrary component weight, which distinguishes it from the neutral hybrid whose weight=1.0 remained a confound.

The mechanism to read afterwards is gradient saturation. Brier's gradient carries the extra `σ'(s)` factor, so confidently wrong cells — disproportionately rare positives on low-prevalence labels late in training — receive vanishing pressure, while BCE's grows. This is the idea's bounded/noise-robust premise and its main AUC risk in one term. `grad_norm` (logged unchanged at lightning.py:323-336) and per-label deltas against prevalence separate "bounded loss helped on noisy soft targets" from "rare positives were starved".

Note `train_loss` now reports Brier values and is not comparable in level to the reference run's BCE curve; only its shape and `grad_norm` are usable cross-run.

Inspected changes.diff and both source trees directly. The candidate is a 2-line call-site swap in lightning.py plus a new `training/losses/` package; every edit is inside the allowlist, config_differences is empty, and `configs/training.yaml`/`config.py` are untouched, so `trainer.class_balanced_bce` and the training-fold weight tensor (`fixed_class_balance` over training rows only, lightning.py:55-101) are reused unchanged — no validation or gold label enters the objective.

Substantive checks, not taken from the author's claim:
- Mask and reduction semantics are byte-for-byte the incumbent's: same empty-observed guard, same `torch.where(targets >= 0.5, w_pos, w_neg)` grouping, same pooled `losses[observed].mean()` (balanced_brier.py:11-21 vs lightning.py:110-118). Label weighting by observation count is therefore identical, which makes this a clean single-factor substitution.
- Soft targets are preserved literally: `(sigmoid(s) - y)^2` is minimized at `sigmoid(s) = y` for any `y` in [0,1]; no implicit hardening.
- Numerics: `training_step` already casts logits to float32 before the loss (lightning.py:306), so sigmoid and the square run in float32 under bf16-mixed, and `(p-y)^2` is bounded in [0,1].
- The `GRADIENT_MATCHING_FACTOR = 2.0` claim holds analytically: d/ds of `2(σ-y)^2` is `4(σ-y)σ'(s)`, equal to BCE's `σ-y` exactly at `s=0` where `σ'=0.25`.
- Protected behavior verified unchanged in the candidate file: data loading, model construction, optimizer/scheduler, `on_train_epoch_start` sampling-epoch wiring, checkpoint state-dict swapping, and `validation_step` (sigmoid-only, no loss, lightning.py:410-422), so validation stays independent of the objective.
- `masked_binary_cross_entropy_with_logits` survives only because `profiling.py:19,65` imports it and that file is outside the allowlist; it is dead on the training path, not a fallback.

Runtime: sigmoid plus a squared term replaces a fused BCE kernel with no per-label loop, so no material addition to the ceiling; preparation was 0.17 s.

Queue unchanged. shared-delayed-ranking stays surplus — this pointwise event bears nothing on the scheduling question, and prior traces already showed its mechanism was not inert. shared-ranking-memory-bank remains unresolved.

Unresolved concerns:
- Brier's gradient carries an extra sigmoid'(s) factor, so confidently-wrong cells lose pressure as training proceeds; rare positives on low-prevalence labels are most exposed. Read per-label deltas against prevalence, not just macro.
- Gradient matching to BCE holds exactly only at logits=0; effective step size on the loss drifts downward over the fixed 10-epoch LR schedule, which is unchanged and cannot compensate.
- train_loss now logs Brier, not BCE, so its level is not comparable to the reference run; only curve shape and grad_norm support cross-run reading.
- Triplet accounting is unavailable at review time (accounting_available=false); the mandatory replay and gates apply at run time.
- masked_binary_cross_entropy_with_logits remains in lightning.py because profiling.py:19 imports it from outside the candidate allowlist; unreachable from the training path, but a residual a future consolidation should remove.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/losses/experiments/shared-balanced-brier-20260920T131336Z-3dfa7fa5/changes.diff
- candidate source/src/knee_abnormality/training/losses/balanced_brier.py:1-21 (new masked_balanced_brier_loss, GRADIENT_MATCHING_FACTOR=2.0)
- parent source/src/knee_abnormality/training/lightning.py:104-118 (incumbent masked BCE: identical guard, weight grouping, losses[observed].mean())
- candidate source/src/knee_abnormality/training/lightning.py:305-321 (single call-site swap; float32 cast and train_loss logging unchanged)
- candidate source/src/knee_abnormality/training/lightning.py:410-422 (validation_step sigmoid-only, no loss term)
- parent source/src/knee_abnormality/training/lightning.py:55-101 (fixed_class_balance / validated_class_balance_weights from training rows, unchanged)
- candidate source/src/knee_abnormality/profiling.py:19,65 (sole remaining importer of the old BCE helper)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/losses/jobs/developer-20260920T131337Z-b2d66c9e/report.json

## proposals-20260920T133029Z-fcd80be3

Verified the researcher's measurements against the artifacts they cite; both reproduce and support their proposals.

Gradient geometry: under balanced BCE the AUC-invariant per-label shift direction holds 23.4% of microbatch squared gradient norm at initialization and 16.7% at the converged proxy, against a 12.5% random floor at batch eight. The excess tracks rarity through all-negative batches (Fracture 52.8% at init, P(no positive in 8) = 0.534; Synovitis 36.3%; MCL 30.1%). The systematic part is small (per-label mean/rms of the weighted residual -0.147..+0.051 at init), so this is mostly eight-sample noise in a direction the metric cannot see — which is what makes projection a variance argument rather than a bias correction.

Weighting geometry: whole-cell class weights leave per-study total mass with cv 0.295 and max/min 3.72, monotone in positive-label count (0 positives 0.72x, 8 positives 1.60x, correlation 0.788); top decile carries 16.5% of mass against 7.1% for the bottom. The alternating fit converges exactly on the fold-0 table (per-study mass cv 0.0000, per-label positive/negative ratio 1.000, weights 0.132-8.785 vs 0.539-6.950, correlation 0.898). Plain per-study normalization without refit is confirmed non-equivalent and worse (ratios 0.569-0.911). At the fixed point weights factor as study-factor times the 2x12 beta, so only a 2x12 table ships and the study factor is recomputed from the batch's own targets — no per-study table, no state, and `class_balance_weight_tensor`'s 2x12 shape (used by non-editable profiling scripts) is preserved.

Closed directions also reproduce: grad_norm medians stay 0.574-0.935 across all ten epochs with an 11.5% clipped share and no trend, so loss-scale self-normalization has no target; and only 47 of 3,503 fold-0 training studies are gold (1.34%), while frozen validation targets agree with the supervising blend on 99.82% of cells, so label-noise-robustness arguments cannot demonstrate benefit on this metric.

Both new proposals are admitted; their supporting measurements reproduce from existing artifacts with no GPU or validation-fitted quantity.

shared-shift-projected-bce ranked above the two standing class-weight ideas. It is the only candidate so far aimed at a component of the incumbent's own gradient that provably cannot move the scored metric (per-label shift direction), it carries no coefficient, temperature or pair pool, so a null result indicts the mechanism rather than an unswept scale — the exact confound that made shared-bce-ranking uninformative about magnitude — and its cost (one masked mean per label per step) is below every prior candidate's overhead, well inside the ~182 s of ceiling headroom. Its uncertainty is honest and stated: output-space energy is not parameter-space progress.

shared-biproportional-weights ranked third, above shared-tempered-balance and shared-soft-mass-balance, because it is parameter-free (nothing to sweep, unlike tempering's exponent), it provably preserves the per-label positive/negative mass property the measured balanced-BCE gain rests on (fitted ratio 1.000 for all twelve labels), and it corrects a measured, never-chosen 2.2x influence advantage for abnormal studies that pushes in the same direction as control B's already over-expressed common abnormality factor. It ranks below the projection because equal study influence is a design preference rather than a metric invariance, and a 27.9% mean change in cell weights is a materially larger perturbation of a validated rule, so a loss is a plausible outcome.

Scope ruling on the question the researcher raised: the biproportional fit reads the joint training-fold label table rather than per-label counts alone, but it yields one shared 2x12 table plus one shared study-normalization rule applied identically to every label. That is shared loss weighting, explicitly in scope; it is not per-label loss selection. No validation or gold label enters the fit.

shared-delayed-ranking stays proposed as surplus: three study-axis pair results (one inconclusive, one reject, one awaiting comparison) make another scheduled ranking hybrid low expected value for a queue slot. shared-ranking-memory-bank remains blocked by its unresolved dependency.

Queue head and the two standing class-weight ideas are otherwise unchanged.

Unresolved concerns:
- shared-shift-projected-bce: output-space gradient energy is not parameter-space progress. The projection also perturbs shared-encoder updates, so removing 17-23% of level noise may not move macro AUC in either direction.
- shared-shift-projected-bce: the head bias gradient goes to ~0 and biases take no weight decay, leaving each label's logit level unanchored. AUC-invariant, but sigmoid nonlinearity makes future gradients level-dependent; per-label mean logit and IQR per epoch must be logged, and saturation drift would be a real failure.
- shared-shift-projected-bce: projection discards the true population level component too, not only sampling noise (class balancing zeroes it exactly only for hard targets at s=0). Intended, but it is not pure variance reduction.
- The converged shift shares were measured on fold-0 validation predictions and labels as a late-training proxy. That is diagnostic only and nothing is tuned on it, but the implementation must contain no validation-derived quantity.
- shared-biproportional-weights: a 27.9% mean change in cell weights (range widening to 0.132-8.785) is a large perturbation of the one weighting rule with a measured gain; equal study influence is a preference, not a metric invariance.
- shared-biproportional-weights: validated_class_balance_weights hard-checks version == 1 (lightning.py:93-101) and the checkpoint hyper_parameters path; a version bump must keep that check, count provenance and the profiling scripts' 2x12 tensor shape consistent, and the shipped weights must be verified to reproduce per-study equality and per-label ratio 1.000 on the training fold before launch.

Evidence:
- outputs/autoresearch/losses/jobs/researcher-20260920T131338Z-3dc7e308/report.json (proposals)
- outputs/autoresearch/losses/jobs/researcher-20260920T131338Z-3dc7e308/scratch/gradient-geometry-20260920.md sections 1-4
- outputs/autoresearch/losses/jobs/researcher-20260920T131338Z-3dc7e308/scratch/shift.py:41-50 (shift-energy estimator), shift.txt
- outputs/autoresearch/losses/jobs/researcher-20260920T131338Z-3dc7e308/scratch/sinkhorn.txt (fit converges: study mass cv 0.0000, per-label ratio 1.000, weights 0.132-8.785, corr 0.898)
- outputs/autoresearch/losses/jobs/researcher-20260920T131338Z-3dc7e308/scratch/study_mass.txt (per-study mass cv 0.295, max/min 3.72, corr 0.788 with positive count; common-factor AUC 0.694-0.835)
- baseline/source/src/knee_abnormality/training/lightning.py:60-102 (fixed_class_balance, class_balance_weight_tensor, version check)
- baseline/source/scripts/pipeline/train.py:148-172 (class_balance produced in-process and serialized; no script edit required)
- researcher-20260920T121718Z-2ea25138/scratch/contrast-evidence-20260920.md section 1 (PC1 44.7% of prediction variance vs 27.8% of target variance)

## completion-20260920T134734Z-41741044

Shared within-study cross-label contrast beside balanced BCE at weight 1.0, tau 1.0 is harmful on fold 0: macro AUC 0.89024 vs incumbent 0.89658, delta mean −0.00635, 95% paired report-group bootstrap [−0.01051, −0.00220] over 2,000 valid draws.

The deficit is structural, not a checkpoint-selection accident. Validation macro AUC trailed the incumbent at every epoch (ep3 0.8733 vs 0.8820; ep8 0.8902 vs 0.8966), and the candidate's own BCE component ended only slightly worse (0.6669 → 0.4491 vs the incumbent's 0.6438 → 0.4413), so ~46–48% of the objective was spent on label-axis ordering for a small pointwise cost and a consistent across-study ranking loss. Ten of twelve labels regressed: MCL −0.0221, ACL −0.0145, Lateral OA −0.0116, Fracture −0.0098, Baker's −0.0093, Lateral Meniscus −0.0079, Contusion −0.0076, Medial Meniscus −0.0038, Medial OA −0.0020, Synovitis −0.0005; only PF OA +0.0093 and Effusion +0.0036 gained. The losses concentrate on low-positive-count labels (MCL 134, ACL 184, Fracture 63) and the gains on the two highest-count labels (PF OA 406, Effusion 525), consistent with the unweighted contrast term diluting the class-balanced pressure that the incumbent gives rare positives.

The idea's own motivation moved the wrong way. OA lateral cross-talk rose 0.6220 → 0.6905 and oa margin_auc fell 0.8661 → 0.8556 (bootstrap oa_margin −0.0105, wide); meniscus cross-talk improved slightly 0.6364 → 0.5957 with margin_auc flat (0.97125 → 0.97091). Enforcing within-study cross-label ordering therefore did not reduce the cross-label confusion it targeted on the family with the largest deficit.

Campaign-level reading: both pair-based objectives are now measured (study-axis neutral, label-axis negative) at ~2–3% runtime cost each, and further within-batch pair variants are low value relative to the weighting and gradient-geometry queue.

Contract checks pass, so the fixed reject verdict stands on evidence, not on a defect: config differences are only trainer.loss.contrast.{weight,tau}; accounting is complete and exact (938,460 triplets / 35,030 visits, train cohort b536b7e1…, validation cohort 7167eba1… with the identical selection hash 92f1341c… in all ten epochs, 4,380 steps); seed 169933627 is fresh; prep+compute 1,594.4 s is inside the 1,797 s ceiling (+2.0% over the reference's 1,562.3 s, far cheaper than the looped hybrid's +3.3%).

The traces close the two open review concerns, so the negative result is mechanism evidence rather than a scale artifact: train_contrast_zero_mass_studies is exactly 0.0 in every epoch (no zero-mass dilution), train_contrast_pair_mass_mean is ~26.8 per study, and the contrast term held ~46–48% of the objective from epoch 0 to 9 (0.6058/1.2726 → 0.3858/0.8349). The term was active at high share for the whole run and still cost macro AUC.

I decline to admit shared-delayed-ranking even though its dependency is now resolved. Its only new degree of freedom is the transition point of the same within-batch pair machinery, and the campaign has now measured that machinery twice: study-axis ranking neutral with an active component, label-axis contrast harmful at a comparable share. Neither direction shows a positive signal that a schedule could time better, so its expected value sits below the untested weighting and gradient-geometry ideas already queued. I am not requesting a downweighted contrast variant either: reducing the weight can at best approach neutrality, and the compartment diagnostic below contradicts the idea's decorrelation premise rather than its calibration.

Queue membership and order are unchanged. shared-ranking-memory-bank stays proposed on its unresolved dependency.

Unresolved concerns:
- The decorrelation premise is contradicted, not merely unconfirmed: OA lateral cross-talk worsened 0.6220 -> 0.6905 and oa margin_auc fell 0.8661 -> 0.8556 under a term designed to separate co-occurring labels.
- Harm concentrates on low-positive-count labels (MCL -0.0221, ACL -0.0145, Fracture -0.0098) while the two highest-count labels gained, so the unweighted contrast term plausibly offsets the incumbent's class balancing; this interaction is inferred from the outcome, not directly measured.
- weight=1.0 and tau=1.0 remain unswept, so the result establishes harm at ~46-48% objective share; it does not exclude a small-weight variant being merely neutral. I judge that not worth a run.
- Bootstrap uncertainty is conditional on these two fitted models and excludes training-seed variation; the interval clears zero but its lower magnitude (-0.0022) is within seed-scale noise.
- Per-study mass normalization (equal weight to studies of mass ~1.4 and ~34) is confirmed never to hit the zero-mass path but its evidence-equalizing effect remains unseparated from the term itself.

Evidence:
- comparison/comparison.json in outputs/autoresearch/losses/experiments/shared-label-contrast-ranking-20260920T124509Z-e94c32f9: per_target deltas, family_means, compartments (oa/meniscus cross_talk, margin_auc), paired_report_group_bootstrap (852 groups, 2,000 draws, seed 20260914)
- training/fold_0/logs/version_0/metrics.csv (same experiment): train_contrast_loss_epoch 0.6058 -> 0.3858, train_loss_epoch 1.2726 -> 0.8349, train_bce_loss_epoch 0.6669 -> 0.4491, train_contrast_pair_mass_mean_epoch 26.75, train_contrast_zero_mass_studies_epoch 0.0 in all ten epochs
- outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat/fold_0/logs/version_0/metrics.csv: incumbent train_loss_epoch 0.6438 -> 0.4413 and per-epoch val_macro_auc used for the epoch-by-epoch comparison
- training/fold_0/best-epoch=08-val_macro_auc=0.89024.ckpt and training/fold_0/triplet_accounting.json (complete; selection hashes per epoch, validation selection 92f1341c… identical across epochs)
- assignment facts: seed 169933627, preparation 0.170 s, measured_compute 1594.186 s vs max_run_seconds 1797 and historical 1562.283 s; config_differences limited to trainer.loss.contrast.tau/weight
- jobs/orchestrator-20260920T124825Z-d4f12086/report.json (implementation review whose zero-mass and scale concerns the traces now resolve)
- jobs/orchestrator-20260920T121537Z-6ee62f96/report.json (study-axis hybrid inconclusive result used for the pair-family comparison)
- autoresearch/campaigns/losses/ideas.yaml:59-72,131-137 (prior reasons for retaining shared-delayed-ranking as surplus)

## review-20260920T135625Z-b5de72e8

Verified mechanism: subtracting the detached linear term sum_c gbar_c * sum_{i obs} s_ic / N_obs makes the per-label observed gradient sum exactly zero, i.e. per-label batch-centering of the weighted residual, removing the AUC-irrelevant logit-level direction while leaving the forward BCE value intact.

A consequence not stated in the developer report: every parameter whose effect on logits is a per-label constant across the batch now receives exactly zero gradient. That includes the 12 output biases (`readout_out_bias`, initialized to zeros at modeling/model.py:136, and `head.bias` at model.py:225), which stay at their initialization for all ten epochs. This is AUC-neutral by construction but freezes calibration, so `train_balanced_bce` will sit systematically above the control's BCE curve even if ranking improves; its level is not evidence of worse fit. `train_loss` now reports balanced_bce minus the detached correction and can be negative — only `train_balanced_bce` (bit-identical formula to control) and per-label diagnostics are cross-run readable, and `grad_norm` now measures the projected gradient, so its level is likewise not comparable to control.

`accumulate_grad_batches: 1` (configs/training.yaml:100), so gbar_c is estimated from at most 8 observed cells per label with no averaging across accumulation. In my check at realistic observation density (5-7 observed cells per label) the removed energy share was 1-12% per label, but it is exactly 1.0 when a label has one observed cell in the microbatch, which fully cancels that cell's gradient. `train_removed_energy_share_<target>` is the diagnostic that decides whether the projection acted as mild variance reduction or discarded a large share of sparse labels' gradient signal.

Interpretation to read afterwards: a win supports removing the level direction as variance reduction; a loss most likely means the per-label mean residual carries ranking-relevant signal for shared encoder/attention parameters at batch 8, not that AUC shift-invariance is wrong.

EarlyStopping and ModelCheckpoint monitor `val_macro_auc` (scripts/pipeline/train.py:217,233), so the changed training-loss semantics cannot affect checkpoint selection or stopping.

Approved shared-shift-projected-bce. The diff touches only training_step, two new per-epoch diagnostic hooks, and two new files under training/losses/; data loading, model construction, optimizer/scheduler, forward prediction, validation_step, checkpoint save/load, accounting, and sampling are byte-identical to parent_source, and config.py / configs/training.yaml are unchanged (facts: config_differences=[], candidate_config_changes={}), so the objective is the single changed factor and the candidate is hyperparameter-free.

I did not rely on the developer's numerical claims. Independent autograd check (/tmp/chk_shift.py, random 8x12 batch with a fully-unobserved label, a single-observed-cell label, and asymmetric class weights): forward balanced_bce is allclose to the parent masked_binary_cross_entropy_with_logits; gradient equals the closed form (g_ic − gbar_c)/N_obs with max abs diff 0.0; per-label observed gradient sums are 0 (max 1.9e-9); unobserved cells get exactly 0; the single-observed-cell label gets exactly 0. Mask accounting and soft-target semantics are preserved (weights still grouped by targets >= 0.5, pooled observed-cell mean reduction, float32 via the existing `.to(torch.float32)` cast). No leakage: the correction and diagnostics are computed under no_grad from the current microbatch only, no persistent state, no validation labels, no extra forward pass.

Added cost is one elementwise sigmoid plus O(1) per-label reductions per step, an in-epoch logit buffer of ~170 KB (438 steps x 8 x 12 float32, cleared each epoch), and one torch.quantile per epoch (10 total) — immaterial against the 1,797 s ceiling; preparation_seconds 0.22.

I did not admit shared-delayed-ranking despite satisfied dependencies. The completed hybrid's traces showed the ranking term was active and a large objective share throughout, so a schedule variant of the same within-batch pair term has low expected value relative to the four queued weighting/normalization ideas; it is retained as proposed rather than queued. Queue membership and order are unchanged.

Unresolved concerns:
- gbar_c is estimated from at most 8 observed cells per label (accumulate_grad_batches=1), so the projection subtracts a noisy per-label mean; removed energy share was 1-12% at typical density but exactly 1.0 for a label with a single observed cell in the microbatch, zeroing that cell's gradient entirely.
- The 12 output biases receive exactly zero gradient and remain at initialization for all ten epochs. AUC-neutral by design, but calibration is frozen, so train_balanced_bce will read systematically higher than control without implying worse ranking.
- train_loss now logs balanced_bce minus a detached linear term and can go negative; grad_norm measures the projected gradient. Neither level is comparable across runs, only train_balanced_bce and the new per-label diagnostics.
- Triplet accounting is unavailable at review time (accounting_available=false); the mandatory replay and gates apply at run time.
- The per-epoch diagnostic buffers (_epoch_logits and friends) are created only in on_train_epoch_start, not __init__; safe under the normal hook order, but a resume path that reached on_train_epoch_end without a train-epoch start would raise AttributeError.
- masked_binary_cross_entropy_with_logits remains defined at lightning.py:105 because profiling.py imports it from outside the candidate allowlist; unreachable from the training path, but still a residual for a later consolidation.

Evidence:
- Diff: /home/claudio/src/knee-abnormality/outputs/autoresearch/losses/experiments/shared-shift-projected-bce-20260920T135020Z-231a530c/changes.diff
- New loss: .../shared-shift-projected-bce-20260920T135020Z-231a530c/source/src/knee_abnormality/training/losses/shift_projection.py:1-41 (correction term at lines 36-41)
- Changed training path: .../source/src/knee_abnormality/training/lightning.py:305-372 (training_step, on_train_epoch_start/end); grad_norm logging unchanged at lightning.py:372-382
- My independent autograd verification script: /tmp/chk_shift.py — grad max diff 0.0, per-label grad sums max 1.9e-9, unobserved grad 0.0, single-observed-cell grad 0.0, bce allclose to parent
- Microbatch/accumulation: .../source/configs/training.yaml:99-100 (batch_size 8, accumulate_grad_batches 1)
- Frozen-bias basis: .../source/src/knee_abnormality/modeling/model.py:136 (readout_out_bias zeros) and model.py:225 (head.bias)
- Selection independence: .../source/scripts/pipeline/train.py:217,232-233 (ModelCheckpoint and EarlyStopping monitor val_macro_auc)
- Developer report: /home/claudio/src/knee-abnormality/outputs/autoresearch/losses/jobs/developer-20260920T135020Z-dcfdee2e/report.json

## completion-20260920T141656Z-f5f29c1c

Balanced Brier (2*(σ(s)−y)², BCE-matched gradient at s=0) is not an improvement over balanced BCE on fold 0: macro AUC 0.89417 vs 0.89658, delta mean −0.00240, 95% paired report-group bootstrap [−0.00635, +0.00149] over 2,000 valid draws. The interval includes zero, so the incumbent is retained, but unlike the ranking hybrid the point estimate is negative and 9 of 12 labels are worse.

Contract facts verified: config_differences empty, candidate diff limited to lightning.py:308 plus the new losses package (changes.diff), accounting complete at 938,460 training triplets / 35,030 visits and the pinned cohort hashes, 4,380 steps, compute 1,592.7 s against the 1,797 s ceiling.

The predicted saturation mechanism is visible in the traces. Control grad_norm rises through late training (0.641 at epoch 4 → 0.884 at epoch 8, 0.761 at 9) while Brier's stays flat and lower (0.568 → 0.600 → 0.691). Brier therefore applied less pressure exactly where BCE increases it on confidently-wrong cells, which is the intended bounded behavior and the most plausible source of the mild macro loss.

Per-label deltas do not order by prevalence, so "rare positives starved" is not cleanly supported: Baker's +0.0080, Effusion +0.0055, PF OA +0.0015; Synovitis −0.0081, Medial Meniscus −0.0068, Fracture −0.0066, Lateral Meniscus −0.0065, Contusion −0.0051, MCL −0.0045, ACL −0.0035, Medial OA −0.0020, Lateral OA −0.0009. Notably Baker's and Effusion were also the top gainers under the ranking hybrid, which suggests label-level seed noise rather than a mechanism-specific tilt.

Calibration was essentially preserved: validation soft-target BCE 0.48811 vs control 0.48566, so the non-log surrogate did not degrade probability quality in any practically relevant way.

Campaign-level read: two changes to objective shape (pairwise ranking addition, pointwise curvature) both landed inside noise, the second with a negative point estimate. Cell weighting, not loss shape, is the more promising remaining axis.

Queue membership and order stay as-is; the reserved head (shared-shift-projected-bce) is already prepared and this result does not bear on the three class-weighting ideas behind it.

shared-delayed-ranking has satisfied dependencies but I am not admitting it. The evidence against it strengthened: the always-on hybrid was neutral with a demonstrably active ranking term, and this run shows that changing the pointwise surrogate's curvature at matched initial gradient also fails to move macro AUC upward. Two objective-shape changes in a row landed inside noise, so schedule variants of an already-neutral component are lower value than the untried class-weighting axis (biproportional, tempered, soft-mass) that alters which cells dominate the gradient rather than how the same cells are shaped. It stays available as a surplus proposal if the weighting axis also comes back flat.

shared-ranking-memory-bank remains unresolved (its dependency shared-bce-ranking is inconclusive, not provisional_accept); it is omitted from both lists and stays proposed.

Unresolved concerns:
- Bootstrap uncertainty is conditional on the two fitted models and excludes training-seed variation; a −0.0024 shift with 9/12 labels down is suggestive but not separable from seed noise on fold 0.
- The saturation mechanism is inferred from grad_norm levels across runs with different loss scales; the comparison is directional (flat vs rising late) rather than a calibrated magnitude.
- Baker's and Effusion improved in both this run and the ranking hybrid while both candidates were macro-neutral-to-negative, which points to label-level seed variance and warns against reading per-label wins as mechanism evidence.
- masked_binary_cross_entropy_with_logits remains defined in lightning.py because profiling.py:19 imports it from outside the candidate allowlist; unreachable from training, but a residual for a later consolidation outside this campaign.
- shared-ranking-memory-bank stays blocked by the unresolved shared-bce-ranking dependency; its stale-score and refresh handling remain unreviewed.

Evidence:
- outputs/autoresearch/losses/experiments/shared-balanced-brier-20260920T131336Z-3dfa7fa5/comparison/comparison.json — macro 0.8941694 vs 0.8965817, per-target AUCs, soft_target_bce 0.4881090 vs 0.4856632, selected_epoch 9
- same experiment changes.diff — lightning.py:308 call swap plus src/knee_abnormality/training/losses/{__init__,balanced_brier}.py; no other files touched
- same experiment training/fold_0/logs/version_0/metrics.csv — epoch-mean grad_norm 0.673, 0.698, 0.558, 0.604, 0.568, 0.591, 0.552, 0.664, 0.600, 0.691; train_loss_epoch 0.306 → 0.135
- outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat/fold_0/logs/version_0/metrics.csv — control grad_norm 0.833, 0.711, 0.649, 0.708, 0.641, 0.718, 0.796, 0.882, 0.884, 0.761
- assignment facts: measured_compute_seconds 1592.697 (ceiling 1797), config_differences [], accounting.complete true, seed 1224364060
- outputs/autoresearch/losses/jobs/orchestrator-20260920T131638Z-73c938b4/report.json — pre-launch review predicting the saturation signature now observed
- outputs/autoresearch/losses/jobs/orchestrator-20260920T121537Z-6ee62f96/report.json — shared-bce-ranking completion, per-label gainers Baker's/Effusion

## review-20260920T142358Z-d8acebe1

The IPF fit is well-posed and converges fast at this scale, but it carries a mechanism consequence the developer report does not state: per-study mass normalization does not preserve equal total mass across labels. Reproducing the exact fit code on a 3503x12 synthetic table with realistic masking and prevalences 0.005-0.6, 200 sweeps give row residual 3.6e-15 and per-label positive/negative mass ratio deviation 4.4e-16, and the mean weight over observed cells is exactly 1.0 (so the global loss scale matches control, whose per-label mean weight is also exactly 1 by construction). However each label's total weighted mass divided by its observed count ranges from 0.068 at prevalence 0.005 to ~2.5 at prevalence 0.5. Rare labels get systematically down-weighted relative to control: their large positive betas inflate a study's row mass, and the study factor then shrinks every weight in that row.

This is in-scope (one shared formula over training-fold counts, no per-label choices), but it means the run tests two coupled effects at once: equalizing study influence and a strong re-tilt of relative label emphasis toward prevalent labels. Under a macro-AUC criterion that weights all twelve labels equally, the expected failure mode is degradation concentrated on the rarest labels. Read per-label deltas against prevalence when the result arrives: a loss ordered by prevalence points at the label-mass tilt rather than at study normalization being wrong, and would motivate a variant that restores per-label mass after the fit.

Fitted betas are extreme but representable in float32 (~76 down to ~0.013 in the synthetic at prevalence 0.005), so no precision concern at realistic skew.

Diff is confined to `src/knee_abnormality/training/lightning.py` (`diff -rq` vs parent_source shows no other file differs); `configs/training.yaml` keeps max_epochs 10, batch_size 8, accumulate_grad_batches 1, class_balanced_bce true. Fit input is `data.train_dataset.studies` at scripts/pipeline/train.py:149, i.e. the fold!=0 table (data/labels.py:178), so no fold-0 validation or gold labels enter the weights; the call site is unmodified. Weights are recomputed each run and persisted to `class_balance.json` (train.py:171) and into checkpoint metadata, so the fitted beta table is auditable post-run.

Correctness of the key identity: the train-time per-study factor recomputed at lightning.py:143-148 reproduces the fitted a_i exactly, because each batch item carries the study's full 12-vector of soft targets and its own observed mask (data/dataset.py:268-280), which are the same rows used in the fit. So a_i is not batch-composition dependent and gradient accumulation cannot perturb it. Masked cells are zeroed before the row sum and then dropped by `losses[observed].mean()`; soft targets and the y>=0.5 grouping rule are unchanged in both fit and loss. The NaN guard for a fully-unobserved row (safe_row_mass) is correct and does not alter any valid case. BCE-with-logits returns float32 under autocast and the weight buffer is float32, so the weighted product stays float32.

No repair requested: the two things I would otherwise ask for are already obtainable — convergence and realized per-label mass are exactly recomputable offline from the persisted betas and the fixed training table.

Queue unchanged. I did not admit `shared-delayed-ranking` despite its satisfied dependency: the completed hybrid's traces showed the ranking term was active and non-degenerate yet neutral, so scheduling that same term remains low expected value; it stays available as surplus rather than displacing queued weighting ideas.

Unresolved concerns:
- Per-label total weighted mass is no longer equal across labels (0.068x to ~2.5x of observed count across prevalence 0.005-0.6 in the reproduced fit), so the run confounds study-influence equalization with a prevalence tilt of label emphasis under an equal-weight macro AUC criterion.
- The 200 sweeps are a fixed count with no convergence assertion; convergence was verified only on synthetic tables (developer's and mine), not the actual training fold. Recomputable offline from the persisted class_balance.json betas, so not blocking.
- class_balance version bumped 1 -> 2 with beta values stored in the positive_weight/negative_weight fields; those fields no longer mean 1/(2p), so any cross-run comparison of class_balance.json against control is not like-for-like.
- Triplet accounting is unavailable at review time (accounting_available=false); the mandatory replay and gates still apply at run time.
- Bootstrap uncertainty for the eventual comparison stays conditional on the fitted models and excludes training-seed variation.

Evidence:
- changes.diff, outputs/autoresearch/losses/experiments/shared-biproportional-weights-20260920T141757Z-d567131d/changes.diff (only lightning.py differs; confirmed by diff -rq against baseline/source)
- src/knee_abnormality/training/lightning.py:58-104 (biproportional fit), :121-128 (version 2 validation), :138-149 (per-study factor in the loss)
- scripts/pipeline/train.py:148-150 fit uses data.train_dataset.studies; :171 persists class_balance.json
- src/knee_abnormality/data/labels.py:165-178 training table excludes fold 0; observed_ derived from target nullity
- src/knee_abnormality/data/dataset.py:268-280 per-item full 12-vector target and observed mask, so train-time a_i equals the fitted a_i
- configs/training.yaml:98-100,108 max_epochs 10, batch_size 8, accumulate_grad_batches 1, class_balanced_bce true
- /tmp/ipf_check.py reproduction: row residual 3.6e-15, class-ratio deviation 4.4e-16, mean observed weight 1.0, per-label mass/observed count 0.068..2.516
- developer report /home/claudio/src/knee-abnormality/outputs/autoresearch/losses/jobs/developer-20260920T141757Z-bf797d8e/report.json (NaN-row backward hazard and its fix)

## proposals-20260920T143428Z-31498a3d

Reproduced the key sizing arithmetic: the reported (var_pos+var_neg)/gap^2 of 0.39-0.84 is exactly 1/d'^2 for d' 1.09-1.60, so it is a restatement of d', not independent evidence. Two consequences for the variance idea's sizing claim: (a) the projected 12-17% objective share assumes the term's numerator equals the unweighted variance sum, but the term applies the existing class weights, which heavily upweight rare positives, so the realized share can exceed the estimate and must be read from the logged per-epoch share rather than assumed; (b) the denominator floor at 1.0 binds precisely when the class gap is below one logit, i.e. early in training before separation develops. In that regime the normalization is inactive and uniform logit-scale shrinkage - which is AUC-neutral - genuinely reduces the term, so the degenerate escape route the design intends to close is open during warm-up. The logged EMA gap and d' traces do separate compression from shrinkage after the fact, but the floor choice and the early-training behavior are review points for the implementation, not settled by the proposal.

The DRW closure in the researcher's section 6 is well supported for weighting schedules: three class-balanced runs lead two ordinary-BCE runs at every epoch including epoch 1 (.8079/.8057/.7972 vs .7986/.7859), with the noted preprocessing and seed confounds on the two references. It does not by itself close ranking-term schedules; shared-delayed-ranking is held for the separate reason above.

Admitted both new proposals (dependencies clear per proposal_admission) and demoted the two remaining weighting ideas below them.

Reranking reason: the new geometry evidence identifies a deficit structure that the queued weighting rules provably cannot reach. Within each metric-defined group the score is aligned with the soft target (corr +0.17..+0.50 negatives, +0.20..+0.55 positives), and the v4 blend pins report-unaddressed metric-negatives at 0.25, so a cell's pointwise equilibrium is set by its target, not its weight. 78-99.6% of each label's discordant pairs involve a bottom-decile positive or top-decile negative, and moment-implied macro AUC 0.90583 exceeds realized 0.89658, placing the loss in group tails. shared-tempered-balance and shared-soft-mass-balance stay queued (cheap, untested, and the argument is validation-side inference rather than measurement) but rank last.

Both new ideas are group-relative moment terms, not repeats of the three measured pair enumerations or the two pointwise reshapings, and they reach every observed cell every step where pair coverage was 0.450 (Fracture) to 0.737 (MCL) per step. Variance compression ranks above the centroid margin because the separation axis has already been probed twice (inconclusive, reject) while within-group compression is untested; they share centroid state but are the two halves of the same moment form and must not be combined in one run.

shared-delayed-ranking stays proposed: shared-bce-ranking's traces show the ranking term was active and substantial throughout (~45% -> ~26% of the objective) yet neutral, so scheduling the same term later has low expected value. shared-ranking-memory-bank remains blocked by its unresolved dependency.

Unresolved concerns:
- Variance-term denominator floored at 1.0 is inactive whenever the class gap is under one logit (early training), leaving uniform logit-scale shrinkage - AUC-neutral - as a real minimizer in that regime; the floor and warm-up behavior need explicit review at implementation.
- Projected 12-17% objective share for the variance term is derived from unweighted class-conditional variances, but the term carries the class weights; realized share may be larger and must come from the per-epoch log.
- All geometry (tail concentration, target/score coupling, d') is measured on validation predictions; no training-fold predictions are saved, so the training-time tail structure the terms actually act on is unmeasured.
- The oracle probes (0.9410 shrink, 0.9329 winsorize) apply validation labels at inference and bound nothing achievable; they must not be read as an effect-size estimate for either idea.
- The two new ideas share EMA centroid state and are the two halves of one moment form; running them together would confound compression against separation.
- Both queued weighting ideas are demoted on an analytic argument about soft-target equilibria, not on a measured null; their retention reflects that the argument is validation-side.

Evidence:
- jobs/researcher-20260920T141757Z-27ab0eb1/scratch/group-geometry-20260920.md sections 1-4 (tail concentration, within-group target/score correlation, moment-implied 0.90583 vs realized 0.89658, microbatch pair coverage 0.450-0.737 and active-band shares 8.1/13.5/30.3%)
- Same scratch dir, sizing.py: ratio (var_pos+var_neg)/gap^2 is computed as the reciprocal of d'^2; active-band shares are validation proxies
- Same scratch dir section 6: per-epoch validation macro AUC table, three balanced vs two ordinary-BCE runs, with the stated preprocessing/seed confounds
- Same scratch dir section 2 and docs/public-labels/uncertainty-and-confidence-encoding.md: v4 maps report-unaddressed cells to 0.25, fixing the pointwise equilibrium independently of cell weight
- recent_results shared-bce-ranking-20260920T112958Z-a2e23db7: ranking component 45% -> 26% of objective, ~10/12 labels and ~118 pairs per step, delta +0.00073 [-0.00329, +0.00479], +52 s (+3.3%) runtime
- proposal_admission: shared-ranking-memory-bank unresolved on shared-bce-ranking (completed, inconclusive)

## completion-20260920T144423Z-42a3ee70

shared-shift-projected-bce: macro AUC 0.89250 vs incumbent 0.89658, delta mean −0.00406, 95% paired bootstrap [−0.00867, +0.00062] over 2,000 valid draws. Interval includes zero, but the whole mass sits below the control except a sliver; this is a directionally negative result, not a neutral one.

The decisive diagnostic settles the question left open at review. Epoch-mean `train_removed_energy_share` is 0.15–0.68, far above the 1–12% predicted for realistic observation density: Fracture 0.68, MCL 0.44, Contusion 0.41, Lateral Meniscus 0.35, down to Medial Meniscus 0.17 and PF OA 0.17. At microbatch 8 with accumulate_grad_batches=1 the per-label mean is estimated from very few observed cells, so subtracting it discards a large fraction of the gradient for sparse labels rather than acting as mild variance reduction. Per-label deltas track this: Spearman between removed-energy share and delta is −0.63, with the two worst labels being the two highest shares (Fracture −0.0248, MCL −0.0246); the two clear gains are low-share, prevalent labels (Effusion +0.0119 at 0.19, ACL +0.0059 at 0.24). Synovitis (−0.0153 at share 0.25) is the main off-trend label.

Fit and trajectory agree: candidate `train_balanced_bce` ends at 0.4691 against the control's bit-identical 0.4413, and validation peaked at epoch 6 (0.8925) where the control kept improving to 0.8966 at epoch 8.

Read this as a result about the estimator, not about AUC shift-invariance: centering the weighted residual per label is exact in expectation but destructive at this batch size. Any successor would need a far lower-variance estimate of the level direction (e.g. accumulated or running), which the fixed recipe does not offer.

Runtime 1,586.2 s against the 1,797 s ceiling (+1.5% over the reference run's 1,562.3 s); accounting complete and matching the pinned cohort, visits, triplets and 4,380 steps; seed 2615871332; no config differences.

Queue change (one demotion): shared-group-variance-compression moves from second to last. It rests on the same premise this run tested for the first time — that score structure invisible to AUC (here the per-label level, there the within-group spread) can be suppressed without cost. The premise did not hold: the projection removed 15–70% of per-label gradient energy, not the 1–12% expected at review, and the per-label damage tracks that share (Spearman −0.63). A class-conditional variance term suppresses a second gradient direction under the same batch-8 estimation noise, so its prior drops relative to the three weighting ideas, which are untested by this evidence and cheap. Not parked: the mechanism is distinct enough that one full-budget test remains worthwhile if the weighting line exhausts.

shared-delayed-ranking is admissible (no unresolved dependency) but stays unqueued as retained surplus rather than displacing a weighting idea: shared-bce-ranking already showed an active, non-degenerate ranking component is neutral at full-run scale, and a schedule variant of that same term has lower expected value than the untested weighting formulas. shared-ranking-memory-bank remains blocked by its unresolved dependency on shared-bce-ranking (inconclusive, not provisional_accept).

shared-centroid-margin is unaffected by today's result and keeps its place as the one ranking-pressure idea with a mechanism (dense per-cell pressure) the sampled-pair run did not test.

Unresolved concerns:
- The negative direction is not established: the interval crosses zero and bootstrap uncertainty is conditional on these two fitted models, excluding training-seed variation. Treat 'projection removes too much gradient at batch 8' as supported by the energy diagnostic, not by the macro delta alone.
- The share-vs-delta Spearman of −0.63 over 12 labels is suggestive, not conclusive; Synovitis (share 0.25, −0.0153) does not fit the pattern.
- The review's 'frozen output biases' concern is empirically immaterial here: residual_readout is off in this config (no readout_out_bias in the checkpoint), and the surviving head.bias magnitudes (max 0.103) are comparable to control's (max 0.071), so no calibration divergence is visible in the artifacts.
- train_loss and grad_norm remain cross-run unreadable for this run; only train_balanced_bce and the per-label diagnostics were used above.
- masked_binary_cross_entropy_with_logits is still defined but unreachable in this source (imported by profiling.py, outside the allowlist); a residual for later consolidation, not a contract issue.

Evidence:
- comparison/comparison.json in experiments/shared-shift-projected-bce-20260920T135020Z-231a530c (per_target, paired_report_group_bootstrap)
- training/fold_0/logs/version_0/metrics.csv: train_removed_energy_share_<target>, train_balanced_bce_epoch, val_macro_auc (candidate)
- outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat/fold_0/logs/version_0/metrics.csv: train_loss_epoch (control balanced BCE), val_macro_auc
- training/fold_0/best-epoch=06-val_macro_auc=0.89250.ckpt state_dict model.head.bias vs control best-epoch=08 checkpoint
- src/knee_abnormality/modeling/model.py:136,225 (readout/head bias definitions; residual_readout inactive in this config)
- configs/training.yaml:100 accumulate_grad_batches: 1
- prior orchestrator review orchestrator-20260920T135626Z-7bf18c0f/report.json (energy-share expectation of 1–12%)
- orchestrator-20260920T121537Z-6ee62f96/report.json (shared-bce-ranking neutral with an active ranking component)

## review-20260920T145225Z-7e3ee664

Two implementation details refine how the eventual result should be read.

Readiness gating is stricter than the idea text: a label contributes only when both centroids are initialized (centroid_margin.py:37), whereas positives need only centroid_neg. For the rarest labels the first positive arrives after tens of steps, so a small number of early steps carry no term for those labels; the logged missing-group-step counters make this measurable and the effect is negligible against 4,380 steps.

Anchor staleness is asymmetric by prevalence. centroid_neg updates essentially every step, so the positive-cell term is well anchored. centroid_pos updates only on steps where that label has an observed positive; at low prevalence that is a small fraction of steps, so with momentum 0.9 the positive centroid lags the drifting logit scale by a long effective window for rare labels. The negative-cell term for rare labels is therefore measured against a stale threshold, particularly in early epochs while logit scale moves fastest. The per-label missing_pos_steps counter gives the realized update count and the per-epoch centroid_pos/neg/gap traces give the drift, so this is diagnosable from the run's own logs; read a rare-label-concentrated degradation against those traces before attributing it to the margin mechanism itself.

Logging caveat: active_cell in centroid_margin.py:34 counts cells whose hinge is positive without applying label_valid, so the reported active-band share includes labels not yet contributing to the loss. This inflates the early-epoch share slightly and does not affect the objective.

The diff implements the assigned mechanism as specified and preserves the contract. Edits are confined to the allowlist (configs/training.yaml, config.py, lightning.py, new src/knee_abnormality/training/losses/); the only config differences are the three trainer.loss.centroid_margin keys, and class_balanced_bce is untouched.

Protected behavior in lightning.py is intact: data loading, model construction, forward, optimizer/scheduler, on_before_optimizer_step, on_train_batch_end weight averaging, validation_step, on_save/on_load_checkpoint and accounting are unmodified. The new on_train_epoch_end does not override a baseline hook (baseline has none, lightning.py:299-405). Buffers are registered on the module (not self.model), so the averaging state_dict loop and the checkpoint model/averaged key swap are unaffected, and persistent=False keeps them out of checkpoints and inference.

Mechanism checks: the margin term runs in training_step only; centroids are plain buffers updated under no_grad from detached float32 logits of the same step, so no gradient reaches the anchors and no validation or sanity batch touches them (centroid_margin.py:56-75, lightning.py:337-380). Masks are honored (pos/neg masks require observed); hard y>=0.5 grouping is used only for term eligibility and the BCE soft targets are unchanged; the existing class weights multiply the hinge on both sides, so the term cannot offset class balancing. Computation is float32 under bf16-mixed with no pair loop, consistent with the near-zero expected overhead.

Label reduction is per-label mean over observed cells then macro over ready labels; since observed counts are near-uniform across labels this approximates the incumbent's pooled-cell scale rather than re-tilting label emphasis, so the mixture coefficient 0.25 is readable as an approximate share. Warm-up is safe: before initialization label_valid is empty and the loss is exactly zero (clamp_min(1) denominators), with uninitialized-label counts logged.

Queue membership and order are unchanged; shared-delayed-ranking and shared-ranking-memory-bank keep their existing dispositions.

Unresolved concerns:
- Anchor staleness is prevalence-asymmetric: centroid_pos updates only on steps containing an observed positive, so rare labels' negative-cell terms are anchored to a long-lagging statistic under momentum 0.9; diagnosable from per-label missing_pos_steps and per-epoch centroid traces, not from the macro result alone.
- Class weights multiply a squared hinge at microbatch 8, so a single rare-label positive inside the band can contribute a large spike to that step's term; the logged train_margin_component_share should be checked for early-epoch excursions well above the intended ~10%.
- Readiness gating requires both centroids initialized, slightly stricter than the idea's opposite-group rule; affects only the first steps for the rarest labels.
- Logged active-band share ignores label_valid (centroid_margin.py:34), overstating early-epoch coverage; objective unaffected.
- Triplet accounting is unavailable at review time (accounting_available=false); the mandatory replay and gates still apply at run time.
- Bootstrap uncertainty for the eventual comparison stays conditional on the fitted models and excludes training-seed variation.

Evidence:
- changes.diff at /home/claudio/src/knee-abnormality/outputs/autoresearch/losses/experiments/shared-centroid-margin-20260920T144634Z-e0e80c5b/changes.diff (config.py:221-246, configs/training.yaml:106-113, lightning.py:235-255/324-460, new losses package)
- candidate src/knee_abnormality/training/losses/centroid_margin.py:14-75 (squared hinge, class weights, readiness gating, EMA update)
- baseline lightning.py:104-118 (pooled-cell BCE reduction), :261-297 (checkpoint key swap), :299-322 (epoch start, training_step), :374-403 (weight averaging over self.model.state_dict()) at /home/claudio/src/knee-abnormality/outputs/autoresearch/losses/baseline/source
- idea record autoresearch/campaigns/losses/ideas.yaml:1249-1340 (mechanism, coverage, budget, implementation review points)
- developer report /home/claudio/src/knee-abnormality/outputs/autoresearch/losses/jobs/developer-20260920T144635Z-6d89635c/report.json (ruff, config load, isolated tensor checks; no GPU use)

## completion-20260920T151316Z-28788f9a

Biproportional (IPF) study/class weighting degrades fold 0: macro AUC 0.89224 vs incumbent 0.89658, delta mean -0.00441, 95% paired report-group interval [-0.00837, -0.00029] over 2,000 valid draws. Contract checks pass: no config differences, seed 3135778825 recorded, 1599.5 s against the 1,797 s ceiling, accounting complete and exact (938,460 train / 24,188 val triplets, cohort hashes matching limits), candidate checkpoint from the 10th epoch (index 9), reference from the 9th.

New mechanism fact that corrects the framing in the pre-launch review: the training fold has no missing labels at all — observed = 3503 for all twelve targets in both control and candidate class_balance.json. So per-study "observed mass" is the constant 12, and the row normalization is not about mask heterogeneity; it purely divides each study's weight vector by its total class-weight mass, down-weighting studies that carry many or rare positive findings. The fitted betas show the accompanying label-emphasis tilt: rare-label positive weight compressed relative to control (Fracture 6.95 -> 4.73, Synovitis 3.88 -> 3.28) and prevalent-label positive weight raised (Effusion 0.837 -> 0.926).

Per-label deltas are only loosely ordered by prevalence (rank correlation ~0.56), so the predicted "degradation concentrated on the rarest labels" is not what happened. The two largest losses are ACL -0.0238 (prev 0.21) and MCL -0.0119 (prev 0.15), with Medial Meniscus -0.0096 at prev 0.41; the three rarest lose less (Fracture -0.0046, Synovitis -0.0076). By family: OA is flat-to-positive (+0.0010 [-0.0064, +0.0085]), meniscus -0.0068 and other_eight -0.0052. Validation soft-target BCE also worsens (0.4857 -> 0.5036). The loss therefore reads as broad damage from down-weighting multi-finding studies rather than a clean rare-label mass artifact, which weakens the case for a mass-restoring variant of this fit.

Queue reordered on new evidence: shared-tempered-balance drops from second to last. Biproportional weighting's realized effect on fold 0 was, in part, exactly a tempering of the control weights toward 1 (positive weight for the rarest label 6.95 -> 4.73; for the most prevalent 0.837 -> 0.926, class_balance.json), combined with study normalization, and it lost 0.0044 macro AUC with an interval strictly below zero. That is direct negative evidence for the compressing direction of shared-tempered-balance; only the sharpening direction remains untested, so it is now the weakest of the weighting ideas. shared-soft-mass-balance (threshold-free recount of positive mass, no deliberate emphasis compression) and shared-group-variance-compression (different mechanism family) move ahead of it. Reserved head unchanged.

shared-delayed-ranking is admissible (no unresolved dependencies) but stays proposed rather than queued: the always-on hybrid's logged traces already showed an actively trained ranking term holding ~26-45% of the objective with ~118 usable pairs/step and a neutral outcome, so re-timing the same term has lower expected value than the three untested weighting/variance mechanisms already queued. Retained for later, not rejected. shared-ranking-memory-bank remains blocked by its unresolved dependency on shared-bce-ranking (inconclusive).

Unresolved concerns:
- Per-label losses are not ordered by prevalence (rank corr ~0.56; worst are ACL and MCL at mid prevalence), so the pre-launch prediction that a mass-restoring variant would recover the loss is not supported by this result; a follow-up of that shape would need its own rationale.
- Study-influence equalization and the rare-positive weight compression are still confounded in this single run; neither is separately falsified.
- IPF convergence on the actual fold was never asserted in-run (fixed 200 sweeps); the persisted betas look well-behaved and monotone in prevalence, but the residual was checked only on synthetic tables.
- class_balance.json version 2 stores betas in the positive_weight/negative_weight fields, so those fields are not comparable field-by-field with version 1 control artifacts.
- Bootstrap uncertainty is conditional on these two fitted models and excludes training-seed variation; a -0.0044 shift is larger than the prior hybrid's noise-level delta but still one seed on one fold.
- shared-ranking-memory-bank stays blocked by its unresolved dependency; its stale-score and refresh handling remain unreviewed.

Evidence:
- Comparison and per-family bootstrap: outputs/autoresearch/losses/experiments/shared-biproportional-weights-20260920T141757Z-d567131d/comparison/comparison.json (paired_report_group_bootstrap.intervals: all/meniscus/oa/other_eight)
- Fitted betas and dense observation counts: .../shared-biproportional-weights-20260920T141757Z-d567131d/training/fold_0/class_balance.json (version 2, observed=3503 for all 12 targets)
- Control weights for comparison: outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat/fold_0/class_balance.json (version 1, observed=3503 for all 12 targets)
- Checkpoint selection: candidate fold_0/best-epoch=09-val_macro_auc=0.89224.ckpt vs control best-epoch=08-val_macro_auc=0.89658.ckpt (selected_checkpoint.json; comparison selected_epoch is 1-based)
- Diff reviewed pre-launch: .../shared-biproportional-weights-20260920T141757Z-d567131d/changes.diff (fixed_class_balance IPF sweep; per-row study_factor applied in the weighted BCE)
- Prior hybrid ranking traces and neutrality: orchestrator report outputs/autoresearch/losses/jobs/orchestrator-20260920T121537Z-6ee62f96/report.json
- Pre-launch review of this candidate: outputs/autoresearch/losses/jobs/orchestrator-20260920T142358Z-61f0ef89/report.json

## review-20260920T151803Z-567b86db

The implementation changes two things at once, both inside the recorded idea but worth separating when the result is read: (1) weight assignment moves from whole-cell thresholded to per-term, removing the y≥0.5 discontinuity; (2) the reference statistic moves from thresholded rate to soft mass m_c, which materially rescales rare-label emphasis. Using the researcher's per-label mean y (loss-evidence-20260920.md:42-53) against the control's recorded weights (13-control-repeat/fold_0/class_balance.json), positive weights drop substantially wherever the soft share is high: Fracture 6.95 → 2.67 (rate 0.073 vs mean y 0.187), Synovitis 3.88 → 1.47 (0.126 vs 0.341), Contusion 2.86 → 1.88, Lateral Meniscus 3.16 → 2.19, Baker's 2.02 → 1.40; near-balanced labels barely move (PF OA 1.087 → 1.031, Effusion 0.837 → 0.847). So the run is not only a continuity fix: it also reduces inverse-frequency emphasis for the rarest labels by roughly 2-2.6x, a direction that partially overlaps shared-tempered-balance's magnitude axis even though the formula 1/(2·rate) is unchanged. A per-label outcome concentrated on Fracture/Synovitis/Contusion should be read against this weight shift, not against the discontinuity alone.

The idea's coverage clause asked for realized positive/negative term mass to be reported; the candidate records m_c, the thresholded rate and both weights in class_balance.json but not realized term mass. That is recoverable from the recorded statistics (term mass is balanced by construction under the new rule) and is not a blocker.

The candidate implements the recorded mechanism exactly and is scope-clean: `diff -rq` against parent_source shows only `training/lightning.py` differing (plus the pre-existing empty allowlisted `training/losses/` dir); no config keys changed, `trainer.class_balanced_bce` stays true. The diff touches only `fixed_class_balance` and `masked_binary_cross_entropy_with_logits`; data loading, model construction, optimizer/scheduler, forward, validation, checkpoint selection and accounting are untouched.

Correctness checks I ran directly rather than taking from the developer report: weights are derived only from `data.train_dataset.studies` (train.py:149), so no validation or gold-fold information enters the statistic; masks are preserved (`losses[observed].mean()` unchanged, and the per-label statistic filters `observed_<target>`); unobserved cells are filled 0.0 (dataset.py:268-276), so the new `(1 - targets)` term is finite there and masked out before reduction; logits are cast to float32 before the loss (lightning.py:309) and `F.logsigmoid` is used on both signs, so the weighted terms are numerically stable under bf16 autocast; the `positive>0 and negative>0` guard keeps m_c strictly inside (0,1). The weight tensor shape (2, n_targets) and broadcasting are unchanged from the incumbent. Version bump 1→2 with the matching validator replaces the superseded rule rather than keeping a fallback, consistent with campaign policy; no other consumer reads a class_balance version.

Loss scale is preserved, which I checked because an unintended scale change would act as a hidden learning-rate change: total cell-weight mass per label is exactly n_obs under both rules (old: w_pos·n_pos + w_neg·n_neg = n/2 + n/2; new: mean over cells of w_pos·y + w_neg·(1−y) = 1 by construction of m_c). Preparation cost is 0.17 s and the per-step form is the same elementwise arithmetic, so ceiling headroom is unaffected.

Queue unchanged: no new evidence bears on ranking. shared-delayed-ranking stays a surplus proposal; shared-ranking-memory-bank stays blocked by its unresolved dependency.

Unresolved concerns:
- The candidate conflates threshold-free assignment with a 2-2.6x reduction of rare-label positive weights (Fracture 6.95 to 2.67, Synovitis 3.88 to 1.47); a win or loss cannot be attributed to the continuity fix alone, and it partially anticipates shared-tempered-balance's magnitude axis.
- Realized per-label positive/negative term mass is not logged as the idea's coverage clause requested; only m_c, thresholded rate and weights are recorded in class_balance.json.
- Triplet accounting is unavailable at review time (accounting_available=false); the mandatory replay and gates still apply at run time.
- The eventual bootstrap interval stays conditional on the two fitted models and excludes training-seed variation; the researcher's own analysis expects a small effect (term-mass ratios 0.89-1.46), so an inconclusive interval is the likely outcome and should not be read as refuting the mechanism.
- shared-ranking-memory-bank remains blocked by its unresolved dependency on shared-bce-ranking (inconclusive); its stale-score and refresh handling are still unreviewed.

Evidence:
- changes.diff (sha 4458a979...) limited to src/knee_abnormality/training/lightning.py lines 55-121
- diff -rq parent_source vs candidate source: only lightning.py differs; extra path is the empty allowlisted training/losses/ directory
- candidate lightning.py:65-79 (m_c from observed training cells, version 2), :99 (validator), :114-121 (term-wise logsigmoid loss)
- lightning.py:309 logits cast to float32 before the loss; dataset.py:268-276 unobserved targets filled 0.0
- scripts/pipeline/train.py:148-149 computes class balance from data.train_dataset.studies only (no validation/gold leakage)
- outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat/fold_0/class_balance.json (incumbent weights for the comparison above)
- researcher scratch loss-evidence-20260920.md:36-62 (per-label mean y vs P(y>=0.5), soft-cell shares, term-mass ratios 0.89-1.46)
- developer report outputs/autoresearch/losses/jobs/developer-20260920T151517Z-a1c7af63/report.json (hard-target equivalence check 1.0598822832107544 both forms)

## proposals-20260920T153958Z-e0d82d19

Verified against the repository, not taken from the report: the v4 target table has 10,093 cells at exactly 0.25 (19.08% of 52,884), 26.5% at 0.025, 12.3% at 0.015, and 0.54% in (0.45,0.55); pooled mean 0.3378 against a hard positive rate 0.2708.

New, decision-relevant structural fact for the campaign: the evaluator's hardening threshold (TARGET_THRESHOLD = 0.5, evaluation/metrics.py:44-57) is identical to the y >= 0.5 grouping used for class weights in the incumbent loss (lightning.py:114-116). Any per-cell rule whose asymmetry keys on y >= 0.5 is therefore aligned with the metric's own labels by construction, and one-sided relaxations on that side cannot move a cell across the scored boundary. This is what separates the clamp family from the rejected masking family (-0.011455): masking discarded both directions of a weak cell's signal, while the clamp retains the full boundary-directed penalty and removes only the away-from-boundary pull.

Corollary worth carrying: the clamped objective is no longer strictly proper — the per-cell minimizer is a half-line, so within-group ordering among free cells is set by optimization dynamics rather than the objective. Because both 0.25 and 0.025 cells are evaluation negatives, their relative order does not enter macro AUC, which bounds the risk; but it means train-loss traces are not calibration diagnostics for this candidate, and the floor-subtracted value must be logged for comparability with the control's 0.6438 -> 0.4413.

Admit shared-evidence-margin-bce, rank it second behind the reserved head.

I verified the proposal's two load-bearing facts myself rather than accepting the report. (1) The label table is quasi-discrete as claimed: over 52,884 cells, 0.025 = 26.5%, 0.25 = 19.08% (10,093), 0.015 = 12.3%, only 0.54% in (0.45,0.55); pooled mean 0.3378 vs hard rate 0.2708. (2) The decisive compatibility point: the evaluator hardens targets at TARGET_THRESHOLD = 0.5 (evaluation/metrics.py:57, data/labels.py:12), which is exactly the grouping the incumbent weight rule uses (lightning.py:114-116). Therefore the clamp's free region (s below t for y<0.5, above t for y>=0.5) always lies on the metric-correct side of that cell's own evaluation label, so released motion can never move a cell across the boundary the metric scores. The removed gradient is precisely the up-pull that holds 0.25 cells — evaluation negatives — at -1.10 logits, and the down-push on weak positives. It changes the penalized domain while leaving the soft target, weights, mask and pooled reduction in the pointwise term untouched, which is inside "preserve soft targets in pointwise terms" (contract.md:34). It is parameter-free, elementwise, and negligible against the 182 s of headroom.

Withholding shared-target-logit-margin (kept as a proposal, not parked, not ranked). It replaces y with y' = sigmoid(logit(y) ± 1) inside the pointwise term. That is a target transform, and delta -> infinity is the hard label the contract excludes; "preserve soft targets in pointwise terms" reads as constraining the term to the unmodified soft target, so a run risks an invalid verdict on contract grounds rather than an interpretable result. The researcher raised the tension itself; I rule it not admissible under the current contract without a user amendment. It also overlaps in motivation with the prepared shared-soft-mass-balance run, which would confound both.

shared-delayed-ranking stays an unqueued surplus proposal: the completed hybrid's traces show the ranking term was active and not drowned out, so a schedule variant of the same term has low expected value relative to the four queued ideas.

shared-tempered-balance moves to last because the prepared soft-mass-balance candidate already cuts rare-label positive weights 2-2.6x, so its magnitude axis is partly pre-empted and should be read after that result.

Unresolved concerns:
- The clamped objective is not strictly proper: free cells have a half-line of minimizers, so within-group score spread is decided by optimization dynamics, not the loss. Bounded for AUC because free regions stay on the metric-correct side, but train-loss and calibration traces lose diagnostic meaning.
- Residual risk shared with the rejected training-label masking result (-0.011455): if a large fraction of weak cells goes free early, the effective supervised set shrinks. The proposal's own estimate is 19.3% of negative and 11.2% of positive cells free at control fit quality; realized per-label free fractions must be logged per epoch and read before attributing any loss to the mechanism.
- shared-target-logit-margin is blocked on contract interpretation only (dependency gate passes). It needs a user ruling on whether an additive logit shift of the pointwise target preserves soft targets; until then it cannot be prepared, and it must not be combined with shared-soft-mass-balance, which attacks the same soft-vs-hard mass mismatch.
- Both new proposals rest on the same validation-side prediction analysis of one control run; the per-cell gradient-energy figures (4.7% wrong-direction) are lower bounds from a single fit and are not independent evidence of a training-fold defect.
- shared-ranking-memory-bank remains unadmitted: its dependency shared-bce-ranking is inconclusive, not provisional_accept, and its stale-score and refresh handling are still unreviewed.
- Concerns already recorded for the prepared shared-soft-mass-balance and running shared-centroid-margin candidates are unchanged and still govern how those results are read.

Evidence:
- Researcher report: outputs/autoresearch/losses/jobs/researcher-20260920T151517Z-cbfc2d1d/report.json (both proposals, exhausted_directions)
- Independent recount of the v4 table: data/labels/public/stevenleehans/version-6/llm_labels_v4_blend.csv, 4,407 studies x 12 labels = 52,884 cells; value_counts 0.025=14,004, 0.25=10,093, 0.015=6,528; mean 0.3378, (y>=0.5) rate 0.2708
- Evaluation threshold identity: src/knee_abnormality/evaluation/metrics.py:44-57 with src/knee_abnormality/data/labels.py:12 (TARGET_THRESHOLD = 0.5)
- Incumbent loss to be preserved outside the clamp: src/knee_abnormality/training/lightning.py:104-118 (weight selection on targets >= 0.5, losses[observed].mean())
- Contract clause under interpretation: autoresearch/campaigns/losses/contract.md:34
- Masking precedent (-0.011455): docs/modeling/supervision.md#training-label-masking
- Prior campaign readings reused for the review pattern: autoresearch/campaigns/losses/findings.md:240-248 (focal), :352 (Brier gradient saturation), :577 (group-tail geometry)
- Completed hybrid traces underpinning the delayed-ranking deferral: recent_results shared-bce-ranking-20260920T112958Z-a2e23db7 (ranking share ~45% -> ~26%, ~10.0/12 labels covered, +52 s)

## completion-20260920T154209Z-f558fb24

The class-centroid margin (margin 1.0, momentum 0.9, weight 0.25) does not improve fold 0: macro AUC 0.89368 vs incumbent 0.89658, delta mean −0.00293, 95% paired report-group bootstrap [−0.00730, +0.00119] over 2,000 valid draws. The interval includes zero but is ~86% below it; the point estimate is the largest negative seen this campaign.

The two review concerns that would have made the result uninterpretable are resolved by the run's own traces, so this reads as evidence against the mechanism rather than against the implementation:

1. Anchor staleness is much milder than feared. missing_pos_steps per epoch out of 438 steps: Fracture 230–245, Synovitis 140–159, Contusion 87–110, ACL 59–74, Medial OA 8–13. Even the rarest label updates its positive centroid on ~45% of steps, so under momentum 0.9 the effective lag is roughly 20 steps, not a long window. Centroid traces are monotone and well separated (e.g. ACL pos 0.067→1.98, neg −0.28→−1.62).
2. No runaway component spikes: epoch-mean margin share 20.2% (epoch 0) decaying monotonically to 3.0% (epoch 9); the maximum logged step-level share is 24.5%, in epoch 0. The term ran roughly 2x the intended ~10% early and then faded, so its influence is concentrated in early training — the opposite profile from shared-delayed-ranking's premise.

Per-label deltas concentrate the loss in high-AUC labels rather than the rare/stale ones: ACL −0.0129, Fracture −0.0087, Lateral Meniscus −0.0068, Medial Meniscus −0.0062, Contusion −0.0060; gains Baker's +0.0048, Lateral OA +0.0043, MCL +0.0019. That pattern is not the prevalence-ordered degradation staleness would predict.

Contract facts: 1,609.8 s measured against the 1,797 s ceiling (+3.0% over reference, leaving ~187 s headroom); accounting complete and matching limits (938,460 training triplets, 35,030 visits, 24,188 validation triplets, 4,380 steps, 10 epochs); fresh seed 3261009277; only the three trainer.loss.centroid_margin keys differ.

Queue change driven by the second consecutive null-or-negative auxiliary score-geometry term. shared-bce-ranking was neutral with a demonstrably active term; shared-centroid-margin is negative-leaning (delta mean −0.00293, interval upper bound +0.00119) with its margin component also demonstrably active early (20.2% of the objective in epoch 0). Two distinct always-on logit-geometry auxiliaries have now failed to produce gain, so shared-group-variance-compression — a third class-conditional logit-geometry auxiliary of the same family — drops to last; it is not parked, since its mechanism (compressing within-group spread rather than pushing cells across an opposite-class anchor) remains distinct and the evidence is suggestive rather than decisive.

Admitting shared-target-logit-margin (dependencies empty, satisfied): it is a pointwise target-shaping change with no batch statistics, no state, and negligible runtime, in the weighting/target-shaping family that has not yet been tested this campaign and where the remaining queue is concentrated. It is ranked below shared-evidence-margin-bce because the two overlap on the target-shaping axis and the one-sided variant is the cleaner first probe of it; running the one-sided candidate first makes the symmetric-shift result interpretable against it.

shared-delayed-ranking is retained as a surplus proposal rather than queued: it is a schedule over the ranking term whose always-on version was neutral, and the centroid-margin traces show an auxiliary term that was early-heavy by construction still did not help, which weakens but does not refute the representation-maturity premise. It is worth keeping for the case where target-shaping also fails.

Unresolved concerns:
- Bootstrap uncertainty is conditional on the two fitted models and excludes training-seed variation; a −0.0029 shift on fold 0 is not separable from seed noise, so this is directional evidence against the mechanism, not a refutation.
- Step-level component share is logged sparsely (87 aggregated points over 4,380 steps), so a single-step rare-positive spike from the class-weighted squared hinge would not be visible; the 24.5% epoch-0 maximum is an aggregate.
- The margin term is early-heavy by construction (20.2% → 3.0% share), so this run tests early auxiliary pressure and says little about a late-applied geometry term.
- Logged active-band share still ignores label_valid (centroid_margin.py:34), overstating early-epoch coverage; the objective is unaffected.
- shared-target-logit-margin's critical review point stands: the shifted target must be derived inside training_step so validation BCE, AUC, checkpoint selection and saved predictions keep the untouched targets.
- shared-ranking-memory-bank remains blocked by its unresolved dependency on shared-bce-ranking (inconclusive); its stale-score and refresh handling are still unreviewed.

Evidence:
- comparison/comparison.json in experiments/shared-centroid-margin-20260920T144634Z-e0e80c5b: macro 0.893679 vs 0.896582, paired_report_group_bootstrap delta_mean −0.0029253, 95% [−0.0072953, +0.0011853], 2,000 valid draws
- training/fold_0/logs/version_0/metrics.csv: train_margin_component_share_epoch 0.2020→0.0303 over epochs 0–9; max logged train_margin_component_share_step 0.2454 (epoch 0)
- training/fold_0/logs/version_0/metrics.csv: train_margin_missing_pos_steps_* per epoch (Fracture 230–245, Synovitis 140–159, Contusion 87–110, ACL 59–74, Medial OA 8–13 of 438 steps)
- training/fold_0/logs/version_0/metrics.csv: train_margin_loss_epoch 0.6755→0.0577 vs train_bce_loss_epoch 0.6535→0.4450
- training/fold_0/timing.json: setup 0.295 s, training 1572.74 s, final prediction 20.84 s; preparation 0.170 s; total 1609.76 s vs 1797 s ceiling
- training/fold_0/selected_checkpoint.json: best-epoch=08, score 0.8936789
- Prior context: orchestrator-20260920T121537Z-6ee62f96 report (shared-bce-ranking inconclusive, +0.00073 [−0.00329, +0.00479])
- ideas.yaml entries for shared-target-logit-margin (created_revision 148) and shared-delayed-ranking (created_revision 4); both have dependencies: []

## completion-20260920T161203Z-14ae6e65

Threshold-free soft-mass class balancing is clearly worse than thresholded inverse-frequency balancing on fold 0: macro AUC 0.88742 vs 0.89658, delta mean −0.00918, 95% paired report-group bootstrap [−0.01345, −0.00484] over 2,000 valid draws. This is roughly ten times the size of the neutral shared-bce-ranking shift, and the interval excludes zero comfortably.

New attribution evidence from the candidate's recorded statistics (training/fold_0/class_balance.json): the realized positive-weight ratio candidate/control equals rate/mean-target per label — Fracture 0.39 (6.95→2.71), Synovitis 0.38 (3.88→1.47), Baker's 0.70, Contusion 0.67, Lateral Meniscus 0.70, ACL 0.89, MCL 0.85, Medial Meniscus 0.88, Lateral OA 0.84, Medial OA 0.92, PF OA 0.95, Effusion 1.02. The per-label deltas are not ordered by that ratio: the two largest shrinks give −0.0197 (Fracture) and only −0.0092 (Synovitis), while near-unchanged ACL and MCL lose −0.0134 and −0.0143, and Contusion (0.67) loses only −0.0030. Eleven of twelve labels decline; the only gain is PF OA (+0.0082), one of the two labels whose weights barely moved, and the other (Effusion) is flat at −0.0024.

So the loss is broad rather than concentrated where rare-label emphasis dropped. That refutes the specific implementation and weakens, without isolating, the pure weight-magnitude explanation: the per-term (continuity) change itself alters the gradient on every soft cell in every label and plausibly carries much of the damage. Practically, the whole-cell thresholded assignment plus full inverse-frequency weighting is now better supported as the incumbent's operating point, and the y≥0.5 discontinuity is not a defect worth removing this way.

Contract checks on the completed run pass, so the recorded numbers are interpretable: accounting complete (938460 triplets / 35030 visits, cohort hashes equal to the limits), validation selection hash identical in all ten epochs, fresh seed 788737321, 1594.5 s of measured setup+training+prediction against the 1797 s ceiling, diff confined to lightning.py. The review-time worry about reduced-precision logsigmoid is resolved: logits are cast to float32 before the loss (lightning.py:309), and the hard-target limit of the per-term formula reduces exactly to the control's weighted BCE, so the loss for y in {0,1} is unchanged.

Queue change: shared-tempered-balance moves below shared-target-logit-margin. Its axis (magnitude of inverse-frequency weighting) has now been partially exercised in the shrink direction and coincided with a clear macro loss, so its expected value at the current exponent choice is lower than the untouched additive-margin axis. It stays queued rather than parked because the evidence is confounded (below) and the tempering exponent can also sharpen. If it runs, its implementation should keep the control's thresholded per-cell weight assignment and vary only the shared exponent/normalization, so it does not repeat this run's two-change confound.

shared-delayed-ranking remains a surplus proposal. It was held pending shared-bce-ranking, which has now reported: the traces showed the ranking term active and well covered throughout (not early-noise dominated) yet macro-neutral, so scheduling its onset has no evidence base and I do not queue it. shared-ranking-memory-bank stays unadmitted on its unresolved dependency.

Unresolved concerns:
- Attribution remains confounded: the candidate changed both the weight assignment rule (per-term, continuous) and the reference statistic (soft mass), and the per-label pattern does not separate them. The reject applies to the combination, not to either change alone.
- Realized per-label positive/negative term mass was still not logged as the idea's coverage clause asked; only m_c, thresholded rate and both weights are recorded. Balanced term mass holds by construction but was not observed.
- Bootstrap uncertainty is conditional on these two fitted models and excludes training-seed and checkpoint-selection variation; the interval bounds the paired difference, not the seed-to-seed spread.
- PF OA's +0.0082 is an exploratory unadjusted interval that includes zero and must not be read as a label-specific opportunity under the shared-loss contract.
- shared-ranking-memory-bank stays blocked by its unresolved dependency on shared-bce-ranking (inconclusive); its stale-score, refresh and detachment handling remain unreviewed.

Evidence:
- comparison/comparison.json and comparison/per_label.json under experiments/shared-soft-mass-balance-20260920T151517Z-d10564bb (macro 0.88742 vs 0.89658; delta −0.00918, [−0.01345, −0.00484], 2000 valid draws)
- experiments/shared-soft-mass-balance-20260920T151517Z-d10564bb/training/fold_0/class_balance.json (version 2; positive_fraction vs mean_target and both weights for all twelve labels)
- experiments/shared-soft-mass-balance-20260920T151517Z-d10564bb/changes.diff (lightning.py only: mean_target weights, per-term logsigmoid loss, class_balance version bump)
- source/src/knee_abnormality/training/lightning.py:309 (logits cast to float32 before the loss) and :106-122 (per-term masked objective)
- training/fold_0/triplet_accounting.json plus packet facts.accounting (10x93846 triplets, 10x3503 visits, cohort hashes equal to limits; validation selection hash 92f1341c… in every epoch)
- training/fold_0/selected_checkpoint.json and best-epoch=08-val_macro_auc=0.88742.ckpt (checkpoint selection consistent with the reported candidate macro AUC)
- facts.timing (setup 0.30 s, training 1572.73 s, prediction 21.31 s, 4380 optimizer steps) against max_run_seconds 1797
- autoresearch/campaigns/losses/ideas.yaml:59-72 (prior rationale for holding shared-delayed-ranking) and :141-168 (shared-tempered-balance mechanism and open exponent choice)

## review-20260920T161744Z-16a1e61f

The candidate implements the one-sided target margin by clamping the logit toward the target's own boundary before BCE: for target>=0.5, effective=min(logit, logit(t)); otherwise effective=max(logit, logit(t)). This removes the control's reverse pull-back that soft targets impose once a logit passes the target's logit, while preserving the full penalty toward the decision boundary. Free regions for positives (above logit(t)) and negatives (below logit(t)) never overlap, so pairwise separation needed for AUC is preserved rather than erased.

New quantitative facts from my own check on a random [8,12] batch with 20% masking: about 16% of observed cells were already free at random initialization, and those cells contribute exactly zero gradient. This is the mechanism's main operational risk — the observed-cell mean denominator is unchanged, so as the free fraction grows over training the effective gradient magnitude falls with no learning-rate compensation. The added per-label `train_free_positive_fraction` / `train_free_negative_fraction` logs make this directly observable per epoch, which is what the interpretation of any inconclusive outcome will hinge on.

Cost is small relative to the 1,797 s ceiling: seven small no_grad accumulator ops per step and 50 scalar logs per epoch, versus the 52 s (3.3%) that the 12-label per-step pair loop cost in the ranking experiment. Accounting, timing and evaluation artifacts do not yet exist (the run has not been launched), so runtime and triplet-identity confirmation remain pending post-run evidence.

Approve the shared-evidence-margin-bce candidate. Scope is clean: `diff -rq` against parent_source shows only `src/knee_abnormality/training/lightning.py` plus the new allowlisted `training/losses/` package; no config.py or configs/training.yaml change, and `trainer.class_balanced_bce` stays true (the weights are reused unchanged), matching facts.config_differences=[].

Protected behavior is intact: the diff touches only the import, the training_step call site, an extension of on_train_epoch_start (diagnostic buffer resets), a new module-level on_train_epoch_end, and a no_grad accumulator helper. Data loading, model construction, optimizer/scheduler, forward, validation_step, checkpoint selection (`checkpoint_sampling_epoch`; no monitored ModelCheckpoint in training/) and the accounting Callback hooks are untouched. The module-level on_train_epoch_end did not previously exist, so it overrides nothing; the accounting callback's own on_train_epoch_end is a separate Callback.

I re-derived the numerics independently rather than relying on the developer's claims (see evidence_refs): free-side cells get exactly 0 gradient, active-side gradients match control's weighted masked BCE bitwise, masked (unobserved) cells get exactly 0 gradient, and exact 0/1 targets sit at boundary |logit|≈13.8 so they retain full pressure. Weighting is `where(target>=0.5, w[0], w[1])` with w shaped [2, n_targets] broadcast over the batch — identical to the control rule; reduction is the same pooled observed-cell mean; loss is computed in float32 on logits already cast to float32 under bf16-mixed.

The objective is one shared formula over all twelve labels with no new tunable (eps is a module constant), so the shared-loss boundary holds. Diagnostics are training-only, per-label logged and reset each epoch, and include an unclamped weighted-BCE trace that reproduces the control formula exactly, giving a comparable curve against the control's 0.6438 → 0.4413.

No queue change: this event produced no evidence bearing on ranking. shared-delayed-ranking remains a retained proposal rather than a queue entry, since the completed hybrid showed the ranking term was already active and non-degenerate at weight 1.0, so schedule variants of it carry modest expected value relative to the queued margin/weighting ideas; shared-ranking-memory-bank stays blocked by its unresolved dependency.

Unresolved concerns:
- Effective gradient scale shrinks as the free fraction grows, since the observed-cell mean denominator still counts zero-gradient cells; a neutral result could reflect reduced effective step size rather than the margin idea failing. The new per-label free-fraction logs must be read before interpreting the outcome.
- No config key records the objective: trainer.class_balanced_bce stays true and candidate_config_changes is empty, so effective_config alone cannot distinguish this run from the control. Identification depends entirely on the recorded source snapshot hash (changes.diff sha256 48b07fb1...).
- Removing the upper pull-back lets logits on confidently-correct cells drift without bound; loss stays finite, but logit magnitude growth under bf16 and its interaction with weight averaging is unmeasured. The epoch-level train_balanced_bce trace is the available proxy.
- Accounting, timing, input-identity and evaluation artifacts are all absent (evidence packet reports available=false); triplet-identity and the runtime ceiling remain unverified until the run completes.
- Targets between 0.3 and 0.7 get free regions starting very close to the decision boundary, so ambiguous cells receive almost no training pressure; whether this is metric-neutral is the hypothesis under test, not an established property.
- shared-ranking-memory-bank remains blocked by its dependency on the inconclusive shared-bce-ranking; its stale-score and refresh handling are still unreviewed.

Evidence:
- Candidate diff: outputs/autoresearch/losses/experiments/shared-evidence-margin-bce-20260920T154409Z-239c3836/changes.diff (sha256 48b07fb1563271638c8259d01aaabed0705063d0d644ca035de71a334a55bd8b, 6868 bytes)
- Candidate loss: .../source/src/knee_abnormality/training/losses/evidence_margin.py:1-44 (boundary/effective-logit construction lines 13-20; weights lines 24-28; returned diagnostics lines 36-43)
- Candidate wiring: .../source/src/knee_abnormality/training/lightning.py:305-311 (call-site swap), :302-312 (epoch-start buffer resets), :332-355 (no_grad accumulation), :356-395 (on_train_epoch_end logging)
- Control reference: baseline/source/src/knee_abnormality/training/lightning.py:104-118 (masked_binary_cross_entropy_with_logits; weight rule line 114-116, reduction line 118) — unchanged and still used by protected profiling.py:60-69
- Class-balance shape: baseline/source/src/knee_abnormality/training/lightning.py:80-88 ([2, n_targets]) and :221-230 buffer registration; observed mask is torch.bool per data/dataset.py:277-280
- My independent CPU verification (venv python, random [8,12] batch, 20% masked, seed 0): free&observed max |grad| = 0.0; active-side gradients allclose to control weighted BCE (atol 1e-7) = True; unobserved-cell max |grad| = 0.0; boundaries for exact targets 0.0/1.0 = -13.8155 / +13.8023; free fraction at init = 0.1558
- Scope check: `diff -rq` baseline/source vs candidate/source reports only training/lightning.py differing plus new training/losses directory; ast.parse succeeds on both changed/new files (py_compile blocked by read-only snapshot)
- Developer report: outputs/autoresearch/losses/jobs/developer-20260920T161153Z-4a13ec60/report.json (status prepared; CPU microbenchmark 0.170 s control vs 0.540 s candidate over 4,380 synthetic steps)

## review-20260920T162350Z-92cad5b2

New quantitative evidence on how active this mechanism actually is, from the Steven v4 blend labels (data/labels/public/stevenleehans/version-6/llm_labels_v4_blend.csv, 52,884 non-null cells): 99.95% of targets are strictly soft, exact 1.0 is 0.045%, exact 0.0 is absent. The passthrough branch for exact 0/1 is therefore nearly inert and the margin applies to essentially every observed cell — this is not a no-op experiment.

Magnitude: mean |shifted − original| is 0.061, max 0.231. The transform is a one-logit outward push, monotone within each side of 0.5, so it preserves target ordering while widening the across-boundary gap; at the optimum the model is asked for a logit ±1 beyond the control's target logit.

The map is discontinuous at exactly 0.5 (0.4999→0.269 vs 0.5→0.731), so near-boundary labels receive sharply separated targets. I quantified the exposure: only 0.75% of cells fall in [0.45, 0.55] and 3.7% in [0.4, 0.6], so the discontinuity touches a small minority; the broader risk is that ambiguous LLM-derived labels are converted into confident training targets across those 3.7%.

The added per-label diagnostics (train_shifted_target_mean, train_entropy_floor, train_balanced_bce against ORIGINAL targets, train_mean_abs_logit) give the traces needed to interpret an inconclusive outcome: train_balanced_bce is directly comparable to the control's train_loss, and train_mean_abs_logit tracks the expected outward logit drift.

Approve shared-target-logit-margin-20260920T161944Z-c1f07123 for launch.

Contract: `diff -rq` against parent_source shows changes confined to configs/training.yaml, config.py, lightning.py and the new training/losses/ package; the only config delta is trainer.loss.target_margin.margin_logits=1.0, class_balanced_bce stays true. Preparation cost 0.17 s; the mechanism is stateless elementwise math on [8,12] tensors plus 50 scalar logs per epoch, so the 1,797 s ceiling is not at risk (contrast: the ranking pair loop cost 52 s).

Substantive review I performed rather than taking from the developer report:
- Protected behavior: validation_step/on_validation_epoch_end still consume raw `batch["target"]` (source lines 490, 500); optimizer/scheduler, data module, checkpoint hooks, masked_binary_cross_entropy_with_logits untouched. No duplicate on_train_epoch_end in the parent, so the new hook does not shadow existing behavior. No training/validation coupling, no fold-0 label use, no added data pass.
- Masking and weights: loss reduces as `(elementwise*weights)[observed].mean()`, byte-equivalent reduction to control; class weights are selected from the ORIGINAL targets, so group assignment is identical to the control (the shift is side-preserving).
- Numerics: targets/logits are float32 before the transform; autocast bf16 does not touch the logit/sigmoid math. Ran the transform: 0.25→0.1092, 0.76→0.8959, 0.98→0.9926, 0.5→0.7311, 0.49→0.2612, exact 1→1. Finite loss and gradients on a masked random batch; all-unobserved batch raises like control.

Ranking and proposals unchanged. shared-delayed-ranking has satisfied dependencies but I am holding it as a surplus proposal rather than queueing: the completed hybrid showed an active, well-scaled ranking term that was still neutral, so scheduling variants of the same pair term have low expected value against the two untried mechanisms already queued.

Unresolved concerns:
- Effective config alone distinguishes this run (trainer.loss.target_margin.margin_logits=1.0), but accounting, timing, input-identity and evaluation artifacts do not yet exist; triplet-accounting replay and the 1,797 s ceiling remain unverified until the run completes.
- Margin 1.0 is unswept and the response may be non-monotone in margin; a neutral macro result will not distinguish 'target sharpening does not help' from 'one logit is the wrong amount'.
- The transform is discontinuous at target=0.5; 3.7% of cells lie in [0.4,0.6] and receive confident targets from ambiguous LLM labels. Whether this amplifies label noise is the hypothesis under test, not established.
- train_mean_abs_logit divides by all cells including unobserved ones, unlike the other diagnostics which use the observed count; read it as an all-cell mean, not an observed-cell mean.
- _diag_* accumulators are created only in on_train_epoch_start, not __init__; safe under the normal Lightning order but would raise if on_train_epoch_end ever ran first.
- shared-ranking-memory-bank remains blocked by its unresolved dependency on the inconclusive shared-bce-ranking; its stale-score and refresh handling are still unreviewed.

Evidence:
- candidate diff: outputs/autoresearch/losses/experiments/shared-target-logit-margin-20260920T161944Z-c1f07123/changes.diff
- candidate source training/losses/target_margin.py:1-54 (shifted_soft_target, target_margin_bce reduction and weight selection)
- candidate source training/lightning.py:305-424 (epoch-start accumulators, training_step branch, new on_train_epoch_end); :490,500 validation uses raw batch['target']
- parent source training/lightning.py:104-118 (control masked_binary_cross_entropy_with_logits; identical weighting and [observed].mean() reduction)
- diff -rq parent vs candidate source: only configs/training.yaml, config.py, training/lightning.py differ, plus new training/losses/
- target distribution check on data/labels/public/stevenleehans/version-6/llm_labels_v4_blend.csv: soft 99.95%, exact1 0.045%, exact0 0%, mean |shift| 0.061, max 0.231, [0.4,0.6] share 3.7%
- developer report: outputs/autoresearch/losses/jobs/developer-20260920T161945Z-e044cd1a/report.json
- prior ranking-hybrid cost/outcome basis for holding shared-delayed-ranking: outputs/autoresearch/losses/experiments/shared-bce-ranking-20260920T112958Z-a2e23db7/comparison/comparison.json

## proposals-20260920T163651Z-6cdab02f

Verified the two motivating measurements behind the admitted proposals against the control artifacts, rather than relying on the researcher's summary.

Class-balance statistics are as cited: thresholded positive rates 0.0719 (Fracture) to 0.5975 (Effusion), positive weights 0.837 to 6.950, all 12 labels observed on all 3,503 training-fold studies (fold_0/class_balance.json, version 1). So the prior offsets a_c span -2.557 to +0.395 as claimed, and the "24 constants" and "one constant per label" both derive from an artifact that already exists and is already validated against the model's target list.

The systematic level shift motivating shared-prior-offset-bce reproduces exactly on the control's 904 validation studies: mean predicted probability ACL 0.317, MCL 0.291, Contusion 0.354, Fracture 0.263, Effusion 0.495, i.e. 11 of 12 labels sit above their target means while Effusion sits below. This confirms the weight table buys balance with a per-label level the macro AUC cannot use, and it is the one motivation for that candidate that does not depend on the disputed capacity argument (per-label heads can absorb a constant level, so this is suggestive, not decisive).

I did not independently reproduce the gradient-allocation decomposition (21.5% weighted gradient mass vs 28.3% cell share for the 0.2-0.8 band) or the sub-AUC split by target confidence; those remain the researcher's computation and the primary quantitative basis for ranking shared-boundary-proximity-weighting second.

Admitted two of three new proposals; queue stays at five with the reserved head unchanged.

shared-boundary-proximity-weighting ranked second. It is the only weighting proposal that leaves every (label, group) weight mass bit-identical, so the attribution recorded for the two rejected weighting candidates (changed group magnitudes: Fracture positive weight 6.95 -> 2.71 / 4.73) does not transfer. It also leaves the per-cell optimum at logit(y), so it is a separable channel from the two in-flight target-margin runs rather than a duplicate: those move the optimum, this moves only allocation. Pinned variant, not a developer choice: k=1, factor from |2y-1|, normalizer per (label, group), no cap. No sweep.

shared-prior-offset-bce ranked third. It is the only untested channel for imbalance (additive link offset replacing the multiplicative table), zero cost, and AUC-neutral at the population optimum by construction. Pinned variant: a_c = log(pi_c/(1-pi_c)) from the stored thresholded rate, tau=1, no class weights retained. I am choosing the externally prescribed statistic over the soft-mean alternative because the point of the run is to bound the additive channel against the multiplicative one; the researcher's measured initialization diagnostic (Fracture pos/neg gradient-mass ratio 0.58 vs control 0.86-1.78) must be recorded pre-launch as the known risk, since it points the same direction as the rejected shared-soft-mass-balance.

shared-tempered-balance demoted to fifth: it re-derives multiplicative group magnitudes, the exact channel bounded by two recorded rejections (-0.00918, -0.00441), so its expected value is now the lowest in the queue.

shared-bounded-target-instruction retained as a surplus proposal, not queued. Its premise overlaps the already-queued shared-group-variance-compression (same within-group-spread hypothesis, different channel) and is directly weakened or strengthened by the two in-flight target-margin results; it also confounds content with schedule. Running it before those report would spend budget on a correlated bet.

shared-delayed-ranking stays proposed: the completed shared-bce-ranking traces show the ranking term was active and non-degenerate (~45% -> ~26% of total loss, ~118 usable pairs/step) and still neutral, so the marginal value of further ranking variants is low even with a schedule.

Unresolved concerns:
- The 0.2-0.8 band decomposition (43.0% of pairs carrying 58.5% of the macro deficit; 21.5% gradient mass vs 28.3% cell share) is unverified by me and is the sole quantitative basis for ranking boundary-proximity-weighting second.
- shared-prior-offset-bce at tau=1 shifts per-label positive/negative gradient mass in the same direction as the rejected shared-soft-mass-balance (Fracture init pos/neg ratio 0.58 vs control 0.86-1.78); this must be recorded before launch, and a negative result will be hard to attribute between 'additive channel fails' and 'rare-positive de-emphasis repeats a known failure'.
- Both admitted candidates must apply their change in training_step only; validation BCE, validation AUC, checkpoint selection and saved predictions must use untouched targets, untouched weights and unoffset logits. This is a review point, not yet verified.
- shared-prior-offset-bce sets trainer.class_balanced_bce=false, so the effective config cannot distinguish it from a plain unweighted control without the new trainer.loss key recording the offset statistic.
- Hypothesis overlap remains across four live items on the boundary/within-group-spread axis (two in-flight target-margin runs, queued variance-compression, admitted boundary-proximity-weighting); if the two margin runs both return clearly negative, the axis should be narrowed rather than extended.
- shared-ranking-memory-bank stays blocked by its unresolved dependency on the inconclusive shared-bce-ranking; its stale-score and refresh handling remain unreviewed.

Evidence:
- Researcher report: /home/claudio/src/knee-abnormality/outputs/autoresearch/losses/jobs/researcher-20260920T161945Z-d22695c8/report.json
- Control class balance verified: .../runs/13-control-repeat/fold_0/class_balance.json — Fracture positive_fraction 0.07193833856694262, positive_weight 6.950396825396825; Effusion 0.5974878675421068 / 0.8368370759675107; observed 3503 for all 12 targets
- Control validation mean predictions computed from .../runs/13-control-repeat/oof_predictions.csv (904 rows): ACL 0.317, MCL 0.291, Medial Meniscus 0.457, Lateral Meniscus 0.333, Medial OA 0.452, Lateral OA 0.407, PF OA 0.499, Effusion 0.495, Synovitis 0.403, Baker's 0.411, Contusion 0.354, Fracture 0.263
- Ranking-channel evidence for deprioritising shared-delayed-ranking: /home/claudio/src/knee-abnormality/outputs/autoresearch/losses/experiments/shared-bce-ranking-20260920T112958Z-a2e23db7/comparison/comparison.json (delta +0.00073, [-0.00329, +0.00479], 2000 valid draws)
- Prior weighting rejections bounding the multiplicative channel (shared-soft-mass-balance -0.00918, shared-biproportional-weights -0.00441): /home/claudio/src/knee-abnormality/autoresearch/campaigns/losses/prior-work.md
- In-flight target-margin candidates whose results should precede shared-bounded-target-instruction: experiments/shared-evidence-margin-bce-20260920T154409Z-239c3836 and experiments/shared-target-logit-margin-20260920T161944Z-c1f07123

## completion-20260920T164612Z-9e410ee2

Resolution of the review's central open question: the free-fraction traces show the mechanism was genuinely active and gradient starvation is bounded, so this is an informative null rather than an uninterpretable one. Free (zero-gradient) observed cells rose monotonically from 2.2%/3.2% (positive/negative) at epoch 0 to 16.9%/24.1% at epoch 9, i.e. roughly a 20% reduction in effective gradient mass by the end of training — real, but far from degenerate. Train balanced BCE fell 0.6489 → 0.4497, closely tracking the control-parented ranking run's BCE trajectory (0.6369 → 0.4482), so removing the pull-back did not measurably slow optimization.

Free fractions vary widely across labels at epoch 9 (negatives: Synovitis 0.497, Fracture 0.408, Baker's 0.394 vs PF OA 0.134, MCL 0.122; positives: Baker's 0.352 vs PF OA 0.066), yet the per-label deltas do not track that spread — Baker's +0.0024 despite the largest free fractions, Medial Meniscus −0.0095 with a middling one. So the gradient-shrink pathway does not explain the outcome.

Result is the tightest null in the campaign: macro AUC 0.89653 candidate vs 0.89658 reference, delta mean −0.00003, 95% paired bootstrap [−0.00360, +0.00361] over 2,000 valid draws.

Contract evidence now complete and clean: triplet accounting complete with 938,460 training triplets / 35,030 visits and cohort hashes matching the pinned limits; validation selection hash identical across all ten epochs (92f1341c…); 4,380 optimizer steps; measured compute 1,584.6 s against the 1,797 s ceiling (+22.3 s, ~1.4%, over the 1,562.3 s reference). Fresh seed 3906936043 recorded.

Queue membership and order stay unchanged (ranked_ids=null). The one-sided margin outcome is a tight null, not a signal that reorders the remaining pointwise-weighting ideas: it says removing the soft target's reverse pull-back is metric-neutral, which is orthogonal to how gradient mass is allocated across cells (boundary-proximity weighting, prior offset, tempered balance) or to within-group spread (variance compression). The reserved head shared-target-logit-margin stays first; it tests the opposite manipulation (targets pushed outward, pressure retained) on the same axis and is the natural complement rather than a duplicate — a neutral result there would be the one that closes the whole target-geometry family.

No proposals admitted. shared-bounded-target-instruction is further weakened by this null (it bets on the same target-geometry channel and additionally confounds content with schedule), and it still overlaps queued shared-group-variance-compression. shared-delayed-ranking keeps the low expected value established from the shared-bce-ranking traces; this event adds nothing for it. shared-ranking-memory-bank remains ineligible: its dependency shared-bce-ranking is decision=inconclusive, and its stale-score/refresh handling is still unreviewed.

Unresolved concerns:
- Removing the soft-target pull-back is measurably active yet macro-neutral; this bounds the value of further 'free region' variants on the same axis, but it does not rule out that a partial (scaled rather than fully clamped) pull-back reduction behaves differently. That variant is not queued and I am not proposing it on this evidence alone.
- Identification of this run still rests solely on the source snapshot hash (changes.diff sha256 48b07fb1…); trainer.class_balanced_bce stayed true and candidate_config_changes is empty, so effective_config cannot distinguish it from the control. The recorded provenance is sufficient here, but future objective edits should surface a config key.
- Logit drift from the removed upper pull-back was never measured directly; this candidate logged no mean-abs-logit diagnostic (unlike the prepared shared-target-logit-margin). The BCE trace is only an indirect proxy, so unbounded-drift interaction with bf16 and weight averaging remains unverified.
- Medial Meniscus −0.0095 has an exploratory interval strictly below zero ([−0.0168, −0.0026]); it is unadjusted across twelve labels, cannot override the macro verdict, and must not motivate per-label treatment.
- Bootstrap uncertainty is conditional on these two fitted models and excludes training-seed variation and repeated selection on fold 0; a ±0.0036 interval still cannot exclude effects of the campaign's 0.001 practical scale.
- shared-ranking-memory-bank remains blocked by its unresolved dependency on the inconclusive shared-bce-ranking; its stale-score and refresh handling are still unreviewed.

Evidence:
- training/fold_0/logs/version_0/metrics.csv in outputs/autoresearch/losses/experiments/shared-evidence-margin-bce-20260920T154409Z-239c3836 — per-epoch train_free_positive_fraction 0.0218→0.1687, train_free_negative_fraction 0.0325→0.2407, train_balanced_bce 0.6489→0.4497, plus per-label free-fraction columns
- Same metrics.csv, epoch-9 per-label free fractions (negatives Synovitis 0.497 / Fracture 0.408 / Baker's 0.394 vs PF OA 0.134; positives Baker's 0.352 vs PF OA 0.066)
- .../shared-evidence-margin-bce-20260920T154409Z-239c3836/comparison/comparison.json — macro 0.8965264 vs 0.8965817, delta mean −3.385e−05, 95% [−0.0035956, +0.0036097], 2,000 valid draws
- .../shared-evidence-margin-bce-20260920T154409Z-239c3836/comparison/per_label.json — Medial Meniscus delta −0.00951, interval [−0.01680, −0.00255]
- assignment facts.accounting — complete=true, 938,460 triplets / 35,030 visits, train cohort b536b7e1…, validation cohort 7167eba1… with identical selection hash 92f1341c… across ten epochs
- assignment facts.timing — setup 0.29 s + training 1548.03 s + prediction 20.34 s = 1584.6 s vs 1,797 s ceiling; 4,380 optimizer steps; seed 3906936043
- training/fold_0/epochs/metrics_epoch_00*.json — validation macro AUC 0.8051→0.8965, best checkpoint epoch 08 at 0.89653
- Prior orchestrator report jobs/orchestrator-20260920T161744Z-2ccad34d/report.json (implementation review whose free-fraction concern this completion resolves)

## review-20260920T165239Z-0f7050b1

The mechanism is active but modest, quantified directly from the Steven v4 blend labels (data/labels/public/stevenleehans/version-6/llm_labels_v4_blend.csv, 4,407 rows x 12 labels). With k=1 the confidence factor spans [1,2] before normalization; after dividing by each group's mean the realized per-cell weight multiplier runs roughly 0.68-1.80 across labels, with a typical p10-p90 band of about 0.85-1.35. So a near-boundary cell gets up to ~2x the gradient of a confident cell in the same group, while group mass is preserved.

Group normalizers differ sharply by label and side: Synovitis negatives Z=1.506 and Baker's negatives Z=1.336 (their negative cells already cluster near 0.5, so their spread is compressed, p10-p90 ~0.84-1.14 and 0.77-1.12), whereas PF OA and Effusion positives are near-saturated (Z=1.130/1.150, p90 only 1.06). The reweighting is therefore unevenly distributed across labels even though the formula is shared, which is the expected consequence of shared normalization over heterogeneous label-confidence distributions.

Direction note for interpretation: this candidate upweights ambiguous labels, the opposite treatment to the concurrently running shared-target-logit-margin, which sharpens them. The two runs bracket the same question about LLM-derived label confidence and should be read together once both complete.

Verification that group mass is preserved: with Z set to the group mean of confidence, the mean factor within each (label, group) over the training fold is exactly 1 by construction, matching the developer's synthetic check.

Diff review (changes.diff, 4 files) confirms the candidate implements the assigned topic and keeps the contract. Edits are confined to configs/training.yaml, config.py, training/lightning.py and the new training/losses/boundary_weighting.py; the only effective config difference is trainer.loss.boundary_weighting.enabled, with class_balanced_bce unchanged at true.

Shared-formula compliance: BOUNDARY_WEIGHTING_K=1.0 is a module constant, not a config knob, and the same confidence(y)=1+k(1-|2y-1|) applies to all twelve labels. Only the per-group normalizers Z are label-derived, from training-fold cells, exactly the pattern the existing class weights already use; the contract permits this.

Protected behavior in lightning.py is untouched: the diff adds a buffer registration in __init__, one extra argument threaded through masked_binary_cross_entropy_with_logits, and the training_step call site. Data loading, model construction, optimizer/scheduler, forward/validation, checkpoint selection and accounting are unmodified. validation_step does not pass the factor, so validation stays identical to the control; the standalone profiling.py caller keeps the default None.

Mask and soft-target semantics verified: observed is defined as target non-null (data/labels.py:174), so the refactor of positive/negative counting in fixed_class_balance from height-difference to explicit value filters is exactly equivalent — no null can reach the negative group. Unobserved cells receive target 0.0 (data/dataset.py:270-276), so the factor is finite there and those cells are still dropped by losses[observed]. Weights are float32 buffers and BCE-with-logits autocasts to fp32, so the reweighting is numerically stable under bf16-mixed. Soft targets themselves are unchanged; only gradient weight moves.

No leakage: normalizers come from train_dataset.studies only, computed once at setup (0.33 s preparation), and no validation or gold labels enter. Runtime impact is elementwise and negligible against the 1,797 s ceiling.

Queue and rankings unchanged; this review event provides no evidence bearing on ordering, and I admit no proposals here.

Unresolved concerns:
- Group weight mass is preserved only in the fold-level, uniform-exposure sense. Within any microbatch of eight, and under the stochastic study/triplet sampling, the realized mean factor deviates from 1, so effective step size fluctuates slightly relative to the control. The same approximation already applies to the control's class weights.
- No diagnostic of the applied factor or of an unweighted comparison loss is logged, unlike the margin experiment. train_loss now measures the reweighted objective and is not directly comparable to the control's train_loss curve. The weight distribution is reconstructible offline from class_balance.json plus the label file, so this is an interpretation cost, not a validity problem.
- k=1.0 is pinned and unswept. A neutral macro result will not distinguish 'boundary reweighting does not help' from 'this strength is wrong', and the response need not be monotone in k.
- Accounting, timing and evaluation artifacts do not exist yet (accounting_available=false). The triplet-accounting replay, input identities and the 1,797 s ceiling remain unverified until the run completes.
- class_balance.json gains keys boundary_weighting_k and the two per-group normalizers. This is self-consistent for fresh runs, but validate_resume_checkpoint (training/checkpoints.py:123) compares the whole dict, so a checkpoint written by the control cannot be resumed under this candidate. Irrelevant for the planned fresh full-budget run.
- shared-ranking-memory-bank remains blocked by its unresolved dependency on the inconclusive shared-bce-ranking; its stale-score and refresh handling are still unreviewed.

Evidence:
- changes.diff: /home/claudio/src/knee-abnormality/outputs/autoresearch/losses/experiments/shared-boundary-proximity-weighting-20260920T164732Z-1a061454/changes.diff (4 files, +config key, +26-line loss module)
- candidate source: .../shared-boundary-proximity-weighting-20260920T164732Z-1a061454/source/src/knee_abnormality/training/losses/boundary_weighting.py:1-26 (k=1.0, confidence, normalizer tensor, factor)
- candidate source: .../source/src/knee_abnormality/training/lightning.py:61-90 (fixed_class_balance normalizers), :119-130 (validated_boundary_weighting_normalizers), :137-152 (weighted loss), :265-274 (buffer), :353-360 (training_step call site)
- mask semantics: .../source/src/knee_abnormality/data/labels.py:174 (observed = target is_not_null) and .../source/src/knee_abnormality/data/dataset.py:270-276 (unobserved target filled with 0.0)
- resume comparison of full class_balance dict: .../source/src/knee_abnormality/training/checkpoints.py:123
- factor distribution computed read-only from data/labels/public/stevenleehans/version-6/llm_labels_v4_blend.csv via /tmp/bw.py (per-label Z, min/max, p10/p90 reported in finding)
- developer report: /home/claudio/src/knee-abnormality/outputs/autoresearch/losses/jobs/developer-20260920T164733Z-3ba05308/report.json (static checks, synthetic mass-preservation check)
- prior related result: /home/claudio/src/knee-abnormality/outputs/autoresearch/losses/experiments/shared-bce-ranking-20260920T112958Z-a2e23db7/comparison/comparison.json

## completion-20260920T171405Z-71d467ec

shared-target-logit-margin (margin_logits=1.0, seed 2166402093) is the campaign's most precisely neutral result: macro AUC 0.89669 vs incumbent 0.89658, delta mean +0.00005, paired 95% interval [-0.00391, +0.00388] over 2,000 valid draws. Incumbent retained.

The mechanism was unambiguously active, so this is an informative null rather than a null result about an inert change. The added diagnostics show train_mean_abs_logit rising 0.551 -> 1.899 across epochs, the expected outward drift of roughly one logit relative to the control's target scale, and train_balanced_bce measured against the ORIGINAL targets ends at 0.4541 versus the control's train_loss 0.4413 at the same epoch (control metrics.csv, fold_0/logs/version_0). So the margin cost real fit on the untransformed targets, moved the score scale substantially, and bought nothing in ranking. train_entropy_floor is constant at 0.14748 by construction, confirming the intended floor reduction from the control's 0.2611.

Interpretation: a shared, order-preserving outward shift of soft targets changes calibration and logit scale but leaves within-label ordering essentially untouched, which is what a rank metric measures. This is a general lesson for target-space ideas: only maps that change *relative* spacing between cells (ties, clipping, reweighting) can plausibly move AUC.

Per-label deltas are mixed and small: Baker's +0.0104 and Lateral OA +0.0046 up, ACL -0.0090 and Synovitis -0.0094 down; every label interval covers zero. Baker's and Synovitis moved in the same directions as in the ranking hybrid, which is consistent with label-level noise shared across runs rather than a mechanism-specific tradeoff.

Contract checks pass on the artifacts: accounting complete with 938,460 training triplets / 35,030 visits and the pinned cohort hashes; all ten validation epochs carry the identical selection hash 92f1341c..., so validation selection is independent of the objective change; total measured 1,574.0 s (setup 0.297 + train 1,553.3 + predict 20.4), +0.75% over the reference and well under the 1,797 s ceiling, so the per-label diagnostics are nearly free.

Reranking: shared-group-variance-compression drops below shared-tempered-balance. The margin run gives the first direct evidence about score-geometry manipulation in this campaign: training pushed the mean absolute head logit from 0.551 (epoch 0) to 1.899 (epoch 9), i.e. a large, deliberate change in score scale and spread, while validation macro AUC moved +0.00005. That does not refute a within-group variance penalty (which is not a monotone rescale and can reorder studies), but it weakens the premise that reshaping the score distribution in a metric-irrelevant direction buys ranking quality, so the two untested class-imbalance-handling mechanisms (prior offset, tempering) now rank ahead of it. Queue membership and the reserved head are unchanged.

Proposals: neither admissible proposal is queued. shared-bounded-target-instruction is another shared monotone target map; the concurrently running shared-boundary-proximity-weighting tests the same label-confidence premise from the opposite direction, and its outcome should determine whether a third target-space variant is worth a full budget. shared-delayed-ranking stays surplus because the always-on hybrid was active (confirmed traces) yet neutral, so the ranking channel has not earned a second slot ahead of untested mechanisms. Both remain proposed with their records intact. shared-ranking-memory-bank remains blocked by its dependency on the inconclusive shared-bce-ranking; I did not alter that dependency.

Unresolved concerns:
- Margin 1.0 is unswept and the response need not be monotone, but the logit-drift trace makes 'the shift was too small to matter' implausible; a repeat at another margin would most likely re-measure the same order-preserving null.
- Bootstrap uncertainty is conditional on these two fitted models; it excludes training-seed variation and repeated selection on fold 0. Two consecutive near-zero macro deltas do not establish that the fold-0 harness can resolve gains near +0.001.
- train_mean_abs_logit averages over all cells including unobserved ones; read it as an all-cell mean. It still tracks the intended drift, since the drift is model-wide.
- The discontinuity at target=0.5 and the 3.7% of cells in [0.4,0.6] remain untested as a noise-amplification risk; the neutral macro result neither confirms nor rules it out.
- _diag_* accumulators are still created only in on_train_epoch_start; harmless here but fragile if that hook order ever changes. Not a validity issue for this completed run.
- shared-ranking-memory-bank stays blocked by its unresolved dependency on the inconclusive shared-bce-ranking; its stale-score, refresh and detachment handling remain unreviewed.

Evidence:
- Candidate diagnostics per epoch: outputs/autoresearch/losses/experiments/shared-target-logit-margin-20260920T161944Z-c1f07123/training/fold_0/logs/version_0/metrics.csv (train_mean_abs_logit 0.551->1.899; train_balanced_bce 0.6496->0.4541; train_entropy_floor 0.147484 constant)
- Control comparison curve: outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat/fold_0/logs/version_0/metrics.csv (train_loss_epoch 0.6438->0.4413)
- Macro comparison and bootstrap: experiments/shared-target-logit-margin-.../comparison/comparison.json and evaluation.json (delta mean 5.05e-05, 95% [-0.0039105, +0.0038818], 2,000 valid draws)
- Per-label exploratory intervals: experiments/shared-target-logit-margin-.../comparison/per_label.json
- Checkpoint selection: training/fold_0/selected_checkpoint.json (best-epoch=07, val_macro_auc 0.896687)
- Timing: training/fold_0/timing.json (setup 0.297 s, training 1553.28 s, prediction 20.37 s, 4,380 steps, peak CUDA 20.65 GB)
- Accounting and validation independence: training/fold_0/triplet_accounting.json via evidence.json accounting (10x93,846 train triplets, cohort b536b7e1..., validation selection 92f1341c... identical across all ten epochs)
- Prior ranking-hybrid interpretation for cross-run label patterns: outputs/autoresearch/losses/jobs/orchestrator-20260920T121537Z-6ee62f96/report.json

## review-20260920T172207Z-edbd9d87

The candidate implements the additive prior-logit offset a_c = log(f_c/(1−f_c)) exactly as specified, verified numerically against the control's fold-0 statistics, and leaves validation, checkpoint selection, predictions, and all protected Lightning behavior (data loading, model construction, optimizer/scheduler, accounting) unchanged.

Interpretation note for the eventual result: this run contrasts the offset against the control's hard-thresholded multiplicative weight table, not against unweighted BCE. Two effects are bundled. First, the weighting's dependence on a 0.5 threshold over soft targets disappears. Second, the effective gradient scale on rare labels drops: at z≈0 the control gives a rare positive weight 1/(2f) (Fracture 6.95) times a 0.5 residual, whereas the offset gives residual ≈ (f−1) ≈ −1 unweighted. Both losses have per-label mean weight 1 in the control's construction, so the aggregate step size is comparable, but the rare-positive tilt is materially milder under the offset. A macro difference in either direction therefore cannot separate "prior offset is a better link-level treatment of imbalance" from "the control's rare-positive upweighting was too strong or too weak at the fixed learning rate."

No diagnostic of the offset-adjusted versus unadjusted loss is logged, and train_loss now measures the offset objective, so its curve is not directly comparable to the control's train_loss. The offsets are fully reconstructible offline from class_balance.json, so this is an interpretation cost only.

Reviewed changes.diff and the candidate source directly. The edit is confined to src/knee_abnormality/training/lightning.py (rtk diff -rq over src/ and configs/ shows only that file differs; the empty training/losses/ directory is scaffolding). No config change, and config_differences is empty, so the numerical gate's recipe fields are untouched; require_source_change is satisfied by the lightning edit.

Mechanism is implemented correctly and matches the idea: fixed_class_balance now stores prior_offset = log(f/(1-f)) from training-fold observed counts only, and masked_binary_cross_entropy_with_logits adds the (12,) offset to logits before BCE instead of multiplying per-cell losses. I recomputed the offsets from the control's fold-0 class_balance.json and reproduced the developer's cited values (ACL −1.328, MCL −1.707, Fracture −2.557), confirming the standard logit-adjustment direction: rare labels require a larger learned logit to compensate the negative prior offset.

Correctness checks that matter here all pass. Logits are cast to float32 in training_step before the offset and BCE, so the add and the loss are float32 under mixed precision. Masking is unchanged: the offset is applied to the full tensor and the mean is still taken over observed cells only. Soft targets are preserved and, unlike the control's targets>=0.5 weight lookup, the offset has no hard threshold; stationary point z = logit(t) − a_c stays monotone in t. The prior uses train_dataset.studies only, so no fold-0 validation leakage. validation_step (lightning.py:402-422) is untouched and scores raw un-offset logits, so checkpoint selection and saved predictions keep control semantics; a per-label constant shift would not change per-label AUC in any case. class_balance is consumed only by training and the resume check — no inference or harness consumer of positive_weight exists (rg over src/, scripts/, and the autoresearch harness), so the version bump to 2 is self-consistent for a fresh run. Added cost is one broadcast add per step, negligible against the 1,797 s ceiling; preparation_seconds 0.17.

Queue membership and order unchanged; no proposals event, so none admitted.

Unresolved concerns:
- The comparison bundles two changes: removal of the 0.5-threshold weight lookup over soft targets, and a milder rare-positive gradient tilt (control 1/(2f) up to 6.95 for Fracture vs unweighted residual under the offset). A macro difference will not separate the link-level prior mechanism from effective rare-class step size at the fixed learning rate.
- train_loss now measures the offset objective and is not comparable to the control's train_loss curve; no unadjusted comparison loss or offset diagnostic is logged. Offsets are recoverable offline from class_balance.json, so this is interpretive only.
- class_balance.json becomes version 2 with prior_offset replacing positive_weight/negative_weight, and validate_resume_checkpoint (training/checkpoints.py:123) compares the whole dict, so a control-written checkpoint cannot be resumed under this candidate. Irrelevant for the planned fresh full-budget run.
- trainer.class_balanced_bce keeps the value true while its meaning changes to prior-offset adjustment, and validated_class_balance_weights keeps a weights name while returning offsets. Documentation is not candidate-editable, so the artifact naming will misdescribe the mechanism if this candidate is ever adopted.
- Accounting, timing, input identity and evaluation artifacts do not exist yet (accounting_available=false); the triplet replay and the 1,797 s ceiling remain unverified until the run completes.
- shared-ranking-memory-bank remains blocked by its unresolved dependency on the inconclusive shared-bce-ranking; its stale-score and refresh handling are still unreviewed.

Evidence:
- changes.diff (sha256 f3e0000430d3312a...) at outputs/autoresearch/losses/experiments/shared-prior-offset-bce-20260920T171555Z-7de7d85a/changes.diff: single-file diff against parent_source
- candidate src/knee_abnormality/training/lightning.py:55-110 (fixed_class_balance version 2 with prior_offset, class_balance_offset_tensor, offset applied pre-BCE with observed-mean reduction)
- candidate src/knee_abnormality/training/lightning.py:297-314 (training_step casts logits to float32 before the offset) and :402-422 (validation_step unchanged, raw logits)
- rtk diff -rq baseline/source/src vs candidate/source/src: only lightning.py differs, plus an empty training/losses/ directory; configs/ identical
- Offsets recomputed from outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat/fold_0/class_balance.json: ACL -1.328, MCL -1.707, Medial Meniscus -0.377, Fracture -2.557, matching the developer's claim
- rg for class_balance/positive_weight across src/, scripts/ and src/knee_abnormality/autoresearch/: no inference, evaluator or harness consumer of the weight fields; scripts/pipeline/train.py:148-184 writes and passes class_balance for training only
- developer report outputs/autoresearch/losses/jobs/developer-20260920T171556Z-7375ddbb/report.json (tau fixed at 1, no new trainer.loss key)
- evidence packet outputs/autoresearch/losses/jobs/orchestrator-20260920T172207Z-2510324f/evidence.json: config_differences empty, preparation_seconds 0.170, accounting/evaluation artifacts unavailable

## proposals-20260920T173352Z-fe29ee0b

Independent verification of the three structural claims the anchor proposal rests on: `batch["study_uid"]` is present in `training_step` and the loss already runs on float32 logits (lightning.py:305-320); a non-persistent buffer is already registered for class_balance_weights (lightning.py:222-228), so training-only state that never reaches state_dict, checkpoint selection or inference has an in-file precedent; and triplet selection is seeded with (study uid, training_seed, epoch) (dataset.py:190-197), so each epoch genuinely presents a different draw of the same study. The mechanism is therefore implementable inside the candidate allowlist with no change to protected behavior, and the buffers cannot leak into inference.

New interpretive limit on the headroom figure: the +0.0052/+0.0075 estimate is an m→infinity extrapolation from two checkpoints that score 0.8847/0.8865 at multiplier 1 against control B's 0.89658, so the draw-noise fraction on the current recipe is unmeasured. A training-time term can only reduce the function's draw sensitivity, never average draws, so any capture is partial and the figure is an upper bound on an upper bound. The run's per-label deviation and drift logs are valuable as the first measurement of this quantity on the current recipe even if the macro interval includes zero.

Separately, the scratch analysis closes four weighting leads with measurement rather than opinion (per-label loss spread max/min 1.26; batch class-weight mass CV 0.104 vs grad-norm spread 0.57-1.78; 4,203/4,258 report groups singleton; 58 gold studies). These should not be re-derived by later researchers.

Admit shared-draw-consistency-anchor and rank it second, behind the reserved head. It is the first candidate on an axis with a quantified, measured headroom rather than a hypothesis (+0.0052 to +0.0075 at multiplier 1 from the binormal fit, draw-noise/draw-noise-20260920.md §1), it is cheap (~168 KB training-only buffers, elementwise [8,12] work, expected ~1% of the 1,562.3 s reference), and it reaches every observed cell of every label at every step from epoch 1, so it is not exposed to the microbatch-8 pair sparsity that limited shared-bce-ranking. Its design also answers shared-shift-projected-bce's recorded failure with a running drift EMA instead of a per-step microbatch mean, and the variance-ratio normaliser blocks the trivial logit-shrink solution that shared-target-logit-margin showed to be AUC-neutral in the opposite direction.

Hold shared-draw-reliability-weighting as a surplus proposal rather than queueing it. It reads the same v statistic that the anchor run will log for the first time on the current recipe; until that run reports the realized deviation and drift traces, its expected effect is indirect and its statistic is confounded with genuine label difficulty (the researcher records this and itself ranks it second). Queueing it now would spend a full-budget slot on a weaker variant of a measurement we are about to take.

Demote shared-tempered-balance below shared-group-variance-compression. Tempering sits on the class-frequency axis that has now been measured negative twice (soft-mass −0.00918, biproportional −0.00441) and is being probed again from the link side by the prepared shared-prior-offset-bce run; its result will inform the tempering strength question, so running tempering before that lands is wasteful. shared-group-variance-compression stays ahead of it: shared-evidence-margin-bce (−0.00003 while leaving 17%/24% of cells gradient-free) shows that releasing within-group placement is neutral, which does not refute a term that changes relative spacing.

No other queue changes; shared-ranking-memory-bank remains unresolved on shared-bce-ranking (inconclusive) and stays proposed.

Unresolved concerns:
- The anchor lags the model by about one epoch (one visit per study per epoch), so the deviation mixes irreproducible draw noise with genuine per-study learning. The per-label drift EMA removes only the common level shift; control B climbs 0.806 -> 0.897 over epochs, so mid-training the term will also penalise legitimate per-study improvement. A null cannot separate the two causes.
- The +0.0052/+0.0075 headroom is extrapolated (m->inf) from two checkpoints that are not the current class-balanced control (0.8847/0.8865 vs 0.89658 at multiplier 1); the draw-noise fraction on the present recipe is unmeasured until this run logs it.
- lambda=0.1, anchor momentum 0.5, variance floor 0.25 and drift momentum are all pinned and unswept. Campaign shares measured so far (45-48% neutral or harmful, 20%->3% negative) only bracket the target; a neutral macro result will not distinguish mechanism failure from wrong strength.
- Review of the eventual candidate must confirm the class grouping rule matches the BCE class-weight rule, that anchors/drift/V_c are detached, that the variance floor is applied before division, that epoch 0 and unseen studies are inactive, and that validation and any sanity pass never write the buffers.
- shared-draw-reliability-weighting's statistic is elevated for genuinely ambiguous studies as well as noisy ones, so it partly down-weights hard cells; shared-balanced-focal (-0.00260) only establishes that the opposite direction lost, not that this one wins.
- shared-ranking-memory-bank stays blocked by its unresolved dependency on the inconclusive shared-bce-ranking; its stale-score, refresh and detachment handling remain unreviewed.

Evidence:
- Proposal source: outputs/autoresearch/losses/jobs/researcher-20260920T171556Z-b263ac05/report.json
- outputs/autoresearch/losses/jobs/researcher-20260920T171556Z-b263ac05/scratch/draw-noise-20260920.md sections 1-4 and draw_noise.py (CPU-only analysis of saved artifacts)
- docs/validation/results/2026-09-19-relative-sampler-width-result.md#count-multipliers: recorded x1..x4 fold-0 table for control-check and sigma-fraction-0867
- src/knee_abnormality/training/lightning.py:222-228 (non-persistent buffer), :305-320 (training_step, study_uid, float32 logits) in baseline/source, verified directly
- src/knee_abnormality/data/dataset.py:190-197: triplet selection seeded by (study uid, training_seed, epoch), verified directly
- shared-bce-ranking-20260920T112958Z-a2e23db7 finding and comparison/comparison.json: +52 s (3.3%) pair-loop cost, ~182 s ceiling headroom, macro delta [-0.00329, +0.00479]
- Prior-work findings for shared-soft-mass-balance (-0.00918), shared-biproportional-weights (-0.00441), shared-balanced-focal (-0.00260), shared-balanced-brier (-0.00240), shared-target-logit-margin (+0.00005), shared-evidence-margin-bce (-0.00003), shared-shift-projected-bce: autoresearch/campaigns/losses/prior-work.md
- Laine & Aila, Temporal Ensembling, ICLR 2017 (openreview BJ6oOfqge); Batch Inverse-Variance Weighting, arXiv:2107.04497 - cited precedents for the two proposals

## completion-20260920T174237Z-5183276d

Boundary-proximity reweighting (k=1.0, seed 1618163065) is neutral-to-slightly-negative on fold 0: macro AUC 0.89551 vs incumbent 0.89658, delta mean -0.00112, paired 95% interval [-0.00503, +0.00287] over 2,000 valid draws.

Contract checks pass on the artifacts. Accounting complete: 938,460 training triplets / 35,030 visits, pinned cohort hashes, and all ten validation epochs carry the identical selection hash 92f1341c..., the same one the margin run recorded, so validation selection is independent of the objective change and of the training seed. Measured 1,599.2 s (setup 0.30 + train 1,562.4 + predict 20.2), 102.4% of the 1,562.3 s reference, well under the 1,797 s ceiling; the reweighting is essentially free.

Paired with shared-target-logit-margin, the bracket the prior report anticipated now closes. Sharpening ambiguous targets gave +0.00005; upweighting them gives -0.00112. Both mechanisms were verified active, and both land inside noise. Redistributing shared gradient mass according to the Steven v4 confidence distribution, in either direction, does not move fold-0 macro AUC. Combined with the margin lesson that only maps changing relative spacing can move a rank metric, this weakens the whole label-confidence-weighting family, not just k=1.

The candidate's late curve sits uniformly below the control (its best epoch 0.8955 vs the control's 0.8966 at two epochs), which is weakly consistent with a small real cost rather than pure selection noise, but the macro interval covers zero and checkpoint selection is still fold-0 selection.

Per-label: Synovitis -0.0130 [-0.0248, -0.0008] and Medial Meniscus -0.0098 [-0.0174, -0.0017] are the only intervals excluding zero (exploratory, unadjusted); MCL +0.0036, Lateral Meniscus +0.0042, Baker's +0.0074 up.

No decision or queue change. Contract checks pass on the completed artifacts, so the two open review concerns that forced orchestrator reporting (missing accounting/timing, unlogged diagnostics) resolve as follows: accounting is complete and matches the pinned limits, and the missing weight diagnostic remains an interpretation cost only, since the applied factors are reconstructible offline from class_balance.json plus the label file.

Queue membership and order stay as they are; nothing in this result reorders the remaining ideas. I admit no proposals on a completion event: the three admissible proposals were not supplied with content here, and the live queue already holds four ideas against a target of five.

The one forward-looking implication worth recording is negative rather than reordering: with the margin run (sharpening ambiguous targets, delta +0.00005) and this run (upweighting them, delta -0.00112) both landing inside noise, the "use the LLM label-confidence distribution to redistribute gradient" family has now been probed in both directions with active, verified mechanisms and no signal. A k-sweep of boundary weighting, or further target-confidence variants, has low expected value relative to the queued ideas. That is my judgment on expected value, not a closure of the idea.

Unresolved concerns:
- Synovitis has now fallen in all three completed candidates (-0.0079 ranking, -0.0094 margin, -0.0130 here) and Baker's has risen in all three. This is more consistent with the control's selected epoch being favorably/unfavorably noisy on those two labels than with any mechanism effect, and it means per-label diagnostics across this campaign are correlated, not independent evidence.
- Synovitis carries the largest boundary normalizer (Z=1.506, its negatives cluster near 0.5) and lost the most. The mechanistic hint that heavier reweighting hurts is confounded by the cross-run Synovitis pattern above and cannot be separated on this evidence.
- k=1.0 remains unswept, so this result does not formally exclude that another strength helps; the response need not be monotone in k. I judge a sweep low-value given the paired null with the margin run, but that is a judgment, not a measurement.
- Bootstrap uncertainty is conditional on the two fitted models and excludes training-seed and checkpoint-selection variation; a -0.0011 macro shift is inside plausible seed noise.
- No applied-factor or unweighted comparison loss was logged, so train_loss is not comparable to the control's curve and the realized in-batch weight distribution was never observed directly, only reconstructed from the label file. Future weighting candidates should log the realized factor.
- shared-ranking-memory-bank stays blocked by its unresolved dependency on the inconclusive shared-bce-ranking; its stale-score and refresh handling remain unreviewed.

Evidence:
- comparison/comparison.json (experiment dir shared-boundary-proximity-weighting-20260920T164732Z-1a061454): settings.allowed_config_differences limited to name/output_root/seed/trainer.loss.boundary_weighting.enabled; reference selected_epoch 9 macro 0.8965817, candidate selected_epoch 8 macro 0.8955109
- Candidate epoch curve in the same file: epochs 7-10 macro 0.89486 / 0.89551 / 0.89437 / 0.89525 vs reference 0.89658 / 0.89650 at its two best epochs
- Assignment facts.accounting: 10x93,846 train triplets and 3,503 visits per epoch, cohort b536b7e1..., validation 24,188 triplets with identical selection_sha256 92f1341ce45b14ee27882fd20892bd416b7945d85ddcd297055441fbc1784059 in all ten epochs
- Assignment facts.timing: setup 0.2977 s, training 1,562.373 s, validation prediction 20.217 s, 4,380 optimizer steps, peak CUDA 20,556,645,376 B; measured_compute_seconds 1,599.19 against the 1,797 s ceiling
- comparison/per_label.json and facts.per_label: Synovitis [-0.02477, -0.00083], Medial Meniscus [-0.01739, -0.00165]; all other label intervals cover zero
- History index entry for shared-target-logit-margin-20260920T161944Z-c1f07123: macro 0.89669, delta +0.00005, interval [-0.00391, +0.00388], mechanism verified active via train_mean_abs_logit 0.551 -> 1.899
- Prior review report orchestrator-20260920T165239Z-194f2eb1 for this experiment (group normalizers, k=1 pinned, resume-compatibility note on training/checkpoints.py:123)

## review-20260920T174940Z-cea31713

The implementation realizes cross-epoch draw invariance as intended given the sampling structure: each training study is drawn exactly once per epoch (dataset length = study count, 8×438 steps/epoch), so the term is inert in epoch 0 and active in epochs 1-9, with the anchor an exponential average (momentum 0.5) of that study's logits from prior epochs.

Interpretation for the eventual result: because a study's two compared draws are a full epoch apart, the penalized deviation mixes triplet-draw variation with genuine per-study learning between epochs. The drift EMA removes only a per-label global mean shift, not per-study progress, so this behaves closer to a temporal-ensembling/self-consistency regularizer than to a pure draw-invariance penalty. A macro difference will not separate "draw noise suppression helps" from "cross-epoch shrinkage of per-study logits helps".

Scale is non-stationary and unlogged in closed form: the denominator is a per-label EMA of between-study logit variance clamped at 0.25, initialized at 0, and the EMA half-life at drift_momentum=0.98 is about 50 steps (~1/8 epoch), not the several-epoch window the developer report assumed. Early training is therefore governed by the 0.25 floor, and the effective weight of the component decays as the logit spread grows. Per-label variance, drift, deviation and inactive-cell counts are logged per epoch, which makes this checkable after the run.

Cost is small: dict lookup over 8 UIDs plus elementwise ops on [8,12]; the main addition is 48 extra per-step self.log calls with epoch aggregation, which should be well inside the ~182 s of ceiling headroom left by the previous candidate but is not yet measured.

Reviewed changes.diff and the helper module directly, plus a static numeric exercise of the component with the supplied interpreter.

Contract: edits are confined to configs/training.yaml, config.py (new LossConfig + TrainerConfig.loss), lightning.py (init buffers, training_step add-on and logging) and the new training/losses/draw_consistency.py. Config differences are only the four trainer.loss keys.

Protected behavior preserved: data loading, model construction, configure_optimizers/scheduler, forward, validation_step, prediction writing, checkpoint save/selection and accounting are untouched in the diff. All four new buffers are persistent=False, so state_dict and checkpoint comparison are unaffected; validation is unchanged and reads no anchor state, so validation independence holds.

Correctness verified: the class weight lookup (targets>=0.5 → class_balance_weights[0]/[1]) matches masked_binary_cross_entropy_with_logits (lightning.py:112-118); logits are cast to float32 before any loss math; anchor/drift/variance enter the component detached (buffers, updated under no_grad), so gradients flow only through the current logits. Numeric check: epoch-0 (all unseen) component is exactly 0.0 with no active labels; after the update the second pass yields a positive component with gradient exactly zero at masked cells; per-label averaging skips labels with zero usable weight and divides by the active-label count.

Capacity and identity: training_studies = len(train_dataset) = studies.height (scripts/pipeline/train.py:147), and the dataset yields one item per study per epoch, so the sequential index map cannot overflow and study UIDs never repeat inside a batch. The observed mask is built from static per-study label columns (dataset.py:277-279), so the per-study "seen" flag combined with per-cell anchor initialization cannot activate a never-initialized anchor cell.

No leakage: anchors, drift and variance are computed only from training logits and training masks; no validation or gold labels enter. No extra data pass or model update; the added per-step work is elementwise on [8,12] tensors.

Queue and proposals unchanged; no new evidence bears on ranking, and the available proposals stay in reserve rather than enlarging the queue.

Unresolved concerns:
- The penalized quantity spans a full epoch between a study's consecutive draws, so it conflates triplet-draw variation with per-study learning progress; the drift EMA removes only a per-label global shift. Any macro effect cannot be attributed to draw-noise suppression alone.
- Effective component scale is non-stationary: the variance EMA starts at 0 and the 0.25 floor governs early training, while drift_momentum=0.98 gives a ~50-step (~0.12 epoch) half-life, not the several-epoch smoothing the developer report assumed. The chosen weight 0.1 is therefore not a stable effective weight.
- drift_momentum is reused for both the drift and variance EMAs; no separate variance momentum was specified by the idea, so the two timescales are tied by implementation choice rather than design.
- Per-label diagnostics log normalised_deviation as 0.0 for labels with no usable cells, which dilutes the epoch-mean diagnostic; read it together with the logged inactive-cell counts.
- All four state buffers are persistent=False, so a genuine checkpoint resume would silently restart with empty anchors and an inert first resumed epoch. Irrelevant for the planned fresh full-budget run only.
- Accounting, timing and evaluation artifacts do not exist yet (accounting_available=false); the triplet replay and the 1,797 s ceiling, including the 48 added per-step log calls, remain unverified until the run completes.

Evidence:
- changes.diff at outputs/autoresearch/losses/experiments/shared-draw-consistency-anchor-20260920T174358Z-295d5924/changes.diff (full reviewed diff: configs/training.yaml, config.py LossConfig, lightning.py init+training_step, new training/losses/draw_consistency.py)
- candidate src/knee_abnormality/training/losses/draw_consistency.py:33-62 (component: active=seen&observed, class-weighted squared deviation, per-label variance-floor normalisation, active-label averaging)
- candidate src/knee_abnormality/training/losses/draw_consistency.py:64-111 (no_grad state update: drift EMA on active cells, variance EMA on observed cells, anchor init-on-first-touch then momentum blend)
- candidate src/knee_abnormality/training/lightning.py:236-258 (non-persistent buffer registration and class-balance requirement guard)
- baseline src/knee_abnormality/training/lightning.py:104-118 (weight convention matched by the consistency term)
- baseline src/knee_abnormality/data/dataset.py:68 (len = studies.height, one draw per study per epoch) and :268-280 (static per-study target/observed columns)
- baseline scripts/pipeline/train.py:147,177-186 (training_studies = len(train_dataset), anchor-table capacity)
- developer report outputs/autoresearch/losses/jobs/developer-20260920T174359Z-c47fa68f/report.json (implementation choices; its drift_momentum smoothing-window justification is inaccurate)

## proposals-20260920T180215Z-b2bab486

Source verification of the second-pooling capability both proposals depend on:

- model.py:322-323 confirms forward == pool_encoded(encode_batch(batch)); both methods public, so no modeling edit is needed and the candidate file allowlist suffices.
- RNG neutrality holds but for a narrower reason than proposed. All stochastic draws live in encode_batch: series dropout (model.py:157-160, series_dropout=0.2 in configs/training.yaml:84) and reflection sampling (model.py:244-248). pool_encoded and _attention_pool draw nothing; the nn.Dropout at model.py:114 is p=0.0 (configs/training.yaml:83) and is a no-op. So extra poolings are RNG-neutral only while dropout stays 0.0, and a view must never re-enter encode_batch.
- pool_encoded raises ValueError when any study has no visually valid token (model.py:216), and the membership tensor is built over the whole batch. A half-view is therefore a full-batch pooling: every one of the eight studies must retain ≥1 valid token in every view, including studies the term itself excludes for having <2 valid acquisitions. Those studies must be duplicated into both halves (contributing nothing to the term) rather than masked out, or the step crashes. Neither proposal states this.
- Cost framing is sound. A view shares the tokens tensor and adds one attention map [8,12,N] plus one btn,nf->btf einsum (f=256); with N~10.5k that is ~0.26 GFLOP and ~4 MB per view against the rexnet_200 pass. Pooling is already forced to float32 by the autocast-disabled block (model.py:219), so the term inherits stable precision without extra work.

Interaction not flagged by the proposals: series_dropout=0.2 already removes evidence before the split, so a half-view can be a substantially degraded readout, which raises the per-view-accuracy risk that shared-subview-supervision creates and inflates the disagreement shared-split-view-consistency measures. Both should log realised acquisitions per view alongside the planned diagnostics so the term's operating point is recoverable.

Scope ruling (mine to make, both proposals asked): a second pooling of tokens already produced by one encode_batch call is allowed supporting loss computation. Verified in modeling/model.py:322-323 that forward is exactly pool_encoded(encode_batch(batch)), both public, so a candidate can split the primary forward inside lightning.py without touching modeling code (not candidate-editable). Encoder, attention module, classifiers, data preparation, triplet selection, augmentation, study exposure, batch composition, optimizer, schedule and inference stay fixed; no extra data pass, no extra optimizer update, validation path untouched. Conditions for review: exactly one encode_batch call per step, primary logits bit-identical, extra views built only by masking visual_valid, and the split RNG drawn from a dedicated generator so the augmentation and series-dropout streams are unchanged.

Both admitted (dependencies empty; admission table confirms). Ranking changed only to place them; reserved head unchanged.

shared-split-view-consistency ranked third rather than second: it targets the same headroom as the reserved head via a lag-free estimate, and the anchor run's logged per-label deviation/variance traces will say whether a draw-invariance term moves anything at all and at what effective scale. Sequencing it after that evidence avoids committing two full-budget runs to one mechanism family with zero measurements, and its coefficient/floor choices are exactly what the anchor traces inform. shared-group-variance-compression is independent of that evidence and already supplies the reviewed detached-normaliser pattern both ideas reuse.

shared-subview-supervision fourth: independent lever, no adaptive state and no collapse mode, but it imposes a stronger assumption (each half carries the label) that series_dropout=0.2 already stresses.

shared-tempered-balance stays last. The researcher's analysis (exhausted_directions item 2) shows per-label-constant reweighting leaves the within-label minimiser monotone, so it cannot change population AUC and acts only through finite-sample optimisation. Not parked — that finite-sample channel is the stated rationale and is untested — but it does not outrank a measured-headroom mechanism.

Closed directions recorded from this report: cross-label decorrelation/severity-factor terms, per-study logit normalisation, and the GCE/symmetric-CE/Huberised-BCE/forward-noise-correction gradient-profile family. No further treatment of the 0.25 unaddressed block beyond the queued bounded-target-instruction.

Unresolved concerns:
- Half-view pooling is a full-batch operation: pool_encoded raises if any of the eight studies has no valid token (model.py:216). Studies with <2 valid acquisitions must be duplicated into both views, not masked out. Neither proposal specifies this and it is a hard crash, not a quality issue.
- RNG neutrality of the extra poolings depends on model.dropout remaining 0.0 (configs/training.yaml:83); the attention Dropout at model.py:114 is otherwise a live draw. Review must confirm the candidate does not change it and calls encode_batch exactly once per step.
- series_dropout=0.2 already removes evidence before any split, so a half-view is a doubly-degraded readout. This inflates measured split-half disagreement and strengthens the hallucination pressure of per-view supervision; realised per-view acquisition counts must be logged.
- Both admitted ideas and the reserved head attack the same draw-noise headroom, whose +0.005 to +0.0075 estimate is an extrapolation from multiplier profiles of checkpoints that are not the current control (draw-noise-20260920.md section 1). A neutral anchor result does not by itself retire the family, but it bounds expected value for the successors.
- shared-ranking-memory-bank remains unadmitted: its only dependency shared-bce-ranking is completed/inconclusive, not provisional_accept. Its stale-score, refresh and detachment handling stay unreviewed.
- The exhausted-directions analysis closing the weighting axis is a single linear probe plus an analytic monotonicity argument on one fitted model's outputs; it justifies ranking shared-tempered-balance last but is not measurement, so the idea is not parked.

Evidence:
- baseline/source/src/knee_abnormality/modeling/model.py:322-323 - forward == pool_encoded(encode_batch(batch))
- baseline/source/src/knee_abnormality/modeling/model.py:242-305 - all stochastic draws (reflection, series dropout) occur in encode_batch only
- baseline/source/src/knee_abnormality/modeling/model.py:152-168 - _effective_presence applies series_dropout in training
- baseline/source/src/knee_abnormality/modeling/model.py:204-240 - _attention_pool membership over whole batch, ValueError on empty study (line 216), float32 autocast-disabled block (line 219)
- baseline/source/src/knee_abnormality/modeling/model.py:114 - nn.Dropout(model.dropout) inside attention; configs/training.yaml:83-84 - dropout 0.0, series_dropout 0.2
- Researcher report: outputs/autoresearch/losses/jobs/researcher-20260920T174359Z-e30eeb84/report.json (proposals, exhausted_directions items 1-4)
- Prior candidate cost/headroom evidence: shared-bce-ranking-20260920T112958Z-a2e23db7 comparison/comparison.json and its recorded ~52 s (3.3%) overhead
- Draw-noise headroom estimate and its caveats: draw-noise-20260920.md sections 1-2, 4

## completion-20260920T181036Z-45163c52

shared-prior-offset-bce is rejected as an improvement: macro AUC 0.88473 vs incumbent 0.89658, paired delta mean −0.01186, 95% report-group bootstrap [−0.01664, −0.00730] over 2,000 valid draws. Incumbent retained.

The per-epoch validation trace shows this is a uniform optimization deficit, not a late-training or selection artifact: candidate trails the control at every one of the ten epochs (0.79136/0.83539/0.85553/0.86328/0.87488/0.88036/0.88343/0.88278/0.88473/0.88449 vs 0.80574/0.84638/0.86054/0.88196/0.88123/0.89139/0.89338/0.89252/0.89658/0.89650), gap ~0.010–0.019 from epoch 0 onward, and both runs select the same late epoch. Combined with the pre-run analysis that the offset gives a materially milder rare-positive gradient tilt than the control's 1/(2f) weights (up to 6.95 for Fracture), the evidence favours "reduced effective rare-class emphasis at the fixed learning rate slows fitting" over "link-level prior handling is intrinsically worse". The flagged confound therefore stands: this rejects the implementation, not the additive-prior mechanism in isolation.

Per-label deltas are broadly negative and roughly track rarity: MCL −0.0364, Baker's −0.0221, ACL −0.0174 (all with intervals strictly below zero), Fracture −0.0164, Lateral Meniscus −0.0165, Contusion −0.0155, Lateral OA −0.0119; only PF OA +0.0056 and Effusion +0.0052 (prevalences 0.46 and 0.60, both intervals spanning zero) improved. The rare/frequent split is the signature expected from weaker rare-positive upweighting.

Offsets are correct: class_balance.json v2 records a_c = log(f_c/(1−f_c)) exactly (Fracture f=0.07194 → −2.5573). Candidate soft_target_bce 0.6019 vs reference 0.4857 reflects expected miscalibration of the unadjusted link and does not affect rank-based AUC.

Queue membership and order stay as they are. This rejection is informative for the ranking but does not reorder it: the one idea it touches, shared-tempered-balance, is already last. Tempering generally moves effective rare-positive emphasis toward 1, the same direction that just cost ~0.012 macro AUC, so it should be re-specified to sweep across (not merely below) the control's emphasis, or dropped, before any implementation; I am leaving it in place rather than parking it because the tempering family also contains sharpening.

No proposals admitted. shared-delayed-ranking, shared-bounded-target-instruction and shared-draw-reliability-weighting have satisfied dependencies but add no expected value over the five live entries right now; they stay in reserve. shared-ranking-memory-bank remains unresolved on shared-bce-ranking (inconclusive) and is therefore omitted from both lists.

Contract checks on the completed run: accounting complete and exact (938,460 training triplets, 35,030 visits, 24,188 validation triplets, 4,380 steps, cohort hashes matching the pinned limits); measured compute 1,596.05 s against the 1,797 s ceiling (+2.2% over control); no config differences outside name/output_root/seed; fresh recorded seed 4,159,665,861. The reject verdict rests on a valid run.

Unresolved concerns:
- The rejection cannot separate the link-level prior mechanism from effective rare-class step size at the fixed learning rate; the uniform per-epoch deficit makes the step-size explanation the better-supported one, but learning rate is not candidate-editable, so this confound is not resolvable inside the campaign.
- shared-tempered-balance moves rare-positive emphasis in the same direction that just failed. It should be re-specified to span emphasis above as well as below the control's weights, or dropped, rather than implemented as stated.
- Bootstrap uncertainty is conditional on these two fitted models and excludes training-seed variation; the deficit (~0.012, interval well clear of zero) is large relative to the seed noise implied by the earlier neutral candidate, so the direction is credible even so.
- No offset diagnostic or unadjusted comparison loss was logged, so train_loss is not comparable to the control's curve; offsets are recoverable offline from class_balance.json, leaving this an interpretation cost only.
- shared-ranking-memory-bank remains blocked by its unresolved dependency on the inconclusive shared-bce-ranking; its stale-score and refresh handling are still unreviewed.
- Runtime headroom is now the binding constraint on pair- and state-based successors: this run used 1,596 s of the 1,797 s ceiling despite adding no per-step loop.

Evidence:
- comparison/comparison.json (macro 0.88473 vs 0.89658; bootstrap mean −0.011858, 95% [−0.016635, −0.007303], 2,000 valid draws; allowed diffs name/output_root/seed only)
- training/fold_0/epochs/metrics_epoch_000..009.json (candidate per-epoch macro AUC) vs outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat/fold_0/epochs/metrics_epoch_*.json (reference)
- training/fold_0/class_balance.json version 2: prior_offset values equal log(f/(1−f)) per label (Fracture 0.071938 → −2.557289)
- training/fold_0/selected_checkpoint.json: best-epoch=08-val_macro_auc=0.88473, same late-epoch selection rule as the control
- assignment facts: measured_compute_seconds 1596.05 vs max_run_seconds 1797; preparation 0.17 s; seed 4159665861; accounting complete with matching cohort/triplet counts
- comparison/per_label.json and supplied per-label intervals (MCL, Baker's, ACL strictly below zero; PF OA and Effusion positive but spanning zero)
- Prior review of this candidate: jobs/orchestrator-20260920T172207Z-2510324f/report.json (rare-positive tilt analysis, 1/(2f) up to 6.95)
- Prior completion: jobs/orchestrator-20260920T121537Z-6ee62f96/report.json (shared-bce-ranking inconclusive; ~182 s ceiling headroom note)

## completion-20260920T184207Z-f1434206

The cross-epoch anchor trained actively and its logged traces resolve the scale concern raised at review. The consistency component (logged unweighted, weight 0.1) was inert in epoch 0 (mean inactive cells 8.00/8, active labels 0) and active with all 12 labels in epochs 1–9: epoch-mean component 1.070 → 0.171, i.e. 0.107 → 0.017 of a total loss falling 0.717 → 0.495 (≈15% → ≈3.5%). So the term was neither dead nor dominant, but its effective weight decayed roughly fourfold over training because the per-label variance EMA denominator grew from mean 0.34 (still near the 0.25 floor early, min 0.15) to mean 1.14 (max 2.06). The review concern that weight=0.1 is not a stable effective weight is confirmed empirically; normalised deviation fell 0.385 (epoch 1) → 0.116 (epoch 3) then drifted back up to ~0.18, and the per-label drift EMA stayed small (|drift| ≈ 0.05–0.09), so the anchor removed little global shift.

Cost is materially larger than the developer and review estimates: 1,712.0 s measured compute vs the 1,562.3 s control, +149.7 s (+9.6%), against +52 s (+3.3%) for the previous pairwise candidate. Only ~85 s of ceiling headroom remains relative to this candidate; a successor still has ~235 s over the control. Per-step logging of 48 scalars is the plausible driver but was not isolated, so future per-label diagnostics should be aggregated in buffers rather than logged per step.

Cross-experiment pattern (new): both completed regularizer candidates move the same labels in the same directions — Effusion (+0.0082, +0.0062), Medial OA (+0.0084, +0.0038) and PF OA/Lateral OA up or flat, while Lateral Meniscus (−0.0075, −0.0213), MCL (−0.0042, −0.0080) and Synovitis (−0.0079, −0.0063) go down; here ACL also falls −0.0177. Adding a shared smoothing/consistency term appears to trade focal ligament and meniscus discrimination for diffuse, prevalent OA/effusion signal. This is exploratory and unadjusted, but it is consistent across two independent objectives and is the most useful signal for designing the queued subview ideas.

Checkpoint selection picked epoch 9 (0.89357) with a monotone validation curve, so no early-stopping artifact.

No decision changes. The result is inconclusive by the fixed criterion (macro delta mean −0.00304, 95% paired interval [−0.00679, +0.00062] spanning zero), so the historical control is retained as incumbent. Accounting replay is exact (938,460 training triplets / 35,030 visits, 4,380 steps, matching cohort hashes; validation selection hash constant across all ten epochs), config differences are confined to the four allowed `trainer.loss` keys, and runtime 1,712.0 s sits under the 1,797 s ceiling, so there is no contract breach — the run is valid, not invalid.

Queue membership and order are unchanged (ranked_ids null, park_ids empty): the reserved head is developing and the remaining entries are unaffected by this outcome. Two queued ideas (shared-split-view-consistency, shared-subview-supervision) target within-step subview invariance rather than cross-epoch anchoring, so this result does not by itself demote them, but the per-label pattern below plus the measured cost are directly relevant when they are reviewed.

I did not re-derive the completion reporter's extraction; the escalation was due to unresolved implementation concerns, which the logged diagnostics now largely resolve or sharpen (see finding).

Unresolved concerns:
- Attribution remains unresolved as predicted at review: a study's compared draws are a full epoch apart, so the penalized deviation mixes triplet-draw noise with per-study learning; the small logged drift (|drift| 0.05-0.09) shows the drift EMA removed little, so this behaved as cross-epoch shrinkage, not draw-invariance. The near-negative macro result does not falsify draw-invariance as a mechanism.
- Effective component weight decayed roughly fourfold (15% -> 3.5% of total loss) as the variance denominator grew 0.34 -> 1.14; the tested configuration is therefore a decaying schedule, not weight 0.1. A constant-effective-weight or larger-weight variant is untested, but is not obviously worth a full-budget run given the negative-leaning mean.
- Runtime overhead is +149.7 s (+9.6%) over control, about three times the previous candidate's and well above the review's 'small cost' estimate; 48 per-step self.log calls are the likely cause but were not isolated. Queued subview ideas add real forward-pass work and must budget within ~235 s over control.
- ACL (-0.0177, interval [-0.0319, -0.0042]) and Lateral Meniscus (-0.0213, [-0.0374, -0.0065]) are individually clearly negative; these are unadjusted exploratory intervals and cannot override the macro verdict, but the consistent OA/effusion-up, ligament/meniscus-down pattern across two candidates deserves weight in future design.
- Bootstrap uncertainty is conditional on these two fitted models and excludes training-seed variation; a -0.0030 macro shift is within plausible seed noise on fold 0, so 'no effect' and 'a small real harm' are not separated.
- State buffers remain persistent=False, so a genuine checkpoint resume would silently restart with empty anchors. Harmless for this completed fresh run, but the defect is now in the accepted-source lineage if this file is reused.

Evidence:
- evaluation.json and comparison/comparison.json: macro 0.89357 candidate vs 0.89658 reference, delta mean -0.0030, 95% [-0.00679, +0.00062], 2000 valid draws (experiments/shared-draw-consistency-anchor-20260920T174358Z-295d5924/)
- training/fold_0/logs/version_0/metrics.csv: train_consistency_component_epoch 1.0696 -> 0.1715, active_labels 12 in epochs 1-9 and 0 in epoch 0, train_loss_epoch 0.7171 -> 0.4951, train_balanced_bce_epoch 0.6453 -> 0.4779
- Same metrics.csv, per-label train_consistency_variance_* : epoch-mean 0.340 -> 1.144 (min 0.15, max 2.06); train_consistency_inactive_* epoch-0 mean 7.998/8, zero thereafter; deviation epoch-mean 0.385 -> 0.116 -> 0.180; |drift| 0.048-0.093
- training/fold_0/timing.json: setup 0.310 s, training 1674.53 s, validation prediction 20.77 s, 4380 optimizer steps, peak CUDA 20.36 GB; measured_compute 1711.997 s vs ceiling 1797 s and control 1562.283 s
- training/fold_0/selected_checkpoint.json and metrics.csv val_macro_auc: monotone 0.8171 -> 0.8936, epoch 9 selected at 0.893568
- triplet accounting in assignment facts (and training/fold_0/triplet_accounting.json): 93,846 triplets / 3,503 visits per training epoch with constant cohort hash b536b7e1..., validation 24,188 / 904 with identical selection hash 92f1341c... in all ten epochs
- comparison/per_label.json: ACL -0.0177 [-0.0319,-0.0042], Lateral Meniscus -0.0213 [-0.0374,-0.0065], Effusion +0.0062, PF OA +0.0066, Medial OA +0.0038, Fracture +0.0058
- Prior candidate for cross-experiment comparison: experiments/shared-bce-ranking-20260920T112958Z-a2e23db7/comparison/comparison.json (Effusion +0.0082, Medial OA +0.0084, Lateral Meniscus -0.0075, Synovitis -0.0079; +52 s runtime)

## user-adopt-learned-strength-20260921T115849Z-b19e375d

User adopted learned anatomical strength as the reference and cleaned source for future experiments; campaign remains stopped.

Matched-seed control gain +0.0023308413 with positive 80% interval. This is an explicit reference replacement, not a retroactive campaign verdict. Original limits, baseline snapshots, historical run references and decisions are preserved. Pending old-reference events are archived; no jobs launched.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/adopted/user-adopt-learned-strength-20260921T115849Z-b19e375d/adoption.json
- /home/claudio/src/knee-abnormality/outputs/analysis/anatomical-strength-vs-control-repeat-ci-levels/adoption_assessment.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/losses/decisions/user-adopt-learned-strength-20260921T115849Z-b19e375d/pending-events.json
