# Campaign findings

## proposals-20260921T215250Z-888fe844

Source check (baseline `sampling.py:469-489`): training percentiles are `(arange(count) + U)/count`, with U drawn from the per-type "targets" stream. Suppliers and band order use separate per-type streams. Validation uses the fixed 0.5 offset and sorts by target. So plane-coupled-depth-targets only has to re-key one stream. But when a plane's two slots have different `count`, the shared RNG gives the same leading uniforms divided by different denominators, so they will not be co-located. The co-location gain has to be measured on the capacity-limited slots, not assumed.

Physical-span and anatomy-anchored placement both change prepared pixels. They therefore change the deterministic validation identity, and anatomy placement also changes inference. Their cache re-preparation and landmark cost count against the runtime ceiling. The other three ideas must reproduce the incumbent's validation selection exactly.

I admitted all five imported ideas. None has a dependency, and all fit the sampling scope and the editable files. The queue was empty, so this ranking builds it.

1. physical-span-standardized-triplets: It rests on measured spacing spread. It adds no compute, is a fixed standardization rather than a tunable, and changes channel content in every dense series. It is the cleanest test that has never been run.
2. anatomy-anchored-crop-placement: This one has the largest potential effect. The joint line sits a median 12.8 to 17.5 mm off centre, and edge labels are sensitive to the crop. It ranks second because the joint-line heuristic has never been validated against ground truth. It also changes the preprocessing identity and inference, and the landmark cost counts against the 1,847 s ceiling. Its label-free coverage check should come before any GPU run.
3. run-level-stratified-jitter: It is cheap. It gets its first test at the incumbent's seed, on the new split and labels, and the result decides the whole draw-coupling family.
4. plane-coupled-depth-targets: The premise is confirmed in source. At baseline `sampling.py:471-477`, the "targets" stream is seeded with the full `kind`, so a plane's two slots draw independent jitter. Validation uses the fixed 0.5 offset. The expected effect is small and of unknown sign.
5. model-feedback-depth-targets: This is the most speculative idea. Earlier attention-based selection gained only +0.004, and attention entropy was about 0.97. It also touches the training loop and the persistent workers.

Unresolved concerns:
- physical-span: the p1 candidate gap of 0.03 mm probably comes from degenerate or duplicate positions, not true dense acquisitions. Inspect it before choosing the stride, or large strides may pair unrelated slices. Re-measure the anchor span on the sealed cohort; do not reuse 7.11 mm from the old cache.
- anatomy-anchored-crop-placement: the joint-line heuristic has only a cross-plane agreement check (96-98% within 10 mm), with no ground truth. It needs the label-free reach and patella-clipping check first. The up-down offset must come from anatomy measurements, never from AUC.
- anatomy-anchored-crop-placement: the landmark must run inside the measured process and at inference, through the shared dataset path. Re-preparing the cache plus the landmark could approach the 1,847 s ceiling.
- plane-coupled-depth-targets: when a plane's two slots have different counts, the shared jitter does not align them. Measure realized co-location against the incumbent's training figures (0.60/0.57/0.58) and validation figures (0.93/0.95/0.94).
- model-feedback-depth-targets: the per-epoch map has to reach persistent workers without adding stalls. Attention is computed on augmented training inputs. The mixture floor must be fixed before the run, not swept.
- All results come from single matched-seed runs, and a same-seed sampling perturbation has moved fold-0 AUC by up to 0.0031. Interpret gains with that noise in mind, without changing the adoption rule.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/baseline/source/src/knee_abnormality/data/preprocessing/sampling.py:469-499 (per-kind RNG streams: targets, suppliers, band_order)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/baseline/source/src/knee_abnormality/data/preprocessing/sampling.py:21-29 (SAMPLING_IDENTITY targets and suppliers strings)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/baseline/source/src/knee_abnormality/data/preprocessing/spatial.py:62 (resample_crop, the insertion point for the anatomy offset)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/imports/ideas-20260921T215249Z-b680c878/report.json
- docs/validation/results/sampling/2026-09-17-ranked-filtered-result.md (same-seed sampling noise of up to 0.0031)

## proposals-20260921T215250Z-f9c94d51

Two user-imported proposals are held as proposed surplus, not queued, because the five-idea queue is full with ideas ranked higher. The earlier inconclusive run of epoch-paced-bag-size is not evidence against it: that run was unpaired (fresh seed) and compared against a withdrawn incumbent. The training loader currently shuffles uniformly (lightning.py:187-188), so label-diverse-batch-composition would replace only that shuffle. Validation (shuffle=False, lightning.py:203) would be unaffected.

Both imported proposals are in scope and pass admission because neither has dependencies. The live queue already holds the five-idea target, and both proposals describe themselves as lower priority, with small or speculative expected effects, than every queued seed idea. I leave them proposed as surplus and do not displace any queued idea. The queue is unchanged, so ranked_ids is null. Either proposal can be admitted when a queue slot opens.

epoch-paced-bag-size: its only valid run was inconclusive (0.895606). That run used a fresh seed against the withdrawn lucky incumbent, so it is weak evidence either way, and the mechanism is not a previously rejected direction. The contract allows moving triplets between epochs as long as the run total stays at the sealed 937,810 with zero tolerance. The proposal's own concavity argument predicts a slightly worse average bag, so any gain has to come from training dynamics.

label-diverse-batch-composition: this falls under the in-scope 'how studies are ordered into batches'. The current training loader shuffles uniformly (`shuffle=True`, lightning.py:187-188), so a stratified order changes only batch membership and order. Visits per epoch and triplet counts stay the same. It is the cheapest idea available and a good fill-in candidate.

Unresolved concerns:
- epoch-paced-bag-size: the large-bag epochs raise peak activation memory (about +19% tokens per study at +1 on every slot) and may lengthen epochs. The replay must show the realized run total exactly equal to 937,810 and a wall clock within the 1,847 s ceiling.
- label-diverse-batch-composition: this could clash with the class_balance_exponent 0.5 weighting only if the implementation touches the weights. Weights must stay fixed, and the stratified order must be drawn from the training seed and epoch, using training-fold labels only.
- The two mechanisms are small, and the contract notes same-seed sampling noise of up to 0.0031 in macro AUC, so a single-run gain of 0.001 or more from either is hard to tell apart from noise.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/imports/ideas-20260921T215250Z-0dcf6942/report.json
- /home/claudio/src/knee-abnormality/autoresearch/campaigns/sampling/seed-ideas-2.yaml
- src/knee_abnormality/training/lightning.py:187-203 (train shuffle=True, val shuffle=False)
- outputs/autoresearch/triplet-sampling/experiments/epoch-paced-bag-size-20260919T200307Z-0bdb46b8 (inconclusive, fresh seed)
- docs/validation/results/sampling/2026-09-17-ranked-filtered-result.md (same-seed sampling noise up to 0.0031)

## review-20260921T220856Z-edb9a511

Gate failure is a naming/policy issue, not a mechanism defect: `data.sampling_half_span_mm` fails the dotted-prefix match for `data.sampling`. The stride generalization is otherwise implemented coherently and falls back to stride 1 for thick-slice runs. With stride >1 a run has len−2·stride candidate triplets instead of len−2. Short, finely spaced runs therefore offer fewer distinct triplets, which can matter for the fixed per-study budgets and for the deterministic x1 validation count. Only gate accounting can confirm the counts. Validation inputs change by design (thin-slice series get wider-span triplets), so the selection identity must record this.

The gate failed, so this cannot be approved. The recipe adds `data.sampling_half_span_mm`, and the policy prefix `data.sampling` only admits `data.sampling` or `data.sampling.*` (gates.py:82 checks `startswith(f"{prefix}.")`). The underscore key is outside the allowed keys. The mechanism itself looks sound, so I am asking for one bounded repair rather than blocking.

Repair: remove the `sampling_half_span_mm` field from config.py and configs/training.yaml. Make the 3.555 mm half-span a fixed module-level constant in sampling.py, and drop the argument from `build_pool` and from the dataset.py call. That matches the repo rule that accepted decisions are fixed behavior, not options. Add the numeric value to the `triplet_span` string in SAMPLING_IDENTITY so the changed validation selection is recorded. Keep the rest of the diff as it is. After the repair, the recipe should be byte-identical to the parent apart from anything already allowed. Re-run the gates so that accounting (training visits 35,020, training triplets 937,810, validation triplets 24,253 at x1) is actually produced. The current facts show accounting_available=false and config_available=false.

Substantive review of the diff so far. The stride is derived per run as round(3.555 / median gap), clamped to (len−1)//2. Runs always have ≥3 slices (sampling.py:226), so stride ≥1. It reduces to the old consecutive triplets when spacing ≥ about 2.37 mm. The support envelope generalizes the old formula correctly: the midpoint cell for a single middle, and the min/max of the middles otherwise. The preparation.py check now accepts any consistent positive stride. The added randomness doesn't touch the RNG streams, and validation stays unseeded. Labels, predictions and validation data don't enter the stride rule, so I see no leakage. The added cost is negligible (preparation took 0.12 s).

Unresolved concerns:
- Accounting and config facts are unavailable (accounting_available=false), so it is unverified whether the exact training triplet count (937,810), visit count (35,020) and x1 validation count (24,253) still hold now that stride >1 shrinks per-run candidate pools.
- The origin of the 3.555 mm half-span is undocumented in the diff. It must come from measured acquisition geometry, not from a score sweep; the developer should state its source.
- scripts/adhoc/analyze_physical_sampling.py:194 calls build_pool with the old signature. It is outside editable files, but it breaks if the signature changes; removing the parameter (the requested repair) avoids this.
- The run median gap uses depths[i] differences; a run with non-finite depths silently falls back to stride 1. Acceptable, but coverage diagnostics should report how many runs got stride >1.

Evidence:
- /home/claudio/src/knee-abnormality/src/knee_abnormality/autoresearch/gates.py:82 (prefix match requires `prefix.`)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/physical-span-standardized-triplets-20260921T220140Z-dbfa1835/changes.diff
- candidate source src/knee_abnormality/data/preprocessing/sampling.py:268-300 (stride derivation, support, triplet construction)
- candidate source src/knee_abnormality/data/preprocessing/preparation.py:125-131 (consistent-stride validation)
- candidate source src/knee_abnormality/config.py:120 and configs/training.yaml:42 (out-of-policy key)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/developer-20260921T220140Z-725cc906/report.json

## review-20260921T221636Z-1c4adc80

Per run, stride k = clamp(round(3.555 / median gap), 1, (L−1)//2), which cuts the candidate windows in that run from L−2 to L−2k. Groups are keyed on middle-slice geometry, so capacity can only shrink. select_triplets draws min(budget, groups) per slot (sampling.py:474), so any slot where the smaller capacity falls below its 15–18 budget changes the realized training and validation triplet totals. The idea specified reducing k until min(budget, groups_k) = min(budget, groups_1); this candidate does not implement that. Runs with sub-mm gaps and few slices, or runs split at failed paths, clamp to one or a few middles and are the most exposed. Expected exposure is low: 2D series have a median of 30 slices at 3.5 mm, and 3D series are long. It is unmeasured, though, because no CPU replay of this candidate's pools was run. Without the guard, a count mismatch will make the run invalid, not merely inconclusive. If it fails that way, the retry should add the capacity-preserving reduction of k, not change the span.

The prior gate failure is resolved. The out-of-policy key data.sampling_half_span_mm is removed, the half-span is a fixed module constant, and gate_error is null. I found no correctness, leakage or validation-independence defect. The stride comes only from each run's measured slice depths, with no labels or predictions involved. Validation still goes through the unseeded path, so it stays deterministic and independent of the training seed, and its changed selection is captured in the identity. The added preparation cost is negligible (0.17 s). The middle slice still sets depth, so peripheral reach and exploitation (targets, widths, supplier rules) are unchanged; only channel content changes. In practice the mechanism affects series with spacing below about 2.37 mm, mainly the 3.6% of series that are 3D (0.4–1 mm, stride 4–9, typically with long runs). The median 2D series at 3.5 mm stays at stride 1. The 3.555 mm half-span is half the measured 7.11 mm median realized span, so it comes from acquisition geometry rather than a score sweep. The implementation leaves out the idea's stated capacity safeguard (see concerns). The zero-tolerance accounting gate (gates.py:157-183) catches any count change after the run, so the omission risks an invalid run, not a silently misread result. This event is already the one allowed repair, and the expected shortfall exposure is small, so I approve rather than block.

Unresolved concerns:
- The capacity-preserving stride reduction from the idea specification (reduce k until min(budget, groups_k) equals min(budget, groups_1)) is not implemented. Any slot pushed below budget changes training_triplets (937,810) or validation_triplets (24,253), which have zero tolerance, and makes the run invalid.
- Accounting is available only after training (training/fold_0/triplet_accounting.json), so exact count preservation is unverified before launch.
- Runs with very small within-run gaps (valid_runs allows gaps down to 0.01 mm) clamp the stride to (L−1)//2 and collapse to one or two middles. That is a coverage loss for such runs, and the sampler logs no per-run stride diagnostic to show how often it happens.
- scripts/adhoc/analyze_physical_sampling.py still assumes consecutive triplets in its analysis. It is outside the editable files and has no training impact.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/physical-span-standardized-triplets-20260921T220931Z-71613a41/changes.diff
- candidate source src/knee_abnormality/data/preprocessing/sampling.py:268-300 (stride, support, triplet construction); :74-96 (valid_runs gap bounds); :474 (count = min(budget, len(groups)))
- candidate source src/knee_abnormality/data/preprocessing/preparation.py:114-131 (distinct middles, consistent-stride check)
- /home/claudio/src/knee-abnormality/src/knee_abnormality/autoresearch/gates.py:150-184 (post-run exact accounting gate)
- /home/claudio/src/knee-abnormality/autoresearch/campaigns/sampling/ideas.yaml:3-32 (coverage clause requiring capacity-preserving k)
- /home/claudio/src/knee-abnormality/docs/research/geometry-and-views/input-resolution.md:45-46 (spacing median 3.5 mm, 30 slices, 3.6% 3D series)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/developer-20260921T221451Z-fcbcf735/report.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/orchestrator-20260921T220856Z-354bd177/report.json

## review-20260921T223613Z-a58173d9

Probe results from 30 real studies (anchor before clipping, in mm; order is row then column). Axial: median 3/−1, 5–95% range [−4, 12] / [−7, 7]. Coronal SI: median 12, range [−34, 25]. Sagittal SI: median 9.5, range [−9, 37]. The zero-anchor fallback rate is ≤11% per axis. The median FOV is 160–169 mm, so clipping limits any shift to about ±10–15 mm. The expected effect is therefore a modest, systematic downward shift of the window toward the joint in sagittal and coronal series. The mechanism tests a small anatomical recentring, not a large change of field of view. In training the reference slice is the middle of the drawn slices, so the anchor varies slightly between epochs. Validation uses one fixed anchor.

The candidate implements the assigned mechanism: it places the fixed 140 mm window from anatomy, meaning the joint line along the physical SI axis for sagittal and coronal series and the tissue-extent midpoint otherwise, with the anterior edge preserved when AP tissue exceeds the window. Crop size, spacing, FOV, labels, folds and scoring are unchanged. Validation stays deterministic because the anchor depends only on the pixels of a reference slice chosen by a fixed rule from the deterministic selection, and it does not use the training seed. There is no leakage: the estimator takes no labels or metadata beyond orientation. The developer measured no per-study overhead (0.123 s vs 0.124 s per study).

I ran a CPU probe on 30 real training studies (195 series; the estimator was wrapped only to log its results). Fallbacks are rare (0–11% per axis), and the median SI offsets are plausible: coronal +12 mm, sagittal +9.5 mm, consistent with the documented joint position about 15 mm below centre. There are outlier detections: coronal 5th percentile −34 mm and sagittal 95th percentile +37 mm. _crop_geometry clips them to ±(FOV−140)/2, which is about ±10 mm at the median 160–169 mm FOV. That bounds both the potential harm and the potential gain, and the realized crop change is at most about 16 px for most series. No gate failed. The remaining issues are minor and do not justify a repair.

Unresolved concerns:
- Joint-line detection produces outliers in the wrong direction (coronal 5th percentile −34 mm, sagittal 95th percentile +37 mm). Clipping bounds them to ±(FOV−140)/2, but the estimator applies no plausibility clamp to the offset.
- The per-slot cache survives preparation retries: after a failed path, newly decoded slices of the same series get an anchor from a different reference subset than slices already cached, so crops within one series can be inconsistent. Retries are rare; the effect is unmeasured.
- In training the anchor's reference slice depends on the stochastic draw, adding small crop jitter between epochs that validation does not have. This is not augmentation by design, but it does perturb the training inputs.
- scripts/adhoc/profile_threaded_loading.py overrides _decode_series with the old three-argument signature and would break. It is outside the editable files and has no effect on training.
- The anterior-preservation branch depends on the LPS sign of IOP. It was verified synthetically, not against real headers; in the probe, AP tissue extent rarely exceeds 140 mm.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/anatomy-anchored-crop-placement-20260921T221837Z-3b5416fc/changes.diff
- candidate source/src/knee_abnormality/data/preprocessing/spatial.py:33-50 (clipped shift in _crop_geometry), :231-260 (estimate_anchor), :300-308 (output_to_native includes the shift)
- candidate source/src/knee_abnormality/data/preprocessing/preparation.py:104-151 (decode_series: reference frame nearest the middle, one anchor per call)
- candidate source/src/knee_abnormality/data/dataset.py:187-217 (per-slot cache shared across retries)
- candidate source/src/knee_abnormality/identity.py:16-47 (hash of preprocessing sources changes the identity)
- candidate source/src/knee_abnormality/inference/engine.py:229 (inference uses KneeStudyDataset, so the anchor also runs at inference)
- /tmp/anchor_probe.py (orchestrator CPU probe: 30 training studies from sample_dataset, 195 series)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/developer-20260921T221837Z-c52338e5/report.json

## completion-20260921T224455Z-5fedd8cd

Standardizing each triplet's through-plane half-span to 3.555 mm gave no macro gain at fixed counts (−0.00047). The labels split in a structured way, within the known same-seed sampling noise of up to 0.0031 macro. Ligament labels rose: ACL +0.0098 (95% interval [−0.0005, 0.0218]) and MCL +0.0129 [−0.0065, 0.0327]. Lateral Meniscus rose +0.0076. PF OA fell −0.0163, the only label whose 95% interval excludes zero ([−0.0281, −0.0048]). Contusion fell −0.0116. This is consistent with a wider physical context helping to trace ligaments along their length. It may cost fine through-plane detail for cartilage and bone surfaces, which matters most in thin-slice 3D series, where the stride grows most. The per-label intervals are exploratory and unadjusted for multiplicity, so this is a hypothesis only. A span that varies by plane or series type might keep the ligament signal without the PF OA loss. It would need a new idea with a physically motivated span, not a sweep. The validation selection identity changed, as expected, because the triplet channels changed: the candidate's is b7f8338f…, deterministic across all epochs, while the baseline replay's validation selection hashes differ. The zero-tolerance counts were preserved even without the capacity guard. For the same span rule, the capacity-shrink risk has now been shown not to bind on this cohort.

This is a completion event, and the fixed eligible_decision is reject. The run is valid. Accounting is complete: 937,810 training triplets, 35,020 visits and 24,253 validation triplets, all matching the sealed counts exactly. Compute was 1,575 s against the 1,847 s ceiling. The main review concern was that the missing capacity-preserving stride reduction could shrink slot capacity below budget and invalidate the run. That did not happen on this cohort. The concern is resolved empirically, not by the code. Macro AUC was 0.898569 against 0.899042, a delta of −0.00047. The central 80% interval [−0.0029, +0.0022] straddles zero, so the result is a clear null-to-negative outcome. No ranking change is needed: the remaining queued ideas do not depend on this mechanism.

Unresolved concerns:
- There is no per-run stride diagnostic, so the share of runs that clamp to (L−1)//2 and the resulting coverage loss in sub-mm-gap series are still unmeasured. The PF OA drop may come from that collapse rather than from the span itself.
- The capacity guard is still missing from the code. Counts matched on this cohort, but any later idea built on this source should add it rather than rely on this result.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/physical-span-standardized-triplets-20260921T220931Z-71613a41/comparison/comparison.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/physical-span-standardized-triplets-20260921T220931Z-71613a41/comparison/per_label.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/baseline/triplet_accounting.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/orchestrator-20260921T221636Z-c521d628/report.json
- docs/validation/results/sampling/2026-09-17-ranked-filtered-result.md (same-seed sampling noise ≤0.0031)

## review-20260921T224801Z-710e4f6f

The diff implements cross-epoch stratification correctly. For each study and kind, one permutation of 10 sub-strata per band is shared across the 10-epoch cycle, so over the run every within-epoch band covers its full width once, in 1/10 slices. This is a coverage-only change: no exploitation signal and no validation change. The only defect is gate-level: a new top-level data field outside the `data.sampling.` path namespace. Preparation cost is negligible (one extra small RNG per selection). No accounting or config facts exist yet because the gate failed before replay.

The candidate fails the recipe gate, so it cannot be approved. `validate_config` (src/knee_abnormality/autoresearch/gates.py:78-84) accepts a prefix key only when it equals the prefix or starts with `prefix.`. The flat key `data.sampling_stratification_depth` matches neither form of `data.sampling`, and it is not in `config_keys`, so the gate rejects it.

The sampling mechanism itself looks correct and on topic:
- The permutation is drawn once per cycle (`rng("strata", cycle)` with cycle = epoch // depth) and indexed by `phase = epoch % depth`.
- With depth 10 over 10 epochs, each within-epoch band therefore lands in each of 10 sub-strata exactly once across the run. The uniform jitter inside each sub-stratum stays random.
- The validation branch is unchanged: deterministic percentiles, independent of the training seed.
- Counts and budgets are untouched and the identity string is updated.

Requested repair (one concrete change): remove the new recipe option.
- Delete `DataConfig.sampling_stratification_depth` from config.py and `data.sampling_stratification_depth` from configs/training.yaml.
- Make depth fixed behavior equal to the run's epoch count, 10. Take it from the existing trainer epoch setting if the dataset already receives it; otherwise use a module constant in sampling.py.
- Drop the `stratification_depth=1` default parameter branch or keep it internal. Either way, no config field.

This follows the repo rule that an accepted implementation is fixed behavior, not a configurable alternative, and it leaves the recipe unchanged so the gate passes. Keep the metadata field and the identity rename. The integration declaration (add mode, nothing removed) is otherwise consistent with the empty `active_incumbent_ideas`.

Unresolved concerns:
- The row-wise permutation assumes `count` (triplets per band set) is the same for a study and kind in every epoch. If budget redistribution changes `count` between epochs, rows no longer map to the same bands and exact once-per-run coverage breaks. After the repair, the developer should confirm that the budgets are constant across epochs.
- The same idea was rejected in the stopped triplet-sampling campaign (run-level-stratified-jitter-20260919T183826Z-f211c7ce), under the old labels and split. Expect a small effect relative to the ~0.003 same-seed sampling noise.
- Realized accounting (training visits 35020, triplets 937810) is unverified because the gate failed before replay.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/run-level-stratified-jitter-20260921T224525Z-4957db3c/changes.diff
- /home/claudio/src/knee-abnormality/src/knee_abnormality/autoresearch/gates.py:72-87 (prefix match requires `prefix.`)
- candidate sampling.py select_triplets ~L439-490 (strata permutation, validation branch unchanged)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/developer-20260921T224526Z-ff9bc7d5/report.json

## review-20260921T225137Z-66de33ef

Approved diff: cross-epoch stratification with depth fixed at 10, matching the 10-epoch schedule. For each study, kind and band, one permutation of 10 sub-strata is drawn per cycle from an epoch-independent stream and indexed by epoch phase. Uniform jitter stays inside the assigned 1/(10·count) sub-interval of percentile space. Over the run, each band's percentile interval is covered once in tenths, and the truncated-Gaussian target mapping is unchanged. Validation selection identity is unchanged: the deterministic midpoint percentiles are untouched. Only the training identity string changed. The realized accounting replay is not yet available (accounting_available=false).

The repair does the single thing the previous review asked for. The recipe option is gone and depth is fixed behavior (a module constant of 10), so the recipe gate now passes (gate_error null, no config differences). The mechanism is on topic: stratified depth coverage across epochs. It is coverage-only, with no exploitation signal.

Correctness:
- Lightning's current_epoch is 0-based (lightning.py:306-310 via dataset.set_epoch). Epochs 0-9 therefore share cycle 0, and each within-epoch band visits each of its 10 sub-strata exactly once over the run.
- `count = min(budget, len(groups))` comes from study_budgets, which depends only on per-slot source counts (dataset.py:169-178, 256), not on the epoch. Rows map to the same bands every epoch. This resolves the earlier concern about constant budgets, except when preparation retries change the pool.
- The validation path is unchanged, deterministic and independent of the training seed.
- No labels or model feedback are used, so there is no leakage path.
- Selection counts are unchanged by construction: count and budgets are untouched and only target positions move.
- Added preparation cost is one small RNG and permutation per selection (preparation 0.17 s).

The queue is unchanged. I admitted no proposals this event; this is a review, not a proposal event.

Unresolved concerns:
- Exact coverage across the run assumes the same pool and count for a study and kind in every epoch. A preparation retry that swaps sources or changes geometry_groups capacity for one epoch breaks the row-to-band mapping for that study. This is rare and its effect is unmeasured.
- The realized training visits (35020) and triplets (937810) have not yet been verified by replay (accounting_available=false). The counts should be unchanged by construction.
- The same mechanism was rejected in the stopped triplet-sampling campaign (run-level-stratified-jitter-20260919T183826Z-f211c7ce) under the old labels and split. The expected effect is small relative to the ~0.003 same-seed sampling noise in fold-0 macro AUC.
- Depth is hardwired to 10 rather than read from trainer.max_epochs. If the epoch schedule ever changed, coverage would silently become partial or span several cycles. The schedule is fixed under this contract.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/run-level-stratified-jitter-20260921T224831Z-d1fcb8e3/changes.diff
- candidate src/knee_abnormality/data/preprocessing/sampling.py:38 (RUN_STRATIFICATION_DEPTH), :473-488 (strata permutation, training/validation branches)
- candidate src/knee_abnormality/data/preprocessing/sampling.py:413-433 study_budgets and :449 count=min(budget,len(groups)): epoch-independent
- candidate src/knee_abnormality/data/dataset.py:169-178,198-206,256 (budgets from source counts; epoch passed via shared Value)
- candidate src/knee_abnormality/training/lightning.py:306-310 (0-based current_epoch -> set_epoch)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/orchestrator-20260921T224801Z-7d2de893/report.json (repair request satisfied)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/developer-20260921T224832Z-6abe5eca/report.json

## proposals-20260921T230559Z-e6d7e514

The admitted idea gives denser per-visit bags (up to +6 triplets, capped at slot capacity − 1) to the third of studies with the highest EMA misfit. The misfit is class-balance weight × (BCE − H(y)), taken from training batches. The bottom third pays for it, and every present slot keeps at least one triplet. The type mix and depth law are unchanged. It starts at epoch 1, and epoch 0 stays on the incumbent.

Checks against source:
- The baseline model pools each series with a masked softmax over its slices (baseline source src/knee_abnormality/modeling/model.py:243-247). The pooled feature is therefore close to invariant to bag size, which weakens any density shortcut the model could learn in training but not use at validation.
- k_05's peak_cuda_allocated_bytes is 20,999,941,632 (19.56 GiB), which confirms that uniformly denser budgets are memory-bound. The per-batch planned-total cap therefore has to be measured in the replay, not estimated.

Misfit is label-derived, so recipient status will correlate with rare-label positives. Among recipients, density is weakly confounded with the label. This is the same shortcut class the researcher lists as exhausted for label-conditioned allocation. Softmax pooling limits it but does not remove it.

I'm admitting misfit-steered-study-bag-density. It has no dependencies, and proposal_admission shows it is satisfied. It is the only registry idea that uses the contract's between-study triplet redistribution, and it does so at exact per-epoch visit and triplet totals (3,502 visits and 93,781 triplets per epoch). Triplet count is also the only bag property with a measured training effect: 20% fewer slices cost −0.0041 and 35% fewer cost −0.0051. Its signal is a detached per-study scalar taken from training batches only. Validation is untouched, and the proposal commits to a validation selection hash equal to the incumbent's.

I rank it third. It is ahead of model-feedback-depth-targets because the stopped campaign found per-triplet attention feedback near-uniform and falsified (triplet-sampling proposals-20260919T190312Z-3039dad1 item 4a). It stays behind the reserved head and plane-coupled-depth-targets because it needs more plumbing: a shared-memory allocation table, a batch-level memory guard and checkpointed EMA state. Its sign is also uncertain.

I'm not admitting epoch-paced-bag-size or label-diverse-batch-composition this event. Neither has new evidence, and epoch-paced-bag-size was inconclusive in the stopped campaign.

The proposal leaves several choices open: the misfit signal, the EMA factor, thirds versus proportional allocation, donor cuts, capacity exclusions and the memory-guard value. The implementation must fix each one as a constant before launch. The review will require a single signal, with no variants behind options.

Unresolved concerns:
- Label-derived misfit makes bag density correlate with rare positives in training. Softmax attention pooling (model.py:243-247) mostly hides count, but more slices still raise the chance that a positive bag contains its focal evidence. That is partly the intended effect, but it also means training sees positives at a density validation never does. Diagnostics should report recipient positive rates per label.
- The memory guard is the only barrier to OOM or a runtime-ceiling breach. Its cap must come from the replay's largest per-batch total. Batch swaps must preserve the once-per-epoch visit rule and keep the ordering stochastic.
- Donor cuts that hit near-capacity multi-source slots, or retries that change capacity, could produce assignment shortfall and break the zero-tolerance triplet total. The CPU replay with a synthetic signal must prove the exact 937,810 triplets.
- RHO-LOSS-style risk: high entropy-corrected loss can mark label-noisy studies, so the densest bags could go to mislabelled cases. The sign is uncertain, and the gain may fall below the ~0.003 same-seed sampling noise.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/researcher-20260921T224526Z-92a4b780/report.json (proposal misfit-steered-study-bag-density)
- outputs/autoresearch/sampling/baseline/source/src/knee_abnormality/modeling/model.py:243-247 (masked softmax attention pooling)
- outputs/training/k_05/fold_0/timing.json:9 peak_cuda_allocated_bytes 20999941632
- docs/validation/run-comparison-and-diagnostics.md#overnight-launcher (fewer-slice training cost −0.0041/−0.0051)
- outputs/autoresearch/triplet-sampling/jobs/researcher-20260919T183826Z-62f4f17b/report.json item 8 (prior between-study reallocation objection)

## completion-20260921T231157Z-8a801ac0

Anatomy-anchored crop placement (joint-line SI anchor for sagittal and coronal series, tissue-extent midpoint for axial, clipped to ±(FOV−140)/2): macro AUC 0.89809 vs 0.89904, a change of −0.00095. The 80% bootstrap interval is [−0.0034, +0.0017]. The decision is reject. The run was valid: exact counts and runtime within the ceiling.

Exploratory per-label tradeoff: regions the downward shift pushed out of the window lost AUC. PF OA fell 0.011 and Contusion 0.017, and both 95% intervals exclude zero. Baker's fell 0.007 and Medial OA 0.004. Joint-line labels gained, but their intervals include zero: Lateral Meniscus +0.009, Synovitis +0.008, MCL +0.006, ACL +0.003.

Interpretation: with the 140 mm window fixed, the default centre already balances patellofemoral or suprapatellar content against tibiofemoral content. Recentring on the joint line only redistributes coverage between these regions. Any future crop-placement idea would need to preserve patellofemoral and superior coverage rather than recentre on one landmark. The estimator's unclamped outliers (coronal −34 mm, sagittal +37 mm before clipping) and its training-only anchor jitter remain unmeasured confounds. They are small relative to the per-label pattern.

The fixed eligible decision is reject, and I interpret this run as a valid negative result. Accounting is complete and matches the sealed counts exactly: 937,810 triplets and 35,020 visits, with 93,781 triplets and 3,502 visits in every epoch. Validation used the x1 count of 24,253 triplets with the same selection hash in every epoch. Measured compute was 1,591 s, under the 1,847 s ceiling, with no config differences. The run did not break the evaluation or the contract.

The review concerns that sent this to the orchestrator do not change that. The unclamped joint-line outliers are bounded by crop clipping. Anchor mismatch after a retry is rare. The training-only anchor jitter is a mechanism property, not leakage. The broken adhoc profiler script and the unverified IOP sign on the anterior branch are outside the measured path. None of these makes the comparison invalid; they are small confounds that could only have moved the result a little in either direction.

The result: macro AUC fell by 0.00095 (0.89804 vs 0.89904). The bootstrap mean is −0.00087, the 80% interval is [−0.0034, +0.0017] and the 95% interval is [−0.0046, +0.0031]. This sits inside the ~0.003 same-seed sampling noise, and there is no sign of a macro gain.

The per-label pattern (exploratory and unadjusted) fits the anatomy. The recentring moved sagittal and coronal windows about 10 mm down toward the joint line, which removes superior content: the patella, suprapatellar recess and distal femoral shaft. The labels that depend on that region fell:
- PF OA: −0.0111, 95% interval [−0.0220, −0.0009]
- Contusion: −0.0174, [−0.0363, −0.0004]
- Baker's: −0.0074
- Medial OA: −0.0043

Joint-line labels rose, but none of their intervals excludes zero:
- Lateral Meniscus: +0.0090
- Synovitis: +0.0076
- MCL: +0.0056
- ACL: +0.0027

A single run does not establish this tradeoff. Its direction is consistent with the mechanism: at a fixed 140 mm window, recentring trades superior and patellofemoral coverage for tibiofemoral coverage rather than adding information.

The queue is unchanged, since this idea was not in it.

Unresolved concerns:
- Single-seed fold-0 run: the per-label intervals are unadjusted and do not include training-noise uncertainty. Seed-matched sampling changes alone move macro AUC by up to 0.0031, so the patellofemoral and contusion loss is suggestive, not established.
- The implementation's residual issues were not fixed before the run: no plausibility clamp on joint-line outliers, anchor mismatch after preparation retries, and anchor jitter in training that validation lacks. They could have slightly biased the result either way, and their effect was never measured.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/anatomy-anchored-crop-placement-20260921T221837Z-3b5416fc/comparison/comparison.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/anatomy-anchored-crop-placement-20260921T221837Z-3b5416fc/comparison/per_label.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/anatomy-anchored-crop-placement-20260921T221837Z-3b5416fc/changes.diff
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/orchestrator-20260921T223613Z-a491b298/report.json (implementation review and the 30-study anchor probe)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/developer-20260921T221837Z-c52338e5/report.json

## review-20260921T231519Z-5fec5880

The coupling happens in percentile space, not in physical depth. target_depths maps each slot's percentile through a truncated Gaussian built from that slot's own pool, with sigma = 0.867 * limiting_reach(pool). The two contrasts land at the same physical depth only when their pool extents and centres coincide, which is typical for same-session series but not guaranteed. When a plane's two slots differ in count (study_budgets redistribution or a geometry_groups capacity limit), the shared stream still gives the same u prefix. But dividing by different counts makes the percentiles diverge, so the coupling is only partial. The opposite direction, cross-slot band interleaving (anti-correlated bands), was inconclusive in the stopped triplet-sampling campaign under the old labels and split. This run tests the correlated end. The expected effect is small relative to the ~0.003 fold-0 noise from same-seed sampling.

I approve because the diff implements the assigned mechanism as declared and nothing else changes. Each plane has exactly two slots in configs/training.yaml:16-39 (non-fluid and fluid/fat-sat), and they have equal max_images. The two slots of a plane now share the "targets" seed, so when their counts are equal they draw identical per-band jitter u, giving identical stratified percentiles. That reproduces the cross-slot percentile pairing that validation's midpoint percentiles already have. Each slot's marginal target distribution is unchanged: each slot's u is still uniform on [0,1). Stratification, budgets, supplier randomness and band_order stay independent per slot, so only the joint cross-slot correlation changes. Coverage across the run is unchanged per slot. Exploitation shifts toward depth-matched cross-contrast views, and cross-slot depth-pair diversity goes down. The change uses no labels and no validation information, and it adds no preparation cost (0.32 s). Queue order and membership are unchanged.

Unresolved concerns:
- The realized accounting replay is unavailable (accounting_available=false). Visit counts are fixed by construction. Realized triplets are not: linear_sum_assignment can leave a row unfilled (a padding column), and whether that happens depends on the drawn targets. So the 937,810 total at zero tolerance must still be verified by the gate, not assumed.
- Physical depth-matching is only approximate when a plane's two pools differ in extent or count. The coupling strength actually achieved (for example, the paired target-depth difference per plane) is not recorded, so a null result cannot separate 'coupling doesn't help' from 'coupling was weak'.
- Coupling reduces the diversity of cross-slot depth pairs the model sees in training. That could hurt robustness to the misaligned series pairs that occur at test time, and no diagnostic measures it.

Evidence:
- outputs/autoresearch/sampling/experiments/plane-coupled-depth-targets-20260921T231233Z-c682d9f5/changes.diff (full diff: sampling.py:28, 471-474)
- candidate source src/knee_abnormality/data/preprocessing/sampling.py:469-492 (validation path skips rng for targets and band_order via `if training`)
- candidate source src/knee_abnormality/data/preprocessing/sampling.py:65-68 (operation_seed hashes the JSON of seed, study, kind, operation, epoch)
- candidate source src/knee_abnormality/data/dataset.py:169-206 (study budgets per slot; kind tuple passed to select_triplets)
- candidate source configs/training.yaml:16-39 (two contrast slots per plane, equal max_images)
- outputs/autoresearch/sampling/jobs/developer-20260921T231233Z-c130888e/report.json

## proposals-20260921T232604Z-694e0a2f

Checked the baseline source for feasibility. `_crop_geometry` (spatial.py:18-27) centres the crop and takes no offset. `output_to_native` is built in `resample_crop` (spatial.py:99), and preparation.py:192 builds patient_maps from `cache[r.path].output_to_native`. So the cache is keyed on path alone and must become (path, window), as the proposal states. Otherwise a slice shared between an A triplet and a B triplet would silently reuse the wrong crop and the wrong patient map. All three slices of a triplet must share one window. position_2d (model.py:351) stays window-relative.

The researcher also argues the anchor run's regional per-label pattern was a reference artifact: its per-label deltas correlate at r=0.72 with the unrelated span run and at r=0.59 with k1−k05. That supports reading it as k_05's own per-label deviation, not a placement cost. These are exploratory per-label numbers and carry no weight in adoption.

Admitted complementary-window-bags. It has no dependencies. It works on "where the in-plane window sits", which the contract puts in scope, and it leaves the sealed 224 px / 0.625 mm / 140 mm per-triplet geometry alone. Placement is set by measured tissue extent from images only, with an image-centre fallback. The same logic runs in validation and inference through the shared dataset path. The only randomness is the training band phase, drawn from a new stream, so the existing targets, suppliers and band_order streams stay as they are. Counts do not change.

Of the in-plane lever, this is the version the registry has not tested: coverage of the whole bag rather than moving a single window. The single-window anchor came out at −0.00095. That result limits the expected size of a placement effect, but it does not predict this idea's sign.

I ranked it third, after misfit-steered density and ahead of model-feedback-depth-targets. The researcher measured model-feedback's per-triplet steering signal as near-uniform, so it has less reach. Complementary windows need more preprocessing plumbing than misfit steering, and their expected gain is modest.

I did not admit the older epoch-paced-bag-size and label-diverse-batch-composition entries. This event adds no evidence for them, and the researcher's gradient-clip measurement weakens the case for batch composition. They stay proposed.

Unresolved concerns:
- complementary-window-bags: the cache in preparation.py:192 is keyed by path. If the window is not added to the key, a slice shared across windows gets the wrong pixels and a wrong patient_map with no error. Review must confirm the key change and that a triplet's three slices share one window.
- complementary-window-bags: the claim that per-window re-resamples are 'rare' is unmeasured. Adjacent selected triplets in one series may share slices often. Preparation time should be measured against the 1,847 s ceiling, plus the tissue-mask cost if the reference slice is the middle-index slice.
- complementary-window-bags: the label-free CPU coverage probe (distribution of s, share of two-window series, union tissue coverage) should be recorded before launch. Otherwise a null result cannot tell 'coverage doesn't help' apart from 'coverage barely changed'.
- complementary-window-bags: the minimum shift s that triggers two windows and the reference-slice choice must be fixed in advance, never swept. The validation preprocessing identity changes and must be recorded.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/researcher-20260921T231233Z-92b631e0/report.json
- outputs/autoresearch/sampling/baseline/source/src/knee_abnormality/data/preprocessing/spatial.py:18-27 (centred _crop_geometry, no offset)
- outputs/autoresearch/sampling/baseline/source/src/knee_abnormality/data/preprocessing/spatial.py:99-108 (output_to_native construction)
- outputs/autoresearch/sampling/baseline/source/src/knee_abnormality/data/preprocessing/preparation.py:192-235 (path-keyed cache feeding patient_maps)
- outputs/autoresearch/sampling/experiments/anatomy-anchored-crop-placement-20260921T221837Z-3b5416fc (single moved window, −0.00095)
- docs/research/geometry-and-views/input-resolution.md#field-of-view-and-joint-position

## completion-20260921T233858Z-d932a52a

Stratifying the training depth draw across epochs (10 sub-strata per band, one per epoch, permuted per study, type and band) is valid at exact counts but did not beat the reference on fold 0. Macro AUC was +0.00093 (80% CI [-0.0012, +0.0032]; 95% CI [-0.0024, +0.0043]). This is the second reject of the mechanism: once under the old labels and split, now under combined_v1 and the scanner split. Per-label results (exploratory, unadjusted) point in mixed directions: ACL +0.0102 (95% CI just above 0), Baker's +0.0082, Lateral OA +0.0044, Effusion +0.0033, Medial OA +0.0030; MCL -0.0092, Contusion -0.0069, Medial Meniscus -0.0046. Gains and losses offset at macro level, which fits a reshuffle from sampling noise rather than a systematic coverage benefit. Replay confirmed that the accounting is unchanged (per-epoch triplets, visits and cohorts identical) and that validation selection is constant across epochs. Runtime was 1,585.8 s, about the same as the reference, so the mechanism costs nothing, but there is no evidence it helps. Cross-epoch depth coverage alone is unlikely to deliver ≥0.001 here. Within-epoch stratification already covers each band, and the extra coverage across epochs changes inputs by less than the model's sampling noise.

This completion was escalated because the implementation review left concerns open. The realized evidence resolves the most important one. The replay is complete: 937,810 training triplets and 35,020 visits, exactly 93,781 triplets and 3,502 visits in every epoch, with training and validation cohort hashes matching the sealed limits. Validation used 24,253 triplets and 905 visits with the same selection_sha256 (f7ad4380…) in all 10 epochs, and the diff leaves the deterministic midpoint percentile branch untouched, so validation does not depend on the training seed. Measured compute was 1,585.8 s, under the 1,847 s ceiling, with 4,380 optimizer steps. Preparation added 0.17 s. I inspected the diff again: it changes only sampling.py (stratified jitter across epochs within each training draw, the identity string and a manifest field). No labels, folds or model code changed. The run is valid.

Interpretation of the fixed reject: macro AUC went from 0.899042 to 0.899976 (Δ +0.00093). That is below the 0.001 minimum gain, and the 80% paired-bootstrap interval [-0.00119, +0.00319] includes 0. The result sits inside the ~0.003 noise that a same-seed sampling change produces, so it cannot tell a small real effect from noise. It agrees with the earlier reject of this mechanism in the stopped triplet-sampling campaign under the old labels and split. I added no queue changes. The admissible proposals (epoch-paced-bag-size, label-diverse-batch-composition) stay proposed for a later ranking event.

Unresolved concerns:
- It is still unmeasured whether a preparation retry that changes a study's pool or count in one epoch breaks the exact row-to-stratum coverage. This does not matter for a rejected result, but applies if the mechanism is reused.
- This is a single fold-0 run with a same-seed sampling divergence of ~0.003, so a true effect of roughly ±0.002 cannot be excluded. Per the policy, no confirmation run is scheduled.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/run-level-stratified-jitter-20260921T224831Z-d1fcb8e3/comparison/comparison.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/run-level-stratified-jitter-20260921T224831Z-d1fcb8e3/comparison/per_label.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/run-level-stratified-jitter-20260921T224831Z-d1fcb8e3/changes.diff (sampling.py select_triplets: the training-only strata block; the validation midpoint branch is unchanged)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/run-level-stratified-jitter-20260921T224831Z-d1fcb8e3/manifest.json (accounting: 93,781 triplets and 3,502 visits in each of 10 epochs; validation selection f7ad4380… constant)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/orchestrator-20260921T225138Z-a3be6dd7/report.json (approval review)
- triplet-sampling decision run-level-stratified-jitter-20260919T183826Z-f211c7ce: reject

## review-20260921T235100Z-3461e649

This is the campaign's first training-feedback candidate. From epoch 1, the top third of training studies by misfit EMA gains up to 6 triplets, spread across slots in proportion to base budgets and capped at capacity-1. The bottom third pays exactly the same total, with each slot floored at 1. That is about ±22% of a ~27-triplet bag. The misfit signal is class-balance-weighted BCE minus label entropy, averaged over observed labels. Each study is seen once per epoch, so an EMA decay of 0.9 acts per epoch, and the first observation initialises it. At epoch 5 the epoch-0 value still carries 0.59 of the weight. The steering therefore reflects early-training misfit more than current misfit. Because positive labels carry the larger class-balance weight, recipients will likely be concentrated among positive and label-hard studies. That makes this partly an implicit positive/hard-example density shift at fixed compute. Studies that stay persistently misfit may be label noise, which this mechanism would amplify. With ~0.003 same-seed sampling noise on fold 0, a single-run difference below that is weak evidence either way.

Approve for launch. I inspected the diff directly, beyond the author's claim. Visits are exactly once per epoch: the batch sampler emits a permutation, the step count is unchanged at ceil(3502/8)=438, and the ledger rejects duplicate visits. Triplet totals are conserved by construction at the requested-budget level. Realized-level conservation rests on the empirical fact that the baseline never pads rows when budget<capacity, which the per-epoch constancy of 93,781 in the baseline ledger supports. If a raised recipient ever pads, the run fails the ledger gate as invalid; it cannot pass silently. Peak batch memory is bounded by the baseline's worst-case top-8 base totals. Swap-pass cost is negligible (at most 438 batches). Validation is untouched and independent of the training seed. There is no leakage: feedback uses training labels and logits only. Queue unchanged.

Unresolved concerns:
- Realized-count conservation assumes that select_triplets never leaves a row unfilled when count<capacity. The baseline's constant 93,781 per epoch supports this, but no proof exists for the raised (capacity-1) regime. The ledger gate is the only guarantee, and a violation would make the run invalid.
- The misfit EMA is dominated by stale early-epoch values (0.9 decay per visit, initialised to the first observation), so this is not really 'current' misfit. No diagnostic logs the recipient/donor identity or how it overlaps with positive labels. A null result therefore cannot separate 'density doesn't help' from 'steered toward the wrong studies'.
- Epoch-0 shuffle now uses the global torch RNG (randperm) instead of the loader's generator. It is still seeded via seed_everything, so the run is reproducible, but the epoch-0 order differs from the incumbent, which contradicts the integration claim that epoch 0 is unchanged. This is minor and consistent with the expected sampling divergence.
- Persistently misfit studies may carry noisy labels, and denser bags would amplify their gradient share.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/misfit-steered-study-bag-density-20260921T233929Z-2f854803/changes.diff
- candidate source data/preprocessing/sampling.py:433-500 (bag_density_offsets: raise_room=capacity-1-base, lower_room=base-1, zero-sum donor/recipient allocation)
- candidate source data/preprocessing/sampling.py:559-569 (count=min(budget, len(groups)); capacity=len(groups))
- candidate source data/dataset.py:105-113, 326-328, 457-475, 497-507 (budget offsets, epoch-0 tables, batch sampler, loader branch)
- candidate source training/lightning.py:359-405 (epoch-start recompute; misfit EMA from training logits, same weight convention as loss at lightning.py:121-123)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/baseline/triplet_accounting.json (93,781 triplets × 10 epochs, 3,502 visits each)
- /home/claudio/src/knee-abnormality/src/knee_abnormality/autoresearch/accounting.py:28-128 (ledger enforces exact realized totals, cohort, validation determinism)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/developer-20260921T233929Z-9a6f1458/report.json

## proposals-20260922T000413Z-711ec936

The researcher's cache-only probe exists and matches its claims (probe/SUMMARY.md). ±2-slice channel reach is 0.700 against 0.491 for ±1, and 0.803 against 0.582 within ±30 mm. The median within-series gap is 3.72 mm. Draw variance scales as n^−1.64…−1.75. The best between-study reallocation ranking removes about 5% of draw variance, misfit ranking removes about 0%, and random ranking adds about 5%.

Limits: every probe number comes from the old all_5fold_v3 split, the old labels and frozen ranked-filtered checkpoints with the sigma-40 snapshot sampler, on the validation cohort. None of it is a training result under combined_v1 or the scanner split. The +0.007 from averaging 12 draws is test-time ensembling of fixed middles, not evidence that half-contrast context channels help in training. Reach counts context slices at half weight, so the effective information gain is less than 49→70%.

The implication for the prepared misfit candidate: any gain has to come from emphasising hard or positive studies, not from reducing variance. A null result is therefore the expected outcome, not a surprise.

I admitted slab-context-triplet-channels. It has no dependencies (proposal_admission satisfied) and falls inside the contract's "how each triplet's channels are composed" scope. It leaves selection, counts, validation x1, the 224 px input, 0.625 mm spacing, 140 mm FOV and augmentation untouched. What changes is the preprocessing and pixel identity, in the same way for training, validation and inference. That is allowed if it is recorded under preprocessing_identity. It is also the only queued mechanism aimed at per-visit through-plane reach. That lever has the strongest measured evidence: 20% fewer slices cost −0.0041. Selection stays identical and the validation selection hash is unchanged, so the result is easy to attribute.

I ranked it second, right behind the reserved head (misfit, already prepared). It goes ahead of complementary-window-bags because it needs less plumbing and reaches every series. The researcher's new bound says between-study reallocation cannot gain through draw variance; the misfit ranking comes out neutral at 0.995–1.002. That lowers the prior for misfit density and suggests the same for model-feedback-depth-targets, whose steering signal is also noisy. So model-feedback-depth-targets stays last.

epoch-paced-bag-size and label-diverse-batch-composition stay proposed and unqueued. For epoch-paced, the new probe says ±1 per slot raises draw variance by about 10%, and the earlier campaign was inconclusive. Neither adds new evidence that would justify queueing it over the four ideas above.

Unresolved concerns:
- Slab averaging over index ±2 blurs outer channels unevenly across series. In thick 2D series, context reaches about ±7–8 mm, while thin 3D series barely change. The rejected physical-span test (−0.00047) also changed outer-channel content, so the sign is genuinely uncertain.
- Validation and inference now decode up to 5 slices per triplet. Validation waves may be loader-bound, so the full replay must be measured against the 1,847 s ceiling. Training being GPU-bound does not cover validation.
- The implementation must normalise context slices with the unchanged bounds, average before uint8 quantisation, and fall back exactly to s±1 at run edges or on decode failures. A disabled-context degeneration check equal to the baseline tensors must be shown at review, together with per-plane fallback and both-side-context rates.
- The probe evidence comes from the old split, old labels and frozen checkpoints. It motivates the idea but does not predict a fold-0 gain above the ~0.003 same-seed sampling noise.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/researcher-20260921T233929Z-cdc19f71/report.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/researcher-20260921T233929Z-cdc19f71/scratch/probe/SUMMARY.md (reach table; allocation_global bound; headroom; cohort all_5fold_v3, frozen ranked-filtered seeds 42–44)
- docs/validation/results/sampling/2026-09-17-ranked-filtered-result.md (same-seed sampling noise up to 0.0031)
- docs/validation/results/campaigns-and-queues/2026-09-16-restored-code-hyperparameter-runs.md (fewer slices −0.0041/−0.0051)

## completion-20260922T000556Z-fa36b368

Drawing a plane's two contrast series from a shared depth-target stream (correlated percentiles, matching validation's cross-slot pairing) gave no macro gain on fold 0: -0.0004 [80% interval -0.0028, +0.0020]. Together with the inconclusive anti-correlated band interleaving in the stopped triplet-sampling campaign, both ends of cross-slot depth correlation now look null at this noise level. Whether depth pairing across contrasts matters is not resolved, because the coupling strength actually achieved was not measured.

Per-label results are exploratory and unadjusted. There were gains for ACL (+0.0095) and Lateral Meniscus (+0.0100), and losses for PF OA (-0.0139; its 95% interval [-0.0261, -0.0027] is the only one that excludes 0), Fracture (-0.0085) and Baker's (-0.0068). One loose hypothesis is that coupling helps ligament and meniscus reading, which benefits from co-registered contrasts, and hurts patellofemoral and posterior findings, which benefit from diverse depths. Given multiplicity and the size of the sampling noise, this should not motivate a per-label variant on its own.

Runtime cost is effectively zero: 1,583 s of measured compute and 0.3 s of preparation.

The run is valid. I checked the executed diff directly (sampling.py:28, 472-474): the only change is the plane-only `stream_key` for the "targets" RNG operation, plus the identity string. It matches the declared add-mode integration, and it removes nothing. The earlier escalation said realized accounting was unavailable. That no longer applies: the recorded ledger is complete. Training has exactly 93,781 triplets and 3,502 visits in each epoch, for 937,810 and 35,020 in total, matching the sealed counts at zero tolerance. The validation selection hash is the same in every epoch (f7ad4380…), and validation keeps its 24,253 triplets over 905 studies. Measured compute was 1,583 s, under the 1,847 s ceiling, with 4,380 steps. The resolved config does not differ from the reference.

The fixed decision is reject. Macro AUC was 0.898659 against 0.899042, a change of -0.00038. The bootstrap mean change was -0.00039, with an 80% interval of [-0.00283, +0.00197]. That is well inside the ~0.003 noise from same-seed sampling perturbations. No queue change is warranted: the result says nothing new about the priority of the remaining queued mechanisms. None of them builds on coupled depth targets.

Unresolved concerns:
- The coupling strength actually achieved (the paired physical target-depth difference per plane) was never logged. So the null result cannot tell 'coupling is ineffective' apart from 'coupling was weak' when the two pools differ in extent or count.
- PF OA's -0.0139, with a 95% interval excluding 0, is one of 12 unadjusted label intervals. It comes from a single run under ~0.003 macro same-seed sampling noise, so treat it as exploratory only.

Evidence:
- Diff: outputs/autoresearch/sampling/experiments/plane-coupled-depth-targets-20260921T231233Z-c682d9f5/source/src/knee_abnormality/data/preprocessing/sampling.py:28,472-474 vs baseline/source
- Comparison: outputs/autoresearch/sampling/experiments/plane-coupled-depth-targets-20260921T231233Z-c682d9f5/comparison/comparison.json (macro 0.898659 vs 0.899042; 80% [-0.00283, 0.00197])
- Per-label: .../plane-coupled-depth-targets-20260921T231233Z-c682d9f5/comparison/per_label.json (PF OA -0.0139 [-0.0261, -0.0027]; ACL +0.0095; LM +0.0100)
- Accounting in the evidence packet outputs/autoresearch/sampling/jobs/orchestrator-20260922T000556Z-0db71c11/evidence.json: 937,810 triplets and 35,020 visits, complete, zero tolerance met
- Prior review: outputs/autoresearch/sampling/jobs/orchestrator-20260921T231519Z-b109e0d4/report.json

## review-20260922T001642Z-c2ad878d

The candidate replaces each outer channel of a triplet with the float mean of the adjacent slice (middle±1) and the next slice out (middle±2), applied before quantisation. This is a thicker effective slab, not a new selection. The model's input is the same shape at the same GPU cost. The main cost is on the CPU: up to two extra decodes per triplet, run as a second decoder call after the triplet decode. That is roughly 40% more unique slices decoded (the 49%→70% slice-coverage claim), and it also applies to validation. The composition applies in both training and validation, so a result measures train-and-evaluate on slab-context inputs, not a pure training-sampling effect. Near the ends of a run, a channel falls back to the adjacent slice alone; `context_requested` and `context_applied` report how often. Relative to the shared ~0.003 same-seed noise, a single-run difference of that size is weak evidence either way.

The review gate failed because the candidate edits `src/knee_abnormality/data/preprocessing/intensity.py`, which is not in `campaign_policy.editable_files`. A failed gate cannot be approved. The mechanism itself looks sound, so I am asking for one bounded repair instead of blocking it.

Repair: restore `intensity.py` byte-identical to the parent. Move the `_scale_pixels` and `_quantize_scaled` logic into `preparation.py` as private helpers. Keep the arithmetic identical: float32 clip, subtract, gain*255, `rint`, uint8, and zeroing outside the rect. Then non-context channels stay numerically equal to the parent's `_normalize_pixels`. In `preparation.py`, import only `_linear_quantiles` from intensity, plus `_normalize_pixels` if it is still used. Everything else in the diff can stay as it is.

Substantive review of the rest of the diff:
- Selection is unchanged. `Triplet.context` is `run[middle±2]` inside the same contiguous run, while the triplet is `run[middle-1:middle+2]` (`sampling.py:275-288`). Selection, budgets and counts are unaffected, so visits and triplet counts should match the ledger.
- Intensity bounds still come only from triplet slices. Context pixels are clipped to those bounds.
- Geometry, positions and `selected_paths` still come from the original triplet records.
- A context slice that fails to decode falls back to the adjacent slice alone.
- The blended rect is the intersection of the two slices' rects.
- No labels are used. The composition is deterministic, so validation stays independent of the training seed.
- Validation and inference get the same composition. The change is recorded in `SAMPLING_IDENTITY['composition']`, which the contract requires for a changed deterministic identity.

Unresolved concerns:
- Gate failure: `intensity.py` is outside the editable files. The helpers must move into `preparation.py`, and `intensity.py` must match the parent byte for byte.
- Runtime ceiling risk: extra context decodes (about 40% more unique slices) happen in a second, sequential decoder call inside `prepare_series`, for both training and validation. The ceiling is 1,847 s against a 1,605.7 s reference, so the headroom is only 15%.
- The blended outer channel's rect is the intersection of both slices' rects. Where the two differ, this can shrink the valid region compared with the parent. No diagnostic logs the lost area.
- The composition also changes the validation and inference inputs. Adoption would therefore mean an inference-time input change, not only a training-sampling change.

Evidence:
- Gate error: review-20260922T001642Z-c2ad878d reports 'Changes outside the campaign workspace: [src/knee_abnormality/data/preprocessing/intensity.py]'.
- experiments/slab-context-triplet-channels-20260922T000626Z-2a4a7330/workspace/src/knee_abnormality/data/preprocessing/intensity.py:33-65 splits `_normalize_pixels` into `_scale_pixels`/`_quantize_scaled` (numerically equivalent refactor).
- workspace/.../preprocessing/sampling.py:275-288 builds `context` as `run[middle-2]`/`run[middle+2]` next to the triplet `run[middle-1:middle+2]`; selection code is unchanged.
- workspace/.../preprocessing/preparation.py:198-264 decodes the extra context slices, clips them to the triplet-derived bounds, and blends outer channels 0 and 2 on the intersection rect.
- workspace/.../data/dataset.py:208-241 passes contexts to each uid and aggregates `context_requested`/`context_applied` into slot metadata.
- Developer report: outputs/autoresearch/sampling/jobs/developer-20260922T000627Z-5138b3d1/report.json

## review-20260922T001934Z-b214d50d

The approved candidate replaces each outer channel with the float mean of the slices at middle±1 and middle±2, before quantisation, and clips the context slice to the triplet's own intensity bounds. It applies identically in training, validation and inference, so any result measures a change in input representation, not training sampling alone. The extra cost is CPU decoding only: up to two additional unique slices per triplet, decoded in a second, sequential decoder call inside `prepare_series`. It hits validation as well. GPU cost is unchanged. The per-visit diagnostic `context_applied/context_requested` shows how much blending actually happens; triplets at the ends of a run fall back to the parent composition. A single-run difference of about 0.003 or less is within the recorded same-seed sampling noise.

The earlier gate failure is repaired exactly as requested. `intensity.py` has been restored, and the helpers moved into `preparation.py` with the same arithmetic. The rest of the diff matches what the earlier review found sound. Validation independence, leakage, selection accounting and preprocessing all hold (details in `integration_review`). This is a channel-composition change, which the contract explicitly places in scope. It adds no configuration and keeps no legacy branch. Visit and triplet ledger counts should match the baseline because selection is untouched. The supervisor's ledger and runtime gates remain the binding checks. The queue does not change.

Unresolved concerns:
- Runtime risk. The extra context decodes run after the triplet decode, serially, in both training and validation. Headroom against the 1,847 s ceiling is only 15%, and no timing measurement was supplied (`accounting_available` is false).
- `decoded_unique_paths` in the prepare metadata now counts context slices as well, so it is not comparable with the parent's value.
- The intersection rect can shrink an outer channel's valid region, and if the two rects are disjoint the channel becomes all zeros. No diagnostic logs the area lost. This is unlikely within a single contiguous run, but it is unmeasured.
- If adopted, this changes the inference-time input composition, not only training. Any separate inference path outside `dataset.py`/`prepare_series` would need the same composition.

Evidence:
- experiments/slab-context-triplet-channels-20260922T001723Z-f7832c5b/changes.diff
- source/src/knee_abnormality/data/preprocessing/preparation.py:21-45 (the helpers) and :220-285 (the blending)
- source/src/knee_abnormality/data/preprocessing/sampling.py:277-291 (the Triplet context) and :569-574 (the diagnostics)
- `intensity.py` is byte-identical to `baseline/source` (checked with `diff`)
- jobs/orchestrator-20260922T001642Z-264c3b88/report.json (the earlier repair request)
- jobs/developer-20260922T001723Z-a65cc68b/report.json (the bit-identical arithmetic check)

## proposals-20260922T002228Z-671f797c

The researcher's probe on k_05's per-epoch validation predictions is diagnostic, not a training result. It found that per-study loss residuals persist across epochs: R^2 is 0.82 from epoch 0 to 1 and 0.95-1.00 from epoch 5 on.

With stale residuals, greedy batch packing cuts batch-mean residual error to 0.24-0.53 of a random shuffle. With per-visit noise it cuts error less:
- 0.51-0.56 at an 80% clean share
- 0.72-0.78 at 60%

Other orderings do worse. Static label-balanced batches reach 0.51-0.69. GraB prefix herding stays at 0.79-0.82 even with an oracle.

I verified in source that training selection is order-independent:
- The RNG is keyed per study, type and epoch (`sampling.py:471-474`).
- The accounting digests sort per-study rows (`accounting.py`).

A batch-composition-only candidate is therefore auditable. Its training selection hashes must equal the baseline replay's exactly. Only worker-side augmentation and series-dropout streams, and batch membership, may differ.

Two things remain unmeasured:
- How much noise augmentation and series dropout add to training-visit residuals.
- How much of the full-parameter gradient variance comes from the logit-space component.

I admitted residual-balanced-study-batches. It has no dependencies, and proposal_admission is satisfied. I placed it third: after the reserved head and complementary-window-bags, and ahead of model-feedback-depth-targets.

Why admit it:
- It is the only in-scope lever that leaves every visit's inputs unchanged. Only batch membership and order change, so attribution is cleaner than for any idea already in the registry.
- It costs almost nothing: CPU greedy packing, under a second per epoch.
- It is a feedback mechanism, which the campaign lacks apart from the running misfit-density candidate.

I checked the two claims it rests on:
- Selection RNG is keyed per (seed, study, type, operation, epoch), not by loader order (`sampling.py:471-474`). Per-study bags therefore stay byte-identical under reordering.
- The accounting digests sort per-study rows (`accounting.py:28-78`). Training selection_sha256 and triplet/visit counts must therefore equal the baseline replay's. That gives a hard pre-launch check.

Why third and not higher:
- How much a logit-space batch-variance reduction translates into macro AUC is unmeasured.
- The expected effect is plausibly below the ~0.003 same-seed sampling noise.
- The complementary-window and slab ideas change what the model sees and carry larger prior-backed upside.

It goes ahead of model-feedback-depth-targets because it cannot create a label shortcut and it is cheaper to execute.

I left the other two proposals unadmitted:
- label-diverse-batch-composition: the researcher's probe shows it strictly weaker than residual balancing (0.51-0.69 against 0.24-0.53 of random batch error) and fading late in training. It stays proposed. It is not queued, so parking does not apply.
- epoch-paced-bag-size: no new evidence this event.

Unresolved concerns:
- There is no measured path from lower logit-space batch residual variance to AUC. The probe used deterministic validation predictions, not noisy training visits, and a null result below ~0.003 will be uninformative.
- It overlaps with the running misfit-steered-study-bag-density: both replace the training batch sampler. If misfit density is adopted first, this idea must be rebuilt on its planned totals and cap. It must not run a second sampler alongside it.
- Pre-launch gate: the CPU replay must show 438 batches (437x8 + 1x6), one visit per study, and training selection hashes equal to the baseline's. The candidate must also save and restore both the residual table and the bag-total table in checkpoints.
- The lower Adam second-moment estimates that balancing causes cannot be separated from the variance-reduction effect, so a positive result would not isolate the mechanism.
- label-diverse-batch-composition is effectively superseded but stays proposed. Retiring it is a registry action outside this event's park semantics.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/researcher-20260922T000627Z-cfe5d8f5/report.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/researcher-20260922T000627Z-cfe5d8f5/scratch/probe/SUMMARY.md
- baseline/source/src/knee_abnormality/data/preprocessing/sampling.py:471-474 (per-study/type/operation/epoch RNG keying)
- baseline/source/src/knee_abnormality/autoresearch/accounting.py:28-78 (order-independent sorted selection digests)
- baseline/source/src/knee_abnormality/training/lightning.py:178-189 (uniform shuffle training loader)
- docs/validation/results/sampling/2026-09-17-ranked-filtered-result.md (same-seed sampling noise up to 0.0031)

## completion-20260922T003303Z-15bde4f3

Misfit-steered study bag density shifted about ±6 triplets per bag, from the top-third-misfit studies to the bottom third, from epoch 1 onward. Per-epoch and total counts stayed exact. Macro AUC did not change (Δ −0.0002; 80% interval [−0.0027, +0.0023]).

Per-label results (exploratory, unadjusted) point to a redistribution rather than a gain:
- Up: Synovitis +0.013 (95% interval excludes 0), Lateral Meniscus +0.008, ACL +0.005, PF OA +0.005.
- Down: MCL −0.012, Lateral OA −0.009, Medial OA −0.006, Medial Meniscus −0.003, Contusion −0.005.

Both OA compartments and MCL lost ground, which fits the idea that density moved toward studies whose misfit is driven by class-balance-weighted positives. This cannot be confirmed: recipient and donor identities were not logged, and are only held in the checkpoint's bag_density_state. The best epoch moved to the final epoch (9 vs 7), a weak hint that the reallocation slows convergence or delays the selected checkpoint.

The EMA was dominated by stale values: with a per-epoch decay of 0.9, the epoch-0 misfit still carries 0.59 of the weight at epoch 5. So the test is closer to "early-misfit studies get denser bags" than to current-misfit feedback. The result does not separate "density reallocation is inert" from "steered by a stale or noisy signal".

Any retry would first need fresher misfit (a faster decay or the last epoch only), a label-noise guard, and logs of recipient/donor overlap with positives. It would still face the ~0.003 noise floor.

This is a completion event, and the code-applied decision is reject. I checked the escalated review concerns against the run artifacts. None of them makes the run invalid:
- Count conservation held. The ledger is complete, with 93,781 triplets and 3,502 visits in every one of the 10 epochs, for totals of 937,810 and 35,020. This matches the sealed limits exactly, so the capacity-1 fill risk did not happen.
- Validation stayed independent. Its selection_sha256 (f7ad4380…) is the same in every epoch and matches the value in the baseline's triplet_accounting.json. Validation composition is therefore unchanged and does not depend on the training seed.
- Runtime was 1,593.8 s against the 1,847 s ceiling, with 4,380 steps and 0.17 s of preparation.
- The epoch-0 order comes from the global torch RNG `randperm` (changes.diff:145). This is reproducible because seed_everything seeds it, and it is part of the expected sampling divergence. The integration claim that epoch 0 is unchanged is inaccurate, but that does not affect validity.

The result is null. Macro AUC is 0.898828 vs 0.899042 (Δ −0.00021, bootstrap mean −0.00017). The 80% interval is [−0.0027, +0.0023], well inside the ~0.003 same-seed sampling noise. Checkpoint selection moved to epoch 9, the last epoch; the reference selected epoch 7.

No queue change. The two admissible proposals are not admitted in this event:
- epoch-paced-bag-size was already inconclusive in triplet-sampling.
- label-diverse-batch-composition overlaps with the queued residual-balanced-study-batches.

Unresolved concerns:
- No recipient/donor diagnostics were logged, so the per-label tradeoff (OA/MCL down, Synovitis/ACL/LM up) cannot be attributed to positive-heavy reallocation versus sampling noise.
- The integration statement said epoch 0 was unchanged, but the epoch-0 shuffle source changed (global randperm vs loader generator). Future feedback candidates should declare this accurately.
- This single-seed fold-0 null result is inside the recorded ~0.0031 same-seed sampling noise, so it cannot rule out a small effect in either direction.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/misfit-steered-study-bag-density-20260921T233929Z-2f854803/comparison/comparison.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/misfit-steered-study-bag-density-20260921T233929Z-2f854803/comparison/per_label.json
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/baseline/triplet_accounting.json (validation selection f7ad4380… matches candidate every epoch)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/misfit-steered-study-bag-density-20260921T233929Z-2f854803/changes.diff:145 (global randperm shuffle)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/misfit-steered-study-bag-density-20260921T233929Z-2f854803/training/fold_0/best-epoch=09-val_macro_auc=0.89883.ckpt
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/orchestrator-20260921T235100Z-120c94f0/report.json

## proposals-20260922T004615Z-dbe57880

- The header probe (128 studies, not the sealed cohort) supports prescription-level registration for fluid and T1/PD pairs in the same plane:
  - 83–92% of pairs are parallel.
  - 71–73% have at least 90% of slices within 0.5 mm.
  - 83% share the in-plane centre within 1 mm.
  - About 0.66–0.73 of T1/PD middles have both neighbour positions matched in the fluid series.
- In the baseline model, tokens interact only through the attention pool. Any cross-contrast conjunction at a pixel location therefore has to enter through the input channels.
- This makes the idea the only in-scope route to early fusion. Its result will be a change in input representation that applies in training, validation and inference, not a pure change to training sampling.

I admitted registered-fluid-context-t1pd-triplets. It has no dependencies (proposal_admission: satisfied). It falls within the contract's scope, which covers how each triplet's channels are composed. Selection, counts, visits and the fluid slots stay byte-identical, so the triplet and visit seals are untouched by construction. The model is unchanged.

The mechanism is new to the campaign: no earlier idea puts two contrasts into one token. It also addresses a gap recorded in triplet-sampling findings item 2: registered middles coincide across contrast slots, but late fusion cannot exploit them.

I checked the premise in the baseline source. In `model.py:220-264`, tokens meet only through the softmax-weighted pool. The optional residual readout adds a non-linearity after pooling, but not a conjunction at a given location. So cross-contrast features at a location can only come from the channels, and the premise holds.

It ranks third: ahead of model-feedback-depth-targets and behind residual-balanced-study-batches. The CPU cost is net-zero in decodes, attribution is clean because selection hashes don't change, and 25–30% of triplets are reached. The sign is uncertain, as the proposer says: it loses T1/PD neighbour context, which matters for OA labels. It shares a composition-level code path with the running slab-context candidate and the developing complementary-window-bags candidate. So whichever of those is adopted first changes its implementation base; the proposal already describes how to rebase onto both.

I did not admit epoch-paced-bag-size: the same mechanism was inconclusive in triplet-sampling. I did not admit label-diverse-batch-composition: it largely overlaps the queued residual-balanced batches, which use training feedback.

Unresolved concerns:
- Registration was measured on a 128-study header sample, not the sealed cohort. Inter-series patient motion cannot be seen in headers. The implementation should report, per plane, the conversion share and the fallback reasons (orientation, centre, position, decode) for both training and validation.
- In validation and inference, converted T1/PD tokens lose their own ±1 neighbours. OA labels, which lean on coronal and sagittal T1/PD, may regress. Review the per-label changes on medial and lateral OA, not just the macro result.
- The change moves the preprocessing and composition identity. It must record a new composition identity and prove degeneration: with matching disabled, tensors must equal the baseline's. The partner bounds must come only from partner slices in the same slot, at the same quantiles.
- It overlaps in code with slab-context (running) and complementary-window-bags (developing): outer-channel rules and a window-keyed cache. If either is adopted first, the implementation must rebase onto it and declare the integration explicitly.
- Fluid content that appears in both the fluid slot and the T1/PD outer channels weakens the effect of series dropout on fluid information. This is unmeasured.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/researcher-20260922T003339Z-44ea2746/report.json (proposal, probe figures, exhausted directions)
- outputs/autoresearch/sampling/baseline/source/src/knee_abnormality/modeling/model.py:220-264 (token interaction only via attention pool; residual readout after pooling)
- outputs/autoresearch/sampling/baseline/source/src/knee_abnormality/data/preprocessing/preparation.py:103-130 (prepare_series consecutive-slice check that the idea must restrict to own middles)
- docs/validation/results/pooling-and-attention/2026-09-17-type-contribution.md and 2026-09-17-test-time-allocation.md (fluid efficiency; T1/PD removal cost <=0.002)
- autoresearch/campaigns/triplet-sampling findings item 2 (cross-slot middle co-location; no pixel-level registration exploited)
- outputs/autoresearch/triplet-sampling/state.json: epoch-paced-bag-size-20260919T200307Z-0bdb46b8 inconclusive

## review-20260922T004641Z-0f1a2977

This candidate changes the in-plane crop in both training and validation. Wherever the measured tissue extent on an axis exceeds 140 mm, about half of each slot's triplets see the A-end window and half see the B-end window, instead of the centred crop. A result therefore measures a combined train-and-inference change in input framing, not training sampling alone. The windows are placed symmetrically about the image centre, as the idea specifies, not about the tissue centre. So they sit flush with the tissue ends only when the tissue is centred in the image. When the knee is off-centre, one window gets clipped and the other falls short of its tissue end. For sagittal and coronal series where the leg crosses both edges of the image, the SI extent equals the full image height, so the shift is (FOV−140)/2. The central strip of width 140−2s stays visible in every triplet. No diagnostic records how often a shift is non-zero or how large it is. A single-run difference of about 0.003 or less is within the recorded same-seed sampling noise.

Approving. The diff implements the stated mechanism as the idea specifies. Each window is shifted by s=(extent−140)/2 from the image centre and clipped. Window A is superior/anterior, derived from the IOP sign (the AP sign is flipped for LPS). The A/B labels alternate over the depth-sorted rows (the targets come from ascending percentiles). Validation always uses phase 0 and never calls rng('windows'), so validation stays deterministic and independent of the training seed. Its triplet count is unchanged, and the changed validation identity is recorded as SAMPLING_IDENTITY['windows']. The mechanism uses no labels or validation predictions. The landmark and mask computation runs inside the measured loader on already-decoded frames, with no extra DICOM reads. Its cost is one ~2 mm-grid Gaussian filter, one opening and one labelling per series per decoder call. On failure it falls back to the image centre. Everything is threaded through dataset.py, so training, validation and inference share one path. The candidate modules import cleanly with the supplied interpreter.

Unresolved concerns:
- No diagnostic records how often a shift is actually applied or how large it is, and measure_windows swallows every exception and falls back silently to zero offsets. If the mask fails widely, the run collapses to a near-baseline replay (apart from stream divergence), and that cannot be seen from the result.
- The reference frame is the middle unique record of the requests missing from the cache in each decoder call. On retries with a partial cache, triplets from the same series can end up with offsets measured on different slices. The effect is minor but not deterministic per series.
- Rows skipped because they have no assignment still consume A/B labels, so the realized A/B split within a slot can be uneven.
- Slices shared between adjacent triplets with different windows are resampled twice, and unique_slice_count/intensity pixels now count (path, window) pairs. Runtime headroom was not measured (accounting_available=false); the expected cost is small.
- The adhoc scripts that call decode_series(records, ...) with the old signature (scripts/adhoc/inspect_physical_preparation.py:123, the benchmark and profile scripts) would break if the candidate were adopted. They are outside the editable scope.

Evidence:
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/experiments/complementary-window-bags-20260922T003339Z-8e49982f/changes.diff
- source/src/knee_abnormality/data/preprocessing/spatial.py:122-154 (measure_windows; image-centre-relative shift s=(extent-window)/2; broad except fallback)
- source/src/knee_abnormality/data/preprocessing/spatial.py:171-203 (clipped shift; output_to_native carries source_top/left)
- source/src/knee_abnormality/data/preprocessing/sampling.py:479-483,493,541 (depth-sorted rows; training-only 'windows' stream, validation phase 0; label attached after assignment)
- source/src/knee_abnormality/data/preprocessing/preparation.py decode_series (single decode per path, reference = middle of missing unique records)
- /home/claudio/src/knee-abnormality/outputs/autoresearch/sampling/jobs/developer-20260922T003339Z-63d1d1eb/report.json (static-only verification; no timing)
