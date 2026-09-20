# Robust study centre (removed 2026-09-18)

A finished line of work, kept for the geometry it measured. The option
`data.robust_center_outlier_mm` was implemented on 2026-09-17, trained at 10, 20 and 30 mm, and
removed on 2026-09-18; the implemented behaviour and the reason for removal are in
[current preparation](../data-preparation/current-preparation.md#robust-study-centre-removed). The
rejection counts, rescue analysis and design below are kept because they describe the geometry,
not the option.

- *Problem:* 178 of 4,407 train centers are rejected (170 RMS above 10 mm, 8 with one series above
  20 mm), often because of one or two stray series, so the whole study falls back to local
  sampling. An in-plane residual does not affect depth sampling.
- *Proposal:* iteratively drop the worst series while it lies more than 20 mm from the center and
  refit, keeping the RMS, condition and two-plane checks. Each series then samples physically when
  the center projects inside its acquired range, and locally otherwise. The model no longer reads
  3D geometry, so only the previews' `geometry_valid` still depends on `& physical` in
  `src/knee_abnormality/data/dataset.py`.
- *CPU analysis* ([summary](../../outputs/analysis/study-center-robust-20260915/summary.txt)):
  it rescues 131 studies (21 of 31 on fold 0), dropping 159 series, mostly small-FOV axials. Of
  the 47 still rejected, 44 have no series above 20 mm. Centers of accepted studies do not move,
  by construction. Of 24,371 series, 350 would sample locally (75 on fold 0).
- *Gate on range rather than retention:* retained series from the other knee in bilateral
  studies, or from an ankle series filed in a knee slot, project up to 184 mm outside their range
  (see the [series-filtering census](../metadata/series-filtering-census.md#implemented-rule-2026-09-17); the ankle and stray other-knee series are now dropped before the fit).
- *Cost:* setting the option adds a `robust_center` entry to the preprocessing identity, so the
  run starts fresh on fold 0.

### Robust-centre re-score (2026-09-17)

Measured, inference only. The three fold-0 control checkpoints of
`overnight-20260916T214520Z-t8k_kw2z` (runs 00–02) re-scored the rescued fold-0 studies at batch
size 1 with bf16 autocast; the rest of fold 0 keeps its saved predictions. The comparison uses a
paired report-group bootstrap (2,000 draws), with Δ macro AUC ×1000 averaged over the three
checkpoints. Tables: [macro](../../outputs/analysis/series-filtering-20260917/robust-centre-rescore/macro_deltas.csv),
[per target](../../outputs/analysis/series-filtering-20260917/robust-centre-rescore/per_target_deltas.csv),
[series paths](../../outputs/analysis/series-filtering-20260917/robust-centre-rescore/series_sampling.parquet),
[sanity](../../outputs/analysis/series-filtering-20260917/robust-centre-rescore/sanity_slots.csv).
Script: [`robust_rescore.py`](../history/series-and-sampling-sources/series-filtering/robust_rescore.py.txt).

- *Patch:* a `KneeStudyDataset` subclass replaces a rejected reference with the robust fit. Series
  dropped from the fit remain model inputs. A wrapper around `build_pool` passes the robust centre
  only to series whose acquired slice range contains it. Every other series gets a reference with
  no centre, so it samples locally.
- *Paths:* none of the rescued series sample locally. When any series in a slot samples
  physically, `build_pool` keeps only the physical triplets. Each of the 12 out-of-range series
  (all in the 10 range-gated studies, 8 of them axial fluid) shares its slot with an in-range
  series, so it leaves the input. The other 129 of 141 series sample physically. Triplets stay at
  1,102 with no shortfall, no preparation failures and valid geometry.

| Rescued studies | n | Δ ×1000 (95% CI) | Per checkpoint |
|---|---:|---:|---|
| 20 mm robust fit | 21 | −0.107 (−0.256, +0.036) | −0.126, −0.044, −0.150 |
| Range-gated fit | 10 | −0.001 (−0.114, +0.110) | −0.057, +0.060, −0.006 |
| 20 mm fit only | 11 | −0.105 (−0.202, −0.024) | −0.069, −0.104, −0.143 |

- On their 10 shared studies, the two fits give identical predictions. The whole loss therefore
  comes from the 11 studies that only the 20 mm fit rescues. There, the far series project inside
  their range, so all 73 series switch from local to physical sampling. The range-gated rescue,
  which also removes the out-of-range series, is neutral.
- Per target (20 mm fit, 21 studies): medial and lateral OA fall by 0.56 ×1000 each and effusion
  rises by 0.56, with the same sign on all three checkpoints. These studies contain only 1–4
  positives for each of these targets.

### Implementation (2026-09-17)

The option reproduces the 20 mm patch above. The implemented behaviour, the identity effects and
the CPU verification (null against HEAD, 20 mm and 10 mm) are recorded in
[current preparation](../data-preparation/current-preparation.md#robust-study-centre-removed).
The runs were made on 2026-09-18 and all landed inside the control spread
([results](../validation/results/2026-09-18-overnight-results.md)), which is what removed the
option.

- *Counts:* after the ankle exclusion, 177 training studies are rejected, and 20 mm rescues 130 of
  them (21 of 31 on fold 0). The 10 mm threshold rescues all 177 and excludes 256 series from the
  fits, against 155 at 20 mm. No inference check has been run at 10 mm.
- *Re-score (measured, inference only):* the `ranked-filtered-seed42` epoch-9 checkpoint was run
  with the option at 20 mm, with the identity check bypassed.
  - The 21 rescued fold-0 studies were re-scored at batch size 1 with bf16 autocast; the other
    883 studies keep their saved predictions.
  - Macro AUC moves from 0.889573 to 0.889228 (−0.00035).
  - A batch-1 recompute without the option gives 0.889574, with a maximum probability difference
    of 0.0019 from the saved predictions.
  - All 105 non-empty slots of these studies switch from local to physical sampling, and the mean
    absolute probability change is 0.028.
  - The three control checkpoints above averaged −0.0001 (from −0.00004 to −0.00015), so this
    checkpoint loses more than any of them. The result comes from one checkpoint and 21 studies
    (`outputs/analysis/robust-center-implementation-20260917/rescore.json`).
- *Expectation:* the run is expected to tie. Seed-to-seed noise is 0.0029 ([strategy](../validation/validation-strategy.md#what-a-comparison-can-resolve)), so a single run
  cannot resolve a change of this size.
