# Evaluation tie-break investigation

**Implemented (2026-09-17)** with crop coverage added as the third criterion
([implementation](#implementation-2026-09-17)). Before it, the only seed-dependent step of
`select_triplets` at inference time was a seeded `source_order` permutation that broke exact ties
in `entitlement - assigned` when a study had more than one series of the same plane/contrast type,
so validation and test scoring depended on the training seed. This investigation characterizes
that sensitivity, audits why the repeated series tie, and designs the deterministic ranking that
replaced the permutation; training keeps its seeded target, band and supplier draws.

Scripts: [snapshots](../../history/sources/series-and-sampling/README.md#tiebreak-design) (`analyze_sensitivity.py`, `pool_stats.py`,
`metadata_analysis.py`, `analyze_rules.py`, `headers.py`, `simulate_rules.py`; logs not kept),
fold-0 validation split.

## How much the seed moves

436 of 904 fold-0 validation studies have 661 repeated-type slots (631 with two series, 28 with
three, 2 with four). Of 6,868 repeated-slot targets, 3,165 are exact `entitlement - assigned`
ties and 15 more are float near-ties.

| Comparison | Slots differ | Studies differ | Triplets differ (repeated slots) |
|---|---:|---:|---:|
| seed 42 vs 43 | 329/661 | 252/904 (0.578 of repeated) | 3,046/6,868 |
| seed 42 vs 44 | 328/661 | 246/904 (0.564) | 3,005/6,868 |
| seed 43 vs 44 | 305/661 | 247/904 (0.567) | 2,808/6,868 |

Two seeds change 305–329 slots, 246–252 studies, about 44% of repeated-slot triplets — about 6.6%
of all ~45.8k validation triplets. 660 of 661 slots differ across at least one pair of 10 tested
seeds. The order changes supplier (series) counts in only 84 slots, and there by one; otherwise it
changes only which depths each supplying series covers. The final triplet is always the preferred
series' closest slice to the target depth.

## Why the series tie

92 of 904 studies contribute non-physical (local) sources to a slot (44 of 661 slots), the rest are
physical Gaussian draws. Repeated series are mostly complementary rather than duplicate
acquisitions:

| Composition | Slots | Share |
|---|---:|---:|
| Same geometry, different protocol (e.g. PD+T2 pairs) | 280 | 42% |
| Different geometry and protocol (e.g. standard plus thin ACL-oblique coronal; 44 mix 2D/3D) | 247 | 37% |
| Same geometry and TR/TE (mostly retakes) | 67 | 10% |
| Minimal headers, no TR/TE | 58 | 9% |
| Remaining (same geometry+TE range, multi-echo, plain retake) | 9 | 1% |

Header availability across 5,935 repeated-group series: Acquisition/series times, `ProtocolName`
and derivation tags are absent everywhere. `SeriesNumber` and `AcquisitionNumber` are always
present. `SeriesDescription`, TR/TE/`AcquisitionMatrix`/`MRAcquisitionType` are present for 92%
(862 of 5,935 `SeriesDescription` values are the placeholder `DummySeriesDesc!`). `ImageType` is
always `ORIGINAL\PRIMARY` except Philips `AX PD FS MPR` reformats (9 slots — the only PD-FS
contrast present there) and one Siemens DESS MPR. 51 slots are true same-protocol retakes with a
later `SeriesNumber`; 13 GE deep-learning-reconstruction pairs (series *N* and *40000+N*) are the
same acquisition reconstructed twice.

## Proposed deterministic rule

The design as first proposed; the implemented ranking inserts crop coverage as step 3 (from the
[follow-up](#follow-up-measurements-2026-09-17)) and orders a same-signature chain newest first
instead of flagging every superseded retake equally. One ranking per slot, used as `source_order` in both training and evaluation; training still draws
its random target, band and supplier picks on top of that order.

1. Originals before reformats (decides 1.6% of tie targets).
2. Latest same-signature retake (8.2%): signature is TR, TE rounded to 1 ms, echo train length,
   flip angle, `MRAcquisitionType`, `SequenceVariant`, `ScanOptions`, rows, columns.
3. Finer acquired in-plane resolution, clipped at `data.spacing_mm` (26%).
4. `SeriesInstanceUID` (64%, the residual arbitrary but reproducible fallback).

Exact-fraction entitlements are kept (no rounding change).

Deciding criterion at tie rows, this rule (`quality` in `analyze_rules_final.log`):

| Criterion | Tie rows decided | Share |
|---|---:|---:|
| SeriesInstanceUID | 2,028 | 0.639 |
| Acquired resolution (worst axis) | 832 | 0.262 |
| Superseded same-protocol retake | 261 | 0.082 |
| Reformat | 52 | 0.016 |

## Rejected alternatives

- **Closest slice first:** mean depth error only improves from 0.875 mm (seed 42) to 0.844 mm
  (`closest_then_quality`), and it changes the deciding rule for 44% of triplets against the
  proposed order (`analyze_rules_final.log`) for a use of first-slice bias that swamps the other
  criteria.
- **Thinner slices**: dropped; it does not track the observed retakes/reformats and would not
  decide the bulk (64%) of ties, which have no quality differential at all. **Coverage** was first
  dropped for the same reason, then adopted after the follow-up showed that it removes every
  low-coverage tie win.

Quality proxies barely move between rule choices: mean clipped worst acquired resolution 0.6862 mm
(proposed) vs 0.6868 (seed 42) vs 0.6869 (UID-only) vs 0.6873 (reversed quality); superseded-retake
triplets 258/259/260/263; reformat triplets 54/55/55/56 (proposed/seed 42/UID/reversed). The
proposed rule and seed 42 pick the same triplet in only about half of repeated-slot targets
(`quality` vs `seed42` triplets differ 0.504), so determinism itself, not a large quality gain, is
the point.

**The only real quality lever** is dropping superseded same-protocol retakes: 51 slots, about 258
triplets, 0.6% of validation triplets. That is a separate, smaller experiment from the tie-break
rule itself.

## Follow-up measurements (2026-09-17)

Measured, CPU only, on fold 0 at the validation budget. Script:
`followup_coverage.py` (`041bd16:docs/history/series-and-sampling-sources/tiebreak-design/followup_coverage.py.txt`). Tables:
`outputs/analysis/s/series-filtering-20260917/followup/tiebreak_coverage.json` and
`tiebreak_coverage_series.parquet`. The re-run reproduces the proposed rule's triplets in all 661
slots.

Two coverage measures are used:

- **Crop coverage:** the fraction of a 140 mm × 140 mm square that lies inside the series'
  acquired field of view. The square sits in the series' image plane, centred on the projection of
  the study centre. The loader's centre is used, or the 20 mm robust centre when the loader
  rejects it. 10 slots have neither and are excluded, which leaves 651 slots and 3,122 tie rows.
  The model's crop is different: `_crop_geometry` in
  `src/knee_abnormality/data/preprocessing/spatial.py` centres it on the image centre. This
  measure therefore shows how much of the knee-centred region a series contains. It is not the
  fill of the model input: the image-centred crop is incompletely filled for only 37 of 1,354
  series.
- **Depth coverage** (physical pools only): the series' share of the truncated-Gaussian target
  mass over the slot's union support.

Series crop coverage has a mean of 0.983. 24 series fall below 0.8. Half of them are small-FOV
patellar axials (mean 0.29), and all of those sit in local-path slots, where the loader's centre
is rejected. Within a slot, crop coverage differs by 20 percentage points or more in 24 slots (12
of them contain a patellar axial). After rounding to 10%, it differs in 53 slots.

| Ranking | Slots where rank 1 has ≥ 20 pp less coverage (patellar) | Tie rows | Tie rows won by ≥ 20 pp less coverage (patellar) | Tie-won mean crop coverage | Mean crop coverage, all 6,766 repeated-slot triplets | Tie-won mean depth coverage |
|---|---:|---:|---:|---:|---:|---:|
| Proposed | 16 (10) | 3,122 | 89 (60) | 0.9777 | 0.98278 | 0.9693 |
| Coverage before resolution | 0 (0) | 3,122 | 0 (0) | 0.9973 | 0.98283 | 0.9710 |
| Seed 42 | not computed | 3,114 | 64 (30) | 0.9810 | 0.98271 | 0.9695 |

The variant ranks by reformat, superseded retake, crop coverage (descending, rounded to 10%),
resolution, then UID. Its deciding criteria at tie rows (651 slots) compare with the proposed
rule as follows:

| Criterion | Proposed | Coverage variant |
|---|---:|---:|
| Reformat | 52 (1.7%) | 52 (1.7%) |
| Superseded retake | 255 (8.2%) | 255 (8.2%) |
| Crop coverage | — | 193 (6.2%) |
| Acquired resolution | 832 (26.6%) | 687 (22.0%) |
| SeriesInstanceUID | 1,983 (63.5%) | 1,935 (62.0%) |

- Coverage decides 145 rows that resolution decided before and 48 that UID decided before.
- 22 of 651 slots change their triplets.
- Slots whose rank-1 series has at least 20 pp less depth coverage: 40 under the proposed rule,
  37 under the variant.

The variant removes every low-coverage tie win. Entitlement still fixes how many triplets each
series supplies, so the mean crop coverage of all realized repeated-slot triplets moves by only
+0.00005. The low-coverage series, mostly patellar axials in local-path slots, still supply their
share at different depths. The effect on AUC is untested.

## Implementation (2026-09-17)

Preprocessing identity `ranked_filtered_physical_gaussian_overlap_spatial_tokens_reflection_v10`,
shipped together with the [series exclusions](../current-preparation.md#series-indexing-guards-and-exclusions).
The ranking is described in the [implemented sampler](implemented-sampler.md).

- `read_header` (`dicom.py`) keeps 12 series tags from the header it already parses; `index_series`
  (`series.py`) stores the most common set as a `SeriesSummary` (series number, reformat flag,
  signature, acquired resolution, receive coil). No extra file is read.
- `build_pool` (`sampling.py`) stores `CandidatePool.ranking`: reformat, number of newer
  same-signature retakes, crop coverage (loader's accepted centre only, rounded to 0.1), acquired
  resolution (clipped at `data.spacing_mm`, rounded to 0.01 mm), UID. A value missing for any
  ranked series ties that step for the slot.
- `select_triplets` lists covering sources in that order in both modes and accumulates exact
  `Fraction` entitlements. The `source_order` stream is gone; training keeps its target, supplier
  and band streams. Evaluation takes no seed (`training_seed=None`), and checkpoint validation no
  longer compares the configured seed.

Two deliberate differences from the analysed variant: a study without an accepted loader centre
ties at coverage (the follow-up used the 20 mm robust centre there, which ranked the patellar
axials of local-path slots last), and a chain of three same-signature retakes is ordered newest
first rather than flagging both older series equally.

### Verification

CPU, [snapshot scripts](../../history/sources/series-and-sampling/README.md#deterministic-selection) (`index_census.py`, `check_design_orders.py`,
`determinism.py`); outputs in `outputs/analysis/d/deterministic-selection-20260917/`
(`index_report.json`, `index_slots.parquet`, `determinism.json`).

- **Against the design:** on the 661 fold-0 slots of the follow-up, the implemented ranking equals
  the coverage variant (accepted centre) or the proposed rule (no accepted centre) in 658. The
  other 3: the stray other-knee series of `…42012883` is now dropped, and two three-retake chains
  (`…28177364` sagittal T1/PD, series 15/16/17; `…21522084` axial fluid, series 2/7/8) are ordered
  newest first.
- **Against UID-only order** (validation budget, after the series exclusions):

  | Scope | Repeated slots | Ranking differs | Triplets differ | First differing criterion (reformat / newer retake / coverage / resolution / UID only) |
  |---|---:|---:|---:|---|
  | Fold 0 | 660 | 127 | 126 | 9 / 49 / 27 / 173 / 402 |
  | All training | 2,894 | 617 | 597 | 53 / 251 / 133 / 706 / 1,751 |
  | Example test | 2 | 0 | 0 | 0 / 0 / 0 / 0 / 2 |

- **Determinism** (full `__getitem__` path with a stub decoder): fold-0 evaluation selections of
  all 904 studies are identical for configs with seed 42 and 43, across two separate process
  pools, and for epoch 0 and 3. Training selections of 400 fold-0 training studies are identical
  when repeated and differ in all 400 studies between seeds 42 and 43 and between epochs 0 and 1.
- **Missing tags (all 24,386 series):** no `SeriesNumber` 0, no TR/TE signature 1,211, no
  acquired resolution 1,213, no receive coil 5,672; 54 series are reformats.

### Re-score of the 2026-09-17 checkpoints

Inference only, RTX 3090, bf16, batch size 8 in dataset order, fold 0 (904 studies), new code with
the preprocessing-identity check bypassed in the scratch script only (
`rescore.py` (`041bd16:docs/history/series-and-sampling-sources/deterministic-selection/rescore.py.txt`); `outputs/analysis/d/deterministic-selection-20260917/rescore/`
`rescore.json` and per-checkpoint predictions). "Saved" is each run's selected-epoch validation
(its own training seed decided the old ties); the saved prediction files reproduce the selected
scores. Paired report-group bootstrap, 2,000 draws.

| Checkpoint | Saved macro AUC | New macro AUC | Δ | 95% CI |
|---|---:|---:|---:|---|
| `00-control` (seed 42) | 0.892995 | 0.892697 | −0.00030 | [−0.00079, +0.00017] |
| `01-control-seed-43` | 0.889837 | 0.889796 | −0.00004 | [−0.00052, +0.00046] |
| `02-control-seed-44` | 0.891627 | 0.891729 | +0.00010 | [−0.00041, +0.00063] |
| `03-ema-0.999` (seed 42) | 0.893155 | 0.892854 | −0.00030 | [−0.00076, +0.00014] |

- The three controls average −0.00008 [−0.00041, +0.00025]; no checkpoint moves significantly.
- Control spread: 0.00316 (sd 0.00158) before, 0.00290 (sd 0.00148) after. The evaluation inputs
  now coincide, so the remaining 0.0029 is model (training-seed) noise; the seed-dependent
  evaluation ties contributed little of the earlier spread.
- EMA minus the seed-42 control stays +0.00016 (+0.00016 before).
- These checkpoints were trained with the old seeded order and with every series present, so the
  deltas measure a change of evaluation inputs, not the value of training on the new identity;
  a seed-42 control retrained on the new identity was never run as a separate test (no record of one
  exists); every later training used the deterministic ranking.
