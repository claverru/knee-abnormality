# 2026-09-13 new public label comparison

**Follow-up:** the recommended Tonylin fold-0 comparison completed with a
public regression, and Steven v4 was restored. See the
[training result](../../validation/results/labels-and-losses/2026-09-13-tonylin-label-comparison.md).
The original source-screening evidence below is retained for provenance.
Zhen V7 and Laymond v2 do not improve overall
reference agreement over Steven v4. Rayan's original release offers a narrower
Fracture-source experiment, but its newer release cannot be justified using the
older 0.8991 score. Those alternative sources have not been adopted or scheduled.

**Re-screened on 2026-09-21:** the newer tables were scored in the
[2026-09-21 re-screen](2026-09-21-public-label-rescreen.md); none beats Steven v4 outside
Fracture, and stricter sources expose Steven v4's over-broad MCL.

**Screened on 2026-09-20, not audited:** newer public tables keep appearing — two
gpt-luna releases on 09-18 among them — and none was scored here. The reason is
the follow-up above plus the [local-to-public correlations](../../research/data-and-validation/val-test-correlations/INDEX.md):
the best gold candidate of 09-13 beat Steven v4 by 0.010 on the 58 studies and
lost 0.008 publicly when trained on, and gold-only AUC correlates with nothing
across our nineteen comparable submissions. A table's gold score does not
justify a run. The public label work worth reading is summarized in the
[2026-09-20 sweep](../../research/competition-and-prior-art/public-research.md#the-2026-09-20-re-sweep).

## Published claims and reproduced scores

Scores below are label-table macro ROC-AUC against the original official
binary labels, except where explicitly described as image-model scores.
Each source is aligned by `StudyInstanceUID`, with Steven rescored on the same
available reference studies. No public gold predictions are overwritten by
this audit. An artificial 1.0 caused by an already overwritten source is
excluded from candidate ranking.

| Source and pinned Kaggle release | Published claim | Recomputed label AUC | Interpretation |
|---|---|---:|---|
| [Steven v4, release 6](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels/versions/6) | 0.8927 on gold | **0.892707**, 58 studies | Current comparison source; claim reproduced. |
| [Tonylin soft labels, release 3](https://www.kaggle.com/datasets/tonylin1026/rsna-knee-refined-soft-labels/versions/3) | No numerical claim in retrieved card or released documentation | **0.902700**, 58 | Best measured candidate, +0.009993. Score the `*_soft` columns. |
| [Zhen V7, release 1](https://www.kaggle.com/datasets/zhen931/rsna-knee-weak-labels-v7/versions/1) | No numerical claim in retrieved card or released documentation | **0.879325**, 57 | Steven scores **0.893237** on the same 57; delta -0.013912. |
| [Laymond Qwen2.5-32B v2 filled, release 4](https://www.kaggle.com/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels/versions/4) | Filled 0.8686; raw 0.8521 on 58 gold studies | **0.868606** filled; **0.852107** raw | Both claims reproduce, below Steven. Score `*__confidence`, which this v2 defines as P(present). |
| [Tasmeem corrected, release 2](https://www.kaggle.com/datasets/tasmeemreza/rsna-knee-refined-llm-labels/versions/2) | **Image-model** gold AUC 0.9062 to 0.9101; public LB 0.919 to 0.926 | Not a usable label benchmark | All 58 rows exactly copy official gold. Direct CSV AUC is 1.0. The model gains are author reports, not reproduced here. |
| [Rayan, original release 1](https://www.kaggle.com/datasets/rayanbabur/rsna-knee-2026-calibrated-soft-targets/versions/1) | 0.8991 on 58 gold studies | **0.899131**, 58 | Claim reproduced from the older release; all improvement is Fracture. |
| [Rayan, current release 2](https://www.kaggle.com/datasets/rayanbabur/rsna-knee-2026-calibrated-soft-targets/versions/2) | Card and bundled report retain 0.8991 | Not a usable label benchmark | All 58 gold rows now copy official labels; silver targets also changed substantially. The older score does not measure this release. |
| [Yunus four-source merged, release 2](https://www.kaggle.com/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged/versions/2) | Card reports 0.8798 for a Gemini source | Not a usable label benchmark | Card describes `report_labels_v4.csv`, while the archive contains merged `report_labels_v5.csv`; all 58 gold rows copy official labels. |

Every table covers all **4,349 report-only studies**, with unique IDs, finite
targets in `[0, 1]`, and no unexpected study IDs. Zhen's two tables and Pilkwang
omit the same one gold study; other compared tables cover all 4,407 studies.
None of Tonylin's reference rows is wholly binary or exactly equal to the
official target vector. This rules out wholesale hard-gold replacement in its
soft columns, but cannot rule out tuning or indirect use of gold labels.

## What the apparent gains contain

Tonylin improves 8 of 12 reference AUCs. Fracture contributes about **77%** of
its net macro gain; excluding Fracture, mean improvement is only **0.0025**.

| Target | Steven v4, 58 | Tonylin soft, 58 | Difference |
|---|---:|---:|---:|
| Fracture | 0.793056 | 0.885417 | +0.092361 |
| MCL | 0.968254 | 0.981859 | +0.013605 |
| Medial Meniscus | 0.948317 | 0.959736 | +0.011418 |
| ACL | 0.987132 | 0.998162 | +0.011029 |
| Lateral Meniscus | 0.878882 | 0.889441 | +0.010559 |
| Lateral OA | 0.832689 | 0.820116 | -0.012573 |
| Contusion | 0.859649 | 0.851552 | -0.008097 |

Across the report-only pool, Tonylin changes **50,642 / 52,188** numerical
targets at tolerance `1e-12`, but only **1,158 / 52,188 (2.22%)** cross the
`>= 0.5` decision threshold. Mean absolute difference is **0.08884**, mean
per-target Spearman correlation **0.90706**, and 311 cells differ by more than
0.5. These numerical differences combine label content and score semantics;
they do not establish clinical errors in either source. Tonylin provides no
generation code or detailed methodology in the inspected release.

A paired study bootstrap with seed **20260913** and **5,000** draws gives a
95% percentile interval of **[+0.00090, +0.02028]** for Tonylin's macro gain.
One draw lacking a class in a target is excluded. This measures resampling
uncertainty on the reused 58-study endpoint, not source-selection bias, prior
gold tuning, patient clustering or variability in training. It supports trying
the source, not declaring a downstream improvement.

Zhen V7 changes only Synovitis and Fracture relative to its bundled V4:
3,670 and 2,460 report-only cells respectively. Their values change without
crossing the 0.5 threshold. Its shared-57 Fracture AUC is strong
(0.896324 versus Steven's 0.791176), but its losses in Contusion, Effusion,
Lateral OA, Lateral Meniscus and Synovitis outweigh that gain. The bundled
Zhen V4 has the same gold AUC as Pilkwang (0.870040) while changing 5,923
report-only target values; equal gold AUC does not imply identical labels.

Rayan release 1 reconstructs as **Steven v4 clipped to `[0.005, 0.995]` on
eleven targets**, to floating-point precision, with **Pilkwang's Fracture
scores exactly on all 4,406 shared studies**. Its Fracture AUC increases from
0.793056 to 0.870139; the other eleven reference AUCs are identical to Steven.
Release 2 changes 51,862 report-only cells relative to release 1, so its
retained numerical claim cannot be carried forward to those new targets.

## Reproduction and experiment boundary

The [comparison source manifest](../../../configs/labels/public-label-comparison-sources.json) pins
the eleven compared tables, dataset versions, explicit target-column mappings,
claims, metadata snapshots, archive/file SHA-256 hashes and official inputs.
Version-specific public Kaggle API download URLs are recorded per source.
Downloaded generator notebooks are stored as source artifacts and were not
executed. The existing production label manifest remains the source of the
active Steven v4 identity.

The [comparison script](../../../scripts/adhoc/compare_public_labels.py) produced
`data/labels/audit/new-public-20260913/final/`: `audit.json`,
`gold_per_target.csv`, `pairwise.csv` and `large_silver_disagreements.csv`.
The report records script/manifest hashes, dependency versions and bootstrap
configuration. Outputs contain IDs and numerical targets, without report text.
Reproduce into a fresh directory:

```bash
uv run --locked python -m scripts.adhoc.compare_public_labels --output-dir data/labels/audit/new-public-20260913-reproduction
```

A training comparison should vary only report-derived training targets while
retaining the official training gold labels, seed, fold membership, sampling,
model and training budget. Keep the `all_5fold_v3` frozen validation targets:
fold 0 has 904 validation studies, of which only 11 are gold. The other 893
validation targets are frozen report-derived labels, so their score also
measures agreement with the existing teacher. Report the small gold-only
diagnostic separately and assess any later public-LB evidence without treating
the all-study validation score as independent adjudication of label quality.
No training, label-source integration, tests or submission was performed.
