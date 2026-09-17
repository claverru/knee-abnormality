# Historical series and sampling analysis sources

Scratch analysis code from the 2026-09-17 session that produced the series filtering, source ranking,
plane/contrast importance, run-agreement and robust-centre implementation findings. The scripts ran from a session scratch
directory (`/tmp/claude-1000/<session>/scratchpad/<folder>/`) against the uncommitted tree of that day,
so their hardcoded paths, sibling imports (`common.py`) and bypassed identity checks are
snapshots, not maintained commands; they sit outside the executable
[script catalog](../../../scripts/INDEX.md). Python files carry a `.txt` suffix; review
verdict files are kept as written. Logs, pickles and intermediate tables were not kept unless the
outputs column names them.

## tiebreak-design

Evaluation tie-break investigation: seed sensitivity, candidate-pool statistics, DICOM header inventory, ranking rules and the coverage follow-up. Findings: [record](../../data-preparation/physical-sampling/evaluation-tie-break-investigation.md). Outputs: `outputs/analysis/series-filtering-20260917/followup/ (coverage); logs and pickles were not kept`.

| Original file | Snapshot | SHA-256 |
|---|---|---|
| `analyze_rules.py` | [analyze_rules.py.txt](tiebreak-design/analyze_rules.py.txt) | `d1555ddb8125dbe2b97ae3c8ca76c867976f02e34a8208cca1715ff6d1270e38` |
| `analyze_sensitivity.py` | [analyze_sensitivity.py.txt](tiebreak-design/analyze_sensitivity.py.txt) | `a4fdcac6f3e68791282e941465e681a8755493d63a81d38bd9bea70e76d23799` |
| `followup_coverage.py` | [followup_coverage.py.txt](tiebreak-design/followup_coverage.py.txt) | `544d434f2bac496fc95c84fe99f786ae5a3d74ca789ce1059988d7b5ed742a02` |
| `headers.py` | [headers.py.txt](tiebreak-design/headers.py.txt) | `e8a651fb3c2178e8f563138b24aff70773500ccb1054982e9749f00bc031060c` |
| `metadata_analysis.py` | [metadata_analysis.py.txt](tiebreak-design/metadata_analysis.py.txt) | `59eeeb85ae7e17efafaf61ac3983c4569cc323b4e286932930bf9b369640aa98` |
| `pool_stats.py` | [pool_stats.py.txt](tiebreak-design/pool_stats.py.txt) | `212784e3de0fbb0d631655012537ea652b77f0c5a73a03c71d7cd3d0c36f8e16` |
| `sensitivity.py` | [sensitivity.py.txt](tiebreak-design/sensitivity.py.txt) | `1c10a9c99b1a0766c43e52160b2081db8aecafecefb0c4599fe37d4e8febb5b7` |
| `simulate_rules.py` | [simulate_rules.py.txt](tiebreak-design/simulate_rules.py.txt) | `f24d83eb3ac6eaa1c1e4106aa028321520f0a877f3efb36e445b67d29c3ca5db` |

## type-importance

Plane/contrast importance: leave-one-type-out inference on the three 2026-09-17 controls, attention shares, availability and triplet capacity. Findings: [record](../../validation/results/2026-09-17-plane-contrast-importance.md). Outputs: `outputs/analysis/type-importance-20260917/`.

| Original file | Snapshot | SHA-256 |
|---|---|---|
| `analyze.py` | [analyze.py.txt](type-importance/analyze.py.txt) | `399dd54878afd4a352c182f3faef5a9af75e84123eee8cffa9811ab0065662dd` |
| `capacity.py` | [capacity.py.txt](type-importance/capacity.py.txt) | `419a375b9529f827c9ebbebd7087a6bff0d755c5c8fb9dd014ed1785946e8789` |
| `infer.py` | [infer.py.txt](type-importance/infer.py.txt) | `eba02d0e59f397a8929d1712060d1437391d2e8f45f5761799d0e9b050c566e0` |
| `numerics.py` | [numerics.py.txt](type-importance/numerics.py.txt) | `54bf66a3897e78945b5afaac484f5037612991503b951c787c7c62e1d57014dd` |
| `numerics_no_tf32.py` | [numerics_no_tf32.py.txt](type-importance/numerics_no_tf32.py.txt) | `01ab5fd06b534a4810c2129b6c5b3e80a1644c31381aca663433911f8e849292` |

## series-filtering

Series filtering and cleaning census: indexing of all training series, geometry and contrast signals, review contact sheets, flag precision, removal inference check, centre-rescue follow-up and robust-centre re-score. Findings: [record](../../metadata/series-filtering-census.md). Outputs: `outputs/analysis/series-filtering-20260917/`.

| Original file | Snapshot | SHA-256 |
|---|---|---|
| `analyze.py` | [analyze.py.txt](series-filtering/analyze.py.txt) | `8665d9d9826b9eb24d4d144fa95c24e88b315791b8ee7c036d94bff9674b2da4` |
| `census.py` | [census.py.txt](series-filtering/census.py.txt) | `4483201e4b3ac9d66cbbdbde7649c2feec2e237dd6d0a08220ab2eba08ed4d6c` |
| `common.py` | [common.py.txt](series-filtering/common.py.txt) | `79e0120266b62173ed64500423e5f09efd593d10535d77794b7987e4f2ccbdc4` |
| `contrast.py` | [contrast.py.txt](series-filtering/contrast.py.txt) | `b8de857f90a4e1cfe2db205a7f48166978ab90d84dcf734737bbff14899a973a` |
| `flags.py` | [flags.py.txt](series-filtering/flags.py.txt) | `a9ea6201e2fcd8b02a28d0be8c61dffd21b2add65f3cf8779c86021b65e6b4ee` |
| `followup_geometry.py` | [followup_geometry.py.txt](series-filtering/followup_geometry.py.txt) | `8ebadd1c24fe3443c6b70e17672dbc261bfe4d60fec4333e6d39750a5434ab62` |
| `geometry_signals.py` | [geometry_signals.py.txt](series-filtering/geometry_signals.py.txt) | `111c999c32032e395b16ba141bd82dbc6bb94baff724aa3a0b7c64d06164353c` |
| `index_all.py` | [index_all.py.txt](series-filtering/index_all.py.txt) | `64728367e64911b9cdbb8b35a60a24f78c06e2108336cbd29eb49e6be97f5fe7` |
| `infer.py` | [infer.py.txt](series-filtering/infer.py.txt) | `0c808bda976248e911dcf3f3e7c01027f4eef98c8709b6c8151c042a55845c17` |
| `make_geo_review.py` | [make_geo_review.py.txt](series-filtering/make_geo_review.py.txt) | `9e3007c5ee6c1a21c69e27cf58fd50ba2e7b22387b5f4632c92088cb851ae90f` |
| `make_random_review.py` | [make_random_review.py.txt](series-filtering/make_random_review.py.txt) | `b2140f327e4f355de5b1f98fdf130bc49464d938079ccfce711e3aa404193f2a` |
| `make_review_sets.py` | [make_review_sets.py.txt](series-filtering/make_review_sets.py.txt) | `fe1cef8461e9d0a25fcd1bf15f5cfb5dcb244f47cfba007e143c5d3a79a635f0` |
| `make_rows.py` | [make_rows.py.txt](series-filtering/make_rows.py.txt) | `81cbd2b02019628f7a8a5d7e9c0f69d7bfbcb6d1ebcf1dc9dcebda41d4ba84db` |
| `planes.py` | [planes.py.txt](series-filtering/planes.py.txt) | `b95983b3baf22ee6e1757a9204d1ba1ef636b88805b1ac63c18c292f5d9da646` |
| `precision.py` | [precision.py.txt](series-filtering/precision.py.txt) | `49db77b38e841355dddf768d55b7f5b80f5c7223fa690470ed35738147153c9e` |
| `robust_rescore.py` | [robust_rescore.py.txt](series-filtering/robust_rescore.py.txt) | `3299ace9dee5339ae24e02bc2611bc4af83da77bde4b5fb05b718498b6ac3e00` |
| `sheets.py` | [sheets.py.txt](series-filtering/sheets.py.txt) | `e3a5a54afd896c12c7e90d623cf6d546298f8fb9d3cfdce0a036a98103a4a70c` |
| `study_sheets.py` | [study_sheets.py.txt](series-filtering/study_sheets.py.txt) | `d8ceabe7c2492f31c467836f73208b5c2eedd2578e54b7cc35a65a60d3d7a22f` |
| `verdicts.py` | [verdicts.py.txt](series-filtering/verdicts.py.txt) | `dbd9fbc207e46ac3039416af4367de96070258e880e0be8b65fc3843d1a60b06` |
| `zoom.py` | [zoom.py.txt](series-filtering/zoom.py.txt) | `95ca855125a8da0aad4be9758b1d5ef63b81759b8f8e9457eaf6b9b2ac5359ad` |
| `verdicts_contrast.txt` | [verdicts_contrast.txt](series-filtering/verdicts_contrast.txt) | `420459e7f7fe922d845b6a9de6bf87175eb86908d6865a0dd1103c8c58010ad0` |
| `verdicts_geo.txt` | [verdicts_geo.txt](series-filtering/verdicts_geo.txt) | `767097bc8959ae6eccb095fb1afe3d0fe871bdda75bb6e0dc1c112aa8572860d` |
| `verdicts_random.txt` | [verdicts_random.txt](series-filtering/verdicts_random.txt) | `8c65082eba8f00c7c5f990703e95a657edce9c11983c7ee31fd52d0cc4b4fe14` |
| `verdicts_rejected.txt` | [verdicts_rejected.txt](series-filtering/verdicts_rejected.txt) | `e06c43aedbb3948cd28d96d9acd1d37172f4c305fbd6b1e47cecdd82c45e0e27` |

## deterministic-selection

Implementation verification of series filtering and source ranking: indexing census, design-order check, determinism check and re-score of the 2026-09-17 checkpoints. Findings: [record](../../data-preparation/physical-sampling/evaluation-tie-break-investigation.md#implementation-2026-09-17). Outputs: `outputs/analysis/deterministic-selection-20260917/`.

| Original file | Snapshot | SHA-256 |
|---|---|---|
| `check_design_orders.py` | [check_design_orders.py.txt](deterministic-selection/check_design_orders.py.txt) | `2d39a6a4f4b0845b36cac102add1dd1a1d52e98acda06e39a8d4b9e6a8e6db9f` |
| `determinism.py` | [determinism.py.txt](deterministic-selection/determinism.py.txt) | `d43da8ba062ef0168d24af4916dcfa166c4a2ebf87da89009057a04c167cccab` |
| `index_census.py` | [index_census.py.txt](deterministic-selection/index_census.py.txt) | `ea6eec0e65b357275b23bb356c7b89bf43fadbf97ffb18d07a500450dd65403e` |
| `rescore.py` | [rescore.py.txt](deterministic-selection/rescore.py.txt) | `a73499b4d9ab38aeddbb4d7c4e2ed4b8cf1b2e9ef1596fa4d1226ae8b157a77e` |

## prediction-agreement

Pairwise agreement of fold-0 predictions between the ranked-filtered runs and the seed controls. Findings: [record](../../validation/results/2026-09-17-ranked-filtered-result.md#prediction-agreement). Outputs: `outputs/analysis/prediction-agreement-20260917/`.

| Original file | Snapshot | SHA-256 |
|---|---|---|
| `analyze.py` | [analyze.py.txt](prediction-agreement/analyze.py.txt) | `20890354e9981467c1becc01f6544da8c6652198fbd64a1f4bfe4dedfa31bbe2` |

## offknee-review

Rough visual check of three off-knee series against their studies' sagittal series. Findings: [record](../../metadata/series-filtering-census.md#labelled-findings-in-off-knee-series). Outputs: `outputs/analysis/series-filtering-20260917/offknee-review.png`.

| Original file | Snapshot | SHA-256 |
|---|---|---|
| `render.py` | [render.py.txt](offknee-review/render.py.txt) | `33813fce8e491d144c8d3f70d4a4bae44cefb3ce74dc6e4b0e6d44239f60c5d0` |

## robust-center-implementation

Verification of the `data.robust_center_outlier_mm` implementation: study centres and evaluation pools for all training studies under HEAD and under the option at null, 20 and 10 mm (run as `census.py <src root> <output pickle> <options JSON>`), their comparison with the re-score patch, and the fold-0 re-score of `ranked-filtered-seed42` with the identity check bypassed. Findings: [record](../../data-preparation/current-preparation.md#robust-study-centre-option). Outputs: `outputs/analysis/robust-center-implementation-20260917/` (`report.json`, `rescore.json`); pickles were not kept.

| Original file | Snapshot | SHA-256 |
|---|---|---|
| `analyze.py` | [analyze.py.txt](robust-center-implementation/analyze.py.txt) | `2a16ba88dca9a6a04d01d19258560689f98813369bc998a48197b746ad7b1abf` |
| `census.py` | [census.py.txt](robust-center-implementation/census.py.txt) | `8c6c7a4a08fdbf6bb3a548a1460361adcdca0340f6518f4c2b2406e6e8996cca` |
| `rescore.py` | [rescore.py.txt](robust-center-implementation/rescore.py.txt) | `94af73c74fe3b5bb98746d48b812ece152a204b44378def2eb4f41a18b1f3009` |
