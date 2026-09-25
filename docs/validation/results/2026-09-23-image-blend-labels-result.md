# 2026-09-23 image-blended training labels: gold gain, public tie

**Blending an out-of-fold image teacher into `combined_v1` at 0.5/0.5 improved the label table's
gold agreement (+0.020) and fold 0 (+0.0035), then tied publicly.** S50 scored 0.945 against 0.942
for S48, the same code and seed on `combined_v1`, which is within run-to-run noise. The teacher came
from an older recipe (Steven v4 labels, ordinary BCE, another split), which hurt PF OA. A matched
teacher is deferred to the
[endgame](../../backlog/endgame.md#out-of-fold-self-distillation), where fold training produces
it anyway. This was the first test of that entry, run at 0.5/0.5 rather than its 0.7/0.3 plan.

## Why an image teacher

`combined_v1` already orders the report calls as gold does (high and medium, then low, then not
addressed, then negative). A rule that re-spaced the calls lowered gold agreement, so the remaining
label error sits within call classes and in report omissions
([report attributes](../../public-labels/report-attributes.md#consensus-mean-rule-tested-on-gold)).
Report text cannot fix that; images can. Public participants at the top of the board report the same
lever: mixing image-model predictions into the extracted label beat both replacing and masking
([public research](../../research/public-research.md#labels-are-the-reported-lever-at-the-top-and-extractor-tuning-is-not)).

## Table: `image_blend_v1`

- **Rule:** 0.5 × `combined_v1` + 0.5 × teacher, on every cell. The weight was fixed before any
  check and not tuned.
- **Teacher:** the mean of the ReXNet-200 and EfficientNet-B0 out-of-fold probabilities from the
  2026-09-18 five-fold runs on `all_5fold_v3`
  (`outputs/training/overnight-20260918T232636Z-mt7ouybk/runs/{rexnet,b0}-fold{0..4}`). Those runs
  used ordinary BCE, Steven v4 labels and ×1. Each study's value comes from the two models that held
  it out.
- **Tool:** [`prepare_image_blend_labels`](../../../scripts/tools/prepare_image_blend_labels.py)
  (`rtk proxy uv run --locked python -m scripts.tools.prepare_image_blend_labels`), CPU only, about a
  minute, deterministic and requiring fresh output paths. It fails unless each teacher run is the
  matching `all_5fold_v3` fold and its predictions hold exactly that fold's studies, and it records
  the silver change counts and the gold check below in `preparation.json` and on stdout.
- **Outputs:** `data/labels/prepared/image-blend-v1/` (labels SHA-256 `7bffba41…`) with
  `preparation.json`. The manifest is `configs/labels/image-blend-v1-label-sources.json`.
- **Gold labels** still override the table in training and evaluation.

### Gold check (58 studies)

| Table | Macro AUC | Eight Steven v4 targets |
|---|---:|---:|
| `combined_v1` | 0.9075 | 0.8828 |
| Teacher alone | 0.9063 | 0.9012 |
| `image_blend_v1` | **0.9275** | **0.9128** |

The blend beats `combined_v1` by +0.020 (paired study bootstrap: 80% [+0.010, +0.030]; 95%
[+0.006, +0.036]; positive in 99.6% of 2,000 draws). The teacher alone is not better (−0.0015, 95%
[−0.027, +0.025]), so the two sources complement each other.

| Label | `combined_v1` → blend |
|---|---|
| Contusion | +0.069 |
| Effusion | +0.049 |
| Medial OA | +0.039 |
| Baker's | +0.036 |
| Lateral Meniscus | +0.032 |
| Medial Meniscus | +0.029 |
| Fracture | +0.017 |
| MCL | 0.000 |
| ACL, Synovitis | −0.004 each |
| Lateral OA | −0.009 |
| PF OA | −0.013 |

Within report-call classes, the teacher separates gold positives better than the report labels do.
Among low-grade cells it scores AUC 0.75, against 0.65 for the labels' within-label rank.

### Change to training

On non-gold studies, the mean absolute change per label is 0.05–0.11 and the mean targets barely
move. Positive rates at 0.5 hold except PF OA (0.32 → 0.37), Synovitis (0.12 → 0.15) and Fracture
(0.061 → 0.048).

Fold-0 class-balance weights (k = 0.5) shift mildly:

| Label | Positive weight, `combined_v1` → blend |
|---|---|
| Fracture | 3.22 → 3.64 |
| MCL | 3.56 → 3.82 |
| Synovitis | 2.15 → 1.94 |
| PF OA | 1.27 → 1.17 |

## Fold-0 result

Run `outputs/training/image-blend` was trained on the `label-grades` worktree. Its recipe was
`k05-clean`'s (`def54f9`, seed 2263738815, k = 0.5, no oblique groups) with only
`labels.source: image_blend_v1` and its manifest changed. It selected epoch 9, trained in 1,592 s,
and scored 0.904133 at ×1 and 0.906294 at ×4.

| Metric | `k05-clean` | `image-blend` | Δ (paired report-group bootstrap, 95%) |
|---|---:|---:|---|
| Fold-0 macro AUC, ×4 τ 1 | 0.902760 | 0.906294 | +0.0035 [+0.0007, +0.0062] |
| [Calibrated judge](2026-09-22-calibrated-judge.md) | 0.7121 | 0.7156 | +0.0035 [+0.0018, +0.0051] |
| Fold-0 soft AUC against the frozen soft targets, ×4 | 0.7985 | 0.8050 | +0.0065 [+0.0046, +0.0084] |

Soft AUC weights every ordered pair of distinct studies by y_i(1 − y_j) of the frozen targets
([`compare_soft_auc`](../../../scripts/adhoc/compare_soft_auc.py)). It rises on eleven labels and
falls only on PF OA (−0.0015); MCL, flat on hard AUC, gains +0.014. Same-study pairs are excluded,
and the score still inherits the report labellers' bias. The script runs on CPU in about 30 s,
refuses an existing output directory and writes `summary.json` and `per_target.csv`. Its bootstrap
mirrors `compare_fold_runs` (multinomial report-group counts, seed 20260914, 2,000 draws, draws
dropped when a label loses a hard class), so the hard delta reproduces that tool's macro interval:

```bash
rtk proxy uv run --locked python -m scripts.adhoc.compare_soft_auc --reference outputs/training/k05-clean --candidate outputs/training/image-blend --output-dir outputs/analysis/soft-auc-new
```

Hard AUC by label:

| Label | Δ |
|---|---:|
| Effusion | +0.014 |
| Baker's | +0.006 |
| ACL | +0.005 |
| Medial Meniscus, Medial OA, Synovitis, Lateral OA, Contusion | +0.0035 to +0.0045 |
| Fracture | +0.002 |
| MCL | 0.000 |
| Lateral Meniscus, PF OA | −0.003 each |

All three fold-0 numbers are upper bounds. The teachers trained on studies that
`all_5fold_scanner_v1` fold 0 holds out, so the child learned indirectly from fold-0 labels.

- **Artifacts:** `outputs/analysis/image-blend-vs-k05-clean/`,
  `outputs/analysis/soft-auc-image-blend-vs-k05-clean/` and
  `outputs/analysis/calibrated-judge-image-blend/`.
- **Allowed differences:** the comparison allowed the label source and manifest, the
  `oblique_tilt_deg` field, and the preprocessing identity. `k05-clean` ran the main tree's
  uncommitted oblique code with the feature off, which follows `HEAD`'s code path.
- **Log warnings:** the run's "preparation conditions" all come from the near-empty-frame guard added
  with series cleaning (`b483140`). Every logged study was traced to a frame under 1% non-zero pixels.

## Public result

S50 (`k05 clean image-blend labels fold0`, ×4 τ 0.75 bs4, bundle `0fdf0b4b`) scored **0.945**,
against 0.942 for S48 (`k05-clean` on `combined_v1`) and 0.945 for `k_05` (S41) and `oblique`
(S47).

The rule fixed before the result was: 0.946 or more means proceed to a matched teacher, 0.944–0.945
is a tie, and 0.943 or less means drop the idea. The result is a tie. The +0.003 equals the gap
between `k_05` and `k05-clean`, two same-seed runs of nearly the same recipe.

## Why the teacher held it back

- **Label definitions:** the teachers learned Steven v4. That agrees with `combined_v1` on eight
  targets but is looser on ACL, MCL, PF OA and Fracture (MCL positives 15% against 5%, PF OA 46%
  against 32%).
- **Loss:** the teachers used ordinary BCE, not class-balanced k = 0.5, so their rare-label
  probabilities sit lower in scale.

PF OA shows the effect. The blend raises its training positive rate from 0.317 to 0.369, and it
loses on gold (−0.013), fold-0 hard AUC (−0.0025) and fold-0 soft AUC (−0.0015).

## Caveats

- **Gold is small:** 58 studies, enriched for disease. A gold gain has failed to transfer before
  (Tonylin: +0.010 on gold, −0.008 publicly), although that table was text-derived.
- **One fold, one seed:** the public difference is one comparison of single fold-0 models.
- **Self-distillation** can reinforce a teacher's errors; the report half of the target limits it.
