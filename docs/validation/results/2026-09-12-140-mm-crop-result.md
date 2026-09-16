# 2026-09-12 140-mm crop result

**`crop140mm` scored 0.925 publicly**, improving on the 150.08-mm
`04-higher-lr` model by 0.007. Submission `56182948` completed successfully,
confirmed through the Kaggle API on 2026-09-12. Prefer its 140-mm geometry
for subsequent comparisons; the current baseline YAML already uses 0.625 mm
spacing at 224 pixels.

| Selected checkpoint | Full-fold AUC | Gold AUC | Soft-target BCE | Public AUC |
|---|---:|---:|---:|---:|
| Higher LR, 150.08 mm, epoch 10 | 0.88294 | 0.91756 | 0.45404 | 0.918 |
| Higher LR, 140 mm, epoch 8 | 0.88612 | 0.92698 | 0.45296 | 0.925 |

The [saved configuration](../../../outputs/training/crop140mm/effective_config.json) differs
from `04-higher-lr` only in spacing, apart from run name/output path:
0.625 versus 0.67 mm. Both used seed 42, `all_5fold_v3` fold 0, the same
input identities, ReXNet-200, ten training epochs, backbone/head LR
`2e-4`/`6e-4`, weight decay 0.01 and series dropout 0.2. Training source
snapshots have identical pipeline/package Python files; only an unrelated
ad hoc benchmark differs. The crop run records commit
`e313b5c92a12e20ecf74f8f3ca61e488ea46fa56` with a dirty working tree in
[`provenance.json`](../../../outputs/training/crop140mm/provenance.json); its
`source_snapshot.zip` preserves the executed source.

Local AUC increased by 0.00318, with nine of twelve targets improving.
Largest gains were Contusion +0.01088, MCL +0.00962 and Lateral Meniscus
+0.00885; Synovitis fell by 0.00835. Gold AUC and BCE also improved, although
the gold diagnostic still contains only eleven studies. These are selected
checkpoint comparisons on the same 904-study development fold; they do not
establish statistical significance or private leaderboard performance.

Training took 35.5 minutes. Both submissions used 2× images and batch size two.
The input dimensions and model are unchanged, so this improvement adds no CNN
pixel workload; official hidden-run runtime is unavailable. Reducing spacing
both tightens coverage and increases spatial sampling density, so the result
does not identify which effect caused the gain. A separate 336-pixel comparison
at fixed 140-mm coverage would use `140 / 336` mm spacing.
See the [submission artifact](../../inference/submissions/2026-09-12-140-mm-crop-submission.md).
