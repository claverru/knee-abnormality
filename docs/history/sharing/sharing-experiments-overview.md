# Sharing and supervision experiment results

Historical measurements recovered from branch `1C` at `3b8c2cc1c7f7fbfe76ef2b149c91dba0c354002f`.
The experiments below used the sigma-80 reflection reference unless stated
otherwise. Their earlier recommendations are retained only where they explain
the next comparison; the [reflection/sigma-40 result](../../validation/results/sampling/2026-09-14-reflection-sigma-40-result.md)
now selects independent classifiers and attention at **0.892360 validation AUC
and 0.936 public AUC**. Current `main` uses that recipe with optional CoAtNet
backbone support; it does not implement 1C.

Read the [generalization investigation](generalization-investigation.md) for the
combined evidence, the [disagreement audit](oa-and-meniscus-disagreement-audit.md) for
OA labels and acquisition sensitivity, and [transfer research](../../research/geometry-and-views/medial-lateral-transfer.md)
for the retained future approaches. No new experiment is scheduled by this record.
[Historical source records](../sources/sharing/README.md) preserve the
original analysis commands and 1C architecture; saved run artifacts own exact
checkpoint reproduction.

## 2026-09-14 sharing generalization investigation

1A submission `56226273` completed with **0.930 public AUC**, versus reflection
0.931, 1B 0.925, original 1C 0.928 and sigma-40 1C 0.931. These are the
historical sigma-80-reference comparisons; reflection/sigma 40 later scored 0.936. The
[comprehensive investigation](generalization-investigation.md) records all five
local/public results, per-target changes, paired report-group bootstrap
uncertainty, weak-label sensitivity checks and the existing factor interventions.
It separates measured results from hypotheses about endpoint/case-mix shift,
checkpoint selection and optimization under shared attention. CPU analysis reads
saved predictions only; no GPU work, inference, training or tests ran.

1B's gain over reflection comes from four targets while eight decline; its
advantage disappears on extreme-score labels but survives original-report
coverage and source-agreement restrictions. The constraint can preserve a
trained 1B solution, and sigma-40 1C ties reflection publicly. These findings do
not justify automatically restoring 1B or rejecting compartment sharing. The later
[reflection/sigma-40 result](../../validation/results/sampling/2026-09-14-reflection-sigma-40-result.md)
selects the independent-head recipe for current development.
