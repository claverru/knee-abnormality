# Why baseline1 had twelve validation studies

The historical **`gold_5fold_v1`**, protocol version 1, stratified only the 58
reference studies, producing gold validation counts **12, 12, 11, 12, 11**
and gold training counts **46, 46, 47, 46, 47**. Their MCL positive counts were
**2, 1, 2, 2, 2**. All targets had both classes in those reference folds; the
58 studies had 54 distinct target combinations and 58 distinct report groups.

Its report-only pool was shared across folds (`fold == -1`), except four report
matches assigned to fold 0 and excluded from that model's training. Fold 0 used
4,345 report-labeled training studies; the other folds used 4,349. That design
was reference-only CV rather than an 80/20 split of the entire dataset, and is
superseded by option A. Its manifest hash remains:

```text
f32934f4e13d55fa53aa4245cc7ad26b6747ababe35b7c54e52ef20f07eef176
```

The current label loader and scorer accept only protocol version 2; version-1 support was
removed on 2026-09-16. Scoring a historical 58-study OOF table against
`data/folds/gold_5fold_v1` needs the archived baseline1-era sources; its membership
and artifacts are not regenerated or changed. Its 12-row baseline1 output is
only partial OOF and cannot satisfy either complete OOF endpoint.

The Lightning pipeline gathers study-level predictions over the full validation
epoch before scoring. Averaging batch AUCs is not equivalent to the official
metric.

The completed `outputs/training/baseline1` run on 2026-09-10 selected epoch 4 at
**0.8439966380070546** over the twelve fold-0 reference studies. Its saved
checkpoint predictions exactly match the selected epoch's predictions. The
[baseline review](../history/baseline1-review.md) records full run identity, per-target
metrics and limitations. Its root OOF CSV contains only twelve studies;
baseline3 has since completed the 903-study all-study fold 0, while the complete
five-fold OOF baseline remains pending. Its `last.ckpt` also contains epoch 4,
although logs extend through epoch 9, so it is not the final training state.
This historical artifact remains unchanged; current training saves a separate
latest checkpoint every epoch. The [correction verification](../history/baseline1-review.md#implemented-corrections)
also reproduced baseline1's saved predictions with fresh inference.
