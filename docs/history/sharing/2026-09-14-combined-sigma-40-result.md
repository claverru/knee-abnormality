# 2026-09-14 combined sigma-40 result

Retain sigma 40 alone over the combined candidate on current local evidence.
`outputs/training/sigma40-higher-lr-shared-normalization` combines sigma 40 with
backbone/head LRs 0.0003/0.0009 and channel mean/std 0.449/0.226. All other
non-path settings match `02-central-sampling` in the completed overnight queue;
archived package Python files and complete input identities match exactly.
Both use 1C, Steven v4, seed 42, fold 0, reflection 0.5, 224 pixels, 140 mm,
2× validation images, ten epochs and 4,380 optimizer steps.

| Metric | Sigma 40 | Combined | Combined minus sigma 40 |
|---|---:|---:|---:|
| Full-fold AUC, 904 studies | 0.892538 | 0.891035 | -0.001503 |
| Soft-target BCE | 0.448244 | 0.448816 | +0.000572 |
| Gold AUC, 11 studies | 0.917229 | 0.927216 | +0.009987 |

Both select epoch 10 (index 9). Combined training took 1,738.80 seconds and
peaked at 20,681,703,936 CUDA bytes, effectively identical to sigma 40's
1,739.16 seconds and identical peak allocation. Three targets improve and nine
decline. Largest gains are lateral meniscus (+0.007634), ACL (+0.001857) and
Baker's (+0.000962); largest losses are fracture (-0.008003), contusion
(-0.006388) and MCL (-0.005951). The individual hyperparameter gains did not
combine into a better result in this run; this single-seed comparison cannot
separate interaction effects from training variability. The small gold subset
does not override the full-fold decline. The candidate has not been submitted.

Evidence is the two runs' saved `effective_config.json`, `source_snapshot.zip`,
`fold_0/input_identity.json`, `selected_checkpoint.json`, epoch metrics and
`timing.json`. The combined selected checkpoint is
`fold_0/best-epoch=09-val_macro_auc=0.89104.ckpt`. No new training, inference or
tests ran for this comparison. Sigma 40's completed public score is 0.931,
tying reflection at the reported precision.
