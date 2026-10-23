# Pooling and attention results

| Read when | Document |
|---|---|
| Choosing the attention temperature τ for an ensemble, or understanding why local silver validation and the board preferred different τ: the board τ ranking of the late ensembles, the gold-scored-on-silver bug, sharpening where report and image disagree, the OAI fit, attention width matching, Contusion and ×8 | [2026-10-22 attention temperature](2026-10-22-attention-temperature.md) |
| Reviewing dense-bag pooling on the three-fold ResNet-50: the token cache, one temperature per multiplier, per-series and per-view attention share, per-study temperature rules, a temperature per label within one model and across folds | [Dense-bag pooling on the ResNet-50](2026-10-12-dense-bag-pooling-on-the-resnet-50.md) |
| Reviewing the side-routed shared-head line (routing of the meniscus and OA labels by knee side, regions, cuts and frames): build-by-build results, the router drawings, the adopted run and the run table | [Side-routed heads results](side-routed-heads/INDEX.md) |
| Reading the submissions that tested attention temperature and the evaluation image count on fixed weights | [2026-09-18 temperature and count submissions](2026-09-18-temperature-and-count-submissions.md) |
| Checking how the control behaves when it sees one plane only, which plane each label relies on, without retraining | [2026-09-27 single-plane inference](2026-09-27-single-plane-inference.md) |
| Checking the trained medial/lateral side mask and global token self-attention with 3D RoPE replacing the 1D/2D embeddings | [2026-09-23 side mask and token attention](2026-09-23-side-mask-and-token-attention.md) |
| Reviewing stride-16/stride-32 fusion (token sum, deep-branch dropout, per-scale heads): per-label AUC against `k_05`, fine/deep branch ablation and the pair-ensemble ceiling | [2026-09-22 stride-16 fusion](2026-09-22-stride16-fusion-result.md) |
| Reviewing ReXNet stride-16 truncation (14×14 tokens): per-label AUC against `k_05`, runtime, memory and parameters | [2026-09-22 stride-16 truncation](2026-09-22-stride16-truncation-result.md) |
| Checking the side-aware anatomical attention on `all_5fold_v3`: fixed bias, bias-off ablation, learned strength, learned width, why historical control B is confounded, how the 80% adoption interval was chosen, and the learned-strength adoption record | [2026-09-21 anatomical attention](2026-09-21-anatomical-attention-result.md) |
| Reviewing appearance-only pooled values (untied position path) and the repaired-DICOM confound | [2026-09-21 untied position](2026-09-21-untied-position-result.md) |
| Reviewing the shared meniscus compartment adapter | [2026-09-20 compartment adapter](2026-09-20-compartment-adapter-result.md) |
| Reviewing count-normalized attention | [2026-09-17 count-normalised attention](2026-09-17-count-normalized-attention-result.md) |
| Choosing τ for the RadImageNet ResNet-18 incumbent | [2026-09-29 RN18 attention temperature](2026-09-29-rn18-attention-temperature.md) |
| Reviewing attention temperature and bag size | [2026-09-17 attention temperature](2026-09-17-attention-temperature.md) |
| Reviewing per-type logit contributions | [2026-09-17 per-type contribution](2026-09-17-type-contribution.md) |
| Reviewing plane and contrast importance | [2026-09-17 plane/contrast importance](2026-09-17-plane-contrast-importance.md) |
| Reviewing the attention prior | [2026-09-15 attention-prior result](2026-09-15-attention-prior-result.md) |
| Reviewing the token graph | [2026-09-14 token-graph result](2026-09-14-token-graph-result.md) |
| Reviewing knee-side embeddings | [2026-09-14 knee-side embedding result](2026-09-14-knee-side-embedding-result.md) |
| Diagnosing per-plane fusion usage | [2026-09-13 per-plane fusion usage diagnosis](2026-09-13-per-plane-fusion-usage-diagnosis.md) |
| Comparing max and average pooling | [2026-09-13 max-pooling validation comparison](2026-09-13-max-pooling-validation-comparison.md) |
