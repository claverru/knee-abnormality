# Submission ledger

All 27 Kaggle competition submissions, newest first, with submission times in UTC from `kaggle competitions submissions` (2026-09-16, plus the five 2026-09-17 submissions). The 19 dated records in this folder cover 25 of them (original 1C, the September 12 overnight queue and the baseline3 artifact each cover several); the baseline4 2× and 3× density comparisons are recorded in [Inference image count and memory](../inference-image-count-and-memory.md).

| Submitted (UTC) | Submission | Public AUC | ID | Record |
|---|---|---:|---:|---|
| 2026-09-18 06:28 | `ranked-filtered-seed42`, attention temperature 0.75 in the weights, 4x images, batch 2 | **0.939** | – | [Temperature and count](2026-09-18-temperature-and-count-submissions.md) |
| 2026-09-18 06:22 | `07-series-share-1.0` (`repeated_series_share: 1.0`), default inference | 0.929 | – | [Temperature and count](2026-09-18-temperature-and-count-submissions.md) |
| 2026-09-18 06:11 | `ranked-filtered-seed42`, attention temperature 0.75 in the weights, 3x images, batch 2 (submitted twice) | 0.938 | 56322551 | [Temperature and count](2026-09-18-temperature-and-count-submissions.md) |
| 2026-09-17 11:40 | Ranked-filtered seed 42 (series filtering and deterministic source ranking, fold-0 0.889573); 13m 16s launch to score | 0.933 | 56303522 | [Ranked-filtered](2026-09-17-ranked-filtered-submission.md) |
| 2026-09-17 07:58 | Seed-44 control (`02-control-seed-44`, fold-0 0.891627) | 0.931 | 56299154 | [Seed controls](2026-09-17-seed-control-submissions.md) |
| 2026-09-17 07:58 | Seed-43 control (`01-control-seed-43`, fold-0 0.889837) | 0.931 | 56299151 | [Seed controls](2026-09-17-seed-control-submissions.md) |
| 2026-09-17 07:32 | Seed-42 control (`00-control`, fold-0 0.892995), the EMA run's live weights; 12m 16s launch to score | 0.928 | 56298600 | [Seed controls](2026-09-17-seed-control-submissions.md) |
| 2026-09-17 07:17 | EMA 0.999 (`03-ema-0.999`, fold-0 0.893155), fast runtime; 13m 47s launch to score | 0.929 | 56298280 | [EMA](2026-09-17-ema-submission.md) |
| 2026-09-15 21:28 | Reflection/sigma-40 champion with 16 loader workers and automatic FP16 backbone chunks; 10m 05s submission to score | 0.936 | 56262847 | [Fast champion](2026-09-15-fast-champion-submission.md) |
| 2026-09-15 08:27 | Champion with every 3D embedding replaced by `unknown_3d` at inference | 0.936 | 56249900 | [Champion without 3D](2026-09-15-champion-without-3d-at-inference.md) |
| 2026-09-15 07:58 | Overnight `00-baseline`: graph off, no 3D, trained on current code | 0.934 | 56249358 | [Overnight 00-baseline](2026-09-15-overnight-00-baseline-submission.md) |
| 2026-09-14 12:16 | Reflection/sigma-40 champion, two GPUs, four studies per GPU, four total workers | **0.936** | 56230573 | [Reflection sigma-40](2026-09-14-reflection-sigma-40-submission.md) |
| 2026-09-14 08:25 | 1A shared meniscus classifier, dual-GPU retry after the worker request-loading failure | 0.930 | 56226273 | [1A with two GPU processes](2026-09-14-1a-submission-with-two-gpu-processes.md) |
| 2026-09-14 07:37 | Original 1C disease-compartment attention, batch-4 retry | 0.928 | 56225269 | [Original 1C comparison](2026-09-14-original-1c-comparison-submission.md) |
| 2026-09-14 07:24 | Original 1C disease-compartment attention, batch 8; hidden rerun failed | no score | 56224993 | [Original 1C comparison](2026-09-14-original-1c-comparison-submission.md) |
| 2026-09-14 07:15 | 1C sigma-40 overnight candidate `02-central-sampling` | 0.931 | 56224822 | [Sigma-40 overnight](2026-09-14-sigma-40-overnight-submission.md) |
| 2026-09-13 21:23 | 1B shared meniscus and OA classifiers, with the later bundle audit | 0.925 | 56215113 | [Meniscus and OA sharing](2026-09-13-meniscus-and-oa-sharing-submission.md) |
| 2026-09-13 17:27 | Spatial tokens with study reflection (sigma 80) | 0.931 | 56211972 | [Reflection](2026-09-13-reflection-submission.md) |
| 2026-09-13 15:31 | 3D physical sampling (`3dsampling`) | 0.930 | 56210103 | [3dsampling](2026-09-13-3dsampling-submission.md) |
| 2026-09-13 13:57 | Tonylin soft labels v3 | 0.921 | 56208424 | [Tonylin](2026-09-13-tonylin-submission.md) |
| 2026-09-13 11:36 | Spatial tokens with five embeddings, after the notebook-name retry | 0.929 | 56206027 | [Spatial token](2026-09-13-spatial-token-submission.md) |
| 2026-09-12 17:38 | Regional spatial fusion, 27 queries, with the failed runtime gate | 0.920 | 56190442 | [Spatial-fusion runtime gate](2026-09-12-spatial-fusion-runtime-gate-and-submission.md) |
| 2026-09-12 09:37 | Frozen `crop140mm` control | 0.925 | 56182948 | [140-mm crop](2026-09-12-140-mm-crop-submission.md) |
| 2026-09-12 08:27 | Overnight `05-faster-head` | 0.903 | 56181919 | [Overnight fold-0](2026-09-12-overnight-fold-0-submissions.md) |
| 2026-09-12 08:21 | Overnight `04-higher-lr`, the queue winner | 0.918 | 56181821 | [Overnight fold-0](2026-09-12-overnight-fold-0-submissions.md) |
| 2026-09-12 08:20 | Overnight `02-longer` | 0.913 | 56181804 | [Overnight fold-0](2026-09-12-overnight-fold-0-submissions.md) |
| 2026-09-11 16:52 | Baseline4 ReXNet-200, 3× images, batch 2 | 0.912 | 56169445 | [Image count and memory](../inference-image-count-and-memory.md) |
| 2026-09-11 16:51 | Baseline4 ReXNet-200, 2× images, batch 2 | 0.913 | 56169438 | [Image count and memory](../inference-image-count-and-memory.md) |
| 2026-09-11 14:59 | Baseline4 ReXNet-200, 1× images, batch 8 | 0.910 | 56167669 | [Baseline3 fold-0 artifact](2026-09-11-baseline3-fold-0-submission-artifact.md), [image count](../inference-image-count-and-memory.md) |
| 2026-09-11 11:46 | Baseline3, bundle parity and codec wheels | 0.878 | 56164711 | [Baseline3 fold-0 artifact](2026-09-11-baseline3-fold-0-submission-artifact.md) |
