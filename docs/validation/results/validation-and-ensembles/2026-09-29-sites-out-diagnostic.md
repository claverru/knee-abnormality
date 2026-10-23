# 2026-09-29 sites-out diagnostic

**An unseen site cost the recipe of 2026-09-29 about 0.009–0.010 macro AUC, replicated by a local run.** A ResNet-18 trained without
five inferred training sites scored 0.0101 lower than the incumbent on those sites' fold-0
studies, but only 0.0009 lower on the seen-site studies, although it trained on 23% less data.
The site cost is −0.0092 (80% [−0.0144, −0.0039], 95% [−0.0172, −0.0008]) on hard AUC and
−0.0059 on soft AUC. That lies between the thresholds set before the run: 0.005 to shelve the
line, 0.02 to make site generalization the main hedge. One run, with a report-group bootstrap
only: moderate evidence that the cost is real, weak on its size.
Why the run was made: [sites and test composition](../../../metadata/sites-and-test-composition.md).

## Setup

- **Held-out sites:** the five single-language clusters (Bulgarian, German, Greek, Dutch,
  French; 1,036 studies, 224 in fold 0, 10 gold).
- **Split:** `all_5fold_scanner_v2_sites_out`, which is `all_5fold_scanner_v2` without those
  studies. It trains on 2,683 studies (the incumbent on 3,495) and selects the checkpoint on the
  688 seen-site fold-0 studies.
- **Recipe:** `configs/resnet50.yaml` (RadImageNet ResNet-18, seed 1533697213), with only the
  split changed.
- **Run:** `sites-out-rn18-20260929`, on one Kaggle T4 in `16-mixed` with cuDNN off.
  - Timings: training with per-epoch validation 9,506 s, ×4 fold-0 prediction 332 s, ×4
    held-out prediction 737 s, 10,671 s in all.
  - Selected epoch 9, ×1 0.92567, ×4 0.92976 on the seen-site fold 0.
- **Reference:** `rn18-no-series-dropout`, the incumbent: fold 0 at ×4, 3090 in bf16, trained
  from the same code.
- **Artifacts:** `outputs/kaggle-training/sites-out-rn18-20260929-output/` and
  `outputs/analysis/s/sites-out-20260929/score.json`. Launch records:
  `outputs/kaggle-training/sites-out-rn18-20260929.json`.
- **Code:** branch `sites-out` (worktree `../knee-abnormality-sites-out`): the Kaggle-training
  hardware options, `scripts/adhoc/build_sites_out_split.py` (removed), `launch_kaggle_sites_out.py`,
  `predict_held_out_sites.py` and `score_sites_out.py`. Kaggle: dataset
  `claverru/knee-sites-out-training` v3, series index `claverru/knee-series-index` v1, notebook
  `claverru/knee-sites-out-train` v3.

```bash
uv run --locked python -m scripts.adhoc.score_sites_out --run-dir outputs/kaggle-training/sites-out-rn18-20260929-output/outputs/sites-out-rn18-20260929 --output outputs/analysis/sites-out-new/score.json
```

## Results

×4 macro AUC against the `rebuild_state_v2` evaluation targets, on the same fold-0 studies for
both models. The paired report-group bootstrap uses 2,000 draws.

| Studies | Sites-out run | Incumbent | Δ | 80% | 95% |
|---|---:|---:|---:|---|---|
| Held-out sites (224) | 0.9087 | 0.9188 | −0.0101 | [−0.0148, −0.0051] | [−0.0177, −0.0022] |
| Seen sites (688) | 0.9298 | 0.9307 | −0.0009 | [−0.0028, +0.0010] | [−0.0037, +0.0020] |
| **Site cost** (first Δ minus second) | | | **−0.0092** | [−0.0144, −0.0039] | [−0.0172, −0.0008] |
| Site cost, soft AUC | | | −0.0059 | [−0.0088, −0.0031] | [−0.0105, −0.0016] |

On all 1,036 held-out studies the run scores 0.8984. By site: Bulgarian 0.9108 (220), German
0.9021 (262), Dutch 0.9258 (152), Greek 0.9005 (321), French 0.9356 (81). There is no
seen-site counterpart for studies outside fold 0, so these are levels, not costs.

## Local replication

The campaign control `sites-out-control` is the same recipe and split trained locally (3090,
bf16, cuDNN on; epoch 7 selected, ×1 0.92517, training 916 s). Its held-out prediction took
51 s. Scored the same way (`outputs/analysis/s/sites-out-control-20260929/score.json`):

| | Kaggle run | Local control |
|---|---:|---:|
| Held-out Δ (224) | −0.0101 | −0.0115 [−0.0167, −0.0061] |
| Seen Δ (688) | −0.0009 | −0.0016 [−0.0036, +0.0005] |
| **Site cost, hard** | **−0.0092** | **−0.0099** [−0.0156, −0.0044] |
| Site cost, soft | −0.0059 | −0.0041 [−0.0072, −0.0013] |
| ×4 on all 1,036 held-out studies | 0.8984 | 0.8996 |
| ×4 on the 688 seen fold-0 studies | 0.9298 | 0.9291 |

The site cost replicates across two independent runs (seed shared, hardware and precision
different). The campaign's primary metric, held-out ×4 on 1,036 studies, differs by 0.0012
between them, which is a first yardstick for its run-to-run noise. Per-site AUCs move more,
up to 0.008 for the Bulgarian cluster.

## Reading

- **Losing 23% of the training data cost almost nothing on seen sites** (−0.0009, inside
  noise). Nearly all of the held-out loss is the site being unseen.
- **It replaces the leave-one-scanner-out diagnostic** once proposed in the big-bets backlog.
  An unverified competitor README claimed a random split inflates AUC by about 0.05 through
  scanner memorization; holding out whole sites costs about 0.01 here.
- **The bootstrap covers evaluation noise only.** Training-seed noise for macro AUC on a full
  fold is about 0.0017. On 224 studies it is likely larger, and it enters the held-out Δ
  directly, so the true interval is wider than shown.
- **Part of the cost may be reporting conventions, not images.** Held-out sites' labels come from
  their own reports, whose conventions the model never learned. The test labels are
  image-derived, so that part would not apply on the test. The 10 held-out gold studies are
  too few to separate the two.
- **What it means for the leaderboard:** a test share *s* of studies from unseen sites would
  cost roughly *s* × 0.009 on those labels, for example about 0.002 if a fifth of the private
  set came from new sites. That is well beyond the 0.001 resolution, and it is a hedge only
  against a scenario that remains unconfirmed. Competitors without site generalization would lose
  the same.
- **Decision (user, 2026-09-29):** run the
  [site-generalization campaign](../../../../autoresearch/campaigns/site-generalization/INDEX.md).
  It found the cost to be mostly between-site label-rate offsets, and led to the
  [L2-SP adoption](2026-09-30-rn18-l2sp-no-confusion.md).
