# 2026-09-12 overnight results

**`04-higher-lr` is the preferred recipe for subsequent controlled comparisons.**
Its learning rates were adopted in `configs/baseline.yaml` on 2026-09-12.
It scored **0.918** publicly, versus **0.913** for `02-longer` and **0.903**
for `05-faster-head`. All three competition submissions completed successfully;
the Kaggle API scores and saved submission records were checked on 2026-09-12.
This is development-selection evidence from three submissions, not a private
leaderboard result or a statistically established ranking.

The queue is
[`outputs/training/overnight-20260911T234416Z-8du9a4nh`](../../../outputs/training/overnight-20260911T234416Z-8du9a4nh).
All seven runs used ReXNet-200 with attention MIL, adjacent triplets, 224-pixel
inputs at 0.67 mm, center-weighted sampling, series dropout 0.2, Steven v4
labels, seed 42 and the fixed `all_5fold_v3` fold 0: 3,503 training and 904
validation studies, including eleven gold and 893 report-derived validation
studies. Training batch size was eight; validation density was 2×.
The [control configuration](../../../outputs/training/overnight-20260911T234416Z-8du9a4nh/runs/01-control/effective_config.json)
and the table below define the comparison. The launcher, `scripts/adhoc/train_overnight.py`
at `ea19772`, fixed batch size eight with no accumulation, series dropout 0.2 and
validation density 2×. Each run changed one setting from the control:

| Run | Backbone LR | Head LR | Weight decay | Epochs / patience |
|---|---:|---:|---:|---:|
| `01-control` | 0.0001 | 0.0003 | 0.01 | 10 / 10 |
| `02-longer` | 0.0001 | 0.0003 | 0.01 | 20 / 20 |
| `03-lower-lr` | 0.00005 | 0.00015 | 0.01 | 10 / 10 |
| `04-higher-lr` | 0.0002 | 0.0006 | 0.01 | 10 / 10 |
| `05-faster-head` | 0.0001 | 0.001 | 0.01 | 10 / 10 |
| `06-lower-decay` | 0.0001 | 0.0003 | 0.003 | 10 / 10 |
| `07-higher-decay` | 0.0001 | 0.0003 | 0.03 | 10 / 10 |

Run provenance records commit `14940b06344345e333386a3b2aad3645d0d5d5f9`
with a dirty working tree; each run's `source_snapshot.zip`,
`effective_config.json`, `provenance.json` and `fold_0/input_identity.json`
preserve the actual source/configuration and input identities. Input identities
match across all seven runs, including membership hash
`c709cd27ff54dd10d833abfa966622f9a5ecadef5d37670da658d81c26f7b3ed`.
Every saved prediction table contains the expected 904 unique fold-0 IDs,
matches its selected epoch's probabilities exactly, and reproduces the saved
full-fold AUC. This comparison used saved predictions, without fresh inference.

Metrics below are evaluated at the checkpoint selected by full-fold AUC.
Epochs are numbered from one; checkpoint filenames and the
[plot](../../../outputs/training/overnight-20260911T234416Z-8du9a4nh/validation_metrics.png)
number epochs from zero. Public scores are reported to the API's three decimals.

| Run | Selected epoch / budget | Full-fold AUC | Gold AUC | Soft-target BCE | Training minutes | Public AUC |
|---|---:|---:|---:|---:|---:|---:|
| `01-control` | 10 / 10 | 0.87848 | 0.91961 | 0.45870 | 35.8 | Not submitted |
| `02-longer` | 12 / 20 | 0.88354 | 0.94183 | 0.46144 | 79.2 | 0.913 |
| `03-lower-lr` | 10 / 10 | 0.86097 | 0.91634 | 0.47215 | 35.0 | Not submitted |
| `04-higher-lr` | 10 / 10 | 0.88294 | 0.91756 | 0.45404 | 37.4 | 0.918 |
| `05-faster-head` | 10 / 10 | 0.87724 | 0.93995 | 0.46091 | 36.0 | 0.903 |
| `06-lower-decay` | 10 / 10 | 0.87866 | 0.92814 | 0.45850 | 35.7 | Not submitted |
| `07-higher-decay` | 8 / 10 | 0.87944 | 0.92794 | 0.45947 | 34.7 | Not submitted |

All submissions used 2× images and inference batch size two. Exact checkpoint,
bundle, notebook and submission identities are linked in the
[submission artifact record](../../inference/submissions/2026-09-12-overnight-fold-0-submissions.md).

- Doubling both learning rates to backbone `2e-4` and head `6e-4` improved
  local full-fold AUC by 0.00446 over control and improved nine targets. Largest
  gains were Lateral Meniscus +0.02486, PF OA +0.01328 and Synovitis +0.01066.
  It also had the lowest BCE and the highest public score among submitted runs.
  Prefer its ten-epoch recipe with weight decay 0.01 for the next comparisons.
- `02-longer`'s local advantage over higher LR was only 0.00060 and reversed
  publicly by 0.005. It peaked at epoch 12, ending at 0.87990 local AUC with
  worsening late BCE despite falling training loss, consistent with overfitting.
  The twenty-epoch budget also stretches the cosine schedule; a fresh
  twelve-epoch schedule would be a different experiment. Longer training did
  not deliver the best public checkpoint here, but this does not establish
  that every longer schedule is worse.
- The eleven-study gold diagnostic ranked longer and faster-head above
  higher-LR, whereas higher-LR won publicly. Faster-head's gold AUC of 0.93995
  did not transfer to a strong public result. Keep gold AUC diagnostic rather
  than using it to override the full-fold endpoint. BCE is useful supporting
  evidence; three selected submissions do not justify replacing AUC with BCE.
- Lower LR underperformed at this epoch budget. Weight-decay changes produced
  small local gains without public evidence; they do not establish a reason
  to change weight decay from 0.01.

The previous best public submission, baseline4 at 2×, scored 0.913, so
higher-LR improved the reported best by 0.005. That historical comparison also
changes the split from `all_5fold_v1` to `all_5fold_v3` and series dropout from
0.1 to 0.2. The overnight control was not submitted, so the public gain over
baseline4 cannot be attributed solely to learning rate. Tiny local differences,
an eleven-study gold subset and repeated public selection should not be treated
as precise estimates of hidden-data generalization.
