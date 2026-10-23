# 2026-09-13 spatial-token validation comparison

The completed `spatial-tokens-five-embeddings` run selected epoch 10 (checkpoint
index 9), with **0.88950565 macro AUC** and **0.45062525 soft-target BCE** on
the same 904 validation studies as `crop140mm`. Both metrics are the run's best
at the final epoch. This is a promising improvement on the development fold;
the paired uncertainty interval does not establish a repeatable advantage.

Kaggle submission **56206027** completed with **0.929 public AUC**, improving
on `crop140mm` at 0.925 by 0.004. It was later surpassed by
[3dsampling](2026-09-13-3dsampling-validation-comparison.md) at 0.930.
The public improvement agrees with the local direction; it does not isolate
individual architectural contributions. Exact artifact and API identities live
in the [submission record](../../../inference/submissions/early-submission-details.md).

| Selected checkpoint | Epoch | Macro AUC | Soft-target BCE | Training seconds |
|---|---:|---:|---:|---:|
| `crop140mm` | 8 | 0.886122 | 0.452961 | 2,130.23 |
| `spatial-fusion` | 10 | 0.884169 | 0.453061 | 1,855.21 |
| `spatial-fusion-max` | 10 | 0.879078 | 0.456062 | 1,665.07 |
| `spatial-tokens-five-embeddings` | 10 | **0.889506** | **0.450625** | 1,628.20 |

Relative to `crop140mm`, macro AUC increases by 0.00338400 and BCE decreases
by 0.00233603. Eight targets improve and four decline. Largest gains are
Baker's (+0.01357), MCL (+0.00915), Contusion (+0.00831) and ACL (+0.00760).
Declines are Effusion (−0.00502), Medial OA (−0.00275), Fracture (−0.00262)
and Synovitis (−0.00149). Gold AUC is 0.93112, but this subset has only eleven
studies and does not resolve small differences.

The saved-prediction analysis uses 2,000 paired study-level bootstrap replicates,
seed 20260913, keeping all twelve targets together. Candidate-minus-control
percentile 95% intervals are [−0.000249, +0.006887] for macro AUC and
[−0.005022, +0.000551] for BCE; all 2,000 replicates are defined. Both intervals
include zero. They condition on the selected checkpoints and this held-out
fold, excluding training-seed variability and checkpoint-selection uncertainty.

Validation AUC reaches 0.888696 at epoch 7 and 0.889506 at epoch 10, a late gain
of 0.000810. The last epochs improve slightly without an obvious late validation
collapse; this is not evidence that a longer training budget would improve it.
Peak allocated CUDA memory is 20,089,207,808 bytes, close to the control's
20,017,470,464 bytes. Full training is 23.6% faster, but strict determinism differs
(`false` here, `true` in the control) and the older control predates native-reader
changes. These full-run timings do not isolate architecture or inference cost.

Seed 42, the 3,503/904 training/validation membership, `all_5fold_v3`, Steven v4
labels, ReXNet-200 initialization, 224-pixel/140-mm inputs, triplet budgets,
2× validation density, optimizer and ten-epoch budget are unchanged. Saved
label/split/membership/series/training/validation identities match. Architecture
and execution changes prevent attributing the difference to 3D coordinates or
any one embedding. The manual encoding diagnostic below has not been run.

The comparison artifact (`outputs/analysis/s/spatial-tokens-vs-crop140mm-20260913/comparison.json`, deleted)
pins prediction, label and run-artifact identities, all per-target results,
bootstrap settings and analysis-source identity. The
learning curves and target deltas (`outputs/analysis/s/spatial-tokens-vs-crop140mm-20260913/validation-comparison.png`, deleted)
are also available as an
SVG (`outputs/analysis/s/spatial-tokens-vs-crop140mm-20260913/validation-comparison.svg`, deleted).
This analysis reads saved CSVs and metrics on CPU; no model execution, tests,
new training, encoding diagnostic or blend submission was performed.
