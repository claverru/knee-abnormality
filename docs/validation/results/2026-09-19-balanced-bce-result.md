# Fixed class-balanced BCE result — 2026-09-19

**Completed: best fold-0 macro AUC 0.895282, up 0.008796 over the closest same-seed historical reference.** The saved
[configuration](../../../outputs/training/balanced-bce/effective_config.json) confirms
`trainer.class_balanced_bce: true`, seed 42, fold 0 of `all_5fold_v3`, Steven v4 supervision,
ReXNet-200, batch size 8 and 10 epochs. `execution.json` records no training-study limit or
resume checkpoint. All ten epoch metrics and the selected checkpoint are present.

The [supervision documentation](../../modeling/supervision.md#fixed-class-balanced-bce) owns the
fixed weighting formula and reduction. Run artifacts live under
`outputs/training/balanced-bce/`: `effective_config.json`, source snapshot and provenance,
`fold_0/input_identity.json`, `fold_0/class_balance.json`, epoch metrics and checkpoints.
Training was launched by the user, not an agent. The working YAML is currently enabled for this
experiment; the schema default remains `false`.

## Measured result

Metrics come from `fold_0/epochs/metrics_epoch_*.json`. Epochs below count completed epochs
from one; checkpoint names count from zero.

| Run | Best macro AUC | Best epoch | Last macro AUC | Ordinary BCE at best AUC |
|---|---:|---:|---:|---:|
| `balanced-bce` | 0.895281691 | 8 | 0.894498047 | 0.487386848 |
| `sigma-fraction-0867` | 0.886485919 | 6 | 0.886400544 | 0.455942468 |
| `overnight-20260918T232636Z-mt7ouybk/runs/rexnet-fold0` | 0.883670901 | 10 | 0.883670901 | 0.453629074 |

Best-to-best gains are **+0.008795772** against sigma and **+0.011610790** against the newer
ReXNet reference. The last-to-last gain against sigma is +0.008097503. Selection points to
[`best-epoch=07-val_macro_auc=0.89528.ckpt`](../../../outputs/training/balanced-bce/fold_0/best-epoch=07-val_macro_auc=0.89528.ckpt).
`selected_checkpoint.json` stores score 0.895281672 (the Lightning float32 representation).
Final-epoch ordinary BCE is 0.483042210. Probability fitting worsened while ranking improved;
ordinary BCE is not the acceptance metric for this experiment.

### Per-label AUC at each run's selected epoch

| Label | Balanced BCE | Same-seed reference | Delta |
|---|---:|---:|---:|
| ACL | 0.915904 | 0.910711 | +0.005193 |
| MCL | 0.846831 | 0.838641 | +0.008190 |
| Medial meniscus | 0.920600 | 0.921118 | −0.000518 |
| Lateral meniscus | 0.902751 | 0.885836 | +0.016915 |
| Medial OA | 0.908154 | 0.903539 | +0.004615 |
| Lateral OA | 0.867821 | 0.859991 | +0.007829 |
| Patellofemoral OA | 0.869804 | 0.863845 | +0.005960 |
| Effusion | 0.874974 | 0.874828 | +0.000146 |
| Synovitis | 0.886968 | 0.892555 | −0.005588 |
| Baker's cyst | 0.913013 | 0.891722 | +0.021291 |
| Contusion | 0.903658 | 0.876844 | +0.026814 |
| Fracture | 0.932903 | 0.918200 | +0.014703 |

Ten of twelve labels improved. This is a strong positive local result across labels, with the
largest gains in contusion, Baker's cyst, lateral meniscus and fracture. It is one run against
historical references, with the source and seed limitations below. No public result is assessed
here. `fold_0/timing.json` was absent at inspection, so runtime and peak memory are not compared.

## Existing references (checked 2026-09-19)

The closest same-seed ordinary-BCE reference is
[`sigma-fraction-0867`](../../../outputs/training/sigma-fraction-0867/effective_config.json).
A scan of 113 saved ReXNet fold-0 configurations found no other differences in its numerical
recipe after excluding run/storage fields and the newly added balancing flag. The old source
uses ordinary masked BCE. Seed 42, 10 epochs, batch size 8, sigma fraction 0.867, total 26,
label source, split, validation ×1 and optimizer settings match.

| Reference | Best macro AUC | Completed epoch (1-based) | Last-epoch AUC | Limitation |
|---|---:|---:|---:|---|
| `sigma-fraction-0867` | 0.886485919 | 6 | 0.886400544 | Same seed/settings, earlier preprocessing implementation |
| `overnight-20260918T232636Z-mt7ouybk/runs/rexnet-fold0` | 0.883670901 | 10 | 0.883670901 | Same recipe and preprocessing source files, seed 471048644 |

Both references share the candidate's training/validation-row hashes, source-label identity,
metadata identities, dependency versions and recorded pretrained-backbone initialization.
Their input-identity differences are confined to the preprocessing source digest.
The best metrics above come from `fold_0/epochs/metrics_epoch_*.json`; filenames/checkpoints use
zero-based epoch numbers, so the sigma checkpoint says `epoch=05`.

Archived sources show the same-seed reference predates shared-memory collation, support-limited
float32 conversion/resampling and the exact quantile partition optimization. These have bounded
[bit-exact evidence](../../backlog/loading-speedup/ranked-by-gain.md), but full training equivalence
was not established by this comparison. The newer `rexnet-fold0` has identical data/preprocessing
sources to `balanced-bce`; its model difference is the later inference-only `force_reflection`
argument, defaulting to false, in addition to the new training-loss plumbing. Neither is an
exact same-source, same-seed control. Use sigma as the closest historical same-seed benchmark and
the newer ReXNet run as a secondary reference, not as proof of a causal weighting gain.

`control-check` (best 0.884769195) is less comparable: it uses fixed 40 mm sampling rather than
the candidate's relative-width 0.867 rule.

## Remaining attribution limit

No exact same-source, same-seed ordinary-BCE control was found among the saved runs. A fresh
ordinary-BCE control (`trainer.class_balanced_bce: false`), seed 42, fold 0, validation ×1,
using the candidate's saved sources and all other settings would isolate the weighting change.
Preserve study exposure, soft targets, augmentations, optimizer, schedule and optimizer-step
budget. No further run was launched or scheduled during this assessment.

## Validation predictions, recovered 2026-09-19

The run finished its ten epochs and saved its checkpoint but **never wrote
`validation_predictions.csv`**: `scripts/pipeline/train.py` calls `predict_loader` at the very end,
and that function's signature had changed to take a list of models earlier the same day without the
caller being updated. The run crashed there, after paying the full training time. The caller is
fixed; a process already running keeps the old code, so any run launched before the fix hits it.

Predictions were recovered from the saved checkpoint without retraining. Because the checkpoint is
pinned to the `modeling/model.py` that existed while it trained — since reverted — the recovery ran
against the run's own `source_snapshot.zip`, which is what that archive is for.

Scored against the frozen fold targets at 0.5, next to `rexnet-fold0` from
[the same day's fold queue](2026-09-19-fold-ensembles-and-sigma-sweep.md) (same sampler, same
`study_triplet_total`, ×1, ordinary BCE, different seed):

| Label | balanced BCE | `rexnet-fold0` | Δ | positives |
|---|---:|---:|---:|---:|
| Lateral Meniscus | 0.9028 | 0.8736 | **+0.0292** | 138 |
| Baker's | 0.9130 | 0.8855 | **+0.0275** | 216 |
| Contusion | 0.9037 | 0.8833 | +0.0203 | 153 |
| MCL | 0.8468 | 0.8321 | +0.0147 | 134 |
| Fracture | 0.9329 | 0.9193 | +0.0136 | 63 |
| Lateral OA | 0.8678 | 0.8550 | +0.0128 | 241 |
| Medial OA | 0.9081 | 0.8984 | +0.0098 | 329 |
| ACL | 0.9159 | 0.9074 | +0.0085 | 184 |
| Effusion | 0.8750 | 0.8692 | +0.0057 | 525 |
| Medial Meniscus | 0.9206 | 0.9151 | +0.0055 | 356 |
| PF OA | 0.8698 | 0.8690 | +0.0008 | 406 |
| Synovitis | 0.8870 | 0.8961 | −0.0092 | 115 |
| **macro** | **0.89528** | **0.88367** | **+0.01161** | |

Eleven of twelve labels improve, and **the largest gains sit on the rarest labels**, which is the
mechanism class-balanced BCE is supposed to have. The macro gap is three times the 0.0037 seed
spread measured on fold 0. The attribution limit above still stands — this is a different seed, not
a matched control — but the size and the rare-label pattern both point the same way.

Not yet submitted: the bundle is exported and its notebook runs clean, but the day's fifth
submission had already been made
([submissions](../../inference/submissions/2026-09-19-ensemble-submissions.md)).
