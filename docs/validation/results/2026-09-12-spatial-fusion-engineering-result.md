# 2026-09-12 spatial-fusion engineering result

Historical subsequent decision: restore [27-query, width-64 eager fusion](../../modeling/historical-regional-fusion.md)
for training, later replaced by spatial-token MIL. The rejection below records the earlier
engineering comparison and is superseded as an implementation-selection
decision. Its measurements and untrained status remain historical facts.

**Reject this candidate and retain the midpoint `crop140mm` control.**
The implementation reached the plan's engineering stop condition: warm
end-to-end inference exceeded the 5% overhead limit at the submitted batch
size two. No full fold-0 training, accuracy comparison, geometry-free ablation
or Kaggle submission was started. This is a runtime rejection, not an accuracy
result or evidence against spatial fusion generally.

The initial implementation used one ReXNet feature pass, unchanged global
256-dimensional tokens, 27 patient-space queries, width 64, dense local
cross-attention with finite geometric bias, and the existing target-specific
MIL. Profiling led to compiled local tensor operations, fixed triplet packing,
factored geometry projection, nine queries (eight ±35-mm corners plus the
center), width 32, and removal of redundant indexing and host synchronization.
Every valid local observation remained available; no spatial shortlist was used.
The rejected implementation and audit utility were archived together and removed
from production. The pre-existing shared geometry foundation remains unchanged.

Final paired profile on the RTX 3090, BF16, float32 matmul policy `high`, seed 42,
32 fold-0 training studies, 2× images, batch two and 16 persistent workers:

| Measurement | Submitted bundle control | Final fusion candidate | Change |
|---|---:|---:|---:|
| Fixed-input mean batch inference, 16 warm repetitions | 62.53 ms | 65.30 ms | +4.44% |
| Complete 32-study warm loader pass, mean of three passes | 1.366 s | 1.468 s | **+7.47%** |
| Warm loader-pass range | 1.353–1.383 s | 1.448–1.482 s | Non-overlapping in this sample |
| Peak allocated inference memory | 1,633,286,656 bytes | 1,609,677,824 bytes | −1.45% |

The final input manifests match every selected triplet, prepared pixel hash,
position and input count in the submitted bundle. Training-mode manifests also
matched the archived training control on the earlier 16-study profile. Labels
and split provenance are saved with each measurement. The original 27-query
candidate passed the preliminary batch-eight training checks: 350.08 versus
361.73 ms per step (+3.33%) and 19.496 versus 19.618 GB peak allocation (+0.63%).
Those training measurements do not describe the final compiled candidate;
its failed inference gate stopped further training work.

The [decision artifact](../../../outputs/analysis/spatial-fusion-20260912/decision.json)
records the limits, ratios and exact profile directories. Each profile retains
source hashes, selected-input manifests, all timing samples and an operator
trace. The final candidate is `candidate-inference-b2-fused`; its reference is
`submitted-inference-b2-high`, executing the actual submitted bundle. Earlier
batch-eight profiles used the training snapshot and are explicitly preliminary.
The [source archive](../../../outputs/analysis/spatial-fusion-20260912/source_snapshot.zip)
contains the complete candidate, audit utility and executed plan; SHA-256
`8c5b2c04f841246ebc04fa6402cc70940db7df77027f4d2a9f561135b811a1d7`.
It is also extracted under that directory's `candidate-source/`. Reproduction
uses the archived `src` on `PYTHONPATH`, the saved `candidate.json`, and the
[archived profiler source](../../history/regional-sources/profile_spatial_fusion.py.txt).

The [image audit](../../data-preparation/spatial-token-geometry-and-mapping-audit.md#spatial-fusion-mapping-audit) supports
coordinate bookkeeping on a bounded real sample, not complete registration
validation. Initial finite forward/backward and deterministic-validation checks
passed. Final trained-checkpoint/export behavior and attention attribution were
not evaluated because no training run was warranted. No tests were written or run.
