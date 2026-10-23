# Historical knee-side sources

The snapshot files were removed from the tree on 2026-09-25. Each reference below is a
`commit:path`; read it with `git show <commit>:<path>`.

These text snapshots preserve the retired knee-side embedding (research approach 1),
implemented and removed on 2026-09-14 on branch `meniscus-soft-coupling` at parent
commit `ea19772`. They are historical records, not active entry points. The
[result](../../../validation/results/pooling-and-attention/2026-09-14-knee-side-embedding-result.md),
[side interventions](../../../validation/results/pooling-and-attention/2026-09-14-knee-side-embedding-result.md#side-interventions-the-checkpoint-ignores-knee-side),
[policy audit](../../../metadata/knee-side/knee-side-policy-audit.md) and
[modeling summary](../../../modeling/retired-side-and-compartment-variants.md#knee-side-embedding) own the findings.

| Original path | Source snapshot | SHA-256 |
|---|---|---|
| Tracked changes to `src/`, `configs/baseline.yaml` (removed), `scripts/adhoc/analyze_physical_sampling.py` and `scripts/tools/diagnose_encoding_usage.py` | integration.patch.txt (`f076a09:docs/history/knee-side-sources/integration.patch.txt`) | `f191411fe06c68e0d3df44bc91cfb082d8828ade77f8f15bc9edc5eddfb997bd` |
| `src/knee_abnormality/data/preprocessing/laterality.py` | laterality.py.txt (`f076a09:docs/history/knee-side-sources/laterality.py.txt`) | `c9df46e39d7621a4b887248fa46c46cf24348c42fa039f1e82b0b50157aeda34` |
| `scripts/adhoc/audit_knee_side_policy.py` (removed) | audit_knee_side_policy.py.txt (`f076a09:docs/history/knee-side-sources/audit_knee_side_policy.py.txt`) | `3755b7c05f3078e7c3bd5797b7e3bda5687c472236af5a1f64882ff03ed10240` |
| `knee-side-embedding-plan.md` | knee-side-embedding-plan.md.txt (`f076a09:docs/history/knee-side-sources/knee-side-embedding-plan.md.txt`) | `b91d187762b4d9fcff67333afcca7c29188c6eb08084ccf1dd1e6bb830e19eb4` |
| `scripts/adhoc/analyze_knee_side_embedding.py` (removed) | analyze_knee_side_embedding.py.txt (`f076a09:docs/history/knee-side-sources/analyze_knee_side_embedding.py.txt`) | `2737f01f10687f3a3e7f7e8837547eb18eaaa6f725751ab9cdf4b25b5d744f09` |

To reproduce, check out `ea19772` in an isolated worktree, apply the patch, and restore
`laterality.py` and the audit script. Alternatively, use `outputs/training/knee-side-embedding/source_snapshot.zip`,
which captured the training sources; the patch's diagnostic side interventions were added after that
snapshot. The two knee-side metadata audits imported the parser from `laterality.py`
while it existed; they have been restored to their original reviewed copies.
The comparison analysis left the script catalog on 2026-09-16, when the run's checkpoints
were deleted; it loads both runs' selected checkpoints, so it can no longer be rerun.
Run outputs, side manifests and analyses remain under `outputs/training/knee-side-embedding/`
and `outputs/analysis/` and are not version controlled.
