# Historical token-graph sources

The snapshot files were removed from the tree on 2026-09-25. Each reference below is a
`commit:path`; read it with `git show <commit>:<path>`.

These text snapshots preserve the retired local kNN token graph, implemented on
2026-09-14 and removed on 2026-09-15 on branch `meniscus-soft-coupling`, as uncommitted
changes on parent commit `ea19772`. They are historical records, not active entry points.
The [result](../../../validation/results/pooling-and-attention/2026-09-14-token-graph-result.md), including the
[forced-training follow-up](../../../validation/results/pooling-and-attention/2026-09-14-token-graph-result.md#2026-09-15-forced-graph-training-still-unused),
and the [modeling summary](../../model-lineage.md#retired-token-graph)
own the findings.

| Original path | Source snapshot | SHA-256 |
|---|---|---|
| `src/knee_abnormality/token_graph.py` (removed) | token_graph.py.txt (`f076a09:docs/history/token-graph-sources/token_graph.py.txt`) | `955ddf06f8e904155d502c1c44e21c6a805634bf362670e3ca4f5ad77a7d4632` |
| `src/knee_abnormality/modeling/model.py` | model.py.txt (`f076a09:docs/history/token-graph-sources/model.py.txt`) | `2384e0477d06e6857f33c0e51e413e0f98a266c8598aa4b3287840df48503dd3` |
| `src/knee_abnormality/config.py` | config.py.txt (`f076a09:docs/history/token-graph-sources/config.py.txt`) | `339a684551b330696fcf486167f485e5d893fcec19f2be4f169c8dec59ea2e09` |
| `src/knee_abnormality/training.py` (removed) | training.py.txt (`f076a09:docs/history/token-graph-sources/training.py.txt`) | `11de09015a073f4ccaeb29209ca2e50ed5ac22ca7893ca5d5e6f83bf34da4cc0` |
| `src/knee_abnormality/modeling/spatial_tokens.py` | spatial_tokens.py.txt (`f076a09:docs/history/token-graph-sources/spatial_tokens.py.txt`) | `ce8ad960321c61513d7133ac5cc99ceec677cac92a4a9c55a9288d605175e84f` |
| `src/knee_abnormality/preprocessing_identity.py` (removed) | preprocessing_identity.py.txt (`f076a09:docs/history/token-graph-sources/preprocessing_identity.py.txt`) | `247895280ca37ebfa4581233803bf038790331742ccb32ac33fc1e57c65bd512` |
| `scripts/tools/diagnose_encoding_usage.py` | diagnose_encoding_usage.py.txt (`f076a09:docs/history/token-graph-sources/diagnose_encoding_usage.py.txt`) | `e0ab9e10b5f98c14b8614b7a10face9f2d83d3e0532b67813fd39f2da6ab24cf` |
| `configs/graph-forced.yaml` (removed) | graph-forced.yaml.txt (`f076a09:docs/history/token-graph-sources/graph-forced.yaml.txt`) | `9277015d8fd6441d1226307db17fa8c373eedd8001e4af48e999fdf947d4086b` |
| `token-graph-plan.md` | token-graph-plan.md.txt (`f076a09:docs/history/token-graph-sources/token-graph-plan.md.txt`) | `ec9920b20a32f9d1468ee27792f1ee9610f8cb697ef91c3f6512c81fb016ce4a` |

The source snapshots are the working tree immediately before removal, so they also
contain the later attention prior: `model.py`, `config.py`, `training.py` and
`graph-forced.yaml` differ from the `graph-forced` run archive only by the
attention-prior additions, and `token_graph.py` is identical to it. `model.py` was
restored from a full read taken immediately before editing and checked against the
edited file and that archive. The plan's opening status was updated to record the
retirement before it was moved here; its links are relative to the repository root.

To reproduce a run, use its training source archive rather than these snapshots:
`outputs/training/token-graph/source_snapshot.zip` for `token-graph` and
`outputs/training/graph-forced/source_snapshot.zip` for `graph-forced`, both predating the
attention prior. Current source, spatial-token and preprocessing identities reject
both checkpoints, so evaluate them only in an isolated checkout with the archived
sources. Diagnostic artifacts are under `outputs/analysis/e/encoding-usage-token-graph/`
and `outputs/analysis/e/encoding-usage-graph-forced/`. Run outputs, checkpoints and
analyses remain under their local `outputs/` paths and are not version controlled.
