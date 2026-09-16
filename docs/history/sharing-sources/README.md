# Historical sharing sources

These exact text snapshots preserve the 1C implementation and investigations
from branch `1C` at `3b8c2cc1c7f7fbfe76ef2b149c91dba0c354002f`. They are historical records,
not active entry points in the script catalog. Current training retains
independent classifiers and attention, with sigma 40 and optional CoAtNet support.
The [experiment results](../sharing/INDEX.md),
[generalization investigation](../sharing/generalization-investigation.md) and
[disagreement audit](../sharing/oa-and-meniscus-disagreement-audit.md) own the findings.

| Original path | Source snapshot | SHA-256 |
|---|---|---|
| `scripts/adhoc/analyze_sharing_generalization.py` | [analyze_sharing_generalization.py.txt](analyze_sharing_generalization.py.txt) | `e3d9878c8126e730e858b8478674baf7d76b2313376e3920195ddf6b6490cee6` |
| `scripts/adhoc/analyze_sharing_case_mix.py` | [analyze_sharing_case_mix.py.txt](analyze_sharing_case_mix.py.txt) | `63819a3cb2ba6048000256db759eceb9328958df56f0b82f3a45fd3fb7d73aea` |
| `scripts/adhoc/audit_sharing_disagreements.py` | [audit_sharing_disagreements.py.txt](audit_sharing_disagreements.py.txt) | `7eb07d3981245cd8bbdbd495c0024a0de4dc7348c1cd6e9fe8affa13fe37697b` |
| `scripts/adhoc/diagnose_compartment_attention.py` | [diagnose_compartment_attention.py.txt](diagnose_compartment_attention.py.txt) | `5e069680454e83da7a2d21328c92a60fd266dbdc6ac3404d9c62ac048f2bd2ca` |
| `scripts/adhoc/train_overnight.py` | [train_overnight.py.txt](train_overnight.py.txt) | `93de1e15229c8d1e258c83540eb8d10bf252948e5706b13cb639e0d8413f2c78` |
| `src/knee_abnormality/model.py` | [model.py.txt](model.py.txt) | `b12b03574c17ca42dffa2accc8f107add1f90d4c379bb8f1ed63b8b5ea87d57d` |
| `src/knee_abnormality/preprocessing_identity.py` | [preprocessing_identity.py.txt](preprocessing_identity.py.txt) | `2eb9ec9b4442ff9f435f6c2696f9d2699bbe00bdcf765c326ecb5c838de5e7a7` |

Reproduce historical commands in an isolated checkout of the pinned revision
or the matching run/analysis source archive. Copying individual files into
current `main` is insufficient: source/config identities and the 1A, 1B and
1C checkpoint architectures differ. The compartment diagnostic additionally
needs each model's own archived runtime and its documented architecture argument.
The eleven-run overnight definitions here differ from the seven-run launcher
still present on `main`; configs alone do not select an architecture or freeze
the source imported by successive queue jobs.

Completed runs retain `effective_config.json`, `source_snapshot.zip`, input
identities, checkpoints and predictions under their original `outputs/` paths.
Analysis artifacts and locked report reviews also remain local and are not
included in version control. The sources above preserve the procedures, not
a replacement for those data or a complete standalone environment.

The retired root disease-sharing plan is superseded by the historical
specification in [transfer research](../../research/medial-lateral-transfer.md#historical-implementation-and-reproduction).
It is not restored as active work.
