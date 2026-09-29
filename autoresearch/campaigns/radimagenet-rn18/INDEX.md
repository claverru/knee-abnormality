# RadImageNet ResNet-18 campaign

**Closed on 2026-09-29 by topic rotation after 23 decided runs.** It adopted no series dropout
(+0.0023) and 36 training triplets (+0.0018). The committed recipe
`configs/radimagenet-resnet18.yaml` takes only the first; 36 triplets wait for the
[endgame](../../../docs/backlog/endgame.md#test-time-augmentation).

| Read when | Document |
|---|---|
| Checking the objective, budget, fixed and permitted mechanisms and evidence requirements | [Contract](contract.md) |
| Checking the runs, curves, drift and noise the seeds build on | [Prior work](prior-work.md) |
| Checking the sealed recipe, reference, budget, scope and role instructions | [Campaign definition](campaign.yaml) |
| Reading the regularization and per-layer seeds (series dropout, DSU, layer-wise decay, linear probe then fine-tune, L2-SP) | [Seeds 1](seed-ideas.yaml) |
| Reading the input, freezing, rate and token seeds (mean-initialized adjacent slices, frozen stem, 36 triplets, backbone LR 1e-4, stride-16 tokens) | [Seeds 2](seed-ideas-2.yaml) |
| Reading the schedule, ResNet-10t batch and plane-backbone seeds | [Seeds 3](seed-ideas-3.yaml) |
| Reading the two user seeds outside the topic (asymmetric focal loss, 101-bin head) | [Seeds 4](seed-ideas-4.yaml) |
| Checking results | [Findings](findings.md) |

## Setup

- **Replaces [`radimagenet`](../radimagenet/INDEX.md)** (user, 2026-09-28), whose reference was the
  ReXNet control, so a single change had to clear the whole 0.003 gap. Its first two seeds (14
  epochs, two warmup epochs) are prior work here; the other 13 seeds carry over in the same order.
- **Reference:** `radimagenet-resnet18` (×4 0.925476), with the `sources_sha256` identity exception
  below. The ReXNet control (0.928288) is reported with every result as the target.
- **Budget:** range 0.5–1.5 of the reference's 1,085 s, the only budget rule.

## Reference parity

The reference was trained before the `data/splits.py` move changed a hashed preprocessing source,
so its recorded `sources_sha256` (`043683b4…`) differs from the current tree's (`02ef3566…`).
[`check_reference_source_parity.py`](../../../scripts/adhoc/check_reference_source_parity.py) ran its
selected checkpoint on CPU over the first 16 validation studies with its own `source_snapshot.zip`
and with the current tree: the 16 × 12 predictions are bit-identical (2026-09-28).

The reference was also launched from the `radimagenet` worktree, so its recorded
`paths.label_manifest` and `paths.report_attribute_manifest` point into that checkout. The files are
byte-identical to main's (SHA-256 `08339ec5…` and `f712c551…`), so both paths are declared
`reference_config_exceptions`.
