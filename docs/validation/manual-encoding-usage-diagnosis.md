# Manual encoding usage diagnosis

After training a run, run the separate command manually against the selected checkpoint:

```bash
rtk proxy uv run --locked python -m scripts.adhoc.diagnose_encoding_usage outputs/training/run --output-dir outputs/analysis/encoding-usage-run --workers 8
```

`--checkpoint` selects an explicit checkpoint; otherwise the run's fold selection
is used. The output directory must be fresh. Configuration, source/checkpoint and
input identities are checked and recorded. No training weights are changed and
no callback or training recipe launches this command. It requires the current
[spatial-token model](../modeling/current-model.md) and preprocessing identity `v9`. Earlier
checkpoints need their archived sources, including every checkpoint trained before the
2026-09-16 package restructure, the retired token graph's and those of the 2026-09-14
overnight queue, `00-baseline` among them.

Every variant reuses one batch's projected backbone features and deterministic held-out
images. It then recomputes the embedding sum, LayerNorm and MIL with the same visual token
membership. The graph variants and graph statistics of the `00-baseline` version were removed
with the graph. The six full-fold variants are:
- intact;
- without the 1D, 2D, plane or contrast embedding; and without all position
  information (1D and 2D).

The attention-prior variants were removed with the
[prior](../history/model-lineage.md#retired-attention-prior) on 2026-09-16; the
[attention-prior result](results/2026-09-15-attention-prior-result.md) keeps their measurements.

No production removal flags are added. The complete held-out fold supplies per-study
predictions, per-target and macro AUC, soft-target BCE and absolute probability
changes. Undefined target/subset AUCs are omitted as null values.

Axial-, coronal- and sagittal-only variants pool the intact tokens of
one plane by excluding other planes' tokens from attention. They cover only studies
with visually valid tokens of that plane. Their prediction files, metrics, per-target
AUCs and bootstrap intervals are paired with intact on the same studies, and study
counts are recorded. Every variant also reports compartment metrics from frozen
targets thresholded at 0.5, for the medial/lateral meniscus and OA pairs:
- **Margin AUC:** the AUC of logit(medial) − logit(lateral) for medial-only against
  lateral-only exclusive-positive studies, with a paired bootstrap interval.
- **Lateral cross-talk:** flag the top-k lateral predictions, where k is the lateral
  positive count. It is the share of flagged lateral negatives that are medial-positive,
  with counts and the medial-positive base rate among lateral negatives.

**The champion plane-only baseline was skipped** by user decision on 2026-09-14; the
pre-graph diagnostic was never run on `reflection-sigma40`. Plane-only margins exist
only for later checkpoints, so a sagittal-only gain cannot be compared with the
champion. The champion's full-fold references come from its saved predictions through
the [fold-run comparison](fold-run-comparison.md): macro AUC 0.892360, margins 0.9690
(meniscus) and 0.8791 (OA), and lateral cross-talk 68.9% (31/45) and 63.9% (53/83)
against medial-positive base rates of 36.7% and 21.6%.

Paired study-level bootstrap resampling retains all twelve targets together.
The default is 2,000 replicates with seed 20260913 and percentile 95% intervals
on variant-minus-intact differences, recording valid replicate counts. These
intervals describe this checkpoint and held-out fold, not training-seed variability.
The intact output is compared with saved validation probabilities; investigate
any disagreement before interpreting small intervention effects.

The pre-graph version replaced 3D with `unknown_3d` and measured the
[knee-side checkpoint](results/2026-09-14-knee-side-embedding-result.md#side-interventions-the-checkpoint-ignores-knee-side); its
source is snapshotted under `outputs/analysis/encoding-usage-knee-side-20260914/source/`.
Its side interventions (suppression, swapping and seeded corruption) are preserved by
the [archived patch](../history/knee-side-sources/README.md). The token-graph version
added graph and edge-type removals and per-plane edge statistics; it is preserved with
the [historical token-graph sources](../history/token-graph-sources/README.md), and its
artifacts are under `outputs/analysis/encoding-usage-token-graph/` and
`outputs/analysis/encoding-usage-graph-forced/`.

These interventions measure reliance. Embeddings and planes
overlap and interact, effects need not add, and they cannot be assigned unique
contribution percentages. Removals never made in training can create unfamiliar inputs.
Attention weights or embedding magnitudes alone are not contribution scores. Separately trained
models would be needed to establish learning benefits. The original spatial-token
decisions and method references are preserved in the
[completed plan snapshot](../history/spatial-token-plan.md.txt).
