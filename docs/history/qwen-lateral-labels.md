# Qwen Lateral OA label pilot

**Closed 2026-09-28, not adopted: implemented in the `qwen-lateral-labels` worktree on 2026-09-27;
the Qwen candidate tied the control and was not submitted.** The worktree and its branch were
removed on 2026-09-28 after its work reached `main`.
The worktree is `/home/claudio/src/knee-abnormality-qwen-lateral-labels`, based on
`d6c2f32`. Main kept `combined_v1` during the pilot and has used `rebuild_state_v2` since
2026-09-28 ([label rebuild](../public-labels/label-rebuild.md)). This follows the
[public-label investigation](../public-labels/2026-09-27-public-label-investigation.md), with no image-model training started.

## Implementation

The fixed recipe is `qwen_lateral_v1`: on silver studies with a Qwen value,
`Lateral OA = (combined_v1 + Qwen3-8B) / 2`. The other eleven targets, all gold source
rows and the one missing silver Qwen value retain `combined_v1`. The existing loader applies
official gold overrides to training and takes validation targets from the frozen
`all_5fold_scanner_v1` split. There is no new training flag or blend-weight option.

[Preparation command](../../scripts/tools/prepare_lateral_oa_blend_labels.py) takes a pinned
public-source input manifest (name, label table, column, file hashes) and validates hashes,
unique IDs, official coverage, probabilities, unchanged other targets and unchanged gold rows.
It refuses existing output paths and records input/output hashes and target changes. On Sep 28
it replaced the Qwen-only preparer; rerun on the Qwen input it reproduces the same `labels.csv`
hash below.
The [Qwen input manifest](../../configs/labels/qwen-label-input.json) pins dataset version 1,
updated Sep 26, with CSV SHA-256
`2806eb6bf8504236cab05c8e4a5d05f7065a9a537898f60f702f84d8df41e5a9`.
The [prepared manifest](../../configs/labels/qwen-lateral-label-sources.json) selects
`data/labels/prepared/qwen-lateral-v1/labels.csv`, SHA-256
`3136a893a698eeb8583747e9c8beaf8ab144ad4e8261e8c05f54a07dbc84caef`.
Datasets remain ignored by Git. Both `data` and `outputs` are symlinked to main's directories.
Qwen uses separate snapshot/prepared directories and the `qwen-lateral-candidate` run directory.
The 34 local investigation and preflight artifacts were preserved in shared outputs on Sep 28.

## Prepared targets and data audit

All 4,407 studies are present: 58 gold and 4,349 silver. The pinned Qwen CSV covers 4,406.
The missing silver study is
`1.2.826.0.1.3680043.8.498.88620225126852321191690360547272730103`.

- 4,337 silver Lateral OA probabilities change; 181 cross the 0.5 threshold.
- Silver positive rate changes 27.3856% → 26.1669%; mean absolute change is 0.084360.
- Actual prepared gold source rows retain the baseline predictions. The investigation's gold
  AUC 0.910439 evaluates a hypothetical blend on gold before overrides; it is not the raw
  prepared table's gold score and does not measure image-model performance.
- CPU loading of both configurations on all five folds found identical frozen validation
  targets, identical gold training targets and identical other training targets.

The selected comparison uses **fold 4, seed 703791404**, inherited from main at worktree creation.
On its 3,538 training studies, Lateral OA positives change 962 → 921. Positive class weight
changes 1.394999 → 1.430362 (+2.53%); negative weight changes 0.852489 → 0.848543 (−0.46%).
The exponent remains 0.5. The normal training run records its full class-balance and label
provenance. A fixed-original-weight sensitivity is not implemented or scheduled.

Local evidence: [preparation report](../../data/labels/prepared/qwen-lateral-v1/preparation.json)
and [five-fold data audit](../../outputs/analysis/qwen-lateral-v1/preflight.json).
Ruff and maintained-script help/index checks passed; no tests were written or run.

## Preparation and execution

Dependencies are synced in the worktree's own environment. Prepared labels and the manifest
already exist. To reproduce them into fresh paths:

```bash
rtk proxy uv run --locked python -m scripts.tools.prepare_lateral_oa_blend_labels --input-manifest configs/labels/qwen-label-input.json --output-dir data/labels/prepared/qwen-lateral-reproduced --output-manifest outputs/analysis/qwen-lateral-reproduced-manifest.json
```

For a clean machine, first use the version-pinned download command in the Qwen input manifest
and provide the existing combined labels and frozen split. Keep datasets out of Git.

The control config, `configs/qwen-lateral-control.yaml` (removed 2026-09-29; in git history),
snapshotted the main recipe of the time, and the candidate, then `configs/training.yaml`, differed only in experiment name, training-label
manifest and source. Both use the same fold, seed, model, augmentation, ten-epoch budget and
×4 prediction setting. Output directories are shared with main through the `outputs` symlink.

The completed `outputs/training/scanner-confusion-control` run is the matched control:
fold 4, seed 703791404, ×4 local AUC 0.909733. It was submitted as S73 on Sep 28,
scoring 0.946 public at ×4 and τ 0.75. Its training configuration, all 77 saved source files
compared, label/split hashes and report-attribute manifest match this worktree's control.
Reuse this run. The user launched the candidate on Sep 28 at 09:33 CEST as
`outputs/training/qwen-lateral-candidate` (same commit and source digest `952e0d5d` as the
control), without the pre-training report review below:

```bash
rtk proxy nohup uv run --locked python -u -m scripts.pipeline.train configs/training.yaml --run-id qwen-lateral-candidate > outputs/training/qwen-lateral-candidate.log 2>&1 < /dev/null &
```

Match S73's ×4 and τ 0.75 for the Kaggle comparison.
This is a manual controlled comparison: existing autoresearch gates require `combined_v1`,
and were not relaxed for this label experiment. Do not regenerate folds or evaluation targets.
Use the normal comparison/submission procedure and existing adoption rule; do not select
another blend weight, source gate or checkpoint criterion on the result.

## Bhavesh comparison labels

Prepared on Sep 28 with the same fixed rule, as `bhavesh_lateral_v1`: half
[Bhavesh LLM labels](https://www.kaggle.com/datasets/bhaveshkolhe/knee-llm-labels/versions/2)
Lateral OA, half `combined_v1`, on silver only. The
[input manifest](../../configs/labels/bhavesh-label-input.json) pins dataset version 2 (updated
Sep 24), two shards whose SHA-256 match the investigation's copies. Bhavesh covers all 4,407
studies. The [prepared manifest](../../configs/labels/bhavesh-lateral-label-sources.json)
selects `data/labels/prepared/bhavesh-lateral-v1/labels.csv`, SHA-256
`f5d3f54b16db30960634126fc456ea6779b6f712cd74eba672db827f1dc86a9b`.

- 4,327 silver Lateral OA probabilities change; 159 cross 0.5; mean absolute change 0.086848.
- Silver positive rate 27.3856% → 25.8910%.
- The hypothetical gold blend reproduces the investigation's Lateral OA AUC 0.832689 → 0.911025.
- On fold 4, training Lateral OA positives change 962 → 916 (Qwen 921). Every other training
  target column and every validation target column are identical to the control's.

**Not planned to train (user decision, 2026-09-28):** a one-label change moves macro AUC less
than run noise. The table is kept; the twelve-target [label rebuild](../public-labels/label-rebuild.md) supersedes
this line. For reference, the launch command would be:

```bash
rtk proxy nohup uv run --locked python -u -m scripts.pipeline.train configs/training.yaml --name bhavesh-lateral-labels --paths.label-manifest configs/labels/bhavesh-lateral-label-sources.json --labels.source bhavesh_lateral_v1 --run-id bhavesh-lateral-candidate > outputs/training/bhavesh-lateral-candidate.log 2>&1 < /dev/null &
```

## Result: inconclusive, not submitted

`qwen-lateral-candidate` (best epoch 9, ×1 0.902772) scored ×4 macro **0.909021** against
0.909733 for `scanner-confusion-control`: **−0.0007**, 95% report-group interval
[−0.0032, +0.0021], 852 groups, 2,000 draws. Soft macro 0.79607 against 0.79595. It fails the
adoption rule. Lateral OA moved −0.0058, scored against the frozen `combined_v1` targets it was
trained away from, while unchanged targets moved by up to ±0.006 (Synovitis +0.0059, Contusion
−0.0047): a single-target label change sits inside run-to-run noise, so a one-label pilot cannot
be judged locally or on the three-decimal leaderboard. It was not submitted, and the Bhavesh
single-target run is not planned; label work continues as the all-target
[label rebuild](../public-labels/label-rebuild.md). Comparison:
`outputs/analysis/qwen-lateral-v1/compare-x4/comparison.json` (config differences allowed:
name, label source/manifest, report-attribute manifest path; input identity differences: label
manifest, source, training/validation row hashes, the latter only through the informational
source column).
