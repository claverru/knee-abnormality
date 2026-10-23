# Label rebuild: training results and A2

## Training results

**B (`rebuild-mean-candidate`), fold 4, seed 703791404:** best epoch 9 (×1 0.901624 against the
frozen targets). Checkpoint selection is unaffected by the frozen targets: epoch 9 is best for both
runs against either label set. **Public 0.946 (S75), tying S73 (0.946)**, bundle `1c3e486e`, ×4,
τ 0.75, batch 4.

Cross-evaluation at ×4 (report-group bootstrap, 2,000 draws, seed 20260914):

| Metric, targets | Control | B | Δ (80% / 95%) |
|---|---:|---:|---|
| Hard macro, `combined_v1` | 0.909733 | 0.908746 | −0.0010 ([−0.0030, +0.0012] / [−0.0043, +0.0022]) |
| Hard macro, B | 0.916103 | 0.919106 | +0.0030 ([+0.0009, +0.0050] / [−0.0001, +0.0060]) |
| Soft macro, `combined_v1` | 0.7959 | 0.7913 | −0.0047 |
| Soft macro, B | 0.7800 | 0.7806 | +0.0006 |

Each model wins on its own targets: uninformative by the pre-stated rule. Against
`combined_v1`, B loses where it moved levels (Effusion −0.024, Baker's −0.012 hard) and gains
Fracture +0.013. About 3,300 silver threshold changes across twelve targets left the public score
unchanged. Training BCE is not comparable across tables: target entropy (the BCE floor) is 0.293
for `combined_v1`, 0.348 for B and 0.305 for A.

**A (`rebuild_state_v1`), fold 4:** best epoch 9 against either label set (×1 0.897726 on the
frozen targets). At ×4 it scored 0.903713 against `combined_v1`, −0.0060 (95% [−0.0093, −0.0025];
Effusion and Baker's −0.028, Medial OA −0.011), and +0.0045 (95% [+0.0013, +0.0079]) against its
own targets: again each model wins on its own targets. Rank-averaged with the control it scored
0.90950 against `combined_v1`; the control plus B scored 0.91131. **Public 0.947 (S76)** against
0.946 for S73 and S75, bundle `08d56aac`, ×4, τ 0.75, batch 4: one rounding step up, despite
−0.0060 locally against the frozen `combined_v1` targets.

## A2: common state scale and the Zhen lineage fix (2026-09-28)

A's final probability is mostly `offset + state effect + 0.84 × z`, where z is the logit of the
equal-weight family mean, so the raw values entering that mean set most of A's ranking. Two
defects there, found after S76 by the source audit and a per-code inspection:

- **Inconsistent "not mentioned", below-threshold and hedged values.** Not-mentioned cells
  entered the mean at 0.50 (Steven v1, 13,438 cells), 0.34 (Pearse, Josh), 0.28 (Pilkwang UNK),
  0.20–0.24 (Ramit, Zhen) and 0.02–0.12 elsewhere, and as missing for Vmohit. Below-threshold
  codes entered at 0.68 (Pilkwang "minimal", 6,461 cells), 0.60 (Ramit `pos_LOW`) and 0 (Vmohit
  `B`); hedged at 0.86 (Ramit) or missing (Vmohit `U`). The pooled gold positive rates by state
  from A are 0.135 not mentioned, 0.345 below threshold and 0.613 hedged.
- **Zhen v7 duplicates Pilkwang.** Its verdicts agree with Pilkwang's on 100% of 52,872 cells;
  only the minimal and UNK cells are rescored. The kappa rule missed it (rank correlation 0.976),
  so it voted as a separate family in A.

A2 (`rebuild_state_v2`, maintained tool, v1 generator snapshotted at
`outputs/analysis/l/label-rebuild-20260928/source/prepare_rebuilt_labels_v1.py`, SHA-256
`0a6226d2`) changes only these: codes whose anchor map or silent symbol (`nm`, `unk`,
`NOT_MENTIONED…`) means not mentioned, below threshold or hedged take those three rates before
averaging, negative and positive codes keep each source's value, and Zhen v7 joins the Pilkwang
family. Everything else (τ 0.15 re-selected, calibration refit, gold protocol) is unchanged. The
three rates come from A's gold fit: five pooled numbers, no new fitting on gold.

| Gold (58) | Macro AUC | Log loss |
|---|---:|---:|
| `combined_v1` | 0.9075 | 0.447 |
| A2 out-of-fold | 0.8798 (A: 0.8838) | 0.365 (A: 0.361) |
| B2 family mean | 0.9015 (B: 0.9022) | 0.373 (B: 0.373) |

Gold cannot separate A2 from A. On silver, A2 stays close to A: rank correlation 0.95–0.999 per
target and 264 threshold changes in all, mostly Medial OA (91), PF OA (57) and Lateral OA (34).
Fold 4: validation and gold training targets identical to `combined_v1`; training positives
Medial OA 760 (A 689), Lateral OA 370 (A 345). Table
`data/labels/prepared/rebuild-state-v2/labels.csv` (`5877acaf…`), manifest
`configs/labels/rebuild-state-v2-label-sources.json`, analysis
`outputs/analysis/l/label-rebuild-20260928/v2/`. The control `rebuild-state-v2-control` trained on fold 0 of `all_5fold_scanner_v2` (seed 1533697213): ×4 0.928288, ×1 0.924637 (epoch 9 of 10, 912 validation studies), submitted as S78 (public 0.944).


## Reproducing A2

Rerun on 2026-09-28 into scratch directories, the tool and the split builder reproduce the
adopted files byte for byte: `rebuild-state-v2/labels.csv` (`5877acaf…`) and
`rebuild-mean-v2/labels.csv` (`a550cd46…`) in 42 s, and all four files of
`all_5fold_scanner_v2` from `scripts.tools.validation prepare --config configs/resnet50.yaml`.

```bash
uv run --locked python -m scripts.tools.prepare_rebuilt_labels --output-dir <scratch>/prepared --manifest-dir <scratch>/manifests --analysis-dir <scratch>/analysis
uv run --locked python -m scripts.tools.validation prepare --config configs/resnet50.yaml --split-dir <scratch>/split
```

- **Public inputs:** the label manifest pins every input by SHA-256 but records Kaggle versions
  for only some. [`rebuild-public-source-downloads.json`](../../../configs/labels/rebuild-public-source-downloads.json)
  gives the dataset, version and download command for all 19 public files; the 11 unversioned
  `downloaded-DATE` files were each matched to their current Kaggle version by download and
  SHA-256 on 2026-09-28.
- **JevK5 v3 attributes:** `outputs/labels/jevk5-attributes-v3/attributes.parquet` (`4e2eb3fc…`)
