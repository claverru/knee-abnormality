# Label tables: lineage and what the final models trained on

**The final models trained on two silver label tables, `rebuild_state_v2` and `rebuild_state_v3`,
plus teacher-mixed soft targets; no model trained on `v3.1` or `v3.2`.** Every table is silver:
derived from the training reports (gold, the 58 official studies, overrides them where the loader
applies it). Evaluation targets stay frozen per split. The training recipe and the teacher chain
are described in the [final solution](../final-solution.md).

## Tables in order

| Table | Dates | What it is | Fate |
|---|---|---|---|
| Steven Lee `v1`, `v2`, `v4` (lixin73 dependency) | audited 2026-09-10 | Public LLM report extractions; v2 = v1 with Synovitis inferred from Effusion, v4 = 0.5 v2 + 0.5 lixin73 | Steven v4 was the first baseline label source ([decision](source-consistency-and-first-baseline-choice.md#implication-for-the-first-baseline), [audit](public-label-audit-overview.md)) |
| Tonylin soft labels | 2026-09-13 | Best candidate by reference agreement among newer public tables | Public 0.921 against Steven v4's 0.929; not adopted ([comparison](../validation/results/labels-and-losses/2026-09-13-tonylin-label-comparison.md)) |
| `combined_v1` | 2026-09-21 | Steven v4 with ACL, MCL, PF OA and Fracture replaced by a consensus mean of nine public extractors | Training and validation labels until 2026-09-28 ([combined labels](combined-labels.md)) |
| `combined_v2` | 2026-09-23 | `combined_v1` with a consensus-mean rule extended to more targets | Tested on gold, worse than `combined_v1`; never trained on ([report attributes](report-attributes.md#consensus-mean-rule-tested-on-gold)) |
| `image_blend_v1`, `oof_blend_v1`, `oof_blend_v2` | 2026-09-23, 2026-09-30, 2026-10-04 | Out-of-fold image predictions blended into the labels (self-distillation) | `image_blend_v1`: S50 public tie ([result](../validation/results/labels-and-losses/2026-09-23-image-blend-labels-result.md)); `oof_blend_v1` (ResNet-18 teachers, 0.5/0.5): gold agreement 0.9397 against 0.9075, S87 0.944 ([result](../validation/results/labels-and-losses/2026-10-04-two-family-oof-distillation.md#earlier-round-resnet-18-at-0505-2026-09-30)); `oof_blend_v2` children: S103 0.955, S106 0.957 ([result](../validation/results/labels-and-losses/2026-10-04-two-family-oof-distillation.md)); superseded by the teacher chain |
| `rebuild_mean_v1`, `rebuild_state_v1` | 2026-09-28 | Rebuild from every published report-label table: family mean (B) and anchored Dawid–Skene report states with gold calibration (A) | Trained on fold 4: B public 0.946 (S75), A 0.947 (S76) ([record](label-rebuild/training-and-a2.md)) |
| **`rebuild_state_v2`** (A2) | 2026-09-28 | A with consistent not-mentioned, below-threshold and hedged rates and the Zhen lineage fix; split `all_5fold_scanner_v2` | **Adopted 2026-09-28 (user decision).** The v2 chain trained on it: ResNet-50, ConvNeXt-Tiny and NextViT-S pretraining, gen 2, 3 and 4 (`configs/*-rebuild-*.yaml`, `resnet50.yaml`, `nextvit.yaml`) |
| **`rebuild_state_v3`** | 2026-10-19 | `rebuild_state_v2` plus report-reading rules: Claude subagents read all 4,407 reports for per-label rule questions, silver rows only (748 silver flips, 40 studies masked), gold untouched; split `all_5fold_scanner_v3` (folds of v2) | **Adopted 2026-10-19 (the final ensemble kept both v2 and v3 members).** The v3 chain trained on it: pretraining, gen 2, 3, 4, v2v3 gen 4 and the S188 students (`configs/*-v3-*.yaml`, `*-v2v3-gen4.yaml`, `*-s188-student.yaml`) |
| `rebuild_state_v3_1` | 2026-10-19 | v3 with 62 silver cells moved to 0.45 or 0.55 where the image ensemble, the reader verdict and a blind review agreed they were wrong | Not adopted; no kept config used it ([audit](audits/2026-10-19-v31-label-audit.md)) |
| `rebuild_state_v3_2` | 2026-10-19 | v3 with 81 changed silver cells in 78 studies, 35 of them ensemble-guided | Not adopted; Spearman correlation with the public scores of a fixed 26-submission panel 0.410608 against v3's 0.429769 (paired interval includes zero), gold AUC 0.904136 against 0.906330 ([v3.2 labels](audits/2026-10-19-v32-labels.md)) |

The v3 choice over v3.1 rested on board-order correlation of local silver scores over 26
re-scorable submissions: Spearman v2 0.20, v3 0.43, v3.1 0.41; v3 above v2 in 99.6% of bootstrap
draws, v3.1 minus v3 −0.017 [−0.046, +0.005]. Gold cross-fitted macro AUC moved 0.880 to 0.906
under v3, which is optimistic because some rules were written after reading gold reports. The v3.1
and v3.2 tables and their manifests are archived outside the tree (see the banner of the
[v3.2 document](audits/2026-10-19-v32-labels.md)).

## Silver labels from teachers

The teacher chain turned model predictions into training targets on top of the report-derived
silver table (the [final solution](../final-solution.md) has the full chain):

- **Gen-1 teachers** (ConvNeXt-Tiny, ResNet-50 and NextViT-S, five folds each) predicted the
  external fastMRI and OAI studies at ×4, τ 0.85 with mirror TTA, giving soft training labels for
  the pretraining of the next generation (`rebuild-gen1-…-oai-soft-tissue` for v2,
  `rebuild-v3-gen1-…` for v3; 19,265 external studies in the v3 table).
- **Teacher tables** (`kaggle_teacher_table.parquet`, from out-of-fold ×4 predictions of the previous
  generation on the Kaggle studies) are mixed with the silver labels at `teacher_share`. The v2 chain
  trained gen 2 and gen 3 (`*-rebuild-gen2/3.yaml`) at 0.75. The gen-2 optimum on the v3 gen-1
  table was 0.80 (bootstrap-averaged), and the v3 gen 2 and gen 3 configs and both gen-4 configs
  (`*-rebuild-gen4.yaml`, `*-v3-gen4.yaml`) train at 0.80. The `*-v2v3-gen4.yaml` models and the
  S188 students (`*-s188-student.yaml`) train at 1.0, with gold studies keeping gold targets.
- A student trained on its teachers' folds leaks the teachers' gold exposure, so the generations'
  gold readings were treated as optimistic and the board decided.

## Reading order for the audits

[Public report-label audit](public-label-audit-overview.md) covers the first sources and
reproduction; [label rebuild](label-rebuild/INDEX.md) the construction of v1 and v2;
[gold versus silver](gold-versus-silver.md) why training and test labels differ; and the
[dated audits](audits/INDEX.md) the sweeps and the v3 line.
