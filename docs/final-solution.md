# Final solution

**The competition ended (2026-10-22 23:59 UTC) with S188 as our best public submission: an
equal-weight ensemble of 60 models, three backbone families (ConvNeXt-Tiny, Refined-RadImageNet
ResNet-50, NextViT-S) × five folds × two student generations (gen 2, gen 3) × two silver label
versions (v2, v3), read at ×4 images with mirror TTA and attention temperature τ 0.4.** It scored
**0.968 public**, first in Kaggle's order among our seven 0.968 submissions (S188 > S187 > S197 >
S196 > S192 > S189 > S194: `public_rank` 1 in `outputs/submission/kaggle-ledger.json`, written by the
[ledger](inference/submissions/ledger.md) script); S187,
the same 60 at τ 0.5, is second. Every member is a student of the same teacher-student chain:
gen-1 teachers on Kaggle labels → pseudo-labels for 19,265 fastMRI and OAI studies → pretraining on
them → Kaggle-only fine-tunes whose targets mix silver labels with teacher predictions.

Not recorded in this repository: the private-leaderboard score, the team's final public or
private rank, which two submissions were selected for the private leaderboard (the Kaggle API
does not expose selections), and the final efficiency-leaderboard row.

## The ensemble

| Group | Label table | Generation | Teacher table, share | Splits (ConvNeXt, ResNet-50, NextViT) | Runs |
|---|---|---|---|---|---|
| v2 gen 2 (S181's members) | `rebuild_state_v2` | 2 | `rebuild-t2-x4-tau085-tta`, 0.75 | `all_5fold_scanner_v2_r7`, `_r8`, `_r12` | `{family}-gen2-fold-{0..4}` |
| v2 gen 3 (S182's members) | `rebuild_state_v2` | 3 | `rebuild-mixed-x4-tau085-tta`, 0.75 | `all_5fold_scanner_v2_r13`, `_r14`, `_r18` | `{family}-gen3-fold-{0..4}` |
| v3 gen 2 (S183's members) | `rebuild_state_v3` | 2 | `rebuild-v3-t2-x4-tau085-tta`, 0.80 | `all_5fold_scanner_v3_r1`, `_r2`, `_r3` | `{family}-v3-gen2-fold-{0..4}` |
| v3 gen 3 | `rebuild_state_v3` | 3 | `rebuild-v3-mixed-x4-tau085-tta`, 0.80 | `all_5fold_scanner_v3_r4`, `_r5`, `_r6` | `{family}-v3-gen3-fold-{0..4}` |

`{family}` is `convnext`, `resnet50` or `nextvit`; each group has 15 members, one training seed per
run (`outputs/logs/rebuild/rebuild-seeds.tsv` and each run's `effective_config.json`). The S188
bundle manifest (SHA-256 prefix `9122b7bd`) lists exactly these 60 checkpoints, taken from the
S181, S182 and S183 bundles and the v3 gen-3 export. Members are averaged with equal weights; per-label
weights and stackers did not carry to the board
([retired pieces](history/fastmri-chain-retired-pieces.md#per-label-ensemble-weights),
[gold-fitted stacking](validation/results/validation-and-ensembles/2026-10-16-gold-fitted-stacking.md)).

## How the members were made

The chain is described in [teacher-student chain](modeling/external-data-and-distillation/teacher-student-chain.md)
and the external studies in [external data](modeling/external-data-and-distillation/external-data.md).
The scripts, job tables and table specs that ran it on the rented machine are kept as a frozen
record in [scripts/adhoc/final_chain/](../scripts/adhoc/final_chain/README.md). In short, per label version:

1. **Gen-1 teachers:** each family × five folds on Kaggle labels only, on the shared split.
   ConvNeXt + ResNet-50 + NextViT was the best of 63 teacher tables over six families on the 58 gold
   studies.
2. **Teacher tables:** the teachers predict their own validation folds (Kaggle table T2) and all
   external studies at ×4, τ 0.85, mirror TTA; OAI keeps only the meniscus and OA labels
   ([predictions](../scripts/adhoc/final_chain/README.md#3-teacher-predictions),
   [table builders and specs](../scripts/adhoc/final_chain/README.md#4-label-tables)).
3. **Pretraining:** one run per family on the external studies only
   (`configs/{family}-{rebuild,v3}-pretrain.yaml`; every training's seed, split and start
   weights are rows of the [job queue](../scripts/adhoc/final_chain/README.md#5-training-and-the-job-queue)).
4. **Gen 2:** Kaggle-only fine-tunes from the pretraining with targets
   (1 − share) × silver + share × teacher, gold studies untouched, each family on its own split.
5. **Gen 3:** the same with the mixed table (half gen-1 table, half gen-2 predictions).

Why each step is there, with the board evidence: pretraining +0.004 on one ConvNeXt fold (S158
0.957 → S159/S160 0.961); teacher share 0.75 +0.003 (S160 0.961 → S167 0.964); five folds (S170
0.965); gen 3 over gen 2 and the ResNet-50 beside ConvNeXt (S174 > S173 > S170, S175 3rd of the
eleven 0.965s); NextViT as third family (S178, 2nd of them), while EfficientViT, MambaVision and FocalNet each moved
the ensemble down (S179, S180, S177); one seed and split per run (S182 first of the eleven 0.965s); v2 and
v3 together with τ 0.5 and 0.4 (S187, S188). Round by round:
[teacher-student results](validation/results/teacher-student/INDEX.md).

## Model and per-family recipes

All members share one architecture, the [current model](modeling/current-model.md): spatial
tokens of every visually valid backbone location, attention pooling per label with side-routed
shared meniscus and OA heads, plane-contrast readouts, a view label bias, an anatomy map and a
soft-tissue map taught from segmenter masks, posed sagittal and coronal views and a 64 mm sagittal
fluid detail view. The family recipes differ only in backbone settings:

| Family | Base config | Backbone and weights | Input statistics | `backbone_lr` | L2-SP early / late | Other |
|---|---|---|---|---|---|---|
| [ConvNeXt-Tiny](modeling/backbones/convnext-tiny-backbone.md) | `configs/convnext.yaml` | `convnext_tiny.in12k_ft_in1k`, timm | ImageNet | 1e-4, ×0.7 per stage toward the stem | 0.01 / 0.0001 | stochastic depth 0.1 |
| [ResNet-50](modeling/backbones/resnet-radimagenet-backbones.md) | `configs/resnet50.yaml` | `resnet50.a1_in1k`, Refined-RadImageNet | 0.3162 / 0.3213 | 5e-5 | 0.01 / 0.001 | average-pool shortcuts |
| [NextViT-S](modeling/backbones/focalnet-and-efficientvit-backbones.md#nextvit-s) | `configs/nextvit.yaml` | `nextvit_small.bd_ssld_6m_in1k`, timm | ImageNet | 2e-4 | 0.001 / 0.00001, late anchor follows the rate | linear decay |

Common to all: repeated middle slice at 224 px, 120 mm frame, coronal T1/PD 84 mm and fluid 93.33 mm, sagittal T1/PD
93.33 mm and fluid 84 mm, plus a 64 mm sagittal fluid detail view, 26 training slices per study, ten epochs, batch 8, head rate 6e-4, class-balanced
BCE at exponent 0.5, anatomy and meniscus map losses at 0.1. The chain configs are the base config
with paths, label source, split, teacher table and share changed:
`configs/{family}-{rebuild,v3}-{pretrain,gen2,gen3}.yaml` for the ensemble,
`configs/{family}-{rebuild,v3,v2v3}-gen4.yaml` and `configs/{family}-s188-student.yaml` for the
all-data models after it. The committed configs carry a placeholder fold and seed; the real ones
are in each run's `effective_config.json`.

## Labels

Silver labels `rebuild_state_v2` (from 2026-09-28) and `rebuild_state_v3` (2026-10-19: report-reading
rules on silver rows, 748 flips, 40 studies masked, gold untouched); gold labels for the 58 gold
studies in both. v3 raised the gen-1 teachers' gold AUC by +0.0067 but tied v2 on the board at gen
2 (S183 one place below S181), so the final ensemble kept both. Lineage and the tables not adopted
(v3.1, v3.2): [label tables](public-labels/label-tables.md).

## Inference

| Setting | S188 |
|---|---|
| Image multiplier | ×4 |
| Attention temperature | τ 0.4 effective: the 45 members exported at τ 0.85 carry a scale of 0.4706, the 15 v3 gen-3 members exported at τ 1.0 carry 0.4 |
| TTA | mirror (study-level reflection) |
| Batch, workers | 2 per device, 8 in total |
| Backbone mode | eager (no CUDA graph; from the bundle name `v2v3-gen2gen3-x60-eager-x4-tau04eff-tta`), precision by the manifest's automatic policy (FP16 on the T4) |
| Bundle | `outputs/submission/exports/v2v3-gen2gen3-x60-eager-x4-tau04eff-tta` (manifest only locally), built 2026-10-20 from code at `6f78b4f` with local modifications |

τ divides the main attention scorer, view label bias and position prior and multiplies the shared
routing temperature. Local validation always preferred τ 0.85; the board preferred 0.4–0.5 for
every large ensemble, because sharpening moves predictions toward image findings exactly where
silver report labels and radiologist labels disagree
([attention temperature](validation/results/pooling-and-attention/2026-10-22-attention-temperature.md)).
S187's visible run peaked at 9.7 of 14.6 GiB GPU memory at about 21 s per study; the hidden rerun
took about 5 h 15 min from submission to score. How bundles, τ and the backbone mode work:
[model bundle](inference/model-bundle-and-offline-environment.md),
[inference overview](inference/inference-overview.md).

## Scores

| Submission | What | Public | Place among equals |
|---|---|---|---|
| **S188** | 60 members, ×4, τ 0.4 | **0.968** | 1st of seven 0.968s |
| S187 | the same 60, τ 0.5 | 0.968 | 2nd |
| S197 | S188's 60 + nine gen-4 models, τ 0.43 | 0.968 | 3rd |
| S185 | 15 v3 gen-2 members, τ 0.5 | 0.967 | 2nd of two 0.967s |
| S182 | 15 v2 gen-3 members, τ 0.85 | 0.965 | 1st of eleven 0.965s |
| S195 | one ResNet-50 (`resnet50-v2v3-gen4`), ×4, τ 0.85, no TTA, compiled graph | 0.964 | 2nd of thirteen 0.964s, best single model |

The full list with every probe is in `outputs/submission/kaggle-ledger.json` and the
[final results](validation/results/teacher-student/2026-10-19-labels-v3-final-ensembles-and-students.md).

## Efficiency board

The efficiency leaderboard ranks a team by its most efficient **selected** submission
([efficiency leaderboard](inference/efficiency-leaderboard.md)). Single-model students were
submitted for it: S193 (ConvNeXt student of S188, ×4, τ 0.5, compiled CUDA-graph backbone, FP16
weights) 0.960 in 544 s, and S195 (ResNet-50 v2v3 gen 4, ×4, τ 0.85, same acceleration) 0.964 in
483 s submit-to-scored. Our last recorded appearance is version 81 (rank 111, S165); which
submission held the final row is not recorded.

## Where the weights are

- **Kaggle:** each submission's private bundle dataset, named in `outputs/submission/kaggle-ledger.json`; S188's is
  `claverru/knee-v2v3-gen2gen3-x60-tau04eff` (notebook
  `claverru/s188-v2-v3-gen2-gen3-cnx-rn50-nextvit-x60-tau0-4`), S187's
  `claverru/knee-v2v3-gen2gen3-x60-tau05eff`.
- **Local, main checkout:** every member run with its checkpoint under `outputs/training/`
  (`{family}-gen2-fold-{k}`, `{family}-gen3-fold-{k}`, `{family}-v3-gen2-fold-{k}`,
  `{family}-v3-gen3-fold-{k}`), the six pretrainings (`{family}-{rebuild,v3}-pretrain`, kept with
  their `last.ckpt`, the start weights of every later stage), and the rented-machine archive
  `outputs/rented-machine-archive-20261020` (predictions, label tables, folds, export trees, run
  metadata, scripts, job queue; the chain's scripts are copied into
  [scripts/adhoc/final_chain/](../scripts/adhoc/final_chain/README.md)). The rented machine itself was destroyed on 2026-10-20; the
  fastMRI and OAI images were not copied home.

## Runs to keep

The disk cleanup keeps the selected checkpoint of every run named in this section, and the
pretrainings also keep their `last.ckpt`; the run names must stay exact. These are the local runs under `outputs/training/` that still held checkpoints when the
competition ended.

- **S188 members (generations 2 and 3, labels v2 and v3):** `convnext-gen2-fold-0`, `convnext-gen2-fold-1`, `convnext-gen2-fold-2`, `convnext-gen2-fold-3`, `convnext-gen2-fold-4`, `convnext-gen3-fold-0`, `convnext-gen3-fold-1`, `convnext-gen3-fold-2`, `convnext-gen3-fold-3`, `convnext-gen3-fold-4`, `convnext-v3-gen2-fold-0`, `convnext-v3-gen2-fold-1`, `convnext-v3-gen2-fold-2`, `convnext-v3-gen2-fold-3`, `convnext-v3-gen2-fold-4`, `convnext-v3-gen3-fold-0`, `convnext-v3-gen3-fold-1`, `convnext-v3-gen3-fold-2`, `convnext-v3-gen3-fold-3`, `convnext-v3-gen3-fold-4`, `nextvit-gen2-fold-0`, `nextvit-gen2-fold-1`, `nextvit-gen2-fold-2`, `nextvit-gen2-fold-3`, `nextvit-gen2-fold-4`, `nextvit-gen3-fold-0`, `nextvit-gen3-fold-1`, `nextvit-gen3-fold-2`, `nextvit-gen3-fold-3`, `nextvit-gen3-fold-4`, `nextvit-v3-gen2-fold-0`, `nextvit-v3-gen2-fold-1`, `nextvit-v3-gen2-fold-2`, `nextvit-v3-gen2-fold-3`, `nextvit-v3-gen2-fold-4`, `nextvit-v3-gen3-fold-0`, `nextvit-v3-gen3-fold-1`, `nextvit-v3-gen3-fold-2`, `nextvit-v3-gen3-fold-3`, `nextvit-v3-gen3-fold-4`, `resnet50-gen2-fold-0`, `resnet50-gen2-fold-1`, `resnet50-gen2-fold-2`, `resnet50-gen2-fold-3`, `resnet50-gen2-fold-4`, `resnet50-gen3-fold-0`, `resnet50-gen3-fold-1`, `resnet50-gen3-fold-2`, `resnet50-gen3-fold-3`, `resnet50-gen3-fold-4`, `resnet50-v3-gen2-fold-0`, `resnet50-v3-gen2-fold-1`, `resnet50-v3-gen2-fold-2`, `resnet50-v3-gen2-fold-3`, `resnet50-v3-gen2-fold-4`, `resnet50-v3-gen3-fold-0`, `resnet50-v3-gen3-fold-1`, `resnet50-v3-gen3-fold-2`, `resnet50-v3-gen3-fold-3`, `resnet50-v3-gen3-fold-4`.
- **Generation-1 v3 runs:** `convnext-v3-gen1-fold-0`, `convnext-v3-gen1-fold-1`, `convnext-v3-gen1-fold-2`, `convnext-v3-gen1-fold-3`, `convnext-v3-gen1-fold-4`, `nextvit-v3-gen1-fold-0`, `nextvit-v3-gen1-fold-1`, `nextvit-v3-gen1-fold-2`, `nextvit-v3-gen1-fold-3`, `nextvit-v3-gen1-fold-4`, `resnet50-v3-gen1-fold-0`, `resnet50-v3-gen1-fold-1`, `resnet50-v3-gen1-fold-2`, `resnet50-v3-gen1-fold-3`, `resnet50-v3-gen1-fold-4`.
- **Pretrainings, kept with their `last.ckpt`:** `convnext-rebuild-pretrain`, `convnext-v3-pretrain`, `nextvit-rebuild-pretrain`, `nextvit-v3-pretrain`, `resnet50-rebuild-pretrain`, `resnet50-v3-pretrain`.
- **Generation-4 runs and S188 students:** `convnext-gen4`, `convnext-s188-student`, `convnext-v2v3-gen4`, `convnext-v3-gen4`, `nextvit-gen4`, `nextvit-s188-student`, `nextvit-v2v3-gen4`, `nextvit-v3-gen4`, `resnet50-gen4`, `resnet50-s188-student`, `resnet50-v2v3-gen4`, `resnet50-v3-gen4`.

## Decisions that held to the end

- **Per-label rules allowed (2026-09-28):** the earlier no-per-label-tuning rule was lifted; under
  it per-label attention temperatures and per-label loss selection had been dropped on 2026-09-23
  ([per-label loss research](research/training-objectives/per-label-losses.md)). Per-label τ and
  per-label ensemble weights were later tried and did not carry to the board (S191, S164).
- **Research backlog review (2026-09-21):** AdamP and plane/contrast-specific feature adapters were
  accepted as small changes; terminal-1×1-only trimming was rejected; layer-wise CoAtNet
  learning-rate decay, standalone BatchNorm recalibration, ordered within-series aggregation and
  SAM were not selected. Bias-field augmentation, also not selected then, was later measured in the
  [augmentation campaign](validation/results/campaigns-and-queues/2026-09-26-augmentation-campaign.md).
- **No adoption of a generalization risk on a local fold (2026-10-13):** a change that frees
  capacity from the L2-SP anchor or removes a robustness augmentation is not adopted on local AUC
  alone.
- **Students judged on the board:** share students' local and gold readings leak through their
  teachers, so families and generations were added only when a submission showed they helped.
