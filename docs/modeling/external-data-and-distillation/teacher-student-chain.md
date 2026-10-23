# Teacher-student chain

**Every member of the final ensemble came out of one chain per label version:** gen-1 teachers
trained on Kaggle labels alone → their predictions label the external studies → one pretraining
per family on those studies → gen-2 students fine-tuned on Kaggle with a teacher share → a mixed
table of gen-1 and gen-2 predictions → gen-3 students. The chain ran twice, on the v2 labels
(`rebuild_state_v2`) and on the v3 labels (`rebuild_state_v3`), for ConvNeXt-Tiny, the
Refined-RadImageNet ResNet-50 and NextViT-S. The final submission S188 averages the gen-2 and gen-3
students of both chains: 2 label versions × 2 generations × 3 families × 5 folds = 60 models
([final solution](../../final-solution.md)). Measured outcomes per round are in the
[teacher-student results](../../validation/results/teacher-student/INDEX.md); the external studies
are described in [external data](external-data.md).

## The chain

| Step | What | Configs | Outputs |
|---|---|---|---|
| Gen 1 (teachers) | Each family × five folds on Kaggle labels only (teacher share 0), on the shared split, one seed per run | `configs/{convnext,resnet50,nextvit}.yaml` (v2), the same with the v3 label manifest, source and split (v3) | `outputs/training/{family}-gen1-fold-{k}`, `{family}-v3-gen1-fold-{k}` |
| Kaggle teacher table T2 | Each gen-1 model predicts its own validation fold at ×4, τ 0.85, mirror TTA; the table is the logit of the mean probability of the chosen teachers | | `data/labels/prepared/rebuild-t2-x4-tau085-tta`, `rebuild-v3-t2-x4-tau085-tta` |
| External table | The same teachers predict all 19,265 fastMRI + OAI studies (×4, τ 0.85, TTA); OAI keeps only menisci and OA ([rule](external-data.md#oai)) | | `rebuild-gen1-x4-tau085-tta-oai-soft-tissue`, `rebuild-v3-gen1-x4-tau085-tta-oai-soft-tissue` |
| Pretraining | One run per family on the external studies only, nominal fold 0, ten epochs | `configs/{family}-{rebuild,v3}-pretrain.yaml` | `{family}-rebuild-pretrain`, `{family}-v3-pretrain` (their `last.ckpt` starts every later stage and is kept) |
| Gen 2 | Kaggle-only fine-tune from the pretraining, targets (1 − share) × silver + share × sigmoid(T2 logit), gold studies keep gold targets | `configs/{family}-{rebuild,v3}-gen2.yaml` | `{family}-gen2-fold-{k}`, `{family}-v3-gen2-fold-{k}` |
| Mixed table | Half the gen-1 table, half the mean of the gen-2 students' own-fold predictions (probabilities averaged) | | `rebuild-mixed-x4-tau085-tta`, `rebuild-v3-mixed-x4-tau085-tta` |
| Gen 3 | As gen 2 with the mixed table | `configs/{family}-{rebuild,v3}-gen3.yaml` | `{family}-gen3-fold-{k}`, `{family}-v3-gen3-fold-{k}` |

The rented-machine code of these steps is kept as a frozen record in
[scripts/adhoc/final_chain/](../../../scripts/adhoc/final_chain/README.md):

- **Predictions:** `predict_teacher_x4.py` for T2 and the gen-2 own-fold predictions, and
  `external_predict_rebuild.py` for the external table
  ([predictions](../../../scripts/adhoc/final_chain/README.md#3-teacher-predictions)).
- **Tables:** `build_kaggle_table.py` with `spec-{t2,mixed}-{rebuild,v3}.json` for T2 and the
  mixed tables, and `build_external_table_rebuild.py` with `teachers-{rebuild-default,v3}.list`
  for the external tables ([label tables](../../../scripts/adhoc/final_chain/README.md#4-label-tables)).
- **Trainings and predictions:** they ran as rows of the job queue (`jobs-*.tsv`,
  `dispatch/jobs.tsv`). `v3_chain.sh` built the v3 tables as soon as their inputs were done
  ([job queue](../../../scripts/adhoc/final_chain/README.md#5-training-and-the-job-queue)).

- **Teacher share:** 0.75 in the v2 chain, 0.80 in the v3 chain (gen 2 and gen 3). The share
  mixes probabilities: `target = (1 − share) × silver + share × sigmoid(teacher logit)`, with gold
  studies untouched (`distilled()` in `data/labels.py`). The share was chosen on the gold agreement
  of the label table, not on fold scores, which cannot judge it
  ([share choice](#choosing-the-teacher-share)).
- **Splits and seeds:** gen 1 uses the shared split (`all_5fold_scanner_v2`, or
  `all_5fold_scanner_v3` with the same folds for the v3 labels), so its out-of-fold gold readings
  stay clean for choosing teachers. Gen 2 and gen 3 use one new scanner-stratified split per
  family and generation, so the families do not leave out the same studies: v2 gen 2 ConvNeXt
  `all_5fold_scanner_v2_r7`, ResNet-50 `_r8`, NextViT `_r12`; v2 gen 3 `_r13`, `_r14`, `_r18`; v3
  gen 2 `all_5fold_scanner_v3_r1`, `_r2`, `_r3`; v3 gen 3 `_r4`, `_r5`, `_r6`. The v3 splits were
  picked from 300 candidates to minimise the worst fold-overlap χ² against each other and the 19
  earlier splits (worst 29.6 against 45–66 for a random pick). Every run has its own training
  seed. Split seeds and run seeds are in `outputs/logs/rebuild/rebuild-seeds.tsv` (main checkout)
  and in each run's `effective_config.json`; the committed configs carry a placeholder fold and
  seed that the job lists overrode (the lists, with each run's seed, split and start weights:
  [final-chain job queue](../../../scripts/adhoc/final_chain/README.md#5-training-and-the-job-queue)).
- **`distilled()` across splits:** the teacher table is out of fold on its own split, so a
  student on another split has validation studies its teachers trained on. `distilled()`
  therefore requires a teacher value for every training study rather than an equal fold. Local
  validation of a student is inflated by this leak, so students were judged on the public board.
- **Code at the time:** the chain ran on main's code with the uncommitted fastMRI branch changes;
  the S188 bundle records `included_git_commit` `6f78b4f` with local modifications. The kept
  pieces were committed on 2026-10-23 (`9f277d4`, `363b027`); the pieces not adopted are in the
  [retired pieces](../../history/fastmri-chain-retired-pieces.md).

## Families and teachers

- **Gen-1 teacher choice (2026-10-19):** ConvNeXt + ResNet-50 + NextViT, first on gold (0.94666,
  58 studies, ×4 τ 0.85 TTA out of fold) of all 63 tables over six families and 0.93807 on silver,
  within about 0.0002 of the best; inclusion effects silver / gold: ResNet-50 +0.0010 / +0.0022,
  NextViT +0.0009 / +0.0009, ConvNeXt +0.0001 / +0.0008, EfficientViT −0.0001 / +0.0008, FocalNet
  +0.0002 / −0.0016, MambaVision −0.0001 / −0.0006
  (`outputs/logs/rebuild/teacher-selection-full.txt`, per-label breakdown
  `outputs/logs/rebuild/ensemble-breakdown.md`). The v3 chain kept the same three families; a
  rescoring of the gen-1 teachers on the v3.1 silver labels left the choice unchanged (first on
  gold and silver).
- **Student families from the board:** with ConvNeXt + ResNet-50 + NextViT as the core (S178,
  0.965, first of six 0.965s), adding EfficientViT (S179) ranked 5th and MambaVision (S180) 7th of
  the 0.965s, and all six families (S177) last; FocalNet added to a 15-model ensemble had cost a
  step earlier (S166). MambaVision, EfficientViT and FocalNet were dropped after gen 1.
- **Recipes per family:** [ConvNeXt-Tiny](../backbones/convnext-tiny-backbone.md),
  [ResNet-50](../backbones/resnet-radimagenet-backbones.md),
  [NextViT-S](../backbones/focalnet-and-efficientvit-backbones.md#nextvit-s); the shared model is
  the [current model](../current-model.md).

## Choosing the teacher share

Gold agreement of the label table (silver mixed with the ResNet-50 + ConvNeXt teachers' out-of-fold
predictions, 58 gold studies): 0.0 0.9075, 0.25 0.9370, 0.5 0.9437, 0.65 0.9456, 0.70 0.9463,
**0.75 0.9471**, 0.80 0.9469, 0.85 0.9458, 0.90 0.9454, 1.0 0.9428. Per-label peaks ranged from
0.10 (ACL) to 1.0 (Contusion) and were not used: per-label values chosen on 58 studies overfit. On
the v3 gen-1 table with v3 cross-fitted gold values the optimum was 0.80 (0.9584; 0.75 = 0.9571;
per-draw argmax 10/50/90% 0.70/0.80/0.90, `outputs/logs/rebuild/share_curve_v3.py`); on the v3
mixed table 0.78 peaked at 0.9595 with 0.80 at 0.9593. The decisive evidence was S167: a single
ConvNeXt with share 0.75 scored 0.964 against 0.961 for the same recipe at share 0 (S160), while
its fold-0 ×4 said −0.0003.

## How students were judged

- **Fold scores cannot judge a share or a pretraining:** teachers trained on the fold being
  scored, and the share moves the model away from silver, which is what validation scores.
- **Gold out of fold (58 studies) predicted ensemble changes of 0.002–0.003** (S162, S166, each a
  full step down on the board) while silver did not; differences under about 0.0005 ended in a
  tie with an unpredictable order (S154 against S155). One fold holds only about 12 gold studies,
  and one study moves the score by about 0.01, so a single-fold gold reading decides nothing.
- **Share students' gold readings leak** through teachers that saw each fold's gold labels, and
  the leak differs by family: S171 (MambaVision + ResNet-50, gold +0.0031 over ConvNeXt alone)
  scored 0.964 below ConvNeXt × 5 (S170, 0.965). From then on share students were judged on the
  board only, and new ensembles started from the board's best and added a family only when a
  submission showed it helped.
- **Seed noise:** a two-family table moved about 0.0001–0.0002 on silver and about 0.002 on gold
  from the training seeds alone, so gold gaps below about 0.002–0.003 between tables are within
  seed noise; single-run reruns land within about ±0.0003 on silver.
- **Equal weights:** per-label or gold-fitted ensemble weights and stackers did not carry to the
  board (S164 0.963 against S163 0.964; [gold-fitted stacking](../../validation/results/validation-and-ensembles/2026-10-16-gold-fitted-stacking.md)).

## Earlier rounds

Before the final chain the line went through a fastMRI joint round with one ResNet-50 teacher
family, a second round on averaged labels, a first external round with 15 silver teachers
(ResNet-50, ConvNeXt, MambaVision) and joint fine-tuning with a fade, a share cycle on the
"teachers2" labels and a third generation on new splits. They set the decisions above; their
results are in the [teacher-student results](../../validation/results/teacher-student/INDEX.md),
and their configs are listed in the [retired pieces](../../history/fastmri-chain-retired-pieces.md#configs-of-rounds-and-families-not-in-the-final-recipe).
Their label builders (`build_silver15_labels.py`, `build_teacher_labels.py`) and share-cycle runners
(`cycle*.sh`, `cycle_configs.py`) are in the [final-chain record](../../../scripts/adhoc/final_chain/README.md#4-label-tables).

## Literature behind the out-of-fold distillation

References only; no measured gain for multi-label AUC on report-derived labels was found in them.

- The Plant Pathology 2020 first-place solution trained on five-fold out-of-fold predictions at
  0.7 truth + 0.3 prediction ([write-up](https://hamonk.github.io/2020/12/05/plant_pathology.html);
  no isolated gain given). 0.7/0.3 was the only weight with a winning precedent.
- Same-architecture children beat their parents on CIFAR (born-again networks,
  [arXiv 1805.04770](https://arxiv.org/abs/1805.04770)).
- For linear and logistic models under label noise, theory says when the child beats the parent,
  and under heavy noise the optimal teacher weight exceeds the label's (Das and Sanghavi, ICML
  2023, [arXiv 2301.13304](https://arxiv.org/abs/2301.13304)).
- A few distillation rounds reduce overfitting and more underfit (Mobahi et al., NeurIPS 2020,
  [arXiv 2002.05715](https://arxiv.org/abs/2002.05715)).
- Noisy Student: removing the student's input noise removed its gain, so children keep their
  augmentation ([arXiv 1911.04252](https://arxiv.org/abs/1911.04252)).

## After the chain

- **Gen 4 (2026-10-22):** one model per family per label version on all Kaggle data
  (`train_on_validation`, fold 0), teacher tables `rebuild-gen4-x4-tau085-tta` (v2) and
  `rebuild-v3-gen4-x4-tau085-tta` (v3) = logit of the equal mean of the gen-1, gen-2 and gen-3
  groups of that version, share 0.80, from the pretrainings
  (`configs/{family}-{rebuild,v3}-gen4.yaml`, seeds `outputs/logs/rebuild/gen4-seeds.tsv`); and a
  v2v3 gen-4 model per family with the equal mean of the six v2/v3 groups, share 1.0, v3 labels,
  from the v2 pretraining (`configs/{family}-v2v3-gen4.yaml`, seeds `gen4mix-seeds.tsv`).
- **S188 students (2026-10-21/22):** one per family from the v3 pretraining on
  `s188-x60-oof-x4-tau04-tta` (each study: the logit of the mean probability of the 12 S188
  members that held it out, re-predicted at ×4, τ 0.4, mirror TTA), share 1.0 because τ's board
  gain sits on cells where silver and gold disagree, `train_on_validation`, seed 1110484336
  (`configs/{family}-s188-student.yaml`).
- Their board results (S193, S195, S196, S197) are in the
  [final results](../../validation/results/teacher-student/2026-10-19-labels-v3-final-ensembles-and-students.md).
