# Third round: 15 silver teachers, fastMRI + OAI pretraining and the teacher share (2026-10-14 to 2026-10-17)

**Pretraining on fastMRI + OAI pseudo-labels lifted a single ConvNeXt fold from 0.957 (S158) to
0.961 (S159/S160); joint and Kaggle-only fine-tuning tied on the board; a 0.75 teacher share on
the Kaggle rows added another step (S167, 0.964) while fold scores said −0.0003; and five ConvNeXt
folds of that recipe gave 0.965 (S170), the best score until the generation and label-version
ensembles. Gold out of fold called two ensemble changes correctly (S162, S166), silver did not.**
The external data and the OAI label rule are in
[external data](../../../modeling/external-data-and-distillation/external-data.md); the decisions
this round fixed are summarised in the
[teacher-student chain](../../../modeling/external-data-and-distillation/teacher-student-chain.md).

## Teachers, labels and pretraining

- **Teachers:** five folds each of main's ResNet-50 (folds 0 to 3 on the local 3090), ConvNeXt-Tiny
  and MambaVision-T configs, Kaggle silver labels only. ×4 per fold: ConvNeXt 0.93616, 0.92928,
  0.93612, 0.93183, 0.93265; MambaVision 0.93485, 0.93030, 0.93516, 0.93021, 0.92979; ResNet-50
  fold 4 0.93314. Out of fold (×4), mean of folds: ResNet-50 0.935157, ConvNeXt 0.933210,
  MambaVision 0.932061; equal logit blend of the three 0.936992 (+0.0018 over ResNet-50, ahead on
  every fold by +0.0007 to +0.0024).
- **Labels:** the 15 teachers' probability mean at ×4, τ 0.85, mirror TTA on 9,828 fastMRI and
  9,437 OAI studies (`data/labels/prepared/silver15-avg-x4-tau085-tta`).
- **Masks:** bones on 1,946 fastMRI knees (`anatomy-mrannotator-sample550-fastmri1966-joints`),
  soft tissue on 1,819 OAI DESS knees with Kaggle's 263 (`soft-tissue-nnunet-knee-dess265-oai`).
- **Pretraining** (`configs/{family}-silver15-pretrain.yaml`, external only, fold 0 nominal,
  inflated through the teachers): best ×1 / ×4 ConvNeXt 0.93458 / 0.93636, MambaVision 0.93474 /
  0.93684, ResNet-50 0.93239 / 0.93511. Training-time meniscus overlap 0.66 to 0.67 against 0.59
  for a Kaggle-only ConvNeXt. Persistent DataLoader workers of three concurrent runs over the
  19k-study index filled the 251 GB host and killed one run at epoch 2; the runs resumed from
  their last epoch with `trainer.persistent_workers: false`.
- **Fade fix:** the quarter-cosine weight measured progress against `estimated_stepping_batches`,
  which counts both batches of a pair, so it would have ended at 0.71; it now uses the optimizer
  step count (deployed before the first joint run).
- **S158, `convnext-silver-fold-0` (no pretraining), ×4, τ 0.85, no TTA: 0.957 public**, first of
  three at 0.957 (S158 > S106 > S119). Baseline of the fine-tuning probe on ConvNeXt fold 0.
- **S159, `convnext-silver15-joint-fold-0` (pretraining, then joint with the fade), ×4, τ 0.85, no
  TTA: 0.961 public**, +0.004 over S158 on the same fold; fourth of five at 0.961 (S150 > S140 >
  S141 > S159 > S149). Local ×4 0.945443 against the teacher's 0.936161 (inflated through the
  teachers). Joint ×4 against the fold's silver teacher so far: ConvNeXt fold 1 0.939754 (+0.0105),
  ResNet-50 fold 0 0.944509 (+0.0054).

## The fine-tuning grid on ConvNeXt fold 0

Local validation scores against silver labels, so it cannot judge choices that move the model
toward or away from silver (teacher share above all); only the public leaderboard can. Budget:
five submissions a day (reset 00:00 UTC), one kept every day for a big ensemble, so at most four
probes a day, each with about 0.001 of rounding and test noise.

Fold 0, ×4, τ 0.85, no TTA; local ×4 with pretraining is inflated through the teachers.

| Cell | Pretraining | Fine-tuning | Kaggle targets | Status | Local ×4 | Public |
|---|---|---|---|---|---|---|
| C1 ConvNeXt | No | Kaggle only | Silver | submitted, S158 | 0.936161 | 0.957 |
| C2 ConvNeXt | No | Kaggle only | Silver + teacher share | not trained | | |
| C3 ConvNeXt | No | Joint + fade | Silver | not trained | | |
| C4 ConvNeXt | No | Joint + fade | Silver + teacher share | not trained | | |
| C5 ConvNeXt | Yes | Kaggle only | Silver | submitted, S160 (trained on the local 3090, 4.7 min per epoch) | 0.945279 | 0.961, above S159 in the tie |
| C6 ConvNeXt | Yes | Kaggle only | Silver + teacher share | trained 2026-10-16 on the local 3090: S160 + share 0.75 of `teachers2-kaggle-x4-tau085-tta` (run `convnext-silver15-pretrain-share075-finetune-fold-0`); submitted, S167 | 0.944995 | **0.964, first in score-sorted order, above S163** |
| C7 ConvNeXt | Yes | Joint + fade | Silver | submitted, S159 | 0.945443 | 0.961 |
| C8 ConvNeXt | Yes | Joint + fade | Silver + teacher share | trained 2026-10-17 on the rented machine (`convnext-silver15-joint-share075-fold-0`, S159 config + share 0.75), submitted, S168 | 0.944662 | 0.964, below S167 in the tie (5th of eight 0.964s) |
| R1 ResNet-50 | No | Kaggle only | Silver | trained (`resnet50-silver-fold0`) | 0.939145 | |
| R3 ResNet-50 | No | Joint + fade | Silver | not trained | | |
| R7 ResNet-50 | Yes | Joint + fade | Silver | trained (`resnet50-silver15-joint-fold-0`) | 0.944509 | |

R2, R4, R5, R6 and R8 mirror the ConvNeXt cells and are not trained. MambaVision fold 0: silver
teacher ×4 0.93485, joint 0.946309, neither submitted. The teacher share is a dial (0.25, 0.5,
0.75, …), not a switch, so the grid cannot be covered.


Questions the grid was built for, with what was measured at the time (the outcomes follow in
[how they were settled](#how-they-were-settled)):

1. **Pretraining:** yes for ConvNeXt (C7 against C1, +0.004 public).
2. **Joint + fade or Kaggle-only fine-tuning after pretraining:** C5 against C7: both 0.961 public
   (S160 above S159 among six at 0.961); locally ×4 0.945279 against 0.945443, logit correlation
   0.992. Per label they split: silver-only minus joint is +0.0027 on bone + fluid (Fracture,
   Contusion, Effusion, Synovitis) and −0.0019 on menisci + OA, contrast +0.0046 (study bootstrap
   95% [+0.0010, +0.0092]). Equal blend +0.0004 over C7, routing joint on menisci/OA and silver-only
   elsewhere +0.0006 (chosen on the same fold). Hypothesis: OAI is an osteoarthritis cohort with
   noisy injury pseudo-labels, amplified by Kaggle-prevalence class weights; a per-label external
   weight could give both in one training.
3. **Teacher share on the Kaggle rows** (soft targets from the 3-family out-of-fold mean of the
   15 silver teachers): undecided at the time; the student-distillation campaign's 0.25 was chosen on silver
   validation and S156's 0.964 confounds it with full data and other teachers. One point far
   from zero (for example 0.5) on the current best path, against it; default 0 (S159).
4. **ResNet-50 pretraining:** pretraining + joint (R7) or joint alone from RadImageNet (R3); it
   gains less from the external stage (+0.005 against +0.008 to +0.012).
5. **ResNet-50 as a student in the final ensemble:** good teacher (best single family out of fold,
   +0.0015 in pairs) but weakest student; out-of-fold ensembles of the 15 joint models with and
   without it, then the ensemble slots (10 ConvNeXt + MambaVision joint models one day, all 15
   the next).
6. **Final form:** fold models or full-data students, and the ensemble's members and weights.
7. **Per-label ensemble weights (specialists):** on the 15 silver teachers' out-of-fold predictions,
   weights fitted per label on four folds and scored on the fifth gain +0.0004 macro over equal
   weights (per fold −0.0003, +0.0002, +0.0003, +0.0004, +0.0012), against +0.0001 for one global
   weight set. Specialists (weights on all folds, ResNet-50 / ConvNeXt / MambaVision): ResNet-50 for
   Lateral OA (0.8/0.2/0.0), Fracture (0.8/0.0/0.2), Medial OA (0.7/0.2/0.1) and PF OA; MambaVision
   for Synovitis (0.3/0.1/0.6); ConvNeXt for MCL (0.2/0.6/0.2). Judged on silver; repeat on the joint
   models and confirm through an ensemble slot.

8. **OAI labels only where they carry signal (agreed 2026-10-15, joint recipe of the following night):** per
   source and label, the 15-teacher pseudo-labels show OAI (an osteoarthritis cohort) has almost no
   injuries (≥ 0.5 on 0.1% of studies for Fracture and MCL, 0.2% Contusion, 2.4% ACL) and
   uninformative fluid labels (confident, < 0.1 or > 0.9, on 0.1% for Effusion and 0% for
   Synovitis; Baker's 13% against 55% on fastMRI), while menisci and OA have rates close to
   Kaggle's. Recipe: fastMRI keeps all twelve labels; OAI keeps only Medial Meniscus, Lateral
   Meniscus, Medial OA, Lateral OA and PF OA (Baker's dropped with the fluid labels); OAI masks
   supervise the maps as before. Implementation: empty values for the dropped labels on the OAI
   rows of the external label table, and the external dataset marks empty labels unobserved (the
   masked loss already skips them); joint stage only, from the existing pretrainings.
   The existing pretrainings stay: C5 fine-tuned from the same pretraining was the better model on
   bone and fluid, so a Kaggle-only stage removes whatever the pretraining took from OAI, and
   re-pretraining three families plus redoing every joint run would cost most of a day. OAI's
   injury labels are mostly correct easy negatives; the fluid labels are the uninformative ones.
   Any future pretraining (for example a final round with new teachers) uses the same per-label
   rule from the start, more so if that night's joint run showed bone and fluid recovering while
   menisci and OA keep their gain.

## Ensembles of the 15 joint models

- **15 joint models (2026-10-15 evening):** ×4 per fold ConvNeXt 0.945443, 0.939754, 0.944991,
  0.939364, 0.940230; MambaVision 0.946309, 0.937811, 0.946394, 0.939305, then fold 4; ResNet-50
  0.944509, 0.938544, 0.942666, 0.937400, then fold 4 (all inflated through the teachers). Out of
  fold, mean of folds: ConvNeXt 0.941956, MambaVision 0.941629, ResNet-50 0.940112, all fifteen at
  equal weights 0.943208, per-label family weights pulled half-way to equal 0.943400 (fitted on
  four folds, scored on the fifth; weights in
  `outputs/submission/label-weights/joint15-per-label-halfway.json`). Submitted as S161 (equal); S162
  (per-label) was cancelled by the user before scoring (bundle kept at
  `outputs/submission/exports/joint15-3fam-perlabel-x4-tau085-tta`); S161 is ×4, τ 0.85, mirror TTA, exported on the rented machine from this worktree's
  code (the per-label weights are an inference-only addition: `export.py --label-weights`,
  `models/label_weights.json`, a weighted per-label mean in the engine).
- **S161 = 0.964 public**, a tie with S154–S156. Against the old round (Kaggle out of fold, silver
  / gold): teachers 0.935120 / 0.938638 old against 0.936719 / 0.940859 new; students S153/S154
  0.940316 / 0.947643 against S161's 0.942929 / 0.948511. The new students gain on every label on
  silver but lose Fracture (−0.0167) and Baker's (−0.0091) on gold, the labels whose OAI
  pseudo-labels are unreliable. The next submission (user decision) was S161's exact configuration with the
  OAI-soft-tissue joint models (S163).
- **S162, the 10 MambaVision + ConvNeXt joint models (no ResNet-50), same settings: 0.963 public**,
  one step below S161 (order: S154 > S156 > S161 > S155 at 0.964, then S152 > S162 at 0.963). Silver
  out of fold predicted +0.0001 for dropping ResNet-50; gold out of fold predicted −0.0021 (0.946379
  against 0.948511), the direction the board took.
- **OAI labels only for menisci and OA, 15 joint models (2026-10-16 night):** ×4 per fold within
  ±0.0007 of S161's members except MambaVision fold 2 (−0.0020). Out of fold, 15 equal: silver
  0.943143 against 0.943196, gold 0.949377 against 0.948511 (+0.0009, study bootstrap 95%
  [−0.0024, +0.0040]); on gold the injury labels recover (MCL +0.0091, Fracture +0.0042, Contusion
  +0.0040, Baker's +0.0018, PF OA +0.0064), Lateral OA falls (−0.0116, 11 positives); MambaVision
  alone rises on gold from 0.940918 to 0.944501. Submitted in S161's configuration as the first
  submission of 2026-10-16.
- **S163, the 15 OAI-soft-tissue joint models in S161's configuration: 0.964 public**, our top
  submission inside the tie (S163 > S154 > S156 > S161 > S155, all 0.964), above S161 as gold out
  of fold predicted.
- **S164, S163's members with per-label family weights chosen for the best gold out-of-fold AUC
  (in-sample gold +0.0046, silver −0.0007): 0.963 public**, one step below S163 (0.964); second of
  three at 0.963 (S152 > S164 > S162). Gold-tuned ensemble weights did not carry to the board; keep
  equal weights unless a weighting also holds on held-out silver.
- **Stacking on S163's members (session knee-abnormality-e9, gold only, leave-pair-out per
  label):** equal weights 0.949380; logistic regression on the label's 3 family logits 0.948893
  (−0.0005); TabPFN v2 on them 0.945299 (−0.0041); TabPFN on all 36 logits 0.934439; logistic
  regression on all 36 0.909443. Pooled leave-one-out AUC was biased against every stacker.
  Decision basis: equal weights for the final ensembles, no gold-fitted weights or stackers.
- **Kaggle-row teacher share against gold (same session; label table, not a student result):**
  gold agreement of the label table with ResNet-50 + ConvNeXt teachers' out-of-fold predictions:
  silver 0.9075, share 0.25 0.9379, 0.5 0.9422, 0.75 0.9473, teacher alone 0.9416 (0.75 picked
  among three on the same 58 studies, slightly optimistic). A per-label gold-fitted correction
  failed (0.9417, leave-pair-out). Bears on question 3: a teacher share of about 0.75.
- **C6, teacher share 0.75 after the pretraining (2026-10-16):** new Kaggle teacher table
  `teachers2-kaggle-x4-tau085-tta` (the 10 ResNet-50 + ConvNeXt silver teachers on their own validation
  folds, ×4, τ 0.85, mirror TTA, mean probability as a logit; each teacher run with its archived source
  because the teachers span three preprocessing identities). Logit r against the τ 1.0 out-of-fold mean
  0.9935 to 0.9988 per label, probabilities slightly higher (up to +0.024 on Contusion); gold agreement of
  the label table at share 0.75 0.9471 (τ 1.0: 0.9473). C6 = S160 + share 0.75 (only change; S160's
  archived source, seed 1110484336, 47 min, peak 18.9 GiB): ×1 0.94300 against S160's 0.94181, ×4
  0.944995 against S160 0.945279 (−0.0003) and S159 0.945443 (−0.0004). Per label against S160: Lateral
  Meniscus +0.0079, MCL +0.0053, Medial OA +0.0016, Baker's +0.0013, ACL +0.0007, Medial Meniscus
  −0.0013, PF OA −0.0018, Contusion −0.0018, Effusion −0.0020, Fracture −0.0030, Lateral OA −0.0035,
  Synovitis −0.0069. Fold 0's 12 gold studies: 0.9628, equal to S160 (S159 0.9643). Silver validation
  cannot judge a share (it moves the model away from silver); the board decides.

## Workflow from 2026-10-15 (user decision)

1. Local: search new architectures, kept only once shown to be good teachers.
2. Teachers: evaluate on silver and gold out-of-fold predictions (58 gold studies), per family and
   per combination, and build pseudo-labels only from the best teachers.
3. Students: train them on those labels and keep the best (ensemble contribution out of fold, then
   the leaderboard).
4. Recipe: pretraining or not, joint or not, teacher share on Kaggle rows, and the rest of the
   questions above.

Teachers and students are judged separately. First reading on the silver teachers (out-of-fold
macro, silver / gold): ResNet-50 0.934632 / 0.938411, ConvNeXt 0.932376 / 0.937444, MambaVision
0.931554 / 0.928401 (gold 95% about ±0.02); ResNet-50 + ConvNeXt 0.936153 / 0.941863, all three
0.936719 / 0.940859. MambaVision is the weakest teacher on gold and the most valuable joint student
(+0.0010 to the other two). FocalNet's five silver folds and their fastMRI and OAI predictions are
queued on the rented machine for the next teacher reading.

## Share students on the board

- **S167 = C6 = 0.964 public, first in score-sorted order (above S163, the 15-model ensemble).**
  A single ConvNeXt fold-0 model, ×4, τ 0.85, no TTA: S160's recipe (0.961) with only the Kaggle
  targets changed to 0.25 silver + 0.75 teacher (ResNet-50 + ConvNeXt out of fold, ×4, τ 0.85,
  mirror TTA). +0.003 public for one change, while fold-0 ×4 said −0.0003 (validation inflated
  through the teachers): the local fold score cannot judge a teacher share; the label table's gold
  agreement (0.9075 → 0.9471) predicted it. Bears on questions 3 and 6: the share goes into every
  student, and an ensemble of share-0.75 students is the next candidate.
- **S166 = 0.963 public** (S163's 15 + the 5 round-2 FocalNet joint students, 20 equal members, ×4,
  τ 0.85, mirror TTA; shared CUDA-graph pool after a private-pool OOM on the T4): −0.001 against S163,
  first among the 0.963s. Out of fold it predicted silver +0.0003, gold −0.0026: the gold reading
  called the direction a second time (after S162), silver did not. Ensemble members are chosen by the
  gold out-of-fold reading (within 0.001 of the best silver).
- **When gold is trustworthy:** the readings that called S162 and S166 used all 58 gold studies,
  out of fold over five folds. One fold holds only about 12 gold studies, and one study moves the score by
  about 0.01, so a single-fold gold reading decides nothing (C6 tied S160 there at 0.9628 and still gained
  +0.003 public). For a teacher-share student, even the 58-study reading is somewhat inflated: the
  teachers that labelled its training rows trained on the fold that holds those gold studies. The
  label table's gold agreement (58 studies, 0.9075 → 0.9471) is what predicted S167.
- **How large a gold difference counts (S154 against S155):** adding the five ResNet-50s to the five
  ConvNeXt models read silver −0.0004, gold −0.0003 out of fold (S155 0.940901 / 0.947541, S154
  0.940478 / 0.947291), and the board tied at 0.964 with S154 sorted first. Gold differences of about
  0.002 to 0.003 (S162, S166) were followed by a full step on the board both times; differences under
  about 0.0005 end in a tie, with the score-sorted order not predictable from them.
- **S168 = C8 = 0.964 public, below S167 (C6) in score-sorted order** (S167 > S163 > S154 > S165 >
  S168 > S156 > S161 > S155). Same settings as S167 (single ConvNeXt fold 0, ×4, τ 0.85, no TTA); the
  only difference is the joint stage. Once the Kaggle rows carry the 0.75 teacher share, the joint stage
  adds nothing on the board, so the cheaper Kaggle-only fine-tune (C6) is the recipe of the next cycle:
  five folds of every family from its round-2 pretraining with share 0.75, ConvNeXt first.
- **S169 (round-2 ensemble):** round 1's ResNet-50 + round 2's ConvNeXt and MambaVision, 15 equal
  members, picked by gold out of fold within 0.001 of the best silver (silver 0.942741, gold 0.949980
  against S163's 0.943175 / 0.949380, a tie-sized +0.0006).
- **S169 = 0.964 public, first in score-sorted order** (S169 > S167 > S163 > S154 > S165 > S168 > S156 >
  S161 > S155): round 1's ResNet-50 + round 2's ConvNeXt and MambaVision, 15 equal members, ×4, τ 0.85,
  mirror TTA, same bundled code as S163. A tie with S163 sorted above it, as the tie-sized gold reading
  (+0.0006) suggested. The round-2 labels (ResNet-50 + ConvNeXt teachers, OAI rule from the pretraining)
  are at least as good as round 1's for ConvNeXt and MambaVision.
- **S170 = 0.965 public, new best** (the only 0.965): the five ConvNeXt folds of the share cycle
  (`convnext-share075-finetune-fold-0..4`: round-2 pretraining, Kaggle-only fine-tune with teacher share
  0.75), equal, ×4, τ 0.85, mirror TTA. One family of five share students beats every 0.964, including
  the 15-member ensembles S169 and S163 and the single share model S167: the share carries over to the
  folds of an ensemble.

## How they were settled

1. **Pretraining:** yes for every family of the final chain.
2. **Joint or Kaggle-only fine-tuning:** Kaggle-only; once the share was in, the joint stage added
   nothing (S168 below S167), and the joint mode was retired.
3. **Teacher share:** 0.75 (S167), later 0.80 on the v3 labels.
4. **ResNet-50 pretraining:** the final chain pretrains every family, the ResNet-50 included.
5. **ResNet-50 as a student:** it stayed in every final ensemble (S173 above S170, S175 above
   S174).
6. **Final form:** fold models, 60 of them (S188); all-data students went to the efficiency
   submissions.
7. **Per-label weights:** dropped; equal weights everywhere (S164).
8. **OAI labels only for menisci and OA:** adopted for every later external table.

## Local runs to keep (main checkout `outputs/training`)

Pretrainings, including their `last.ckpt` (start weights of the joint runs): convnext-silver15-pretrain,
convnext-silver15-pretrain-resumed, mambavision-silver15-pretrain, mambavision-silver15-pretrain-resumed,
resnet50-silver15-pretrain, resnet50-silver15-pretrain-resumed, convnext-teachers2-pretrain,
focalnet-teachers2-pretrain, mambavision-teachers2-pretrain.

Silver teachers: resnet50-silver-fold0, resnet50-silver-fold1, resnet50-silver-fold2, resnet50-silver-fold3,
rn50-silver-fold-4, convnext-silver-fold-0, convnext-silver-fold-1, convnext-silver-only-imagenet-fold-2,
convnext-silver-fold-3, convnext-silver-fold-4, mamba-silver-fold-0, mamba-silver-fold-1, mamba-silver-fold-2,
mamba-silver-fold-3, mamba-silver-fold-4, focalnet-silver-fold-0, focalnet-silver-fold-1, focalnet-silver-fold-2,
focalnet-silver-fold-3, focalnet-silver-fold-4, effvit-b2-best-folds-0-1-3-4, efficientvit-sd02-fold-2.

Students: convnext-silver15-joint-oai-soft-fold-0 to 4, mambavision-silver15-joint-oai-soft-fold-0 to 4,
resnet50-silver15-joint-oai-soft-fold-0 to 4 (S163), convnext-silver15-joint-fold-0 to 4,
mambavision-silver15-joint-fold-0 to 4, resnet50-silver15-joint-fold-0 to 4 (S161),
resnet50-teachers2-joint-fold-0 to 4, focalnet-teachers2-joint-fold-0 to 4 and the later teachers2 joint runs.
