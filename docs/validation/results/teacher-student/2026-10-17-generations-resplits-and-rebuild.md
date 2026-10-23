# Generations, new splits and the rebuild with one seed per run (2026-10-17 to 2026-10-19)

**Each generation learned its Kaggle targets from the previous generation's out-of-fold
predictions mixed with silver at a 0.75 share. A third generation beat the second on the board
(S174 above S173 above S170), training each family on its own split and seed helped (S175), and
of six families only ConvNeXt, ResNet-50 and NextViT added to the ensemble (S178 first of the
0.965s; EfficientViT, MambaVision and FocalNet each moved it down). The rebuild of the whole chain
with one seed per run gave S181 (gen 2) and S182 (gen 3, first of ten 0.965s) on the v2 labels.**

## Generations before the rebuild

| Generation | Teachers | Kaggle targets | Start weights | Public |
|---|---|---|---|---|
| Gen 0 silver teachers | none | silver | ImageNet / RadImageNet | S158 0.957 (ConvNeXt fold 0) |
| Share children (`*-share075-finetune-fold-k`) | gen-0 ResNet-50 + ConvNeXt, table T2 (`teachers2-kaggle-x4-tau085-tta`) | 0.25 silver + 0.75 T2 | round-2 pretrainings (`*-teachers2-pretrain`) | S170 0.965, ConvNeXt × 5 |
| Gen 3 | the five families' share children | 0.25 silver + 0.75 table (two variants, below) | same round-2 pretrainings | S174 0.965 (ConvNeXt mixed × 5) |

Every child was a Kaggle-only fine-tune (no external data) of its family's round-2 pretraining. The
joint stage with fastMRI/OAI added nothing once the share was in (S168 sorted below S167).

The two gen-3 variants, both at share 0.75:

- **all5**, table `data/labels/prepared/gen3-all5-kaggle-x4-tau085-tta`: mean of the five families' share
  children, each predicting its own validation fold at ×4, τ 0.85, mirror TTA.
- **mixed**, table `data/labels/prepared/gen3-mixed-kaggle-x4-tau085-tta`: half that mean, half the gen-0
  table T2 (probabilities averaged), the literature's remedy against students converging (share
  children agree at 0.99 logit correlation).

Runs are named `<family>-gen3all5-finetune-fold-k` and `<family>-gen3mixed-finetune-fold-k`.

## Selecting the third generation (2026-10-17)

The T2 table's share curve (gold agreement): 0.0 0.9075, 0.25 0.9370, 0.5 0.9437, 0.65 0.9456,
0.70 0.9463, **0.75 0.9471 (peak)**, 0.80 0.9469, 0.85 0.9458, 0.90 0.9454, 1.0 0.9428. Per-label peaks
range from 0.10 (ACL) to 1.0 (Contusion), not used: per-label values chosen on 58 studies overfit.
No new pretraining in this generation: better pretraining labels left the round-2 ConvNeXt students
level, and the joint stage adds nothing once the share is in (S168 below S167).
- **S171 = 0.964 public, below S170 (0.965)**, 9th of ten 0.964s: the share children of MambaVision +
  ResNet-50 (10 equal), picked as the best gold overall out of fold (gold 0.952163 against ConvNeXt
  alone's 0.949055, +0.0031; silver equal, 0.941281 against 0.941280). The first time a gold gain above
  0.002 did not carry to the board. The share inflates every child's gold reading through teachers that
  saw each fold's gold labels, so among share children gold is a weaker guide than it was for S162 and
  S166.
- **Selection error (2026-10-17):** S171 and S172 were picked on the share children's gold out-of-fold
  readings and both left out ConvNeXt, the only family the board had confirmed (S170 0.965). The gold
  readings of share students leak through teachers that saw each fold's gold labels, and the leak differs
  by family, so it reorders them. Rule from here: share students are judged on the board only; new
  ensembles start from the board's best (ConvNeXt × 5) and add a family only when a submission shows it
  helps; the next generation's teachers include ConvNeXt. A clean local reading needs cross-fitted
  teachers (no teacher has seen the gold labels of the fold it labels).
- **S172 = 0.964 public**, just above S171 (MambaVision + ResNet-50 + EfficientViT share children, 15
  equal). Neither share-children ensemble without ConvNeXt beats ConvNeXt × 5 (S170, 0.965).

## Third generation on new splits (2026-10-18)

- **S174 = 0.965 public, first among the 0.965s** (ConvNeXt mixed × 5, `convnext-gen3mixed-finetune-fold-0..4`);
  **S173 = 0.965, second** (ConvNeXt + ResNet-50 share children, 10 equal); S170 third.
- The gen-3 queue was stopped by the user at 04:05 UTC on 2026-10-18. Of the third generation only ConvNeXt mixed is kept;
  every other gen-3 run (ConvNeXt all5, the four other families' mixed and all5) was deleted locally and on the
  machine. Their fold readings on the old split, x1 validation macro AUC: MambaVision mixed 0.94130 0.93541 0.94126
  0.93626 0.93607; EfficientViT mixed 0.94014 0.93444 0.93999 0.93519 0.93592; FocalNet mixed 0.94129 0.93386
  0.93944 0.93452 0.93608; ResNet-50 all5 0.93755 0.93057 0.93501 (folds 0-2).
- Reason: every family trained on the same folds and seed leaves out the same studies, so the families add
  correlated members to ConvNeXt (share children agree at 0.99 logit correlation). The four other families are
  retrained on the mixed table, share 0.75, each on its own new split and seed, for ensemble diversity.
- Splits (scanner-stratified, `scripts/tools/validation.py prepare`, `data/folds/`), one per family:
  `all_5fold_scanner_v2_r1` ResNet-50 (split seed 1533777639), `_r2` MambaVision (3082927044), `_r3` EfficientViT
  (2128871193), `_r4` FocalNet (2597761050), `_r5` ConvNeXt (2423726674). Largest overlap of any fold of one split
  with a fold of another 0.23-0.26 (chance 0.20).
- One training seed per run (25 seeds), folds 0-4: ResNet-50 1203997460 643699413 3046000151 2761326745 3154334900;
  ConvNeXt 3311724702 765003022 4108126569 206399518 2150820031; MambaVision 2424155815 558766553 3500031755
  1497311460 3863367298; EfficientViT 1404199329 3988543562 2583390982 3384567468 3942274526; FocalNet 662050637
  246703576 2685161398 982647884 3735886651.
- `distilled()` in `data/labels.py` required the teacher table's fold to equal the student's fold, which only
  holds on the teacher's own split; it now requires a teacher value for every training study. The table stays
  out-of-fold on its own split; a student's validation studies were already seen by its teachers on the old
  split, so the board stays the only judge.
- Runs `<family>-gen3mixed-r<k>-finetune-fold-<f>`, queue `/root/cycle_gen3_resplit.sh` on the rented machine (started 04:31 UTC): ResNet-50, ConvNeXt, MambaVision, EfficientViT, FocalNet. The old-split ConvNeXt mixed runs (S174)
  stay.
- **NextViT-S as a sixth gen-3 family (2026-10-18, user decision):** the backbone-families-2 campaign's incumbent
  (fold 2 x4 0.938851, ties FocalNet SRF; third family over ResNet-50 + FocalNet +0.00093, logit correlation 0.958
  with the ResNet-50). Tree `/home/claudio/src/knee-abnormality-nextvit` on the machine (this worktree's code, main
  `318eba2` + the `distilled()` fix; sync-free venv). `configs/nextvit-teachers2-pretrain.yaml` (removed) and
  `configs/nextvit-gen3mixed-finetune.yaml` (removed) differ from `configs/nextvit.yaml` exactly as ConvNeXt's do from
  `configs/convnext.yaml`. Pretraining seed 4213577004 (old split fold 0, external only, 10 epochs); fine-tunes on
  `all_5fold_scanner_v2_r6` (split seed 1624582548), seeds 4081590439 3631603500 763392539 14540636 952220979.
  Chain `/root/cycle_nextvit.sh`, real-batch preflight before each run.
- **S175 = 0.965 public, first among four 0.965s** (S175 > S174 > S173 > S170): ConvNeXt r5 + ResNet-50 r1, 10 equal.
  The board order (full-precision public score) reads as a 2x2, family by generation: gen 2 ConvNeXt S170 (4th),
  gen 2 ConvNeXt + ResNet-50 S173 (3rd), gen 3 ConvNeXt S174 (2nd), gen 3 ConvNeXt + ResNet-50 S175 (1st). Adding
  ResNet-50 helps in both generations and the newer generation helps with and without it; the generation step is the
  larger (S174 above S173). The new splits and seeds are not separated in this reading (S175 changed them too).

## Rebuild from gen 1 with a seed per run (2026-10-18 to 2026-10-19)

The plan: every generation from gen 1 retrained with one seed per run; gen 1 keeps the shared split
`all_5fold_scanner_v2` (its gold out-of-fold readings are clean, so teachers can still be
selected), gen 2 and gen 3 use one new split per family and per generation. The chain as it
finally ran is in the
[teacher-student chain](../../../modeling/external-data-and-distillation/teacher-student-chain.md).

- Clean gen-1 reading (x4 out-of-fold, 4,407 studies, 58 gold): NextViT silver teachers (`nextvit-silver-fold-0,1,3,4`
  and the campaign's fold 2, one seed per fold) 0.9352 all / 0.9376 gold, the best single teacher on all studies;
  ResNet-50 0.9346 / 0.9384, ConvNeXt 0.9324 / 0.9374, FocalNet 0.9328 / 0.9308, MambaVision 0.9316 / 0.9284.
  Over all 63 tables of six families (EfficientViT from `effvit-b2-best-folds-0-1-3-4` + `efficientvit-sd02-fold-2`)
  the top six lie within 0.00015 on all studies and all contain ResNet-50 and NextViT (inclusion effects +0.0012 and
  +0.0009; MambaVision -0.0003, EfficientViT -0.0001); ConvNeXt + ResNet-50 + NextViT is first on gold (0.94302 against
  T2's 0.94161) and 0.00015 below the best on all studies. T2 + NextViT gains +0.0016 on all studies (bootstrap 80%
  [+0.0013, +0.0019]) and on every label.
- Gen 1 retrained: ConvNeXt, ResNet-50, FocalNet, MambaVision and EfficientViT, 25 runs, main's configs (identical to
  the gen-0 teachers' recipe), one new seed each; NextViT's is done. The table was chosen afterwards from all 63
  combinations of the rebuilt teachers, since run-to-run variation can reorder the close tables (single-run reruns land
  within about ±0.0003; a five-run table is estimated near 0.00013; the bootstrap SD of a table difference on 4,407 studies
  is about 0.0002).
- Gen-1 teacher choice (user, 2026-10-18): a reasoned judgement over gold (clean at gen 1, proven reliable, 58 studies),
  silver (4,407 studies), the measured seed noise (old vs new gen-1 sets), ranking stability across both sets, per-label
  breadth, diversity and external-prediction cost; default ConvNeXt + ResNet-50 + NextViT when unclear. The chain ran
  unattended on the defaults.
- **S177 = 0.965, last of five 0.965s** (S175 > S174 > S173 > S170 > S177): all six gen-3 families, 30 equal. More families sorted below ConvNeXt + ResNet-50 (S175);
- **S176 = 0.964, first of the 0.964s** (five families, no NextViT). With S175 > S177 > S176: NextViT helps (0.964 -> 0.965), MambaVision + EfficientViT + FocalNet weaken ConvNeXt + ResNet-50; leading evidence for the final members and the gen-2 table.
- Seed noise at the table setting (2026-10-19, x4 τ0.85 TTA, own folds, shared split): ConvNeXt + ResNet-50 table with the
  old seeds (T2) 0.93654 silver / 0.94278 gold, with the new gen-1 seeds 0.93637 / 0.94509 (−0.00017 / +0.0023), logit
  correlation 0.988. Silver seed noise of a two-family table about 0.0001-0.0002; gold moves about 0.002 from the seed
  alone, so gold gaps below about 0.002-0.003 between tables are within seed noise. New single teachers: ResNet-50
  0.93445 / 0.94357, ConvNeXt 0.93322 / 0.94084.
- **Plan change after S175/S176/S177 (2026-10-19, reviewed with the user):** MambaVision, EfficientViT and FocalNet were
  dropped after gen 1 (their gen-3 members weaken the ensemble on the board, and they are the weakest gen-1 teachers;
  FocalNet lowers the four-family tables by 0.0022 on gold). Pretraining, gen 2 and gen 3 continued with ConvNeXt,
  ResNet-50 and NextViT only (3 pretrainings, 15 gen-2 runs, 15 gen-3 runs); the gen-2 half of the mixed table and the
  final ensemble use the same three families. Their gen-1 runs stayed for the teacher analysis.
- **Gen-1 teacher decision (2026-10-19 ~08:20 CEST): ConvNeXt + ResNet-50 + NextViT.** Full six-family reading at x4 τ0.85
  TTA (`outputs/logs/rebuild/teacher-selection-full.txt`): first on gold (0.94666) of 63 tables, 0.93807 on silver within
  about 0.0002 of the best; the gold runners-up with EfficientViT tie on gold (−0.0001 to +0.0002, inside the ~0.002 seed
  noise) but lose about 0.001 on silver in every bootstrap resample. Inclusion effects silver / gold: ResNet-50 +0.0010 /
  +0.0022, NextViT +0.0009 / +0.0009, ConvNeXt +0.0001 / +0.0008, EfficientViT −0.0001 / +0.0008, FocalNet +0.0002 /
  −0.0016, MambaVision −0.0001 / −0.0006. Tables: `data/labels/prepared/rebuild-t2-x4-tau085-tta` (Kaggle, 15 teachers)
  and `rebuild-gen1-x4-tau085-tta-oai-soft-tissue` (external, 19,265 studies).
- **S178 (probe A) = 0.965, first of six 0.965s** (S178 > S175 > S174 > S173 > S170 > S177): old-cycle gen-3 ConvNeXt + ResNet-50 + NextViT, 15 equal. NextViT added to ConvNeXt + ResNet-50 beats S175: the three-family core is confirmed as members.
- **S179 (probe C) = 0.965, 5th of seven 0.965s**, below probe A (S178): EfficientViT added to ConvNeXt + ResNet-50 + NextViT moves the ensemble down (gen 1 predicted a tie or slight loss). EfficientViT stays out of gen 2 and gen 3; its queued pretraining was removed.
- **S180 (probe B) = 0.965, 7th of eight 0.965s**, below probe A: MambaVision added to the three moves the ensemble down further than EfficientViT. MambaVision and FocalNet pretrainings removed; the rebuild is ConvNeXt + ResNet-50 + NextViT only. Board family tests, all against S178: + EfficientViT 5th, + MambaVision 7th, + all others (S177) 8th.
- **S181 (rebuild gen 2, ConvNeXt + ResNet-50 + NextViT, 15) = 0.965, 7th of the 0.965s; S182 (rebuild gen 3, same families) = 0.965, first of ten 0.965s**, above S178: gen 3 adds over gen 2 under the v2 labels.
