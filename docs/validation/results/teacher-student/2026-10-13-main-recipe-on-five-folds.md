# Main's ResNet-50 recipe at `288e50a` on five folds (2026-10-13)

**The recipe selected on fold 2 over 29 campaign runs did not carry to the other folds: over the
four fresh folds it was −0.0001 at ×4 and −0.0002 at ×1 against the previous teachers, and
removing any one of its three adoptions (OA detector, average-pool shortcuts, mask decoder) did
not move fold 0.** These five runs became the teachers of the second pseudo-label round
([second round](2026-10-14-second-round-averaged-labels.md)).

Trained as the five teachers for a new pseudo-label round: main's exact tree (`git archive`,
`/workspace/repo-main-288e50a` on the rented machine), `configs/resnet50.yaml`, seed 1110484336,
Kaggle only. Reference per fold: the teacher of the current pseudo-labels (folds 0 to 2
`rn50-aligned-mixing-folds-0-2`, fold 3 `rn50-main-0bd598f-folds-3-4`, fold 4
`rn50-main-0bd598f-fold-4-workers-24`), same seed and split.

| Fold | Run | ×1 new / previous | Δ ×1 | ×4 new / previous | Δ ×4 |
|---|---|---|---:|---|---:|
| 0 | `rn50-main-288e50a-folds-0-4` | 0.934235 / 0.933184 | +0.0011 | 0.940416 / 0.940906 | −0.0005 |
| 1 | `rn50-main-288e50a-folds-0-4` | 0.923958 / 0.925323 | −0.0014 | 0.932858 / 0.933392 | −0.0005 |
| 2 | campaign run 27, copied as `rn50-main-288e50a-fold-2-campaign-run-27` (same source identity, seed and settings as main) | 0.930902 / 0.928460 | +0.0024 | 0.939634 / 0.937000 | +0.0026 |

Folds 3 and 4 were stopped by the user at fold 3's fourth epoch. Training takes 27.7 minutes a
fold on the rented machine; on the local one the new and the previous recipe both take 44.9.

×4 by label, new against previous (fold 0, fold 1, fold 2): PF OA −0.0035, −0.0025, −0.0029;
Lateral OA −0.0040, −0.0020, +0.0065; Medial OA +0.0001, +0.0017, +0.0014; MCL +0.0022, −0.0066,
+0.0108; Fracture −0.0026, +0.0009, +0.0041; Baker's −0.0034, +0.0004, +0.0036; Synovitis +0.0026,
−0.0016, +0.0008; Lateral Meniscus +0.0020, +0.0035, −0.0016; Contusion −0.0003, +0.0040, +0.0054.

The recipe was selected on fold 2 over 29 campaign runs, each adoption a step of about one repeat
distance there (0.0005, 0.0011, 0.0010 at ×4; a repeat moves 0.0010).

**Fold 0 without the OA detector** (`rn50-main-288e50a-no-oa-detector-fold-0`, the same tree with
the line adding the detector's correction to the OA logits removed, so every other weight starts
equal): ×1 0.933310 (−0.0009 against the run with it, +0.0001 against the previous recipe), ×4
0.940484 (+0.0001, −0.0004). ×4 by label against the run with the detector: Baker's +0.0034, MCL
+0.0027, Lateral OA +0.0012, PF OA +0.0010, Effusion +0.0009, ACL +0.0009, Medial Meniscus
+0.0002, Synovitis −0.0004, Medial OA −0.0012, Lateral Meniscus −0.0013, Contusion −0.0015,
Fracture −0.0051. Against the previous recipe PF OA is still −0.0025 and Lateral OA −0.0028, so
the detector is not what lowers them on this fold; the mask decoder and the average-pool
shortcuts are not separated.

**Fold 0 without the average-pool shortcuts** (`rn50-main-288e50a-no-avg-pool-shortcuts-fold-0`,
the same tree with the two lines that switch them on removed; mask decoder and OA detector kept):
×1 0.933821 (−0.0004 against the new recipe, +0.0006 against the previous), ×4 0.940342 (−0.0001,
−0.0006). ×4 by label against the new recipe: MCL +0.0022, Baker's +0.0013, ACL +0.0013, Lateral
Meniscus +0.0012, Contusion +0.0010, PF OA +0.0007, Synovitis +0.0004, Medial OA −0.0004, Medial
Meniscus −0.0011, Effusion −0.0018, Lateral OA −0.0027, Fracture −0.0029.

Fold 0 at ×4, four trainings: previous recipe 0.940906, new recipe 0.940416, without the OA
detector 0.940484, without the shortcuts 0.940342. The three runs from main's tree are within
0.00014 of each other and 0.0004 to 0.0006 under the previous teacher, and share its per-label
differences: PF OA −0.0035, −0.0025, −0.0028; Lateral OA −0.0040, −0.0028, −0.0067; Fracture
−0.0026, −0.0077, −0.0055; MCL +0.0022, +0.0049, +0.0044; Synovitis +0.0026, +0.0023, +0.0031.
What the three share and the previous teacher lacks: the mask decoder, the commits between the
teacher's code and `288e50a` other than the three adoptions, and the machine (the previous
teacher was trained elsewhere). Those are not separated; neither the OA detector nor the
shortcuts moves fold 0.

**Fold 0 without the mask decoder** (`rn50-main-288e50a-no-mask-decoder-fold-0`, the same tree
with the line adding the decoder's loss removed; OA detector and shortcuts kept): ×1 0.934069
(−0.0002 against the new recipe, +0.0009 against the previous), ×4 0.940614 (+0.0002, −0.0003).
×4 by label against the new recipe: Baker's +0.0046, Synovitis +0.0020, Lateral Meniscus +0.0018,
MCL +0.0014, Contusion +0.0002, Effusion 0.0000, ACL −0.0001, Medial Meniscus −0.0005, Fracture
−0.0011, PF OA −0.0014, Medial OA −0.0015, Lateral OA −0.0029.

**All five fold 0 trainings, ×1 / ×4:** previous teacher 0.933184 / 0.940906; new recipe 0.934235
/ 0.940416; without the OA detector 0.933310 / 0.940484; without the shortcuts 0.933821 /
0.940342; without the mask decoder 0.934069 / 0.940614. The four runs from main's tree span
0.0003 at ×4 and 0.0009 at ×1: no adoption changes fold 0 when removed. Against the previous
teacher all four keep PF OA (−0.0025 to −0.0048), Lateral OA (−0.0028 to −0.0069) and Fracture
(−0.0026 to −0.0077) lower and MCL (+0.0022 to +0.0049) and Synovitis (+0.0023 to +0.0047)
higher, so that pattern belongs to none of the three adoptions; what remains is the code between
`0bd598f` (with its uncommitted changes) and `288e50a` other than the adoptions, and the machine.

**Folds 3 and 4, trained afterwards** (`rn50-main-288e50a-folds-3-4`, main's tree unchanged):

| Fold | ×1 new / previous | Δ ×1 | ×4 new / previous | Δ ×4 | Training |
|---|---|---:|---|---:|---:|
| 3 | 0.925043 / 0.925655 | −0.0006 | 0.931805 / 0.931023 | +0.0008 | 27.4 min |
| 4 | 0.926211 / 0.926103 | +0.0001 | 0.933838 / 0.934064 | −0.0002 | 28.8 min |

×4 by label, fold 3: Synovitis +0.0036, Lateral Meniscus +0.0033, Baker's +0.0032, Effusion
+0.0017, MCL +0.0009, Lateral OA +0.0006, ACL 0.0000, PF OA −0.0002, Fracture −0.0004, Contusion
−0.0006, Medial Meniscus −0.0012, Medial OA −0.0016. Fold 4: Contusion +0.0033, Effusion +0.0031,
PF OA +0.0028, Fracture +0.0024, Lateral OA +0.0016, Medial Meniscus +0.0011, Baker's −0.0003,
Medial OA −0.0003, ACL −0.0011, Lateral Meniscus −0.0026, Synovitis −0.0064, MCL −0.0064.

**Five folds of main's recipe at `288e50a` against the previous teachers:** mean ×4 0.935710
against 0.935277 (+0.0004), mean ×1 0.928070 against 0.927745 (+0.0003). Without fold 2, where
the recipe was selected: ×4 −0.0001, ×1 −0.0002 over four folds. No label moves the same way on
all four fresh folds. The fold 3 run stopped earlier from the same code, seed and machine had
0.900969 and 0.911379 at its first two epochs against 0.901889 and 0.908213 here: two identical
launches differ by up to 0.003 within an epoch.

The five teachers of the new recipe: folds 0 and 1 in `rn50-main-288e50a-folds-0-4`, fold 2 in
`rn50-main-288e50a-fold-2-campaign-run-27`, folds 3 and 4 in `rn50-main-288e50a-folds-3-4`, all
on the rented machine with main's source identity.
