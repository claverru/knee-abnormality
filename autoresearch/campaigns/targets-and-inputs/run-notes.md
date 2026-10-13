# Targets-and-inputs campaign: run notes

One section per run: both multipliers against the incumbent, per-label changes from largest gain to largest loss, cost, and the lesson.

## 1. soft-prevalence-class-weights (2026-10-12)

Class weights from each label's mean soft target instead of its thresholded fraction, on the reference.

- **×4:** 0.937509 against 0.937000, +0.000509, 80% interval [−0.000222, +0.001285].
- **×1:** 0.929788 against 0.928460, +0.001327 (epoch 10). Behind at epochs 1 to 3 (−0.0043 at epoch 3), ahead from epoch 4.
- **Soft macro AUC:** 0.797401 against 0.796529, +0.000871.
- **Cost:** training 2,540 s (0.94 of the reference), prediction 65 s, peak GPU memory 16.2 GB.
- **Decision:** the runtime adopted it (above at both multipliers). **The keeper reset the incumbent to the reference the same evening on the user's concern** that class-weight changes have disagreed with the public board; the run stays a recorded candidate until checked publicly, and seed 2 onwards build on the reference (`decisions/keeper-incumbent-reset-20261012T1745Z`).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Fracture | +0.0042 | +0.0031 |
| Contusion | +0.0031 | +0.0023 |
| Effusion | +0.0017 | +0.0008 |
| Lateral OA | +0.0014 | +0.0014 |
| Baker's | +0.0009 | +0.0011 |
| Medial OA | +0.0009 | +0.0008 |
| MCL | +0.0007 | +0.0009 |
| Medial Meniscus | +0.0001 | +0.0001 |
| Lateral Meniscus | −0.0000 | +0.0014 |
| ACL | −0.0005 | +0.0009 |
| PF OA | −0.0028 | −0.0016 |
| Synovitis | −0.0036 | −0.0007 |

**Lesson:** of the four labels whose weights changed most, Fracture, Effusion and MCL rose and Synovitis, whose positive weight fell furthest (2.16 to about 1.29), lost the most; PF OA lost without a large weight change. Past public pairs on class weighting (ReXNet-200, fold 0, September) never followed the local hard-AUC sign, so this local gain is not read as a public one.

**Proxies for the public direction** (subagent, 2026-10-12; fold 2, 875 studies, 80% report-group bootstrap; track record = public sign right on the five recorded class-weight pairs, ReXNet-200 fold 0):

| Proxy | Track record | ×4 Δ | ×1 Δ |
|---|---|---:|---:|
| Hard AUC | 2/5 | +0.0005 [−0.0002, +0.0012] | +0.0013 |
| Soft AUC | 4/5 | +0.0009 [+0.0005, +0.0012] | +0.0009 |
| Nine other extractors, soft | 4/5 | +0.0009 [+0.0006, +0.0013] | +0.0009 |
| Soft, same-side pairs | 3/5 | +0.0013 [+0.0009, +0.0018] | +0.0006 |
| Negatives the report addressed | none | +0.0010 [+0.0003, +0.0018] | +0.0013 |
| Gold only (12 studies) | unusable | −0.0026 | +0.0039 |

- In the five old class-weight pairs hard and same-side soft AUC moved in opposite directions; here both rise. In the 19 recorded ×4 pairs with small same-sign hard and soft deltas, public went up 11, tied 3, down 5.
- Synovitis: the hard loss is +0.0017 on addressed negatives; 724 of its 761 negatives were unaddressed by the original extractor. PF OA loses on every proxy although its weight barely changed.
- Open: a paired submission of candidate and reference fold 2 at one inference setting; the reference fold 2 has no public score.

**Public score (S150, 2026-10-13):** 0.960 at ×4, τ 0.9, single model, no mirror TTA. In Kaggle's list sorted by public score it is fifth of all submissions and first of the four at 0.960, above S144 (the same recipe without this change, fold 1), S143 and S138, and above the five at 0.959 (S126, S145, S146, S147 on fold 0 with per-label τ and mirror TTA, S142). Its own reference, fold 2 of `rn50-aligned-mixing-folds-0-2`, has no public score, so the pair that isolates the change is open: against S144 the fold differs as well. Read: the soft-prevalence weights did not cost public score, the outcome the proxies pointed to; the size of the change is not known without the fold-2 reference.

**The pair (S151, 2026-10-13):** the reference itself, fold 2 of `rn50-aligned-mixing-folds-0-2`, at the same settings (single model, ×4, τ 0.9, exported from its own source snapshot) scored 0.959 and sits 13th in Kaggle's score order, fifth of the six at 0.959; S150 is 5th at 0.960. One pair on one fold: the soft-prevalence class weights are +0.001 public and eight places up, the same sign as locally (×4 +0.0005, ×1 +0.0013, soft +0.0009) and as every proxy with a track record. This is the first class-weight change whose public and local signs agree. 0.959 is also the public reference of the campaign's fold-2 recipe.

## 2. stable-intensity-bounds (2026-10-12)

A series' clipping bounds from a fixed set of up to seven of its slices instead of the slices drawn, on the reference.

- **×4:** 0.936730 against 0.937000, −0.000270, 80% interval [−0.001035, +0.000511].
- **×1:** 0.929069 against 0.928460, +0.000608 (epoch 10). Behind at epochs 2 and 3 (−0.0026), ahead at 4, 5, 9 and 10.
- **Soft macro AUC:** −0.000057.
- **Cost:** training 2,580 s (0.96 of the reference), prediction 65 s, peak GPU memory 16.2 GB.
- **Decision:** not adopted (below the incumbent at ×4).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Baker's | +0.0030 | +0.0012 |
| Contusion | +0.0010 | +0.0000 |
| Effusion | +0.0007 | −0.0003 |
| Lateral OA | +0.0006 | +0.0009 |
| Medial Meniscus | −0.0000 | −0.0004 |
| Synovitis | −0.0001 | +0.0005 |
| Fracture | −0.0001 | +0.0002 |
| Medial OA | −0.0004 | −0.0000 |
| Lateral Meniscus | −0.0005 | +0.0004 |
| ACL | −0.0008 | −0.0005 |
| PF OA | −0.0018 | −0.0010 |
| MCL | −0.0048 | −0.0016 |

**Lesson:** the seed expected more at ×4 than at ×1 and the run shows the opposite: +0.0006 at ×1 and −0.0003 at ×4, so the step from ×1 to ×4 shrank by 0.0009. Open: whether the fixed set of seven slices gives the ×4 bag, which reads most of the series, worse bounds than its own slices did; MCL (45 positives) carries the largest loss. PF OA lost here and in run 1.

## 3. uncertain-report-targets-weigh-less (2026-10-12)

Each report-derived training cell weighted by its target's distance from 0.5, per label at equal total loss mass, on the reference.

- **×4:** 0.936047 against 0.937000, −0.000953, 80% interval [−0.002044, +0.000157].
- **×1:** 0.928319 against 0.928460, −0.000141 (epoch 10). Behind at every epoch but 4 and 9; −0.0038 at epochs 2, 3 and 5.
- **Soft macro AUC:** +0.000178.
- **Cost:** training 2,623 s (0.97 of the reference), prediction 67 s, peak GPU memory 16.2 GB.
- **Decision:** not adopted (below the incumbent at both multipliers).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Synovitis | +0.0047 | +0.0023 |
| Effusion | +0.0017 | +0.0022 |
| MCL | +0.0005 | −0.0005 |
| Medial Meniscus | +0.0002 | +0.0005 |
| Baker's | −0.0002 | +0.0017 |
| Medial OA | −0.0006 | +0.0006 |
| Lateral OA | −0.0015 | +0.0006 |
| Lateral Meniscus | −0.0018 | −0.0003 |
| ACL | −0.0019 | −0.0012 |
| Contusion | −0.0028 | +0.0002 |
| Fracture | −0.0045 | −0.0017 |
| PF OA | −0.0050 | −0.0023 |

**Lesson:** two of the three aimed labels moved as the seed expected (Synovitis +0.0047, Effusion +0.0017, both also on soft AUC), and the third, PF OA, lost the most (−0.0050, interval below zero) with 24% of its cells between 0.25 and 0.75. PF OA has now lost in all three runs (−0.0028, −0.0018, −0.0050). Labels with few uncertain cells (Fracture, Contusion, ACL, Lateral Meniscus) lost although their weights barely changed, so the cost is not confined to the reweighted labels. Open: a per-label rule fixed before the run (Synovitis and Effusion only) and what makes PF OA's uncertain cells informative where Synovitis's are not.

## 4. hard-report-targets (2026-10-12)

Report-derived training targets set to 0 or 1 at 0.5 (41,832 cells changed), on the reference.

- **×4:** 0.934129 against 0.937000, −0.002871, 95% interval [−0.005393, −0.000292].
- **×1:** 0.925784 against 0.928460, −0.002676 (epoch 10). Behind at every epoch, −0.0054 at epoch 2.
- **Cost:** training 2,564 s (0.95 of the reference), prediction 66 s, peak GPU memory 16.2 GB.
- **Decision:** recorded by the runtime as `invalid`, not as a rejection: the comparison gate refused the run because `training_rows_sha256` differs, which this seed changes by design (it hashes the training targets). The keeper re-ran `compare_fold_runs` on the saved predictions with that one identity difference allowed; the figures here come from it. Below the incumbent at both multipliers, so no adoption either way and no retraining.

| Label | ×4 Δ |
|---|---:|
| Contusion | +0.0063 |
| Synovitis | +0.0000 |
| Baker's | -0.0010 |
| Medial Meniscus | -0.0016 |
| Effusion | -0.0025 |
| ACL | -0.0026 |
| Medial OA | -0.0027 |
| PF OA | -0.0043 |
| Fracture | -0.0047 |
| Lateral OA | -0.0052 |
| Lateral Meniscus | -0.0071 |
| MCL | -0.0091 |

**Lesson:** hard targets lose on the macro at both multipliers, with the interval below zero: the soft values carry information the model uses. With run 3 (uncertain cells weighted down, −0.0010) the two runs agree that removing or flattening the uncertain cells costs AUC on this recipe. Harness: a seed that changes training target values needs `training_rows_sha256` among the allowed identity differences, or its comparison fails after a full training.

## 5. mask-decoder-beside-the-labels (2026-10-12)

A decoder on the second, third and fourth stage maps predicting the bone and soft-tissue masks on the 28 by 28 grid, in training only, on the reference.

- **×4:** 0.937529 against 0.937000, +0.000529, 80% interval [−0.000239, +0.001311].
- **×1:** 0.929885 against 0.928460, +0.001424 (epoch 10). Behind at epochs 1 to 3, ahead from epoch 4.
- **Soft macro AUC:** +0.000389.
- **Cost:** training 2,617 s (0.97 of the reference, inside the seed's 1.10 ceiling), prediction 65 s, peak GPU memory 16.7 GB allocated (17.1 GB on the card).
- **Decision:** adopted by the runtime (above at both multipliers); the incumbent from run 6 on.

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| MCL | +0.0031 | +0.0018 |
| Lateral OA | +0.0025 | +0.0002 |
| Contusion | +0.0021 | +0.0016 |
| Effusion | +0.0015 | −0.0002 |
| Baker's | +0.0008 | +0.0003 |
| Medial Meniscus | +0.0007 | +0.0005 |
| Medial OA | +0.0006 | +0.0004 |
| Fracture | +0.0004 | +0.0013 |
| Lateral Meniscus | −0.0004 | +0.0006 |
| Synovitis | −0.0005 | −0.0001 |
| ACL | −0.0006 | +0.0011 |
| PF OA | −0.0039 | −0.0028 |

**Lesson:** nine labels up, spread over ligament, OA and bone labels rather than the meniscus and OA labels the seed named first. **PF OA lost in all five runs** (−0.0028, −0.0018, −0.0050, −0.0043, −0.0039 at ×4), under five unrelated mechanisms, here with the interval below zero. What the five share and the reference lacks: they were trained by this campaign from the worktree's source (loader changes of `c1d973b`, one fold per run, the campaign's launcher) while the reference is fold 2 of a three-fold run from main at `0bd598f`. Candidate cause: a PF OA difference between the reference training and any training from this source, independent of the seed. A retraining of the reference recipe inside the campaign would separate it; not launched (control runs are the user's call).

### Run 5, the audit (subagent, 2026-10-12; saved metrics, predictions and CPU-loaded weights, no forward pass)

Compared with the reference and with the three no-decoder campaign runs (1, 2, 3; "pool"), fold 2, 875 studies, one run.

- **Against the pool the decoder is +0.0008 at ×4** (80% interval [−0.0001, +0.0016]) and +0.0008 at ×1 ([−0.0002, +0.0018]); the pool is −0.0002 at ×4 and +0.0006 at ×1 against the reference, so about 0.0006 of the +0.0014 ×1 gain is campaign against reference. PF OA is −0.0007 against the pool at ×4 while the pool is −0.0032 against the reference: its loss is mostly not the decoder's.
- **The existing maps got better:** `val_anatomy_loss` 0.6333 against 0.6400 to 0.6431, `val_meniscus_loss` 0.5921 against 0.6023 to 0.6049, at every epoch; overlaps of the kneecap, fibula, medial tibial cartilage and lateral meniscus outside the no-decoder range.
- **Geometry and routing did not move:** plane error 1.6136 mm against 1.6213 to 1.6772, centre and crop shift inside the range, region routing weights within ±0.001. The soft-tissue routing opens about 10% more lateral meniscus cells and 2% more lateral OA cells; medial unchanged.
- **Where the AUC moved:** the labels with no soft-tissue routed read (ACL, MCL, Effusion, Synovitis, Baker's) are +0.0024 at ×1 against the pool [+0.0010, +0.0038] and +0.0007 at ×4; the five routed labels −0.0004 at ×1 and +0.0007 at ×4; Contusion and Fracture +0.0001 and +0.0012. MCL +0.0086 at ×1.
- **Weights:** distance to the reference relative to a pool run's: stages one to three 1.01 to 1.03, stage four 1.14, the token projection 1.37. No regulariser signature in the training loss.
- **The decoder's task is exhausted early:** bone loss 0.8804 to 0.2730, soft-tissue 0.5652 to 0.1921, flat from epoch 7; its two terms carry 0.0465 of the training loss against 0.1281 for the two map heads.
- **Reading:** best supported is a feature effect in stage four and the token projection that helps the unrouted labels (medium); the better maps are a fact whose link to AUC is a guess. Follow-up seeds: [sixth seeds](seed-ideas-6.yaml).

## 6. intensity-augmentation-off (2026-10-12)

`data.intensity_augmentation.probability` 0, on the mask-decoder incumbent (run 5).

- **×4:** 0.937148 against 0.937529, −0.000381, 80% interval [−0.001422, +0.000680].
- **×1:** 0.928970 against 0.929885, −0.000914 (epoch 10). +0.0063 at epoch 1, +0.0020 at 3, +0.0012 at 5, then behind from epoch 6 (−0.0014 at epoch 9).
- **Soft macro AUC:** −0.000108.
- **Cost:** training 2,592 s (0.99 of the incumbent), prediction 66 s, peak GPU memory 16.7 GB.
- **Decision:** not adopted (below the incumbent at both multipliers).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| ACL | +0.0039 | +0.0020 |
| PF OA | +0.0024 | +0.0018 |
| Baker's | +0.0021 | −0.0002 |
| Fracture | +0.0012 | −0.0012 |
| Medial Meniscus | +0.0005 | +0.0002 |
| MCL | +0.0001 | −0.0006 |
| Synovitis | −0.0003 | +0.0004 |
| Medial OA | −0.0011 | −0.0003 |
| Lateral OA | −0.0024 | +0.0013 |
| Contusion | −0.0029 | −0.0022 |
| Effusion | −0.0031 | −0.0007 |
| Lateral Meniscus | −0.0049 | −0.0016 |

**Lesson:** without the intensity augmentation the model learns faster (ahead through epoch 5) and ends behind from epoch 6: the augmentation costs early fit and pays in the late epochs. The losses sit on fluid-read labels (Effusion, interval below zero; Contusion) and on Lateral Meniscus (interval below zero); ACL and PF OA gained. Open: the augmentation only on the fluid-sensitive series, or its range rather than its presence.

## 7. noise-augmentation-off (2026-10-13)

`data.noise_augmentation.probability` 0, on the mask-decoder incumbent (run 5).

- **×4:** 0.937905 against 0.937529, +0.000376, 80% interval [−0.000666, +0.001392]. The highest ×4 of the campaign so far.
- **×1:** 0.929498 against 0.929885, −0.000387 (epoch 10). Ahead at epochs 1 to 5 (+0.0028 at epoch 5), behind at 6 (−0.0013) and 9 (−0.0006).
- **Soft macro AUC:** −0.000029.
- **Cost:** training 2,598 s (0.99 of the incumbent), prediction 65 s, peak GPU memory 16.7 GB.
- **Decision:** not adopted: above the incumbent at ×4, below at ×1, and the rule asks for both.

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Fracture | +0.0052 | +0.0001 |
| Synovitis | +0.0035 | +0.0013 |
| ACL | +0.0027 | +0.0032 |
| Baker's | +0.0024 | +0.0014 |
| PF OA | +0.0022 | +0.0024 |
| Lateral Meniscus | +0.0012 | +0.0008 |
| Medial Meniscus | −0.0010 | −0.0007 |
| Medial OA | −0.0011 | −0.0019 |
| MCL | −0.0016 | −0.0039 |
| Contusion | −0.0023 | −0.0014 |
| Lateral OA | −0.0031 | −0.0017 |
| Effusion | −0.0035 | +0.0001 |

**Lesson:** the two augmentation removals share a shape: faster early epochs, a turn at epoch 6, and the same per-label signs on these seven labels: ACL, PF OA and Baker's up in both (+0.0039 and +0.0027, +0.0024 and +0.0022, +0.0021 and +0.0024), Effusion, Contusion, Lateral OA and Medial OA down in both (Effusion −0.0031 and −0.0035, both intervals below zero). The intensity-changing augmentations help the fluid-read labels and cost ACL, PF OA and Baker's. The step from ×1 to ×4 grew by 0.0008 without noise. Open: both augmentations kept only for the fluid series, where Effusion and Contusion are read.

## 8. hard-pair-final-phase (2026-10-13)

In epochs 8 to 10 the worst-ordered positive-negative pairs share a batch and a pair term at weight 0.1 is added; every study still seen once per epoch. On the mask-decoder incumbent (run 5).

- **×4:** 0.937728 against 0.937529, +0.000199, 80% interval [−0.000590, +0.001025].
- **×1:** 0.929220 against 0.929885, −0.000665 (epoch 10). Epochs 1 to 7, where the recipe is the incumbent's, differ from it by −0.0013 to +0.0012 (0.0000 at epoch 7); the mined epochs are +0.0001, −0.0011 and −0.0007.
- **Soft macro AUC:** −0.000094.
- **Cost:** training 2,592 s (0.99 of the incumbent), prediction 65 s, peak GPU memory 16.8 GB.
- **Decision:** not adopted: above the incumbent at ×4, below at ×1.

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| PF OA | +0.0031 | +0.0014 |
| MCL | +0.0015 | +0.0003 |
| Fracture | +0.0012 | −0.0018 |
| Baker's | +0.0008 | +0.0008 |
| Effusion | +0.0006 | +0.0012 |
| Medial OA | +0.0006 | +0.0004 |
| ACL | +0.0006 | −0.0012 |
| Medial Meniscus | +0.0004 | +0.0009 |
| Synovitis | −0.0003 | −0.0001 |
| Contusion | −0.0006 | −0.0013 |
| Lateral OA | −0.0015 | +0.0007 |
| Lateral Meniscus | −0.0040 | −0.0025 |

**Lesson:** the run reached epoch 7 level with the incumbent and the three mined epochs left ×1 0.0007 lower and ×4 0.0002 higher: no gain from the phase at this weight and pair count. Eight labels rose at ×4 and Lateral Meniscus carried the loss (−0.0040), as in run 6 (−0.0049). The first seven epochs give a yardstick for this recipe at ×1: two trainings that differ only in the stored-logit buffer are up to 0.0013 apart at one epoch. Runs 7 and 8 are both above the incumbent at ×4 and below at ×1.

## 9. plane-contrast-readouts-off (2026-10-13)

`model.plane_readout` off, on the mask-decoder incumbent (run 5).

- **×4:** 0.937778 against 0.937529, +0.000249, 80% interval [−0.000769, +0.001305].
- **×1:** 0.929504 against 0.929885, −0.000381 (epoch 10). −0.0042 at epoch 1, behind through epoch 4, +0.0015 at epoch 5, then within 0.0006.
- **Soft macro AUC:** −0.000308.
- **Cost:** training 2,609 s (1.00 of the incumbent), prediction 65 s, peak GPU memory 16.7 GB.
- **Decision:** not adopted: above the incumbent at ×4, below at ×1.

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| ACL | +0.0025 | +0.0003 |
| Synovitis | +0.0022 | +0.0006 |
| Fracture | +0.0009 | −0.0018 |
| Baker's | +0.0005 | +0.0008 |
| Lateral Meniscus | +0.0005 | +0.0020 |
| PF OA | +0.0005 | +0.0002 |
| MCL | +0.0004 | −0.0025 |
| Medial OA | −0.0002 | −0.0000 |
| Medial Meniscus | −0.0008 | −0.0013 |
| Effusion | −0.0011 | −0.0014 |
| Contusion | −0.0012 | −0.0014 |
| Lateral OA | −0.0013 | +0.0008 |

**Lesson:** the model without the plane-contrast readouts lands level with the incumbent (no label beyond ±0.0025, every interval across zero), with a slower start (−0.0042 at epoch 1): the readouts speed the first epochs and are not needed at the end. Runs 7, 8 and 9, three unrelated changes on the incumbent, all land at ×4 +0.0002 to +0.0004 and ×1 −0.0004 to −0.0007: the incumbent's ×1 (0.929885) sits above what three trainings near its recipe reach, and its ×4 (0.937529) below. The both-multipliers rule then rejects on the ×1 side each time.

## 10. oa-detector-beside-the-shared-head (2026-10-13)

The three-compartment OA detector restored as a correction on the three OA logits, beside the routed shared OA head. On the mask-decoder incumbent (run 5). A user seed, an exception to the contract's exclusion of new readout structure.

- **×4:** 0.938618 against 0.937529, +0.001089, 80% interval [+0.000431, +0.001775], 95% interval [+0.000141, +0.002144]: the first run of the campaign with the interval above zero.
- **×1:** 0.931247 against 0.929885, +0.001362 (epoch 10). Ahead from epoch 3; +0.0015 to +0.0019 at epochs 6 to 8.
- **Soft macro AUC:** +0.000570.
- **Cost:** training 2,602 s (0.99 of the incumbent), prediction 65 s, peak GPU memory 16.7 GB.
- **Decision:** adopted by the runtime (above at both multipliers); the incumbent from run 11 on. Against the campaign's reference: ×4 +0.001618, ×1 +0.002787.

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| MCL | +0.0044 | +0.0005 |
| PF OA | +0.0019 | +0.0021 |
| ACL | +0.0016 | +0.0013 |
| Medial OA | +0.0013 | +0.0007 |
| Lateral OA | +0.0013 | +0.0017 |
| Contusion | +0.0012 | −0.0004 |
| Baker's | +0.0010 | +0.0012 |
| Synovitis | +0.0007 | +0.0003 |
| Fracture | +0.0007 | −0.0010 |
| Medial Meniscus | +0.0000 | −0.0001 |
| Effusion | −0.0002 | +0.0005 |
| Lateral Meniscus | −0.0011 | −0.0001 |

**Lesson:** the three OA labels the detector corrects all rise (PF OA +0.0019, Medial OA +0.0013, Lateral OA +0.0013, also on soft AUC), and ten labels rise in all; the three OA labels account for 0.0004 of the +0.0011 macro and the labels the detector does not touch for the rest (MCL +0.0044, ACL +0.0016). Runs 7, 8 and 9 on the same incumbent were below it at ×1 by 0.0004 to 0.0007; this one is above by 0.0014, so the ×1 gain is outside what those three trainings spanned. PF OA, which lost in runs 1 to 5 against the reference, is back to 0.9136 (reference 0.9155).

## 11. oa-detector-instead-of-the-shared-head (2026-10-13)

Three separate OA heads plus the OA detector, in place of the routed shared OA head. On the OA-detector incumbent (run 10). A user seed.

- **×4:** 0.937965 against 0.938618, −0.000652, 80% interval [−0.001664, +0.000277].
- **×1:** 0.929208 against 0.931247, −0.002039 (epoch 10). Level through epoch 6 (±0.0008 after epoch 1), then −0.0019 to −0.0020 from epoch 7.
- **Soft macro AUC:** +0.000164.
- **Cost:** training 2,593 s (1.00 of the incumbent), prediction 65 s, peak GPU memory 16.7 GB.
- **Decision:** not adopted (below the incumbent at both multipliers).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Lateral Meniscus | +0.0025 | +0.0008 |
| PF OA | +0.0010 | +0.0003 |
| Contusion | +0.0007 | +0.0004 |
| Medial Meniscus | +0.0004 | +0.0005 |
| Synovitis | +0.0003 | −0.0004 |
| Baker's | −0.0003 | +0.0002 |
| ACL | −0.0004 | −0.0009 |
| Fracture | −0.0008 | +0.0019 |
| Lateral OA | −0.0008 | +0.0013 |
| Effusion | −0.0010 | −0.0005 |
| Medial OA | −0.0042 | −0.0005 |
| MCL | −0.0053 | −0.0011 |

**Lesson:** the routed shared OA head is worth keeping beside the detector: without it Medial OA loses 0.0042 (interval below zero) and Lateral OA 0.0008, while PF OA, the label the detector was first adopted for, gains 0.0010. MCL (45 positives) gives back the 0.0044 it gained in run 10 (−0.0053 here) although neither change reads MCL; its interval spans [−0.0171, +0.0030]. With run 10: detector beside the shared head +0.0011, detector in its place −0.0007.

## 12. report-attribute-loss-on (2026-10-13)

`trainer.report_attribute_loss_weight` 0.1 (it was 0), on the OA-detector incumbent (run 10). A user seed.

- **×4:** 0.938323 against 0.938618, −0.000295, 80% interval [−0.000977, +0.000329].
- **×1:** 0.929382 against 0.931247, −0.001865 (epoch 10). +0.0031 at epoch 2 and +0.0011 at epoch 5, then −0.0013 to −0.0023 from epoch 6.
- **Soft macro AUC:** −0.000102.
- **Cost:** training 2,654 s (1.02 of the incumbent), prediction 66 s, peak GPU memory 16.8 GB.
- **Decision:** not adopted (below the incumbent at both multipliers).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Fracture | +0.0020 | +0.0013 |
| Synovitis | +0.0019 | +0.0002 |
| Baker's | +0.0008 | +0.0010 |
| Lateral Meniscus | +0.0008 | −0.0011 |
| PF OA | +0.0005 | −0.0004 |
| Effusion | −0.0002 | −0.0004 |
| Medial Meniscus | −0.0002 | +0.0000 |
| Medial OA | −0.0009 | −0.0003 |
| MCL | −0.0014 | +0.0013 |
| ACL | −0.0019 | −0.0015 |
| Lateral OA | −0.0022 | −0.0009 |
| Contusion | −0.0027 | −0.0005 |

**Lesson:** the report-attribute heads at weight 0.1 leave the ResNet-50 on this recipe level at ×4 (every label's interval across zero) and 0.0019 lower at ×1, the same sign as their removal on the ResNet-18 (2026-10-08). Runs 11 and 12, both on the run-10 incumbent, are 0.0019 to 0.0020 below it at ×1 and 0.0003 to 0.0007 below at ×4: against this incumbent the ×1 gap is three times the ×4 gap, as it was the other way round against the run-5 incumbent (runs 7 to 9).

## 13. scanner-confusion-head-beside-l2sp (2026-10-13)

`trainer.scanner_confusion_weight` 0.1 (it was 0), L2-SP kept, on the OA-detector incumbent (run 10). A user seed.

- **×4:** 0.937043 against 0.938618, −0.001575, 80% interval [−0.002398, −0.000713], 95% interval [−0.002846, −0.000274]: the interval is below zero.
- **×1:** 0.929955 against 0.931247, −0.001292 (epoch 10). +0.0018 at epoch 2, then −0.0012 to −0.0027 from epoch 3 except epoch 5 (+0.0003).
- **Soft macro AUC:** −0.000963.
- **Cost:** training 2,638 s (1.01 of the incumbent), prediction 66 s, peak GPU memory 16.7 GB.
- **Decision:** not adopted (below the incumbent at both multipliers).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Medial Meniscus | +0.0002 | +0.0001 |
| Baker's | +0.0000 | −0.0008 |
| PF OA | −0.0002 | −0.0012 |
| MCL | −0.0010 | +0.0005 |
| Synovitis | −0.0011 | +0.0001 |
| Lateral Meniscus | −0.0012 | −0.0009 |
| Fracture | −0.0014 | −0.0023 |
| Medial OA | −0.0015 | −0.0009 |
| Effusion | −0.0025 | −0.0020 |
| Lateral OA | −0.0028 | −0.0001 |
| ACL | −0.0034 | −0.0034 |
| Contusion | −0.0042 | −0.0006 |

**Lesson:** beside L2-SP the scanner confusion head costs this recipe 0.0016 at ×4, with ten labels down and none up by more than 0.0002; the loss is spread, not on one label, and Effusion's interval is below zero. Unlike runs 11 and 12 the ×4 loss is as large as the ×1 loss, so this one is a loss of the recipe and not only of the incumbent's high ×1. The head's public gain of 2026-09-27 was measured without L2-SP; with L2-SP in place the two do not add.

## 14. view-label-bias-off (2026-10-13)

`model.view_label_bias` false, on the OA-detector incumbent (run 10). A user seed.

- **×4:** 0.938022 against 0.938618, −0.000596, 80% interval [−0.001303, +0.000130].
- **×1:** 0.929805 against 0.931247, −0.001442 (epoch 10). +0.0017 at epoch 1, level at epochs 5 and 6, −0.0011 to −0.0017 from epoch 7.
- **Soft macro AUC:** −0.000374.
- **Cost:** training 2,605 s (1.00 of the incumbent), prediction 65 s, peak GPU memory 16.7 GB.
- **Decision:** not adopted (below the incumbent at both multipliers).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Lateral OA | +0.0022 | +0.0011 |
| Effusion | +0.0007 | −0.0001 |
| Fracture | +0.0007 | +0.0004 |
| Contusion | +0.0005 | +0.0013 |
| Lateral Meniscus | +0.0004 | −0.0002 |
| Baker's | +0.0000 | −0.0010 |
| Medial Meniscus | −0.0005 | −0.0004 |
| ACL | −0.0007 | −0.0006 |
| Synovitis | −0.0008 | −0.0001 |
| PF OA | −0.0015 | −0.0013 |
| Medial OA | −0.0017 | −0.0007 |
| MCL | −0.0066 | −0.0028 |

**Lesson:** without the view label bias the recipe is 0.0006 lower at ×4, and MCL alone (−0.0066, 45 positives) is 0.0005 of that macro change; the other eleven labels sum to −0.0001. The bias stays. MCL has carried the largest per-label move in three of the five runs from run 10 on (+0.0044, −0.0053 and −0.0066; −0.0014 and −0.0010 in the other two): with 45 positives it moves the macro by up to 0.0005 on its own, which is the size of most macro differences in this campaign.

## 15. anatomical-attention-bias-back (2026-10-13)

The learned medial-lateral attention bias restored for the two meniscus and the two tibiofemoral OA labels, on the OA-detector incumbent (run 10). A user seed.

- **×4:** 0.938003 against 0.938618, −0.000615, 80% interval [−0.001297, +0.000059].
- **×1:** 0.929677 against 0.931247, −0.001570 (epoch 10). +0.0022 at epoch 1, level at epoch 3, −0.0010 to −0.0019 from epoch 6.
- **Soft macro AUC:** −0.000338.
- **Cost:** training 2,605 s (1.00 of the incumbent), prediction 66 s, peak GPU memory 16.7 GB.
- **Decision:** not adopted (below the incumbent at both multipliers).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Effusion | +0.0014 | +0.0006 |
| Contusion | +0.0004 | +0.0009 |
| Medial Meniscus | +0.0001 | +0.0002 |
| Baker's | +0.0001 | −0.0004 |
| PF OA | +0.0000 | −0.0002 |
| Fracture | −0.0002 | −0.0010 |
| Synovitis | −0.0005 | −0.0001 |
| Lateral OA | −0.0010 | −0.0003 |
| Lateral Meniscus | −0.0013 | −0.0010 |
| Medial OA | −0.0016 | −0.0009 |
| ACL | −0.0017 | −0.0009 |
| MCL | −0.0032 | −0.0010 |

**Lesson:** of the four labels the bias acts on, three lose (Medial OA −0.0016, Lateral Meniscus −0.0013, Lateral OA −0.0010) and Medial Meniscus is level: the routed heads already place these labels by the knee frame and the bias adds nothing on top, as the seed guessed. With run 9 (plane-contrast readouts off, level) both halves of the 2026-10-01 adoption are now read on the ResNet-50: neither moves it by more than 0.0006 at ×4. Runs 11 to 15, five changes on the run-10 incumbent, are all below it, by 0.0003 to 0.0016 at ×4 and 0.0013 to 0.0020 at ×1.

## 16. sagittal-fluid-detail-view-off (2026-10-13)

Training without the 64 mm second view of the sagittal fluid slices, on the OA-detector incumbent (run 10). A user seed.

- **×4:** 0.937424 against 0.938618, −0.001193, 80% interval [−0.002095, −0.000334] (below zero), 95% interval [−0.002649, +0.000099].
- **×1:** 0.928284 against 0.931247, −0.002963 (epoch 10). Ahead at epochs 1, 2 and 5, then −0.0009 to −0.0030, widening every epoch from 6.
- **Soft macro AUC:** −0.000979.
- **Cost:** training 2,343 s (0.90 of the incumbent), prediction 59 s (0.90), peak GPU memory 14.5 GB (16.7 GB with the view).
- **Decision:** not adopted (below the incumbent at both multipliers).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Baker's | +0.0021 | +0.0013 |
| Lateral Meniscus | +0.0015 | −0.0001 |
| Effusion | +0.0009 | −0.0001 |
| Lateral OA | −0.0003 | −0.0009 |
| PF OA | −0.0004 | −0.0012 |
| Medial Meniscus | −0.0008 | −0.0003 |
| ACL | −0.0010 | −0.0012 |
| Contusion | −0.0011 | −0.0009 |
| Fracture | −0.0011 | −0.0007 |
| Synovitis | −0.0021 | −0.0009 |
| Medial OA | −0.0023 | −0.0009 |
| MCL | −0.0098 | −0.0058 |

**Lesson:** on this recipe the 64 mm detail view is worth 0.0012 at ×4 and 0.0030 at ×1 for 10% of the training and prediction time and 2.2 GB of memory; it stays. On the plain ResNet-50 recipe the view and the routed cell read together had given +0.00004 at ×4; with the mask decoder and the OA detector in place it carries more. MCL loses most (−0.0098, interval below zero), a ligament read on coronal fluid, not on the view removed: nine labels fall, so the view feeds the shared features, not only the sagittal reads. Medial OA's interval is below zero too.

## 17. detail-view-embedding-off (2026-10-13)

Training without the learned vector added to the tokens of every view that is not at 120 mm, on the OA-detector incumbent (run 10). A user seed.

- **×4:** 0.938106 against 0.938618, −0.000512, 80% interval [−0.001275, +0.000224].
- **×1:** 0.930052 against 0.931247, −0.001195 (epoch 10). +0.0043 at epoch 1, within ±0.0004 at epochs 3, 4, 6 and 7, −0.0010 to −0.0013 from epoch 8.
- **Soft macro AUC:** −0.000159.
- **Cost:** training 2,618 s (1.01 of the incumbent), prediction 66 s, peak GPU memory 16.7 GB.
- **Decision:** not adopted (below the incumbent at both multipliers).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Effusion | +0.0010 | +0.0005 |
| Baker's | +0.0008 | +0.0007 |
| Lateral OA | +0.0004 | −0.0001 |
| ACL | +0.0000 | −0.0000 |
| Fracture | −0.0000 | −0.0001 |
| Synovitis | −0.0002 | +0.0001 |
| Lateral Meniscus | −0.0003 | −0.0002 |
| Medial Meniscus | −0.0006 | −0.0007 |
| Medial OA | −0.0015 | −0.0005 |
| PF OA | −0.0017 | −0.0009 |
| Contusion | −0.0017 | −0.0005 |
| MCL | −0.0024 | −0.0003 |

**Lesson:** the detail-view embedding is a small part: without it no label moves by more than 0.0024 and every interval crosses zero; the macro is 0.0005 lower at ×4, the smallest change of the six removals and restorations run on this incumbent. It stays, at no cost. Runs 11 to 17, seven changes on the run-10 incumbent, are all below it at both multipliers (×4 −0.0003 to −0.0016, ×1 −0.0012 to −0.0030).

## 18. gin-augmentation-off (2026-10-13)

`data.gin_augmentation.probability` 0 (it was 0.3), on the OA-detector incumbent (run 10). A user seed.

- **×4:** 0.939728 against 0.938618, +0.001111, 80% interval [−0.000198, +0.002423]. The highest ×4 of the campaign.
- **×1:** 0.929887 against 0.931247, −0.001359 (epoch 10). +0.0117 at epoch 1, +0.0036 at 2, +0.0019 at 5, +0.0008 at 7, then −0.0005, −0.0011 and −0.0014.
- **Soft macro AUC:** +0.000699.
- **Step from ×1 to ×4:** 0.009841 against the incumbent's 0.007371, 0.0025 larger.
- **Cost:** training 2,599 s (1.00 of the incumbent), prediction 65 s, peak GPU memory 16.7 GB.
- **Decision:** not adopted by the rule: above the incumbent at ×4, below at ×1.

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Fracture | +0.0079 | +0.0020 |
| Lateral Meniscus | +0.0039 | +0.0015 |
| Contusion | +0.0025 | +0.0017 |
| Synovitis | +0.0015 | −0.0002 |
| PF OA | +0.0014 | +0.0011 |
| Lateral OA | +0.0010 | +0.0012 |
| MCL | +0.0006 | +0.0005 |
| Medial Meniscus | −0.0003 | −0.0008 |
| Baker's | −0.0003 | −0.0001 |
| Medial OA | −0.0011 | −0.0004 |
| ACL | −0.0017 | +0.0008 |
| Effusion | −0.0019 | +0.0011 |

**Lesson:** without GIN the model is the best of the campaign at ×4 and soft AUC and 0.0014 lower at ×1: the two multipliers disagree by 0.0025, the largest split of the campaign, so what GIN changes is how much the model gains from reading more slices. The gain sits on the bone and meniscus labels (Fracture +0.0079, Lateral Meniscus +0.0039, Contusion +0.0025); Effusion loses at ×4 as it did when the intensity and noise augmentations were removed (runs 6 and 7), but not on soft AUC here. All eight runs on the run-10 incumbent (11 to 18) are below it at ×1 by 0.0012 to 0.0030, while their ×4 ranges from −0.0016 to +0.0011: the ×1 side of the rule does not separate them and the ×4 side does. Whether to adopt it by hand is the user's decision; ×4 is the multiplier of the submissions.

## 19. resolution-augmentation-off (2026-10-13)

`data.resolution_augmentation.probability` 0 (it was 0.25), on the OA-detector incumbent (run 10). A user seed.

- **×4:** 0.937574 against 0.938618, −0.001044, 80% interval [−0.002223, +0.000098].
- **×1:** 0.930560 against 0.931247, −0.000687 (epoch 10). +0.0104 at epoch 1, +0.0038 at 2, then between −0.0031 and +0.0019.
- **Soft macro AUC:** −0.000603.
- **Cost:** training 2,611 s (1.00 of the incumbent), prediction 66 s, peak GPU memory 16.7 GB.
- **Decision:** not adopted (below the incumbent at both multipliers).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Fracture | +0.0015 | +0.0000 |
| Lateral Meniscus | +0.0005 | +0.0001 |
| Baker's | +0.0003 | +0.0001 |
| Synovitis | +0.0000 | −0.0002 |
| PF OA | −0.0000 | −0.0005 |
| Medial Meniscus | −0.0002 | −0.0004 |
| Lateral OA | −0.0004 | +0.0009 |
| ACL | −0.0008 | −0.0005 |
| Effusion | −0.0019 | −0.0002 |
| Medial OA | −0.0021 | −0.0005 |
| Contusion | −0.0027 | −0.0017 |
| MCL | −0.0066 | −0.0045 |

**Lesson:** the one-axis resolution degradation is worth 0.0010 at ×4 on this recipe and stays; MCL carries 0.0006 of that (−0.0066). The two augmentations of the half stack removed alone on the ResNet-50 go opposite ways at ×4: GIN off +0.0011 (run 18), resolution degradation off −0.0010, both with the same large first-epoch lead (+0.0117 and +0.0104). Four augmentation removals are now read (runs 6, 7, 18, 19): Effusion falls at ×4 in all four (−0.0031, −0.0035, −0.0019, −0.0019), so every intensity or resolution augmentation helps Effusion, and Fracture rises in all four (+0.0012, +0.0052, +0.0079, +0.0015).

## 20. mask-decoder-loss-weight-equal-to-the-map-heads (2026-10-13)

The mask decoder's two losses at weight 0.3 (they were 0.1), on the OA-detector incumbent (run 10). The keeper's follow-up to run 5.

- **×4:** 0.937622 against 0.938618, −0.000996, 80% interval [−0.001690, −0.000317] (below zero), 95% interval [−0.002069, +0.000052].
- **×1:** 0.929725 against 0.931247, −0.001522 (epoch 10). Ahead at epochs 1, 2 and 5, level at 6, −0.0010 to −0.0017 from epoch 7.
- **Soft macro AUC:** −0.000718.
- **Cost:** training 2,600 s (1.00 of the incumbent), prediction 66 s, peak GPU memory 16.7 GB.
- **Decision:** not adopted (below the incumbent at both multipliers).
- **Activation, epoch 10 against the incumbent:** decoder bone loss 0.2669 against 0.2728, soft-tissue 0.1883 against 0.1926, `val_anatomy_loss` 0.6286 against 0.6340, `val_meniscus_loss` 0.5854 against 0.5939. `train_label_loss` 0.4339 (new metric, no incumbent value).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Lateral Meniscus | +0.0010 | −0.0012 |
| Baker's | +0.0007 | −0.0008 |
| Lateral OA | +0.0004 | +0.0001 |
| Medial OA | −0.0001 | −0.0003 |
| Fracture | −0.0005 | −0.0011 |
| Medial Meniscus | −0.0007 | −0.0003 |
| PF OA | −0.0008 | +0.0001 |
| ACL | −0.0014 | −0.0019 |
| MCL | −0.0016 | +0.0010 |
| Effusion | −0.0016 | −0.0012 |
| Contusion | −0.0035 | −0.0014 |
| Synovitis | −0.0039 | −0.0015 |

**Lesson:** three times the decoder weight made every map better again (decoder losses 2% lower, the 7 by 7 map 0.0054 lower, the 14 by 14 map 0.0085 lower) and the labels worse (−0.0010 at ×4, interval below zero, nine labels down). Better maps do not carry to AUC beyond what weight 0.1 gave: the audit's reading that the run-5 gain was a feature effect and not the maps' accuracy holds, and that effect is already saturated or over-driven at 0.1. The labels that lose most are the unrouted ones the decoder had helped in run 5 (Synovitis −0.0039, interval below zero; Effusion, ACL, MCL), so at 0.3 the side task takes capacity from them. The decoder weight stays at 0.1.

## 21. mask-decoder-stage-one-level (2026-10-13)

A 56 by 56 level added to the mask decoder from the first stage's map, with the mask heads on it. On the OA-detector incumbent (run 10). The keeper's follow-up to run 5.

- **×4:** 0.938075 against 0.938618, −0.000543, 80% interval [−0.001279, +0.000209].
- **×1:** 0.930124 against 0.931247, −0.001122 (epoch 10). Ahead at epochs 1, 2 and 5, −0.0006 to −0.0018 from epoch 6.
- **Soft macro AUC:** −0.000121.
- **Cost:** training 2,845 s (1.09 of the incumbent), prediction 66 s, peak GPU memory 17.4 GB allocated (18.0 GB on the card).
- **Decision:** not adopted (below the incumbent at both multipliers).
- **Activation:** decoder bone loss 1.1674 at epoch 1 to 0.2601 at epoch 10, still flat over epochs 7 to 10 (0.2546, 0.2573, 0.2573, 0.2601); soft-tissue 0.6971 to 0.1688. `val_anatomy_loss` 0.6351 against the incumbent's 0.6340, `val_meniscus_loss` 0.5940 against 0.5939. The supervised-cell metric reads 5.7 million per epoch, not the expected 57 million: check its reduction before reusing it.

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Fracture | +0.0036 | +0.0015 |
| Baker's | +0.0016 | +0.0007 |
| Lateral OA | +0.0007 | +0.0003 |
| Medial Meniscus | −0.0003 | −0.0001 |
| Contusion | −0.0004 | −0.0002 |
| Medial OA | −0.0006 | −0.0006 |
| ACL | −0.0011 | −0.0005 |
| Synovitis | −0.0013 | −0.0006 |
| Lateral Meniscus | −0.0013 | −0.0010 |
| Effusion | −0.0015 | −0.0006 |
| PF OA | −0.0017 | −0.0010 |
| MCL | −0.0041 | −0.0004 |

**Lesson:** the finer decoder level changed nothing it was aimed at: the coarse maps are as accurate as the incumbent's (7 by 7 and 14 by 14 losses within 0.001), the decoder's loss is flat from epoch 7 as at 28 by 28, and the labels are 0.0005 lower at ×4 for 9% more training time. With run 20 (decoder weight 0.3, −0.0010) the decoder line is read: the 28 by 28 decoder at weight 0.1 is where it gives what it gives; neither more weight nor finer cells adds. What is left untested on this line is the label path reading the decoder's features, which the contract keeps out of bounds.

## 22. squeeze-excitation-in-the-resnet-blocks (2026-10-13)

A squeeze-and-excitation module in each of the sixteen bottlenecks, started as the identity (gate 2·sigmoid, last layer zero), new parameters at the head rate without L2-SP. On the OA-detector incumbent (run 10). A user seed.

- **×4:** 0.937619 against 0.938618, −0.000998, 80% interval [−0.001972, −0.000012].
- **×1:** 0.929609 against 0.931247, −0.001638 (epoch 10). Ahead at epochs 1 to 5 (+0.0021 at epoch 5), behind from epoch 6.
- **Soft macro AUC:** −0.000287.
- **Cost:** training 2,973 s (1.14 of the incumbent), prediction 71 s (1.09), peak GPU memory 19.9 GB allocated, 20.1 GB on the card (16.7 GB for the incumbent); no spill, full power throughout.
- **Decision:** not adopted (below the incumbent at both multipliers).
- **Activation:** `val_se_gate_abs_deviation` 0.052 at epoch 1 to 0.122 at epoch 10: the gates moved 12% from 1 on average.

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Lateral Meniscus | +0.0027 | +0.0017 |
| ACL | +0.0017 | −0.0004 |
| PF OA | +0.0009 | +0.0006 |
| Effusion | +0.0007 | −0.0001 |
| Contusion | +0.0006 | +0.0003 |
| Baker's | −0.0002 | +0.0004 |
| Synovitis | −0.0008 | −0.0002 |
| Lateral OA | −0.0014 | +0.0005 |
| Medial Meniscus | −0.0018 | −0.0007 |
| Medial OA | −0.0020 | −0.0003 |
| Fracture | −0.0020 | −0.0012 |
| MCL | −0.0105 | −0.0040 |

**Lesson:** MCL alone (−0.0105, interval below zero, 45 positives) is 0.0009 of the 0.0010 macro loss at ×4; the other eleven labels sum to −0.0001. So the SE network is level with the plain one on eleven labels and loses MCL. Signs of what the gates did: the training loss is lower at every epoch (0.5672 against 0.5717 at epoch 10) while the validation loss is not (0.3983 against 0.3979), and the L2-SP penalty is lower (0.0191 against 0.0210): the unanchored gates at the head rate absorbed adaptation the anchored convolutions would have made, and fitted the training set a little more. Candidate follow-ups, one change each: the gates at the backbone rate; SE in stages three and four only (also 5% memory instead of 19%).

## 23. global-context-in-the-resnet-blocks (2026-10-13)

An additive global-context block in the thirteen bottlenecks of stages two to four, started at zero, new parameters free at the head rate without L2-SP. On the OA-detector incumbent (run 10). A user seed.

- **×4:** 0.938277 against 0.938618, −0.000341, 80% interval [−0.001423, +0.000722].
- **×1:** 0.930657 against 0.931247, −0.000590 (epoch 10), the closest ×1 to this incumbent of the runs built on it. −0.0052 at epoch 1, behind through epoch 6, level at epochs 7 and 8.
- **Soft macro AUC:** −0.000351.
- **Cost:** training 2,897 s (1.11 of the incumbent), prediction 70 s (1.08), peak GPU memory 18.5 GB allocated, 18.7 GB on the card.
- **Decision:** not adopted (below the incumbent at both multipliers).
- **Logged values at epoch 10, against the incumbent and run 22:** `val_global_context_rms_ratio` 0.487 (0.526 at epoch 1): the added term is about half the size of the block output it is added to. L2-SP penalty 0.0184 (incumbent 0.0210, run 22 0.0191). Stage-four drift 0.0968 (0.1068, 0.0986). Training loss 0.5664 (0.5717, 0.5672). Validation loss 0.3975 (0.3979, 0.3983).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| PF OA | +0.0026 | +0.0013 |
| Baker's | +0.0025 | +0.0015 |
| Synovitis | +0.0020 | +0.0007 |
| Lateral Meniscus | +0.0019 | −0.0011 |
| Lateral OA | +0.0003 | +0.0012 |
| ACL | −0.0008 | −0.0010 |
| MCL | −0.0010 | −0.0004 |
| Fracture | −0.0010 | −0.0018 |
| Effusion | −0.0011 | −0.0006 |
| Medial OA | −0.0015 | +0.0000 |
| Medial Meniscus | −0.0024 | −0.0014 |
| Contusion | −0.0057 | −0.0027 |

**Lesson:** level with the incumbent on the macro (every multiplier inside its interval), with Contusion (−0.0057) and Medial Meniscus (−0.0024) below zero on their intervals. The free block is large from the first epoch and the anchored weights move less with it (lowest L2-SP penalty and stage-four drift of the three runs, lowest training loss): the same pattern as run 22's gates, stronger. The user's reading on 2026-10-13: this works against what L2-SP is in the recipe for, and one fold cannot show the cost. Consequence for the campaign: modules added inside the backbone are pulled toward the identity at the L2-SP strength (seeds `global-context-pulled-to-identity`, `coordinate-attention-late-stages-pulled-to-identity`, `squeeze-excitation-late-stages-pulled-to-identity`).

## 24. soft-prevalence-class-weights-on-the-incumbent (2026-10-13)

Run 1's change (class weights from the mean soft target) ported unchanged onto the OA-detector incumbent (run 10). The seed that applies the user's adoption decision after S150 against S151.

- **×4:** 0.937680 against 0.938618, −0.000938, 80% interval [−0.001728, −0.000183] (below zero), 95% interval [−0.002146, +0.000224].
- **×1:** 0.929182 against 0.931247, −0.002065 (epoch 10). +0.0062 at epoch 1, crossing at epochs 3, 4 and 6, −0.0016 to −0.0022 from epoch 6.
- **Soft macro AUC:** +0.000018.
- **Cost:** training 2,643 s (1.02 of the incumbent), peak GPU memory 16.7 GB.
- **Activation:** the twelve class weights equal run 1's to every digit (MCL positive weight 2.52 against the incumbent's 3.66).
- **Decision:** rejected by the rule. **Not adopted by hand:** the user's adoption decision was taken on the public pair and on run 1's local gain; this run is a local loss at ×4 with the interval below zero, which the decision did not have before it. The keeper resumed the campaign on the unchanged incumbent and put the result to the user.

| Label | ×4 Δ here | ×4 Δ in run 1 |
|---|---:|---:|
| Lateral Meniscus | +0.0016 | −0.0000 |
| Fracture | +0.0009 | +0.0042 |
| Medial Meniscus | +0.0008 | +0.0001 |
| Baker's | +0.0008 | +0.0009 |
| Effusion | +0.0008 | +0.0017 |
| PF OA | +0.0007 | −0.0028 |
| ACL | −0.0012 | −0.0005 |
| Lateral OA | −0.0015 | +0.0014 |
| Medial OA | −0.0015 | +0.0009 |
| Contusion | −0.0024 | +0.0031 |
| MCL | −0.0050 | +0.0007 |
| Synovitis | −0.0055 | −0.0036 |

**Lesson:** the same weights on two bases give +0.0005 on the reference and −0.0009 on the incumbent with the mask decoder and the OA detector. What repeats: Synovitis falls in both (−0.0036, −0.0055), the label whose positive weight is cut most (2.16 to 1.29), and Fracture, Effusion and Baker's rise in both, by less here. What does not repeat: Contusion (+0.0031 then −0.0024), MCL (+0.0007 then −0.0050) and the two tibiofemoral OA labels turn negative. Contusion and MCL are labels the mask decoder raised in run 5 (+0.0021, +0.0031) and the OA labels are the OA detector's: candidate cause, the lower positive weights take back part of what those two additions gave on the rare labels. Soft AUC is level (+0.00002), as it was up in run 1 (+0.0009). Public evidence (S150 0.960 against S151 0.959) is for the weights on the reference recipe only.

**Adoption of run 24 dropped (user, 2026-10-13):** the weights aim at the same labels the mask decoder and the OA detector already raised, so on this recipe they are not adopted. They remain a publicly confirmed gain (+0.001, S150 against S151) on the recipe without those two additions.

## 25. global-context-pulled-to-identity (2026-10-13)

Run 23's global-context blocks with a pull toward zero on each block's last convolution at the L2-SP strength (0.001) and no weight decay on the new parameters. On the OA-detector incumbent (run 10). A user seed.

- **×4:** 0.938402 against 0.938618, −0.000216, 80% interval [−0.001110, +0.000621].
- **×1:** 0.930128 against 0.931247, −0.001119 (epoch 10). Level at epochs 2 and 4, +0.0012 at 5, −0.0007 to −0.0012 from epoch 6.
- **Soft macro AUC:** −0.000598.
- **Cost:** training 2,896 s (1.11 of the incumbent), prediction 71 s, peak GPU memory 18.5 GB allocated.
- **Decision:** not adopted (below the incumbent at both multipliers).
- **The pull, per epoch:** `val_global_context_rms_ratio` 0.392, 0.218, 0.127, 0.102, 0.085, 0.076, 0.072, 0.068, 0.066, 0.066 (the free run: 0.526 to 0.487). Identity penalty 0.0050 at epoch 1 to 0.0008 at epoch 10. L2-SP penalty 0.0203 (incumbent 0.0210, free run 0.0184), stage-four drift 0.1039 (0.1068, 0.0968), training loss 0.5721 (0.5717, 0.5664).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Fracture | +0.0036 | +0.0006 |
| Synovitis | +0.0016 | +0.0001 |
| Lateral Meniscus | +0.0014 | −0.0010 |
| PF OA | +0.0008 | −0.0001 |
| Medial OA | −0.0001 | +0.0001 |
| Medial Meniscus | −0.0004 | −0.0004 |
| Lateral OA | −0.0004 | +0.0008 |
| Baker's | −0.0006 | −0.0011 |
| Effusion | −0.0012 | −0.0010 |
| ACL | −0.0019 | −0.0009 |
| MCL | −0.0020 | −0.0022 |
| Contusion | −0.0035 | −0.0020 |

**Lesson:** the pull works as a mechanism: the block's added term falls from 0.39 of the feature size to 0.066 and the anchored weights, the L2-SP penalty and the training loss return to the incumbent's values. With the block held that small the model is the incumbent again (×4 −0.0002, every label's interval across zero). Runs 23 and 25 together: free, the block is large and the model is level locally; held near the identity, the block is small and the model is level. Global context adds nothing to this recipe either way, for 11% more training time and 1.8 GB. Contusion is the largest loss in both (−0.0057, −0.0035).

## 26. coordinate-attention-late-stages-pulled-to-identity (2026-10-13)

Coordinate attention in the nine bottlenecks of stages three and four, started as the identity, with a pull toward the identity on its two output convolutions at the L2-SP strength (0.001) and no weight decay. On the OA-detector incumbent (run 10). A user seed.

- **×4:** 0.937951 against 0.938618, −0.000667, 80% interval [−0.001384, +0.000033].
- **×1:** 0.930072 against 0.931247, −0.001175 (epoch 10).
- **Soft macro AUC:** −0.000501.
- **Cost:** training 2,885 s (1.11 of the incumbent), prediction 71 s, peak GPU memory 18.9 GB allocated, 19.4 GB on the card.
- **Decision:** not adopted (below the incumbent at both multipliers).
- **The gates did not open:** `coordinate_gate_deviation` 0.0083 at epoch 1, falling every epoch to 0.0004 at epoch 10; the identity penalty falls from 0.0023 to 0.00009. L2-SP penalty 0.0211 (incumbent 0.0210), stage-four drift 0.1070 (0.1068), training loss 0.5718 (0.5717).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| Contusion | +0.0006 | +0.0003 |
| Lateral Meniscus | +0.0004 | −0.0005 |
| ACL | +0.0002 | −0.0011 |
| PF OA | +0.0000 | −0.0001 |
| Medial OA | −0.0001 | −0.0003 |
| Effusion | −0.0003 | −0.0004 |
| Fracture | −0.0005 | −0.0003 |
| Baker's | −0.0005 | −0.0008 |
| Lateral OA | −0.0007 | −0.0003 |
| Medial Meniscus | −0.0009 | −0.0005 |
| Synovitis | −0.0024 | −0.0008 |
| MCL | −0.0037 | −0.0011 |

**Lesson:** this run did not test coordinate attention: at this pull the gates stay within 0.8% of 1 and close to 0.04% by the end, so the network that trained is the incumbent's recipe with an inert module. Its anchor penalty, drift and training loss equal the incumbent's to the third decimal. **That makes it the nearest thing to a retraining of the incumbent recipe the campaign has: ×4 0.937951 and ×1 0.930072, 0.0007 and 0.0012 below the incumbent's own numbers.** It supports reading the incumbent's ×1 (0.931247) and ×4 (0.938618) as high for its recipe by about those amounts, and re-reads the fifteen runs built on it: against this run GIN off is +0.0018 at ×4, run 23 (free global context) +0.0003, run 25 (pulled global context) +0.0005, runs 11 to 17 between −0.0009 and +0.0004. Caveat: one run, and the module's presence changes memory layout and the forward pass by rounding only, not by design. For the coordinate-attention idea itself: the pull at 0.001 on a zero-started gate leaves it no room; a weaker pull would test it and is a generalization risk the user has ruled out adopting.

## 27. average-pool-shortcuts-in-the-resnet (2026-10-13)

The three stride-two shortcuts of the ResNet-50 as (2 by 2 average pool, the same 1 by 1 convolution at stride 1, batch norm), ResNet-D style, on the same weights with the keys renamed. No new parameter. On the OA-detector incumbent (run 10). A user seed.

- **×4:** 0.939634 against 0.938618, +0.001016, 80% interval [+0.000172, +0.001885] (above zero), 95% interval [−0.000265, +0.002336].
- **×1:** 0.930902 against 0.931247, −0.000345 (epoch 10), the closest ×1 to this incumbent of the seventeen runs built on it.
- **Soft macro AUC:** +0.000244.
- **Cost:** training 2,696 s (1.04 of the incumbent), prediction 68 s, peak GPU memory 16.9 GB.
- **Decision:** rejected by the rule: above the incumbent at ×4, below at ×1 by 0.0003.
- **Anchor and fit:** L2-SP penalty 0.0207 (incumbent 0.0210), stage-four drift 0.1056 (0.1068), training loss 0.5710 (0.5717): the anchored weights stay where the incumbent's are; nothing unanchored was added.

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| MCL | +0.0032 | +0.0014 |
| Fracture | +0.0030 | +0.0010 |
| Lateral OA | +0.0027 | +0.0012 |
| Contusion | +0.0021 | +0.0009 |
| Baker's | +0.0018 | +0.0004 |
| Medial Meniscus | +0.0007 | +0.0006 |
| Synovitis | +0.0006 | −0.0001 |
| Effusion | −0.0001 | +0.0003 |
| Lateral Meniscus | −0.0002 | −0.0011 |
| ACL | −0.0002 | −0.0003 |
| Medial OA | −0.0005 | −0.0001 |
| PF OA | −0.0009 | −0.0012 |

**Lesson:** the second run of the campaign with the ×4 interval above zero (after run 10), from a change with no new parameters and the L2-SP anchor intact. Seven labels rise and none falls by more than 0.0009. Against run 26, the near-retraining of the incumbent recipe (×4 0.937951, ×1 0.930072), it is +0.0017 at ×4 and +0.0008 at ×1. The rule rejects it on a ×1 gap of 0.0003 to an incumbent whose ×1 reads about 0.0012 high for its recipe. Whether to adopt it by hand is the user's decision; it carries none of the generalization risk of the free modules (it averages where the plain shortcut discards three positions in four).

**Run 27 adopted by the user (2026-10-13 16:06 CEST):** the incumbent is the average-pool-shortcuts run (×4 0.939634, ×1 0.930902), on top of the OA detector and the mask decoder. The campaign was stopped with the running jobs cancelled to apply it: the first blur-pool run was cancelled 275 s into training and is recorded invalid without a result; the seed is requeued on the new incumbent as `blur-pool-at-the-stage-transitions-on-average-pool-shortcuts` ([thirteenth seeds](seed-ideas-13.yaml)). Against the campaign's starting reference the incumbent is +0.0026 at ×4 and +0.0024 at ×1.

## 28. squeeze-excitation-late-stages-pulled-to-identity (2026-10-13)

Squeeze-and-excitation in the nine bottlenecks of stages three and four, started as the identity, pulled toward it at the L2-SP strength, no weight decay. On the average-pool incumbent (run 27). A user seed; the user closed the SE line after it (no third try).

- **×4:** 0.938639 against 0.939634, −0.000994, 80% interval [−0.001610, −0.000421], 95% interval [−0.001952, −0.000122] (below zero).
- **×1:** 0.929592 against 0.930902, −0.001310 (epoch 10).
- **Soft macro AUC:** −0.000260.
- **Cost:** training 2,812 s (1.04 of the incumbent), prediction 70 s, peak GPU memory 17.8 GB allocated, 18.2 GB on the card.
- **Decision:** not adopted (below the incumbent at both multipliers).
- **The gates stayed shut:** `val_se_gate_abs_deviation` 0.0022, 0.0005, 0.0001, then of the order of 0.00005 to the end. L2-SP penalty 0.0206 (incumbent 0.0207), training loss 0.5714 (0.5710).

| Label | ×4 Δ | Soft Δ |
|---|---:|---:|
| ACL | +0.0017 | +0.0008 |
| Lateral Meniscus | +0.0013 | +0.0020 |
| PF OA | +0.0007 | +0.0003 |
| Effusion | +0.0004 | +0.0000 |
| Synovitis | +0.0003 | +0.0006 |
| Medial Meniscus | +0.0002 | +0.0003 |
| Fracture | −0.0009 | −0.0016 |
| Contusion | −0.0014 | +0.0002 |
| Medial OA | −0.0019 | −0.0004 |
| Baker's | −0.0020 | −0.0013 |
| MCL | −0.0052 | −0.0021 |
| Lateral OA | −0.0053 | −0.0021 |

**Lesson:** not a test of squeeze-and-excitation: the gates are 1 to four decimals, so the network that trained is the run-27 recipe with nine modules that multiply by one. Read as a repeat of that recipe it is 0.0010 below the incumbent at ×4, with the paired interval below zero, and 0.0013 below at ×1. Run 26 was the same kind of repeat of the run-10 recipe and came out 0.0007 and 0.0012 below run 10. **Two inert-module repeats, two incumbents, both 0.0007 to 0.0010 below at ×4 and 0.0012 to 0.0013 below at ×1.** Two readings fit: (a) each adopted run is the upper draw of its recipe, because adoption selects high runs, and a repeat of this ResNet-50 recipe differs by about 0.001 at ×4, three times the ±0.0003 taken from the ResNet-18; (b) an inert module in the block is not free, for instance through its effect on memory format, mixed-precision rounding or the initialisation stream. Repeat against repeat, the average-pool shortcuts are +0.0007 at ×4 (0.938639 against run 26's 0.937951) and −0.0005 at ×1 (0.929592 against 0.930072): smaller and less clear than run 27's own +0.0010 and −0.0003. The labels that lose most here, MCL and Lateral OA, are the two that gained most in run 27 (+0.0032, +0.0027): the repeat gives those gains back.

## 29. blur-pool-at-the-stage-transitions-on-average-pool-shortcuts (2026-10-13)

Blur-pool on the three stride-two 3 by 3 convolutions (convolution at stride 1, then a fixed 3 by 3 blur at stride 2), stem and shortcuts as the incumbent has them. On the average-pool incumbent (run 27). A user seed; its first run was cancelled for the adoption of run 27 and this is the requeued one.

- **×4:** 0.938444 against 0.939634, −0.001189, 80% interval [−0.001997, −0.000369] (below zero), 95% interval [−0.002453, +0.000027].
- **×1:** 0.930305 against 0.930902, −0.000597 (epoch 10).
- **Soft macro AUC:** −0.000400.
- **Cost:** training 3,053 s (1.13 of the incumbent, inside the 3,364 s ceiling; the first epoch alone was slow), ×4 prediction 154 s (68 s for the incumbent), peak GPU memory 18.0 GB allocated.
- **Decision:** below the incumbent at both multipliers. The campaign was stopped by the user as this run ended, so the runtime's decision is still pending in the state; the figures are from the run's `evaluation.json`.

| Label | ×4 Δ |
|---|---:|
| ACL | +0.0006 |
| Synovitis | +0.0005 |
| PF OA | +0.0001 |
| Medial Meniscus | −0.0003 |
| Effusion | −0.0004 |
| Medial OA | −0.0012 |
| Contusion | −0.0012 |
| MCL | −0.0019 |
| Fracture | −0.0020 |
| Baker's | −0.0024 |
| Lateral Meniscus | −0.0030 |
| Lateral OA | −0.0031 |

**Lesson:** blur-pool on top of the average-pool shortcuts gives nothing: against the incumbent −0.0012 at ×4, and against run 28, the inert repeat of the incumbent's recipe (×4 0.938639, ×1 0.929592), −0.0002 at ×4 and +0.0007 at ×1, for 13% more training time and more than twice the ×4 prediction time. The thin structures it was aimed at do not gain (Lateral Meniscus −0.0030, MCL −0.0019, ACL +0.0006).

## Where the campaign stands (stopped by the user, 2026-10-13 17:58 CEST)

Twenty-nine runs on fold 2. **Incumbent: run 27, the average-pool shortcuts, on top of the OA detector (run 10) and the mask decoder (run 5): ×4 0.939634, ×1 0.930902**, against the starting reference's 0.937000 and 0.928460.

- **Adopted:** mask decoder at 28 by 28, weight 0.1 (run 5); the three-compartment OA detector beside the routed shared OA head (run 10); average-pool shortcuts in the ResNet-50 (run 27, by the user).
- **Publicly read:** soft-prevalence class weights on the reference recipe, S150 0.960 against S151 0.959, +0.001; not adopted, because on the incumbent they lose 0.0009 (run 24) on the labels the decoder and detector already raised.
- **Rejected with the interval below zero at ×4:** hard report targets (run 4), the scanner confusion head beside L2-SP (13), the detail view removed (16, so the view stays), decoder weight 0.3 (20).
- **Level or within the spread:** stable intensity bounds, uncertain targets weighing less, the hard-pair phase, plane-contrast readouts off, the anatomical attention bias back, the view label bias off, the detail-view embedding off, the report-attribute loss on, the 56 by 56 decoder level, global context free or pulled, blur-pool.
- **Augmentations:** removing intensity, noise or resolution augmentation costs Effusion and loses or ties; removing GIN gave the best ×4 of its incumbent (+0.0011) with ×1 below, left unadopted by the user as a generalization risk.
- **Backbone modules:** free of any anchor they move the function far from the pretrained one while L2-SP reads lower (SE gates 12% from 1, global-context term 0.49 of the features); pulled toward the identity at the L2-SP strength the gates shut (SE, coordinate attention) or the block is held to 7% (global context), and the model is the incumbent again. The user's rule: no adoption of a change that risks generalization.
- **The yardstick:** two inert-module repeats landed 0.0007 and 0.0010 below their incumbents at ×4 and 0.0012 and 0.0013 below at ×1. A repeat of this ResNet-50 recipe differs by about 0.001 at ×4, not the ±0.0003 taken from the ResNet-18, or an adopted run is the high draw of its recipe; either way most single-run differences of this campaign are inside that distance, and the both-multipliers rule rejected sixteen runs in a row against one high incumbent.
- **Open:** the public score of the incumbent (no submission of it yet); the reference's PF OA (0.9155, above 25 of the 26 campaign runs), unexplained.
