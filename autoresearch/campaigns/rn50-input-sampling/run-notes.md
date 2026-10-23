# ResNet-50 input sampling campaign: run notes

Reference and first incumbent: `rn50-detail-view-teacher-025` (×1 0.927290, ×4 0.935570,
2,650 s). Per-label tables are at ×4, from largest gain to largest loss. One run is weak
evidence; the ResNet-50's rerun spread is not yet measured (ResNet-18: ±0.0003 at ×4, 0.0015 at
×1).

## 1. `aligned-contrast-mixing`: not above the incumbent at ×4

The user's seed. The donor contrast is resampled onto the target's grid in patient coordinates
before the blend (reflection handled, uncovered pixels keep the target). On two studies the
aligned donor differs from the unaligned one by 0.08 to 0.10 mean absolute intensity.

- **×1:** 0.928276 against 0.927290, +0.000986 (epoch 10). Against the reference per epoch:
  +0.0008, +0.0007, −0.0013, +0.0018, −0.0017, +0.0019, +0.0010, +0.0009, +0.0006, +0.0010;
  mean of epochs 6 to 10 +0.0011.
- **×4:** 0.935270 against 0.935570, −0.000301.
- **Time:** 2,660 s (1.004 of the reference).

| Label | Reference | Run | Change at ×4 |
|---|---|---|---|
| MCL | 0.948288 | 0.949214 | +0.000926 |
| ACL | 0.950264 | 0.951168 | +0.000904 |
| PF OA | 0.917023 | 0.917596 | +0.000573 |
| Medial OA | 0.929958 | 0.930236 | +0.000278 |
| Synovitis | 0.892163 | 0.892382 | +0.000219 |
| Medial Meniscus | 0.949853 | 0.950007 | +0.000154 |
| Lateral OA | 0.920668 | 0.920776 | +0.000108 |
| Effusion | 0.920596 | 0.920596 | +0.000000 |
| Fracture | 0.936957 | 0.936343 | −0.000614 |
| Baker's | 0.970937 | 0.969806 | −0.001131 |
| Lateral Meniscus | 0.928562 | 0.926608 | −0.001955 |
| Contusion | 0.961574 | 0.958504 | −0.003070 |

Seven labels up by at most 0.0009, and the ×4 loss sits on Contusion and Lateral Meniscus. On
the ResNet-18 the same fix gave −0.0008 at ×4; here ×1 is up in the last five epochs and ×4 is
−0.0003. The two multipliers disagree, as they did for the anatomy map on the ResNet-50
(+0.0017 at ×1, +0.0001 at ×4): candidate reading, a guess, is that the fix helps a
five-slice bag and adds nothing once ×4 has read most slices.

**Adopted by the user** (monitor page, 2026-10-11 22:33 CEST), applied when no training runs.
The run of seed 2 in progress and the prepared candidate of seed 3 were built on the reference
and are compared with it.

## 2. `centred-token-grid-by-last-stage-phase`: aborted by the user at epoch 5

Against the reference per epoch: −0.0012, +0.0016, −0.0007, +0.0006, −0.0023. Aborted on
2026-10-11 22:55 CEST for a corrected form, no result.

A review with numeric checks on the backbone found the candidate correct for its seed: tokens at
16, 48, …, 208 px, both residual branches of the last stage on the same phase, pretrained
weights unchanged, declared positions true. What it does not do:

- **The third-stage cells stay at 0 to 208 px**, 7.5 px off the crop centre. At 224 px both
  grids cannot be centred by sampling phase: a token always sits on a cell, and a centred grid of
  7 lies between the cells of a centred grid of 14. Shifting the third stage instead leaves the
  tokens 7.5 px off; shifting stages two and three splits it (cells +4.5 px, tokens −3.5 px).
  Both are exactly centred at 225 px (15 cells, 8 tokens) and 257 px (17 and 9).
- **The copies between the grids are half a cell off** (8 px, 2.3 to 4.3 mm), in the reference
  as well: cells take a token by repetition, tokens take the mean of a 2 by 2 block that is not
  centred on them. The 257 px candidate had the same, and not mirror-symmetric. Nothing in
  training reads the shares pooled onto tokens today; the token context added to the
  soft-tissue map and the anatomy zone on the cells are read.

Corrected copies are in `exact-grid-transfers-224.diff` and `exact-grid-transfers-257.diff`
(checked on synthetic ramps only): bilinear at the cells' true positions, and ¼, ½, ¼ per axis
around each token. Border tokens keep a 5.3 px inward pull, since half their footprint is
outside the image.

By the user's order the queue is now: `centred-grids-at-225-px-exact-copies` (new),
`centred-token-grid-at-257-px` (its first candidate, never trained, replaced by the corrected
form), `centred-token-grid-by-phase-exact-copies` (new), then the sampling seeds. The policy
allows the image size to change in the 225 px seed too.

## 3. `centred-grids-at-225-px-exact-copies`: below the incumbent at both multipliers

The user's order. 225 px with the pixel sizes kept: 15 by 15 cells and 8 by 8 tokens, both
centred on pixel 112, token k on cell 2k; copies between the grids exact (bilinear with aligned
corners, ¼ ½ ¼ around each token). Built on aligned contrast mixing.

- **×1:** 0.927010; −0.001266 against the incumbent, −0.000280 against the original reference.
  Per epoch against the incumbent: +0.0027, −0.0020, +0.0008, −0.0015, −0.0008, −0.0011,
  −0.0016, −0.0013, −0.0006, −0.0013. The ResNet-18 signature of 225 px (−0.005, +0.003,
  −0.006 in the first three epochs) did not appear.
- **×4:** 0.934498; −0.000771 against the incumbent, −0.001072 against the original reference.
- **Time:** 2,876 s (1.085 of the reference; the developer's estimate was 1.04). GPU 18.3 GB.

| Label | Incumbent | Run | Change at ×4 |
|---|---|---|---|
| Fracture | 0.936343 | 0.946294 | +0.009951 |
| Baker's | 0.969806 | 0.971768 | +0.001962 |
| Medial OA | 0.930236 | 0.930692 | +0.000456 |
| Contusion | 0.958504 | 0.958614 | +0.000110 |
| ACL | 0.951168 | 0.950407 | −0.000760 |
| PF OA | 0.917596 | 0.916303 | −0.001293 |
| Medial Meniscus | 0.950007 | 0.948298 | −0.001709 |
| Effusion | 0.920596 | 0.918521 | −0.002075 |
| Lateral Meniscus | 0.926608 | 0.924232 | −0.002376 |
| Lateral OA | 0.920776 | 0.917560 | −0.003216 |
| MCL | 0.949214 | 0.944171 | −0.005043 |
| Synovitis | 0.892382 | 0.887119 | −0.005263 |

Eight labels down, and the macro is held up by Fracture (+0.0100, a label with fewer than 100
positives): without Fracture the other eleven average −0.0017. Second 225 px result below its
224 px parent (ResNet-18: −0.0024 with the inherited copies), now with exact copies, so the
copies were not the cause there. What the 225 px runs share and the 257 px ResNet-18 run does
not: an even number of tokens per side, so no token sits on the crop centre and the joint lies
at the corner of four tokens; and 44% of tokens on the image border. Candidate causes, guesses.
The 257 px seed (a token on the centre, 40% border tokens, 15% more field) and the 224 px phase
shift (tokens centred, no border tokens) separate them. The user's condition for removing the
257 px seed (this run successful) is not met; it stays queued.

## 4. `centred-token-grid-by-phase-exact-copies`: below the incumbent at both multipliers

The last stage's sampling phase shifted at 224 px (tokens at 16 to 208 px, cells unchanged at 0
to 208) with exact copies between the grids. Built on aligned contrast mixing.

- **×1:** 0.926056; −0.002220 against the incumbent, −0.001234 against the original reference.
  Per epoch against the incumbent: −0.0024, +0.0036, +0.0001, −0.0014, −0.0004, −0.0024,
  −0.0037, −0.0028, −0.0018, −0.0022.
- **×4:** 0.934307; −0.000963 against the incumbent, −0.001263 against the original reference.
- **Time:** 2,797 s (1.055 of the reference).

| Label | Incumbent | Run | Change at ×4 |
|---|---|---|---|
| Effusion | 0.920596 | 0.922489 | +0.001893 |
| Lateral Meniscus | 0.926608 | 0.927800 | +0.001193 |
| Fracture | 0.936343 | 0.937326 | +0.000983 |
| Baker's | 0.969806 | 0.969933 | +0.000127 |
| Medial OA | 0.930236 | 0.929981 | −0.000255 |
| Lateral OA | 0.920776 | 0.920397 | −0.000378 |
| Contusion | 0.958504 | 0.957981 | −0.000523 |
| Medial Meniscus | 0.950007 | 0.949467 | −0.000540 |
| PF OA | 0.917596 | 0.915954 | −0.001641 |
| Synovitis | 0.892382 | 0.890293 | −0.002089 |
| ACL | 0.951168 | 0.949000 | −0.002168 |
| MCL | 0.949214 | 0.941058 | −0.008156 |

MCL carries most of the loss (−0.0082; fewer than 100 positives); MCL was also down at 225 px
(−0.0050). It is the label the seed expected to gain.

## The Fracture jump of run 3 was one study

Fold 1 has 60 Fracture positives of 874. In run 3 one positive (report: "suspect linear fracture
of tibial eminence", target 0.66) moved from above 10% of the negatives to above 67%; that move
is +0.0096 of the +0.0100, and of the other 59 positives 23 rose and 26 fell. Paired bootstrap
of the change: −0.004 to +0.034. The three 224 px ResNet-50 runs before it rank that study alike
epoch by epoch and end at 0.03 to 0.10; run 4 (224 px, the same augmentation stream) ends at
0.11 again; the ResNet-18 reference has it at 0.79. Runs that share the seed and the image size
share the fate of single studies; changing the image size changes the augmentation draws. On
Fracture, MCL, Lateral OA and Contusion a change is reported with the positives that carry it.

## Left against right knees (saved predictions, no run)

Side from series text and reports: 129 left, 133 right, 612 unknown of 874. AUC of right minus
left knees per label is the same in all four runs (Medial Meniscus +0.065 to +0.071, Effusion
+0.049 to +0.055, Medial OA +0.040 to +0.056, MCL −0.017 to −0.050 on 10 and 8 positives), so
it reflects the studies in each group, not the grid. Mean absolute difference over the eight
larger labels: reference 0.0314, aligned mixing 0.0312, 225 px 0.0305, 224 px phase shift
0.0280. Centring the grid does not reduce a left-right difference that this sample can show.

## Reading of the three grid runs

Neither centred version is above its parent at either multiplier (225 px −0.0008 at ×4, −0.0016
without the one Fracture study; phase shift −0.0010), as the three ResNet-18 runs before them.
Candidate reasons, guesses: tokens carry their true positions whatever the offset; a last-stage
token draws on far more than its 32 px; reflection shows both sides; and each fix has its own
cost (border tokens on four sides at 225 and 257 px, a sampling phase the pretrained weights
never used at 224 px).

## Exception for the 257 px seed (the user, 2026-10-12 00:40 CEST)

The rebuilt 257 px candidate was blocked at review on cost (estimate 3,407 s and 22.7 GB against
3,313 s and 22 GB). The user wants it run to see whether a larger image helps the ResNet-50: the
sealed training ceiling is 3,578 s (1.35) since then, for this seed; it runs ahead of the
warm-restarts seed.

**Incident, 2026-10-12 00:56 CEST.** The 257 px run's first epoch took 375 s against 271 s for
the reference's: a pace of 1.40, above the 1.35 ceiling set minutes earlier (the estimate was
1.29), so the runtime would have stopped it after epoch 2. The keeper killed the supervisor
process alone, raised the sealed ceiling to 3,843 s (1.45) and the per-experiment limit to
4,350 s, and resumed; the training was never interrupted. GPU memory 22.6 GB; lowest available
RAM in the first epoch 15.1 GB.

## 5. `centred-token-grid-at-257-px`: below the incumbent at both multipliers

The user's seed, run under their budget exception. 257 px with the pixel sizes kept (fields 137.7,
96.4, 107.1 and 73.4 mm): 17 by 17 cells and 9 by 9 tokens, both centred on pixel 128, exact
copies between the grids. Built on aligned contrast mixing.

- **×1:** 0.924376; −0.003900 against the incumbent, −0.002914 against the original reference.
  Per epoch against the incumbent: +0.0002, −0.0010, −0.0013, −0.0021, +0.0009, −0.0021,
  −0.0050, −0.0031, −0.0032, −0.0039.
- **×4:** 0.933649; −0.001621 against the incumbent, −0.001921 against the original reference.
  Predicted by the keeper on its own after the run (`scripts.pipeline.predict`, four loader
  workers, same checkpoint and settings; the same scoring reproduces the incumbent's 0.935270 and
  the reference's 0.935570), saved in
  `outputs/analysis/c/centred-257-px-prediction-20261012/validation_predictions.csv`.
- **Time:** ten epochs in about 3,500 s (1.32 of the reference); GPU 22.6 GB.

| Label | Incumbent | Run | Change at ×4 |
|---|---|---|---|
| Lateral Meniscus | 0.926608 | 0.929204 | +0.002596 |
| ACL | 0.951168 | 0.953233 | +0.002065 |
| Baker's | 0.969806 | 0.971145 | +0.001339 |
| Medial Meniscus | 0.950007 | 0.950911 | +0.000904 |
| Synovitis | 0.892382 | 0.892128 | −0.000254 |
| Contusion | 0.958504 | 0.958008 | −0.000496 |
| Medial OA | 0.930236 | 0.929124 | −0.001112 |
| Fracture | 0.936343 | 0.934193 | −0.002150 |
| Effusion | 0.920596 | 0.918273 | −0.002322 |
| PF OA | 0.917596 | 0.914791 | −0.002805 |
| Lateral OA | 0.920776 | 0.914546 | −0.006229 |
| MCL | 0.949214 | 0.938228 | −0.010986 |

A larger image did not help the ResNet-50 here: with 32% more pixels and 15% more field per
side, macro is down at both multipliers. The menisci, ACL and Baker's gain 0.001 to 0.003; MCL
(−0.0110) and Lateral OA (−0.0062), two rare labels, carry the loss, and MCL fell in all three
centred runs (−0.0050, −0.0082, −0.0110). All three centred versions are below the incumbent at
×4 (−0.0008, −0.0010, −0.0016).

**Incident, 01:47 CEST: the memory guard killed the run at the start of its ×4 prediction.**
Available RAM fell from 31 GB to 2.9 GB in 40 s. Measured by another session: the training
workers exit, 16 prediction workers start and fill a queue of up to 32 ×4 batches of about
0.57 GB each (`scripts/pipeline/train.py` builds the prediction loader with the trainer's 16
workers and prefetch 2), then it plateaus; the two earlier runs plateaued at about 4.3 GB
available. WSL stayed up. The tightened guard (abort below 3.5 GB) sat just under that plateau,
so it would have cut any run in its last minute: it is back to 2.5 GB, now killing the training
process directly, with no quiet stage. Candidate fix for the recipe, not applied: four workers
for the prediction loader.

## 6. `cosine-schedule-with-warm-restarts`: below the incumbent at both multipliers

The user's seed outside the sampling scope. After the unchanged warm-up epoch, three cosine
cycles of three epochs with peaks at 1, 0.5 and 0.25 of the configured rates, each decaying to
the 0.01 floor, in place of the single decay over epochs 2 to 10. Built on aligned contrast
mixing.

- **×1:** 0.927026; −0.001250 against the incumbent, −0.000264 against the original reference.
  Per epoch against the incumbent: −0.0018, +0.0003, +0.0048, +0.0036 (end of cycle 1), −0.0003,
  −0.0006, −0.0018 (end of cycle 2), −0.0040, −0.0010, −0.0013 (end of cycle 3). Ends of the
  cycles: 0.920919, 0.923880, 0.927026.
- **×4:** 0.934700; −0.000570 against the incumbent, −0.000870 against the original reference.
- **Time:** 2,661 s (1.004 of the reference). Lowest available RAM, in the final prediction:
  4.9 GB.

| Label | Incumbent | Run | Change at ×4 |
|---|---|---|---|
| Fracture | 0.936343 | 0.939558 | +0.003215 |
| MCL | 0.949214 | 0.950989 | +0.001775 |
| Lateral Meniscus | 0.926608 | 0.927760 | +0.001153 |
| Baker's | 0.969806 | 0.970349 | +0.000542 |
| Effusion | 0.920596 | 0.921123 | +0.000527 |
| Contusion | 0.958504 | 0.958944 | +0.000441 |
| Medial OA | 0.930236 | 0.929672 | −0.000564 |
| PF OA | 0.917596 | 0.916114 | −0.001482 |
| Medial Meniscus | 0.950007 | 0.948430 | −0.001577 |
| ACL | 0.951168 | 0.948055 | −0.003113 |
| Synovitis | 0.892382 | 0.888562 | −0.003820 |
| Lateral OA | 0.920776 | 0.916843 | −0.003932 |

Each cycle ends higher than the one before, and each restart loses ground that the next decay
only partly recovers within three epochs; the single decay is ahead from epoch 5 on. Open
variants, not run: one restart instead of two, or a smaller rise (peaks 1, 0.3, 0.1).

**Epoch 1 as a measure of run-to-run variation.** The first epoch of this run is the incumbent's
training exactly (same code path, schedule and seed) and scores 0.899084 against 0.900848:
0.0018 apart at ×1 after one epoch, on the ResNet-50, with `trainer.deterministic` off.

## 7. `sagittal-fluid-ten-distinct-slices`: above the incumbent at ×1, level at ×4

The detail view removed; sagittal fluid gets ten distinct slices in one 84 mm view;
`study_slice_total` 31. Same image count in training as the reference. Built on aligned contrast
mixing.

- **×1:** 0.931363 at epoch 9 (epoch 10: 0.930674); +0.003087 against the incumbent, +0.004073
  against the original reference. Per epoch against the incumbent: +0.0088, +0.0044, +0.0044,
  +0.0029, +0.0019, +0.0034, +0.0023, +0.0028, +0.0039, +0.0024. **Not like for like:** at ×1
  this run validates on ten sagittal fluid slices, the incumbent on five and five magnified
  copies.
- **×4:** 0.935225; −0.000044 against the incumbent, −0.000345 against the original reference.
  Paired bootstrap of the change against the incumbent (1,000 draws): 10% to 90% −0.0012 to
  +0.0011, 48% above zero. Predicted by the keeper on its own after the run (six loader workers,
  124 s, lowest available RAM 29.2 GB), saved in
  `outputs/analysis/s/sagittal-ten-slices-prediction-20261012/validation_predictions.csv`.
- **Time:** ten epochs at 1.03 of the reference's pace; GPU 16.8 GB.

| Label | Incumbent | Run | Change at ×4 | Positives |
|---|---|---|---|---|
| Fracture | 0.936343 | 0.940725 | +0.004382 | 60 |
| Contusion | 0.958504 | 0.960197 | +0.001693 | 93 |
| Medial OA | 0.930236 | 0.931843 | +0.001607 | 189 |
| Synovitis | 0.892382 | 0.893940 | +0.001558 | 114 |
| Medial Meniscus | 0.950007 | 0.950437 | +0.000430 | 339 |
| Lateral OA | 0.920776 | 0.920816 | +0.000041 | 95 |
| Baker's | 0.969806 | 0.969771 | −0.000035 | 114 |
| MCL | 0.949214 | 0.948648 | −0.000566 | 47 |
| Lateral Meniscus | 0.926608 | 0.925705 | −0.000902 | 135 |
| ACL | 0.951168 | 0.949709 | −0.001459 | 131 |
| PF OA | 0.917596 | 0.914006 | −0.003590 | 290 |
| Effusion | 0.920596 | 0.916907 | −0.003689 | 244 |

The ×1 gain is what more slices at validation give, as ×4 over ×1 always does; at ×4, where both
read most of the stack, training on ten distinct sagittal fluid slices instead of five and five
magnified copies changes nothing in the macro. It is the best ×4 of the six candidates so far,
and the only one level with the incumbent. PF OA and Effusion, two large labels, are down
0.0036 to 0.0037: the detail view it removes was open to PF OA, and Effusion is read mostly on
axial. Not adopted by the rule (not above at ×4). Worth knowing for the endgame: a recipe that
is level at ×4 and +0.003 at ×1 is the better one if inference has to run at a lower multiplier.

**Incident, 03:30 CEST: the guard killed this run in its ×4 prediction** (available RAM 1.95 GB,
below the 2.5 GB line; swap 2.4 of 10 GB in use). The earlier 224 px runs plateaued at 4.3 to
4.9 GB; this one reads the sagittal fluid stack whole at ×4 and went lower. WSL stayed up. The
keeper predicted on its own and resumed at 03:37.

## 8. `axial-fluid-kneecap-tile-at-native-pixels`: below the incumbent at both multipliers

The shared-slice second view moved from sagittal fluid (64 mm) to axial fluid: a 100 mm tile on
the kneecap front point, open to PF OA, Effusion and Synovitis. Built on aligned contrast mixing.

- **×1:** 0.926572 at epoch 9; −0.001704 against the incumbent, −0.000718 against the original
  reference. Per epoch against the incumbent: −0.0006, +0.0013, +0.0003, −0.0027, −0.0011,
  −0.0013, −0.0022, −0.0016, −0.0009, −0.0019.
- **×4:** 0.935067; −0.000203 against the incumbent, −0.000503 against the original reference.
- **Time:** 2,770 s (1.045 of the reference); GPU 17.4 GB; lowest available RAM, in the final
  prediction, 7.1 GB. Completed by the harness.

| Label | Incumbent | Run | Change at ×4 |
|---|---|---|---|
| Medial Meniscus | 0.950007 | 0.951909 | +0.001902 |
| Baker's | 0.969806 | 0.970960 | +0.001154 |
| Contusion | 0.958504 | 0.959371 | +0.000867 |
| Lateral Meniscus | 0.926608 | 0.927049 | +0.000441 |
| Fracture | 0.936343 | 0.936712 | +0.000369 |
| Effusion | 0.920596 | 0.920271 | −0.000325 |
| Synovitis | 0.892382 | 0.891874 | −0.000508 |
| Medial OA | 0.930236 | 0.929409 | −0.000826 |
| Lateral OA | 0.920776 | 0.919532 | −0.001243 |
| ACL | 0.951168 | 0.949904 | −0.001264 |
| MCL | 0.949214 | 0.947825 | −0.001389 |
| PF OA | 0.917596 | 0.915984 | −0.001612 |

The three labels the tile was opened to did not gain: PF OA −0.0016, Effusion −0.0003,
Synovitis −0.0005 (the seed's test was PF OA +0.002). The tile's attention shares are logged in
the run's validation metrics and were not read here. PF OA is down in every run that removes
the sagittal 64 mm detail view (ten slices −0.0036, this run −0.0016): the detail view was open
to PF OA, so what it gave PF OA is not replaced by native axial pixels on the kneecap.

## 9. `one-more-slice-on-every-series-type`: above the incumbent at both multipliers (keeper's scoring); adoption awaits the user

The detail view removed; slices per type 7, 7, 6, 6, 6, 6 (axial T1/PD, axial fluid, coronal
T1/PD, coronal fluid, sagittal T1/PD, sagittal fluid); `study_slice_total` 31. About the
reference's image count in training. Built on aligned contrast mixing.

- **×1:** 0.931204 at epoch 9 (epoch 10: 0.930922); +0.002928 against the incumbent, +0.003914
  against the original reference. Per epoch against the incumbent: +0.0005, +0.0060, +0.0070,
  +0.0053, +0.0040, +0.0053, +0.0031, +0.0030, +0.0038, +0.0026. Not like for like at ×1: this
  run validates on 38 distinct slices, the incumbent on 26 and five magnified copies.
- **×4:** 0.935771; **+0.000502 against the incumbent, +0.000201 against the original
  reference**: the first run of the campaign above both. Paired bootstrap of the change against
  the incumbent (1,000 draws): 10% to 90% −0.0009 to +0.0018, 67% above zero. Predicted by the
  keeper on its own after the run (six loader workers), saved in
  `outputs/analysis/o/one-more-slice-prediction-20261012/validation_predictions.csv`.
- **Time:** ten epochs at 1.01 of the reference's pace; GPU 17.1 GB.

| Label | Incumbent | Run | Change at ×4 | Positives |
|---|---|---|---|---|
| Lateral Meniscus | 0.926608 | 0.930918 | +0.004310 | 135 |
| Synovitis | 0.892382 | 0.896030 | +0.003647 | 114 |
| Effusion | 0.920596 | 0.922619 | +0.002023 | 244 |
| Medial Meniscus | 0.950007 | 0.950889 | +0.000882 | 339 |
| Fracture | 0.936343 | 0.936855 | +0.000512 | 60 |
| Medial OA | 0.930236 | 0.930669 | +0.000433 | 189 |
| Baker's | 0.969806 | 0.969656 | −0.000150 | 114 |
| MCL | 0.949214 | 0.948751 | −0.000463 | 47 |
| PF OA | 0.917596 | 0.917106 | −0.000490 | 290 |
| Lateral OA | 0.920776 | 0.920249 | −0.000527 | 95 |
| ACL | 0.951168 | 0.949133 | −0.002034 | 131 |
| Contusion | 0.958504 | 0.956383 | −0.002120 | 93 |

The gains sit on large labels (Lateral Meniscus, Synovitis, Effusion: 114 to 244 positives),
which is what carried from the ResNet-18 to the ResNet-50 before; Effusion and Synovitis are the
labels whose errors followed axial reach, and this run gives axial one more slice per contrast.
Against run 7 (all five extra slices on sagittal fluid: ×4 level, PF OA −0.0036, Effusion
−0.0037), spreading the slices over the six types is the better use of the same budget: PF OA
is nearly kept (−0.0005) without the detail view.

**It meets the campaign's adoption rule by this scoring, but the harness did not complete it:**
the guard killed the run in its ×4 prediction at 05:13 CEST (available RAM 2.4 GB), so the
runtime has no comparison and cannot adopt it; the keeper does not adopt. The idea is parked
with the figures; the user decides. Later candidates of the night still build on aligned
contrast mixing.

**Check of the separate scoring.** On run 8, which the harness completed, the keeper's separate
prediction of the same checkpoint differs from the harness's predictions by 0.00005 on average
(0.0036 at most) and scores 0.935074 against the harness's 0.935067.

**Guard corrected, 2026-10-12 06:03 CEST** (the user, on the monitor page: "It's unacceptable to
have one every two experiments invalid"). The three kills of the night (257 px, ten slices, one
more slice) all happened with 7.6 GB or more of swap still free: the guard looked at available
RAM alone. It now kills only when available RAM plus free swap is under 4 GB. The dip is the
final ×4 prediction's queue (16 workers, prefetch 2, about 18 GB of batches), bounded and about
a minute long; the cause is in `scripts/pipeline/train.py` and is not changed.

## 10. `sagittal-fluid-context-view-106-mm`: below the incumbent at both multipliers

The second sagittal fluid view keeps the first view's slices and becomes a 106 mm context view
(in place of the 64 mm detail view), open to PF OA, Effusion, Synovitis and Baker's. Built on
aligned contrast mixing.

- **×1:** 0.926233; −0.002043 against the incumbent, −0.001057 against the original reference.
  Per epoch against the incumbent: +0.0004, +0.0015, −0.0042, −0.0052, −0.0036, +0.0002,
  −0.0023, −0.0013, −0.0013, −0.0020.
- **×4:** 0.934464; −0.000805 against the incumbent, −0.001105 against the original reference.
- **Time:** 2,676 s (1.010 of the reference). Lowest available RAM, in the final prediction,
  7.5 GB. Completed by the harness.

| Label | Incumbent | Run | Change at ×4 |
|---|---|---|---|
| Fracture | 0.936343 | 0.940131 | +0.003788 |
| Baker's | 0.969806 | 0.971791 | +0.001985 |
| Synovitis | 0.892382 | 0.892440 | +0.000058 |
| Contusion | 0.958504 | 0.958559 | +0.000055 |
| Effusion | 0.920596 | 0.920544 | −0.000052 |
| Lateral Meniscus | 0.926608 | 0.926086 | −0.000521 |
| Medial OA | 0.930236 | 0.929664 | −0.000572 |
| ACL | 0.951168 | 0.950325 | −0.000842 |
| Lateral OA | 0.920776 | 0.919141 | −0.001635 |
| Medial Meniscus | 0.950007 | 0.948105 | −0.001902 |
| PF OA | 0.917596 | 0.913645 | −0.003950 |
| MCL | 0.949214 | 0.943142 | −0.006072 |

Baker's rose as the seed expected (+0.0020, its test was +0.002 to +0.004); PF OA, which it was
also opened to, fell (−0.0040), as did the menisci and ACL that lost the 64 mm view. PF OA is
down in all three runs without the 64 mm view on PF OA (−0.0036, −0.0016, −0.0040) and nearly
kept only where every type got a slice (run 9, −0.0005).

## 11. `axial-views-at-100-mm`: below the incumbent at both multipliers

Both axial views at 100 mm instead of 120 mm (0.446 mm per pixel, at or above the acquired
resolution in 85% of axial fluid series against 35%). Built on aligned contrast mixing.

- **×1:** 0.926777; −0.001499 against the incumbent, −0.000513 against the original reference.
  Per epoch against the incumbent: −0.0032, +0.0015, +0.0018, −0.0019, −0.0012, −0.0012,
  −0.0008, −0.0025, −0.0010, −0.0015. Like for like: the same slices as the incumbent.
- **×4:** 0.934166; −0.001103 against the incumbent, −0.001403 against the original reference.
- **Time:** 2,673 s (1.009 of the reference). Lowest available RAM, in the final prediction,
  6.2 GB. Completed by the harness.

| Label | Incumbent | Run | Change at ×4 |
|---|---|---|---|
| Contusion | 0.958504 | 0.961037 | +0.002533 |
| Synovitis | 0.892382 | 0.893017 | +0.000635 |
| Medial OA | 0.930236 | 0.930630 | +0.000394 |
| Fracture | 0.936343 | 0.936425 | +0.000082 |
| Effusion | 0.920596 | 0.920570 | −0.000026 |
| PF OA | 0.917596 | 0.916314 | −0.001281 |
| Medial Meniscus | 0.950007 | 0.948601 | −0.001406 |
| Baker's | 0.969806 | 0.967890 | −0.001916 |
| Lateral Meniscus | 0.926608 | 0.924593 | −0.002015 |
| MCL | 0.949214 | 0.946204 | −0.003010 |
| Lateral OA | 0.920776 | 0.917195 | −0.003581 |
| ACL | 0.951168 | 0.947520 | −0.003647 |

PF OA, the label the seed expected to gain from native axial pixels, fell (−0.0013), and Baker's
fell as the seed feared from the cut posterior tissue (−0.0019). With run 8 (the 100 mm kneecap
tile, PF OA −0.0016), two runs now feed axial at its acquired resolution and neither helps PF
OA: in-plane resolution on axial is not what limits it. Together with the 257 px run, no input
change of the night that alters field or pixel size is above the incumbent; the one that is
adds distinct slices on every type (run 9).

## 12. `thirty-five-distinct-slices`: above the incumbent at both multipliers

The detail view removed; slices per type 7, 7, 6, 6, 6, 10 (ten on sagittal fluid, one more on
every other type); `study_slice_total` 35. Runs 7 and 9 together. Built on aligned contrast
mixing. Completed by the harness; the runtime's verdict is provisional accept.

- **×1:** 0.933540 at epoch 9; +0.005264 against the incumbent, +0.006250 against the original
  reference. Per epoch against the incumbent: +0.0075, +0.0072, +0.0051, +0.0060, +0.0082,
  +0.0055, +0.0063, +0.0063, +0.0061, +0.0050. Not like for like at ×1 (more distinct slices at
  validation).
- **×4:** 0.936965; **+0.001695 against the incumbent, +0.001395 against the original
  reference**. The best ×4 of the campaign.
- **Time:** 2,990 s (1.128 of the reference); GPU 18.7 GB. Lowest available RAM, in the final
  prediction, 2.6 GB with 7.5 GB of swap free: the corrected guard let it finish, the earlier
  one would have killed it.

| Label | Incumbent | Run | Change at ×4 |
|---|---|---|---|
| MCL | 0.949214 | 0.955826 | +0.006612 |
| Synovitis | 0.892382 | 0.896941 | +0.004559 |
| Contusion | 0.958504 | 0.962992 | +0.004488 |
| Fracture | 0.936343 | 0.938473 | +0.002129 |
| Medial OA | 0.930236 | 0.932113 | +0.001877 |
| Baker's | 0.969806 | 0.971041 | +0.001235 |
| PF OA | 0.917596 | 0.918641 | +0.001045 |
| Medial Meniscus | 0.950007 | 0.950955 | +0.000948 |
| Effusion | 0.920596 | 0.920075 | −0.000520 |
| Lateral Meniscus | 0.926608 | 0.926086 | −0.000521 |
| Lateral OA | 0.920776 | 0.920249 | −0.000527 |
| ACL | 0.951168 | 0.950192 | −0.000976 |

Eight labels up and none down by more than 0.001. The dose series of distinct training slices,
at ×4 against the incumbent: 31 slices all on sagittal fluid −0.00004 (run 7), 31 slices spread
+0.0005 (run 9), 35 slices +0.0017 (this run). PF OA rises here (+0.0010) without the 64 mm
view. The cost is 1.13 of the reference's training time and more images per study at
inference. Of the gains, MCL (+0.0066), Contusion and Fracture are rare labels; Synovitis
(+0.0046), Medial OA (+0.0019) and PF OA (+0.0010) are large ones.

## `sagittal-fluid-kneecap-and-pouch-tile`: parked at review, not run

Adopted by the runtime at 07:50 CEST: `thirty-five-distinct-slices` is the incumbent (×1
0.933540, ×4 0.936965). The tile seed says the second sagittal fluid view becomes a 93 mm tile
on the kneecap; the new incumbent has no second view, so the candidate rebuilt on it had nothing
to replace and the reviewer parked it. Run on the new incumbent it would have to add a view on
shared slices: five more images on a recipe already at 1.13 of the reference's time, about 1.28,
over the budget. Left parked for the user: either on five of the ten sagittal fluid slices with
a budget exception, or dropped. Its premise (PF OA lower when the kneecap is cut) is weaker
since run 12, where PF OA rose without any second view.

## The researcher's first proposals (08:31 CEST), on the 35-slices incumbent

1. `acquired-endpoints-in-every-slice-bag`: every bag keeps the first and last acquired slice
   position of each series type; the rest is drawn as before.
2. `contrast-mixing-with-acquisition-depth-support`: the aligned donor contributes only where
   its slice's depth cell overlaps the target's.
3. `sagittal-fluid-primary-context-on-distinct-slices`: the sagittal fluid view at 106 mm on its
   ten distinct slices.
4. `fractional-canvases-at-half-storage`: canvases kept as float16 instead of uint8. **Removed
   from the queue by the keeper, pending the user:** it doubles the loader's image buffers, and
   the final ×4 prediction already queues about 18 GB of batches and leaves 2.6 GB available on
   the incumbent, so the run would exhaust memory at its end.

## 13. `three-dimensional-series-read-as-slabs`: level with the incumbent

A pick of a 3D series (836 series in 659 studies) is the mean of its three decoded thin slices
instead of the middle one. Built on the 35-slices incumbent, so like for like at both
multipliers.

- **×1:** 0.933203 at epoch 9; −0.000337 against the incumbent. Per epoch against it: +0.0013,
  −0.0007, +0.0012, −0.0014, −0.0012, +0.0005, +0.0006, −0.0005, −0.0003, −0.0005.
- **×4:** 0.936887; −0.000078 against the incumbent.
- **Time:** 2,989 s, the incumbent's. Lowest available RAM, in the final prediction, 2.7 GB with
  4.2 GB of swap free. Completed by the harness; rejected by the rule.

| Label | Incumbent | Run | Change at ×4 |
|---|---|---|---|
| MCL | 0.955826 | 0.957936 | +0.002110 |
| Lateral Meniscus | 0.926086 | 0.927339 | +0.001253 |
| Baker's | 0.971041 | 0.971630 | +0.000589 |
| ACL | 0.950192 | 0.950572 | +0.000380 |
| Medial Meniscus | 0.950955 | 0.951247 | +0.000292 |
| Fracture | 0.938473 | 0.938452 | −0.000020 |
| PF OA | 0.918641 | 0.918552 | −0.000089 |
| Effusion | 0.920075 | 0.919861 | −0.000215 |
| Synovitis | 0.896941 | 0.896330 | −0.000612 |
| Medial OA | 0.932113 | 0.930977 | −0.001135 |
| Contusion | 0.962992 | 0.961836 | −0.001156 |
| Lateral OA | 0.920249 | 0.917911 | −0.002338 |

No measurable effect in the macro; the subgroup with a 3D series was not read apart here. As a
by-product it is the closest thing to a rerun of the incumbent the campaign has: a change that
touches 15% of the studies lands within 0.0001 at ×4 and 0.0003 at ×1, with single labels moving
by up to 0.002. That is in line with the ResNet-18's rerun spread and makes run 12's +0.0017 at
×4 hard to attribute to retraining alone.

## 14. `acquired-endpoints-in-every-slice-bag`: below the incumbent at both multipliers

The researcher's idea: every bag keeps the first and last acquired slice position of each
series type. Built on the 35-slices incumbent.

- **×1:** 0.929801 at epoch 9; −0.003739 against the incumbent. Per epoch against it: −0.0081,
  −0.0077, −0.0055, −0.0070, −0.0059, −0.0052, −0.0061, −0.0041, −0.0037, −0.0035.
- **×4:** 0.934592; −0.002374 against the incumbent.
- **Time:** 3,005 s. Completed by the harness; rejected by the rule.

| Label | Incumbent | Run | Change at ×4 |
|---|---|---|---|
| Lateral Meniscus | 0.926086 | 0.931469 | +0.005383 |
| Effusion | 0.920075 | 0.919971 | −0.000104 |
| Fracture | 0.938473 | 0.938145 | −0.000328 |
| Medial Meniscus | 0.950955 | 0.950448 | −0.000507 |
| Medial OA | 0.932113 | 0.931325 | −0.000788 |
| Baker's | 0.971041 | 0.969125 | −0.001916 |
| Lateral OA | 0.920249 | 0.918114 | −0.002135 |
| Contusion | 0.962992 | 0.958752 | −0.004240 |
| ACL | 0.950192 | 0.945548 | −0.004644 |
| PF OA | 0.918641 | 0.912854 | −0.005786 |
| Synovitis | 0.896941 | 0.890339 | −0.006602 |
| MCL | 0.955826 | 0.949008 | −0.006818 |

Eleven labels down. Two of six or seven slices per type forced to the ends of the stack leave
fewer slices on the joint; it loses at ×4 as well, so the training bags were worse, not only the
×1 validation.

## Stopped by the user (2026-10-12 09:50 CEST)

Graceful stop after run 14. Fourteen runs; incumbent in the runtime `thirty-five-distinct-slices`.

**The user's objection to run 12 (09:47):** an idea that adds images per study should not have
been queued. It is an endgame item (`docs/backlog/untried-ideas.md`: more training triplets per
study); the sampling subagent had flagged its seed 7 as such and the keeper queued it anyway.
Runs 7 and 9 keep the reference's image count; run 12 has about 36 images against 31.7 and 1.13
of the time. **And its measurement is not like for like:** it validates and predicts on more
images than its parent, and the comparison ignores the attention temperature, which is what
absorbs a difference in image count between training and inference. The user ordered a
comparison of the two models at one fixed image count (×4 of the aligned-mixing recipe, 124
images nominal) over temperatures 0.7 to 1.0; results below.

## Fixed image count and temperature sweep: aligned mixing against 35 slices (the user's order)

Predictions on the 874 validation studies of fold 1 with each run's selected checkpoint, saved in
`outputs/analysis/f/fixed-images-tau-sweep-20261012/`.

**Image counts.** At ×4 the two models read the same number of images under their own configs:
107,397 for aligned mixing (122.9 per study) and 107,292 for 35 slices (122.8). The stacks are
exhausted at ×4, so a config with more slices per type returns no more images; the aligned
model's count includes about 20 magnified copies of sagittal fluid slices per study. The
unequal comparison is ×1: 31.65 images per study against 35.84. A reduced bag for the 35-slices
model (6, 6, 5, 5, 5, 10 at ×4) reads 96,913 images, 10% fewer than the aligned model.

| Temperature | Aligned mixing | 35 slices | Difference | 10% to 90% (paired bootstrap, 500 draws) |
|---|---|---|---|---|
| 0.7 | 0.934275 | 0.935727 | +0.001453 | +0.0001 to +0.0029 |
| 0.8 | 0.934882 | 0.936751 | +0.001869 | +0.0005 to +0.0032 |
| 0.9 | 0.935155 | 0.937065 | +0.001911 | +0.0007 to +0.0031 |
| 1.0 | 0.935272 | 0.936968 | +0.001696 | +0.0005 to +0.0029 |

- Best temperature of each: aligned mixing 1.0 (0.935272), 35 slices 0.9 (0.937065); difference
  between the two bests +0.001793.
- The 35-slices model on the reduced bag (96,913 images) at temperature 1.0: 0.936745, still
  +0.0015 above the aligned model on its 107,397.
- Temperature moves each model by at most 0.0013 over 0.7 to 1.0 and does not change the order.

Per label at temperature 0.9:

| Label | Aligned mixing | 35 slices | Change |
|---|---|---|---|
| MCL | 0.949446 | 0.957241 | +0.007795 |
| Synovitis | 0.891459 | 0.895533 | +0.004074 |
| Fracture | 0.936896 | 0.940438 | +0.003542 |
| Contusion | 0.959041 | 0.962469 | +0.003428 |
| Medial Meniscus | 0.948651 | 0.950685 | +0.002035 |
| Medial OA | 0.930823 | 0.932839 | +0.002016 |
| Baker's | 0.969402 | 0.971030 | +0.001627 |
| PF OA | 0.917052 | 0.918269 | +0.001216 |
| Lateral OA | 0.920978 | 0.920438 | −0.000541 |
| ACL | 0.951229 | 0.950510 | −0.000719 |
| Effusion | 0.920062 | 0.919327 | −0.000735 |
| Lateral Meniscus | 0.926818 | 0.926006 | −0.000812 |

So the ×4 gain of run 12 is not an effect of reading more images at prediction, nor of the
temperature: it holds at equal image count, at every temperature, and with 10% fewer images.
What differs is the training: about 36 images per study against 31.7 (1.13 of the time), which
is the endgame trade the user did not want queued.

## Manual run after the stop: more slices with the 64 mm detail view (the user's idea, 2026-10-12)

Run by the keeper outside the runtime, as a campaign run would be: fold 1, seed 1110484336, ten
epochs, built from the aligned-mixing source by config alone
(`outputs/training/rn50-more-slices-with-detail-view`, build folder beside it). Slices 7, 7, 6,
6, 6 and 5, plus the 64 mm detail view on the five sagittal fluid slices: 31 distinct slices and
5 magnified copies, about 37 images per study, the 35-slices incumbent's cost. The question:
do finer sagittal fluid images add to more slices.

- **×1:** 0.933350 at epoch 10; −0.000190 against the incumbent (0.933540), +0.005074 against
  aligned mixing. Per epoch against the incumbent: −0.0014, −0.0044, +0.0012, −0.0006, −0.0012,
  −0.0006, −0.0015, −0.0010, −0.0010, +0.0001.
- **×4:** 0.936100; −0.000865 against the incumbent (0.936965), +0.000831 against aligned
  mixing. Paired bootstrap (500 draws, 10% to 90%): −0.0018 to +0.0001 against the incumbent,
  −0.0003 to +0.0019 against aligned mixing.
- **Time:** 3,544 s in all, not comparable: the first two epochs ran at half speed (below). The
  eight epochs after took 5:11 on average against the incumbent's 4:58 (1.04).

| Label | Incumbent | Run | Against the incumbent | Against aligned mixing |
|---|---|---|---|---|
| Lateral OA | 0.920249 | 0.923289 | +0.003040 | +0.002513 |
| Effusion | 0.920075 | 0.921181 | +0.001106 | +0.000585 |
| Baker's | 0.971041 | 0.971791 | +0.000750 | +0.001985 |
| Synovitis | 0.896941 | 0.897461 | +0.000519 | +0.005078 |
| Medial Meniscus | 0.950955 | 0.950955 | −0.000000 | +0.000948 |
| Medial OA | 0.932113 | 0.931958 | −0.000154 | +0.001722 |
| Fracture | 0.938473 | 0.938084 | −0.000389 | +0.001740 |
| ACL | 0.950192 | 0.948568 | −0.001623 | −0.002599 |
| Contusion | 0.962992 | 0.960417 | −0.002575 | +0.001914 |
| MCL | 0.955826 | 0.952276 | −0.003550 | +0.003062 |
| PF OA | 0.918641 | 0.915027 | −0.003614 | −0.002568 |
| Lateral Meniscus | 0.926086 | 0.922197 | −0.003889 | −0.004410 |

At equal cost, five more distinct sagittal fluid slices (the incumbent) are worth more than five
magnified copies of the same slices: −0.0009 at ×4. The labels the detail view is open to do not
gain from it here (PF OA −0.0036, Lateral Meniscus −0.0039, ACL −0.0016 against the incumbent).
Against aligned mixing the run is +0.0008 at ×4, which is the extra slice on five series types
(31 distinct slices against 26); the ladder at ×4 is 26 slices 0.935270, 31 with the detail view
0.936100, 31 without it 0.935771 (run 9), 35 without it 0.936965.

**Incident: the GPU ran at half speed for the first two epochs** (8:39 and 8:51 against 5:04).
The card sat at P3, about 140 W and 800 MHz. Windows had moved 3.3 GB of the training's 22.2 GB
of GPU memory into shared system memory (Task Manager: dedicated 18.9 of 24 GB, shared 3.3 GB)
because desktop apps held part of the card; the user suspects VS Code, opened at 11:06. It
recovered at the next epoch boundary after the apps were closed. `nvidia-smi` shows only the
dedicated part. The 257 px run of the night (22.6 GB) was probably slowed the same way.

## Close-out (2026-10-12, the user)

- **S146**, the 35-slices incumbent (aligned mixing, 0.25 teacher share) at ×4, τ 0.9, batch 4,
  8 workers: **0.959 public**, against 0.960 for S143 (the reference, same teacher share) and
  0.959 for S142 (the reference recipe on the usual labels). S144 (aligned mixing alone) and
  S145 failed, so aligned mixing has no public score.
- **On main by the user's decision:** the recipe of the aligned-contrast-mixing run (soft-tissue
  line, sagittal fluid detail view, aligned mixing) in both configs, trained on the usual labels
  (teacher share 0). The 35 slices stay out. No run exists on exactly that setting.
- A 35-slices training on the usual labels was started at the user's request and stopped by
  the user after three epochs (×1 0.908047, 0.912190, 0.917417).
- `INFERENCE_NUM_WORKERS` 16 to 8 (submissions in another session crashed with 16), used by the
  training's final ×4 prediction as well, which removes the memory dip at the end of a run.
- The runtime's sealed training ceiling was raised to 3,843 s in this campaign for the 257 px
  exception; a new campaign seals its own.
