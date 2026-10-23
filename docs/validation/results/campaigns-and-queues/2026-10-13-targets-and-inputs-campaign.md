# 2026-10-13 targets-and-inputs campaign

Autoresearch campaign on the Refined-RadImageNet ResNet-50, 2026-10-12 18:36 to 2026-10-13 17:58
CEST, stopped by the user after 29 runs. Fold 2 of `all_5fold_scanner_v2` (875 validation
studies), seed 1110484336, ten epochs, labels `rebuild_state_v2`, teacher share 0. Reference:
fold 2 of `outputs/training/rn50-aligned-mixing-folds-0-2`, ×4 0.937000, ×1 0.928460, 2,691 s.
**Adopted on main: a mask decoder trained beside the labels, the three-compartment OA detector
beside the routed shared OA head, and average-pool shortcuts in the ResNet-50: ×4 0.939634, ×1
0.930902 (run 27).** The user removed the mask decoder again on 2026-10-14, without a run
([current model](../../../modeling/current-model.md#adoption-history)). Per run, per label, logged diagnostics and incidents:
[run notes](../../../../autoresearch/campaigns/targets-and-inputs/run-notes.md).

## Runs

Each run is one training against the incumbent of its time ("Against"). Intervals are 80% paired
report-group bootstrap intervals of the ×4 change unless marked. The runtime's rule adopts a run
strictly above its incumbent at ×1 and ×4.

| Run | Change | Against | ×4 | ×4 Δ | 80% interval | ×1 Δ | Outcome |
|---:|---|---|---:|---:|---|---:|---|
| 1 | Class weights from the mean soft target | reference | 0.937509 | +0.0005 | [−0.0002, +0.0013] | +0.0013 | passed the rule; held out, see below |
| 2 | Clipping bounds from a fixed set of slices | reference | 0.936730 | −0.0003 | [−0.0010, +0.0005] | +0.0006 | rejected |
| 3 | Report cells weighted by distance from 0.5 | reference | 0.936047 | −0.0010 | [−0.0020, +0.0002] | −0.0001 | rejected |
| 4 | Report targets hardened at 0.5 | reference | 0.937529 | +0.0005 | [−0.0002, +0.0013] | +0.0014 | rejected |
| 5 | Mask decoder on the 28 by 28 grid, training only | reference | 0.937529 | +0.0005 | [−0.0002, +0.0013] | +0.0014 | adopted |
| 6 | Intensity augmentation off | run 5 | 0.937148 | −0.0004 | [−0.0014, +0.0007] | −0.0009 | rejected |
| 7 | Noise augmentation off | run 5 | 0.937905 | +0.0004 | [−0.0007, +0.0014] | −0.0004 | rejected |
| 8 | Hard-pair batches and pair term in epochs 8 to 10 | run 5 | 0.937728 | +0.0002 | [−0.0006, +0.0010] | −0.0007 | rejected |
| 9 | Plane-contrast readouts off | run 5 | 0.937778 | +0.0002 | [−0.0008, +0.0013] | −0.0004 | rejected |
| 10 | OA detector beside the routed shared OA head | run 5 | 0.938618 | +0.0011 | [+0.0004, +0.0018] | +0.0014 | adopted |
| 11 | Separate OA heads plus the detector, no shared head | run 10 | 0.937965 | −0.0007 | [−0.0017, +0.0003] | −0.0020 | rejected |
| 12 | Report-attribute loss at 0.1 | run 10 | 0.938323 | −0.0003 | [−0.0010, +0.0003] | −0.0019 | rejected |
| 13 | Scanner confusion head at 0.1 beside L2-SP | run 10 | 0.937043 | −0.0016 | [−0.0024, −0.0007] | −0.0013 | rejected |
| 14 | View label bias off | run 10 | 0.938022 | −0.0006 | [−0.0013, +0.0001] | −0.0014 | rejected |
| 15 | Medial-lateral attention bias restored | run 10 | 0.938003 | −0.0006 | [−0.0013, +0.0001] | −0.0016 | rejected |
| 16 | 64 mm sagittal fluid detail view off | run 10 | 0.937424 | −0.0012 | [−0.0021, −0.0003] | −0.0030 | rejected |
| 17 | Detail-view embedding off | run 10 | 0.938106 | −0.0005 | [−0.0013, +0.0002] | −0.0012 | rejected |
| 18 | GIN augmentation off | run 10 | 0.939728 | +0.0011 | [−0.0002, +0.0024] | −0.0014 | not adopted (user) |
| 19 | Resolution degradation off | run 10 | 0.937574 | −0.0010 | [−0.0022, +0.0001] | −0.0007 | rejected |
| 20 | Mask decoder losses at 0.3 | run 10 | 0.937622 | −0.0010 | [−0.0017, −0.0003] | −0.0015 | rejected |
| 21 | Mask decoder with a 56 by 56 level | run 10 | 0.938075 | −0.0005 | [−0.0013, +0.0002] | −0.0011 | rejected |
| 22 | SE in all sixteen bottlenecks, free | run 10 | 0.937619 | −0.0010 | [−0.0020, −0.0000] | −0.0016 | rejected |
| 23 | Global context in stages two to four, free | run 10 | 0.938277 | −0.0003 | [−0.0014, +0.0007] | −0.0006 | rejected |
| 24 | Class weights from the mean soft target, again | run 10 | 0.937680 | −0.0009 | [−0.0017, −0.0002] | −0.0021 | not adopted |
| 25 | Global context pulled toward the identity | run 10 | 0.938402 | −0.0002 | [−0.0011, +0.0006] | −0.0011 | rejected |
| 26 | Coordinate attention pulled toward the identity (gates stayed shut) | run 10 | 0.937951 | −0.0007 | [−0.0014, +0.0000] | −0.0012 | rejected |
| 27 | Average-pool shortcuts (ResNet-D) | run 10 | 0.939634 | +0.0010 | [+0.0002, +0.0019] | −0.0003 | adopted by the user |
| 28 | SE in stages three and four pulled toward the identity (gates stayed shut) | run 27 | 0.938639 | −0.0010 | [−0.0016, −0.0004] | −0.0013 | rejected |
| 29 | Blur-pool at the three stride-two convolutions | run 27 | 0.938444 | −0.0012 | [−0.0020, −0.0004] | −0.0006 | below at both; undecided when the campaign stopped |

- **Run 1** passed the rule and was taken out of the incumbent the same evening by the keeper on
  the user's concern about class weights on the public board; see
  [the public pair](#soft-prevalence-class-weights-read-publicly).
- **Run 4** was recorded `invalid` by the comparison gate, because hardening the targets changes
  `training_rows_sha256`; its figures come from `compare_fold_runs` re-run on the saved
  predictions with that identity difference allowed.
- **A first blur-pool run** was cancelled 275 s into training when the campaign was stopped to
  adopt run 27; run 29 is the same seed on the new incumbent.
- **Dropped before running:** balanced target columns (user); the free coordinate-attention seed,
  replaced by the pulled one (user).

## What was adopted

- **Mask decoder (run 5).** In training only, a decoder on the second, third and fourth stage
  maps predicts the seven bone and seven soft-tissue classes on the 28 by 28 grid of every image,
  each loss at weight 0.1. The existing 7 by 7 and 14 by 14 maps become more accurate
  (`val_anatomy_loss` 0.6333 against 0.6400 to 0.6431 without it); plane, centre, pose and
  routing weights do not move; the gain sits in the labels with no routed read. Weight 0.3 (run
  20) made the maps better again and the labels worse (−0.0010); a 56 by 56 level (run 21)
  changed neither the maps nor the labels. Reading: a feature effect, saturated at 0.1.
- **OA detector beside the routed shared OA head (run 10).** One linear detector on the tokens,
  a smooth maximum over the medial half, the lateral half and the front of the knee, times a
  per-label scale from zero, added to the three OA logits. The first run with the ×4 interval
  above zero; all three OA labels rise. In place of the shared head (run 11) it loses (Medial OA
  −0.0042).
- **Average-pool shortcuts (run 27).** The three stride-two shortcuts become (2 by 2 average
  pool, the same 1 by 1 convolution at stride 1, batch norm) on the same weights with renamed
  keys; no new parameter, L2-SP anchor intact. Rejected by the rule on ×1 −0.0003 and adopted by
  the user. Against the inert repeat of the previous recipe (run 26) it is +0.0017 at ×4; repeat
  against repeat (run 28 against run 26) +0.0007 at ×4 and −0.0005 at ×1.

## Soft-prevalence class weights, read publicly

Class weights (exponent 0.5) from each label's mean soft target instead of its thresholded
fraction; positive weights fall (Synovitis 2.16 to 1.29, MCL 3.66 to 2.52).

| | Local ×4 | Local ×1 | Public, ×4 τ 0.9, single model | Place in Kaggle's score order |
|---|---:|---:|---:|---:|
| Reference, fold 2 (S151) | 0.937000 | 0.928460 | 0.959 | 13th |
| Soft-prevalence weights on it (run 1, S150) | 0.937509 | 0.929788 | 0.960 | 5th |
| The same weights on the run-10 incumbent (run 24) | 0.937680 against 0.938618 | 0.929182 against 0.931247 | not submitted | |

- **The first class-weight change whose public and local signs agree** (one pair, one fold). On
  the six earlier pairs (ReXNet-200, fold 0, September) local hard AUC never gave the public
  sign; soft AUC and soft AUC against other extractors did four times in five and pointed up
  here (+0.0009 at ×4).
- **Not adopted (user):** on the recipe with the mask decoder and the OA detector the weights
  lose 0.0009 at ×4 with the interval below zero. Contusion, MCL and the tibiofemoral OA labels,
  which gained in run 1, turn negative; they are labels the two additions already raised.
  Synovitis, whose weight is cut most, falls in both runs (−0.0036, −0.0055).
- **Where it still applies:** models on the recipe without the decoder and the detector. The
  campaign's final recipe, with the detector and without the decoder, is unmeasured.

## Targets

Hard report targets lose 0.0029 at ×4 (95% interval below zero) and cells weighted by distance
from 0.5 lose 0.0010: the soft values of uncertain cells carry information the model uses. The
graded weight raises Synovitis (+0.0047) and Effusion (+0.0017), its aims, and lowers PF OA
(−0.0050) and labels with few uncertain cells.

## Inputs and augmentation

- **Clipping bounds from a fixed set of up to seven slices** (run 2): ×1 +0.0006, ×4 −0.0003;
  the step from ×1 to ×4 shrank by 0.0009, the opposite of the seed's expectation.
- **Four augmentation removals** (intensity, noise, GIN, resolution degradation; runs 6, 7, 18,
  19): every one starts faster (GIN off +0.0117 and resolution off +0.0104 at epoch 1) and is
  below its incumbent at ×1 by the end. Effusion falls at ×4 in all four (−0.0019 to −0.0035) and
  Fracture rises in all four (+0.0012 to +0.0079).
- **GIN off** is the best ×4 against the run-10 incumbent (+0.0011, soft AUC +0.0007) with ×1
  −0.0014; its step from ×1 to ×4 is 0.0098 against 0.0074. Left unadopted by the user: GIN is
  there for unseen scanners and one fold cannot show that cost.
- **The 64 mm sagittal fluid detail view** is worth 0.0012 at ×4 and 0.0030 at ×1 on this recipe
  for 10% of training and prediction time and 2.2 GB (run 16); it stays.

## Modules inside the backbone

All keep every Refined-RadImageNet weight and start as the identity.

| Module | Treatment | How far it moved | ×4 Δ |
|---|---|---|---:|
| SE, sixteen blocks (run 22) | head rate, no anchor | gates 12% from 1 | −0.0010, MCL −0.0105 |
| Global context, stages 2 to 4 (run 23) | head rate, no anchor | added term 0.49 of the block output | −0.0003 |
| Global context (run 25) | pulled to zero at the L2-SP strength | 0.39 at epoch 1 to 0.066 | −0.0002 |
| Coordinate attention, stages 3 and 4 (run 26) | pulled | gates 0.04% from 1 | −0.0007 |
| SE, stages 3 and 4 (run 28) | pulled | gates 0.005% from 1 | −0.0010 |

- **Free modules work against L2-SP.** With them the L2-SP penalty, the drift of stage four and
  the training loss all read lower than the incumbent's while the function moves further from
  the pretrained one. User rule from this campaign: a change that risks generalization is not
  adopted on a local fold.
- **Pulled at the L2-SP strength** (0.001 on the module's last layer, no weight decay) a
  multiplicative gate stays shut and an additive block is held to 7%; the model is the incumbent
  again. The pull is a working mechanism; none of the three modules adds to this recipe under it.
- **Blur-pool** on the three stride-two convolutions (run 29) gives nothing on top of the
  average-pool shortcuts, for 13% more training time and twice the ×4 prediction time.
- **Cost of the free modules:** SE 19% more GPU memory (20.1 GB on the card) and 14% more time;
  global context 1.8 GB and 11%.

## Removals and restorations of adopted parts

Plane-contrast readouts off (run 9), the medial-lateral attention bias back (15), the view label
bias off (14), the detail-view embedding off (17) and the report-attribute loss on (12) all land
within 0.0007 of their incumbent at ×4 with every label's interval across zero, apart from MCL
(45 positives), which alone carries up to 0.0005 of the macro. The scanner confusion head beside
L2-SP loses 0.0016 with the interval below zero (run 13). The hard-pair final phase is level
(run 8).

## How far two trainings of one recipe are apart

Runs 26 and 28 trained an incumbent's recipe with modules that stayed at the identity; their
L2-SP penalty, drift and training loss equal the incumbent's to the third decimal.

| Incumbent | Its ×4 / ×1 | The inert repeat's ×4 / ×1 | Difference |
|---|---|---|---|
| Run 10 | 0.938618 / 0.931247 | 0.937951 / 0.930072 (run 26) | −0.0007 / −0.0012 |
| Run 27 | 0.939634 / 0.930902 | 0.938639 / 0.929592 (run 28) | −0.0010 / −0.0013 |

- **Two readings fit:** an adopted run is the high draw of its recipe, since adoption selects
  high runs, and a repeat of this ResNet-50 recipe differs by about 0.001 at ×4 (the ±0.0003
  taken from the ResNet-18 is too narrow); or an inert module in the block is not free.
- **Consequence:** most single-run differences of this campaign are inside that distance, and
  the both-multipliers rule rejected sixteen runs in a row against the run-10 incumbent, all on
  the ×1 side.
- **The first seven epochs of run 8** share the incumbent's recipe and differ from it by −0.0013
  to +0.0012 at ×1 from one epoch to the next.

## Open

- **The incumbent has no public score.** S151 (0.959) is the starting recipe on the same fold at
  ×4 τ 0.9.
- **The reference's PF OA** (0.9155) is above 25 of the 26 campaign runs (0.9105 to 0.9162); a
  high draw on one label, or a difference between a fold of a three-fold run from main and a
  single-fold campaign run, not separated.
- **The label path reading the decoder's features** (tokens or routing) was proposed by the audit
  of run 5 and kept out by the contract.
