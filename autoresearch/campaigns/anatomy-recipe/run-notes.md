# Anatomy-recipe campaign: run notes

One entry per run, against the incumbent at the time. The control and first incumbent is
`rn18-anatomy-map-centre-control`: ×4 0.926762, ×1 0.917074, training 1,391 s.

## Harness

- **`report-attribute-loss-off` blocked before training (2026-10-08):** the review asks every
  candidate for a logged proof that its change is active, and the seed forbade any source change.
  Requeued as `report-attribute-loss-off-2` with one validation-only metric allowed. A seed that
  changes only a config value has to allow that metric.

## Run 1: `jitter-shared-by-the-two-contrasts-of-a-plane` — not adopted

One jitter draw per plane instead of one per series, so the two contrasts of a plane stay
registered under the posed crops.

| | Control | Run | Δ |
|---|---:|---:|---:|
| ×4 | 0.926762 | 0.926031 | −0.000731 (95% −0.0022 to +0.0008) |
| ×1 (best epoch) | 0.917074 (epoch 9) | 0.917226 (epoch 8) | +0.000152 |

- **Rule:** higher at ×1, lower at ×4, so not adopted.
- **Cost:** training 1,406 s, 1.011 times the control, as the seed stated.
- **Per label at ×4:** down Fracture −0.0031, Lateral Meniscus −0.0024, Synovitis −0.0021, MCL
  −0.0016, Lateral OA −0.0014; up Baker's +0.0015, Medial Meniscus +0.0009, Contusion +0.0005.
  No interval excludes zero.
- **Per epoch at ×1:** behind by 0.001 in epochs 1, 2, 5 and 6, level in 3 and 4, ahead only at
  epoch 8.
- **Reading (one run):** registering the two contrasts under the jitter gave no gain here. The
  independent jitter also acts as an augmentation of the pair's relative position, which the
  shared draw removes; the labels that fell are not ones that depend on comparing the two
  contrasts at a point, so this run does not separate the two effects.

## Run 2: `report-attribute-loss-off-2` — adopted

The report-attribute loss at weight 0 beside the anatomy map (the heads stay, untrained).

| | Control | Run | Δ |
|---|---:|---:|---:|
| ×4 | 0.926762 | 0.927291 | +0.000529 (95% −0.0004 to +0.0015; 80% −0.0001 to +0.0012) |
| ×1 (best epoch) | 0.917074 (epoch 9) | 0.917342 (epoch 9) | +0.000268 |

- **Rule:** at least the control at both multipliers and prediction 50.8 s against 51.8 s, so
  adopted by hand under the contract; the runtime's own verdict was reject (its 80% interval
  reaches −0.0001). It is the incumbent for every later seed.
- **Cost:** training 1,352 s, 0.972 times the control.
- **Per label at ×4:** up Contusion +0.0034, Lateral OA +0.0025, MCL +0.0016, Fracture +0.0015,
  ACL +0.0011; down Synovitis −0.0017, Medial Meniscus −0.0016 (95% −0.0032 to +0.0000), Baker's
  −0.0009. No interval excludes zero.
- **The map did not change:** overlap femur 0.904, tibia 0.897, joint 0.759, kneecap 0.734,
  kneecap joint 0.669, plane error 1.78 mm, against 0.905, 0.899, 0.759, 0.735, 0.675 and 1.73 mm
  for the control. The two auxiliary losses were not competing for the map.
- **Against the earlier measurement:** removing the loss cost −0.0006 at ×4 before the 84/93 mm
  views and the map; beside the map it is +0.0005. One run each; the sign changed, and Synovitis,
  which the seed expected to gain, fell.
- **Keeper:** the next training (coronal 108 mm) had launched on the control before the adoption;
  aborted after two minutes and requeued unchanged as `coronal-fluid-crops-at-108-mm-2`.

## Harness: `sagittal-fluid-crops-at-112-mm` blocked before training

The candidate ran two full-batch bicubic samplings every step (a parent-canvas copy and the wide
cut) and chose between them with `torch.where`; the review blocked it on the cost rule after one
repair left it unchanged. Requeued as `sagittal-fluid-crops-at-112-mm-2` with a one-pass cut of my own wording, which was
blocked too: masking the larger canvas does not reproduce the unwidened views exactly and added
boolean indexing. Requeued again as `sagittal-fluid-crops-at-112-mm-3`, pointing the developer at
the cut the review approved for the coronal seed (series listed once in the collate, each slice
sampled once, the unwidened views from a window of their former canvas size). Lesson: when one
seed's implementation of a shared mechanism passes review, the sibling seed names that source
instead of describing the mechanism again.

## Run 3: `coronal-fluid-crops-at-108-mm-2` — not adopted

The fluid coronal crop at 108 mm instead of 93 mm (scale 1.157 logged on validation), canvas
margin 24.5 mm. Against the new incumbent (run 2).

| | Incumbent | Run | Δ |
|---|---:|---:|---:|
| ×4 | 0.927291 | 0.925609 | −0.001682 (95% −0.0036 to +0.0001; 80% −0.0029 to −0.0005) |
| ×1 (best epoch) | 0.917342 | 0.915523 (epoch 8) | −0.001819 |

- **Rule:** lower at both multipliers, not adopted.
- **Cost:** training 1,434 s, 1.061 times the incumbent (allowed 1.08); ×4 prediction 57.0 s
  against 50.8 s, 1.12 times.
- **Per label at ×4:** down Fracture −0.0064, Lateral OA −0.0043, MCL −0.0030, Synovitis −0.0029,
  Lateral Meniscus −0.0023, Baker's −0.0018; up Contusion +0.0020, Medial Meniscus +0.0014. No
  single interval excludes zero; ten of twelve labels fell.
- **Per epoch at ×1:** level for three epochs, then behind by 0.0007 to 0.0021 from epoch 4 on.
- **The map and pose did not change:** overlaps and plane error (1.74 mm) are the incumbent's;
  12% of the fluid coronal series sit at the shift limit.
- **Reading (one run):** the wider field costs resolution on every label (0.48 mm per pixel
  instead of 0.415), and the collateral ligaments and condyle edges it was meant to bring inside
  the image did not pay for it: MCL and Lateral OA fell most among the labels read there. Same
  direction as the anatomy-masks run that widened both series of a plane by a ninth.

## Run 4: `femur-front-in-the-kneecap-routes` — not adopted

The femur's front at kneecap height read in full, instead of at a quarter, in the kneecap and
front-core routes.

| | Incumbent | Run | Δ |
|---|---:|---:|---:|
| ×4 | 0.927291 | 0.926764 | −0.000528 (95% −0.0014 to +0.0003) |
| ×1 (best epoch) | 0.917342 | 0.916965 (epoch 9) | −0.000377 |

- **Rule:** lower at both multipliers, not adopted.
- **Cost:** training 1,371 s, 1.014 times the incumbent; ×4 prediction 51.9 s against 50.8 s.
- **Its own targets at ×4:** PF OA −0.0001, Effusion −0.0009, Synovitis +0.0010. Elsewhere Lateral
  Meniscus −0.0035 (95% −0.0075 to +0.0001), Lateral OA −0.0021, ACL −0.0018; Medial Meniscus
  +0.0010.
- **The change was small in effect:** the applied increment to the route weights averaged 0.017,
  and the three targets already spent 44%, 47% and 22% of their attention on that membership
  (Effusion, PF OA, Synovitis).
- **Reading (one run):** the routes were already reading the femur's front; raising its weight
  from a quarter to full on 4.6% of the tokens moved the three targets by 0.001 or less. The
  labels that fell are not read through these routes, so their movement is not explained by the
  mechanism.

## Run 5: `sagittal-fluid-crops-at-112-mm-3` — not adopted

The fluid sagittal crop at 112 mm instead of 93 mm, canvas margin 26.5 mm, with the cut approved
for the coronal seed. First run decided by the runtime's 80% rule.

| | Incumbent | Run | Δ |
|---|---:|---:|---:|
| ×4 | 0.927291 | 0.926423 | −0.000868 (95% −0.0024 to +0.0006; 80% −0.0019 to +0.0001) |
| ×1 (best epoch) | 0.917342 | 0.917443 (epoch 9) | +0.000101 |

- **Rule:** the 80% interval does not lie above zero; rejected by the runtime.
- **Cost:** training 1,515 s, 1.121 times the incumbent, above the 1.1 the seed allowed (its
  first epoch shared the GPU with a short analysis pass); ×4 prediction 60.6 s against 50.8 s,
  1.19 times.
- **Per label at ×4:** the labels the wider field was meant for moved the right way, PF OA +0.0032
  and Lateral OA +0.0030, ACL +0.0015; against them Fracture −0.0060, MCL −0.0042, Effusion
  −0.0029 (95% −0.0064 to +0.0006), Lateral Meniscus −0.0021, Contusion −0.0018. No interval
  excludes zero.
- **Reading (one run):** unlike the coronal widening, which lowered ten labels, the sagittal one
  trades: the kneecap joint gains from being inside the image and the fine-detail labels pay for
  0.50 mm pixels instead of 0.417. That is the trade the per-knee size seed and the kneecap seed
  try to avoid, by not widening every knee.

## Run 6: `kneecap-inside-the-fluid-sagittal-crop-per-knee-2` — not adopted

The fluid sagittal crop moved forward per knee until the kneecap is inside (mean applied move
4.1 mm on validation), field unchanged at 93 mm, canvas margin 22.5 mm.

| | Incumbent | Run | Δ |
|---|---:|---:|---:|
| ×4 | 0.927291 | 0.927094 | −0.000198 (95% −0.0014 to +0.0009; 80% −0.0010 to +0.0006) |
| ×1 (best epoch) | 0.917342 | 0.917025 (epoch 9) | −0.000317 |

- **Rule:** the 80% interval does not lie above zero; rejected by the runtime.
- **Cost:** training 1,433 s, 1.060 times the incumbent (1.08 allowed); ×4 prediction 56.4 s
  against 50.8 s.
- **Per label at ×4:** nine labels up, by 0.0001 to 0.0015: PF OA +0.0015 (the target), Lateral
  Meniscus +0.0011, MCL +0.0006. Fracture −0.0046 and Baker's −0.0013 (95% −0.0035 to +0.0005)
  carry the macro below the incumbent. No interval excludes zero.
- **Per epoch at ×1:** ahead of the incumbent in epochs 1 to 7 (by up to 0.0014), behind by
  0.0003 in the last three.
- **Reading (one run):** moving the crop instead of widening it kept the resolution and the PF OA
  gain of run 5 in part (+0.0015 against +0.0032) without run 5's losses on MCL, Effusion and
  the menisci. Baker's falling fits the mechanism: the crop moves forward, so the back of the
  knee leaves it. Fracture fell in five of the six runs of this campaign by 0.003 to 0.006 (60
  positives); it is the noisiest label here and is not evidence about any one of them.

## Run 7: `crop-size-per-knee-from-the-first-look` — not adopted by the 80% rule

The field of the four sagittal and coronal views scaled per knee by the first look's size estimate
(size ÷ 17.087, clamped 0.87 to 1.13), canvas margin 23.5 mm, axial untouched. Recipe and
measurements from the knee-size study (`outputs/analysis/k/knee-size-crops/`).

| | Incumbent | Run | Δ |
|---|---:|---:|---:|
| ×4 | 0.927291 | 0.927608 | +0.000317 (95% −0.0015 to +0.0022; 80% −0.0009 to +0.0015) |
| ×1 (best epoch) | 0.917342 | 0.917680 (epoch 9) | +0.000338 |

- **Rule:** the 80% interval does not lie above zero; rejected by the runtime. It is the only run
  of the campaign above the incumbent at both multipliers, and the highest at each; under the
  both-multipliers rule of runs 1 to 4 it would have been adopted.
- **Cost:** training 1,435 s, 1.061 times the incumbent (1.2 allowed); ×4 prediction 57.6 s
  against 50.8 s, 1.13 times; peak memory 4.0 GB.
- **Activation on the 874 validation knees, last epoch:** mean scale 1.000, spread 0.067, 5.1%
  of knees at a clamp, as the study predicted on 150 knees (1.00, 0.07, 5.3%). The map and plane
  are the incumbent's (plane error 1.81 mm against 1.78 mm).
- **Per label at ×4:** up MCL +0.0085 (95% −0.0056 to +0.0246, 47 positives), Synovitis +0.0025,
  Medial OA +0.0018, Medial Meniscus +0.0013, Baker's +0.0013, Effusion +0.0006; down Fracture
  −0.0041, Lateral OA −0.0032, Lateral Meniscus −0.0028, Contusion −0.0017. No interval excludes
  zero; the macro gain rests largely on MCL.
- **Per epoch at ×1:** ahead of the incumbent in seven of ten epochs, by up to 0.0007.
- **Reading (one run):** the first field change of the campaign that did not lose, and the only
  one that keeps the mean resolution. The lateral labels fell in this run and in run 3; the study
  showed small knees lose the margin at the collateral edges with per-knee sizing (34.8% to
  17.4% of series with that region inside), which fits a lateral-edge cost. Next steps the result
  points to: the axial centring the study measured (coverage 84.3% to 91.9%, 98.5% with
  resizing), and a clamp that only shrinks the field for no knee below today's margin.

## Adoption rule from run 5 on

The user set adoption to the runtime's verdict on 2026-10-08 (80% paired-bootstrap interval of
the ×4 improvement strictly above zero). Runs 1 to 4 were decided by "at least the incumbent at
both multipliers", applied by hand, which needed a stop at the end of each run and cost 38 idle
minutes once, when a training launched before an adoption. No hand adoptions or stops since.

## Stopped by the user, 2026-10-08 17:20 CEST

Forced stop. The joint-centre offset training (`joint-centre-offset-from-the-training-knees-2`)
was aborted in its first minutes and gives no result. Not run: that seed and
`tokens-on-the-14-by-14-grid-2` (in development when the campaign stopped). Seven runs decided,
one adoption (run 2); the incumbent is `report-attribute-loss-off-2` (×4 0.927291, ×1 0.917342).
