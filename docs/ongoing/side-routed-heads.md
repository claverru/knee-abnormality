# Side-routed shared heads

**Status (2026-10-05): built in worktree `knee-abnormality-side-routed-heads`, branch
`side-routed-heads`, uncommitted; [first run](#first-run-2026-10-05) at 0.923102 against
0.924049, [second run](#second-run) at 0.923534 and 0.952 public (S112), [fourth run](#fourth-run)
at 0.923952 and **0.953 public (S113)**; the design below is the [fifth build](#fifth-build-2026-10-05),
whose [run](#fifth-run) reached 0.924476 and 0.952 public (S114).** One shared meniscus classifier and one shared OA
classifier score every token without knowing its compartment, and geometry decides which target
the evidence counts for. Designed with the user on 2026-10-05, after the
[fixes-and-removals campaign](../../autoresearch/campaigns/fixes-and-removals/INDEX.md) left
every position encoding at a tie.

## Runs at a glance

**Reference since 2026-10-06 19:20: run 24** (×4 0.925067, ×1 0.917070, S118 0.953). The fixes-and-removals
worktree was merged into main by the user that evening; this line continues alone and every later run is
compared with run 24.

Fold 1 of `all_5fold_scanner_v2`, seed 1110484336, ResNet-18. Old recipe on this fold: ×4 0.924049,
public 0.950 (S108, S111). Same-fold sibling runs differ by about 0.0004 of macro (SD).

| # | Run (`outputs/training/…`) | Main change | ×4 macro | Public |
|---|---|---|---:|---:|
| 1 | `20261005-181012-side-routed-heads` | Shared heads routed by a learned boundary and side | 0.923102 | |
| 2 | `20261005-185535-side-routed-knee-frame` | Knee frame, learned offsets, position code | 0.923534 | S112 0.952 |
| 3 | `20261005-194836-side-routed-attention-prior` | Attention prior, extra auxiliary loss | 0.923085 | |
| 4 | `20261005-202001-side-routed-run4` | Prior kept, auxiliary loss removed | 0.923952 | S113 0.953 |
| 5 | `20261005-205251-side-routed-run5` | Octants from the fitted centre, learned rotation, no offsets | 0.924476 | S114 0.952 |
| – | campaign `separate-oa-frame-with-knee-level-term` | Second frame for OA, PF OA front-and-up, knee-level term | 0.924485 | S115 0.953 |
| 6 | `20261006-073158-side-routed-hard-octants` | One frame, hard cuts, front-and-up, knee-level term | 0.924432 | |
| 7 | `20261006-081303-side-routed-soft-gradient-own-routes` | Soft gradient through the cuts; MCL, Effusion, Baker's routed | **0.924845** | S116 0.952 |
| 8 | `20261006-084210-side-routed-study-frames` | Frame inferred per study (six regressed numbers) | 0.924320 | S117 0.952 |
| 9 | `20261006-092326-side-routed-fine-landmarks` | 14×14 tokens for the shared heads, landmark frame, slow frame | 0.922842 | |
| 10 | `20261006-095814-side-routed-fine-fused-routed-reports` | Deep features on the fine tokens, frame at head rate, routed v4 report targets | 0.924423 | |
| 11 | `20261006-103317-side-routed-study-frames-landmarks` | Run 8 plus the landmark frame, 7×7 tokens, v3.1 reports | 0.924119 | |
| 12 | `20261006-112944-side-routed-eighteen-regions` | Eighteen regions with learned sizes, overlap assignment, routed v4 reports, own position in the fixed frame | 0.923687 | |
| 13 | `20261006-115457-side-routed-regions-v31-reports` | Run 12 with the v3.1 reports on unrouted heads | 0.924122 | |
| 14 | `20261006-121814-side-routed-regions-acl-unrouted` | Run 13 with ACL unrouted | 0.923936 | |
| 15 | `20261006-124148-side-routed-regions-shared-frame` | Run 14 on one shared learned frame, no per-knee head | 0.924141 | |
| 16 | `20261006-130551-side-routed-regions-mirrored-position` | Run 14 with compartment-relative position for the shared heads | 0.924172 | |
| 17 | `20261006-132943-rn50-side-routed-regions` | ResNet-50 attempt, stopped by the user after three minutes | none | |
| 18 | `20261006-135237-side-routed-regions-soft-gradient-knee-level` | Run 12 with hard cuts and the soft gradient, knee-level term on every routed target | 0.923121 | |
| 19 | `20261006-141739-side-routed-regions-core-kept-by-sides` | Run 18 with the core no longer taken from the side labels | 0.923473 | |
| 20 | `20261006-144444-side-routed-regions-wider-table` | Run 19 with the wider region table, Synovitis unrouted | 0.923483 | |
| 21 | `20261006-151245-side-routed-regions-nod-per-knee` | Run 20 with per-knee centre, nod and sizes; roll and turn shared | 0.923947 | |
| 22 | `20261006-153716-side-routed-regions-kneecap-above-lower-cut` | Run 21 with PF OA and Effusion on the front above the lower cut | 0.924358 | |
| 23 | `20261006-160557-side-routed-meniscus-spot-frame-eight-sections` | Frame hung on the two meniscus spots; eight sections per level | 0.924251 | |
| 24 | `20261006-163650-side-routed-spot-frame-ring-sectors` | Spot frame with eight sectors centred on the axes, core and outer ring; region table from the attention | **0.925067** |S118 0.953 |
| 25 | `20261006-171738-side-routed-four-anchor-frame` | Run 24 with the notch and the kneecap as further anchors; no free tilt | 0.924419 | |
| 26 | `20261006-180530-side-routed-acl-all-heights` | Run 24 with ACL reading its region at all heights | 0.923636 | |
| 27 | `20261006-183038-side-routed-routed-detectors` | Run 24 with the old tear and OA detectors over the regions; stopped by the user | | |
| 28 | `20261006-184303-side-routed-midway-cuts` | Run 24 with the four cuts placed midway between attention spots, no gradient | 0.924384 | |
| 29 | `20261006-192417-side-routed-held-cuts-balanced` | Only the three cuts labels hold from both sides (midline, one height cut, core), equal pull per side | 0.925098 | |
| 30 | `20261006-200553-side-routed-mirrored-held-division` | Run 24 with the mirrored 15-section division found by the search (three heights, core 21 mm, four sectors) | 0.923564 | |
| 31 | `20261006-204225-side-routed-pf-oa-front-half` | Run 24 with PF OA and Effusion on the front half and the upper core, as in run 23 | 0.924552 | |

The 24 campaign runs are in `autoresearch/campaigns/side-routing/run-notes.md`. The model in the
worktree is the eighteen-region build (last section); the code of runs 10 and 11 is in their source
snapshots. The
Design section below describes the fifth run, and each later build section lists what it changed.

## Principle

**The fitted centre and a learned rotation cut the knee into eight octants; routing is a table
from octants to targets.** A shared classifier reads position-free tokens, so it cannot tell
which compartment a token is in. A cut is used for routing only where targets compete across
it, which holds it in place: medial against lateral targets across the midline, PF OA against
the two tibiofemoral OA targets across the front cut. No target pair is split by height, so the
height cut routes nothing.

- **Shared classifiers.** Medial and lateral meniscal tears, and medial, lateral and
  patellofemoral OA, are each one disease in several places. One classifier per disease learns
  from every compartment; on fold 1 Lateral Meniscus has 135 positives against 339 for medial,
  and Lateral OA 95 against 189.
- **Routing, because the classifier is shared.** One output has to become two or three targets,
  and the image cannot say which: the 84 and 93 mm views are centred on the knee, and the cue for
  the side (the fibula) is at their edge or outside.
- **Earlier evidence.** The half-knee loss is a shared detector pooled per half, adopted at
  +0.0031 with the own-side mask (2026-09-24). The three-compartment OA detector, shared over
  medial, lateral and patellofemoral regions, gave PF OA +0.0102 when adopted (2026-09-25).
  Neither was ever the predictor.

## Design

### Channels

The 12 targets are read from 9 channels. ACL, MCL, Effusion, Synovitis, Baker's, Contusion and
Fracture keep one channel each and pool unrouted over all valid tokens. Two shared channels feed
the five compartment targets:

| Shared channel | Targets it feeds | Routing weight of a token |
|---|---|---|
| `meniscus` | Medial Meniscus, Lateral Meniscus | medial, lateral |
| `oa` | Medial OA, Lateral OA, PF OA | medial, lateral, front |

The attention scorer, the plane-contrast readout and the view label bias have 9 outputs instead
of 12. Each of the 12 targets keeps its own bias.

### Routing

For a compartment target `k` of shared channel `h`, with `r_k` the token's routing weight in
[0, 1]:

```text
attention_k = softmax over the study's valid tokens of (score_h + prior_k + log r_k)
logit_k     = sum over tokens of attention_k * evidence_h + bias_k
```

| Target | Region | Routing weight |
|---|---|---|
| Medial Meniscus | medial | `r_medial` |
| Lateral Meniscus | lateral | `1 - r_medial` |
| Medial OA | medial and back | `r_medial * (1 - r_front)` |
| Lateral OA | lateral and back | `(1 - r_medial) * (1 - r_front)` |
| PF OA | front, both sides | `r_front` |

- `r_medial = sigmoid(m / width_m)` and `r_front = sigmoid(f / width_f)`, with `(m, f, h)` the
  token's coordinates in the knee frame below. A cut has no band: every token belongs to one
  side, softly near the cut.
- The two widths set how soft each cut is: a token one width from a cut counts 73% for its
  side. They are learned scalars, start at 5 mm and are kept within 1 to 40 mm.
- A token without a fitted centre takes 0.5 for every routing weight. If a study's routing
  weights for a target sum to less than one token, that target pools unrouted.

### Knee frame

```text
(m, f, h) = R * (s * u, a, z)
```

- `(u, a, z)` are the token's left-right, anterior and head-foot offsets from the fitted centre
  in mm, after the rotation from the headers: the scanner's left-right axis is rotated about
  the head-foot axis to the mean row direction of the study's axial and coronal series
  ([census](#visual-inspection-and-rotation-census)); a study without one keeps the scanner's
  axes.
- `s` is the side sign, so `m` is medial-positive for every knee.
- **`R` is one learned rotation** about the fitted centre, common to all studies: a rotation
  vector of three angles (nod about the medial axis, roll about the front axis, turn about the
  vertical axis), each `90° * tanh(parameter)`, starting at zero. The nod tilts the front cut so
  that the kneecap and the trochlea fall in front and the tibiofemoral joint line behind
  ([sketches](#octant-sketches)).
- **There is no offset and nothing per study.** A cut pinned to the centre can turn but cannot
  slide, so PF OA cannot widen its region to the whole knee. The earlier builds' lateral offset
  learned 2 to 3 mm and their front boundary moved between 12 and 27 mm with PF OA improving
  throughout.

### Position prior on the attention

```text
position           = (m, f, h), the knee-frame coordinates                in mm
score of target k  = content score of its channel + view bias
                     + log r_k                       (routed targets)
                     + prior_k(position)
prior              = Linear(42, 10)(sin and cos of the position at 7 wavelengths, 8 to 512 mm)
```

| Targets | Prior map | Medial coordinate it reads |
|---|---|---|
| The seven own-channel targets | one each | signed, medial positive |
| Medial and Lateral Meniscus | one shared | depth into the target's own compartment |
| Medial and Lateral OA | one shared | depth into the target's own compartment |
| PF OA | one | absolute |

- The map is linear, starts at zero and adds straight to the attention score, with no hidden
  layer or normalisation between the position and the score.
- Medial and lateral targets share one map read from their own side, so the prior carries no
  compartment either.
- A token without a fitted centre has a prior of zero.
- At inference the temperature divides the content score, the view bias, the prior and the
  routing term together.

### Knee side

The side is used by the routing and by the sign of the medial coordinate in the position prior,
and nowhere else.

- **Certain studies** (`explicit_coordinates_agree`, 2,072 of 4,407 in training): the metadata
  side routes. These are the only studies that train the side head.
- **Every other study** routes by a probability, as a mixture,
  `r_medial = p_left * r_medial(as a left knee) + (1 - p_left) * r_medial(as a right knee)`,
  with the gradient cut so the disease loss cannot move the head. At 0.5 the evidence is shared
  equally between the two sides.
  - **Side from coordinates alone (2,082 studies):** the image checks the metadata. The
    coordinates are a prior worth 4.33 in log-odds (their 98.7% agreement with reports), added to
    the side head's logit. When the two agree the side is near certain; a confident image
    overrules the coordinates, which is the case of a scanner with another coordinate
    convention. It also routes these studies from the first step, while the head is untrained.
  - **No side (170 studies):** the side head's probability alone.
- **Side head:** a small network (256 to 128, tanh, to 1) scores each of the study's axial and
  coronal tokens with a fitted centre; the logit for "left knee" is the mean of those scores
  weighted by `tanh(u / width)`. Its input is detached, so the side loss trains this network and
  its width and nothing else. The width is learned like the routing widths. Binary
  cross-entropy on the certain studies, weight 0.1. A study without such tokens has no head
  logit.
- **Reflection:** a mirrored study looks like the other knee, so the head reads `-u` and its
  target flips when
  the training reflection is applied, and the probability used for routing is flipped back,
  because the coordinates stay those of the source. `reflect_batch` marks the mirrored pass so
  reflection TTA does the same.
- Series whose side comes from the bilateral split keep it.

### Removed in the same change

- The meniscus own-side mask and its midline band, the half-knee loss and its detector
  (`trainer.half_knee_loss_weight`), the OA detector and its regions, the anatomical attention
  bias (`model.anatomical_bias`), and the per-compartment attention and readout weights of the
  five compartment targets.
- The 1D slice position and the 2D image position on the tokens; the position prior on the
  attention and the token position code of the own-channel classifiers replace them.

### Token position code

A token read by an own-channel classifier carries a fixed sinusoidal code of its knee-frame
position (42 wavelengths from 8 to 512 mm per axis, norm 11.2, the scale of the reference's
slice-position code), added before the normalisation; the position is detached there. A token
read by a shared classifier carries none. A token without a fitted centre has a code of zero.
The report-attribute heads read the own-channel version.

Kept: the centre fit, the metadata side policy, the report-attribute heads.

### Logged per epoch

The three frame angles; the mean absolute header rotation of the knee frame; the three widths; the side head's accuracy on certain studies, in training and validation;
for coordinates-only studies, how often the final probability sides with the coordinates; the
share of studies in each side state; the mean routing weight per compartment; the attention
mass of each target per view.

## Reference and first run

The reference is the campaign incumbent `slab-rule-and-positive-std-requeue`
(`outputs/autoresearch/fixes-and-removals/experiments/slab-rule-and-positive-std-requeue-20261005T155303Z-59280bbc/training`,
fold 1 of `all_5fold_scanner_v2`, seed 1110484336, ×4 0.924049, ×1 0.916134). The worktree
starts from that run's code and `configs/training.yaml` carries its fold and seed.

```bash
rtk proxy uv run --locked just train configs/training.yaml --name side-routed-heads
```

One run of this fold moves the macro by about 0.0005 and a 60-positive label by about 0.01 from
a changed data stream alone
([run notes](../../autoresearch/campaigns/fixes-and-removals/run-notes.md)), so read the five
compartment targets together and the validation loss beside the macro.

## First run (2026-10-05)

`outputs/training/20261005-181012-side-routed-heads`, fold 1, seed 1110484336, one run.

| | Run | Reference | Δ |
|---|---:|---:|---:|
| Macro AUC ×4 | 0.923102 | 0.924049 | −0.0009 |
| Macro AUC ×1 (epoch 10) | 0.914604 | 0.916134 | −0.0015 |
| Soft-target BCE ×4 | 0.4025 | 0.4037 | −0.0012 |

| Target | Run | Reference | Δ | Positives | Routed |
|---|---:|---:|---:|---:|---|
| ACL | 0.9415 | 0.9458 | −0.0043 | 131 | no |
| MCL | 0.9177 | 0.9176 | +0.0001 | 47 | no |
| Medial Meniscus | 0.9460 | 0.9469 | −0.0009 | 339 | yes |
| Lateral Meniscus | 0.9054 | 0.9081 | −0.0027 | 135 | yes |
| Medial OA | 0.9241 | 0.9229 | +0.0012 | 189 | yes |
| Lateral OA | 0.9016 | 0.9035 | −0.0019 | 95 | yes |
| PF OA | 0.9002 | 0.8994 | +0.0008 | 290 | yes |
| Effusion | 0.9190 | 0.9182 | +0.0007 | 244 | no |
| Synovitis | 0.8783 | 0.8900 | −0.0118 | 114 | no |
| Baker's | 0.9630 | 0.9636 | −0.0007 | 114 | no |
| Contusion | 0.9531 | 0.9468 | +0.0063 | 93 | no |
| Fracture | 0.9275 | 0.9257 | +0.0018 | 60 | no |

- The five routed targets sum to −0.0035, which is −0.0003 of macro. Synovitis alone is −0.0010
  of macro; a label of its size moves by about 0.01 on this fold from a changed data stream.
- At ×1 the run led at epoch 1 (+0.0037) and trailed by 0.001 to 0.004 afterwards. Lateral
  Meniscus led through epoch 7 (up to +0.019) and PF OA through epoch 4; neither lead held.

### What the new parts learned

| | Epoch 1 | Epoch 9 |
|---|---:|---:|
| Side head accuracy on certain validation studies | 100% | 100% |
| Side head agreement with coordinates-only sides | 99.5% | 100% |
| Front boundary correction, mean | 13.5 mm | 27.0 mm |
| Midline correction, spread over studies | 0.6 mm | 3.2 mm |
| Medial/lateral width | 4.8 mm | 4.4 mm (epoch 6) |
| Front width | 4.9 mm | 4.7 mm (epoch 6) |

### Visual inspection and rotation census

Eight validation studies rendered with `scripts/adhoc/render_routing_views.py`
(`outputs/analysis/side-routed-views-20261005/`), and the rotation of every training series
measured from headers with `scripts/adhoc/census_knee_rotation.py`
(`outputs/analysis/knee-rotation-census-20261005/`).

- **Front boundary:** on the patellofemoral joint in all eight, 15 to 39 mm ahead of the fitted
  centre over 60 studies (mean 27.2, SD 5.6).
- **Fitted midline:** on the intercondylar notch on every coronal slice.
- **Learned midline:** a constant, not a per-study correction. Over 60 studies it moves lateral
  in 82%, by 2.1 mm on average (SD 2.6), and that drift is the same in every rotation band
  (correlation with the knee's rotation 0.04). The two largest corrections (8 and 9 mm) leave
  the notch for the lateral tibial spine.
- **Crops:** every coronal and sagittal crop holds the joint, both condyles and the plateau. The
  joint line sits at 44% to 69% of the crop height, mostly 55% to 60%, about ±5 mm between
  studies.
- **Rotation:** the routing uses the scanner's left-right axis, and the knee is rotated against
  it. Over 4,237 studies with a side, taking the axial and coronal row directions: 76% rotated
  outward, mean 4.5°; over 5° in 49%, over 10° in 19%, over 15° in 6%, over 20° in 2%. At 30 mm
  behind the centre, 10° misplaces the boundary by 5.3 mm and 20° by 10.9 mm. Axial and coronal
  agree (correlation 0.83, median difference 1.8°); sagittal series are angled on purpose and
  do not track them (0.43 to 0.50).

## Second build (2026-10-05)

Built after the first run, with the user: the knee frame, the learned offsets in place of the
per-study midline landmark, and the position code read unsigned by the shared classifiers and
signed by the others. The first run had no position code, a per-study midline landmark, a
front landmark centred on the fitted centre, and the scanner's axes.

Checked on four training studies on the GPU: the position code is zero at the start; the
rotation matches the census per study; with random position weights a side flip swaps the
medial and lateral logits of both shared channels exactly, leaves PF OA unchanged and moves the
own-channel logits; a mirrored test-time pass keeps each token's position; the offsets, the
front scorer and the position output receive gradients.

```bash
rtk proxy uv run --locked just train configs/training.yaml --name side-routed-knee-frame
```

Same fold, seed and reference as the first run.

### Second run

`outputs/training/20261005-185535-side-routed-knee-frame`, one run.

| | Second run | First run | Reference |
|---|---:|---:|---:|
| Macro AUC ×4 | 0.923534 | 0.923102 | 0.924049 |
| Macro AUC ×1 (epoch 10) | 0.914544 | 0.914604 | 0.916134 |
| Soft-target BCE ×4 | 0.4026 | 0.4025 | 0.4037 |

| Target | Second − reference | First − reference |
|---|---:|---:|
| ACL | −0.0060 | −0.0043 |
| MCL | +0.0005 | +0.0001 |
| Medial Meniscus | −0.0006 | −0.0009 |
| Lateral Meniscus | −0.0007 | −0.0027 |
| Medial OA | +0.0003 | +0.0012 |
| Lateral OA | −0.0002 | −0.0019 |
| PF OA | +0.0044 | +0.0008 |
| Effusion | +0.0015 | +0.0007 |
| Synovitis | −0.0098 | −0.0118 |
| Baker's | +0.0000 | −0.0007 |
| Contusion | +0.0037 | +0.0063 |
| Fracture | +0.0006 | +0.0018 |

- The five routed targets sum to +0.0032 (first run −0.0035). ACL and Synovitis together are
  −0.0158, which is −0.0013 of macro and more than the whole gap of −0.0005; the signed
  position code did not recover them.
- Learned at the end: lateral offset 2.2 mm, front offset 4.8 mm, up offset 1.3 mm; front
  boundary 22.4 mm on average (SD 4.7); widths 4.4 and 4.7 mm; side head 100%.
- **Kaggle: S112, 0.952 public** (×4, τ 0.9, no reflection TTA; bundle
  `outputs/submission/exports/side-routed-knee-frame-f1-x4-tau09`). The same fold and inference
  setting on the campaign recipe scored 0.950 with the OA detector off (S111) and 0.950 with it
  on (S108).
- The trained position code is about a tenth of the token's norm (1.7 and 1.3 against 15.6 on
  40 validation studies) and removing it changes the logits by 0.03 to 0.14, so this run barely
  used position.
- At ×1 the run led the reference through epoch 3 (+0.0041, −0.0002, +0.0007) and trailed by
  0.0012 to 0.0033 afterwards, as the first run did from epoch 2.

Not built: a learned crop shift from the same centre, with the zoom fixed. The joint line
varies by about ±5 mm between studies, under half a token, so it runs after this one.

## Third build (2026-10-05)

Built after the second run and an investigation of both runs
([findings](#investigation-of-the-first-two-runs)), as one combined change:

| Change | Reason |
|---|---|
| Side head reads detached tokens, pooled with a left-right weight | The side loss trained the trunk: a probe for the side scores 87% on the reference's features and 100% on the second run's |
| Front scorer reads detached tokens | Same reason |
| Position acts on the attention through a linear prior per target | The second run's token code ended at a tenth of the token's norm |
| Tokens carry no position code | The classifiers stay blind to the compartment by construction |
| The up offset is gone | It only shifted the token code |
| Routed auxiliary loss on the meniscus targets | The reference's half-knee loss was the one trunk-level term removed |
| Temperature also divides the prior and the routing term | At τ 0.9, S112 sharpened the content score only |

Checked on four training studies on the GPU: the side loss has no gradient on the trunk; the
prior is zero at the start; with a random prior a side flip swaps the medial and lateral logits
of both shared channels exactly, leaves PF OA unchanged and moves the own-channel logits; the
prior, the detector, the offsets, the three widths, the front scorer and the side head receive
gradients; a temperature of 0.5 sets the routing temperature and changes the logits.

```bash
rtk proxy uv run --locked just train configs/training.yaml --name side-routed-attention-prior
```

Same fold, seed and reference. Checkpoints of the first two runs no longer load in this code.

### Third run

`outputs/training/20261005-194836-side-routed-attention-prior`, one run: **×4 0.923085**
(reference 0.924049, second run 0.923534, first run 0.923102); ×1 0.914630; BCE 0.4027. Not
submitted.

Each run minus the reference at ×4, beside the 12 same-fold campaign runs minus the reference:

| Target | Third | Second | First | Siblings, mean ± SD | Third below how many of 12 |
|---|---:|---:|---:|---:|---:|
| ACL | −0.0039 | −0.0060 | −0.0043 | −0.0016 ± 0.0014 | 11 |
| MCL | −0.0128 | +0.0005 | +0.0001 | +0.0005 ± 0.0029 | 12 |
| Medial Meniscus | −0.0011 | −0.0006 | −0.0009 | −0.0002 ± 0.0007 | 11 |
| Lateral Meniscus | +0.0009 | −0.0007 | −0.0027 | −0.0015 ± 0.0024 | 3 |
| Medial OA | −0.0011 | +0.0003 | +0.0012 | +0.0005 ± 0.0008 | 11 |
| Lateral OA | +0.0031 | −0.0002 | −0.0019 | −0.0024 ± 0.0034 | 0 |
| PF OA | +0.0062 | +0.0044 | +0.0008 | −0.0007 ± 0.0005 | 0 |
| Effusion | +0.0010 | +0.0015 | +0.0007 | +0.0007 ± 0.0008 | 3 |
| Synovitis | −0.0042 | −0.0098 | −0.0118 | −0.0041 ± 0.0022 | 6 |
| Baker's | −0.0031 | +0.0000 | −0.0007 | +0.0001 ± 0.0015 | 12 |
| Contusion | +0.0026 | +0.0037 | +0.0063 | +0.0025 ± 0.0013 | 7 |
| Fracture | +0.0008 | +0.0006 | +0.0018 | +0.0072 ± 0.0044 | 10 |

- **Synovitis is back at the sibling mean** and ACL moved half-way; the five routed targets sum
  to +0.0080 (second run +0.0032), with PF OA and Lateral OA above all 12 siblings.
- **MCL fell to −0.0128, below all 12 siblings**, and is the whole macro gap (−0.0011); Baker's
  is also below all 12. Both were level in the first two runs.
- **The side head is weaker on detached tokens:** 96.5% on certain validation studies (93.5% in
  training) and 94.2% agreement with coordinates-only sides, against 100% before; its width
  went from 5 to 3.2 mm. The final side probability still agrees with the coordinates in 100%
  of coordinates-only studies; the 6.2% of studies with no side route by the head alone.
- **The front boundary sits at 12.4 mm** (SD 3.6), against 22.4 mm in the second run; the front
  offset is 2.8 mm and the lateral offset 2.9 mm. Widths 4.2 and 4.9 mm.
- **The position prior learned anatomical places.** Where each target's prior peaks, median
  over 60 validation studies, in mm (medial, ahead of the front boundary, up), with its spread
  over a study's tokens in logits:

  | Target | Peak | Spread |
  |---|---|---:|
  | ACL | −4, −20, −15 | 0.29 |
  | MCL | +39, −10, +3 | 0.25 |
  | Medial / Lateral Meniscus | 34 / 32 into the own side, −19, −17 | 0.38 |
  | Medial / Lateral OA | 21 into the own side, −18, −16 | 0.35 |
  | PF OA | −1, +45, +21 | 0.46 |
  | Effusion | +10, +4, +41 | 0.21 |
  | Synovitis | +9, +29, +26 | 0.20 |
  | Baker's | +26, −42, +8 | 0.34 |
  | Contusion, Fracture | no clear place | 0.15 |

  Removing the prior changes the logits by 0.02 to 0.19.
- The routed auxiliary loss fell from 0.549 to 0.437 over training, as the reference's
  half-knee loss did (0.559 to 0.436).

## Fourth build (2026-10-05)

The third build with two changes, from what the three runs showed:

| Change | Reason |
|---|---|
| The routed auxiliary loss and its detector are removed | Added on weak evidence; the shared routed classifier already is the per-half detector. MCL and Baker's fell below all 12 sibling runs only in the run that had it, and both sit on the medial side where it acts (a suspicion, not shown) |
| The side head is a per-token network instead of one linear layer on the pooled tokens | On detached tokens the linear head reached 96.5%, against 100% when it trained the trunk |

Kept from the third build: the detached side head and front scorer, the position prior on the
attention, position-free tokens, the knee frame, the two learned offsets and the temperature
on the prior and routing term.

The same GPU check as the third build passes: no side-loss gradient on the trunk, a zero prior
at the start, the exact medial/lateral swap under a side flip, gradients on every new part.

```bash
rtk proxy uv run --locked just train configs/training.yaml --name side-routed-run4
```

Same fold, seed and reference. Checkpoints of the earlier runs no longer load in this code.

### Fourth run

`outputs/training/20261005-202001-side-routed-run4`, one run: **×4 0.923952** (reference
0.924049; the 12 same-fold campaign runs span 0.9234 to 0.9248); ×1 0.915445; BCE at ×1 0.4085
against 0.4092. **Kaggle: S113, 0.953 public** (×4, τ 0.9, no reflection TTA; bundle
`outputs/submission/exports/side-routed-run4-f1-x4-tau09`), against 0.952 for the second run
(S112) and 0.950 for the campaign recipe on the same fold (S108, S111).

| Target | Fourth − reference | Siblings − reference, mean | `position-embeddings-off` − reference |
|---|---:|---:|---:|
| ACL | −0.0034 | −0.0016 | −0.0018 |
| MCL | −0.0031 | +0.0005 | −0.0018 |
| Medial Meniscus | −0.0002 | −0.0002 | |
| Lateral Meniscus | −0.0001 | −0.0015 | |
| Medial OA | +0.0030 | +0.0005 | |
| Lateral OA | +0.0022 | −0.0024 | |
| PF OA | +0.0082 | −0.0007 | |
| Effusion | +0.0012 | +0.0007 | +0.0021 |
| Synovitis | −0.0054 | −0.0041 | −0.0035 |
| Baker's | −0.0057 | +0.0001 | −0.0028 |
| Contusion | +0.0036 | +0.0025 | +0.0036 |
| Fracture | −0.0015 | +0.0072 | +0.0002 |

- **The five routed targets sum to +0.0131** (third run +0.0080, second +0.0033, first
  −0.0035); PF OA and Medial OA are above all 12 siblings.
- **MCL came back** from −0.0128 to −0.0031 with the auxiliary loss removed.
- **The side head is back at 99.8%** on certain validation studies (99.2% agreement with
  coordinates-only sides) as a per-token network on detached tokens.
- **The own-channel targets carry the rest.** Against the sibling mean they sum to −0.0196
  (−0.0016 of macro), most of it Fracture (−0.0087) and Baker's (−0.0058), both below all 12
  siblings. `position-embeddings-off`, the reference recipe without the token position codes,
  is the one sibling with the same profile: the lowest sibling on Baker's and, at +0.0002
  against +0.0043 to +0.0117 for the others, on Fracture. One run each.
- Learned: lateral offset 2.0 mm, front offset 2.9 mm, front boundary 11.5 mm, widths 4.5, 4.9
  and 4.3 mm.
- At ×1 the run led the reference through epoch 4 (+0.0070, +0.0022, +0.0027, +0.0022) and
  trailed by 0.0006 to 0.0013 afterwards.

## Fifth build (2026-10-05)

Designed with the user after the fourth run, as one combined change:

| Change | Reason |
|---|---|
| The frame is the fitted centre plus one learned rotation; the lateral offset, the front offset and the per-study front scorer are removed | The user's octant picture; the offsets were small or unneeded, and a cut that cannot slide cannot be abused |
| Medial and Lateral OA take the back of their side only | The front cut then has targets competing across it |
| Own-channel classifiers read tokens with a position code at the reference's scale | Fracture and Baker's were below all 12 siblings in the fourth run, the profile of `position-embeddings-off` |

Kept from the fourth build: the detached per-token side head, the position prior on the
attention, position-free tokens for the shared classifiers, the header rotation and the
temperature on the prior and routing term.

Checked on four training studies on the GPU: the frame starts as the identity and the three
angles receive gradients; with a random prior a side flip swaps the medial and lateral logits
of both shared channels exactly; the three back and front routing weights sum to one; the
token code has norm 11.2; no side-loss gradient on the trunk.

At the start the front cut is vertical through the centre, so Medial and Lateral OA begin on
the back half of their side until the nod is learned.

```bash
rtk proxy uv run --locked just train configs/training.yaml --name side-routed-run5
```

Same fold, seed and reference. Checkpoints of the earlier runs no longer load in this code.

### Fifth run

`outputs/training/20261005-205251-side-routed-run5`, one run: **×4 0.924476** (reference
0.924049, fourth run 0.923952); ×1 0.915823. **Kaggle: S114, 0.952 public** (×4, τ 0.9, no
reflection TTA; bundle `outputs/submission/exports/side-routed-run5-f1-x4-tau09`), against
0.953 for the fourth run (S113) and 0.952 for the second (S112). The public score followed the
routed targets' sum (+0.0033, +0.0131, +0.0090 for S112, S113, S114) and not the local macro
(0.9235, 0.9240, 0.9245); three submissions at a resolution of 0.001.

| Target | Fifth − reference | Fourth − reference | Siblings − reference, mean | Fifth below how many of 12 |
|---|---:|---:|---:|---:|
| ACL | −0.0041 | −0.0034 | −0.0016 | 11 |
| MCL | +0.0012 | −0.0031 | +0.0005 | 4 |
| Medial Meniscus | −0.0012 | −0.0002 | −0.0002 | 11 |
| Lateral Meniscus | +0.0015 | −0.0001 | −0.0015 | 3 |
| Medial OA | +0.0013 | +0.0030 | +0.0005 | 3 |
| Lateral OA | +0.0003 | +0.0022 | −0.0024 | 3 |
| PF OA | +0.0071 | +0.0082 | −0.0007 | 0 |
| Effusion | +0.0031 | +0.0012 | +0.0007 | 0 |
| Synovitis | −0.0027 | −0.0054 | −0.0041 | 2 |
| Baker's | −0.0035 | −0.0057 | +0.0001 | 12 |
| Contusion | +0.0021 | +0.0036 | +0.0025 | 8 |
| Fracture | +0.0000 | −0.0015 | +0.0072 | 11 |

- **First run of the line above the reference** (+0.0004), with the own-channel targets summing
  to −0.0039 against −0.0143 in the fourth run: MCL, Synovitis, Effusion, Baker's and Fracture
  all moved up with the token position code. Baker's is still below all 12 siblings, Fracture
  and ACL below 11.
- **The routed five sum to +0.0090** (fourth run +0.0131). Medial OA and Lateral OA, now on the
  back of their side only, gave back 0.0017 and 0.0019; PF OA stays above all siblings.
- **The frame barely turned:** nod −5.1° (the front cut leans 5° towards front-and-up, against
  the 25° to 45° of the sketches), roll −2.4°, turn −1.0°. The front width grew from 5.0 to
  5.7 mm; the medial width ended at 4.5 mm.
- The side head reached 99.8% on certain validation studies.
- At ×1 the run led the reference through epoch 5 and ended 0.0003 behind.

### Octant sketches

`outputs/analysis/octant-sketches-20261005/` (loader only): the cuts drawn on midline sagittal,
coronal and axial slices of two studies, and `tilt-sheet/sheet.png` with the front cut at 0°,
25° and 45° on ten validation studies. The fitted centre sits at the notch, about 15 mm above
the joint line on the coronal slice. With the cut through the centre, a nod of 25° to 45°
leaves the kneecap, the trochlea and the front of the femur in front and the joint line, the
tibial plateau and the back of the condyles behind in all ten. The angles are illustrations,
not learned values.

### Patient motion between series

Measured by a subagent on the full training set, 2026-10-05
(`outputs/analysis/inter-series-motion-20261005/`; scripts in `scripts/adhoc/inter_series_motion/`). Every pair of same-plane series is resampled by
header geometry onto a 1 mm grid and aligned by a normalised-gradient-field correlation over
±20 mm; 12,083 comparable pairs in 4,398 studies (pairs angled more than 3° apart excluded).

| | Median | p90 | p95 | p99 | Over 3 mm | Over 5 mm |
|---|---:|---:|---:|---:|---:|---:|
| Same-plane shift, all pairs | 0.41 mm | 1.15 | 1.48 | 2.71 | 0.84% | 0.29% |
| Worst pair per study | 0.66 mm | | 1.87 | 3.77 | 1.61% of studies | 0.61% (27 studies) |
| Crossing planes, 200 studies, along the intersection | 0.36 mm | 1.33 | 1.82 | 3.20 | 1.1% | 0.21% |

- Validation: a synthetic shift is recovered with a median error of 0.08 mm; a triangle
  closure over same-plane and crossing estimates leaves a median residual of 0.14 mm (p99
  1.11); adjacent slices of one series give a median of 0.36 mm.
- The nine largest cases viewed are real repositioning, some beyond the 20 mm search limit.
  Shift grows with the gap in series number (median 0.36 mm for neighbours, 1.08 mm seven
  apart); repeated same-contrast series carry the heavier tail.
- **Decision: no per-series correction.** Tokens are 12 to 17 mm apart and the cuts are soft
  over about 4.5 mm; 99% of pairs move under 2.7 mm. Not measured: through-plane shift and
  rotation.

### Investigation of the first two runs

A subagent compared both runs with the reference and with 12 same-fold campaign runs.

- **ACL and Synovitis:** both runs are below all 12 sibling runs on Synovitis and below 11 or
  12 on ACL. The reference is a high draw on both: against the sibling mean the second run is
  −0.0044 (ACL) and −0.0057 (Synovitis). PF OA is above all 12 (+0.0051 against their mean).
- **The two runs are one draw:** their per-study logit changes against the reference correlate
  0.89 (ACL) and 0.93 (Synovitis), against about 0.48 for sibling pairs.
- **Position codes are not the cause:** `position-embeddings-off` on the reference recipe gave
  Synovitis −0.0035 and ACL −0.0018, the sibling mean.
- **Attention left the axial fluid view** for every own-channel target, outside the range of
  13 campaign runs (training, last epoch): ACL 0.091 to 0.059, Synovitis 0.415 to 0.335,
  Effusion 0.487 to 0.425. Synovitis' coronal-fluid single-view AUC fell from 0.812 to 0.766.
- **Routing localises:** attention centres at +22 mm (Medial Meniscus), −21 (Lateral Meniscus),
  +19 (Medial OA, reference +11), −17 (Lateral OA, reference −11), 34 mm anterior (PF OA,
  reference 16).
- **OA alone and together (×4, second run minus reference):** against studies with OA only in
  another compartment, PF OA +0.0099 (reference 0.7377) and Medial OA +0.0047 (0.8288), Lateral
  OA −0.0002 (0.8134); isolated Medial OA against OA-free knees −0.0096 (53 positives).
- The anatomical attention bias was already off in the reference (`model.anatomical_bias:
  false`), so it was not removed by this line.
- No coding bug found in initialisation, parameter groups, the routing fallback, the view-bias
  indexing or the detached side probability.

## Variants considered

Each is one change against the design:

| Variant | State |
|---|---|
| MCL routed from medial tokens and ACL from a window on the midline | Built, then removed: both have their own classifier, so they have nothing to route |
| No position at all | The first run |
| A learned lateral offset, a learned front offset, a per-study front scorer; front tokens also counting for Medial and Lateral OA; no token position for the own-channel classifiers | The first four runs |
| A position code on the tokens, unsigned for the shared classifiers | The second run: the code stayed at a tenth of the token's norm |
| The side head on the mean of undetached tokens | The first two runs: it trained the trunk |
| A routed half-knee auxiliary loss on the meniscus targets, weight 0.075; a linear side head on pooled detached tokens | The third run |
| A signed 3D position | Rejected: the classifiers would know the compartment |
| A hard split in place of the soft one | Not run |
| A per-study midline landmark | The first run: it learned a constant |
| The fitted centre with nothing learned | Not run |
| PF OA on its own channel; the front tokens excluded from Medial and Lateral OA | Not run |
| The side head alone for coordinates-only studies, without the coordinate prior | Not run |
| A centre correction per plane | Not run |

## Open

- The same shared channels on the Refined ResNet-50, the submission model.

## Hard octants on one frame (2026-10-06)

Run `outputs/training/20261006-073158-side-routed-hard-octants`, fold 1, seed 1110484336: run 5 plus
PF OA from front-and-up, the knee-level term, one frame, and hard cuts (no boundary widths; the frame
learns only through the position prior). The separate OA frame is dropped: its two frames turned
+6.3° and −5.9°, which bends the midline per disease instead of following the anatomy.

- ×4 macro 0.924432 against run 5's 0.924476; ×1 0.915933 against 0.915823.
- Per label against run 5: Fracture +0.0055, Synovitis +0.0021, Lateral OA −0.0023, PF OA −0.0013,
  Medial Meniscus −0.0012, the rest within ±0.0011.
- Learned angles: nod +2.50°, roll +3.52°, turn +3.06° (run 5 with soft cuts: −5.11°, −2.39°, −0.98°).
  The frame turns a few degrees with or without soft boundaries, and the signs are not stable.
- Knee-level strength: meniscus −0.119, OA +0.0002.
- Drawings on real slices: `outputs/analysis/octant-sketches-20261006/hard-octants-*.png`
  (two-frame model: `tilted-octants-*.png`).

S115, the two-frame adopted model (`separate-oa-frame-with-knee-level-term`, ×4, τ 0.9): 0.953 public,
equal to S113 (run 4) and against 0.952 for S114 (run 5).

### Turning the trained frame by hand

`outputs/analysis/side-routing-rotation-sweep-20261006/sweep.csv`: the hard-octant model at ×1, each
angle moved −30° to +30° around its learned value, inference only. `cuts` turns only the routing;
`all` also turns the position prior and the own heads' position code.

- Within ±10° on any axis the macro stays in 0.91551 to 0.91597 around 0.91589, and the routed sum
  within 0.005 of 4.5330.
- At ±30° the largest losses are nod −30° (PF OA 0.8897 to 0.8727) and roll +30° (Lateral Meniscus
  −0.006, Lateral OA −0.007).
- Roll towards −30° raises PF OA steadily (0.8897, 0.8912, 0.8927, 0.8939 at 0, −10, −20, −30) and
  lowers Medial OA by 0.002; routed sum +0.0037, macro +0.0003.
- No better angle away from the learned one beyond that. The heads were trained at the learned
  angle, so the sweep favours it.

## Build: soft gradient and three own-label routes (2026-10-06)

On top of the hard-octant run, still 7×7 tokens and one frame, no shifts:

- **Soft gradient through the hard cuts.** The forward pass keeps the hard octants; the gradient of
  each cut is that of `sigmoid(coordinate / token pitch)`, with the pitch measured per slice from
  the token positions (12 to 17 mm), so no width is set or learned. The frame's angles now receive
  gradient from the routing of every routed label (first step: 0.0009, 0.0021, 0.0007 against
  exactly zero before).
- **Own-label routes** (`OWN_ROUTES`): MCL from the four medial octants, Effusion from the two
  front-up octants, Baker's from the two medial-back octants. Their heads, tokens and priors are
  unchanged. Chosen from the hard-octant model's learned position prior, which already points there
  (Baker's medial-back +0.37/+0.28, Effusion front-up +0.28/+0.22, MCL medial +0.04 to +0.18);
  ACL, Contusion, Synovitis and Fracture stay unrouted.

All nine guardrails pass.

Run `outputs/training/20261006-081303-side-routed-soft-gradient-own-routes`, fold 1, seed 1110484336,
same speed as before (about 4 steps per second):

- ×4 macro 0.924845 against 0.924432 for the hard-octant run (+0.0004), 0.924476 for run 5 and
  0.924049 for the old recipe; ×1 0.917361 against 0.915933 (+0.0014). Highest of the line on both.
- Per label against the hard-octant run: Contusion +0.0021, Fracture +0.0018, Medial OA +0.0015,
  Lateral Meniscus +0.0013, Lateral OA +0.0010, Medial Meniscus +0.0005, MCL +0.0005, Baker's
  +0.0003, Effusion −0.0002, PF OA −0.0004, ACL −0.0014, Synovitis −0.0020. The three newly routed
  labels did not move; the routed five gained 0.0039 summed.
- Learned angles: nod +0.85°, roll −0.63°, turn +2.11°. With the routing pulling on the frame it
  turns less than without (2.5°, 3.5°, 3.1°): the shared angles are one correction for every study,
  so per-knee differences cancel. Next build: a frame inferred per study.
- Knee-level strength: meniscus −0.118, OA +0.0006.

## Build: a frame inferred per study (2026-10-06)

On top of the soft-gradient build (hard octants, eight routed labels, 7×7 tokens):

- **Frame head** (`study_frames`): reads the study's detached tokens plus the sinusoid code of their
  position in the fitted, header-aligned coordinates, attention-pools them (256 → 128) and outputs
  six numbers per study: a centre shift along medial, front and up, and nod, roll and turn.
- The shift is in token pitches of that study, the angles keep the ±90° range. The output layer
  starts at zero, so the first step is exactly the previous build (same loss, 0.874439). The shared
  angles and a shared shift (`frame_shift`) act as the head's bias.
- Every use of position follows the inferred frame: routing, the attention prior and the own heads'
  position code.
- Trained only by the diagnosis loss through the soft gradient and the prior; no auxiliary loss, no
  gradient into the backbone.
- Logged per epoch, train and validation: `study_frame_{nod,roll,turn}_deg` and `_abs_deg`,
  `study_frame_shift_{medial,front,up}_mm` and `_abs_mm`.
- `frame` and `routing_weights` now take the encoded batch.

All nine guardrails pass. The mirror guardrail holds at the zero start; once trained, the mirrored
pass infers its own frame from the mirrored image.

S116, the soft-gradient run (×4, τ 0.9): 0.952 public, with S112 and S114; S113 and S115 are 0.953.

Run `outputs/training/20261006-084210-side-routed-study-frames`, fold 1, seed 1110484336:

- ×4 macro 0.924320 against 0.924845 for the soft-gradient run (−0.0005); ×1 0.916573 against
  0.917361 (−0.0008), behind by 0.0005 to 0.0010 from epoch 3 on.
- Per label against the soft-gradient run: Baker's +0.0040, PF OA +0.0017, Medial Meniscus +0.0004,
  Synovitis 0.0000, Lateral OA −0.0002, Medial OA −0.0003, ACL −0.0004, Lateral Meniscus −0.0007,
  Effusion −0.0008, Contusion −0.0015, Fracture −0.0019, MCL −0.0066.
- Inferred frames over the validation knees at the last epoch, mean (mean absolute): nod +8.1°
  (8.4°), roll −2.2° (2.7°), turn −2.2° (3.3°); centre shift medial −1.0 (2.2) mm, front +1.3
  (1.4) mm, up −1.4 (1.9) mm. The shared part is nod +3.3°, roll −1.0°, turn 0.0°.
- The nod is common to all knees (same sign everywhere): the height cut drops towards the front, so
  the front-up region reaches the lower kneecap. Turn and the medial and up shifts differ in sign
  between knees. The values swung in the first epochs (roll −19° at epoch 2) and settled after 8.
- Four studies drawn before and after: `outputs/analysis/octant-sketches-20261006/study-frames-*.png`
  (rows 5, 9, 21, 45: nod +5.1° to +12.8°, turn −1.5° to −5.5°, shifts up to 4.6 mm).
- S117, this run (×4, τ 0.9): 0.952 public, with S112, S114 and S116; S113 and S115 are 0.953.

## Build: finer tokens, landmark frame, slower frame learning (2026-10-06)

On top of the per-study frame build:

- **Finer tokens for the shared heads.** The meniscus and OA heads and the frame head read the
  stride-16 stage (`layer3`, 14×14, 196 tokens per slice, 6 to 8.6 mm apart) through their own
  projection and LayerNorm; the seven own heads, the side head and the report heads keep the 7×7
  stage. The stride-16 token positions come from the same verified ResNet mapping, cut at stage 3.
  The pooling runs once per grid and each target takes its own grid's result.
- **Frame from landmarks.** The frame head produces three attention maps over a study's fine tokens
  (centre, front, up), starting uniform. Each landmark is the attention-weighted mean position; its
  displacement from the plain mean of the tokens is what moves the frame: the centre landmark's
  displacement is the centre shift, and the front and up axes are the unit axes scaled by the
  study's token spread plus the front and up landmarks' displacements relative to the centre's,
  made orthonormal. The shared angles still multiply on top. Zero start: the first frame is the
  geometric centre and header alignment.
- **Slower frame.** Parameters named `frame_*` train at the backbone learning rate (5e-5 instead of
  6e-4).
- Logged as before (`study_frame_*`), with the angles read back from the rotation matrix.

All nine guardrails pass; a training step, validation step and the per-view prediction path run;
peak memory on four studies 1.8 GB.

Run `outputs/training/20261006-092326-side-routed-fine-landmarks`, fold 1, seed 1110484336, about two
minutes per epoch:

- ×4 macro 0.922842 against 0.924845 for the soft-gradient run (−0.0020) and 0.924049 for the old
  recipe; ×1 0.915148 against 0.917361 (−0.0022). Lowest of the line.
- The loss is in the shared heads on the stride-16 tokens: Lateral OA −0.0155, PF OA −0.0126, Medial
  Meniscus −0.0042, Medial OA +0.0010, Lateral Meniscus +0.0012 (routed five −0.0301 summed). The
  seven own labels gained 0.0061 summed (Synovitis +0.0037, Fracture +0.0035, Effusion +0.0013,
  Baker's +0.0011, MCL −0.0002, ACL −0.0010, Contusion −0.0023).
- The frame is not the cause. Resetting the landmark maps to flat at inference (the starting frame)
  gives ×1 macro 0.91458 against 0.91509 with the learned frame, routed five 4.5050 against 4.5071.
- The landmark maps stayed flat at the backbone learning rate: each landmark effectively uses
  99.7% to 99.9% of a study's tokens, score range 0.3 to 0.45 logits, weight norm 0.03 to 0.05.
  Over 150 validation knees the frame is one common rotation, not per knee: nod 0.4° (SD 0.3),
  roll 2.6° (SD 0.4), turn 4.1° (SD 0.3), shifts under 0.5 mm (SD 0.1).
- Lessons: stride-16 features alone are too shallow for the OA head, which the meniscus head
  tolerates better; a zero-started head at 5e-5 does not move in ten epochs.

## Build: deep features on the fine tokens, frame at head rate, routed v4 report targets (2026-10-06)

On top of the fine-tokens and landmark build:

- **Deep features on the fine tokens.** The stride-16 tokens read by the shared heads and the
  frame head are their own projection plus the stride-32 projection, linearly interpolated to the
  fine token anchors (even anchors coincide with a coarse one, odd ones take the mean of two).
- **Frame parameters back at the head learning rate** (6e-4). At 5e-5 the landmark maps stayed flat.
- **Report attributes: target set 4.0** (`configs/labels/jevk5-report-attribute-targets-v4.0.json`,
  37 questions, brought from the worktree `knee-abnormality-jevk5-attributes-v4` with its question
  set and recipe; spec of the next extraction in that worktree's `attributes_v5.md`), **routed as
  "4.1" at training time** (`training/report_attributes.py`):
  - Ten compartment questions share one head per family and are routed like their label:
    meniscus status and tear type by side; cartilage depth and extent by medial rest, lateral rest
    and front-and-up. They read the position-free fine tokens.
  - The three location questions become nine yes/no region targets on shared heads instead of
    whole-knee "where" heads: meniscus tear in the front or back octants of each side, marrow edema
    in medial rest, lateral rest or front-and-up, patellar facet in the medial or lateral half of
    front-and-up. A region's target is the summed probability of the options placing the finding
    there; its weight is the question's weight times one minus the mass on options that state no
    place.
  - The other 23 questions keep their own heads on the 7×7 tokens.
  - Five routes added: medial-front, lateral-front, lateral-back, front-and-up medial and lateral.
    All of these pull on the frame through the soft gradient.
- Region targets over the 4,407 studies (target > 0.5): meniscus tear medial front 196, medial back
  1,343, lateral front 277, lateral back 398; marrow edema medial 1,207, lateral 790, front-and-up
  118; patellar facet medial 591, lateral 517.
- Logged: `report_attribute_bce_<family>_<route>` per region, beside the per-question KL.

All nine guardrails pass; a training step, validation step and the per-view prediction path run;
peak memory on four studies 2.0 GB.

Run `outputs/training/20261006-095814-side-routed-fine-fused-routed-reports`, fold 1, seed 1110484336:

- ×4 macro 0.924423 against 0.924845 for the soft-gradient run (−0.0004); ×1 0.916674 against
  0.917361 (−0.0007). Level within run-to-run noise; not above.
- Per label against the soft-gradient run: PF OA +0.0018, Synovitis +0.0019, Baker's +0.0018,
  Effusion +0.0009, Medial Meniscus +0.0003, ACL +0.0001, Medial OA −0.0004, Fracture −0.0008,
  Lateral OA −0.0016, Contusion −0.0017, Lateral Meniscus −0.0033, MCL −0.0040. Routed five −0.0032
  summed, against −0.0301 for the fine tokens without deep features: the deep features removed
  that loss, and the fine grid as a whole bought nothing on the score.
- **The frames are per knee now.** Over 150 validation knees, mean (SD, range): nod 10.6° (2.8, 7 to
  28), roll −1.1° (5.2, −19 to 12), turn −2.2° (2.2, −12 to 3); centre shift medial −1.1 mm (2.6,
  −8.6 to 5.6), front +1.2 mm (2.1), up +1.0 mm (2.5, −7.3 to 7.2). The centre and front landmark
  maps use 79% and 82% of a study's tokens (score range 2.5 logits); the up map 98%. Shared part:
  nod 4.8°.
- **Report targets (training set, last epoch), BCE explained against the base rate:** meniscus tear
  medial back 34%, lateral front 28%, lateral back 27%; marrow edema lateral 30%, medial 29%,
  front-and-up 3%; patellar facet lateral 19%, medial 15%; meniscus tear medial front low. The 33
  question heads all improved (KL −13% to −45% from the first epoch), the shared routed ones in the
  same range as the others.
- **Moving the trained centre by hand** (`outputs/analysis/side-routing-centre-sweep-20261006/`,
  ×1, −20 to +15 mm): the learned position is at the top. PF OA 0.8915 at the learned centre,
  0.8925 at 5 mm lower, 0.8883 at 20 mm lower, 0.8737 at 15 mm higher; macro 0.91671 at the learned
  centre and lower everywhere else, on both the up and the front axis. So the frame is not missing a
  better place for PF OA; the nod of about 10° already lowers the height cut at the front.
- Drawings: `outputs/analysis/octant-sketches-20261006/side-routed-fine-fused-routed-reports-*.png`.
- Lesson: the frame can now be placed per knee and sits at a local optimum, and the score does not
  follow. Inside its region each label's attention already selects the tokens that matter, so a
  more exact region changes little. The fine grid also gives back the bag reduction the octants
  bought (2× the original tokens for half-knee labels).

## Build: run 8 plus landmarks (2026-10-06)

The worktree code is restored from run 8's source snapshot (7×7 tokens for every head, v3.1 report
targets on unrouted heads, the regressed per-study frame) and gets the landmark frame on top:

- The frame head keeps its attention-pooled regression of six numbers per study (shift in token
  pitches, three angles) and gains three zero-started landmark maps over the same 7×7 tokens.
- The two compose: the centre shift is the centre landmark's displacement plus the regressed shift;
  the rotation is the regressed rotation applied to the landmark axes.
- Zero start: the first step's loss equals run 8's (0.874440).
- The fine grid, its projection, the routed report heads and the v4 targets are not in this build;
  they are in run 10's source snapshot and `scratchpad/run10-build`. The v4 manifest, question set
  and recipe stay in `configs/labels/`, unused.

All nine guardrails pass.

Run `outputs/training/20261006-103317-side-routed-study-frames-landmarks` (run 11), fold 1, seed
1110484336:

- ×4 macro 0.924119 against 0.924845 for run 7 (−0.0007) and 0.924320 for run 8 (−0.0002); ×1
  0.916769 against 0.917361 (−0.0006).
- Per label against run 7: Synovitis +0.0019, PF OA +0.0016, Contusion +0.0011, ACL +0.0009, Baker's
  +0.0009, Effusion +0.0002, Medial Meniscus 0.0000, Medial OA −0.0007, Lateral OA −0.0008, Lateral
  Meniscus −0.0014, Fracture −0.0035, MCL −0.0088.
- Frames over all validation knees, mean (SD, range): nod 10.9° (5.2, −5 to 25), roll 0.2° (5.2,
  −13 to 18), turn 3.3° (6.4, −17 to 19); shift medial −0.2 mm (2.6, −10 to 6), front +0.7 mm
  (2.2), up +0.8 mm (3.2, −6 to 10). The widest per-knee spread of any run.
- Frame reset at inference (×1): learned frame 0.91673, starting frame 0.91614 (−0.0006), starting
  centre with learned angles 0.91633, learned centre with starting angles 0.91642. With the
  learned frame PF OA is +0.0025, Contusion +0.0028 and MCL +0.0027 above the starting frame, and
  Fracture −0.0025. So inside this model the frame helps; MCL is low with either frame, which makes
  its loss a training effect.

Across the four runs with a per-knee frame (8, 9, 10, 11) against run 7: macro −0.0002 to −0.0007
(run 9 aside), PF OA +0.0016 to +0.0018 in the three where the frame learned, MCL −0.0040 to
−0.0088 in the same three, Fracture −0.0008 to −0.0035.

## Where each label's attention falls (2026-10-06)

`outputs/analysis/side-routing-attention-by-region-20261006/attention.json`: run 11's model, the 874
validation knees with a fitted centre, attention of each label on its positive knees, binned by
the fifteen regions of the proposed division (four quadrants plus a core of 14 mm radius; joint
band ±10 mm around 14 mm below the geometric centre). Geometric coordinates, 7×7 tokens. Routed
labels are confined to their octants in this model, so for them only the height and core split is
free.

| Label | Femur band | Joint band | Tibia band | Core (all heights) | Notes |
|---|---:|---:|---:|---:|---|
| ACL (unrouted) | 57% | 37% | 6% | 42% | core 6 to 7 times its token share; lateral quadrants 43%, medial 12% |
| MCL | 52% | 36% | 12% | 15% | medial back at the joint line 4.2 times its share |
| Medial Meniscus | 40% | 49% | 11% | 14% | medial back at the joint line 6.8 times its share |
| Lateral Meniscus | 45% | 45% | 10% | 10% | lateral back at the joint line 6.1 times |
| Medial OA | 32% | 49% | 19% | 18% | core at the joint line 4.5 times its share |
| Lateral OA | 38% | 47% | 14% | 15% | |
| PF OA | 96% | 4% | 0% | 7% | lateral front 54%, medial front 35% |
| Effusion | 98% | 2% | 0% | 8% | |
| Baker's | 59% | 32% | 9% | 5% | 93% in medial back |
| Synovitis (unrouted) | 69% | 22% | 10% | 12% | front femur 51% |
| Contusion (unrouted) | 46% | 30% | 24% | 18% | lateral 54%, medial 28% |
| Fracture (unrouted) | 42% | 33% | 25% | 22% | core 3.9 times its share at the joint line |

Consequences for the region table: ACL reads the lateral quadrants as much as the core; the side
labels use their half of the core, so the core must not be taken away from them; a joint band of
this size holds only half of the meniscus attention, the femur band the other 40% to 45%; the
tibia band is the one most labels barely use.

### The same measurement on the highest-attention tokens only

All-token attention includes a long diffuse tail. Taking only each knee's three highest-attention
tokens per label (they hold 4% to 18% of the attention; ten tokens hold 10% to 46%) changes the
picture. Token share of the bands: femur 51%, joint 22%, tibia 27%; core 9%.

| Label | Femur | Joint | Tibia | Core | Note |
|---|---:|---:|---:|---:|---|
| ACL | 77% | 23% | 1% | 77% | all-token view had only 42% in the core; the lateral quadrants were tail |
| MCL | 56% | 37% | 7% | 6% | |
| Medial Meniscus | 34% | 65% | 2% | 1% | medial back of the joint band alone 50% |
| Lateral Meniscus | 39% | 59% | 2% | 2% | lateral back of the joint band 39% |
| Medial OA | 17% | 63% | 20% | 11% | |
| Lateral OA | 31% | 59% | 9% | 10% | |
| PF OA | 97% | 3% | 0% | 1% | |
| Effusion | 100% | 0% | 0% | 13% | |
| Synovitis | 99% | 1% | 0% | 6% | all-token view had 69% in the femur band |
| Baker's | 62% | 31% | 7% | 1% | |
| Contusion | 38% | 26% | 36% | 9% | spread |
| Fracture | 36% | 23% | 41% | 22% | spread; the core at 2.5 times its share |

So the decisive tokens support the three heights and the core: the joint band (22% of tokens) holds
59% to 65% of the top tokens of the menisci and the femorotibial OA labels; the menisci's top tokens
avoid the core and the tibia; ACL's sit in the core; Synovitis's sit in the femur band. The femur
share of the menisci (34% to 39%, below the band's token share) is consistent with a joint band of
±10 mm being thinner than the tokens.

## Build: eighteen regions, everything about the frame learned (2026-10-06)

On top of run 11 (7×7 tokens, per-study frame from landmarks plus regression):

- **Regions.** Around the per-study frame: four quadrants (medial/lateral × front/back) and a core
  cylinder split at the midline, at three heights (femur, joint, tibia): 18 regions.
- **Nine numbers per study.** The frame head regresses three more: the upper cut, the lower cut and
  the core radius, in token pitches of the study, on top of shared biases (`frame_size`). They
  start at 0, −2 and 1 pitches (`FRAME_SIZE_START_PITCHES`): the upper cut at the centre, the
  lower cut about 27 mm below it, the core about 14 mm in radius. With the geometric centre about
  15 mm above the joint line that puts the joint band around the joint line.
- **Assignment by overlap.** A token's weight for a cut is 0.5 plus its signed distance over twice
  its half-extent along the cut's normal, clamped to 0..1. The half-extent comes from the token's
  box: its two in-plane steps and the step to the next slice, projected on the frame axes. So a
  token inside counts 1, a straddling one its share, and a thin slice stack is cut sharply along
  its normal. This replaces the hard cuts with a soft gradient.
- **Region table** (`_regions`), from the highest-attention tokens of run 11:

  | Target | Reads |
  |---|---|
  | ACL | core, femur and joint bands |
  | MCL | medial quadrants, femur and joint bands |
  | Medial / Lateral Meniscus | its quadrants, joint band |
  | Medial / Lateral OA | its side with its half of the core, all heights, without the kneecap part |
  | PF OA, Effusion | front quadrants and the core, femur band |
  | Synovitis | femur band |
  | Baker's | medial-back quadrant, femur and joint bands |
  | Contusion, Fracture | unrouted |

- **Report attributes: target set 4.0, routed** on the position-free 7×7 tokens: the ten
  compartment questions shared and routed like their label, ACL status and PCL routed to the core,
  and the nine region targets (meniscus tear in the front or back of each side's joint band,
  marrow edema by compartment, patellar facet by side of the kneecap part).
- **Own heads read position in the fixed geometric frame** (token position code and the attention
  prior of own targets); routing and the shared targets' prior follow the inferred frame.
- **Logged:** `study_frame_{band_top,band_bottom,core_radius}_mm`, and on the validation set
  `val_report_attribute_fit` and `val_report_attribute_bce_<region>`.
- Guardrail change: the meniscus pair may not overlap and every weight lies in 0..1; the shared
  channels no longer partition every token (the menisci leave the core and the other bands, and the
  core of the femur band is read by PF OA and both OA sides).

First step on four studies: token share per route 15% for a meniscus (50% before), 7% for the
core, 30% for the kneecap part, 47% for the femur band; all three sizes receive gradient. All nine
guardrails pass.

Run `outputs/training/20261006-112944-side-routed-eighteen-regions` (run 12), fold 1, seed 1110484336:

- ×4 macro 0.923687 against 0.924845 for run 7 (−0.0012, three sibling SDs); ×1 0.915856 against
  0.917361 (−0.0015).
- Per label against run 7 (sibling SD in brackets): Baker's +0.0020 (0.0015), Lateral OA +0.0017
  (0.0034), Lateral Meniscus +0.0016 (0.0024), PF OA +0.0007, Effusion −0.0001, Medial OA −0.0002,
  ACL −0.0008 (0.0014), Medial Meniscus −0.0008, Synovitis −0.0019 (0.0022), Contusion −0.0045
  (0.0013), MCL −0.0046 (0.0029), Fracture −0.0071 (0.0044).
- The routed five are +0.0030 summed while reading 15% to 38% of the tokens; ACL holds on the core
  alone (7% of the tokens). The loss is in Contusion and Fracture, which are unrouted and in this
  build do not depend on the frame at all, and in MCL.
- Learned sizes over the validation knees: upper cut +4.9 mm, lower cut −35.2 mm (the joint band
  widened from 27 to 40 mm), core radius 15.0 mm. Nod +20.1° in every knee, roll 8.6° and turn
  3.8° in absolute value, centre shifts 2.4 to 3.2 mm.
- **Report region targets on the validation set** (BCE explained against the base rate, last
  epoch): meniscus tear medial back 33%, lateral back 22%, lateral front 17%, medial front −9%;
  marrow edema medial 25%, lateral 25%, kneecap 2%; patellar facet lateral 20%, medial 15%. The
  validation fit improves through training (0.763 to 0.719). The regions hold on unseen knees.
- Decision: Contusion's supervision changed with the v4 targets (its status and location questions
  became marrow edema of any cause, and the location became region targets), and it lost 0.0017
  in run 10 as well. Next run isolates the reports: this build with the v3.1 targets on unrouted
  heads.

### Run 13: the eighteen regions with the v3.1 reports, unrouted

Run `outputs/training/20261006-115457-side-routed-regions-v31-reports`: run 12's build with the
report heads and targets of run 11 (v3.1, 35 questions, unrouted). Launched by the assistant in the
unattended loop the user asked for on 2026-10-06 13:30.

- ×4 macro 0.924122 against 0.924845 for run 7 (−0.0007) and 0.923687 for run 12 (+0.0004); ×1
  0.916925 against 0.917361 (−0.0004).
- Per label against run 7: Lateral OA +0.0031, Lateral Meniscus +0.0021, Baker's +0.0016, PF OA
  +0.0007, Effusion +0.0006, Medial OA −0.0003, Medial Meniscus −0.0004, Synovitis −0.0005, ACL
  −0.0009, MCL −0.0033, Contusion −0.0042, Fracture −0.0072. Routed five +0.0052 summed.
- Against run 12 every label is within ±0.0015: Contusion +0.0003, Fracture −0.0001. So the v4
  routed reports did not cause the Contusion and Fracture losses, and on this fold the two report
  setups are level.
- Learned sizes: upper cut +3.4 mm, lower cut −34.2 mm, core radius 15.6 mm, nod +15.5°.
- Contusion and Fracture lose the same amount in both runs (−0.0045/−0.0042 and −0.0071/−0.0072),
  so it is systematic to the eighteen-region build and not noise. They are unrouted and read the
  fixed frame, as in run 7, so the cause is indirect. Leading candidate: ACL now reads only the
  core, while unrouted it put 43% of its attention on the lateral quadrants, where the bruises and
  fractures that accompany its tears are; that trained the backbone on them. Next run: the same
  build with ACL unrouted.

### Run 14: ACL unrouted

Run `outputs/training/20261006-121814-side-routed-regions-acl-unrouted`: run 13 with ACL reading the
whole knee again.

- ×4 macro 0.923936 against 0.924845 for run 7 (−0.0009) and 0.924122 for run 13 (−0.0002); ×1
  0.915416.
- Against run 13: Fracture +0.0027, Contusion +0.0018, ACL +0.0016, PF OA +0.0011, Lateral
  Meniscus −0.0015, Lateral OA −0.0018, Effusion −0.0019, MCL −0.0048. Against run 7: Contusion
  −0.0024, Fracture −0.0044, MCL −0.0080.
- Contusion and Fracture move in the predicted direction and recover about a third of their loss
  (1.4 and 0.6 sibling SDs); the macro does not move. Weak support for the ACL explanation; ACL
  stays unrouted.
- Across the six runs with a per-knee frame head that learned (8, 10, 11, 12, 13, 14) MCL is below
  run 7 every time (−0.0066, −0.0040, −0.0088, −0.0046, −0.0033, −0.0080; sibling SD 0.0029) and
  Fracture every time (−0.0019, −0.0008, −0.0035, −0.0071, −0.0072, −0.0044), while the region
  table gives the routed five +0.0030 to +0.0052. Next run separates the two: the eighteen regions
  on one shared learned frame (shared centre shift, rotation and sizes, no per-knee head).

### Run 15: the eighteen regions on one shared frame

Run `outputs/training/20261006-124148-side-routed-regions-shared-frame`: run 14 without the per-knee
frame head; one learned centre shift, rotation and set of sizes for all knees.

- ×4 macro 0.924141 against 0.924845 for run 7 (−0.0007) and 0.923936 for run 14 (+0.0002); ×1
  0.915650.
- Per label against run 7: Baker's +0.0016, Lateral Meniscus +0.0009, ACL +0.0008, Medial OA
  +0.0008, Medial Meniscus +0.0004, PF OA +0.0003, Lateral OA −0.0010, Synovitis −0.0010, Effusion
  −0.0015, Contusion −0.0022, Fracture −0.0029, MCL −0.0047.
- The shared frame stayed where it started: upper cut −0.1 mm, lower cut −29.0 mm, core radius
  13.7 mm, nod 0.4°, roll −2.0°, turn 1.4°, shifts under 0.7 mm.
- MCL and Fracture are below run 7 here too, without any per-knee head. So the per-knee head is
  not what lowers them.

**Yardstick check.** The ×4 macros of every run of the line since the fifth are 0.924476, 0.924432,
0.924845, 0.924320, 0.924423, 0.924119, 0.923687, 0.924122, 0.923936, 0.924141 (run 9 left out):
mean 0.92425, SD 0.0003, the same as the sibling SD of the old recipe (0.0004). Run 7 is the top
of that spread and run 12 the bottom. Measured against the spread and not against the best draw,
none of the changes since the fifth run has moved the fold-1 macro: hard cuts, soft gradient,
per-knee frame, landmarks, fine tokens with deep features, eighteen regions, learned sizes, overlap,
routed v4 reports. MCL and Fracture being "below run 7" in every later run says as much about
run 7's draw on those two labels as about the later runs. What did move, beyond single-label
noise and in every run that had it: PF OA with a per-knee frame (+0.0016 to +0.0018, three runs),
and the report region targets fitting on the validation set (run 12).

Next run tries the one idea with a mechanism the heads do not yet have: position for the shared
heads, in compartment-relative (mirrored) coordinates.

### Run 16: position for the shared heads, in compartment-relative coordinates

Run `outputs/training/20261006-130551-side-routed-regions-mirrored-position`: run 14 with the shared
meniscus and OA heads reading each token plus the fixed sinusoid of its frame position, the medial
axis as is for a medial target, mirrored for a lateral one and in absolute value for PF OA and the
knee-level term. The guardrail "shared tokens are position-free" becomes "shared tokens mirror
between sides"; the side-flip check still passes (mismatch 3e-8).

- ×4 macro 0.924172 against 0.923936 for run 14 (+0.0002) and 0.924845 for run 7; ×1 0.915906.
- Against run 14: MCL +0.0033, ACL +0.0031, Lateral OA +0.0030, Fracture +0.0021, Effusion +0.0013,
  Medial Meniscus +0.0003, Lateral Meniscus −0.0004, Medial OA −0.0005, Contusion −0.0010, Baker's
  −0.0013, PF OA −0.0024, Synovitis −0.0045. The five shared-head labels sum to 0.0000.
- Position does not help the shared heads inside their regions; inside the noise of the fold.

### Run 17: the agreed design on the Refined ResNet-50

Since ten ResNet-18 variants sit inside one noise band on fold 1, the loop moves to a change large
enough to read: the same design on the backbone the best public result uses. Build: run 12's
(eighteen regions, per-knee frame with learned sizes, overlap assignment, routed v4 reports, own
position in the fixed frame) with ACL unrouted as in run 14. Config `configs/refined-resnet50.yaml`
with `--validation.folds [1]`, seed 1533697213. All nine guardrails pass with this config.

Loose reference, not a control: `20261003-122711-refined-rn50-views-label-bias-folds-1-4`, fold 1,
same seed, the old whole-knee recipe of 2026-10-03 (ImageNet normalization, half-knee loss 0.075):
×4 0.929567, ×1 0.921679. On the ResNet-18 the old recipe's fold 1 is 0.924049.

Run 17 was stopped by the user three minutes into its first epoch (2026-10-06 15:33): moving to the
ResNet-50 was the assistant's own decision and outside what the loop was for. Nothing is kept from
it; its partial folder is `outputs/training/20261006-132943-rn50-side-routed-regions`.

### Run 18 build: hard cuts with the soft gradient, knee-level term on every routed target

Decided with the user on 2026-10-06 (the per-knee frame is settled for every run; the reports stay
routed). On run 12's build, ACL in the core as in the agreed table:

- **Assignment.** Hard steps at every cut (midline, front cut, the two height cuts, the core
  radius), with the gradient of a sigmoid of the signed distance over the token box's extent along
  the cut's normal. This is run 7's mechanism in place of the overlap shares; the three sizes learn
  through it.
- **Knee-level term on every routed target.** One learned number per channel, zero at the start:
  the target's score is its region's evidence plus that share of the same head's evidence pooled
  over the whole knee. New for ACL, MCL, Effusion, Synovitis and Baker's; Contusion and Fracture are
  unrouted and have none. It lets ACL keep the core as its region and still read the bruises on the
  lateral side, so its loss reaches those tokens again.

All nine guardrails pass. First step on four studies: routing weights are 0 or 1 on side-certain
knees; the knee-level numbers of the seven routed channels and the three sizes receive gradient.

Run `outputs/training/20261006-135237-side-routed-regions-soft-gradient-knee-level` (run 18):

- ×4 macro 0.923121 against 0.924845 for run 7 (−0.0017) and 0.923687 for run 12 (−0.0006); ×1
  0.915100. The lowest of the eighteen-region runs.
- Against run 12 (the same build with overlap shares and the knee-level term on the shared heads
  only): MCL +0.0059, Fracture +0.0033, Contusion +0.0019, Lateral Meniscus −0.0001, ACL −0.0008,
  Effusion −0.0008, Medial OA −0.0010, Medial Meniscus −0.0024, PF OA −0.0025, Lateral OA −0.0029,
  Synovitis −0.0032, Baker's −0.0043. The five shared-head labels lose 0.0089 summed; Medial
  Meniscus is 0.0033 below run 7, against a sibling SD of 0.0007.
- **The sizes hardly move under the soft gradient:** upper cut −0.2 mm, lower cut −28.8 mm, core
  radius 16.3 mm (start 0, −27.4, 13.7). With overlap shares they went to +4.9, −35.2 and 15.0.
  Nod 4.6° against 20.1°. So the joint band stays 28 mm thick.
- **Every knee-level number learned negative:** ACL −0.027, MCL −0.008, Effusion −0.039, Synovitis
  −0.064, Baker's −0.033, meniscus −0.096, OA −0.021. The term acts as a contrast (region minus
  knee), not as a wider view; it did not make ACL read the lateral bruises. Contusion and Fracture
  still rose against run 12 (+0.0019, +0.0033), within their noise.
- Report fit on the validation set 0.7204 against 0.7194 in run 12: unchanged.

**Drawings** (`outputs/analysis/side-routing-loop-20261006/side-routed-regions-soft-gradient-knee-level/`):

- `attention-*.png`: each label looks where it should. Medial and Lateral Meniscus on the joint
  line of their side, ACL on the notch, MCL on the medial edge, PF OA on the kneecap and trochlea,
  Effusion and Synovitis on the fluid above the kneecap, Baker's on the cyst behind the medial
  side. A meniscus puts 20% to 43% of its attention on one slice, on two to four tokens.
- `regions-*.png`: the core is the problem. It is a cylinder of 16 mm mean radius (22 mm in study
  45) around the inferred up axis; on the axial slices it covers the notch and the inner half of
  both condyles and plateaus, which is where the meniscal roots and inner posterior horns are. With
  hard cuts the menisci, MCL and Baker's lose those tokens entirely; with overlap shares (runs 12
  to 16) the tokens on the core's edge still counted in part, and Medial Meniscus stayed within
  0.0008 of run 7, even with a 29 mm band (run 15).

Decision for run 19: the side labels keep their half of the core (Medial and Lateral Meniscus, MCL,
Baker's and the tear-location targets no longer exclude it). The core then defines ACL's region and
is added to the kneecap part for PF OA and Effusion; it is taken away from no one.

### Run 19: the side labels keep their half of the core

Run `outputs/training/20261006-141739-side-routed-regions-core-kept-by-sides`: run 18 with the core no
longer taken from the menisci, MCL, Baker's and the tear-location targets.

- ×4 macro 0.923473 against 0.923121 for run 18 (+0.0004) and 0.924845 for run 7 (−0.0014); ×1
  0.915175.
- Against run 18: MCL +0.0039, Fracture +0.0039, Medial Meniscus +0.0018, Contusion +0.0015, Medial
  OA +0.0009, Effusion +0.0007, Lateral OA +0.0006, ACL −0.0004, PF OA −0.0012, Baker's −0.0013,
  Synovitis −0.0022, Lateral Meniscus −0.0041.
- Against run 7, beyond their sibling SD: Synovitis −0.0073 (SD 0.0022), PF OA −0.0030 (0.0005),
  Baker's −0.0036 (0.0015), Lateral Meniscus −0.0025 (0.0024); MCL +0.0052 (0.0029).
- Sizes as in run 18: upper cut +0.4 mm, lower cut −27.9 mm, core radius 15.3 mm, nod 4.7°.

**The frame head, measured on this model** (200 validation knees):

- Angles, mean (SD, range): nod 4.5° (4.4, −9 to 20), roll −7.1° (5.4, −19 to 9), turn −4.7° (5.8,
  −21 to 6); 34% of knees have more than 10° of roll and 23% more than 10° of turn. Shift SD 2.9,
  3.3 and 4.9 mm. Sizes: lower cut −27.9 mm (SD 4.1, −36 to −19), upper cut and radius nearly
  constant.
- Both parts contribute: regression alone gives SD 3.5°, 4.1°, 5.2°; landmarks alone 3.4°, 3.0°,
  2.5°.
- The inferred turn does not follow the header rotation of the knee (correlation 0.06).
- **The frame is a consistent function of the knee, not noise:** inferred from a knee and from its
  mirror image (100 side-certain knees), nod, turn, the front and up shifts and the three sizes
  correlate 0.97 to 0.99. Roll (0.86, difference SD 3.0°) and the medial shift (0.75, 2.1 mm), the
  two that move the midline, are the least reproducible.
- Drawing `regions-021.png`: a knee that stands upright in the coronal image gets a roll of −13.8°;
  the midline and the core lean across the lateral condyle and the height cuts slope across the
  joint. So a consistent frame is not always an anatomical one.

**Reading of runs 18 and 19.** With hard cuts, a label loses whatever its region leaves out, and
nothing pulls a region wider (the sizes do not move under the soft gradient). The regions of the
table came from each label's three highest-attention tokens, which is too tight for that: Synovitis
keeps 69% of its all-token attention in the femur band, the menisci about half in the joint band,
Baker's 91% without the tibia band. Run 20 uses the table from the all-token attention instead:
menisci on the femur and joint bands of their side, MCL and Baker's at all heights, ACL on the core
and the lateral quadrants (femur and joint bands), Synovitis unrouted; OA, PF OA and Effusion
unchanged.

### Run 20: the wider region table

Run `outputs/training/20261006-144444-side-routed-regions-wider-table`: run 19 with the region table
from the all-token attention (menisci on the femur and joint bands of their side, MCL and Baker's
at all heights, ACL on the core and the lateral quadrants above the tibia band, Synovitis unrouted).

- ×4 macro 0.923483 against 0.923473 for run 19, 0.924119 for run 11 and 0.924845 for run 7; ×1
  0.915464.
- Against run 19: Synovitis +0.0056, PF OA +0.0029, ACL +0.0027, Medial Meniscus +0.0015, Effusion
  +0.0013, Lateral OA +0.0012, Baker's +0.0010, Lateral Meniscus +0.0009, Contusion −0.0013, Medial
  OA −0.0020, Fracture −0.0034, MCL −0.0102. The labels whose regions were widened all recovered;
  the macro did not move because MCL fell by 0.010 (it has ranged from 0.9096 to 0.9236 over the
  day's runs, 47 positives).
- Against run 11 (octants with the same per-knee frame head): Contusion −0.0036, Synovitis −0.0036,
  Baker's −0.0034, Medial OA −0.0016, PF OA −0.0016, MCL +0.0038, the rest within ±0.0014.
- Frame: nod −3.6°, |roll| 8.3°, upper cut −1.7 mm, lower cut −27.5 mm, core radius 12.5 mm.
- Family means of the ×4 macro: octants (runs 5, 6, 7, 8, 10, 11) 0.92444; eighteen regions with
  overlap (12 to 16) 0.92401; eighteen regions with hard cuts (18, 19, 20) 0.92336. With a run-to-run
  SD of 0.0003 to 0.0004 the last family is about three SDs below the first.
- Drawings (`outputs/analysis/side-routing-loop-20261006/side-routed-regions-wider-table/`): in study 5
  the head moves the centre 8 mm down, onto the joint line, which is anatomically right; and it
  turns the frame −12.9° and rolls it −9.6°, so on the axial slices the midline no longer runs
  along the notch and on the coronal slices the midline and the core lean across a knee that
  stands upright. The start frame (headers) looks better aligned than the inferred one in turn and
  roll.

### Run 21 build: per-knee centre, nod and sizes; roll and turn shared

Agreed with the user on 2026-10-06 after the drawings of runs 19 and 20. On run 20's build, the
frame head still infers a frame for every knee, but only the parts a knee really differs in: the
centre (three shifts), the nod (how bent the knee is) and the three sizes. Roll and turn are one
learned value each for all knees, as in run 7. The landmarks now give the nod only: the tilt of the
front and up landmarks in the front-up plane. Zero start unchanged; all nine guardrails pass.

Run `outputs/training/20261006-151245-side-routed-regions-nod-per-knee` (run 21):

- ×4 macro 0.923947 against 0.923483 for run 20 (+0.0005), 0.924119 for run 11 and 0.924845 for
  run 7; ×1 0.915177. The best of the hard-cut eighteen-region runs.
- Against run 20: Synovitis +0.0027, Baker's +0.0024, MCL +0.0023, Lateral Meniscus +0.0014, Lateral
  OA +0.0014, Medial OA +0.0012, Medial Meniscus +0.0001, Effusion −0.0003, Contusion −0.0004, PF OA
  −0.0015, ACL −0.0017, Fracture −0.0020. Against run 7 the routed five sum to −0.0006; Contusion
  −0.0029 and Fracture −0.0053 remain.
- Frame over the validation knees: shared roll −0.9° and turn −1.9°; nod −15.3° (every knee
  negative); centre shift up −3.6 mm; upper cut −0.2 mm, lower cut −30.8 mm, core radius 13.5 mm.
- Drawings (`outputs/analysis/side-routing-loop-20261006/side-routed-regions-nod-per-knee/`): the
  midline is upright on the coronal slices and runs along the notch on the axial ones. On the
  sagittal slices the height cuts now slope down towards the back of the knee, following the
  backward slope of the tibial plateau, so the joint band lies along the plateau. The same slope
  lifts the upper cut at the front, which takes the lower kneecap out of the PF OA and Effusion
  region; in the octant runs the nod went the other way (+10°) for exactly that label.
- So one height cut cannot both follow the plateau and reach under the kneecap. Next run: PF OA
  and Effusion read the front quadrants above the lower cut (femur and joint bands) plus the core
  of the femur band, so their region no longer depends on the upper cut's slope.

### Run 22: the kneecap labels read the front above the lower cut

Run `outputs/training/20261006-153716-side-routed-regions-kneecap-above-lower-cut`: run 21 with PF OA
and Effusion on the front quadrants above the lower cut plus the core of the femur band.

- ×4 macro 0.924358 against 0.923947 for run 21 (+0.0004) and 0.924845 for run 7 (−0.0005); ×1
  0.916091. The hard-cut region runs have climbed 0.923121, 0.923473, 0.923483, 0.923947, 0.924358.
- Against run 21: Fracture +0.0043, PF OA +0.0030, Lateral OA +0.0019, Medial Meniscus +0.0007,
  Effusion +0.0005, Medial OA −0.0009, Baker's −0.0009, Contusion −0.0011, Lateral Meniscus −0.0019.
  PF OA moved as predicted. Against run 7 the routed five are +0.0019 summed; Contusion −0.0040.
- Frame: nod −14.1°, centre shift up −3.5 mm; lower cut −16.5 mm (it moved from −30.8), upper cut
  −5.4 mm, core radius 17.8 mm.

**Where the heads look in knees with and without the finding** (this model, 400 validation knees,
attention-weighted position of each label's ten highest-attention tokens, geometric frame, mm;
`outputs/analysis/side-routing-loop-20261006/spots-side-routed-regions-kneecap-above-lower-cut.json`):

| Label | Positive knees: mean (SD) | Negative knees: mean (SD) |
|---|---|---|
| Medial Meniscus | 26.5, −5.3, −5.0 (4.9, 5.8, 6.8) | 25.0, −7.1, −5.9 (4.1, 5.0, 5.6) |
| Lateral Meniscus | −29.1, −3.9, −6.7 (5.8, 5.3, 6.8) | −26.3, −4.7, −5.9 (4.3, 5.2, 6.4) |
| ACL | −5.8, −4.3, −1.7 (4.5, 4.5, 5.1) | −5.4, −1.4, −0.3 (4.2, 3.9, 5.3) |
| PF OA | −2.5, 36.4, 13.1 (5.3, 9.8, 7.9) | −1.4, 31.6, 13.3 (4.4, 8.4, 8.1) |
| Medial OA | 21.6, −1.6, −18.8 (4.3, 5.2, 8.3) | 21.6, −3.9, −14.7 (5.4, 4.8, 8.6) |
| Lateral OA | −23.9, −3.6, −15.5 (6.5, 6.0, 8.2) | −20.4, −5.0, −12.8 (4.6, 5.3, 8.1) |

The heads look at the same place whether the finding is there or not, and the ten tokens hold 18%
to 36% of the attention in both groups. From the two meniscus spots: distance 52.8 mm (SD 6.5),
midpoint −0.6, −5.5, −5.8 (SD 3.2, 4.7, 5.9), turn of the line between them −2.0° (SD 5.4), roll
0.6° (SD 4.8). A spot's SD of 4 to 6 mm mixes real differences between knees with the token grid
(12 to 17 mm), so the turn and roll from two spots carry an uncertainty of a few degrees.

### Run 23 build: the frame hung on the two meniscus spots, eight sections per level

Agreed with the user on 2026-10-06 ("learn everything" again, tied to structures, and eight sections).
On run 22's build:

- **Frame from the heads' own attention.** For each knee, the shared meniscus head's attention over
  the medial half and over the lateral half of the geometric frame gives two spots (centroids
  weighted by the squared attention, no gradient). The centre is their midpoint plus one shared
  learned shift; the side axis is the line between them, which fixes turn and roll per knee. The nod
  and the three sizes stay regressed per knee as in run 21. The free regression of the centre and
  the centre landmark are gone.
- **Eight sections per level.** The two diagonals split each quadrant into a 45° section next to
  the side axis and one next to the front-back axis. Every diagnosis label keeps its union. The
  tear-location targets become three per side: anterior horn (anterior section), body (the two
  sections next to the side axis) and posterior horn (posterior section), in the joint band.
- Guardrails: the frame no longer starts at the geometric one (untrained heads give spots near
  each half's centroid: first step on four studies, centre within 7 mm and axes within 3°), so
  "learned geometry starts at zero" now checks the shared rotation and the prior only; the mirror
  check compares the geometric position, since a frame read from the image differs between a knee
  and its mirror image; the side-flip check mirrors the frame with the side. All nine pass.

## The shape read from the heads' attention (2026-10-06)

Asked by the user: extract where the heads look and find the division of the knee that matches it.
Scripts in `scripts/adhoc/side_routed_session/` (`extract_attention.py`, `analyse_attention.py`); outputs in
`outputs/analysis/side-routing-attention-shape-20261006/` (`attention.npz`, `heads-1..4.png`,
`emergent-{8,12,18}-{geometric,spots}.png`, `shapes.json`).

- **Source.** Run 11's model on the 874 validation knees with a fitted centre: the 35 report
  question heads (v3.1, each over the whole knee) and ACL, Synovitis, Contusion and Fracture are free;
  the other eight labels were routed in that model and are kept apart. Per knee and head, the ten
  most attended tokens with their attention, on knees where the finding is present.
- **Score.** The knee in 6 mm cells; a division is scored by how much of what separates the heads
  it keeps (mutual information between section and head, as a share of the value at single cells).
  Cuts, core radius and sector orientation are placed where they score best.

| Division | Sections | Free heads | Diagnosis labels | Best cuts (mm from the geometric centre) |
|---|---:|---:|---:|---|
| Octants | 8 | 43.0% | 46.4% | |
| Quadrants × 3 heights | 12 | 46.4% | 51.4% | 0, +18 |
| Quadrants × 4 heights | 16 | 49.1% | 54.5% | −24, 0, +12 |
| Quadrants + core × 3 heights | 15 | 50.9% | 55.2% | core 12 |
| 6 sectors × 3 heights | 18 | 56.4% | 60.1% | −24, +12; sectors centred on the axes |
| 8 sectors × 3 heights | 24 | 57.7% | 61.9% | −24, +12; sectors centred on the axes |
| 12 sectors × 3 heights | 36 | 61.2% | 65.1% | −24, +12 |
| 8 sectors + core × 3 heights | 27 | 61.4% | 65.0% | core 12 |
| 8 sectors + core + outer ring × 3 heights | 51 | 64.4% | 68.5% | core 12, ring from 30 |
| Emergent (neighbouring cells grouped by who looks there) | 8 / 12 / 18 / 24 | 55.2 / 59.4 / 64.4 / 66.3% | 57.7 / 62.6 / 66.9 / 68.4% | |

- Going round the axis matters most (4 to 8 sectors: +11 points), then the core (+4 to +5) and the
  outer ring (+3); a third height adds +3. Eight emergent sections keep 55% where the octants keep
  43%.
- The best sectors are centred on the axes (a medial sector, an anterior one, and so on), not
  bounded by them: the midline and the front cut run through the kneecap sector and the medial
  sector.
- Best height cuts for the sector divisions: 24 mm below and 12 mm above the geometric centre,
  about 10 mm below and 26 mm above the joint line.
- Centring every knee on its own two free meniscus spots (distance 50.2 mm, SD 5.7; midpoint SD 3.0,
  4.3, 6.2 mm) raises the sector divisions by 1.5 to 2.5 points and the eight emergent sections from
  55.2% to 58.7%.
- **What emerges with eight sections:** a central column (the notch); the inner medial compartment;
  an outer medial shell that also wraps the back; the inner lateral compartment; the outer lateral
  and lower part; the front upper region (kneecap and pouch) with a small centre on the kneecap; an
  anterolateral piece at the joint. With twelve: the tibia below, the posteromedial corner and the
  kneecap split further.
- **Each label is a compact blob about 15 to 20 mm across** (`heads-1.png`): ACL at the centre; MCL
  at 35 to 45 mm medial; the menisci at 20 to 35 mm on their side, just behind the side axis and
  5 to 15 mm below the centre; the OA sides at about 20 mm, 5 to 15 mm below; PF OA within 10 mm of
  the midline, 25 to 45 mm in front, 10 to 25 mm up; Effusion and Synovitis in front and 20 to 50 mm
  up; Baker's 20 to 40 mm medial, 25 to 35 mm behind; Contusion and Fracture spread, more lateral.

### Run 23 result, and logging once per epoch

Run `outputs/training/20261006-160557-side-routed-meniscus-spot-frame-eight-sections`:

- ×4 macro 0.924251 against 0.924358 for run 22 (−0.0001) and 0.924845 for run 7 (−0.0006); ×1
  0.916249 against 0.916091. Level with run 22.
- Against run 22: Lateral Meniscus +0.0015, Medial OA +0.0014, ACL +0.0011, Contusion +0.0008,
  Baker's +0.0006, MCL +0.0004, PF OA +0.0003, Effusion −0.0002, Medial Meniscus −0.0014, Synovitis
  −0.0014, Fracture −0.0017, Lateral OA −0.0027. Against run 7: PF OA +0.0018, Effusion +0.0012,
  Lateral OA +0.0011; Contusion −0.0032, Fracture −0.0027, MCL −0.0021.
- Frame from the two meniscus spots, over the validation knees at epoch 8: centre 7.4 mm down and
  2.4 mm back, roll 0.0° (4.0° in absolute value), turn −2.5° (4.9°), nod −17.8°; upper cut −4.8 mm,
  lower cut −13.5 mm, core radius 16.6 mm.
- Drawings (`outputs/analysis/side-routing-loop-20261006/side-routed-meniscus-spot-frame-eight-sections/`):
  the midline stays upright on the coronal slices and close to the notch on the axial ones (study
  21: roll −3.2°, turn −5.9°, centre 8.8 mm down, on the joint line). The joint band is thin (9 mm)
  and slopes 18°.
- Training took longer than usual because the attention extraction shared the GPU.

**Logging.** Runs 12 to 23 logged about 70 more values than run 7 (607 against 530), most of them on
every training and validation step through Lightning's logger. From the next run on, every
diagnostic value is accumulated in plain tensors and written once per epoch
(`_log_epoch_metric`, `_flush_epoch_metrics`); only `train_loss` and the gradient norms stay per
step. The values in the log are the same weighted means as before.

### Run 24 build: the spot frame with the ring-and-sector division

Agreed with the user on 2026-10-06. On run 23's build (frame on the two meniscus spots, routed v4
reports, hard assignment with the soft gradient, knee-level term on every routed target, per-epoch
logging):

- **Division.** Around the frame's up axis: eight 45° sectors centred on the axes (medial,
  anteromedial, anterior, anterolateral, lateral, posterolateral, posterior, posteromedial), a core
  cylinder and an outer ring, at three heights. The sector cuts are fixed angles; the four sizes
  (upper cut, lower cut, core radius, ring radius) are regressed per knee and start at the values
  the attention analysis placed them, in token pitches: +1.3, −0.9, 1.1 and 2.2 (about +18 mm,
  −12 mm, 15 mm and 30 mm in the spot-centred frame).
- **Region table**, from each label's map and its free report heads, made generous, medial and
  lateral mirrored:

  | Target | Sectors | Rings | Heights |
  |---|---|---|---|
  | Medial / Lateral Meniscus, Medial / Lateral OA | the three on its side | inner, outer, and its half of the core | tibia, joint |
  | PF OA, Effusion | the three front ones | inner, outer | joint, upper |
  | Synovitis | the three front ones | inner, outer, core | joint, upper |
  | ACL | all (core); anterior, posterior and the three lateral ones (inner ring) | core, inner | joint, upper |
  | MCL | medial, anteromedial | inner, outer | all |
  | Baker's | posterior, posteromedial, medial | inner, outer | all |
  | Contusion, Fracture | unrouted | | |

- **Report targets.** The tear places are now single sectors in the joint band (anteromedial =
  anterior horn, medial = body, posteromedial = posterior horn, and the lateral mirror); edema by
  compartment and the facets as before.
- First step on four studies: token share per region 28% for a meniscus or OA side, 23% for the
  kneecap labels, 17% for ACL, 23% for MCL, 31% for Baker's; all four sizes receive gradient. All
  nine guardrails pass.

Run `outputs/training/20261006-163650-side-routed-spot-frame-ring-sectors` (run 24):

- ×4 macro **0.925067**, the highest of the line: 0.924845 for run 7 (+0.0002), 0.924251 for run 23
  (+0.0008), 0.924049 for the old recipe (+0.0010). ×1 0.917070 against 0.917361 for run 7. Within
  one sibling SD of run 7, so level with the best and not shown to be above it.
- Per label against run 7: Synovitis +0.0043, Baker's +0.0021, MCL +0.0018, Fracture +0.0016,
  Lateral Meniscus +0.0015, Lateral OA +0.0012, Medial OA −0.0002, Medial Meniscus −0.0007, Effusion
  −0.0011, PF OA −0.0016, ACL −0.0021, Contusion −0.0041.
- Frame at epoch 6: roll and turn 3.9° and 4.3° in absolute value; centre 4.8 mm down; nod +17.8°
  (it was about −15° in runs 21 to 23); upper cut +14.6 mm, lower cut +0.2 mm, core radius
  11.6 mm, ring radius 29.0 mm.
- Ten epochs took about 20 minutes with the diagnostics logged once per epoch (4.1 steps per second
  against 3.5 to 3.7).
- S118 (run 24, ×4, τ 0.9): 0.953 public, with S113 and S115; S112, S114, S116 and S117 are 0.952 and
  the old recipe 0.950. The first notebook attempt errored on Kaggle after a minute with an empty
  log (the bundle runs locally), the second was refused for a notebook-name conflict, the third went
  through; one submission was used.
- Drawings of three knees (`…/side-routed-spot-frame-ring-sectors/regions-*.png`): side-to-side and
  height are right and the roll is 1° to 6°; the centre slides front to back (7 mm back in one knee,
  10 mm forward in another, where the core ends up in front of the notch), the turn is 2° to 8°, and
  the nod is +10° to +14° where the three runs before had −15°. Two spots fix a line; the position
  along the menisci and the tilt around the line are not anchored.

### Run 25 build: the frame on four anchors

Agreed with the user on 2026-10-06 after run 24 became the reference. On run 24's build, the frame
uses two more spots, each the squared-attention centroid of a head, without gradient:

- **Notch:** where the ACL head looks over the whole knee. The centre's front-back position is the
  mean of the two meniscus spots and the notch along the level front axis; side-to-side and height
  still come from the midpoint of the meniscus spots.
- **Kneecap:** where the shared OA head looks in the front-upper part of the geometric frame. The
  front axis points from the centre to that spot, lowered by one learned angle shared by all knees
  (`kneecap_elevation`, starting at 26°, the elevation measured between the PF OA spot and the
  meniscus midpoint in run 22). This fixes the nod per knee; the per-knee nod regression and the
  three free landmarks are removed. The four sizes stay regressed per knee.
- With untrained heads the kneecap spot is the centroid of the front-upper region, so the first
  frames are nodded by up to 26°; they should settle as the OA head learns where the kneecap is.
- Logged: `kneecap_elevation_deg`. All nine guardrails pass.

### Attention temperature on the reference (run 24), ×4

Sweep at inference on fold 1's validation knees, with run 24's own code (the temperature also
divides the scores the frame's spots are read from, so it moves the frame slightly):

| τ | Macro | Largest movers against τ 1.0 |
|---:|---:|---|
| 0.70 | 0.92463 | MCL +0.0031, Fracture +0.0025; Synovitis −0.0057, Baker's −0.0034 |
| 0.80 | 0.92501 | MCL +0.0026, Fracture +0.0017; Synovitis −0.0034, Baker's −0.0022 |
| 0.90 | 0.92505 | MCL +0.0015, Fracture +0.0011; Synovitis −0.0017, Baker's −0.0010 |
| 1.00 | 0.92508 | |
| 1.10 | 0.92479 | MCL −0.0009, Fracture −0.0016 |
| 1.25 | 0.92442 | MCL −0.0029, Fracture −0.0033; Baker's +0.0012, Synovitis +0.0013 |

The macro is flat from 0.8 to 1.0 (0.00007 apart); the local ×4 score is the τ 1.0 one, and S118 was
submitted at 0.9 like the earlier bundles. Sharper attention helps the small own-head labels and
softer helps the routed front and back-medial ones, and they cancel.

Run `outputs/training/20261006-171738-side-routed-four-anchor-frame` (run 25):

- ×4 macro 0.924419 against 0.925067 for the reference (−0.0006); ×1 0.915438 against 0.917070
  (−0.0016). Not better; the reference stays run 24 and the worktree code is back on run 24's build
  (run 25's is in its source snapshot and `scratchpad/run25-build`).
- Per label against the reference: ACL +0.0028, Contusion +0.0027, PF OA +0.0009, Lateral Meniscus
  +0.0005, Effusion +0.0003, Medial OA 0.0000, Medial Meniscus −0.0001, Baker's −0.0009, Fracture
  −0.0022, MCL −0.0023, Lateral OA −0.0034, Synovitis −0.0059.
- Frame over the validation knees at epoch 7: nod −10.3° and stable from epoch 4 (the reference
  wandered between +8° and +18°); roll 3.8° and turn 6.0° in absolute value (4.5°, 5.6°);
  front-to-back shift 3.9 mm in absolute value, the same as the reference; the shared kneecap
  elevation went from 26° to 28.7°.
- So the kneecap anchor fixes the tilt, at −10°, and the notch does not tighten the centre. The
  reference's free tilt sat at the opposite sign (+10° to +18°), which is where the labels put it;
  the shared elevation is one slow number and could not follow in ten epochs.
- Drawing of study 45: the centre is 6.8 mm forward (9.6 mm in the reference), roll −6°, and the
  lower cut has risen above the centre (+6 mm) with the upper cut at +14 mm, leaving a joint band
  8 mm thick. The two height cuts act as the kneecap labels' floor and the menisci's ceiling, not
  as the joint line.

## The old recipe's attention against the reference's (2026-10-06)

`outputs/analysis/side-routing-old-vs-reference-20261006/` (`old.npz`, `reference.npz`,
`old-vs-reference.png`; scripts `extract_label_attention.py`, `compare_old_reference.py` in the
`scripts/adhoc/side_routed_session/`). The old recipe (`slab-rule-and-positive-std-requeue`) and run 24 on the 874
validation knees; both see identical images and tokens, so attention is compared token by token.
Positive knees, the ten most attended tokens.

| Label | Old model's tokens outside the reference's region | Same tokens in both top tens | Spots apart | ×1 AUC, reference − old |
|---|---:|---:|---:|---:|
| ACL | 35% | 40% | 10.0 mm | −0.0045 |
| Synovitis | 34% | 25% | 9.1 mm | 0.0000 |
| MCL | 29% | 43% | 9.5 mm | +0.0090 |
| Effusion | 24% | 47% | 7.6 mm | +0.0035 |
| PF OA | 13% | 45% | 6.0 mm | +0.0002 |
| Lateral OA | 13% | 35% | 3.9 mm | +0.0007 |
| Lateral Meniscus | 6% | 56% | 2.7 mm | +0.0013 |
| Baker's | 5% | 61% | 5.7 mm | −0.0010 |
| Medial OA | 4% | 32% | 9.7 mm | −0.0037 |
| Medial Meniscus | 2% | 61% | 1.3 mm | −0.0021 |
| Contusion, Fracture | 0% (unrouted) | 49% | 5.3, 4.2 mm | +0.0046, +0.0047 |

- **ACL.** The old model's spot is at −6.5, +2.5, −3.3 mm and the reference's at −6.6, −2.7, +5.3:
  the same place in the plane and 8.6 mm higher. ACL's region leaves out the tibia band, and the
  lower cut rose during training from 12 mm below the centre to the centre, so the lower part of
  the ligament and its tibial attachment are closed to it.
- **Medial Meniscus** reads the same tokens as the old model (2% outside, 1.3 mm apart) and is
  0.0021 lower while Lateral Meniscus is 0.0013 higher: the effect of one head shared between 339
  and 135 positives, not of the region.
- MCL, Effusion and Synovitis read a narrower area than the old model and score the same or
  better; the old recipe also restricted each meniscus to its side plus 10 mm across the midline.
- Next build: ACL reads its sectors and rings at all heights; nothing else changes.

### Run 26 build: ACL at all heights

The reference build (run 24) with one change: ACL's region (the core, and the inner ring of the
anterior, posterior and three lateral sectors) is read at all three heights instead of leaving out
the tibia band. All nine guardrails pass.

Run `outputs/training/20261006-180530-side-routed-acl-all-heights` (run 26):

- ×4 macro 0.923636 against 0.925067 for the reference (−0.0014); ×1 0.915117 against 0.917070.
  The reference stays run 24 and the worktree code is back on its build.
- Per label against the reference: Contusion +0.0021, Lateral Meniscus +0.0019, ACL +0.0010,
  Medial Meniscus +0.0010, PF OA +0.0009, Medial OA −0.0007, Effusion −0.0009, Baker's −0.0011,
  Lateral OA −0.0042, MCL −0.0050, Fracture −0.0050, Synovitis −0.0072.
- ACL got the tokens below the cut back and gained 0.0010, inside its noise (0.0014): the height
  of its region is not what holds it below the old recipe.
- The lower cut, no longer held by ACL, ended at +9.6 mm with the upper cut at +15.3 mm (reference
  −1.8 and +13.7): a joint band 6 mm thick, above the joint line; it crossed the upper cut at
  epochs 3 and 4. Synovitis, which reads joint and upper, lost 0.0072 (three times its noise).

### ACL and Medial Meniscus were below the old recipe from run 1

ACL at ×4 across the line: old recipe 0.9458; run 1 (ACL on its own head, unrouted) 0.9415; every
later run between 0.9387 and 0.9446, run 14 (ACL unrouted on the region build) 0.9415. Medial
Meniscus: old recipe 0.9469, runs 0.9408 to 0.9466. So the gap dates from the first build, not
from any region. Outside the routing, the first build removed two parts of the old recipe that had
measured value when adopted:

- the half-knee loss (a shared tear detector, smooth maximum over the tokens of each half of the
  knee, trained on the meniscus labels at weight 0.075): +0.0031 at adoption, interval +0.0015 to
  +0.0048 ([knee-geometry campaign](../validation/results/2026-09-24-knee-geometry-campaign.md));
- the three-compartment OA detector (a linear detector pooled over the medial, lateral and front
  regions and added to the three OA logits through a learned scale): +0.0013, PF OA +0.0102
  ([near-misses campaign](../validation/results/2026-09-25-near-misses-campaign.md)).

Script: `label_history.py` in `scripts/adhoc/side_routed_session/`.

### Run 27, stopped

`outputs/training/20261006-183038-side-routed-routed-detectors` was stopped by the user in its
first minutes: the reference with the old recipe's tear detector and OA detector pooled over the
learned regions. Not pursued: both do what the shared heads already do (one reader shared across
the sides, on each side's region, with that side's label) and differ only in pooling by a soft
maximum instead of attention; their gains at adoption were measured against separate heads.

### Run 28 build: cuts placed midway between the spots

The problem addressed: a cut settles in an anatomical place only when labels pull on both sides of
it (medial against lateral, through the shared heads); a cut with a label on one side only drifts
(the lower cut in run 26), and labels with more cases pull harder. This build stops moving the four
region sizes by gradient and places them per knee from the heads' attention spots.

- Four spots per knee, squared-attention centroids without gradient: the two meniscus spots (as in
  the reference), the notch (ACL channel on its own tokens, whole knee) and the kneecap (OA channel
  over the fitted front-up region). Each has a spread, the weighted standard deviation of its
  tokens.
- A cut between two neighbours sits at `near + (far − near) · spread_near / (spread_near +
  spread_far)`, in the knee's frame: core radius between the notch's and the menisci's distance
  from the axis; ring radius between the menisci's and the kneecap's; upper cut between the
  menisci's and the kneecap's height; lower cut the mirror of the upper one about the menisci's
  height (nothing has a spot in the tibia).
- Removed: the shared `frame_size` and the per-knee size regression. The frame itself (centre on
  the meniscus spots, side axis, learned nod, shared shift and rotation) is the reference's, and
  the soft gradient still reaches it through the token positions.
- With the reference's trained weights loaded into this build (146 validation knees,
  `midway_cut_values.py` in `scripts/adhoc/side_routed_session/`): upper cut +11.2 mm (SD 3.1), lower −12.3
  (3.1), core radius 12.3 (2.0), ring radius 19.8 (3.0); joint band 23.5 mm, never empty. The
  reference had learned +13.7 and −1.8; the attention's best fit was +18, −12, core 15, ring 30.
- All nine guardrails pass.

Run `outputs/training/20261006-184303-side-routed-midway-cuts` (run 28):

- ×4 macro 0.924384 against 0.925067 for the reference (−0.0007, under two sibling SDs); ×1
  0.916204 against 0.917070 (−0.0009). The reference stays run 24. The worktree code is left on
  this build (copy in `scratchpad/run28-build`; the reference's in `scratchpad/run24-build`).
- Per label against the reference: Contusion +0.0020, Effusion +0.0012, MCL +0.0010, Synovitis
  +0.0008, PF OA +0.0001, Medial OA −0.0004, Medial Meniscus −0.0010, Baker's −0.0016, Lateral
  Meniscus −0.0017, ACL −0.0019, Lateral OA −0.0026, Fracture −0.0041. None beyond 1.5 of its own
  sibling SDs. The kneecap labels did not pay for their floor moving down 10 mm.
- The cuts hold still over the epochs: lower −12.3 mm, upper +10.9, core radius 12.6, ring radius
  21.2 at the last epoch (reference −1.8, +13.7, 14.7, 26.5). The nod settles at +5° to +6°, where
  the reference swung between +8° and +18°.
- Drawing of study 45: the joint band lies on the joint line in both sagittal views and in the
  coronal ones, the first run where it does.
- ACL reads the smallest region of the line here (core 12.6 mm, inner ring to 21.2 mm, against 14.7
  and 26.5): the ring cut is placed between the menisci (about 20 mm from the axis) and the kneecap
  spot (about 22 mm), two neighbours at nearly the same distance, so it falls on the meniscus spots
  themselves.

### Which targets hold each cut

Drawing: `outputs/analysis/side-routing-cut-forces-20261006/cut-forces.png`. A cut is held when
targets read only one side of it from each side. In the reference's division: medial against
lateral is held by labels on both sides; front against back by three labels in front and Baker's
alone behind; the core radius by ACL inside and MCL, Baker's, PF OA and Effusion outside. Not held:
the ring radius (ACL alone), the upper cut (menisci and OA from below, nothing above) and the lower
cut (kneecap labels and ACL from above, nothing below). Those three are the cuts that drifted or
landed badly in runs 24, 26 and 28. The model works regardless because the cuts start from the
attention's fit, the regions are generous, and the attention inside each region does the fine work.

### Run 29 build: only the held cuts, with equal pull from the two sides

Drawing per label: `outputs/analysis/side-routing-cut-forces-20261006/held-cuts-sectors-per-target-3d.png`.
The reference's construction (frame on the two meniscus spots, learned nod, sizes regressed per
knee, hard cuts with the soft gradient) with the division reduced to the three cuts that labels
hold from both sides:

- **Midline:** medial against lateral.
- **One height cut** (`cut_height`, from 0.45 pitch, about 6 mm above the centre, the middle of the
  reference's two learned cuts) in place of the two; the frame's nod tilts it.
- **Core radius** (from 1.1 pitch, as before). The ring radius is gone.
- Regions: Medial Meniscus, Medial OA, MCL and Baker's read the medial side below the cut and
  outside the core; Lateral Meniscus and Lateral OA the lateral side likewise; PF OA, Effusion and
  Synovitis everything above the cut outside the core; ACL the core at all heights, alone (the
  menisci and OA no longer read their half of it); Contusion and Fracture everything.
- Report targets: the routed questions follow their label's region; edema by compartment and the
  patellar facets use the three compartments; the tear places by horn keep their sectors below the
  cut, and send no gradient to the three cuts.
- **Equal pull:** at each of the three cuts the soft gradient of the side that reads inside and of
  the side that reads outside are rescaled to their mean norm in every step (`BalancedSides`), so a
  side with more cases or more targets does not outweigh the other. Checked: sides at 323 and 3
  come out at 163 each; a silent side leaves the other unchanged.
- All nine guardrails pass.

Run `outputs/training/20261006-192417-side-routed-held-cuts-balanced` (run 29):

- ×4 macro 0.925098 against 0.925067 for the reference (+0.00003: equal); ×1 0.915496 against
  0.917070 (−0.0016). Five sections against 54. The worktree code is on this build (copy in
  `scratchpad/run29-build`).
- Per label against the reference: Effusion +0.0025, Medial OA +0.0024, Lateral Meniscus +0.0023,
  Contusion +0.0021, Fracture +0.0015, ACL +0.0013, Baker's +0.0001, PF OA −0.0002, Synovitis
  −0.0002, Medial Meniscus −0.0008, Lateral OA −0.0019, MCL −0.0085 (three of its sibling SDs; it
  lost its own corner and reads the whole medial compartment).
- The frame nodded far more than in any earlier run: 62° at the last epoch, steady from epoch 8
  (reference 14°). With the cut 12 mm along the tilted axis, the single cut is closer to a
  front-against-back plane leaning backwards than to a height: the kneecap labels read the front
  and top, the menisci and OA sides the back and bottom.
- The core shrank from 14 mm to 9 mm, smaller than one token: with ACL alone inside, the outside
  wins. Roll 5.2° and turn 4.3° in absolute value, as in the reference.
- Drawings pending: `draw_regions.py` reads the four old sizes and has to be adapted.

### The finest division in which labels hold every cut (search, 2026-10-06)

Scripts `search_held_division.py` and `search_symmetric_division.py` in `scripts/adhoc/side_routed_session/`;
results and drawings in `outputs/analysis/side-routing-cut-forces-20261006/`
(`symmetric-division-<threshold>.json`, `symmetric-division-per-target-3d.png`).

- Input: the old recipe's free attention (ten most attended tokens on positive knees) of the ten
  routed labels, centred per knee on the midpoint of its two meniscus spots. A label reads a part
  when it holds at least a threshold share of its attention; a cut is held when one label reads
  only the part before it and another only the part after. Heights, radii and sector angles are
  searched separately (3 mm and 15° steps, cuts at least 12 mm or 45° apart, every label keeping
  at least 80% of its attention), maximising the number of parts and then the weaker side's cases.
- Without mirroring the medial and lateral pairs: 18 sections at 10% and 15%, 15 at 20%. Its
  regions are not mirror images, which one shared head cannot read (the guardrail
  `shared_classifiers_blind_to_the_compartment` fails), so the search was repeated with sector
  cuts mirrored about the front-back axis and each pair reading the union of its two sides.
- Mirrored: 15 sections at 10% (three heights, five sectors, no core), 10 at 15% (two heights,
  four sectors, core), 15 at 20% (three heights at +9 and +27 mm, core 21 mm, four sectors cut at
  30°, 90°, 150° and 270° from medial through the front). The four sectors and the core are the
  same at 15% and 20%.
- Constant findings: three heights that are the labels' own levels (menisci, kneecap, pouch), not
  tibia, joint and femur; a core of 21 to 24 mm, not 14; four to six uneven sectors, one wide part
  at the back on each side.
- Weakest cuts at 20%: the front midline (Medial OA against Lateral OA, 95 cases), the core (ACL
  inside, 131), the upper cut (Effusion above, 244).

### Run 30 build: the mirrored 15-section division

The reference build with the division of the 20% search and no balancing of the gradient:

- Sizes per knee, learned as in the reference: upper cut from 2.1 pitch (27 mm), lower cut from
  0.7 pitch (9 mm), core radius from 1.65 pitch (21 mm). No ring radius.
- Sectors: front-medial 30° to 90°, front-lateral 90° to 150°, back-lateral 150° to 270°,
  back-medial 270° to 30°.
- Regions: each meniscus its half of the core and its back sector, bottom band; each OA side its
  half of the core and both sectors of its side, bottom and middle bands; PF OA the two front
  sectors, middle and top; Effusion the two front sectors, top (its own route `pouch`); Synovitis
  the core and the front sectors, middle and top; ACL the core, bottom and middle; MCL and Baker's
  the back-medial sector, bottom and middle.
- Report targets: tear place by horn on the front and back sector of its side in the bottom band,
  the body on both; the patellar facets on the front-medial and front-lateral sectors.
- All nine guardrails pass.

Run `outputs/training/20261006-200553-side-routed-mirrored-held-division` (run 30):

- ×4 macro 0.923564 against 0.925067 for the reference (−0.0015); ×1 0.914537 against 0.917070
  (−0.0025). Lower at both. The reference stays run 24 and the worktree code is back on its build
  (this run's in `scratchpad/run30-build`).
- Per label against the reference: Contusion +0.0042, Lateral Meniscus +0.0009, Medial Meniscus
  +0.0007, Medial OA +0.0006, Baker's +0.0005, ACL −0.0002, Lateral OA −0.0011, PF OA −0.0022,
  Effusion −0.0031, Synovitis −0.0050, Fracture −0.0055, MCL −0.0078.
- The cuts stay where the search placed them without any balancing: upper 27.0 mm, lower 12.9 mm,
  core radius 20.2 mm at the last epoch (starts 27, 9, 21); nod 15°, as in the reference.
- The losses are on the labels whose regions were made tightest: Effusion (pouch only), PF OA and
  Synovitis (front sectors, middle and top), MCL (back-medial sector only). The menisci, read in
  the bottom band and their back sector with their half of a 20 mm core, are unchanged or slightly
  up. Held cuts hold; regions cut at 20% of the attention leave too much out.

### ×1 beside ×4 (2026-10-06)

Over the line the gain from ×1 to ×4 is 0.0072 to 0.0090. Run 29 gains 0.0096, the largest: its ×4
score (0.925098) equals the reference's while its ×1 (0.915496) is 0.0016 below. A run counts as
level with the reference only when both multipliers agree, so run 29 is not, and the five-section
shape is not taken as a base. The seven public scores of the line (0.952 or 0.953) are too coarse
to say which multiplier predicts them. Script `x1_x4.py` in `scripts/adhoc/side_routed_session/`.

### What the thirty runs show

- The routing is worth +0.002 to +0.003 on public (0.952 or 0.953 on all seven submissions against
  0.950) and about +0.001 locally, and that was there from the first builds. Local scores of the
  whole line lie between 0.9228 and 0.9251 at ×4 against a yardstick of 0.0004.
- Worked: shared heads on each side's region; hard cuts with a soft gradient; own routes for MCL,
  Effusion and Baker's (run 7); the frame on the meniscus spots with the division from the
  attention (run 24).
- Did not: finer tokens (9, 10); tight regions (18, 19, 30); more anatomical frames (21, 25, 28:
  stable frames, same or lower score); ACL through its region (14, 26); routed v4 reports against
  v3.1 (12, 13).
- Leaving evidence out costs and including too much is free; a frame that looks right does not
  score better; fold 1 cannot rank most of these variants.
- Open: ACL (−0.007) and Medial Meniscus (−0.003) below the old recipe since run 1; the first
  build's change that touches ACL is the token position (slice and in-image position replaced by
  the position code and the attention prior).

### The runs sorted, and which labels separate them (2026-10-06)

Script `rank_runs.py` in `scripts/adhoc/side_routed_session/`; ×1 and ×4 standardised and summed.

- Order: 24, 7, 10, 13, 8, 29, 11, 28, 23, 22, 6, 5, old recipe, 16, 25, 15, 12, 4, 14, 21, 20, 26,
  19, 30, 2, 18, 9, 3, 1. ×1 and ×4 correlate at 0.72 across the routed runs.
- Top six against bottom six, per label at ×4: PF OA +0.0040 (8 of its sibling SDs), Synovitis
  +0.0050 (2.3), Medial OA +0.0011 (1.4), Effusion +0.0015 (1.0), Contusion −0.0020 (−1.6), ACL and
  MCL 0.0000, the rest under one SD. PF OA and Synovitis carry about half of the macro's variance
  between runs (shares 0.20 and 0.28).
- Spread of each label across the 28 routed runs over its sibling SD: PF OA 5.8, Medial Meniscus
  1.8, Synovitis 1.6, the other nine 0.7 to 1.4. Thirty designs moved nine labels no more than a
  rerun does.
- PF OA: old recipe 0.8994; 0.9067 to 0.9076 in runs 4, 8, 10 and 23; 0.9033 in run 24 and 0.9031
  to 0.9042 in runs 25 to 29; 0.8924 with fine tokens (run 9). Synovitis: 0.878 to 0.880 in runs 1
  and 2, 0.8918 in run 24, 0.8926 in run 28. No run has both at their best.
- Against the old recipe the routed runs gain on PF OA, Effusion (0.9182 against 0.917 to 0.922)
  and Medial OA (0.9229 against 0.922 to 0.927) and lose on ACL (0.9458 against a mean of 0.9406
  with a spread equal to its noise) and Medial Meniscus (0.9469 against 0.941 to 0.947).

### Build ready, not launched: PF OA and Effusion on the front half

The reference build with one change. The route of PF OA and Effusion goes from the three front
sectors outside the core above the lower cut (135° wide) to what it was in run 23: the front half
outside the core above the lower cut (180° wide) plus the core above the upper cut. Synovitis, the
patellar facets and everything else are the reference's. Guardrails and the training-step check
pass. The user asked for it to be built and not run yet.

Run `outputs/training/20261006-204225-side-routed-pf-oa-front-half` (run 31, launched by the user):

- ×4 macro 0.924552 against 0.925067 for the reference (−0.0005); ×1 0.916597 against 0.917070
  (−0.0005). Lower at both: not adopted, the reference stays run 24 and the worktree code is back
  on its build (this run's in `scratchpad/run31-build`).
- Per label against the reference: Effusion +0.0026, ACL +0.0022, Lateral Meniscus +0.0014,
  Contusion +0.0011, PF OA +0.0004, Medial Meniscus −0.0005, Baker's −0.0006, Medial OA −0.0013,
  Synovitis −0.0016, Lateral OA −0.0023, MCL −0.0028, Fracture −0.0049.
- PF OA did not come back: 0.9037 against 0.9033 in the reference and 0.9067 in run 23. The width
  of its region (front half and the core above the upper cut, against three front sectors) is not
  what separates run 23 from run 24 for PF OA. Effusion, on the same route, gained 0.0026.

### The reference recipe on the ResNet-50 (2026-10-06, launched by the user)

`outputs/training/20261006-210555-rn50-side-routed-reference`: `configs/refined-resnet50.yaml` on
the run 24 build, fold 1, seed 1110484336, ten epochs. ×4 0.933209, ×1 0.922934, against 0.925067
and 0.917070 for the ResNet-18 reference (+0.0081 and +0.0059). Submitted as **S119: 0.957
public** (×4, τ 0.9, batch 4, no reflection; bundle
`outputs/submission/exports/rn50-side-routed-ring-sectors-f1-x4-tau09`, record
`outputs/submission/records/rn50-side-routed-ring-sectors-f1-x4-tau09.json`). It equals the best
public score so far, S106 (five distilled ResNet-50 children, ×6, mirror TTA), with one model at
×4. No ResNet-50 without routing exists on this fold and seed.

## Crop offsets without a locator (2026-10-07, manual run after the campaign stopped)

Run `outputs/training/20261007-081431-side-routed-crop-offsets-3mm`, fold 1, seed 1110484336, on
the campaign's incumbent (`height-cuts-from-the-spots-requeue`, ×4 0.925389, ×1 0.917411). The
loader cuts each 84 and 93 mm view with a 3 mm margin (uint8 canvases, as in the campaign's light
port); the model cuts the 224 px crop. Training: each slice's crop is offset at random, uniform in
a disc of 3 mm, from its own generator. Prediction: slice k of the T valid slices of a series sits
at sunflower point k of T (radius 3·sqrt((k+0.5)/T) mm, angle k·137.5°), a single slice at the
anchor. The 120 mm axial views do not move. The affine augmentation is unchanged. Build backup:
scratchpad `crop-offsets-build`; the worktree is on this build.

- **×4 0.925012 (−0.0004), ×1 0.914850 (−0.0026): not adopted.** Guardrails passed, training
  1,263 s. Per label at ×4: Medial OA +0.0019, Lateral OA +0.0015, Synovitis −0.0038, MCL −0.0029,
  Baker's −0.0017, the rest within 0.0009.
- **The two halves, by predicting with the pattern and at the anchor**
  (`outputs/analysis/routing-and-shifts-frames-20261007/mcl-spatial-maps/crop-offsets-on-off-x{1,4}.npz`,
  `incumbent-pattern-on-off-x{1,4}.npz`):

  | Weights | Multiplier | Pattern | Anchor | Pattern's effect |
  |---|---|---:|---:|---:|
  | Trained with offsets | ×4 | 0.9250 | 0.9242 | +0.0008 |
  | Trained with offsets | ×1 | 0.9148 | 0.9154 | −0.0006 |
  | Incumbent, no retraining | ×4 | 0.9259 | 0.9253 | +0.0005 |
  | Incumbent, no retraining | ×1 | 0.9167 | 0.9174 | −0.0007 |

  Training with re-cut offsets costs 0.0012 at ×4 and 0.0020 at ×1 (anchor against the
  incumbent), in line with the learnable-views jitter runs 17 and 43 (−0.0008, −0.0009). The
  pattern at prediction helps at ×4 and hurts at ×1 on both sets of weights. On the incumbent at
  ×4: Lateral Meniscus +0.0040, Lateral OA +0.0032, MCL +0.0012, Medial OA +0.0012, Synovitis
  −0.0017, Fracture −0.0012, ACL −0.0011, PF OA −0.0010. The spot-shift locator's shift this
  morning had the same size and labels at ×4 (+0.0007).
- **Lesson.** A spread of crop positions over a series's slices is a prediction-time gain of
  0.0005 to 0.0008 at ×4 that needs no training, concentrated on the lateral labels; random
  re-cut offsets in training do not help.
- **Pattern by plane, incumbent's weights** (`pattern-by-plane-*-x{1,4}.npz` in the same folder).
  Macro against the anchor at ×4: coronal only +0.0004 (0.925787), the other detail plane only
  +0.0002, both +0.0005. At ×1: −0.0001, −0.0004, −0.0007. Coronal only at ×4: Lateral Meniscus
  +0.0032, Lateral OA +0.0019, MCL +0.0014, ACL −0.0009, the rest within 0.0007; the Synovitis
  (−0.0012) and Fracture (−0.0018) losses come from the other plane. The two planes' effects
  add up, so restricting the pattern to coronal views keeps most of the gain at ×4 and removes
  the ×1 cost, without gaining more than moving both.
- **What the pattern's gain is made of, incumbent's weights, coronal views, ×4**
  (`pattern-directions-coronal-x4.npz`, `pattern-radius-coronal-x4.npz`). Macro against the
  anchor (0.9253):

  | Variant | Macro | Variant | Macro |
  |---|---:|---|---:|
  | Sunflower 3 mm | +0.0004 | All slices 3 mm lateral | +0.0002 |
  | Sunflower 1.5 mm | +0.0002 | All slices 3 mm medial | −0.0003 |
  | Sunflower 4.5 mm | −0.0000 | All slices 3 mm up | −0.0005 |
  | Sunflower 6 mm | +0.0002 | All slices 3 mm down | −0.0003 |
  | Sunflower, across only | +0.0001 | Alternating ±3 mm across | −0.0001 |
  | Sunflower, down only | +0.0002 | Alternating ±3 mm down | −0.0002 |
  | | | Alternating ±6 mm across | −0.0001 |
  | | | Alternating ±6 mm down | −0.0000 |

  The fourteen variants average 0.0000 with a spread of about 0.00025, and the gain does not
  grow with the radius, so +0.0004 for the 3 mm sunflower is inside what any small displacement
  of the crops does. No direction carries it, a constant shift to any side does not help (no new
  anatomy is being reached), and two fixed positions per plane do not help. Per label the one
  repeat is Lateral Meniscus, above the anchor in 11 of 14 variants (mean +0.0011, largest
  +0.0032). Moving every coronal crop 3 mm medially costs Lateral OA 0.0043, the largest single
  effect: the lateral edge of the coronal crop matters to it, and moving 3 mm laterally does not
  add to it (−0.0002).
- **Lesson.** The prediction pattern is not a finding on the incumbent: an earlier line here
  calls it a gain of 0.0005 to 0.0008, measured with one pattern on three sets of weights; across
  patterns on one set it is zero on average. It stays out of the endgame list.
- **Where the pattern's effect on the lateral labels comes from** (incumbent's weights, ×4, the
  3 mm sunflower on both detail planes; `pattern-frame-swap-*.npz`,
  `pattern-content-or-position-*.npz`). Three interventions, each against the anchor:

  | Label | Pattern | Pattern crops, anchor's frame | Anchor crops, pattern's frame | Pattern pictures, anchor positions | Anchor pictures, pattern positions |
  |---|---:|---:|---:|---:|---:|
  | Lateral Meniscus | +0.0040 | +0.0038 | +0.0002 | +0.0036 | −0.0000 |
  | Lateral OA | +0.0032 | +0.0032 | −0.0005 | +0.0034 | −0.0005 |
  | MCL | +0.0012 | +0.0012 | −0.0004 | +0.0005 | +0.0001 |
  | Macro | +0.0005 | +0.0005 | −0.0001 | +0.0006 | −0.0001 |

  The effect is in the pictures the backbone sees. It is not the frame (which moves 0.5 mm and
  1.1° at the median between the two passes), and not the token positions used by the routing and
  the position code. In logits the pattern lowers scores slightly; for Lateral Meniscus the
  negatives in the top tenth of scores fall by 0.056, positives in the same score range by 0.025
  and the other negatives by 0.013, so confident false positives lose more than true positives.
  Lateral OA shows the same order with a smaller gap (0.064, 0.049). The frame-precision
  explanation proposed first is refuted by the frame swap.
- **Sensitivity to the crop position per label** (incumbent, ×4, coronal crops at the anchor and
  3 mm lateral, medial, up and down; spread of a knee's logit over the five, divided by the gap
  between the mean logit of positives and negatives; `pattern-directions-coronal-x4.npz`):
  Lateral Meniscus 0.029, MCL 0.024, Lateral OA 0.023, Medial OA 0.022, Medial Meniscus 0.022,
  then Fracture, Contusion and ACL at 0.014, Synovitis 0.013, Effusion 0.011, PF OA 0.010,
  Baker's 0.007. The five labels read on the coronal views by side are the sensitive ones, to the
  same degree within 1.3 times; the lateral ones are not a class apart. No relation to the number
  of positives (correlation −0.14), so the imbalance explanation is dropped. All four directions
  move a score by the same amount, except MCL (0.08 across against 0.05 up or down). Only the
  coronal views were moved, so the ranking also reflects how much each label reads from them.

## Crops cut by each knee's frame (2026-10-07, prediction only, no training)

The user's idea: cut the crops following the frame the incumbent infers for each knee. Checked on
the incumbent's weights with the canvas loader (15 mm margin): a first pass at the anchor gives
the knee's frame; the 84 and 93 mm views are then re-cut and predicted again. Centring moves the
window by the frame centre's deviation from the fold mean, projected into the slice; rotation
turns the window in its plane so that the frame's side axis is level on coronal views and its up
axis is vertical on the other detail plane (clamped at 15°; 120 mm axial views untouched). Scripts
`frame_crops.py` and `frame_rotation_controls.py` in `scripts/adhoc/side_routed_session/`; outputs in
`outputs/analysis/routing-and-shifts-frames-20261007/frame-crops/`.

| Variant | ×4 (anchor 0.925344) | ×1 (anchor 0.917415) |
|---|---:|---:|
| Centred | −0.0004 | |
| Rotated | +0.0011 | −0.0002 |
| Centred and rotated | −0.0007 | |
| Rotated the wrong way (minus the angle) | −0.0007 | −0.0009 |
| Rotated by half the angle | +0.0010 | +0.0007 |
| Rotated, coronal views only | +0.0002 | +0.0002 |
| Rotated, other detail plane only | +0.0010 | −0.0004 |

- Rotation applied: median 3.6° on coronal (90th percentile 8.7°), 5.9° on the other plane
  (11.5°); the frame's axis is level in the re-cut crop to 0.00°. Centring moves the window 5 to
  7 mm at the median and 6% of slices reach the margin.
- The sign of the rotation matters: the frame's angle against its opposite is +0.0018 at ×4 and
  +0.0007 at ×1, with the same resampling in both, so this is alignment and not the picture
  disturbance measured earlier today (spread 0.00025 over fourteen offset variants).
- Per label at ×4, full rotation: Fracture +0.0041, Lateral OA +0.0033, Lateral Meniscus
  +0.0031, MCL +0.0027, Contusion +0.0018, ACL and Baker's −0.0008. The wrong way costs ACL
  0.0048 and Fracture 0.0024.
- Half the angle is as good at ×4 and better at ×1, where it is the only variant above the anchor
  at both multipliers. The gain comes from the plane that shows the forward tilt, whose angle is
  the one still learned from the labels.
- Centring costs Fracture 0.0088 and PF OA 0.0037 and gains Medial Meniscus, Contusion, Effusion
  and ACL by about 0.002 each; it overrides the per-view offsets of the view table.
- All of this is with weights trained on unaligned crops (and ±10° of rotation augmentation).
- **The rotation on the two reference models** (worktree build: run 24's code with the canvas
  loader at a 6 mm margin and the rotation inside the model as a first look and a re-cut;
  `rotation-check-*-x4.npz`), ×4 against the same model at the anchor:

  | Weights | Anchor | Half angle | Full angle |
  |---|---:|---:|---:|
  | ResNet-18 run 24 | 0.925091 | +0.0007 | +0.0005 |
  | ResNet-50 on run 24's recipe (S119) | 0.933158 | +0.0002 | +0.0001 |

  Three sets of weights now gain at ×4 with half the angle (+0.0010, +0.0007, +0.0002), the
  ResNet-50 least. On all three Lateral Meniscus and Lateral OA gain 0.002 to 0.004; on the
  ResNet-50 ACL, Medial OA and PF OA each lose 0.0011. The anchor passes reproduce the recorded
  scores to 0.00005.
- **Not submitted.** The export refuses the ResNet-50 checkpoint with the new loader and model
  code: its `preprocessing_identity` differs from the executing sources, and the check says not
  to rewrite checkpoint identities. A model trained with this build would export normally.

## The knee's bend between femur and tibia (2026-10-07, measurement attempt)

Asked while discussing a frame of three attached segments instead of one cylinder. Two rule-based
estimates of each bone's axis on the 874 validation knees, from the series at the full 120 mm
field (`measure_bend.py`, `measure_bend_marrow.py` in `scripts/adhoc/side_routed_session/`; outputs in
`outputs/analysis/routing-and-shifts-frames-20261007/knee-bend/`):

- **Centre of the limb's outline per height.** Below the joint the centre drifts backwards by
  14° on average (SD 9.5°): that is the calf, not the tibia. Above the joint it is straight up to
  +50 mm and then bent by the end of the field. Not a bone axis.
- **Bright pixels well inside the limb on the T1/PD series.** They follow Hoffa's fat pad in
  front of the joint (+14 mm at the joint line) and the popliteal fat behind the femur
  (−15 mm above +40 mm); slopes have an SD of 21 to 35°. Not a bone axis either.
- Neither correlates with the frame's forward tilt (|r| ≤ 0.12). No number for the bend came out
  of this.
- **By eye**, on eighteen sagittal slices through a condyle (`sagittal-gallery.png`): the tibia
  stands close to vertical in nearly all of them, and the femur leans by anything from about 0°
  to 30° or more (rows 532 and 200 clearly bent; 370, 374 and 284 nearly straight). The 120 mm
  field shows about 60 mm of each bone, mostly condyle and plateau and little shaft.

## Femur segment with its own lean (2026-10-07, manual run on the incumbent)

Run `outputs/training/20261007-101112-side-routed-femur-lean`, fold 1, seed 1110484336. The frame
above the upper cut is laid out around an axis that leans forward or back from the centre of that
cut: `front' = front − max(height − upper cut, 0)·tan(lean)`. Lean per knee = `35°·tanh((start +
correction)/35°)`; start = atan of half the kneecap spot's forward offset from a shared learned
distance (1.57 token pitches at the start, the measured mean) over its rise above the cut, no
gradient through the spot; correction regressed per knee from the pooled frame tokens, zero at the
start, label gradient. Build backup: scratchpad `femur-lean-build`; the worktree is on this build.

- Before training, on the incumbent's weights: start lean mean −0.3°, SD 9.8°; five knees bent by
  eye +2.5° on average, five straight ones −9.6°.
- **×4 0.925202 (−0.0002), ×1 0.916509 (−0.0009): not adopted.** Guardrails passed, 1,213 s.
- Per label at ×4: Effusion +0.0021, Synovitis +0.0021, Medial OA +0.0010, MCL −0.0024, Fracture
  −0.0021, Lateral Meniscus −0.0018, Contusion −0.0013, PF OA −0.0005, ACL −0.0002.
- **The lean went to one side for every knee.** Validation mean −2° after epoch 1, −30° after
  epoch 2, −25° at the end, with the mean size equal to the mean, so the start's spread between
  knees was overridden by the regressed correction; the shared distance stayed at 1.60. The whole
  frame's forward tilt ended at 15° against 6° for the incumbent, the core at 17.4 mm (16.3) and
  the ring at 28.2 mm (29.8).
- Drawing of the frame per label at epoch 7:
  `outputs/analysis/routing-and-shifts-frames-20261007/femur-lean-per-label-3d.png`.
- **Lesson.** A lean learned from the labels with a start from the spots does not keep the start:
  the labels reading the upper band (PF OA, Effusion, ACL and the top of MCL and Baker's) pull it
  one way and nothing pulls back. Effusion gains with it and PF OA does not.

