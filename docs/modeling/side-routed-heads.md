# Side-routed shared heads

One shared meniscus classifier and one shared OA classifier scored every token without knowing its
compartment; a knee frame inferred per knee from the heads' own meniscus attention spots, with the
knee side, decided which medial, lateral or patellofemoral target the evidence counted for. The
adopted version is run 24 of a 31-run line on ResNet-18 (fold 1 of `all_5fold_scanner_v2`, seed
1110484336): ×4 0.925067, ×1 0.917070, 0.953 public (S118), against ×4 0.924049 and 0.950 public
(S108, S111) for the recipe without routing; the ResNet-50 on run 24's recipe scored 0.957 public
(S119).

Built 2026-10-05 and 2026-10-06 in worktree `knee-abnormality-side-routed-heads` (branch
`side-routed-heads`), designed with the user after the
[fixes-and-removals campaign](../../autoresearch/campaigns/fixes-and-removals/INDEX.md) left every
position encoding at a tie. Adopted on main on 2026-10-07 (`609a489`, first as run 24 with the
height cuts from the spots; `99dfcee` kept run 24 itself as the recipe). The anatomy-map and
posed-crop recipe of 2026-10-08 kept the shared heads and replaced their frame and sections, and
later recipes followed ([adoption history](current-model.md#adoption-history)). Runs and
measurements: [side-routed heads results](../validation/results/pooling-and-attention/side-routed-heads/INDEX.md).

The session's one-off analysis and drawing scripts other than the ones those results name were deleted;
the last commit holding them is `d8def44:scripts/adhoc/side_routed_session/`.

## Principle

- **Shared classifiers.** Medial and lateral meniscal tears, and medial, lateral and
  patellofemoral OA, are each one disease in several places. One classifier per disease learned
  from every compartment; on fold 1 Lateral Meniscus had 135 positives against 339 for medial,
  and Lateral OA 95 against 189.
- **Routing, because the classifier was shared.** One output had to become two or three targets,
  and the image could not say which: the 84 and 93 mm views are centred on the knee, and the cue
  for the side (the fibula) is at their edge or outside.
- **Blind to the compartment.** A shared classifier read position-free tokens, so it could not
  tell which compartment a token was in; geometry alone assigned its output to a target.
- **A cut holds only where targets compete across it.** Medial against lateral targets held the
  midline; a cut with a label on one side only drifted
  ([which targets hold each cut](../validation/results/pooling-and-attention/side-routed-heads/2026-10-06-held-cuts-and-summary.md#which-targets-hold-each-cut)).
- **Earlier evidence.** The half-knee loss was a shared detector pooled per half, adopted at
  +0.0031 with the own-side mask (2026-09-24). The three-compartment OA detector, shared over
  medial, lateral and patellofemoral regions, gave PF OA +0.0102 when adopted (2026-09-25).
  Neither was ever the predictor.

## Design as adopted (run 24)

Run `outputs/training/20261006-163650-side-routed-spot-frame-ring-sectors`. Each part below
names the build that introduced it; the octant routing it replaced is described under the
[fifth build](../validation/results/pooling-and-attention/side-routed-heads/2026-10-05-fourth-and-fifth-builds.md#fifth-build-2026-10-05).

### Channels

The 12 targets were read from 9 channels. ACL, MCL, Effusion, Synovitis, Baker's, Contusion and
Fracture kept one channel each. Two shared channels fed the five compartment targets:

| Shared channel | Targets it feeds |
|---|---|
| `meniscus` | Medial Meniscus, Lateral Meniscus |
| `oa` | Medial OA, Lateral OA, PF OA |

The attention scorer, the plane-contrast readout and the view label bias had 9 outputs instead
of 12; each of the 12 targets kept its own bias. In run 24 every target except Contusion and
Fracture read a region of the knee; in the first five builds only the five compartment targets
were routed.

### Pooling

For a routed target `k` of channel `h`, with `r_k` the token's routing weight in [0, 1]:

```text
attention_k = softmax over the study's valid tokens of (score_h + prior_k + log r_k)
logit_k     = sum over tokens of attention_k * evidence_h + bias_k
```

- **Knee-level term on every routed channel** (run 18): one learned number per channel, zero at
  the start; the target's score was its region's evidence plus that share of the same head's
  evidence pooled over the whole knee. In run 18 every number learned negative, so the term acted
  as a contrast (region minus knee), not as a wider view.
- At inference the temperature divided the content score, the view bias, the prior and the
  routing term together (third build on).

### Knee frame (run 23 on)

- **Geometric frame.** The token's left-right, anterior and head-foot offsets from the fitted
  centre in mm, after the rotation from the headers: the scanner's left-right axis rotated about
  the head-foot axis to the mean row direction of the study's axial and coronal series; a study
  without one kept the scanner's axes. The side sign made the medial coordinate medial-positive
  for every knee.
- **Two spots per knee.** The shared meniscus head's attention over the medial half and over the
  lateral half of the geometric frame gave two spots, centroids weighted by the squared
  attention, with no gradient.
- **Centre and side axis.** The centre was the spots' midpoint plus one shared learned shift; the
  side axis was the line between them, which fixed turn and roll per knee.
- **Nod and sizes per knee.** The frame head (attention-pooled over the study's detached tokens,
  256 to 128) regressed the nod and the four sizes per knee, on top of shared values, as in run
  21. It was trained only by the diagnosis loss through the soft gradient and the prior, with no
  gradient into the backbone.
- With untrained heads the spots sat near each half's centroid (first step on four studies:
  centre within 7 mm and axes within 3° of the geometric frame).
- Two spots fixed a line; the position along the menisci and the tilt around the line were not
  anchored. Run 24's drawings showed the centre sliding 7 mm back in one knee and 10 mm forward
  in another.

### Division and region table (run 24)

Around the frame's up axis: eight 45° sectors centred on the axes (medial, anteromedial,
anterior, anterolateral, lateral, posterolateral, posterior, posteromedial), a core cylinder, an
inner ring (out to the ring radius) and an outer ring, at three heights. The sector cuts were
fixed angles. The four sizes (upper cut, lower cut, core radius, ring radius) were regressed per
knee and started where the [attention analysis](../validation/results/pooling-and-attention/side-routed-heads/2026-10-06-hard-cut-regions.md#the-shape-read-from-the-heads-attention-2026-10-06)
placed them, in token pitches +1.3, −0.9, 1.1 and 2.2 (about +18 mm, −12 mm, 15 mm and 30 mm in
the spot-centred frame).

The table came from each label's attention map and its free report heads, made generous, medial
and lateral mirrored:

| Target | Sectors | Rings | Heights |
|---|---|---|---|
| Medial / Lateral Meniscus, Medial / Lateral OA | the three on its side | inner, outer, and its half of the core | tibia, joint |
| PF OA, Effusion | the three front ones | inner, outer | joint, upper |
| Synovitis | the three front ones | inner, outer, core | joint, upper |
| ACL | all (core); anterior, posterior and the three lateral ones (inner ring) | core, inner | joint, upper |
| MCL | medial, anteromedial | inner, outer | all |
| Baker's | posterior, posteromedial, medial | inner, outer | all |
| Contusion, Fracture | unrouted | | |

First step on four studies: token share per region 28% for a meniscus or OA side, 23% for the
kneecap labels, 17% for ACL, 23% for MCL, 31% for Baker's.

### Assignment: hard cuts with a soft gradient (run 18)

The forward pass took hard steps at every cut; the gradient of each cut was that of a sigmoid of
the token's signed distance over its box's extent along the cut's normal (the box from the
token's two in-plane steps and the step to the next slice, projected on the frame axes). The
sizes learned through it. This was run 7's mechanism (there with the token pitch, 12 to 17 mm,
as the scale), replacing the overlap shares of runs 12 to 16. Routing weights were 0 or 1 on
side-certain knees.

### Position prior on the attention

```text
position           = (m, f, h), the frame coordinates                     in mm
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

- The map was linear, started at zero and added straight to the attention score. Medial and
  lateral targets shared one map read from their own side, so the prior carried no compartment
  either. A token without a fitted centre had a prior of zero.
- From run 12 on, the own-channel targets' prior used the fixed geometric frame and the shared
  targets' prior the inferred frame.

### Token position code (fifth build)

A token read by an own-channel classifier carried a fixed sinusoidal code of its position (42
wavelengths from 8 to 512 mm per axis, norm 11.2, the scale of the reference's slice-position
code), added before the normalisation, with the position detached; from run 12 on, in the fixed
geometric frame. A token read by a shared classifier carried none. A token without a fitted
centre had a code of zero. The report-attribute heads read the own-channel version.

### Knee side

The side was used by the routing and by the sign of the medial coordinate, and nowhere else.

- **Certain studies** (`explicit_coordinates_agree`, 2,072 of 4,407 in training): the metadata
  side routed. These were the only studies that trained the side head.
- **Every other study** routed by a probability, as a mixture,
  `r_medial = p_left * r_medial(as a left knee) + (1 - p_left) * r_medial(as a right knee)`,
  with the gradient cut so the disease loss could not move the head. At 0.5 the evidence was
  shared equally between the two sides.
  - **Side from coordinates alone (2,082 studies):** the coordinates were a prior worth 4.33 in
    log-odds (their 98.7% agreement with reports), added to the side head's logit. A confident
    image overruled the coordinates (a scanner with another coordinate convention); the prior
    also routed these studies from the first step, while the head was untrained.
  - **No side (170 studies):** the side head's probability alone.
- **Side head** (fourth build): a small network (256 to 128, tanh, to 1) scored each of the
  study's axial and coronal tokens with a fitted centre; the logit for "left knee" was the mean
  of those scores weighted by `tanh(u / width)`, `u` the left-right offset. Its input was
  detached, so the side loss trained this network and its learned width and nothing else. Binary
  cross-entropy on the certain studies, weight 0.1. It reached 99.8% on certain validation
  studies.
- **Reflection:** the head read `-u` and its target flipped when the training reflection was
  applied, and the routing probability was flipped back, because the coordinates stayed those of
  the source. `reflect_batch` marked the mirrored pass so reflection TTA did the same.
- Series whose side came from the bilateral split kept it.

### Report-attribute targets (run 10 on, as in run 24)

Target set 4.0 (`configs/labels/jevk5-report-attribute-targets-v4.0.json`, 37 questions), routed
as "4.1" at training time (`training/report_attributes.py`):

- The ten compartment questions (meniscus status and tear type; cartilage depth and extent)
  shared one head per family and were routed like their label, on the position-free tokens. ACL
  status and PCL were routed to the core.
- The three location questions became region targets on shared heads: a region's target was the
  summed probability of the options placing the finding there, weighted by the question's weight
  times one minus the mass on options that state no place. In run 24 the tear places were single
  sectors in the joint band (anteromedial = anterior horn, medial = body, posteromedial = posterior
  horn, and the lateral mirror); marrow edema by compartment; the patellar facet by side of the
  kneecap part.
- The other questions kept their own heads. Logged: `report_attribute_bce_<family>_<route>`,
  `val_report_attribute_fit` and `val_report_attribute_bce_<region>`.

### Removed by the line

- First build: the meniscus own-side mask and its midline band, the half-knee loss and its
  detector (`trainer.half_knee_loss_weight`), the OA detector and its regions, the
  per-compartment attention and readout weights of the five compartment targets, and the 1D slice
  position and 2D image position on the tokens. The anatomical attention bias
  (`model.anatomical_bias`) was already off in the reference.
- Fourth build: the routed auxiliary loss of the third build. Fifth build: the learned lateral and
  front offsets and the per-study front scorer. Run 6: the boundary widths. Run 23: the free
  regression of the centre and the centre landmark.

Kept from the reference: the centre fit, the metadata side policy, the report-attribute heads.

### Logged per epoch

The frame angles and their absolute means (`study_frame_{nod,roll,turn}_deg`, `_abs_deg`), the
centre shifts (`study_frame_shift_{medial,front,up}_mm`, `_abs_mm`), the sizes
(`study_frame_{band_top,band_bottom,core_radius}_mm` from run 12); the side head's accuracy on
certain studies in training and validation; for coordinates-only studies, how often the final
probability sided with the coordinates; the share of studies in each side state; the mean
routing weight per compartment; the attention mass of each target per view. From run 24 every
diagnostic value was accumulated in plain tensors and written once per epoch
(`_log_epoch_metric`, `_flush_epoch_metrics`); only `train_loss` and the gradient norms stayed
per step. Runs 12 to 23 had logged 607 values against 530 for run 7, most on every step; ten
epochs took about 20 minutes once logged per epoch (4.1 steps per second against 3.5 to 3.7).

### Guardrails

Nine checks ran before every launch: among them, a side flip swapped the medial and lateral
logits of both shared channels exactly and left PF OA unchanged; no side-loss gradient reached the
trunk; shared classifiers stayed blind to the compartment
(`shared_classifiers_blind_to_the_compartment`); learned geometry started at zero (from run 23,
the shared rotation and the prior only); the mirrored pass kept each token's geometric position.

## How the design evolved

Results docs: [first to third builds](../validation/results/pooling-and-attention/side-routed-heads/2026-10-05-first-to-third-builds.md),
[fourth and fifth builds](../validation/results/pooling-and-attention/side-routed-heads/2026-10-05-fourth-and-fifth-builds.md),
[octant builds](../validation/results/pooling-and-attention/side-routed-heads/2026-10-06-octant-builds.md),
[eighteen regions](../validation/results/pooling-and-attention/side-routed-heads/2026-10-06-eighteen-regions.md),
[hard-cut regions](../validation/results/pooling-and-attention/side-routed-heads/2026-10-06-hard-cut-regions.md),
[spot-frame reference](../validation/results/pooling-and-attention/side-routed-heads/2026-10-06-spot-frame-reference.md),
[held cuts and summary](../validation/results/pooling-and-attention/side-routed-heads/2026-10-06-held-cuts-and-summary.md),
[manual runs of 2026-10-07](../validation/results/pooling-and-attention/side-routed-heads/2026-10-07-manual-runs-after-the-campaign.md).

1. First build: shared heads routed by a learned boundary and the side head, no position (×4 0.923102).
2. Second build: knee frame from the header rotation, learned offsets, token position code (0.923534, S112 0.952).
3. Third build: detached side head and front scorer, position prior on the attention, routed auxiliary loss (0.923085).
4. Fourth build: auxiliary loss removed, per-token side head (0.923952, S113 0.953).
5. Fifth build: octants from the fitted centre and one learned rotation, own-channel position code (0.924476, S114 0.952).
6. Runs 6 to 11: hard octants, soft gradient and own routes for MCL, Effusion and Baker's (run 7, 0.924845, S116 0.952), frame per study (S117 0.952), fine tokens, landmarks.
7. Runs 12 to 17: eighteen regions with learned sizes and overlap assignment, routed v4.0 reports.
8. Runs 18 to 23: hard cuts with the soft gradient, knee-level term, wider tables, per-knee nod, the meniscus-spot frame.
9. Run 24: the spot frame with eight sectors, core and rings (0.925067, S118 0.953); runs 25 to 31 tried four anchors, ACL at all heights, midway cuts, held cuts only, the searched division and the front half for PF OA, none adopted.
10. ResNet-50 on run 24's recipe: ×4 0.933209, S119 0.957.

## Variants considered

| Variant | State |
|---|---|
| MCL routed from medial tokens and ACL from a window on the midline (first build) | Built, then removed: both had their own classifier, so nothing to route; MCL was routed again from run 7 |
| No position at all | The first run |
| A learned lateral offset, a learned front offset, a per-study front scorer; front tokens also counting for Medial and Lateral OA; no token position for the own-channel classifiers | The first four runs |
| A position code on the tokens, unsigned for the shared classifiers | The second run: the code stayed at a tenth of the token's norm |
| The side head on the mean of undetached tokens | The first two runs: it trained the trunk |
| A routed half-knee auxiliary loss on the meniscus targets, weight 0.075; a linear side head on pooled detached tokens | The third run |
| A signed 3D position for the shared classifiers | Rejected: the classifiers would know the compartment |
| A per-study midline landmark | The first run: it learned a constant |
| Compartment-relative (mirrored) position for the shared heads | Run 16: level |
| The old tear and OA detectors pooled over the regions | Run 27, stopped by the user: they do what the shared heads already do |
| The fitted centre with nothing learned; PF OA on its own channel; the side head alone for coordinates-only studies; a centre correction per plane | Not run |
