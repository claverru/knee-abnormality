# Side-routed heads: octant builds, runs 6 to 11 (2026-10-06)

**Hard octants with a soft gradient and own routes for MCL, Effusion and Baker's (run 7) reached
×4 0.924845 and ×1 0.917361, the line's best until run 24 (S116 0.952 public).** A frame inferred
per knee (runs 8, 10, 11) gained PF OA +0.0016 to +0.0018 and lost MCL, with the macro level;
stride-16 tokens alone cost 0.0020 (run 9), and adding deep features removed that loss without
gaining. All runs: fold 1 of `all_5fold_scanner_v2`, seed 1110484336, ResNet-18. Design:
[side-routed heads](../../../../modeling/side-routed-heads.md).

## Hard octants on one frame (run 6)

Run `outputs/training/20261006-073158-side-routed-hard-octants`: run 5 plus PF OA from
front-and-up, the knee-level term, one frame, and hard cuts (no boundary widths; the frame learned
only through the position prior). The separate OA frame of the campaign run
`separate-oa-frame-with-knee-level-term` (×4 0.924485) was dropped: its two frames turned +6.3° and
−5.9°, which bends the midline per disease instead of following the anatomy.

- ×4 macro 0.924432 against run 5's 0.924476; ×1 0.915933 against 0.915823.
- Per label against run 5: Fracture +0.0055, Synovitis +0.0021, Lateral OA −0.0023, PF OA −0.0013,
  Medial Meniscus −0.0012, the rest within ±0.0011.
- Learned angles: nod +2.50°, roll +3.52°, turn +3.06° (run 5 with soft cuts: −5.11°, −2.39°,
  −0.98°). The frame turned a few degrees with or without soft boundaries, and the signs were not
  stable.
- Knee-level strength: meniscus −0.119, OA +0.0002.
- Drawings on real slices: `outputs/analysis/o/octant-sketches-20261006/hard-octants-*.png`
  (two-frame model: `tilted-octants-*.png`).

S115, the two-frame model (`separate-oa-frame-with-knee-level-term`, ×4, τ 0.9): 0.953 public,
equal to S113 (run 4) and against 0.952 for S114 (run 5).

### Turning the trained frame by hand

`outputs/analysis/s/side-routing-rotation-sweep-20261006/sweep.csv`: the hard-octant model at ×1,
each angle moved −30° to +30° around its learned value, inference only. `cuts` turns only the
routing; `all` also turns the position prior and the own heads' position code.

- Within ±10° on any axis the macro stayed in 0.91551 to 0.91597 around 0.91589, and the routed
  sum within 0.005 of 4.5330.
- At ±30° the largest losses were nod −30° (PF OA 0.8897 to 0.8727) and roll +30° (Lateral
  Meniscus −0.006, Lateral OA −0.007).
- Roll towards −30° raised PF OA steadily (0.8897, 0.8912, 0.8927, 0.8939 at 0, −10, −20, −30) and
  lowered Medial OA by 0.002; routed sum +0.0037, macro +0.0003.
- No better angle away from the learned one beyond that. The heads were trained at the learned
  angle, so the sweep favours it.

## Soft gradient and three own-label routes (run 7)

On top of the hard-octant run, still 7×7 tokens and one frame, no shifts:

- **Soft gradient through the hard cuts.** The forward pass kept the hard octants; the gradient of
  each cut was that of `sigmoid(coordinate / token pitch)`, with the pitch measured per slice from
  the token positions (12 to 17 mm), so no width was set or learned. The frame's angles then
  received gradient from the routing of every routed label (first step: 0.0009, 0.0021, 0.0007
  against exactly zero before).
- **Own-label routes** (`OWN_ROUTES`): MCL from the four medial octants, Effusion from the two
  front-up octants, Baker's from the two medial-back octants. Their heads, tokens and priors were
  unchanged. Chosen from the hard-octant model's learned position prior, which already pointed
  there (Baker's medial-back +0.37/+0.28, Effusion front-up +0.28/+0.22, MCL medial +0.04 to
  +0.18); ACL, Contusion, Synovitis and Fracture stayed unrouted.

All nine guardrails passed. Run
`outputs/training/20261006-081303-side-routed-soft-gradient-own-routes`, same speed as before
(about 4 steps per second):

- ×4 macro 0.924845 against 0.924432 for the hard-octant run (+0.0004), 0.924476 for run 5 and
  0.924049 for the old recipe; ×1 0.917361 against 0.915933 (+0.0014). Highest of the line on
  both at the time.
- Per label against the hard-octant run: Contusion +0.0021, Fracture +0.0018, Medial OA +0.0015,
  Lateral Meniscus +0.0013, Lateral OA +0.0010, Medial Meniscus +0.0005, MCL +0.0005, Baker's
  +0.0003, Effusion −0.0002, PF OA −0.0004, ACL −0.0014, Synovitis −0.0020. The three newly routed
  labels did not move; the routed five gained 0.0039 summed.
- Learned angles: nod +0.85°, roll −0.63°, turn +2.11°. With the routing pulling on the frame it
  turned less than without (2.5°, 3.5°, 3.1°): the shared angles are one correction for every
  study, so per-knee differences cancel. This motivated a frame inferred per study.
- Knee-level strength: meniscus −0.118, OA +0.0006.
- S116 (×4, τ 0.9): 0.952 public, with S112 and S114; S113 and S115 were 0.953.

## A frame inferred per study (run 8)

On top of the soft-gradient build (hard octants, eight routed labels, 7×7 tokens):

- **Frame head** (`study_frames`): read the study's detached tokens plus the sinusoid code of their
  position in the fitted, header-aligned coordinates, attention-pooled them (256 → 128) and output
  six numbers per study: a centre shift along medial, front and up, and nod, roll and turn.
- The shift was in token pitches of that study, the angles kept the ±90° range. The output layer
  started at zero, so the first step was exactly the previous build (same loss, 0.874439). The
  shared angles and a shared shift (`frame_shift`) acted as the head's bias.
- Every use of position followed the inferred frame: routing, the attention prior and the own
  heads' position code.
- Trained only by the diagnosis loss through the soft gradient and the prior; no auxiliary loss,
  no gradient into the backbone.
- Logged per epoch, train and validation: `study_frame_{nod,roll,turn}_deg` and `_abs_deg`,
  `study_frame_shift_{medial,front,up}_mm` and `_abs_mm`. `frame` and `routing_weights` took the
  encoded batch.

All nine guardrails passed. The mirror guardrail held at the zero start; once trained, the
mirrored pass inferred its own frame from the mirrored image.

Run `outputs/training/20261006-084210-side-routed-study-frames`:

- ×4 macro 0.924320 against 0.924845 for run 7 (−0.0005); ×1 0.916573 against 0.917361
  (−0.0008), behind by 0.0005 to 0.0010 from epoch 3 on.
- Per label against run 7: Baker's +0.0040, PF OA +0.0017, Medial Meniscus +0.0004, Synovitis
  0.0000, Lateral OA −0.0002, Medial OA −0.0003, ACL −0.0004, Lateral Meniscus −0.0007, Effusion
  −0.0008, Contusion −0.0015, Fracture −0.0019, MCL −0.0066.
- Inferred frames over the validation knees at the last epoch, mean (mean absolute): nod +8.1°
  (8.4°), roll −2.2° (2.7°), turn −2.2° (3.3°); centre shift medial −1.0 (2.2) mm, front +1.3
  (1.4) mm, up −1.4 (1.9) mm. The shared part: nod +3.3°, roll −1.0°, turn 0.0°.
- The nod was common to all knees (same sign everywhere): the height cut dropped towards the front,
  so the front-up region reached the lower kneecap. Turn and the medial and up shifts differed in
  sign between knees. The values swung in the first epochs (roll −19° at epoch 2) and settled
  after 8.
- Four studies drawn before and after:
  `outputs/analysis/o/octant-sketches-20261006/study-frames-*.png` (rows 5, 9, 21, 45: nod +5.1° to
  +12.8°, turn −1.5° to −5.5°, shifts up to 4.6 mm).
- S117 (×4, τ 0.9): 0.952 public, with S112, S114 and S116; S113 and S115 were 0.953.

## Finer tokens, landmark frame, slower frame learning (run 9)

On top of the per-study frame build:

- **Finer tokens for the shared heads.** The meniscus and OA heads and the frame head read the
  stride-16 stage (`layer3`, 14×14, 196 tokens per slice, 6 to 8.6 mm apart) through their own
  projection and LayerNorm; the seven own heads, the side head and the report heads kept the 7×7
  stage. The stride-16 token positions came from the same verified ResNet mapping, cut at stage 3.
  The pooling ran once per grid and each target took its own grid's result.
- **Frame from landmarks.** The frame head produced three attention maps over a study's fine
  tokens (centre, front, up), starting uniform. Each landmark was the attention-weighted mean
  position; its displacement from the plain mean of the tokens moved the frame: the centre
  landmark's displacement was the centre shift, and the front and up axes were the unit axes
  scaled by the study's token spread plus the front and up landmarks' displacements relative to
  the centre's, made orthonormal. The shared angles still multiplied on top. Zero start: the
  first frame was the geometric centre and header alignment.
- **Slower frame.** Parameters named `frame_*` trained at the backbone learning rate (5e-5
  instead of 6e-4). Logged as before, with the angles read back from the rotation matrix.

All nine guardrails passed; peak memory on four studies 1.8 GB.

Run `outputs/training/20261006-092326-side-routed-fine-landmarks`, about two minutes per epoch:

- ×4 macro 0.922842 against 0.924845 for run 7 (−0.0020) and 0.924049 for the old recipe; ×1
  0.915148 against 0.917361 (−0.0022). Lowest of the line.
- The loss was in the shared heads on the stride-16 tokens: Lateral OA −0.0155, PF OA −0.0126,
  Medial Meniscus −0.0042, Medial OA +0.0010, Lateral Meniscus +0.0012 (routed five −0.0301
  summed). The seven own labels gained 0.0061 summed (Synovitis +0.0037, Fracture +0.0035,
  Effusion +0.0013, Baker's +0.0011, MCL −0.0002, ACL −0.0010, Contusion −0.0023).
- The frame was not the cause. Resetting the landmark maps to flat at inference (the starting
  frame) gave ×1 macro 0.91458 against 0.91509 with the learned frame, routed five 4.5050 against
  4.5071.
- The landmark maps stayed flat at the backbone learning rate: each landmark effectively used
  99.7% to 99.9% of a study's tokens, score range 0.3 to 0.45 logits, weight norm 0.03 to 0.05.
  Over 150 validation knees the frame was one common rotation, not per knee: nod 0.4° (SD 0.3),
  roll 2.6° (SD 0.4), turn 4.1° (SD 0.3), shifts under 0.5 mm (SD 0.1).
- Lessons: stride-16 features alone are too shallow for the OA head, which the meniscus head
  tolerates better; a zero-started head at 5e-5 does not move in ten epochs.

## Deep features on the fine tokens, frame at head rate, routed v4 report targets (run 10)

On top of the fine-tokens and landmark build:

- **Deep features on the fine tokens.** The stride-16 tokens read by the shared heads and the
  frame head were their own projection plus the stride-32 projection, linearly interpolated to the
  fine token anchors (even anchors coincide with a coarse one, odd ones take the mean of two).
- **Frame parameters back at the head learning rate** (6e-4).
- **Report attributes: target set 4.0** (`configs/labels/jevk5-report-attribute-targets-v4.0.json`,
  37 questions, brought from the worktree `knee-abnormality-jevk5-attributes-v4` with its question
  set and recipe; the spec of the next extraction is that worktree's `attributes_v5.md`), **routed as "4.1" at training time** (`training/report_attributes.py`):
  - Ten compartment questions shared one head per family and were routed like their label:
    meniscus status and tear type by side; cartilage depth and extent by medial rest, lateral rest
    and front-and-up. They read the position-free fine tokens.
  - The three location questions became nine yes/no region targets on shared heads instead of
    whole-knee "where" heads: meniscus tear in the front or back octants of each side, marrow
    edema in medial rest, lateral rest or front-and-up, patellar facet in the medial or lateral
    half of front-and-up. A region's target was the summed probability of the options placing the
    finding there; its weight was the question's weight times one minus the mass on options that
    state no place.
  - The other 23 questions kept their own heads on the 7×7 tokens.
  - Five routes added: medial-front, lateral-front, lateral-back, front-and-up medial and lateral.
    All of these pulled on the frame through the soft gradient.
- Region targets over the 4,407 studies (target > 0.5): meniscus tear medial front 196, medial
  back 1,343, lateral front 277, lateral back 398; marrow edema medial 1,207, lateral 790,
  front-and-up 118; patellar facet medial 591, lateral 517.
- Logged: `report_attribute_bce_<family>_<route>` per region, beside the per-question KL.

All nine guardrails passed; peak memory on four studies 2.0 GB.

Run `outputs/training/20261006-095814-side-routed-fine-fused-routed-reports`:

- ×4 macro 0.924423 against 0.924845 for run 7 (−0.0004); ×1 0.916674 against 0.917361
  (−0.0007). Level within run-to-run noise; not above.
- Per label against run 7: PF OA +0.0018, Synovitis +0.0019, Baker's +0.0018, Effusion +0.0009,
  Medial Meniscus +0.0003, ACL +0.0001, Medial OA −0.0004, Fracture −0.0008, Lateral OA −0.0016,
  Contusion −0.0017, Lateral Meniscus −0.0033, MCL −0.0040. Routed five −0.0032 summed, against
  −0.0301 for the fine tokens without deep features: the deep features removed that loss, and
  the fine grid as a whole bought nothing on the score.
- **The frames were per knee.** Over 150 validation knees, mean (SD, range): nod 10.6° (2.8, 7
  to 28), roll −1.1° (5.2, −19 to 12), turn −2.2° (2.2, −12 to 3); centre shift medial −1.1 mm
  (2.6, −8.6 to 5.6), front +1.2 mm (2.1), up +1.0 mm (2.5, −7.3 to 7.2). The centre and front
  landmark maps used 79% and 82% of a study's tokens (score range 2.5 logits); the up map 98%.
  Shared part: nod 4.8°.
- **Report targets (training set, last epoch), BCE explained against the base rate:** meniscus
  tear medial back 34%, lateral front 28%, lateral back 27%; marrow edema lateral 30%, medial 29%,
  front-and-up 3%; patellar facet lateral 19%, medial 15%; meniscus tear medial front low. The 33
  question heads all improved (KL −13% to −45% from the first epoch), the shared routed ones in
  the same range as the others.
- **Moving the trained centre by hand** (`outputs/analysis/s/side-routing-centre-sweep-20261006/`,
  ×1, −20 to +15 mm): the learned position was at the top. PF OA 0.8915 at the learned centre,
  0.8925 at 5 mm lower, 0.8883 at 20 mm lower, 0.8737 at 15 mm higher; macro 0.91671 at the
  learned centre and lower everywhere else, on both the up and the front axis. So the frame was
  not missing a better place for PF OA; the nod of about 10° already lowered the height cut at
  the front.
- Drawings: `outputs/analysis/o/octant-sketches-20261006/side-routed-fine-fused-routed-reports-*.png`.
- Lesson: the frame could be placed per knee and sat at a local optimum, and the score did not
  follow. Inside its region each label's attention already selects the tokens that matter, so a
  more exact region changes little. The fine grid also gave back the bag reduction the octants
  bought (2× the original tokens for half-knee labels).

## Run 8 plus landmarks (run 11)

The worktree code was restored from run 8's source snapshot (7×7 tokens for every head, v3.1
report targets on unrouted heads, the regressed per-study frame) and got the landmark frame on
top:

- The frame head kept its attention-pooled regression of six numbers per study (shift in token
  pitches, three angles) and gained three zero-started landmark maps over the same 7×7 tokens.
- The two composed: the centre shift was the centre landmark's displacement plus the regressed
  shift; the rotation was the regressed rotation applied to the landmark axes.
- Zero start: the first step's loss equalled run 8's (0.874440).
- The fine grid, its projection, the routed report heads and the v4 targets were left out; they
  are in run 10's source snapshot and `scratchpad/run10-build`. The v4 manifest, question set
  and recipe stayed in `configs/labels/`, unused by this run.

All nine guardrails passed. Run `outputs/training/20261006-103317-side-routed-study-frames-landmarks`:

- ×4 macro 0.924119 against 0.924845 for run 7 (−0.0007) and 0.924320 for run 8 (−0.0002); ×1
  0.916769 against 0.917361 (−0.0006).
- Per label against run 7: Synovitis +0.0019, PF OA +0.0016, Contusion +0.0011, ACL +0.0009,
  Baker's +0.0009, Effusion +0.0002, Medial Meniscus 0.0000, Medial OA −0.0007, Lateral OA
  −0.0008, Lateral Meniscus −0.0014, Fracture −0.0035, MCL −0.0088.
- Frames over all validation knees, mean (SD, range): nod 10.9° (5.2, −5 to 25), roll 0.2° (5.2,
  −13 to 18), turn 3.3° (6.4, −17 to 19); shift medial −0.2 mm (2.6, −10 to 6), front +0.7 mm
  (2.2), up +0.8 mm (3.2, −6 to 10). The widest per-knee spread of any run.
- Frame reset at inference (×1): learned frame 0.91673, starting frame 0.91614 (−0.0006),
  starting centre with learned angles 0.91633, learned centre with starting angles 0.91642. With
  the learned frame PF OA was +0.0025, Contusion +0.0028 and MCL +0.0027 above the starting frame,
  and Fracture −0.0025. So inside this model the frame helped; MCL was low with either frame,
  which makes its loss a training effect.

Across the four runs with a per-knee frame (8, 9, 10, 11) against run 7: macro −0.0002 to −0.0007
(run 9 aside), PF OA +0.0016 to +0.0018 in the three where the frame learned, MCL −0.0040 to
−0.0088 in the same three, Fracture −0.0008 to −0.0035.
