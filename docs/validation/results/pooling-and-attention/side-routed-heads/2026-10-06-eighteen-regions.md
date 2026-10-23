# Side-routed heads: eighteen regions, runs 12 to 17 (2026-10-06)

**Run 11's attention, binned by height, core and quadrant, gave a region table; the eighteen-region
build with per-knee sizes and overlap assignment (runs 12 to 16) scored ×4 0.923687 to 0.924172,
inside the line's spread.** The yardstick check after run 15 (mean 0.92425, SD 0.0003 over ten
runs since the fifth) showed that none of the changes since the fifth run had moved the fold-1
macro; Contusion, Fracture and MCL losses against run 7 were systematic to the region builds.
Run 17, the ResNet-50, was stopped. Fold 1 of `all_5fold_scanner_v2`, seed 1110484336, ResNet-18.
Design: [side-routed heads](../../../../modeling/side-routed-heads.md).

## Where each label's attention falls (2026-10-06)

`outputs/analysis/s/side-routing-attention-by-region-20261006/attention.json`: run 11's model, the
874 validation knees with a fitted centre, attention of each label on its positive knees, binned
by the fifteen regions of the proposed division (four quadrants plus a core of 14 mm radius; joint
band ±10 mm around 14 mm below the geometric centre). Geometric coordinates, 7×7 tokens. Routed
labels were confined to their octants in this model, so for them only the height and core split
was free.

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

All-token attention includes a long diffuse tail. Taking only each knee's three
highest-attention tokens per label (they hold 4% to 18% of the attention; ten tokens hold 10% to
46%) changes the picture. Token share of the bands: femur 51%, joint 22%, tibia 27%; core 9%.

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

So the decisive tokens support the three heights and the core: the joint band (22% of tokens)
holds 59% to 65% of the top tokens of the menisci and the femorotibial OA labels; the menisci's
top tokens avoid the core and the tibia; ACL's sit in the core; Synovitis's sit in the femur band.
The femur share of the menisci (34% to 39%, below the band's token share) is consistent with a
joint band of ±10 mm being thinner than the tokens.

## Build: eighteen regions, everything about the frame learned (run 12)

On top of run 11 (7×7 tokens, per-study frame from landmarks plus regression):

- **Regions.** Around the per-study frame: four quadrants (medial/lateral × front/back) and a core
  cylinder split at the midline, at three heights (femur, joint, tibia): 18 regions.
- **Nine numbers per study.** The frame head regressed three more: the upper cut, the lower cut
  and the core radius, in token pitches of the study, on top of shared biases (`frame_size`).
  They started at 0, −2 and 1 pitches (`FRAME_SIZE_START_PITCHES`): the upper cut at the centre,
  the lower cut about 27 mm below it, the core about 14 mm in radius. With the geometric centre
  about 15 mm above the joint line that put the joint band around the joint line.
- **Assignment by overlap.** A token's weight for a cut was 0.5 plus its signed distance over
  twice its half-extent along the cut's normal, clamped to 0..1. The half-extent came from the
  token's box: its two in-plane steps and the step to the next slice, projected on the frame
  axes. So a token inside counted 1, a straddling one its share, and a thin slice stack was cut
  sharply along its normal. This replaced the hard cuts with a soft gradient.
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
  prior of own targets); routing and the shared targets' prior followed the inferred frame.
- **Logged:** `study_frame_{band_top,band_bottom,core_radius}_mm`, and on the validation set
  `val_report_attribute_fit` and `val_report_attribute_bce_<region>`.
- Guardrail change: the meniscus pair may not overlap and every weight lies in 0..1; the shared
  channels no longer partitioned every token (the menisci left the core and the other bands, and
  the core of the femur band was read by PF OA and both OA sides).

First step on four studies: token share per route 15% for a meniscus (50% before), 7% for the
core, 30% for the kneecap part, 47% for the femur band; all three sizes received gradient. All
nine guardrails passed.

Run `outputs/training/20261006-112944-side-routed-eighteen-regions` (run 12):

- ×4 macro 0.923687 against 0.924845 for run 7 (−0.0012, three sibling SDs); ×1 0.915856 against
  0.917361 (−0.0015).
- Per label against run 7 (sibling SD in brackets): Baker's +0.0020 (0.0015), Lateral OA +0.0017
  (0.0034), Lateral Meniscus +0.0016 (0.0024), PF OA +0.0007, Effusion −0.0001, Medial OA
  −0.0002, ACL −0.0008 (0.0014), Medial Meniscus −0.0008, Synovitis −0.0019 (0.0022), Contusion
  −0.0045 (0.0013), MCL −0.0046 (0.0029), Fracture −0.0071 (0.0044).
- The routed five were +0.0030 summed while reading 15% to 38% of the tokens; ACL held on the core
  alone (7% of the tokens). The loss was in Contusion and Fracture, which were unrouted and in this
  build did not depend on the frame at all, and in MCL.
- Learned sizes over the validation knees: upper cut +4.9 mm, lower cut −35.2 mm (the joint band
  widened from 27 to 40 mm), core radius 15.0 mm. Nod +20.1° in every knee, roll 8.6° and turn
  3.8° in absolute value, centre shifts 2.4 to 3.2 mm.
- **Report region targets on the validation set** (BCE explained against the base rate, last
  epoch): meniscus tear medial back 33%, lateral back 22%, lateral front 17%, medial front −9%;
  marrow edema medial 25%, lateral 25%, kneecap 2%; patellar facet lateral 20%, medial 15%. The
  validation fit improved through training (0.763 to 0.719). The regions held on unseen knees.
- Contusion's supervision changed with the v4 targets (its status and location questions became
  marrow edema of any cause, and the location became region targets), and it lost 0.0017 in run
  10 as well, so run 13 isolated the reports.

### Run 13: the eighteen regions with the v3.1 reports, unrouted

Run `outputs/training/20261006-115457-side-routed-regions-v31-reports`: run 12's build with the
report heads and targets of run 11 (v3.1, 35 questions, unrouted). The first run of the
unattended loop the user asked for on 2026-10-06 13:30.

- ×4 macro 0.924122 against 0.924845 for run 7 (−0.0007) and 0.923687 for run 12 (+0.0004); ×1
  0.916925 against 0.917361 (−0.0004).
- Per label against run 7: Lateral OA +0.0031, Lateral Meniscus +0.0021, Baker's +0.0016, PF OA
  +0.0007, Effusion +0.0006, Medial OA −0.0003, Medial Meniscus −0.0004, Synovitis −0.0005, ACL
  −0.0009, MCL −0.0033, Contusion −0.0042, Fracture −0.0072. Routed five +0.0052 summed.
- Against run 12 every label was within ±0.0015: Contusion +0.0003, Fracture −0.0001. So the v4
  routed reports did not cause the Contusion and Fracture losses, and on this fold the two report
  setups were level.
- Learned sizes: upper cut +3.4 mm, lower cut −34.2 mm, core radius 15.6 mm, nod +15.5°.
- Contusion and Fracture lost the same amount in both runs (−0.0045/−0.0042 and
  −0.0071/−0.0072), so it was systematic to the eighteen-region build and not noise. They were
  unrouted and read the fixed frame, as in run 7, so the cause was indirect. Leading candidate:
  ACL read only the core, while unrouted it put 43% of its attention on the lateral quadrants,
  where the bruises and fractures that accompany its tears are; that trained the backbone on
  them. Run 14 tested it.

### Run 14: ACL unrouted

Run `outputs/training/20261006-121814-side-routed-regions-acl-unrouted`: run 13 with ACL reading
the whole knee again.

- ×4 macro 0.923936 against 0.924845 for run 7 (−0.0009) and 0.924122 for run 13 (−0.0002); ×1
  0.915416.
- Against run 13: Fracture +0.0027, Contusion +0.0018, ACL +0.0016, PF OA +0.0011, Lateral
  Meniscus −0.0015, Lateral OA −0.0018, Effusion −0.0019, MCL −0.0048. Against run 7: Contusion
  −0.0024, Fracture −0.0044, MCL −0.0080.
- Contusion and Fracture moved in the predicted direction and recovered about a third of their
  loss (1.4 and 0.6 sibling SDs); the macro did not move. Weak support for the ACL explanation.
- Across the six runs with a per-knee frame head that learned (8, 10, 11, 12, 13, 14) MCL was
  below run 7 every time (−0.0066, −0.0040, −0.0088, −0.0046, −0.0033, −0.0080; sibling SD
  0.0029) and Fracture every time (−0.0019, −0.0008, −0.0035, −0.0071, −0.0072, −0.0044), while
  the region table gave the routed five +0.0030 to +0.0052. Run 15 separated the two.

### Run 15: the eighteen regions on one shared frame

Run `outputs/training/20261006-124148-side-routed-regions-shared-frame`: run 14 without the
per-knee frame head; one learned centre shift, rotation and set of sizes for all knees.

- ×4 macro 0.924141 against 0.924845 for run 7 (−0.0007) and 0.923936 for run 14 (+0.0002); ×1
  0.915650.
- Per label against run 7: Baker's +0.0016, Lateral Meniscus +0.0009, ACL +0.0008, Medial OA
  +0.0008, Medial Meniscus +0.0004, PF OA +0.0003, Lateral OA −0.0010, Synovitis −0.0010,
  Effusion −0.0015, Contusion −0.0022, Fracture −0.0029, MCL −0.0047.
- The shared frame stayed where it started: upper cut −0.1 mm, lower cut −29.0 mm, core radius
  13.7 mm, nod 0.4°, roll −2.0°, turn 1.4°, shifts under 0.7 mm.
- MCL and Fracture were below run 7 here too, without any per-knee head. So the per-knee head was
  not what lowered them.

**Yardstick check.** The ×4 macros of every run of the line since the fifth were 0.924476,
0.924432, 0.924845, 0.924320, 0.924423, 0.924119, 0.923687, 0.924122, 0.923936, 0.924141 (run 9
left out): mean 0.92425, SD 0.0003, the same as the sibling SD of the old recipe (0.0004). Run 7
was the top of that spread and run 12 the bottom. Measured against the spread and not against the
best draw, none of the changes since the fifth run moved the fold-1 macro: hard cuts, soft
gradient, per-knee frame, landmarks, fine tokens with deep features, eighteen regions, learned
sizes, overlap, routed v4 reports. MCL and Fracture being "below run 7" in every later run says as
much about run 7's draw on those two labels as about the later runs. What did move, beyond
single-label noise and in every run that had it: PF OA with a per-knee frame (+0.0016 to
+0.0018, three runs), and the report region targets fitting on the validation set (run 12).

### Run 16: position for the shared heads, in compartment-relative coordinates

Run `outputs/training/20261006-130551-side-routed-regions-mirrored-position`: run 14 with the
shared meniscus and OA heads reading each token plus the fixed sinusoid of its frame position,
the medial axis as is for a medial target, mirrored for a lateral one and in absolute value for
PF OA and the knee-level term. The guardrail "shared tokens are position-free" became "shared
tokens mirror between sides"; the side-flip check still passed (mismatch 3e-8).

- ×4 macro 0.924172 against 0.923936 for run 14 (+0.0002) and 0.924845 for run 7; ×1 0.915906.
- Against run 14: MCL +0.0033, ACL +0.0031, Lateral OA +0.0030, Fracture +0.0021, Effusion
  +0.0013, Medial Meniscus +0.0003, Lateral Meniscus −0.0004, Medial OA −0.0005, Contusion
  −0.0010, Baker's −0.0013, PF OA −0.0024, Synovitis −0.0045. The five shared-head labels summed
  to 0.0000.
- Position did not help the shared heads inside their regions; inside the noise of the fold.

### Run 17: the agreed design on the Refined ResNet-50, stopped

With ten ResNet-18 variants inside one noise band on fold 1, the loop moved to a change large
enough to read: the same design on the backbone of the best public result. Build: run 12's
(eighteen regions, per-knee frame with learned sizes, overlap assignment, routed
v4 reports, own position in the fixed frame) with ACL unrouted as in run 14, on
`configs/refined-resnet50.yaml` (removed) with `--validation.folds [1]`, seed 1533697213. All nine
guardrails passed with this config. Loose reference, not a control:
`20261003-122711-refined-rn50-views-label-bias-folds-1-4`, fold 1, same seed, the old whole-knee
recipe of 2026-10-03 (ImageNet normalization, half-knee loss 0.075): ×4 0.929567, ×1 0.921679.

The user stopped it three minutes into its first epoch (2026-10-06 15:33): moving to the
ResNet-50 had been the assistant's own decision, outside what the loop was for. Nothing was kept;
its partial folder is `outputs/training/20261006-132943-rn50-side-routed-regions`.
