# Frame-and-targets: keeper's run notes

Written by the keeper after each result: both multipliers against the incumbent, the per-target
table, the geometry, repairs and adoption decisions.

## Started 2026-10-07 13:47 CEST (Codex)

Base run 24 (×4 0.925067, ×1 0.917070). Eleven forced seeds as first written.

## Queue refocused on the user's direction (2026-10-07 14:05)

The user restated the campaign's direction after the first seed had started: the frame's tilt
may be wrong because there are two tilts, one for the tibia and one for the femur; the v5
extraction was run to get a landmark on each bone; once the frame has both, the crops are turned
and moved by it. Seeds outside that were removed, and the user allowed seeds to build on the
previous campaign's incumbent (the height cuts placed from the spots).

- **Removed, not run:** `kneecap-region-on-the-kneecap-spot`,
  `extensor-tendon-targets-above-and-below-the-kneecap`, `acl-on-its-unrouted-head`,
  `five-unplaced-report-questions-without-heads`; and the two frame seeds on run 24's learned
  cuts, `femur-section-leaning-to-the-femur-spots` and `tibia-and-femur-sections-on-their-own-axes`,
  replaced by the versions below.
- **Could not be removed:** `forward-tilt-from-the-meniscus-front-and-back-spots` was already
  training and `frame-read-from-the-spots` already built. Both rest on a tilt read from the front
  and back of each meniscus, whose premise (that the meniscus head looks at both ends of each
  meniscus) was never checked; they run for information and their GPU time overlaps the
  developer's work on the next seed.
- **Queue from there:** `edema-bone-targets-with-the-head-in-the-model` (the femur and tibia
  spots, reported with and without edema, nothing reading them yet);
  `femur-section-on-spot-placed-cuts`; `tibia-and-femur-tilts-on-spot-placed-cuts` (the direction
  itself); then the three crop seeds.
- **Why the two frame seeds build on the spot-placed cuts:** the femur section hinges at the
  upper cut. In run 24 the cuts are regressed from the labels (+13.7 and −1.8 mm) and a learned
  cut can slide to trade against a new angle, as the learned tilt did against the learned lean.
  Placed from the spots they held at +10.8 and −12.9 mm.

## After every frame experiment (user instruction, 2026-10-07 14:10)

The keeper draws the run's frame on the same sample knees (validation rows 532 and 200, bent by
eye; 370 and 374, straight; 45 and 5), with the spots and each section's axis on the slices,
looks at the drawings, and evaluates them with the scores and the logged geometry. The evaluation
is written here and decides the queue: seeds that are no longer worth running are removed, and
seeds the result calls for are added.

## 1. `forward-tilt-from-the-meniscus-front-and-back-spots`: not adopted (2026-10-07 14:20)

- ×4 0.924060 (−0.0010), ×1 0.916607 (−0.0005) against run 24. Below at both.
- Per target at ×4: ACL +0.0035 (2.5 sibling SDs), Synovitis −0.0091 (4), Lateral OA −0.0039,
  Medial OA −0.0022 (2.7), Fracture −0.0018, the rest within 0.0009.
- Geometry, validation, last epoch: measured tilt 14.6° with a spread of 10.0° across knees,
  measured in 99% of knees. The frame's forward angle ends at −8.7° on average (run 24: +13.6°), so
  the measured plateau line and the tilt run 24 learned differ by about 22°; the shared learned
  rotation has moved the frame about 6° back toward the learned side. Upper cut +15.5 mm, lower cut
  +3.9 mm (run 24: +13.7, −1.8): both learned cuts now lie above the joint level.
- **Drawings** (`outputs/analysis/f/frame-and-targets/forward-tilt-from-the-meniscus-front-and-back-spots/frames.png`,
  rows 532, 200, 370, 374, 45, 5): in all six knees the whole-meniscus spot, its front spot and
  its back spot lie on the joint line, about 10 mm apart on each side, and the frame's joint level
  follows the visible joint space in both planes, including the knee whose plateau slopes 14°
  (row 374). The meniscus attention is spread along each meniscus (no token holds more than 8% of
  a region's squared weight), not concentrated at the back. The premise doubted before the run
  holds on these knees.
- **What is wrong in the drawings is the cuts, not the tilt:** the lower cut sits 2 to 6 mm above
  the joint level, so the joint band is a 10 to 13 mm slab through the condyles and everything
  from the joint line down is the tibia band. Synovitis, which reads above the lower cut, loses
  most.
- **Evaluation.** The tilt read from the menisci follows the tibial plateau; the tilt run 24
  learns does not (22° away), which fits the user's reading that one learned tilt was standing for
  something other than the tibia. With the tilt fixed to the plateau the learned cuts moved to
  compensate. Consequences for the queue: `frame-read-from-the-spots` (this tilt with the cuts
  placed from the spots) is now the direct test of that and stays; the two-tilts seed keeps its
  tibia tilt from the tibia spots until the bone-head run shows where those spots fall, and a
  variant with this plateau tilt for the tibia is added if they fall badly.

## Repair: the bone-head seed and its two followers parked by a requirement of mine (2026-10-07 14:45)

`edema-bone-targets-with-the-head-in-the-model` was built completely (four targets, 399 / 282 /
490 / 328 training studies above 0.5 for femur medial, femur lateral, tibia medial, tibia lateral;
the head in the model; the spots; guardrails passed) and then parked at review because the seed's
text demanded that a mirrored pass give mirror-image spots, and they were 5.2 mm apart on the one
study checked. The demand was wrong: a mirrored image is a different picture to the backbone.
`femur-section-on-spot-placed-cuts` was parked for importing the same mechanism. Requeued as
`edema-bone-targets-with-the-head-in-the-model-2`, `femur-section-on-spot-placed-cuts-2` and
`tibia-and-femur-tilts-on-spot-placed-cuts-2`
([bone-head requeue](bone-head-requeue-ideas.yaml), [direction requeues](direction-requeue-ideas.yaml))
with the requirement withdrawn and the parked workspace named for reuse. The GPU idled about
fifteen minutes.

## 2. `frame-read-from-the-spots`: not adopted (2026-10-07 14:42)

- ×4 0.923697 (−0.0014), ×1 0.915767 (−0.0013) against run 24.
- Per target at ×4: MCL +0.0028, Lateral Meniscus +0.0012; Lateral OA −0.0073 (2 sibling SDs),
  Synovitis −0.0037, Fracture −0.0030, ACL −0.0018, Medial OA −0.0017 (2 SDs).
- Geometry, validation, last epoch: measured tilt 20.8° (spread 10.2°), against 14.6° in run 1
  with the same tilt rule; the frame's forward angle −15.8° (run 24 +13.6°); cuts from the spots
  +7.6 and −8.5 mm (the previous campaign's model: +10.8, −12.9); core 14.0 mm, ring 25.7 mm.
- **Drawings** (`outputs/analysis/f/frame-and-targets/frame-read-from-the-spots/frames.png`): the
  two cuts now bracket the joint, as intended. The tilt overshoots: in rows 374 (−26°), 5 (−11°)
  and 200 (−8°) the frame's joint level is steeper than the visible joint line, with the front
  spot sitting above the meniscus on the condyle and the back spot below it. In run 1 the same
  knees read −14°, −5° and 0° and followed the joint.
- **Evaluation.** The tilt read from the heads' own attention is not an anatomical constant: it
  rises over the first five epochs in both runs (9° to 15°, 14° to 21°) and settles 6° apart
  depending on what else in the frame is read from spots. The heads' attention is trained through
  the regions the frame defines, and the frame is read from that attention: a closed loop with
  nothing outside it holding the angle. It does not run away, but it is not a measurement of the
  plateau either. Reading more of the frame from label-trained attention makes it worse, not
  better (this run against run 1, and both against run 24).
- **For the queue:** this is the argument for the direction's version, where the landmarks come
  from the report-supervised bone head rather than from the label-trained heads. That head is
  pooled over routed regions too, so the same loop is possible: the bone-head run's report of
  where its spots fall, with and without edema, and how they move across epochs is the thing to
  read before the two-tilts run. If they drift, the two frame seeds are requeued with the bone
  targets pooled over regions fixed in the scan's coordinates.

## 3. `edema-bone-targets-with-the-head-in-the-model-2`: not adopted; the spots mark the bones, not their directions (2026-10-07 15:12)

- ×4 0.924495 (−0.0006), ×1 0.917017 (−0.00005) against run 24. Below at both, by a rerun's width.
- Per target at ×4: Lateral Meniscus +0.0015, ACL +0.0010, Contusion +0.0010; Fracture −0.0046,
  Synovitis −0.0032, Lateral OA −0.0026; the rest within 0.0005.
- **The targets are learnable:** on the validation knees the head explains 27% (femur medial), 29%
  (femur lateral), 29% (tibia medial) and 31% (tibia lateral), against a bar of 10%.
- **Where the spots fall** (all 874 validation knees, relative to the meniscus midpoint): femur
  spots 15.0 mm up (SD 3.7) and 13.2 mm forward (SD 7.1); tibia spots 11.2 mm down (SD 3.1) and
  2.3 mm forward (SD 4.0). Found in every knee. Knees whose report has edema (392) and knees
  without (482) give the same places within a millimetre or two, so the head looks at the marrow
  of each bone whether or not there is edema.
- **Drawings** (`outputs/analysis/f/frame-and-targets/edema-bone-targets-with-the-head-in-the-model-2/bone-spots.png`):
  in all six knees the femur spot lies in the femoral condyle and the tibia spot in the plateau
  just under the joint line. The head tells the two bones apart.
- **But the spots sit next to the joint, and a line through them is not the bone's direction.** The
  line from the menisci to the femur spots leans +38° on average (SD 13), the line from the tibia
  spots to the menisci −12° (SD 19), and their difference, the would-be bend, is 51° on average
  with a spread of 28°, against 0° to 30° by eye. It does not sort the six knees: 52° on a knee
  judged straight (row 374), 2° on the other straight one, 38° and 48° on the two bent ones, 77°
  on row 5. With a lever arm of 11 to 15 mm, where on the condyle the attention sits decides the
  angle. The condyle is round and the femur turns about it when the knee bends, so a landmark
  there cannot show the shaft's direction.
- **A second spot further up each bone, same weights** (`far-bone-spots-18-22.csv`: the head's
  squared-attention centroid within 18 mm of the menisci and beyond 22 mm, giving a lever of about
  21 mm): femur axis 27° on average (SD 18), tibia axis 9° (SD 16), bend 18° (SD 25, 5th to 95th
  percentile −25° to +55°). The mean is plausible; the spread is not, and it reads 2° on one bent
  knee and 64° on row 45. The head puts 18% of its attention beyond 22 mm in the femur and 7% in
  the tibia, so the far spots rest on little.
- **Evaluation.** The report-supervised head gives a real landmark for each bone's end, which
  nothing else in the model had, and no usable direction. Two tilts read from these spots would be
  set by noise. Queue: `tibia-and-femur-tilts-on-spot-placed-cuts-2` removed before running.
  `femur-section-on-spot-placed-cuts-2` was already built and launched on the resume; it cannot be
  removed and runs for information (its lean, from a femur spot 4 mm above the upper cut, will be
  large and noisy). The three crop seeds follow.

## 4. `femur-section-on-spot-placed-cuts-2`: not adopted, as predicted from the spots (2026-10-07 15:38)

- ×4 0.921832 (−0.0032), ×1 0.912677 (−0.0044) against run 24: the largest loss of the campaign.
- Per target at ×4: Lateral OA −0.0092, Synovitis −0.0075, PF OA −0.0064 (twelve sibling SDs),
  MCL −0.0044, Fracture −0.0041; only Contusion above the incumbent (+0.0009).
- Geometry, validation, last epoch: femur lean 60° with a spread of 27° across knees, applied in
  85% of knees; the frame's learned forward tilt went from −17° at epoch 1 to −38°; cuts from the
  spots +2.6 and −5.8 mm, drifting inwards over training.
- **Drawings** (`outputs/analysis/f/frame-and-targets/femur-section-on-spot-placed-cuts-2/bone-spots.png`):
  in every knee the two cuts run diagonally across the joint at 35° to 40°, through femur and
  tibia alike. The femur spot sits a few millimetres above the upper cut, so the lean it gives is
  near its limit for most knees, and the labels turned the whole frame the other way to
  compensate: the same trade as the learned lean of 2026-10-07 morning, larger.
- **Evaluation.** Confirms run 3's reading on a trained model: a landmark in the condyle cannot set
  the femur section's direction. Nothing in the queue depends on it. The frame topic now needs a
  landmark on the shaft or another source for the bend; the crop seeds run on run 24's frame.

## 5. `crops-turned-to-the-frame`: best ×4 of the line, not adopted on ×1; submitted as S121 at the user's request (2026-10-07 16:01)

- ×4 0.926581 (+0.0015), ×1 0.916478 (−0.0006) against run 24. Above at ×4, below at ×1: not
  adopted by the two-multiplier rule (the runtime's own verdict is a provisional accept).
- Per target at ×4: ACL +0.0055 (4 sibling SDs), MCL +0.0040, Contusion +0.0040 (3 SDs), Lateral
  OA +0.0022, Fracture +0.0019, Lateral Meniscus +0.0013, Effusion +0.0012; Synovitis −0.0030;
  the rest within 0.0007.
- Time: training 1,418 s (1.16 times run 24's 1,227 s, limit 1,595 s); validation prediction
  40.9 s against 33.1 s (1.23 times, limit 1.3).
- Angles applied (half the frame's angle): on the validation knees the sagittal crops are turned
  −4.2° on average (spread 2.6°) at the end, the coronal crops 2.1° in size. Training turns by
  the frame remembered from the last epoch and validation by the current one; while the learned
  tilt swung in the first epochs they differed (−1.9° against −5.0° at epoch 3, where the
  validation macro fell 0.0046 behind before recovering), and from epoch 8 they agree within a
  degree. The frame's learned tilt ends at 9.9° (run 24: 13.6°).
- The user asked why the remembered frame is used in training; a keeper's seed for the frame of
  the same step was imported and withdrawn within two minutes on the user's instruction.
- Submitted as S121 (bundle `outputs/submission/exports/side-routed-crops-turned-f1-x4-tau09`,
  ×4, τ 0.9, batch 4), exported from the run's frozen source.

## Queue after the user's instruction of 2026-10-07 16:00

"We cannot move forward to crops and shifts until the double cylinder is done": the last crop seed
(`smaller-rotation-augmentation-on-turned-crops`) was removed; `crops-turned-and-centred-on-the-frame`
was already built and runs while the developer prepares the double cylinder. Queued for the
double cylinder: `double-cylinder-with-a-learned-femur-lean`
([seed](double-cylinder-ideas.yaml)), `double-cylinder-with-one-shared-bend` and
`tendon-targets-with-the-head-in-the-model` ([seeds 2](double-cylinder-ideas-2.yaml)). A research
subagent was sent to find further ways to obtain the femur's axis; its ideas are queued when it
reports.

## Repair: the runtime's own acceptance undone, and the last crop seed retired (2026-10-07 16:10)

- The supervisor made `crops-turned-to-the-frame` the incumbent on its shared verdict (×4 alone).
  The contract adopts only at both multipliers and its ×1 is 0.0006 below run 24, so the keeper
  stopped the campaign, set the incumbent back to the historical control in `state.json` and
  marked the run rejected (backup `state.before-reverting-the-automatic-acceptance-20261007.json`;
  decision recorded under `outputs/autoresearch/frame-and-targets/decisions/`). Under this
  campaign the runtime accepts on its own, unlike the previous one where every run fell short of
  its interval: the keeper checks the incumbent after every run.
- `crops-turned-and-centred-on-the-frame` was with the developer and could not be removed by
  command; with the campaign stopped the keeper retired it in `state.json` (backup
  `state.before-retiring-the-centred-crop-seed-20261007.json`), on the user's instruction that no
  crop seed runs before the double cylinder is right.
- Queue: `double-cylinder-with-a-learned-femur-lean`, `double-cylinder-with-one-shared-bend`,
  `tendon-targets-with-the-head-in-the-model`.

## S121 and the research review (2026-10-07 16:25)

- **S121, `crops-turned-to-the-frame`: 0.952 public**, against 0.953 for run 24 (S118) and 0.952
  for the height-cuts model (S120). Its +0.0015 at ×4 locally did not show on the public board;
  its ×1 was 0.0006 below.
- **Research review of the double cylinder** (a read-only subagent over the notes, code,
  drawings and outputs): two attention spots cannot give a direction at this token size (4 to 6
  mm of noise per spot; the three femur estimates correlate 0.07, 0.02 and −0.24 with each other;
  5° of precision would need a 70 mm lever); the femur spot is the hinge, not a direction; the
  −25° learned lean matches the line from the PF OA spot to the Effusion spot (−23°), soft tissue
  rather than bone; the nod and the lean are nearly one parameter, which nothing below the cut
  pins; the queued learned-lean seed's opposing reader has a lever of under 3 mm. The scan
  headers do not carry the bend. Proposals: align the femur band's tokens to a running template;
  a per-token position head with a two-segment fit (needs a new loss term); hinge on the femur
  spots with a true rotation; hand marks on 100 to 200 knees as a truth; front readers on the
  kneecap spot along the measured line.
- **Check of the template idea without training**
  (`outputs/analysis/f/frame-and-targets/template-bend/`, bone-head weights, 874 validation knees,
  ×2): the rotation about the femur-spot hinge that best matches a knee's sagittal tokens to the
  average token per 8 mm cell, estimated separately from the T1/PD and the fluid series of the
  same knee, correlates at 0.76 between the two (796 knees), spread 12°. On fifteen knees drawn
  (`template-bend.png`) the direction follows the visible femur where the two agree; the three
  knees at the lower search limit have misplaced femur spots. Axial tokens do not agree with the
  sagittal ones (0.17). The same on the tibia band, hinged at the tibia spots: 0.59, spread 10°;
  tibia and femur angles correlate 0.25.
- **Queued from it:** `femur-section-turned-to-a-running-template` and
  `tibia-and-femur-sections-turned-to-running-templates` ([template seeds](template-ideas.yaml)),
  ahead of the shared-bend control. `tendon-targets-with-the-head-in-the-model` removed: a spot
  pair again.
- A second subagent is looking at what RadImageNet's knee classes and public knee segmentations
  could give as anatomical landmarks (user's request).

## Stopped by the user (2026-10-07: stop asked 16:56, run aborted 17:05 CEST)

- **Template femur section (first candidate): aborted at epoch 4 by the keeper.** Its angles
  saturated: 34 to 56% of knees at a limit of the search and a spread of 28 to 31°, against 2% and
  12° in the offline check; the two series groups still agreed (0.72 to 0.85). Cause, in the
  keeper's seed: every sagittal token took part, so at large angles tokens from below the joint
  turned into the grid, and the score was a sum divided by a count that did not depend on the
  angle. Corrected in that candidate's workspace (only femur-band tokens; mean similarity over the
  tokens that land; at least 40 of them) and requeued as `femur-section-turned-to-a-running-template-2`
  and `tibia-and-femur-sections-turned-to-running-templates-2`
  ([template requeues](template-requeue-ideas.yaml)). Neither ran.
- **`double-cylinder-with-a-learned-femur-lean`:** first launch crashed at the first validation
  step (a diagnostic rewritten by the developer mixed half and full precision); the keeper patched
  the workspace and the repair relaunched. Aborted at the user's request after epoch 7 of 10
  (validation macro 0.9142 against 0.9152 for the incumbent at that epoch; no final score). Its
  lean: +6.0° with a spread of 2.4° across knees after epoch 1, −19.0° with a spread of 0.7° after
  epoch 2. With one free angle (the frame's tilt stayed at zero), a reader inside the bone and its
  own view of the femur tokens, the gradient still sets one bend for all knees, as the research
  review predicted from the reader's short lever.
- **State at the stop:** incumbent run 24 (×4 0.925067, ×1 0.917070); six decided runs and two
  invalid; no adoption. Queued and not run: `double-cylinder-with-one-shared-bend`,
  `femur-section-turned-to-a-running-template-2`,
  `tibia-and-femur-sections-turned-to-running-templates-2`.
- **Open direction, raised by the user:** femur and tibia masks from a pretrained knee segmenter,
  run offline on the training studies, as a truth for the two tilts and as targets for a head in
  the model (the user's survey: `knee_mri_anatomy_pretrained_models.md`; a subagent's review of
  RadImageNet's knee classes and of public segmenters is in the session). The recorded rules allow
  pretrained models that are free and publicly accessible without institutional sign-off;
  non-commercial terms alone do not disqualify.

