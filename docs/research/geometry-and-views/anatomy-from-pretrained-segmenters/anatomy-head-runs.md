# Anatomy head and frame runs (2026-10-07)

- 2026-10-07: anatomy head built : `src/knee_abnormality/training/anatomy.py`
  (targets packed on the GPU, per-token shares from 16 points, linear head on the tokens before
  the position code), loss in `training/lightning.py`, `patient_position` and `token_steps`
  outputs in `modeling/model.py`, settings `paths.anatomy_targets` and
  `trainer.anatomy_loss_weight`.
- 2026-10-07: first run, `outputs/training/rn18-anatomy-head-sample550` (run 24's config, fold 1,
  seed 1110484336, anatomy weight 0.1, 400 supervised training knees; launched by
  `launch_anatomy_head.sh`). Against run 24: ×4 0.924947 against 0.925067 (−0.00012), ×1 0.916815
  against 0.917070 (−0.00026); sibling SD 0.0004; same 2 min 3 s per epoch. Anatomy on the 150
  validation knees at epoch 9: main class right for 94.0% of tokens, overlap femur 0.892, tibia
  0.883, kneecap 0.665, fibula 0.538, share error 0.071 (epoch 0: 0.845, 0.837, 0.414, 0.132);
  training agreement 93.3%. Labels at ×4 (yardstick in brackets): Lateral Meniscus +0.0042
  (0.0024), Medial OA +0.0017 (0.0008), Contusion +0.0017 (0.0013), Lateral OA +0.0015 (0.0034),
  Fracture +0.0011 (0.0044), ACL +0.0003, Baker's +0.0002, Medial Meniscus −0.0004, Effusion
  −0.0017 (0.0015), PF OA −0.0018 (0.0005), MCL −0.0037 (0.0029), Synovitis −0.0045 (0.0022).
  Lesson: the model learns femur and tibia per token from 400 knees at no cost in speed or macro
  score; the kneecap and fibula stay weak, the kneecap being outside most sagittal and coronal
  crops. The label heads do not use the prediction yet.
  A first launch lost its last two options to a wrapped paste and trained a plain repeat for one
  epoch (0.89281); its folder was deleted.
- 2026-10-07: joint plane from the predicted bone map, checked without training on the 150
  validation knees with the run above (`anatomy_joint_plane_check.py`,
  `anatomy-joint-plane-check.csv`). Plane fitted to femur share minus tibia share over bone
  tokens, refitted three times with weights near the plane (12 mm): level spread 1.7 mm with a
  constant offset (the plane sits 6.4 mm above the plateau's surface), forward tilt error sd 2.8°
  (real tilt sd 4.5°), sideways tilt error sd 2.0° (real sd 4.0°). Without the refit: 2.1 mm,
  3.3°, 3.3°. A separating plane (logistic): 1.8 mm, 3.2°, 2.5°.
- 2026-10-07: second run, `outputs/training/rn18-anatomical-frame-sample550`
  (`model.anatomical_frame`; `launch_anatomical_frame.sh`): a height head in the model predicts
  each token's height above the plateau, one plane per knee is fitted to the predictions, and the
  frame takes its height origin and tilt from that plane, with a joint row of half a token each
  side (learnable offsets) and the label gradient kept out of the plane. Drawings
  `anatomical-frame-3d.png`, `anatomical-frame-cylinders.png`.
  **The height target does not work as the plane's source.** At epoch 3 on the 150 validation
  knees (`anatomical_frame_knee_errors.py`): level spread 5.9 mm (7.6 mm from the loader's knee
  centre alone), forward tilt error sd 4.1° (real 4.5°), sideways 2.7° (real 2.9°); the height
  head's error is one shift per knee (sd 5.9 mm between knees, 3.8 mm inside a knee), so more
  tokens do not average it out. Reading: local appearance says what a token is, not how far it
  is from the joint. **Lesson for the next build: take the plane from the predicted bone map
  (1.7 mm, 2.8°, 2.0°), with the bone head inside the model, and drop the height head.**
  Final: ×4 0.923643 (−0.00142 against run 24), ×1 0.915436 (−0.00163).
- 2026-10-07: can the femur's and tibia's own tilts be known? Fitting a centre line through the
  femur tokens (`anatomy_bone_axes_check.py`) does not give the femur's direction: correlation
  with the real bend 0.04 to 0.33 from the model's map and 0.03 to 0.45 from the segmenter's own
  masks at token level; the crops show about 45 mm of femur, which is condyle. A linear probe on
  the trained model's bone-weighted mean features (`anatomy_tilt_probe.py`, 399 training knees,
  150 unseen): forward bend of femur against plateau correlation 0.70 (error sd 5.4° of 7.4°),
  plateau forward tilt 0.63, femur forward lean 0.53, sideways tilts 0.26 to 0.46.
- 2026-10-07: third run, `outputs/training/rn18-tiltable-frame-sample550`
  (`launch_tiltable_frame.sh`; height head removed). Bone head inside the model; joint plane
  fitted to predicted femur share minus tibia share over bone tokens with three refits near the
  plane; frame's height origin and tilt from that plane; joint row half a token each side
  (learnable); a bend head on bone-weighted pooled features (two bones by three planes) predicts
  the femur's forward lean, trained on the real shaft from the masks; the femur section above
  the upper cut is sheared forward by it. Checks before launch: `tiltable_frame_check.py`.
  Against run 24: ×4 0.924810 (−0.00026), ×1 0.916333 (−0.00074). On the validation knees at
  epoch 9: joint plane error 1.64 mm (1.85 at epoch 0), bend error 2.8° against 5.2° for always
  predicting no bend (5.1° at epoch 0), overlap femur 0.887, tibia 0.873, kneecap 0.640.
  Labels at ×4: MCL +0.0048, Contusion +0.0029, ACL +0.0010, Effusion +0.0009, Medial Meniscus
  +0.0003, Baker's +0.0003, Lateral Meniscus −0.0002, Lateral OA −0.0010, Medial OA −0.0012,
  Synovitis −0.0028, PF OA −0.0039 (yardstick 0.0005), Fracture −0.0042 (0.0044).
  Lesson: the plane from the bone map is accurate from the first epoch and the bend is learnable
  from the picture; the frame anchored on them scores level with run 24 overall, with PF OA the
  one clear loss.
- 2026-10-07: joint classes derived from the bone masks (`joint_class_candidates.py`,
  `joint_class_scale_free.py`; `joint-class-candidates.png`, `joint-class-scale-free.png`).
  Distance rules: within 4, 6, 8 mm of both bones give 1.2, 10.0, 20.3 ml; distances adding to
  12 mm or less 22 ml; straight between the bones 14.7 ml but 8 to 10 mm narrower. Direction
  rule (nearest femur and tibia surfaces on opposite sides): 10.9 ml at 150°, 31.1 ml at 120°,
  no millimetres, and it also fills the notch between the condyles. Scaling by the knee's own
  size is too noisy. Kneecap joint within 6 mm of kneecap and femur: 4 ml. Masks need the
  largest piece per bone kept (a stray tibia piece on row 374) and holes filled.
- 2026-10-07: seven-class targets, `data/labels/prepared/anatomy-mrannotator-sample550-joints/`
  (`segmenter_export_joint_targets.py`, 3 minutes on four workers at 1 mm): bones closed (largest
  piece, 4 mm closing by distance transforms, holes filled); joint = points with femur and tibia
  on opposite sides (135° or more), closed 4 mm, bones removed; kneecap joint the same between
  kneecap and femur. Medians: joint 29.7 ml, kneecap joint 18.3 ml. Checks
  `joint-class-closed-twice.png`, `anatomy-lookup-check-joints.png`,
  `joint-targets-pixel-view.png`: the joint is right and generous front and back; the kneecap
  class also covers the pouch above the kneecap and the gutters; classes are blocky from side to
  side (3 to 4 mm sagittal slices).
- 2026-10-07: fourth run, `outputs/training/rn18-anatomy-sections-sample550`
  (`launch_anatomy_sections.sh`; bend head and femur lean removed). Seven classes inside the
  model; sections by class (femur, joint and tibia shares, the rest placed by height against the
  joint plane); per-route section weights as parameters starting at 0.95 or 0.05; kneecap route
  also reads kneecap and kneecap-joint tokens. Against run 24: ×4 0.924694 (−0.00037, bootstrap
  95% interval −0.0019 to +0.0012), ×1 0.916203 (−0.00087, −0.0031 to +0.0012).
  Audit in `outputs/analysis/frame-and-targets/anatomy-sections-audit/` (`audit.txt`,
  `map-prediction.png`, `attention-per-label.png`, `route-weights.png`; scripts
  `audit_extract.py`, `audit_analyse.py`, `audit_visual.py`; run 24 extracted with its archived
  source):
  - Labels, paired bootstrap on 874 knees: Synovitis −0.0052 at ×4 (4% of draws above zero) and
    −0.0098 at ×1 (interval excludes zero); ACL −0.0062 at ×1 (excludes zero), −0.0018 at ×4;
    Medial OA +0.0051 at ×1 (excludes zero); PF OA −0.0026 at ×4 (15% above zero); the rest
    inside their intervals.
  - Map over the 150 validation knees: main class right 92.7%; overlap femur 0.897, tibia
    0.888, joint 0.741, kneecap 0.687, kneecap joint 0.633, fibula 0.568; joint tokens mistaken
    for femur 8%, tibia 7%, other 10%; plane error 1.58 mm.
  - Learned: the section weights did not move (all within 0.02 of their start); cuts +6.4 and
    −5.9 mm; core radius 9.3 mm against 14.8 in run 24, ring 21.9 against 26.3.
  - Attention by real anatomy: menisci and compartment OA moved from femur (0.34 to 0.39 of
    attention in run 24) to tibia and joint (femur 0.16 to 0.18, tibia 0.30 to 0.33, joint 0.28
    to 0.35) and became more focused (59 to 98 effective tokens against 89 to 135). ACL: femur
    0.63 to 0.40, joint 0.13 to 0.29, tibia 0.00 to 0.08. PF OA: femur 0.34 to 0.49, kneecap
    0.12 to 0.07, kneecap joint 0.15 to 0.13; its region grew (mean weight 0.25 to 0.31) and
    routing moves less of its attention (0.66 to 0.50). Synovitis: joint 0.05 to 0.11, kneecap
    and kneecap joint 0.14 to 0.09, region 0.30 to 0.34.
  - Sides consistent: fibula tokens weigh 0.95 in the lateral meniscus route and 0.00 in the
    medial one; no knee with an uncertain side; routing fallback never used.
  Lessons: sections by anatomy work for the joint labels; the front regions (kneecap,
  front_core) are a broad block of the front half of the knee and got broader, which is where
  PF OA and Synovitis lost; the section weights need a faster learning rate or a milder start to
  be learned at all; the core shrank and ACL lost femur attention.
- 2026-10-07: fifth run, `outputs/training/rn18-anatomy-front-sample550`
  (`launch_anatomy_front.sh`, `apply_front_regions_edits.py`): kneecap route = kneecap and
  kneecap-joint tokens (full at 30% share) plus the front block above the joint at a quarter;
  front_core above the joint row plus kneecap tokens; section preferences starting at 0.88 and
  0.12 and learning five times faster; core and ring fixed at 1.1 and 2.2 pitches. Against run
  24: ×4 0.924984 (−0.00008, interval −0.0016 to +0.0014), ×1 0.917087 (+0.00002, −0.0020 to
  +0.0018). Audit `audit-front.txt`. Labels: Medial OA +0.0062 at ×1 (interval excludes zero)
  and +0.0015 at ×4; Contusion +0.0030 at ×4 (94% of draws above zero); Baker's +0.0018 (87%);
  Lateral Meniscus +0.0039 at ×1 (88%); ACL back to level (0.0000 at ×4, −0.0012 at ×1); PF OA
  −0.0014 at ×4, +0.0005 at ×1, its attention on the kneecap joint 0.15 to 0.19; Synovitis
  −0.0068 at ×4 and −0.0110 at ×1, both intervals excluding zero. The section preferences now
  move (tear routes up to 0.4 from their start; label routes within 0.03).
  **Synovitis fell in every run that trains the anatomy map, including the first, where the
  frame was untouched** (×4: −0.0045, −0.0028, −0.0052, −0.0068), with its attention by anatomy
  now equal to run 24's. Reading: the cost comes from the anatomy loss shaping the backbone, not
  from the regions. Without Synovitis this run would be about +0.0005 at ×4 and +0.0009 at ×1.
  Next check: train the map without its gradient reaching the backbone.
- 2026-10-07: a map head on detached tokens (two layers, with Dice; `rn18-anatomy-reader-sample550`,
  stopped by the user's decision after one epoch): femur 0.52, tibia 0.44, joint 0.40, plane error
  4.6 mm against 0.85, 0.84, 0.68 and 1.8 mm with the gradient. The map's fast learning is the
  backbone adapting.
- 2026-10-07: sixth run, `outputs/training/rn18-anatomy-crops-sample550` (`launch_anatomy_crops.sh`;
  `apply_pose_crops_edits.py` on the crops-turned build of the frame-and-targets campaign): Dice
  added to the map's loss (gradient to the backbone, linear head) and posed crops: canvases with a
  17 mm margin for sagittal and coronal views, a first look without gradient in training and at
  prediction, each crop shifted so the joint centre lands at the average spot and turned so the
  joint is level (limits 12° and three quarters of the margin; training jitter ±3° and ±3 mm).
  Check with the true pose: `pose_crops_check.py`, `pose-crops-check.png`. Against run 24: ×4
  0.925885 (+0.00082, interval −0.0020 to +0.0035), ×1 0.917130 (+0.00006); **0.953 public as S122**,
  equal to run 24's S118. Training 1,863 s (a job ran beside it) and ×4 prediction 55 s against
  1,274 s and 33 s. Map: femur 0.905, tibia 0.899, joint 0.759, kneecap 0.735, kneecap joint 0.675,
  plane error 1.73 mm. Per target and per view: `target-analysis.txt`; audit `audit-crops.txt`;
  drawings `frame-prediction.png`, `resulting-crops.png`. Gains: Medial OA +0.0033 / +0.0112,
  Baker's +0.0041 / +0.0049, MCL +0.0065 / +0.0096, Lateral OA +0.0053 / +0.0041, Medial Meniscus
  +0.0021 / +0.0039. Losses: Lateral Meniscus −0.0046 / −0.0148, Synovitis −0.0056 / −0.0083,
  Contusion −0.0058 and Fracture −0.0057 at ×1, ACL −0.0030 at ×4.
  Flaws found in the audit: (1) the turned crops are cut bilinearly and keep 0.81 of the fine
  detail, bicubic keeps 0.98 (`posed_crop_sharpness.py`); (2) the shift across removes only a third
  of its spread (correlation 0.76 to 0.81 with the real one) while the shift up and down removes
  three quarters (0.97) (`pose_accuracy.py`); (3) the turn's correlation with the real one is 0.73
  to 0.80; (4) Synovitis pays for the map in every run; (5) the canvases covered the 95th
  percentile of the need (45 and 39 px), so they were not oversized; (6) the audit compared run
  24's attention with the anatomy looked up at the posed tokens' positions, corrected to its own
  positions (the figures moved by up to 0.05).
- 2026-10-07: campaign `autoresearch/campaigns/anatomy-masks/` created with the sixth run as
  reference and incumbent and seven forced seeds.
