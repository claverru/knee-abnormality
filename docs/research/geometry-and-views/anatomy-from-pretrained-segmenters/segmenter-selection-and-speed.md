# Segmenter selection, benchmark and speed

Worked on 2026-10-07 as a list at the repository root, moved here on 2026-10-10. The
one-off scripts it names under `scripts/adhoc/side_routed_session/` were not kept.
Sources: the user's survey (`knee_mri_anatomy_pretrained_models.md`), two research reviews of the
same day, and the frame-and-targets campaign
([run notes](../../../../autoresearch/campaigns/frame-and-targets/run-notes.md)).

## Why

Every frame we built reads the knee's geometry from where label-trained heads look. Those spots
carry 4 to 6 mm of noise, move with training, and cannot give a bone's direction. A pretrained
segmenter gives the bones themselves.

## Rules we keep

- The segmenter runs **offline, on training and validation studies only**. Nothing from it goes
  into the loader or the Kaggle bundle; at prediction the model uses only what it learned.
- Models must be free and publicly accessible without institutional sign-off (host rulings in
  `docs/modeling/supervision.md`). Clear: TotalSegmentator `total_mr` (Apache-2.0, femur only).
  Passes the host's test: MRAnnotator (CC-BY-NC, public weights). Unchecked: the others.

### Models cleared by the user's licence and rules pass (2026-10-07)

In priority order: `aagatti/nnunet_knee` (femur, tibia, patella, cartilage, menisci); SA-Profile
(knee MRI landmarks); MRAnnotator (femur, tibia, patella, fibula; yes with a caveat); PYCAD Knee
(ACL, PCL, meniscus, cartilage); SKM-TEA models (cartilage, meniscus); kneeseg SKI10;
MultiPlanarUNet knee; Denizlab IWOAI; SAM-Med3D, MedSAM2, nnInteractive (promptable refinement);
TotalSegmentator `total_mr` (legal, less knee-specific); kneeseg OAI (weak).
Removed: SAT, RobustCartilageSegmentation, BiomedParse, TotalSegmentator
`appendicular_bones_mr`, DOSMA weights, LDSeg, KOA-Segmentator, AutoMedLabel, KneePreM.

### Benchmark so far (idea 0)

- `aagatti/nnunet_knee`, full resolution, 24 sagittal volumes of validation studies: on
  fluid-sensitive fat-suppressed series femur 11/11, tibia 11/11, patella 10/11, clean outlines;
  on T1/PD series 1/13 for each bone; menisci found in none. About 9 cm of femur and 6 cm of tibia
  visible. 45 s per volume at full resolution, about 10 s with the coarse model and a larger step.
- 4,150 of the 4,407 training studies (94%) have a fat-suppressed sagittal series; 257 have only a
  T1/PD sagittal.
- MRAnnotator's weights could not be downloaded from the command line (the Google Drive link
  returns "not found" without a browser session).
- Scripts and volumes: `/home/claudio/.cache/knee-segmenters/`; drawings:
  `outputs/analysis/frame-and-targets/segmenter-benchmark/`.

## What we compute

- **One sagittal series per study** (largest field, most slices), as a 3D volume. Every series of
  a study shares patient coordinates, and series move under 2.7 mm against each other in 99% of
  pairs, so one mask places the bones in all slices of the study.
- **Four structures:** femur, tibia, patella, fibula.
- **Kept:** one small volume per study in patient coordinates at 2 to 3 mm.
- **Stages:** about 30 studies (benchmark), the 874 validation knees (yardstick), all training
  studies (targets).

## Benchmark and speed log (2026-10-07)

- 2026-10-07 18:40: benchmark of `aagatti/nnunet_knee` with the coarse model (6 s per volume) on
  34 fat-suppressed sagittal volumes: femur and tibia 34/34, patella 21/34, menisci 5/34; femur
  axis repeats to 1.6° between two series of one knee (median, 6 pairs), tibia axis to 9.1°.
  Keep only each bone's largest connected piece (stray pieces in muscle on row 532).
- 2026-10-07 18:50: dropped SA-Profile (only the kneecap groove of the femur; no tibia, no tilts)
  and idea 6 (knee side from the fibula): the user considers medial and lateral already right.
  MRAnnotator's only remaining use is the 6% of studies without a fat-suppressed sagittal.
- 2026-10-07 19:20: tibia direction from the plateau (a plane fitted to the top of the tibia
  mask) instead of the shaft: repeats to 1.4° in the sagittal plane and 0.9° in the coronal plane
  between two series of one knee (median, 5 pairs); the tibia shaft fit stays poor (4.8°, wrong on
  row 370 where the picture cuts the bone). Femur shaft 2.5°. A rule "the tibia is the piece
  right under the femur" fixes row 532 and rejects 3 of 34 volumes. On 26 knees: femur lean sd
  4.8°, plateau tilt sd 3.7°, uncorrelated (−0.18); bend (femur against the plateau's normal)
  mean −1.7°, sd 6.6°, range −12 to +16°, beyond 5° in 54% of knees; bend repeats to 1.9°.
  Scripts `bone_geometry.py`, `tibia_fit.py`, `bend.py` in `/home/claudio/.cache/knee-segmenters/`;
  drawings `bone-axes-first-fit.png`, `bone-axes-second-fit.png`.
- 2026-10-07 19:50: masks projected onto run 24's crops and 7 by 7 token grid for 26 knees
  (`scripts/adhoc/side_routed_session/token_anatomy_maps.py` (removed), CPU only; `token-anatomy-maps.png`,
  `token-anatomy-shares.parquet`). The outlines land on the bones in all three planes, T1/PD
  series included, from the one fat-suppressed sagittal mask. Per sagittal or coronal image about
  10 tokens are mostly femur and 7 mostly tibia; half the tokens hold some bone, and in 70% of
  those one bone fills at least half the token. The kneecap is at the crop's edge or outside
  (0.3 to 0.7 tokens per image). The sagittal and coronal crops show almost no femur shaft.
- 2026-10-07 20:05: run 24's frame against the bones on the same 26 knees
  (`scripts/adhoc/side_routed_session/frame_against_bones.py` (removed), `frame-against-bones.png`, `.csv`).
  Centre: the real joint centre is 1.2 mm lateral (sd 3.0) and 4.4 mm in front (sd 4.6) of the
  frame's. Heights: the lower cut is 11.0 mm above the joint level (sd 3.1, range 6.5 to 18.3),
  the upper cut 26.1 mm above (sd 2.5); neither is between the bones. Nod: frame +13.7° (sd 4.1),
  plateau +1.0° (sd 3.9), femur −0.2° (sd 4.7); frame minus plateau +12.7° (sd 6.1); correlation of
  the frame's nod with the plateau's −0.16 and with the femur's −0.06. Roll: frame sd 5.8°,
  plateau sd 3.2°, correlation +0.33.
- 2026-10-07: bone coverage of run 24's crops on the 26 knees, inside each crop's rectangle
  (`scripts/adhoc/side_routed_session/crop_bone_coverage.py` (removed), `crop-bone-coverage.csv`). Sagittal
  84 mm (T1/PD): condyles 99%, plateau 97%, femur seen to 42 mm above the joint and tibia to 39 mm
  below (90 and 62 mm segmented), kneecap 52% inside on average and whole in 11% of knees.
  Sagittal 93 mm (fluid): femur 49 mm, tibia 42 mm, kneecap 74%, whole in 38%. Coronal 84 and
  93 mm: femur 42 and 48 mm, tibia 40 and 44 mm. Axial 120 mm: kneecap whole in 96 to 100%.
- 2026-10-07: a speed test with four segmenter processes at once ran WSL out of memory twice
  (about 28 processes). Keep one process (6 s per volume, about 14 GB); no parallel parts.
- 2026-10-07: single-process fast segmenter (`scripts/adhoc/side_routed_session/segmenter_fast.py` (removed),
  run with the segmenter environment): reading in three threads, preparation and restoring on the
  GPU, the model's own sliding window. 0.56 s per volume on the 67 benchmark volumes (0.51 s of it
  the model), 4.0 GB of memory, 4 GB of GPU memory, read-ahead bounded to 12 volumes and a 14 GB
  memory guard. Overlap with the original pipeline's masks on the 34 fat-suppressed volumes
  (`segmenter_fast_agreement.py`): femur median 0.999 (lowest 0.983), tibia 0.998 (0.944),
  kneecap 0.993 (0.941). All 4,150 studies: about 40 minutes plus the series conversion.
- 2026-10-07: tried to go below that (`segmenter_fast_settings.sh`, 67 volumes each). Patches per
  call 1, 2, 4: 0.51, 0.50, 0.51 s per volume (4.3, 5.5, 7.9 GB of GPU memory): the GPU's compute
  is already full with one patch. Channels-last memory format: slower, 0.72 s. Coarser input
  breaks the masks: 1.3 times coarser gives overlap 0.964 femur, 0.919 tibia, 0.744 kneecap
  (lowest 0.33, 0.12, 0.06); 1.6 times gives 0.941, 0.835, 0.615. Kept: native scale, one patch
  per call, 0.51 s per volume. Next cost is reading: read the series with the repository's own
  DICOM reader and keep only a small bone volume, instead of converting to files.
- 2026-10-07: MRAnnotator. The README's visible weights link is dead; the address behind it works
  (Google Drive id `15gQuvmaDlbs7HlGpISWNqnnWYGp9-Xc4`, 2.03 GB, no login). Licence: CC-BY-NC,
  stated in the README and the paper (Radiology Advances 2024); weights public; only their
  dataset needs a usage agreement. Run on the 67 benchmark volumes in LAS orientation with a
  single-process script (`segmenter_mrannotator.py`, `segmenter_mrannotator_summary.py`,
  `mrannotator-samples.png`): femur, tibia, kneecap and fibula found in 34/34 fat-suppressed and
  33/33 T1/PD volumes, every knee usable (row 532 included), one connected piece per bone.
  Between the T1/PD and the fat-suppressed series of the same knee (28 pairs): plateau tilt
  differs by 0.4° at the median (90th percentile 1.4°), femur axis by 0.3° (1.0°). Overlap with
  the first model on fat-suppressed series: femur 0.87, tibia 0.85, kneecap 0.56 (its bones are
  smaller in volume, 157 against 195 ml of femur; its kneecap is complete where the first
  model's was partial). 3.3 s per volume (84 patches at 0.36 mm), 3.1 GB of memory, 5.3 GB of GPU.
  MRAnnotator replaces `aagatti/nnunet_knee` for the bones.
- 2026-10-07: MRAnnotator speed (`STEP` and `SKIP_AIR` settings of the script,
  `segmenter_compare.py`). No overlap between patches (`STEP=1.0`): 1.85 s per volume, 47 patches
  instead of 84; against the overlapped run the plateau tilt moves by 0.16° at the median (largest
  1.1°) and the femur axis by 0.09° (largest 3.1°); overlap femur 0.988, tibia 0.987, kneecap
  0.971, fibula 0.978 (lowest 0.92, 0.86, 0.65, 0.81). Skipping patches of air saves under two
  patches per volume (1.81 s) with identical masks. The cost is structural: the model works at
  0.36 mm in every direction, so our 3 to 4 mm slices are repeated about ten times.
- 2026-10-07: sample of 550 knees (400 training, 150 validation with the six sample knees; one
  sagittal series each, the one with most slices; `segmenter_convert_sample.py`, seed 20261007)
  segmented with `STEP=1.0`: 18.9 minutes, 2.06 s per knee, 5.3 GB of memory. Targets exported
  by `segmenter_export_targets.py` to `data/labels/prepared/anatomy-mrannotator-sample550/`
  (2.5 mm bone volume per knee, `knees.json` with joint centre, plateau normal, femur axis).
- 2026-10-07: copying each slice fewer times (`SQUASH`), against the `STEP=1.0` masks on the 67
  benchmark volumes. Half the copies: 1.3 s per volume, overlap femur 0.957, tibia 0.968, kneecap
  0.941, fibula 0.913 (lowest 0.83, 0.81, 0.73, 0.00); plateau tilt moves 0.32° at the median
  (largest 1.5°), femur axis 0.24° (90th percentile 1.4°, largest 13.9°). A third of the copies:
  0.7 s, overlap 0.856, 0.890, 0.863, 0.667; plateau 0.44°, femur axis 1.75° (90th percentile
  6.5°). Half the copies degrades mildly, a third clearly; kept at full copies.
