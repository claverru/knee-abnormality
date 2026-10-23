# Study 3b: our 3D series, census and alignment with 2D series

836 3D series in 659 of 4,407 training studies (headers from the 2026-09-17 census); alignment
on 14 studies (7 DESS, 7 sagittal 3D PD fat-suppressed, 5 scanner models) with the bone
segmenter.

**How many studies could supply a mask from a 3D series: 413 with a fluid-sensitive one.**

| Family | Series / studies | Plane | Slice (mm) | Pixel (mm) | Slab (mm) | Scanners |
|---|---|---|---|---|---:|---|
| DESS | 268 / 265 | axial | 0.6 | 0.25 | 86 | 98.9% one Siemens Avanto fit 1.5T, German reports |
| 3D PD fat-suppressed | 148 / 148 | sagittal | 0.4 to 0.89 | 0.28 to 0.89 | 128 | Philips Achieva 57%, Siemens Skyra 3T 42% |
| 3D T1/T2* gradient echo | 216 / 214 | sagittal, axial | 1.0 | 0.31 | varies | four scanner groups |
| Zero-TE (GE) | 130 / 88 | sagittal | 1.0 | 0.39 | 120 | 94% one GE Architect 3T |
| Reformats of the 3D PD series | 50 / 50 | axial | 0.9 | 0.28 | 167 | Philips Achieva |

- **Concentration:** the 413 come from three scanner and site groups. There is no institution
  tag; "site" is report language plus scanner model.
- **Every 3D series holds one knee** (the 320 mm DESS field is 124 mm of body at the median).
- **They are already model inputs:** no filter uses the acquisition type; they are loaded as
  ordinary views in their plane and contrast slot (3.06% of training triplets).
- **Their 2D companions:** DESS studies have sagittal T1/PD (264 of 265), coronal fluid (265)
  and axial T1/PD (263), but a 2D sagittal fluid series in only 20. The 3D PD studies have
  sagittal T1/PD, coronal T1/PD and axial fluid, and a 2D sagittal or coronal fluid series in 1.
- **Folds:** DESS 53 / 52 / 40 / 60 / 60; 3D PD 30 / 33 / 26 / 36 / 23. 59 of the 413 are among
  the 550 knees with bone masks.
- **Their labels differ from the rest** (percent positive):

| Group | n | ACL | Lateral Meniscus | Medial Meniscus | PF OA | Effusion | Synovitis | Contusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All training | 4,407 | 14.9 | 15.4 | 38.8 | 33.1 | 27.8 | 12.9 | 10.5 |
| DESS | 265 | 8.3 | 7.2 | 29.1 | 32.5 | 41.9 | 17.0 | 15.8 |
| 3D PD fat-suppressed | 148 | 7.4 | 14.2 | 36.5 | 43.2 | 7.4 | 6.1 | 6.8 |

**Coverage of the 3D slabs.**

- DESS (43 studies): the slab is centred on the kneecap, 65 mm above the joint line and 26 mm
  below at the median. Whole kneecap inside in 43 of 43; the cruciate band in 42 of 43; under
  20 mm of tibia below the joint line in 7 of 43. It covers 42 to 57% of a 2D series' voxels.
- Sagittal 3D PD (30 masks): femur, tibia, kneecap and the joint band inside in 30 of 30, with
  10 to 29 mm beyond the bones on each side.
- Menisci and cruciates were placed from bone landmarks, not segmented.

**Optional extra: a 3D-made mask on the same study's 2D series.**

| 3D type | 2D plane | Dice femur / tibia / kneecap | Surface distance, median (mm) | Best rigid shift, median / max (mm) |
|---|---|---|---|---|
| DESS | sagittal | 0.92 / 0.93 / 0.84 | 0.7 to 1.1 | 1.4 / 2.25 |
| DESS | coronal | 0.94 / 0.88 / 0.82 | 1.0 to 2.0 | 1.1 / 2.1 |
| 3D PD | sagittal | 0.94 / 0.93 / 0.90 | 0.6 to 0.9 | 1.1 / 2.5 |
| 3D PD | coronal | 0.86 / 0.81 / 0.79 | 0.9 to 1.8 | 2.1 / 3.7 |

- Good enough for 12 to 17 mm tokens; marginal for menisci and cartilage, which need 1 to 2 mm,
  the size of the shift between series. About half the error is motion between series.
- **The bone segmenter is unreliable on 2D coronal series** (shafts cut into boxes, one kneecap
  misplaced by 30 mm on the sheets), so the coronal figures understate the 3D mask.

**Limits.** Translation only; soft-tissue alignment not measured; 8 of 14 comparisons used
sagittal T1/PD because these studies lack 2D sagittal fluid; 3D volumes were halved in-plane to
about 0.5 mm before segmentation (8.8 s per volume, 9 GB).
