# Study 3: soft-tissue segmenters

**`aagatti/nnunet_knee` is good on our 3D DESS series and not on routine 2D series.** The user's
fallback case holds: the mask model works on 3D series only.

**Why it failed before: contrast, not a pipeline error.** It was trained on DESS (OAI-ZIB).
Ruled out by test: configuration and label ids, axis order, the pipeline itself (its own test
image gives menisci of 2.3 and 1.6 ml), slice thickness (averaging that image to 3.2 mm slices
changes little), resampling our 2D volumes to 0.8 mm slices, and test-time mirroring.

**Results on our volumes** (found / passing a label-free acceptance test: volume in range, one
main piece, in place against the bone masks, correct side):

| Model, series (n) | Medial meniscus | Lateral meniscus | Femoral cartilage | Medial tibial | Lateral tibial | Kneecap cartilage | s per volume |
|---|---|---|---|---|---|---|---|
| nnunet_knee, DESS (10) | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 9 | 10 / 8 | 19 slow, 2.0 to 2.5 fast |
| nnunet_knee, sagittal 3D PD FS (10) | 7 / 5 | 9 / 3 | 10 / 10 | 9 / 9 | 10 / 9 | 10 / 10 | 27 |
| nnunet_knee, 2D FS sagittal (20) | 3 / 0 | 4 / 1 | 20 / 8 | 7 / 3 | 12 / 5 | 17 / 8 | 22 |
| nnunet_knee, 2D T1/PD sagittal (10) | 0 | 0 | 0 | 0 | 0 | 0 | 22 |
| nnunet_knee, 2D coronal (10) | 0 / 0 | 1 / 0 | 9 / 3 | 5 / 4 | 5 / 1 | 5 / 0 | 2 |
| SKM-TEA echo-1 U-Net, DESS (10) | 10 / 1 | 10 / 0 | 10 / 5 | 10 / 7 | 10 / 4 | 10 / 5 | 2.7 |
| SKM-TEA echo-1 U-Net, 3D PD FS (10) | 10 / 4 | 10 / 4 | 10 / 0 | 10 / 4 | 10 / 2 | 5 / 0 | 3.0 |

- DESS by eye: the passing masks are good (menisci as horn triangles and peripheral wedges,
  cartilage as a layer on each surface; median menisci 2.6 and 2.2 ml, femoral cartilage
  12.5 ml). With n 10, 10 of 10 only bounds the true rate above about 74%.
- 3D PD FS: cartilage good, menisci undersized and unstable (fast and slow runs agree only to
  Dice 0.65 to 0.80 on thin classes).
- 2D fat-suppressed sagittal: cartilage sits on the bones but covers about half the true volume.
- SKM-TEA: fragmented from slice to slice on our DESS; worse than nnunet_knee everywhere.
- The DESS acceptance test used bones from the same study's 2D sagittal T1/PD series, so it
  rests on alignment through patient coordinates (about 1 to 2 mm, study 3b).

**What exists and what does not.**

| Class | Provider | Usable knees |
|---|---|---|
| Menisci, medial and lateral | nnunet_knee on DESS | about 240 to 264 of the 264 DESS studies |
| Femoral, tibial, kneecap cartilage | nnunet_knee on DESS; cartilage only from sagittal 3D PD FS | DESS as above, plus about 130 of 147 |
| ACL, PCL, MCL, LCL, effusion, pouch, Baker's cyst, tendons, fat pad | **no released weights found** | 0 |

All DESS comes from one scanner and protocol (Siemens Avanto fit 1.5T).

**Other candidates** (weights' licence, per the rule): SKM-TEA Apache-2.0 (run); kneeseg SKI10
MIT (random forest, not run); MedSAM2 CC-BY-SA, promptable (not run); DOSMA, KneeXNet-2.5D,
CartiMorph licence unclear (not run); TotalSegmentator MR sub-tasks need a licence key; Denizlab
IWOAI and rocaseg AGPL or research-only. Not checked: MONAI model zoo, MRSegmentator, SAM-Med3D.

**Fast runner:** `fast_nnunet_knee.py` in the session scratchpad (`soft-tissue/`): one model
load, threaded reading, crop to foreground, no window overlap, no mirroring, mixed precision,
argmax from interpolated logits on the GPU. 2.46 s per DESS volume (5 GB): 264 DESS in about
11 minutes. Against the standard pipeline on DESS: Dice 0.96 to 0.98 on menisci and cartilage,
acceptance rates unchanged. Restoring by nearest label dropped thin classes to 0.85 to 0.93 and
was discarded. Not validated on 3D PD FS.

Drawings: `dess_full.png`, `dess_zoom.png`, `pd3d_full.png`, `native_full.png`,
`fast_coronal_s10.png`, `skm_dess_crop_U-Net_E1.png`.
