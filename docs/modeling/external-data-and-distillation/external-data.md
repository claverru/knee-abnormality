# External data: fastMRI and OAI

**The final models were pretrained on 19,265 external knee MRI studies (9,828 fastMRI and 9,437
OAI) labelled by the project's own teacher ensembles, then fine-tuned on Kaggle studies only.**
The external images never reached a Kaggle submission bundle; only weights trained on them did.
How the labels were made and used is in the [teacher-student chain](teacher-student-chain.md);
measured outcomes are in the [teacher-student results](../../validation/results/teacher-student/INDEX.md).
Whether OAI was admissible under the competition's external-data rule was debated publicly and
never ruled on by the host ([2026-10-09 re-sweep](../../research/competition-and-prior-art/2026-10-09-public-resweep.md),
[2026-09-28 re-sweep](../../research/competition-and-prior-art/2026-09-28-public-resweep.md));
this repository does not record a ruling.

## fastMRI

- **Data:** fastMRI knee DICOMs, 10,009 studies and 47,552 series (47,585 catalogued series
  later), kept on the rented machine at `/workspace/data/fastmri` (Vast instance 55528957, then
  the successor instance). Preprocessing was audited against the Kaggle path: decoded pixels
  identical, geometry and intensity statistics matching. Download, layout (`fastmri_layout.py`),
  loader and geometry screens and the joined fastMRI + OAI root (`external_root.py`):
  [final-chain record](../../../scripts/adhoc/final_chain/README.md#1-external-data-on-the-host).
- **Excluded from training:** 181 studies (166 with a single series) whose knee geometry cannot
  be fitted. Their positions are all zero and `torch.hypot` in the routing weights has no finite
  gradient there, so one such study corrupts the weights. Listed in `excluded_from_training.csv`;
  the training table has 9,828 studies. Inference on them is fine.
- **Stored uncompressed since 2026-10-13:** the JPEG 2000 decoder (`pylibjpeg-openjpeg` 2.5.0)
  crashed a loader worker with a segmentation fault in three of twelve campaign trainings, after
  29 to 55 minutes each, inside `openjpeg`'s `decode_pixel_data`, sporadically and not tied to a
  file. The decoder was removed from the pinned dependencies (Pillow's decoder gives identical
  pixels on 1,500 images but takes 13.8 ms per image against 7.2 ms) and the series were
  rewritten in place as uncompressed DICOM keeping every identifier: 1,575,673 images converted in
  27 minutes, 18 already uncompressed, 0 failed; 4,000 sampled images hashed before and after, 0
  differ; none of 1,500 checked still compressed (`uncompress_fastmri.py`). fastMRI then took 432 GB instead of 164 GB, and
  the compressed originals were gone from that machine. Training no longer decoded fastMRI
  images, so the `fastmri-joint` autoresearch campaign's time limits no longer fitted and it was
  never run.
- **No 3D series:** fastMRI has no DESS or other 3D series (0 of 47,585), so the soft-tissue
  segmenter cannot read it; MRAnnotator can.

### Scanners against Kaggle

One DICOM header per study, read on 2026-10-13 (`/workspace/data/fastmri/study_headers.parquet`
on the rented machine); 9,828 training studies. Institution and station names are blank
(de-identified); 39 distinct device serial numbers.

| | fastMRI | Kaggle (4,407 studies) |
|---|---|---|
| Siemens | 81% | 44% |
| GE | 9% | 21% |
| Philips | 2% | 30% |
| Hitachi / Fujifilm | 5% | 0.5% |
| Toshiba / Canon | 0% | 5% |
| Unknown | 3% | 0% |
| 1.5 T | 69% | 57% |
| 3 T | 30% | 36% |

fastMRI's models are mostly older Siemens generations (Symphony 1,628, Skyra 1,423, Espree 1,309,
Aera 1,273, Verio 1,006, Sonata 567, Avanto 455), then Hitachi Echelon 387, GE Signa HDe 297 and
Excite 269, Philips Achieva 233. Kaggle's are newer and more mixed (Ingenia 741, Vida 447, Aera
400, Achieva dStream 327, Avanto fit 262, Signa Explorer 213, Signa Artist 208). The two share
Aera, Skyra, Signa Excite and Prisma. fastMRI widens the Siemens and Hitachi side and adds next to
nothing for Philips, Toshiba or Canon.

### Bone masks on fastMRI

Without mask targets the pretraining's bone map gets no supervision and the poses built from it
are arbitrary (first pretraining, 2026-10-13: 0 mask tokens, validation agreement 0.08, crops 33
px sagittal and 28 px coronal too high on average, turns of 8° against 3.4°, frame nod 14° against
3.5°). Poses are built from the bone map only (study frame, then crop turn, shift and scale); the
soft-tissue map only routes label reads. So bone masks were made for a fastMRI sample with
MRAnnotator:

- **First sample:** 1,230 of the 9,828 training studies, 12.5% as for Kaggle (550 of about 4,410),
  seed 20261013, allocated across scanner models by the square root of their study counts
  (Siemens 66% of the masks against 81% of the studies), one sagittal series each, the one with
  most images; 541 fat-suppressed, 689 not.
- **Chain, on the rented machine** (`~/.cache/knee-segmenters/fastmri1230`, segmenter environment
  rebuilt there with the local versions): conversion, MRAnnotator with `STEP=1.0` (14.3 minutes,
  0.7 s per knee), `segmenter_export_targets.py`, `segmenter_export_joint_targets.py` (4 mm
  closing, 135°, 2.5 mm) and `mask_measurements.py` of `scripts/adhoc/side_routed_session/`; the
  choice and conversion scripts (`fastmri_choose.py`, `fastmri_convert.py`) sit beside the sample
  (`fastmri_convert.py` is also in the [final-chain record](../../../scripts/adhoc/final_chain/README.md#2-mask-targets-for-the-pretrainings)).
- **Against the Kaggle 550, medians (Kaggle in brackets):** joint 27.3 ml (29.7), kneecap joint
  15.1 (18.3), femur 150.7 (167.1), tibia 97.2 (109.7), kneecap 12.1 (12.9), fibula 7.8 (8.6);
  kneecap front 54.3 mm (55.9), back 38.7 (40.0), top 53.5 (54.6), bottom 21.1 (21.3), joint gap
  6.1 (6.5 and 6.3). Two contact sheets (twelve scanners at random, and the smallest bones) showed
  clean bones and joint classes on every scanner model looked at.
- **Dropped, 10 knees:** two with no fitted plateau and eight with femur under 80 ml or tibia
  under 45 ml (one is not a knee image, the others are partial bones).
- **Targets:** `data/labels/prepared/anatomy-mrannotator-fastmri1230-joints` (all 1,230) and
  `anatomy-mrannotator-sample550-fastmri1230-joints` (550 Kaggle and 1,220 fastMRI volumes
  linked, `sources.json` with the rule and the dropped knees). With them, the fastMRI-only stage's
  first epoch reached 0.90 bone-map agreement on 1.81 million mask tokens, crops at +0.4 px
  (sagittal) and +2.7 px (coronal) and 3.7° turns.
- **Final:** bones on 1,946 fastMRI knees, `anatomy-mrannotator-sample550-fastmri1966-joints`,
  the `paths.anatomy_targets` of every pretraining config (second batch `fastmri_masks_extra.sh`,
  merged by `merge_masks.py`: [final-chain record](../../../scripts/adhoc/final_chain/README.md#2-mask-targets-for-the-pretrainings)).

## OAI

- **Data:** 9,437 OAI studies on the rented machine (`/workspace/data/external` with fastMRI).
  The baseline images were downloaded from signed links (`oai_download.sh`, `oai_fetch.py`) and
  laid out as one study per knee, `oai-<patient>-<side>` (`oai_layout.py`;
  [final-chain record](../../../scripts/adhoc/final_chain/README.md#1-external-data-on-the-host)); the 2026-10-21 unseen-data τ test
  fetched OAI baseline images from package 1228997 through signed links
  ([attention temperature](../../validation/results/pooling-and-attention/2026-10-22-attention-temperature.md)).
- **Unseen-data baseline (2026-10-21):** the existing τ 0.85 OAI predictions of the gen-1 models
  (which never saw OAI) scored against MOAKS on the 9 mapped labels, over the 1,339 baseline knees
  with complete MOAKS forms (labels from the OAI "MR Image Assessments" release in
  `data/external/oai-assessments/`): macro AUC 0.9126 for one model, 0.9156 for the five folds of
  one family and 0.9182 for all 15. The τ sweep that followed scored 1,335 knees
  ([attention temperature](../../validation/results/pooling-and-attention/2026-10-22-attention-temperature.md#unseen-data-oai)).
- **Soft-tissue masks:** 1,819 OAI DESS knees segmented with Kaggle's 263
  (`soft-tissue-nnunet-knee-dess265-oai`, the `paths.meniscus_targets` of every pretraining
  config; [soft-tissue segmenter](../../history/soft-tissue-segmenter.md); scripts
  `oai_dess_full.sh`, `oai_soft_tissue_final.py` in the [final-chain record](../../../scripts/adhoc/final_chain/README.md#2-mask-targets-for-the-pretrainings)).
- **Labels only where they carry signal (agreed 2026-10-15):** per source and label, the
  15-teacher pseudo-labels showed OAI, an osteoarthritis cohort, has almost no injuries (≥ 0.5 on
  0.1% of studies for Fracture and MCL, 0.2% Contusion, 2.4% ACL) and uninformative fluid labels
  (confident, < 0.1 or > 0.9, on 0.1% for Effusion and 0% for Synovitis; Baker's 13% against 55% on
  fastMRI), while menisci and OA have rates close to Kaggle's. Rule: fastMRI keeps all twelve
  labels; OAI keeps only Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA and PF OA
  (Baker's dropped with the fluid labels); OAI masks supervise the maps as before. The dropped
  labels are empty values on the OAI rows of the external label table, and the external dataset
  marks empty labels unobserved, so the masked loss skips them. Every external table from the
  second round on (`*-oai-soft-tissue`) follows it ([table builders](../../../scripts/adhoc/final_chain/README.md#4-label-tables)). On the first use (15 joint models, gold out of
  fold) the injury labels recovered: MCL +0.0091, Fracture +0.0042, Contusion +0.0040, Baker's
  +0.0018, PF OA +0.0064, with Lateral OA −0.0116 on 11 positives
  ([result](../../validation/results/teacher-student/2026-10-15-silver15-round-and-teacher-share.md)).

## Training on external studies

`extra_training` (`data_root`, `labels`) in a config trains on the external studies only: one
epoch is a full pass over the external table, batches of 8 (2,409 batches per epoch for the
19,265 studies, about 26–30 minutes on the RTX PRO 6000), class weights from the external labels'
mean soft target as prevalence (`fixed_class_balance(..., soft_prevalence=True)`), and no L2-SP
penalty in this stage (the training step skips the anchor when `extra_training` is set); the
Kaggle fine-tunes start from its last checkpoint (`train --initial-weights`) and anchor L2-SP to
those loaded weights, not to the original ImageNet or RadImageNet ones. The Kaggle validation fold
(fold 0 nominal) is only scored, never trained on; its scores are inflated because the teachers
that labelled the external studies trained on that fold. The joint mode that paired every Kaggle
batch with an external batch was retired
([retired pieces](../../history/fastmri-chain-retired-pieces.md#joint-training-with-external-batches)).
Persistent DataLoader workers of three concurrent runs over the 19k-study index filled the 251 GB
host and killed one run at epoch 2; those runs resumed with `trainer.persistent_workers: false`.
