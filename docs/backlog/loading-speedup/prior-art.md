# Prior art in past RSNA DICOM competitions

Public solutions checked 2026-09-17. Each shows a technique working in a Kaggle notebook; none is a
measurement on our data, and two of them have since been closed here by our own measurements.

- **`dicomsdl` for parsing and decoding.** The 2023 abdominal-trauma winner ships it as an offline
  Kaggle dataset ([repo](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution)); the 2024
  lumbar-spine MRI solution reads `ImagePositionPatient` through `dicomsdl.open` and pixels through
  `pixelData(storedvalue=True)`
  ([repo](https://github.com/yujiariyasu/rsna_2024_lumbar_spine_degenerative_classification)); the
  2022 cervical-spine solution uses it as well
  ([repo](https://github.com/louis-she/rsna-2022-public)). **Closed here:** our raw parser is
  3.7–4.2x faster ([closed](closed.md#ruled-out-by-arithmetic-or-by-the-environment)).
- **A parallel metadata pass, then a parallel pixel pass.** The 2025 intracranial-aneurysm solution
  (MRI, MRA and CTA) replaced SimpleITK's series reader with `joblib.Parallel` over
  `dcmread(stop_before_pixels=True, force=True)` for pose, spacing and rescale, sorted slices by
  their projection on the slice normal, then ran a second parallel pass for pixels
  ([repo](https://github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution),
  `nnunetv2/dataset_conversion/kaggle_2025_rsna/official_data_to_nnunet.py`). Independent support on
  MRI for the threaded header and frame reads that now ship.
- **Payload taken from bytes already read.** The breast-cancer winner and the 2022 cervical-spine
  solution locate the bitstream inside `PixelData` rather than opening or decoding again
  ([repo](https://github.com/dangnh0611/kaggle_rsna_breast_cancer)). Feeds the open
  [one-file-access candidate](candidates.md#bit-exact-batch-into-the-retrain); we cannot copy it
  directly, because selection happens after indexing and whole-file pre-reading is measured slower.
- **Not applicable: GPU decoding.** DALI and nvJPEG2000 carry those competitions' JPEG 2000 and
  JPEG-LS images. Our frames are uncompressed Explicit VR Little Endian, so there is nothing to
  offload.
- **Converting to PNG or npy for training**, which those solutions all do, cannot serve our
  inference at all and is worth 1–4 s per training run
  ([closed](closed.md#ruled-out-by-arithmetic-or-by-the-environment)).
- **Already done here.** Stopping before pixel data, recording the payload offset and sorting on a
  minimal tag set are standard in those solutions and are what `read_header` and `read_frame`
  already do.

## This competition, checked 2026-09-18

Self-reported by competitors, with no linked runs
([sweep](../../research/public-research.md#slice-density-and-the-board-2026-09-18-re-sweep),
[timings](../../inference/runtime-evidence-and-leads.md)).

- **The team leading both boards submits in about five minutes** and says reading is still 60–70%
  of that, so a loader-dominated profile is normal here and ours is not pathological. Their model
  is one single-fold CNN, which is where most of the difference lives.
- **Four CPU threads behind a queue took one competitor's preprocessing from about an hour to
  15 minutes.** Nothing to take: that is the step we are already past. We run 16 loader workers
  with a shared io pool, decoding overlaps the forward pass, and the box is
  [CPU-saturated at 95.4% with the GPUs at 24%](attribution.md#the-box-is-out-of-cpu-not-waiting-on-nfs),
  so more concurrency has nothing left to hide.
- **A bounded decode queue per GPU** is what the public two-T4 notebook keeps ahead of each device.
  Same shape as our open [bounded ordered payload window](candidates.md#bit-exact-batch-into-the-retrain),
  which is motivated by transient RAM rather than by wall — their use of it is weak support for the
  formulation, not evidence that it is fast.
- **One header scan reused across every model arm** bought that notebook 13.5% and 8.8% on two
  stages with identical predictions. We already parse each header exactly once per file per run, so
  the equivalent saving is
  [already banked](closed.md#ruled-out-by-arithmetic-or-by-the-environment); what it confirms is
  that re-reading headers per arm is the common mistake, which matters if we ever blend arms.
- **Two environment details worth keeping.** A recursive search for auxiliary files under the
  mounted competition directory walks every test DICOM; and `CUDNN_CONV_WSCAP_DBG=1024`, set before
  the first torch import, caps eager convolution workspaces under the T4's 16 GiB. Neither is a
  lever for us today — our lookups are explicit and the backbone is small — but the second is the
  guard to remember if a larger backbone ever ships.

## Published dataloader profiles for comparison

Our split is ordinary for a CPU-bound pipeline and unusual only among medical-imaging pipelines,
whose published gains mostly come from multi-epoch caching that single-pass inference cannot use.

- FFCV's diagnosis guidance: page caching gives diminishing returns past RAM, and a CPU-bound
  pipeline is fixed by cheaper or compiled transforms rather than by caching
  ([Bottleneck Doctor](https://docs.ffcv.io/bottleneck_doctor.html), accessed 2026-09-18).
- MONAI reports up to ~10x from `CacheDataset`, explicitly a multi-epoch effect
  ([Data docs](https://docs.monai.io/en/stable/data.html), accessed 2026-09-18). Inference reads each
  study once, so none of it transfers.
- Network-filesystem guidance converges on the same two levers we use: fewer file reopens, more
  client concurrency, because an open/close round trip can cost as much as reading megabytes
  ([AWS EFS performance tips](https://docs.aws.amazon.com/efs/latest/ug/performance-tips.html),
  accessed 2026-09-18).
