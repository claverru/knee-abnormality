# View pose studies: scope and agreements

Studies behind "a crop per knee and per plane-contrast", written on 2026-10-08 and
collected here on 2026-10-10. The one-off scripts they used (session scratch under `/tmp`,
and `scripts/adhoc/meniscus_masks/`) were not kept: a WSL restart on 2026-10-08 wiped them. Their
outputs, tables and figures under `outputs/analysis/` were intact at the time; where a study names a
script (including the fast segmenter runner `fast_nnunet_knee.py`), it would have to be rewritten to
be run again.

**Terms.** *Pose*: a crop's position, rotation and scale. *Canvas*: the fixed image the loader
cuts before the model sees anything; it only has to be large enough and sharp enough. The crop is
made from the first look's 3D anatomy map of the knee, inside the canvas.

**Agreed so far (user, 2026-10-08).**

- All three planes get a pose (shift, turn, scale), per plane-contrast, not only sagittal and
  coronal.
- No hand-measured correction constants on the centre, plane or pose; improve the learning.
- The label loss must not choose the crop (it zoomed out in the past).
- The current model's attention is biased by the crops it was trained on; it cannot decide what
  each view should frame.
- Masks are needed on a subset of knees only, as today; a segmenter has to be good on some
  studies, with a way to tell which.
- Idea (user): if the soft-tissue segmenter only works on 3D series, make the masks on the 3D
  series in our data (836 series in 659 training studies: 268 Siemens DESS, about 147 studies
  with a sagittal 3D PD fat-suppressed series), supervise the map on those series' tokens only,
  and let the model generalise to 2D series. They are already model inputs (DESS in the axial
  fluid slot, 3D PD fat-suppressed in the sagittal fluid slot). Study 3 tests the segmenters on
  2D and on 3D series to tell which case holds. Transferring a 3D-made mask to the 2D series of
  the same study is an optional extra, checked in study 3b.

**Outside models: the rule is about the model, not its training data (user, 2026-10-08).** A
segmenter qualifies by its weights' own licence and free access. Checked: `aagatti/nnunet_knee`
weights MIT (model card; femur, tibia, kneecap, cartilage, menisci; trained on OAI-ZIB DESS);
SKM-TEA weights Apache-2.0 and ungated (`huggingface.co/arjundd/skm-tea-models` metadata; they
take qDESS echoes as input, so they suit our 3D DESS series). Regions inferred from the bone
masks are enough for the crops and the routing; real soft-tissue masks would only sharpen the
map's extra signal.

## Outputs by study

| Study | Outputs |
|---|---|
| 1. Knee size and per-knee scale | `outputs/analysis/knee-size-crops/` |
| 2. Attention per view in the knee frame | `outputs/analysis/view-attention/` |
| 3. Soft-tissue segmenters | `outputs/analysis/soft-tissue-segmenters/` |
| 3b. Census of our 3D series; alignment of 3D-made masks with 2D series | `outputs/analysis/soft-tissue-segmenters/3d-series/` |
| 4. Which targets are seen in which plane-contrast | `outputs/analysis/targets-by-view/` |
| 5. Do the errors sit where the crop cut the finding? | `outputs/analysis/errors-vs-coverage/` |
| Meniscus masks from DESS | `data/labels/prepared/meniscus-nnunet-knee-dess265/` |
