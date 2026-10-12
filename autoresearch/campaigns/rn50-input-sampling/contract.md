# ResNet-50 input sampling campaign contract

Written 2026-10-11 with the user. Started 2026-10-11 21:26 CEST.

## Objective

Raise the macro AUC of a Refined-RadImageNet ResNet-50 by changing what the model is fed: which
slices, which fields, at which pixel size, and how faithfully the input is placed and mixed. The
development backbone is the ResNet-50 itself, because three generations of ResNet-18 gains did
not reach it ([prior work](prior-work.md)).

The reference and first incumbent is `outputs/training/rn50-detail-view-teacher-025` (trained by
the keeper on 2026-10-11 at the user's request: fold 1 of `all_5fold_scanner_v2`, seed
1110484336, ResNet-50, ten epochs, the sagittal fluid detail-view recipe, 0.25 teacher share from
the five ResNet-50 teachers). Its scores and training time are in [prior work](prior-work.md).

Eleven forced seeds, the user's three first, then the researcher without end:

1. `aligned-contrast-mixing`: the donor contrast is aligned to the target before blending (user).
2. `centred-token-grid-by-last-stage-phase`: the last stage samples the odd positions, so the
   7 by 7 tokens sit at 16 to 208 px of the 224 px crop (user).
3. `centred-token-grid-at-257-px`: the image is 257 px, where the grid is symmetric (user).
4. `sagittal-fluid-ten-distinct-slices`: the detail view's five images become five more sagittal
   fluid slices.
5. `axial-fluid-kneecap-tile-at-native-pixels`: the detail-view mechanism on the one plane fed
   below its acquired resolution.
6. `one-more-slice-on-every-series-type`: the detail view's images spread over the six types.
7. `sagittal-fluid-context-view-106-mm`: the second sagittal fluid view shows the anatomy the
   84 mm crop cuts.
8. `axial-views-at-100-mm`: both axial views at 100 mm instead of 120 mm.
9. `thirty-five-distinct-slices`: seeds 4 and 6 together.
10. `sagittal-fluid-kneecap-and-pouch-tile`: a 93 mm tile placed on the kneecap.
11. `three-dimensional-series-read-as-slabs`: a pick of a 3D series is the mean of its three
    decoded thin slices.

## Settled

- **The backbone is `resnet50.a1_in1k` with the Refined-RadImageNet weights.** No other backbone.
- **Labels are 0.75 silver + 0.25 raw joint teacher** from
  `data/labels/prepared/rn50-distillation-teachers`. The share and the table are not subjects.
- **Prediction settings are not subjects:** ×1 validation and ×4 prediction as the recipe has
  them; no test-time augmentation, no ensembling, no weight averaging.
- **Everything is computable for an unseen knee** from its own images through the model.
- **Ten epochs.** Never fewer; the epoch count is not a subject.

## Principles

1. **A seed feeds the model acquired information it does not have**, or corrects how the input
   is placed or mixed. Showing the same pixels again at another magnification is not a gain: the
   64 mm detail view is finer than both acquired axes in 98% of sagittal fluid series.
2. **A number set by hand comes from a cited measurement** (the acquisition tables in prior
   work, a measured extent, an earlier run), never from a sweep on the validation labels.
3. **A seed states its images per study** in training and at ×4, and its expected training time.
4. **A seed can be proved active** by metrics it names, at most six. A mechanism that acts only
   in training may name a training metric the recipe already logs.
5. **A superseded behaviour is removed**, not kept behind an option.
6. **Bundles are allowed** when a seed names the runs it combines.

## How a result is read

One run is weak evidence. On the ResNet-18, three trainings of one recipe gave ×4 within ±0.0003
and ×1 within 0.0015, and labels with fewer than 100 positives moved by 0.003 to 0.007. The
ResNet-50's own spread is not known; the second forced run of the campaign that leaves the
recipe's scores unchanged in expectation will be read as its first estimate. A per-label effect
is read on the ×1 AUC averaged over epochs 6 to 10 together with ×4, on the eight larger labels
first: gains that sat only on MCL, Lateral OA, Contusion and Fracture did not carry from the
ResNet-18 in three of three steps.

## Seeds the user ordered outside the scope

- `cosine-schedule-with-warm-restarts` (2026-10-11): the learning-rate schedule restarted twice
  within the ten epochs. The optimizer, its rates and the epochs stay. It opens no direction for
  the researcher.

## Out of bounds

Another backbone; ensembling, weight averaging, test-time augmentation; fewer epochs; changes to
the labels, the teacher share or the teacher table; a segmenter or a prepared target at
validation or prediction; trained models in the loader; new routing, attention or readout
structure on the same features (the soft-tissue reads campaign's subject); image sizes other
than 224 px outside the seeds `centred-token-grid-at-257-px` and
`centred-grids-at-225-px-exact-copies`; multi-seed, confirmation or control
runs not asked for by the user; choosing anything on the campaign fold's validation labels;
Kaggle submissions.

## Cost

**The budget is 0.75 to 1.25 of the reference's training time.** A run adds no time its
mechanism does not need: no new diagnostics in the training step, no host synchronisation, no
second evaluation of maps, routing or tokens. A ResNet-50 run on this recipe peaks at about
16.5 GB of GPU memory and leaves about 7 GB of the machine's 39 GB of RAM free at its tightest:
a seed that raises the images per study states its GPU and RAM estimate, and the developer runs
nothing heavy beside a training.

**Exception by the user (2026-10-12), for the seed `centred-token-grid-at-257-px` only:** training
time up to 1.45 of the reference's and GPU memory up to the card's 24 GB, to see whether a larger
image helps the ResNet-50. The sealed training ceiling is 3,843 s since then (first set to 3,578 s;
raised when the first epoch took 375 s, a pace of 1.40); every other seed
is still held to 1.25 by its epoch pace.

## Duration and adoption

Started with `start`; no end. After the forced seeds the researcher keeps proposing inside this
contract until the user stops it. Adoption is by the runtime under the
[shared rule](../../protocol.md#adoption-policy): strictly above the incumbent at both ×1 and
×4. The keeper repairs and requeues; it does not adopt.

## Evaluation

Every run is read against the incumbent at ×4 and ×1, per label from largest gain to largest
loss, the eight larger labels and the four rare ones apart, with the images per study, the
training and prediction seconds, the peak GPU memory and the lowest available RAM.
