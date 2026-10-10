# 2026-10-10 view sizes: sagittal and coronal fields on the recipe in force

**Sagittal fluid at 84 mm is the only field setting that moves the model, and the recipe already
has it.** Four runs changed the in-plane field of the sagittal or coronal views on the recipe in
force; none beat it. The coronal fields and the sagittal T1/PD field are level at ×4, and a
second view on sagittal fluid that takes slices from the first lost 0.0027. Nothing was adopted
and no code was kept. Per-epoch curves, single-view readouts and the full per-label tables are in
the [view-poses run notes](../../../autoresearch/campaigns/view-poses/run-notes.md).

## Setup

- **Recipe:** main at `7f7c30f` (meniscus map and posed canvases, the
  [recipe in force](../../ongoing/state.md#recipe-in-force)), ResNet-18, `configs/training.yaml`
  with only the view table changed (`data.series_types[*].views[*].field_mm`, and for the last
  run the number of views and their slices).
- **Split and seed:** `all_5fold_scanner_v2`, validation fold 1 (874 studies), seed 1110484336,
  one run per setting.
- **Yardstick:** the recipe's own run `sagittal-fluid-canvas-fine` (×4 0.928121) and a retraining
  of the same code and configuration, `rn18-reference-retrained` (×4 0.928120). The two differ by
  up to 0.0046 on a single label at ×4 and by 0.0010 at best ×1, so runs are judged at ×4 and
  per-label changes against both.
- **Scores:** ×4 is the saved prediction of the selected checkpoint at attention temperature 1,
  without reflection averaging; public scores are ×4 at temperature 0.9, batch 4.

## Runs

Fields in millimetres; 93 is 93.33 (0.417 mm per pixel) and 84 is 0.375 mm per pixel. Axial is
120 mm throughout.

| Sagittal T1/PD | Sagittal fluid | Coronal T1/PD | Coronal fluid | Run (`outputs/training/`) | ×4 | Best ×1 | Public |
|---|---|---|---|---|---|---|---|
| 84 | 93 | 84 | 93 | parent, view-poses run 3 | 0.926563 | 0.917013 | – |
| 93 | 84 | 84 | 93 | **recipe**, view-poses run 7 | 0.928121 | 0.919392 | 0.955 (S129) |
| 93 | 84 | 93 | 84 | `rn18-coronal-fluid-canvas-fine` | 0.928026 | 0.918941 | 0.954 (S131) |
| 93 | 84 | 93 | 93 | `rn18-coronal-both-93mm` | 0.928226 | 0.918769 | 0.954 (S133) |
| 93 | 93 | 84 | 93 | `rn18-sagittal-both-93mm` | 0.926621 | 0.918390 | – |
| 93 | 72 with 3 slices and 112 with 2 | 84 | 93 | `rn18-sagittal-fluid-72-and-112mm` | 0.925384 | 0.918100 | – |

Not run: both sagittal views at 84 mm (prepared, one line from the recipe).

## Findings

- **Sagittal fluid at 84 mm: +0.0015 at ×4.** Four trainings have it at 84 mm (three
  configurations) and two at 93 mm; all eight run-against-run differences lie between +0.0014
  and +0.0017. Per label, repeating in at least seven of the eight and larger than the
  retraining's difference: MCL +0.0056, Contusion +0.0043, Fracture +0.0038, Lateral OA +0.0034,
  Synovitis −0.0023.
- **Sagittal T1/PD is level.** Both sagittal views at 93 mm equals the parent (+0.00006), so the
  swap that made the recipe gained through the fluid view, not through the wider T1/PD view. One
  pair of runs.
- **The coronal fields are level.** Three settings within 0.0002 at ×4, each one board step below
  the recipe on Kaggle. They trade labels, one run each: coronal T1/PD at 93 mm gave Fracture
  +0.0053 and Baker's +0.0024 and cost Lateral Meniscus −0.0042.
- **Why coronal cannot go finer (measured on 549 mask knees):** the two menisci span 85 mm side
  to side and the medial collateral band reaches 53 mm medial of the plateau centre, so an 84 mm
  coronal crop holds the whole band in 15% of knees and a 93 mm one in 56%. In the sagittal
  plane both menisci, the joint zone and the notch fit a 64 mm box.
- **Single-view readouts do not predict the model.** Each sagittal and coronal view read alone
  prefers 93 mm (sagittal fluid 0.860 against 0.851, coronal T1/PD 0.742 against 0.721), while
  the model prefers 84 mm for sagittal fluid and is flat elsewhere; per-label directions agree
  on 2 to 7 of 12 labels. Do not set fields from them.
- **A second view that takes slices loses.** Sagittal fluid as a 72 mm view with three slices
  and a 112 mm view with two: −0.0027, with MCL −0.0177, Contusion −0.0089 and Baker's −0.0048
  carrying −0.0026 of it, and ACL +0.0047 the one gain. The same split lost 0.0020 on the recipe
  of 3 October ([learnable views](2026-10-03-learnable-views-campaign.md)). It was not a clean
  test of two scales: the two views pick slices independently and overlap at ×4, the first look
  at ×4 reads their end slices, and contrast mixing and the feature-statistics augmentation
  blend across the two zooms. A placed view cannot be as wide as `data.reference_field_mm`
  (120 mm), which is why the wide view is 112 mm.
- **Cost of the second view, unexplained:** 2,232 s against about 1,500 s at the same 32 images
  per study. Loader time per study is equal and nothing in the code scales worse than linearly;
  seven- to twelve-view runs before the posed canvases were not slower.

## As ensemble members

Means of the saved ×4 probabilities on fold 1:

| Mean of | ×4 | Public |
|---|---|---|
| recipe + its retraining | 0.92833 | – |
| recipe + coronal swap | 0.928549 | 0.955 (S132) |
| retraining + coronal swap | 0.92861 | – |
| retraining + coronal swap + both coronal 93 | 0.92872 | – |
| all seven trainings | 0.92841 | – |

- Pairs that differ only in coronal fields correlate less than two trainings of one
  configuration (logit correlation 0.9943 against 0.9969) and average to about +0.0004 more
  (bootstrap probability above zero 0.84 for the best triple). Sagittal variants correlate less
  still (0.9902) but start 0.0015 or more lower and net out below the best member.
- Reading each label from its best run is selection optimism: 0.93013 chosen and scored on the
  same studies, −0.0004 against the recipe when chosen on one half and scored on the other (500
  splits).
- S132 could not be one bundle: an ensemble bundle requires one `data` section, and the members'
  view tables differ. It was one notebook running the two existing bundle datasets in turn and
  averaging the two files.

## Artifacts

- Runs under `outputs/training/` as named above, each with its source snapshot, effective
  configuration, per-epoch metrics and ×4 predictions; the parent and the recipe under
  `outputs/autoresearch/view-poses/experiments/`.
- Submission records under `outputs/submission/records/`:
  `rn18-coronal-fluid-canvas-fine-f1-x4-tau09.json` (S131),
  `rn18-ensemble-sag-fine-coronal-fine-f1-x4-tau09.json` (S132),
  `rn18-coronal-both-93mm-f1-x4-tau09.json` (S133).
- The scripts were one-offs in a session scratch folder and were not kept; every figure here
  comes from the saved predictions and logs and can be recomputed from them.
