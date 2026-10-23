# 2026-10-12 soft-tissue reads and ResNet-50 input sampling

Two autoresearch campaigns and the runs around them, 2026-10-10 to 2026-10-12, on fold 1 of
`all_5fold_scanner_v2`, seed 1110484336, ten epochs. **Adopted on main: the soft-tissue line
(seven-class map, routed third-stage cell read, bone routing), the sagittal fluid detail view
and aligned contrast mixing, trained on the usual labels.** Thirty-five distinct slices per
study gained +0.0014 locally and scored 0.959 public against 0.960 for its reference; it is not
on main. Per run, per label and incidents:
[soft-tissue reads run notes](../../../../autoresearch/campaigns/soft-tissue-reads/run-notes.md)
and [input sampling run notes](../../../../autoresearch/campaigns/rn50-input-sampling/run-notes.md).

## Soft-tissue reads campaign (ResNet-18, 34 runs)

Reference `rn18-soft-tissue-added-read-bone-routes`: ×4 0.928909, ×1 0.918680, 1,599 s. One
adoption, the sagittal fluid detail view (a 64 mm second crop of each sagittal fluid slice, same
slices, pose and augmentation, open to the menisci, the OA labels and ACL): ×4 0.929467, ×1
0.919142, 1,775 s.

- **Three trainings of one recipe** (the reference, an unchanged rerun, a near no-op): ×4
  0.928909, 0.928586, 0.928618; ×1 0.918680, 0.920187, 0.918625. Labels with fewer than 100
  positives move by 0.003 to 0.007 between identical trainings.
- **The cell read adds nothing measurable** on its five labels: four trainings without it match.
- **The detail view's three variants** (all twelve labels, three of five slices, another
  statistics draw) land at +0.0006, −0.0004 and −0.0003 against the plain recipe.
- **`trainer.deterministic` is off**; the shared epoch-curve pattern of the campaign's runs was
  the reference's own curve.

## What reaches the ResNet-50

| Recipe, ResNet-50, fold 1 | Local ×4 | Public |
|---|---|---|
| Run 24 (side-routed) | 0.933209 | 0.957 (S119) |
| Anatomy map and posed crops | 0.933265 | 0.959 (S126) |
| View poses | 0.933256 | – |
| Soft-tissue line with the detail view, usual labels | 0.933292 | 0.959 (S142) |
| The same with 0.25 teacher share | 0.935570 | 0.960 (S143) |
| Aligned contrast mixing and 35 distinct slices, 0.25 teacher share | 0.936965 | 0.959 (S146) |

Structure, routing, maps and poses gained on the ResNet-18 through its four rare labels (MCL,
Lateral OA, Contusion, Fracture) and did not carry; teacher labels and input sampling at
acquired resolution did.

## ResNet-50 input sampling campaign (14 runs)

Reference `rn50-detail-view-teacher-025` (the detail-view recipe at 0.25 teacher share): ×1
0.927290, ×4 0.935570, 2,650 s. Each run against the incumbent at its time.

| Run | ×1 | ×4 | Change at ×1 | Change at ×4 | Decision |
|---|---|---|---|---|---|
| Aligned contrast mixing | 0.928276 | 0.935270 | +0.0010 | −0.0003 | adopted by the user |
| 225 px, both grids centred, exact copies | 0.927010 | 0.934498 | −0.0013 | −0.0008 | rejected |
| 224 px last-stage phase shift, exact copies | 0.926056 | 0.934307 | −0.0022 | −0.0010 | rejected |
| 257 px, both grids centred, exact copies | 0.924376 | 0.933649 | −0.0039 | −0.0016 | rejected |
| Cosine schedule with two warm restarts | 0.927026 | 0.934700 | −0.0013 | −0.0006 | rejected |
| Ten distinct sagittal fluid slices, no detail view | 0.931363 | 0.935225 | +0.0031 | −0.0000 | rejected |
| Axial fluid kneecap tile at 100 mm | 0.926572 | 0.935067 | −0.0017 | −0.0002 | rejected |
| One more slice on every type, no detail view | 0.931204 | 0.935771 | +0.0029 | +0.0005 | above at both; not completed by the harness |
| Sagittal fluid 106 mm context view | 0.926233 | 0.934464 | −0.0020 | −0.0008 | rejected |
| Axial views at 100 mm | 0.926777 | 0.934166 | −0.0015 | −0.0011 | rejected |
| 35 distinct slices (7, 7, 6, 6, 6, 10), no detail view | 0.933540 | 0.936965 | +0.0053 | +0.0017 | adopted by the runtime |
| 3D series read as slabs (against 35 slices) | 0.933203 | 0.936887 | −0.0003 | −0.0001 | rejected |
| First and last slice in every bag (against 35 slices) | 0.929801 | 0.934592 | −0.0037 | −0.0024 | rejected |
| Manual: 31 slices with the detail view (against 35 slices) | 0.933350 | 0.936100 | −0.0002 | −0.0009 | not adopted |

- **Distinct slices spread over the series types is the one local gain**; ×1 figures of the slice
  runs are not like for like (more images at ×1 validation).
- **At ×4 the image counts are equal** (107,397 for aligned mixing, 107,292 for 35 slices: the
  stacks are exhausted), and the gain holds at every attention temperature:

| Temperature | Aligned mixing | 35 slices | Difference |
|---|---|---|---|
| 0.7 | 0.934275 | 0.935727 | +0.0015 |
| 0.8 | 0.934882 | 0.936751 | +0.0019 |
| 0.9 | 0.935155 | 0.937065 | +0.0019 |
| 1.0 | 0.935272 | 0.936968 | +0.0017 |

- **The public board does not confirm it:** S146 (35 slices and aligned mixing) 0.959 against
  0.960 for S143 (the reference). Aligned mixing alone has no public score (S144 failed).
- **Centred token grids do not help:** at 224 px both grids cannot be centred by sampling phase;
  225 and 257 px centre both and lose. A larger image (257 px, 32% more pixels) loses as well.
- **Native axial pixels do not help PF OA** (two runs).
- **Token grid copies between the third and fourth stage are half a cell off** in the recipe
  (8 px); the exact form is in the campaign folder and changed nothing measurable.

## What main carries and what has no run

Both configs (`configs/resnet50.yaml`, ResNet-18; `configs/refined-resnet50.yaml` (removed)) carry the
aligned-mixing recipe with the usual labels (teacher share 0). No run exists on exactly that
setting: the ResNet-50 runs of the recipe used the 0.25 teacher share (×4 0.935270) or the
unaligned blend (usual labels, ×4 0.933292, S142), and the ResNet-18 has not run aligned mixing.
A 35-slices run on the usual labels was stopped by the user after three epochs
(`outputs/training/rn50-thirty-five-slices-silver`, ×1 0.908047, 0.912190, 0.917417).

## Operations

- **The final ×4 prediction of a training** queued up to 32 batches (16 workers, prefetch 2,
  about 18 GB) and took available RAM to 2.6 to 4.9 GB for a minute; three runs were killed
  there by a memory guard that ignored free swap. `INFERENCE_NUM_WORKERS` is 8 since this
  close-out and the training's final prediction uses it.
- **GPU memory that does not fit beside the Windows desktop** is moved by the driver into shared
  system memory and the training runs at about half speed (card at P3, about 150 W);
  `nvidia-smi` shows only the dedicated part. Seen with a 22 GB training and desktop
  applications open.
