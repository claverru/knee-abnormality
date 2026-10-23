# 2026-10-19 torch.compile backbone

What `torch.compile` of the backbone gains on Kaggle's T4s once TensorRT is gone, per backbone
family, for one model and for ensembles. **The compiled backbone (default mode, static 64-image
chunks padded with zeros, channels-last) is 13–25% faster per batch than the CUDA-graph path on
every family and 32% faster on a 10-member ensemble, with predictions within 0.013. On one model
×4, compiled alone does not pay (its kernel launches take the loaders' CPU), but compiled inside
the existing CUDA graph cuts the worker wall by 30–42 s in paired same-session runs and S167's
session by 34 s on average over three fresh sessions per arm. Adopted on 2026-10-20 on the
`inference-speed` branch and in main on 2026-10-23 as the default `--backbone graph` export:
`use_fixed_backbone_chunks` compiles the backbone before its capture, and `scripts.pipeline.submit`
builds the compile cache on a Kaggle T4 and ships it in the bundle. Compiled graphs also keep
smaller pools: a 20-member ensemble that runs out of memory with plain graphs fits with compiled
ones.**

Code under test: `use_compiled_backbone` in `inference/engine.py` (probe bundles; production still
uses `use_fixed_backbone_chunks`). Labs: `profile_kaggle_inference` conditions `compile-lab`,
`compile-families`, `compile-ensemble` (one T4, prefetched batches, no loaders, warm-up then a
timed pass). Outputs under `outputs/profiling/kaggle-*compile*-20261019*`.

## Per family

One member of each gen-3 family from `gen3mixed-6fam-resplit-x30-eager-x4-tau085-tta`, 32 studies,
×4, one T4 (`kaggle-compile-families-20261019b-3`):

| Family | Graph, s/batch | Compiled, s/batch | Change | Cold compile | Max \|Δ\| |
|---|---:|---:|---:|---:|---:|
| FocalNet-T SRF | 1.599 | 1.206 | −25% | 37.1 s | 0.0005 |
| ConvNeXt-T | 1.311 | 0.995 | −24% | 26.1 s | 0.0013 |
| EfficientViT-B2 | 1.220 | 0.945 | −23% | 33.8 s | 0.0124 |
| ResNet-50 | 1.124 | 0.886 | −21% | 16.7 s | 0.0082 |
| NextViT-S | 1.515 | 1.261 | −17% | 29.0 s | 0.0008 |
| MambaVision-T | 1.099 | 0.951 | −13% | 24.7 s | 0.0009 |

- Compiled ResNet-50 gen1 fold 0, 912 studies: macro AUC 0.937886 against 0.937892 for the FP16
  graph path and 0.937887 for FP32 (`kaggle-compiled-resnet50-gen1-f0-20261019-4`).

## Ensembles

Five ConvNeXt and five ResNet-50 gen-3 members, 32 studies, ×4, one T4
(`kaggle-compile-ensemble-20261019-2`):

| | Graph | Compiled |
|---|---:|---:|
| Seconds per batch | 9.92 | 6.76 (−32%) |
| Compile | — | 29.7 s for the first ConvNeXt, 14.0 s for the first ResNet-50, 0.04–0.06 s for every later member of a family |
| Whole process | 380 s | 195 s |
| Max \|Δ\| | — | 0.0008 |

Compile cost grows with the number of families, not members: later members of a family reuse the
first one's compiled code. Compile keeps no CUDA-graph memory pool per member, so it is not
bounded by the 20-member graph limit.

## One model, end to end

S156 ×4, Kaggle 2×T4, 1,300 studies, `fp16-b4-w8`, one run each:

| Run | Command | Model ready | Forward per GPU | Loader wait per GPU | GPU-process CPU-s |
|---|---:|---:|---:|---:|---:|
| Branch, CUDA graph | 318.9 s | 39.1 s | 251, 275 s | 2.4, 2.3 s | 126 |
| Branch, CUDA graph, repeat | 303.2 s | 38.8 s | 259, 258 s | 2.3, 2.4 s | 128 |
| Compiled, cold | 295.7 s | 67.4 s | 192, 194 s | 34, 28 s | 178 |
| Compiled, cold, repeat | 299.8 s | 66.5 s | 194, 202 s | 37, 27 s | 183 |
| Compiled, shipped cache | 306.9 s | 57.3 s | 196, 199 s | 51, 43 s | 190 |
| Compiled, shipped cache, repeat | 307.9 s | 57.0 s | 194, 201 s | 54, 44 s | 185 |

- **The loaders become the limit.** Forward drops by 55–75 s per GPU, but loader wait rises from
  2 s to 28–54 s, spread over the whole run.
- **Compiled kernels launch from Python:** the GPU processes use about 55 CPU-s more, which the
  loaders lose on four CPUs.
- **Cache:** shipping the inductor and Triton caches (16 MB, built on a T4) cuts model load from
  37 to 23 s. A first cache held only the inductor part and gained nothing; Triton's compiled
  kernels sit in a separate directory. Shipped as 892 files it adds 3–6 s to bundle verification;
  it would have to ship as one archive.

## Compiled inside the CUDA graph, paired

The compiled backbone captured in the existing CUDA graph (`use_fixed_backbone_chunks` compiling
`_backbone_features` before the warm-up and capture), against the graph path, both arms in one
Kaggle session (`paired-*` conditions: page cache of the studies and the bundle dropped before each
arm), shipped inductor and Triton cache, S156 ×4, 1,300 studies, `fp16-b4-w8`:

| Session | Arm | Worker wall | Forward per GPU | Loader wait per GPU | Model load |
|---|---|---:|---:|---:|---:|
| Graph first (`kaggle-s156-paired-ab-20261019-1`) | graph | 334.1 s | 257, 266 s | 2, 2 s | 7.6 s |
| | compiled in graph | **291.8 s** | 201, 215 s | 14, 3 s | 21.5 s |
| Compiled first (`kaggle-s156-paired-ba-20261019-1`) | compiled in graph | **279.4 s** | 195, 203 s | 19, 7 s | 21.9 s |
| | graph | 309.9 s | 257, 264 s | 3, 2 s | 10.2 s |

- **−42.3 and −30.5 s of worker wall, both orders;** two graph arms in one session differed by
  4.9 s. Predictions against the graph arm: max \|Δ\| 0.0127–0.0128, mean 7.9e-5.
- GPU-process CPU falls back near the graph path's, so the loaders keep up (wait 3–19 s).
- Five runs of this variant completed; the earlier 30-minute "hang" coincided with a Kaggle session
  that also stayed "running" for a notebook that fails at once, so it is attributed to Kaggle.
- Without the shipped cache model load is about 14 s longer (cold compile).

## S167 in fresh sessions

S167 (`convnext-silver15-share075-ft-f0-x4-tau085`) in three fresh Kaggle sessions per arm, rotating
order, 1,300 studies, `fp16-b4-w8`; session time is Kaggle's `RUNNING` to `COMPLETE`, polled every
10 s. A is the submitted bundle; B is its own code with the branch's inference files
(`engine.py`, `predict_worker.py`, `predict.py`, the buffer pool and the page pre-read) and its
checkpoint slimmed to FP16 network weights (213.9 → 54.1 MiB); C is B compiled inside the graph
with the shipped cache. S167's config cannot load under current main (its preprocessing identity
predates main's), so the branch was ported onto its code (`outputs/profiling/kaggle-s167-real-*`).

| Arm | Session, three runs | Mean | Command, mean | Worker wall, mean | Worker start to model import |
|---|---|---:|---:|---:|---|
| A, submitted | 426.6, 467.4, 427.0 s | 440.3 s | 396.3 s | 375.3 s | 31.7, 43.0, 31.8 s |
| B, branch | 497.7, 427.0, 376.2 s | 433.6 s | 380.0 s | 366.8 s | 17.3, 15.1, 12.5 s |
| C, branch compiled | 426.6, 406.2, 386.0 s | 406.3 s | 369.2 s | 353.5 s | 14.5, 14.5, 13.2 s |

- C ties A in the first round and beats it by 61 and 41 s in the others; its forward falls from
  about 270 to 164–194 s per GPU, and most of the rest goes to loader wait (98–143 s against
  A's 20–61 s).
- B's earlier start (the page pre-read cuts the import phase from 16–21 s to under 0.1 s) is
  mostly lost to loader wait; its −6.7 s per session is inside the host spread.
- **The adopted code with the cache built by the submission job** (`_build_compile_cache` on
  Kaggle's visible test, 16.3 MB): one fresh session 396.3 s, command 358.9 s, model load 22.4 s
  (a cold compile takes about 37 s), predictions within 0.0126 of the submitted bundle's
  (`kaggle-s167-adopted-20261020-3`).

## Three families in fresh sessions

Each family's single-model bundle as exported (A) against the same weights on the adopted code
with a compile cache built by the submission job's stage (B; ConvNeXt's B is S167's C above), three
fresh Kaggle sessions per arm in rotating order, 1,300 studies, `fp16-b4-w8`. ResNet-50:
`fp32-resnet50-gen1-fold-0` (gen-1 code); NextViT: `probe-nextvit-f2-x4` (main's code at 10-18).
Outputs `outputs/profiling/kaggle-{s167,rn50,nextvit}-real-*-20261020-*`.

| Family | A session, mean | B session, mean | Δ session | Δ at 69.2 s per 0.001 | A worker wall | B worker wall |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-T (S167) | 440.3 s | 406.3 s | −34.0 s | 0.0005 | 375.3 s | 353.5 s |
| ResNet-50 | 467.7 s | **352.3 s** | −115.4 s | 0.0017 | 371.9 s | 304.6 s |
| NextViT-S | 474.3 s | 413.2 s | −61.1 s | 0.0009 | 377.7 s | 314.8 s |

- B's three ResNet-50 sessions span 345.9–355.6 s, against 417.7–508.0 s for A.
- Forward per GPU on B: ResNet-50 146–174 s, ConvNeXt 164–194 s, NextViT 204–243 s.
- The families differ in more than the backbone (ResNet-50's bundle carries gen-1 code), so the
  cross-family order holds for these bundles, not for the architectures alone.

## First submission: S193

S193 (2026-10-22 01:16 UTC) is the ConvNeXt-Tiny student distilled from S188
(`convnext-s188-student`, ×4, τ 0.5, no mirror TTA), exported from its training tree and ported to
the adopted inference (`convnext-s188-student-x4-tau05-fast`, 54.1 MiB FP16 checkpoint). It scored
**0.960 public**, 4th of five 0.960s.

| Stage | S193 | S167 | S156 |
|---|---:|---:|---:|
| Compile-cache build (upload, notebook on the visible test) | 38 + 123 s | — | — |
| Final upload and dataset ready | 38 s | 45 s | 54 s |
| Visible notebook | 122 s | 91 s | 424 s |
| Submitted to scored (hidden rerun plus Kaggle's queue) | 544 s | 393 s | 424 s |

- In the visible run the cache hit: model load 22.0 s (a cold compile takes about 37 s); bundle
  verification 12.4 s on the gcsfuse mount.
- Submitted to scored carries at least 128 s of queue noise. 544 s is within that of S156 and far
  from S165's 2,690 s; the board's own runtime appears in the next efficiency-board version.

## Second submission: S195

S195 (2026-10-22 09:46 UTC) is the ResNet-50 v2v3 gen-4 student (`resnet50-v2v3-gen4`, ×4, τ 0.85,
no mirror TTA), exported on the rented machine (its checkpoint records `/workspace` paths) and
ported to the adopted inference (`resnet50-v2v3-gen4-x4-tau085-fast`, 47.1 MiB FP16 checkpoint). It
scored **0.964 public**, 2nd of thirteen 0.964s, above every other single model at 0.964 (S167
4th).

| Stage | S195 | S193 | S167 |
|---|---:|---:|---:|
| Compile-cache build (notebook on the visible test) | 154 s | 123 s | — |
| Final upload and dataset ready | 37 s | 38 s | 45 s |
| Visible notebook | 121 s | 122 s | 91 s |
| Submitted to scored | 483 s | 544 s | 393 s |

- The cache notebook sat in Kaggle's queue twice and was relaunched (records `-r2`, `-r3`).
- Visible run: model load 18.6 s with the cache, verification 12.6 s, τ 0.85 confirmed in the
  runtime log.

## A graph-sized ensemble

20 members of the S188 pool (7 ConvNeXt-T, 7 NextViT-S, 6 ResNet-50, FP32 weights) as one bundle,
×4, batch 2, 8 workers, mirror TTA, 40 studies, Kaggle 2×T4, one session per arm
(`outputs/profiling/kaggle-x20-compiled-graph-20261023`):

| | Plain graphs | Compiled graphs | `--backbone graph` (compiled, FP16 backbone) | `--backbone eager` (FP16 backbone) |
|---|---|---|---|---|
| Result | CUDA out of memory | complete | complete | complete |
| Graph pools | 9.86 GiB (from the error) | about 6.25 GiB (reserved 9.76 GiB after capture, 3.51 GiB allocated) | — | none |
| Peak reserved per GPU | above 14.4 GiB | 12.29, 12.60 GiB | 11.51, 11.58 GiB | 8.61, 9.20 GiB |
| Model load | — | 96 s, cold | 81 s, cold | 12 s |
| Prediction, 20 studies per GPU | — | 163, 152 s | 139, 150 s | 223, 224 s |
| Max \|Δ\| against compiled graphs | — | — | 0.0020 (mean 7.5e-5) | 0.0021 (mean 8.2e-5) |

The last two columns are main's code at 2026-10-23 on the same members
(`outputs/profiling/kaggle-x20-{graph,eager}-mode-20261023`), without a compile cache; the largest
batches held 272 and 296 images.

- **Compiling cuts the per-member pool from about 0.49 to 0.31 GiB.** S166 ran 20 members with
  plain graphs; this mix did not fit, so the graph limit depends on the families, not on a member
  count.
- **The FP16 backbone takes another 0.8–1.0 GiB off the peak** and leaves 3 GiB of headroom on
  40 studies; predictions move by at most 0.002.
- **Eager forwards are about 50% slower** (223 against 139–150 s per GPU) but load in 12 s; on
  40 studies the cold compile evens the two walls (250 against 254 s), and the compile cache and
  a full test set favour the graph.

## Rejected

| Variant | Result |
|---|---|
| `max-autotune-no-cudagraphs` | 66.6 s cold compile, slower per batch (1.09 against 1.04 s) |
| C++ wrapper (`cpp_wrapper`) | GPU-process CPU 172 against 178 CPU-s, cold model load 62.5 against 37 s; it needs a `libcuda.so` link on Kaggle's image (only `libcuda.so.1` is installed) (`kaggle-s156-branch-compiled-cpp2-20261019-2`) |
| `reduce-overhead` (CUDA graphs on top of compile) | 1.149 against 1.158 s per batch; brings back the per-member graph pools |
