# Inference: distinguish the measured clocks

| Model | GPUs used | Loader workers, total | Studies per GPU batch | Submit → observed completion | Visible notebook measured total |
|---|---:|---:|---:|---:|---:|
| Reflection | 1 T4 | 8 | 2 | 13m 36s | 54.22s |
| 1A | 2 T4 | 4, two per GPU | 4 | 16m 07s | 90.62s |
| 1B | 1 T4 | 8 | 2 | 14m 36s | 51.98s |
| Original 1C | 1 T4 | 4 | 4 | 17m 38s | 54.57s |

Submission wait includes unknown queue/evaluation delays and approximately
30-second polling granularity. It is not the efficiency-prize runtime. Kaggle
staff explicitly define that runtime as notebook execution from start to end,
including setup, model loading and DICOM reading; GPUs are eligible.
[Staff clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733475).
The visible notebook totals are our instrumented portion of that execution,
not an official hidden-data runtime measurement.

All visible logs process three example studies; the competition describes roughly
1,300 hidden test studies. The 1A example partitions as two studies plus one,
never filling either batch limit of four. It establishes use of both T4s and
successful output, not sustained scaling. The failed original-1C batch-8 rerun
has no usable timing or score; memory pressure is plausible but its hidden error
does not establish an OOM diagnosis.

## Ranked runtime hypotheses

1. **Shared CPU and storage supply limits scaling.** Each GPU process has two
   loader workers, versus eight for the old single-GPU configuration. Two GPUs
   do not double the machine's CPU or storage bandwidth. DICOM enumeration,
   header parsing, decoding, geometry and tensor preparation can leave either
   GPU waiting. Study complexity and codec variation make hidden-data performance
   different from the three examples. This is a strong source-based hypothesis,
   not a measured fraction of hidden runtime.
2. **Startup and duplicated work offset compute gains.** The coordinator imports
   the model/inference stack, verifies the bundle and selects metadata before
   starting its worker timer. Each worker then initializes another interpreter,
   dependencies, dataset and model. On 1A, worker wall is 24.64s but the full
   prediction subprocess is 83.65s: about 59.01s lies outside that worker timer.
   The present logs cannot allocate that gap among imports, parent setup and
   shutdown. Each child builds an index for the entire test series metadata and,
   without a saved test image-count inventory, enumerates all series directories
   to count DICOMs even though study assignments are disjoint. Later reads enumerate
   selected series again. The duplication is confirmed statically; its cost is
   unmeasured on hidden data.
3. **Studies per batch is a poor compute unit.** A study expands into many
   triplet images and spatial tokens. Batch two may already provide a large CNN
   workload; doubling studies need not improve occupancy. It increases memory,
   preprocessing task granularity and waiting behind expensive studies. The
   current dense study-membership attention calculation also increases its
   masked work faster than linearly with study batch size, although the backbone
   may dominate. The share of time in this pooling is unknown.
4. **Static sharding leaves an uneven tail.** Study-count partitioning ignores
   acquisition counts, slice/header counts, codecs and image/token workload.
   Completion waits for the slower process. This is obvious with the two-plus-one
   example; the size of the effect on the hidden cohort is unmeasured. Persistent
   per-worker caches help repeated local validation more than a cold one-pass
   hidden run.
5. **T4 compute is being used in FP32.** Automatic precision selects native BF16
   when supported and otherwise FP32; it does not select FP16 on T4. This is
   confirmed in every saved Kaggle log. FP16 is a concrete optimization candidate:
   T4 has substantially greater mixed-precision Tensor Core peak throughput than
   FP32 throughput. Peak hardware figures are not an application speedup prediction.
   [NVIDIA T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf).

The small sharing heads do not materially reduce the expensive image backbone,
triplet count or spatial-token count. Replacing these heads should not itself
deliver a substantial inference speedup. Local saved validation prediction times
of roughly 32–34 seconds for 904 studies use different GPU hardware/precision and
a different cache context; they cannot forecast cold T4 notebook execution.
