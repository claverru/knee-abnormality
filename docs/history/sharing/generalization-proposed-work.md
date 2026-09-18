# Proposed discriminating work

These historical proposals explain how the investigation could discriminate
between causes. The [later research shortlist](../../research/medial-lateral-transfer.md)
supersedes their priority order. Current code uses independent heads at sigma 40;
no training, GPU diagnostics or submissions are scheduled by this record.

## Generalization and architecture

1. Preserve frozen full-fold AUC, but report its target contributions, report
   coverage/source agreement, study complexity and available site/protocol strata
   alongside it. Predetermine these diagnostics before the next architectural
   comparison; do not keep selecting whichever post-hoc subset agrees with public.
   Inspect disagreement cases, especially contusion/fracture and lateral OA,
   against report text and explicit official severity definitions. This can expose
   labeling mechanisms without pretending that another LLM is gold truth.
2. When training resources become available, separate initialization and data-order
   randomness, copy identical shared initial tensors into matched comparisons and
   compare fixed epochs as well as selected best epochs. A matched seed comparison
   is the first step; additional seeds/folds would resolve different uncertainties
   but remain deferred under the current project strategy. An independent
   adjudicated validation panel would address endpoint validity more directly than
   more report-derived folds, if it becomes feasible.
3. The most informative targeted architecture experiment is to start from the
   saved 1B representation and compare its independent scorer against the known
   near-lossless shared-meniscus-contrast projection, with matched fine-tuning
   conditions. Measure whether the constrained solution deteriorates during
   adaptation. This distinguishes a difficult training path from immediate
   capacity loss; it is a new experiment, not a replacement for the original
   from-pretrained comparison.
4. A separate candidate is shared anatomical contrast with disease-specific
   strength, motivated by 1B's 0.781 versus 0.474 contrast SD and the opposing
   meniscus/OA response to scaling. It would relax 1C's identical-strength
   assumption. Also measure per-family gradient norms/alignment when GPU diagnosis
   is available before choosing learning-rate or gradient-balance changes.
5. The original analysis kept reflection/sigma 80 as the independent-head reference
   and sigma-40 1C as an equally scoring public candidate. The completed
   reflection/sigma-40 run later surpassed both publicly. Do not restore 1B merely because its original
   fold AUC is higher, or reject all compartment sharing because original 1C loses
   locally. Compare architecture at matched sampling before attributing sigma 40's
   benefit specifically to the factor. Avoid using further public probes as an
   unlimited substitute for validation.

## Efficiency

Deferred by user decision after this investigation, then resumed on 2026-09-15 as
[loading speedup](../../backlog/loading-speedup/INDEX.md).

Create one Kaggle profiling notebook with a fixed checkpoint and a representative
set of training studies available on Kaggle, including expensive study/codec
tails. The local machine does not need two GPUs for that controlled measurement.
Use the same source, precision and preprocessing for a 1-GPU/2-GPU ×
2-study/4-study batch matrix with four total loader workers; add the old
1-GPU/eight-worker condition separately. Record startup and both cold and warm
passes, counterbalance run order and repeat enough to see machine variability.
This does not require consuming scored competition submissions.

Measure imports/setup, full metadata indexing, per-batch loader wait, DICOM/header
work, transfer, CNN, pooling, write and per-GPU completion tails. Record actual
images/tokens and study counts, not only study batch size. Use CUDA events or
synchronization for device timing because GPU launches are asynchronous.
[PyTorch timing guidance](https://docs.pytorch.org/docs/main/notes/cuda.html).

If the loader dominates, prioritize one shared/prefiltered metadata inventory,
eliminating repeated header/enumeration work, CPU-budget-aware preparation and
overlap with GPU work. If partition tails dominate, use cost-aware assignment or
a dynamic work queue. If GPU compute dominates, measure FP16 with sensitive
geometry/pooling operations retained in FP32 and validate saved prediction AUC
and numerical stability before adopting it. Batch or chunk by actual image/token
budget to decouple study size from peak CNN memory.

ONNX/TensorRT can accelerate the regular 224×224 image backbone, but do not
eliminate Python DICOM reading, startup or variable-study aggregation. The backbone
is the natural first export boundary; dynamic image batches need suitable engine
profiles. Choose that work after measuring the time fraction it can affect.
For example, if only 30% of notebook time benefits from a second GPU, even ideal
halving of that part gives only 1/(0.7 + 0.3/2) ≈ 1.18× overall speedup, before
extra process costs.

Efficiency also rewards accuracy. With D = maxAUC − benchmark, the official
minimized score is −AUC/D + runtime/32400. An AUC improvement δ has the same score
benefit as saving 32400δ/D seconds. For a hypothetical D = 0.45, +0.001 AUC is
worth 72 seconds and +0.006 is worth 432 seconds. This example does not assume
the final private maximum is known. Optimizing runtime while sacrificing several
AUC decimals may lose the efficiency prize as well as leaderboard position.
[Official efficiency formula](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation).
