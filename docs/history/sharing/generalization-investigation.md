# Sharing experiments: generalization and inference investigation

Investigated on 2026-09-14 using saved predictions, training archives, submission
records and Kaggle notebook logs. New computation was CPU-only tabular analysis;
no model execution, image decoding, training, GPU queries or tests were run.
The intervention results below come from the previously completed
[compartment diagnosis](2026-09-13-compartment-attention-diagnosis.md).

## Conclusions and scope

This investigation preserves the earlier five-model comparison recovered from
`1C` at `3b8c2cc`. The later [reflection/sigma-40 comparison](../../validation/results/2026-09-14-reflection-sigma-40-result.md)
added **0.892360 validation AUC and 0.936 public AUC**, selecting independent
heads at sigma 40 as the current baseline. Tables and correlations below retain
the original analysis cohort. The [transfer research](../../research/medial-lateral-transfer.md)
owns the current shortlist; historical proposals below are not scheduled work.

The subsequent [blinded disagreement and acquisition audit](oa-and-meniscus-disagreement-audit.md)
completed the report-review and metadata follow-up. It finds confident OA source
labels for explicitly low-grade report findings and shows a manufacturer-mixture
change that reverses 1B versus reflection locally. Removing the flagged report
cells does not remove 1B's original local advantage. These results refine the
hypotheses below without identifying the hidden test distribution.

The public reversal is real at the reported score precision, but its hidden
per-target cause is unobserved. The strongest combined explanation is that
weak-label validation rewards some changes that do not transfer to the official
annotation criteria and case mix, while repeated selection on one fold and
different optimization trajectories make close architectural comparisons noisy.
This is a limitation on causal confidence, not on the competitive value of the
score differences. A 0.006 public loss is consequential.

1C does not add an optional location embedding to 1B. It replaces independent
attention scores with a shared constraint, changing both representational
freedom and gradient coupling. Existing interventions show that the learned
factor is useful and that the constraint can preserve a good 1B solution.
Optimization and disease-specific contrast strength therefore deserve more
attention than a categorical conclusion that compartment sharing is bad.

The inference comparisons do not establish that a second GPU failed to improve
hidden-data throughput. Submission completion time mixes execution, queuing and
scoring. The saved visible notebooks contain only three studies. They do expose
substantial startup costs and confirm FP32 execution on both T4s. Static source
inspection identifies duplicated metadata work, shared CPU/I/O limits and
variable study costs as concrete candidates for poor scaling.

## Reproduction and artifacts

The archived CPU analysis command (`f076a09:docs/history/sharing-sources/analyze_sharing_generalization.py.txt`)
requires its [original runtime](../sharing-sources/README.md), not current
`main`. It joins the saved predictions to frozen
targets, checks study identity/probabilities, calculates target/subset metrics,
resamples report groups and fingerprints its inputs. It refuses an existing
output directory:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked python -m scripts.adhoc.analyze_sharing_generalization --output outputs/analysis/sharing-generalization-new
```

The completed evidence is in
[`outputs/analysis/s/sharing-generalization-20260914/`](../../../outputs/analysis/s/sharing-generalization-20260914):

- `analysis.json`, `target_metrics.csv` and the archived analysis source: primary
  comparison, bootstrap, epoch curves, input hashes and submission timing.
- `report_evidence_subsets.json`: supplementary target-specific coverage and
  source-agreement restrictions, with label-source hashes. Coverage means gold
  or Steven-v1 score unequal to 0.5; agreement additionally requires the original
  Steven-v1 and lixin target classifications to agree at 0.5. Evaluation still
  uses the frozen target, not a substituted label.
- `expanded_correlation.json`: the historical 12-checkpoint fixed-label cohort.
- `bundle_audit.json`: source/config/input-identity and export-integrity comparison.
- `notebook_timings.json`: structured measurements extracted from saved visible
  Kaggle notebook logs. These supplementary files are separate derivations; the
  main CLI regenerates the primary comparison, not those auxiliary audits.

Earlier GPU interventions and their intact-prediction parity are preserved in
`outputs/analysis/c/compartment-diagnosis-20260913/`. No intervention was rerun for
this investigation. New script lint/format and catalog checks are static checks,
not model correctness tests or performance benchmarks.
