# Ongoing work

Active work and tasks planned for the short term. Follow the
[work-tracking rules](../AGENTS.md#work-tracking).

Follow the [competition repository architecture](architecture.md), with
on-the-fly pixel preparation and small development runs before full comparisons.
Use the maintained workflow in `scripts/pipeline/`; keep one-off
investigations in `scripts/adhoc/` and reusable utilities in `scripts/tools/`.
Tasks below are pending, in implementation order.

The [modeling notes](modeling.md#implemented-baseline) own the current
baseline recipe and fold-0 result; `configs/baseline.yaml` owns its defaults.
Do not duplicate those values here. Saved checkpoint configuration, rather than
the mutable baseline YAML, governs reproduction and submission.

Current strategy, selected on 2026-09-11: iterate on **fold 0** until near the
end of the competition. Additional-fold training and ensembling are deferred.
Keep fold membership, validation targets and the comparison recipe fixed.

- [ ] **Integrate the measured DICOM reader.** The native descriptor experiment
  is complete with exact outputs and a 1.357× median loader ratio. Follow the
  narrow production-integration scope in [optimization.md](../optimization.md)
  and retain the [experiment evidence](data-preparation.md#dicom-descriptor-experiment).

The unverified settings and results remain hypotheses to reproduce and improve,
not established requirements. When a task is completed, move durable
findings to its linked canonical document and remove the completed task here.
