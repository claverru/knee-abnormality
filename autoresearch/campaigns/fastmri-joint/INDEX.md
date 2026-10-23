# fastMRI joint training campaign

**Written 2026-10-12, not started.** The campaign changes how the pseudo-labelled fastMRI half
enters the training of a Refined-RadImageNet ResNet-50 that trains on the Kaggle fold and on
fastMRI together, on fold 3 of `all_5fold_scanner_v2` with fold 3 of `rn50-joint-fastmri-fold-3`
as the reference. Nine forced seeds in the order of the ranking by potential agreed with the user, then the
researcher without end.
Adoption by the runtime alone: strictly above the incumbent at ×1 and ×4. Training time 0.75 to
1.25 of the reference's. Worktree `knee-abnormality-fastmri-pretraining`, branch
`fastmri-pretraining`.

The [remote training host](../../../docs/backlog/refactors/remote-training-host.md) option is
built and `validate` accepts the campaign; the recipe is this worktree's `configs/training.yaml`.

**Ready to start; the start is the user's call.** Checked on 2026-10-12: the launcher end to end
with a trivial command, the joint recipe for one epoch on 64 studies from this worktree's code on
the rented machine (8 optimizer steps for 16 batches), and the fifteen hashed preprocessing
sources identical to the reference's. No calibration, training or comparison has run through
the training host yet.

Two places exist for the seeds, since a campaign may edit only `src/knee_abnormality`: the module
returns the gradient-accumulation schedule and says which batch is a fastMRI batch
(`configure_callbacks`, `extra_batch`), and the data module exposes the frame the class balance is
computed from (`class_balance_studies`).

| Read when | Document |
|---|---|
| Checking the objective, what is settled, the principles a seed follows, how a result is read, what is out of bounds, what is held for the user, cost and adoption | [Contract](contract.md) |
| Reading the first joint run against the plain control per epoch and per label, the pseudo-labels' mean and teacher spread per label, the L2-SP record and the precedents | [Prior work](prior-work.md) |
| Reading the first five seeds: Kaggle-only final epochs, confident pseudo-labels weighing more, batch-norm statistics from Kaggle, the fastMRI weight following the learning rate, class weights over both datasets | [Seeds](seed-ideas.yaml) |
| Reading the last four seeds: no loader augmentation on fastMRI, label-rate matching by sampling, hard pseudo-labels, L2-SP off | [Second seeds](seed-ideas-2.yaml) |
| Reading the keeper's rerun of the confidence-weighting seed, whose first run was lost to a loader crash | [Third seeds](seed-ideas-3.yaml) |
| Reading the keeper's rerun of the Kaggle-only final epochs seed, lost to the same kind of crash | [Fourth seeds](seed-ideas-4.yaml) |
| Reading the keeper's seed that applies the scheduled fastMRI weight to the adopted incumbent | [Fifth seeds](seed-ideas-5.yaml) |
| Reading the user's seed that applies batch-norm statistics from Kaggle to the scheduled-weight incumbent | [Sixth seeds](seed-ideas-6.yaml) |
| Reading the seeds after the public check of 2026-10-13, which keep or increase fastMRI's part on the reference: doubled fastMRI weight, no loader augmentation, label-rate matching | [Seventh seeds](seed-ideas-7.yaml) |
| Reading the keeper's rerun of the doubled fastMRI weight, lost to the decoder crash | [Eighth seeds](seed-ideas-8.yaml) |
| Checking the reference, the budget, the editable files, the config permissions and the role instructions | [Campaign definition](campaign.yaml) |
| Reading each run against the incumbent per epoch and per label, and the record of the launcher failure and the loader-worker crashes | [Run notes](run-notes.md) |
