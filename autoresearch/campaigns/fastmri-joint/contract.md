# fastMRI joint training campaign contract

Written 2026-10-12 with the user. Not started: it waits for the
[remote training host](../../../autoresearch/runtime.md#training-host) option.

## Objective

Raise the macro AUC of a Refined-RadImageNet ResNet-50 that trains on the Kaggle fold and on the
pseudo-labelled fastMRI studies together, by changing how the fastMRI half enters the training:
when it is used, how its cells are weighted, what it does to the regularisation and to the
statistics shared with the Kaggle half. The first joint run ended level with the plain recipe on
the fold (×1 +0.0006, ×4 −0.0003), ahead in every epoch with the lead fading as the learning
rate fell and with PF OA the one large loss, and scored 0.962 publicly (S148), the best public
score at the time ([prior work](prior-work.md)).

The reference and first incumbent is fold 3 of `outputs/training/rn50-joint-fastmri-fold-3`,
trained on the rented machine on 2026-10-12. Trainings run on that machine, the only place where
the fastMRI studies are; agents, review, comparison records and every document stay local.

Nine forced seeds in the order of the ranking by potential agreed with the user on 2026-10-12,
then the researcher without end:

1. `kaggle-only-final-epochs`: the last three of the ten epochs train on Kaggle alone.
2. `confident-pseudo-labels-weigh-more`: a fastMRI cell weighs by how far its pseudo-label is
   from 0.5.
3. `batch-norm-statistics-from-kaggle`: fastMRI batches do not update the running statistics.
4. `fastmri-weight-follows-learning-rate`: the fastMRI batch's loss is scaled by the
   learning-rate curve.
5. `class-weights-over-both-datasets`: the class weights count the fastMRI cells too.
6. `no-loader-augmentation-on-fastmri`: fastMRI studies load without contrast mixing, intensity
   and noise augmentation.
7. `label-rate-matching-by-sampling`: fastMRI studies are drawn so their label rates match the
   Kaggle fold's.
8. `hard-pseudo-labels`: fastMRI pseudo-labels are thresholded at 0.5.
9. `l2-sp-off`: no anchor to the pretrained weights.

Seeds 1, 5 and 9 come from the user's questions on the first run; the others are the keeper's
proposals from its result and are marked as guesses. Dropped in the ranking: mixed batches (the
training ledger counts whole batches, so it is not feasible as a seed) and fastMRI's first part
alone (it discards 22% of the studies on weak evidence).

## Direction after the public check of 2026-10-13

By the user's decision the incumbent is the starting reference again and the campaign changes
direction. S149, the scheduled fastMRI weight, gained +0.0018 at ×4 on the fold and scored 0.961
against S148's 0.962 ([run notes](run-notes.md)). The fold holds only sites seen in training and
cannot see what fastMRI adds for unseen ones, so:

- **No seed reduces fastMRI's part:** no fading or scheduled-down weight, no Kaggle-only epochs,
  no Kaggle-only statistics, no exclusion of fastMRI studies beyond the 181 without geometry.
- **Seeds keep or increase it:** more weight or exposure within the budget, cleaner supervision
  that stays soft, which studies are drawn, regularisation tied to both populations.
- **An adoption is a candidate for a public check, not a conclusion.** The runtime's rule still
  decides the incumbent on the fold.

## Reference exceptions

The reference was trained on the rented machine from a checkout under `/workspace`, so seven
recorded path values differ from this worktree's recipe and are declared as
`reference_config_exceptions`: `paths.data_root`, `paths.label_manifest`,
`paths.report_attribute_manifest`, `paths.anatomy_targets`, `paths.meniscus_targets`,
`extra_training.data_root` and `extra_training.labels`. They are the same files and folders: the
campaign's paths are links to those locations on that machine, and the label manifest's hash is
part of the input identity the comparison checks. **Authorized by the user on 2026-10-12.**

## Training host

Calibration, trainings and comparisons run on the rented machine named in `campaign.yaml`
([remote training host](../../../autoresearch/runtime.md#training-host)); their records
come back after every poll and checkpoints stay there. **The campaign's code runs there with
network access, accepted by the user on 2026-10-12 for this machine:** its container cannot cut
the network, and it holds no credentials, agents or records.

## Settled

- **The backbone is `resnet50.a1_in1k` with the Refined-RadImageNet weights**, on main's recipe
  with labels `rebuild_state_v2` and teacher share 0.
- **Fold 3 of `all_5fold_scanner_v2`, seed 1110484336, ten epochs.** Never fewer.
- **The pseudo-labels are fixed:** `fastmri-rn50-main-x4-tau09-tta`, the held-out teacher's row
  per study (teacher 3 for fold 3), 9,828 studies. No relabelling inside the campaign.
- **The 181 fastMRI studies without fitted knee geometry stay out** of training.
- **Validation is the Kaggle fold alone**, with the control's studies and targets. fastMRI is
  never scored.
- **Prediction settings are not subjects:** ×1 validation and ×4 prediction as the recipe has
  them.
- **Everything is computable for an unseen knee** from its own images through the model.

## Principles

1. **One mechanism per run**, applied to the incumbent it is assigned.
2. **A number set by hand comes from a cited measurement or from the mechanism's definition**,
   never from a sweep on the validation labels.
3. **A seed can be proved active** by metrics it names, at most six.
4. **A superseded behaviour is removed**, not kept behind an option.
5. **A rule is shared by the twelve labels** unless the seed states a per-label rule fixed before
   the run. A per-label rule is never chosen from the campaign fold's validation result.

## How a result is read

One run is weak evidence. Every run is read against the incumbent and against the plain control
(fold 3 of `rn50-main-0bd598f-folds-3-4`, ×1 0.925655, ×4 0.931023), at ×4 and ×1, per label from
largest gain to largest loss, with PF OA, MCL, Effusion and Fracture named apart: they are the
labels the first joint run lost most on. A per-label effect is read on the ×1 AUC averaged over
epochs 6 to 10 together with ×4. Reruns of one recipe land within ±0.0003 at ×4. A change of the
loss weights moves the losses by construction: read AUC, never the loss.

**The fold cannot see unseen sites.** L2-SP was adopted for them and is level on fold validation
([prior work](prior-work.md#l2-sp)); an adoption of `l2-sp-off` is provisional until the user
checks it publicly. The same holds for the seeds that change class or cell weights or the
label mix drawn (2, 5, 7 and 8), since class weighting has disagreed with the public board
before.

## Out of bounds

Another backbone; ensembling, weight averaging, test-time augmentation; fewer epochs; another
label table, changes to gold labels or to the evaluation targets, the teacher share; new
pseudo-labels or teachers; training on the fold's validation studies; trained models in the
loader; new routing, attention or readout structure; image sizes other than 224 px, other
fields, slices or views; reflection probabilities; gold-study weighting; fixed measured offsets
on the joint centre or pose; multi-seed, confirmation or control runs not asked for by the user;
choosing anything on the campaign fold's validation labels; Kaggle submissions; credentials,
agents or records on the rented machine.

## Held for the user

- **Teacher maps as fastMRI targets:** the anatomy and soft-tissue maps of the teacher as extra
  targets on fastMRI. It needs a labelling pass that saves the maps; not a seed until that exists.
- **Children distilled from joint teachers:** needs the five joint folds; after the campaign.

## Cost

**The budget is 0.75 to 1.25 of the reference's training time** (4,530 s on the rented machine).
A run adds no time its mechanism does not need. The joint run peaks at 16.6 GB of GPU memory.

## Duration and adoption

Started with `start`; no end. After the forced seeds the researcher keeps proposing inside this
contract until the user stops it. Adoption is by the runtime under the
[shared rule](../../protocol.md#adoption-policy): strictly above the incumbent at both ×1 and
×4. The keeper repairs and requeues; it does not adopt.
