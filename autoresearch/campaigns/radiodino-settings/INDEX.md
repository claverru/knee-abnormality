# RadioDINO settings campaign

**Closed by the user 2026-10-15 ~17:00 CEST, no code adopted.** Ran from 01:36 with Codex, Claude as keeper: 19 runs scored (4 invalid). Seed 1 (RadioDINO on the MambaVision recipe, ×4 0.929220) adopted by the user's decision; final incumbent `top-down-backbone-warm-up` (×4 0.929641, ×1 0.924256), 0.0082 below the MambaVision-T reference and about 0.0055 below a decoder-free MambaVision-T. Rates, schedules, anchors, read depths, a learned final merge and a locality line (convolutional spatial prior, overlapping patches, depthwise convolution in the MLPs) all landed within about ±0.002. Tunes the training settings of RadioDINO-S/16 (RadImageNet
DINO ViT-S/16) that its differences from MambaVision-T make suspect. Reference: the MambaVision-T
incumbent `mambavision-t-combined-12-epochs` (fold 2, ×4 0.937801, ×1 0.929880). Six forced
seeds: the first puts RadioDINO on the MambaVision recipe and is adopted by the user's decision
whatever its verdict; the other five apply to it. Training time 0.5 to 1.1 of the reference's,
twelve epochs fixed. Ran from the worktree `knee-abnormality-radiodino`, which held the adapter.

The worktree `knee-abnormality-radiodino`, its branch and the runtime artifacts were removed at closing; the adapter code is kept as [radiodino-adapter.patch](radiodino-adapter.patch).

| Read when | Document |
|---|---|
| Checking the objective, the seeds, what is settled, the principles, how a result is read, what is out of bounds, cost, and the keeper's forced adoption of seed 1 | [Contract](contract.md) |
| Reading the model, the adapter, how it differs from MambaVision-T, the weight scale and relative steps, the precedents, and the reference's exceptions against today's source | [Prior work](prior-work.md) |
| Reading the first five seeds: RadioDINO on the MambaVision recipe, rates falling with depth, anchors by weight scale, feature statistics before the position embedding, the middle read at block 10 | [Seeds](seed-ideas.yaml) |
| Reading the sixth seed: stochastic depth 0.1 | [Second seeds](seed-ideas-2.yaml) |
| Reading the five rate and schedule ideas forced on the user's message of 2026-10-15 03:54 and run before seeds 5 and 6: stage rates from 4e-4 to 2e-4, the backbone held for the first epoch, top-down warm-up per stage, one rate of 1.4e-4, the position embedding's own rate | [Keeper rate ideas](keeper-ideas-rates.yaml) |
| Checking the reference, the budget, the editable files, the config permissions and the role instructions | [Campaign definition](campaign.yaml) |
| Reading the two fixes from the diagnosis of the gap to MambaVision-T (user's message of 08:12): a learned 2x2 merge for the final read, the early anchor by gradient scale | [Keeper fix ideas](keeper-ideas-fixes.yaml) |
| Reading the convolutional spatial prior (user's choice of 2026-10-15 12:00): the RadImageNet ResNet-50 stem to layer2 added to the ViT tokens before each stage through zero-initialised gates | [Keeper spatial-prior idea](keeper-ideas-spatial-prior.yaml) |
| Reading the spatial prior with its pretrained ResNet layers frozen, after the first version ran over budget | [Keeper frozen spatial-prior idea](keeper-ideas-spatial-prior-frozen.yaml) |
| Reading the overlapping patch embedding (user's choice, 2026-10-15): a 32x32 stride-16 projection started from the pretrained kernel | [Keeper overlap idea](keeper-ideas-overlap.yaml) |
| Reading the depthwise convolution in the block MLPs (PVTv2/LocalViT locality, user's line of 2026-10-15) | [Keeper local-MLP idea](keeper-ideas-local-mlp.yaml) |
| Reading each run's result against the incumbent and the reference, per label, and the keeper's repairs (the seal exceptions, the restored reference checkpoint, seed 1's forced adoption) | [Run notes](run-notes.md) |
| Reading the adapter's source change (staged backbone, spatial-token geometry), not adopted | [Adapter patch](radiodino-adapter.patch) |
| Reading the keeper-written spatial-prior patches (trainable, frozen, stride 4) | [Spatial prior](spatial-prior.patch), [frozen](spatial-prior-frozen.patch), [stride 4](spatial-prior-stride-four.patch) |
| Reading the runtime's decision summary | [Findings](findings.md) |
