# MambaVision settings campaign

**Closed: stopped by the user 2026-10-14 18:40 CEST (forced). Five adoptions: one rate of 2e-4, anchors matched to the weight scale, twelve epochs, the late anchor following the learning rate, linear decay. The five trained together outside the campaign (`mambavision-t-combined-12-epochs`, ×4 0.937801, ×1 0.929880, blend with the ResNet-50 0.941419) became the incumbent by the user's decision and are `configs/mambavision-t.yaml` on main. The four seeds then run on it (anchors per stage, stochastic depth 0.1, two warm-up epochs, twice the head rate) all rose at ×1 and fell at ×4; the fifth was stopped. First stop 08:58 CEST after 11 decided runs.** Started 2026-10-13 23:05 CEST with Claude as keeper and Codex as the provider. The campaign tunes the training settings of MambaVision-T
that its differences from the ResNet-50 make suspect, on the line of its reference,
`mambavision-t-rn50-recipe` (the ResNet-50 recipe unchanged on fold 2: ×4 0.922130 against
0.939634 for the ResNet-50, with a higher training loss at every epoch). Seven forced seeds, then
the researcher without end. Adoption by the runtime under the shared rule. Training time 0.75 to
1.3 of the reference's. It runs from the worktree `knee-abnormality-new-backbones`, whose
environment holds the model code and compiled kernels
([dependencies](../../../docs/modeling/mambavision-backbone.md#dependencies)).

| Read when | Document |
|---|---|
| Checking the objective, what is settled, the principles a seed follows, how a result is read, what is out of bounds, cost and adoption | [Contract](contract.md) |
| Reading the reference per epoch and per label against the ResNet-50, why it underfits (weight scale, drift, anchor, gradient norm), the seeds removed before the start, the precedents and the code a seed can port | [Prior work](prior-work.md) |
| Reading the first four seeds: stage-wise rates matched to the weight scale, one rate of 2e-4, one rate of 4e-4, anchors matched to the weight scale | [Seeds](seed-ideas.yaml) |
| Reading the last three seeds: twelve epochs, feature-statistics augmentation off, gradient clipping at 10 | [Second seeds](seed-ideas-2.yaml) |
| Reading the seed the keeper added after run 3: the stage-wise rates again, on the incumbent with the weight-scale anchors | [Third seeds](seed-ideas-3.yaml) |
| Reading the seed the keeper added after the user scoped the generalisation rule: the scheduled late anchor of run 9 on the linear-decay incumbent | [Fourth seeds](seed-ideas-4.yaml) |
| Reading the five seeds on the full recipe (user, 2026-10-14): anchors per stage by weight scale, stochastic depth 0.1, two warm-up epochs, twice the head rate, the early anchor following the rate | [Fifth seeds](seed-ideas-5.yaml) |
| Checking the reference, the budget, the editable files, the config permissions and the role instructions | [Campaign definition](campaign.yaml) |
| Reading each run's result against the incumbent and the ResNet-50, per label | [Run notes](run-notes.md) |
| Reading the runtime's decision summary | [Findings](findings.md) |
