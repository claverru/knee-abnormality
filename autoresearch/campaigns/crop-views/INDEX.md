# Crop views campaign

**Ran 2026-10-16 22:00 to 2026-10-17 10:30 CEST with Codex, stopped by the user after 13 decided runs.** The
wider coronal crops became the incumbent (x4 0.939467 against the control's 0.937820); the per-knee scale
off was a smaller solo gain; no crop variant added more than +0.0004 to the five-family blend; the ImageNet
ResNet-50 reproduced the other families' per-label pattern, and the framing effect depended on the
pretraining for Lateral Meniscus and Synovitis. Retrains earlier crop and pose framings on the ResNet-50
recipe to find views that add to the recipe in an ensemble. The control is
`resnet50-silver-fold2` (fold 2, ×4 0.937820). Adoption is by a modified, uncommitted runtime in
the worktree `knee-abnormality-crop-views`: a run is adopted when its equal logit blend with the
control gains more ×4 macro AUC over the control than the incumbent's does (the control's own bar
+0.00059). Ten forced seeds, five of them with a ResNet-10t auxiliary backbone, then the researcher
without end. Training time 0.3 to 1.2 of the
control's 2,617 s.

The runtime change, the seed sources and the worktree were not merged; results and cross-checks are in
[2026-10-17 crop views campaign](../../../docs/validation/results/crops-and-views/2026-10-17-crop-views-campaign.md).

| Read when | Document |
|---|---|
| Checking the objective, the seeds, the complement rule, what is settled, how a result is read, what is out of bounds and the cost | [Contract](contract.md) |
| Reading the control and its exceptions, the yardstick for the complement gain, the fold-2 family runs, the earlier crop and pose runs the seeds rest on, and the coverage of today's views | [Prior work](prior-work.md) |
| Reading the five seeds: two views wide and posed, unposed 120 mm everywhere, per-knee scale off, coronal crops a ninth wider, soft bone shares for the joint plane | [Seeds](seed-ideas.yaml) |
| Reading the five ResNet-10t seeds: the full-canvas locator, the wide fluid stream, the locator with its tokens in the pools, the stream with its own loss, the full-canvas context tokens | [ResNet-10t seeds](seed-ideas-2.yaml) |
| Reading the keeper-written source change of the first seed (a posed flag per view, zero pose for unposed views, four validation metrics) | [Two views patch](two-views-wide-and-posed.patch) |
| Reading the eleventh seed, added after the start: the incumbent's posed and wide views drawn as one selection and split by depth, after their x4 overlap was measured | [Distinct depths seed](seed-ideas-3.yaml) |
| Reading the keeper's seeds after the first two results: both axial views unposed alone, and the incumbent with unposed axial views (the second removed for the scale-off versions) | [Axial pose seeds](seed-ideas-4.yaml) |
| Reading the keeper's seeds after per-knee-scale-off was adopted: the wide views on the scale-off base, and with unposed axial views too | [Scale-off combination seeds](seed-ideas-5.yaml) |
| Reading the pretraining by framing 2x2 (user request 2026-10-17): the ResNet-50 on ImageNet weights, posed and unposed at 120 mm | [Pretraining seeds](seed-ideas-6.yaml) |
| Reading the keeper repair of the ImageNet seeds: each convolution rescaled to the RadImageNet scale with its BatchNorm statistics, function preserved, so the recipe's rate and anchors fit | [ImageNet conv scale patch](imagenet-conv-scale.patch) |
| Reading the combination of the campaign's two solo gains: wider coronal crops with the per-knee scale off | [Coronal and scale seed](seed-ideas-7.yaml) |
| Reading the requeued context stream after the time gate blocked it: 3 slices per fluid type, and its own-loss follow-up | [3-slice stream seeds](seed-ideas-8.yaml) |
| Checking the control, the budget, the editable files, the config permissions and the role instructions | [Campaign definition](campaign.yaml) |
| Reading the runtime's decision summary | [Findings](findings.md) |
