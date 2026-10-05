# Refined-RadImageNet campaign

**Closed: stopped by the user on 2026-10-04 at 20:43 CEST after 18 scored runs and two manual ones; the one adoption is the checkpoint normalization, on main since 2026-10-05 ([results](../../../docs/validation/results/2026-10-04-refined-radimagenet-campaign.md)).** It tested how to get more out of Refined-RadImageNet
pretraining on the ResNet-18 recipe: the checkpoint's own input statistics, the authors'
policy-gradient mask (their mask-trained weights and the mask in our fine-tuning), the pretrained
165-class head with its 18 knee classes, and adjacent slices through Cross-D and ACS
convolutions. Fourteen forced seeds, then an endless researcher.

| Read when | Document |
|---|---|
| Checking the objective, the seed order and its exceptions, the budget, fixed and permitted mechanisms, the build facts a developer needs and the target-to-class mapping | [Contract](contract.md) |
| Checking what the refinement is, the checkpoints and their weight scales, the pretrained head measured on our slices, the authors' mask method, Cross-D-Conv and what ResNet-18 has already measured | [Prior work](prior-work.md) |
| Checking the sealed recipe, reference, budget, scope and role instructions | [Campaign definition](campaign.yaml) |
| Reading seeds 1–5 (checkpoint normalization, the two mask-trained ResNet-10t checkpoints, the policy-gradient mask for all epochs and in two stages) | [Seeds 1](seed-ideas.yaml) |
| Reading seeds 6–10 (pretrained policy network, random smooth mask, knee logits as token features, knee-class readout, frozen slice teacher) | [Seeds 2](seed-ideas-2.yaml) |
| Reading seeds 11–14 (knee-logit attention prior, Cross-D ResNet-18 per slice, Cross-D 3D stem over adjacent slices, ACS stem and layer1) | [Seeds 3](seed-ideas-3.yaml) |
| Reading the user seed that keeps the backbone in eval mode so BatchNorm uses the pretrained running statistics, forced on 2026-10-04 after the normalization seed was accepted | [Backbone eval seed](backbone-eval-ideas.yaml) |
| Reading the two keeper seeds that repair the Cross-D seeds (static shifted kernels as a plain ResNet-18, and the gated three-slice stem), forced on 2026-10-04 | [Cross-D static seeds](crossd-static-ideas.yaml) |
| Reading the keeper seed that replaces the ACS seed with a gated, function-preserving form in the stem and the first block of layer1, forced on 2026-10-04 | [Gated ACS seed](acs-gated-ideas.yaml) |
| Reading the keeper seed that keeps gated ACS in the stem only, after the stem-and-block form ran over the time bound, forced on 2026-10-04 | [Gated ACS stem seed](acs-gated-stem-ideas.yaml) |
| Reading the keeper seed that reruns the gated ACS stem with an unclamped gate, after the clamped gate died at zero, forced on 2026-10-04 | [ACS gate fix seed](acs-gate-fix-ideas.yaml) |
| Checking results | [Findings](findings.md) |
| Reading each run's per-label deltas, curve, mechanism readings and lesson, and the keeper's repairs | [Run notes](run-notes.md) |

## Setup

- **Branch and worktree:** `refined-radimagenet` in `../knee-abnormality-refined-radimagenet`,
  off `5edd795`. The build is uncommitted (the runtime works from uncommitted source):
  the `radimagenet_policy`, `radimagenet_policy_ft` and `radimagenet_crossd` weight entries,
  the classifier loader and knee-class table in `modeling/radimagenet.py`, `modeling/crossd.py`,
  the config values and the provenance lines of `scripts/pipeline/train.py`.
  `modeling/model.py` stays byte-identical to main: it is a hashed preprocessing source, and a
  first start on 2026-10-04 paused at calibration because an edit to it changed the baseline's
  identity; the seeds that need other weights carry the model change themselves.
- **Links:** `outputs` and `data` in the worktree are links to the main checkout's.
- **Reference:** `rn18-view-label-bias` (`outputs/training/20261003-100844-rn18-view-label-bias`,
  ×4 0.932477, 1,181 s, 0.951 public as S101). Its `paths.label_manifest` and
  `paths.report_attribute_manifest` name the removed view-config worktree and are declared
  config exceptions.
- **ResNet-10t reference:** `rn10t-view-label-bias`
  (`outputs/training/20261004-094416-rn10t-view-label-bias`, ×4 0.927543, ×1 0.921848, 1,141 s),
  trained from main on 2026-10-04 for seeds 2 and 3.
- **Budget:** range 0.75–2.0 of the reference's 1,181 s training time (886–2,362 s; user,
  2026-10-04). Epochs, batch size and slices per study are not editable.
- **Seed order:** forced, in file order. The normalization seed is first because the head seeds
  need it and it costs nothing.
- **Endless research** (user, 2026-10-04): `rotate_after: 1000000`, and the researcher is told
  never to return an empty refill and to open a new family when one closes. Only the user stops
  the campaign.
- **Weights in the torch hub cache:** `policy_radimagenet_resnet10t.pth`,
  `policy_ft_radimagenet_resnet10t.pth`, `policy_radimagenet_resnet50.pth`,
  `policy_ft_radimagenet_resnet50.pth`, `crossd_radimagenet_resnet18.pth`, beside the three
  standard Refined files; hashes in `PRETRAINED_WEIGHTS`.

## Settled before the start

- **Normalization (label-free, 2026-10-04):** the checkpoints were trained with mean 0.3162 and
  std 0.3213; under the ImageNet values our recipe uses, the frozen pretrained head puts 0.088
  of its probability on knee classes for our slices, against 0.978 under its own
  ([prior work](prior-work.md#pretrained-head-on-our-slices)). Seed 1 tests it.
- **Class order:** the 165 classes follow the sorted folder names (the authors' loader is
  torchvision's `ImageFolder`); under the checkpoint's normalization a knee class is on top for
  98.5% of our slices, which confirms indices 83–100.
- **Policy network:** not published anywhere; the mask seeds train their own.
- **Cross-D layer:** `modeling/crossd.py` reproduces the authors' layer to rounding and the
  checkpoint loads strictly into the converted ResNet-18.
