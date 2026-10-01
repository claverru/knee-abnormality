# Revisits campaign

**Ran 2026-09-30 22:15 to 2026-10-01 16:57 CEST, stopped by the user.** It re-tested on the
RadImageNet ResNet-18 recipe the ideas whose verdict changed when past comparisons were re-scored on
`rebuild_state_v2`, plus untried neighbours, then searched hyperparameters and measured the
backbones the user forced in. Outcome: plane-contrast readouts and removal of the anatomical bias
accepted (×4 0.929835 against 0.928062); the Refined-RadImageNet ResNet-50 +0.0046, kept as a
measurement; one backbone per plane ties; nothing else passed
([result](../../../docs/validation/results/2026-10-01-revisits-campaign.md)).

| Read when | Document |
|---|---|
| Checking the objective, seeds, budget, fixed and permitted mechanisms and evidence requirements | [Contract](contract.md) |
| Checking where each seed comes from and what ResNet-18 has already measured | [Prior work](prior-work.md) |
| Checking the sealed recipe, reference, budget, scope and role instructions | [Campaign definition](campaign.yaml) |
| Reading seeds 1–5 (dense per-plane readouts, plane/contrast readouts, no DSU, no anatomical bias, no bilateral smoothing) | [Seeds 1](seed-ideas.yaml) |
| Reading seeds 6–9 (no resolution degradation, plane/contrast feature adapters, plane-conditioned BatchNorm, stochastic depth) | [Seeds 2](seed-ideas-2.yaml) |
| Reading the two EMA seeds (decay 0.999 and 0.9995) the user forced on 2026-09-30, lifting the weight-averaging exclusion for them only | [EMA seeds](ema-ideas.yaml) |
| Reading the EMA retakes with a BatchNorm refresh on a fixed quarter of the training studies, forced by Claude as campaign keeper after both EMA seeds ran over the time budget and the final repair was blocked by the one-repair review allowance | [EMA subset refresh](ema-subset-refresh-ideas.yaml) |
| Reading the timm-style EMA runs (existing option, BatchNorm statistics averaged with the weights, no refresh) that replaced the refresh retakes on the user's instruction, 2026-10-01 | [EMA buffer average](ema-timm-ideas.yaml) |
| Reading the two ResNet-10t user seeds (incumbent recipe at backbone LR 5e-5, and at the weight-RMS-scaled 6e-5), forced on 2026-10-01 | [ResNet-10t seeds](resnet10t-ideas.yaml) |
| Reading the withdrawn original-RadImageNet ResNet-50 seeds (5e-5 and 1.27e-3, nine epochs, [−1, 1] input scaling), replaced before launch by the Refined checkpoint | [ResNet-50 seeds](resnet50-ideas.yaml) |
| Reading the Refined-RadImageNet ResNet-50 user seed (incumbent recipe, 5e-5, ten epochs under the raised 2,016 s limit), with the checkpoint's hash and inspection | [Refined ResNet-50 seed](refined-resnet50-ideas.yaml) |
| Reading the Refined-RadImageNet ResNet-50 at its weight-RMS-scaled backbone LR (5.5e-5, ten epochs), forced by the user on 2026-10-01 | [Refined ResNet-50 scaled LR](refined-resnet50-scaled-ideas.yaml) |
| Reading the four plane-backbone user seeds (one ResNet-18 or ResNet-10t per plane on the incumbent recipe; A keeps the joint pool and plane-contrast readouts, B uses dense per-plane readouts with learned fusion), forced on 2026-10-01 | [Plane backbones](plane-backbone-ideas.yaml) |
| Reading the last user seed, removing the report-attribute heads, after which the user stops the campaign (2026-10-01) | [No attribute heads](no-report-attributes-ideas.yaml) |
| Checking results | [Findings](findings.md) |

## Setup

- **Branch and worktree:** `revisits` in `../knee-abnormality-revisits`, off `00f2f32`.
- **Reference:** `rn18-l2sp-no-confusion` (×4 0.928062, 0.943 public as S86). It was trained from
  the since-removed `rn18-l2sp-confusion` worktree, so its `paths.label_manifest` and
  `paths.report_attribute_manifest` are declared config exceptions: its recorded
  `label_manifest_sha256` (`08339ec5…`) and `report_attribute_manifest_sha256` (`f712c551…`) match
  this tree's files.
- **Budget:** range 0.5–1.65 of the reference's 1,222 s training time (1.5 until the user raised it on 2026-10-01).
- **No topic rotation:** `rotate_after: 1000` (user, 2026-09-30).
