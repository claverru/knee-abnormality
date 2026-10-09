# Pretraining-input campaign

**Closed: stopped by the user on 2026-10-05 at about 11:33 CEST after 32 scored runs; nothing adopted ([results](../../../docs/validation/results/2026-10-05-pretraining-input-campaign.md)).** It tried to close the gap between what we feed the
Refined-RadImageNet backbone and what it was pretrained on, on a ResNet-10t reference: the
intensity mapping of the fluid-sensitive series, the pretraining's own augmentations, our
augmentations that move the input away from the pretraining statistics, and the authors' mask
procedure on the corrected input. Twenty-one forced seeds, twenty of them configuration-only on
options already built, and seventeen more added during the run by the keeper and the user; the
researcher never ran.

| Read when | Document |
|---|---|
| Checking the objective, the seed order, the budget, fixed and permitted mechanisms, the table of built options with their keys and logged scalars, the fixed activation signature of each option, and the build facts for source changes | [Contract](contract.md) |
| Checking the pretraining recipe, the label-free gap measurements (per series type, intensity mappings, the pretraining and our augmentations, field of view) and what the mask-trained weights and masks have already measured | [Prior work](prior-work.md) |
| Checking the sealed recipe, reference, budget, scope and role instructions | [Campaign definition](campaign.yaml) |
| Reading seeds 1–5 (loader bright-end quantile 0.99 and 0.98, heated weights, per-slice quantile, AutoAugment intensity operations) | [Seeds 1](seed-ideas.yaml) |
| Reading seeds 6–10 (per-type quantile, fluid brightness, training without bilateral smoothing, contrast mixing or GIN) | [Seeds 2](seed-ideas-2.yaml) |
| Reading seeds 11–15 (rotation 30°, CutMix inside a study, union bags of two studies, mask centred at 1, heated weights with the mask released after three epochs) | [Seeds 3](seed-ideas-3.yaml) |
| Reading seeds 16–20 (chimera knees from the medial half of one study and the lateral half of another, with the other labels left out, averaged or OR'd; the front-to-back chimera; the finding transplant) | [Seeds 4](seed-ideas-4.yaml) |
| Reading the keeper seed added after runs 1 and 2 (0.98 for the non-fluid series only) | [Keeper seeds 1](keeper-ideas-1.yaml) |
| Reading the keeper seed added after run 4 (per-slice quantile on the non-fluid series only; a source change) | [Keeper seeds 2](keeper-ideas-2.yaml) |
| Reading the keeper seed added after run 5 (AutoAugment intensity operations at probability 0.25 with our intensity augmentation kept) | [Keeper seeds 3](keeper-ideas-3.yaml) |
| Reading the keeper seed added after run 8 (per-slice stretch with a smooth roll-off instead of the hard clip; a source change) | [Keeper seeds 4](keeper-ideas-4.yaml) |
| Reading the keeper seed added after run 11 (one stretch factor per series instead of per slice; a source change) | [Keeper seeds 5](keeper-ideas-5.yaml) |
| Reading the keeper seed added after run 12 (bilateral smoothing off on the incumbent without contrast mixing) | [Keeper seeds 6](keeper-ideas-6.yaml) |
| Reading the keeper seed added after run 17 (AutoAugment at 0.25 with only its mild operations; a one-line source change) | [Keeper seeds 7](keeper-ideas-7.yaml) |
| Reading the keeper seed added after run 22 (union bags at loss weight 0.1) | [Keeper seeds 8](keeper-ideas-8.yaml) |
| Reading the incumbent rerun ordered by the user and the two seeds that give each label group its own intensity scale (stretched and unstretched channels; no stretch on the axial fluid series) | [Keeper seeds 9](keeper-ideas-9.yaml) |
| Reading the heated weights on the current incumbent, ordered by the user | [Keeper seeds 10](keeper-ideas-10.yaml) |
| Reading the two single-model seeds for the heated weights' strengths (feature sparsity penalty; anchor toward the heated checkpoint), asked by the user | [Keeper seeds 11](keeper-ideas-11.yaml) |
| Reading the sagittal-plane version of the split-by-plane seed, taken from the per-view scores | [Keeper seeds 12](keeper-ideas-12.yaml) |
| Reading the frozen heated teacher seed (one model, the heated checkpoint used only during training), asked by the user | [Keeper seeds 13](keeper-ideas-13.yaml) |
| Reading the blend-teacher seed added after run 32 (the student distilled from the mean of the incumbent and the heated model) | [Keeper seeds 14](keeper-ideas-14.yaml) |
| Reading the sparsity penalty at ten times the weight, added when the first run did not raise the zero share | [Keeper seeds 15](keeper-ideas-15.yaml) |
| Reading seed 21 (mask policy with a baseline, per-cell mean and coarse grid; the only source change) | [Seeds 5](seed-ideas-5.yaml) |
| Checking results | [Findings](findings.md) |
| Reading each run's per-label deltas, mechanism scalars and lesson, and the keeper's repairs | [Run notes](run-notes.md) |

## Setup

- **Branch and worktree:** `refined-radimagenet` in `../knee-abnormality-refined-radimagenet`,
  removed on 2026-10-05. The build was never committed; a full copy is frozen in
  `outputs/autoresearch/pretraining-input/baseline/source/`.
- **Build:** the options of the [contract's table](contract.md#built-options) in `config.py`,
  `data/dataset.py`, `modeling/model.py`, `modeling/pretraining_augmentation.py` and
  `training/lightning.py`, all off by default; `configs/training.yaml` set to the ResNet-10t
  recipe with the checkpoint normalization. Checked on CPU with a four-study batch: the default
  recipe runs, and every option switched on does what its row says.
- **Reference:** `20261004-203750-rn10t-radimagenet-normalization`, trained before the build;
  the options are off by default, so the source hash is a declared exception.
- **Gap-meter scripts:** `outputs/analysis/p/pretraining-gap-20261004/`.
- **Start (as it was run):** `rtk proxy uv run --locked just autoresearch start pretraining-input --provider codex`
  from the worktree; `start` imports the seed files itself.

