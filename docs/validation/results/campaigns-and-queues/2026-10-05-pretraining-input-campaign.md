# 2026-10-05 pretraining-input campaign

**Nothing from this campaign is adopted: its two acceptances on the ResNet-10t did not carry over
to the ResNet-18.** The `pretraining-input` autoresearch campaign tried to bring our input
preparation and augmentation closer to what the Refined-RadImageNet backbone was pretrained on,
on a ResNet-10t with the checkpoint normalization. It scored 32 runs (one more was aborted) from
2026-10-04 23:47 to 2026-10-05 11:33 CEST, when the user stopped it. Two runs met the adoption
rule on the ResNet-10t: dividing each slice by its own 0.98 quantile (+0.0007) and switching
contrast mixing off (+0.0020), together ×4 0.930247 against 0.927579 at the start. On the
ResNet-18 the pair scored −0.0013 and the per-slice quantile alone −0.0007, neither separated
from zero, and removing contrast mixing had already been +0.0006 and not separated in an earlier
campaign. The recipe on main keeps only the
[checkpoint normalization](../backbones/2026-10-04-refined-radimagenet-campaign.md).

Fold 0 of `all_5fold_scanner_v2` (912 studies), `rebuild_state_v2` labels, seed 1533697213, ×4
predictions, paired report-group bootstrap with 2,000 draws. Adoption rule: the lower end of the
central 80% interval above zero
([protocol](../../../../autoresearch/protocol.md#shared-evaluation-and-adoption)). Every result is
one run at one seed. Training-time range 0.75 to 1.35 of the reference's 1,162 s (user,
2026-10-04).

- **Campaign record** (contract, prior work with the label-free gap measurements, seeds,
  decision summary): [pretraining-input campaign](../../../../autoresearch/campaigns/pretraining-input/INDEX.md).
- **Per-run details** (per-label deltas, curves, mechanism scalars, lessons, the manual runs and
  the keeper's repairs): [run notes](../../../../autoresearch/campaigns/pretraining-input/run-notes.md).
- **Code:** the build (the options of the campaign's
  [contract](../../../../autoresearch/campaigns/pretraining-input/contract.md#built-options) and one
  logged scalar per option) is not on main; its worktree was removed on 2026-10-05 after
  the campaign closed, and a full copy is frozen in `outputs/autoresearch/pretraining-input/baseline/source/`, and
  each run's source, with the developer's change where it had one, under
  `experiments/<run>/source/`.
- **Artifacts:** `outputs/autoresearch/pretraining-input/` (state, and per run the frozen source,
  `changes.diff`, config, logs, predictions and comparison). All checkpoints were removed on
  2026-10-05 ([cleanup log](../../../architecture/configs-and-experiment-artifacts.md)), so new
  comparisons against these runs need them retrained. The first start's folder, with no
  training, is `outputs/autoresearch/pretraining-input-aborted-20261004`.

## Yardstick

The incumbent recipe trained twice at the same seed (`incumbent-rerun`) differs by 0.0001 at ×4
and by at most 0.0008 on any label, and by at most 0.0003 at ×1 over the ten epochs. The deltas
below are not GPU noise down to about 0.0003. That does not measure how much a different seed
moves a run: the campaign pins the incumbent's seed.

## Runs

References: the historical control `20261004-203750-rn10t-radimagenet-normalization` (×4
0.927579, ×1 0.921891), then `slice-upper-quantile-098` (0.928294) from run 5 and
`no-contrast-mixing` (0.930247) from run 13. Runs are in the order they were scored; "keeper"
and "user" mark seeds added during the campaign.

| # | Run | What changes | Reference | Δ ×4 | 80% interval | Verdict |
|---|---|---|---|---:|---|---|
| 1 | `loader-upper-quantile-099` | Loader bright-end quantile 0.99 (from 0.995) | historical control | +0.000439 | [−0.000048, +0.000911] | reject |
| 2 | `loader-upper-quantile-098` | Loader bright-end quantile 0.98 | historical control | +0.000420 | [−0.000230, +0.001087] | reject |
| 3 | `heated-weights` | Mask-trained (heated) checkpoint, matched rate and anchors | historical control | +0.001039 | [−0.000650, +0.002647] | reject |
| 4 | `slice-upper-quantile-098` | Each slice divided by its own 0.98 quantile and clamped | historical control | **+0.000716** | [+0.000009, +0.001413] | **accepted on the ResNet-10t** |
| 5 | `autoaugment-intensity` | AutoAugment intensity operations on every slice, our intensity augmentation off | `slice-upper-quantile-098` | −0.002907 | [−0.004085, −0.001671] | reject |
| 6 | `fluid-type-upper-quantile` | Loader quantile 0.98 on the fluid-sensitive series only | `slice-upper-quantile-098` | −0.000247 | [−0.000703, +0.000205] | reject |
| 7 | `other-type-upper-quantile` | Loader quantile 0.98 on the non-fluid series only (keeper) | `slice-upper-quantile-098` | −0.000217 | [−0.000477, +0.000031] | reject |
| 8 | `slice-quantile-other-types` | Per-slice stretch on the non-fluid series only (keeper) | `slice-upper-quantile-098` | −0.000952 | [−0.001677, −0.000240] | reject |
| 9 | `fluid-brightness` | Fluid-sensitive series multiplied by 1.2 and clamped | `slice-upper-quantile-098` | −0.001079 | [−0.001827, −0.000349] | reject |
| 10 | `no-bilateral-smoothing` | Bilateral smoothing augmentation off | `slice-upper-quantile-098` | +0.000902 | [−0.000073, +0.001923] | reject |
| 11 | `slice-quantile-soft-knee` | Per-slice stretch with a tanh roll-off above 0.8 instead of the clamp (keeper) | `slice-upper-quantile-098` | +0.000025 | [−0.000266, +0.000314] | reject |
| 12 | `no-contrast-mixing` | Contrast mixing augmentation off | `slice-upper-quantile-098` | **+0.001952** | [+0.000994, +0.002938] | **accepted on the ResNet-10t** |
| 13 | `no-gin` | GIN augmentation off | `no-contrast-mixing` | −0.000560 | [−0.001830, +0.000706] | reject |
| 14 | `no-bilateral-smoothing-on-no-mixing` | Bilateral smoothing off as well (keeper) | `no-contrast-mixing` | −0.001948 | [−0.002988, −0.000937] | reject |
| 15 | `series-quantile-stretch` | One stretch factor per series instead of per slice (keeper) | `no-contrast-mixing` | −0.000560 | [−0.001196, +0.000060] | reject |
| 16 | `rotation-30` | Rotation up to 30° (from 10°) | `no-contrast-mixing` | −0.001765 | [−0.002582, −0.000974] | reject |
| 17 | `autoaugment-intensity-025` | AutoAugment on a quarter of the slices, our intensity augmentation kept (keeper) | `no-contrast-mixing` | −0.003720 | [−0.005028, −0.002470] | reject |
| 18 | `study-cutmix` | CutMix between slices of the same study and series type, probability 0.5 | `no-contrast-mixing` | −0.003917 | [−0.005064, −0.002773] | reject |
| 19 | `union-bag-loss` | Extra loss on union bags of two studies with OR labels, weight 0.5 | `no-contrast-mixing` | −0.003706 | [−0.005447, −0.001899] | reject |
| 20 | `input-mask-centred-one` | Random smooth input mask in [0.5, 1.5] | `no-contrast-mixing` | −0.002893 | [−0.004008, −0.001777] | reject |
| 21 | `heated-weights-mask-release` | Heated checkpoint with a random mask for three epochs | `no-contrast-mixing` | −0.002152 | [−0.003888, −0.000445] | reject |
| 22 | `split-knee-bag-loss` | Chimera of a medial and a lateral half, five side labels, weight 0.5 | `no-contrast-mixing` | −0.004919 | [−0.006380, −0.003556] | reject |
| 23 | `split-knee-bag-average` | The chimera with the other seven labels averaged | `no-contrast-mixing` | −0.002651 | [−0.003977, −0.001359] | reject |
| 24 | `split-knee-bag-or` | The chimera with the other seven labels OR'd | `no-contrast-mixing` | −0.003378 | [−0.004968, −0.001745] | reject |
| 25 | `front-back-bag-loss` | Chimera cut front to back (PF OA anterior, Baker's posterior) | `no-contrast-mixing` | −0.004358 | [−0.006103, −0.002634] | reject |
| 26 | `transplant-bag-loss` | A donor's most attended tokens pooled with a whole host, OR labels | `no-contrast-mixing` | −0.003555 | [−0.005228, −0.001847] | reject |
| 27 | `learned-mask-policy` | Mask policy with a baseline, per-cell mean and an 8 × 8 grid | `no-contrast-mixing` | — | — | aborted at epoch 6 by the user |
| 28 | `incumbent-rerun` | The incumbent recipe again at the same seed (user's control) | `no-contrast-mixing` | −0.000119 | [−0.000346, +0.000124] | reject |
| 29 | `slice-quantile-keep-axial-fluid` | No stretch on the axial fluid-sensitive series (keeper) | `no-contrast-mixing` | −0.000370 | [−0.000980, +0.000236] | reject |
| 30 | `heated-weights-on-incumbent` | Heated checkpoint on the incumbent (user) | `no-contrast-mixing` | −0.001669 | [−0.003376, −0.000016] | reject |
| 31 | `slice-quantile-keep-sagittal-fluid` | No stretch on the sagittal fluid-sensitive series (keeper) | `no-contrast-mixing` | −0.000934 | [−0.001449, −0.000422] | reject |
| 32 | `heated-teacher-distillation` | Frozen trained heated model as a teacher, weight 0.5 (keeper) | `no-contrast-mixing` | +0.000105 | [−0.000486, +0.000680] | reject |
| 33 | `slice-window-channels` | Channels as stretched, as loaded, stretched (keeper) | `no-contrast-mixing` | −0.001046 | [−0.001748, −0.000385] | reject |

Run by hand after the stop, from frozen sources with the supervisor's command and environment:
the feature-sparsity penalty at weight 0.05 (the campaign's last run, scored −0.000819, 95%
[−0.001641, −0.000013], decision not recorded) and at 0.5 (stopped at epoch 4, 0.027 below the
incumbent at ×1), and the L2-SP anchor pointed at the heated checkpoint (stopped after two
epochs, level with the incumbent). Not run: the blend-teacher distillation, AutoAugment with
only its mild operations, and the union bags at weight 0.1.

## On the ResNet-18

Reference: the adopted ResNet-18 recipe, `radimagenet-input-statistics-20261004T103433Z-ff2f3786`
(×4 0.933058, ×1 0.929948). One run each, trained from the worktree's build.

| Run | What changes | ×4 | Δ | 95% interval |
|---|---|---:|---:|---|
| `outputs/training/20261005-094913-rn18-slice-quantile-no-mixing` | Per-slice quantile 0.98 and contrast mixing off | 0.931789 | −0.001269 | [−0.003347, +0.000695] |
| `outputs/training/20261005-101600-rn18-slice-quantile-098` | Per-slice quantile 0.98 | 0.932317 | −0.000741 | [−0.001987, +0.000450] |
| `rn18-native-contrast-preservation` (`radimagenet-rn18` campaign, older recipe) | Contrast mixing off | — | +0.000551 | 80% [−0.001057, +0.002199] |

The pair is behind its reference from epoch 5 at ×1 and the quantile alone at every epoch, by
about 0.001. Fracture, the largest gain of `no-contrast-mixing` on the ResNet-10t (+0.0066), is
the largest loss of the pair on the ResNet-18 (−0.0080); MCL rises on both backbones. On the
ResNet-10t every one of the sixteen later candidates measured against `no-contrast-mixing` lost
MCL or Fracture or both, whatever its mechanism, which reads as that single run sitting high on
those two small labels (48 and 59 positives).

## Findings

- **Clipping the bright end trades labels.** Stretching the fluid-sensitive series raises MCL,
  Lateral OA, Lateral Meniscus and ACL and lowers Baker's and Effusion (runs 1, 2, 4). Taking the
  stretch off those series reverses both (run 8). Six ways of reducing the clip (non-fluid only,
  a roll-off, a per-series factor, the axial or the sagittal plane left out, two scales in the
  channels) show the same trade in proportion; none kept MCL. Leaving the axial fluid series
  unstretched is the narrowest: Baker's +0.0049 for MCL −0.0068, the other ten labels within
  0.002. Single-view validation scores from a pair of runs did not predict which plane carries
  the effect.
- **Our augmentations.** Removing contrast mixing passed on the ResNet-10t and removing
  bilateral smoothing missed by 0.00007; together they give the gain back (+0.0000 against the
  same reference). Removing GIN lost 0.0006: faster early, lower at the end.
- **The pretraining's augmentations lose.** AutoAugment's intensity operations (−0.0029 on every
  slice, −0.0037 on a quarter of them), rotation to 30° (−0.0018), CutMix inside a study
  (−0.0039).
- **Bag losses lose at weight 0.5.** Six runs (union bags, three medial-lateral chimeras, the
  front-back chimera, the transplant) between −0.0027 and −0.0049, each with a constant deficit
  from the second epoch. Medial OA is the one label they help (five of six).
- **Masks.** A random mask centred at 1 lost 0.0029. The policy fixed in `learned-mask-policy`
  does learn (Beta parameters 1.2 to 2.6–3.4, gradient norm 0.4–0.7), but its 8 × 8 mask is four
  times stronger than the random one and the run sat 0.008 to 0.014 below the incumbent.
- **Heated weights.** +0.0010 over the first reference and −0.0017 over the last incumbent, with
  the same label pattern in five runs across two campaigns: ligaments, lateral meniscus and
  contusion up, the OA labels down.
- **The heated model as a second view.** An equal-weight blend of the incumbent and a heated run
  scores +0.0011 to +0.0012 over the incumbent on fold 0 (two heated runs, 0.0017 and 0.0022
  below it alone); a blend with a standard-weights run adds 0.0000 to 0.0005 and with the
  same-seed rerun nothing. A standard-weights student distilled from the frozen heated model
  took its MCL and ACL gains and kept Lateral OA, and ended level with the incumbent (+0.0001).
  Follow-up: [untried ideas](../../../backlog/untried-ideas.md#ensembles).
- **A sparsity penalty does not make the features sparse.** An L1 penalty on the mean of the
  final feature map shrinks the activations (the share of zero features fell from 0.46 to 0.42
  at weight 0.05 and to 0.33 at 0.5; the heated checkpoint is at 0.88).

## Harness notes

- **A configuration-only seed needs a logged scalar that proves its option acted.** The first
  seed was blocked at review for lack of one; the build got one scalar per option and the
  contract a fixed activation signature per option, and the campaign was started again (it
  freezes its source at start).
- **Same-seed reruns are near-reproducible** (see the yardstick), and the campaign cannot run a
  different seed.
