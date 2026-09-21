# Combined labels (`combined_v1`)

**Training and validation labels in force since 2026-09-21.** Steven v4 supplies eight targets.
ACL, MCL, PF OA and Fracture instead take the per-study mean of nine independent public report
extractors. Official gold labels still override every source on the 58 gold studies. On gold, the
table scores 0.9075 macro AUC against Steven v4's 0.8927. Gold helped choose the four targets, so
that figure is optimistic, and the image-model effect is untested.

## What the table contains

| Target | Source | Silver positive rate, Steven v4 → combined | Gold AUC, Steven v4 → combined |
|---|---|---|---|
| ACL | Consensus mean | 20.6% → 14.7% | 0.987 → 0.996 |
| MCL | Consensus mean | 15.2% → 5.2% | 0.968 → 0.995 |
| PF OA | Consensus mean | 45.9% → 31.7% | 0.902 → 0.929 |
| Fracture | Consensus mean | 6.8% → 6.1% | 0.793 → 0.907 |
| Other eight | Steven v4 | unchanged | unchanged |

The consensus mean averages whichever of these sources cover a study. At least five always do.

| Source | Model | Value used |
|---|---|---|
| Pilkwang v2 | GPT-4 | probability |
| Laymond v2 filled | Qwen2.5-32B | `__confidence` column |
| Zhen v7 | not stated | probability |
| Yunus | Gemini Flash | severity levels 0.05 / 0.25 / 0.55 / 0.9 |
| JEV text | OpenJev ensemble | probability |
| Josh | gpt-5.6-luna, DSPy consensus | 0.5 ± half the relative self-reported confidence |
| Ramit | GPT-luna | score; not mentioned is 0.2 |
| Dreaddevelopment | not stated | probability |
| Vmohit | not stated, rubric-thresholded | positive 1; absent or below threshold 0; uncertain or unmentioned left out |

The source files and their provenance are in the
[public label re-screen](2026-09-21-public-label-rescreen.md). The Josh mapping is this project's
choice, since that release gives a binary annotation with a confidence rather than a probability.

## How the targets were chosen

1. **Remove duplicates.** Binary agreement on silver studies at 0.5 groups the tables into
   families. The Steven v4 family agrees 97–100% with Lixin GPT-5.6-Sol, Niklita, Rayan, Shingo,
   the two Fracture hybrids, Tonylin and JEV HYBRID. The Pilkwang family includes the Yunus merges
   and Zhen v8. Josh and Vmohit agree 98.6%. Tables that copy gold into their gold rows were
   dropped: Zaid, Tasmeem and seven others. The nine sources above are one member per remaining
   family, excluding Steven v4.
2. **Two candidates per target:** Steven v4, or the consensus mean. The consensus is the plain
   mean of the sources' own values, with no weights fitted on gold.
3. **Rule.** Take the consensus only when all three checks hold on the 58 gold studies:
   - it beats Steven v4 in more than 90% of 2,000 gold-study bootstrap draws;
   - its lowest leave-one-source-out AUC still beats Steven v4;
   - it still beats Steven v4 without the two sources whose prompts were developed on gold, JEV
     and Josh.

   The four chosen targets:

   | Target | P(better) | Leave-one-out minimum | Without gold-tuned sources |
   |---|---:|---:|---:|
   | ACL | 0.917 | 0.995 | 0.998 |
   | MCL | 0.950 | 0.989 | 0.991 |
   | PF OA | 0.944 | 0.921 | 0.916 |
   | Fracture | 0.998 | 0.881 | 0.918 |

   No other target has P above 0.62. Baker's cyst, Medial OA and Lateral OA, Effusion and the
   menisci are worse under the consensus.
4. **Mechanism.** For ACL, MCL and PF OA the consensus is stricter than Steven v4, in line with the
   rubric's high-grade definitions ([rubric](../metadata/official-label-criteria.md)). For
   Fracture, Steven v4 is the weakest of the seven sources that can be benchmarked. Where stricter
   sources lose on gold, as for Effusion and Baker's "moderate or large", the radiologists'
   calls are broader than a strict report reading, so Steven v4 stays.

The rule was set after the per-target results were seen; it is not pre-registered. With 9–35 gold
positives per target, some chosen gains can be selection noise. The two mechanisms and the
agreement between independent sources are what the choice rests on.

## Alternatives measured and rejected

- **Latent-class aggregation (Dawid–Skene) for all targets.** It was fitted on silver studies
  only, with Steven v4 plus the nine sources as binary voters. Gold macro is 0.8903, below Steven
  v4. The posteriors are nearly hard, because correlated sources look more reliable than they are.
  The latent class also follows whichever definition has the most voters, not the rubric.
- **A first pass that included Zaid** showed gains on all twelve targets. Zaid copies gold into its
  gold rows, so that result was an artifact.
- **Whole-table replacements** remain rejected: see the
  [re-screen](2026-09-21-public-label-rescreen.md#whole-table-comparison) and the
  [Tonylin run](../validation/results/2026-09-13-tonylin-label-comparison.md).

## Literature behind the design

- **Correlated sources.** Aggregation that ignores correlated sources rates duplicate labelers as
  more reliable than they are ([Snorkel](https://arxiv.org/abs/1711.10160),
  [Cachay 2021](https://arxiv.org/abs/2106.10302), [AMA](https://arxiv.org/abs/2210.02441)). That is
  why families are removed first.
- **Simple averaging.** With full coverage and independent sources, majority vote and averaging
  are competitive with learned label models ([WRENCH](https://arxiv.org/abs/2109.11377)).
- **Soft targets.** Soft targets generally beat rounded ones (WRENCH;
  [Pham 2021](https://arxiv.org/abs/1911.06475)).
- **Report versus image.** Report-derived and image-derived labels disagree systematically, through
  different thresholds and omitted findings. [VisualCheXbert](https://arxiv.org/abs/2102.11467) fitted a
  report-to-image mapping on 500 image-labelled studies; 58 cannot support a fitted mapping, only
  a choice between two variants per target.
- **Small-sample selection.** Choosing among many variants on a small set is optimistic
  ([Cawley and Talbot 2010](https://jmlr.org/papers/v11/cawley10a.html)). CheXpert's per-finding
  uncertainty policies were mostly not significant even at 200 studies
  ([Irvin 2019](https://arxiv.org/abs/1901.07031)).

Aggregating report extractors cannot close the report-to-image gap. Synovitis stays
the clearest case: every text source misses 14 or more of its 27 gold positives.

## Reproduction

- Table: `rtk proxy uv run --locked python -m scripts.tools.prepare_combined_labels` writes
  `data/labels/prepared/combined-v1/labels.csv` (SHA-256 `1e586b57…be215fb`) and
  [`configs/labels/combined-label-sources.json`](../../configs/labels/combined-label-sources.json).
  The manifest pins every input's hash and also lists `steven_v4`, so one manifest serves both
  label sources.
- Evidence: `scripts.adhoc.analyze_label_combination` writes
  `outputs/analysis/label-combination-20260921/`, with `per_target.csv`,
  `silver_agreement.csv` and `summary.json`. It fails if its selection differs from the prepared
  table.
- The validation split built on these labels is
  [`all_5fold_scanner_v1`](../validation/validation-strategy.md).
