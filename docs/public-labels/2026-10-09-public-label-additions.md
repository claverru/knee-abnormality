# 2026-10-09 public label additions

**Nothing new is worth adding to the label rebuild.** A Kaggle sweep found eleven label or
prediction tables published since the [10-03 sweep](2026-10-03-public-label-additions.md). One
publisher (Kiritohayate) adds three new independent report readers; they score below
`rebuild_state_v2` on gold and on our blind-judged cells. The rest are copies, derivatives,
rule-based lexicon readers or image-model predictions weaker than ours. No image-derived labels,
regions of interest or masks on the competition studies were published.

- **Search:** Kaggle datasets matching knee, RSNA, labels, LLM, pseudo, soft, teacher, report,
  OOF, distill, gold, mask, segmentation, annotations and JEV, sorted by update date, up to three
  pages per term; kept those updated from 2026-10-03 with a knee or RSNA name (77), then those
  holding a table.
- **Snapshots:** `data/labels/public/<owner>/<slug>/downloaded-20261009/`.
- **Checks:** scratch analysis, not a maintained script; same three as on 10-03. Gold macro AUC
  on the official 58 studies; Spearman and binary kappa at 0.5 against held roots on silver;
  accuracy and AUC on the 371 report cells judged blind in the
  [source audit](label-rebuild.md#source-audit-result-2026-09-28) (POS and HEDGED positive).

## Tables

| Dataset | What it is | Verdict |
|---|---|---|
| `kiritohayate95/rsna-knee-medgemma-grades` (Oct 6) | Three readers on all 4,407 reports, MedGemma-27B, Qwen-32B and Mistral, with one prompt: an integer 0–10 per finding for "how likely expert radiologists re-reading the images would mark it present and clinically significant", with a severity rubric per finding. Scripts included. | New roots (kappa 0.65–0.70 with Steven v4, Lixin and Pilkwang v2; 0.73–0.76 with `rebuild_state_v2`). Below the labels in force; see below. Not added. |
| `tannguyenngoc/label-soft-rsna` (Oct 5) | Four soft tables with per-cell weights, report labels blended with an image teacher; gold copied into the 58 gold rows. | Teacher derivative. Blind accuracy 0.830, AUC 0.908. Excluded. |
| `karttikjangid05/rsna-knee-public-llm-label-sets-compared` (Oct 8) | Plain mean of six held roots (Steven v2, Pilkwang, Lixin, Laymond, Yunus, Dreaddev) and per-source gold AUCs with bootstrap intervals. | Derivative of held roots. Gold 0.8886. Excluded. |
| `siamrahman29/rsna-knee-assets-public` (Oct 7) | A lexicon report reader (`report_labels.parquet`) and a consensus table that is Steven v4 in effect (Spearman 0.98), with a MIL training package. | Lexicon reader gold 0.826, blind accuracy 0.704. Excluded. |
| `codeandodev/rsna-knee-etiquetas-debiles` (Oct 6) | Lexicon reader with per-finding confidence and state; gold copied. | Blind accuracy 0.693. Excluded. |
| `omarelhabashy/knee-labels-v3` (Oct 7) | Four-state LLM read (abnormal, normal, uncertain, not mentioned), prompt included. | Gold 0.790, blind accuracy 0.655. Excluded. |
| `mandavisingh/rsna-knee-labels-v1` (Oct 6) | Spearman 1.00 with Steven v4. | Copy. |
| `tunminhhunh/rsna-knee-dinov2-labels` (Oct 6) | `report_labels_v2.csv` identical in scores to Pilkwang v2, with a DINOv2 checkpoint. | Copy. |
| `zhuowamg/rsna-knee-e13-train` (Oct 8) | Out-of-fold predictions of an image model on all 4,407 studies. | Gold 0.858, below our own out-of-fold predictions. Excluded. |
| `bijaychdas/rsna-knee-2026-g1-v2-holdout-preds` (Oct 4) | Holdout predictions (881 studies) of a DINO model at 0.796 on its own labels. | Excluded. |
| `hengck23/hengck23-rnsa-knee` (Oct 8) | Code, one checkpoint and the gold rows of `train.csv` with their series metadata. | No labels. |

| Source | Gold macro AUC | Blind accuracy (n 371) | Without HEDGED (n 260) | Blind AUC | Silver positive rate |
|---|---:|---:|---:|---:|---:|
| `rebuild_state_v2` (in force) | 0.9075 | 0.887 | 0.862 | 0.950 | 0.173 |
| Steven v4 | 0.8927 | 0.811 | 0.750 | 0.919 | 0.269 |
| Karttik consensus | 0.8886 | 0.887 | 0.846 | 0.917 | 0.270 |
| Kiritohayate Qwen-32B | 0.8856 | 0.792 | 0.781 | 0.849 | 0.212 |
| Kiritohayate Mistral | 0.8790 | 0.787 | 0.769 | 0.853 | 0.224 |
| Kiritohayate MedGemma-27B | 0.8790 | 0.760 | 0.738 | 0.829 | 0.225 |

The blind cells favour the labels in force, as noted on
[10-03](2026-10-03-public-label-additions.md#tables).

## The Kiritohayate graded readers

**Weak evidence, 58 gold studies.** The rank mean of the three readers scores 0.8982 on gold
against 0.9075 for `rebuild_state_v2`. Per label, largest gain to largest loss: Synovitis +0.051
(0.841 against 0.790, 27 positives), Lateral OA +0.027, Medial Meniscus +0.005, Baker's +0.003,
Lateral Meniscus +0.001, ACL −0.003, MCL −0.017, Fracture −0.024, Medial OA −0.025, Effusion
−0.034, PF OA −0.045, Contusion −0.050. Mixing the mean into `rebuild_state_v2` at rank weight
0.25 or 0.5 gives 0.9066 for both, level with 0.9075 alone.

The Synovitis difference is inside the gold noise (a 27-positive AUC has a 95% interval of about
±0.11). What differs is the rubric, which is the one usable idea here: Synovitis is raised for
haemarthrosis, Hoffa fat-pad oedema or impingement, synovial proliferation and a moderate or large
effusion with debris, and a small plain effusion alone stays low. It is a candidate question for
the [report attribute extraction](../ongoing/report-attribute-extraction.md), not a table to merge.

## Forum reports on labels

From the [2026-10-09 re-sweep](../research/2026-10-09-public-resweep.md#labels):
a participant's bootstrap on gold finds no public set clearly better than the median set on 11
of 12 findings, and several top participants say label extraction quality is no longer what
separates scores. Neither changes the labels in force.
