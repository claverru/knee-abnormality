# 2026-09-14 knee-side embedding result

**A fold-0 tie with the champion; the side code is retired.** The `knee-side-embedding` run selected epoch 7
(index 6) at **0.892410051**, **+0.000050** over reflection/sigma 40's 0.892360168.
The paired 95% interval for the macro difference is [−0.003386, +0.003197].
It was not submitted. The champion stays, and the
[implementation is archived](../../modeling/retired-side-and-compartment-variants.md#knee-side-embedding).

| Selected model | Macro AUC | Meniscus mean | OA pair mean | Other eight | Soft-target BCE | Gold (11) |
|---|---:|---:|---:|---:|---:|---:|
| Reflection/sigma 40 | 0.892360 | **0.901105** | 0.882491 | 0.892641 | 0.451855 | 0.922156 |
| Knee-side embedding | **0.892410** | 0.900460 | **0.883162** | **0.892710** | **0.450670** | **0.922884** |

Five of twelve targets improved: MCL +0.0068, medial meniscus +0.0055, lateral OA
+0.0048, Synovitis +0.0068 and Contusion +0.0046. The declines were ACL −0.0009,
lateral meniscus −0.0068, medial OA −0.0035, PF OA −0.0040, Effusion −0.0056,
Baker's −0.0055 and Fracture −0.0017. Report-derived AUC moved from 0.891725 to
0.891893. The candidate led at seven of ten matched epochs and finished at 0.892380,
compared with 0.891347. Mean absolute probability change was 0.046, and per-target
Spearman correlations were 0.91–0.98. All family intervals include zero: meniscus
[−0.00862, +0.00701], OA [−0.00528, +0.00663] and other eight [−0.00410, +0.00383].

Exclusive-positive side ranking did not improve. For the 344 exclusive meniscus
studies, the medial-minus-lateral logit-margin AUC was 0.9693 versus 0.9690. For the 198
exclusive OA studies it fell from 0.8791 to 0.8567, and correct-side fraction fell
from 0.813 to 0.773. The small OA subset limits this signal, but it contradicts the
hypothesis that side context mainly improves medial/lateral interpretation.

| Side stratum (studies) | Champion macro | Candidate macro | Paired 95% delta interval |
|---|---:|---:|---|
| Left (412) | 0.8888 | 0.8896 | [−0.00481, +0.00589] |
| Right (450) | 0.8950 | 0.8937 | [−0.00622, +0.00333] |
| Unknown (42) | 0.8878 | 0.8903 | [−0.01194, +0.01921], 1,973 valid draws |
| Known side (862) | — | — | [−0.00355, +0.00338] |

By source, coordinate-assigned studies scored 0.9017 then 0.9009 (440), and explicit
studies 0.8787 then 0.8789 (422). Vendor macro changes were GE −0.0003 (201),
Philips +0.0029 (255), Siemens −0.0018 (386) and Toshiba −0.0069 (48). These strata
come from metadata, not anatomical labels; they show no gain concentrated in one scanner or side group.
The learned vectors were used moderately: norms were left 0.186 and right 0.172,
with distance 0.202 and cosine 0.364. Plane and contrast embeddings had norms of 0.35–0.39.

Training took 1,680.3 s, compared with 1,699.5 s. Validation prediction took 33.1 s,
compared with 31.7 s, and peak CUDA allocation was 20.55 GB versus 20.46 GB. Each run
had 4,380 optimizer steps; one run per model cannot establish a speed difference.
All ten epoch side manifests were identical. Together with the training
manifests, they match the [side policy audit](../../metadata/knee-side-policy-audit.md) for all 4,407 studies.

Configurations differ only in run name and `model.side_dropout`; label, split,
membership and series identities match. Source differences add the side
implementation. They also include the statically reviewed
[CoAtNet adapter refactor](../../modeling/coatnet-224-spatial-backbone.md) in
`checkpoints.py`/identities, and the reference's since-removed
`configs/reflection-sigma40.yaml`. Uncertainty uses 2,000 report-group draws (852 groups,
seed 20260914). It excludes training-seed and checkpoint-selection variability; one seed and
fold cannot establish repeatability or public ranking. Metadata inference,
embedding and side dropout were changed together.

The [comparison artifact](../../../outputs/analysis/k/knee-side-embedding-20260914/comparison.json)
has per-target, conditional, exclusive-margin, side/source/vendor strata, epoch,
timing, vector and fingerprint details. It was produced on CPU from saved predictions and checkpoint tensors by
[the historical analysis](../../history/knee-side-sources/README.md), which did not
need the removed side code. It left the script catalog on 2026-09-16, when the run's
checkpoints were deleted.

## Side interventions: the checkpoint ignores knee side

The [encoding diagnostic](../manual-encoding-usage-diagnosis.md) on the selected checkpoint
reproduced saved probabilities (mean absolute difference 3.1e−5, maximum 0.0015)
with no preparation conditions. Every side intervention left predictions essentially unchanged:

| Variant | Macro AUC change | Mean absolute probability change | Paired 95% interval |
|---|---:|---:|---|
| Side suppressed to unknown | +0.000045 | 0.00177 | [−0.00007, +0.00016] |
| Known sides swapped (862) | +0.000052 | 0.00176 | [−0.00012, +0.00023] |
| 1/5/10% flipped, uniform | +0.000011/+0.000011/+0.000021 | ≤0.00018 | all include zero |
| 1/5/10% flipped, GEHC/Philips first | +0.000008/−0.000003/+0.000007 | ≤0.00015 | all include zero |

No target changed by more than 0.001 when sides were swapped, and left, right,
explicit and coordinate strata moved by at most 0.0002. Swapping every known side
had no cost, so the model does not use side to decide medial versus lateral.
Plane (+0.00181) and contrast (+0.00179) removal produced similar probability changes.
Replacing 3D with unknown changed AUC by +0.000037, with probability change 0.0062,
although 873 of 904 studies and 96.7% of visual tokens have valid 3D. Removing all
position information cost −0.001461 [−0.00247, −0.00048]; 1D and 2D removals cost
−0.000461 and −0.000244.

Because the 3D term also carries little weight, side-conditioned coordinates
(research approaches 2 and 3) would act through a branch this checkpoint barely
uses. The side code was therefore retired rather than extended. These are removal measurements on one checkpoint, not trained ablations.
Artifacts are in `outputs/analysis/e/encoding-usage-knee-side-20260914/`
(`results.json`, per-variant predictions and the side manifest copy).
