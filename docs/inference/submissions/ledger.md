# Submission ledger

All 102 Kaggle competition submissions, newest first, with submission times in UTC. The table is
regenerated from the live submission list and the local bundles by
[`update_submission_ledger`](../../../scripts/tools/update_submission_ledger.py),
which also writes `outputs/submission/kaggle-ledger.json`. The dated records in this folder cover
S01–S32; the baseline4 image-count comparisons are recorded in
[Inference image count and memory](../inference-image-count-and-memory.md), and the 2026-09-19
ensemble submissions S33–S37 are recorded in
[2026-09-19 ensemble submissions](2026-09-19-ensemble-submissions.md); S94–S97 are recorded in
[2026-10-02 learned-view and blend submissions](2026-10-02-blend-submissions.md).

## Naming

Every submission carries one line, and Kaggle keeps it forever: **a submission description cannot
be edited after submitting**, and neither the API nor the site offers a way to change it. The
sequence number, the dataset and notebook title and that line are all built by
[`submit`](../../../scripts/pipeline/submit.py) from the bundle, so the only thing chosen by hand
is `--label`.

```text
S33 | b0-fold0 | x4 tau0.75 bs4 | val0(x1) 0.886354 | bundle c99de9cc
```

| Field | Meaning |
|---|---|
| `S<nn>` | Position in the submission list, counted from the first submission; stable because submissions only append |
| label | The run and its one change against the run it is meant to beat, 2 to 45 characters, the single `--label` argument; the bound keeps the title within 50 characters, since a 52-character notebook title failed `SaveKernel` with HTTP 400 |
| inference | `x<images> tau<effective temperature> bs<batch per device>`, plus `+mirrorTTA`, read from the bundle manifest |
| `val0(x<m>)` | Fold-0 macro AUC of the exported checkpoint at validation multiplier `m`; an ensemble reports `<k>x mean` over its members |
| bundle | First eight characters of the bundle manifest SHA-256, which identifies the exact submitted artifact |

The title is `S<nn> <label>`, shared by the notebook and the private bundle dataset, which keeps
the Kaggle notebook list, the dataset list and this ledger on the same handles. Each bundle
dataset also carries the same line as its description and `Submission S<nn>, public LB <score>` as
its subtitle.

The sequence number counts only submissions already on Kaggle, so a job launched while another
has not yet submitted would reuse its number; pass `--sequence` explicitly for overlapping
launches. The ledger script numbers rows by submission time, so an explicit `--sequence` that
disagrees with time order is swapped in `kaggle-ledger.json`: S59 was submitted 28 s before S58.
The descriptions on Kaggle, and this table, carry the correct numbers.

Two caveats apply to the rows before S33. Kaggle notebook titles were set from their slugs until
this scheme, so the notebook names on Kaggle still read `Knee Rf Tau075 4X 20260918 Inference`
rather than `S31 …` until the ledger script's `--apply` renames them. And the three `ranked-filtered … + tau0.75` bundles carry
the temperature folded into the scorer weights rather than in a runtime field, so their `tau0.75`
comes from the table in the ledger script rather than from the manifest.

### Regenerating the ledger

```bash
rtk proxy uv run --locked python -m scripts.tools.update_submission_ledger
rtk proxy uv run --locked python -m scripts.tools.update_submission_ledger --apply
```

Without `--apply` it writes only the ledger JSON and prints the planned titles. Labels for S01–S37
come from a table in the script; later ones are recovered from the notebook slug. With `--apply` it:

- rewrites each private bundle dataset's title, subtitle and description, refusing any dataset
  that is not private and keeping the `Private inference bundle <sha>` line, so `submit` still
  recognizes an unchanged bundle and reuses its version;
- renames each notebook by saving its current source and metadata back as a quick save, a new
  version that is not executed, so no GPU quota is spent and submitted versions are untouched;
- skips datasets and notebooks that already match, and any notebook whose latest run has not
  finished;
- pauses between calls and retries HTTP 429 with exponential backoff, since a full pass is
  rate limited.

## Ledger

S numbers follow the live submission order. From S66 to S71 the permanent Kaggle labels run one
higher, and S72 carries "S66": the reliability-filter submission was first labelled S66, failed,
and was resubmitted later. Other documents cite the Kaggle labels (S67 for the scanner confusion
head); the Label column gives the Kaggle label wherever it differs.

| S | Submitted (UTC) | Label | Inference | Fold 0 | Public | ID | Record |
|---|---|---|---|---|---:|---:|---|
| S107 | 2026-10-04 19:34 | RN50 of S102 + RN18 of S101 (view label bias, fold 0), equal weights, one bundle | x4 tau0.9 bs4 | (x4 blend, fold 0, τ 1) 0.936891 | 0.954 | — | [2026-10-04 distillation](../../validation/results/2026-10-04-two-family-oof-distillation.md)#resnet-18-beside-the-resnet-50-s107 |
| S106 | 2026-10-04 18:54 | oof-blend-v2 children, the five Refined RN50 only (S103 without the ReXNets) | x6 tau0.8 bs4 +mirrorTTA | (x6 pooled out of fold) 0.936272 | **0.957** | — | [2026-10-04 distillation](../../validation/results/2026-10-04-two-family-oof-distillation.md)#resnet-50-children-alone-s106 |
| S105 | 2026-10-04 16:22 | S103 with τ 1.1 for effusion and synovitis | x6 tau0.7 (RN50) / 1.0 (ReXNet) bs4 +mirrorTTA | (x6 pooled out of fold) 0.939208 | 0.954 | — | [2026-10-04 distillation](../../validation/results/2026-10-04-two-family-oof-distillation.md)#a-temperature-per-label |
| S104 | 2026-10-04 11:24 | S103's ten checkpoints with type share in place of the tuned temperatures | x6 tau1 bs4 +mirrorTTA +typeShare | not measured | 0.951 | — | [2026-10-04 distillation](../../validation/results/2026-10-04-two-family-oof-distillation.md)#type-share-in-place-of-the-temperatures-s104 |
| S103 | 2026-10-04 10:27 | oof-blend-v2 children: 5 Refined RN50 + 5 ReXNet-200, folds 0–4, one bundle | x6 tau0.7 (RN50) / 1.0 (ReXNet) bs4 +mirrorTTA | (x6 pooled out of fold) 0.938887 | **0.955** | — | [2026-10-04 distillation](../../validation/results/2026-10-04-two-family-oof-distillation.md)#submission |
| S102 | 2026-10-03 11:43 | RN50 84/93 views + label bias f0 t0.9 (`refined-rn50-views-label-bias`) | x4 tau0.9 bs4 | (x1) 0.931222, x4 0.937129 | **0.954** | 56796920 | [adoption](../../modeling/current-model.md#adoption-history) |
| S101 | 2026-10-03 10:44 | RN18 main + label bias f0 t0.9 (`rn18-view-label-bias`, the control since 2026-10-03) | x4 tau0.9 bs4 | (x1) 0.929026, x4 0.932477 | 0.951 | 56796002 | [adoption](../../modeling/current-model.md#adoption-history) |
| S100 | 2026-10-03 10:06 | RN18 hand views + label bias f0 t0.9, the same recipe on the branch's code | x4 tau0.9 bs4 | (x1) 0.929876, x4 0.933307 | 0.950 | 56795386 | `view-optimisation` branch, learnable-views campaign; [adoption](../../modeling/current-model.md#adoption-history) |
| S99 | 2026-10-03 08:40 | ens of five RN18 learnable-views fold-0 runs, equal weights, own code each (oa3d, sagroute, centre, both3d, sagwidths) | x4 tau0.9 bs4, no TTA | (x4 blend) 0.935015 | 0.951 | not recorded | `view-optimisation` branch, learnable-views campaign |
| S98 | 2026-10-03 08:22 | RN18 learned sagittal routing f0 t0.9 | x4 tau0.9 bs4 | (x1) 0.930065, x4 0.933143 | 0.947 | 56793440 | `view-optimisation` branch, learnable-views campaign |
| S97 | 2026-10-02 21:06 | ens of 28 networks in 13 bundles, mirror TTA on all, weighted by architecture: Refined RN50 f0 0.30; RN18 84/93 mm f0 0.18 + RN18 L2-SP folds 1–4 0.12; ReXNet f0, 3 × f3, 4 bundles f4 0.30; B3 + ReXNet f3 0.05; 5 B0 + 5 ReXNet of 2026-09-19 0.05 | x4 (B3 x5) mirror TTA bs4 | none (members of other folds) | **0.954** | 56782385 | [2026-10-02 blends](2026-10-02-blend-submissions.md#s97-members-and-weights) |
| S96 | 2026-10-02 19:23 | ens ReXNet v2 control (τ 0.75) + Refined RN50 (τ 0.9), both with reflection TTA, + RN18 84/93 mm (τ 0.9), fold 0; three bundles, one notebook | x4 bs4 | (x4 blend, fold 0, τ 1, no TTA) 0.936926 | 0.953 | 56780901 | [2026-10-02 blends](2026-10-02-blend-submissions.md) |
| S95 | 2026-10-02 19:23 | ens RN18 84/93 mm + RN18 learned views (τ 0.9), fold 0; two bundles, one notebook | x4 tau0.9 bs4 | (x4 blend, fold 0, τ 1) 0.933996 | 0.950 | 56780899 | [2026-10-02 blends](2026-10-02-blend-submissions.md) |
| S94 | 2026-10-02 17:45 | RN18 learned views budget f0 t0.9 | x4 tau0.9 bs4 | (x1, fold 0) 0.927404 | 0.947 | 56779307 | [2026-10-02 blends](2026-10-02-blend-submissions.md#what-the-numbers-say) |
| S93 | 2026-10-02 07:11 | RN18 84mm T1PD 93mm fluid f0 t0.9 | x4 tau0.9 bs4 | (x1, fold 0) 0.929261 | 0.950 | 56768246 | [2026-10-02 smart-crops campaign](../../validation/results/2026-10-02-smart-crops-campaign.md) |
| S92 | 2026-10-01 20:51 | RN18 120mm plane readouts f0 t0.9 | x4 tau0.9 bs4 | (x1, fold 0) 0.927479 | 0.947 | 56758831 | [2026-10-01 mixed-field campaign](../../validation/results/2026-10-01-mixed-field-campaign.md#control-on-the-adopted-recipe) |
| S91 | 2026-10-01 19:18 | RN18 T1PD112 fluid140 mid-aug f0 t0.9 | x4 tau0.9 bs4 | (x1, fold 0) 0.924854 | 0.945 | 56757734 | [2026-10-01 mixed-field campaign](../../validation/results/2026-10-01-mixed-field-campaign.md) |
| S90 | 2026-10-01 19:16 | RN18 120mm mid-slice aug f0 t0.9 | x4 tau0.9 bs4 | (x1, fold 0) 0.925649 | 0.947 | 56757704 | [2026-10-01 mixed-field campaign](../../validation/results/2026-10-01-mixed-field-campaign.md) |
| S89 | 2026-10-01 09:25 | ens ReXNet v2 control (τ 0.75) + Refined RN50 (τ 0.9), fold 0, reflection TTA; two bundles, one notebook | x4 refl-TTA bs4 | (x4 blend, fold 0, no TTA) 0.935999 | 0.951 | 56748424 | [2026-10-01 revisits campaign](../../validation/results/2026-10-01-revisits-campaign.md) |
| S88 | 2026-10-01 09:07 | Refined RN50 f0 t0.9 | x4 tau0.9 bs4 | (x1, fold 0) 0.929393 | 0.948 | 56748142 | [2026-10-01 revisits campaign](../../validation/results/2026-10-01-revisits-campaign.md) |
| S87 | 2026-09-30 20:00 | RN18 OOF-blend 0.5 f0 t0.9 | x4 tau0.9 bs4 | (x1, fold 0) 0.926842 | 0.944 | 56717610 | [Next](../../ongoing/next.md) |
| S86 | 2026-09-30 17:27 | RN18 L2-SP no confusion f0 t0.9 | x4 tau0.9 bs4 | (x1, fold 0) 0.924507 | 0.943 | 56714526 | [RN18 L2-SP without confusion head](../../validation/results/2026-09-30-rn18-l2sp-no-confusion.md) |
| S85 | 2026-09-30 12:20 | RN18 sites-out L2-SP+ArcFace f0 t0.9 (sites-out split) | x4 tau0.9 bs4 | (x1, fold 0 seen sites) 0.924518 | 0.940 | 56707163 | [Site-generalization prior work](../../../autoresearch/campaigns/site-generalization/prior-work.md) |
| S84 | 2026-09-30 12:18 | RN18 sites-out L2-SP f0 t0.9 (sites-out split) | x4 tau0.9 bs4 | (x1, fold 0 seen sites) 0.925349 | 0.943 | 56707112 | [Site-generalization prior work](../../../autoresearch/campaigns/site-generalization/prior-work.md) |
| S83 | 2026-09-30 12:18 | RN18 sites-out control f0 t0.9 (sites-out split) | x4 tau0.9 bs4 | (x1, fold 0 seen sites) 0.925173 | 0.941 | 56707111 | [Site-generalization prior work](../../../autoresearch/campaigns/site-generalization/prior-work.md) |
| S82 | 2026-09-29 21:00 | RN18 control no-SD 26trip f0 t0.9 | x4 tau0.9 bs4 | (x1, fold 0) 0.923840 | 0.943 | 56687489 | [RN18 L2-SP without confusion head](../../validation/results/2026-09-30-rn18-l2sp-no-confusion.md) |
| S81 | 2026-09-29 16:52 | RN18 attributes v4 f0 t0.9 | x4 tau0.9 bs4 | (x1, fold 0) 0.922590 | 0.942 | 56682502 | [Report attribute extraction](../../ongoing/report-attribute-extraction.md) |
| S80 | 2026-09-29 07:33 | ens ReXNet v2 control (τ 0.75) + RadImageNet RN18 (τ 0.9), both 36 triplets, fold 0; two bundles, two passes | x4 36trip bs4 | (x1, fold 0) 0.924637 / 0.924370 | 0.949 | — | [Single-pass mixed-source ensembles](../../backlog/refactors/single-pass-mixed-source-ensembles.md) |
| S79 | 2026-09-29 07:13 | RadImageNet RN18 36trip no-SD fold0 (Kaggle description says S78) | x4 tau0.9 bs4 | (x1, fold 0) 0.924370 | 0.944 | 56666712 | [RN18 attention temperature](../../validation/results/2026-09-29-rn18-attention-temperature.md) |
| S78 | 2026-09-29 06:37 | rexnet rebuild-state-v2 control fold0 ep09 (Kaggle description says S79) | x4 tau0.75 bs4 | (x1, fold 0) 0.924637 | 0.944 | 56665597 | [Label rebuild](../../public-labels/label-rebuild.md) |
| S77 | 2026-09-28 11:46 | 3xReXNet plane dense fusion fold4 ep09 | x4 tau0.75 bs4 | (x1, fold 4) 0.907878 | 0.945 | 56639652 | [Plane-specialization campaign](../../validation/results/2026-09-28-plane-specialization-campaign.md) |
| S76 | 2026-09-28 11:00 | rexnet rebuild-state labels fold4 ep09 | x4 tau0.75 bs4 | (x1, fold 4) 0.897726 | 0.947 | 56638577 | [Label rebuild](../../public-labels/label-rebuild.md) |
| S75 | 2026-09-28 10:26 | rexnet rebuild-mean labels fold4 ep09 | x4 tau0.75 bs4 | (x1, fold 4) 0.901624 | 0.946 | 56637797 | [Label rebuild](../../public-labels/label-rebuild.md) |
| S74 | 2026-09-28 09:51 | 3xB0 plane dense learned fusion fold4 ep08 | x4 tau0.75 bs4 | (x1, fold 4) 0.900042 | 0.943 | 56636971 | [Plane-specialization campaign](../../validation/results/2026-09-28-plane-specialization-campaign.md) |
| S73 | 2026-09-28 06:43 | rexnet scanner confusion control fold4 ep09 | x4 tau0.75 bs4 | (x1, fold 4) 0.902061 | 0.946 | 56632006 | [Plane-specialization campaign](../../validation/results/2026-09-28-plane-specialization-campaign.md) |
| S72 | 2026-09-27 11:47 | attr reliability filter fold3 epoch 09 retry (Kaggle label S66) | x4 tau0.75 bs4 | (x1, fold 3) 0.902138 | 0.943 | 56607593 | [Auxiliary-signals campaign](../../validation/results/2026-09-27-auxiliary-signals-campaign.md) |
| S71 | 2026-09-27 10:20 | B3 lr4.5e-4 + ReXNet half-stack fold3 (Kaggle label S70) | x5 tau0.75 bs4 +mirrorTTA | (x1, fold 3) 2x mean 0.900756 | 0.946 | 56605776 | [B3 overnight queues](../../validation/results/2026-09-27-b3-overnight-queue.md) |
| S70 | 2026-09-27 10:16 | scanner confusion stronger head fold3 (Kaggle label S71) | x4 tau0.75 bs4 | (x1, fold 3) 0.895089 | 0.943 | 56605710 | [Auxiliary-signals campaign](../../validation/results/2026-09-27-auxiliary-signals-campaign.md) |
| S69 | 2026-09-27 10:04 | scanner-model confusion fold3 epoch 09 (Kaggle label S70) | x4 tau0.75 bs4 | (x1, fold 3) 0.898365 | 0.944 | 56605445 | [Auxiliary-signals campaign](../../validation/results/2026-09-27-auxiliary-signals-campaign.md) |
| S68 | 2026-09-27 00:07 | per-site bias fold3 epoch 09 (Kaggle label S69) | x4 tau0.75 bs4 | (x1, fold 3) 0.901322 | 0.947 | 56592463 | [Auxiliary-signals campaign](../../validation/results/2026-09-27-auxiliary-signals-campaign.md) |
| S67 | 2026-09-26 18:33 | half-stack control fold3 epoch 09 (Kaggle label S68) | x4 tau0.75 bs4 | (x1, fold 3) 0.902067 | 0.947 | 56586402 | [Auxiliary-signals campaign](../../validation/results/2026-09-27-auxiliary-signals-campaign.md) |
| S66 | 2026-09-26 18:15 | scanner confusion head fold3 epoch 09 (Kaggle label S67) | x4 tau0.75 bs4 | (x1, fold 3) 0.900348 | 0.948 | 56586087 | [Auxiliary-signals campaign](../../validation/results/2026-09-27-auxiliary-signals-campaign.md) |
| S65 | 2026-09-26 11:41 | half-stack augmentation fold2 epoch 09 | x4 tau0.75 bs4 | (x1, fold 2) 0.893525 | 0.943 | 56578320 | [Augmentation campaign](../../validation/results/2026-09-26-augmentation-campaign.md) |
| S64 | 2026-09-26 10:45 | half-stack augmentation fold2 epoch 07 | x4 tau0.75 bs4 | (x1, fold 2) 0.893622 | 0.943 | 56577204 | [Augmentation campaign](../../validation/results/2026-09-26-augmentation-campaign.md) |
| S63 | 2026-09-26 09:15 | smoothing + DSU augmentation fold2 | x4 tau0.75 bs4 | (x1, fold 2) 0.891370 | 0.940 | 56575386 | [Augmentation campaign](../../validation/results/2026-09-26-augmentation-campaign.md) |
| S62 | 2026-09-25 19:45 | report-attribute heads v3.1 fold1 | x4 tau0.75 bs4 | (x1, fold 1) 0.895294 | 0.940 | 56559056 | [Report-attribute heads](../../validation/results/2026-09-25-report-attribute-heads.md) |
| S61 | 2026-09-25 18:29 | report-attribute heads fold1 | x4 tau0.75 bs4 | (x1, fold 1) 0.896116 | 0.938 | 56557729 | [Report-attribute heads](../../validation/results/2026-09-25-report-attribute-heads.md) |
| S60 | 2026-09-25 08:44 | OA detector control fold1 | x4 tau0.75 bs4 | (x1, fold 1) 0.895202 | 0.938 | 56546031 | [Adoption history](../../modeling/current-model.md#adoption-history) |
| S59 | 2026-09-25 07:29 | token dropout random fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.900029 | 0.939 | 56544445 | [Near-misses campaign](../../validation/results/2026-09-25-near-misses-campaign.md) |
| S58 | 2026-09-25 07:30 | three-compartment OA detector fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.897369 | 0.944 | 56544456 | [Near-misses campaign](../../validation/results/2026-09-25-near-misses-campaign.md) |
| S57 | 2026-09-24 17:05 | early learning reg fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.900934 | 0.943 | 56527174 | [Early-learning regularization](../../history/early-learning-regularization.md) |
| S56 | 2026-09-24 11:14 | knee-geometry adopted new seed fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.896343 | 0.940 | 56519764 | [Current model](../../modeling/current-model.md) |
| S55 | 2026-09-24 09:34 | knee-geometry merged new seed fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.897581 | 0.940 | 56517876 | [Knee-geometry findings](../../../autoresearch/campaigns/knee-geometry/findings.md) |
| S54 | 2026-09-24 08:08 | knee-geometry half-knee MIL fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.900879 | 0.944 | 56516135 | [Knee-geometry findings](../../../autoresearch/campaigns/knee-geometry/findings.md) |
| S53 | 2026-09-24 08:20 | class-balance schedule fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.899067 | 0.941 | 56516372 | [Noisy-labels campaign](../../validation/results/2026-09-23-noisy-labels-campaign.md#class-weighting-changes-need-the-public-board) |
| S52 | 2026-09-23 11:09 | image-blend oblique embedding fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.905319 | 0.943 | 56491424 | [Oblique groups](../../validation/results/2026-09-22-oblique-groups-result.md#group-embedding-on-image-blend-labels-s52) |
| S51 | 2026-09-23 10:17 | oblique group embedding fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.901635 | 0.942 | 56490382 | [Oblique groups](../../validation/results/2026-09-22-oblique-groups-result.md#group-embedding-on-the-cleaned-source-s51) |
| S50 | 2026-09-23 09:29 | k05 clean image-blend labels fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.904133 | **0.945** | 56489466 | – |
| S49 | 2026-09-23 09:10 | k05 epoch9 soft-AUC pick fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.898724 | 0.944 | 56489076 | [Validation AUC variants](../../validation/results/2026-09-23-validation-auc-variants.md#checkpoint-selection-check-s49) |
| S48 | 2026-09-23 07:09 | k05 cleaned source fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.899419 | 0.942 | 56486378 | [Series cleaning](../../validation/results/2026-09-22-series-cleaning-rules.md#retrained-with-cleaning) |
| S47 | 2026-09-22 11:29 | oblique groups fold0 | x5 tau0.7 bs4 | (x1, scanner split) 0.897527 | **0.945** | 56463079 | [Oblique groups](../../validation/results/2026-09-22-oblique-groups-result.md) |
| S46 | 2026-09-22 11:07 | oblique groups fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.897527 | 0.944 | 56462626 | [Oblique groups](../../validation/results/2026-09-22-oblique-groups-result.md) |
| S45 | 2026-09-22 07:20 | class-balance exponent k0.75 fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.901737 | 0.942 | 56457101 | [Class-balance exponent](../../validation/results/2026-09-21-class-balance-exponent.md) |
| S44 | 2026-09-22 06:27 | class-balance exponent k0.25 fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.898255 | 0.944 | 56455715 | [Class-balance exponent](../../validation/results/2026-09-21-class-balance-exponent.md) |
| S43 | 2026-09-22 06:13 | stratified jitter on k0.5 fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.899976 | 0.944 | 56455304 | [Sampling campaign](../../../autoresearch/campaigns/sampling/prior-work.md#user-submissions) |
| S42 | 2026-09-21 21:43 | unweighted BCE fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.891925 | 0.943 | 56444039 | [Class-balance exponent](../../validation/results/2026-09-21-class-balance-exponent.md) |
| S41 | 2026-09-21 20:58 | class-balance exponent k0.5 fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.899042 | **0.945** | 56443418 | [Class-balance exponent](../../validation/results/2026-09-21-class-balance-exponent.md) |
| S40 | 2026-09-21 18:47 | combined-labels scanner-split fold0 | x4 tau0.75 bs4 | (x1, scanner split) 0.902049 | **0.940** | – | [First run](../../validation/results/2026-09-21-combined-labels-first-run.md) |
| S39 | 2026-09-20 08:18 | stratified-sampling fold0 | x4 tau0.75 bs4 | (x1) 0.897515 | 0.933 | – | – |
| S38 | 2026-09-20 08:07 | control-B fold0 | x4 tau0.75 bs4 | (x1) 0.896582 | 0.933 | – | – |
| S37 | 2026-09-19 10:29 | ens-10x mixed + mirror TTA | x4 tau0.75 bs4 +mirrorTTA | (x1) 10x mean 0.877121 | no score | 56356227 | – |
| S36 | 2026-09-19 10:12 | ens-5xb0 + mirror TTA | x4 tau0.75 bs4 +mirrorTTA | (x1) 5x mean 0.877023 | **0.941** | 56355850 | – |
| S35 | 2026-09-19 09:59 | ens-5xb0 | x4 tau0.75 bs4 | (x1) 5x mean 0.877023 | 0.940 | 56355572 | – |
| S34 | 2026-09-19 08:41 | rexnet-fold0 | x4 tau0.75 bs4 | (x1) 0.883671 | 0.936 | 56353814 | – |
| S33 | 2026-09-19 08:40 | b0-fold0 | x4 tau0.75 bs4 | (x1) 0.886354 | 0.934 | 56353769 | – |
| S32 | 2026-09-18 08:04 | robust-center-20 topup-26 | x4 tau0.75 bs2 | (x1) 0.888021 | 0.938 | 56325019 | [Temperature and count](2026-09-18-temperature-and-count-submissions.md) |
| S31 | 2026-09-18 06:28 | ranked-filtered seed42 + tau0.75 | x4 tau0.75 bs2 | (x2) 0.889573 | 0.939 | 56322931 | [Temperature and count](2026-09-18-temperature-and-count-submissions.md) |
| S30 | 2026-09-18 06:22 | overnight 07-series-share-1.0 | x2 tau1 bs4 | (x1) 0.889582 | 0.929 | 56322795 | [Temperature and count](2026-09-18-temperature-and-count-submissions.md) |
| S29 | 2026-09-18 06:19 | ranked-filtered seed42 + tau0.75 (dup) | x3 tau0.75 bs2 | (x2) 0.889573 | 0.938 | 56322720 | [Temperature and count](2026-09-18-temperature-and-count-submissions.md) |
| S28 | 2026-09-18 06:11 | ranked-filtered seed42 + tau0.75 | x3 tau0.75 bs2 | (x2) 0.889573 | 0.938 | 56322551 | [Temperature and count](2026-09-18-temperature-and-count-submissions.md) |
| S27 | 2026-09-17 11:40 | ranked-filtered seed42 | x2 tau1 bs4 | (x2) 0.889573 | 0.933 | 56303522 | [Ranked-filtered](2026-09-17-ranked-filtered-submission.md) |
| S26 | 2026-09-17 07:58 | control seed 44 | x2 tau1 bs4 | (x2) 0.891627 | 0.931 | 56299154 | [Seed controls](2026-09-17-seed-control-submissions.md) |
| S25 | 2026-09-17 07:58 | control seed 43 | x2 tau1 bs4 | (x2) 0.889837 | 0.931 | 56299151 | [Seed controls](2026-09-17-seed-control-submissions.md) |
| S24 | 2026-09-17 07:32 | control seed 42 | x2 tau1 bs4 | (x2) 0.892995 | 0.928 | 56298600 | [Seed controls](2026-09-17-seed-control-submissions.md) |
| S23 | 2026-09-17 07:17 | EMA 0.999 | x2 tau1 bs4 | (x2) 0.893155 | 0.929 | 56298280 | [EMA](2026-09-17-ema-submission.md) |
| S22 | 2026-09-15 21:28 | champion fast runtime | x2 tau1 bs4 | (x2) 0.892360 | 0.936 | 56262847 | [Fast champion](2026-09-15-fast-champion-submission.md) |
| S21 | 2026-09-15 08:27 | champion with unknown_3d inference | x2 tau1 bs4 | (x2) 0.892360 | 0.936 | 56249900 | [Champion without 3D](2026-09-15-champion-without-3d-at-inference.md) |
| S20 | 2026-09-15 07:58 | overnight 00-baseline no 3D | x2 tau1 bs4 | (x2) 0.892763 | 0.934 | 56249358 | [Overnight 00-baseline](2026-09-15-overnight-00-baseline-submission.md) |
| S19 | 2026-09-14 12:16 | reflection sigma40 champion | x2 tau1 bs4 | (x2) 0.892360 | 0.936 | 56230573 | [Reflection sigma-40](2026-09-14-reflection-sigma-40-submission.md) |
| S18 | 2026-09-14 08:25 | 1A shared meniscus head | x2 tau1 bs4 | (x2) 0.891192 | 0.930 | 56226273 | [1A with two GPU processes](2026-09-14-1a-submission-with-two-gpu-processes.md) |
| S17 | 2026-09-14 07:37 | 1C compartment attention retry | x2 tau1 bs4 | (x2) 0.889430 | 0.928 | 56225269 | [Original 1C](2026-09-14-original-1c-comparison-submission.md) |
| S16 | 2026-09-14 07:24 | 1C compartment attention | x2 tau1 bs8 | (x2) 0.889430 | no score | 56224993 | [Original 1C](2026-09-14-original-1c-comparison-submission.md) |
| S15 | 2026-09-14 07:15 | sigma 40 central sampling | x2 tau1 bs2 | (x2) 0.892538 | 0.931 | 56224822 | [Sigma-40 overnight](2026-09-14-sigma-40-overnight-submission.md) |
| S14 | 2026-09-13 21:23 | 1B shared meniscus and OA heads | x2 tau1 bs2 | (x2) 0.892022 | 0.925 | 56215113 | [Meniscus and OA](2026-09-13-meniscus-and-oa-sharing-submission.md) |
| S13 | 2026-09-13 17:27 | study reflection sigma 80 | x2 tau1 bs2 | (x2) 0.890976 | 0.931 | 56211972 | [Reflection](2026-09-13-reflection-submission.md) |
| S12 | 2026-09-13 15:31 | 3D physical sampling | x2 tau1 bs2 | (x2) 0.889964 | 0.930 | 56210103 | [3dsampling](2026-09-13-3dsampling-submission.md) |
| S11 | 2026-09-13 13:57 | tonylin soft labels v3 | x2 tau1 bs2 | (x2) 0.892306 | 0.921 | 56208424 | [Tonylin](2026-09-13-tonylin-submission.md) |
| S10 | 2026-09-13 11:36 | spatial tokens five embeddings | x2 tau1 bs2 | (x2) 0.889506 | 0.929 | 56206027 | [Spatial token](2026-09-13-spatial-token-submission.md) |
| S09 | 2026-09-12 17:38 | regional spatial fusion 27 queries | x2 tau1 bs2 | (x2) 0.884169 | 0.920 | 56190442 | [Spatial-fusion runtime gate](2026-09-12-spatial-fusion-runtime-gate-and-submission.md) |
| S08 | 2026-09-12 09:37 | crop140mm control | x2 tau1 bs2 | (x2) 0.886122 | 0.925 | 56182948 | [140-mm crop](2026-09-12-140-mm-crop-submission.md) |
| S07 | 2026-09-12 08:27 | overnight 05-faster-head | x2 tau1 bs2 | (x2) 0.877244 | 0.903 | 56181919 | [Overnight fold-0](2026-09-12-overnight-fold-0-submissions.md) |
| S06 | 2026-09-12 08:21 | overnight 04-higher-lr | x2 tau1 bs2 | (x2) 0.882938 | 0.918 | 56181821 | [Overnight fold-0](2026-09-12-overnight-fold-0-submissions.md) |
| S05 | 2026-09-12 08:20 | overnight 02-longer | x2 tau1 bs2 | (x2) 0.883536 | 0.913 | 56181804 | [Overnight fold-0](2026-09-12-overnight-fold-0-submissions.md) |
| S04 | 2026-09-11 16:52 | baseline4 rexnet200 x3 | x3 tau1 bs2 | (x2) 0.872510 | 0.912 | 56169445 | [Image count](../inference-image-count-and-memory.md) |
| S03 | 2026-09-11 16:51 | baseline4 rexnet200 x2 | x2 tau1 bs2 | (x2) 0.872510 | 0.913 | 56169438 | [Image count](../inference-image-count-and-memory.md) |
| S02 | 2026-09-11 14:59 | baseline4 rexnet200 x1 | tau1 | (x2) 0.872510 | 0.910 | 56167669 | [Baseline3 artifact](2026-09-11-baseline3-fold-0-submission-artifact.md), [image count](../inference-image-count-and-memory.md) |
| S01 | 2026-09-11 11:46 | baseline3 resnet50d | tau1 | 0.849819 | 0.878 | 56164711 | [Baseline3 artifact](2026-09-11-baseline3-fold-0-submission-artifact.md) |
