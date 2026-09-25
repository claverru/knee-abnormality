# Submission ledger

All 56 Kaggle competition submissions, newest first, with submission times in UTC. The table is
regenerated from the live submission list and the local bundles by
[`update_submission_ledger`](../../../scripts/tools/update_submission_ledger.py),
which also writes `outputs/submission/kaggle-ledger.json`. The dated records in this folder cover
S01–S32; the baseline4 image-count comparisons are recorded in
[Inference image count and memory](../inference-image-count-and-memory.md), and the 2026-09-19
ensemble submissions S33–S37 are recorded in
[2026-09-19 ensemble submissions](2026-09-19-ensemble-submissions.md).

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
launches.

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

| S | Submitted (UTC) | Label | Inference | Fold 0 | Public | ID | Record |
|---|---|---|---|---|---:|---:|---|
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
