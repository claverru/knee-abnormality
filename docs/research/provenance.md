# Provenance

Some competition strategy sections contain unverified working notes gathered
during early project planning. They retain useful hypotheses, unsuccessful
attempts, and changing conclusions, but lack the complete configurations,
checkpoints, and run artifacts needed for independent reproduction. Treat their
measurements as experiment leads rather than project results.

The [public-source investigation](public-research.md), checked on 2026-09-09 and
re-swept on 2026-09-18 and 2026-09-20, has separate provenance: direct Kaggle discussion text,
official pages, and downloaded public notebook source. It distinguishes reported
measurements from inspected implementations and project proposals. It did not
reproduce training or submission scores. Notebook snapshot identities are
recorded in its [source manifest](public-notebook-sources.json). The re-sweep
read threads through the forum API
(`discussions.DiscussionsService/GetForumTopicById`) and listed notebooks,
datasets and both leaderboards through the Kaggle CLI 2.0.2; no notebook was
executed and no attached weights or caches were audited. The 2026-09-20 sweep
repeated that method — all 122 topics listed through
`discussions.DiscussionsService/GetTopicListByForumId`, the twelve active since
09-18 read in full, notebook listings by date and by votes, the dataset listing
by date, the public leaderboard, and five notebook snapshots — and additionally
re-downloaded the two corrupted competition DICOMs plus a seeded eight-file
sample to check the host's repair
([data contract](../data-preparation/public-data-contract-and-decoding.md)).
The [2026-09-28 re-sweep](2026-09-28-public-resweep.md) used the same two forum calls, read only
threads and comments dated from 2026-09-20, listed notebooks by score through
`kernels.KernelsService/ListKernels` (`SCORE_DESCENDING`), downloaded the leaderboard with the CLI
and pinned the four cited notebooks in the manifest. It decoded, without running, the base64
inference script embedded in one of them.

The [public-label audit](../public-labels/INDEX.md), completed on 2026-09-10,
downloads and fingerprints the Steven Lee and Pilkwang label datasets plus
Steven v4's lixin73 dependency. Its local Polars audit measures coverage,
source transformations, uncertainty and reference agreement, separately from
the earlier public experiment reports. It does not reproduce label generation
or establish downstream training results. Dataset versions and file hashes
are pinned in [the label-source manifest](../../configs/labels/public-label-sources.json).

The [September 13 label comparison](../public-labels/2026-09-13-new-public-label-comparison.md)
adds six candidate publishers, reproduces available reference-score claims and
distinguishes copied gold targets from scoreable predictions. Its separate
[comparison manifest](../../configs/labels/public-label-comparison-sources.json) pins the inspected
releases. Its initial Tonylin recommendation was followed by a separate
[training comparison](../validation/results/2026-09-13-tonylin-label-comparison.md):
public AUC fell from 0.929 to 0.921 despite higher local AUC, and the baseline
returned to Steven v4. Label-table agreement and image-model results remain
separate evidence.

The [broader competition research](INDEX.md), reviewed on 2026-09-10,
has separate primary-source evidence from past RSNA and MRI competitions,
original papers, and author repositories. Its topic documents distinguish
reported results, inspected implementations, and proposed transfers to this
task. Four notebooks and four Python files were inspected without execution;
the [code manifest](source-manifest.json) pins their revisions and hashes.
This research does not establish new project experiment results or approve
external datasets for the current competition.

Sections explicitly marked as official or verified in the topic notes now
incorporate the 2026-09-09 public-source investigation. External links without
their own verification record remain research leads. Recheck volatile competition
requirements when preparing submissions or adding external data.

The environment document has separate provenance: direct host checks and
published Kaggle image metadata.

The metadata and reference-label sections also have independent evidence:
a Polars audit of five official competition CSVs downloaded on 2026-09-09.
[Competition metadata](../metadata/INDEX.md) records file fingerprints, reproduction
instructions, and limits; pixel and experiment claims were not verified by
that audit.

## How to use these notes

Use unverified working notes to propose experiments. Their settings are candidates to reproduce
and improve; results can change with the backbone, labels, sampling or implementation.

Start from the [current model](../modeling/current-model.md), the
[validation strategy](../validation/validation-strategy.md) and the fixed
[validation protocol](../validation/project-validation-protocol.md). Use the
[preprocessing notes](../data-preparation/unverified-preprocessing-notes.md) and
[modeling evidence](../modeling/INDEX.md) to identify comparisons, recording throughput and
inference cost alongside validation results. The [endgame backlog](../backlog/endgame.md)
records work still deferred to the final phase.

Retain useful negative results to avoid repeating unsuccessful experiments without a reason to
expect a different outcome. Record validated findings in their canonical topic documents and
keep them distinct from unverified working notes.
