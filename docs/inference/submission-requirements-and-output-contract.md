# Submission requirements and output contract

## Verified submission requirements

Reviewed on 2026-09-10, with the indexed official rules, timeline, evaluation
and data pages rechecked on 2026-09-11. The code-requirements page was read through
the Kaggle API on 2026-09-15 (see below); recheck it before final submission selection.

- The [official code requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements)
  require notebook submissions, disabled Internet, a maximum of nine hours
  for either CPU or GPU execution, and an output named `submission.csv`.
  Rechecked on 2026-09-15 through the Kaggle API, which returns every competition
  page's text (`ListCompetitionPages` in `kagglesdk`, via
  `KaggleApi().build_kaggle_client()`) where the web page does not render. Only CPU
  and GPU notebooks are listed, so **TPU notebooks cannot submit**; no page mentions
  TPUs or other hardware. Kaggle's TPU image also lacks kornia, polars,
  pydantic-settings, pydicom and Lightning.
- The [rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
  allow five submissions per day and two final selections. External data/models
  must meet the competition's accessibility and licensing conditions. Retain
  the public-label and pretrained-weight source/version records already used
  by training.
- The [timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline)
  currently sets entry/team-merger deadlines at October 15, 2026, final
  submission at October 22, and winners' deliverables at November 5, all at
  23:59 UTC. Winning solutions require code, weights and documentation, with
  additional presentation/public-release obligations described by the host.
- The [evaluation page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)
  scores the mean ROC-AUC across twelve targets. Write study-level confidence
  scores, preserving the official column names and order; do not threshold
  predictions into binary decisions.
- The [official efficiency evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/efficiency-prize-evaluation)
  considers final-selected (or automatically selected) submissions that beat
  the sample-submission benchmark on the private leaderboard. It minimizes:

  ```text
  Efficiency = AUC / (Benchmark - maxAUC) + RuntimeSeconds / 32400
  ```

  Here `maxAUC` is the best private-leaderboard AUC and `Benchmark` is the
  sample-submission score. This is not an AUC-per-second ratio. The public
  efficiency notebook updates daily and exposes ranks rather than full
  efficiency scores during development.

The tiny example-test run displayed on a public notebook page does not measure
hidden-test runtime. Measure the full path: imports, input discovery, decoding,
preprocessing, model loading, forward passes, ensembling, and CSV writing.
Training caches can save repeated experiments but do not predecode unseen
test studies. See [public runtime evidence](../research/public-research.md#data-access-and-runtime-lessons).

## Inputs and output contract

The [official data description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
states that scoring replaces the three example studies, series descriptors
and DICOMs with the actual test data, approximately 1,300 studies. Reports
are absent at test time. Discover study IDs from the mounted test files on
every execution; do not package example predictions or hard-code their IDs.
The two contrast flags are not guaranteed to coincide. The listed image
encodings include uncompressed little-endian, JPEG Lossless and JPEG 2000.

The local official template was also inspected on 2026-09-10. Its header is:

```text
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
```

The output should have exactly one row for every template study, twelve finite
probabilities in `[0, 1]`, no missing/duplicate IDs and no extra index or fold
column. Use the mounted template to order rows and columns. Write the final
file to `/kaggle/working/submission.csv`.
