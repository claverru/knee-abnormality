# Evidence and coverage

The review was completed on **2026-09-10**, covering selected primary solution
discussions/write-ups, author repositories, public notebooks, organizer pages,
and original papers. It is a targeted review of transferable methods, not a
complete enumeration of every RSNA competition, forum post, or MRI paper.
Radiograph-only competitions received less attention because the volumetric and
multi-series problems are better matched by the sources above.

- **Author report:** a solution author's claimed score, ablation, or experience.
  It was not reproduced here. A winning ensemble does not isolate the value of
  each component, and multiple solutions from the same authors are not
  independent replications.
- **Paper result:** the authors' experiment on its stated dataset and endpoint.
  External evaluation and controlled ablations are stronger evidence than a
  leaderboard anecdote, but neither establishes transfer to this knee task.
- **Inspected implementation:** downloaded source establishes what the reviewed
  code contains. It does not authenticate attached weights, OOF arrays, reported
  scores, or the exact code used for a historical submission.
- **Project proposal:** our synthesis, to be evaluated locally before adoption.

Some Kaggle pages returned empty browser bodies. Indexed first-party text and
author-maintained repository versions supplied the cited content; partial
coverage is identified where it limits a claim. The public code review covers
four notebook files and four Python files, plus solution prose. Notebook source
cells are numbered from zero. Git commit IDs and downloaded-byte hashes are in
the manifest; external datasets and model weights were not downloaded or audited.

The current competition overview, data, and rules URLs were consulted but did
not expose readable bodies in this review. Current-task requirements therefore
retain their explicit **2026-09-09 verification date** in
[supervision](../modeling/supervision.md#official-labeling-context) and
[inference requirements](../inference/submission-requirements-and-output-contract.md#verified-submission-requirements).
This review does not newly authorize any external dataset. Revisit source
availability and competition rulings before incorporating one.
