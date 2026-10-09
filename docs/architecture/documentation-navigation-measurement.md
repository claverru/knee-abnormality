# Documentation navigation measurement

Measured 2026-09-20. A small paired pilot observed fewer retrieval operations with the revised
navigation, but does not establish that the indexes or lookup rules caused the difference.
Both participants located the canonical documents; neither opened an index directly, and the
participant assigned the previous rules skipped their mandatory root-index traversal.

## Observed retrieval cost

| Measure | Previous version | Revised version | Change |
|---|---:|---:|---:|
| Canonical documents cited | 12/12 | 12/12 | Same |
| Lookup operations | 39 | 31 | −20.5% |
| Retrieved characters | 139,534 | 130,438 | −6.5% |
| Retrieved whitespace-delimited words | 15,538 | 15,434 | −0.7% |
| Characters including the loaded skill | 146,169 | 138,094 | −5.5% |
| Direct index opens | 0 | 0 | Same |
| Searches returning index entries | 6 | 5 | — |

Operations are instrumented file reads, heading reads, section reads, searches and file listings,
not outer tool-call counts. Characters include emitted paths and line-number formatting. They
are not model-token counts. No conclusion follows about total inference cost, reasoning tokens,
agent elapsed time or index maintenance effort.

## Method and scope

Two fresh `luna_explorer` agents (`gpt-5.6-luna`, low reasoning) received the same twelve questions
in the same order and separate frozen documentation copies. One received the saved documentation
and docs-maintainer skill from before the navigation refinements; the other received the revised
versions. Both versions already contained the earlier index-only cleanup, so this comparison does
not measure that cleanup. Model sampling seeds were not exposed.

The questions covered two known paths, two exact terms, sampler determinism, BCE weighting,
OOF scoring, a historical overnight result, two canonical write destinations, ensemble behavior
and official CSV relationships. They were selected by the parent agent and include topics whose
navigation had just changed. Participants could reuse context between questions; this was one
session per arm, not twelve independent trials. Corpus inspection used the same instrumented
read-only helper. Task instructions restricted participants to their assigned copy and prohibited
inspection of the other arm or the live repository; this was not a filesystem access restriction.

A rubric identifying canonical documents and required facts was saved before either answer report
was read. Review was performed by the parent agent and was not blind to the condition. Canonical
citation recall is distinct from answer correctness and completeness: the previous-version report
omitted the final source-ranking tie-break and missing-field behavior, and did not supply the
requested runnable OOF scoring command. The revised report supplied those details. Both reports
also have smaller omissions or wording qualifications recorded with the artifacts.

## Per-question retrieval output

| Question | Previous operations | Revised operations | Previous characters | Revised characters |
|---|---:|---:|---:|---:|
| Known path: seeds and resume | 4 | 1 | 17,468 | 15,734 |
| Known path: detached submission logs | 2 | 2 | 4,603 | 5,793 |
| Exact term: sampling_sigma_fraction | 2 | 2 | 4,363 | 11,180 |
| Exact term: source_ranking | 2 | 2 | 1,980 | 2,537 |
| Sampler determinism | 3 | 2 | 11,743 | 4,590 |
| Class-balanced BCE | 3 | 2 | 7,537 | 5,742 |
| OOF scoring | 7 | 4 | 30,591 | 20,764 |
| Historical EMA/SWA controls | 5 | 4 | 33,737 | 33,691 |
| Write destination: host memory | 4 | 2 | 4,219 | 7,299 |
| Ensemble compatibility | 2 | 2 | 5,181 | 3,998 |
| Official CSV relationships | 2 | 5 | 5,183 | 14,881 |
| Write destination: vendor label rates | 3 | 3 | 12,929 | 4,229 |

## Interpretation

This is a preliminary retrieval-cost observation, not a validated comparison of faithfully
executed navigation strategies. The old-policy participant did not follow the prescribed index
walk. Search could still retrieve index descriptions, but the effects of revised wording,
section headings, navigation rules and ordinary variation between agent runs are not isolated.
The updated participant did not use the section helper; it read files or line ranges instead.
Consequently, this pilot does not demonstrate that the new direct section links were used.

The observed operation reduction is encouraging. The much smaller change in retrieved words,
noncompliance with the old rule, small selected sample and lack of repeated runs prevent a strong
claim of token savings or better answer quality. The appropriate decision is to retain the
measurement as provisional evidence, without treating its percentages as a general efficiency gain.

## Artifacts

Frozen documents and policy files, SHA-256 manifests, questions, the prespecified rubric,
instrumented traces, both answer reports, the answer review, helper source and metric summary are
stored under
`outputs/analysis/d/documentation-navigation-20260920T104254Z/`.

- [Setup](../../outputs/analysis/d/documentation-navigation-20260920T104254Z/setup.json)
- [Metrics](../../outputs/analysis/d/documentation-navigation-20260920T104254Z/summary.json)
- [Answer review](../../outputs/analysis/d/documentation-navigation-20260920T104254Z/answer-review.json)

The evidence directory is ignored by Git. The tracked report retains the measurements and their
limits; raw artifacts require the local output directory.
