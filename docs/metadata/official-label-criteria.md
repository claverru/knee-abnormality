# Official label criteria (host primary source)

Verbatim host text for the twelve target definitions and the label/report
relationship. Use this document as the rubric's primary source; the paraphrases in
[supervision](../modeling/supervision.md#official-labeling-context) and in
`z_labels/PROMPT.txt` derive from it.

Retrieved 2026-09-17 from the competition forum. Host posts are by
Po-Hao "Howard" Chen (`javaduke95`).

## Label Description (discussion 733343, pinned Overview)

> Models are evaluated on twelve binary labels, each indicating the presence or
> absence of a specific finding in the imaged knee. The labels, and the criteria
> used by the annotating radiologists, are summarized below. In each case,
> ambiguous or borderline findings ("on the fence") were graded as negative to
> favor specificity.
>
> **ACL tear:** A high-grade partial or full-thickness tear of the anterior
> cruciate ligament, meaning complete discontinuity of the ligament, or more than
> 50 percent of fibers disrupted, with or without secondary signs such as
> characteristic pivot-shift bone contusions. Mild signal change, degeneration, or
> thickening without discontinuity is graded negative.
>
> **MCL tear:** A high-grade partial or complete acute tear of the medial
> collateral ligament, with disrupted fibers and edema within and adjacent to the
> ligament. Low-grade sprains and chronic or remote stress changes are graded
> negative.
>
> **Medial meniscus tear:** Abnormal signal that definitely contacts the meniscal
> surface on at least two images, or a morphologic abnormality such as a
> truncated, diminutive, or displaced fragment, involving the medial meniscus.
> Intrasubstance degeneration that does not reach the surface is negative.
>
> **Lateral meniscus tear:** The same criteria applied to the lateral meniscus.
>
> **Medial compartment osteoarthritis:** A moderate or large area (roughly 1 cm or
> greater) of high-grade cartilage loss, defined as greater than 50 percent of
> cartilage thickness, in the medial compartment, with or without underlying
> subchondral marrow changes.
>
> **Lateral compartment osteoarthritis:** The same criteria applied to the lateral
> compartment.
>
> **Patellofemoral compartment osteoarthritis:** The same criteria applied to the
> patellofemoral compartment.
>
> **Joint effusion:** A moderate or large amount of fluid distending the joint.
>
> **Synovitis:** Inflammation and thickening of the synovial lining of the joint.
>
> **Baker (popliteal) cyst:** A moderate or large fluid collection in the
> characteristic location behind the knee.
>
> **Contusion:** A bone contusion, seen as bone marrow edema-like signal from
> impact, without a discrete fracture line.
>
> **Acute fracture:** An acute cortical break or fracture line.
>
> Each study in the annotated reference set was independently labeled by two
> subspecialty-trained MSK radiologists, with disagreements adjudicated by a third
> radiologist to produce a single consensus ground truth. Labels are assigned at
> the level of the whole examination, for a single knee.

A participant asked in the same thread whether "the same criteria applied" under
*Lateral meniscus tear* should read *medial*; that question is unanswered as of
retrieval. The text reads consistently if *Lateral meniscus tear* inherits the
medial definition with the laterality swapped.

## Labels versus report text (discussion 733826, host reply 2026-08-08)

Answering an audit of 20 of the 58 reference studies, the host confirmed, verbatim:

> **Were the labels assigned independently from the MRI images, rather than
> extracted from the reports?** Yes
>
> **If image interpretation and report text disagree, should the image-derived
> label be considered authoritative?** Yes. Note that only a small sample of
> provided data contains both. It is intended to help participants surface this
> conclusion.
>
> **Do negative labels mean that the finding was confirmed absent, or can they
> also mean that it was not annotated?** It means the finding was annotated as
> absent using methods outlined in Overview discussion post under Label
> Description section.
>
> **Are multiple reports or bilateral examinations sometimes associated with one
> StudyInstanceUID? If so, how were the study-level labels consolidated?** Yes. In
> clinical practice, both knees may occasionally be scanned under one
> StudyInstanceUID. For the challenge, each bilateral study or bilateral report was
> individually reviewed, and the released report text or DICOM metadata was
> adjusted as needed to provide sufficient information for participants to
> disambiguate the relevant study/studies.

The third answer is the load-bearing one for supervision design: **there is no
missingness in the target.** Every finding was adjudicated for every reference
study, so a report's silence is an unobserved label, never an absent one. This is
the requirement-side counterpart to the measured rejection of
[training-label masking](../history/sharing/2026-09-13-label-masking-result.md).

## Report text does not determine the label (discussion 733491, host reply 2026-08-07)

> The radiology reports in the training set are deidentified versions of the
> original contributor reports. They are provided as clinical text and may contain
> ambiguous wording, internal inconsistencies, or findings that do not map
> one-to-one to the challenge's binary targets.
>
> Similarly, marrow edema, cartilage findings, and terminology in the narrative or
> impression do not by themselves determine the contusion, osteoarthritis, or
> other challenge labels. Those labels reflect the standardized image-review
> criteria.
>
> A report may therefore use language that appears inconsistent with the
> corresponding image-based label because the original report and the image-based
> labels were generated through different processes.

The same thread records a participant observation that reports themselves
mislocalize compartments — a chondral lesion of the lateral patellar facet whose
impression reads "lateral compartment chondromalacia". Compartment terms in report
text are therefore not reliable evidence of compartment, which bounds
report-derived Lateral OA and PF OA quality independently of extractor quality.

## Reported ceiling for report-only labeling

**Participant claim, not host-verified** (discussion 733826, opening post): a
manual report-only review of 20 of the 58 reference studies under a strict textual
rule, 240 decisions, reported 82.5% overall agreement with the provided labels,
73.1% positive predictive agreement (68/93) and 80.0% positive recall (68/85).
Small sample, single reviewer, selection unstated. Treat as an order-of-magnitude
ceiling for report-only extraction rather than a measured bound.

## Retrieving competition discussions

Forum pages render only a JavaScript shell to unauthenticated fetches, and the
`kaggle` CLI has no discussions command. The authenticated SDK reaches them by
topic id:

```python
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.discussions.services.discussions_api_service import (
    ApiGetTopicRequest,
    ApiListCommentsRequest,
)

api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as kc:
    request = ApiGetTopicRequest()
    request.id = 733343
    topic = kc.discussions.discussion_api_client.get_topic(request).topic
    request = ApiListCommentsRequest()
    request.topic_id = 733343
    request.page_size = 50
    comments = kc.discussions.discussion_api_client.list_comments(request).comments
```

`get_topic` returns the opening post and `list_comments` the replies, both as JSON
with HTML in `content`. **`ListTopics` returns HTTP 403** for a personal token, so
topics cannot be enumerated or searched this way; reach them by known id or by
following `/discussion/<id>` links inside retrieved content.
