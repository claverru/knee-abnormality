# Supervision

Evidence: unverified working notes, explicitly identified official/public
sources, and the local CSV audit completed on 2026-09-09;
see [provenance](INDEX.md#provenance).

The supplied reference labels cover only 58 studies. Keep missing labels
distinct from negatives when preparing supervision for the remaining studies.
Public report-derived labels are a starting point to investigate; reported
extraction improvements have not consistently improved downstream models.

**Selected for the first baseline, 2026-09-10:**
[Steven Lee's v4 blend](public-label-audit.md#implication-for-the-first-baseline),
using the pinned file from Kaggle dataset version 6. Keep official reference
labels separate and apply them only to eligible training rows under the fixed folds.

The current [all-study protocol](validation.md#project-validation-protocol)
assigns all 4,407 studies to five folds without using gold status as a
stratification feature. Training uses gold where available and the selected
report-derived probabilities otherwise. The split's frozen effective label
table supplies validation targets, with a fixed 0.5 cutoff for AUC and separate
gold/report-derived diagnostics. Label-source experiments change training
supervision while preserving the evaluation table.

The [2026-09-10 public-label audit](public-label-audit.md) now pins and measures
Steven Lee's v1/v2/v4, Pilkwang v2, and the lixin73 component of Steven v4.
All cover the 4,349 report-only studies. Pilkwang omits one reference study,
so its extraction score uses 57 rather than 58 studies. Their uncertainty
encodings differ; consult the audit before constructing training targets.

## Official labeling context

The [host's annotation overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)
describes examination-level reference labels from two MSK radiologists, with
disagreements adjudicated by a third. Borderline findings were graded negative.
The rubric distinguishes high-grade ACL/MCL tears from mild degeneration or
sprain; meniscal surface-reaching tears from intrasubstance degeneration; and
moderate/large effusions and Baker's cysts from minor findings. OA requires a
substantial area of high-grade cartilage loss. Use the complete rubric when
designing extraction, rather than marking every abnormality mention positive.

The [official data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
confirms that reports are training-only and are not provided for test studies.
Keep report extraction offline and train the submission model to predict from
available image inputs.

The host explicitly permits hosted LLM/API processing of competition reports
for label extraction, subject to accessibility, cost, competition rules, and
service terms. Such processing is not by itself prohibited private sharing.
The clarification also distinguishes straightforward dataset registration from
substantial institutional approval barriers; non-commercial restrictions alone
do not disqualify a dataset under the competition's rules.
[Host clarification, including 2026-08-27 replies](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965).

**KneeCoT is specifically disallowed** under the host's 2026-08-25 ruling because
of its institutional agreement requirement. The host allows for reconsideration
if access terms change; no reversal appeared in the retrieved thread.
[KneeCoT ruling](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109).

For actual public label-table comparisons, missing-label experiments and
negative results, read [public supervision evidence](public-research.md#report-derived-supervision).

## Available supervision

The [local metadata audit](metadata.md#inventory-and-acquisition) confirms
4,407 training studies, each with a nonblank `Report`:

- 58 studies have all 12 reference labels, each binary `0` or `1`.
- 4,349 studies have all 12 labels missing; no studies are partially labeled.
- Missing cells mean absent supervision, not a negative label. Preserve a
  label-observation mask; keep any inferred label source/version separate
  from the reference labels.
- Exact report text repeats in 46 groups covering 177 studies. Identical
  wording does not establish the same patient or duplicated images. No
  conflicting reference labels were found among identical reports within
  the small reference-labeled subset; this is not evidence of label correctness.

In these notes, **gold** means supplied reference labels and **silver** means
labels inferred from reports or other model predictions. The downloaded
example `test.csv` contains only study IDs, with no reports or target labels.

Unverified context and research leads:

- Reports were said to be unavailable at inference, so text extraction is an
  offline supervision step for the image model.
- This [public-label discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932)
  is a relevant public resource. The [public review](public-research.md#report-derived-supervision)
  identifies concrete sources and author-reported version comparisons.

## Reference label distribution

Observed in `train.csv`, in the column order of `sample_submission.csv`:

| Target | Positive | Negative |
|---|---:|---:|
| ACL | 24 | 34 |
| MCL | 9 | 49 |
| Medial Meniscus | 26 | 32 |
| Lateral Meniscus | 23 | 35 |
| Medial OA | 15 | 43 |
| Lateral OA | 11 | 47 |
| PF OA | 21 | 37 |
| Effusion | 35 | 23 |
| Synovitis | 27 | 31 |
| Baker's | 12 | 46 |
| Contusion | 19 | 39 |
| Fracture | 18 | 40 |

Each target has 58 observed labels and 4,349 missing labels, with no values
outside `{0, 1}` among observed cells. These counts describe the supplied
reference subset, not prevalence across all training studies. MCL has only
nine positive examples, so check target coverage when designing
[validation splits](validation.md). External silver-label files and the official
evaluator were not part of this CSV audit.

## Label extraction results

These unverified scores were recorded as AUC on the 58 gold-labeled studies,
not accuracy or image-model public leaderboard scores. Exact checkpoints and
evaluation code are unavailable, and model names may be imprecise.

| Label source | Reported gold-label AUC on 2026-08-28 |
|---|---|
| Public regex extraction | 0.86 |
| Qwen3.8 27B | 0.89 |
| Each of two `ox alpha (glm 5.3 flash)` runs | About 0.90 |
| Combined extraction ensemble | 0.91 |
| Kimi K3 alone | 0.91; considered too expensive to run broadly |

One unnamed target was around 0.75 AUC; the rest were described as relatively
uniform. Per-target tables are unavailable.

An earlier local setup used the name `EschaLabs/Qwen3.8-27B-Escha-W2`, described
as a 2-bit model occupying about 10 GB, with thinking disabled and approximately
batch size 8 on an RTX 4090. These are reported setup details, not verified model
specifications. The exact checkpoint has not been identified.

## Downstream benefit changed over time

- **Early positive result, 2026-08-28:** adding two GLM label sources to regex
  plus Qwen reportedly moved the image-model public score from 0.918 to 0.923.
  This was associated with improved extraction AUC on gold labels.
- **Later qualification, 2026-09-08–09:** further label improvements reportedly
  did little for image-model performance. Models trained on improved labels
  also seemed more correlated and contributed less to ensembles.

These describe different stages and should both be retained. The available
evidence does not establish that label work always helps or never matters. Preserve
the label version when comparing training runs and distinguish label-extraction
quality from downstream model and ensemble quality.

## Attempts with little or no reported benefit

- DSPy/GEPA prompt optimization, more thinking, English translation, and
  bilingual prompting did not improve extraction in the reported attempts.
  A simple prompt with thinking disabled was preferred.
- Training a separate NLP label predictor performed worse than regex/LLM
  extraction.
- Smoothing labels where extractors disagreed left the public score unchanged.
- An LLM judge was tried, but no isolated result was provided.
- Adding textbook material was considered but not demonstrated. Matching the
  report-labeling methodology using the report text remains the stronger lead.

## Refinement and earlier eligibility uncertainty

Using out-of-fold image-model predictions reportedly improved inferred labels
against gold labels. Later analysis identified circular reasoning in attempts
to create more strong labels. See [validation](validation.md#local-to-public-score-relationship)
before treating these improvements as independent evidence.

Hosted LLM eligibility was initially uncertain. The
[official clarification above](#official-labeling-context) resolves that
uncertainty, subject to its conditions. One unidentified external dataset was
considered potentially usable but probably not worthwhile.
