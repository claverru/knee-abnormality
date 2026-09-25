# Knee side, appearance and anatomical location

Recorded on 2026-09-14. These are eight approaches, with stable numbers for
discussion. Approach 1 was implemented, [tied the champion](../validation/results/2026-09-14-knee-side-embedding-result.md)
and was retired because its checkpoint ignored side; its
[plan and sources are archived](../history/knee-side-sources/README.md). The other
approaches remain unimplemented; see the [big bets](../backlog/big-bets.md#position-and-knee-side).
The simplest candidate is an additional left/right embedding; coordinates
with a common anatomical orientation are the strongest inexpensive alternative.
Neither has a measured validation or public result.

## Fundamental hypothesis

Knowing whether an image depicts a left or right knee may help interpret its
spatial features. Appearance describes tissue and possible abnormality;
position describes where that evidence lies; knee side connects patient
directions to medial/lateral anatomy. Side may already be inferable from images,
so explicit metadata could simplify learning without supplying fundamentally
new information. Its incremental value is unmeasured.

This aligns right-knee medial anatomy with left-knee medial anatomy, and
likewise for lateral. It does not require medial and lateral disease classifiers
to become equal or similar. Keep this hypothesis separate from the
[classifier-sharing experiments](../history/sharing/INDEX.md) and
[soft-coupling regression](../validation/results/2026-09-14-meniscus-soft-coupling-result.md).

The [champion model](../history/model-lineage.md#five-embedding-predecessor), current when this was written, adds appearance,
1D position, 2D position, 3D position, plane and contrast embeddings before
LayerNorm. Target-specific nonlinear attention pools those combined vectors;
independent linear classifiers produce twelve predictions. It has no explicit
knee-side input or token-to-token attention in the pooling head.
These facts were checked in `model.py`, `spatial_tokens.py` and `dataset.py`
on `meniscus-soft-coupling`, parent commit `ea19772`. The subsequent removal of
the soft-coupling training code restored that champion's training recipe;
the model and loader behavior described here are unchanged. The production
3D encoder is a 3→64→256 GELU MLP.

Centered 3D coordinates retain patient LPS directions. In DICOM's human
coordinate convention, increasing X points toward the patient's left.
Interpreting that direction as medial or lateral depends on knee side and
joint pose. The sign of an absolute DICOM coordinate does not itself identify
the knee. [DICOM image-plane geometry](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html).

## Eight approaches

### 1. Add a knee-side embedding

Add a learned left/right vector to each study token alongside the existing
embeddings, before LayerNorm and nonlinear attention:

```text
token_i = appearance_i + position(xyz_i) + side_embedding(side) + other_embeddings_i
```

The same side vector is broadcast to every token of the study. Two vectors
at width 256 add 512 parameters. The
archived first-experiment plan (`f076a09:docs/history/knee-side-sources/knee-side-embedding-plan.md.txt`) specified conservative
metadata assignments, zero-initialized vectors, a fixed zero contribution for
unknown side and study-level side dropout. All attention outputs and disease
classifiers remain independent.
The downstream network learns how side changes the meaning of appearance and
position. Added inference work is negligible relative to the backbone.

This is the smallest direct extension of our embedding sum. It supplies useful
context without enforcing a spatial interpretation. That freedom also permits
irrelevant associations. Placing side only after pooling cannot influence which
evidence is selected; adding a study-constant scalar only to final attention
scores cancels under the token softmax. The proposed placement precedes the
nonlinear attention computation.

### 2. Condition the position encoder on knee side

Pass side into the position MLP together with coordinates:

```text
token_i = appearance_i + position(xyz_i, side) + other_embeddings_i
```

The coordinate mapping can learn a different interpretation for each knee.
Side-dependent scaling and shifting of spatial features is another conditioning
variant, related to [FiLM](https://arxiv.org/abs/1709.07871). FiLM is a general
mechanism demonstrated on other visual reasoning tasks; it is not knee-specific
evidence for this proposal.

The change stays inexpensive and puts side directly into spatial processing.
The anatomical transformation remains learned. Restricting side to the 3D
branch reduces its usefulness when that branch is unavailable or dropped.
Our current model substitutes a learned unknown vector for unreliable geometry
and drops available 3D for 10% of eligible training studies.

#### Difference between 1 and 2

In approach 1, the position encoder produces the same vector for a given
coordinate regardless of knee side. The left/right embedding supplies separate
context. Before LayerNorm, changing side adds the same vector difference at
every position; subsequent nonlinear layers can nevertheless produce different
effects at different positions and appearances.

In approach 2, changing side can change the positional vector differently at
each coordinate, inside the position encoder itself. For example, the mapping
can learn to reverse the interpretation of the patient-left direction before
position is combined with appearance.

Both approaches can ultimately learn interactions between side and position.
They differ in where those interactions are learned and how computation is
organized, not in a guarantee that one will outperform the other. Approach 2
can still encode a constant side effect and influence the final prediction
through the embedding sum; it does not prevent prevalence shortcuts. Approach 3
below supplies a known coordinate transformation explicitly instead of asking
either network to discover it.

### 3. Use coordinates with a common anatomical orientation

Multiply the centered patient-X coordinate by a side-dependent sign: +1 for
a right knee and −1 for a left knee. Feed the transformed coordinates into
the existing positional MLP. This makes the signed axis approximately point
toward medial anatomy for both sides under ordinary knee positioning.

The change requires essentially no extra learned capacity and directly removes
a directional ambiguity. It does not identify an exact compartment boundary or
compensate for all joint rotations. The accepted study center is a geometric
reference, not a guaranteed anatomical landmark. Images retain their original
orientation, so the backbone still handles mirrored appearances. Unknown side
requires an explicit handling rule before implementation.

### 4. Put images into a common knee orientation

Reflect one side into a chosen common orientation before feature extraction,
transforming pixels, triplet channels, positions and geometry consistently.
This could simplify both appearance recognition and location interpretation.

It changes the backbone input distribution and relies more strongly on correct
laterality. Its interaction with successful reflection augmentation must be
designed deliberately: canonicalizing immediately after a synthetic reflection
can undo that augmentation. This is a broader experiment than changing token
metadata or coordinates alone.

### 5. Separate attention inputs from pooled appearance

Use appearance, position and knee side to compute attention, while pooling
appearance features for the independent disease classifiers:

```text
attention_i = function(appearance_i, position_i, side)
pooled_appearance = sum_i(normalized_attention_i * appearance_i)
prediction = independent_classifier(pooled_appearance)
```

This gives location a clear role in selecting evidence. It preserves independent
decisions and does not require a shared meniscus classifier. Appearance features
can still carry spatial context, and attention can still exploit shortcuts;
the separation does not guarantee anatomical localization.

Simply applying the same linear classifier before rather than after weighted
pooling is algebraically equivalent. The substantive change is separating the
information used for selection from the vectors passed to the classifier.

### 6. Add soft anatomical preferences to attention

Use side and coordinates to give medial and lateral targets broad spatial
preferences, combined with learned visual attention. Retain independent disease
heads. This reduces how much coarse anatomy must be learned from study labels.

Center errors, joint rotation and overlapping receptive fields make rigid masks
risky. Soft preferences retain more flexibility. This introduces an attention
constraint and should be evaluated carefully in light of 1C; geometric grounding
is a different source of information, not proof that the constraint is beneficial.

### 7. Learn anatomical landmarks or regions

Localize joint compartments or menisci, then express positions relative to
those structures and classify their pooled features or localized images.
This addresses pose and center variation more directly than patient axes alone.

It introduces a localization task, possibly requiring annotations, pseudo-labels
or another model; localization errors can propagate into classification.
[Tack et al.](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2021.747217/full)
reported improved meniscus classification with auxiliary localization supervision
on OAI data. Different data and anatomical annotations prevent treating that
result as an expected gain in our competition.

### 8. Model relationships across regions and views

Let a relational module combine appearance, relative position and knee side
across regions or acquisitions. It could connect evidence likely to describe the
same structure and use surrounding anatomy to interpret ambiguous local features.
This includes the original idea of recognizing disease evidence and assigning it
to a compartment from its spatial relationship to other meniscus evidence.

It is the most expressive proposal and adds compute and attribution difficulty.
Study-level supervision can still reward co-occurrence shortcuts. A few anatomical
regions or queries are a more economical starting point than full attention over
all spatial tokens; no implementation or runtime estimate has been established.

## Metadata and reflection requirements

The [completed all-study tag/text audit](../metadata/2026-09-14-knee-side-metadata-audit.md)
supersedes the preliminary coverage sample below: those fields support possible
nonconflicting assignments in 56.98% of training studies, with manufacturer-dependent
missingness. The subsequent [coordinate investigation](../metadata/knee-side-coordinate-investigation.md)
adds strong evidence from absolute physical positions and reaches 96.85%
potential combined coverage under an exploratory margin rule. That is not
validated anatomical accuracy. It also provides candidate right sides for the
two example-test studies without tags. The linked audits own counts, conflicts,
coordinate-convention limits and reproduction. Image-based side inference remains
deferred by user instruction; metadata-only confidence remains under investigation.

A read-only Polars count on the existing geometry audit found recorded `L` or
`R` in **68 of 128 sampled studies**; 60 had neither. The 68 had no conflicting
L/R values among the sampled headers. `ImageLaterality` was absent in all 26,804
saved header rows, covering 721 series. This is not a full-dataset coverage or
correctness audit. The input is
[`headers.parquet`](../../outputs/analysis/geometry-metadata-20260912/headers.parquet);
study coverage counts unique nonempty L/R values in `Laterality`, grouped by
`StudyInstanceUID`. Its SHA-256 is
`2523ae738e05b068ae2991bc87a12a3f537863c00126e91b34593448d39ad5ea`.
The [DICOM definition](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.3.html)
distinguishes series laterality from image/frame laterality.

The retired embedding used an [audited policy](../metadata/knee-side-policy-audit.md)
for unknown or contradictory values, covering 96.14% of training studies. Hidden-test coverage
remains unmeasured. Training reports are not an inference-time source of
side. A learned side predictor is a separate possible response to missing data,
with its own accuracy and domain-shift questions.

Our whole-study reflection changes the effective knee side presented to the
model. For approaches using side, transform that input with the pixels and
coordinates; preserve the original source metadata separately. All medial/lateral
target identities remain unchanged. For approach 3, flipping both centered X
and the side sign preserves the canonical coordinate of corresponding tissue.

Correct 50% reflection reduces direct effective-side prevalence correlations
for eligible known-side studies. It does not remove metadata-missingness or
acquisition cues. Unknown-side behavior and consistency across acquisitions are
part of the data contract, not evidence that laterality must be complete.

## Comparison priorities

Approach 1 was compared first and [retired](../validation/results/2026-09-14-knee-side-embedding-result.md).
Its checkpoint ignored side and barely used the 3D position term, which weakens
approaches 2 and 3; approach 3 was also declined on 2026-09-14. If this direction is
revisited, approach 5 is the preferred structural separation. Complexity is not a ranking of expected
performance, and none of these proposals automatically fixes weak-label bias.

Use the existing fixed-fold recipe: Steven v4 with gold overrides, seed 42,
ReXNet, reflection 0.5, sigma 40, unchanged optimization and independent heads.
Do not combine the initial comparison with classifier coupling or a backbone
change. Record the metadata policy and identities as part of each run.

Keep full-fold and per-target AUC primary. Add left/right/unknown subgroup
results and retain exclusive-positive, co-occurrence and vendor diagnostics.
Improvement within each knee side would better support anatomical recognition
than a gain confined to changing rankings between side groups. Once compute is
available, paired reflection predictions with consistently transformed side
can measure orientation robustness; this is a diagnostic proposal, not a request
to run inference. One seed and fold cannot establish repeatability or hidden
leaderboard performance.

A 7×7 token is anchored at a coordinate but has an overlapping receptive field;
it is not an isolated patch. Normal-looking local evidence also cannot establish
that the entire compartment is normal. All approaches must retain aggregation
over relevant regions, and both compartments may independently be normal or
abnormal.
