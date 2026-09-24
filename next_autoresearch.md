# Next autoresearch campaign

Defined as the [near-misses campaign](autoresearch/campaigns/near-misses/INDEX.md) (validated, not
started); its seed files are the canonical specs, and this list is the selection record.

These ideas are ranked by expected gain. Ideas 1, 2 and 5 rest on measured evidence; the
estimates for the rest are guesses. All of them fit the current recipe, and each runs against
`knee-geometry-merged-2` (seed 638120755).

1. **Shared + residual meniscus head:** about +0.002. It measured +0.0025 over `k05-clean`, with
   an 80% interval of [+0.0008, +0.0041]. Against the incumbent it lost only in Fracture, which is
   seed noise. Retrain it without the fallback change it was trained with.
2. **Folded medial position:** about +0.002. In the training-only half-knee detector, the
   medial-lateral coordinate becomes distance from the midline, and its gradients stay out of the
   shared position and normalization layers (the gradient-isolation variant). It measured +0.0027
   over `k05-clean` (80% lower bound +0.0011); an earlier version without gradient isolation
   measured +0.0020. The menisci improved in both. Retrain it without the fallback change.
3. **#1 + #2 combined:** about +0.002 to +0.004 (guess). A follow-up run after both. They change
   different parameters (the main classifier and the training-only detector) but target the same
   two labels, so the gains may overlap rather than add up.
4. **Full side-split meniscus readout:** up to about +0.002 (guess). A follow-up after #3,
   from the RSNA lumbar solution (claverru/RSNA-lumbar), which shared one head per condition across
   sides and levels. One meniscus attention row and one classifier, plus a per-side correction,
   read the medial-half tokens and the lateral-half tokens separately, with folded coordinates, at
   inference as well as training. Medial and lateral knee compartments are not mirror images, so
   keep the per-side correction and do not mirror pixels between halves.
5. **Early-learning regularization:** about +0.001. Local +0.0013; public 0.943 (S57) against
   0.942 (S48). Its ligament gain should add to the current model's gains. Details:
   [early-learning regularization](docs/ongoing/early-learning-regularization.md).
6. **Side-normalized 3D position:** up to about +0.001. Never run. Each token gets its 3D patient
   position relative to the study (or knee) centre in mm, with x flipped for left knees, encoded
   with fixed sine features and a small MLP. It **replaces** the 1D and 2D embeddings. Unknown
   side keeps y and z and uses a learned x vector; untrusted geometry uses a learned
   unknown-position vector. The related token-attention run raised menisci and OA, and its losses
   look like seed noise; that run also added an attention block, which this idea does not. Do
   not require a shared frame-of-reference UID across series: every accepted study fit has
   several, so that gate left the study-anchored depth run with zero active studies. Log how many
   studies and tokens use the encoding each epoch.
7. **Front-back attention bias for PF OA:** about +0.001 (guess, backed by the adopted bias). The
   adopted medial-lateral anatomical bias (fixed 20 mm width, learned strength, +0.0023 at
   adoption) extended to the front-back axis: PF OA attention gets a signed pull toward the front of
   the knee, measured along patient y from the knee centre. In the spatial importance map PF OA has
   the strongest front-back contrast of any label (+1.85). The retired attention prior failed
   differently: it learned position and width for all labels from a nearly flat start, and they
   barely moved. This bias can also give #8 its patellofemoral region.
8. **Three-compartment OA detector:** about 0 to +0.002 (guess). One cartilage-degeneration
   detector shared across the medial, lateral and patellofemoral compartments, each with its own
   readout, the way the lumbar solution shared across spinal levels. It targets PF OA (about
   0.865), which would learn from the more frequent tibiofemoral cases. It needs a patellofemoral
   region from the study geometry and is scoped to the OA family.
9. **Attention-token dropout, random masking:** about 0 to +0.001 (guess). During training,
   drop a few randomly chosen valid tokens from each label's attention softmax and renormalize;
   evaluation is unchanged. It aims to stop attention collapsing onto a few tokens. The only
   support is ACMIL in histopathology; there is no project evidence. Every study–label pair keeps
   at least one valid token inside the meniscus own-side mask. Spec:
   [small changes](docs/backlog/small-changes.md#attention-token-dropout).
10. **Attention-token dropout, top-attention masking:** about 0 to +0.001 (guess). A follow-up to
    #9 that drops the same number of tokens, drawn from the highest-attention ones instead of at
    random. It is kept only if it beats #9.
11. **Plane/contrast-specific readouts:** 0 to +0.001 (guess). Each plane/contrast type gets its
    own readout before the results are combined; it is the same for every label. It suits PF OA,
    whose evidence comes from axial fluid (29%) and sagittal (54%), with almost none from coronal.
    Spec: [small changes](docs/backlog/small-changes.md#planecontrast-specific-readouts).
12. **Any-OA auxiliary target:** 0 to +0.001 (guess). A training-only whole-knee head predicts the
    maximum of the three OA soft labels. Reports often say "tricompartmental" or leave the
    compartment vague, so a knee-level target is denser and less noisy, and PF OA features learn
    from every OA case.
13. **Slice orientation in the 3D position:** about 0 (guess). A follow-up after #6 that adds
    each slice's normal vector to the 3D position input, as the lumbar solution did. It mostly
    matters for oblique series, whose line is closed.
14. **Plane-negative half-knee MIL:** about 0 to +0.001 (guess). Never run. In the training-only
    half-knee loss, a positive half still needs evidence somewhere, but a negative half must
    show no tear evidence in every plane it has, scored separately per plane. Both menisci use
    the same rule, with the existing class-balance weight and auxiliary coefficient. It is the
    only idea that refines what the adopted half-knee loss teaches rather than which tokens it
    sees.
15. **Attention-coupled meniscus MIL:** at or below 0. It crashed on a logging bug before training.
    Detector-ranked pooling, the closest idea tried, lost −0.0018.

## Process

- Compare new runs against the re-seeded control, not the lucky incumbent.
- Family-scoped rules are authorized (user decision 2026-09-24) for the meniscus family and the
  OA family, including PF OA in #7 and #8: one rule per family, as each idea specifies. Every
  other label keeps its path.
- Judge loss changes on soft AUC.
- Seeds run first, in this order; the researcher wakes only after all of them finish
  (`backlog.seed_order: forced`, implemented 2026-09-24).
