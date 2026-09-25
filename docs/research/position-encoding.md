# Position information for compartment discrimination

Recorded on 2026-09-14. This combines our measured evidence, a code-level analysis of the
current model, a literature review of position encoding and geometric aggregation,
graph neural networks included, and first-principles ideas. The Phase 2 geometric
interaction layer was implemented as a token graph on 2026-09-14 and
[retired unused](../history/model-lineage.md#retired-token-graph) on
2026-09-15. The Phase 1 anchored
[attention prior](../history/model-lineage.md#retired-attention-prior) was implemented
on 2026-09-15, regressed and was retired on 2026-09-16, together with the per-token patient points; nothing else proposed here is implemented or scheduled. Measured numbers come from saved fold-0 artifacts. The CPU
analysis scripts that produced the new numbers are preserved in
`outputs/analysis/position-research-20260914/scripts/` but not cataloged.

## Answer

**Position can plausibly help, but not in the form we have tried, and the measurable
headroom for medial/lateral discrimination is small under the current labels.**

1. Every position change so far left compartment discrimination within noise.
   The changes were regional fusion, spatial tokens, 3D embeddings, sampling, reflection and knee side.
2. The model has three structural blockers that explain this without assuming
   position is useless:
   - position is swamped inside an additive sum;
   - the pooling head cannot relate tokens;
   - "medial" requires a product of knee side and coordinate sign, which reflection
     randomizes and additive embeddings struggle to learn.
3. Meniscus is already largely separated. OA is predicted as a shared degenerative
   score, which matches its shared weak labels.
4. The highest-value next step is a set of cheap diagnostics that show whether errors
   are position errors, variance or label noise. After that come head changes that give
   position a separate route to token selection. The most principled structural option
   is one geometric token-interaction layer (a graph or relative-bias transformer).

## 1. Measured evidence

### No position change moved compartment discrimination

The **margin AUC** below is the AUC of logit(medial) − logit(lateral) on
exclusive-positive fold-0 studies: 344 meniscus studies (281 medial-only, 63 lateral-only)
and 198 OA studies (143 medial-only, 55 lateral-only).

| Run | Macro | Lateral meniscus | Lateral OA | Meniscus margin | OA margin |
|---|---:|---:|---:|---:|---:|
| crop140mm (global pooling) | 0.8861 | 0.8802 | 0.8602 | 0.9669 | 0.8512 |
| spatial-fusion-radius (27 regional queries) | 0.8864 | 0.8775 | 0.8615 | 0.9652 | 0.8469 |
| spatial-tokens-five-embeddings | 0.8895 | 0.8853 | 0.8672 | 0.9718 | 0.8507 |
| 3dsampling | 0.8900 | 0.8969 | 0.8692 | 0.9697 | 0.8740 |
| reflection-sigma40 (champion) | 0.8924 | 0.8871 | 0.8623 | 0.9690 | 0.8791 |
| knee-side-embedding | 0.8924 | 0.8803 | 0.8671 | 0.9693 | 0.8567 |

- **Noise band:** the champion's bootstrap 95% intervals are [0.952, 0.984] for the
  meniscus margin and [0.825, 0.927] for OA. Across about 20 fold-0 runs, including
  hyperparameter-only changes, margins span 0.958–0.980 and 0.837–0.897. No position
  change leaves that band.
- **Regional fusion:** it received 4.9% attention against 36.6% under uniform weighting.
  Omitting it moved margins by at most 0.003
  ([fusion diagnosis](../validation/results/2026-09-13-per-plane-fusion-usage-diagnosis.md)).
- **Gains came from other targets:** the spatial-token gain concentrated in Baker's,
  MCL, contusion and ACL. Sigma 40 lowered all four medial/lateral AUCs while raising
  macro and public scores.

### Removal diagnostic on a trained checkpoint

These are removal measurements on the knee-side checkpoint, not trained ablations
([side interventions](../validation/results/2026-09-14-knee-side-embedding-result.md#side-interventions-the-checkpoint-ignores-knee-side)).

| Removed | Macro change | Lateral meniscus | Lateral OA | Largest losses |
|---|---:|---:|---:|---|
| 3D (replaced by unknown) | +0.00004 | +0.0007 | ≈0 | none; 873/904 studies have valid 3D |
| 1D | −0.00046 | +0.0019 | +0.0023 | mostly calibration (BCE +0.012) |
| All position | **−0.00146** [−0.0025, −0.0005] | +0.0033 | +0.0028 | Baker's −0.0079, fracture −0.0074, contusion −0.0053 |
| Knee side swapped | +0.00005 | ≈0 | ≈0 | none |

Position that matters helps **localized, less common findings**. Removing it slightly
*improves* the lateral targets and both margins.

### Headroom on saved champion predictions

- **Swap oracle:** flip medial/lateral logits on the 27 meniscus and 37 OA exclusive
  studies ranked wrong-side. Lateral meniscus rises from 0.887 to 0.909 and lateral OA
  from 0.862 to 0.871. Perfect side assignment is worth at most **about +0.004 macro**.
- **Label confidence matters more:** restricting AUC to confidently labelled studies
  (soft labels ≤0.3 or ≥0.7) raises meniscus AUCs by 0.02–0.03.
- **Wrong-side errors look like variance:** only 8 of 45 meniscus and 16 of 75 OA
  wrong-side studies are wrong in all 8 saved runs. An 8-run logit average lifts the
  meniscus margin from 0.969 to 0.977 but leaves OA at 0.878.
- **Cross-talk exists but is not linearly removable:** 69% of lateral-meniscus false
  positives are medial-positive, against a 37% base rate. Stacking the medial logit
  into a lateral logistic model gains nothing, so any fix must act at token level.
- **OA is a shared score:**
  - OA prediction correlation is 0.855, against a label correlation of 0.51.
  - Regressing the lateral OA logit on both labels gives 1.20·own + 1.01·medial.
  - Among studies negative in both OA compartments, medial and lateral soft training
    labels are identical in 99% (meniscus 77%).
  - The blinded audit found OA severity conflicts
    ([disagreement audit](../history/sharing/oa-and-meniscus-disagreement-audit.md)).
  - These coefficients barely change from global pooling through the champion, and
    removing position leaves them unchanged.
- **Routing is learnable:** 1C's learned compartment factor is used. Removing it drops
  macro AUC from 0.8894 to 0.8723; reversing it drops macro to 0.839 and meniscus mean
  to 0.655 ([sharing experiments](../history/sharing/INDEX.md)). Stronger structural
  coupling did not transfer to public scores.

## 2. Why the model cannot use position well

These points come from `model.py`, `spatial_tokens.py` and `augmentation.py`, plus a CPU
magnitude probe of the champion checkpoint on 6 studies.

1. **Swamped additive fusion.** Token norms:

   | Term | Norm | Varying part |
   |---|---:|---:|
   | Visual | 34.0 | 35.1 |
   | 1D | 11.3 | 7.1 |
   | 2D | 4.4 | 2.1 |
   | 3D | 2.4 | 1.1 |
   | Plane, contrast | 0.37 | — |

   - The 3D MLP's statistics match a freshly initialized MLP, so it barely trained.
   - After LayerNorm, the per-token logit spread is visual 1.90 vs 3D 0.06.
   - LayerNorm divides by the visual σ, which dilutes position most in strongly activated
     tokens, likely the lesions.
   - The fixed 1D sinusoid cannot be scaled down, and high-frequency channels alias at
     3 mm slice spacing.
2. **No token interaction.** Attention scores each token independently through a shared
   256→128 tanh layer, and classifiers are linear in pooled features. The head cannot
   express relative position ("on the fibular side of this plateau"), compartment
   boundaries, or that sagittal and coronal tokens show the same 3D point.
3. **Medial is a product, and reflection randomizes it.**
   - Medial ≈ knee side × sign(X − notch). Left/right knees are balanced
     (412/450 in fold 0) and reflection flips X with probability 0.5, so sign(X), the
     1D slice offset sign and side each carry no compartment information alone.
   - Learning such a parity from a zero-initialized additive input has almost no
     gradient signal ([Barak et al. 2022](https://arxiv.org/abs/2207.08799)). This fits
     the ignored side embedding better than "side is useless".
   - Reflection-invariant quantities (|X|, anterior-posterior Y, superior-inferior Z)
     barely separate the paired compartments but can locate unpaired targets.
4. **Content already encodes location.**
   - A token cell spans 20 mm; ReXNet's theoretical receptive field (317 mm) exceeds the
     140-mm crop; squeeze-excitation and zero padding add global and absolute in-image
     position ([Kayhan & van Gemert 2020](https://arxiv.org/abs/2003.07064);
     [Islam et al. 2020](https://arxiv.org/abs/2001.08248)).
   - Coronal and axial tokens see the fibula, MCL/LCL and plateau shape, so their
     compartment is visible.
   - **Sagittal is the likely gap.** A near-midline sagittal slab looks alike in both
     compartments, and only lateral slices show the fibular head or popliteus hiatus.
     Sagittal tokens cannot borrow handedness from other tokens.
5. **Noisy frame.** The study center is the scanner field-of-view center, with 10–20 mm
   tolerance, not the notch. The crop is centered on the native image, and inter-series
   motion is unmodeled.

## 3. External methods

### Evidence from comparable problems

- **Kaggle winners localized explicitly rather than encoding cleverly.**
  - RSNA 2022 cervical spine and RSNA 2023 abdominal trauma used segmentation crops and
    auxiliary segmentation.
  - [RSNA 2024 lumbar spine 2nd place](https://github.com/brendanartley/RSNA-2024-Competition)
    regressed separate left/right coordinates.
  - [1st place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution)
    used 3D localization and auxiliary attention losses. Its cross-series projection
    details are unverified.
- **[AnyMC3D (2025)](https://arxiv.org/abs/2512.12887):** query attention pooling beat
  transformer and LSTM slice fusion, and auxiliary segmentation added +0.011 AUC.
  In its reimplemented [Medical Slice Transformer](https://arxiv.org/abs/2411.15802), the
  best configuration used no positional embeddings.
- **Knee:**
  - [Tack et al. 2021](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2021.747217/full)
    localized medial/lateral meniscus subregions with box regression as an auxiliary loss.
  - [Astuto et al. 2021](https://pubs.rsna.org/doi/abs/10.1148/ryai.2021200165)
    segmented compartments before classification.
  - A [2024 meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11364796/) pooled
    tear location below detection (0.905 vs 0.939, location from two studies).
- **Why auxiliary inputs are ignored:** shortcut/simplicity bias, modality competition
  ([OGM-GE](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.pdf)),
  and noisy content–position cross-terms in summed embeddings
  ([TUPE](https://arxiv.org/abs/2006.15595)).

### Methods and fit

| Family | Examples | Fit here |
|---|---|---|
| Untied or gated position | [TUPE](https://arxiv.org/abs/2006.15595) separate position logits; [PETRv2](https://arxiv.org/abs/2206.01256) feature-gated 3D embedding | Cheap direct fix for swamping; the repo's approach 5 |
| Distance-biased attention | [Swin](https://arxiv.org/abs/2103.14030) relative bias, [ALiBi](https://arxiv.org/abs/2108.12409), learned Gaussian/RBF priors, [Query2Label](https://arxiv.org/abs/2107.10834) and DETR queries | Per-target or per-compartment anchor bias on attention logits; negligible cost |
| Continuous rotary encodings | [RoPE-Mixed](https://arxiv.org/abs/2403.13298), [STRING](https://arxiv.org/abs/2502.02562), LieRE | Native mm coordinates, translation-invariant; need query–key interaction |
| Multi-view geometry | [PETR](https://arxiv.org/abs/2203.05625), [Plücker rays](https://arxiv.org/abs/2402.14817), [GTA](https://arxiv.org/abs/2310.10375), [PRoPE](https://arxiv.org/abs/2507.10496), epipolar masks, BEVFormer grids | DICOM affines are exact, so orthogonal planes intersect along known lines; enables cross-plane intersection attention or a common 3D feature grid |
| Graph neural networks | [Patch-GCN](https://arxiv.org/abs/2107.13048), [PointNet++](https://arxiv.org/abs/1706.02413)/[DGCNN](https://arxiv.org/abs/1801.07829) EdgeConv, [GPS](https://arxiv.org/abs/2205.12454); knee [LGF-Net](https://link.springer.com/chapter/10.1007/978-3-031-16437-8_53), [MVGNN](https://link.springer.com/chapter/10.1007/978-981-96-2914-5_22) | kNN/radius graph over grid-pooled tokens in patient mm with cross-plane edges; sparse local version of a relative-bias transformer |
| Conditional convolutions | [CPE/PEG](https://arxiv.org/abs/2102.10882), TransMIL PPEG, [PTv3](https://arxiv.org/abs/2312.10035) | Context within a series only; no cross-series geometry |
| Localization supervision | Auxiliary segmentation, attention-mass losses, coordinate heads | Strongest Kaggle evidence; needs external segmenter or landmarks and a rules check |
| Position pretext | [MP3](https://arxiv.org/abs/2207.07611), [SAM embeddings](https://arxiv.org/abs/2012.02383), cross-plane correspondence contrast | Training-only; classification gain unverified |
| Content-derived frames | [Invariant Slot Attention](https://arxiv.org/abs/2302.04973), [learned canonicalization](https://arxiv.org/abs/2211.06489) | Origin/orientation from the model's own attention; speculative for knee MRI |

The knee GNN papers are small MRNet-scale studies, and whether their graphs use DICOM 3D
geometry is unverified.

**Reflection-equivariant models are unsuitable.** [EGNN](https://arxiv.org/abs/2102.09844)
and Clifford/[GATr](https://arxiv.org/abs/2305.18415) networks cannot separate a medial
from a lateral pattern by geometry. Every relative scheme is translation-invariant but
not mirror-invariant.

**GNN verdict:** worth testing only as the single interaction layer in section 4,
roughly interchangeable with a relative-bias transformer layer. It gives a
meniscus token context from nearby plateau, fibula or MCL tokens, which yields "lateral"
from relative anatomy without metadata side and stays consistent under reflection.
At k=16 over about 500 pooled nodes, compute is trivial. Risks: extra capacity fits
acquisition shortcuts under weak labels, and a large radius oversmooths medial into lateral.

**Not recommended:**
- Fourier, hash or grid-cell upgrades to the additive 3D MLP (frequency is not the bottleneck);
- per-study neural fields or re-slicing optimization (too costly for Kaggle inference);
- rigid atlas resampling (harms thick sagittal slices);
- location hypernetworks or mixture-of-experts before a frame exists (they inherit the parity problem).

## 4. Unconventional ideas worth keeping

- **Frame from the model's own attention.** The MCL is medial and the ACL sits in the
  notch, so the MCL minus ACL attention centroid points medial and the ACL centroid
  approximates the notch. Every token gets a signed, notch-relative coordinate without
  metadata side or image side inference. Diagnostic: how often the centroid sign agrees
  with audited metadata side.
- **Multiplicative handedness gate.** A study scalar h, constrained so reflection flips
  its sign, multiplies signed coordinates through FiLM on the attention input. The
  side × position product becomes explicit, giving non-zero gradient from the start.
- **Chirality-consistency pretext.** Occasionally mirror pixels but not coordinates, and
  predict per token whether image and geometry agree. This directly supervises the
  missing product; a frozen-feature probe first tests whether handedness is already known.
- **Hemi-knee counterfactual mixing.** Build training studies from one study's medial
  half and another's lateral half, with compartment labels from each and whole-knee
  targets masked. This breaks the measured cross-talk. Build it only if attention
  diagnostics show token-level misattribution.
- **Coronal teacher for sagittal tokens.** Distill a scalar medial-ness from coronal
  tokens into sagittal tokens at the nearest 3D point, so only one scalar crosses planes.
- **Mirror-pair asymmetry features** for OA (a token minus its counterpart across the
  midline). Plausible for OA, weak for menisci, and alignment is poor at 20 mm tokens.
- **Swap-noise likelihood** for paired targets if report laterality errors prove common.

Speculative ideas likely to fail at this data scale: implicit 3D disease fields,
position hypernetworks, rigid atlas canonicalization and cross-plane contrastive pretraining.

## 5. Recommended sequence

### Phase 0: diagnostics without training

| Test | What it decides | Cost |
|---|---|---|
| Read reports for the persistent wrong-side studies (8 meniscus, 16 OA) | If half or more are report or extraction laterality errors, stop compartment position work | CPU, about 1 hour |
| Plane-restricted inference (sagittal-only, coronal-only, axial-only presence masks) | If the sagittal margin is clearly below coronal at similar detection AUC, sagittal lacks handedness and position headroom exists; if margins match, content already solves compartment | 3 inference passes |
| Dump attention and token 3D positions | Share of lateral-head attention on medial tokens in medial-only false positives; MCL/ACL centroid frame agreement with metadata side | 1 inference pass |
| Frozen token probes | Compartment accuracy from features vs features plus a signed coordinate, by plane; stop if the sagittal gain is under about 3 points | CPU |

### Phase 1: head-only changes

Each is one fold-0 run judged by margin AUC, lateral cross-talk and macro AUC. Single-seed
macro noise is about ±0.003.

1. **Untied position path.** LayerNorm the appearance features alone, and give position
   its own normalized channel or feature gate into the attention logits only.
   Zero-initialize it, and add position-only auxiliary logits if it still goes unused.
   Appearance-only pooled values were [rejected](../validation/results/2026-09-21-untied-position-result.md)
   (−0.0018, confounded by repaired data).
2. **Anchored attention priors.** Add per-target learned 3D Gaussian bias on attention
   logits, outside LayerNorm:
   - |X|, Y and Z priors without a frame;
   - signed priors when the attention-derived frame from Phase 0 is reliable.

   Unknown 3D gets no bias.

### Phase 2: structural, if Phase 0 or 1 shows usable signal

3. **One geometric interaction layer** before target attention. Grid-pool tokens in
   patient mm (15–20 mm voxels), then apply a kNN EdgeConv graph or a relative-bias/RoPE
   transformer on Δxyz, with cross-plane intersection neighbours. It needs no side and is
   reflection-consistent. It changes how location is used, not just how it is encoded.
   On 2026-09-14 the user chose to pursue it ahead of Phase 0/1 as a deliberate risk; the
   token graph plan (`f076a09:docs/history/token-graph-sources/token-graph-plan.md.txt`) used raw
   tokens and a kNN graph. Two trained runs did not use the graph, so it was
   [retired](../validation/results/2026-09-14-token-graph-result.md) on 2026-09-15. The Phase 0
   plane-restricted inference on the champion was skipped by user decision, so
   plane-only margins exist only for the graph checkpoints.
4. **Localization supervision** (attention mass inside pseudo compartment masks or
   keypoints). Best external evidence, but it depends on a license- and rules-compatible
   knee segmenter.
5. **Hemi-knee mixing or the chirality pretext,** conditional on the diagnostics.

**Independent of position:** reflection test-time augmentation and ensembling target the
measured wrong-side variance directly.

## Relation to other documents

- The [knee-side research](knee-side-and-location.md) owns approaches 1–8. Approach 1
  was retired, approach 3 was declined, and approach 5 corresponds to Phase 1 item 1.
- The additive 3D MLP, untrained and unused, was deleted with the token graph
  implementation and stays removed after the
  [graph's retirement](../history/model-lineage.md#retired-token-graph). As
  recommended here, the geometry pipeline was kept for the attention prior's 3D term; the
  model's patient points were removed with the prior on 2026-09-16.
- Label quality limits OA regardless of architecture; see the
  [disagreement audit](../history/sharing/oa-and-meniscus-disagreement-audit.md).
- All measurements are single-seed and fold-0; no public or hidden per-target evidence exists.
