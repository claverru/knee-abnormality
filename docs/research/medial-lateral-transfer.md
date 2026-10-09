# Medial and lateral disease transfer

## Recommendation

Current status, 2026-09-21: the original proposals below are historical design
notes owned by the [stopped autoresearch campaign](../../autoresearch/campaigns/medial-lateral-transfer/INDEX.md),
not an active queue. Learned anatomical family strength with independent heads
is now adopted, while the pooled adapter and other rejected implementations are
removed. [The current result record](../validation/results/2026-09-21-anatomical-attention-result.md)
supersedes the next-experiment recommendations below. No confirmation branch is pending.

The medial/lateral similarity hypothesis remains useful. The completed experiments reject treating stronger parameter sharing as a reliable improvement; they do not establish that the model should learn the two compartments independently at every level. The successful reflection model already shares its image backbone, token projection and part of its attention network across all targets. Additional sharing must improve on that existing transfer.

The first experiment was **weak meniscus classifier coupling at sigma 40, preserving independent attention, independent classifier vectors and independent biases**. The [completed coefficient-0.1 comparison](../validation/results/2026-09-14-meniscus-soft-coupling-result.md) reduced classifier distance by 20.5% relative to the champion, but lowered macro AUC by 0.001492 and meniscus-pair AUC by 0.004085. Retain reflection/sigma 40; this run does not justify adding OA or an immediate coefficient sweep. It tests a finite sharing constraint without adding inference computation; it does not rule out every coupling strength.

Candidate 2, **a small shared nonlinear disease feature module with independent compartment classifiers**, was implemented and [rejected on 2026-09-20](../validation/results/2026-09-20-compartment-adapter-result.md): macro AUC landed inside the control spread while the intended meniscus family fell −0.0114 [−0.0200, −0.0030] and the meniscus margin −0.0191 [−0.0392, −0.0034] against `13-control-repeat`. Two different sharing mechanisms have now regressed the same family, and the margin diagnostic says the cost is exactly the loss of medial/lateral distinguishability that unequal prevalence predicts. **A training-only shared auxiliary detector (candidate 3) is the remaining distinct proposal**; it encourages transferable features without sharing anything at inference. Neither approach automatically removes label or acquisition bias.

Any candidate needs two assessments: does it improve recognition of the intended disease, and does that gain survive the existing acquisition and disease-combination diagnostics? An increase in aggregate local AUC alone did not distinguish the sharing experiments correctly. Conversely, an isolated unfavorable public result does not identify the hidden causal mechanism.

Candidate 2's `model.compartment_adapter` option is a removal; its `"independent"` arm was never run because it is a reparametrization of the already-rejected `model.residual_readout`. Candidate 1 was [implemented, evaluated and retired](../modeling/retired-side-and-compartment-variants.md#meniscus-classifier-coupling) on `meniscus-soft-coupling`, using coefficient 0.1; its public score is unknown. Its training code was removed after the regression; archived sources and the completed analysis remain available. The other candidates remain proposals, with no new run scheduled. Reflection/sigma 40 remains the submitted champion. The recommendations concern the archived ReXNet comparison and should not be combined with a backbone change.

## The six relevant experiments

All six use Steven v4 supervision, the same gold-balanced fold 0, seed 42, reflection augmentation and twice the training image density for validation. Validation contains 904 studies, including 893 with report-derived labels and eleven with gold labels. The metric is macro ROC AUC over twelve targets. Public scores retain the precision recorded by Kaggle.[^1][^2]

| Experiment | Sigma | Shared final classifiers | Shared disease/compartment attention | Validation AUC | Kaggle public |
|---|---:|---|---|---:|---:|
| Reflection | 80 mm | None | No | 0.890976 | 0.931 |
| 1A | 80 mm | Meniscus | No | 0.891192 | 0.930 |
| 1B | 80 mm | Meniscus and OA, separately | No | 0.892022 | 0.925 |
| 1C | 80 mm | Meniscus and OA, separately | Yes | 0.889430 | 0.928 |
| 1C, sigma 40 | 40 mm | Meniscus and OA, separately | Yes | **0.892538** | 0.931 |
| Reflection, sigma 40 | 40 mm | None | No | 0.892360 | **0.936** |

At 80 mm, classifier sharing improves local macro AUC while public performance deteriorates. Adding the 1C attention constraint reverses those directions relative to 1B: local AUC falls while public AUC improves. At 40 mm, 1C leads locally by 0.000178 but loses publicly by 0.005. These differences matter competitively.

The pattern is not evidence that every component of sharing is harmful. Tightening sampling improves both local and public results for both architectures. The untested cells include meniscus-only sharing at 40 mm, classifier-only sharing of both families at 40 mm, and disease/compartment attention with independent final classifiers. The six runs do not isolate those interactions.[^1]

Public evaluation covers approximately 30% of the test set. Its ranking is relevant external evidence, not the final private ranking. The local fold has more studies but predominantly different label provenance. Sample size and annotation fidelity are separate considerations; neither validation set is automatically the more accurate estimator of private performance.[^3]

### Historical implementation and reproduction

The original two-stage proposal became three experiments: 1A and 1B evaluated shared classifiers sequentially; 1C evaluated attention sharing while preserving both selected classifier pairs. Each shared classifier is one actual 256-dimensional parameter vector trained by both compartment outputs, with independent biases. Meniscus and OA never share the same disease classifier. PF OA and the other unrelated targets remain independent. All twelve binary outputs retain official target ordering and allow both compartments to be positive.

The saved runs are `outputs/training/meniscus-classifier`, `outputs/training/meniscus-oa-classifiers` and `outputs/training/disease-compartment-attention`. Their effective configurations, source snapshots and input identities own reproduction. The original comparison used ReXNet-200, 224-pixel/140-mm inputs, sigma 80, reflection probability 0.5 and ten epochs, starting from pretrained initialization rather than fine-tuning the selected reflection checkpoint. Equal seeds do not guarantee identical stochastic sequences after architecture changes. The original choices to retain 1A and 1B were controls for the next experiment; the later public results and reflection/sigma-40 result supersede those choices as production recommendations.

In 1C, two disease vectors and one compartment vector act on the existing 128-dimensional attention hidden representation. Each of the four targets separately normalizes scores over valid study tokens before pooling. The other eight attention scorers remain independent. The learned compartment sign is not a fixed patient-coordinate sign or an image-half mask. Reflection preserves target identities; it never swaps medial and lateral labels. With only one disease family, the additive decomposition can re-express two independent vectors; sharing the contrast across both families creates the additional constraint examined below.[^5]

## What the local evidence actually supports

### There is some evidence of useful disease transfer

In 1A, both meniscus AUCs improve over reflection: medial by 0.002737 and lateral by 0.003406. The overall gain is only 0.000217 because other targets move in both directions. This is compatible with useful meniscus transfer plus unfavorable changes elsewhere in the shared representation. It does not establish that meniscus itself caused the public decrease.[^4]

The 1B result is less supportive of the intended mechanism than its aggregate score suggests. It improves four targets and worsens eight relative to reflection. Excluding contusion and fracture, the other ten targets' mean declines by 0.000903. Its shared OA pair also averages below reflection. Much of the apparent success is therefore outside the disease family the intervention was intended to help.[^4]

This distinction matters for experiment selection. A model can improve macro AUC through unrelated tasks while its proposed disease-sharing mechanism fails. A useful follow-up should report both meniscus targets, both tibiofemoral OA targets and the other eight targets explicitly, while retaining the competition metric as the primary endpoint.

### 1C imposed a real attention constraint

The four attention score vectors in 1C have the following form:

```text
meniscus medial: disease_meniscus + compartment
meniscus lateral: disease_meniscus - compartment
OA medial:       disease_OA + compartment
OA lateral:      disease_OA - compartment
```

The expressions denote linear scores on the same attention hidden features. Meniscus and OA must therefore have the same token-wise medial/lateral score contrast. Setting the compartment vector to zero does not recover the independent attention model: within each disease family, both attention distributions become identical. With a shared classifier, their logits then differ only by their fixed biases.[^5]

The saved interventions also argue against a simple capacity-only explanation. Removing the compartment factor reduced 1C AUC from 0.889430 to 0.872267; reversing it reduced AUC to 0.839237. Yet projecting trained 1B attention to a shared meniscus contrast changed macro AUC by approximately -0.000000081. A useful shared-contrast solution existed in that trained representation.[^5]

The more specific issue is the training path and the strength of the anatomical contrast. In 1B, the median attention-score contrast standard deviations were 0.781 for meniscus and 0.474 for OA. In 1C, both were 0.784. Scaling the learned factor affected the two families in opposite directions. These results motivate softer or disease-dependent sharing, but do not identify why Kaggle preferred one trained model.[^5]

### Two demonstrated vulnerabilities remain distinct

The blinded report audit found nineteen report-derived OA cells with explicit low-grade findings but confident positive training labels. The discrepancies occurred in both underlying label sources, so source agreement and confidence thresholds do not identify them reliably. Excluding those cells barely changed 1B's local advantage: +0.001047 became +0.001038. The audited errors are real evidence of a supervision problem, but not a complete explanation of the public reversal.[^6]

Acquisition sensitivity is a separate observation. A training-derived vendor-only prior achieved 0.617 validation AUC for lateral OA without images. Reweighting validation manufacturers could reverse the 1B/reflection ordering. This demonstrates a possible case-mixture mechanism; it does not prove the image model used vendor cues or reveal Kaggle's manufacturer mixture.[^6]

The meniscus audit agreed with the frozen labels on all 81 scoreable cells. That selected report sample cannot establish a population error rate, but it provides a concrete reason to investigate meniscus transfer before increasing OA coupling. A disease family should earn its own sharing decision.[^6]

## Why biological similarity does not require identical classifiers

There are three different propositions:

1. Some visual features of a disease transfer between compartments.
2. The optimal classifier of the pooled features is the same in both compartments.
3. The same medial/lateral attention contrast transfers between different diseases.

The first proposition is the motivating intuition. The second and third are stronger modeling assumptions. Related anatomy can share useful features while differing in background structures, informative feature combinations, severity distributions and reporting practices. Independent attention does not guarantee that its pooled outputs will be aligned well enough for one final weight vector.

Shared-plus-private multi-task models provide a formal middle ground: retain a common component while allowing each task to depart from it. Classic regularized multi-task learning explicitly models task vectors as a shared vector plus task-specific deviations. Cross-stitch networks instead learn mixtures of shared and task-specific features. These establish design principles, not expected gains for this competition.[^7][^8]

AUC also distinguishes two meanings of distribution shift. For a fixed target, changing only its positive fraction while leaving positive and negative score distributions unchanged does not change population AUC. Changing severity, acquisition, disease co-occurrence or annotation criteria can change those conditional distributions and therefore change AUC. A constant bias correction or positive temperature scaling cannot repair a ranking error.

The literature contains an unusually relevant example: Zech and colleagues manipulated between-hospital pneumonia prevalence and observed internal performance improvements that failed to transfer externally. It shows how source-specific disease associations can create misleading internal gains. It concerns chest radiographs, not knee MRI, and is evidence for a mechanism rather than an attribution of these results.[^9]

## Candidate 1: weak classifier coupling

Keep the reflection model's prediction equation for each compartment:

```text
logit_medial  = w_medial · pooled_medial + b_medial
logit_lateral = w_lateral · pooled_lateral + b_lateral
```

Add a training penalty for the meniscus pair:

```text
loss = existing_loss + lambda * ||w_medial - w_lateral||² / feature_width
```

The attention distributions and biases remain independent. The shared backbone receives gradients through both disease tasks as before. The additional penalty encourages compatible classifiers without requiring exact equality. The equivalent interpretation is a common disease vector with penalized compartment-specific deviations, following the regularized multi-task principle.[^7]

**Why this was selected first:** it changes one assumption, preserves all original prediction functions, introduces no extra inference work and requires no new labels. With lambda zero, the objective and parameterization recover the independent-head control. This is different from 1C, where removing the compartment term destroys the original ability to select separate evidence.

That recovery property is not a guarantee that a positive lambda will generalize better. Both tasks may still agree on a shortcut. A finite penalty can also pull classifiers together when their pooled feature distributions should remain different. Weight distance is a diagnostic of coupling, not a measure of anatomical correctness.

Use one weak, predefined coupling strength initially. Its normalization and reduction must be recorded, because the same numerical lambda means different things under different BCE and parameter reductions. A training-only gradient-scale check can verify that the penalty influences learning without dominating it. Track vector distance, vector norms and per-target AUC; a candidate whose vectors barely change does not meaningfully test sharing.

Do not implement an unconstrained shared vector plus two unconstrained private vectors and call it a new source of information. Without a constraint or regularizer, that decomposition is largely a redundant parameterization. Keeping the current independent vectors and adding the explicit penalty makes the experimental change easier to interpret.

Meniscus-only hard sharing at 40 mm is a useful later comparison if the weakly coupled model improves: it distinguishes a benefit of partial coupling from an interaction between meniscus sharing and sigma 40. It is an untested control, not a known inferior model.

## Candidate 2: shared disease features, independent decisions

**Measured and rejected 2026-09-20** ([result](../validation/results/2026-09-20-compartment-adapter-result.md)). The design below is retained to explain what was built and why the stopping rule closed it; it is not a candidate for a width or coefficient sweep.

A more substantial alternative preserves the independent output classifiers and inserts a small shared nonlinear module into the separately pooled representations of one disease family:

```text
adapted_compartment = pooled_compartment
                      + U_disease * GELU(V_disease * pooled_compartment)
logit_compartment = w_compartment · adapted_compartment + b_compartment
```

Both menisci use the same disease module, but their attention and final classifier vectors remain different. OA would receive its own module only in a subsequent experiment. A bottleneck width of 32 on 256-dimensional pooled features requires about 16,384 matrix weights per family, plus any biases. It operates after pooling, so its work scales with studies rather than all image tokens. Actual runtime remains unmeasured.

The shared module can learn nonlinear combinations useful to both compartments while the final heads assign them different importance. Initializing its output projection to zero preserves the original model's predictions at initialization. It does not preserve the later optimization trajectory or guarantee that the module will remain harmless.

This design is a project proposal inspired by feature-sharing methods. MTAN retains task-specific feature selection over a shared feature pool, while MMoE uses task-specific gates over shared experts. Their experiments support allowing tasks to select common information differently. Neither paper evaluates this particular residual adapter or these knee targets.[^10][^11]

Two details prevent misleading conclusions. First, a purely linear shared adapter before independent linear classifiers can often be absorbed into the classifier matrices; the nonlinear transformation supplies a meaningful new feature family. Second, an adapter may help simply by increasing head capacity. Compare a shared module with otherwise equivalent independent modules if the initial result merits attribution to sharing. The parameter counts differ and should be reported.

This is preferable to immediately introducing a large expert-routing network: there are only two compartments per family, limited training studies and a strong existing shared backbone. Flexible routing also learns from the same biased labels and can route using acquisition or co-occurrence cues. Flexibility supplies options, not a safeguard against shortcuts.

## Candidate 3: training-only shared disease supervision

Attach a shared auxiliary classifier to both separately pooled meniscus representations. Keep the main twelve outputs unchanged and retain all ordinary losses. The auxiliary classifier predicts each compartment's own label; it never asks medial and lateral predictions to be equal within a study.

```text
main_logit_c = independent_head_c(pooled_c)
aux_logit_c  = shared_meniscus_weight · pooled_c + aux_bias_c
loss        = main_loss + lambda * mean(aux_meniscus_losses)
```

The auxiliary objective encourages features that one disease decoder can recognize in either compartment. The independent main heads remain free to use other features or different combinations. Removing the auxiliary head at export adds no inference computation to the original model.

This separates a training preference from a mandatory final decision rule. However, its gradients still alter attention and the shared backbone. It can reproduce the same negative transfer or amplify the same label error when given excessive weight. Auxiliary-task research explicitly recognizes that an auxiliary loss may help at one stage and hurt at another; gradient similarity has been proposed as one way to moderate that influence.[^12]

The essential control is an otherwise equivalent pair of independent auxiliary heads at the same loss weight. Without that comparison, an improvement could be explained by emphasizing meniscus more strongly, rather than sharing a detector. The proposal therefore costs more experiments to interpret than the direct coupling penalty.

A study-level auxiliary label for disease in either compartment would test a different idea. It can reward general disease burden without improving localization. With soft marginal targets, the joint probability of either compartment being positive is also not determined by the two marginals; the usual probabilistic-OR formula assumes independence. Do not silently introduce that assumption into supervision.

A teacher distilled from 1B or 1C is not automatically a solution: it would teach the student the same predictions whose transfer is in question. Distillation becomes more persuasive only with a teacher or objective that adds independently supported knowledge.

## Mechanisms that address bias more directly

### Disease-combination coverage

A shared detector trained mostly on correlated disease can succeed without learning isolated disease well. Preserve the four joint-label groups for each family: neither, medial only, lateral only and both. Evaluate each target within strata of the other target whenever both classes are represented; report support and undefined AUCs rather than calculating an AUC inside a constant-label group.

If training evidence establishes inadequate isolated-disease coverage, a separate experiment can moderately rebalance the relevant training loss across joint-label groups. Normalize weights within the target or family, and avoid changing the other ten task losses merely because a study was oversampled. Uncertain group assignments should not receive large weights. This is a stress against dependence on co-occurrence, not an estimate of Kaggle's unknown disease prevalence.

Disease co-occurrence is not inherently an error or an invalid predictor. It may provide useful context. The concern is excessive reliance that fails when the association changes; forcing disease predictions to be independent could discard useful information. Conditional evaluation should establish the tradeoff before a training intervention attempts to suppress it.

Group DRO provides evidence that attention to predefined groups can improve worst-group generalization, but its original work also shows that naive worst-group training is insufficient without appropriate regularization. Just Train Twice reweights early errors, yet its published model selection still uses group-labeled validation. Neither establishes that upweighting our hardest weak-label cases is beneficial; those cases can be annotation errors.[^13][^14]

### OA annotation criteria

The strongest OA-specific evidence concerns severity semantics, not a generic shortage of label confidence. Before an OA intervention, examine a predefined random training-only sample using the same severity rubric as the completed audit. Distinguish explicit below-threshold disease from insufficient evidence, and preserve gold labels. A confident report-derived positive is not necessarily a reliable official-endpoint positive.[^6]

If the training sample confirms the mechanism, change OA supervision separately from sharing and sampling. That is a different causal experiment. Do not revise the frozen validation targets to make the new model appear better, and do not treat a report review as image adjudication. Improved annotation alignment may change the direction of the local metric precisely because the original proxy endpoint is unchanged.

Hidden-stratification research supports examining clinically meaningful subgroups rather than relying entirely on aggregate metrics. It does not establish that an automatically generated subgroup or a disagreement-enriched sample is a valid replacement endpoint.[^15]

### Acquisition robustness

Use manufacturer and field strength as diagnostic proxies, not as hospital identities or training priors to insert into inference. Report original and fixed reweighted evaluations. A training intervention should attempt to reduce documented conditional reliance without deleting medically relevant image variation.

Blanket domain-adversarial removal of manufacturer or compartment information is not the default recommendation. Domain-invariance theory gives counterexamples where aligned representations and good source accuracy still fail under conditional shift. Compartment identity is also necessary to locate the target. Removing all information correlated with it can remove useful anatomy.[^16]

A recent multicentre knee study provides a relevant caution. Its primary-abnormality AUC decreased from 0.898 internally to 0.852 and 0.812 on two external sets. Despite its multi-task title, the methods describe nine independently trained classifiers plus a segmentation component. It should not be cited as evidence that tied disease heads solve generalization.[^17]

### Optimization interference

PCGrad and related methods modify conflicting task gradients. Their evidence supports optimization interference as a plausible mechanism, not a test-distribution bias correction. Two tasks can have aligned gradients because both reward the same incorrect label convention. More recent task-priority work also questions treating gradient realignment as sufficient across parameters.[^18][^19]

Measure gradient alignment before choosing such a method, when compute becomes available. Record the parameter block: conflict in a shared classifier is different from conflict in the backbone. Repeated per-task backward passes can be costly for a twelve-target MRI model. The current saved results do not justify making gradient surgery the next default optimizer.

## Diagnostics with high information value

The most useful representation diagnostic is to freeze a trained model's backbone and attention, then fit independent final classifiers using training data only. Applied to 1C/sigma 40, this asks whether its existing representation contains usable information that the shared output rule does not expose. Applied to reflection/sigma 40 with tied, weakly coupled and independent readouts, it asks how compatible its existing pooled feature spaces are.

This is inspired by Deep Feature Reweighting, where last-layer retraining on a suitable reweighting set recovers useful features that the original classifier underuses. The published method relies on data where the problematic correlation is broken; ordinary refitting on the same weak-label training population is only a diagnostic adaptation, not a reproduction of that robustness guarantee.[^20]

The original 904-study validation set must never become fitting data. Reserving a subset after a backbone has already trained on it does not make that subset unseen by the representation. A genuine held-out reweighting design requires defining it before the corresponding representation training. Avoid changing the current validation protocol merely to make this diagnostic possible.

Saved prediction CSVs are insufficient for fitting new readouts: pooled feature vectors are needed. Extracting those features requires model execution and is deferred while the GPU is occupied. Once features exist, these small fits can run on CPU. No feature extraction, training or model intervention accompanies this report.

Attention plots alone are weaker evidence than paired prediction interventions. A disease-looking map does not establish that the prediction depends on disease morphology. Likewise, success at predicting vendor from an embedding establishes available information, not that the disease head relies on it. Keep such observations separate from intervention results.

The existing meniscus margin AUC favors 1B over reflection, 0.980173 versus 0.972321, despite its worse public score. Within-study ordering and per-target ranking across different studies are different tasks. A better margin diagnostic alone cannot establish better external disease recognition, and the exclusive-positive subset omits both-positive cases.[^4]

## Priorities and comparison design

| Priority | Candidate | Assumption being examined | Main remaining risk | Inference addition |
|---|---|---|---|---|
| Completed; no improvement at 0.1 | Weak meniscus classifier coupling, sigma 40 | Related classifiers should be similar rather than identical | Coupled shortcuts or incompatible pooled features | None |
| Second architectural option | Shared nonlinear meniscus adapter, independent heads | Transferable disease features need different final combinations | Extra capacity or shared nuisance features | Small module after pooling |
| Alternative second option | Shared auxiliary meniscus detector | Transfer should guide training without tying final outputs | Extra task weight and gradient interference | None after removing auxiliary head |
| Diagnostic when compute is free | Frozen representation, independent readout fits | Useful information exists but the shared readout underuses it | Weak labels and in-sample representation fitting | No extra backbone pass at deployment |
| Conditional on evidence | Moderate disease-combination weighting | Co-occurrence dependence contributes to failure | Noisy rare groups and changed training objective | None |
| Later | Disease-dependent attention contrast strength | Equal meniscus/OA contrast is overly restrictive | Continues the 1C training constraint | Small attention change |

Anatomical localization has more direct knee-specific evidence than several generic transfer methods. Tack and colleagues improved meniscus classification with an auxiliary bounding-box objective, using localization targets during training and avoiding a required localization preprocessing step at inference. Their study uses OAI sequences, six meniscal subregions and localization information not currently available in this pipeline. It is evidence about the value of anatomical supervision, not an isolated medial/lateral sharing result.[^21]

Anatomical crops, new localization supervision and a large label-query transformer remain outside the previously selected scope. They should not be silently added to a small sharing experiment.

### Completed candidate-1 comparison design

Use the archived reflection/sigma-40 ReXNet recipe as the reference. Preserve Steven v4, gold labels, fold membership, seed, pretrained initialization, image sampling, augmentation, optimizer and epoch-selection rule. Add only the meniscus weight-distance penalty. Preserve independent OA and all attention scorers. Architecture and loss identity belong in the run provenance even though inference computation is unchanged.

Before training, fix the diagnostic report: full-fold macro AUC, both meniscus AUCs, OA pairs, the other eight targets, BCE, exclusive-positive margin AUC, performance conditioned on co-occurring disease and the established manufacturer strata. Keep the original full-fold endpoint primary. Do not choose a favorable subgroup after seeing the result.

A convincing result would improve the intended family without concentrating the benefit in the same vendor or co-occurrence cases that motivated the concern. That pattern would make a controlled public comparison more informative; it would not predict a guaranteed public gain. A macro improvement driven by unrelated targets should not be presented as proof of successful meniscus transfer.

If weak coupling helps, compare hard meniscus sharing at 40 mm only if determining the required coupling strength affects the next decision. If it fails despite meaningful coupling, move to independent decisions over shared nonlinear features or a training-only auxiliary detector. Do not immediately add OA to a failed meniscus design.

To test whether a data intervention specifically fixes sharing, the eventual design is a two-by-two comparison: independent versus selected sharing method, under original versus revised training weighting or supervision. Running only the combined change cannot distinguish a sharing benefit from a label or sampling benefit. This factorial comparison is a later decision, not a request to launch four runs now.

### Stopping and interpretation rules

A successful architecture experiment demonstrates a useful way to transfer knowledge under the measured conditions. It does not prove the suspected hidden bias was removed. A failed experiment should distinguish ineffective coupling, inability to fit the source task, improvements restricted to a local proxy and external transfer failure.

If several forms of partial sharing improve only weak-label validation while deteriorating across the fixed conditional diagnostics or public comparisons, further architectural complexity has diminishing value. The next information would come from annotation alignment or better independent evaluation. If OA-specific supervision is the dominant issue, changing the sharing layer alone cannot manufacture the missing target information.

The practical objective is to retain the independent model's ability to make different compartment decisions while testing a limited additional route for transferable disease evidence. Shared weights, shared attention and shared training supervision are distinct experiments; their failures should remain distinct too.

## Sources

External research below supplies mechanisms and transferable design principles. No cited paper establishes a gain for this competition's twelve targets under Steven v4 supervision. Evidence was reviewed through 2026-09-14; the exact proposed implementations remain untested. Competition-specific annotation and public-fraction statements use the repository's prior official-source records; the corresponding Kaggle pages did not expose readable bodies on the current access.

[^1]: Project record, [Reflection sigma-40 result](../validation/results/2026-09-14-reflection-sigma-40-result.md), 2026-09-14; [comparison artifact](../../outputs/analysis/r/reflection-sigma40-20260914/comparison.json). Four-cell comparison, frozen inputs, checkpoint selection and public outcomes.
[^2]: Project records, [sharing comparison](../../outputs/analysis/s/sharing-generalization-20260914/analysis.json), 2026-09-14; [1A submission](../../outputs/submission/records/knee-1a-dual-gpu-20260914-v2.json), [1B submission](../../outputs/submission/records/knee-meniscus-oa-20260913.json), [1C submission](../../outputs/submission/records/knee-1c-batch4-20260914.json), [1C sigma-40 submission](../../outputs/submission/records/knee-sigma40-20260914.json), [reflection sigma-40 submission](../../outputs/submission/records/knee-reflection-sigma40-20260914.json). Selected-model results, not independent replications.
[^3]: Kaggle, [competition leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard) and [data description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data); prior verified details retained in [local comparison](../validation/results/2026-09-14-reflection-sigma-40-result.md) and the [generalization investigation](../history/sharing/generalization-public-subset-and-case-mixture.md). Public fraction and population limits.
[^4]: Project record, [sharing analysis](../../outputs/analysis/s/sharing-generalization-20260914/analysis.json), 2026-09-14. Per-target contributions and exclusive-positive ranking. [Narrative and limitations](../history/sharing/generalization-investigation.md).
[^5]: Project record, [compartment intervention summary](../../outputs/analysis/c/compartment-diagnosis-20260913/summary.json), 2026-09-13; [historical implementation specification](#historical-implementation-and-reproduction). The original plan was retired on 2026-09-14; its exact local archive is `outputs/archive/maintenance/disease-sharing-plan-retired-20260914-f1038525ba6f.md`, SHA-256 `f1038525ba6f351c836f86edd6bfe25e4e80464fc47d667cfe770da199661360`. Previously executed interventions; no new model execution.
[^6]: Project record, [blinded disagreement and acquisition audit](../../outputs/analysis/s/sharing-disagreement-20260914/summary.json), 2026-09-14. [Full audit narrative](../history/sharing/oa-and-meniscus-disagreement-audit.md). Report review is not radiologist image adjudication. Rubric source: Kaggle organizers, [annotation clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343), prior repository verification.
[^7]: Theodoros Evgeniou and Massimiliano Pontil, [Regularized Multi-Task Learning](https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/evgeniou%2Bpontil-2004.pdf), KDD 2004. Shared vector plus regularized task-specific deviations; kernel/SVM setting, not deep knee classification.
[^8]: Ishan Misra, Abhinav Shrivastava, Abhinav Gupta and Martial Hebert, [Cross-Stitch Networks for Multi-task Learning](https://publications.ri.cmu.edu/storage/publications/pub_files/2016/6/multi-task.pdf), CVPR 2016. Learned mixtures of task-specific and shared features.
[^9]: John R. Zech et al., [Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683), PLOS Medicine, 2018-11-06. Primary experiments on hospital-specific prevalence and external generalization.
[^10]: Shikun Liu, Edward Johns and Andrew J. Davison, [End-To-End Multi-Task Learning With Attention](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.html), CVPR 2019. MTAN combines shared features with task-specific attention; not the same factorization as 1C.
[^11]: Jiaqi Ma et al., [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/), KDD 2018. Task-dependent selection of shared experts; recommendation and other classification settings.
[^12]: Yunshu Du et al., [Adapting Auxiliary Losses Using Gradient Similarity](https://arxiv.org/abs/1812.02224), 2018 preprint. Adaptive influence of auxiliary objectives; guarantees concern the stated optimization setting, not hidden-domain accuracy.
[^13]: Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto and Percy Liang, [Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization](https://arxiv.org/abs/1911.08731), ICLR 2020; preprint 2019. Group robustness and the importance of regularization.
[^14]: Evan Zheran Liu et al., [Just Train Twice: Improving Group Robustness without Training Group Information](https://proceedings.mlr.press/v139/liu21f/liu21f.pdf), ICML 2021. Error-based reweighting; group-labeled validation used for selection.
[^15]: Luke Oakden-Rayner, Jared Dunnmon, Gustavo Carneiro and Christopher Ré, [Hidden Stratification Causes Clinically Meaningful Failures in Machine Learning for Medical Imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC7665161/), CHIL 2020. Subgroup failure, label quality and limits of aggregate evaluation.
[^16]: Han Zhao, Rémi Tachet des Combes, Kun Zhang and Geoffrey J. Gordon, [On Learning Invariant Representations for Domain Adaptation](https://proceedings.mlr.press/v97/zhao19a.html), ICML 2019. Conditional shift and limits of unconditional feature alignment.
[^17]: Zhuoyao Xie et al., [Development of a multi-task deep learning system for classification of nine common knee abnormalities on MRI: a large-scale, multicentre, stepwise validation study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12508577/), eClinicalMedicine, 2025-09-27. External knee evaluation; methods describe nine independently trained classification models.
[^18]: Tianhe Yu et al., [Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html), NeurIPS 2020. PCGrad and task-gradient interference.
[^19]: Wooseong Jeong and Kuk-Jin Yoon, [Quantifying Task Priority for Multi-Task Optimization](https://openaccess.thecvf.com/content/CVPR2024/html/Jeong_Quantifying_Task_Priority_for_Multi-Task_Optimization_CVPR_2024_paper.html), CVPR 2024. Limitations of gradient realignment and parameter-specific task priority.
[^20]: Polina Kirichenko, Pavel Izmailov and Andrew Gordon Wilson, [Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations](https://arxiv.org/pdf/2204.02937), ICLR 2023. Deep Feature Reweighting; importance of a suitable held-out reweighting set.
[^21]: Alexander Tack, Alexey Shestakov, David Lüdke and Stefan Zachow, [A Multi-Task Deep Learning Method for Detection of Meniscal Tears in MRI Data from the Osteoarthritis Initiative Database](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2021.747217/full), Frontiers in Bioengineering and Biotechnology, 2021-12-02. Meniscal localization as auxiliary supervision; different data, labels and anatomical annotations.
