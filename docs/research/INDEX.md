# Research and evidence sources

| Read when | Document |
|---|---|
| Getting the review's main recommendations, which competitions transfer and how they change the research direction | [Research overview](research-overview.md) |
| Comparing public notebook implementations and discussion findings: reported successes, failures, uncertainties, report-derived supervision, and the 2026-09-18 sweep of slice density, board standings and single-model scores | [Public discussion and notebook findings](public-research.md) |
| Judging the review's date, coverage and evidence types before relying on a claim | [Evidence and coverage](evidence-and-coverage.md) |
| Checking where project evidence came from: unverified working notes, public sources, label audits, research reviews, environment and metadata audits | [Provenance](provenance.md) |
| Choosing input pixel spacing, field of view, crop placement and slice density from acquisition, image, crop-run, internal and external evidence | [Input resolution](input-resolution.md) |
| Revisiting learned crop or zoom selection after the dropped investigation: literature limits, RSNA winners' supervised localize-then-crop recipes, global-local fusion, overengineering in the scout design and ranked alternatives | [Learned crop selection](learned-crop-selection.md) |
| Deciding whether and how position can improve medial/lateral discrimination: evidence, model blockers, encodings, GNNs, unconventional ideas and diagnostics | [Position information](position-encoding.md) |
| Using left/right knee identity to interpret spatial features and comparing side embeddings, coordinate transformations and anatomical modeling | [Knee side and anatomical location](knee-side-and-location.md) |
| Choosing partial medial/lateral sharing after the local/public reversal, including independent readouts and experiments to distinguish bias from constraints | [Medial and lateral disease transfer](medial-lateral-transfer.md) |
| Choosing training objectives for mean per-label ROC-AUC: BCE weighting, study sampling, batch composition and adaptive allocation, plus the earlier loss-only ranking and its evidence | [Training objectives, weighting and sampling for mean per-label ROC-AUC](auc_loss_approaches.md) |
| Choosing optimizer experiments: AdamW learning-rate and beta2 tuning, EMA, SAM, AdamP and AUC-loss optimizers, with a proposed run order | [Optimizer research](optimizer-research.md) |
| Comparing the closest RSNA MRI tasks and their successful and unsuccessful approaches | [RSNA MRI competitions](rsna-mri.md) |
| Borrowing sequence modeling, crop supervision, or feature aggregation from volumetric CT competitions | [Transfer from RSNA CT](rsna-ct.md) |
| Understanding knee MRI classification, MRI domain shift, segmentation, and reconstruction challenges | [Other MRI benchmarks](mri-benchmarks.md) |
| Judging which backbone families have held up on MRI, for the main backbone or a small scout: this dataset, RSNA lumbar localizers and classifiers, papers and a verdict per family | [Backbone MRI evidence](backbone-mri-evidence.md) |
| Considering genuine 3D CNNs (timm_3d or MONAI ResNet/EfficientNet with inflated or MedicalNet weights): volume construction, axis conventions, study fusion and a gated, unexecuted experiment plan | [Volumetric CNNs](volumetric-cnns.md) |
| Training a pretrained timm EfficientNet: matching checkpoint preprocessing, BatchNorm statistics policy, the spatial-mapping blocker and an unexecuted learning-rate × BatchNorm plan | [EfficientNet training](efficientnet-training.md) |
| Revisiting a scout backbone if a scout design returns: the historical timm survey with learned-zoom size and compute deltas, adaptation category and MRI evidence | [Scout backbone candidates](scout-backbones.md) |
| Filtering series that should not reach the classifier (other anatomy, contralateral knee, localizers, reformats, post-contrast) or checking plane and contrast labels: ranked header, geometry, image and model signals with cost, tag coverage, reported reliability and candidate rules | [Series-filtering signals](series-filtering-signals.md) |
| Choosing external knee data or evaluating pretrained model ideas | [Pretraining and external data](pretraining-and-external-data.md) |
| Finding inspectable implementations and avoiding misleading notebook defaults | [Public code and notebooks](code-and-notebooks.md) |
| Recovering exact reviewed code versions | [Code source manifest](source-manifest.json) |
| Recovering the public notebook snapshots inspected by the public-source investigation | [Notebook source manifest](public-notebook-sources.json) |
