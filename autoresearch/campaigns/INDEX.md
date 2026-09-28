# Campaigns

| Read when | Document |
|---|---|
| Recovering the stopped plane-specialization campaign (one EfficientNet-B0 per plane: label-to-plane partitions and routing, dense per-plane readouts, linear, MLP, joint and evidence fusion, per-plane heads, radiology mapping, per-plane OA detector, attribute heads, half-knee MIL and scanner confusion, DSU per plane; dense readouts accepted, learned fusion adopted, the line closed without adoption into the recipe) | [Plane specialization](plane-specialization/INDEX.md) |
| Recovering the stopped auxiliary-signals campaign (JevK5 attribute-target variants, scanner gradient reversal, the bounded scanner confusion head, per-site and per-model logit biases, forced ideas added on the fly; confusion head adopted, 18 rejected, 2 invalid) | [Auxiliary signals](auxiliary-signals/INDEX.md) |
| Recovering the closed augmentation campaign (13 user-ordered seeds, reruns and researcher refills: GIN, k-space resolution, sharpening, image/token/pooled MixUp, late affine shutdown, bias field, channel reversal, bilateral smoothing, intra-study contrast mixing, DSU and follow-ups, stochastic depth; smoothing and DSU adopted, the half stack adopted on main) | [Augmentation](augmentation/INDEX.md) |
| Recovering the stopped near-misses campaign (15 user-ordered seeds, then researcher ideas: meniscus readouts, early-learning regularization, 3D position, PF OA bias and three-compartment detector, token dropout, plane readouts, half-knee refinements; token dropout adopted, 18 rejected) | [Near misses](near-misses/INDEX.md) |
| Recovering the stopped knee-geometry campaign (study centre, bilateral studies, midline and knee frame, meniscus and OA compartment readouts; half-knee MIL adopted, 16 rejected, 1 invalid) | [Knee geometry](knee-geometry/INDEX.md) |
| Recovering the stopped noisy-labels campaign (prediction-memory target correction and down-weighting, early-learning regularization, class-balance schedule; nothing adopted) | [Noisy labels](noisy-labels/INDEX.md) |
| Recovering the retired cleaned-source oblique campaign (ACL gate, coverage split, frame depth and training-only views rejected; group embedding compared by hand and closed with the oblique line) | [Oblique (cleaned source)](oblique-clean/INDEX.md) |
| Recovering the retired x4 oblique campaign on the source before series cleaning (view bias, mixture pooling and combined identity rejected; ACL gate trained but uncompared) | [Oblique (x4 predictions)](oblique-x4/INDEX.md) |
| Recovering the retired first oblique campaign scored on x1 predictions (embedding, view bias and residual readout results) | [Oblique](oblique/INDEX.md) |
| Recovering the retired sampling campaign on the shared evaluation (slice, triplet, series, batch-composition, crop and feedback-driven sampling; six rejected experiments) | [Sampling](sampling/INDEX.md) |
| Recovering the historical triplet-selection campaign on `all_5fold_v3` that sampling succeeds | [Triplet sampling](triplet-sampling/INDEX.md) |
| Recovering the historical shared-loss campaign on `all_5fold_v3` | [Shared losses](losses/INDEX.md) |
| Recovering the historical medial/lateral disease-transfer campaign on `all_5fold_v3` | [Medial and lateral disease transfer](medial-lateral-transfer/INDEX.md) |
