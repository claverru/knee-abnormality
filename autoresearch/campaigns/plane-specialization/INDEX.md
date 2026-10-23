# Plane specialization

| Read when | Document |
|---|---|
| Checking the objective, forced seed order, fixed B0 plane-backbone recipe, plane-exclusive pooling, learned fusion, the radiology label-to-plane mapping, authorized scope and exclusions | [Campaign contract](contract.md) |
| Resolving the reference, seed order, config prefixes and editable scope | [Campaign configuration](campaign.yaml) |
| Checking seeds 1–5: random label-to-plane partition, its learned fusion, dense per-plane readouts, their learned fusion, per-plane heads | [Seeds 1–5](seed-ideas.yaml) |
| Checking seeds 6–9: radiology plane mapping, auxiliary per-plane readouts, winner-take-all routing, a full independent head per plane | [Seeds 6–9](seed-ideas-2.yaml) |
| Checking the user-forced nonlinear fusion idea: a per-label MLP with GELU on detached plane logits, residual on the learned fusion | [Nonlinear fusion idea](nonlinear-fusion-ideas.yaml) |
| Checking the user-forced MLP fusion without triplet dropout, which isolates the MLP from nonlinear-plane-logit-fusion | [No-dropout MLP fusion idea](nonlinear-fusion-no-dropout-ideas.yaml) |
| Checking the user-forced cross-label fusion: one MLP with GELU over all 36 detached plane logits, residual on the learned fusion | [Cross-label fusion idea](cross-label-fusion-ideas.yaml) |
| Checking the user-forced DSU inside each plane backbone and the per-plane OA detector | [DSU and OA-detector ideas](dsu-and-oa-detector-ideas.yaml) |
| Checking the user-forced per-plane auxiliaries (report-attribute heads, half-knee MIL, scanner confusion) and the shared stride-8 stem | [Plane auxiliary ideas](plane-auxiliary-ideas.yaml) |
| Checking the user-forced evidence-statistics fusion: detached attention shape, token scores, slice consistency and coverage per plane through a per-label residual MLP | [Evidence fusion idea](evidence-fusion-ideas.yaml) |
| Finding where the seeds come from, the single-plane evidence, the reference against the ReXNet control, the precedent table and closed directions | [Prior work](prior-work.md) |
| Recording ideas, their ancestry and completed outcomes | [Idea registry](ideas.yaml) |
| Checking the live queue | [Backlog](backlog.yaml) |
| Reviewing campaign decisions | [Findings](findings.md) |
| Reading the outcome: every result, the adoptions, the transfer to three ReXNet-200s (S73, S74, S77), the τ sweep and the closure | [Campaign result](../../../docs/validation/results/campaigns-and-queues/2026-09-28-plane-specialization-campaign.md) |
