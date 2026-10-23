# Auxiliary signals

| Read when | Document |
|---|---|
| Checking the objective, forced seed order, fixed recipe, authorized scope (attribute targets, loss, heads, metadata heads, acquisition inputs) and exclusions | [Campaign contract](contract.md) |
| Resolving the reference, seed order, config prefixes and editable scope | [Campaign configuration](campaign.yaml) |
| Checking seeds 1–5: attribute coefficient 0.2, group-balanced attribute loss, reliability-filtered questions, removal of report-completeness options, removal of hedged options | [Seeds 1–5](seed-ideas.yaml) |
| Checking seed 6: scanner-group adversarial head (gradient reversal) | [Seed 6](seed-ideas-2.yaml) |
| Checking the user-ordered retry of seed 6 with the head trained at full strength and 0.1 only on the reversed gradient | [Adversarial retry](adversarial-retry-ideas.yaml) |
| Checking the user-ordered scanner confusion idea: detached scanner head and a bounded KL confusion loss replacing gradient reversal | [Confusion idea](confusion-ideas.yaml) |
| Checking the user-ordered confusion follow-ups: stronger detached head with warm-up, and a head on backbone features before normalization | [Confusion follow-ups](confusion-followup-ideas.yaml) |
| Checking the user-ordered bias ideas: training-only per-site logit bias, scanner-model confusion target, report-completeness bias | [Bias ideas](bias-ideas.yaml) |
| Checking the user-ordered per-scanner token harmonization (ComBat-style location and scale, applied at inference) | [Harmonization idea](harmonization-ideas.yaml) |
| Checking the user-ordered completeness bias rebuilt on v3 not-mentioned mass after the report-length version was blocked | [Completeness from v3](completeness-v3-ideas.yaml) |
| Checking the user-ordered site-bias follow-ups: per-model training-only bias, and the site bias combined with scanner-model confusion | [Site-bias follow-ups](site-bias-followup-ideas.yaml) |
| Finding where the seeds come from, the seed-noise yardstick, the precedent table (attribute heads, header stacker), the v3 answer weaknesses, site priors and closed directions | [Prior work](prior-work.md) |
| Recording ideas, their ancestry and completed outcomes | [Idea registry](ideas.yaml) |
| Checking the live queue | [Backlog](backlog.yaml) |
| Reviewing campaign decisions | [Findings](findings.md) |
| Reading the outcome: every result, the adoption, the public scores S67–S72 and the findings | [Campaign result](../../../docs/validation/results/campaigns-and-queues/2026-09-27-auxiliary-signals-campaign.md) |
