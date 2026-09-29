# Preserve and normalize MRI signal before augmentation

Researcher topic `mri-signal-preservation`.

**Stopped by the user on 2026-09-29 after 8 decided runs, none adopted.** Five runs exceeded the
1,627 s training ceiling: per-image intensity work in GPU preparation added 17–115% per epoch
against about 10% headroom. That led to the epoch-2 projection stop and the rule against cutting
epochs to fit ([developer brief](../../roles/developer.md)). Two completed runs were recorded
`invalid` by a comparison bug on nested identity keys, since fixed. Rescored, they are
`sig-magnitude-power-debiasing` −0.0008 (95% [−0.0022, +0.0007]) and the 4-epoch
`sig-foreground-quantile-bounds` −0.0137. The one clean comparison,
`sig-black-anchored-foreground-upper`, was a reject at +0.0002 (80% [−0.0007, +0.0011]).

| Read when | Document |
|---|---|
| Checking the live queue | [backlog.yaml](backlog.yaml) |
| Resolving the reference, chain, backlog settings and editable scope | [campaign.yaml](campaign.yaml) |
| Checking the objective, exclusions, permitted mechanisms, cited weaknesses and starting point | [contract.md](contract.md) |
| Checking the researcher's complete draft: cited weaknesses, literature, families and seed metadata | [draft.yaml](draft.yaml) |
| Reviewing campaign decisions | [findings.md](findings.md) |
| Recording ideas, their ancestry and completed outcomes | [ideas.yaml](ideas.yaml) |
| Checking precedents, closed directions and each seed's closest precedent | [prior-work.md](prior-work.md) |
| Checking seed ideas imported at startup | [seed-ideas.yaml](seed-ideas.yaml) |
| Checking the topic definition as taken from its researcher source | [topic.yaml](topic.yaml) |
