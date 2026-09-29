# Autoresearch

Topic-scoped, autonomous experimentation. The user starts one campaign, or the topic loop; a
detached supervisor then coordinates research, implementation, training and evaluation within the
campaign's fixed contract, and the loop moves to a new topic on its own when one stops paying. Every campaign uses the
[shared evaluation and adoption rule](protocol.md#shared-evaluation-and-adoption).

**Only local training and validation are authorized. No submissions, uploads, publication or
remote evaluation jobs.**

## Where to start

- [Index](INDEX.md): every document in this folder.
- [Runtime operation](runtime.md): start, inspect, stop and resume a campaign or the
  [topic loop](runtime.md#topic-loop) with either provider (Codex or Claude subscription CLI),
  before the first launch.
- [Topic rotation](topic-rotation.md): how the loop rotates and chains topics;
  [topics](topics/INDEX.md) for its queue and seed pool.
- [Protocol](protocol.md): evaluation, adoption, seeds, roles and ownership.
- [Campaigns](campaigns/INDEX.md): each topic's contract, ideas and findings.
- [Campaign setup](campaign-setup.md): add a topic through configuration, without runner edits.

Model and effort per role live in [agents.yaml](agents.yaml). Do not fall back to paid API usage
when subscription quota runs out.

## Current state

No campaign or topic loop is running; all campaigns are stopped. Pending runner work is in
[status](status.md).

## Ownership

Campaign agents write only inside `autoresearch/`; they may read and link project docs but not
edit them. Runtime state, sources, logs, predictions and checkpoints stay under the ignored
`outputs/autoresearch/<campaign-id>/`.
