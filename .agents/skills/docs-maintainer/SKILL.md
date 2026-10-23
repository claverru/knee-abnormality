---
name: docs-maintainer
description: Find, create, update, and audit repository documentation. Use when consulting project docs, recording durable knowledge, or cleaning up duplicated, obsolete, misleading, or poorly organized documentation.
---

# Documentation Maintainer

Keep project knowledge easy to find, accurate, and concise. Give each concept one canonical home. Distinguish current guidance from proposals and historical evidence.

## Choose the scope

- **Consult:** answer a question using relevant documentation. Keep this read-only; report material discrepancies without starting a cleanup.
- **Update:** create or change documentation for the requested work. Review the affected topics and their references.
- **Audit:** when asked to review or reorganize a documentation set, inspect that whole set in bounded batches.

A lookup or local edit does not require a global audit. Perform requested edits unless the user asks for a proposal or analysis only.

## Consult efficiently

1. Choose the entry point from the question: open a known path or section directly; use `rg --files` or targeted `rg` for a filename, symbol or specific concept; use the relevant folder index for a broad topic. Start at the root documentation index, normally `docs/INDEX.md`, when the destination is unclear.
2. Read only the relevant sections and enough surrounding context to understand their scope and qualifications. Inspect headings before loading a long body. Follow evidence links as needed; direct links and search may skip index levels.
3. Broaden searches across plausible documentation locations and relevant READMEs when necessary. Try alternate terms or an index if a search is inconclusive; an empty first search does not establish absence.
4. Check applicability: distinguish current instructions, proposals, superseded decisions, and results tied to particular versions. Cite the document or evidence supporting the answer; surface unresolved contradictions.

Do not preload the documentation tree. Expand context when the question requires it.

## Create and update

1. Locate the canonical document and check overlapping topics using the lookup rules above. Update the existing location unless the new topic answers a distinct question worth discovering independently.
2. Read the affected content before editing, and the whole document before a substantial rewrite or merge. Preserve unique constraints, exceptions, rationale, and unrelated working changes.
3. Lead with the answer or current procedure. Use descriptive headings and stable concept-based filenames. Add meaningful section headings and direct anchor links for recurring questions within a longer document; preserve existing anchors or repair their incoming links when headings change. Link to canonical explanations rather than repeating them. Merge overlapping purposes; split only at meaningful conceptual boundaries.
4. Separate verified facts, hypotheses, and plans. Support consequential claims with relevant code/tests, official sources, or reproducible evidence. Record versions or check dates for volatile external claims and review triggers where useful. A recent edit date does not prove verification.
5. Update docs when work changes a reusable procedure, constraint, decision, or conclusion. Avoid session transcripts and routine progress reports. For experimental conclusions, retain the configuration, code revision, dataset/split identity, seed where relevant, metrics, and artifact references needed to assess them.
6. Keep navigation hierarchical:
   - Each documentation folder has one `INDEX.md` (or the repository's existing equivalent) listing every document and subfolder directly inside it, each with a link and a one-line "read when". Use distinctive concepts, experiment terms and likely search vocabulary so readers can distinguish neighboring entries.
   - Index files contain navigation only. Keep scores, conclusions and procedures in the canonical documents. Link to specific sections where they answer recurring questions; useful cross-topic links may point outside the folder.
   - Update index entries when a document or folder is added, moved, removed, changes purpose, or needs a repaired or more useful navigation link. A change to a result or fact inside a document does not require an index edit when its purpose and links still fit. Update affected navigation in the same change; do not propagate child-level details through ancestor indexes.
   - Keep documents single-topic and small. Split a document that grows large (roughly 300+ lines) or accumulates independent entries into a folder with its own index, and repair links.
   - Use `README.md` for its conventional purpose, not as a folder index, unless the repository already does.
   - Follow an established repository convention unless asked to reorganize.

Match evidence to the claim: requirements describe obligations, code shows implementation, and experiments support measured outcomes. Make disagreements explicit instead of silently selecting a convenient source.

## Correct and prune

Before deleting or substantially replacing content, read it completely, find incoming references, and ensure removed information is recoverable from a recorded version or local snapshot. Tracking a file does not preserve its uncommitted changes. Preserve unique useful material and repair links in the same change.

- **Duplicate or obsolete guidance:** consolidate useful details, then remove redundant or unused material. Keep redirects or stubs only for a concrete compatibility need.
- **Wrong claims:** replace incorrect guidance in place using evidence and repair dependent conclusions. Mark unresolved uncertainty when verification is unavailable; do not invent a replacement.
- **Completed plans:** move durable findings and decisions to their canonical homes, then delete completed tasks and plans with no remaining work.
- **Historical context:** retain useful decision rationale with explicit supersession links, required migration guidance, and release records. Use version history for ordinary editorial history.
- **Experiments:** preserve useful negative results that prevent repeated work. Mark invalid measurements and withdraw conclusions that relied on them; retain enough context to explain the failure when useful.

Age, length, low readership, or mere existence alone never determines retention. Ask whether the information helps someone act, understand a constraint, reproduce evidence, or avoid a known mistake.

## Audit and finish

For an audit, inventory the requested documentation set, review each document and substantial section by topic, then reconcile cross-topic duplication, competing claims, and misplaced information. Choose keep, rewrite, move, merge, split, or delete based on reader needs. Optimize the final organization, not the smallest diff.

After edits, review the diff and affected navigation, links, and anchors; run existing documentation checks where available. Verify that indexes list all their documents and contain only navigation, and flag unlisted or oversized multi-topic documents. Structural checks cannot establish factual correctness. Complete the requested scope with canonical ownership, useful boundaries, and explicit unresolved uncertainties.

Report substantive changes and unresolved issues briefly. Do not claim to have audited content outside the reviewed scope.
