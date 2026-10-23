# Public label audit

| Read when | Document |
|---|---|
| Finding which label tables existed (Steven v4, combined, rebuild_state v1/v2/v3, the v3.1/v3.2 tables not adopted, teacher-mixed silver targets) and which ones the final models trained on | [Label tables](label-tables.md) |
| Reviewing the first audit of label sources, coverage and audit reproduction | [Public report-label audit](public-label-audit-overview.md) |
| Comparing label agreement on reference studies | [Reference agreement](reference-agreement.md) |
| Interpreting source uncertainty encodings and report-state supervision evidence | [What uncertainty and confidence actually encode](uncertainty-and-confidence-encoding.md) |
| Checking source consistency and label-selection rationale | [Source consistency and first-baseline label choice](source-consistency-and-first-baseline-choice.md) |
| Checking `combined_v1`, the labels of 2026-09-21 to 09-28: per-target sources, consensus method, selection rule and reproduction | [Combined labels](combined-labels.md) |
| Reading a dated public-label sweep, comparison, rebuild audit or v3 label review | [Dated public-label audits](audits/INDEX.md) |
| Understanding how `rebuild_state_v2` was built: lineage-level public report sources, report states, anchored Dawid–Skene, gold semantic calibration, source audit, the A2 fix, and how to reproduce the tables and split | [Label rebuild](label-rebuild/INDEX.md) |
| Understanding why training labels (report-derived silver) and test labels (image-derived, like the 58 gold studies) differ: over-called low grades, silence as negative, site conventions, local-versus-public sign flips and what to try | [Gold versus silver labels](gold-versus-silver.md) |
| Using severity, hedging or report-state attributes from public tables (Ramit, Zaid, Pilkwang): agreement, redundancy with targets, overlabelled low-grade positives on gold | [Report attributes in public label tables](report-attributes.md) |
| Reproducing or auditing the JevK5 report-attribute extraction: the 40-question v3 set and its design rules (facts not targets, not-mentioned vs normal, options at official thresholds), the v1/v2 audits, v3 full-run analysis and gold check, the rejected reliability filter (S72), runner and pinned inputs | [JevK5 report-attribute extraction](jevk5-report-attribute-extraction.md) |
