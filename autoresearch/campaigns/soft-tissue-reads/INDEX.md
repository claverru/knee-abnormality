# Soft-tissue reads campaign

**Stopped by the user 2026-10-11 20:37 CEST after 34 runs; one adoption (the sagittal fluid detail view), at the plain recipe's level by its variants and by the ResNet-50 submission. Development moves to the ResNet-50.** The campaign exploits the soft-tissue masks better on the
line of its reference, `rn18-soft-tissue-added-read-bone-routes` (×4 0.928909, ×1 0.918680,
1,599 s): a seven-class map of the menisci and cartilage taught from the masks, and a read of
the third-stage cells routed by that map (B) added to the fourth-stage read (A) for the two
meniscus and three OA labels. Five forced seeds, in order: A trained alone with B as a
correction on a detached A and the wide regions; the OA cartilage and bone zone pooled apart; a
position inside the structure for the cell read; a cell reader of its own for PF OA; the map on
the 28 by 28 grid. Five more forced seeds were added on 2026-10-11. Then the researcher without end. Adoption by the runtime alone: strictly
above the incumbent at ×1 and ×4. Training time 0.75 to 1.25 of the reference's. Keeper: Claude,
with Codex as the provider.

| Read when | Document |
|---|---|
| Checking the objective, what is settled, the rules a seed must follow, what is out of bounds, cost and adoption | [Contract](contract.md) |
| Reading the masks, the reference's design, the four runs of the line, what the cell read sees that the fourth-stage read does not, the precedent table, what was dropped and the open leads | [Prior work](prior-work.md) |
| Reading the five forced seeds | [Seeds](seed-ideas.yaml) |
| Reading the five seeds added on 2026-10-11 (OA read on the second-stage cells, bone zone taught on the cell grid, cell read relative to the average open cell, own projection for the cell read, map overlap term per structure) and the measurements behind them | [Second seeds](seed-ideas-2.yaml) |
| Reading the follow-up seed forced after run 3 (the position for the meniscus reader only) | [Third seeds](seed-ideas-3.yaml) |
| Reading the four bundle seeds forced on 2026-10-11 at the user's request (combinations of runs 3, 4, 5, 6, 7, 9, 12 and 16 chosen by their per-label results) | [Bundle seeds](seed-ideas-4.yaml) |
| Reading the fifth bundle, forced after run 18 (the share-transition map loss with the area-normalised attention and the detail adapter) | [Fifth bundle](seed-ideas-5.yaml) |
| Reading the PF OA bundle forced after the audit of the OA labels (unpadded rim, zone on the cells, own reader, transition map loss) | [PF OA bundle](seed-ideas-6.yaml), removed from the queue the same day |
| Reading the bundle whose parts were chosen per label from the epoch curves and ×4 together (area-normalised attention, transition map loss, own projection, unpadded rim) | [Curve-chosen bundle](seed-ideas-7.yaml) |
| Reading the seeds of the wider scope, forced on 2026-10-11: the map taught on every study from the reference's pseudo-volumes, dense context slices, cross-plane map agreement, structure tokens in the fourth-stage bag, the map as a side task without the cell read | [Wider-scope seeds](seed-ideas-8.yaml) |
| Reading the sagittal fluid detail view seed | [Detail view](seed-ideas-9.yaml) |
| Reading the follow-up to the pseudo-volume run (pseudo-targets only on sagittal fluid and coronal T1/PD, the real masks at half of the map loss) | [Pseudo-volume follow-up](seed-ideas-10.yaml) |
| Reading the control the user asked for: the incumbent's recipe retrained unchanged by the harness | [Control](seed-ideas-11.yaml) |
| Reading the twelve-epoch look the user asked for (not for adoption) | [Twelve epochs](seed-ideas-12.yaml) |
| Reading the user's seed of a small label loss on each plane's read (weight 0.05) | [Plane read loss](seed-ideas-13.yaml) |
| Reading the three detail-view variants forced on 2026-10-11 (open to all twelve labels, three of five slices, feature-statistics draw) | [Detail-view variants](seed-ideas-14.yaml) |
| Reusing the wide regions of run 4 (the first seed takes its routing hunks only) | [Wide-regions change](wide-regions.diff) |
| Checking the reference, the budget, the editable files, the config permissions and the role instructions | [Campaign definition](campaign.yaml) |
| Reading each run's result against the incumbent, its diagnostics and the incidents | [Run notes](run-notes.md) |
| Changing the mean epoch curve of the earlier runs against the reference, which the keeper's watch subtracts to print each epoch against the family | [Family curve](family-curve.txt) |
| Reading the proposals, the queue and the decision summary once the campaign runs | [Ideas](ideas.yaml), [Backlog](backlog.yaml), [Findings](findings.md) |
