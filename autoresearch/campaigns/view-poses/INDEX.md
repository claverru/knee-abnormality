# View-poses campaign

**Ran 2026-10-08 to 2026-10-09 with Codex; stopped by the user after seven runs and three adoptions; the incumbent is `sagittal-fluid-canvas-fine` (×4 0.928121, ×1 0.919392).** The campaign runs smart crops (shift, turn and
scale per knee) in every view and tries different canvases for the six plane-contrast views. Its
reference is `rn18-meniscus-map`, the 224 px recipe with the report-attribute loss off and a
meniscus head. Eleven forced seeds, in order: the meniscus head on the 14 by 14 grid and the
knee's centre taught from the masks; the 368 px canvases with scale and the axial pose, fields
unchanged; a crop centre for each plane-contrast and two seeds that each move one view's
centre; four canvas combinations; the meniscus labels read by meniscus share. Then the
researcher without end. Adoption by the runtime: at least the incumbent at both ×1 and ×4; by the keeper,
the meniscus head whatever its macro result and the canvas seed unless clearly worse. Training
time 0.5 to 1.4 of the reference's.

| Read when | Document |
|---|---|
| Checking the objective, the stages, what is settled and what is still open, the rules a seed must follow and adoption | [Contract](contract.md) |
| Reading the measurements the campaign rests on: knee size, where findings are seen per view, errors against crop coverage, the token grid, the soft-tissue segmenter, the 3D series and the canvas sizes | [Prior work](prior-work.md) |
| Reading forced seeds 1 to 5: the meniscus head on 14 by 14, the taught joint centre, the 368 px canvases with scale and the axial pose, a crop centre per plane-contrast, the T1/PD sagittal crop on the notch | [Seeds](seed-ideas.yaml) |
| Reading forced seeds 6 to 10: the T1/PD joint crops on the menisci, and the four canvas combinations | [Seeds 2](seed-ideas-2.yaml) |
| Reading forced seed 11: the meniscus labels read by meniscus share | [Seeds 3](seed-ideas-3.yaml) |
| Reading the two seeds re-submitted with their measured thresholds: the T1/PD joint crops on the menisci and the meniscus labels read by meniscus share | [Seeds 4](seed-ideas-4.yaml), [Seeds 5](seed-ideas-5.yaml), [calibration values](meniscus-calibration.json) |
| Reading the plan behind the seeds: the base, the canvas combinations as trades and the ones dropped, the checks still to run and the seeds planned after | [Seed outline](seed-outline.md) |
| Checking the reference, the budget, the editable files, the config permissions and the role instructions | [Campaign definition](campaign.yaml) |
| Reading each run's result against the incumbent, its diagnostics and the incidents | [Run notes](run-notes.md) |
| Reading the proposals, the queue and the decision summary once the campaign runs | [Ideas](ideas.yaml), [Backlog](backlog.yaml), [Findings](findings.md) |
