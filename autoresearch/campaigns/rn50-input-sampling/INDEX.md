# ResNet-50 input sampling campaign

**Stopped by the user 2026-10-12 09:50 CEST after 14 runs. Aligned contrast mixing (adopted by the user) is on main; 35 distinct slices per study (adopted by the runtime) is not: it scored 0.959 public (S146) against 0.960 for the reference (S143).** The campaign changes what a Refined-RadImageNet ResNet-50
is fed (distinct slices, fields, pixel size against the acquired resolution, alignment of the
input) on the line of its reference, `rn50-detail-view-teacher-025` (the sagittal fluid
detail-view recipe at 0.25 teacher share). Eleven forced seeds, the user's three first, then the
researcher without end. Adoption by the runtime alone: strictly above the incumbent at ×1 and
×4. Training time 0.75 to 1.25 of the reference's. Keeper: Claude, with Codex as the provider.

| Read when | Document |
|---|---|
| Checking the objective, what is settled, the principles a seed follows, how a result is read, what is out of bounds, cost and adoption | [Contract](contract.md) |
| Reading the reference, what reached the ResNet-50 from the ResNet-18 and what did not, the acquired-against-fed tables, the precedents, the token grid at 224 px and what was dropped | [Prior work](prior-work.md) |
| Reading the user's three seeds: aligned contrast mixing, the centred token grid by the last stage's sampling phase, the centred grid at 257 px | [Seeds](seed-ideas.yaml) |
| Reading the five sampling seeds: ten sagittal fluid slices, the axial fluid kneecap tile, one more slice on every type, the 106 mm sagittal context view, axial at 100 mm | [Second seeds](seed-ideas-2.yaml) |
| Reading the last three seeds: 35 distinct slices, the kneecap and pouch tile, 3D series as slabs | [Third seeds](seed-ideas-3.yaml) |
| Checking the reference, the budget, the editable files, the config permissions and the role instructions | [Campaign definition](campaign.yaml) |
| Reading each run's result against the incumbent, per label, with the incidents | [Run notes](run-notes.md) |
| Reading the two grid seeds added on 2026-10-11 by the user's order: both grids centred at 225 px, and the 224 px phase shift requeued, each with exact copies between cells and tokens | [Fourth seeds](seed-ideas-4.yaml) |
| Reusing the corrected copies between the third-stage cells and the fourth-stage tokens | [Exact copies at 257 px](exact-grid-transfers-257.diff), [exact copies at 224 px](exact-grid-transfers-224.diff) |
| Reading the user's seed of 2026-10-11 outside the sampling scope: the cosine schedule restarted twice within the ten epochs | [Fifth seeds](seed-ideas-5.yaml) |
