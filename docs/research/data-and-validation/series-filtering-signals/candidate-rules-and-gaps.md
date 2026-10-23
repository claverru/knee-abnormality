# Series filtering: candidate rules and gaps

Every rule keeps a series when its inputs are missing. None may remove the last series of a
study. Flags go to the visual census first; only confirmed rules become exclusions.

1. **Defensive per-file and per-series filters (H0/H1; expected 0 training hits).**
   - Keep the modal matrix, pixel spacing and orientation within a series.
   - Drop `LOCALIZER` image types and localizer descriptions.
   - Count hits on train and on the three example-test studies; any training hit is a finding.
2. **Side groups (H0).**
   - Compute each series' median physical center X.
   - When a study has at least two series at X > +40 mm and at least two at X < −40 mm, label it bilateral: 7 expected.
   - When exactly one series sits opposite at least three others, label it a stray: 1 expected. Candidates for exclusion.
   - Check that each group is a contiguous `SeriesNumber` block.
   - Report GEHC and near-origin studies separately; the sign convention is weak there.
3. **Coil anatomy (H1).**
   - Normalize `ReceiveCoilName` to anatomy words (KNEE, ANKLE, FOOT, WRIST, HAND, SHOULDER, HIP, SPINE, BODY, HEAD). Combined coils such as `KNEE_FOOT` count as knee.
   - Flag a minority group whose words exclude KNEE when the study majority includes KNEE: 5 series expected.
   - Whole-study non-knee coils are not flagged.
4. **Geometric outlier (H0).**
   - Flag series dropped by the 20 mm robust fit whose center projects outside their acquired range.
   - Count how many rows 2–3 already explain, and send the rest to the visual census.
5. **Label checks, flag only (H1).**
   - Fat-suppression evidence versus label, with the known-0 groups (T1 fat-suppressed, water-excited cartilage) excluded.
   - T1-like weighting (TR < 800 ms, not gradient echo) under label 1.
   - Post-contrast words or agent names.
   - Dominance margin below 0.1.
   - Expect a few dozen series; review them visually before any relabeling.
6. **Decision gates.**
   - Exclude a series only when two independent signals agree (e.g. rows 2 or 3 plus row 4) or the census confirms it visually.
   - Bilateral studies were left pending the [open question](findings-by-case.md#contralateral-knee).
   - Each rule's effect on validation studies and triplets was to be measured before use.

## Open questions and gaps

- Which knee a bilateral study's labels describe, and whether the one-sided `Laterality=L` tags are the host's disambiguation.
- Whether the hidden test contains localizers, reformats or other anatomy that training lacks (unknowable locally). The defensive rules exist for that case.
- Whether GE `SMART FAT` series without "Water" are water or fat images.
- The open-source tool survey was stopped early. What it found is in [open-source tool rules](external-evidence.md#open-source-tool-rules); GE conformance statements (the `*_GEMS` scan-option meanings, deep-learning reconstruction numbering) remain unread.
- Knee dataset curation reports give no exclusion counts: fastMRI, MRNet, SKM-TEA and KneeMRI describe manual checks without numbers. The lumbar organizers built per-sequence collages to catch wrong sequence labels but give no counts ([LumbarDISC](https://arxiv.org/abs/2506.09162)).
- No knee or MSK series classifier, bilateral-detection method or MRI knee-laterality model was found. Every reported header-classifier number comes from brain, body, prostate or radiograph data.
