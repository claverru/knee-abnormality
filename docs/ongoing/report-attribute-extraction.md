# Report attribute extraction with JevK5

**Status, 2026-09-25:** the [report-attribute heads](../modeling/report-attribute-heads.md) on
target set 3.1 are part of the recipe (user decision,
[result](../validation/results/2026-09-25-report-attribute-heads.md)). The targets come from the
v3 extraction below. Open work: a v4 question set (Baker's size first, then the weaknesses in the
[v3 analysis](#v3-full-run-analysis-2026-09-25)), and a separate target-repair experiment on
low-grade effusion, Baker's and cartilage findings.
**Reliability filter, reevaluated 2026-09-27:** the question-level reliability filter
(`reliability-filtered-attribute-questions` in the
[auxiliary-signals campaign](../../autoresearch/campaigns/auxiliary-signals/INDEX.md): drop questions
whose mean option-order agreement is below 0.80, plus their gated children, 7 of 35) scored +0.00104
locally, 80% [−0.0004, +0.0024], against `half-stack-control`, the campaign's only local near miss,
but **0.943 public (S72, bundle `f2a642bb`) against 0.947 for its paired control S68** (same fold and
seed). Kaggle shows S72 with a stale "S66 … retry" description. Not adopted; the cell-level
version lost locally (−0.00062). This entry replaces
the backlog big bet "Richer report supervision: OA attributes", widened from OA to all twelve
labels.

**v3 has two consumers (2026-09-28):** the report-attribute heads (target set 3.1, training only)
and the adopted labels `rebuild_state_v2`, where JevK5 v3 is one root family and one of the state
anchors ([label rebuild](../public-labels/label-rebuild.md#reproducing-a2)). The labels pin
`outputs/labels/jevk5-attributes-v3/attributes.parquet` by SHA-256 and the folder is read-only, so
v4 must never replace v3 in place. Order for v4 (user decision): first a label-free comparison of
v4 against v3 answers on v3's known weak spots (hedged findings, 15 of JevK5's 18 wrong cells in the
source audit, and Baker's size); then v4 targets in the heads as a normal candidate against
`rebuild-state-v2-control`, which keeps the evaluation targets fixed; only then, if v4 clearly fixes
the hedged-finding misses, a new label version `rebuild_state_v3` on the same split, judged with a
new control and a public comparison because it moves the evaluation targets.

## Goal

Extract structured facts from every training report and use them to supervise the image model
during training only. Reports are never model inputs, and the evaluation targets stay unchanged.

The attributes can serve four purposes:

- **Targets:** better soft targets for the twelve labels, using grade, hedging and whether the
  report mentions the finding at all.
- **Location:** where each finding sits (meniscus horn, bone and compartment, patellofemoral
  surface) as spatial supervision, extending the half-knee idea beyond the menisci.
- **Extra findings:** auxiliary heads for image-visible findings outside the twelve labels (PCL,
  lateral ligaments, extensor tendons, bursae and fat pad, cysts, patellar instability,
  osteochondral lesions, loose bodies).
- **Reliability:** per-cell trust from hedging and report silence.

**Premise, measured 2026-09-22:** severity alone mostly repeats what the soft targets already
encode (Spearman 0.66–0.79 among positives), and the worst-ranked cells are low-grade positives,
which is label noise ([report attributes](../public-labels/report-attributes.md)). No public table
holds location, extra findings or a "not mentioned" state separate from "normal". PF OA predictions
barely follow report severity (Spearman 0.09–0.24), so severity may still help there.

## Extractor

JevK5 (`alibiserikbay/JevK5`, revision `05b7ad94`) is Qwen3.5-4B with a distilled LoRA. It
answers one lettered multiple-choice question per forward pass: a softmax over the answer letters'
logits, divided by a stored temperature (1.532). It is a general-purpose decision model trained on
English. On raw multilingual reports its 16-question predecessor reached 0.877 gold macro AUC
(`z_labels/jevk5_kaggle`). Its authors measured that averaging two option orders lowers
calibration error at a small accuracy cost; the attributes are used as probabilities, so every
question is read in both orders and averaged.

## Question set design

The set is [`configs/labels/jevk5-report-attributes-v3.json`](../../configs/labels/jevk5-report-attributes-v3.json):
40 questions.

- **Facts, not targets.** The output is each option's probability. The mapping to targets, heads
  or masks happens downstream and can change without re-running the model.
- **"Not mentioned" is separate from "normal" only where the model can tell them apart:** the
  status questions for ACL, MCL, both menisci, the three cartilage depths and effusion. Silence is
  an unobserved label, never an absent one
  ([criteria](../metadata/official-label-criteria.md#labels-versus-report-text-discussion-733826-host-reply-2026-08-08)).
- **Hedging is an option.** "Possible" means the report itself is uncertain; hedged wording that
  still names a finding or grade ("in favor of", "consistent with") takes the finding.
- **Options sit at the official thresholds** ([criteria](../metadata/official-label-criteria.md)):
  ACL above half the fibres, MCL high-grade and acute, meniscal signal reaching the surface, OA
  depth above half the thickness and extent of about 1 cm or more, effusion and Baker's moderate
  or larger.
- **The model sees option descriptions only, never the keys.**
- **Exclusions live in options, not in the criterion.** A sentence in the criterion saying what
  does not count was mostly ignored; an option that names the look-alike absorbs it (another cyst,
  cartilage-only, bone-only, damage only in other compartments, other soft-tissue edema,
  suprapatellar fluid, non-traumatic edema).
- **Locations are coarse.** One question per meniscus for the tear site, and one contusion
  location question over compartments. Per-bone marrow questions were dropped: the model copied
  one bone's finding to the others in both v1 and v2.
- **Report text is normalised** before encoding: look-alike symbols (U+2206, U+00B5) become Greek
  letters, and German VKB and HKB and the abbreviation FCL get their English names appended.
- **Study context:** knee side and the stated indication.

### Downstream consistency rules

The consumer applies these instead of trusting the dependent answers. On the v2 check they gained
about 30 correct answers per question:

- a meniscus whose status is not mentioned, normal or degeneration has no tear location and no
  tear type;
- a meniscus whose status is a tear but whose location is "no tear" has an unstated location;
- MCL timing is "no injury" when MCL status is not mentioned or intact.

## v1 audit, 2026-09-25

v1 had 50 questions. The first 400 reports (two shards) were judged by hand against the report text,
one reviewer per question group, all 400 reports for most groups. The judgments and the v1 answers
are in `outputs/labels/jevk5-attributes-v1-audit/`. Findings that shaped v2:

- **Option keys leaked into answers.** The model read `key: description`. `not_stated` was taken
  as "the report says nothing" (86 `mcl_timing` and 101 medial tear-type errors), and
  `effusion_debris_only` absorbed plain effusions (6 of 50 right).
- **Findings were copied to other locations,** the largest error source. Marrow region positives
  were 14–53% right, with 178 errors from one bone's finding repeated in others; meniscal horn
  positives were 25–75% right; patellar facet lesions leaked into the femorotibial compartments.
  The two option orders agree on these errors, so order disagreement does not flag them.
- **Missing options for confusers.** Meniscal and ganglion cysts became Baker's cysts; chondral
  lesions and bruises became osteochondral lesions (90 false positives); bruises, necrosis and OA
  edema became fractures (fracture positives 42% right); cartilage lesions became OA grades.
- **General "normal" statements were ignored** despite the preamble, mostly on the PCL, lateral
  ligaments, extensor tendons and bones.
- **Language:** German VKB/HKB, Turkish "SOL" (left, 18 of 18 missed), and Greek reports written
  with U+2206 for Δ. The runner now replaces the look-alike symbols.
- **Reliable questions:** ACL status (87%), MCL status (95%), meniscus status (85% medial, 92%
  lateral), effusion (93%), patellar instability, cartilage depth grading scales.
- **Order disagreement flags doubtful answers** on status questions (ACL right in 6 of 31
  disagreements) but not on the systematic location errors.
- **Dropped:** limited examination (no positives in 400), prior surgery (all 7 positives wrong),
  fibula marrow (4 of 29 positives right, no label uses it).

## v2 audit, 2026-09-25

The same reviewers scored v2 on the same 400 reports against their v1 judgments. The v2 answers
and scoring scripts are in `outputs/labels/jevk5-attributes-v2-audit/`.

- **Better almost everywhere.** Order agreement rose from 0.81 to 0.88. Correct answers out of 400,
  v1 to v2: ACL 347 to 371, PCL 330 to 391, extensor tendons 349 to 393, side 374 to 395, context
  296 to 383, medial meniscus status 338 to 362, medial tear location 288 to 336, extrusion 385 to
  397, meniscal cyst 346 to 378, fracture 356 to 398, loose bodies 384 to 398, synovitis 364 to
  396. Cartilage extent and overall OA gained 12 to 27 points of accuracy on 200 reports.
- **Hiding the keys and adding look-alike options did the work:** synovitis positive precision
  went from 0.60 to 0.96, extrusion false positives from 10 to 0, osteochondral false positives
  from 50 to 15 on 200 reports.
- **Worse:** per-bone marrow (copying 198 to 190 errors, invented "degenerative" 213 to 257; four
  of five regions lost accuracy), lateral meniscus status (367 to 362, medial findings leaking in),
  patellofemoral surface, and Hoffa (63 false positives unchanged).
- **Over-broad preamble rules backfired:** "a negated finding is absent" turned "degenerated, no
  rupture" into intact ACL and MCL, and "bones normal covers every bone" was read as normal
  cartilage.
- **The abbreviation line in the preamble did not work;** VKB, HKB and FCL were still missed.
- **Cartilage depth became more order-sensitive** (medial disagreements 81 to 135, lateral 75 to
  150) while its accuracy held.

## v3 full-run analysis, 2026-09-25

Label-free checks over all 4,407 v3 answers (scripts in the session scratchpad; answers in
`outputs/labels/jevk5-attributes-v3/`). Inputs for a v4:

- **Dead options:** bone-on-bone (1–3 answers per compartment), context "other" (2), synovitis
  "possible" (9), ACL graft (4), and the osteophyte "only elsewhere" option, which never exceeds
  0.43.
- **Position bias:** the first option gains up to +0.27 in the forward order (cartilage and OA
  "not mentioned") and the last option up to +0.26 (osteophyte "unspecified"). The biggest
  forward/reverse flips are between options with the same disease meaning: overall OA "not
  mentioned" vs "none" (1,369 reports, 1,333 without any arthrosis word) and cartilage depth "not
  mentioned" vs "normal" (599–954 per compartment).
- **Osteophytes are mostly noise:** 3,778 reports never mention osteophytes yet carry 0.32 mean
  osteophyte probability (medial); 1,339 of 1,354 medial forward/reverse flips have no osteophyte
  word. Geometric instead of arithmetic combination of the two orders changes 1% of answers and
  does not fix it.
- **Children answer without their parent:** medial tear location says tear on 642 reports whose
  status says no tear, contusion location on 969, PF surface damage on 1,267 reports whose PF depth
  says no loss, cartilage extent on 291–519. The target gates cover the first two; PF surface and
  extent are ungated.
- **Language:** order agreement is lowest in Croatian, Greek and Turkish (0.78–0.80) and highest in
  English (0.87). A quarter of Spanish reports are short positive-only summaries (median 201
  characters), so their "not mentioned" answers are correct: 255 of 260 Spanish ACL "not mentioned"
  reports never name the cruciates.
- **Training targets sit below the official thresholds:** in `combined_v1`, 95% of reports the
  extractor reads as a small effusion are Effusion-positive (1,619 reports), 61% of small Baker's
  cysts are positive, and 67–75% of reports with softening or low-grade medial cartilage loss are
  Medial OA positive. The official criteria grade all of these negative.
- **Gold check (58 studies, 2026-09-25):** partly confirmed. Gold is positive for 15 of 28
  reported-small effusions (54%, against 95% in the targets and 0% under the official text), 3 of 9
  small Baker's cysts (33%, targets 61%), 1 of 5 low-grade or softened medial cartilage (targets
  67–75%) and 3 of 11 low-grade or softened PF cartilage. Scoring the same v3 answers against gold,
  a rule at the official threshold beats "any finding" for PF OA (AUC 0.947 against 0.842) and
  Effusion (0.755 against 0.614) but not for Baker's (0.832 against 0.871), Medial OA (0.848 against
  0.913) or Lateral OA (0.832 against 0.841). Low grades are over-called in the targets, but a hard
  official cut is too strict for effusion; small samples throughout.

## Procedure

1. **Check on the first two shards:** run the new version, stop after two shards (the same 400
   reports as the audits), and re-judge them. Resume the same run if the check passes; a changed
   question set restarts it.
2. **Gold confirmation** at the end only.
3. **Training-only use:** every question except knee side becomes an auxiliary head with one
   fixed coefficient, trained on the full option distribution with children gated on their parent
   ([design](../modeling/report-attribute-heads.md)). The twelve disease targets, gold overrides
   and disease loss stay unchanged, and the heads are dropped at inference.
   - Compare against a fresh current-recipe control with the same split, evaluation targets,
     seed, image workload and training budget. Attribute accuracy is diagnostic, not the adoption
     metric.
   - **Known weak targets in v3** (spot check of the audited cases, 2026-09-25): Hoffa and Baker's
     size still invent findings, the patellar-facet leak into the femorotibial cartilage questions
     remains, osteochondral still counts bone lesions, and unexplained edema is now called a
     contusion. Against the training targets, Baker's size agrees (Spearman 0.73) and its errors
     are mostly small cysts; Hoffa is unrelated to any target. Contusion location conflicted
     with the Contusion target until gated on contusion status.

## Running

The runner, [`score_jevk5_report_attributes`](../../scripts/adhoc/score_jevk5_report_attributes.py),
reproduces JevK5's prompt and readout with
`transformers==5.17.0` and `flash-linear-attention==0.5.0` added through uv's `--with` overlay,
with torch pinned to the project's cu128 build. Outputs go to `outputs/labels/jevk5-attributes-v3/`.
A shard of 200 reports takes about 12 minutes alone on the 3090 (v1 took 10.6; v2 and v3 prompts
are about 13% longer), so the full run is about 4.5 hours and the two-shard check about 25 minutes.

The target-table step (`scripts.tools.prepare_report_attribute_targets`) was retired on 2026-10-14 with the report-attribute heads; it is at `7770ce9:scripts/tools/prepare_report_attribute_targets.py`.

```bash
rtk proxy uv run --locked --with "transformers==5.17.0" --with "flash-linear-attention==0.5.0" --with "torch==2.10.0+cu128" --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match python -m scripts.adhoc.score_jevk5_report_attributes --output-dir outputs/labels/jevk5-attributes-v3
rtk proxy uv run --locked python -m scripts.adhoc.analyze_jevk5_report_attributes answers
rtk proxy uv run --locked python -m scripts.adhoc.analyze_jevk5_report_attributes gold
```

## Reproducing the targets

- **Extraction inputs are pinned:** `run.json` records the JevK5 revision (`05b7ad94`), the
  question-set and report SHA-256s and the seed; `summary.json` records torch 2.10.0+cu128,
  transformers 5.17.0 and the RTX 3090; `source/` holds the exact runner. The committed runner
  differs from that snapshot only in its CLI descriptions. The model weights (7.9 GB) are in the
  local Hugging Face cache.
- **Extraction is not proven bit-identical:** it runs in bf16 on the GPU and has never been
  repeated, so a rerun may differ in low-order digits. Measuring that costs one shard, about 12
  minutes of GPU.
- **Preparation is deterministic:** rebuilding target set 3.1 from the saved answers reproduced
  the pinned table's SHA-256 (`5db55757…`) exactly.
- **The answers are the costly artifact:** `outputs/labels/jevk5-attributes-v3/attributes.parquet`
  (23 MB) is not in version control and has no off-machine copy yet.
