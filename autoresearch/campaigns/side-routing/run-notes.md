# Side-routing campaign: run notes

Keeper's notes: repairs, adoption decisions with their run report, and seeds added during the
run. Clock times are CEST.

## Summary of the first night (2026-10-06 07:40, 20 runs)

**State:** running, Codex; incumbent `separate-oa-frame-with-knee-level-term` (×4 0.924485,
adopted 04:50); 20 runs decided, 1 adopted; GPU idle about 50 minutes in eight hours (15 at the
start, 12 at the adoption, two per guarded decision until 05:00, none since).

**On the plain macro nothing moved.** Seventeen of the twenty runs are within ±0.0005 of their
incumbent, which is the run-to-run spread on this fold; the adopted run is +0.00001. Three
runs are clearly below (−0.0007 to −0.0009).

**What repeated, and so is learned:**

| Finding | Runs |
|---|---|
| Routing PF OA from front-and-up raises its compartment localisation by 0.011 to 0.018 | 5 |
| A knee-level term is learned as a contrast (negative strength) and raises meniscus localisation by 0.002 to 0.006 | 6 |
| Given their own frames, meniscus and OA turn about ±6° in opposite, anatomical directions | 2, and 1 control |
| Better localisation has not raised the plain AUC of the routed targets | every run with it |
| Position codes on the own-channel tokens other than the fixed knee-frame one do not restore Fracture (four variants); the old recipe's image-frame codes cost the routed targets 0.010 to 0.012 | 4, 2 |
| A position prior that combines the axes costs 0.0008 of macro | 2 |
| The own-channel attention uses the token position (removing it there costs 0.011 over those targets) | 1 |

**Decisions waiting for the user:**

1. **The adoption rule.** A mark taken as the highest value any run reached keeps lucky draws
   (the fourth run's Medial Meniscus localisation blocked nine runs). The own-targets flag has
   no noise tolerance and blocked the two best-scoring runs on ACL and four runs on Baker's.
   Proposal: marks as the lineage's mean, and two sibling SDs of tolerance on the own-targets
   flag.
2. **Whether localisation pays on the hidden test.** Candidates for one submission: the
   incumbent, or `localising-design-all-three-parts` (PF OA localisation +0.018 and Medial
   Meniscus +0.006 over the fifth run, macro +0.0001). The public score followed the routed
   targets across S112 to S114.
3. **Where to go.** Refinements of the geometry are below what one run on one fold resolves.
   The researcher's proposals have become small variations of the priors and the knee-level
   term. Options: carry the design to the ResNet-50; or stop the campaign.

## Keeper repairs

### 2026-10-05 23:50 — the guardrail check needed a GPU the developer does not have

- **What happened:** the first forced seed, `routed-report-attribute-heads`, was implemented
  and then blocked twice: `scripts.tools.check_side_routing` called `.cuda()`, and the
  developer's environment has no GPU access ("GPU access blocked by the operating system").
  The orchestrator used the seed's one repair and blocked it. No training ran.
- **Cause:** mine. The contract makes the check mandatory before a run and I had only run the
  tool from a shell with a GPU.
- **Repair:** the tool now runs on the GPU when one is visible and on the CPU otherwise (four
  studies, nine seconds, 6.6 GB peak, all nine guardrails pass on the incumbent). The same file
  was installed in the campaign's parent source
  (`outputs/autoresearch/side-routing/baseline/source/scripts/tools/`), so every new workspace
  carries it and compares identical to its parent. The guardrails themselves are unchanged.
- **Requeue:** `routed-report-attribute-heads-requeue`, forced, with the orchestrator's request
  for an activation count that measures a real change of attention, and with the route names
  left to the incumbent's at the time of the run.
- **Cost:** about 15 minutes of idle GPU before the first training.

### 2026-10-06 08:50 — a third retry withdrawn from the queue

The researcher queued `bounded-hidden-interactions-for-own-attention`, a third way of
restructuring how the own-channel attention uses position, after runs 18 and 22 lost 0.0007 and
0.0022 on that direction (own-channel sum −0.011 and −0.027). Withdrawn with `remove-idea`
before the developer took it: the direction is closed, and its zero-started addition would be
expected to stay near zero. Earlier in the night I left a retry of the cross-axis priors in the
queue and it repeated its loss (run 20); this time the run was not spent.

## Adoption decisions

### 1. `front-and-up-region-for-pf-oa` — not adopted (2026-10-06 00:14)

×4 0.924270 against the incumbent's 0.924476: −0.000206 (80% [−0.0010, +0.0006]); ×1 0.915453.
Flags: macro below the incumbent; routed sum 0.0090 below the lineage's best (tolerance
0.0090); Medial Meniscus localisation 0.0060 below the best (tolerance 0.0025).

| Target | Δ against the incumbent | Localisation Δ |
|---|---:|---:|
| PF OA | +0.0013 | **+0.0153** |
| Medial OA | −0.0009 | −0.0014 |
| Lateral OA | −0.0027 | −0.0092 |
| Medial Meniscus | −0.0014 | −0.0026 |
| Lateral Meniscus | −0.0012 | −0.0017 |
| Fracture | +0.0036 | |
| ACL, MCL, Effusion, Synovitis, Baker's, Contusion | −0.0009 to +0.0005 | |

- **PF OA localises far better from front-and-up** (+0.0153, tolerance 0.0057): the region
  from the drawings is the right one for PF OA.
- **The OA sides did not recover** although they got the front-and-low region back; they fell
  a little further. So what they lost after the fourth run is not their own front but the PF
  region's evidence: OA in one compartment predicts OA in another (166 of 363 OA studies have
  several), and a target with an exclusive region cannot see it. The fourth run's overlapping
  regions gave that evidence to the sides (+0.0030, +0.0022, 0.953 public); the fifth run's
  and this run's exclusive regions took it away.
- **Lesson:** exclusive regions buy localisation and cost the plain AUC. The next design keeps
  the exclusive regions and returns the knee-level evidence explicitly
  (`knee-level-term-with-front-and-up-routing`).
- Learned: nod 3.6°, roll −0.5°, turn 1.3°; widths 4.5 (medial), 5.4 (front), 5.1 (up) mm;
  side head 100%.
- GPU idle around the decision: about 2 minutes (guard armed at epoch 8, resumed at once).

### 2. `learned-position-code-for-own-channels` — not adopted (2026-10-06 00:36)

×4 0.924062 against the incumbent's 0.924476: −0.000414; ×1 0.916001 at epoch 9. Flags: macro
below the incumbent; Fracture newly below every sibling run; Medial Meniscus localisation
0.0046 below the best (tolerance 0.0025).

| Target | Δ against the incumbent |
|---|---:|
| ACL | +0.0028 |
| Contusion | +0.0014 |
| Baker's, Effusion | +0.0009 |
| Synovitis | −0.0006 |
| Fracture | −0.0037 |
| MCL | −0.0047 |
| Routed five, summed | −0.0020 |

- **The seed's premise did not hold.** A learned position code did not lift Fracture; it fell
  0.0037, with MCL. I had read three campaign runs with a learned code as worth +0.010 on
  Fracture; all 12 sibling runs average +0.0072 there whatever their position code, so the
  learned code was not what set those three apart. What keeps Fracture, Baker's and ACL under
  the sibling runs in this design is still unknown: a question for measurement, not for
  another position variant.
- The fixed sinusoid stays.
- GPU idle around the decision: about 2 minutes.

### 3. `routed-report-attribute-heads-requeue` — not adopted (2026-10-06 00:59)

×4 0.924428 against the incumbent's 0.924476: −0.000048. Flags: macro below the incumbent (by
0.00005); Medial Meniscus localisation 0.0058 below the best (tolerance 0.0025).

- **Neutral on every number.** All twelve targets within ±0.0013 of the incumbent; routed sum
  −0.0021, own-channel sum +0.0015; localisation within −0.0024 to +0.0008. Sharing and routing
  12 report questions and dropping 2 neither helped nor cost.
- It is the cleaner structure (the report heads no longer ask the backbone for the
  compartment) at no measurable price, and the rule keeps it out by 0.00005 of macro and the
  Medial Meniscus mark. A candidate to carry into a later combined design.
- Learned: nod −4.5°, roll −2.8°, turn −0.6°; widths 4.5 and 5.7 mm; side head 100%.
- GPU idle around the decision: about 2 minutes.

### 4. `knee-level-term-with-front-and-up-routing` — not adopted (2026-10-06 01:20)

×4 0.924123 against the incumbent's 0.924476: −0.000353. Flag: macro below the incumbent. No
localisation flag: this is the first run to clear the Medial Meniscus mark's tolerance.

| Target | Δ against the incumbent | Localisation Δ |
|---|---:|---:|
| PF OA | +0.0005 | **+0.0129** |
| Lateral Meniscus | −0.0004 | +0.0042 |
| Medial Meniscus | −0.0019 | +0.0018 |
| Medial OA | −0.0013 | −0.0022 |
| Lateral OA | −0.0019 | −0.0074 |
| Own-channel targets | −0.0008 to +0.0012 | |

- **The knee-level strengths went negative:** meniscus −0.138, OA −0.035, settled by epoch 5.
  The model subtracts the knee-level evidence from each compartment's own, which sharpens the
  contrast between compartments (meniscus localisation up for both sides) and is the opposite
  of what the seed proposed. **My reading of the first run was wrong on the mechanism:** the
  OA sides are not short of the other compartments' evidence in a form a shared additive term
  can return; given the choice, training uses that term to tell compartments apart.
- **PF OA's localisation gain from front-and-up repeats** (+0.0153, then +0.0129), with its
  plain AUC level (+0.0013, +0.0005).
- **The OA sides are below the incumbent in both front-and-up runs** (−0.0009/−0.0027, then
  −0.0013/−0.0019), each within one to two sibling SDs for those targets.
- Learned: nod 3.6°, roll −0.2°, turn 1.4°; widths 4.7 (medial), 5.5 (front), 5.1 (up) mm.
- GPU idle around the decision: about 2 minutes.

### Where the campaign stands after four runs (2026-10-06 01:25)

| Run | ×4 macro | Δ against the incumbent |
|---|---:|---:|
| Incumbent (fifth run) | 0.924476 | |
| `front-and-up-region-for-pf-oa` | 0.924270 | −0.0002 |
| `learned-position-code-for-own-channels` | 0.924062 | −0.0004 |
| `routed-report-attribute-heads-requeue` | 0.924428 | −0.0000 |
| `knee-level-term-with-front-and-up-routing` | 0.924123 | −0.0004 |

All four are within 0.0004 of the incumbent, which is the standard deviation of the macro over
the 12 sibling runs of the old recipe. One run per design on one fold does not separate
refinements of this size on the macro; what the runs do separate is the localisation values
(PF OA +0.013 to +0.015 from front-and-up, twice) and the learned quantities (the sign of the
knee-level strengths).

### 5. `knee-level-term-on-incumbent-routing` — not adopted, by one flag (2026-10-06 01:45)

×4 **0.924834** against the incumbent's 0.924476: **+0.000358**, the highest of the line. One
flag: ACL newly below every sibling run.

| | Δ against the incumbent | Against the lineage's best | Tolerance |
|---|---:|---:|---:|
| Macro | +0.0004 | | must not fall |
| Routed sum | −0.0018 | −0.0058 | 0.0090 |
| Medial Meniscus localisation | +0.0041 | **+0.0007** | 0.0025 |
| Lateral Meniscus localisation | +0.0059 | +0.0047 | 0.0061 |
| Medial OA, Lateral OA, PF OA localisation | +0.0004, −0.0009, +0.0016 | +0.0004, −0.0009, −0.0012 | |
| Own-channel sum | +0.0060 (Fracture +0.0036, Synovitis +0.0022) | | |
| ACL | −0.0012 | 0.0002 below the lowest sibling | none |

- **The knee-level strengths are negative again** (meniscus −0.138, OA −0.021), the same values
  as with front-and-up routing: the model subtracts the knee-level evidence from each
  compartment's own. With the incumbent's regions that contrast lifts meniscus localisation
  above the fourth run's mark, the number that had blocked every earlier run.
- **Why it is not adopted:** the incumbent's ACL was already below 11 of the 12 sibling runs
  (−0.0041 against the old reference; the lowest sibling is −0.0051). This run's ACL is 0.0012
  lower, less than one sibling SD for ACL (0.0014), and that puts it 0.0002 under the lowest
  sibling. The own-targets flag has no noise tolerance, unlike the routed-sum and localisation
  flags.
- **For the user:** by the rule as written the run stays out, and I applied it. It is the
  design I would have adopted: the macro rose, the shared heads localise better than any run
  before, and the one flag is a move of under one SD across a threshold the incumbent already
  touched. Giving the own-targets flag the same two-SD tolerance as the others would adopt it.
- The term is carried forward in the next keeper design.
- Learned: nod −4.5°, roll −2.3°, turn −1.3°; widths 4.7 and 5.7 mm; side head 100%.
- GPU idle around the decision: about 2 minutes.

### 6. `image-frame-position-codes-for-own-channels` — not adopted (2026-10-06 02:07)

×4 0.924035 against the incumbent's 0.924476: −0.000441. Flags: macro below the incumbent;
routed sum 0.0136 below the best (tolerance 0.0090); Medial Meniscus localisation; Fracture
newly below every sibling.

| Target | Δ against the incumbent |
|---|---:|
| Baker's | +0.0026 |
| Synovitis | +0.0024 |
| ACL | +0.0013 |
| Fracture | −0.0016 |
| Own-channel sum | +0.0042 |
| Routed sum | −0.0095 (Lateral OA −0.0044, Lateral Meniscus −0.0018, PF OA −0.0017) |

- **The old recipe's codes did not bring Fracture back either.** At ×1 it led the incumbent by
  0.006 to 0.007 at epochs 7 and 8 and ended 0.0016 below at ×4. Three position variants for
  the own-channel tokens now (fixed knee-frame, learned knee-frame, the old image-frame pair):
  Fracture at +0.0000, −0.0037 and −0.0016. Fracture has 60 positives and moves about 0.004
  between sibling runs, so none of these separates from the others; what the old recipe has on
  Fracture is not recovered by the token position alone.
- Baker's, Synovitis and ACL rose a little; the routed targets fell although their tokens and
  routing did not change, which is the size of run-to-run movement on this fold.
- The combined design with the knee-level term was queued after all, on other grounds: the
  knee-level run was blocked only by ACL, and this run raised ACL by 0.0013.
- GPU idle around the decision: about 2 minutes.

### 7. `spatial-priors-for-routed-report-heads` (researcher) — not adopted, by one flag (2026-10-06 02:28)

×4 **0.924807** against the incumbent's 0.924476: **+0.000331**. One flag: Medial Meniscus
localisation 0.0036 under the fourth run's mark (tolerance 0.0025); against the incumbent it
is level (−0.0002).

- Routed sum −0.0001; own-channel sum +0.0040 (Fracture +0.0023, Synovitis +0.0013); no
  own-channel target below every sibling, Baker's included; ACL −0.0004.
- The researcher's design: the five shared report families keep their routing, get one
  zero-started position prior each, read detached coordinates and routing weights, and all 35
  questions stay.
- **Second run above the incumbent on the macro, second run kept out by a single flag.** This
  one is stopped by the mark that looks like a lucky draw (see below); the knee-level run
  cleared that mark and was stopped by ACL instead. Each clears the other's flag, so the two
  are combined in `knee-level-term-with-routed-report-heads-and-priors`.
- GPU idle around the decision: about 2 minutes.

### 8. `residual-position-code-on-fixed-basis` (researcher) — not adopted (2026-10-06 02:52)

×4 0.924474 against the incumbent's 0.924476: −0.000002. Flags: macro below the incumbent (by
two millionths); Medial Meniscus localisation 0.0075 under the mark. Routed sum −0.0033,
own-channel sum +0.0033 (Fracture +0.0044). A fourth position variant for the own-channel
tokens, and neutral like the others.

### 9. `knee-level-term-with-routed-report-heads-and-priors` — not adopted (2026-10-06 03:15)

×4 0.924400 against the incumbent's 0.924476: −0.000076. One flag: macro below the incumbent.
Every other condition holds: Medial and Lateral Meniscus localisation +0.0048 and +0.0040
(above the mark by 0.0014), ACL +0.0011, no new own-channel flag, routed sum within tolerance.

- **The two macro gains did not add up.** Its parts scored +0.000358 and +0.000331 alone; the
  combination scores −0.000076. Gains of that size are inside the run-to-run spread of the
  macro on this fold (SD 0.0004 over the sibling runs), and this run says so directly.
- **What did carry over is the localisation effect of the knee-level term:** meniscus
  localisation rose in all three runs that have it (+0.0018/+0.0042, +0.0041/+0.0059,
  +0.0048/+0.0040), with negative strengths each time.
- GPU idle around the decision: about 2 minutes.

### Where the campaign stands after nine runs (2026-10-06 03:15)

| Run | ×4 macro Δ | Robust effect |
|---|---:|---|
| `knee-level-term-on-incumbent-routing` | +0.0004 | Meniscus localisation +0.004 to +0.006 |
| `spatial-priors-for-routed-report-heads` | +0.0003 | |
| `residual-position-code-on-fixed-basis` | −0.0000 | |
| `routed-report-attribute-heads-requeue` | −0.0000 | |
| `knee-level-term-with-routed-report-heads-and-priors` | −0.0001 | Meniscus localisation +0.004 to +0.005 |
| `front-and-up-region-for-pf-oa` | −0.0002 | PF OA localisation +0.015 |
| `knee-level-term-with-front-and-up-routing` | −0.0004 | PF OA localisation +0.013, meniscus +0.002 to +0.004 |
| `learned-position-code-for-own-channels` | −0.0004 | |
| `image-frame-position-codes-for-own-channels` | −0.0004 | |

Mean −0.0001, all within ±0.0004. On the plain macro the nine designs are indistinguishable
from the incumbent and from each other with one run each. Two effects repeat and are larger
than their noise: front-and-up routing on PF OA localisation (twice), and the knee-level term,
learned as a contrast, on meniscus localisation (three times).

### 10. `knee-level-term-with-image-frame-position-codes` — not adopted (2026-10-06 03:38)

×4 0.923531 against the incumbent's 0.924476: −0.000945, the lowest of the campaign. Flags:
macro; routed sum 0.0163 below the best (tolerance 0.0090); PF OA localisation; Fracture newly
below every sibling.

- **The image-frame codes cost the routed targets, twice.** Routed sum −0.0095 alone and
  −0.0122 here (Lateral OA −0.0044 and −0.0070, PF OA −0.0017 and −0.0019), although the shared
  classifiers do not read the codes; the knee-level term's two other runs had −0.0018 and
  −0.0035. The effect goes through the backbone. With Fracture at −0.0016 and −0.0031, the old
  recipe's codes on the own-channel tokens are closed for this design.
- ACL +0.0024, Baker's +0.0032 and Synovitis +0.0014 rose again, as in the codes' own run.
- Meniscus localisation +0.0036 and +0.0020: the knee-level term's effect, a fourth time.
- GPU idle around the decision: about 2 minutes.

### 11. `separate-oa-frame-for-octant-routing` (researcher) — not adopted (2026-10-06 04:02)

×4 **0.924876** against the incumbent's 0.924476: **+0.000400**, the highest macro of the
campaign. Flags: Medial Meniscus localisation 0.0045 under the mark (tolerance 0.0025) and
Medial OA localisation 0.0031 under the best (tolerance 0.0030). Not guarded: the next training
launched at once and nothing was adopted, so no run was wasted.

- The researcher's design: the OA targets (PF OA from front-and-up) get a learned frame of
  their own; the meniscus targets, the priors and the own-channel position keep the other.
- **The two frames turned apart:** about the vertical axis the meniscus frame ended at +6.2°
  and the OA frame at −6.1°; nods −2.9° and +4.1°. With one shared frame the turn had stayed
  within ±1.6° in every run. The two diseases ask for different orientations of the midline,
  and a single frame was averaging them out.
- PF OA localisation +0.0137 (a third time with front-and-up); PF OA +0.0016; own-channel sum
  +0.0067; routed sum −0.0019.
- Three runs are now above the incumbent on the macro (+0.0004, +0.0003, +0.0004) and none
  joined; the first combination of two of them scored −0.0001.

### 12. `localising-design-all-three-parts` — not adopted (2026-10-06 04:24)

×4 0.924613 against the incumbent's 0.924476: +0.000137. One flag: the routed sum is 0.0107
under the fourth run's mark (tolerance 0.0090); against the incumbent it is −0.0066.

| Target | Δ against the incumbent | Localisation Δ | Localisation against the best |
|---|---:|---:|---:|
| PF OA | +0.0012 | **+0.0179** | +0.0152 |
| Medial Meniscus | −0.0012 | **+0.0059** | +0.0025 |
| Lateral Meniscus | −0.0024 | +0.0029 | +0.0018 |
| Medial OA | −0.0013 | −0.0009 | −0.0009 |
| Lateral OA | −0.0028 | −0.0100 | −0.0100 |
| Own-channel sum | +0.0082 (Fracture +0.0035, MCL +0.0031) | | |

- **The localisation effects hold together.** PF OA localisation is the best of every run
  (+0.0179), Medial Meniscus localisation the best too (+0.0059, 0.0025 above the fourth run's
  mark), with the knee-level strengths negative again. The parts do not interfere.
- **The plain AUC of the routed targets does not follow:** four of the five are 0.001 to 0.003
  under the incumbent, as in every run with front-and-up routing (routed sum −0.0049, −0.0048,
  −0.0019, −0.0066). Better localisation of a compartment has not meant a better score for it
  on this fold.
- This is the campaign's candidate for the hidden test: checkpoint under
  `outputs/autoresearch/side-routing/experiments/localising-design-all-three-parts-20261006T013826Z-2f113c93/training/fold_1/`.
- GPU idle around the decision: about 2 minutes.

### 13. `separate-oa-frame-with-knee-level-term` — **adopted** (2026-10-06 04:50)

×4 0.924485 against the incumbent's 0.924476: +0.000010. **No flag**: the first run of the
campaign whose report says `joins_lineage: true`. It is the new incumbent.

| | Value | Condition |
|---|---:|---|
| Macro | +0.00001 | not below the incumbent |
| Routed sum against the best | −0.0088 | tolerance 0.0090 |
| Medial Meniscus localisation | +0.0062 (0.0028 above the fourth run's mark) | |
| Lateral Meniscus localisation | +0.0039 | |
| PF OA localisation | +0.0114 | |
| Medial OA, Lateral OA localisation | −0.0019, −0.0061 | tolerances 0.0030, 0.0126 |
| Own-channel targets | sum +0.0048; none below every sibling (ACL +0.0014, Baker's +0.0008) | |
| Side head | 99.8% | |

- **The design:** the researcher's separate frame for OA with PF OA routed from front-and-up,
  plus the keeper's knee-level term. The frames turned apart again (meniscus +6.3°, OA −5.9°
  about the vertical axis; nods −2.5° and +3.7°) and the knee-level strengths were negative
  again: both learned behaviours repeated.
- **What it is and is not.** On the plain macro it equals the fifth run; it was adopted because
  it holds that macro while telling compartments apart better than any earlier base on PF OA
  and both meniscus targets. It is not evidence of a better score.
- **Mechanics.** The supervisor stopped with this run awaiting its decision, and `adopt` needs
  a decided run. I resumed, let the decision be recorded, aborted the training that launched
  in the same pass (`calibrated-shared-report-outputs`, built on the old incumbent, seconds
  in), adopted, and resumed. That idea is requeued for the new incumbent. GPU idle: about 12
  minutes, until the first candidate on the new incumbent is built.

### 14. `independent-pf-cut-normals` (researcher) — not adopted (2026-10-06 05:20)

First run on the new incumbent. ×4 0.924565 against 0.924485: +0.000079. Flags: Medial Meniscus
localisation 0.0029 under the best, which is now the incumbent's own value (tolerance 0.0025);
Baker's newly below every sibling. PF OA localisation +0.0047, routed sum +0.0012. Not guarded:
the next training launched at once.

### 15. `calibrated-shared-report-outputs-requeue` (researcher) — not adopted (2026-10-06 05:42)

×4 0.924630 against the incumbent's 0.924485: +0.000144. Flags: Medial OA localisation under the
best; Baker's newly below every sibling. Routed sum +0.0031, own-channel sum −0.0014. Not
guarded; no idle.

### 16. `cross-axis-attention-priors` (researcher) — not adopted (2026-10-06 06:05)

×4 0.923650 against the incumbent's 0.924485: −0.000836, the second lowest of the campaign.
Every kind of flag: macro, routed sum, Medial Meniscus and Medial OA localisation, and ACL,
Baker's and Contusion below every sibling. Own-channel sum −0.0064. Letting the position prior
combine the axes cost on both the own-channel and the routed targets; the additive prior
stays.

### 17. `family-mean-prior-for-knee-reference` (researcher) — not adopted (2026-10-06 06:28)

×4 0.924575 against the incumbent's 0.924485: +0.000090. Flags: Medial OA localisation under the
best; Baker's newly below every sibling (the third run on this incumbent stopped by Baker's).
Routed sum +0.0004, own-channel sum +0.0007. No idle.

### 18. `position-free-own-attention-with-positioned-values` (researcher) — not adopted (2026-10-06 06:50)

×4 0.923821 against the incumbent's 0.924485: −0.000665. Own-channel sum −0.0110 with Fracture
and MCL below every sibling; routed sum +0.0030. Taking the position code out of what the
own-channel attention reads, and leaving it only in what the classifiers read, cost the
own-channel targets: their attention uses the token position. No idle.

### 19. `rotation-before-side-frame-composition` (researcher) — not adopted (2026-10-06 07:12)

×4 0.924211 against the incumbent's 0.924485: −0.000275. Flags: macro; Baker's below every
sibling. Routed sum +0.0018, own-channel sum −0.0051, no localisation flag.

- The learned rotation is applied in the scanner's frame, before the knee side mirrors the
  coordinates. The two frames then no longer turn apart: +2.7° (meniscus) and +2.1° (OA)
  about the vertical axis, against +6.3° and −5.9° in the incumbent. A turn that is the same
  for left and right knees in the scanner's frame is not an anatomical turn, so this confirms
  what the opposite turns of the incumbent are: anatomical, mirrored between left and right
  knees, and different for the two diseases. The incumbent's order (mirror, then rotate) stays.

### 20. `cross-axis-priors-with-isolated-frame-gradients` (researcher) — not adopted (2026-10-06 07:35)

×4 0.923694 against the incumbent's 0.924485: −0.000791. The retry of the cross-axis priors
repeats their first result (−0.000836) with the same own-channel flags (ACL, Baker's,
Contusion): own-channel sum −0.0075. A cross-axis position prior costs about 0.0008 of macro,
twice; closed.

### 21. `mixture-of-member-priors-for-knee-reference` (researcher) — not adopted (2026-10-06 07:58)

×4 0.924483 against the incumbent's 0.924485: −0.000003. Neutral on every number (routed sum
−0.0003, own-channel sum +0.0002); kept out by the macro, the routed sum at its tolerance and
Baker's. No idle.

### 22. `detached-content-position-interactions-for-own-attention` (researcher) — not adopted (2026-10-06 08:22)

×4 0.922289 against the incumbent's 0.924485: **−0.002197**, the lowest of the campaign by far.
Own-channel sum −0.0266 with ACL, Baker's, Fracture and Synovitis below every sibling; routed
sum +0.0002. A second attempt at restructuring how the own-channel attention reads position,
after run 18 (−0.0007); both cost the own-channel targets and leave the routed ones alone. The
incumbent's arrangement (position code on the tokens the own-channel attention and classifiers
both read) stays.

### 23. `side-equivariant-shared-frames-with-independent-own-frame` (researcher) — not adopted (2026-10-06 08:45)

×4 0.924390 against the incumbent's 0.924485: −0.000095. A third frame, for the own-channel
targets. Neutral: routed sum +0.0021, own-channel sum −0.0032, no localisation flag; kept out
by the macro and Baker's. No idle.

### 24. `side-posterior-expectation-for-shared-position-priors` (researcher) — not adopted (2026-10-06 09:20)

×4 0.924249 against the incumbent's 0.924485: −0.000236. Routed sum +0.0029, own-channel sum
−0.0057; kept out by the macro and Baker's. The last run before the stop.

### Stopped by the user (2026-10-06 09:22)

The user stopped the campaign gracefully after 24 decided runs and one adoption. The current
training finished and nothing new launched; one candidate was prepared
(`side-hypothesis-attention-marginalization`) and two queued. On reading the adopted design the
user did not want the second frame, which departs from the single rotating frame of the fifth
build; whether the adoption is undone is the user's open decision.

### A standing obstacle: the fourth run's Medial Meniscus mark

Both runs raise the Medial Meniscus localisation flag although neither changed the meniscus
routing. The mark comes from the fourth run; the fifth run, the starting code, is already
0.0034 below it against a tolerance of 0.0025 (twice the sibling SD of 0.0013). So a run on the
fifth run's code joins the lineage only if it also improves Medial Meniscus localisation by
about 0.001. Between the fourth and the fifth run the meniscus cut lost its learned lateral
offset (2 mm) and gained the frame's rotation. The rule stays as the user set it; the routed
report-attribute heads, next in the queue, put six more meniscus questions on that cut.

**Probe (2026-10-06 00:45, inference only, ×1, the incumbent's checkpoint):** shifting the
midline towards lateral does not bring the mark back
(`outputs/analysis/side-routing-offset-probe-20261006/offsets.csv`).

| Lateral shift | Medial Meniscus localisation | Lateral Meniscus localisation | Medial OA localisation |
|---:|---:|---:|---:|
| −2 mm | 0.9062 | 0.8410 | 0.8255 |
| 0 | 0.9064 | 0.8440 | 0.8234 |
| +2 mm | 0.9059 | 0.8459 | 0.8206 |
| +5 mm | 0.9028 | 0.8470 | 0.8155 |

Every change within ±2 mm is under 0.003 and the medial values fall as the cut moves lateral.
The missing lateral offset is not what the fifth run lost on Medial Meniscus; a model trained
with the offset was not tested.

**Is the mark a level or a lucky draw? (2026-10-06 00:50)** Medial Meniscus localisation at ×4,
every run of the line beside the 13 runs of the old recipe on this fold:

| Runs | Medial Meniscus localisation |
|---|---|
| Old recipe, 13 runs | mean 0.9239, SD 0.0013, highest 0.9256 |
| Runs 1, 2, 3 | 0.9253, 0.9276, 0.9264 |
| **Run 4 (the mark)** | **0.9317** |
| Run 5 (the incumbent) | 0.9283 |
| `front-and-up-region-for-pf-oa`, `learned-position-code-for-own-channels` | 0.9256, 0.9271 |

Six runs of the line sit between 0.9253 and 0.9283; the fourth run alone is at 0.9317, about
three sibling SDs above them. Taking the highest value any run reached as the mark picks up
such a draw and keeps it: with one mark per number and several numbers, some mark will be a
lucky one. **For the user:** as it stands the lineage needs a run to reach about 0.929 on this
one number before anything can be adopted. Options: keep the rule; take the fourth run out of
the marks; or use the mean of the lineage's runs as the mark in place of the highest.

## Seeds added by the keeper

| Seed | When | Why |
|---|---|---|
| `routed-report-attribute-heads-requeue` | 2026-10-05 23:55 | Requeue after the harness failure above |
| `knee-level-term-with-front-and-up-routing` | 2026-10-06 00:20 | The first run showed that exclusive regions cost the OA sides the other compartments' evidence; this keeps the regions and adds a shared knee-level term |
| `knee-level-term-on-incumbent-routing` | 2026-10-06 01:00 | The knee-level term on the incumbent's own regions, so the combined design and this one show what the term is worth with each region; also keeps a candidate ready while the researcher starts |
| `image-frame-position-codes-for-own-channels` | 2026-10-06 01:23 | The largest local deficit left (Fracture, Baker's, ACL below the sibling runs): the one sibling without the old recipe's slice-position and in-image codes reproduces it, and the learned knee-frame code did not cure it |
| `knee-level-term-with-image-frame-position-codes` | 2026-10-06 02:10 | The best run so far (knee-level term, +0.0004, blocked only by ACL) with the one change that raised ACL (+0.0013), ahead of the researcher's two weaker proposals |
| `knee-level-term-with-routed-report-heads-and-priors` | 2026-10-06 02:30 | The two runs above the incumbent (+0.0004, +0.0003), each kept out by a flag the other clears |
| `localising-design-all-three-parts` | 2026-10-06 03:42 | The one design carrying every repeated localisation gain (front-and-up PF OA, the knee-level term) plus the routed report heads with priors; the candidate to judge on the hidden test, since the macro of single runs no longer ranks designs |
| `separate-oa-frame-with-knee-level-term` | 2026-10-06 04:05 | The highest macro of the campaign (separate OA frame, +0.0004) with the mechanism that cleared its main flag, the Medial Meniscus mark, in three of four runs |
| `calibrated-shared-report-outputs-requeue` | 2026-10-06 04:52 | Requeue of the researcher's idea on the new incumbent, after its launch on the old one was cancelled at the adoption |
