# Student distillation: run notes

Written after each result: the overall and the gold score against the incumbent and the
reference, the per-label table of both, what the label table of the run looked like, repairs and
adoption decisions.

## Scores across the campaign

One row per run, in the order run. Differences are against the incumbent at the time of the run.

| Run | Recipe choice | Overall ×1 | Δ | Overall ×4 | Δ | Gold ×4 | Δ | Adopted |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `teacher-share-050` | 0.5 silver + 0.5 raw joint teacher | 0.928982 | n/a | 0.934171 | n/a | 0.938740 | n/a | yes, forced |
| `teacher-share-025` | 0.75 silver + 0.25 raw joint teacher | 0.930031 | +0.001048 | 0.935438 | +0.001267 | 0.937752 | −0.000988 | no: comparison failed in the runtime, requeued as `teacher-share-025-rerun` |
| `teacher-share-100` | teacher predictions alone | 0.926529 | −0.002453 | 0.931358 | −0.002813 | 0.935659 | −0.003081 | no |
| `teacher-share-025-rerun` | 0.75 silver + 0.25 raw joint teacher | 0.930532 | +0.001549 | 0.935809 | +0.001638 | 0.938714 | −0.000026 | yes |
| `teacher-share-075-rerun` | 0.25 silver + 0.75 raw joint teacher | 0.928151 | −0.002380 | 0.932562 | −0.003247 | 0.937050 | −0.001664 | no |
| `teacher-share-per-label` | share 0.5 / 0.25 / 0 by the teachers' AUC per label | 0.930547 | +0.000015 | 0.935444 | −0.000365 | 0.937075 | −0.001639 | no |
| `recalibrated-fold-teachers-rerun` | 0.25 share, teachers recalibrated per fold and label | 0.929470 | −0.001061 | 0.935983 | +0.000174 | 0.938705 | −0.000010 | no |
| `share-from-plane-agreement` | share 0.125 to 0.375 per cell from the agreement of the teacher's planes | 0.929954 | −0.000577 | 0.935635 | −0.000174 | 0.938092 | −0.000622 | no |
| `reranked-fold-teachers-rerun` | 0.25 share, teachers reranked per fold onto the pooled distribution | 0.929954 | −0.000578 | 0.935415 | −0.000394 | 0.937680 | −0.001034 | no |
| `target-per-plane` | 0.25 share plus a per-plane target from the teacher's single-plane readings, weight 0.5 | 0.930126 | −0.000406 | 0.935184 | −0.000625 | 0.937384 | −0.001331 | no |
| `teacher-beyond-repeat-noise` | 0.25 share only where the teacher differs from silver by more than 0.18 | 0.929884 | −0.000648 | 0.935639 | −0.000170 | 0.936252 | −0.002462 | no |
| `teacher-residual-at-silver-level` | 0.25 share of the teacher's residual at its silver level (20 bins) | 0.930722 | +0.000190 | 0.936106 | +0.000297 | 0.938511 | −0.000204 | no |

## Before the first run (2026-10-10)

- **Teachers:** ×4 macro AUC on their own fold 0.925641 (fold 1), 0.922001 (fold 2), 0.934859
  (fold 3), 0.922936 (fold 4). Folds 1 to 3 are in `outputs/training/student-distillation-teachers`,
  fold 4 in `outputs/training/student-distillation-teacher-fold-4`.
- **Why two teacher runs:** the four-fold run was killed by the kernel for memory at 02:42 CEST
  during fold 4 (32 loader workers at 28 GB, 19.6 GB of shared memory, swap full) and WSL shut down
  with it. Memory available falls inside a fold (27 GB to 14 GB in ten minutes) and is not given
  back between folds of one process. Fold 4 was retrained alone. One fold per process on this
  machine.
- **Teacher table:** 3,475 studies and joint rows; per series type 3,475 axial fluid, 3,368 sagittal
  T1/PD, 3,362 coronal fluid, 3,275 sagittal fluid, 2,688 coronal T1/PD, 665 axial T1/PD; logits
  between −5.81 and 5.60, none missing. `predict_teacher_table.py` now skips a fold without a
  selected checkpoint.
- **Repair, activation metric:** the first start blocked `teacher-share-050` twice without
  training: every candidate must name a logged metric that proves its mechanism ran, the recipe
  logged none for teacher mixing, and a configuration-only seed cannot add one. The recipe now logs
  `teacher_target_shift` at every epoch (mean absolute change of the training labels against the
  silver ones, measured in `data/labels.py`: 0.0283 at share 0.25, 0.0567 at 0.5, 0.1133 at 1.0), and
  the developer instructions name it. The campaign was started afresh at 08:13 CEST; the blocked
  start is in `outputs/autoresearch/student-distillation-blocked-start-20261010`.

## teacher-share-050 (adopted by force, the first incumbent)

Overall ×1 0.928982, overall ×4 0.934171, gold ×4 0.938740 (58 studies). No comparison: it is the
incumbent every later seed is scored against. Training 1,491 s of 1,648 allowed; best epoch the
last one. Label table: mean absolute change per label from 0.043 (MCL, ACL) to 0.083 (PF OA),
3.2% of cells cross 0.5, training-label macro AUC against silver 0.9952.

| Label | Overall ×4 | Gold ×4 |
|---|---:|---:|
| Baker's cyst | 0.9613 | 0.9837 |
| Medial OA | 0.9562 | 0.9922 |
| ACL | 0.9552 | 0.9914 |
| Medial meniscus | 0.9463 | 0.9639 |
| Fracture | 0.9423 | 0.9431 |
| MCL | 0.9399 | 0.9478 |
| Contusion | 0.9366 | 0.9447 |
| Lateral meniscus | 0.9258 | 0.9068 |
| Lateral OA | 0.9210 | 0.8859 |
| Effusion | 0.9208 | 0.9702 |
| PF OA | 0.9133 | 0.9048 |
| Synovitis | 0.8913 | 0.8303 |

Lesson: the forced adoption and the sealing against a reference from another split both work.
The runtime's own results listing shows the forced run's scores as unavailable, because it has no
comparison; the scores here come from its saved ×4 predictions.

## teacher-share-025 (invalid by a runtime failure, requeued)

The run trained and passed its checks; its comparison job failed with `Unexpected input identity
differences: ['training_rows_sha256']` and the runtime recorded it invalid. Every seed of this
campaign changes the training labels, so that identity always differs from the incumbent's; the
policy listed no identity key allowed to differ.

- **Repair (09:22 CEST):** campaign stopped with `--abort` (one minute of `teacher-share-075` lost),
  `policy.input_identity_keys: [training_rows_sha256]` set in `campaign.yaml` and in the sealed
  policy of `state.json` (copy of the old state: `state.before-identity-key.json`), the validation
  identities stay protected. `teacher-share-025-rerun` and `teacher-share-075-rerun` imported as
  forced seeds ([rerun seeds](seed-ideas-reruns.yaml)) and placed behind `teacher-share-100`, which
  the developer already held.
- **Scores of the invalid run, computed by hand from its saved predictions** (not the runtime's
  decision; no bootstrap interval): overall ×1 0.930031 (+0.001048), overall ×4 0.935438
  (+0.001267), gold ×4 0.937752 (−0.000988), against `teacher-share-050`. All three are inside the
  adoption rule; the rerun decides.

| Label | Overall ×4 Δ | Gold ×4 Δ |
|---|---:|---:|
| Synovitis | +0.0042 | +0.0060 |
| Fracture | +0.0038 | −0.0097 |
| Medial meniscus | +0.0020 | −0.0048 |
| Lateral meniscus | +0.0018 | +0.0025 |
| Lateral OA | +0.0017 | −0.0097 |
| MCL | +0.0016 | −0.0023 |
| Effusion | +0.0014 | +0.0075 |
| Medial OA | +0.0002 | −0.0047 |
| ACL | +0.0001 | +0.0012 |
| PF OA | −0.0000 | +0.0103 |
| Baker's cyst | −0.0007 | −0.0109 |
| Contusion | −0.0008 | +0.0027 |

Overall and gold point in opposite directions (overall up, gold slightly down), with the per-label
gold changes of both signs and up to 0.011 on 9 to 35 positives.

## teacher-share-100 (rejected)

Teacher predictions alone, against `teacher-share-050`: overall ×1 0.926529 (−0.002453), overall
×4 0.931358 (−0.002813, 95% [−0.004023, −0.001723]), gold ×4 0.935659 (−0.003081). All three
measures fail the rule and point the same way. First comparison on the repaired identity check:
the three measures were written and read as designed.

| Label | Overall ×4 Δ | Gold ×4 Δ |
|---|---:|---:|
| Contusion | +0.0023 | +0.0013 |
| Fracture | -0.0000 | -0.0056 |
| Medial meniscus | -0.0013 | -0.0024 |
| Medial OA | -0.0015 | -0.0031 |
| Baker's cyst | -0.0019 | -0.0072 |
| Effusion | -0.0023 | +0.0000 |
| Lateral meniscus | -0.0027 | +0.0012 |
| MCL | -0.0030 | +0.0023 |
| PF OA | -0.0032 | -0.0013 |
| Lateral OA | -0.0048 | +0.0019 |
| ACL | -0.0049 | -0.0110 |
| Synovitis | -0.0105 | -0.0131 |

Eleven of twelve labels fall on the whole fold; synovitis loses most (−0.0105, interval excludes
zero), then ACL and lateral OA. With the hand-scored `teacher-share-025` (+0.0013 at ×4) the
share reads so far: 0.25 above 0.5 above 1.0, so the silver labels carry information the teachers
lose, most of it on synovitis.

## teacher-share-025-rerun (adopted, the incumbent since 10:21 CEST)

0.75 silver + 0.25 raw joint teacher, against `teacher-share-050`: overall ×1 0.930532 (+0.001549),
overall ×4 0.935809 (+0.001638, 95% [+0.000553, +0.002808]), gold ×4 0.938714 (−0.000026). All three
inside the rule; adopted by the runtime. Training 1,512 s. The first attempt of the same recipe,
scored by hand, gave +0.001048, +0.001267 and −0.000988: the two trainings differ by 0.0004 to
0.0005 on the whole fold and by 0.0010 on gold.

| Label | Overall ×4 Δ | Gold ×4 Δ |
|---|---:|---:|
| Synovitis | +0.0050 | +0.0096 |
| Fracture | +0.0038 | −0.0125 |
| Lateral OA | +0.0023 | −0.0039 |
| Medial meniscus | +0.0020 | −0.0024 |
| Effusion | +0.0020 | +0.0087 |
| Contusion | +0.0019 | −0.0013 |
| MCL | +0.0017 | +0.0023 |
| Lateral meniscus | +0.0016 | +0.0087 |
| ACL | +0.0003 | +0.0012 |
| Medial OA | +0.0003 | −0.0031 |
| PF OA | +0.0001 | +0.0051 |
| Baker's cyst | −0.0013 | −0.0127 |

Eleven of twelve labels rise on the whole fold and gold is flat, so overall and gold do not
disagree here. Synovitis gains most in both attempts and in both scores, and is the label that
lost most at share 1.0: it is the label where the teacher's values cost most. Fracture and
Baker's cyst fall on gold in both attempts (18 and 12 gold positives). The share ordering is now
0.25 > 0.5 > 1.0 on every measure; nothing below 0.25 has been run, and the reference with no
teacher is on another fold, so the campaign has not yet shown that any teacher share beats none.

## teacher-share-075-rerun (rejected)

0.25 silver + 0.75 raw joint teacher, against `teacher-share-025-rerun`: overall ×1 0.928151
(−0.002380), overall ×4 0.932562 (−0.003247, 95% [−0.004665, −0.001901]), gold ×4 0.937050
(−0.001664). Overall and gold point the same way. Training 1,506 s.

| Label | Overall ×4 Δ | Gold ×4 Δ |
|---|---:|---:|
| Contusion | −0.0002 | −0.0013 |
| Lateral meniscus | −0.0004 | −0.0050 |
| Baker's cyst | −0.0008 | +0.0036 |
| PF OA | −0.0016 | −0.0051 |
| Medial OA | −0.0018 | +0.0031 |
| Medial meniscus | −0.0021 | +0.0036 |
| ACL | −0.0030 | −0.0049 |
| Effusion | −0.0030 | −0.0062 |
| MCL | −0.0045 | −0.0136 |
| Lateral OA | −0.0050 | +0.0155 |
| Fracture | −0.0061 | +0.0083 |
| Synovitis | −0.0105 | −0.0179 |

Every label falls on the whole fold. The share range is monotone at ×4 on fold 0: 0.25 0.935809,
0.5 0.934171, 0.75 0.932562, 1.0 0.931358, about −0.0016 per quarter of share, and the same order at
×1 (0.930532, 0.928982, 0.928151, 0.926529). Gold ×4: 0.938714, 0.938740, 0.937050, 0.935659.
Synovitis carries the largest part at every step. The line has not turned yet at 0.25;
`teacher-share-0125` is the next point.

## teacher-share-per-label (rejected by the rule; decision record pending at the stop)

Shares from the teachers' AUC per label on folds 1 to 4: 0.5 for ACL, medial meniscus, Baker's cyst
and contusion; 0.25 for lateral meniscus, medial OA, effusion and fracture; 0 for MCL, lateral OA,
PF OA and synovitis. Against `teacher-share-025-rerun`: overall ×1 0.930547 (+0.000015), overall ×4
0.935444 (−0.000365, 95% [−0.001240, +0.000511]), gold ×4 0.937075 (−0.001639). The ×4 measure fails
the rule. Training 1,539 s. The comparison finished as the campaign stopped; the orchestrator
records the decision at the next resume.

| Label | Share | Overall ×4 Δ | Gold ×4 Δ |
|---|---:|---:|---:|
| Synovitis | 0 | +0.0037 | −0.0048 |
| MCL | 0 | +0.0019 | −0.0113 |
| Lateral OA | 0 | +0.0015 | −0.0019 |
| Baker's cyst | 0.5 | +0.0008 | +0.0072 |
| PF OA | 0 | +0.0001 | +0.0064 |
| Medial meniscus | 0.5 | −0.0006 | +0.0000 |
| Medial OA | 0.25 | −0.0006 | −0.0016 |
| Contusion | 0.5 | −0.0015 | −0.0054 |
| Effusion | 0.25 | −0.0016 | −0.0012 |
| ACL | 0.5 | −0.0022 | −0.0037 |
| Lateral meniscus | 0.25 | −0.0026 | −0.0062 |
| Fracture | 0.25 | −0.0033 | +0.0028 |

Mean overall change by group: share 0 (no teacher) +0.0018, share 0.5 −0.0009, share 0.25
(unchanged recipe) −0.0020. On the whole fold the four labels that lost the teacher all rose or
held, and the four given more teacher fell on average, which is what the seed said would show it
wrong; it also agrees with the share range (less teacher is better). The four unchanged labels
moved by −0.0006 to −0.0033 with an identical label recipe, so a single label moves by up to
0.003 between two trainings. Gold does not follow the same grouping (share 0: −0.0029 on average).

## Stop of 11:18 CEST (user)

- The user stopped the campaign after `teacher-share-per-label`; it is not to be resumed without
  the user.
- `teacher-share-0125`, a seed the keeper had added, was removed at the user's request before it
  trained: retired in the stopped state, since `remove-idea` refuses a candidate the developer has
  prepared (old state: `state.before-remove-0125.json`; record under `decisions/`).
- `reranked-fold-teachers` is the queue head. Its implementation is prepared and checked on the
  label table: prepared candidate (`prepared/reranked-fold-teachers/report.md`, deleted). The seed's
  implementation text in the state points the developer at it.
- S134 (share 0.25) and S135 (share 0.75) were submitted at the user's request, ×4, τ 0.9, no TTA.

## Resumes, the time stop and the budget (12:45 to 13:26 CEST)

- **12:45, resumed by the user.** `reranked-fold-teachers` (the prepared patch, applied by the
  developer in two minutes) trained two epochs at 219 and 218 s against the usual 150 s; the runtime
  projected 2,182 s against the 1,648 s ceiling, stopped it and recorded it invalid, no evidence
  against the idea. The developer found nothing to repair inside the rules and the orchestrator
  parked the seed.
- **Cause of the slow epochs: not established.** Seconds per epoch that day: 147 to 154 from 08:20
  to 10:25, 150 to 158 for `teacher-share-per-label` at 10:50, about 220 for another session's run
  at 12:00 and 218 for the reranked run at 12:48, then 150 for the incumbent's source at 13:01 (a
  two-epoch check, `outputs/training/machine-speed-check-20261010`) and 151 to 155 after the machine
  was restarted. The slow period matches the machine and not the patch, which adds no per-batch
  work; the reranked rerun on a normal machine is the direct test.
- **12:57, stopped with abort by the user**; `recalibrated-fold-teachers` lost one minute of
  training and was requeued as `recalibrated-fold-teachers-rerun`.
- **13:16, resumed by the user after restarting the machine.** `reranked-fold-teachers-rerun`
  requeued with the same patch once the first epoch showed 155 s.
- **13:26, time budget raised from 1.1 to 1.33 by the user**: training ceiling 1,992 s, 2,290 s per
  experiment. Applied by restarting only the supervisor, which re-adopted the running training
  (old state and limits: `state.before-budget-133.json`, `limits.before-budget-133.json`).

## recalibrated-fold-teachers-rerun (rejected)

Each fold teacher's joint logit refitted per label on the silver labels of its fold (logistic,
slope and offset), share 0.25, against `teacher-share-025-rerun`: overall ×1 0.929470 (−0.001061),
overall ×4 0.935983 (+0.000174, 95% [−0.000642, +0.001000]), gold ×4 0.938705 (−0.000010). The ×4
gain is below +0.0009 and ×1 is below zero, so two measures fail. Training 1,518 s, epochs at 151
to 155 s.

| Label | Overall ×4 Δ | Gold ×4 Δ |
|---|---:|---:|
| MCL | +0.0023 | −0.0045 |
| Lateral meniscus | +0.0018 | −0.0050 |
| Synovitis | +0.0016 | −0.0084 |
| Baker's cyst | +0.0007 | +0.0091 |
| Medial OA | +0.0003 | −0.0031 |
| Medial meniscus | +0.0003 | +0.0060 |
| Lateral OA | +0.0002 | +0.0058 |
| PF OA | +0.0001 | +0.0039 |
| Effusion | −0.0005 | +0.0000 |
| ACL | −0.0005 | −0.0025 |
| Fracture | −0.0020 | −0.0028 |
| Contusion | −0.0021 | +0.0013 |

Level with the incumbent at ×4 and on gold; the two trainings of share 0.25 differed by 0.0004 at
×4, so +0.0002 is inside what a retraining gives. ×1 and ×4 point in opposite directions (−0.0011
and +0.0002). Eight labels rise a little on the whole fold, none by more than 0.0023.

## share-from-plane-agreement (rejected)

Share per study and label from the spread d of the teacher's single-plane sigmoid readings:
s = clip(0.25 × (1 − d / d90) + 0.125, 0, 1), d90 the label's 90th percentile of d on folds 1 to 4,
so 0.375 where the planes agree and 0.125 for the most divided tenth; the value mixed in stays the
raw joint one. Against `teacher-share-025-rerun`: overall ×1 0.929954 (−0.000577), overall ×4
0.935635 (−0.000174, 95% [−0.000990, +0.000596]), gold ×4 0.938092 (−0.000622). All three slightly
below; level within what a retraining gives (0.0004 at ×4). Training 1,558 s. Label table: mean
absolute change against the incumbent 0.00872, against silver 0.02304 (incumbent 0.02833), so the
table carries a little less teacher overall; 0.31% of cells cross 0.5.

| Label | Overall ×4 Δ | Gold ×4 Δ |
|---|---:|---:|
| MCL | +0.0038 | +0.0091 |
| Lateral OA | +0.0020 | −0.0019 |
| Lateral meniscus | +0.0020 | −0.0062 |
| Synovitis | +0.0015 | +0.0000 |
| PF OA | +0.0014 | +0.0064 |
| Baker's cyst | +0.0000 | −0.0000 |
| Medial meniscus | −0.0005 | +0.0000 |
| ACL | −0.0010 | −0.0012 |
| Effusion | −0.0017 | −0.0025 |
| Medial OA | −0.0019 | −0.0016 |
| Fracture | −0.0036 | −0.0042 |
| Contusion | −0.0041 | −0.0054 |

Overall and gold point the same way. MCL, lateral OA, synovitis and PF OA rise again: they are the
four labels that rose when the teacher was removed in `teacher-share-per-label` and three of them
rose under recalibration. Contusion and fracture fall in this run and in the recalibrated one
(−0.0021 and −0.0020 there). Three label-table variants at share 0.25 (per label, recalibrated,
plane agreement) all land within 0.0004 of the incumbent at ×4: the way the 0.25 of teacher is
shaped moves the macro score less than the amount of teacher did.

## Seed added at the user's request (14:07 CEST, from the monitor page)

`teacher-beyond-repeat-noise` ([gate seed](seed-ideas-gate.yaml)), proposed by a sub-agent and
queued forced: the teacher enters only where |teacher − silver| > 0.18, twice the 0.092 standard
deviation of the teacher's value between studies that share a report (35 groups, folds 1 to 4);
20.33% of cells keep the teacher, target shift 0.0172 against the incumbent's 0.0283 (recomputed by
the keeper). Its table moves the labels as much as a plain share of 0.152 would, so a gain is read
with the per-label table.

## Second seed and the end of the campaign (user, 14:31 CEST, from the monitor page)

- `teacher-residual-at-silver-level` ([level seed](seed-ideas-level.yaml)), proposed by a second
  sub-agent and queued forced as the last run: label = clip(silver + 0.25 × (p − m), 0, 1), with m
  the mean teacher value of the study's silver bin (20 equal-count bins per label, folds 1 to 4). It
  keeps the teacher's ordering inside a silver level and removes its pull of the silver values toward
  the middle (+0.089 where silver < 0.1, −0.167 where silver ≥ 0.5; 32% of the squared
  teacher-silver difference). Mean absolute change against the incumbent 0.02158 and 1.68% of cells
  clipped at 0, both recomputed by the keeper.
- The user ends the campaign once the queue has run: `reranked-fold-teachers-rerun`,
  `target-per-plane`, `teacher-beyond-repeat-noise`, `teacher-residual-at-silver-level`. The keeper
  asks for the stop when the last one starts training, so the researcher starts no idea of its own.

## reranked-fold-teachers-rerun (rejected)

The prepared patch, unchanged, on the restarted machine: epochs at about 150 s (first epoch 157 s),
training 1,502 s. This clears the patch of the slow epochs of 12:48; they were the machine's.
Against `teacher-share-025-rerun`: overall ×1 0.929954 (−0.000578), overall ×4 0.935415
(−0.000394, 95% [−0.001167, +0.000287]), gold ×4 0.937680 (−0.001034). All three below, the ×4
difference at the size of what a retraining gives (0.0004).

| Label | Overall ×4 Δ | Gold ×4 Δ |
|---|---:|---:|
| Baker's cyst | +0.0015 | +0.0054 |
| Lateral meniscus | +0.0007 | −0.0000 |
| Contusion | +0.0007 | −0.0027 |
| Synovitis | +0.0006 | +0.0012 |
| Effusion | +0.0000 | −0.0037 |
| Lateral OA | −0.0003 | −0.0039 |
| Medial meniscus | −0.0004 | +0.0012 |
| Medial OA | −0.0006 | −0.0016 |
| PF OA | −0.0010 | +0.0026 |
| ACL | −0.0012 | +0.0000 |
| MCL | −0.0022 | −0.0068 |
| Fracture | −0.0025 | −0.0042 |

The spread of the per-fold teacher mean (0.008 to 0.042 per label) was removed and the student did
not gain: the level differences between the four fold teachers are not what limits the recipe at
share 0.25, where they move a label by 0.0025 on average. Both teacher preparations are now run
(reranked −0.0004, recalibrated +0.0002): the answer to the contract's second choice is that the
raw values are as good as either. Fracture falls for the third run in a row (−0.0020, −0.0036,
−0.0025) while the teacher's 0.25 is reshaped three different ways, so it is not following the
preparation.

## target-per-plane (rejected)

Added loss: 0.5 × the soft-target BCE of each series type's own readout (`pool_encoded` with the
view) against the sigmoid of the teacher's reading from the same series type, averaged over the
types present; the joint label stays the incumbent's. Against `teacher-share-025-rerun`: overall ×1
0.930126 (−0.000406), overall ×4 0.935184 (−0.000625, 95% [−0.001716, +0.000455]), gold ×4 0.937384
(−0.001331). All three below. Training 1,520 s, epochs at normal speed.

| Label | Overall ×4 Δ | Gold ×4 Δ |
|---|---:|---:|
| Baker's cyst | +0.0018 | +0.0000 |
| Effusion | +0.0013 | +0.0025 |
| ACL | +0.0009 | −0.0012 |
| Fracture | +0.0004 | +0.0028 |
| Medial meniscus | −0.0001 | +0.0060 |
| PF OA | −0.0004 | +0.0013 |
| Contusion | −0.0007 | −0.0108 |
| Lateral OA | −0.0011 | +0.0039 |
| Medial OA | −0.0015 | +0.0000 |
| Synovitis | −0.0016 | −0.0024 |
| Lateral meniscus | −0.0024 | −0.0112 |
| MCL | −0.0038 | −0.0068 |

Learning curve at ×1 against the incumbent, by epoch: +0.0041, +0.0021, +0.0013, +0.0003, +0.0003,
+0.0010, −0.0001, −0.0007, −0.0001, −0.0004. The per-plane target speeds up the first epochs and
the lead is gone by epoch 7: it helps the student reach the teacher's readings sooner and adds
nothing the joint label does not give by the end of the schedule. The per-plane targets are raw
teacher values at full weight inside their loss term, so this run also carries more teacher than
the incumbent; MCL and synovitis, which rise whenever the teacher is reduced, fall here.

## teacher-beyond-repeat-noise (rejected)

Label = silver + 0.25 × (teacher − silver) where |teacher − silver| > 0.18, silver elsewhere (20.3%
of cells keep the teacher). Against `teacher-share-025-rerun`: overall ×1 0.929884 (−0.000648),
overall ×4 0.935639 (−0.000170, 95% [−0.000874, +0.000514]), gold ×4 0.936252 (−0.002462). Level at
×4, below at ×1, and the gold measure fails the rule on its own (−0.0025 against −0.002), the only
run of the campaign to do so beside `teacher-share-100`. Training 1,499 s.

| Label | Overall ×4 Δ | Gold ×4 Δ |
|---|---:|---:|
| ACL | +0.0013 | +0.0000 |
| Lateral OA | +0.0011 | −0.0058 |
| PF OA | +0.0009 | +0.0026 |
| Baker's cyst | +0.0008 | +0.0054 |
| MCL | +0.0005 | −0.0023 |
| Medial OA | +0.0002 | +0.0000 |
| Medial meniscus | −0.0001 | −0.0072 |
| Synovitis | −0.0005 | −0.0119 |
| Lateral meniscus | −0.0009 | −0.0062 |
| Effusion | −0.0015 | +0.0012 |
| Contusion | −0.0018 | −0.0067 |
| Fracture | −0.0020 | +0.0014 |

The seed's own prediction fails: it expected the largest gains on synovitis, MCL, fracture and
lateral OA, the labels with the most sub-noise jitter, and synovitis (−0.0005, gold −0.0119) and
fracture (−0.0020) fall. Removing the teacher's small moves, 80% of the cells and 39% of the label
shift, changed the whole-fold score by nothing measurable and cost on gold, so the small
differences are not harmful jitter; on gold they look useful. The table carried as much shift as a
plain share of 0.152 and did not gain, which is one indirect reading against "less teacher is
better below 0.25", with the shift concentrated in a fifth of the cells.

## Stop requested (15:41 CEST)

The last seed, `teacher-residual-at-silver-level`, started training at 15:40. The keeper asked for
the stop at 15:41, as the user instructed: its training and comparison finish and nothing else
launches. The researcher had started one job when the forced seeds ran out; its proposals are not
run.

## teacher-residual-at-silver-level (rejected by the rule; the campaign's last run)

Label = clip(silver + 0.25 × (teacher − m), 0, 1), m the mean teacher value of the study's silver
bin (20 equal-count bins per label). Against `teacher-share-025-rerun`: overall ×1 0.930722
(+0.000190), overall ×4 0.936106 (+0.000297, 95% [−0.000499, +0.001041]), gold ×4 0.938511
(−0.000204). The ×4 gain is below +0.0009, so it is not adopted; ×1 and gold pass. It is the
highest ×4 and ×1 of the campaign and the only variant at share 0.25 above the incumbent on both,
by less than the 0.0004 a retraining gives. Training 1,507 s. Its comparison finished as the
campaign stopped; the decision record is written at a next resume.

| Label | Overall ×4 Δ | Gold ×4 Δ |
|---|---:|---:|
| Synovitis | +0.0027 | +0.0000 |
| Lateral OA | +0.0016 | +0.0000 |
| Baker's cyst | +0.0015 | +0.0036 |
| MCL | +0.0012 | −0.0045 |
| Lateral meniscus | +0.0002 | −0.0037 |
| ACL | +0.0001 | +0.0000 |
| Contusion | −0.0002 | +0.0000 |
| Effusion | −0.0002 | −0.0075 |
| Medial OA | −0.0002 | +0.0000 |
| Medial meniscus | −0.0003 | +0.0048 |
| PF OA | −0.0006 | +0.0090 |
| Fracture | −0.0022 | −0.0042 |

The seed predicted the largest gains on synovitis, PF OA, MCL, fracture and lateral OA. Synovitis
(+0.0027), lateral OA (+0.0016) and MCL (+0.0012) rise as predicted; PF OA is flat and fracture
falls (−0.0022), the label it named as the one that would separate it from plain less-teacher.
Fracture has now fallen in five of the six variants at share 0.25. Removing the teacher's pull
toward the middle did not hurt and did not clearly help: the smoothing is not what distillation
gains from, and it is not what cost the higher shares either, at the resolution of one run.

## Campaign closed (2026-10-10 16:07 CEST, by the user's instruction)

**Recipe found:** 0.75 × silver + 0.25 × the raw joint teacher prediction. Incumbent
`teacher-share-025-rerun`: overall ×1 0.930532, overall ×4 0.935809, gold ×4 0.938714.

Answers to the contract's four choices, each from one run per cell on fold 0 (932 studies, 58
gold); two trainings of one recipe differ by 0.0004 at ×4 and 0.0010 on gold:

1. **What the teacher gives:** the joint prediction. The single-plane readings did not help as a
   per-plane target (−0.0006) nor as a source of the share (−0.0002).
2. **How its values are prepared:** raw. Reranked −0.0004, recalibrated +0.0002.
3. **How teacher and silver are mixed:** one share, and less teacher is better over the range run:
   0.25 0.935809, 0.5 0.934171, 0.75 0.932562, 1.0 0.931358 at ×4. A share per label (−0.0004), a
   gate on the size of the disagreement (−0.0002, gold −0.0025) and the residual at the silver
   level (+0.0003) do not beat it.
4. **What the student is trained on:** the joint target only.

**Open:** no run below 0.25 and none with share 0 on this split, so the campaign shows the order of
the shares it ran and not that a teacher beats none; the user removed the 0.125 seed and left the
share-0 control undecided. `teacher-residual-at-silver-level` is the one variant above the incumbent
on ×1 and ×4.

**Counts:** 16 forced seeds (9 of the user's, 4 requeues, 2 from sub-agents at the user's request,
1 keeper seed removed), 11 compared runs, 2 adoptions (the forced first seed and share 0.25), 4
invalid runs from harness and machine causes, all rerun. S134 (share 0.25) and S135 (share 0.75)
submitted at the user's request, still pending on Kaggle at the close.

## Public scores (submitted at the user's request, ×4, τ 0.9, no TTA)

| Submission | Model | Fold-0 ×4 | Gold ×4 | Public |
|---|---|---:|---:|---:|
| S136 | `teacher-share-050` (share 0.5) | 0.934171 | 0.938740 | 0.953 |
| S134 | `teacher-share-025-rerun` (share 0.25) | 0.935809 | 0.938714 | pending since 11:02 CEST |
| S135 | `teacher-share-075-rerun` (share 0.75) | 0.932562 | 0.937050 | pending since 11:02 CEST |

S136's notebook sat in Kaggle's queue; re-pushing it every five minutes got version 6 to run
(17:18 CEST) and it was scored within minutes. These students train on folds 1 to 4 of
`gold_holdout_5fold_v1` (3,475 silver studies, no gold study in training), so 0.953 is not
comparable with S129 (0.955), the same ResNet-18 recipe trained on 3,533 studies of
`all_5fold_scanner_v2` with the gold labels in training.

## After the campaign: ResNet-50 teachers and a first ResNet-50 child (2026-10-10, evening)

Outside the campaign, on the user's instruction, on main's split `all_5fold_scanner_v2` and `configs/refined-resnet50.yaml`.

- **Five ResNet-50 teachers** (`outputs/training/rn50-distillation-teachers`, the first ResNet-50 on main's recipe). ×1 at the best epoch and ×4 per fold: 0 0.933188 / 0.940517; 1 0.924407 / 0.933256; 2 0.928762 / 0.937675; 3 0.924309 / 0.931475; 4 0.925093 / 0.934282. Pooled out of fold at ×4: 0.934934. About 39 minutes of training per fold, 14.3 GB peak VRAM.
- **Fold 1 against the earlier runs of the same fold and seed, ×4:** ResNet-18 on main's recipe (`sagittal-fluid-canvas-fine`, S129) 0.928121, so +0.005135 for the ResNet-50; ResNet-50 on the anatomy-map recipe (`rn50-anatomy-map-centre`, S126, 0.959 public) 0.933265, so −0.000009. The recipe changes adopted since S126 gave the ResNet-18 +0.001038 and the ResNet-50 nothing on this fold. The gap between the backbones over three recipes: +0.008142 (run 24), +0.006182 (anatomy map), +0.005135 (main's).
- **Teacher table:** `data/labels/prepared/rn50-distillation-teachers/teacher_table.parquet`, 4,407 studies, ×6, reflection TTA, τ 0.85 (user decision; 0.8 is the value measured on the ResNet-50 in the prior work).
- **Child `rn50-student-share-050`** (fold 1, 0.5 × silver + 0.5 × raw joint teacher, gold studies keep their labels; mean label shift against silver 0.0539): ×4 0.935635, ×1 0.927680, training 2,315 s. Against its fold 1 teacher +0.002379 at ×4 and +0.003273 at ×1; against S126 +0.002370 at ×4. One run. Per label against the teacher at ×4, largest gain to largest loss: MCL +0.0114, Contusion +0.0093, Fracture +0.0058, Lateral meniscus +0.0028, Effusion +0.0025, Medial OA +0.0022, ACL +0.0009, Medial meniscus −0.0003, Lateral OA −0.0004, Baker's cyst −0.0005, PF OA −0.0011, Synovitis −0.0040. Submitted as S138 on 2026-10-11 00:13 CEST (×4, τ 0.9): **0.960 public**, the best submission, against 0.959 for S126 (same fold and seed, no distillation). The share the campaign found best on the ResNet-18, 0.25, is not run on the ResNet-50.
- **Memory across folds, fixed:** `scripts/pipeline/train.py` now runs each fold in its own spawned process, with the loader workers set back to `fork` inside it. Without that last line the workers were spawned too, which cost about 5 GB more RAM and 3.5 GB more swap at the same epoch and filled the swap in the first attempt. With the fix every fold peaked near 31 GB of RAM and swap together and the next one started from about 10 GB used; the five folds ran in one process tree on 40 GB of RAM.
- **Ensemble of the five ResNet-50 teachers** (x6, reflection TTA, tau 0.85, probability average): S139 failed in Kaggle's hidden rerun after about 50 minutes ("unhandled error", no log) with 4 studies per batch and 16 loader workers; the same bundle with 2 studies per batch and 8 workers, S140 on 2026-10-11, scored **0.961 public**, the best submission (S138, the distilled child alone, 0.960; S126 0.959). The cause of the S139 failure is not established; the lighter loader is the only difference.
- **Ensemble-combination study (2026-10-11, `outputs/analysis/e/ensemble-combination-20261011`):** measured on the campaign's four ResNet-18 teachers and its held-out fold 0 (932 studies, 58 gold), all at x6, reflection TTA and tau 0.85; baseline, the probability average, overall 0.937811 and gold 0.939980. Five sub-agents recorded 546 methods, a sixth tried TabPFN. Teacher weights of any kind, other means (logit, rank, geometric, median), per-teacher calibration or alignment, rank fusion and label co-occurrence models gain nothing or lose (the teachers correlate at 0.98). The only extra signal is each label's own six single-view readings used as a small correction to its joint reading: own-label kernel ridge on the Platt residual +0.00115 overall [+0.0005, +0.0019] and +0.0010 gold, cross-validated +0.0003 inside folds 1-4; greedy per-label view weights +0.00097 and +0.0009; a fixed mix at weight 0.1 +0.00024 and +0.0003. Stackers on all 84 logits, series counts or thresholded targets (tree boosters up to +0.0021 overall) gain on the silver studies only and lose on gold (-0.001 to -0.005). TabPFN in the own-label residual form: +0.00055 at its cross-validated strength. Every gold interval spans zero.
- **The correction on the five ResNet-50 teachers:** refitted on their out-of-fold table; cross-validated gain per teacher +0.00026 at strength 0.25 (5 of 5 folds), +0.00011 at 0.5, -0.00135 at 1.0; the fixed view mix +0.00017 to +0.00025. About half the signal the ResNet-18 teachers showed. Submitted as S141 (x6, TTA, tau 0.85, batch 2, strength 0.25): **0.961 public, equal to S140** without it. The stacker is an export option (`--stacker`, `src/knee_abnormality/inference/stacker.py`), uncommitted; scoring took about 32 minutes against about 30.
