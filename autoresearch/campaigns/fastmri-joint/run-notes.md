# fastMRI joint training campaign: run notes

Fold 3, one run per seed, against the incumbent `rn50-joint-fastmri-fold-3` (×1 0.926211, ×4
0.930745) and the plain control (×1 0.925655, ×4 0.931023). Trainings on the rented machine.

## Infrastructure, 2026-10-12

- **Launcher bug (keeper's):** the first launch of `kaggle-only-final-epochs` exited after 30 s
  because the launcher had created the run's training folder on the host. Fixed; the launcher
  also runs from the campaign root now, so a fix reaches every run.
- **Loader-worker segmentation faults:** `confident-pseudo-labels-weigh-more` died at batch 791
  of its fourth epoch (minute 29) and `kaggle-only-final-epochs` at batch 145 of its fifth
  (minute 31); both are recorded invalid and requeued as `-rerun` seeds. The third run went
  through the same data order and elapsed time unharmed, and a per-minute record of its
  processes shows nothing accumulating (35 processes, 185 open files, 148 threads, 73 to 78 GB
  available, no out-of-memory event), so the fault is sporadic and in native code of a worker.
  The by-hand reference run and the pseudo-labelling had none. Unresolved; from the fourth run on
  every job prints the Python stack of a dying worker, the thread-limit variables of the local
  harness are no longer passed to the host, and the launcher restarts a run once when it dies
  early enough for a fresh run to fit.
- Two-run yardstick from these runs: single early epochs of the same recipe differ by up to
  0.0014 at ×1 (the lost run of seed 1 against the incumbent, epochs 1 to 4: +0.0006, +0.0014,
  +0.0009, −0.0015).

## Run 1: `batch-norm-statistics-from-kaggle` (rejected by the runtime, adopted by the user)

fastMRI batches normalise with their own statistics and do not update the running ones.
**×1 0.926095 against 0.926211 (−0.000116); ×4 0.931370 against 0.930745 (+0.000626, 80% interval
+0.0001 to +0.0012).** Below at ×1, so not adopted. ×4 soft macro +0.0003. Training 4,275 s
against 4,533 s, prediction 77 s, peak GPU memory 16.5 GB. Against the plain control: ×1 +0.0004,
×4 +0.0003.

×1 by epoch, candidate minus incumbent: +0.0005, +0.0011, +0.0005, +0.0001, +0.0012, −0.0003,
+0.0012, +0.0002, +0.0005, −0.0001.

×4 by label: Lateral OA +0.0035, ACL +0.0018, Fracture +0.0015, MCL +0.0014, Lateral Meniscus
+0.0009, Medial Meniscus +0.0002, Contusion +0.0001, Effusion +0.0001, Baker's −0.0001,
Synovitis −0.0001, Medial OA −0.0003, PF OA −0.0014.

Reading: the step from ×1 to ×4 grew (+0.0053 against the incumbent's +0.0045), which is where
the seed expected its effect, and MCL and Fracture, two of the four labels the first joint run
lost on, came back. PF OA did not. Keeping fastMRI out of the running statistics is one of the
seeds that pulls the model toward Kaggle; the user wants such changes checked publicly before
they are trusted.

**Adopted by the user on 2026-10-12 23:55 CEST** after the runtime's rejection: the ×1 miss of 0.0001 is
inside the rerun spread and ×4 is the submitted setting. The request was filed while
`fastmri-weight-follows-learning-rate` trained on the first incumbent; the supervisor applies it when
no training runs. It has no public score.

## Run 2: `fastmri-weight-follows-learning-rate` (rejected by the runtime on ×1)

The fastMRI batch's loss is scaled by the learning rate over its peak, on the first incumbent.
**×4 0.932544 against 0.930745 (+0.001799, 80% interval +0.0011 to +0.0025); ×1 0.926148 against
0.926211 (−0.000063).** Rejected because it is 0.0001 below at ×1. ×4 soft macro +0.0012. Training
4,292 s, prediction 77 s, peak GPU memory 15.3 GB. No crash. Against the plain control: ×1 +0.0005,
×4 +0.0015. Against the incumbent the user adopted afterwards (run 1): ×1 +0.0001, ×4 +0.0012.

×1 by epoch, candidate minus first incumbent: +0.0007, +0.0019, +0.0023, +0.0002, +0.0015, +0.0011,
+0.0013, +0.0007, +0.0002, −0.0001.

×4 by label: Synovitis +0.0049, Lateral Meniscus +0.0040, Lateral OA +0.0037, ACL +0.0035, MCL
+0.0026, Contusion +0.0010, Baker's +0.0010, Effusion +0.0007, Medial Meniscus +0.0005, PF OA
+0.0002, Medial OA −0.0001, Fracture −0.0005.

Reading: the largest ×4 gain of the campaign and the first whose 80% interval excludes zero by a
margin; ten labels up and none down by more than 0.0005. It is ahead at ×1 in nine of ten epochs
and loses the last by 0.0001, as run 1 did, so the ×1 side of the rule is deciding on differences
inside the rerun spread. It fades fastMRI in the second half, so it is one of the seeds the user
wants checked publicly. Keeper's follow-up: `fastmri-weight-follows-learning-rate-on-incumbent`
applies it to the adopted incumbent.

## Run 3: `kaggle-only-final-epochs-rerun` (meets the rule)

On the incumbent adopted from run 1 (×1 0.926095, ×4 0.931370): epochs 0 to 6 paired, epochs 7 to
9 on Kaggle alone. **×4 0.931686 against 0.931370 (+0.000316, 80% interval −0.0004 to +0.0011);
×1 0.926177 against 0.926095 (+0.000082), selected at epoch 8, the first Kaggle-only epoch.**
Above at both, so it meets the runtime's rule. ×4 soft macro −0.0003. Training 3,559 s (the
Kaggle-only epochs take about two minutes each, since the loader no longer waits on fastMRI
decoding), 160 s above the campaign's lower limit; prediction 77 s; peak GPU memory 15.3 GB. No
crash. Against the first incumbent: ×4 +0.0009. Against the plain control: ×4 +0.0007.

×1 by epoch, candidate minus incumbent: +0.0007, −0.0006, −0.0001, −0.0008, −0.0004, +0.0007,
−0.0001, then +0.0006, +0.0002, −0.0001 in the Kaggle-only epochs.

×4 by label: PF OA +0.0053, Fracture +0.0039, Lateral OA +0.0020, MCL +0.0015, Synovitis +0.0008,
Effusion +0.0007, Medial Meniscus +0.0001, Medial OA 0.0000, Lateral Meniscus −0.0006, Baker's
−0.0022, ACL −0.0034, Contusion −0.0043.

Reading: a macro change the size of the rerun spread made of large moves in both directions. PF
OA, the label the joint training lost most on, comes back by +0.0053, with Fracture and MCL;
Contusion, ACL and Baker's, three labels the joint training had gained on, give part of it back.
That is the pattern of moving back toward the Kaggle-only model, which is the user's concern
about this family of seeds: it needs a public check before it is trusted. The ×1 score peaks in
the first Kaggle-only epoch and falls in the next two.

Run 3 was adopted by the runtime at 02:11 CEST on 2026-10-13: the incumbent is batch-norm
statistics from Kaggle plus Kaggle-only final epochs, ×1 0.926178, ×4 0.931686.

## Run 4: `fastmri-weight-follows-learning-rate-on-incumbent` (rejected)

The scheduled fastMRI weight of run 2 applied to the incumbent of run 3, so it acts in the seven
paired epochs only. **×4 0.931456 against 0.931686 (−0.000231, 80% interval −0.0010 to +0.0006);
×1 0.925971 against 0.926178 (−0.000207).** Below at both. ×4 soft macro +0.0003. Training 3,588 s,
prediction 82 s, peak GPU memory 15.3 GB. No crash.

×1 by epoch, candidate minus incumbent: −0.0018, +0.0016, +0.0011, +0.0014, −0.0003, −0.0006,
+0.0005, then −0.0012, −0.0001, −0.0003 in the Kaggle-only epochs.

×4 by label: ACL +0.0036, Contusion +0.0023, Lateral Meniscus +0.0013, Effusion +0.0013, Medial
Meniscus +0.0012, Synovitis +0.0002, PF OA −0.0002, Baker's −0.0003, Medial OA −0.0005, MCL
−0.0018, Lateral OA −0.0035, Fracture −0.0064.

Reading: the +0.0018 that the scheduled weight gave on the first incumbent does not add to the
incumbent's two changes. The three mechanisms (statistics from Kaggle, Kaggle-only final epochs,
a fading fastMRI weight) all reduce fastMRI's part late in training, so they overlap; candidate
cause for the missing gain. The labels this run moves are the mirror of run 3: it gives back
Fracture, Lateral OA and MCL, which run 3 had gained, and recovers ACL and Contusion, which run
3 had lost.

## ×4 of every completed run so far

| Recipe | ×1 | ×4 |
|---|---:|---:|
| Plain control, no fastMRI | 0.925655 | 0.931023 |
| Joint, first incumbent (S148, 0.962 public) | 0.926211 | 0.930745 |
| + batch-norm statistics from Kaggle (run 1) | 0.926095 | 0.931370 |
| + scheduled fastMRI weight, on the first incumbent (run 2) | 0.926148 | **0.932544** |
| + statistics from Kaggle + Kaggle-only final epochs (run 3, incumbent) | 0.926178 | 0.931686 |
| + all three (run 4) | 0.925971 | 0.931456 |

The best ×4 of the campaign is run 2, which the runtime rejected by 0.0001 at ×1 and which is
not on the adopted path.

## Run 5: `confident-pseudo-labels-weigh-more-rerun` (rejected)

On the incumbent of run 3: a fastMRI cell weighs by how far its pseudo-label is from 0.5,
normalised per label. **×4 0.931689 against 0.931686 (+0.000003, 80% interval −0.0008 to +0.0008);
×1 0.925453 against 0.926178 (−0.000725).** Below at ×1 by more than the earlier near-misses, and
behind in eight of ten epochs. ×4 soft macro +0.0003. Training 3,598 s, prediction 84 s, peak GPU
memory 15.3 GB. No crash.

×1 by epoch, candidate minus incumbent: −0.0012, −0.0018, −0.0001, +0.0009, +0.0006, −0.0015,
−0.0007, then −0.0008, −0.0006, −0.0012 in the Kaggle-only epochs.

×4 by label: Contusion +0.0044, Lateral Meniscus +0.0041, ACL +0.0039, Baker's +0.0024, Effusion
+0.0003, Synovitis −0.0002, PF OA −0.0012, Medial OA −0.0014, Medial Meniscus −0.0023, MCL −0.0026,
Fracture −0.0031, Lateral OA −0.0042.

Reading: level at ×4 and worse at ×1. Its target was PF OA, the label with the least confident
pseudo-labels, and PF OA went down (−0.0012), so the mechanism did not do what it was proposed
for. The per-label pattern is again the mirror of run 3: Contusion, ACL and Baker's back up,
Fracture, Lateral OA and MCL back down. Runs 3, 4 and 5 together show two groups of labels
trading against each other at an unchanged macro: (Contusion, ACL, Baker's, Lateral Meniscus)
against (Fracture, Lateral OA, MCL, PF OA). The first run of this seed, on the first incumbent,
was level with it at epoch 3 when it crashed.

## Run 6: `class-weights-over-both-datasets` (rejected)

On the incumbent of run 3: the class weights are computed from the Kaggle and the fastMRI
training rows together. **×4 0.931484 against 0.931686 (−0.000202, 80% interval −0.0011 to
+0.0007); ×1 0.925505 against 0.926178 (−0.000672).** Below at both. ×4 soft macro +0.0007.
Training 3,569 s, prediction 76 s, peak GPU memory 15.3 GB. No crash.

×1 by epoch, candidate minus incumbent: −0.0006, −0.0012, +0.0003, −0.0001, +0.0001, −0.0009,
−0.0010, then −0.0007, −0.0006, −0.0005 in the Kaggle-only epochs.

×4 by label: Contusion +0.0050, ACL +0.0045, Lateral Meniscus +0.0028, Effusion +0.0009, Medial
Meniscus +0.0004, Medial OA 0.0000, Synovitis 0.0000, PF OA −0.0002, Baker's −0.0007, Lateral OA
−0.0027, Fracture −0.0060, MCL −0.0064.

## What runs 4, 5 and 6 share

Three unrelated changes (a scheduled fastMRI weight, confidence weights on fastMRI cells, class
weights over both datasets), each on the incumbent of run 3, give nearly the same per-label
differences against it at ×4:

| Label | Run 4 | Run 5 | Run 6 |
|---|---:|---:|---:|
| ACL | +0.0036 | +0.0039 | +0.0045 |
| Contusion | +0.0023 | +0.0044 | +0.0050 |
| Lateral Meniscus | +0.0013 | +0.0041 | +0.0028 |
| Lateral OA | −0.0035 | −0.0042 | −0.0027 |
| MCL | −0.0018 | −0.0026 | −0.0064 |
| Fracture | −0.0064 | −0.0031 | −0.0060 |
| Macro | −0.0002 | 0.0000 | −0.0002 |

Their ×1 curves are also below the incumbent's in the Kaggle-only epochs every time (−0.0012 to
−0.0005 at epoch 8, where the incumbent jumped by +0.0014 in one epoch). What the three share is
the reference they are measured against, and that each changes the size of the fastMRI loss in
the paired epochs. Candidate causes, not separated yet: (a) any change to the fastMRI loss scale
shifts the same balance between these label groups; (b) the incumbent's per-label profile and its
epoch-8 jump belong to that one training and a second training of the same recipe would not
repeat them, in which case every candidate is being compared with a profile it cannot match. A
second training of the incumbent's recipe would separate the two; it is a control run and the
user's to order.

## Incumbent changed by the user, 2026-10-13 06:06 CEST

Message on the monitor page: "Adopt fast mri follows learning rate instead of kaggle only final
epochs". The keeper filed the adoption of run 2
(`fastmri-weight-follows-learning-rate-20261012T214811Z-76841b3f`, ×1 0.926148, ×4 0.932544); the
supervisor applies it when no training runs. That run was trained on the first incumbent, so the
new incumbent is the joint recipe with the scheduled fastMRI weight only: it does not carry
batch-norm statistics from Kaggle (run 1) or the Kaggle-only final epochs (run 3). Runs 4, 5 and 6
were measured against run 3's incumbent; `no-loader-augmentation-on-fastmri` was training on it
when the change was filed.

## Run 7: `no-loader-augmentation-on-fastmri` (rejected)

On the incumbent of run 3: fastMRI studies load without contrast mixing, intensity and noise
augmentation. **×4 0.931334 against 0.931686 (−0.000352, 80% interval −0.0014 to +0.0007); ×1
0.926511 against 0.926178 (+0.000334), the highest ×1 of the campaign.** Above at ×1, below at ×4.
×4 soft macro 0.0000. Training 3,491 s, 90 s above the lower limit; prediction 79 s. No crash.

×1 by epoch, candidate minus incumbent: −0.0010, −0.0019, +0.0010, +0.0042, −0.0012, −0.0005,
−0.0003, then −0.0007, −0.0004, +0.0005 in the Kaggle-only epochs. The +0.0042 at epoch 4 is the
incumbent's dip there, which this run did not have.

×4 by label: ACL +0.0039, Lateral Meniscus +0.0019, Medial Meniscus +0.0017, Synovitis +0.0017,
Contusion +0.0012, Effusion +0.0004, Baker's +0.0001, Medial OA −0.0007, PF OA −0.0019, Lateral OA
−0.0032, Fracture −0.0044, MCL −0.0047.

Reading: the fourth candidate on run 3's incumbent with the same per-label differences (ACL up,
Fracture, MCL and Lateral OA down) and a ×4 within 0.0004 of it. Four unrelated changes giving
one pattern points at the reference they share: the ×4 of that recipe is about 0.9313 to 0.9317
whatever is changed on top, and its Fracture, MCL and Lateral OA were high in that one training.
The step from ×1 to ×4 of this run is the smallest of the campaign (+0.0048 against +0.0055 for
the incumbent).

The user's adoption of run 2 was applied at 06:24 CEST: the incumbent is
`fastmri-weight-follows-learning-rate` (×1 0.926148, ×4 0.932544).

## Run 8: `label-rate-matching-by-sampling` (rejected by the runtime on ×4)

On the scheduled-weight incumbent (×1 0.926148, ×4 0.932544): fastMRI studies are drawn with
probabilities fitted so that the twelve positive rates among drawn studies equal the Kaggle
fold's, with replacement. **×1 0.927656 against 0.926148 (+0.001508), the highest ×1 of the
campaign; ×4 0.932420 against 0.932544 (−0.000123, 80% interval −0.0011 to +0.0008).** Rejected
because it is 0.0001 below at ×4. ×4 soft macro −0.0006. Training 4,280 s, prediction 81 s, peak
GPU memory 15.4 GB. No crash.

Sampling: the fit converged in 17 iterations to all twelve rates (for example ACL 0.149 against
0.091 under uniform drawing, PF OA 0.331, Fracture 0.068); effective sample size 7,660 of 9,828
studies; 9,066 distinct studies planned over the ten epochs, 2,888 distinct of 3,528 draws in the
first epoch. Coverage was not the cost.

×1 by epoch, candidate minus incumbent: −0.0016, −0.0039, −0.0012, +0.0041, −0.0011, −0.0006,
+0.0008, +0.0009, +0.0019, +0.0015.

×4 by label: Fracture +0.0042, PF OA +0.0029, Medial Meniscus +0.0009, Effusion +0.0004, Lateral OA
−0.0002, Contusion −0.0004, Lateral Meniscus −0.0004, Medial OA −0.0004, Synovitis −0.0011, MCL
−0.0014, Baker's −0.0023, ACL −0.0036.

Reading: the mirror of the earlier near-misses. Runs 1 and 2 gained at ×4 and missed ×1 by
0.0001; this one gains 0.0015 at ×1 and misses ×4 by 0.0001. Its step from ×1 to ×4 is +0.0048
against the incumbent's +0.0064, so what it gains with one image set per study it does not keep
when four are averaged. PF OA, the label joint training lost most on, gains +0.0029 with
Fracture; ACL and Baker's give some back. The epoch-4 dip that every run sharing the incumbent's
loader draws shows is absent here and in run 7, the two runs that change what the loader
delivers, so that dip belongs to the data served in that epoch under the shared seed.

## Run 9: `hard-pseudo-labels` (rejected)

On the scheduled-weight incumbent: fastMRI targets become 0 or 1 at 0.5; the candidate's change
was checked and touches only the fastMRI targets. **×4 0.928935 against 0.932544 (−0.003609, 80%
interval −0.0048 to −0.0024); ×1 0.923158 against 0.926148 (−0.002990).** Below the plain control
without fastMRI at both (×1 0.925655, ×4 0.931023). ×4 soft macro −0.0036. Training 4,417 s,
prediction 81 s. No crash.

×1 by epoch, candidate minus incumbent: −0.0076, −0.0078, −0.0097, −0.0033, −0.0070, −0.0059,
−0.0044, −0.0040, −0.0025, −0.0030. The gap closes as the scheduled fastMRI weight fades.

×4 by label: PF OA +0.0024, Lateral Meniscus −0.0004, Contusion −0.0007, ACL −0.0007, Medial OA
−0.0009, Medial Meniscus −0.0016, Baker's −0.0020, Fracture −0.0025, Effusion −0.0026, Lateral OA
−0.0078, MCL −0.0105, Synovitis −0.0161.

Reading: the first large, unambiguous effect of the campaign, ten times the rerun spread and
with an interval that excludes zero. The teacher's graded probabilities carry information the
student needs; with thresholded targets the joint training is worse than no fastMRI. The labels
hit hardest are Synovitis, MCL and Lateral OA; Synovitis and MCL have few confident positives, so
thresholding turns most of their evidence into plain negatives. PF OA alone improves (+0.0024):
it is the label whose pseudo-labels sit closest to 0.5 (mean 0.48), the one case where soft
targets say least. Lesson for later seeds: keep the pseudo-labels soft; anything that sharpens
them should be expected to cost.

## Public check of the incumbent, 2026-10-13

By the user's order the incumbent of run 2 was submitted with S148's settings (×4, τ 0.9, batch 4,
no mirror TTA):

| Submission | Recipe | Fold ×1 | Fold ×4 | Public |
|---|---|---:|---:|---:|
| S148 | joint, first incumbent | 0.926211 | 0.930745 | 0.962 |
| S149 | + fastMRI weight following the learning rate | 0.926148 | 0.932544 | 0.961 |

A fold gain of +0.0018 at ×4, the largest of the campaign, came with −0.001 on the public board.
One pair of submissions on a board shown to three decimals: weak evidence on its own, but it is
the direction the user had warned about, since the scheduled weight removes fastMRI from the late
epochs. The fold contains only sites seen in training, so it cannot see what fastMRI adds for
unseen ones; the same was recorded for L2-SP. Until a way to measure that locally exists, a fold
gain from reducing fastMRI's part is not evidence of a better submission.

## Reset and new direction, 2026-10-13 09:35 CEST

By the user's decision after S149: the campaign was stopped, the run in progress
(`batch-norm-statistics-from-kaggle-on-incumbent`, at epoch 3 and 0.0005 below the scheduled-weight
incumbent) was cancelled, and the incumbent was reset to the starting reference (decision record
`keeper-incumbent-reset-20261013T0730Z`). Three forced seeds were imported that keep or increase
fastMRI's part on the reference ([seventh seeds](seed-ideas-7.yaml)): the fastMRI batch at twice
the weight, no loader augmentation on fastMRI, and label-rate matching by sampling. `l2-sp-off`
stays queued. The contract carries the direction; the sealed role instructions could not be
changed, so the researcher receives it through the recorded finding and the seeds' text, and the
keeper removes any proposal that reduces fastMRI's part.

Known gap found on the way: cancelling a run locally does not end it on the host at once, because
the launcher is killed before it can signal; the host's guard ends it when the heartbeat lapses
(five minutes), or immediately when the keeper touches the run's `cancel` file.

## Run 10: `l2-sp-off` (met the fold rule; not adopted after its public check)

On the reference: both anchor strengths 0, so the backbone is no longer pulled to its pretrained
weights and takes the recipe's weight decay. **×4 0.932550 against 0.930745 (+0.001806, 80%
interval +0.0012 to +0.0025); ×1 0.927625 against 0.926211 (+0.001414), ahead in all ten epochs.**
×4 soft macro +0.0011. Training 4,350 s, prediction 75 s. No crash.

×1 by epoch, candidate minus reference: +0.0006, +0.0001, +0.0017, +0.0021, +0.0017, +0.0012,
+0.0019, +0.0013, +0.0014, +0.0014.

×4 by label: ACL +0.0041, Synovitis +0.0032, Lateral Meniscus +0.0029, Contusion +0.0029, Effusion
+0.0027, Lateral OA +0.0027, Medial Meniscus +0.0018, Medial OA +0.0014, PF OA +0.0007, Fracture
+0.0002, Baker's −0.0003, MCL −0.0007.

**Public check, by the user's order before any adoption:** `S150 | rn50 f3 joint fastmri l2-sp
off` (×4, τ 0.9, batch 4, no mirror TTA) scored **0.961** and ranks second in the list sorted by
public score, below S148 (0.962) and above S140, S141 and S149 (all 0.961). Not adopted; the
keeper recorded the rejection by hand while the campaign was stopped, so the runtime's fold rule
did not apply it. L2-SP stays.

| Submission | Recipe | Fold ×1 | Fold ×4 | Public | Rank by score |
|---|---|---:|---:|---:|---:|
| S148 | joint, reference | 0.926211 | 0.930745 | 0.962 | 1 |
| S150 | + L2-SP off | 0.927625 | 0.932550 | 0.961 | 2 |
| S149 | + scheduled fastMRI weight | 0.926148 | 0.932544 | 0.961 | 5 |

Reading: the cleanest fold gain of the campaign, ten labels up and ahead in every epoch, and the
public board still puts it below the reference. That is twice a fold gain of +0.0018 at ×4 with a
lower public score. It is what the record of L2-SP predicted (level or better on seen sites,
worse on unseen ones), and fastMRI's scanners, 81% Siemens with almost no Philips, Toshiba or
Canon, do not replace it. The fold cannot rank candidates for this competition's hidden set at
the 0.002 level.

Two submissions carry the number S150: another session's `rn50 f2 soft-prevalence class weights`
was submitted 17 minutes before this one.
