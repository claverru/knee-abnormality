# Run notes

## Start, 2026-10-13

- **23:03 CEST, calibration failed in 3 s:** `transformers` 4.50.0 beside the locked
  `huggingface-hub` 1.30 raised on import. Removed `transformers` (MambaVision-T does not need it)
  and resumed at 23:05; calibration passed.
- **23:18, monitor page:** registered with the fastMRI worktree's `monitor.sh`, by the user's
  order.

## Queue order changed by the keeper, 2026-10-13 23:22

`anchors-by-weight-scale` moved from fourth to directly behind the already prepared
`backbone-rate-2e-4`, ahead of `backbone-rate-4e-4`.

Reason, from the first two epochs of `stage-rates-by-weight-scale` against the reference:

| | Stage rates, epoch 1 | Epoch 2 | Reference, epoch 1 | Epoch 2 |
|---|---:|---:|---:|---:|
| Validation AUC | 0.8153 | 0.8567 | 0.8305 | 0.8759 |
| Training loss | 0.949 | 0.799 | 0.934 | 0.705 |
| Anchor penalty | 0.040 | 0.079 | 0.005 | 0.018 |
| Drift, last stage | 0.0094 | 0.0071 | 0.0051 | 0.0053 |

With twelve times the rate on the last stage its drift rose by a third and was already shrinking
in epoch 2, while the anchor penalty was four to eight times the reference's. The anchor holds the
weights; a higher rate against it adds movement the anchor undoes. The rate seeds are therefore
read on an incumbent that the anchor seed may change, and that seed should run before the second
of them.

## Researcher instructions amended, 2026-10-13 23:56

By the user's order ("make sure the researcher takes into account past experiments when proposing
new: curves, performance per label, etc"): the researcher reads every decided run and the
reference before proposing, tabulates per run the curves, diagnostics and per-label changes, reads
across runs, and names for each idea the runs, curve features and labels it rests on. Applied to
`campaign.yaml` and the sealed policy together while the campaign was stopped (about one minute,
after the first training ended); decision recorded under the campaign's `decisions/`.

## Run 1, `stage-rates-by-weight-scale`: rejected

Rates 6e-4 on the last stage, 0.6 times less per stage. ×4 0.919368 against 0.922130
(−0.002762, 80% interval [−0.004675, −0.000919]); ×1 0.913465 against 0.915009 (−0.001544).
Against the ResNet-50 −0.020266; equal blend with it 0.936056 (−0.003578 against the ResNet-50
alone). Training 2,604 s, prediction 63 s, peak 12.7 GB.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation AUC | 0.8153 | 0.8567 | 0.8704 | 0.8776 | 0.8877 | 0.8924 | 0.9009 | 0.9092 | 0.9123 | 0.9135 |
| Reference | 0.8305 | 0.8759 | 0.8907 | 0.9012 | 0.9028 | 0.9061 | 0.9113 | 0.9141 | 0.9142 | 0.9150 |
| Training loss | 0.949 | 0.799 | 0.723 | 0.706 | 0.684 | 0.663 | 0.637 | 0.624 | 0.614 | 0.603 |
| Reference loss | 0.934 | 0.705 | 0.658 | 0.641 | 0.629 | 0.618 | 0.605 | 0.600 | 0.599 | 0.593 |
| Anchor penalty | 0.040 | 0.079 | 0.059 | 0.055 | 0.046 | 0.039 | 0.030 | 0.024 | 0.020 | 0.018 |
| Drift, last stage | 0.0094 | 0.0071 | 0.0081 | 0.0072 | 0.0063 | 0.0054 | 0.0045 | 0.0035 | 0.0029 | 0.0026 |

Per label at ×4 against the reference, largest gain to largest loss: Baker's +0.0036, Synovitis
+0.0013, Medial OA +0.0010, Lateral OA +0.0007, Effusion −0.0002, MCL −0.0006, Medial Meniscus
−0.0012, Lateral Meniscus −0.0026, ACL −0.0026, Contusion −0.0079, PF OA −0.0112, Fracture −0.0135.

**What it shows:** the higher rates did not buy movement. The last stage ended closer to its
pretrained weights (0.26%) than in the reference (0.36%), with the same final anchor penalty
(0.018): while the rate was high the anchor penalty was two to four times the reference's, and as
the rate decayed the anchor pulled the weights back. Training loss stayed above the reference's
at every epoch. The validation curve gained most late (epochs 7 to 9), when the rate was low.
The run measures a high rate against the stiff anchor, not the rate itself; the three labels that
lost most are the ones the reference already lost most on (Fracture, PF OA) plus Contusion, its
one lead over the ResNet-50.

**Next:** `backbone-rate-2e-4` is training; `anchors-by-weight-scale` follows. If the anchor seed
frees the weights, the stage-wise rates deserve a second run on that incumbent.

## Run 2, `backbone-rate-2e-4`: accepted, new incumbent (2026-10-14 00:44)

One backbone rate of 2e-4. ×4 0.924571 against 0.922130 (+0.002441); ×1 0.919012 against
0.915009 (+0.004003). Against the ResNet-50 −0.015062; equal blend with it 0.936867 (−0.002767
against the ResNet-50 alone; the reference's blend was −0.003199). Training 2,609 s, peak 12.7 GB.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation AUC | 0.8163 | 0.8687 | 0.8900 | 0.8962 | 0.9012 | 0.9065 | 0.9130 | 0.9175 | 0.9187 | 0.9190 |
| Reference | 0.8305 | 0.8759 | 0.8907 | 0.9012 | 0.9028 | 0.9061 | 0.9113 | 0.9141 | 0.9142 | 0.9150 |
| Training loss | 0.928 | 0.741 | 0.687 | 0.668 | 0.653 | 0.637 | 0.617 | 0.607 | 0.602 | 0.593 |
| Reference loss | 0.934 | 0.705 | 0.658 | 0.641 | 0.629 | 0.618 | 0.605 | 0.600 | 0.599 | 0.593 |
| Anchor penalty | 0.023 | 0.050 | 0.044 | 0.039 | 0.035 | 0.030 | 0.025 | 0.021 | 0.019 | 0.018 |
| Drift, last stage | 0.0083 | 0.0063 | 0.0059 | 0.0053 | 0.0052 | 0.0046 | 0.0040 | 0.0035 | 0.0032 | 0.0031 |
| Gradient norm | 4.67 | 1.27 | 1.36 | 1.61 | 1.05 | 1.60 | 1.56 | 1.70 | 1.48 | 1.63 |

Per label at ×4 against the reference, largest gain to largest loss: Lateral OA +0.0124, PF OA
+0.0058, Medial OA +0.0057, ACL +0.0055, Lateral Meniscus +0.0050, Baker's +0.0039, Medial
Meniscus +0.0017, MCL +0.0017, Synovitis +0.0012, Effusion +0.0006, Contusion −0.0066, Fracture
−0.0075.

**What it shows:**

- Behind the reference until epoch 5, ahead from epoch 6, with the lead growing as the rate decays
  (+0.0004, +0.0017, +0.0033, +0.0045, +0.0040).
- The training loss ends level with the reference's (0.593) and the last stage again ends near
  its pretrained weights (0.31% against 0.36%): the gain did not come from the weights ending
  further away. The anchor still pulls them back by the end.
- The gradient norm is 1.0 to 1.7 from epoch 2, against 4 to 9 in the reference and 2.4 to 4.2 in
  run 1. The stem moved twice as far in epoch 1 (drift 0.021 against 0.008) and the large stem
  gradients went with it; the clip at 1.0 now binds lightly, which weakens the case for
  `gradient-clipping-at-10` on this incumbent.
- Both rate runs lose on Contusion (−0.0066, −0.0079) and Fracture (−0.0075, −0.0135), the two
  bone-signal labels; run 1 also lost PF OA, which this run gained.

**Next:** `anchors-by-weight-scale` on this incumbent, then `backbone-rate-4e-4`.

## Run 3, `anchors-by-weight-scale`: accepted, new incumbent (2026-10-14 01:33)

Anchors 0.001 early and 0.00001 late on the 2e-4 incumbent. ×4 0.934657 against 0.924571
(+0.010086); ×1 0.928335 against 0.919012 (+0.009323). Against the ResNet-50 −0.004977 at ×4 and
−0.002567 at ×1; equal blend with it 0.940011, +0.000377 over the ResNet-50 alone, the first
blend of this line above it. Training 2,593 s, peak 12.7 GB.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation AUC | 0.8414 | 0.8879 | 0.9083 | 0.9107 | 0.9147 | 0.9207 | 0.9240 | 0.9268 | 0.9281 | 0.9283 |
| Run 2 | 0.8163 | 0.8687 | 0.8900 | 0.8962 | 0.9012 | 0.9065 | 0.9130 | 0.9175 | 0.9187 | 0.9190 |
| ResNet-50 | 0.8954 | 0.9105 | 0.9129 | 0.9164 | 0.9214 | 0.9264 | 0.9274 | 0.9297 | 0.9304 | 0.9309 |
| Training loss | 0.896 | 0.676 | 0.623 | 0.607 | 0.598 | 0.586 | 0.571 | 0.566 | 0.563 | 0.558 |
| Run 2 loss | 0.928 | 0.741 | 0.687 | 0.668 | 0.653 | 0.637 | 0.617 | 0.607 | 0.602 | 0.593 |
| ResNet-50 loss | 0.848 | 0.663 | 0.631 | 0.618 | 0.608 | 0.595 | 0.583 | 0.579 | 0.572 | 0.571 |
| Anchor penalty | 0.0010 | 0.0054 | 0.0076 | 0.0087 | 0.0093 | 0.0095 | 0.0095 | 0.0094 | 0.0093 | 0.0093 |
| Drift, stem | 0.030 | 0.043 | 0.044 | 0.044 | 0.043 | 0.040 | 0.039 | 0.038 | 0.037 | 0.037 |
| Drift, second stage | 0.045 | 0.076 | 0.089 | 0.096 | 0.100 | 0.102 | 0.103 | 0.103 | 0.102 | 0.102 |
| Drift, last stage | 0.026 | 0.034 | 0.037 | 0.037 | 0.037 | 0.037 | 0.037 | 0.037 | 0.036 | 0.036 |
| Gradient norm | 6.81 | 1.10 | 1.07 | 0.85 | 0.77 | 0.92 | 1.34 | 1.37 | 0.93 | 0.95 |

Per label at ×4 against run 2, largest gain first, with the distance to the ResNet-50 in brackets:
PF OA +0.0257 (+0.0016), Fracture +0.0145 (−0.0230), Lateral Meniscus +0.0114 (−0.0061), ACL
+0.0112 (−0.0074), Medial OA +0.0105 (−0.0029), MCL +0.0104 (−0.0081), Effusion +0.0089
(−0.0003), Medial Meniscus +0.0087 (−0.0011), Contusion +0.0059 (+0.0081), Lateral OA +0.0056
(−0.0169), Synovitis +0.0054 (−0.0013), Baker's +0.0029 (−0.0023).

**What it shows:**

- The anchor was the main cause of the underfit. With it at the strength the weight scale gives,
  the weights moved and stayed moved (the drift no longer shrinks as the rate decays), and the
  training loss fell below the ResNet-50's from epoch 3.
- Ahead of run 2 from the first epoch (+0.025) and at every epoch after.
- Every label gained. Fracture and Contusion, which both rate runs lost, gained here: their loss
  belonged to a high rate against the stiff anchor.
- What is left against the ResNet-50 sits on Fracture and Lateral OA; five labels are within 0.003.
- The last stage moved 3.6% against the ResNet-50's 10.6%, the stem 3.7% against 0.8%: one rate
  for every stage is still not the ResNet-50's relative step anywhere but in the middle.
- The gradient norm is about 1 from epoch 2, so the clip at 1.0 is roughly at the norm.
- The anchors exist for unseen sites, which this fold does not measure. The seed kept them, scaled.

## Seed added by the keeper, 2026-10-14 01:36

`stage-rates-on-freed-anchors` ([third seeds](seed-ideas-3.yaml)), forced: the stage-wise rates of
run 1 (6e-4 on the last stage, 0.6 per stage) on the incumbent of run 3. Run 1 measured those
rates against the stiff anchor and could not show what they do; run 3's drift per stage shows the
single rate is too low for the last stage and high for the stem.

## Run 4, `backbone-rate-4e-4`: rejected, level (2026-10-14 02:19)

One backbone rate of 4e-4 on the incumbent of run 3. ×4 0.935009 against 0.934657 (+0.000352,
80% interval [−0.000931, +0.001514]); ×1 0.927992 against 0.928335 (−0.000343): above at ×4, below
at ×1, so not adopted. Against the ResNet-50 −0.004625; equal blend with it 0.940100 (+0.000466
over the ResNet-50 alone; run 3's blend +0.000377).

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation AUC | 0.8367 | 0.8768 | 0.9050 | 0.9089 | 0.9123 | 0.9167 | 0.9223 | 0.9257 | 0.9277 | 0.9280 |
| Run 3 | 0.8414 | 0.8879 | 0.9083 | 0.9107 | 0.9147 | 0.9207 | 0.9240 | 0.9268 | 0.9281 | 0.9283 |
| Training loss | 0.889 | 0.693 | 0.643 | 0.625 | 0.614 | 0.600 | 0.585 | 0.577 | 0.573 | 0.565 |
| Run 3 loss | 0.896 | 0.676 | 0.623 | 0.607 | 0.598 | 0.586 | 0.571 | 0.566 | 0.563 | 0.558 |
| Anchor penalty | 0.0023 | 0.0104 | 0.0142 | 0.0161 | 0.0170 | 0.0172 | 0.0168 | 0.0164 | 0.0161 | 0.0159 |
| Drift, stem | 0.045 | 0.057 | 0.056 | 0.056 | 0.054 | 0.050 | 0.046 | 0.044 | 0.043 | 0.043 |
| Drift, second stage | 0.072 | 0.120 | 0.140 | 0.151 | 0.157 | 0.159 | 0.159 | 0.158 | 0.157 | 0.157 |
| Drift, last stage | 0.034 | 0.038 | 0.039 | 0.038 | 0.037 | 0.036 | 0.035 | 0.034 | 0.034 | 0.034 |
| Gradient norm | 1.70 | 0.83 | 0.90 | 0.61 | 0.60 | 0.67 | 0.75 | 0.81 | 0.77 | 0.78 |

Per label at ×4 against run 3, largest gain to largest loss: Lateral OA +0.0082, Lateral Meniscus
+0.0030, Medial Meniscus +0.0017, Synovitis +0.0003, ACL +0.0001, Baker's −0.0002, Contusion
−0.0004, PF OA −0.0007, MCL −0.0012, Fracture −0.0019, Medial OA −0.0022, Effusion −0.0027.

**What it shows:**

- Twice the rate ends level: behind run 3 at every epoch (−0.011 at epoch 2, −0.004 at epoch 6),
  closing to −0.0003 as the rate decays. The single rate is near its optimum between 2e-4 and
  4e-4 on these anchors.
- The training loss is higher than run 3's at every epoch from the second (0.565 against 0.558 at
  the end): the higher rate fits less, not more.
- The second stage moved half as far again (15.7% against 10.2%) and the stem more (4.3% against
  3.7%), but the last stage did not (3.4% against 3.6%), with the anchor penalty 1.7 times run
  3's. Doubling the rate did not move the last stage further: its distance is not set by the rate
  at these values.
- Lateral OA gained 0.008 (now −0.0087 from the ResNet-50, from −0.0169); Fracture did not move
  (−0.0248 from the ResNet-50).
- The gradient norm is 0.6 to 0.9, below the clip of 1.0 from epoch 2: `gradient-clipping-at-10`
  would change almost no step on a rate this high, and few on the incumbent's (0.8 to 1.4).

**For `stage-rates-on-freed-anchors`:** its 6e-4 on the last stage is unlikely to move that stage
further by itself, given this run; what it still tests is the lower rate on the stem and first
stage (7.8e-5 and 1.3e-4 against 2e-4) with a higher one above.

## Run 5, `twelve-epochs`: accepted, new incumbent (2026-10-14 03:13)

Twelve epochs on the incumbent of run 3. ×4 0.936187 against 0.934657 (+0.001530, 80% interval
[+0.000437, +0.002629]); ×1 0.930634 against 0.928335 (+0.002300). Against the ResNet-50
−0.003447 at ×4 and −0.000268 at ×1; equal blend with it 0.940656, +0.001022 over the ResNet-50
alone (run 3's blend +0.000377). Training 3,118 s, 1.20 of the reference's.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation AUC | 0.8414 | 0.8844 | 0.9068 | 0.9098 | 0.9174 | 0.9183 | 0.9247 | 0.9264 | 0.9298 | 0.9296 | 0.9303 | 0.9306 |
| Run 3 | 0.8414 | 0.8879 | 0.9083 | 0.9107 | 0.9147 | 0.9207 | 0.9240 | 0.9268 | 0.9281 | 0.9283 | | |
| Training loss | 0.896 | 0.672 | 0.625 | 0.608 | 0.600 | 0.590 | 0.577 | 0.572 | 0.569 | 0.559 | 0.555 | 0.549 |
| Run 3 loss | 0.896 | 0.676 | 0.623 | 0.607 | 0.598 | 0.586 | 0.571 | 0.566 | 0.563 | 0.558 | | |
| Drift, second stage | 0.047 | 0.076 | 0.090 | 0.098 | 0.103 | 0.106 | 0.108 | 0.108 | 0.108 | 0.108 | 0.108 | 0.108 |
| Drift, last stage | 0.026 | 0.033 | 0.036 | 0.037 | 0.038 | 0.038 | 0.038 | 0.038 | 0.037 | 0.037 | 0.037 | 0.037 |

Per label at ×4 against run 3, largest gain first, with the distance to the ResNet-50 in brackets:
Lateral Meniscus +0.0059 (−0.0003), Fracture +0.0050 (−0.0180), Synovitis +0.0036 (+0.0023),
Lateral OA +0.0026 (−0.0143), ACL +0.0025 (−0.0049), Effusion +0.0009 (+0.0006), Medial Meniscus
+0.0002 (−0.0009), Contusion +0.0001 (+0.0083), Medial OA −0.0001 (−0.0031), PF OA −0.0007
(+0.0009), Baker's −0.0007 (−0.0030), MCL −0.0009 (−0.0090).

**What it shows:**

- The two extra epochs add: ×1 was already at run 3's final value by epoch 9 (0.9298) and rose a
  little further; the training loss ended lower (0.549 against 0.558) with the weights no further
  from their start (the drift is flat from epoch 7).
- No label lost more than 0.001. The gains are on Lateral Meniscus, Fracture and Synovitis.
- Against the ResNet-50 the model is ahead on four labels (Contusion, Synovitis, PF OA, Effusion),
  within 0.001 on two more, and behind mainly on Fracture (−0.0180), Lateral OA (−0.0143) and MCL
  (−0.0090).
- The blend is +0.0010 over the ResNet-50 alone, the size of the distance between two ResNet-50
  trainings on this fold.
- Runs from here take about 3,120 s on this incumbent, inside the 3,372 s bound.

## Run 6, `stage-rates-on-freed-anchors`: rejected, level (2026-10-14 04:11)

Rates 6e-4 on the last stage and 0.6 per stage down, on the twelve-epoch incumbent. ×4 0.935863
against 0.936187 (−0.000324, 80% interval [−0.001684, +0.001033]); ×1 0.930656 against 0.930634
(+0.000022). Against the ResNet-50 −0.003771; equal blend with it 0.940368 (+0.000734 over the
ResNet-50 alone; the incumbent's blend +0.001022).

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation AUC | 0.8325 | 0.8802 | 0.8993 | 0.9027 | 0.9138 | 0.9156 | 0.9177 | 0.9255 | 0.9287 | 0.9304 | 0.9307 | 0.9306 |
| Run 5 | 0.8414 | 0.8844 | 0.9068 | 0.9098 | 0.9174 | 0.9183 | 0.9247 | 0.9264 | 0.9298 | 0.9296 | 0.9303 | 0.9306 |
| Training loss | 0.899 | 0.709 | 0.648 | 0.630 | 0.618 | 0.606 | 0.590 | 0.584 | 0.580 | 0.570 | 0.565 | 0.558 |
| Run 5 loss | 0.896 | 0.672 | 0.625 | 0.608 | 0.600 | 0.590 | 0.577 | 0.572 | 0.569 | 0.559 | 0.555 | 0.549 |
| Anchor penalty | 0.0022 | 0.0107 | 0.0147 | 0.0162 | 0.0171 | 0.0173 | 0.0170 | 0.0167 | 0.0163 | 0.0160 | 0.0157 | 0.0157 |
| Drift, stem | 0.013 | 0.023 | 0.026 | 0.027 | 0.028 | 0.028 | 0.028 | 0.027 | 0.027 | 0.027 | 0.027 | 0.027 |
| Drift, second stage | 0.045 | 0.085 | 0.100 | 0.110 | 0.115 | 0.118 | 0.119 | 0.120 | 0.120 | 0.119 | 0.119 | 0.119 |
| Drift, last stage | 0.047 | 0.051 | 0.047 | 0.045 | 0.042 | 0.039 | 0.037 | 0.035 | 0.034 | 0.032 | 0.032 | 0.032 |
| Gradient norm | 9.25 | 2.41 | 2.76 | 2.14 | 1.21 | 1.47 | 2.36 | 2.19 | 1.80 | 2.30 | 1.81 | 1.64 |

Per label at ×4 against run 5, largest gain to largest loss: MCL +0.0079, ACL +0.0037, Baker's
+0.0011, PF OA +0.0006, Medial Meniscus +0.0002, Lateral OA −0.0004, Medial OA −0.0015, Lateral
Meniscus −0.0015, Synovitis −0.0019, Effusion −0.0023, Contusion −0.0031, Fracture −0.0068.

**What it shows:**

- Level at both multipliers. Behind run 5 until epoch 9 (−0.007 at epochs 3, 4 and 7), level from
  epoch 10.
- The last stage moved further early (5.1% at epoch 2 against 3.3%) and came back to 3.2% by the
  end, below run 5's 3.7%, as the rate decayed: the late anchor at 0.00001 still pulls it back.
  Runs 4 and 6 agree that a higher rate does not leave the last stage further from its start.
- The stem moved less (2.7% against 3.8%), as intended, and the training loss ended higher (0.558
  against 0.549).
- The labels split by structure: the two ligaments gained (MCL +0.0079, ACL +0.0037, both now
  within 0.0012 of the ResNet-50) and the bone-signal and fluid labels lost (Fracture −0.0068,
  Contusion −0.0031, Effusion −0.0023). Runs 1 and 2, the other runs that raised the late stages'
  rate against an anchor, also lost Fracture and Contusion.
- The gradient norm is back at 1.2 to 2.8, so the clip at 1.0 binds on most steps in this run.

**Read with runs 1, 2 and 4:** the rate family is closed at one rate of 2e-4. What the higher late
rate changes is which labels win, not the macro. Fracture is the label furthest from the ResNet-50
in every run (−0.018 to −0.044) and drops whenever the late stages are pushed harder.

## Run 7, `feature-statistics-augmentation-off`: rejected (2026-10-14 05:03)

The feature-statistics augmentation off, on the twelve-epoch incumbent. ×4 0.935080 against
0.936187 (−0.001107, 80% interval [−0.002457, +0.000383]); ×1 0.928986 against 0.930634
(−0.001648). Against the ResNet-50 −0.004554; equal blend with it 0.940416 (+0.000783 over the
ResNet-50 alone; the incumbent's +0.001022).

Validation AUC per epoch: 0.8501, 0.8873, 0.9065, 0.9046, 0.9158, 0.9182, 0.9254, 0.9253, 0.9275,
0.9280, 0.9293, 0.9290 (run 5: 0.8414, 0.8844, 0.9068, 0.9098, 0.9174, 0.9183, 0.9247, 0.9264,
0.9298, 0.9296, 0.9303, 0.9306). Training loss ended at 0.549, the same as run 5's; drift and
gradient norm as in run 5.

Per label at ×4 against run 5, largest gain to largest loss: MCL +0.0043, Fracture +0.0021,
Lateral OA +0.0017, Medial Meniscus +0.0009, Lateral Meniscus −0.0005, PF OA −0.0007, ACL −0.0008,
Baker's −0.0016, Medial OA −0.0027, Synovitis −0.0028, Effusion −0.0028, Contusion −0.0102.

**What it shows:**

- Ahead in the first two epochs (+0.0087, +0.0029), level from epoch 3 to 7, behind from epoch 8.
  The augmentation slows the start and helps the end.
- The training loss ends the same with or without it: the measured underfit was not the
  augmentation's doing, and at 2.5 to 2.9 times its ResNet-50 strength it does not cost this
  model its fit.
- Contusion lost most (−0.0102) and leads the ResNet-50 only with the augmentation on. The fluid and
  diffuse labels (Effusion, Synovitis, Baker's) lost; MCL, Fracture and Lateral OA gained a little.
- The augmentation stays on, as it should for unseen scanners.

## Run 8, `gradient-clipping-at-10`: rejected, level (2026-10-14 05:57)

The clip at 10 on the twelve-epoch incumbent. ×4 0.935853 against 0.936187 (−0.000334, 80%
interval [−0.001451, +0.000727]); ×1 0.930383 against 0.930634 (−0.000251). Against the ResNet-50
−0.003781; equal blend with it 0.940829 (+0.001195 over the ResNet-50 alone, the highest blend so
far; the incumbent's +0.001022).

Validation AUC per epoch: 0.8451, 0.8909, 0.9063, 0.9067, 0.9159, 0.9170, 0.9228, 0.9266, 0.9295,
0.9300, 0.9302, 0.9304. Training loss ended at 0.547 (run 5: 0.549). Gradient norm 3.1 in epoch 1,
then 0.8 to 1.3.

Per label at ×4 against run 5, largest gain to largest loss: Synovitis +0.0035, ACL +0.0033,
Fracture +0.0019, PF OA +0.0005, Lateral OA +0.0003, Baker's +0.0003, Lateral Meniscus −0.0000,
Medial OA −0.0003, Effusion −0.0005, Medial Meniscus −0.0010, Contusion −0.0017, MCL −0.0101.

**What it shows:** the clip changes only the first epoch, where the norm is 3 to 5; from epoch 2
the norm sits around the old clip of 1.0. The run is level at both multipliers, ahead in epochs 1
and 2 and level after. MCL lost 0.010, the one label that moved beyond 0.0035; MCL also moved by
+0.008 in run 6 and +0.004 in run 7, runs that changed nothing aimed at it, so it is the most
volatile label of the campaign.

## Forced seeds done, 2026-10-14 06:00

Eight runs, three adoptions (one rate of 2e-4, anchors matched to the weight scale, twelve
epochs). Incumbent `twelve-epochs`: ×4 0.936187, ×1 0.930634; −0.003447 from the ResNet-50 at ×4
and level at ×1 (−0.000268); blend with it +0.001022. The researcher's first refill proposed five
ideas grounded in the runs above; `late-anchor-follows-learning-rate` trains first.

## Run 9, `late-anchor-follows-learning-rate`: adopted (2026-10-14 06:51; marked adopted by the user's order at 09:20 after the keeper had taken it out in error)

The researcher's first idea: the late anchor's strength multiplied by the backbone rate over its
peak, so 0.1 of it in the warm-up's start and about 1% at the end. ×4 0.937634 against 0.936187
(+0.001447, 80% interval [+0.000462, +0.002492]); ×1 0.930689 (selected epoch 11) against
0.930634 (+0.000054). The runtime adopted it.

**Taken back out:** the late anchor is the recipe's site-generalisation device, and this run
removes most of it in the last epochs. Under the user's standing rule (no adoption of a run whose
mechanism could cost site generalisation, even when it passes the local rule; the keeper takes such
a run back out), `twelve-epochs` was re-adopted at 06:52 with `adopt`, rationale recorded. The gain
it showed is small (×1 +0.00005) and one fold cannot show its cost on other sites. The user can
re-adopt it.

Validation AUC per epoch: 0.8443, 0.8916, 0.9071, 0.9108, 0.9167, 0.9190, 0.9237, 0.9281, 0.9297,
0.9298, 0.9307, 0.9303 (run 5: … 0.9298, 0.9296, 0.9303, 0.9306).

## Queue, 2026-10-14 07:11

`late-anchor-decay-with-protected-warmup` (researcher, second refill) removed from the queue by the
keeper under the same rule: it weakens the late anchor to 1% by the end, as run 9 did. The rule is
written into prior-work.md's blocked directions so the researcher stops proposing it.

## Run 10, `twelve-epoch-linear-decay`: accepted, new incumbent (2026-10-14 07:51)

The researcher's idea: twelve epochs with linear decay in place of cosine, same peak rate, warm-up
and floor. ×4 0.937208 against 0.936187 (+0.001020); ×1 0.931826 against 0.930634 (+0.001192).
Against the ResNet-50 −0.002426 at ×4 and +0.000924 at ×1, the first run above it at ×1. Equal
blend with it 0.941156, +0.001522 over the ResNet-50 alone, the best blend of the campaign. No
site-generalisation concern: only the shape of the rate schedule changed.

Per label at ×4 against run 5, largest gain first, with the distance to the ResNet-50 in brackets:
Fracture +0.0041 (−0.0139), Synovitis +0.0036 (+0.0059), Contusion +0.0035 (+0.0118), ACL +0.0027
(−0.0022), PF OA +0.0011 (+0.0021), Lateral OA +0.0008 (−0.0135), Effusion +0.0005 (+0.0011),
Baker's +0.0005 (−0.0025), MCL −0.0002 (−0.0092), Lateral Meniscus −0.0014 (−0.0016), Medial OA
−0.0014 (−0.0044), Medial Meniscus −0.0017 (−0.0026).

Validation AUC per epoch: 0.8344, 0.8882, 0.9103, 0.9103, 0.9176, 0.9202, 0.9232, 0.9271, 0.9308,
0.9299, 0.9306, 0.9318. Behind run 5 in epoch 1 only; ahead or level after, and still rising at
the end where cosine had flattened.

## Run 11, `feature-statistics-after-first-epoch`: rejected (2026-10-14 08:47)

The feature-statistics augmentation skipped in the first epoch only, on the linear-decay incumbent.
×4 0.936888 against 0.937208 (−0.000320, 80% interval [−0.001435, +0.000748]); ×1 0.929495 against
0.931826 (−0.002331). Equal blend with the ResNet-50 0.940960 (+0.001326; the incumbent's
+0.001522).

Validation AUC per epoch: 0.8454, 0.8932, 0.9081, 0.9108, 0.9171, 0.9174, 0.9227, 0.9252, 0.9271,
0.9271, 0.9284, 0.9295. Ahead in epochs 1 and 2 (+0.011, +0.005), behind from epoch 3 and by 0.002
to 0.004 from epoch 8.

Per label at ×4 against the incumbent, largest gain to largest loss: Lateral OA +0.0033, Medial OA
+0.0013, Lateral Meniscus +0.0011, Synovitis +0.0008, Fracture +0.0007, Medial Meniscus +0.0006,
Effusion +0.0005, PF OA +0.0001, ACL −0.0007, Baker's −0.0012, MCL −0.0046, Contusion −0.0057.

**Read with run 7:** both runs without the augmentation for some epochs start faster and finish
lower, and both lose Contusion most (−0.0102, −0.0057). The augmentation's value is in the late
epochs and on Contusion; the faster start does not carry to the end. The family is closed.

## Generalisation rule scoped by the user, 2026-10-14 08:58

The user: the rule that holds out generalisation risks applies "only for models trained on
radimagenet weights". It does not apply to MambaVision-T, so run 9 should not have been taken
back out and `late-anchor-decay-with-protected-warmup` should not have been removed. Since then run
10 (linear decay) was adopted on the cosine incumbent, so run 9 cannot simply be re-adopted
without dropping run 10. The blocked direction in prior-work.md is lifted, and run 9's mechanism
is forced on the current incumbent as `late-anchor-follows-rate-on-linear-decay`
([fourth seeds](seed-ideas-4.yaml)). The researcher may propose the protected-warm-up variant
again.

## Stopped by the user, 2026-10-14 08:58

Forced stop (`stop --abort`): `two-epoch-warmup-on-weight-scale-anchors` was aborted in its second
epoch and has no result. Queued and not run: `gradient-clipping-relaxed-after-warmup` (prepared),
`late-anchor-follows-rate-on-linear-decay` (forced), `linear-backbone-cosine-head-decay`.

| Run | Verdict | ×4 | ×1 | vs ResNet-50 ×4 | Blend vs ResNet-50 |
|---|---|---:|---:|---:|---:|
| Reference (ResNet-50 recipe) | | 0.922130 | 0.915009 | −0.0175 | −0.0032 |
| 1 Stage rates by weight scale | rejected | 0.919368 | 0.913465 | −0.0203 | −0.0036 |
| 2 Rate 2e-4 | adopted | 0.924571 | 0.919012 | −0.0151 | −0.0028 |
| 3 Anchors by weight scale | adopted | 0.934657 | 0.928335 | −0.0050 | +0.0004 |
| 4 Rate 4e-4 | rejected | 0.935009 | 0.927992 | −0.0046 | +0.0005 |
| 5 Twelve epochs | adopted | 0.936187 | 0.930634 | −0.0034 | +0.0010 |
| 6 Stage rates on freed anchors | rejected | 0.935863 | 0.930656 | −0.0038 | +0.0007 |
| 7 Feature statistics off | rejected | 0.935080 | 0.928986 | −0.0046 | +0.0008 |
| 8 Clip at 10 | rejected | 0.935853 | 0.930383 | −0.0038 | +0.0012 |
| 9 Late anchor follows the rate | adopted | 0.937634 | 0.930689 | −0.0020 | |
| 10 Linear decay | adopted, incumbent | 0.937208 | 0.931826 | −0.0024 | +0.0015 |
| 11 Feature statistics after epoch 1 | rejected | 0.936888 | 0.929495 | −0.0027 | +0.0013 |

## Run 9 marked adopted by the user, 2026-10-14 09:20

The user ordered `late-anchor-follows-learning-rate` marked as adopted. The runtime had kept its
verdict as an acceptance throughout; only the incumbent pointer was moved back to `twelve-epochs` by
the keeper's error. The pointer stays on `twelve-epoch-linear-decay`, because run 9 was trained on
`twelve-epochs` without linear decay and re-pointing to it would drop run 10. The adopted lineage
is therefore rate 2e-4, weight-scale anchors, twelve epochs, the late anchor following the rate
and linear decay. That full combination is being trained outside the campaign since 09:13 as
`outputs/training/mambavision-t-combined-12-epochs` (the recipe equals run 10's apart from the
late-anchor schedule), which is also what the queued `late-anchor-follows-rate-on-linear-decay`
would have run.


## All adopted changes together, outside the campaign (2026-10-14 10:06)

`outputs/training/mambavision-t-combined-12-epochs`: rate 2e-4, weight-scale anchors, twelve
epochs, the late anchor following the rate and linear decay (run 10's recipe plus run 9's
schedule). ×4 0.937801 against 0.937208 for run 10 (+0.000593) and 0.937634 for run 9; ×1 0.929880
(epoch 12) against 0.931826 (−0.001946). Against the ResNet-50 −0.001833 at ×4, the closest of the
line; equal blend with it 0.941419, +0.001785 over the ResNet-50 alone, the best blend. Training
3,106 s, peak 12.7 GB.

Validation AUC per epoch: 0.8156, 0.8903, 0.9073, 0.9110, 0.9168, 0.9178, 0.9213, 0.9263, 0.9284,
0.9278, 0.9291, 0.9299 (run 10: 0.8344, 0.8882, 0.9103, 0.9103, 0.9176, 0.9202, 0.9232, 0.9271,
0.9308, 0.9299, 0.9306, 0.9318).

Per label at ×4 against run 10, largest gain to largest loss, with the distance to the ResNet-50
in brackets: Fracture +0.0058 (−0.0081), MCL +0.0039 (−0.0054), Medial Meniscus +0.0023
(−0.0003), Medial OA +0.0008 (−0.0036), ACL +0.0008 (−0.0014), PF OA +0.0007 (+0.0028), Baker's
+0.0007 (−0.0018), Effusion −0.0004 (+0.0008), Synovitis −0.0008 (+0.0051), Lateral OA −0.0015
(−0.0149), Lateral Meniscus −0.0015 (−0.0031), Contusion −0.0036 (+0.0082).

**What it shows:** above run 10 at ×4 and below at ×1, the same split run 9 showed against its
parent; under the campaign's rule it would not be adopted over run 10. Fracture, the label furthest
from the ResNet-50 all campaign, gained most here (−0.008 from the ResNet-50, from −0.044 at the
start). The blend is the best of the line.

## Full recipe entered as incumbent; new seeds, 2026-10-14 11:55

By the user's order ("the full recipe, just modify the campaign so that it appears in it as done"),
the run with all adopted changes (`outputs/training/mambavision-t-combined-12-epochs`) was entered
into the stopped campaign as experiment
`late-anchor-follows-rate-on-linear-decay-20261014T095230Z-c63b90a0`, under the forced idea it
equals, with its own frozen source (the run's snapshot, checked against the worktree), a copy of
the run, the campaign's comparison against `twelve-epoch-linear-decay` and per-label diagnostics.
The shared rule alone gives `reject` (×4 +0.000593, ×1 −0.001946, 80% interval
[−0.000418, +0.001691]); it is recorded as accepted by user decision and is the incumbent. The
previous state is kept as `state.before-<experiment>.json`, and the decision under `decisions/`.

By the user's order ("remove the existing seeds and add these"), every queued, prepared and
proposed idea was retired, the aborted `two-epoch-warmup-on-weight-scale-anchors` closed as
invalid, and five forced seeds imported ([fifth seeds](seed-ideas-5.yaml)), in this order:
`anchors-per-stage-by-weight-scale`, `stochastic-depth-0-1-on-a-fitted-model`, `two-warm-up-epochs`,
`head-rate-twice`, `early-anchor-follows-learning-rate`. The campaign stays stopped until the user
resumes it.


## Resumed with Codex, 2026-10-14 12:00; one repair

The completion event of the aborted `two-epoch-warmup-on-weight-scale-anchors` was still pending
in the state. The supervisor asked the orchestrator to decide it three times in two minutes and
rejected each answer, because that run's reference is no longer the incumbent. The run was already
closed as invalid, so the keeper stopped the campaign, removed that one event (previous state in
`state.before-stale-event-removal.json`) and resumed at 12:04. The developer then started
`anchors-per-stage-by-weight-scale`.

## Stopped by the user, 2026-10-14 12:20

Stopped while the developer prepared `anchors-per-stage-by-weight-scale`; no new seed trained. The
five forced seeds stay queued on the full-recipe incumbent (×4 0.937801, ×1 0.929880).

## Resumed with Codex, 2026-10-14 14:17; absolute-paths repair

The review sent `anchors-per-stage-by-weight-scale` back twice with `Recipe changed outside campaign
policy: ['paths...']`. Cause: the full-recipe incumbent's `configs/training.yaml` came from the
run's source snapshot with relative paths (`../data`, `labels/...`), which resolve inside each
experiment folder; the harness's earlier sources carry absolute ones. The keeper stopped the
campaign, wrote the absolute paths into the incumbent's and the candidate's `training.yaml` (the
incumbent's resolved config is identical to its run's effective config), refreshed the incumbent
manifest and the recorded source digests, set the candidate back to preparation and resumed
(previous state in `state.before-absolute-paths-repair.json`). It launched at 14:32.

## anchors-per-stage-by-weight-scale: reject (×4 −0.0007, ×1 +0.0022)

Stem 0.0046, stages.0 0.00074, stages.1 3.6e-5, stages.2 2.1e-5, stages.3 6.2e-6 instead of
0.001/0.00001. ×4 0.937090 against 0.937801 (80% interval [−0.0019, +0.0004]); ×1 0.932050 against
0.929880, the best ×1 of the campaign. Ahead of the incumbent at ×1 from epoch 3 on except epoch 5
(level), +0.0016 to +0.0036. Per label at ×4, gain to loss: Lateral Meniscus +0.0051, Lateral OA
+0.0024, Effusion +0.0005, Baker's +0.0005, Medial OA +0.0003, PF OA +0.0002, Synovitis +0.0001,
ACL −0.0027, Medial Meniscus −0.0030, Contusion −0.0031, Fracture −0.0039, MCL −0.0049. Lateral OA,
the label furthest behind the ResNet-50, moved toward it; the ligament, fracture and contusion
labels lost. The ×1 and ×4 split in opposite directions: the per-stage anchors help the single-pass
read and lose under the ×4 bag.

## stochastic-depth-0-1-on-a-fitted-model: reject (×4 −0.0009, ×1 +0.0011)

×4 0.936885 against 0.937801 (80% interval [−0.0025, +0.0006]); ×1 0.931018 against 0.929880.
Behind the incumbent at ×1 only at epochs 8 and 9. Per label at ×4, gain to loss: Lateral Meniscus
+0.0062, PF OA +0.0049, Medial OA +0.0024, Synovitis +0.0005, MCL −0.0001, Effusion −0.0008,
Baker's −0.0011, Lateral OA −0.0019, Medial Meniscus −0.0020, ACL −0.0032, Contusion −0.0039,
Fracture −0.0119. Lateral Meniscus gained most in both runs of the day, Fracture, Contusion and ACL
lost in both.

Second run in a row with ×1 up and ×4 down. The ×4 − ×1 gap: incumbent +0.0079, anchors per stage
+0.0050, stochastic depth +0.0059, the previous incumbent (twelve-epoch-linear-decay) +0.0054. The
incumbent's ×4 lift is the outlier, not the candidates'. Shared by both candidates and not by the
incumbent: trained by the harness (the incumbent was trained outside it, from the worktree).
Candidate causes to check: the incumbent's ×4 lift itself (late-anchor schedule on linear decay, its
only change over twelve-epoch-linear-decay); a difference between the harness's and the direct
run's final ×4 prediction (workers, batch, environment).
Checked: the incumbent's effective config equals the stochastic-depth run's except stochastic depth,
name and output root, and the ×4 prediction took the same time (63.7 s against 63.9 s); same
environment. The prediction settings do not explain the gap; the incumbent's ×4 lift belongs to
its model.

## Early epochs of two-warm-up-epochs (keeper's look, user's question at epoch 10)

The +0.025 at epoch 1 is against the incumbent's epoch 1 (0.8156), the lowest of the five runs on
the same schedule (rate 2e-4, one warm-up epoch, linear decay): the others are 0.834–0.845, median
0.8448, with identical first-epoch settings (the incumbent and late-anchor-follows-learning-rate
differ only after the warm-up and score 0.816 and 0.844). Against that median, two-warm-up-epochs
(rate 1.1e-4 at the end of epoch 1) is −0.0045 at epoch 1, −0.0155 at epoch 2 (when it reaches the
peak), −0.006 at epochs 3–4, −0.002 to −0.0016 at epochs 5–7, then −0.0009 to +0.0006 from epoch 8.
The aborted two-epoch warm-up on the older recipe gave 0.834 at epoch 1. The rate 4e-4 run was also
lower early (0.837, 0.877, 0.905 at epochs 1–3) and level at the end. Across runs, an epoch's AUC
falls behind while the rate is at its peak and catches up as it decays; the first epoch's AUC spreads
by 0.03 between runs with identical settings.

## two-warm-up-epochs: reject (×4 −0.0004, ×1 +0.0013)

×4 0.937373 against 0.937801 (80% interval [−0.0017, +0.0008]); ×1 0.931222 against 0.929880.
Per label at ×4, gain to loss: Lateral Meniscus +0.0041, Synovitis +0.0033, Lateral OA +0.0029, PF
OA +0.0011, ACL +0.0006, Baker's +0.0000, Contusion −0.0001, Effusion −0.0011, Medial OA −0.0014,
Medial Meniscus −0.0016, Fracture −0.0048, MCL −0.0082.

Third run in a row above the incumbent at ×1 (0.93205, 0.93102, 0.93122 against 0.92988) and below
it at ×4 (0.93709, 0.93689, 0.93737 against 0.93780). Lateral Meniscus gained in all three (+0.0041
to +0.0062), Fracture lost in all three (−0.0039 to −0.0119), MCL in all three (−0.0001 to −0.0082).
The incumbent pairs the lowest ×1 of the four with the highest ×4.

## head-rate-twice: reject (×4 −0.0011, ×1 +0.0004)

×4 0.936739 against 0.937801 (80% interval [−0.0024, +0.0002]); ×1 0.930309 against 0.929880.
Per label at ×4, gain to loss: Contusion +0.0073, PF OA +0.0024, Lateral OA +0.0013, Synovitis
+0.0011, Effusion +0.0010, Lateral Meniscus +0.0005, Baker's −0.0014, Medial OA −0.0016, Medial
Meniscus −0.0019, MCL −0.0032, ACL −0.0036, Fracture −0.0145. Fourth run in a row with ×1 up and ×4
down; Fracture lost in all four (−0.0039 to −0.0145), MCL in all four.

## Researcher off, stop after the last seed (user, 2026-10-14 17:35)

"stop the campaign after the queued seeds are done. don't let the researcher wake up". The keeper
stopped gracefully (head-rate-twice finished training and comparison), set `refill_below` to 0 in
the state (backup `state.before-researcher-off.json`) so the refill never asks the researcher, and
resumed at 18:10; early-anchor-follows-learning-rate launched at once. A keeper script stops the
campaign once it is decided.

## Stopped by the user with --abort, 2026-10-14 18:40

"stop the campaign forced": early-anchor-follows-learning-rate was aborted after epoch 6 (×1 0.9183
against the incumbent's 0.9178 at that epoch), no result. `refill_below` stays 0. Incumbent
unchanged: the full recipe, ×4 0.937801, ×1 0.929880.
