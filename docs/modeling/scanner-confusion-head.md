# Scanner confusion head

**Part of the ReXNet recipe (`configs/rexnet-200.yaml`) since 2026-09-27; removed from the
ResNet-18 recipe on 2026-09-30** (user decisions): a training-only head learns the study's
scanner group from detached tokens, and a bounded confusion loss pushes the backbone's tokens
toward carrying no more scanner information than the group frequencies. It is dropped at
inference. At `trainer.scanner_confusion_weight: 0` the head is not built. It never engaged: its
accuracy kept rising and its KL term stayed about 0.1% of the loss on both backbones, and removing
it was neutral ([removal](../validation/results/2026-09-30-rn18-l2sp-no-confusion.md)). The evidence and the rejected variants are in [bias_free.md](../../bias_free.md).

## Why

The hidden-test labels come from the images; the training labels mostly from reports, which carry
site reporting conventions ([gold versus silver](../public-labels/gold-versus-silver.md)). The
head discourages the backbone from encoding the scanner, the closest image-visible stand-in for
the site.

## Configuration

Required in every experiment config; 0 disables the head:

```yaml
trainer:
  scanner_confusion_weight: 0.1
```

## Mechanism

`ScannerConfusionHead` (`training/auxiliary/scanner_confusion.py`) lives in the Lightning module,
outside `KneeAbnormalityModel`, so prediction, export and checkpoint loading are unchanged.

- **Targets:** the 8 scanner groups of the split protocol (Siemens, Philips and GE × 1.5T, 3T and
  unknown field strength; every other vendor pooled as minor), from
  `outputs/analysis/sharing-disagreement-20260914/study_headers.parquet`, checked against the
  scanner-table SHA-256 in the configured split's `protocol.json` (the same table for
  `all_5fold_scanner_v1` and `v2`).
- **Head:** one attention query (256 → 128 → 1) over the tokens the disease head pools, then a
  linear 8-way classifier; forked-RNG initialization, head learning rate.
- **Head term:** cross-entropy on **detached** tokens, unweighted; only the head learns from it.
- **Confusion term:** the head runs again on the live tokens with its parameters detached
  (`torch.func.functional_call`); the backbone receives
  `0.1 × ramp × KL(frequencies ‖ prediction)`, where the frequencies are the training fold's group
  shares and `ramp = 2 / (1 + exp(-10 p)) − 1` over training progress `p`. The KL is minimized when
  the head can do no better than the frequencies and cannot be pushed further.
- **Logged per epoch:** head loss and accuracy (baselines: majority rate 0.263, frequency entropy
  1.851), confusion KL, ramp and effective weight.

## Evidence

Fold 3, seed 2617320043, against `half-stack-control` (×4 0.905205, public 0.947, S68): ×4
0.904442 (−0.00076, 80% [−0.0027, +0.0011]) and **0.948 public (S67)**. The head learned the
scanner beyond the frequencies (accuracy 0.314, loss 1.724) while disease AUC on the major vendors
held (+0.0002), and minor-site predictions shifted against that site's label inflation (Synovitis
−0.27 logits). The public gain is one rounding step. Heavier variants lost publicly in the order of
their disease cost: scanner-model target 0.944 (S70), stronger head 0.943 (S71).

## Rejected alternatives

- **Gradient reversal** on the same head lost 0.004–0.006 locally: it maximizes an unbounded
  cross-entropy, so the backbone bent features to make the head confidently wrong.
- **Stronger pressure, finer targets, earlier features, per-site and per-model logit biases:**
  measured in the auxiliary-signals campaign; none beat this head publicly
  ([bias_free.md](../../bias_free.md)).
