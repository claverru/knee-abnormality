# Label rebuild: state and family-mean arms (2026-09-28)

## Results (2026-09-28)

**By the pre-registered floor, B passes and A fails.** Only B was recommended
for training by the rule (see [decisions](#decisions-recorded)). The user trained both and later
adopted A2, the fixed version of A ([A2](training-and-a2.md#a2-common-state-scale-and-the-zhen-lineage-fix-2026-09-28)).

Selected τ = 0.15 (held-out-source log-likelihood −0.4775; 1.0: −0.5111, 0.25: −0.4858,
0.1: −0.5027).

### Gold evaluation (pre-registered)

58 gold studies, 696 cells; gold-tuned and gold-copying roots removed on gold rows; A cross-fitted
over the 58 gold report groups (one study each); 10,000 valid bootstrap draws.

| Arm | Macro AUC | Δ vs `combined_v1` (95%) | P(better) | Log loss | Δ log loss (95%) |
|---|---:|---|---:|---:|---|
| `combined_v1` | 0.9075 | — | — | 0.4471 | — |
| **A**, state + calibration, out-of-fold | 0.8838 | −0.0238 [−0.0372, −0.0109] | 0.000 | 0.3606 | −0.0865 [−0.1286, −0.0447] |
| **B**, family mean | 0.9022 | −0.0054 [−0.0181, +0.0072] | 0.193 | 0.3732 | −0.0739 [−0.1047, −0.0444] |
| A0, state rubric (diagnosis) | 0.9006 | −0.0069 [−0.0200, +0.0059] | 0.137 | 0.7323 | +0.2851 [+0.1474, +0.4343] |

| Target | `combined_v1` | A (OOF) | B | A0 |
|---|---:|---:|---:|---:|
| ACL | 0.996 | 0.994 | 0.995 | 0.995 |
| MCL | 0.996 | 0.975 | 0.986 | 0.982 |
| Medial Meniscus | 0.948 | 0.928 | 0.940 | 0.953 |
| Lateral Meniscus | 0.879 | 0.862 | 0.886 | 0.880 |
| Medial OA | 0.932 | 0.933 | 0.943 | 0.924 |
| Lateral OA | 0.833 | 0.822 | 0.849 | 0.851 |
| PF OA | 0.929 | 0.909 | 0.923 | 0.906 |
| Effusion | 0.877 | 0.826 | 0.867 | 0.853 |
| Synovitis | 0.790 | 0.773 | 0.811 | 0.794 |
| Baker's | 0.944 | 0.909 | 0.917 | 0.946 |
| Contusion | 0.860 | 0.854 | 0.872 | 0.838 |
| Fracture | 0.907 | 0.819 | 0.838 | 0.886 |

The `combined_v1` row is favoured: its gold rows contain JEV and Josh, and its four consensus
targets were chosen on these studies. B uses neither.

**Diagnostics, after the fact and changing nothing:** fitted on all gold, A scores 0.9022
in-sample, the same as B, so most of the out-of-fold loss comes from refitting without each
single held-out study. A likely mechanism is leave-one-out pessimism: removing a positive lowers
its own target offset. This is plausible, not proven. Log loss, which the calibration targets, is
best for A.

### Report states and calibration

Gold-positive rate by argmax state, pooled: NEG 0.03 (185 cells), NOT_MENTIONED 0.13 (193),
LOW 0.34 (87), HEDGED 0.61 (80), POS 0.85 (151). The states keep their intended order and meaning.

Silver cells by argmax state (NEG / NOT_MENTIONED / LOW / HEDGED / POS):

| Target | NEG | NM | LOW | HEDGED | POS |
|---|---:|---:|---:|---:|---:|
| ACL | 2,804 | 347 | 533 | 220 | 445 |
| MCL | 3,187 | 410 | 528 | 175 | 49 |
| Medial Meniscus | 1,674 | 241 | 682 | 116 | 1,636 |
| Lateral Meniscus | 2,774 | 443 | 475 | 79 | 578 |
| Medial OA | 1,751 | 976 | 483 | 793 | 346 |
| Lateral OA | 1,943 | 1,295 | 433 | 480 | 198 |
| PF OA | 1,554 | 807 | 540 | 964 | 484 |
| Effusion | 1,291 | 429 | 1,449 | 593 | 587 |
| Synovitis | 49 | 3,763 | 153 | 0 | 384 |
| Baker's | 1,110 | 2,152 | 534 | 0 | 553 |
| Contusion | 1,774 | 1,280 | 398 | 487 | 410 |
| Fracture | 820 | 3,239 | 0 | 57 | 233 |

Calibration (final fit on all gold): state effects NEG −2.19, NM −2.01, LOW −2.11, HEDGED −1.53,
POS −1.28; slope γ 0.84 on z; target offsets −0.82 (MCL) to +0.98 (Synovitis). Because z carries
much of the signal, the map is read at each state's median silver z:

| Target | NEG | NM | LOW | HEDGED | POS |
|---|---:|---:|---:|---:|---:|
| ACL | 0.02 | 0.10 | 0.18 | 0.63 | 0.93 |
| MCL | 0.02 | 0.11 | 0.24 | 0.72 | 0.93 |
| Medial Meniscus | 0.01 | 0.07 | 0.13 | 0.47 | 0.87 |
| Lateral Meniscus | 0.03 | 0.13 | 0.24 | 0.62 | 0.91 |
| Medial OA | 0.02 | 0.07 | 0.22 | 0.57 | 0.87 |
| Lateral OA | 0.02 | 0.07 | 0.20 | 0.49 | 0.80 |
| PF OA | 0.04 | 0.12 | 0.31 | 0.69 | 0.89 |
| Effusion | 0.03 | 0.14 | 0.32 | 0.76 | 0.87 |
| Synovitis | 0.05 | 0.20 | 0.61 | — | 0.90 |
| Baker's | 0.02 | 0.07 | 0.30 | — | 0.81 |
| Contusion | 0.02 | 0.06 | 0.12 | 0.36 | 0.78 |
| Fracture | 0.02 | 0.08 | — | 0.53 | 0.82 |

### Silver changes against `combined_v1`

Positive rate at 0.5 and changed calls on the 4,349 silver studies:

| Target | `combined_v1` | A | A changed | B | B changed |
|---|---:|---:|---:|---:|---:|
| ACL | 14.7% | 14.3% | 23 | 15.8% | 55 |
| MCL | 5.2% | 5.0% | 14 | 6.0% | 35 |
| Medial Meniscus | 40.4% | 38.3% | 130 | 41.0% | 70 |
| Lateral Meniscus | 15.4% | 15.0% | 25 | 15.4% | 24 |
| Medial OA | 37.3% | 19.4% | 782 | 28.6% | 385 |
| Lateral OA | 27.4% | 9.8% | 763 | 16.6% | 470 |
| PF OA | 31.7% | 31.8% | 68 | 35.2% | 157 |
| Effusion | 59.4% | 27.1% | 1,411 | 29.0% | 1,328 |
| Synovitis | 12.4% | 12.3% | 7 | 12.5% | 13 |
| Baker's | 24.7% | 12.8% | 522 | 22.3% | 107 |
| Contusion | 17.2% | 10.6% | 289 | 17.9% | 197 |
| Fracture | 6.1% | 6.2% | 10 | 6.3% | 11 |

Fold 4 (3,538 training studies): validation targets and gold training targets are identical to
`combined_v1` for both tables. Training positives change most for A in Lateral OA 962 → 345,
Medial OA 1,310 → 689, Effusion 2,097 → 978 and Baker's 868 → 451, and for B in Effusion
2,097 → 1,047, Lateral OA 962 → 590 and Medial OA 1,310 → 1,007. Class-balance weights move with
them (`fold4_training_positives.json`).

### Decisions recorded

1. **A is not recommended by the rule.** Training it anyway rests on the in-sample tie with B and
   the better log loss.
2. **Effusion** falls to about 28% positive in both tables, the same direction as the
   [`combined_v2` rule](../report-attributes.md#consensus-mean-rule-tested-on-gold)
   that lost gold AUC; B's Effusion AUC is 0.867 against 0.877.
3. **B loses Fracture** (0.838 against 0.907, whose consensus was selected on these studies) and
   Baker's (0.917 against 0.944), and gains OA, Synovitis and Contusion.
4. **Our JevK5 attributes** are a root and an anchor; they are not a Kaggle table.
5. **Hosseini** is treated as possibly gold-tuned; the Tonylin-style derivatives, Nekkon NLI,
   achelijndiamantidis, lvthanhxun, Freshtime and the new narisetti image teacher are excluded.
6. **Josh sits in the Vmohit family,** so the state model never reads it (Vmohit represents it).
7. **The two pre-gold revisions** (anchor constraints, τ grid extension) were made after silver
   diagnostics and before any gold number.

### Outputs and launch

- Tool: [`prepare_rebuilt_labels`](../../../scripts/tools/prepare_rebuilt_labels.py), CPU, about
  40 s, deterministic, refuses existing outputs:
  `uv run --locked python -m scripts.tools.prepare_rebuilt_labels --analysis-dir outputs/analysis/label-rebuild-new`
  (move the two prepared directories and manifests aside first).
- Tables: `data/labels/prepared/rebuild-state-v1/labels.csv` (SHA-256 `d47b631d…`) and
  `data/labels/prepared/rebuild-mean-v1/labels.csv` (`0db443ab…`); manifests
  [`rebuild-state-v1`](../../../configs/labels/rebuild-state-v1-label-sources.json) and
  [`rebuild-mean-v1`](../../../configs/labels/rebuild-mean-v1-label-sources.json).
- Analysis: `outputs/analysis/l/label-rebuild-20260928/` (`report.json` with families, kappa,
  tempering, states, calibration and gold evaluation; `state_model.json`;
  `fold4_training_positives.json`).
- Both arms were launched by the user on fold 4 (seed 703791404), pairing with `scanner-confusion-control`, and compared publicly with S73 at ×4, τ 0.75. Local validation scored against the frozen `combined_v1` targets, which both tables move away from; the public score decided.
