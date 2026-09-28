# Medial and lateral transfer: prior work and launch state

Current status: stopped on learned-strength-20260921 after the campaign and manual
follow-ups completed. The cancelled extra run is finalized invalid. See the
[closure record](../../checkpoint-closure.md) for current reference/source and
state; the startup details below preserve original provenance. The user refreshed
the queue and broadened the objective on 2026-09-21 as recorded below.

## Revisit queue on learned strength

The objective is now to improve compartment-specific meniscus and medial/lateral
OA predictions. Shared disease features, anatomical assignment, knee-side and
position encoding are all eligible; an encoding proposal need not add disease
sharing. The user explicitly authorized side-encoding solutions and anatomy-based
assignment, superseding the old scope deferral for the queued anatomy gate.

The current control remains `outputs/training/anatomical-attention/learned-strength`
(macro AUC 0.8974429314502609), seed 2263738815, with the cleaned frozen source in
the closure record. The recent `control-repeat` without anatomical attention is
supporting evidence, not the incumbent. Original startup recipe/reference fields
and sealed resource limits remain historical budget provenance.

| Order | Queued implementation | Reason to revisit or investigate |
|---|---|---|
| 1 | Position-rich shared meniscus readout | Older 1A improved both meniscus labels by about 0.003 on a different recipe. Test classifier sharing with the adopted anatomical separation, keeping position-rich values. The later appearance-only tied classifier harmed meniscus and is not restored. |
| 2 | Anatomical-attention attribution exclusivity | Historical medial-OA gain was +0.005466 versus B, but macro fell 0.000710 and other compartment labels regressed. Anatomically separated attention changes which evidence its erasure penalty removes. |
| 3 | Anatomical-attention local lesion evidence | The stride-16 shared-detector attempt was cancelled and invalid, not a measured failure. Its ungated local evidence can now be read by anatomical attention. |
| 4 | Explicit knee-side/coordinate encoding | Give attention a learned side-location interaction, retaining the accepted bias and position-rich values. This differs from the retired standalone side embedding. |
| 5 | Anatomy-supervised compartment evidence split | Reuse the working geometry to supervise assignment; retain the anatomical incumbent and review overlap with any newly accepted evidence branch. |

Each historical adaptation has a new ID and links to its original source and
results. No historical decision is relabeled, no cancelled checkpoint is resumed,
and no old source becomes the implementation parent. Incumbent changes require
renewed integration review. The other three formerly queued proposals remain
`proposed`, available for later admission rather than scientifically rejected.

The old `shared-family-level-channel` is deliberately not queued unchanged: its
four selected-checkpoint scales had norm about 0.0092 and its targeted family AUCs
barely moved. The +0.000893 macro difference mainly came from other labels. That
is weak evidence for the proposed sharing mechanism, despite the favorable macro
ranking. Self-oriented hemifield gating also substantially overlaps the now-adopted
anatomical solution. Neither deserves an automatic replay or coefficient sweep.

The historical candidates used different seeds and older data/source states;
their scores prioritize hypotheses, not proven gains on learned strength. New runs
inherit the incumbent seed and use the fixed gain >= 0.001 plus positive central
80% interval rule. Per-label diagnostics do not override it. Only this campaign's
queue/guidance changed; all three campaigns remain stopped.

The append-only [decision and before-state](../../../outputs/autoresearch/medial-lateral-transfer/decisions/user-compartment-revisit-20260921-1789998712642067442/decision.json)
record the user authorization, queue, scope amendment and unchanged reference.

## Active run and launch readiness

Historical startup information follows. Current user-requested follow-up:

### Corrected exclusivity follow-up

On 2026-09-21 the user explicitly requested another attribution-exclusivity attempt,
with the inactive mechanism fixed. The completed anatomical-attention attempt had
zero auxiliary loss for the first three epochs and exactly reproduced all control
label AUCs and BCE. Later weighted epoch-mean penalty peaked near 1.35e-6 versus
main loss around 0.47. Its margin of 1.0 left almost all positive logit drops
unpenalized. Preserve its `reject` verdict, but do not interpret it as evidence
against an active exclusivity objective or as a clean estimate of seed variance.

The [corrected proposal](corrected-exclusivity.yaml) replaces the dormant squared
hinge with a linear zero-margin positive-drop penalty. That removes both the 1.0
deadband and the tiny-gradient behavior for small violations. Activity and weighted
loss are logged for interpretation during the ordinary full run; there is no
monitor, adaptive setting or special early stop. Use the current incumbent and its
seed; all accounting and adoption gates remain unchanged. This specific user
authorization supersedes the historical finding that a margin correction is
inadmissible. It does not authorize coefficient sweeps, validation-based tuning or
extra training passes.

After the user explicitly stopped the campaign, the corrected proposal and a fresh
retry of the interrupted local-lesion implementation were inserted at the front of
the live queue. The corrected exclusivity run is first and the local-lesion retry is
second. Neither has launched. The aborted two-epoch local-lesion attempt remains an
invalid operational record and carries no scientific verdict.

Historical startup record: the campaign was prepared and unstarted at this point. Its draft was written while the external
`outputs/training/compartment-control` run was active on 2026-09-20. That run has
now completed and the [result note](../../../docs/validation/results/2026-09-20-compartment-adapter-result.md)
rejects the pooled meniscus adapter. The unstarted campaign now uses control B as
its reference; no campaign state, acceptance or sealed budget existed to migrate.
The draft's adapter-off and independent-adapter proposals were withdrawn before
admission. Future work retains the user's broad medial/lateral-transfer scope.

| Field | Completed external adapter run |
|---|---|
| Run | `outputs/training/compartment-control` |
| Actual mechanism | `model.compartment_adapter: shared`, despite the run name |
| Fold and seed | `all_5fold_v3`, fold 0, seed 2685821520 |
| Selected checkpoint | Epoch 7; macro AUC approximately 0.895099 |
| Recorded time and steps | 1,600.681170 seconds; 4,380 optimizer steps |
| Prediction SHA-256 | `753249bffe7ba8c6538318ba32abb65437bcdc563bff86345eb56aeff7659981` |
| Saved config SHA-256 | `3fa37e1f25e0b35aa83f3ae62ee5ff6fabb04569ab907c9d1ca5052f5f22c6e9` |
| Source archive SHA-256 | `982f16b5d611323c9cec22fdd45515ee32b301444bc6cc97533d2849d7d30dd4` |
| Input identity SHA-256 | `058599b9f93a3ebe21ade57afb5c5b230d2c50bb94cb8cbb12a718c76f05040c` |

The recorded result has an inconclusive macro interval against both controls.
Against control B, the meniscus-family delta is about -0.0114 with interval
[-0.0200, -0.0030], and within-study meniscus discrimination declines. The external
investigation rejected this mechanism under its stated diagnostic criterion.
That is historical evidence, not a campaign verdict or a change to this campaign's
fixed macro acceptance rule. Do not restore the adapter, repeat its shared arm,
or run the retired independent arm or width/coefficient sweeps.

### Pinned starting reference

Use `outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat`
(control B), seed 2263738815, fold 0. Its selected epoch is 8 and saved-prediction
macro AUC is 0.8965816694263277. The recipe is ReXNet-200, 224 px, sigma fraction
0.867, study-total floor 26, reflection 0.5, class-balanced BCE, batch 8, ten epochs,
validation x1, without the pooled adapter. The campaign reads its saved
`effective_config.json` so changes to the user's training YAML do not silently
change the starting recipe. New campaign runs now inherit the incumbent's seed under the shared protocol.

| Reference artifact | SHA-256 |
|---|---|
| `effective_config.json` | `17cdd9c5af428ffca1ea9334e02016ec996709dbd23d838ab363386846a5af42` |
| `source_snapshot.zip` | `dad2e83f2e1a005aab7c949c24f50bde2361e0cbd10de9d10f48db46401dc0a8` |
| `fold_0/input_identity.json` | `2066a2ef4500505cd1a5f38698ffc6d6e01d32e7856d9305ae2c9afc01c572fd` |
| `fold_0/validation_predictions.csv` | `d8de136017222ddb930638674aa543e959b148983f97f1866c077b7889217a24` |

Completed reference time is 1,562.282713 seconds and there are 4,380 optimizer
steps; the 115% ceiling is 1,797 seconds. CPU campaign validation checks the saved
recipe, required files, seed schemas and derived limits. It does not perform the
baseline replay or establish source/input compatibility. Source is captured from
the main tree at initialization: baseline sealing must match the saved reference's
input identity, and source drift must be resolved without relabeling artifacts.
The refreshed definition passed CPU validation with two seed proposals and these
derived limits. No baseline replay or campaign initialization was run.

Before startup, review subsequent external work and avoid duplicate proposals.
The two remaining seeds are recommendations, not an approved execution queue.
Revalidate the definition:

```bash
rtk proxy uv run --locked python -m scripts.tools.autoresearch validate medial-lateral-transfer
```

The user starts the detached campaign when ready:

```bash
rtk proxy uv run --locked just autoresearch-start claude medial-lateral-transfer
```

`codex` is also supported. Startup imports proposals for review and seals the
original budget. No watcher starts the campaign automatically; no runtime artifact
directory was created during this work.

## What has already been tried

The [transfer investigation](../../../docs/research/medial-lateral-transfer.md)
and [retired implementations](../../../docs/modeling/retired-side-and-compartment-variants.md)
own the detailed history. These old comparisons used the archived recipe,
ordinary BCE and x2 validation; their scores are not direct controls for this
class-balanced-BCE, x1 campaign.

| Mechanism | Evidence and interpretation |
|---|---|
| Hard meniscus classifier sharing, 1A | Both meniscus AUCs improved locally by about 0.003; macro gain was only 0.000217 and public score went 0.931 to 0.930. No reliable broad win established. |
| Separate shared meniscus and OA classifiers, 1B | Local macro 0.892022 versus reflection 0.890976, public 0.925 versus 0.931. Much of the local gain came from unrelated labels; the OA pair did not justify the intended transfer. |
| Disease plus signed compartment attention, 1C | Forced the same token-wise medial/lateral attention contrast across the two diseases. At sigma 40, local 0.892538 versus 0.892360 but public 0.931 versus 0.936. A restrictive sharing design with mixed outcomes, not a rejection of shared morphology. |
| Weak meniscus classifier coupling, coefficient 0.1 | Classifier distance fell 20.5%; macro delta was -0.001491595 and meniscus mean -0.004085. Both paired intervals crossed zero. Retired; do not restart a coefficient sweep. |
| Knee-side embedding | The trained model ignored it and tied the old champion. It addressed left/right orientation, a related but different question from medial/lateral disease transfer. |
| Attention priors and token graph | The position/attention-prior line regressed or tied and was retired. These do not establish that every anatomically grounded transfer mechanism fails. |
| Nonlinear pooled meniscus adapter | Completed in `compartment-control`, mode `shared`; rejected after the meniscus-family and within-study discrimination regressions. Keep as evidence, not a starting winner; do not repeat or restore its retired independent arm. |

For exact classifier-coupling diagnostics and uncertainty, use the
[completed result](../../../docs/validation/results/2026-09-14-meniscus-soft-coupling-result.md).
For anatomy-related precedents, use the
[position backlog](../../../docs/backlog/big-bets.md#position-and-knee-side).
Historical proposals marked unimplemented in older research notes may have later
results; follow the linked implementation and outcome before calling them new.

## Open mechanisms and ownership

The campaign owns future work on the user's medial/lateral-transfer hypothesis.
The completed external adapter run retains its original identity and provenance;
its result is not a campaign experiment. The result note and transfer research are
read-only inputs, not a second queue to execute in parallel. At startup, reconcile
any externally completed follow-ups before admitting seeds.

Initial proposals address training-only shared disease supervision and shared
token-level disease evidence with separate compartment routing. They are distinct
questions; a pair of independent auxiliary heads may later be needed to attribute
an auxiliary-objective gain. Neither disease family is excluded from future ideas.

The external result retires the independent pooled adapter using its connection
to the older residual readout and the failed shared result. Respect that retirement.
For a materially different future mechanism, choose attribution controls for its
own feature path and objective; an old capacity result does not settle all future
sharing-versus-capacity questions.

The [disagreement audit](../../../docs/history/sharing/oa-and-meniscus-disagreement-audit.md)
found OA annotation issues and acquisition sensitivity. These are interpretation
limits, not proof of the cause of earlier sharing failures. OA label repair was later
superseded by the [label rebuild](../../../docs/public-labels/label-rebuild.md).
Generic shared-loss and triplet-selection work remain in their respective
campaigns. Anatomical routing belongs here only when it serves the transfer
hypothesis and fits this campaign's fixed data and compute rules.
