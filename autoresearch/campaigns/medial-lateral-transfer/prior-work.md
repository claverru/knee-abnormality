# Medial and lateral transfer: prior work and launch state

Current status: stopped on learned-strength-20260921 after the campaign and manual
follow-ups completed. The cancelled extra run is finalized invalid. See the
[closure record](../../checkpoint-closure.md) for current reference/source and
state; the startup details below preserve original provenance.

## Active run and launch readiness

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
limits, not proof of the cause of earlier sharing failures. OA label repair remains
owned by [its separate plan](../../../docs/backlog/big-bets.md#oa-only-rubric-aligned-label-repair).
Generic shared-loss and triplet-selection work remain in their respective
campaigns. Anatomical routing belongs here only when it serves the transfer
hypothesis and fits this campaign's fixed data and compute rules.
