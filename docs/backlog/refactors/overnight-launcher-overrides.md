# Overnight launcher on config overrides

**Proposed, unimplemented.** `scripts/tools/train_overnight.py` still merges each
`EXPERIMENTS` change into the control YAML and writes one derived config per run. Since
2026-09-25 `scripts.pipeline.train` accepts any experiment field as a dotted flag
([configs and artifacts](../../architecture/configs-and-experiment-artifacts.md)), so each
queue entry could be a list of flags passed to one control YAML, such as
`--optimizer.head-lr 0.0012 --seed null`. Training already records the flags in
`provenance.json`. This changes how runs are launched, not what they train.

## Related decision (not behavior-neutral)

`configs/training.yaml` pins the current control's seed (1533697213 since 2026-09-28), so a fresh-seed run
passes `--seed null`. Dropping the pinned seed would make fresh seeds the default but changes
the seed policy for autoresearch and manual runs; it needs a decision before this refactor
relies on it.
