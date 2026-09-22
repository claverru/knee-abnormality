# Command-line config overrides for training

**Proposed, unimplemented.** Let `scripts/pipeline/train.py` override any experiment field from
the command line, so a one-change run needs no derived YAML:

```bash
rtk proxy uv run --locked python -m scripts.pipeline.train configs/training.yaml \
  --run-id exponent-k05 --trainer.class-balance-exponent 0.5 --seed null
```

Today `train.py` accepts only `--data-root`, `--output-root`, `--folds`, `--run-id`, `--resume`
and `--limit-train`. Any other change means writing a new YAML, and it must sit in `configs/`
because relative paths such as `../data` resolve against the config file's directory.

## Native pydantic-settings support (checked 2026-09-21, pydantic-settings 2.15.0)

Probed in a scratch script against the real `ExperimentConfig`, without repository changes:

- `CliSettingsSource` over a `BaseSettings` subclass of `ExperimentConfig` exposes every nested
  field as a dotted kebab-case flag (`--trainer.class-balance-exponent`,
  `--data.affine-augmentation.scale-range '[0.97,1.03]'`, `--seed null`) and lists them in
  `--help`.
- Calling the source returns only the flags actually given, as a nested dict. Unknown flags such
  as `--trainer.typo` are rejected.
- No hand-written `key=value` parser is needed.

## Design

1. Parse the override flags into that sparse dict, deep-merge it into the raw YAML (the
   overnight launcher's `merge` helper in `scripts/adhoc/train_overnight.py` does this), then
   validate with the existing `config_dir` context in `load_config`. pydantic-settings' own YAML
   source cannot pass that context, so it would break relative-path resolution.
2. Convert values before merging. The CLI source returns scalars as strings (`'0.5'`, `'4'`).
   Floats accept that, but the `StrictInt` fields (`seed`, `trainer.batch_size`,
   `trainer.max_epochs`, …) reject `'4'`; this failure was reproduced. Parse each string leaf
   with `yaml.safe_load`, or relax those fields to `int`.
3. Combine with `TrainCLI`'s own flags: parse `TrainCLI` ignoring unknown arguments, and hand
   the remainder to the override source.
4. Record the given overrides in `provenance.json`. `input_config.yaml` stays the unmodified
   YAML and `effective_config.json` the validated result.
5. Pass trailing arguments through the `train` recipe in the `justfile`.

Scope is training only. Prediction, validation and export load the configuration saved with the
checkpoint, so overrides there could contradict the model. Estimated size: about 15 lines across
`config.py` and `train.py`, plus the `justfile`.

## Follow-ons

- The overnight launcher's `EXPERIMENTS` nested dicts could become override lists.
- `configs/training.yaml` fixes `seed: 2263738815`, while every experiment run should draw its
  own random seed. Removing that line would make `--seed null` unnecessary.
