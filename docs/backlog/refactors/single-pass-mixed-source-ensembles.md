# Single-pass ensembles of checkpoints from different sources

**Proposed, unimplemented.** A multi-checkpoint bundle already predicts in one data pass: the
loader decodes each batch once and every member runs on it before the probabilities are averaged
(`load_ensemble_models` and the stacked sigmoid in `inference/engine.py`), with a baked
τ per member (`export --attention-temperature` takes one value per checkpoint). Two checks keep
checkpoints out of that path when they come from different campaigns:

- **Source identity.** The exporter and the Kaggle loader reject a checkpoint whose
  `preprocessing_identity.sources_sha256` differs from the bundled source
  (`identity.validate_preprocessing_identity`), and one bundle carries one source tree.
- **Data config.** `_validate_members` in `scripts/pipeline/export.py` and `load_ensemble_models`
  require identical `data` sections, so members trained at different `data.study_slice_total`
  cannot share a bundle.

## Why

S80 (2026-09-29) ensembled the ReXNet `rebuild_state_v2` fold-0 control (source `952e0d…`, 26
triplets) with the RadImageNet ResNet-18 incumbent (source `a95abf…`, 36 triplets). It shipped
two bundles in one dataset and a hand-written notebook (scratchpad `ens/submit_ensemble.py`,
outside the repository) that ran `scripts.pipeline.predict` once per bundle and averaged the
CSVs. It scored 0.949 public, against 0.944 for each member alone (S78, S79), but the scoring run took about 41 min because it decodes every test study twice. Its bundle search also walked all of `/kaggle/input`
(345 s before inference on the three public studies). The hashed-source differences between the
two members are additive: an import move in `data/dataset.py`, RadImageNet and ResNet support
in `modeling/model.py`, and ResNet mappings and MaxPool handling in
`modeling/spatial_tokens.py`. None of these touches a ReXNet code path.

## Change

1. **Recorded source parity.** For a member whose recorded `sources_sha256` differs from the
   bundled source, the exporter requires a parity proof: predictions from
   [`check_reference_source_parity`](../../../scripts/adhoc/check_reference_source_parity.py)
   under both source trees, bitwise equal on the first validation studies. The exporter stores
   the proof in the member's manifest record, next to the recorded and bundled hashes. The loader
   accepts a member whose identity is either the executing one or one listed with its proof in
   the manifest. Checkpoint identities stay unmodified. This is the inference-side counterpart of
   the campaigns' `reference_identity_exceptions`. Each use needs a user authorization, as those
   do.
2. **Bundle-level input settings.** `export --study-triplet-total` sets the triplet count for
   the whole bundle, like `--image-multiplier`. Member `data` sections must then be equal apart
   from that key, and the portable config writes the override into each exported member.
3. **Retire the two-bundle notebook.** `submit` keeps one bundle per submission. The scratchpad
   script is not promoted.

## Alternative: separate data and model identities

Replace the single `sources_sha256` with two recorded hashes:

- **Data identity:** the loading sources (`data/dataset.py`, `data/preprocessing/*`,
  `modeling/augmentation.py`), which define the tensors the loader yields. A bundle requires one
  data identity for all members, because the decode is shared.
- **Model identity:** the forward sources (`modeling/model.py`, `modeling/spatial_tokens.py`,
  `inference/reflection.py`), checked per member against the bundled tree.

One process imports one `knee_abnormality.modeling`, so every member still runs on the bundled
model code. A differing model identity therefore still needs the parity proof of item 1, scoped to
the model files. A differing data identity needs it once per bundle. A pure model-side change then
never blocks a shared pass, and a proof covers only the half that changed. Source hashes still flag
non-functional edits: the S80 pair's `dataset.py` differs only by an import move. A data identity
computed from loader outputs on fixed studies would avoid that, at the cost of a decode at export.
Changing the recorded identity invalidates every existing checkpoint's comparison against the
bundled source, unless the loader also accepts the old single hash, which the no-legacy rule
excludes. So adopting this means retraining or exporting from archived sources until new runs
replace them.

## Checks before adoption

- The parity run for the S80 pair: ReXNet under `952e0d…` and under the bundled tree on CPU.
- One Kaggle fold-0 ×4 pass of the ReXNet at 36 triplets against its 26-triplet predictions.
  Running a member off its trained bag size is untested; if it loses AUC, members keep their
  own bag and this route does not apply to them.
- The rebuilt S80 bundle's Kaggle fold-0 predictions match the two-pass averages. The expected
  cost is one decode plus two forward passes: about 3.7 min on fold 0 at ×4, against about
  8 min plus the 6 min scan for S80.

## Not covered

Members with different image sizes, spacing, series types or multipliers still need separate
passes. Final-phase ensembles retrained from one source tree need only item 2.
