# Physical-sampling runtime identity

New runs use `physical_gaussian_overlap_spatial_tokens_v1`, including fixed gap,
complete-link grouping, local-fallback, seed, retry and unique-slice intensity
rules in the preprocessing identity. Source hashing includes
`data/preprocessing/sampling.py` (one entry of `identity.PREPROCESSING_SOURCES`); export copies the package and records SciPy in its
runtime dependency manifest. SciPy is already installed in the pinned Kaggle
runtime and is a locked project dependency. No new package is introduced.

Checkpoint compatibility checks require the configured seed to match the saved
seed because it affects evaluation supplier selection for overlapping
acquisitions. A mismatch reports `seed`; use the run's `effective_config.json`
to preserve its original validation and test inputs.

Old checkpoints and bundles retain their original config and source; strict
identity validation rejects using them with this sampler. Keep the 2× inference
multiplier and existing tensor shapes. Loader capacity ranking now estimates
possible overlapping triplets (`max(n_images - 2, 0)`) rather than dividing unique
files into non-overlapping inputs; geometry and run constraints can lower this
estimate. Actual input counts come from preparation.

On the pinned accepted-center cohort, validation realizes 6,248 triplets versus
5,659 historically, **10.4% more CNN inputs** at the same nominal limits. This is
metadata/preparation evidence, not an inference runtime, memory or accuracy
measurement. Training and model execution were excluded from implementation.
See [verification](../data-preparation/physical-sampling/implementation-verification.md).
