# Model submodules

`KneeAbnormalityModel` in `modeling/model.py` is one `nn.Module` of about 2,000 lines whose methods
share state through attributes. The backbone stages, view windows and mask decoder already live in
their own files (2026-10-14). The rest falls into four groups (line counts of 2026-10-14):

| Group | Methods | Lines |
|---|---|---:|
| Anatomical frame | `frame_rotation`, `rotation_angles`, `study_frames`, `anatomical_frames`, `frame`, `position_code` | ~270 |
| Routing and regions | `routing_widths`, `position_priors`, `side_probability`, `_regions`, `route_sections`, routing weights, `_attention_pool` | ~390 |
| Soft-tissue routing | `soft_tissue_routing_weights`, `_soft_tissue_routing_weights`, `_soft_tissue_tokens`, `_soft_tissue_pool` | ~200 |
| Image encoding | `_encode_images`, `_augment_images`, `_mix_contrast`, `_effective_presence`, `first_look_pose`, `_encode_crops` | ~600 |

- **Change:** turn each group into a small `nn.Module` in its own file that owns its parameters,
  buffers and constants, with `KneeAbnormalityModel` composing them.
- **Keeps:** every output and the random stream. Each submodule must be built at the point of
  `__init__` where its parameters are made now, so the initial weights stay identical; proved by
  the same seed giving the same initial state and the same eval and seeded training forward on a
  validation batch (the check used for the 2026-10-14 merge: initial-state digest, optimizer groups,
  and forward outputs within main's own rerun spread).
- **Costs:** parameter names gain the submodule prefix, so checkpoints trained before it need a key
  map or their own `source_snapshot.zip`; the change touches hashed preprocessing sources, so the
  preprocessing identity changes.
