# Final chain: rented-machine scripts

These are the scripts, job tables and specs that built the external data, the label tables and the
teacher predictions behind the 60 S188 members. They ran on the rented Vast.ai machine between
2026-10-13 and 2026-10-20. The machine was destroyed on 2026-10-20. The files are a **frozen
record** of what ran there. They were copied byte for byte from
`outputs/rented-machine-archive-20261020/root/`, and `dispatch/jobs.tsv` came from the archive's
`dispatch/`; on 2026-10-23 the Python files were reformatted with `ruff format`, which changed
layout only (the unformatted copies are in commit `b3f3f62` and in the archive). They were not refactored to current conventions, and they assume the host layout
described below. The chain itself is described in the
[teacher-student chain](../../../docs/modeling/external-data-and-distillation/teacher-student-chain.md),
the external studies in [external data](../../../docs/modeling/external-data-and-distillation/external-data.md),
and the result in the [final solution](../../../docs/final-solution.md).

**Secrets:** none was found in the copied files, so nothing is redacted and there are no
placeholders. The download links lived in input files that were deleted on the machine after use
(`/root/kaggle_url.txt`, `/root/fastmri_urls.txt`, `/root/oai_aria2_input.txt`,
`/root/oai_<part>.txt`). The archive's `.vast_api_key` and `.ssh/authorized_keys` were not copied.

## Host paths

| On the host | What | Here or locally |
|---|---|---|
| `/root/<file>` | these scripts and their inputs (lists, specs, limits files) | this folder |
| `/home/claudio/src/knee-abnormality-<tree>` | checkouts of this repository. The rebuild (v2) and v3 chains ran in `-nextvit` with `-sync-free/.venv`. The silver-15 and teachers2 rounds used `-sync-free`, with FocalNet in `-teachers2`, EfficientViT in `-effvit` and FocalNet silver teachers in `-focalnet`. The OAI screens ran in `-fastmri-pretraining` | the repository root. The code state is in [Code at the time](../../../docs/modeling/external-data-and-distillation/teacher-student-chain.md#the-chain); snapshots of some trees are in `outputs/rented-machine-archive-20261020/export-src/` |
| `/workspace/dispatch/{jobs.tsv,claims/,done/}` | the job queue: rows, claim folders, `done/<id>` markers (`<host> <exit code> <time>`) | `dispatch/jobs.tsv` here (final state); claims, markers and the `jobs-before-*.tsv` snapshots in the archive's `dispatch/` |
| `/workspace/downloads`, `/workspace/data/raw/{fastmri,oai}` | downloaded archives, extracted DICOMs | not copied home |
| `/workspace/data/{fastmri,oai}` | laid-out roots (`raw/test_series/` links, `test_series.csv`, `sample_submission.csv`, `studies.csv`, `series_catalog.parquet`) | catalogs and study lists in the archive's `data/fastmri`; images not copied |
| `/workspace/data/external` | fastMRI + OAI joined, the `extra_training.data_root` that `configs/*-pretrain.yaml` still name | the archive's `data/external` (`studies.csv`, `excluded_from_training.csv`) |
| `/workspace/data/labels/prepared/<table>` | label tables and mask targets | `data/labels/prepared/<table>`, same names; also the archive's `data/labels/prepared` |
| `/workspace/data/folds/<split>` | splits | `data/folds/<split>` |
| `/workspace/outputs/training/<run>` | runs | `outputs/training/<run>` |
| `/workspace/predictions/{silver15,rebuild,rebuild-external,gen3-teachers,pretrain-all}` | teacher predictions | the archive's `predictions/` (except `pretrain-all`) |
| `/workspace/outputs/logs/{night2..night6,rebuild}`, `/workspace/logs` | state logs (`chain.state`, `rebuild.state`) and run logs | the archive's `logs/` |
| `/home/claudio/.cache/knee-segmenters` (`/root/.cache/...`) | segmenter environment and converted volumes | `~/.cache/knee-segmenters` |

## Order

### 1. External data on the host

| File | What it did |
|---|---|
| `new_download.sh` | Installed aria2 and fetched the Kaggle zip and fastMRI `knee_DICOMs_batch{1,2}.tar.xz` from URL files. |
| `rebuild_process.sh` | Built a layout environment, unzipped Kaggle with `parallel_unzip.py`, extracted the fastMRI batches, then ran `fastmri_layout.py` and `uncompress_fastmri.py sample/convert/verify`. |
| `parallel_unzip.py` | Extracted a zip with parallel workers and checked the file sizes. |
| `fastmri_layout.py` | Cataloged every fastMRI series (plane, fat suppression from header and name, geometry) and named studies `fastmri-<patient>-<study>`. It linked the kept series into the competition layout with a 0.5 placeholder `sample_submission.csv`, and dropped the studies in `fastmri_rejected.csv`. |
| `fastmri_rejected.csv` | The two studies that the pipeline loader could not decode. |
| `uncompress_fastmri.py` | Rewrote fastMRI as uncompressed DICOM with Pillow's decoder, hashing 4,000 sampled images before and after ([why](../../../docs/modeling/external-data-and-distillation/external-data.md#fastmri)). |
| `fastmri_study_headers.py` | Wrote `study_headers.parquet` (manufacturer, model, field strength per study). |
| `oai_download.sh`, `oai_extract.sh` | Downloaded OAI with aria2 from signed links and extracted each archive when it finished. |
| `oai_fetch.py`, `oai_fetch_run.sh` | Did the same with a Python fetcher, for link files split into parts. |
| `oai_layout.py` | Laid out OAI baseline (`image03/00m`) with `fastmri_layout.describe`. Studies are `oai-<patient>-<side>`, with series SAG_IW_TSE, COR_IW_TSE, COR_MPR and AX_MPR. |
| `screen_loader.py`, `screen_geometry_any.py`, `oai_screens.sh` | Ran every study through the loader (rejections) and the training dataset (studies without a valid series geometry), for OAI through `oai_screens.sh`. |
| `external_root.py` | Joined fastMRI and OAI into `/workspace/data/external` with one `studies.csv` and one `excluded_from_training.csv`. The fastMRI exclusions came from `ensemble10-avg-labels-x4-tau085-tta`, the OAI ones from `/workspace/data/oai`. |

### 2. Mask targets for the pretrainings

| File | What it did |
|---|---|
| `fastmri_masks_extra.sh` | The second fastMRI bone-mask batch, in this order: headers, segmenter environment, `fastmri_choose_extra.py <name> 0.2`, `fastmri_convert.py`, MRAnnotator, then the export, joint and measurement scripts of [`side_routed_session/`](../side_routed_session/). |
| `fastmri_choose_extra.py` | Filled each scanner group up to its square-root quota of 20% of the fastMRI training studies (seed 20261014), skipping the first 1,230 knees. |
| `fastmri_convert.py` | Converted the chosen sagittal series to NIfTI for the segmenter. It also served the first 1,230-knee sample. |
| `merge_masks.py` | Merged the first sample and the extra batch into `anatomy-mrannotator-sample550-fastmri1966-joints`, the `paths.anatomy_targets` of every pretraining config. It dropped knees with femur under 80 ml or tibia under 45 ml. |
| `oai_dess_full.sh`, `oai_dess_full2.sh` | OAI soft-tissue masks: `oai_convert_dess.py`, four `oai_segment_shard.py` shards, then [`meniscus_masks/export_dess.py`](../meniscus_masks/export_dess.py) to `soft-tissue-nnunet-knee-oai1887`. The second script redid the segmentation and the export. |
| `oai_convert_dess.py` | Converted the selected DESS series (`/workspace/data/oai/selection_dess.csv`) with [`meniscus_masks/convert_dess.py`](../meniscus_masks/convert_dess.py)'s orientation and crop, and recorded whether each DESS shares the frame of reference of the knee's other series. |
| `oai_segment_shard.py` | Ran nnU-Net (`nnunet_knee`, fold 1) on one shard of the volumes. |
| `oai_soft_tissue_final.py` | Removed knees whose DESS frame of reference differs from their other series, then linked OAI with Kaggle's DESS knees into `soft-tissue-nnunet-knee-dess265-oai`, the `paths.meniscus_targets` of every pretraining config. |

### 3. Teacher predictions

| File | What it did |
|---|---|
| `predict_teacher_x4.py <run> <fold> <out.parquet>` | Predicted one run's own validation fold at ×4, τ 0.85 with mirror TTA, and wrote probabilities plus a `.json` record. It fed the Kaggle tables (`rsna` jobs). |
| `external_predict_rebuild.py <ckpt> <out dir> <workers>` | Predicted every fastMRI and OAI study at ×4, τ 0.85 with mirror TTA, and wrote `fastmri.parquet`, `oai.parquet` and `record.json` (`external` jobs). |
| `external_predict.py`, `external_predict_lane2.py` | The same for the silver-15 round, serialized on a GPU lock file (two lanes). |
| `predict_pretrain_all.py`, `cycle_pretrain_predictions.sh` | Predicted every Kaggle study with each teachers2 pretraining (all five folds). |

### 4. Label tables

| File | What it did |
|---|---|
| `build_external_table_rebuild.py <teachers list> <out dir>` | External table: the mean probability of the listed teachers' external predictions, without the excluded studies. OAI rows get empty ACL, MCL, Effusion, Synovitis, Baker's, Contusion and Fracture ([rule](../../../docs/modeling/external-data-and-distillation/external-data.md#oai)). The table is repeated for folds 0–4 as `external_training_table.parquet`. |
| `build_kaggle_table.py <spec> <out dir>` | Kaggle teacher table: for each group, the mean over families of their own-fold predictions, then the weighted mean of the groups, stored as logits in `kaggle_teacher_table.parquet` (4,407 rows, `reading` = `joint`). |
| `build_teacher_labels.py <teachers list> <out dir>` | The external-table builder of the teachers2 round (reads `/workspace/predictions/silver15`; run by `night5.sh` with `teachers2.list`). |
| `build_silver15_labels.py` | The silver-15 round's external table: 15 fixed teachers, every label kept (run by `night2.sh`). |
| `teachers-rebuild-default.list` → `rebuild-gen1-x4-tau085-tta-oai-soft-tissue` | v2 external table (NextViT teachers: `nextvit-silver-fold-{0,1,3,4}` and `nextvit-gen1-fold-2`). |
| `spec-t2-rebuild.json` → `rebuild-t2-x4-tau085-tta`; `spec-mixed-rebuild.json` → `rebuild-mixed-x4-tau085-tta` | v2 gen-2 and gen-3 teacher tables (mixed = 0.5 gen 1 + 0.5 gen 2). |
| `teachers-v3.list`, `spec-t2-v3.json`, `spec-mixed-v3.json` | The same three tables for v3, built by `v3_chain.sh`. |
| `teachers2.list` | The ten silver teachers of the teachers2 external table. |

### 5. Training and the job queue

The rebuild (v2) and v3 chains ran as rows of `/workspace/dispatch/jobs.tsv`. Each row has nine tab-separated fields:

1. id
2. family
3. config, or the run to predict
4. fold
5. seed
6. kind
7. host: `any`, `remote`, `local` or `machine2`
8. extra training arguments (`@T@` = `/workspace/outputs/training`)
9. done markers it waits for, comma-separated

The kinds are:

- `teacher`, `finetune` and `pretrain` train through `scripts/tools/preflight_training.py --memory-cap-gib 40`, with `KNEE_AUTORESEARCH_LIMITS=nextvit-limits.json` (10 epochs).
- `rsna` runs `predict_teacher_x4.py`.
- `external` runs `external_predict_rebuild.py`.

Some markers were written by hand (`teacher-decision`, `v3-gen2-share`, `v3-pretrain-local`, `gen3-table`). Others came from `v3_chain.sh` or from finished jobs.

| File | What it did |
|---|---|
| `jobs-gen1.tsv` | v2 gen-1 teachers, six families × five folds on `configs/{family}.yaml`. |
| `jobs-v2.tsv` | v2 own-fold (`rsna-*`) and external (`ext-*`) predictions of the gen-1 teachers. |
| `jobs-gen2-gated.tsv` | v2 pretrainings (`configs/{family}-rebuild-pretrain.yaml`) and gen-2 fine-tunes (`configs/{family}-rebuild-gen2.yaml`, splits `_r7`, `_r8`, `_r12`), gated on the teacher decision. |
| `extra-pretrains.tsv` | The FocalNet, MambaVision and EfficientViT pretrainings, later dropped. |
| `jobs-gen3.tsv` | v2 gen 3 (`configs/{family}-rebuild-gen3.yaml`, splits `_r13`, `_r14`, `_r18`). |
| `jobs-v31-gen1.tsv` | Gen-1 teachers on the v3.1 labels (not adopted; their label manifest is no longer in the repository). |
| `jobs-v3-chain.tsv` | v3 predictions, pretrainings (`configs/{family}-v3-pretrain.yaml`), gen 2 and gen 3 (`configs/{family}-v3-gen{2,3}.yaml`, splits `all_5fold_scanner_v3_r1`…`_r6`). |
| `dispatch/jobs.tsv` | The queue's final state, 246 rows. It is the fullest record of seeds, splits and starting weights, and adds the v3 gen-1 teachers and the NextViT silver predictions. |
| `dispatch_machine.sh`, `dispatch_machine2.sh`, `dispatch_remote.sh` | The machine-side workers. Each claimed a row, ran it by kind and wrote the done marker. `dispatch_remote.sh` also predicted a run's own fold right after training. The local 3090 ran `local` rows with `outputs/logs/rebuild/dispatch_local.sh`. |
| `v3_chain.sh` | Built the v3 external, T2 and mixed tables as soon as their prediction jobs were done, and wrote the markers that released the v3 pretrainings and fine-tunes. |
| `nextvit-limits.json`, `effvit-limits.json` | Preflight limits files (10 epochs, no time bound). |
| `cycle_configs.py` | Wrote `{family}-share075-{finetune,joint}.yaml` (share 0.75 on `teachers2-kaggle-x4-tau085-tta`) into the host trees. |
| `cycle.sh`, `cycle_gen3.sh`, `cycle_gen3_variants.sh`, `cycle_gen3_resplit.sh`, `cycle_nextvit.sh` | The teachers2-round share cycle and its third-generation variants (gen3-a, all5, mixed, resplit with one seed per run, NextViT). Their configs are the retired ones listed in [retired pieces](../../../docs/history/fastmri-chain-retired-pieces.md#configs-of-rounds-and-families-not-in-the-final-recipe). |

## Against current code

- The prediction scripts' `knee_abnormality` imports all resolve on current main. `predict_teacher_x4.py` and `build_kaggle_table.py` match today's function signatures.
- `screen_loader.py` and `screen_geometry_any.py` load `configs/training.yaml`, which no longer exists.
- `oai_layout.py` imports `fastmri_layout` from `/root`, which is this folder.
- `oai_convert_dess.py` and `fastmri_masks_extra.sh` call `scripts/adhoc/meniscus_masks/` and `scripts/adhoc/side_routed_session/`, both still present.
- The cycle scripts' configs (`*-teachers2-*`, `*-share075-*`, `*-gen3mixed-*`, `*-silver15-*`) and `configs/labels/rebuild-state-v3-1-label-sources.json` are not in the repository.
- The following are not in the archive:
  - the first fastMRI sample's chooser (`fastmri_choose.py`, kept beside the sample in `~/.cache/knee-segmenters/fastmri1230` on the rented machine and lost with it)
  - whatever wrote OAI's `selection_dess.csv`
  - the step that turned the geometry screens into `excluded_from_training.csv`
  - the builders of `teachers2-kaggle-x4-tau085-tta` and `gen3-*-kaggle-x4-tau085-tta`, which were built from local predictions

## Not copied

These stay in `outputs/rented-machine-archive-20261020/root/`.

- **Machine plumbing (54 files):**
  - job claiming: `claim_job.sh` and its `.bak`, `finish_job.sh`, `release_job.sh`, `running_jobs.sh`
  - restarts and guards: `revive.sh` and its `.bak`, `onstart.sh`, `ram_guard.sh`, `pause_until_converted.sh`
  - host vetting: `vet_host.sh`, `ports.log`, `hasbooted`
  - benchmarks and smoke tests: `scaling.py`, `sustained.py`, `gpu_bench.py`, `focal_smoke.py`, `smoke_labels.py`, `smoke_external_data.py`
  - data and logs: `uncompress_sample.json` and the three `v3_*_table.log` outputs
  - home dotfiles, including the Vast API key and `authorized_keys`
  - the earlier rounds' queue wrappers: `night2.sh` to `night6.sh`, `switch_to_night3.sh`, `silver_queue.sh`, `silver2.sh`, `convnext_silver.sh`, `c8.sh`, `spec_c6.sh`, `effvit.sh`, `effvit-pretrain.sh`, `r50-pretrain.sh`, `focalnet_queue.sh`, `focalnet_predict.sh`, `focalnet_predict2.sh`, `main_convnext_kaggle_only.sh`, `main_rn50_student.sh`, `repredict.sh`, `pred_lane2.sh`, `pred_r0.sh`, `pred_rerun.sh`
- **One-off analyses and checks (28 files):**
  - the τ diagnostics and mechanics: `predict_tau.py`, `tau_ext.py`, `tau_val.py`, `tau_mechanics_machine.py`, `run_tau*.sh`, `run_taumech.sh`, `sched_taumech*.sh`, `stop_taumech.sh`, `run_gamma.sh`, `tau_tasks.txt` ([attention temperature](../../../docs/validation/results/pooling-and-attention/2026-10-22-attention-temperature.md))
  - ×1/×4 reference checks: `x4_mirror_reference.py`, `random_x1_*.py`
  - code parity and sync audits: `parity_*`, `merged*_check.sh`, `sync_audit.py`
  - OAI inspections: `oai_inspect.py`, `oai_frames.py`
  - the OAI DESS pilots: `oai_dess_pilot*.sh`
  - training monitors: `mask_metrics.py`, `pretrain_update.py`

The local half of the chain is in `outputs/logs/rebuild/` (main checkout, gitignored):

- `dispatch_local.sh`, `make_gen2_jobs.py`, `combine_teachers.py`
- `gen4.sh` and the `gen4*-table-spec.json` files
- `teacher_selection.py`, `share_curve*.py`
- the seed tables
