import json
import sys
import time
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
import torch
from pydantic import Field
from pydantic_settings import SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.config import INFERENCE_IO_THREADS, load_config
from knee_abnormality.data.dataset import KneeStudyDataset, build_data_loader
from knee_abnormality.data.labels import STUDY, TARGET_THRESHOLD, load_fold_labels
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.preprocessing.dicom import read_frame, read_header
from knee_abnormality.data.preprocessing.series import index_series
from knee_abnormality.data.sources import SERIES, StudySeries, build_series_index
from knee_abnormality.evaluation.metrics import aucs, bootstrap_auc
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.provenance import copy_sources, sha256_file

NEAR_EMPTY_FRACTION = 0.01
POSITION_DECIMALS = 2


class Settings(CLISettings):
    run_dir: Path = Path("outputs/training/oblique-x4")
    output_dir: Path = Field(description="Fresh output directory")
    census: Path = Path(
        "outputs/analysis/series-filtering-20260917/series_flags_with_triplets.parquet"
    )
    files: Path = Path("outputs/analysis/repeated-series-20260922/files.parquet")
    fold: int = 0
    threads: int = 16
    workers: int = 4
    draws: int = 2000
    seed: int = 20260914


class FrameFilterDataset(KneeStudyDataset):
    def __init__(self, *args, dropped: frozenset[str], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropped = dropped

    def _indexed(self, candidate):
        indexed = super()._indexed(candidate)
        if not self.dropped:
            return indexed
        records = tuple(
            r for r in indexed.records if str(r.path.resolve()) not in self.dropped
        )
        if len(records) == len(indexed.records):
            return indexed
        filtered = replace(indexed, records=records)
        self._slice_index[candidate.directory] = filtered
        return filtered


def predict(model, config, frame, index, multiplier, settings, dropped=frozenset()):
    dataset = FrameFilterDataset(
        frame,
        index,
        config.data,
        (),
        None,
        multiplier,
        INFERENCE_IO_THREADS,
        dropped=dropped,
    )
    loader = build_data_loader(dataset, config.trainer, 1, num_workers=settings.workers)
    outputs = {}
    with torch.inference_mode():
        for batch in loader:
            outputs[batch["study_uid"][0]] = torch.sigmoid(
                model(with_images(batch)).float()
            ).numpy()[0]
    return outputs


def with_slots(study: StudySeries, slots: list[list]) -> StudySeries:
    return StudySeries(study.uid, tuple(tuple(values) for values in slots))


def relabel_variants(index, census, studies):
    flagged = census.filter(
        (
            pl.col("f_fluidfs_in_t1pd_confirmed").fill_null(False)
            | pl.col("f_nonfs_in_fluid_confirmed").fill_null(False)
        )
        & pl.col(STUDY).is_in(studies)
    )
    variant, rows = {}, []
    for row in flagged.iter_rows(named=True):
        study = variant.get(row[STUDY], index[row[STUDY]])
        slots = [list(values) for values in study.slots]
        candidate = next(c for c in slots[row["slot"]] if c.uid == row[SERIES])
        slots[row["slot"]].remove(candidate)
        slots[row["relabel_slot"]].append(candidate)
        variant[row[STUDY]] = with_slots(study, slots)
        rows.append(
            {
                STUDY: row[STUDY],
                SERIES: row[SERIES],
                "from_slot": row["slot"],
                "to_slot": row["relabel_slot"],
                "description": row["desc"],
            }
        )
    return {"relabel": variant}, rows


def _middle(indexed) -> np.ndarray:
    record = indexed.records[len(indexed.records) // 2]
    return read_frame(record.path, record.descriptor).pixels


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape:
        return None
    a = left.astype(np.float64).ravel()
    b = right.astype(np.float64).ravel()
    spread = a.std() * b.std()
    return float(np.mean((a - a.mean()) * (b - b.mean())) / spread) if spread else None


def signature_variants(index, studies):
    keep_newest, keep_oldest, rows = {}, {}, []
    for uid in studies:
        study = index[uid]
        newest = [list(values) for values in study.slots]
        oldest = [list(values) for values in study.slots]
        changed = False
        for slot, values in enumerate(study.slots):
            if len(values) < 2:
                continue
            groups = {}
            for candidate in values:
                indexed = index_series(
                    candidate.directory,
                    INFERENCE_IO_THREADS,
                    list(candidate.listing) if candidate.listing is not None else None,
                )
                summary = indexed.summary
                if not indexed.records or summary.signature is None:
                    continue
                key = (
                    len(indexed.records),
                    summary.signature,
                    tuple(
                        round(r.physical_position, POSITION_DECIMALS)
                        for r in indexed.records
                    ),
                )
                groups.setdefault(key, []).append((candidate, indexed))
            for members in groups.values():
                if len(members) < 2:
                    continue
                frames = {c.uid: _middle(indexed) for c, indexed in members}
                if any(
                    np.array_equal(frames[a.uid], frames[b.uid])
                    for (a, _), (b, _) in combinations(members, 2)
                ):
                    continue
                ordered = sorted(
                    members,
                    key=lambda item: (
                        item[1].summary.series_number
                        if item[1].summary.series_number is not None
                        else -np.inf,
                        item[0].uid,
                    ),
                )
                for candidate, _ in ordered[:-1]:
                    newest[slot].remove(candidate)
                for candidate, _ in ordered[1:]:
                    oldest[slot].remove(candidate)
                changed = True
                numbers = [indexed.summary.series_number for _, indexed in ordered]
                rows.append(
                    {
                        STUDY: uid,
                        "slot": slot,
                        "members": len(ordered),
                        "series_numbers": [n for n in numbers if n is not None],
                        "gap": None
                        if None in numbers
                        else float(numbers[-1] - numbers[0]),
                        "middle_correlation": _correlation(
                            frames[ordered[0][0].uid], frames[ordered[-1][0].uid]
                        ),
                        "series": [candidate.uid for candidate, _ in ordered],
                    }
                )
        if changed:
            keep_newest[uid] = with_slots(study, newest)
            keep_oldest[uid] = with_slots(study, oldest)
    return {"keep_newest": keep_newest, "keep_oldest": keep_oldest}, rows


def near_empty_frames(
    files: Path, data_root: Path
) -> tuple[frozenset[str], pl.DataFrame]:
    flagged = pl.read_parquet(files).filter(
        (pl.col("split") == "train") & pl.col("chunk_constant")
    )
    rows = []
    for row in flagged.iter_rows(named=True):
        path = data_root / "train_series" / row[STUDY] / row[SERIES] / row["file"]
        pixels = read_frame(path, read_header(path).descriptor).pixels
        rows.append(
            {
                STUDY: row[STUDY],
                SERIES: row[SERIES],
                "path": str(path.resolve()),
                "nonzero_fraction": float((pixels > 0).mean()),
            }
        )
    frame = pl.from_dicts(rows)
    near = frame.filter(pl.col("nonzero_fraction") < NEAR_EMPTY_FRACTION)
    return frozenset(near["path"].to_list()), frame


def neighbours(files: Path, data_root: Path, series: set[str]) -> dict[str, set[str]]:
    frame = (
        pl.read_parquet(files)
        .filter(pl.col(SERIES).is_in(list(series)))
        .sort(SERIES, "physical_position", "file")
    )
    result = {}
    for (uid,), group in frame.group_by(SERIES, maintain_order=True):
        paths = [
            str((data_root / "train_series" / study / uid / name).resolve())
            for study, name in group.select(STUDY, "file").iter_rows()
        ]
        for position, path in enumerate(paths):
            result[path] = set(paths[max(position - 1, 0) : position + 2])
    return result


def bootstrap_counts(groups, truth, draws, seed):
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(unique), np.full(len(unique), 1 / len(unique)), size=draws
    )[:, inverse]
    positive = counts @ truth
    return counts[
        ((positive > 0) & (positive < counts.sum(axis=1, keepdims=True))).all(axis=1)
    ]


def full_fold_delta(ctx, affected, intact, values):
    reference = ctx["saved"].copy()
    candidate = ctx["saved"].copy()
    for uid in affected:
        reference[ctx["position"][uid]] = intact[uid]
        candidate[ctx["position"][uid]] = values[uid]
    truth, targets = ctx["truth"], ctx["targets"]
    base = aucs(truth.astype(np.int8), reference, targets)
    variant = aucs(truth.astype(np.int8), candidate, targets)
    delta = bootstrap_auc(truth, candidate, ctx["counts"]) - bootstrap_auc(
        truth, reference, ctx["counts"]
    )
    return {
        "full_reference_macro": base["macro_auc"],
        "full_delta": variant["macro_auc"] - base["macro_auc"],
        "full_low95": float(np.percentile(delta, 2.5)),
        "full_high95": float(np.percentile(delta, 97.5)),
        **{
            f"d_{name}": variant["per_target_auc"][name] - base["per_target_auc"][name]
            for name in targets
        },
    }


def shifts(affected, intact, values, targets):
    return pl.DataFrame(
        {
            STUDY: affected,
            "mean_abs_shift": [
                float(np.abs(values[u] - intact[u]).mean()) for u in affected
            ],
            "max_abs_shift": [
                float(np.abs(values[u] - intact[u]).max()) for u in affected
            ],
            "max_shift_label": [
                targets[int(np.abs(values[u] - intact[u]).argmax())] for u in affected
            ],
        }
    )


def run(settings: Settings) -> dict:
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(settings.threads)
    config = load_config(settings.run_dir / "effective_config.json")
    labels = load_fold_labels(config, settings.fold)
    targets = labels.targets
    validation = labels.validation.sort(STUDY)
    studies = validation[STUDY].to_list()
    selected = json.loads(
        (
            settings.run_dir / f"fold_{settings.fold}/selected_checkpoint.json"
        ).read_text()
    )
    multiplier = int(selected["prediction_image_multiplier"])
    model = load_checkpoint_model(
        config,
        targets,
        Path(selected["path"]),
        torch.device("cpu"),
        expected_fold=settings.fold,
    )
    model.eval()
    saved = (
        pl.read_csv(
            settings.run_dir / f"fold_{settings.fold}/validation_predictions.csv"
        )
        .join(validation.select(STUDY), on=STUDY, how="semi")
        .sort(STUDY)
    )
    if saved[STUDY].to_list() != studies:
        raise ValueError("Saved predictions do not cover the validation fold")
    truth = (
        validation.select(f"target_{t}" for t in targets).to_numpy() >= TARGET_THRESHOLD
    ).astype(np.float64)
    ctx = {
        "saved": saved.select(targets).to_numpy().astype(np.float64),
        "position": {uid: i for i, uid in enumerate(studies)},
        "truth": truth,
        "targets": targets,
        "counts": bootstrap_counts(
            validation["report_group"].to_numpy(), truth, settings.draws, settings.seed
        ),
    }
    data_root = Path(config.paths.data_root)
    raw_root = data_root / "raw" if (data_root / "raw").is_dir() else data_root
    index = build_series_index(config.paths.data_root, config.data, "train", studies)

    census = pl.read_parquet(settings.census).filter(pl.col("split") == "train")
    relabel, relabel_rows = relabel_variants(index, census, studies)
    signature, signature_rows = signature_variants(index, studies)
    near_empty, flagged_frames = near_empty_frames(settings.files, raw_root)
    flagged_frames.write_parquet(settings.output_dir / "blank_centre_frames.parquet")
    near_series = set(
        flagged_frames.filter(pl.col("path").is_in(list(near_empty)))[SERIES].to_list()
    )
    touching = neighbours(settings.files, raw_root, near_series)
    near_studies = sorted(
        set(flagged_frames.filter(pl.col("path").is_in(list(near_empty)))[STUDY])
    )
    all_train = build_series_index(
        config.paths.data_root, config.data, "train", near_studies
    )
    usage_frame = pl.DataFrame({STUDY: near_studies})
    blank_dataset = KneeStudyDataset(
        usage_frame, all_train, config.data, (), None, multiplier, INFERENCE_IO_THREADS
    )
    usage_rows = []
    for position, uid in enumerate(near_studies):
        item = blank_dataset[position]
        middles = [
            assignment["middle_path"]
            for meta in item["sampling"]
            for assignment in meta.get("assignments", [])
        ]
        hits = sum(
            1
            for middle in middles
            if touching.get(str(Path(middle).resolve()), set()) & near_empty
        )
        usage_rows.append(
            {STUDY: uid, "triplets": len(middles), "near_empty_triplets": hits}
        )
    usage = pl.from_dicts(usage_rows)
    usage.write_parquet(settings.output_dir / "near_empty_usage.parquet")
    blank_affected = sorted(
        set(usage.filter(pl.col("near_empty_triplets") > 0)[STUDY]) & set(studies)
    )

    plans = {
        "relabel": (relabel["relabel"], frozenset()),
        "keep_newest": (signature["keep_newest"], frozenset()),
        "keep_oldest": (signature["keep_oldest"], frozenset()),
        "drop_near_empty": ({uid: index[uid] for uid in blank_affected}, near_empty),
    }
    affected_all = sorted({uid for variant, _ in plans.values() for uid in variant})
    intact = predict(
        model,
        config,
        validation.filter(pl.col(STUDY).is_in(affected_all)),
        index,
        multiplier,
        settings,
    )
    results, frames = [], []
    for name, (variant, dropped) in plans.items():
        affected = sorted(variant)
        if not affected:
            results.append({"variant": name, "studies": 0})
            continue
        modified = {**index, **variant}
        values = predict(
            model,
            config,
            validation.filter(pl.col(STUDY).is_in(affected)),
            modified,
            multiplier,
            settings,
            dropped,
        )
        shift = shifts(affected, intact, values, targets).with_columns(
            pl.lit(name).alias("variant")
        )
        frames.append(shift)
        results.append(
            {
                "variant": name,
                "studies": len(affected),
                "mean_abs_shift": float(shift["mean_abs_shift"].mean()),
                "median_max_abs_shift": float(shift["max_abs_shift"].median()),
                "max_abs_shift": float(shift["max_abs_shift"].max()),
                "studies_max_shift_over_0.05": int(
                    (shift["max_abs_shift"] > 0.05).sum()
                ),
                **full_fold_delta(ctx, affected, intact, values),
            }
        )
        print(json.dumps(results[-1]), flush=True)
    pl.concat(frames).write_parquet(settings.output_dir / "study_shifts.parquet")
    pl.DataFrame(relabel_rows).write_parquet(
        settings.output_dir / "relabel_series.parquet"
    )
    pl.from_dicts(signature_rows).write_parquet(
        settings.output_dir / "signature_groups.parquet"
    )
    summary = {
        "run_dir": str(settings.run_dir),
        "checkpoint": selected["path"],
        "fold": settings.fold,
        "multiplier": multiplier,
        "device": "cpu",
        "near_empty_fraction": NEAR_EMPTY_FRACTION,
        "near_empty_frames": len(near_empty),
        "near_empty_studies": len(near_studies),
        "near_empty_triplets": int(usage["near_empty_triplets"].sum()),
        "near_empty_total_triplets": int(usage["triplets"].sum()),
        "results": results,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    source = Path(__file__).resolve()
    manifest = {
        "inputs": {
            str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (
                settings.census,
                settings.files,
                Path(selected["path"]),
                source,
            )
        },
        "source_snapshot": copy_sources([source], settings.output_dir / "source"),
    }
    (settings.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return summary


def main() -> None:
    try:
        summary = run(Settings())
    except (OSError, ValueError, SettingsError) as error:
        print(f"Series cleaning rule scoring error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
