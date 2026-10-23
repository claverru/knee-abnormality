import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, Runtime, describe
from knee_abnormality.config import PositiveFiniteFloat
from knee_abnormality.inference.bundle import discover_kaggle_data_root, verify_bundle
from knee_abnormality.provenance import sha256_file

SCALES = ("half", "double")
CONTRASTS = {0: "t1pd", 1: "fluid"}


class Settings(CLISettings, Runtime):
    model_config = describe(
        "Predict the fold-0 validation studies once with the bundle's per-type triplet "
        "budgets and once per series type with that type's per-study budget halved "
        "and doubled, recording the realised triplets per study and type. Runs with "
        "PYTHONPATH set to the bundle's code directory."
    )

    bundle: Path = Field(description="Exported bundle directory")
    inputs: Path = Field(
        description="Directory holding studies.csv, targets.csv, protocol.json, "
        "labels.csv and the label manifest"
    )
    label_manifest: str = Field(
        default="rebuild-state-v2-label-sources.json",
        description="Label manifest file name inside the inputs directory",
    )
    raw: Path | None = Field(
        default=None,
        description="Competition data directory; defaults to the Kaggle input",
    )
    patch_dir: Path = Field(description="Directory holding the sitecustomize patch")
    adapter: Path = Field(description="Data root of symbolic links to build")
    output_dir: Path = Field(description="Directory for predictions and pass records")
    fold: int = Field(default=0, ge=0, description="Validation fold")
    image_multiplier: int = Field(default=1, gt=0, description="Image multiplier")
    attention_temperature: PositiveFiniteFloat = Field(
        default=1.0, description="Pooling softmax temperature of every pass"
    )
    batch_size: int = Field(default=4, gt=0, description="Studies per GPU batch")
    num_workers: int = Field(default=8, ge=0, description="Total loader workers")
    passes: list[str] | None = Field(
        default=None, description="Pass names to run; defaults to all thirteen"
    )


def link(target: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        path.unlink()
    path.symlink_to(target)


def build_adapter(settings: Settings, experiment: dict, raw: Path) -> None:
    protocol = json.loads((settings.inputs / "protocol.json").read_text())
    for name, expected in protocol["input_sha256"].items():
        if sha256_file(raw / name) != expected:
            raise ValueError(f"{name} differs from the frozen split source")
    link(raw, settings.adapter / "raw")
    split = settings.adapter / "folds" / experiment["validation"]["split"]
    for name in ("studies.csv", "targets.csv", "protocol.json"):
        link(settings.inputs / name, split / name)
    manifest = json.loads((settings.inputs / settings.label_manifest).read_text())
    records = [
        file
        for source in manifest["sources"]
        for file in source["files"]
        if file.get("label_table") == experiment["labels"]["source"]
    ]
    if len(records) != 1:
        raise ValueError("Expected one label table for the bundle's label source")
    if sha256_file(settings.inputs / "labels.csv") != records[0]["sha256"]:
        raise ValueError("labels.csv differs from the label manifest")
    link(
        settings.inputs / "labels.csv",
        settings.adapter.joinpath(*Path(records[0]["path"]).parts[1:]),
    )


def pass_plan(experiment: dict) -> list[dict]:
    plan = [{"name": "baseline", "slot": None, "scale": None, "series_type": None}]
    for slot, kind in enumerate(experiment["data"]["series_types"]):
        label = f"{kind['plane'].lower()}_{CONTRASTS[kind['fluid_sensitive']]}"
        for scale in SCALES:
            plan.append(
                {
                    "name": f"{label}_{scale}",
                    "slot": slot,
                    "scale": scale,
                    "series_type": label,
                }
            )
    return plan


def summarise_trace(trace_dir: Path, experiment: dict, counts_path: Path) -> dict:
    records = [
        json.loads(line)
        for path in sorted(trace_dir.glob("trace-*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    names = ("sources", "base", "budget", "triplets", "capacity", "requested")
    frame = pl.DataFrame(
        [
            {
                "StudyInstanceUID": record["study"],
                **{
                    f"{name}_{slot}": value
                    for name in names
                    for slot, value in enumerate(record[name])
                },
            }
            for record in records
        ]
    ).sort("StudyInstanceUID")
    frame.write_csv(counts_path)
    types = []
    for slot, kind in enumerate(experiment["data"]["series_types"]):
        present = frame.filter(pl.col(f"sources_{slot}") > 0)
        types.append(
            {
                "slot": slot,
                "series_type": f"{kind['plane'].lower()}_{CONTRASTS[kind['fluid_sensitive']]}",
                "studies_with_type": present.height,
                "mean_triplets_all_studies": float(frame[f"triplets_{slot}"].mean()),
                "mean_triplets_present": float(present[f"triplets_{slot}"].mean()),
                "mean_base_budget_present": float(present[f"base_{slot}"].mean()),
                "mean_budget_present": float(present[f"budget_{slot}"].mean()),
                "studies_below_budget": present.filter(
                    pl.col(f"triplets_{slot}") < pl.col(f"budget_{slot}")
                ).height,
            }
        )
    total = frame.select(
        pl.sum_horizontal(f"triplets_{slot}" for slot in range(len(types))).alias(
            "total"
        )
    )["total"]
    return {
        "traced_studies": frame.height,
        "distinct_studies": frame["StudyInstanceUID"].n_unique(),
        "triplets_total": int(total.sum()),
        "triplets_per_study": float(total.mean()),
        "types": types,
    }


def run_pass(settings: Settings, experiment: dict, entry: dict) -> dict:
    name = entry["name"]
    output = settings.output_dir / f"predictions_{name}.csv"
    trace_dir = settings.adapter.parent / f"{settings.adapter.name}-trace-{name}"
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{settings.patch_dir}:{settings.bundle / 'code'}"
    environment["KNEE_BUDGET_LOG"] = str(trace_dir)
    for key in ("KNEE_BUDGET_SLOT", "KNEE_BUDGET_SCALE"):
        environment.pop(key, None)
    if entry["slot"] is not None:
        environment["KNEE_BUDGET_SLOT"] = str(entry["slot"])
        environment["KNEE_BUDGET_SCALE"] = entry["scale"]
    command = [
        sys.executable,
        "-m",
        "scripts.pipeline.predict",
        str(settings.bundle / "config.json"),
        "--checkpoint",
        str(settings.bundle / "model.ckpt"),
        "--output",
        str(output),
        "--split",
        "validation",
        "--fold",
        str(settings.fold),
        "--data-root",
        str(settings.adapter),
        "--label-manifest",
        str(settings.inputs / settings.label_manifest),
        "--image-multiplier",
        str(settings.image_multiplier),
        "--attention-temperature",
        str(settings.attention_temperature),
        "--batch-size",
        str(settings.batch_size),
        "--num-workers",
        str(settings.num_workers),
        "--device",
        settings.device,
        "--precision",
        settings.precision,
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    seconds = time.perf_counter() - started
    record = {**entry, "seconds": seconds, "returncode": completed.returncode}
    if completed.returncode != 0:
        record["stderr"] = completed.stderr.decode(errors="replace")[-12000:]
        return record
    runtime = json.loads(output.with_suffix(".runtime.json").read_text())
    record.update(
        attention_temperature=runtime["attention_temperature"],
        image_multiplier=runtime["image_multiplier"],
        reflection_tta=runtime["reflection_tta"],
        studies=runtime["studies"],
        device=runtime["device"],
        devices=runtime["devices"],
        precision=runtime["precision"],
        gpu_names=[worker["gpu_name"] for worker in runtime["workers"]],
        worker_wall_seconds=runtime["worker_wall_seconds"],
        prediction_seconds=[
            worker["prediction_seconds"] for worker in runtime["workers"]
        ],
        images=sum(worker["images"] for worker in runtime["workers"]),
        **summarise_trace(
            trace_dir, experiment, settings.output_dir / f"counts_{name}.csv"
        ),
    )
    shutil.rmtree(trace_dir)
    return record


def main() -> None:
    started = time.perf_counter()
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = verify_bundle(settings.bundle)
    experiment = json.loads((settings.bundle / "config.json").read_text())["experiment"]
    raw = settings.raw if settings.raw is not None else discover_kaggle_data_root()
    build_adapter(settings, experiment, raw.expanduser().resolve())
    plan = pass_plan(experiment)
    if settings.passes is not None:
        unknown = set(settings.passes) - {entry["name"] for entry in plan}
        if unknown:
            raise ValueError(f"Unknown passes: {sorted(unknown)}")
        plan = [entry for entry in plan if entry["name"] in settings.passes]
    records = []
    for entry in plan:
        record = run_pass(settings, experiment, entry)
        (settings.output_dir / f"pass_{entry['name']}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        records.append(record)
        print(
            json.dumps(
                {
                    key: record.get(key)
                    for key in (
                        "name",
                        "returncode",
                        "seconds",
                        "precision",
                        "devices",
                        "triplets_total",
                        "stderr",
                    )
                }
            ),
            flush=True,
        )
        if record["returncode"] != 0 and entry["name"] == "baseline":
            break
    summary = {
        "bundle_checkpoint_sha256": manifest["checkpoint"]["exported_sha256"],
        "raw": str(raw),
        "attention_temperature": settings.attention_temperature,
        "image_multiplier": settings.image_multiplier,
        "batch_size": settings.batch_size,
        "num_workers": settings.num_workers,
        "total_seconds": time.perf_counter() - started,
        "failed": [record["name"] for record in records if record["returncode"] != 0],
        "passes": [record["name"] for record in records],
    }
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
