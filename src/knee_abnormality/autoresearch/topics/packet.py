import math
import time
from pathlib import Path, PurePosixPath

from knee_abnormality.autoresearch.campaign.definition import (
    PROTECTED_CONFIG,
    PROTECTED_SOURCES,
    RANGE_BUDGET_CONFIG,
    RECIPE,
    overlaps,
)
from knee_abnormality.autoresearch.reporting.curves import run_curves
from knee_abnormality.autoresearch.reporting.progress import adoption_result
from knee_abnormality.autoresearch.runtime.layout import PREDICTIONS, fold_dir
from knee_abnormality.autoresearch.runtime.storage import read_json
from knee_abnormality.autoresearch.supervisor.context import family_ledger
from knee_abnormality.autoresearch.topics.history import outcome
from knee_abnormality.config import STUDY_HEADERS

DECIMALS = 6


def rounded(value):
    if isinstance(value, float):
        return round(value, DECIMALS) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def stratum(hard, values, targets, mask):
    from knee_abnormality.evaluation.metrics import aucs

    scores = aucs(hard[mask], values[mask], targets, strict=False)["per_target_auc"]
    defined = [score for score in scores.values() if score is not None]
    return {
        "studies": int(mask.sum()),
        "positives": dict(zip(targets, hard[mask].sum(axis=0).tolist(), strict=True)),
        "per_label": scores,
        "macro_over_defined_labels": sum(defined) / len(defined) if defined else None,
    }


def incumbent_profile(root: Path, run: Path, config: dict):
    import numpy as np
    import polars as pl

    from knee_abnormality.data.sources import resolve_raw_dir
    from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD
    from knee_abnormality.evaluation.metrics import aucs, compartment_metrics

    fold = config["validation"]["folds"][0]
    data = Path(config["paths"]["data_root"])
    split = data / "folds" / config["validation"]["split"]
    labels = pl.read_csv(split / "targets.csv")
    targets = labels.columns[1:]
    frame = (
        pl.read_csv(split / "studies.csv")
        .filter(pl.col("fold") == fold)
        .join(labels, on=STUDY, validate="1:1")
        .join(
            pl.read_parquet(root / STUDY_HEADERS).select(STUDY, "vendor"),
            on=STUDY,
            validate="1:1",
            how="left",
        )
        .sort(STUDY)
    )
    predictions = pl.read_csv(fold_dir(run, fold) / PREDICTIONS).sort(STUDY)
    if predictions[STUDY].to_list() != frame[STUDY].to_list():
        raise ValueError(f"Incumbent predictions do not cover fold {fold}: {run}")
    values = predictions.select(targets).to_numpy().astype(float)
    truth = frame.select(targets).to_numpy().astype(float)
    hard = (truth >= TARGET_THRESHOLD).astype(int)
    overall = aucs(hard, values, targets)
    vendors = frame["vendor"].fill_null("unknown").to_numpy()
    series = pl.read_csv(resolve_raw_dir(data, "train") / "train_series.csv")
    kinds = {
        (item["plane"], item["fluid_sensitive"], item["fat_suppression"])
        for item in config["data"]["series_types"]
    }
    present = {
        study: set(zip(planes, fluid, fat, strict=True))
        for study, planes, fluid, fat in series.group_by(STUDY)
        .agg("Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression")
        .iter_rows()
    }
    studies = [present.get(study, set()) for study in frame[STUDY].to_list()]
    planes = sorted({kind[0] for kind in kinds})
    masks = {
        "complete_recipe_series": np.array([kinds <= found for found in studies]),
        **{
            f"missing_plane={plane}": np.array(
                [all(kind[0] != plane for kind in found) for found in studies]
            )
            for plane in planes
        },
        **{
            f"missing_series={plane}/fluid_sensitive={fluid}/fat_suppression={fat}": np.array(
                [(plane, fluid, fat) not in found for found in studies]
            )
            for plane, fluid, fat in sorted(kinds)
        },
    }
    selected = read_json(fold_dir(run, fold) / "selected_checkpoint.json")
    return rounded(
        {
            "run": str(run),
            "fold": fold,
            "split": config["validation"]["split"],
            "labels_source": config["labels"]["source"],
            "prediction_image_multiplier": config["validation"][
                "prediction_image_multiplier"
            ],
            "studies": frame.height,
            "macro_auc": overall["macro_auc"],
            "recorded_macro_auc": selected.get("prediction_macro_auc"),
            "per_label": overall["per_target_auc"],
            "positives": dict(zip(targets, hard.sum(axis=0).tolist(), strict=True)),
            "slices": {
                "vendor": {
                    vendor: stratum(hard, values, targets, vendors == vendor)
                    for vendor in sorted(set(vendors.tolist()))
                },
                "plane": {
                    name: stratum(hard, values, targets, mask)
                    for name, mask in masks.items()
                    if mask.any()
                },
                "compartment": compartment_metrics(truth, values, targets),
            },
        }
    )


def evaluation(state):
    config = state["config"]
    return {
        "split": config["validation"]["split"],
        "labels_source": config["labels"]["source"],
        "fold": config["validation"]["folds"][0],
    }


def ledgers(records, incumbent_run: Path):
    retried = {
        state["ideas"][exp["idea"]]["mechanism_id"]
        for record in records
        if (state := record["state"])
        for exp in state["experiments"].values()
        if exp.get("reference")
        and Path(exp["reference"]["run"]).resolve() == incumbent_run.resolve()
    }
    campaigns, near_misses, closed = [], [], set()
    for record in records:
        state = record["state"]
        if state is None:
            continue
        verdicts = state.get("research", {}).get("verdicts", {})
        closed.update(
            family for family, item in verdicts.items() if item["verdict"] == "close"
        )
        campaigns.append(
            {
                "campaign": record["id"],
                "evaluation": evaluation(state),
                "families": family_ledger(state),
            }
        )
        for exp in state["experiments"].values():
            delta, bounds = adoption_result(exp)
            family = state["ideas"][exp["idea"]]["mechanism_id"]
            if (
                exp.get("decision") in {"reject", "inconclusive"}
                and delta is not None
                and delta > 0
                and family not in retried
            ):
                near_misses.append(
                    {
                        "campaign": record["id"],
                        "idea": exp["idea"],
                        "family": family,
                        "experiment": exp["id"],
                        "macro_delta": delta,
                        "adoption_interval_80": bounds,
                        "evaluation": evaluation(state),
                    }
                )
    return rounded(campaigns), rounded(near_misses), sorted(closed)


def scope(policy):
    return {
        "editable_files": policy["editable_files"],
        "editable_directories": policy["editable_directories"],
        "config_keys": policy["config_keys"],
        "config_prefixes": policy["config_prefixes"],
        "list_item_fields": sorted(policy["list_item_fields"]),
    }


def families(record):
    if record["state"]:
        return sorted(
            {idea["mechanism_id"] for idea in record["state"]["ideas"].values()}
        )
    return []


def past_topics(records, origins):
    rows = []
    for record in records:
        definition, state = record["definition"], record["state"]
        rows.append(
            {
                "id": record["id"],
                "origin": origins.get(record["id"], "user"),
                "status": state["status"] if state else "not_started",
                "evaluation": evaluation(state) if state else None,
                "scope": scope(definition["policy"])
                if "policy" in definition
                else None,
                "definition_error": definition.get("error"),
                "families": families(record),
                "outcome": outcome(state, record["seeded"]) if state else None,
            }
        )
    return rows


def protected_source(path: str):
    path = PurePosixPath(path)
    return any(path.is_relative_to(protected) for protected in PROTECTED_SOURCES)


def covered_source(path: str, policy):
    return path in policy["editable_files"] or any(
        PurePosixPath(path).is_relative_to(directory)
        for directory in policy["editable_directories"]
    )


def covered_config(section: str, policy):
    return any(
        overlaps(section, item)
        for item in (
            *policy["config_keys"],
            *policy["config_prefixes"],
            *policy["list_item_fields"],
        )
    )


def codebase_map(source: Path, config: dict, records):
    policies = [
        record["definition"]["policy"]
        for record in records
        if "policy" in record["definition"]
    ]
    modules = sorted(
        path.relative_to(source).as_posix()
        for path in (source / "src/knee_abnormality").rglob("*.py")
        if "__pycache__" not in path.parts and path.name != "__init__.py"
    )
    sections = sorted(
        f"{top}.{key}" if isinstance(group, dict) else top
        for top, group in config.items()
        for key in (group if isinstance(group, dict) else [top])
    )
    return {
        "unexplored_modules": [
            path
            for path in modules
            if not protected_source(path)
            and not any(covered_source(path, policy) for policy in policies)
        ],
        "unexplored_config_sections": [
            section
            for section in sections
            if not any(
                overlaps(section, item)
                for item in (*PROTECTED_CONFIG, *RANGE_BUDGET_CONFIG)
            )
            and not any(covered_config(section, policy) for policy in policies)
        ],
        "protected_sources": [str(path) for path in PROTECTED_SOURCES],
        "protected_config": list(PROTECTED_CONFIG),
        "range_budget_config": list(RANGE_BUDGET_CONFIG),
        "editable_non_python": [RECIPE],
    }


def build_packet(root: Path, incumbent: dict, records, origins, pool):
    run = Path(incumbent["run"])
    config = read_json(run / "effective_config.json")
    campaigns, near_misses, closed = ledgers(records, run)
    return {
        "created_at": time.time(),
        "incumbent": {
            **incumbent_profile(root, run, config),
            "id": incumbent["id"],
            "campaign": incumbent["campaign"],
            "source": incumbent["source"],
            "seed": config["seed"],
            "curves": run_curves(fold_dir(run, config["validation"]["folds"][0])),
        },
        "campaigns": campaigns,
        "closed_families": closed,
        "near_misses": near_misses,
        "topics": past_topics(records, origins),
        "codebase_map": codebase_map(Path(incumbent["source"]), config, records),
        "seed_pool": pool,
    }


def resolve(packet, pointer: str):
    value = packet
    for part in pointer.split("/")[1:]:
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not key.isdigit() or int(key) >= len(value):
                raise KeyError(pointer)
            value = value[int(key)]
        elif isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise KeyError(pointer)
    return value
