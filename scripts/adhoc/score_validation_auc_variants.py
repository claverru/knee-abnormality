import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field
from pydantic_settings import SettingsError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.splits import STUDY, load_split
from scripts.tools.prepare_combined_labels import consensus_sources
from scripts.tools.validation import scanner_groups

STEVEN_V4 = "data/labels/public/stevenleehans/version-6/llm_labels_v4_blend.csv"
LABEL_TABLES = {
    "steven_v4": STEVEN_V4,
    "combined_v1": "data/labels/prepared/combined-v1/labels.csv",
}
HEADERS = "outputs/analysis/s/sharing-disagreement-20260914/study_headers.parquet"
COHORTS = {
    "v3_x2": (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 23, 24, 25, 26, 27),
    "v3_x1_tau075": (32, 33, 34, 38, 39),
    "scanner_x1": (40, 41, 42, 43, 44, 45, 46, 48),
}
EXTRA = {48: ("outputs/training/k05-clean/fold_0", 8, 0.942)}
REFERENCE = "hard05"
STRATA = ("scanner_group", "vendor", "field", "machine", "thickness", "pixel_spacing", "sex")
HEDGES = r"possibl|cannot be excluded|may represent|likely|questionabl|suspect|probabl|equivocal|subtle|suggest"


class VariantsCLI(CLISettings):
    ledger: Path = Path("outputs/submission/kaggle-ledger.json")
    raw_dir: Path = Path("data/raw")
    folds_dir: Path = Path("data/folds")
    comparison_manifest: Path = Path(
        "configs/labels/public-label-comparison-sources.json"
    )
    output_dir: Path = Path("outputs/analysis/validation-auc-variants")
    permutations: int = Field(default=20000, ge=1000)
    seed: int = 20260923


def submissions(settings: VariantsCLI) -> dict[int, tuple[Path, int, float]]:
    result = {}
    for row in json.loads(settings.ledger.read_text()):
        manifest = Path(row["bundle"]) / "manifest.json"
        if row["public_score"] is None or not manifest.exists():
            continue
        source = json.loads(manifest.read_text()).get("checkpoint", {}).get(
            "source_path"
        )
        if source is None:
            continue
        source = Path(source)
        run = next(
            (
                path
                for path in (
                    source.parent,
                    Path(str(source.parent).replace("/outputs/", "/outputs/training/")),
                )
                if path.exists()
            ),
            None,
        )
        match = re.search(r"epoch=(\d+)", source.name)
        if run is not None and match:
            result[row["sequence"]] = (run, int(match.group(1)), float(row["public_score"]))
    for sequence, (run, epoch, public) in EXTRA.items():
        result[sequence] = (Path(run), epoch, public)
    return result


def sharpen(target: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(target, 1e-4, 1 - 1e-4)
    logit = np.log(clipped) - np.log1p(-clipped)
    return 1 / (1 + np.exp(-logit / temperature))


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-3, 1 - 1e-3)
    return np.log(clipped) - np.log1p(-clipped)


def gold_calibration(
    settings: VariantsCLI, split, targets: list[str]
) -> tuple[float, float]:
    official = pl.read_csv(settings.raw_dir / "train.csv", infer_schema_length=None)
    gold = (
        split.membership.filter(pl.col("is_gold") & (pl.col("fold") != 0))
        .select(STUDY)
        .join(official.select(STUDY, *targets).unique(STUDY), on=STUDY)
    )
    table = split.protocol["label_source"]["label_table"]
    source = pl.read_csv(LABEL_TABLES[table], infer_schema_length=None).select(
        pl.col(STUDY).cast(pl.String), *[pl.col(t).cast(pl.Float64) for t in targets]
    )
    joined = gold.join(source, on=STUDY, suffix="_source")
    truth = joined.select(targets).to_numpy().ravel().astype(float)
    report = joined.select([f"{t}_source" for t in targets]).to_numpy().ravel()
    keep = np.isfinite(truth) & np.isfinite(report)
    model = LogisticRegression(C=1e6).fit(logit(report[keep])[:, None], truth[keep])
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def acquisition_strata() -> pl.DataFrame:
    headers = pl.read_parquet(HEADERS)
    field = pl.col("MagneticFieldStrength").cast(pl.Float64, strict=False)
    spacing = (
        pl.col("PixelSpacing").str.strip_chars("[]").str.split(",").list.first()
        .str.strip_chars().cast(pl.Float64, strict=False)
    )
    thickness = pl.col("SliceThickness").cast(pl.Float64, strict=False)
    return headers.select(
        STUDY,
        pl.col("vendor"),
        pl.when(field.is_null()).then(pl.lit("unknown"))
        .when(field >= 2).then(pl.lit("3T"))
        .when(field >= 1.4).then(pl.lit("1.5T"))
        .otherwise(pl.lit("low")).alias("field"),
        pl.col("ManufacturerModelName").alias("machine"),
        pl.when(thickness < 3).then(pl.lit("thin")).when(thickness <= 3.5)
        .then(pl.lit("mid")).otherwise(pl.lit("thick")).alias("thickness"),
        pl.when(spacing < 0.3).then(pl.lit("fine")).when(spacing < 0.45)
        .then(pl.lit("mid")).otherwise(pl.lit("coarse")).alias("pixel_spacing"),
        pl.col("PatientSex").alias("sex"),
    ).join(scanner_groups(headers), on=STUDY)


def study_properties(settings: VariantsCLI, membership: pl.DataFrame) -> pl.DataFrame:
    reports = pl.read_csv(settings.raw_dir / "train.csv", infer_schema_length=None).select(
        STUDY,
        pl.col("Report").str.len_chars().log().alias("log_report_length"),
        pl.col("Report").str.to_lowercase().str.count_matches(HEDGES).alias("hedges"),
    )
    series = pl.read_csv(settings.raw_dir / "train_series.csv").group_by(STUDY).agg(
        pl.len().alias("series")
    )
    properties = (
        membership.select(STUDY, "is_gold", "report_group")
        .join(reports, on=STUDY, how="left")
        .join(series, on=STUDY, how="left")
        .with_columns(pl.col("series").fill_null(0))
    )
    features = properties.select("log_report_length", "hedges", "series").to_numpy()
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    model = LogisticRegression(C=1.0).fit(features, properties["is_gold"].to_numpy())
    propensity = model.predict_proba(features)[:, 1]
    return properties.with_columns(
        pl.Series("gold_odds", propensity / (1 - propensity)),
        pl.col("report_group").count().over("report_group").alias("group_size"),
    )


def soft_weights(target: np.ndarray) -> np.ndarray:
    return np.outer(target, 1 - target)


def hard_weights(target: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    positive = target >= threshold
    return np.outer(positive, ~positive).astype(np.float64)


def balance(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return 1 / counts[inverse]


def stratified(base: dict, strata: dict) -> dict:
    result = {}
    for axis, values in strata.items():
        same = values[:, None] == values[None, :]
        balanced = np.outer(balance(values), balance(values))
        for name, weight in base.items():
            result[f"{name}_same_{axis}"] = weight * same
            result[f"{name}_across_{axis}"] = weight * ~same
            result[f"{name}_balanced_{axis}"] = weight * balanced
    return result


def pair_weights(
    target: np.ndarray, gold: np.ndarray, study: dict, strata: dict, agreement: np.ndarray
) -> dict:
    positive = target >= 0.5
    cross = np.outer(positive, ~positive)
    soft = soft_weights(target)
    confident = (target <= 0.2) | (target >= 0.8)
    return {
        "hard05": cross.astype(np.float64),
        "hard03": hard_weights(target, 0.3),
        "hard07": hard_weights(target, 0.7),
        "confident20": cross * np.outer(confident, confident),
        "soft": soft,
        "soft_pow05": np.sqrt(soft),
        "soft_pow2": soft**2,
        "soft_cross_only": soft * cross,
        "soft_within_only": soft * ~cross,
        "soft_sharpen_t05": soft_weights(sharpen(target, 0.5)),
        "soft_flatten_t2": soft_weights(sharpen(target, 2.0)),
        "gap": np.maximum(target[:, None] - target[None, :], 0),
        "kendall_gap005": (target[:, None] - target[None, :] > 0.05).astype(np.float64),
        "soft_gold_x10": soft * np.where(np.add.outer(gold, gold) > 0, 10.0, 1.0),
        "hard05_extractor_agreement": cross * np.outer(agreement, agreement),
        "soft_extractor_agreement": soft * np.outer(agreement, agreement),
        **{
            f"{base}_{name}": weight * np.outer(study[name], study[name])
            for name in ("gold_odds", "report_length", "report_dedup")
            for base, weight in (("hard05", cross.astype(np.float64)), ("soft", soft))
        },
        **stratified({"hard05": cross.astype(np.float64), "soft": soft}, strata),
    }


def prepared(weights: dict) -> dict:
    result = {}
    for name, weight in weights.items():
        weight = weight.astype(np.float32)
        np.fill_diagonal(weight, 0)
        result[name] = (weight, float(weight.sum(dtype=np.float64)))
    return result


def labels_for(settings: VariantsCLI, split_name: str) -> dict:
    split = load_split(settings.raw_dir, settings.folds_dir / split_name)
    targets = split.protocol["targets"]
    evaluation = (
        split.evaluation.join(split.membership.select(STUDY, "is_gold", "fold"), on=STUDY)
        .filter(pl.col("fold") == 0)
        .join(acquisition_strata(), on=STUDY, how="left")
        .sort(STUDY)
    )
    _, sources = consensus_sources(
        json.loads(settings.comparison_manifest.read_text()), targets
    )
    steven = pl.read_csv(STEVEN_V4, infer_schema_length=None).select(
        pl.col(STUDY).cast(pl.String), *[pl.col(t).cast(pl.Float64) for t in targets]
    )
    stacked = evaluation.select(STUDY)
    for name, source in {"steven_v4": steven, **sources}.items():
        stacked = stacked.join(
            source.select(STUDY, *[pl.col(t).alias(f"{name}|{t}") for t in targets]),
            on=STUDY,
            how="left",
            validate="1:1",
        )
    names = ["steven_v4", *sources]
    gold = evaluation["is_gold"].to_numpy().astype(bool)
    frozen = evaluation.select(targets).to_numpy().astype(float)

    def consensus(members: list[str]) -> np.ndarray:
        values = stacked.select(
            pl.mean_horizontal([f"{n}|{t}" for n in members]).alias(t) for t in targets
        ).to_numpy()
        values[gold] = frozen[gold]
        return values

    intercept, slope = gold_calibration(settings, split, targets)
    calibrated = 1 / (1 + np.exp(-(intercept + slope * logit(frozen))))
    calibrated[gold] = frozen[gold]
    print(f"{split_name} gold calibration a={intercept:.3f} b={slope:.3f}", file=sys.stderr)
    tables = {"": frozen, "consensus10_": consensus(names), "independent9_": consensus(names[1:])}
    properties = evaluation.select(STUDY).join(
        study_properties(settings, split.membership), on=STUDY, how="left"
    )
    study = {
        "gold_odds": properties["gold_odds"].to_numpy(),
        "report_length": np.exp(properties["log_report_length"].to_numpy()),
        "report_dedup": 1 / properties["group_size"].to_numpy(),
    }
    strata = {
        axis: evaluation[axis].fill_null("missing").cast(pl.String).to_numpy()
        for axis in STRATA
    }
    strata["series_count"] = np.digitize(properties["series"].to_numpy(), [4.5, 5.5, 6.5]).astype(str)
    spread = np.stack(
        [
            stacked.select(
                pl.concat_list([f"{n}|{t}" for n in names]).list.drop_nulls().list.std()
            ).to_series().to_numpy()
            for t in targets
        ],
        axis=1,
    )
    agreement = np.clip(1 - 2 * np.nan_to_num(spread, nan=0.0), 0.05, 1)
    agreement[gold] = 1
    weights = []
    for k in range(len(targets)):
        label = pair_weights(frozen[:, k], gold, study, strata, agreement[:, k])
        for prefix in ("consensus10_", "independent9_"):
            column = tables[prefix][:, k]
            label[f"{prefix}hard05"] = hard_weights(column)
            label[f"{prefix}soft"] = soft_weights(column)
        label["gold_calibrated_soft"] = soft_weights(calibrated[:, k])
        weights.append(prepared(label))
    return {
        "targets": targets,
        "studies": evaluation[STUDY].to_list(),
        "gold": gold,
        "frozen": frozen,
        "weights": weights,
    }


def predictions(path: Path, labels: dict) -> np.ndarray:
    frame = pl.read_csv(path).sort(STUDY)
    if frame[STUDY].to_list() != labels["studies"]:
        raise ValueError(f"Validation studies differ from fold 0: {path}")
    return frame.select(labels["targets"]).to_numpy().astype(float)


def per_label(
    score: np.ndarray, labels: dict, names: tuple[str, ...] | None = None
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for k, weights in enumerate(labels["weights"]):
        column = score[:, k]
        above = (column[:, None] > column[None, :]).astype(np.float32)
        above += 0.5 * (column[:, None] == column[None, :])
        for name, (weight, norm) in weights.items():
            if names is None or name in names:
                value = float(np.einsum("ij,ij->", weight, above, dtype=np.float64)) / norm
                result.setdefault(name, []).append(value)
    return result


def variants(score: np.ndarray, labels: dict) -> dict[str, float]:
    frozen, gold = labels["frozen"], labels["gold"]
    aucs = per_label(score, labels)
    values = {name: float(np.mean(v)) for name, v in aucs.items()}
    values["soft_median_label"] = float(np.median(aucs["soft"]))
    values["micro"] = float(roc_auc_score((frozen >= 0.5).ravel(), score.ravel()))
    values["report_only"] = float(
        np.mean(
            [roc_auc_score(frozen[~gold, k] >= 0.5, score[~gold, k]) for k in range(12)]
        )
    )
    values["partial_fpr20"] = float(
        np.mean([roc_auc_score(frozen[:, k] >= 0.5, score[:, k], max_fpr=0.2) for k in range(12)])
    )
    values["spearman_target"] = float(
        np.mean(
            [
                np.corrcoef(
                    pl.Series(score[:, k]).rank().to_numpy(),
                    pl.Series(frozen[:, k]).rank().to_numpy(),
                )[0, 1]
                for k in range(12)
            ]
        )
    )
    probability = np.clip(score, 1e-7, 1 - 1e-7)
    values["neg_soft_bce"] = float(
        (frozen * np.log(probability) + (1 - frozen) * np.log1p(-probability)).mean()
    )
    return values


def ranks(values: np.ndarray) -> np.ndarray:
    return pl.Series(values).rank("average").to_numpy()


def concordance(local: np.ndarray, public: np.ndarray) -> tuple[int, int]:
    i, j = np.triu_indices(len(local), 1)
    ranked = (local[i] != local[j]) & (public[i] != public[j])
    agree = (local[i] > local[j]) == (public[i] > public[j])
    return int((agree & ranked).sum()), int(ranked.sum())


def score(settings: VariantsCLI) -> None:
    if settings.output_dir.exists():
        raise ValueError(f"Output directory already exists: {settings.output_dir}")
    runs = submissions(settings)
    labels = {}
    rows = []
    for cohort, sequences in COHORTS.items():
        for sequence in sequences:
            run, epoch, public = runs[sequence]
            split = json.loads((run.parent / "effective_config.json").read_text())[
                "validation"
            ]["split"]
            if split not in labels:
                labels[split] = labels_for(settings, split)
            epochs = run / "epochs"
            selected = predictions(epochs / f"validation_epoch_{epoch:03d}.csv", labels[split])
            values = variants(selected, labels[split])
            last = [
                predictions(path, labels[split])
                for path in sorted(epochs.glob("validation_epoch_*.csv"))[-3:]
            ]
            pair = ("hard05", "soft")
            last_values = [per_label(p, labels[split], pair) for p in last]
            ensemble = per_label(np.mean(last, axis=0), labels[split], pair)
            for name in pair:
                values[f"{name}_last3_mean"] = float(
                    np.mean([np.mean(v[name]) for v in last_values])
                )
                values[f"{name}_last3_ensemble"] = float(np.mean(ensemble[name]))
            rows.append(
                {"cohort": cohort, "sequence": sequence, "run": str(run), "epoch": epoch,
                 "public": public, **values}
            )
            print(f"scored S{sequence}", file=sys.stderr)
    frame = pl.DataFrame(rows)
    metrics = [c for c in frame.columns if c not in ("cohort", "sequence", "run", "epoch", "public")]
    parts = [frame.filter(pl.col("cohort") == c) for c in COHORTS]
    local = {m: [p[m].to_numpy() for p in parts] for m in metrics}
    public = [p["public"].to_numpy() for p in parts]

    def total(metric: str, targets: list[np.ndarray]) -> int:
        return sum(concordance(x, y)[0] for x, y in zip(local[metric], targets))

    observed = {m: total(m, public) - total(REFERENCE, public) for m in metrics}
    generator = np.random.default_rng(settings.seed)
    exceed = {m: 0 for m in metrics}
    exceed_max = {m: 0 for m in metrics}
    for _ in range(settings.permutations):
        shuffled = [generator.permutation(y) for y in public]
        base = total(REFERENCE, shuffled)
        gains = {m: total(m, shuffled) - base for m in metrics}
        best = max(gains[m] for m in metrics if m != REFERENCE)
        for m in metrics:
            exceed[m] += gains[m] >= observed[m]
            exceed_max[m] += best >= observed[m]
    summary = {}
    for metric in metrics:
        summary[metric] = {
            "concordant": total(metric, public),
            "ranked": sum(concordance(x, y)[1] for x, y in zip(local[metric], public)),
            "gain_vs_reference": observed[metric],
            "p_gain": (exceed[metric] + 1) / (settings.permutations + 1),
            "p_gain_max_corrected": (exceed_max[metric] + 1) / (settings.permutations + 1),
            "cohorts": {},
        }
        for cohort, x, y in zip(COHORTS, local[metric], public):
            concordant, ranked = concordance(x, y)
            summary[metric]["cohorts"][cohort] = {
                "pearson": float(np.corrcoef(x, y)[0, 1]),
                "spearman": float(np.corrcoef(ranks(x), ranks(y))[0, 1]),
                "concordant": concordant,
                "ranked": ranked,
            }
    settings.output_dir.mkdir(parents=True)
    frame.write_csv(settings.output_dir / "submissions.csv")
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for metric in sorted(metrics, key=lambda m: -summary[metric]["concordant"]):
        s = summary[metric]
        cells = " | ".join(
            f"{c}: r={v['pearson']:+.3f} rho={v['spearman']:+.3f} {v['concordant']}/{v['ranked']}"
            for c, v in s["cohorts"].items()
        )
        print(
            f"{metric:28s} {s['concordant']:3d} p={s['p_gain']:.3f} "
            f"pmax={s['p_gain_max_corrected']:.3f} | {cells}"
        )


if __name__ == "__main__":
    try:
        score(VariantsCLI())
    except SettingsError as error:
        sys.exit(str(error))
