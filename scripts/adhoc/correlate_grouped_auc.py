import json
import re
from itertools import combinations, product
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD
from knee_abnormality.evaluation.metrics import soft_auc
from scripts.adhoc.correlate_submission_metrics import label_sets, submitted_checkpoint
from scripts.tools.validation import MAJOR_VENDORS, scanner_groups

HEADERS = Path("outputs/analysis/sharing-disagreement-20260914/study_headers.parquet")
CURRENT = "rebuild_state_v2|hard|pooled"
SCANNER_GROUP_SOFT = "rebuild_state_v2|soft|scanner_group|equal|m5"
REFERENCES = [
    CURRENT,
    "rebuild_state_v2|soft|pooled",
    "rebuild_state_v2|hard|major",
    SCANNER_GROUP_SOFT,
    "rebuild_state_v2|soft|scanner_group|equal|m10",
]
GROUPINGS = {
    "pooled": ("all", False),
    "major": ("all", True),
    "site": ("site", False),
    "vendor3": ("site", True),
    "scanner_group": ("scanner_group", False),
    "scanner_group_major": ("scanner_group", True),
    "field": ("field", False),
    "vendor": ("vendor", False),
    "scanner": ("scanner", False),
    "scanner_major": ("scanner", True),
}
WEIGHTS = ("equal", "sqrt", "size")
MIN_CLASS = (3, 5, 10, 20)
TARGET_TYPES = ("hard", "soft")
ADVERSARIAL = {
    "half-stack-control": "outputs/training/half-stack-control",
    "gradient-reversal": "outputs/autoresearch/auxiliary-signals/experiments/scanner-group-adversarial-head-20260926T141522Z-0818ece2/training",
    "gradient-reversal-full-head": "outputs/autoresearch/auxiliary-signals/experiments/scanner-adversarial-decoupled-weight-20260926T152823Z-d509df07/training",
    "confusion-head": "outputs/autoresearch/auxiliary-signals/experiments/scanner-confusion-detached-head-20260926T165429Z-55c4db64/training",
    "confusion-stronger-head": "outputs/autoresearch/auxiliary-signals/experiments/scanner-confusion-stronger-head-20260926T191359Z-8e1f5021/training",
    "confusion-backbone-features": "outputs/autoresearch/auxiliary-signals/experiments/scanner-confusion-backbone-features-20260926T191751Z-045f7078/training",
    "confusion-scanner-model": "outputs/autoresearch/auxiliary-signals/experiments/scanner-model-confusion-target-20260926T202630Z-e9304533/training",
    "site-bias": "outputs/autoresearch/auxiliary-signals/experiments/site-bias-product-of-experts-20260926T195143Z-21b4b14d/training",
    "model-bias": "outputs/autoresearch/auxiliary-signals/experiments/model-bias-product-of-experts-20260926T224928Z-44f7ac95/training",
    "site-bias-model-confusion": "outputs/autoresearch/auxiliary-signals/experiments/site-bias-with-model-confusion-20260926T233021Z-841d86a2/training",
}


NARROW = [
    f"rebuild_state_v2|soft|{grouping}" + ("" if grouping in ("pooled", "major") else "|equal|m5")
    for grouping in GROUPINGS
]


class GroupedAUCCLI(CLISettings):
    model_config = describe(
        "Search AUC variants averaged over site, scanner group, scanner model, vendor and field "
        "strength for the one that best orders submissions like the public leaderboard, estimate "
        "the search out of sample by leave-one-cohort-out selection and a max-statistic "
        "permutation test, and score the scanner adversarial runs on the chosen variants"
    )

    ledger: Path = Field(default=Path("outputs/submission/kaggle-ledger.json"), description="Machine submission ledger")
    raw_dir: Path = Field(default=Path("data/raw"), description="Official competition files")
    folds_dir: Path = Field(default=Path("data/folds"), description="Frozen split directories")
    output_dir: Path = Field(description="Fresh output directory")
    permutations: int = Field(default=2000, ge=100, description="Within-cohort public-score permutations")
    bootstrap: int = Field(default=500, ge=100, description="Report-group bootstrap draws for the adversarial deltas")
    top: int = Field(default=20, ge=2, description="Base variants whose pairwise averages are searched")
    decisive: float = Field(default=0.003, gt=0, description="Public difference counted as beyond seed noise")
    seed: int = Field(default=20260929, description="Permutation and bootstrap seed")


def study_groups() -> pl.DataFrame:
    headers = pl.read_parquet(HEADERS).unique(STUDY, keep="first")
    field = pl.col("MagneticFieldStrength").cast(pl.Float64, strict=False)
    return headers.select(
        STUDY,
        pl.lit("all").alias("all"),
        pl.col("vendor").is_in(MAJOR_VENDORS).alias("major"),
        pl.when(pl.col("vendor").is_in(MAJOR_VENDORS)).then(pl.col("vendor")).otherwise(pl.lit("minor")).alias("site"),
        pl.col("vendor"),
        pl.when(field.is_null()).then(pl.lit("unknown")).when(field >= 2).then(pl.lit("3T"))
        .when(field >= 1.4).then(pl.lit("1.5T")).otherwise(pl.lit("low")).alias("field"),
        pl.col("ManufacturerModelName").fill_null("unknown").alias("scanner"),
    ).join(scanner_groups(headers), on=STUDY)


def target_tables(settings: GroupedAUCCLI) -> tuple[dict[str, pl.DataFrame], list[str]]:
    frames, targets, _ = label_sets(settings)
    stacked = pl.concat([frame.select(STUDY, *targets) for frame in frames.values()])
    frames["mean3"] = stacked.group_by(STUDY).agg(pl.col(targets).mean())
    return frames, targets


def label_cells(probability, soft, groups, grouping) -> dict:
    column, major_only = GROUPINGS[grouping]
    keep = groups["major"] if major_only else np.ones(len(soft), bool)
    names = groups[column]
    cells = {"auc": {"hard": [], "soft": []}, "positives": [], "negatives": [], "size": []}
    hard = (soft >= TARGET_THRESHOLD).astype(np.float64)
    for group in np.unique(names[keep]):
        mask = keep & (names == group)
        positives = hard[mask].sum(axis=0)
        cells["positives"].append(positives)
        cells["negatives"].append(mask.sum() - positives)
        cells["size"].append(mask.sum())
        for kind, target in (("hard", hard), ("soft", soft)):
            cells["auc"][kind].append([
                soft_auc(target[mask, k], probability[mask, k]) if 0 < positives[k] < mask.sum() else np.nan
                for k in range(soft.shape[1])
            ])
    return {
        "auc": {kind: np.array(values) for kind, values in cells["auc"].items()},
        "positives": np.array(cells["positives"]),
        "negatives": np.array(cells["negatives"]),
        "size": np.array(cells["size"], dtype=np.float64),
    }


def variant_names() -> list[tuple]:
    variants = []
    for labels, kind in product(("rebuild_state_v2", "steven_v4", "combined_v1", "mean3"), TARGET_TYPES):
        for grouping in GROUPINGS:
            if grouping in ("pooled", "major"):
                variants.append((labels, kind, grouping, "equal", 1))
            else:
                variants.extend((labels, kind, grouping, weight, m) for weight, m in product(WEIGHTS, MIN_CLASS))
    return variants


def variant_key(variant: tuple) -> str:
    labels, kind, grouping, weight, m = variant
    return f"{labels}|{kind}|{grouping}" if grouping in ("pooled", "major") else f"{labels}|{kind}|{grouping}|{weight}|m{m}"


def aggregate(cells: dict, kind: str, weight: str, m: int) -> float:
    auc = cells["auc"][kind]
    valid = (cells["positives"] >= m) & (cells["negatives"] >= m) & np.isfinite(auc)
    size = cells["size"][:, None] * np.ones_like(auc)
    w = {"equal": np.ones_like(auc), "sqrt": np.sqrt(size), "size": size}[weight] * valid
    per_label = np.nansum(np.where(valid, auc, 0) * w, axis=0) / np.where(w.sum(axis=0) > 0, w.sum(axis=0), np.nan)
    return float(np.nanmean(per_label)) if np.isfinite(per_label).any() else np.nan


def metric_values(predictions, frames, targets, groups, variants, index=None) -> dict:
    result, cache = {}, {}
    needed = {(v[0], v[2]) for v in variants}
    for labels in sorted({labels for labels, _ in needed}):
        joined = predictions.join(frames[labels].rename({t: f"y_{t}" for t in targets}), on=STUDY).join(groups, on=STUDY).sort(STUDY)
        if index is not None:
            joined = joined[index]
        arrays = {c: joined[c].to_numpy() for c in ("all", "major", "site", "vendor", "field", "scanner", "scanner_group")}
        probability = joined.select(targets).to_numpy().astype(np.float64)
        soft = joined.select([f"y_{t}" for t in targets]).to_numpy().astype(np.float64)
        for grouping in [g for l, g in needed if l == labels]:
            cache[labels, grouping] = label_cells(probability, soft, arrays, grouping)
    for variant in variants:
        labels, kind, grouping, weight, m = variant
        result[variant_key(variant)] = aggregate(cache[labels, grouping], kind, weight, m)
    return result


def prediction_sources(run: Path, fold: int, epoch: int, validation_multiplier, is_selected: bool, prediction_multiplier) -> dict:
    sources = {"val": run / f"fold_{fold}/epochs/validation_epoch_{epoch:03d}.csv"}
    if is_selected and prediction_multiplier == 4:
        sources["x4"] = run / "oof_predictions.csv"
    return {name: path for name, path in sources.items() if path.exists()}


def submission_rows(settings: GroupedAUCCLI) -> list[dict]:
    rows = []
    for entry in json.loads(settings.ledger.read_text()):
        row = submitted_checkpoint(entry)
        if row is not None:
            rows.append(row)
    return rows


def cohort_pairs(table: pl.DataFrame, source: str) -> list[tuple]:
    frame = table.filter(pl.col("source") == source)
    key = ["split", "fold", "inference"] + (["validation_multiplier"] if source == "val" else [])
    pairs = []
    for (family, members) in frame.group_by(key, maintain_order=True):
        rows = members["row"].to_list()
        for a, b in combinations(rows, 2):
            pairs.append((a, b, f"{family[0]}/f{family[1]}/{family[2]}"))
    return pairs


def weighted_concordance(signs: np.ndarray, public_difference: np.ndarray) -> np.ndarray:
    return 0.5 + 0.5 * (signs @ public_difference) / np.abs(public_difference).sum()


def decisive_concordance(signs: np.ndarray, public_difference: np.ndarray, decisive: float) -> tuple[float, int]:
    mask = np.abs(public_difference) >= decisive - 1e-9
    return float(0.5 + 0.5 * (signs[mask] @ np.sign(public_difference[mask])) / mask.sum()), int(mask.sum())


def agreement(signs: np.ndarray, public_difference: np.ndarray) -> str:
    nonzero = public_difference != 0
    return f"{int((signs[nonzero] == np.sign(public_difference[nonzero])).sum())}/{int(nonzero.sum())}"


def search(differences, public_difference, members, top, combos) -> tuple[list[str], float, np.ndarray, list[str]]:
    names = list(members)
    base = weighted_concordance(np.sign(np.stack([differences[n] for n in names])), public_difference)
    candidates = [[n] for n in names]
    scores = [base]
    if combos:
        best = [names[i] for i in np.argsort(-base)[:top]]
        pairs = [[a, b] for a, b in combinations(best, 2)]
        candidates += pairs
        scores.append(weighted_concordance(np.sign(np.stack([differences[a] + differences[b] for a, b in pairs])), public_difference))
    scores = np.concatenate(scores)
    index = int(np.argmax(scores))
    return candidates[index], float(scores[index]), scores, [" + ".join(c) for c in candidates]


def combined(differences, components) -> np.ndarray:
    return sum(differences[c] for c in components)


def permuted_differences(table, pairs, generator, count) -> list[np.ndarray]:
    submission_of = table["submission"].to_numpy()
    by_submission = {int(s): p for s, p in zip(submission_of, table["public"].to_numpy())}
    cohort_submissions = {}
    for a, b, family in pairs:
        cohort_submissions.setdefault(family, set()).update((int(submission_of[a]), int(submission_of[b])))
    result = []
    for _ in range(count):
        mapping = {}
        for family, submissions in cohort_submissions.items():
            ordered = sorted(submissions)
            mapping |= {(family, s): by_submission[t] for s, t in zip(ordered, generator.permutation(ordered))}
        result.append(np.array([mapping[(f, int(submission_of[a]))] - mapping[(f, int(submission_of[b]))] for a, b, f in pairs]))
    return result


def run(settings: GroupedAUCCLI) -> None:
    if settings.output_dir.exists():
        raise ValueError(f"Output directory already exists: {settings.output_dir}")
    frames, targets = target_tables(settings)
    groups = study_groups()
    variants = variant_names()
    names = [variant_key(v) for v in variants]
    records = []
    for row in submission_rows(settings):
        sources = prediction_sources(Path(row["run"]), row["fold"], row["epoch"], row["validation_multiplier"], row["is_selected"], row["prediction_multiplier"])
        for source, path in sources.items():
            predictions = pl.read_csv(path).select(STUDY, *targets)
            records.append(row | {"source": source} | metric_values(predictions, frames, targets, groups, variants))
    table = pl.DataFrame(records, infer_schema_length=None).with_row_index("row")
    settings.output_dir.mkdir(parents=True)
    table.write_parquet(settings.output_dir / "submission_metrics.parquet")

    pairs = [(a, b, family) for source in ("val", "x4") for a, b, family in cohort_pairs(table, source)]
    left, right = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    public = table["public"].to_numpy()
    public_difference = public[left] - public[right]
    source_of = table["source"].to_numpy()[left]
    family_of = np.array([p[2] for p in pairs])
    cohort = ["source", "split", "fold", "inference", "validation_multiplier"]
    scale = table.select((pl.col(names) - pl.col(names).mean().over(cohort)).std()).row(0, named=True)
    differences = {}
    for name in names:
        values = table[name].to_numpy()
        difference = (values[left] - values[right]) / scale[name]
        if np.isfinite(difference).all():
            differences[name] = difference
    families = {
        "wide search": (list(differences), True),
        "wide search, single variants": (list(differences), False),
        "grouping only (rebuild soft, equal, m5)": ([n for n in NARROW if n in differences], False),
    }
    permutations = permuted_differences(table, pairs, np.random.default_rng(settings.seed), settings.permutations)

    _, _, scores, labels = search(differences, public_difference, families["wide search"][0], settings.top, True)
    order = np.argsort(-scores)
    ranking = []
    for label in list(dict.fromkeys([labels[i] for i in order[:40]] + REFERENCES)):
        d = combined(differences, label.split(" + "))
        entry = {"variant": label, "weighted": float(weighted_concordance(np.sign(d), public_difference))}
        for source in ("val", "x4"):
            mask = source_of == source
            entry[f"{source}_weighted"] = float(weighted_concordance(np.sign(d[mask]), public_difference[mask]))
            entry[f"{source}_decisive"], entry[f"{source}_decisive_pairs"] = decisive_concordance(np.sign(d[mask]), public_difference[mask], settings.decisive)
        ranking.append(entry)
    ranking = pl.DataFrame(ranking)
    ranking.write_csv(settings.output_dir / "ranking.csv")

    current = np.sign(differences[CURRENT])
    family_rows, loco_rows = [], []
    for family_name, (members, combos) in families.items():
        chosen, in_sample, _, _ = search(differences, public_difference, members, settings.top, combos)
        gain = in_sample - weighted_concordance(current, public_difference)
        exceed = sum(
            search(differences, permuted, members, settings.top, combos)[1] - weighted_concordance(current, permuted) >= gain - 1e-12
            for permuted in permutations
        )
        numerator = weight = 0.0
        for held_out in sorted(set(family_of)):
            train, test = family_of != held_out, family_of == held_out
            if np.abs(public_difference[test]).sum() == 0:
                continue
            picked = search({k: v[train] for k, v in differences.items()}, public_difference[train], members, settings.top, combos)[0]
            held = public_difference[test]
            score = float(weighted_concordance(np.sign(combined(differences, picked)[test]), held))
            numerator += score * np.abs(held).sum()
            weight += np.abs(held).sum()
            loco_rows.append({"family": family_name, "held_out": held_out, "chosen": " + ".join(picked), "held_out_weighted": score,
                              "current_weighted": float(weighted_concordance(current[test], held)), "public_weight": float(np.abs(held).sum())})
        family_rows.append({"family": family_name, "candidates": len(members), "chosen_on_all": " + ".join(chosen), "in_sample": in_sample,
                            "max_statistic_p": (exceed + 1) / (len(permutations) + 1), "leave_one_cohort_out": numerator / weight})
    for reference in REFERENCES:
        d = np.sign(differences[reference])
        family_rows.append({"family": "fixed", "candidates": 1, "chosen_on_all": reference,
                            "in_sample": float(weighted_concordance(d, public_difference)), "max_statistic_p": None,
                            "leave_one_cohort_out": float(weighted_concordance(d, public_difference))})
    families_table = pl.DataFrame(family_rows)
    families_table.write_csv(settings.output_dir / "families.csv")
    pl.DataFrame(loco_rows).write_csv(settings.output_dir / "leave_one_cohort_out.csv")

    sg, soft = np.sign(differences[SCANNER_GROUP_SOFT]), np.sign(differences["rebuild_state_v2|soft|pooled"])
    observed = weighted_concordance(sg, public_difference) - weighted_concordance(soft, public_difference)
    pairwise_p = (sum(weighted_concordance(sg, p) - weighted_concordance(soft, p) >= observed - 1e-12 for p in permutations) + 1) / (len(permutations) + 1)

    cohort_rows = []
    for held_out in sorted(set(family_of)):
        for source in ("val", "x4"):
            mask = (family_of == held_out) & (source_of == source)
            if mask.any() and np.abs(public_difference[mask]).sum() > 0:
                cohort_rows.append({"cohort": held_out, "source": source} | {
                    reference: agreement(np.sign(differences[reference][mask]), public_difference[mask]) for reference in REFERENCES
                })
    pl.DataFrame(cohort_rows).write_csv(settings.output_dir / "cohorts.csv")

    sensitivity = []
    for labels_name, kind, grouping, weight, m in variants:
        key = variant_key((labels_name, kind, grouping, weight, m))
        if labels_name == "rebuild_state_v2" and key in differences:
            sensitivity.append({"kind": kind, "grouping": grouping, "weight": weight, "min_class": m,
                                "weighted": float(weighted_concordance(np.sign(differences[key]), public_difference))})
    pl.DataFrame(sensitivity).write_csv(settings.output_dir / "sensitivity.csv")

    best_single = next(label for label in ranking["variant"] if " + " not in label)
    report_groups = pl.read_csv(settings.folds_dir / "all_5fold_scanner_v1/studies.csv").select(STUDY, "report_group")
    adversarial = adversarial_scores(settings, frames, targets, groups, variants, report_groups, list(dict.fromkeys(REFERENCES + [best_single])))
    adversarial.write_csv(settings.output_dir / "adversarial.csv")
    (settings.output_dir / "summary.json").write_text(json.dumps({
        "variants": len(differences), "pairs": len(pairs), "best_single": best_single,
        "scanner_group_soft_over_pooled_soft": float(observed), "scanner_group_soft_over_pooled_soft_p": pairwise_p,
    }, indent=2))
    with pl.Config(tbl_rows=-1, tbl_width_chars=250, fmt_str_lengths=110, float_precision=3):
        print(ranking.head(10))
        print(families_table)
        print(pl.DataFrame(cohort_rows))
        print(adversarial.filter(pl.col("source") == "x4"))
    print(f"scanner-group soft over pooled soft {observed:+.3f}, p {pairwise_p:.4f}")


def adversarial_scores(settings, frames, targets, groups, variants, report_groups, metrics) -> pl.DataFrame:
    wanted = [v for v in variants if variant_key(v) in metrics]
    ledger = {}
    for entry in json.loads(settings.ledger.read_text()):
        row = submitted_checkpoint(entry)
        if row is not None:
            ledger[str(Path(row["run"]).resolve())] = (entry["sequence"], row["public"])
    generator = np.random.default_rng(settings.seed)
    control_predictions, rows = {}, []
    for name, path in ADVERSARIAL.items():
        run = Path(path)
        selected = json.loads((run / "fold_3/selected_checkpoint.json").read_text())
        epoch = int(re.search(r"epoch=(\d+)", Path(selected["path"]).name)[1])
        sequence, public = ledger.get(str(run.resolve()), (None, None))
        for source, file in prediction_sources(run, 3, epoch, 1, True, selected.get("prediction_image_multiplier")).items():
            predictions = pl.read_csv(file).select(STUDY, *targets).sort(STUDY)
            point = metric_values(predictions, frames, targets, groups, wanted)
            if name == "half-stack-control":
                control_predictions[source] = (predictions, point)
                rows.extend({"run": name, "source": source, "metric": m, "value": point[m], "delta": 0.0, "low80": None, "high80": None,
                             "ledger_sequence": sequence, "public": public} for m in metrics)
                continue
            control, control_point = control_predictions[source]
            codes = predictions.join(groups.select(STUDY), on=STUDY).join(report_groups, on=STUDY).sort(STUDY)["report_group"].to_numpy()
            unique_groups = np.unique(codes)
            members = {g: np.flatnonzero(codes == g) for g in unique_groups}
            draws = []
            for _ in range(settings.bootstrap):
                index = np.concatenate([members[g] for g in generator.choice(unique_groups, len(unique_groups))])
                a = metric_values(predictions, frames, targets, groups, wanted, index)
                b = metric_values(control, frames, targets, groups, wanted, index)
                draws.append({k: a[k] - b[k] for k in a})
            for metric in metrics:
                spread = np.array([d[metric] for d in draws])
                rows.append({"run": name, "source": source, "metric": metric, "value": point[metric], "delta": point[metric] - control_point[metric],
                             "low80": float(np.nanquantile(spread, 0.1)), "high80": float(np.nanquantile(spread, 0.9)),
                             "ledger_sequence": sequence, "public": public})
    return pl.DataFrame(rows)


if __name__ == "__main__":
    run(GroupedAUCCLI())
