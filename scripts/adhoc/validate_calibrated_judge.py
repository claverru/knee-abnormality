import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from pydantic_settings import SettingsError
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.labels import STUDY, TARGET_THRESHOLD
from knee_abnormality.provenance import copy_sources, sha256_file
from scripts.tools.prepare_combined_labels import consensus_sources

PUBLIC = Path("data/labels/public")
STEVEN_V4 = PUBLIC / "stevenleehans/version-6/llm_labels_v4_blend.csv"
COMPARISON_MANIFEST = Path("configs/labels/public-label-comparison-sources.json")
SUBMISSIONS = Path("scripts/adhoc/submission_public_scores.json")
JUDGE_FILES = {
    "hosseini_jev_v2": PUBLIC
    / "hosseini7798/rsna-knee-label-extraction-jev/downloaded-20260922/Label_Extraction_Jev_v2.csv",
    "pearse_weak_v2": PUBLIC
    / "pearsejim01/rsna-2026-eda/downloaded-20260922/weak_labels_v2.parquet",
    "lvt_teacher": PUBLIC
    / "lvthanhxun/rsna-knee-v9-teacher/downloaded-20260922/official_teacher_fused_v9.csv",
    "nekkon_nli": PUBLIC
    / "nekkon/rsna-knee-nli-entailment-scores/downloaded-20260922/nli_entailment_scores.csv",
    "ramit_luna": PUBLIC
    / "ramitnandi/rsna-knee-labels-gpt-luna/downloaded-20260921/report_labels.csv",
    "pilkwang_v2": PUBLIC / "pilkwang/version-1/report_labels_v2.csv",
}
CONSTRUCTION_ONLY = ("ramit_luna", "pilkwang_v2")
MIN_GOLD_AUC = 0.80
MAX_CONSTRUCTION_KAPPA = 0.90
MIN_STATE_SHARE = 0.02
MAX_STATE_SHARE = 0.80
PAIR_GAP = 0.002
COHORTS = {
    "scanner_combined_x1": [
        ("k=1", "outputs/training/fold0-combined-scanner", 0.940),
        ("k=0.75", "outputs/training/k_075", 0.942),
        ("k=0.5", "outputs/training/k_05", 0.945),
        ("k=0.25", "outputs/training/k_025", 0.944),
        ("k=0", "outputs/training/k_00", 0.943),
        ("oblique", "outputs/training/oblique", 0.944),
        (
            "stratified jitter k=0.5",
            "outputs/autoresearch/sampling/experiments/run-level-stratified-jitter-20260921T224831Z-d1fcb8e3/training",
            0.944,
        ),
    ],
    "v3_steven_x1": [
        (
            "rexnet ordinary BCE",
            "outputs/training/overnight-20260918T232636Z-mt7ouybk/runs/rexnet-fold0",
            0.936,
        ),
        (
            "b0 ordinary BCE",
            "outputs/training/overnight-20260918T232636Z-mt7ouybk/runs/b0-fold0",
            0.934,
        ),
        (
            "control B balanced BCE",
            "outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/00-control",
            0.933,
        ),
        (
            "stratified sampler balanced BCE",
            "outputs/autoresearch/triplet-sampling/experiments/run-level-stratified-jitter-20260919T183826Z-f211c7ce/training",
            0.933,
        ),
        ("robust centre", "outputs/training/robust-center-20-topup-26", 0.938),
    ],
}
CROSS_SPLIT = [
    "outputs/training/fold0-combined-scanner",
    "outputs/training/k_05",
    "outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/00-control",
    "outputs/autoresearch/triplet-sampling/experiments/run-level-stratified-jitter-20260919T183826Z-f211c7ce/training",
    "outputs/training/overnight-20260918T232636Z-mt7ouybk/runs/rexnet-fold0",
]
CROSS_SPLIT_PUBLIC = [0.940, 0.945, 0.933, 0.933, 0.936]


class JudgeCLI(CLISettings):
    output_dir: Path
    reference: Path | None = None
    candidate: Path | None = None
    bootstrap_draws: int = 2000
    seed: int = 2263738815


def long_frame(frame: pl.DataFrame, targets: list[str], column, name: str):
    return pl.concat(
        frame.select(
            pl.col(STUDY).cast(pl.Utf8),
            pl.lit(target).alias("target"),
            column(target).alias(name),
        )
        for target in targets
    )


def judge_states(targets: list[str]) -> dict[str, pl.DataFrame]:
    hosseini = pl.read_csv(JUDGE_FILES["hosseini_jev_v2"], infer_schema_length=0)
    pearse = pl.read_parquet(JUDGE_FILES["pearse_weak_v2"])
    teacher = pl.read_csv(JUDGE_FILES["lvt_teacher"], infer_schema_length=0)
    nli = pl.read_csv(JUDGE_FILES["nekkon_nli"], infer_schema_length=0)
    ramit = pl.read_csv(JUDGE_FILES["ramit_luna"], infer_schema_length=0)
    pilkwang = pl.read_csv(JUDGE_FILES["pilkwang_v2"], infer_schema_length=0)

    def argmax(target: str) -> pl.Expr:
        positive, negative, unknown = (
            pl.col(f"{target}_{k}").cast(pl.Float64) for k in (1, 0, 9)
        )
        return (
            pl.when((positive >= negative) & (positive >= unknown))
            .then(pl.lit("positive"))
            .when(negative >= unknown)
            .then(pl.lit("negative"))
            .otherwise(pl.lit("unknown"))
        )

    def mentioned(target: str) -> pl.Expr:
        return (
            pl.when(~pl.col(f"{target}__mentioned").cast(pl.Boolean))
            .then(pl.lit("unmentioned"))
            .when(pl.col(target).cast(pl.Float64) >= TARGET_THRESHOLD)
            .then(pl.lit("positive"))
            .otherwise(pl.lit("negative"))
        )

    def nli_bin(target: str) -> pl.Expr:
        return (
            (pl.col(target).cast(pl.Float64) * 5)
            .floor()
            .clip(0, 4)
            .cast(pl.Int64)
            .cast(pl.Utf8)
        )

    ramit_long = ramit.select(
        pl.col(STUDY),
        pl.col("CLASS")
        .str.replace_all("_", " ")
        .str.replace("Bakers", "Baker's")
        .alias("target"),
        pl.when(pl.col("classification") == "POSITIVE")
        .then(pl.col("severity"))
        .when(pl.col("classification") == "NA")
        .then(pl.lit("UNADDRESSED"))
        .otherwise(pl.col("classification"))
        .alias("state"),
    )
    return {
        "hosseini_jev_v2": long_frame(hosseini, targets, argmax, "state"),
        "pearse_weak_v2": long_frame(pearse, targets, mentioned, "state"),
        "lvt_teacher_llm": long_frame(
            teacher, targets, lambda t: pl.col(f"{t}__llm_state"), "state"
        ),
        "lvt_teacher_rule": long_frame(
            teacher, targets, lambda t: pl.col(f"{t}__rule_state"), "state"
        ),
        "nekkon_nli": long_frame(nli, targets, nli_bin, "state"),
        "ramit_luna": ramit_long,
        "pilkwang_v2": long_frame(
            pilkwang, targets, lambda t: pl.col(f"{t}__verdict"), "state"
        ),
    }


def calibrate(states: pl.DataFrame, gold: pl.DataFrame) -> pl.DataFrame:
    return (
        states.join(gold, on=[STUDY, "target"])
        .group_by("state")
        .agg(pl.len().alias("gold_cells"), pl.col("gold").mean().alias("p"))
        .sort("state")
    )


def eligibility(
    states: dict[str, pl.DataFrame],
    gold: pl.DataFrame,
    construction: dict[str, pl.DataFrame],
    targets: list[str],
) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    gold_ids = gold[STUDY].unique()
    rows, tables = [], {}
    for name, frame in states.items():
        table = calibrate(frame, gold)
        tables[name] = table
        scored = frame.join(table, on="state").join(gold, on=[STUDY, "target"])
        silver = frame.filter(~pl.col(STUDY).is_in(gold_ids.to_list()))
        shares = silver["state"].value_counts(normalize=True)["proportion"]
        positive = frame.join(table, on="state").with_columns(
            (pl.col("p") >= 0.5).alias("called")
        )
        kappas = {}
        for source, labels in construction.items():
            joined = positive.filter(~pl.col(STUDY).is_in(gold_ids.to_list())).join(
                labels.unpivot(
                    index=STUDY, on=targets, variable_name="target", value_name="v"
                ).drop_nulls("v"),
                on=[STUDY, "target"],
            )
            kappas[source] = float(
                cohen_kappa_score(
                    joined["called"].to_numpy(),
                    joined["v"].to_numpy() >= TARGET_THRESHOLD,
                )
            )
        closest = max(kappas, key=kappas.get)
        gold_auc = float(roc_auc_score(scored["gold"] >= 0.5, scored["p"]))
        row = {
            "source": name,
            "gold_studies": scored[STUDY].n_unique(),
            "states": table.height,
            "gold_auc": gold_auc,
            "largest_state_share": float(shares.max()),
            "smallest_state_share": float(shares.min()),
            "closest_construction_source": closest,
            "closest_kappa": kappas[closest],
            "construction_source": name in CONSTRUCTION_ONLY,
        }
        row["eligible"] = (
            not row["construction_source"]
            and row["gold_studies"] == gold[STUDY].n_unique()
            and gold_auc >= MIN_GOLD_AUC
            and row["closest_kappa"] < MAX_CONSTRUCTION_KAPPA
            and row["largest_state_share"] <= MAX_STATE_SHARE
            and row["smallest_state_share"] >= MIN_STATE_SHARE
        )
        rows.append(row)
    return pl.DataFrame(rows), tables


def expected_truth(
    states: dict[str, pl.DataFrame],
    tables: dict[str, pl.DataFrame],
    sources: list[str],
    gold: pl.DataFrame,
) -> pl.DataFrame:
    calibrated = pl.concat(
        states[name]
        .join(tables[name].select("state", "p"), on="state")
        .select(STUDY, "target", "p")
        for name in sources
    )
    return (
        calibrated.group_by(STUDY, "target")
        .agg(pl.col("p").mean())
        .join(gold, on=[STUDY, "target"], how="full", coalesce=True)
        .select(STUDY, "target", pl.coalesce("gold", "p").alias("truth"))
    )


def soft_auc(scores: np.ndarray, truth: np.ndarray, weight: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    s, q, w = scores[order], truth[order], weight[order]
    negative = w * (1 - q)
    positive = w * q
    cumulative = np.concatenate([[0.0], np.cumsum(negative)])
    start = np.searchsorted(s, s, side="left")
    end = np.searchsorted(s, s, side="right")
    below = cumulative[start]
    tied = cumulative[end] - cumulative[start] - negative
    numerator = (positive * (below + 0.5 * tied)).sum()
    denominator = (positive * (negative.sum() - negative)).sum()
    return float(numerator / denominator)


def predictions(run: Path) -> pl.DataFrame:
    fold = run / "fold_0"
    for name in ("validation_predictions.csv.x1", "validation_predictions.csv"):
        if (fold / name).exists():
            frame = pl.read_csv(fold / name)
            return (
                frame.filter(pl.col("fold") == 0) if "fold" in frame.columns else frame
            )
    raise FileNotFoundError(f"No validation predictions under {fold}")


def split_of(run: Path) -> Path:
    config = json.loads((run / "effective_config.json").read_text())
    return Path("data/folds") / config["validation"]["split"]


def scored_frame(
    run: Path, targets: list[str], truth: pl.DataFrame, studies=None
) -> pl.DataFrame:
    frame = predictions(run)
    if studies is not None:
        frame = frame.filter(pl.col(STUDY).is_in(studies))
    frozen = pl.read_csv(split_of(run) / "targets.csv").unpivot(
        index=STUDY, on=targets, variable_name="target", value_name="frozen"
    )
    return (
        frame.select(STUDY, *targets)
        .unpivot(index=STUDY, on=targets, variable_name="target", value_name="score")
        .join(frozen, on=[STUDY, "target"])
        .join(truth, on=[STUDY, "target"], how="left")
    )


def macro_scores(frame: pl.DataFrame, targets: list[str]) -> dict[str, float]:
    fold, judge = [], []
    for target in targets:
        part = frame.filter(pl.col("target") == target)
        fold.append(
            roc_auc_score(part["frozen"] >= TARGET_THRESHOLD, part["score"].to_numpy())
        )
        known = part.filter(pl.col("truth").is_not_null())
        judge.append(
            soft_auc(
                known["score"].to_numpy(),
                known["truth"].to_numpy(),
                np.ones(known.height),
            )
        )
    return {"fold0_macro_auc": float(np.mean(fold)), "judge": float(np.mean(judge))}


def agreement(names, public, values) -> dict:
    public, values = np.asarray(public), np.asarray(values)
    pairs = [
        (i, j)
        for i, j in combinations(range(len(public)), 2)
        if abs(public[i] - public[j]) >= PAIR_GAP - 1e-9
    ]
    concordant = sum(
        np.sign(values[i] - values[j]) == np.sign(public[i] - public[j])
        for i, j in pairs
    )
    return {
        "runs": len(names),
        "kendall_tau_b": float(kendalltau(values, public).statistic),
        "between_run_sd": float(np.std(values, ddof=1)),
        "pairs_at_least_0.002_apart": len(pairs),
        "concordant_pairs": int(concordant),
    }


def cohort_runs() -> dict[str, list[tuple[str, Path, float]]]:
    manifest = json.loads(SUBMISSIONS.read_text())
    plateau = manifest["plateau_from"]
    early = [
        (s["name"], Path(s["run"]), s["public"], s["submitted"])
        for s in manifest["submissions"]
        if s["excluded"] is None or s["name"] == "tonylin-soft-v3"
    ]
    cohorts = {
        name: [(n, Path(r), p) for n, r, p in runs] for name, runs in COHORTS.items()
    }
    cohorts["v3_steven_x2"] = [(n, r, p) for n, r, p, _ in early]
    cohorts["v3_steven_x2_plateau"] = [
        (n, r, p) for n, r, p, d in early if d >= plateau
    ]
    return cohorts


def replicate_noise(targets, judges: dict[str, pl.DataFrame]) -> pl.DataFrame:
    manifest = json.loads(SUBMISSIONS.read_text())
    rows = []
    for judge, truth in judges.items():
        deviations = {"fold0_macro_auc": [], "judge": []}
        for group in manifest["replicate_groups"]:
            values = [
                macro_scores(scored_frame(Path(run), targets, truth), targets)
                for run in group["runs"]
            ]
            for key, collected in deviations.items():
                series = np.array([v[key] for v in values])
                collected.extend(series - series.mean())
        freedom = sum(len(g["runs"]) - 1 for g in manifest["replicate_groups"])
        rows.append(
            {
                "judge": judge,
                "degrees_of_freedom": freedom,
                "fold0_macro_auc_sd": float(
                    np.sqrt(np.sum(np.square(deviations["fold0_macro_auc"])) / freedom)
                ),
                "judge_sd": float(
                    np.sqrt(np.sum(np.square(deviations["judge"])) / freedom)
                ),
            }
        )
    return pl.DataFrame(rows)


def compare(settings: JudgeCLI, targets, truth: pl.DataFrame) -> dict:
    reference = scored_frame(settings.reference, targets, truth)
    candidate = scored_frame(settings.candidate, targets, truth)
    groups = pl.read_csv(split_of(settings.reference) / "studies.csv").select(
        pl.col(STUDY), "report_group"
    )
    studies = reference[STUDY].unique().sort()
    if set(candidate[STUDY].unique().to_list()) != set(studies.to_list()):
        raise ValueError("Reference and candidate predict different studies")
    group_index = (
        pl.DataFrame({STUDY: studies})
        .join(groups, on=STUDY, how="left")
        .with_columns(pl.col("report_group").rank("dense").cast(pl.Int64) - 1)
    )
    study_group = dict(zip(group_index[STUDY], group_index["report_group"]))
    group_count = int(group_index["report_group"].max()) + 1
    parts = []
    for target in targets:
        pair = []
        for frame in (reference, candidate):
            known = frame.filter(
                (pl.col("target") == target) & pl.col("truth").is_not_null()
            ).sort(STUDY)
            pair.append(known)
        index = np.array([study_group[s] for s in pair[0][STUDY]])
        parts.append((pair, index))

    def delta(group_weight: np.ndarray) -> float:
        values = []
        for pair, index in parts:
            weight = group_weight[index]
            values.append(
                soft_auc(
                    pair[1]["score"].to_numpy(), pair[1]["truth"].to_numpy(), weight
                )
                - soft_auc(
                    pair[0]["score"].to_numpy(), pair[0]["truth"].to_numpy(), weight
                )
            )
        return float(np.mean(values))

    rng = np.random.default_rng(settings.seed)
    observed = delta(np.ones(group_count))
    draws = np.array(
        [
            delta(
                np.bincount(
                    rng.integers(0, group_count, group_count), minlength=group_count
                ).astype(float)
            )
            for _ in range(settings.bootstrap_draws)
        ]
    )
    return {
        "reference": str(settings.reference),
        "candidate": str(settings.candidate),
        "judge_reference": macro_scores(reference, targets)["judge"],
        "judge_candidate": macro_scores(candidate, targets)["judge"],
        "delta": observed,
        "interval_80": [float(np.quantile(draws, 0.1)), float(np.quantile(draws, 0.9))],
        "interval_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "draws": settings.bootstrap_draws,
    }


def validate(settings: JudgeCLI) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    targets = json.loads(
        Path("data/folds/all_5fold_scanner_v1/protocol.json").read_text()
    )["targets"]
    train = pl.read_csv("data/raw/train.csv", infer_schema_length=None)
    gold = (
        train.filter(pl.col(targets[0]).is_not_null())
        .select(STUDY, *targets)
        .unpivot(index=STUDY, on=targets, variable_name="target", value_name="gold")
        .with_columns(pl.col("gold").cast(pl.Float64))
    )
    _, construction = consensus_sources(
        json.loads(COMPARISON_MANIFEST.read_text()), targets
    )
    construction["steven_v4"] = pl.read_csv(STEVEN_V4, infer_schema_length=None).select(
        pl.col(STUDY).cast(pl.Utf8), *[pl.col(t).cast(pl.Float64) for t in targets]
    )
    states = judge_states(targets)
    eligible_table, tables = eligibility(states, gold, construction, targets)
    eligible = eligible_table.filter("eligible")["source"].to_list()
    if not eligible:
        raise ValueError("No judge source passed eligibility")
    judges = {"frozen": expected_truth(states, tables, eligible, gold)}
    for name in states:
        judges[name] = expected_truth(states, tables, [name], gold)
    history_rows, agreement_rows = [], []
    for cohort, runs in cohort_runs().items():
        scored = {
            name: {
                label: macro_scores(scored_frame(run, targets, truth), targets)
                for label, run, _ in runs
            }
            for name, truth in judges.items()
        }
        for label, run, public in runs:
            history_rows.append(
                {
                    "cohort": cohort,
                    "run": label,
                    "path": str(run),
                    "public": public,
                    "fold0_macro_auc": scored["frozen"][label]["fold0_macro_auc"],
                    **{
                        f"judge_{name}": scored[name][label]["judge"] for name in judges
                    },
                }
            )
        labels = [label for label, _, _ in runs]
        public = [p for _, _, p in runs]
        agreement_rows.append(
            {
                "cohort": cohort,
                "metric": "fold0_macro_auc",
                **agreement(
                    labels,
                    public,
                    [scored["frozen"][l]["fold0_macro_auc"] for l in labels],
                ),
            }
        )
        for name in judges:
            agreement_rows.append(
                {
                    "cohort": cohort,
                    "metric": f"judge_{name}",
                    **agreement(
                        labels, public, [scored[name][l]["judge"] for l in labels]
                    ),
                }
            )
    shared = None
    for run in CROSS_SPLIT:
        ids = set(predictions(Path(run))[STUDY].to_list())
        shared = ids if shared is None else shared & ids
    shared = sorted(shared)
    cross_rows = []
    for run, public in zip(CROSS_SPLIT, CROSS_SPLIT_PUBLIC):
        row = {"path": run, "public": public, "shared_studies": len(shared)}
        for name, truth in judges.items():
            row[f"judge_{name}"] = macro_scores(
                scored_frame(Path(run), targets, truth, shared), targets
            )["judge"]
        cross_rows.append(row)
    outputs = {
        "eligibility": eligible_table,
        "calibration": pl.concat(
            table.with_columns(pl.lit(name).alias("source"))
            for name, table in tables.items()
        ),
        "history": pl.DataFrame(history_rows),
        "agreement": pl.DataFrame(agreement_rows),
        "cross_split_shared_studies": pl.DataFrame(cross_rows),
        "replicate_noise": replicate_noise(targets, {"frozen": judges["frozen"]}),
    }
    for name, table in outputs.items():
        table.write_csv(settings.output_dir / f"{name}.csv")
    judges["frozen"].write_csv(settings.output_dir / "frozen_expected_truth.csv")
    summary = {
        "judge": "fold-0 macro soft AUC against expected truth: gold labels on gold studies, "
        "otherwise the mean over eligible sources of P(gold positive | source state), "
        "pooled across labels on the 58 gold studies",
        "eligible_sources": eligible,
        "rules": {
            "min_gold_auc": MIN_GOLD_AUC,
            "max_construction_kappa": MAX_CONSTRUCTION_KAPPA,
            "state_share_range": [MIN_STATE_SHARE, MAX_STATE_SHARE],
            "construction_only": list(CONSTRUCTION_ONLY),
            "pair_gap": PAIR_GAP,
        },
    }
    if settings.reference is not None and settings.candidate is not None:
        summary["comparison"] = compare(settings, targets, judges["frozen"])
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    files = [*JUDGE_FILES.values(), STEVEN_V4, COMPARISON_MANIFEST, SUBMISSIONS]
    manifest = {
        "inputs": {str(path): sha256_file(path) for path in files},
        "sources": copy_sources((Path(__file__),), settings.output_dir / "source"),
    }
    (settings.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with pl.Config(tbl_rows=80, tbl_cols=20, tbl_width_chars=260, float_precision=4):
        print(json.dumps(summary, indent=2))
        for name in (
            "eligibility",
            "agreement",
            "cross_split_shared_studies",
            "replicate_noise",
        ):
            print(name)
            print(outputs[name])


def main() -> None:
    try:
        settings = JudgeCLI()
        if (settings.reference is None) != (settings.candidate is None):
            raise ValueError("Pass both --reference and --candidate, or neither")
        validate(settings)
    except (KeyError, OSError, TypeError, ValueError, SettingsError) as error:
        print(f"Calibrated judge error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
