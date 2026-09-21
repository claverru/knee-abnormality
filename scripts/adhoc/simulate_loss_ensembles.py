import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field
from pydantic_settings import SettingsError
from scipy.stats import rankdata, spearmanr

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.labels import STUDY, TARGET_THRESHOLD
from knee_abnormality.evaluation.metrics import aucs, bootstrap_auc
from knee_abnormality.provenance import copy_sources, sha256_file

CONTROLS = (
    Path("outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat"),
    Path("outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/00-control"),
    Path("outputs/training/balanced-bce"),
)
ALLOWED_CONFIG_PREFIXES = ("name", "seed", "paths.output_root", "trainer.loss")
TOP_SUBSETS = 25

state: dict = {}


class EnsembleCLI(CLISettings):
    campaign: Path = Path("outputs/autoresearch/losses/experiments")
    controls: list[Path] = Field(default_factory=lambda: list(CONTROLS))
    split_dir: Path = Path("data/folds/all_5fold_v3")
    output_dir: Path
    fold: int = Field(default=0, ge=0)
    batch: int = Field(default=512, ge=1)
    splits: int = Field(default=200, ge=10)
    selection_max_size: int = Field(default=5, ge=2)
    workers: int = Field(default=16, ge=1)
    bootstrap_draws: int = Field(default=2000, ge=100)
    seed: int = 20260921


def flatten(value: object, prefix: str = "") -> dict:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: json.dumps(value, sort_keys=True)}


def differences(reference: dict, candidate: dict) -> list[str]:
    return sorted(
        key
        for key in set(reference) | set(candidate)
        if reference.get(key) != candidate.get(key)
    )


def load_members(settings: EnsembleCLI, frame: pl.DataFrame, targets: list[str]):
    roots = [(root.name, "control", root) for root in settings.controls] + [
        (root.parent.name.rsplit("-", 2)[0], "loss", root)
        for root in sorted(settings.campaign.glob("*/training"))
        if (root / f"fold_{settings.fold}" / "validation_predictions.csv").exists()
    ]
    anchor = settings.controls[0]
    reference_config = flatten(
        json.loads((anchor / "effective_config.json").read_text())
    )
    reference_identity = flatten(
        json.loads(
            (anchor / f"fold_{settings.fold}" / "input_identity.json").read_text()
        )
    )
    members, predictions = [], []
    for name, kind, root in roots:
        fold_dir = root / f"fold_{settings.fold}"
        config = json.loads((root / "effective_config.json").read_text())
        config_diff = differences(reference_config, flatten(config))
        if rejected := [
            key for key in config_diff if not key.startswith(ALLOWED_CONFIG_PREFIXES)
        ]:
            raise ValueError(f"{name}: unexpected configuration differences {rejected}")
        identity_diff = differences(
            reference_identity,
            flatten(json.loads((fold_dir / "input_identity.json").read_text())),
        )
        evaluation = root.parent / "evaluation.json"
        prediction_path = fold_dir / "validation_predictions.csv"
        prediction = pl.read_csv(prediction_path).sort(STUDY)
        if prediction[STUDY].to_list() != frame[STUDY].to_list():
            raise ValueError(f"{name}: prediction membership differs from the fold")
        predictions.append(prediction.select(targets).to_numpy().astype(np.float64))
        members.append(
            {
                "name": name,
                "kind": kind,
                "decision": json.loads(evaluation.read_text())["eligible_decision"]
                if kind == "loss"
                else "control",
                "seed": config["seed"],
                "root": str(root),
                "predictions_sha256": sha256_file(prediction_path),
                "config_differences": [
                    key for key in config_diff if not key.startswith(("name", "paths"))
                ],
                "input_identity_differences": identity_diff,
            }
        )
    return members, np.stack(predictions)


def per_target_auc(scores: np.ndarray, hard: np.ndarray) -> np.ndarray:
    positives = hard.sum(axis=0)
    negatives = len(hard) - positives
    ranks = rankdata(scores, axis=1)
    rank_sum = np.einsum("snt,nt->st", ranks, hard)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def masks_to_weights(masks: np.ndarray, count: int) -> np.ndarray:
    bits = ((masks[:, None] >> np.arange(count)) & 1).astype(np.float64)
    return bits / bits.sum(axis=1, keepdims=True)


def evaluate(masks: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    predictions, hard, batch = state["predictions"], state["hard"], state["batch"]
    if rows is not None:
        predictions, hard = predictions[:, rows], hard[rows]
    result = []
    for start in range(0, len(masks), batch):
        weights = masks_to_weights(masks[start : start + batch], len(predictions))
        scores = np.einsum("sm,mnt->snt", weights, predictions)
        result.append(per_target_auc(scores, hard))
    return np.concatenate(result)


def initialize(predictions: np.ndarray, hard: np.ndarray, batch: int) -> None:
    state.update(predictions=predictions, hard=hard, batch=batch)


def evaluate_chunk(masks: np.ndarray) -> np.ndarray:
    return evaluate(masks)


def greedy(rows: np.ndarray | None, count: int) -> list[tuple[int, float]]:
    singles = evaluate(1 << np.arange(count), rows).mean(axis=1)
    mask, best = 1 << int(singles.argmax()), float(singles.max())
    path = [(mask, best)]
    while True:
        candidates = np.array(
            [mask | (1 << index) for index in range(count) if not mask >> index & 1]
        )
        if not len(candidates):
            return path
        scores = evaluate(candidates, rows).mean(axis=1)
        if scores.max() <= best:
            return path
        mask, best = int(candidates[scores.argmax()]), float(scores.max())
        path.append((mask, best))


def selection_split(arguments: tuple) -> dict:
    split, rows_a, rows_b, fixed, small, count = arguments
    small_a = evaluate(small, rows_a).mean(axis=1)
    singles = small[np.bitwise_count(small) == 1]
    single_a = small_a[np.bitwise_count(small) == 1]
    order = singles[np.argsort(-single_a)]
    chosen = {
        **fixed,
        "best_single": int(order[0]),
        "top3": int(np.bitwise_or.reduce(order[:3])),
        "top5": int(np.bitwise_or.reduce(order[:5])),
        "exhaustive": int(small[small_a.argmax()]),
        "greedy": greedy(rows_a, count)[-1][0],
    }
    masks = np.array(list(chosen.values()))
    score_a = evaluate(masks, rows_a).mean(axis=1)
    score_b = evaluate(masks, rows_b).mean(axis=1)
    return {
        "split": split,
        **{
            f"{strategy}_{field}": value
            for strategy, mask, a, b in zip(
                chosen, masks, score_a, score_b, strict=True
            )
            for field, value in (("mask", int(mask)), ("a", float(a)), ("b", float(b)))
        },
    }


def label(mask: int, names: list[str]) -> str:
    return " + ".join(name for index, name in enumerate(names) if mask >> index & 1)


def interval(delta: np.ndarray) -> dict:
    return {
        "delta_mean": float(delta.mean()),
        "central_80": np.percentile(delta, [10, 90]).tolist(),
        "central_95": np.percentile(delta, [2.5, 97.5]).tolist(),
        "positive_fraction": float((delta > 0).mean()),
    }


def simulate(settings: EnsembleCLI) -> None:
    if settings.output_dir.exists():
        raise ValueError(f"Output directory exists: {settings.output_dir}")
    labels_path = settings.split_dir / "targets.csv"
    studies_path = settings.split_dir / "studies.csv"
    labels = pl.read_csv(labels_path)
    targets = labels.columns[1:]
    frame = (
        pl.read_csv(studies_path)
        .filter(pl.col("fold") == settings.fold)
        .join(labels, on=STUDY, validate="1:1")
        .sort(STUDY)
    )
    hard = (frame.select(targets).to_numpy() >= TARGET_THRESHOLD).astype(np.float64)
    members, predictions = load_members(settings, frame, targets)
    names = [member["name"] for member in members]
    count = len(members)
    initialize(predictions, hard, settings.batch)
    for member, values in zip(members, predictions, strict=True):
        exact = aucs(hard.astype(np.int8), values, targets)
        member["macro_auc"] = exact["macro_auc"]
        member.update(exact["per_target_auc"])
    fast = evaluate(1 << np.arange(count)).mean(axis=1)
    exact = np.array([member["macro_auc"] for member in members])
    if np.abs(fast - exact).max() > 1e-9:
        raise ValueError("Rank AUC disagrees with the metric implementation")

    masks = np.arange(1, 1 << count, dtype=np.int64)
    chunks = np.array_split(masks, settings.workers * 8)
    with ProcessPoolExecutor(
        settings.workers,
        initializer=initialize,
        initargs=(predictions, hard, settings.batch),
    ) as pool:
        table = np.concatenate(list(pool.map(evaluate_chunk, chunks)))
    macro = table.mean(axis=1)
    sizes = np.bitwise_count(masks)
    single = np.zeros(1 << count)
    single[masks] = macro
    bits = (masks[:, None] >> np.arange(count)) & 1
    member_auc = exact
    mean_member = bits @ member_auc / sizes
    best_member = np.where(bits == 1, member_auc, -np.inf).max(axis=1)
    controls = bits[:, [member["kind"] == "control" for member in members]].sum(axis=1)

    settings.output_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                key: value
                for key, value in member.items()
                if not key.endswith("differences")
            }
            for member in members
        ]
    ).write_csv(settings.output_dir / "members.csv")
    subsets = pl.DataFrame(
        {
            "mask": masks,
            "size": sizes,
            "controls": controls,
            "members": [label(int(mask), names) for mask in masks],
            "macro_auc": macro,
            "mean_member_auc": mean_member,
            "gain_vs_mean_member": macro - mean_member,
            "gain_vs_best_member": macro - best_member,
            **{target: table[:, index] for index, target in enumerate(targets)},
        }
    )
    subsets.write_parquet(settings.output_dir / "subsets.parquet")

    kinds = [member["kind"] for member in members]
    pair_rows = []
    for first in range(count):
        for second in range(first + 1, count):
            mask = (1 << first) | (1 << second)
            correlation = np.mean(
                [
                    spearmanr(predictions[first][:, t], predictions[second][:, t])[0]
                    for t in range(len(targets))
                ]
            )
            pair_rows.append(
                {
                    "first": names[first],
                    "second": names[second],
                    "type": "-".join(sorted((kinds[first], kinds[second]))),
                    "macro_auc": single[mask],
                    "gain_vs_mean_member": single[mask]
                    - (member_auc[first] + member_auc[second]) / 2,
                    "gain_vs_best_member": single[mask]
                    - max(member_auc[first], member_auc[second]),
                    "mean_spearman": float(correlation),
                }
            )
    pairs = pl.DataFrame(pair_rows).sort("macro_auc", descending=True)
    pairs.write_csv(settings.output_dir / "pairs.csv")

    marginal_rows = []
    for index, name in enumerate(names):
        bit = 1 << index
        without = masks[(masks & bit) == 0]
        delta = single[without | bit] - single[without]
        size = np.bitwise_count(without)
        marginal_rows.append(
            {
                "member": name,
                "decision": members[index]["decision"],
                "macro_auc": member_auc[index],
                "mean_gain_all": float(delta.mean()),
                **{
                    f"mean_gain_size_{k}": float(delta[size == k].mean())
                    for k in range(1, 5)
                },
                "anchor_pair_gain": float(
                    single[1 | bit] - single[1] if index else 0.0
                ),
            }
        )
    marginal = pl.DataFrame(marginal_rows).sort("mean_gain_size_2", descending=True)
    marginal.write_csv(settings.output_dir / "marginal.csv")

    best_rows, frequency_rows = [], []
    for k in range(1, count + 1):
        order = np.flatnonzero(sizes == k)
        order = order[np.argsort(-macro[order])]
        best = order[0]
        best_rows.append(
            {
                "size": k,
                "subsets": len(order),
                "macro_auc": macro[best],
                "gain_vs_mean_member": macro[best] - mean_member[best],
                "controls": int(controls[best]),
                "members": label(int(masks[best]), names),
            }
        )
        top = bits[order[:TOP_SUBSETS]].mean(axis=0)
        frequency_rows.append(
            {"size": k, **dict(zip(names, top.tolist(), strict=True))}
        )
    best_by_size = pl.DataFrame(best_rows)
    best_by_size.write_csv(settings.output_dir / "best_by_size.csv")
    pl.DataFrame(frequency_rows).write_csv(settings.output_dir / "top_frequency.csv")

    greedy_path = greedy(None, count)
    pl.DataFrame(
        [
            {"step": step + 1, "macro_auc": score, "members": label(mask, names)}
            for step, (mask, score) in enumerate(greedy_path)
        ]
    ).write_csv(settings.output_dir / "greedy.csv")

    control_mask = int(sum(1 << i for i, kind in enumerate(kinds) if kind == "control"))
    promising_mask = int(
        sum(1 << i for i, m in enumerate(members) if m["decision"] != "reject")
    )
    cohorts = {
        "seed_only": (masks & ~control_mask) == 0,
        "non_rejected_losses": (controls == 0) & ((masks & ~promising_mask) == 0),
        "all_losses": controls == 0,
    }
    matched_size = []
    for k in range(2, len(settings.controls) + 1):
        seed_mean = macro[(sizes == k) & cohorts["seed_only"]].mean()
        for cohort, selected in cohorts.items():
            selected = selected & (sizes == k)
            matched_size.append(
                {
                    "size": k,
                    "cohort": cohort,
                    "subsets": int(selected.sum()),
                    "mean_macro_auc": float(macro[selected].mean()),
                    "mean_member_auc": float(mean_member[selected].mean()),
                    "mean_gain_vs_mean_member": float(
                        (macro - mean_member)[selected].mean()
                    ),
                    "beats_seed_only_mean": float((macro[selected] > seed_mean).mean()),
                }
            )
    fixed = {
        "anchor": 1,
        "control_seeds": control_mask,
        "non_rejected": promising_mask,
        "all": (1 << count) - 1,
    }
    small = masks[sizes <= settings.selection_max_size]
    groups, inverse = np.unique(frame["report_group"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(settings.seed)
    jobs = []
    for split in range(settings.splits):
        half = rng.permutation(len(groups))[: len(groups) // 2]
        in_a = np.isin(inverse, half)
        rows_a, rows_b = np.flatnonzero(in_a), np.flatnonzero(~in_a)
        jobs.append((split, rows_a, rows_b, fixed, small, count))
    with ProcessPoolExecutor(
        settings.workers,
        initializer=initialize,
        initargs=(predictions, hard, settings.batch),
    ) as pool:
        selection = pl.DataFrame(list(pool.map(selection_split, jobs)))
    selection.write_csv(settings.output_dir / "selection_splits.csv")
    strategies = [*fixed, "best_single", "top3", "top5", "exhaustive", "greedy"]
    selection_summary = {
        strategy: {
            "held_out_mean": float(selection[f"{strategy}_b"].mean()),
            "selection_half_mean": float(selection[f"{strategy}_a"].mean()),
            "optimism": float(
                (selection[f"{strategy}_a"] - selection[f"{strategy}_b"]).mean()
            ),
            "held_out_vs_control_seeds": float(
                (selection[f"{strategy}_b"] - selection["control_seeds_b"]).mean()
            ),
            "wins_vs_control_seeds": float(
                (selection[f"{strategy}_b"] > selection["control_seeds_b"]).mean()
            ),
            "held_out_vs_anchor": float(
                (selection[f"{strategy}_b"] - selection["anchor_b"]).mean()
            ),
            "mean_size": float(
                np.bitwise_count(selection[f"{strategy}_mask"].to_numpy()).mean()
            ),
        }
        for strategy in strategies
    }

    counts = rng.multinomial(
        len(groups),
        np.full(len(groups), 1 / len(groups)),
        size=settings.bootstrap_draws,
    )[:, inverse]
    positive = counts @ hard
    counts = counts[
        ((positive > 0) & (positive < counts.sum(axis=1, keepdims=True))).all(axis=1)
    ]
    candidates = {
        **fixed,
        "greedy_full": greedy_path[-1][0],
        **{
            f"best_size_{k}": int(
                masks[np.flatnonzero(sizes == k)[macro[sizes == k].argmax()]]
            )
            for k in range(2, 6)
        },
    }

    bootstrap = {
        name: bootstrap_auc(
            hard,
            np.tensordot(masks_to_weights(np.array([mask]), count)[0], predictions, 1),
            counts,
        )
        for name, mask in candidates.items()
    }
    intervals = {
        name: {
            "members": label(mask, names),
            "macro_auc": float(single[mask]),
            "selected_on_this_fold": name.startswith(("best_size", "greedy")),
            "vs_anchor": interval(bootstrap[name] - bootstrap["anchor"]),
            "vs_control_seeds": interval(bootstrap[name] - bootstrap["control_seeds"]),
        }
        for name, mask in candidates.items()
        if name != "anchor"
    }

    sources = copy_sources(
        (Path(__file__), Path("src/knee_abnormality/evaluation/metrics.py")),
        settings.output_dir / "source",
    )
    summary = {
        "settings": json.loads(settings.model_dump_json()),
        "fold": settings.fold,
        "studies": len(frame),
        "targets": targets,
        "ensemble": "unweighted probability mean, as in exported bundles",
        "anchor": names[0],
        "members": [
            {
                key: member[key]
                for key in ("name", "kind", "decision", "seed", "macro_auc")
            }
            | {
                "config_differences": member["config_differences"],
                "input_identity_differences": member["input_identity_differences"],
            }
            for member in members
        ],
        "pair_types": {
            row["type"]: {
                "pairs": row["len"],
                "mean_gain_vs_mean_member": row["gain_vs_mean_member"],
                "mean_spearman": row["mean_spearman"],
            }
            for row in pairs.group_by("type")
            .agg(
                pl.len(),
                pl.col("gain_vs_mean_member").mean(),
                pl.col("mean_spearman").mean(),
            )
            .to_dicts()
        },
        "matched_size": matched_size,
        "best_by_size": best_by_size.to_dicts(),
        "greedy": [
            {"members": label(mask, names), "macro_auc": score}
            for mask, score in greedy_path
        ],
        "selection_check": {
            "splits": settings.splits,
            "exhaustive_max_size": settings.selection_max_size,
            "strategies": selection_summary,
        },
        "paired_report_group_bootstrap": {
            "draws": len(counts),
            "intervals": intervals,
        },
        "inputs": {
            str(path): sha256_file(path) for path in (labels_path, studies_path)
        },
        "sources": sources,
    }
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with pl.Config(tbl_rows=40, tbl_cols=12, fmt_str_lengths=160, tbl_width_chars=220):
        print(pl.DataFrame(summary["members"]).select("name", "decision", "macro_auc"))
        print(marginal)
        print(best_by_size)
        print(pairs.head(15))
    print(json.dumps(summary["pair_types"], indent=2))
    print(json.dumps(matched_size, indent=2))
    print(json.dumps(summary["greedy"], indent=2))
    print(json.dumps(summary["selection_check"], indent=2))
    print(json.dumps(intervals, indent=2))


def main() -> None:
    try:
        simulate(EnsembleCLI())
    except (KeyError, OSError, TypeError, ValueError, SettingsError) as error:
        print(f"Ensemble simulation error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
