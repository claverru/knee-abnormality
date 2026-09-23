import json
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import polars as pl
import torch
from pydantic import Field
from pydantic_settings import SettingsError
from scipy.optimize import linear_sum_assignment

from knee_abnormality.cli import CLISettings
from knee_abnormality.config import INFERENCE_IO_THREADS, load_config
from knee_abnormality.data import dataset as dataset_module
from knee_abnormality.data.dataset import KneeStudyDataset, build_data_loader
from knee_abnormality.data.labels import STUDY, TARGET_THRESHOLD, load_fold_labels
from knee_abnormality.data.preprocessing.sampling import (
    CandidatePool,
    Selection,
    geometry_groups,
    limiting_reach,
    operation_seed,
    source_ranking,
    target_depths,
    union_support,
)
from knee_abnormality.data.sources import build_series_index
from knee_abnormality.evaluation.metrics import aucs, bootstrap_auc
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import move_batch
from knee_abnormality.training.cuda import configure_cuda_allocator

TILTED = {
    "standard": ("standard", None),
    "tilted": ("tilted", None),
    "additive": ("additive", None),
    "additive_acl": ("additive", "acl"),
    "intact_acl": ("intact", "acl"),
    "intact_hidden": ("intact", "hide"),
}
TILTED_POST = ("mean_standard_tilted", "route_acl")
GENERAL = (
    "passes",
    "best_only",
    "additive_all",
    "intact_double",
    "coverage_weighted",
)
GATE_LABELS = {"acl": ("ACL",), "hide": ()}


class PolicyCLI(CLISettings):
    run_dirs: list[Path] = Field(
        default_factory=lambda: [
            Path("outputs/training/k_05"),
            Path("outputs/training/k_025"),
            Path("outputs/training/k_075"),
            Path("outputs/training/k_00"),
            Path("outputs/training/fold0-combined-scanner"),
        ]
    )
    census: Path = Path(
        "outputs/analysis/series-filtering-20260917/series_flags_with_triplets.parquet"
    )
    output_dir: Path
    thresholds: list[float] = Field(default_factory=lambda: [30.0, 20.0])
    settings: list[str] = Field(default_factory=lambda: ["x4-tau0.75"])
    variants: list[str] = Field(
        default_factory=lambda: [*TILTED, *TILTED_POST, *GENERAL]
    )
    fold: int = Field(default=0, ge=0)
    workers: int = Field(default=12, ge=0)
    draws: int = Field(default=2000, ge=200)
    seed: int = 20260914


def coverage_weighted_select_triplets(
    pool: CandidatePool,
    budget: int,
    sigma_fraction: float,
    radius: float,
    study: str,
    kind: tuple,
    training_seed: int | None,
    epoch: int = 0,
) -> Selection:
    if budget < 1:
        raise ValueError("Sampling requires a positive budget")
    sigma = sigma_fraction * limiting_reach(pool) if pool.physical else None
    if pool.physical and (not np.isfinite(sigma) or sigma <= 0):
        raise ValueError(
            "Sampling requires a finite positive sigma for a physical pool"
        )
    groups = geometry_groups(pool, radius)
    count = min(budget, len(groups))
    metadata = {
        **pool.diagnostics,
        "requested_triplets": budget,
        "capacity": len(groups),
        "sampling_sigma_fraction": sigma_fraction,
        "sampling_sigma_mm": sigma,
        "triplets": 0,
        "channel_appearances": 0,
        "unique_source_slices": 0,
        "distinct_middle_geometry": 0,
        "shortfall": budget,
        "targets": [],
        "assignments": [],
    }
    if not count:
        return Selection((), metadata)

    training = training_seed is not None

    def rng(operation):
        return np.random.default_rng(
            operation_seed(training_seed, study, kind, operation, epoch)
        )

    percentiles = (
        np.arange(count) + (rng("targets").random(count) if training else 0.5)
    ) / count
    targets = target_depths(pool, percentiles, sigma)
    sources = pool.ranking
    supplier_rng = rng("suppliers") if training else None
    entitlement = dict.fromkeys(sources, Fraction(0))
    assigned = dict.fromkeys(sources, 0)
    preferred = [None] * count
    eligible = [
        {v.uid for v in pool.candidates if v.support[0] <= t <= v.support[1]}
        for t in targets
    ]
    coverage = {
        s: max(
            float(
                np.diff(
                    union_support(tuple(v for v in pool.candidates if v.uid == s)),
                    axis=1,
                ).sum()
            ),
            1e-3,
        )
        for s in sources
    }
    order = rng("band_order").permutation(count) if training else np.argsort(targets)
    for row in order:
        choices = [s for s in sources if s in eligible[row]]
        total = sum(coverage[s] for s in choices)
        for source in choices:
            entitlement[source] += Fraction(coverage[source] / total)
        source = (
            choices[int(supplier_rng.integers(len(choices)))]
            if training
            else max(choices, key=lambda s: entitlement[s] - assigned[s])
        )
        preferred[row] = source
        assigned[source] += 1
    ordered = [
        [
            (value, value.uid, value.support[0], value.support[1], value.depth, middle)
            for value in group
            for middle in (str(value.records[1].path),)
        ]
        for group in groups
    ]
    alternatives, errors = {}, []
    for row, target in enumerate(targets):
        wanted = preferred[row]
        for column, group in enumerate(ordered):
            best = None
            best_key = None
            for value, uid, low, high, depth, middle in group:
                if low <= target <= high:
                    key = (uid != wanted, (depth - target) ** 2, uid, middle)
                    if best_key is None or key < best_key:
                        best, best_key = value, key
            if best is not None:
                error = (best.depth - target) ** 2
                alternatives[row, column] = (best, error)
                errors.append(error)
    scale = max(errors, default=0.0) or 1.0
    change_cost = count + 1.0
    missing_cost = count * (change_cost + 1) + 1
    costs = np.full((count, len(groups) + count), missing_cost)
    costs[:, : len(groups)] = np.inf
    for (row, col), (value, error) in alternatives.items():
        costs[row, col] = (value.uid != preferred[row]) * change_cost + error / scale
    rows, cols = linear_sum_assignment(costs)
    selected, placements = [], []
    for row, col in zip(rows, cols, strict=True):
        if col >= len(groups):
            continue
        value, error = alternatives[row, col]
        selected.append(value)
        placements.append(
            {
                "band": int(row),
                "target": float(targets[row]),
                "depth": value.depth,
                "error": float(np.sqrt(error)),
                "preferred_uid": preferred[row],
                "uid": value.uid,
                "supplier_changed": value.uid != preferred[row],
                "middle_path": str(value.records[1].path),
                "run": value.run,
                "support": value.support,
            }
        )
    metadata.update(
        triplets=len(selected),
        channel_appearances=3 * len(selected),
        unique_source_slices=len({r.path for v in selected for r in v.records}),
        distinct_middle_geometry=len(selected),
        shortfall=budget - len(selected),
        targets=targets.tolist(),
        assignments=placements,
        target_entitlements={s: float(value) for s, value in entitlement.items()},
        preferred_counts=assigned,
        realized_counts={s: sum(v.uid == s for v in selected) for s in sources},
    )
    return Selection(tuple(selected), metadata)


class PolicyDataset(KneeStudyDataset):
    def __init__(self, *args, groups: dict, mode: tuple | None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.groups = groups
        self.mode = mode

    def _ranked(self, study, sources):
        return source_ranking(
            dict(sources),
            self._study_references[study.uid].point,
            self.data.spacing_mm,
            self.data.image_size * self.data.spacing_mm / 2,
        )

    def _plan(self, study, slot, sources):
        explicit = self.groups.get((study.uid, slot))
        if explicit is not None:
            return explicit
        if self.mode is None or len(sources) < 2:
            return None
        ranked = self._ranked(study, sources)
        if self.mode[0] == "pass":
            return (frozenset({ranked[self.mode[1] % len(ranked)]}),)
        if self.mode[0] == "best":
            return (frozenset({ranked[0]}),)
        if self.mode[0] == "additive_all":
            return tuple(frozenset({uid}) for uid in ranked)
        raise ValueError(f"Unknown mode {self.mode}")

    def _prepare_slot(self, study, slot, candidates, sources, budget, failures):
        plan = self._plan(study, slot, sources)
        if plan is None:
            return super()._prepare_slot(
                study, slot, candidates, sources, budget, failures
            )
        values, metas = [], []
        for uids in plan:
            subset = [source for source in sources if source[0] in uids]
            if not subset:
                continue
            value, meta = super()._prepare_slot(
                study,
                slot,
                {uid: c for uid, c in candidates.items() if uid in uids},
                subset,
                budget,
                failures,
            )
            values.extend(value)
            metas.append(meta)
        merged = {
            **(metas[0] if metas else {}),
            "groups": len(metas),
            "triplets": sum(meta.get("triplets", 0) for meta in metas),
            "assignments": [a for meta in metas for a in meta.get("assignments", [])],
        }
        return tuple(values), merged


def install_gate(model):
    original = model._attention_scores
    model.tilted_gate = None

    def gated(tokens):
        scores = original(tokens)
        if model.tilted_gate is None:
            return scores
        image_tilted, allowed = model.tilted_gate
        per_image = tokens.shape[0] // image_tilted.numel()
        if per_image * image_tilted.numel() != tokens.shape[0]:
            raise ValueError("Token count does not divide into images")
        blocked = image_tilted.repeat_interleave(per_image)[:, None] & ~allowed[None]
        return scores.masked_fill(blocked, -1e4)

    model._attention_scores = gated


def gate_state(batch, tilted_uids, allowed, device):
    uids = [Path(paths[0][0]).parent.name for paths in batch["selected_paths"]]
    series_tilted = torch.tensor([uid in tilted_uids for uid in uids])
    keep = batch["series_presence"][
        batch["series_study_index"], batch["series_type_index"]
    ]
    image_series = batch["slice_mask"][keep].nonzero(as_tuple=True)[0]
    return series_tilted[keep][image_series].to(device), allowed.to(device)


def slot_plans(index, studies, series_types, obliquity, threshold):
    plans = {"standard": {}, "tilted": {}, "additive": {}}
    tilted_by_study, slots = {}, []
    for uid in studies:
        for number, values in enumerate(index[uid].slots):
            plane = series_types[number].plane
            tilted = frozenset(
                c.uid for c in values if obliquity.get(c.uid, -1.0) >= threshold
            )
            standard = frozenset(c.uid for c in values) - tilted
            if not tilted:
                continue
            modified = plane != "Axial" and bool(standard)
            slots.append(
                {
                    "study": uid,
                    "slot": number,
                    "plane": plane,
                    "tilted": sorted(tilted),
                    "standard": sorted(standard),
                    "modified": modified,
                }
            )
            if modified:
                plans["standard"][(uid, number)] = (standard,)
                plans["tilted"][(uid, number)] = (tilted,)
                plans["additive"][(uid, number)] = (standard, tilted)
                tilted_by_study[uid] = tilted_by_study.get(uid, frozenset()) | tilted
    return plans, tilted_by_study, slots


def predict(
    model,
    config,
    targets,
    studies,
    index,
    multiplier,
    device,
    workers,
    *,
    groups=None,
    mode=None,
    gate=None,
    allowed=None,
):
    dataset = PolicyDataset(
        studies,
        index,
        config.data,
        targets,
        None,
        multiplier,
        INFERENCE_IO_THREADS,
        groups=groups or {},
        mode=mode,
    )
    loader = build_data_loader(dataset, config.trainer, 1, num_workers=workers)
    outputs, triplets = {}, {}
    with torch.inference_mode():
        for batch in loader:
            uid = batch["study_uid"][0]
            triplets[uid] = sum(
                sum(Counter(a["uid"] for a in meta.get("assignments", [])).values())
                for meta in batch["sampling"][0]
            )
            model.tilted_gate = (
                gate_state(batch, gate[uid], allowed, device)
                if gate is not None and uid in gate
                else None
            )
            device_batch = move_batch(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs[uid] = (
                    torch.sigmoid(model(device_batch).float()).cpu().numpy()[0]
                )
    model.tilted_gate = None
    return outputs, triplets


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


def compare(truth, reference, candidate, counts, targets):
    base = aucs(truth.astype(np.int8), reference, targets)
    variant = aucs(truth.astype(np.int8), candidate, targets)
    delta = bootstrap_auc(truth, candidate, counts) - bootstrap_auc(
        truth, reference, counts
    )
    return {
        "reference_macro": base["macro_auc"],
        "macro": variant["macro_auc"],
        "delta": variant["macro_auc"] - base["macro_auc"],
        "low95": float(np.percentile(delta, 2.5)),
        "high95": float(np.percentile(delta, 97.5)),
        "positive_fraction": float((delta > 0).mean()),
        **{
            f"d_{name}": variant["per_target_auc"][name] - base["per_target_auc"][name]
            for name in targets
        },
    }


def score(ctx, name, threshold, affected, values, triplets):
    full = ctx["reference"].copy()
    for u in affected:
        full[ctx["position_of"][u]] = values[u]
    rows_idx = [ctx["position_of"][u] for u in affected]
    sub_truth = ctx["truth"][rows_idx]
    defined = [
        i
        for i in range(len(ctx["targets"]))
        if 0 < sub_truth[:, i].sum() < len(rows_idx)
    ]
    whole = compare(
        ctx["truth"], ctx["reference"], full, ctx["full_counts"], ctx["targets"]
    )
    part = compare(
        sub_truth[:, defined],
        ctx["reference"][rows_idx][:, defined],
        full[rows_idx][:, defined],
        bootstrap_counts(
            ctx["groups"][rows_idx],
            sub_truth[:, defined],
            ctx["settings"].draws,
            ctx["settings"].seed,
        ),
        [ctx["targets"][i] for i in defined],
    )
    ctx["rows"].append(
        {
            "run": ctx["run"],
            "setting": ctx["setting"],
            "threshold": threshold,
            "variant": name,
            "studies": len(affected),
            "triplets": None
            if triplets is None
            else int(sum(triplets[u] for u in affected)),
            "triplets_intact": int(sum(ctx["intact_triplets"][u] for u in affected)),
            **{f"full_{k}": v for k, v in whole.items()},
            "subset_delta": part["delta"],
            "subset_low95": part["low95"],
            "subset_high95": part["high95"],
        }
    )
    ctx["frames"].append(
        pl.DataFrame(
            {
                "run": ctx["run"],
                "setting": ctx["setting"],
                "threshold": threshold,
                "variant": name,
                STUDY: affected,
                **{t: full[rows_idx][:, i] for i, t in enumerate(ctx["targets"])},
            }
        )
    )
    print(
        f"{ctx['run']} {ctx['setting']} {name}"
        f"{'' if threshold is None else f' ge{threshold:g}'}: "
        f"full {whole['delta']:+.5f} [{whole['low95']:+.5f}, {whole['high95']:+.5f}] "
        f"subset {part['delta']:+.4f} ({len(affected)} studies)",
        flush=True,
    )


def run(settings: PolicyCLI) -> None:
    started = time.perf_counter()
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    configure_cuda_allocator()
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    census = pl.read_parquet(settings.census).filter(pl.col("split") == "train")
    obliquity = dict(
        census.filter(pl.col("obliquity_deg").is_not_null())
        .select("SeriesInstanceUID", "obliquity_deg")
        .iter_rows()
    )
    rows, frames, meta = [], [], {"runs": {}}
    for run_dir in settings.run_dirs:
        config = load_config(run_dir / "effective_config.json")
        labels = load_fold_labels(config, settings.fold)
        targets = labels.targets
        validation = labels.validation.sort(STUDY)
        studies = validation[STUDY].to_list()
        position_of = {uid: i for i, uid in enumerate(studies)}
        truth = (
            validation.select(f"target_{t}" for t in targets).to_numpy()
            >= TARGET_THRESHOLD
        ).astype(np.float64)
        groups = validation["report_group"].to_numpy()
        full_counts = bootstrap_counts(groups, truth, settings.draws, settings.seed)
        selected = json.loads(
            (run_dir / f"fold_{settings.fold}/selected_checkpoint.json").read_text()
        )
        checkpoint = run_dir / f"fold_{settings.fold}" / Path(selected["path"]).name
        index = build_series_index(
            config.paths.data_root, config.data, "train", studies
        )
        tilted_plans = {}
        for threshold in settings.thresholds:
            plans, tilted_by_study, slots = slot_plans(
                index, studies, config.data.series_types, obliquity, threshold
            )
            tilted_plans[threshold] = (plans, tilted_by_study)
            (
                settings.output_dir / f"slots_{run_dir.name}_ge{threshold:g}.json"
            ).write_text(json.dumps(slots, indent=1) + "\n")
        multi = [u for u in studies if any(len(s) > 1 for s in index[u].slots)]
        passes = {u: max(len(s) for s in index[u].slots) for u in multi}
        meta["runs"][run_dir.name] = {
            "checkpoint": str(checkpoint),
            "multi_series_studies": len(multi),
        }
        for setting in settings.settings:
            multiplier = int(setting.split("-")[0][1:])
            tau = float(setting.split("-tau")[1])
            model = load_checkpoint_model(
                config,
                targets,
                checkpoint,
                device,
                expected_fold=settings.fold,
                attention_temperature=tau,
            )
            model.eval()
            install_gate(model)
            clock = time.perf_counter()
            common = (model, config, targets)
            intact, intact_triplets = predict(
                *common, validation, index, multiplier, device, settings.workers
            )
            reference = np.stack([intact[u] for u in studies]).astype(np.float64)
            frames.append(
                pl.DataFrame(
                    {
                        "run": run_dir.name,
                        "setting": setting,
                        "threshold": None,
                        "variant": "intact",
                        STUDY: studies,
                        **{t: reference[:, i] for i, t in enumerate(targets)},
                    }
                )
            )

            ctx = {
                "reference": reference,
                "position_of": position_of,
                "truth": truth,
                "groups": groups,
                "targets": targets,
                "full_counts": full_counts,
                "run": run_dir.name,
                "setting": setting,
                "intact_triplets": intact_triplets,
                "rows": rows,
                "frames": frames,
                "settings": settings,
            }

            acl = targets.index("ACL")
            for threshold, (plans, tilted_by_study) in tilted_plans.items():
                affected = sorted(tilted_by_study)
                subset = validation.filter(pl.col(STUDY).is_in(affected))
                made = {}
                for name, (sampling, gate) in TILTED.items():
                    needed = name in settings.variants or (
                        name in ("standard", "tilted")
                        and any(v in settings.variants for v in TILTED_POST)
                    )
                    if not needed:
                        continue
                    allowed = None
                    if gate is not None:
                        allowed = torch.tensor(
                            [t in GATE_LABELS[gate] for t in targets]
                        )
                    made[name] = predict(
                        *common,
                        subset,
                        index,
                        multiplier,
                        device,
                        settings.workers,
                        groups=None if sampling == "intact" else plans[sampling],
                        gate=tilted_by_study if gate else None,
                        allowed=allowed,
                    )
                    if name in settings.variants:
                        score(ctx, name, threshold, affected, *made[name])
                if "mean_standard_tilted" in settings.variants:
                    mean = {
                        u: (made["standard"][0][u] + made["tilted"][0][u]) / 2
                        for u in affected
                    }
                    score(ctx, "mean_standard_tilted", threshold, affected, mean, None)
                if "route_acl" in settings.variants:
                    routed = {}
                    for u in affected:
                        value = made["standard"][0][u].copy()
                        value[acl] = (
                            made["standard"][0][u][acl] + made["tilted"][0][u][acl]
                        ) / 2
                        routed[u] = value
                    score(ctx, "route_acl", threshold, affected, routed, None)
            subset = validation.filter(pl.col(STUDY).is_in(multi))
            if "passes" in settings.variants:
                sums, counts, total = {}, Counter(), Counter()
                for k in range(max(passes.values())):
                    members = [u for u in multi if passes[u] > k]
                    values, triplets = predict(
                        *common,
                        subset.filter(pl.col(STUDY).is_in(members)),
                        index,
                        multiplier,
                        device,
                        settings.workers,
                        mode=("pass", k),
                    )
                    for u in members:
                        sums[u] = sums.get(u, 0) + values[u]
                        counts[u] += 1
                        total[u] += triplets[u]
                score(
                    ctx,
                    "passes",
                    None,
                    multi,
                    {u: sums[u] / counts[u] for u in multi},
                    total,
                )
            if "best_only" in settings.variants:
                score(
                    ctx,
                    "best_only",
                    None,
                    multi,
                    *predict(
                        *common,
                        subset,
                        index,
                        multiplier,
                        device,
                        settings.workers,
                        mode=("best",),
                    ),
                )
            if "additive_all" in settings.variants:
                score(
                    ctx,
                    "additive_all",
                    None,
                    multi,
                    *predict(
                        *common,
                        subset,
                        index,
                        multiplier,
                        device,
                        settings.workers,
                        mode=("additive_all",),
                    ),
                )
            if "coverage_weighted" in settings.variants:
                original = dataset_module.select_triplets
                dataset_module.select_triplets = coverage_weighted_select_triplets
                try:
                    values, triplets = predict(
                        *common, subset, index, multiplier, device, settings.workers
                    )
                finally:
                    dataset_module.select_triplets = original
                score(ctx, "coverage_weighted", None, multi, values, triplets)
            if "intact_double" in settings.variants:
                score(
                    ctx,
                    "intact_double",
                    None,
                    multi,
                    *predict(
                        *common,
                        subset,
                        index,
                        2 * multiplier,
                        device,
                        settings.workers,
                    ),
                )
            meta["runs"][run_dir.name][setting] = {
                "intact_macro": aucs(truth.astype(np.int8), reference, targets)[
                    "macro_auc"
                ],
                "seconds": time.perf_counter() - clock,
            }
            print(
                f"{run_dir.name} {setting} {meta['runs'][run_dir.name][setting]}",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()
    table = pl.DataFrame(rows)
    table.write_csv(settings.output_dir / "results.csv")
    pl.concat(frames, how="vertical_relaxed").write_parquet(
        settings.output_dir / "predictions.parquet"
    )
    aggregate = (
        table.group_by("setting", "threshold", "variant")
        .agg(
            pl.col("studies").first(),
            pl.col("full_delta").mean().alias("mean_full_delta"),
            (pl.col("full_delta") > 0).sum().alias("runs_positive"),
            pl.col("full_delta").min().alias("min_full_delta"),
            pl.col("full_delta").max().alias("max_full_delta"),
            pl.col("subset_delta").mean().alias("mean_subset_delta"),
            (pl.col("triplets") / pl.col("triplets_intact"))
            .mean()
            .alias("triplet_ratio"),
        )
        .sort(
            "setting", "threshold", "mean_full_delta", descending=[False, False, True]
        )
    )
    aggregate.write_csv(settings.output_dir / "aggregate.csv")
    meta.update(
        census=str(settings.census),
        fold=settings.fold,
        thresholds=settings.thresholds,
        settings=settings.settings,
        variants=settings.variants,
        tilted_rule="coronal and sagittal slots holding both a series at or above the threshold and another series; axial slots untouched",
        general_rule="every slot with two or more usable series; series ordered by the production source ranking",
        gate="tilted-series tokens receive attention score -1e4 for every label outside the allowed set before the per-label softmax",
        precision="bf16 autocast, batch size 1",
        bootstrap={
            "unit": "report_group",
            "draws": settings.draws,
            "seed": settings.seed,
        },
        seconds=time.perf_counter() - started,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
    )
    (settings.output_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    with pl.Config(tbl_rows=80, tbl_cols=12, tbl_width_chars=220, float_precision=5):
        print(aggregate)


def main() -> None:
    try:
        run(PolicyCLI())
    except (KeyError, OSError, TypeError, ValueError, SettingsError) as error:
        print(f"Tilted slot policy scoring error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
