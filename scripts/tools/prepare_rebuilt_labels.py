import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.data.labels import STUDY, TARGET_THRESHOLD, require_unique_ids
from knee_abnormality.kaggle import COMPETITION
from scripts.tools.prepare_combined_labels import fingerprint
from scripts.tools.prepare_image_blend_labels import base_labels

STATES = ("NEG", "NOT_MENTIONED", "LOW", "HEDGED", "POS")
NEG, NM, LOW, HEDGED, POS = range(len(STATES))
TEMPERATURES = (1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1)
BIN_EDGES = (0.1, 0.3, 0.5, 0.7, 0.9)
NATIVE_SYMBOLS = 12
FAMILY_KAPPA = 0.9
ANCHOR_LEAK = 0.02
EM_ITERATIONS = 300
EM_TOLERANCE = 1e-7
STATE_PRIOR_SD = 2.5
TARGET_PRIOR_SD = 0.5
SLOPE_PRIOR_SD = 1.0
Z_CLIP = 0.02
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260927
LOSS_CLIP = 1e-4
STATE_TABLE = "rebuild_state_v2"
MEAN_TABLE = "rebuild_mean_v2"
STATE_RATE = {NM: 0.1347, LOW: 0.3448, HEDGED: 0.6125}
SILENT_SYMBOLS = ("nm", "unk")
DERIVED = {"zhen_v7": "pilkwang"}
PUBLIC = Path("data/labels/public")
SNAPSHOT = "downloaded-{}"


@dataclass
class Root:
    name: str
    paths: list[Path]
    gold_excluded: str | None = None
    symbols: dict[str, pl.Series] = field(default_factory=dict)
    values: dict[str, pl.Series] = field(default_factory=dict)
    anchors: dict[str, dict[str, dict[int, float]]] = field(default_factory=dict)


class PrepareCLI(CLISettings):
    model_config = describe(
        "Rebuild all twelve training-label targets from lineage-level public report sources "
        "and JevK5 attributes on a common state scale: rebuild_state_v2 (anchored Dawid-Skene "
        "report states with a pooled gold semantic calibration) and rebuild_mean_v2 "
        "(family-mean control), with "
        "cross-fitted gold evaluation"
    )

    base_manifest: Path = Field(
        default=Path("configs/labels/combined-label-sources.json"),
        description="Pinned combined_v1 label manifest; its gold rows fill the tables' gold rows",
    )
    jevk5_attributes: Path = Field(
        default=Path("outputs/labels/jevk5-attributes-v3/attributes.parquet"),
        description="JevK5 v3 per-question option probabilities for every report",
    )
    split_studies: Path = Field(
        default=Path("data/folds/all_5fold_scanner_v1/studies.csv"),
        description="Split study table supplying gold report groups for cross-fitting",
    )
    output_dir: Path = Field(
        default=Path("data/labels/prepared"),
        description="Parent directory for the two fresh prepared table directories",
    )
    manifest_dir: Path = Field(
        default=Path("configs/labels"),
        description="Directory for the two fresh label manifests",
    )
    analysis_dir: Path = Field(
        description="Fresh directory for families, tempering, states, calibration and gold results",
    )


def public(owner: str, slug: str, date: str, name: str) -> Path:
    return PUBLIC / owner / slug / SNAPSHOT.format(date) / name


def binned(values: pl.Series) -> pl.Series:
    labels = [f"b{i}" for i in range(len(BIN_EDGES) + 1)]
    return values.cut(list(BIN_EDGES), labels=labels, left_closed=True).cast(pl.String)


def native_or_binned(values: pl.Series) -> pl.Series:
    if values.drop_nulls().n_unique() <= NATIVE_SYMBOLS:
        return values.round(3).cast(pl.String)
    return binned(values)


def value_root(
    root: Root,
    frame: pl.DataFrame,
    targets: list[str],
    columns: dict[str, str] | None = None,
    not_mentioned: float | None = None,
) -> Root:
    for target in targets:
        values = frame[(columns or {}).get(target, target)].cast(pl.Float64)
        symbols = native_or_binned(values)
        if not_mentioned is not None:
            silent = (values == not_mentioned).fill_null(False).to_numpy()
            symbols = pl.Series(
                np.where(silent, "nm", symbols.to_numpy().astype(object)),
                dtype=pl.String,
            )
        root.symbols[target] = symbols
        root.values[target] = values
    return root


def aligned(frame: pl.DataFrame, studies: pl.DataFrame) -> pl.DataFrame:
    frame = frame.with_columns(pl.col(STUDY).cast(pl.String))
    require_unique_ids(frame, "root source")
    return studies.join(frame, on=STUDY, how="left", validate="1:1")


def load_roots(studies: pl.DataFrame, targets: list[str], jevk5: Path) -> list[Root]:
    snake = {t: t.lower().replace(" ", "_").replace("'s", "s") for t in targets}
    under = {t: t.replace(" ", "_").replace("'", "") for t in targets}
    roots = []

    root = Root("jevk5", [jevk5])
    roots.append(jevk5_root(root, studies, targets, jevk5))

    path = public(
        "ramitnandi", "rsna-knee-labels-gpt-luna", "20260921", "report_labels.csv"
    )
    long = pl.read_csv(path, infer_schema_length=None).with_columns(
        pl.col("CLASS").replace({v: k for k, v in under.items()}),
        pl.when(pl.col("classification") == "NA")
        .then(pl.lit("nm"))
        .when(pl.col("classification") == "NEGATIVE")
        .then(
            pl.when(pl.col("hedged"))
            .then(pl.lit("neg_hedged"))
            .otherwise(pl.lit("neg"))
        )
        .otherwise(
            pl.when(pl.col("hedged"))
            .then(pl.lit("pos_hedged"))
            .otherwise(pl.lit("pos_") + pl.col("severity"))
        )
        .alias("symbol"),
        pl.col("SCORE").cast(pl.Float64),
    )
    symbols = aligned(long.pivot(on="CLASS", index=STUDY, values="symbol"), studies)
    scores = aligned(long.pivot(on="CLASS", index=STUDY, values="SCORE"), studies)
    root = Root("ramit", [path])
    for target in targets:
        root.symbols[target] = symbols[target]
        root.values[target] = scores[target]
    state = {"nm": {NM: 1}, "neg": {NEG: 1}, "neg_hedged": {NEG: 0.5, HEDGED: 0.5}}
    state |= {"pos_LOW": {LOW: 1}, "pos_MEDIUM": {POS: 1}, "pos_HIGH": {POS: 1}}
    state |= {"pos_hedged": {HEDGED: 1}}
    root.anchors = {t: state for t in targets}
    roots.append(root)

    path = public(
        "vmohitrao", "rsna-knee-report-labels", "20260921", "label_states.csv"
    )
    frame = aligned(pl.read_csv(path, infer_schema_length=None), studies)
    root = Root("vmohit", [path])
    for target in targets:
        root.symbols[target] = frame[target]
        root.values[target] = frame.select(
            pl.when(pl.col(target) == "P")
            .then(1.0)
            .when(pl.col(target).is_in(["N", "B"]))
            .then(0.0)
            .otherwise(None)
        ).to_series()
    state = {
        "P": {POS: 1},
        "N": {NEG: 1},
        "B": {LOW: 1},
        "U": {HEDGED: 1},
        "M": {NM: 1},
    }
    root.anchors = {t: state for t in targets}
    roots.append(root)

    path = Path("data/labels/public/pilkwang/version-1/report_labels_v2.csv")
    frame = aligned(pl.read_csv(path, infer_schema_length=None), studies)
    root = value_root(Root("pilkwang", [path]), frame, targets)
    state = {"0.08": {NEG: 1}, "0.28": {NM: 0.8, HEDGED: 0.2}, "0.68": {LOW: 1}}
    state |= {"0.82": {POS: 1}, "0.94": {POS: 1}}
    root.anchors = {t: state for t in targets}
    roots.append(root)

    path = Path("data/labels/public/stevenleehans/version-6/llm_labels_full.csv")
    frame = aligned(pl.read_csv(path, infer_schema_length=None), studies)
    roots.append(
        value_root(Root("steven_v1", [path]), frame, targets, not_mentioned=0.5)
    )

    path = Path("data/labels/public/lixin73/version-1/report_labels_gpt56sol.csv")
    frame = aligned(pl.read_csv(path, infer_schema_length=None), studies)
    root = Root("lixin73", [path])
    for target in targets:
        root.values[target] = frame[target].cast(pl.Float64)
        confident = (frame[f"{target}__conf"] >= 0.9).fill_null(False).to_numpy()
        values = native_or_binned(root.values[target]).to_numpy().astype(object)
        root.symbols[target] = pl.Series(
            [
                None if v is None else v + ("|hi" if c else "|lo")
                for v, c in zip(values, confident)
            ],
            dtype=pl.String,
        )
    roots.append(root)

    path = Path("data/labels/public/zhen931/version-1/report_labels_v7.csv")
    frame = aligned(pl.read_csv(path, infer_schema_length=None), studies)
    root = Root("zhen_v7", [path])
    for target in targets:
        values = frame[target].cast(pl.Float64)
        root.values[target] = values
        root.symbols[target] = frame.select(
            pl.when(pl.col(f"{target}__verdict") == "UNK")
            .then(pl.lit("nm"))
            .otherwise(pl.col(f"{target}__verdict") + pl.lit("|") + binned(values))
        ).to_series()
    roots.append(root)

    path = Path("data/labels/public/laymond/version-4/weak_labels_v2_filled.csv")
    frame = aligned(pl.read_csv(path, infer_schema_length=None), studies)
    root = Root("laymond", [path])
    for target in targets:
        confidence = frame[f"{target}__confidence"].cast(pl.Float64)
        root.values[target] = confidence
        root.symbols[target] = frame.select(
            pl.when(pl.col(f"{target}__mentioned") == 0)
            .then(pl.lit("nm"))
            .when(pl.col(f"{target}__negated") == 1)
            .then(pl.lit("neg"))
            .otherwise(
                pl.lit("l")
                + pl.col(f"{target}__label").cast(pl.String)
                + pl.lit("|")
                + binned(confidence)
            )
        ).to_series()
    roots.append(root)

    path = public(
        "yunusgmsoy", "rsna-knee-llm-report-labels", "20260921", "yunus_llm_labels.csv"
    )
    frame = aligned(pl.read_csv(path, infer_schema_length=None), studies)
    roots.append(value_root(Root("yunus", [path]), frame, targets))

    path = public(
        "riadmohamed42", "jev-knee-labels", "20260921", "FINAL2_labels_scores.csv"
    )
    frame = aligned(pl.read_csv(path, infer_schema_length=None), studies)
    roots.append(value_root(Root("jev_text", [path], "gold-tuned"), frame, targets))

    path = public(
        "dreaddevelopment", "rsna-knee-labels", "20260921", "labels_llm_soft.csv"
    )
    frame = aligned(pl.read_csv(path, infer_schema_length=None), studies)
    roots.append(value_root(Root("dreaddev", [path]), frame, targets))

    path = public(
        "joshuaziel",
        "rsna-knee-mri-report-labels-gpt-5-6-luna",
        "20260921",
        "all_raw_predictions.json",
    )
    records = pl.DataFrame(json.loads(path.read_text()))
    top = records["confidence"].max()
    findings = {t.replace(" ", "_").replace("'s", ""): t for t in targets}
    records = records.with_columns(
        pl.col("Finding").replace(findings),
        (
            pl.col("evidence_basis")
            + pl.lit("|")
            + pl.col("annotation").cast(pl.String)
        ).alias("symbol"),
        pl.when(pl.col("annotation") == 1)
        .then(0.5 + pl.col("confidence") / (2 * top))
        .otherwise(0.5 - pl.col("confidence") / (2 * top))
        .alias("score"),
    )
    symbols = aligned(
        records.pivot(on="Finding", index=STUDY, values="symbol"), studies
    )
    scores = aligned(records.pivot(on="Finding", index=STUDY, values="score"), studies)
    root = Root("josh", [path], "gold-tuned")
    for target in targets:
        root.symbols[target] = symbols[target]
        root.values[target] = scores[target]
    roots.append(root)

    path = public(
        "nartaa", "rsna-knee-hpo-assets", "20260922", "train_labels_v1.parquet"
    )
    frame = aligned(pl.read_parquet(path), studies)
    root = Root("nartaa", [path])
    for target in targets:
        values = frame[f"y_{under[target]}"].cast(pl.Float64)
        root.values[target] = values
        root.symbols[target] = frame.select(
            pl.when(pl.col(f"m_{under[target]}") == 0)
            .then(pl.lit("nm"))
            .otherwise(native_or_binned(values))
        ).to_series()
    roots.append(root)

    path = public("xhaitanya", "qwen38b-labels-rsnaknee", "20260927", "llm_labels.csv")
    frame = aligned(pl.read_csv(path, infer_schema_length=None), studies)
    roots.append(value_root(Root("qwen3_8b", [path]), frame, targets, snake))

    paths = [
        public(
            "bhaveshkolhe", "knee-llm-labels", "20260928", f"labels_full_shard{i}.csv"
        )
        for i in (0, 1)
    ]
    frame = aligned(
        pl.concat([pl.read_csv(p, infer_schema_length=None) for p in paths]), studies
    )
    roots.append(value_root(Root("bhavesh", paths), frame, targets))

    path = public(
        "hosseini7798",
        "rsna-knee-label-extraction-jev",
        "20260922",
        "Label_Extraction_Jev_v2.csv",
    )
    frame = aligned(pl.read_csv(path, infer_schema_length=None).drop("Report"), studies)
    root = Root("hosseini", [path], "possibly gold-tuned")
    for target in targets:
        negative, positive, unknown = (pl.col(f"{target}_{k}") for k in (0, 1, 9))
        root.values[target] = frame[f"{target}_1"].cast(pl.Float64)
        root.symbols[target] = frame.select(
            pl.when(unknown.is_null())
            .then(None)
            .when((unknown > negative) & (unknown > positive))
            .then(pl.lit("unk"))
            .when(negative >= positive)
            .then(pl.lit("neg"))
            .when(positive >= 0.9)
            .then(pl.lit("pos_hi"))
            .otherwise(pl.lit("pos"))
        ).to_series()
    roots.append(root)

    path = public(
        "pearsejim01", "rsna-2026-eda", "20260922", "weak_labels_frontier.parquet"
    )
    frame = aligned(pl.read_parquet(path), studies)
    root = Root("pearse", [path])
    for target in targets:
        values = frame[target].cast(pl.Float64)
        root.values[target] = values
        root.symbols[target] = frame.select(
            pl.when(pl.col(f"{target}__npos") + pl.col(f"{target}__nneg") == 0)
            .then(pl.lit("nm"))
            .otherwise(binned(values))
        ).to_series()
    roots.append(root)

    path = public(
        "dhiyaneshbabu", "rsna-knee-labelled-train", "20260928", "labelled_train.csv"
    )
    frame = aligned(pl.read_csv(path, infer_schema_length=None), studies)
    roots.append(value_root(Root("opus_sonnet", [path], "gold copied"), frame, targets))

    path = public("yehezkielhaganta", "psuedo-labeling-knee", "20260928", "data.csv")
    frame = aligned(pl.read_csv(path, infer_schema_length=None).drop("Report"), studies)
    root = value_root(Root("yehezkiel", [path], "gold copied"), frame, targets)
    for target in targets:
        root.symbols[target] = frame.select(
            pl.when(pl.col(target) == -1)
            .then(pl.lit("nm"))
            .otherwise(pl.col(target).cast(pl.Float64).round(3).cast(pl.String))
        ).to_series()
        root.values[target] = frame.select(
            pl.when(pl.col(target) == -1)
            .then(None)
            .otherwise(pl.col(target).cast(pl.Float64))
        ).to_series()
    roots.append(root)
    return roots


def status_map(
    options: dict[str, int | dict[int, float]],
) -> dict[str, dict[int, float]]:
    return {k: v if isinstance(v, dict) else {v: 1.0} for k, v in options.items()}


SILENT = {NEG: 0.5, NM: 0.5}
JEVK5_STATUS: dict[str, tuple[str, dict[str, dict[int, float]]]] = {
    "ACL": (
        "acl_status",
        status_map(
            {
                "not_mentioned": NM,
                "intact": NEG,
                "degeneration_or_sprain": LOW,
                "possible_tear": HEDGED,
                "tear_unspecified": POS,
                "partial_unspecified": HEDGED,
                "partial_low_grade": LOW,
                "partial_high_grade": POS,
                "complete_tear": POS,
                "graft": NEG,
            }
        ),
    ),
    "MCL": (
        "mcl_status",
        status_map(
            {
                "not_mentioned": NM,
                "intact": NEG,
                "sprain_or_thickening": LOW,
                "possible_tear": HEDGED,
                "partial_unspecified": HEDGED,
                "partial_high_grade": POS,
                "complete_tear": POS,
            }
        ),
    ),
    **{
        target: (
            question,
            status_map(
                {
                    "not_mentioned": NM,
                    "normal": NEG,
                    "degeneration": LOW,
                    "possible_tear": HEDGED,
                    "tear": POS,
                    "displaced_or_truncated": POS,
                }
            ),
        )
        for target, question in (
            ("Medial Meniscus", "medial_meniscus_status"),
            ("Lateral Meniscus", "lateral_meniscus_status"),
        )
    },
    "Effusion": (
        "effusion_size",
        status_map(
            {
                "not_mentioned": NM,
                "none": NEG,
                "small": LOW,
                "moderate": POS,
                "large": POS,
                "unspecified": HEDGED,
            }
        ),
    ),
    "Synovitis": (
        "synovitis_status",
        status_map({"absent": SILENT, "present": POS, "possible": HEDGED}),
    ),
    "Baker's": (
        "baker_size",
        status_map(
            {
                "absent": SILENT,
                "other_cyst_only": NEG,
                "small": LOW,
                "moderate": POS,
                "large": POS,
                "unspecified": HEDGED,
            }
        ),
    ),
    "Contusion": (
        "contusion_status",
        status_map(
            {
                "none": SILENT,
                "contusion": POS,
                "possible": HEDGED,
                "other_edema_only": LOW,
            }
        ),
    ),
    "Fracture": (
        "fracture_status",
        status_map(
            {
                "none": SILENT,
                "acute": POS,
                "possible": HEDGED,
                "stress_or_insufficiency": LOW,
                "old": LOW,
            }
        ),
    ),
}
JEVK5_OA = {"Medial OA": "medial", "Lateral OA": "lateral", "PF OA": "pf"}
DEPTH_STATE = {
    "not_mentioned": NM,
    "normal": NEG,
    "only_elsewhere": NEG,
    "softening": LOW,
    "partial_low": LOW,
    "unspecified_loss": HEDGED,
}
HIGH_DEPTH = ("partial_high", "full_thickness", "bone_on_bone")
WIDE_EXTENT = ("large", "diffuse")


def oa_state(depth: str, extent: str) -> dict[int, float]:
    if depth in DEPTH_STATE:
        return {DEPTH_STATE[depth]: 1.0}
    if extent in WIDE_EXTENT:
        return {POS: 1.0}
    if extent == "focal":
        return {LOW: 1.0}
    return {HEDGED: 1.0}


def expected_positive(
    probabilities: dict[str, np.ndarray], states: dict[str, dict[int, float]]
) -> np.ndarray:
    return sum(
        probabilities[option] * (mass.get(POS, 0.0) + 0.5 * mass.get(HEDGED, 0.0))
        for option, mass in states.items()
    )


def jevk5_root(
    root: Root, studies: pl.DataFrame, targets: list[str], path: Path
) -> Root:
    answers = pl.read_parquet(path).with_columns(pl.col(STUDY).cast(pl.String))
    wide = answers.pivot(on=["question", "option"], index=STUDY, values="probability")
    wide = studies.join(
        wide.rename(
            {
                c: c.replace('{"', "").replace('"}', "").replace('","', "__")
                for c in wide.columns
            }
        ),
        on=STUDY,
        how="left",
        validate="1:1",
    )

    def options(question: str) -> dict[str, np.ndarray]:
        prefix = f"{question}__"
        return {
            c[len(prefix) :]: wide[c].to_numpy()
            for c in wide.columns
            if c.startswith(prefix)
        }

    def argmax(probabilities: dict[str, np.ndarray]) -> np.ndarray:
        names = list(probabilities)
        return np.array(names)[
            np.stack(list(probabilities.values()), axis=1).argmax(axis=1)
        ]

    for target in targets:
        if target in JEVK5_OA:
            side = JEVK5_OA[target]
            depth = options(f"{side}_cartilage_depth")
            extent = options(f"{side}_cartilage_extent")
            depth_choice, extent_choice = argmax(depth), argmax(extent)
            symbols = [
                f"{d}|{e if d in HIGH_DEPTH else '-'}"
                for d, e in zip(depth_choice, extent_choice)
            ]
            states = {}
            for d in depth:
                for e in extent:
                    states[f"{d}|{e if d in HIGH_DEPTH else '-'}"] = oa_state(d, e)
            high = sum(depth[d] for d in HIGH_DEPTH if d in depth)
            wide_mass = sum(extent[e] for e in WIDE_EXTENT)
            stated = (
                1
                - extent.get("not_stated", 0)
                - extent.get("no_loss", 0)
                - extent.get("only_elsewhere", 0)
            )
            hedged = depth["unspecified_loss"] + high * (1 - stated)
            value = high * wide_mass + 0.5 * hedged
        else:
            question, states = JEVK5_STATUS[target]
            probabilities = options(question)
            symbols = list(argmax(probabilities))
            value = expected_positive(probabilities, states)
        root.symbols[target] = pl.Series(symbols, dtype=pl.String)
        root.values[target] = pl.Series(np.clip(value, 0, 1), dtype=pl.Float64)
        root.anchors[target] = states
    return root


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    x, y = a[valid] >= TARGET_THRESHOLD, b[valid] >= TARGET_THRESHOLD
    observed = (x == y).mean()
    expected = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return float((observed - expected) / (1 - expected)) if expected < 1 else 1.0


def families(
    roots: list[Root], targets: list[str], silver: np.ndarray
) -> tuple[list[list[int]], np.ndarray]:
    count = len(roots)
    similarity = np.zeros((count, count))
    for a in range(count):
        for b in range(a + 1, count):
            similarity[a, b] = similarity[b, a] = np.mean(
                [
                    kappa(
                        roots[a].values[t].to_numpy()[silver],
                        roots[b].values[t].to_numpy()[silver],
                    )
                    for t in targets
                ]
            )
    groups: list[list[int]] = [[i] for i in range(count)]
    while True:
        best, pair = FAMILY_KAPPA, None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                link = min(similarity[a, b] for a in groups[i] for b in groups[j])
                if link >= best:
                    best, pair = link, (i, j)
        if pair is None:
            break
        i, j = pair
        groups[i] = groups[i] + groups.pop(j)
    names = [r.name for r in roots]
    for child, parent in DERIVED.items():
        source = next(g for g in groups if names.index(child) in g)
        destination = next(g for g in groups if names.index(parent) in g)
        if source is not destination:
            destination.extend(source)
            groups.remove(source)
    return sorted(sorted(g) for g in groups), similarity


def scaled_values(root: Root, target: str) -> np.ndarray:
    values = root.values[target].to_numpy().astype(float)
    symbols = root.symbols[target].to_numpy().astype(object)
    anchors = root.anchors.get(target, {})
    result = values.copy()
    for symbol in {s for s in symbols if s is not None}:
        mass = anchors.get(symbol)
        if mass is None and (
            symbol in SILENT_SYMBOLS or str(symbol).startswith("NOT_MENTIONED")
        ):
            mass = {NM: 1.0}
        if mass is None or not any(state in STATE_RATE for state in mass):
            continue
        rows = symbols == symbol
        result[rows] = sum(
            weight * (STATE_RATE[state] if state in STATE_RATE else values[rows])
            for state, weight in mass.items()
        )
    return result


@dataclass
class Encoded:
    codes: list[np.ndarray]
    alphabets: list[list[str]]
    allowed: list[np.ndarray]
    anchor: list[np.ndarray | None]


def encode(roots: list[Root], target: str, masks: list[np.ndarray]) -> Encoded:
    encoded = Encoded([], [], [], [])
    for root, mask in zip(roots, masks):
        symbols = root.symbols[target].to_numpy().astype(object)
        present = np.array([s is not None for s in symbols]) & mask
        alphabet = sorted({s for s, p in zip(symbols, present) if p})
        index = {s: i for i, s in enumerate(alphabet)}
        codes = np.array([index[s] if p else -1 for s, p in zip(symbols, present)])
        allowed = np.ones((len(STATES), len(alphabet)))
        anchor = None
        if target in root.anchors:
            anchor = np.zeros((len(alphabet), len(STATES)))
            for symbol, i in index.items():
                for state, mass in root.anchors[target].get(symbol, {}).items():
                    anchor[i, state] = mass
            allowed = np.where(anchor.T > 0, 1.0, ANCHOR_LEAK)
        encoded.codes.append(codes)
        encoded.alphabets.append(alphabet)
        encoded.allowed.append(allowed)
        encoded.anchor.append(anchor)
    return encoded


def log_likelihoods(
    encoded: Encoded, thetas: list[np.ndarray], skip: int | None = None
) -> np.ndarray:
    rows = len(encoded.codes[0])
    total = np.zeros((rows, len(STATES)))
    for r, (codes, theta) in enumerate(zip(encoded.codes, thetas)):
        if r == skip:
            continue
        seen = codes >= 0
        total[seen] += np.log(theta[:, codes[seen]]).T
    return total


def posterior(
    encoded: Encoded,
    prior: np.ndarray,
    thetas: list[np.ndarray],
    tau: float,
    skip: int | None = None,
) -> np.ndarray:
    scores = np.log(prior) + tau * log_likelihoods(encoded, thetas, skip)
    return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))


def fit_states(
    encoded: Encoded, tau: float
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    rows = len(encoded.codes[0])
    start = np.full((rows, len(STATES)), 0.01)
    for codes, anchor in zip(encoded.codes, encoded.anchor):
        if anchor is not None:
            seen = codes >= 0
            start[seen] += anchor[codes[seen]]
    q = start / start.sum(axis=1, keepdims=True)
    for _ in range(EM_ITERATIONS):
        prior = (q.sum(axis=0) + 1) / (rows + len(STATES))
        thetas = []
        for codes, allowed in zip(encoded.codes, encoded.allowed):
            theta = np.ones_like(allowed)
            seen = codes >= 0
            np.add.at(theta.T, codes[seen], q[seen])
            theta *= allowed
            thetas.append(theta / theta.sum(axis=1, keepdims=True))
        updated = posterior(encoded, prior, thetas, tau)
        change = np.abs(updated - q).max()
        q = updated
        if change < EM_TOLERANCE:
            break
    return prior, thetas, q


def held_out_likelihood(
    encoded: Encoded,
    prior: np.ndarray,
    thetas: list[np.ndarray],
    tau: float,
    rows: np.ndarray,
) -> float:
    total, count = 0.0, 0
    for r, (codes, theta) in enumerate(zip(encoded.codes, thetas)):
        seen = (codes >= 0) & rows
        if not seen.any():
            continue
        others = posterior(encoded, prior, thetas, tau, skip=r)[seen]
        total += np.log((others * theta[:, codes[seen]].T).sum(axis=1)).sum()
        count += int(seen.sum())
    return total / count


def centred_logit(values: np.ndarray, centre: float) -> np.ndarray:
    return logit(np.clip(values, Z_CLIP, 1 - Z_CLIP)) - centre


def calibrated(
    params: np.ndarray, q: np.ndarray, target: np.ndarray, z: np.ndarray, targets: int
) -> np.ndarray:
    beta, alpha, gamma = (
        params[: len(STATES)],
        params[len(STATES) : len(STATES) + targets],
        params[-1],
    )
    return (q * expit(beta[None] + alpha[target][:, None] + gamma * z[:, None])).sum(
        axis=1
    )


def fit_calibration(
    q: np.ndarray, target: np.ndarray, z: np.ndarray, y: np.ndarray, targets: int
) -> np.ndarray:
    k = len(STATES)
    sd = np.r_[
        np.full(k, STATE_PRIOR_SD), np.full(targets, TARGET_PRIOR_SD), SLOPE_PRIOR_SD
    ]

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        beta, alpha, gamma = params[:k], params[k : k + targets], params[-1]
        s = expit(beta[None] + alpha[target][:, None] + gamma * z[:, None])
        p = np.clip((q * s).sum(axis=1), 1e-9, 1 - 1e-9)
        loss = (
            -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
            + 0.5 * ((params / sd) ** 2).sum()
        )
        dp = -(y / p - (1 - y) / (1 - p))
        slope = q * s * (1 - s) * dp[:, None]
        gradient = (
            np.r_[
                slope.sum(axis=0),
                np.bincount(target, weights=slope.sum(axis=1), minlength=targets),
                (slope.sum(axis=1) * z).sum(),
            ]
            + params / sd**2
        )
        return loss, gradient

    return minimize(objective, np.zeros(k + targets + 1), jac=True, method="L-BFGS-B").x


def pair_matrix(
    scores: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive, negative = truth == 1, truth == 0
    difference = scores[positive][:, None] - scores[negative][None]
    return (difference > 0) + 0.5 * (difference == 0), positive, negative


def gold_metrics(
    arms: dict[str, np.ndarray], truth: np.ndarray, targets: list[str]
) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    studies = truth.shape[0]
    draws = np.stack(
        [
            np.bincount(rng.integers(0, studies, studies), minlength=studies)
            for _ in range(BOOTSTRAP_DRAWS)
        ]
    )
    valid = np.ones(BOOTSTRAP_DRAWS, dtype=bool)
    for t in range(len(targets)):
        valid &= (draws[:, truth[:, t] == 1].sum(axis=1) > 0) & (
            draws[:, truth[:, t] == 0].sum(axis=1) > 0
        )
    draws = draws[valid].astype(float)
    results = {}
    for name, scores in arms.items():
        aucs, boot_auc = [], np.zeros((len(draws), len(targets)))
        for t in range(len(targets)):
            pairs, positive, negative = pair_matrix(scores[:, t], truth[:, t])
            aucs.append(float(pairs.mean()))
            wp, wn = draws[:, positive], draws[:, negative]
            boot_auc[:, t] = ((wp @ pairs) * wn).sum(axis=1) / (
                wp.sum(axis=1) * wn.sum(axis=1)
            )
        clipped = np.clip(scores, LOSS_CLIP, 1 - LOSS_CLIP)
        cell = -(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped))
        results[name] = {
            "macro_auc": float(np.mean(aucs)),
            "per_target_auc": dict(zip(targets, aucs)),
            "log_loss": float(cell.mean()),
            "boot_macro": boot_auc.mean(axis=1),
            "boot_loss": (draws @ cell.sum(axis=1))
            / (draws.sum(axis=1) * len(targets)),
        }
    reference = results["combined_v1"]
    report = {"valid_draws": int(valid.sum())}
    for name, result in results.items():
        entry = {k: v for k, v in result.items() if not k.startswith("boot")}
        if name != "combined_v1":
            auc_delta = result["boot_macro"] - reference["boot_macro"]
            loss_delta = result["boot_loss"] - reference["boot_loss"]
            entry |= {
                "macro_auc_delta": result["macro_auc"] - reference["macro_auc"],
                "macro_auc_delta_95": [
                    float(np.percentile(auc_delta, 2.5)),
                    float(np.percentile(auc_delta, 97.5)),
                ],
                "p_macro_auc_better": float((auc_delta > 0).mean()),
                "log_loss_delta": result["log_loss"] - reference["log_loss"],
                "log_loss_delta_95": [
                    float(np.percentile(loss_delta, 2.5)),
                    float(np.percentile(loss_delta, 97.5)),
                ],
            }
        report[name] = entry
    return report


def write_table(
    name: str,
    values: np.ndarray,
    studies: pl.DataFrame,
    base: pl.DataFrame,
    targets: list[str],
    settings: PrepareCLI,
    base_manifest: dict,
    inputs: dict,
    method: str,
) -> dict:
    directory = settings.output_dir / name.replace("_", "-")
    manifest_path = (
        settings.manifest_dir / f"{name.replace('_', '-')}-label-sources.json"
    )
    table = studies.select(STUDY).with_columns(
        [pl.Series(t, values[:, i]) for i, t in enumerate(targets)]
    )
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError(f"{name} targets must be finite probabilities")
    directory.mkdir(parents=True)
    path = directory / "labels.csv"
    table.sort(STUDY).write_csv(path)
    prepared = {**fingerprint(path), "label_table": name}
    manifest = {
        "prepared_at": datetime.now(UTC).isoformat(),
        "competition": COMPETITION,
        "method": method,
        "preparer": fingerprint(Path(__file__).resolve()),
        "inputs": inputs,
        "rows": table.height,
        "sources": [
            *base_manifest["sources"],
            {"dataset": f"local/{name}", "dataset_version": 1, "files": [prepared]},
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    silver = ~studies["is_gold"].to_numpy()
    old = base.select(targets).to_numpy()[silver]
    new = values[silver]
    return {
        "label_manifest": fingerprint(manifest_path),
        "prepared_labels": prepared,
        "silver_positive_rate": {
            t: [
                float((old[:, i] >= TARGET_THRESHOLD).mean()),
                float((new[:, i] >= TARGET_THRESHOLD).mean()),
            ]
            for i, t in enumerate(targets)
        },
        "silver_changed_calls": {
            t: int(
                (
                    (old[:, i] >= TARGET_THRESHOLD) != (new[:, i] >= TARGET_THRESHOLD)
                ).sum()
            )
            for i, t in enumerate(targets)
        },
        "silver_mean_absolute_change": {
            t: float(np.abs(old[:, i] - new[:, i]).mean())
            for i, t in enumerate(targets)
        },
    }


def prepare(settings: PrepareCLI) -> dict:
    outputs = (
        [settings.output_dir / n.replace("_", "-") for n in (STATE_TABLE, MEAN_TABLE)]
        + [
            settings.manifest_dir / f"{n.replace('_', '-')}-label-sources.json"
            for n in (STATE_TABLE, MEAN_TABLE)
        ]
        + [settings.analysis_dir]
    )
    if any(p.exists() for p in outputs):
        raise FileExistsError("Preparation requires fresh output paths")
    base_manifest = json.loads(settings.base_manifest.read_text())
    base, base_record = base_labels(base_manifest)
    targets = pl.read_csv("data/raw/sample_submission.csv").columns[1:]
    official = pl.read_csv("data/raw/train.csv", infer_schema_length=None).select(
        STUDY, *targets
    )
    split = pl.read_csv(settings.split_studies, infer_schema_length=None).select(
        STUDY, "report_group"
    )
    studies = (
        official.select(STUDY, pl.col("ACL").is_not_null().alias("is_gold"))
        .join(split, on=STUDY, how="left", validate="1:1")
        .sort(STUDY)
    )
    base = studies.select(STUDY).join(base, on=STUDY, how="left", validate="1:1")
    gold = studies["is_gold"].to_numpy()
    silver = ~gold
    truth = (
        studies.select(STUDY).join(official, on=STUDY).select(targets).to_numpy()[gold]
    )

    roots = load_roots(studies.select(STUDY), targets, settings.jevk5_attributes)
    groups, similarity = families(roots, targets, silver)
    representatives = [roots[min(g)] for g in groups]
    clean = [~gold | (r.gold_excluded is None) for r in representatives]

    def family_mean() -> np.ndarray:
        result = np.full((len(gold), len(targets)), np.nan)
        for i, t in enumerate(targets):
            stack = []
            for g in groups:
                members = [
                    np.where(
                        ~gold | (roots[m].gold_excluded is None),
                        scaled_values(roots[m], t),
                        np.nan,
                    )
                    for m in g
                ]
                stack.append(np.nanmean(np.stack(members), axis=0))
            result[:, i] = np.nanmean(np.stack(stack), axis=0)
        return result

    with np.errstate(all="ignore"):
        mean_clean = family_mean()
    base_values = base.select(targets).to_numpy()
    mean_clean = np.where(np.isfinite(mean_clean), mean_clean, base_values)

    tempering = {}
    fits = {}
    for tau in TEMPERATURES:
        score = 0.0
        fits[tau] = {}
        for t in targets:
            encoded = encode(representatives, t, clean)
            prior, thetas, _ = fit_states(encoded, tau)
            fits[tau][t] = (encoded, prior, thetas)
            score += held_out_likelihood(encoded, prior, thetas, tau, silver)
        tempering[str(tau)] = score / len(targets)
    tau = float(max(tempering, key=tempering.get))

    q_clean = np.zeros((len(targets), len(gold), len(STATES)))
    thetas_report = {}
    for i, t in enumerate(targets):
        encoded, prior, thetas = fits[tau][t]
        q_clean[i] = posterior(encoded, prior, thetas, tau)
        thetas_report[t] = {
            "state_prior": dict(zip(STATES, prior.tolist())),
            "roots": {
                representatives[r].name: {
                    s: dict(zip(STATES, np.round(theta[:, j], 4).tolist()))
                    for j, s in enumerate(encoded.alphabets[r])
                }
                for r, theta in enumerate(thetas)
            },
        }

    centre = np.array(
        [
            logit(np.clip(mean_clean[silver, i], Z_CLIP, 1 - Z_CLIP)).mean()
            for i in range(len(targets))
        ]
    )
    z_clean = centred_logit(mean_clean, centre)
    cell_target = np.repeat(np.arange(len(targets))[None], gold.sum(), axis=0)
    gold_q = q_clean[:, gold].transpose(1, 0, 2)
    gold_z = z_clean[gold]
    report_groups = studies["report_group"].to_numpy()[gold]
    oof = np.zeros((gold.sum(), len(targets)))
    for group in np.unique(report_groups):
        train, held = report_groups != group, report_groups == group
        params = fit_calibration(
            gold_q[train].reshape(-1, len(STATES)),
            cell_target[train].ravel(),
            gold_z[train].ravel(),
            truth[train].ravel(),
            len(targets),
        )
        oof[held] = calibrated(
            params,
            gold_q[held].reshape(-1, len(STATES)),
            cell_target[held].ravel(),
            gold_z[held].ravel(),
            len(targets),
        ).reshape(-1, len(targets))
    params = fit_calibration(
        gold_q.reshape(-1, len(STATES)),
        cell_target.ravel(),
        gold_z.ravel(),
        truth.ravel(),
        len(targets),
    )
    rubric = gold_q[..., POS] + 0.5 * gold_q[..., HEDGED]
    evaluation = gold_metrics(
        {
            "combined_v1": base_values[gold],
            "A_state_calibrated_oof": oof,
            "B_family_mean": mean_clean[gold],
            "A0_state_rubric": rubric,
        },
        truth,
        targets,
    )

    all_target = np.repeat(np.arange(len(targets))[None], len(gold), axis=0)
    state_values = calibrated(
        params,
        q_clean.transpose(1, 0, 2).reshape(-1, len(STATES)),
        all_target.ravel(),
        z_clean.ravel(),
        len(targets),
    ).reshape(len(gold), len(targets))
    state_values = np.where(gold[:, None], base_values, state_values)
    mean_values = np.where(gold[:, None], base_values, mean_clean)

    beta, alpha, gamma = params[: len(STATES)], params[len(STATES) : -1], params[-1]
    calibration = {
        "state_effects": dict(zip(STATES, beta.tolist())),
        "target_offsets": dict(zip(targets, alpha.tolist())),
        "slope": float(gamma),
    }
    argmax_states = q_clean.argmax(axis=2)
    calibration["map_at_state_median_z"] = {
        t: {
            s: round(
                float(
                    expit(
                        beta[k]
                        + alpha[i]
                        + gamma
                        * np.median(z_clean[silver & (argmax_states[i] == k), i])
                    )
                ),
                3,
            )
            for k, s in enumerate(STATES)
            if (silver & (argmax_states[i] == k)).sum() >= 20
        }
        for i, t in enumerate(targets)
    }
    gold_states = argmax_states[:, gold].T
    calibration["gold_positive_rate_by_state"] = {
        s: [int((gold_states == k).sum()), float(truth[gold_states == k].mean())]
        for k, s in enumerate(STATES)
    }
    distribution = {
        t: {s: int((argmax_states[i, silver] == k).sum()) for k, s in enumerate(STATES)}
        for i, t in enumerate(targets)
    }
    inputs = {
        "combined_v1": base_record,
        "base_manifest": fingerprint(settings.base_manifest),
        "split_studies": fingerprint(settings.split_studies),
        "official": fingerprint(Path("data/raw/train.csv")),
        "roots": {r.name: [fingerprint(p) for p in r.paths] for r in roots},
    }
    common = (
        "Silver rows only; gold rows keep combined_v1 values and the loader applies official "
        "gold overrides. Roots are lineage-level public report extractions plus JevK5 v3 "
        f"attributes, merged into families at mean silver kappa >= {FAMILY_KAPPA}, with "
        f"derived roots {DERIVED} merged into their parent's family. Before averaging, "
        "not-mentioned, below-threshold and hedged codes (anchor maps and silent symbols) "
        f"take the pooled gold positive rate of their state from rebuild_state_v1 {STATE_RATE}; "
        "negative and positive codes keep each source's own value."
    )
    state_method = (
        "Anchored Dawid-Skene over five latent report states per target (NEG, NOT_MENTIONED, "
        "LOW, HEDGED, POS) on each family representative's native symbols, fitted by EM on "
        f"all reports without labels, tempering tau={tau} chosen by held-out-source "
        "likelihood; P(y) = sum_k q_k sigmoid(beta_k + alpha_t + gamma z) fitted by MAP on "
        "gold with gold-tuned and gold-copying roots removed on gold rows. " + common
    )
    mean_method = (
        "Per target mean of each root family's probability on the common state scale. "
        + common
    )
    settings.analysis_dir.mkdir(parents=True)
    tables = {
        STATE_TABLE: write_table(
            STATE_TABLE,
            state_values,
            studies,
            base,
            targets,
            settings,
            base_manifest,
            inputs,
            state_method,
        ),
        MEAN_TABLE: write_table(
            MEAN_TABLE,
            mean_values,
            studies,
            base,
            targets,
            settings,
            base_manifest,
            inputs,
            mean_method,
        ),
    }
    report = {
        "roots": [
            {
                "name": r.name,
                "gold_excluded": r.gold_excluded,
                "anchored": bool(r.anchors),
            }
            for r in roots
        ],
        "families": [[roots[m].name for m in g] for g in groups],
        "family_representatives": [r.name for r in representatives],
        "mean_silver_kappa": {
            roots[a].name: {
                roots[b].name: round(float(similarity[a, b]), 3)
                for b in range(len(roots))
                if b != a
            }
            for a in range(len(roots))
        },
        "tempering_held_out_log_likelihood": tempering,
        "tau": tau,
        "silver_state_argmax": distribution,
        "calibration": calibration,
        "gold_evaluation": evaluation,
        "tables": tables,
    }
    (settings.analysis_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (settings.analysis_dir / "state_model.json").write_text(
        json.dumps(thetas_report, indent=2) + "\n"
    )
    return report


def main() -> None:
    report = prepare(PrepareCLI())
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "families",
                    "tau",
                    "tempering_held_out_log_likelihood",
                    "gold_evaluation",
                    "calibration",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
