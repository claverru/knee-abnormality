import ast
import json
import zipfile
from glob import glob
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, Field
from scipy import stats
from threadpoolctl import threadpool_limits

from knee_abnormality.cli import Bootstrap, CLISettings, describe
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD

TYPES = (
    "axial_t1pd",
    "axial_fluid",
    "coronal_t1pd",
    "sagittal_t1pd",
    "coronal_fluid",
    "sagittal_fluid",
)
KINDS = {
    "axial_t1pd": ("Axial", 0),
    "axial_fluid": ("Axial", 1),
    "coronal_t1pd": ("Coronal", 0),
    "sagittal_t1pd": ("Sagittal", 0),
    "coronal_fluid": ("Coronal", 1),
    "sagittal_fluid": ("Sagittal", 1),
}
SLICE_SHARE = {
    "axial_t1pd": 0.046,
    "axial_fluid": 0.243,
    "coronal_t1pd": 0.150,
    "sagittal_t1pd": 0.187,
    "coronal_fluid": 0.189,
    "sagittal_fluid": 0.185,
}
BASE_WIDTH = 120.0
WIDE_STEP = 140.0
FLUID = 280 / 3
CANDIDATE_WIDTHS = (84.0, FLUID, 100.0, 112.0, 120.0, 130.0, 140.0, 160.0)
STRENGTHS = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
OLD = "shared-readout"
NEW = "plane-readout"
REFERENCE = "rn18-l2sp-no-confusion"
CONTROL = "rn18-field-120-control"
WIDE_CONTROL = "no-anatomical-bias"
T84 = "t1pd-detail-84"
INCUMBENT = "fluid-detail-93-replacement"
CORONAL_93 = "coronal-fluid-93-sagittal-fluid-120"
SAGITTAL_93 = "sagittal-fluid-93-coronal-fluid-120"
MIXED = "outputs/autoresearch/mixed-field/experiments"
MANUAL = "outputs/autoresearch/mixed-field/manual"
SMART = "outputs/autoresearch/smart-crops/experiments"
REVISITS = "outputs/autoresearch/revisits/experiments"
ALL_120 = (120,) * 6
ALL_140 = (140,) * 6
ALL_160 = (160,) * 6
SPLIT = (112, 140, 112, 112, 140, 140)
DETAIL = (120, 120, 84, 84, FLUID, FLUID)
FORWARD = "sagittal_fluid_no_forward_shift"
VERTICAL = "sagittal_no_vertical_shift"
MIDDLE = "also middle-slice augmentation"
DEAD_STREAM = "its second, 140 mm stream never trained; in effect 120 mm"


class Run(BaseModel):
    name: str
    family: Literal["shared-readout", "plane-readout"]
    reference: str | None
    widths: tuple[float, float, float, float, float, float]
    root: str
    rule: bool = False
    placement: tuple[str, ...] = ()
    readable: bool = True
    note: str = ""


class Group(BaseModel):
    name: str
    members: tuple[str, ...]
    split: bool = False
    narrow: bool = True


class Term(BaseModel):
    name: str
    members: tuple[str, ...]
    side: Literal["narrow", "wide"]
    step: float
    description: str


def detail(**changed):
    return tuple({**dict(zip(TYPES, DETAIL, strict=True)), **changed}.values())


def old(name, widths, base=MIXED, **extra):
    return Run(
        name=name,
        family=OLD,
        reference=REFERENCE,
        widths=widths,
        root=f"{base}/{name}-2026*/training",
        **extra,
    )


def smart(name, reference, widths, **extra):
    return Run(
        name=name,
        family=NEW,
        reference=reference,
        widths=widths,
        root=f"{SMART}/{name}-2026*/training",
        rule=True,
        **extra,
    )


RUNS = (
    Run(
        name=REFERENCE,
        family=OLD,
        reference=None,
        widths=ALL_140,
        root=f"outputs/training/{REFERENCE}",
    ),
    old("field-160-single-logged", ALL_160),
    old("field-120-single", ALL_120),
    old("field-120-middle-slice-augmentation", ALL_120, MANUAL, note=MIDDLE),
    old("two-stream-paired", ALL_120, readable=False, note=DEAD_STREAM),
    old("contrast-split-112-140", SPLIT),
    old("middle-slice-augmentation", SPLIT, MANUAL, note=MIDDLE),
    Run(
        name=WIDE_CONTROL,
        family=NEW,
        reference=None,
        widths=ALL_140,
        root=f"{REVISITS}/{WIDE_CONTROL}-2026*/training",
        note="trained before middle-slice augmentation",
    ),
    Run(
        name=CONTROL,
        family=NEW,
        reference=WIDE_CONTROL,
        widths=ALL_120,
        root=f"outputs/training/{CONTROL}",
    ),
    smart(T84, CONTROL, detail(coronal_fluid=120, sagittal_fluid=120)),
    smart(INCUMBENT, T84, DETAIL),
    smart(CORONAL_93, INCUMBENT, detail(sagittal_fluid=120)),
    smart(SAGITTAL_93, INCUMBENT, detail(coronal_fluid=120)),
    smart("axial-130-detail-84-93", INCUMBENT, detail(axial_t1pd=130, axial_fluid=130)),
    smart("axial-140-detail-84-93", INCUMBENT, detail(axial_t1pd=140, axial_fluid=140)),
    smart("sagittal-fluid-93-centred", INCUMBENT, DETAIL, placement=(FORWARD,)),
    smart("sagittal-fluid-100", INCUMBENT, detail(sagittal_fluid=100)),
    smart("t1pd-detail-93", INCUMBENT, detail(coronal_t1pd=FLUID, sagittal_t1pd=FLUID)),
    smart("coronal-t1pd-84-sagittal-t1pd-120", INCUMBENT, detail(sagittal_t1pd=120)),
    smart("sagittal-t1pd-84-coronal-t1pd-120", INCUMBENT, detail(coronal_t1pd=120)),
    smart("sagittal-no-vertical-shift", INCUMBENT, DETAIL, placement=(VERTICAL,)),
)
EXCLUDED = (
    (
        OLD,
        "the width changes between slices of one type",
        (
            "field-jitter",
            "mixed-field-140-160",
            "mixed-field-120-160",
            "mixed-field-readout-weights",
            "field-backbones-routed",
            "plane-weighted-fields",
        ),
    ),
    (OLD, "140 or 160 mm chosen per series", ("extent-ruled-field",)),
    (
        NEW,
        "T1/PD width chosen per series, 84 to 120 mm",
        ("t1pd-detail-acquisition-matched", "t1pd-detail-acquired-matrix"),
    ),
    (
        NEW,
        "non-uniform window: 120 mm with a fine 80 mm core",
        ("t1pd-foveated-native-view", "fluid-foveated-native-view"),
    ),
    (NEW, "two T1/PD views, 120 and 84 mm, fused", ("t1pd-shallow-detail-fusion",)),
    (
        NEW,
        "same widths as t1pd-detail-84 with a denser token grid, not a width change",
        ("t1pd-detail-dense-final-grid",),
    ),
    (
        NEW,
        "a 60 mm T1/PD crop added beside or taught from the 120 mm view",
        (
            "crop-rule",
            "crop-attention",
            "crop-compartments",
            "crop-own-loss",
            "crop-evidence-utility-placement",
            "crop-rule-overlap-replacement",
            "crop-context-teacher-distillation",
            "crop-full-context-ranking-locator",
        ),
    ),
    (
        NEW,
        "a second 140 mm view of the fluid slices through another backbone",
        ("two-views-fluid-140", "two-views-112-fluid-140"),
    ),
    (
        NEW,
        "a 93 x 67 mm window added beside the 120 mm sagittal fluid view",
        (
            "sagittal-fluid-detail-rectangle",
            "sagittal-fluid-correspondence-fusion",
            "sagittal-fluid-content-residual",
        ),
    ),
    (
        NEW,
        "two widths or crops on one type, routed by label (queued)",
        ("sagittal-fluid-wide-for-bakers", "oa-compartment-readers"),
    ),
    (
        NEW,
        "changes normalisation, not a width (queued)",
        ("spacing-specific-batch-statistics",),
    ),
)
GROUPS = (
    Group(name="axial", members=("axial_t1pd", "axial_fluid"), narrow=False),
    Group(
        name="coronal_sagittal_t1pd",
        members=("coronal_t1pd", "sagittal_t1pd"),
        split=True,
    ),
    Group(name="coronal_fluid", members=("coronal_fluid",)),
    Group(name="sagittal_fluid", members=("sagittal_fluid",)),
)
WIDE_TERMS = (
    ("axial_wide", ("axial_t1pd", "axial_fluid")),
    ("coronal_sagittal_t1pd_wide", ("coronal_t1pd", "sagittal_t1pd")),
    ("coronal_sagittal_fluid_wide", ("coronal_fluid", "sagittal_fluid")),
)
THREE_120 = (
    "field-120-single",
    "field-120-middle-slice-augmentation",
    "two-stream-paired",
)
AXIAL_130 = np.log2(130 / 120) / np.log2(140 / 120)
WIDE_160 = np.log2(160 / 140) / np.log2(140 / 120)
SAGITTAL_100 = np.log2(120 / 100) / np.log2(120 / FLUID)
T1PD_93 = np.log2(120 / FLUID) / np.log2(120 / 84)
CLOSURES = (
    (
        "fluid_additivity",
        (
            "both fluid planes at 93 mm, minus coronal only, minus sagittal only, plus "
            "neither; zero if the two planes add up"
        ),
        {INCUMBENT: 1, CORONAL_93: -1, SAGITTAL_93: -1, T84: 1},
    ),
    (
        "every_type_140_to_120_between_families",
        (
            "140 to 120 mm on every type in the plane-readout recipe (one pair) minus the "
            "same step in the shared-readout recipe (mean of three 120 mm runs)"
        ),
        {
            CONTROL: 1,
            WIDE_CONTROL: -1,
            REFERENCE: 1,
            **dict.fromkeys(THREE_120, -1 / 3),
        },
    ),
    (
        "axial_130_on_the_120_140_line",
        "axial at 130 mm minus the log-linear point between axial 120 and 140 mm",
        {
            "axial-130-detail-84-93": 1,
            INCUMBENT: -(1 - AXIAL_130),
            "axial-140-detail-84-93": -AXIAL_130,
        },
    ),
    (
        "every_type_160_on_the_120_140_line",
        (
            "every type at 160 mm minus the log-linear continuation of the 120 to 140 mm "
            "step, shared-readout recipe"
        ),
        {
            "field-160-single-logged": 1,
            REFERENCE: -(1 + WIDE_160),
            **dict.fromkeys(THREE_120, WIDE_160 / 3),
        },
    ),
    (
        "sagittal_fluid_100_on_the_93_120_line",
        (
            "sagittal fluid at 100 mm minus the log-linear point between 93 and 120 mm, "
            "coronal fluid at 93 mm"
        ),
        {
            "sagittal-fluid-100": 1,
            CORONAL_93: -(1 - SAGITTAL_100),
            INCUMBENT: -SAGITTAL_100,
        },
    ),
    (
        "t1pd_93_on_the_84_120_line",
        (
            "T1/PD at 93 mm minus the log-linear point between 84 and 120 mm; the 120 to "
            "84 mm step is taken at fluid 120 mm, the 93 mm run at fluid 93 mm"
        ),
        {"t1pd-detail-93": 1, INCUMBENT: -1, T84: 1 - T1PD_93, CONTROL: -(1 - T1PD_93)},
    ),
    (
        "t1pd_plane_additivity",
        (
            "coronal T1/PD zoom alone plus sagittal T1/PD zoom alone (fluid at 93 mm) "
            "minus both together (fluid at 120 mm)"
        ),
        {
            INCUMBENT: 2,
            "coronal-t1pd-84-sagittal-t1pd-120": -1,
            "sagittal-t1pd-84-coronal-t1pd-120": -1,
            T84: -1,
            CONTROL: 1,
        },
    ),
)


def pair(effect, path, after, before, scale=1.0):
    return effect, path, {after: scale, before: -scale}


PATHS = (
    pair("sagittal fluid 120 to 93 mm", "coronal fluid at 120 mm", SAGITTAL_93, T84),
    pair(
        "sagittal fluid 120 to 93 mm", "coronal fluid at 93 mm", INCUMBENT, CORONAL_93
    ),
    pair("coronal fluid 120 to 93 mm", "sagittal fluid at 120 mm", CORONAL_93, T84),
    pair(
        "coronal fluid 120 to 93 mm", "sagittal fluid at 93 mm", INCUMBENT, SAGITTAL_93
    ),
    pair("coronal + sagittal T1/PD 120 to 84 mm", "fluid at 120 mm", T84, CONTROL),
    pair("axial 120 to 140 mm", "140 mm run", "axial-140-detail-84-93", INCUMBENT),
    pair(
        "axial 120 to 140 mm",
        "130 mm run scaled to the 140 mm step",
        "axial-130-detail-84-93",
        INCUMBENT,
        1 / AXIAL_130,
    ),
    *(
        pair("every type 120 to 140 mm", f"shared-readout, {name}", REFERENCE, name)
        for name in THREE_120
    ),
    pair("every type 120 to 140 mm", "plane-readout", WIDE_CONTROL, CONTROL),
    pair(
        "every type 140 to 160 mm",
        "shared-readout",
        "field-160-single-logged",
        REFERENCE,
    ),
    *(
        pair(
            "T1/PD 140 to 112 mm on all planes, fluid at 140 mm", name, name, REFERENCE
        )
        for name in ("contrast-split-112-140", "middle-slice-augmentation")
    ),
    pair(
        "sagittal fluid forward shift removed",
        "incumbent widths",
        "sagittal-fluid-93-centred",
        INCUMBENT,
    ),
    pair(
        "sagittal fluid 93 to 100 mm",
        "incumbent widths",
        "sagittal-fluid-100",
        INCUMBENT,
    ),
    pair(
        "coronal + sagittal T1/PD 84 to 93 mm",
        "incumbent widths",
        "t1pd-detail-93",
        INCUMBENT,
    ),
    pair(
        "sagittal T1/PD 84 to 120 mm",
        "incumbent widths",
        "coronal-t1pd-84-sagittal-t1pd-120",
        INCUMBENT,
    ),
    pair(
        "coronal T1/PD 84 to 120 mm",
        "incumbent widths",
        "sagittal-t1pd-84-coronal-t1pd-120",
        INCUMBENT,
    ),
    pair(
        "sagittal vertical shift removed",
        "incumbent widths",
        "sagittal-no-vertical-shift",
        INCUMBENT,
    ),
)


class Settings(CLISettings, Bootstrap):
    model_config = describe(
        "Fit one ridge regression per label, and for the macro, of fold-0 AUC on the view "
        "width of each series type across every finished run with one width per type, "
        "with closure checks, leave-one-run-out error, validation-sample and "
        "training-noise intervals, the preferred widths and an identifiability table"
    )

    split_dir: Path = Field(
        default=Path("data/folds/all_5fold_scanner_v2"),
        description="Frozen split with targets.csv and studies.csv",
    )
    output_dir: Path = Field(
        default=Path("outputs/analysis/v/view-regression-20261002"),
        description="Output directory",
    )
    threads: int = Field(default=6, ge=1, le=6, description="Numerical threads")


def resolve(run: Run) -> Path | None:
    matches = sorted(glob(run.root))
    return Path(matches[-1]) if matches else None


def finished(root: Path | None) -> bool:
    return root is not None and all(
        (root / "fold_0" / name).exists()
        for name in ("validation_predictions.csv", "selected_checkpoint.json")
    )


def namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: namespace(item) for key, item in value.items()})
    return value


def evaluate(node, scope):
    return eval(
        compile(ast.Expression(node), "frozen", "eval"), {"__builtins__": {}}, scope
    )


class Scope(dict):
    def __init__(self, assigned, base):
        super().__init__(base)
        self.assigned = assigned

    def __missing__(self, name):
        if name not in self.assigned:
            raise KeyError(name)
        self[name] = evaluate(self.assigned[name], self)
        return self[name]


def name_assignments(body):
    nodes = [
        node
        for node in body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ]
    return {
        node.targets[0].id: node.value
        for node in sorted(nodes, key=lambda node: node.lineno)
    }


def code_views(root: Path) -> tuple[dict[str, tuple[float, bool]], dict]:
    data = json.loads((root / "effective_config.json").read_text())["data"]
    with zipfile.ZipFile(root / "source_snapshot.zip") as archive:
        dataset = ast.parse(archive.read("src/knee_abnormality/data/dataset.py"))
        spatial = ast.parse(
            archive.read("src/knee_abnormality/data/preprocessing/spatial.py")
        )
    constants = {}
    for name, value in name_assignments(spatial.body).items():
        if name.isupper():
            try:
                constants[name] = evaluate(value, dict(constants))
            except (NameError, TypeError, KeyError, AttributeError):
                continue
    assigned = {}
    for function in ast.walk(dataset):
        if isinstance(function, ast.FunctionDef):
            names = name_assignments(ast.walk(function))
            if "field_mm" in names or (
                "spacing" in names and not isinstance(names["spacing"], ast.Constant)
            ):
                assigned = names
                break
    views = {}
    for kind, (plane, fluid) in KINDS.items():
        scope = Scope(
            assigned,
            {
                **constants,
                "self": SimpleNamespace(data=namespace(data)),
                "plane": plane,
                "kind": SimpleNamespace(fluid_sensitive=fluid, fat_suppression=fluid),
            },
        )
        if "field_mm" in assigned:
            width = float(scope["field_mm"])
        elif "spacing" in assigned:
            width = float(scope["spacing"]) * data["image_size"]
        else:
            width = data["spacing_mm"] * data["image_size"]
        views[kind] = (width, bool(scope["detail"]) if "detail" in assigned else False)
    placement = {
        name: value
        for name, value in constants.items()
        if any(word in name for word in ("OFFSET", "ANTERIOR", "VERTICAL", "SHIFT"))
    }
    return views, placement


def check_views(run: Run, root: Path | None) -> dict:
    row = {"run": run.name, "status": "no source yet", "placement_constants": ""}
    declared = {
        kind: (width, run.rule and width < BASE_WIDTH)
        for kind, width in zip(TYPES, run.widths, strict=True)
    }
    row |= {f"declared|{kind}": width for kind, (width, _) in declared.items()}
    if root is None or not (root / "source_snapshot.zip").exists():
        return row
    if not run.readable:
        return row | {"status": f"declared only: {run.note}"}
    views, placement = code_views(root)
    row |= {f"code|{kind}": width for kind, (width, _) in views.items()}
    row |= {f"rule_placed|{kind}": placed for kind, (_, placed) in views.items()}
    wrong = [
        kind
        for kind in TYPES
        if abs(views[kind][0] - declared[kind][0]) > 0.01
        or views[kind][1] != declared[kind][1]
    ]
    if wrong:
        raise ValueError(f"{run.name}: declared views differ from its code on {wrong}")
    return row | {
        "status": "matches the frozen code",
        "placement_constants": json.dumps(placement, default=str),
    }


def label_aucs(truth: np.ndarray, scores: np.ndarray, counts: np.ndarray) -> np.ndarray:
    result = np.empty((len(counts), truth.shape[1] + 1))
    for target in range(truth.shape[1]):
        positive = truth[:, target] == 1
        difference = scores[positive, target, None] - scores[~positive, target]
        wins = (difference > 0) + 0.5 * (difference == 0)
        pos, neg = counts[:, positive], counts[:, ~positive]
        result[:, target] = (pos * (neg @ wins.T)).sum(1) / (pos.sum(1) * neg.sum(1))
    result[:, -1] = result[:, :-1].mean(1)
    return result


def load_scores(settings: Settings, used: list[Run], roots: dict) -> dict:
    targets_frame = pl.read_csv(settings.split_dir / "targets.csv")
    targets = [column for column in targets_frame.columns if column != STUDY]
    first = pl.read_csv(roots[CONTROL] / "fold_0" / "validation_predictions.csv")
    studies = sorted(first[STUDY].to_list())
    truth_frame = (
        pl.read_csv(settings.split_dir / "studies.csv")
        .select(STUDY, "report_group")
        .join(targets_frame, on=STUDY, validate="1:1")
        .filter(pl.col(STUDY).is_in(studies))
        .sort(STUDY)
    )
    truth = (truth_frame.select(targets).to_numpy() >= TARGET_THRESHOLD).astype(
        np.int64
    )
    unique, inverse = np.unique(
        truth_frame["report_group"].to_numpy(), return_inverse=True
    )
    rng = np.random.default_rng(settings.bootstrap_seed)
    resamples = rng.multinomial(
        len(unique),
        np.full(len(unique), 1 / len(unique)),
        size=settings.bootstrap_draws,
    )[:, inverse]
    counts = np.vstack([np.ones((1, len(studies))), resamples.astype(np.float64)])
    point = {1: [], 4: []}
    resampled = {1: [], 4: []}
    recorded = []
    for item in used:
        fold = roots[item.name] / "fold_0"
        selected = json.loads((fold / "selected_checkpoint.json").read_text())
        epoch = int(selected["path"].split("best-epoch=")[1][:2])
        files = {
            4: fold / "validation_predictions.csv",
            1: fold / "epochs" / f"validation_epoch_{epoch:03d}.csv",
        }
        for multiplier, path in files.items():
            predictions = pl.read_csv(path).sort(STUDY)
            if predictions[STUDY].to_list() != studies:
                raise ValueError(
                    f"{item.name}: validation studies differ at x{multiplier}"
                )
            values = label_aucs(
                truth, predictions.select(targets).to_numpy().astype(np.float64), counts
            )
            point[multiplier].append(values[0])
            resampled[multiplier].append(values[1:])
        logged = json.loads(
            (fold / "epochs" / f"metrics_epoch_{epoch:03d}.json").read_text()
        )["per_target_auc"]
        recorded.append(
            max(
                abs(point[4][-1][-1] - selected["prediction_macro_auc"]),
                *(
                    abs(point[1][-1][column] - logged[target])
                    for column, target in enumerate(targets)
                ),
            )
        )
    return {
        "labels": [*targets, "macro"],
        "scores": {key: np.array(values) for key, values in point.items()},
        "draws": {key: np.stack(values, axis=1) for key, values in resampled.items()},
        "recorded": float(max(recorded)),
        "studies": len(studies),
        "report_groups": len(unique),
        "positives": dict(zip(targets, map(int, truth.sum(0)), strict=True)),
    }


def part(width: float, side: str) -> float:
    value = np.log2(BASE_WIDTH / width)
    return max(value, 0.0) if side == "narrow" else max(-value, 0.0)


def term_value(term: Term, widths) -> float:
    by_kind = dict(zip(TYPES, widths, strict=True))
    weight = sum(SLICE_SHARE[kind] for kind in term.members)
    return sum(
        SLICE_SHARE[kind] * part(by_kind[kind], term.side) for kind in term.members
    ) / (weight * part(term.step, term.side))


def build_terms(runs: list[Run]) -> tuple[list[Group], list[Term]]:
    groups = []
    for group in GROUPS:
        indices = [TYPES.index(kind) for kind in group.members]
        differ = any(len({run.widths[index] for index in indices}) > 1 for run in runs)
        if group.split and differ:
            groups += [Group(name=kind, members=(kind,)) for kind in group.members]
        else:
            groups.append(group)
    terms = []
    for group in groups:
        indices = [TYPES.index(kind) for kind in group.members]
        narrowest = min(run.widths[index] for run in runs for index in indices)
        if group.narrow and narrowest < BASE_WIDTH:
            terms.append(
                Term(
                    name=f"{group.name}_narrow",
                    members=group.members,
                    side="narrow",
                    step=narrowest,
                    description=f"{' + '.join(group.members)} "
                    f"{BASE_WIDTH:.0f} to {narrowest:.1f} mm",
                )
            )
    for name, members in WIDE_TERMS:
        terms.append(
            Term(
                name=name,
                members=members,
                side="wide",
                step=WIDE_STEP,
                description=f"{' + '.join(members)} "
                f"{BASE_WIDTH:.0f} to {WIDE_STEP:.0f} mm",
            )
        )
    return groups, terms


def design_row(widths, family, placement, terms, flags) -> list[float]:
    return [
        *(float(family == name) for name in (OLD, NEW)),
        *(term_value(term, widths) for term in terms),
        *(float(flag in placement) for flag in flags),
    ]


def solver(design: np.ndarray, penalised: np.ndarray, strength: float) -> np.ndarray:
    augmented = np.vstack([design, np.sqrt(strength) * np.diag(penalised)])
    return np.linalg.pinv(augmented)[:, : len(design)]


def leave_one_out(design, penalised, strength, scores) -> np.ndarray:
    predictions = np.empty_like(scores)
    for index in range(len(design)):
        keep = np.arange(len(design)) != index
        predictions[index] = (
            design[index] @ solver(design[keep], penalised, strength) @ scores[keep]
        )
    return predictions


def strength_path(design, penalised, scores, labels) -> list[dict]:
    rows = []
    for strength in STRENGTHS:
        error = (leave_one_out(design, penalised, strength, scores) - scores) ** 2
        rows.append(
            {
                "strength": strength,
                "pooled_label_rmse": float(np.sqrt(error[:, :-1].mean())),
                **{
                    f"rmse|{label}": float(np.sqrt(error[:, column].mean()))
                    for column, label in enumerate(labels)
                },
            }
        )
    return rows


def replicate_noise(design: np.ndarray, scores: np.ndarray):
    keys = [tuple(np.round(row, 9)) for row in design]
    squares = np.zeros(scores.shape[1])
    widest = np.zeros(scores.shape[1])
    freedom = 0
    groups = []
    for key in dict.fromkeys(keys):
        members = [index for index, other in enumerate(keys) if other == key]
        if len(members) > 1:
            block = scores[members]
            squares += ((block - block.mean(0)) ** 2).sum(0)
            widest = np.maximum(widest, block.max(0) - block.min(0))
            freedom += len(members) - 1
            groups.append(members)
    variance = squares / freedom
    shrunk = variance.copy()
    shrunk[:-1] = (variance[:-1] + variance[:-1].mean()) / 2
    degrees = np.full(scores.shape[1], 2.0 * freedom)
    degrees[-1] = freedom
    return np.sqrt(variance), np.sqrt(shrunk), degrees, widest, groups


def interval(estimate, validation, training, degrees) -> dict:
    row = {
        "estimate": estimate,
        "se_validation": validation,
        "se_training": training,
        "se_both": float(np.hypot(validation, training)),
    }
    for level in (80, 95):
        tail = 0.5 + level / 200
        half = float(
            np.hypot(
                stats.norm.ppf(tail) * validation,
                stats.t.ppf(tail, degrees) * training,
            )
        )
        row |= {
            f"low_{level}": estimate - half,
            f"high_{level}": estimate + half,
            f"distinguishable_{level}": bool(abs(estimate) > half),
        }
    row["distinguishable_95_validation_only"] = bool(
        abs(estimate) > stats.norm.ppf(0.975) * validation
    )
    return row


def contrast_rows(specs, names, labels, scores, draws, sigma, degrees) -> list[dict]:
    rows = []
    index = {name: position for position, name in enumerate(names)}
    for group, description, weights in specs:
        head = {"check": group, "description": description, "runs": ", ".join(weights)}
        if not all(name in index for name in weights):
            rows.append(head | {"label": "pending"})
            continue
        vector = np.zeros(len(names))
        for name, weight in weights.items():
            vector[index[name]] += weight
        value = vector @ scores
        spread = np.einsum("dnl,n->dl", draws, vector).std(0)
        for position, label in enumerate(labels):
            rows.append(
                head
                | {"label": label}
                | interval(
                    float(value[position]),
                    float(spread[position]),
                    float(sigma[position] * np.linalg.norm(vector)),
                    degrees[position],
                )
            )
    return rows


def frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)


def analyse(
    scores, draws, design, penalised, strength, names, labels, features, effects
):
    hat = solver(design, penalised, strength)
    plain = solver(design, penalised, 0.0) @ scores
    estimate = hat @ scores
    estimate_draws = np.einsum("pn,dnl->dpl", hat, draws)
    sigma_raw, sigma, degrees, widest, replicates = replicate_noise(design, scores)
    leverage = np.sqrt((hat**2).sum(1))
    coefficients = [
        {"feature": feature, "effect": effects.get(feature, feature), "label": label}
        | interval(
            float(estimate[row, column]),
            float(estimate_draws[:, row, column].std()),
            float(sigma[column] * leverage[row]),
            degrees[column],
        )
        | {"unpenalised_estimate": float(plain[row, column])}
        for row, feature in enumerate(features)
        for column, label in enumerate(labels)
    ]
    fitted = design @ estimate
    effective = len(design) - np.trace(design @ hat)
    noise = [
        {
            "label": label,
            "replicate_sd": float(sigma_raw[column]),
            "replicate_sd_shrunk": float(sigma[column]),
            "degrees_of_freedom": float(degrees[column]),
            "largest_replicate_gap": float(widest[column]),
            "fit_residual_sd": float(
                np.sqrt(((scores - fitted)[:, column] ** 2).sum() / effective)
            ),
            "fit_residual_degrees": float(effective),
        }
        for column, label in enumerate(labels)
    ]
    return {
        "hat": hat,
        "estimate": estimate,
        "estimate_draws": estimate_draws,
        "sigma": sigma,
        "degrees": degrees,
        "fitted": fitted,
        "held_out": leave_one_out(design, penalised, strength, scores),
        "coefficients": coefficients,
        "noise": noise,
        "replicates": [[names[index] for index in group] for group in replicates],
    }


def linear(result, vector, column) -> dict:
    return interval(
        float(vector @ result["estimate"][:, column]),
        float((result["estimate_draws"][:, :, column] @ vector).std()),
        float(result["sigma"][column] * np.linalg.norm(vector @ result["hat"])),
        result["degrees"][column],
    )


def held_out_tables(used, labels, scores, result) -> tuple[list[dict], list[dict]]:
    position = {item.name: index for index, item in enumerate(used)}
    referenced = [
        index for index, item in enumerate(used) if item.reference in position
    ]
    reference = np.full_like(scores, np.nan)
    family = np.empty_like(scores)
    for index, item in enumerate(used):
        if item.reference in position:
            reference[index] = scores[position[item.reference]]
        others = [
            other
            for other, candidate in enumerate(used)
            if candidate.family == item.family and other != index
        ]
        family[index] = scores[others].mean(0)
    rows = [
        {
            "run": item.name,
            "label": label,
            "measured": float(scores[index, column]),
            "fitted": float(result["fitted"][index, column]),
            "left_out_prediction": float(result["held_out"][index, column]),
            "reference_prediction": float(reference[index, column]),
            "family_mean_prediction": float(family[index, column]),
        }
        for index, item in enumerate(used)
        for column, label in enumerate(labels)
    ]
    every = list(range(len(used)))
    errors = {
        "model_all_runs": (result["held_out"] - scores)[every],
        "model_runs_with_reference": (result["held_out"] - scores)[referenced],
        "reference_score": (reference - scores)[referenced],
        "family_mean_all_runs": (family - scores)[every],
        "family_mean_runs_with_reference": (family - scores)[referenced],
    }
    head = {"runs": len(every), "runs_with_reference": len(referenced)}
    summary = [
        {"label": label}
        | head
        | {
            f"rmse|{name}": float(np.sqrt((error[:, column] ** 2).mean()))
            for name, error in errors.items()
        }
        for column, label in enumerate(labels)
    ]
    summary.append(
        {"label": "pooled over the 12 labels"}
        | head
        | {
            f"rmse|{name}": float(np.sqrt((error[:, :-1] ** 2).mean()))
            for name, error in errors.items()
        }
    )
    return rows, summary


def isolating_pairs(used, groups, kind, width, family=None) -> list[tuple[str, str]]:
    index = TYPES.index(kind)
    group = next(group for group in groups if kind in group.members)
    allowed = {TYPES.index(member) for member in group.members}
    pairs = set()
    for one in used:
        if one.widths[index] != width or (family and one.family != family):
            continue
        for other in used:
            moved = {
                place
                for place in range(len(TYPES))
                if one.widths[place] != other.widths[place]
            }
            if (
                other.family == one.family
                and other.placement == one.placement
                and index in moved
                and moved <= allowed
            ):
                pairs.add((one.name, other.name))
    return sorted(pairs)


def identifiability_rows(used, groups, terms) -> list[dict]:
    rows = []
    for index, kind in enumerate(TYPES):
        tested = sorted({item.widths[index] for item in used})
        for width in CANDIDATE_WIDTHS:
            at = [item for item in used if abs(item.widths[index] - width) < 0.01]
            pairs = (
                isolating_pairs(used, groups, kind, at[0].widths[index]) if at else []
            )
            side = "narrow" if width < BASE_WIDTH else "wide"
            modelled = [
                term.name
                for term in terms
                if kind in term.members and term.side == side
            ]
            placed = sorted(
                {
                    "rule-placed"
                    if item.rule and width < BASE_WIDTH
                    else "image-centred"
                    for item in at
                }
            )
            if not at:
                status = (
                    "untested, inside the tested range: interpolation"
                    if tested[0] < width < tested[-1]
                    else "untested, outside the tested range: extrapolation"
                )
            elif width == BASE_WIDTH:
                status = "reference width"
            elif pairs:
                status = "isolated: a pair of runs differs in this group alone"
            else:
                status = "only moved together with other types"
            notes = []
            if at and width != BASE_WIDTH and not modelled:
                notes.append("not modelled: no term covers this cell")
            if at and width < BASE_WIDTH:
                notes.append(
                    "narrower, finer and rule-placed at once"
                    if placed == ["rule-placed"]
                    else "narrower and finer at once, image-centred"
                )
            if at and width > BASE_WIDTH:
                notes.append("wider and coarser at once, image-centred")
            if at and {item.family for item in at} == {OLD}:
                notes.append("only in the shared-readout recipe")
            rows.append(
                {
                    "series_type": kind,
                    "width_mm": round(width, 2),
                    "runs": len(at),
                    "runs_shared_readout": sum(item.family == OLD for item in at),
                    "runs_plane_readout": sum(item.family == NEW for item in at),
                    "isolating_pairs": len(pairs),
                    "status": status,
                    "placement": ", ".join(placed),
                    "term": ", ".join(modelled) if width != BASE_WIDTH else "",
                    "notes": "; ".join(notes),
                    "run_names": ", ".join(item.name for item in at),
                }
            )
    return rows


def preference_tables(used, groups, terms, flags, labels, measured, result):
    tested = {
        kind: {item.widths[index] for item in used} for index, kind in enumerate(TYPES)
    }
    scopes = {
        "every tested width": [
            sorted(set.intersection(*(tested[kind] for kind in group.members)))
            for group in groups
        ],
        "widths isolated in the plane-readout recipe": [
            sorted(
                {BASE_WIDTH}
                | {
                    width
                    for width in tested[group.members[-1]]
                    if isolating_pairs(used, groups, group.members[-1], width, NEW)
                }
            )
            for group in groups
        ],
    }
    incumbent = np.array(design_row(DETAIL, NEW, (), terms, flags))
    estimate = result["estimate"]
    combinations = []
    preferences = []
    for scope, options in scopes.items():
        choices = list(product(*options))
        rows = []
        for choice in choices:
            widths = dict.fromkeys(TYPES, BASE_WIDTH)
            for group, width in zip(groups, choice, strict=True):
                widths |= dict.fromkeys(group.members, width)
            rows.append(
                design_row([widths[kind] for kind in TYPES], NEW, (), terms, flags)
            )
        rows = np.array(rows)
        predicted = rows @ estimate
        own = predicted.argmax(0)
        for column, label in enumerate(labels):
            best = int(own[column])
            step = rows[best] - incumbent
            sigma = result["sigma"][column]
            record = {
                "scope": scope,
                "label": label,
                **{
                    f"width|{group.name}": round(width, 2)
                    for group, width in zip(groups, choices[best], strict=True)
                },
                "fitted_at_incumbent_widths": float(incumbent @ estimate[:, column]),
                "measured_incumbent": float(measured[column]),
                "predicted": float(predicted[best, column]),
                "predicted_se_training": float(
                    sigma * np.linalg.norm(rows[best] @ result["hat"])
                ),
                "worst_predicted": float(predicted[:, column].min()),
                "sd_one_run_against_measured_incumbent": float(
                    sigma * np.hypot(np.linalg.norm(step @ result["hat"]), np.sqrt(2))
                ),
                **{
                    f"delta_{key}": value
                    for key, value in linear(result, step, column).items()
                },
            }
            (combinations if label == "macro" else preferences).append(record)
        steps = rows[own[:-1]] - incumbent
        gain = float(np.einsum("lp,pl->", steps, estimate[:, :-1]) / len(steps))
        routed = np.einsum("lp,dpl->d", steps, result["estimate_draws"][:, :, :-1])
        combinations.append(
            {
                "scope": scope,
                "label": "macro if every label had its own widths (upper bound)",
                "fitted_at_incumbent_widths": float(incumbent @ estimate[:, -1]),
                "measured_incumbent": float(measured[-1]),
                "predicted": float(incumbent @ estimate[:, -1]) + gain,
                "delta_estimate": gain,
                "delta_se_validation": float(routed.std() / len(steps)),
            }
        )
    return combinations, preferences


def single_change_rows(used, groups, terms, flags, labels, result) -> list[dict]:
    base = dict(zip(TYPES, DETAIL, strict=True))
    incumbent = np.array(design_row(DETAIL, NEW, (), terms, flags))
    rows = []
    for group in groups:
        tested = set.intersection(
            *(
                {item.widths[TYPES.index(kind)] for item in used}
                for kind in group.members
            )
        )
        start = base[group.members[0]]
        for width in sorted(tested - {start}):
            widths = base | dict.fromkeys(group.members, width)
            step = (
                np.array(
                    design_row([widths[kind] for kind in TYPES], NEW, (), terms, flags)
                )
                - incumbent
            )
            isolated = bool(
                isolating_pairs(used, groups, group.members[-1], width, NEW)
            )
            rows += [
                {
                    "group": group.name,
                    "from_mm": round(start, 2),
                    "to_mm": round(width, 2),
                    "isolated_in_plane_readout": isolated,
                    "label": label,
                }
                | linear(result, step, column)
                for column, label in enumerate(labels)
            ]
    step = np.array(design_row(ALL_120, NEW, (), terms, flags)) - incumbent
    rows += [
        {
            "group": "every coronal and sagittal type",
            "to_mm": BASE_WIDTH,
            "isolated_in_plane_readout": False,
            "label": label,
        }
        | linear(result, step, column)
        for column, label in enumerate(labels)
    ]
    return rows


def conflict_tables(coefficients, features) -> tuple[list[dict], list[dict]]:
    keep = (
        "feature",
        "effect",
        "label",
        "estimate",
        "low_95",
        "high_95",
        "distinguishable_80",
        "distinguishable_95",
    )
    conflicts = [
        {key: row[key] for key in keep}
        for row in coefficients
        if not row["feature"].startswith("family:") and row["label"] != "macro"
    ]
    summary = []
    for feature in features:
        rows = sorted(
            (row for row in conflicts if row["feature"] == feature),
            key=lambda row: -row["estimate"],
        )
        if not rows:
            continue
        gains = [row for row in rows if row["estimate"] > 0]
        losses = [row for row in reversed(rows) if row["estimate"] < 0]
        summary.append(
            {
                "feature": feature,
                "effect": rows[0]["effect"],
                "labels_gaining": len(gains),
                "labels_losing": len(losses),
                "spread": rows[0]["estimate"] - rows[-1]["estimate"],
                "gaining": ", ".join(
                    f"{row['label']} {row['estimate']:+.4f}" for row in gains
                ),
                "losing": ", ".join(
                    f"{row['label']} {row['estimate']:+.4f}" for row in losses
                ),
            }
        )
    return conflicts, summary


def run_table(roots, names, scores) -> list[dict]:
    rows = []
    for item in RUNS:
        root = roots[item.name]
        row = {
            "run": item.name,
            "family": item.family,
            "reference": item.reference or "",
            **dict(zip(TYPES, (round(width, 2) for width in item.widths), strict=True)),
            "narrow_windows": "rule-placed" if item.rule else "image-centred",
            "placement_flags": ", ".join(item.placement),
            "status": "in the regression" if item.name in names else "pending",
            "note": item.note,
            "path": str(root) if root else item.root,
        }
        if item.name in names:
            index = names.index(item.name)
            row |= {
                "macro_x4": float(scores[4][index, -1]),
                "macro_x1": float(scores[1][index, -1]),
            }
        rows.append(row)
    return rows


def score_table(names, labels, scores) -> pl.DataFrame:
    return frame(
        [
            {"run": name, **dict(zip(labels, map(float, row), strict=True))}
            for name, row in zip(names, scores, strict=True)
        ]
    )


def run(settings: Settings) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    roots = {item.name: resolve(item) for item in RUNS}
    checks = [check_views(item, roots[item.name]) for item in RUNS]
    used = [item for item in RUNS if finished(roots[item.name])]
    names = [item.name for item in used]
    pending = [item.name for item in RUNS if item.name not in names]
    loaded = load_scores(settings, used, roots)
    labels, scores, draws = loaded["labels"], loaded["scores"], loaded["draws"]
    groups, terms = build_terms(used)
    flags = sorted({flag for item in used for flag in item.placement})
    features = [
        f"family:{OLD}",
        f"family:{NEW}",
        *(term.name for term in terms),
        *(f"placement:{flag}" for flag in flags),
    ]
    effects = {term.name: term.description for term in terms}
    penalised = np.array([0.0] * 2 + [1.0] * len(terms) + [0.0] * len(flags))
    design = np.array(
        [
            design_row(item.widths, item.family, item.placement, terms, flags)
            for item in used
        ]
    )
    path = strength_path(design, penalised, scores[4], labels)
    strength = min(path, key=lambda row: row["pooled_label_rmse"])["strength"]
    results = {
        multiplier: analyse(
            scores[multiplier],
            draws[multiplier],
            design,
            penalised,
            strength,
            names,
            labels,
            features,
            effects,
        )
        for multiplier in (4, 1)
    }
    main = results[4]
    held_out, held_out_summary = held_out_tables(used, labels, scores[4], main)
    closures = {
        multiplier: contrast_rows(
            CLOSURES,
            names,
            labels,
            scores[multiplier],
            draws[multiplier],
            results[multiplier]["sigma"],
            results[multiplier]["degrees"],
        )
        for multiplier in (4, 1)
    }
    paths = contrast_rows(
        PATHS, names, labels, scores[4], draws[4], main["sigma"], main["degrees"]
    )
    combinations, preferences = preference_tables(
        used, groups, terms, flags, labels, scores[4][names.index(INCUMBENT)], main
    )
    conflicts, conflict_summary = conflict_tables(main["coefficients"], features)
    tables = {
        "runs": frame(run_table(roots, names, scores)),
        "view_check": frame(checks),
        "excluded_runs": frame(
            [
                {"run": name, "family": family, "reason": reason}
                for family, reason, members in EXCLUDED
                for name in members
            ]
        ),
        "run_label_auc_x4": score_table(names, labels, scores[4]),
        "run_label_auc_x1": score_table(names, labels, scores[1]),
        "design": score_table(names, features, design),
        "ridge_path": frame(path),
        "coefficients_x4": frame(main["coefficients"]),
        "coefficients_x1": frame(results[1]["coefficients"]),
        "noise_x4": frame(main["noise"]),
        "noise_x1": frame(results[1]["noise"]),
        "leave_one_run_out": frame(held_out),
        "leave_one_run_out_summary": frame(held_out_summary),
        "closure_x4": frame(closures[4]),
        "closure_x1": frame(closures[1]),
        "effect_paths": frame(paths),
        "best_combination": frame(combinations),
        "label_preferences": frame(preferences),
        "single_changes": frame(
            single_change_rows(used, groups, terms, flags, labels, main)
        ),
        "conflicts": frame(conflicts),
        "conflict_summary": frame(conflict_summary),
        "identifiability": frame(identifiability_rows(used, groups, terms)),
    }
    for name, table in tables.items():
        table.write_csv(settings.output_dir / f"{name}.csv", float_precision=6)
    model = {
        "runs": names,
        "pending": pending,
        "labels": labels,
        "positives": loaded["positives"],
        "features": features,
        "terms": [term.model_dump() for term in terms],
        "groups": [group.model_dump() for group in groups],
        "base_width_mm": BASE_WIDTH,
        "slice_share": SLICE_SHARE,
        "strength": strength,
        "strength_grid": list(STRENGTHS),
        "bootstrap": {
            "draws": settings.bootstrap_draws,
            "seed": settings.bootstrap_seed,
            "report_groups": loaded["report_groups"],
            "studies": loaded["studies"],
        },
        "largest_auc_difference_from_recorded": loaded["recorded"],
        "design": design.tolist(),
        "replicate_groups": main["replicates"],
        **{
            f"x{multiplier}": {
                "coefficients": {
                    feature: dict(zip(labels, map(float, row), strict=True))
                    for feature, row in zip(features, result["estimate"], strict=True)
                },
                "training_noise_sd": dict(
                    zip(labels, map(float, result["sigma"]), strict=True)
                ),
                "training_noise_degrees": dict(
                    zip(labels, map(float, result["degrees"]), strict=True)
                ),
                "hat": result["hat"].tolist(),
            }
            for multiplier, result in results.items()
        },
    }
    (settings.output_dir / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    print(f"runs in the regression: {len(names)}")
    print(f"pending: {', '.join(pending) or 'none'}")
    print(f"terms: {', '.join(term.name for term in terms)}")
    print(f"ridge strength: {strength}")
    print(f"largest AUC difference from the recorded values: {loaded['recorded']:.1e}")
    print(f"replicate groups: {main['replicates']}")
    with pl.Config(
        tbl_rows=60,
        tbl_cols=20,
        tbl_width_chars=220,
        float_precision=6,
        fmt_str_lengths=60,
    ):
        print(
            tables["coefficients_x4"]
            .filter(pl.col("label") == "macro")
            .select("feature", "estimate", "low_95", "high_95", "unpenalised_estimate")
        )
        print(
            tables["leave_one_run_out_summary"].filter(
                pl.col("label").is_in(["macro", "pooled over the 12 labels"])
            )
        )
        print(
            tables["best_combination"].select(
                "scope",
                "label",
                pl.col("^width.*$"),
                "predicted",
                "delta_estimate",
                "delta_low_95",
                "delta_high_95",
            )
        )
    print(f"tables and model.json written to {settings.output_dir}")


def main() -> None:
    settings = Settings()
    with threadpool_limits(limits=settings.threads):
        run(settings)


if __name__ == "__main__":
    main()
