from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD
from knee_abnormality.evaluation.metrics import aucs

TYPES = (
    "axial_t1pd",
    "axial_fluid",
    "coronal_t1pd",
    "coronal_fluid",
    "sagittal_t1pd",
    "sagittal_fluid",
)
MIXED = "outputs/autoresearch/mixed-field/experiments"
MANUAL = "outputs/autoresearch/mixed-field/manual"
SMART = "outputs/autoresearch/smart-crops/experiments"
RUNS = {
    "rn18-l2sp-no-confusion": "outputs/training/rn18-l2sp-no-confusion",
    "no-anatomical-bias": "outputs/autoresearch/revisits/experiments/"
    "no-anatomical-bias-20260930T213145Z-58126164/training",
    "rn18-field-120-control": "outputs/training/rn18-field-120-control",
    "field-160-single-logged": f"{MIXED}/field-160-single-logged-20261001T123626Z-d61f0fe9/training",
    "field-120-single": f"{MIXED}/field-120-single-20261001T124248Z-36c2ff33/training",
    "field-jitter": f"{MIXED}/field-jitter-20261001T130426Z-a1cbdc93/training",
    "mixed-field-140-160": f"{MIXED}/mixed-field-140-160-20261001T132509Z-95b83f62/training",
    "mixed-field-120-160": f"{MIXED}/mixed-field-120-160-20261001T134624Z-da64b1ee/training",
    "mixed-field-readout-weights": f"{MIXED}/mixed-field-readout-weights-20261001T140827Z-4be68530/training",
    "plane-weighted-fields": f"{MIXED}/plane-weighted-fields-20261001T143031Z-6b4e022b/training",
    "extent-ruled-field": f"{MIXED}/extent-ruled-field-20261001T145233Z-e5c2374a/training",
    "field-backbones-routed": f"{MIXED}/field-backbones-routed-20261001T151354Z-0c7e4217/training",
    "contrast-split-112-140": f"{MIXED}/contrast-split-112-140-20261001T153730Z-e26452f3/training",
    "two-stream-paired": f"{MIXED}/two-stream-paired-20261001T172835Z-5c197a3e/training",
    "middle-slice-augmentation": f"{MANUAL}/middle-slice-augmentation-20261001/training",
    "field-120-middle-slice-augmentation": f"{MANUAL}/field-120-middle-slice-augmentation-20261001/training",
    "t1pd-detail-84": f"{SMART}/t1pd-detail-84-20261001T220724Z-00bb3343/training",
    "crop-rule": f"{SMART}/crop-rule-20261001T223846Z-d6c1d1d9/training",
    "crop-attention": f"{SMART}/crop-attention-20261001T230037Z-fbfeebae/training",
    "crop-compartments": f"{SMART}/crop-compartments-20261001T234621Z-4f217655/training",
    "crop-own-loss": f"{SMART}/crop-own-loss-20261001T234049Z-fde20105/training",
    "two-views-fluid-140": f"{SMART}/two-views-fluid-140-20261002T002445Z-5ee08c36/training",
    "two-views-112-fluid-140": f"{SMART}/two-views-112-fluid-140-20261002T003538Z-5a04bae3/training",
    "sagittal-fluid-detail-rectangle": f"{SMART}/sagittal-fluid-detail-rectangle-20261002T012346Z-23fe8ce0/training",
    "t1pd-detail-acquisition-matched": f"{SMART}/t1pd-detail-acquisition-matched-20261002T014021Z-ab17c719/training",
    "crop-evidence-utility-placement": f"{SMART}/crop-evidence-utility-placement-20261002T020617Z-b1dc5ca2/training",
    "crop-rule-overlap-replacement": f"{SMART}/crop-rule-overlap-replacement-20261002T022812Z-c8358a22/training",
    "sagittal-fluid-correspondence-fusion": f"{SMART}/sagittal-fluid-correspondence-fusion-20261002T030451Z-5c8e4a3c/training",
    "t1pd-detail-acquired-matrix": f"{SMART}/t1pd-detail-acquired-matrix-20261002T031657Z-4c29e2ee/training",
    "crop-context-teacher-distillation": f"{SMART}/crop-context-teacher-distillation-20261002T034308Z-ce5b8d32/training",
    "crop-full-context-ranking-locator": f"{SMART}/crop-full-context-ranking-locator-20261002T045147Z-789cb96d/training",
    "t1pd-foveated-native-view": f"{SMART}/t1pd-foveated-native-view-20261002T045517Z-2216951a/training",
    "t1pd-shallow-detail-fusion": f"{SMART}/t1pd-shallow-detail-fusion-20261002T052310Z-68f4ae79/training",
    "sagittal-fluid-content-residual": f"{SMART}/sagittal-fluid-content-residual-20261002T054942Z-99f47393/training",
    "fluid-detail-93-replacement": f"{SMART}/fluid-detail-93-replacement-20261002T061326Z-c4e4d640/training",
    "t1pd-detail-dense-final-grid": f"{SMART}/t1pd-detail-dense-final-grid-20261002T064737Z-8c9118c6/training",
}
ALL_140 = dict.fromkeys(TYPES, "140")
ALL_120 = dict.fromkeys(TYPES, "120")
T84 = {**ALL_120, "coronal_t1pd": "84 rule", "sagittal_t1pd": "84 rule"}
CROP = "120 + 60 crop"
CROPS = {**ALL_120, "coronal_t1pd": CROP, "sagittal_t1pd": CROP}
T1PD = ("axial_t1pd", "coronal_t1pd", "sagittal_t1pd")
FLUID = ("axial_fluid", "coronal_fluid", "sagittal_fluid")
DETAIL_T1PD = ("coronal_t1pd", "sagittal_t1pd")
SAGITTAL_RECTANGLE = {**T84, "sagittal_fluid": "120 + 93x67 rule"}
PAIRS = (
    (
        "every type, one field",
        "field-160-single-logged",
        "rn18-l2sp-no-confusion",
        ALL_140,
        dict.fromkeys(TYPES, "160"),
        "",
    ),
    (
        "every type, one field",
        "field-120-single",
        "rn18-l2sp-no-confusion",
        ALL_140,
        ALL_120,
        "",
    ),
    (
        "every type, one field",
        "field-120-middle-slice-augmentation",
        "rn18-l2sp-no-confusion",
        ALL_140,
        ALL_120,
        "also middle-slice augmentation",
    ),
    (
        "every type, one field",
        "two-stream-paired",
        "rn18-l2sp-no-confusion",
        ALL_140,
        ALL_120,
        "second 140 mm stream never trained",
    ),
    (
        "every type, one field",
        "rn18-field-120-control",
        "no-anatomical-bias",
        ALL_140,
        ALL_120,
        "plane-contrast readouts on both",
    ),
    (
        "every type, one field",
        "field-120-single",
        "field-160-single-logged",
        dict.fromkeys(TYPES, "160"),
        ALL_120,
        "derived pair",
    ),
    (
        "split by contrast",
        "contrast-split-112-140",
        "rn18-l2sp-no-confusion",
        ALL_140,
        {**ALL_140, **dict.fromkeys(T1PD, "112")},
        "",
    ),
    (
        "split by contrast",
        "middle-slice-augmentation",
        "rn18-l2sp-no-confusion",
        ALL_140,
        {**ALL_140, **dict.fromkeys(T1PD, "112")},
        "also middle-slice augmentation",
    ),
    (
        "split by contrast",
        "contrast-split-112-140",
        "field-120-single",
        ALL_120,
        {**ALL_140, **dict.fromkeys(T1PD, "112")},
        "derived pair: T1/PD narrower, fluid wider",
    ),
    (
        "coronal and sagittal detail views",
        "t1pd-detail-84",
        "rn18-field-120-control",
        ALL_120,
        T84,
        "",
    ),
    (
        "coronal and sagittal detail views",
        "fluid-detail-93-replacement",
        "t1pd-detail-84",
        T84,
        {**T84, "coronal_fluid": "93 rule", "sagittal_fluid": "93 rule"},
        "S93",
    ),
    (
        "coronal and sagittal detail views",
        "fluid-detail-93-replacement",
        "rn18-field-120-control",
        ALL_120,
        {**T84, "coronal_fluid": "93 rule", "sagittal_fluid": "93 rule"},
        "S93 against S92",
    ),
    (
        "coronal and sagittal detail views",
        "t1pd-detail-acquisition-matched",
        "t1pd-detail-84",
        T84,
        {**T84, **dict.fromkeys(DETAIL_T1PD, "84-120 by stored pixel")},
        "wider where the series is coarser",
    ),
    (
        "coronal and sagittal detail views",
        "t1pd-detail-acquired-matrix",
        "t1pd-detail-84",
        T84,
        {**T84, **dict.fromkeys(DETAIL_T1PD, "84-120 by acquired matrix")},
        "wider where the series is coarser",
    ),
    (
        "coronal and sagittal detail views",
        "t1pd-foveated-native-view",
        "t1pd-detail-84",
        T84,
        {**T84, **dict.fromkeys(DETAIL_T1PD, "120 foveated, 80 core at 0.4167")},
        "",
    ),
    (
        "coronal and sagittal detail views",
        "t1pd-shallow-detail-fusion",
        "t1pd-detail-84",
        T84,
        {**T84, **dict.fromkeys(DETAIL_T1PD, "120 fused with 84 at stride 8")},
        "",
    ),
    (
        "coronal and sagittal detail views",
        "t1pd-detail-dense-final-grid",
        "t1pd-detail-84",
        T84,
        {**T84, **dict.fromkeys(DETAIL_T1PD, "84 rule, stride-16 final grid")},
        "same field, denser tokens",
    ),
    (
        "mixed fields in one model",
        "field-jitter",
        "rn18-l2sp-no-confusion",
        ALL_140,
        dict.fromkeys(TYPES, "train 120/140/160, predict 140"),
        "",
    ),
    (
        "mixed fields in one model",
        "mixed-field-140-160",
        "rn18-l2sp-no-confusion",
        ALL_140,
        dict.fromkeys(TYPES, "140/160 alternating"),
        "",
    ),
    (
        "mixed fields in one model",
        "mixed-field-120-160",
        "rn18-l2sp-no-confusion",
        ALL_140,
        dict.fromkeys(TYPES, "120/160 alternating"),
        "",
    ),
    (
        "mixed fields in one model",
        "mixed-field-readout-weights",
        "rn18-l2sp-no-confusion",
        ALL_140,
        dict.fromkeys(TYPES, "120/160 alternating"),
        "per-field readout weights",
    ),
    (
        "mixed fields in one model",
        "field-backbones-routed",
        "rn18-l2sp-no-confusion",
        ALL_140,
        dict.fromkeys(TYPES, "120/160 alternating"),
        "one backbone per field",
    ),
    (
        "mixed fields in one model",
        "plane-weighted-fields",
        "rn18-l2sp-no-confusion",
        ALL_140,
        {
            **ALL_120,
            "axial_t1pd": "120, half at 160",
            "axial_fluid": "120, half at 160",
            "sagittal_t1pd": "120, quarter at 160",
            "sagittal_fluid": "120, quarter at 160",
        },
        "",
    ),
    (
        "mixed fields in one model",
        "extent-ruled-field",
        "rn18-l2sp-no-confusion",
        ALL_140,
        dict.fromkeys(TYPES, "140, 160 if tissue overhangs"),
        "",
    ),
    (
        "second view beside the main one",
        "two-views-fluid-140",
        "rn18-field-120-control",
        ALL_120,
        {**ALL_120, **dict.fromkeys(FLUID, "120 + 140 ResNet-10t")},
        "",
    ),
    (
        "second view beside the main one",
        "two-views-112-fluid-140",
        "rn18-field-120-control",
        ALL_120,
        {**dict.fromkeys(TYPES, "112"), **dict.fromkeys(FLUID, "112 + 140 ResNet-10t")},
        "",
    ),
    (
        "second view beside the main one",
        "sagittal-fluid-detail-rectangle",
        "t1pd-detail-84",
        T84,
        SAGITTAL_RECTANGLE,
        "",
    ),
    (
        "second view beside the main one",
        "sagittal-fluid-correspondence-fusion",
        "t1pd-detail-84",
        T84,
        SAGITTAL_RECTANGLE,
        "fused into the main tokens",
    ),
    (
        "second view beside the main one",
        "sagittal-fluid-content-residual",
        "t1pd-detail-84",
        T84,
        SAGITTAL_RECTANGLE,
        "fused as a residual",
    ),
    (
        "second view beside the main one",
        "crop-rule",
        "rn18-field-120-control",
        ALL_120,
        CROPS,
        "",
    ),
    (
        "second view beside the main one",
        "crop-attention",
        "rn18-field-120-control",
        ALL_120,
        CROPS,
        "attention-placed",
    ),
    (
        "second view beside the main one",
        "crop-compartments",
        "rn18-field-120-control",
        ALL_120,
        CROPS,
        "two crops per coronal slice",
    ),
    (
        "second view beside the main one",
        "crop-own-loss",
        "rn18-field-120-control",
        ALL_120,
        CROPS,
        "crop loss",
    ),
    (
        "second view beside the main one",
        "crop-evidence-utility-placement",
        "rn18-field-120-control",
        ALL_120,
        CROPS,
        "evidence-placed",
    ),
    (
        "second view beside the main one",
        "crop-rule-overlap-replacement",
        "rn18-field-120-control",
        ALL_120,
        CROPS,
        "wide tokens under the crop removed",
    ),
    (
        "second view beside the main one",
        "crop-context-teacher-distillation",
        "rn18-field-120-control",
        ALL_120,
        {**ALL_120, **dict.fromkeys(DETAIL_T1PD, "120, 60 crop in training only")},
        "crop only as a training teacher",
    ),
    (
        "second view beside the main one",
        "crop-full-context-ranking-locator",
        "rn18-field-120-control",
        ALL_120,
        CROPS,
        "learned locator",
    ),
)


class Settings(CLISettings):
    model_config = describe(
        "Tabulate the per-label x4 fold-0 AUC delta of every recorded run that changed "
        "the field or view of a series type against its reference, with the field each "
        "series type had on both sides"
    )

    split_dir: Path = Field(
        default=Path("data/folds/all_5fold_scanner_v2"),
        description="Frozen split with targets.csv",
    )
    output_dir: Path = Field(description="Output directory")


def run(settings: Settings) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    truth = pl.read_csv(settings.split_dir / "targets.csv")
    targets = truth.columns[1:]
    scores = {}
    for name, path in RUNS.items():
        frame = pl.read_csv(Path(path) / "fold_0" / "validation_predictions.csv")
        joined = frame.select(STUDY, *targets).join(
            truth, on=STUDY, how="left", suffix="|truth", validate="1:1"
        )
        y = (
            joined.select([f"{t}|truth" for t in targets]).to_numpy()
            >= TARGET_THRESHOLD
        ).astype(np.int64)
        result = aucs(y, joined.select(targets).to_numpy(), targets)
        scores[name] = {
            "studies": joined.height,
            "macro": result["macro_auc"],
            **result["per_target_auc"],
        }
    pl.DataFrame([{"run": name, **value} for name, value in scores.items()]).write_csv(
        settings.output_dir / "run_label_auc.csv", float_precision=6
    )
    rows = []
    for group, name, reference, before, after, note in PAIRS:
        rows.append(
            {
                "group": group,
                "run": name,
                "reference": reference,
                "runs": 1,
                "note": note,
                **{
                    kind: after[kind]
                    if after[kind] == before[kind]
                    else f"{before[kind]} -> {after[kind]}"
                    for kind in TYPES
                },
                "macro_reference": scores[reference]["macro"],
                "macro_run": scores[name]["macro"],
                "delta_macro": scores[name]["macro"] - scores[reference]["macro"],
                **{
                    f"delta|{target}": scores[name][target] - scores[reference][target]
                    for target in targets
                },
            }
        )
    frame = pl.DataFrame(rows)
    frame.write_csv(settings.output_dir / "run_label_deltas.csv", float_precision=6)
    with pl.Config(tbl_rows=100, tbl_cols=30, tbl_width_chars=300, float_precision=4):
        print(frame.drop(*TYPES, "note", "runs", "group"))


def main() -> None:
    run(Settings())


if __name__ == "__main__":
    main()
