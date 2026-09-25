import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.data.labels import STUDY, require_unique_ids
from knee_abnormality.data.report_attributes import (
    FileRecord,
    Gate,
    Merge,
    QuestionSet,
    TargetManifest,
    TargetQuestion,
    check_distributions,
    value_column,
    weight_column,
)

EXCLUDED = (
    "study_side",
    "medial_osteophytes",
    "lateral_osteophytes",
    "pf_osteophytes",
    "extra_hoffa",
)
MERGES = (
    Merge(
        question="overall_osteoarthritis",
        option="no_osteoarthritis",
        sources=("not_mentioned", "none"),
    ),
    *(
        Merge(
            question=f"{compartment}_cartilage_depth",
            option="no_damage",
            sources=("not_mentioned", "normal", "only_elsewhere"),
        )
        for compartment in ("medial", "lateral")
    ),
    Merge(
        question="pf_cartilage_depth",
        option="no_damage",
        sources=("not_mentioned", "normal"),
    ),
    *(
        Merge(
            question=f"{compartment}_cartilage_depth",
            option="full_thickness",
            sources=("full_thickness", "bone_on_bone"),
        )
        for compartment in ("medial", "lateral", "pf")
    ),
    *(
        Merge(
            question=f"{compartment}_cartilage_extent",
            option="no_loss",
            sources=("no_loss", "only_elsewhere"),
        )
        for compartment in ("medial", "lateral")
    ),
)
MENISCUS_TEARS = ("possible_tear", "tear", "displaced_or_truncated")
CARTILAGE_LOSS = ("unspecified_loss", "partial_low", "partial_high", "full_thickness")
GATES = (
    *(
        Gate(
            child=f"{side}_meniscus_{child}",
            parent=f"{side}_meniscus_status",
            parent_options=MENISCUS_TEARS,
            empty_option="no_tear",
            fallback_option="not_stated",
        )
        for side in ("medial", "lateral")
        for child in ("tear_location", "tear_type")
    ),
    Gate(
        child="mcl_timing",
        parent="mcl_status",
        parent_options=(
            "sprain_or_thickening",
            "possible_tear",
            "partial_unspecified",
            "partial_high_grade",
            "complete_tear",
        ),
        empty_option="no_injury",
        fallback_option="not_stated",
    ),
    Gate(
        child="contusion_location",
        parent="contusion_status",
        parent_options=("contusion", "possible"),
        empty_option="none",
        fallback_option="not_stated",
    ),
    Gate(
        child="baker_ruptured",
        parent="baker_size",
        parent_options=("small", "moderate", "large", "unspecified"),
        empty_option="no_cyst",
        fallback_option="intact",
    ),
    *(
        Gate(
            child=f"{compartment}_cartilage_extent",
            parent=f"{compartment}_cartilage_depth",
            parent_options=CARTILAGE_LOSS,
            empty_option="no_loss",
            fallback_option="not_stated",
        )
        for compartment in ("medial", "lateral", "pf")
    ),
    Gate(
        child="pf_cartilage_surface",
        parent="pf_cartilage_depth",
        parent_options=("softening", *CARTILAGE_LOSS),
        empty_option="no_damage",
        fallback_option="not_stated",
    ),
)
READINGS = ("forward", "reverse", "probability")
METHOD = (
    "JevK5 option probabilities averaged over both option orders, one column per question and "
    "option. Questions in excluded are dropped: knee side contradicts reflection augmentation, "
    "and the osteophyte and Hoffa answers were mostly position noise or invented findings. Each "
    "merge sums its source options into one option in both orders. Each question's weight is one "
    "minus the total-variation distance between its forward and reverse distributions after "
    "merging. Each gate conditions a child question on its parent: with p the parent's "
    "probability mass on parent_options, the child becomes (1 - p) on empty_option plus p times "
    "the child's own distribution with its empty_option mass moved to fallback_option, and its "
    "weight becomes (1 - p) times the parent's weight plus p times its own."
)


class PrepareCLI(CLISettings):
    model_config = describe(
        "Build the report-attribute target table for the auxiliary heads from a finished "
        "JevK5 extraction run (drops, merges, order-agreement weights, parent gates) and "
        "its pinned manifest"
    )

    attributes: Path = Field(
        default=Path("outputs/labels/jevk5-attributes-v3/attributes.parquet"),
        description="Extraction output; its run.json must name the question set",
    )
    question_set: Path = Field(
        default=Path("configs/labels/jevk5-report-attributes-v3.json"),
        description="Question set the extraction used",
    )
    output_dir: Path = Field(
        default=Path("data/labels/prepared/jevk5-report-attributes-v3.1"),
        description="New directory for the prepared target table",
    )
    output_manifest: Path = Field(
        default=Path("configs/labels/jevk5-report-attribute-targets-v3.1.json"),
        description="New manifest pinning the table, question set and source run",
    )


def target_questions(question_set: QuestionSet) -> tuple[TargetQuestion, ...]:
    absorbed = {
        (merge.question, source)
        for merge in MERGES
        for source in merge.sources
        if source != merge.option
    }
    questions = []
    for question in question_set.questions:
        if question.id in EXCLUDED:
            continue
        options = []
        for option in question.options:
            key = (question.id, option)
            if key in absorbed:
                first = next(
                    merge.option
                    for merge in MERGES
                    if merge.question == question.id and option in merge.sources
                )
                if first not in options:
                    options.append(first)
            elif option not in options:
                options.append(option)
        questions.append(TargetQuestion(id=question.id, options=tuple(options)))
    return tuple(questions)


def wide_readings(
    long: pl.DataFrame, question_set: QuestionSet, reading: str
) -> pl.DataFrame:
    options = [
        (question.id, option)
        for question in question_set.questions
        if question.id not in EXCLUDED
        for option in question.options
    ]
    studies = long.select(STUDY).unique()
    if long.height != studies.height * len(options):
        raise ValueError("Every study must answer every question option once")
    wide = long.with_columns(
        pl.format("attribute_{}__{}", "question", "option").alias("column")
    ).pivot(on="column", index=STUDY, values=reading)
    if set(wide.columns) != {STUDY, *(value_column(q, o) for q, o in options)}:
        raise ValueError("Attribute options differ from the question set")
    return wide.select(
        STUDY, *(pl.col(value_column(q, o)).cast(pl.Float64) for q, o in options)
    ).sort(STUDY)


def apply_merges(frame: pl.DataFrame) -> pl.DataFrame:
    for merge in MERGES:
        sources = [value_column(merge.question, source) for source in merge.sources]
        frame = frame.with_columns(
            pl.sum_horizontal(sources).alias(value_column(merge.question, merge.option))
        ).drop(
            [
                column
                for column in sources
                if column != value_column(merge.question, merge.option)
            ]
        )
    return frame


def order_weights(
    forward: pl.DataFrame, reverse: pl.DataFrame, questions: tuple[TargetQuestion, ...]
) -> pl.DataFrame:
    joined = forward.join(reverse, on=STUDY, suffix="__reverse")
    return joined.select(
        STUDY,
        *(
            (
                1
                - 0.5
                * pl.sum_horizontal(
                    (
                        pl.col(value_column(question.id, option))
                        - pl.col(f"{value_column(question.id, option)}__reverse")
                    ).abs()
                    for option in question.options
                )
            )
            .clip(0, 1)
            .alias(weight_column(question.id))
            for question in questions
        ),
    )


def apply_gate(
    frame: pl.DataFrame, gate: Gate, questions: dict[str, TargetQuestion]
) -> pl.DataFrame:
    child = questions[gate.child]
    parent = questions[gate.parent]
    if not set(gate.parent_options) <= set(parent.options) or not {
        gate.empty_option,
        gate.fallback_option,
    } <= set(child.options):
        raise ValueError(f"Gate {gate.child} names options its questions lack")
    mass = pl.sum_horizontal(
        pl.col(value_column(parent.id, option)) for option in gate.parent_options
    )
    empty = pl.col(value_column(child.id, gate.empty_option))

    def gated(option: str) -> pl.Expr:
        if option == gate.empty_option:
            return 1 - mass
        own = pl.col(value_column(child.id, option))
        if option == gate.fallback_option:
            own = own + empty
        return mass * own

    return frame.with_columns(
        *(
            gated(option).alias(value_column(child.id, option))
            for option in child.options
        ),
        (
            (1 - mass) * pl.col(weight_column(parent.id))
            + mass * pl.col(weight_column(child.id))
        ).alias(weight_column(child.id)),
    )


def prepare(settings: PrepareCLI) -> dict:
    if settings.output_dir.exists() or settings.output_manifest.exists():
        raise ValueError("Output paths must not exist")
    question_set = QuestionSet.model_validate_json(settings.question_set.read_text())
    run = json.loads((settings.attributes.parent / "run.json").read_text())
    if Path(run["identity"]["questions"]).name != settings.question_set.name:
        raise ValueError("The attribute run used a different question set")
    questions = target_questions(question_set)
    by_id = {question.id: question for question in questions}
    long = pl.read_parquet(settings.attributes).filter(
        pl.col("question").is_in(list(by_id))
    )
    readings = {
        reading: apply_merges(wide_readings(long, question_set, reading))
        for reading in READINGS
    }
    frame = readings["probability"].join(
        order_weights(readings["forward"], readings["reverse"], questions), on=STUDY
    )
    require_unique_ids(frame, "Report attributes")
    official = pl.read_csv("data/raw/train.csv", infer_schema_length=0).select(STUDY)
    if set(frame[STUDY]) != set(official[STUDY]):
        raise ValueError("Report attributes must cover every training study")
    for gate in GATES:
        frame = apply_gate(frame, gate, by_id)
    frame = frame.select(
        STUDY,
        *(
            pl.col(value_column(question.id, option)).cast(pl.Float32)
            for question in questions
            for option in question.options
        ),
        *(
            pl.col(weight_column(question.id)).cast(pl.Float32)
            for question in questions
        ),
    )
    check_distributions(frame, questions, "Prepared report attributes")
    settings.output_dir.mkdir(parents=True)
    targets_path = settings.output_dir / "targets.parquet"
    frame.write_parquet(targets_path)
    manifest = TargetManifest(
        version=1,
        prepared_at=datetime.now(UTC).isoformat(),
        method=METHOD,
        source_run=run["identity"],
        attributes=FileRecord.of(settings.attributes),
        question_set=FileRecord.of(settings.question_set),
        targets=FileRecord.of(targets_path),
        excluded=EXCLUDED,
        merges=MERGES,
        gates=GATES,
        questions=questions,
    )
    settings.output_manifest.write_text(manifest.model_dump_json(indent=2) + "\n")
    weights = frame.select(pl.col("^weight_.*$"))
    return {
        "manifest": str(settings.output_manifest),
        "targets": manifest.targets.model_dump(),
        "studies": frame.height,
        "questions": len(questions),
        "options": sum(len(question.options) for question in questions),
        "mean_weight": float(weights.to_numpy().mean()),
        "weights_below_half": float((weights.to_numpy() < 0.5).mean()),
    }


def main() -> None:
    report = prepare(PrepareCLI())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
