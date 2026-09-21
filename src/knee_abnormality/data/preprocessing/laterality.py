import json
import math
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from knee_abnormality.data.preprocessing.geometry import image_center, valid_pose

UNKNOWN, LEFT, RIGHT = 0, 1, 2
SIDE_NAMES = ("unknown", "left", "right")
EXPLICIT_FIELDS = ("Laterality", "ImageLaterality")
TEXT_FIELDS = (
    "StudyDescription",
    "SeriesDescription",
    "ProtocolName",
    "BodyPartExamined",
)
EXPLICIT_VALUES = {
    "L": "L",
    "LEFT": "L",
    "R": "R",
    "RIGHT": "R",
    "B": "B",
    "BOTH": "B",
    "BILATERAL": "B",
}
WORDS = {
    "L": {
        "LEFT",
        "LINKS",
        "LINKE",
        "LINKEN",
        "LINKES",
        "IZQUIERDA",
        "IZQUIERDO",
        "GAUCHE",
        "ESQUERDO",
        "ESQUERDA",
        "SINISTRO",
        "SINISTRA",
    },
    "R": {
        "RIGHT",
        "RECHTS",
        "RECHTE",
        "RECHTEN",
        "RECHTES",
        "DERECHA",
        "DERECHO",
        "DROIT",
        "DROITE",
        "DIREITO",
        "DIREITA",
        "DESTRO",
        "DESTRA",
    },
}
ABBREVIATIONS = {
    "L": {"L", "LT", "LI", "IZQ", "IZDA", "ESQ", "SX"},
    "R": {"R", "RT", "RE", "DER", "DCHA", "DIR", "DX"},
}
ANATOMY = {"KNEE", "KNIE", "RODILLA", "GENOU", "JOELHO", "GINOCCHIO"}
SIDE_SHORTHAND = {"L": {"LT", "IZQ", "IZDA", "ESQ"}, "R": {"RT", "DCHA"}}
MRI_CONTEXT = {
    "SAG",
    "SAGITTAL",
    "COR",
    "CORONAL",
    "AX",
    "AXI",
    "AXIAL",
    "TRA",
    "PD",
    "T1",
    "T2",
    "T1W",
    "T2W",
    "PDW",
    "TSE",
    "FSE",
}
MINIMUM_COORDINATE_SERIES = 2
MINIMUM_ABS_CENTER_X_MM = 40.0

SIDE_POLICY = {
    "algorithm": "explicit_then_coordinate_consensus_knee_side_v1",
    "states": list(SIDE_NAMES),
    "explicit_fields": list(EXPLICIT_FIELDS),
    "explicit_values": EXPLICIT_VALUES,
    "text_fields": list(TEXT_FIELDS),
    "text_parser": "reviewed_words_shorthand_anatomy_or_mri_context_tokens_v1",
    "headers": "all_readable_loader_indexed_headers_of_supported_series",
    "series_coordinates": "first_middle_last_indexed_records_by_filename_all_valid_native_poses_median_image_center_x",
    "study_coordinates": "median_of_series_centers",
    "minimum_coordinate_series": MINIMUM_COORDINATE_SERIES,
    "minimum_abs_center_x_mm": MINIMUM_ABS_CENTER_X_MM,
    "series_sign_agreement": "all_usable_series",
    "coordinate_sign": {"positive_x": "left", "negative_x": "right"},
    "precedence": [
        "bilateral_or_conflicting_explicit_or_text_unknown",
        "qualified_coordinates_contradicting_explicit_or_text_unknown",
        "consistent_explicit_side",
        "qualified_coordinates_with_absent_or_agreeing_text",
        "otherwise_unknown",
    ],
    "excluded_inputs": [
        "reports",
        "labels",
        "study_identifiers",
        "manual_lookup_tables",
        "manufacturer",
        "pixels",
    ],
}


@dataclass(frozen=True, slots=True)
class HeaderSide:
    explicit: frozenset[str]
    text: frozenset[str]
    manufacturer: str
    model: str


@dataclass(frozen=True, slots=True)
class SeriesSide:
    explicit: tuple[str, ...]
    text: tuple[str, ...]
    manufacturers: tuple[str, ...]
    models: tuple[str, ...]
    center_x_mm: float | None
    coordinate_reason: str


@dataclass(frozen=True, slots=True)
class KneeSide:
    value: int
    source: str
    reason: str
    explicit: tuple[str, ...]
    text: tuple[str, ...]
    coordinate_side: int
    coordinate_reason: str
    center_x_mm: float | None
    center_range_mm: tuple[float, float] | None
    coordinate_series: tuple[str, ...]
    omitted_series: tuple[tuple[str, str], ...]
    manufacturers: tuple[str, ...]
    models: tuple[str, ...]


@lru_cache(maxsize=4096)
def text_sides(value: str) -> tuple[str, ...]:
    normalized = (
        unicodedata.normalize("NFKD", value.upper()).encode("ascii", "ignore").decode()
    )
    tokens = set(re.findall(r"[A-Z0-9]+", normalized))
    return tuple(
        sorted(
            side
            for side in WORDS
            if tokens & (WORDS[side] | SIDE_SHORTHAND[side])
            or (tokens & ANATOMY and tokens & ABBREVIATIONS[side])
            or (tokens & MRI_CONTEXT and side in tokens)
        )
    )


def explicit_side(value: object) -> str | None:
    return EXPLICIT_VALUES.get(str(value).strip().upper())


def header_side(values: dict[str, object]) -> HeaderSide:
    explicit = {
        side
        for field in EXPLICIT_FIELDS
        if (side := explicit_side(values.get(field) or "")) is not None
    }
    text = {
        side
        for field in TEXT_FIELDS
        if values.get(field)
        for side in text_sides(str(values[field]))
    }
    return HeaderSide(
        frozenset(explicit),
        frozenset(text),
        str(values.get("Manufacturer") or ""),
        str(values.get("ManufacturerModelName") or ""),
    )


def series_side(records: tuple, headers: tuple[HeaderSide, ...]) -> SeriesSide:
    explicit = tuple(sorted({side for header in headers for side in header.explicit}))
    text = tuple(sorted({side for header in headers for side in header.text}))
    manufacturers = tuple(
        sorted({header.manufacturer for header in headers if header.manufacturer})
    )
    models = tuple(sorted({header.model for header in headers if header.model}))
    ordered = sorted(records, key=lambda record: record.path.name)
    if not ordered:
        return SeriesSide(explicit, text, manufacturers, models, None, "empty_series")
    poses = [
        ordered[index].pose
        for index in sorted({0, len(ordered) // 2, len(ordered) - 1})
    ]
    if any(pose is None or not valid_pose(pose) for pose in poses):
        return SeriesSide(
            explicit, text, manufacturers, models, None, "invalid_native_pose"
        )
    center = float(np.median([image_center(pose)[0] for pose in poses]))
    if not math.isfinite(center):
        return SeriesSide(
            explicit, text, manufacturers, models, None, "nonfinite_center"
        )
    return SeriesSide(explicit, text, manufacturers, models, center, "usable")


def _coordinate_side(
    centers: list[float],
) -> tuple[int, str, float | None, tuple[float, float] | None]:
    if not centers:
        return UNKNOWN, "no_usable_series", None, None
    center = float(np.median(centers))
    extent = (min(centers), max(centers))
    if len(centers) < MINIMUM_COORDINATE_SERIES:
        return UNKNOWN, "insufficient_series", center, extent
    if abs(center) <= MINIMUM_ABS_CENTER_X_MM:
        return UNKNOWN, "near_origin", center, extent
    if not (extent[0] > 0 or extent[1] < 0):
        return UNKNOWN, "series_sign_disagreement", center, extent
    return LEFT if center > 0 else RIGHT, "qualified", center, extent


def resolve_knee_side(series: tuple[tuple[str, SeriesSide], ...]) -> KneeSide:
    ordered = sorted(series, key=lambda item: item[0])
    explicit = tuple(sorted({side for _, value in ordered for side in value.explicit}))
    text = tuple(sorted({side for _, value in ordered for side in value.text}))
    usable = [(uid, value) for uid, value in ordered if value.center_x_mm is not None]
    coordinate, coordinate_reason, center, extent = _coordinate_side(
        [value.center_x_mm for _, value in usable]
    )
    metadata = {side for side in (*explicit, *text) if side != "B"}
    if "B" in explicit:
        value, source, reason = UNKNOWN, "none", "bilateral_explicit"
    elif len(metadata) > 1:
        value, source, reason = UNKNOWN, "none", "conflicting_explicit_or_text"
    else:
        metadata_side = (
            {"L": LEFT, "R": RIGHT}[next(iter(metadata))] if metadata else UNKNOWN
        )
        if coordinate and metadata_side and coordinate != metadata_side:
            value, source, reason = UNKNOWN, "none", "coordinates_contradict_metadata"
        elif explicit:
            value, source = metadata_side, "explicit"
            reason = (
                "explicit_coordinates_agree"
                if coordinate
                else "explicit_coordinates_inconclusive"
            )
        elif coordinate:
            value, source = coordinate, "coordinates"
            reason = "coordinates_agree_text" if text else "coordinates_only"
        elif text:
            value, source, reason = UNKNOWN, "none", "text_only"
        else:
            value, source, reason = UNKNOWN, "none", "no_qualified_evidence"
    return KneeSide(
        value,
        source,
        reason,
        explicit,
        text,
        coordinate,
        coordinate_reason,
        center,
        extent,
        tuple(uid for uid, _ in usable),
        tuple(
            (uid, value.coordinate_reason)
            for uid, value in ordered
            if value.center_x_mm is None
        ),
        tuple(sorted({name for _, value in ordered for name in value.manufacturers})),
        tuple(sorted({name for _, value in ordered for name in value.models})),
    )


def side_record(study_uid: str, side: KneeSide) -> dict:
    return {
        "StudyInstanceUID": study_uid,
        "knee_side": side.value,
        "side": SIDE_NAMES[side.value],
        "source": side.source,
        "reason": side.reason,
        "explicit_sides": "".join(side.explicit),
        "text_sides": "".join(side.text),
        "coordinate_side": SIDE_NAMES[side.coordinate_side],
        "coordinate_reason": side.coordinate_reason,
        "center_x_mm": side.center_x_mm,
        "min_series_center_x_mm": side.center_range_mm[0]
        if side.center_range_mm
        else None,
        "max_series_center_x_mm": side.center_range_mm[1]
        if side.center_range_mm
        else None,
        "coordinate_series": json.dumps(side.coordinate_series),
        "omitted_series": json.dumps(side.omitted_series),
        "manufacturers": json.dumps(side.manufacturers),
        "models": json.dumps(side.models),
        "policy": SIDE_POLICY["algorithm"],
    }
