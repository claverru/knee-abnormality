import itertools
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from pydantic_settings import BaseSettings, SettingsConfigDict

sys.path.insert(0, str(Path(__file__).parent))
import motion_lib as ml

FRACS = (0.3, 0.4, 0.5, 0.6, 0.7)
SYNTH = (-4.5, 7.0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)
    data_root: Path = Path("data")
    output: Path
    limit: int = 200
    seed: int = 0
    workers: int = 4
    extras: bool = False
    skip_from: Path | None = None


def summarise(a, b, ests, kind, meta):
    row = dict(meta)
    row["kind"] = kind
    row["n_slices_used"] = len(ests)
    row["slice_offset_mm"] = float(np.median([e["slice_offset"] for e in ests]))
    row["overlap"] = float(np.median([e["overlap"] for e in ests]))
    for name in ("ngf", "ncc"):
        sy = np.array([e[name]["sy"] for e in ests])
        sx = np.array([e[name]["sx"] for e in ests])
        my, mx = float(np.median(sy)), float(np.median(sx))
        vec = mx * a.u + my * a.v
        row[f"{name}_sx_slices"] = sx.tolist()
        row[f"{name}_sy_slices"] = sy.tolist()
        row[f"{name}_sx"] = mx
        row[f"{name}_sy"] = my
        row[f"{name}_lr"] = float(vec[0])
        row[f"{name}_ap"] = float(vec[1])
        row[f"{name}_si"] = float(vec[2])
        row[f"{name}_mag"] = float(np.hypot(mx, my))
        row[f"{name}_spread"] = float(
            max(np.hypot(sy[i] - sy[j], sx[i] - sx[j]) for i in range(len(ests)) for j in range(len(ests)))
        )
        row[f"{name}_z"] = float(np.median([e[name]["z"] for e in ests]))
        row[f"{name}_peak"] = float(np.median([e[name]["peak"] for e in ests]))
        row[f"{name}_second"] = float(np.nanmedian([e[name]["second"] for e in ests]))
        row[f"{name}_edge"] = bool(any(e[name]["edge"] for e in ests))
        c = ml.combine(ests, name)
        if c is not None:
            cv = c["sx"] * a.u + c["sy"] * a.v
            row[f"{name}_c_sx"] = float(c["sx"])
            row[f"{name}_c_sy"] = float(c["sy"])
            row[f"{name}_c_lr"] = float(cv[0])
            row[f"{name}_c_ap"] = float(cv[1])
            row[f"{name}_c_si"] = float(cv[2])
            row[f"{name}_c_mag"] = float(np.hypot(c["sx"], c["sy"]))
            row[f"{name}_c_peak"] = float(c["peak"])
            row[f"{name}_c_second"] = float(c["second"])
            row[f"{name}_c_edge"] = bool(c["edge"])
            row[f"{name}_c_dev"] = float(np.median(np.hypot(sy - c["sy"], sx - c["sx"])))
    return row


def run_pair(a, b, kind, meta, offset=(0.0, 0.0), null=False):
    ests = []
    for frac in FRACS:
        ia = int(round(frac * (len(a.files) - 1)))
        try:
            if null:
                res = ml.pair_slice(a, ia, a, force_b=min(ia + 1, len(a.files) - 1))
            else:
                res = ml.pair_slice(a, ia, b, offset)
        except Exception:
            res = None
        if res is not None and (null or abs(res[0]["slice_offset"]) <= 0.6 * b.gap):
            ests.append(res[0])
    if len(ests) < 3:
        return None
    return summarise(a, b, ests, kind, meta)


def work(args):
    study, series_rows, root, extras = args
    start = time.time()
    rows = []
    loaded = {}
    for r in series_rows:
        try:
            s = ml.load_series(Path(root) / study / r["SeriesInstanceUID"])
        except Exception:
            s = None
        if s is not None:
            loaded[r["SeriesInstanceUID"]] = (s, r)
    numbers = sorted(s.number for s, _ in loaded.values())
    by_plane = {}
    for s, r in loaded.values():
        by_plane.setdefault(r["Anatomical_Plane"], []).append((s, r))
    for plane, items in by_plane.items():
        items.sort(key=lambda t: (t[0].number, t[0].uid))
        for (a, ra), (b, rb) in itertools.combinations(items, 2):
            meta = {
                "study": study,
                "plane": plane,
                "series_a": a.uid,
                "series_b": b.uid,
                "fluid_a": ra["Fluid_Sensitive"],
                "fluid_b": rb["Fluid_Sensitive"],
                "fs_a": ra["Fat_Suppression"],
                "fs_b": rb["Fat_Suppression"],
                "number_a": a.number,
                "number_b": b.number,
                "rank_a": numbers.index(a.number),
                "rank_b": numbers.index(b.number),
                "n_series_study": len(loaded),
                "desc_a": a.description,
                "desc_b": b.description,
                "manufacturer": a.manufacturer,
                "field": a.field,
                "phase_dir_a": a.phase_dir,
                "phase_dir_b": b.phase_dir,
                "bandwidth_a": a.bandwidth,
                "bandwidth_b": b.bandwidth,
                "gap_a": a.gap,
                "gap_b": b.gap,
                "ps_a": a.ps[0],
                "ps_b": b.ps[0],
                "normal_angle_deg": float(np.degrees(np.arccos(np.clip(abs(a.n @ b.n), 0, 1)))),
                "inplane_angle_deg": float(np.degrees(np.arccos(np.clip(abs(a.u @ b.u), 0, 1)))),
                "ua": a.u.tolist(),
                "va": a.v.tolist(),
            }
            row = run_pair(a, b, "pair", meta)
            if row is not None:
                rows.append(row)
            if extras:
                row = run_pair(a, b, "synthetic", meta, offset=SYNTH)
                if row is not None:
                    rows.append(row)
        if extras:
            for s, r in items:
                meta = {"study": study, "plane": plane, "series_a": s.uid, "series_b": s.uid, "gap_a": s.gap, "gap_b": s.gap}
                row = run_pair(s, s, "null_adjacent", meta, null=True)
                if row is not None:
                    rows.append(row)
    elapsed = time.time() - start
    for row in rows:
        row["study_seconds"] = elapsed
    return rows


def main():
    cfg = Settings()
    series = pl.read_csv(cfg.data_root / "raw" / "train_series.csv")
    studies = pl.read_csv(cfg.data_root / "folds" / "all_5fold_scanner_v2" / "studies.csv")["StudyInstanceUID"].sort().to_list()
    rng = np.random.default_rng(cfg.seed)
    order = [studies[i] for i in rng.permutation(len(studies))]
    if cfg.limit > 0:
        order = order[: cfg.limit]
    grouped = {k[0]: v.to_dicts() for k, v in series.group_by("StudyInstanceUID")}
    root = str(cfg.data_root / "raw" / "train_series")
    tasks = [(s, grouped[s], root, cfg.extras) for s in order if s in grouped]
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.time()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(cfg.workers, maxtasksperchild=200) as pool:
        for i, res in enumerate(pool.imap_unordered(work, tasks, chunksize=2)):
            rows.extend(res)
            if (i + 1) % 100 == 0:
                print(i + 1, len(tasks), round(time.time() - start), flush=True)
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(cfg.output)
    print("done", len(rows), round(time.time() - start), flush=True)


if __name__ == "__main__":
    main()
