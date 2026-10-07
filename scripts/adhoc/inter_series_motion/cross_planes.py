import itertools
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from pydantic_settings import BaseSettings, SettingsConfigDict
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

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
    synthetic: bool = False


def load_volume(s):
    pos = s.ipp @ s.n
    drift = s.ipp - s.ipp[0] - np.outer(pos - pos[0], s.n)
    if np.abs(drift).max() > 0.5 or np.any(np.diff(pos) <= 0.1):
        return None
    sig = (max(0.5 / s.ps[0], 0.01), max(0.5 / s.ps[1], 0.01))
    vol = np.stack([gaussian_filter(ml.read_pixels(f), sig) for f in s.files])
    return vol, pos


def reformat(a, ia, b, vol, pos, shape, offset=(0.0, 0.0)):
    origin = a.ipp[ia] - b.ipp[0]
    ii, jj = np.meshgrid(
        np.arange(shape[0], dtype=np.float64) + offset[0], np.arange(shape[1], dtype=np.float64) + offset[1], indexing="ij"
    )
    col = (origin @ b.u + jj * (a.u @ b.u) + ii * (a.v @ b.u)) / b.ps[1]
    row = (origin @ b.v + jj * (a.u @ b.v) + ii * (a.v @ b.v)) / b.ps[0]
    t = pos[0] + origin @ b.n + jj * (a.u @ b.n) + ii * (a.v @ b.n)
    k = np.interp(t, pos, np.arange(len(pos)), left=-1, right=-1)
    mask = (k >= 0) & (row >= 0) & (row <= b.rows - 1) & (col >= 0) & (col <= b.cols - 1)
    out = map_coordinates(vol, [np.maximum(k, 0), row, col], order=1, mode="constant", cval=0.0)
    out = out * mask
    scale = np.percentile(out[mask], 99) if mask.any() else 1.0
    return np.sqrt(np.clip(out / max(scale, 1e-6), 0, 2)).astype(np.float32), mask


def measure(a, b, vol, pos, offset=(0.0, 0.0)):
    shape = ml.grid_shape(a)
    proj = np.array([b.n @ a.v, b.n @ a.u])
    axis = int(np.argmax(np.abs(proj)))
    ests = []
    for frac in FRACS:
        ia = int(round(frac * (len(a.files) - 1)))
        img_a, mask_a = ml.sample_on_grid(a, ia, a, ia, shape)
        img_a = gaussian_filter1d(img_a, b.gap / 2.0, axis=axis)
        img_b, mask_b = reformat(a, ia, b, vol, pos, shape, offset)
        fa, fb = ml.features(img_a, mask_a), ml.features(img_b, mask_b)
        if fa is None or fb is None:
            continue
        e = ml.estimate(fa, fb)
        if e is not None and e["overlap"] > 0.25:
            ests.append(e)
    if len(ests) < 3:
        return None
    c = ml.combine(ests, "ngf")
    c2 = ml.combine(ests, "ncc")
    if c is None or c2 is None:
        return None
    sy = np.array([e["ngf"]["sy"] for e in ests])
    sx = np.array([e["ngf"]["sx"] for e in ests])
    return {
        "sx": float(c["sx"]),
        "sy": float(c["sy"]),
        "ncc_sx": float(c2["sx"]),
        "ncc_sy": float(c2["sy"]),
        "edge": bool(c["edge"]),
        "peak": float(c["peak"]),
        "second": float(c["second"]),
        "dev": float(np.median(np.hypot(sy - c["sy"], sx - c["sx"]))),
        "overlap": float(np.median([e["overlap"] for e in ests])),
        "n_slices_used": len(ests),
    }


def work(args):
    study, series_rows, root, synthetic = args
    start = time.time()
    loaded = {}
    for r in series_rows:
        try:
            s = ml.load_series(Path(root) / study / r["SeriesInstanceUID"])
        except Exception:
            s = None
        if s is not None:
            loaded[s.uid] = (s, r)
    numbers = sorted(s.number for s, _ in loaded.values())
    items = sorted(loaded.values(), key=lambda t: (t[0].number, t[0].uid))
    volumes = {}
    rows = []
    for (a, ra), (b, rb) in itertools.combinations(items, 2):
        if ra["Anatomical_Plane"] == rb["Anatomical_Plane"]:
            continue
        angle = float(np.degrees(np.arccos(np.clip(abs(a.n @ b.n), 0, 1))))
        if angle < 60:
            continue
        if b.uid not in volumes:
            try:
                volumes[b.uid] = load_volume(b)
            except Exception:
                volumes[b.uid] = None
        if volumes[b.uid] is None:
            continue
        vol, pos = volumes[b.uid]
        line = np.cross(a.n, b.n)
        line = line / np.linalg.norm(line)
        across = np.cross(a.n, line)
        for kind, offset in (("pair", (0.0, 0.0)), ("synthetic", SYNTH)) if synthetic else (("pair", (0.0, 0.0)),):
            try:
                m = measure(a, b, vol, pos, offset)
            except Exception:
                m = None
            if m is None:
                continue
            vec = m["sx"] * a.u + m["sy"] * a.v
            vec2 = m["ncc_sx"] * a.u + m["ncc_sy"] * a.v
            rows.append(
                {
                    "study": study,
                    "kind": kind,
                    "plane_a": ra["Anatomical_Plane"],
                    "plane_b": rb["Anatomical_Plane"],
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
                    "desc_a": a.description,
                    "desc_b": b.description,
                    "gap_a": a.gap,
                    "gap_b": b.gap,
                    "angle_deg": angle,
                    "line": line.tolist(),
                    "along": float(vec @ line),
                    "across": float(vec @ across),
                    "ncc_along": float(vec2 @ line),
                    "ncc_across": float(vec2 @ across),
                    "lr": float(vec[0]),
                    "ap": float(vec[1]),
                    "si": float(vec[2]),
                    **m,
                }
            )
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
    tasks = [(s, grouped[s], root, cfg.synthetic) for s in order if s in grouped]
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.time()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(cfg.workers, maxtasksperchild=50) as pool:
        for i, res in enumerate(pool.imap_unordered(work, tasks, chunksize=1)):
            rows.extend(res)
            if (i + 1) % 50 == 0:
                print(i + 1, len(tasks), round(time.time() - start), flush=True)
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(cfg.output)
    print("done", len(rows), round(time.time() - start), flush=True)


if __name__ == "__main__":
    main()
