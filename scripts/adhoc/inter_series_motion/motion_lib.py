from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from scipy import fft as sfft
from scipy.ndimage import binary_erosion, gaussian_filter, map_coordinates

SEARCH = 20
TAGS = [
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
    "Rows",
    "Columns",
    "SeriesNumber",
    "SeriesDescription",
    "Manufacturer",
    "MagneticFieldStrength",
    "InPlanePhaseEncodingDirection",
    "PixelBandwidth",
    "SpacingBetweenSlices",
]


@dataclass
class Series:
    uid: str
    files: list
    ipp: np.ndarray
    u: np.ndarray
    v: np.ndarray
    n: np.ndarray
    ps: tuple
    rows: int
    cols: int
    number: int
    description: str
    manufacturer: str
    field: float
    phase_dir: str
    bandwidth: float
    gap: float


def load_series(folder):
    folder = Path(folder)
    recs = []
    for f in sorted(folder.iterdir()):
        try:
            d = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=TAGS)
            recs.append(
                (
                    f,
                    np.array(d.ImagePositionPatient, dtype=float),
                    np.array(d.ImageOrientationPatient, dtype=float),
                    tuple(float(x) for x in d.PixelSpacing),
                    int(d.Rows),
                    int(d.Columns),
                    d,
                )
            )
        except Exception:
            continue
    if len(recs) < 3:
        return None
    keys = [(tuple(np.round(r[2], 3)), tuple(np.round(r[3], 4)), r[4], r[5]) for r in recs]
    modal = max(set(keys), key=keys.count)
    recs = [r for r, k in zip(recs, keys) if k == modal]
    if len(recs) < 3:
        return None
    iop = recs[0][2]
    u, v = iop[:3], iop[3:]
    n = np.cross(u, v)
    pos = np.array([r[1] @ n for r in recs])
    order = np.argsort(pos)
    recs = [recs[i] for i in order]
    pos = pos[order]
    d = recs[0][6]

    def num(name, default=float("nan")):
        try:
            return float(d.get(name))
        except Exception:
            return default

    return Series(
        uid=folder.name,
        files=[r[0] for r in recs],
        ipp=np.array([r[1] for r in recs]),
        u=u,
        v=v,
        n=n,
        ps=recs[0][3],
        rows=recs[0][4],
        cols=recs[0][5],
        number=int(num("SeriesNumber", -1)),
        description=str(d.get("SeriesDescription", "")),
        manufacturer=str(d.get("Manufacturer", "")),
        field=num("MagneticFieldStrength"),
        phase_dir=str(d.get("InPlanePhaseEncodingDirection", "")),
        bandwidth=num("PixelBandwidth"),
        gap=float(np.median(np.diff(pos))) if len(pos) > 1 else float("nan"),
    )


def read_pixels(path):
    d = pydicom.dcmread(path)
    return d.pixel_array.astype(np.float32)


def grid_shape(s):
    return int(np.floor((s.rows - 1) * s.ps[0])) + 1, int(np.floor((s.cols - 1) * s.ps[1])) + 1


def sample_on_grid(ref, ref_index, src, src_index, shape, offset=(0.0, 0.0)):
    img = read_pixels(src.files[src_index])
    img = gaussian_filter(img, (max(0.5 / src.ps[0], 0.01), max(0.5 / src.ps[1], 0.01)))
    origin = ref.ipp[ref_index] - src.ipp[src_index]
    ii, jj = np.meshgrid(
        np.arange(shape[0], dtype=np.float64) + offset[0],
        np.arange(shape[1], dtype=np.float64) + offset[1],
        indexing="ij",
    )
    col = (origin @ src.u + jj * (ref.u @ src.u) + ii * (ref.v @ src.u)) / src.ps[1]
    row = (origin @ src.v + jj * (ref.u @ src.v) + ii * (ref.v @ src.v)) / src.ps[0]
    out = map_coordinates(img, [row, col], order=1, mode="constant", cval=0.0)
    mask = (row >= 0) & (row <= src.rows - 1) & (col >= 0) & (col <= src.cols - 1)
    scale = np.percentile(out[mask], 99) if mask.any() else 1.0
    out = np.sqrt(np.clip(out / max(scale, 1e-6), 0, 2))
    return out.astype(np.float32), mask


def nearest_slice(ref, ref_index, src):
    centre = ref.ipp[ref_index] + ref.u * (ref.cols - 1) * ref.ps[1] / 2 + ref.v * (ref.rows - 1) * ref.ps[0] / 2
    dist = (centre - src.ipp) @ src.n
    k = int(np.argmin(np.abs(dist)))
    return k, float(dist[k])


def features(img, mask):
    sm = gaussian_filter(img, 1.0)
    gy, gx = np.gradient(sm)
    mag = np.hypot(gx, gy)
    m = binary_erosion(mask, iterations=4)
    if m.sum() < 100:
        return None
    eps = 0.5 * np.percentile(mag[m], 90) + 1e-6
    den = np.sqrt(mag**2 + eps**2)
    nx, ny = gx / den, gy / den
    chans = [nx * nx * m, ny * ny * m, np.sqrt(2.0) * nx * ny * m]
    return {"ngf": chans, "gm": mag * m, "mask": m.astype(np.float64)}


def _corr_factory(shape):
    size = (sfft.next_fast_len(shape[0] + SEARCH + 2), sfft.next_fast_len(shape[1] + SEARCH + 2))

    def f(x):
        return sfft.rfft2(x, size)

    def corr(fa, fb):
        c = sfft.irfft2(np.conj(fa) * fb, size)
        c = np.roll(c, (SEARCH, SEARCH), axis=(0, 1))
        return c[: 2 * SEARCH + 1, : 2 * SEARCH + 1]

    return f, corr


def _peak(score, valid):
    s = np.where(valid, score, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(s)), s.shape)
    peak = float(s[iy, ix])

    def sub(a, b, c):
        d = a - 2 * b + c
        if not np.isfinite(d) or d >= 0:
            return 0.0
        return float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))

    oy = sub(s[iy - 1, ix], s[iy, ix], s[iy + 1, ix]) if 0 < iy < 2 * SEARCH else 0.0
    ox = sub(s[iy, ix - 1], s[iy, ix], s[iy, ix + 1]) if 0 < ix < 2 * SEARCH else 0.0
    yy, xx = np.mgrid[: 2 * SEARCH + 1, : 2 * SEARCH + 1]
    far = valid & (np.hypot(yy - iy, xx - ix) > 6)
    second = float(s[far].max()) if far.any() else float("nan")
    vals = score[valid]
    z = float((peak - vals.mean()) / (vals.std() + 1e-12))
    edge = bool(iy in (0, 2 * SEARCH) or ix in (0, 2 * SEARCH))
    return iy - SEARCH + oy, ix - SEARCH + ox, peak, second, z, edge


def estimate(fa, fb):
    shape = fa["mask"].shape
    f, corr = _corr_factory(shape)
    fma, fmb = f(fa["mask"]), f(fb["mask"])
    count = corr(fma, fmb)
    valid = count > 0.5 * min(fa["mask"].sum(), fb["mask"].sum())
    if not valid.any():
        return None
    count = np.maximum(count, 1.0)
    ngf = sum(corr(f(a), f(b)) for a, b in zip(fa["ngf"], fb["ngf"])) / count
    a, b = fa["gm"], fb["gm"]
    fa_, fb_ = f(a), f(b)
    sa, sb = corr(fa_, fmb), corr(fma, fb_)
    num = corr(fa_, fb_) - sa * sb / count
    da = corr(f(a * a), fmb) - sa * sa / count
    db = corr(fma, f(b * b)) - sb * sb / count
    ncc = num / np.sqrt(np.maximum(da * db, 1e-12))
    out = {}
    for name, score in (("ngf", ngf), ("ncc", ncc)):
        sy, sx, peak, second, z, edge = _peak(score, valid)
        out[name] = {"sy": sy, "sx": sx, "peak": peak, "second": second, "z": z, "edge": edge}
        vals = score[valid]
        out[name]["map"] = np.where(valid, (score - vals.mean()) / (vals.std() + 1e-12), 0.0)
    out["valid"] = valid
    out["overlap"] = float(count[SEARCH, SEARCH] / fa["mask"].sum())
    return out


def pair_slice(a, ia, b, offset=(0.0, 0.0), force_b=None):
    shape = grid_shape(a)
    ib, dist = nearest_slice(a, ia, b) if force_b is None else (force_b, float((a.ipp[ia] - b.ipp[force_b]) @ b.n))
    img_a, mask_a = sample_on_grid(a, ia, a, ia, shape)
    img_b, mask_b = sample_on_grid(a, ia, b, ib, shape, offset)
    fa, fb = features(img_a, mask_a), features(img_b, mask_b)
    if fa is None or fb is None:
        return None
    est = estimate(fa, fb)
    if est is None:
        return None
    est["ib"] = ib
    est["slice_offset"] = dist
    return est, (img_a, mask_a, img_b, mask_b)


def combine(ests, name):
    valid = np.logical_and.reduce([e["valid"] for e in ests])
    if not valid.any():
        return None
    score = sum(e[name]["map"] for e in ests) / len(ests)
    sy, sx, peak, second, z, edge = _peak(score, valid)
    return {"sy": sy, "sx": sx, "peak": peak, "second": second, "edge": edge}
