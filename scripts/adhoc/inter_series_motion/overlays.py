import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pydantic_settings import BaseSettings, SettingsConfigDict
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))
import motion_lib as ml


class Settings(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)
    pairs: Path
    out_dir: Path
    top: int = 12
    data_root: Path = Path("data")
    min_mag: float = 0.0
    max_mag: float = 1e9
    prefix: str = "top"


def edges(img):
    sm = gaussian_filter(img, 1.0)
    gy, gx = np.gradient(sm)
    m = np.hypot(gx, gy)
    return np.clip(m / (np.percentile(m, 99.5) + 1e-6), 0, 1)


def composite(a, b):
    return np.stack([a, b, a], axis=-1)


def main():
    cfg = Settings()
    d = pl.read_parquet(cfg.pairs).filter(
        (pl.col("kind") == "pair") & (pl.col("normal_angle_deg") <= 3) & pl.col("ngf_c_mag").is_not_null()
    )
    d = d.filter((pl.col("ngf_c_mag") >= cfg.min_mag) & (pl.col("ngf_c_mag") <= cfg.max_mag))
    d = d.sort("ngf_c_mag", descending=True).unique("study", keep="first", maintain_order=True).head(cfg.top)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    root = cfg.data_root / "raw" / "train_series"
    for rank, r in enumerate(d.iter_rows(named=True)):
        a = ml.load_series(root / r["study"] / r["series_a"])
        b = ml.load_series(root / r["study"] / r["series_b"])
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        for k, frac in enumerate((0.35, 0.5, 0.65)):
            ia = int(round(frac * (len(a.files) - 1)))
            ib, dist = ml.nearest_slice(a, ia, b)
            shape = ml.grid_shape(a)
            img_a, _ = ml.sample_on_grid(a, ia, a, ia, shape)
            img_b, _ = ml.sample_on_grid(a, ia, b, ib, shape)
            img_c, _ = ml.sample_on_grid(a, ia, b, ib, shape, (r["ngf_c_sy"], r["ngf_c_sx"]))
            ea, eb, ec = edges(img_a), edges(img_b), edges(img_c)
            panels = [
                (img_a, f"A #{r['number_a']} {r['desc_a']} slice {ia}", "gray"),
                (img_b, f"B #{r['number_b']} {r['desc_b']} slice {ib} (offset {dist:.1f} mm)", "gray"),
                (composite(ea, eb), "edges by header: A magenta, B green", None),
                (composite(ea, ec), f"edges after shift sx={r['ngf_c_sx']:.1f} sy={r['ngf_c_sy']:.1f} mm", None),
            ]
            for ax, (im, title, cmap) in zip(axes[k], panels):
                ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
        fig.suptitle(
            f"{r['study']} {r['plane']} shift {r['ngf_c_mag']:.1f} mm (1 px = 1 mm) dev {r['ngf_c_dev']:.2f} ncc ({r['ncc_c_sx']:.1f},{r['ncc_c_sy']:.1f})",
            fontsize=11,
        )
        fig.tight_layout()
        name = f"{cfg.prefix}{rank:02d}_{r['plane'].lower()}_{r['ngf_c_mag']:.1f}mm_{r['study'][-8:]}.png"
        fig.savefig(cfg.out_dir / name, dpi=70)
        plt.close(fig)
        print(name, flush=True)


if __name__ == "__main__":
    main()
