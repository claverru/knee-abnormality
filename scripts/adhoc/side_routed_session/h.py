import polars as pl, numpy as np
f = pl.read_parquet("outputs/analysis/geometry-metadata-20260912/headers.parquet")
f = f.filter(pl.col("ImageOrientationPatient").list.len() == 6)
s = f.group_by("StudyInstanceUID","SeriesInstanceUID","Anatomical_Plane").agg(pl.col("ImageOrientationPatient").first(), pl.len().alias("n"))
o = np.array(s["ImageOrientationPatient"].to_list())
n = np.cross(o[:, :3], o[:, 3:])
axis = {"Axial": 2, "Coronal": 1, "Sagittal": 0}
ax = np.array([axis[p] for p in s["Anatomical_Plane"]])
sign = np.sign(n[np.arange(len(n)), ax])
dom = np.abs(n).argmax(1) == ax
s = s.with_columns(pl.Series("sign", sign), pl.Series("dom", dom),
    pl.Series("row_dom", [ "xyz"[i]+("+" if v>0 else "-") for i,v in zip(np.abs(o[:,:3]).argmax(1), o[np.arange(len(o)), np.abs(o[:,:3]).argmax(1)])]),
    pl.Series("col_dom", [ "xyz"[i]+("+" if v>0 else "-") for i,v in zip(np.abs(o[:,3:]).argmax(1), o[np.arange(len(o)), 3+np.abs(o[:,3:]).argmax(1)])]))
print(s.height, "series", s["StudyInstanceUID"].n_unique(), "studies")
print(s.group_by("Anatomical_Plane","sign","dom").len().sort("Anatomical_Plane","sign","dom"))
print(s.group_by("Anatomical_Plane","row_dom","col_dom","sign").len().sort("Anatomical_Plane","len", descending=[False,True]))
