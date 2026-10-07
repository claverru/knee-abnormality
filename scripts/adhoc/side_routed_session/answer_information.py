import json
import math
from pathlib import Path

import numpy as np
import polars as pl

manifest = json.loads(Path("configs/labels/jevk5-report-attribute-targets-v4.0.json").read_text())
table = pl.read_parquet(manifest["targets"]["path"])
questions = manifest["questions"]
print(len(table), "reports;", len(table.columns), "columns; first columns", table.columns[:6])
print("question entry sample:", json.dumps(questions[0] if isinstance(questions, list) else list(questions.items())[0])[:400])
names = [q["id"] for q in questions]
rows = []
for name in names:
    prefix = f"attribute_{name}__"
    columns = [c for c in table.columns if c.startswith(prefix)]
    if not columns:
        rows.append((name, 0, "", 0.0, 0.0, 0.0, 0))
        continue
    values = table.select(columns).to_numpy().astype(float)
    values = values / np.clip(values.sum(1, keepdims=True), 1e-9, None)
    mean = values.mean(0)
    top = int(mean.argmax())
    winner = values.argmax(1)
    share = float((winner == top).mean())
    entropy = float(-(mean * np.log(np.clip(mean, 1e-12, None))).sum() / math.log(len(columns))) if len(columns) > 1 else 0.0
    other = int((winner != top).sum())
    rows.append((name, len(columns), columns[top][len(prefix):], float(mean[top]), share, entropy, other))
print(f"{'question':34}{'options':>8}  {'commonest answer':28}{'its mass':>9}{'reports':>9}{'others':>8}{'entropy':>9}")
for row in sorted(rows, key=lambda r: -r[4]):
    print(f"{row[0]:34}{row[1]:8d}  {row[2][:27]:28}{row[3]:9.3f}{row[4]:9.1%}{row[6]:8d}{row[5]:9.2f}")
