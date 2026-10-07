from pathlib import Path

p = Path("docs/ongoing/side-routed-heads.md")
s = p.read_text()


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


sub(
    "the design below is the [fifth build](#fifth-build-2026-10-05),\nnot yet run.** ",
    "the design below is the [fifth build](#fifth-build-2026-10-05),\nwhose [run](#fifth-run) reached 0.924476 (S114 submitted).** ",
)
sub(
    "scripts in the session scratchpad under\n`motion/`, to be moved to `scripts/adhoc/`)",
    "scripts in `scripts/adhoc/inter_series_motion/`)",
)
sub(
    "### Octant sketches",
    """### Fifth run

`outputs/training/20261005-205251-side-routed-run5`, one run: **×4 0.924476** (reference
0.924049, fourth run 0.923952); ×1 0.915823. Submitted as S114 (×4, τ 0.9, no reflection TTA;
bundle `outputs/submission/exports/side-routed-run5-f1-x4-tau09`).

| Target | Fifth − reference | Fourth − reference | Siblings − reference, mean | Fifth below how many of 12 |
|---|---:|---:|---:|---:|
| ACL | −0.0041 | −0.0034 | −0.0016 | 11 |
| MCL | +0.0012 | −0.0031 | +0.0005 | 4 |
| Medial Meniscus | −0.0012 | −0.0002 | −0.0002 | 11 |
| Lateral Meniscus | +0.0015 | −0.0001 | −0.0015 | 3 |
| Medial OA | +0.0013 | +0.0030 | +0.0005 | 3 |
| Lateral OA | +0.0003 | +0.0022 | −0.0024 | 3 |
| PF OA | +0.0071 | +0.0082 | −0.0007 | 0 |
| Effusion | +0.0031 | +0.0012 | +0.0007 | 0 |
| Synovitis | −0.0027 | −0.0054 | −0.0041 | 2 |
| Baker's | −0.0035 | −0.0057 | +0.0001 | 12 |
| Contusion | +0.0021 | +0.0036 | +0.0025 | 8 |
| Fracture | +0.0000 | −0.0015 | +0.0072 | 11 |

- **First run of the line above the reference** (+0.0004), with the own-channel targets summing
  to −0.0039 against −0.0143 in the fourth run: MCL, Synovitis, Effusion, Baker's and Fracture
  all moved up with the token position code. Baker's is still below all 12 siblings, Fracture
  and ACL below 11.
- **The routed five sum to +0.0090** (fourth run +0.0131). Medial OA and Lateral OA, now on the
  back of their side only, gave back 0.0017 and 0.0019; PF OA stays above all siblings.
- **The frame barely turned:** nod −5.1° (the front cut leans 5° towards front-and-up, against
  the 25° to 45° of the sketches), roll −2.4°, turn −1.0°. The front width grew from 5.0 to
  5.7 mm; the medial width ended at 4.5 mm.
- The side head reached 99.8% on certain validation studies.
- At ×1 the run led the reference through epoch 5 and ended 0.0003 behind.

### Octant sketches""",
)
p.write_text(s)

p = Path("scripts/adhoc/INDEX.md")
s = p.read_text()
row = "| [census_knee_rotation.py](census_knee_rotation.py) |"
assert s.count(row) == 1
a = s.index(row)
b = s.index("\n", a) + 1
s = (
    s[:b]
    + "| [inter_series_motion/](inter_series_motion/) | How far does the anatomy shift between two series of one study (same-plane pairs and crossing planes, normalised-gradient-field correlation over ±20 mm, with synthetic-shift and triangle-closure validation)? Run `run_pairs.py` then `analyze.py`; `cross_planes.py`, `analyze_cross.py`, `union.py` and `overlays.py` for the crossing planes and the drawings. | [Patient motion between series](../../docs/ongoing/side-routed-heads.md#patient-motion-between-series) |\n"
    + s[b:]
)
p.write_text(s)
