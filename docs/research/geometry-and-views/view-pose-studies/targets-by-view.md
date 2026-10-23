# Study 4: which targets are seen in which plane-contrast

Two sources independent of our model: radiology literature, and visual inspection of wide,
uncropped contact sheets of 102 training studies (8 positives per target, 6 negatives; fold 1
excluded), read by agents with the report as a guide. **Not a radiologist's reading.** Calls are
reliable for conspicuous findings (fluid, cysts, marrow oedema, gross fractures) and weak for
fine ones: medial-meniscus calls were at chance against negatives and PF OA was never clear.

**Target by view** (inspection counts clear/subtle/not/cannot-tell are in
`synthesis-target-view.csv`):

| Target | Sag fluid | Sag T1/PD | Cor fluid | Cor T1/PD | Ax fluid | Ax T1/PD |
|---|---|---|---|---|---|---|
| ACL | **primary** | useful | useful | little | useful | little |
| MCL | little | none | **primary** | useful | useful | little |
| Medial Meniscus | **primary** | **primary** | useful | useful | little | none |
| Lateral Meniscus | **primary** | useful | useful | little | little | none |
| Medial OA | useful | useful | **primary** | useful | little | none |
| Lateral OA | **primary** | little | **primary** | useful | little | none |
| PF OA | useful | little | none | none | **primary** | little |
| Effusion | **primary** | useful | useful | little | **primary** | little |
| Synovitis | **primary** | little | useful | none | **primary** | none |
| Baker's | **primary** | useful | useful | little | **primary** | little |
| Contusion | **primary** | little | **primary** | useful | **primary** | little |
| Fracture | useful | useful | **primary** | **primary** | useful | useful |

Medial Meniscus rests on literature alone. Where the sources disagree: axial fluid showed ACL as
clearly as sagittal; coronal fluid showed the largest effusion area and the most conspicuous
synovitis, which the literature does not use; literature rates short-TE coronal T1/PD useful for
the lateral meniscus and inspection saw nothing there.

**What each view must contain** (distances from the joint line and plateau centre; landmarks
from 400 mask knees: kneecap top 55 mm above the joint line, 95th percentile 66; kneecap front
56 mm ahead of centre; posterior condyles 23 mm behind; bone width 73 mm):

| View | Wide need | Fine need | Conflict |
|---|---|---|---|
| Sag fluid | pouch to 80 to 125 mm above the joint line; cyst from 45 mm above to 35 mm below, 30 mm behind the condyles; bone 65 mm above to 50 to 60 mm below | meniscus and cartilage within ±10 mm of the joint line, 45 mm front to back; ACL on 1 to 2 slices within 7 mm of mid; 1 mm tear line, 2 to 3 mm cartilage | **yes:** about 180 mm of height against a 60 to 90 mm box |
| Sag T1/PD | adds little | 1 mm line at the joint | no |
| Cor fluid | skin to skin for both collaterals; MCL from 40 to 50 mm above to 60 to 70 mm below the joint line | joint space ±25 mm; 2 to 3 mm ligament and cartilage | less: about 110 mm high, centred slightly above the joint line, holds both; pouch and cyst are in the front and back slices, not in a larger field |
| Cor T1/PD | tibia to 60 mm below the joint line | 1 to 2 mm fracture line, 3 to 5 mm osteophytes | no |
| Ax fluid | whole limb cross-section for cyst and pouch, 130 to 150 mm | 70 × 40 mm box at the kneecap, 3 mm cartilage band, in 7 to 12 slices | **yes** |
| Ax T1/PD | – | fracture line, fat-fluid level | serves no target alone |

**Views.**

- Irreplaceable: sagittal fluid (ACL, menisci, pouch height), coronal fluid (MCL, the
  tibiofemoral joint), axial fluid (kneecap cartilage, the Baker's neck).
- Near-redundant with their fluid partner: sagittal and coronal T1/PD. They add the fracture
  line and osteophytes and, by literature only, short-TE meniscal signal.
- Axial T1/PD: never above "little"; present in 19% of training studies; the least consistent
  content (3D gradient echoes, a two-leg volume, a kneecap cartilage series). No target depends
  on it.
- **The "T1/PD" slot is not one contrast.** Sagittal held T2 in 45 of 102, T1 in 31, PD in 26;
  coronal T1 in 88; axial T1 in 36 of 46. Most clear fluid or cyst calls in sagittal T1/PD came
  from T2 or gradient-echo series.

**Scanned coverage** (mask-based over 120 to 433 series per view, or inspection of 102 studies):

| Need | Sagittal | Coronal | Axial |
|---|---|---|---|
| ≥ 30 mm above the kneecap top | 96 to 98% | 93 to 95% | 37% fluid, 22% T1/PD |
| ≥ 40 mm above the kneecap top | 78 to 83% | 73 to 74% | 16% / 6% |
| ≥ 30 mm behind the posterior condyles | 100% | 17 to 19% | 98 to 99% |
| ≥ 60 mm below the joint line (MCL's tibial end) | 70 to 79% | 62 to 68% | 3 to 5% |
| 10 mm beyond the bone edge, medial or lateral | 67 to 73% | 100% | 99 to 100% |
| Front of the kneecap inside | 100% | 62 to 71% | 99% |

The pouch is the most-cut region: median 45 mm of sagittal field above the kneecap top against
pouches reaching 25 to 70 mm above it; whole in the field in 46 of 102 studies. Coronal stacks
ended inside the Baker's cyst in 3 of 8. Axial stacks start a median 18 to 24 mm above the
kneecap.

**Per target, where and how big** (inspection, n 8 each): effusion bulk is off the midline in 5
of 8, 3 to 14 mm thick; Baker's 45 to 110 mm long, 5 to 40 mm medial of mid; contusions 20 to
35 mm patches, at the condyle periphery in 5 of 8, so on the 1 to 4 outermost sagittal slices;
MCL injury in the femoral half, 40 to 45 mm medial of the notch; ACL in a box 0 to 30 mm above
the joint line; fracture line seen in 2 of 8, to 55 mm below the joint line.

**Limits.** Sheets show 12 of about 30 slices per series; sizes are ±5 mm; tear lines,
cartilage depth and non-displaced fracture lines were not resolved, so those definitions were
not tested; some figures in the literature summary came from search summaries, not opened pages.
Sources are linked in the agent's report; ESSR 2024 protocol, Radiology Assistant, Radiopaedia,
MOAKS.
