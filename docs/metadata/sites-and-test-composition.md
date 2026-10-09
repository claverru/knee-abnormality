# Sites and test composition

**The hidden test probably contains sites absent from training, but no official statement says so.**
The organizers list 19 contributing institutions and, separately, 3 "additional contributing
sites" (Mongolia, Australia, Italy). The training reports contain no Italian or Mongolian text,
and they cover 9 languages where RSNA describes "a dozen". Unseen vendors remain unknown. The
local split `all_5fold_scanner_v2` is stratified, not grouped, by site, so every local score
measures seen-site performance only. Checked 2026-09-29; recheck if the hosts answer
[discussion 734681](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734681).

## Official statements

- **Data page:** studies come from "a diverse international mix of imaging sites" spanning "a
  wide range of scanners, protocols, and populations". The test has about 1,300 studies, with
  the public leaderboard on 30% of them. Label prevalence "is not guaranteed to be the same"
  across training, public and private sets. Nothing is said about sites or scanners.
- **Acknowledgements page (data contributors):**
  - Belgium: AZ Delta.
  - Argentina: Centro Rossi. Chile: Clínica Alemana.
  - Thailand: Chiang Mai University, Khon Kaen University.
  - Taiwan: China Medical University Hospital.
  - Morocco: CHU Mohamed VI.
  - Türkiye: Hacettepe University, Koç University, Samsun Training and Research Hospital.
  - Malta: Mater Dei. Canada: McGill, Unity Health Toronto. USA: Thomas Jefferson.
  - Bulgaria: Sofia University. Croatia: University Hospital Dubrava. Greece: University Hospital
    of Heraklion. Germany: Würzburg. Bosnia and Herzegovina: University of Sarajevo.

  That is 19 institutions. A separate list thanks "additional contributing sites": Intermed
  Hospital (Ulaanbaatar, Mongolia), Liverpool Hospital (NSW, Australia) and Salus Vigevano
  (Italy).
- **RSNA challenge page:** "more than 5,000 knee MRI exams from 19 sites worldwide" with reports
  in "a dozen different languages". The RSNA news release says 16 sites; the discrepancy is
  unexplained.
- **Forum:** discussion 734681 (2026-08-12) asked whether whole sites are held out of the
  private set. As of 2026-09-29 no host has replied, and none of the 23 host posts across the
  forum's 140 threads addresses sites, scanners or test composition.

Pages were read through Kaggle's page and forum APIs
(`competitions.PageService/ListPages`, `discussions.DiscussionsService/GetForumTopicById`),
as in the [research provenance](../research/provenance.md).

## Training languages and inferred sites

[`infer_training_sites.py`](../../scripts/adhoc/infer_training_sites.py) detects each training
report's language (`langdetect`, seed 0, probability ≥ 0.8), runs script and vocabulary checks,
and tabulates language by vendor, field strength and model. Artifacts are in
`outputs/analysis/t/training-sites-20260929/`.

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= uv run --locked --with langdetect python -m scripts.adhoc.infer_training_sites --output-dir outputs/analysis/training-sites-new
```

| Language | Studies | Scanners | Inferred institution | Confidence |
|---|---:|---|---|---|
| Bulgarian | 220 | Philips Ingenia 3T | Sofia University | high |
| German | 262 | Siemens Avanto fit 1.5T | Würzburg | high |
| Greek | 321 | Siemens SonataVision 1.5T, Prisma 3T | Heraklion | high |
| Dutch | 152 | Siemens Vida 3T, Sola 1.5T | AZ Delta | high |
| French | 81 | Siemens Aera 1.5T | CHU Mohamed VI, Marrakech | moderate |
| Croatian | 406 | Siemens Vida 3T; Philips Ingenia 1.5T; Siemens Aera/Altea 1.5T | Dubrava and Sarajevo (Bosnian is detected as Croatian); split unknown | moderate |
| Turkish | 546 | Siemens Aera 1.5T; GE Explorer 1.5T; GE Architect 3T; Philips Achieva 1.5T | Hacettepe, Koç, Samsun; split unknown | moderate |
| Spanish | 680 | Philips 1.5T; Siemens Skyra/Lumina 3T; GE Artist 1.5T; Siemens Sempra; Philips 3T | Centro Rossi and Clínica Alemana; split unknown | moderate |
| English | 1,735 | all vendors, including every Toshiba, Canon, Fujifilm and Hitachi study | Thai, Taiwanese, Maltese, Canadian and US sites; 3 reports contain Thai script (Philips) | low per site |

Four reports have uncertain language.

- **Absent languages:** no report contains Italian knee vocabulary (`ginocchio`, `versamento
  articolare`, `legamento crociato`). No report contains Mongolian letters (ө, ү); all 220
  Cyrillic reports are the Bulgarian cluster. Salus Vigevano and Intermed Hospital therefore
  left no Italian- or Mongolian-language study in training. Liverpool Hospital would report in
  English and cannot be separated.
- **Test sample:** the three example test studies are a Siemens MAGNETOM Avanto fit, a Toshiba
  Vantage and a GE SIGNA Artist, all models present in training. Three studies say nothing about
  the rest.

## What it implies

- **Most likely reading, moderate evidence:** the three additional sites contributed studies
  that appear only in the test. Alternatives: their reports were written in English, their
  studies were excluded, or the list separates them for another reason. Whether any of them
  falls in the public or the private part is unknown. If only in the private part, the public
  leaderboard cannot show how the model handles them.
- **Unseen vendors:** unknown. The additional sites' scanners are not published.
- **Local validation** keeps every training site in every fold, so it cannot measure new-site
  generalization. The five cleanly inferred sites (Bulgarian, German, Greek, Dutch and French
  clusters, 1,036 studies, 224 in fold 0 of `all_5fold_scanner_v2`) are the natural held-out set
  for a leave-sites-out measurement
  ([sites-out diagnostic](../validation/results/2026-09-29-sites-out-diagnostic.md): an unseen site costs
  about 0.009 macro AUC).
- **Scanner memorization in metadata:** a public thread reports metadata-only macro AUC of 0.652
  under random folds and 0.598 under scanner-grouped folds
  ([discussion 734004](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734004)).
  This is the metadata alone, not image models.
