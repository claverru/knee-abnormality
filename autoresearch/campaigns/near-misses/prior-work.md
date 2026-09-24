# Prior work

Where each seed comes from, the evidence behind its order, and the directions already closed.

- **Selection review, 2026-09-24:** `next_autoresearch.md` at the repository root lists the 15
  seeds with expected gains. Ideas 1–5 carry measured evidence; the rest are guesses.
- **Seed noise:** re-seeding the adopted knee-geometry recipe (`adopted-new-seed` against
  `meniscus-half-knee-mil`) moved macro AUC by −0.0063. The largest per-label moves were
  Contusion −0.0205, PF OA −0.0154, Fracture −0.0137 and MCL −0.0129. Treat single-label moves of
  that size in those labels as noise.
- **Knee-geometry campaign:** [findings](../knee-geometry/findings.md) and
  [idea registry](../knee-geometry/ideas.yaml). Seeds 1–2, 6, 14 and 15 come from it. Against the
  pre-adoption control `k05-clean`, 16 of its 17 candidates were positive; they lost mainly to
  the adopted run's lucky seed.
- **Noisy-labels campaign:** [findings](../noisy-labels/findings.md). Seed 5 comes from it; its
  per-label analysis is in `docs/ongoing/early-learning-regularization.md`.
- **RSNA lumbar solution** (claverru/RSNA-lumbar): side-free per-side samples, heads shared
  across sides and levels, and side-normalized coordinates with slice normals. Seeds 4, 8 and 13
  adapt it.
- **Closed directions:**
  - study-level incomplete-geometry fallbacks;
  - changes to which tokens the half-knee loss sees (axis consensus, fractional footprint,
    detector ranking);
  - the oblique line;
  - sampler shape and width;
  - forcing OA evidence apart without shared context;
  - report-derived supervision (JevK5 line, separate).
