# Loading speedup

Everything about accelerating DICOM loading over Kaggle's NFS mount: where the cold-start time
actually goes, what is left to try, what is closed, and how to measure any of it. Started 2026-09-17
with the user; four identity-changing changes and three retrain-free ones shipped 2026-09-18 and
the rest is open.

**Why it matters:** the loader is 57–59% of prediction time on the two T4s even after the shipped
changes, so loading and not the GPUs sets inference throughput, and inference runtime counts toward
the efficiency prize. Active status lives in [ongoing work](../../ongoing/loading-speedup.md).

| Read when | Document |
|---|---|
| Reading the cold-start profile, checking what the shipped changes bought, finding what a stage really contains, or before trusting any number the profiler reports | [Where the cold-start time goes](attribution.md) |
| Picking the next change, checking whether one needs a retrain or a validation run, or following the proposed experiment order | [Open candidates](candidates.md) |
| Before proposing anything: the measured regressions, the approaches ruled out by arithmetic or by Kaggle's environment, and why | [Closed and rejected](closed.md) |
| Designing a probe or a benchmark for a loading change, or deciding whether a local result means anything | [How to measure](measuring.md) |
| Checking what past RSNA DICOM solutions did, and how published medical-imaging dataloader profiles compare | [Prior art](prior-art.md) |
