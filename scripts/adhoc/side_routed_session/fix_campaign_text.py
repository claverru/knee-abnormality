from pathlib import Path

root = Path("autoresearch/campaigns/routing-and-shifts")
p = root / "campaign.yaml"
s = p.read_text()
old_start = s.index("    deterministic. Topic 2: cutting a crop inside the model is not in this worktree;")
old_end = s.index("Per-epoch diagnostics go through the module's epoch accumulators")
new = """    deterministic. Reusing earlier code is allowed and expected; three sources, all readable
    from the candidate. (a) Runs of this line, for the routing seeds: each run directory under
    outputs/training/ has its full source in source_snapshot.zip (unzip it into a temporary
    directory outside the candidate); the seeds name the run to read. (b) Earlier candidates of
    this campaign: outputs/autoresearch/routing-and-shifts/experiments/<experiment>/ holds
    source/ and changes.diff; a topic 2 seed that follows another reuses its port, cutter and
    locator instead of rebuilding them. (c) The learned-view work of 2026-10-02 and 03, for
    topic 2: the branch view-optimisation (commit b290da6; read files with `git show
    view-optimisation:PATH`, never check it out), whose src/knee_abnormality/modeling/views.py
    has the in-model cutter (sample_windows, normalise_windows, ViewWindows with a clamped
    two-axis shift per view), and the experiments under
    outputs/autoresearch/learnable-views/experiments/, each with source/ and changes.diff:
    hand-sizes-learned-shifts-20261003T183532Z-8d11d7a5 (sizes pinned, shifts learned, the
    closest to this campaign's per-series seed), knee-anchored-learned-offsets-20261002T194645Z-d4e331e1
    and hand-views-routed-shift-learning-20261003T212020Z-0863da22 (built and reviewed, never
    run). That code was written on an older recipe: take the grid construction, the shift
    parameter, the normalisation after the cut and the logging, and adapt them to this
    worktree; do not take its single global canvas at its own spacing, its bicubic reads, its
    learned width, label bias or slice budget, and do not copy files whole. Topic 2 facts
    about the incumbent: prepare_images in data/preprocessing/device.py resamples each slice's
    view window to 224 px with an index-based plan, normalises each series between two
    quantiles of its windows' valid pixels and rounds to uint8; the view window's position
    comes from view_position in data/preprocessing/spatial.py; the affine augmentation and the
    reflection run in the model afterwards. The first topic 2 seed
    (learned-shift-per-series-type) states the port in five numbered steps; every other topic
    2 seed keeps steps 1 to 4 exactly. Required of every topic 2 candidate, with the numbers
    in its report: with every shift at zero, the crops entering the augmentation equal the
    incumbent's prepared images on one batch, in training and in evaluation mode, largest
    absolute difference at most 1 grey level of 255; the per-study seeds also show that a
    mirrored pass and a plain pass give mirror-image crops. Never decode more slices per study
    than the incumbent, and do not change widths, image size or slice counts. Each topic 2 run
    logs once per epoch, for training and validation, the mean and the standard deviation
    across knees of every learned shift in millimetres, and reports its prediction time.
    Training time must stay within 0.75 to 1.3 times the reference's; do not reduce epochs,
    slices or studies. """
p.write_text(s[:old_start] + new + s[old_end:])

p = root / "prior-work.md"
s = p.read_text()
old = "- **Shifts per series type looked random:** means within ±2.5 mm of the start, 3 to 7 mm apart\n  between runs.\n"
assert s.count(old) == 1
s = s.replace(
    old,
    old
    + "- **Sizes pinned, shifts learned** (`hand-sizes-learned-shifts`, 8 mm shift jitter): ×4 0.932905,\n"
    "  −0.0004 against its reference, 80% interval [−0.0014, +0.0005], ×1 0.9286.\n",
)
s = s.rstrip("\n") + """

Reusable code for topic 2 (older recipe; adapt, do not copy whole): the branch `view-optimisation`
(commit `b290da6`), whose `src/knee_abnormality/modeling/views.py` has the in-model cutter; and,
under `outputs/autoresearch/learnable-views/experiments/`, each with `source/` and `changes.diff`:
`hand-sizes-learned-shifts-20261003T183532Z-8d11d7a5`,
`knee-anchored-learned-offsets-20261002T194645Z-d4e331e1` and
`hand-views-routed-shift-learning-20261003T212020Z-0863da22` (built and reviewed, never run).
Runs of this line keep their full source in `source_snapshot.zip` in the run directory.
"""
p.write_text(s)
print("updated")
