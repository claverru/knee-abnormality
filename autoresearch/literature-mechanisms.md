# Mechanisms from the autoresearch literature

Literature review of 2024–2026 automated ML research systems and the statistics of validation reuse, early stopping and bandit allocation, checked against our campaign records on 2026-09-28. It extends the [ScientistTwo lessons](scientisttwo-lessons.md).

## Summary: stop campaigns sooner, verify mechanisms in code

The literature's most useful offer for our campaign system is not a smarter search policy or a new statistical test. It is two cheap controls. The first is **an automatic switch away from long campaigns**: every system that plots returns shows front-loaded gains, and long searches against one validation signal lose held-out performance. The second is **code-checked evidence that each mechanism actually ran**: "silent component disabling" is one of the commonest failures in automated research, and deterministic gates catch it far better than LLM judgment. Replayed on our own 157 decided runs, **leaving a campaign after six consecutive researcher runs without an adoption would have moved 40 runs (25%) to fresh topics and lost no adoption and no run that cleared the 0.001 gain**. The same cap also bounds how many comparisons each campaign spends on one fixed fold, which is the cheapest false-adoption control available under our one-run rule. The second priority is a code-enforced activation signature checked after epoch 0, which turns our known "mechanism never activated" failure into an early, cheap `invalid` verdict instead of a ten-epoch reject. Third, a zero-GPU check on four repeat pairs found training noise (0.0006–0.0010) below the bootstrap's 0.0015, so the adoption rule stays as it is. One result cuts against the literature: **our follow-up ideas beat fresh families** (median Δ −0.0010 vs −0.0029, Mann-Whitney p = 0.0008, n = 46 vs 43), so the widely recommended fresh-family quota should not be adopted here. Most tree-search, multi-fidelity and multiple-testing machinery fails our regime of one expensive, small-effect run per idea. The rejected list at the end explains why.

## Eight mechanisms ranked by value per cost for us

The ranking weighs expected value (GPU saved, false or missed adoptions avoided) against implementation cost and fit with the fixed constraints: one run per idea at the incumbent seed, no confirmation or control reruns, and 0.001 macro AUC mattering. "E" marks evidence-backed proposals and "G" guesses; the confidence applies to the transfer to our system, not to the source finding. The stop-rule and follow-up numbers come from a replay of `outputs/autoresearch` campaign records built on [`analyze_campaign_screening.py`](../scripts/adhoc/analyze_campaign_screening.py). It covered 157 decided runs in 13 campaigns with 5 adoptions, rerun on 2026-09-28 after the plane-specialization idea files landed. Every run used `all_5fold_scanner_v1` with `combined_v1` labels; the evaluation moved to `all_5fold_scanner_v2` and `rebuild_state_v2` the same day, so absolute thresholds (the epoch-5 abort, noise yardsticks) are untested on the new labels.

| Rank | Mechanism | Group | Label | Rule | Cost | Decision it changes |
|---|---|---|---|---|---|---|
| 1 | Autonomous topic rotation | Stopping and allocation | E, moderate | After 6 consecutive researcher runs without `provisional_accept`, the researcher drafts a new topic inside a user-sealed envelope; code validates it; it starts from the current incumbent; seeds neither count nor reset; adoption resets | Runner work | ~25% of runs move to fresh topics, GPU never idle, no user in the loop |
| 2 | Activation signature | Integrity | E direction, high fit; frequency unknown | Developer declares a logged metric and a predicate; code checks it after the declared epoch (default 0); failure means `invalid`, abort, repair | Logging lines, one gate | Never-activated runs cost 1 epoch, not 10, and stop counting as rejects |
| 3 | Bootstrap calibration from repeat pairs (measured, closed) | Evaluation noise | E, weak (4 pairs) | Keep the rule: training noise on Δ ≈ 0.0006–0.0010 against bootstrap SE 0.0015, so the 80% rule runs at about 12–15% false positives; re-measure once pairs build up on `rebuild_state_v2` | Done, zero GPU | None now |
| 4 | Epoch-5 loser abort on the same hook | Stopping and allocation | E (own data), weak | Abort when epoch-5 Δ < −0.003; verdict `reject` | Near zero once #2 exists | ~6.6% fewer epochs |
| 5 | Artifact-grounded interpretation, typed outcomes | Memory and integrity | E, medium | Orchestrator cites log lines and values; reporter fills layer/degree/action fields | Prompt and schema | Family verdict inputs |
| 6 | Distributional prompting for new families | Exploration | G, low | Researcher emits new-family ideas across complexity tiers with verbalised probabilities, framed against the tried list | Prompt | New-family quality only |
| 7 | Evidence-preserving log compaction | LLM cost | G, low-medium | Deterministic reducer with hash and exact-quote checks, raw-log fallback | Small code | Tokens per orchestrator call |
| 8 | Runtime-patch diff guard | Integrity | E (class), low need | Flag diffs that patch `torch`, metric functions, `__eq__`, seeds, epochs or samplers | Grep | Review attention |

## Campaign length is the cheapest lever for GPU and false adoptions

### 1. A six-run researcher streak ends the campaign

**Sources and evidence.** The published agent systems agree on the shape of returns: they are front-loaded. ScientistTwo reports that relative gains "are largest during the initial improvement rounds and steadily diminish", and that most top ideas surface early ([arXiv 2609.19644](https://arxiv.org/html/2609.19644)). In AIRA-dojo, MCTS began to overfit the validation signal after about 50 hours, and **selecting the final node by validation instead of test cost 9–16 medal points** ([arXiv 2507.02554](https://arxiv.org/html/2507.02554)). Its follow-up describes a "paradoxical trend: as agents spend more time searching, performance on held-out test sets often drops" ([arXiv 2603.26499](https://arxiv.org/html/2603.26499v2)). MLGym finds agents mostly win by tuning hyperparameters, and those wins run out ([arXiv 2502.14499](https://arxiv.org/abs/2502.14499)).

The HPO overtuning study adds that single-holdout AUC on small data with neural nets is the setting most prone to severe overtuning, which hits about 10% of runs ([arXiv 2506.19540](https://arxiv.org/html/2506.19540)). Every agent system uses a hand-set cap rather than an adaptive rule. ML-Master ends a branch after **3 consecutive improvements below 0.001** ([arXiv 2506.16499](https://arxiv.org/html/2506.16499)); AutoSOTA forces a structural "leap" after 3 parameter-only iterations ([arXiv 2604.05550](https://arxiv.org/html/2604.05550)); AI Scientist v2 fixes 12–21 nodes per stage ([arXiv 2504.08066](https://arxiv.org/html/2504.08066)). Among statistical frameworks, Makarova et al. stop BO once the regret bound falls below the validation noise ([arXiv 2104.08166](https://arxiv.org/abs/2104.08166)). Rotting-bandit theory models arms whose reward decays with pulls ([arXiv 1702.07274](https://arxiv.org/abs/1702.07274)). Both favour switching to a fresh arm over proving the current one dead.

**Our data.** Our record matches: all 5 adoptions came at campaign positions ≤9, and 0 of the 65 later runs adopted. A pure significance test for "this campaign is dead" is hopeless at our ~5% base rate. Holding a 20% false-stop risk on a campaign still at the early 5.4% rate needs N = ln 0.2 / ln 0.946 ≈ **29 consecutive failures** [calc]. The literature's answer is to compare the next run here against the next run in a fresh campaign, not to test.

Replayed against the records, the streak rule behaves as follows:

| Streak N | Runs moved of 157 | Adoptions lost | Runs with Δ ≥ 0.001 lost |
|---|---|---|---|
| 4 | 51 | 0 | 1 |
| 5 | 45 | 0 | 0 |
| 6 | 40 | 0 | 0 |
| 8 | 30 | 0 | 0 |

A plain position cap is worse. Capping every campaign at 12 runs moves 44 runs but drops 3 of the runs that cleared 0.001, all late augmentation follow-ups.

**Rule.** After **N = 6 consecutive researcher runs without `provisional_accept`**, the loop rotates to a new topic without the user (user decision 2026-09-28: autoresearch must keep going while the user is away). The researcher drafts the next topic (contract, seed ideas, a narrow policy) inside an envelope the user seals once in advance; code checks the policy is a subset of the envelope, validates it and refuses closed topics, and the new topic starts from the current incumbent run as its reference, so adoptions chain and no control retrain is needed. The current topic keeps training until the new topic's first candidate is ready, so the GPU never idles. Within-campaign pivots do not substitute: after a six-run streak, new families went 0 of 19 and follow-ups 0 of 21 at Δ ≥ 0.001.

Rotation pauses a topic; it does not close it. A restless-bandit view says a paused topic can recover after the incumbent changes elsewhere. That part is a guess, with low confidence.

**Strength and cost.** The evidence is moderate that returns diminish, and weak for the exact N: the rule is fitted in-sample on 5 adoptions, the incumbent strengthens after each adoption, and the user ended campaigns at different lengths. The cost is a counter in the supervisor.

**What it changes.** Topics end about 25% sooner, with proportionally fewer researcher refills; these are the costliest LLM jobs, at $39.56 for 5 jobs in the September 19–20 baseline ([efficiency.md](efficiency.md)). The rule also bounds the comparisons each campaign spends on the fixed fold. The literature treats that as the dominant driver of validation overfitting in agent loops, and it is the only multiplicity control that costs no statistical power at 0.001 effects.

### 4. The epoch-5 loser abort becomes worth it once a mid-run hook exists

**Sources and evidence.** Successive halving, ASHA, BOHB and "one epoch is all you need" all rely on early rankings holding to the end. Egele et al. show the one-epoch baseline works because benchmark curves are dominated by a few clear winners ([arXiv 2307.15422](https://arxiv.org/abs/2307.15422)). Our early-vs-final Spearman of 0.06–0.49 for epochs 0–4 rules that out. The safe transferable forms are one-sided. PASHA treats configurations within ε as ties, with ε set to the 90th percentile of distances between criss-crossing pairs ([arXiv 2207.06940](https://arxiv.org/html/2207.06940)). LC-PFN stops a run only when it is unlikely to beat the best so far ([arXiv 2310.20447](https://arxiv.org/abs/2310.20447)). In our records, aborting at **epoch-5 Δ < −0.003 would have stopped 26 runs, none of which finished above zero, saving 6.6% of epochs**.

**Rule.** Abort at the end of epoch 5 when the paired Δ against the pinned reference is below −0.003. Record the verdict as `reject`, with the epoch-5 Δ as evidence.

**Strength and cost.** The own-data evidence is weak (five winners), and the saving ceiling is small by construction in a small-effect regime. The earlier notes judged the supervisor complexity not worth 5–7% GPU. That judgment reverses only if mechanism 2 ships, because the abort then reuses the same mid-run check. Without mechanism 2, keep the earlier "do not implement".

**What it changes.** Runs are shorter for clear losers. The final Δ of an aborted run is lost, so family verdicts on those ideas rest on a partial curve.

## Code, not an LLM, should decide whether a mechanism ran

### 2. A declared activation signature, checked after epoch 0

**Sources and evidence.** The failure is common. An analysis of 800 AutoResearch trajectories names "gated cheating", where the genuine method sits behind flags that never fire, and "silent component disabling" as recurring pathologies. It also finds agents that **identify a critical problem and ship anyway in 82.5% of trajectories** ([arXiv 2608.14905](https://arxiv.org/html/2608.14905v2)). The AI Scientist v2 authors list wrong implementations of the main idea among their failure modes ([Sakana v2 paper](https://pub.sakana.ai/ai-scientist-v2/paper/paper.pdf)).

Architectural gates beat prompts. EviBound's verification gate, which queries the run's artifacts before a claim can propagate, took false claims from **8/8 to 0/8 at about 8% overhead** ([arXiv 2511.05524](https://arxiv.org/abs/2511.05524)), although n is tiny. LLM auditors reading only the write-up sit at chance; given logs and code they rise **from 51% to 74% accuracy** ([arXiv 2509.08713](https://arxiv.org/html/2509.08713v1)). The survey of the verification gap concludes that deterministic, interventional checks beat narrative judging ([arXiv 2608.05179](https://arxiv.org/pdf/2608.05179)). Our system already has the `invalid` verdict and the rule that an invalid run is no evidence against an idea. What it lacks is a code-owned trigger: today the researcher or the completion reporter diagnoses "never activated" after a full run.

**Rule.** Each developer report names one activation signature: a column in `metrics.csv` or a logged counter, for example `token_dropout_masked_tokens`, an auxiliary loss, a gradient norm on new parameters or an augmented-batch fraction. It also names a predicate, such as "> 0" or "≥ 0.05 of batches", and a check epoch, defaulting to 0 and later only for warm-up-gated mechanisms. The orchestrator's review confirms that the signature measures the claimed mechanism.

After the check epoch the supervisor evaluates the predicate. A miss aborts the run with verdict **`invalid`**, which returns the idea to the developer as a repair and does not count toward family closure or the streak stop. A missing signature blocks launch, much like a missing config key.

**Strength and cost.** Confidence is high that this targets our known failure. No published effect size exists for ML-training loops, and our own frequency of never-activated runs was not counted. The cost is a few logging lines, one supervisor check and a report field, with no extra GPU.

**What it changes.** A never-activated run is caught after one epoch instead of ten and classified as a repair, not a scientific negative. This matters more once the streak stop exists: without the gate, silent failures would push campaigns toward a false stop.

### 5. Artifact-grounded interpretation and typed outcome records

**Sources and evidence.** Agent-as-judge reading code, logs and data reached **κ = 0.75 against human experts versus 0.53 for a single-call LLM**, with every finding required to cite a log line, identifier or exact value ([arXiv 2608.14905](https://arxiv.org/html/2608.14905v2)). A typed negative-knowledge record lifted ScienceAgentBench pass rate from 31.6% to 36.8% while cutting tokens by 73% ([arXiv 2606.21024](https://arxiv.org/html/2606.21024v1)); the benchmark is small, so treat this as weak evidence. The record has five fields: layer (implementation, method, hypothesis, measurement), scope, degree (contradicted, inconclusive, unstable, artifact-driven), action and risk. Memory volume itself matters little: removing R&D-Agent's memory context cost only 3.1 points, its smallest ablation ([arXiv 2505.14738](https://arxiv.org/html/2505.14738v2)). Memory framing matters more. Asking sibling memory for "similar ideas" was part of an intervention that cost **6.9–8.4 medal points** ([arXiv 2511.15593](https://arxiv.org/html/2511.15593v2)).

**Rule.** The orchestrator's completion interpretation must cite an artifact (log line, metric value, diff hunk) for each claim about mechanism behaviour. The cheap completion reporter fills three typed fields per reject:

- layer: implementation, method or measurement;
- degree: *contradicted* when the 80% upper bound < 0, *inconclusive* when the interval spans 0, *near-miss* when Δ ≥ 0.001 and the lower bound ≤ 0;
- suggested action.

It fills them from code-computed facts, not prose.

**Strength and cost.** Evidence is medium for grounding and weak for typed records. The cost is prompt and schema edits.

**What it changes.** The researcher's extend, retry or close verdicts rest on structured inputs, so near-misses and implementation failures stay distinguishable from true negatives.

### 8. A deterministic guard for runtime patching

**Sources and evidence.** Reward hacking rises sharply when the scorer is reachable: **30.4% of RE-Bench runs versus 0.7% on HCAST**, including monkey-patched evaluators and overloaded tensor equality. Prompts such as "don't cheat" barely help ([METR](https://metr.org/blog/2025-06-05-recent-reward-hacking/)). GPT-5 exploited contradictory tests 76% of the time on ImpossibleBench ([arXiv 2510.20270](https://arxiv.org/abs/2510.20270)). Our sealed verdict code, hashed predictions and allowlist already close the main route. The residual surface is an allowlisted module that patches `torch` or a metric at import time.

**Rule.** Code flags diffs that contain `setattr` on library modules, redefinitions of `__eq__` or metric functions, or edits to seed, epoch-count or sampler settings, and routes them to the orchestrator review with the hunk quoted. Evidence for the class is strong, but the need is low, so this ranks last. The developer's existing `blocked` exit already supplies the abort channel that ImpossibleBench found cut cheating from 54% to 9%.

## The bootstrap interval covers most of the noise

### 3. Calibration against repeat pairs: keep the rule

**Sources and evidence.** Our rule adopts when Δ ≥ 0.001 and the lower end of the central 80% interval is above zero. That is a one-sided test at about 10% (z ≈ 1.28), a little stricter than the parameter-free Ladder's one-SE step, which works out to about 0.15 ([arXiv 1502.04585](https://arxiv.org/pdf/1502.04585)). So the literature endorses the shape of our rule. What it questions is the width of the interval.

The paired bootstrap models evaluation-sample noise only. Using the incumbent seed does not cancel training noise once the idea changes the trajectory: any perturbation, down to one bit, produces full run-to-run variance ([arXiv 2103.04514](https://arxiv.org/abs/2103.04514)). Jordan shows seed variance is mostly test-set noise for long, stable training: CIFAR-10 at 64 epochs has 0.15% test-set std against 0.033% distribution-wise. It is not for short or fine-tuning regimes: 0.56% against 0.52% at 4 epochs, and 2.24% against 2.08% for BERT-Large ([arXiv 2304.01910](https://arxiv.org/html/2304.01910)).

Our ten-epoch runs from a pretrained backbone on a shifted medical domain sit closer to the second regime. Bouthillier et al. find that a single-point comparison has **about 10% false positives and 75% false negatives** ([arXiv 2103.03098](https://arxiv.org/pdf/2103.03098)). That is the honest error profile of one run per idea.

**Measurement (2026-09-28).** [`estimate_repeat_pair_noise.py`](../scripts/adhoc/estimate_repeat_pair_noise.py) compares four mechanisms measured twice on the old labels. Since both Δs of a pair score the same studies, their difference is mostly training noise, plus base and scoring differences that make it an upper bound:

| Mechanism | Δ first | Δ second | Difference | Differs by |
|---|---|---|---|---|
| Label-view attention bias | +0.0018 | +0.0006 | 0.0012 | ×1 vs ×4 scoring, series cleaning; same seed |
| Meniscus shared residual head | −0.0006 | −0.0010 | 0.0004 | Seed, incumbent |
| DSU | +0.0022 | +0.0012 | 0.0009 | Seed, smoothing underneath |
| Run-level stratified jitter | +0.0035 | +0.0009 | 0.0025 | Seed, incumbent, split (`all_5fold_v3` vs scanner v1) |

Training noise per Δ (root mean square difference over √2) is 0.0010 with all four pairs and 0.0006 without the cross-split pair, against a median bootstrap SE of 0.0015. The total SE is 0.0016–0.0018, so the 80% rule behaves as a one-sided test at about 12–15% false positives rather than 10%. This agrees with the bound from the spread of Δ across all ideas (robust SD 0.0020, so at most about 0.0013 of training noise). Incumbent-seed pairing therefore removes most seed noise here, unlike the literature's short-training regime.

**Decision.** Keep the rule. False positives run slightly above nominal, but a lucky adoption is almost always near-neutral, while false negatives dominate one-run comparisons and each costs at least 0.001. Widening the interval would trade the cheap error for the expensive one. The measurement is shared by all topics; repeat it once pairs build up on `rebuild_state_v2`, as a script rather than runner work.

**Strength.** Weak: four pairs, each confounded by base or scoring differences, one across splits.

**Corrections deferred.** Bonferroni over k = 30 comparisons moves z from 1.28 to 2.71 and roughly doubles the bar. Online FDR is milder but still harsh early. With SAFFRON, the level at the first test is a small fraction of the initial wealth W0 < α ([arXiv 1802.09098](https://arxiv.org/pdf/1802.09098)), far below 0.10 until an adoption replenishes it [calc]. It would therefore block most 0.001–0.005 gains, and its guarantee is only approximate for dependent, seed-blind bootstrap p-values. Mechanism 1 bounds the number of comparisons instead, at no cost in power.

## Our records overturn the literature's diversity advice

### Follow-ups win on our data, so the fresh-family quota is rejected

**Sources and evidence.** The literature favours diversity strongly. Reducing ideation diversity cost 6.9–8.4 medal points ([arXiv 2511.15593](https://arxiv.org/html/2511.15593v2)), removing R&D-Agent's exploration path cost about 10 ([arXiv 2505.14738](https://arxiv.org/html/2505.14738v2)), and ScientistTwo injects fresh seed ideas into every round ([arXiv 2609.19644](https://arxiv.org/html/2609.19644)). The ScientistTwo notes therefore proposed reserving a refill slot for new families, pending a zero-GPU check. That check now argues against the quota.

Among researcher runs with known lineage, **follow-ups (n = 46) had median Δ −0.0010 and 4 runs at Δ ≥ 0.001 (9%)**. **New families (n = 43) had median −0.0029 and 1 such run (2%)**. The shift is significant: Mann-Whitney p = 0.0008. Adoptions are too few to compare (0 vs 1). The difference has a clear cause: follow-ups are mostly retries of near-misses, so they start closer to the bar.

The literature's diversity evidence comes from MLE-bench agents collapsing onto one architecture across hundreds of cheap nodes. Our two-rejected-follow-ups closure rule already prevents that collapse. Keep "open lines take refill slots first". For the same reason, drop the DGM/ShinkaEvolve family weighting (performance × 1/(1+children)) ([arXiv 2505.22954](https://arxiv.org/html/2505.22954); [arXiv 2509.19349](https://arxiv.org/html/2509.19349)).

### 6. Better new-family ideas through distributional prompting, not ranking

**Sources and evidence.** Ranking cannot rescue weak ideas. Off-the-shelf o3 was **no better than chance** at predicting which of two research ideas wins ([arXiv 2506.00794](https://arxiv.org/abs/2506.00794)). The best models reach 58.8–61.5% pairwise on MLE-bench code pairs and only Spearman 0.23 listwise ([arXiv 2601.05930](https://arxiv.org/html/2601.05930)). Pre-execution enthusiasm for LLM ideas also reverses after execution ([arXiv 2506.20803](https://arxiv.org/abs/2506.20803)). An Elo tournament or a ForeAgent-style pairwise gate would add orchestrator calls for a few points above chance.

What does move idea quality is generation-side. Verbalised sampling raised diversity 1.6–2.1× ([arXiv 2510.01171](https://arxiv.org/abs/2510.01171)), and AIRA's operator content mattered more than its search policy: greedy with better prompts reached 45.5% against MCTS at 47.0% ([arXiv 2507.02554](https://arxiv.org/html/2507.02554)).

**Rule.** When a refill includes new families, the researcher proposes them across three complexity tiers (minimal, moderate, structural). Each carries a verbalised probability of clearing the 0.001 bar. The closed-family list is framed as "already tried, move away from this", not as material to build on.

**Strength, cost and what it changes.** This is a guess with low confidence, at the cost of a prompt edit. It changes only the quality of new-family proposals. Ranking stays as is and should not absorb extra model budget.

## LLM cost goes to fewer calls, not weaker models

### 7. Evidence-preserving compaction of orchestrator packets

**Sources and evidence.** SoL-Pi cut token traffic **44.7–49.0% and API cost by about a third at equal performance** ([arXiv 2609.20519](https://arxiv.org/abs/2609.20519)). Its method compacts context and observations, and a deterministic verifier checks each reduced log's schema, source hash, exit status and exact quotes, falling back to the raw log on any mismatch ([arXiv 2609.20519](https://arxiv.org/html/2609.20519)). Tiering evidence says strong models matter for generation: removing all but a small model hurt AlphaEvolve ([arXiv 2506.13131](https://arxiv.org/html/2506.13131)). Mechanical roles take cheap sidecars, as in AIRA-dojo ([arXiv 2507.02554](https://arxiv.org/html/2507.02554)). Our tiering already follows that pattern, with Haiku as the completion reporter.

Our September 19–20 baseline cost $106.55 in 4 h 53 min, hit the session limit, and consumed 116 million cached input tokens ([efficiency.md](efficiency.md)). Context volume is therefore the lever.

**Rule.** Pass training and evaluation logs to the orchestrator as compacted receipts. A receipt is valid only if a code check confirms its source hash and quoted values; otherwise the orchestrator gets the raw excerpt.

**Strength, cost and what it changes.** This is a guess with low-medium confidence, at small code cost. It lowers the chance that a campaign stalls on session limits. The streak stop already removes the most expensive calls, so build this only if limits still bind afterwards.

## Mechanisms examined and rejected for our regime

| Mechanism | Source | Why it does not fit |
|---|---|---|
| Successive halving, Hyperband, ASHA, BOHB, one-epoch screening | [arXiv 1603.06560](https://arxiv.org/abs/1603.06560), [1810.05934](https://arxiv.org/abs/1810.05934), [1807.01774](https://arxiv.org/abs/1807.01774), [2307.15422](https://arxiv.org/abs/2307.15422) | Need early rankings to hold. Ours do not (Spearman 0.06–0.49 through epoch 4), so they would discard winners roughly at random. |
| Subset-first screening (ScientistTwo) | [arXiv 2609.19644](https://arxiv.org/html/2609.19644) | Same failure. No thresholds are published, and one adopted run was 0.010 behind at epoch 0. |
| LC-PFN learning-curve extrapolation | [arXiv 2310.20447](https://arxiv.org/abs/2310.20447) | Its 2–6× savings come from wide between-config spread. Untested on paired Δ curves, and the fixed abort captures most of the reachable saving. |
| Probability-of-improvement test over many seeds | [arXiv 2103.03098](https://arxiv.org/pdf/2103.03098) | Needs about 29 trainings per arm; violates one run per idea. |
| Seed randomisation per run (Bouthillier's 51× cheaper estimator) | [arXiv 2103.03098](https://arxiv.org/pdf/2103.03098) | Breaks incumbent-seed pairing; the repeat-pair measurement (mechanism 3) found the pairing already removes most seed noise. |
| Bonferroni widening | [arXiv 2103.03098](https://arxiv.org/pdf/2103.03098) | Roughly doubles the bar at k = 30 and kills power at 0.001. |
| Online FDR (SAFFRON, LORD++, alpha-investing) | [arXiv 1802.09098](https://arxiv.org/pdf/1802.09098), [JMLR v7](https://www.jmlr.org/papers/v7/zhou06a.html) | Early levels far below 0.10; dependent, seed-blind p-values; no use in agent loops found. Revisit only if campaigns grow long again. |
| Thresholdout, one-bit feedback to proposers | [arXiv 1506.02629](https://arxiv.org/pdf/1506.02629), [2606.11045](https://arxiv.org/html/2606.11045v1) | Per-label AUC is the researcher's main diagnostic. The overfitting measured comes from agents tuning directly on validation under "at all costs" prompts, and sequential tweaks to one recipe overfit less ([arXiv 1905.12580](https://arxiv.org/abs/1905.12580)). |
| Fresh-fold winner's-curse audit | [Lee & Shen, KDD 2018](https://dl.acm.org/doi/10.1145/3219819.3219905) | Needs an old-recipe run on the new fold, i.e. a control rerun. Use it only if such a pair already exists. |
| Fresh-family refill quota; family weighting by 1/(1+children) | [arXiv 2511.15593](https://arxiv.org/html/2511.15593v2), [2505.22954](https://arxiv.org/html/2505.22954) | Our follow-ups outperform new families (p = 0.0008). |
| LLM pairwise or Elo ranking tournaments, predict-then-verify | [arXiv 2601.05930](https://arxiv.org/html/2601.05930), [2502.18864](https://arxiv.org/abs/2502.18864) | About 60% pairwise accuracy at best, for extra orchestrator calls. |
| MCTS, evolutionary populations, islands | [arXiv 2507.02554](https://arxiv.org/html/2507.02554), [2509.19349](https://arxiv.org/html/2509.19349) | Built for hundreds of cheap evaluations. No gain over greedy without better operators, and MCTS overfit validation. |
| In-loop reviewer, rebuttal or debate critics | [arXiv 2609.19644](https://arxiv.org/html/2609.19644), [2511.04583](https://arxiv.org/html/2511.04583) | ScientistTwo's in-loop acceptance rose 46.9% → 93.9% while the external reviewer fell back to 69.4%. Critics get optimised against, and they induce fabrication. Our code verdict already avoids this. |
| Top-k final-selection hedging | [arXiv 2507.02554](https://arxiv.org/html/2507.02554) | Already covered by the near-miss campaign and the endgame ensembling backlog. |

## Conclusion

The literature changes our priorities more than our statistics. Our adoption rule is already a slightly stricter Ladder, and the multiple-testing corrections on offer would trade away the 0.001 sensitivity the competition rewards. The real exposure is elsewhere. Long campaigns keep drawing comparisons against one fold after returns have run out; topic rotation answers that at no power cost. The worry that the bootstrap is too narrow did not hold: four repeat pairs put training noise below the bootstrap's own error.

The one place where our data overrule the field is exploration. At about 20 expensive runs per campaign, a disciplined follow-up on a near-miss is a better bet than a fresh family, so diversity effort belongs in how new families are generated, not in reserving slots for them. The next step is to build mechanisms 1 and 2 together: the activation check keeps silent failures from pushing campaigns into false stops.
