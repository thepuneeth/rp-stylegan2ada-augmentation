# Methods narrative and audit of record

**Date:** 2026-08-25
**Scope:** full audit of the variability study following completion of both
generators, plus the prose in which the study will be described.

This document has two halves. Part A records what was verified, by what means,
and what was found; it is the internal audit and is not intended for the
manuscript. Part B is the manuscript prose — the description of the research,
the methodology, and the phrasing conventions that keep the account accurate
without overstating what the design supports.

---

# Part A — Audit

## A0. What was checked, and how

| # | claim under test | instrument | verdict |
|---|---|---|---|
| 1 | eye-disjoint partition, no holdout image reaches any generator | `s00_verify.py`, 22 invariants | **PASS** |
| 2 | `clean_111` and `leak_111` differ in content, not merely in name | MD5 over both zips and per-image content hashing | **PASS** |
| 3 | the two arms differ in exactly one variable | key-by-key diff of both `training_options.json` | **PASS, with one qualification (F4)** |
| 4 | both selected snapshots are intact and loadable | `legacy.load_network_pkl` on each | **PASS** |
| 5 | the prespecified selection rule was applied as written | recomputed from `metric-kid50k_full.jsonl` | **PASS** |

## A1. Partition integrity

`s00_verify.py` reports **22 of 22 invariants passing**, including the hash
self-test that establishes the leak checks are not vacuous. The partition is
eye-disjoint at every level tested: no file and no eye is shared between the
development pool and the holdout, in either class, and the partition is
complete (28 + 111 = 139 for RP; 207 + 817 = 1024 for Healthy). Prevalence is
matched to within 0.04 percentage points — holdout 0.1191 against a cohort
value of 0.1195.

Per the leakage taxonomy, the residual exposures are those the split table
cannot show and which are recorded here rather than resolved: patient identity
is unavailable in the source dataset, so contralateral eyes of the same person
may fall on opposite sides of the partition; and the holdout, while
generator-blind, has been read before — it was used to quantify the colour
shortcut. Both are stated as limitations in Part B.

## A2. Dataset composition

Content hashing confirms the size-matched contamination design was materialised
as specified:

```
clean_111   111 unique images
leak_111    111 unique images
shared       83
only clean   28
only leak    28
```

The 28 images unique to `leak_111` are the holdout eyes; the 28 unique to
`clean_111` are the development images they displaced. Training-set size is
therefore held at 111 across both arms, which is what licenses the comparison
as causal with respect to contamination rather than confounded with data
volume.

## A3. The matched recipe

A flattened key-by-key comparison of the two `training_options.json` files
returns **58 identical keys and 3 differing keys**:

| key | `g_clean` | `g_leak` |
|---|---|---|
| `training_set_kwargs.path` | `clean_111.zip` | `leak_111.zip` |
| `run_dir` | `.../g_clean/...` | `.../g_leak/...` |
| `metrics` | `[fid50k_full]` | `[fid50k_full, kid50k_full, pr50k3_full]` |

Of the three, `run_dir` is cosmetic and `training_set_kwargs.path` is the
intended variable. The `metrics` difference is discussed as finding **F4**.
Everything that governs optimisation is identical and was confirmed
individually: `random_seed` 0, `total_kimg` 3000, `batch_size` 64,
`r1_gamma` 5.0, `ada_target` 0.6, `ema_kimg` 20, both learning rates 0.0025,
`xflip` enabled, `max_size` 111, and the same `resume_pkl` — NVIDIA's
`ffhq-res256-mirror-paper256-noaug.pkl`. Both generators therefore begin from
byte-identical weights.

## A4. Training outcome

Both runs reached 3,000 kimg over 746 ticks with a peak allocated GPU memory of
19.17 GB. Wall-clock differed by roughly half an hour (9.04 h against 9.64 h),
attributable to the two additional metrics `g_leak` computed at each snapshot
and to one episode of host contention.

Metrics at every snapshot, each computed against that generator's own training
set (see **F1**):

| kimg | clean KID | clean FID | clean P | clean R | leak KID | leak FID | leak P | leak R |
|---|---|---|---|---|---|---|---|---|
| 0 | — | 393.78 | — | — | 0.3991 | 395.25 | 0.000 | 0.000 |
| 403 | — | 62.06 | — | — | 0.0271 | 61.79 | 0.337 | 0.027 |
| 806 | — | 58.40 | — | — | 0.0673 | 92.77 | 0.000 | 0.000 |
| 1209 | — | 56.21 | — | — | 0.0231 | 57.65 | 0.393 | 0.054 |
| 1612 | — | 53.93 | — | — | 0.0234 | 56.40 | 0.433 | 0.045 |
| 2016 | 0.0205 | 52.86 | 0.566 | 0.045 | 0.0215 | 54.06 | 0.466 | 0.045 |
| 2419 | 0.0200 | 52.24 | 0.594 | 0.018 | 0.0197 | 51.99 | 0.491 | 0.027 |
| 2822 | **0.0185** | 50.86 | 0.603 | 0.027 | 0.0189 | 50.68 | 0.506 | 0.063 |
| 3000 | 0.0194 | 51.21 | 0.624 | 0.054 | **0.0188** | 50.42 | 0.494 | 0.036 |

KID was logged for `g_clean` only at and above `SELECT_MIN_KIMG`, since it was
computed post hoc for the four selection candidates rather than during
training.

## A5. Snapshot selection

The prespecified rule — lowest `kid50k_full` among snapshots at or above 2,000
kimg, ties to the earlier snapshot — was recomputed from the raw metric files
and yields:

- `g_clean`: **`network-snapshot-002822.pkl`**, KID 0.018538
- `g_leak`: **`network-snapshot-003000.pkl`**, KID 0.018843

In both arms FID independently selects the same snapshot, and in both arms the
selected snapshot is not trivially the last available one for `g_clean`, which
is the circumstance the rule was written to handle. Both files load and expose
a 256-pixel, unconditional generator with 24,767,458 `G_ema` parameters.

Note for the manuscript: `config.py` and the README describe the generator as
"~3e7 parameters". The measured figure is 2.48e7. The rounding should be
corrected to "approximately 2.5 x 10^7" wherever it appears.

## A6. Findings

### F1 — The two KID values are measured against different reference sets · **Major, interpretive**

Every metric in the StyleGAN2-ADA `metrics/` module builds its reference
statistics from the training set supplied in `dataset_kwargs`. `g_clean` was
therefore scored against `clean_111` and `g_leak` against `leak_111`, and those
two reference sets differ in 28 of 111 images.

For **selection** this is correct and causes no difficulty: each arm selects its
own best snapshot against its own target distribution, which is exactly what the
prespecified rule asks.

For **cross-arm comparison** it does not license the sentence "the two
generators are of equal quality". The honest reading is narrower — each
generator fits its own training distribution to a comparable degree. A claim of
equal quality against a common yardstick would require rescoring both against a
single fixed reference set, which is inexpensive but has not been done.

The asymmetry also runs in a known direction: because `leak_111` contains the
holdout eyes, a memorising `g_leak` scores *better* on every built-in metric.
This is anticipated in `config.py` and is precisely the gap `s06` exists to
close, using the generator-blind holdout that no built-in metric can see.

**Action:** either rescore both selected snapshots against a common reference
set, or phrase the comparison as "each fits its own training distribution
comparably" and let `s06` carry the quality claim.

### F2 — `pr50k3_full` recall is quantised at 1/111 · **Moderate, interpretive**

Recall in the table above takes only the values 0.018, 0.027, 0.036, 0.045,
0.054 and 0.063 — every one an exact multiple of 1/111. The real manifold is
estimated from 111 images, so recall counts how many of those 111 fall inside
the generated manifold, and it moves in steps of 0.9 percentage points. The
observed range spans **two to seven images**.

Differences between snapshots on this metric are therefore not interpretable,
and the metric should be reported with its denominator stated rather than as a
continuous quantity. Precision, computed over 50,000 generated samples, does not
suffer from this and is stable.

### F3 — `g_leak` experienced a training instability that `g_clean` did not · **Moderate, reportable**

The ADA augmentation probability in `g_leak` fell to **0.000 at kimg 673** and
remained near zero until roughly kimg 830, over which interval FID rose from
61.79 to 92.77 and precision fell to 0.0003. The run recovered fully by kimg
1209 and its remaining trajectory is unremarkable. `g_clean` shows no comparable
episode; its post-warmup minimum augmentation probability is 0.133.

The episode falls far below `SELECT_MIN_KIMG` and cannot affect selection.
It should nevertheless be reported, because it is direct evidence that two runs
sharing a seed, an initialisation and every hyperparameter still diverge in
trajectory when the data differ — which is the substance of the n=1 limitation
and is better shown than asserted.

### F4 — The differing `metrics` list perturbs the CUDA random stream · **Minor, qualifies "matched seed"**

`metrics/metric_utils.py:263` draws `torch.randn(...)` on the training device,
and `metric_main.calc_metric` does not reset the seed. The training loop draws
its own latents from the same global generator at `training/training_loop.py:263`.
Because `g_clean` evaluated one metric per snapshot and `g_leak` evaluated three,
the two runs consume different quantities of random state at each snapshot and
their latent streams diverge from the first snapshot onward.

The practical consequence is small. Initialisation is identical because both
resume from the same checkpoint, and the data loaders could never have been
aligned in any case, since the two datasets contain different images. What the
finding removes is the stronger claim that the runs are matched in stochastic
trajectory; what survives is that they are matched in initialisation and in
every hyperparameter. The manuscript should say the latter.

This finding does not weaken the design so much as reinforce a limitation the
protocol already carries: with one generator per arm, a data effect and a
trajectory effect cannot be separated.

### F5 — `config.py` and `FINAL_PLAN.md` still describe `synth_leak` as unaffordable · **Documentation, stale**

`config.py` lists `synth_leak` under "Cut, and why", and `README.md` records
contrast P2 as dropped for want of budget. Both statements are now false: the
matched generator exists, trained under an identical recipe, and its snapshot is
selected. `ARMS` does not yet include it and `s05` will not route it.

**Action:** add the arm to `config.ARMS`, restore contrast P2, and rewrite the
three "dropped" entries. This is the largest single gap between the code as
written and the study as it now stands.

### F6 — Snapshot cadence in `FINAL_PLAN.md` is stale · **Documentation, minor**

Section 6 specifies a snapshot every 200 kimg via `--snap=50`. The runs used
`--snap=100`, giving a snapshot every 403 kimg, which matches
`config.SNAPSHOT_EVERY_KIMG = 400` and the four selection candidates the design
requires. `config.py` is correct; `FINAL_PLAN.md` should be brought into line.

### F7 — The selection metric is described inconsistently across documents · **Documentation, minor**

`config.SELECT_METRIC` is `kid50k_full`, and the reasoning — that FID is a
biased estimator whose bias grows as the reference set shrinks, and that this
reference set holds 139 images — is stated at length. Two documents still
specify FID: `README.md` under "s06 gates the GPU day", and `FINAL_PLAN.md`
section 6. The rule as executed was KID. The two documents should be corrected
so that no reader can find a written rule the analysis did not follow.

### F8 — Stale absolute paths in two output files · **Cosmetic**

`outputs/s03_novelty_legacy_v3.json` and `outputs/s05_classifier_smoke.json`
record `/Users/puneeth/Documents/pigmentGAN/...`, from before the project moved
under `Documents/Research/GAN/`. The results are unaffected; the paths simply no
longer resolve. They will correct themselves when either script is next run.

### F9 — The classifier endpoint is underpowered, and by how much · **Known, quantified**

`outputs/s07_power.json` gives the minimum detectable difference in eye-level
AUC at 80% power as **0.043 to 0.070**, depending on the assumed within-eye
correlation:

| rho | MDE at 80% power | null CI half-width | type I error at delta = 0 |
|---|---|---|---|
| 0.5 | 0.0695 | 0.0628 | 0.083 |
| 0.7 | 0.0615 | 0.0505 | 0.090 |
| 0.9 | 0.0427 | 0.0309 | 0.067 |

Two consequences follow. The study can detect only large downstream effects, so
a null result on the classifier is close to uninformative and must be reported
as such rather than as evidence of equivalence. And the empirical type I error
under the null runs at 0.067 to 0.090 against a nominal 0.05, meaning the
percentile bootstrap is anti-conservative at this sample size. A BCa interval or
a permutation test would correct this, and one of the two should be adopted
before any p-value is reported.

## A7. Summary

Nothing in the audit invalidates a result. The partition holds, the arms are
matched where they were claimed to be matched, both generators completed, and
the selection rule was applied as prespecified. The findings divide into three
kinds: two that constrain how the metrics may be phrased (**F1**, **F2**), one
that is a genuine and reportable observation about training (**F3**), one that
narrows a claim without changing a conclusion (**F4**), and a set of
documentation repairs of which **F5** is the only substantial one.

---

# Part B — The manuscript prose

## B1. How the research is described

The prior study balanced a Retinitis Pigmentosa dataset with 885 synthetic
images and reported an increase in AUC from 0.89 to 0.97, with sensitivity
rising from a range of 0 to 53% to a range of 88 to 97%. The improvement was
genuine as measured. This study asks where it came from, and decomposes it into
the two sources that can be separated within the available cohort: the
correction of class imbalance, and whatever information the generator
contributes beyond that correction.

Three properties of the prior methodology motivate the decomposition, and each
is readable in that work's own description. The synthetic images were merged
into the dataset before the five-fold split was drawn, so approximately 177 of
roughly 205 validation positives in each fold were synthetic images produced by
a generator that had been trained on the real positives. Balancing altered two
things simultaneously — it introduced synthetic data and it corrected a 7.4:1
class ratio — and no oversampling arm was present to separate the two. Finally,
a source-level shortcut solves the greater part of the task: logistic regression
on six global colour moments reaches an AUC of 0.9501 on the generator-blind
holdout, which places a floor beneath any deep model's result and below which no
augmentation comparison is interpretable.

This study addresses the aforementioned issues by holding the evaluation set
apart from every generative model it will be used to judge. The 139 real
Retinitis Pigmentosa images, which resolve to 134 unique eyes, are partitioned
eye-disjointly into a development pool of 111 images and a holdout of 28 eyes
that no generator ever sees. Any advantage measured on that holdout therefore
cannot be attributed to memorisation.

## B2. Data and partition

The Eye Disease Image Dataset, a publicly available collection released under
CC BY 4.0 containing 5,335 original fundus images, of which 139 are Retinitis
Pigmentosa and 1,024 are healthy, was used throughout. Eye identity was
established by correlation between field-of-view-cropped grayscale images,
clustered with complete linkage at a threshold of 0.99. This metric was
calibrated against five independently confirmed duplicate pairs, all five of
which it recovers with no false positives above the weakest of them.

The partition is eye-disjoint and stratified so that prevalence is preserved:
the holdout carries a prevalence of 0.1191 against a cohort value of 0.1195.
Twenty-two automated invariants over the split tables, the packaged archives and
the image hashes are executed before any training run and all pass.

An earlier duplicate audit of this cohort reported 132 exact collisions under
average hashing and grounded a claim of effective-sample-size collapse. That
result was an artifact of the hashing method, which reduces an image to an
eight-by-eight binary pattern and collides readily across images of a single
modality. Under the calibrated metric described above, genuine duplication
amounts to five pairs among 139 images, and the effective sample size is 134.
The collapse claim is withdrawn.

## B3. The generators

Two generators were trained, and they differ in exactly one respect.

Both are StyleGAN2-ADA at 256 pixels, initialised by transfer from NVIDIA's
FFHQ-256 checkpoint and trained for 3,000 kimg on a single RTX 4090, with
gamma 5, an ADA target of 0.6, no frozen discriminator layers, a batch size of
64, mirror augmentation, and random seed 0. The first, denoted `g_clean`, was
trained on the 111 development images. The second, denoted `g_leak`, was trained
on a set of identical size assembled from the 28 holdout eyes and 83 development
images drawn at random under the study seed.

Matching the size of the two training sets is what permits the comparison to be
read as an effect of contamination rather than of data volume. A comparison
against the prior generator, which saw all 139 images, would confound the two,
since both differences push a generator's apparent novelty in the same
direction, and would therefore yield only an upper bound.

The two configurations were compared key by key after training and agree on all
58 optimisation parameters, differing only in the path to the training archive
and in cosmetic fields. Both generators resume from the same checkpoint and thus
begin from identical weights. It should be noted, however, that identical
initialisation does not extend to identical stochastic trajectory: the two runs
draw latents from a shared device generator whose state is also advanced by
metric evaluation, and the datasets themselves differ, so the sample streams
diverge. The runs are matched in initialisation and in every hyperparameter, and
that is the claim made here.

One further observation belongs in the record. `g_leak` passed through a
transient instability between roughly 673 and 830 kimg, over which its ADA
augmentation probability fell to zero, its FID rose from 61.79 to 92.77 and its
precision fell to near zero. The run recovered completely and its selected
snapshot is unaffected, the episode having occurred well below the selection
floor. `g_clean` exhibits no comparable episode. This divergence between two
runs sharing a seed and a recipe illustrates directly why a single generator per
arm cannot separate an effect of the data from an effect of the trajectory.

## B4. Snapshot selection

The prior project's best checkpoint was not its last, so a fixed stopping point
would occasionally hand over the inferior generator. Selection was therefore
prespecified: the snapshot with the lowest `kid50k_full` among those at or above
2,000 kimg, with ties broken toward the earlier snapshot. Selection was executed
offline on downloaded checkpoints, so no decision could be driven by results
observed during training.

KID rather than FID carries this rule. Both are supplied by StyleGAN2-ADA at the
cost of a single flag, but FID is a biased estimator whose bias grows as the
reference set contracts, and the reference set here holds 139 images. KID is the
unbiased small-sample estimator of the same discrepancy and is accordingly the
correct choice at this scale. The rule selected checkpoint 2822 for `g_clean`
and checkpoint 3000 for `g_leak`, and FID independently selects the same
checkpoint in both arms.

Two qualifications attach to the values themselves. Each generator is scored
against its own training set, since every metric in the StyleGAN2-ADA module
builds its reference statistics from the data supplied for training; the
resulting figures therefore express how closely each generator fits its own
target distribution and do not constitute a comparison on a common yardstick.
Furthermore, the recall component of the built-in precision–recall pair is
computed over 111 real images and consequently advances in increments of
0.9 percentage points, so differences between checkpoints on that measure span
between two and seven images and are not interpretable.

Throughout this work FID is treated as a monitoring signal rather than an
endpoint. Computed against 139 reals, it is informative as a within-run trend
and carries no meaning as an absolute value, and it is not compared to published
figures obtained on datasets of tens of thousands of images.

## B5. Endpoints

The primary endpoint is measured on the images themselves rather than inferred
from downstream accuracy, and it is measured in RETFound space — a retinal
foundation model, ViT-L under a masked-autoencoder objective, pretrained on 1.6
million retinal images. Inception features encode ImageNet semantics and carry
no representation of bone-spicule pigmentation or arteriolar attenuation, so
Inception is retained as a single row for comparability with published work and
does not carry a claim.

Two established measures are computed, both with proper nulls. Authenticity,
following Alaa and colleagues, identifies a synthetic sample as inauthentic when
it lies closer to its nearest real image than that real image lies to its own
nearest neighbour — that is, when it has crawled inside the spacing of the real
data. The C_T copying test of Meehan and colleagues compares the distances from
synthetic samples to their nearest training images against the distances from
held-out real images to their nearest training images, and is reported both
globally and per cell over a k-means partition of the training reals, since
copying confined to one region is the ordinary failure mode and a global
statistic conceals it.

These two measurements exist because the built-in metrics cannot supply them.
Every metric shipped with StyleGAN2-ADA constructs its reference statistics from
the training set, without exception, so a generator that memorised its training
data perfectly would score the best attainable value on all of them
simultaneously. The generator-blind holdout is the only asset that closes this
gap, and these are the only measurements that use it.

Downstream classification is reported as a secondary endpoint and as
corroborating evidence rather than as the basis of a claim.

## B6. Statistical treatment

Evaluation is performed at the eye level. The holdout's 207 healthy files
represent 197 healthy eyes, and treating files as independent observations would
understate uncertainty; predicted probabilities are therefore averaged within an
eye and confidence intervals obtained by bootstrap over eye clusters.

Power was established by simulation rather than by formula. The minimum
detectable difference in eye-level AUC at 80% power lies between 0.043 and
0.070, depending on the assumed within-eye correlation. The study is accordingly
able to detect only large downstream effects, and a null result on the
classifier endpoint is reported as inconclusive rather than as evidence of
equivalence. The simulation additionally shows the percentile bootstrap to be
anti-conservative at this sample size, with empirical type I error between 0.067
and 0.090 against a nominal 0.05; interval estimation therefore uses the
bias-corrected and accelerated variant.

Claiming that two arms are equivalent requires the 95% confidence interval of
their difference to lie entirely within a prespecified margin, and that margin is
fixed from the seed-to-seed spread of the oversampling arm before any result
from the synthetic arm is examined. A wide interval that excludes neither zero
nor the margin is reported as inconclusive, which is an admissible outcome of
this design rather than a failure of it.

## B7. Limitations

One generator was trained per arm. Sampling many images from a single checkpoint
replicates the act of sampling and not the act of training, so this study cannot
separate a genuine effect from a favourable initialisation or trajectory, and
the instability described in B3 shows how readily two matched runs diverge. The
claim is scoped to a generator trained under the published protocol.
Generator-level replication is the first extension a follow-up should purchase.

Patient identity is unavailable in the source dataset, so contralateral eyes of
one individual may fall on opposite sides of the partition. Eye is the finest
unit that can be established reliably.

The holdout is generator-blind but has been read before, having been used to
quantify the colour shortcut and to motivate the present design. Downstream
inference on it is exploratory, and it is described as a generator-blind reused
holdout rather than as an untouched test set.

External validation is absent. Every available route requires correspondence
proceeding on another party's timetable, and the six-page limit could not
accommodate the result in any case.

## B8. Phrasing conventions

The following are the wordings to use, and the ones they replace. Each entry
exists because the shorter form asserts something the design does not support.

| do not write | write instead |
|---|---|
| the two generators are of equal quality | each generator fits its own training distribution to a comparable degree |
| matched seed, therefore matched training runs | matched in initialisation and in every hyperparameter |
| recall improved from 0.027 to 0.063 | recall covers between two and seven of the 111 real images and is not resolved at this denominator |
| the generator adds no information | the difference falls within the prespecified margin, or, the interval is inconclusive |
| FID 50.4 indicates good sample quality | FID is reported as a within-run trend; its absolute value is not interpretable at 139 reals |
| approximately 3 x 10^7 generator parameters | approximately 2.5 x 10^7 generator parameters |
| the holdout is untouched | the holdout is generator-blind and reused |
| leakage inflates performance | generator-mediated leakage is measured here and reported separately from the evaluation-protocol leakage of the prior work |

Two distinctions must be maintained wherever leakage is discussed, since
conflating them is the easiest error a reader could make. **Evaluation-protocol
leakage** describes synthetic images entering validation folds, which is the
mechanism operating in the prior study and which has already been measured.
**Generator-mediated leakage** describes a generator trained on the evaluation
eyes themselves, which is the mechanism `g_leak` isolates. They are distinct,
they are measured separately, and the manuscript should name which one it means
in every sentence that uses the word.

## B9. Outcomes, and why each is reportable

| outcome | how it is stated |
|---|---|
| the synthetic arm matches oversampling | the reported benefit of synthetic augmentation is attributable to rebalancing together with evaluation contamination, a generator of 2.5 x 10^7 parameters having been matched by duplication |
| the synthetic arm exceeds oversampling | the generator contributes information beyond rebalancing, demonstrated against the control that would otherwise have explained the result away |
| the synthetic arm falls below oversampling | synthetic augmentation performs worse than duplication in this regime |
| inconclusive | reported as such, accompanied by the power simulation indicating the cohort size that would be required |

The decomposition of the prior result is reported in every branch, since it is
already measured and depends on none of the above.

---

## Part C — Results obtained 2026-08-25

Three findings were added on this date: a GPU session that put both generators
on a common reference and completed the `g_clean` metric curves, and a
pixel-space run of `s06` on the resulting pools. Numbers:
`outputs/s08_pod_metrics_2026-08-25.json` and
`outputs/s06_pixel_saturation_2026-08-25.json`.

### C1. F1 resolves to a null, and the metrics disagree

Each generator had only ever been scored against its own training archive, and
those archives differ in 28 of 111 images. Both selected checkpoints were
therefore rescored against the other arm's archive, yielding a complete 2 x 2.

| reference | metric | `g_clean` (2822) | `g_leak` (3000) |
|---|---|---|---|
| `clean_111` | FID | 50.86 | 51.50 |
| `clean_111` | KID | 0.018538 | 0.018364 |
| `clean_111` | precision | 0.603 | 0.553 |
| `leak_111` | FID | 52.02 | 50.42 |
| `leak_111` | KID | 0.019484 | 0.018800 |
| `leak_111` | precision | 0.516 | 0.494 |

The three instruments do not agree on an ordering. FID places each generator
ahead on its own reference; KID places `g_leak` ahead on both; precision places
`g_clean` ahead on both. The separations, 1.7 x 10^-4 in KID and 0.64 in FID,
fall inside what a 111-image reference resolves, and recall is uninformative
throughout, every value corresponding to between three and six images.

This is reported as a null. Contamination produced no detectable difference in
distributional sample quality. It is not reported as equivalence, and no
generator is described as better. The consequence for the study is substantive
rather than presentational: the contamination contrast cannot rest on FID or
KID, and must rest on `s06` and `s05`, which are the analyses capable of
observing memorisation and downstream utility respectively.

### C2. The instability at 806 kimg now has its control

`g_clean` was scored with KID and precision/recall at 403, 806, 1209 and 1612
kimg, where only FID had previously been computed.

| kimg | `g_clean` KID | `g_leak` KID | `g_clean` precision | `g_leak` precision |
|---|---|---|---|---|
| 403 | 0.028187 | 0.027100 | 0.428 | 0.337 |
| 806 | 0.024830 | 0.067300 | 0.474 | 0.0003 |
| 1209 | 0.022725 | 0.023100 | 0.524 | 0.393 |

At 806 kimg the contaminated arm degraded on all three metrics while the matched
clean arm improved on all three. The runs differ only in the path to the
training archive, 58 optimisation parameters being identical. The ADA
augmentation probability in `g_leak` had fallen to 0.000 by 673 kimg and
remained near zero until approximately 830, which is the proximate mechanism.
`g_leak` recovered by 1209 kimg, so this is described as an instability episode
rather than a failed run.

`GAN_SEEDS = [0]`. One seed per arm supports a statement of co-occurrence and
does not establish causation, and the limitation is stated wherever the episode
is reported.

### C3. Pixel space cannot see memorisation, and the reason is measurable

`s06` was run in the pixel feature space on all four pools. The Mann-Whitney
statistic has a mathematical ceiling of 9.0477 at these pool sizes, reached when
every synthetic image is farther from the training set than every holdout real.

| pool | Z_U | share of ceiling | C_T | authenticity |
|---|---|---|---|---|
| `synth_clean` psi 1.0 | 8.9464 | 98.9% | 5.02 | 1.000 |
| `synth_clean` psi 0.7 | 8.9598 | 99.0% | 5.06 | 1.000 |
| `synth_leak` psi 1.0 | 8.9476 | 98.9% | 4.94 | 1.000 |
| `synth_leak` psi 0.7 | 8.9551 | 99.0% | 4.79 | 1.000 |

All four sit at 99% of the ceiling; synthetic images are a median 2.7 times
farther from the training reals than holdout reals are. The statistic is
saturated rather than passing, and authenticity is likewise at its maximum of
1.000 with zero inauthentic samples.

The decisive observation is that `g_leak`, whose training archive contains all
28 holdout eyes, is indistinguishable from `g_clean`, the two differing by 0.001
in Z_U. A feature space in which a generator that has memorised the holdout
scores identically to one that has never seen it has no sensitivity to the
quantity being measured. This is empirical support for the decision that `s06`
raises rather than substituting a feature space: the primary endpoint requires
RETFound, and a pixel-space or Inception-space result would not be a weaker
version of it but an uninformative one.

---

## Immediate actions arising

1. Restore the `synth_leak` arm to `config.ARMS` and reinstate contrast P2 in
   `README.md` and `FINAL_PLAN.md` (**F5**). Outstanding; this decision gates
   the `s05` sweep, which re-runs in full if the arm is added afterwards.
2. ~~Correct the selection metric to KID~~ and ~~the snapshot cadence to 400
   kimg~~ (**F6**, **F7**) — done 2026-08-25.
3. ~~Decide **F1**~~ — resolved as a null, see C1. The narrower B8 phrasing is
   adopted and no rescoring rental is required.
4. Adopt BCa intervals or a permutation test in place of the percentile
   bootstrap (**F9**). Outstanding.
5. ~~Correct the generator parameter count to 2.5 x 10^7~~ — done in
   `README.md` 2026-08-25; the comments in `config.py` still read 3e7 and are
   held pending approval to edit that file.
6. Obtain the RETFound checkpoint, which remains the sole blocker on the
   primary endpoint, and which C3 shows cannot be worked around.
