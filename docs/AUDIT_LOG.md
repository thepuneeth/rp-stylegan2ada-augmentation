# Audit log — variability study

Append-only. Every step records what was checked, what failed, and what changed
as a result. A step that produced a wrong answer stays in the record with its
correction, so the same mistake is not repeated.

---

## S01 — unique eye count

**Question:** how many independent eyes do the 139 RP and 1024 healthy files
represent? Needed because the whole study rests on an eye-disjoint,
generator-blind holdout.

### Attempt 1 — FAILED

Mean absolute difference on uncropped 128×128 RGB, single-linkage at MAE < 12.0
(the threshold inherited from `corrected_experiments/unique_eye_count.py`).

Result: 1024 healthy images collapsed into **10 "eyes"**, one component holding
**842 images**. RP collapsed to 44 eyes with a 63-image component.

Two independent causes, both confirmed:

1. **No separation.** The pairwise MAE histogram rises smoothly from 3.93 with
   no bimodality — 3.5–4.0: 1 pair, 7.0–7.5: 19, 10.0–10.5: 67, 15.0–15.5: 116.
   There is no threshold that splits duplicates from distinct eyes, because
   every fundus from this single camera shares a circular field of view, a dark
   surround and a bright optic disc. That shared template dominates the
   distance, not eye identity.
2. **Chaining.** Single-linkage merges a~b and b~c even when a and c are
   unrelated. With 34,425 sub-threshold healthy pairs this cascaded into one
   giant component.

### Attempt 2 — calibration (`s01a`)

Four candidate features were scored against five duplicate pairs confirmed
independently by the earlier cosine audit (RP5~RP7, RP4~RP6, RP1~RP3,
RP17~RP18, RP113~RP114). Metric: how many *extra* pairs sneak above the weakest
true pair, i.e. false positives at full recall.

| feature | weakest known pair | false positives at full recall |
|---|---|---|
| whole-frame grayscale correlation | 0.9915 | 7 |
| **FOV-cropped grayscale correlation** | **0.9905** | **0** |
| FOV-cropped high-pass | 0.8907 | 1225 |
| FOV-cropped gradient magnitude | 0.9401 | 391 |

Cropping to the illuminated field of view removes the shared black canvas that
inflated every distance. High-pass and gradient — intended to isolate
vasculature — performed *worse*: the high-frequency content is noisier than the
cropped global structure.

Largest gap in the similarity ranking: 0.9986 → 0.9905.

### Attempt 3 — ACCEPTED

FOV-cropped grayscale correlation at 256×256, **complete linkage** (every pair
inside a cluster must exceed threshold, so chaining is impossible by
construction), threshold 0.99.

| | files | unique eyes | largest cluster | multi-image eyes |
|---|---|---|---|---|
| RP | 139 | **134** | 2 | 5 |
| Healthy | 1024 | **963** | 4 | 56 |

All 5 known duplicate pairs recovered (5/5). Threshold sweep is stable:

- RP: 0.995→135, 0.99→134, 0.985→131, 0.98→124
- Healthy: 0.995→989, 0.99→963, 0.985→831, 0.98→610

Source consistency verified: the 139 filenames in
`Eye Disease Image Dataset/Original Dataset/Retinitis Pigmentosa` are identical
to the 139 non-synthetic filenames in `BalancedDataset/Retinitis Pigmentosa`.

### Finding that overturns a prior conclusion

The earlier `corrected_experiments/outputs/duplicate_audit.json` reported **132
exact aHash collisions** and was used to argue that the effective sample size
had collapsed far below 139 — the basis of the "effective-N collapse" claim.

**That was an artifact.** aHash reduces an image to an 8×8 binary pattern, which
collides trivially across same-modality images. Under a metric calibrated to
recover known duplicates with zero false positives, real duplication in RP is
**5 pairs among 139 images (3.6%)**, and effective N is **134**, not a
collapsed number.

Consequence: "effective-N collapse" is **not** a defensible paper claim and is
withdrawn. It survives only as a one-line methods statement — folds are
eye-disjoint, and 139 files represent 134 eyes.

This does not weaken the study. It strengthens it: 134 independent RP eyes is a
healthier base for a generator-blind split than a collapsed count would have
been.

---

## S02 — eye-disjoint split

Built and assertion-guarded. No failures.

| | files | eyes |
|---|---|---|
| RP holdout (generator-blind) | 28 | 28 |
| RP dev (GAN training set) | 111 | 106 |
| Healthy holdout | 207 | 197 |
| Healthy dev | 817 | 766 |

Prevalence: cohort 0.1195, holdout 0.1191, dev 0.1196.
25 folds (5 repeats x 5) inside DEV, 21–24 RP per validation fold, no eye
spanning train and val in any fold.

Healthy holdout is excluded from the GAN pretraining corpus as well as RP
holdout. A generator that had seen holdout negatives could bias the negative
class as readily as the positive one.

---

## S03 — novelty metrics, and a bias I introduced

### Dry run on the legacy v3 synthetic pool

Run before spending any GPU, to validate the apparatus and to obtain the leaky
comparison arm for free. The v3 generator was trained on all 139 real RP
including the 28 holdout eyes, so its novelty figures are an UPPER BOUND
contaminated by memorisation, never evidence of genuine novelty.

### FAILURE — set-size bias in Novel Coverage Gain

First implementation compared `d_synth`, a minimum over **885** synthetic
images, against `d_train`, a minimum over **111** reals. A minimum over more
draws is mechanically smaller, so the metric manufactured novelty out of sample
size alone.

| | unmatched (wrong) | size-matched (correct) |
|---|---|---|
| Inception NCG | 0.714 | **0.426** [0.286, 0.536] |
| binomial p | 0.0178 | **0.714** |
| Wilcoxon p | 0.0078 | **0.952** |

The "significant novelty" was entirely artifact. Fixed by bootstrapping
synthetic subsets of exactly |R| = 111 images, 500 replicates. The unmatched
value is still written to the output, explicitly labelled an artifact, so it
cannot be quoted by accident.

Two safeguards added:

- **`copying_control`** — replaces the synthetic set with a bootstrap resample
  of the training reals. Copies cannot be closer to an unseen eye than the
  originals, so NCG must be exactly 0. Returns 0.000 in both feature spaces;
  the run asserts and aborts otherwise.
- **Two separate tests, never conflated.** `beyond_copying` asks whether the CI
  excludes 0 (is there any novelty at all). `better_than_real_train` asks
  whether it excludes 0.5 (does synthetic serve unseen eyes better than the
  real training set). Very different claims.

### Baseline result — the number to beat

| metric | pixel space | Inception space |
|---|---|---|
| novel coverage gain | 0.041 [0.000, 0.107] | 0.426 [0.286, 0.536] |
| beyond copying (CI excludes 0) | **no** | yes |
| better than real train (CI excludes 0.5) | no | **no** |
| precision vs train | 0.072 | 0.938 |
| **recall vs train** | 0.081 | **0.297** |
| coverage vs train | 0.270 | 0.964 |
| coverage vs holdout | 0.500 | 1.000 |
| intra-set diversity, synth / real | 1.636 | 0.940 |
| memorisation flagged | **0 / 885** | — |

Diagnosis: the v3 generator produces plausible RP-looking images
(precision 0.938) and is demonstrably **not copying** (0 of 885 flagged at the
duplicate threshold calibrated to zero false positives), but covers only ~30%
of the real RP manifold. It is narrow, not dishonest. **Recall 0.297 is the
target.**

The two feature spaces disagree, and that is itself informative: high pixel
diversity (1.636) with very low pixel precision (0.072) means the synthetic
variation is not anatomically structured. It varies in ways that do not
correspond to real vessel and disc geometry.

---

## S00 — pipeline verification, and two vacuous checks

A standing verifier for all 20 invariants. It caught two bugs in itself before
it caught anything in the data — both of the same dangerous kind.

**Bug 1.** `img_hashes` compared raw originals (2004x1690) against s04's output
(square-padded 256x256 PNG). No digest could ever match, so
`clean_111: contains NO holdout image` **passed vacuously** — it would have
passed on a real leak.

**Bug 2.** After adding square padding, hashes still could not match: the
reference path went 2004 -> 64 while the pipeline path went
2004 -> 256 (LANCZOS) -> 64. Different resampling chains, different pixels,
different MD5.

Fixed by replicating s04's transform exactly, and by adding a **self-test that
proves the hash can detect a match before any leak check is trusted**. A
leak-detection check that cannot detect a leak is worse than no check, because
it manufactures confidence.

Final state: **all 20 checks pass**, including the self-test, so the negative
results (`clean_111` contains no holdout image; `pretrain_fundus` contains no
holdout image) are meaningful rather than vacuous.

---

## Stage-2 corpus — licensing resolved

The Eye Disease Image Dataset (Mendeley DOI `10.17632/s9bfhswzjb.1`) is
**CC BY 4.0**, verified at source on 2026-07-30. It has ten classes totalling
5,335 originals; only Healthy (1,024) and RP (139) were ever downloaded.

Roughly **4,172 further fundus images are available under a license this
project has already accepted**, no authentication and no scraping. That
unblocks staged domain-adjacent pretraining, which directly targets the
recall 0.297 weakness.

Caveat recorded in `PRETRAIN_CORPUS.md`: same hospitals, same camera, so this
will teach fundus anatomy but will not break the documented source shortcut
(colour moments alone reach AUC 0.947). Pterygium is excluded as anterior
segment rather than fundus.

---

## Redesign — from "does the GAN add variability" to "where does the help come from"

Triggered by a literature audit against research practice. Three findings forced
the change, and a fourth correction restored scope that had been cut too hard.

### Finding 1 — the metric was reinvented

Two peer-reviewed frameworks already cover most of what NCG was built for:

- **Meehan et al., AISTATS 2020** — three-sample data-copying test using
  training, held-out and generated samples, Mann-Whitney U on nearest-neighbour
  distances. Has a proper null distribution.
- **Alaa et al., ICML 2022** — alpha-precision, beta-recall and **authenticity**,
  the standard three-axis framework; authenticity is a per-sample "is this a
  copy" score.

**Resolution: report all three.** They answer *are samples too close to
training*; NCG asks *are samples closer to unseen real eyes than training
samples are*. Copying-absence and novelty-toward-the-true-manifold are different
questions. A new metric validated against established ones is stronger than a
new metric alone.

### Finding 2 — the feature space had no clinical meaning

Inception encodes ImageNet semantics; FOV-cropped pixel correlation encodes
field-of-view geometry. **Neither represents bone-spicule pigmentation,
arteriolar attenuation or waxy disc pallor.** "Covers 30% of the RP manifold"
was a statement about ImageNet space.

Published critique states it directly: synthetic images can "increase visual
diversity without meaningfully expanding coverage of clinically relevant concept
combinations."

**Resolution: RETFound** (ViT-L MAE, 1.6M retinal images, Nature 2023) becomes
the primary feature space. Inception is retained for comparability only.

### Finding 3 — the real gap is the information source, not the generator

**An Information-Theoretic Criterion for Efficient Data Synthesis**, arXiv
2605.16379 (2026, ~2 months old): the data processing inequality applies to
**information-closed** pipelines. Synthetic data can only add value when the
pipeline is *not* closed -- when external signals shape it.

This reframes everything. A generator trained only on 111 images is closed and
cannot add information. A generator initialised from pretrained weights is open.
**Every prior study varies generator capacity; none varies the information
source.** That is the gap, and the criterion explaining it is untested in
medical imaging.

New design: an eight-arm ladder separating **capacity** (oversample vs PCA
sampler vs scratch-GAN) from **information** (scratch-GAN vs transfer-GAN), plus
the decisive control -- routing the same corpus **directly into classifier
pretraining with no generator at all**.

### Correction — scope had been cut too aggressively

Prior use of a method is a citation, not a prohibition. Two items were wrongly
dropped and are restored:

- **Leakage tax.** UCSF *avoided* generator-mediated leakage; nobody has
  measured what it costs. Different contribution, and at ~$1.10 on a 4090 it is
  cheap. Restored.
- **NCG.** Restored as a complement to Meehan and Alaa, per Finding 1.

### Also corrected

- **GPU selection was on $/hr instead of $/kimg.** RTX 4090 is ~5x cheaper per
  kimg than A100 for StyleGAN2-ADA at 256px, which is raw-FP32-bound. Budget
  falls from ~$40 to ~$8. Risk noted: 4090 is sm_89 and the repo pins torch
  1.7.1; a fifteen-minute compile-and-throughput gate is specified before
  committing.
- **kimg budget was padded.** 6,600 -> 5,000 across four runs.

---

# 2026-07-30 — External audit, and the redesign it forced

An independent audit of `FINAL_PLAN.md` (`docs/FINAL_PLAN_AUDIT.md`) returned
"major revision required before GPU spending". Its findings were checked against
the code rather than accepted. Most held.

## Confirmed against source

**`ada_kimg` and `ema_rampup` diverge on resume.** `stylegan2-ada-pytorch-main/train.py:313-315`:

```python
if resume != 'noresume':
    args.ada_kimg = 100      # default is 500 (training_loop.py:111)
    args.ema_rampup = None   # default 0.05
```

The scratch-vs-transfer comparison was advertised as "identical except
initialization". It is not: resume silently changes ADA adaptation speed by 5x
and disables EMA rampup. The audit caught two of these; `ema_rampup` is a third
found while verifying. Any E/F difference measured under the current commands
would have been uninterpretable. **Both must be pinned explicitly in both arms.**

**The pipeline is not information-closed.** `scripts/s05_classifier.py:114` uses
`models.resnet18(weights="IMAGENET1K_V1")`. A scratch generator does not make
the end-to-end pipeline closed.

**Corpus arithmetic wrong.** 101 + 1509 + 127 + 1349 + 444 + 500 + 125 = 4,155
non-RP fundus pathology images (Pterygium excluded, anterior segment). Plus 817
DEV Healthy = **4,972**, not the 4,989 previously stated.

**Internal contradiction.** `FINAL_PLAN.md` specified S2 at 2,000 kimg while the
executable command in `RUNPOD_RECIPE.md` used 3,000.

**`s05_classifier.py` does not implement the advertised protocol.** It trains on
the full DEV set and evaluates repeated runs against the fixed holdout; it never
consumes `dev_folds.json`. Only `oversample` and the legacy `synthetic` arm are
routed; unrecognised arm names fall through to real-data behaviour.

## Accepted without dispute

- **n=1 generator per condition is pseudoreplication.** Thousands of samples
  from one checkpoint replicate sampling, not training. Fixed: 3 seeds each for
  E and F, seed as the unit of inference.
- **Evaluation unit mismatch.** 207 Healthy holdout files represent 197 eyes.
  Eye-level colour control reads AUC 0.9561, cluster bootstrap 95% CI
  [0.9204, 0.9850]. All evaluation moves to eye level.
- **The analytic MDE of 0.046 is not reproducible.** ρ=0.8 was assumed without
  justification. Replaced by a paired cluster-bootstrap simulation.
- **NCG's interval is not a confidence interval.** It resamples size-matched
  synthetic subsets from one fixed pool, capturing subset-selection variability
  only — not variation across holdout eyes, training eyes, or generator runs.
  Demoted to appendix, exploratory.
- **Non-significance is not equivalence.** The four-outcome table treated
  apparent equality as a result. An equivalence margin and an explicit
  *inconclusive* outcome are now prespecified.
- **The holdout is reused.** It was used to quantify the colour shortcut and
  motivate the redesign. Now described as a *generator-blind reused holdout*.

## Rejected

**The audit's resource tradeoff.** It recommended cutting the leakage arm, extra
truncation values, or NCG in order to afford generator seeds. At $0.0011/kimg on
a 4090 that tradeoff does not exist: 3xE + 3xF + F_leak + S2 = 9,000 kimg
~= $10. Cost was never the binding constraint. Six pages is.

## The prior paper's three flaws — the study's motivation

Read directly from the v3 manuscript
(`~/Documents/orginial_paper/*.tex`). None appears in its limitations section.

1. **Synthetic images entered the validation folds.** Synthetic images were
   merged into the dataset *before* the 5-fold split ("added with the positive
   images... to make the dataset balanced", then "partitions the dataset into
   80% for training and 20% for validation"). Roughly 177 of ~205 validation
   positives per fold are synthetic, produced by a generator trained on the real
   positives. The headline sensitivity of 88-97% is measured largely on those.
2. **Balancing confounds two interventions.** It adds synthetic data and
   corrects a 7.4:1 class ratio simultaneously. No oversampling or
   class-weighting arm exists to separate them.
3. **A source shortcut solves most of the task.** Six colour moments reach
   AUC 0.9501 on the generator-blind holdout.

## Redesign — eight arms down to five

The eight-arm information ladder was correct in its controls and unaffordable in
its breadth. Eight arms on a 28-eye endpoint inside six pages yields overlapping
intervals and no space for method detail.

Cut: arm C (traditional augmentation), arm G (classifier pretraining on the
corpus), arm H. G is a different pipeline rather than a controlled comparison;
defining its label access, initialization, objective and budget honestly costs a
page. Recorded as future work.

Cut as headline claims: NCG (appendix) and the data-processing-inequality
argument (one motivating sentence). The DPI prediction that closed-pipeline arms
must be equal does not follow — regularization, class balancing and induced
invariance all improve finite-sample risk without adding population information.
This reverses the earlier "scope had been cut too aggressively" correction for
NCG specifically: the reason is now power and page budget, not precedent.

Kept: `F_leak` (renamed from `G_leak`, which collided with the old arm G).

Primary contrast is now **F vs B** — does the generator add anything beyond
rebalancing — with **F_leak vs F** quantifying the leakage tax.

---

## 2026-08-02 — plan rebuilt on measured throughput; six arms down to four

### The error

The 2026-07-30 plan budgeted **1,000 kimg per generator**. That figure came from
an online estimate and was never checked against this project's own logs.

`training-runs/*/stats.jsonl` says the **earliest FID ever measured in this
project is at 1,983 kimg** (49.58). The entire per-run budget therefore sat
*below the first measured point*, with no evidence anything useful exists there.
FID was still falling ~8 points between 2,000 and 3,800 kimg. Corrected by
re-deriving every GPU number from the logs.

### What the logs say

| run | config | kimg | sec/kimg | hours | ADA aug | signs/real |
|---|---|---|---|---|---|---|
| 00004 | resume ffhq256, γ5, target 0.6, freezed 0 | 290 | 21.7 | 1.80 | 0.0→0.195 | 0.00→0.59 |
| 00005 | continue, γ5 | 2,750 | 21.7 | 16.92 | 0.0→0.32 | 0.91→0.59 |
| 00006 | γ3, target 0.35, freezed 4 | 770 | 28.8 | 6.29 | 0.0→0.604 | 0.89→0.31 |
| 00007 | γ4, target 0.25, freezed 5 | 484 | 27.8 | 3.79 | 0.0→0.924 | 1.00→0.26 |
| | **total** | **4,294** | 24.1 | **28.80** | | |

FID50k_full by cumulative kimg: 49.58 @ 1,983 · 46.85 @ 3,031 · 41.98 @ 3,765 ·
**41.69 @ 3,806 (best)** · 43.01 @ 4,293.

**Metric overhead is not the explanation for 28.8 h.** Splitting
`Timing/total_sec` into summed `Timing/sec_per_tick` plus
`Timing/maintenance_sec`: maintenance totals **0.59 h** across all four runs and
the residual gap is zero. 28.8 h is pure training.

**Reconciliation of the "<20 GPU-hours" figure:** runs 00004 + 00005 alone are
**3,040 kimg in 18.72 h**. The 10.08-hour tail (00006 + 00007) bought only 1,254
kimg because `freezed4`/`freezed5` cut throughput from 21.7 to ~28 sec/kimg —
and its final phase made FID *worse* (41.69 → 43.01).

### The redesign

Hard ceiling **20 GPU-hours for the entire study**. At 21.7 sec/kimg that is
~3,320 kimg total, i.e. **one generator near convergence**.

- **`g_clean`**, 111 dev eyes, resume `ffhq256`, γ5 / target 0.6 / freezed 0
  throughout, **3,000 kimg ≈ 18.1 h**, plus ~0.5 h of post-hoc FID. Expected FID
  ≈47 — the measured value at 3,031 kimg, not a projection.
- Arms reduce to **A `real_only` · B `oversample` · D `pca_sampler` ·
  C `synth_clean`**, plus `legacy_v3` as a labelled contaminated historical row.
- **P1 (C vs B) is the only confirmatory test.** S1 (B vs A) and S2 (D vs C) are
  descriptive.

### Cut, with reasons

| cut | reason |
|---|---|
| stage-2 fundus corpus pretrain | not what was achieved — v3 went ffhq256 → RP directly. Removes the 4,972-image download, the `pretrain_fundus.zip` rebuild, and §12 contamination audit |
| arm E `synth_scratch` | StyleGAN2-ADA from random init on 111 images at 256px produces noise at any budget reachable in 20 GPU-hours |
| **the `train.py:313-315` patch** | it existed *only* because E did not resume while F did, so `ada_kimg` 500→100 and `ema_rampup` 0.05→None applied to one arm. With every remaining run resumed the shift applies uniformly and cancels. **The ugliest implementation item in the project exits at zero cost.** |
| arm F_leak | needs a second matched generator (28 holdout + 83 dev, N matched at 111). No budget. `gan_data/leak_111.zip` is already built and retained |
| 3 generator seeds | n=1, with the scope limit stated rather than concealed |
| scarcity sweep | 28 blind eyes cannot resolve a curve |
| the `freezed` phase schedule | ~30% throughput cost; its final phase made FID worse |

### Two rules added, both free

**Snapshot selection.** The best snapshot was **not the last** (41.69 @ ~3,806,
degrading to 43.01 @ 4,293), so a fixed-kimg endpoint would sometimes hand over
the worse generator. Prespecified: `--snap=50` (every 200 kimg), `--metrics=none`
during training, FID computed after training on candidates ≥ 2,000 kimg,
selection offline on downloaded snapshots, ties to the earlier snapshot.

**Surplus rule.** If the pod's first tick beats 21.7 sec/kimg, the surplus buys
**more kimg on the same generator**, capped at 3,800. Fixed in advance from
throughput so the stopping point is never results-driven.

### Standing caveat recorded

**FID is a monitoring signal here, not an endpoint.** It is computed against only
139 reals: valid as a within-run trend, meaningless as an absolute number, and
not comparable to published FIDs on 70k-image datasets. Primary endpoints remain
Alaa α-precision / β-recall / authenticity and Meehan C_T in RETFound space.

### Also noted

The v3 paper's reported **FID 40.97 appears in no log.** The closest logged value
is 41.69 at ~3,806 cumulative kimg. Already on the concede list in
`DEFENSE_AGENT_BRIEF.md` §6; it now has a source.

### Files rewritten

`docs/FINAL_PLAN.md` · `docs/RUNPOD_RECIPE.md` · `config.py` ·
`docs/copilot_panel.html`. `docs/DEFENSE_AGENT_BRIEF.md` updated (§2, §3 tiers,
§6 concessions 7–8). `docs/FINAL_PLAN_AUDIT.md` left as-is and marked stale on
budget arithmetic.

---

## 2026-08-26 — Scope reduced to essential analyses

The study is trimmed to the analytic vocabulary of the original v3 paper plus the
one new claim, on the owner's instruction to keep only the needed analytics.

**Removed from scope (scripts + outputs deleted):** `s01a` duplicate-metric
calibration, `s03` novelty metrics, `s04b` pca_sampler arm, `s06`
authenticity / coverage / Meehan C_T, `s07` power simulation. Classifier arms
`pca_sampler` and `colour_lr` dropped; `s05` reduced from six arms to four
(`real_only`, `oversample`, `synth_clean`, `synth_leak`). `s12` reduced to the two
prespecified contrasts P1 and P2. The `gan_data/pca_pool` image directory (1,110
images) deleted.

**Endpoint change.** The primary endpoint is no longer image-level authenticity /
β-recall in RETFound space (that apparatus is removed). It is now the downstream
classifier contrasts P1 (`synth_clean` vs `oversample`) and P2 (`synth_leak` vs
`synth_clean`), with FID/KID/SWD and the cross-score F1/F3 as distributional
support and Grad-CAM as the attention check. RETFound is no longer required.

**Living docs rewritten to match:** `README.md`, `RUNBOOK.md`,
`docs/RUNPOD_RECIPE.md`, `docs/PRETRAIN_CORPUS.md`, `docs/copilot_panel.html`,
`docs/FINAL_PLAN.md`, `docs/METHODS_NARRATIVE.md` (Part B; Part C section C3, the
pixel-space `s06` result, removed as out of scope).

**Deleted as superseded:** `docs/JUDGE_REPORT.md`, `docs/FINAL_PLAN_AUDIT.md`,
`docs/LIT_REVIEW_AND_REPOSITIONING_2026-08-18.md`, `docs/DEFENSE_AGENT_BRIEF.md`.

Prior entries in this append-only log are left intact: they record what was
genuinely run on their dates. This entry is the correction, not a rewrite of the
history above it. The `provenance/2026-08-26_pod_s05/` run record (raw pod log +
runner for the six-arm sweep) was deleted on the owner's instruction as a
superseded session log; the result it produced survives as
`outputs/s05_classifier_main5.json`, filtered to the four reported arms, and the
current run command is in `RUNBOOK.md`.
