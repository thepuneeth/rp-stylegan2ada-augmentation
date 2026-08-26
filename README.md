# Variability study

**The prior study balanced an RP dataset with 885 synthetic images and reported
AUC 0.89 -> 0.97. The gain was real. This study asks where it came from.**

> Scope is fixed by `config.py` and `docs/FINAL_PLAN.md`: **four arms, one new
> generator, <=20 GPU-hours.** An earlier six-arm / eight-run / three-seed draft
> is superseded — see *Dropped* below before reusing anything written against it.

---

## The question, made testable

Every arm trains on the same 111 real RP images and is scored on the same 28
**generator-blind** eyes.

| # | contrast | isolates | expectation | role |
|---|---|---|---|---|
| **P1** | **`synth_clean` vs `oversample`** | what the generator adds *beyond rebalancing* | small or null | **confirmatory — the only one** |
| S1 | `oversample` vs `real_only` | class balance alone | most of the prior gain | descriptive |
| S2 | `pca_sampler` vs `synth_clean` | does a 22-direction linear subspace match ~2.5e7 weights? | — | descriptive |

**P1 is the primary contrast** — the direct form of "does the GAN beat
oversampling", which the prior work never asked and a reviewer asks first.
S1 and S2 are descriptive and carry no multiplicity correction.

### Why the prior result needs decomposing

Three problems, all readable in that paper's own methodology, none in its
limitations:

1. **Synthetic images entered the validation folds.** They were merged into the
   dataset *before* the 5-fold split, so ~177 of ~205 validation positives per
   fold are synthetic images from a generator trained on the real positives.
2. **Balancing changed two things at once** — adds synthetic data *and* corrects
   a 7.4:1 class ratio. No oversampling arm to separate them.
3. **A source shortcut solves most of the task** — six colour moments reach
   **AUC 0.9501** on the generator-blind holdout.

### What makes this answerable

**1. The holdout is generator-blind.** 139 real RP images (134 unique eyes)
split eye-disjoint into 111 the generator trains on and **28 eyes no generator
ever sees**. Any advantage measured there cannot be memorisation.

**2. Novelty is measured on the images, not inferred from accuracy** — in
**RETFound** space (retinal foundation model), not ImageNet Inception, because
Inception features encode ImageNet semantics rather than bone-spicule
pigmentation or arteriolar attenuation. Primary metrics are Alaa et al.
(ICML 2022) alpha-precision / beta-recall / **authenticity** and Meehan et al.
(AISTATS 2020) **C_T** copying test — both established, both with proper nulls.

Novel Coverage Gain is retained in the **appendix only**, exploratory. Its
interval resamples size-matched synthetic subsets from one fixed pool, so it
captures subset-selection variability and not variation across holdout eyes,
training eyes, or generator runs. The `copying_control` (NCG must read exactly 0
on resampled reals) stays as an implementation self-test.

## The arms

`config.ARMS`, in full:

| arm | training set addition | role |
|---|---|---|
| `real_only` | -- | floor |
| **`oversample`** | minority duplicated to balance | class balance, zero new pixels |
| `pca_sampler` | linear manifold samples (k=22, 4.5e6 stored floats) | cheap comparator, free, no GPU |
| **`synth_clean`** | StyleGAN2-ADA (~2.5e7 params), ffhq256 -> 111 dev eyes | the generator as normally used |

Plus one historical row, **not an arm**: `legacy_v3`, the prior generator, which
saw all 139 real RP including every evaluation eye. Labelled contaminated
wherever it appears and supports no clean claim. Costs no GPU — already on disk.

**n = 1 generator.** Sampling thousands of images from one checkpoint replicates
*sampling*, not *training*, so this study cannot separate a real effect from a
favourable initialization. The claim is scoped to "for a generator trained under
the published protocol", stated in limitations rather than concealed by a large
sample size. Generator-level replication is the first thing a follow-up buys.

### Dropped

| dropped | why |
|---|---|
| arm `synth_scratch` | StyleGAN2-ADA from random init on 111 images at 256px is noise at any budget reachable in 20 GPU-hours |
| stage-2 fundus corpus pretrain | not what was achieved — v3 went ffhq256 -> RP directly. Deletes the 4,972-image download and the contamination audit |
| the `train.py:313-315` patch | only mattered when one generator resumed and another did not. The single remaining generator resumes, so the `ada_kimg`/`ema_rampup` shift cancels |
| generator seeds x3 | see n=1 above |
| scarcity sweep | 28 blind eyes cannot resolve a curve |

**Generator-mediated leakage is measured here.** `g_leak` was trained on the
28 holdout eyes plus 83 dev eyes, N matched at 111, under a recipe identical to
`g_clean` in 58 of 61 `training_options.json` keys. At the distribution level
the 2x2 cross-score is near-diagonal (F1), and the one snapshot where the
generators diverge is a training instability the ADA augment trace identifies
(F3). At the classifier level the arm exists as `synth_leak`, routed through
`s05 --arms ... synth_leak --synth-leak-dir`; it is deliberately **not** in
`config.ARMS`, because adding an arm after a sweep re-runs the sweep.

---

## Status

| step | state | result |
|---|---|---|
| s01 unique eyes | **done** | 139 files = **134 eyes**; 1024 healthy = 963 |
| s01a metric calibration | **done** | FOV-crop correlation, 5/5 known duplicates, 0 false positives |
| s02 eye-disjoint split | **done** | 111 dev / **28 blind holdout**; prevalence matched to 4 decimals |
| s03 novelty metrics (Inception) | **done, baseline measured** | v3 recall **0.297** — the number to beat |
| s04 GAN data prep | **done** | `clean_111.zip`, `leak_111.zip` |
| s00 verification | **done** | all 20 invariants pass, incl. hash self-test |
| s05 classifier harness | **done, smoke-tested** | `colour_lr` = AUC 0.9501 on holdout |
| **s06 memorisation check** | **built, selftested; pixel space run** | authenticity + Meehan C_T only; fidelity/diversity delegated to built-in `pr50k3_full` |
| eye-level evaluation | **done** | s05 reports image and eye level; `colour_lr` reproduces 0.9501 image / 0.9561 eye |
| `pca_sampler` arm | **done** | `s04b_pca_sampler.py`, k=22, 1,110 images, selftested |
| arm routing in s05 | **done** | all four arms routed; an unrouted arm now raises instead of silently training as `real_only` |
| `dev_folds.json` | **withdrawn** | vestigial. The protocol trains on all of DEV and evaluates on the blind holdout across seeds; it never cross-validates. s05 is right to ignore it |
| power simulation | **built** | `s07_power.py`, `outputs/s07_power.json` |
| GPU run | **running** | `g_clean`, RTX 4090, 3,000 kimg, ~$7 — launched 2026-08-23 |
| external RP test set | **blocked** | 6 images; needs human correspondence on someone else's timetable |

**s06 gates the GPU day.** Without a prespecified snapshot-selection criterion,
14 GPU-hours produce ~15 `.pkl` snapshots with no principled way to choose one.
Selection runs offline on downloaded snapshots: lowest **KID**
(`config.SELECT_METRIC = kid50k_full`) among snapshots >= `SELECT_MIN_KIMG`,
ties to the earlier snapshot. KID rather than FID because FID's bias grows as
the reference set shrinks and this one is 111 images.

### Baseline — legacy v3 generator, size-matched, measured in Inception space

Superseded as a target once RETFound replaces Inception, but retained: it is the
**information-open** reference point, since v3 transferred from FFHQ.

| metric | value | reading |
|---|---|---|
| Inception precision | 0.938 | fidelity is fine |
| **Inception recall** | **0.297** | **spans only 30% of the real manifold — the weakness** |
| novel coverage gain | 0.426 [0.286, 0.536] | beyond copying, but *not* better than the real training set |
| pixel-space NCG | 0.041 [0.000, 0.107] | no structural novelty at all |
| memorisation | 0 / 885 flagged | clean — it is not copying |

The diagnosis is specific: v3's generator makes plausible RP-looking images
(precision 0.938, memorisation 0) that cover too little of the disease's
variation (recall 0.297). It is not cheating; it is narrow. **Recall is the
target.**

Note this baseline is an **upper bound** — the v3 generator saw the holdout, so
its novelty figures are contaminated by memorisation. A clean generator must
beat 0.297 while starting from a genuine disadvantage.

---

## Running anything

Scripts need `torch`, `sklearn`, `scipy`, `PIL`. The system `python3` has only
some of them; the project virtualenv has all of them:

```bash
cd variability_study
../venv/bin/python scripts/s00_verify.py      # 20 invariants, read-only
../venv/bin/python scripts/common.py          # shared-primitive selftest
```

`scripts/` is on `sys.path` when a script is run directly, so `from common
import C, ...` resolves without a path hack.

## Layout

```
config.py                     single source of truth; every constant is justified in place
scripts/
  common.py                   fov_crop, corr_features, list_images + selftest  <-- shared
  s00_verify.py               20 invariants over splits, zips and hashes
  s01_unique_eyes.py          eye identity via FOV-crop correlation + complete linkage
  s01a_calibrate_...py        why that metric and not the three alternatives
  s02_make_splits.py          eye-disjoint holdout + repeated CV folds, assertion-guarded
  s03_novelty_metrics.py      NCG, precision/recall/density/coverage, memorisation
  s04_prepare_gan_data.py     clean/leak image folders
  s06_metrics.py              authenticity + Meehan C_T on the blind holdout
  s05_classifier.py           downstream arms on the blind holdout
splits/     eye_labels.json, holdout.json, dev.json, dev_folds.json
gan_data/   clean_111.zip, leak_111.zip   (gitignored; regenerate with s04)
outputs/    every result as JSON
docs/
  FINAL_PLAN.md         the design of record — arms, budget, endpoints
  AUDIT_LOG.md          what was checked, what failed, what changed  <-- read this
  RUNPOD_RECIPE.md      GPU commands, snapshot integrity, the numbers to beat
  FINAL_PLAN_AUDIT.md   adversarial review of FINAL_PLAN
  PRETRAIN_CORPUS.md    stage-2 licensing. SUPERSEDED — stage 2 is dropped
  LIT_REVIEW_...md      prior-work positioning
```

`common.py` exists because `fov_crop` was copied into three scripts and the
correlation feature into two. s03's distances are only comparable to the
duplicate threshold s01a calibrated if that feature is identical — three copies
made that an unenforced comment. Verified behaviour-preserving: s01 re-run after
the extraction reproduces `outputs/s01_unique_eyes.json` and
`splits/eye_labels.json` byte-for-byte.

---

## Two errors caught during setup

Recorded because both would have produced a wrong paper, and both were caught by
checking rather than by assuming.

**The duplicate audit was an artifact.** The earlier
`corrected_experiments/outputs/duplicate_audit.json` reported 132 exact aHash
collisions and grounded an "effective-N collapse" claim. aHash reduces an image
to an 8x8 binary pattern and collides trivially across same-modality images.
Under a metric calibrated to recover five independently confirmed duplicate
pairs with zero false positives, real duplication is **5 pairs among 139 images
(3.6%)**. Effective N is 134, not a collapsed number. **"Effective-N collapse"
is withdrawn as a claim.**

**The novelty metric was biased by set size, and that bias was mine.** The first
implementation compared a minimum over 885 synthetic images against a minimum
over 111 reals. A minimum over more draws is mechanically smaller, so it
manufactured novelty from sample size alone: Inception NCG read 0.714 with
p=0.0078. After size-matched bootstrapping the same data reads **0.426 with
p=0.71** — nothing. Every reported figure is now size-matched, and the unmatched
value is retained in the output labelled as an artifact so it can never be
quoted by accident.

---

## Why this survives every outcome

No paper should depend on a result that cannot be guaranteed:

| outcome | headline |
|---|---|
| **`synth_clean` ~= `oversample`** | the reported benefit is rebalancing plus evaluation contamination |
| **`synth_clean` > `oversample`** | the generator adds real information beyond rebalancing, against the control that would have explained it away |
| **`synth_clean` < `oversample`** | synthetic augmentation is actively worse than duplication in this regime |
| **inconclusive** | reported as such, with a power simulation showing what cohort size would be needed |

Non-significance is not equivalence. Claiming "`synth_clean` ~= `oversample`"
requires a prespecified equivalence margin, set from the seed-to-seed spread of
the `oversample` arm before any `synth_clean` result is read.

---

## Next actions

1. **Fetch the RETFound checkpoint** (`RETFOUND_WEIGHTS`, CFP variant, ~1.2 GB,
   github.com/rmaphoh/RETFound_MAE). `s06_metrics.py` is built and selftested
   but has only been run in pixel space. `PRIMARY_FEATURE_SPACE = "retfound"`
   and the primary endpoint is beta-recall in that space, so this is the one
   remaining blocker on the headline number. s06 raises rather than falling
   back to Inception, by design. — *do first; licence acceptance is the
   owner's*
2. ~~Move evaluation to the eye level, build the `pca_sampler` arm, make s05
   route all four arms.~~ **Done 2026-08-23.** The `dev_folds.json` half of this
   item is withdrawn: the protocol does not cross-validate, so there is nothing
   to consume.
3. Power simulation script.
4. **Provision GPU**: one generator, `clean_111.zip`, 3,000 kimg. — *needs
   explicit approval*. Confirm UGHS first-author and in-person-attendance
   eligibility before any spend.

See `docs/FINAL_PLAN.md` for the full design, prior-work positioning and
power analysis.

---

## Runtime

Measured on MPS, per classifier training run (30 epochs, ~930-1,630 images):

| backbone | per run | full sweep |
|---|---|---|
| `toycnn` | ~6 min | ~5 h |
| `resnet18` | ~25 min | ~21 h |

Use `--epochs 2 --backbone toycnn` to smoke-test the harness in ~90 s; those
numbers are not results.

GAN training, measured on this project's own `training-runs/*/stats.jsonl` and
not estimated: **21.7 sec/kimg** on the gamma=5 / ADA target=0.6 / freezed=0
recipe, so 3,000 kimg is ~18.1 hours on one Ampere-class GPU. FID50k_full by
cumulative kimg was 49.58 @ 1983, 46.85 @ 3031, 41.69 @ 3806 (best),
43.01 @ 4293 (worse). **FID here is a monitoring signal, not an endpoint** — it
is computed against only 139 reals, so it is valid as a within-run trend and
meaningless as an absolute number.

### One result that already matters

`colour_lr` — logistic regression on six global colour moments — reaches
**AUC 0.9501** on the generator-blind holdout. The documented source shortcut
(every image shares JPEG quantization hash `56535cd0d590` and resolution
2004x1690) survives the new eye-disjoint split intact. Any deep arm that does
not clear 0.95 has learned nothing beyond colour, and no augmentation
comparison among such arms is interpretable. This is why the control arm runs
first in every invocation.
