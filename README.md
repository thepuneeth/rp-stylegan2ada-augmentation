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
| **P1** | **`synth_clean` vs `oversample`** | what the generator adds *beyond rebalancing* | small or null | **confirmatory** |
| **P2** | **`synth_leak` vs `synth_clean`** | generator-mediated leakage: did training the generator on the holdout help | small or null | **confirmatory** |
| S1 | `oversample` vs `real_only` | class balance alone | most of the prior gain | descriptive |

**P1 is the primary contrast** — the direct form of "does the GAN beat
oversampling", which the prior work never asked and a reviewer asks first. P2
tests whether a deliberately contaminated generator gains anything. Both are run
by `s12_contrasts.py` with BCa cluster bootstrap and permutation; S1 is
descriptive and carries no multiplicity correction.

### Why the prior result needs decomposing

Three problems, all readable in that paper's own methodology, none in its
limitations:

1. **Synthetic images entered the validation folds.** They were merged into the
   dataset *before* the 5-fold split, so ~177 of ~205 validation positives per
   fold are synthetic images from a generator trained on the real positives.
2. **Balancing changed two things at once** — adds synthetic data *and* corrects
   a 7.4:1 class ratio. No oversampling arm to separate them.

### What makes this answerable

**1. The holdout is generator-blind.** 139 real RP images (134 unique eyes)
split eye-disjoint into 111 the generator trains on and **28 eyes no generator
ever sees**. Any advantage measured there cannot be memorisation.

**2. Two generators, matched recipe.** `g_clean` never sees the holdout;
`g_leak` is trained on 28 holdout eyes plus 83 dev eyes, N matched at 111, under
a recipe identical to `g_clean` in 58 of 61 `training_options.json` keys. The
pair separates evaluation-protocol leakage from generator-mediated leakage —
whether training the generator itself on the holdout confers any advantage.

## The arms

`config.ARMS`, in full:

| arm | training set addition | role |
|---|---|---|
| `real_only` | -- | floor |
| **`oversample`** | minority duplicated to balance | class balance, zero new pixels |
| **`synth_clean`** | StyleGAN2-ADA (~2.5e7 params), ffhq256 -> 111 dev eyes | the generator as normally used |
| `synth_leak` | the size-matched CONTAMINATED generator (28 holdout eyes swapped in) | opt-in, contrast P2; not in `config.ARMS` |

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
| s00 verification | **done** | all invariants pass, incl. hash self-test |
| s01 unique eyes | **done** | 139 files = **134 eyes**; 1024 healthy = 963 |
| s02 eye-disjoint split | **done** | 111 dev / **28 blind holdout**; prevalence matched to 4 decimals |
| s04 GAN data prep | **done** | `clean_111.zip`, `leak_111.zip` |
| GAN training | **done** (GPU) | `g_clean`, `g_leak`, RTX 4090, 3,000 kimg |
| s05 classifier | **done** | 4 arms, eye-level; `outputs/s05_classifier_main5.json` |
| s09 SWD | **done** | vs noise floor and real-to-real anchor |
| s11 Grad-CAM | **done** | attention on the retina, above the disc-area null |
| s12 contrasts | **done** | P1 + P2, BCa cluster bootstrap + permutation |
| external RP test set | **blocked** | 6 images; needs human correspondence on someone else's timetable |

Snapshot selection ran offline on downloaded snapshots: lowest **KID**
(`config.SELECT_METRIC = kid50k_full`) among snapshots >= `SELECT_MIN_KIMG`,
ties to the earlier snapshot. KID rather than FID because FID's bias grows as
the reference set shrinks and this one is 111 images.

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
  s00_verify.py               invariants over splits, zips and hashes
  s01_unique_eyes.py          eye identity via FOV-crop correlation + complete linkage
  s02_make_splits.py          eye-disjoint holdout, assertion-guarded
  s04_prepare_gan_data.py     clean/leak image folders
  s05_classifier.py           downstream arms on the blind holdout
  s09_swd.py                  sliced Wasserstein distance, real-pixel realism check
  s10_figures.py              paper figures
  s11_gradcam.py              attention check: is the model on the retina?
  s12_contrasts.py            P1 + P2, BCa cluster bootstrap + permutation
  stats_boot.py               BCa cluster bootstrap + permutation helpers
splits/     eye_labels.json, holdout.json, dev.json, dev_folds.json
gan_data/   clean_111.zip, leak_111.zip   (gitignored; regenerate with s04)
outputs/    every result as JSON
docs/
  FINAL_PLAN.md         the design of record — arms, budget, endpoints
  AUDIT_LOG.md          what was checked, what failed, what changed  <-- read this
  RUNPOD_RECIPE.md      GPU commands, snapshot integrity, the numbers to beat
  METHODS_NARRATIVE.md  audit of record + manuscript prose
  PRETRAIN_CORPUS.md    stage-2 licensing. SUPERSEDED — stage 2 is dropped
```

`common.py` exists because `fov_crop` and the correlation feature were copied
across scripts. Any distance compared to the duplicate threshold is only
comparable if that feature is byte-identical, so one definition makes it
structural. Verified behaviour-preserving: s01 re-run after the extraction
reproduces `outputs/s01_unique_eyes.json` and `splits/eye_labels.json`
byte-for-byte.

---

## An error caught during setup

Recorded because it would have produced a wrong paper, and it was caught by
checking rather than by assuming.

**The duplicate audit was an artifact.** The earlier
`corrected_experiments/outputs/duplicate_audit.json` reported 132 exact aHash
collisions and grounded an "effective-N collapse" claim. aHash reduces an image
to an 8x8 binary pattern and collides trivially across same-modality images.
Under a metric calibrated to recover five independently confirmed duplicate
pairs with zero false positives, real duplication is **5 pairs among 139 images
(3.6%)**. Effective N is 134, not a collapsed number. **"Effective-N collapse"
is withdrawn as a claim.**

---

## Why this survives every outcome

No paper should depend on a result that cannot be guaranteed:

| outcome | headline |
|---|---|
| **`synth_clean` ~= `oversample`** | the reported benefit is rebalancing plus evaluation contamination |
| **`synth_clean` > `oversample`** | the generator adds real information beyond rebalancing, against the control that would have explained it away |
| **`synth_clean` < `oversample`** | synthetic augmentation is actively worse than duplication in this regime |
| **inconclusive** | reported as such, scoped to what 28 blind eyes can resolve |

Non-significance is not equivalence. Claiming "`synth_clean` ~= `oversample`"
requires a prespecified equivalence margin, set from the seed-to-seed spread of
the `oversample` arm before any `synth_clean` result is read.

---

## Next actions

The pipeline is complete: both generators trained, `s05` swept over four arms,
`s09`/`s11`/`s12` run. The only open item is the **external RP test set** (6
images, blocked on human correspondence).

See `docs/FINAL_PLAN.md` for the full design and prior-work positioning.

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
