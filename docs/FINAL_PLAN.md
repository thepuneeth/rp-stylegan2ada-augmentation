# Finalized plan — one generator, 20 GPU-hours

**Venue:** IEEE BIBM 2026 UGHS Symposium · 6 pages incl. references · single-blind

Supersedes the 2026-07-30 six-arm / 9,000-kimg plan. That plan budgeted 1,000
kimg per generator from an online estimate. The project's own training logs say
the earliest FID ever measured in this project is at **1,983 kimg** — the entire
budget sat below the first measured point, with no evidence anything useful
exists there. Every number in this document is now taken from
`training-runs/*/stats.jsonl`, not from an estimate.

---

## 1. What the logs actually say

Four runs, FFHQ256 → 139 RP images, single A5000-class GPU.

| run | config | kimg | sec/kimg | hours |
|---|---|---|---|---|
| 00004 | resume ffhq256, γ5, target 0.6, freezed 0 | 290 | 21.7 | 1.80 |
| 00005 | continue, γ5 | 2,750 | 21.7 | 16.92 |
| 00006 | γ3, target 0.35, **freezed 4** | 770 | 28.8 | 6.29 |
| 00007 | γ4, target 0.25, **freezed 5** | 484 | 27.8 | 3.79 |
| | **total** | **4,294** | 24.1 avg | **28.80** |

FID50k_full by cumulative kimg:

| cum kimg | FID |
|---|---|
| 1,983 | 49.58 |
| 3,031 | 46.85 |
| 3,765 | 41.98 |
| **3,806** | **41.69 ← best** |
| 4,293 | 43.01 *(worse)* |

Three facts follow, and they set this whole design.

**(a) 3,040 kimg costs 18.7 hours.** Runs 00004 + 00005 alone. That is the
"<20 GPU-hours" figure this project remembers, and it is a real measured point,
not a target.

**(b) The last 10 hours bought 5 FID points and then gave 2 back.** The
`freezed` phases dropped throughput 21.7 → ~28 sec/kimg for 1,254 kimg. Half the
GPU time went to the tail.

**(c) The best snapshot was not the last one.** 41.69 at ~3,806, degrading to
43.01 by 4,293. Any fixed-kimg stopping rule would sometimes hand over the worse
generator.

Metric overhead is not the explanation for the 28.8 hours: summed
`Timing/maintenance_sec` is 0.59 h across all four runs and the gap between
total and ticks is zero. 28.8 h is pure training.

---

## 2. Budget

**Ceiling: 20 GPU-hours for the entire study.**

| item | kimg | hours |
|---|---|---|
| `g_clean` training @ 21.7 s/kimg | 3,000 | 18.1 |
| FID50k_full at 2,000 / 2,500 / 3,000 | — | ~0.7 |
| sampling ψ ∈ {0.7, 1.0}, 1,500 images each | — | <0.1 |
| **total** | **3,000** | **≈18.8** |

Expected FID ≈ 47. Not a hope — it is the measured value at 3,031 kimg on the
same recipe.

Everything else in the study runs on the local machine at zero GPU cost.

---

## 3. Design

Two matched generators. Four classifier arms. Every arm trains on the same 111
real RP images and is scored on the same 28 **generator-blind** eyes.
Eye-disjoint, built and verified (`s00_verify.py`).

| arm | training set addition | GPU |
|---|---|---|
| **A** `real_only` | — | 0 |
| **B** `oversample` | minority duplicated to balance | 0 |
| **C** `synth_clean` | **the one new generator**, 111 dev eyes only | 18.8 h |
| — `synth_leak` | the size-matched CONTAMINATED generator, opt-in | 0 (already on disk) |

`synth_leak` is the second matched generator (trained on 28 holdout + 83 dev
eyes). It is wired but not in `config.ARMS`, added on the command line for
contrast P2.

### Prespecified contrasts

| # | contrast | isolates | prior expectation |
|---|---|---|---|
| **P1** | **C vs B** | what the generator adds *beyond rebalancing* | small or null |
| **P2** | **`synth_leak` vs C** | generator-mediated leakage: did training on the holdout help | small or null |
| S1 | B vs A | class balance alone | most of the prior gain |

**P1 and P2 are confirmatory.** P1 is the direct version of "does the GAN beat
oversampling" — the question the prior work never asked and the reviewer asks
first. P2 tests whether a deliberately contaminated generator gains anything.
S1 is a descriptive, unadjusted interval, labelled as such.

### The leakage result

The prior study's inflation came from **evaluation-protocol leakage**: synthetic
images were merged into the dataset *before* the 5-fold split, so each validation
fold held roughly 177 synthetic positives out of ~205, produced by a generator
trained on the real positives. Removing synthetic images from the validation
folds and rerunning collapses the effect —
`corrected_experiments/outputs/corrected_cv_summary.json`, synthetic ≈
oversampling. That is the paper's leakage result and it cost nothing.

**Generator-mediated leakage** — training the generator itself on the holdout
eyes — is a separate mechanism, and it **is now measured**. The second matched
generator was trained (`g_leak`: 28 holdout + 83 dev, N matched at 111, and 58
of 61 `training_options.json` keys identical to `g_clean` — the two runs differ
only in `--data`). Two results follow from it.

*Distribution level (F1).* Each generator is scored against each 111-image
reference set, giving a 2x2. If training on the holdout conferred an advantage,
`g_leak` would score better against `leak_111` than `g_clean` does, by more than
the reference sets differ from each other. It does not: the KID gap between the
two generators on a fixed reference is 0.00017, an order of magnitude under the
gap induced by swapping the reference set. The 2x2 is reported in full, since
the near-diagonal structure is the result.

*Control (F3).* The one snapshot where the two generators genuinely diverge is
806 kimg, where `g_leak` precision falls to 0.0003 and its KID rises to 0.0673.
This is not leakage; it is a training instability, and the ADA augment
probability collapsing in the same window identifies it as such (Fig. 1). The
`g_clean` backfill at the matched kimg is what licenses that reading, which is
why the backfill was run.

The classifier-level counterpart is the `synth_leak` arm in `s05`, which is
wired but **not** in `config.ARMS` — see §Arm decision.

### Arm decision — `synth_leak`

`s05` sweeps `config.ARMS`. An arm added after a sweep re-runs the whole sweep,
so the arm list is fixed once, before launch, and the decision is recorded here
rather than inferred from a command line.

**Include the arm.** It costs 70 fits to 100, no GPU, and it is the only thing
that carries the generator-mediated-leakage question across to the endpoint a
clinical reader cares about. F1 and F3 establish that `g_leak` and `g_clean` are
indistinguishable *as distributions*; without `synth_leak` the paper cannot say
whether that indistinguishability survives into downstream accuracy, which is
precisely the inference the prior work got wrong. A reviewer who accepts the
distribution result will ask the classifier question next, and the answer costs
one flag decided now versus a full re-run decided later.

The arm carries contrast **P2**, prespecified alongside P1. The primary P1
endpoint remains `synth_clean` versus `oversample` on the blind eyes and stays
that way.

Launch is therefore:

```
python3 scripts/s05_classifier.py \
  --arms real_only oversample synth_clean synth_leak \
  --synth-clean-dir <psi1.0 clean sample dir> \
  --synth-leak-dir <psi1.0 leak sample dir>
```

`config.ARMS` stays at three, so the default sweep remains the prespecified one
and `synth_leak` is visible in the command that produced it.

### Dropped from the previous plan

| dropped | why |
|---|---|
| Stage-2 fundus corpus pretrain | not what was achieved — v3 went ffhq256 → RP directly. Deletes the 4,972-image download and the contamination-audit section |
| arm E `synth_scratch` | StyleGAN2-ADA from random init on 111 images at 256px produces noise at any budget reachable here |
| the `train.py:313-315` patch | only mattered because E did not resume while F did. Both remaining generators resume, so the `ada_kimg`/`ema_rampup` shift hits identically and cancels |
| generator seeds ×3 | n=1. See §6 |
| scarcity sweep | 28 blind eyes cannot resolve a curve |
| the DPI argument | one motivating sentence, scoped to the generator stage, phrased as a hypothesis |

Dropping arm E deletes the ugliest implementation item in the project for free.

---

## 4. Endpoints

### Primary — downstream classification, on the 28 blind eyes

The confirmatory endpoints are the classifier contrasts P1 (`synth_clean` vs
`oversample`) and P2 (`synth_leak` vs `synth_clean`), evaluated at eye level and
inferred by `s12_contrasts.py` with BCa cluster bootstrap and permutation. See
§5 for what this endpoint can and cannot resolve.

### Distributional realism — supporting

The two generators are compared as distributions with FID and KID
(StyleGAN2-ADA built-ins, within-run signals) and **SWD** (`s09`, raw-pixel,
Inception-free). None of these is an endpoint on its own; together with the
cross-score F1/F3 (`s08`) they say whether the clean and contaminated
generators are distinguishable at all.

**FID is a monitoring signal, not an endpoint.** It is computed against only 139
reals, so it is valid as a within-run trend and meaningless as an absolute
number. Never compare it to published FIDs on 70k-image datasets.

### Attention — supporting

Grad-CAM (`s11`) checks that a high AUC reflects attention on the retina rather
than the black border or acquisition artefacts, against the disc-area null.

---

## 5. What the classifier endpoint can and cannot do

Holdout: 28 RP eyes and 207 Healthy **files** representing **197 Healthy eyes**.

- **Evaluate at the eye level.** Average predicted probability per eye. Treating
  207 files as independent controls understates uncertainty. Eye-level colour
  control reads AUC 0.9561, stratified eye-cluster bootstrap 95% CI
  **[0.9204, 0.9850]** — the width of that interval on a *saturated* control is
  the honest picture of available precision.
- **The benchmark is near-saturated.** Even a simple colour baseline reaches AUC
  ~0.95 on this cohort, which shares a JPEG quantization signature and a single
  resolution. An arm that does not clear that has not demonstrated anything
  retinal — which is why Grad-CAM (`s11`) checks where attention actually lands.
- **Patient identity is unavailable** in the source dataset. Contralateral eyes
  may cross partitions. Eye is the finest reliable unit.
- **The holdout is generator-blind but reused.** It motivated this design.
  Describe it as a *generator-blind reused holdout*; downstream inference is
  interpreted against the width of the cluster-bootstrap interval.

Consequence: the classifier contrasts P1 and P2 carry the claim, scoped to what
28 blind eyes can resolve.

---

## 6. Statistical decision rules — prespecified

Fixed before results are seen.

- **Primary contrasts:** P1 (C vs B) and P2 (`synth_leak` vs C). Endpoint:
  eye-level holdout AUC difference, BCa cluster bootstrap + permutation.
- **Equivalence margin:** claiming "C ≈ B" requires the 95% CI of the difference
  to lie entirely within ±δ. δ is set from the B-vs-B resampling spread measured
  before any C result is read.
- **Non-significance is not equivalence.** A wide interval excluding neither zero
  nor δ is reported as **inconclusive** — an admissible outcome of this study,
  not a failure of it.
- **One generator, n=1.** Sampling thousands of images from one checkpoint
  replicates sampling, not training. This study therefore cannot separate a real
  effect from a favourable initialization or trajectory. The claim scopes to
  *"for a generator trained under the published protocol."* Stated in
  limitations, not concealed by a large sample size. Generator-level replication
  is named as the first thing a follow-up should buy.

### Snapshot selection — prespecified, costs nothing

The prior project's best snapshot was not its last. So:

1. `--snap` writes a snapshot every **~400 kimg** (`--snap=50` ticks). The
   as-run cadence was 403 kimg, giving 9 snapshots over 3,000 kimg, not the
   15 an even 200 kimg would have produced.
2. All snapshots are downloaded and verified to load **locally**.
3. Selection runs offline on the local machine, on the saved snapshots — no GPU
   time, no results-driven stopping.
4. Selection criterion: lowest **KID** (`kid50k_full`, `config.SELECT_METRIC`)
   among snapshots ≥ 2,000 kimg (`config.SELECT_MIN_KIMG`). Ties break to the
   earlier snapshot. KID, not FID: FID is a biased estimator whose bias grows
   as the reference set shrinks, and this reference set is 111 images. KID is
   the unbiased small-sample estimator, so it is the defensible choice here.

### Surplus rule — prespecified

Measure sec/kimg from the first tick on the pod. If throughput beats 21.7
sec/kimg, spend the surplus on **more kimg on the same generator**, capped at
**3,800 kimg** — the point where the prior project's FID bottomed at 41.69 — not
on a second generator. Fixed in advance so it is not a results-driven choice.

At 14 sec/kimg the 20-hour budget reaches ~5,000 kimg and the cap binds first.

---

## 7. Outcomes, and why each is publishable

| outcome | headline |
|---|---|
| **C ≈ B** | the reported benefit of synthetic RP augmentation is rebalancing plus evaluation contamination — a ~3×10⁷-parameter generator matched by duplication |
| **C > B** | the generator adds real information beyond rebalancing, measured against the control that would have explained it away |
| **C < B** | synthetic augmentation is actively worse than duplication in this regime |
| **inconclusive** | reported as such, scoped to what 28 blind eyes can resolve |

The evaluation-leakage decomposition (§3) is reported in every branch, because
it is already measured.

---

## 8. Six pages

**In:** the decomposition, four arms (A, B, C + `synth_leak`), classifier
contrasts P1 and P2 with eye-level cluster bootstrap, FID/KID/SWD distributional
realism, Grad-CAM attention, the prespecified decision rules, the leakage
decomposition.

**Out:** external validation, focal loss, the full v3 comparator table (cite
it), the scratch/corpus generators, the scarcity sweep.

**If it still does not fit, cut in this order:** the ψ sweep (keep 1.0) → SWD →
Grad-CAM. Never cut A, B or C — they are the spine.

---

## 9. Go/no-go before any spend

UGHS requires an undergraduate or high-school **first author and primary
contributor**, and at least one student author must **attend and present in
person in Dallas** for inclusion in the proceedings. Confirm both.

---

## 10. External data — out of scope for this submission

Six pages cannot hold it, and every route below needs human correspondence
on someone else's timetable. Stated as a limitation. Longest lead time first:

1. **Iowa StoneRounds** — ~490 images / ~178 patients, CC BY-NC-ND 3.0; site
   states "Do NOT scrape". Email for nonprofit research access.
2. **PhysioNet / BRSET** — begin CITI training and the data-use agreement.
3. **UCSF Sci Rep 2026 authors** — request their 132-image RP cohort.

---

## Sources

- [IEEE BIBM 2026 UGHS Symposium](https://bibm2026-hs.github.io/)
- [Deep learning detection of RP inheritance forms through synthetic data expansion — Sci Rep 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12687845/)
- [Do Synthetic Brain MRIs Reliably Improve Tumour Classification? — arXiv 2605.23094](https://arxiv.org/pdf/2605.23094)
- [Eye Disease Image Dataset — Mendeley, CC BY 4.0](https://data.mendeley.com/datasets/s9bfhswzjb/1)
