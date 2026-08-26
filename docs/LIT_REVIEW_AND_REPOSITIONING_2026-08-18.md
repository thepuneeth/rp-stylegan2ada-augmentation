# Literature review and repositioning assessment

**Date:** 2026-08-18 · **Author:** automated scheduled review (`retina-lit-review`)
**Scope:** independent literature review + critical evaluation of the current
research path (`docs/FINAL_PLAN.md`, 2026-08-02) against the stated goal:
*quantify how much generative methods improve performance in a data-scarce
disease setting*, with a result that is novel, defensible, and interesting.

**Standing constraint respected:** this is analysis only. No code edited, no
script run, no data changed, no GPU provisioned. Every recommendation below is
text awaiting a "go".

---

## 0. Executive verdict

**The plan is scientifically sound and is the right *skeleton*. It is not yet
the strongest *claim*.** Three findings drive this:

1. **The primary endpoint drifted off the research question.** `FINAL_PLAN.md`
   makes β-recall in a RETFound embedding the primary endpoint and demotes
   downstream classification to "corroborating evidence, reported as
   underpowered." But the project's own philosophy is about *performance* in a
   data-scarce disease. A paper whose headline number is a generative-diversity
   metric answers a different question than the one being asked. **Fix: do not
   abandon the downstream endpoint — change its shape from a single
   underpowered point contrast to a paired scarcity curve, which is both better
   powered and a stronger claim.** (§5.1, §6.1)

2. **The binary claim "GAN beats oversampling" was effectively scooped in
   2026 at the top end, and the null was scooped at the bottom end.** EyeDiff
   (npj Digital Medicine, 2026) reports a rare-disease AUROC of 0.919 vs 0.893
   for oversampling — with retinitis pigmentosa named as one of the nine rare
   subtypes. Independently, a StyleGAN2-ADA brain-MRI study (arXiv 2605.23094,
   May 2026) ran essentially the rigorous version of the null and found utility
   is architecture- and ratio-dependent, not guaranteed. **A binary win/null on
   "does synthetic beat oversampling" is now a crowded 2026 result.** What is
   *not* crowded is the quantity: how much, per real image, under what leakage
   discipline. (§3, §4)

3. **The one genuinely unoccupied gap the project is uniquely positioned to
   fill is the leakage tax** — an experimental measurement of how much of the
   reported benefit of generative augmentation comes from the generator having
   seen the evaluation cases. The literature acknowledges this mechanism but
   the only quantitative treatments found are in language models and mass
   spectrometry, not medical imaging. `FINAL_PLAN.md` cut this arm for budget.
   **The stated budget now permits two generators. Restore it — this is the
   single highest-value use of GAN #2.** (§3.4, §6.2)

Recommended repositioning, in one sentence:

> **Not** "does a GAN beat oversampling in a rare retinal disease" (crowded,
> binary, and scooped at both ends), **but** "*what does synthetic data actually
> buy* in a rare retinal disease — measured as a synthetic-to-real exchange
> rate, a leakage tax, and a shortcut-amplification cost, under a
> generator-blind protocol."

That reframing keeps the original philosophy exactly (it is still *"how much do
generative methods help under data scarcity"*), converts a binary into three
quantities, and remains publishable under every outcome including a null.

---

## 1. Venue — read this before anything else

This is the most actionable finding in the document and it is not scientific.

| venue | format | eligibility |
|---|---|---|
| **IEEE UEMCON 2026** (named in the review request) | double-blind, 7 pages regular / 6 pages short | general ubiquitous-computing / electronics venue with an applied-ML track. **No undergraduate track is advertised.** |
| **IEEE BIBM 2026 UGHS** (named in `FINAL_PLAN.md`) | single-blind, 6 pages incl. references | biomedical. Undergraduate or high-school first author required; in-person presentation required. |

**The review request names UEMCON; the plan targets BIBM UGHS. These are
different papers with different eligibility rules.** BIBM UGHS is biomedical and
its scope matches this work far better. **Decide which one this is.** If UEMCON
is genuinely the target, the GPU work in §6 does not fit its shorter runway and
the only honest UEMCON submission is the already-computed, zero-GPU audit
material (§6.4).

---

## 2. What the literature review covered

Searched: StyleGAN2-ADA and transfer learning for fundus/rare retinal disease;
GAN vs oversampling in medical imaging (positive and null results); generative
data leakage and test-set contamination; staged/intermediate-domain pretraining;
diffusion vs GAN under small-data; synthetic-to-real exchange rates and
dose-response saturation; RP-specific deep learning; shortcut learning and
device/colour confounding in fundus imaging; fidelity-vs-utility metric
disconnect; few-shot GAN adaptation and memorization; TSTR protocols; public
fundus corpora licensing; public pretrained fundus StyleGAN checkpoints.

Full source list in §9.

---

## 3. Findings — the six clusters that matter

### 3.1 The direct competitor at the top end: EyeDiff (npj Digital Medicine, 2026)

A text-conditioned latent diffusion model built on Stable Diffusion v1.5,
trained on 42,048 images across 8 datasets and 14 modalities. On a rare-disease
cohort spanning nine subtypes — **retinitis pigmentosa explicitly among them** —
it raises AUROC from 0.871 to 0.919, and **beats traditional oversampling
(0.893)**. It also improves on top of RETFound as a backbone.

**Impact on this project.** The headline "generative augmentation can beat
oversampling for rare retinal disease" is now a published 2026 result in a
high-visibility venue. Any submission claiming that as its novelty will be
asked why it is not simply a weaker replication.

**But the differentiation is real and defensible**, and it must be stated
explicitly in the introduction rather than left for a reviewer to find:

- EyeDiff needs a 42k-image multimodal corpus. **This project's regime is the
  one where that corpus does not exist**: one hospital source, 139 RP images, no
  external RP data beyond 6 images. That is the actual data-scarce setting the
  original research philosophy targets. EyeDiff answers "what can a foundation
  model do", not "what can a lab with 139 images do".
- **EyeDiff does not control generator-mediated leakage.** Neither does any
  other work found. Its generator saw the disease distributions it is later
  evaluated on augmenting.
- EyeDiff reports a *win*, not an *exchange rate*. It does not say how many
  synthetic images equal one real image, or where the benefit saturates.

**Action: cite EyeDiff prominently, concede the top-end result, and claim the
scarce-regime and the leakage-controlled measurement.** Do not pretend it does
not exist — a reviewer who finds it uncited will treat the whole framing as
uninformed.

### 3.2 The direct methodological neighbour: StyleGAN2-ADA class-plane study (arXiv 2605.23094, May 2026)

Twelve StyleGAN2-ADA generators on constrained BRISC-2025 brain-MRI partitions,
1:1 and 1:2 real-to-synthetic ratios, three classifier families (random forest
on InceptionV3 features, compact CNN, MobileViTV2), paired multi-seed testing
with Holm correction.

Results: RF gained nothing; CNN mean gains **did not survive Holm correction**;
MobileViTV2 gained +1.02% absolute (Holm p = 0.0104). Conclusion: *"augmentation
utility was found to be architecture- and ratio-dependent, not guaranteed by
visual fidelity alone."*

**Impact.** This is the closest existing paper to the current plan's design —
same generator family, same "does it actually help held-out classification"
question, same statistical seriousness. Two things follow.

- **A plain null on one architecture at one ratio is now a weaker contribution
  than it was in July.** This paper already established that the answer moves
  with architecture and ratio.
- **It hands over two free axes worth adopting.** (a) *Architecture
  dependence*: `s05_classifier.py` already supports `--backbone` with
  `resnet18` and `toycnn`; running both is zero GPU cost and directly addresses
  the strongest known confound in this result class. (b) *Training efficiency*:
  the study found augmented conditions selected their checkpoint 42–64% earlier.
  That is a real, reportable, zero-cost secondary endpoint that survives a null
  on accuracy — "synthetic data did not raise the ceiling but reached it in
  fewer epochs" is a publishable sentence.

### 3.3 Why GAN and not diffusion — now answerable in one sentence

The obvious reviewer question ("2026, why a GAN?") has a citable answer that
does not sound defensive:

- Akbar et al. (Machine Learning: Science and Technology, 2024): **diffusion
  models memorize training images more than StyleGAN, especially on small
  datasets** — directly the regime here (111 images).
- Akbar et al. also report that **FID and Inception Score do not reliably
  predict downstream task performance**, which independently supports the
  plan's decision to treat FID as a monitoring signal rather than an endpoint.
- Multiple 2023–2025 comparisons (Müller-Franzes et al.; Medfusion) find
  diffusion wins on *diversity/recall* while needing ~1,000+ samples for stable
  convergence.

**Recommended framing:** "In the ≤150-image regime, the model class that
memorizes least is the correct choice, and the memorization literature points
to StyleGAN2-ADA over diffusion. We verify this empirically with Meehan's C_T
data-copying test rather than asserting it." That converts a perceived weakness
into a deliberate, evidenced design decision.

### 3.4 The gap that is genuinely open: the leakage tax

This is the strongest finding of the review.

- The *mechanism* is acknowledged in the literature: training generators on
  disjoint subsets is recommended precisely because training on all real data
  "could lead to an underestimate of generalization error" (Synthetic data, real
  errors, 2023). Per-fold generator retraining appears in isolated protocols.
- The *magnitude* is essentially unmeasured in medical imaging. The only
  quantitative treatments found are outside imaging: test-set contamination
  in generative LLM evaluation (arXiv 2601.04301), and MassSpecGym, which
  reports that a generative model trained on test data shows *"significant
  performance inflation compared to leakage-free approaches."*
- **No medical-imaging paper was found that trains a matched pair of
  generators — identical architecture, identical N, differing only in whether
  the evaluation cases were visible — and reports the difference.**

The project already has both datasets built and verified: `gan_data/clean_111`
(111 dev eyes) and `gan_data/leak_111` (28 holdout eyes + 83 dev, N matched at
111), with `s00_verify.py` passing 20/20 invariants including the positive
hash self-test. **The experiment is one GPU day away and nobody has run it.**

This is where the "wow factor" actually lives — not in a better generator, but
in a number the field currently does not have: *how much of the published
benefit of generative augmentation is contamination?* It is also perfectly
aligned with the original philosophy, because a leakage tax is precisely a
correction to "how much do generative methods help."

### 3.5 The second open gap: shortcut amplification

The project's own audit found `global_colour` alone reaches AUC 0.947–0.956 on
this cohort, and that every image shares one JPEG quantization signature and one
resolution. Colour statistics are also what a GAN transfers most faithfully.

The literature has the two halves but not the join:

- Shortcut learning and device/colour confounding in fundus classification is
  well documented (cross-camera DR validation; mask-of-truth sensitivity work).
- Bias inheritance from generative augmentation is documented in *natural*
  images ("From Fake to Real", ECCV 2024; "Generative models improve fairness of
  medical classifiers under distribution shifts"), where the concern is that
  synthetic data introduces a real-vs-synthetic bias of its own.
- **No work was found that measures whether generative augmentation
  *increases a classifier's reliance on a known source shortcut* in fundus
  imaging.**

This is a zero-GPU experiment on data that already exists, and it is the
mechanistic explanation that makes a null *interesting* rather than
disappointing: if synthetic augmentation raises internal AUC while increasing
colour dependence, the paper has explained *why* synthetic gains do not
generalize, not merely reported that they do not.

### 3.6 The dose/exchange-rate literature supports the reframing

- Sagers et al. (latent diffusion augmentation, arXiv 2308.12453): with 16–32
  real images per class, 10 synthetic per real gave up to **+13.3% accuracy**;
  pushing to 75× was "neither monotonic nor statistically significant." A
  saturating dose-response is the expected shape.
- The optimal synthetic:real ratio **decreases as real training size increases**
  — i.e. the effect this project cares about is strongest exactly in the scarce
  regime and vanishes as data grows. That is a *curve*, and it is the natural
  quantitative object for "how much do generative methods help under scarcity."
- Reported synthetic-to-real exchange rates in the general vision literature
  cluster around an order of ~10:1 and are highly task-dependent — no fundus or
  rare-retinal-disease exchange rate was found.

`s05_classifier.py` already implements a `dose` sweep. The scarcity ladder is
the missing half.

### 3.7 RP-specific prior art

- **Sci Rep 2026 (UCSF)** — 132 fundus photographs, 105 eyes, 53 RP patients;
  VAE latent-mixing expansion; ViT classifier; patient-level split discipline
  (both eyes of a patient in the same split); pooled AUC 0.67 → 0.75 (noise
  expansion) → 0.79 (pairwise combinatorial). Different task (inheritance form,
  not RP detection) and a much weaker generative model, but it is the closest
  RP-specific synthetic-expansion paper and **must be cited and distinguished**.
  Already correctly listed in `FINAL_PLAN.md` sources.
- **MDPI Applied Sciences 2025** — RP classification on 2,798 FAF images with
  integrated-gradients analysis. Establishes that RP classification itself is
  not novel; only the scarce-regime generative question is.
- Older work: pigment-sign segmentation, AI-assisted early RP detection.

**Consequence:** the paper must not present "detecting RP with deep learning" as
any part of its contribution. The contribution is entirely about what synthetic
data does.

---

## 4. Scorecard — does the current plan survive the literature?

| element of `FINAL_PLAN.md` | verdict |
|---|---|
| Eye-disjoint 111/28 generator-blind split, verified | **Keep — this is the paper's backbone and is better than most published practice.** |
| Evaluation-protocol leakage decomposition (already computed, zero GPU) | **Keep.** Concrete, honest, and explains a specific published number. |
| Arms A (real_only), B (oversample), C (synth_clean) | **Keep — the spine.** |
| Colour-shortcut control reported first | **Keep, and extend** (§6.3) — currently used as a validity check; it is worth more as a finding. |
| Alaa α-precision/β-recall/authenticity + Meehan C_T | **Keep.** Correct, current, and answers the memorization question properly. |
| RETFound as the embedding | **Keep** — but note it is **not yet implemented**; `s03_novelty_metrics.py` currently computes Inception-based Kynkäänniemi/Naeem metrics only. This is the largest unimplemented item. |
| FID demoted to monitoring signal | **Keep** — now directly supported by Akbar et al. |
| **β-recall as the *primary* endpoint** | **Change.** Answers a different question than the project's own. (§5.1) |
| **Leak generator dropped to future work** | **Reverse.** Budget now allows 2 GANs; this is the highest-value one. (§3.4) |
| **Scarcity sweep dropped** ("28 blind eyes cannot resolve a curve") | **Reverse, with a correction — the reasoning contains an error.** (§5.2) |
| Stage-2 fundus-corpus pretraining dropped | **Endorse the drop as a core arm**, with one caveat (§5.4). |
| Arm D (pca_sampler) | **Keep if space** — it is a genuinely good "capacity does not equal information" control, and cheap. |
| n=1 generator, acknowledged as pseudoreplication | **Unresolved and unresolvable at this budget — see §5.3 for the honest handling.** |
| Arm E (scratch GAN) dropped | **Endorse.** Correct call; the earlier judge report reached the same conclusion. |

---

## 5. The five weaknesses, in order of severity

### 5.1 The primary endpoint no longer matches the research question

`FINAL_PLAN.md` §4 makes β-recall in RETFound space primary and states the
classifier "is corroborating evidence, not the basis of a claim."

The reasoning that produced this is sound (28 positive eyes cannot support a
well-powered point contrast). The consequence is not. The project's stated
philosophy is about **performance improvement under data scarcity**. A paper
whose confirmatory result is a generative-diversity statistic will be read as a
generative-modelling paper with a weak clinical section, and reviewers at a
biomedical venue will ask the obvious question: *if the classifier cannot tell
the difference, why does β-recall matter?*

There is a good answer to that question — β-recall measures whether the
generator produced new information at all, independent of whether a small
classifier can exploit it — but it is a *supporting* argument, not a headline.

**Recommendation:** keep every generative metric exactly as specified, and
report them as the *mechanism* layer. Move the confirmatory claim back to the
downstream side, but change its shape (§6.1) so it is actually powered.

### 5.2 "28 blind eyes cannot resolve a curve" is not correct as stated

The plan drops the scarcity sweep on this basis. The reasoning conflates two
different sources of uncertainty.

- The **absolute** AUC on 28 positive eyes is imprecise — correct, and the
  measured eye-level colour-control CI [0.9204, 0.9850] shows it.
- The **difference between two arms evaluated on the same 28 eyes** is *paired*.
  The eye-level noise is largely common-mode and cancels. This is exactly why
  the plan itself already specifies paired cluster bootstrap for P1 (C vs B) on
  those same 28 eyes.

A scarcity ladder is a set of paired contrasts on that same shared holdout, plus
repeated eye-disjoint CV inside the 111 dev eyes — which cumulatively supplies
far more evaluation events than 28. **If a single paired contrast is powered
enough to be the primary endpoint, a ladder of paired contrasts is powered
enough to be a curve.** The correct statement is narrower and should be kept:
*absolute* AUC at any single ladder rung has wide intervals, and the curve
should be presented with its uncertainty band rather than as point estimates.

### 5.3 One generator per condition is pseudoreplication, and adding a second condition does not fix it

This was flagged in the July audit and the judge report and remains true. With
two generators used for two *different* conditions (clean and leak), there are
zero replicates of either. An observed clean-vs-leak difference can be leakage,
initialization, or trajectory.

There is no budget-feasible fix. There are three honest mitigations, and all
three should be applied:

1. **Shared initialization and identical recipe.** Both generators resume from
   the same base checkpoint with identical hyperparameters, seed, kimg budget,
   and snapshot-selection rule. This removes initialization as an explanation
   and leaves only trajectory divergence.
2. **Snapshot-level dispersion as a variance floor.** Evaluate the metric at
   the last K snapshots of each run (≥2,000 kimg) and report the within-run
   spread. This is not a substitute for seed replication, but it bounds how
   much of any observed gap could be trajectory noise. It is free.
3. **Scope the claim in the title and abstract.** "A matched-pair case study"
   rather than "we show that leakage inflates performance by X." State
   explicitly that generator-level replication is the first thing a follow-up
   should buy.

Framed this way the result is still worth reporting: no matched pair exists at
all in the medical imaging literature, and a well-controlled n=1 pair with an
honest label is a contribution. An n=1 pair presented as a population estimate
is a rejection.

### 5.4 Dropping stage-2 pretraining is right, but the reasoning left value on the table

Dropping stage-2 as a *core arm* is correct: it costs GPU days, it dilutes the
confirmatory question, and the audit correctly found the FFHQ→corpus→RP path
does not isolate the corpus.

However, the review found **public StyleGAN2-ADA checkpoints pretrained on
fundus images** — the k-SALSA release (ECCV 2022) provides generators trained on
**APTOS** and **EyePACS**. If usable, this converts stage-2 from "3–4 GPU-days
and a licensing audit of a 4,972-image corpus" into "download a checkpoint",
and would likely reach a given FID in far fewer kimg than the FFHQ→RP path the
project measured (FID 46.85 at 3,031 kimg; best 41.69 at 3,806).

**Two frictions make this a spike, not a plan:** the checkpoints are at
**512×512** (the project trains at 256, and StyleGAN2-ADA cannot `--resume`
across resolutions without dropping a synthesis block), and they are distributed
in **rosinality `.pt` format**, requiring reverse conversion to NVIDIA `.pkl`.

**Recommendation: timeboxed spike of at most 3 hours, before the GPU rental,
with a hard fallback to FFHQ256 (status quo).** If it works, both generators get
a better initialization at zero extra GPU cost and the paper gains a free
sentence about domain-adjacent initialization efficiency. If it does not, the
plan is unchanged. **Do not make any claim depend on it.**

Note also: a descriptive comparison of the FFHQ→RP FID trajectory (already in
`training-runs/*/stats.jsonl`) against whatever the new run produces is free.
It is not a controlled contrast — different N, different data — and must be
labelled descriptive, but "the field's default face-pretrained initialization
spent most of its budget unlearning faces" is worth one honest figure.

### 5.5 The 6-page limit is a harder constraint than the plan assumes

`FINAL_PLAN.md` §8 already lists a cut order, which is good practice. But the
plan as written — decomposition, four arms, a historical row, RETFound + Alaa +
Meehan, eye-level cluster bootstrap, colour control, prespecified decision
rules, leakage decomposition — is already at or beyond six pages *including
references*, before adding anything from §6 below. The §6 recommendations are
therefore presented **with explicit page costs and a cut order** (§7), not as an
open-ended wish list.

---

## 6. Recommended repositioning and experimental plan

**Central question, unchanged:** how much do generative methods improve
performance in a data-scarce disease setting?

**Answer format, changed:** three numbers instead of one verdict.

> **What does synthetic data actually buy in a rare retinal disease?**
> An exchange rate, a leakage tax, and a shortcut cost — measured
> generator-blind.

Each is a quantity, each is publishable under a null, and each is aligned with
the original philosophy. Two generators, one GPU day each.

### 6.1 Deliverable 1 — the synthetic-to-real exchange rate (0 GPU)

Replace the single point contrast C-vs-B with a **paired scarcity ladder**.

- Real-training-size rungs: **N_dev ∈ {20, 40, 70, 111}** eyes (eye-disjoint
  subsamples of the 111, nested so smaller rungs are subsets of larger ones,
  repeated over several draws).
- At each rung, run arms A (real_only), B (oversample), C (synth_clean), and
  optionally D (pca_sampler), at the dose sweep `s05_classifier.py` already
  supports.
- Evaluate on (a) the fixed generator-blind 28-eye holdout, paired across all
  rungs and arms, and (b) repeated eye-disjoint CV inside the dev pool.

**The headline number:** the horizontal distance between the A curve and the C
curve — *how many additional real eyes the synthetic pool is worth*, and at what
N that equivalence falls to zero.

Why this is stronger than the current primary contrast:
- It is a *curve*, so it is robust to any single rung being noisy, and its shape
  (monotone, saturating, crossing zero) is itself the finding.
- It directly answers "how much", which is the actual research question.
- The saturation literature (§3.6) predicts a decreasing benefit with N; either
  confirming that shape in a rare retinal disease or contradicting it is a
  result.
- It is entirely on data already on disk. **Zero GPU cost.**
- Under a complete null it still produces a publishable statement: *"the
  exchange rate is indistinguishable from zero at every scarcity level tested,
  including the most extreme (N=20)"* — which is a far stronger null than "no
  difference at N=111", because it closes the "you just didn't test a scarce
  enough regime" escape hatch.

Run both `--backbone resnet18` and `--backbone toycnn` (§3.2). Report the
checkpoint-epoch/efficiency secondary endpoint; it is free and survives a null.

### 6.2 Deliverable 2 — the leakage tax (1 GPU day; GAN #2)

Restore the matched leaky twin. `gan_data/leak_111/` is built and verified.

- `G_clean`: 111 dev eyes. `G_leak`: 28 holdout eyes + 83 dev eyes.
- **N matched at 111. Same base checkpoint, same hyperparameters, same seed,
  same kimg budget, same snapshot-selection rule, same ψ, same dose.**
- Both evaluated on the same 28 holdout eyes.
- The difference is the leakage tax, with training-set size controlled — the
  confound that made v3's existing 885-image pool unusable for this purpose.

Report at both layers: downstream AUC/sensitivity **and** the Alaa/Meehan
generative metrics against the holdout (does `G_leak` show elevated
authenticity failure or C_T data-copying against the holdout that `G_clean`
does not?). The second layer is what turns "the number went up" into a
mechanism.

Handle the n=1 problem exactly as in §5.3: shared init, snapshot dispersion,
matched-pair case-study framing.

**Why this is the right use of GAN #2** rather than a seed replicate of
`G_clean`: a seed replicate improves the confidence interval on an already
crowded question (§3.1, §3.2). The leaky twin produces a number the field does
not have. Given the stated preference for a genuinely novel result over a safe
one, the twin wins — provided its n=1 limitation is stated plainly.

### 6.3 Deliverable 3 — shortcut amplification (0 GPU)

Promote the colour shortcut from validity check to finding. For each arm's
trained classifier, measure **dependence on the source shortcut**, not just AUC:

- AUC under colour standardization / channel-statistic matching at test time —
  how much does each arm lose when the shortcut is removed?
- Agreement between each arm's predictions and the colour-only logistic
  regression already implemented in `s05_classifier.py`.
- The same colour-moment statistics computed on real RP vs synthetic RP: does
  the generator reproduce the source colour signature more tightly than real
  data does?

**The claim, if it holds:** synthetic augmentation increases reliance on a
source-specific shortcut relative to oversampling. That is a novel mechanistic
finding (§3.5), it explains why internal gains fail to transfer, and it is the
kind of result that makes a null memorable instead of forgettable.

**If it does not hold**, it is a one-paragraph negative control that
strengthens every other claim in the paper. Either way it costs nothing.

### 6.4 The zero-GPU fallback paper

If the GPU day slips or the pod fails, deliverables 1 and 3 plus the already-
computed evaluation-protocol leakage decomposition, the duplicate/effective-N
collapse, the trivial-baseline saturation result, and the legacy-v3 contaminated
row constitute a complete, honest short paper with **no new GAN training at
all**. That is the minimal-runway UEMCON option if that venue is the real target.

---

## 7. Triage

Ordered by value per unit of effort. Everything above the line is the paper.

| # | item | effort | GPU | why this order |
|---|---|---|---|---|
| 1 | Eye-level evaluator + scarcity ladder + dose sweep in `s05_classifier.py`, both backbones | large | 0 | Deliverable 1. The confirmatory result. Runs on existing data. |
| 2 | Shortcut-amplification analysis | small | 0 | Deliverable 3. Small extension of existing colour code. |
| 3 | Paired cluster-bootstrap + prespecified decision rules frozen **in writing before any GPU spend** | small | 0 | Without this the whole design's credibility argument collapses. |
| 4 | k-SALSA fundus-checkpoint spike, **hard 3-hour cap**, fallback FFHQ256 | small, hard-capped | 0 | §5.4. Cheap upside, no dependency. |
| 5 | **GPU day: `G_clean` + `G_leak`, matched** | one session | ~2 GPU-days' worth | Deliverables 2. Run both on the same rental. |
| 6 | Snapshot selection offline; sample ψ ∈ {0.7, 1.0} | small | 0 | Prespecified rule, no results-driven stopping. |
| 7 | All arms + metrics rerun with the real synthetic pool | medium | 0 | |
| 8 | Writing + figures | largest | 0 | The real bottleneck. |
| — | *line* | | | |
| 9 | RETFound embedding for Alaa/Meehan (currently Inception only) | medium | 0 | **Cut first if late** — Inception is defensible with the Akbar caveat stated; RETFound is better but not load-bearing. |
| 10 | Arm D (pca_sampler) | small | 0 | Cut second. |
| 11 | ψ sweep beyond ψ=1.0 | trivial | 0 | Cut third; keep ψ=1.0 only. |

Note items 1–4 are all local and could start immediately.

**Hard sequencing constraint:** item 3 (freezing decision rules) must complete
before item 5 (GPU). This is what makes the prespecification claim true rather
than decorative, and it is the single cheapest credibility purchase in the plan.

---

## 8. Risks and things this review did not verify

- **Not verified: k-SALSA checkpoint usability.** Resolution mismatch (512 vs
  256) and format conversion (rosinality `.pt` → NVIDIA `.pkl`) are both real
  engineering risk. Treated as an optional spike for exactly this reason.
- **Not verified: fundus-corpus licensing.** The 817-image `pretrain_fundus/`
  set on disk was not license-audited in this review. It is not load-bearing in
  the recommended plan.
- **Not verified: whether EyeDiff's rare-disease cohort overlaps this project's
  data source.** Worth 10 minutes before writing the related-work paragraph.
- **Unchanged open item: the reference standard for the 139 RP labels is
  undefined.** This remains the most serious unfixable limitation and must be
  stated in the limitations section. No amount of experimental design repairs
  it.
- **Unchanged open item: external RP data is 6 images.** Correctly deferred to
  a future submission in `FINAL_PLAN.md`. The long-lead requests (Iowa StoneRounds email,
  PhysioNet/BRSET credentialing, the UCSF Sci Rep authors) should go out
  regardless of which venue this paper targets — they are human actions on
  someone else's timetable, and nothing in the current plan depends on them.
- **Statistical caution:** the scarcity ladder introduces multiple contrasts.
  Designate **one** rung and one arm pair as confirmatory (recommend C vs B at
  the full N=111, matching the existing plan's P1) and label every other rung
  descriptive with unadjusted intervals, or apply Holm correction across rungs
  as the brain-MRI study did. Do not let a ladder become a fishing expedition —
  that is the failure mode reviewers look for in exactly this design.

---

## 9. Sources

**Direct competitors and prior art**
- [Boosting foundation models for rare eye disease diagnosis via a multimodal text-to-image generative framework (EyeDiff) — npj Digital Medicine 2026](https://www.nature.com/articles/s41746-026-02560-2)
- [Do Synthetic Brain MRIs Reliably Improve Tumour Classification? A StyleGAN2-ADA Class-Plane Augmentation Study on BRISC 2025 — arXiv 2605.23094](https://arxiv.org/pdf/2605.23094)
- [Deep learning detection of retinitis pigmentosa inheritance forms through synthetic data expansion of a rare disease dataset — Sci Rep 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12687845/)
- [Retinitis Pigmentosa Classification with Deep Learning and Integrated Gradients Analysis — Appl. Sci. 2025](https://www.mdpi.com/2076-3417/15/4/2181)
- [Deep Learning-Based Detection of Pigment Signs for Analysis and Diagnosis of Retinitis Pigmentosa](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7349531/)
- [A Future Picture: Current GANs in Vitreoretinal Pathologies — Biomedicines 2025](https://doi.org/10.3390/biomedicines13020284)
- [Synthetic AI using GAN for retinal imaging in detection of AMD — Front. Med. 2023](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2023.1184892/full)

**GAN vs diffusion, memorization, and metric validity**
- [Beware of diffusion models for synthesizing medical images — a comparison with GANs in terms of memorizing brain MRI and chest x-ray images — Mach. Learn.: Sci. Technol. 2024](https://iopscience.iop.org/article/10.1088/2632-2153/ad9a3a)
- [A multimodal comparison of latent denoising diffusion probabilistic models and GANs for medical image synthesis — 2023](https://pubmed.ncbi.nlm.nih.gov/37495660/)
- [Metrics that Matter: Evaluating Image Quality Metrics for Medical Image Generation — arXiv 2505.07175](https://arxiv.org/html/2505.07175)
- [Balancing Fidelity, Utility, and Privacy in Synthetic Cardiac MRI Generation — arXiv 2603.04340](https://arxiv.org/html/2603.04340v1)

**Null and cautionary results on synthetic augmentation**
- [A Critical Assessment of Generative Models for Synthetic Data Augmentation on Limited Pneumonia X-ray Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC10741143/)
- [Using Synthetic Images to Augment Small Medical Image Datasets — arXiv 2503.00962](https://arxiv.org/abs/2503.00962)
- [Data augmentation strategies for GAN-based medical image analysis: an empirical review — 2026](https://www.tandfonline.com/doi/full/10.1080/21642583.2026.2634452)

**Leakage and contamination**
- [Synthetic data, real errors: how (not) to publish and use synthetic data — arXiv 2305.09235](https://arxiv.org/pdf/2305.09235)
- [Quantifying the Effect of Test Set Contamination on Generative Evaluations — arXiv 2601.04301](https://arxiv.org/abs/2601.04301)
- [MassSpecGym in the Wild: Uncovering and Correcting Evaluation Pitfalls in AI-Driven Molecule Discovery — arXiv 2606.19624](https://arxiv.org/pdf/2606.19624)
- [When Privacy Isn't Synthetic: Hidden Data Leakage in Generative AI Models — arXiv 2512.06062](https://arxiv.org/html/2512.06062)

**Dose-response, exchange rate, scaling**
- [Augmenting medical image classifiers with synthetic data from latent diffusion models — arXiv 2308.12453](https://arxiv.org/pdf/2308.12453)
- [Scaling Laws of Synthetic Images for Model Training ... for Now — arXiv 2312.04567](https://arxiv.org/pdf/2312.04567)
- [Diversify, Don't Fine-Tune: Scaling Up Visual Recognition Training with Synthetic Images — arXiv 2312.02253](https://arxiv.org/pdf/2312.02253)

**Shortcut learning, bias inheritance, dataset hygiene**
- [Mask of truth: model sensitivity to unexpected regions of medical images — arXiv 2412.04030](https://arxiv.org/pdf/2412.04030)
- [Cross-Camera External Validation for AI Software in Diagnosis of Diabetic Retinopathy](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8926465/)
- [From Fake to Real: Pretraining on Balanced Synthetic Images to Prevent Spurious Correlations — ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-73636-0_14)
- [Generative models improve fairness of medical classifiers under distribution shifts — arXiv 2304.09218](https://arxiv.org/pdf/2304.09218)
- [Benchmarking Pretrained Vision Embeddings for Near- and Duplicate Detection in Medical Images — arXiv 2312.07273](https://arxiv.org/pdf/2312.07273)

**Few-shot generative modelling under limited data**
- [Few-shot Adaptation of Generative Adversarial Networks — arXiv 2010.11943](https://arxiv.org/pdf/2010.11943)
- [Adversarial Semantic Augmentation for Training GANs under Limited Data — arXiv 2502.00800](https://arxiv.org/html/2502.00800v1)
- [Smoothness Similarity Regularization for Few-Shot GAN Adaptation — arXiv 2308.09717](https://arxiv.org/pdf/2308.09717)

**Resources and infrastructure**
- [k-SALSA — pretrained StyleGAN2-ADA generators on APTOS and EyePACS (ECCV 2022)](https://github.com/hcholab/k-salsa)
- [RETFound — a foundation model for generalizable disease detection from retinal images, Nature 2023](https://www.nature.com/articles/s41586-023-06555-x)
- [How Faithful is your Synthetic Data? — Alaa et al., ICML 2022](https://proceedings.mlr.press/v162/alaa22a.html)
- [A Non-Parametric Test to Detect Data-Copying in Generative Models — Meehan et al., AISTATS 2020](https://proceedings.mlr.press/v108/meehan20a.html)

**Venues**
- [IEEE UEMCON 2026 — Call for Papers](https://ieee-uemcon.org/call-for-papers/)
- [IEEE BIBM 2026 UGHS Symposium](https://bibm2026-hs.github.io/)
