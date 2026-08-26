# Audit of `FINAL_PLAN.md`

**Audit date:** 2026-07-30  
**Verdict:** Major revision required before GPU spending  
**Scope:** Scientific premise, experimental design, statistics, implementation readiness, data integrity, literature claims, GPU recipe, and venue requirements.

## Executive summary

The project has unusually strong dataset hygiene: its generator-blind RP split, packaged GAN datasets, and leakage checks all passed verification. The proposed comparison between scratch and transferred generators is also potentially useful.

The current plan is not execution-ready, however. Its information-theoretic premise is stronger than the cited theory supports; the decisive transferred-GAN-versus-direct-pretraining comparison relies on an underpowered classifier endpoint; several arms do not isolate their claimed factors; and one GAN run per condition does not provide generator-level replication. Statistical inference, eye-level evaluation, and the direct-pretraining protocol must be specified before GPU runs begin.

## Work performed

1. Read and cross-checked:
   - `docs/FINAL_PLAN.md`
   - `docs/PRETRAIN_CORPUS.md`
   - `docs/RUNPOD_RECIPE.md`
   - `config.py`
   - `scripts/s00_verify.py` through `scripts/s05_classifier.py`
   - Local StyleGAN2-ADA training configuration and training loop
   - Existing splits, outputs, GAN datasets, and synthetic pools
2. Ran `python3 scripts/s00_verify.py` from the project root.
3. Inspected current classifier-arm routing, transforms, pretrained weights, and evaluation flow.
4. Inspected the NCG implementation and its uncertainty calculations.
5. Checked the StyleGAN2-ADA resume behavior against the scratch configuration.
6. Recalculated dataset totals from the source publication.
7. Reanalysed the colour-only holdout control at the eye level with a stratified cluster bootstrap.
8. Checked the cited information-theory, RETFound, RP synthetic-expansion, generative-auditing, source-dataset, and venue materials.
9. Checked IEEE BIBM 2026 UGHS eligibility, page-limit, attendance, and deadline requirements.

No project code, data, configuration, or original planning documents were changed during the audit.

## Verified strengths

### Dataset integrity

`python3 scripts/s00_verify.py` completed successfully with every invariant passing, including:

- 139 original RP images accounted for.
- Every RP and Healthy file has an eye label.
- RP DEV and holdout sets share no files or identified eyes.
- Healthy DEV and holdout sets share no files or identified eyes.
- All 25 configured folds are eye-disjoint and contain validation RP cases.
- Holdout prevalence matches the source cohort closely.
- `clean_111` contains 111 DEV RP images and no holdout image.
- `leak_111` contains all 28 holdout RP eyes as intended.
- `pretrain_fundus` contains no exact holdout image.
- Packaged clean and leaky ZIP files differ in content.
- The legacy synthetic pool contains no exact real image.
- The positive hash self-test confirms that leakage detection is non-vacuous.

### Other strengths

- The plan explicitly recognizes the severe same-source colour shortcut.
- The generator-blind RP holdout is substantially better than the legacy generator's contaminated evaluation.
- Sample-size matching corrected the earlier synthetic-set-size bias in NCG.
- RETFound is a more defensible primary embedding than ImageNet Inception for retinal images.
- Adding established Alaa and Meehan metrics reduces reliance on a project-specific metric.
- The GPU plan includes a compatibility and throughput gate before full runs.
- The proposed deadline, six-page limit, and biomedical-imaging scope match the venue materials.

## Blocking findings

### 1. The information-theoretic conclusion is overstated

`FINAL_PLAN.md` predicts that scratch GAN, PCA, and oversampling should be approximately equal regardless of model capacity because none receives an external signal.

The cited information-theory work establishes a data-processing bound for closed synthetic-training loops. It does not establish equality of downstream risk among different learning algorithms. A transformation of a finite dataset can improve a restricted classifier through regularization, class balancing, invariance, or optimization without adding information about the unknown population.

The current classifier also uses ImageNet-pretrained ResNet18 weights in `scripts/s05_classifier.py:114`. Therefore the end-to-end pipeline is not information-closed even when the generator is trained from scratch. Any use of “closed” should be limited explicitly to the generator stage.

**Required change:** Present the closed-arm ordering as an empirical hypothesis, not a theorem or necessary consequence of the data-processing inequality.

### 2. The decisive F-versus-G question is not supported by the primary metrics

The plan identifies transferred synthesis versus direct classifier pretraining as the decisive comparison. RETFound image metrics cannot evaluate direct classifier pretraining because arm G generates no images. The comparison therefore depends on downstream classifier performance—the endpoint the plan itself says is saturated and underpowered.

Scarcity regimes can increase headroom, but they do not increase the 28 positive test eyes or guarantee larger effects. Repeated model seeds do not create additional independent clinical test observations.

**Required change:** Either obtain substantially more generator-blind/external RP test eyes, or label F-versus-G exploratory. A defensible primary generator question is E versus F in a fixed retinal embedding, with downstream classification secondary.

### 3. “Same information, two routes” is not yet a matched experiment

Arm F follows FFHQ initialization, stage-2 fundus training, and RP fine-tuning. Arm G is not operationally defined. The plan does not specify:

- Classifier architecture and starting weights.
- Supervised versus self-supervised corpus pretraining.
- Whether corpus labels are available to G.
- Training objective and schedule.
- Compute budget.
- Class balancing.
- Checkpoint selection.
- Whether other arms retain ImageNet initialization.

If G uses corpus labels, it receives information unavailable to an unconditional GAN. If it is self-supervised, a specific method must be implemented and validated. Starting F from FFHQ while starting G from a different base also prevents a clean same-information interpretation.

The stage-2 corpus deliberately excludes RP, so it cannot contain external RP-specific information. It supplies general fundus and pathology information, which is then adapted using the same target RP data.

**Required change:** Define G completely and match data access, label access, base initialization, evaluation budget, and downstream class balance. Otherwise describe F versus G as a pragmatic comparison of pipelines rather than a causal test of delivery route.

### 4. One trained generator per condition is pseudoreplication

The plan trains one scratch RP GAN and one transferred RP GAN. Thousands of samples from one checkpoint do not replicate generator training. Differences may be caused by a favorable or unfavorable initialization, training trajectory, or checkpoint.

**Required change:** Prefer at least three independent E runs and three independent F runs. If this is impossible, call the result a checkpoint-level case study and avoid population-level claims about scratch versus transfer training. Replication should take priority over the leakage-tax run, extra truncation values, and the new NCG metric.

### 5. Several contrasts do not isolate their stated factors

- B, D, and E differ in model family, assumptions, optimization, and output behavior—not only capacity.
- The claimed PCA parameter count of approximately `10^4` is undefined and likely misleading for image-space PCA.
- Added synthetic quantities are not fixed in the primary arm table.
- A fixed number of epochs produces different optimizer-step counts as training-set size changes.
- F and G may produce different class balances and numbers of RP presentations.
- The FFHQ-to-corpus-to-RP path means F includes both FFHQ and corpus transfer; it does not isolate the corpus alone.

A concrete implementation confound exists in the local StyleGAN code:

- `stylegan2-ada-pytorch-main/training/training_loop.py:111` defaults `ada_kimg` to 500.
- `stylegan2-ada-pytorch-main/train.py:314` changes it to 100 whenever `--resume` is used.

Thus E and F do not currently differ only in initialization; ADA adapts five times faster in F.

**Required change:** Harmonize ADA settings explicitly, define a fixed primary added-data quantity or optimizer-step budget, and soften all “isolates capacity” language.

### 6. The statistical decision framework is incomplete

The four-outcome table treats apparent equality as a publishable null result. Nonsignificance is not evidence of equivalence. Claims such as E approximately equals B require a prespecified equivalence margin and adequate power.

The table also omits:

- E outperforming F.
- F and G both failing.
- Endpoint disagreement.
- Dose or scarcity interactions.
- Wide confidence intervals supporting no conclusion.
- Arm H outperforming or failing to outperform its components.

The plan does not identify one primary contrast and one primary endpoint or state how multiplicity will be handled.

**Required change:** Add an explicit inconclusive outcome, define effect-size thresholds/equivalence margins, and prespecify the primary contrast before seeing new results.

## Statistical and measurement findings

### Holdout unit mismatch

The holdout contains 28 RP eyes and 207 Healthy files representing 197 identified Healthy eyes. Treating all 207 files as independent controls understates uncertainty.

An audit reanalysis that averaged the colour-model probabilities by eye found:

- File-level AUC: 0.9501 using 28 positive and 207 negative files.
- Eye-level AUC: 0.9561 using 28 positive and 197 negative eyes.
- Stratified eye-cluster bootstrap 95% CI: 0.9204–0.9850.

This interval illustrates the limited precision available from 28 positive eyes despite the large number of negative files.

**Required change:** Aggregate predictions by eye or use cluster-aware bootstrap/permutation inference. State explicitly that patient-level independence cannot be guaranteed because patient identifiers are unavailable; contralateral eyes may cross partitions.

### The stated minimum detectable AUC gain is not reproducible

The claimed 0.046 MDE uses an assumed paired correlation of 0.8 without justification and does not specify one- versus two-sided testing clearly. A standard Hanley–McNeil sanity calculation near baseline AUC 0.9501 does not provide 80% two-sided power even for a gain approaching the maximum possible 0.0499.

**Required change:** Replace the analytic claim with a transparent paired simulation or cluster bootstrap based on plausible score distributions. Do not claim that scarcity restores power automatically.

### The holdout is generator-blind but no longer untouched

The holdout was used to quantify the colour shortcut, motivate the redesign, and calibrate expectations. This does not invalidate generator-blind novelty analysis, but it prevents describing the holdout as a pristine confirmatory test set.

**Required change:** Use “generator-blind reused holdout” and classify inference as exploratory unless an untouched external set is acquired.

### NCG uncertainty is currently invalid as a confidence interval

In `scripts/s03_novelty_metrics.py:220–222`, the NCG interval is the percentile spread across repeated size-matched subsets drawn from one fixed synthetic pool. It measures subset-selection variability only. It does not capture uncertainty across:

- Holdout eyes.
- Training eyes.
- GAN runs.
- Learned feature representations.

The copying control is a useful implementation self-test, but its exact zero is a deterministic reference for literal copies, not a statistical null for “no useful information.” Interpolation, jitter, nuisance variation, or chance can produce NCG above zero without clinically meaningful novelty.

**Required change:** Demote NCG to exploratory unless a valid eye- and generator-level resampling/permutation procedure is developed. Compare it against oversampling, traditional augmentation, and PCA—not only literal copying.

### RETFound does not automatically establish clinical novelty

RETFound is retinally relevant, but the plan does not demonstrate that the selected feature layer distinguishes RP morphology rather than camera, colour, crop, or other pathology. Related-pathology contamination could increase apparent coverage rather than necessarily “failing safe.” High precision alone does not prove the absence of such contamination.

**Required change:** Freeze and document preprocessing and layer selection. Add sanity checks using confirmed duplicate pairs, known eye relationships, and real disease/source controls. Restrict conclusions to embedding-space novelty unless expert clinical review is available.

### Shortcut learning remains load-bearing

All local cohort images share a JPEG quantization signature and standardized resolution, and six colour moments achieve holdout AUC around 0.95. Scarcity experiments do not remove this shortcut.

**Required change:** Keep downstream conclusions methodological and same-source. Do not imply clinical validity or generalization without an external cohort. Require every deep arm to exceed the colour-only control before interpreting it as disease learning.

## Data and factual corrections

### Stage-2 corpus arithmetic

The source dataset contains:

- CSCR: 101
- Diabetic retinopathy: 1,509
- Disc edema: 127
- Glaucoma: 1,349
- Healthy: 1,024
- Macular scar: 444
- Myopia: 500
- Pterygium: 17
- Retinal detachment: 125
- RP: 139

The seven additional fundus pathology classes, excluding anterior-segment Pterygium, total 4,155 images. Adding the existing 817 DEV Healthy images gives an expected stage-2 corpus of approximately **4,972**, not 4,989.

The source publication reports two Topcon camera models and two original resolutions before standardization. “Same camera” should therefore not be assumed for all downloaded classes.

### Corpus construction is not yet implemented

The current `pretrain_fundus` directory contains only 817 DEV Healthy images. The existing preparation script rebuilds that directory from Healthy data; it does not download or incorporate the seven additional fundus classes. Manually added files may be deleted if the script is rerun.

**Required change:** Implement and verify corpus ingestion before declaring stage 2 unblocked. Add exact- and near-duplicate checks across all downloaded classes and both holdouts.

### Internal inconsistencies

- `FINAL_PLAN.md` specifies S2 at 2,000 kimg, while the executable command in `RUNPOD_RECIPE.md` uses 3,000 kimg.
- The plan specifies truncation ψ values `{0.7, 1.0}`; configuration and recipe materials also include 0.5.
- Arm G and `G_leak` use conflicting names; the leakage arm should be renamed, for example `F_leak` or `L`.
- “Arm G is free” ignores implementation and compute for classifier pretraining.
- The exact quoted claim about “clinically relevant concept combinations” has no traceable citation in the plan. Add a source and page/passage or remove quotation marks.
- Broad priority claims such as “no prior work compares...” should be phrased as “to our knowledge” with a documented search date and scope.

## Implementation readiness

`config.py` lists the new eight-arm ladder, but `scripts/s05_classifier.py` does not implement it fully:

- Oversampling is implemented.
- A legacy synthetic arm is implemented.
- Traditional augmentation, PCA sampling, scratch synthesis, transferred synthesis, direct pretraining, and combined routes do not yet have distinct complete paths.
- Unsupported arm names can fall through to effectively the same real-data behavior.

The classifier script also does not currently execute the claimed 5 repeats × 5 folds. It trains on the full DEV set and evaluates repeated runs on the fixed holdout rather than consuming `dev_folds.json` as described.

**Required change:** Complete the arm routing and decide whether the design is fixed-holdout repeated training or repeated cross-validation. Do not describe one as the other.

## GPU and operational findings

- Four GAN runs are computationally plausible after the protocol is fixed, but scientifically insufficient for generator-level replication.
- RTX 4090 execution needs the planned custom-operation compile and throughput gate because the pinned legacy PyTorch environment does not natively target Ada Lovelace kernels.
- The S2 budget conflict changes the total from 5,000 to 6,000 kimg.
- RETFound, Alaa, and Meehan implementations should be completed and tested on toy/current data before GPU generation; otherwise expensive outputs may be produced before the primary apparatus works.
- Given the prior incomplete snapshot problem, checkpoint persistence must be verified. A persistent volume or tested object-storage transfer is preferable to relying on an unverified manual `runpodctl send` workflow.

## Venue requirements

The IEEE BIBM 2026 Undergraduate and High School Symposium materials indicate:

- Deadline: 2026-08-31 AoE.
- Undergraduate papers: maximum six pages including figures, tables, and references.
- High-school papers: maximum five pages including figures, tables, and references.
- Review: single-blind.
- Biomedical imaging is in scope.
- An undergraduate or high-school student must be first author and primary contributor.
- At least one student author must attend and present in person in Dallas for inclusion in the proceedings.

**Go/no-go requirement:** Confirm first-author eligibility, primary-contributor status, travel, and in-person attendance before spending on GPU runs.

## Minimum revision before execution

1. Reframe the information-theory argument as testable hypotheses.
2. Choose one primary contrast and endpoint; E versus F in RETFound space is the cleanest current option.
3. Mark F versus G exploratory unless more independent test eyes are obtained.
4. Define arm G completely, including objective, labels, initialization, schedule, and balance.
5. Harmonize resumed and scratch StyleGAN settings, especially `ada_kimg`.
6. Add independent GAN seeds; cut leakage, extra ψ values, or NCG first if resources are constrained.
7. Evaluate and resample at the eye level.
8. Replace the MDE claim with transparent paired/cluster-aware power simulation.
9. Demote or repair NCG inference and avoid equating embedding proximity with clinical novelty.
10. Implement every classifier arm and align the written evaluation protocol with actual code.
11. Build and verify the corrected 4,972-image stage-2 corpus.
12. Resolve all kimg, ψ, naming, quote, and citation inconsistencies.
13. Confirm UGHS student and attendance eligibility.

## Recommended interpretation after revision

A defensible paper can ask whether external retinal-corpus pretraining improves the fidelity and coverage of an RP generator relative to matched random initialization, and explore whether synthetic delivery or direct representation pretraining is more useful downstream under scarcity.

It should not claim that the data-processing inequality forces all closed augmentation methods to be equal, that a nonsignificant contrast establishes equivalence, or that same-source embedding and classifier results demonstrate clinical generalization.

## Sources checked

- Project plan and supporting files listed under “Work performed.”
- Information-theoretic synthetic-training paper: https://arxiv.org/html/2605.16379
- RETFound paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10550819/
- Alaa et al., fidelity/diversity/generalization metrics: https://proceedings.mlr.press/v162/alaa22a.html
- Meehan et al., data-copying test: https://proceedings.mlr.press/v108/meehan20a.html
- RP synthetic-expansion study: https://pubmed.ncbi.nlm.nih.gov/41965358/
- Source retinal dataset paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11528549/
- IEEE BIBM 2026 UGHS: https://bibm2026-hs.github.io/
