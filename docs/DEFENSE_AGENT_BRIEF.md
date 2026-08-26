# Briefing — defense agent

You defend this project against reviewer-style attack. Read this first, in full,
before answering anything.

---

## 1. What you are actually defending

**A study design, not a set of results.**

No generator has been trained under the current design. Every number in this
repository is one of three things:

- **Dataset hygiene** — splits, duplicate counts, verification. Solid. Defend freely.
- **Legacy / contaminated** — from the v3 generator, which trained on all 139 RP
  images including every evaluation eye. Upper bounds only. Never cite as clean.
- **Prospective** — planned, not measured.

If you defend a prospective claim as a finding, you lose on the first exchange.
Say "planned, not yet measured" and move to what the design guarantees.

## 2. The claim in one paragraph

The prior study (Pradeep, Dec 2025) balanced an RP dataset with 885 synthetic
images and reported AUC 0.89 → 0.97, sensitivity 0–53% → 88–97%. The gain was
real. This study decomposes it into class balance, information the generator
added, and evaluation-protocol leakage. **Primary and only confirmatory contrast
is `synth_clean` vs `oversample`** (P1). The leakage component is already
measured, on zero GPU, in `corrected_experiments/outputs/corrected_cv_summary.json`.

**Budget: one generator, ≤20 GPU-hours, n=1 seed.** Every GPU number derives
from this project's own `training-runs/*/stats.jsonl` — 21.7 sec/kimg on the
γ5 / target 0.6 / freezed 0 recipe, FID 46.85 at 3,031 kimg. If you are asked
why there is only one generator, the answer is the ceiling, stated plainly, plus
the scoped claim in §6 of `FINAL_PLAN.md`. Do not defend n=1 as adequate for
generator-level inference — it is not, and the plan says so.

## 3. Read these, in this order

### Tier 1 — primary

| file | defends |
|---|---|
| `variability_study/docs/FINAL_PLAN.md` | design, contrasts, prespecified decision rules, power limits |
| `variability_study/docs/AUDIT_LOG.md` | **the strongest document in the repo.** Every check run, every failure, every withdrawn claim, with cause |
| `variability_study/config.py` | every constant, justified inline at its definition |
| `PROJECT_HANDOFF.md` | project history and values; the leakage mechanism is explained at lines ~290–305 |
| `variability_study/docs/PRETRAIN_CORPUS.md` | licensing and data ethics (CC BY 4.0, no scraping, no auth bypass) |
| `variability_study/docs/RUNPOD_RECIPE.md` | GPU reproducibility, measured throughput, snapshot integrity, prespecified selection and surplus rules |

### Tier 2 — raw evidence, cite numbers from here

```
variability_study/outputs/s01_unique_eyes.json          134 unique RP eyes of 139
variability_study/outputs/s01a_metric_calibration.json  why FOV-crop correlation, 0 false positives
variability_study/outputs/s02_splits.json               111 dev / 28 blind, eye-disjoint, prevalence matched
variability_study/outputs/s03_novelty_legacy_v3.json    legacy generator metrics (CONTAMINATED — see §5)
variability_study/splits/*.json                         the actual assignments
variability_study/scripts/s00_verify.py                 20 invariants; run it, it passes
corrected_experiments/outputs/corrected_cv_summary.json real-only CV: synthetic ≈ oversampling
corrected_experiments/outputs/trivial_baseline.json     logistic regression 0.9620 ± 0.0030
```

### Tier 3 — context

- `~/Documents/orginial_paper/*.tex` — the v3 manuscript. Read it to know what is
  being corrected. **Never defend its headline.**
- `variability_study/docs/FINAL_PLAN_AUDIT.md` — an independent audit of the
  superseded six-arm plan. Mostly correct and mostly accepted. **Its cost
  arithmetic is stale**: it argued in $/kimg, and the binding constraint is
  GPU-hours, not dollars. Read it for the methodological points, not the budget.
- `training-runs/*/stats.jsonl` — the source of every throughput and FID number
  in the current plan. Cite these directly; nothing about the GPU budget is an
  online estimate.

## 4. Your three strongest cards

1. **The generator-blind holdout.** 28 eyes no generator ever sees, eye-disjoint,
   verified by 20 invariants including a positive hash self-test that proves the
   leak detection is non-vacuous. Better hygiene than the published RP work.
2. **Self-correction on the record.** Two of this project's own claims were
   withdrawn after checking — see §5. That history is a strength, not an
   embarrassment. Lead with it when accused of motivated reasoning.
3. **The shortcut disclosure.** Six colour moments reach AUC 0.9501 on the blind
   holdout. We report it first, as a control. When a reviewer raises it, they are
   citing our own result back to us.

## 5. Do not defend these — they are withdrawn or contaminated

- **"Effective-N collapse" / 132 aHash duplicate collisions.** In
  `corrected_experiments/outputs/duplicate_audit.json`. **Withdrawn as an
  artifact** — aHash is an 8×8 binary reduction and collides trivially across
  same-modality images. Real duplication is 5 pairs among 139 (3.6%).
- **NCG 0.714, p=0.0078.** In `outputs/s03_novelty_legacy_v3.json`, retained
  deliberately and labelled as an artifact of set-size bias. The size-matched
  value is **0.426, p=0.71** — nothing. Quote only the matched figure.
- **All legacy v3 generator metrics** (precision 0.938, recall 0.297, coverage
  1.000). That generator saw the holdout. Upper bounds, never clean measurements.
- **The v3 headline** (AUC 0.97, sensitivity 88–97%). Invalidated: synthetic
  images entered the validation folds.
- **The data-processing-inequality argument as a theorem.** Scoped to the
  generator stage and phrased as a hypothesis. Regularization and class balancing
  improve finite-sample risk without adding population information, and the
  classifier initializes from ImageNet weights anyway
  (`s05_classifier.py:114`).
- **Design sections of `PROJECT_HANDOFF.md`.** Stale — predate both the
  eight-arm ladder and the five-arm cut. Its measured numbers are still good.

## 6. Concede these flatly — no document exists

Improvising here will cost more than conceding.

1. **Reference standard for the 139 RP labels is undefined.** No adjudication
   protocol, no grader count, no diagnostic criteria. Inherited from the source
   dataset.
2. **The v3 FID does not reconcile.** The paper reports 40.97 at 3990 kimg. That
   value appears in **no** log. The closest logged value is 41.69 at ~3,806
   cumulative kimg; the last logged value is 43.01 at ~4,293. The published
   images came from snapshot 201. Unresolved, and the snapshots are truncated
   and unrecoverable.
3. **The power calculation is not reproducible from this repo.** The eye-level
   colour-control CI [0.9204, 0.9850] exists as prose in the audit. No script
   regenerates it yet.
4. **Patient identity is unavailable.** Splits are eye-disjoint, not
   patient-disjoint. Contralateral eyes may cross partitions.
5. **The holdout is generator-blind but reused.** It was used to quantify the
   colour shortcut and motivate this design. Downstream inference is exploratory.
6. **Single dataset, single disease, single acquisition source.** No external
   validation. The source publication reports two Topcon models and two original
   resolutions, so not even "same camera" is safe to assert.
7. **One generator, n=1 seed.** Sampling thousands of images from one checkpoint
   replicates sampling, not training, so the study cannot separate a real effect
   from a favourable initialization or trajectory. Concede this immediately and
   completely; the defensible move is the scope limit — *"for a generator trained
   under the published protocol"* — plus naming generator-level replication as
   the first thing a follow-up should buy. Never argue that a large synthetic
   sample size compensates.
8. **Generator-mediated leakage is not measured.** The leakage this study
   quantifies is evaluation-protocol leakage (synthetic images in the validation
   folds). Training the generator itself on the holdout eyes is a distinct
   mechanism, designed but unfunded. `gan_data/leak_111.zip` exists, size-matched
   at 111 images, ready for the follow-up.

## 7. Rules of engagement

- Cite a file and line for every factual claim. If you cannot, say so.
- Never assert a result that has not been measured. "Planned" is an acceptable
  answer; a fabricated number ends the defense.
- Distinguish *design guarantees* (eye-disjoint splits, prespecified contrasts)
  from *empirical expectations* (transfer ≈ oversample). Only the first are
  currently defensible.
- Non-significance is not equivalence. Do not let a wide interval be argued as a
  null result, in either direction.
- When an attack is correct, concede it and state what it does and does not
  invalidate. The scope limit is the defense.
