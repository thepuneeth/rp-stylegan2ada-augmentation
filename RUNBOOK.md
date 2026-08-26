# Runbook — from raw images to the paper

Everything here runs on CPU/MPS and is free, except where marked **GPU**.
Run from the repo root with `./venv/bin/python`.

## Environment

```bash
export SSL_CERT_FILE=$(./venv/bin/python -c "import certifi;print(certifi.where())")
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
```

**Not optional.** Without it `torchvision` cannot fetch the ImageNet ResNet18
weights and `s05` dies at `build_model`. The weights are cached after the first
success (`~/.cache/torch/hub/checkpoints/`), so this bites once, on a machine
that has never run the classifier.

## Pipeline

| step | script | status | produces |
|---|---|---|---|
| 0 | `s00_verify.py` | **done**, 22/22 pass | integrity gate; run after ANY data change |
| 1 | `s01_unique_eyes.py` | done | eye clusters from FOV-cropped correlation |
| 1a | `s01a_calibrate_duplicate_metric.py` | done | the 0.99 threshold, calibrated not assumed |
| 2 | `s02_make_splits.py` | done | eye-disjoint dev / holdout |
| 3 | `s03_novelty_metrics.py` | done | legacy v3 novelty baseline |
| 4 | `s04_prepare_gan_data.py` | done | `clean_111.zip`, `leak_111.zip`, size-matched |
| 4b | `s04b_pca_sampler.py` | done | `pca_pool`, the cheap-fake control |
| — | GAN training | **done** (GPU) | `runs/g_clean`, `runs/g_leak`, 9 snapshots each |
| — | pools + cross-score | **done** (GPU) | 4 pools, `s08_pod_metrics_*.json` |
| 5 | `s05_classifier.py` | **PENDING** | the downstream utility endpoint |
| 6 | `s06_metrics.py` | **BLOCKED** on RETFound | authenticity + Meehan C_T, primary endpoint |
| 7 | `s07_power.py` | done | MDE 0.043-0.070; type I 0.067-0.090 |
| 9 | `s09_swd.py` | done | SWD vs noise floor and real-to-real anchor |
| 10 | `s10_figures.py` | done | all paper figures that have inputs |
| 11 | `s11_gradcam.py` | ready, needs s05 | shortcut check: is attention on the retina? |
| — | `stats_boot.py` | done | BCa cluster bootstrap + permutation test (F9) |

Every script with non-trivial logic has `--self-test`. Run it before trusting it.

## The pending run

`s05` is the only thing left that takes real time. 70 fits without `synth_leak`,
100 with it. The arm is routed but **not** in `config.ARMS`, so it is opt-in.

Decided: **run the five-arm form.** The reasoning is in
`docs/FINAL_PLAN.md` §Arm decision. Short version — F1 and F3 show the two
generators are indistinguishable as distributions, and `synth_leak` is the only
thing that carries that finding through to downstream accuracy.

```bash
# THE RUN — five arms, including the contaminated generator (contrast P2)
./venv/bin/python scripts/s05_classifier.py --tag main5 \
  --arms real_only oversample pca_sampler synth_clean synth_leak \
  --synth-clean-dir gan_data/synth_clean_psi1.0 \
  --synth-leak-dir  gan_data/synth_leak_psi1.0 \
  --pca-dir gan_data/pca_pool
```

`config.ARMS` deliberately stays at four, so the fifth arm is visible in the
command that produced the result rather than buried in config. Adding an arm
after a sweep re-runs the sweep, which is why the decision is made here first.

Before launching, in the same shell:

```bash
export SSL_CERT_FILE=$(./venv/bin/python -c "import certifi;print(certifi.where())")
```

without it `build_model` dies fetching the ImageNet weights.

## Reading the results

`s05` writes `outputs/s05_classifier_<tag>.json` with per-run image-level and
eye-level metrics plus raw probabilities. Eye level is the reportable one
(`config.EVAL_UNIT = "eye"`); use `stats_boot.bca_ci` for intervals and
`stats_boot.permutation_test` for arm differences, not the percentile bootstrap.

Watch `colour_lr` in the header line. It reached **AUC 0.950** on colour moments
alone in the smoke run. Any arm not clearly above that has not demonstrated it
learned anything retinal, and `s11_gradcam.py` is the follow-up that says whether
it did.

## What the paper rests on

Four distributional instruments have now been applied to the clean-vs-contaminated
question and none separates the generators: FID splits by reference, KID favours
`g_leak`, precision and SWD favour `g_clean`, every margin at or below the
resolution of a 111-image reference. Pixel-space `s06` is saturated at 99% of its
statistic's ceiling and cannot see memorisation at all.

So the contamination claim rests on `s06` in RETFound space and on `s05`. Those
two are the paper. Everything above is the apparatus that makes them credible.

Full reasoning and phrasing: `docs/METHODS_NARRATIVE.md`, Parts A, B and C.
