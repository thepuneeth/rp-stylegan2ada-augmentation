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
| 0 | `s00_verify.py` | **done**, all pass | integrity gate; run after ANY data change |
| 1 | `s01_unique_eyes.py` | done | eye clusters from FOV-cropped correlation |
| 2 | `s02_make_splits.py` | done | eye-disjoint dev / holdout |
| 4 | `s04_prepare_gan_data.py` | done | `clean_111.zip`, `leak_111.zip`, size-matched |
| — | GAN training | **done** (GPU) | `runs/g_clean`, `runs/g_leak`, 9 snapshots each |
| — | pools + cross-score | **done** (GPU) | pools, `s08_pod_metrics_*.json` |
| 5 | `s05_classifier.py` | **done** | the downstream utility endpoint (4 arms) |
| 9 | `s09_swd.py` | done | SWD vs noise floor and real-to-real anchor |
| 10 | `s10_figures.py` | done | all paper figures that have inputs |
| 11 | `s11_gradcam.py` | done | shortcut check: is attention on the retina? |
| 12 | `s12_contrasts.py` | done | P1 + P2 contrasts, BCa cluster bootstrap + permutation |
| — | `stats_boot.py` | done | BCa cluster bootstrap + permutation helpers |

Every script with non-trivial logic has `--self-test`. Run it before trusting it.

## The classifier run

`s05` is the endpoint that takes real time. Four arms: `real_only`,
`oversample`, `synth_clean`, and the contaminated generator `synth_leak`
(carries the F1/F3 distribution finding through to downstream accuracy, contrast
P2). `synth_leak` is routed but **not** in `config.ARMS`, so it is opt-in on the
command line.

```bash
# THE RUN — four arms, including the contaminated generator (contrast P2)
./venv/bin/python scripts/s05_classifier.py --tag main5 \
  --arms real_only oversample synth_clean synth_leak \
  --synth-clean-dir gan_data/synth_clean_psi1.0 \
  --synth-leak-dir  gan_data/synth_leak_psi1.0
```

`config.ARMS` stays at three, so `synth_leak` is visible in the command that
produced the result rather than buried in config. Adding an arm after a sweep
re-runs the sweep.

Before launching, in the same shell:

```bash
export SSL_CERT_FILE=$(./venv/bin/python -c "import certifi;print(certifi.where())")
```

without it `build_model` dies fetching the ImageNet weights.

## Reading the results

`s05` writes `outputs/s05_classifier_<tag>.json` with per-run image-level and
eye-level metrics plus raw probabilities. Eye level is the reportable one
(`config.EVAL_UNIT = "eye"`); `s12_contrasts.py` does the inference — P1
(`synth_clean` vs `oversample`) and P2 (`synth_leak` vs `synth_clean`) with BCa
cluster bootstrap and permutation, not the percentile bootstrap.

`s11_gradcam.py` is the follow-up that says whether a high AUC reflects
attention on the retina rather than the border.

## What the paper rests on

The distributional instruments applied to the clean-vs-contaminated question do
not separate the generators: FID splits by reference, KID favours `g_leak`,
precision and SWD (`s09`) favour `g_clean`, every margin at or below the
resolution of a 111-image reference. The cross-score F1/F3 (`s08`) says the same.

So the contamination claim rests on `s05` (downstream utility, contrast P2 via
`s12`) and `s11` (attention on the retina). Those are the paper. Everything above
is the apparatus that makes them credible.

Full reasoning and phrasing: `docs/METHODS_NARRATIVE.md`, Parts A, B and C.
