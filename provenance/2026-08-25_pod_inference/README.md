# Pod inference session — 2026-08-25

RTX 4090 secure, RunPod, ~72 min wall clock, **$0.939** ($6.073 -> $5.134).
Three pod starts; the first two failed on container gaps and cost $0.14 between
them. Every pod was torn down and verified (`pod list` -> `[]`).

## What ran

| phase | output |
|---|---|
| A. pools | 4 x 1110 images, seeds `0-1109`, psi 1.0 and 0.7, both generators |
| B. cross-score | each selected checkpoint scored against the OTHER arm's training zip |
| C. backfill | `g_clean` kid50k/pr50k3 at kimg 403, 806, 1209, 1612 |

Results: `outputs/s08_pod_metrics_2026-08-25.json`.
Pools: `gan_data/synth_{clean,leak}_psi{1.0,0.7}/`.

## Why B existed

Each generator had only ever been scored against its OWN training set, and those
sets differ in 28 of 111 images. The two KID values were therefore not on a
common yardstick (audit finding F1). B puts both checkpoints on both references.

## Two container gaps, for the next rental

`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` ships with neither:

- `click` -- `calc_metrics.py` and `generate.py` will not start.
- `ninja` -- the fused CUDA ops silently fall back to the slow reference path.

`pod_run.sh` installs both. Its gate checks the RETURN VALUE of
`bias_act._init()`; calling it bare reports success even on a fallback, because
`_init()` warns and returns False rather than raising.

## Utilisation note

`generate.py` runs at batch 1 and `metric_utils` at `batch_gen = min(batch_size, 4)`,
so nothing approached the card's 24 GB (training peaked at 19.17 GB; this did not).
Raising `--batch-gen` would have saved more than choosing a cheaper card.
No `nvidia-smi` was captured -- this is inferred from the code, not measured.

## Files

- `session3.sh` -- local orchestrator; creates the pod, uploads, runs, downloads,
  and deletes unconditionally via an EXIT/INT/TERM trap.
- `pod_run.sh` -- the pod-side job.
- `strip_pkl.py` -- reduces a snapshot to `G_ema` only, 296 MB -> 100 MB, with a
  self-check that the stripped copy reproduces the original output exactly.
- `pod_run.log`, `logs.tgz` -- raw stdout and per-metric logs.
