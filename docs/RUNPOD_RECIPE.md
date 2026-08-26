# GPU recipe — one run, one day, ≤20 GPU-hours

One generator. Nothing here runs without explicit approval.

| run | data | init | kimg | seeds | hours |
|---|---|---|---|---|---|
| **`g_clean`** | `clean_111.zip` | `ffhq256` | 3,000 | 1 | ≈18.8 |

That is the whole GPU plan. Rationale, contrasts and what n=1 costs the claims:
`FINAL_PLAN.md` §2, §3, §6.

---

## Throughput is measured, not estimated

From this project's own `training-runs/*/stats.jsonl`:

| run | config | kimg | sec/kimg | hours |
|---|---|---|---|---|
| 00004 | resume ffhq256, γ5, target 0.6, freezed 0 | 290 | 21.7 | 1.80 |
| 00005 | continue, γ5 | 2,750 | 21.7 | 16.92 |
| 00006 | γ3, target 0.35, freezed 4 | 770 | **28.8** | 6.29 |
| 00007 | γ4, target 0.25, freezed 5 | 484 | **27.8** | 3.79 |

**Use the phase-1 recipe only.** γ5 / target 0.6 / `freezed=0` ran at 21.7
sec/kimg. The `freezed` phases cost ~30% throughput and are cut. That is where
the previous project's budget went: 10.1 hours for 1,254 kimg.

Budget at 21.7 sec/kimg:

| item | hours |
|---|---|
| 3,000 kimg training | 18.1 |
| 4 × `fid50k_full`, after training | ~0.5 |
| sampling ψ ∈ {0.7, 1.0} | <0.1 |
| **total** | **≈18.7** |

Expected FID ≈ 47 — the measured value at 3,031 kimg on this recipe.

### GPU choice

Select on **$/kimg**, not $/hr. StyleGAN2-ADA at 256px is raw-FP32-bound, not
tensor-core or memory bound.

**Both rows below are MEASURED**, 24 kimg each, `--batch=64`, identical flags,
ported gradfix, zero fallbacks. Not projections.

| GPU | sec/kimg | 3,000 kimg | $/hr | run cost | verdict |
|---|---|---|---|---|---|
| **RTX 4090 secure** | **10.7** | **8.9 h** | $0.74 | **$6.56** | **use this** |
| RTX 4090 community | 10.7 | 8.9 h | $0.34 | $3.01 | cheaper, unreliable — see below |
| RTX 4000 Ada secure | 30.9 | 25.8 h | $0.28 | $7.21 | slower AND dearer per run |

The 4090 is 2.9x the 4000 Ada. Selecting on $/hr picks the wrong card here:
the 4000 Ada is less than half the hourly rate and still costs more per finished
run, because it is on the clock three times as long.

**Community 4090 failed twice in a row on 2026-08-23** and is not the default
despite winning on price. First pod: accepted the rent, reported `RUNNING`,
never booted its container (`uptimeSeconds: 0` after 10 min) — $0.06 for
nothing. Retry 90 s later: `"There are no longer any instances available"`,
though the catalog had just reported stock **High**. Community listings are
advisory, not a reservation. Try it if you like — a failed create costs $0 —
but budget for secure and treat a working community pod as a windfall.

**RTX A5000 is no longer rentable on RunPod at all.** Earlier drafts of this
file made it the safe default; it is not in the catalog.

**RTX 5090 looks good on paper and is a trap.** sm_120 (Blackwell) has no
kernels in the torch 2.4 / cu124 container this recipe uses; the custom ops
would not compile.

**Budget check:** at $6.56 secure, one generator fits a ~$7.80 balance with
~$1.20 of margin. Two generators (`g_clean` + `g_leak`) do NOT fit — top up
first, or cut `--kimg`.

**Do not rent a network volume.** At ~$0.07/GB/month it costs more than the
compute.

---

## Fifteen-minute gate

Do not debug on a rented clock.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
cd stylegan2-ada-pytorch-main && python -c "
from torch_utils.ops import bias_act, upfirdn2d
bias_act._init(); upfirdn2d._init(); print('CUDA ops OK')"
```

Then start training and read the first tick's `sec/kimg`.

| observed | action |
|---|---|
| ops fail to build | kill the pod (~$0.07 spent) — the container is wrong, not the code |
| first tick `gpumem` near 3 GB | `--batch=64` did not take. Kill and re-launch with it |
| sec/kimg ≈ 10.7 (4090) or 30.9 (4000 Ada) | on target, proceed |
| sec/kimg > 35 | wrong card or a throttled host. Kill and re-rent |

Read `sec/kimg` from **tick 1, not tick 0** — tick 0 carries plugin-compile and
startup time and reads ~9x high. Ticks 1-5 on the 4090 were 10.64, 10.77, 10.78,
10.79, 10.73: this number converges immediately and does not drift.

**Torch 2.x port — REQUIRED, and verified before renting.** Two files in
`torch_utils/ops/` were written against torch 1.7–1.9 and silently disable
themselves on anything newer, falling back to ops that cannot do the
second-order gradients R1 regularization needs. Measured on a rented pod: the
unported code CRASHES (`derivative for aten::grid_sampler_2d_backward is not
implemented`), and merely disabling `conv2d_gradfix` instead runs but costs
**33.8 sec/kimg against the 21.7 baseline — 56% slower**, plus 645,040
`warnings.warn` calls inside the training loop.

| file | change |
|---|---|
| `grid_sample_gradfix.py` | version gate widened; `_jit_get_operation` now returns `(op, names)`; `grid_sampler_2d_backward` gained a required `output_mask` |
| `conv2d_gradfix.py` | version gate widened; `aten::cudnn_convolution_backward_weight` was REMOVED in torch 1.13 and is replaced by `aten::convolution_backward` with `output_mask=[False, True, False]` |

Verify before spending anything — runs on CPU, no GPU required:

```bash
cd stylegan2-ada-pytorch-main && PYTHONPATH=$PWD python test_gradfix_port.py
```

Expected `ALL PASS`: conv2d matches native autograd to 0.0 on forward, all three
gradients and double-backward in both transpose modes; grid_sample matches on
first order, and its second-order path — which native torch cannot compute at
all — matches central finite differences to ~6e-12.

**GPU note.** RTX A5000 is not currently rentable on RunPod, and community 4090
reports stock it does not have. sm_89 (Ada) is fine on torch 2.x: both custom
CUDA ops compile cleanly, verified on an RTX 4000 Ada.

---

## 0. Upload

```
variability_study/gan_data/clean_111.zip      21 MB
stylegan2-ada-pytorch-main/                   the training code
```

That is all. `pretrain_fundus.zip` and `leak_111.zip` are **not used** — the
stage-2 corpus pretrain and the leakage generator are cut (`FINAL_PLAN.md` §3).
Both files are kept on the local machine for the follow-up study.

```bash
runpodctl get pod    # confirm the key works and nothing is already running
```

No source patch is required. The previous plan needed one because arm E did not
resume while arm F did, so `train.py:313-315` silently changed `ada_kimg` and
`ema_rampup` in one arm only. With a single resumed run the shift applies
uniformly and cancels. Record the observed values anyway:

```bash
grep -H "ada_kimg\|ema_rampup" /workspace/runs/*/*/training_options.json
```

---

## 1. Train

```bash
python train.py \
  --outdir=/workspace/runs/g_clean \
  --data=/workspace/clean_111.zip \
  --gpus=1 --cfg=paper256 --mirror=1 --batch=64 \
  --resume=ffhq256 --seed=0 \
  --gamma=5 --aug=ada --target=0.6 --freezed=0 \
  --snap=50 --metrics=fid50k_full \
  --kimg=3000
```

- `--batch=64` is **NOT optional and not cosmetic.** Without it, `train.py:191`
  computes `batch_gpu = spec.mb // spec.ref_gpus = 64 // 8 = 8`, because the
  `paper256` spec is written for an 8-GPU reference machine. On one GPU that
  silently becomes **8x gradient accumulation**: same effective batch, but eight
  micro-batches per step. Passing `--batch` explicitly takes the branch at
  `train.py:220`, `batch_gpu = batch // gpus = 64`, i.e. a single pass.

  Measured, RTX 4000 Ada, both runs 24 kimg, identical in every other respect:

  | | batch_gpu | gpumem | sec/kimg | 3,000 kimg | cost @ $0.28/hr |
  |---|---|---|---|---|---|
  | omitted (default) | 8 | 2.97 GB | 32.6 | 27.2 h | $7.61 |
  | `--batch=64` | 64 | 16.4 GB | 30.9 | 25.8 h | $7.21 |

  **The flag is worth 5%, not 33%.** An earlier draft of this file predicted it
  would recover a 21.7 sec/kimg baseline. It does not, and 21.7 is not reachable
  on this card at all -- that figure came from the prior project on different
  hardware. Gradient accumulation costs far less than assumed: the eight
  micro-batches are still eight full GPU passes either way, and the only saving
  is per-step overhead. Pass `--batch=64` anyway -- it is free and it is what
  the config intends -- but budget from 30.9, not 21.7.

  16.4 GB of the card's 20 GB is used, so `--batch=64` fits but leaves no room
  for a second job or a larger batch.

  The prior project's run directory is named `...-batch64-ada-resumecustom`;
  that `-batch64` suffix is stamped by `train.py:218` and is the record that
  `--batch=64` was passed. **Check for it in the new run's directory name --
  it is the cheapest possible confirmation that the fast path is active.**
  A gpumem near 3 GB in the first tick line means it is not.
- `--snap=50` = a snapshot every **200 kimg** (4 kimg/tick), so 15 snapshots.
- `--metrics=fid50k_full` **during** training, and only that one. It is not an
  endpoint here -- it is the COST BRAKE. Every snapshot is written to disk, so
  stopping early saves money and never saves quality: a late stop wastes dollars
  but cannot lose the good checkpoint. FID's within-run curve is what tells you
  when to stop paying, which is exactly how it was used on the previous project.
  15 evaluations is ~1.7 h of the 20 (~$0.46 on an A5000). Worth it.
- Do NOT add `kid50k_full` or `pr50k3_full` to this flag. They would triple the
  in-training metric cost for no decision you make on the pod. Both run offline
  from the downloaded snapshots via `calc_metrics.py` for zero rental time.
- γ5 / target 0.6 / `freezed=0` for the whole run. No phase schedule. If the
  discriminator overfits — ADA probability climbing toward 1.0, real/fake logit
  gap widening — **record it in `AUDIT_LOG.md` and do not intervene mid-run.**
  Changing hyperparameters on observed behaviour makes the run unreproducible
  and the stopping point results-driven.

---

## 2. Measure, after training, on the candidates only

Run this LOCALLY, on the downloaded snapshots, after the pod is destroyed.
Zero rental time.

```bash
for K in 2000 2400 2800 3000; do
  python calc_metrics.py --metrics=kid50k_full,pr50k3_full \
    --data=/path/to/clean_111.zip --mirror=1 \
    --network=/path/to/g_clean/network-snapshot-$(printf %06d $K).pkl
done
```

**KID, not FID, decides the snapshot.** FID is a BIASED estimator whose bias
grows as the reference set shrinks, and this reference set is 139 images. KID is
the unbiased small-sample estimator of the same discrepancy and is the correct
headline at this n. Both ship with the repo for the cost of one flag; the
previous project logged only FID, which was an oversight rather than a choice.

`pr50k3_full` (Kynkaanniemi precision/recall) covers fidelity and diversity, so
`s06_metrics.py` does not recompute them.

**FID here is a monitoring signal, not an endpoint.** It is computed against 139
reals, so it is valid as a within-run trend and meaningless as an absolute
number. Do not compare it to published FIDs on 70k-image datasets. The paper's memorisation
endpoints are authenticity and Meehan C_T, computed locally by
`scripts/s06_metrics.py` against the generator-blind holdout.

Why that module exists at all: EVERY metric in NVIDIA's `metrics/` module builds
its reference statistics from the TRAINING set -- there is no held-out data
anywhere in it. So a generator that perfectly memorises scores FID ~ 0,
precision 1.0 and recall 1.0, the best possible value on all of them at once.
`s06_metrics.py --selftest` demonstrates this numerically on toy data: a pure
copier scores precision 1.000 / recall 0.973 against an honest generator's
0.835 / 0.811. Closing that blind spot is s06's only job.

**Selection rule, prespecified:** lowest `kid50k_full` among snapshots
≥ 2,000 kimg; ties break to the earlier snapshot. The prior project's best snapshot was 41.69 at
~3,806 kimg, degrading to 43.01 by 4,293 — a fixed endpoint would have handed
over the worse generator.

---

## 3. Snapshot integrity — non-negotiable

All 110 snapshots from the previous project were silently truncated to identical
byte sizes (9,699,328 and 9,437,184) and are unrecoverable. The generator that
produced the published images no longer exists.

Sync off the pod **continuously**, not at the end:

```bash
runpodctl send /workspace/runs/g_clean/00000-*/network-snapshot-*.pkl
```

Verify on the pod after every snapshot:

```bash
python - <<'PY'
import glob, pickle, hashlib, os
for p in sorted(glob.glob('/workspace/runs/**/*.pkl', recursive=True)):
    try:
        with open(p,'rb') as f: d = pickle.load(f)
        ok = all(k in d for k in ('G','D','G_ema'))
    except Exception as e:
        ok = f'LOAD FAILED: {e}'
    h = hashlib.md5(open(p,'rb').read()).hexdigest()[:12]
    print(f"{os.path.getsize(p):>12} {h} {'OK' if ok is True else ok}  {p}")
PY
```

Identical sizes across different kimg means truncation. Stop and fix.

Then verify each transferred file loads **locally**, and compare md5 against the
pod, **before terminating the pod**. 15 snapshots ≈ 5 GB. A pod destroyed with
the only good copy on it repeats the original loss exactly.

---

## 4. Generate

v3 sampled at a single truncation, ψ=0.6, which suppresses the very diversity
this study measures.

```bash
BEST=/workspace/runs/g_clean/00000-*/network-snapshot-XXXXXX.pkl   # from §2
for PSI in 0.7 1.0; do
  python generate.py --outdir=/workspace/gen/clean_psi$PSI \
    --trunc=$PSI --seeds=0-1499 --network=$BEST
done
```

1,500 images per ψ supports doses up to 10× the 111 real RP with room to
resample. Seconds of GPU time.

Sync both directories off, verify locally, **then terminate the pod.**

---

## 5. Back on the local machine — no GPU from here

```bash
python variability_study/scripts/s03_novelty_metrics.py \
  --synth-dir gen/clean_psi1.0 --tag clean_psi1.0

python variability_study/scripts/s05_classifier.py \
  --synth-dir gen/clean_psi1.0 --tag clean_psi1.0
```

### Reference numbers — legacy v3, contaminated

Size-matched, Inception space. That generator trained on all 139 real RP
including every evaluation eye. **Upper bounds only.**

| metric | v3 (leaky) |
|---|---|
| Inception precision | 0.938 |
| Inception recall | **0.297** |
| coverage vs holdout | 1.000 |
| novel coverage gain, size-matched | 0.426 [0.286, 0.536] |
| memorisation flagged | 0 / 885 |

**Precision gates recall.** A recall gain accompanied by a precision or
authenticity drop is not novelty. Never report the two as co-equal numbers.
