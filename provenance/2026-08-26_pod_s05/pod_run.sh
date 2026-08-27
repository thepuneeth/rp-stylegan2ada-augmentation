#!/usr/bin/env bash
# s05 five-arm classifier sweep on a Runpod community RTX 4090.
# Runs INSIDE the pod. Assumes /workspace/payload.tar has been unpacked to /workspace/s05.
set -euo pipefail

cd /workspace/s05

# 100 fits x N workers exhausts the default 1024 fd limit.
ulimit -n 65535

echo "=== deps ==="
pip install -q scikit-learn 2>&1 | tail -2

echo "=== gate ==="
python -c "
import torch, sys
assert torch.cuda.is_available(), 'no CUDA -- s05 would silently fall back to CPU'
print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))
print('vCPU', __import__('os').cpu_count())
"
# Workers. NOT os.cpu_count() and NOT nproc: inside a Runpod container both
# report the HOST's core count (96 on the box this was written against) while
# the cgroup quota allowed 10.2. Spawning 94 workers onto 10 cores thrashes.
# cpu.max is the allocation that actually binds.
export S05_WORKERS=$(python - <<'PY'
import os
def cores():
    try:                                          # cgroup v2
        q, p = open('/sys/fs/cgroup/cpu.max').read().split()
        if q != 'max':
            return int(q) / int(p)
    except OSError:
        pass
    try:                                          # cgroup v1 -- what Runpod used
        q = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read())
        p = int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read())
        if q > 0:
            return q / p
    except OSError:
        pass
    return os.cpu_count()
print(max(2, int(cores()) - 2))
PY
)
echo "S05_WORKERS=$S05_WORKERS"

echo "=== sweep ==="
python -u scripts/s05_classifier.py --tag main5 \
  --arms real_only oversample pca_sampler synth_clean synth_leak \
  --synth-clean-dir gan_data/synth_clean_psi1.0 \
  --synth-leak-dir  gan_data/synth_leak_psi1.0 \
  --pca-dir gan_data/pca_pool

echo "=== done ==="
ls -la outputs/s05_classifier_main5.json
