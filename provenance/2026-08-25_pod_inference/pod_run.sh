#!/usr/bin/env bash
# Runs ON the pod. Ordered so the highest-value output lands first:
# gate -> pools (deliverable) -> cross-score (fixes F1) -> backfill (optional tail).
set -uo pipefail
cd /workspace
mkdir -p out/pools out/logs
export PYTHONPATH=/workspace/stylegan2-ada-pytorch-main
cd /workspace/stylegan2-ada-pytorch-main

echo "=== DEPS ==="
# bare pytorch container: none of these ship with it. ninja is not optional --
# without it the fused CUDA ops silently fall back to the slow reference path.
pip install -q ninja click requests scipy Pillow psutil imageio imageio-ffmpeg tqdm 2>&1 | tail -3
python -c "import click,scipy,PIL,psutil,requests;print('deps OK')" || exit 1

echo "=== GATE ==="
python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))" || exit 1
# _init() WARNS and returns False on failure rather than raising, so the return
# value must be checked -- calling it bare reports success on a silent fallback.
python -c "
from torch_utils.ops import bias_act, upfirdn2d
ok = bias_act._init() and upfirdn2d._init()
assert ok, 'custom CUDA ops fell back to reference implementation'
print('CUDA ops OK (compiled, not fallback)')" || { echo 'GATE FAIL: custom ops'; exit 1; }

echo "=== A. POOLS ==="
for spec in "clean-002822 synth_clean" "leak-003000 synth_leak"; do
  set -- $spec; PK=$1; NAME=$2
  for PSI in 1.0 0.7; do
    D=/workspace/out/pools/${NAME}_psi${PSI}
    echo "--- $D"
    python generate.py --network=/workspace/pkl/${PK}-Gema.pkl \
      --seeds=0-1109 --trunc=$PSI --noise-mode=const --outdir=$D \
      > /workspace/out/logs/gen_${NAME}_psi${PSI}.log 2>&1
    echo "    $(ls $D | wc -l) images"
  done
done
tar -czf /workspace/out/pools.tgz -C /workspace/out pools && echo "pools.tgz $(du -h /workspace/out/pools.tgz|cut -f1)"

echo "=== B. CROSS-SCORE (F1) ==="
M=fid50k_full,kid50k_full,pr50k3_full
python calc_metrics.py --network=/workspace/pkl/clean-002822-Gema.pkl \
  --data=/workspace/data/leak_111.zip --mirror=1 --metrics=$M --gpus=1 \
  2>&1 | tee /workspace/out/logs/cross_clean-002822_vs_leak111.log
python calc_metrics.py --network=/workspace/pkl/leak-003000-Gema.pkl \
  --data=/workspace/data/clean_111.zip --mirror=1 --metrics=$M --gpus=1 \
  2>&1 | tee /workspace/out/logs/cross_leak-003000_vs_clean111.log

echo "=== C. BACKFILL g_clean kid+pr ==="
for K in 000403 000806 001209 001612; do
  [ -f /workspace/pkl/clean-$K-Gema.pkl ] || { echo "skip $K (not uploaded)"; continue; }
  python calc_metrics.py --network=/workspace/pkl/clean-$K-Gema.pkl \
    --data=/workspace/data/clean_111.zip --mirror=1 \
    --metrics=kid50k_full,pr50k3_full --gpus=1 \
    2>&1 | tee /workspace/out/logs/backfill_clean-$K.log
done

tar -czf /workspace/out/logs.tgz -C /workspace/out logs
echo "=== POD JOB COMPLETE ==="
