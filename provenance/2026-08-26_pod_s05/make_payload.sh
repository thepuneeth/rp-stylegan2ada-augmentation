#!/usr/bin/env bash
# Build the upload tarball. Runs LOCALLY, from the repo root.
# Ships only what s05 opens: code, splits, the two real-image classes, the
# three synthetic pools, and the legacy v3 pool. Not the other 4 disease
# classes of the Eye Disease Image Dataset, which s05 never reads.
set -euo pipefail
OUT=${1:?usage: make_payload.sh <out.tar.gz>}

tar czf "$OUT" \
  config.py scripts/ splits/ \
  "Eye Disease Image Dataset/Original Dataset/Retinitis Pigmentosa" \
  "Eye Disease Image Dataset/Original Dataset/Healthy" \
  generated_rp_images_0007_201_trunc_6 \
  gan_data/synth_clean_psi1.0 gan_data/synth_leak_psi1.0 gan_data/pca_pool

echo "payload: $(du -h "$OUT" | cut -f1)"
