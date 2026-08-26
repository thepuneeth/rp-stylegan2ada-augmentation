"""
s04b - the pca_sampler arm: synthetic RP images from a linear manifold.

Why this arm exists
-------------------
Contrast S2 asks whether a StyleGAN2-ADA generator (~3e7 parameters, ~9 GPU
hours) buys anything over the cheapest possible generative model fitted to the
same 111 images. PCA is that model: a mean image plus k orthogonal directions,
sampled as a Gaussian on the retained subspace. No GPU, seconds to fit.

If `synth_clean` cannot beat this, the deep generator is not adding structure
the data's linear subspace already contains.

Fitted on the SAME 111 dev eyes the generator trains on, so the two arms are
information-matched. It never sees a holdout eye.

    ../venv/bin/python scripts/s04b_pca_sampler.py --n 1110

Reproducible under C.SEED. Writes PNGs named pca0000.png.. so the pool is
interchangeable with generate.py's seed0000.png.. output in s05.
"""
import os
import sys
import json
import argparse

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import C  # noqa: E402


# macOS Accelerate BLAS emits spurious divide/overflow warnings from matmul.
# Same reason every s0*.py sets this.
np.seterr(all="ignore")


def load_dev_rp(size):
    """The 111 dev RP images as rows of a float64 matrix, plus their shape."""
    with open(os.path.join(C.SPLITS, "dev.json")) as fh:
        dev = json.load(fh)
    paths = [os.path.join(C.REAL_RP_DIR, f) for f in sorted(dev["RP"])]
    rows = []
    for p in paths:
        with Image.open(p) as im:
            a = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR),
                           dtype=np.float64)
        rows.append(a.ravel())
    return np.asarray(rows), (size, size, 3), paths


def fit_pca(X, var_keep):
    """
    Mean, components and per-component standard deviations.

    Plain SVD rather than sklearn.PCA so the returned singular values make the
    parameter count explicit -- the whole point of this arm is that its
    parameter count is small and stated.
    """
    mu = X.mean(0)
    Xc = X - mu
    # n=111 rows, so at most 110 non-degenerate directions exist regardless of
    # the 196,608 pixel dimensions.
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    keep = int(np.searchsorted(np.cumsum(var) / var.sum(), var_keep) + 1)
    keep = min(keep, Vt.shape[0] - 1)          # drop the last, always ~0
    # Per-component sd of the data's projection, the sampling scale.
    sd = S[:keep] / np.sqrt(len(X) - 1)
    return mu, Vt[:keep], sd


def sample(mu, comps, sd, n, shape, rng):
    """n images drawn as mu + sum_j z_j sd_j v_j, z ~ N(0, I)."""
    Z = rng.standard_normal((n, len(sd))) * sd
    out = Z @ comps + mu
    return np.clip(out, 0, 255).astype(np.uint8).reshape((n,) + shape)


def _selftest():
    """Cheap invariants; runs on `--selftest`."""
    rng = np.random.default_rng(0)
    # A rank-2 dataset embedded in 50 dimensions must be recovered as rank 2.
    basis = rng.standard_normal((2, 50))
    X = (rng.standard_normal((40, 2)) @ basis) * 10 + 128
    mu, comps, sd = fit_pca(X, 0.999)
    assert comps.shape[0] == 2, f"expected rank 2, got {comps.shape[0]}"
    # Components are orthonormal.
    assert np.allclose(comps @ comps.T, np.eye(len(comps)), atol=1e-8)
    # Samples land in byte range and carry the right shape.
    s = sample(mu, comps, sd, 5, (5, 5, 2), rng)
    assert s.shape == (5, 5, 5, 2) and s.dtype == np.uint8
    # Sampled spread tracks the fitted spread, not an arbitrary scale.
    big = sample(mu, comps, sd, 4000, (50,), rng).astype(np.float64)
    ratio = big.std(0).mean() / X.std(0).mean()
    assert 0.7 < ratio < 1.3, f"sample spread off by {ratio:.2f}x"
    print("s04b selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1110,
                    help="pool size; max DOSE_LEVELS (10.0) x 111 real RP")
    ap.add_argument("--size", type=int, default=C.IMG_SZ,
                    help="output resolution; C.IMG_SZ matches the 256px GAN pool")
    ap.add_argument("--var", type=float, default=0.95,
                    help="fraction of variance retained")
    ap.add_argument("--outdir", default=os.path.join(C.PROJECT, "variability_study",
                                                     "gan_data", "pca_pool"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return

    X, shape, paths = load_dev_rp(a.size)
    mu, comps, sd = fit_pca(X, a.var)
    rng = np.random.default_rng(C.SEED)
    imgs = sample(mu, comps, sd, a.n, shape, rng)

    os.makedirs(a.outdir, exist_ok=True)
    for i, im in enumerate(imgs):
        Image.fromarray(im).save(os.path.join(a.outdir, f"pca{i:04d}.png"))

    # State the parameter count rather than asserting it. comps dominates:
    # k x (size^2 x 3) floats. The claim "~1e4 params" in README/FINAL_PLAN
    # refers to the LATENT budget (n_train x k), not the stored basis.
    d = int(np.prod(shape))
    params = {"n_train": len(X), "k_components": int(len(comps)),
              "pixel_dim": d,
              "stored_floats_basis_and_mean": int(len(comps) * d + d),
              "latent_floats_n_train_x_k": int(len(X) * len(comps)),
              "variance_retained": float(a.var)}
    out = {"outdir": a.outdir, "n_generated": int(a.n), "size": a.size,
           "seed": C.SEED, "params": params,
           "fitted_on": [os.path.basename(p) for p in paths]}
    p = os.path.join(C.OUT, "s04b_pca_sampler.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"fitted on {len(X)} dev RP eyes, k={len(comps)} components "
          f"({a.var:.0%} variance)")
    print(f"  basis+mean floats {params['stored_floats_basis_and_mean']:,}")
    print(f"  latent floats     {params['latent_floats_n_train_x_k']:,}")
    print(f"wrote {a.n} images to {a.outdir}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
