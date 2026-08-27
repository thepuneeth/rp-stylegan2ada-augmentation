"""
Shared image primitives for the variability study.

Extracted because eye identity (s01) and any distance compared to the
duplicate threshold depend on the SAME crop and the SAME correlation feature.
They are only comparable if the feature is byte-identical, so one definition
makes that structural rather than an unenforced comment.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402  (path must be set first)


def fov_crop(im_gray, thresh=12):
    """
    Crop to the illuminated circular field of view.

    Fundus photographs sit on a near-black rectangular canvas. Distances
    computed over that canvas are inflated by the shared background, so the
    bounding box of the illuminated region is taken first. Falls back to the
    full frame if the mask is empty or implausibly small.
    """
    a = np.asarray(im_gray, dtype=np.float32)
    mask = a > thresh
    if mask.sum() < 0.05 * mask.size:
        return im_gray
    ys, xs = np.where(mask)
    return im_gray.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def corr_features(paths, sz=C.CORR_SZ, dtype=np.float64):
    """
    Zero-mean unit-norm FOV-cropped grayscale, one row per path.

    Row dot product = Pearson correlation, so 1 - dot is the distance the
    duplicate threshold (C.CORR_THRESHOLD) was calibrated against.
    Accumulation is always float64; `dtype` sets only the storage precision.
    """
    X = np.empty((len(paths), sz * sz), dtype=dtype)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            g = fov_crop(im.convert("L")).resize((sz, sz), Image.BILINEAR)
        a = np.asarray(g, dtype=np.float64)
        a -= a.mean()
        n = np.linalg.norm(a)
        X[i] = (a / (n if n > 1e-6 else 1.0)).ravel()
    return X


def list_images(d):
    """Sorted filenames in `d`, skipping dotfiles (.DS_Store)."""
    return sorted(f for f in os.listdir(d) if not f.startswith("."))


def _selftest():
    """Cheap invariants. Runs on `python scripts/common.py`."""
    # macOS Accelerate BLAS emits spurious divide/overflow warnings from
    # matmul on small arrays. Same reason every s0*.py sets this.
    np.seterr(all="ignore")
    # fov_crop finds the illuminated box.
    a = np.zeros((64, 64), dtype=np.uint8)
    a[10:40, 20:50] = 200
    assert fov_crop(Image.fromarray(a)).size == (30, 30)

    # A mask below the 5% floor falls back to the full frame.
    b = np.zeros((64, 64), dtype=np.uint8)
    b[0:5, 0:5] = 200
    assert fov_crop(Image.fromarray(b)).size == (64, 64)

    # Rows are unit-norm and zero-mean, so self-dot is 1 and a shifted-and-
    # scaled copy still correlates at 1 -- the property the threshold assumes.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rng = np.random.default_rng(0)
        img = (rng.random((64, 64)) * 200 + 30).astype(np.uint8)
        p1 = os.path.join(td, "a.png")
        p2 = os.path.join(td, "b.png")
        Image.fromarray(img).save(p1)
        Image.fromarray(np.clip(img * 0.5 + 20, 0, 255).astype(np.uint8)).save(p2)
        X = corr_features([p1, p2], sz=32)
    assert abs(np.linalg.norm(X[0]) - 1.0) < 1e-9
    assert abs(X[0].mean()) < 1e-9
    assert X[0] @ X[1] > 0.99

    print("common.py selftest OK")


if __name__ == "__main__":
    _selftest()
