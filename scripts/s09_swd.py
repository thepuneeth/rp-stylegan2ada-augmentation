"""
Sliced Wasserstein Distance between a synthetic pool and a real reference set.

Carried over from the v3 paper's `metrics_code/realism_verfication_metrics.ipynb`,
which reported SWD alongside FID and KID. It is added here because F1 came back
a null in which FID, KID and precision disagreed on the ordering of the two
generators (see METHODS_NARRATIVE Part C1). SWD is a useful third opinion
precisely because its inductive bias is different: it compares multi-scale RAW
PIXEL PATCH statistics with random projections, and never touches Inception. A
metric that disagrees with FID and KID for a reason unrelated to Inception
features is informative; a fourth Inception-based metric would not be.

Lower is better. Reported per patch size, since fine and coarse scales fail in
different ways: small patches see texture and noise, large patches see layout.

    ../venv/bin/python scripts/s09_swd.py --self-test
    ../venv/bin/python scripts/s09_swd.py
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import config as C

PATCH_SIZES = (4, 8, 16, 32)
PATCHES_PER_IMAGE = 64
MAX_PATCHES = 200_000
N_PROJECTIONS = 512


def load_dir(d, size=256, limit=None):
    fs = sorted(f for f in os.listdir(d)
                if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if limit:
        fs = fs[:limit]
    out = []
    for f in fs:
        with Image.open(os.path.join(d, f)) as im:
            im = im.convert("RGB").resize((size, size), Image.BILINEAR)
            out.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0)
    return torch.stack(out)


def sample_patches(images, patch, per_image, gen):
    u = F.unfold(images, kernel_size=patch, stride=patch).transpose(1, 2)
    if per_image is not None and u.size(1) > per_image:
        idx = torch.randperm(u.size(1), generator=gen)[:per_image]
        u = u[:, idx]
    return u.reshape(-1, u.size(-1))


def swd(real_p, fake_p, n_proj, gen):
    """Mean absolute difference of sorted 1-D projections. Lower is closer."""
    m = min(real_p.size(0), fake_p.size(0))
    r = real_p[torch.randperm(real_p.size(0), generator=gen)[:m]]
    f = fake_p[torch.randperm(fake_p.size(0), generator=gen)[:m]]
    r = r - r.mean(0, keepdim=True)
    f = f - f.mean(0, keepdim=True)
    P = torch.randn(n_proj, r.size(1), generator=gen)
    P = P / (P.norm(dim=1, keepdim=True) + 1e-8)
    return float((torch.sort(r @ P.T, dim=0).values
                  - torch.sort(f @ P.T, dim=0).values).abs().mean())


def swd_multiscale(real_imgs, fake_imgs, seed=C.SEED):
    out = {}
    for p in PATCH_SIZES:
        g = torch.Generator().manual_seed(seed)      # same draw at every scale
        rp = sample_patches(real_imgs, p, PATCHES_PER_IMAGE, g)
        fp = sample_patches(fake_imgs, p, PATCHES_PER_IMAGE, g)
        for t in (rp, fp):
            if t.size(0) > MAX_PATCHES:
                pass
        out[f"patch{p}"] = swd(rp, fp, N_PROJECTIONS, g)
    out["mean"] = float(np.mean([out[f"patch{p}"] for p in PATCH_SIZES]))
    return out


def noise_floor(imgs, seed=C.SEED):
    """
    SWD of a reference set against ITSELF under independent patch draws.

    This is not zero. Patches are subsampled, so two draws from the same images
    differ by chance, and that chance difference is the floor below which no
    comparison means anything. Any pool-to-pool gap smaller than the floor is
    sampling noise being read as a result -- the same failure the saturated
    pixel-space Z_U statistic exhibited (METHODS_NARRATIVE Part C3).
    """
    return swd_multiscale(imgs, imgs.clone(), seed=seed)


def _self_test():
    """Identical inputs must sit at the noise floor; unrelated ones far above it."""
    g = torch.Generator().manual_seed(0)
    a = torch.rand(16, 3, 64, 64, generator=g)
    same = swd_multiscale(a, a.clone())["mean"]
    unrelated = swd_multiscale(a, torch.rand(16, 3, 64, 64, generator=g))["mean"]
    shifted = swd_multiscale(a, (a * 0.5 + 0.25))["mean"]
    assert unrelated > 5 * same, f"unrelated must clear the floor: {unrelated} vs {same}"
    assert shifted > 5 * same, f"contrast shift must be detected: {shifted} vs {same}"
    print(f"self-test OK  floor={same:.4f}  unrelated={unrelated:.4f}  "
          f"contrast-shifted={shifted:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test(); sys.exit(0)

    refs = {n: load_dir(os.path.join(C.GAN_DATA, n)) for n in ("clean_111", "leak_111")}
    floors = {n: noise_floor(imgs) for n, imgs in refs.items()}
    for n, f_ in floors.items():
        print(f"noise floor {n:10s} mean={f_['mean']:.4f}  "
              f"(differences below this are not readable)", flush=True)
    print()
    pools = ["synth_clean_psi1.0", "synth_clean_psi0.7",
             "synth_leak_psi1.0", "synth_leak_psi0.7"]
    res = {}
    for pool in pools:
        imgs = load_dir(os.path.join(C.GAN_DATA, pool))
        res[pool] = {rn: swd_multiscale(rimgs, imgs) for rn, rimgs in refs.items()}
        for rn in refs:
            r = res[pool][rn]
            print(f"{pool:22s} vs {rn:10s} " +
                  "  ".join(f"p{p}={r[f'patch{p}']:.4f}" for p in PATCH_SIZES) +
                  f"  mean={r['mean']:.4f}", flush=True)
    p = os.path.join(C.OUT, "s09_swd.json")
    json.dump({"patch_sizes": list(PATCH_SIZES), "n_projections": N_PROJECTIONS,
               "patches_per_image": PATCHES_PER_IMAGE, "seed": C.SEED,
               "noise_floor": floors, "results": res}, open(p, "w"), indent=2)
    print("wrote", p)
