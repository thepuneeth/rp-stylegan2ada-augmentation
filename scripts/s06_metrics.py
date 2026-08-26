"""
s06 - Snapshot-selection metrics: fidelity, coverage, authenticity, copying.

Why this exists
---------------
FID50k_full is computed here against 139 real images. It is usable as a
within-run trend and meaningless as an absolute number, so it cannot be the
rule that picks which training snapshot becomes the study's one generator.
This module supplies the criterion that can.

Three measurements, each answering a question FID conflates.

WHAT THIS DOES *NOT* DUPLICATE
-----------------------------
StyleGAN2-ADA already ships fid50k_full, kid50k_full and pr50k3_full
(Kynkaanniemi precision/recall), all computed on the GPU during training for
the cost of one flag. Fidelity and diversity are therefore NOT this module's
job, and an earlier draft that recomputed Alaa's alpha-precision/beta-recall
here was deleted as redundant.

Use the built-ins, with KID as the headline:

    --metrics=fid50k_full,kid50k_full,pr50k3_full

FID is a biased estimator and this dataset has 139 reals, so its absolute
value is meaningless. KID is the unbiased small-sample estimator of the same
quantity and is the correct headline at this n.

THE GAP THE BUILT-INS CANNOT CLOSE
----------------------------------
Every metric in NVIDIA's metrics/ module builds its reference statistics with
compute_feature_stats_for_dataset(opts) -- the TRAINING SET. Without
exception. No built-in metric has any notion of held-out data.

So a generator that perfectly memorises its training set scores FID ~ 0,
precision 1.0 and recall 1.0: the best possible value on every built-in
metric simultaneously. Measured on toy data in _selftest below, a pure copier
scores alpha-precision 0.973 and beta-recall 0.979 -- above an honest
generator on both.

This module exists only to close that gap, and it is the only place the
generator-blind holdout built in s01/s02 is used. Two measurements.

2. AUTHENTICITY  (Alaa et al., same paper, exact rule)
   Precision and recall are both maximised by a generator that memorises. A
   copy sits exactly on the real manifold (perfect precision) and, copied
   often enough, spans it (perfect recall). Authenticity is the third axis
   that separates the two: a synthetic sample is INAUTHENTIC when it is
   closer to its nearest real than that real is to its own nearest real
   neighbour -- i.e. it has crawled inside the real data's own spacing.

3. C_T DATA-COPYING TEST  (Meehan et al., AISTATS 2020)
   Authenticity is a per-sample heuristic. C_T is a hypothesis test, and it
   needs exactly the asset this study spent s01/s02 building: a
   generator-blind holdout.

     L(S, R)  distance from each synthetic sample to the nearest GAN-TRAINING
              real
     L(H, R)  distance from each HOLDOUT real to the nearest training real

   Under a generator that models the distribution rather than the sample,
   these two are drawn from the same law. Z_U is the standardised
   Mann-Whitney statistic between them, and it is SIGNED:

     Z_U << 0   synthetic sits closer to the training reals than genuinely
                new eyes do  ->  data copying
     Z_U ~  0   indistinguishable from a fresh draw  ->  no evidence of copying
     Z_U >> 0   synthetic is FURTHER from the training reals than new eyes
                are  ->  underfitting, off-manifold

   Global Z_U misses copying that is confined to one region, which is the
   normal failure mode. Meehan's fix is to partition instance space (k-means
   on the training reals) and aggregate per-cell Z_U weighted by the holdout
   mass in each cell. Both are reported. With 28 holdout eyes the per-cell
   test is thin, so cells below MIN_CELL are dropped and the dropped mass is
   reported rather than hidden.

Distances
---------
Euclidean throughout, in every space. Pixel features from common.corr_features
are unit-norm, so euclidean distance there is a monotone function of
correlation distance (d^2 = 2 - 2r) -- every rank-based statistic below is
therefore identical to the correlation-space version, and comparability with
the s01a-calibrated threshold survives.

Outputs outputs/s06_metrics_<tag>.json
"""
import os, sys, json, argparse
import numpy as np
from scipy.stats import mannwhitneyu

from common import C, corr_features
from s03_novelty_metrics import inception_features, prd_metrics

np.seterr(all="ignore")

MIN_CELL = 5          # holdout points required before a k-means cell is tested
N_CELLS = 3           # Meehan k; 28 holdout eyes will not support more


# ------------------------------------------------------------------ features
def retfound_features(paths, batch=16):
    """
    RETFound (ViT-L MAE, 1.6M retinal images, Nature 2023) CLS embedding.

    The primary space per config.PRIMARY_FEATURE_SPACE. Inception encodes
    ImageNet semantics and has no notion of bone-spicule pigmentation or
    arteriolar attenuation; a retinal foundation model does.

    Raises rather than falling back. Silently scoring the primary contrast in
    a substitute feature space would be the kind of error this package exists
    to prevent.
    """
    try:
        import timm, torch
    except ImportError as e:
        raise RuntimeError(
            "RETFound needs timm: `../venv/bin/pip install timm`. "
            f"({e})") from e
    import torch
    from PIL import Image
    from torchvision import transforms

    w = os.environ.get("RETFOUND_WEIGHTS")
    if not w or not os.path.isfile(w):
        raise RuntimeError(
            "Set RETFOUND_WEIGHTS to the RETFound_mae_natureCFP checkpoint. "
            "Weights: https://github.com/rmaphoh/RETFound_MAE (CFP variant). "
            "Not bundled -- ~1.2 GB, and licence acceptance is the owner's.")

    net = timm.create_model("vit_large_patch16_224", pretrained=False,
                            num_classes=0, global_pool="")
    sd = torch.load(w, map_location="cpu")
    sd = sd.get("model", sd)
    missing = net.load_state_dict(
        {k: v for k, v in sd.items() if not k.startswith("decoder")},
        strict=False)
    print(f"  RETFound loaded; missing={len(missing.missing_keys)} "
          f"unexpected={len(missing.unexpected_keys)}")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net.eval().to(dev)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            ims = []
            for p in paths[i:i + batch]:
                with Image.open(p) as im:
                    ims.append(tf(im.convert("RGB")))
            out.append(net(torch.stack(ims).to(dev))[:, 0].cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def pixel_features(paths):
    return corr_features(paths, dtype=np.float32)


FEATURES = {"retfound": retfound_features,
            "inception": inception_features,
            "pixel": pixel_features}


# ------------------------------------------------------------------- helpers
def cdist(A, B):
    """Euclidean. See module docstring on why this is correct in every space."""
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.sqrt(np.maximum(d2, 0.0))


def nn_dist(A, B, exclude_self=False):
    """Distance from each row of A to its nearest row of B."""
    D = cdist(A, B)
    if exclude_self:
        np.fill_diagonal(D, np.inf)
    return D.min(axis=1), D.argmin(axis=1)


# --------------------------------------------------------- Alaa authenticity
def _auth_frac(R, S, d_rr):
    d_sr, idx = nn_dist(S, R)
    return float((d_sr > d_rr[idx]).mean())


def authenticity(R, S, H=None, n_boot=200, seed=C.SEED):
    """
    Alaa's exact rule. A synthetic sample is INAUTHENTIC when it is nearer to
    its closest real than that real is to its own nearest real neighbour: it
    has landed inside the real data's own spacing, which sampling from the
    distribution has no reason to do but copying-with-noise does.

    THE RAW FRACTION HAS NO INTERPRETABLE SCALE, and quoting it alone is an
    error. The statistic is a minimum over |S| draws, so it falls as |S| grows
    for reasons that have nothing to do with copying -- on pure Gaussian data
    with 400 synthetic against 111 real it reads 0.66, not 1.0. Two corrections,
    both mandatory, both mirroring the size-matching discipline s03 applies to
    novel coverage gain:

      matched   authenticity of random |H|-sized subsets of S, bootstrapped
      null      the same statistic with H -- genuinely fresh REAL eyes -- in
                place of S. This is what "not copying" actually scores on this
                cohort at this sample size.

    `authenticity_vs_null` is the number to read: >= 0 is indistinguishable
    from a fresh real draw, and negative is the copying direction.
    """
    d_rr, _ = nn_dist(R, R, exclude_self=True)
    d_sr, idx = nn_dist(S, R)
    auth = d_sr > d_rr[idx]
    out = {"authenticity_raw": float(auth.mean()),
           "n_inauthentic": int((~auth).sum()),
           "n_synth": int(len(S))}
    if H is None:
        return out

    rng = np.random.default_rng(seed)
    m = min(len(H), len(S))
    vals = [_auth_frac(R, S[rng.choice(len(S), m, replace=False)], d_rr)
            for _ in range(n_boot)]
    null = _auth_frac(R, H, d_rr)
    out.update({
        "matched_subset_size": int(m),
        "authenticity_matched": float(np.mean(vals)),
        "authenticity_matched_ci95": [float(np.percentile(vals, 2.5)),
                                      float(np.percentile(vals, 97.5))],
        "authenticity_null_real_holdout": null,
        "authenticity_vs_null": float(np.mean(vals) - null),
        "below_null": bool(np.percentile(vals, 97.5) < null),
    })
    return out


# ---------------------------------------------------------- Meehan et al. C_T
def _zu(dS, dH):
    """
    Standardised Mann-Whitney U between synthetic-to-train and
    holdout-to-train nearest-neighbour distances. Sign convention as in the
    module docstring: negative means copying.
    """
    m, n = len(dS), len(dH)
    if m < 2 or n < 2:
        return None
    U = mannwhitneyu(dS, dH, alternative="two-sided", method="asymptotic").statistic
    mu = m * n / 2.0
    sd = np.sqrt(m * n * (m + n + 1) / 12.0)
    return float((U - mu) / sd)


def ct_test(H, R, S, n_cells=N_CELLS, seed=C.SEED):
    """
    Global and per-cell C_T. Cells come from k-means on the TRAINING reals,
    never on the synthetic pool -- the partition must not depend on what is
    being tested.
    """
    dS, _ = nn_dist(S, R)
    dH, _ = nn_dist(H, R)
    out = {"n_synth": int(len(S)), "n_holdout": int(len(H)),
           "n_train": int(len(R)),
           "median_d_synth_to_train": float(np.median(dS)),
           "median_d_holdout_to_train": float(np.median(dH)),
           "Z_U_global": _zu(dS, dH)}

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_cells, n_init=10, random_state=seed).fit(R)
    lS, lH = km.predict(S), km.predict(H)

    cells, used_mass = [], 0.0
    for c in range(n_cells):
        mS, mH = lS == c, lH == c
        z = _zu(dS[mS], dH[mH]) if mH.sum() >= MIN_CELL else None
        cells.append({"cell": c, "n_synth": int(mS.sum()),
                      "n_holdout": int(mH.sum()), "Z_U": z,
                      "tested": z is not None})
        if z is not None:
            used_mass += mH.sum()

    if used_mass > 0:
        out["C_T"] = float(sum(c["Z_U"] * c["n_holdout"] / used_mass
                               for c in cells if c["tested"]))
    else:
        out["C_T"] = None
    out["cells"] = cells
    out["n_cells"] = n_cells
    out["min_cell"] = MIN_CELL
    out["holdout_mass_tested"] = float(used_mass / len(H))
    return out


# ------------------------------------------------------------------- driver
def score(H, R, S):
    d = authenticity(R, S, H=H)
    d["ct"] = ct_test(H, R, S)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", action="append", required=True,
                    help="repeatable; one per snapshot to rank")
    ap.add_argument("--space", default=C.PRIMARY_FEATURE_SPACE,
                    choices=list(FEATURES))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--leaky", action="store_true",
                    help="generator saw the holdout; C_T and authenticity are "
                         "upper bounds and support no clean claim")
    a = ap.parse_args()
    tag = a.tag or a.space

    with open(os.path.join(C.SPLITS, "dev.json")) as fh:
        dev = json.load(fh)
    with open(os.path.join(C.SPLITS, "holdout.json")) as fh:
        hold = json.load(fh)
    feat = FEATURES[a.space]
    R = feat([os.path.join(C.REAL_RP_DIR, f) for f in sorted(dev["RP"])])
    H = feat([os.path.join(C.REAL_RP_DIR, f) for f in sorted(hold["RP"])])
    print(f"space={a.space}  train={len(R)}  holdout={len(H)}")

    res = {"space": a.space, "leaky": a.leaky, "seed": C.SEED,
           "n_train": len(R), "n_holdout": len(H), "snapshots": {}}
    if a.leaky:
        res["interpretation_warning"] = (
            "Generator trained on holdout images. Authenticity and C_T are "
            "contaminated by memorisation; upper bounds only.")

    for d in a.synth_dir:
        name = os.path.basename(d.rstrip("/"))
        paths = [os.path.join(d, f) for f in sorted(os.listdir(d))
                 if not f.startswith(".")]
        S = feat(paths)
        r = score(H, R, S)
        res["snapshots"][name] = r
        ct = r["ct"]
        print(f"\n{name}  n_synth={len(S)}")
        print(f"  authenticity {r['authenticity_matched']:.3f} "
              f"vs real-holdout null {r['authenticity_null_real_holdout']:.3f} "
              f"(delta {r['authenticity_vs_null']:+.3f}"
              f"{', BELOW NULL' if r['below_null'] else ''})")
        ctv = "n/a" if ct["C_T"] is None else f'{ct["C_T"]:.2f}'
        print(f"  C_T {ctv}   Z_U_global {ct['Z_U_global']:.2f}"
              f"   holdout mass tested {ct['holdout_mass_tested']:.2f}")

    # No ranking here. Snapshot selection is by kid50k_full, logged by the
    # training run itself; this module only says whether the selected snapshot
    # memorised, which is a pass/fail check rather than a score to sort on.
    p = os.path.join(C.OUT, f"s06_metrics_{tag}.json")
    with open(p, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {p}")


def _selftest():
    """
    Gaussian toy data with known ground truth. Runs on `python scripts/s06_metrics.py --selftest`.
    """
    rng = np.random.default_rng(0)
    R = rng.normal(size=(111, 16))
    H = rng.normal(size=(28, 16))

    # A copier: synthetic = training reals plus negligible jitter.
    S_copy = R[rng.integers(0, len(R), 400)] + rng.normal(scale=1e-3, size=(400, 16))
    # An honest generator: fresh draws from the same law.
    S_good = rng.normal(size=(400, 16))
    # A collapsed generator: one mode only.
    S_collapse = R[0] + rng.normal(scale=1e-2, size=(400, 16))

    c, g, k = score(H, R, S_copy), score(H, R, S_good), score(H, R, S_collapse)

    # Copying is caught on both axes: samples crawl inside the reals' own
    # spacing, well below what a fresh REAL draw scores, and they sit nearer
    # the training set than fresh eyes do.
    assert c["below_null"], (c["authenticity_matched"], c["authenticity_null_real_holdout"])
    assert c["authenticity_vs_null"] < -0.5, c["authenticity_vs_null"]
    assert c["ct"]["Z_U_global"] < -5, c["ct"]["Z_U_global"]

    # An honest generator is indistinguishable from a fresh real draw on both.
    # Note its RAW authenticity is only ~0.66 -- the null is what makes that
    # readable, and is why the raw number must never be quoted alone.
    assert not g["below_null"], (g["authenticity_matched"], g["authenticity_null_real_holdout"])
    assert abs(g["authenticity_vs_null"]) < 0.15, g["authenticity_vs_null"]
    assert abs(g["ct"]["Z_U_global"]) < 3, g["ct"]["Z_U_global"]

    # The justification for this module existing, kept as a live check.
    # prd_metrics is Kynkaanniemi k-NN precision/recall -- the same estimator
    # StyleGAN2-ADA ships as pr50k3_full, and like it, referenced against the
    # TRAINING set. The copier must beat the honest generator on both, which
    # is exactly why a built-in-only protocol cannot detect memorisation.
    pc = prd_metrics(R, S_copy, "euclidean")
    pg = prd_metrics(R, S_good, "euclidean")
    pk = prd_metrics(R, S_collapse, "euclidean")
    assert pc["precision"] >= pg["precision"], (pc, pg)
    assert pc["recall"] >= pg["recall"], (pc, pg)
    # Mode collapse the built-ins DO catch, via recall.
    assert pk["recall"] < pg["recall"] - 0.3, (pk, pg)

    for nm, r in [("copy", c), ("honest", g), ("collapsed", k)]:
        print(f"{nm:10s} authen raw {r['authenticity_raw']:.3f} "
              f"matched {r['authenticity_matched']:.3f} "
              f"null {r['authenticity_null_real_holdout']:.3f} "
              f"delta {r['authenticity_vs_null']:+.3f}  "
              f"Z_U {r['ct']['Z_U_global']:+.2f}")
    print(f"built-in pr50k3-style (training-set reference, NO holdout):")
    for nm, pr in [("copy", pc), ("honest", pg), ("collapsed", pk)]:
        print(f"{nm:10s} precision {pr['precision']:.3f}  recall {pr['recall']:.3f}")
    print("s06 selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
