"""
s03 - Does the generator produce genuinely NEW variability, or does it copy?

This module is the measurement apparatus for the study's central question. It is
deliberately separate from any classifier, so "the GAN adds variability" can be
tested directly on the images rather than inferred from downstream accuracy.

The question, made precise
--------------------------
Random oversampling duplicates existing minority images. It adds zero new
information -- every "new" sample sits exactly on top of a training image. So
"beyond copying" has an exact operational meaning, and the generator has to
clear a bar that copying cannot.

    NOVEL COVERAGE GAIN (the headline metric)

    For each generator-blind held-out real RP image h:
        d_train(h) = min distance from h to any GAN-TRAINING real image
        d_synth(h) = min distance from h to any SYNTHETIC image
    NCG = fraction of h with d_synth(h) < d_train(h)

    Under pure copying NCG is exactly 0: synthetic images coincide with training
    reals, so d_synth == d_train. NCG significantly above 0 means the generator
    produced images that are closer to REAL EYES IT NEVER SAW than any of its
    training images were. That is new variability which is also correct -- it
    lands on the real manifold rather than in noise.

    This is only meaningful because the holdout is generator-blind. If the
    generator had seen h, a low d_synth(h) would just be memorisation. That is
    exactly the confound the s02 split removes, and it is why the legacy v3
    generator (trained on all 139, holdout included) can only ever furnish an
    upper bound.

Supporting measurements
-----------------------
1. MEMORISATION, a necessary condition. Using the duplicate metric calibrated in
   s01a -- FOV-cropped correlation, threshold 0.99, zero false positives against
   five confirmed duplicate pairs -- count synthetic images that would be
   clustered as the SAME EYE as a training real. This is a sharper test than the
   Inception-distance argument used in v3, because the threshold was validated
   against known duplicates rather than assumed.

2. PRECISION / RECALL / DENSITY / COVERAGE in Inception feature space, computed
   against both the training reals and the blind holdout reals. Precision drops
   if the generator hallucinates off-manifold; recall and coverage rise only if
   it spans regions the reference set occupies. Reported for comparability with
   the generative-modelling literature.

3. INTRA-SET DIVERSITY. Mean pairwise distance within the synthetic pool against
   the same within the training reals. A mode-collapsed generator scores low
   here even when precision looks healthy.

Outputs outputs/s03_novelty_<tag>.json
"""
import os, sys, json, argparse
import numpy as np
from PIL import Image
from scipy.stats import wilcoxon, binomtest

from common import C, corr_features

np.seterr(all="ignore")


# ------------------------------------------------------------------ features
def pixel_features(paths, sz=C.CORR_SZ):
    """
    The s01 eye-identity feature, imported rather than reimplemented, so
    distances here are directly comparable to the validated duplicate
    threshold. float32 storage: 139 x 65536 float64 is 73 MB per set.
    """
    return corr_features(paths, sz=sz, dtype=np.float32)


def inception_features(paths, batch=32):
    """
    2048-d pool3 features from ImageNet InceptionV3, the standard space for
    FID/precision/recall. Weights are already cached locally.
    """
    import torch, torchvision
    from torchvision import transforms

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = torchvision.models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
    net.fc = torch.nn.Identity()
    net.eval().to(dev)

    tf = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            ims = []
            for p in paths[i:i + batch]:
                with Image.open(p) as im:
                    ims.append(tf(im.convert("RGB")))
            feats.append(net(torch.stack(ims).to(dev)).cpu().numpy())
    return np.concatenate(feats).astype(np.float32)


def pairwise(A, B, kind):
    """Distance matrix. Correlation distance for pixels, Euclidean for Inception."""
    if kind == "pixel":
        return 1.0 - (A @ B.T)
    a2 = (A ** 2).sum(1)[:, None]
    b2 = (B ** 2).sum(1)[None, :]
    return np.sqrt(np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0))


# ------------------------------------------------------- generative manifolds
def knn_radius(D_self, k):
    """Distance to the k-th nearest OTHER member; defines each ball's radius."""
    D = D_self.copy()
    np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1)[:, k - 1]


def prd_metrics(R, S, kind, k=3):
    """
    Kynkaanniemi precision/recall plus Naeem density/coverage.

    precision  synthetic points inside the real manifold      (fidelity)
    recall     real points inside the synthetic manifold      (mode span)
    density    robust precision, counts overlapping balls
    coverage   fraction of real balls containing a synthetic  (robust recall)
    """
    Drr, Dss = pairwise(R, R, kind), pairwise(S, S, kind)
    Drs = pairwise(R, S, kind)
    rr, rs = knn_radius(Drr, k), knn_radius(Dss, k)

    in_real = (Drs <= rr[:, None]).any(axis=0)      # per synthetic
    in_synth = (Drs <= rs[None, :]).any(axis=1)     # per real
    return {
        "precision": float(in_real.mean()),
        "recall": float(in_synth.mean()),
        "density": float((Drs <= rr[:, None]).sum(axis=0).mean() / k),
        "coverage": float((Drs.min(axis=1) <= rr).mean()),
        "k": k,
    }


def intra_diversity(X, kind):
    D = pairwise(X, X, kind)
    iu = np.triu_indices(len(X), 1)
    return {"mean_pairwise": float(D[iu].mean()),
            "median_pairwise": float(np.median(D[iu]))}


# ------------------------------------------------------------------ the study
def novel_coverage(H, R, S, kind, n_boot=C.NCG_BOOTSTRAP, seed=C.SEED):
    """
    The headline metric. H = generator-blind holdout reals, R = GAN training
    reals, S = synthetic.

    SIZE MATCHING IS MANDATORY. d_synth is a minimum over |S| samples and
    d_train a minimum over |R|. A minimum over more draws is mechanically
    smaller, so an unmatched comparison of 885 synthetic against 111 real
    manufactures novelty out of sample size alone. Every reported figure below
    therefore uses synthetic subsets of exactly |R| images, bootstrapped n_boot
    times. The unmatched value is retained only to expose the size of the
    artifact, and must never be quoted as a result.

    The copying null is enforced by construction elsewhere: replacing S with a
    resample of R yields NCG = 0, since d_synth then equals d_train exactly.
    """
    rng = np.random.default_rng(seed)
    d_train = pairwise(H, R, kind).min(axis=1)
    D_hs = pairwise(H, S, kind)

    d_unmatched = D_hs.min(axis=1)
    m = min(len(R), len(S))

    gains, ratios, wins = [], [], np.zeros(len(H))
    for _ in range(n_boot):
        idx = rng.choice(len(S), size=m, replace=False)
        d_s = D_hs[:, idx].min(axis=1)
        closer = d_s < d_train
        gains.append(closer.mean())
        ratios.append(np.median(d_s / d_train))
        wins += closer
    gains, ratios = np.array(gains), np.array(ratios)

    # Test on the per-eye win rate: how often each holdout eye is better served
    # by a size-matched synthetic subset than by the whole training set.
    per_eye = wins / n_boot
    n_closer = int((per_eye > 0.5).sum())
    bt = binomtest(n_closer, len(H), p=0.5, alternative="greater")
    try:
        wp = float(wilcoxon(per_eye - 0.5, alternative="greater").pvalue)
    except ValueError:
        wp = 1.0

    return {
        "n_holdout": int(len(H)),
        "n_train_real": int(len(R)),
        "n_synth_total": int(len(S)),
        "matched_subset_size": int(m),
        "n_bootstrap": int(n_boot),
        "novel_coverage_gain": float(gains.mean()),
        "novel_coverage_gain_ci95": [float(np.percentile(gains, 2.5)),
                                     float(np.percentile(gains, 97.5))],
        # Two distinct questions, and the paper must not conflate them.
        #   vs 0.0  "is there ANY novelty beyond copying?"  Copying gives
        #           exactly 0, so a CI excluding 0 is evidence of novelty.
        #   vs 0.5  "does synthetic cover unseen eyes BETTER than the real
        #           training set does?"  A far stronger claim.
        "beyond_copying": bool(np.percentile(gains, 2.5) > 0.0),
        "frac_bootstraps_at_zero": float((gains == 0.0).mean()),
        "better_than_real_train": bool(np.percentile(gains, 2.5) > 0.5),
        "novel_coverage_gain_UNMATCHED_artifact": float((d_unmatched < d_train).mean()),
        "n_eyes_favouring_synth": n_closer,
        "median_d_train": float(np.median(d_train)),
        "median_ratio_synth_over_train": float(ratios.mean()),
        "binomial_p_vs_half": float(bt.pvalue),
        "wilcoxon_p": wp,
        "per_holdout_win_rate": per_eye.tolist(),
        "per_holdout_d_train": d_train.tolist(),
    }


def copying_control(H, R, kind, n_boot=C.NCG_BOOTSTRAP, seed=C.SEED):
    """
    Sanity check on the metric itself. "Synthetic" is replaced by a bootstrap
    resample of the training reals, i.e. literal oversampling. A correct
    implementation must return a novel coverage gain of 0: copies cannot be
    closer to an unseen eye than the originals they were copied from.
    A non-zero value here means the metric is broken, not that copying works.
    """
    rng = np.random.default_rng(seed)
    d_train = pairwise(H, R, kind).min(axis=1)
    D_hr = pairwise(H, R, kind)
    gains = []
    for _ in range(n_boot):
        idx = rng.choice(len(R), size=len(R), replace=True)
        gains.append((D_hr[:, idx].min(axis=1) < d_train).mean())
    return {"novel_coverage_gain": float(np.mean(gains)),
            "expected": 0.0,
            "passes": bool(np.mean(gains) == 0.0)}


def memorisation(R, S):
    """
    Synthetic images that the validated eye-identity metric would call the SAME
    EYE as a training real. Pixel features only -- the threshold is calibrated
    in that space and nowhere else.
    """
    sim = S @ R.T                                   # correlation
    best = sim.max(axis=1)
    return {
        "threshold": C.CORR_THRESHOLD,
        "n_synth": int(len(S)),
        "n_flagged_same_eye": int((best >= C.CORR_THRESHOLD).sum()),
        "frac_flagged": float((best >= C.CORR_THRESHOLD).mean()),
        "max_similarity_to_train": float(best.max()),
        "p99_similarity_to_train": float(np.percentile(best, 99)),
        "median_similarity_to_train": float(np.median(best)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", default=C.LEGACY_SYNTH_DIR)
    ap.add_argument("--tag", default="legacy_v3")
    ap.add_argument("--leaky", action="store_true",
                    help="generator saw the holdout; results are an upper bound only")
    a = ap.parse_args()

    with open(os.path.join(C.SPLITS, "dev.json")) as fh:
        dev = json.load(fh)
    with open(os.path.join(C.SPLITS, "holdout.json")) as fh:
        hold = json.load(fh)

    train_rp = [os.path.join(C.REAL_RP_DIR, f) for f in sorted(dev["RP"])]
    hold_rp = [os.path.join(C.REAL_RP_DIR, f) for f in sorted(hold["RP"])]
    synth = [os.path.join(a.synth_dir, f) for f in
             sorted(os.listdir(a.synth_dir)) if not f.startswith(".")]

    print(f"tag={a.tag}  train_rp={len(train_rp)}  holdout_rp={len(hold_rp)}  "
          f"synth={len(synth)}  leaky={a.leaky}")

    res = {"tag": a.tag, "synth_dir": a.synth_dir, "leaky": a.leaky,
           "n_train_rp": len(train_rp), "n_holdout_rp": len(hold_rp),
           "n_synth": len(synth)}
    if a.leaky:
        res["interpretation_warning"] = (
            "Generator was trained on all 139 real RP including the holdout. "
            "Novel coverage here is an UPPER BOUND contaminated by memorisation "
            "and is not evidence of genuine novelty.")

    for kind in ["pixel", "inception"]:
        print(f"\n--- {kind} space ---")
        f = pixel_features if kind == "pixel" else inception_features
        R, H, S = f(train_rp), f(hold_rp), f(synth)

        nc = novel_coverage(H, R, S, kind)
        ctrl = copying_control(H, R, kind)
        assert ctrl["passes"], (
            f"copying control failed in {kind} space: NCG="
            f"{ctrl['novel_coverage_gain']}, expected 0. Metric is broken.")
        prd = prd_metrics(R, S, kind)
        prd_hold = prd_metrics(H, S, kind)
        div_s, div_r = intra_diversity(S, kind), intra_diversity(R, kind)

        res[kind] = {"novel_coverage": nc, "copying_control": ctrl,
                     "prd_vs_train": prd, "prd_vs_holdout": prd_hold,
                     "intra_diversity_synth": div_s,
                     "intra_diversity_train_real": div_r}
        if kind == "pixel":
            res["memorisation"] = memorisation(R, S)

        lo, hi = nc["novel_coverage_gain_ci95"]
        print(f"  NOVEL COVERAGE GAIN {nc['novel_coverage_gain']:.3f} "
              f"[{lo:.3f}, {hi:.3f}]  (size-matched: {nc['matched_subset_size']} "
              f"synthetic vs {nc['n_train_real']} real, {nc['n_bootstrap']} boots)")
        print(f"    unmatched artifact would have read "
              f"{nc['novel_coverage_gain_UNMATCHED_artifact']:.3f}  <- not a result")
        print(f"    {nc['n_eyes_favouring_synth']}/{nc['n_holdout']} holdout eyes "
              f"favour synthetic;  binomial p={nc['binomial_p_vs_half']:.4g}  "
              f"wilcoxon p={nc['wilcoxon_p']:.4g}  "
              f"median ratio {nc['median_ratio_synth_over_train']:.3f}")
        print(f"    beyond copying (CI excludes 0): {nc['beyond_copying']}   "
              f"better than real train (CI excludes 0.5): {nc['better_than_real_train']}")
        print(f"    copying control NCG={ctrl['novel_coverage_gain']:.3f} (must be 0) OK")
        print(f"  vs TRAIN   precision {prd['precision']:.3f} recall {prd['recall']:.3f} "
              f"density {prd['density']:.3f} coverage {prd['coverage']:.3f}")
        print(f"  vs HOLDOUT precision {prd_hold['precision']:.3f} "
              f"recall {prd_hold['recall']:.3f} coverage {prd_hold['coverage']:.3f}")
        print(f"  intra-set mean pairwise  synth {div_s['mean_pairwise']:.4f}  "
              f"real {div_r['mean_pairwise']:.4f}  "
              f"ratio {div_s['mean_pairwise']/div_r['mean_pairwise']:.3f}")
        if kind == "pixel":
            m = res["memorisation"]
            print(f"  MEMORISATION  {m['n_flagged_same_eye']}/{m['n_synth']} synthetic "
                  f"flagged same-eye as a training real (thr {m['threshold']}), "
                  f"max sim {m['max_similarity_to_train']:.4f}")

    p = os.path.join(C.OUT, f"s03_novelty_{a.tag}.json")
    with open(p, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
