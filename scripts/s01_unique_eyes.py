"""
s01 - How many UNIQUE eyes does the cohort actually contain?

Why this runs first
-------------------
"139 RP images" is a file count, not an eye count. If the same eye lands in both
the GAN training set and the generator-blind holdout, the holdout is not blind
and the central claim of this study collapses. Every downstream step needs an
EYE-DISJOINT split, which needs eye identity.

Method, and why it is this one
------------------------------
A first attempt used mean absolute difference on 128x128 RGB with single-linkage
at MAE < 12. It failed loudly: 1024 healthy images collapsed to 10 "eyes", one
component holding 842. Recorded here so the failure is not repeated.

  Cause 1, no separation. The pairwise MAE histogram rises smoothly from 3.9
  with no bimodality. Every fundus from one camera shares a circular field of
  view, a dark surround and a bright optic disc, and that shared template
  dominates the distance instead of eye identity.

  Cause 2, chaining. Single-linkage merges a~b and b~c even when a and c are
  unrelated. Thousands of spurious links cascade into one giant component.

Both are fixed here.

  Feature: crop to the illuminated field of view, grayscale, resize, zero-mean
  and unit-norm. The dot product between two rows is then Pearson correlation.
  Cropping removes the shared black canvas, which is what inflated the raw
  distances. Calibration in s01a compared this against whole-frame, high-pass
  and gradient variants: cropping recovered all five independently confirmed
  duplicate pairs with ZERO false positives above the weakest of them, while
  high-pass admitted 1225 and gradient 391.

  Linkage: COMPLETE, which requires every pair inside a cluster to exceed the
  threshold. Chaining is impossible by construction.

Threshold is swept, not assumed, and the sweep is written out.

Outputs
-------
outputs/s01_unique_eyes.json   counts, sweep, diagnostics, chaining check
splits/eye_labels.json         {class: {filename: eye_id}}  <-- load-bearing

Read-only with respect to the image data.
"""
import os, sys, json, time
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from common import C, corr_features, list_images

np.random.seed(C.SEED)
np.seterr(all="ignore")

KNOWN_RP_PAIRS = [("5", "7"), ("4", "6"), ("1", "3"), ("17", "18"), ("113", "114")]
RP_FMT = "Retinitis Pigmentosa{}.jpg"


def cluster(S, thr):
    """
    Complete-linkage on correlation distance. Every pair within a returned
    cluster satisfies similarity >= thr, so no chaining.
    """
    D = np.clip(1.0 - S, 0, None)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0                      # enforce exact symmetry for squareform
    Z = linkage(squareform(D, checks=False), method="complete")
    lab = fcluster(Z, t=1.0 - thr, criterion="distance")
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def audit_source_consistency():
    """
    The pristine originals must hold exactly the real images that the legacy
    BalancedDataset carries, or eye labels computed here will not apply
    downstream. BalancedDataset copies preserve filenames, so names suffice.
    """
    bal_rp = os.path.join(C.PROJECT, "BalancedDataset", "Retinitis Pigmentosa")
    if not os.path.isdir(bal_rp):
        return {"checked": False, "reason": "BalancedDataset absent"}
    bal_real = {f for f in os.listdir(bal_rp)
                if not f.startswith(".") and "seed" not in f}
    orig = set(list_images(C.REAL_RP_DIR))
    return {"checked": True, "n_original": len(orig), "n_balanced_real": len(bal_real),
            "identical_filenames": sorted(orig) == sorted(bal_real),
            "only_in_original": sorted(orig - bal_real)[:10],
            "only_in_balanced": sorted(bal_real - orig)[:10]}


def main():
    res = {"seed": C.SEED, "corr_size": C.CORR_SZ, "threshold": C.CORR_THRESHOLD,
           "linkage": "complete", "feature": "fov_crop_gray_zeromean_unitnorm",
           "sweep": {}}

    res["source_consistency"] = audit_source_consistency()
    print("source consistency:", json.dumps(res["source_consistency"]))

    eye_labels = {}
    for name, dirpath in [("RP", C.REAL_RP_DIR), ("Healthy", C.HEALTHY_DIR)]:
        files = list_images(dirpath)
        print(f"\n=== {name}: {len(files)} files ===", flush=True)
        t0 = time.time()
        X = corr_features([os.path.join(dirpath, f) for f in files])
        S = X @ X.T
        print(f"  features + similarity in {time.time()-t0:.0f}s", flush=True)

        iu = np.triu_indices(len(files), 1)
        off = S[iu]

        sweep = {}
        for thr in C.CORR_SWEEP:
            lab = cluster(S, thr)
            sizes = np.bincount(lab)
            sweep[str(thr)] = {"n_unique": int(lab.max() + 1),
                               "largest_cluster": int(sizes.max()),
                               "multi_image_eyes": int((sizes > 1).sum())}
        res["sweep"][name] = sweep

        lab = cluster(S, C.CORR_THRESHOLD)
        sizes = np.bincount(lab)
        eye_labels[name] = {f: int(l) for f, l in zip(files, lab)}

        # Recall check against independently confirmed duplicates.
        if name == "RP":
            idx = {f: i for i, f in enumerate(files)}
            recovered = []
            for a, b in KNOWN_RP_PAIRS:
                ia, ib = idx[RP_FMT.format(a)], idx[RP_FMT.format(b)]
                recovered.append({"pair": f"RP{a}~RP{b}",
                                  "similarity": float(S[ia, ib]),
                                  "same_eye": bool(lab[ia] == lab[ib])})
            res["known_pair_recall"] = recovered
            miss = [r["pair"] for r in recovered if not r["same_eye"]]
            print("  known duplicate pairs recovered: "
                  f"{len(recovered)-len(miss)}/{len(recovered)}"
                  + (f"  MISSED {miss}" if miss else ""))

        print(f"  correlation  median {np.median(off):.4f}  p99 {np.percentile(off,99):.4f}"
              f"  p99.9 {np.percentile(off,99.9):.4f}  max {off.max():.4f}")
        print(f"  pairs >= {C.CORR_THRESHOLD}: {int((off >= C.CORR_THRESHOLD).sum())}")
        print(f"  UNIQUE EYES: {lab.max()+1} of {len(files)} files")
        print(f"  largest cluster {sizes.max()}, singletons {(sizes==1).sum()}, "
              f"multi-image eyes {(sizes>1).sum()}")
        print("  sweep: " + "  ".join(
            f"{t}->{sweep[str(t)]['n_unique']}(max{sweep[str(t)]['largest_cluster']})"
            for t in C.CORR_SWEEP))

        res[name] = {"n_files": len(files), "n_unique": int(lab.max() + 1),
                     "singletons": int((sizes == 1).sum()),
                     "multi_image_eyes": int((sizes > 1).sum()),
                     "largest_cluster": int(sizes.max()),
                     "median_corr": float(np.median(off)),
                     "p999_corr": float(np.percentile(off, 99.9)),
                     "max_corr": float(off.max()),
                     "cluster_size_hist": np.bincount(sizes).tolist()}
        del X, S

    # With complete-linkage a huge component cannot arise from chaining, so a
    # large one would mean the threshold itself is too permissive.
    warn = [n for n in ("RP", "Healthy") if res[n]["largest_cluster"] > 8]
    res["threshold_warning"] = warn
    if warn:
        print(f"\n!! WARNING: largest cluster > 8 in {warn}; threshold may be "
              f"too permissive. Inspect the sweep.")

    with open(os.path.join(C.SPLITS, "eye_labels.json"), "w") as fh:
        json.dump(eye_labels, fh, indent=1)
    with open(os.path.join(C.OUT, "s01_unique_eyes.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {C.SPLITS}/eye_labels.json")
    print(f"wrote {C.OUT}/s01_unique_eyes.json")


if __name__ == "__main__":
    main()
