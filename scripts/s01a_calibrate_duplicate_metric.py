"""
s01a - Calibrate a duplicate-detection metric against known ground truth.

Why this exists
---------------
The first attempt clustered by mean absolute difference on 128x128 RGB with
single-linkage at MAE < 12. It failed: 1024 healthy images collapsed into 10
"eyes", one component holding 842 images. Two independent causes.

1. NO SEPARATION. The pairwise MAE histogram rises smoothly from 3.9 with no
   bimodality, so no threshold splits duplicates from distinct eyes. Every
   fundus from one camera looks alike at low resolution: same circular field of
   view, same dark surround, same bright optic disc. That shared template, not
   eye identity, dominates the distance.

2. CHAINING. Single-linkage merges a~b and b~c into one component even when a
   and c are unrelated. With thousands of spurious links this cascades.

Fix for (1): isolate the retinal vasculature, which is the actual per-eye
fingerprint. Crop to the field of view, then high-pass filter to strip global
illumination, then correlate.

Fix for (2): complete-linkage, which requires EVERY pair inside a cluster to be
above threshold. Chaining becomes impossible by construction.

Ground truth used for calibration comes from the earlier duplicate audit, which
confirmed these five RP pairs by cosine similarity >= 0.9975.

Writes outputs/s01a_metric_calibration.json. Selects nothing on its own; the
chosen metric and threshold are promoted into config by hand after review.
"""
import os, sys, json
import numpy as np
from PIL import Image, ImageFilter

from common import C, fov_crop

np.seterr(all="ignore")

KNOWN_RP_PAIRS = [("5", "7"), ("4", "6"), ("1", "3"), ("17", "18"), ("113", "114")]
RP_FMT = "Retinitis Pigmentosa{}.jpg"


def features(dirpath, files, mode, sz=256, blur=8):
    """
    Unit-norm, zero-mean feature vectors. Dot product between two rows is then
    the Pearson correlation, so a threshold on the dot product is a threshold on
    correlation.

    modes
      plain     grayscale, whole frame
      crop      grayscale, field-of-view cropped
      crophp    field-of-view cropped, then high-pass  <- isolates vasculature
      cropgrad  field-of-view cropped, then gradient magnitude
    """
    X = np.empty((len(files), sz * sz), dtype=np.float64)
    for i, f in enumerate(files):
        with Image.open(os.path.join(dirpath, f)) as im:
            g = im.convert("L")
            if mode != "plain":
                g = fov_crop(g)
            g = g.resize((sz, sz), Image.BILINEAR)
            a = np.asarray(g, dtype=np.float64)
            if mode == "crophp":
                a = a - np.asarray(g.filter(ImageFilter.GaussianBlur(blur)),
                                   dtype=np.float64)
            elif mode == "cropgrad":
                gy, gx = np.gradient(a)
                a = np.hypot(gx, gy)
        a = a - a.mean()
        n = np.linalg.norm(a)
        X[i] = (a / (n if n > 1e-6 else 1.0)).ravel()
    return X


def largest_gap(sorted_desc, lo=0.90, hi=0.9999):
    """
    Biggest drop between consecutive similarities inside a plausible range.
    A clean duplicate mode shows up as a visible cliff; its absence is itself
    a finding and is reported rather than papered over.
    """
    s = sorted_desc[(sorted_desc >= lo) & (sorted_desc <= hi)]
    if len(s) < 3:
        return None
    d = s[:-1] - s[1:]
    k = int(np.argmax(d))
    return {"gap": float(d[k]), "above": float(s[k]), "below": float(s[k + 1]),
            "n_above": int(k + 1)}


def main():
    files = sorted(f for f in os.listdir(C.REAL_RP_DIR) if not f.startswith("."))
    idx = {f: i for i, f in enumerate(files)}
    known = [(idx[RP_FMT.format(a)], idx[RP_FMT.format(b)]) for a, b in KNOWN_RP_PAIRS]

    res = {"known_pairs": [RP_FMT.format(a) + " ~ " + RP_FMT.format(b)
                           for a, b in KNOWN_RP_PAIRS], "modes": {}}

    for mode in ["plain", "crop", "crophp", "cropgrad"]:
        X = features(C.REAL_RP_DIR, files, mode)
        M = X @ X.T
        iu = np.triu_indices(len(files), 1)
        off = M[iu]
        ks = [float(M[i, j]) for i, j in known]
        s = np.sort(off)[::-1]

        # Separation: how much of the bulk sneaks above the weakest true pair?
        # A perfect metric admits exactly the true duplicates and nothing else.
        n_above_weakest = int((off >= min(ks)).sum())

        res["modes"][mode] = {
            "known_pair_similarity": ks,
            "weakest_known": min(ks),
            "bulk_median": float(np.median(off)),
            "bulk_p99": float(np.percentile(off, 99)),
            "bulk_p999": float(np.percentile(off, 99.9)),
            "pairs_above_weakest_known": n_above_weakest,
            "false_positives_at_weakest_known": n_above_weakest - len(ks),
            "largest_gap": largest_gap(s),
            "top20": [float(v) for v in s[:20]],
        }
        g = res["modes"][mode]["largest_gap"]
        print(f"{mode:9s} known {' '.join(f'{v:.4f}' for v in ks)}")
        print(f"          median {np.median(off):.4f}  p99.9 {np.percentile(off,99.9):.4f}"
              f"  extra pairs above weakest known: {n_above_weakest - len(ks)}")
        if g:
            print(f"          largest gap {g['gap']:.4f} at {g['above']:.4f}"
                  f" -> {g['below']:.4f}, {g['n_above']} pairs above")

    best = min(res["modes"], key=lambda m: res["modes"][m]["false_positives_at_weakest_known"])
    res["recommended_mode"] = best
    print(f"\nfewest false positives at full recall of known pairs: {best}")

    with open(os.path.join(C.OUT, "s01a_metric_calibration.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {C.OUT}/s01a_metric_calibration.json")


if __name__ == "__main__":
    main()
