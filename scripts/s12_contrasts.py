"""s12 -- prespecified contrasts on the s05 sweep, with cluster inference.

s05 reports each arm's AUC as a mean over seeds. That summary cannot support a
claim that one arm beats another: the holdout is 235 files over 225 eyes, so
files are not independent, and the seed-to-seed spread is not the sampling
uncertainty that matters. This script does the inference s05 deliberately does
not.

Unit of resampling is the EYE, matching config.EVAL_UNIT. Per arm, the five
seeds are averaged into one probability per eye, so a contrast compares two
score vectors over the same 225 eyes.

Read the output with the sampling geometry in mind. The holdout carries 28 RP
eyes, so eye-level AUC is coarsely quantised: in a check on the dose=5.0
contrast, 30.4% of bootstrap resamples returned a difference of EXACTLY zero
because both arms ranked every resampled eye identically. Intervals therefore
pin to 0.0000 on one side. That is the ceiling effect reaching the inference,
not a defect in the estimator.

The permutation is deliberately conservative. Swapping whole arm assignments
within an eye breaks the strong across-eye correlation the two arms share, so
the null is more variable than the paired difference and p runs high. It will
not manufacture an effect; it can miss a small true one.

Two tests per contrast:
  BCa cluster bootstrap on the AUC difference -- interval on the effect.
  Paired permutation that sign-flips ARM LABELS WITHIN AN EYE and recomputes
  the AUC difference. Sign-flipping per-eye score differences instead would
  test whether mean predicted probability shifts, which is a different claim.
"""
import json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from scripts.stats_boot import bca_ci

RES = os.path.join(C.OUT, "s05_classifier_main5.json")


def holdout_eye_index():
    """Per-file eye id and label, in the exact order s05 wrote `probs`."""
    with open(os.path.join(C.SPLITS, "holdout.json")) as fh:
        hold = json.load(fh)
    with open(os.path.join(C.SPLITS, "eye_labels.json")) as fh:
        eye_labels = json.load(fh)
    items = ([(f, "Healthy", 0) for f in sorted(hold["Healthy"])]
             + [(f, "RP", 1) for f in sorted(hold["RP"])])
    keys = np.array([f"{cls}:{eye_labels[cls][f]}" for f, cls, _ in items])
    y = np.array([lab for _, _, lab in items])
    return keys, y


def per_eye(probs, keys, y):
    """Mean probability per eye, and that eye's label."""
    order = sorted(set(keys))
    p = np.empty(len(order)); lab = np.empty(len(order), dtype=int)
    for i, k in enumerate(order):
        m = keys == k
        u = np.unique(y[m]); assert u.size == 1, f"eye {k} carries both labels"
        p[i], lab[i] = probs[m].mean(), u[0]
    return p, lab


def contrast(pa, pb, y, n_boot=C.BOOTSTRAP_N, n_perm=10000, seed=C.SEED):
    """AUC(a) - AUC(b) on paired per-eye scores, with BCa CI and permutation p."""
    clusters = {i: np.array([i]) for i in range(len(y))}   # eye IS the unit here

    def stat(idx):
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            return np.nan
        return roc_auc_score(yy, pa[idx]) - roc_auc_score(yy, pb[idx])

    theta, (lo, hi), diag = bca_ci(np.arange(len(y)), clusters, stat,
                                   n_boot=n_boot, seed=seed)

    rng = np.random.default_rng(seed)
    obs = stat(np.arange(len(y)))
    null = np.empty(n_perm)
    for t in range(n_perm):
        swap = rng.random(len(y)) < 0.5
        a2 = np.where(swap, pb, pa); b2 = np.where(swap, pa, pb)
        null[t] = roc_auc_score(y, a2) - roc_auc_score(y, b2)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"delta_auc": float(theta), "ci95": [lo, hi], "p_perm": float(p),
            "z0": diag.get("z0"), "a": diag.get("a")}


def main():
    d = json.load(open(RES))
    keys, y_file = holdout_eye_index()
    n_files = len(keys)

    pools = {}
    for r in d["runs"]:
        if r["arm"] == "colour_lr":
            continue
        pools.setdefault((r["arm"], r["dose"]), []).append(np.asarray(r["probs"]))
    scores = {}
    for k, v in pools.items():
        assert all(len(p) == n_files for p in v), "probs length != holdout files"
        pe, y_eye = per_eye(np.mean(v, axis=0), keys, y_file)
        scores[k] = pe
    _, y_eye = per_eye(np.asarray(d["runs"][0]["probs"]), keys, y_file)

    print(f"{len(scores)} arm-dose cells | {len(y_eye)} eyes "
          f"({int(y_eye.sum())} RP) | {n_files} files")

    auc = {k: roc_auc_score(y_eye, v) for k, v in scores.items()}
    best = lambda arm: max((k for k in scores if k[0] == arm), key=lambda k: auc[k])

    tests = []
    # P1, prespecified: the generator arm against the cheap baseline it must beat.
    tests.append(("synth_clean(best) vs oversample", best("synth_clean"), ("oversample", None)))
    # P2: contamination at matched dose -- the generator-mediated leakage probe.
    for dose in C.DOSE_LEVELS:
        if ("synth_leak", dose) in scores and ("synth_clean", dose) in scores:
            tests.append((f"synth_leak vs synth_clean @dose={dose}",
                          ("synth_leak", dose), ("synth_clean", dose)))
    # Every arm against doing nothing.
    for arm in ("oversample", "pca_sampler", "synth_clean", "synth_leak"):
        k = best(arm) if arm != "oversample" else ("oversample", None)
        tests.append((f"{arm}(best) vs real_only", k, ("real_only", None)))

    out = {}
    print(f"\n{'contrast':44}{'dAUC':>9}{'95% CI':>22}{'p':>9}")
    for name, ka, kb in tests:
        if ka not in scores or kb not in scores:
            print(f"{name:44}  SKIP (missing cell)"); continue
        r = contrast(scores[ka], scores[kb], y_eye)
        out[name] = r | {"arm_a": list(ka), "arm_b": list(kb),
                         "auc_a": auc[ka], "auc_b": auc[kb]}
        ci = f"({r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f})"
        star = " *" if r["p_perm"] < 0.05 else ""
        print(f"{name:44}{r['delta_auc']:>+9.4f}{ci:>22}{r['p_perm']:>9.4f}{star}")

    dest = os.path.join(C.OUT, "s12_contrasts.json")
    json.dump({"eval_unit": "eye", "n_eyes": len(y_eye),
               "n_boot": C.BOOTSTRAP_N, "contrasts": out}, open(dest, "w"), indent=1)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
