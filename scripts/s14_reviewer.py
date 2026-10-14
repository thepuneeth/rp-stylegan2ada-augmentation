"""s14 -- camera-ready reviewer analyses. No retraining.

Both BIBM UGHS reviewers converged on one defect. The prespecified endpoint is
AUC; AUC is saturated (0.9843-0.9930 across the four arms) while sensitivity
spans 0.521-0.921. So every prespecified contrast is small by construction, and
every null is uninformative about effect size. Reviewer 1: "the paper notices
the metric is the reason the contrasts are small, then keeps it as the deciding
test." Reviewer 2: "the prespecified AUC contrasts have essentially no power --
the paper acknowledges this but still leans on them."

This does NOT rewrite the prespecification; that is the paper's credibility.
It adds, labelled secondary and camera-ready:

  A  the same paired contrasts on endpoints that are not saturated --
     sensitivity at the fixed 0.5 threshold, and partial AUC over the
     specificity >= 0.95 region.
  B  P1 at FIXED doses. The submitted P1 takes each arm's best dose on the very
     holdout it is scored on; that is a selection the 28 RP eyes cannot pay for.
     Dose 1.0 is the matched-balance point (both arms reach exact class balance);
     dose 10.0 is the grid maximum. Best-dose is retained, labelled optimistic.
  C  Holm across the declared P2 family (dose x backbone), so the one
     significant cell is judged against the grid it was found in.
  D  a THIRD backbone (toycnn, already swept, never reported) so the
     two-backbone disagreement is adjudicated by a third vote instead of by
     choosing which answer to believe.
  E  Brier score per condition. The submission reports no calibration.
  F  headline contrasts recomputed under a STRICTER eye-clustering threshold,
     so residual same-eye leakage is bounded instead of assumed absent.
  G  exact-McNemar granularity on the 28 RP eyes: the smallest sensitivity
     difference that CAN reach significance at this n, whatever the truth is.

Every number comes from the per-eye probabilities s05 already stored.

    ../venv/bin/python scripts/s14_reviewer.py            # A-E, G
    ../venv/bin/python scripts/s14_reviewer.py --alt-eyes # adds F (reclusters)
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import binom
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from stats_boot import bca_ci  # noqa: E402

BACKBONES = {"strong": "s05_classifier_main5.json",      # resnet18, ImageNet init
             "lowparam": "s05_classifier_micro5.json",   # microresnet, ~77k params
             "third": "s05_classifier_toy5.json"}        # toycnn, swept not reported

N_PERM = 10000
THR = 0.5          # the operating threshold s05 itself used for sens/spec


# ------------------------------------------------------------------ endpoints
def m_auc(y, p):
    return roc_auc_score(y, p)


def m_sens(y, p):
    return float((p[y == 1] >= THR).mean())


def m_pauc(y, p):
    """McClish-standardised partial AUC over specificity >= 0.95."""
    return roc_auc_score(y, p, max_fpr=0.05)


METRICS = {"auc": m_auc, "sens": m_sens, "pauc": m_pauc}


def brier(y, p):
    return float(np.mean((p - y) ** 2))


# ------------------------------------------------------------- eye machinery
def eye_index(eye_labels):
    """Per-file eye key and label, in the exact order s05 wrote `probs`."""
    with open(os.path.join(C.SPLITS, "holdout.json")) as fh:
        hold = json.load(fh)
    items = ([(f, "Healthy", 0) for f in sorted(hold["Healthy"])]
             + [(f, "RP", 1) for f in sorted(hold["RP"])])
    keys = np.array([f"{cls}:{eye_labels[cls][f]}" for f, cls, _ in items])
    y = np.array([lab for _, _, lab in items])
    return keys, y


def per_eye(probs, keys, y):
    order = sorted(set(keys))
    p = np.empty(len(order))
    lab = np.empty(len(order), dtype=int)
    for i, k in enumerate(order):
        m = keys == k
        u = np.unique(y[m])
        assert u.size == 1, f"eye {k} carries both labels"
        p[i], lab[i] = probs[m].mean(), u[0]
    return p, lab


def load_scores(path, keys, y_file):
    """(arm, dose) -> per-eye probability, seeds pooled as s12 pools them."""
    d = json.load(open(path))
    pools = {}
    for r in d["runs"]:
        pools.setdefault((r["arm"], r["dose"]), []).append(np.asarray(r["probs"]))
    scores, y_eye = {}, None
    for k, v in pools.items():
        assert all(len(p) == len(keys) for p in v), f"{path}: probs length mismatch"
        scores[k], y_eye = per_eye(np.mean(v, axis=0), keys, y_file)
    return scores, y_eye, d.get("backbone")


# ------------------------------------------------------------------ inference
def contrast(pa, pb, y, metric, n_boot=C.BOOTSTRAP_N, n_perm=N_PERM, seed=C.SEED):
    """metric(a) - metric(b) on paired per-eye scores: BCa CI + paired permutation.

    The permutation sign-flips ARM LABELS WITHIN AN EYE, which is the
    exchangeability the design supports, and recomputes the metric difference --
    identical to s12 except that the metric is an argument rather than AUC.
    """
    f = METRICS[metric]
    clusters = {i: np.array([i]) for i in range(len(y))}

    def stat(idx):
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            return np.nan
        return f(yy, pa[idx]) - f(yy, pb[idx])

    theta, (lo, hi), diag = bca_ci(np.arange(len(y)), clusters, stat,
                                   n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed)
    obs = stat(np.arange(len(y)))
    null = np.empty(n_perm)
    for t in range(n_perm):
        swap = rng.random(len(y)) < 0.5
        a2 = np.where(swap, pb, pa)
        b2 = np.where(swap, pa, pb)
        null[t] = f(y, a2) - f(y, b2)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"metric": metric, "delta": float(theta), "ci95": [lo, hi],
            "p_perm": float(p), "a": diag.get("a"), "z0": diag.get("z0")}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def mcnemar_granularity(n_pos=28):
    """Smallest one-directional discordance that can reach p<0.05, exact binomial.

    Sensitivity on this holdout is a paired proportion over n_pos RP eyes. If b
    eyes disagree between two arms and EVERY ONE favours the same arm -- the most
    favourable configuration that exists -- the exact two-sided McNemar p is
    2 * 0.5^b. So there is a floor on what any sensitivity contrast can detect
    here, independent of the truth.
    """
    for b in range(1, n_pos + 1):
        p = 2 * binom.sf(b - 1, b, 0.5)
        if p < 0.05:
            return {"min_discordant_eyes": b, "exact_p": float(p),
                    "min_detectable_delta_sens": b / n_pos, "n_pos_eyes": n_pos}
    return {"min_discordant_eyes": None, "n_pos_eyes": n_pos}


# ----------------------------------------------------------------------- main
def run(scores_by_bb, y_eye, tag=""):
    out = {}

    # ---- E: per-condition endpoint table, including calibration
    table = {}
    for bb, (scores, _) in scores_by_bb.items():
        for (arm, dose), p in sorted(scores.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            table[f"{bb}|{arm}|{dose}"] = {
                "auc": m_auc(y_eye, p), "sens": m_sens(y_eye, p),
                "spec": float((p[y_eye == 0] < THR).mean()),
                "pauc_spec95": m_pauc(y_eye, p), "brier": brier(y_eye, p)}
    out["conditions"] = table

    # ---- A+B: P1, fixed doses and best dose, on three endpoints
    p1 = {}
    for bb, (scores, _) in scores_by_bb.items():
        auc_of = {k: m_auc(y_eye, v) for k, v in scores.items()}
        cands = [k for k in scores if k[0] == "synth_clean"]
        best = max(cands, key=lambda k: auc_of[k])
        variants = {"fixed_dose1.0": ("synth_clean", 1.0),
                    "fixed_dose10.0": ("synth_clean", 10.0),
                    "best_dose_OPTIMISTIC": best}
        for vname, ka in variants.items():
            kb = ("oversample", None)
            if ka not in scores or kb not in scores:
                continue
            for metric in METRICS:
                r = contrast(scores[ka], scores[kb], y_eye, metric)
                p1[f"{bb}|{vname}|{metric}"] = r | {"arm_a": list(ka), "arm_b": list(kb)}
    out["P1"] = p1

    # ---- C+D: P2 across the full declared family, then Holm
    p2 = {}
    for bb, (scores, _) in scores_by_bb.items():
        for dose in C.DOSE_LEVELS:
            ka, kb = ("synth_leak", dose), ("synth_clean", dose)
            if ka not in scores or kb not in scores:
                continue
            for metric in METRICS:
                r = contrast(scores[ka], scores[kb], y_eye, metric)
                p2[f"{bb}|dose{dose}|{metric}"] = r | {"arm_a": list(ka), "arm_b": list(kb)}
    # The declared P2 family is the paper's two backbones x six doses. The third
    # backbone is reported as a robustness check with its own family, so adding
    # it can neither rescue nor sink a cell of the declared one.
    for metric in METRICS:
        fam = [k for k in p2 if k.endswith("|" + metric) and not k.startswith("third|")]
        own = [k for k in p2 if k.endswith("|" + metric) and k.startswith("third|")]
        for k, a in zip(own, holm([p2[k]["p_perm"] for k in own]) if own else []):
            p2[k]["p_holm"] = float(a)
            p2[k]["holm_family_size"] = len(own)
        if not fam:
            continue
        adj = holm([p2[k]["p_perm"] for k in fam])
        for k, a in zip(fam, adj):
            p2[k]["p_holm"] = float(a)
            p2[k]["holm_family_size"] = len(fam)
    out["P2"] = p2
    return out


def alt_eye_labels(thr):
    """Recluster both classes at a stricter similarity threshold (item F)."""
    from common import corr_features, list_images
    from s01_unique_eyes import cluster
    labels = {}
    for name, dirpath in [("RP", C.REAL_RP_DIR), ("Healthy", C.HEALTHY_DIR)]:
        files = list_images(dirpath)
        X = corr_features([os.path.join(dirpath, f) for f in files])
        lab = cluster(X @ X.T, thr)
        labels[name] = {f: int(l) for f, l in zip(files, lab)}
        print(f"  {name}: {len(files)} files -> {lab.max()+1} eyes at thr={thr}")
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt-eyes", type=float, default=None,
                    help="also rerun headline contrasts at this stricter "
                         "eye-clustering threshold, e.g. 0.985")
    a = ap.parse_args()

    with open(os.path.join(C.SPLITS, "eye_labels.json")) as fh:
        eye_labels = json.load(fh)
    keys, y_file = eye_index(eye_labels)

    scores_by_bb = {}
    for bb, fname in BACKBONES.items():
        path = os.path.join(C.OUT, fname)
        if not os.path.exists(path):
            print(f"SKIP {bb}: {fname} absent")
            continue
        sc, y_eye, name = load_scores(path, keys, y_file)
        scores_by_bb[bb] = (sc, name)
        print(f"{bb:9s} {name:12s} {len(sc)} arm-dose cells")
    _, y_eye = per_eye(np.zeros(len(keys)), keys, y_file)
    print(f"{len(y_eye)} eyes ({int(y_eye.sum())} RP) from {len(keys)} files "
          f"at eye threshold {C.CORR_THRESHOLD}")

    res = {"eval_unit": "eye", "n_eyes": int(len(y_eye)),
           "n_pos_eyes": int(y_eye.sum()), "threshold": THR,
           "n_boot": C.BOOTSTRAP_N, "n_perm": N_PERM, "seed": C.SEED,
           "eye_corr_threshold": C.CORR_THRESHOLD,
           "mcnemar_floor": mcnemar_granularity(int(y_eye.sum()))}
    res["primary"] = run(scores_by_bb, y_eye)

    if a.alt_eyes:
        print(f"\nreclustering eyes at thr={a.alt_eyes} ...")
        alt = alt_eye_labels(a.alt_eyes)
        akeys, ay_file = eye_index(alt)
        alt_scores = {}
        for bb, fname in BACKBONES.items():
            path = os.path.join(C.OUT, fname)
            if os.path.exists(path):
                sc, _, name = load_scores(path, akeys, ay_file)
                alt_scores[bb] = (sc, name)
        _, ay_eye = per_eye(np.zeros(len(akeys)), akeys, ay_file)
        print(f"  holdout collapses to {len(ay_eye)} eyes "
              f"({int(ay_eye.sum())} RP)")
        res["alt_eye_threshold"] = a.alt_eyes
        res["alt_n_eyes"] = int(len(ay_eye))
        res["alt_n_pos_eyes"] = int(ay_eye.sum())
        res["alt_clustering"] = run(alt_scores, ay_eye)

    dest = os.path.join(C.OUT, "s14_reviewer.json")
    json.dump(res, open(dest, "w"), indent=1)
    print(f"\nwrote {dest}")
    report(res)


def report(res):
    f = res["mcnemar_floor"]
    print(f"\n--- G  sensitivity floor at n={f['n_pos_eyes']} RP eyes")
    print(f"    needs >= {f['min_discordant_eyes']} eyes disagreeing ALL ONE WAY "
          f"(exact p={f['exact_p']:.4f})")
    print(f"    -> no sensitivity difference below "
          f"{f['min_detectable_delta_sens']:.3f} can reach p<0.05 here")

    for block, label in [("primary", f"eye thr {res['eye_corr_threshold']}"),
                         ("alt_clustering", f"eye thr {res.get('alt_eye_threshold')}")]:
        if block not in res:
            continue
        print(f"\n================ {block}  ({label}) ================")
        for name in ("P1", "P2"):
            print(f"\n--- {name}")
            print(f"{'cell':44}{'delta':>9}{'95% CI':>22}{'p':>8}{'p_holm':>9}")
            for k, r in res[block][name].items():
                ci = f"({r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f})"
                ph = f"{r['p_holm']:.4f}" if "p_holm" in r else ""
                star = " *" if r.get("p_holm", r["p_perm"]) < 0.05 else ""
                print(f"{k:44}{r['delta']:>+9.4f}{ci:>22}"
                      f"{r['p_perm']:>8.4f}{ph:>9}{star}")


if __name__ == "__main__":
    main()
