"""
Cluster bootstrap with BCa intervals, and an exact-style permutation test.

Addresses audit finding F9. The percentile bootstrap already in use is
anti-conservative here: the s07 power simulation measured its type I error at
0.067 to 0.090 against a nominal 0.05, because the estimator is skewed at this
sample size and percentile intervals do not correct for that.

BCa corrects two things the percentile method ignores:
  z0 (bias correction)  -- the median of the bootstrap distribution is not the
                           point estimate, and the offset is measurable.
  a  (acceleration)     -- the variance of the estimator changes with the true
                           value, estimated by jackknife skewness.

Resampling is over EYES, not images (config.EVAL_UNIT = "eye"): two images of
the same eye are not independent, and resampling images would understate the
variance by treating them as if they were.

For a difference between two arms, prefer permutation_test: it needs no
distributional assumption at all, and under the null of exchangeable labels its
p-value is exact up to the number of permutations drawn.

    ../venv/bin/python scripts/stats_boot.py --self-test
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import norm
import config as C


def _resample_clusters(clusters, rng):
    keys = list(clusters)
    drawn = rng.choice(len(keys), size=len(keys), replace=True)
    return np.concatenate([clusters[keys[i]] for i in drawn])


def bca_ci(values, clusters, stat, n_boot=C.BOOTSTRAP_N, alpha=0.05,
           seed=C.SEED):
    """
    BCa interval for `stat` under a cluster bootstrap.

    values   : (n,) array the statistic is computed on
    clusters : dict cluster_key -> integer index array into `values`
    stat     : callable taking an index array, returning a float
    """
    rng = np.random.default_rng(seed)
    keys = list(clusters)
    theta_hat = stat(np.arange(len(values)))

    boots = np.array([stat(_resample_clusters(clusters, rng))
                      for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    if boots.size < 100:
        return theta_hat, (np.nan, np.nan), {"reason": "too few finite boots"}

    # bias correction: where the point estimate sits in the bootstrap dist
    prop = np.mean(boots < theta_hat)
    prop = min(max(prop, 1.0 / len(boots)), 1.0 - 1.0 / len(boots))
    z0 = norm.ppf(prop)

    # acceleration: jackknife over CLUSTERS, matching the resampling unit
    jack = []
    for i in range(len(keys)):
        idx = np.concatenate([clusters[k] for j, k in enumerate(keys) if j != i])
        jack.append(stat(idx))
    jack = np.asarray(jack, dtype=float)
    jack = jack[np.isfinite(jack)]
    d = jack.mean() - jack
    denom = 6.0 * (np.sum(d ** 2) ** 1.5)
    a = float(np.sum(d ** 3) / denom) if denom > 0 else 0.0

    zl, zu = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    def adj(z):
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = np.percentile(boots, [100 * adj(zl), 100 * adj(zu)])
    return float(theta_hat), (float(lo), float(hi)), {"z0": float(z0), "a": a,
                                                      "n_boot": int(boots.size)}


def permutation_test(a_vals, b_vals, n_perm=10000, seed=C.SEED,
                     alternative="two-sided"):
    """
    Paired permutation test on per-eye values from two arms.

    Sign-flips the paired differences, which is the exchangeability the design
    supports: the same holdout eyes are scored under both arms, so arm labels
    are exchangeable within an eye and not across eyes.
    """
    a, b = np.asarray(a_vals, float), np.asarray(b_vals, float)
    assert a.shape == b.shape, "paired test needs equal-length inputs"
    d = a - b
    obs = d.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = (signs * d).mean(axis=1)
    if alternative == "two-sided":
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    else:
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return {"observed_diff": float(obs), "p": float(p), "n_perm": int(n_perm),
            "n_pairs": int(d.size)}


def _self_test():
    rng = np.random.default_rng(0)
    # 40 eyes, 3 images each, clustered signal
    eye_means = rng.normal(0.6, 0.15, size=40)
    vals, clusters, k = [], {}, 0
    for e, m in enumerate(eye_means):
        idx = []
        for _ in range(3):
            vals.append(m + rng.normal(0, 0.02)); idx.append(k); k += 1
        clusters[e] = np.array(idx)
    vals = np.array(vals)
    mean_stat = lambda idx: float(vals[idx].mean())
    theta, (lo, hi), info = bca_ci(vals, clusters, mean_stat, n_boot=2000)
    assert lo < theta < hi, f"point estimate must lie inside its own CI: {lo},{theta},{hi}"
    assert lo < eye_means.mean() < hi, "CI must cover the truth on a clean sample"
    print(f"BCa  theta={theta:.4f} CI=({lo:.4f},{hi:.4f}) z0={info['z0']:+.3f} a={info['a']:+.4f}")

    # null: identical arms must not be called significant
    x = rng.normal(0.7, 0.1, 28)
    p_null = permutation_test(x, x + rng.normal(0, 0.001, 28))["p"]
    # real shift must be detected
    p_shift = permutation_test(x + 0.08, x)["p"]
    assert p_null > 0.05, f"identical arms flagged significant: p={p_null}"
    assert p_shift < 0.01, f"clear shift missed: p={p_shift}"
    print(f"perm  null p={p_null:.3f}   shifted p={p_shift:.4f}")

    # a cluster bootstrap must be WIDER than an image bootstrap that ignores
    # clustering -- if it is not, the clustering is not being respected
    flat = {i: np.array([i]) for i in range(len(vals))}
    _, (flo, fhi), _ = bca_ci(vals, flat, mean_stat, n_boot=2000)
    assert (hi - lo) > (fhi - flo), "cluster CI must exceed the naive image CI"
    print(f"width cluster={hi-lo:.4f} > naive-image={fhi-flo:.4f}  (clustering respected)")
    print("self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    if ap.parse_args().self_test:
        _self_test()
