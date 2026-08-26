"""
s07 - Can this study detect anything? Power for the primary contrast.

Why this runs BEFORE any GPU is rented
--------------------------------------
P1 -- synth_clean vs oversample on the generator-blind holdout -- is the only
confirmatory test in the study. The holdout is 28 RP eyes and 197 Healthy eyes,
fixed by the cohort and not enlargeable without new data. If a difference of the
size a generator plausibly produces cannot be distinguished from zero at that n,
then renting a GPU buys a result that was uninterpretable before it was measured.

That is a cheaper thing to learn now than after the training run.

This module answers two questions, and they are NOT the same question.

  DETECTION   what AUC difference would P1 detect, at 80% power?
              If the minimum detectable effect is enormous, a null result means
              nothing and the study cannot claim the generator failed.

  EQUIVALENCE if the true difference IS zero, how tight an interval can this
              design put around it?  This is the more likely outcome and the
              more useful one: "the generator adds no more than X AUC beyond
              rebalancing, on this cohort" is a real finding. It requires a
              prespecified margin, which this module exists to inform.

Why the pairing matters more than the sample size
-------------------------------------------------
Both arms are scored on the SAME holdout eyes by classifiers differing only in
training data. Their per-eye scores are therefore strongly correlated, and the
comparison is paired. Var(A - B) = Var(A) + Var(B) - 2*rho*sd(A)*sd(B), so at
rho = 0.9 the variance of the DIFFERENCE is a fifth of what an unpaired
comparison of the same two arms would carry. Power here is governed by rho far
more than by n, which is why quoting a textbook two-sample AUC power formula
would badly understate what this design can do.

rho is unknown before the experiment. It is swept, not assumed.

Simulation model
----------------
Binormal. Negatives ~ N(0,1); positives ~ N(mu,1) with mu = sqrt(2)*Phi^-1(AUC),
the exact binormal relation. The two arms' latent scores are drawn with
correlation rho, giving one score per eye per arm.

Inference matches the analysis this study will actually run -- the eye-level
bootstrap already fixed in config (EVAL_UNIT="eye", BOOTSTRAP_N) -- rather than
a closed-form test that the real analysis will not use. Resampling is over EYES,
stratified by class, because eyes are the independent unit.

Outputs outputs/s07_power.json
"""
import os, sys, json, argparse
import numpy as np
from scipy.stats import norm

from common import C

np.seterr(all="ignore")

N_POS = 28        # holdout RP eyes
N_NEG = 197       # holdout Healthy eyes
BASE_AUC = 0.90   # oversample arm; colour_lr already reaches 0.9501 on this
                  # cohort, so a high baseline is the realistic regime and the
                  # ceiling squeezes any increment the generator could add.


def auc(pos, neg):
    """Mann-Whitney AUC. Ties count half."""
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(np.float64) + 1
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.bincount(inv, weights=r)
        r = (sums / cnt)[inv]
    n1 = len(pos)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * len(neg))


def simulate_scores(auc_a, auc_b, rho, rng, n_pos=N_POS, n_neg=N_NEG):
    """
    One trial. Returns (pos_a, neg_a, pos_b, neg_b).

    The SAME eye carries both arms' scores, correlated at rho -- that is what
    makes the comparison paired.
    """
    mu_a = np.sqrt(2.0) * norm.ppf(auc_a)
    mu_b = np.sqrt(2.0) * norm.ppf(auc_b)
    L = np.linalg.cholesky([[1.0, rho], [rho, 1.0]])

    z = rng.normal(size=(n_pos + n_neg, 2)) @ L.T
    a, b = z[:, 0].copy(), z[:, 1].copy()
    a[:n_pos] += mu_a
    b[:n_pos] += mu_b
    return a[:n_pos], a[n_pos:], b[:n_pos], b[n_pos:]


def boot_ci(pa, na, pb, nb, n_boot, rng):
    """
    Percentile CI for AUC(A) - AUC(B), resampling EYES stratified by class.
    Both arms are resampled with the SAME indices, which preserves the pairing.
    """
    d = np.empty(n_boot)
    npos, nneg = len(pa), len(na)
    for i in range(n_boot):
        ip = rng.integers(0, npos, npos)
        ineg = rng.integers(0, nneg, nneg)
        d[i] = auc(pa[ip], na[ineg]) - auc(pb[ip], nb[ineg])
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def power_at(delta, rho, n_trials, n_boot, seed, base=BASE_AUC):
    """
    Fraction of trials whose 95% CI for the paired AUC difference excludes 0.

    delta = 0 gives the FALSE POSITIVE RATE, which must land near 0.05 or the
    procedure is miscalibrated and every other number here is worthless.
    """
    rng = np.random.default_rng(seed)
    hits, widths, ests = 0, [], []
    for _ in range(n_trials):
        pa, na, pb, nb = simulate_scores(min(base + delta, 0.999), base, rho, rng)
        lo, hi = boot_ci(pa, na, pb, nb, n_boot, rng)
        hits += (lo > 0) or (hi < 0)
        widths.append(hi - lo)
        ests.append(auc(pa, na) - auc(pb, nb))
    return {"delta": float(delta), "rho": float(rho),
            "power": hits / n_trials,
            "mean_ci_halfwidth": float(np.mean(widths) / 2.0),
            "mean_estimate": float(np.mean(ests)),
            "n_trials": int(n_trials)}


def mde(rows, target=0.80):
    """
    Minimum detectable effect: smallest simulated delta reaching `target`
    power. Interpolates between the bracketing grid points. None if the grid
    never gets there -- reported as such rather than extrapolated.
    """
    rows = sorted(rows, key=lambda r: r["delta"])
    for prev, cur in zip(rows, rows[1:]):
        if prev["power"] < target <= cur["power"]:
            f = (target - prev["power"]) / (cur["power"] - prev["power"])
            return float(prev["delta"] + f * (cur["delta"] - prev["delta"]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--boot", type=int, default=400,
                    help=f"inner bootstrap; the real analysis uses {C.BOOTSTRAP_N}")
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10])
    a = ap.parse_args()

    res = {"n_pos_eyes": N_POS, "n_neg_eyes": N_NEG, "base_auc": BASE_AUC,
           "n_trials": a.trials, "n_boot": a.boot, "seed": C.SEED,
           "grids": {}, "mde_80": {}, "null_ci_halfwidth": {}}

    for rho in a.rhos:
        rows = [power_at(d, rho, a.trials, a.boot, C.SEED + i)
                for i, d in enumerate(a.deltas)]
        res["grids"][str(rho)] = rows
        res["mde_80"][str(rho)] = mde(rows)
        res["null_ci_halfwidth"][str(rho)] = rows[0]["mean_ci_halfwidth"]

        print(f"\nrho = {rho}   (per-eye score correlation between the two arms)")
        for r in rows:
            tag = "  <- false positive rate" if r["delta"] == 0 else ""
            print(f"  delta AUC {r['delta']:.3f}   power {r['power']:.2f}"
                  f"   mean CI halfwidth {r['mean_ci_halfwidth']:.3f}{tag}")
        m = res["mde_80"][str(rho)]
        print(f"  MDE at 80% power: "
              + (f"{m:.3f} AUC" if m else f"> {max(a.deltas):.3f} AUC (off grid)"))
        print(f"  under a TRUE NULL the CI is +/- "
              f"{rows[0]['mean_ci_halfwidth']:.3f} -> the tightest equivalence "
              f"margin this design can support")

    print("\nRead this as: an equivalence margin BELOW the halfwidth cannot be "
          "concluded even when the truth is exactly zero.")
    p = os.path.join(C.OUT, "s07_power.json")
    with open(p, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {p}")


def _selftest():
    """Invariants that fail loudly if the machinery is wrong."""
    rng = np.random.default_rng(0)

    # AUC against a known case: perfectly separated, and perfectly tied.
    assert auc(np.array([3., 4., 5.]), np.array([0., 1., 2.])) == 1.0
    assert auc(np.array([0., 0.]), np.array([0., 0.])) == 0.5

    # The binormal draw reproduces the AUC it was asked for.
    pa, na, _, _ = simulate_scores(0.90, 0.90, 0.9, rng, n_pos=4000, n_neg=4000)
    assert abs(auc(pa, na) - 0.90) < 0.02, auc(pa, na)

    # Pairing is real: at high rho the two arms' per-eye scores co-vary.
    pa, na, pb, nb = simulate_scores(0.9, 0.9, 0.95, rng, n_pos=2000, n_neg=2000)
    assert np.corrcoef(np.r_[pa, na], np.r_[pb, nb])[0, 1] > 0.9

    # Calibration: at delta = 0 the CI must exclude 0 about 5% of the time.
    # A procedure that fails here makes every power number meaningless.
    fp = power_at(0.0, 0.7, 120, 200, 1)["power"]
    assert fp < 0.15, f"false positive rate {fp} -- procedure miscalibrated"

    # Power must rise with the effect, and pairing must help.
    lo = power_at(0.02, 0.7, 120, 200, 2)["power"]
    hi = power_at(0.10, 0.7, 120, 200, 2)["power"]
    assert hi > lo, (lo, hi)
    weak = power_at(0.05, 0.3, 150, 200, 3)["power"]
    strong = power_at(0.05, 0.95, 150, 200, 3)["power"]
    assert strong > weak, (weak, strong)

    print(f"false positive rate at delta=0: {fp:.3f} (target ~0.05)")
    print(f"power at delta=0.02 {lo:.2f}  at delta=0.10 {hi:.2f}")
    print(f"power at delta=0.05: rho=0.3 {weak:.2f}  rho=0.95 {strong:.2f}")
    print("s07 selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
