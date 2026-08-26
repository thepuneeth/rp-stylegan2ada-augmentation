"""
s02 - Build the eye-disjoint, generator-blind split.

The design in one paragraph
---------------------------
The 139 real RP images (134 unique eyes, per s01) are partitioned EYE-DISJOINT
into a HOLDOUT that no generator will ever see, and a DEV pool used both to
train the generator and to cross-validate classifiers. Every arm -- imbalanced,
class-weighted, oversampled, focal, synthetic -- is scored on the same holdout.
Because the generator is fit only on DEV, a synthetic advantage measured on
HOLDOUT cannot be explained by the generator having memorised the test eyes.
That is the whole point: it converts "synthetic helps" into a claim that
survives the leakage objection.

Why eye-disjoint rather than image-disjoint
-------------------------------------------
Five RP eyes contribute two images each. Splitting at the image level would put
the same eye on both sides, and the holdout would silently stop being blind.

Why a fixed holdout rather than per-fold generation
---------------------------------------------------
Removing generator-mediated leakage by retraining the generator once per fold
costs five GAN runs. A single fixed generator-blind holdout costs one, and
answers the same question. Model selection happens inside DEV via repeated
eye-disjoint cross-validation, so the holdout is touched once per arm at the
end and never used to choose anything.

Outputs
-------
splits/holdout.json      filenames + eye ids, both classes
splits/dev.json          filenames + eye ids, both classes
splits/dev_folds.json    repeated eye-disjoint CV folds inside DEV
outputs/s02_splits.json  audit record

Assertions abort the run rather than emit a subtly broken split.
"""
import os, sys, json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from common import C

rng = np.random.default_rng(C.SEED)


def load_eye_labels():
    with open(os.path.join(C.SPLITS, "eye_labels.json")) as fh:
        return json.load(fh)


def carve_holdout(labels, n_target, rng):
    """
    Move whole eyes into the holdout until the image target is reached.

    Eyes are shuffled and taken in order, so selection is unbiased with respect
    to cluster size. The final count lands near, not exactly on, the target
    because eyes move as units -- that is intentional; splitting an eye to hit a
    round number would defeat the purpose.
    """
    by_eye = {}
    for fname, eye in labels.items():
        by_eye.setdefault(eye, []).append(fname)
    eyes = sorted(by_eye)
    rng.shuffle(eyes)

    hold, n = [], 0
    for e in eyes:
        if n >= n_target:
            break
        hold.append(e)
        n += len(by_eye[e])
    hold = set(hold)

    hold_files = sorted(f for e in hold for f in by_eye[e])
    dev_files = sorted(f for e in by_eye if e not in hold for f in by_eye[e])
    return hold_files, dev_files, hold, set(by_eye) - hold


def main():
    labels = load_eye_labels()
    res = {"seed": C.SEED}

    holdout, dev = {}, {}
    eyes_hold, eyes_dev = {}, {}

    for cls, target in [("RP", C.HOLDOUT_RP_TARGET),
                        ("Healthy", C.HOLDOUT_HEALTHY_TARGET)]:
        hf, df, he, de = carve_holdout(labels[cls], target,
                                       np.random.default_rng(C.SEED))
        holdout[cls] = {f: labels[cls][f] for f in hf}
        dev[cls] = {f: labels[cls][f] for f in df}
        eyes_hold[cls], eyes_dev[cls] = he, de

        # Non-negotiable: an eye may not straddle the boundary.
        assert not (he & de), f"{cls}: eye appears in both holdout and dev"
        assert not (set(hf) & set(df)), f"{cls}: file in both holdout and dev"

        print(f"{cls:8s} holdout {len(hf):4d} files / {len(he):4d} eyes   "
              f"dev {len(df):4d} files / {len(de):4d} eyes")
        res[cls] = {"holdout_files": len(hf), "holdout_eyes": len(he),
                    "dev_files": len(df), "dev_eyes": len(de),
                    "target": target}

    n_h_rp, n_h_he = len(holdout["RP"]), len(holdout["Healthy"])
    n_d_rp, n_d_he = len(dev["RP"]), len(dev["Healthy"])
    prev_cohort = 139 / (139 + 1024)
    res["prevalence"] = {
        "cohort": prev_cohort,
        "holdout": n_h_rp / (n_h_rp + n_h_he),
        "dev": n_d_rp / (n_d_rp + n_d_he),
    }
    print(f"\nprevalence  cohort {prev_cohort:.4f}  "
          f"holdout {res['prevalence']['holdout']:.4f}  "
          f"dev {res['prevalence']['dev']:.4f}")

    # ------------------------------------------------ CV folds inside DEV only
    # Eye ids are namespaced across classes so a healthy eye and an RP eye can
    # never collide into the same group.
    files, y, groups = [], [], []
    off = max(labels["Healthy"].values()) + 1
    for f, e in sorted(dev["Healthy"].items()):
        files.append(("Healthy", f)); y.append(0); groups.append(e)
    for f, e in sorted(dev["RP"].items()):
        files.append(("RP", f)); y.append(1); groups.append(e + off)
    y, groups = np.array(y), np.array(groups)

    folds = []
    for rep in range(C.N_REPEATS):
        sgkf = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True,
                                    random_state=C.SEED + rep)
        for k, (tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
            # An eye in both halves of a fold would leak inside DEV, which would
            # corrupt model selection even though the holdout stays clean.
            assert not (set(groups[tr]) & set(groups[va])), \
                f"rep{rep} fold{k}: eye spans train and val"
            folds.append({"rep": rep, "fold": k,
                          "train": [list(files[i]) for i in tr],
                          "val": [list(files[i]) for i in va],
                          "n_train_rp": int(y[tr].sum()),
                          "n_val_rp": int(y[va].sum())})

    rp_val = [f["n_val_rp"] for f in folds]
    print(f"\n{len(folds)} folds ({C.N_REPEATS} repeats x {C.N_FOLDS})   "
          f"RP per val fold: min {min(rp_val)} max {max(rp_val)} "
          f"mean {np.mean(rp_val):.1f}")
    res["folds"] = {"n": len(folds), "rp_val_min": min(rp_val),
                    "rp_val_max": max(rp_val), "rp_val_mean": float(np.mean(rp_val))}

    # A fold with no positives makes AUC undefined for that fold.
    assert min(rp_val) > 0, "a validation fold contains zero RP images"

    for fname, obj in [("holdout.json", holdout), ("dev.json", dev),
                       ("dev_folds.json", folds)]:
        with open(os.path.join(C.SPLITS, fname), "w") as fh:
            json.dump(obj, fh, indent=1)
        print(f"wrote {C.SPLITS}/{fname}")

    with open(os.path.join(C.OUT, "s02_splits.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {C.OUT}/s02_splits.json")

    print(f"\nGAN training set = DEV RP = {n_d_rp} images / {len(eyes_dev['RP'])} eyes")
    print(f"Generator-blind holdout = {n_h_rp} RP + {n_h_he} healthy")


if __name__ == "__main__":
    main()
