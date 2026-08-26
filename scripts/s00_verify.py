"""
s00 - Verify every invariant the study depends on.

Run after any change to splits, data or config. Each check is something that,
if silently violated, would produce a publishable-looking but wrong result.
Exit code is non-zero if anything fails.

The invariants
--------------
1. Source integrity      the 139 real RP filenames match between the pristine
                         originals and the legacy BalancedDataset
2. Eye labels complete   every image has an eye id
3. Holdout is disjoint   no eye and no file appears in both holdout and dev
4. Folds are disjoint    no eye spans train and val within any fold
5. Every fold has RP     otherwise AUC is undefined for that fold
6. Prevalence matched    holdout prevalence tracks the cohort
7. GAN data is honest    clean_111 contains NO holdout image; leak_111 contains
                         EVERY holdout image; both are the same size
8. Pretrain is clean     the pretraining corpus contains no holdout image
9. Zips are distinct     clean and leak differ in content, not merely in name
10. Synthetic is separate the legacy pool contains no real image
"""
import os, sys, json, zipfile, hashlib
import numpy as np
from PIL import Image

from common import C

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def img_hashes(paths, square_pad):
    """
    Content hashes for identity comparison across the pipeline.

    square_pad=True must REPLICATE s04's transform exactly -- pad onto a black
    square, then resize to 256 with LANCZOS -- because exact-pixel hashing does
    not survive a different resampling chain. Padding alone is not enough:
    2004 -> 64 and 2004 -> 256 (LANCZOS) -> 64 produce different pixels and
    therefore different digests.

    Two versions of this function were wrong before this one, and both failed
    the same way: every "contains no holdout image" check passed VACUOUSLY,
    because no digest could ever match. A leak-detection check that cannot
    detect a leak is worse than no check, so the transform is spelled out here
    rather than approximated.

    square_pad=False is for files already written by s04 at 256x256, which are
    hashed as stored. PNG is lossless, so a padded original and its saved copy
    hash identically.
    """
    out = {}
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            if square_pad:
                w, h = im.size
                s = max(w, h)
                canvas = Image.new("RGB", (s, s), (0, 0, 0))
                canvas.paste(im, ((s - w) // 2, (s - h) // 2))
                im = canvas.resize((256, 256), Image.LANCZOS)
            a = np.asarray(im, dtype=np.uint8)
        out[hashlib.md5(a.tobytes()).hexdigest()] = p
    return out


def selftest_hashing():
    """
    Prove the hash can actually match before trusting any result from it.
    A holdout original, transformed as s04 transforms it, must collide with its
    own copy inside leak_111. If this fails, every leak check below is vacuous.
    """
    d = os.path.join(C.GAN_DATA, "leak_111")
    hold = json.load(open(os.path.join(C.SPLITS, "holdout.json")))
    if not os.path.isdir(d):
        return None
    one = os.path.join(C.REAL_RP_DIR, sorted(hold["RP"])[0])
    ref = set(img_hashes([one], square_pad=True))
    got = set(img_hashes([os.path.join(d, f) for f in sorted(os.listdir(d))
                          if f.endswith(".png")], square_pad=False))
    return bool(ref & got)


def main():
    print("=== s00 verification ===\n")

    for f in ["eye_labels.json", "holdout.json", "dev.json", "dev_folds.json"]:
        if not os.path.exists(os.path.join(C.SPLITS, f)):
            print(f"missing {f} -- run s01 and s02 first")
            return 1

    eyes = json.load(open(os.path.join(C.SPLITS, "eye_labels.json")))
    hold = json.load(open(os.path.join(C.SPLITS, "holdout.json")))
    dev = json.load(open(os.path.join(C.SPLITS, "dev.json")))
    folds = json.load(open(os.path.join(C.SPLITS, "dev_folds.json")))

    print("-- source integrity")
    orig = sorted(f for f in os.listdir(C.REAL_RP_DIR) if not f.startswith("."))
    bal = os.path.join(C.PROJECT, "BalancedDataset", "Retinitis Pigmentosa")
    bal_real = sorted(f for f in os.listdir(bal)
                      if not f.startswith(".") and "seed" not in f)
    check("139 real RP originals", len(orig) == 139, f"{len(orig)}")
    check("originals match BalancedDataset reals", orig == bal_real)

    print("\n-- eye labels")
    for cls, d in [("RP", C.REAL_RP_DIR), ("Healthy", C.HEALTHY_DIR)]:
        files = {f for f in os.listdir(d) if not f.startswith(".")}
        check(f"{cls}: every image labelled",
              files == set(eyes[cls]), f"{len(eyes[cls])} labels / {len(files)} files")

    print("\n-- holdout disjointness")
    for cls in ("RP", "Healthy"):
        hf, df = set(hold[cls]), set(dev[cls])
        he = {eyes[cls][f] for f in hf}
        de = {eyes[cls][f] for f in df}
        check(f"{cls}: no shared files", not (hf & df))
        check(f"{cls}: no shared EYES", not (he & de),
              f"holdout {len(he)} eyes, dev {len(de)} eyes")
        check(f"{cls}: partition is complete",
              hf | df == set(eyes[cls]), f"{len(hf)}+{len(df)}={len(hf)+len(df)}")

    print("\n-- cross-validation folds")
    bad_fold, no_rp = 0, 0
    off = max(eyes["Healthy"].values()) + 1
    for fd in folds:
        gtr = {eyes[c][f] + (off if c == "RP" else 0) for c, f in fd["train"]}
        gva = {eyes[c][f] + (off if c == "RP" else 0) for c, f in fd["val"]}
        if gtr & gva:
            bad_fold += 1
        if fd["n_val_rp"] == 0:
            no_rp += 1
    check(f"{len(folds)} folds: no eye spans train/val", bad_fold == 0,
          f"{bad_fold} violations")
    check("every fold has RP in validation", no_rp == 0)

    print("\n-- prevalence")
    ph = len(hold["RP"]) / (len(hold["RP"]) + len(hold["Healthy"]))
    pc = 139 / (139 + 1024)
    check("holdout prevalence within 2pp of cohort", abs(ph - pc) < 0.02,
          f"holdout {ph:.4f} vs cohort {pc:.4f}")

    print("\n-- GAN training data")
    st = selftest_hashing()
    check("hash self-test: a known holdout image IS detectable in leak_111",
          st is not False,
          "if this fails every leak check below is vacuous")
    hold_rp_paths = [os.path.join(C.REAL_RP_DIR, f) for f in sorted(hold["RP"])]
    hold_h = img_hashes(hold_rp_paths, square_pad=True)

    for name, must_contain_holdout in [("clean_111", False), ("leak_111", True)]:
        d = os.path.join(C.GAN_DATA, name)
        if not os.path.isdir(d):
            check(f"{name} exists", False)
            continue
        paths = [os.path.join(d, f) for f in sorted(os.listdir(d))
                 if f.endswith(".png")]
        h = img_hashes(paths, square_pad=False)
        overlap = set(h) & set(hold_h)
        check(f"{name}: 111 images", len(paths) == 111, f"{len(paths)}")
        if must_contain_holdout:
            check(f"{name}: contains ALL 28 holdout eyes",
                  len(overlap) == len(hold_h), f"{len(overlap)}/{len(hold_h)}")
        else:
            check(f"{name}: contains NO holdout image", not overlap,
                  f"{len(overlap)} leaked")

    pre = os.path.join(C.GAN_DATA, "pretrain_fundus")
    if os.path.isdir(pre):
        hold_all = img_hashes(
            hold_rp_paths + [os.path.join(C.HEALTHY_DIR, f) for f in sorted(hold["Healthy"])],
            square_pad=True)
        ph_ = img_hashes([os.path.join(pre, f) for f in sorted(os.listdir(pre))
                          if f.endswith(".png")], square_pad=False)
        check("pretrain_fundus: contains no holdout image",
              not (set(ph_) & set(hold_all)), f"{len(set(ph_) & set(hold_all))} leaked")

    print("\n-- packaged zips")
    zs = {}
    for name in ("clean_111", "leak_111"):
        p = os.path.join(C.GAN_DATA, f"{name}.zip")
        if not os.path.exists(p):
            check(f"{name}.zip exists", False)
            continue
        zf = zipfile.ZipFile(p)
        names = sorted(n for n in zf.namelist() if n.endswith(".png"))
        m = hashlib.md5()
        for n in names:
            m.update(zf.read(n))
        zs[name] = m.hexdigest()
        check(f"{name}.zip: 111 images", len(names) == 111, f"{len(names)}")
    if len(zs) == 2:
        check("clean and leak zips differ in CONTENT",
              zs["clean_111"] != zs["leak_111"],
              "identical byte size alone is not evidence of duplication")

    print("\n-- synthetic pool")
    if os.path.isdir(C.LEGACY_SYNTH_DIR):
        sp = [os.path.join(C.LEGACY_SYNTH_DIR, f)
              for f in sorted(os.listdir(C.LEGACY_SYNTH_DIR)) if not f.startswith(".")]
        real_h = img_hashes([os.path.join(C.REAL_RP_DIR, f) for f in orig], square_pad=False)
        sh = img_hashes(sp, square_pad=False)
        check("legacy synthetic pool contains no real image",
              not (set(sh) & set(real_h)), f"{len(sp)} synthetic")

    print(f"\n=== {'ALL CHECKS PASSED' if not FAIL else f'{len(FAIL)} FAILURES: ' + ', '.join(FAIL)} ===")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
