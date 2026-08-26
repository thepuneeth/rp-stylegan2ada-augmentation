"""
s04 - Materialise the image folders each generator will train on.

Three folders, three jobs
-------------------------
clean_111/     The 111 DEV RP images (106 eyes). The generator trained here has
               NEVER seen the 28 holdout eyes, so any novelty it shows on them
               is real. This is the study's primary generator.

leak_111/      The 28 HOLDOUT eyes plus 83 DEV images, 111 total. Deliberately
               contaminated, and matched to clean_111 in size so that the ONLY
               difference between the two arms is whether the generator saw the
               test eyes. The gap between them is the leakage tax, measured with
               training-set size controlled.

               Size matching is what makes this causal. Comparing against the
               legacy v3 pool instead would confound leakage with training-set
               size, 139 versus 111, both pushing the same direction, and would
               yield only an upper bound.

pretrain_fundus/  DEV healthy fundus, 817 images, for stage-2 domain-adjacent
               pretraining. See docs/PRETRAIN_CORPUS.md for why this stage
               exists and what a larger corpus would add.

Every folder is written as 256x256 PNG with the field of view preserved, which
is what StyleGAN2-ADA's dataset_tool expects to package.

Outputs the folders plus outputs/s04_gan_data.json, and prints the exact
dataset_tool and train commands for each arm.
"""
import os, sys, json, shutil
import numpy as np
from PIL import Image

from common import C

SZ = 256


def prep(src_paths, dst_dir):
    """Square-pad to preserve aspect, resize to 256, write PNG."""
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)
    for i, p in enumerate(sorted(src_paths)):
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = max(w, h)
            canvas = Image.new("RGB", (s, s), (0, 0, 0))
            canvas.paste(im, ((s - w) // 2, (s - h) // 2))
            canvas.resize((SZ, SZ), Image.LANCZOS).save(
                os.path.join(dst_dir, f"img{i:05d}.png"))
    return len(src_paths)


def main():
    with open(os.path.join(C.SPLITS, "dev.json")) as fh:
        dev = json.load(fh)
    with open(os.path.join(C.SPLITS, "holdout.json")) as fh:
        hold = json.load(fh)

    dev_rp = sorted(dev["RP"])
    hold_rp = sorted(hold["RP"])
    res = {}

    # ---- clean arm: DEV only, holdout never seen
    clean = [os.path.join(C.REAL_RP_DIR, f) for f in dev_rp]
    n = prep(clean, os.path.join(C.GAN_DATA, "clean_111"))
    res["clean_111"] = {"n": n, "eyes": len(set(dev["RP"].values())),
                        "contains_holdout": False}
    print(f"clean_111        {n} images, {res['clean_111']['eyes']} eyes")

    # ---- leaky arm: all holdout eyes + DEV filler, size-matched to clean
    rng = np.random.default_rng(C.SEED)
    filler = list(dev_rp)
    rng.shuffle(filler)
    filler = sorted(filler[: n - len(hold_rp)])
    leak_files = sorted(hold_rp + filler)
    assert len(leak_files) == n, "leak arm not size-matched to clean arm"
    assert set(hold_rp) <= set(leak_files), "leak arm must contain every holdout eye"
    leak = ([os.path.join(C.REAL_RP_DIR, f) for f in hold_rp]
            + [os.path.join(C.REAL_RP_DIR, f) for f in filler])
    prep(leak, os.path.join(C.GAN_DATA, "leak_111"))
    res["leak_111"] = {"n": len(leak_files), "n_holdout_included": len(hold_rp),
                       "n_dev_filler": len(filler), "contains_holdout": True}
    print(f"leak_111         {len(leak_files)} images "
          f"({len(hold_rp)} holdout + {len(filler)} dev filler)")

    # ---- stage-2 pretraining corpus: healthy fundus from DEV only
    pre = [os.path.join(C.HEALTHY_DIR, f) for f in sorted(dev["Healthy"])]
    n_pre = prep(pre, os.path.join(C.GAN_DATA, "pretrain_fundus"))
    res["pretrain_fundus"] = {"n": n_pre, "source": "DEV healthy only",
                              "contains_holdout": False}
    print(f"pretrain_fundus  {n_pre} images (DEV healthy, holdout excluded)")

    # Holdout healthy is excluded so the classifier holdout stays clean too:
    # a generator that had seen holdout healthy images could bias the negative
    # class just as easily as the positive one.
    assert not (set(dev["Healthy"]) & set(hold["Healthy"])), "healthy split leaked"

    with open(os.path.join(C.OUT, "s04_gan_data.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    sg = os.path.join(C.PROJECT, "stylegan2-ada-pytorch-main")
    print("\n--- package for StyleGAN2-ADA ---")
    for name in ["clean_111", "leak_111", "pretrain_fundus"]:
        print(f"python {sg}/dataset_tool.py --source {C.GAN_DATA}/{name} "
              f"--dest {C.GAN_DATA}/{name}.zip")
    print(f"\nwrote {C.OUT}/s04_gan_data.json")


if __name__ == "__main__":
    main()
