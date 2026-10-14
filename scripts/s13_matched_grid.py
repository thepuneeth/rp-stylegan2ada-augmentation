"""
Matched sample grid: one holdout eye per column.

The old fig_sample_grid stacked three INDEPENDENT random draws, so column c of
the real row had nothing to do with column c of the synthetic rows and nothing
could be compared downwards. Here each column starts from a real holdout eye,
and the synthetic rows show the sample from each generator that lies closest to
it in Inception pool3 feature space -- the same features FID and KID are built
on, so "closest" means the same thing in the figure as it does in the numbers.

This is RETRIEVAL, not conditional generation: nothing asked either generator to
reproduce a given eye. The caption has to say so, or the figure reads as a
memorization claim.

    ../venv/bin/python scripts/s13_matched_grid.py
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import config as C

ROWS = [("clean-GAN", "synth_clean_psi1.0"),
        ("contaminated-GAN", "synth_leak_psi1.0")]


def _inception(device):
    from torchvision import models
    net = models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
    net.fc = torch.nn.Identity()                 # 2048-d pool3 features
    return net.eval().to(device)


def _features(net, paths, device, bs=32):
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = torch.stack([tf(Image.open(p).convert("RGB"))
                                 for p in paths[i:i + bs]]).to(device)
            f = net(batch)
            out.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(out).numpy()


def main(a):
    device = "cpu"
    hold = json.load(open(os.path.join(C.SPLITS, "holdout.json")))
    reals = [os.path.join(C.REAL_RP_DIR, f) for f in sorted(hold["RP"])][:a.n_cols]
    net = _inception(device)
    fr = _features(net, reals, device)

    picks, sims = [], []
    for _, sub in ROWS:
        d = os.path.join(C.GAN_DATA, sub)
        pool = [os.path.join(d, f) for f in sorted(os.listdir(d))
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        fs = _features(net, pool, device)
        # a non-finite feature row would WIN argmax rather than lose it, so it is
        # pushed below every real candidate instead of being left to propagate.
        n_bad = int((~np.isfinite(fs).all(1)).sum())
        if n_bad:
            print(f"  warning: {n_bad} non-finite feature rows excluded")
            fs = np.where(np.isfinite(fs), fs, 0.0)
        s = fr @ fs.T                                    # cosine, rows normalised
        s = np.where(np.isfinite(s), s, -np.inf)
        idx = s.argmax(1)
        picks.append([pool[i] for i in idx])
        sims.append([float(s[r, i]) for r, i in enumerate(idx)])
        print(f"  {sub}: {len(pool)} candidates, "
              f"cosine {np.mean(sims[-1]):.3f} mean over columns")

    n = len(reals)
    fig, ax = plt.subplots(3, n, figsize=(1.35 * n, 1.45 * 3))
    for c, p in enumerate(reals):
        with Image.open(p) as im:
            ax[0, c].imshow(im.convert("RGB").resize((C.IMG_SZ, C.IMG_SZ)))
    ax[0, 0].set_ylabel("real RP", fontsize=7)
    for r, ((lab, _), row) in enumerate(zip(ROWS, picks), start=1):
        for c, p in enumerate(row):
            with Image.open(p) as im:
                ax[r, c].imshow(im.convert("RGB").resize((C.IMG_SZ, C.IMG_SZ)))
        ax[r, 0].set_ylabel(lab, fontsize=7)
    for axis in ax.ravel():
        axis.set_xticks([]); axis.set_yticks([])
    fig.subplots_adjust(wspace=.02, hspace=.02)
    out = os.path.join(C.ROOT, "figures", "fig_matched_grid.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    json.dump({"reals": reals, "picks": {ROWS[i][0]: picks[i] for i in range(2)},
               "cosine": {ROWS[i][0]: sims[i] for i in range(2)}},
              open(os.path.join(C.OUT, "s13_matched_grid.json"), "w"), indent=2)
    print("wrote figures/fig_matched_grid.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cols", type=int, default=6)
    main(ap.parse_args())
