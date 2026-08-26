"""
Grad-CAM over the classifier arms.

Carried over from the v3 paper (`metrics_code/gradcams.ipynb`), but used here as
an INSTRUMENT rather than an illustration. The v3 notebook carries the author's
own observation that the model was "adapting to the outside color / abnormalities
of the synthesized images" -- shortcut learning on synthetic border artefacts,
fought at the time with aggressive augmentation.

That makes the question here: does training on synthetic images move the
classifier's attention OFF the retina? It matters because s05's own colour_lr
control reaches AUC 0.950 from colour moments alone, so a high AUC is not by
itself evidence that anything retinal was learned.

Reported per arm:
  on_disc_fraction   share of CAM mass inside the fundus disc (the retina).
                     A model attending to the black border or the bright
                     acquisition notch scores low regardless of its AUC.

This trains its own model per arm at a fixed seed rather than reusing an s05
checkpoint, because s05 does not persist models. The models are therefore
comparable to each other but are not the exact s05 fits.

    ../venv/bin/python scripts/s11_gradcam.py --self-test
    ../venv/bin/python scripts/s11_gradcam.py --arms real_only synth_clean
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


class GradCAM:
    """Hooks a conv layer, weights its activations by mean gradient."""

    def __init__(self, model, target_layer):
        self.model, self.acts, self.grads = model, None, None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, class_idx=None):
        out = self.model(x)
        if class_idx is None:
            class_idx = int(out.argmax(1)[0])
        one = torch.zeros_like(out)
        one[0, class_idx] = 1.0
        self.model.zero_grad()
        out.backward(gradient=one, retain_graph=True)
        w = self.grads.mean(dim=(2, 3), keepdim=True)       # GAP over spatial
        cam = torch.relu((w * self.acts).sum(1))[0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).cpu().numpy(), class_idx


def disc_mask(path, size):
    """Fundus disc as a boolean mask: non-black pixels of the raw image."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR))
    return g > 12                                            # same cut as common.fov_crop


def on_disc_fraction(cam, mask):
    """Share of CAM mass falling inside the retina rather than on the border."""
    from PIL import Image as _I
    c = np.asarray(_I.fromarray((cam * 255).astype(np.uint8))
                   .resize(mask.shape[::-1], _I.BILINEAR)).astype(float)
    tot = c.sum()
    return float(c[mask].sum() / tot) if tot > 0 else float("nan")


def _self_test():
    """The CAM machinery must run and the disc metric must behave sanely."""
    from torchvision import models
    m = models.resnet18(weights="IMAGENET1K_V1").eval()
    cam_fn = GradCAM(m, m.layer4[-1])
    cam, idx = cam_fn(torch.randn(1, 3, 224, 224))
    assert cam.ndim == 2 and 0.0 <= cam.min() and abs(cam.max() - 1.0) < 1e-5, \
        f"CAM must be a normalised 2-D map, got {cam.shape} max={cam.max()}"

    centre = np.zeros((16, 16)); centre[6:10, 6:10] = 1.0
    border = np.zeros((16, 16)); border[0, :] = 1.0
    mask = np.zeros((64, 64), bool); mask[12:52, 12:52] = True
    fc, fb = on_disc_fraction(centre, mask), on_disc_fraction(border, mask)
    assert fc > 0.9, f"centred CAM should sit on the disc, got {fc}"
    assert fb < 0.1, f"border CAM should sit off the disc, got {fb}"
    print(f"self-test OK  cam{cam.shape} class={idx}  "
          f"on_disc centred={fc:.3f} border={fb:.3f}")


def _train(S, items, loss_name, weights, backbone, device, seed):
    """
    s05's training loop, reproduced because s05.run_once trains and evaluates in
    one call and returns only predictions -- there is no model to hand back.
    Kept deliberately as a thin wrapper over s05's OWN components (build_model,
    ImgDS, TRAIN_TF, FocalLoss, C.EPOCHS, C.LR, C.BATCH) so it cannot silently
    drift from the arm it is supposed to explain.
    """
    import torch.nn as nn
    from torch.utils.data import DataLoader
    torch.manual_seed(seed); np.random.seed(seed)
    model = S.build_model(backbone, device)
    if loss_name == "focal":
        crit = S.FocalLoss()
    elif weights == "balanced":
        y = np.array([b for _, b in items])
        w = torch.tensor([len(y) / (2 * (y == 0).sum()),
                          len(y) / (2 * (y == 1).sum())],
                         dtype=torch.float32, device=device)
        crit = nn.CrossEntropyLoss(weight=w)
    else:
        crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    dl = DataLoader(S.ImgDS(items, S.TRAIN_TF), batch_size=C.BATCH, shuffle=True)
    model.train()
    for _ in range(C.EPOCHS):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    return model.eval()


def main(a):
    import s05_classifier as S
    from torchvision import models
    device = C.DEVICE if torch.backends.mps.is_available() else "cpu"
    dev = json.load(open(os.path.join(C.SPLITS, "dev.json")))
    hold = json.load(open(os.path.join(C.SPLITS, "holdout.json")))
    pools = {"synth_clean": S.load_pool(a.synth_clean_dir) if a.synth_clean_dir else [],
             "synth_leak": S.load_pool(a.synth_leak_dir) if a.synth_leak_dir else [],
             "pca_sampler": S.load_pool(a.pca_dir) if a.pca_dir else []}
    rp_hold = [os.path.join(C.REAL_RP_DIR, f) for f in sorted(hold["RP"])][:a.n_images]

    res, fig_rows = {}, []
    for arm in a.arms:
        rng = np.random.default_rng(C.SEED)
        items, w, loss = S.build_train_items(dev, arm, a.dose, pools.get(arm), rng)
        print(f"[{arm}] training on {len(items)} images ...", flush=True)
        model = _train(S, items, loss, w, a.backbone, device, C.SEED)
        layer = model.layer4[-1] if a.backbone == "resnet18" else model.f[-3]
        cam_fn, fracs, cams = GradCAM(model, layer), [], []
        for p in rp_hold:
            x = S.EVAL_TF(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            cam, _ = cam_fn(x, class_idx=1)
            fracs.append(on_disc_fraction(cam, disc_mask(p, C.IMG_SZ)))
            cams.append(cam)
        res[arm] = {"on_disc_fraction_mean": float(np.nanmean(fracs)),
                    "on_disc_fraction": fracs, "n_train": len(items)}
        fig_rows.append((arm, rp_hold, cams))
        print(f"  on-disc CAM mass {np.nanmean(fracs):.3f}")

    os.makedirs(os.path.join(C.ROOT, "figures"), exist_ok=True)
    n = len(rp_hold)
    fig, ax = plt.subplots(len(fig_rows), n, figsize=(1.6 * n, 1.7 * len(fig_rows)))
    ax = np.atleast_2d(ax)
    for r, (arm, paths, cams) in enumerate(fig_rows):
        for c, (p, cam) in enumerate(zip(paths, cams)):
            with Image.open(p) as im:
                ax[r, c].imshow(im.convert("RGB").resize((C.IMG_SZ, C.IMG_SZ)))
            ax[r, c].imshow(np.asarray(Image.fromarray((cam * 255).astype(np.uint8))
                                       .resize((C.IMG_SZ, C.IMG_SZ), Image.BILINEAR)),
                            cmap="jet", alpha=.45)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        ax[r, 0].set_ylabel(arm, fontsize=8)
    fig.savefig(os.path.join(C.ROOT, "figures", "fig_gradcam.png"),
                dpi=200, bbox_inches="tight")
    json.dump(res, open(os.path.join(C.OUT, "s11_gradcam.json"), "w"), indent=2)
    print("wrote figures/fig_gradcam.png and outputs/s11_gradcam.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--arms", nargs="*", default=["real_only", "synth_clean"])
    ap.add_argument("--dose", type=float, default=1.0)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--n-images", type=int, default=6)
    ap.add_argument("--synth-clean-dir", default=None)
    ap.add_argument("--synth-leak-dir", default=None)
    ap.add_argument("--pca-dir", default=None)
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        main(a)
