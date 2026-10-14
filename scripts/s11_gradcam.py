"""
Grad-CAM over the classifier arms.

Carried over from the v3 paper (`metrics_code/gradcams.ipynb`), but used here as
an INSTRUMENT rather than an illustration. The v3 notebook carries the author's
own observation that the model was "adapting to the outside color / abnormalities
of the synthesized images" -- shortcut learning on synthetic border artefacts,
fought at the time with aggressive augmentation.

That makes the question here: does training on synthetic images move the
classifier's attention OFF the retina? A high AUC is not by itself evidence
that anything retinal was learned, so attention is checked directly.

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

    def __call__(self, x, class_idx=None, normalize_after_resize=False):
        out = self.model(x)
        if class_idx is None:
            class_idx = int(out.argmax(1)[0])
        one = torch.zeros_like(out)
        one[0, class_idx] = 1.0
        self.model.zero_grad()
        out.backward(gradient=one, retain_graph=True)
        w = self.grads.mean(dim=(2, 3), keepdim=True)       # GAP over spatial
        cam = torch.relu((w * self.acts).sum(1))[0]
        if normalize_after_resize:
            cam = torch.nn.functional.interpolate(
                cam[None, None], size=x.shape[2:], mode="bilinear",
                align_corners=False)[0, 0]
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


def _train(S, items, loss_name, weights, backbone, device, seed, train_tf=None):
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
    dl = DataLoader(S.ImgDS(items, train_tf or S.TRAIN_TF), batch_size=C.BATCH, shuffle=True,
                    **S.loader_kw(device, seed))
    model.train()
    for _ in range(C.EPOCHS):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    return model.eval()


# reader-facing row labels; the arms keep their on-disk names everywhere else
ARM_LABEL = {"real_only": "real-only", "oversample": "oversampled",
             "synth_clean": "clean-GAN", "synth_leak": "contaminated-GAN"}


def main(a):
    import s05_classifier as S
    from torchvision import models
    device = S.pick_device()
    dev = json.load(open(os.path.join(C.SPLITS, "dev.json")))
    hold = json.load(open(os.path.join(C.SPLITS, "holdout.json")))
    pools = {"synth_clean": S.load_pool(a.synth_clean_dir) if a.synth_clean_dir else [],
             "synth_leak": S.load_pool(a.synth_leak_dir) if a.synth_leak_dir else []}
    rp_hold = [os.path.join(C.REAL_RP_DIR, f) for f in sorted(hold["RP"])][:a.n_images]

    # Same decode-once cache s05 uses. The sources are 2004x1690 JPEGs and each
    # arm retrains from scratch, so without this the run is all JPEG decode.
    S.prefill_cache([os.path.join(C.HEALTHY_DIR, f) for f in dev["Healthy"]]
                    + [os.path.join(C.REAL_RP_DIR, f) for f in dev["RP"]]
                    + rp_hold
                    + [p for arm in a.arms for p in (pools.get(arm) or [])],
                    workers=int(os.environ.get("S05_WORKERS", 16)) or 16)

    # The null on-disc fraction. A CAM that attends to NOTHING in particular --
    # uniform over the frame -- scores the disc's share of the image, not zero.
    # Without this number an on-disc fraction of 0.8 cannot be read as "on the
    # retina": if the disc covers 80% of the frame, 0.8 IS indifference.
    disc_area = float(np.mean([disc_mask(p, C.IMG_SZ).mean() for p in rp_hold]))
    print(f"null: disc covers {disc_area:.3f} of frame "
          f"(a uniform CAM scores this)")

    res, fig_rows = {}, []
    for arm in a.arms:
        rng = np.random.default_rng(C.SEED)
        items, w, loss = S.build_train_items(dev, arm, a.dose, pools.get(arm), rng)
        print(f"[{arm}] training on {len(items)} images ...", flush=True)
        model = _train(S, items, loss, w, a.backbone, device, C.SEED,
                       train_tf=S.AUGS[a.aug])
        if a.backbone == "resnet18":
            layer = model.layer4[-1]
        elif a.backbone == "microresnet":
            # f[-1] is 16x16; f[-2] is 32x32, matching the original CNN CAM.
            layer = model.f[-2] if a.cam_layer == "32" else model.f[-1]
        else:
            layer = model.f[-3]          # last conv of the ToyCNN / OrigCNN stack
        cam_fn, fracs, cams, preds = GradCAM(model, layer), [], [], []
        for p in rp_hold:
            x = S.EVAL_TF(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            # class_idx=None -> the PREDICTED class, which is what the original
            # study's Grad-CAM did (grad_cam(batch, pred_idx)). Forcing class 1
            # asks a model that predicts Healthy to justify RP, and returns
            # noise rather than that model's actual evidence.
            cam, k = cam_fn(
                x, class_idx=None if a.cam_class == "pred" else 1,
                normalize_after_resize=a.cam_norm == "after_resize")
            fracs.append(on_disc_fraction(cam, disc_mask(p, C.IMG_SZ)))
            cams.append(cam); preds.append(int(k))
        res[arm] = {"on_disc_fraction": fracs, "n_train": len(items),
                    "pred": preds}
        fig_rows.append((arm, rp_hold, cams))

    # A constant CAM makes on_disc_fraction undefined (total mass 0). Averaging
    # each arm over whatever survived would compare arms on different images, so
    # an image undefined in ANY arm is dropped from ALL of them and the count is
    # reported. Previously these were nan-dropped per arm, which silently left
    # the arms averaged over different holdout eyes.
    bad = sorted({i for v in res.values()
                  for i, x in enumerate(v["on_disc_fraction"])
                  if not np.isfinite(x)})
    keep = [i for i in range(len(rp_hold)) if i not in bad]
    print(f"excluded {len(bad)} of {len(rp_hold)} holdout eyes with an "
          f"undefined CAM in some arm: {bad}")
    for arm, v in res.items():
        f = [v["on_disc_fraction"][i] for i in keep]
        v["on_disc_fraction_mean"] = float(np.mean(f))
        v["n_scored"] = len(f)
        v["excluded_idx"] = bad
        print(f"[{arm}] on-disc CAM mass {v['on_disc_fraction_mean']:.3f} "
              f"over {len(f)} eyes (null {disc_area:.3f}, "
              f"lift {v['on_disc_fraction_mean']-disc_area:+.3f})")

    os.makedirs(os.path.join(C.ROOT, "figures"), exist_ok=True)
    # every holdout eye is scored, but the figure only shows the first --n-cols
    # of them; a 28-wide grid is unreadable at column width.
    n = min(len(rp_hold), a.n_cols or len(rp_hold))
    # Which eyes the figure shows. "first" = the first n, unselected.
    # "orig" ports the original study's _select_rp_example heuristic: prefer
    # eyes the baseline calls Healthy and a synthetic arm calls RP, i.e. the
    # ones where augmentation changed the decision. This is a SELECTION and is
    # recorded in the JSON as such so the figure is not read as a random sample.
    order = list(range(len(rp_hold)))
    if a.select == "orig" and "real_only" in res:
        base = res["real_only"]["pred"]
        syn = next((res[k]["pred"] for k in ("synth_clean", "synth_leak")
                    if k in res), None)
        if syn is not None:
            order.sort(key=lambda i: (not (base[i] == 0 and syn[i] == 1), i))
    order = [i for i in order if i not in bad]
    shown = order[:n]
    print(f"figure shows eyes {shown} (select={a.select})")
    fig, ax = plt.subplots(len(fig_rows) + 1, n,
                           figsize=(1.6 * n, 1.7 * (len(fig_rows) + 1)))
    ax = np.atleast_2d(ax)
    for c, p in enumerate([rp_hold[i] for i in shown]):  # the eyes themselves
        with Image.open(p) as im:
            ax[0, c].imshow(im.convert("RGB").resize((C.IMG_SZ, C.IMG_SZ)))
        ax[0, c].set_xticks([]); ax[0, c].set_yticks([])
    ax[0, 0].set_ylabel("holdout eye", fontsize=8)
    for r, (arm, paths, cams) in enumerate(fig_rows, start=1):
        for c, (p, cam) in enumerate([(paths[i], cams[i]) for i in shown]):
            with Image.open(p) as im:
                ax[r, c].imshow(im.convert("RGB").resize((C.IMG_SZ, C.IMG_SZ)))
            ax[r, c].imshow(np.asarray(Image.fromarray((cam * 255).astype(np.uint8))
                                       .resize((C.IMG_SZ, C.IMG_SZ), Image.BILINEAR)),
                            cmap="jet", alpha=.45)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        ax[r, 0].set_ylabel(ARM_LABEL.get(arm, arm), fontsize=8)
    sfx = f"_{a.tag}" if a.tag else ""
    png = os.path.join(C.ROOT, "figures", f"fig_gradcam{sfx}.png")
    js = os.path.join(C.OUT, f"s11_gradcam{sfx}.json")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    json.dump({"disc_area_null": disc_area, "backbone": a.backbone,
               "aug": a.aug, "dose": a.dose, "cam_class": a.cam_class,
               "cam_layer": a.cam_layer, "cam_norm": a.cam_norm,
               "select": a.select, "shown_idx": shown, "arms": res},
              open(js, "w"), indent=2)
    print(f"wrote {png} and {js}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--arms", nargs="*", default=["real_only", "synth_clean"])
    ap.add_argument("--dose", type=float, default=1.0)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--n-images", type=int, default=6)
    ap.add_argument("--n-cols", type=int, default=0,
                    help="columns to draw (0 = all scored images)")
    ap.add_argument("--synth-clean-dir", default=None)
    ap.add_argument("--synth-leak-dir", default=None)
    ap.add_argument("--aug", default="current", choices=["current", "orig"],
                    help="'orig' = the original notebooks' train transform")
    ap.add_argument("--cam-class", default="pred", choices=["pred", "rp"],
                    help="pred = the model's predicted class, as in the "
                         "original study; rp = force class 1.")
    ap.add_argument("--cam-layer", default="auto", choices=["auto", "32"],
                    help="auto = existing target layer; 32 = MicroResNet's "
                         "earlier 32x32 block")
    ap.add_argument("--cam-norm", default="native",
                    choices=["native", "after_resize"],
                    help="normalise at native CAM resolution or after "
                         "upsampling, as in the original study")
    ap.add_argument("--select", default="first", choices=["first", "orig"],
                    help="which holdout eyes the figure shows. orig ports the "
                         "original study's selection heuristic and is recorded "
                         "in the JSON.")
    ap.add_argument("--tag", default="",
                    help="suffix for the output files; without it every run "
                         "overwrites fig_gradcam.png")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        main(a)
