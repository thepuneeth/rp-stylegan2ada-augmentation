"""Render the original five-column Grad-CAM layout on held-out eyes.

This is an exploratory layout check: each arm trains five seeded models on the
same eye-disjoint dev split, then all five CAMs use one holdout RP eye. The
five fits are not presented as cross-validation estimates.
"""
import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import s05_classifier as S
from s11_gradcam import ARM_LABEL, GradCAM, _train


MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def _layer(model, backbone, cam_layer):
    if backbone == "resnet18":
        return model.layer4[-1]
    if backbone == "microresnet":
        return model.f[-2] if cam_layer == "32" else model.f[-1]
    return model.f[-3]


def _overlay(cam, image):
    """Original-study rendering: resize, normalize, JET, 0.45/0.55 blend."""
    cam = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (C.IMG_SZ, C.IMG_SZ), Image.BILINEAR), dtype=np.float32) / 255.0
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(0.45 * heat + 0.55 * image, 0, 1)


def _prediction(model, paths, device):
    out = []
    with torch.no_grad():
        for p in paths:
            x = S.EVAL_TF(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            out.append(int(model(x).argmax(1)[0]))
    return out


def _choose(preds):
    """Port original outcome-conditioned selection across the five models.

    If no eye meets the strict preference, use its deterministic count-based
    fallback rather than silently selecting the first file.
    """
    base = np.asarray(preds["real_only"])
    syn = np.asarray(preds["synth_clean"] if "synth_clean" in preds
                     else preds["synth_leak"])
    candidates = [i for i in range(base.shape[1])
                  if base[:, i].sum() <= 1 and syn[:, i].sum() >= 4]
    if candidates:
        return candidates[0]
    return min(range(base.shape[1]),
               key=lambda i: (base[:, i].sum(), -syn[:, i].sum(), i))


def main(a):
    dev = json.load(open(os.path.join(C.SPLITS, "dev.json")))
    hold = json.load(open(os.path.join(C.SPLITS, "holdout.json")))
    paths = [os.path.join(C.REAL_RP_DIR, f) for f in sorted(hold["RP"])]
    pools = {
        "synth_clean": S.load_pool(a.synth_clean_dir),
        "synth_leak": S.load_pool(a.synth_leak_dir),
    }
    device = S.pick_device()
    S.prefill_cache(
        [os.path.join(C.HEALTHY_DIR, f) for f in dev["Healthy"]]
        + [os.path.join(C.REAL_RP_DIR, f) for f in dev["RP"]]
        + paths
        + [p for arm in a.arms for p in pools.get(arm, [])],
        workers=int(os.environ.get("S05_WORKERS", 16)) or 16,
    )

    fits, preds = {}, {arm: [] for arm in a.arms}
    for arm in a.arms:
        rng = np.random.default_rng(C.SEED)
        items, weights, loss = S.build_train_items(
            dev, arm, a.dose, pools.get(arm), rng)
        fits[arm] = []
        for fold in range(a.n_models):
            seed = C.SEED + fold
            print(f"[{arm}] model {fold + 1}/{a.n_models}, seed={seed}", flush=True)
            # Weights are cached so changing only which eye is visualised is
            # seconds rather than a full refit. Key covers everything that
            # affects the fit; the eye index deliberately does not.
            ck = os.path.join(
                C.ROOT, "outputs", "foldgrid_weights",
                f"{a.backbone}_{a.aug}_dose{a.dose}_{arm}_seed{seed}.pt")
            os.makedirs(os.path.dirname(ck), exist_ok=True)
            if os.path.exists(ck):
                model = S.build_model(a.backbone, device)
                model.load_state_dict(torch.load(ck, map_location=device))
                model.eval()
                print("  loaded cached weights", flush=True)
            else:
                model = _train(S, items, loss, weights, a.backbone, device,
                               seed, train_tf=S.AUGS[a.aug])
                torch.save(model.state_dict(), ck)
            p = _prediction(model, paths, device)
            fits[arm].append((seed, model))
            preds[arm].append(p)

    picks = ([int(v) for v in str(a.eye_idx).split(",")]
             if a.eye_idx is not None else [_choose(preds)])
    for selected in picks:
        _render(a, S, C, device, paths, fits, preds, selected)


def _render(a, S, C, device, paths, fits, preds, selected):
    p = paths[selected]
    print(f"selected holdout idx {selected}: {p}")
    fig, ax = plt.subplots(len(a.arms) + 1, a.n_models,
                           figsize=(7.0, 1.8 * (len(a.arms) + 1)))
    ax = np.atleast_2d(ax)
    with Image.open(p) as im:
        raw = np.asarray(im.convert("RGB").resize((C.IMG_SZ, C.IMG_SZ))) / 255.0
    for c, (seed, _) in enumerate(fits[a.arms[0]]):
        ax[0, c].imshow(raw)
        ax[0, c].set_title(f"seed {seed}", fontsize=8, pad=2)
    ax[0, 0].set_ylabel("holdout eye", fontsize=8)

    for r, arm in enumerate(a.arms, start=1):
        for c, (seed, model) in enumerate(fits[arm]):
            x = S.EVAL_TF(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            cam, k = GradCAM(model, _layer(model, a.backbone, a.cam_layer))(x)
            image = x[0].detach().cpu().permute(1, 2, 0).numpy() * STD + MEAN
            ax[r, c].imshow(_overlay(cam, np.clip(image, 0, 1)))
            ax[r, c].set_title(f"pred {k}", fontsize=7, pad=2)
        ax[r, 0].set_ylabel(ARM_LABEL.get(arm, arm), fontsize=8)

    for axis in ax.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    fig.subplots_adjust(left=0.10, right=0.995, top=0.96, bottom=0.01,
                        wspace=0.015, hspace=0.08)
    os.makedirs(os.path.join(C.ROOT, "figures"), exist_ok=True)
    tag = (a.tag or f"foldgrid_{a.backbone}_{a.aug}") + f"_eye{selected}"
    png = os.path.join(C.ROOT, "figures", f"fig_gradcam_{tag}.png")
    js = os.path.join(C.OUT, f"s11_gradcam_{tag}.json")
    fig.savefig(png, dpi=300)
    json.dump({
        "backbone": a.backbone, "aug": a.aug, "dose": a.dose,
        "cam_class": "pred", "cam_layer": a.cam_layer,
        "cam_norm": "after_resize", "overlay": "cv2_jet_0.45_cam_0.55_image",
        "n_models": a.n_models, "seeds": [C.SEED + i for i in range(a.n_models)],
        "selection": ("manual --eye-idx, chosen on VISIBLE PIGMENTATION, "
                      "not on model outcome"
                      if a.eye_idx is not None else
                      "original outcome heuristic over five seeded models"),
        "selected_idx": selected, "selected_path": p,
        "pred_selected": {arm: [v[selected] for v in vals]
                          for arm, vals in preds.items()},
        "predictions": preds,
    }, open(js, "w"), indent=2)
    print(f"wrote {png} and {js}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*",
                    default=["real_only", "synth_clean", "synth_leak"])
    ap.add_argument("--dose", type=float, default=10.0)
    ap.add_argument("--backbone", default="microresnet")
    ap.add_argument("--aug", default="current", choices=["current", "orig"])
    ap.add_argument("--cam-layer", default="auto", choices=["auto", "32"])
    ap.add_argument("--n-models", type=int, default=5)
    ap.add_argument("--synth-clean-dir", required=True)
    ap.add_argument("--synth-leak-dir", required=True)
    ap.add_argument("--eye-idx", default=None,
                    help="holdout RP index to visualise, overriding the "
                         "outcome heuristic. Selecting on VISIBLE PATHOLOGY "
                         "rather than on model outcome; record which and why.")
    ap.add_argument("--tag", default="")
    main(ap.parse_args())
