"""
s05 - Does synthetic augmentation beat copying, on eyes no generator ever saw?

Protocol
--------
Train on DEV. Evaluate on the generator-blind HOLDOUT. Repeat over seeds. Every
arm sees the identical holdout, so arms are compared paired.

    imbalanced   no correction, the untreated control
    weighted     class-weighted cross-entropy
    oversample   random oversampling of real RP  <-- COPYING. the baseline to beat
    focal        focal loss, gamma = 2
    synthetic    real RP plus synthetic RP at a given dose and truncation
    colour_lr    logistic regression on global colour moments, a control

Why "oversample" is the baseline and not "imbalanced"
-----------------------------------------------------
Oversampling duplicates existing minority images and adds zero new information.
It is the exact operationalisation of "copying the existing RP images". A
generator that only reproduces its training distribution can at best match it.
Beating it is the whole claim, so beating "imbalanced" proves nothing.

Why colour_lr is included
-------------------------
A source-shortcut audit found that global colour moments alone separate these
classes at AUC 0.947, and that a logistic regression matched the CNN. If every
deep arm lands near the colour baseline, the benchmark is saturated and no
augmentation difference is interpretable. This arm is a validity check on the
whole comparison, not a competitor.

Augmentation bugs inherited from corrected_cv_v2.py, fixed here
---------------------------------------------------------------
    RandomAffine((-15, -15), ...)      a CONSTANT -15 degree rotation, never
                                       random. Now degrees=15, a real range.
    ColorJitter(contrast=(0.2, 1.0))   one-sided, could only ever darken. Now
                                       symmetric about 1.0.
    shear=40                           anatomically meaningless on a fundus.
                                       Removed.
    rng_seed = SEED + 1000 * rep       no fold term, so every fold in a repeat
                                       shared an initialisation. Now includes it.
    synth[:need]                       the same synthetic images every fold.
                                       Now resampled per run.
    Linear(32*16*16, n)                position-sensitive flatten head. Now
                                       global average pooling, so a lesion is
                                       recognised wherever it appears.

Outputs outputs/s05_classifier_<tag>.json
"""
import os, sys, json, argparse, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score

from common import C

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Symmetric, anatomically plausible. Fundus images have no canonical up beyond
# roughly +/- 15 degrees of head tilt, and horizontal flip maps left eye to
# right eye, which is legitimate.
TRAIN_TF = transforms.Compose([
    transforms.Resize((C.IMG_SZ, C.IMG_SZ)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.05, 0.05),
                            scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
    transforms.ToTensor(), NORM,
])
EVAL_TF = transforms.Compose([
    transforms.Resize((C.IMG_SZ, C.IMG_SZ)), transforms.ToTensor(), NORM,
])


class ImgDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        with Image.open(p) as im:
            x = self.tf(im.convert("RGB"))
        return x, y


class ToyCNN(nn.Module):
    """
    The v3 architecture, but with the flatten head replaced by global average
    pooling. Retained so results remain comparable to the published baseline.
    """
    def __init__(self, n=2):
        super().__init__()
        ch, layers = 3, []
        for out in (8, 16, 32, 32):
            layers += [nn.Conv2d(ch, out, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2)]
            ch = out
        self.f = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, n)

    def forward(self, x):
        return self.fc(self.pool(self.f(x)).flatten(1))


def build_model(name, device):
    if name == "toycnn":
        return ToyCNN().to(device)
    m = models.resnet18(weights="IMAGENET1K_V1")
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m.to(device)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, y):
        ce = nn.functional.cross_entropy(logits, y, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


def colour_moments(paths):
    """Per-channel mean and std at low resolution: the documented shortcut."""
    X = np.empty((len(paths), 6), dtype=np.float64)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            a = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.float64) / 255
        X[i] = np.concatenate([a.mean((0, 1)), a.std((0, 1))])
    return X


# Arms that add generated images from a pool at a dose. The three differ only
# in WHICH pool, never in how it is drawn from, so they share one branch.
#   synthetic     legacy v3 pool -- contaminated, saw every holdout eye
#   synth_clean   this study's generator, trained on the 111 dev eyes only
#   pca_sampler   linear-manifold samples, ~1e4 params, no GPU
POOL_ARMS = ("synthetic", "synth_clean", "pca_sampler")

# Arms that add nothing and differ only in loss / sampling of the reals.
PLAIN_ARMS = ("real_only", "imbalanced", "oversample", "weighted", "focal")


def build_train_items(dev, arm, dose, pool, rng):
    """Returns (items, sample_weights_or_None, loss_name).

    `pool` is the arm's generated-image pool and is required for POOL_ARMS,
    ignored otherwise. An unrecognised arm RAISES: the previous version fell
    through to the plain-CE branch, so `synth_clean` and `pca_sampler` trained
    as `real_only` and reported a null difference that was an artifact of
    routing, not a result.
    """
    healthy = [(os.path.join(C.HEALTHY_DIR, f), 0) for f in sorted(dev["Healthy"])]
    rp = [(os.path.join(C.REAL_RP_DIR, f), 1) for f in sorted(dev["RP"])]

    if arm == "oversample":
        # Copying: duplicate real RP until the classes balance.
        k = len(healthy) - len(rp)
        extra = [rp[i] for i in rng.integers(0, len(rp), size=max(k, 0))]
        return healthy + rp + extra, None, "ce"

    if arm in POOL_ARMS:
        if not pool:
            raise ValueError(f"arm '{arm}' needs a non-empty image pool")
        # Dose is a multiple of the real RP count, resampled every run so no
        # single frozen synthetic subset drives the result.
        n_add = int(round(dose * len(rp)))
        n_add = min(n_add, len(pool))
        idx = rng.choice(len(pool), size=n_add, replace=False)
        extra = [(pool[i], 1) for i in idx]
        return healthy + rp + extra, None, "ce"

    if arm == "weighted":
        return healthy + rp, "balanced", "ce"
    if arm == "focal":
        return healthy + rp, None, "focal"
    if arm in ("real_only", "imbalanced"):
        # Same training set; two names because `real_only` is this study's
        # floor arm and `imbalanced` is the legacy sweep's untreated control.
        return healthy + rp, None, "ce"
    raise ValueError(f"unrouted arm '{arm}' -- add it to build_train_items")


def run_once(items, loss_name, weights, holdout_items, backbone, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(backbone, device)
    if loss_name == "focal":
        crit = FocalLoss()
    elif weights == "balanced":
        y = np.array([b for _, b in items])
        w = torch.tensor([len(y) / (2 * (y == 0).sum()),
                          len(y) / (2 * (y == 1).sum())],
                         dtype=torch.float32, device=device)
        crit = nn.CrossEntropyLoss(weight=w)
    else:
        crit = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    dl = DataLoader(ImgDS(items, TRAIN_TF), batch_size=C.BATCH, shuffle=True,
                    num_workers=0)

    model.train()
    for _ in range(C.EPOCHS):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in DataLoader(ImgDS(holdout_items, EVAL_TF), batch_size=C.BATCH):
            probs.append(torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def score(p, y, thr=0.5):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return {"auc": float(roc_auc_score(y, p)),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "precision": tp / max(tp + fp, 1),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "kappa": float(cohen_kappa_score(y, pred))}


def eye_keys(items, eye_labels):
    """One cluster id per holdout file: '<class>:<eye>'.

    Eye ids are numbered within a class, so 'RP:12' and 'Healthy:12' are
    different eyes. Prefixing keeps them apart.
    """
    out = []
    for p, y in items:
        cls = "RP" if y == 1 else "Healthy"
        out.append(f"{cls}:{eye_labels[cls][os.path.basename(p)]}")
    return np.array(out)


def by_eye(p, y, keys):
    """Mean predicted probability per eye.

    C.EVAL_UNIT is 'eye'. The holdout is 235 files but only 225 eyes -- 207
    healthy files are 197 eyes -- so scoring per file treats repeat photographs
    of one eye as independent evidence and understates uncertainty.
    """
    order = sorted(set(keys))
    pe = np.empty(len(order))
    ye = np.empty(len(order), dtype=int)
    for i, k in enumerate(order):
        m = keys == k
        lab = np.unique(y[m])
        assert lab.size == 1, f"eye {k} carries both labels"
        pe[i], ye[i] = p[m].mean(), lab[0]
    return pe, ye


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", default=C.LEGACY_SYNTH_DIR,
                    help="pool for the legacy 'synthetic' arm")
    ap.add_argument("--synth-clean-dir", default=None,
                    help="pool for 'synth_clean' (generate.py output)")
    ap.add_argument("--pca-dir", default=None,
                    help="pool for 'pca_sampler' (s04b output)")
    ap.add_argument("--tag", default="legacy_v3")
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "toycnn"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--doses", type=float, nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override C.EPOCHS; use a small value to smoke-test the harness")
    a = ap.parse_args()

    if a.epochs is not None:
        C.EPOCHS = a.epochs
    device = C.DEVICE if torch.backends.mps.is_available() else "cpu"
    with open(os.path.join(C.SPLITS, "dev.json")) as fh:
        dev = json.load(fh)
    with open(os.path.join(C.SPLITS, "holdout.json")) as fh:
        hold = json.load(fh)

    with open(os.path.join(C.SPLITS, "eye_labels.json")) as fh:
        eye_labels = json.load(fh)

    holdout_items = ([(os.path.join(C.HEALTHY_DIR, f), 0) for f in sorted(hold["Healthy"])]
                     + [(os.path.join(C.REAL_RP_DIR, f), 1) for f in sorted(hold["RP"])])
    hold_keys = eye_keys(holdout_items, eye_labels)

    def load_pool(d):
        if not d or not os.path.isdir(d):
            return []
        return sorted(os.path.join(d, f) for f in os.listdir(d)
                      if not f.startswith("."))

    # One pool per pool-arm. A pool-arm with no directory is skipped loudly
    # rather than silently degrading to real_only.
    pools = {"synthetic": load_pool(a.synth_dir),
             "synth_clean": load_pool(a.synth_clean_dir),
             "pca_sampler": load_pool(a.pca_dir)}
    doses = a.doses if a.doses is not None else C.DOSE_LEVELS
    arms = a.arms if a.arms is not None else C.ARMS
    missing = [x for x in arms if x in POOL_ARMS and not pools[x]]
    if missing:
        raise SystemExit(f"arms {missing} requested but their image pools are "
                         f"empty. Pass --synth-clean-dir / --pca-dir, or drop "
                         f"them from --arms.")

    print(f"device={device} backbone={a.backbone} tag={a.tag}")
    print(f"train on DEV: {len(dev['RP'])} RP + {len(dev['Healthy'])} healthy")
    print(f"eval on generator-blind HOLDOUT: {len(hold['RP'])} RP + "
          f"{len(hold['Healthy'])} healthy")
    print(f"holdout eyes: {len(set(hold_keys))} clusters over "
          f"{len(holdout_items)} files")
    for k, v in pools.items():
        if v:
            print(f"pool {k}: {len(v)}")
    print()

    res = {"tag": a.tag, "backbone": a.backbone, "synth_dir": a.synth_dir,
           "synth_clean_dir": a.synth_clean_dir, "pca_dir": a.pca_dir,
           "n_seeds": a.seeds, "device": device,
           "eval_unit": C.EVAL_UNIT, "n_holdout_eyes": len(set(hold_keys)),
           "runs": []}

    # Control arm: colour moments only. If the deep arms cannot beat this, the
    # benchmark is saturated and no augmentation comparison means anything.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    tr = ([(os.path.join(C.HEALTHY_DIR, f), 0) for f in sorted(dev["Healthy"])]
          + [(os.path.join(C.REAL_RP_DIR, f), 1) for f in sorted(dev["RP"])])
    Xtr = colour_moments([p for p, _ in tr]); ytr = np.array([y for _, y in tr])
    Xte = colour_moments([p for p, _ in holdout_items])
    yte = np.array([y for _, y in holdout_items])
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(Xtr, ytr)
    pc = lr.predict_proba(Xte)[:, 1]
    m = score(pc, yte)
    pe, ye = by_eye(pc, yte, hold_keys)
    me = score(pe, ye)
    res["runs"].append({"arm": "colour_lr", "dose": None, "seed": 0, **m,
                        "eye": me, "probs": pc.tolist()})
    print(f"colour_lr        AUC {m['auc']:.4f} image / {me['auc']:.4f} eye  "
          f"<- shortcut control")

    for arm in arms:
        arm_doses = doses if arm in POOL_ARMS else [None]
        for dose in arm_doses:
            for seed in range(a.seeds):
                rng = np.random.default_rng(C.SEED + 1000 * seed)
                items, w, loss = build_train_items(dev, arm, dose or 0.0,
                                                   pools.get(arm), rng)
                t0 = time.time()
                p, y = run_once(items, loss, w, holdout_items, a.backbone,
                                device, C.SEED + 1000 * seed)
                m = score(p, y)
                pe, ye = by_eye(p, y, hold_keys)
                me = score(pe, ye)
                res["runs"].append({"arm": arm, "dose": dose, "seed": seed,
                                    "n_train": len(items), **m, "eye": me,
                                    "probs": p.tolist()})
                print(f"{arm:11s} dose={str(dose):5s} seed={seed}  "
                      f"AUC {m['auc']:.4f} image / {me['auc']:.4f} eye  "
                      f"f1 {m['f1']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    p = os.path.join(C.OUT, f"s05_classifier_{a.tag}.json")
    with open(p, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
