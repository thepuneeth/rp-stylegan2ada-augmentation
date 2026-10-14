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

Why "oversample" is the baseline and not "imbalanced"
-----------------------------------------------------
Oversampling duplicates existing minority images and adds zero new information.
It is the exact operationalisation of "copying the existing RP images". A
generator that only reproduces its training distribution can at best match it.
Beating it is the whole claim, so beating "imbalanced" proves nothing.

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
from concurrent.futures import ThreadPoolExecutor
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

# The original research's train transform, verbatim from
# pigmentGAN/synthesized_dataset_CNN.ipynb. Retained unmodified -- including the
# constant -15 degree rotation and the one-sided contrast, both of which the
# current TRAIN_TF deliberately corrected -- because the question is whether the
# published attention result depends on them. shear=40 with scale (0.75, 1.2)
# distorts the frame every epoch, which is the candidate reason the border was
# not usable as a shortcut there.
ORIG_TRAIN_TF = transforms.Compose([
    transforms.Resize((C.IMG_SZ, C.IMG_SZ)),
    transforms.RandomAffine((-15, -15), translate=(0.05, 0.1),
                            scale=(0.75, 1.2), shear=40),
    transforms.ColorJitter(contrast=(0.2, 1.0)),
    transforms.Resize((C.IMG_SZ, C.IMG_SZ)),
    transforms.ToTensor(), NORM,
])

AUGS = {"current": None, "orig": ORIG_TRAIN_TF}   # None -> module TRAIN_TF


# Decode-once cache, path -> 256x256x3 uint8.
#
# The real fundus sources are 2004x1690 JPEGs, so every access paid a 20 ms
# decode to produce a 256x256 tensor -- 30 epochs x 70 fits of the same
# handful of images. Resize is the first step of both transforms and Pillow
# returns a plain copy when the requested size already matches, so caching the
# resized array and letting the transform resize it a second time is
# bit-identical (verified: max abs diff 0) and about 300x cheaper.
#
# Prefilled in the parent before any DataLoader is built, so forked workers
# inherit it copy-on-write instead of each decoding its own copy.
_CACHE = {}


def prefill_cache(paths, workers=16):
    todo = [p for p in dict.fromkeys(paths) if p not in _CACHE]
    if not todo:
        return
    rs = transforms.Resize((C.IMG_SZ, C.IMG_SZ))

    def load(p):
        with Image.open(p) as im:
            return p, np.asarray(rs(im.convert("RGB")))

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for p, a in ex.map(load, todo):
            _CACHE[p] = a
    mb = sum(a.nbytes for a in _CACHE.values()) / 1e6
    print(f"cached {len(todo)} images in {time.time() - t0:.0f}s "
          f"({mb:.0f} MB resident)", flush=True)


class ImgDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        a = _CACHE.get(p)
        if a is not None:
            return self.tf(Image.fromarray(a)), y
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


class OrigCNN(nn.Module):
    """
    The architecture from the original research notebooks, verbatim:
    pigmentGAN/{original,synthesized}_dataset_CNN.ipynb.

    Four Conv-ReLU-MaxPool blocks at 4/8/16/32 channels and a POSITION-SENSITIVE
    flatten head, Linear(32*16*16, 2). ToyCNN differs on both counts -- wider
    early channels and global average pooling -- so it is not the model that
    produced the published Grad-CAMs. Kept separate rather than folded into
    ToyCNN so the two remain independently runnable.
    """

    def __init__(self, n=2):
        super().__init__()
        ch, layers = 3, []
        for out in (4, 8, 16, 32):
            layers += [nn.Conv2d(ch, out, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)]
            ch = out
        self.f = nn.Sequential(*layers)
        self.fc = nn.Linear(32 * 16 * 16, n)

    def forward(self, x):
        return self.fc(self.f(x).flatten(1))


class _Basic(nn.Module):
    """A pre-activation-free BasicBlock, ResNet-18's, at a fraction of the width."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(c_out)
        self.c2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(c_out)
        self.skip = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm2d(c_out))

    def forward(self, x):
        h = torch.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return torch.relu(h + self.skip(x))


class MicroResNet(nn.Module):
    """
    A residual network held under 100k parameters -- about 77k, against
    ResNet-18's 11.2M and ToyCNN's 15k.

    ToyCNN is a plain four-conv stack with no normalisation, and at 15k
    parameters it settles on whatever the crudest class-correlated cue in the
    frame happens to be. This keeps the "deliberately weakened" intent, since it
    remains two orders of magnitude below the strong backbone, while restoring
    the two things ToyCNN lacks: residual connections, so depth is usable, and
    BatchNorm, whose absence is the likeliest source of ToyCNN's seed-to-seed
    swing.
    """

    def __init__(self, n=2, widths=(16, 32, 64)):
        super().__init__()
        w0 = widths[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, w0, 3, 2, 1, bias=False),
            nn.BatchNorm2d(w0), nn.ReLU(),
            nn.MaxPool2d(2))
        blocks, c = [], w0
        for i, w in enumerate(widths):
            blocks.append(_Basic(c, w, stride=1 if i == 0 else 2))
            c = w
        self.f = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c, n)

    def forward(self, x):
        return self.fc(self.pool(self.f(self.stem(x))).flatten(1))


def build_model(name, device):
    if name == "toycnn":
        return ToyCNN().to(device)
    if name == "microresnet":
        return MicroResNet().to(device)
    if name == "origcnn":
        return OrigCNN().to(device)
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


# Arms that add generated images from a pool at a dose. The three differ only
# in WHICH pool, never in how it is drawn from, so they share one branch.
#   synthetic     legacy v3 pool -- contaminated, saw every holdout eye
#   synth_clean   this study's generator, trained on the 111 dev eyes only
#   synth_leak    the size-matched CONTAMINATED generator (trained on the 111
#                 dev eyes with 28 of them swapped for holdout eyes). Routed
#                 here so the arm is available, but it is NOT in config.ARMS:
#                 enable it per run with --arms ... synth_leak --synth-leak-dir.
POOL_ARMS = ("synthetic", "synth_clean", "synth_leak")

# Arms that add nothing and differ only in loss / sampling of the reals.
PLAIN_ARMS = ("real_only", "imbalanced", "oversample", "weighted", "focal")


def load_pool(d):
    """Sorted image paths in a pool directory; empty list if there is none.

    Module level so s11_gradcam can build the same pools s05 does, rather than
    keeping a second copy that could drift out of step.
    """
    if not d or not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if not f.startswith("."))


def build_train_items(dev, arm, dose, pool, rng):
    """Returns (items, sample_weights_or_None, loss_name).

    `pool` is the arm's generated-image pool and is required for POOL_ARMS,
    ignored otherwise. An unrecognised arm RAISES: the previous version fell
    through to the plain-CE branch, so `synth_clean` trained
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


def pick_device():
    """cuda > mps > cpu.

    The previous form was `C.DEVICE if torch.backends.mps.is_available() else
    "cpu"`, which silently selected CPU on a CUDA box -- the sweep would have
    run, slowly, and reported device="cpu" in the output JSON.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return C.DEVICE
    return "cpu"


def loader_kw(device, seed, train=True):
    """DataLoader settings.

    num_workers was 0, so JPEG decode and augmentation ran serially in the main
    process and the GPU waited on one core: a measured 18.1 ms per image, of
    which most was decode. Parallel decode is the single change that makes this
    sweep affordable.

    Note this shifts the augmentation RNG. With workers, each worker seeds its
    own torch RNG rather than drawing from the parent, so the sequence of random
    crops and flips differs from a num_workers=0 run. The draws are still a
    valid sample and still reproducible -- the loader generator is pinned to C.SEED + seed below -- but fits are NOT bit-identical to the earlier serial run. The
    endpoint averages over C seeds, so this changes the noise, not the estimate.
    """
    if not C.NUM_WORKERS:
        return {"num_workers": 0}
    kw = {"num_workers": C.NUM_WORKERS,
          "pin_memory": device == "cuda",
          "prefetch_factor": 4}
    if train:
        # persistent_workers ONLY on the training loader, which is reused across
        # C.EPOCHS. Setting it on the short-lived eval loader too leaked worker
        # processes: run_once builds fresh loaders on every fit, and at ~100
        # fits x 8 workers the sweep died with "[Errno 24] Too many open files".
        kw["persistent_workers"] = True
        # A seeded generator is enough: torch derives each worker's seed as
        # base_seed + worker_id from this generator, so worker RNG is already
        # deterministic. An explicit worker_init_fn would be redundant, and a
        # lambda one is not picklable under the spawn start method.
        g = torch.Generator(); g.manual_seed(C.SEED + seed)
        kw["generator"] = g
    return kw


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
                    **loader_kw(device, seed))

    model.train()
    for _ in range(C.EPOCHS):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

    # Tear the training workers down before eval. Without this the loader's
    # worker processes survive until GC, and across a 100-fit sweep the file
    # descriptors accumulate until the process hits its limit.
    if hasattr(dl, "_iterator") and dl._iterator is not None:
        dl._iterator._shutdown_workers()
    del dl

    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in DataLoader(ImgDS(holdout_items, EVAL_TF), batch_size=C.BATCH,
                               **loader_kw(device, seed, train=False)):
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
    ap.add_argument("--synth-leak-dir", default=None,
                    help="pool for 'synth_leak' (the contaminated generator)")
    ap.add_argument("--tag", default="legacy_v3")
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "toycnn", "microresnet"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--doses", type=float, nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override C.EPOCHS; use a small value to smoke-test the harness")
    a = ap.parse_args()

    if a.epochs is not None:
        C.EPOCHS = a.epochs
    device = pick_device()
    with open(os.path.join(C.SPLITS, "dev.json")) as fh:
        dev = json.load(fh)
    with open(os.path.join(C.SPLITS, "holdout.json")) as fh:
        hold = json.load(fh)

    with open(os.path.join(C.SPLITS, "eye_labels.json")) as fh:
        eye_labels = json.load(fh)

    holdout_items = ([(os.path.join(C.HEALTHY_DIR, f), 0) for f in sorted(hold["Healthy"])]
                     + [(os.path.join(C.REAL_RP_DIR, f), 1) for f in sorted(hold["RP"])])
    hold_keys = eye_keys(holdout_items, eye_labels)

    # One pool per pool-arm. A pool-arm with no directory is skipped loudly
    # rather than silently degrading to real_only.
    pools = {"synthetic": load_pool(a.synth_dir),
             "synth_clean": load_pool(a.synth_clean_dir),
             "synth_leak": load_pool(a.synth_leak_dir)}
    doses = a.doses if a.doses is not None else C.DOSE_LEVELS
    arms = a.arms if a.arms is not None else C.ARMS
    missing = [x for x in arms if x in POOL_ARMS and not pools[x]]
    if missing:
        raise SystemExit(f"arms {missing} requested but their image pools are "
                         f"empty. Pass --synth-clean-dir / --synth-leak-dir, or "
                         f"drop them from --arms.")

    print(f"device={device} backbone={a.backbone} tag={a.tag}")
    print(f"train on DEV: {len(dev['RP'])} RP + {len(dev['Healthy'])} healthy")
    print(f"eval on generator-blind HOLDOUT: {len(hold['RP'])} RP + "
          f"{len(hold['Healthy'])} healthy")
    print(f"holdout eyes: {len(set(hold_keys))} clusters over "
          f"{len(holdout_items)} files")
    for k, v in pools.items():
        if v:
            print(f"pool {k}: {len(v)}")

    # Decode every image the sweep can touch, once, before any worker forks.
    prefill_cache([p for p, _ in holdout_items]
                  + [os.path.join(C.HEALTHY_DIR, f) for f in dev["Healthy"]]
                  + [os.path.join(C.REAL_RP_DIR, f) for f in dev["RP"]]
                  + [p for arm in arms if arm in POOL_ARMS
                     for p in (pools.get(arm) or [])],
                  workers=int(os.environ.get("S05_WORKERS", 16)) or 16)
    print()

    res = {"tag": a.tag, "backbone": a.backbone, "synth_dir": a.synth_dir,
           "synth_clean_dir": a.synth_clean_dir,
           "n_seeds": a.seeds, "device": device,
           "eval_unit": C.EVAL_UNIT, "n_holdout_eyes": len(set(hold_keys)),
           "runs": []}

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
