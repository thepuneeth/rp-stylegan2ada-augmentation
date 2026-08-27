"""
Paper figures, built from whatever outputs already exist.

Every figure is skipped with a message rather than crashing if its inputs are
missing, so this runs usefully before s05 and s09 have produced anything.

    ../venv/bin/python scripts/s10_figures.py            # all available
    ../venv/bin/python scripts/s10_figures.py --only curves

Figures:
  curves     training trajectories for both arms + the ADA augment probability.
             This is the F3 figure: the contaminated arm collapses at 806 kimg
             where the matched clean arm improves.
  samples    a grid of real / synth_clean / synth_leak images.
  crossscore the F1 2x2, each generator against both reference sets.
  swd        SWD against its noise floor and the real-to-real anchor.
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import config as C

FIGDIR = os.path.join(C.ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)
ARMC = {"g_clean": "#1f77b4", "g_leak": "#d62728"}


def _save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png and .svg")


def _metric_series(run, metric):
    """(kimg, value) from a run's metric jsonl, plus the s08 backfill."""
    p = os.path.join(C.ROOT, "runs", run, f"metric-{metric}.jsonl")
    pts = {}
    if os.path.isfile(p):
        for line in open(p):
            d = json.loads(line)
            kimg = int(os.path.basename(d["snapshot_pkl"])
                       .replace("network-snapshot-", "").replace(".pkl", ""))
            v = d["results"]
            pts[kimg] = v.get(metric, v.get(f"{metric}_precision"))
    bf = os.path.join(C.OUT, "s08_pod_metrics_2026-08-25.json")
    if run == "g_clean" and os.path.isfile(bf):
        d = json.load(open(bf)).get("backfill_g_clean_vs_clean_111", {})
        key = "precision" if metric.startswith("pr50k3") else metric
        for k, rec in d.items():
            if key in rec:
                pts.setdefault(int(k), rec[key])
    return np.array(sorted(pts)), np.array([pts[k] for k in sorted(pts)])


def fig_curves():
    rows = [("kid50k_full", "KID (lower better)"),
            ("fid50k_full", "FID (lower better)"),
            ("pr50k3_full", "Precision (higher better)")]
    fig, ax = plt.subplots(4, 1, figsize=(7.2, 10), sharex=True)
    any_data = False
    for i, (m, lab) in enumerate(rows):
        for run in ("g_clean", "g_leak"):
            k, v = _metric_series(run, m)
            if len(k) == 0:
                continue
            any_data = True
            ax[i].plot(k, v, "o-", color=ARMC[run], label=run, ms=4, lw=1.6)
        ax[i].set_ylabel(lab); ax[i].grid(alpha=.3)
        ax[i].axvspan(673, 830, color="grey", alpha=.15, lw=0)
        # kimg 0 is the untrained FFHQ init (FID ~394) and compresses everything
        # after it into a flat line. Log scale keeps the 806 excursion readable.
        if m in ("kid50k_full", "fid50k_full"):
            ax[i].set_yscale("log")
    # ADA augment probability, the mechanism behind the episode
    for run in ("g_clean", "g_leak"):
        p = os.path.join(C.ROOT, "runs", run, "stats.jsonl")
        if not os.path.isfile(p):
            continue
        kk, aa = [], []
        for line in open(p):
            d = json.loads(line)
            if "Progress/augment" in d and "Progress/kimg" in d:
                kk.append(d["Progress/kimg"]["mean"])
                aa.append(d["Progress/augment"]["mean"])
        if kk:
            any_data = True
            ax[3].plot(kk, aa, color=ARMC[run], label=run, lw=1.3)
    ax[3].axvspan(673, 830, color="grey", alpha=.15, lw=0)
    ax[3].set_ylabel("ADA augment p"); ax[3].set_xlabel("kimg"); ax[3].grid(alpha=.3)
    ax[0].legend(frameon=False)
    ax[0].set_title("Matched runs differing only in --data.\n"
                    "Shaded: kimg 673-830, where g_leak destabilised.",
                    fontsize=9, loc="left")
    if not any_data:
        print("  skip curves: no metric jsonl found"); plt.close(fig); return
    _save(fig, "fig_training_curves")


def fig_samples(n=6, seed=C.SEED):
    srcs = [("Real RP (dev)", C.REAL_RP_DIR),
            ("synth_clean", os.path.join(C.GAN_DATA, "synth_clean_psi1.0")),
            ("synth_leak", os.path.join(C.GAN_DATA, "synth_leak_psi1.0"))]
    srcs = [(t, d) for t, d in srcs if os.path.isdir(d)]
    if not srcs:
        print("  skip samples: no pools found"); return
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(len(srcs), n, figsize=(1.5 * n, 1.55 * len(srcs)))
    ax = np.atleast_2d(ax)
    for r, (title, d) in enumerate(srcs):
        fs = sorted(f for f in os.listdir(d)
                    if f.lower().endswith((".png", ".jpg", ".jpeg")))
        pick = rng.choice(len(fs), size=min(n, len(fs)), replace=False)
        for c in range(n):
            ax[r, c].axis("off")
            if c < len(pick):
                with Image.open(os.path.join(d, fs[pick[c]])) as im:
                    ax[r, c].imshow(im.convert("RGB").resize((256, 256)))
        ax[r, 0].set_ylabel(title)
        ax[r, 0].axis("on"); ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
        ax[r, 0].set_ylabel(title, fontsize=8)
    _save(fig, "fig_sample_grid")


def fig_crossscore():
    p = os.path.join(C.OUT, "s08_pod_metrics_2026-08-25.json")
    if not os.path.isfile(p):
        print("  skip crossscore: s08 json missing"); return
    cs = json.load(open(p))["cross_score_f1"]
    mets = [("fid50k_full", "FID"), ("kid50k_full", "KID"), ("precision", "Precision")]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for i, (mk, lab) in enumerate(mets):
        vals, labels, colors = [], [], []
        for gen, key in (("clean-002822", "g_clean"), ("leak-003000", "g_leak")):
            for ref in ("clean_111", "leak_111"):
                k = [kk for kk in cs[gen] if kk.startswith(f"vs_{ref}")][0]
                vals.append(cs[gen][k][mk]); labels.append(f"{key}\nvs {ref}")
                colors.append(ARMC[key])
        ax[i].bar(range(4), vals, color=colors, alpha=.85)
        ax[i].set_xticks(range(4)); ax[i].set_xticklabels(labels, fontsize=7)
        ax[i].set_title(lab, fontsize=10); ax[i].grid(alpha=.3, axis="y")
        lo, hi = min(vals), max(vals); pad = (hi - lo) * .6 + 1e-9
        ax[i].set_ylim(lo - pad, hi + pad)          # margins are tiny: show that
    fig.suptitle("F1: each generator against BOTH reference sets. "
                 "Note the axis ranges -- all separations are small.",
                 fontsize=9)
    _save(fig, "fig_crossscore_2x2")


def fig_swd():
    p, pa = os.path.join(C.OUT, "s09_swd.json"), os.path.join(C.OUT, "s09_swd_anchors.json")
    if not (os.path.isfile(p) and os.path.isfile(pa)):
        print("  skip swd: s09 json missing"); return
    d, anc = json.load(open(p)), json.load(open(pa))
    floor = anc["floor_clean111"]["mean"]
    rr = anc["real_vs_real_clean111_leak111"]["mean"]
    names = list(d["results"])
    vals = [d["results"][n]["clean_111"]["mean"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    cols = [ARMC["g_leak"] if "leak" in n else ARMC["g_clean"] for n in names]
    ax.bar(range(len(names)), vals, color=cols, alpha=.85)
    ax.axhline(floor, color="k", ls="--", lw=1.2, label=f"noise floor {floor:.4f}")
    ax.axhline(rr, color="grey", ls=":", lw=1.4, label=f"real vs real {rr:.4f}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("synth_", "") for n in names], fontsize=8)
    ax.set_ylabel("SWD vs clean_111"); ax.set_ylim(floor * .97, max(vals) * 1.02)
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.3, axis="y")
    ax.set_title("SWD separates synthetic from real, but not the two generators.",
                 fontsize=9, loc="left")
    _save(fig, "fig_swd")


FIGS = {"curves": fig_curves, "samples": fig_samples,
        "crossscore": fig_crossscore, "swd": fig_swd}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(FIGS))
    a = ap.parse_args()
    for name in (a.only or list(FIGS)):
        print(f"[{name}]")
        FIGS[name]()
