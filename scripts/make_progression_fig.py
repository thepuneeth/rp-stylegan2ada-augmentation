"""Training-progression figure for the transfer-learning fine-tuning.

Columns are training snapshots (kimg), rows are fixed grid positions, so each
row follows the same latent vector from the FFHQ initialisation onwards.
Tiles come from the top row of the StyleGAN2-ADA fakes*.png grids of the
pigmentGAN runs, which sampled image snapshots every 40 kimg and therefore
capture the face-to-fundus transition. Run 00005 resumes directly from run
00004's 282 kimg snapshot and shares its random seed, so the latent grid is
continuous across the two runs.

The source grids are truncated on disk (the PNG streams end part-way through
the IDAT chunks), but PNG rows decode top-down and only the first grid row is
needed, so LOAD_TRUNCATED_IMAGES recovers exactly the tiles used here.
"""
import os
from PIL import Image, ImageFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "pigmentGAN",
                    "training-runs")
A = os.path.join(BASE, "00004-rp_dataset_256-mirror-paper256-gamma5-"
                       "kimg5000-batch64-ada-resumeffhq256")
B = os.path.join(BASE, "00005-rp_dataset_256-mirror-paper256-gamma5-"
                       "kimg5000-batch64-ada-resumecustom")
OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "fig_progression.png")

TILE = 256
COLS = [3, 8, 11]
ROWS = [(A, "fakes_init.png", "0 kimg"),
        (A, "fakes000040.png", "40 kimg"),
        (A, "fakes000121.png", "121 kimg"),
        (A, "fakes000282.png", "282 kimg"),
        (B, "fakes000806.png", "806 kimg"),
        (B, "fakes001612.png", "1612 kimg"),
        (B, "fakes002741.png", "2741 kimg")]


def tile(run, fname, col):
    im = Image.open(os.path.join(run, fname)).convert("RGB")
    return im.crop((col * TILE, 0, (col + 1) * TILE, TILE))


fig, axes = plt.subplots(len(COLS), len(ROWS),
                         figsize=(3.4, 3.4 / len(ROWS) * len(COLS) + 0.16))
fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.005,
                    wspace=0.02, hspace=0.02)

for c, (run, fname, label) in enumerate(ROWS):
    for r, col in enumerate(COLS):
        ax = axes[r][c]
        ax.imshow(tile(run, fname, col))
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
    axes[0][c].set_title(label, fontsize=6, pad=3)

# arrow across the top spanning the snapshot sequence
first = axes[0][0].get_position()
last = axes[0][-1].get_position()
y = 0.965
fig.patches.append(matplotlib.patches.FancyArrow(
    first.x0, y, last.x1 - first.x0, 0,
    transform=fig.transFigure, figure=fig,
    width=0.004, head_width=0.030, head_length=0.012,
    length_includes_head=True, color="0.25"))

fig.savefig(OUT, dpi=400)
print("wrote", os.path.normpath(OUT))
