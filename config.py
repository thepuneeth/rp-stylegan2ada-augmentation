"""
Single source of truth for the variability study.

Research question
-----------------
The prior study (v3) balanced an RP dataset with 885 synthetic images and
reported AUC 0.89 -> 0.97, sensitivity 0-53% -> 88-97%. The gain was real.
This study decomposes it into two sources it can actually afford to measure:

    class balance          -- oversample vs real_only
    generator information  -- synth_clean vs oversample     <- PRIMARY

v3 merged its synthetic images into the dataset BEFORE the 5-fold split, so
roughly 177 of ~205 validation positives per fold were synthetic images from a
generator trained on the real positives. It also had no oversampling arm, so
"add synthetic" and "fix a 7.4:1 class ratio" were never separated.

Here every arm trains on the same 111 real RP images and is scored on 28 eyes
NO generator ever saw, so no result can be explained by the generator having
memorised the evaluation set.

Budget
------
ONE new generator, <=20 GPU-hours total. Every number in the GPU plan comes
from this project's own training-runs/*/stats.jsonl, not from an estimate:
21.7 sec/kimg on the gamma=5 / target=0.6 / freezed=0 recipe, FID 46.85 at
3,031 kimg. See docs/FINAL_PLAN.md and docs/RUNPOD_RECIPE.md.

NOTE on scope: "information-closed" applies to the GENERATOR STAGE only, and as
a hypothesis rather than a consequence of the data processing inequality.
Regularization, class balancing and induced invariance all improve
finite-sample risk without adding population information -- and the classifier
itself initialises from ImageNet weights.

Everything in this package is deterministic under SEED.
"""
import os

SEED = 42

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.abspath(__file__))
# This study is self-contained: the source images, the legacy v3 pool, the
# BalancedDataset cross-check and the StyleGAN2-ADA code all live under ROOT.
# It was split out of pigmentGAN/variability_study on 2026-08-25, where
# PROJECT pointed one level up at the shared parent project.
PROJECT = ROOT

# Pristine originals. NOT BalancedDataset/, which mixes real and synthetic.
ORIG = os.path.join(PROJECT, "Eye Disease Image Dataset", "Original Dataset")
REAL_RP_DIR = os.path.join(ORIG, "Retinitis Pigmentosa")
HEALTHY_DIR = os.path.join(ORIG, "Healthy")

# Legacy synthetic pool from the v3 paper (StyleGAN2-ADA, snapshot 201, psi=0.6,
# generator trained on ALL 139 real RP -> leaky by construction). Retained only
# as a historical comparison row; never used as a clean arm.
LEGACY_SYNTH_DIR = os.path.join(PROJECT, "generated_rp_images_0007_201_trunc_6")

OUT = os.path.join(ROOT, "outputs")
SPLITS = os.path.join(ROOT, "splits")
GAN_DATA = os.path.join(ROOT, "gan_data")
DOCS = os.path.join(ROOT, "docs")
for _d in (OUT, SPLITS, GAN_DATA, DOCS):
    os.makedirs(_d, exist_ok=True)

# ------------------------------------------------- duplicate / eye clustering
# Eye identity is established by correlation between field-of-view-cropped
# grayscale images, clustered with COMPLETE linkage.
#
# An earlier attempt used mean absolute difference on uncropped 128x128 RGB with
# single linkage; it merged 842 of the 1024 healthy images into one "eye". Two
# causes, both addressed here. (a) Uncropped low-resolution fundus distances are
# dominated by the shared black canvas and circular field of view, so the
# histogram has no duplicate mode to threshold. (b) Single linkage chains a~b,
# b~c into one cluster even when a and c are unrelated.
#
# A calibration compared four candidate features against five independently
# confirmed duplicate pairs. FOV-cropped grayscale correlation recovered all five with
# ZERO false positives above the weakest of them; high-pass admitted 1225 and
# gradient magnitude 391. The weakest confirmed pair sits at 0.9905 and the
# largest gap in the similarity ranking falls at 0.9986 -> 0.9905.
CORR_SZ = 256
CORR_THRESHOLD = 0.99
CORR_SWEEP = [0.995, 0.99, 0.985, 0.98]

# ------------------------------------------------------------- the split
# 139 real RP are partitioned EYE-DISJOINT into:
#   - a generator-blind holdout, which no GAN ever sees, and
#   - a development pool used for GAN training and classifier cross-validation.
# Target sizes are approximate because whole eyes move together.
HOLDOUT_RP_TARGET = 28
HOLDOUT_HEALTHY_TARGET = 206      # keeps holdout prevalence ~= cohort prevalence

# Evaluate at EYE level. The holdout's 207 Healthy files represent 197 Healthy
# eyes; treating files as independent controls understates uncertainty. Average
# the predicted probability within an eye, then bootstrap over eye clusters.
EVAL_UNIT = "eye"
BOOTSTRAP_N = 2000

# ------------------------------------------------------- classifier protocol
N_FOLDS = 5
N_REPEATS = 5
IMG_SZ = 256
BATCH = 32
EPOCHS = 30
LR = 1e-4

# Synthetic dose, as a multiple of the real RP count in the training split.
# 1.0x is the exact-balance point the v3 paper tested; it confounded dose with
# class balance. Sweeping past it separates the two.
DOSE_LEVELS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

# Truncation. v3 used 0.6 only, which suppresses the very diversity the study
# is trying to measure. 1.0 is untruncated.
PSI_LEVELS = [0.7, 1.0]

# The arms. Every arm trains on the same 111 real RP and is scored on the same
# 28 generator-blind eyes.
#
#   arm           training set addition                  role
#   ------------  -------------------------------------  --------------------
#   real_only     --                                     floor
#   oversample    minority duplicated to balance         balance, no new pixels
#   synth_clean   StyleGAN2-ADA (~2.5e7 params), the ONE   the generator as
#                 new run, ffhq256 -> 111 dev eyes       normally used
#
# Prespecified contrasts:
#   P1  synth_clean vs oversample  -- what the generator adds beyond
#                                     rebalancing.  THE PRIMARY CONTRAST,
#                                     and the only confirmatory test.
#   S1  oversample vs real_only    -- class balance alone.        descriptive
ARMS = ["real_only", "oversample", "synth_clean"]

# Historical row, not an arm. The v3 generator saw all 139 real RP including
# every evaluation eye. Labelled contaminated wherever it appears; supports no
# clean claim.
LEGACY_ROW = "legacy_v3"
LEGACY_ARMS = ["imbalanced", "weighted", "oversample", "focal", "synthetic"]

# ------------------------------------------------------------ generator plan
# ONE generator, n=1. Sampling thousands of images from one checkpoint
# replicates SAMPLING, not TRAINING, so this study cannot separate a real
# effect from a favourable initialization or trajectory. The claim scopes to
# "for a generator trained under the published protocol", stated in
# limitations rather than concealed by a large sample size. Generator-level
# replication is the first thing a follow-up should buy.
#
# Cut, and why (docs/FINAL_PLAN.md section 3):
#   synth_scratch  -- StyleGAN2-ADA from random init on 111 images at 256px
#                     produces noise at any budget reachable in 20 GPU-hours.
#   stage-2 corpus -- not what was achieved; v3 went ffhq256 -> RP directly.
#   synth_leak     -- needs a second matched generator (28 holdout + 83 dev,
#                     N matched at 111). No budget. gan_data/leak_111.zip is
#                     already built and kept for the follow-up study.
GAN_SEEDS = [0]

# Measured on this project's own runs, not estimated. gamma=5 / ADA target=0.6
# / freezed=0 ran at 21.7 sec/kimg; the later freezed=4/5 phases dropped to
# ~28 and are cut. FID50k_full by cumulative kimg was 49.58 @ 1983,
# 46.85 @ 3031, 41.69 @ 3806 (best), 43.01 @ 4293 (worse).
# 2026-08-23: 21.7 is NOT reachable on any currently rentable card -- it came
# from the prior project's hardware. Both figures below are measured here, 24
# kimg each, --batch=64, identical flags:
#     RTX 4090        10.7 sec/kimg   ->  3000 kimg = 8.9 h
#     RTX 4000 Ada    30.9 sec/kimg   ->  3000 kimg = 25.8 h
SEC_PER_KIMG = 10.7
GPU_HOUR_CEILING = 20.0
TRAIN_KIMG = 3000
# 400 kimg, not 200. --snap sets image snapshots, network snapshots AND metric
# evaluations from one number (train.py:86-87), so halving it halves FID cost.
# Leaves 4 selection candidates at or above SELECT_MIN_KIMG, which is enough.
SNAPSHOT_EVERY_KIMG = 400

# Snapshot selection, prespecified. The prior project's best snapshot was NOT
# its last, so a fixed-kimg endpoint would sometimes hand over the worse
# generator. Selection runs offline on downloaded snapshots: lowest
# kid50k_full among snapshots >= SELECT_MIN_KIMG, ties break to the earlier
# snapshot.
#
# KID, NOT FID. Both ship with StyleGAN2-ADA for the cost of one flag. FID is
# a BIASED estimator whose bias grows as the reference set shrinks, and this
# reference set is 139 images. KID is the unbiased small-sample estimator of
# the same discrepancy, which makes it the correct headline at this n. The
# prior runs logged only fid50k_full; that was a mistake, not a choice.
#
#   --metrics=fid50k_full,kid50k_full,pr50k3_full
#
# pr50k3_full (Kynkaanniemi precision/recall) covers fidelity and diversity.
# Every metric in NVIDIA's metrics/ module references the TRAINING set, so it is
# a within-run signal only and cannot detect memorisation of the holdout.
SELECT_METRIC = "kid50k_full"
TRAIN_METRICS = "fid50k_full,kid50k_full,pr50k3_full"
SELECT_MIN_KIMG = 2000

# WITHDRAWN 2026-08-23. The surplus rule ("if the first tick beats
# SEC_PER_KIMG, raise kimg toward 3800") was assistant-authored, never the
# owner's intent, and it is now actively harmful: it keys off a stale 21.7
# baseline, so a 4090's 10.7 fires it automatically, and 3800 kimg costs $8.36
# against a balance under $8. It was written when GPU-HOURS were the binding
# constraint. DOLLARS bind now. Do not reinstate without a budget that covers
# it.
TRAIN_KIMG_CAP = TRAIN_KIMG  # no surplus extension

# FID is a MONITORING SIGNAL, not an endpoint. It is computed against only 139
# reals: valid as a within-run trend, meaningless as an absolute number. Never
# compare it to published FIDs on 70k-image datasets.

DEVICE = "mps"

# DataLoader worker processes. 0 means decode in the main process, which on the
# local box measured 18.1 ms per image with the GPU idle behind it. Set from the
# environment so one pod recipe fits any core count: S05_WORKERS=8 etc.
NUM_WORKERS = int(os.environ.get("S05_WORKERS", "0"))
