# Evaluating StyleGAN2-ADA Augmentation for Retinitis Pigmentosa Fundus Images Using 111 Training Images

Code, splits, and result files for the paper accepted at the **IEEE BIBM 2026
Undergraduate and High School Symposium (UGHS)**.

Retinitis Pigmentosa (RP) is rare, so diagnostic models for it are trained on
very few images. This study fine-tunes StyleGAN2-ADA on 111 RP fundus images and
measures what the synthetic images are worth to a diagnostic classifier, on a
holdout of 28 RP eyes that the generator never saw.

## Findings

| condition (low-parameter classifier, 10x dose) | sensitivity | specificity | AUC | Brier |
|---|---|---|---|---|
| real-only | 0.521 | 1.000 | 0.9843 | 0.043 |
| oversampled | **0.921** | 0.951 | 0.9922 | 0.039 |
| clean-GAN | 0.629 | 0.998 | 0.9905 | 0.037 |
| contaminated-GAN | 0.757 | 0.996 | 0.9930 | 0.025 |

Eye-level, mean over five classifier seeds.

- **Synthetic images do not beat oversampling.** Eye-level AUC is near its
  ceiling for every condition and shows no difference (clean-GAN minus
  oversampled, +0.0004, p=0.94). On sensitivity, oversampling detects
  significantly more RP eyes at every dose choice (p <= 0.017).
- **A contaminated generator, trained on the holdout eyes on purpose, gains no
  advantage that survives correction** under the low-parameter classifier. The
  one significant cell (pretrained ResNet-18, 1.0x dose) rests on the rank of a
  single RP eye that every ResNet-18 condition misclassifies.
- **An ImageNet-pretrained ResNet-18 detects 26 of 28 RP eyes from the real
  images alone**, so augmentation of either kind has little left to improve.
- **With 28 RP eyes, no sensitivity difference below 0.21 can reach p < 0.05**,
  and AUC has under 0.01 of headroom. The null results are bounds.

## Design

```
139 RP images  -> 134 eyes (correlation clustering, complete linkage)
               -> 111 development images (106 eyes) + 28 generator-blind holdout eyes
1024 healthy   -> 817 development + 207 holdout images (197 eyes)

g_clean  StyleGAN2-ADA, FFHQ-256 init, trained on the 111 development images
g_leak   same recipe, 28 holdout + 83 development images (N matched at 111)

classifier conditions  real-only | oversampled | clean-GAN | contaminated-GAN
synthetic doses        0, 0.5, 1, 2, 5, 10x the 111 real RP images
backbones              MicroResNet (~77k params, from scratch) | ResNet-18 (ImageNet init)
```

Every metric is computed per **eye**, not per image. Contrasts use a BCa cluster
bootstrap over eyes and a paired permutation test that swaps condition labels
within an eye.

- **P1**: clean-GAN vs oversampled. Does the generator add more than duplicating
  the real RP images?
- **P2**: contaminated-GAN vs clean-GAN. Does exposure to the holdout help?

## Repository

| path | contents |
|---|---|
| `config.py` | every constant: paths, split sizes, doses, seeds, bootstrap counts |
| `scripts/s00_verify.py` | integrity gate: split disjointness, size matching, hash checks |
| `scripts/s01_unique_eyes.py` | eye identity by field-of-view correlation clustering |
| `scripts/s02_make_splits.py` | eye-disjoint, generator-blind split |
| `scripts/s04_prepare_gan_data.py` | builds the `clean_111` and `leak_111` training sets |
| `scripts/s05_classifier.py` | the classifier sweep: conditions x doses x seeds, eye-level scoring |
| `scripts/s07_power.py` | power simulation for the AUC contrast |
| `scripts/s09_swd.py` | sliced Wasserstein distance against a real-to-real anchor |
| `scripts/s10_figures.py`, `make_*_fig.py` | paper figures |
| `scripts/s11_gradcam*.py` | Grad-CAM over the classifier conditions |
| `scripts/s12_contrasts.py` | prespecified P1/P2 contrasts on AUC |
| `scripts/s13_matched_grid.py` | matched sample grid, one holdout eye per column |
| `scripts/s14_reviewer.py` | camera-ready analyses: secondary endpoints, fixed-dose P1, Holm across P2, calibration, stricter eye clustering |
| `scripts/stats_boot.py` | BCa cluster bootstrap and permutation test, with a self-test |
| `splits/` | eye labels, development / holdout split, folds |
| `outputs/` | every result file the paper's numbers come from |
| `figures/` | paper figures |
| `provenance/` | GPU pod scripts and logs for generator sampling and metrics |
| `docs/` | planning and audit history; superseded by the paper where they differ |

Result files behind the paper:

- `outputs/s05_classifier_micro5.json`: low-parameter classifier (Table I)
- `outputs/s05_classifier_main5.json`: ResNet-18
- `outputs/s05_classifier_toy5.json`: third backbone, robustness check only
- `outputs/s12_contrasts*.json`: prespecified contrasts
- `outputs/s14_reviewer.json`: Table II and Section III-D
- `outputs/s07_power_base099.json`: power at the observed AUC

## Reproducing

Data is not redistributed. Download the **Eye Disease Image Dataset**
(T. Khatun, Mendeley Data, V1, 2024) and place it so that
`Eye Disease Image Dataset/Original Dataset/Retinitis Pigmentosa/` and
`.../Healthy/` exist at the repository root.

```bash
pip install -r requirements.txt
python scripts/s01_unique_eyes.py
python scripts/s02_make_splits.py
python scripts/s00_verify.py
```

Generator training needs a GPU and
[NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
placed at `stylegan2-ada-pytorch-main/`. The recipe is in
`docs/RUNPOD_RECIPE.md`. Network snapshots (about 5.7 GB) are not included.

Everything after the classifier sweep runs on CPU from the stored per-eye
probabilities:

```bash
python scripts/s12_contrasts.py
python scripts/s14_reviewer.py --alt-eyes 0.985    # about 15 min
python scripts/s07_power.py --base 0.99 --deltas 0.0 0.002 0.004 0.006 0.009
```

`RUNBOOK.md` has the full step-by-step pipeline. Everything is deterministic
under `config.SEED = 42`.

## Citation

```bibtex
@inproceedings{pradeep2026rpstylegan,
  title     = {Evaluating {StyleGAN2-ADA} Augmentation for Retinitis Pigmentosa
               Fundus Images Using 111 Training Images},
  author    = {Pradeep, Puneeth},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine
               (BIBM) Workshops, Undergraduate and High School Symposium},
  year      = {2026}
}
```
