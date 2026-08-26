> **SUPERSEDED.** The stage-2 fundus corpus pretrain was dropped in
> `FINAL_PLAN.md` §3 — v3 went ffhq256 -> RP directly, so a corpus stage is
> not what the study reproduces. Kept for the licensing research, which
> stays valid if the stage is ever revived.

# Stage-2 pretraining corpus — licensing and availability

> **STATUS 2026-08-02 — NOT USED IN THE CURRENT STUDY.** The stage-2 corpus
> pretrain was cut when the plan was rebuilt to a 20-GPU-hour ceiling
> (`AUDIT_LOG.md` 2026-08-02). The published v3 result went FFHQ256 → RP
> directly, and this study reproduces what was practically achieved rather than
> adding an unmeasured stage. The licensing and provenance work below stands and
> is retained for the follow-up study; the 7 missing Mendeley classes do **not**
> need to be downloaded for the BIBM submission.

## Why a stage-2 corpus at all

The v3 generator transferred **FFHQ256 → 139 RP**. FFHQ is human faces. Faces
and retinal fundus share almost no low-level statistics: no vasculature, no
optic disc, no circular field of view, no illumination falloff from a ring
flash. Most of the 4,272 kimg of training was therefore spent un-learning
faces before any RP-specific structure could be acquired. The result was
FID 40.97, where fundus at 256×256 should reach 10–20.

Staged transfer inserts a middle step:

```
FFHQ256  →  large fundus corpus  →  111 RP
```

Stage 2 teaches *what a fundus is*. Stage 3 then only has to learn *what RP
does to one* — bone-spicule pigmentation, arteriolar attenuation, waxy disc
pallor. 111 images is ample for a delta and nowhere near enough for a whole
imaging modality.

This matters directly for the study's question. The measured weakness of the
v3 generator is **recall 0.297** in Inception space: good fidelity
(precision 0.938) but coverage of only ~30% of the real RP manifold. A
generator that never learned the modality cannot span it. Stage 2 targets
exactly that number.

---

## Primary corpus — AVAILABLE, already licensed

**Eye Disease Image Dataset**, Mendeley Data, DOI `10.17632/s9bfhswzjb.1`.

- **License: CC BY 4.0** (verified at source, 2026-07-30). Permits use,
  redistribution and derivatives with attribution. No non-commercial or
  no-derivatives restriction.
- **This is the project's existing data source.** The 139 RP and 1,024 healthy
  images already in use came from here. No new access agreement, no new ethics
  question, no credentialing.
- 5,335 original images from Anwara Hamida Eye Hospital and BNS Zahrul Haque
  Eye Hospital, Faridpur district, Bangladesh.

**Ten classes.** Only two were ever downloaded:

| class | status |
|---|---|
| Healthy | downloaded — 1,024 |
| Retinitis Pigmentosa | downloaded — 139 |
| Diabetic Retinopathy | **not downloaded** |
| Glaucoma | **not downloaded** |
| Macular Scar | **not downloaded** |
| Optic Disc Edema | **not downloaded** |
| Central Serous Chorioretinopathy | **not downloaded** |
| Retinal Detachment | **not downloaded** |
| Myopia | **not downloaded** |
| Pterygium | **EXCLUDE** — anterior segment, not fundus |

Roughly **4,172 additional fundus images** are therefore available under a
license this project has already accepted. Retrieving them is a straight
download from Mendeley, no authentication, no scraping.

### Action required

Download the eight missing fundus classes from
`https://data.mendeley.com/datasets/s9bfhswzjb/1`, excluding Pterygium.
Target corpus for stage 2:

```
~4,172 pathology fundus  +  817 DEV healthy  ≈  4,989 images
```

The 817 healthy figure is DEV-only. **Holdout healthy images are excluded from
the pretraining corpus**, exactly as holdout RP is — a generator that had seen
holdout negatives could bias the negative class as easily as the positive one.
Holdout RP is excluded from every stage.

---

## The honest limitation, stated up front

This corpus comes from the **same two hospitals and the same camera** as the
target data. A source-shortcut audit found every image in the cohort shares
JPEG quantization hash `56535cd0d590` and resolution 2004×1690, and that global
colour moments alone separate RP from healthy at **AUC 0.947**.

Same-source pretraining will therefore teach fundus anatomy but will **not**
break the source shortcut — it may reinforce it. Two consequences, both of
which the study must state rather than hide:

1. Stage 2 is justified as *modality learning*, not as domain diversification.
2. The `colour_lr` control arm in `s05` exists precisely to detect this. If
   every deep arm lands near AUC 0.947, the benchmark is saturated and no
   augmentation difference is interpretable regardless of generator quality.

A genuinely multi-source corpus would be stronger. See below.

---

## Secondary corpora — would strengthen, need verification

Not required for the experiment to run. Pursue in parallel; each needs its
license read at source before any download, per the project's data-ethics rule.

| corpus | approx. size | status |
|---|---|---|
| ODIR-5K | ~5,000 pairs, 6 ocular diseases | license **unverified** |
| MESSIDOR-2 | ~1,748 | redistribution restricted; at least one benchmark explicitly cannot redistribute it |
| APTOS 2019 | ~3,662 | Kaggle competition terms — **verify before use** |
| IDRiD | ~516 | license **unverified** |
| EyePACS | ~88,000 | Kaggle competition terms — **verify before use** |
| FGADR | — | **cannot be redistributed**, license forbids it |

**Rule that governs all of these:** read the license at the official
repository, not from a secondary aggregator or a Kaggle re-upload. Kaggle
mirrors frequently strip or misstate the original terms. If a license cannot be
confirmed, the corpus is not used. Nothing unlicensed is substituted.

---

## External test data — separate problem, unchanged

Stage-2 pretraining does not address external validation. That remains the
binding constraint on main-track credibility, and the position is unchanged:

- **6 real external RP images total** (RFMiD 2.0 = 4, RFMiD 1.0 = 2). A sanity
  anchor, not a powered test set.
- **Iowa StoneRounds / EyeRounds** — ~490 images from ~178 patients, the richest
  RP source found. CC BY-NC-ND 3.0, no bulk API, site states *"Do NOT scrape"*.
  Legitimate path is to email the maintainers for nonprofit research use and
  confirm that evaluation-only use is compatible with NC-ND.
- **BRSET** — PhysioNet credentialed access, requires CITI human-subjects
  training and a signed data-use agreement. RP count unverified.

Both require a human to correspond and sign, with weeks of latency, and neither
is agent-automatable. **Start them now, in parallel with GPU work.**

---

## Sources

- [Eye Disease Image Dataset — Mendeley Data](https://data.mendeley.com/datasets/s9bfhswzjb/1)
- [A dataset of color fundus images for the detection and classification of eye diseases — Data in Brief](https://www.sciencedirect.com/science/article/pii/S2352340924009417)
- [A Global Review of Publicly Available Datasets Containing Fundus Images — J. Clin. Med.](https://www.mdpi.com/2077-0383/12/10/3587)
- [GDRBench redistribution restrictions](https://github.com/chehx/DGDR/blob/main/GDRBench/README.md)
