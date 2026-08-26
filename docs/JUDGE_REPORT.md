# Judge report — prioritization of GAN training arms

**Recommendation: major revision before GPU training**

## Summary judgment

The non-transferred (`synth_scratch`) GAN arm adds limited value to the central
research question. It tests whether fundus-corpus pretraining improves a GAN,
but that is only the secondary S2 contrast. The study's confirmatory questions
are whether a transferred GAN adds value beyond oversampling (P1) and how much
evaluation-eye exposure inflates that value (P2) (`FINAL_PLAN.md:61-72`).

Training three scratch GANs therefore spends substantial experimental effort on
a secondary mechanism while the confirmatory leakage arm has only one training
seed (`FINAL_PLAN.md:152-156, 177-182`). This allocation should be reversed.

## Main concern

A single GAN checkpoint cannot represent generator-training variability.
Generating thousands of samples from one checkpoint replicates sampling, not
training, so the protocol correctly states that generator contrasts must use
training seed as the unit of inference (`FINAL_PLAN.md:152-156`;
`RUNPOD_RECIPE.md:17-20`).

However, `synth_leak` currently has one seed even though `synth_leak` versus
`synth_transfer` is confirmatory P2 (`FINAL_PLAN.md:65-67, 152-156, 177-182`).
With one leaky run, an observed difference could reflect leakage, initialization,
or training trajectory. Size-matching the clean and leaky datasets controls
sample count, but it does not control seed-to-seed GAN variation
(`RUNPOD_RECIPE.md:13-20`).

## Required revision

Remove `synth_scratch` from the core experiment and reallocate its runs to the
leakage contrast:

| run | seeds | purpose |
|---|---:|---|
| Stage-2 fundus pretraining | 1 | shared initialization |
| `synth_transfer` | 3 | clean transferred generator |
| `synth_leak` | 3 | replicated leakage condition |

Keep `real_only` and `oversample`; they establish the floor and isolate class
balance without introducing new pixels (`FINAL_PLAN.md:52-58`). Keep
`pca_sampler` only if page space permits, because the plan already identifies it
as the first arm to cut (`FINAL_PLAN.md:227-228`).

The scratch-GAN comparison may be moved to future work or an appendix after P1
and P2 are complete. It should not receive three runs while a confirmatory
contrast remains unreplicated.

## Effect on the contribution

This revision narrows the paper to its strongest defensible claim: separating
class rebalancing, clean transferred synthesis, and generator-mediated leakage.
It does **not** claim that scratch GANs are empirically ineffective; that has not
been measured under the current design. The judgment is that they have low
marginal value for this paper's primary question.

## Decision

**Major revision, likely acceptable after reallocation.** Do not begin the
scratch-GAN runs unless the confirmatory clean-versus-leaky comparison has first
received adequate independent generator replication.
