# Roadmap

First written 2026-06-07, before the experiments. Cleaned up and updated
2026-06-09 after the proof-of-concept cycle. Results of that cycle are in
`LOG.md`; this file is only about what comes next and why.

## The idea in one paragraph

Cloud et al. (2025) found that subliminal transfer needs teacher and student to
share initialization. But one pretraining seed fixes two things at once: the
initial weights and the order of the training data. The PolyPythias release has
`pythia-160m` siblings where only one of the two changes:
`pythia-160m-data-seed{1,2,3}` (same init, different data order) and
`pythia-160m-weight-seed{1,2,3}` (different init, same data order), plus
`pythia-160m-seed{1..9}` where both change, and 154 intermediate checkpoints per
run. So one can ask which half of "same initialization" actually carries the
channel. As far as I know, nobody has used the decoupled variants for this.

## Done: proof of concept

Before any of the plan makes sense, the transfer must be measurable on a 160M
base model at all. That was not obvious (the original works on GPT-4.1-scale
chat models). The PoC showed: yes, measurable, with full fine-tuning for both
the teacher induction and the student training. LoRA gives a stable null, which
was a finding in itself. Numbers and dead ends: `LOG.md`.

## Next: the 2x2 experiment

One owl teacher on `pythia-160m`. One pool of filtered numbers from it. Distill
into students from each variant:

| Student base                   | Init      | Data order | Question                              |
|--------------------------------|-----------|------------|---------------------------------------|
| `pythia-160m`                  | same      | same       | upper bound (= the PoC setting)        |
| `pythia-160m-data-seed{1-3}`   | same      | different  | does the channel live in the init?     |
| `pythia-160m-weight-seed{1-3}` | different | same       | or in the shared data stream?          |
| `pythia-160m-seed{1-3}`        | different | different  | lower bound                            |

Several students per cell, same training and eval as in the PoC. Any outcome is
informative: the channel follows the init, or the data order, or needs both, or
survives everywhere (which would mean "same base" in the original paper is a
stronger condition than necessary).

## After that, if the 2x2 gives a signal

Roughly in order:

- Mechanism of the LoRA null: measure the dot product between the teacher's
  induction step and the student's distillation step under LoRA vs full FT,
  sweep the LoRA rank. This is the quantity the original paper's theorem is
  about, and the LoRA result predicts it should be near zero there.
- Cheap similarity measures between teacher and student computed before
  distillation (activation similarity, gradient cosine): do they predict the
  measured transfer across the 2x2 cells?
- Checkpoints: teachers from different points of the pretraining run. When does
  the ability to carry the channel appear? PolyPythias has 154 checkpoints per
  run and known phase events (induction heads around step 1k) to compare
  against. This is also one of the example tasks in the project description.
- 410M: combined seeds only, but enough to check the effect is not a 160M
  oddity. Known problem from the PoC: on 410M only the owl teacher kept
  generating clean numbers, so the control needs care there.

## Fixed choices so far

- Trait: animal preference (owl), measured by log-odds against alternative
  animals. Free-form eval is impossible at this scale. A second animal is
  queued as a control on owl-specificity.
- Method: full fine-tuning everywhere from now on. LoRA only as an object of
  study, not as the workhorse.
- Models: the released decoupled 160M set, no own pretraining.
- Compute: a single consumer GPU (RTX 3060 class) is enough for everything
  above, except maybe the checkpoint sweep.
