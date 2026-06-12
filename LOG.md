# Log

Working notes from the proof-of-concept cycle, in the order things happened.
Dead ends included, because they explain why the final setup looks the way it
does. The setup itself is described in `README.md`.

## All runs so far

Note added 2026-06-11: rows 1-7 below were measured with the old eval, which
turned out to have only 15 distinct prefixes (entry 11). All seven settings were
re-run on the same training data under the fixed eval; those re-runs are the
current numbers and live in entry 11. The rows below stay as history.

| # | Method  | Model, seed         | Control         | Teacher delta | Contrast (95% CI)       | Transfer | Result file |
|---|---------|---------------------|-----------------|---------------|-------------------------|----------|-------------|
| 1 | LoRA    | 160M, seed 0        | reference       | +9.75         | -0.51 (CI not recorded) | no       | not kept, see entry 9 |
| 2 | LoRA    | 160M, seed 0        | dolphin teacher | +8.27         | -0.47 [-0.95, -0.004]   | no       | not kept |
| 3 | LoRA    | 410M, seed 0        | reference       | +10.5         | -0.34 [-0.46, -0.23]    | no       | not kept |
| 4 | full FT | 160M, seed 0, run 1 | reference       | +11.35        | +1.18 [+0.60, +1.79]    | yes      | not kept |
| 5 | full FT | 160M, seed 0, run 2 | reference       | +12.20        | +0.65 [+0.18, +1.16]    | yes      | `results/strength_pythia-160m_full_owl_ind256x3.0_ctrl-ref_seed0.json` |
| 6 | full FT | 160M, seed 1, run 1 | reference       | +10.16        | +0.79 [+0.14, +1.41]    | yes      | not kept |
| 7 | full FT | 160M, seed 1, run 2 | reference       | +12.92        | +1.14 [+0.47, +1.83]    | yes      | `results/strength_pythia-160m_full_owl_ind256x3.0_ctrl-ref_seed1.json` |

"Teacher delta" is how much the teacher's owl log-odds moved from the base model
after induction. It is my own working proxy for teacher strength, not a metric
from the literature: the same log-odds score I use on students, applied to the
teacher. (The original paper measured teacher traits by free-form answer rates,
which a small base model cannot produce.) The delta alone can mislead, a
destroyed model also has a huge delta, so I always read it together with the
generation yield (entry 2).

"Contrast" is owl-student minus control-student, mean over 50 prefixes,
bootstrap 95% CI. Every run trained each student on 10k filtered sequences
(entry 10 describes one exception that got caught and fixed), and every run had
0 owl tokens in the filtered training data.

So the full-FT setting ran four times (two seeds, two runs each) and came out
significantly positive every time, with the size moving between +0.65 and +1.18.
The size is noisy because full-FT training is not bit-deterministic on this GPU
(entry 10); the sign and the significance were stable. The three LoRA runs are
negative-to-zero at every teacher strength I tried.

About "not kept": result files written before 2026-06-09 had two bugs (entry 9).
The file name did not include method/control/seed, so different runs could
overwrite each other, and the JSON recorded the default config instead of the
one that actually ran. I deleted those files rather than publish artifacts with
wrong labels; the numbers above come from the run logs. Rows 5 and 7 were run
under the fixed pipeline and their artifacts are committed, together with the
exact training data they used (`results/gen_cache/`). Re-running the LoRA rows
the same way is on the list.

## Why LoRA may block the transfer (a guess, not a result)

Theorem 1 of Cloud et al.: if the student and the teacher start from the same
parameters, one sufficiently small gradient step on the teacher's outputs either
has a zero dot product between the student step and the teacher step, or it
moves the student toward the teacher in the teacher's loss. So the transfer
rides on that dot product, and the paper notes it vanishes when the teacher's
loss depends only on parameters that do not affect the student's loss.

My guess: LoRA confines the teacher's induction step and the student's
distillation step to two different low-rank adapter subspaces. That pushes the
dot product toward zero, and the channel closes. Full fine-tuning takes the
full-parameter step the theorem is actually about.

I have not measured the dot product under the two regimes. That measurement is
the clean next experiment, together with a LoRA rank sweep (does the transfer
come back as the rank grows?). Other explanations exist, for example that LoRA
steps are simply too small or too constrained. The empirical part (LoRA no,
full FT yes) does not depend on which explanation is right. The original paper
used full SFT everywhere and, as far as I can tell, never compared LoRA against
full FT.

## Chronology

**1. Pipeline bring-up.** Built the full loop: induce teacher, generate filtered
numbers, train student, eval. Compute: one RTX 3060 rented on vast.ai. Two early
practical findings that shaped everything after:

- base pythia-160m passes the strict numbers-only filter only ~10-12% of the
  time (its raw generations are messy), so generation needs heavy oversampling,
  about 200k attempts for 10k kept sequences;
- `max_new_tokens` is the main speed lever: cutting it from 64 to 24 nearly
  triples generation throughput and even raises the yield a little.

**2. Over-induction.** First teacher: 512 examples x 10 epochs, teacher delta
about +20. Looks impressive, but the model is overwritten: it stops producing
numbers and rambles owl text instead, so almost nothing passes the filter. The
channel needs a teacher that is biased and still able to do the task. Calibrated
down (`calibrate_induction.py`) to 256 examples x 1 epoch, delta about +2.5,
with healthy yield.

**3. Wrong metric (important).** The first version compared the student against
the untouched baseline model. This turned out to be wrong: fine-tuning on ANY
number data, the control data too, drops the owl log-odds by about 4 points.
This drift has nothing to do with the trait, and it swamps the signal. Fix:
compare the trait student against the control student. Both get the same
numbers treatment, so the drift cancels, and what remains is the difference
caused by which teacher made the numbers. All numbers in this log are such
contrasts.

**4. Eval degeneracy.** Under aggressive student training (3 epochs, LoRA rank
16) the student becomes number-fixated: the most likely next token after "my
favorite animal is the" is a digit, and " owl" falls to rank 300-700. The
log-odds still contains some signal, but it sits near the floor.
`diagnose_eval.py` checks for this. Gentle student training (LoRA: 1 epoch,
rank 4; full FT: small lr) keeps the eval alive.

**5. Teacher strength sweep (LoRA).** Maybe the delta +2.5 teacher was just too
weak? I induced a 4x stronger teacher, delta +9.75 (row 1). Still no transfer
under LoRA. So teacher strength was not the blocker.

**6. Symmetric control (LoRA).** Maybe the flat or negative contrast is a
confound: trait numbers come from a fine-tuned model, control numbers from an
untouched one, and these two distributions can differ in other ways than the
trait. I replaced the control with a second teacher induced on "dolphin" at the
same strength, so the only difference between the two channels is the animal.
The contrast barely moved (-0.47 vs -0.51, row 2). Not the cause.

**7. Scale to 410M (LoRA).** Same null (-0.34, row 3). A side finding that
matters for experiment design: on 410M, owl is the only animal I tried whose
induced teacher still generates clean numbers. Cat, eagle, wolf, lizard, dolphin
all collapse to about 0% yield at the same dose (`yield_screen_410m.py`). So a
symmetric-animal control is not possible on 410M, and the reference control was
used there (on 160M the two controls gave nearly the same answer).

**8. Switch to full fine-tuning.** The theorem is about a full gradient step,
and LoRA was my own addition for cheapness; the original never used it. So both
stages, teacher induction and student training, switched to full FT with a small
lr. Calibration first (`calibrate_fullft_160m.py`): 256 examples x 3 epochs at
lr 5e-5 gives delta about +11-12 with 4-7% generation yield. Full FT at LoRA's
lr 2e-4 simply destroys the model; the working range is about 2e-5 to 5e-5.
Result: contrast +1.18 [+0.60, +1.79] (row 4). The first positive run.

**9. Result-file hygiene.** While preparing the repo for publishing I found two
bugs in the result writing. First, file names were templated only on the
induction parameters, so two different runs (for example LoRA and full FT at the
same strength) wrote to the same path and overwrote each other. Second, the
provenance block recorded the default config instead of the configs that
actually ran; it said method "lora" even for full-FT runs. Both fixed in
`strength_probe.py`: names now carry model/method/target/control/seed, and the
actual configs are written back before dumping. Files produced before the fix
were deleted, which is why most table rows say "not kept". The run-1 file of
row 4 was lost to exactly this overwrite problem.

**10. Re-runs and a second seed.** With the fixed pipeline I re-ran the full-FT
setting on both seeds. Seed 0 again positive but smaller: +0.65 [+0.18, +1.16]
(row 5, archived). Full-FT training is not bit-deterministic on this GPU even
with fixed seeds (deterministic kernels are requested, but not all bf16 ops
comply), so run-to-run variation of this size is apparently normal; the sign
held in all four full-FT runs. Seed 1 had its own lesson first: the initial run
looked null at +0.30, but it had trained on only 5.4k sequences. That seed's
teacher had about 2% generation yield, and the attempts cap stopped generation
early. After regenerating to the full 10k, seed 1 gave +0.79 (row 6), and the
re-run under the fixed pipeline gave +1.14 [+0.47, +1.83] (row 7, archived).
Lesson: check n before comparing anything.

**11. Eval bug: the 50 prefixes were really 15 (plus a seeding bug).**
2026-06-11. While cleaning the repo for publishing I found two real bugs, both
mine.

First, the eval. `make_prefixes(50)` had only 15 templates and silently filled
the rest with duplicates. The eval is deterministic, so a duplicated prefix
gives byte-identical scores: my "50 prefixes" carried 15 independent
observations, and every bootstrap CI in rows 1-7 is too narrow, roughly by
sqrt(50/15) = 1.8x. It also explains an oddity in the archived seed-0 run:
"21/50 prefixes favor trait" together with a clearly positive mean. That
fraction was weighted by how often each template happened to be duplicated.
Fix (eval v2): 50 distinct templates, the old 15 kept as the first 15, no
sampling, and `make_prefixes` now refuses n > templates instead of duplicating.
Result files carry `_eval-v2` in the name and store all per-prefix scores, so
this kind of check is possible after the fact without retraining anything.

Second, seeding. Corpus building, generation prompts and prefix sampling
derived their RNG from `(seed, "...", n).__hash__()`. Python salts string
hashes per process, and setting PYTHONHASHSEED at runtime (which
`seed_everything` did) does nothing. So at a fixed `--seed` the induction
corpus and the generation prompts still differed between processes. Part of
the run-to-run spread that entry 10 blamed on bf16 nondeterminism was simply
this. Fix: string seeds (`random.Random(f"{seed}:gen:{n}")` goes through
sha512 and is process-stable). One consequence: the exact teachers behind the
archived data caches cannot be reconstructed (their corpora came from the
salted path). The re-runs below retrain the teacher with the same recipe and
train the students on the archived cached numbers, which is what the claim is
about anyway.

Re-runs: every setting from rows 1-7, same cached training data, eval v2, one
pass each, archived as `results/strength_*_eval-v2.json` (training data for
the LoRA rows is now committed to `results/gen_cache/` too):

| Setting             | Teacher Δ | Contrast (95% CI)       | CI excludes 0  |
|---------------------|-----------|-------------------------|----------------|
| LoRA 160M, ref ctrl (row 1)     | +9.6  | +0.36 [-0.18, +0.91] | no |
| LoRA 160M, dolphin ctrl (row 2) | +9.6  | -0.02 [-0.55, +0.49] | no |
| LoRA 410M, ref ctrl (row 3)     | +9.6  | -0.25 [-0.40, -0.10] | yes, negative |
| full FT 160M, seed 0 (rows 4-5) | +12.1 | +1.24 [+0.72, +1.75] | yes |
| full FT 160M, seed 1 (rows 6-7) | +16.3 | +0.81 [+0.19, +1.42] | yes |

What changed in the picture. Full FT holds on both seeds, and seed 0 got
stronger under the fixed eval (35/50 prefixes favor the trait, median +1.5).
The 160M LoRA rows are nulls now, not negatives: the old negative point
estimates were partly the prefix-weighting artifact. The 410M LoRA row stays
genuinely negative, and I do not have an explanation; the symmetric dolphin
control is impossible there (entry 7), so a distribution confound cannot be
excluded. A paired per-prefix comparison on the same 50 prefixes, full-FT
seed 0 minus LoRA row 1, gives +0.88 [+0.17, +1.61] - the method difference
itself is significant, with the old caveat that the teachers are still not
strength-matched (Δ +12.1 vs +9.6).

A bonus finding. I ran the whole verification twice, in two separate pod
sessions on the same GPU model and driver (RTX 3060, driver 570, torch
2.11.0+cu128), and the two passes came out identical in every digit I compared
- contrasts, CIs, medians, per-prefix counts. So with the seeding fixed the
pipeline is exactly reproducible on the same hardware, and the run-to-run
spread of rows 4-7 (+0.65 vs +1.18 at one config) was most likely the salted
seeding and the prefix-weight lottery, not bf16 nondeterminism as entry 10
guessed. Different hardware may still shift digits; that is now the remaining
unknown.

Housekeeping in the same pass: unit tests for the filter, the prefixes and the
corpus (`tests/`, no GPU needed); one bootstrap implementation (5000
resamples, recorded as actually ran - the old result JSONs recorded 2000 while
the script computed with 5000); `sync_to_pod.sh` now writes a GIT_HASH file,
so future results from the pod will carry a real commit hash (the runs in this
entry were made before the commit, so their provenance says "unknown", same as
the v1 artifacts); `strength_probe.py` defaults are the
archived recipe, so a bare run reproduces it. One verification pass is five
runs and about 40 minutes on the usual 3060; both passes together cost ~$0.10.

## Practical notes

- Models: `EleutherAI/pythia-160m`, `EleutherAI/pythia-410m`. Next cycles will
  use the decoupled PolyPythias variants (`pythia-160m-data-seed*`,
  `pythia-160m-weight-seed*`).
- Induction recipe on 160M (full FT): 256 examples x 3 epochs at lr 5e-5 gives
  delta about +11-13 and generation yield 4-7%. Push harder and the teacher
  loses the number task (the +20 collapse of entry 2).
- Channel: the number format and the strict filter follow Section 3 of the
  original paper. Yield is 2-18% depending on model, animal and seed. The leak
  check (no target-word substring in any kept sequence) ran in every reported
  experiment and was always 0.
- Eval: 50 plain-text prefixes (genuinely distinct since the entry-11 fix),
  log-odds of " owl" against 8 alternatives, multi-token words scored by summed
  token log-probs. Free-form eval does not work at this scale; see entry 4 for
  the related trap.
- Results: one JSON per run, named
  `strength_<model>_<method>_<target>_ind<...>_ctrl-<...>_seed<N>.json`, with
  the actually-used configs and package versions inside. The generation cache
  resumes from partial JSONL files, so a crash never loses progress.
- Compute: RTX 3060 12 GB, about $0.06/hour. Full FT of 160M fits in ~4 GB
  VRAM. One `strength_probe` run takes 30-60 minutes depending on yield; with
  the committed data cache more like 20-30.

## Open questions / next steps

1. A second trait (eagle) on 160M: how much of this is owl-specific?
2. Measure the teacher-step / student-step dot product under LoRA vs full FT,
   and sweep the LoRA rank.
3. A matched comparison: LoRA and full-FT teachers with the same delta and the
   same data volume, to separate method from strength (the entry-11 teachers
   are Δ +12-16 full vs +9.6 LoRA, still not matched).
4. Why is the 410M LoRA contrast slightly negative? It survived the eval fix
   (entry 11), so it is not the prefix artifact.
5. Then the actual project: the init vs data-order 2x2 on the decoupled models
   (see `ROADMAP.md`), now that there is a regime where transfer is measurable.
