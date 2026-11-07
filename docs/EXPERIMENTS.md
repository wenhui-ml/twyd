# Experiment log — budget-limited reproduction (2026-11)

Everything here runs on one 24 GB ROCm card (RX 7900 XTX class), teacher
`Llama-3.2-3B-Instruct`, student = same model with the hidden dim halved
(3072 → 1536, **936.6M** params). The paper's headline uses ~10B compression-stage
tokens on 4×A100-80G; these runs are at **1/20–1/75 of that budget**, so read them
as a scaling study, not as a replication.

## TL;DR — where the student stands on the paper's 8-task suite

| model | stage tokens | Avg8 (acc_norm) | gap to teacher |
|---|---:|---:|---:|
| teacher `Llama-3.2-3B-Instruct` (3.2B) | – | **.7135** | – |
| student @PT (`band-6L`) | 524M | .6633 | −5.02 |
| student @SFT (paper 8-dataset mix) | +131M | **.6693** | **−4.42** |

Two independent bases (the 1.6B `LRC-1.5B-Base` line and this 936M line) agree:
**SFT buys 0.4–0.6 pt, never 5 pt.** The teacher gap is a PT-budget problem.

## 1. SFT on the paper's 8-dataset mixture (2026-11-03/04)

Data: `data/sft_paper8.jsonl` — 205,000 samples from 7 ModelScope sources,
built by `scripts/build_sft_paper8.py`. Recipe: 2,000 steps, `max_length=2048`, lr 1e-5,
cosine, default cross-entropy,
layers 0-2 dense (34.87M trainable = 3.72%), teacher on CPU,
**22.86 GiB peak, 8.1 s/step**.

Budget correction (2026-11-05): `train_core_lr_pilot.sh` has **no** `TOKENS -> MAX_STEPS`
conversion, so `TOKENS=130000000` was ignored and the run stopped at the default
2,000 steps = **0.63 epoch ≈ 78M tokens** (`trainer_state.json: epoch 0.631`), not the
131M implied by `tokens/step=65,536`. The paper's SFT stage is **0.62B tokens ≈ 5 epochs**
of the same file, i.e. this run was ~8x under the paper's SFT budget. `MAX_STEPS` is now
passed through explicitly by `scripts/run_sft_mixture.sh`.

| task | n | teacher | PT only | + SFT | Δ | gap to teacher |
|---|---:|---:|---:|---:|---:|---:|
| arc_challenge (acc_norm) | 1,172 | .4608 | .4428 | .4471 | +0.43 | −1.37 |
| arc_easy (acc_norm) | 2,376 | .7109 | .7176 | **.7332** | +1.56 | **+2.23** |
| boolq | 3,270 | .7869 | .7358 | .7404 | +0.46 | −4.65 |
| commonsense_qa | 1,221 | .7387 | .6568 | .6577 | +0.09 | −8.10 |
| piqa (acc_norm) | 1,838 | .7704 | .7155 | .7258 | +1.03 | −4.46 |
| sciq (acc_norm) | 1,000 | .9340 | .9180 | .9250 | +0.70 | −0.90 |
| winogrande | 1,267 | .6843 | .6204 | .6283 | +0.79 | −5.60 |
| **mmlu** | 14,042 | **.6222** | .4999 | .4969 | **−0.30** | **−12.53** |
| **Avg8** | | **.7135** | .6633 | **.6693** | **+0.60** | **−4.42** |

Reading:

* SFT gains are exactly where behaviour/format matters — ARC-Easy (+1.56), PIQA
  (+1.03), WinoGrande (+0.79), SciQ (+0.70). **ARC-Easy ends above the teacher.**
* MMLU is 54% of the suite's items and it *falls* during SFT. It is the single
  reason Avg8 stays at .67: knowledge capacity is bought with PT tokens and width.
* Same conclusion as the 1.6B line (Avg8 .6584 → .6625 after SFT, `+0.41`).

## 2. PT scaling run — 2B tokens (in flight, started 2026-11-04 10:55)

Motivation: §1 says the only lever left is PT tokens. This run is the first point
above 524M on the same arm, and it saves every ~0.5B tokens so the scaling curve
(0.5 / 1.0 / 1.5 / 2.0B) can be scored without waiting for the end.

| knob | value |
|---|---|
| arm | CORE-LR (`BRANCH=core K=64 LORA_MODE=core FACTORIZED=1`) |
| layers / modules | `0-2,11-13` (the `band-6L` band) / `gate_proj,up_proj` |
| trainable | **5.11M (0.54%)**, 2,768M frozen buffers |
| data | `data/fw_score4_1B.jsonl` (957,524 docs, 4.89 GB) |
| tokens / steps | 2.0B / **15,258** (seq 2048, bs 1, grad-accum 32, 2 cards) |
| loss | `lrc_distill` (teacher pinned on device) |
| peak memory | 22.65 GiB |
| step time | ~28 s (teacher forward dominates; both cards at 100%) |
| checkpoints | every 1,900 steps = 0.25B tokens ≈ 15 h → `output/product/pt-2b-band6L/0-2_11-13/*/checkpoint-1900…15258` |

The first launch used `SAVE_STEPS=3815` (~30 h between checkpoints, nothing on disk for
a day and a quarter); it was stopped at step 625 and restarted at 15:54 with
`SAVE_STEPS=1900`, so an eval-able checkpoint lands every ~15 h. The aborted run is
kept as `output/product/pt-2b-band6L/archived-abort-625steps{,.log}`.

Launch (reproduce):

```bash
GPU=0,1 NPROC=2 SEQ=2048 SAVE_STEPS=1900 BS=1 GAS=32 GATE=0 USE_AUX=0 STAGE=pt \
  DATA=$PWD/data/fw_score4_1B.jsonl PLANS=0-2,11-13 LORA_MODULES=gate_proj,up_proj \
  SEED=42 BRANCH=core K=64 LORA_MODE=core FACTORIZED=1 FREEZE_COL=true \
  PERP_BASIS=teacher TOKENS=2000000000 WARM_START=slice \
  OUTROOT=$PWD/output/product/pt-2b-band6L \
  bash scripts/run_v15_w1.sh
```

Scoring a checkpoint (paper suite = 8 tasks, 0-shot):

```bash
PRESET=paper GPU=0 BATCH=16 bash scripts/run_lm_eval.sh <ckpt> paper_pt2b_<tag>
```

Then the SFT stage on top, with the recipe from §1 (`PT_CKPT=<ckpt>`,
`TOKENS=130000000` — `scripts/run_sft_mixture.sh`).

## 3. What had to be fixed to make the SFT stage fit 24 GB

| symptom | root cause | fix |
|---|---|---|
## 4b. Correction (2026-11-05): the `WARM_START=slice` init was **broken**

Every run in §1-§2 (and the first SFT-only ladder) started from `_shrink_student_`,
which copies the **top-left block** of each teacher tensor (`t[:r, :r]`). The residual
stream has no privileged coordinate basis, so truncating coordinates is not a
projection of the function — it is a random-ish subspace that the layer norms and the
MLP `W @ x` half-sum then amplify. Measured on the same sentence:

| model | residuals | CE (plain English) | generation |
|---|---|---:|---|
| teacher `Llama-3.2-3B-Instruct` | 3072 | **2.855** | coherent |
| slice r=1536 (top-left block) | 1536 | **12.462** | gibberish |
| slice r=1536 with `master_p=embed_svd` (Alg.1 (i) SVD basis) | 1536 | 12.773 | gibberish |
| slice r=3072 (identity) | 3072 | 2.855 | coherent |
| `LRC-1.5B-Base` (paper release) | 1536 | **3.050** | coherent |

Consequences:

* **The 524M-token PT in §2 was distillation into a random init**, not fine-tuning a
  teacher projection — which is why it needed so many tokens and why it plateaued
  around the base model's MMLU. The rank ceiling argument in §5 still stands (it is
  anchored on `LRC-1.5B-Base`, LoRA r=256 and the teacher — all valid models), but
  the token-scaling result of §2 is confounded and cannot be read as "tokens do not
  help a good init".
* **The first SFT-only ladder (S0/S1) is void**: S0 trained from noise (loss 9.73 ->
  6.37 after 1,700 steps). The MMLU-vs-rank sweep in §5 is void too — it scored
  random models (MMLU .22-.28).
* **Working init, zero cost**: the paper's released `LRC-1.5B-Base` (CE 3.050,
  paper-8 Avg8 .6550, MMLU .5006). SFT from it reaches **loss 1.245 at step 20**, vs
  6.37 at step 1,700 from the broken slice — a 2-orders-of-magnitude difference in
  what the same budget buys.
* **Still open**: a principled width-r warm start (paper Alg.1 ridge/SVD with
  calibration activations, `TWYD_MASTER_P` + `calib_ids`) is needed for a real rank
  sweep. Our embed-SVD attempt (`build_master_P` + `project_student_with_P_`) does not
  reproduce it, so the capacity question can currently only be probed at r=1536.

## 4c. First *valid* SFT-only result (2026-11-06): behaviour up, MMLU down

Init: `LRC-1.5B-Base` (the paper's released student, CE 3.050). 1 epoch of
`data/sft_paper8.jsonl` (3,170 steps, lr 1e-4, CE), **all 28 layers** adapted
(34.87M trainable — see the `PLANS` bug below). Paper-8, 0-shot:

| task | LRC base | + SFT | delta |
|---|---:|---:|---:|
| arc_challenge (acc_norm) | .4258 | **.4437** | +1.79 |
| arc_easy (acc_norm) | .6970 | **.7125** | +1.55 |
| boolq | .7330 | .7385 | +0.55 |
| commonsense_qa | .6519 | .6470 | −0.49 |
| piqa (acc_norm) | .7155 | **.7258** | +1.03 |
| sciq (acc_norm) | .9140 | **.9230** | +0.90 |
| winogrande | .6022 | **.6101** | +0.79 |
| **mmlu** | .5006 | **.4781** | **−2.25** |
| **7 behaviour tasks** | .6771 | **.6858** | **+0.87** |
| **Avg8** | **.6550** | **.6598** | **+0.48** |

Reading: SFT does what it is supposed to on behaviour (+0.87 pt on 7 tasks, every
one of them up except CSQA) but costs 2.25 pt of MMLU, which eats 80% of the gain.

**Layer count only trades gain against damage — it does not move the frontier.**
Same init, same data, same lr, 1 epoch:

| arm | layers | trainable | loss | 7 behaviour | MMLU | Avg8 | dAvg8 |
|---|---:|---:|---|---:|---:|---:|---:|
| `LRC-1.5B-Base` | – | – | – | .6771 | .5006 | .6550 | – |
| L1 | 28 (all) | 34.87M | CE | **.6858** | .4781 | **.6598** | **+0.48** |
| M9 | 9 | 11.21M | CE | .6823 | **.4957** | .6589 | +0.39 |
| M3 | 3 | 3.74M | CE | .6805 | .4956 | .6574 | +0.24 |
| KD3 | 3 | 3.74M | distill | .6820 | .4955 | .6587 | +0.37 |

Three things follow from the completed 2x2:

1. **CE scales monotonically with adapter capacity**: +0.24 (3L) -> +0.39 (9L) -> +0.48
   (28L). Behaviour gain and MMLU damage move together, so the net stays in a narrow
   band: SFT with generic instruction data buys **at most ~+0.5 pt Avg8**, and no layer
   choice escapes that.
2. **Distillation is ~3x more parameter-efficient than CE**: KD3 gets +0.37 with 3.74M
   params, which is what CE needs 9 layers (11.21M) to reach. But it does *not* reduce
   the MMLU damage (-0.51, same as CE at the same size) and it does not raise the
   ceiling. Expected: `LRC-1.5B-Base` is *already* the logit-distillation product of
   this same teacher, so re-distilling on instruction data adds no new knowledge.
3. **The gap to the teacher is untouched**: best SFT-only Avg8 .6598 vs teacher .7135
   (-5.4 pt), and MMLU .478-.496 vs .6222. Every route we have tried lands on the same
   trade-off curve -- the binding constraint is the student's rank (r=1536), not the
   adapter, the loss, the token count or the layer plan.

Adapting fewer layers buys back 1.76 pt of MMLU but gives up 0.35 pt of behaviour,
and the two effects cancel: the net is +0.4~0.5 pt either way. **Behaviour gain and
knowledge damage scale together with adapter capacity**, i.e. generic instruction
data slides the model along a trade-off curve instead of pushing it outward. That
is the empirical answer to "will SFT alone lift the average?": yes, by ~0.4 pt, and
no more, no matter how the layers are chosen.

The remaining SFT-stage lever that can add *knowledge* (not just behaviour) is the
distillation objective: `LOSS_TYPE=lrc_distill` on the SFT data, teacher on device.
That is the next arm (`KD3`), and it is the only cheap test of "can SFT close the
MMLU gap without a PT stage?".

**Bug (fixed 2026-11-06).** `scripts/run_sft_mixture.sh` passed `PLANS=$PLANS_ARM`,
but `train_core_lr_pilot.sh` has no `PLANS` variable — the tuner reads
`LORA_LAYERS`. Every SFT run before this date therefore adapted **all 28 layers**
(`lora_layers=all`), not the 0-2 plan its log line claimed. Passing
`LORA_LAYERS="${PLANS_ARM}"` makes the plan real: 3 layers -> 3.7M trainable,
9 layers -> 11.2M, all layers -> 34.87M.

## 4d. The other compression axis, measured: depth pruning without recovery also collapses

`scripts/prune_depth.py` keeps the teacher at full width (hidden 3072, so every MLP stays
rank 3072) and drops whole layers. Zero training, scored on one sentence (CE, plain
English; teacher = 2.855):

| variant | kept layers | CE |
|---|---|---:|
| teacher | all 28 | 2.855 |
| even-spaced | 0,2,...,26 (14) | 10.705 |
| first half | 0-13 (14) | 11.699 |
| drop middle | 0-6 + 21-27 (14) | **8.500** |
| last half | 14-27 (14) | 10.305 |

Even the best variant (dropping the middle) lands at CE 8.5, i.e. unusable, and the
full eval on the exported even-spaced model scored MMLU .2304 / Avg8 ~.28 (chance).

Conclusion: **both compression axes need PT-scale recovery**, not SFT-scale. The
width axis already has one because the paper shipped it (`LRC-1.5B-Base`, CE 3.050,
repaired with ~10B distillation tokens). Depth pruning has no such checkpoint here, so
at SFT budgets it is strictly worse than just starting from `LRC-1.5B-Base`.

This closes the decision tree for the SFT-only question: with a 100M-1B-token budget
the only usable student is the already-repaired width-halved one, and SFT on top of it
is worth **+0.4~0.5 pt Avg8** (§4c), not parity.

## 4e. Paired significance test: the gain is real but the *net* is zero

McNemar on the saved per-item samples (`output/lm_eval/paper_*`, base = `LRC-1.5B-Base`):

| arm | 7 behaviour tasks (n=12,144) | MMLU (n=14,042) | all items (n=26,186) |
|---|---|---|---|
| L1 (28L, 34.87M) | **+0.89 pt, p=2.0e-05** | −2.25 pt, p=1.3e-15 | **−0.79 pt, p=9.8e-06** |
| M9 (9L, 11.21M) | +0.49 pt, p=0.012 | −0.50 pt, p=0.026 | −0.04 pt, p=0.80 (tie) |
| M3 (3L, 3.74M) | +0.30 pt, p=0.085 | −0.51 pt, p=0.016 | −0.13 pt, p=0.34 (tie) |
| KD3 (3L, distill) | +0.38 pt, p=0.021 | −0.51 pt, p=0.016 | −0.10 pt, p=0.48 (tie) |

The behaviour gain is **not noise** (p = 2e-05 for L1, and it holds at every capacity).
But the metric decides the verdict:

* **Paper metric (Avg8 = unweighted mean of the 8 task accuracies)**: +0.24 ~ +0.48 pt.
  Every task has equal weight, so the 7 behaviour tasks (which improve) count 7/8.
* **Deployment metric (pool all 26,186 items)**: −0.79 ~ −0.04 pt, because MMLU is 54%
  of the items and it drops.

So the honest one-line answer to "does SFT help?": **it moves 7 tasks up by 0.3-0.9 pt
and MMLU down by 0.5-2.3 pt; the pooled effect is a tie at best.** The best product
choice among these arms is **M9 or KD3** (statistically indistinguishable from the
base model on the pooled metric, small behaviour gain, and KD3 needs 1/9 of M9's
adapter parameters). L1 is the only arm that is significantly *worse* overall.

## 4f. Knowledge-injection arm (running, 2026-11-07): teacher-written CoT data

§4e says the SFT stage only moves behaviour, because CE on instruction data never asks
the student to *recall a fact*. The one SFT-stage lever that can carry knowledge is
teacher-written text: sequence-level distillation on a corpus of our own.

Pipeline (`scripts/gen_teacher_cot.py` -> `scripts/build_sft_cot_mix.py` -> SFT):

| step | detail |
|---|---|
| prompts | fineweb-edu passages (`data/fw_score4_1B.jsonl`); **no paper-8 eval item is ever shown to the teacher** |
| tasks | 60% "write one 4-option MCQ + answer + reasoning", 40% "write a study note" |
| volume | 90,000 samples on 2 GPUs, ~20M generated tokens, ~516 tok/s, ~11 h |
| quality gate | format check on 32 samples: MCQ prompts 19/19 well formed after moving from "2 questions" to "1 question" + `repetition_penalty=1.08`, `temperature=0.6` (the first prompt version looped) |
| mix | `data/sft_paper8.jsonl` (205k) + CoT x2 -> 385k samples, ~22% CoT tokens |
| train | `LRC-1.5B-Base`, 9 layers (`0-2,11-13,25-27` = 11.2M trainable), 1 epoch, lr 1e-4, CE |
| eval | paper-8, paired against base / M9 / M3 / KD3 |

Success criterion, stated before the result: **MMLU must hold at >= .50 (base) while
behaviour stays at M9's level (+0.49 pt)**. Anything less means sequence-level
distillation at a 20M-token budget cannot buy knowledge either, and the SFT-only
question is settled (answer: no parity, ship the behaviour-only win).

## 5. Why more tokens did nothing: the rank ceiling (and the sweep that measures it)

`§1–§2` showed 322M and 524M tokens are indistinguishable, and LoRA r=256 with
44.8M trainable params is no better than our 5.11M-param CORE-LR arm. Both facts
point at the same constraint, and it is not the adapter or the token budget:

| arm (same paper-8 suite, same harness) | width | trainable | Avg8 | MMLU |
|---|---:|---:|---:|---:|
| teacher `Llama-3.2-3B-Instruct` | 3072 | – | **.7135** | **.6222** |
| CORE-LR k=64, 524M tok | 1536 | 5.11M | .6633 | .4999 |
| CORE-LR k=64 + SFT | 1536 | 5.11M | .6693 | .4969 |
| LoRA r=256, 9 layers, 524M tok | 1536 | **44.83M** | .6611 | .4947 |
| `LRC-1.5B-Base` (paper release) | 1536 | – | .6550 | .5006 |

Giving the adapter **8.8×** more parameters changed nothing, so the ceiling is the
student, not the update. The mechanism is **rank, not non-linearity**:

* SiLU/GELU is element-wise — a 1536-wide model is just as non-linear as a 3072-wide one.
* Every layer's MLP composite `W_down · W_up` has rank ≤ `min(d_hidden, d_ff)`: 3072
  for the teacher, **1536** for the student. The residual stream is the inter-layer
  bus, and its width halves with it.
* Width reduction is a **projection**: the discarded directions cannot be written
  back by any amount of training (data-processing inequality). LoRA never projects —
  `W = W₀(full rank) + BA` is a *superset* of the original model, which is exactly
  why LoRA can match or beat a teacher that it literally contains.

That also explains the CE/MMLU split we measure: the slice keeps the high-variance
principal subspace (next-token prediction is dominated by frequent patterns — CE
2.4318–2.4437 is *better* than `LRC-1.5B-Base`'s 2.4771) and drops the low-variance
long tail where specific facts live (MMLU −12 pt). **Lower CE does not mean the
knowledge survived.**

**Measurement (`scripts/export_slice.py`).** The zero-training slice init is
exported at several widths and scored on the same suite; every point costs no
training, only an eval:

```bash
python scripts/export_slice.py --r 2048 --out output/slices/r2048
PRESET=paper GPU=0 BATCH=16 bash scripts/run_lm_eval.sh output/slices/r2048 paper_r2048
```

Widths under test: **768 / 1152 / 1536 / 1920 / 2304** (3072 = the teacher itself).
They must be multiples of the teacher's 24 attention heads: the slice keeps the head
count and shrinks `head_dim`, so 1024/2048/2560 fail HF's `hidden % num_heads`
config validation. Exported sizes are 803M / 1,205M / 1,606M / 2,008M / 2,410M; the
r=1536 point (1,606M) matches the paper's released `LRC-1.5B-Base` exactly, which is
a useful sanity check that the export is the same architecture we train.
If MMLU rises monotonically with `r` while the 8-task behaviour tasks saturate
early, then "half the parameters" must be bought by **pruning depth at full width**,
not by halving the residual stream — and the SFT-only path can only repair what the
projection left behind.

| OOM at 23.19 GiB, *before* the first step, insensitive to `MAX_LENGTH` | complement basis `U⊥` materialised for **every** MLP module (84 × ≈109 MB ≈ 9 GiB) although `LORA_LAYERS` selected 6 | build it lazily — only for modules that will train (`in_lora_set and with_perp`), both `left` and `right` paths (`twyd/tuner.py`) |
| `KeyError: 'nll'` when switching the SFT loss | `nll` is not a registered `--loss_type` | `LOSS_TYPE=` (empty) omits the flag → swift default cross-entropy, which is what the paper's SFT uses |
| teacher resident during SFT | distillation loss is a PT objective | `TWYD_TEACHER_DEVICE=cpu` + no `--loss_type` for SFT |
| 2 of 28 CPU threads used | `DATA_PROC=1`, `dataloader_num_workers=0` | `DATA_PROC=12 DATALOADER_WORKERS=4` (`scripts/train_core_lr_pilot.sh`) |

The lazy-basis change is a pure win: PT peaks drop by up to ~9 GiB whenever the
layer plan is selective, and `tests/` (48 tests, incl. the CORE ≡ dense ≡ plain
initialisation invariants) still pass.

## 4. Where the numbers live

* `output/RESULTS_sft_paper8.md` — the SFT table above, with reproduce commands.
* `output/RESULTS_m525.md` — the 1.6B / 524M-token matched-budget study.
* `output/lm_eval/paper_{teacher,band6L,sft_paper8}/…/results_*.json` — raw harness output.
