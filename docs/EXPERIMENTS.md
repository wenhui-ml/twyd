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
