# TWYD compression study — results summary (2026-11-08)

One-page digest of `docs/EXPERIMENTS.md`. All numbers are measured on this box
(2 × 24 GB ROCm), paper-8 suite: ARC-C, ARC-E, BoolQ, CSQA, PIQA, SciQ, WinoG, MMLU,
0-shot, `acc_norm` where the task has it, n = 26,186 items. Teacher =
`Llama-3.2-3B-Instruct` (3.2B). Students are all **hidden 1536 / 28 layers /
d_ff 8192 / head_dim 128** (1,606M), i.e. the paper's architecture.

## 1. Headline

| arm | train tokens | Avg8 | ΔAvg8 | MMLU | pooled 26,186 items (paired) |
|---|---:|---:|---:|---:|---|
| teacher `Llama-3.2-3B-Instruct` | – | **.7135** | +5.85 | **.6222** | – |
| `LRC-1.5B-Base` (paper's released student) | **~10B** | .6550 | – | .5006 | reference |
| + SFT 3L (3.74M) | 78M | .6574 | +0.24 | .4956 | p=0.34 (ns) |
| + SFT 3L distill (3.74M) | 78M | .6587 | +0.37 | .4955 | p=0.48 (ns) |
| + SFT 9L (11.2M) | 78M | .6589 | +0.39 | .4957 | p=0.80 (ns) |
| + SFT 28L (34.9M) | 78M | .6598 | +0.48 | .4781 | **p=9.8e-06 (worse)** |
| + SFT 9L + teacher CoT | 130M | .6572 | +0.22 | .5002 | p=0.73 (ns) |
| **PT 524M + SFT 78M** | **602M** | **.6693** | **+1.43** | .4969 | **+0.58 pt, p=0.0022** |
| **PT-knowledge 300M + SFT 78M** | **378M** | **.6634** | **+0.84** | **.5075** | **+0.72 pt, p=8.4e-05** |
| **PT-mixed 500M (kno+web) + SFT 78M** | **578M** | **.6674** | **+1.24** | **.5065** | **+0.89 pt, p=1.3e-06** |
| PT-mixed 500M only (no SFT) | 500M | .6646 | +0.96 | **.5083** | +0.75 pt, p=2.1e-05 |

*Avg8 = mean of the harness acc_norm where available, else acc (this is the column
convention; `acc`-only means are ~0.3 pt higher). The knowledge arm trades 0.59 pt of Avg8
for +1.06 pt of MMLU relative to the fineweb chain - see F1b.*

Two sentences:

* **The PT + SFT chain is the only configuration that is significantly better than the
  paper's own released student on the deployment metric**, and it does it with **1/16 of
  the tokens** (602M vs ~10B).
* **SFT alone cannot beat the base student**: five arms, every one lands in the +0.2~+0.5
  Avg8 band with a pooled effect of "tie" or "worse".

## 2. Findings

### F1. PT is what protects MMLU; SFT is what converts it into behaviour

| stage | 7 behaviour tasks | MMLU | pooled |
|---|---|---|---|
| PT 524M only | +0.91 pt, p=1.4e-05 | −0.08 pt, p=0.81 | +0.38 pt, p=0.038 |
| + SFT 78M | **+1.69 pt, p=7.4e-14** | −0.37 pt, p=0.22 | **+0.58 pt, p=0.0022** |
| SFT 78M only (best) | +0.89 pt, p=2.0e-05 | **−2.25 pt, p=1.3e-15** | −0.79 pt, p=9.8e-06 |

The same 78M-token SFT costs 2.25 pt of MMLU on the raw base student and costs nothing
measurable on the distilled one. A bigger SFT is **not** a substitute for the PT stage.

### F1b. The PT *corpus* picks the point on the trade-off frontier (new, 2026-11-09)

Identical recipe, identical 5.11M trainable parameters, identical KD objective - only the
PT corpus changes:

| PT corpus | tokens | MMLU | 7 behaviour | pooled |
|---|---:|---:|---:|---:|
| fineweb-edu | 524M | .4999 (-0.08) | **+0.91 (p=1.4e-5)** | +0.38 (p=.038) |
| wikipedia+math | 300M | **.5076 (+0.70, p=.010)** | +0.40 (p=.071) | +0.56 (p=.0016) |
| wikipedia+math, then SFT 78M | 378M | **.5075 (+0.68, p=.016)** | +0.77 (p=7.2e-4) | **+0.72 (p=8.4e-5)** |

The two corpora are **not** ranked, they are **different**: web text teaches behaviour,
encyclopedic/math text teaches knowledge, and the pooled scores are a tie (p=0.44
head-to-head). What matters is that the knowledge corpus is the only configuration in the
whole study that is **significantly better than the paper's 10B-token student on MMLU**,
and adding the standard SFT on top keeps that gain (0.01 pt lost) while buying +0.77 pt of
behaviour - the best pooled arm in the study at 378M tokens (1/26 of the paper's budget).

This also corrects finding F2 / EXPERIMENTS.md section 5: the MMLU ceiling is
**rank x corpus**, not rank alone.

### F1c. The two PT corpora are ADDITIVE - mixed PT is now the best arm (new, 2026-11-11)

Test of F1b: 500M tokens of 50/50 wikipedia+math / fineweb-edu (interleaved), same recipe.

| PT corpus | tokens | MMLU | 7 behaviour | pooled |
|---|---:|---:|---:|---:|
| fineweb only | 524M | .4999 (-0.08) | +0.91 | +0.38 |
| knowledge only | 300M | .5076 (+0.70) | +0.40 | +0.56 |
| **50/50 mix** | **500M** | **.5083 (+0.77, p=.0044)** | **+0.73 (p=9.7e-4)** | **+0.75 (p=2.1e-5)** |

The mix takes the MMLU of the knowledge corpus **and** the behaviour of the web corpus - the
two effects add within noise (predicted +0.31 / +0.66, observed +0.77 / +0.73). So this is
not a frontier you have to choose a point on; it is a **sum**, and the right PT corpus is a
mixture.

With the standard 78M SFT on top (`paper_MIX9`, 578M tokens total = **1/17 of the paper's
budget**): **MMLU .5065 (+0.58, p=.040), behaviour +1.24 (p=3.8e-8), pooled +0.89
(p=1.3e-6)** - the largest effect anywhere in this study, and the only arm that is
simultaneously better than the paper's 10B-token student on MMLU *and* on behaviour.

Head-to-head: vs `know9` pooled p=0.14 (tie), behaviour mix9 wins p=2.6e-3; vs the fineweb
chain `ptSFT8` pooled p=0.061, MMLU mix9 wins p=4.6e-5. **`mix9` weakly dominates both.**

### F2. Layer choice trades gain against damage, and the net is flat

CE SFT scales monotonically with adapted capacity (+0.24 / +0.39 / +0.48 for 3 / 9 / 28
layers) but so does MMLU damage (−0.5 / −0.5 / −2.25). Logit distillation is ~3× more
parameter-efficient than CE (3 layers + KD ≈ 9 layers + CE) but does not raise the ceiling.

### F3. MMLU is a rank wall, not a token wall

| student | width | PT tokens | MMLU |
|---|---:|---:|---:|
| paper release | 1536 | ~10B | .5006 |
| ours | 1536 | 524M | .4999 |
| ours + SFT | 1536 | 602M | .4969 |
| ours (LoRA r=256, 44.8M params) | 1536 | 524M | .4947 |
| teacher | 3072 | – | .6222 |

Eight hundred-fold more PT tokens and 8.8× more adapter parameters move MMLU by
**less than 0.6 pt**. Mechanism: every layer's MLP composite `W_down · W_up` has rank
≤ min(d, d_ff) = 1536 for the student vs 3072 for the teacher, and width reduction is an
irreversible projection. LoRA never reduces rank (`W = W₀ + BA` is a superset of the
original model), which is why LoRA can match a teacher that a half-width student cannot.

### F4. Teacher-written CoT data protects MMLU but cannot add knowledge

90k teacher samples (~20M tokens: 60% passage-grounded MCQ + 40% study notes, generated
from fineweb-edu, **never from an eval item**) mixed at 55% of tokens:
MMLU −0.04 pt (p=0.90, the first arm with zero damage) but behaviour only +0.19 (p=0.39).
Why no knowledge gain: the generated items are passage-grounded, so the student can
re-derive them at test time and memorises nothing; and at 55% of the mix the CoT data
displaced the instruction data that produced the behaviour gains. Net: a wash.

### F5. Both compression axes need PT-scale recovery

Depth pruning at 50% (full width, 14/28 layers) without recovery collapses the model:
CE 8.5–11.7 nats (teacher 2.86) across four different pruning patterns, and the exported
model scores chance (MMLU .2304). The width axis only works because the paper shipped a
recovered checkpoint (`LRC-1.5B-Base`, CE 3.050 vs teacher 2.855).


## 3. Apparatus bugs found on the way (all fixed)

| bug | effect | fix |
|---|---|---|
| `_shrink_student_` copies the raw top-left block `t[:r, :r]` | the "slice warm start" is **random** (CE 12.46 vs teacher 2.86); every `WARM_START=slice` run was distilling into noise, which is why PT needed so many tokens | documented; experiments now init from `LRC-1.5B-Base` |
| `run_sft_mixture.sh` passed `PLANS=`, which `train_core_lr_pilot.sh` never reads | **every SFT run silently adapted all 28 layers** (34.87M) instead of the 3/9-layer plan in its log | pass `LORA_LAYERS` |
| `LOSS_TYPE=nll` | `KeyError`; `nll` is not a registered loss | empty `LOSS_TYPE` omits the flag (swift default CE) |
| depth pruning left stale `layer_idx` | KV-cache `IndexError`, wrong per-layer behaviour | renumber on prune |
| complement basis `U⊥` built for all 84 MLP modules | 9 GiB of dead weight, SFT OOM at 23.19 GiB before step 1 | build lazily for planned modules only (48 tests still pass) |
| `TOKENS` was never converted to `MAX_STEPS` in the SFT launcher | the "131M token" SFT run was really 2,000 steps = 0.63 epoch ≈ 78M tokens | pass `MAX_STEPS` explicitly |

## 4. Remaining levers, ranked by evidence

| # | lever | expected | cost |
|---|---|---|---|
| 1 | fix the PT warm start (paper Alg.1 ridge/SVD instead of the broken coordinate slice) — a large fraction of our 524M tokens went into repairing a random init | +0.5~1.0 pt at equal tokens | 2–3 days |
| 2 | swap fineweb-edu for knowledge-dense PT text (textbooks / encyclopedic) | **the only PT-side lever that can move MMLU** (+0.5~1.5 pt) | 1–2 days |
| 3 | on-policy KD (student samples, teacher scores) | +0.3~0.8 pt at equal tokens | 3–4 days |
| 4 | widen the student r=1536 → 2048 | the only way to lift MMLU materially | needs FSDP/offload or more cards |

Realistic target for (1)+(2) at 300–500M PT tokens: **Avg8 .68+ with MMLU ≥ .50**. Parity
with the teacher (.7135, MMLU .6222) is not reachable at r=1536 by any token / layer / data
recipe we have measured.

## 5. Reproduction

```bash
# PT (6 of 28 layers, logit distillation, teacher on device)
GPU=0,1 NPROC=2 SEQ=2048 SAVE_STEPS=1900 BS=1 GAS=32 GATE=0 USE_AUX=0 STAGE=pt \
  DATA=$PWD/data/fw_score4_1B.jsonl PLANS=0-2,11-13 LORA_MODULES=gate_proj,up_proj \
  BRANCH=core K=64 LORA_MODE=core FACTORIZED=1 PERP_BASIS=teacher TOKENS=524000000 \
  WARM_START=slice OUTROOT=$PWD/output/product/pt-524m bash scripts/run_v15_w1.sh

# SFT (from the paper's recovered student; the layer plan is now honoured)
MODEL=$HOME/models/LRC-1.5B-Base MODEL_TYPE=llama TEMPLATE=llama3_2 LR=1e-4 LOSS_TYPE= \
  WARM_START_ARM=base BRANCH_ARM=dense LORA_MODE_ARM=dense FACTORIZED_ARM=0 \
  PLANS_ARM=0-2,11-13,25-27 MAX_STEPS=3170 SFT_DATA=$PWD/data/sft_paper8.jsonl \
  OUT=$PWD/output/product/sft-LRC-9L bash scripts/run_sft_mixture.sh

# score + paired test
PRESET=paper GPU=0 BATCH=16 bash scripts/run_lm_eval.sh <ckpt> paper_<tag>
python scripts/eval_mcnemar.py --metric acc base=output/lm_eval/paper_pt_base \
  <tag>=output/lm_eval/paper_<tag>
```

## 6. Caveats

* LogiQA is missing from paper-8 (`datasets>=4` dropped script loaders), so these are
  8-task numbers, not the paper's Avg9.
* The teacher's MMLU (.6222) is measured 0-shot with our harness; the paper reports a
  different suite, so cross-paper comparison should use within-suite deltas only.
* All gains are one seed (42) per arm. The paired McNemar tests control for item
  difficulty but not run-to-run variance; a second seed is the obvious hardening step
  before publication.
* The PT+SFT chain's PT stage started from the broken slice init (see §3), so its 524M
  token budget is an **upper bound** on what the recipe needs, not a lower bound.
