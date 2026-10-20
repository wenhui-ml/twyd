# twyd
Train What You Deploy: Closing the MLP Reachability Gap in Low-Rank Clone Distillation

`twyd` 在 [ms-swift](https://github.com/modelscope/ms-swift) 之上实现
[Low-Rank Clone (LRC)](https://github.com/CURRENTF/LowRankClone) 风格的 low-rank clone 蒸馏，
补上 MLP reachability gap：把训练对象从 teacher slice 换成整个 deployed matrix。

## 原理（TWYD.pdf §4 / A.5）

- Plain-LRC：`W = T Z_col^T`，`T: d_ff×H` 冻结，只训 `Z_col: r×H`；
  利用率 `u = 1/ρ`，`ρ = d_ff/d_model^(T)`（2.67 / 3.50 / 5.375）。
- Dense-LRC（§4.1）：直接训整块 `d_ff×r` dense，warm start 在 merged plain-LRC 上；推理零新增。
- CORE-LRC（§4.2）：`W = T Z_col^T + U⊥ Z⊥^T`，`[T U⊥]` 完备正交，`Z⊥` 零初值；
  gate/up 左补，down 右补（`Z⊥ V⊥^T`）；merge 后与 Dense 同形同权重（Prop.1）。
- Loss（Eq.6-7）：`T^2·KL(T=40) + CE + λ_aux·L_align`，`λ_aux=0.2`。

## CORE-LR 技术方案（2026-10）

> 目标：把论文的 **full CORE** 推进到 **CORE-LR(k)**——用更少的可训坐标逼近
> full CORE 的质量，并让训练成本真实下降。**不是**用 rank 去追 teacher parity。

### 1. 目标与边界

- **Qwen2.5 是主战场**：`3B→1.7B, r=2048`，`d_ff/H=5.375`，`stranded=81.4%`；
  宽 MLP 让 full CORE 比 Dense 高约 4 Avg9，是 CORE-LR 最该拿下的 regime。
- **Llama 3.2 是效率对照**：`3B→1.5B, r=1536`，`d_ff/H=2.67`，`stranded=62.5%`；
  论文里 Dense/CORE tie，CORE-LR 的目标是“更少坐标打平”，不强求超过 Dense。
- **teacher parity 不是本项目指标**：Qwen Table 8 的 full CORE 也只有 63.44/63.82，
  teacher 是 69.11；CORE-LR 的目标是逼近 full CORE，而不是 69.11。
- **k 只影响训练成本**：部署时 merge 成 width-r dense 权重，推理延迟/显存由
  width-r student 决定；CORE-LR 不增加推理开销，也不因 k 变小而额外加速推理。

### 2. 论文参考点（Qwen2.5 3B→1.7B，r=2048）

| 模型 | Avg9 | 说明 |
|---|---:|---|
| Teacher Qwen2.5-3B | 69.11 | 上限 |
| Full CORE 10B PT | 63.44 | CORE 参考 |
| Full CORE 10B PT+SFT | 63.82 | CORE 参考 |
| LRC ~20B PT | 63.43 | CORE 在 10B 追平 20B LRC |
| Dense 10B PT | 59.38 | 同 token 对照 |
| LRC 10B PT | 52.99 | matched baseline |

结论：CORE 的承诺是 **2× token efficiency + 相对 matched LRC +10.45**，
不是 teacher parity。CORE-LR 的成功标准是：

```text
CORE-LR(k<=256) ≈ Full CORE (63.4/63.8)
且可训坐标 <= Full CORE 的 25%
且 step time / optimizer memory 明显下降
```

### 3. 模型对与算力现实

| 用途 | Teacher | Student | r | 说明 |
|---|---|---|---:|---|
| Qwen 精确目标 | Qwen2.5-3B | 1.7B | 2048 | 优先用官方 LRC/PT base |
| Qwen 2 卡代理 | Qwen2.5-1.5B | width-r | 1024 | 保留宽 MLP 几何，适合测谱/rank |
| Llama 精确目标 | Llama-3.2-3B | 1.5B | 1536 | 需要官方 LRC/PT base |
| Llama 2 卡代理 | Llama-3.2-1B | width-r | 512 | full CORE 约 150M 可训，适合 pilot |

2×7900 XTX 的现实预算（参考 `docs/ROADMAP.md`）：45M tokens ≈ 2.3h，
0.2B ≈ 10h，0.5B ≈ 25h，10B ≈ 500h。**不要在 2 卡上硬跑 10B**；用论文
checkpoint 或租卡做最终放大。

### 4. 四阶段实验

#### Stage 0：gate（0 token）

```bash
python scripts/check_init.py \
  --teacher /path/to/teacher \
  --model /path/to/student_base \
  --r 1024 --branch core --seq-len 128
```

必须 `CE <= 6.0`、`mean cos >= 0.5`、`token_acc >= 0.20`。不过 gate 就先换
官方 LRC base / PT checkpoint；否则 dense/core/CORE-LR 的差异会被坏初始化淹没。

#### Stage 1：小规模 full CORE，测真实补空间谱

在代理模型上跑 20–50M tokens 的 full CORE（`TWYD_LORA_RANK=0`, `branch=core`），
然后用：

```bash
python scripts/measure_complement_rank.py \
  --student output/full-core-pilot \
  --teacher /path/to/teacher \
  --layers all --topk 32,64,128,256,512,1024 \
  --json-out output/rank_spectrum.json
```

得到每层/每投影 `Z_perp` 的 top-k energy、eRank、stable rank。这是 rank 的金标准；
没有 full CORE checkpoint 时，才退回梯度/激活先验。

#### Stage 2：CORE-LR(k) sweep

同一 base、同一 token 预算：

```text
k = 64, 128, 256, 512
```

记录 CE/token_acc、Avg9 子集、可训坐标、optimizer memory、peak memory、
step time、merge parity。判断 k 是否能在 `<=256` 内逼近 full CORE。

#### Stage 3：rank allocator + factorized forward

- 用 Stage 1 的 `E_{l,p}(k)` 做预算分配：

```text
maximize  sum E_{l,p}(k_{l,p})
s.t.      sum k_{l,p}(r + d_ff-H) <= K_total
```

- 当前只支持全局 `TWYD_LORA_RANK`；下一步加 `TWYD_RANK_MAP`（per-module k）。
- 实现 cached `W_col` + factorized forward：

```text
y = F.linear(x, W_col)
  + ((x @ A) @ (B.T @ basis_perp.T)) * scaling
```

否则 k 只省参数/优化器，不省主计算，融资故事不完整。

#### Stage 4：放大与 MoE

- 代理模型验证后，在 Qwen2.5-3B→1.7B / Llama-3.2-3B→1.5B 上做 0.2–0.5B token
  的放大验证；
- dense CORE-LR 通过 gate 后，再做 MoE：per-expert `k_e`、router-aware allocation、
  merge parity；
- 不在 dense 未证明前把 MoE 当主线。

### 5. 代码改造清单

1. **ModelSpec 适配层**：Llama/Qwen 的 target modules、attention bias、tie、
   GQA/kv_map、vocab、merge/logits parity。
2. **per-module rank map**：`TWYD_RANK_MAP` + `CoreLinearLoRA` 按模块名取 k。
3. **rank allocator**：读 `rank_spectrum.json`，贪心/背包分配，输出 rank map。
4. **factorized forward**：缓存 frozen `W_col`，只对低秩项建 autograd 图。
5. **eval harness**：CE/token_acc、Avg9 子集、tokens-to-target、peak memory、step time。

### 6. Go / No-Go

| Gate | 标准 |
|---|---|
| G0 初始化 | CE <= 6.0, cos >= 0.5 |
| G1 full CORE | Qwen: CORE > Dense；Llama: CORE ≈ Dense |
| G2 低秩假设 | k <= 256，与 full CORE 差距 <= 0.5，坐标 <= 25% |
| G3 成本 | step time 降低 >= 1.3x 或 optimizer memory 降低 >= 30% |
| G4 MoE | 只有 G2/G3 通过才启动 |

## 结构

```
twyd/  config.py core_basis.py tuner.py loss.py init.py merge.py plugin.py
scripts/ train_dense.sh train_core.sh convert_ckpt.py
tests/ test_core_equiv.py
_upstream/  只读参考（ms-swift + LowRankClone，不导入）
```

## 用法

```bash
pip install -e /home/chenwh/Papers/twyd/_upstream/ms-swift
pip install -e /home/chenwh/Papers/twyd
swift pt --model <student> --dataset <data> \
  --tuner_type core_lrc --loss_type lrc_distill \
  --external_plugins twyd.plugin \
  --tuner_kwargs '{"teacher_model":"<teacher>","student_hidden":1536}'
# Dense 把 core_lrc 换成 dense_lrc；plain_lrc 为 matched 基线
```

### 训练前先过 step-0 体检（必须 PASS）

```bash
python scripts/check_init.py \
  --teacher /home/chenwh/models/Llama-3.2-3B-Instruct \
  --model   /home/chenwh/models/Llama-3.2-3B-Instruct \
  --r 1536 --branch dense --seq-len 256
```

脚本会打印 teacher/student 的 CE、token acc 和逐层 hidden cosine。当前
“teacher 权重 top-left slice” 初始化在 3072→512/1024 上会直接 FAIL
（student CE≈11.8≈均匀分布，第 3 层 cos≈0.15），说明 warm start 没有跟踪
teacher；这种情况下继续训练只会学到高频 token，评测必接近随机。通过
P/ridge（论文 Alg.1 / App. A.4）或 LRC co-train→convert 的 warm start
后再开训。

### Zero-PT tensor-space warm start（Alg.1 / App. A.4）

`scripts/calib_ridge.py` 已实现 **流式、逐层 greedy 的 ridge 校准**：
只缓存 `C^T C` 和 `C^T H_T`，显存与校准 token 数无关；拟合 deployed `W`
后精确分解成 `Z_col`（col path）和 `Z_perp`（补空间残差）。这是
**校准分布上的最小二乘等价**，不是对所有输入的精确函数等价；`r < H`
的压缩残差正是 `Z_perp` 和后续训练要处理的部分。

```bash
# 1) 无 HF 下载的数学 sanity check
python scripts/calib_ridge.py --synthetic

# 2) 从 teacher 直接构造 width-r student，校准并保存 merged HF + ridge_weights.pt
#    先准备几百条 packed sequences（jsonl 里放 {"text": ...}）：
python scripts/make_calib_ids.py --tokenizer /path/to/Llama-3.2-3B \
  --input calib.jsonl --num-seqs 200 --seq-len 1024 --out outputs/calib_ids.pt

python scripts/calib_ridge.py \
  --teacher /path/to/Llama-3.2-3B \
  --r 1536 --branch core --perp residual \
  --teacher-device cpu \
  --calib-ids /path/to/calib_ids.pt \
  --num-seqs 200 --seq-len 1024 \
  --out outputs/ridge-r1536
# 24GB 卡上建议加 --teacher-device cpu，避免 teacher+student+basis 同时占 GPU

# 3) 训练时在线校准（不需要离线 ckpt）
TWYD_WARM_START=ridge \
TWYD_CALIB_IDS=/path/to/calib_ids.pt \
TWYD_RIDGE_PERP=residual \
swift pt --model /path/to/teacher --tuner_type core_lrc ...

# 4) 使用离线校准结果（保留补空间残差）
TWYD_WARM_START=ridge_ckpt \
TWYD_RIDGE_CKPT=outputs/ridge-r1536/ridge_weights.pt \
swift pt --model /path/to/width-r-base --tuner_type core_lrc ...
```

- `--master-p embed_svd` 可选：先做 embedding-SVD 的全局 P 投影（head-aware），
  但在 1B 上单独 P 仍过不了 gate，默认关闭；不要把它当作 zero-PT 的充分条件。
- `--calib-attn` 可选：在 MLP 之前对 q/k/v/o 做 head-projected ridge；1B 上
  与 P/MLP 组合仍 FAIL，默认关闭，仅作对照。
- `--perp zero` 复现论文默认（`Z_perp=0`，step 0 精确等于 plain LRC）；
  `--perp residual` 把 teacher 残差写进补空间（更快，但不再是 strict plain）。
  日志的 `rel before -> deployed (fit=...)`：`deployed` 是合并后真实权重的误差，
  `perp=zero` 时它可能明显高于 `fit`（down 尤其明显）。
- 校准数据只需几百条 packed sequences；`.pt/.npy/.jsonl/.txt` 均可，
  文本文件会自动加载 teacher tokenizer。
- `--joint`（默认）每层一遍前向；`--greedy` gate/up 更新后重跑 down，
  更精确但慢一倍。
- `warm_start="ridge"` 只校准 MLP gate/up/down（论文 App. B 默认）；
  attention/embedding 的 zoom/ridge 仍是下一步。
- 一个重要的结构性事实：teacher 的 gate/up 输出天然落在 `col(T)` 内，
  所以校准残差 `Z_perp` 在 gate/up 上通常≈0；补空间容量主要靠训练打开。
  down 的目标是 `H→r` 的投影，残差会填充 `V⊥`，因此 residual init 对
  down 更明显。不要因为 gate/up 的 `z_perp=0` 就认为校准失败。

### 官方 LRC 权重和 TWYD SVD/ridge 的关系（重要）

官方 LRC 发布的 1.5B 权重本质上是一个 **hidden-size 减半的 dense student**：
它的 MLP 是 `d_ff × r` 的普通权重，**没有** `T`、`U⊥`、`Z_col`、`Z_perp`，
也没有 SVD/ridge。SVD/ridge 是 TWYD 的后处理，不是官方权重自带的。

TWYD 做的事情是：拿 teacher 的 SVD 基 `[T U⊥]`（它在 `R^{d_ff}` 里是完备的），
把任意 width-r 的官方权重**精确重写**成

```
W_official = T Z_col^T + U⊥ Z_perp^T
```

这是坐标变换，不要求官方模型当初用 SVD 训练。随后可以选择：

- `warm_start="base"`：**原样保留**官方权重（包括 `Z_perp`），只做精确重分解，
  然后训 CORE-LR/补空间；这是推荐的 zero-PT 路径。
- `warm_start="ridge"`：在校准集上重新拟合 MLP，会**覆盖**官方 MLP；
  这是 TWYD 的可选重初始化，不是“官方 LRC 自带 ridge”，默认不建议直接用在
  已经训练好的官方 base 上。

官方 LRC base + CORE-LR 的正确用法：

```bash
# 1) 如果是 CoTrainLM/zoom 格式，先用 scripts/convert_ckpt.py 转成 merged HF
# 2) 用 width-r 官方 checkpoint 作为 --model，warm_start=base 保留权重
TWYD_WARM_START=base \
TWYD_R=1536 \
TWYD_FREEZE_COL=1 \
TWYD_LORA_RANK=128 \
TWYD_LORA_MODE=core \
swift pt --model /path/to/official-lrc-r1536 --tuner_type core_lrc ...
```

`warm_start="base"` 只做重分解，不改 deployed 权重；`freeze_col=1` 冻结官方
col path，只训低秩补空间。若想完整 CORE 训练，把 `freeze_col` 设为 0。

### 7900 XTX / ROCm FlashAttention

`twyd-flash` 里已经装了 `flash_attn` 的 **Triton 后端**（gfx1100），LRC 训练可以这样启用：

```bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export SWIFT=/home/chenwh/miniconda3/envs/twyd-flash/bin/swift
export PYTORCH_ALLOC_CONF=expandable_segments:True
ATTN_IMPL=flash_attention_2 bash scripts/train_core_lr_pilot.sh
```

脚本默认 `ATTN_IMPL=sdpa`，不会改变现有行为；`--attn_impl` 会被
`_shrink_student_` 保留（否则 width-shrink 会退回 teacher 的默认 attention）。

在 7900 XTX 上的实测（Llama-3.2-1B，r=512，seq=2048，batch=1）：

> **注意：下面这组数字是在 GPU1 仍被其他任务占用（GPU use ~75%）时测的，
> 不是空闲 GPU 的干净 benchmark；绝对毫秒数会受共享负载影响，只能作方向性参考。**
> `peak` 是当前进程的 `max_memory_allocated`，受外部任务影响较小。

| attention | step | peak |
|---|---:|---:|
| `sdpa` + AOTriton | 192.6 ms | 6.34 GiB |
| `flash_attention_2` (Triton) | 220.5 ms | 6.34 GiB |
| `eager` / SDPA-math | OOM | >24 GiB |

**为什么整步显存没降？** 因为 `sdpa` 本身已经走 AOTriton flash，且整步峰值不在
attention，而在 loss/backward。r=512、seq=2048 的逐阶段峰值两种后端完全相同：

```text
forward  3.377 GiB
loss     5.334 GiB   # float32 logits（vocab=128k）
backward 6.312 GiB   # MLP d_ff=8192 激活 + 梯度
```

纯 attention 对照（8 heads、d=64、seq=4096、bf16、前向+反向）则能看出 flash 的价值：

```text
SDPA math       2.102 GiB
SDPA AOTriton   0.036 GiB
flash_attn      0.043 GiB
```

所以在 LRC 里，flash 只在 **attention 物化 S×S 会 OOM** 时救场；当 SDPA 已经使用
AOTriton 时，它不会降低整步峰值，也不会额外提速。真正的显存大头是
**teacher/student 权重、d_ff=8192 的 MLP 激活、以及 float32 logits/loss**。

> 速度数字（192.6/220.5 ms）是在 GPU1 仍有其他任务时测的，绝对值受干扰；
> 这里只能确认“整步显存峰值相同”，速度排名需空闲 GPU 成对复测。

### 环境选择：twyd vs twyd-flash

- **`twyd` 就够用于 LRC 训练 + SDPA/AOTriton flash**：它有
  `torch 2.9.1+rocm6.4`、`aotriton.images/amd-gfx11xx`、`twyd` 和 `swift`。
- **`twyd-flash` 不是 AOTriton 的必要条件**：它只是额外装了
  `flash_attn` 的 Triton 后端（用于 `flash_attention_2` / `padding_free` /
  `packing` / CK 编译实验）；目前没有原生 CK `.so`。
- 因此：`ATTN_IMPL=sdpa` 用 `twyd`；只有 `ATTN_IMPL=flash_attention_2`
  或要测 `flash_attn` API 时才需要 `twyd-flash`。

### 2048 context 的建议

2048 不需要额外安装 `flash_attn`：PyTorch SDPA 的 AOTriton 后端已经提供
FlashAttention 的 `O(S)` attention 显存。主要脚本支持 `MAX_LENGTH`：

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
MAX_LENGTH=2048 ATTN_IMPL=sdpa \
R=512 K=128 bash scripts/train_core_lr_pilot.sh
```

2048 时真正的显存大头仍是 MLP（`d_ff=8192`）、float32 logits（vocab=128k）、
teacher 和梯度；建议保持 `gradient_checkpointing=true`，teacher 放 CPU，
必要时 `TWYD_USE_AUX=false` 或减小 micro-batch/用 Adafactor。

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
ATTN_IMPL=sdpa bash scripts/train_core_lr_pilot.sh
```

只有需要 `flash_attn` / `padding_free` / `packing`，或 SDPA 退回 math 导致 OOM 时，
再用 `flash_attention_2`。

### 真实 1B gate 结果（2026-10-20，Llama-3.2-1B → r=512）


同一 79-token 文本、同一 tokenizer、`branch=dense`：

| 初始化 | student CE | mean cos | 备注 |
|---|---:|---:|---|
| slice（top-left） | 11.90 | 0.168 | hidden norm 16–20，attention 后爆掉 |
| MLP ridge（无 P） | 11.15 | 0.219 | CE 有改善，但仍 FAIL |
| embedding-SVD P only（select） | 11.95 | 0.146 | norm 正常（0.68–3.9），方向仍不对 |
| embedding-SVD P only（merge heads） | 11.76 | 0.171 | 比 select 略好，仍 FAIL |
| P + MLP ridge | 15.60 | 0.114 | 局部 ridge 破坏全局 residual，反而更差 |
| P + attention ridge + MLP ridge | 15.88 | 0.117 | 逐层 q/k/v/o ridge 仍不等于 attention 语义 |

结论：**MLP-only ridge、全局 P、以及逐层 q/k/v/o ridge 都不足以从零构造一个可训练的学生**；
真正的 blocker 是 attention 的 head reduction/softmax 与逐层非线性耦合，线性 ridge
无法保持 attention 语义。`master_p` 和 `--calib-attn` 保留为实验选项，默认关闭；
zero-PT 的现实路径是“官方 LRC width-r base + `warm_start=base` + CORE-LR”，
或者实现论文的 attention zoom / LRC co-training。若在官方 base 上用 `ridge`，
那是在 TWYD 坐标系里重拟合，不是官方权重自带的能力。


详见 `twyd/config.py`（`tuner_kwargs` 直透）与 `scripts/`。
