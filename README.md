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
