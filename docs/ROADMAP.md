# TWYD on ms-swift：技术复现、CORE-LR、TWYD-MoE 与创业计划

> 目标：在 ms-swift 上把 TWYD 从“论文 + 私有代码”做成可复现、可开源、可融资的技术资产。
> 原则：先让 step-0 初始化过 gate，再谈参数效率；先在小模型上拿到可信的 Pareto 曲线，再上 MoE 和大预算。

## 0. 现状（2026-10）

- 论文：`TWYD.pdf`（Llama3.2-3B→1.5B，r=1536；CORE-LRC 在 Qwen 上 +10.45 Avg9）。
- 当前实现：`twyd/` + `_upstream/ms-swift`，`dense_lrc/core_lrc/plain_lrc` 注册正常，merge 正常。
- 实际 run：`dense-lora128-r1024-ddp-2gpu-100k`，3B teacher、r=1024、LoRA rank=128、SFT-only 45M tokens。
- 结果：step 1 `token_acc=1.9e-4`，1368 步后 13.2%；merged CE≈11.3、token acc=0；9 任务全部随机。
- 根因：width-r 学生由 teacher 权重 top-left slice 构造，不是 LRC warm start；attention/embedding 冻结；aux 失效。

## 0.1 最新结论（2026-10-21）

- **Qwen Table 8 是 CORE 的核心证据，但不是 teacher parity**：
  `Qwen2.5-3B→1.7B, r=2048, ρ=5.375, stranded=81.4%`；teacher 69.11，
  full CORE 10B PT 63.44，PT+SFT 63.82，Dense 10B 59.38，LRC 20B 63.43。
  CORE 的承诺是 **2× token efficiency + 相对 matched LRC +10.45**。
- **Llama 是效率对照**：`3B→1.5B, r=1536, ρ=2.67, stranded=62.5%`；
  论文里 Dense/CORE tie，CORE-LR 目标是“更少坐标打平”，不强求超过 Dense。
- **1B 从零构造仍未过 gate**：slice/MLP/P/attention ridge 都 FAIL；
  现阶段必须先拿官方 LRC base / PT checkpoint，否则 dense/core 对比无效。
- **CORE-LR 低秩假设尚未证明**：下一步的关键 gate 是
  `k<=256` 逼近 full CORE，且可训坐标 `<=25%`、step time/optimizer memory 下降。
- **FlashAttention/AOTriton**：2048 context 用 `twyd` + SDPA + AOTriton 即可，
  不需要依赖 `flash_attn` 包；详见 `README.md` 的环境/2048 节。
- **技术方案**（模型对、四阶段实验、rank allocator、factorized forward、Go/No-Go）
  已写入 `README.md` 的 “CORE-LR 技术方案（2026-10）”。

## 1. 技术判断：TWYD-LR 应该是 CORE-LR，不是 dense-LoRA

论文的可达集论证（r=1536，H=3072，d_ff=8192，L=28，3 个 MLP 投影）：

| 方案 | 每个 MLP 可训坐标 | 总可训 | 占 dense | 更新秩 |
|---|---:|---:|---:|---:|
| plain LRC `T Z_col^T` | rH = 4.72M | 396M | 37.5% | ≤ r |
| Dense-LRC 全矩阵 | r·d_ff = 12.58M | 1.057B | 100% | 满秩 |
| CORE 全补空间 | r(d_ff−H) = 7.86M | 661M | 62.5% | ≤ r |
| **CORE-LR k=128** | 0.85M | **71.6M** | **6.8%** | ≤ k |
| CORE-LR k=256 | 1.70M | 143M | 13.5% | ≤ k |
| CORE-LR k=512 | 3.41M | 286M | 27.1% | ≤ k |
| dense-LoRA k=128 | 1.25M | 105M | 9.9% | ≤ k |

> CORE-LR 假设 col path `T Z_col^T` 已由 plain LRC 冻结/预训练；若同时训 col，加上 plain LRC 的 396M（r=1536）。
> k 的有效上限是 `min(r, d_ff−H)`；k=r 时 `A B^T` 因子化是冗余的，会退化到接近 dense 的参数量。

关键结论：

1. **dense-LoRA 不是 TWYD 哲学**。它训练的是 `W0 + A B^T`，更新秩只有 k；k=128 < r=1536，可达集反而比 plain LRC 更小。它只是“在 dense 权重上做 PEFT”，没有打开 5120 维的 stranded complement。
2. **CORE-LR 是 TWYD-native 的高效变体**：`W = T Z_col^T + U⊥ (A B^T)^T`，所有可训坐标都在 `col(T)^⊥`，merge 后仍是一个 `d_ff×r` dense 矩阵，推理零新增。
3. 同一 k 下，CORE-LR 比 dense-LoRA 更省（`d_ff−H=5120 < d_ff=8192`），而且方向选在“老师有、plain LRC 学不到”的补空间，假设更干净。
4. 研究方向：**CORE-LR rank sweep + 层级 rank 分配 + 等参数对照（把同样的坐标限制在 col(T)）**，才能回答“增益来自可达集还是参数量”。

## 2. Phase 0（1–2 周）：让初始化过 gate

验收标准（`scripts/check_init.py`）：

- student CE ≤ 6.0（teacher≈2，均匀≈11.76）
- 逐层 mean cosine ≥ 0.5（当前 r=1536 只有 0.144）
- token_acc ≥ 0.20

任务：

- [x] 实现论文 Alg.1 / App. A.4 的 MLP warm start：
  - [ ] embedding SVD → `P ∈ R^{r×H}`：`build_master_P` + `project_student_with_P_` 已实现（head-aware），但 1B 真实 gate 仍未通过；默认关闭
  - 逐层 greedy：teacher/student 同层前向，streaming `C^T C` / `C^T H_T`，拟合 deployed `W` 后分解 `Z_col` / `Z_perp`
  - `scripts/calib_ridge.py --synthetic` 与 `tests/test_ridge_calib.py` 已通过（38 tests）
  - 训练路径：`warm_start="ridge"`（在线校准）或 `warm_start="ridge_ckpt"`（离线 `.pt` / merged HF）
  - `pt_ckpt` SFT 现在也会保留 PT 合并权重里的补空间残差（`ridge_perp != zero`）
- [ ] 真实 1B gate 结论（2026-10-20）：slice CE 11.90/cos 0.168 → MLP ridge 11.15/0.219，仍 FAIL；P select 11.95/0.146；P merge 11.76/0.171；P+ridge 15.60/0.114；P+attn ridge+MLP ridge 15.88/0.117。**线性 MLP/P/attention ridge 都不足以从零构造学生，attention 语义耦合是 blocker**；优先用官方 LRC base，或实现论文的 attention zoom / LRC co-training
- [ ] attention 用 LRC 的 zoom/ridge；当前只校准 MLP gate/up/down（论文 App. B 的默认 target）
- [ ] 若时间紧：直接用官方 LowRankClone `CoTrainLM` + `convert_ckpt.py` 产出一个 width-r checkpoint，作为 `--base`/`--pt_ckpt`
- [ ] 固定 `r=1536`（不是 1024）；`--model` 用 3B（或官方 1.5B LRC ckpt），不要 1B+3B
- [ ] 加 logit parity：live wrapper vs reload merged，`max_abs_diff < 1e-2`


## 3. Phase 1（2–4 周）：TWYD-LR / CORE-LR 实验矩阵

先在小模型上跑通，再放大。建议 pilot：Llama-3.2-1B teacher → r=512/768 student，0.2–0.5B tokens；或 3B→1.5B（r=1536）0.3B tokens 的短跑。

### 3.1 变体

| 代号 | 参数化 | 目的 |
|---|---|---|
| P0 | plain LRC `T Z_col^T` | 官方 baseline |
| P1 | Dense 全矩阵 | 论文主方法（上限） |
| P2 | CORE full `Z⊥` | 论文高效实现 |
| P3 | **CORE-LR(k)** `Z⊥=A B^T`，k=64/128/256/512/1024 | 你的核心创新：效率前沿 |
| P4 | dense-LoRA(k) | PEFT 对照（不是 TWYD-native） |
| P5 | CORE-LR + 层级 rank 分配 | 第二个创新点 |
| P6 | 等参数对照：同样 k 的坐标限制在 col(T) | 排除“纯加参数”解释（论文 §6.1） |

### 3.2 指标

- 语言建模：CE / token_acc / PPL（teacher、P0–P6 同一 batch）
- 下游：Avg9（ARC-E/C、LogiQA、CSQA、PIQA、WinoGrande、BoolQ、SciQ、MMLU），0-shot、同一 lm-eval 版本
- 成本：可训坐标、GPU 小时、峰值显存、tokens-to-target
- 结构：`u = 可训坐标 / (r·d_ff)`、每层 rank、merge parity、生成退化检查
- 统计：至少 2 seeds；报告均值/极差；单 seed 只用于 debug

### 3.3 关键消融

1. **rank sweep**：CORE-LR k 从 64 到 r，画“eval vs 可训坐标”的 Pareto 曲线；预期在 k≈256–512 出现拐点。
2. **层级分配**：用校准集的梯度/激活 eRank 估计每层补空间的重要性，`k_l ∝ score_l`，总预算固定；对比 uniform rank。
3. **等参数对照 P6**：同样的 k、同样的参数量，但限制在 col(T)，验证增益来自补空间而不是参数量。
4. **basis 选择**：teacher SVD `U⊥` vs 随机正交补 vs 数据感知 P；验证是否“basis 不 privileged”（论文结论）。
5. **SFT vs PT**：先 PT 再 SFT；不要用 SFT-only 45M token 下结论。

### 3.4 现实算力

当前 2×7900 XTX：45M tokens 约 2.3h。0.5B tokens ≈ 25h；1B ≈ 50h；10B ≈ 500h（约 21 天）。
所以：

- Phase 1 用 0.2–0.5B tokens 的 pilot（1B teacher / r=512–768），先看趋势；
- 复现论文 3B→1.5B 的 10B PT 需要租 A100/H100 或融资后做，不要在 2 卡上硬跑；
- 商业计划书里写“pilot 的 Pareto 曲线 + 租赁算力的 scaling 预算”，不要承诺 2 卡 10B。

## 4. Phase 2（6–10 周）：TWYD-MoE

### 4.1 设定

先做 **MoE teacher → 更窄 MoE student**（保留专家数与路由），而不是 dense→MoE。候选小模型：

- OLMoE-1B-7B（64 experts, top-8）：单机可跑，社区 eval 完整
- Qwen1.5-MoE-A2.7B（60 experts, top-4）：中文/多语言友好
- DeepSeek-MoE-16B：更大，融资后

### 4.2 研究点

1. **per-expert reachability**：每个 expert 的 MLP 都是 `d_ff×r`；统计每个 expert 的 eRank / 负载 / 补空间能量，验证“长尾 expert 的 stranded capacity 更大还是更小”。
2. **router-aware CORE-LR**：`k_e ∝ load_e` 或 `k_e ∝ gradient_e`；总补空间预算固定；对比 uniform-k。
3. **shared expert 优先**：shared expert 一直激活，先开它的 complement；expert 层再按负载分配。
4. **mergeable**：训练后把 `U⊥ Z⊥^T` 折进每个 expert 的 `d_ff×r` 权重；推理 FLOPs/显存不变，路由不变。
5. **router 蒸馏**：student router 输入维度变了，需要 P；用 teacher 的 top-k routing distribution 做 KL，加 load-balance 正则。
6. **basis 共享**：每 expert 存 `U⊥` 内存爆炸（L×E×d_ff×(d_ff−H)）；研究 expert 间子空间重叠，尝试共享 basis + per-expert coefficients，或按 expert cluster 共享。

### 4.3 验收

- 同参数量下，MoE student 的 Avg9 / 困惑度 / 专家负载不劣化；
- 训练显存/时间可接受（至少 1B 级 MoE pilot）；
- merge 后推理吞吐与原始 width-r MoE 一致；
- 给出“每 expert 可训坐标 vs 增益”的曲线，作为第二个商业 proof point。

## 5. Phase 3：产品化 / 开源 / 融资

### 5.1 IP 与合规（先做，不要跳过）

- 你和前司的劳动合同/知识产权归属条款；TWYD 论文、代码、实验记录是否属于职务成果；
- `_upstream/LowRankClone` 的 license（读 LICENSE，不要默认 MIT/Apache）；ms-swift 是 Apache-2.0；
- 创业公司代码要 clean-room：只依据公开论文 + 公开 repo 的接口/算法重写，不复制前司私有代码/数据/权重；
- 找律师做一次 IP opinion；融资尽调一定会问“论文归谁、代码是否原创、训练数据授权”。
- 在 README/仓库中记录 provenance（论文、公开 repo、license、独立实现声明）。

### 5.2 开源策略

- 开源：ms-swift plugin、CORE-LR 实现、eval/复现脚本、pilot 权重（如 license 允许）；Apache-2.0；
- 保留：数据管线、MoE 调度/rank 分配、评测 harness、领域数据与客户适配；
- 不做“又一个 LoRA”叙事；做“可证明的 reachability + 零推理开销 + 可合并”的压缩/蒸馏工具链。

### 5.3 融资叙事

- 问题：企业要在固定推理成本下提升模型能力；MoE/大模型的训练参数浪费在不可达方向；
- 方案：TWYD / CORE-LR 在**不增加推理成本**的前提下，把训练预算花在“部署矩阵真正能到达”的方向；
- 证据：同一 harness 的 Pareto 曲线（plain/dense/CORE/CORE-LR）、tokens-to-target、GPU 小时、merge parity；
- 壁垒：评测/数据管线 + MoE rank 分配 + 可复现的工程实现；不是单个公式；
- 里程碑：M0 gate + pilot（1–2 月）→ M1 CORE-LR Pareto（2–3 月）→ M2 MoE pilot（4–6 月）→ 融资/客户 POC。

> 详细的客户、产品形态、商业模式、90 天/12 个月里程碑、融资与护城河见 §7。

## 6. 本周就做的 5 件事

1. 跑 `scripts/check_init.py`，把 r=1536 的 FAIL 记录成 issue；在 gate 通过前冻结所有训练。
2. 选 A 或 B：A=官方 LRC convert 产出 1.5B ckpt；B=实现 Alg.1 ridge。先做 A，最快拿到可信 baseline。
3. 固定 pilot：1B teacher→r=512/768，0.2B tokens，plain vs dense vs CORE vs CORE-LR(128/256/512)，2 seeds。
4. 加 `scripts/param_budget.py` 到实验报告；每个 run 记录可训坐标、GPU 小时、merge parity、CE/acc。
5. 写 2 页技术 memo（给融资/合作方）：问题、方法、Pareto 图、复现命令、IP 声明。

## 7. 创业计划（CORE-LR 楔子，TWYD-MoE 扩张）

> 技术是楔子，不是公司。公司卖的是“**同样的任务质量，更低的训练/部署成本**”，
> 不是“一个 rank 公式”。

### 7.1 一句话定位

> **TWYD 让企业用更少的训练参数和显存，把 Qwen/Llama 压缩成可部署的小模型；
> 训练只学关键补空间，推理零额外开销，并可扩展到 MoE。**

### 7.2 客户与楔子

优先级：

1. **私有化/行业模型团队**：有 Qwen/Llama 教师、需要 1.5B–2B 级可部署学生；
2. **中文企业知识库/客服/金融/医疗**：成本敏感、数据不能出域；
3. **边缘/端侧**：参数量、显存、推理延迟有硬约束；
4. **MoE 团队**：已有 MoE 教师，想把 expert MLP 的 stranded capacity 用起来。

第一个 wedge 不要做“通用大模型压缩平台”，而是：

```text
一个行业 + 一个教师模型 + 一个 1.5B–2B 学生 + 一套 eval
```

### 7.3 产品形态

- **开源核心**：ms-swift plugin、CORE-LR、merge/eval/复现脚本（Apache-2.0）。
- **商业层**：
  - rank allocator（per-layer / per-projection / per-expert）；
  - factorized CORE kernel（真实训练成本下降）；
  - MoE router-aware allocation；
  - 领域数据/校准/评测管线；
  - 企业适配与支持。

### 7.4 必须拿出的 proof points

1. **Qwen2.5 3B→1.7B r=2048**：
   - full CORE 10B PT 63.44 / PT+SFT 63.82（论文参考）；
   - CORE-LR(k<=256) 与 full CORE 差距 <=0.5；
   - 可训坐标 <= full CORE 的 25%；
   - step time 或 optimizer memory 明显下降。
2. **Llama 3.2 3B→1.5B r=1536**：
   - CORE-LR 用更少坐标打平 Dense/full CORE；
   - 证明方法不是 Qwen 特例。
3. **部署**：
   - merge 后是普通 dense width-r 模型；
   - 推理框架零改动；
   - prefill/decode 延迟、显存、吞吐有实测。
4. **MoE pilot**：
   - per-expert rank 优于 uniform rank（同总预算）；
   - merge 后路由/输出一致。

### 7.5 商业模式

- **License**：企业版插件 + rank allocator + MoE 模块；
- **压缩服务**：按模型/项目交付，包含 eval 和部署验证；
- **订阅/支持**：模型升级、数据/评测管线、私有化部署；
- **联合研发**：与云厂商/模型厂商/行业 ISV 合作。

不要按“训练 token 数”收费；按“交付的模型质量 + 成本节省 + 部署价值”收费。

### 7.6 里程碑

#### 0–90 天（Pre-seed 证据）

- G0：官方 LRC base 过 gate；
- G1：Qwen 代理 full CORE 复现 CORE > Dense；
- G2：CORE-LR(k<=256) 逼近 full CORE；
- G3：factorized forward 让 step time/optimizer memory 下降；
- 2 个 design partners 的领域 eval；
- 一页 Pareto 图 + 一页成本模型。

#### 3–6 个月（Seed 证据）

- Qwen/Llama 目标规模 0.2–0.5B token 放大；
- 一个客户 POC：同质量、训练成本降低、部署零额外开销；
- MoE pilot：per-expert rank 可行；
- 开源 repo 有外部用户/issue/PR。

#### 6–12 个月（A 轮叙事）

- 2–3 个付费客户/LOI；
- MoE 压缩 recipe；
- 模型 zoo（Qwen/Llama/领域模型）；
- 可复现的 tokens-to-target / $/quality 证据。

### 7.7 融资阶段与资金用途

- **Pre-seed**：技术 de-risk + 2 个 design partners；资金用于 GPU/云、1–2 名工程师、IP/法务。
- **Seed**：CORE-LR 目标规模 + MoE pilot + 客户 POC；资金用于算力、BD、评测/数据。
- **A 轮**：模型 zoo + 商业化；资金用于销售、算力、MoE 规模化。
- 不要在 2×7900 XTX 上承诺 10B 全量；把“租赁算力/云 GPU”写进预算。

### 7.8 护城河

公式是公开的，CORE 本身不是护城河。护城河是：

1. **rank allocator**：per-layer / per-projection / per-expert 的预算分配；
2. **factorized CORE execution**：真实训练成本下降的工程实现；
3. **MoE router-aware allocation**：expert 负载/梯度/子空间共享；
4. **评测/数据/校准管线**：可复现的 Pareto 和客户 eval；
5. **ms-swift/HF 集成**：一键训练、merge、部署；
6. **模型 zoo / 领域 recipe**：Qwen/Llama/行业模型的可复用配方。

### 7.9 风险与 Go/No-Go

| 风险 | 触发条件 | 应对 |
|---|---|---|
| CORE-LR 低秩不成立 | k 必须 1024–2048 才接近 full CORE | 退回 full CORE/服务化，或改结构化低秩 |
| Qwen teacher gap 被误读 | 把 63.8 当成应到 69.1 | 明确“vs full CORE/20B LRC”指标 |
| attention/embedding 是瓶颈 | 逐层 cosine 在早期就崩 | attention zoom / LRC co-training / 容量 |
| 2 卡算力不足 | 10B 全量不可行 | 代理模型 + 论文 checkpoint + 租卡 |
| 前司 IP 风险 | 代码/权重/数据来源不清 | clean-room、license 审核、律师 opinion |
| MoE 分散主线 | dense 未过 G2 就做 MoE | 冻结 MoE，直到 G2/G3 通过 |

### 7.10 不做什么

- 不做“又一个 LoRA”；
- 不承诺 Qwen teacher parity；
- 不把“参数少”直接等同于“推理快”；
- 不在 dense CORE-LR 未证明前做大规模 MoE；
- 不使用前司未公开的代码/权重/数据；
- 不在 2 卡上硬跑 10B 全量。

### 7.11 一页 pitch

```text
问题：企业要在固定推理成本下提升模型能力；full CORE/大模型训练浪费在不可达方向。
方案：CORE-LR 只学关键补空间 + rank allocator + factorized execution；
      部署仍是普通 dense 小模型，推理零额外开销；MoE 按 expert 分配 rank。
证据：Qwen 3B→1.7B full CORE 10B 追平 20B LRC；CORE-LR k<=256 逼近 full CORE；
      Llama 作为效率对照；merge parity + 实测训练/推理成本。
壁垒：allocator + factorized kernel + MoE + eval/data + ms-swift 集成。
阶段：Pre-seed 证据（90 天）→ Seed 客户 POC（6 个月）→ A 轮模型 zoo/MoE（12 个月）。
```
