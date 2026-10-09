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

详见 `twyd/config.py`（`tuner_kwargs` 直透）与 `scripts/`。

