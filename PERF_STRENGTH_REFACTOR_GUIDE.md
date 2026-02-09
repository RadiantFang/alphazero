# AlphaZero9 重构执行手册（性能 + 棋力）

> 适用项目：`/mnt/data/alphazero9`  
> 文档目标：你打开后可以直接照着改、照着跑、照着验收。  
> 阅读顺序：`1.一页速览 -> 2.直接可用命令 -> 3.逐项改造说明 -> 4.A/B验收模板`

---

## 1. 一页速览（先看这个）

## 1.1 你当前已经完成的优化（代码已落地）

1. `ReplayBuffer.sample` 低开销采样（缓存概率 + `replace=True`）
2. `GoState.legal_actions` 实例缓存
3. 异步自博弈空队列参数错误修复
4. `MCTS` 根并行（Root Parallelism）
5. Policy Target Smoothing（策略目标平滑）
6. Opening Book 混合自博弈（开局重放）
7. `GoState.to_tensor` 缓存 + 向量化构建

## 1.2 现在最该继续做的 4 件事（按投入产出排序）

1. **MCTS 批叶子去重推理**（同状态只前向一次）
2. **TreeNode 级 legal 缓存**（避免重复合法性扫描）
3. **ReplayBuffer 权重增量更新**（减少 `extend` 后全量重建）
4. **晋级判定抗噪声升级**（动态门槛 + 关键对手 hard gate）

## 1.3 预期收益（保守）

- 运行性能：每轮耗时下降 `15%~35%`
- 棋力：同等训练预算提升 `+30~+120 Elo`

---

## 2. 直接可用命令（复制就能跑）

## 2.1 环境

```bash
export PATH="$HOME/miniforge3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate alphazero9
cd /mnt/data/alphazero9
```

## 2.2 最小冒烟（覆盖 1/3/4/6）

```bash
python scripts/train.py \
  --board-size 5 \
  --device cpu \
  --iterations 2 \
  --selfplay-games-per-iter 2 \
  --selfplay-workers 1 \
  --mcts-simulations 4 \
  --eval-simulations 4 \
  --mcts-inference-batch 4 \
  --mcts-root-parallelism 2 \
  --opening-book-prob 0.7 \
  --opening-book-max-moves 3 \
  --policy-target-smoothing 0.02 \
  --train-steps-per-iter 2 \
  --batch-size 8 \
  --replay-size 512 \
  --channels 32 \
  --num-blocks 1 \
  --value-hidden-dim 128 \
  --arena-games 4 \
  --save-dir checkpoints/smoke_opt_1364
```

## 2.3 CPU 推荐模板（先稳，再快）

```bash
python scripts/train.py \
  --device cpu \
  --iterations 100 \
  --selfplay-games-per-iter 80 \
  --selfplay-workers 8 \
  --async-selfplay \
  --async-prefetch-batches 2 \
  --train-torch-threads 8 \
  --worker-torch-threads 1 \
  --mcts-simulations 256 \
  --eval-simulations 320 \
  --mcts-inference-batch 16 \
  --mcts-root-parallelism 2 \
  --mcts-topk-expand 48 \
  --policy-target-smoothing 0.02 \
  --opening-book-prob 0.35 \
  --opening-book-max-moves 3 \
  --train-steps-per-iter 1200 \
  --batch-size 256 \
  --channels 192 \
  --num-blocks 12 \
  --value-hidden-dim 256
```

---

## 3. 参数速查（看表就够）

| 目标 | 参数 | 推荐范围 | 说明 |
|---|---|---|---|
| 提速 | `--mcts-root-parallelism` | `1~4` | 根并行，CPU 核多时收益更明显 |
| 提速 | `--mcts-inference-batch` | CPU:`4~16` | 增大批推理吞吐 |
| 提棋力 | `--policy-target-smoothing` | `0.01~0.05` | 降低策略头过拟合 |
| 网络容量 | `--value-hidden-dim` | `64~256` | 控制 value head MLP 隐层大小 |
| 提棋力+多样性 | `--opening-book-prob` | `0.2~0.5` | 部分对局从开局库启动 |
| 控制开局深度 | `--opening-book-max-moves` | `2~6` | 开局库最多走几手 |
| 性能/质量平衡 | `--mcts-simulations` | `128~512` | 越高越强越慢 |
| 稳定评估 | `--eval-simulations` | `>= mcts_simulations` | 防误晋级 |

---

## 4. 已落地改造详解（你现在项目里真正生效的）

## 4.1 Root Parallelism（已完成）

**文件位置**
- `go_zero/mcts.py`
- `go_zero/train.py`

**做了什么**
1. `MCTS` 新增 `root_parallelism`。
2. 总模拟数按 root 数拆分，使用线程池并发执行多个 root 搜索。
3. 最终策略分布仍由合并后的 visits 归一化得到。
4. root value 按各 root 分配到的 simulation 数做加权合并。

**为什么能提速/提棋力**
- 提速：多 root 并发搜索提升 CPU 利用率，并减少单树热点偏置。
- 棋力：在同预算下增加根层探索多样性，常见于复杂局面更稳。

**风险**
- `root_parallelism` 过高可能导致每棵树模拟太少，策略噪声上升。

**建议默认**
- CPU：`2`
- 小预算训练：`1~2`

**实现位置（当前代码）**
- `go_zero/mcts.py`：`ThreadPoolExecutor` 并发提交 `_run_single_root_search(...)`
- `go_zero/mcts.py`：收集各 root 的 `visits` 后做向量求和

---

## 4.2 Policy Target Smoothing（已完成）

**文件位置**
- `go_zero/train.py::train_candidate`

**做了什么**
- 训练时把目标策略做平滑：
- `target = target*(1-s) + s/|A|`

**收益**
- 减少“过尖目标”导致的过拟合与梯度不稳定。
- 对小样本阶段和开局分布偏窄阶段更友好。

**建议范围**
- `0.01~0.03`（建议从 `0.02` 起）

**风险**
- 过大（如 `>0.08`）会冲淡 MCTS 信号。

---

## 4.3 Opening Book 混合自博弈（已完成）

**文件位置**
- `go_zero/train.py::opening_book_lines`
- `go_zero/train.py::maybe_play_opening_book`
- `go_zero/train.py::selfplay_game`

**做了什么**
1. 构造中心和近星位的短序列开局线。
2. 按概率 `opening_book_prob` 对局前置若干步。
3. 非法则提前结束开局注入，回到正常 MCTS。

**收益**
- 增加前中盘高价值样本密度。
- 早期训练更快摆脱随机开局噪声。

**建议范围**
- `opening_book_prob: 0.2~0.5`
- `opening_book_max_moves: 2~4`

**风险**
- 概率过高会导致开局分布过窄。

---

## 4.4 State Tensor 缓存与向量化（已完成）

**文件位置**
- `go_zero/go.py::to_tensor`

**做了什么**
1. 增加 `_tensor_cache`，同状态重复请求直接返回。
2. 改为预分配 `planes` 后批量填充历史平面。
3. 减少 Python list + `np.stack` 的重复分配。

**收益**
- `MCTS` 大量 `to_tensor()` 调用时明显降低 CPU 与内存分配压力。

**风险**
- 无状态突变风险（`GoState` 为 frozen dataclass，不可变语义安全）。

---

## 5. 下一批建议重构（还没做，建议按顺序）

## 5.1 A1: `_expand_batch` 叶子去重推理（最高优先级）

**目标**
- 同一批叶子里重复状态只跑一次网络。

**实施步骤**
1. 在 `_expand_batch` 为每个 `state` 生成 key。
2. 建 `key -> leaf_indices` 映射。
3. 对 unique states 运行 `_predict_batch`。
4. 把结果回填到所有重复叶子。

**验收指标**
- `_predict_batch` 每轮调用次数下降。
- 同配置 self-play 耗时下降。

---

## 5.2 A2: TreeNode 保存 legal actions

**目标**
- 减少 `state.legal_actions()` 重复调用。

**实施步骤**
1. `TreeNode` 增加 `legal_actions` 字段。
2. 扩展节点时写入该字段。
3. 选择/扩展优先读节点缓存。

**验收指标**
- profiling 中 `GoState.is_legal` 占比下降。

---

## 5.3 A3: ReplayBuffer 权重增量更新

**目标**
- `extend` 后不再全量重建全池权重。

**实施步骤**
1. 维护 `difficulty/timestamp/weight` 数组。
2. `extend` 时仅计算新增段。
3. 每 `N` 次做一次全量校准。

**验收指标**
- `sample` 平均耗时进一步下降。

---

## 5.4 B1: 晋级判定抗噪声升级

**目标**
- 防止偶然晋级造成 Elo 曲线抖动。

**实施步骤**
1. `promote_threshold` 动态化（依据最近窗口方差）。
2. 增加关键 benchmark 对手 hard gate。
3. 对边界局面增加复验（不同 seed）。

**验收指标**
- 晋级后回退概率下降。
- 10 轮滑动 Elo 方差下降。

---

## 6. A/B 实验模板（不走弯路）

## 6.1 实验设计

1. Baseline：当前主干参数
2. Baseline + 单项改造
3. 每组至少 `10` 迭代
4. 固定随机种子集合（如 `42/314/2718`）

## 6.2 每轮必记指标

- `time_selfplay`
- `time_train`
- `time_arena`
- `selfplay_games_per_hour`
- `arena_w/l/d`
- `benchmark_winrate`
- `pool_best_elo`

## 6.3 保留/回滚规则

- 保留：连续 5 轮平均耗时下降 `>=10%` 且棋力不降
- 回滚：任意 2 轮 benchmark 明显下滑（例如 `>3%`）

---

## 7. 常见问题与建议

1. **Q：提速后棋力下降怎么办？**
- 先降 `mcts-root-parallelism`，再降低 `policy-target-smoothing`。

2. **Q：开局库会不会把策略带偏？**
- 控制 `opening-book-prob <= 0.5`，并保留随机开局对局。

3. **Q：CPU 很满但速度没上去？**
- 检查 `mcts-inference-batch`、`worker-torch-threads=1`、`train-torch-threads` 是否过高互抢。

---

## 8. 当前建议默认值（可直接抄）

```text
mcts-root-parallelism=2
policy-target-smoothing=0.02
opening-book-prob=0.35
opening-book-max-moves=3
```

如果你要做“冲分模式”（棋力优先）：

```text
mcts-simulations↑
eval-simulations>=mcts-simulations
arena-games↑
```

如果你要做“速度模式”（吞吐优先）：

```text
mcts-simulations↓
mcts-inference-batch适度↑
selfplay-workers按物理核调优
```

---

## 9. 文档维护规则（建议）

每次改动请补三行：

1. 改了什么（文件 + 函数）
2. 为什么改（预期收益）
3. 如何验证（命令 + 指标）

这样后续任何人接手都能“一眼看明白”。
