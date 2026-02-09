# AlphaZero 9x9 Go (Competition Edition)

> 注：本项目主要由 AI（Codex）完成与维护，人工负责需求与验收。

面向比赛的围棋强化学习项目（9x9）：
- `PyTorch` 策略价值网络（ResNet + SE，value head 隐层可配）
- `MCTS`（PUCT + batched inference + virtual loss + top-k expand）
- `自博弈/训练解耦`（异步预取 + 多进程 worker）
- `模型池 + Elo + Benchmark` 评估体系
- `SPRT` 晋级早停 + `PBT` 自动调参
- `superko` + 历史特征 + 样本去重 + 优先采样

---

## 1. 3 分钟上手（先跑起来）

### 1.1 环境准备

```bash
export PATH="$HOME/miniforge3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate alphazero
cd /home/fm/alphazero9
```

### 1.2 快速功能验证（2 轮，约几十秒）

```bash
python scripts/train.py \
  --board-size 5 \
  --device cpu \
  --iterations 2 \
  --selfplay-games-per-iter 2 \
  --selfplay-workers 2 \
  --async-selfplay \
  --mcts-simulations 2 \
  --eval-simulations 2 \
  --mcts-inference-batch 4 \
  --train-steps-per-iter 1 \
  --batch-size 8 \
  --replay-size 256 \
  --channels 32 \
  --num-blocks 1 \
  --value-hidden-dim 128
```

看到每轮输出 `saved=checkpoints/iter_xxx.pt` 即表示链路正常。

### 1.3 正式训练（9x9 推荐起步）

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
  --mcts-topk-expand 48 \
  --mcts-virtual-loss 1.0 \
  --temp-start 1.0 \
  --temp-mid 0.4 \
  --temp-end 0.05 \
  --temp-moves 20 \
  --temp-mid-moves 40 \
  --resign-threshold 0.98 \
  --train-steps-per-iter 1200 \
  --batch-size 256 \
  --channels 192 \
  --num-blocks 12 \
  --value-hidden-dim 256 \
  --arena-games 60 \
  --promote-threshold 0.55 \
  --arena-sprt-delta 0.05 \
  --pbt \
  --pbt-patience 3 \
  --pbt-sigma 0.15 \
  --value-huber-delta 1.0 \
  --policy-entropy-coeff 1e-3 \
  --pool-max-size 20 \
  --pool-opponents 4 \
  --benchmark-opponents 4 \
  --benchmark-games-per-opponent 8 \
  --elo-games-per-opponent 12 \
  --elo-k 24
```

---

## 2. 项目结构（看这里知道代码在哪）

- `go_zero/go.py`
- 围棋状态、合法性判定、`superko`、特征平面编码

- `go_zero/model.py`
- 策略价值网络（ResNet + SE）

- `go_zero/mcts.py`
- MCTS 核心：PUCT、根噪声、batched inference、virtual loss、top-k expand

- `go_zero/train.py`
- 训练主控：并行自博弈、异步解耦、回放池、arena、Elo、benchmark、PBT

- `scripts/train.py`
- 训练入口脚本

- `scripts/play.py`
- 人机对弈入口（自动读取 checkpoint 网络配置）

- `scripts/battle_ui_pyqt.py`
- PyQt 桌面对战界面（真人/人机/机机、自动模式、悔棋）

- `checkpoints/`
- 训练输出目录（模型、模型池、Elo 元数据）

---

## 3. 训练流程（逻辑全景）

每轮迭代顺序：

1. `best model` 生成自博弈数据
2. 过滤低信息样本 + 去重
3. 写入回放池（优先采样：新鲜度 + 难度）
4. 训练 `candidate model`
5. `arena` 对战 best（支持 SPRT 早停）
6. 若晋级：
- 更新 `best model`
- 写入模型池
- 与池中对手打 Elo
- 做 benchmark 对手集评估
7. 若不晋级且启用 PBT：
- 自动扰动关键超参
8. 保存 `iter_xxx.pt` 和 `best.pt`

---

## 4. 输出文件说明

### 4.1 Checkpoint

- `checkpoints/iter_001.pt`
- 每轮完整状态（模型、配置、评估信息）

- `checkpoints/best.pt`
- 当前最强模型（用于对弈和部署）

### 4.2 模型池与 Elo

- `checkpoints/model_pool/`
- 历史晋级模型文件

- `checkpoints/model_pool.json`
- 模型 Elo、对局数、当前 best 标记

---

## 5. 常用命令

### 5.1 继续训练（恢复）

```bash
python scripts/train.py \
  --device cpu \
  --resume checkpoints/iter_020.pt \
  --selfplay-workers 8 \
  --async-selfplay \
  --async-prefetch-batches 2
```

### 5.2 对弈（人 vs AI）

```bash
python scripts/play.py \
  --checkpoint checkpoints/best.pt \
  --human-color black \
  --sims 320
```

### 5.3 桌面对战界面（PyQt）

```bash
python scripts/battle_ui_pyqt.py
```

输入格式：
- 落子：`row col`，例如 `3 4`
- 停一手：`pass`

---

## 6. 参数速查（重点参数一眼懂）

### 6.1 性能相关

- `--selfplay-workers`
- 自博弈并行进程数（CPU 建议 `4~16`）

- `--async-selfplay`
- 开启“后台自博弈 + 前台训练”解耦

- `--async-prefetch-batches`
- 异步预取深度（建议 `2~4`）

- `--train-torch-threads`
- 主训练进程 PyTorch 线程数（`0` 为自动）

- `--worker-torch-threads`
- 每个自博弈 worker 线程数（建议 `1`）

- `--mcts-inference-batch`
- MCTS 批推理大小（CPU `4~16`，GPU `16~64`）

- `--mcts-topk-expand`
- 仅扩展先验 top-k 子节点（加速搜索）

- `--mcts-virtual-loss`
- 虚拟损失强度（并行搜索常用 `0.5~2.0`）

### 6.2 棋力相关

- `--mcts-simulations`
- 自博弈搜索次数（越高越强越慢）

- `--eval-simulations`
- 评估搜索次数（建议 >= 训练）

- `--temp-start --temp-mid --temp-end`
- 三段温度策略（探索->收敛）

- `--temp-moves --temp-mid-moves`
- 温度切换步数

- `--resign-threshold`
- 投子阈值（低阈值更激进）

- `--value-huber-delta`
- 价值头 Huber 损失参数

- `--policy-entropy-coeff`
- 策略熵正则系数（防策略塌缩）

- `--value-hidden-dim`
- value head MLP 隐层维度（9x9 常用 `64/128/256`）

### 6.3 评估/晋级相关

- `--arena-games`
- 候选 vs best 晋级赛局数

- `--promote-threshold`
- 晋级胜率阈值

- `--arena-sprt-delta`
- SPRT 早停容忍带（越小越严格）

- `--pool-opponents`
- Elo 更新时采样对手数

- `--elo-games-per-opponent`
- 与每个 Elo 对手对局数

- `--benchmark-opponents --benchmark-games-per-opponent`
- 晋级后基准池评估配置

### 6.4 自动调参相关（PBT）

- `--pbt`
- 开启停滞时自动超参扰动

- `--pbt-patience`
- 连续多少轮未晋级触发扰动

- `--pbt-sigma`
- 扰动强度

### 6.5 全量参数说明（含默认值）

以下为 `python scripts/train.py` 的完整参数清单：

- `--board-size`（默认 `9`）：棋盘大小。
- `--komi`（默认 `5.5`）：贴目。
- `--history-len`（默认 `8`）：历史特征平面长度。

- `--device`（默认 `cpu`）：训练设备，如 `cpu` / `cuda`。
- `--seed`（默认 `42`）：随机种子。

- `--iterations`（默认 `50`）：总训练迭代轮数。
- `--selfplay-games-per-iter`（默认 `40`）：每轮自博弈局数。
- `--selfplay-workers`（默认 `1`）：自博弈并行进程数。
- `--async-selfplay`（默认 `False`）：开启异步自博弈。
- `--async-prefetch-batches`（默认 `2`）：异步自博弈预取深度。
- `--train-torch-threads`（默认 `0`）：训练进程线程数，`0` 表示自动。
- `--worker-torch-threads`（默认 `1`）：每个自博弈 worker 的 PyTorch 线程数。
- `--mcts-simulations`（默认 `192`）：自博弈 MCTS 模拟次数。
- `--eval-simulations`（默认 `256`）：评估/对局 MCTS 模拟次数。
- `--mcts-inference-batch`（默认 `8`）：MCTS 批推理大小。
- `--mcts-topk-expand`（默认 `0`）：仅扩展先验 top-k 子节点，`0` 表示不限制。
- `--mcts-virtual-loss`（默认 `1.0`）：并行搜索虚拟损失强度。
- `--mcts-root-parallelism`（默认 `1`）：根并行数（内部至少为 `1`）。
- `--temp-moves`（默认 `20`）：前段温度持续步数。
- `--temp-mid-moves`（默认 `40`）：中段温度截止步数。
- `--temp-start`（默认 `1.0`）：前段温度。
- `--temp-mid`（默认 `0.4`）：中段温度。
- `--temp-end`（默认 `0.05`）：后段温度。
- `--resign-threshold`（默认 `0.98`）：投子阈值。
- `--disable-resign`（默认 `False`）：禁用投子逻辑。

- `--replay-size`（默认 `200000`）：回放池容量。
- `--batch-size`（默认 `256`）：训练 batch 大小。
- `--train-steps-per-iter`（默认 `800`）：每轮训练 step 数。

- `--lr`（默认 `1e-3`）：学习率。
- `--weight-decay`（默认 `1e-4`）：权重衰减。
- `--grad-clip-norm`（默认 `1.0`）：梯度裁剪阈值。
- `--use-amp`（默认 `False`）：显式请求开启 AMP。
- `--no-amp`（默认 `False`）：强制关闭 AMP（优先级高于 `--use-amp`）。
- `--value-huber-delta`（默认 `1.0`）：价值头 Huber delta。
- `--policy-entropy-coeff`（默认 `1e-3`）：策略熵正则系数。
- `--policy-target-smoothing`（默认 `0.0`）：策略目标平滑系数（内部下限裁剪到 `0`）。

- `--channels`（默认 `192`）：主干通道数。
- `--num-blocks`（默认 `12`）：残差块数量。
- `--no-se`（默认 `False`）：关闭 SE 模块。
- `--value-hidden-dim`（默认 `256`）：value head MLP 隐层维度（内部至少为 `1`）。

- `--arena-games`（默认 `40`）：候选与 best 的 arena 局数。
- `--promote-threshold`（默认 `0.55`）：晋级分数阈值。
- `--no-arena-sprt`（默认 `False`）：关闭 SPRT 提前停止。
- `--arena-sprt-delta`（默认 `0.05`）：SPRT 判定容忍带。

- `--no-augment-symmetry`（默认 `False`）：关闭对称增强。

- `--c-base`（默认 `19652.0`）：PUCT `c_base`。
- `--c-init`（默认 `1.25`）：PUCT `c_init`。
- `--dirichlet-alpha`（默认 `0.3`）：根节点 Dirichlet 噪声 alpha。
- `--dirichlet-eps`（默认 `0.25`）：根节点噪声混合比例。

- `--pool-max-size`（默认 `20`）：模型池最大容量。
- `--pool-opponents`（默认 `4`）：Elo 更新采样对手数。
- `--elo-games-per-opponent`（默认 `12`）：每个 Elo 对手对局数。
- `--benchmark-opponents`（默认 `4`）：benchmark 采样对手数。
- `--benchmark-games-per-opponent`（默认 `8`）：每个 benchmark 对手对局数。
- `--elo-k`（默认 `24.0`）：Elo K 因子。
- `--elo-init`（默认 `1500.0`）：新模型初始 Elo。
- `--pbt`（默认 `False`）：启用 PBT 自动扰动。
- `--pbt-patience`（默认 `3`）：连续未晋级多少轮触发扰动。
- `--pbt-sigma`（默认 `0.15`）：PBT 扰动强度。
- `--opening-book-prob`（默认 `0.0`）：开局库注入概率（内部裁剪到 `[0, 1]`）。
- `--opening-book-max-moves`（默认 `0`）：开局库最多注入步数（内部下限裁剪到 `0`）。

- `--save-dir`（默认 `checkpoints`）：checkpoint 保存目录。
- `--resume`（默认空）：恢复训练的 checkpoint 路径。
- `--save-buffer`（默认 `False`）：将回放池随 checkpoint 一起保存。

---

## 7. 调参建议（实战）

### 7.1 CPU 机器

- 优先保证吞吐：
- 增加 `--selfplay-workers`
- `--worker-torch-threads 1`
- 中等 `--mcts-inference-batch`

- 如果卡住或抖动：
- 降低 `--mcts-simulations`
- 降低 `--train-steps-per-iter`
- 开启 `--pbt`

### 7.2 GPU 机器

- 提升推理批量：
- 增加 `--mcts-inference-batch`
- 增加 `--batch-size`
- 打开 `--use-amp`

- 注意显存：
- OOM 时先降 `batch-size`，再降 `channels/num-blocks`

---

## 8. 常见问题排查

### 8.1 训练很慢

- 检查 `--selfplay-workers` 是否过低
- 检查 `--worker-torch-threads` 是否不是 `1`
- 检查 `--mcts-simulations` 是否过高

### 8.2 棋力不涨

- 提高 `--selfplay-games-per-iter`
- 提高 `--arena-games` 与 `--eval-simulations`
- 开启 `--pbt`
- 检查是否频繁误晋级（可收紧 `promote-threshold`）

### 8.3 评估波动大

- 提高 `--elo-games-per-opponent`
- 增加 `--benchmark-games-per-opponent`
- 减小 `--elo-k`

---

## 9. 当前项目已经实现的冠军向能力

- 强规则：`superko`
- 强特征：历史平面 + last move + capture planes
- 强搜索：batched MCTS + virtual loss + top-k
- 强训练：解耦流水线 + 优先采样 + 去重过滤 + Huber + 熵正则
- 强评估：SPRT 晋级 + Elo 模型池 + benchmark
- 强运维：可恢复训练、可长期迭代

如果你要继续冲榜，下一步建议：
- 常驻 worker + 共享内存权重广播（进一步降通信开销）
- 分布式多机自博弈
- 推理编译优化（`torch.compile`/ONNX）
