# AlphaZero 9x9 围棋 - 速查（同步版）

> 注：本项目主要由 AI（Codex）完成与维护，人工负责需求与验收。

## 1) 环境

```bash
export PATH="$HOME/miniforge3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate alphazero9
cd /mnt/data/alphazero9
```

## 2) 最小冒烟（覆盖新优化）

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

## 3) 正式训练（CPU 推荐模板）

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
  --mcts-virtual-loss 1.0 \
  --policy-target-smoothing 0.02 \
  --opening-book-prob 0.35 \
  --opening-book-max-moves 3 \
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
  --pbt-sigma 0.15
```

## 4) 恢复训练

```bash
python scripts/train.py \
  --device cpu \
  --resume checkpoints/iter_020.pt \
  --selfplay-workers 8 \
  --async-selfplay \
  --async-prefetch-batches 2
```

## 5) 对弈

```bash
python scripts/play.py \
  --checkpoint checkpoints/best.pt \
  --human-color black \
  --sims 320
```

## 5.1) 桌面对战界面（PyQt）

```bash
python scripts/battle_ui_pyqt.py
```

输入：
- 落子：`row col`（如 `3 4`）
- 停一手：`pass`

## 6) 你现在新增可用的关键参数

- 运行性能：`mcts-root-parallelism`（线程池并发 root 搜索） `mcts-inference-batch` `selfplay-workers`
- 棋力稳定：`policy-target-smoothing` `eval-simulations` `arena-games`
- 网络规模：`value-hidden-dim`（value head 隐层维度）
- 样本多样性：`opening-book-prob` `opening-book-max-moves`

建议起步值：
- `mcts-root-parallelism=2`
- `policy-target-smoothing=0.02`
- `value-hidden-dim=128`
- `opening-book-prob=0.35`
- `opening-book-max-moves=3`

## 7) 输出文件

- `checkpoints/iter_xxx.pt`：每轮 checkpoint
- `checkpoints/best.pt`：当前最强模型
- `checkpoints/model_pool/`：模型池
- `checkpoints/model_pool.json`：Elo 元数据

## 8) 完整重构手册

- 详版文档：`PERF_STRENGTH_REFACTOR_GUIDE.md`

## 9) 全量参数速查（精简）

- 棋局与特征：`board-size=9` 棋盘大小；`komi=5.5` 贴目；`history-len=8` 历史特征长度。
- 设备与随机：`device=cpu` 训练设备；`seed=42` 随机种子。
- 训练循环：`iterations=50` 总轮数；`selfplay-games-per-iter=40` 每轮自博弈局数；`train-steps-per-iter=800` 每轮训练步数。
- 自博弈并行：`selfplay-workers=1` worker 数；`async-selfplay=False` 异步自博弈；`async-prefetch-batches=2` 异步预取深度；`train-torch-threads=0` 训练线程；`worker-torch-threads=1` worker 线程。
- MCTS：`mcts-simulations=192` 自博弈模拟数；`eval-simulations=256` 评估模拟数；`mcts-inference-batch=8` 批推理；`mcts-topk-expand=0` top-k 扩展；`mcts-virtual-loss=1.0` 虚拟损失；`mcts-root-parallelism=1` 根并行。
- 温度与投子：`temp-start=1.0`、`temp-mid=0.4`、`temp-end=0.05` 三段温度；`temp-moves=20`、`temp-mid-moves=40` 切换步数；`resign-threshold=0.98` 投子阈值；`disable-resign=False` 禁用投子。
- 回放与 batch：`replay-size=200000` 回放池容量；`batch-size=256` 批大小。
- 优化器与损失：`lr=1e-3` 学习率；`weight-decay=1e-4` 权重衰减；`grad-clip-norm=1.0` 梯度裁剪；`use-amp=False` 请求 AMP；`no-amp=False` 强制关 AMP（优先级更高）；`value-huber-delta=1.0`；`policy-entropy-coeff=1e-3`；`policy-target-smoothing=0.0`。
- 网络：`channels=192` 通道；`num-blocks=12` 残差块；`no-se=False` 关闭 SE；`value-hidden-dim=256` value head 隐层。
- Arena 晋级：`arena-games=40` 对局数；`promote-threshold=0.55` 晋级阈值；`no-arena-sprt=False` 关闭 SPRT；`arena-sprt-delta=0.05` SPRT 容忍带。
- 数据增强：`no-augment-symmetry=False` 关闭对称增强。
- PUCT 与噪声：`c-base=19652.0`、`c-init=1.25`；`dirichlet-alpha=0.3`、`dirichlet-eps=0.25`。
- 模型池与 Elo：`pool-max-size=20`；`pool-opponents=4`；`elo-games-per-opponent=12`；`benchmark-opponents=4`；`benchmark-games-per-opponent=8`；`elo-k=24.0`；`elo-init=1500.0`。
- PBT 与开局：`pbt=False`；`pbt-patience=3`；`pbt-sigma=0.15`；`opening-book-prob=0.0`；`opening-book-max-moves=0`。
- 保存恢复：`save-dir=checkpoints` 输出目录；`resume=''` 恢复路径；`save-buffer=False` 连同回放池保存。
