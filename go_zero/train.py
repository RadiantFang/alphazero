from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from .go import BLACK, WHITE, GoState, decode_action
from .mcts import MCTS
from .model import AlphaZeroNet

_WORKER_THREADS_CONFIGURED = False
_WORKER_MODEL: AlphaZeroNet | None = None
_WORKER_MODEL_VERSION: int = -1


@dataclass
class TrainConfig:
    board_size: int = 9
    komi: float = 5.5
    history_len: int = 8

    device: str = "cpu"
    seed: int = 42

    iterations: int = 50
    selfplay_games_per_iter: int = 40
    selfplay_workers: int = 1
    async_selfplay: bool = False
    train_torch_threads: int = 0
    worker_torch_threads: int = 1

    mcts_simulations: int = 192
    eval_simulations: int = 256
    mcts_inference_batch: int = 8
    mcts_topk_expand: int = 0
    mcts_virtual_loss: float = 1.0
    mcts_root_parallelism: int = 1
    temp_moves: int = 20
    temp_mid_moves: int = 40
    temp_start: float = 1.0
    temp_mid: float = 0.4
    temp_end: float = 0.05
    resign_threshold: float = 0.98
    disable_resign: bool = False

    replay_size: int = 200000
    batch_size: int = 256
    train_steps_per_iter: int = 800

    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    value_huber_delta: float = 1.0
    policy_entropy_coeff: float = 1e-3
    policy_target_smoothing: float = 0.0

    channels: int = 192
    num_blocks: int = 12
    use_se: bool = True
    value_hidden_dim: int = 256

    arena_games: int = 40
    promote_threshold: float = 0.55
    arena_sprt: bool = True
    arena_sprt_delta: float = 0.05

    augment_symmetry: bool = True

    c_base: float = 19652.0
    c_init: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25

    pool_max_size: int = 20
    pool_opponents: int = 4
    elo_games_per_opponent: int = 12
    benchmark_opponents: int = 4
    benchmark_games_per_opponent: int = 8
    elo_k: float = 24.0
    elo_init: float = 1500.0

    async_prefetch_batches: int = 2
    pbt: bool = False
    pbt_patience: int = 3
    pbt_sigma: float = 0.15
    opening_book_prob: float = 0.0
    opening_book_max_moves: int = 0

    save_dir: str = "checkpoints"
    resume: str = ""
    save_buffer: bool = False


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=capacity)
        self.counter = 0
        self._sampling_dirty = True
        self._sample_probs: np.ndarray | None = None
        self._data_cache: list[tuple[np.ndarray, np.ndarray, float, float, int]] = []

    def extend(self, items):
        for s, p, v, difficulty in items:
            self.data.append((s, p, v, float(difficulty), self.counter))
            self.counter += 1
        self._sampling_dirty = True

    def __len__(self):
        return len(self.data)

    def _rebuild_sampling_cache(self):
        self._data_cache = list(self.data)
        n = len(self._data_cache)
        if n == 0:
            self._sample_probs = None
            self._sampling_dirty = False
            return

        idx = np.arange(n, dtype=np.float64)
        latest = max(1.0, float(self._data_cache[-1][4]))
        freshness = 0.2 + 0.8 * ((idx + 1.0) / n)
        difficulties = np.fromiter((max(1e-3, float(x[3])) for x in self._data_cache), dtype=np.float64, count=n)
        recency = np.fromiter(
            (0.5 + 0.5 * (float(x[4]) / latest) for x in self._data_cache),
            dtype=np.float64,
            count=n,
        )
        weights = difficulties * freshness * recency
        total = float(weights.sum())
        if total <= 0.0:
            self._sample_probs = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            self._sample_probs = weights / total
        self._sampling_dirty = False

    def sample(self, batch_size: int, board_size: int, augment_symmetry: bool):
        n = len(self.data)
        if n == 0:
            raise RuntimeError("ReplayBuffer is empty.")

        if self._sampling_dirty or self._sample_probs is None or len(self._data_cache) != n:
            self._rebuild_sampling_cache()

        if self._sample_probs is None:
            raise RuntimeError("ReplayBuffer sampling cache is unavailable.")

        choice_idx = np.random.choice(n, size=batch_size, replace=True, p=self._sample_probs)
        batch = [self._data_cache[int(i)] for i in choice_idx]
        states = []
        policies = []
        values = []

        for s, p, v, _, _ in batch:
            if augment_symmetry:
                sym = random.randint(0, 7)
                s = apply_symmetry_to_tensor(s, sym)
                p = apply_symmetry_to_policy(p, board_size, sym)
            states.append(s)
            policies.append(p)
            values.append(v)

        states_t = torch.from_numpy(np.stack(states, axis=0)).to(torch.float32)
        policies_t = torch.from_numpy(np.stack(policies, axis=0)).to(torch.float32)
        values_t = torch.tensor(values, dtype=torch.float32)
        return states_t, policies_t, values_t


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def transform_2d(board: np.ndarray, sym: int) -> np.ndarray:
    if sym < 4:
        out = np.rot90(board, k=sym)
    else:
        out = np.rot90(np.fliplr(board), k=sym - 4)
    return np.ascontiguousarray(out)


def apply_symmetry_to_tensor(state_tensor: np.ndarray, sym: int) -> np.ndarray:
    out = np.stack([transform_2d(ch, sym) for ch in state_tensor], axis=0)
    return out.astype(np.float32, copy=False)


def apply_symmetry_to_policy(policy: np.ndarray, board_size: int, sym: int) -> np.ndarray:
    board_policy = policy[: board_size * board_size].reshape(board_size, board_size)
    pass_prob = policy[board_size * board_size]
    board_policy = transform_2d(board_policy, sym)
    out = np.zeros_like(policy, dtype=np.float32)
    out[: board_size * board_size] = board_policy.reshape(-1)
    out[board_size * board_size] = pass_prob
    return out


def position_key(state_tensor: np.ndarray) -> bytes:
    return state_tensor.tobytes()


def filter_selfplay_samples(
    samples: list[tuple[np.ndarray, np.ndarray, float, float]],
    recent_keys: deque[bytes],
    recent_set: set[bytes],
    max_recent: int = 200000,
) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    out = []
    for s, p, z, d in samples:
        # Low-information filter: skip near-certain pass distributions.
        if float(p[-1]) > 0.995:
            continue
        k = position_key(s)
        if k in recent_set:
            continue
        recent_keys.append(k)
        recent_set.add(k)
        out.append((s, p, z, d))
        while len(recent_keys) > max_recent:
            old = recent_keys.popleft()
            recent_set.discard(old)
    return out


def build_model(cfg: TrainConfig) -> AlphaZeroNet:
    return AlphaZeroNet(
        board_size=cfg.board_size,
        history_len=cfg.history_len,
        channels=cfg.channels,
        num_blocks=cfg.num_blocks,
        use_se=cfg.use_se,
        value_hidden_dim=cfg.value_hidden_dim,
    )


def create_mcts(model: torch.nn.Module, cfg: TrainConfig, sims: int, add_noise: bool) -> MCTS:
    del add_noise  # kept for call-site readability
    return MCTS(
        model=model,
        board_size=cfg.board_size,
        num_simulations=sims,
        c_base=cfg.c_base,
        c_init=cfg.c_init,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_eps=cfg.dirichlet_eps,
        inference_batch_size=cfg.mcts_inference_batch,
        topk_expand=cfg.mcts_topk_expand,
        virtual_loss=cfg.mcts_virtual_loss,
        root_parallelism=cfg.mcts_root_parallelism,
        device=cfg.device,
    )


def opening_book_lines(board_size: int) -> list[list[tuple[int, int]]]:
    if board_size < 5:
        return []
    c = board_size // 2
    near = max(0, c - 1)
    far = min(board_size - 1, c + 1)
    lines = [
        [(c, c), (near, c), (c, near), (far, c), (c, far)],
        [(c, c), (c, near), (near, c), (c, far), (far, c)],
        [(near, near), (far, far), (near, far), (far, near), (c, c)],
        [(near, c), (far, c), (c, near), (c, far), (c, c)],
    ]
    out = []
    for line in lines:
        uniq = []
        seen = set()
        for r, col in line:
            if 0 <= r < board_size and 0 <= col < board_size and (r, col) not in seen:
                uniq.append((r, col))
                seen.add((r, col))
        if uniq:
            out.append(uniq)
    return out


def maybe_play_opening_book(state: GoState, cfg: TrainConfig) -> tuple[GoState, int]:
    if cfg.opening_book_prob <= 0.0 or cfg.opening_book_max_moves <= 0:
        return state, 0
    if random.random() >= cfg.opening_book_prob:
        return state, 0

    lines = opening_book_lines(state.board_size)
    if not lines:
        return state, 0

    line = random.choice(lines)
    max_moves = min(len(line), cfg.opening_book_max_moves)
    played = 0
    for move in line[:max_moves]:
        if not state.is_legal(move):
            break
        state = state.play(move)
        played += 1
        if state.is_terminal():
            break
    return state, played


def selfplay_game(model: torch.nn.Module, cfg: TrainConfig):
    state = GoState.new_game(board_size=cfg.board_size, komi=cfg.komi, history_len=cfg.history_len)
    state, opening_moves = maybe_play_opening_book(state, cfg)
    mcts = create_mcts(model, cfg, sims=cfg.mcts_simulations, add_noise=True)

    trajectory: list[tuple[np.ndarray, np.ndarray, int, float]] = []
    move_idx = opening_moves
    resigned_winner = 0
    while not state.is_terminal():
        if move_idx < cfg.temp_moves:
            temp = cfg.temp_start
        elif move_idx < cfg.temp_mid_moves:
            temp = cfg.temp_mid
        else:
            temp = cfg.temp_end

        pi, action, root_v = mcts.run(
            state,
            temperature=temp,
            add_exploration_noise=True,
            return_root_value=True,
        )
        difficulty = 1.0 - float(np.max(pi))
        trajectory.append((state.to_tensor(), pi, state.to_play, max(0.01, difficulty)))

        if not cfg.disable_resign and abs(root_v) >= cfg.resign_threshold and move_idx >= 12:
            resigned_winner = -state.to_play if root_v < 0 else state.to_play
            break
        state = state.play(decode_action(cfg.board_size, action))
        move_idx += 1

    winner = resigned_winner if resigned_winner != 0 else state.winner()
    game_data = []
    for s, pi, player, difficulty in trajectory:
        if winner == 0:
            z = 0.0
        else:
            z = 1.0 if winner == player else -1.0
        difficulty = max(difficulty, abs(z) * 0.25)
        game_data.append((s, pi, z, difficulty))
    return game_data, winner


def selfplay_worker(model_path: str, model_version: int, cfg_dict: dict, num_games: int, seed: int):
    global _WORKER_THREADS_CONFIGURED, _WORKER_MODEL, _WORKER_MODEL_VERSION
    cfg = TrainConfig(**cfg_dict)
    cfg.device = "cpu"
    if not _WORKER_THREADS_CONFIGURED and cfg.worker_torch_threads > 0:
        torch.set_num_threads(cfg.worker_torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _WORKER_THREADS_CONFIGURED = True
    set_seed(seed)

    if _WORKER_MODEL is None:
        _WORKER_MODEL = build_model(cfg).to("cpu")
        _WORKER_MODEL.eval()
        _WORKER_MODEL_VERSION = -1
    if _WORKER_MODEL_VERSION != model_version:
        ckpt = torch.load(model_path, map_location="cpu")
        _WORKER_MODEL.load_state_dict(ckpt["model"])
        _WORKER_MODEL.eval()
        _WORKER_MODEL_VERSION = model_version

    samples = []
    bw = ww = dd = 0
    for _ in range(num_games):
        data, winner = selfplay_game(_WORKER_MODEL, cfg)
        samples.extend(data)
        if winner == BLACK:
            bw += 1
        elif winner == WHITE:
            ww += 1
        else:
            dd += 1

    return samples, bw, ww, dd


def run_selfplay(
    best_model: torch.nn.Module,
    cfg: TrainConfig,
    iter_idx: int,
    pool: ProcessPoolExecutor | None = None,
    model_path: str | None = None,
    model_version: int = 0,
):
    if cfg.selfplay_workers <= 1:
        best_model.eval()
        new_samples = []
        bw = ww = dd = 0
        for _ in trange(cfg.selfplay_games_per_iter, desc=f"Iter {iter_idx} selfplay", leave=True):
            data, winner = selfplay_game(best_model, cfg)
            new_samples.extend(data)
            if winner == BLACK:
                bw += 1
            elif winner == WHITE:
                ww += 1
            else:
                dd += 1
        return new_samples, bw, ww, dd

    games = cfg.selfplay_games_per_iter
    workers = min(cfg.selfplay_workers, games)
    parts = [games // workers] * workers
    for i in range(games % workers):
        parts[i] += 1

    cfg_dict = dict(cfg.__dict__)
    cfg_dict["device"] = "cpu"

    new_samples = []
    bw = ww = dd = 0
    local_pool = None
    ex = pool
    if ex is None:
        local_pool = ProcessPoolExecutor(max_workers=workers)
        ex = local_pool
    try:
        futures = []
        future_games: dict = {}
        for wid, g in enumerate(parts):
            if model_path is None:
                raise RuntimeError("model_path is required for multiprocess self-play.")
            fut = ex.submit(
                selfplay_worker,
                model_path,
                model_version,
                cfg_dict,
                g,
                cfg.seed + iter_idx * 1000 + wid,
            )
            futures.append(fut)
            future_games[fut] = g

        with tqdm(total=games, desc=f"Iter {iter_idx} selfplay", leave=True) as pbar:
            for f in as_completed(futures):
                s, b, w, d = f.result()
                new_samples.extend(s)
                bw += b
                ww += w
                dd += d
                pbar.update(future_games.get(f, 0))
    finally:
        if local_pool is not None:
            local_pool.shutdown(wait=True)

    return new_samples, bw, ww, dd


def run_selfplay_from_state(
    model_path: str,
    model_version: int,
    cfg_dict: dict,
    iter_idx: int,
    pool: ProcessPoolExecutor | None = None,
):
    cfg = TrainConfig(**cfg_dict)
    cfg.device = "cpu"
    model = build_model(cfg).to("cpu")
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return run_selfplay(
        model,
        cfg,
        iter_idx,
        pool=pool,
        model_path=model_path,
        model_version=model_version,
    )


def submit_async_selfplay(
    executor: ThreadPoolExecutor,
    best_model: torch.nn.Module,
    cfg: TrainConfig,
    iter_idx: int,
    model_path: str,
    model_version: int,
    pool: ProcessPoolExecutor | None = None,
):
    del best_model
    cfg_dict = dict(cfg.__dict__)
    cfg_dict["device"] = "cpu"
    return executor.submit(run_selfplay_from_state, model_path, model_version, cfg_dict, iter_idx, pool)


def train_candidate(candidate: torch.nn.Module, buffer: ReplayBuffer, cfg: TrainConfig):
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.train_steps_per_iter),
        eta_min=cfg.lr * 0.1,
    )

    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    candidate.train()
    losses = []
    policy_losses = []
    value_losses = []

    for _ in trange(cfg.train_steps_per_iter, desc="train", leave=True):
        states, target_policies, target_values = buffer.sample(
            batch_size=cfg.batch_size,
            board_size=cfg.board_size,
            augment_symmetry=cfg.augment_symmetry,
        )
        states = states.to(device=cfg.device)
        target_policies = target_policies.to(device=cfg.device)
        target_values = target_values.to(device=cfg.device)
        if cfg.policy_target_smoothing > 0.0:
            smooth = float(np.clip(cfg.policy_target_smoothing, 0.0, 0.25))
            target_policies = target_policies * (1.0 - smooth) + (smooth / target_policies.shape[1])

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            logits, values = candidate(states)
            policy_loss = -(target_policies * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.huber_loss(values, target_values, delta=cfg.value_huber_delta)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss = policy_loss + value_loss - cfg.policy_entropy_coeff * entropy

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.append(float(loss.item()))
        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))

    return float(np.mean(losses)), float(np.mean(policy_losses)), float(np.mean(value_losses))


def play_match(model_a: torch.nn.Module, model_b: torch.nn.Module, cfg: TrainConfig, games: int) -> tuple[int, int, int]:
    wins = 0
    losses = 0
    draws = 0

    for game_id in trange(games, desc="elo", leave=True):
        a_black = (game_id % 2 == 0)
        state = GoState.new_game(board_size=cfg.board_size, komi=cfg.komi, history_len=cfg.history_len)

        mcts_a = create_mcts(model_a, cfg, sims=cfg.eval_simulations, add_noise=False)
        mcts_b = create_mcts(model_b, cfg, sims=cfg.eval_simulations, add_noise=False)

        while not state.is_terminal():
            a_turn = (state.to_play == BLACK and a_black) or (state.to_play == WHITE and not a_black)
            mcts = mcts_a if a_turn else mcts_b
            _, action = mcts.run(state, temperature=1e-6, add_exploration_noise=False)
            state = state.play(decode_action(cfg.board_size, action))

        winner = state.winner()
        if winner == 0:
            draws += 1
        elif (winner == BLACK and a_black) or (winner == WHITE and not a_black):
            wins += 1
        else:
            losses += 1

    return wins, losses, draws


def _wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    den = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / den
    margin = (z / den) * np.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)
    return max(0.0, center - margin), min(1.0, center + margin)


def _score_interval_from_counts(wins: int, losses: int, draws: int, z: float = 1.96) -> tuple[float, float]:
    total = wins + losses + draws
    if total <= 0:
        return 0.0, 1.0

    mean = (float(wins) + 0.5 * float(draws)) / float(total)
    # Outcome support is {0.0, 0.5, 1.0}; estimate standard error from empirical variance.
    var = (
        float(wins) * (1.0 - mean) ** 2
        + float(draws) * (0.5 - mean) ** 2
        + float(losses) * (0.0 - mean) ** 2
    ) / float(total)
    se = float(np.sqrt(max(0.0, var) / float(total)))
    margin = z * se
    return max(0.0, mean - margin), min(1.0, mean + margin)


def play_match_sprt(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    cfg: TrainConfig,
    games: int,
    threshold: float,
    delta: float,
) -> tuple[int, int, int, bool]:
    wins = 0
    losses = 0
    draws = 0
    accepted = False

    for game_id in trange(games, desc="arena", leave=True):
        a_black = (game_id % 2 == 0)
        state = GoState.new_game(board_size=cfg.board_size, komi=cfg.komi, history_len=cfg.history_len)
        mcts_a = create_mcts(model_a, cfg, sims=cfg.eval_simulations, add_noise=False)
        mcts_b = create_mcts(model_b, cfg, sims=cfg.eval_simulations, add_noise=False)

        while not state.is_terminal():
            a_turn = (state.to_play == BLACK and a_black) or (state.to_play == WHITE and not a_black)
            mcts = mcts_a if a_turn else mcts_b
            _, action = mcts.run(state, temperature=1e-6, add_exploration_noise=False)
            state = state.play(decode_action(cfg.board_size, action))

        winner = state.winner()
        if winner == 0:
            draws += 1
        elif (winner == BLACK and a_black) or (winner == WHITE and not a_black):
            wins += 1
        else:
            losses += 1

        total_games = wins + losses + draws
        if total_games >= 6:
            low, high = _score_interval_from_counts(wins, losses, draws)
            if low > (threshold + delta):
                accepted = True
                break
            if high < (threshold - delta):
                accepted = False
                break

    total_games = max(1, wins + losses + draws)
    if total_games >= 1 and not accepted:
        mean_score = (float(wins) + 0.5 * float(draws)) / float(total_games)
        accepted = mean_score >= threshold
    return wins, losses, draws, accepted


def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def update_elo_pair(r_a: float, r_b: float, score_a: float, games: int, k: float) -> tuple[float, float]:
    exp_a = expected_score(r_a, r_b)
    delta = k * (score_a - games * exp_a)
    return r_a + delta, r_b - delta


def apply_pbt_mutation(cfg: TrainConfig):
    # Mutate a small set of impactful knobs when training stalls.
    sigma = max(0.01, cfg.pbt_sigma)
    lr_scale = float(np.exp(np.random.normal(0.0, sigma)))
    c_init_scale = float(np.exp(np.random.normal(0.0, sigma)))
    dir_shift = float(np.random.normal(0.0, sigma * 0.2))
    temp_shift = int(round(np.random.normal(0.0, max(1.0, cfg.temp_moves * sigma * 0.2))))

    cfg.lr = float(np.clip(cfg.lr * lr_scale, 1e-5, 5e-3))
    cfg.c_init = float(np.clip(cfg.c_init * c_init_scale, 0.5, 3.5))
    cfg.dirichlet_eps = float(np.clip(cfg.dirichlet_eps + dir_shift, 0.05, 0.4))
    cfg.temp_moves = int(np.clip(cfg.temp_moves + temp_shift, 4, 40))


def load_model_pool(pool_path: str) -> dict:
    if not os.path.exists(pool_path):
        return {"entries": {}, "best_name": ""}
    with open(pool_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model_pool(pool_path: str, pool: dict):
    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=True, indent=2)


def init_or_load_pool(cfg: TrainConfig, best_model: torch.nn.Module, start_iter: int) -> tuple[dict, str, str]:
    pool_dir = os.path.join(cfg.save_dir, "model_pool")
    os.makedirs(pool_dir, exist_ok=True)
    pool_path = os.path.join(cfg.save_dir, "model_pool.json")

    pool = load_model_pool(pool_path)
    entries = pool.setdefault("entries", {})
    best_name = pool.get("best_name", "")

    if best_name and best_name in entries:
        return pool, pool_dir, pool_path

    init_name = f"model_iter_{start_iter - 1:03d}"
    init_path = os.path.join(pool_dir, f"{init_name}.pt")
    torch.save({"model": best_model.state_dict(), "cfg": cfg.__dict__, "iter": start_iter - 1}, init_path)
    entries[init_name] = {"path": init_path, "elo": cfg.elo_init, "games": 0, "iter": start_iter - 1}
    pool["best_name"] = init_name
    save_model_pool(pool_path, pool)
    return pool, pool_dir, pool_path


def sample_pool_opponents(pool: dict, exclude_name: str, k: int) -> list[str]:
    entries = pool.get("entries", {})
    names = [n for n in entries.keys() if n != exclude_name]
    if not names or k <= 0:
        return []

    names.sort(key=lambda n: entries[n]["elo"], reverse=True)
    top_take = min(len(names), max(1, k // 2))
    top = names[:top_take]

    rest = names[top_take:]
    rand_take = min(len(rest), k - len(top))
    rand = random.sample(rest, rand_take) if rand_take > 0 else []
    out = top + rand
    if len(out) < min(k, len(names)):
        fill = [n for n in names if n not in out]
        out.extend(fill[: min(k, len(names)) - len(out)])
    return out


def prune_pool(pool: dict, keep_name: str, max_size: int):
    entries = pool.get("entries", {})
    if len(entries) <= max_size:
        return

    names = [n for n in entries.keys() if n != keep_name]
    names.sort(key=lambda n: entries[n]["iter"])  # oldest first

    remove_n = max(0, len(entries) - max_size)
    for name in names[:remove_n]:
        try:
            os.remove(entries[name]["path"])
        except OSError:
            pass
        entries.pop(name, None)


def load_model_from_file(path: str, device: str) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("cfg", {})
    model = AlphaZeroNet(
        board_size=cfg.get("board_size", 9),
        history_len=cfg.get("history_len", 8),
        channels=cfg.get("channels", 192),
        num_blocks=cfg.get("num_blocks", 12),
        use_se=cfg.get("use_se", True),
        value_hidden_dim=cfg.get("value_hidden_dim", 256),
    ).to(device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


def runtime_model_path_for_version(save_dir: str, version: int) -> str:
    return os.path.join(save_dir, f"_runtime_best_v{version:06d}.pt")


def save_runtime_model_snapshot(model: torch.nn.Module, save_dir: str, version: int):
    path = runtime_model_path_for_version(save_dir, version)
    torch.save({"model": model.state_dict()}, path)
    return path


def run_training(cfg: TrainConfig):
    set_seed(cfg.seed)
    if cfg.train_torch_threads > 0:
        torch.set_num_threads(cfg.train_torch_threads)
        torch.set_num_interop_threads(1)
    os.makedirs(cfg.save_dir, exist_ok=True)

    best_model = build_model(cfg).to(cfg.device)
    buffer = ReplayBuffer(cfg.replay_size)
    start_iter = 1

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=cfg.device)
        if "best_model" in ckpt:
            best_model.load_state_dict(ckpt["best_model"])
        else:
            best_model.load_state_dict(ckpt["model"])
        start_iter = int(ckpt.get("iter", 0)) + 1
        if cfg.save_buffer and "buffer" in ckpt:
            buffer.extend(ckpt["buffer"])
        print(f"[Resume] from {cfg.resume}, next_iter={start_iter}, buffer={len(buffer)}")

    pool, pool_dir, pool_path = init_or_load_pool(cfg, best_model, start_iter)
    runtime_model_version = 0
    runtime_model_path = save_runtime_model_snapshot(best_model, cfg.save_dir, runtime_model_version)

    async_executor = None
    selfplay_pool = None
    pending_selfplay: deque = deque()
    next_selfplay_iter = start_iter
    no_promote_iters = 0
    recent_keys: deque[bytes] = deque()
    recent_set: set[bytes] = set()

    if cfg.selfplay_workers > 1:
        selfplay_pool = ProcessPoolExecutor(max_workers=cfg.selfplay_workers)

    if cfg.async_selfplay:
        async_executor = ThreadPoolExecutor(max_workers=1)
        prefetch = max(1, cfg.async_prefetch_batches)
        while next_selfplay_iter <= cfg.iterations and len(pending_selfplay) < prefetch:
            pending_selfplay.append(
                submit_async_selfplay(
                    async_executor,
                    best_model,
                    cfg,
                    next_selfplay_iter,
                    runtime_model_path,
                    runtime_model_version,
                    selfplay_pool,
                )
            )
            next_selfplay_iter += 1

    try:
        for it in range(start_iter, cfg.iterations + 1):
            best_model.eval()

            if cfg.async_selfplay:
                if not pending_selfplay:
                    if next_selfplay_iter <= cfg.iterations:
                        pending_selfplay.append(
                            submit_async_selfplay(
                                async_executor,
                                best_model,
                                cfg,
                                next_selfplay_iter,
                                runtime_model_path,
                                runtime_model_version,
                                selfplay_pool,
                            )
                        )
                        next_selfplay_iter += 1
                    else:
                        raise RuntimeError("Async self-play queue underrun with no remaining iterations to schedule.")
                fut = pending_selfplay.popleft()
                new_samples, black_wins, white_wins, draws = fut.result()
                filtered = filter_selfplay_samples(new_samples, recent_keys, recent_set)
                buffer.extend(filtered)
            else:
                new_samples, black_wins, white_wins, draws = run_selfplay(
                    best_model,
                    cfg,
                    it,
                    pool=selfplay_pool,
                    model_path=runtime_model_path,
                    model_version=runtime_model_version,
                )
                filtered = filter_selfplay_samples(new_samples, recent_keys, recent_set)
                buffer.extend(filtered)

            if len(buffer) < cfg.batch_size:
                print(f"[Iter {it}] buffer too small: {len(buffer)} < {cfg.batch_size}, skip training")
                continue

            candidate_model = build_model(cfg).to(cfg.device)
            candidate_model.load_state_dict(copy.deepcopy(best_model.state_dict()))

            loss_mean, pol_mean, val_mean = train_candidate(candidate_model, buffer, cfg)

            promoted = True
            arena_w = arena_l = arena_d = 0
            if cfg.arena_games > 0:
                candidate_model.eval()
                best_model.eval()
                if cfg.arena_sprt:
                    arena_w, arena_l, arena_d, promoted = play_match_sprt(
                        candidate_model,
                        best_model,
                        cfg,
                        cfg.arena_games,
                        cfg.promote_threshold,
                        cfg.arena_sprt_delta,
                    )
                else:
                    arena_w, arena_l, arena_d = play_match(candidate_model, best_model, cfg, cfg.arena_games)
                    total_games = max(1, arena_w + arena_l + arena_d)
                    mean_score = (float(arena_w) + 0.5 * float(arena_d)) / float(total_games)
                    promoted = mean_score >= cfg.promote_threshold

            elo_logs = []
            benchmark_logs = []
            if promoted:
                best_model.load_state_dict(copy.deepcopy(candidate_model.state_dict()))
                no_promote_iters = 0
                runtime_model_version += 1
                runtime_model_path = save_runtime_model_snapshot(best_model, cfg.save_dir, runtime_model_version)

                best_name = f"model_iter_{it:03d}"
                best_path = os.path.join(pool_dir, f"{best_name}.pt")
                torch.save({"model": best_model.state_dict(), "cfg": cfg.__dict__, "iter": it}, best_path)

                entries = pool["entries"]
                prev_best = pool.get("best_name", "")
                base_elo = entries.get(prev_best, {}).get("elo", cfg.elo_init)
                entries[best_name] = {"path": best_path, "elo": float(base_elo), "games": 0, "iter": it}
                pool["best_name"] = best_name

                opp_names = sample_pool_opponents(pool, exclude_name=best_name, k=cfg.pool_opponents)
                for opp_name in opp_names:
                    opp_path = entries[opp_name]["path"]
                    opp_model, opp_cfg = load_model_from_file(opp_path, cfg.device)
                    if int(opp_cfg.get("board_size", cfg.board_size)) != cfg.board_size:
                        continue

                    w, l, d = play_match(best_model, opp_model, cfg, cfg.elo_games_per_opponent)
                    games = w + l + d
                    score = float(w) + 0.5 * float(d)

                    r_new = float(entries[best_name]["elo"])
                    r_opp = float(entries[opp_name]["elo"])
                    r_new, r_opp = update_elo_pair(r_new, r_opp, score_a=score, games=games, k=cfg.elo_k)
                    entries[best_name]["elo"] = r_new
                    entries[opp_name]["elo"] = r_opp
                    entries[best_name]["games"] += games
                    entries[opp_name]["games"] += games

                    elo_logs.append((opp_name, w, l, d, r_new, r_opp))

                prune_pool(pool, keep_name=best_name, max_size=cfg.pool_max_size)
                save_model_pool(pool_path, pool)

                torch.save({"model": best_model.state_dict(), "cfg": cfg.__dict__, "iter": it}, os.path.join(cfg.save_dir, "best.pt"))
                bench_names = sample_pool_opponents(pool, exclude_name=best_name, k=cfg.benchmark_opponents)
                for bench in bench_names:
                    bench_model, bench_cfg = load_model_from_file(pool["entries"][bench]["path"], cfg.device)
                    if int(bench_cfg.get("board_size", cfg.board_size)) != cfg.board_size:
                        continue
                    bw, bl, bd = play_match(best_model, bench_model, cfg, cfg.benchmark_games_per_opponent)
                    decisive = max(1, bw + bl)
                    benchmark_logs.append((bench, bw, bl, bd, bw / decisive))
            else:
                no_promote_iters += 1
                if cfg.pbt and no_promote_iters >= max(1, cfg.pbt_patience):
                    old = (cfg.lr, cfg.c_init, cfg.dirichlet_eps, cfg.temp_moves)
                    apply_pbt_mutation(cfg)
                    no_promote_iters = 0
                    print(
                        "[PBT] mutate "
                        f"lr {old[0]:.2e}->{cfg.lr:.2e}, "
                        f"c_init {old[1]:.3f}->{cfg.c_init:.3f}, "
                        f"dir_eps {old[2]:.3f}->{cfg.dirichlet_eps:.3f}, "
                        f"temp_moves {old[3]}->{cfg.temp_moves}"
                    )

            ckpt = {
                "iter": it,
                "model": best_model.state_dict(),
                "best_model": best_model.state_dict(),
                "candidate_model": candidate_model.state_dict(),
                "cfg": cfg.__dict__,
                "buffer_size": len(buffer),
                "promoted": promoted,
                "arena": {"wins": arena_w, "losses": arena_l, "draws": arena_d},
                "pool_best": pool.get("best_name", ""),
            }
            if cfg.save_buffer:
                ckpt["buffer"] = list(buffer.data)

            iter_ckpt = os.path.join(cfg.save_dir, f"iter_{it:03d}.pt")
            torch.save(ckpt, iter_ckpt)

            log_line = (
                f"[Iter {it}] selfplay B/W/D={black_wins}/{white_wins}/{draws} workers={cfg.selfplay_workers} "
                f"async={cfg.async_selfplay} "
                f"buffer={len(buffer)} loss={loss_mean:.4f} pol={pol_mean:.4f} val={val_mean:.4f} "
                f"arena={arena_w}/{arena_l}/{arena_d} promoted={promoted} saved={iter_ckpt}"
            )
            print(log_line)

            if promoted and elo_logs:
                entries = pool["entries"]
                best_name = pool.get("best_name", "")
                print(f"[ELO] {best_name} rating={entries[best_name]['elo']:.1f}")
                for opp_name, w, l, d, r_new, r_opp in elo_logs:
                    print(f"[ELO] vs {opp_name}: {w}/{l}/{d}, new={r_new:.1f}, opp={r_opp:.1f}")
            if promoted and benchmark_logs:
                for bench, bw, bl, bd, wr in benchmark_logs:
                    print(f"[Benchmark] vs {bench}: {bw}/{bl}/{bd}, winrate={wr:.3f}")

            if cfg.async_selfplay:
                prefetch = max(1, cfg.async_prefetch_batches)
                while next_selfplay_iter <= cfg.iterations and len(pending_selfplay) < prefetch:
                    pending_selfplay.append(
                        submit_async_selfplay(
                            async_executor,
                            best_model,
                            cfg,
                            next_selfplay_iter,
                            runtime_model_path,
                            runtime_model_version,
                            selfplay_pool,
                        )
                    )
                    next_selfplay_iter += 1
    finally:
        if async_executor is not None:
            async_executor.shutdown(wait=False, cancel_futures=True)
        if selfplay_pool is not None:
            selfplay_pool.shutdown(wait=False, cancel_futures=True)


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Competition-oriented AlphaZero-style 9x9 Go training")

    p.add_argument("--board-size", type=int, default=9)
    p.add_argument("--komi", type=float, default=5.5)
    p.add_argument("--history-len", type=int, default=8)

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--selfplay-games-per-iter", type=int, default=40)
    p.add_argument("--selfplay-workers", type=int, default=1)
    p.add_argument("--async-selfplay", action="store_true")
    p.add_argument("--async-prefetch-batches", type=int, default=2)
    p.add_argument("--train-torch-threads", type=int, default=0)
    p.add_argument("--worker-torch-threads", type=int, default=1)
    p.add_argument("--mcts-simulations", type=int, default=192)
    p.add_argument("--eval-simulations", type=int, default=256)
    p.add_argument("--mcts-inference-batch", type=int, default=8)
    p.add_argument("--mcts-topk-expand", type=int, default=0)
    p.add_argument("--mcts-virtual-loss", type=float, default=1.0)
    p.add_argument("--mcts-root-parallelism", type=int, default=1)
    p.add_argument("--temp-moves", type=int, default=20)
    p.add_argument("--temp-mid-moves", type=int, default=40)
    p.add_argument("--temp-start", type=float, default=1.0)
    p.add_argument("--temp-mid", type=float, default=0.4)
    p.add_argument("--temp-end", type=float, default=0.05)
    p.add_argument("--resign-threshold", type=float, default=0.98)
    p.add_argument("--disable-resign", action="store_true")

    p.add_argument("--replay-size", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-steps-per-iter", type=int, default=800)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--value-huber-delta", type=float, default=1.0)
    p.add_argument("--policy-entropy-coeff", type=float, default=1e-3)
    p.add_argument("--policy-target-smoothing", type=float, default=0.0)

    p.add_argument("--channels", type=int, default=192)
    p.add_argument("--num-blocks", type=int, default=12)
    p.add_argument("--no-se", action="store_true")
    p.add_argument("--value-hidden-dim", type=int, default=256)

    p.add_argument("--arena-games", type=int, default=40)
    p.add_argument("--promote-threshold", type=float, default=0.55)
    p.add_argument("--no-arena-sprt", action="store_true")
    p.add_argument("--arena-sprt-delta", type=float, default=0.05)

    p.add_argument("--no-augment-symmetry", action="store_true")

    p.add_argument("--c-base", type=float, default=19652.0)
    p.add_argument("--c-init", type=float, default=1.25)
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--dirichlet-eps", type=float, default=0.25)

    p.add_argument("--pool-max-size", type=int, default=20)
    p.add_argument("--pool-opponents", type=int, default=4)
    p.add_argument("--elo-games-per-opponent", type=int, default=12)
    p.add_argument("--benchmark-opponents", type=int, default=4)
    p.add_argument("--benchmark-games-per-opponent", type=int, default=8)
    p.add_argument("--elo-k", type=float, default=24.0)
    p.add_argument("--elo-init", type=float, default=1500.0)
    p.add_argument("--pbt", action="store_true")
    p.add_argument("--pbt-patience", type=int, default=3)
    p.add_argument("--pbt-sigma", type=float, default=0.15)
    p.add_argument("--opening-book-prob", type=float, default=0.0)
    p.add_argument("--opening-book-max-moves", type=int, default=0)

    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save-buffer", action="store_true")

    args = p.parse_args()
    return TrainConfig(
        board_size=args.board_size,
        komi=args.komi,
        history_len=args.history_len,
        device=args.device,
        seed=args.seed,
        iterations=args.iterations,
        selfplay_games_per_iter=args.selfplay_games_per_iter,
        selfplay_workers=args.selfplay_workers,
        async_selfplay=args.async_selfplay,
        async_prefetch_batches=args.async_prefetch_batches,
        train_torch_threads=args.train_torch_threads,
        worker_torch_threads=args.worker_torch_threads,
        mcts_simulations=args.mcts_simulations,
        eval_simulations=args.eval_simulations,
        mcts_inference_batch=args.mcts_inference_batch,
        mcts_topk_expand=args.mcts_topk_expand,
        mcts_virtual_loss=args.mcts_virtual_loss,
        mcts_root_parallelism=max(1, args.mcts_root_parallelism),
        temp_moves=args.temp_moves,
        temp_mid_moves=args.temp_mid_moves,
        temp_start=args.temp_start,
        temp_mid=args.temp_mid,
        temp_end=args.temp_end,
        resign_threshold=args.resign_threshold,
        disable_resign=args.disable_resign,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        train_steps_per_iter=args.train_steps_per_iter,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        use_amp=(not args.no_amp) and (args.use_amp or args.device.startswith("cuda")),
        value_huber_delta=args.value_huber_delta,
        policy_entropy_coeff=args.policy_entropy_coeff,
        policy_target_smoothing=max(0.0, args.policy_target_smoothing),
        channels=args.channels,
        num_blocks=args.num_blocks,
        use_se=not args.no_se,
        value_hidden_dim=max(1, args.value_hidden_dim),
        arena_games=args.arena_games,
        promote_threshold=args.promote_threshold,
        arena_sprt=not args.no_arena_sprt,
        arena_sprt_delta=args.arena_sprt_delta,
        augment_symmetry=not args.no_augment_symmetry,
        c_base=args.c_base,
        c_init=args.c_init,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps,
        pool_max_size=args.pool_max_size,
        pool_opponents=args.pool_opponents,
        elo_games_per_opponent=args.elo_games_per_opponent,
        benchmark_opponents=args.benchmark_opponents,
        benchmark_games_per_opponent=args.benchmark_games_per_opponent,
        elo_k=args.elo_k,
        elo_init=args.elo_init,
        pbt=args.pbt,
        pbt_patience=args.pbt_patience,
        pbt_sigma=args.pbt_sigma,
        opening_book_prob=float(np.clip(args.opening_book_prob, 0.0, 1.0)),
        opening_book_max_moves=max(0, args.opening_book_max_moves),
        save_dir=args.save_dir,
        resume=args.resume,
        save_buffer=args.save_buffer,
    )


def main():
    cfg = parse_args()
    run_training(cfg)


if __name__ == "__main__":
    main()
