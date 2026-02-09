from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np


BLACK = 1
WHITE = -1
EMPTY = 0


def action_size(board_size: int) -> int:
    return board_size * board_size + 1  # + pass


def encode_action(board_size: int, move: Optional[tuple[int, int]]) -> int:
    if move is None:
        return board_size * board_size
    r, c = move
    return r * board_size + c


def decode_action(board_size: int, action: int) -> Optional[tuple[int, int]]:
    if action == board_size * board_size:
        return None
    return divmod(action, board_size)


def tensor_channels(history_len: int) -> int:
    # own history + opp history + to-play + legal + last move + own caps + opp caps
    return history_len * 2 + 5


@dataclass(frozen=True)
class GoState:
    board: np.ndarray
    to_play: int = BLACK
    consecutive_passes: int = 0
    move_count: int = 0
    komi: float = 5.5
    history_len: int = 8
    board_history: tuple[np.ndarray, ...] = ()
    seen_hashes: frozenset[int] = frozenset()
    black_captures: int = 0
    white_captures: int = 0
    last_move: Optional[tuple[int, int]] = None
    _legal_actions_cache: tuple[int, ...] | None = field(default=None, init=False, repr=False, compare=False)
    _tensor_cache: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    @staticmethod
    def _state_hash(board: np.ndarray, to_play: int) -> int:
        # Situational superko hash: board + side to play.
        h = hashlib.blake2b(digest_size=8)
        h.update(board.tobytes())
        h.update(bytes([to_play + 2]))
        return int.from_bytes(h.digest(), "little", signed=False)

    @staticmethod
    def new_game(board_size: int = 9, komi: float = 5.5, history_len: int = 8) -> "GoState":
        board = np.zeros((board_size, board_size), dtype=np.int8)
        init_hash = GoState._state_hash(board, BLACK)
        return GoState(
            board=board,
            komi=komi,
            history_len=history_len,
            board_history=(board.copy(),),
            seen_hashes=frozenset({init_hash}),
            black_captures=0,
            white_captures=0,
            last_move=None,
        )

    @property
    def board_size(self) -> int:
        return self.board.shape[0]

    def in_bounds(self, r: int, c: int) -> bool:
        n = self.board_size
        return 0 <= r < n and 0 <= c < n

    def neighbors(self, r: int, c: int) -> Iterable[tuple[int, int]]:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc):
                yield nr, nc

    def _group_and_liberties(self, board: np.ndarray, r: int, c: int) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
        color = board[r, c]
        stack = [(r, c)]
        seen = {(r, c)}
        group: list[tuple[int, int]] = []
        liberties: set[tuple[int, int]] = set()
        while stack:
            cr, cc = stack.pop()
            group.append((cr, cc))
            for nr, nc in self.neighbors(cr, cc):
                v = board[nr, nc]
                if v == EMPTY:
                    liberties.add((nr, nc))
                elif v == color and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return group, liberties

    def _remove_dead_groups_around(self, board: np.ndarray, r: int, c: int, captured_color: int) -> int:
        captured = 0
        checked: set[tuple[int, int]] = set()
        for nr, nc in self.neighbors(r, c):
            if board[nr, nc] != captured_color or (nr, nc) in checked:
                continue
            group, libs = self._group_and_liberties(board, nr, nc)
            checked.update(group)
            if not libs:
                for gr, gc in group:
                    board[gr, gc] = EMPTY
                captured += len(group)
        return captured

    def is_legal(self, move: Optional[tuple[int, int]]) -> bool:
        if move is None:
            return True

        r, c = move
        if not self.in_bounds(r, c) or self.board[r, c] != EMPTY:
            return False

        board = self.board.copy()
        board[r, c] = self.to_play
        self._remove_dead_groups_around(board, r, c, -self.to_play)

        _, my_libs = self._group_and_liberties(board, r, c)
        if not my_libs:
            return False  # suicide

        # Situational superko: no repeated (board + side to play).
        next_hash = self._state_hash(board, -self.to_play)
        if next_hash in self.seen_hashes:
            return False

        return True

    def legal_actions(self) -> list[int]:
        if self._legal_actions_cache is not None:
            return list(self._legal_actions_cache)

        n = self.board_size
        acts: list[int] = []
        for r in range(n):
            for c in range(n):
                if self.is_legal((r, c)):
                    acts.append(encode_action(n, (r, c)))
        acts.append(encode_action(n, None))
        object.__setattr__(self, "_legal_actions_cache", tuple(acts))
        return acts

    def play(self, move: Optional[tuple[int, int]]) -> "GoState":
        if move is None:
            new_board = self.board.copy()
            new_to_play = -self.to_play
            next_hash = self._state_hash(new_board, new_to_play)
            new_hist = (new_board.copy(),) + self.board_history[: self.history_len - 1]
            return GoState(
                board=new_board,
                to_play=new_to_play,
                consecutive_passes=self.consecutive_passes + 1,
                move_count=self.move_count + 1,
                komi=self.komi,
                history_len=self.history_len,
                board_history=new_hist,
                seen_hashes=self.seen_hashes | {next_hash},
                black_captures=self.black_captures,
                white_captures=self.white_captures,
                last_move=None,
            )

        if not self.is_legal(move):
            raise ValueError(f"Illegal move: {move}")

        r, c = move
        new_board = self.board.copy()
        new_board[r, c] = self.to_play
        captured = self._remove_dead_groups_around(new_board, r, c, -self.to_play)

        new_to_play = -self.to_play
        next_hash = self._state_hash(new_board, new_to_play)
        new_hist = (new_board.copy(),) + self.board_history[: self.history_len - 1]

        return GoState(
            board=new_board,
            to_play=new_to_play,
            consecutive_passes=0,
            move_count=self.move_count + 1,
            komi=self.komi,
            history_len=self.history_len,
            board_history=new_hist,
            seen_hashes=self.seen_hashes | {next_hash},
            black_captures=(self.black_captures + captured) if self.to_play == BLACK else self.black_captures,
            white_captures=(self.white_captures + captured) if self.to_play == WHITE else self.white_captures,
            last_move=move,
        )

    def is_terminal(self) -> bool:
        return self.consecutive_passes >= 2 or self.move_count >= self.board_size * self.board_size * 2

    def _territory_score(self) -> tuple[int, int]:
        n = self.board_size
        visited = np.zeros((n, n), dtype=bool)
        black_territory = 0
        white_territory = 0

        for r in range(n):
            for c in range(n):
                if visited[r, c] or self.board[r, c] != EMPTY:
                    continue

                region = [(r, c)]
                stack = [(r, c)]
                visited[r, c] = True
                borders = set()

                while stack:
                    cr, cc = stack.pop()
                    for nr, nc in self.neighbors(cr, cc):
                        v = self.board[nr, nc]
                        if v == EMPTY and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                            region.append((nr, nc))
                        elif v != EMPTY:
                            borders.add(v)

                if borders == {BLACK}:
                    black_territory += len(region)
                elif borders == {WHITE}:
                    white_territory += len(region)

        return black_territory, white_territory

    def score(self) -> tuple[float, float]:
        # Area-like scoring (stones + territory) with komi.
        black_stones = int(np.sum(self.board == BLACK))
        white_stones = int(np.sum(self.board == WHITE))
        bt, wt = self._territory_score()
        black = black_stones + bt
        white = white_stones + wt + self.komi
        return float(black), float(white)

    def winner(self) -> int:
        black, white = self.score()
        if black > white:
            return BLACK
        if white > black:
            return WHITE
        return 0

    def outcome_for_current_player(self) -> float:
        w = self.winner()
        if w == 0:
            return 0.0
        return 1.0 if w == self.to_play else -1.0

    def to_tensor(self) -> np.ndarray:
        if self._tensor_cache is not None:
            return self._tensor_cache

        n = self.board_size
        planes = np.zeros((tensor_channels(self.history_len), n, n), dtype=np.float32)

        hist = np.zeros((self.history_len, n, n), dtype=np.int8)
        hist_take = min(self.history_len, len(self.board_history))
        for i in range(hist_take):
            hist[i] = self.board_history[i]
        own_hist = (hist == self.to_play).astype(np.float32, copy=False)
        opp_hist = (hist == -self.to_play).astype(np.float32, copy=False)

        planes[: self.history_len] = own_hist
        planes[self.history_len : 2 * self.history_len] = opp_hist

        to_play_black = planes[2 * self.history_len]
        to_play_black.fill(1.0 if self.to_play == BLACK else 0.0)

        legal = planes[2 * self.history_len + 1]
        for a in self.legal_actions():
            if a == n * n:
                continue
            r, c = divmod(a, n)
            legal[r, c] = 1.0

        last_move_plane = planes[2 * self.history_len + 2]
        if self.last_move is not None:
            lr, lc = self.last_move
            last_move_plane[lr, lc] = 1.0

        max_pts = float(max(1, n * n))
        own_caps = float(self.black_captures if self.to_play == BLACK else self.white_captures) / max_pts
        opp_caps = float(self.white_captures if self.to_play == BLACK else self.black_captures) / max_pts
        planes[2 * self.history_len + 3].fill(own_caps)
        planes[2 * self.history_len + 4].fill(opp_caps)

        object.__setattr__(self, "_tensor_cache", planes)
        return planes
