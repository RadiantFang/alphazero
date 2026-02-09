from __future__ import annotations

import argparse

import torch

from go_zero.go import GoState, decode_action
from go_zero.mcts import MCTS
from go_zero.model import AlphaZeroNet


def print_board(state: GoState):
    n = state.board_size
    print("   " + " ".join(f"{i}" for i in range(n)))
    for r in range(n):
        row = []
        for c in range(n):
            v = state.board[r, c]
            if v == 1:
                row.append("X")
            elif v == -1:
                row.append("O")
            else:
                row.append(".")
        print(f"{r:2d} " + " ".join(row))


def parse_move(s: str, n: int):
    s = s.strip().lower()
    if s in {"pass", "p"}:
        return None
    parts = s.split()
    if len(parts) != 2:
        return "invalid"
    try:
        r, c = int(parts[0]), int(parts[1])
    except ValueError:
        return "invalid"
    if not (0 <= r < n and 0 <= c < n):
        return "invalid"
    return (r, c)


def model_from_checkpoint(args):
    if not args.checkpoint:
        return AlphaZeroNet(board_size=args.board_size, value_hidden_dim=args.value_hidden_dim).to(args.device), None

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt.get("cfg", {})

    model = AlphaZeroNet(
        board_size=cfg.get("board_size", args.board_size),
        history_len=cfg.get("history_len", 8),
        channels=cfg.get("channels", 192),
        num_blocks=cfg.get("num_blocks", 12),
        use_se=cfg.get("use_se", True),
        value_hidden_dim=cfg.get("value_hidden_dim", args.value_hidden_dim),
    ).to(args.device)

    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    return model, cfg


def main():
    ap = argparse.ArgumentParser(description="Play 9x9 Go against AlphaZero-style MCTS agent")
    ap.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint .pt")
    ap.add_argument("--board-size", type=int, default=9)
    ap.add_argument("--komi", type=float, default=5.5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--sims", type=int, default=256)
    ap.add_argument("--human-color", type=str, choices=["black", "white"], default="black")
    ap.add_argument("--value-hidden-dim", type=int, default=256)
    args = ap.parse_args()

    model, cfg = model_from_checkpoint(args)
    model.eval()

    board_size = cfg.get("board_size", args.board_size) if cfg else args.board_size
    history_len = cfg.get("history_len", 8) if cfg else 8
    komi = cfg.get("komi", args.komi) if cfg else args.komi

    mcts = MCTS(model=model, board_size=board_size, num_simulations=args.sims, device=args.device)
    human = 1 if args.human_color == "black" else -1
    state = GoState.new_game(board_size=board_size, komi=komi, history_len=history_len)

    while not state.is_terminal():
        print_board(state)
        side = "Black(X)" if state.to_play == 1 else "White(O)"
        print(f"Turn: {side}")

        if state.to_play == human:
            raw = input("Your move (row col / pass): ")
            move = parse_move(raw, board_size)
            if move == "invalid" or not state.is_legal(move):
                print("Invalid move, try again.")
                continue
            state = state.play(move)
        else:
            _, action = mcts.run(state, temperature=1e-6, add_exploration_noise=False)
            move = decode_action(board_size, action)
            print(f"AI move: {'pass' if move is None else move}")
            state = state.play(move)

    print_board(state)
    b, w = state.score()
    winner = state.winner()
    if winner == 1:
        print(f"Game over. Black wins. Score B={b:.1f}, W={w:.1f}")
    elif winner == -1:
        print(f"Game over. White wins. Score B={b:.1f}, W={w:.1f}")
    else:
        print(f"Game over. Draw. Score B={b:.1f}, W={w:.1f}")


if __name__ == "__main__":
    main()
