from __future__ import annotations

import glob
import os
import re
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from PyQt6.QtCore import QPointF, Qt, QTimer
    from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QRadialGradient
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QSizePolicy,
    )
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyQt6 is required. Install with: pip install PyQt6"
    ) from e

from go_zero.go import BLACK, WHITE, GoState, decode_action
from go_zero.mcts import MCTS
from go_zero.model import AlphaZeroNet

ROLE_HUMAN = "__human__"


@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    cfg: dict


class BoardWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._state: GoState | None = None
        self.margin = 28
        self.cell = 56
        self.stone_radius = 22
        self._x0 = 28
        self._y0 = 28
        self._board_w = 0
        self.on_click = None
        self.setMinimumSize(560, 560)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_state(self, state: GoState):
        self._state = state
        self.update()

    def _update_geometry(self):
        if self._state is None:
            return
        n = self._state.board_size
        min_side = min(self.width(), self.height())
        outer_margin = max(20, int(min_side * 0.08))
        board_px = max(120, min_side - 2 * outer_margin)
        self.cell = max(16, board_px // max(1, n - 1))
        self._board_w = self.cell * (n - 1)
        # Keep the board centered when window is resized.
        self._x0 = (self.width() - self._board_w) // 2
        self._y0 = (self.height() - self._board_w) // 2
        self.stone_radius = max(6, int(self.cell * 0.42))

    def mousePressEvent(self, event):
        if self._state is None or event.button() != Qt.MouseButton.LeftButton:
            return
        self._update_geometry()
        n = self._state.board_size
        px = event.position().x()
        py = event.position().y()
        c = round((px - self._x0) / self.cell)
        r = round((py - self._y0) / self.cell)
        if not (0 <= r < n and 0 <= c < n):
            return
        ix = self._x0 + c * self.cell
        iy = self._y0 + r * self.cell
        if abs(px - ix) <= self.cell * 0.45 and abs(py - iy) <= self.cell * 0.45 and self.on_click:
            self.on_click(r, c)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#e3c48f"))
        if self._state is None:
            return

        n = self._state.board_size
        self._update_geometry()
        x0 = self._x0
        y0 = self._y0
        board_w = self._board_w

        pen = QPen(QColor("#51341f"))
        pen.setWidth(2)
        p.setPen(pen)
        for i in range(n):
            x = x0 + i * self.cell
            y = y0 + i * self.cell
            p.drawLine(x0, y, x0 + board_w, y)
            p.drawLine(x, y0, x, y0 + board_w)

        p.setPen(QColor("#312010"))
        p.setFont(QFont("Sans Serif", max(14, int(self.cell * 0.34))))
        top_y = y0 - max(8, int(self.cell * 0.35))
        left_x = x0 - max(16, int(self.cell * 0.4))
        for i in range(n):
            p.drawText(x0 + i * self.cell - 4, top_y, str(i))
            p.drawText(left_x, y0 + i * self.cell + 4, str(i))

        last_move = self._state.last_move
        for r in range(n):
            for c in range(n):
                v = int(self._state.board[r, c])
                if v == 0:
                    continue
                cx = x0 + c * self.cell
                cy = y0 + r * self.cell
                if v == BLACK:
                    grad = QRadialGradient(QPointF(cx - 6, cy - 6), self.stone_radius + 2)
                    grad.setColorAt(0.0, QColor("#6e6e6e"))
                    grad.setColorAt(1.0, QColor("#090909"))
                    p.setPen(QPen(QColor("#161616"), 1))
                else:
                    grad = QRadialGradient(QPointF(cx - 6, cy - 6), self.stone_radius + 2)
                    grad.setColorAt(0.0, QColor("#ffffff"))
                    grad.setColorAt(1.0, QColor("#d8d8d8"))
                    p.setPen(QPen(QColor("#888888"), 1))
                p.setBrush(grad)
                p.drawEllipse(QPointF(cx, cy), self.stone_radius, self.stone_radius)

                if last_move is not None and (r, c) == last_move:
                    marker = QColor("#d33f2a") if v == BLACK else QColor("#2a2a2a")
                    p.setBrush(marker)
                    p.setPen(QPen(marker, 1))
                    marker_radius = max(6, int(self.cell * 0.14))
                    p.drawEllipse(QPointF(cx, cy), marker_radius, marker_radius)


class BattleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlphaZero Go Battle (PyQt)")
        self.resize(1180, 760)
        self.state: GoState | None = None
        self.history: list[GoState] = []
        self.logs: list[str] = []
        self.model_cache: dict[tuple[str, str], ModelBundle] = {}
        self.mcts_cache: dict[tuple[str, str, int, int], MCTS] = {}
        self.auto_playing = False
        self.manual_game_end = False
        self.game_started = False
        self.infer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ui_mcts")
        self.pending_infer_future: Future | None = None
        self.pending_infer_snapshot: tuple[int, int, bytes] | None = None
        self.pending_infer_token = 0
        self.log_timer = QTimer(self)
        self.log_timer.setSingleShot(True)
        self.log_timer.timeout.connect(self._flush_log_view)
        self._log_dirty = False

        self.board = BoardWidget()
        self.board.on_click = self.on_board_click
        self.turn_label = QLabel("")
        self.score_label = QLabel("")
        self.terminal_label = QLabel("")
        self.ai_status_label = QLabel("")
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        self.board_size = QSpinBox()
        self.board_size.setRange(5, 19)
        self.board_size.setValue(9)
        self.komi = QDoubleSpinBox()
        self.komi.setRange(0.0, 20.0)
        self.komi.setSingleStep(0.5)
        self.komi.setValue(5.5)
        self.sims = QSpinBox()
        self.sims.setRange(1, 2048)
        self.sims.setValue(192)
        self.device = QComboBox()
        self.device.addItems(["cuda", "cpu"])
        if not torch.cuda.is_available():
            self.device.setCurrentText("cpu")
        self.auto_mode = QComboBox()
        self.auto_mode.addItem("手动", False)
        self.auto_mode.addItem("智能自动", True)

        self.black_role = QComboBox()
        self.white_role = QComboBox()
        self.refresh_btn = QPushButton("刷新模型列表")
        self.new_btn = QPushButton("开始对局")
        self.end_btn = QPushButton("结束对局")
        self.undo_btn = QPushButton("悔棋")
        self.pass_btn = QPushButton("停一手")
        self.ai_btn = QPushButton("AI 下一手")
        self.auto_btn = QPushButton("自动到终局")
        self.export_sgf_btn = QPushButton("导出 SGF")
        self.import_sgf_btn = QPushButton("导入 SGF")

        self._build_ui()
        self._apply_font_theme()
        self._configure_runtime_threads()
        self._bind_events()
        self.refresh_checkpoints()
        self._refresh_view()

    def _apply_font_theme(self):
        # Larger default typography for high-DPI and zoomed windows.
        self.setFont(QFont("Microsoft YaHei UI", 14))
        self.setStyleSheet(
            """
            QLabel { font-size: 17px; }
            QGroupBox { font-size: 17px; font-weight: 600; }
            QComboBox, QSpinBox, QDoubleSpinBox, QPushButton { font-size: 16px; min-height: 36px; }
            QTextEdit { font-size: 16px; }
            """
        )
        self.turn_label.setStyleSheet("font-size: 22px; font-weight: 700;")
        self.score_label.setStyleSheet("font-size: 19px;")
        self.terminal_label.setStyleSheet("font-size: 19px; font-weight: 600; color: #8b1a1a;")
        self.ai_status_label.setStyleSheet("font-size: 18px; font-weight: 600; color: #8a5a00;")

    def _build_ui(self):
        root = QWidget()
        layout = QHBoxLayout(root)
        self.setCentralWidget(root)

        left = QVBoxLayout()
        left.addWidget(self.board, stretch=8)
        left.addWidget(self.turn_label)
        left.addWidget(self.score_label)
        left.addWidget(self.terminal_label)
        left.addWidget(self.ai_status_label)
        layout.addLayout(left, stretch=7)

        right = QVBoxLayout()
        cfg_box = QGroupBox("对局设置")
        cfg_grid = QFormLayout(cfg_box)
        cfg_grid.addRow("棋盘大小", self.board_size)
        cfg_grid.addRow("贴目", self.komi)
        cfg_grid.addRow("MCTS 模拟", self.sims)
        cfg_grid.addRow("设备", self.device)
        cfg_grid.addRow("自动模式", self.auto_mode)
        cfg_grid.addRow("黑方角色", self.black_role)
        cfg_grid.addRow("白方角色", self.white_role)
        right.addWidget(cfg_box)

        btn_grid = QGridLayout()
        btn_grid.addWidget(self.refresh_btn, 0, 0)
        btn_grid.addWidget(self.new_btn, 0, 1)
        btn_grid.addWidget(self.end_btn, 1, 0)
        btn_grid.addWidget(self.undo_btn, 1, 1)
        btn_grid.addWidget(self.pass_btn, 2, 0)
        btn_grid.addWidget(self.ai_btn, 2, 1)
        btn_grid.addWidget(self.auto_btn, 3, 0, 1, 2)
        btn_grid.addWidget(self.export_sgf_btn, 4, 0)
        btn_grid.addWidget(self.import_sgf_btn, 4, 1)
        right.addLayout(btn_grid)

        right.addWidget(QLabel("着手记录"))
        right.addWidget(self.log_view, stretch=4)
        layout.addLayout(right, stretch=4)

    def _bind_events(self):
        self.refresh_btn.clicked.connect(self.refresh_checkpoints)
        self.new_btn.clicked.connect(self.new_game)
        self.end_btn.clicked.connect(self.end_game)
        self.undo_btn.clicked.connect(self.undo_move)
        self.pass_btn.clicked.connect(self.pass_move)
        self.ai_btn.clicked.connect(self.ai_step)
        self.auto_btn.clicked.connect(self.toggle_auto_play)
        self.export_sgf_btn.clicked.connect(self.export_sgf)
        self.import_sgf_btn.clicked.connect(self.import_sgf)
        self.auto_mode.activated.connect(self._on_option_activated)
        self.black_role.activated.connect(self._on_option_activated)
        self.white_role.activated.connect(self._on_option_activated)
        self.auto_mode.currentIndexChanged.connect(self._on_option_changed)
        self.black_role.currentIndexChanged.connect(self._on_option_changed)
        self.white_role.currentIndexChanged.connect(self._on_option_changed)

    def _configure_runtime_threads(self):
        cpu_count = max(1, int(os.cpu_count() or 1))
        torch_threads = max(1, min(4, cpu_count // 2))
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch may reject resetting interop threads after initialization.
            pass

    def _cancel_pending_inference(self):
        self.pending_infer_token += 1
        if self.pending_infer_future is not None:
            self.pending_infer_future.cancel()
            self.pending_infer_future = None
            self.pending_infer_snapshot = None
        self._set_ai_status("")

    def refresh_checkpoints(self):
        self._cancel_pending_inference()
        self.model_cache.clear()
        self.mcts_cache.clear()
        options = [("真人", ROLE_HUMAN)]
        for p in sorted(glob.glob("checkpoints/**/*.pt", recursive=True)):
            if Path(p).name.startswith("_runtime_best_"):
                continue
            options.append((p, p))

        cur_b = self._role_value(self.black_role)
        cur_w = self._role_value(self.white_role)
        self.black_role.clear()
        self.white_role.clear()
        for txt, val in options:
            self.black_role.addItem(txt, val)
            self.white_role.addItem(txt, val)

        self._restore_role(self.black_role, cur_b)
        self._restore_role(self.white_role, cur_w)
        if self.white_role.currentIndex() == 0 and self.white_role.count() > 1:
            self.white_role.setCurrentIndex(1)

    @staticmethod
    def _restore_role(combo: QComboBox, target: Optional[str]):
        if target is None:
            return
        for i in range(combo.count()):
            if combo.itemData(i) == target:
                combo.setCurrentIndex(i)
                return

    @staticmethod
    def _role_value(combo: QComboBox) -> str:
        return str(combo.currentData())

    def _auto_mode_enabled(self) -> bool:
        return bool(self.auto_mode.currentData())

    def _on_option_activated(self, _index: int):
        combo = self.sender()
        if isinstance(combo, QComboBox):
            QTimer.singleShot(0, combo.hidePopup)
            QTimer.singleShot(0, combo.clearFocus)
        QTimer.singleShot(120, self._maybe_auto_step)

    def _on_option_changed(self, _index: int):
        QTimer.singleShot(120, self._maybe_auto_step)

    def _infer_history_len(self) -> int:
        for combo in (self.black_role, self.white_role):
            role = self._role_value(combo)
            if role == ROLE_HUMAN:
                continue
            cached = self.model_cache.get((role, self.device.currentText()))
            if cached is not None:
                return int(cached.cfg.get("history_len", 8))
            try:
                ckpt = torch.load(role, map_location="cpu")
                cfg = ckpt.get("cfg", {})
                return int(cfg.get("history_len", 8))
            except Exception:
                return 8
        return 8

    def new_game(self):
        self._cancel_pending_inference()
        self.auto_playing = False
        self.manual_game_end = False
        self.game_started = True
        self.auto_btn.setText("自动到终局")
        self.state = GoState.new_game(
            board_size=int(self.board_size.value()),
            komi=float(self.komi.value()),
            history_len=self._infer_history_len(),
        )
        self.history = [self.state]
        self.logs = []
        self._refresh_view()
        self._maybe_auto_step()

    def _moves_from_history(self) -> list[Optional[tuple[int, int]]]:
        if len(self.history) <= 1:
            return []
        return [st.last_move for st in self.history[1:]]

    @staticmethod
    def _move_to_sgf_coord(move: Optional[tuple[int, int]]) -> str:
        if move is None:
            return ""
        r, c = move
        return f"{chr(ord('a') + c)}{chr(ord('a') + r)}"

    @staticmethod
    def _sgf_coord_to_move(coord: str, board_size: int) -> Optional[tuple[int, int]]:
        if coord == "":
            return None
        if len(coord) != 2:
            raise ValueError(f"非法 SGF 坐标: {coord}")
        c = ord(coord[0]) - ord("a")
        r = ord(coord[1]) - ord("a")
        if not (0 <= r < board_size and 0 <= c < board_size):
            raise ValueError(f"SGF 坐标越界: {coord}")
        return (r, c)

    @staticmethod
    def _extract_sgf_prop(sgf_text: str, key: str) -> str:
        m = re.search(rf"{re.escape(key)}\[(.*?)\]", sgf_text, flags=re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    def export_sgf(self):
        if (not self.game_started) or self.state is None:
            QMessageBox.information(self, "提示", "当前没有可导出的对局。")
            return
        default_name = f"go_{self.state.board_size}x{self.state.board_size}.sgf"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出 SGF",
            default_name,
            "SGF Files (*.sgf);;All Files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".sgf"):
            path = f"{path}.sgf"

        moves = self._moves_from_history()
        sgf_parts = [
            "(",
            f";FF[4]GM[1]SZ[{int(self.state.board_size)}]KM[{float(self.state.komi):g}]CA[UTF-8]",
        ]
        color = "B"
        for mv in moves:
            sgf_parts.append(f";{color}[{self._move_to_sgf_coord(mv)}]")
            color = "W" if color == "B" else "B"
        sgf_parts.append(")")
        sgf_text = "".join(sgf_parts)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(sgf_text)
            QMessageBox.information(self, "导出成功", f"SGF 已保存到:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def import_sgf(self):
        self._cancel_pending_inference()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入 SGF",
            "",
            "SGF Files (*.sgf);;All Files (*)",
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                sgf_text = f.read()

            if "GM[1]" not in sgf_text and "gm[1]" not in sgf_text.lower():
                raise ValueError("仅支持围棋 SGF（GM[1]）。")

            size_raw = self._extract_sgf_prop(sgf_text, "SZ")
            if not size_raw:
                board_size = 19
            elif ":" in size_raw:
                left, right = size_raw.split(":", 1)
                if left != right:
                    raise ValueError("仅支持方形棋盘 SGF（如 SZ[9]）。")
                board_size = int(left)
            else:
                board_size = int(size_raw)
            if not (5 <= board_size <= 19):
                raise ValueError(f"棋盘大小超出界面支持范围: {board_size}")

            komi_raw = self._extract_sgf_prop(sgf_text, "KM")
            komi = float(komi_raw) if komi_raw else 5.5
            history_len = self._infer_history_len()

            move_tokens = re.findall(r";\s*([BW])\s*\[([^\]]*)\]", sgf_text, flags=re.IGNORECASE)
            moves: list[tuple[str, Optional[tuple[int, int]]]] = []
            for color_str, coord in move_tokens:
                color = color_str.upper()
                if color not in {"B", "W"}:
                    continue
                coord_clean = coord.strip().lower()
                moves.append((color, self._sgf_coord_to_move(coord_clean, board_size)))

            self.auto_playing = False
            self.auto_btn.setText("自动到终局")
            self.manual_game_end = False
            self.game_started = True
            self.board_size.setValue(board_size)
            self.komi.setValue(komi)
            self.state = GoState.new_game(board_size=board_size, komi=komi, history_len=history_len)
            self.history = [self.state]
            self.logs = [f"[系统] 已导入 SGF: {Path(path).name}"]

            for idx, (color, mv) in enumerate(moves, start=1):
                if self.state is None:
                    raise RuntimeError("内部状态异常：state 为空。")
                if self.state.is_terminal():
                    raise ValueError(f"第 {idx} 手后棋局已终局，后续着手无效。")
                expected = "B" if self.state.to_play == BLACK else "W"
                if color != expected:
                    raise ValueError(f"第 {idx} 手颜色不匹配: SGF={color}, 轮到={expected}")
                if not self.state.is_legal(mv):
                    raise ValueError(f"第 {idx} 手不合法: {mv}")
                self._apply_move(mv, trigger_auto=False)

            self._refresh_view()
            QMessageBox.information(self, "导入完成", f"已复现 {len(moves)} 手。")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))

    def end_game(self):
        if not self.game_started or self.state is None:
            return
        self._cancel_pending_inference()
        self.auto_playing = False
        self.auto_btn.setText("自动到终局")
        if not self.manual_game_end and not self.state.is_terminal():
            b, w = self.state.score()
            winner = self.state.winner()
            if winner == BLACK:
                result = "黑方胜"
            elif winner == WHITE:
                result = "白方胜"
            else:
                result = "和棋"
            self.logs.append(f"[系统] 对局已手动结束，结算：{result}（黑 {b:.1f} / 白 {w:.1f}）")
        self.manual_game_end = True
        self._refresh_view()

    def _game_over(self) -> bool:
        if (not self.game_started) or self.state is None:
            return True
        return self.manual_game_end or self.state.is_terminal()

    def undo_move(self):
        if (not self.game_started) or len(self.history) <= 1:
            return
        self._cancel_pending_inference()
        self.auto_playing = False
        self.manual_game_end = False
        self.auto_btn.setText("自动到终局")
        self.history.pop()
        if self.logs:
            self.logs.pop()
        self.state = self.history[-1]
        self._refresh_view()

    def pass_move(self):
        if self._game_over():
            return
        if not self._human_turn():
            return
        self._apply_move(None)

    def on_board_click(self, r: int, c: int):
        if self._game_over():
            return
        if not self._human_turn():
            return
        self._apply_move((r, c))

    def _human_turn(self) -> bool:
        if self.state is None:
            return False
        role = self._role_for_side(self.state.to_play)
        return role == ROLE_HUMAN

    def _role_for_side(self, to_play: int) -> str:
        if to_play == BLACK:
            return self._role_value(self.black_role)
        return self._role_value(self.white_role)

    def _apply_move(self, move: Optional[tuple[int, int]], trigger_auto: bool = True):
        if self.state is None:
            return
        if not self.state.is_legal(move):
            QMessageBox.warning(self, "非法落子", f"该落子不合法: {move}")
            return
        prev_state = self.state
        by = "黑方" if self.state.to_play == BLACK else "白方"
        label = "pass" if move is None else f"{move[0]},{move[1]}"
        self.state = self.state.play(move)
        if prev_state.to_play == BLACK:
            captured = self.state.black_captures - prev_state.black_captures
        else:
            captured = self.state.white_captures - prev_state.white_captures
        captured_n = max(0, int(captured))
        if captured_n > 0:
            self.logs.append(f"{by}: {label} | 提子 {captured_n} 子")
        else:
            self.logs.append(f"{by}: {label}")
        self.history.append(self.state)
        self._refresh_view()
        if trigger_auto:
            self._maybe_auto_step()

    def _set_ai_status(self, text: str, flush: bool = False):
        self.ai_status_label.setText(text)
        if flush:
            QApplication.processEvents()

    def _build_bundle(self, ckpt_path: str, device: str) -> ModelBundle:
        key = (ckpt_path, device)
        if key in self.model_cache:
            return self.model_cache[key]

        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt.get("cfg", {})
        model = AlphaZeroNet(
            board_size=int(cfg.get("board_size", self.board_size.value())),
            history_len=int(cfg.get("history_len", self._infer_history_len())),
            channels=int(cfg.get("channels", 192)),
            num_blocks=int(cfg.get("num_blocks", 12)),
            use_se=bool(cfg.get("use_se", True)),
            value_hidden_dim=int(cfg.get("value_hidden_dim", 256)),
        ).to(device)
        state_dict = ckpt.get("model", ckpt.get("best_model", ckpt))
        model.load_state_dict(state_dict)
        model.eval()
        bundle = ModelBundle(model=model, cfg=cfg)
        self.model_cache[key] = bundle
        return bundle

    def _build_mcts(self, role: str, bundle: ModelBundle, device: str, board_size: int, sims: int) -> MCTS:
        key = (role, device, board_size, sims)
        cached = self.mcts_cache.get(key)
        if cached is not None:
            return cached
        mcts = MCTS(
            model=bundle.model,
            board_size=board_size,
            num_simulations=sims,
            device=device,
        )
        self.mcts_cache[key] = mcts
        return mcts

    @staticmethod
    def _infer_move_job(mcts: MCTS, state: GoState) -> Optional[tuple[int, int]]:
        _, action = mcts.run(state, temperature=1e-6, add_exploration_noise=False)
        return decode_action(state.board_size, action)

    def _poll_inference_result(self, token: int):
        fut = self.pending_infer_future
        if fut is None:
            return
        if not fut.done():
            QTimer.singleShot(10, lambda: self._poll_inference_result(token))
            return

        snapshot = self.pending_infer_snapshot
        self.pending_infer_future = None
        self.pending_infer_snapshot = None
        if token != self.pending_infer_token:
            return
        if snapshot is None or self.state is None:
            return
        current_snapshot = (self.state.move_count, self.state.to_play, self.state.board.tobytes())
        if current_snapshot != snapshot:
            self._set_ai_status("")
            return

        try:
            move = fut.result()
        except Exception as e:
            QMessageBox.critical(self, "AI 推理失败", str(e))
            self.auto_playing = False
            self.auto_btn.setText("自动到终局")
            self._set_ai_status("")
            return

        self._set_ai_status("")
        self._apply_move(move)

    def ai_step(self):
        if self._game_over():
            return
        if self.pending_infer_future is not None:
            return
        role = self._role_for_side(self.state.to_play)
        if role == ROLE_HUMAN:
            return

        self._set_ai_status("AI 推理中...", flush=True)

        device = self.device.currentText()
        if device == "cuda" and not torch.cuda.is_available():
            QMessageBox.warning(self, "CUDA 不可用", "当前环境未检测到可用 CUDA，已改用 CPU。")
            self.device.setCurrentText("cpu")
            device = "cpu"

        try:
            bundle = self._build_bundle(role, device)
        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", str(e))
            self.auto_playing = False
            self.auto_btn.setText("自动到终局")
            self._set_ai_status("")
            return

        model_board = int(bundle.cfg.get("board_size", self.state.board_size))
        if model_board != self.state.board_size:
            QMessageBox.warning(
                self,
                "棋盘大小不匹配",
                f"模型棋盘={model_board}，当前棋盘={self.state.board_size}。",
            )
            self.auto_playing = False
            self.auto_btn.setText("自动到终局")
            self._set_ai_status("")
            return

        state_snapshot = (self.state.move_count, self.state.to_play, self.state.board.tobytes())
        mcts = self._build_mcts(
            role=role,
            bundle=bundle,
            device=device,
            board_size=self.state.board_size,
            sims=int(self.sims.value()),
        )
        self.pending_infer_token += 1
        token = self.pending_infer_token
        self.pending_infer_snapshot = state_snapshot
        self.pending_infer_future = self.infer_executor.submit(self._infer_move_job, mcts, self.state)
        QTimer.singleShot(10, lambda: self._poll_inference_result(token))

    def toggle_auto_play(self):
        self.auto_playing = not self.auto_playing
        self.auto_btn.setText("停止自动" if self.auto_playing else "自动到终局")
        self._maybe_auto_step()

    def _maybe_auto_step(self):
        auto_triggered = self.auto_playing or self._auto_mode_enabled()
        if not auto_triggered:
            return
        if self._game_over():
            self.auto_playing = False
            self.auto_btn.setText("自动到终局")
            return
        if self._human_turn():
            return
        QTimer.singleShot(10, self.ai_step)

    def _update_log_view(self):
        self._log_dirty = True
        if not self.log_timer.isActive():
            self.log_timer.start(80)

    def _flush_log_view(self):
        if not self._log_dirty:
            return
        self._log_dirty = False
        self.log_view.setPlainText("\n".join(self.logs[-200:]))
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _refresh_view(self):
        if (not self.game_started) or self.state is None:
            self.turn_label.setText("当前行棋: 未开始")
            self.score_label.setText("比分: -")
            self.terminal_label.setText("请点击“开始对局”")
            self._update_log_view()
            self.undo_btn.setEnabled(False)
            self.pass_btn.setEnabled(False)
            return
        if self.state is None:
            return
        self.board.set_state(self.state)
        side = "黑方" if self.state.to_play == BLACK else "白方"
        self.turn_label.setText(f"当前行棋: {side}")
        b, w = self.state.score()
        self.score_label.setText(f"比分: 黑方 {b:.1f} | 白方 {w:.1f}")
        if self.manual_game_end:
            winner = self.state.winner()
            if winner == BLACK:
                result = "黑方胜"
            elif winner == WHITE:
                result = "白方胜"
            else:
                result = "和棋"
            self.terminal_label.setText(f"终局: 手动结束（{result}）")
        elif self.state.is_terminal():
            winner = self.state.winner()
            if winner == BLACK:
                self.terminal_label.setText("终局: 黑方胜")
            elif winner == WHITE:
                self.terminal_label.setText("终局: 白方胜")
            else:
                self.terminal_label.setText("终局: 和棋")
        else:
            self.terminal_label.setText("")
        self._update_log_view()
        self.undo_btn.setEnabled(len(self.history) > 1)
        self.pass_btn.setEnabled((not self._game_over()) and self._human_turn())

    def closeEvent(self, event):
        self.auto_playing = False
        self._cancel_pending_inference()
        self.infer_executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = BattleWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
