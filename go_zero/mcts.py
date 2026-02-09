from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch

from .go import GoState, decode_action


@dataclass
class TreeNode:
    prior: float
    to_play: int
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "TreeNode"] = field(default_factory=dict)

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        board_size: int = 9,
        num_simulations: int = 128,
        c_base: float = 19652.0,
        c_init: float = 1.25,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        inference_batch_size: int = 1,
        topk_expand: int = 0,
        virtual_loss: float = 1.0,
        root_parallelism: int = 1,
        device: str = "cpu",
    ):
        self.model = model
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.c_base = c_base
        self.c_init = c_init
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.inference_batch_size = max(1, inference_batch_size)
        self.topk_expand = max(0, topk_expand)
        self.virtual_loss = float(max(0.0, virtual_loss))
        self.root_parallelism = max(1, int(root_parallelism))
        self.device = device

    @torch.inference_mode()
    def _predict_batch(self, states: list[GoState]) -> tuple[list[np.ndarray], list[float], list[list[int]]]:
        tensors = np.stack([s.to_tensor() for s in states], axis=0)
        x = torch.from_numpy(tensors).to(self.device)
        with torch.autocast(device_type="cuda", enabled=self.device.startswith("cuda")):
            logits, value = self.model(x)
        logits_np = logits.detach().cpu().numpy().astype(np.float64)
        values_np = value.detach().cpu().numpy().astype(np.float64)

        all_probs: list[np.ndarray] = []
        all_values: list[float] = []
        all_legal: list[list[int]] = []

        for i, state in enumerate(states):
            legal = state.legal_actions()
            logit = logits_np[i]
            mask = np.full_like(logit, -1e9)
            mask[legal] = 0.0
            masked = logit + mask
            masked -= np.max(masked)
            probs = np.exp(masked)
            probs_sum = probs.sum()
            if probs_sum <= 0:
                probs = np.zeros_like(probs)
                probs[legal] = 1.0 / len(legal)
            else:
                probs /= probs_sum

            all_probs.append(probs)
            all_values.append(float(values_np[i]))
            all_legal.append(legal)

        return all_probs, all_values, all_legal

    def _predict(self, state: GoState) -> tuple[np.ndarray, float, list[int]]:
        probs, values, legal = self._predict_batch([state])
        return probs[0], values[0], legal[0]

    def _select_child(self, node: TreeNode) -> tuple[int, TreeNode]:
        best_score = -1e18
        best_action = -1
        best_child = None

        parent_visits = node.visit_count
        for action, child in node.children.items():
            pb_c = math.log((parent_visits + self.c_base + 1.0) / self.c_base) + self.c_init
            pb_c *= math.sqrt(parent_visits + 1.0) / (child.visit_count + 1.0)
            prior_score = pb_c * child.prior
            value_score = -child.value()
            score = prior_score + value_score
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_child is None:
            raise RuntimeError("No child selected in MCTS")
        return best_action, best_child

    def _expand(self, node: TreeNode, state: GoState, add_noise: bool = False) -> float:
        priors, value, legal = self._predict(state)

        if add_noise and legal:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
            for i, a in enumerate(legal):
                priors[a] = (1 - self.dirichlet_eps) * priors[a] + self.dirichlet_eps * noise[i]

        expand_legal = legal
        if self.topk_expand > 0 and len(legal) > self.topk_expand:
            pass_action = self.board_size * self.board_size
            ranked = sorted(legal, key=lambda a: float(priors[a]), reverse=True)
            expand_legal = ranked[: self.topk_expand]
            if pass_action in legal and pass_action not in expand_legal:
                expand_legal = expand_legal[:-1] + [pass_action]

        for a in expand_legal:
            node.children[a] = TreeNode(prior=float(priors[a]), to_play=-node.to_play)

        return value

    def _expand_batch(
        self,
        leaves: list[tuple[TreeNode, GoState, bool]],
    ) -> list[float]:
        states = [s for _, s, _ in leaves]
        priors_list, values, legal_list = self._predict_batch(states)

        out_values: list[float] = []
        for (node, _, add_noise), priors, legal, value in zip(leaves, priors_list, legal_list, values):
            if add_noise and legal:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
                for i, a in enumerate(legal):
                    priors[a] = (1 - self.dirichlet_eps) * priors[a] + self.dirichlet_eps * noise[i]
            expand_legal = legal
            if self.topk_expand > 0 and len(legal) > self.topk_expand:
                pass_action = self.board_size * self.board_size
                ranked = sorted(legal, key=lambda a: float(priors[a]), reverse=True)
                expand_legal = ranked[: self.topk_expand]
                if pass_action in legal and pass_action not in expand_legal:
                    expand_legal = expand_legal[:-1] + [pass_action]
            for a in expand_legal:
                if a not in node.children:
                    node.children[a] = TreeNode(prior=float(priors[a]), to_play=-node.to_play)
            out_values.append(value)
        return out_values

    def _run_single_root_search(
        self,
        root_state: GoState,
        add_exploration_noise: bool,
        simulations: int,
    ) -> tuple[np.ndarray, float]:
        root = TreeNode(prior=1.0, to_play=root_state.to_play)
        self._expand(root, root_state, add_noise=add_exploration_noise)

        sims_done = 0
        while sims_done < simulations:
            batch_n = min(self.inference_batch_size, simulations - sims_done)
            batch_paths: list[list[TreeNode]] = []
            batch_terminal_values: list[float | None] = []
            batch_leaves: list[tuple[TreeNode, GoState, bool]] = []

            for _ in range(batch_n):
                node = root
                state = root_state
                search_path = [node]
                while node.expanded() and not state.is_terminal():
                    action, node = self._select_child(node)
                    state = state.play(decode_action(self.board_size, action))
                    search_path.append(node)

                batch_paths.append(search_path)
                if state.is_terminal():
                    batch_terminal_values.append(state.outcome_for_current_player())
                else:
                    batch_terminal_values.append(None)
                    batch_leaves.append((node, state, False))
                if self.virtual_loss > 0.0:
                    for pnode in search_path:
                        pnode.visit_count += 1
                        pnode.value_sum -= self.virtual_loss

            leaf_values: list[float] = []
            if batch_leaves:
                leaf_values = self._expand_batch(batch_leaves)

            leaf_ptr = 0
            for i, path in enumerate(batch_paths):
                value = batch_terminal_values[i]
                if value is None:
                    value = leaf_values[leaf_ptr]
                    leaf_ptr += 1
                if self.virtual_loss > 0.0:
                    for pnode in path:
                        pnode.visit_count -= 1
                        pnode.value_sum += self.virtual_loss
                for path_node in reversed(path):
                    path_node.visit_count += 1
                    path_node.value_sum += value
                    value = -value

            sims_done += batch_n

        visits = np.zeros(self.board_size * self.board_size + 1, dtype=np.float64)
        for a, child in root.children.items():
            visits[a] = child.visit_count
        return visits, float(root.value())

    def run(
        self,
        root_state: GoState,
        temperature: float = 1.0,
        add_exploration_noise: bool = True,
        return_root_value: bool = False,
    ) -> tuple[np.ndarray, int] | tuple[np.ndarray, int, float]:
        if root_state.is_terminal():
            pi = np.zeros(self.board_size * self.board_size + 1, dtype=np.float32)
            pi[-1] = 1.0
            if return_root_value:
                return pi, int(pi.argmax()), float(root_state.outcome_for_current_player())
            return pi, int(pi.argmax())
        roots = min(self.root_parallelism, self.num_simulations)
        if roots <= 1:
            visits, root_value = self._run_single_root_search(
                root_state,
                add_exploration_noise=add_exploration_noise,
                simulations=self.num_simulations,
            )
        else:
            sims_parts = [self.num_simulations // roots] * roots
            for i in range(self.num_simulations % roots):
                sims_parts[i] += 1
            visits = np.zeros(self.board_size * self.board_size + 1, dtype=np.float64)
            root_value_sum = 0.0
            futures = []
            with ThreadPoolExecutor(max_workers=roots) as ex:
                for sims_i in sims_parts:
                    futures.append(
                        ex.submit(
                            self._run_single_root_search,
                            root_state,
                            add_exploration_noise,
                            sims_i,
                        )
                    )
                for sims_i, fut in zip(sims_parts, futures):
                    v_i, r_i = fut.result()
                    visits += v_i
                    root_value_sum += r_i * sims_i
            root_value = root_value_sum / float(max(1, self.num_simulations))

        if temperature <= 1e-2:
            pi = np.zeros_like(visits)
            pi[np.argmax(visits)] = 1.0
        else:
            visits = np.power(visits, 1.0 / temperature)
            s = visits.sum()
            if s <= 0:
                pi = np.ones_like(visits) / len(visits)
            else:
                pi = visits / s

        action = int(np.random.choice(len(pi), p=pi))
        if return_root_value:
            return pi.astype(np.float32), action, root_value
        return pi.astype(np.float32), action
