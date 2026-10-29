"""Monte Carlo tree search with PUCT selection (AlphaZero style).

Many games are searched at once: every round selects one leaf in each tree
and evaluates all of those leaves with a single batched network call. The
search in each tree is identical to a sequential search, so no virtual loss
is needed, and batching keeps the network cost per position low on a CPU.

Values are always from the point of view of the side to move at a node.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Sequence

import chess
import numpy as np

from .encoding import encode_board, legal_moves_and_indices

EvalFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


class Node:
    __slots__ = ("moves", "indices", "priors", "N", "W", "children", "value", "terminal", "noised")

    def __init__(self):
        self.moves: list[chess.Move] | None = None
        self.indices: np.ndarray | None = None  # policy index of each move
        self.priors: np.ndarray | None = None
        self.N: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.children: list[Node | None] | None = None
        self.value = 0.0  # network value estimate when expanded
        self.terminal: float | None = None  # exact value if the game is over here
        self.noised = False

    @property
    def expanded(self) -> bool:
        return self.moves is not None

    def visits(self) -> int:
        return int(self.N.sum()) if self.N is not None else 0

    def q(self) -> float:
        """Mean value of this node for its side to move."""
        n = self.visits()
        return float(self.W.sum() / n) if n else self.value


def _key(board: chess.Board):
    return board._transposition_key()


class SearchTree:
    """A game in progress plus the search tree rooted at its current position."""

    def __init__(self, board: chess.Board | None = None):
        self.board = board.copy() if board is not None else chess.Board()
        self.root = Node()
        replay = self.board.copy()
        keys = [_key(replay)]
        while replay.move_stack:
            replay.pop()
            keys.append(_key(replay))
        self.history = Counter(keys)

    def is_repetition(self) -> bool:
        """True if the current position has occurred before in this game."""
        return self.history[_key(self.board)] > 1

    def advance(self, move: chess.Move) -> None:
        """Play `move`, keeping the matching subtree for reuse."""
        child = None
        if self.root.expanded and move in self.root.moves:
            child = self.root.children[self.root.moves.index(move)]
        self.board.push(move)
        self.history[_key(self.board)] += 1
        # A node the search scored as a draw by repetition is not the end of
        # the real game, so it cannot serve as a root: start a fresh one.
        self.root = child if child is not None and child.terminal is None else Node()

    def visit_distribution(self) -> tuple[list[chess.Move], np.ndarray]:
        root = self.root
        total = root.N.sum()
        probs = root.N / total if total > 0 else root.priors
        return root.moves, probs

    def top_moves(self, k: int = 5) -> list[dict]:
        """Best root moves by visit count, for display."""
        root = self.root
        if not root.expanded or not root.moves:
            return []
        total = max(1, int(root.N.sum()))
        order = np.lexsort((-root.priors, -root.N))[:k]
        out = []
        for i in order:
            move = root.moves[i]
            n = int(root.N[i])
            out.append({
                "uci": move.uci(),
                "san": self.board.san(move),
                "visits": n,
                "prob": n / total,
                "prior": float(root.priors[i]),
                "q": float(root.W[i] / n) if n else None,
            })
        return out


class MCTS:
    def __init__(
        self,
        evaluate: EvalFn,
        c_puct: float = 1.5,
        fpu_reduction: float = 0.2,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        rng: np.random.Generator | None = None,
    ):
        self.evaluate = evaluate
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = rng or np.random.default_rng()
        self.leaf_evals = 0  # positions sent to the network, for throughput stats

    # -- public API ---------------------------------------------------------

    def run(self, trees: Sequence[SearchTree], num_sims: int, add_noise: bool = False) -> None:
        """Search until every root has at least `num_sims` visits (reusing old ones)."""
        unexpanded = [t for t in trees if not t.root.expanded]
        if unexpanded:
            self._expand_roots(unexpanded, add_noise)
        if add_noise:
            for t in trees:
                if t.root.expanded and not t.root.noised:
                    self._add_noise(t.root)
        active = [t for t in trees if t.root.moves and t.root.terminal is None and t.root.visits() < num_sims]
        while active:
            pending = []
            for tree in active:
                leaf = self._select(tree)
                if leaf is not None:
                    pending.append(leaf)
            if pending:
                self._evaluate_and_backup(pending)
            active = [t for t in active if t.root.visits() < num_sims]

    def choose_move(self, tree: SearchTree, temperature: float = 0.0) -> chess.Move:
        """Pick a root move: proportional to visits^(1/T), or the most visited if T == 0."""
        root = tree.root
        if not root.moves:
            raise ValueError("no legal moves at the root")
        visits = root.N.astype(np.float64)
        if temperature > 0 and visits.sum() > 0:
            p = visits ** (1.0 / temperature)
            p /= p.sum()
            return root.moves[int(self.rng.choice(len(p), p=p))]
        best = np.flatnonzero(visits == visits.max())
        if len(best) > 1:  # tie-break on the prior
            best = [best[int(np.argmax(root.priors[best]))]]
        return root.moves[int(best[0])]

    # -- internals ----------------------------------------------------------

    def _add_noise(self, node: Node) -> None:
        if len(node.moves) > 1:
            noise = self.rng.dirichlet([self.dirichlet_alpha] * len(node.moves))
            node.priors = (1 - self.dirichlet_eps) * node.priors + self.dirichlet_eps * noise
        node.noised = True

    def _expand_roots(self, trees: Sequence[SearchTree], add_noise: bool) -> None:
        pending = []
        for tree in trees:
            moves, idx = legal_moves_and_indices(tree.board)
            if not moves:
                tree.root.moves = []
                tree.root.terminal = -1.0 if tree.board.is_check() else 0.0
                continue
            planes = encode_board(tree.board, repetition=tree.is_repetition())
            pending.append((tree, [], tree.root, moves, idx, planes))
        if pending:
            self._evaluate_and_backup(pending)
        if add_noise:
            for tree in trees:
                if tree.root.moves:
                    self._add_noise(tree.root)

    def _pick(self, node: Node) -> int:
        n = node.N
        sqrt_total = math.sqrt(n.sum() + 1)
        q = np.where(n > 0, node.W / np.maximum(n, 1), node.value - self.fpu_reduction)
        u = self.c_puct * node.priors * (sqrt_total / (1.0 + n))
        return int(np.argmax(q + u))

    def _select(self, tree: SearchTree):
        """Descend to a leaf. Terminal leaves are backed up immediately (returns None);
        otherwise returns the data needed to evaluate and expand the leaf."""
        board = tree.board
        node = tree.root
        path: list[tuple[Node, int]] = []
        path_keys = []
        while node.expanded and node.terminal is None:
            i = self._pick(node)
            path.append((node, i))
            board.push(node.moves[i])
            path_keys.append(_key(board))
            child = node.children[i]
            if child is None:
                child = node.children[i] = Node()
            node = child

        if node.terminal is None:
            moves, idx = legal_moves_and_indices(board)
            key = path_keys[-1]
            if not moves:
                node.terminal = -1.0 if board.is_check() else 0.0
            elif (
                tree.history[key] > 0
                or key in path_keys[:-1]
                or board.halfmove_clock >= 100
                or board.is_insufficient_material()
            ):
                node.terminal = 0.0  # repetition, 50-move rule or dead position
            else:
                planes = encode_board(board)
                for _ in path:
                    board.pop()
                return (tree, path, node, moves, idx, planes)

        for _ in path:
            board.pop()
        self._backup(path, node.terminal)
        return None

    def _evaluate_and_backup(self, pending) -> None:
        planes = np.stack([p[5] for p in pending])
        logits, values = self.evaluate(planes)
        self.leaf_evals += len(pending)
        for row, (tree, path, node, moves, idx, _) in enumerate(pending):
            legal_logits = logits[row, idx].astype(np.float64)
            legal_logits -= legal_logits.max()
            priors = np.exp(legal_logits)
            priors /= priors.sum()
            node.moves = moves
            node.indices = idx
            node.priors = priors
            node.N = np.zeros(len(moves), dtype=np.int64)
            node.W = np.zeros(len(moves), dtype=np.float64)
            node.children = [None] * len(moves)
            node.value = float(values[row])
            self._backup(path, node.value)

    @staticmethod
    def _backup(path, value: float) -> None:
        for node, i in reversed(path):
            value = -value
            node.N[i] += 1
            node.W[i] += value


def search(
    evaluate: EvalFn,
    boards: Sequence[chess.Board],
    num_sims: int,
    c_puct: float = 1.5,
) -> list[SearchTree]:
    """Convenience wrapper: search several positions without noise."""
    trees = [SearchTree(b) for b in boards]
    MCTS(evaluate, c_puct=c_puct).run(trees, num_sims)
    return trees
