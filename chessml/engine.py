"""A UCI engine (Stockfish) as a training opponent and teacher.

Engine games are mixed into self-play. On the engine's turn one multi-PV
search gives its best few moves with scores. They serve twice:
    teacher   a soft policy target over those moves (the network imitates
              Stockfish's best moves, never its deliberate mistakes)
    opponent  the move actually played, weakened by a strength level 0-20:
              level 0 moves at random, the share of random moves falls to
              zero at level 10, and the rest are drawn from the candidates by
              a softmax whose temperature falls to zero at level 20
The trainer adapts the level to the network's recent score (EngineLadder).
"""

from __future__ import annotations

import os
import shutil
from collections import deque

import chess
import chess.engine
import numpy as np

from .encoding import move_to_index

MAX_LEVEL = 20
MATE_CP = 10_000
TARGET_TEMPERATURE = 100.0  # centipawns, for the soft imitation target

INSTALL_HINT = ("Stockfish was not found. Install it (Ubuntu/Debian: sudo apt install stockfish; "
                "other systems: https://stockfishchess.org/download/) or point to the binary with "
                "--engine-path or the CHESSML_ENGINE environment variable.")


def find_engine(path: str | None = None) -> str:
    """Absolute path of the engine binary: `path`, $CHESSML_ENGINE, or stockfish on PATH."""
    candidates = [path] if path else [os.environ.get("CHESSML_ENGINE"), shutil.which("stockfish"),
                                      "/usr/games/stockfish"]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate) or (candidate if os.path.isfile(candidate) else None)
        if resolved and os.access(resolved, os.X_OK):
            return os.path.abspath(resolved)
    if path:
        raise FileNotFoundError(f"engine {path!r} not found or not executable. {INSTALL_HINT}")
    raise FileNotFoundError(INSTALL_HINT)


def engine_available(path: str | None = None) -> str | None:
    try:
        return find_engine(path)
    except FileNotFoundError:
        return None


def check_engine_config(cfg) -> None:
    """Validate the opponent settings of a TrainConfig and resolve its engine path.

    Raises ValueError for bad values, FileNotFoundError if the engine is missing.
    """
    if not 0 <= cfg.engine_share <= 1:
        raise ValueError("engine share must be between 0 and 1")
    if not 0 <= cfg.engine_level <= MAX_LEVEL:
        raise ValueError(f"engine level must be between 0 and {MAX_LEVEL}")
    if cfg.engine_time <= 0:
        raise ValueError("engine time per move must be positive")
    if cfg.engine_multipv < 1:
        raise ValueError("engine candidate moves must be at least 1")
    if cfg.engine_share > 0:
        cfg.engine_path = find_engine(cfg.engine_path)


def open_engine(path: str) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(path)
    options = {"Threads": 1, "Hash": 16}
    engine.configure({k: v for k, v in options.items() if k in engine.options})
    return engine


def analyse(engine: chess.engine.SimpleEngine, board: chess.Board, seconds: float,
            multipv: int) -> list[tuple[chess.Move, float]]:
    """The engine's best moves with centipawn scores for the side to move, best first."""
    infos = engine.analyse(board, chess.engine.Limit(time=seconds), multipv=multipv)
    scored = []
    for info in infos:
        if info.get("pv"):
            scored.append((info["pv"][0], float(info["score"].pov(board.turn).score(mate_score=MATE_CP))))
    scored.sort(key=lambda s: -s[1])
    return scored


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = (scores - scores.max()) / temperature
    p = np.exp(z)
    return p / p.sum()


def policy_target(scored: list[tuple[chess.Move, float]], turn: chess.Color) -> tuple[np.ndarray, np.ndarray]:
    """Sparse policy target (indices, probabilities) over the engine's candidate moves."""
    idx = np.array([move_to_index(m, turn) for m, _ in scored], dtype=np.int16)
    p = _softmax(np.array([s for _, s in scored]), TARGET_TEMPERATURE).astype(np.float32)
    return idx, p


def random_move_prob(level: int) -> float:
    return max(0.0, (10 - level) / 10)


def temperature(level: int) -> float:
    """Softmax temperature (centipawns) for picking among the candidates; 0 = always the best."""
    return 300.0 * (1 - level / MAX_LEVEL)


def choose_engine_move(scored: list[tuple[chess.Move, float]], board: chess.Board, level: int,
                       rng: np.random.Generator) -> chess.Move:
    """The move the engine plays at strength `level`."""
    if not scored or rng.random() < random_move_prob(level):
        moves = list(board.legal_moves)
        return moves[int(rng.integers(len(moves)))]
    t = temperature(level)
    if t <= 0 or len(scored) == 1:
        return scored[0][0]
    p = _softmax(np.array([s for _, s in scored]), t)
    return scored[int(rng.choice(len(scored), p=p))][0]


class EngineLadder:
    """Raises the engine level when the network scores well, lowers it when it scores badly."""

    def __init__(self, level: int, adaptive: bool = True, window: int = 20,
                 up: float = 0.6, down: float = 0.3):
        self.level = max(0, min(MAX_LEVEL, level))
        self.adaptive = adaptive
        self.window = window
        self.up, self.down = up, down
        self.scores: deque = deque(maxlen=window)
        self.decided_score: float | None = None  # window score behind the last level change

    def score(self) -> float | None:
        return sum(self.scores) / len(self.scores) if self.scores else None

    def record(self, net_score: float, level: int) -> int | None:
        """Add a finished game played at `level`; returns the new level if it changed."""
        if level != self.level:
            return None  # games started before the last change say nothing about this level
        self.scores.append(net_score)
        if not self.adaptive or len(self.scores) < self.window:
            return None
        score = self.score()
        new = self.level
        if score >= self.up:
            new = min(MAX_LEVEL, self.level + 1)
        elif score <= self.down:
            new = max(0, self.level - 1)
        if new == self.level:
            return None
        self.level = new
        self.decided_score = score
        self.scores.clear()
        return new
