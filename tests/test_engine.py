import sqlite3
import sys

import chess
import numpy as np
import pytest

from chessml.config import TrainConfig
from chessml.encoding import move_to_index
from chessml.engine import (EngineLadder, analyse, check_engine_config, choose_engine_move, engine_available,
                            find_engine, open_engine, policy_target, random_move_prob)
from chessml.metrics import MetricsDB, read_history
from chessml.runs import read_json
from chessml.trainer import Control, Trainer
from test_training import tiny_config

# A minimal UCI engine: scores every legal move by the material it wins, so
# tests can run engine games without Stockfish installed.
FAKE_ENGINE = '''
import sys
import chess

VALUES = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900}
board, multipv = chess.Board(), 1
for line in sys.stdin:
    words = line.split()
    if not words:
        continue
    if words[0] == "uci":
        print("id name FakeFish")
        print("option name MultiPV type spin default 1 min 1 max 500")
        print("option name Threads type spin default 1 min 1 max 1024")
        print("option name Hash type spin default 16 min 1 max 1024")
        print("uciok")
    elif words[0] == "isready":
        print("readyok")
    elif words[:3] == ["setoption", "name", "MultiPV"]:
        multipv = int(words[4])
    elif words[0] == "position":
        board = chess.Board() if words[1] == "startpos" else chess.Board(" ".join(words[2:8]))
        if "moves" in words:
            for uci in words[words.index("moves") + 1:]:
                board.push_uci(uci)
    elif words[0] == "go":
        scored = []
        for move in board.legal_moves:
            victim = board.piece_type_at(move.to_square)
            scored.append((VALUES.get(victim, 0), move))
        scored.sort(key=lambda s: -s[0])
        for i, (cp, move) in enumerate(scored[:multipv]):
            print(f"info depth 1 multipv {i + 1} score cp {cp} nodes 1 pv {move.uci()}")
        print(f"bestmove {scored[0][1].uci()}")
    elif words[0] == "quit":
        break
    sys.stdout.flush()
'''


@pytest.fixture
def fake_engine(tmp_path):
    path = tmp_path / "fakefish"
    path.write_text(f"#!{sys.executable}\n{FAKE_ENGINE}")
    path.chmod(0o755)
    return str(path)


def test_policy_target_indexes_both_colours():
    e4, e5 = chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")
    idx, p = policy_target([(e4, 30.0), (chess.Move.from_uci("d2d4"), -70.0)], chess.WHITE)
    assert idx[0] == move_to_index(e4, chess.WHITE) and np.isclose(p.sum(), 1) and p[0] > p[1]
    idx, _ = policy_target([(e5, 0.0)], chess.BLACK)
    assert idx[0] == move_to_index(e5, chess.BLACK) == move_to_index(e4, chess.WHITE)  # mirrored


def test_levels_weaken_the_engine():
    board = chess.Board()
    scored = [(chess.Move.from_uci("e2e4"), 40.0), (chess.Move.from_uci("a2a3"), -60.0)]
    rng = np.random.default_rng(0)
    assert random_move_prob(0) == 1 and random_move_prob(5) == 0.5 and random_move_prob(10) == 0
    assert all(choose_engine_move(scored, board, 20, rng) == scored[0][0] for _ in range(50))
    weak = {choose_engine_move(scored, board, 0, rng) for _ in range(200)}
    assert len(weak) > 2  # random legal moves at level 0


def test_ladder_moves_with_results():
    ladder = EngineLadder(0, window=4)
    assert [ladder.record(1.0, 0) for _ in range(4)] == [None, None, None, 1]
    assert ladder.record(0.0, 0) is None  # a game started at the old level is ignored
    assert [ladder.record(0.0, 1) for _ in range(4)][-1] == 0
    assert [ladder.record(0.0, 0) for _ in range(4)][-1] is None  # clamped at 0
    fixed = EngineLadder(3, adaptive=False, window=2)
    assert fixed.record(1.0, 3) is None and fixed.record(1.0, 3) is None and fixed.score() == 1.0


def test_find_engine(tmp_path, fake_engine, monkeypatch):
    with pytest.raises(FileNotFoundError, match="apt install stockfish"):
        find_engine(str(tmp_path / "missing"))
    assert find_engine(fake_engine) == fake_engine
    monkeypatch.setenv("CHESSML_ENGINE", fake_engine)
    assert engine_available() == fake_engine
    with pytest.raises(ValueError):
        check_engine_config(TrainConfig(engine_share=1.5))
    with pytest.raises(ValueError):
        check_engine_config(TrainConfig(engine_level=21))


def test_analyse_with_uci_engine(fake_engine):
    engine = open_engine(fake_engine)
    try:
        board = chess.Board("4k3/8/q7/8/8/8/8/R3K3 w - - 0 1")
        scored = analyse(engine, board, 0.01, 3)
    finally:
        engine.quit()
    assert len(scored) == 3 and scored[0] == (chess.Move.from_uci("a1a6"), 900.0)


def test_old_metrics_db_gets_new_columns(tmp_path):
    path = tmp_path / "metrics.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE games (ts REAL, elapsed REAL, game INTEGER, result TEXT, plies INTEGER, "
                 "termination TEXT, worker INTEGER)")
    conn.execute("INSERT INTO games VALUES (0, 0, 1, '1-0', 10, 'checkmate', 0)")
    conn.commit()
    conn.close()
    db = MetricsDB(path)
    db.insert("games", {"game": 2, "result": "0-1", "opponent": "engine", "engine_level": 3, "net_score": 0.0})
    rows = db.fetch("games")
    db.close()
    assert rows[0]["opponent"] is None and rows[1]["engine_level"] == 3


def _engine_run(tmp_path, monkeypatch, engine_path):
    monkeypatch.setenv("CHESSML_RUNS", str(tmp_path / "runs"))
    cfg = tiny_config(max_games=4, engine_share=1.0, engine_path=engine_path, engine_time=0.005, engine_window=2)
    trainer = Trainer(cfg, Control(), None, run_name="e")
    assert trainer.run() == "completed"
    games = read_history(trainer.run_dir)["games"]
    assert len(games) == 4 and all(g["opponent"] == "engine" and g["net_score"] is not None for g in games)
    assert read_json(trainer.run_dir / "state.json")["engine_level"] is not None
    return trainer


def test_training_against_engine(tmp_path, monkeypatch, fake_engine):
    trainer = _engine_run(tmp_path, monkeypatch, fake_engine)
    assert trainer.buffer.size > 0


@pytest.mark.skipif(engine_available() is None, reason="Stockfish is not installed")
def test_training_against_stockfish(tmp_path, monkeypatch):
    _engine_run(tmp_path, monkeypatch, None)
