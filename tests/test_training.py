import threading
import time

import chess
import numpy as np
import pytest

from chessml.config import TrainConfig, parse_duration, preset_values
from chessml.evaluation import MaterialPlayer, RandomPlayer, compute_ratings, play_match
from chessml.metrics import read_history
from chessml.runs import latest_checkpoint, read_json, run_checkpoints
from chessml.selfplay import game_status, material_balance
from chessml.trainer import Control, ReplayBuffer, Trainer


def tiny_config(**overrides):
    values = dict(preset="light", workers=1, threads=1, sims=8, batch_size=32, nice=0, parallel_games=2,
                  blocks=1, channels=16, max_plies=16, min_buffer=16, eval_minutes=0, seed=0)
    values.update(overrides)
    return TrainConfig(**values)


def test_parse_duration():
    assert parse_duration("2h") == 7200
    assert parse_duration("90m") == 5400
    assert parse_duration("1h30m") == 5400
    assert parse_duration("45s") == 45
    assert parse_duration("10") == 600
    with pytest.raises(ValueError):
        parse_duration("soon")


def test_presets_scale_with_cores():
    assert preset_values("hardcore", 12)["workers"] + preset_values("hardcore", 12)["threads"] == 12
    assert preset_values("light", 12)["workers"] < preset_values("balanced", 12)["workers"]
    assert preset_values("balanced", 12)["workers"] < preset_values("heavy", 12)["workers"]
    for name in ("light", "balanced", "heavy", "hardcore"):
        assert preset_values(name, 1)["workers"] >= 1
    cfg = TrainConfig.from_preset("heavy", sims=10, workers=None)
    assert cfg.sims == 10 and cfg.workers == preset_values("heavy")["workers"]


def test_adjudication_and_material():
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    assert material_balance(board) == 5
    history = {board._transposition_key(): 1}
    from collections import Counter

    assert game_status(board, Counter(history), max_plies=0, adjudicate_material=3) == ("1-0", "adjudicated")
    assert game_status(board, Counter(history), max_plies=500, adjudicate_material=3) is None


def test_replay_buffer_wraps_and_round_trips(tmp_path):
    buf = ReplayBuffer(5)
    planes = np.random.randint(0, 2, size=(7, 20, 8, 8)).astype(np.uint8)
    idx = [np.array([i, i + 1], dtype=np.int16) for i in range(7)]
    probs = [np.array([0.25, 0.75], dtype=np.float32) for _ in range(7)]
    buf.add(planes, idx, probs, np.arange(7, dtype=np.float32))
    assert buf.size == 5
    buf.save(tmp_path / "b.npz")
    other = ReplayBuffer(5)
    other.load(tmp_path / "b.npz")
    assert sorted(other.z.tolist()) == [2, 3, 4, 5, 6]
    x, target, z = other.sample(4, np.random.default_rng(0))
    assert x.shape == (4, 20, 8, 8) and np.allclose(target.sum(1), 1)


def test_baselines_and_ratings():
    import random

    rng = random.Random(0)
    res = play_match(MaterialPlayer(rng), RandomPlayer(rng), 6, max_plies=60, rng=rng)
    assert res["wins"] + res["draws"] + res["losses"] == 6
    ratings = compute_ratings([("a", "random", 9, 1, 0), ("b", "a", 5, 0, 5)])
    assert ratings["random"] == 0 and ratings["a"] > 200 and abs(ratings["a"] - ratings["b"]) < 50


def test_training_run_game_limit_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESSML_RUNS", str(tmp_path))
    events = []
    trainer = Trainer(tiny_config(max_games=4), Control(), events.append, run_name="t")
    assert trainer.run() == "completed"
    run = trainer.run_dir
    state = read_json(run / "state.json")
    assert state["status"] == "completed" and state["games"] == 4
    assert latest_checkpoint(run) is not None and len(run_checkpoints(run)) >= 2
    assert (run / "buffer.npz").exists()
    history = read_history(run)
    assert len(history["games"]) == 4 and history["train"] and history["system"]
    assert any(e["type"] == "status" for e in events)

    resumed = Trainer(tiny_config(max_games=2), Control(), None, resume=str(run))
    assert resumed.run() == "completed"
    assert resumed.run_dir == run
    assert read_json(run / "state.json")["games"] == 6
    assert len(read_history(run)["games"]) == 6


def test_graceful_stop_saves_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESSML_RUNS", str(tmp_path))
    control = Control()
    trainer = Trainer(tiny_config(max_plies=400), control, None, run_name="s")
    threading.Timer(6.0, control.stop.set).start()
    start = time.time()
    assert trainer.run() == "stopped"
    assert time.time() - start < 60
    state = read_json(trainer.run_dir / "state.json")
    assert state["status"] == "stopped"
    assert latest_checkpoint(trainer.run_dir).exists()
