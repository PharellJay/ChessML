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
                  blocks=1, channels=16, max_plies=16, min_buffer=16, eval_minutes=0, seed=0,
                  full_search_prob=1.0, resign_threshold=-2)  # every move a sample, no resignation
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


def test_live_grid_and_focus(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESSML_RUNS", str(tmp_path))
    control, events = Control(), []
    control.focus[0], control.focus[1] = 0, 1  # watch the second game of worker 0
    trainer = Trainer(tiny_config(max_plies=400), control, events.append, run_name="g")
    threading.Timer(5.0, control.stop.set).start()
    assert trainer.run() == "stopped"
    grids = [e for e in events if e["type"] == "grid"]
    assert grids and len(grids[-1]["games"]) == 2  # 1 worker x 2 parallel games
    assert {(g["worker"], g["slot"]) for g in grids[-1]["games"]} == {(0, 0), (0, 1)}
    assert all("/" in g["board"] and g["ply"] >= 0 for g in grids[-1]["games"])
    lives = [e for e in events if e["type"] == "live"]
    assert lives and all(e["slot"] == 1 for e in lives)


def test_machines_start_untrained_or_from_a_checkpoint(tmp_path, monkeypatch):
    from chessml.runs import create_run, list_runs

    monkeypatch.setenv("CHESSML_RUNS", str(tmp_path))
    with pytest.raises(ValueError):
        create_run("../escape")
    fresh = create_run("fresh")
    with pytest.raises(FileExistsError):
        create_run("fresh")
    assert read_json(fresh / "state.json")["status"] == "new" and latest_checkpoint(fresh) is None

    trainer = Trainer(tiny_config(max_games=2), Control(), None, resume=str(fresh))
    assert trainer.run() == "completed" and trainer.run_dir == fresh
    source = latest_checkpoint(fresh)
    games = read_json(fresh / "state.json")["games"]

    fork = create_run("fork", source)
    assert latest_checkpoint(fork).name == f"ckpt_{games:06d}g.pt"
    assert {r["name"]: r["origin"] for r in list_runs()}["fork"] == str(source.resolve())
    trainer = Trainer(tiny_config(max_games=2), Control(), None, resume=str(fork))
    assert trainer.run() == "completed"
    assert read_json(fork / "state.json")["games"] == games + 2  # it continued from the copied checkpoint
    assert read_json(fresh / "state.json")["games"] == games  # and left the original untouched



def test_machine_with_its_own_size_and_pretraining_rules(tmp_path, monkeypatch):
    from chessml.runs import create_run

    monkeypatch.setenv("CHESSML_RUNS", str(tmp_path))
    empty = create_run("empty")
    with pytest.raises(FileNotFoundError):  # no saved replay buffer to pre-train on
        create_run("a", blocks=2, channels=16, bootstrap="empty")
    with pytest.raises(ValueError):
        create_run("b", blocks=2)  # size needs both numbers
    with pytest.raises(ValueError):
        create_run("c", blocks=2, channels=4096)
    sized = create_run("sized", blocks=2, channels=16)
    assert read_json(sized / "state.json")["net"] == {"blocks": 2, "channels": 16}
    trainer = Trainer(tiny_config(max_games=2), Control(), None, resume=str(sized))
    assert trainer.run() == "completed"
    from chessml.network import load_checkpoint
    assert load_checkpoint(str(latest_checkpoint(sized)))["net"] == {"blocks": 2, "channels": 16}
    assert read_json(sized / "state.json")["net"] == {"blocks": 2, "channels": 16}  # kept by later saves
    with pytest.raises(ValueError):  # a fork keeps its checkpoint's size
        create_run("d", latest_checkpoint(sized), blocks=4, channels=48)
    assert not (tmp_path / "a").exists() and not (tmp_path / "d").exists()
    assert empty.exists()


def test_pretraining_on_another_machines_buffer(tmp_path, monkeypatch):
    from chessml.network import load_checkpoint
    from chessml.runs import create_run, list_runs

    monkeypatch.setenv("CHESSML_RUNS", str(tmp_path))
    source = Trainer(tiny_config(max_games=6, max_plies=24), Control(), None, run_name="src")
    assert source.run() == "completed"
    positions = int(np.load(source.run_dir / "buffer.npz")["z"].shape[0])

    big = create_run("big", blocks=2, channels=24, bootstrap="src", bootstrap_epochs=2)
    info = read_json(big / "state.json")["bootstrap"]
    assert info == {"source": "src", "positions": positions, "epochs": 2, "steps_done": 0, "done": False}
    assert {r["name"]: r for r in list_runs()}["big"]["net"] == {"blocks": 2, "channels": 24}

    # stop after the first logged losses: the pre-training is saved part-way
    control, events = Control(), []

    def emit(event):
        events.append(event)
        if event["type"] == "train":
            control.stop.set()

    batch = 2  # small batches: enough steps per pass, and a held-out share on a tiny buffer
    trainer = Trainer(tiny_config(max_games=2, batch_size=batch), control, emit, resume=str(big))
    assert trainer.run() == "stopped"
    state = read_json(big / "state.json")
    assert state["status"] == "stopped" and state["games"] == 0
    assert 0 < state["bootstrap"]["steps_done"] < state["bootstrap"]["steps_total"]
    assert not state["bootstrap"]["done"]
    assert not any(e["type"] == "game" for e in events)  # no self-play before the pre-training ends

    events = []
    trainer = Trainer(tiny_config(max_games=2, batch_size=batch), Control(), events.append, resume=str(big))
    assert trainer.run() == "completed"
    state = read_json(big / "state.json")
    assert state["bootstrap"]["done"] and state["games"] == 2
    held = positions // 20
    assert state["bootstrap"]["steps_total"] == 2 * ((positions - held) // batch)
    assert 1 <= state["bootstrap"]["best_epoch"] <= 2
    logs = [e["message"] for e in events if e["type"] == "log"]
    assert any("continuing at step" in m for m in logs)
    assert any("held-out loss" in m for m in logs) and any("Pre-training finished" in m for m in logs)
    first = load_checkpoint(str(big / "checkpoints" / "ckpt_000000g.pt"))
    assert first["net"] == {"blocks": 2, "channels": 24} and first["steps"] >= state["bootstrap"]["steps_total"]
    assert not (big / "pretrain_best.pt").exists()
    # a later session does not pre-train again
    events = []
    assert Trainer(tiny_config(max_games=1, batch_size=batch), Control(), events.append, resume=str(big)).run() == "completed"
    assert not any("pre-training" in e.get("message", "").lower() for e in events if e["type"] == "log")

def _rated(*ratings, step=100):
    return [{"checkpoint": f"c{i * step}", "games": i * step, "rating": r} for i, r in enumerate(ratings)]


def test_learning_rate_drops_on_a_rating_plateau():
    from chessml.trainer import lr_after_eval

    points = _rated(0, 300, 250, 290, 280, 350)
    assert lr_after_eval(points, "c100", 3, window=1) == (0, False)  # a new best
    assert lr_after_eval(points, "c200", 3, window=1) == (1, False)
    assert lr_after_eval(points, "c300", 3, window=1) == (2, False)
    assert lr_after_eval(points, "c400", 3, window=1) == (3, True)  # three evaluations without beating 300
    assert lr_after_eval(points, "c500", 3, window=1) == (0, False)
    assert lr_after_eval(points, "c400", 0, window=1) == (3, False)  # patience 0: never drops
    assert lr_after_eval(points, "unknown", 3) == (0, False)


def test_learning_rate_ignores_a_single_lucky_rating():
    from chessml.trainer import lr_after_eval

    # a steady climb with one spike: compared one by one, nothing after the spike beats it
    points = _rated(100, 200, 300, 700, 350, 400, 450, 500, 550)
    assert lr_after_eval(points, "c800", 3, window=1) == (5, True)
    # averaged over three evaluations the spike is diluted: the climb overtakes it before patience runs out
    assert lr_after_eval(points, "c700", 3, window=3) == (2, False)
    assert lr_after_eval(points, "c800", 3, window=3) == (0, False)
    # a real plateau still drops: smoothed 300 345 330 330 300 300 300, nothing beats 345 for five evaluations
    flat = _rated(300, 390, 300, 300, 300, 300, 300)
    assert lr_after_eval(flat, "c600", 3, window=3) == (5, True)


def test_learning_rate_rule_restarts_after_a_change():
    from chessml.trainer import lr_after_eval

    points = _rated(900, 500, 520, 540, 530, 560)  # an old best of 900 that nothing reaches again
    assert lr_after_eval(points, "c500", 3, window=1)[1]  # compared with it, every later checkpoint is stale
    # compared only from the last change (game 100) on, the run is still improving
    assert lr_after_eval(points, "c500", 3, window=1, since=100) == (0, False)
    assert lr_after_eval(points, "c400", 3, window=1, since=100) == (1, False)
    assert lr_after_eval(points, "c100", 3, window=1, since=100) == (0, False)  # the first one is the best
    assert lr_after_eval(points, "c0", 3, since=100) == (0, False)  # from before the change: no verdict


def test_learning_rate_scale_survives_a_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESSML_RUNS", str(tmp_path))
    trainer = Trainer(tiny_config(max_games=2), Control(), None, run_name="lr")
    trainer._setup()
    trainer.lr_scale = 0.3
    trainer.games = 1
    trainer.ladder.level = 0
    trainer._save_checkpoint()
    trainer.db.close()
    resumed = Trainer(tiny_config(max_games=2, lr=1e-3), Control(), None, resume=str(trainer.run_dir))
    resumed._setup()
    assert abs(resumed.optimizer.param_groups[0]["lr"] - 3e-4) < 1e-12
    resumed.db.close()


def test_reset_lr_undoes_earlier_drops(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESSML_RUNS", str(tmp_path))
    trainer = Trainer(tiny_config(max_games=2), Control(), None, run_name="reset")
    trainer._setup()
    trainer.lr_scale, trainer.lr_since = 0.09, 0
    trainer.games = 7
    trainer.ladder.level = 0
    trainer._save_checkpoint()
    trainer.db.close()
    # without a reset the drops stay, but never below the floor: 1e-3 * 0.09 = 9e-5 runs at 1e-4
    kept =Trainer(tiny_config(lr=1e-3), Control(), None, resume=str(trainer.run_dir))
    kept._setup()
    assert abs(kept.optimizer.param_groups[0]["lr"] - 1e-4) < 1e-12
    assert kept.lr_note is None
    kept.db.close()
    reset = Trainer(tiny_config(lr=1e-3, reset_lr=True), Control(), None, resume=str(trainer.run_dir))
    reset._setup()
    assert abs(reset.optimizer.param_groups[0]["lr"] - 1e-3) < 1e-12
    assert (reset.lr_scale, reset.lr_since) == (1.0, 7)  # the plateau rule compares from game 7 on
    assert "reset" in reset.lr_note
    reset.db.close()


def test_fast_moves_are_not_training_samples(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESSML_RUNS", str(tmp_path))
    trainer = Trainer(tiny_config(max_games=4, full_search_prob=0.3, fast_sims=2, resign_threshold=-2),
                      Control(), None, run_name="cap")
    assert trainer.run() == "completed"
    state = read_json(trainer.run_dir / "state.json")
    plies = sum(g["plies"] for g in read_history(trainer.run_dir)["games"])
    assert 0 < state["positions"] < 0.6 * plies  # about 30% of the moves were recorded


def test_resignation_ends_lost_games_and_checks_itself(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESSML_RUNS", str(tmp_path))
    # a threshold above every possible value: every side resigns at its first full search
    trainer = Trainer(tiny_config(max_games=8, full_search_prob=1.0, resign_threshold=1.5, resign_min_ply=0,
                                  resign_moves=1, resign_check_share=0.5), Control(), None, run_name="resign")
    assert trainer.run() == "completed"
    games = read_history(trainer.run_dir)["games"]
    resigned = [g for g in games if g["termination"] == "resignation"]
    played_out = [g for g in games if g["termination"] != "resignation"]
    assert resigned and all(g["plies"] == 0 and g["result"] == "0-1" for g in resigned)  # White resigned at once
    # in every game played out, White "would have resigned"; each one is checked against the real result
    assert len(trainer.resign_checks) == len(played_out)


def test_resign_threshold_from_played_out_games():
    from chessml.trainer import choose_resign_threshold

    # sides whose value fell to -0.95 or lower always lost; around -0.85 a third of them did not
    samples = [(-0.97, True)] * 15 + [(-0.95, True)] * 10 + [(-0.85, False)] * 10 + [(-0.85, True)] * 20
    chosen = choose_resign_threshold(samples, ceiling=-0.8)
    assert -0.95 < chosen < -0.85  # resign at -0.95, not at -0.85
    assert choose_resign_threshold(samples, ceiling=-0.96) is None  # too few sides below -0.96 to tell
    assert choose_resign_threshold([(-0.95, False)] * 30, ceiling=-0.9) == -1.0  # never safe: switch it off
