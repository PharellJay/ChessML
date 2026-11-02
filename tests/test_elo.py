import math
import queue
import threading

from chessml.elo import (bracket_players, calibrate, calibration_pairs, calibration_summary, default_players, job_main,
                         load_calibration, load_results, measure, save_calibration)
from chessml.evaluation import checkpoint_ratings, compute_ratings, rating_errors
from chessml.network import ChessNet, NetConfig, save_checkpoint

PLAYERS = [
    {"name": "random", "kind": "random"},
    {"name": "material", "kind": "material"},
    {"name": "sf-level-10", "kind": "level", "level": 10},
    {"name": "sf-elo-1320", "kind": "elo", "elo": 1320},
]


def test_fixed_anchors_and_errors():
    few = [("a", "anchor", 5, 0, 5), ("b", "a", 9, 0, 1)]
    ratings = compute_ratings(few, fixed={"anchor": 1500.0})
    assert ratings["anchor"] == 1500 and abs(ratings["a"] - 1500) < 1 and ratings["b"] > ratings["a"] + 200
    errors = rating_errors(few, ratings, fixed={"anchor": 1500.0})
    assert "anchor" not in errors and errors["b"] > errors["a"]  # b's rating also carries a's uncertainty
    many = [("a", "anchor", 50, 0, 50), ("b", "a", 90, 0, 10)]
    assert rating_errors(many, compute_ratings(many, fixed={"anchor": 1500.0}), fixed={"anchor": 1500.0})["a"] \
        < errors["a"]
    # without fixed players the old behaviour is unchanged: random play anchored at 0
    assert compute_ratings([("x", "random", 5, 0, 5)])["random"] == 0


def test_calibration_pairs_link_the_ladder_to_the_anchors():
    pairs = calibration_pairs(default_players())
    names = {(a["name"], b["name"]) for a, b in pairs}
    assert ("random", "sf-level-3") in names and ("material", "sf-elo-1320") in names
    assert not any(a["kind"] == b["kind"] == "elo" for a, b in pairs)


def test_calibrate_and_measure(tmp_path, monkeypatch, fake_engine):
    monkeypatch.setenv("CHESSML_RUNS", str(tmp_path / "runs"))
    cal = calibrate(fake_engine, games=2, processes=2, players=PLAYERS, seed=0, refine_games=4)
    assert cal["ratings"]["sf-elo-1320"] == 1320 and set(cal["ratings"]) == {p["name"] for p in PLAYERS}
    assert len(cal["records"]) == len(calibration_pairs(PLAYERS)) + 2 and cal["refined_games"] == 4
    save_calibration(cal)
    cal = load_calibration()
    assert cal["engine"] == "FakeFish"

    ckpt = tmp_path / "ckpt_000000g.pt"
    save_checkpoint(str(ckpt), ChessNet(NetConfig(1, 16)))
    res = measure(ckpt, cal, games=2, sims=4, processes=1, seed=0)
    assert len(res["records"]) == len(PLAYERS) and math.isfinite(res["elo"]) and res["error"] > 0
    assert 0 < res["own_error"] <= res["error"]  # its own games are only part of the uncertainty
    assert "sf-level-10" in bracket_players(cal, cal["ratings"]["sf-level-10"])

    evals = [{"checkpoint": r["player"], "opponent": r["opponent"], "wins": r["wins"], "draws": r["draws"],
              "losses": r["losses"]} for r in res["records"]]
    fit = checkpoint_ratings(evals, cal)
    assert fit["scale"] == "elo" and abs(fit["checkpoints"][0]["rating"] - res["elo"]) < 1
    assert checkpoint_ratings(evals)["scale"] == "relative"


def _run_job(kind, params):
    events, stop = queue.Queue(), threading.Event()
    job_main(kind, params, events, stop)
    out = []
    while not events.empty():
        out.append(events.get())
    return out


def test_jobs_report_progress_and_store_results(tmp_path, monkeypatch, fake_engine):
    monkeypatch.setenv("CHESSML_RUNS", str(tmp_path / "runs"))
    base = {"engine_path": fake_engine, "processes": 1, "nice": 0}
    events = _run_job("calibrate", {**base, "games": 2, "players": PLAYERS})
    kinds = [e["event"] for e in events]
    assert kinds[0] == "started" and kinds[-1] == "completed"
    assert [e["stage"] for e in events if e["event"] == "stage"] == ["calibrating", "refining"]
    assert kinds.count("match") == 2 * len(calibration_pairs(PLAYERS))  # the refinement round is as large
    assert any(e["event"] == "boards" and e["games"] for e in events)  # live positions of the matches
    assert events[-1]["result"]["ladder"][-1]["elo"] >= events[-1]["result"]["ladder"][0]["elo"]
    assert calibration_summary(load_calibration())["matches"] == 2 * len(calibration_pairs(PLAYERS))

    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(str(ckpt), ChessNet(NetConfig(1, 16)))
    events = _run_job("measure", {**base, "checkpoint": str(ckpt), "games": 2, "sims": 4})
    stages = [e["stage"] for e in events if e["event"] == "stage"]
    assert stages == ["scouting", "measuring"] and events[-1]["event"] == "completed"
    results = load_results(load_calibration())
    assert len(results) == 1 and results[0]["checkpoint"] == "ckpt.pt"
    assert abs(results[0]["elo"] - events[-1]["result"]["elo"]) < 1e-6

    # a refinement round can be added to the saved calibration later
    before = load_calibration()
    events = _run_job("calibrate", {**base, "games": 4, "refine": True})
    assert events[-1]["event"] == "completed"
    assert len(load_calibration()["records"]) == len(before["records"]) + 2

    # stored results are refitted when the calibration changes
    current = load_results(load_calibration())[0]["elo"]
    cal = load_calibration()
    cal["anchors"] = {"sf-elo-1320": 1520.0}  # the same games, with the anchor 200 Elo higher
    assert abs(load_results(cal)[0]["elo"] - current - 200) < 5


def test_cancelled_and_failed_jobs(tmp_path, monkeypatch, fake_engine):
    monkeypatch.setenv("CHESSML_RUNS", str(tmp_path / "runs"))
    events, stop = queue.Queue(), threading.Event()
    stop.set()
    job_main("calibrate", {"engine_path": fake_engine, "processes": 1, "nice": 0, "games": 2, "players": PLAYERS},
             events, stop)
    out = [events.get() for _ in range(events.qsize())]
    assert out[-1]["event"] == "cancelled" and load_calibration() is None
    failed = _run_job("measure", {"checkpoint": "x.pt", "games": 2, "sims": 4, "processes": 1, "nice": 0})
    assert failed[-1]["event"] == "error" and "calibration" in failed[-1]["message"]


def test_gif_of_a_game(tmp_path, monkeypatch, fake_engine):
    from PIL import Image

    from chessml.gif import render_gif

    monkeypatch.setenv("CHESSML_ENGINE", fake_engine)
    monkeypatch.setenv("CHESSML_RUNS", str(tmp_path / "runs"))
    ckpt = tmp_path / "ckpt_000000g.pt"
    save_checkpoint(str(ckpt), ChessNet(NetConfig(1, 16)))
    for opponent in ("self", "random", "sf-level-5"):
        res = render_gif(ckpt, tmp_path / f"{opponent}.gif", opponent, sims=4, max_plies=6, size=160, seed=0)
        with Image.open(res["path"]) as gif:
            assert gif.n_frames == res["plies"] + 2  # start position, every ply, the result
        assert res["plies"] <= 6 and res["result"] in ("1-0", "0-1", "1/2-1/2")


def test_refinement_targets_the_bottom_of_the_ladder():
    from chessml.elo import _refit, refinement_pairs

    players = default_players()
    names = [p["name"] for p in players]
    ladder = [n for n in names if not n.startswith("sf-elo")]
    true = dict(zip(ladder, range(0, 2200, 200)), **{"sf-elo-1320": 1320, "sf-elo-1600": 1600, "sf-elo-1900": 1900})
    records = []
    for a, b in calibration_pairs(players):  # expected results of 20 games under the true ratings
        e = 1 / (1 + 10 ** ((true[b["name"]] - true[a["name"]]) / 400))
        wins = round(20 * e)
        records.append((a["name"], b["name"], wins, 0, 20 - wins))
    cal = _refit({"players": players, "anchors": {n: float(true[n]) for n in names if n.startswith("sf-elo")},
                  "records": records, "games_per_pair": 20})
    plan = refinement_pairs(cal, 200, 20)
    assert len(plan) == 10
    floor = min(cal["anchors"].values())
    # every planned match involves a player rated below the anchors
    assert all(min(cal["ratings"].get(a["name"], 9e9), cal["ratings"].get(b["name"], 9e9)) < floor for a, b in plan)


def test_training_levels_map_to_calibrated_elo():
    from chessml.elo import level_elo

    cal = {"level_nodes": 5000, "multipv": 5, "players": default_players(),
           "ratings": {"random": -200.0, "sf-level-3": 100.0, "sf-level-5": 400.0, "sf-level-15": 1800.0}}
    assert level_elo(cal, 0, 5000, 5) == -200 and level_elo(cal, 3, 5000, 5) == 100
    assert level_elo(cal, 4, 5000, 5) == 250  # halfway between levels 3 and 5
    assert level_elo(cal, 18, 5000, 5) is None  # above the ladder
    assert level_elo(cal, 4, 20000, 5) is None  # a different search: not the calibrated players


def test_bracket_players_surround_a_rating():
    cal = {"players": default_players(),
           "ratings": {"random": -200.0, "sf-level-3": 100.0, "sf-level-5": 400.0, "material": 600.0,
                       "sf-elo-1320": 1320.0}}
    assert bracket_players(cal, 450) == ["sf-level-5", "sf-elo-1320"]  # Stockfish players only
    assert bracket_players(cal, -500) == ["sf-level-3", "sf-level-5"]  # below the ladder: the two nearest
    assert bracket_players(cal, 2000) == ["sf-elo-1320", "sf-level-5"]


def test_evaluation_brackets_the_estimate(tmp_path, monkeypatch, fake_engine):
    from chessml.elo import engine_setup
    from chessml.evaluation import evaluate_checkpoint

    monkeypatch.setenv("CHESSML_RUNS", str(tmp_path / "runs"))
    cal = calibrate(fake_engine, games=2, processes=1, players=PLAYERS, seed=0, refine_games=0)
    setup = {**cal, **engine_setup(cal, fake_engine)}
    ckpt = tmp_path / "ckpt_000000g.pt"
    save_checkpoint(str(ckpt), ChessNet(NetConfig(1, 16)))
    records = evaluate_checkpoint(str(ckpt), ["random", "material"], 2, 4, max_plies=20, calibration=setup,
                                  calibrated_games=4, bracket=True, seed=0)
    stockfish = [r for r in records if r["opponent"].startswith("sf-")]
    assert [r["opponent"] for r in records[:2]] == ["random", "material"]
    assert len(stockfish) == 2  # the test ladder has two Stockfish players, one on each side or the two nearest
    assert all(r["wins"] + r["draws"] + r["losses"] == 4 for r in stockfish)
    assert all(r["wins"] + r["draws"] + r["losses"] == 2 for r in records[:2])  # the baselines keep `games`
