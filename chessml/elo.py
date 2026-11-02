"""Absolute strength in Elo, calibrated against Stockfish's UCI_Elo mode.

Stockfish can limit its strength to a given Elo (UCI_LimitStrength /
UCI_Elo, 1320-3190, calibrated by the Stockfish team against CCRL 40/4).
A young network is far weaker than 1320, so a ladder of bridge players
connects it to those anchors: random moves, the greedy material player and
node-limited Stockfish levels (the strength scale of engine.py).

`calibrate` plays the ladder once, each player against its neighbours and
the strongest bridges against the anchors, and saves the games. A
measurement fits ratings over those games plus the new ones, with the
anchors held at their nominal Elo, so the network lands on the anchors'
scale. Engine moves are limited by nodes rather than time, so the players'
strength does not depend on CPU speed or load.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import random
import signal
import threading
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

from .engine import analyse, choose_engine_move, engine_available, find_engine, open_engine
from .evaluation import (ABSOLUTE_PRIOR, MaterialPlayer, NetPlayer, RandomPlayer, compute_ratings, own_error,
                         play_match, rating_errors)
from .runs import list_checkpoints, read_json, runs_dir, write_json

CALIBRATION_FILE = "elo_calibration.json"
RESULTS_FILE = "elo_results.json"
LEVEL_NODES = 5_000  # search per move of a bridge level
ELO_NODES = 50_000  # enough for a UCI_Elo player to reach the depth at which it picks its move
MULTIPV = 5
SCOUT_GAMES = 2  # games per calibrated player in a measurement's first, scouting stage
WINDOW = 350  # the full measurement plays players within this many Elo of the scouting estimate
MIN_OPPONENTS = 4
SPAN = 2  # each ladder player meets the next SPAN players
ANCHOR_SPAN = 7  # each anchor meets the strongest ANCHOR_SPAN ladder players


def default_players() -> list[dict]:
    """The calibration ladder from weakest to strongest, then the Elo anchors.

    Each step should be small enough that neighbours still score against
    each other; one more random move in ten costs roughly 100-200 Elo. The
    ladder is densest from about 700 to 1600, where the network spends most
    of its training. The order comes from a trial calibration.
    """
    level = lambda n: {"name": f"sf-level-{n}", "kind": "level", "level": n}  # noqa: E731
    players = [{"name": "random", "kind": "random"}, level(3), level(5), level(6),
               {"name": "material", "kind": "material"}, level(7), level(8), level(9), level(10), level(12), level(15)]
    players += [{"name": f"sf-elo-{elo}", "kind": "elo", "elo": elo} for elo in (1320, 1600, 1900)]
    return players


def calibration_pairs(players: list[dict]) -> list[tuple[dict, dict]]:
    ladder = [p for p in players if p["kind"] != "elo"]
    anchors = [p for p in players if p["kind"] == "elo"]
    pairs = [(a, b) for i, a in enumerate(ladder) for b in ladder[i + 1:i + 1 + SPAN]]
    pairs += [(a, b) for b in anchors for a in ladder[-ANCHOR_SPAN:]]
    return pairs


# -- players ------------------------------------------------------------------


class LevelPlayer:
    """Stockfish at one of engine.py's strength levels, searching a fixed number of nodes."""

    def __init__(self, spec: dict, setup: dict, rng: random.Random):
        self.name = spec["name"]
        self.level = spec["level"]
        self.limit = chess.engine.Limit(nodes=setup["level_nodes"])
        self.multipv = setup["multipv"]
        self.rng = np.random.default_rng(rng.randrange(2**32))
        self.engine = open_engine(setup["engine_path"])

    def choose(self, boards: list[chess.Board]) -> list[chess.Move]:
        return [choose_engine_move(analyse(self.engine, b, self.limit, self.multipv), b, self.level, self.rng)
                for b in boards]

    def close(self) -> None:
        self.engine.quit()


class EloPlayer:
    """Stockfish limited to a nominal Elo by its own UCI_LimitStrength mode."""

    def __init__(self, spec: dict, setup: dict):
        self.name = spec["name"]
        self.limit = chess.engine.Limit(nodes=setup["elo_nodes"])
        self.engine = open_engine(setup["engine_path"], UCI_LimitStrength=True, UCI_Elo=spec["elo"])

    def choose(self, boards: list[chess.Board]) -> list[chess.Move]:
        return [self.engine.play(b, self.limit).move for b in boards]

    def close(self) -> None:
        self.engine.quit()


def make_player(spec: dict, setup: dict | None, rng: random.Random):
    """Build a player from its spec; engine players read `setup` (a calibration)."""
    kind = spec["kind"]
    if kind == "random":
        return RandomPlayer(rng)
    if kind == "material":
        return MaterialPlayer(rng)
    if kind == "net":
        return NetPlayer(spec["path"], spec["sims"], spec["name"])
    if kind == "level":
        return LevelPlayer(spec, setup, rng)
    if kind == "elo":
        return EloPlayer(spec, setup)
    raise ValueError(f"unknown player kind {kind!r}")


# -- running matches ------------------------------------------------------------


class Cancelled(Exception):
    """Raised when a job's stop event is set."""


# Live board snapshots for the web page: match processes put them on this queue
# (set by _init_worker, or by job_main when matches run in its own process).
_live_queue = None
LIVE_INTERVAL = 1.0  # seconds between snapshots of one match


def _live_sender(index: int, name_a: str, name_b: str, games: int):
    """An on_update callback for play_match that sends this match's boards at most once a second."""
    last = 0.0

    def send(boards, player_white, results) -> None:
        nonlocal last
        now = time.time()
        if _live_queue is None or now - last < LIVE_INTERVAL:
            return
        last = now
        items = []
        for i, (board, a_white) in enumerate(zip(boards, player_white)):
            items.append({"key": f"{index}-{i}", "board": board.board_fen(), "ply": board.ply(),
                          "last_move": board.peek().uci() if board.move_stack else None,
                          "white": name_a if a_white else name_b, "black": name_b if a_white else name_a,
                          "game": i + 1, "games": games, "result": results[i][0] if results[i] else None})
        try:
            _live_queue.put_nowait({"task": index, "games": items})
        except queue.Full:
            pass

    return send


def _init_worker(nice: int, live=None) -> None:
    from .config import apply_niceness

    global _live_queue
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent handles Ctrl+C
    apply_niceness(nice)
    _live_queue = live


def _play_pair(task) -> dict:
    index, spec_a, spec_b, games, seed, setup = task
    if "net" in (spec_a["kind"], spec_b["kind"]):
        import torch

        torch.set_num_threads(1)
    rng = random.Random(seed)
    players = []
    try:
        for spec in (spec_a, spec_b):
            players.append(make_player(spec, setup, rng))
        res = play_match(players[0], players[1], games, rng=rng,
                         on_update=_live_sender(index, spec_a["name"], spec_b["name"], games))
    finally:
        for p in players:
            if hasattr(p, "close"):
                p.close()
        if _live_queue is not None:
            try:
                _live_queue.put_nowait({"task": index, "done": True})
            except queue.Full:
                pass
    return {"player": spec_a["name"], "opponent": spec_b["name"], **res}


def run_matches(pairs: list[tuple[dict, dict]], games: int, setup: dict, processes: int = 1, nice: int = 10,
                seed: int | None = None, progress=None, stop=None, live=None) -> list[dict]:
    """Play every pair `games` times, spread over `processes` processes; returns one record per pair.

    Setting the `stop` event raises Cancelled; with several processes the
    running matches are terminated rather than finished. With a `live`
    queue, the matches send snapshots of their boards to it.
    """
    global _live_queue
    base = seed if seed is not None else random.randrange(2**31)
    tasks = [(i, a, b, games, base + i, setup) for i, (a, b) in enumerate(pairs)]
    records = []
    if processes <= 1:
        _live_queue = live
        for task in tasks:
            if stop is not None and stop.is_set():
                raise Cancelled
            records.append(_play_pair(task))
            if progress:
                progress(records[-1], len(records), len(tasks))
        return records
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(processes, len(tasks)), initializer=_init_worker, initargs=(nice, live)) as pool:
        results = pool.imap_unordered(_play_pair, tasks)
        while len(records) < len(tasks):  # leaving the block early terminates the workers
            try:
                record = results.next(timeout=0.5)
            except mp.TimeoutError:
                if stop is not None and stop.is_set():
                    raise Cancelled
                continue
            records.append(record)
            if progress:
                progress(record, len(records), len(tasks))
    return records


def _rows(records: list[dict]) -> list[tuple]:
    return [(r["player"], r["opponent"], r["wins"], r["draws"], r["losses"]) for r in records]


# -- calibration ------------------------------------------------------------------


def calibration_path() -> Path:
    return runs_dir() / CALIBRATION_FILE


def load_calibration(path: str | Path | None = None) -> dict | None:
    """The saved calibration, its ratings refitted from its games (so they match the current fit)."""
    calibration = read_json(Path(path) if path else calibration_path())
    return _refit(calibration) if calibration else None


def engine_setup(calibration: dict, engine_path: str | None = None) -> dict:
    """The calibration's engine settings, with an engine path that exists on this machine."""
    path = find_engine(engine_path) if engine_path else (engine_available(calibration["engine_path"])
                                                        or find_engine(None))
    return {"engine_path": path, "level_nodes": calibration["level_nodes"],
            "elo_nodes": calibration["elo_nodes"], "multipv": calibration["multipv"]}


def _refit(calibration: dict) -> dict:
    rows = [tuple(r) for r in calibration["records"]]
    ratings = compute_ratings(rows, fixed=calibration["anchors"], prior=ABSOLUTE_PRIOR)
    return {**calibration, "records": rows, "ratings": ratings,
            "errors": rating_errors(rows, ratings, fixed=calibration["anchors"], prior=ABSOLUTE_PRIOR)}


def calibrate(engine_path: str | None = None, games: int = 20, processes: int = 1, nice: int = 10,
              players: list[dict] | None = None, seed: int | None = None, progress=None, stop=None,
              live=None, refine_games: int | None = None, on_stage=None) -> dict:
    """Play the calibration ladder and fit its ratings with the UCI_Elo anchors held fixed.

    A refinement round follows (see refine_calibration). It plays
    `refine_games` more games, by default as many as the first round.
    """
    on_stage = on_stage or (lambda stage, info: None)
    players = players or default_players()
    anchors = {p["name"]: float(p["elo"]) for p in players if p["kind"] == "elo"}
    if not anchors:
        raise ValueError("calibration needs at least one UCI_Elo anchor")
    path = find_engine(engine_path)
    engine = open_engine(path)
    name = engine.id.get("name", "unknown")
    engine.quit()
    setup = {"engine_path": path, "level_nodes": LEVEL_NODES, "elo_nodes": ELO_NODES, "multipv": MULTIPV}
    pairs = calibration_pairs(players)
    on_stage("calibrating", {"games": games, "matches": len(pairs)})
    start = time.time()
    records = run_matches(pairs, games, setup, processes, nice, seed, progress, stop, live)
    calibration = _refit({"engine": name, **setup, "created": time.time(), "seconds": time.time() - start,
                          "games_per_pair": games, "refined_games": 0, "players": players, "anchors": anchors,
                          "records": _rows(records)})
    if refine_games is None:
        refine_games = len(pairs) * games
    if refine_games > 0:
        calibration = refine_calibration(calibration, refine_games, processes, nice,
                                         None if seed is None else seed + 10_000, progress, stop, live, on_stage)
    return calibration


def refinement_pairs(calibration: dict, games: int, block: int) -> list[tuple[dict, dict]]:
    """The matches whose games narrow the bottom of the ladder most, `block` games each.

    Every ladder player below the lowest anchor is rated through a chain of
    matches, so its uncertainty grows with the distance to the anchors. A
    game's information depends only on the players' expected score, so the
    matches can be planned before they are played. The plan greedily adds,
    one block at a time, the pairing (played or not) that most reduces the
    mean variance of those players, using the current ratings.
    """
    from .evaluation import information

    rows = [tuple(r) for r in calibration["records"]]
    ratings, anchors = calibration["ratings"], calibration["anchors"]
    names = [p["name"] for p in calibration["players"]]
    floor = min(anchors.values())
    target = [n for n in names if n not in anchors and ratings[n] < floor] or [n for n in names if n not in anchors]
    candidates = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if not (a in anchors and b in anchors)]

    def cost(extra) -> float:
        info, pos = information(rows, ratings, anchors, ABSOLUTE_PRIOR, extra)
        cov = np.linalg.pinv(info)
        return float(np.mean([cov[pos[n], pos[n]] for n in target]))

    chosen: list[tuple[str, str, int]] = []
    for _ in range(games // block):
        best = min(candidates, key=lambda ab: cost(chosen + [(ab[0], ab[1], block)]))
        chosen.append((best[0], best[1], block))
    by_name = {p["name"]: p for p in calibration["players"]}
    return [(by_name[a], by_name[b]) for a, b, _ in chosen]


def refine_calibration(calibration: dict, games: int, processes: int = 1, nice: int = 10, seed: int | None = None,
                       progress=None, stop=None, live=None, on_stage=None, engine_path: str | None = None) -> dict:
    """Play about `games` more calibration games where they narrow the ladder most, and refit."""
    on_stage = on_stage or (lambda stage, info: None)
    block = calibration["games_per_pair"]
    calibration = _refit(calibration)
    pairs = refinement_pairs(calibration, games, block)
    if not pairs:
        return calibration
    setup = engine_setup(calibration, engine_path)
    on_stage("refining", {"games": block, "matches": len(pairs)})
    start = time.time()
    records = run_matches(pairs, block, setup, processes, nice, seed, progress, stop, live)
    return _refit({**calibration, "records": calibration["records"] + _rows(records),
                   "refined_games": calibration.get("refined_games", 0) + block * len(pairs),
                   "seconds": calibration["seconds"] + time.time() - start})


def save_calibration(calibration: dict, path: str | Path | None = None) -> Path:
    path = Path(path) if path else calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, calibration)
    return path


def calibration_summary(calibration: dict | None) -> dict | None:
    """What the web page shows about a calibration: its settings and the rated ladder."""
    if not calibration:
        return None
    kinds = {p["name"]: p["kind"] for p in calibration["players"]}
    ladder = [{"name": name, "kind": kinds.get(name), "elo": rating,
               "error": 1.96 * calibration["errors"][name] if name in calibration["errors"] else None,
               "anchor": name in calibration["anchors"]}
              for name, rating in sorted(calibration["ratings"].items(), key=lambda kv: kv[1])]
    games = sum(r[2] + r[3] + r[4] for r in calibration["records"])
    return {"engine": calibration["engine"], "created": calibration["created"], "seconds": calibration["seconds"],
            "games_per_pair": calibration["games_per_pair"], "matches": len(calibration["records"]),
            "games": games, "refined_games": calibration.get("refined_games", 0), "ladder": ladder}


# -- measuring a checkpoint ------------------------------------------------------------


def fit_with_calibration(calibration: dict, records: list[dict]) -> tuple[dict, dict]:
    """Ratings and standard errors over the calibration games plus `records`."""
    rows = [tuple(r) for r in calibration["records"]] + _rows(records)
    ratings = compute_ratings(rows, fixed=calibration["anchors"], prior=ABSOLUTE_PRIOR)
    return ratings, rating_errors(rows, ratings, fixed=calibration["anchors"], prior=ABSOLUTE_PRIOR)


def measure(checkpoint: str | Path, calibration: dict, games: int = 10, sims: int = 64, processes: int = 1,
            nice: int = 10, opponents: list[str] | None = None, engine_path: str | None = None,
            seed: int | None = None, progress=None, on_stage=None, stop=None, live=None) -> dict:
    """Play `checkpoint` against calibrated players and fit its Elo.

    Unless `opponents` are given, a scouting stage first plays a few games
    against every calibrated player. The measurement proper then plays only
    the players near the scouting estimate. Lopsided matches carry little
    information and slow the measurement down, so the scouting games are not
    part of the final fit. `on_stage(stage, info)` is called as each stage starts.
    """
    on_stage = on_stage or (lambda stage, info: None)
    name = Path(checkpoint).name
    net = {"name": name, "kind": "net", "path": str(checkpoint), "sims": sims}
    chosen = [p for p in calibration["players"] if opponents is None or p["name"] in opponents]
    if not chosen:
        raise ValueError("no calibrated opponents selected")
    needs_engine = any(p["kind"] in ("level", "elo") for p in chosen)
    setup = engine_setup(calibration, engine_path) if needs_engine else None
    estimate = None
    if opponents is None:
        on_stage("scouting", {"games": SCOUT_GAMES, "opponents": [p["name"] for p in chosen]})
        scout = run_matches([(net, p) for p in chosen], SCOUT_GAMES, setup, processes, nice, seed, progress, stop,
                            live)
        estimate = fit_with_calibration(calibration, scout)[0][name]
        ratings = calibration["ratings"]
        ranked = sorted(chosen, key=lambda p: abs(ratings[p["name"]] - estimate))
        chosen = [p for i, p in enumerate(ranked) if i < MIN_OPPONENTS or abs(ratings[p["name"]] - estimate) <= WINDOW]
    on_stage("measuring", {"games": games, "opponents": [p["name"] for p in chosen], "estimate": estimate})
    records = run_matches([(net, p) for p in chosen], games, setup, processes, nice,
                          None if seed is None else seed + 1000, progress, stop, live)
    ratings, errors = fit_with_calibration(calibration, records)
    return {"checkpoint": name, "elo": ratings[name], "error": errors.get(name), "records": records,
            "own_error": own_error(_rows(records), ratings, name), "estimate": estimate, "ratings": ratings}


def level_elo(calibration: dict | None, level: float, nodes: int, multipv: int) -> float | None:
    """Calibrated Elo of a Stockfish training level, interpolated between the ladder's levels.

    Level 0 always moves at random, so it is the random player. Only defined
    when the engine searches like the ladder (same nodes and candidates), and
    within the ladder's range of levels.
    """
    if not calibration or calibration["level_nodes"] != nodes or calibration["multipv"] != multipv:
        return None
    ratings = calibration["ratings"]
    points = sorted([(0.0, ratings["random"])] if "random" in ratings else [])
    points = sorted(points + [(float(p["level"]), ratings[p["name"]]) for p in calibration["players"]
                              if p["kind"] == "level" and p["name"] in ratings])
    for (l0, r0), (l1, r1) in zip(points, points[1:]):
        if l0 <= level <= l1:
            return r0 + (r1 - r0) * (level - l0) / (l1 - l0) if l1 > l0 else r0
    return None


def bracket_players(calibration: dict, rating: float) -> list[str]:
    """The calibrated Stockfish players just below and just above `rating` (the two nearest at either end).

    Games against a player on each side carry the most information about a
    rating in between, and keep the estimate honest in both directions.
    """
    ratings = calibration["ratings"]
    ladder = sorted((p["name"] for p in calibration["players"]
                     if p["kind"] in ("level", "elo") and p["name"] in ratings), key=ratings.get)
    below = [n for n in ladder if ratings[n] <= rating]
    above = [n for n in ladder if ratings[n] > rating]
    if below and above:
        return [below[-1], above[0]]
    return sorted(ladder, key=lambda n: abs(ratings[n] - rating))[:2]


# -- stored measurements ----------------------------------------------------------------


def results_path() -> Path:
    return runs_dir() / RESULTS_FILE


def checkpoint_label(path: str | Path) -> str:
    """The label the web page uses for a checkpoint (e.g. models/best.pt or run/ckpt.pt)."""
    path = Path(path).resolve()
    for item in list_checkpoints():
        if Path(item["path"]).resolve() == path:
            return item["label"]
    return str(path)


def save_result(result: dict, checkpoint: str | Path, games: int, sims: int) -> dict:
    """Append a measurement to the results file; returns the stored entry."""
    half = lambda se: 1.96 * se if se is not None else None  # noqa: E731
    entry = {"label": checkpoint_label(checkpoint), "path": str(Path(checkpoint).resolve()),
             "checkpoint": result["checkpoint"], "elo": result["elo"], "error": half(result["error"]),
             "own_error": half(result.get("own_error")),
             "estimate": result.get("estimate"), "games": games, "sims": sims, "created": time.time(),
             "records": [{k: r[k] for k in ("player", "opponent", "wins", "draws", "losses")}
                         for r in result["records"]]}
    path = results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, (read_json(path) or []) + [entry])
    return entry


def load_results(calibration: dict | None) -> list[dict]:
    """Stored measurements, newest first, refitted against the current calibration."""
    results = read_json(results_path()) or []
    if calibration:
        for entry in results:
            ratings, errors = fit_with_calibration(calibration, entry["records"])
            name = entry["checkpoint"]
            entry["elo"] = ratings[name]
            error, own = errors.get(name), own_error(_rows(entry["records"]), ratings, name)
            entry["error"] = 1.96 * error if error is not None else None
            entry["own_error"] = 1.96 * own if own is not None else None
    return sorted(results, key=lambda e: -e["created"])


def store_in_run(result: dict, checkpoint: str | Path, sims: int) -> bool:
    """Add the measurement games to the checkpoint's run, so its charts use them too."""
    from .metrics import MetricsDB
    from .network import load_checkpoint

    checkpoint = Path(checkpoint)
    if checkpoint.parent.name != "checkpoints":
        return False
    db = MetricsDB(checkpoint.parent.parent / "metrics.db")
    meta = load_checkpoint(str(checkpoint))
    for r in result["records"]:
        db.insert("evals", {"elapsed": meta.get("elapsed", 0.0), "games": meta.get("games", 0),
                            "step": meta.get("steps", 0), "checkpoint": r["player"], "opponent": r["opponent"],
                            "wins": r["wins"], "draws": r["draws"], "losses": r["losses"], "sims": sims})
    db.close()
    return True


# -- background jobs for the web interface ------------------------------------------------


def job_main(kind: str, params: dict, events, stop) -> None:
    """Run a calibration or a measurement in its own process, reporting through `events`.

    Events are {"type": "strength", "kind": kind, "event": ...} with event
    started, stage, match, completed, cancelled or error.
    """
    from .config import apply_niceness

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the server stops us through `stop`
    nice = params.get("nice", 10)
    apply_niceness(nice)

    def emit(event: str, **data) -> None:
        events.put({"type": "strength", "kind": kind, "event": event, "time": time.time(), **data})

    def progress(record: dict, done: int, total: int) -> None:
        emit("match", record={k: record[k] for k in ("player", "opponent", "wins", "draws", "losses")},
             done=done, total=total)

    emit("started", params={k: v for k, v in params.items() if k != "players"})
    live = mp.get_context("spawn").Queue(maxsize=2000)
    forwarding = threading.Event()
    forwarding.set()
    forwarder = threading.Thread(target=_forward_live, args=(live, emit, forwarding), daemon=True)
    forwarder.start()
    final: tuple[str, dict] = ("error", {"message": "job ended unexpectedly"})
    try:
        stage = lambda name, info: emit("stage", stage=name, **info)  # noqa: E731
        if kind == "calibrate" and params.get("refine"):
            current = load_calibration()
            if not current:
                raise ValueError("there is no calibration to refine yet")
            calibration = refine_calibration(current, params["games"], params["processes"], nice, progress=progress,
                                             stop=stop, live=live, on_stage=stage,
                                             engine_path=params.get("engine_path"))
            save_calibration(calibration)
            final = ("completed", {"result": calibration_summary(load_calibration())})
        elif kind == "calibrate":
            calibration = calibrate(params.get("engine_path"), params["games"], params["processes"], nice,
                                    players=params.get("players"), progress=progress, stop=stop, live=live,
                                    refine_games=params.get("refine_games"), on_stage=stage)
            save_calibration(calibration)
            final = ("completed", {"result": calibration_summary(load_calibration())})
        elif kind == "measure":
            calibration = load_calibration()
            if not calibration:
                raise ValueError("no Elo calibration yet")
            checkpoint = params["checkpoint"]
            result = measure(checkpoint, calibration, params["games"], params["sims"], params["processes"], nice,
                             engine_path=params.get("engine_path"), progress=progress, stop=stop, live=live,
                             on_stage=stage)
            store_in_run(result, checkpoint, params["sims"])
            final = ("completed", {"result": save_result(result, checkpoint, params["games"], params["sims"])})
        else:
            raise ValueError(f"unknown job kind {kind!r}")
    except Cancelled:
        final = ("cancelled", {})
    except Exception as exc:  # report instead of dying silently
        final = ("error", {"message": str(exc) or repr(exc)})
    finally:  # the live boards go first, so the job's last event is how it ended
        forwarding.clear()
        forwarder.join(timeout=2)
        emit("boards", games=[])
        emit(final[0], **final[1])


def _forward_live(live, emit, forwarding) -> None:
    """Merge the matches' board snapshots into one "boards" event per second."""
    active: dict[int, list] = {}
    last = 0.0
    while forwarding.is_set():
        try:
            msg = live.get(timeout=0.2)
            if msg.get("done"):
                active.pop(msg["task"], None)
            else:
                active[msg["task"]] = msg["games"]
        except queue.Empty:
            pass
        if time.time() - last >= LIVE_INTERVAL:
            last = time.time()
            emit("boards", games=[g for task in sorted(active) for g in active[task]])
