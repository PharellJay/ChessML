"""Evaluation: baseline players, matches and Elo-style ratings.

Ratings are fitted by maximum likelihood over all evaluation games of a run
(logistic Elo model, draws count half), with one virtual draw per pairing so
a 100% score gives a finite rating. Without a calibration they are relative,
anchored at random play = 0. With one (elo.py) the Stockfish UCI_Elo
anchors are held at their nominal ratings, which puts everything on an
absolute Elo scale.
"""

from __future__ import annotations

import math
import random
import re
import signal
from collections import Counter
from pathlib import Path

import chess
import numpy as np

from .selfplay import PIECE_VALUES, game_status

ANCHOR = "random"
# Virtual draws per pairing in an absolute (calibrated) fit. The calibration
# ladder is connected by real results, so it needs almost no prior, and a full
# virtual draw would pull the ratings of lopsided pairings (0-20) together.
ABSOLUTE_PRIOR = 0.01


class RandomPlayer:
    name = "random"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def choose(self, boards: list[chess.Board]) -> list[chess.Move]:
        return [self.rng.choice(list(b.legal_moves)) for b in boards]


class MaterialPlayer:
    """Greedy one-ply baseline: mate if possible, else the biggest capture, else random."""

    name = "material"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def _score(self, board: chess.Board, move: chess.Move) -> float:
        gain = 0
        if board.is_en_passant(move):
            gain = 1
        else:
            victim = board.piece_type_at(move.to_square)
            if victim:
                gain = PIECE_VALUES[victim]
        if move.promotion:
            gain += PIECE_VALUES[move.promotion] - 1
        board.push(move)
        if board.is_checkmate():
            gain = 1000
        board.pop()
        return gain

    def choose(self, boards: list[chess.Board]) -> list[chess.Move]:
        out = []
        for board in boards:
            scored = [(self._score(board, m), m) for m in board.legal_moves]
            best = max(s for s, _ in scored)
            out.append(self.rng.choice([m for s, m in scored if s == best]))
        return out


class NetPlayer:
    """A checkpoint searching with MCTS (no noise, most-visited move)."""

    def __init__(self, checkpoint: str | Path, sims: int, name: str | None = None):
        from .mcts import MCTS
        from .network import Evaluator, model_from_checkpoint

        self.name = name or Path(checkpoint).name
        self.sims = sims
        self.mcts = MCTS(Evaluator(model_from_checkpoint(str(checkpoint))))

    def choose(self, boards: list[chess.Board]) -> list[chess.Move]:
        from .mcts import SearchTree

        trees = [SearchTree(b) for b in boards]
        self.mcts.run(trees, self.sims)
        return [self.mcts.choose_move(t) for t in trees]


def play_match(player, opponent, games: int, max_plies: int = 300, adjudicate_material: int = 3,
               opening_plies: int = 4, rng: random.Random | None = None, on_update=None) -> dict:
    """Play `games` games (colours alternate, all games in parallel).

    The first `opening_plies` plies are random so deterministic players do not
    repeat the same game. Returns wins/draws/losses from `player`'s view.
    `on_update(boards, player_white, results)` is called after every round of
    moves, for live views.
    """
    rng = rng or random.Random()
    boards = [chess.Board() for _ in range(games)]
    histories = [Counter([b._transposition_key()]) for b in boards]
    player_white = [i % 2 == 0 for i in range(games)]
    results: list[tuple[str, str] | None] = [None] * games

    for board, history in zip(boards, histories):
        for _ in range(opening_plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            history[board._transposition_key()] += 1

    def finished(i):
        if results[i] is None:
            results[i] = game_status(boards[i], histories[i], max_plies, adjudicate_material)
        return results[i] is not None

    while True:
        active = [i for i in range(games) if not finished(i)]
        if not active:
            break
        for who in (player, opponent):
            to_move = [i for i in active
                       if (boards[i].turn == chess.WHITE) == (player_white[i] == (who is player))]
            if not to_move:
                continue
            moves = who.choose([boards[i] for i in to_move])
            for i, move in zip(to_move, moves):
                boards[i].push(move)
                histories[i][boards[i]._transposition_key()] += 1
            active = [i for i in active if not finished(i)]
            if on_update:
                on_update(boards, player_white, results)

    wins = draws = losses = 0
    terminations = Counter()
    for (result, termination), white in zip(results, player_white):
        terminations[termination] += 1
        if result == "1/2-1/2":
            draws += 1
        elif (result == "1-0") == white:
            wins += 1
        else:
            losses += 1
    return {"wins": wins, "draws": draws, "losses": losses, "terminations": dict(terminations)}


def expected_score(diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


_K = math.log(10) / 400


def _pairs(matches, prior: float = 1.0) -> dict[tuple[str, str], list[float]]:
    pairs: dict[tuple[str, str], list[float]] = {}
    for a, b, w, d, l in matches:
        if a == b:
            continue
        key, score = ((a, b), w + d / 2) if a < b else ((b, a), l + d / 2)
        entry = pairs.setdefault(key, [prior / 2, prior])  # `prior` virtual draws
        entry[0] += score
        entry[1] += w + d + l
    return pairs


def compute_ratings(matches: list[tuple[str, str, int, int, int]], anchor: str = ANCHOR,
                    fixed: dict[str, float] | None = None, prior: float = 1.0) -> dict[str, float]:
    """Fit Elo ratings from (player, opponent, wins, draws, losses) records.

    `fixed` holds players whose ratings are known (they replace `anchor`).
    `prior` is the number of virtual draws added to every pairing.
    """
    fixed = fixed or {anchor: 0.0}
    pairs = _pairs(matches, prior)
    players = {p for key in pairs for p in key} | set(fixed)
    start = sum(fixed.values()) / len(fixed)
    ratings = {p: fixed.get(p, start) for p in players}
    k = _K
    for _ in range(200):
        for p in players:
            if p in fixed:
                continue
            residual = curvature = 0.0
            for (a, b), (score, n) in pairs.items():
                if p not in (a, b):
                    continue
                s = score if p == a else n - score
                e = expected_score(ratings[p] - ratings[b if p == a else a])
                residual += s - n * e
                curvature += n * e * (1 - e) * k
            if curvature > 0:
                ratings[p] += max(-400.0, min(400.0, residual / curvature))
    return ratings


def information(matches: list[tuple[str, str, int, int, int]], ratings: dict[str, float], fixed: dict[str, float],
                prior: float = 1.0, extra: list[tuple[str, str, float]] = ()) -> tuple[np.ndarray, dict[str, int]]:
    """Fisher information matrix of the free ratings, and each free player's row.

    A game between players with expected score e carries k^2 e (1 - e) of
    information, whatever its result. `extra` adds expected games
    (a, b, games) that have not been played, to plan which ones to play.
    """
    free = sorted(p for p in ratings if p not in fixed)
    pos = {p: i for i, p in enumerate(free)}
    info = np.zeros((len(free), len(free)))
    played = [(a, b, n) for (a, b), (_, n) in _pairs(matches, prior).items()]
    for a, b, n in played + list(extra):
        e = expected_score(ratings[a] - ratings[b])
        w = _K * _K * n * e * (1 - e)
        for p, q in ((a, b), (b, a)):
            if p in pos:
                info[pos[p], pos[p]] += w
                if q in pos:
                    info[pos[p], pos[q]] -= w
    return info, pos


def rating_errors(matches: list[tuple[str, str, int, int, int]], ratings: dict[str, float],
                  fixed: dict[str, float] | None = None, anchor: str = ANCHOR, prior: float = 1.0) -> dict[str, float]:
    """Standard error of each fitted rating, from the inverse Fisher information.

    It accounts for the uncertainty of every other free rating in the fit
    (for example the calibrated bridge players), not only the player's own games.
    """
    fixed = fixed or {anchor: 0.0}
    info, pos = information(matches, ratings, fixed, prior)
    if not pos:
        return {}
    cov = np.linalg.pinv(info)
    return {p: float(math.sqrt(max(cov[i, i], 0.0))) for p, i in pos.items()}


def own_error(matches: list[tuple[str, str, int, int, int]], ratings: dict[str, float], player: str) -> float | None:
    """Standard error of `player`'s rating if all its opponents' ratings were exact:
    the part of its uncertainty that more games of its own would reduce."""
    weight = 0.0
    for a, b, w, d, l in matches:
        if player in (a, b) and a != b:
            e = expected_score(ratings[a] - ratings[b])
            weight += _K * _K * (w + d + l) * e * (1 - e)
    return 1 / math.sqrt(weight) if weight > 0 else None


def checkpoint_ratings(evals: list[dict], calibration: dict | None = None) -> dict:
    """One rating fit over all evaluation records of a run.

    Returns the evaluated checkpoints ordered by self-play games (parsed from
    the ckpt_<games>g.pt name) with standard errors, and the material
    baseline's rating. With a `calibration` (elo.load_calibration) its games
    join the fit and the ratings are absolute Elo ("scale": "elo"),
    otherwise relative to random play ("scale": "relative").
    """
    matches = [(e["checkpoint"], e["opponent"], e["wins"], e["draws"], e["losses"]) for e in evals]
    fixed, prior = None, 1.0
    if calibration:
        matches += [tuple(r) for r in calibration["records"]]
        fixed, prior = calibration["anchors"], ABSOLUTE_PRIOR
    ratings = compute_ratings(matches, fixed=fixed, prior=prior)
    errors = rating_errors(matches, ratings, fixed=fixed, prior=prior)
    points = []
    for name, rating in ratings.items():
        m = re.match(r"ckpt_(\d+)g\.pt$", name)
        if m:
            points.append({"checkpoint": name, "games": int(m.group(1)), "rating": rating,
                           "error": errors.get(name)})
    points.sort(key=lambda p: p["games"])
    return {"checkpoints": points, "material": ratings.get("material"), "all": ratings, "errors": errors,
            "scale": "elo" if calibration else "relative"}


def evaluate_checkpoint(checkpoint: str, opponents: list[str], games: int, sims: int,
                        max_plies: int = 300, adjudicate_material: int = 3,
                        include_material_reference: bool = False, seed: int | None = None,
                        calibration: dict | None = None, calibrated_games: int | None = None,
                        bracket: bool = False, history: list[dict] | None = None,
                        hint: float | None = None) -> list[dict]:
    """Play `checkpoint` against each opponent ('random', 'material', a checkpoint
    path, or the name of a player in `calibration`).

    Returns one record per match. With `include_material_reference`, also
    plays the material baseline against random so it can be rated.
    Calibrated Stockfish opponents get `calibrated_games` games (default: `games`).
    With `bracket`, the checkpoint's Elo is then estimated from these games
    (with the run's earlier evaluation records `history`, and the last known
    Elo `hint` as a fallback, see elo.bracket_estimate), and it also plays the
    two calibrated Stockfish players around it.
    """
    rng = random.Random(seed)
    player = NetPlayer(checkpoint, sims)
    calibrated = {p["name"]: p for p in calibration["players"]} if calibration else {}
    records = []
    if include_material_reference:
        res = play_match(MaterialPlayer(rng), RandomPlayer(rng), games * 2, max_plies, adjudicate_material, rng=rng)
        records.append({"player": "material", "opponent": "random", **res})
    opponents = list(opponents)
    i = 0
    while i < len(opponents):
        opp = opponents[i]
        i += 1
        if opp == "random":
            opponent = RandomPlayer(rng)
        elif opp == "material":
            opponent = MaterialPlayer(rng)
        elif opp in calibrated:
            from .elo import make_player

            opponent = make_player(calibrated[opp], calibration, rng)
        else:
            opponent = NetPlayer(opp, sims)
        stockfish = opp in calibrated and calibrated[opp]["kind"] in ("level", "elo")
        n = calibrated_games if stockfish and calibrated_games else games
        try:
            res = play_match(player, opponent, n, max_plies, adjudicate_material, rng=rng)
        finally:
            if hasattr(opponent, "close"):
                opponent.close()
        records.append({"player": player.name, "opponent": opponent.name, **res})
        if bracket and calibration and i == len(opponents):
            from .elo import bracket_estimate, bracket_players

            bracket = False
            estimate = bracket_estimate(calibration, records, player.name, history, hint)
            opponents += [p for p in bracket_players(calibration, estimate) if p not in opponents]
    return records


def eval_process_main(checkpoint, opponents, games, sims, max_plies, adjudicate_material,
                      include_material_reference, nice, result_queue, tag, calibration=None, calibrated_games=None,
                      bracket=False, history=None, hint=None):
    """Entry point for the background evaluation process started by the trainer."""
    from .config import apply_niceness

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    apply_niceness(nice)
    import torch

    torch.set_num_threads(1)
    try:
        records = evaluate_checkpoint(checkpoint, opponents, games, sims, max_plies,
                                      adjudicate_material, include_material_reference, calibration=calibration,
                                      calibrated_games=calibrated_games, bracket=bracket, history=history,
                                      hint=hint)
        result_queue.put({"tag": tag, "records": records})
    except Exception as exc:  # report instead of dying silently
        result_queue.put({"tag": tag, "error": repr(exc)})


def score(record: dict) -> float:
    n = record["wins"] + record["draws"] + record["losses"]
    return (record["wins"] + record["draws"] / 2) / n if n else float("nan")
