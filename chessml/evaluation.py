"""Evaluation: baseline players, matches and Elo-style ratings.

Ratings are anchored at random play = 0. They are fitted by maximum
likelihood over all evaluation games of a run (logistic Elo model, draws
count half), with one virtual draw per pairing so a 100% score gives a
finite rating.
"""

from __future__ import annotations

import math
import random
import signal
from collections import Counter
from pathlib import Path

import chess

from .selfplay import PIECE_VALUES, game_status

ANCHOR = "random"


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
               opening_plies: int = 4, rng: random.Random | None = None) -> dict:
    """Play `games` games (colours alternate, all games in parallel).

    The first `opening_plies` plies are random so deterministic players do not
    repeat the same game. Returns wins/draws/losses from `player`'s view.
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


def compute_ratings(matches: list[tuple[str, str, int, int, int]], anchor: str = ANCHOR) -> dict[str, float]:
    """Fit Elo ratings from (player, opponent, wins, draws, losses) records."""
    pairs: dict[tuple[str, str], list[float]] = {}
    for a, b, w, d, l in matches:
        if a == b:
            continue
        key, score = ((a, b), w + d / 2) if a < b else ((b, a), l + d / 2)
        entry = pairs.setdefault(key, [0.5, 1.0])  # one virtual draw
        entry[0] += score
        entry[1] += w + d + l
    players = {p for key in pairs for p in key} | {anchor}
    ratings = {p: 0.0 for p in players}
    k = math.log(10) / 400
    for _ in range(200):
        for p in players:
            if p == anchor:
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


def evaluate_checkpoint(checkpoint: str, opponents: list[str], games: int, sims: int,
                        max_plies: int = 300, adjudicate_material: int = 3,
                        include_material_reference: bool = False, seed: int | None = None) -> list[dict]:
    """Play `checkpoint` against each opponent ('random', 'material' or a checkpoint path).

    Returns one record per match. With `include_material_reference`, also
    plays the material baseline against random so it can be rated.
    """
    rng = random.Random(seed)
    player = NetPlayer(checkpoint, sims)
    records = []
    if include_material_reference:
        res = play_match(MaterialPlayer(rng), RandomPlayer(rng), games * 2, max_plies, adjudicate_material, rng=rng)
        records.append({"player": "material", "opponent": "random", **res})
    for opp in opponents:
        if opp == "random":
            opponent = RandomPlayer(rng)
        elif opp == "material":
            opponent = MaterialPlayer(rng)
        else:
            opponent = NetPlayer(opp, sims)
        res = play_match(player, opponent, games, max_plies, adjudicate_material, rng=rng)
        records.append({"player": player.name, "opponent": opponent.name, **res})
    return records


def eval_process_main(checkpoint, opponents, games, sims, max_plies, adjudicate_material,
                      include_material_reference, nice, result_queue, tag):
    """Entry point for the background evaluation process started by the trainer."""
    from .config import apply_niceness

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    apply_niceness(nice)
    import torch

    torch.set_num_threads(1)
    try:
        records = evaluate_checkpoint(checkpoint, opponents, games, sims, max_plies,
                                      adjudicate_material, include_material_reference)
        result_queue.put({"tag": tag, "records": records})
    except Exception as exc:  # report instead of dying silently
        result_queue.put({"tag": tag, "error": repr(exc)})


def score(record: dict) -> float:
    n = record["wins"] + record["draws"] + record["losses"]
    return (record["wins"] + record["draws"] / 2) / n if n else float("nan")
