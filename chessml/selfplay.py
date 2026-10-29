"""Self-play worker processes and the game rules shared with evaluation."""

from __future__ import annotations

import queue
import signal
import time
from collections import Counter

import chess
import numpy as np

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material_balance(board: chess.Board) -> int:
    """White material minus Black material, in pawn units."""
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        total += value * (len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK)))
    return total


def game_status(board: chess.Board, history: Counter, max_plies: int, adjudicate_material: int):
    """(result, termination) if the game is over, else None.

    `history` counts how often each position occurred (transposition keys).
    Games reaching `max_plies` are adjudicated: a material lead of at least
    `adjudicate_material` pawns wins, anything less is a draw.
    """
    outcome = board.outcome(claim_draw=False)
    if outcome is not None:
        names = {
            chess.Termination.CHECKMATE: "checkmate",
            chess.Termination.STALEMATE: "stalemate",
            chess.Termination.INSUFFICIENT_MATERIAL: "insufficient material",
            chess.Termination.SEVENTYFIVE_MOVES: "fifty-move rule",
            chess.Termination.FIVEFOLD_REPETITION: "repetition",
        }
        return outcome.result(), names.get(outcome.termination, outcome.termination.name.lower())
    if board.halfmove_clock >= 100:
        return "1/2-1/2", "fifty-move rule"
    if history[board._transposition_key()] >= 3:
        return "1/2-1/2", "repetition"
    if board.ply() >= max_plies:
        balance = material_balance(board)
        if balance >= adjudicate_material:
            return "1-0", "adjudicated"
        if balance <= -adjudicate_material:
            return "0-1", "adjudicated"
        return "1/2-1/2", "move limit"
    return None


def result_value(result: str, turn: chess.Color) -> float:
    """Game result from the point of view of the side `turn`."""
    if result == "1-0":
        return 1.0 if turn == chess.WHITE else -1.0
    if result == "0-1":
        return -1.0 if turn == chess.WHITE else 1.0
    return 0.0


class _Game:
    def __init__(self):
        from .mcts import SearchTree

        self.tree = SearchTree()
        self.planes: list[np.ndarray] = []
        self.pol_idx: list[np.ndarray] = []
        self.pol_p: list[np.ndarray] = []
        self.turns: list[bool] = []
        self.moves: list[str] = []


def _ignore_sigint_and_nice(nice: int) -> None:
    # Ctrl+C is handled by the parent, which stops children gracefully.
    from .config import apply_niceness

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    apply_niceness(nice)


def worker_main(worker_id, cfg_dict, weights_path, weights_version, counters,
                game_queue, live_queue, pause_event, stop_event, seed):
    """Play self-play games until `stop_event` is set; unfinished games are discarded."""
    from .config import TrainConfig
    from .encoding import encode_board
    from .mcts import MCTS
    from .network import ChessNet, Evaluator, NetConfig, load_checkpoint

    import torch

    cfg = TrainConfig.from_dict(cfg_dict)
    _ignore_sigint_and_nice(cfg.nice)
    live_queue.cancel_join_thread()  # live snapshots are disposable; never block exit on them
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = ChessNet(NetConfig(cfg.blocks, cfg.channels))
    loaded_version = -1
    mcts = MCTS(Evaluator(model), c_puct=cfg.c_puct, dirichlet_alpha=cfg.dirichlet_alpha,
                dirichlet_eps=cfg.dirichlet_eps, rng=rng)
    games = [_Game() for _ in range(cfg.parallel_games)]
    last_live = 0.0

    while not stop_event.is_set():
        if pause_event.is_set():
            time.sleep(0.1)
            continue
        if weights_version.value != loaded_version:
            loaded_version = weights_version.value
            model.load_state_dict(load_checkpoint(weights_path)["model"])
            model.eval()

        evals_before = mcts.leaf_evals
        mcts.run([g.tree for g in games], cfg.sims, add_noise=True)
        counters[worker_id] += mcts.leaf_evals - evals_before

        if worker_id == 0 and time.time() - last_live > 0.25:
            last_live = time.time()
            _send_live(live_queue, games[0])

        for slot, game in enumerate(games):
            if stop_event.is_set():
                break
            tree = game.tree
            board = tree.board
            moves, probs = tree.visit_distribution()
            game.planes.append(encode_board(board, repetition=tree.is_repetition()))
            game.pol_idx.append(tree.root.indices.astype(np.int16))
            game.pol_p.append(probs.astype(np.float32))
            game.turns.append(board.turn)
            temperature = 1.0 if board.ply() < cfg.temperature_plies else 0.0
            move = mcts.choose_move(tree, temperature)
            game.moves.append(move.uci())
            tree.advance(move)
            status = game_status(tree.board, tree.history, cfg.max_plies, cfg.adjudicate_material)
            if status is None:
                continue
            result, termination = status
            z = np.array([result_value(result, t) for t in game.turns], dtype=np.float32)
            game_queue.put({
                "worker": worker_id,
                "planes": np.stack(game.planes),
                "pol_idx": game.pol_idx,
                "pol_p": game.pol_p,
                "z": z,
                "result": result,
                "termination": termination,
                "plies": len(game.moves),
                "moves": game.moves,
            })
            games[slot] = _Game()


def _send_live(live_queue, game: _Game) -> None:
    tree = game.tree
    root = tree.root
    if not root.moves:
        return
    board = tree.board
    sign = 1.0 if board.turn == chess.WHITE else -1.0
    top = tree.top_moves(5)
    by_prior = sorted(range(len(root.moves)), key=lambda i: -root.priors[i])[:5]
    msg = {
        "type": "live",
        "fen": board.fen(),
        "ply": board.ply(),
        "last_move": board.peek().uci() if board.move_stack else None,
        "turn": "white" if board.turn == chess.WHITE else "black",
        "value_white": sign * root.value,  # raw network value
        "search_white": sign * root.q(),  # value after search
        "top": top,
        "policy": [{"uci": root.moves[i].uci(), "san": board.san(root.moves[i]),
                    "prior": float(root.priors[i])} for i in by_prior],
    }
    try:
        live_queue.put_nowait(msg)
    except queue.Full:
        pass
