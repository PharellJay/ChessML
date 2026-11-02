"""Self-play worker processes (including games against an engine) and the game
rules shared with evaluation."""

from __future__ import annotations

import queue
import signal
import time
from collections import Counter

import chess
import numpy as np

from .encoding import NUM_PLANES

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
    def __init__(self, engine_color: chess.Color | None = None, engine_level: int = 0):
        from .mcts import SearchTree

        self.tree = SearchTree()
        self.planes: list[np.ndarray] = []
        self.pol_idx: list[np.ndarray] = []
        self.pol_p: list[np.ndarray] = []
        self.turns: list[bool] = []
        self.moves: list[str] = []
        self.engine_color = engine_color  # None for a self-play game
        self.engine_level = engine_level
        self.full = True  # this move gets the full search (and becomes a training sample)
        self.resign_allowed = True
        self.would_resign: chess.Color | None = None  # in games played out: who would have resigned first
        self.low = {chess.WHITE: 0, chess.BLACK: 0}  # consecutive full searches below the threshold
        self.min_value = {chess.WHITE: 1.0, chess.BLACK: 1.0}  # lowest full-search value each side saw

    def engine_to_move(self) -> bool:
        return self.engine_color is not None and self.tree.board.turn == self.engine_color

    def opponent_label(self) -> str:
        if self.engine_color is None:
            return "self-play"
        net = "White" if self.engine_color == chess.BLACK else "Black"
        return f"vs Stockfish level {self.engine_level} (network plays {net})"


def _ignore_sigint_and_nice(nice: int) -> None:
    # Ctrl+C is handled by the parent, which stops children gracefully.
    from .config import apply_niceness

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    apply_niceness(nice)


def worker_main(worker_id, cfg_dict, weights_path, weights_version, engine_level, resign_threshold, counters,
                game_queue, live_queue, focus, pause_event, stop_event, seed):
    """Play games until `stop_event` is set; unfinished games are discarded.

    A share `engine_share` of the games is played against the UCI engine at
    the level the trainer publishes in `engine_level`. For the live view every
    worker sends a snapshot of all its games once a second, and the worker
    owning the game selected in `focus` (worker, slot) sends its details.

    Search effort is randomised per move (playout cap randomisation): with
    probability `full_search_prob` a move gets the full `sims` search, with
    Dirichlet noise, and becomes a training sample; otherwise a quick
    `fast_sims` search only picks the move. A side whose full-search value
    stays below the published `resign_threshold` for `resign_moves` full
    searches in a row resigns, except in a share `resign_check_share` of
    games: those are played out, and report how low each side's value went,
    so the trainer can set the threshold from real outcomes.
    """
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
    engine = None

    def new_game() -> _Game:
        if cfg.engine_share > 0 and rng.random() < cfg.engine_share:
            game = _Game(chess.WHITE if rng.random() < 0.5 else chess.BLACK, engine_level.value)
        else:
            game = _Game()
        game.resign_allowed = rng.random() >= cfg.resign_check_share
        return game

    def finish(slot: int, game: _Game, result: str, termination: str) -> None:
        z = np.array([result_value(result, t) for t in game.turns], dtype=np.float32)
        net_score = None
        if game.engine_color is not None:
            net_score = (result_value(result, not game.engine_color) + 1) / 2
        game_queue.put({
            "worker": worker_id,
            "planes": np.stack(game.planes) if game.planes else np.zeros((0, NUM_PLANES, 8, 8), np.uint8),
            "pol_idx": game.pol_idx,
            "pol_p": game.pol_p,
            "z": z,
            "result": result,
            "termination": termination,
            "plies": len(game.moves),
            "moves": game.moves,
            "opponent": "self" if game.engine_color is None else "engine",
            "engine_level": None if game.engine_color is None else game.engine_level,
            "net_score": net_score,
            "resign_check": not game.resign_allowed,
            "would_resign": None if game.would_resign is None else
            ("white" if game.would_resign == chess.WHITE else "black"),
            "min_value": None if game.resign_allowed else
            {"white": game.min_value[chess.WHITE], "black": game.min_value[chess.BLACK]},
        })
        games[slot] = new_game()

    games = [new_game() for _ in range(cfg.parallel_games)]
    last_live = last_grid = 0.0

    try:
        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.1)
                continue
            if weights_version.value != loaded_version:
                loaded_version = weights_version.value
                model.load_state_dict(load_checkpoint(weights_path)["model"])
                model.eval()

            searching = [g for g in games if not g.engine_to_move()]
            for g in searching:
                g.full = rng.random() < cfg.full_search_prob
            evals_before = mcts.leaf_evals
            mcts.run([g.tree for g in searching], [cfg.sims if g.full else cfg.fast_sims for g in searching],
                     add_noise=[g.full for g in searching])
            counters[worker_id] += mcts.leaf_evals - evals_before

            now = time.time()
            if focus[0] == worker_id and now - last_live > 0.25:
                last_live = now
                slot = min(max(focus[1], 0), len(games) - 1)
                _send_live(live_queue, games[slot], worker_id, slot)
            if now - last_grid > 1.0:
                last_grid = now
                _send_grid(live_queue, games, worker_id)

            threshold = resign_threshold.value
            for slot, game in enumerate(games):
                if stop_event.is_set():
                    break
                tree = game.tree
                board = tree.board
                if game.engine_to_move():
                    if engine is None:
                        from .engine import open_engine

                        engine = open_engine(cfg.engine_path)
                    move = _engine_move(engine, game, cfg, rng)
                else:
                    if game.full and board.ply() >= cfg.resign_min_ply and tree.root.moves:
                        side, value = board.turn, tree.root.q()
                        game.min_value[side] = min(game.min_value[side], value)
                        game.low[side] = game.low[side] + 1 if value < threshold else 0
                        if game.low[side] >= cfg.resign_moves:
                            if game.resign_allowed:
                                finish(slot, game, "0-1" if side == chess.WHITE else "1-0", "resignation")
                                continue
                            if game.would_resign is None:
                                game.would_resign = side
                    if game.full:
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
                if status is not None:
                    finish(slot, game, *status)
    finally:
        if engine is not None:
            engine.quit()


def _engine_move(engine, game: _Game, cfg, rng: np.random.Generator) -> chess.Move:
    """The engine's move; with imitation on, its position is recorded with its best moves as target."""
    import chess.engine

    from .encoding import encode_board
    from .engine import analyse, choose_engine_move, policy_target

    tree = game.tree
    board = tree.board
    scored = analyse(engine, board, chess.engine.Limit(nodes=cfg.engine_nodes), cfg.engine_multipv)
    if cfg.engine_imitate and scored:
        idx, p = policy_target(scored, board.turn)
        game.planes.append(encode_board(board, repetition=tree.is_repetition()))
        game.pol_idx.append(idx)
        game.pol_p.append(p)
        game.turns.append(board.turn)
    return choose_engine_move(scored, board, game.engine_level, rng)


def _send_grid(live_queue, games: list[_Game], worker_id: int) -> None:
    """A small snapshot of every game of this worker: enough to draw a thumbnail."""
    items = []
    for slot, game in enumerate(games):
        board = game.tree.board
        items.append({"worker": worker_id, "slot": slot, "board": board.board_fen(), "ply": board.ply(),
                      "last_move": board.peek().uci() if board.move_stack else None,
                      "engine": None if game.engine_color is None else
                      ("white" if game.engine_color == chess.WHITE else "black"),
                      "engine_level": game.engine_level if game.engine_color is not None else None})
    try:
        live_queue.put_nowait({"type": "grid", "worker": worker_id, "games": items})
    except queue.Full:
        pass


def _send_live(live_queue, game: _Game, worker_id: int = 0, slot: int = 0) -> None:
    tree = game.tree
    root = tree.root
    if not root.moves:
        return  # the engine is to move here; the next snapshot comes on the network's turn
    board = tree.board
    sign = 1.0 if board.turn == chess.WHITE else -1.0
    top = tree.top_moves(5)
    by_prior = sorted(range(len(root.moves)), key=lambda i: -root.priors[i])[:5]
    msg = {
        "type": "live",
        "worker": worker_id,
        "slot": slot,
        "fen": board.fen(),
        "ply": board.ply(),
        "last_move": board.peek().uci() if board.move_stack else None,
        "turn": "white" if board.turn == chess.WHITE else "black",
        "opponent": game.opponent_label(),
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
