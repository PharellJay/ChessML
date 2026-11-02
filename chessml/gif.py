"""`python -m chessml gif`: one game of a checkpoint as an animated GIF, for the README.

The checkpoint plays from the start position against itself, a baseline,
a calibrated Stockfish player or another checkpoint, with the same rules as
training (move limit and material adjudication). Every ply becomes a frame,
and the last frame holds the result.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import chess
from PIL import Image, ImageDraw, ImageFont

from .evaluation import MaterialPlayer, NetPlayer, RandomPlayer
from .runs import PACKAGE_DIR
from .selfplay import game_status

PIECES_DIR = PACKAGE_DIR / "static" / "vendor" / "img" / "chesspieces" / "wikipedia"
LIGHT, DARK = (240, 217, 181), (181, 136, 99)  # the web board's colours
LAST_LIGHT, LAST_DARK = (247, 204, 104), (219, 162, 62)  # last-move squares
BG, INK, MUTED = (252, 252, 251), (11, 11, 11), (111, 110, 105)


def make_opponent(spec: str, net: NetPlayer, sims: int, rng: random.Random):
    """'self', 'random', 'material', a calibrated Stockfish player (sf-level-N, sf-elo-N) or a checkpoint path."""
    if spec == "self":
        return net
    if spec == "random":
        return RandomPlayer(rng)
    if spec == "material":
        return MaterialPlayer(rng)
    if spec.startswith("sf-level-") or spec.startswith("sf-elo-"):
        from .elo import ELO_NODES, LEVEL_NODES, MULTIPV, engine_setup, load_calibration, make_player
        from .engine import find_engine

        value = spec.rsplit("-", 1)[1]
        player = ({"name": spec, "kind": "level", "level": float(value) if "." in value else int(value)}
                  if spec.startswith("sf-level-") else {"name": spec, "kind": "elo", "elo": int(value)})
        calibration = load_calibration()
        setup = engine_setup(calibration) if calibration else {
            "engine_path": find_engine(None), "level_nodes": LEVEL_NODES, "elo_nodes": ELO_NODES, "multipv": MULTIPV}
        return make_player(player, setup, rng)
    return NetPlayer(spec, sims)


def play_game(white, black, max_plies: int = 200, adjudicate_material: int = 3) -> tuple[list[chess.Move], str, str]:
    """Play from the start position; returns the moves, the result and how the game ended."""
    board = chess.Board()
    history = Counter([board._transposition_key()])
    while (status := game_status(board, history, max_plies, adjudicate_material)) is None:
        player = white if board.turn == chess.WHITE else black
        board.push(player.choose([board])[0])
        history[board._transposition_key()] += 1
    return list(board.move_stack), status[0], status[1]


class _Renderer:
    def __init__(self, size: int):
        self.sq = size // 8
        self.board_px = self.sq * 8
        self.pad = max(10, self.sq // 3)
        self.head = self.sq + self.pad
        self.font = ImageFont.load_default(size=max(11, self.sq // 3))
        self.title_font = ImageFont.load_default(size=max(13, int(self.sq / 2.4)))
        self.pieces = {}
        for color in "wb":
            for kind in "KQRBNP":
                img = Image.open(PIECES_DIR / f"{color}{kind}.png").convert("RGBA")
                self.pieces[color + kind] = img.resize((self.sq, self.sq), Image.LANCZOS)

    def frame(self, board: chess.Board, title: str, status: str) -> Image.Image:
        width = self.board_px + 2 * self.pad
        img = Image.new("RGB", (width, self.head + self.board_px + self.pad), BG)
        draw = ImageDraw.Draw(img)
        draw.text((self.pad, self.pad // 2), title, font=self.title_font, fill=INK)
        draw.text((self.pad, self.pad // 2 + self.sq // 2), status, font=self.font, fill=MUTED)
        last = board.peek() if board.move_stack else None
        marked = {last.from_square, last.to_square} if last else set()
        for square in chess.SQUARES:
            f, r = chess.square_file(square), chess.square_rank(square)
            x, y = self.pad + f * self.sq, self.head + (7 - r) * self.sq
            light = (f + r) % 2 == 1
            color = (LAST_LIGHT if light else LAST_DARK) if square in marked else (LIGHT if light else DARK)
            draw.rectangle([x, y, x + self.sq - 1, y + self.sq - 1], fill=color)
            piece = board.piece_at(square)
            if piece:
                key = ("w" if piece.color == chess.WHITE else "b") + piece.symbol().upper()
                img.paste(self.pieces[key], (x, y), self.pieces[key])
        return img


def _describe(result: str, termination: str, net_white: bool, opponent: str) -> str:
    if result == "1/2-1/2":
        return f"Draw ({termination})"
    white_won = result == "1-0"
    if opponent == "self":
        return f"{'White' if white_won else 'Black'} wins ({termination})"
    return f"{'Network' if white_won == net_white else 'Opponent'} wins ({termination})"


def render_gif(checkpoint: str | Path, out: str | Path, opponent: str = "self", color: str = "white",
               sims: int = 64, max_plies: int = 200, size: int = 360, frame_ms: int = 300,
               title: str | None = None, seed: int | None = None) -> dict:
    """Play one game and save it as a GIF; returns a summary of the game."""
    rng = random.Random(seed)
    net = NetPlayer(checkpoint, sims)
    other = make_opponent(opponent, net, sims, rng)
    net_white = color == "white"
    white, black = (net, other) if net_white else (other, net)
    try:
        moves, result, termination = play_game(white, black, max_plies)
    finally:
        if hasattr(other, "close"):
            other.close()
    name = Path(checkpoint).name
    if title is None:
        title = f"{name} vs itself" if opponent == "self" else (
            f"{name} ({color}) vs {Path(opponent).name if opponent.endswith('.pt') else opponent}")
    renderer = _Renderer(size)
    board = chess.Board()
    frames = [renderer.frame(board, title, "Start position")]
    for i, move in enumerate(moves, 1):
        san = board.san(move)
        board.push(move)
        number = (i + 1) // 2
        frames.append(renderer.frame(board, title, f"{number}{'.' if i % 2 else '...'} {san}"))
    outcome = _describe(result, termination, net_white, opponent)
    frames.append(renderer.frame(board, title, f"{result} · {outcome} · {len(moves)} plies"))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample = Image.new("RGB", (frames[0].width * 2, frames[0].height))  # every piece, and the last-move colours
    sample.paste(frames[0], (0, 0))
    sample.paste(frames[-1], (frames[0].width, 0))
    palette = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    images = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    durations = [frame_ms * 3] + [frame_ms] * len(moves) + [5000]
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    return {"path": out, "plies": len(moves), "result": result, "termination": termination, "outcome": outcome,
            "bytes": out.stat().st_size}
