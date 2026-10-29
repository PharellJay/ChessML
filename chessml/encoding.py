"""Board encoding and the move <-> policy-index mapping.

Everything is expressed from the point of view of the side to move: when it
is Black's turn the board is mirrored vertically and the colours are swapped,
so the network always "plays upwards" as White.

Input planes (NUM_PLANES x 8 x 8, uint8):
    0-5    our pawns, knights, bishops, rooks, queens, king
    6-11   their pawns, knights, bishops, rooks, queens, king
    12-15  castling rights: our O-O, our O-O-O, their O-O, their O-O-O
    16     en-passant target square
    17     half-move clock (raw count, scaled by 1/100 in `planes_to_float`)
    18     position has occurred before (repetition flag)
    19     all ones (lets the convolutions see the board edge)

Policy (AlphaZero layout, 73 move planes x 64 from-squares = 4672 logits):
    planes 0-55   queen-like moves: 8 directions x distance 1..7
                  (queen promotions use these planes too)
    planes 56-63  knight moves
    planes 64-72  under-promotions to knight, bishop, rook x 3 directions
    index = plane * 64 + from_square   (squares in side-to-move perspective)
"""

from __future__ import annotations

import chess
import numpy as np

NUM_PLANES = 20
NUM_MOVE_PLANES = 73
POLICY_SIZE = NUM_MOVE_PLANES * 64  # 4672
HALFMOVE_PLANE = 17

PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

# (file delta, rank delta) for the 8 queen directions: N, NE, E, SE, S, SW, W, NW
QUEEN_DIRECTIONS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
KNIGHT_DELTAS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def _plane_for(from_sq: int, to_sq: int, promotion: int | None) -> int | None:
    """Move plane for a move given in side-to-move (White) perspective."""
    df = chess.square_file(to_sq) - chess.square_file(from_sq)
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)
    if promotion in UNDERPROMOTIONS:
        if dr != 1 or abs(df) > 1:
            return None
        return 64 + UNDERPROMOTIONS.index(promotion) * 3 + (df + 1)
    if (df, dr) in KNIGHT_DELTAS:
        return 56 + KNIGHT_DELTAS.index((df, dr))
    dist = max(abs(df), abs(dr))
    if dist == 0 or (df != 0 and dr != 0 and abs(df) != abs(dr)):
        return None
    direction = (df // dist, dr // dist)
    return QUEEN_DIRECTIONS.index(direction) * 7 + (dist - 1)


def _build_tables():
    """Precompute index lookups for every geometrically possible move."""
    move_to_idx: dict[tuple[int, int, int | None], int] = {}
    idx_to_move: dict[int, tuple[int, int, int | None]] = {}
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq == to_sq:
                continue
            promos: list[int | None] = [None]
            if chess.square_rank(from_sq) == 6 and chess.square_rank(to_sq) == 7:
                promos = [None, chess.QUEEN, *UNDERPROMOTIONS]
            for promo in promos:
                plane = _plane_for(from_sq, to_sq, promo)
                if plane is None:
                    continue
                idx = plane * 64 + from_sq
                move_to_idx[(from_sq, to_sq, promo)] = idx
                # Queen promotions share the plain move's index.
                idx_to_move[idx] = (from_sq, to_sq, promo if promo in UNDERPROMOTIONS else None)
    return move_to_idx, idx_to_move


_MOVE_TO_IDX, _IDX_TO_MOVE = _build_tables()


def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    """Policy index of `move` played by the side `turn`."""
    from_sq, to_sq = move.from_square, move.to_square
    if turn == chess.BLACK:
        from_sq, to_sq = chess.square_mirror(from_sq), chess.square_mirror(to_sq)
    return _MOVE_TO_IDX[(from_sq, to_sq, move.promotion)]


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Inverse of `move_to_index` for the side to move on `board`.

    A queen-like move of a pawn onto the last rank decodes as a queen
    promotion. The returned move is not checked for legality.
    """
    from_sq, to_sq, promo = _IDX_TO_MOVE[index]
    if board.turn == chess.BLACK:
        from_sq, to_sq = chess.square_mirror(from_sq), chess.square_mirror(to_sq)
    if promo is None and board.piece_type_at(from_sq) == chess.PAWN:
        if chess.square_rank(to_sq) in (0, 7):
            promo = chess.QUEEN
    return chess.Move(from_sq, to_sq, promotion=promo)


def legal_moves_and_indices(board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
    """All legal moves of `board` and their policy indices."""
    moves = list(board.legal_moves)
    turn = board.turn
    if turn == chess.WHITE:
        idx = [_MOVE_TO_IDX[(m.from_square, m.to_square, m.promotion)] for m in moves]
    else:
        mirror = chess.square_mirror
        idx = [_MOVE_TO_IDX[(mirror(m.from_square), mirror(m.to_square), m.promotion)] for m in moves]
    return moves, np.asarray(idx, dtype=np.int64)


def _bitboards_to_planes(bbs: list[int]) -> np.ndarray:
    arr = np.asarray(bbs, dtype="<u8")
    bits = np.unpackbits(arr.view(np.uint8).reshape(-1, 8), axis=1, bitorder="little")
    return bits.reshape(-1, 8, 8)


def encode_board(board: chess.Board, repetition: bool = False) -> np.ndarray:
    """Encode `board` as a (NUM_PLANES, 8, 8) uint8 array."""
    us, them = board.turn, not board.turn
    bbs = [board.pieces_mask(pt, us) for pt in PIECE_TYPES]
    bbs += [board.pieces_mask(pt, them) for pt in PIECE_TYPES]
    if us == chess.BLACK:
        bbs = [chess.flip_vertical(bb) for bb in bbs]
    ep = 0
    if board.ep_square is not None and board.has_legal_en_passant():
        ep_sq = board.ep_square if us == chess.WHITE else chess.square_mirror(board.ep_square)
        ep = chess.BB_SQUARES[ep_sq]
    bbs.append(ep)

    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.uint8)
    pieces = _bitboards_to_planes(bbs)
    planes[0:12] = pieces[0:12]
    planes[16] = pieces[12]
    planes[12] = board.has_kingside_castling_rights(us)
    planes[13] = board.has_queenside_castling_rights(us)
    planes[14] = board.has_kingside_castling_rights(them)
    planes[15] = board.has_queenside_castling_rights(them)
    planes[HALFMOVE_PLANE] = min(board.halfmove_clock, 100)
    planes[18] = 1 if repetition else 0
    planes[19] = 1
    return planes


def planes_to_float(planes: np.ndarray) -> np.ndarray:
    """Convert a batch (N, NUM_PLANES, 8, 8) of uint8 planes to network input."""
    x = planes.astype(np.float32)
    x[:, HALFMOVE_PLANE] *= 0.01
    return x
