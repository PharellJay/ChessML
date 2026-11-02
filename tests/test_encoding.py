import random

import chess
import numpy as np
import pytest

from chessml.encoding import (
    HALFMOVE_PLANE,
    NUM_PLANES,
    POLICY_SIZE,
    encode_board,
    index_to_move,
    legal_moves_and_indices,
    move_to_index,
    planes_to_float,
)


def random_positions(n, seed=0, max_plies=120):
    """Positions reached by random play, including some near the end of games."""
    rng = random.Random(seed)
    positions = []
    while len(positions) < n:
        board = chess.Board()
        for _ in range(rng.randrange(max_plies)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        positions.append(board)
    return positions


SPECIAL_FENS = [
    # promotions and under-promotions with captures, both colours
    "1n2k3/P7/8/8/8/8/7p/4K1N1 w - - 0 1",
    "1n2k3/P7/8/8/8/8/7p/4K1N1 b - - 0 1",
    # en passant available
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    # castling both sides
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
]


def test_planes_shape_and_start_position():
    planes = encode_board(chess.Board())
    assert planes.shape == (NUM_PLANES, 8, 8)
    assert planes.dtype == np.uint8
    assert planes[0].sum() == 8 and planes[6].sum() == 8  # pawns
    assert planes[5].sum() == 1 and planes[11].sum() == 1  # kings
    assert planes[0][1].all()  # our pawns on rank 2
    assert planes[6][6].all()  # their pawns on rank 7
    assert planes[12:16].all()  # all castling rights
    assert not planes[16].any()  # no en passant
    assert planes[19].all()


def test_encoding_is_side_to_move_relative():
    # A position and its colour-flipped mirror look identical to the side to move.
    for board in random_positions(200, seed=1) + [chess.Board(f) for f in SPECIAL_FENS]:
        assert np.array_equal(encode_board(board), encode_board(board.mirror()))


def test_en_passant_and_halfmove_planes():
    board = chess.Board(SPECIAL_FENS[2])
    planes = encode_board(board)
    assert planes[16].sum() == 1 and planes[16][5][5] == 1  # f6
    board = chess.Board("8/8/8/4k3/8/8/8/4K2R w K - 37 80")
    x = planes_to_float(encode_board(board)[None])
    assert x[0, HALFMOVE_PLANE, 0, 0] == pytest.approx(0.37)
    assert encode_board(board, repetition=True)[18].all()


def test_policy_indices_unique_and_in_range():
    for board in random_positions(300, seed=2) + [chess.Board(f) for f in SPECIAL_FENS]:
        moves, idx = legal_moves_and_indices(board)
        assert len(set(idx.tolist())) == len(moves)
        assert ((idx >= 0) & (idx < POLICY_SIZE)).all()


def test_move_index_round_trip():
    for board in random_positions(300, seed=3) + [chess.Board(f) for f in SPECIAL_FENS]:
        for move in board.legal_moves:
            index = move_to_index(move, board.turn)
            assert index_to_move(index, board) == move


def test_promotions_get_distinct_indices():
    board = chess.Board(SPECIAL_FENS[0])
    promos = [m for m in board.legal_moves if m.promotion]
    assert len(promos) == 8  # a8 push and axb8 capture, 4 pieces each
    assert len({move_to_index(m, board.turn) for m in promos}) == 8


def test_perspective_symmetry_of_indices():
    # e2e4 for White and e7e5 for Black are the same move from the mover's view.
    white = move_to_index(chess.Move.from_uci("e2e4"), chess.WHITE)
    black = move_to_index(chess.Move.from_uci("e7e5"), chess.BLACK)
    assert white == black
