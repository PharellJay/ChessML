import sys

import pytest

# A minimal UCI engine: scores every legal move by the material it wins, so
# tests can run engine games without Stockfish installed.
FAKE_ENGINE = '''
import sys
import chess

VALUES = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900}
board, multipv = chess.Board(), 1
for line in sys.stdin:
    words = line.split()
    if not words:
        continue
    if words[0] == "uci":
        print("id name FakeFish")
        print("option name MultiPV type spin default 1 min 1 max 500")
        print("option name Threads type spin default 1 min 1 max 1024")
        print("option name Hash type spin default 16 min 1 max 1024")
        print("option name UCI_LimitStrength type check default false")
        print("option name UCI_Elo type spin default 1320 min 1320 max 3190")
        print("uciok")
    elif words[0] == "isready":
        print("readyok")
    elif words[:3] == ["setoption", "name", "MultiPV"]:
        multipv = int(words[4])
    elif words[0] == "position":
        board = chess.Board() if words[1] == "startpos" else chess.Board(" ".join(words[2:8]))
        if "moves" in words:
            for uci in words[words.index("moves") + 1:]:
                board.push_uci(uci)
    elif words[0] == "go":
        scored = []
        for move in board.legal_moves:
            victim = board.piece_type_at(move.to_square)
            scored.append((VALUES.get(victim, 0), move))
        scored.sort(key=lambda s: -s[0])
        for i, (cp, move) in enumerate(scored[:multipv]):
            print(f"info depth 1 multipv {i + 1} score cp {cp} nodes 1 pv {move.uci()}")
        print(f"bestmove {scored[0][1].uci()}")
    elif words[0] == "quit":
        break
    sys.stdout.flush()
'''


@pytest.fixture
def fake_engine(tmp_path):
    path = tmp_path / "fakefish"
    path.write_text(f"#!{sys.executable}\n{FAKE_ENGINE}")
    path.chmod(0o755)
    return str(path)
