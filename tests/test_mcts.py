import chess
import numpy as np
import torch

from chessml.encoding import POLICY_SIZE
from chessml.mcts import MCTS, SearchTree
from chessml.network import ChessNet, Evaluator, NetConfig, load_checkpoint, model_from_checkpoint, save_checkpoint

from test_encoding import SPECIAL_FENS, random_positions


def uniform_eval(planes):
    n = len(planes)
    return np.zeros((n, POLICY_SIZE), dtype=np.float32), np.zeros(n, dtype=np.float32)


def small_net():
    torch.manual_seed(0)
    return Evaluator(ChessNet(NetConfig(blocks=1, channels=16)))


def test_network_shapes_and_value_range():
    net = ChessNet(NetConfig(blocks=2, channels=16))
    x = torch.randn(5, 20, 8, 8)
    policy, value = net(x)
    assert policy.shape == (5, POLICY_SIZE)
    assert value.shape == (5,)
    assert (value.abs() <= 1).all()


def test_checkpoint_round_trip(tmp_path):
    net = ChessNet(NetConfig(blocks=1, channels=16)).eval()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, net, step=7)
    loaded = model_from_checkpoint(path)
    x = torch.randn(3, 20, 8, 8)
    with torch.no_grad():
        assert torch.allclose(net(x)[0], loaded(x)[0])
    assert load_checkpoint(path)["step"] == 7


def test_mcts_returns_only_legal_moves():
    boards = random_positions(60, seed=5) + [chess.Board(f) for f in SPECIAL_FENS]
    boards = [b for b in boards if not b.is_game_over()]
    for evaluate in (uniform_eval, small_net()):
        mcts = MCTS(evaluate, rng=np.random.default_rng(0))
        trees = [SearchTree(b) for b in boards]
        mcts.run(trees, num_sims=24, add_noise=True)
        for tree, board in zip(trees, boards):
            for temperature in (0.0, 1.0):
                move = mcts.choose_move(tree, temperature)
                assert move in board.legal_moves
            assert tree.root.visits() >= 24
            moves, probs = tree.visit_distribution()
            assert set(moves) == set(board.legal_moves)
            assert abs(probs.sum() - 1) < 1e-6


def test_mcts_legal_through_a_whole_game_with_tree_reuse():
    mcts = MCTS(small_net(), rng=np.random.default_rng(1))
    tree = SearchTree()
    for _ in range(80):
        if tree.board.is_game_over():
            break
        mcts.run([tree], num_sims=16, add_noise=True)
        move = mcts.choose_move(tree, temperature=1.0)
        assert move in tree.board.legal_moves
        tree.advance(move)


def test_mcts_finds_mate_in_one():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")  # Ra8#
    tree = SearchTree(board)
    MCTS(uniform_eval).run([tree], num_sims=200)
    assert MCTS(uniform_eval).choose_move(tree) == chess.Move.from_uci("a1a8")


def test_terminal_root_has_no_moves():
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    tree = SearchTree(stalemate)
    MCTS(uniform_eval).run([tree], num_sims=10)
    assert tree.root.moves == [] and tree.root.terminal == 0.0
