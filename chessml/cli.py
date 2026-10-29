"""Command line interface: python -m chessml {train,evaluate,serve,report,list}."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from .config import CPU_COUNT, PRESET_NAMES, TrainConfig, all_presets, format_duration, parse_duration


def _add_training_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("intensity (preset values can be overridden individually)")
    g.add_argument("--preset", choices=PRESET_NAMES, default="balanced")
    g.add_argument("--workers", type=int, help="self-play worker processes")
    g.add_argument("--threads", type=int, help="torch threads for the trainer process")
    g.add_argument("--sims", type=int, help="MCTS simulations per move in self-play")
    g.add_argument("--batch-size", type=int, help="training batch size")
    g.add_argument("--nice", type=int, help="process niceness of trainer and workers (0-19)")
    d = p.add_argument_group("duration (default: until stopped)").add_mutually_exclusive_group()
    d.add_argument("--games", type=int, help="stop after N self-play games")
    d.add_argument("--time", help="stop after a wall-clock time such as 2h, 90m, 1h30m")
    d.add_argument("--until-stop", action="store_true", help="run until Ctrl+C (default)")
    o = p.add_argument_group("opponent (default: self-play only)")
    o.add_argument("--engine-share", type=float, metavar="FRACTION",
                   help="share of games played against Stockfish, 0-1 (e.g. 0.5; 1 = engine games only)")
    o.add_argument("--engine-path", help="UCI engine binary (default: $CHESSML_ENGINE or stockfish on PATH)")
    o.add_argument("--engine-level", type=int, help="engine strength 0-20 to start from (default 0)")
    o.add_argument("--engine-fixed", action="store_true", help="keep the engine level fixed instead of adapting it")
    o.add_argument("--engine-time", type=float, help="seconds of engine search per move (default 0.02)")
    o.add_argument("--no-imitate", action="store_true",
                   help="do not train on the engine's positions (learn only from the network's own moves)")
    r = p.add_argument_group("run")
    r.add_argument("--resume", metavar="RUN_OR_CHECKPOINT",
                   help="continue a run (name or path) or start a new run from a checkpoint file")
    r.add_argument("--name", help="name for a new run directory")
    a = p.add_argument_group("advanced")
    a.add_argument("--parallel-games", type=int, help="games searched together per worker")
    a.add_argument("--blocks", type=int, help="residual blocks (new runs only)")
    a.add_argument("--channels", type=int, help="channels per block (new runs only)")
    a.add_argument("--max-plies", type=int, help="adjudicate self-play games at this length")
    a.add_argument("--lr", type=float, help="learning rate")
    a.add_argument("--checkpoint-minutes", type=float, help="minutes between checkpoints")
    a.add_argument("--eval-minutes", type=float, help="minutes between evaluations (0 = off)")
    a.add_argument("--eval-games", type=int, help="evaluation games per opponent")
    a.add_argument("--seed", type=int)


def config_from_args(args) -> TrainConfig:
    overrides = {
        "workers": args.workers, "threads": args.threads, "sims": args.sims,
        "batch_size": args.batch_size, "nice": args.nice, "parallel_games": args.parallel_games,
        "blocks": args.blocks, "channels": args.channels, "max_plies": args.max_plies, "lr": args.lr,
        "checkpoint_minutes": args.checkpoint_minutes, "eval_minutes": args.eval_minutes,
        "eval_games": args.eval_games, "seed": args.seed,
        "max_games": args.games, "max_seconds": parse_duration(args.time) if args.time else None,
        "engine_share": args.engine_share, "engine_path": args.engine_path, "engine_level": args.engine_level,
        "engine_time": args.engine_time,
        "engine_adaptive": False if args.engine_fixed else None, "engine_imitate": False if args.no_imitate else None,
    }
    return TrainConfig.from_preset(args.preset, **overrides)



class ConsolePrinter:
    """Prints training events: a status line every few seconds plus logs and evaluations."""

    def __init__(self, every: float = 10.0):
        self.every = every
        self.last = 0.0
        self.loss = None

    def __call__(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "train":
            self.loss = event
        elif kind == "log":
            print(f"[{time.strftime('%H:%M:%S')}] {event['message']}", flush=True)
        elif kind == "status":
            if event["state"] in ("running", "paused") and time.time() - self.last < self.every:
                return
            self.last = time.time()
            loss = (f"loss {self.loss['total_loss']:.3f} (policy {self.loss['policy_loss']:.3f}, "
                    f"value {self.loss['value_loss']:.3f})" if self.loss else "loss -")
            eta = f" | eta {format_duration(event['eta'])}" if event.get("eta") is not None else ""
            engine = ""
            if event.get("engine_share"):
                score = event.get("engine_score")
                engine = (f" | stockfish level {event['engine_level']}"
                          + (f", score {100 * score:.0f}%" if score is not None else ""))
            print(f"[{format_duration(event['elapsed'])}] {event['state']} | games {event['games']} "
                  f"({event['games_per_sec']:.2f}/s) | steps {event['steps']} | {loss} | "
                  f"buffer {event['buffer']}{engine}{eta}", flush=True)


def cmd_train(args) -> int:
    from .trainer import Control, Trainer

    from .engine import check_engine_config

    cfg = config_from_args(args)
    try:
        check_engine_config(cfg)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    control = Control()
    presses = 0

    def on_signal(signum, frame):
        nonlocal presses
        presses += 1
        if presses == 1:
            print("\nStopping gracefully: discarding unfinished games and saving a checkpoint "
                  "(press Ctrl+C again to quit immediately)", flush=True)
            control.stop.set()
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    print(f"Training: preset {cfg.preset} ({CPU_COUNT} CPUs) workers={cfg.workers} threads={cfg.threads} "
          f"sims={cfg.sims} batch={cfg.batch_size} nice={cfg.nice} | {cfg.duration_text()} | {cfg.opponent_text()}",
          flush=True)
    trainer = Trainer(cfg, control, ConsolePrinter(), resume=args.resume, run_name=args.name)
    reason = trainer.run()
    print(f"Run directory: {trainer.run_dir}")
    return 0 if reason in ("completed", "stopped") else 1


def cmd_evaluate(args) -> int:
    from .evaluation import compute_ratings, evaluate_checkpoint, score
    from .metrics import MetricsDB
    from .network import load_checkpoint
    from .runs import resolve_checkpoint, run_checkpoints

    import torch

    torch.set_num_threads(args.threads)
    ckpt = resolve_checkpoint(args.checkpoint)
    run_dir = ckpt.parent.parent if ckpt.parent.name == "checkpoints" else None
    opponents = []
    for opp in args.opponents.split(","):
        opp = opp.strip()
        if opp == "previous":
            earlier = [p for p in run_checkpoints(run_dir)] if run_dir else []
            earlier = [p for p in earlier if p.stat().st_mtime < ckpt.stat().st_mtime]
            if earlier:
                opponents.append(str(earlier[0] if args.previous == "first" else earlier[-1]))
        elif opp in ("random", "material"):
            opponents.append(opp)
        else:
            opponents.append(str(resolve_checkpoint(opp)))
    db = MetricsDB(run_dir / "metrics.db") if run_dir else None
    have_ref = bool(db) and any(e["checkpoint"] == "material" for e in db.fetch("evals"))
    print(f"Evaluating {ckpt} ({args.games} games per opponent, {args.sims} simulations per move)")
    start = time.time()
    records = evaluate_checkpoint(str(ckpt), opponents, args.games, args.sims,
                                  include_material_reference=not have_ref, seed=args.seed)
    print(f"{'player':<22} {'opponent':<22} {'W':>4} {'D':>4} {'L':>4} {'score':>6}  terminations")
    for r in records:
        print(f"{r['player']:<22} {r['opponent']:<22} {r['wins']:>4} {r['draws']:>4} {r['losses']:>4} "
              f"{100 * score(r):>5.1f}%  {r['terminations']}")
    print(f"({time.time() - start:.0f}s)")
    if db:
        meta = load_checkpoint(str(ckpt))
        for r in records:
            db.insert("evals", {"elapsed": meta.get("elapsed", 0.0), "games": meta.get("games", 0),
                                "step": meta.get("steps", 0), "checkpoint": r["player"],
                                "opponent": r["opponent"], "wins": r["wins"], "draws": r["draws"],
                                "losses": r["losses"], "sims": args.sims})
        evals = db.fetch("evals")
        ratings = compute_ratings([(e["checkpoint"], e["opponent"], e["wins"], e["draws"], e["losses"])
                                   for e in evals])
        db.insert("ratings", {"elapsed": meta.get("elapsed", 0.0), "games": meta.get("games", 0),
                              "step": meta.get("steps", 0), "checkpoint": ckpt.name,
                              "rating": ratings.get(ckpt.name, 0.0), "material_rating": ratings.get("material")})
        print("Ratings (random = 0): " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(ratings.items(), key=lambda kv: kv[1])))
        db.close()
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    serve(args.host, args.port, open_browser=not args.no_browser)
    return 0


def cmd_report(args) -> int:
    from .report import generate_report

    generate_report(run=args.run, readme=Path(args.readme), out_dir=Path(args.out),
                    export_best=args.export_best)
    return 0


def cmd_list(args) -> int:
    from .runs import list_checkpoints, list_runs

    print("Presets on this machine:")
    for name, values in all_presets().items():
        print(f"  {name:<9} " + " ".join(f"{k}={v}" for k, v in values.items()))
    print("\nRuns:")
    for r in list_runs():
        print(f"  {r['name']:<28} {r['status']:<10} games={r['games']:<7} steps={r['steps']:<7} "
              f"time={format_duration(r['elapsed'])} checkpoints={r['checkpoints']}")
    print("\nCheckpoints:")
    for c in list_checkpoints():
        print(f"  {c['label']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m chessml", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="run self-play training")
    _add_training_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="play a checkpoint against baselines and earlier checkpoints")
    p.add_argument("checkpoint", help="checkpoint file, run name (latest checkpoint) or label")
    p.add_argument("--games", type=int, default=20, help="games per opponent")
    p.add_argument("--sims", type=int, default=64, help="MCTS simulations per move")
    p.add_argument("--opponents", default="random,material,previous",
                   help="comma-separated: random, material, previous, or checkpoint paths")
    p.add_argument("--previous", choices=("first", "last"), default="first",
                   help="which earlier checkpoint 'previous' means (default: the untrained start)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("serve", help="start the web interface")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("report", help="regenerate result charts and the README results section")
    p.add_argument("--run", help="run to report on (default: the run with the most games)")
    p.add_argument("--readme", default="README.md")
    p.add_argument("--out", default="docs/results", help="directory for chart images")
    p.add_argument("--export-best", action="store_true",
                   help="also copy the best-rated checkpoint (weights only) to models/best.pt")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("list", help="show presets, runs and checkpoints")
    p.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
