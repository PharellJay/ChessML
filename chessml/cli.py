"""Command line interface: python -m chessml {train,evaluate,calibrate,elo,gif,serve,report,list}."""

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
    g.add_argument("--buffer-size", type=int, help="replay buffer size in positions (about 1.85 KB each)")
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
    o.add_argument("--engine-nodes", type=int, help="engine search per move in nodes (default 5000, as in the Elo ladder)")
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
    a.add_argument("--full-search-prob", type=float,
                   help="share of moves searched with all --sims and used for training (default 0.25)")
    a.add_argument("--fast-sims", type=int, help="simulations for the other moves (default 16)")
    a.add_argument("--resign-threshold", type=float,
                   help="resign below this search value (default -0.9; -1 turns resignation off)")
    a.add_argument("--lr", type=float, help="learning rate")
    a.add_argument("--checkpoint-minutes", type=float, help="minutes between checkpoints")
    a.add_argument("--eval-minutes", type=float, help="minutes between evaluations (0 = off)")
    a.add_argument("--eval-games", type=int, help="evaluation games per opponent")
    a.add_argument("--eval-calibrated-games", type=int,
                   help="evaluation games per calibrated Stockfish opponent (default 20)")
    a.add_argument("--seed", type=int)


def config_from_args(args) -> TrainConfig:
    overrides = {
        "workers": args.workers, "threads": args.threads, "sims": args.sims,
        "batch_size": args.batch_size, "nice": args.nice, "parallel_games": args.parallel_games,
        "blocks": args.blocks, "channels": args.channels, "max_plies": args.max_plies, "lr": args.lr,
        "full_search_prob": args.full_search_prob, "fast_sims": args.fast_sims,
        "resign_threshold": args.resign_threshold,
        "checkpoint_minutes": args.checkpoint_minutes, "eval_minutes": args.eval_minutes,
        "eval_games": args.eval_games, "eval_calibrated_games": args.eval_calibrated_games, "seed": args.seed,
        "max_games": args.games, "max_seconds": parse_duration(args.time) if args.time else None,
        "engine_share": args.engine_share, "engine_path": args.engine_path, "engine_level": args.engine_level,
        "engine_nodes": args.engine_nodes, "buffer_size": args.buffer_size,
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


def _print_match(record: dict, done: int, total: int) -> None:
    print(f"[{done}/{total}] {record['player']} vs {record['opponent']}: "
          f"+{record['wins']} ={record['draws']} -{record['losses']}", flush=True)


def _print_stage(stage: str, info: dict) -> None:
    if stage == "scouting":
        print(f"Scouting: {info['games']} games against each of {len(info['opponents'])} calibrated players",
              flush=True)
    else:
        estimate = "" if info["estimate"] is None else f"Scouting estimate {info['estimate']:.0f}; "
        print(f"{estimate}measuring against {', '.join(info['opponents'])}", flush=True)


def cmd_calibrate(args) -> int:
    from .elo import calibrate, calibration_pairs, default_players, load_calibration, refine_calibration, save_calibration

    def stage(name: str, info: dict) -> None:
        what = "Refining the ladder" if name == "refining" else "Calibrating the Elo ladder"
        print(f"{what}: {info['matches']} matches of {info['games']} games on {args.processes} processes", flush=True)

    try:
        if args.refine:
            current = load_calibration()
            if not current:
                print("error: no calibration to refine yet. Run: python -m chessml calibrate", file=sys.stderr)
                return 2
            games = args.refine_games if args.refine_games is not None else (
                len(calibration_pairs(current["players"])) * current["games_per_pair"])
            cal = refine_calibration(current, games, args.processes, args.nice, args.seed, progress=_print_match,
                                     on_stage=stage, engine_path=args.engine_path)
        else:
            print("This takes about 4 hours of CPU time with the refinement round: "
                  "30 minutes on 10 processes, two hours on 2.", flush=True)
            cal = calibrate(args.engine_path, args.games, args.processes, args.nice, seed=args.seed,
                            progress=_print_match, refine_games=args.refine_games, on_stage=stage)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    path = save_calibration(cal)
    print(f"\nRatings ({cal['engine']}, anchors at their UCI_Elo):")
    for name, rating in sorted(cal["ratings"].items(), key=lambda kv: -kv[1]):
        err = cal["errors"].get(name)
        print(f"  {name:<14} {rating:6.0f}  " + ("anchor" if name in cal["anchors"] else f"± {1.96 * err:.0f}"))
    print(f"Saved {path} ({format_duration(cal['seconds'])})")
    return 0


def cmd_elo(args) -> int:
    from .elo import load_calibration, measure, save_result, store_in_run
    from .runs import resolve_checkpoint

    calibration = load_calibration()
    if not calibration:
        print("error: no Elo calibration yet. Run once: python -m chessml calibrate", file=sys.stderr)
        return 2
    ckpt = resolve_checkpoint(args.checkpoint)
    opponents = [o.strip() for o in args.opponents.split(",")] if args.opponents else None
    print(f"Measuring {ckpt} ({args.games} games per opponent, {args.sims} simulations per move)",
          flush=True)
    start = time.time()
    try:
        res = measure(ckpt, calibration, args.games, args.sims, args.processes, args.nice, opponents,
                      args.engine_path, args.seed, progress=_print_match, on_stage=_print_stage)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"\n{'opponent':<14} {'its Elo':>7} {'W':>4} {'D':>4} {'L':>4} {'score':>6}")
    by_name = {r["opponent"]: r for r in res["records"]}
    for name, rating in sorted(calibration["ratings"].items(), key=lambda kv: kv[1]):
        if name in by_name:
            r = by_name[name]
            n = r["wins"] + r["draws"] + r["losses"]
            print(f"{name:<14} {rating:7.0f} {r['wins']:>4} {r['draws']:>4} {r['losses']:>4} "
                  f"{100 * (r['wins'] + r['draws'] / 2) / n:5.1f}%")
    err = f" ± {1.96 * res['error']:.0f} (95%)" if res["error"] else ""
    print(f"\n{res['checkpoint']}: Elo {res['elo']:.0f}{err}   ({time.time() - start:.0f}s)")
    if res["error"] and res["own_error"]:  # the two parts add in quadrature
        total, own = 1.96 * res["error"], 1.96 * res["own_error"]
        ladder = max(total ** 2 - own ** 2, 0) ** 0.5
        hint = ("refining the calibration (calibrate --refine) would narrow it more than more games"
                if ladder > own else "more games per opponent (--games) would narrow it most")
        print(f"Of the ± {total:.0f}: ± {own:.0f} from these games, ± {ladder:.0f} from the calibrated ladder; {hint}.")
    print("Elo on the scale of Stockfish's UCI_Elo (CCRL 40/4), not FIDE or online ratings.")
    store_in_run(res, ckpt, args.sims)  # the run's charts use these games too
    save_result(res, ckpt, args.games, args.sims)  # and the web page's Strength tab lists the result
    return 0


def cmd_gif(args) -> int:
    from .gif import render_gif
    from .runs import resolve_checkpoint

    ckpt = resolve_checkpoint(args.checkpoint)
    out = args.out or f"docs/games/{ckpt.stem}.gif"
    print(f"Playing {ckpt} vs {args.opponent} ({args.sims} simulations per move)...", flush=True)
    try:
        res = render_gif(ckpt, out, args.opponent, args.color, args.sims, args.max_plies, args.size, args.frame_ms,
                         args.title, args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{res['result']} · {res['outcome']} after {res['plies']} plies -> {res['path']} "
          f"({res['bytes'] / 1e6:.1f} MB)")
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

    procs = max(1, CPU_COUNT - 2)
    p = sub.add_parser("calibrate", help="calibrate the Elo ladder against Stockfish's UCI_Elo (once per machine)")
    p.add_argument("--games", type=int, default=20, help="games per calibration match (default 20)")
    p.add_argument("--refine-games", type=int,
                   help="games of the refinement round, played where they narrow the ladder most "
                        "(default: as many as the first round; 0 skips it)")
    p.add_argument("--refine", action="store_true",
                   help="only add a refinement round to the existing calibration")
    p.add_argument("--processes", type=int, default=procs, help=f"parallel matches (default {procs})")
    p.add_argument("--engine-path", help="Stockfish binary (default: $CHESSML_ENGINE or stockfish on PATH)")
    p.add_argument("--nice", type=int, default=10, help="niceness of the match processes")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("elo", help="measure a checkpoint's Elo against the calibrated ladder")
    p.add_argument("checkpoint", help="checkpoint file, run name (latest checkpoint) or label")
    p.add_argument("--games", type=int, default=20,
                   help="games per opponent near the scouting estimate (default 20)")
    p.add_argument("--sims", type=int, default=64, help="MCTS simulations per move")
    p.add_argument("--opponents", help="comma-separated calibrated players to play, skipping the scouting stage "
                                       "(default: the players near the scouting estimate)")
    p.add_argument("--processes", type=int, default=procs, help=f"parallel matches (default {procs})")
    p.add_argument("--engine-path", help="Stockfish binary (default: the calibration's, or stockfish on PATH)")
    p.add_argument("--nice", type=int, default=10, help="niceness of the match processes")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_elo)

    p = sub.add_parser("gif", help="render one game of a checkpoint as an animated GIF")
    p.add_argument("checkpoint", help="checkpoint file, run name (latest checkpoint) or label")
    p.add_argument("--opponent", default="self",
                   help="self (default), random, material, sf-level-N or sf-elo-N (Stockfish), or a checkpoint path")
    p.add_argument("--color", choices=("white", "black"), default="white", help="the checkpoint's colour")
    p.add_argument("--sims", type=int, default=64, help="MCTS simulations per move")
    p.add_argument("--max-plies", type=int, default=200, help="adjudicate at this length, as in training")
    p.add_argument("--size", type=int, default=360, help="board size in pixels")
    p.add_argument("--frame-ms", type=int, default=300, help="milliseconds per move")
    p.add_argument("--title", help="text above the board (default: checkpoint vs opponent)")
    p.add_argument("--out", help="output file (default: docs/games/<checkpoint>.gif)")
    p.add_argument("--seed", type=int, help="seed for random and Stockfish opponents")
    p.set_defaults(func=cmd_gif)

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
