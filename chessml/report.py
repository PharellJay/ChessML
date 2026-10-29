"""`python -m chessml report`: result charts and the README results section.

Everything is read from a run's metrics database, so the README only ever
shows numbers that a real run produced.
"""

from __future__ import annotations

import os
import platform
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import format_duration  # noqa: E402
from .evaluation import checkpoint_ratings, compute_ratings  # noqa: E402
from .metrics import read_history  # noqa: E402
from .runs import MODELS_DIR, list_runs, read_json, resolve_run  # noqa: E402

START, END = "<!-- results:start -->", "<!-- results:end -->"

# reference palette, light mode: categorical slots in fixed order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def machine_info() -> dict:
    """Hardware and software of the current machine."""
    import psutil
    import torch

    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": "none (CPU only)" if not torch.cuda.is_available() else torch.cuda.get_device_name(0),
    }


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def _rolling(games, window=100):
    xs, white, draw, black, length = [], [], [], [], []
    for i, g in enumerate(games):
        chunk = games[max(0, i - window + 1): i + 1]
        n = len(chunk)
        w = sum(c["result"] == "1-0" for c in chunk)
        b = sum(c["result"] == "0-1" for c in chunk)
        xs.append(g["game"])
        white.append(100 * w / n)
        black.append(100 * b / n)
        draw.append(100 * (n - w - b) / n)
        length.append(sum(c["plies"] for c in chunk) / n)
    return xs, white, draw, black, length


def make_charts(h: dict, out_dir: Path) -> list[tuple[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = []

    if h["train"]:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        t = [r["elapsed"] / 60 for r in h["train"]]
        for i, key in enumerate(("policy_loss", "value_loss", "total_loss")):
            ax.plot(t, [r[key] for r in h["train"]], color=COLORS[i], lw=2, label=key.replace("_", " "))
        _style(ax, "Training loss", "training time (minutes)", "loss")
        ax.legend(frameon=False)
        _save(fig, out_dir / "loss.png")
        charts.append(("loss.png", "Policy, value and total loss"))

    if h["system"]:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        t = [r["elapsed"] / 60 for r in h["system"]]
        ax.plot(t, [r["games_per_sec"] for r in h["system"]], color=COLORS[0], lw=2)
        _style(ax, "Self-play throughput (60 s window)", "training time (minutes)", "games / second")
        ax.set_ylim(bottom=0)
        _save(fig, out_dir / "throughput.png")
        charts.append(("throughput.png", "Self-play games per second"))

        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.plot(t, [r["cpu_total"] for r in h["system"]], color=COLORS[0], lw=2)
        _style(ax, "CPU usage, all cores", "training time (minutes)", "%")
        ax.set_ylim(0, 105)
        _save(fig, out_dir / "cpu.png")
        charts.append(("cpu.png", "Overall CPU usage"))

    if h["games"]:
        xs, white, draw, black, length = _rolling(h["games"])
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for i, (label, ys) in enumerate((("White wins", white), ("draws", draw), ("Black wins", black))):
            ax.plot(xs, ys, color=COLORS[i], lw=2, label=label)
        _style(ax, "Self-play outcomes (last 100 games)", "game", "%")
        ax.set_ylim(0, 100)
        ax.legend(frameon=False)
        _save(fig, out_dir / "outcomes.png")
        charts.append(("outcomes.png", "Self-play outcomes"))

        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.plot(xs, length, color=COLORS[0], lw=2)
        _style(ax, "Average game length (last 100 games)", "game", "plies")
        ax.set_ylim(bottom=0)
        _save(fig, out_dir / "game_length.png")
        charts.append(("game_length.png", "Average self-play game length"))

    if h["evals"]:
        # final fit over all evaluation games, so the chart matches the ratings table
        fit = checkpoint_ratings(h["evals"])
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.plot([p["games"] for p in fit["checkpoints"]], [p["rating"] for p in fit["checkpoints"]],
                color=COLORS[0], lw=2, marker="o", label="checkpoint")
        if fit["material"] is not None:
            ax.axhline(fit["material"], color=COLORS[1], lw=2, ls="--", label="material-capture baseline")
        ax.axhline(0, color=COLORS[2], lw=2, ls=":", label="random play (anchor)")
        _style(ax, "Strength of checkpoints (Elo-style, random = 0)", "self-play games", "rating")
        ax.legend(frameon=False)
        _save(fig, out_dir / "strength.png")
        charts.append(("strength.png", "Elo-style rating of checkpoints during training"))
    return charts


def _final_evals(evals: list[dict]) -> tuple[str | None, list[dict]]:
    """Evaluation rows of the last checkpoint that was evaluated."""
    names = [e["checkpoint"] for e in evals if e["checkpoint"] != "material"]
    if not names:
        return None, []
    last = names[-1]
    rows = [e for e in evals if e["checkpoint"] == last]
    # keep the most recent match per opponent
    latest = {}
    for e in rows:
        latest[e["opponent"]] = e
    return last, list(latest.values())


def results_markdown(run_dir: Path, h: dict, charts, image_dir: str, hardware: dict, hardware_note: str,
                     exported: dict | None = None) -> str:
    cfg = read_json(run_dir / "config.json", {}) or {}
    state = read_json(run_dir / "state.json", {}) or {}
    sessions = h["sessions"]
    elapsed = state.get("elapsed", 0.0)
    games, positions, steps = state.get("games", 0), state.get("positions", 0), state.get("steps", 0)
    lines = [START, "", f"Generated by `python -m chessml report` from run `{run_dir.name}`.", ""]

    lines += ["### The run", "", "| | |", "|---|---|"]
    lines.append(f"| Hardware | {hardware['cpu']}, {hardware['logical_cpus']} logical CPUs, "
                 f"{hardware['ram_gb']} GB RAM, GPU: {hardware['gpu']} |")
    lines.append(f"| Software | {hardware['os']}, Python {hardware['python']}, torch {hardware['torch']} |")
    lines.append(f"| Preset | `{cfg.get('preset')}`: {cfg.get('workers')} self-play workers, {cfg.get('threads')} "
                 f"trainer threads, {cfg.get('sims')} simulations/move, batch {cfg.get('batch_size')}, "
                 f"nice {cfg.get('nice')} |")
    lines.append(f"| Network | {cfg.get('blocks')} residual blocks x {cfg.get('channels')} channels |")
    starts = [s["detail"] for s in sessions if s["event"] == "start"]
    lines.append(f"| Training sessions | {len(starts)}: " + "; ".join(starts) + " |")
    lines.append(f"| Training time | {format_duration(elapsed)} |")
    lines.append(f"| Self-play games | {games:,} ({games / elapsed:.2f} games/s on average) |" if elapsed else
                 f"| Self-play games | {games:,} |")
    lines.append(f"| Positions generated | {positions:,} |")
    lines.append(f"| Training steps | {steps:,} |")
    if h["system"]:
        evals_ps = [r["evals_per_sec"] for r in h["system"] if r["evals_per_sec"]]
        if evals_ps:
            lines.append(f"| Network evaluations/s (all workers, median) | {sorted(evals_ps)[len(evals_ps) // 2]:,.0f} |")
    if h["train"]:
        last = h["train"][-1]
        first = h["train"][0]
        lines.append(f"| Loss, first -> last logged | total {first['total_loss']:.3f} -> {last['total_loss']:.3f} "
                     f"(policy {first['policy_loss']:.3f} -> {last['policy_loss']:.3f}, "
                     f"value {first['value_loss']:.3f} -> {last['value_loss']:.3f}) |")
    if h["games"]:
        tail = h["games"][-100:]
        n = len(tail)
        term = {}
        for g in tail:
            term[g["termination"]] = term.get(g["termination"], 0) + 1
        lines.append(f"| Last {n} self-play games | " + ", ".join(f"{k} {v}" for k, v in
                                                                   sorted(term.items(), key=lambda kv: -kv[1]))
                     + f"; average length {sum(g['plies'] for g in tail) / n:.0f} plies |")
        engine = [g for g in h["games"] if g.get("opponent") == "engine"]
        if engine:
            last = engine[-50:]
            score = 100 * sum(g["net_score"] for g in last) / len(last)
            lines.append(f"| Games against Stockfish | {100 * len(engine) / len(h['games']):.0f}% of games; level "
                         f"{engine[0]['engine_level']} -> {engine[-1]['engine_level']}; network score over the "
                         f"last {len(last)}: {score:.0f}% |")
    if hardware_note:
        lines += ["", hardware_note]
    lines.append("")

    evals = h["evals"]
    ckpt, final = _final_evals(evals)
    if ckpt:
        lines += [f"### Strength of the final checkpoint (`{ckpt}`)", "",
                  "| Opponent | Games | Wins | Draws | Losses | Score |", "|---|---|---|---|---|---|"]
        for e in final:
            n = e["wins"] + e["draws"] + e["losses"]
            lines.append(f"| {e['opponent']} | {n} | {e['wins']} | {e['draws']} | {e['losses']} | "
                         f"{100 * (e['wins'] + e['draws'] / 2) / n:.1f}% |")
        sims = {e["sims"] for e in final if e["sims"]}
        lines.append("")
        lines.append(f"Simulations per move in evaluation: {', '.join(map(str, sorted(sims))) or 'n/a'}.")
        ratings = compute_ratings([(e["checkpoint"], e["opponent"], e["wins"], e["draws"], e["losses"]) for e in evals])
        lines += ["", "### Elo-style ratings (all evaluation games of the run, random play = 0)", "",
                  "| Player | Rating |", "|---|---|"]
        for name, rating in sorted(ratings.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"| {name} | {rating:.0f} |")
        lines.append("")
        if exported:
            lines += [f"The bundled `models/best.pt` is `{exported['source']}`, the best-rated checkpoint "
                      f"({exported['rating']:.0f}), saved without optimiser state "
                      f"({exported['size_mb']:.1f} MB).", ""]
    else:
        lines += ["No evaluations recorded for this run yet.", ""]

    if charts:
        lines += ["### Charts", ""]
        for file, alt in charts:
            lines.append(f"![{alt}]({image_dir}/{file})")
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def export_best_checkpoint(run_dir: Path, h: dict) -> dict | None:
    """Copy the best-rated checkpoint (weights only) to models/best.pt."""
    from .network import load_checkpoint, model_from_checkpoint, save_checkpoint

    evals = h["evals"]
    ratings = compute_ratings([(e["checkpoint"], e["opponent"], e["wins"], e["draws"], e["losses"]) for e in evals])
    candidates = [(r, n) for n, r in ratings.items() if (run_dir / "checkpoints" / n).exists()]
    if not candidates:
        return None
    rating, name = max(candidates)
    src = run_dir / "checkpoints" / name
    data = load_checkpoint(str(src))
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / "best.pt"
    save_checkpoint(str(dest), model_from_checkpoint(data), games=data.get("games"), steps=data.get("steps"),
                    elapsed=data.get("elapsed"), run=run_dir.name, source=name, rating=rating)
    return {"path": dest, "source": name, "rating": rating, "size_mb": dest.stat().st_size / 1e6}


def generate_report(run: str | None, readme: Path, out_dir: Path, export_best: bool = False) -> None:
    if run:
        run_dir = resolve_run(run)
    else:
        runs = sorted(list_runs(), key=lambda r: r["games"], reverse=True)
        if not runs:
            sys.exit("No runs found. Train first: python -m chessml train")
        run_dir = Path(runs[0]["path"])
    h = read_history(run_dir, limit=5000)
    hardware = read_json(run_dir / "hardware.json")
    note = ""
    if not hardware:
        hardware = machine_info()
        note = "_Hardware details describe the machine that generated this report (the run did not record its own)._"
    exported = export_best_checkpoint(run_dir, h) if export_best else None
    charts = make_charts(h, out_dir)
    try:
        image_dir = out_dir.resolve().relative_to(readme.resolve().parent).as_posix()
    except ValueError:
        image_dir = out_dir.as_posix()
    section = results_markdown(run_dir, h, charts, image_dir, hardware, note, exported)
    text = readme.read_text() if readme.exists() else f"# ChessML\n\n## Results\n\n{START}\n{END}\n"
    if START in text and END in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: section, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n## Results\n\n" + section + "\n"
    readme.write_text(text)
    print(f"Updated {readme} from run {run_dir.name}; charts in {out_dir}")
    if exported:
        print(f"Exported {exported['source']} to {exported['path']} ({exported['size_mb']:.1f} MB)")
    elif export_best:
        print("No rated checkpoint to export")
