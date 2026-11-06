"""Run directories ("machines" in the web interface) and checkpoint discovery.

runs/<name>/
    config.json        training configuration of the latest session
    state.json         status, counters and the latest checkpoint
    metrics.db         SQLite metrics (losses, games, system, evaluations)
    weights.pt         newest weights, read by self-play workers
    buffer.npz         replay buffer saved on stop (used when resuming)
    checkpoints/       ckpt_<games>g.pt files
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_DIR.parent / "models"


def runs_dir() -> Path:
    return Path(os.environ.get("CHESSML_RUNS", "runs")).resolve()


def new_run_dir(name: str | None = None) -> Path:
    base = runs_dir()
    base.mkdir(parents=True, exist_ok=True)
    name = name or time.strftime("run-%Y%m%d-%H%M%S")
    path, n = base / name, 1
    while path.exists():
        n += 1
        path = base / f"{name}-{n}"
    (path / "checkpoints").mkdir(parents=True)
    return path


RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,59}")
BOOTSTRAP_EPOCHS = 4  # passes over the source buffer; four brought an 8x96 net close to its 4x48 source
MAX_BLOCKS, MAX_CHANNELS = 40, 512


def create_run(name: str, from_checkpoint: str | Path | None = None, blocks: int | None = None,
               channels: int | None = None, bootstrap: str | None = None,
               bootstrap_epochs: int = BOOTSTRAP_EPOCHS) -> Path:
    """Create a named run, empty or starting from a copy of a checkpoint.

    The copy becomes the run's first checkpoint, so training the run
    continues from it exactly as it would continue from its own checkpoint.

    An untrained run can instead get its own network size (`blocks` x
    `channels`) and a `bootstrap` source: another run whose saved replay
    buffer is copied in. Its first training session then pre-trains the new
    network on those positions before self-play starts (Trainer._pretrain).
    """
    name = name.strip()
    if not RUN_NAME.fullmatch(name):
        raise ValueError("use 1-60 letters, digits, spaces, dots, dashes or underscores, starting with a letter or digit")
    path = runs_dir() / name
    if path.exists():
        raise FileExistsError(f"a machine named {name!r} already exists")
    if from_checkpoint and (blocks or channels or bootstrap):
        raise ValueError("a machine started from a checkpoint keeps that checkpoint's network size and training")
    if (blocks is None) != (channels is None):
        raise ValueError("give both the number of blocks and the channels per block")
    if blocks is not None and not (1 <= blocks <= MAX_BLOCKS and 8 <= channels <= MAX_CHANNELS):
        raise ValueError(f"network size must be 1-{MAX_BLOCKS} blocks of 8-{MAX_CHANNELS} channels")
    source_buffer = None
    if bootstrap:
        bootstrap_run = resolve_run(bootstrap)
        source_buffer = bootstrap_run / "buffer.npz"
        if not source_buffer.is_file():
            raise FileNotFoundError(f"machine {bootstrap_run.name!r} has no saved replay buffer yet "
                                    "(it is written when a training session stops)")
        if not 1 <= bootstrap_epochs <= 50:
            raise ValueError("pre-training passes must be between 1 and 50")
    (path / "checkpoints").mkdir(parents=True)
    state = {"status": "new", "games": 0, "steps": 0, "positions": 0, "elapsed": 0.0, "checkpoint": None,
             "updated": time.time()}
    if blocks is not None:
        state["net"] = {"blocks": blocks, "channels": channels}
    if source_buffer is not None:
        import numpy as np

        shutil.copyfile(source_buffer, path / "buffer.npz")  # loaded like the run's own buffer when it starts
        with np.load(path / "buffer.npz") as data:
            positions = int(data["z"].shape[0])
        state["bootstrap"] = {"source": bootstrap_run.name, "positions": positions, "epochs": bootstrap_epochs,
                              "steps_done": 0, "done": False}
    if from_checkpoint:
        from .network import load_checkpoint

        source = Path(from_checkpoint)
        data = load_checkpoint(str(source))
        games = data.get("games") or 0
        dest = path / "checkpoints" / f"ckpt_{games:06d}g.pt"
        shutil.copyfile(source, dest)
        state.update(games=games, steps=data.get("steps") or 0, positions=data.get("positions") or 0,
                     elapsed=data.get("elapsed") or 0.0, checkpoint=str(dest), origin=str(source.resolve()))
    write_json(path / "state.json", state)
    return path


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path: Path, data) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def resolve_run(name_or_path: str) -> Path:
    """Find a run directory by name or path."""
    path = Path(name_or_path)
    for candidate in (path, runs_dir() / name_or_path):
        if candidate.is_dir() and (candidate / "checkpoints").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"no run named {name_or_path!r} in {runs_dir()}")


def run_checkpoints(run: Path) -> list[Path]:
    return sorted(Path(run, "checkpoints").glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)


def latest_checkpoint(run: Path) -> Path | None:
    state = read_json(Path(run) / "state.json", {}) or {}
    if state.get("checkpoint") and Path(state["checkpoint"]).exists():
        return Path(state["checkpoint"])
    ckpts = run_checkpoints(run)
    return ckpts[-1] if ckpts else None


def list_runs() -> list[dict]:
    base = runs_dir()
    if not base.is_dir():
        return []
    out = []
    for path in sorted(base.iterdir()):
        if not (path / "checkpoints").is_dir():
            continue
        state = read_json(path / "state.json", {}) or {}
        config = read_json(path / "config.json", {}) or {}
        net = state.get("net") or ({"blocks": config["blocks"], "channels": config["channels"]}
                                   if "blocks" in config and "channels" in config else None)
        out.append({
            "name": path.name,
            "path": str(path),
            "status": state.get("status", "unknown"),
            "games": state.get("games", 0),
            "steps": state.get("steps", 0),
            "elapsed": state.get("elapsed", 0.0),
            "preset": state.get("preset"),
            "checkpoints": len(run_checkpoints(path)),
            "origin": state.get("origin"),
            "net": net,  # None until the first session of a run created without a size
            "bootstrap": state.get("bootstrap"),
            "has_buffer": (path / "buffer.npz").is_file(),
            "updated": path.stat().st_mtime,
        })
    out.sort(key=lambda r: r["updated"], reverse=True)
    return out


def list_checkpoints() -> list[dict]:
    """Every checkpoint that can be played against: bundled models first, then runs."""
    out = []
    if MODELS_DIR.is_dir():
        for p in sorted(MODELS_DIR.glob("*.pt")):
            out.append({"label": f"models/{p.name}", "path": str(p), "run": None})
    for run in list_runs():
        for p in reversed(run_checkpoints(Path(run["path"]))):
            out.append({"label": f"{run['name']}/{p.name}", "path": str(p), "run": run["name"]})
    return out


def resolve_checkpoint(spec: str) -> Path:
    """A checkpoint file, a run name/dir (-> its latest checkpoint) or a label from list_checkpoints."""
    path = Path(spec)
    if path.is_file():
        return path.resolve()
    for item in list_checkpoints():
        if item["label"] == spec:
            return Path(item["path"])
    run = resolve_run(spec)
    ckpt = latest_checkpoint(run)
    if ckpt is None:
        raise FileNotFoundError(f"run {spec!r} has no checkpoints")
    return ckpt
