"""Run directories and checkpoint discovery.

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
        out.append({
            "name": path.name,
            "path": str(path),
            "status": state.get("status", "unknown"),
            "games": state.get("games", 0),
            "steps": state.get("steps", 0),
            "elapsed": state.get("elapsed", 0.0),
            "preset": state.get("preset"),
            "checkpoints": len(run_checkpoints(path)),
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
