"""Training configuration, intensity presets and duration parsing."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, fields

CPU_COUNT = os.cpu_count() or 4

PRESET_NAMES = ("light", "balanced", "heavy", "hardcore")


def preset_values(name: str, cores: int = CPU_COUNT) -> dict:
    """Resolve an intensity preset for a machine with `cores` logical CPUs.

    Self-play workers use one core each; the trainer uses `threads` cores.
    Workers and trainer run niced so the web server always gets CPU first.
    """
    name = name.lower()
    if name == "light":
        return dict(workers=max(1, cores // 6), threads=1, sims=64, batch_size=128, nice=19)
    if name == "balanced":
        return dict(workers=max(1, cores // 3), threads=2, sims=64, batch_size=256, nice=10)
    if name == "heavy":
        return dict(workers=max(1, (2 * cores) // 3), threads=2, sims=64, batch_size=256, nice=8)
    if name == "hardcore":
        return dict(workers=max(1, cores - 2), threads=2, sims=64, batch_size=256, nice=5)
    raise ValueError(f"unknown preset {name!r}; choose from {', '.join(PRESET_NAMES)}")


def apply_niceness(target: int) -> None:
    """Set this process's niceness to `target` (it can only be raised)."""
    try:
        current = os.nice(0)
        if target > current:
            os.nice(target - current)
    except (OSError, AttributeError):
        pass


def all_presets(cores: int = CPU_COUNT) -> dict:
    return {name: preset_values(name, cores) for name in PRESET_NAMES}


_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([hms]?)", re.I)


def parse_duration(text: str) -> float:
    """Parse '2h', '90m', '1h30m', '45s' or a bare number of minutes into seconds."""
    text = text.strip().lower()
    if not text:
        raise ValueError("empty duration")
    total, pos = 0.0, 0
    for m in _DURATION_RE.finditer(text):
        if text[pos:m.start()].strip():
            raise ValueError(f"cannot parse duration {text!r}")
        value, unit = float(m.group(1)), m.group(2) or "m"
        total += value * {"h": 3600, "m": 60, "s": 1}[unit]
        pos = m.end()
    if text[pos:].strip() or total <= 0:
        raise ValueError(f"cannot parse duration {text!r}")
    return total


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


@dataclass
class TrainConfig:
    # intensity
    preset: str = "balanced"
    workers: int = 2
    threads: int = 2
    sims: int = 64
    batch_size: int = 256
    nice: int = 10
    parallel_games: int = 16  # games searched together per worker (network batch size)

    # duration: at most one of games / seconds; neither means "until stopped"
    max_games: int | None = None
    max_seconds: float | None = None

    # network
    blocks: int = 4
    channels: int = 48

    # self-play
    max_plies: int = 200  # longer games are adjudicated
    adjudicate_material: int = 3  # material lead (pawn units) that wins an adjudicated game
    temperature_plies: int = 20  # sample moves by visit count for this many plies
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25

    # opponent: share of games played against a UCI engine (Stockfish) instead of self-play
    engine_share: float = 0.0  # 0 = self-play only, 1 = engine games only
    engine_path: str | None = None  # default: $CHESSML_ENGINE or stockfish on PATH
    engine_time: float = 0.02  # seconds of engine search per move
    engine_multipv: int = 5  # candidate moves per engine search (imitation target)
    engine_level: int = 0  # strength 0-20: starting level, or the fixed level if not adaptive
    engine_adaptive: bool = True  # move the level with the network's recent score
    engine_window: int = 20  # engine games per adaptive decision
    engine_imitate: bool = True  # also train on the engine's positions with its moves as target

    # optimisation
    lr: float = 1e-3
    weight_decay: float = 1e-4
    buffer_size: int = 100_000
    min_buffer: int = 1_000
    sample_reuse: float = 4.0  # times each position is (on average) used for training
    publish_every: int = 25  # training steps between weight updates sent to workers

    # bookkeeping
    checkpoint_minutes: float = 10.0
    eval_minutes: float = 15.0  # 0 disables periodic evaluation
    eval_games: int = 10  # per opponent
    eval_sims: int = 64
    seed: int | None = None

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> "TrainConfig":
        values = preset_values(preset)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(preset=preset.lower(), **values)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def duration_text(self) -> str:
        if self.max_games:
            return f"{self.max_games} games"
        if self.max_seconds:
            return f"{format_duration(self.max_seconds)} wall-clock"
        return "until stopped"

    def opponent_text(self) -> str:
        if self.engine_share <= 0:
            return "self-play"
        level = f"adaptive, from level {self.engine_level}" if self.engine_adaptive else f"level {self.engine_level}"
        return f"{100 * self.engine_share:.0f}% vs Stockfish ({level})"
