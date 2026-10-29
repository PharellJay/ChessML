"""Per-run metrics in SQLite, so charts survive restarts.

`elapsed` is cumulative training time of the run in seconds (paused time
excluded), so curves from resumed sessions continue where they left off.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS train (
    ts REAL, elapsed REAL, step INTEGER, games INTEGER,
    policy_loss REAL, value_loss REAL, total_loss REAL, lr REAL);
CREATE TABLE IF NOT EXISTS games (
    ts REAL, elapsed REAL, game INTEGER, result TEXT, plies INTEGER,
    termination TEXT, worker INTEGER);
CREATE TABLE IF NOT EXISTS system (
    ts REAL, elapsed REAL, games INTEGER, games_per_sec REAL, positions_per_sec REAL,
    evals_per_sec REAL, cpu_total REAL, cpu_per_core TEXT, mem_percent REAL);
CREATE TABLE IF NOT EXISTS evals (
    ts REAL, elapsed REAL, games INTEGER, step INTEGER, checkpoint TEXT, opponent TEXT,
    wins INTEGER, draws INTEGER, losses INTEGER, sims INTEGER);
CREATE TABLE IF NOT EXISTS ratings (
    ts REAL, elapsed REAL, games INTEGER, step INTEGER, checkpoint TEXT, rating REAL,
    material_rating REAL);
CREATE TABLE IF NOT EXISTS sessions (
    ts REAL, elapsed REAL, event TEXT, detail TEXT);
"""

TABLE_COLUMNS = {
    "train": ("ts", "elapsed", "step", "games", "policy_loss", "value_loss", "total_loss", "lr"),
    "games": ("ts", "elapsed", "game", "result", "plies", "termination", "worker"),
    "system": ("ts", "elapsed", "games", "games_per_sec", "positions_per_sec", "evals_per_sec",
               "cpu_total", "cpu_per_core", "mem_percent"),
    "evals": ("ts", "elapsed", "games", "step", "checkpoint", "opponent", "wins", "draws", "losses", "sims"),
    "ratings": ("ts", "elapsed", "games", "step", "checkpoint", "rating", "material_rating"),
    "sessions": ("ts", "elapsed", "event", "detail"),
}


class MetricsDB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def insert(self, table: str, row: dict) -> None:
        cols = TABLE_COLUMNS[table]
        row = dict(row)
        row.setdefault("ts", time.time())
        if table == "system" and not isinstance(row.get("cpu_per_core"), str):
            row["cpu_per_core"] = json.dumps(row.get("cpu_per_core") or [])
        values = [row.get(c) for c in cols]
        self.conn.execute(
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", values
        )
        self.conn.commit()

    def insert_many(self, table: str, rows: list[dict]) -> None:
        for row in rows:
            self.insert(table, row)

    def fetch(self, table: str, limit: int | None = None) -> list[dict]:
        cols = TABLE_COLUMNS[table]
        sql = f"SELECT {','.join(cols)} FROM {table} ORDER BY rowid"
        rows = [dict(zip(cols, r)) for r in self.conn.execute(sql)]
        if table == "system":
            for r in rows:
                r["cpu_per_core"] = json.loads(r["cpu_per_core"] or "[]")
        if limit and len(rows) > limit:  # thin long histories evenly, keep the last row
            stride = len(rows) / limit
            rows = [rows[int(i * stride)] for i in range(limit - 1)] + [rows[-1]]
        return rows

    def history(self, limit: int = 2000) -> dict:
        return {t: self.fetch(t, None if t in ("evals", "ratings", "sessions") else limit) for t in TABLE_COLUMNS}

    def close(self) -> None:
        self.conn.close()


def read_history(run_dir: str | Path, limit: int = 2000) -> dict:
    path = Path(run_dir) / "metrics.db"
    if not path.exists():
        return {t: [] for t in TABLE_COLUMNS}
    db = MetricsDB(path)
    try:
        return db.history(limit)
    finally:
        db.close()
