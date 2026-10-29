"""Web interface: FastAPI app with REST endpoints and a WebSocket for live updates.

Training runs in a separate process (trainer.training_process_main) whose
events are read by a background thread and pushed to every connected
browser. The server process itself is not niced, so it stays responsive
while the niced training processes use the remaining CPU.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import queue
import threading
import time
import webbrowser
from functools import lru_cache
from pathlib import Path

import chess
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .config import CPU_COUNT, TrainConfig, all_presets, parse_duration
from .metrics import read_history
from .runs import list_checkpoints, list_runs, resolve_checkpoint, resolve_run

STATIC_DIR = Path(__file__).resolve().parent / "static"


class TrainingManager:
    """Owns the training process and fans its events out to WebSocket clients."""

    def __init__(self):
        self.ctx = mp.get_context("spawn")
        self.proc = None
        self.control = None
        self.events = None
        self.status: dict = {"type": "status", "state": "idle"}
        self.live: dict | None = None
        self.clients: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def start(self, cfg: TrainConfig, resume: str | None, name: str | None) -> None:
        from .trainer import Control, training_process_main

        with self.lock:
            if self.running:
                raise RuntimeError("training is already running")
            self.control = Control(self.ctx)
            self.events = self.ctx.Queue()
            self.live = None
            self.status = {"type": "status", "state": "starting", "preset": cfg.preset}
            self.proc = self.ctx.Process(target=training_process_main, name="chessml-trainer",
                                         args=(cfg.to_dict(), resume, name, self.control, self.events))
            self.proc.start()
            threading.Thread(target=self._read_events, args=(self.proc, self.events), daemon=True).start()
        self.broadcast(self.status)

    def pause(self) -> None:
        if self.running:
            self.control.pause.set()

    def resume(self) -> None:
        if self.running:
            self.control.pause.clear()

    def stop(self) -> None:
        if self.running:
            self.control.stop.set()

    def shutdown(self, timeout: float = 120.0) -> None:
        """Stop training gracefully (checkpoint included) when the server exits."""
        if self.running:
            print("Stopping training gracefully before exit...", flush=True)
            self.control.stop.set()
            self.proc.join(timeout)
            if self.proc.is_alive():
                self.proc.terminate()

    def _read_events(self, proc, events) -> None:
        while True:
            try:
                event = events.get(timeout=0.5)
            except queue.Empty:
                if not proc.is_alive():
                    break
                continue
            kind = event.get("type")
            if kind == "status":
                self.status = event
            elif kind == "live":
                self.live = event
            self.broadcast(event)
        if self.status.get("state") not in ("stopped", "completed", "error"):
            self.status = {**self.status, "state": "error" if proc.exitcode else "stopped"}
            self.broadcast(self.status)

    def broadcast(self, event: dict) -> None:
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(self._fan_out, event)

    def _fan_out(self, event: dict) -> None:
        for q in list(self.clients):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # a slow client misses an update rather than stalling the others


manager = TrainingManager()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    manager.loop = asyncio.get_running_loop()
    yield
    await run_in_threadpool(manager.shutdown)


app = FastAPI(title="ChessML", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# -- training -----------------------------------------------------------------


class StartRequest(BaseModel):
    preset: str = "balanced"
    workers: int | None = None
    threads: int | None = None
    sims: int | None = None
    batch_size: int | None = None
    nice: int | None = None
    duration: str = "stop"  # "games", "time" or "stop"
    games: int | None = None
    time: str | None = None  # e.g. "2h", "90m"
    resume: str | None = None  # run name or checkpoint path
    name: str | None = None
    eval_minutes: float | None = None
    checkpoint_minutes: float | None = None


@app.get("/api/info")
def info():
    return {"cpu_count": CPU_COUNT, "presets": all_presets(), "defaults": TrainConfig().to_dict(),
            "status": manager.status, "live": manager.live}


@app.post("/api/training/start")
def start_training(req: StartRequest):
    try:
        max_games = max_seconds = None
        if req.duration == "games":
            if not req.games or req.games < 1:
                raise ValueError("number of games must be at least 1")
            max_games = req.games
        elif req.duration == "time":
            max_seconds = parse_duration(req.time or "")
        cfg = TrainConfig.from_preset(
            req.preset, workers=req.workers, threads=req.threads, sims=req.sims,
            batch_size=req.batch_size, nice=req.nice, max_games=max_games, max_seconds=max_seconds,
            eval_minutes=req.eval_minutes, checkpoint_minutes=req.checkpoint_minutes,
        )
        for field in ("workers", "threads", "sims", "batch_size"):
            if getattr(cfg, field) < 1:
                raise ValueError(f"{field} must be at least 1")
        if not 0 <= cfg.nice <= 19:
            raise ValueError("niceness must be between 0 and 19")
        resume = req.resume or None
        if resume and not Path(resume).is_file():
            resume = str(resolve_run(resume))
        manager.start(cfg, resume, req.name or None)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return manager.status


@app.post("/api/training/pause")
def pause_training():
    manager.pause()
    return {"ok": manager.running}


@app.post("/api/training/resume")
def resume_training():
    manager.resume()
    return {"ok": manager.running}


@app.post("/api/training/stop")
def stop_training():
    manager.stop()
    return {"ok": manager.running}


@app.get("/api/training/status")
def training_status():
    return {"status": manager.status, "live": manager.live, "running": manager.running}


@app.get("/api/runs")
def runs():
    return list_runs()


@app.get("/api/runs/{name}/history")
def run_history(name: str, limit: int = 2000):
    try:
        return read_history(resolve_run(name), limit)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/checkpoints")
def checkpoints():
    return list_checkpoints()


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    manager.clients.add(q)
    try:
        await ws.send_json({"type": "hello", "status": manager.status, "live": manager.live})
        while True:
            await ws.send_json(await q.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        manager.clients.discard(q)


# -- play ---------------------------------------------------------------------


class PlayRequest(BaseModel):
    moves: list[str] = []  # UCI moves from the start position
    checkpoint: str
    sims: int = 200


_search_lock = threading.Lock()


@lru_cache(maxsize=4)
def _engine(path: str, mtime: float):
    from .mcts import MCTS
    from .network import Evaluator, model_from_checkpoint

    return MCTS(Evaluator(model_from_checkpoint(path)))


def _board_from(moves: list[str]) -> chess.Board:
    board = chess.Board()
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {uci} in position {board.fen()}")
        board.push(move)
    return board


def _analyse(req: PlayRequest) -> dict:
    import torch

    from .mcts import SearchTree

    if not 1 <= req.sims <= 20000:
        raise ValueError("simulations must be between 1 and 20000")
    board = _board_from(req.moves)
    if board.is_game_over(claim_draw=True):
        return {"game_over": True, "result": board.result(claim_draw=True), "fen": board.fen()}
    path = resolve_checkpoint(req.checkpoint)
    with _search_lock:  # one search at a time keeps the server's CPU use bounded
        torch.set_num_threads(2)
        mcts = _engine(str(path), path.stat().st_mtime)
        tree = SearchTree(board)
        start = time.time()
        mcts.run([tree], req.sims)
        move = mcts.choose_move(tree)
        elapsed = time.time() - start
    sign = 1.0 if board.turn == chess.WHITE else -1.0
    return {
        "game_over": False,
        "move": move.uci(),
        "san": board.san(move),
        "value_white": sign * tree.root.q(),
        "net_value_white": sign * tree.root.value,
        "top": tree.top_moves(5),
        "sims": tree.root.visits(),
        "seconds": elapsed,
        "fen": board.fen(),
    }


@app.post("/api/play/move")
async def engine_move(req: PlayRequest):
    """The engine's move (and its analysis) in the position after `moves`."""
    try:
        return await run_in_threadpool(_analyse, req)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    import uvicorn

    url = f"http://{host}:{port}/"
    print(f"ChessML web interface: {url}  (Ctrl+C stops training gracefully and exits)", flush=True)
    if open_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning", timeout_graceful_shutdown=3)
