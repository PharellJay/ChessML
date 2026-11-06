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
from .elo import SCOUT_GAMES, WINDOW, calibration_summary, load_calibration, load_results
from .engine import check_engine_config, engine_available, find_engine
from .metrics import read_history
from .runs import create_run, list_checkpoints, list_runs, resolve_checkpoint, resolve_run

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
        self.grid: dict | None = None
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
            self.live = self.grid = None
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

    def focus(self, worker: int, slot: int) -> None:
        """Show this game (worker, slot) in the live view's detail panel."""
        workers = self.status.get("workers") or 0
        slots = self.status.get("parallel_games") or 0
        if not (0 <= worker < workers and 0 <= slot < slots):
            raise ValueError(f"no game {worker}/{slot} (training has {workers} workers x {slots} games)")
        if self.running:
            self.control.focus[0], self.control.focus[1] = worker, slot

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
            elif kind == "grid":
                self.grid = event
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


class StrengthManager:
    """Runs one Elo job (calibration or measurement) at a time in a child process.

    Its events reach the browsers through the training manager's WebSocket fan-out.
    """

    def __init__(self):
        self.ctx = mp.get_context("spawn")
        self.proc = None
        self.stop_event = None
        self.job: dict | None = None  # latest state of the current or last job
        self.lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def start(self, kind: str, params: dict) -> dict:
        from .elo import job_main

        with self.lock:
            if self.running:
                raise RuntimeError(f"a {self.job['kind']} job is already running")
            self.stop_event = self.ctx.Event()
            events = self.ctx.Queue()
            self.job = {"kind": kind, "state": "starting", "params": params, "stage": None,
                        "done": 0, "total": 0, "matches": [], "started": time.time()}
            # not a daemon: the job starts a pool of match processes itself
            self.proc = self.ctx.Process(target=job_main, name=f"chessml-{kind}",
                                         args=(kind, params, events, self.stop_event))
            self.proc.start()
            threading.Thread(target=self._read_events, args=(self.proc, events), daemon=True).start()
        return self.job

    def cancel(self) -> None:
        if self.running:
            self.stop_event.set()

    def shutdown(self, timeout: float = 20.0) -> None:
        if self.running:
            self.stop_event.set()
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
            self._apply(event)
            manager.broadcast(event)
        if self.job and self.job["state"] in ("starting", "running"):
            self.job["state"] = "error"
            self.job["message"] = f"job process exited with code {proc.exitcode}"
            manager.broadcast({"type": "strength", "kind": self.job["kind"], "event": "error",
                               "message": self.job["message"]})

    def _apply(self, event: dict) -> None:
        job = self.job
        kind = event["event"]
        if kind == "boards":
            job["boards"] = event["games"]
        elif kind == "started":
            job["state"] = "running"
        elif kind == "stage":
            job.update(stage=event["stage"], stage_info=event, done=0,
                       total=len(event["opponents"]) if "opponents" in event else event.get("matches", 0))
        elif kind == "match":
            job.update(done=event["done"], total=event["total"])
            job["matches"].append({**event["record"], "stage": job["stage"]})
        elif kind in ("completed", "cancelled", "error"):
            job["state"] = kind
            job["result"] = event.get("result")
            job["message"] = event.get("message")


strength = StrengthManager()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    manager.loop = asyncio.get_running_loop()
    yield
    await run_in_threadpool(strength.shutdown)
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
    buffer_size: int | None = None
    duration: str = "stop"  # "games", "time" or "stop"
    games: int | None = None
    time: str | None = None  # e.g. "2h", "90m"
    resume: str | None = None  # run name or checkpoint path
    name: str | None = None
    eval_minutes: float | None = None
    checkpoint_minutes: float | None = None
    engine_share: float | None = None  # 0 or empty = self-play only
    engine_level: int | None = None
    engine_adaptive: bool | None = None
    engine_path: str | None = None
    reset_lr: bool | None = None  # undo earlier learning-rate drops for this session
    full_search_prob: float | None = None  # share of moves with the full search (the training samples)
    eval_games: int | None = None  # evaluation games per opponent


@app.get("/api/info")
def info():
    return {"cpu_count": CPU_COUNT, "presets": all_presets(), "defaults": TrainConfig().to_dict(),
            "engine": engine_available(), "calibrated": load_calibration() is not None,
            "status": manager.status, "live": manager.live, "grid": manager.grid}


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
            batch_size=req.batch_size, nice=req.nice, buffer_size=req.buffer_size, max_games=max_games,
            max_seconds=max_seconds,
            eval_minutes=req.eval_minutes, checkpoint_minutes=req.checkpoint_minutes,
            engine_share=req.engine_share, engine_level=req.engine_level, engine_adaptive=req.engine_adaptive,
            engine_path=req.engine_path or None, reset_lr=req.reset_lr or None,
            full_search_prob=req.full_search_prob, eval_games=req.eval_games,
        )
        for field in ("workers", "threads", "sims", "batch_size", "eval_games"):
            if getattr(cfg, field) < 1:
                raise ValueError(f"{field} must be at least 1")
        if not 0 < cfg.full_search_prob <= 1:
            raise ValueError("full-search share must be above 0 and at most 1")
        if not 1_000 <= cfg.buffer_size <= 5_000_000:
            raise ValueError("replay buffer must hold between 1,000 and 5,000,000 positions")
        if not 0 <= cfg.nice <= 19:
            raise ValueError("niceness must be between 0 and 19")
        check_engine_config(cfg)
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
    return {"status": manager.status, "live": manager.live, "grid": manager.grid, "running": manager.running}


class FocusRequest(BaseModel):
    worker: int
    slot: int


@app.post("/api/training/focus")
def focus_game(req: FocusRequest):
    try:
        manager.focus(req.worker, req.slot)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": manager.running}


@app.get("/api/runs")
def runs():
    return list_runs()


class MachineRequest(BaseModel):
    name: str
    checkpoint: str | None = None  # start from a copy of this checkpoint; empty = untrained
    blocks: int | None = None  # network size of an untrained machine (default: the configured 4x48)
    channels: int | None = None
    pretrain_from: str | None = None  # machine whose replay buffer it pre-trains on before self-play
    pretrain_passes: int | None = None


@app.post("/api/machines")
def create_machine(req: MachineRequest):
    """A new machine (run directory), untrained or starting from a checkpoint."""
    try:
        source = resolve_checkpoint(req.checkpoint) if req.checkpoint else None
        extra = {} if req.pretrain_passes is None else {"bootstrap_epochs": req.pretrain_passes}
        path = create_run(req.name, source, blocks=req.blocks, channels=req.channels,
                          bootstrap=req.pretrain_from or None, **extra)
    except FileExistsError as exc:
        raise HTTPException(409, str(exc))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    return {"name": path.name, "path": str(path)}


@app.get("/api/runs/{name}/history")
def run_history(name: str, limit: int = 2000):
    from .evaluation import checkpoint_ratings

    try:
        history = read_history(resolve_run(name), limit)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    fit = checkpoint_ratings(history["evals"], load_calibration())
    history["fitted"] = {"checkpoints": fit["checkpoints"], "material": fit["material"],
                         "random": fit["all"].get("random", 0.0), "scale": fit["scale"]}
    return history


@app.get("/api/checkpoints")
def checkpoints():
    """Playable checkpoints, with the latest Elo measurement of each where there is one."""
    latest = {}
    for result in reversed(load_results(load_calibration())):  # oldest first, so the newest wins
        latest[str(Path(result["path"]).resolve())] = result
    out = []
    for item in list_checkpoints():
        result = latest.get(str(Path(item["path"]).resolve()))
        out.append({**item, "elo": result["elo"] if result else None,
                    "elo_error": result["error"] if result else None})
    return out


# -- strength (Elo) -------------------------------------------------------------


def _default_processes() -> int:
    free = CPU_COUNT - 2 if not manager.running else CPU_COUNT // 2
    return max(1, free)


@app.get("/api/strength")
def strength_state():
    calibration = load_calibration()
    return {"engine": engine_available(), "calibration": calibration_summary(calibration),
            "results": load_results(calibration), "job": strength.job, "running": strength.running,
            "training": manager.running, "cpu_count": CPU_COUNT, "default_processes": _default_processes(),
            "scout_games": SCOUT_GAMES, "window": WINDOW}


class CalibrateRequest(BaseModel):
    games: int = 20  # per match; with refine, the refinement round's total
    processes: int | None = None
    refine: bool = False  # add a refinement round to the existing calibration


class MeasureRequest(BaseModel):
    checkpoint: str
    games: int = 20
    sims: int = 64
    processes: int | None = None


def _check_processes(processes: int | None) -> int:
    processes = processes or _default_processes()
    if not 1 <= processes <= CPU_COUNT:
        raise ValueError(f"processes must be between 1 and {CPU_COUNT}")
    return processes


def _start_job(kind: str, params: dict):
    try:
        return strength.start(kind, params)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/strength/calibrate")
def start_calibration(req: CalibrateRequest):
    try:
        if req.refine:
            if not load_calibration():
                raise ValueError("there is no calibration to refine yet")
            if not 20 <= req.games <= 10_000:
                raise ValueError("refinement games must be between 20 and 10000")
        elif not 2 <= req.games <= 200:
            raise ValueError("games per match must be between 2 and 200")
        params = {"games": req.games, "refine": req.refine, "processes": _check_processes(req.processes),
                  "engine_path": find_engine(None), "nice": 10}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    return _start_job("calibrate", params)


@app.post("/api/strength/measure")
def start_measurement(req: MeasureRequest):
    from .elo import checkpoint_label, engine_setup

    try:
        calibration = load_calibration()
        if not calibration:
            raise ValueError("calibrate first: the Elo ladder has not been calibrated on this machine")
        if not 2 <= req.games <= 200:
            raise ValueError("games per opponent must be between 2 and 200")
        if not 1 <= req.sims <= 5000:
            raise ValueError("simulations must be between 1 and 5000")
        path = resolve_checkpoint(req.checkpoint)
        params = {"checkpoint": str(path), "label": checkpoint_label(path), "games": req.games, "sims": req.sims,
                  "processes": _check_processes(req.processes),
                  "engine_path": engine_setup(calibration)["engine_path"], "nice": 10}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    return _start_job("measure", params)


@app.post("/api/strength/cancel")
def cancel_strength_job():
    strength.cancel()
    return {"ok": strength.running}


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    manager.clients.add(q)
    try:
        await ws.send_json({"type": "hello", "status": manager.status, "live": manager.live, "grid": manager.grid})
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
