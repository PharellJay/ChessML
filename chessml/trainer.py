"""Training orchestration: self-play workers, replay buffer, optimisation,
checkpoints, periodic evaluation and graceful stop.

Processes (all started with the "spawn" method):
    trainer       this module; owns the network, optimiser and replay buffer
    workers x N   self-play (selfplay.worker_main), one core each; with an
                  engine share, each also drives its own Stockfish process
    evaluator     occasional strength test of the newest checkpoint
The controller (CLI or web server) talks to the trainer through `Control`
(pause/stop events) and receives events through the `emit` callback.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import time
from collections import deque
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from .config import TrainConfig, apply_niceness
from .encoding import NUM_PLANES, POLICY_SIZE, planes_to_float
from .engine import EngineLadder, find_engine
from .evaluation import compute_ratings, eval_process_main
from .metrics import MetricsDB
from .network import ChessNet, NetConfig, load_checkpoint, save_checkpoint
from .runs import latest_checkpoint, new_run_dir, resolve_run, run_checkpoints, write_json
from .selfplay import worker_main


class Control:
    """Pause/stop flags shared by the controller and the training processes."""

    def __init__(self, ctx=None):
        ctx = ctx or mp.get_context("spawn")
        self.pause = ctx.Event()
        self.stop = ctx.Event()


class ReplayBuffer:
    """Ring buffer of (planes, sparse policy target, outcome) samples."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.planes = np.zeros((capacity, NUM_PLANES, 8, 8), dtype=np.uint8)
        self.z = np.zeros(capacity, dtype=np.float32)
        self.pol_idx: list = [None] * capacity
        self.pol_p: list = [None] * capacity
        self.size = 0
        self.pos = 0

    def add(self, planes, pol_idx, pol_p, z) -> None:
        for i in range(len(z)):
            self.planes[self.pos] = planes[i]
            self.z[self.pos] = z[i]
            self.pol_idx[self.pos] = pol_idx[i]
            self.pol_p[self.pos] = pol_p[i]
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int, rng: np.random.Generator):
        ids = rng.integers(0, self.size, size=n)
        target = np.zeros((n, POLICY_SIZE), dtype=np.float32)
        for row, i in enumerate(ids):
            target[row, self.pol_idx[i]] = self.pol_p[i]
        return self.planes[ids], target, self.z[ids]

    def _chronological(self) -> np.ndarray:
        if self.size < self.capacity:
            return np.arange(self.size)
        return (np.arange(self.capacity) + self.pos) % self.capacity

    def save(self, path: Path) -> None:
        order = self._chronological()
        if len(order) == 0:
            return
        tmp = Path(str(path) + ".tmp.npz")
        np.savez_compressed(
            tmp,
            planes=self.planes[order],
            z=self.z[order],
            lens=np.array([len(self.pol_idx[i]) for i in order]),
            pol_idx=np.concatenate([self.pol_idx[i] for i in order]),
            pol_p=np.concatenate([self.pol_p[i] for i in order]),
        )
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        data = np.load(path)
        splits = np.cumsum(data["lens"])[:-1]
        self.add(data["planes"], np.split(data["pol_idx"], splits), np.split(data["pol_p"], splits), data["z"])


class Trainer:
    def __init__(self, cfg: TrainConfig, control: Control, emit=None,
                 resume: str | None = None, run_name: str | None = None):
        self.cfg = cfg
        self.control = control
        self.emit = emit or (lambda event: None)
        self.resume = resume
        self.run_name = run_name
        self.rng = np.random.default_rng(cfg.seed)

    # -- setup ------------------------------------------------------------

    def _setup(self) -> None:
        cfg = self.cfg
        self.buffer = ReplayBuffer(cfg.buffer_size)
        ckpt_path = None
        if self.resume:
            target = Path(self.resume)
            if target.is_file():  # fork a new run from a specific checkpoint
                ckpt_path = target.resolve()
                parent = ckpt_path.parent.parent.name if ckpt_path.parent.name == "checkpoints" else "model"
                self.run_dir = new_run_dir(self.run_name or f"{parent}-from-{ckpt_path.stem}")
            else:  # continue a run in place from its latest checkpoint
                self.run_dir = resolve_run(self.resume)
                ckpt_path = latest_checkpoint(self.run_dir)
                if (self.run_dir / "buffer.npz").exists():
                    self.buffer.load(self.run_dir / "buffer.npz")
        else:
            self.run_dir = new_run_dir(self.run_name)

        if cfg.engine_share > 0:
            cfg.engine_path = find_engine(cfg.engine_path)
        data = load_checkpoint(str(ckpt_path)) if ckpt_path else None
        if data:
            cfg.blocks, cfg.channels = data["net"]["blocks"], data["net"]["channels"]
            if cfg.engine_adaptive and data.get("engine_level") is not None:
                cfg.engine_level = data["engine_level"]  # the ladder continues where it was
        self.ladder = EngineLadder(cfg.engine_level, cfg.engine_adaptive, cfg.engine_window)
        self.model = ChessNet(NetConfig(cfg.blocks, cfg.channels))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.games = self.steps = self.positions = 0
        self.elapsed_base = 0.0
        self.active_seconds = 0.0
        self.checkpoint = None
        if data:
            self.model.load_state_dict(data["model"])
            if "optimizer" in data:
                self.optimizer.load_state_dict(data["optimizer"])
                for group in self.optimizer.param_groups:
                    group["lr"], group["weight_decay"] = cfg.lr, cfg.weight_decay
            self.games = data.get("games", 0)
            self.steps = data.get("steps", 0)
            self.positions = data.get("positions", 0)
            self.elapsed_base = data.get("elapsed", 0.0)
            self.checkpoint = str(ckpt_path)
        self.db = MetricsDB(self.run_dir / "metrics.db")
        write_json(self.run_dir / "config.json", cfg.to_dict())
        from .report import machine_info

        write_json(self.run_dir / "hardware.json", machine_info())
        evals = self.db.fetch("evals")
        self.last_evaluated = next((e["checkpoint"] for e in reversed(evals) if e["checkpoint"] != "material"), None)
        self.have_material_reference = any(e["checkpoint"] == "material" for e in evals)

    # -- main loop --------------------------------------------------------

    def run(self) -> str:
        """Train until a limit is reached or stop is requested. Returns the final state."""
        cfg = self.cfg
        self._setup()
        torch.set_num_threads(cfg.threads)
        apply_niceness(cfg.nice)
        psutil.cpu_percent(percpu=True)  # prime the CPU counters

        ctx = mp.get_context("spawn")
        self.weights_path = self.run_dir / "weights.pt"
        save_checkpoint(str(self.weights_path), self.model)
        self.version = ctx.Value("i", 0)
        self.engine_level = ctx.Value("i", self.ladder.level)
        self.counters = ctx.Array("q", cfg.workers)
        self.game_queue = ctx.Queue()
        self.live_queue = ctx.Queue(maxsize=8)
        self.eval_queue = ctx.Queue()
        self.worker_stop = ctx.Event()
        self.eval_proc = None
        self.eval_meta = None
        self.eval_suspended = False

        if self.games == 0 and not run_checkpoints(self.run_dir):
            self._save_checkpoint()  # untrained reference for "earlier checkpoint" evaluation

        seed = cfg.seed if cfg.seed is not None else int.from_bytes(os.urandom(4), "little")
        self.workers = [
            ctx.Process(
                target=worker_main, daemon=True, name=f"chessml-selfplay-{i}",
                args=(i, cfg.to_dict(), str(self.weights_path), self.version, self.engine_level, self.counters,
                      self.game_queue, self.live_queue, self.control.pause, self.worker_stop, seed + i),
            )
            for i in range(cfg.workers)
        ]
        for w in self.workers:
            w.start()

        self.ctx = ctx
        self.session_games = 0
        self.session_positions = 0
        self.trained_samples = 0
        self.recent_games: deque = deque()
        self.loss_acc = [0.0, 0.0, 0.0, 0]
        self.last_rates = (time.time(), self.positions, 0)
        self.rates = {"games_per_sec": 0.0, "positions_per_sec": 0.0, "evals_per_sec": 0.0}
        self.db.insert("sessions", {"elapsed": self.elapsed_base, "event": "start",
                                    "detail": f"preset={cfg.preset} workers={cfg.workers} {cfg.duration_text()}"
                                              + ("" if cfg.engine_share <= 0 else f", {cfg.opponent_text()}")})
        self.emit({"type": "log", "message": f"Run {self.run_dir.name} started: {cfg.preset} preset, "
                                            f"{cfg.workers} workers, {cfg.duration_text()}, {cfg.opponent_text()}"})

        now = time.time()
        last_tick = last_status = last_ckpt = last_eval = now
        reason = "error"
        state = "running"
        try:
            while True:
                now = time.time()
                paused = self.control.pause.is_set()
                if not paused:
                    self.active_seconds += now - last_tick
                last_tick = now
                if paused != (state == "paused"):
                    state = "paused" if paused else "running"
                    self._suspend_eval(paused)
                    self.emit({"type": "log", "message": "Training paused" if paused else "Training resumed"})
                    last_status = 0
                if self.control.stop.is_set():
                    reason = "stopped"
                    break
                if cfg.max_games and self.session_games >= cfg.max_games:
                    reason = "completed"
                    break
                if cfg.max_seconds and self.active_seconds >= cfg.max_seconds:
                    reason = "completed"
                    break

                got = self._drain_games()
                self._drain_live()
                self._poll_eval()
                trained = 0 if paused else self._train_some(budget=0.5)

                if now - last_status >= 1.0:
                    self._system_metrics(now)
                    self._emit_status(state)
                    last_status = now
                if not paused and now - last_ckpt >= cfg.checkpoint_minutes * 60:
                    self._save_checkpoint()
                    last_ckpt = now
                if (not paused and cfg.eval_minutes > 0 and self.eval_proc is None
                        and now - last_eval >= cfg.eval_minutes * 60):
                    self._start_eval()
                    last_eval = now
                if not got and not trained:
                    time.sleep(0.05)
        finally:
            self._shutdown(reason)
        return reason

    # -- self-play data ---------------------------------------------------

    def _drain_games(self, limit: int = 64) -> int:
        n = 0
        while n < limit:
            try:
                msg = self.game_queue.get_nowait()
            except queue.Empty:
                break
            n += 1
            if self.cfg.max_games and self.session_games >= self.cfg.max_games:
                continue  # finished after the limit was reached: not counted
            self._add_game(msg)
        return n

    def _add_game(self, msg: dict) -> None:
        self.buffer.add(msg["planes"], msg["pol_idx"], msg["pol_p"], msg["z"])
        n = len(msg["z"])
        self.games += 1
        self.session_games += 1
        self.positions += n
        self.session_positions += n
        self.recent_games.append(time.time())
        row = {"elapsed": self._elapsed(), "game": self.games, "result": msg["result"],
               "plies": msg["plies"], "termination": msg["termination"], "worker": msg["worker"],
               "opponent": msg["opponent"], "engine_level": msg["engine_level"], "net_score": msg["net_score"]}
        self.db.insert("games", row)
        self.emit({"type": "game", **row})
        if msg["opponent"] == "engine":
            self._update_engine_level(msg["net_score"], msg["engine_level"])

    def _update_engine_level(self, net_score: float, level: int) -> None:
        new = self.ladder.record(net_score, level)
        if new is None:
            return
        self.engine_level.value = new
        self.emit({"type": "log", "message": f"Stockfish level {'raised' if new > level else 'lowered'} to {new}: "
                                            f"the network scored {100 * self.ladder.decided_score:.0f}% over its "
                                            f"last {self.ladder.window} games at level {level}"})

    def _drain_live(self) -> None:
        latest = None
        while True:
            try:
                latest = self.live_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self.emit(latest)

    # -- optimisation -----------------------------------------------------

    def _train_some(self, budget: float) -> int:
        cfg = self.cfg
        if self.buffer.size < min(cfg.min_buffer, cfg.buffer_size):
            return 0
        steps, start = 0, time.time()
        while (self.trained_samples < self.session_positions * cfg.sample_reuse
               and time.time() - start < budget):
            self._train_step()
            self.trained_samples += cfg.batch_size
            steps += 1
        return steps

    def _train_step(self) -> None:
        planes, target_p, target_z = self.buffer.sample(self.cfg.batch_size, self.rng)
        self.model.train()
        logits, value = self.model(torch.from_numpy(planes_to_float(planes)))
        policy_loss = -(torch.from_numpy(target_p) * F.log_softmax(logits, dim=1)).sum(1).mean()
        value_loss = F.mse_loss(value, torch.from_numpy(target_z))
        loss = policy_loss + value_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.steps += 1

        acc = self.loss_acc
        acc[0] += policy_loss.item()
        acc[1] += value_loss.item()
        acc[2] += loss.item()
        acc[3] += 1
        if acc[3] >= 10:
            self._log_losses()
        if self.steps % self.cfg.publish_every == 0:
            self._publish_weights()

    def _log_losses(self) -> None:
        acc = self.loss_acc
        if not acc[3]:
            return
        row = {"elapsed": self._elapsed(), "step": self.steps, "games": self.games,
               "policy_loss": acc[0] / acc[3], "value_loss": acc[1] / acc[3],
               "total_loss": acc[2] / acc[3], "lr": self.optimizer.param_groups[0]["lr"]}
        self.db.insert("train", row)
        self.emit({"type": "train", **row})
        self.loss_acc = [0.0, 0.0, 0.0, 0]

    def _publish_weights(self) -> None:
        self.model.eval()
        save_checkpoint(str(self.weights_path), self.model)
        with self.version.get_lock():
            self.version.value += 1

    # -- checkpoints and evaluation ----------------------------------------

    def _elapsed(self) -> float:
        return self.elapsed_base + self.active_seconds

    def _save_checkpoint(self) -> Path:
        path = self.run_dir / "checkpoints" / f"ckpt_{self.games:06d}g.pt"
        self.model.eval()
        save_checkpoint(str(path), self.model, self.optimizer, games=self.games, steps=self.steps,
                        positions=self.positions, elapsed=self._elapsed(), run=self.run_dir.name,
                        config=self.cfg.to_dict(), created=time.time(), engine_level=self.ladder.level)
        self.checkpoint = str(path)
        self._write_state("running")
        self.emit({"type": "checkpoint", "path": str(path), "name": path.name, "games": self.games})
        return path

    def _write_state(self, status: str) -> None:
        write_json(self.run_dir / "state.json", {
            "status": status, "games": self.games, "steps": self.steps, "positions": self.positions,
            "elapsed": self._elapsed(), "checkpoint": self.checkpoint, "preset": self.cfg.preset,
            "engine_level": self.ladder.level if self.cfg.engine_share > 0 else None, "updated": time.time(),
        })

    def _start_eval(self) -> None:
        cfg = self.cfg
        ckpt = self._save_checkpoint()
        opponents = ["random", "material"]
        previous = self._previous_checkpoint(ckpt)
        if previous:
            opponents.append(str(previous))
        self.eval_meta = {"checkpoint": ckpt.name, "games": self.games, "steps": self.steps,
                          "elapsed": self._elapsed()}
        self.eval_proc = self.ctx.Process(
            target=eval_process_main, daemon=True, name="chessml-eval",
            args=(str(ckpt), opponents, cfg.eval_games, cfg.eval_sims, 300, cfg.adjudicate_material,
                  not self.have_material_reference, cfg.nice, self.eval_queue, ckpt.name),
        )
        self.eval_proc.start()
        self.emit({"type": "log", "message": f"Evaluating {ckpt.name} vs {', '.join(Path(o).name for o in opponents)}"})

    def _previous_checkpoint(self, current: Path) -> Path | None:
        ckpts = [p for p in run_checkpoints(self.run_dir) if p != current]
        if self.last_evaluated:
            match = [p for p in ckpts if p.name == self.last_evaluated]
            if match:
                return match[0]
        return ckpts[0] if ckpts else None  # the untrained starting point

    def _suspend_eval(self, suspend: bool) -> None:
        if self.eval_proc is None or not self.eval_proc.is_alive() or suspend == self.eval_suspended:
            return
        try:
            proc = psutil.Process(self.eval_proc.pid)
            proc.suspend() if suspend else proc.resume()
            self.eval_suspended = suspend
        except psutil.Error:
            pass

    def _poll_eval(self) -> None:
        if self.eval_proc is None:
            return
        try:
            result = self.eval_queue.get_nowait()
        except queue.Empty:
            if not self.eval_proc.is_alive() and self.eval_proc.exitcode not in (None, 0):
                self.emit({"type": "log", "message": f"Evaluation process exited with code {self.eval_proc.exitcode}"})
                self.eval_proc = None
            return
        self.eval_proc.join(timeout=5)
        self.eval_proc = None
        meta = self.eval_meta
        if "error" in result:
            self.emit({"type": "log", "message": f"Evaluation failed: {result['error']}"})
            return
        for rec in result["records"]:
            self.db.insert("evals", {
                "elapsed": meta["elapsed"], "games": meta["games"], "step": meta["steps"],
                "checkpoint": rec["player"], "opponent": rec["opponent"], "wins": rec["wins"],
                "draws": rec["draws"], "losses": rec["losses"], "sims": self.cfg.eval_sims,
            })
            if rec["player"] == "material":
                self.have_material_reference = True
        ratings = compute_ratings([(e["checkpoint"], e["opponent"], e["wins"], e["draws"], e["losses"])
                                   for e in self.db.fetch("evals")])
        row = {"elapsed": meta["elapsed"], "games": meta["games"], "step": meta["steps"],
               "checkpoint": meta["checkpoint"], "rating": ratings.get(meta["checkpoint"], 0.0),
               "material_rating": ratings.get("material")}
        self.db.insert("ratings", row)
        self.last_evaluated = meta["checkpoint"]
        self.emit({"type": "eval", "records": result["records"], **row})
        summary = ", ".join(f"vs {r['opponent']} +{r['wins']}={r['draws']}-{r['losses']}"
                            for r in result["records"] if r["player"] != "material")
        self.emit({"type": "log", "message": f"Eval {meta['checkpoint']}: {summary}; rating {row['rating']:.0f}"})

    # -- status -----------------------------------------------------------

    def _system_metrics(self, now: float) -> None:
        t0, pos0, ev0 = self.last_rates
        evals = sum(self.counters[:])
        dt = max(1e-6, now - t0)
        while self.recent_games and now - self.recent_games[0] > 60:
            self.recent_games.popleft()
        window = min(60.0, max(1.0, self.active_seconds))
        self.rates = {
            "games_per_sec": len(self.recent_games) / window,
            "positions_per_sec": (self.positions - pos0) / dt,
            "evals_per_sec": (evals - ev0) / dt,
        }
        self.last_rates = (now, self.positions, evals)
        per_core = psutil.cpu_percent(percpu=True)
        row = {"elapsed": self._elapsed(), "games": self.games, **self.rates,
               "cpu_total": sum(per_core) / max(1, len(per_core)), "cpu_per_core": per_core,
               "mem_percent": psutil.virtual_memory().percent}
        self.db.insert("system", row)
        self.emit({"type": "system", **row})

    def _eta(self) -> float | None:
        cfg = self.cfg
        if cfg.max_seconds:
            return max(0.0, cfg.max_seconds - self.active_seconds)
        if cfg.max_games:
            rate = self.rates["games_per_sec"] or (self.session_games / max(1.0, self.active_seconds))
            remaining = cfg.max_games - self.session_games
            return remaining / rate if rate > 0 else None
        return None

    def _emit_status(self, state: str) -> None:
        cfg = self.cfg
        self.emit({
            "type": "status", "state": state, "run": self.run_dir.name, "preset": cfg.preset,
            "workers": cfg.workers, "threads": cfg.threads, "sims": cfg.sims, "batch_size": cfg.batch_size,
            "nice": cfg.nice, "duration": cfg.duration_text(), "max_games": cfg.max_games,
            "max_seconds": cfg.max_seconds, "elapsed": self._elapsed(), "session_elapsed": self.active_seconds,
            "eta": self._eta(), "games": self.games, "session_games": self.session_games,
            "positions": self.positions, "steps": self.steps, "buffer": self.buffer.size,
            "checkpoint": Path(self.checkpoint).name if self.checkpoint else None,
            "evaluating": self.eval_proc is not None, "opponent": cfg.opponent_text(),
            "engine_share": cfg.engine_share, "engine_level": self.ladder.level, "engine_score": self.ladder.score(),
            "engine_window": cfg.engine_window, **self.rates,
        })

    # -- shutdown ---------------------------------------------------------

    def _shutdown(self, reason: str) -> None:
        self._emit_status("stopping")
        self.emit({"type": "log", "message": "Stopping: discarding unfinished games and saving a checkpoint"})
        self.worker_stop.set()
        self.control.pause.clear()
        if self.eval_proc is not None:
            self._suspend_eval(False)
            self.eval_proc.terminate()
            self.eval_proc.join(timeout=5)
            self.eval_proc = None
        deadline = time.time() + 20
        while any(w.is_alive() for w in self.workers) and time.time() < deadline:
            self._drain_games()  # games completed before the stop still count
            self._drain_live()
            time.sleep(0.05)
        for w in self.workers:
            if w.is_alive():
                w.terminate()
            w.join(timeout=5)
        self._drain_games()
        if reason == "completed":  # train on the last games before the final checkpoint
            self._train_some(budget=30.0)
        self._log_losses()
        self._save_checkpoint()
        self.buffer.save(self.run_dir / "buffer.npz")
        final = "completed" if reason == "completed" else "stopped"
        self._write_state(final)
        self.db.insert("sessions", {"elapsed": self._elapsed(), "event": final,
                                    "detail": f"games={self.games} steps={self.steps}"})
        self.db.close()
        self._emit_status(final)
        self.emit({"type": "log", "message": f"Run {self.run_dir.name} {final}: {self.games} games, "
                                            f"{self.steps} training steps, checkpoint {Path(self.checkpoint).name}"})


def training_process_main(cfg_dict: dict, resume, run_name, control: Control, event_queue) -> None:
    """Entry point when training runs in a child process of the web server."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the server stops us gracefully
    trainer = Trainer(TrainConfig.from_dict(cfg_dict), control, event_queue.put, resume, run_name)
    try:
        trainer.run()
    except Exception as exc:
        event_queue.put({"type": "log", "message": f"Training failed: {exc!r}"})
        event_queue.put({"type": "status", "state": "error", "error": repr(exc)})
        raise
