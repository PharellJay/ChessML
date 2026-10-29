# ChessML

A chess engine that learns by playing against itself, built to train on a
CPU-only machine. A local web interface lets you start and control training,
watch it live, and play against any checkpoint.

- **AlphaZero-style learning:** a residual policy/value network in PyTorch,
  guided by Monte Carlo tree search (MCTS). It knows only the rules, which
  come from [python-chess](https://python-chess.readthedocs.io/).
- **Training control:** set the length by number of games, by wall-clock time,
  or "until I press Stop". You can pause, resume and stop gracefully, and
  resume from any run or checkpoint.
- **Intensity presets** (Light / Balanced / Heavy / Hardcore) that scale with
  your core count, plus manual overrides.
- **Live dashboard** over WebSocket: losses, throughput, CPU per core, outcomes,
  game length, Elo-style strength, and the self-play game in progress with
  the network's move probabilities and value estimate.
- **Play mode:** a clickable board, a choice of colour, checkpoint and strength,
  undo, flip, PGN export and optional engine analysis.

## How it works

```
             +--------------------- trainer process ----------------------+
             |  replay buffer (last 100k positions)                       |
 games  ---> |  AdamW on  policy cross-entropy + value MSE                |
 (planes,    |  publishes new weights every 25 steps  ------------------+ |
  visit      |  checkpoint every 10 min, evaluation every 15 min        | |
  counts,    +----------------------------------------------------------|-+
  result)          ^                                                     |
                   |                                                     v
       +-----------+-----------+   ... one per worker ...   +-----------------------+
       | self-play worker 0    |                            | self-play worker N-1  |
       | 16 games at once,     |                            |                       |
       | batched MCTS, 64 sims |                            |                       |
       +-----------------------+                            +-----------------------+

 web server (FastAPI, not niced)  <-- events (mp.Queue) --  trainer
        |  WebSocket push
        v
 browser: controls, live charts, live board, play mode
```

1. **Encode** the position from the side to move's view: 20 planes of 8x8 for
   pieces, castling rights, en passant, the 50-move counter and repetition.
2. **Search.** Each worker plays 16 games at once. Every MCTS round picks one
   leaf per game, and the network scores all 16 leaves in a single batch. That
   gives policy priors over the 4,672 AlphaZero move slots and a value in
   [-1, 1]. The search selects moves by PUCT, adds Dirichlet noise at the root,
   and reuses the tree between moves.
3. **Play.** Moves are sampled from visit counts for the first 20 plies, then
   the most-visited move is played. A game ends by checkmate, stalemate,
   insufficient material, the 50-move rule or threefold repetition. At 200
   plies it is adjudicated: a lead of at least 3 pawns of material wins,
   anything less is a draw.
4. **Learn.** Every position becomes a training sample: the visit
   distribution is the policy target, and the game result (from that
   position's side to move) is the value target. The trainer samples each
   position about 4 times, then sends the new weights to the workers.
5. **Measure.** A separate process regularly plays the newest checkpoint
   against three opponents: a random mover, a greedy material-capture
   baseline, and the previously evaluated checkpoint. Ratings are fitted on
   an Elo scale with random play anchored at 0.

### Sizing for a CPU

- **Network: 4 residual blocks x 48 channels, 265,994 parameters.** On a CPU,
  each search step pays twice: once for Python tree work (move generation,
  encoding, selection) and once for the network call. At this size the
  network call costs about as much as the tree work around it, so neither
  dominates. I measured this once on the run machine (one worker, one thread,
  16 parallel games, 64 simulations per move):

  | Network | Parameters | Network evaluations/s |
  |---|---|---|
  | 4 x 32 | 157,818 | 2,554 |
  | **4 x 48** | **265,994** | **1,810** |
  | 5 x 64 | 489,626 | 1,239 |
  | 6 x 64 | 563,610 | 1,079 |
  | 8 x 96 | 1,504,442 | 479 |
  | 10 x 128 | 3,201,498 | 264 |

  With hours of CPU rather than months of TPU, more games beat more
  capacity. 4x48 costs about 30% of throughput against 4x32, for 70% more
  parameters.
- **Search: 64 simulations per move, 16 games batched per worker.** That is
  enough search for the visit counts to improve on the raw policy, while
  keeping games short to produce. Batching across games (instead of virtual
  loss inside one tree) keeps every search exact and amortises the per-call
  overhead of PyTorch.
- **Move limit with adjudication.** An untrained network almost never
  delivers mate, so with only game results to learn from the value head
  would see mostly draws. Scoring long games by material gives it a learning
  signal from the first minutes. Checkmates take over as the network gets
  stronger.

## Install

Requires Python 3.14 (also tested in a 3.14.6 virtual environment) and works
on CPU only.

```bash
git clone https://github.com/PharellJay/ChessML.git
cd ChessML
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 1. PyTorch, CPU-only build: a much smaller download without the CUDA libraries
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

# 2. Everything else (pinned versions)
pip install -r requirements.txt

# 3. Check the install
python -m pytest -q
```

If you skip step 1, `pip install -r requirements.txt` installs the default
PyPI build of torch. That build bundles CUDA, also runs fine on a CPU-only
machine, and is the one used for the results below, but it downloads
several GB.

The web interface needs no build step and no internet access: chessboard.js,
chess.js, jQuery and the piece images are vendored in `chessml/static/vendor/`.

## Quick start

```bash
python -m chessml serve            # opens http://127.0.0.1:8000/
```

In the browser, pick a preset, pick a duration and press **Start**. Charts and
the live board update as training runs. The **Play** tab plays against any
checkpoint, including the bundled `models/best.pt`, right after cloning.

The same from the terminal:

```bash
python -m chessml train --preset balanced --time 2h      # wall-clock limit
python -m chessml train --preset heavy --games 2000      # game limit
python -m chessml train --preset light                   # until Ctrl+C
python -m chessml train --resume run-20261029-101500     # continue a run
python -m chessml train --resume runs/x/checkpoints/ckpt_000500g.pt   # new run from a checkpoint
python -m chessml evaluate run-20261029-101500 --games 40
python -m chessml list                                   # presets, runs, checkpoints
python -m chessml report --run run-20261029-101500       # regenerate Results below
```

## Training modes and intensity presets

**Duration.** Set exactly one of `--games N`, `--time 2h` (also `90m`,
`1h30m`, `45s`), or nothing, which means until stopped. Limits apply to the
current session: resuming a run with `--games 500` plays 500 more. Paused
time does not count against a time limit. While a limited run is going, the
dashboard shows its ETA.

**Stopping.** Stop, the UI button, or Ctrl+C in the terminal running `train`
or `serve`, ends training gracefully: games still in progress are discarded,
games already finished are kept, and a checkpoint and the replay buffer are
saved. A second Ctrl+C quits immediately. Pause freezes self-play,
training and any running evaluation, and resume continues where it left off.

**Resuming.** `--resume <run>` continues that run in place from its latest
checkpoint: weights, optimiser, counters, replay buffer, and metrics appended
to the same charts. `--resume <file.pt>` starts a new run initialised from
that checkpoint. The UI's "Start from" menu offers both.

**Intensity.** Presets are computed from the number of logical CPUs:

| Preset | Self-play workers | Trainer threads | Sims/move | Batch | Niceness | On 4 CPUs | On 12 CPUs |
|---|---|---|---|---|---|---|---|
| Light | cpus / 6 | 1 | 64 | 128 | 19 | 1 + 1 | 2 + 1 |
| Balanced | cpus / 3 | 2 | 64 | 256 | 10 | 1 + 2 | 4 + 2 |
| Heavy | 2 x cpus / 3 | 2 | 64 | 256 | 8 | 2 + 2 | 8 + 2 |
| Hardcore | cpus - 2 | 2 | 64 | 256 | 5 | 2 + 2 | 10 + 2 |

The last two columns are workers + trainer threads. Each worker keeps one
core busy, so Hardcore uses every core: 10 workers plus 2 trainer threads on
a 12-core machine, plus a short-lived evaluation process every 15 minutes.
Override any value with `--workers`, `--threads`, `--sims`, `--batch-size`
and `--nice`, or in the UI under *Manual overrides*. `python -m chessml list`
prints the presets for your machine.

**Why the UI stays responsive.** Trainer, workers and evaluator are separate
processes that lower their own priority (niceness 5 to 19). The web server
stays at normal priority and only relays events, so the kernel schedules it
ahead of training even when every core is busy.

## UI tour

**Train tab**

![Train tab](docs/ui-train.png)

- *Training* (left): preset with its resolved values, manual overrides,
  duration (until stop / N games / time), "Start from" (new run, continue a
  run, or new run from a checkpoint), and Start / Pause / Resume / Stop.
- *Status*: state, run, preset, elapsed time, ETA, games played, games/s,
  positions, network evaluations/s, training steps, replay buffer size and
  current checkpoint.
- *Live self-play game*: the game worker 0 is playing, with the network's
  value estimate (raw and after search) and its top moves (prior
  probability, and share of search visits).
- *Metrics*: charts for loss (policy, value, total), games/s, overall CPU,
  CPU per core, win/draw/loss and average game length over the last 100
  games, Elo-style rating per evaluated checkpoint, and evaluation scores,
  plus an evaluation table. The **Run** selector shows the history of any
  past run: metrics are stored per run in SQLite (`runs/<run>/metrics.db`),
  so charts survive restarts.

**Play tab**

![Play tab](docs/ui-play.png)

Click a piece, then its target square. Legal targets are highlighted, and
promotions ask which piece you want. You choose your colour, the checkpoint
to play against, and its strength in MCTS simulations per move. **Undo**
takes back your last move together with the engine's reply. **Export PGN**
shows the game for copying or downloading. *Show engine evaluation* adds the
search value and the top candidate moves (visits, share, value), and outlines
the best candidate on the board.

## Results

The table below states the machine and preset of the run it describes.
Rerun the commands in [Reproducing the results](#reproducing-the-results) to
regenerate this section on your own machine.

<!-- results:start -->
<!-- results:end -->

## Reproducing the results

```bash
# 1. Train: Hardcore preset, 58 minutes of training (under 60 minutes of wall clock in total)
python -m chessml train --preset hardcore --time 58m --eval-minutes 10 --seed 1 --name hardcore-60m

# 2. Final evaluation of the last checkpoint: 40 games each against random, the
#    material-capture baseline and the untrained starting network
python -m chessml evaluate hardcore-60m --games 40 --sims 64 \
    --opponents random,material,previous --previous first --seed 1

# 3. Regenerate the Results section and charts, and export the best checkpoint
python -m chessml report --run hardcore-60m --export-best
```

Self-play runs in several processes, so two runs never play identical games,
and throughput depends on the CPU. Expect the same trend, not the same numbers.

## Project layout

```
chessml/
  encoding.py    board planes, move <-> policy index (4,672 AlphaZero move slots)
  network.py     residual policy/value network, checkpoint save/load
  mcts.py        batched PUCT search with tree reuse
  selfplay.py    self-play worker processes, game rules, adjudication
  trainer.py     replay buffer, optimisation, checkpoints, evaluation, graceful stop
  evaluation.py  random and material-capture baselines, matches, Elo fitting
  metrics.py     per-run SQLite metrics
  runs.py        run directories and checkpoint discovery
  server.py      FastAPI app, WebSocket events, play endpoint
  report.py      charts and the Results section of this README
  cli.py         python -m chessml {train,evaluate,serve,report,list}
  static/        web UI (plain HTML/CSS/JS) and vendored libraries
models/best.pt   best checkpoint of the reported run (weights only)
tests/           pytest suite
```

## Limitations

- **The strength ceiling is low.** AlphaZero trained for millions of games
  with 800 simulations per move on thousands of TPUs. An hour on a few CPU
  cores gives a few thousand games with a 266k-parameter network. Expect an
  engine that captures material and mates weak opponents, not one that
  plays sound chess. Longer runs keep improving, but slowly.
- **Material adjudication is domain knowledge.** It speeds up early learning,
  but it also teaches the network to value material the way the adjudication
  rule does. Set `--max-plies` high to rely on real game results instead.
- **Ratings are relative.** They come from short matches (tens of games)
  against random play, a one-ply material-capture player and earlier
  checkpoints, with random play anchored at 0. They are not comparable to
  FIDE or online ratings, and they carry wide error bars.
- **No history planes.** The input encodes only the current position plus a
  repetition flag, unlike AlphaZero's 8-position history.
- **Python search.** The tree search is pure Python, so it scales with
  processes, not threads. On a big machine, more workers help more than more
  threads per worker.
