import { Chess } from "/static/vendor/chess-1.4.0.js";
import { BarChart, LineChart } from "/static/charts.js";
import { MiniBoards } from "/static/boards.js";

const $id = (id) => document.getElementById(id);
const PIECES = "/static/vendor/img/chesspieces/wikipedia/{piece}.png";
const ROLLING = 100; // games in the outcome / length window
const ENGINE_ROLLING = 50; // engine games in the score window

const state = { info: null, status: { state: "idle" }, chartRun: null, games: [], evals: [], runs: [], ckpts: null };

// ---------------------------------------------------------------- helpers

function fmtDuration(s) {
  if (s === null || s === undefined) return "–";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}
const fmtInt = (n) => (n === null || n === undefined ? "–" : Number(n).toLocaleString());
const fmtNum = (n, d = 2) => (n === null || n === undefined ? "–" : Number(n).toFixed(d));
const signed = (v) => (v >= 0 ? "+" : "") + v.toFixed(2);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmtDate = (t) => new Date(t * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
const shortPath = (p) => p.split("/").slice(-2).join("/");

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function log(message, ts = Date.now()) {
  const li = document.createElement("li");
  const t = new Date(ts).toLocaleTimeString();
  li.innerHTML = `<time>${t}</time>`;
  li.append(document.createTextNode(message));
  const list = $id("log");
  list.querySelector(".placeholder")?.remove();
  list.prepend(li);
  while (list.children.length > 200) list.lastChild.remove();
}

function setValueBar(bar, text, value, label) {
  const v = Math.max(-1, Math.min(1, value));
  bar.style.width = `${((v + 1) / 2) * 100}%`;
  text.textContent = label;
}

// Redraw a chessboard.js board only when its position changed: every redraw
// recreates all piece images, which flickers when it happens several times a second.
function showPosition(board, boardId, fen, lastMove) {
  const placement = fen.split(" ")[0];
  if (board.shown !== placement) {
    board.position(placement, false);
    board.shown = placement;
  }
  highlightSquares(boardId, "sq-last", lastMove ? [lastMove.slice(0, 2), lastMove.slice(2, 4)] : []);
}

function highlightSquares(boardId, cls, squares) {
  document.querySelectorAll(`#${boardId} .${cls}`).forEach((el) => el.classList.remove(cls));
  for (const sq of squares) document.querySelector(`#${boardId} .square-${sq}`)?.classList.add(cls);
}

// ---------------------------------------------------------------- tabs

document.querySelectorAll(".tab").forEach((btn) => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${btn.dataset.tab}`));
  liveBoard.resize();
  playBoard.resize();
  jobBoard.resize();
  renderPlay();
  if (btn.dataset.tab === "play") refreshCheckpoints();
  if (btn.dataset.tab === "strength") { refreshCheckpoints(); refreshStrength(); }
  history.replaceState(null, "", `#${btn.dataset.tab}`);  // reloads and bookmarks keep the tab
}));
window.addEventListener("resize", () => { liveBoard.resize(); playBoard.resize(); jobBoard.resize(); renderPlay(); });

// ---------------------------------------------------------------- charts

const minutes = (v) => (v < 10 ? v.toFixed(1) : v.toFixed(0));
const pct = (v) => `${v.toFixed(0)}%`;
const charts = {
  loss: new LineChart($id("chLoss"), { series: ["policy", "value", "total"], xLabel: "min", xFmt: minutes }),
  gps: new LineChart($id("chGps"), { series: ["games/sec"], xLabel: "min", xFmt: minutes, yMin: 0 }),
  cpu: new LineChart($id("chCpu"), { series: ["all cores"], xLabel: "min", xFmt: minutes, yMin: 0, yMax: 100, yFmt: pct }),
  cores: new BarChart($id("chCores"), { yMax: 100, yFmt: pct, labelFmt: (i) => `core ${i}` }),
  wdl: new LineChart($id("chWdl"), { series: ["White wins", "draws", "Black wins"], xLabel: "game", integerX: true, yMin: 0, yMax: 100, yFmt: pct }),
  len: new LineChart($id("chLen"), { series: ["plies"], xLabel: "game", integerX: true, yMin: 0 }),
  elo: new LineChart($id("chElo"), { series: ["checkpoint", "material baseline"], xLabel: "games", integerX: true, markers: true,
    emptyText: "No evaluation yet (runs every few minutes)" }),
  score: new LineChart($id("chScore"), { series: ["vs random", "vs material", "vs earlier checkpoint", "vs Stockfish (calibrated)"], xLabel: "games", integerX: true,
    yMin: 0, yMax: 100, yFmt: pct, markers: true, emptyText: "No evaluation yet" }),
  engScore: new LineChart($id("chEngScore"), { series: ["score"], xLabel: "game", integerX: true, yMin: 0, yMax: 100, yFmt: pct,
    emptyText: "No games against Stockfish in this run" }),
  engLevel: new LineChart($id("chEngLevel"), { series: ["level"], xLabel: "game", integerX: true, yMin: 0, yMax: 20,
    yFmt: (v) => v.toFixed(0), emptyText: "No games against Stockfish in this run" }),
};

function outcomePoints(games) {
  const w = [], d = [], b = [], len = [];
  let ww = 0, bw = 0, plies = 0;
  games.forEach((g, i) => {
    ww += g.result === "1-0"; bw += g.result === "0-1"; plies += g.plies;
    if (i >= ROLLING) {
      const old = games[i - ROLLING];
      ww -= old.result === "1-0"; bw -= old.result === "0-1"; plies -= old.plies;
    }
    const n = Math.min(i + 1, ROLLING);
    w.push([g.game, (100 * ww) / n]); b.push([g.game, (100 * bw) / n]);
    d.push([g.game, (100 * (n - ww - bw)) / n]); len.push([g.game, plies / n]);
  });
  return { w, d, b, len };
}

function engineGamePoints(games) {
  const score = [], level = [];
  const recent = [];
  let total = 0;
  for (const g of games) {
    if (g.opponent !== "engine") continue;
    recent.push(g.net_score); total += g.net_score;
    if (recent.length > ENGINE_ROLLING) total -= recent.shift();
    score.push([g.game, (100 * total) / recent.length]);
    level.push([g.game, g.engine_level]);
  }
  return { score, level };
}

function renderOutcomes() {
  const { w, d, b, len } = outcomePoints(state.games);
  charts.wdl.set(0, w); charts.wdl.set(1, d); charts.wdl.set(2, b);
  charts.len.set(0, len);
  const { score, level } = engineGamePoints(state.games);
  charts.engScore.set(0, score);
  charts.engLevel.set(0, level);
}

function opponentSlot(name) {
  if (name === "random") return 0;
  if (name === "material") return 1;
  if (name.startsWith("sf-")) return 3;
  return 2;
}

function renderEvals() {
  const series = [[], [], [], []];
  const rows = [];
  for (const e of state.evals) {
    if (e.checkpoint === "material") continue;
    const n = e.wins + e.draws + e.losses;
    const s = n ? (100 * (e.wins + e.draws / 2)) / n : null;
    series[opponentSlot(e.opponent)].push([e.games, s]);
  }
  series.forEach((pts, i) => charts.score.set(i, pts));
  for (const e of [...state.evals].reverse()) {
    const n = e.wins + e.draws + e.losses;
    rows.push(`<tr><td>${fmtInt(e.games)}</td><td>${e.checkpoint}</td><td>${e.opponent}</td><td>${e.wins}</td>` +
      `<td>${e.draws}</td><td>${e.losses}</td><td>${n ? ((100 * (e.wins + e.draws / 2)) / n).toFixed(0) + "%" : "–"}</td></tr>`);
  }
  $id("evalTable").innerHTML = rows.join("") || `<tr><td colspan="7">No evaluations yet</td></tr>`;
}

async function loadHistory(name) {
  state.chartRun = name;
  Object.values(charts).forEach((c) => (c.clear ? c.clear() : c.set([])));
  state.games = []; state.evals = [];
  renderEvals();
  if (!name) return;
  let h;
  try { h = await api(`/api/runs/${encodeURIComponent(name)}/history`); } catch (e) { log(`Could not load history: ${e.message}`); return; }
  if (state.chartRun !== name) return;
  const t = (r) => r.elapsed / 60;
  charts.loss.set(0, h.train.map((r) => [t(r), r.policy_loss]));
  charts.loss.set(1, h.train.map((r) => [t(r), r.value_loss]));
  charts.loss.set(2, h.train.map((r) => [t(r), r.total_loss]));
  charts.gps.set(0, h.system.map((r) => [t(r), r.games_per_sec]));
  charts.cpu.set(0, h.system.map((r) => [t(r), r.cpu_total]));
  if (h.system.length) charts.cores.set(h.system[h.system.length - 1].cpu_per_core);
  state.games = h.games;
  renderOutcomes();
  // one rating fit over all evaluation games of the run (same as the README)
  const fit = h.fitted.checkpoints;
  $id("eloCaption").textContent = h.fitted.scale === "elo"
    ? "Strength: Elo, calibrated against Stockfish's UCI_Elo"
    : "Strength: Elo-style rating (random play = 0) · python -m chessml calibrate gives real Elo";
  charts.elo.set(0, fit.map((p) => [p.games, p.rating]));
  const xs = fit.map((p) => p.games);
  charts.elo.set(1, h.fitted.material === null || !xs.length ? []
    : [[Math.min(...xs), h.fitted.material], [Math.max(...xs), h.fitted.material]]);
  state.evals = h.evals;
  renderEvals();
}

async function refreshRuns(select) {
  let runs = [];
  try { runs = await api("/api/runs"); } catch { return; }
  const chartSel = $id("chartRun");
  const current = select || state.chartRun || (runs[0] && runs[0].name);
  chartSel.innerHTML = runs.map((r) => `<option value="${r.name}">${r.name} (${r.status}, ${fmtInt(r.games)} games)</option>`).join("");
  if (current) chartSel.value = current;
  state.runs = runs;
  await refreshCheckpoints();  // every checkpoint list (machines, Play, Strength) from one fetch
  renderMachines();
  if (current && current !== state.chartRun) loadHistory(current);
}
$id("chartRun").addEventListener("change", (e) => loadHistory(e.target.value));

// ---------------------------------------------------------------- machines

// A machine is a run directory: a network that keeps training, with its checkpoints.
function renderMachines() {
  const sel = $id("machine");
  const active = ACTIVE.includes(state.status.state);
  const keep = (active && state.status.run) || sel.value || state.chartRun;
  sel.innerHTML = state.runs.length ? state.runs.map((r) => `<option value="${esc(r.name)}">${esc(r.name)}</option>`).join("")
    : `<option value="">No machines yet</option>`;
  if (state.runs.some((r) => r.name === keep)) sel.value = keep;
  sel.disabled = active || !state.runs.length;
  renderMachine();
}

function renderMachine() {
  const name = $id("machine").value;
  const run = state.runs.find((r) => r.name === name);
  if (!run) {
    $id("machineInfo").textContent = "Create a machine to start training.";
    $id("machineCkpts").hidden = true;
    return;
  }
  const origin = run.origin ? ` · started from ${shortPath(run.origin)}` : "";
  $id("machineInfo").textContent = `${run.status} · ${fmtInt(run.games)} games · ${fmtDuration(run.elapsed)} trained${origin}`;
  const ckpts = (state.ckpts || []).filter((c) => c.run === name);
  $id("machineCkpts").hidden = false;
  $id("ckptCount").textContent = `(${ckpts.length})`;
  $id("ckptList").innerHTML = ckpts.length ? ckpts.map((c) => {
    const m = c.label.match(/ckpt_(\d+)g\.pt$/);
    const meta = [m ? `${fmtInt(Number(m[1]))} games` : "", c.elo === null || c.elo === undefined ? "" : `Elo ${Math.round(c.elo)}`]
      .filter(Boolean).join(" · ");
    return `<li><span class="ckpt-name">${esc(c.label.split("/").pop())}</span><span class="ckpt-meta">${meta}</span>` +
      `<span class="ckpt-actions" data-path="${esc(c.path)}"><button type="button" data-act="play" title="Play against it">Play</button>` +
      `<button type="button" data-act="measure" title="Measure its Elo">Elo</button>` +
      `<button type="button" data-act="fork" title="New machine starting from this checkpoint">Fork</button></span></li>`;
  }).join("") : `<li class="ckpt-empty">No checkpoints yet. The first is saved when training starts.</li>`;
}
$id("machine").addEventListener("change", () => { renderMachine(); loadHistory($id("machine").value); $id("chartRun").value = $id("machine").value; });

$id("ckptList").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-act]");
  if (!btn) return;
  const path = btn.parentElement.dataset.path;
  if (btn.dataset.act === "fork") { openMachineDialog(path); return; }
  const tab = btn.dataset.act === "play" ? "play" : "strength";
  $id(tab === "play" ? "playCkpt" : "msCkpt").value = path;
  document.querySelector(`.tab[data-tab="${tab}"]`).click();
});

function openMachineDialog(from = "") {
  $id("machineFrom").innerHTML = `<option value="">An untrained network</option>` +
    (state.ckpts || []).map((c) => `<option value="${esc(c.path)}">${esc(c.label)}</option>`).join("");
  $id("machineFrom").value = from;
  $id("machineName").value = "";
  $id("machineError").textContent = "";
  $id("machineDialog").showModal();
  $id("machineName").focus();
}
$id("btnNewMachine").addEventListener("click", () => openMachineDialog());
$id("machineCancel").addEventListener("click", () => $id("machineDialog").close());
$id("machineForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const m = await api("/api/machines", { name: $id("machineName").value, checkpoint: $id("machineFrom").value || null });
    $id("machineDialog").close();
    log(`Created machine ${m.name}`);
    await refreshRuns(m.name);
    $id("machine").value = m.name;
    renderMachine();
  } catch (err) { $id("machineError").textContent = err.message; }
});

// ---------------------------------------------------------------- status & controls

const ACTIVE = ["starting", "running", "paused", "stopping"];

function opponentText(s) {
  if (s.engine_share === undefined) return "–";
  if (!s.engine_share) return "self-play";
  const score = s.engine_score == null ? "" : ` · network ${(100 * s.engine_score).toFixed(0)}% at this level`;
  const elo = s.engine_elo == null ? "" : ` ≈ ${Math.round(s.engine_elo)} Elo`;
  return `${Math.round(100 * s.engine_share)}% vs Stockfish · level ${s.engine_level}${elo}${s.engine_adaptive ? " (adaptive)" : ""}${score}`;
}

function renderStatus(s) {
  const prevRun = state.status.run, prevState = state.status.state;
  state.status = s;
  const badge = $id("stateBadge");
  badge.textContent = s.state;
  badge.className = `badge ${s.state}`;
  $id("st-run").textContent = s.run || "–";
  $id("st-preset").textContent = s.preset ? `${s.preset} (${s.workers ?? "?"} workers, ${s.threads ?? "?"} threads, ${s.sims ?? "?"} sims, nice ${s.nice ?? "?"})` : "–";
  $id("st-opponent").textContent = opponentText(s);
  $id("st-duration").textContent = s.duration || "–";
  $id("st-elapsed").textContent = fmtDuration(s.elapsed);
  const showEta = ["running", "paused"].includes(s.state) && s.eta !== null && s.eta !== undefined;
  $id("st-eta").textContent = showEta ? fmtDuration(s.eta) : "–";
  $id("st-games").textContent = s.games === undefined ? "–" : `${fmtInt(s.games)}${s.max_games ? ` (${fmtInt(s.session_games)} / ${fmtInt(s.max_games)} this session)` : ""}`;
  $id("st-gps").textContent = fmtNum(s.games_per_sec);
  $id("st-positions").textContent = fmtInt(s.positions);
  $id("st-eps").textContent = s.evals_per_sec === undefined ? "–" : fmtInt(Math.round(s.evals_per_sec));
  $id("st-steps").textContent = fmtInt(s.steps);
  $id("st-lr").textContent = s.lr == null ? "–" : s.lr.toExponential(1);
  $id("st-resign").textContent = s.resign_threshold === undefined ? "–" : s.resign_threshold === null ? "off"
    : `below ${s.resign_threshold.toFixed(2)}` + (s.resign_false == null ? "" :
      ` · ${Math.round(100 * s.resign_false)}% wrong of ${s.resign_checked} checked`);
  $id("st-buffer").textContent = fmtInt(s.buffer);
  $id("st-ckpt").textContent = (s.checkpoint || "–") + (s.evaluating ? " (evaluating…)" : "");
  $id("st-elo").textContent = s.elo
    ? `Elo ${Math.round(s.elo.elo)}${s.elo.error ? ` ± ${Math.round(s.elo.error)}` : ""} (${s.elo.checkpoint})`
    : s.calibrated === false ? "not calibrated (python -m chessml calibrate)" : "–";
  const active = ACTIVE.includes(s.state);
  $id("btnStart").disabled = active;
  $id("btnPause").disabled = s.state !== "running";
  $id("btnResume").disabled = s.state !== "paused";
  $id("btnStop").disabled = !["running", "paused"].includes(s.state);
  if (ACTIVE.includes(s.state) !== ACTIVE.includes(prevState)) renderMachines();  // lock or unlock the machine picker
  if (s.run && s.run !== prevRun) refreshRuns(s.run);
  if (["stopped", "completed"].includes(s.state) && prevRun) refreshRuns();
}

function updatePresetHint() {
  if (!state.info) return;
  const p = state.info.presets[$id("preset").value];
  $id("presetHint").textContent = `${p.workers} self-play workers · ${p.threads} torch threads · ${p.sims} sims/move · ` +
    `batch ${p.batch_size} · nice ${p.nice} · buffer ${fmtInt(p.buffer_size)} positions ` +
    `(about ${Math.round((p.buffer_size * 1.85) / 1000)} MB) · this machine has ${state.info.cpu_count} CPUs`;
  for (const k of ["workers", "threads", "sims", "batch_size", "nice", "buffer_size"]) $id(k).placeholder = p[k];
}
$id("preset").addEventListener("change", updatePresetHint);

function updateEngineFields() {
  $id("engineFields").hidden = $id("mode").value !== "engine";
  if (!state.info) return;
  $id("engineHint").textContent = state.info.engine ? `Found ${state.info.engine}` :
    "Stockfish not found on PATH. Install it (sudo apt install stockfish) or enter the path of the binary.";
}
$id("mode").addEventListener("change", updateEngineFields);

$id("trainForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  $id("trainError").textContent = "";
  const num = (id) => ($id(id).value === "" ? null : Number($id(id).value));
  const body = {
    preset: $id("preset").value,
    workers: num("workers"), threads: num("threads"), sims: num("sims"), batch_size: num("batch_size"), nice: num("nice"),
    buffer_size: num("buffer_size"),
    duration: document.querySelector("input[name=duration]:checked").value,
    games: num("games"), time: $id("time").value,
    resume: $id("machine").value || null,
  };
  if (!body.resume) { $id("trainError").textContent = "Create a machine first (New machine)."; return; }
  if ($id("mode").value === "engine") {
    Object.assign(body, {
      engine_share: (num("engineShare") ?? 50) / 100, engine_level: num("engineLevel"),
      engine_adaptive: $id("engineAdaptive").checked, engine_path: $id("enginePath").value || null,
    });
  }
  try {
    renderStatus(await api("/api/training/start", body));
    log(`Starting ${body.preset} training of ${body.resume}…`);
  } catch (err) { $id("trainError").textContent = err.message; }
});
for (const [id, action] of [["btnPause", "pause"], ["btnResume", "resume"], ["btnStop", "stop"]]) {
  $id(id).addEventListener("click", async () => {
    try { await api(`/api/training/${action}`, {}); } catch (err) { $id("trainError").textContent = err.message; }
  });
}

// ---------------------------------------------------------------- live board

const liveBoard = Chessboard("liveBoard", { position: "start", pieceTheme: PIECES });

// every running game as a thumbnail; the selected one (worker, slot) is shown above in detail
const live = { focus: [0, 0], items: [] };
const liveGrid = new MiniBoards($id("liveGrid"), {
  emptyText: "Every game in progress appears here while training runs.",
  onSelect: (key) => focusGame(...key.split("-").map(Number)),
});

function gridItem(g) {
  const opponent = g.engine ? `vs Stockfish level ${g.engine_level} (network plays ${g.engine === "white" ? "Black" : "White"})` : "self-play";
  return { key: `${g.worker}-${g.slot}`, board: g.board, last_move: g.last_move, label: `${g.worker}·${g.slot}`,
    badge: g.engine ? `SF${g.engine_level}` : "", title: `Worker ${g.worker}, game ${g.slot} · ${opponent} · ply ${g.ply}` };
}

function renderGrid(ev) {
  if (!ev) return;
  live.items = ev.games.map(gridItem);
  liveGrid.update(live.items, live.focus.join("-"));
  const engine = ev.games.filter((g) => g.engine).length;
  $id("gridInfo").textContent = `${ev.games.length} games${engine ? `, ${engine} against Stockfish` : ""} · click one to watch it above`;
}

async function focusGame(worker, slot) {
  live.focus = [worker, slot];
  liveGrid.update(live.items, `${worker}-${slot}`);
  const g = live.items.find((i) => i.key === `${worker}-${slot}`);
  if (g) {  // show the thumbnail's position at once; the details follow within a quarter second
    showPosition(liveBoard, "liveBoard", g.board, g.last_move);
    $id("liveInfo").textContent = g.title;
  }
  $id("liveBoard").scrollIntoView({ behavior: "smooth", block: "nearest" });
  try { await api("/api/training/focus", { worker, slot }); } catch (err) { log(`Could not select that game: ${err.message}`); }
}

function renderLive(ev) {
  if (!ev) return;
  if (ev.worker !== undefined && (ev.worker !== live.focus[0] || ev.slot !== live.focus[1])) return;  // from before a switch
  showPosition(liveBoard, "liveBoard", ev.fen, ev.last_move);
  const where = ev.worker === undefined ? "" : `Worker ${ev.worker}, game ${ev.slot} · `;
  $id("liveInfo").textContent = `${where}${ev.opponent ? ev.opponent + " · " : ""}ply ${ev.ply}, ${ev.turn} to move`;
  setValueBar($id("liveBar"), $id("liveValue"), ev.value_white,
    `network ${signed(ev.value_white)} · after search ${signed(ev.search_white)} (−1 Black wins … +1 White wins)`);
  const share = Object.fromEntries((ev.top || []).map((m) => [m.uci, m.prob]));
  $id("liveTop").innerHTML = (ev.policy || []).map((m) => {
    const s = share[m.uci];
    return `<tr><td>${m.san}</td><td class="bar-cell"><span class="bar" style="width:${(m.prior * 100).toFixed(1)}%"></span>` +
      `<span class="val">${(m.prior * 100).toFixed(1)}%</span></td><td>${s === undefined ? "–" : (s * 100).toFixed(0) + "%"}</td></tr>`;
  }).join("");
}

// ---------------------------------------------------------------- websocket

function handleEvent(ev) {
  const sameRun = state.chartRun && state.chartRun === state.status.run;
  switch (ev.type) {
    case "hello":
      renderStatus(ev.status);
      if (ev.grid && ev.grid.focus) live.focus = ev.grid.focus;
      renderGrid(ev.grid);
      renderLive(ev.live);
      break;
    case "grid": renderGrid(ev); break;
    case "status": renderStatus(ev); break;
    case "live": renderLive(ev); break;
    case "log": log(ev.message); break;
    case "checkpoint": refreshCheckpoints(); break;
    case "train":
      if (!sameRun) break;
      charts.loss.push(0, [ev.elapsed / 60, ev.policy_loss]);
      charts.loss.push(1, [ev.elapsed / 60, ev.value_loss]);
      charts.loss.push(2, [ev.elapsed / 60, ev.total_loss]);
      break;
    case "system":
      if (!sameRun) break;
      charts.gps.push(0, [ev.elapsed / 60, ev.games_per_sec]);
      charts.cpu.push(0, [ev.elapsed / 60, ev.cpu_total]);
      charts.cores.set(ev.cpu_per_core);
      break;
    case "game":
      if (!sameRun) break;
      state.games.push(ev);
      renderOutcomes();
      break;
    case "eval":
      if (sameRun) loadHistory(state.chartRun);  // ratings are refitted over all evaluations
      break;
    case "strength": applyStrengthEvent(ev); break;
  }
}

function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  const conn = $id("conn");
  ws.onopen = () => { conn.textContent = "live"; conn.className = "conn ok"; };
  ws.onclose = () => { conn.textContent = "reconnecting…"; conn.className = "conn bad"; setTimeout(connect, 2000); };
  ws.onmessage = (m) => handleEvent(JSON.parse(m.data));
}

// ---------------------------------------------------------------- play

const game = new Chess();
const play = { color: "white", checkpoint: null, sims: 200, thinking: false, selected: null, started: false, analysis: null };
const playBoard = Chessboard("playBoard", { position: "start", pieceTheme: PIECES, draggable: false });

async function refreshCheckpoints() {
  let ckpts = [];
  try { ckpts = await api("/api/checkpoints"); } catch { return; }
  state.ckpts = ckpts;
  renderMachine();
  const elo = (c) => (c.elo === null || c.elo === undefined ? "" : ` · Elo ${Math.round(c.elo)}`);
  for (const id of ["playCkpt", "msCkpt"]) {
    const sel = $id(id);
    const keep = sel.value;
    sel.innerHTML = ckpts.length ? ckpts.map((c) => `<option value="${c.path}">${c.label}${elo(c)}</option>`).join("")
      : `<option value="">No checkpoints yet – train first</option>`;
    if ([...sel.options].some((o) => o.value === keep)) sel.value = keep;
  }
  if (strength.data) renderMeasureForm(strength.data);
}

const historyUci = () => game.history({ verbose: true }).map((m) => m.from + m.to + (m.promotion || ""));
const userToMove = () => play.started && !play.thinking && !game.isGameOver() && game.turn() === play.color[0];

function gameOverText() {
  if (game.isCheckmate()) return game.turn() === play.color[0] ? "Checkmate – the engine wins." : "Checkmate – you win!";
  if (game.isStalemate()) return "Draw by stalemate.";
  if (game.isThreefoldRepetition()) return "Draw by threefold repetition.";
  if (game.isInsufficientMaterial()) return "Draw by insufficient material.";
  if (game.isDraw()) return "Draw (fifty-move rule).";
  return "";
}

function renderPlay() {
  playBoard.position(game.fen(), false);
  const hist = game.history({ verbose: true });
  const last = hist[hist.length - 1];
  highlightSquares("playBoard", "sq-last", last ? [last.from, last.to] : []);
  highlightSquares("playBoard", "sq-selected", play.selected ? [play.selected] : []);
  const targets = play.selected ? game.moves({ square: play.selected, verbose: true }).map((m) => m.to) : [];
  highlightSquares("playBoard", "sq-target", targets);
  const san = game.history();
  const items = [];
  for (let i = 0; i < san.length; i += 2) items.push(`<li>${san[i]} ${san[i + 1] || ""}</li>`);
  $id("moveList").innerHTML = items.join("");
  let status = "Choose a checkpoint and press New game.";
  if (play.started) {
    status = game.isGameOver() ? gameOverText()
      : play.thinking ? "Engine is thinking…"
      : `Your move (${play.color})${game.inCheck() ? " – check!" : ""}`;
  }
  $id("playStatus").textContent = status;
  $id("btnUndo").disabled = !play.started || play.thinking || hist.length === 0;
}

function renderAnalysis(a, label) {
  const show = $id("showAnalysis").checked;
  $id("analysisCard").hidden = !show;
  if (!show || !a || a.game_over) return;
  $id("anaInfo").textContent = `${label} · ${fmtInt(a.sims)} positions in ${a.seconds.toFixed(1)}s`;
  setValueBar($id("anaBar"), $id("anaValue"), a.value_white,
    `${signed(a.value_white)} for White (expected score ${(((a.value_white + 1) / 2) * 100).toFixed(0)}%)`);
  $id("anaTop").innerHTML = a.top.map((m) => `<tr><td>${m.san}</td><td>${fmtInt(m.visits)}</td>` +
    `<td>${(m.prob * 100).toFixed(0)}%</td><td>${m.q === null ? "–" : signed(m.q)}</td></tr>`).join("");
  highlightSquares("playBoard", "sq-hint", label === "your candidates" && a.top[0] ? [a.top[0].uci.slice(0, 2), a.top[0].uci.slice(2, 4)] : []);
}

async function requestSearch() {
  return api("/api/play/move", { moves: historyUci(), checkpoint: play.checkpoint, sims: play.sims });
}

async function engineTurn() {
  renderPlay();
  if (!play.started || game.isGameOver()) return;
  if (game.turn() === play.color[0]) {
    if ($id("showAnalysis").checked) {
      const fen = game.fen();
      try {
        const a = await requestSearch();
        if (game.fen() === fen) renderAnalysis(a, "your candidates");
      } catch (err) { $id("playStatus").textContent = err.message; }
    }
    return;
  }
  play.thinking = true;
  renderPlay();
  try {
    const reply = await requestSearch();
    if (!reply.game_over) {
      game.move({ from: reply.move.slice(0, 2), to: reply.move.slice(2, 4), promotion: reply.move[4] });
      $id("playSimsInfo").textContent = `Last move: ${fmtInt(reply.sims)} positions in ${reply.seconds.toFixed(1)}s.`;
      renderAnalysis(reply, "engine's search before its move");
    }
  } catch (err) {
    play.thinking = false;
    renderPlay();
    $id("playStatus").textContent = `Engine error: ${err.message}`;
    return;
  }
  play.thinking = false;
  engineTurn();
}

function askPromotion() {
  const dlg = $id("promoDialog");
  return new Promise((resolve) => {
    const done = (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      dlg.removeEventListener("click", done);
      dlg.close();
      resolve(b.value);
    };
    dlg.addEventListener("click", done);
    dlg.showModal();
  });
}

async function onSquareClick(sq) {
  if (!userToMove()) return;
  const piece = game.get(sq);
  if (play.selected && play.selected !== sq) {
    const legal = game.moves({ square: play.selected, verbose: true }).filter((m) => m.to === sq);
    if (legal.length) {
      const from = play.selected;
      play.selected = null;
      const promotion = legal.some((m) => m.promotion) ? await askPromotion() : undefined;
      game.move({ from, to: sq, promotion });
      highlightSquares("playBoard", "sq-hint", []);
      engineTurn();
      return;
    }
  }
  play.selected = piece && piece.color === play.color[0] && play.selected !== sq ? sq : null;
  renderPlay();
}
$("#playBoard").on("click", ".square-55d63", function () { onSquareClick($(this).attr("data-square")); });

$id("btnNew").addEventListener("click", () => {
  play.checkpoint = $id("playCkpt").value;
  if (!play.checkpoint) { $id("playStatus").textContent = "No checkpoint selected."; return; }
  play.color = $id("playColor").value;
  play.sims = Number($id("playSims").value) || 200;
  $id("playSimsInfo").textContent = "";
  play.started = true;
  play.thinking = false;
  play.selected = null;
  game.reset();
  playBoard.orientation(play.color);
  $id("pgnBox").hidden = true;
  highlightSquares("playBoard", "sq-hint", []);
  engineTurn();
});
$id("playSims").addEventListener("change", () => { play.sims = Number($id("playSims").value) || play.sims; });
$id("btnUndo").addEventListener("click", () => {
  if (play.thinking || !game.history().length) return;
  game.undo();
  if (game.history().length && game.turn() !== play.color[0]) game.undo();
  play.selected = null;
  highlightSquares("playBoard", "sq-hint", []);
  engineTurn();
});
$id("btnFlip").addEventListener("click", () => { playBoard.orientation("flip"); renderPlay(); });
$id("showAnalysis").addEventListener("change", () => {
  $id("analysisCard").hidden = !$id("showAnalysis").checked;
  if (!$id("showAnalysis").checked) highlightSquares("playBoard", "sq-hint", []);
  else if (userToMove()) engineTurn();
});

function pgnText() {
  const engine = `ChessML ${($id("playCkpt").selectedOptions[0] || {}).text || ""} (${play.sims} sims)`;
  game.setHeader("Event", "ChessML game");
  game.setHeader("Date", new Date().toISOString().slice(0, 10).replaceAll("-", "."));
  game.setHeader("White", play.color === "white" ? "Human" : engine);
  game.setHeader("Black", play.color === "black" ? "Human" : engine);
  let result = "*";
  if (game.isCheckmate()) result = game.turn() === "w" ? "0-1" : "1-0";
  else if (game.isDraw() || game.isStalemate()) result = "1/2-1/2";
  game.setHeader("Result", result);
  return game.pgn();
}
$id("btnPgn").addEventListener("click", () => {
  $id("pgnText").value = pgnText();
  $id("pgnBox").hidden = false;
});
$id("btnPgnCopy").addEventListener("click", () => navigator.clipboard?.writeText($id("pgnText").value));
$id("btnPgnSave").addEventListener("click", () => {
  const blob = new Blob([$id("pgnText").value + "\n"], { type: "application/x-chess-pgn" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `chessml-${new Date().toISOString().slice(0, 19).replaceAll(":", "-")}.pgn`;
  a.click();
  URL.revokeObjectURL(a.href);
});

// ---------------------------------------------------------------- strength

const strength = { data: null, selected: null };

function playerName(name) {
  if (name === "random") return "random moves";
  if (name === "material") return "material player";
  let m = name.match(/^sf-level-(.+)$/);
  if (m) return `Stockfish level ${m[1]}`;
  m = name.match(/^sf-elo-(\d+)$/);
  if (m) return `Stockfish Elo ${m[1]}`;
  return name;
}

async function refreshStrength() {
  let d;
  try { d = await api("/api/strength"); } catch { return; }
  strength.data = d;
  if (!d.results.some((r) => r.created === strength.selected)) strength.selected = d.results.length ? d.results[0].created : null;
  renderStrength();
}

function renderStrength() {
  const d = strength.data;
  if (!d) return;
  renderCalibration(d);
  renderMeasureForm(d);
  renderJob();
  renderJobBoards();
  renderResult();
  renderLadder();
  renderHistory();
}

// minutes for `rounds` rounds of 40 matches (one round of 20 games per match is about 2 CPU-hours)
function calibrationEstimate(rounds) {
  const d = strength.data;
  const games = Number($id("calGames").value) || 20;
  const procs = Number($id("calProcs").value) || d.default_processes;
  return Math.max(1, Math.round((rounds * 120 * games) / 20 / procs));
}

const refineGames = (cal) => cal.games - cal.refined_games;  // a refinement round as large as the first round

function renderCalibration(d) {
  const cal = d.calibration, badge = $id("calBadge");
  $id("calProcs").placeholder = d.default_processes;
  if (!d.engine) {
    badge.textContent = "no engine"; badge.className = "badge error";
    $id("calStatus").textContent = "Stockfish was not found. Install it (sudo apt install stockfish) and reload.";
  } else if (!cal) {
    badge.textContent = "not calibrated"; badge.className = "badge";
    $id("calStatus").textContent = "Needed once per machine. A ladder of weaker players (random moves, the material " +
      "player, Stockfish strength levels) plays Stockfish's Elo-calibrated mode, which ties the ladder to real Elo.";
  } else {
    badge.textContent = "calibrated"; badge.className = "badge calibrated";
    const refined = cal.refined_games ? `, ${fmtInt(cal.refined_games)} of them in refinement` : "";
    $id("calStatus").textContent = `${cal.engine} · ${fmtDate(cal.created)} · ${fmtInt(cal.games)} games in ` +
      `${cal.matches} matches${refined} · took ${fmtDuration(cal.seconds)}`;
  }
  const btn = $id("btnCalibrate");
  btn.textContent = cal ? "Recalibrate" : "Calibrate";
  btn.className = cal ? "" : "primary";
  btn.disabled = !d.engine || d.running;
  $id("btnRefine").hidden = !cal;
  $id("btnRefine").disabled = !d.engine || d.running;
  $id("calHint").textContent = !d.engine ? "" : (cal
    ? `Refine plays ${fmtInt(refineGames(cal))} more games where they narrow the lower ladder most (about ` +
      `${calibrationEstimate(1)} min). Recalibrate starts over (about ${calibrationEstimate(2)} min).`
    : `About ${calibrationEstimate(2)} min with these settings, including a refinement round.`) +
    (d.training ? " Training is running, so both run slower." : "");
}

function renderMeasureForm(d) {
  $id("msProcs").placeholder = d.default_processes;
  $id("btnMeasure").disabled = !d.calibration || !d.engine || d.running || !$id("msCkpt").value;
  $id("msHint").textContent = !d.calibration ? "Calibrate first." :
    `Scouts with ${d.scout_games} games against every ladder player, then plays the players within ${d.window} Elo of ` +
    "that estimate. A few minutes on several processes." +
    (d.training ? " Training is running: both share the CPU and run slower, the result is unaffected." : "");
}

function jobTitle(job) {
  return job.kind === "calibrate" ? "Calibration" : `Measuring ${job.params.label || shortPath(job.params.checkpoint)}`;
}

function renderJob() {
  const job = strength.data && strength.data.job;
  $id("jobCard").hidden = !job;
  if (!job) return;
  const running = ["starting", "running"].includes(job.state);
  $id("jobTitle").textContent = jobTitle(job);
  const badge = $id("jobBadge");
  badge.textContent = running ? (job.stage || "starting") : job.state;
  badge.className = `badge ${running ? "running" : job.state}`;
  $id("jobBar").style.width = `${job.total ? (100 * job.done) / job.total : 0}%`;
  let text = "";
  const info = job.stage_info || {};
  if (job.state === "error") text = `Failed: ${job.message || "unknown error"}`;
  else if (job.state === "cancelled") text = "Cancelled.";
  else if (job.state === "completed") text = job.kind === "calibrate" ? "Calibration saved." : "Done. The result is shown on the right.";
  else if (job.stage === "scouting") text = `Scouting: ${info.games} games against each ladder player · ${job.done}/${job.total} matches`;
  else if (job.stage === "measuring") text = `Scouting estimate ${info.estimate === null ? "–" : Math.round(info.estimate)}. ` +
    `Now ${info.games} games against each of ${info.opponents.map(playerName).join(", ")} · ${job.done}/${job.total} matches`;
  else if (job.stage === "calibrating") text = `Round 1: ${info.games} games per match · ${job.done}/${job.total} matches`;
  else if (job.stage === "refining") text = `Refinement: ${info.games} games each for the ${info.matches} matches that ` +
    `narrow the lower ladder most · ${job.done}/${job.total} matches`;
  else text = "Starting…";
  $id("jobText").textContent = text;
  $id("jobMatches").innerHTML = [...job.matches].reverse().map((m) => {
    const who = job.kind === "calibrate" ? `${playerName(m.player)} vs ${playerName(m.opponent)}` : playerName(m.opponent);
    const stage = job.kind === "measure" && m.stage === "scouting" ? " (scouting)" : "";
    return `<li>${esc(who)}${stage}: +${m.wins} =${m.draws} −${m.losses}</li>`;
  }).join("");
  $id("btnCancelJob").hidden = !running;
}

function applyStrengthEvent(ev) {
  const d = strength.data;
  if (!d) return;
  if (ev.event === "boards") {
    if (d.job) { d.job.boards = ev.games; renderJobBoards(); }
    return;
  }
  if (ev.event === "started" || !d.job) {
    refreshStrength();  // a job started elsewhere (another browser tab)
    return;
  }
  const job = d.job;
  if (ev.event === "stage") {
    Object.assign(job, { stage: ev.stage, stage_info: ev, done: 0, total: ev.opponents ? ev.opponents.length : ev.matches || 0 });
  } else if (ev.event === "match") {
    Object.assign(job, { done: ev.done, total: ev.total, state: "running" });
    job.matches.push({ ...ev.record, stage: job.stage });
  } else {
    if (ev.event === "completed" && ev.kind === "measure") strength.selected = ev.result.created;
    refreshStrength();
    refreshCheckpoints();
    return;
  }
  renderJob();
}

// live boards of the running calibration or measurement
const jobBoard = Chessboard("jobBoard", { position: "start", pieceTheme: PIECES });
const jobLive = { selected: null };
const jobGrid = new MiniBoards($id("jobGrid"), {
  emptyText: "Games appear here while a job runs.",
  onSelect: (key) => { jobLive.selected = key; renderJobBoards(); $id("jobBoard").scrollIntoView({ behavior: "smooth", block: "nearest" }); },
});

function shortName(name) {
  if (name === "random") return "rnd";
  if (name === "material") return "mat";
  let m = name.match(/^sf-level-(.+)$/);
  if (m) return `L${m[1]}`;
  m = name.match(/^sf-elo-(\d+)$/);
  return m ? `E${m[1]}` : "net";
}

function renderJobBoards() {
  const job = strength.data && strength.data.job;
  const games = (job && ["starting", "running"].includes(job.state) && job.boards) || [];
  const card = $id("jobLiveCard");
  const wasHidden = card.hidden;
  card.hidden = !games.length;
  if (!games.length) return;
  if (wasHidden) jobBoard.resize();
  if (!games.some((g) => g.key === jobLive.selected)) jobLive.selected = (games.find((g) => !g.result) || games[0]).key;
  const measure = job.kind === "measure";
  jobGrid.update(games.map((g) => ({
    key: g.key, board: g.board, last_move: g.last_move,
    label: measure ? shortName(g.white.endsWith(".pt") ? g.black : g.white) : `${shortName(g.white)}–${shortName(g.black)}`,
    badge: g.result || "",
    title: `${playerName(g.white)} (White) vs ${playerName(g.black)} (Black) · game ${g.game} of ${g.games} · ply ${g.ply}`,
  })), jobLive.selected);
  const g = games.find((x) => x.key === jobLive.selected);
  showPosition(jobBoard, "jobBoard", g.board, g.last_move);
  $id("jg-white").textContent = playerName(g.white);
  $id("jg-black").textContent = playerName(g.black);
  $id("jg-game").textContent = `${g.game} of ${g.games} in this match`;
  $id("jg-ply").textContent = g.ply;
  $id("jg-result").textContent = g.result || "in progress";
  const matches = new Set(games.map((x) => x.key.split("-")[0])).size;
  const done = games.filter((x) => x.result).length;
  $id("jobLiveInfo").textContent = `${playerName(g.white)} vs ${playerName(g.black)}`;
  $id("jobGridInfo").textContent = `${games.length} games in ${matches} running matches, ${done} finished · click one to watch it`;
}

function selectedResult() {
  const d = strength.data;
  return d && d.results.find((r) => r.created === strength.selected);
}

function ladderElo(name) {
  const cal = strength.data.calibration;
  const p = cal && cal.ladder.find((x) => x.name === name);
  return p ? p.elo : null;
}

function renderResult() {
  const r = selectedResult();
  $id("resEmpty").hidden = !!r;
  $id("resBody").hidden = !r;
  $id("resWhen").textContent = r ? `${fmtDate(r.created)} · ${r.games} games per opponent · ${r.sims} simulations` : "";
  if (!r) return;
  $id("resElo").textContent = Math.round(r.elo);
  $id("resCi").textContent = r.error ? `± ${Math.round(r.error)} (95%)` : "";
  $id("resName").textContent = r.label;
  const ladder = strength.data.calibration ? strength.data.calibration.ladder : [];
  const below = ladder.filter((p) => p.elo <= r.elo).pop();
  const above = ladder.find((p) => p.elo > r.elo);
  const place = below && above ? `Between ${playerName(below.name)} (${Math.round(below.elo)}) and ${playerName(above.name)} (${Math.round(above.elo)}).`
    : below ? `Above ${playerName(below.name)} (${Math.round(below.elo)}), the strongest ladder player.`
    : above ? `Below ${playerName(above.name)} (${Math.round(above.elo)}), the weakest ladder player.` : "";
  const range = r.error ? ` With 95% confidence its strength lies between ${Math.round(r.elo - r.error)} and ${Math.round(r.elo + r.error)}.` : "";
  let sources = "";
  if (r.error && r.own_error) {  // the two parts add in quadrature
    const ladder = Math.sqrt(Math.max(r.error ** 2 - r.own_error ** 2, 0));
    sources = ` Of the ± ${Math.round(r.error)}, its own games account for ± ${Math.round(r.own_error)} and the calibrated ladder ` +
      `for ± ${Math.round(ladder)}: ` + (ladder > r.own_error
        ? "refining the calibration would narrow it more than more games would."
        : "more games per opponent would narrow it.");
  }
  $id("resText").textContent = `${place}${range}${sources} Engine Elo on the scale of Stockfish's UCI_Elo (CCRL 40/4), ` +
    "not a FIDE or online rating.";
  const rows = [...r.records].sort((a, b) => (ladderElo(a.opponent) ?? 0) - (ladderElo(b.opponent) ?? 0));
  $id("resTable").innerHTML = rows.map((m) => {
    const n = m.wins + m.draws + m.losses, elo = ladderElo(m.opponent);
    return `<tr><td>${esc(playerName(m.opponent))}</td><td>${elo === null ? "–" : Math.round(elo)}</td><td>${m.wins}</td>` +
      `<td>${m.draws}</td><td>${m.losses}</td><td>${n ? ((100 * (m.wins + m.draws / 2)) / n).toFixed(1) + "%" : "–"}</td></tr>`;
  }).join("");
}

function renderLadder() {
  const d = strength.data, el = $id("ladder");
  if (!d.calibration) {
    el.innerHTML = `<div class="empty-note">Calibrate to see the ladder.</div>`;
    return;
  }
  const rows = d.calibration.ladder.map((p) => ({ name: playerName(p.name), elo: p.elo, error: p.error, cls: p.anchor ? "anchor" : "" }));
  const seen = new Set();
  for (const r of d.results) {  // newest measurement per checkpoint
    if (seen.has(r.label) || (seen.size >= 6 && r.created !== strength.selected)) continue;
    seen.add(r.label);
    rows.push({ name: r.label, elo: r.elo, error: r.error, cls: r.created === strength.selected ? "measured selected" : "measured", title: r.label });
  }
  rows.sort((a, b) => b.elo - a.elo);
  let lo = Math.min(...rows.map((r) => r.elo - (r.error || 0)));
  let hi = Math.max(...rows.map((r) => r.elo + (r.error || 0)));
  const step = hi - lo > 1500 ? 500 : 250;
  lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
  const x = (v) => (100 * (v - lo)) / (hi - lo);
  const ticks = [];
  for (let v = lo; v <= hi; v += step) ticks.push(v);
  const grid = ticks.map((v) => `<div class="ladder-grid" style="left:${x(v)}%"></div>`).join("");
  const axis = `<div class="ladder-head"><div class="ladder-name"></div><div class="ladder-axis">` +
    `${ticks.map((v) => `<span style="left:${x(v)}%">${v}</span>`).join("")}</div><div class="ladder-val"></div></div>`;
  el.innerHTML = axis + rows.map((r) => {
    const ci = r.error ? `<div class="ladder-ci" style="left:${x(r.elo - r.error)}%;width:${x(r.elo + r.error) - x(r.elo - r.error)}%"></div>` : "";
    const val = `${Math.round(r.elo)}${r.error ? ` ± ${Math.round(r.error)}` : ""}`;
    return `<div class="ladder-row ${r.cls}"><div class="ladder-name" title="${esc(r.title || r.name)}">${esc(r.name)}</div>` +
      `<div class="ladder-track">${grid}${ci}<div class="ladder-dot" style="left:${x(r.elo)}%"></div></div>` +
      `<div class="ladder-val">${val}</div></div>`;
  }).join("");
}

function renderHistory() {
  const d = strength.data;
  $id("resHistory").innerHTML = d.results.length ? d.results.map((r) =>
    `<tr data-created="${r.created}" class="${r.created === strength.selected ? "selected" : ""}"><td>${esc(r.label)}</td>` +
    `<td>${Math.round(r.elo)}</td><td>${r.error ? "± " + Math.round(r.error) : "–"}</td>` +
    `<td>${fmtInt(r.records.reduce((n, m) => n + m.wins + m.draws + m.losses, 0))}</td><td>${r.sims}</td><td>${fmtDate(r.created)}</td></tr>`).join("")
    : `<tr><td colspan="6">No measurements yet</td></tr>`;
}

$id("resHistory").addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-created]");
  if (!tr) return;
  strength.selected = Number(tr.dataset.created);
  renderResult(); renderLadder(); renderHistory();
});
for (const id of ["calGames", "calProcs"]) $id(id).addEventListener("input", () => strength.data && renderCalibration(strength.data));
$id("msCkpt").addEventListener("change", () => strength.data && renderMeasureForm(strength.data));

async function startStrengthJob(path, body) {
  $id("strengthError").textContent = "";
  try {
    strength.data.job = await api(path, body);
    strength.data.running = true;
    renderStrength();
  } catch (err) { $id("strengthError").textContent = err.message; }
}

const optNum = (id) => ($id(id).value === "" ? null : Number($id(id).value));
$id("btnCalibrate").addEventListener("click", () => {
  if (strength.data.calibration && !confirm("Replace the current calibration? Stored measurements are refitted against the new one.")) return;
  startStrengthJob("/api/strength/calibrate", { games: optNum("calGames") ?? 20, processes: optNum("calProcs") });
});
$id("btnRefine").addEventListener("click", () => startStrengthJob("/api/strength/calibrate", {
  refine: true, games: refineGames(strength.data.calibration), processes: optNum("calProcs"),
}));
$id("btnMeasure").addEventListener("click", () => startStrengthJob("/api/strength/measure", {
  checkpoint: $id("msCkpt").value, games: optNum("msGames") ?? 20, sims: optNum("msSims") ?? 64, processes: optNum("msProcs"),
}));
$id("btnCancelJob").addEventListener("click", async () => {
  try { await api("/api/strength/cancel", {}); } catch (err) { $id("strengthError").textContent = err.message; }
});

// ---------------------------------------------------------------- start

async function init() {
  try {
    state.info = await api("/api/info");
    updatePresetHint();
    updateEngineFields();
    renderStatus(state.info.status);
    if (state.info.grid && state.info.grid.focus) live.focus = state.info.grid.focus;
    renderGrid(state.info.grid);
    renderLive(state.info.live);
  } catch (err) { log(`Server not reachable: ${err.message}`); }
  await refreshRuns(state.status.run);
  refreshStrength();
  renderPlay();
  connect();
  document.querySelector(`.tab[data-tab="${location.hash.slice(1)}"]`)?.click();
}
init();
