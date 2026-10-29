import { Chess } from "/static/vendor/chess-1.4.0.js";
import { BarChart, LineChart } from "/static/charts.js";

const $id = (id) => document.getElementById(id);
const PIECES = "/static/vendor/img/chesspieces/wikipedia/{piece}.png";
const ROLLING = 100; // games in the outcome / length window
const ENGINE_ROLLING = 50; // engine games in the score window

const state = { info: null, status: { state: "idle" }, chartRun: null, games: [], evals: [] };

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
  renderPlay();
  if (btn.dataset.tab === "play") refreshCheckpoints();
}));
window.addEventListener("resize", () => { liveBoard.resize(); playBoard.resize(); renderPlay(); });

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
  score: new LineChart($id("chScore"), { series: ["vs random", "vs material", "vs earlier checkpoint"], xLabel: "games", integerX: true,
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
  return 2;
}

function renderEvals() {
  const series = [[], [], []];
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
  const resumeSel = $id("resume");
  const keep = resumeSel.value;
  let ckpts = [];
  try { ckpts = await api("/api/checkpoints"); } catch { /* ignore */ }
  resumeSel.innerHTML = `<option value="">New run</option>` +
    (runs.length ? `<optgroup label="Continue run (latest checkpoint)">` +
      runs.map((r) => `<option value="${r.name}">${r.name}</option>`).join("") + `</optgroup>` : "") +
    (ckpts.length ? `<optgroup label="New run from checkpoint">` +
      ckpts.map((c) => `<option value="${c.path}">${c.label}</option>`).join("") + `</optgroup>` : "");
  resumeSel.value = [...resumeSel.options].some((o) => o.value === keep) ? keep : "";
  if (current && current !== state.chartRun) loadHistory(current);
}
$id("chartRun").addEventListener("change", (e) => loadHistory(e.target.value));

// ---------------------------------------------------------------- status & controls

const ACTIVE = ["starting", "running", "paused", "stopping"];

function opponentText(s) {
  if (s.engine_share === undefined) return "–";
  if (!s.engine_share) return "self-play";
  const score = s.engine_score == null ? "" : ` · network ${(100 * s.engine_score).toFixed(0)}% at this level`;
  return `${Math.round(100 * s.engine_share)}% vs Stockfish · level ${s.engine_level}${s.engine_adaptive ? " (adaptive)" : ""}${score}`;
}

function renderStatus(s) {
  const prevRun = state.status.run;
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
  $id("st-buffer").textContent = fmtInt(s.buffer);
  $id("st-ckpt").textContent = (s.checkpoint || "–") + (s.evaluating ? " (evaluating…)" : "");
  const active = ACTIVE.includes(s.state);
  $id("btnStart").disabled = active;
  $id("btnPause").disabled = s.state !== "running";
  $id("btnResume").disabled = s.state !== "paused";
  $id("btnStop").disabled = !["running", "paused"].includes(s.state);
  if (s.run && s.run !== prevRun) refreshRuns(s.run);
  if (["stopped", "completed"].includes(s.state) && prevRun) refreshRuns();
}

function updatePresetHint() {
  if (!state.info) return;
  const p = state.info.presets[$id("preset").value];
  $id("presetHint").textContent = `${p.workers} self-play workers · ${p.threads} torch threads · ${p.sims} sims/move · ` +
    `batch ${p.batch_size} · nice ${p.nice} (this machine has ${state.info.cpu_count} CPUs)`;
  for (const k of ["workers", "threads", "sims", "batch_size", "nice"]) $id(k).placeholder = p[k];
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
    duration: document.querySelector("input[name=duration]:checked").value,
    games: num("games"), time: $id("time").value,
    resume: $id("resume").value || null, name: $id("runName").value || null,
  };
  if ($id("mode").value === "engine") {
    Object.assign(body, {
      engine_share: (num("engineShare") ?? 50) / 100, engine_level: num("engineLevel"),
      engine_adaptive: $id("engineAdaptive").checked, engine_path: $id("enginePath").value || null,
    });
  }
  try {
    renderStatus(await api("/api/training/start", body));
    log(`Starting ${body.preset} training…`);
  } catch (err) { $id("trainError").textContent = err.message; }
});
for (const [id, action] of [["btnPause", "pause"], ["btnResume", "resume"], ["btnStop", "stop"]]) {
  $id(id).addEventListener("click", async () => {
    try { await api(`/api/training/${action}`, {}); } catch (err) { $id("trainError").textContent = err.message; }
  });
}

// ---------------------------------------------------------------- live board

const liveBoard = Chessboard("liveBoard", { position: "start", pieceTheme: PIECES });

function renderLive(ev) {
  if (!ev) return;
  liveBoard.position(ev.fen, false);
  const lm = ev.last_move;
  highlightSquares("liveBoard", "sq-last", lm ? [lm.slice(0, 2), lm.slice(2, 4)] : []);
  $id("liveInfo").textContent = `${ev.opponent ? ev.opponent + " · " : ""}ply ${ev.ply}, ${ev.turn} to move`;
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
      renderLive(ev.live);
      break;
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
  const sel = $id("playCkpt");
  const keep = sel.value;
  sel.innerHTML = ckpts.length ? ckpts.map((c) => `<option value="${c.path}">${c.label}</option>`).join("")
    : `<option value="">No checkpoints yet – train first</option>`;
  if ([...sel.options].some((o) => o.value === keep)) sel.value = keep;
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
  $id("anaInfo").textContent = `${label} · ${a.sims} sims in ${a.seconds.toFixed(1)}s`;
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

// ---------------------------------------------------------------- start

async function init() {
  try {
    state.info = await api("/api/info");
    updatePresetHint();
    updateEngineFields();
    renderStatus(state.info.status);
    renderLive(state.info.live);
  } catch (err) { log(`Server not reachable: ${err.message}`); }
  await refreshRuns(state.status.run);
  refreshCheckpoints();
  renderPlay();
  connect();
}
init();
