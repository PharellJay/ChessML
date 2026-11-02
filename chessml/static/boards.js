// Small canvas chessboards, cheap enough to show every running game at once
// (a chessboard.js widget per game would cost thousands of DOM nodes).

const LIGHT = "#f0d9b5", DARK = "#b58863";  // chessboard.js square colours
const LAST_MOVE = "rgba(237, 161, 0, 0.5)";
const PIECES = "/static/vendor/img/chesspieces/wikipedia/";

const images = {};
const ready = Promise.all([..."wb"].flatMap((c) => [..."KQRBNP"].map((p) => new Promise((resolve) => {
  const img = new Image();
  img.onload = img.onerror = resolve;
  img.src = `${PIECES}${c}${p}.png`;
  images[c + p] = img;
}))));

// Draw a position (the piece-placement part of a FEN) with the last move highlighted.
export function drawBoard(canvas, placement, lastMove) {
  const px = Math.round((canvas.clientWidth || 80) * (window.devicePixelRatio || 1));
  if (canvas.width !== px) { canvas.width = px; canvas.height = px; }
  const ctx = canvas.getContext("2d");
  const sq = px / 8;
  const marked = lastMove ? [lastMove.slice(0, 2), lastMove.slice(2, 4)] : [];
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      ctx.fillStyle = (r + f) % 2 ? DARK : LIGHT;
      ctx.fillRect(f * sq, r * sq, sq, sq);
      if (marked.includes("abcdefgh"[f] + (8 - r))) {
        ctx.fillStyle = LAST_MOVE;
        ctx.fillRect(f * sq, r * sq, sq, sq);
      }
    }
  }
  placement.split("/").forEach((row, r) => {
    let f = 0;
    for (const ch of row) {
      if (ch >= "1" && ch <= "8") { f += Number(ch); continue; }
      const img = images[(ch === ch.toUpperCase() ? "w" : "b") + ch.toUpperCase()];
      if (img && img.complete && img.naturalWidth) ctx.drawImage(img, f * sq, r * sq, sq, sq);
      f++;
    }
  });
}

// A grid of clickable mini boards. Items: {key, board, last_move, label, badge, title}.
export class MiniBoards {
  constructor(el, { onSelect, emptyText = "No games" } = {}) {
    this.el = el;
    this.emptyText = emptyText;
    this.items = [];
    this.selected = null;
    this.keys = null;
    el.addEventListener("click", (e) => {
      const tile = e.target.closest(".mini");
      if (tile && onSelect) onSelect(tile.dataset.key);
    });
    new ResizeObserver(() => this.draw()).observe(el);
    ready.then(() => this.draw());
  }

  update(items, selected) {
    this.items = items;
    this.selected = selected;
    const keys = items.map((g) => g.key).join(",");
    if (keys !== this.keys) {  // rebuild the tiles only when the set of games changes
      this.keys = keys;
      this.el.innerHTML = items.length ? items.map((g) =>
        `<button type="button" class="mini" data-key="${g.key}"><canvas></canvas>` +
        `<span class="mini-label"></span><span class="mini-badge"></span></button>`).join("")
        : `<p class="mini-empty">${this.emptyText}</p>`;
    }
    this.draw();
  }

  draw() {
    const tiles = this.el.querySelectorAll(".mini");
    this.items.forEach((g, i) => {
      const tile = tiles[i];
      if (!tile) return;
      tile.title = g.title || "";
      tile.classList.toggle("selected", g.key === this.selected);
      tile.querySelector(".mini-label").textContent = g.label || "";
      const badge = tile.querySelector(".mini-badge");
      badge.textContent = g.badge || "";
      badge.hidden = !g.badge;
      drawBoard(tile.querySelector("canvas"), g.board, g.last_move);
    });
  }
}
