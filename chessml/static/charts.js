// Small canvas charts: multi-series line charts with a crosshair tooltip, and a bar chart.
// Colors come from CSS custom properties so light and dark themes both work.

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function niceStep(range, target = 4) {
  const raw = range / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const norm = raw / mag;
  return (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
}

function fmt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (Math.abs(v) < 1e-9) return "0";
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

class BaseChart {
  constructor(el, opts) {
    this.el = el;
    this.opts = opts;
    el.innerHTML = "";
    if (opts.series && opts.series.length >= 2) {
      el.classList.add("with-legend");
      const legend = document.createElement("div");
      legend.className = "chart-legend";
      opts.series.forEach((s, i) => {
        const item = document.createElement("span");
        item.innerHTML = `<i style="background:var(--series-${i + 1})"></i>${s}`;
        legend.appendChild(item);
      });
      el.appendChild(legend);
    }
    this.canvas = document.createElement("canvas");
    this.tip = document.createElement("div");
    this.tip.className = "chart-tip";
    this.empty = document.createElement("div");
    this.empty.className = "chart-empty";
    this.empty.textContent = opts.emptyText || "No data yet";
    el.append(this.canvas, this.tip, this.empty);
    this.hover = null;
    this.pending = false;
    new ResizeObserver(() => this.schedule()).observe(el);
    this.canvas.addEventListener("mousemove", (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.hover = { x: e.clientX - r.left, y: e.clientY - r.top };
      this.schedule();
    });
    this.canvas.addEventListener("mouseleave", () => { this.hover = null; this.schedule(); });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.schedule());
  }

  schedule() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => { this.pending = false; this.draw(); });
  }

  setup() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px system-ui, sans-serif";
    return { ctx, w, h };
  }

  yAxis(ctx, box, y0, y1, yFmt) {
    const step = niceStep(y1 - y0);
    ctx.strokeStyle = cssVar("--grid");
    ctx.fillStyle = cssVar("--text-muted");
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let v = Math.ceil(y0 / step) * step; v <= y1 + 1e-9; v += step) {
      const y = box.y(v);
      ctx.beginPath(); ctx.moveTo(box.left, Math.round(y) + 0.5); ctx.lineTo(box.right, Math.round(y) + 0.5); ctx.stroke();
      ctx.fillText(yFmt ? yFmt(v) : fmt(v), box.left - 6, y);
    }
  }

  showTip(html, px, py, w) {
    this.tip.innerHTML = html;
    this.tip.style.display = "block";
    const tw = this.tip.offsetWidth;
    let left = px + 12;
    if (left + tw > w) left = px - tw - 12;
    this.tip.style.left = `${Math.max(0, left)}px`;
    this.tip.style.top = `${Math.max(0, py - 10)}px`;
  }

  hideTip() { this.tip.style.display = "none"; }
}

export class LineChart extends BaseChart {
  // opts: { series: [names], xLabel, xFmt, yFmt, yMin, yMax }
  constructor(el, opts) {
    super(el, opts);
    this.data = opts.series.map(() => []);
  }

  set(i, points) { this.data[i] = points; this.schedule(); }
  push(i, point, max = 5000) {
    const d = this.data[i];
    d.push(point);
    if (d.length > max) d.splice(0, d.length - max);
    this.schedule();
  }
  clear() { this.data = this.opts.series.map(() => []); this.schedule(); }

  draw() {
    const { ctx, w, h } = this.setup();
    const all = this.data.flat().filter((p) => p[1] !== null && !Number.isNaN(p[1]));
    this.empty.style.display = all.length ? "none" : "flex";
    if (!all.length) { this.hideTip(); return; }
    const o = this.opts;
    let x0 = Math.min(...all.map((p) => p[0])), x1 = Math.max(...all.map((p) => p[0]));
    let y0 = o.yMin ?? Math.min(...all.map((p) => p[1]));
    let y1 = o.yMax ?? Math.max(...all.map((p) => p[1]));
    if (o.yMin === undefined && y0 > 0 && y0 < (y1 - y0)) y0 = 0;
    if (x1 === x0) x1 = x0 + 1;
    if (y1 === y0) { y1 += 1; if (o.yMin === undefined) y0 -= 1; }
    const pad = (y1 - y0) * 0.06;
    if (o.yMax === undefined) y1 += pad;
    if (o.yMin === undefined && y0 !== 0) y0 -= pad;

    const box = { left: 46, right: w - 10, top: 8, bottom: h - 22 };
    box.x = (v) => box.left + ((v - x0) / (x1 - x0)) * (box.right - box.left);
    box.y = (v) => box.bottom - ((v - y0) / (y1 - y0)) * (box.bottom - box.top);
    this.yAxis(ctx, box, y0, y1, o.yFmt);

    // x axis labels
    ctx.fillStyle = cssVar("--text-muted");
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    let xs = niceStep(x1 - x0, 5);
    if (o.integerX) xs = Math.max(1, Math.round(xs));
    const labelRoom = o.xLabel ? ctx.measureText(o.xLabel).width + 24 : 0;
    for (let v = Math.ceil(x0 / xs) * xs; v <= x1 + 1e-9; v += xs) {
      if (box.x(v) > box.right - labelRoom) continue;  // keep clear of the axis name
      ctx.fillText(o.xFmt ? o.xFmt(v) : fmt(v), box.x(v), box.bottom + 6);
    }
    if (o.xLabel) {
      ctx.textAlign = "right";
      ctx.fillText(o.xLabel, box.right, box.bottom + 6);
    }

    // series
    this.data.forEach((pts, i) => {
      const valid = pts.filter((p) => p[1] !== null && !Number.isNaN(p[1]));
      if (!valid.length) return;
      ctx.strokeStyle = cssVar(`--series-${i + 1}`);
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.beginPath();
      valid.forEach((p, k) => (k ? ctx.lineTo(box.x(p[0]), box.y(p[1])) : ctx.moveTo(box.x(p[0]), box.y(p[1]))));
      ctx.stroke();
      if (valid.length === 1 || o.markers) {
        ctx.fillStyle = ctx.strokeStyle;
        valid.forEach((p) => { ctx.beginPath(); ctx.arc(box.x(p[0]), box.y(p[1]), 3, 0, Math.PI * 2); ctx.fill(); });
      }
    });

    // hover: crosshair + nearest point per series
    if (!this.hover || this.hover.x < box.left || this.hover.x > box.right) { this.hideTip(); return; }
    const xv = x0 + ((this.hover.x - box.left) / (box.right - box.left)) * (x1 - x0);
    ctx.strokeStyle = cssVar("--text-muted");
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(Math.round(this.hover.x) + 0.5, box.top); ctx.lineTo(Math.round(this.hover.x) + 0.5, box.bottom); ctx.stroke();
    const rows = [];
    let shownX = null;
    this.data.forEach((pts, i) => {
      const valid = pts.filter((p) => p[1] !== null && !Number.isNaN(p[1]));
      if (!valid.length) return;
      let best = valid[0];
      for (const p of valid) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
      shownX = shownX ?? best[0];
      const cx = box.x(best[0]), cy = box.y(best[1]);
      ctx.fillStyle = cssVar("--surface-1");
      ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = cssVar(`--series-${i + 1}`);
      ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
      rows.push(`<div><i style="background:var(--series-${i + 1})"></i>${o.series[i]}: <b>${o.yFmt ? o.yFmt(best[1]) : fmt(best[1])}</b></div>`);
    });
    const head = `${o.xLabel ? o.xLabel + " " : ""}${o.xFmt ? o.xFmt(shownX) : fmt(shownX)}`;
    this.showTip(`<div>${head}</div>${rows.join("")}`, this.hover.x, this.hover.y, w);
  }
}

export class BarChart extends BaseChart {
  // opts: { yMax, labelFmt, yFmt }
  constructor(el, opts) {
    super(el, opts);
    this.values = [];
  }

  set(values) { this.values = values || []; this.schedule(); }

  draw() {
    const { ctx, w, h } = this.setup();
    const vals = this.values;
    this.empty.style.display = vals.length ? "none" : "flex";
    if (!vals.length) { this.hideTip(); return; }
    const o = this.opts;
    const y0 = 0, y1 = o.yMax ?? Math.max(...vals, 1);
    const box = { left: 46, right: w - 10, top: 8, bottom: h - 22 };
    box.y = (v) => box.bottom - ((v - y0) / (y1 - y0)) * (box.bottom - box.top);
    this.yAxis(ctx, box, y0, y1, o.yFmt);
    const slot = (box.right - box.left) / vals.length;
    const bw = Math.max(2, Math.min(36, slot - 2));
    let hovered = -1;
    if (this.hover) hovered = Math.floor((this.hover.x - box.left) / slot);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    vals.forEach((v, i) => {
      const x = box.left + i * slot + (slot - bw) / 2;
      const top = box.y(Math.max(0, Math.min(y1, v)));
      const r = Math.min(4, bw / 2, box.bottom - top);
      ctx.fillStyle = cssVar("--series-1");
      ctx.globalAlpha = hovered === -1 || hovered === i ? 1 : 0.55;
      ctx.beginPath();
      ctx.moveTo(x, box.bottom);
      ctx.lineTo(x, top + r);
      ctx.quadraticCurveTo(x, top, x + r, top);
      ctx.lineTo(x + bw - r, top);
      ctx.quadraticCurveTo(x + bw, top, x + bw, top + r);
      ctx.lineTo(x + bw, box.bottom);
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;
      if (vals.length <= 16 || i % Math.ceil(vals.length / 16) === 0) {
        ctx.fillStyle = cssVar("--text-muted");
        ctx.fillText(String(i), x + bw / 2, box.bottom + 6);
      }
    });
    if (hovered >= 0 && hovered < vals.length) {
      const label = o.labelFmt ? o.labelFmt(hovered) : `#${hovered}`;
      this.showTip(`<div>${label}: <b>${o.yFmt ? o.yFmt(vals[hovered]) : fmt(vals[hovered])}</b></div>`, this.hover.x, this.hover.y, w);
    } else this.hideTip();
  }
}
