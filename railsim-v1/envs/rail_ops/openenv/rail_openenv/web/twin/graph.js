// The two charts under the map, on 2D canvases: the train graph (time across, distance down: every train's path,
// disruptions shaded, the moment being shown) with the plan's events on a strip below it, and the cost so far
// against the optimum's. Both are cursors on the same clock: click or drag to move it, hover to read a value.

import { hhmm } from "./motion.js";

const INK = { primary: "#ffffff", secondary: "#c3c2b7", muted: "#898781", grid: "#2c2c2a", axis: "#383835",
  surface: "#1a1a19" };
export const SERIES = { stop: "#3987e5", fast: "#d95926", plan: "#199e70" };
const STATUS = { critical: "#d03b3b", warning: "#fab219", serious: "#ec835a", good: "#0ca30c" };
const GLYPH = { hold: "‖", turn: "↩", cancel: "✕", violation: "!", closure: "▼", reopen: "▲" };

function fit(canvas) {
  const r = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
  }
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [g, r.width, r.height];
}

export class TrainGraph {
  constructor(canvas, tip, onSeek, onPick) {
    this.c = canvas;
    this.tip = tip;
    this.onSeek = onSeek;
    this.onPick = onPick;
    this.R = null;
    this.t = 0;
    this.sel = null;
    this.hover = null;
    let drag = false;
    canvas.addEventListener("pointerdown", (e) => { drag = true; canvas.setPointerCapture(e.pointerId); this.seekAt(e, true); });
    canvas.addEventListener("pointermove", (e) => { if (drag) this.seekAt(e, false); this.hoverAt(e); });
    canvas.addEventListener("pointerup", () => { drag = false; });
    canvas.addEventListener("pointerleave", () => { this.hover = null; this.tip.hidden = true; this.draw(); });
  }

  set(R, t0, t1) {
    this.R = R;
    this.t0 = t0;
    this.t1 = t1;
    this.km = Object.fromEntries(R.points.map((p) => [p.code, p.km]));
    this.kmMax = Math.max(...R.points.map((p) => p.km));
    // each served run as a polyline of (minute, km)
    this.paths = R.runs.filter((r) => !r.cancelled).map((r) => {
      const pts = [];
      if (r.started && r.fixed_from !== null && r.fixed_from !== undefined) pts.push([r.fixed_from, this.km[r.route[r.fixed_to - 1]]]);
      const from = r.started ? r.fixed_to : r.first;
      for (let i = from; i <= r.last; i++) {
        const k = this.km[r.route[i]];
        if (r.arr[i] !== null && r.arr[i] !== undefined && !(i === from && !r.started)) pts.push([r.arr[i], k]);
        if (r.dep[i] !== null && r.dep[i] !== undefined && i < r.last) pts.push([r.dep[i], k]);
      }
      return { train: r.train, unit: r.unit, kind: r.kind, pts, bad: r.problems.length > 0 };
    });
    this.draw();
  }

  frame() {
    const [g, w, h] = fit(this.c);
    const L = 44, Rp = 10, T = 8, B = 30, strip = 16;
    return { g, w, h, L, Rp, T, B, strip, gh: h - T - B - strip,
      x: (t) => L + (t - this.t0) / (this.t1 - this.t0) * (w - L - Rp),
      tx: (x) => this.t0 + (x - L) / (w - L - Rp) * (this.t1 - this.t0),
      y: (km) => T + km / this.kmMax * (h - T - B - strip) };
  }

  seekAt(e, pick) {
    if (!this.R) return;
    const f = this.frame();
    const r = this.c.getBoundingClientRect();
    const t = Math.max(this.t0, Math.min(this.t1, f.tx(e.clientX - r.left)));
    this.onSeek(t);
    if (pick && this.hover) this.onPick(this.hover.unit);
  }

  hoverAt(e) {
    if (!this.R) return;
    const f = this.frame();
    const r = this.c.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = null, bd = 10;
    for (const p of this.paths) {
      for (let i = 1; i < p.pts.length; i++) {
        const [ta, ka] = p.pts[i - 1], [tb, kb] = p.pts[i];
        const ax = f.x(ta), ay = f.y(ka), bx = f.x(tb), by = f.y(kb);
        const dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy || 1;
        const u = Math.max(0, Math.min(1, ((mx - ax) * dx + (my - ay) * dy) / L2));
        const d = Math.hypot(mx - ax - u * dx, my - ay - u * dy);
        if (d < bd) { bd = d; best = { ...p, t: ta + u * (tb - ta), km: ka + u * (kb - ka) }; }
      }
    }
    this.hover = best;
    if (best) {
      const code = Object.entries(this.km).reduce((a, [c, k]) => (Math.abs(k - best.km) < Math.abs(a[1] - best.km) ? [c, k] : a), ["", 1e9]);
      const near = Math.abs(code[1] - best.km) < 0.3 ? `at ${code[0]}` : `km ${best.km.toFixed(1)}`;
      this.tip.innerHTML = `<b>${best.train}</b> ${best.kind === "fast" ? "fast" : "stopping"} · ${hhmm(best.t)} ${near}`;
      this.tip.style.left = `${Math.min(r.width - 180, mx + 12)}px`;
      this.tip.style.top = `${my - 28}px`;
      this.tip.hidden = false;
    } else {
      this.tip.hidden = true;
    }
    this.draw();
  }

  draw() {
    if (!this.R) return;
    const f = this.frame();
    const { g, w, h, L, Rp, T, B, strip } = f;
    g.clearRect(0, 0, w, h);
    g.font = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";
    // disruptions
    for (const c of this.R.closures) {
      const a = this.km[c.where[0]], b = this.km[c.where[c.where.length - 1]];
      const k0 = c.kind === "track" ? a - 0.4 : Math.min(a, b), k1 = c.kind === "track" ? a + 0.4 : Math.max(a, b);
      g.fillStyle = c.kind === "closed" ? STATUS.critical : c.kind === "slow" ? STATUS.warning : STATUS.serious;
      g.globalAlpha = 0.22;
      g.fillRect(f.x(c.start), f.y(k0), f.x(c.end) - f.x(c.start), Math.max(3, f.y(k1) - f.y(k0)));
      g.globalAlpha = 1;
    }
    // stations: hairlines and names
    g.textAlign = "right";
    g.textBaseline = "middle";
    let lastY = -Infinity;
    for (const p of this.R.points) {
      const y = f.y(p.km);
      g.strokeStyle = INK.grid;
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(L, y + 0.5); g.lineTo(w - Rp, y + 0.5); g.stroke();
      const major = p.kind === "terminal" || p.turn;              // terminals and turn points first
      if (p.kind !== "junction" && (y - lastY >= 11 || major)) {
        g.fillStyle = major ? INK.secondary : INK.muted;
        g.fillText(p.code, L - 6, y);
        lastY = y;
      }
    }
    // hours
    g.textAlign = "center";
    g.textBaseline = "top";
    const step = this.t1 - this.t0 > 300 ? 60 : 30;
    for (let t = Math.ceil(this.t0 / step) * step; t <= this.t1; t += step) {
      const x = f.x(t);
      g.strokeStyle = INK.grid;
      g.beginPath(); g.moveTo(x + 0.5, T); g.lineTo(x + 0.5, T + f.gh); g.stroke();
      g.fillStyle = INK.muted;
      g.fillText(hhmm(t), x, h - B + strip + 6);
    }
    // the window's start and end
    g.strokeStyle = INK.axis;
    for (const t of [this.R.start, this.R.end]) {
      const x = f.x(t);
      g.beginPath(); g.moveTo(x + 0.5, T); g.lineTo(x + 0.5, T + f.gh); g.stroke();
    }
    // train paths
    const focus = this.sel ?? (this.hover && this.hover.unit);
    for (const p of this.paths) {
      const on = !focus || p.unit === focus;
      g.strokeStyle = SERIES[p.kind] || SERIES.stop;
      g.globalAlpha = on ? 1 : 0.28;
      g.lineWidth = focus && p.unit === focus ? 3 : 1.5;
      g.lineJoin = "round";
      g.beginPath();
      p.pts.forEach(([t, k], i) => (i ? g.lineTo(f.x(t), f.y(k)) : g.moveTo(f.x(t), f.y(k))));
      g.stroke();
    }
    g.globalAlpha = 1;
    // rule breaks: a cross where and when each happens
    g.strokeStyle = STATUS.critical;
    g.lineWidth = 2;
    for (const v of this.R.violations) {
      if (!v.time || !v.where) continue;
      const [hh, mm] = v.time.split(":").map(Number);
      let t = hh * 60 + mm;
      if (t < this.R.start - 120) t += 1440;
      const ends = v.where.split("-").filter((c) => this.km[c] !== undefined);
      if (!ends.length) continue;
      const k = ends.reduce((a, c) => a + this.km[c], 0) / ends.length;
      const x = f.x(t), y = f.y(k);
      g.beginPath(); g.moveTo(x - 4, y - 4); g.lineTo(x + 4, y + 4); g.moveTo(x + 4, y - 4); g.lineTo(x - 4, y + 4); g.stroke();
    }
    // events strip
    const sy = T + f.gh + 4;
    g.textAlign = "center";
    g.textBaseline = "top";
    g.font = "bold 11px system-ui, -apple-system, 'Segoe UI', sans-serif";
    for (const ev of this.R.events) {
      if (ev.t === null || ev.t === undefined) continue;
      const x = f.x(ev.t);
      g.fillStyle = ev.kind === "violation" || ev.kind === "closure" ? STATUS.critical : ev.kind === "reopen" ? STATUS.good : INK.secondary;
      g.fillText(GLYPH[ev.kind] || "·", x, sy);
    }
    // the cursor
    const x = f.x(this.t);
    g.strokeStyle = INK.primary;
    g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(x, T); g.lineTo(x, T + f.gh + strip); g.stroke();
  }

  setTime(t) {
    this.t = t;
    this.draw();
  }
}

export class CostChart {
  constructor(canvas, tip, onSeek) {
    this.c = canvas;
    this.tip = tip;
    this.onSeek = onSeek;
    this.t = 0;
    canvas.addEventListener("pointermove", (e) => this.hoverAt(e));
    canvas.addEventListener("pointerleave", () => { this.tip.hidden = true; });
    canvas.addEventListener("pointerdown", (e) => {
      const f = this.frame(), r = canvas.getBoundingClientRect();
      this.onSeek(Math.max(this.t0, Math.min(this.t1, f.tx(e.clientX - r.left))));
    });
  }

  // series: [{name, costs: [[t, amount], ...], color, dashed}]
  set(series, t0, t1) {
    this.series = series.map((s) => {
      let sum = 0;
      const steps = [[t0, 0]];
      for (const [t, a] of s.costs) { sum += a; steps.push([t, sum]); }
      return { ...s, steps, total: sum };
    });
    this.t0 = t0;
    this.t1 = t1;
    this.max = Math.max(10, ...this.series.map((s) => s.total)) * 1.08;
    this.draw();
  }

  frame() {
    const [g, w, h] = fit(this.c);
    const L = 44, Rp = 92, T = 22, B = 22;
    return { g, w, h, L, Rp, T, B,
      x: (t) => L + (t - this.t0) / (this.t1 - this.t0) * (w - L - Rp),
      tx: (x) => this.t0 + (x - L) / (w - L - Rp) * (this.t1 - this.t0),
      y: (v) => h - B - v / this.max * (h - T - B) };
  }

  valueAt(s, t) {
    let v = 0;
    for (const [tt, sum] of s.steps) { if (tt > t) break; v = sum; }
    return v;
  }

  hoverAt(e) {
    if (!this.series) return;
    const f = this.frame(), r = this.c.getBoundingClientRect();
    const t = Math.max(this.t0, Math.min(this.t1, f.tx(e.clientX - r.left)));
    this.tip.innerHTML = `${hhmm(t)}<br>` + this.series.map((s) => `${s.name}: <b>${this.valueAt(s, t).toLocaleString()}</b>`).join("<br>");
    this.tip.style.left = `${Math.min(r.width - 150, e.clientX - r.left + 12)}px`;
    this.tip.style.top = "6px";
    this.tip.hidden = false;
  }

  draw() {
    if (!this.series) return;
    const f = this.frame();
    const { g, w, h, L, Rp, T, B } = f;
    g.clearRect(0, 0, w, h);
    g.font = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";
    g.fillStyle = INK.secondary;
    g.textAlign = "left";
    g.textBaseline = "top";
    g.fillText("Cost so far (stop-minutes)", L, 4);
    // y grid: three hairlines
    g.textAlign = "right";
    g.textBaseline = "middle";
    for (let i = 0; i <= 3; i++) {
      const v = (this.max / 1.08) * i / 3;
      const y = Math.round(f.y(v)) + 0.5;
      g.strokeStyle = i ? INK.grid : INK.axis;
      g.beginPath(); g.moveTo(L, y); g.lineTo(w - Rp, y); g.stroke();
      g.fillStyle = INK.muted;
      g.fillText(Math.round(v).toLocaleString(), L - 6, y);
    }
    // the series as step lines, with the total labelled at the end
    for (const s of this.series) {
      g.strokeStyle = s.color;
      g.lineWidth = 2;
      g.setLineDash(s.dashed ? [5, 4] : []);
      g.beginPath();
      let prev = null;
      for (const [t, v] of s.steps) {
        const x = f.x(Math.min(this.t1, t)), y = f.y(v);
        if (prev === null) g.moveTo(x, y); else { g.lineTo(x, prev); g.lineTo(x, y); }
        prev = y;
      }
      g.lineTo(f.x(this.t1), prev);
      g.stroke();
      g.setLineDash([]);
      g.fillStyle = INK.primary;
      g.textAlign = "left";
      g.fillText(`${s.name} ${s.total.toLocaleString()}`, w - Rp + 6, Math.max(T, Math.min(h - B, f.y(s.total))));
    }
    const x = f.x(this.t);
    g.strokeStyle = INK.primary;
    g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(x, T); g.lineTo(x, h - B); g.stroke();
  }

  setTime(t) {
    this.t = t;
    this.draw();
  }
}
