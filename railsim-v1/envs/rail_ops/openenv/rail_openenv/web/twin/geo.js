// The line on the map: routes (OSM track paths built by tools/twin/build_twin.py) and where a network km lies on them.
//
// A route is one direction's lane from a terminal track to the other terminal track, through the platform tracks that
// direction uses in the record. Every network point has an anchor (its distance along the route), so km -> place is
// piecewise linear between anchors. A train that stands on the other track at a station (a turned train, or a second
// train there) blends onto the other direction's route inside that station's zone, straight across to it: fully
// between the switches that bound a loop, with a ramp where the zone ends on double track (a crossover).
//
// Frame: metres east (x) and north (y) of the twin origin. No DOM and no three.js here (node-testable).

export const CANONICAL = { E: "E-8-5a", W: "W-5a-8" };
const RAMP_SWITCH = 0.03;    // km: zones bounded by a switch, where the two tracks meet
const RAMP_CROSS = 0.25;     // km: zones that end on double track
const END_RUN = 20;          // m: the stretch whose direction a route keeps past its ends

export function smoothstep(u) {
  const x = Math.min(1, Math.max(0, u));
  return x * x * (3 - 2 * x);
}

export class Route {
  constructor(key, data, kmOf) {
    this.key = key;
    this.dir = data.dir;
    const p = data.p;
    const n = p.length / 2;
    this.x = new Float64Array(n);
    this.y = new Float64Array(n);
    this.s = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      this.x[i] = p[2 * i] / 10;
      this.y[i] = p[2 * i + 1] / 10;
      if (i) this.s[i] = this.s[i - 1] + Math.hypot(this.x[i] - this.x[i - 1], this.y[i] - this.y[i - 1]);
    }
    this.length = this.s[n - 1];
    // anchors sorted by km (the route's s grows with km eastbound, shrinks westbound)
    this.anchors = Object.entries(data.anchors)
      .filter(([c]) => kmOf[c] !== undefined)
      .map(([c, s]) => [kmOf[c], s])
      .sort((a, b) => a[0] - b[0]);
  }

  sAtKm(km) {
    const a = this.anchors;
    if (km <= a[0][0]) return a[0][1] + (km - a[0][0]) * 1000 * (this.dir === "E" ? 1 : -1);
    for (let i = 1; i < a.length; i++) {
      if (km <= a[i][0]) {
        const [k0, s0] = a[i - 1], [k1, s1] = a[i];
        return s0 + (s1 - s0) * (km - k0) / (k1 - k0);
      }
    }
    const [kl, sl] = a[a.length - 1];
    return sl + (km - kl) * 1000 * (this.dir === "E" ? 1 : -1);
  }

  kmAtS(s) {
    const a = this.dir === "E" ? this.anchors : [...this.anchors].reverse();
    if (s <= a[0][1]) return a[0][0] + (s - a[0][1]) / 1000 * (this.dir === "E" ? 1 : -1);
    for (let i = 1; i < a.length; i++) {
      if (s <= a[i][1]) {
        const [k0, s0] = a[i - 1], [k1, s1] = a[i];
        return k0 + (k1 - k0) * (s - s0) / (s1 - s0);
      }
    }
    const [kl, sl] = a[a.length - 1];
    return kl + (s - sl) / 1000 * (this.dir === "E" ? 1 : -1);
  }

  // [x, y] at distance s. Routes end at the terminals' stop points, so a train standing there reaches past an end by
  // half its length: past either end the track goes on straight, along its last END_RUN metres. (Clamping folded
  // those cars back onto the end point.)
  pointAtS(s) {
    const S = this.s;
    const n = S.length;
    if (s < 0 || s > this.length) {
      const atEnd = s > this.length;
      const [ex, ey] = atEnd ? [this.x[n - 1], this.y[n - 1]] : [this.x[0], this.y[0]];
      const [bx, by] = this.pointAtS(atEnd ? Math.max(0, this.length - END_RUN) : Math.min(this.length, END_RUN));
      const L = Math.hypot(ex - bx, ey - by);
      if (L < 1e-6) return [ex, ey];
      const d = atEnd ? s - this.length : -s;
      return [ex + (ex - bx) / L * d, ey + (ey - by) / L * d];
    }
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (S[mid] <= s) lo = mid; else hi = mid;
    }
    const L = S[hi] - S[lo];
    const t = L > 0 ? (s - S[lo]) / L : 0;
    return [this.x[lo] + t * (this.x[hi] - this.x[lo]), this.y[lo] + t * (this.y[hi] - this.y[lo])];
  }

  pointAtKm(km) {
    return this.pointAtS(this.sAtKm(km));
  }

  // the point of the route nearest to p, among the segments within `window` metres of sGuess along it (the spot
  // directly across from p when p lies on a parallel track)
  project(p, sGuess, window = 40) {
    const S = this.s, n = S.length;
    const lo = Math.max(0, this.segmentAt(sGuess - window)), hi = Math.min(n - 2, this.segmentAt(sGuess + window));
    let best = null, bd = Infinity;
    for (let i = lo; i <= hi; i++) {
      const ax = this.x[i], ay = this.y[i], dx = this.x[i + 1] - ax, dy = this.y[i + 1] - ay;
      const L2 = dx * dx + dy * dy;
      const u = L2 > 0 ? Math.max(0, Math.min(1, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2)) : 0;
      const qx = ax + u * dx, qy = ay + u * dy, d = Math.hypot(p[0] - qx, p[1] - qy);
      if (d < bd) { bd = d; best = [qx, qy]; }
    }
    return best || this.pointAtS(sGuess);
  }

  // index of the segment [i, i + 1] that holds distance s (clamped to the route)
  segmentAt(s) {
    const S = this.s;
    let lo = 0, hi = S.length - 1;
    if (s <= S[0]) return 0;
    if (s >= S[hi]) return hi - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (S[mid] <= s) lo = mid; else hi = mid;
    }
    return lo;
  }
}

export class Line {
  // twin: the decoded twin.json; points: [{code, km}] from the replay (the network)
  constructor(twin, points) {
    this.twin = twin;
    this.points = points;
    this.kmOf = Object.fromEntries(points.map((p) => [p.code, p.km]));
    this.routes = {};
    for (const [k, d] of Object.entries(twin.routes)) this.routes[k] = new Route(k, d, this.kmOf);
    this.zones = twin.zones || {};
    const ends = Object.entries(this.kmOf).sort((a, b) => a[1] - b[1]);
    // each terminal: its code and km, its platform tracks, and the direction sets leave it in
    this.ends = [{ code: ends[0][0], km: ends[0][1], side: "W", tracks: ["6b", "8"], leave: "E" },
      { code: ends[ends.length - 1][0], km: ends[ends.length - 1][1], side: "E", tracks: ["5a", "6a"], leave: "W" }];
    this.extra = {};                                    // routes into the drawn sidings, made when first asked for
  }

  routeKey(dir, lw, gn) {
    return dir === "E" ? `E-${lw}-${gn}` : `W-${gn}-${lw}`;
  }

  route(dir, lw, gn) {
    const key = this.routeKey(dir, lw, gn);
    if (this.routes[key]) return this.routes[key];
    if (this.extra[key] === undefined) this.extra[key] = this.sidingRoute(dir, lw, gn);
    return this.extra[key] || this.routes[CANONICAL[dir]];
  }

  // a terminal's route out along the line, from one of its platform tracks (s grows away from the terminal)
  outRoute(end, track) {
    return end.side === "W" ? this.route("E", track, "5a") : this.route("W", "8", track);
  }

  // weight in [0, 1] of the other track at km, for a train standing on track `lane` at station `code`
  blendWeight(code, km) {
    const z = this.zones[code];
    if (!z) return 0;
    const [k0, k1, sw0, sw1] = z;
    if (km < k0 || km > k1) return 0;
    const r0 = sw0 ? RAMP_SWITCH : RAMP_CROSS, r1 = sw1 ? RAMP_SWITCH : RAMP_CROSS;
    return Math.min(smoothstep((km - k0) / r0), smoothstep((k1 - km) / r1));
  }

  // [x, y] of a point km along a train's route; lanes: {station: "E" | "W"} where it uses that track
  place(dir, lw, gn, km, lanes) {
    const own = this.route(dir, lw, gn);
    const p = own.pointAtKm(km);
    if (!lanes) return p;
    for (const [code, lane] of Object.entries(lanes)) {
      if (lane === dir) continue;
      const w = this.blendWeight(code, km);
      if (w <= 0) continue;
      // straight across to the other track: the two directions' km scales are anchored at their own platforms and
      // drift up to 16 m apart along the line, so blending towards the other track's same-km point pulled the ends of
      // a train crossing over by different amounts (a car stretched by up to 3.3 m)
      const other = this.routes[CANONICAL[lane]];
      const q = other.project(p, other.sAtKm(km));
      return [p[0] + w * (q[0] - p[0]), p[1] + w * (q[1] - p[1])];
    }
    return p;
  }

  // what motion.js buildSim needs from the map (its opts): slotKm, the km between two sets standing one behind the
  // other at a terminal, `slotM` metres apart on the map (near Leeuwarden a network km is 0.77 km of OSM track);
  // slots, how many sets each terminal track holds clear of the switch where it meets the other one; yard, each
  // terminal's sidings and where their first set stands (km out from the terminal); zones
  simOpts(slotM, trainM) {
    const slotKm = {}, slots = {}, yard = {};
    for (const end of this.ends) {
      slotKm[end.code] = 0;
      for (const tr of end.tracks) {
        const r = this.outRoute(end, tr), s0 = r.sAtKm(end.km);
        slotKm[end.code] = Math.max(slotKm[end.code], Math.abs(r.kmAtS(s0 + slotM) - end.km));
        slots[`${end.code}:${tr}`] = Math.max(1, Math.floor((this.clearOf(end, tr) - 5 - trainM / 2) / slotM) + 1);
      }
      const y = this.yards()[end.code];
      yard[end.code] = y ? y.sidings.map((g) => ({ track: g.track, km: g.km })) : [];
      for (const g of y ? y.sidings : []) slots[`${end.code}:${g.track}`] = YARD.sets;
    }
    return { slotKm, slots, yard, zones: this.zones, turn: this.turnShifts() };
  }

  // {"station:came in:lane": km}: where a set that came in one way and stands on `lane` sits on the other direction's
  // km scale, less the station's km (each direction's scale is anchored at its own platform)
  turnShifts() {
    const out = {};
    for (const p of this.points) {
      if (!p.turn || p.tracks !== 2) continue;
      for (const came of ["E", "W"]) {
        const leaves = came === "E" ? "W" : "E";
        for (const lane of ["E", "W"]) {
          const at = this.place(came, "8", "5a", p.km, { [p.code]: lane });
          let best = 0, bd = Infinity;
          for (let d = -0.03; d <= 0.03; d += 0.0002) {
            const q = this.place(leaves, "8", "5a", p.km + d, { [p.code]: lane });
            const dist = Math.hypot(q[0] - at[0], q[1] - at[1]);
            if (dist < bd) { bd = dist; best = d; }
          }
          out[`${p.code}:${came}:${lane}`] = Math.round(best * 1e4) / 1e4;
        }
      }
    }
    return out;
  }

  // metres from a terminal track's stop point to the switch where it meets the terminal's other track
  clearOf(end, tr) {
    const r = this.outRoute(end, tr), s0 = r.sAtKm(end.km);
    const other = this.outRoute(end, end.tracks.find((x) => x !== tr));
    let free = 0;
    for (let d = 0; d < 2000; d += 2) {
      const p = r.pointAtS(s0 + d), q = other.project(p, other.sAtKm(r.kmAtS(s0 + d)), 80);
      if (Math.hypot(p[0] - q[0], p[1] - q[1]) < 1) break;
      free = d;
    }
    return free;
  }

  // the drawn yard at each terminal ({code: {groups: [{side, uBuf, uSw}], sidings: [{track, D, R, km, y}]}}), laid once
  yards() {
    if (!this._yards) {
      this._yards = {};
      for (const end of this.ends) this._yards[end.code] = layYard(this, end);
    }
    return this._yards;
  }

  // the sidings as track to draw: [{p (decimetres), k: 1, yard: true, end: [x, y, heading of the buffers]}]
  yardTracks() {
    const out = [];
    for (const end of this.ends) {
      const y = this.yards()[end.code];
      if (!y) continue;
      for (const g of y.sidings) {
        const pts = [];
        for (let u = g.y.uBuf - 6; u <= g.y.uSw; u += 2) pts.push(sidingPoint(this, end, g.y, g, u));
        const [a, b] = [pts[0], pts[1]];
        out.push({ p: pts.flatMap(([x, yy]) => [Math.round(x * 10), Math.round(yy * 10)]), k: 1, yard: true,
          end: [a[0], a[1], Math.atan2(a[1] - b[1], a[0] - b[0])] });
      }
    }
    return out;
  }

  // a route that starts or ends in a siding: the route of the terminal's first platform track, taken off the line
  // at the yard's switch and along the siding (null when there is no such siding)
  sidingRoute(dir, lw, gn) {
    const y = this.yards();
    const [west, east] = this.ends;
    const gW = y[west.code] && y[west.code].sidings.find((g) => g.track === lw);
    const gE = y[east.code] && y[east.code].sidings.find((g) => g.track === gn);
    if (!gW && !gE) return null;
    const baseKey = this.routeKey(dir, gW ? west.tracks[1] : lw, gE ? east.tracks[0] : gn);
    const r = this.routes[baseKey];
    if (!r) return null;
    // the base route's vertices, with each siding end's stretch replaced by points every 2 m along the siding
    const spans = [];
    for (const [end, g] of [[west, gW], [east, gE]]) {
      if (!g) continue;
      const sT = r.sAtKm(end.km), sgn = (end.side === "W") === (r.dir === "E") ? 1 : -1;
      const yd = g.y, pts = [];
      for (let u = 0; u <= yd.uSw + 4; u += 2) pts.push([sT + sgn * u, sidingPoint(this, end, yd, g, u)]);
      spans.push({ lo: Math.min(sT, sT + sgn * (yd.uSw + 4)), hi: Math.max(sT, sT + sgn * (yd.uSw + 4)), pts });
    }
    const verts = [];
    for (let i = 0; i < r.x.length; i++) {
      if (!spans.some((sp) => r.s[i] >= sp.lo && r.s[i] <= sp.hi)) verts.push([r.s[i], [r.x[i], r.y[i]]]);
    }
    for (const sp of spans) verts.push(...sp.pts);
    verts.sort((a, b) => a[0] - b[0]);
    // the new distances along it, and every anchor carried over from the base route's distance to the new one
    const sNew = [0];
    for (let i = 1; i < verts.length; i++) {
      const [p, q] = [verts[i - 1][1], verts[i][1]];
      sNew.push(sNew[i - 1] + Math.hypot(q[0] - p[0], q[1] - p[1]));
    }
    const map = (sb) => {
      let i = 1;
      while (i < verts.length - 1 && verts[i][0] < sb) i++;
      const [s0, s1] = [verts[i - 1][0], verts[i][0]];
      return sNew[i - 1] + (sNew[i] - sNew[i - 1]) * (s1 > s0 ? (sb - s0) / (s1 - s0) : 0);
    };
    const anchors = {};
    for (const [code, km] of Object.entries(this.kmOf)) {
      const a = r.anchors.find(([k]) => k === km);
      if (a) anchors[code] = map(a[1]);
    }
    const p = verts.flatMap(([, [x, yy]]) => [x * 10, yy * 10]);
    return new Route(this.routeKey(dir, lw, gn), { dir: r.dir, p, anchors }, this.kmOf);
  }

  // points along a train: [[x, y], ...] at the given distances (metres) ahead (+) or behind (-) its centre, which
  // is where the simulation puts it (a stopped train's centre is at the platform's stop point)
  body(dir, lw, gn, kmCentre, offsets, lanes) {
    const r = this.route(dir, lw, gn);
    const s0 = r.sAtKm(kmCentre);
    return offsets.map((d) => this.place(dir, lw, gn, r.kmAtS(s0 + d), lanes));
  }
}

// --- the drawn yards --------------------------------------------------------------------------------------------
// The rules give a terminal room for every set, but the line reaches only two platform tracks at each end (Leeuwarden
// 6b and 8, Groningen 5a and 6a). A set the platforms cannot take waits in a small yard the replay draws beside the
// line just outside the terminal: YARD.n stub sidings, each holding YARD.sets sets, fanning out of the line at one
// switch per group, on the nearest stretch past the platform tracks that is clear of buildings, water, roads,
// platforms and other track (as many side by side as fit there, the rest in a second group further out).
// (Leeuwarden's real yard lies beyond the platforms and is reached only through 6b, so it cannot take a set while the
// platforms are full.) The sidings are not on the map; SOURCES.md says so.
export const YARD = { n: 3, sets: 2, firsts: [4.5, 9, 13.5], spacing: 4.5, clear: 2, reach: 2500 };
const HOLD = 130;                    // metres of straight siding two sets need: 5 + 55 + 10 + 55 + 5
const ROAD_W = [22, 13, 10, 8, 5.5, 4, 3, 2.6], WATERWAY_W = [22, 14, 4, 1.8];

// where a siding lies u metres out from the terminal's stop point, along the line: the line moved sideways by the
// siding's offset, easing in over its ramp to the switch (a smoothstep); closer in than the yard's buffers it runs on
// straight (no train goes there; the route needs somewhere to start)
function sidingPoint(line, end, y, g, u) {
  y = y || g.y;
  const r = line.outRoute(end, end.side === "W" ? end.tracks[1] : end.tracks[0]), s0 = r.sAtKm(end.km);
  const uE = y.uBuf - 40;
  if (u < uE) {
    const a = sidingPoint(line, end, y, g, uE), b = sidingPoint(line, end, y, g, uE + 2), L = Math.hypot(a[0] - b[0], a[1] - b[1]);
    return [a[0] + (a[0] - b[0]) / L * (uE - u), a[1] + (a[1] - b[1]) / L * (uE - u)];
  }
  const p = r.pointAtS(s0 + u), p0 = r.pointAtS(s0 + u - 1), p1 = r.pointAtS(s0 + u + 1);
  const dx = p1[0] - p0[0], dy = p1[1] - p0[1], L = Math.hypot(dx, dy) || 1;
  const off = u >= y.uSw ? 0 : u <= y.uSw - g.R ? g.D : g.D * smoothstep((y.uSw - u) / g.R);
  return [p[0] - dy / L * off * y.side, p[1] + dx / L * off * y.side];
}

// lay the yard at a terminal: try each side of the line, nearest stretch first, and keep the first that is clear
function layYard(line, end) {
  const twin = line.twin;
  if (!twin.buildings || !twin.roads) return null;
  const r = line.outRoute(end, end.side === "W" ? end.tracks[1] : end.tracks[0]), s0 = r.sAtKm(end.km);
  const uMin = Math.max(...end.tracks.map((tr) => line.clearOf(end, tr))) + 25;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let u = 0; u <= YARD.reach + 400; u += 20) {
    const [x, y] = r.pointAtS(s0 + u);
    x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
  }
  const ob = new Obstacles(twin, x0 - 80, y0 - 80, x1 + 80, y1 + 80);
  // a siding is clear if nothing stands on it or within YARD.clear of it; where it still runs on the line's
  // formation (near its switch) it crosses what the line crosses, and on its way out it may cross other track (as
  // yard tracks do), but its straight part keeps clear of other track
  const clear = (y, g) => {
    for (let u = y.uBuf - 6; u <= y.uSw; u += 3) {
      const [px, py] = sidingPoint(line, end, y, g, u), [qx, qy] = sidingPoint(line, end, y, g, u + 1);
      const L = Math.hypot(qx - px, qy - py) || 1, nx = -(qy - py) / L, ny = (qx - px) / L;
      const off = u >= y.uSw ? 0 : u <= y.uSw - g.R ? g.D : g.D * smoothstep((y.uSw - u) / g.R);
      for (const w of [-YARD.clear, 0, YARD.clear]) {
        if (off + w * y.side < 4.5) continue;
        if (ob.blocked(px + nx * w, py + ny * w, w === 0 && u <= y.uSw - g.R)) return false;
      }
    }
    return true;
  };
  // the nearest group of k sidings side by side, clear of the groups already laid on that side
  const groups = [];
  const nearest = (k) => {
    let best = null;
    for (const side of [1, -1]) {
      for (const first of YARD.firsts) {
        const sidings = Array.from({ length: k }, (_, i) => ({ D: first + i * YARD.spacing }));
        for (const g of sidings) g.R = 40 + 4 * g.D;
        const span = HOLD + sidings[k - 1].R;
        for (let uBuf = uMin; uBuf <= YARD.reach && (!best || uBuf < best.uBuf); uBuf += 10) {
          if (groups.some((o) => o.side === side && uBuf - 10 < o.uSw && o.uBuf < uBuf + span + 10)) continue;
          const y = { side, uBuf, uSw: uBuf + span, sidings };
          if (sidings.every((g) => clear(y, g))) { best = y; break; }
        }
      }
    }
    return best;
  };
  // as many sidings together as fit nearest the terminal, then the rest in another group
  for (let left = YARD.n; left > 0;) {
    let y = null;
    for (let k = left; k >= 1 && !y; k--) y = nearest(k);
    if (!y) break;
    groups.push(y);
    left -= y.sidings.length;
  }
  if (!groups.length) return null;
  const sidings = [];
  for (const y of groups) {
    for (const g of y.sidings) {
      g.y = y;
      g.track = `Y${sidings.length + 1}`;
      // network km out from the terminal where the siding's first set stands (its centre, clear of the buffers)
      g.km = Math.abs(r.kmAtS(s0 + y.uBuf + 5 + 27.5) - end.km);
      sidings.push(g);
    }
  }
  return { groups, sidings };
}

// what a siding must keep clear of, on a grid: buildings, water and platforms (inside), roads, waterways and other
// track (too near the centre line)
class Obstacles {
  constructor(twin, x0, y0, x1, y1, cell = 40) {
    Object.assign(this, { x0, y0, x1, y1, cell, polys: new Map(), segs: new Map() });
    const dec = (f) => {
      const out = [];
      for (let i = 0; i < f.length; i += 2) out.push([f[i] / 10, f[i + 1] / 10]);
      return out;
    };
    const tag = (ring, kind) => Object.assign(ring, { kind });
    for (const f of twin.buildings.p) this.poly(tag(dec(f), "building"));
    for (const w of twin.water || []) this.poly(tag(dec(w.o), "water"));
    for (const pl of twin.platforms || []) this.poly(tag(dec(pl.p), "platform"));
    for (const rd of twin.roads) this.line(dec(rd.p), (ROAD_W[rd.c] ?? 4) / 2 + 1, false, `road${rd.c}`);
    for (const w of twin.waterways || []) this.line(dec(w.p), (WATERWAY_W[w.c] ?? 2) / 2 + 1, false, "waterway");
    for (const t of twin.tracks) this.line(dec(t.p), 4.4, true, "track");
  }

  cells(ax, ay, bx, by, pad) {
    const out = [];
    const c0 = Math.floor((Math.max(Math.min(ax, bx) - pad, this.x0)) / this.cell);
    const c1 = Math.floor((Math.min(Math.max(ax, bx) + pad, this.x1)) / this.cell);
    const r0 = Math.floor((Math.max(Math.min(ay, by) - pad, this.y0)) / this.cell);
    const r1 = Math.floor((Math.min(Math.max(ay, by) + pad, this.y1)) / this.cell);
    for (let c = c0; c <= c1; c++) for (let r = r0; r <= r1; r++) out.push(`${c},${r}`);
    return out;
  }

  poly(ring) {
    if (ring.length < 3) return;
    let ax = Infinity, ay = Infinity, bx = -Infinity, by = -Infinity;
    for (const [x, y] of ring) { ax = Math.min(ax, x); ay = Math.min(ay, y); bx = Math.max(bx, x); by = Math.max(by, y); }
    if (bx < this.x0 || ax > this.x1 || by < this.y0 || ay > this.y1) return;
    for (const k of this.cells(ax, ay, bx, by, 0)) {
      if (!this.polys.has(k)) this.polys.set(k, []);
      this.polys.get(k).push(ring);
    }
  }

  line(pts, half, track, kind) {
    for (let i = 0; i + 1 < pts.length; i++) {
      const [a, b] = [pts[i], pts[i + 1]];
      if (Math.max(a[0], b[0]) + half < this.x0 || Math.min(a[0], b[0]) - half > this.x1
        || Math.max(a[1], b[1]) + half < this.y0 || Math.min(a[1], b[1]) - half > this.y1) continue;
      for (const k of this.cells(a[0], a[1], b[0], b[1], half)) {
        if (!this.segs.has(k)) this.segs.set(k, []);
        this.segs.get(k).push([a, b, half, track, kind]);
      }
    }
  }

  // is (x, y) taken? tracks count only where the siding is off the line (near its switch it runs onto the line)
  blocked(x, y, tracks) {
    return !!this.why(x, y, tracks);
  }

  why(x, y, tracks) {
    const k = `${Math.floor(x / this.cell)},${Math.floor(y / this.cell)}`;
    for (const ring of this.polys.get(k) || []) if (insideRing(x, y, ring)) return ring.kind || "area";
    for (const [a, b, half, track, kind] of this.segs.get(k) || []) {
      if (track && !tracks) continue;
      const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy;
      const t = L2 > 0 ? Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / L2)) : 0;
      if (Math.hypot(x - a[0] - t * dx, y - a[1] - t * dy) < half) return kind || (track ? "track" : "line");
    }
    return null;
  }
}

function insideRing(x, y, ring) {
  let c = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
  }
  return c;
}

// Buildings the trains' track runs through: OSM maps Groningen's train shed (building=train_station) and buildings
// that bridge the line as footprints over the rails. They are drawn from this clearance up, so trains pass under them
// (drawn solid from the ground, they swallowed standing trains and filled the cab view with a wall).
export const CLEARANCE = 5.5;

export function overTrack(twin, step = 4, cell = 50) {
  const grid = new Map();
  for (const r of Object.values(twin.routes)) {
    const P = [];
    for (let i = 0; i < r.p.length; i += 2) P.push([r.p[i] / 10, r.p[i + 1] / 10]);
    for (let i = 0; i + 1 < P.length; i++) {
      const [ax, ay] = P[i], [bx, by] = P[i + 1];
      const n = Math.max(1, Math.floor(Math.hypot(bx - ax, by - ay) / step));
      for (let j = 0; j < n; j++) {
        const x = ax + (bx - ax) * j / n, y = ay + (by - ay) * j / n;
        const key = `${Math.floor(x / cell)},${Math.floor(y / cell)}`;
        let list = grid.get(key);
        if (!list) grid.set(key, (list = []));
        list.push(x, y);
      }
    }
  }
  const inside = (x, y, ring) => {
    let c = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i], [xj, yj] = ring[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
    }
    return c;
  };
  const out = new Set();
  const B = twin.buildings;
  for (let k = 0; k < B.p.length; k++) {
    const ring = [];
    for (let i = 0; i < B.p[k].length; i += 2) ring.push([B.p[k][i] / 10, B.p[k][i + 1] / 10]);
    if (ring.length < 3) continue;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y] of ring) {
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    }
    search:
    for (let cx = Math.floor(x0 / cell); cx <= Math.floor(x1 / cell); cx++) {
      for (let cy = Math.floor(y0 / cell); cy <= Math.floor(y1 / cell); cy++) {
        const list = grid.get(`${cx},${cy}`);
        if (!list) continue;
        for (let i = 0; i < list.length; i += 2) {
          const x = list[i], y = list[i + 1];
          if (x >= x0 && x <= x1 && y >= y0 && y <= y1 && inside(x, y, ring)) { out.add(k); break search; }
        }
      }
    }
  }
  return out;
}
