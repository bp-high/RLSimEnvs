// Trains in time: where every train set is at any moment under a plan, from the replay's simulated times
// (rail_core.replay, which uses the grader's own simulate). Nothing here changes a time: positions between the
// simulated minutes are a smooth monotone curve (stopped trains have zero speed, passing trains keep theirs).
//
// A train set's window is a chain of segments: it stands (at a station, a turn point, a terminal), moves (one run, or
// the part of it the plan keeps), stands again. Track allocation is for the picture only:
//   - terminals: the two tracks the line reaches (Leeuwarden 6b/8, Groningen 5a/6a, as in the record). A set takes
//     its type's usual track, or the other one, or stands behind a set already there: wherever it neither drives
//     through a set nor is driven through (a set that comes in while another stands on its track stands behind it,
//     and must leave first), and only as many deep as the track holds before its switch. The rules give a terminal
//     room for every set, so a plan can bring in a set where neither platform track has room (one set arrives as
//     another leaves with the other track taken): that set waits in the terminal's yard, sidings the replay draws
//     beside the line just outside (geo.js, opts.yard). Only if the yard is full too is it drawn faded while it would
//     share a track (`ghost`);
//   - two-track stations: each direction's own platform (as in the record); a train that finds it taken, or a turned
//     train, uses the other track. A set holds its track while it is in the station's zone (coming in and going out).
//
// No DOM and no three.js here (node-testable). Times are minutes after midnight of the service day; km as the network.

export const SLOT_M = 65;          // a second set on a terminal track stands this far (metres along the track) behind
export const SLOT_KM = SLOT_M / 1000;   // ... in network km where the map has no say (buildSim's default)
export const TRAIN_M = 55;         // a two-car unit, metres
const THROAT_KM = 0.4;             // a set this near a terminal is on its terminal track (or the switches in front)
const PLATFORM_KM = 0.07;          // a set this near a station's stop point is on (or entering) its platform track
const GHOST_RAMP = 0.4;            // minutes over which a set with no room at a terminal fades out (and back in)
const FADED = " · no room at the terminal's platforms or in its sidings: drawn faded";

export function lwTrack(run, arriving) {
  // record (Jan-Jun 2025): westbound arrivals fast -> 8, stopping -> 6b; eastbound departures fast -> 6b, stop -> 8
  if (arriving) return run.kind === "fast" ? "8" : "6b";
  return run.kind === "fast" ? "6b" : "8";
}

export function gnTrack(run) {
  return run.kind === "fast" ? "6a" : "5a";    // record: fast trains mostly 6a, stopping trains 5a
}

export function hhmm(m) {
  if (m === null || m === undefined || !Number.isFinite(m)) return "--:--";
  const t = Math.floor(m + 1e-9);
  return `${String(Math.floor(t / 60) % 24).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`;
}

// --- monotone cubic Hermite through (t, x) knots; stop knots have zero slope ---------------------------------

export function hermite(knots) {
  const n = knots.length;
  const t = knots.map((k) => k.t), x = knots.map((k) => k.x), m = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    if (knots[i].stop) continue;
    const d0 = i > 0 ? (x[i] - x[i - 1]) / (t[i] - t[i - 1]) : null;
    const d1 = i < n - 1 ? (x[i + 1] - x[i]) / (t[i + 1] - t[i]) : null;
    if (d0 === null) m[i] = d1;
    else if (d1 === null) m[i] = d0;
    else if (d0 * d1 <= 0) m[i] = 0;
    else {
      const h0 = t[i] - t[i - 1], h1 = t[i + 1] - t[i];
      const w1 = 2 * h1 + h0, w2 = h1 + 2 * h0;
      m[i] = (w1 + w2) / (w1 / d0 + w2 / d1);
    }
  }
  return { t, x, m };
}

export function evalHermite(H, tq) {
  const { t, x, m } = H;
  const n = t.length;
  if (n === 1 || tq <= t[0]) return [x[0], 0];
  if (tq >= t[n - 1]) return [x[n - 1], 0];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= tq) lo = mid; else hi = mid;
  }
  const h = t[hi] - t[lo];
  const u = (tq - t[lo]) / h, u2 = u * u, u3 = u2 * u;
  const pos = (2 * u3 - 3 * u2 + 1) * x[lo] + (u3 - 2 * u2 + u) * h * m[lo] + (-2 * u3 + 3 * u2) * x[hi]
    + (u3 - u2) * h * m[hi];
  const vel = ((6 * u2 - 6 * u) * x[lo] + (3 * u2 - 4 * u + 1) * h * m[lo] + (-6 * u2 + 6 * u) * x[hi]
    + (3 * u2 - 2 * u) * h * m[hi]) / h;
  return [pos, vel];
}

// --- the simulation picture ------------------------------------------------------------------------------------

// opts, from the map (Line.simOpts): slotKm {terminal code: km between sets standing one behind another} (the map's
// km scale differs from the network's near Leeuwarden), slots {"code:track": how many sets the track holds}, yard
// {terminal: [{track, km}]}, zones {station: [km from, km to, ...]} where a set on a station's other track is drawn on
// it, and turn {"station:came in:lane": km} where a set that turned back there starts its next run
export function buildSim(R, opts = {}) {
  const slotKm = opts.slotKm || {}, slots = opts.slots || {}, yard = opts.yard || {};
  const kmOf = Object.fromEntries(R.points.map((p) => [p.code, p.km]));
  const point = Object.fromEntries(R.points.map((p) => [p.code, p]));
  const codes = R.points.map((p) => p.code);
  const west = codes[0], east = codes[codes.length - 1];
  const runOf = new Map(R.runs.map((r) => [r.train, r]));
  const units = [];

  for (const u of R.units) {
    const runs = u.runs.map((id) => runOf.get(id)).filter(Boolean);
    const segs = [];
    let at = u.at, since = R.start, prev = null;
    for (const r of runs) {
      if (r.cancelled) continue;
      const s = r.first, e = r.last;
      const k = r.started ? r.fixed_to : s;
      let t0;
      if (r.started && r.fixed_from !== null && r.fixed_from !== undefined) {
        t0 = r.fixed_from;                                   // already running when the window starts
      } else {
        const where = r.route[k];
        const dep = r.dep[k];
        // an illegal plan can start a run where its train set is not (a continuity break): the set waits where it
        // is and jumps when the run leaves, as the grader's simulation has it
        const standAt = prev && at && at !== where ? at : where;
        segs.push({ type: "stand", code: standAt, t0: Math.min(since, dep), t1: dep, prev, next: r });
        t0 = dep;
      }
      segs.push({ type: "move", run: r, dir: r.dir, t0, t1: r.arr[e], k, s, e, lanes: {}, startKm: null, endKm: null });
      at = r.route[e];
      since = r.arr[e];
      prev = r;
    }
    if (at) segs.push({ type: "stand", code: at, t0: since, t1: Infinity, prev, next: null });
    units.push({ id: u.id, runs, segs });
  }

  // every move's curve as the plan times it, with terminal stands at the stop point: when is each set near a terminal
  for (const un of units) for (const sg of un.segs) if (sg.type === "move") shape(sg, kmOf);

  // terminals: which track, and which slot on it
  const terminal = (c) => c === west || c === east;
  const stays = [];
  for (const un of units) {
    un.segs.forEach((sg, i) => {
      if (sg.type !== "stand" || !terminal(sg.code)) return;
      const before = un.segs[i - 1], after = un.segs[i + 1];
      const into = before && before.type === "move" ? before : null, out = after && after.type === "move" ? after : null;
      const kT = kmOf[sg.code];
      let prefer;
      if (sg.code === west) prefer = sg.prev && sg.prev.dir === "W" ? lwTrack(sg.prev, true) : sg.next ? lwTrack(sg.next, false) : "8";
      else prefer = sg.prev && sg.prev.dir === "E" ? gnTrack(sg.prev) : sg.next ? gnTrack(sg.next) : "5a";
      // [A, a]: coming in (from where it is within THROAT_KM of the stop); [a, d]: standing; [d, D]: going out
      const a = into ? into.t1 : sg.t0, d = out ? out.t0 : sg.t1;
      stays.push({ sg, into, out, code: sg.code, kT, prefer, a, d,
        A: into ? near(into, kT - THROAT_KM, kT + THROAT_KM)[0] : sg.t0,
        D: out ? near(out, kT - THROAT_KM, kT + THROAT_KM)[1] : sg.t1 });
    });
    // a set already leaving a terminal when the window starts stood on one of its tracks until then
    un.segs.forEach((sg, i) => {
      if (sg.type !== "move") return;
      const x0 = sg.knots[0].x, before = un.segs[i - 1];
      const code = [west, east].find((c) => Math.abs(x0 - kmOf[c]) < 1e-9);
      if (!code || (before && before.type === "stand" && before.code === code)) return;
      const sg0 = (sg.from = { type: "stand", code, prev: null, next: sg.run });
      const kT = kmOf[code];
      stays.push({ sg: sg0, into: null, out: sg, code, kT, a: sg.t0, d: sg.t0, A: sg.t0,
        D: near(sg, kT - THROAT_KM, kT + THROAT_KM)[1], prefer: code === west ? lwTrack(sg.run, false) : gnTrack(sg.run) });
    });
  }
  // in the order sets come in (sets already there at the start: the last to leave the deepest)
  stays.sort((x, y) => x.A - y.A || y.d - x.d || x.a - y.a);
  const placed = [];
  const on = (st, track) => placed.filter((o) => !o.ghost && o.code === st.code && o.track === track && o.A < st.D
    && st.A < o.D);
  const fits = (st, track) => {
    const there = on(st, track);
    // every set on the track while this one is near: it was standing before this one came near, and leaves after
    // this one has gone (this one stands behind it)
    if (!there.every((o) => o.a <= st.A && st.D <= o.d)) return null;
    const slot = there.length ? 1 + Math.max(...there.map((o) => o.slot)) : 0;
    return slot < (slots[`${st.code}:${track}`] ?? Infinity) ? slot : null;
  };
  for (const st of stays) {
    const other = st.code === west ? (st.prefer === "8" ? "6b" : "8") : (st.prefer === "5a" ? "6a" : "5a");
    let pick = null;
    for (const tr of [st.prefer, other]) {
      const slot = fits(st, tr);
      if (slot !== null && (pick === null || slot < pick[1])) pick = [tr, slot];
    }
    // no room at the platforms: the first siding of the terminal's yard that has room
    for (const y of pick === null ? yard[st.code] || [] : []) {
      const slot = fits(st, y.track);
      if (slot !== null) { pick = [y.track, slot]; break; }
    }
    if (pick === null) {
      // no room at the platforms or in the yard: drawn faded on its usual track, behind the sets there
      st.ghost = true;
      const there = on(st, st.prefer), cap = slots[`${st.code}:${st.prefer}`] ?? Infinity;
      pick = [st.prefer, Math.min(there.length ? 1 + Math.max(...there.map((o) => o.slot)) : 0, cap - 1)];
    }
    [st.track, st.slot] = pick;
    placed.push(st);
    st.sg.track = st.track;
    st.sg.slot = st.slot;
    const y = (yard[st.code] || []).find((x) => x.track === st.track);
    st.sg.km = st.kT + (st.code === west ? 1 : -1) * ((y ? y.km : 0) + st.slot * (slotKm[st.code] ?? SLOT_KM));
  }
  // a set with no room is faded only while it shares its track with a set it cannot stand beside (coming in or going
  // out through it, or on its spot), solid before and after
  for (const g of placed) {
    if (!g.ghost) continue;
    const win = [];
    for (const o of placed) {
      if (o === g || o.code !== g.code || o.track !== g.track || !(o.A < g.D && g.A < o.D)) continue;
      const fine = o.slot < g.slot ? o.a <= g.A && g.D <= o.d : o.slot > g.slot ? g.a <= o.A && o.D <= g.d : false;
      if (!fine) win.push([Math.max(g.A, o.A), Math.min(g.D, o.D)]);
    }
    // (a move ends one stay and starts the next: it keeps both's windows)
    for (const sg of [g.sg, g.into, g.out]) if (sg) sg.ghost = (sg.ghost || []).concat(win);
  }
  // moves inherit the terminal tracks of the stands around them
  for (const un of units) {
    un.segs.forEach((sg, i) => {
      if (sg.type !== "move") return;
      const r = sg.run, before = un.segs[i - 1], after = un.segs[i + 1];
      sg.lw = r.dir === "E" ? lwTrack(r, false) : lwTrack(r, true);
      sg.gn = gnTrack(r);
      const from = before && before.type === "stand" && before.track ? before : sg.from;
      if (from && from.track) {
        if (from.code === west) sg.lw = from.track; else sg.gn = from.track;
        sg.startKm = from.km;
      }
      if (after && after.type === "stand" && after.track) {
        if (after.code === west) sg.lw = after.track; else sg.gn = after.track;
        sg.endKm = after.km;
      }
    });
    for (const sg of un.segs) {
      if (sg.type === "stand" && sg.km === undefined) sg.km = kmOf[sg.code];
    }
  }

  // two-track stations: each direction's platform, or the other one when it is taken. A set holds its track for as long
  // as it is in the station's zone (the stretch where a set on the other track is drawn there: opts.zones), else
  // within PLATFORM_KM of the stop
  const zoneOf = (c) => (opts.zones && opts.zones[c] ? opts.zones[c].slice(0, 2)
    : [kmOf[c] - PLATFORM_KM, kmOf[c] + PLATFORM_KM]);
  const occ = {};      // code -> [{t0, t1, prefer, apply(lane)}]
  const twoTrack = (c) => point[c] && point[c].tracks === 2 && !point[c].directional;
  for (const un of units) {
    un.segs.forEach((sg, i) => {
      if (sg.type === "stand" && twoTrack(sg.code)) {
        const prefer = sg.prev ? sg.prev.dir : sg.next ? sg.next.dir : "E";
        const before = un.segs[i - 1], after = un.segs[i + 1];
        const [lo, hi] = zoneOf(sg.code);
        (occ[sg.code] ||= []).push({
          t0: before && before.type === "move" ? near(before, lo, hi)[0] : sg.t0,
          t1: after && after.type === "move" ? near(after, lo, hi)[1] : Math.min(sg.t1, sg.t0 + 24 * 60), prefer,
          apply: (lane) => {
            sg.lane = lane;
            if (before && before.type === "move") before.lanes[sg.code] = lane;
            if (after && after.type === "move") after.lanes[sg.code] = lane;
          },
        });
      } else if (sg.type === "move") {
        const r = sg.run;
        const moving = r.started && r.fixed_from !== null && r.fixed_from !== undefined;
        const from = moving ? sg.k - 1 : r.started ? sg.k : sg.s;     // the first point it leaves in the window
        for (let j = from + 1; j < sg.e; j++) {
          const c = r.route[j];
          if (!twoTrack(c) || r.arr[j] === null) continue;
          const [t0, t1] = near(sg, ...zoneOf(c));
          (occ[c] ||= []).push({ t0, t1, prefer: r.dir, apply: (lane) => { sg.lanes[c] = lane; } });
        }
      }
    });
  }
  for (const list of Object.values(occ)) {
    list.sort((a, b) => a.t0 - b.t0);
    const taken = { E: [], W: [] };
    for (const o of list) {
      // minutes this set would share the lane with the sets already on it: its own lane if it is free, else the
      // other one if that is free, else whichever it shares for less time
      const shared = (lane) => taken[lane].reduce((m, [a, b]) => m + Math.max(0, Math.min(b, o.t1) - Math.max(a, o.t0)),
        0);
      const other = o.prefer === "E" ? "W" : "E";
      const lane = shared(o.prefer) <= 0 ? o.prefer : shared(other) <= 0 ? other
        : shared(other) < shared(o.prefer) ? other : o.prefer;
      taken[lane].push([o.t0, o.t1]);
      o.apply(lane);
    }
  }

  // a set turning back at a station leaves from where it stands. It is drawn there by the route it came in on, and
  // each direction's km scale is anchored at its own platform (up to 16 m apart along the track, Buitenpost), so its
  // next run starts at the km that lands on that spot (opts.turn), not at the station's km: no jump as it leaves
  const turn = opts.turn || {};
  for (const un of units) {
    un.segs.forEach((sg, i) => {
      const out = un.segs[i + 1];
      if (sg.type !== "stand" || !sg.prev || !sg.next || sg.prev.dir === sg.next.dir || !out || out.type !== "move"
        || out.startKm !== null || Math.abs(out.knots[0].x - kmOf[sg.code]) > 1e-9) return;
      const shift = turn[`${sg.code}:${sg.prev.dir}:${sg.lane || sg.prev.dir}`];
      if (shift) out.startKm = kmOf[sg.code] + shift;
    });
  }

  // the final curves, with terminal stands at their slots
  for (const un of units) {
    for (const sg of un.segs) if (sg.type === "move") shape(sg, kmOf);
    // keep stands contiguous with the moves around them
    un.segs.forEach((sg, i) => {
      if (sg.type !== "stand") return;
      const before = un.segs[i - 1], after = un.segs[i + 1];
      if (before && before.type === "move") sg.t0 = before.t1;
      if (after && after.type === "move") sg.t1 = after.t0;
    });
  }

  const tEnd = Math.max(R.end, ...units.flatMap((u) => u.segs.filter((s) => s.type === "move").map((s) => s.t1)));
  return new Sim(R, units, kmOf, point, tEnd);
}

// a move's knots and curve: through every simulated time of its run, ending (starting) at its terminal slot
function shape(sg, kmOf) {
  const r = sg.run, kn = [];
  const push = (t, x, stop) => {
    if (t === null || t === undefined) return;
    const last = kn[kn.length - 1];
    if (last && t <= last.t + 1e-9) {
      if (Math.abs(x - last.x) < 1e-9) { last.stop = last.stop || stop; return; }
      t = last.t + 0.05;                                   // never two places at one instant
    }
    kn.push({ t, x, stop });
  };
  const running = r.started && r.fixed_from !== null && r.fixed_from !== undefined;
  const first = r.started ? sg.k : sg.s;
  const kmAt = (j) => {
    if (j === sg.e && sg.endKm !== null) return sg.endKm;
    if (j === first && !running && sg.startKm !== null) return sg.startKm;
    return kmOf[r.route[j]];
  };
  if (running) push(r.fixed_from, sg.startKm ?? kmOf[r.route[sg.k - 1]], !!r.stops[sg.k - 1]);
  for (let j = first; j <= sg.e; j++) {
    const x = kmAt(j);
    if (j === first && !running) {
      if (r.started) push(r.arr[j], x, true);
      push(r.dep[j], x, true);
    } else if (j === sg.e) {
      push(r.arr[j], x, true);
    } else if (r.stopped[j]) {
      push(r.arr[j], x, true);
      push(r.dep[j], x, true);
    } else {
      push(r.arr[j], x, false);
    }
  }
  sg.H = hermite(kn);
  sg.knots = kn;
  sg.t0 = kn[0].t;
  sg.t1 = kn[kn.length - 1].t;
}

// [first, last] time a move is between km lo and hi (a stretch its route passes or ends in): the curve is monotone,
// so that is one stretch of time, and each end lies between two knots
function near(sg, lo, hi) {
  const kn = sg.knots, H = sg.H, inside = (t) => { const x = evalHermite(H, t)[0]; return x >= lo && x <= hi; };
  const i = kn.findIndex((q) => q.x >= lo && q.x <= hi);
  if (i < 0) return [sg.t1, sg.t1];
  let j = i;
  while (j + 1 < kn.length && kn[j + 1].x >= lo && kn[j + 1].x <= hi) j++;
  const t0 = i === 0 ? kn[0].t : bisect(kn[i - 1].t, kn[i].t, inside);
  const t1 = j === kn.length - 1 ? kn[j].t : bisect(kn[j + 1].t, kn[j].t, inside);
  return [t0, t1];
}

// the edge between `out` (where ok fails) and `inn` (where it holds), either way round in time
function bisect(out, inn, ok) {
  for (let n = 0; n < 40 && Math.abs(inn - out) > 1e-4; n++) {
    const mid = (out + inn) / 2;
    if (ok(mid)) inn = mid; else out = mid;
  }
  return inn;
}

export class Sim {
  constructor(R, units, kmOf, point, tEnd) {
    this.R = R;
    this.units = units;
    this.kmOf = kmOf;
    this.point = point;
    this.t0 = R.start;
    this.t1 = tEnd;
    this.codes = R.points.map((p) => p.code);
  }

  segAt(un, t) {
    const segs = un.segs;
    if (!segs.length) return null;
    let i = 0;
    while (i < segs.length - 1 && t >= segs[i].t1) i++;
    return segs[i];
  }

  // where a train set is and what it is doing at minute t
  stateAt(un, t) {
    const sg = this.segAt(un, t);
    if (!sg) return null;
    if (sg.type === "stand") {
      const facing = sg.prev ? sg.prev.dir : sg.next ? sg.next.dir : "E";
      const ref = sg.prev || sg.next;
      return {
        // cab: a set that will run again is driven from the end it leaves from (the driver changes ends), so the cab
        // looks down the line it will take, not past the buffers it came in against
        unit: un.id, km: sg.km, dir: facing, cab: sg.next ? sg.next.dir : facing, moving: false, speed: 0,
        code: sg.code,
        lw: sg.code === this.codes[0] && sg.track ? sg.track : ref ? lwTrack(ref, facing === "W") : "8",
        gn: sg.code === this.codes[this.codes.length - 1] && sg.track ? sg.track : ref ? gnTrack(ref) : "5a",
        lanes: sg.lane ? { [sg.code]: sg.lane } : null, train: sg.next ? sg.next.train : sg.prev ? sg.prev.train : "",
        kind: ref ? ref.kind : "stop", status: this.standStatus(sg, t), delay: this.standDelay(sg),
        run: sg.next || sg.prev, ghost: this.ghostAt(sg, t),
      };
    }
    const [km, v] = evalHermite(sg.H, t);
    return {
      unit: un.id, km, dir: sg.dir, moving: Math.abs(v) > 1e-4, speed: Math.abs(v) * 60, code: this.atPoint(sg, t),
      lw: sg.lw, gn: sg.gn, lanes: Object.keys(sg.lanes).length ? sg.lanes : null, train: sg.run.train,
      kind: sg.run.kind, status: this.moveStatus(sg, t, km) + (this.ghostAt(sg, t) > 0 ? FADED : ""),
      delay: this.moveDelay(sg, t), run: sg.run, ghost: this.ghostAt(sg, t),
    };
  }

  // how faded a set is drawn at t (0 solid, 1 faded): fully inside its windows (sg.ghost), fading over GHOST_RAMP
  ghostAt(sg, t) {
    let g = 0;
    for (const [a, b] of sg.ghost || []) g = Math.max(g, 1 - (t < a ? a - t : t > b ? t - b : 0) / GHOST_RAMP);
    return g;
  }

  atPoint(sg, t) {
    const kn = sg.knots;
    for (let i = 0; i < kn.length - 1; i++) {
      if (kn[i].stop && kn[i + 1].stop && Math.abs(kn[i].x - kn[i + 1].x) < 1e-9 && kn[i].t <= t && t <= kn[i + 1].t) {
        return this.codeAtKm(kn[i].x);
      }
    }
    return null;
  }

  codeAtKm(km) {
    let best = null, d = Infinity;
    for (const [c, k] of Object.entries(this.kmOf)) {
      if (Math.abs(k - km) < d) { d = Math.abs(k - km); best = c; }
    }
    return d < 0.3 ? best : null;
  }

  moveDelay(sg, t) {
    const r = sg.run;
    let late = 0;
    for (let j = 0; j < r.route.length; j++) {
      const a = r.arr[j] ?? r.dep[j];
      const sch = r.sched_arr[j] ?? r.sched_dep[j];
      if (a === null || a === undefined || sch === null || sch === undefined) continue;
      if (a <= t) late = a - sch;
    }
    return Math.max(0, late);
  }

  standDelay(sg) {
    const r = sg.next;
    if (!r) return 0;
    const j = r.started ? r.fixed_to : r.first;
    return Math.max(0, (r.dep[j] ?? 0) - (r.sched_dep[j] ?? r.dep[j] ?? 0));
  }

  moveStatus(sg, t, km) {
    const r = sg.run;
    const code = this.atPoint(sg, t);
    if (code) {
      const j = r.route.indexOf(code);
      const hold = r.order && r.order.hold && r.order.hold[code];
      const dep = r.dep[j];
      if (hold && dep !== null && dep > (r.sched_dep[j] ?? dep)) return `held at ${code} until ${hold} (plan)`;
      if (r.stops[j]) return `at ${code}, leaves ${hhmm(dep)}`;
      return `waiting at ${code}, leaves ${hhmm(dep)}`;
    }
    let j = 0;
    while (j < r.route.length - 1 && (r.arr[j + 1] === null || r.arr[j + 1] <= t)) j++;
    const nextStop = r.route.slice(j + 1, sg.e + 1).find((c, i) => r.stops[j + 1 + i]) || r.route[sg.e];
    const to = r.route[sg.e];
    if (String(sg.dir === "W" ? sg.lw : sg.gn).startsWith("Y") && sg.endKm !== null && (to === this.codes[0]
      || to === this.codes[this.codes.length - 1])) return `to the sidings outside ${to}`;
    return `to ${to}${to !== r.route[r.route.length - 1] ? " (turned short)" : ""}, next ${nextStop}`;
  }

  standStatus(sg, t) {
    const s = this.standText(sg, t);
    return this.ghostAt(sg, t) > 0 ? s + FADED : s;
  }

  standText(sg, t) {
    const yard = String(sg.track || "").startsWith("Y");
    if (yard) {
      // a drawn siding outside the terminal: the platforms had no room (the rules give a terminal room for every set)
      const r = sg.next;
      return r ? `in the sidings outside ${sg.code}; runs ${r.train} at ${hhmm(r.dep[r.started ? r.fixed_to : r.first])}`
        : `in the sidings outside ${sg.code}: done for this window`;
    }
    if (!sg.next) return `at ${sg.code}: done for this window`;
    const r = sg.next;
    const j = r.started ? r.fixed_to : r.first;
    const hold = r.order && r.order.hold && r.order.hold[sg.code];
    if (hold && r.dep[j] > (r.sched_dep[j] ?? r.dep[j])) return `held at ${sg.code} until ${hold} (plan), then ${r.train}`;
    if (sg.prev && sg.prev.last < sg.prev.route.length - 1 && sg.prev.route[sg.prev.last] === sg.code) {
      return `turned back at ${sg.code}; runs ${r.train} at ${hhmm(r.dep[j])}`;
    }
    return `at ${sg.code}; runs ${r.train} at ${hhmm(r.dep[j])}`;
  }

  // every train set's state at t (those with a window)
  statesAt(t) {
    return this.units.map((u) => this.stateAt(u, t)).filter(Boolean);
  }

  // cancelled runs (shown in the list, never on the map)
  cancelled() {
    return this.R.runs.filter((r) => r.cancelled);
  }

  // which sections hold a train between their end points at t: index -> [unit ids]
  occupancy(states) {
    const occ = {};
    const pts = this.R.points;
    for (const s of states) {
      if (s.code) continue;            // standing at a point: a station, a terminal track or a terminal's sidings
      for (let i = 0; i < pts.length - 1; i++) {
        if (s.km > pts[i].km + 0.03 && s.km < pts[i + 1].km - 0.03) {
          (occ[i] ||= []).push(s.unit);
          break;
        }
      }
    }
    return occ;
  }

  // the cost accrued by minute t (the replay's cost events, which add up to the grader's cost)
  costAt(t) {
    let sum = 0;
    for (const c of this.R.costs) {
      if (c[0] > t) break;
      sum += c[1];
    }
    return sum;
  }
}
