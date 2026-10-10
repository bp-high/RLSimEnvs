// The map replay's geometry and motion against the replays the server serves (node --test; no npm install).
//
//   RAIL_REPLAYS=<dir of replay JSON files> node --test openenv/tests/viewer/*.test.mjs
//
// test_viewer.py writes the replays (the optimum, the baselines and the committed model plans for a sample of tasks)
// and runs this file. Without RAIL_REPLAYS only the twin checks run.

import assert from "node:assert/strict";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import { Line, CANONICAL, overTrack } from "../../rail_openenv/web/twin/geo.js";
import { buildSim, evalHermite, hermite, SLOT_M, TRAIN_M } from "../../rail_openenv/web/twin/motion.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const TWIN = JSON.parse(gunzipSync(readFileSync(join(HERE, "../../rail_openenv/web/twin/twin.json.gz"))).toString());
const NETWORK = JSON.parse(readFileSync(join(HERE, "../../../../../data/gnlw/network.json"), "utf8"));
// what the page passes buildSim: the terminals' slot spacing and capacity and the stations' zones, from the map
const OPTS = new Line(TWIN, NETWORK.points).simOpts(SLOT_M, TRAIN_M);
const simOf = (R) => buildSim(R, OPTS);
const DIR = process.env.RAIL_REPLAYS;
const REPLAYS = DIR && existsSync(DIR)
  ? readdirSync(DIR).filter((f) => f.endsWith(".json")).map((f) => JSON.parse(readFileSync(join(DIR, f), "utf8")))
  : [];

test("every route passes every network point in order, and km -> place -> km round-trips", () => {
  const line = new Line(TWIN, NETWORK.points);
  assert.equal(Object.keys(line.routes).length, 8);
  for (const r of Object.values(line.routes)) {
    const ks = NETWORK.points.map((p) => p.km);
    for (const km of ks) {
      const s = r.sAtKm(km);
      assert.ok(s >= -1 && s <= r.length + 1, `${r.key}: km ${km} off the route`);
      assert.ok(Math.abs(r.kmAtS(s) - km) < 1e-6, `${r.key}: round trip at km ${km}`);
    }
    // s grows in the route's direction of travel
    const s0 = r.sAtKm(ks[0]), s1 = r.sAtKm(ks[ks.length - 1]);
    assert.ok(r.dir === "E" ? s1 > s0 : s0 > s1, `${r.key} runs the wrong way`);
    // no jumps: neighbouring km steps stay close on the map
    let prev = r.pointAtKm(0);
    for (let km = 0.01; km < ks[ks.length - 1]; km += 0.01) {
      const p = r.pointAtKm(km);
      assert.ok(Math.hypot(p[0] - prev[0], p[1] - prev[1]) < 40, `${r.key}: jump at km ${km.toFixed(2)}`);
      prev = p;
    }
  }
});

test("the two directions share the single track and part where the line is double", () => {
  const line = new Line(TWIN, NETWORK.points);
  const e = line.routes[CANONICAL.E], w = line.routes[CANONICAL.W];
  const km = Object.fromEntries(NETWORK.points.map((p) => [p.code, p.km]));
  // lateral distance from the eastbound lane to the westbound one (each direction's stations anchor its own platform
  // centre, so the same km can sit a few metres apart along a shared track)
  const gap = (k) => {
    const a = e.pointAtKm(k), s = w.sAtKm(k);
    let best = Infinity;
    for (let d = -150; d <= 150; d += 0.5) {
      const b = w.pointAtS(s + d);
      best = Math.min(best, Math.hypot(a[0] - b[0], a[1] - b[1]));
    }
    return best;
  };
  for (const sec of NETWORK.sections) {
    const mid = (km[sec.a] + km[sec.b]) / 2;
    if (sec.single) assert.ok(gap(mid) < 1.0, `${sec.a}-${sec.b} is single track but the lanes are ${gap(mid)} m apart`);
    else assert.ok(gap(mid) > 2.5 && gap(mid) < 15, `${sec.a}-${sec.b} is double track but the lanes are ${gap(mid)} m apart`);
  }
  // blending onto the other track lands on it inside a station zone, and changes nothing outside
  for (const [code, [k0, k1]] of Object.entries(line.zones)) {
    const mid = km[code];
    // straight across: on the westbound track, and level with the eastbound position (not at the westbound route's
    // same-km point, which can lie up to 16 m further along)
    const p = line.place("E", "8", "5a", mid, { [code]: "W" }), q = w.project(p, w.sAtKm(mid));
    assert.ok(Math.hypot(p[0] - q[0], p[1] - q[1]) < 0.05, `${code}: blend misses the other track`);
    const own = e.pointAtKm(mid), side = Math.hypot(p[0] - own[0], p[1] - own[1]);
    assert.ok(side > 2.5 && side < 15, `${code}: the other track is ${side.toFixed(1)} m across, not beside it`);
    const out = line.place("E", "8", "5a", k1 + 0.1, { [code]: "W" }), o = e.pointAtKm(k1 + 0.1);
    assert.ok(Math.hypot(out[0] - o[0], out[1] - o[1]) < 1e-6, `${code}: blend leaks out of its zone`);
    assert.ok(k0 < mid && mid < k1);
  }
});

test("a train standing at a terminal keeps its shape: no car folds back past the end of its route", () => {
  // routes end at the terminals' stop points, so half of a train standing there reaches past the end
  const line = new Line(TWIN, NETWORK.points);
  const ends = [NETWORK.points[0], NETWORK.points[NETWORK.points.length - 1]];
  const offs = [23.25, 4.25, -4.25, -23.25];                 // the bogies of a two-car unit, as scene.js places them
  for (const r of Object.values(line.routes)) {
    const [lw, gn] = r.dir === "E" ? r.key.split("-").slice(1) : r.key.split("-").slice(1).reverse();
    for (const [i, { code, km }] of ends.entries()) {
      const n = OPTS.slots[`${code}:${i ? gn : lw}`];
      assert.ok(n >= 2 && n <= 6, `${code} ${i ? gn : lw} holds ${n} sets`);
      for (let slot = 0; slot < n; slot++) {
        const k = km + (i ? -1 : 1) * slot * OPTS.slotKm[code];
        const pts = line.body(r.dir, lw, gn, k, offs, null);
        for (let i = 1; i < pts.length; i++) {
          const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
          assert.ok(Math.abs(d - (offs[i - 1] - offs[i])) < 0.5,
            `${r.key} at km ${k}: bogies ${i - 1}-${i} are ${d.toFixed(1)} m apart, not ${offs[i - 1] - offs[i]}`);
        }
        // and the two cars follow the track: at most a gentle curve between them (a fold is about 180 degrees)
        const a = Math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]);
        const b = Math.atan2(pts[3][1] - pts[2][1], pts[3][0] - pts[2][0]);
        assert.ok(Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b))) < 10 * Math.PI / 180, `${r.key} at km ${k}: bent`);
      }
    }
  }
});

test("buildings over the track are raised, not solid: Groningen's train shed and nothing else of its size", () => {
  const raised = overTrack(TWIN);
  const gn = TWIN.stations.find((s) => s.code === "GN");
  const centre = (k) => {
    const f = TWIN.buildings.p[k];
    let x = 0, y = 0;
    for (let i = 0; i < f.length; i += 2) { x += f[i] / 10; y += f[i + 1] / 10; }
    return [x / (f.length / 2), y / (f.length / 2)];
  };
  assert.ok(raised.size >= 1 && raised.size <= 5, `${raised.size} buildings over the track`);
  const shed = [...raised].filter((k) => TWIN.buildings.k[k] === 7
    && Math.hypot(centre(k)[0] - gn.x, centre(k)[1] - gn.y) < 300);
  assert.equal(shed.length, 1, "Groningen's train shed (building=train_station) is not over the track");
});

test("the yards: sidings off the line near each terminal, on open ground, and routes into them that hold together", () => {
  const line = new Line(TWIN, NETWORK.points);
  const yards = line.yards();
  const dec = (f) => { const o = []; for (let i = 0; i < f.length; i += 2) o.push([f[i] / 10, f[i + 1] / 10]); return o; };
  const inside = (x, y, ring) => {
    let c = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i], [xj, yj] = ring[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
    }
    return c;
  };
  const tracks = line.yardTracks();
  for (const end of line.ends) {
    const y = yards[end.code];
    assert.ok(y && y.sidings.length >= 3, `${end.code}: ${y ? y.sidings.length : 0} sidings`);
    for (const g of y.sidings) {
      assert.ok(g.km > 0.2 && g.km < 1.5, `${end.code} ${g.track}: ${g.km} km out`);
      assert.equal(OPTS.slots[`${end.code}:${g.track}`], 2);
    }
  }
  // no siding point inside a building or water (checked against every footprint, not the search's grid)
  const pts = tracks.flatMap((t) => dec(t.p));
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  const near = (ring) => ring.some(([x, y]) => x > x0 - 300 && x < x1 + 300 && y > y0 - 300 && y < y1 + 300);
  const rings = [...TWIN.buildings.p.map(dec), ...TWIN.water.map((w) => dec(w.o))].filter(near);
  for (const [x, y] of pts) assert.ok(!rings.some((r) => inside(x, y, r)), `a siding runs through a building or water at ${x}, ${y}`);
  // every route into a siding is continuous and the line's own route away from the terminals
  for (const [dir, lw, gn] of [["E", "Y1", "5a"], ["W", "8", "Y2"], ["E", "Y3", "Y3"], ["W", "Y2", "6a"]]) {
    const r = line.route(dir, lw, gn), base = line.route(dir, "8", "5a");
    assert.equal(r.key, line.routeKey(dir, lw, gn));
    let prev = r.pointAtKm(-0.1);
    for (let km = -0.09; km < 55.1; km += 0.01) {
      const p = r.pointAtKm(km);
      assert.ok(Math.hypot(p[0] - prev[0], p[1] - prev[1]) < 15, `${r.key}: jump at km ${km.toFixed(2)}`);
      prev = p;
    }
    for (let km = 2; km < 53; km += 1) {
      const a = r.pointAtKm(km), b = base.pointAtKm(km);
      assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1]) < 1, `${r.key} leaves the line at km ${km}`);
    }
  }
});

test("hermite: hits every knot, zero speed at stops, never runs backwards between monotone knots", () => {
  const H = hermite([{ t: 0, x: 0, stop: true }, { t: 5, x: 4, stop: false }, { t: 9, x: 7, stop: true },
                     { t: 11, x: 7, stop: true }, { t: 15, x: 10, stop: true }]);
  for (const [t, x] of [[0, 0], [5, 4], [9, 7], [11, 7], [15, 10]]) assert.ok(Math.abs(evalHermite(H, t)[0] - x) < 1e-9);
  assert.equal(evalHermite(H, 9)[1], 0);
  let prev = -1;
  for (let t = 0; t <= 15; t += 0.05) {
    const [x] = evalHermite(H, t);
    assert.ok(x >= prev - 1e-9);
    prev = x;
  }
});

test("replays: trains are where the simulation says at every simulated minute", { skip: !REPLAYS.length }, () => {
  for (const R of REPLAYS) {
    const S = simOf(R);
    const km = Object.fromEntries(R.points.map((p) => [p.code, p.km]));
    const ends = new Set([R.points[0].code, R.points[R.points.length - 1].code]);
    for (const un of S.units) {
      for (const sg of un.segs) {
        if (sg.type !== "move") continue;
        const r = sg.run;
        for (let j = sg.s; j <= sg.e; j++) {
          for (const t of [r.arr[j], r.dep[j]]) {
            if (t === null || t === undefined || t < sg.t0 || t > sg.t1) continue;
            const st = S.stateAt(un, t);
            // at a terminal: where the allocation stood it (its platform track's slot, or a siding in the yard)
            const want = j === sg.e && sg.endKm !== null ? sg.endKm
              : j === (r.started ? sg.k : sg.s) && sg.startKm !== null ? sg.startKm : km[r.route[j]];
            if (!ends.has(r.route[j]) && want !== km[r.route[j]]) continue;   // an illegal plan's set leaving where it is
            assert.ok(Math.abs(st.km - want) <= 1e-6,
              `${R.task_id} ${R.plan}: ${r.train} at ${t} is at km ${st.km}, not ${r.route[j]} (${want})`);
          }
        }
        // monotone along its direction, and a believable speed
        let prev = null;
        for (let t = sg.t0; t <= sg.t1; t += 0.1) {
          const st = S.stateAt(un, t);
          if (prev !== null) {
            const dx = (st.km - prev) * (sg.dir === "E" ? 1 : -1);
            assert.ok(dx >= -1e-6, `${R.task_id}: ${r.train} runs backwards at ${t.toFixed(1)}`);
          }
          assert.ok(st.speed < 260, `${R.task_id}: ${r.train} at ${st.speed.toFixed(0)} km/h`);
          prev = st.km;
        }
      }
    }
  }
});

test("replays: never two trains on one station track at once in a legal plan", { skip: !REPLAYS.length }, () => {
  for (const R of REPLAYS.filter((x) => x.feasible)) {
    const sim = simOf(R);
    for (let t = sim.t0; t <= sim.t1; t += 0.25) {
      const seen = new Map();
      for (const s of sim.statesAt(t)) {
        if (s.moving || !s.code) continue;
        const p = sim.point[s.code];
        if (!p || p.tracks !== 2 || p.directional) continue;
        const lane = (s.lanes && s.lanes[s.code]) || s.dir;
        const key = `${s.code}:${lane}`;
        assert.ok(!seen.has(key), `${R.task_id} ${R.plan}: ${seen.get(key)} and ${s.train} share ${key} at ${t}`);
        seen.set(key, s.train);
      }
    }
  }
});

test("replays: no set is drawn through another at a terminal; one with no platform room waits in the yard", { skip: !REPLAYS.length }, () => {
  // A terminal's two tracks hold sets one behind another: a set comes in behind the sets on its track and leaves
  // before them, only as deep as the track runs clear of its switch. The rules give a terminal room for every set, so a
  // plan can need a third track the line does not have (a set comes in as another leaves, the other track taken); that
  // set waits in the yard the replay draws outside the terminal, and only if the yard is full too is it faded while it
  // would share a track. Two bodies within 2 m of each other are on one track. (A plan that breaks a rule can send two
  // sets out together, or one into another: that is drawn as it is.)
  const line = new Line(TWIN, NETWORK.points);
  const offs = [];
  for (let d = -27.5; d <= 27.5; d += 5) offs.push(d);
  let stays = 0, faded = 0;
  for (const R of REPLAYS) {
    const sim = simOf(R);
    const ends = [R.points[0].km, R.points[R.points.length - 1].km];
    for (const un of sim.units) {
      for (const sg of un.segs) {
        if (sg.type !== "stand" || sg.track === undefined) continue;
        stays++;
        if (sg.ghost && sg.ghost.length) faded++;
        assert.ok(sg.slot < OPTS.slots[`${sg.code}:${sg.track}`], `${R.task_id}: ${un.id} stands past ${sg.code} ${sg.track}'s switch`);
      }
    }
    if (!R.feasible) continue;
    for (let t = sim.t0; t <= sim.t1; t += 0.1) {
      const here = sim.statesAt(t).filter((s) => s.ghost < 0.5 && ends.some((k) => Math.abs(s.km - k) < 1.6));
      const bodies = here.map((s) => line.body(s.dir, s.lw, s.gn, s.km, offs, s.lanes));
      for (let i = 0; i < here.length; i++) {
        for (let j = i + 1; j < here.length; j++) {
          let d = Infinity;
          for (const p of bodies[i]) for (const q of bodies[j]) d = Math.min(d, Math.hypot(p[0] - q[0], p[1] - q[1]));
          assert.ok(d >= 2, `${R.task_id} ${R.plan}: ${here[i].train} and ${here[j].train} drawn through each other at ${t.toFixed(1)}`);
        }
      }
    }
  }
  assert.ok(stays > 500 && faded <= 0.002 * stays, `${faded} of ${stays} terminal stays faded`);
});

test("replays: a standing train's cab looks along the line it will take, never past the end", { skip: !REPLAYS.length }, () => {
  const line = new Line(TWIN, NETWORK.points);
  const CAB = [27, 27 + 160];                                // scene.js: CAR + 0.5 and CAR + 160 ahead
  for (const R of REPLAYS) {
    const sim = simOf(R);
    for (const un of sim.units) {
      for (const sg of un.segs) {
        if (sg.type !== "stand" || !sg.next || !Number.isFinite(sg.t1) || sg.t1 - sg.t0 < 0.2) continue;
        const st = sim.stateAt(un, (sg.t0 + sg.t1) / 2);
        assert.equal(st.cab, sg.next.dir, `${R.task_id}: ${st.train} cab`);
        // scene.js: from the train as drawn, along the route it is drawn on (backwards when it will leave the other way)
        const r = line.route(st.dir, st.lw, st.gn), s0 = r.sAtKm(st.km), k = st.cab !== st.dir ? -1 : 1;
        for (const d of CAB) {
          const s = s0 + k * d;
          assert.ok(s >= -1 && s <= r.length + 1, `${R.task_id} ${R.plan}: ${st.train}'s cab at ${st.code} looks past the end`);
        }
      }
    }
  }
});

test("replays: every train keeps its shape and its cab a view, everywhere, every minute", { skip: !REPLAYS.length }, () => {
  // the bogies scene.js places: two 26.5 m cars with a 1 m gap, bogies 9.5 m either side of each car's centre. A fold
  // puts two bogies on one point; blending towards the other track's same-km point stretched a car crossing over at
  // a station by up to 3.3 m. Now a train keeps its length within half a metre everywhere.
  const line = new Line(TWIN, NETWORK.points);
  const offs = [23.25, 4.25, -4.25, -23.25];
  let n = 0;
  for (const R of REPLAYS) {
    const sim = simOf(R);
    for (let t = sim.t0; t <= sim.t1; t += 1) {
      for (const s of sim.statesAt(t)) {
        const p = line.body(s.dir, s.lw, s.gn, s.km, offs, s.lanes);
        for (let i = 1; i < p.length; i++) {
          const d = Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]);
          assert.ok(Math.abs(d - (offs[i - 1] - offs[i])) < 0.5,
            `${R.task_id} ${R.plan}: ${s.train} at ${t} (km ${s.km.toFixed(3)}): bogies ${d.toFixed(1)} m apart`);
        }
        const a = Math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0]), b = Math.atan2(p[3][1] - p[2][1], p[3][0] - p[2][0]);
        assert.ok(Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b))) < 15 * Math.PI / 180,
          `${R.task_id} ${R.plan}: ${s.train} at ${t} (km ${s.km.toFixed(3)}) is folded`);
        // the cab camera and its aim point never meet (they did, past a route's end: a blank view)
        const k = s.cab && s.cab !== s.dir ? -1 : 1;
        const [c0, c1] = line.body(s.dir, s.lw, s.gn, s.km, [27 * k, 187 * k], s.lanes);
        assert.ok(Math.hypot(c1[0] - c0[0], c1[1] - c0[1]) > 100, `${R.task_id} ${R.plan}: ${s.train}'s cab at ${t}`);
        n++;
      }
    }
  }
  assert.ok(n > 10000, `only ${n} train positions checked`);
});

test("replays: a set that turns back at a station leaves from where it stood", { skip: !REPLAYS.length }, () => {
  // drawn by the route it came in on, it leaves by the other direction's, whose km scale is anchored at the other
  // platform (up to 16 m along the track at Buitenpost): its next run starts at the km that lands on the same spot
  const line = new Line(TWIN, NETWORK.points);
  let n = 0;
  for (const R of REPLAYS) {
    const sim = simOf(R);
    for (const un of sim.units) {
      un.segs.forEach((sg, i) => {
        const out = un.segs[i + 1];
        if (sg.type !== "stand" || !sg.prev || !sg.next || sg.prev.dir === sg.next.dir || !out || out.type !== "move") return;
        if (out.run.route[out.run.started ? out.k : out.s] !== sg.code) return;
        const a = sim.stateAt(un, (sg.t0 + sg.t1) / 2), b = sim.stateAt(un, out.t0);
        const pa = line.place(a.dir, a.lw, a.gn, a.km, a.lanes), pb = line.place(b.dir, b.lw, b.gn, b.km, b.lanes);
        const d = Math.hypot(pa[0] - pb[0], pa[1] - pb[1]);
        assert.ok(d < 0.5, `${R.task_id} ${R.plan}: ${un.id} jumps ${d.toFixed(1)} m leaving ${sg.code} as ${sg.next.train}`);
        n++;
      });
    }
  }
  assert.ok(n > 500, `only ${n} turns checked`);
});

test("replays: cost events add up to the plan's cost", { skip: !REPLAYS.length }, () => {
  for (const R of REPLAYS) {
    const sim = simOf(R);
    assert.equal(sim.costAt(1e9), R.cost_if_legal);
    if (R.feasible) assert.equal(R.cost_if_legal, R.cost);
  }
});
