// The line twin: pick a task and a plan, and watch it run on the real line. Data: /api (rail_core.replay, the grader's
// own simulation) and ./twin.json.gz (OpenStreetMap). URL: ?task=<id>&plan=<id>&t=<HH:MM>&follow=<unit>&cam=map|chase|cab

import { Line } from "./geo.js";
import { buildSim, hhmm, SLOT_M, TRAIN_M } from "./motion.js";
import { CostChart, SERIES, TrainGraph } from "./graph.js";
import { KIND_COLOR, TwinScene } from "./scene.js";

const $ = (s) => document.querySelector(s);
// the header wraps onto more rows on narrow screens: the panels and toasts sit below whatever height it has
new ResizeObserver(([e]) => document.documentElement.style.setProperty("--bar-h", `${Math.ceil(e.target.offsetHeight)}px`))
  .observe($("#bar"));
const params = new URLSearchParams(location.search);
const S = { t: 0, playing: false, speed: 60, sim: null, R: null, opt: null, sel: null, cam: "map", lastEvent: -1 };
const TIER = { real: "real disruption", blockage: "blockage", loop: "lost passing track", compound: "compound" };
const FEATURED = "gnlw-2025-03-10-0700-210m-blockage-0";

async function json(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`);
  return r.json();
}

async function loadTwin() {
  const r = await fetch("./twin.json.gz");
  if (!r.ok) throw new Error(`twin.json.gz: HTTP ${r.status}`);
  const buf = new Uint8Array(await r.arrayBuffer());
  if (buf[0] === 0x1f && buf[1] === 0x8b) {
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(stream).text());
  }
  return JSON.parse(new TextDecoder().decode(buf));             // a proxy already decoded it
}

function parseClock(v, R) {
  const m = /^(\d{1,2}):(\d{2})$/.exec(v || "");
  if (!m) return null;
  let t = +m[1] * 60 + +m[2];
  if (t < R.start - 120) t += 1440;
  return t;
}

function weekday(day) {
  const d = new Date(`${day}T12:00:00Z`);
  return d.toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
}

function setURL() {
  const p = new URLSearchParams();
  p.set("task", $("#task").value);
  p.set("plan", $("#plan").value);
  p.set("t", hhmm(S.t));
  if (S.sel) p.set("follow", S.sel);
  if (S.cam !== "map") p.set("cam", S.cam);
  history.replaceState(null, "", `?${p}`);
  $("#graphlink").href = `/viewer/?task=${encodeURIComponent($("#task").value)}&plan=${encodeURIComponent($("#plan").value)}`;
}

// --- panels ----------------------------------------------------------------------------------------------------

function renderScenario(R) {
  const where = (c) => [...new Set([c.where[0], c.where[c.where.length - 1]])].join("–");
  const what = { closed: "closed", slow: "slow running", track: "one platform track out of use" };
  const items = R.closures.map((c) => `<li><b>${where(c)}</b> ${what[c.kind] || c.kind}${c.kind === "slow" ? ` (+${c.extra} min)` : ""}, `
    + `${hhmm(c.start)}–${hhmm(c.end)}<div class="note">${c.cause || ""}${c.real ? " · from the record" : ""}</div></li>`).join("");
  const notes = (R.notices || []).slice(0, 3).map((n) => `<li class="note">${n}</li>`).join("");
  $("#scenario").innerHTML = `<div class="day">${weekday(R.day)}</div>
    <div class="muted">${hhmm(R.start)}–${hhmm(R.end)} · ${R.runs.length} trains<span class="tier">${TIER[R.difficulty] || R.difficulty}</span></div>
    <ul>${items}${notes}</ul>`;
}

function renderResult(R) {
  const rb = R.violations.length;
  const pct = R.cost !== null && R.optimal_cost ? Math.round((R.cost / R.optimal_cost - 1) * 100) : null;
  const viol = R.violations.slice(0, 40).map((v, i) => `<li data-i="${i}">${v.time || ""} ${v.rule.replace("_", " ")}: ${v.detail}</li>`).join("");
  $("#result").innerHTML = `<div class="plan-name">${R.plan}</div>
    <div class="hero">${R.reward.toFixed(3)}<small>reward</small></div>
    <div class="badge ${R.feasible ? "ok" : "bad"}">${R.feasible ? "legal plan" : `${rb} rule break${rb === 1 ? "" : "s"}: scores under 0.2`}</div>
    <dl>
      <dt>Cost</dt><dd>${R.cost !== null ? R.cost.toLocaleString() : `${R.cost_if_legal.toLocaleString()} if it were legal`}${pct !== null && pct > 0 ? ` (+${pct}% over the optimum)` : ""}</dd>
      <dt>Optimum</dt><dd>${R.optimal_cost.toLocaleString()} (floor ${R.floor_cost.toLocaleString()})</dd>
      <dt>Delay</dt><dd>${R.cost_parts.delay.toLocaleString()} stop-min</dd>
      <dt>Unserved</dt><dd>${(R.cost_parts.unserved / (R.rules.cancel_cost || 30)).toLocaleString()} stops (${R.cost_parts.unserved.toLocaleString()})</dd>
      <dt>Handed on</dt><dd>${R.cost_parts.handed_on.toLocaleString()}</dd>
      <dt>Cancelled</dt><dd>${R.cancelled} runs · ${R.short_turned} turned short</dd>
    </dl>
    ${rb ? `<ul class="viol">${viol}</ul>` : ""}`;
  for (const li of document.querySelectorAll("#result .viol li")) {
    li.onclick = () => {
      const v = R.violations[+li.dataset.i];
      const t = parseClock(v.time, R);
      if (t !== null) seek(t - 1);
      if (v.trains && v.trains[0]) selectRun(v.trains[0]);
    };
  }
}

function renderTrains(states) {
  const ol = $("#trains");
  const rows = states.slice().sort((a, b) => a.unit.localeCompare(b.unit));
  $("#ntrains").textContent = `${rows.length} train sets`;
  const html = rows.map((s) => {
    const d = Math.round(s.delay);
    return `<li data-u="${s.unit}" class="${S.sel === s.unit ? "sel" : ""}"><i class="chip" style="background:${KIND_COLOR[s.kind]}"></i>`
      + `<span class="num">${s.train}</span><span class="muted">${s.dir === "E" ? "→ Groningen" : "← Leeuwarden"}</span>`
      + `<span class="dl ${d > 0 ? "late" : ""}">${d > 0 ? `+${d} min` : "on time"}</span><span class="st">${s.status}</span></li>`;
  }).join("");
  if (ol.dataset.html !== html) {
    ol.innerHTML = html;
    ol.dataset.html = html;
    for (const li of ol.children) li.onclick = () => select(li.dataset.u, true);
  }
  const cx = S.sim.cancelled();
  $("#cxl-h").hidden = !cx.length;
  $("#cancelled").innerHTML = cx.map((r) => `<li><i class="chip" style="background:${KIND_COLOR[r.kind]}"></i>`
    + `<span class="num">${r.train}</span><span>${r.route[0]} ${hhmm(r.sched_dep[0])} → ${r.route[r.route.length - 1]}</span>`
    + `<span>${r.stops.filter(Boolean).length} stops</span></li>`).join("");
}

function toast(ev) {
  const box = $("#toasts");
  const el = document.createElement("div");
  el.className = `toast ${ev.kind}`;
  el.innerHTML = `<time>${hhmm(ev.t)}</time>${ev.text}`;
  box.prepend(el);
  while (box.children.length > 4) box.lastChild.remove();
  setTimeout(() => { el.style.opacity = 0; setTimeout(() => el.remove(), 700); }, 4200);
}

// --- selection and cameras -------------------------------------------------------------------------------------

let scene, graph, cost;

function select(unit, fly) {
  S.sel = unit;
  scene.selected = unit;
  graph.sel = unit;
  graph.draw();
  if (fly && unit) setCam(S.cam === "cab" ? "cab" : "chase");
  setURL();
}

function selectRun(train) {
  const r = S.R.runs.find((x) => x.train === train);
  if (r) select(r.unit, true);
}

function setCam(mode) {
  S.cam = mode;
  for (const [id, m] of [["#cam-map", "map"], ["#cam-follow", "chase"], ["#cam-cab", "cab"]]) $(id).classList.toggle("on", S.cam === m);
  if (mode === "map") scene.viewAll();
  else if (S.sel) scene.follow(S.sel, mode);
  else {
    const first = S.sim.statesAt(S.t).find((s) => s.moving) || S.sim.statesAt(S.t)[0];
    if (first) { select(first.unit, false); scene.follow(first.unit, mode); }
  }
  setURL();
}

function seek(t) {
  S.t = Math.max(S.sim.t0, Math.min(S.sim.t1, t));
  S.lastEvent = S.R.events.findIndex((e) => e.t !== null && e.t > S.t) - 1;
  if (S.lastEvent < -1) S.lastEvent = S.R.events.length - 1;
}

// --- loading a task and a plan ----------------------------------------------------------------------------

// the replay shows the models' plans (the eval runs) and live episodes on this server, not the reference methods
// (the CP-SAT optimum, the greedy dispatchers, change nothing); a server without model runs offers everything
async function loadPlans(taskId, want) {
  const all = await json(`/api/tasks/${encodeURIComponent(taskId)}/plans`);
  const models = all.filter((p) => p.id.startsWith("run:") || p.id.startsWith("episode:"));
  const plans = models.length ? models : all;
  $("#plan").innerHTML = plans.map((p) => `<option value="${p.id}">${p.label}${p.reward !== null && p.reward !== undefined ? ` · ${(+p.reward).toFixed(3)}` : ""}</option>`).join("");
  $("#plan").value = plans.some((p) => p.id === want) ? want : plans[0].id;
}

async function loadReplay(keepTime) {
  const taskId = $("#task").value, planId = $("#plan").value;
  const [R, opt] = await Promise.all([
    json(`/api/tasks/${encodeURIComponent(taskId)}/replay?plan=${encodeURIComponent(planId)}`),
    planId === "optimal" ? null : json(`/api/tasks/${encodeURIComponent(taskId)}/replay?plan=optimal`),
  ]);
  const line = scene ? scene.line : new Line(TWIN, R.points);
  if (!scene) TWIN.tracks = TWIN.tracks.concat(line.yardTracks());    // draw the yards' sidings with the track
  const sim = buildSim(R, line.simOpts(SLOT_M, TRAIN_M));
  const prevT = S.t;
  S.R = R; S.sim = sim; S.opt = opt || R;
  if (!scene) {
    scene = new TwinScene($("#stage"), TWIN, line);
    // a click on a train follows it; the end of a drag (panning the map) is not a click
    let down = null;
    const el = scene.labels.domElement;
    el.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
    el.addEventListener("click", (e) => {
      if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
      const u = scene.pick(e.clientX, e.clientY);
      if (u) select(u, true);
    });
    window.twin = { state: S, scene, seek, select, setCam };     // for the browser console and the viewer checks
  }
  scene.setReplay(R, sim);
  graph.set(R, sim.t0, sim.t1);
  const series = [{ name: planId === "optimal" ? "optimum" : "this plan", costs: R.costs, color: SERIES.plan }];
  if (opt) series.push({ name: "optimum", costs: opt.costs, color: "#898781", dashed: true });
  cost.set(series, sim.t0, sim.t1);
  renderScenario(R);
  renderResult(R);
  const clock = parseClock(params.get("t"), R);
  seek(keepTime ? prevT : clock !== null && !S.booted ? clock : sim.t0);
  S.booted = true;
  if (S.sel && !sim.units.some((u) => u.id === S.sel)) S.sel = null;
  setURL();
}

// --- main loop ---------------------------------------------------------------------------------------------

let TWIN;
let last = performance.now(), lastList = 0;

function frame(now) {
  const dt = Math.min(0.5, (now - last) / 1000);
  last = now;
  if (S.sim) {
    if (S.playing) {
      S.t += dt * S.speed / 60;
      if (S.t >= S.sim.t1) { S.t = S.sim.t1; S.playing = false; $("#play").textContent = "▶"; }
      // the plan's moments as they happen
      while (S.lastEvent + 1 < S.R.events.length && S.R.events[S.lastEvent + 1].t !== null && S.R.events[S.lastEvent + 1].t <= S.t) {
        S.lastEvent++;
        if (S.speed <= 120) toast(S.R.events[S.lastEvent]);
      }
    }
    const states = S.sim.statesAt(S.t);
    scene.update(S.t, states, dt);
    graph.setTime(S.t);
    cost.setTime(S.t);
    $("#clock b").textContent = `${hhmm(S.t)}:${String(Math.floor((S.t % 1) * 60)).padStart(2, "0")}`;
    $("#clock span").textContent = `cost so far ${S.sim.costAt(S.t).toLocaleString()}`;
    if (now - lastList > 250) { renderTrains(states); lastList = now; }
    scene.render();
  }
  requestAnimationFrame(frame);
}

async function main() {
  $("#loadmsg").textContent = "OpenStreetMap: track, stations, buildings…";
  const [twin, tasks] = await Promise.all([loadTwin(), json("/api/tasks")]);
  TWIN = twin;
  const label = (t) => `${t.day} ${t.start}–${t.end} · ${TIER[t.difficulty] || t.difficulty} · ${t.runs} trains`;
  // the eval tasks the models were run on; every task when this server has no model runs
  const evals = tasks.filter((t) => t.split === "eval" && t.model_plans);
  const shown = (evals.length ? evals : tasks).sort((a, b) => a.task_id.localeCompare(b.task_id));
  $("#task").innerHTML = shown.map((t) => `<option value="${t.task_id}">${label(t)}</option>`).join("");
  const want = params.get("task");
  $("#task").value = shown.some((t) => t.task_id === want) ? want : shown.some((t) => t.task_id === FEATURED) ? FEATURED : shown[0].task_id;
  $("#station").innerHTML += twin.stations.map((s) => `<option value="${s.code}">${s.name}</option>`).join("");
  graph = new TrainGraph($("#graph"), $("#gtip"), (t) => seek(t), (u) => select(u, false));
  cost = new CostChart($("#cost"), $("#ctip"), (t) => seek(t));
  $("#loadmsg").textContent = "The plan: every train's simulated times…";
  await loadPlans($("#task").value, params.get("plan"));
  await loadReplay(false);
  if (params.get("follow")) select(params.get("follow"), false);
  if (params.get("cam") && params.get("cam") !== "map") setCam(params.get("cam"));
  $("#loading").classList.add("done");

  $("#task").onchange = async () => { S.sel = null; scene.selected = null; await loadPlans($("#task").value, $("#plan").value); await loadReplay(false); setCam("map"); };
  $("#plan").onchange = () => loadReplay(true);
  $("#play").onclick = () => { S.playing = !S.playing; if (S.playing && S.t >= S.sim.t1) seek(S.sim.t0); $("#play").textContent = S.playing ? "⏸" : "▶"; };
  $("#restart").onclick = () => seek(S.sim.t0);
  $("#speed").onchange = () => { S.speed = +$("#speed").value; };
  $("#cam-map").onclick = () => setCam("map");
  $("#cam-follow").onclick = () => setCam("chase");
  $("#cam-cab").onclick = () => setCam("cab");
  $("#panels").onclick = () => { document.body.classList.toggle("clean"); $("#panels").classList.toggle("on", document.body.classList.contains("clean")); };
  $("#station").onchange = () => { if ($("#station").value) { S.cam = "map"; scene.viewStation($("#station").value); $("#station").value = ""; } };
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT") return;
    if (e.code === "Space") { e.preventDefault(); $("#play").click(); }
    else if (e.code === "ArrowRight") seek(S.t + (e.shiftKey ? 10 : 1));
    else if (e.code === "ArrowLeft") seek(S.t - (e.shiftKey ? 10 : 1));
    else if (e.code === "Home") seek(S.sim.t0);
    else if (e.key === "m") setCam("map");
    else if (e.key === "f") setCam("chase");
    else if (e.key === "c") setCam("cab");
    else if (e.key === "h") $("#panels").click();
  });
  setInterval(setURL, 2000);
  requestAnimationFrame(frame);
}

main().catch((e) => {
  $("#loadmsg").textContent = `Could not start: ${e.message}`;
  console.error(e);
});
