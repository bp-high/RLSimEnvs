// The live part of the line twin: sky and sun for the task's date and time, the trains, the signals that guard each
// single-track section, the level crossings, the disruptions, labels, and the cameras. The scenery is world.js.
//
// It only draws: positions come from motion.js (the simulated times), places from geo.js (the OSM track).

import * as THREE from "three";
import { MapControls } from "three/addons/controls/MapControls.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";

import { CANONICAL } from "./geo.js";
import { animateWorld, buildWorld } from "./world.js";

// series colours, validated for the dark interface (dataviz reference palette, dark slots 1 and 2)
export const KIND_COLOR = { stop: "#3987e5", fast: "#d95926" };
export const STATUS = { critical: "#d03b3b", warning: "#fab219", serious: "#ec835a", good: "#0ca30c" };
const CAR = 26.5, GAP = 1.0, BOGIE = 9.5;
const LAT = 53.22, LON = 6.19;

// --- sun ---------------------------------------------------------------------------------------------------------

function amsterdamOffset(day) {
  // CET (UTC+1), CEST (UTC+2) from the last Sunday of March to the last Sunday of October
  const [y, m, d] = day.split("-").map(Number);
  const lastSunday = (month) => { const t = new Date(Date.UTC(y, month, 0)); return t.getUTCDate() - t.getUTCDay(); };
  const date = Date.UTC(y, m - 1, d);
  const start = Date.UTC(y, 2, lastSunday(3)), end = Date.UTC(y, 9, lastSunday(10));
  return date >= start && date < end ? 2 : 1;
}

export function sunPosition(day, minute) {
  // NOAA's simplified solar position: elevation and azimuth (from north, clockwise), degrees
  const [y, m, d] = day.split("-").map(Number);
  const n = Math.round((Date.UTC(y, m - 1, d) - Date.UTC(y, 0, 0)) / 864e5);
  const utcMin = minute - amsterdamOffset(day) * 60;
  const g = (2 * Math.PI / 365) * (n - 1 + (utcMin / 60 - 12) / 24);
  const eq = 229.18 * (0.000075 + 0.001868 * Math.cos(g) - 0.032077 * Math.sin(g) - 0.014615 * Math.cos(2 * g)
    - 0.040849 * Math.sin(2 * g));
  const decl = 0.006918 - 0.399912 * Math.cos(g) + 0.070257 * Math.sin(g) - 0.006758 * Math.cos(2 * g)
    + 0.000907 * Math.sin(2 * g) - 0.002697 * Math.cos(3 * g) + 0.00148 * Math.sin(3 * g);
  const tst = utcMin + eq + 4 * LON;
  const ha = (tst / 4 - 180) * Math.PI / 180;
  const lat = LAT * Math.PI / 180;
  const cosZ = Math.sin(lat) * Math.sin(decl) + Math.cos(lat) * Math.cos(decl) * Math.cos(ha);
  const zen = Math.acos(Math.max(-1, Math.min(1, cosZ)));
  let az = Math.atan2(Math.sin(ha), Math.cos(ha) * Math.sin(lat) - Math.tan(decl) * Math.cos(lat)) + Math.PI;
  az = (az + 2 * Math.PI) % (2 * Math.PI);
  return { elevation: 90 - zen * 180 / Math.PI, azimuth: az * 180 / Math.PI };
}

// --- trains --------------------------------------------------------------------------------------------------

function carGeometries() {
  const s = new THREE.Shape();
  s.moveTo(-1.42, 1.15); s.lineTo(1.42, 1.15); s.lineTo(1.42, 3.25);
  s.quadraticCurveTo(1.4, 3.92, 0.85, 3.98); s.lineTo(-0.85, 3.98); s.quadraticCurveTo(-1.4, 3.92, -1.42, 3.25);
  s.closePath();
  const body = new THREE.ExtrudeGeometry(s, { depth: CAR - 2.2, bevelEnabled: false });
  body.rotateY(Math.PI / 2);
  body.translate(-(CAR - 2.2) / 2 - 1.1, 0, 0);         // spans -CAR/2 .. CAR/2 - 2.2 (the nose adds 2.2)
  const side = new THREE.Shape();
  side.moveTo(0, 1.15); side.lineTo(0, 3.98); side.lineTo(0.7, 3.92); side.lineTo(1.75, 2.35); side.lineTo(2.2, 1.6);
  side.lineTo(2.2, 1.15); side.closePath();
  const nose = new THREE.ExtrudeGeometry(side, { depth: 2.84, bevelEnabled: false });
  nose.translate(CAR / 2 - 2.2, 0, -1.42);
  const screen = new THREE.PlaneGeometry(1.95, 2.4);
  screen.rotateY(Math.PI / 2);
  screen.rotateZ(-0.98);
  screen.translate(CAR / 2 - 2.2 + 1.24, 3.12, 0);
  const band = new THREE.BoxGeometry(CAR - 4.4, 0.8, 2.88);
  band.translate(-1.1, 2.6, 0);
  const stripe = new THREE.BoxGeometry(CAR - 2.6, 0.26, 2.87);
  stripe.translate(-1.0, 1.62, 0);
  const bogie = new THREE.BoxGeometry(2.7, 0.75, 2.2);
  const lamp = new THREE.CircleGeometry(0.16, 10);
  lamp.rotateY(Math.PI / 2);
  return { body, nose, screen, band, stripe, bogie, lamp };
}

class TrainMesh {
  constructor(shared, kind) {
    const { geo, mats } = shared;
    this.group = new THREE.Group();
    this.cars = [];
    for (let c = 0; c < 2; c++) {
      const car = new THREE.Group();
      const inner = new THREE.Group();            // car 1 is car 0 turned round: both ends are cabs
      if (c === 1) inner.rotation.y = Math.PI;
      for (const [g, m] of [[geo.body, mats.body], [geo.nose, mats.body], [geo.screen, mats.glass],
        [geo.band, mats.glass], [geo.stripe, mats.stripe[kind]]]) {
        const mesh = new THREE.Mesh(g, m);
        mesh.castShadow = true;
        inner.add(mesh);
      }
      for (const x of [-BOGIE, BOGIE]) {
        const b = new THREE.Mesh(geo.bogie, mats.bogie);
        b.position.set(x, 0.62, 0);
        inner.add(b);
      }
      const lamps = [];
      for (const z of [-0.95, 0.95]) {
        const l = new THREE.Mesh(geo.lamp, mats.head);
        l.position.set(CAR / 2 + 0.01, 1.55, z);
        inner.add(l);
        lamps.push(l);
      }
      car.add(inner);
      car.userData.lamps = lamps;
      this.group.add(car);
      this.cars.push(car);
    }
    this.halo = new THREE.Mesh(new THREE.RingGeometry(31, 32.5, 64), mats.halo);
    this.halo.rotation.x = -Math.PI / 2;
    this.halo.position.y = 0.4;
    this.halo.visible = false;
    this.group.add(this.halo);
    this.glow = new THREE.Sprite(mats.glow);
    this.glow.scale.set(24, 24, 1);
    this.group.add(this.glow);
    const el = document.createElement("div");
    el.className = "tlabel";
    this.label = new CSS2DObject(el);
    this.label.position.set(0, 9, 0);
    this.group.add(this.label);
    this.kind = kind;
    this.ghostMats = new Map();
    this.ghosted = false;
  }

  setKind(shared, kind) {
    if (kind === this.kind) return;
    this.kind = kind;
    for (const car of this.cars) car.children[0].children[4].material = shared.mats.stripe[kind];
  }

  // g in [0, 1]: how faded the set is drawn (motion.js: a set the terminal's two tracks have no room for). Its meshes
  // swap to see-through copies of their materials, made once per set, and back to the shared ones at 0.
  setGhost(g) {
    if (!(g > 0) && !this.ghosted) return;
    this.ghosted = g > 0;
    const opacity = 1 - 0.65 * g;
    for (const car of this.cars) {
      car.traverse((o) => {
        if (!o.isMesh) return;
        const base = o.material.userData.ghostOf || o.material;
        if (o.userData.cast === undefined) o.userData.cast = o.castShadow;
        if (!this.ghosted) {
          o.material = base;
          o.castShadow = o.userData.cast;
          return;
        }
        let m = this.ghostMats.get(base);
        if (!m) {
          m = base.clone();
          m.userData = { ghostOf: base };
          m.transparent = true;
          m.depthWrite = false;
          this.ghostMats.set(base, m);
        }
        m.opacity = opacity * (base.opacity ?? 1);
        o.material = m;
        o.castShadow = false;
      });
    }
    this.glow.visible = !this.ghosted;
    this.label.element.classList.toggle("ghost", this.ghosted);
  }
}

// --- the scene -----------------------------------------------------------------------------------------------

export class TwinScene {
  constructor(el, twin, line) {
    this.el = el;
    this.twin = twin;
    this.line = line;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true,
      preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    el.appendChild(this.renderer.domElement);
    this.labels = new CSS2DRenderer();
    this.labels.domElement.className = "labels";
    el.appendChild(this.labels.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.5, 300000);
    this.controls = new MapControls(this.camera, this.labels.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.maxPolarAngle = Math.PI * 0.47;
    this.controls.minDistance = 12;
    this.controls.maxDistance = 140000;
    this.controls.zoomToCursor = true;

    this.hemi = new THREE.HemisphereLight(0xeef3ff, 0x7c7a62, 2.0);
    this.ambient = new THREE.AmbientLight(0xffffff, 0.25);
    this.sun = new THREE.DirectionalLight(0xffffff, 3.0);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    const sc = this.sun.shadow.camera;
    sc.left = sc.bottom = -350; sc.right = sc.top = 350; sc.near = 10; sc.far = 3000;
    this.sun.shadow.bias = -0.0004;
    this.scene.add(this.hemi, this.ambient, this.sun, this.sun.target);
    this.scene.fog = new THREE.Fog(0xbcd3e6, 4000, 60000);
    this.scene.background = new THREE.Color(0xbcd3e6);

    this.world = buildWorld(twin, this.scene);
    this.shared = this.sharedAssets();
    this.trains = new Map();
    this.signals = [];
    this.crossings = this.buildCrossings();
    this.closures = [];
    this.stationLabels = this.buildStationLabels();
    this.placeLabels = this.buildPlaceLabels();
    this.followId = null;
    this.mode = "map";
    this.lastFollow = null;
    this.selected = null;
    this.night = 0;
    this.resize();
    window.addEventListener("resize", () => this.resize());
    this.viewAll();
  }

  sharedAssets() {
    const geo = carGeometries();
    const glowTex = (() => {
      const cv = document.createElement("canvas");
      cv.width = cv.height = 64;
      const g = cv.getContext("2d");
      const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
      grd.addColorStop(0, "rgba(255,244,214,1)"); grd.addColorStop(0.25, "rgba(255,230,170,0.5)");
      grd.addColorStop(1, "rgba(255,220,150,0)");
      g.fillStyle = grd; g.fillRect(0, 0, 64, 64);
      return new THREE.CanvasTexture(cv);
    })();
    const mats = {
      body: new THREE.MeshStandardMaterial({ color: 0xeef0f2, roughness: 0.45, metalness: 0.15 }),
      glass: new THREE.MeshStandardMaterial({ color: 0x18222c, roughness: 0.15, metalness: 0.6 }),
      bogie: new THREE.MeshLambertMaterial({ color: 0x2a2a2a }),
      stripe: { stop: new THREE.MeshStandardMaterial({ color: KIND_COLOR.stop, roughness: 0.5 }),
        fast: new THREE.MeshStandardMaterial({ color: KIND_COLOR.fast, roughness: 0.5 }) },
      head: new THREE.MeshBasicMaterial({ color: 0xfff6dc }),
      tail: new THREE.MeshBasicMaterial({ color: 0xff2a1a }),
      halo: new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85, depthWrite: false }),
      glow: new THREE.SpriteMaterial({ map: glowTex, color: 0xffffff, transparent: true, opacity: 0,
        depthWrite: false, blending: THREE.AdditiveBlending }),
    };
    return { geo, mats };
  }

  resize() {
    const w = this.el.clientWidth || window.innerWidth, h = this.el.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h);
    this.labels.setSize(w, h);
    for (const m of [...(this.lineMats || []), ...(this.closureMats || [])]) m.resolution.set(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // --- static labels -----------------------------------------------------------------------------------------

  buildStationLabels() {
    const out = [];
    for (const st of this.twin.stations) {
      const el = document.createElement("div");
      el.className = "slabel";
      el.textContent = st.name;
      const o = new CSS2DObject(el);
      o.position.set(st.x, 14, -st.y);
      this.scene.add(o);
      out.push({ code: st.code, obj: o, x: st.x, y: st.y });
    }
    return out;
  }

  buildPlaceLabels() {
    const out = [];
    const stations = new Set(this.twin.stations.map((s) => s.name.toLowerCase()));
    for (const p of this.twin.places) {
      if (p.r > 3) continue;
      const el = document.createElement("div");
      el.className = `plabel r${p.r}`;
      el.textContent = p.n;
      const o = new CSS2DObject(el);
      o.position.set(p.x, 30, -p.y);
      this.scene.add(o);
      out.push({ obj: o, r: p.r, x: p.x, y: p.y, dup: stations.has(p.n.toLowerCase()) });
    }
    return out;
  }

  // --- crossings, signals, closures ------------------------------------------------------------------------

  buildCrossings() {
    const out = [];
    const cv = document.createElement("canvas");
    cv.width = 64; cv.height = 8;
    const g = cv.getContext("2d");
    for (let i = 0; i < 8; i++) { g.fillStyle = i % 2 ? "#ffffff" : "#d42a2a"; g.fillRect(i * 8, 0, 8, 8); }
    const stripes = new THREE.CanvasTexture(cv);
    const armMat = new THREE.MeshLambertMaterial({ map: stripes });
    const postMat = new THREE.MeshLambertMaterial({ color: 0xdddddd });
    const plateMat = new THREE.MeshLambertMaterial({ color: 0x3e3f42, depthTest: false, depthWrite: false });
    const offMat = new THREE.MeshBasicMaterial({ color: 0x3a1010 });
    this.lampOn = new THREE.MeshBasicMaterial({ color: 0xff3020 });
    const armGeo = new THREE.BoxGeometry(5.6, 0.14, 0.14);
    armGeo.translate(2.8, 0, 0);
    const postGeo = new THREE.CylinderGeometry(0.08, 0.1, 1.3, 6);
    const lampGeo = new THREE.CircleGeometry(0.13, 10);
    for (const c of this.twin.crossings) {
      const grp = new THREE.Group();
      grp.position.set(c.x, 0, -c.y);
      const half = c.n * 2.4 + 2.2;
      const plate = new THREE.Mesh(new THREE.BoxGeometry(half * 2, 0.05, 5.5), plateMat);
      plate.rotation.y = c.r;
      plate.position.y = 0.2;
      plate.renderOrder = -4;                          // painted with the flat layers (see world.js)
      grp.add(plate);
      const arms = [], lamps = [];
      for (const side of [1, -1]) {
        const ux = Math.cos(c.r) * side, uy = Math.sin(c.r) * side;     // along the road, away from the track
        const vx = -uy, vy = ux;                                         // across the road
        const bx = ux * (half + 1.2) + vx * 3.1, by = uy * (half + 1.2) + vy * 3.1;
        const post = new THREE.Mesh(postGeo, postMat);
        post.position.set(bx, 0.65, -by);
        const pivot = new THREE.Group();
        pivot.position.set(bx, 1.05, -by);
        pivot.rotation.y = Math.atan2(-vy, -vx);                         // arm points back across the road
        const tilt = new THREE.Group();
        tilt.rotation.z = Math.PI / 2;                                   // open: arm up
        const arm = new THREE.Mesh(armGeo, armMat);
        tilt.add(arm);
        pivot.add(tilt);
        for (const dz of [-0.2, 0.2]) {
          const l = new THREE.Mesh(lampGeo, offMat);
          l.position.set(bx + ux * 0.01, 1.25, -by + dz);
          l.rotation.y = Math.atan2(-uy, ux) + Math.PI / 2;
          grp.add(l);
          lamps.push(l);
        }
        grp.add(post, pivot);
        arms.push(tilt);
      }
      this.scene.add(grp);
      out.push({ ...c, arms, lamps, offMat, angle: Math.PI / 2 });
    }
    return out;
  }

  clearReplayObjects() {
    for (const s of this.signals) this.scene.remove(s.group);
    for (const c of this.closures) { this.scene.remove(c.mesh); this.scene.remove(c.label); if (c.line) this.scene.remove(c.line); }
    this.closureMats = [];
    for (const v of this.breaks || []) { this.scene.remove(v.mesh); this.scene.remove(v.label); }
    this.signals = [];
    this.closures = [];
    this.breaks = [];
  }

  buildBreaks(R) {
    // where and when the plan breaks a rule: a red ring on the spot for a few minutes around the moment
    const km = Object.fromEntries(R.points.map((p) => [p.code, p.km]));
    const ringGeo = new THREE.RingGeometry(40, 48, 48);
    ringGeo.rotateX(-Math.PI / 2);
    const mat = new THREE.MeshBasicMaterial({ color: STATUS.critical, transparent: true, opacity: 0.9, depthTest: false,
      depthWrite: false });
    for (const v of R.violations) {
      if (!v.time || !v.where) continue;
      const [h, m] = v.time.split(":").map(Number);
      let t = h * 60 + m;
      if (t < R.start - 120) t += 1440;
      const ends = v.where.split("-").filter((c) => km[c] !== undefined);
      if (!ends.length) continue;
      const k = ends.reduce((a, c) => a + km[c], 0) / ends.length;
      const p = this.pose("E", k);
      const mesh = new THREE.Mesh(ringGeo, mat);
      mesh.position.set(p.x, 1.5, -p.y);
      mesh.renderOrder = 6;
      const el = document.createElement("div");
      el.className = "vlabel";
      el.innerHTML = `<b>${v.rule.replace("_", " ")}</b> ${v.trains.join(" · ")} at ${v.where}`;
      const label = new CSS2DObject(el);
      label.position.set(p.x, 30, -p.y);
      this.scene.add(mesh, label);
      this.breaks.push({ t, mesh, label });
    }
  }

  pose(dir, km, lw = "8", gn = "5a", lanes = null) {
    const p = this.line.place(dir, lw, gn, km, lanes);
    const q = this.line.place(dir, lw, gn, km + (dir === "E" ? 0.01 : -0.01), lanes);
    return { x: p[0], y: p[1], h: Math.atan2(q[1] - p[1], q[0] - p[0]) };
  }

  buildSignals(R) {
    // one signal at each entry of every single-track section, on the right of the track; red while a train is
    // inside or the section is closed
    const km = Object.fromEntries(R.points.map((p) => [p.code, p.km]));
    const kind = Object.fromEntries(R.points.map((p) => [p.code, p.kind]));
    const mastGeo = new THREE.CylinderGeometry(0.09, 0.11, 4.2, 6);
    const headGeo = new THREE.BoxGeometry(0.35, 0.95, 0.5);
    const lampGeo = new THREE.CircleGeometry(0.12, 12);
    const grey = new THREE.MeshLambertMaterial({ color: 0x2b2b2b });
    this.sigRed = new THREE.MeshBasicMaterial({ color: 0xff2a1a });
    this.sigGreen = new THREE.MeshBasicMaterial({ color: 0x28e070 });
    this.sigOff = new THREE.MeshBasicMaterial({ color: 0x1a1a1a });
    R.sections.forEach((sec, i) => {
      if (!sec.single) return;
      for (const dir of ["E", "W"]) {
        const at = dir === "E" ? sec.a : sec.b;
        const sign = dir === "E" ? 1 : -1;
        const k = kind[at] === "junction" ? km[at] - sign * 0.18 : km[at] + sign * 0.12;
        const p = this.pose(dir, k);
        const rx = Math.sin(p.h), ry = -Math.cos(p.h);                  // right of the direction of travel
        const group = new THREE.Group();
        group.position.set(p.x + rx * 2.6, 0, -(p.y + ry * 2.6));
        group.rotation.y = p.h + Math.PI;                               // the lamps face oncoming trains
        const mast = new THREE.Mesh(mastGeo, grey);
        mast.position.y = 2.1;
        const head = new THREE.Mesh(headGeo, grey);
        head.position.y = 4.3;
        const red = new THREE.Mesh(lampGeo, this.sigOff), green = new THREE.Mesh(lampGeo, this.sigOff);
        red.rotation.y = green.rotation.y = Math.PI / 2;
        red.position.set(0.18, 4.52, 0); green.position.set(0.18, 4.12, 0);
        group.add(mast, head, red, green);
        this.scene.add(group);
        this.signals.push({ section: i, dir, group, red, green });
      }
    });
  }

  buildClosures(R) {
    const km = Object.fromEntries(R.points.map((p) => [p.code, p.km]));
    const single = R.sections.map((s) => s.single);
    const idx = Object.fromEntries(R.points.map((p, i) => [p.code, i]));
    for (const c of R.closures) {
      let k0, k1;
      if (c.kind === "track") { k0 = km[c.where[0]] - 0.2; k1 = km[c.where[0]] + 0.2; }
      else { k0 = Math.min(km[c.where[0]], km[c.where[c.where.length - 1]]); k1 = Math.max(km[c.where[0]], km[c.where[c.where.length - 1]]); }
      const pos = [], index = [];
      const lanes = ["E"];
      const i0 = Math.min(idx[c.where[0]], idx[c.where[c.where.length - 1]]);
      const i1 = Math.max(idx[c.where[0]], idx[c.where[c.where.length - 1]]);
      if (c.kind === "track" || single.slice(i0, Math.max(i1, i0 + 1)).some((s) => !s)) lanes.push("W");
      for (const lane of lanes) {
        const base = pos.length / 3;
        let n = 0;
        for (let k = k0; k <= k1 + 1e-9; k += Math.max(0.005, (k1 - k0) / 400)) {
          const p = this.pose(lane, k, "8", "5a");
          const nx = -Math.sin(p.h) * 2.6, ny = Math.cos(p.h) * 2.6;
          pos.push(p.x + nx, 0.45, -(p.y + ny), p.x - nx, 0.45, -(p.y - ny));
          n++;
        }
        for (let j = 0; j < n - 1; j++) {
          const v = base + 2 * j;
          index.push(v, v + 2, v + 1, v + 1, v + 2, v + 3);
        }
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      g.setIndex(index);
      const color = c.kind === "closed" ? STATUS.critical : c.kind === "slow" ? STATUS.warning : STATUS.serious;
      const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.7, depthWrite: false,
        side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(g, mat);
      mesh.renderOrder = 5;
      // from high up: a thick line over the overview line while it lasts
      const lpts = [];
      for (let k = k0; k <= k1 + 1e-9; k += Math.max(0.02, (k1 - k0) / 100)) {
        const q = this.pose("E", Math.min(k, k1));
        lpts.push(q.x, 1.2, -q.y);
      }
      const lmat = new LineMaterial({ color, linewidth: 7, depthTest: false, depthWrite: false, transparent: true });
      lmat.resolution.set(this.el.clientWidth || 1, this.el.clientHeight || 1);
      (this.closureMats ||= []).push(lmat);
      const lg = new LineGeometry();
      lg.setPositions(lpts);
      const line = new Line2(lg, lmat);
      line.renderOrder = -1;
      const el = document.createElement("div");
      el.className = `clabel ${c.kind}`;
      const what = c.kind === "closed" ? "Closed" : c.kind === "slow" ? `Slow +${c.extra} min` : "1 track out of use";
      el.innerHTML = `<b>${what}</b> ${c.cause || ""}`;
      const label = new CSS2DObject(el);
      const mid = this.pose("E", (k0 + k1) / 2);
      label.position.set(mid.x, 22, -mid.y);
      this.scene.add(mesh, label, line);
      this.closures.push({ ...c, mesh, label, mat, line });
    }
  }

  buildOverviewLine(R) {
    // the line as seen from high up, where 3 m of track is under a pixel: white where it is single track (one train
    // at a time between stations), grey where it is double; a dark casing under both
    const km = Object.fromEntries(R.points.map((p) => [p.code, p.km]));
    this.lineMats = [];
    this.overview = new THREE.Group();
    const mk = (color, width, order) => {
      const m = new LineMaterial({ color, linewidth: width, depthTest: false, depthWrite: false, transparent: true });
      m.resolution.set(this.el.clientWidth || 1, this.el.clientHeight || 1);
      this.lineMats.push(m);
      m.userData.order = order;
      return m;
    };
    const casing = mk(0x1a1a19, 7, -3), single = mk(0xffffff, 3, -2), double = mk(0x9a9a96, 3, -2);
    for (const sec of R.sections) {
      const pts = [];
      for (let k = km[sec.a]; k <= km[sec.b] + 1e-9; k += Math.max(0.02, (km[sec.b] - km[sec.a]) / 120)) {
        const p = this.line.place("E", "8", "5a", Math.min(k, km[sec.b]));
        pts.push(p[0], 1, -p[1]);
      }
      for (const mat of [casing, sec.single ? single : double]) {
        const g = new LineGeometry();
        g.setPositions(pts);
        const l = new Line2(g, mat);
        l.computeLineDistances();
        l.renderOrder = mat.userData.order;
        this.overview.add(l);
      }
    }
    this.scene.add(this.overview);
  }

  setReplay(R, sim) {
    this.R = R;
    this.sim = sim;
    if (!this.overview) this.buildOverviewLine(R);
    this.clearReplayObjects();
    this.buildSignals(R);
    this.buildClosures(R);
    this.buildBreaks(R);
    const keep = new Set(sim.units.map((u) => u.id));
    for (const [id, tm] of this.trains) {
      if (!keep.has(id)) {
        this.scene.remove(tm.group);
        // a label in a removed group stays on the page (three.js removes a label's element only when the label itself
        // is removed), frozen where it was: a task with fewer sets, then one with more, showed some trains twice
        tm.label.element.remove();
        this.trains.delete(id);
      }
    }
    for (const u of sim.units) {
      if (!this.trains.has(u.id)) {
        const tm = new TrainMesh(this.shared, (u.runs[0] && u.runs[0].kind) || "stop");
        tm.unitId = u.id;
        tm.group.traverse((o) => { o.userData.unit = u.id; });
        this.scene.add(tm.group);
        this.trains.set(u.id, tm);
      }
    }
    this.sunT = null;
  }

  setSun({ elevation, azimuth }) {
    const el = THREE.MathUtils.degToRad(Math.max(elevation, -8));
    const az = THREE.MathUtils.degToRad(azimuth);
    this.sunDir = new THREE.Vector3(Math.sin(az) * Math.cos(el), Math.sin(Math.max(el, 0.03)), -Math.cos(az) * Math.cos(el));
    const day = THREE.MathUtils.smoothstep(elevation, -6, 10);        // 0 night .. 1 day
    const dusk = Math.max(0, 1 - Math.abs(elevation - 1) / 9) * (elevation > -8 ? 1 : 0);
    this.night = 1 - day;
    const sky = new THREE.Color(0x0d1830).lerp(new THREE.Color(0xa9c8e6), day);
    sky.lerp(new THREE.Color(0xe39a62), dusk * 0.45);
    this.scene.background = sky;
    this.scene.fog.color.copy(sky);
    this.hemi.intensity = 0.5 + 2.6 * day;
    this.hemi.color.set(0xeef3ff).lerp(new THREE.Color(0x5a6a9a), 1 - day);
    this.sun.intensity = 3.6 * day + 0.8 * dusk;
    this.sun.color.set(0xffffff).lerp(new THREE.Color(0xffa060), dusk);
    this.renderer.toneMappingExposure = 0.8 + 0.35 * day;
    this.shared.mats.glow.opacity = 0.9 * this.night;
  }

  // --- cameras ---------------------------------------------------------------------------------------------

  viewAll() {
    const [x0, y0, x1, y1] = this.twin.bounds;
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    this.mode = "map";
    this.followId = null;
    this.controls.enabled = true;
    // high enough that the whole line fits the width of the view, whatever its shape
    const half = (x1 - x0) / 2 - 2000;
    const tanH = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * this.camera.aspect;
    const h = Math.max(20000, (half / tanH) * 0.95);
    this.controls.target.set(cx, 0, -cy);
    this.camera.position.set(cx, h, -cy + h * 0.44);
    this.controls.update();
  }

  viewAt(x, y, dist = 700, pitch = 0.9) {
    this.mode = "map";
    this.followId = null;
    this.controls.enabled = true;
    this.controls.target.set(x, 0, -y);
    this.camera.position.set(x - dist * 0.35, dist * Math.sin(pitch), -y + dist * Math.cos(pitch));
    this.controls.update();
  }

  viewStation(code) {
    const st = this.twin.stations.find((s) => s.code === code);
    if (st) this.viewAt(st.x, st.y, 650, 0.75);
  }

  follow(unitId, mode = "chase") {
    this.followId = unitId;
    this.mode = mode;
    this.lastFollow = null;
    this.controls.enabled = mode !== "cab";
    if (mode === "chase") {
      const tm = this.trains.get(unitId);
      if (tm) {
        const p = tm.group.position.clone();
        this.controls.target.copy(p);
        this.camera.position.set(p.x - 140, 95, p.z + 160);
        this.controls.update();
      }
    }
  }

  // --- per frame ---------------------------------------------------------------------------------------------

  update(t, states, dtReal) {
    const sim = this.sim;
    if (!sim) return;
    if (this.sunT === null || Math.abs(t - this.sunT) > 0.5) {      // the sun moves with the clock
      this.setSun(sunPosition(this.R.day, t));
      this.sunT = t;
    }
    // trains
    const seen = new Set();
    for (const s of states) {
      const tm = this.trains.get(s.unit);
      if (!tm) continue;
      seen.add(s.unit);
      tm.group.visible = true;
      tm.setKind(this.shared, s.kind);
      const offs = [CAR / 2 + GAP / 2 + BOGIE, CAR / 2 + GAP / 2 - BOGIE, -(CAR / 2 + GAP / 2) + BOGIE,
        -(CAR / 2 + GAP / 2) - BOGIE];
      const pts = this.line.body(s.dir, s.lw, s.gn, s.km, offs, s.lanes);
      const mid = this.line.place(s.dir, s.lw, s.gn, s.km, s.lanes);
      tm.group.position.set(mid[0], 0, -mid[1]);
      for (let c = 0; c < 2; c++) {
        const [a, b] = [pts[2 * c], pts[2 * c + 1]];          // its front and rear bogie, so cars follow curves
        const car = tm.cars[c];
        car.position.set((a[0] + b[0]) / 2 - mid[0], 0, -((a[1] + b[1]) / 2 - mid[1]));
        car.rotation.y = Math.atan2(a[1] - b[1], a[0] - b[0]);
        // head lamps on the leading cab, tail lamps behind
        for (const l of car.userData.lamps) l.material = c === 0 ? this.shared.mats.head : this.shared.mats.tail;
      }
      const head = pts[0];
      tm.glow.position.set(head[0] - mid[0], 2.0, -(head[1] - mid[1]));
      tm.halo.visible = this.selected === s.unit;
      tm.state = s;
      const delay = Math.round(s.delay);
      tm.label.element.innerHTML = `<i style="background:${KIND_COLOR[s.kind]}"></i>${s.train}`
        + (delay > 0 ? `<span class="late">+${delay}</span>` : "");
      tm.label.element.classList.toggle("sel", this.selected === s.unit);
      tm.setGhost(s.ghost || 0);
    }
    for (const [id, tm] of this.trains) if (!seen.has(id)) tm.group.visible = false;

    // signals
    const occ = sim.occupancy(states);
    for (const sg of this.signals) {
      const sec = this.R.sections[sg.section];
      const closed = this.closures.some((c) => c.kind === "closed" && t >= c.start && t < c.end
        && this.covers(c, sec));
      const red = closed || (occ[sg.section] && occ[sg.section].length > 0);
      sg.red.material = red ? this.sigRed : this.sigOff;
      sg.green.material = red ? this.sigOff : this.sigGreen;
    }
    // level crossings: down while a train is within ~35 s of reaching it, until it has passed
    const soon = states.map((s) => {
      const un = sim.units.find((u) => u.id === s.unit);
      const f = un ? sim.stateAt(un, t + 0.6) : s;
      return [Math.min(s.km, f.km) - 0.035, Math.max(s.km, f.km) + 0.035];
    });
    const blink = Math.floor(performance.now() / 450) % 2;
    for (const c of this.crossings) {
      const down = soon.some(([a, b]) => c.km >= a && c.km <= b);
      const target = down ? 0 : Math.PI / 2;
      c.angle += (target - c.angle) * Math.min(1, dtReal * 2.5);
      for (const a of c.arms) a.rotation.z = c.angle;
      c.lamps.forEach((l, i) => { l.material = down && (i + blink) % 2 ? this.lampOn : c.offMat; });
    }
    // disruptions: visible while they last, pulsing
    const pulse = 0.45 + 0.25 * Math.sin(performance.now() / 300);
    for (const c of this.closures) {
      const on = t >= c.start && t < c.end;
      c.mesh.visible = on;
      c.label.visible = on;
      c.mat.opacity = pulse;
      c.line.visible = on && this.overview && this.overview.visible;
    }
    for (const v of this.breaks) {
      const on = t >= v.t - 1 && t <= v.t + 3;
      v.mesh.visible = v.label.visible = on;
      if (on) v.mesh.scale.setScalar(1 + 0.15 * Math.sin(performance.now() / 160));
    }
    animateWorld(this.world, dtReal);
    this.updateCamera();
    this.updateLabels();
  }

  covers(c, sec) {
    const idx = Object.fromEntries(this.R.points.map((p, i) => [p.code, i]));
    const lo = Math.min(idx[c.where[0]], idx[c.where[c.where.length - 1]]);
    const hi = Math.max(idx[c.where[0]], idx[c.where[c.where.length - 1]]);
    const i = idx[sec.a];
    return i >= lo && i < hi;
  }

  updateCamera() {
    const tm = this.followId ? this.trains.get(this.followId) : null;
    if (tm && tm.group.visible && tm.state) {
      const s = tm.state;
      if (this.mode === "cab") {
        // from the train as drawn (by the route it is on): a standing set that will leave the other way is seen from
        // its other end, looking back along that route. (The other direction's route at the same km is up to 16 m
        // away along the track, which put the camera inside the train.)
        const k = s.cab && s.cab !== s.dir ? -1 : 1;
        const pts = this.line.body(s.dir, s.lw, s.gn, s.km, [k * (CAR + 0.5), k * (CAR + 160)], s.lanes);
        this.camera.position.set(pts[0][0], 3.4, -pts[0][1]);
        this.camera.lookAt(pts[1][0], 2.2, -pts[1][1]);
      } else {
        const p = tm.group.position;
        if (this.lastFollow) {
          const d = p.clone().sub(this.lastFollow);
          this.camera.position.add(d);
          this.controls.target.add(d);
        }
        this.lastFollow = p.clone();
      }
    }
    if (this.mode !== "cab") this.controls.update();
    // shadows follow the view
    const tgt = this.mode === "cab" ? this.camera.position : this.controls.target;
    const h = this.camera.position.y;
    this.sun.castShadow = h < 2500;
    this.sun.target.position.copy(tgt);
    this.sun.position.copy(tgt).addScaledVector(this.sunDir || new THREE.Vector3(0.3, 0.8, 0.3), 1500);
    this.scene.fog.near = Math.max(1500, h * 2);
    this.scene.fog.far = Math.max(20000, h * 7);
    if (this.overview) {                               // the overview line fades in as the track gets too thin to see
      const a = THREE.MathUtils.smoothstep(h, 1200, 4000);
      this.overview.visible = a > 0.01;
      for (const m of this.lineMats) m.opacity = a;
      for (const m of this.closureMats || []) m.opacity = a;
    }
    const trees = this.world.getObjectByName("trees");
    if (trees) {                                       // trees only near the view: there are tens of thousands
      trees.visible = h < 2500;
      if (trees.visible) for (const im of trees.children) im.visible = im.userData.centre.distanceTo(tgt) < 3500 + h;
    }
  }

  updateLabels() {
    const cam = this.camera.position;
    const h = cam.y;
    const narrow = window.innerWidth < 760 && h > 20000;  // a phone's overview: terminals and turn points only
    for (const l of this.placeLabels) {
      const show = !l.dup && !narrow && ((l.r <= 1 && h > 2500) || (l.r === 2 && h > 900 && h < 20000) || (l.r === 3 && h < 6000 && h > 300));
      l.obj.visible = show;
    }
    for (const l of this.stationLabels) {                // far stations only from high up
      l.obj.visible = (h > 2000 || Math.hypot(l.x - cam.x, -l.y - cam.z) < 12000)
        && (!narrow || ["LW", "HDG", "BP", "ZH", "GN"].includes(l.code));
    }
    for (const tm of this.trains.values()) {
      const d = tm.group.position.distanceTo(cam);
      tm.label.visible = tm.group.visible;
      tm.label.element.classList.toggle("far", d > 9000);
    }
    for (const c of this.closures) if (c.label.visible) c.label.element.classList.toggle("far", h > 15000);
  }

  render() {
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
  }

  pick(clientX, clientY) {
    const r = this.renderer.domElement.getBoundingClientRect();
    const v = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(v, this.camera);
    const objs = [];
    for (const tm of this.trains.values()) if (tm.group.visible) objs.push(tm.group);
    const hit = ray.intersectObjects(objs, true).find((h) => h.object.userData.unit);
    if (hit) return hit.object.userData.unit;
    // generous fallback: the train nearest the click on screen
    let best = null, bd = 28;
    for (const tm of this.trains.values()) {
      if (!tm.group.visible) continue;
      const p = tm.group.position.clone().project(this.camera);
      const sx = (p.x + 1) / 2 * r.width + r.left, sy = (1 - p.y) / 2 * r.height + r.top;
      const d = Math.hypot(sx - clientX, sy - clientY);
      if (d < bd && p.z < 1) { bd = d; best = tm.unitId; }
    }
    return best;
  }
}

export { CANONICAL };
