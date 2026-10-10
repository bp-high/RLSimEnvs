// The static scenery of the line twin, built once from twin.json (OpenStreetMap, see SOURCES.md): ground, land cover,
// water, roads, buildings, trees, landmarks, track and platforms. Everything is merged into a few large meshes,
// tiled so the renderer can skip what is off screen.
//
// World frame: x east, z south (-north), y up, metres; the twin's (x, y) maps to (x, -y).

import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { CLEARANCE, overTrack } from "./geo.js";

export const TILE = 2000;          // metres: tile size for culling
const Y = { landcover: 0.02, farmland: 0.01, water: 0.05, waterway: 0.06, road: 0.09, platform: 0, track: 0.12 };

export const COLORS = {
  ground: 0x789a58,
  landcover: {
    0: 0xbfb6a4, 1: 0x82a85c, 2: 0x7fae5a, 3: 0xb4aebb, 4: 0xc8b9aa, 5: 0xa39d92, 6: 0xb3a789, 7: 0x3e6a35,
    8: 0x7ea75a, 9: 0x8a9255, 10: 0x6d9a7f, 11: 0x6f9a55, 12: 0x7d9c6a, 13: 0xd2dbd8, 14: 0x7fa257, 15: 0x98a862,
    16: 0xd8c99a,
  },
  water: 0x4d7da6,
  road: [0x4c4d51, 0x55565a, 0x5b5c60, 0x616265, 0x67686b, 0x6e6e70, 0x9b8b6a, 0xa2584b],
  roadWidth: [22, 13, 10, 8, 5.5, 4, 3, 2.6],
  waterwayWidth: [22, 14, 4, 1.8],
  building: [
    [0x9b5a43, 0x4a3a34], [0xa77d63, 0x585654], [0x7b5d4b, 0x3d3a36], [0x9ea3a8, 0x7c8186], [0xb8a998, 0x6f6a66],
    [0x8a4e3b, 0x3f3936], [0xd5e2de, 0xc3d2ce], [0xb7724f, 0x4b4744],
  ],
  ballast: 0x8b8379, rail: 0xc4c7cb, sleeper: 0x5a4a3d, platform: 0xbab6ae, platformEdge: 0xe0bf38, bufferStop: 0xa8432f,
  trunk: 0x5b4636, crown: 0x3f6b33, crown2: 0x4f7a3b,
};

const col = new THREE.Color();

function decode(flat) {
  const pts = [];
  for (let i = 0; i < flat.length; i += 2) pts.push([flat[i] / 10, flat[i + 1] / 10]);
  return pts;
}

function tileKey(x, y) {
  return `${Math.floor(x / TILE)},${Math.floor(y / TILE)}`;
}

// growable typed buffers for one mesh
class Buf {
  constructor() { this.pos = []; this.col = []; this.uv = null; this.idx = []; }
  get n() { return this.pos.length / 3; }
  vert(x, y, z, c) { this.pos.push(x, y, z); this.col.push(c.r, c.g, c.b); return this.n - 1; }
  // a triangle that faces up (+y), whichever way its vertices were given
  up(a, b, c) {
    const P = this.pos;
    const ax = P[3 * a], az = P[3 * a + 2], bx = P[3 * b], bz = P[3 * b + 2], cx = P[3 * c], cz = P[3 * c + 2];
    const ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
    if (ny >= 0) this.idx.push(a, b, c); else this.idx.push(a, c, b);
  }
  // a triangle that faces down (-y): the underside of a roof over the track
  down(a, b, c) {
    const P = this.pos;
    const ax = P[3 * a], az = P[3 * a + 2], bx = P[3 * b], bz = P[3 * b + 2], cx = P[3 * c], cz = P[3 * c + 2];
    const ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
    if (ny < 0) this.idx.push(a, b, c); else this.idx.push(a, c, b);
  }
  geometry() {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(this.col, 3));
    if (this.uv) g.setAttribute("uv", new THREE.Float32BufferAttribute(this.uv, 2));
    g.setIndex(this.n > 65535 ? new THREE.Uint32BufferAttribute(this.idx, 1) : new THREE.Uint16BufferAttribute(this.idx, 1));
    g.computeVertexNormals();
    g.computeBoundingSphere();
    return g;
  }
}

function tiled(map, key) {
  let b = map.get(key);
  if (!b) { b = new Buf(); map.set(key, b); }
  return b;
}

// Flat layers (ground, land cover, water, roads, track) are painted in a fixed order before anything 3D, without
// depth: they all lie on the ground, and centimetre offsets cannot win a depth test from kilometres away.
function flat(material, order) {
  material.side = THREE.DoubleSide;
  material.depthTest = false;
  material.depthWrite = false;
  material.userData.order = order;
  return material;
}

function meshes(map, material, opts = {}) {
  const group = new THREE.Group();
  for (const b of map.values()) {
    if (!b.n) continue;
    const m = new THREE.Mesh(b.geometry(), material);
    m.receiveShadow = !!opts.receive;
    m.castShadow = !!opts.cast;
    if (material.userData.order !== undefined) m.renderOrder = material.userData.order;
    group.add(m);
  }
  return group;
}

// flat polygons (outer ring + holes), triangulated, at height y, into tiled buffers
function addPolygon(map, outer, holes, y, c) {
  if (outer.length < 3) return;
  const contour = outer.map(([x, yy]) => new THREE.Vector2(x, yy));
  const hs = holes.map((h) => h.map(([x, yy]) => new THREE.Vector2(x, yy)));
  let tris;
  try {
    tris = THREE.ShapeUtils.triangulateShape(contour, hs);
  } catch {
    return;
  }
  const all = [...outer, ...holes.flat()];
  let cx = 0, cy = 0;
  for (const [x, yy] of outer) { cx += x; cy += yy; }
  const b = tiled(map, tileKey(cx / outer.length, cy / outer.length));
  const base = b.n;
  for (const [x, yy] of all) b.vert(x, y, -yy, c);
  for (const [a, bb, cc] of tris) b.up(base + a, base + bb, base + cc);
}

// ribbons along polylines (roads, rivers, ditches) with mitred joins
function addRibbon(map, pts, width, y, c, uvScale = 0) {
  if (pts.length < 2) return;
  const h = width / 2;
  const n = pts.length;
  const mid = pts[n >> 1];
  const b = tiled(map, tileKey(mid[0], mid[1]));
  if (uvScale && !b.uv) b.uv = [];
  const base = b.n;
  let along = 0;
  for (let i = 0; i < n; i++) {
    const p = pts[i], a = pts[Math.max(0, i - 1)], q = pts[Math.min(n - 1, i + 1)];
    let dx = q[0] - a[0], dy = q[1] - a[1];
    const L = Math.hypot(dx, dy) || 1;
    dx /= L; dy /= L;
    let nx = -dy, ny = dx, k = 1;
    if (i > 0 && i < n - 1) {                       // mitre: keep the width constant around the corner
      const ex = p[0] - a[0], ey = p[1] - a[1], el = Math.hypot(ex, ey) || 1;
      const sx = -ey / el, sy = ex / el;
      k = 1 / Math.max(0.35, nx * sx + ny * sy);
    }
    if (i > 0) along += Math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]);
    b.vert(p[0] + nx * h * k, y, -(p[1] + ny * h * k), c);
    b.vert(p[0] - nx * h * k, y, -(p[1] - ny * h * k), c);
    if (uvScale) b.uv.push(0, along / uvScale, 1, along / uvScale);
  }
  for (let i = 0; i < n - 1; i++) {
    const v = base + 2 * i;
    b.up(v, v + 1, v + 2);
    b.up(v + 1, v + 3, v + 2);
  }
}

function sleeperTexture() {
  const cv = document.createElement("canvas");
  cv.width = 64; cv.height = 64;
  const g = cv.getContext("2d");
  g.fillStyle = "#8b8379"; g.fillRect(0, 0, 64, 64);
  for (let i = 0; i < 400; i++) {                  // ballast grain
    const v = 110 + Math.floor(Math.random() * 60);
    g.fillStyle = `rgb(${v},${v - 6},${v - 14})`;
    g.fillRect(Math.random() * 64, Math.random() * 64, 2, 2);
  }
  g.fillStyle = "#5a4a3d";
  g.fillRect(6, 4, 52, 22);                       // one sleeper per 0.6 m (two per texture repeat of 1.2 m)
  g.fillRect(6, 36, 52, 22);
  const t = new THREE.CanvasTexture(cv);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 8;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export function buildWorld(twin, scene, opts = {}) {
  const root = new THREE.Group();
  root.name = "world";
  const [x0, y0, x1, y1] = twin.bounds;
  const vc = (o = {}) => new THREE.MeshLambertMaterial({ vertexColors: true, ...o });

  // ground
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0 + 600000, y1 - y0 + 600000),
    flat(new THREE.MeshLambertMaterial({ color: COLORS.ground }), -10));
  ground.renderOrder = -10;
  ground.rotation.x = -Math.PI / 2;
  ground.position.set((x0 + x1) / 2, 0, -(y0 + y1) / 2);
  ground.receiveShadow = true;
  root.add(ground);

  // land cover: farmland and meadows get a slight per-field tint, so the parcel pattern shows
  const lc = new Map();
  let i = 0;
  for (const p of twin.landcover) {
    const base = COLORS.landcover[p.c] ?? COLORS.ground;
    col.setHex(base);
    if (p.c === 14 || p.c === 15 || p.c === 1) col.offsetHSL(((i * 37) % 11 - 5) * 0.004, 0, ((i * 53) % 9 - 4) * 0.008);
    addPolygon(lc, decode(p.o), p.i.map(decode), p.c >= 14 ? Y.farmland : Y.landcover, col.clone());
    i++;
  }
  root.add(meshes(lc, flat(vc(), -9), { receive: true }));

  // water: lakes, canals and rivers as areas, ditches and drains as ribbons
  const wat = new Map();
  col.setHex(COLORS.water);
  for (const p of twin.water) addPolygon(wat, decode(p.o), p.i.map(decode), Y.water, col);
  for (const w of twin.waterways) addRibbon(wat, decode(w.p), COLORS.waterwayWidth[w.c] ?? 2, Y.waterway, col);
  const waterMat = flat(new THREE.MeshPhongMaterial({ vertexColors: true, shininess: 80, specular: 0x334455 }), -8);
  root.add(meshes(wat, waterMat));

  // roads (Dutch cycle paths in their red asphalt)
  const rd = new Map();
  for (const r of [...twin.roads].sort((a, b) => b.c - a.c)) {
    col.setHex(COLORS.road[r.c] ?? COLORS.road[4]);
    addRibbon(rd, decode(r.p), COLORS.roadWidth[r.c] ?? 4, Y.road + (7 - r.c) * 0.004, col);
  }
  root.add(meshes(rd, flat(vc(), -7), { receive: true }));

  // buildings: walls and flat roofs, coloured by kind; those over the track raised above it
  const bl = new Map(), over = new Map();
  const B = twin.buildings;
  const raised = overTrack(twin);
  const wall = new THREE.Color(), roof = new THREE.Color();
  for (let k = 0; k < B.p.length; k++) {
    const ring = decode(B.p[k]);
    if (ring.length < 3) continue;
    const h = B.h[k];
    const base = raised.has(k) ? Math.min(CLEARANCE, h - 1) : 0;
    const [wc, rc] = COLORS.building[B.k[k]] ?? COLORS.building[0];
    wall.setHex(wc).offsetHSL(0, 0, ((k * 29) % 7 - 3) * 0.012);
    roof.setHex(rc);
    let cx = 0, cy = 0;
    for (const [x, y] of ring) { cx += x; cy += y; }
    const b = tiled(base > 0 ? over : bl, tileKey(cx / ring.length, cy / ring.length));
    // orientation: walls face outwards whichever way the ring runs
    let area = 0;
    for (let j = 0; j < ring.length; j++) {
      const [ax, ay] = ring[j], [bx, by] = ring[(j + 1) % ring.length];
      area += ax * by - bx * ay;
    }
    const ccw = area > 0;
    for (let j = 0; j < ring.length; j++) {
      const [ax, ay] = ring[j], [bx, by] = ring[(j + 1) % ring.length];
      const v = b.n;
      b.vert(ax, base, -ay, wall); b.vert(bx, base, -by, wall); b.vert(bx, h, -by, wall); b.vert(ax, h, -ay, wall);
      if (ccw) b.idx.push(v, v + 1, v + 2, v, v + 2, v + 3);
      else b.idx.push(v, v + 2, v + 1, v, v + 3, v + 2);
    }
    let tris;
    try { tris = THREE.ShapeUtils.triangulateShape(ring.map(([x, y]) => new THREE.Vector2(x, y)), []); }
    catch { tris = []; }
    const v0 = b.n;
    for (const [x, y] of ring) b.vert(x, h, -y, roof);
    for (const [a, bb, c] of tris) b.up(v0 + a, v0 + bb, v0 + c);
    if (base > 0) {                                   // the underside, seen from the trains
      const u0 = b.n;
      for (const [x, y] of ring) b.vert(x, base, -y, wall);
      for (const [a, bb, c] of tris) b.down(u0 + a, u0 + bb, u0 + c);
    }
  }
  const buildings = meshes(bl, vc({ flatShading: true }), { cast: true, receive: true });
  buildings.name = "buildings";
  root.add(buildings);
  // the raised ones are see-through, so the trains standing under Groningen's shed stay visible from above
  root.add(meshes(over, vc({ flatShading: true, transparent: true, opacity: 0.4, depthWrite: false,
    side: THREE.DoubleSide })));

  // track: ballast with sleepers, then the two rails (1435 mm gauge)
  const tex = sleeperTexture();
  const bal = new Map(), rails = new Map();
  const white = new THREE.Color(0xffffff), steel = new THREE.Color(COLORS.rail);
  for (const t of twin.tracks) {
    const pts = decode(t.p);
    addRibbon(bal, pts, t.k ? 2.8 : 3.2, Y.track, white, 1.2);
    for (const off of [-0.7175, 0.7175]) {
      const shifted = offsetLine(pts, off);
      addRibbon(rails, shifted, 0.075, Y.track + 0.17, steel);
    }
  }
  root.add(meshes(bal, flat(new THREE.MeshLambertMaterial({ map: tex, vertexColors: true }), -6), { receive: true }));
  root.add(meshes(rails, flat(new THREE.MeshPhongMaterial({ vertexColors: true, shininess: 90, specular: 0x999999 }), -5)));
  // buffer stops at the drawn yards' siding ends (geo.js yardTracks: [x, y, heading out past the end])
  const stopMat = new THREE.MeshLambertMaterial({ color: COLORS.bufferStop });
  for (const t of twin.tracks) {
    if (!t.end) continue;
    const stop = new THREE.Mesh(new THREE.BoxGeometry(1.0, 1.1, 3.0), stopMat);
    stop.position.set(t.end[0], Y.track + 0.55, -t.end[1]);
    stop.rotation.y = t.end[2];
    stop.castShadow = true;
    root.add(stop);
  }

  // platforms: 76 cm high, concrete, with the yellow safety line
  const pl = new Map(), edge = new Map();
  for (const p of twin.platforms) {
    const ring = decode(p.p);
    if (ring.length < 3) continue;
    extrude(pl, ring, 0.76, new THREE.Color(COLORS.platform));
    addRibbon(edge, offsetRing(ring, -0.6), 0.1, 0.78, new THREE.Color(COLORS.platformEdge));
  }
  root.add(meshes(pl, vc(), { receive: true, cast: true }));
  root.add(meshes(edge, vc()));

  // trees: rows (Frisian alder rows and hedgerows), single trees, woods
  root.add(buildTrees(twin, opts.treeCap ?? 60000));
  root.add(buildLandmarks(twin, opts));
  scene.add(root);
  return root;
}

function offsetLine(pts, d) {
  const out = [];
  for (let i = 0; i < pts.length; i++) {
    const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)];
    const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1;
    out.push([pts[i][0] - dy / L * d, pts[i][1] + dx / L * d]);
  }
  return out;
}

function offsetRing(ring, d) {
  const closed = [...ring, ring[0]];
  return offsetLine(closed, d);
}

function extrude(map, ring, h, c) {
  let cx = 0, cy = 0;
  for (const [x, y] of ring) { cx += x; cy += y; }
  const b = tiled(map, tileKey(cx / ring.length, cy / ring.length));
  let area = 0;
  for (let j = 0; j < ring.length; j++) {
    const [ax, ay] = ring[j], [bx, by] = ring[(j + 1) % ring.length];
    area += ax * by - bx * ay;
  }
  for (let j = 0; j < ring.length; j++) {
    const [ax, ay] = ring[j], [bx, by] = ring[(j + 1) % ring.length];
    const v = b.n;
    b.vert(ax, 0, -ay, c); b.vert(bx, 0, -by, c); b.vert(bx, h, -by, c); b.vert(ax, h, -ay, c);
    if (area > 0) b.idx.push(v, v + 1, v + 2, v, v + 2, v + 3); else b.idx.push(v, v + 2, v + 1, v, v + 3, v + 2);
  }
  let tris;
  try { tris = THREE.ShapeUtils.triangulateShape(ring.map(([x, y]) => new THREE.Vector2(x, y)), []); }
  catch { tris = []; }
  const v0 = b.n;
  for (const [x, y] of ring) b.vert(x, h, -y, c);
  for (const [a, bb, cc] of tris) b.up(v0 + a, v0 + bb, v0 + cc);
}

// deterministic pseudo-random (no Math.random in the scenery: the same twin always looks the same)
function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

function treeGeometry() {
  const trunk = new THREE.CylinderGeometry(0.18, 0.25, 3, 4);
  trunk.translate(0, 1.5, 0);
  const crown = new THREE.IcosahedronGeometry(2.6, 0);
  crown.scale(1, 1.25, 1);
  crown.translate(0, 5.2, 0);
  const paint = (g, c) => {
    const n = g.attributes.position.count, arr = new Float32Array(n * 3);
    col.setHex(c);
    for (let i = 0; i < n; i++) arr.set([col.r, col.g, col.b], i * 3);
    g.setAttribute("color", new THREE.BufferAttribute(arr, 3));
    return g;
  };
  const flatGeo = (g) => (g.index ? g.toNonIndexed() : g);
  const a = paint(flatGeo(trunk), COLORS.trunk), b = paint(flatGeo(crown), COLORS.crown);
  const g = new THREE.BufferGeometry();
  for (const name of ["position", "normal", "color"]) {
    const x = a.attributes[name], y = b.attributes[name];
    const arr = new Float32Array(x.array.length + y.array.length);
    arr.set(x.array); arr.set(y.array, x.array.length);
    g.setAttribute(name, new THREE.BufferAttribute(arr, x.itemSize));
  }
  g.computeVertexNormals();
  return g;
}

function buildTrees(twin, cap) {
  const spots = [];
  const r = rng(7);
  for (const row of twin.trees.rows) {
    const pts = decode(row);
    for (let i = 1; i < pts.length; i++) {
      const [ax, ay] = pts[i - 1], [bx, by] = pts[i];
      const L = Math.hypot(bx - ax, by - ay);
      for (let d = r() * 7; d < L; d += 6.5 + r() * 2) spots.push([ax + (bx - ax) * d / L, ay + (by - ay) * d / L]);
    }
  }
  const P = twin.trees.points;
  for (let i = 0; i < P.length; i += 2) spots.push([P[i] / 10, P[i + 1] / 10]);
  for (const p of twin.landcover) {                   // woods: scattered trees, up to one per 120 m2
    if (p.c !== 7) continue;
    const ring = decode(p.o);
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (const [x, y] of ring) { minx = Math.min(minx, x); maxx = Math.max(maxx, x); miny = Math.min(miny, y); maxy = Math.max(maxy, y); }
    const step = 11;
    for (let x = minx; x < maxx; x += step) {
      for (let y = miny; y < maxy; y += step) {
        const px = x + r() * step, py = y + r() * step;
        if (inside(px, py, ring)) spots.push([px, py]);
        if (spots.length > cap) break;
      }
    }
  }
  // none on the drawn yards' sidings
  const yard = new Set();
  for (const t of twin.tracks) {
    if (!t.yard) continue;
    const pts = decode(t.p);
    for (const [x, y] of pts) for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      yard.add(`${Math.floor(x / 8) + dx},${Math.floor(y / 8) + dy}`);
    }
  }
  const group = new THREE.Group();
  group.name = "trees";
  const byTile = new Map();
  for (const s of spots.filter(([x, y]) => !yard.has(`${Math.floor(x / 8)},${Math.floor(y / 8)}`)).slice(0, cap)) {
    const k = tileKey(s[0], s[1]);
    if (!byTile.has(k)) byTile.set(k, []);
    byTile.get(k).push(s);
  }
  const geo = treeGeometry();
  const mat = new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true });
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3(), pos = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  for (const list of byTile.values()) {
    const im = new THREE.InstancedMesh(geo, mat, list.length);
    list.forEach(([x, y], j) => {
      const s = 0.7 + r() * 0.6;
      q.setFromAxisAngle(up, r() * Math.PI * 2);
      m4.compose(pos.set(x, 0, -y), q, sc.set(s, s * (0.85 + r() * 0.4), s));
      im.setMatrixAt(j, m4);
      const v = 0.82 + r() * 0.3;                    // brightness only: keeps green crowns and brown trunks
      im.setColorAt(j, col.setRGB(v, v * (0.98 + r() * 0.06), v));
    });
    im.computeBoundingSphere();
    im.castShadow = true;
    im.userData.centre = new THREE.Vector3(list[0][0], 0, -list[0][1]);
    group.add(im);
  }
  return group;
}

function inside(x, y, ring) {
  let c = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
  }
  return c;
}

// windmills (the Frisian polder mills), wind turbines, water towers. Static parts are merged into one mesh per
// material; blades and sails are instanced and turn in animateWorld().
function buildLandmarks(twin) {
  const group = new THREE.Group();
  group.name = "landmarks";
  const parts = { white: [], body: [], cap: [], brick: [] };
  const rotors = [];
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
  const put = (kind, geo, x, y, z, yaw = 0) => {
    m4.compose(new THREE.Vector3(x, y, z), q.setFromAxisAngle(up, yaw), new THREE.Vector3(1, 1, 1));
    parts[kind].push((geo.index ? geo.toNonIndexed() : geo).applyMatrix4(m4));
  };
  const YAW = 0.6;                                     // facing the prevailing south-westerly
  for (const m of twin.landmarks) {
    const x = m.x, z = -m.y;
    const dx = Math.cos(YAW), dz = -Math.sin(YAW);     // the rotor side, in the world frame
    if (m.k === "turbine") {
      const h = Math.max(40, Math.min(m.h, 140));
      put("white", new THREE.CylinderGeometry(1.2, 2.2, h, 10), x, h / 2, z);
      put("white", new THREE.BoxGeometry(6, 3, 3), x, h + 1, z, YAW);
      rotors.push({ kind: "blade", n: 3, len: h * 0.45, hub: new THREE.Vector3(x + dx * 3.2, h + 1, z + dz * 3.2), speed: 1.6,
        phase: (x * 0.013) % 6.28 });
    } else if (m.k === "mill") {
      const h = Math.max(8, Math.min(m.h, 30));
      put("body", new THREE.CylinderGeometry(2.2, 4.2, h * 0.7, 8), x, h * 0.35, z);
      put("cap", new THREE.ConeGeometry(2.8, h * 0.18, 8), x, h * 0.79, z);
      rotors.push({ kind: "sail", n: 4, len: h * 0.55, hub: new THREE.Vector3(x + dx * 2.8, h * 0.72, z + dz * 2.8), speed: 0.5,
        phase: (x * 0.017) % 6.28 });
    } else if (m.k === "water_tower") {
      const h = Math.max(15, Math.min(m.h, 50));
      put("brick", new THREE.CylinderGeometry(3, 3.5, h * 0.75, 12), x, h * 0.375, z);
      put("brick", new THREE.CylinderGeometry(7, 5, h * 0.25, 16), x, h * 0.875, z);
    } else {
      const h = Math.max(10, Math.min(m.h, 120));
      put("white", new THREE.CylinderGeometry(1.5, 2.5, h, 8), x, h / 2, z);
    }
  }
  const mats = {
    white: new THREE.MeshLambertMaterial({ color: 0xf2f2f0 }),
    body: new THREE.MeshLambertMaterial({ color: 0x4a4038, flatShading: true }),
    cap: new THREE.MeshLambertMaterial({ color: 0x2f3a30, flatShading: true }),
    brick: new THREE.MeshLambertMaterial({ color: 0x8c5a45, flatShading: true }),
  };
  for (const [k, list] of Object.entries(parts)) {
    if (!list.length) continue;
    const mesh = new THREE.Mesh(mergeGeometries(list), mats[k]);
    mesh.castShadow = true;
    group.add(mesh);
  }
  // one instanced mesh per rotor part; unit length along +y from the hub
  const bladeGeo = new THREE.BoxGeometry(0.012, 1, 0.035);
  bladeGeo.translate(0, 0.5, 0);
  const sailGeo = mergeGeometries([new THREE.PlaneGeometry(0.004, 1).translate(0, 0.5, 0).toNonIndexed(),
    new THREE.PlaneGeometry(0.075, 0.78).translate(0, 0.56, 0.035).toNonIndexed()]);
  sailGeo.rotateY(Math.PI / 2);
  const inst = {};
  for (const [kind, geo, mat] of [["blade", bladeGeo, mats.white],
    ["sail", sailGeo, new THREE.MeshLambertMaterial({ color: 0xe8e0d0, side: THREE.DoubleSide })]]) {
    const count = rotors.filter((r) => r.kind === kind).reduce((a, r) => a + r.n, 0);
    if (!count) continue;
    const im = new THREE.InstancedMesh(geo, mat, count);
    im.frustumCulled = false;
    group.add(im);
    inst[kind] = im;
  }
  group.userData = { rotors, inst, t: 0 };
  setRotors(group, 0);
  return group;
}

const _m = new THREE.Matrix4(), _r = new THREE.Matrix4(), _y = new THREE.Matrix4(), _s = new THREE.Matrix4();

function setRotors(group, t) {
  const { rotors, inst } = group.userData;
  const idx = { blade: 0, sail: 0 };
  _y.makeRotationY(0.6);
  for (const r of rotors) {
    const im = inst[r.kind];
    for (let k = 0; k < r.n; k++) {
      _r.makeRotationX(r.phase + t * r.speed + (k * 2 * Math.PI) / r.n);
      _s.makeScale(r.len, r.len, r.len);
      _m.makeTranslation(r.hub.x, r.hub.y, r.hub.z).multiply(_y).multiply(_r).multiply(_s);
      im.setMatrixAt(idx[r.kind]++, _m);
    }
  }
  for (const im of Object.values(inst)) im.instanceMatrix.needsUpdate = true;
}

export function animateWorld(root, dtReal) {
  const lm = root.getObjectByName("landmarks");
  if (!lm) return;
  lm.userData.t += dtReal;
  setRotors(lm, lm.userData.t);
}
