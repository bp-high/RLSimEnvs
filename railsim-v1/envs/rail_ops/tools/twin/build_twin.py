"""The line's map for the replay viewer (`/twin/`): OpenStreetMap geometry along Leeuwarden - Groningen, the routes
trains follow, and where every network point sits on them.

    python3 -I tools/twin/build_twin.py fetch --cache CACHE     # Overpass -> CACHE/*.json (raw OSM, not committed)
    python3 -I tools/twin/build_twin.py build --cache CACHE     # -> openenv/rail_openenv/web/twin/twin.json.gz

The replay is a picture of a plan. It animates the times `check.simulate` computes and never changes a cost or a
reward.

Frame: metres east (x) and north (y) of 53.22 N, 6.19 E (equirectangular; under 0.2 % scale error along the line).
Geometry is stored as integer decimetres, flattened [x0, y0, x1, y1, ...].

Routes. A train follows a route: its direction's lane through the platform tracks that trains of that direction use in
the record (Rijden de Treinen planned platforms, January-June 2025), from one terminal track to the other
(Leeuwarden 6b or 8, Groningen 5a or 6a). A route is a shortest path over the OSM track graph that never reverses at a
switch, and carries the distance along it of every network point, so a network km maps to a place on the map. Where a
train stands on the other track at a station (a turned train, a second train), the viewer blends to the other
direction's route inside the station's zone.
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
ENV = HERE.parents[2]                                                   # envs/rail_ops
OUT = ENV / "openenv" / "rail_openenv" / "web" / "twin"
NETWORK = HERE.parents[4] / "data" / "gnlw" / "network.json"

LAT0, LON0 = 53.22, 6.19
_R = 6371008.8
KX = _R * math.radians(1) * math.cos(math.radians(LAT0))
KY = _R * math.radians(1)
BBOX = "53.10,5.72,53.33,6.66"

MIRRORS = ("https://overpass.private.coffee/api/interpreter",
           "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
           "https://overpass-api.de/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter")

# The platform each direction uses at each station: the record's planned platforms, January-June 2025 (E = towards
# Groningen). Eastbound / westbound stops on these platforms: LWC 1/1; HDG 1 (5,685) / 2 (5,058); FWD 2 / 1; DWE 2 / 1;
# BP 1 / 2; GK 2 / 1; ZH 1 (5,597) / 2 (5,084).
PLATFORM = {"LWC": {"E": "1", "W": "1"}, "HDG": {"E": "1", "W": "2"}, "FWD": {"E": "2", "W": "1"},
            "DWE": {"E": "2", "W": "1"}, "BP": {"E": "1", "W": "2"}, "GK": {"E": "2", "W": "1"},
            "ZH": {"E": "1", "W": "2"}}
# Terminal tracks: Leeuwarden 6b (11,114 stops) and 8 (10,476); Groningen 5a (10,171) and 6a (8,357).
TERMINAL = {"LW": ("6b", "8"), "GN": ("5a", "6a")}
# OSM maps Groningen as rebuilt (Spoorzone works, 2026): tracks 5 and 6 end at buffer stops on the west side, so a
# train from Leeuwarden cannot reach them. The replay stands 5a trains on OSM track 2 and 6a trains on track 3, the
# island platform next to them that is reachable from the west.
GN_TRACK = {"5a": "2", "6a": "3"}
STATION_REF = {"LW": "Lw", "LWC": "Lwc", "HDG": "Hdg", "FWD": "Fwd", "DWE": "Dwe", "BP": "Bp", "GK": "Gk", "ZH": "Zh",
               "GN": "Gn"}
SERVICE_WEIGHT = {None: 1.0, "crossover": 2.0, "siding": 6.0, "spur": 8.0, "yard": 10.0}
TRAIN_HALF = 28.0                                                       # half a two-car unit, metres


def xy(lat: float, lon: float) -> tuple[float, float]:
    return (lon - LON0) * KX, (lat - LAT0) * KY


def dm(v: float) -> int:
    return int(round(v * 10))


def flat(pts) -> list[int]:
    return [dm(c) for p in pts for c in p]


# --------------------------------------------------------------------------------------------------------- Overpass

def overpass(query: str, path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    for attempt in range(4):
        for url in MIRRORS:
            host = url.split("/")[2]
            try:
                req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": query}).encode(),
                                             headers={"User-Agent": "RLSimEnv-railsim-twin/1.0"})
                with urllib.request.urlopen(req, timeout=420) as r:
                    body = r.read()
                d = json.loads(body)
                if "runtime error" in d.get("remark", ""):
                    raise RuntimeError(d["remark"][:160])
                path.write_bytes(body)
                print(f"  {path.name}: {len(d['elements'])} elements ({host})", flush=True)
                return d
            except Exception as e:  # noqa: BLE001 - any failure: try the next mirror
                print(f"  {path.name}: {host} failed: {str(e)[:120]}", flush=True)
        time.sleep(20 * (attempt + 1))
    raise SystemExit(f"Overpass failed for {path.name}")


RAIL_QUERY = f"""[out:json][timeout:300];
(
  way["railway"~"^(rail|platform)$"]({BBOX});
  way["public_transport"="platform"]["train"="yes"]({BBOX});
  nwr["railway"~"^(station|halt)$"]({BBOX});
  node["railway"~"^(level_crossing|crossing|switch|buffer_stop)$"]({BBOX});
);
out geom;"""


def scenery_queries(line_ll: list[tuple[float, float]]) -> dict[str, str]:
    def a(r: int) -> str:
        return f"around:{r}," + ",".join(f"{lat:.5f},{lon:.5f}" for lat, lon in line_ll)

    head = "[out:json][timeout:400];\n("
    tail = ");\nout geom;"
    return {
        "buildings": head + f'way["building"]({a(700)}); relation["building"]["type"="multipolygon"]({a(700)});' + tail,
        "landmarks": head + f'node["man_made"~"^(windmill|water_tower|tower)$"]({a(4000)}); '
                            f'way["man_made"~"^(windmill|water_tower)$"]({a(4000)}); '
                            f'node["power"="generator"]["generator:source"="wind"]({a(6000)});' + tail,
        "roads": head + f'way["highway"~"^(motorway|trunk|primary|secondary|tertiary)(_link)?$"]({a(2500)}); '
                        f'way["highway"~"^(unclassified|residential|living_street|service|pedestrian|track|'
                        f'cycleway|road)$"]({a(800)});' + tail,
        "water": head + f'way["natural"="water"]({a(3000)}); relation["natural"="water"]({a(3000)}); '
                        f'way["waterway"="riverbank"]({a(3000)}); way["landuse"~"^(basin|reservoir)$"]({a(3000)});'
                 + tail,
        "waterways": head + f'way["waterway"~"^(canal|river|stream|ditch|drain)$"]({a(1200)});' + tail,
        "landcover": head + f'way["landuse"~"^(residential|industrial|commercial|retail|forest|grass|cemetery|'
                            f'allotments|recreation_ground|farmyard|railway|construction|orchard|plant_nursery|'
                            f'greenhouse_horticulture|meadow|farmland|village_green)$"]({a(2500)}); '
                            f'relation["landuse"~"^(residential|industrial|forest|meadow|farmland)$"]({a(2500)}); '
                            f'way["natural"~"^(wood|scrub|heath|wetland|grassland|sand)$"]({a(2500)}); '
                            f'relation["natural"~"^(wood|scrub|heath|wetland)$"]({a(2500)}); '
                            f'way["leisure"~"^(park|pitch|golf_course|sports_centre|garden|nature_reserve)$"]'
                            f'({a(1500)});' + tail,
        "trees": head + f'way["natural"~"^(tree_row|hedge)$"]({a(1500)}); way["barrier"="hedge"]({a(800)}); '
                        f'node["natural"="tree"]({a(600)});' + tail,
        "places": head + f'node["place"~"^(city|town|village|hamlet|suburb|neighbourhood|quarter)$"]({a(5000)});'
                  + tail,
    }


def fetch(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    rail = overpass(RAIL_QUERY, cache / "rail.json")
    net = json.loads(NETWORK.read_text())
    routes = build_routes(rail["elements"], net)
    main = routes["routes"]["E-8-5a"]["xy"]
    line = simplify(main, 60.0)
    line_ll = [(LAT0 + y / KY, LON0 + x / KX) for x, y in line]
    print(f"corridor line: {len(line_ll)} points", flush=True)
    for name, q in scenery_queries(line_ll).items():
        overpass(q, cache / f"{name}.json")


# --------------------------------------------------------------------------------------------------------- geometry

def seg_project(p, a, b) -> tuple[float, float, tuple[float, float]]:
    """(distance, t along a-b in [0, 1], foot point)."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L))
    f = (ax + t * dx, ay + t * dy)
    return math.dist(p, f), t, f


def simplify(pts: list, tol: float) -> list:
    """Douglas-Peucker; closed rings keep their closing point."""
    n = len(pts)
    if n < 3:
        return list(pts)
    if pts[0] == pts[-1]:
        far = max(range(n), key=lambda i: math.dist(pts[0], pts[i]))
        if far in (0, n - 1):
            return list(pts)
        return simplify(pts[:far + 1], tol)[:-1] + simplify(pts[far:], tol)
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        best, bi = 0.0, -1
        for i in range(a + 1, b):
            d = seg_project(pts[i], pts[a], pts[b])[0]
            if d > best:
                best, bi = d, i
        if best > tol:
            keep[bi] = True
            stack += [(a, bi), (bi, b)]
    return [p for p, k in zip(pts, keep) if k]


def ring_area(ring) -> float:
    return 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:])))


def cum_len(pts) -> list[float]:
    s = [0.0]
    for a, b in zip(pts, pts[1:]):
        s.append(s[-1] + math.dist(a, b))
    return s


class Polyline:
    """Projection onto a polyline through a coarse grid index."""

    def __init__(self, pts, cell: float = 200.0):
        self.pts, self.s, self.cell = pts, cum_len(pts), cell
        self.grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            for gx in range(int(min(a[0], b[0]) // cell) - 1, int(max(a[0], b[0]) // cell) + 2):
                for gy in range(int(min(a[1], b[1]) // cell) - 1, int(max(a[1], b[1]) // cell) + 2):
                    self.grid[gx, gy].append(i)

    def project(self, p) -> tuple[float, float]:
        """(distance from the line, distance along it); distance is inf when farther than one grid cell."""
        best = (math.inf, 0.0)
        for i in set(self.grid.get((int(p[0] // self.cell), int(p[1] // self.cell)), [])):
            d, t, _ = seg_project(p, self.pts[i], self.pts[i + 1])
            if d < best[0]:
                best = (d, self.s[i] + t * (self.s[i + 1] - self.s[i]))
        return best

    def at(self, s: float) -> tuple[tuple[float, float], float]:
        """(point, heading in radians) at distance s along the line."""
        import bisect
        s = max(0.0, min(self.s[-1], s))
        i = min(len(self.pts) - 2, max(0, bisect.bisect_right(self.s, s) - 1))
        a, b = self.pts[i], self.pts[i + 1]
        L = self.s[i + 1] - self.s[i]
        t = 0.0 if L == 0 else (s - self.s[i]) / L
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])), math.atan2(b[1] - a[1], b[0] - a[0])


# ------------------------------------------------------------------------------------------------------ track graph

class TrackGraph:
    def __init__(self, ways: list[dict]):
        self.xy: dict[int, tuple[float, float]] = {}
        self.adj: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)   # n -> {m: (length, weight)}
        self.way_nodes: dict[int, list[int]] = {}
        self.ways = {w["id"]: w for w in ways}
        self._virtual = -1
        for w in ways:
            ids = w["nodes"]
            for nid, g in zip(ids, w["geometry"]):
                self.xy[nid] = xy(g["lat"], g["lon"])
            weight = SERVICE_WEIGHT.get(w["tags"].get("service"), 8.0)
            for a, b in zip(ids, ids[1:]):
                L = math.dist(self.xy[a], self.xy[b])
                self.adj[a][b] = self.adj[b][a] = (L, L * weight)
            self.way_nodes[w["id"]] = list(ids)

    def insert(self, way_id: int, p) -> int:
        """A new node on a way at the foot of p; returns its id."""
        ids = self.way_nodes[way_id]
        best = None
        for i, (a, b) in enumerate(zip(ids, ids[1:])):
            d, t, f = seg_project(p, self.xy[a], self.xy[b])
            if best is None or d < best[0]:
                best = (d, i, f)
        _, i, f = best
        a, b = ids[i], ids[i + 1]
        if math.dist(f, self.xy[a]) < 0.5:
            return a
        if math.dist(f, self.xy[b]) < 0.5:
            return b
        v = self._virtual
        self._virtual -= 1
        self.xy[v] = f
        _, w = self.adj[a][b]
        k = w / max(self.adj[a][b][0], 1e-9)
        del self.adj[a][b], self.adj[b][a]
        for m in (a, b):
            L = math.dist(f, self.xy[m])
            self.adj[v][m] = self.adj[m][v] = (L, L * k)
        ids.insert(i + 1, v)
        return v

    def _ok_turn(self, p: int | None, n: int, m: int) -> bool:
        if p is None:
            return True
        if m == p:
            return False
        (px, py), (nx, ny), (mx, my) = self.xy[p], self.xy[n], self.xy[m]
        ax, ay, bx, by = nx - px, ny - py, mx - nx, my - ny
        la, lb = math.hypot(ax, ay), math.hypot(bx, by)
        return la == 0 or lb == 0 or (ax * bx + ay * by) / (la * lb) > 0.7      # under ~45 degrees: no reversing

    def path(self, src: int, dst: int, prev: int | None = None) -> list[int]:
        """Shortest path src -> dst over directed edges with no reversal; `prev` fixes the direction we arrive in."""
        start = (prev, src)
        dist = {start: 0.0}
        back: dict[tuple, tuple] = {}
        tie = 0
        heap = [(0.0, tie, start)]
        while heap:
            d, _, state = heapq.heappop(heap)
            if d > dist.get(state, math.inf):
                continue
            p, n = state
            if n == dst:
                out = [n]
                while state in back:
                    state = back[state]
                    out.append(state[1])
                return out[::-1]
            for m, (_, w) in self.adj[n].items():
                if not self._ok_turn(p, n, m):
                    continue
                ns, nd = (n, m), d + w
                if nd < dist.get(ns, math.inf):
                    dist[ns] = nd
                    back[ns] = state
                    tie += 1
                    heapq.heappush(heap, (nd, tie, ns))
        raise ValueError(f"no path {src} -> {dst}")


def _centroid(e: dict) -> tuple[float, float]:
    if "lat" in e:
        return xy(e["lat"], e["lon"])
    g = e["geometry"]
    return xy(sum(p["lat"] for p in g) / len(g), sum(p["lon"] for p in g) / len(g))


def station_places(els: list[dict]) -> dict[str, dict]:
    """Each station of the line: OSM name and position (by its railway:ref)."""
    by_ref = {}
    for e in els:
        t = e.get("tags", {})
        if t.get("railway") in ("station", "halt") and t.get("railway:ref"):
            by_ref.setdefault(t["railway:ref"], e)
    out = {}
    for code, ref in STATION_REF.items():
        e = by_ref.get(ref)
        if e is None:
            raise SystemExit(f"no OSM station with railway:ref={ref}")
        out[code] = {"name": e["tags"].get("name", code), "xy": _centroid(e)}
    return out


def _way_dist(g: TrackGraph, wid: int, p) -> float:
    ids = g.way_nodes[wid]
    return min(seg_project(p, g.xy[a], g.xy[b])[0] for a, b in zip(ids, ids[1:]))


def platform_tracks(g: TrackGraph, els: list[dict], places: dict) -> tuple[dict, dict]:
    """(station, platform) -> the OSM way of its track: railway:track_ref where mapped (Groningen), else the platform
    polygon's nearest tracks, assigned jointly so two platforms never share one track. Also returns each platform
    polygon's centroid (where trains stop)."""
    plats = [e for e in els if e["type"] == "way" and e.get("tags", {}).get("railway") == "platform"
             and e["tags"].get("ref")]
    out: dict[tuple[str, str], int] = {}
    centre: dict[tuple[str, str], tuple[float, float]] = {}
    rings: dict[tuple[str, str], list] = {}
    for code, place in places.items():
        if code == "GN":
            continue
        refs = set(TERMINAL.get(code, ())) | set(PLATFORM.get(code, {}).values())
        near_ways = [wid for wid, w in g.ways.items() if _way_dist(g, wid, place["xy"]) < 400
                     and w["tags"].get("service") not in ("yard", "siding")]
        tagged = {g.ways[w]["tags"].get("railway:track_ref"): w for w in near_ways
                  if g.ways[w]["tags"].get("railway:track_ref")}
        need = sorted(refs)
        cands = {}
        for ref in need:
            num = ref.rstrip("ab")
            if num in tagged:
                out[code, ref] = tagged[num]
                continue
            ps = [p for p in plats if p["tags"]["ref"] in (ref, num) and math.dist(_centroid(p), place["xy"]) < 500]
            if not ps:
                raise SystemExit(f"no platform {ref} at {code}")
            pc = centre[code, ref] = _centroid(ps[0])
            rings[code, ref] = _ring(ps[0]["geometry"])
            cands[ref] = sorted((_way_dist(g, w, pc), w) for w in near_ways if _way_dist(g, w, pc) < 25)
        left = [r for r in need if (code, r) not in out]
        if len(left) == 1:
            out[code, left[0]] = cands[left[0]][0][1]
        elif len(left) == 2:
            a, b = left
            best = min(((da + db, wa, wb) for da, wa in cands[a][:3] for db, wb in cands[b][:3] if wa != wb))
            out[code, a], out[code, b] = best[1], best[2]
        elif left:
            raise SystemExit(f"{code}: cannot assign {left}")
    return out, centre, rings


def build_routes(els: list[dict], net: dict) -> dict:
    ways = [e for e in els if e["type"] == "way" and e.get("tags", {}).get("railway") == "rail"]
    g = TrackGraph(ways)
    places = station_places(els)
    tracks, centre, rings = platform_tracks(g, els, places)
    # where trains stop: the middle of each station's platform track; at Leeuwarden near the end of the bay; at
    # Groningen in the 'a' half (the west end, where trains from Leeuwarden come in)
    stop_pt: dict[tuple[str, str], tuple[float, float]] = {}
    for (code, ref), wid in tracks.items():
        if code == "LW":                               # a bay: trains come in from the east, stop near its west end
            pl = Polyline([g.xy[n] for n in g.way_nodes[wid]])
            ends = [pl.project(q)[1] for q in rings[code, ref]]
            lo, hi = min(ends), max(ends)
            west = pl.at(lo)[0][0] < pl.at(hi)[0][0]
            stop_pt[code, ref] = pl.at(lo + 15 + TRAIN_HALF if west else hi - 15 - TRAIN_HALF)[0]
        else:
            stop_pt[code, ref] = centre[code, ref]
    plats = [e for e in els if e["type"] == "way" and e.get("tags", {}).get("railway") == "platform"
             and math.dist(_centroid(e), places["GN"]["xy"]) < 400]
    for ref, osm in GN_TRACK.items():         # Groningen: through tracks; trains stand in the west ('a') half
        ring = _ring(next(e for e in plats if e["tags"].get("ref") == osm)["geometry"])
        xs = [q[0] for q in ring]
        target = (min(xs) + 0.25 * (max(xs) - min(xs)), sum(q[1] for q in ring) / len(ring))
        cands = [wid for wid, w in g.ways.items() if w["tags"].get("railway:track_ref") == osm
                 and _way_dist(g, wid, places["GN"]["xy"]) < 400]
        tracks["GN", ref] = min(cands, key=lambda wid: _way_dist(g, wid, target))
        stop_pt["GN", ref] = target
    nodes = {k: g.insert(tracks[k], p) for k, p in stop_pt.items()}
    order = [p["code"] for p in net["points"]]
    stations = [c for c in order if c in PLATFORM]
    routes = {}
    for lw in TERMINAL["LW"]:
        for gn in TERMINAL["GN"]:
            for d in "EW":
                seq = [nodes["LW", lw]] + [nodes[c, PLATFORM[c]["E"]] for c in stations] + [nodes["GN", gn]] \
                    if d == "E" else \
                    [nodes["GN", gn]] + [nodes[c, PLATFORM[c]["W"]] for c in stations[::-1]] + [nodes["LW", lw]]
                path = [seq[0]]
                for a, b in zip(seq, seq[1:]):
                    prev = path[-2] if len(path) > 1 else None
                    path += g.path(a, b, prev)[1:]
                key = f"E-{lw}-{gn}" if d == "E" else f"W-{gn}-{lw}"
                anchors = {}
                s = cum_len([g.xy[n] for n in path])
                pos = {n: i for i, n in enumerate(path)}
                anchors["LW"] = s[pos[nodes["LW", lw]]]
                anchors["GN"] = s[pos[nodes["GN", gn]]]
                for c in stations:
                    anchors[c] = s[pos[nodes[c, PLATFORM[c][d]]]]
                routes[key] = {"nodes": path, "xy": [g.xy[n] for n in path], "s": s, "anchors": anchors, "dir": d}
    # junctions: where the two directions' tracks meet (double track becomes single), on every route
    e, w = routes["E-8-5a"], routes["W-5a-8"]
    shared = set(e["nodes"]) & set(w["nodes"])
    for code, after in (("J1", "GK"), ("J2", "ZH")):
        i0 = e["nodes"].index(nodes[after, PLATFORM[after]["E"]])
        j = next(n for n in e["nodes"][i0:] if n in shared)
        for r in routes.values():
            r["anchors"][code] = r["s"][r["nodes"].index(j)]
    # zones where a train may stand on the other track: between the switches around the station, or 600 m either
    # side where the line is double (a crossover)
    zones = {}
    km = {p["code"]: p["km"] for p in net["points"]}
    for c in stations:
        if c == "LWC":
            continue
        i = e["nodes"].index(nodes[c, PLATFORM[c]["E"]])
        lo = next((k for k in range(i, -1, -1) if e["nodes"][k] in shared), None)
        hi = next((k for k in range(i, len(e["nodes"])) if e["nodes"][k] in shared), None)
        s0 = e["anchors"][c]
        lo_s = e["s"][lo] if lo is not None and s0 - e["s"][lo] < 1500 else None
        hi_s = e["s"][hi] if hi is not None and e["s"][hi] - s0 < 1500 else None
        zones[c] = [round(km[c] - ((s0 - lo_s) if lo_s is not None else 600) / 1000, 3),
                    round(km[c] + ((hi_s - s0) if hi_s is not None else 600) / 1000, 3),
                    lo_s is not None, hi_s is not None]
    return {"graph": g, "places": places, "routes": routes, "zones": zones, "tracks": tracks}


def km_of(route: dict, s: float, net: dict) -> float:
    """Network km at distance s along a route (piecewise linear between the anchors)."""
    km = {p["code"]: p["km"] for p in net["points"]}
    pairs = sorted((route["anchors"][c], km[c]) for c in route["anchors"])
    if s <= pairs[0][0]:
        return pairs[0][1]
    for (s0, k0), (s1, k1) in zip(pairs, pairs[1:]):
        if s <= s1:
            return k0 + (k1 - k0) * (s - s0) / max(s1 - s0, 1e-9)
    return pairs[-1][1]


# ---------------------------------------------------------------------------------------------------------- scenery

def _ring(geom) -> list[tuple[float, float]]:
    return [xy(p["lat"], p["lon"]) for p in geom]


def _assemble(members: list[dict], role: str) -> list[list[tuple[float, float]]]:
    segs = [[(p["lat"], p["lon"]) for p in m["geometry"]] for m in members
            if m.get("type") == "way" and m.get("role", "outer") == role and m.get("geometry")]
    rings = []
    while segs:
        cur = segs.pop()
        while cur[0] != cur[-1]:
            for i, s in enumerate(segs):
                if s[0] == cur[-1]:
                    cur = cur + s[1:]
                elif s[-1] == cur[-1]:
                    cur = cur + s[::-1][1:]
                elif s[-1] == cur[0]:
                    cur = s + cur[1:]
                elif s[0] == cur[0]:
                    cur = s[::-1] + cur[1:]
                else:
                    continue
                segs.pop(i)
                break
            else:
                break
        if cur[0] == cur[-1] and len(cur) >= 4:
            rings.append([xy(a, b) for a, b in cur])
    return rings


def polygons(els: list[dict], tol: float, min_area: float):
    """(outer ring, [inner rings], tags) for closed ways and multipolygon relations."""
    for e in els:
        if e["type"] == "way" and e.get("geometry") and len(e["geometry"]) >= 4 and e["nodes"][0] == e["nodes"][-1]:
            outers, inners = [_ring(e["geometry"])], []
        elif e["type"] == "relation" and e.get("members"):
            outers, inners = _assemble(e["members"], "outer"), _assemble(e["members"], "inner")
        else:
            continue
        for o in outers:
            o = simplify(o, tol)
            if len(o) < 4 or ring_area(o) < min_area:
                continue
            ins = [simplify(i, tol) for i in inners if _inside(i[0], o)]
            yield o, [i for i in ins if len(i) >= 4 and ring_area(i) >= min_area], e.get("tags", {})


def _inside(p, ring) -> bool:
    x, y = p
    c = False
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            c = not c
    return c


def _num(v) -> float | None:
    try:
        return float(str(v).split(";")[0].replace(",", ".").replace("m", "").strip())
    except (TypeError, ValueError):
        return None


BUILDING_HEIGHT = {"house": 7, "detached": 7, "semidetached_house": 7, "terrace": 8, "residential": 7, "bungalow": 4,
                   "apartments": 13, "farm": 8, "barn": 8, "farm_auxiliary": 6, "cowshed": 7, "stable": 6,
                   "industrial": 9, "warehouse": 9, "commercial": 10, "retail": 7, "supermarket": 7, "office": 13,
                   "school": 9, "church": 15, "chapel": 8, "garage": 3, "garages": 3, "shed": 3, "carport": 2.5,
                   "greenhouse": 4, "hut": 3, "roof": 4, "train_station": 8, "hospital": 18, "civic": 10,
                   "public": 10, "hotel": 12, "service": 4}
BUILDING_KIND = {"house": 0, "detached": 0, "semidetached_house": 0, "terrace": 0, "residential": 0, "bungalow": 0,
                 "apartments": 1, "farm": 2, "barn": 2, "farm_auxiliary": 2, "cowshed": 2, "stable": 2,
                 "industrial": 3, "warehouse": 3, "commercial": 4, "retail": 4, "supermarket": 4, "office": 4,
                 "church": 5, "chapel": 5, "greenhouse": 6, "train_station": 7}


def building_height(tags: dict) -> float:
    h = _num(tags.get("height"))
    if h and 2 <= h <= 150:
        return h
    lv = _num(tags.get("building:levels"))
    if lv and 0 < lv < 40:
        return lv * 3.0 + (2.5 if tags.get("roof:shape") not in (None, "flat") else 0.5)
    return float(BUILDING_HEIGHT.get(tags.get("building"), 6))


LANDCOVER = {"residential": 0, "village_green": 1, "grass": 1, "recreation_ground": 1, "park": 1, "garden": 1,
             "pitch": 2, "sports_centre": 2, "golf_course": 1, "industrial": 3, "commercial": 4, "retail": 4,
             "railway": 5, "construction": 5, "farmyard": 6, "forest": 7, "wood": 7, "orchard": 8,
             "plant_nursery": 8, "scrub": 9, "heath": 9, "wetland": 10, "nature_reserve": 11, "cemetery": 12,
             "allotments": 8, "greenhouse_horticulture": 13, "meadow": 14, "grassland": 14, "farmland": 15,
             "sand": 16}
ROAD = {"motorway": 0, "motorway_link": 0, "trunk": 0, "trunk_link": 0, "primary": 1, "primary_link": 1,
        "secondary": 2, "secondary_link": 2, "tertiary": 3, "tertiary_link": 3, "unclassified": 4, "residential": 4,
        "living_street": 4, "road": 4, "pedestrian": 5, "service": 5, "track": 6, "cycleway": 7}
WATERWAY = {"canal": 0, "river": 1, "stream": 2, "drain": 3, "ditch": 3}


def load(cache: Path, name: str) -> list[dict]:
    p = cache / f"{name}.json"
    return json.loads(p.read_text())["elements"] if p.is_file() else []


def build(cache: Path, out: Path) -> dict:
    net = json.loads(NETWORK.read_text())
    rail = load(cache, "rail")
    R = build_routes(rail, net)
    g, routes = R["graph"], R["routes"]
    km = {p["code"]: p["km"] for p in net["points"]}
    report = []
    for key, r in routes.items():
        a = r["anchors"]
        order = [p["code"] for p in (net["points"] if r["dir"] == "E" else net["points"][::-1])]
        ratios = [(abs(a[y] - a[x]) / 1000) / max(abs(km[y] - km[x]), 1e-9) for x, y in zip(order, order[1:])]
        report.append(f"{key}: {r['s'][-1] / 1000:.2f} km, segment ratio {min(ratios):.3f}-{max(ratios):.3f}")
        assert all(a[y] > a[x] for x, y in zip(order, order[1:])), f"{key}: anchors out of order"
    main = Polyline(routes["E-8-5a"]["xy"])
    west = Polyline(routes["W-5a-8"]["xy"])
    corridor = Polyline(simplify(routes["E-8-5a"]["xy"], 20.0), cell=1600.0)

    def near_line(p, d) -> bool:
        return main.project(p)[0] < d or west.project(p)[0] < d

    # tracks to draw: every rail way near the line (the route tracks, loops, yards, other lines' first metres)
    track_out = []
    for w in g.ways.values():
        pts = [xy(p["lat"], p["lon"]) for p in w["geometry"]]
        if not any(corridor.project(p)[0] < 1500 for p in pts[:: max(1, len(pts) // 6)] + [pts[-1]]):
            continue
        svc = w["tags"].get("service")
        track_out.append({"p": flat(simplify(pts, 0.3)), "k": 0 if svc is None else 1})
    # platforms
    plat_out = []
    for e in rail:
        t = e.get("tags", {})
        if e["type"] == "way" and t.get("railway") == "platform" and e["nodes"][0] == e["nodes"][-1]:
            ring = simplify(_ring(e["geometry"]), 0.2)
            if near_line(ring[0], 40):
                plat_out.append({"p": flat(ring), "ref": t.get("ref", "")})
    # level crossings on the line: network km, how many tracks it crosses, the road's angle
    roads = load(cache, "roads")
    road_dir = {}
    for wy in roads:
        ids, geo = wy.get("nodes", []), wy.get("geometry", [])
        if len(ids) < 2 or len(geo) != len(ids):
            continue
        for i, n in enumerate(ids):
            a, b = geo[max(0, i - 1)], geo[min(len(geo) - 1, i + 1)]
            pa, pb = xy(a["lat"], a["lon"]), xy(b["lat"], b["lon"])
            road_dir.setdefault(n, math.atan2(pb[1] - pa[1], pb[0] - pa[0]))
    rail_nodes = {nid for w in g.ways.values() for nid in w["nodes"]}
    lc_out = []
    for e in rail:
        if e["type"] != "node" or e.get("tags", {}).get("railway") != "level_crossing" or e["id"] not in rail_nodes:
            continue
        p = xy(e["lat"], e["lon"])
        de, se = main.project(p)
        dw, _ = west.project(p)
        if min(de, dw) > 6:
            continue
        k = km_of(routes["E-8-5a"], se, net)
        if not (0.3 < k < km["GN"] - 0.3):
            continue
        (fx, fy), head = main.at(se)
        lc_out.append({"km": round(k, 3), "x": round(p[0], 1), "y": round(p[1], 1), "t": round(head, 3),
                       "r": round(road_dir.get(e["id"], head + math.pi / 2), 3),
                       "n": 2 if math.dist(main.at(se)[0], west.at(west.project(p)[1])[0]) > 2.5 else 1})
    lc_out.sort(key=lambda c: c["km"])

    # scenery
    b_rings, b_h, b_k = [], [], []
    for o, _, t in polygons(load(cache, "buildings"), 0.6, 6.0):
        b_rings.append(flat(o[:-1]))
        b_h.append(round(building_height(t), 1))
        b_k.append(BUILDING_KIND.get(t.get("building"), 0))
    lc_polys = []
    for o, ins, t in polygons(load(cache, "landcover"), 2.0, 150.0):
        cls = LANDCOVER.get(t.get("landuse") or t.get("natural") or t.get("leisure"))
        if cls is not None:
            lc_polys.append({"o": flat(o[:-1]), "i": [flat(i[:-1]) for i in ins], "c": cls})
    lc_polys.sort(key=lambda p: p["c"] in (14, 15), reverse=True)         # farmland first (drawn underneath)
    water = [{"o": flat(o[:-1]), "i": [flat(i[:-1]) for i in ins]}
             for o, ins, _ in polygons(load(cache, "water"), 1.5, 40.0)]
    road_out = []
    for wy in roads:
        cls = ROAD.get(wy.get("tags", {}).get("highway"))
        if cls is None or not wy.get("geometry"):
            continue
        road_out.append({"p": flat(simplify(_ring(wy["geometry"]), 0.8)), "c": cls})
    ww_out = []
    for wy in load(cache, "waterways"):
        cls = WATERWAY.get(wy.get("tags", {}).get("waterway"))
        if cls is not None and wy.get("geometry"):
            ww_out.append({"p": flat(simplify(_ring(wy["geometry"]), 1.0)), "c": cls})
    rows, singles = [], []
    for e in load(cache, "trees"):
        if e["type"] == "way" and e.get("geometry"):
            rows.append(flat(simplify(_ring(e["geometry"]), 1.0)))
        elif e["type"] == "node":
            singles += [dm(c) for c in xy(e["lat"], e["lon"])]
    marks = []
    for e in load(cache, "landmarks"):
        t = e.get("tags", {})
        p = _centroid(e) if e["type"] == "way" else xy(e["lat"], e["lon"])
        if t.get("power") == "generator":
            kind, h = "turbine", _num(t.get("height:hub")) or _num(t.get("height")) or 60.0
        elif t.get("man_made") == "windmill":
            kind, h = "mill", _num(t.get("height")) or 16.0
        elif t.get("man_made") == "water_tower":
            kind, h = "water_tower", _num(t.get("height")) or 30.0
        elif t.get("man_made") == "tower" and t.get("tower:type") in ("bell_tower", "communication", None):
            kind, h = "tower", _num(t.get("height")) or 25.0
        else:
            continue
        marks.append({"k": kind, "x": round(p[0], 1), "y": round(p[1], 1), "h": round(h, 1),
                      "n": t.get("name", "")})
    rank = {"city": 0, "town": 1, "village": 2, "suburb": 3, "quarter": 3, "hamlet": 4, "neighbourhood": 4}
    places = []
    for e in load(cache, "places"):
        t = e.get("tags", {})
        if t.get("name") and t.get("place") in rank:
            p = xy(e["lat"], e["lon"])
            places.append({"n": t["name"], "x": round(p[0]), "y": round(p[1]), "r": rank[t["place"]]})

    xs = [p[0] for r in routes.values() for p in r["xy"]]
    ys = [p[1] for r in routes.values() for p in r["xy"]]
    data = {
        "frame": {"lat0": LAT0, "lon0": LON0, "unit": "dm"},
        "bounds": [round(min(xs) - 3000), round(min(ys) - 3000), round(max(xs) + 3000), round(max(ys) + 3000)],
        "routes": {k: {"p": flat(simplify(r["xy"], 0.15)), "dir": r["dir"],
                       "anchors": {c: round(v, 1) for c, v in r["anchors"].items()}} for k, r in routes.items()},
        "zones": R["zones"],
        "stations": [{"code": c, "name": R["places"][c]["name"], "x": round(R["places"][c]["xy"][0], 1),
                      "y": round(R["places"][c]["xy"][1], 1)} for c in R["places"]],
        "crossings": lc_out,
        "tracks": track_out,
        "platforms": plat_out,
        "buildings": {"p": b_rings, "h": b_h, "k": b_k},
        "landcover": lc_polys,
        "water": water,
        "waterways": ww_out,
        "roads": road_out,
        "trees": {"rows": rows, "points": singles},
        "landmarks": marks,
        "places": places,
        "attribution": "© OpenStreetMap contributors (ODbL 1.0); platforms used: Rijden de Treinen (CC BY 4.0)",
    }
    out.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode()
    (out / "twin.json.gz").write_bytes(gzip.compress(body, compresslevel=9, mtime=0))
    stats = {"routes": len(routes), "crossings": len(lc_out), "tracks": len(track_out), "platforms": len(plat_out),
             "buildings": len(b_rings), "landcover": len(lc_polys), "water": len(water), "waterways": len(ww_out),
             "roads": len(road_out), "tree_rows": len(rows), "trees": len(singles) // 2, "landmarks": len(marks),
             "places": len(places), "json_mb": round(len(body) / 1e6, 2),
             "gz_mb": round((out / "twin.json.gz").stat().st_size / 1e6, 2)}
    print("\n".join(report))
    print("zones", R["zones"])
    print(json.dumps(stats))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "build"])
    ap.add_argument("--cache", required=True, help="directory for the raw Overpass responses")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.step == "fetch":
        fetch(Path(args.cache))
    else:
        build(Path(args.cache), Path(args.out))


if __name__ == "__main__":
    main()
