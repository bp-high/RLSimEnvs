# The line twin: sources

`twin.json.gz` is built by `envs/rail_ops/tools/twin/build_twin.py` (`fetch`, then `build`). Frame: metres east and
north of 53.22 N, 6.19 E (equirectangular), stored as integer decimetres.

| What | Source and licence |
|---|---|
| track (every rail way near the line: main line, passing loops, yards), platforms, level crossings, station positions | © OpenStreetMap contributors, [ODbL 1.0](https://www.openstreetmap.org/copyright) (Overpass API, data of July-October 2026) |
| 42,654 buildings (within 700 m of the line), land cover, water, roads and cycle paths, tree rows and trees, wind turbines, windmills, towers, place names | © OpenStreetMap contributors, ODbL 1.0 (same queries) |
| which platform each direction uses at every station, and the usual terminal tracks | Rijden de Treinen train archive, January-June 2025 (planned platforms; CC BY 4.0, "Rijden de Treinen") |
| train times | `rail_core.replay`: the grader's own simulation of the plan (`check.simulate`) |

## Real, derived and drawn

**Real:**
- Track geometry, switches and station layout, as mapped in OSM. Trains follow it per car, so they take the curves
  and the loop tracks.
- Each direction's platform at each station, from the record: Hurdegaryp 1 eastbound / 2 westbound, Feanwâlden 2 / 1,
  De Westereen 2 / 1, Buitenpost 1 / 2, Grijpskerk 2 / 1, Zuidhorn 1 / 2.
- Leeuwarden bays 6b and 8, the tracks Leeuwarden - Groningen trains used there.
- The times: every arrival, departure and passing minute is the simulation's. Between them the picture moves
  smoothly: zero speed at stops, passing speed kept through points the train does not stop at.

**Derived:**
- Groningen. OSM maps the station as rebuilt (Spoorzone works, 2026): tracks 5 and 6, which the record's 5a and 6a
  trains used, end at buffer stops on the west side, so no train from Leeuwarden can reach them. The twin stands those
  trains on OSM tracks 2 and 3, the island platform beside them that is reachable from the west.
- Terminal track per train: the record's usual split by train type (Leeuwarden: westbound fast trains to 8, stopping
  trains to 6b; Groningen: fast 6a, stopping 5a), or the other track, or behind a set already there, wherever no set
  drives through another: a set that comes in while another stands on its track stands 65 m behind it and leaves
  first, and a track holds only as many sets as fit before its switch (Leeuwarden 4, Groningen 3). From the east the
  line reaches only these two tracks at Leeuwarden (OSM, checked October 2026), while the rules give a terminal room
  for every set, so a plan can bring a third set in as another leaves with the other track taken (3.6% of terminal
  stays in the test replays). That set waits in a yard the replay draws (below).
- Which of the two tracks a train uses at a passing station: its direction's platform, or the other one when that is
  taken or the train is turning back there.
- Level crossing barriers close about 35 seconds before a train reaches the crossing and open once it has passed.
- Signals: one at each entry of every single-track section, red while a train is inside or the section is closed.
  They show the rule the grader enforces; they are not the line's real signals.

**Drawn (illustrative):**
- The yards: three stub sidings outside each terminal, two sets each, where a set waits when neither platform track
  has room (Leeuwarden: two sidings about 0.4 km out and one about 0.8 km out; Groningen: three about 0.8 km out).
  They are not on the map. `geo.js` lays them beside the line on the nearest stretch past the platform tracks that is
  clear of buildings, water, roads, platforms and other track, and draws them with buffer stops. Leeuwarden's real
  yard lies beyond the platforms, reached from this line only through 6b, so it cannot take a set while the platforms
  are full. A set the yard cannot take either is drawn faded while it would share a track (1 stay in 5,410 in the
  test replays).
- Building heights: OSM `height` or `building:levels` where tagged, else by building type (a house 7 m, a barn 8 m,
  a warehouse 9 m).
- Buildings the track runs through (Groningen's train shed and two buildings that bridge the line) are drawn from a
  5.5 m clearance up and see-through, so trains pass under them and stay visible.
- The cab view of a standing train is from the end it will leave from (the driver changes ends).
- Heights of landmarks without a tag: windmills 16 m, wind turbines a 60 m hub, towers 25 m.
- Trees: the mapped trees and tree rows, plus woods filled at about one tree per 120 m² (capped at 60,000).
- Train models: every train is drawn as a two-car unit (about 55 m), white, with a blue stripe for stopping trains and
  orange for fast ones. The record does not say which unit ran which train.
- The sky and the sun: the solar position for the task's date and minute at the line (local time, CET/CEST).

The replay never changes a cost or a reward. The viewer loads three.js 0.180 from the jsDelivr CDN.
