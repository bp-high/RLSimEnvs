# Groningen - Leeuwarden: data sources

Everything RailSimEnv v1 knows about the line, where it comes from, and what is real, derived or assumed.

## Sources and licences

| file | what | source | licence |
|---|---|---|---|
| `services_2025h1.csv.gz` | every stop of every Arriva train that runs only between Leeuwarden and Groningen, 1 January - 30 June 2025: schedule, arrival and departure delay, cancellations, planned and actual platform (144,787 stops, 22,700 services, 181 days) | [Rijden de Treinen train archive](https://www.rijdendetreinen.nl/en/open-data/train-archive) (`services-2025-MM.csv.gz`), built from NS real-time data | CC BY 4.0, "Rijden de Treinen" |
| `disruptions_2025h1.csv` | the 21 disruptions Rijden de Treinen linked to the line in the same months: cause, start, end, stations | [Rijden de Treinen disruptions](https://www.rijdendetreinen.nl/en/open-data/disruptions) | CC BY 4.0 |
| `disruption_profiles.csv` | all 123 disruptions on the line in 2024 and 2025 (cause, duration, stations): the distribution seeded disruptions are drawn from | same | CC BY 4.0 |
| `network.json` | points, kilometres, single or double track per section, tracks per station, turn points | OpenStreetMap track geometry and `ref:ProRailSpoortak` tags (each Dutch track is its own way; a `V` suffix is single track), station coordinates from the Rijden de Treinen station list, platforms from the record | © OpenStreetMap contributors, ODbL 1.0 |
| `runtimes.json` | minimum stop-to-stop running time per service type | derived from `services_2025h1.csv.gz` (`tools/calibrate.py`) | CC BY 4.0 (derived) |
| `links_2025h1.csv` | which run each arriving train forms next (rolling-stock circulation) | derived from the schedule (`tools/curate_gnlw.py`) | CC BY 4.0 (derived) |

Re-curate with `tools/curate_gnlw.py` from the monthly files at `https://opendata.rijdendetreinen.nl/public/services/`
(about 31 MB each) and `.../public/disruptions/disruptions-2024.csv`, `disruptions-2025.csv`; then
`tools/calibrate.py`. From July 2025 the line runs on to Groningen Europapark and is partly closed for works, so the
half-year is one timetable.

## Real, derived, assumed

**Real (from the record):**
- every train, its stops, scheduled times, real delays and cancellations;
- each train's real position when a task starts (from its last actual departure);
- the real disruptions on their real days (tier `real`): cause, start and duration;
- the real dispatchers' outcome, scored with the task's cost (`reference.real`; it need not obey the model);
- planned platforms, which give each station's tracks (two platforms used by both directions at the passing
  stations; one at Leeuwarden Camminghaburen);
- turn points: the stations where trains were really short-turned in January-June 2025 (Hurdegaryp 99 times,
  Zuidhorn 41, Buitenpost 17).

**Derived:**
- **Track layout.** Single track Leeuwarden - Feanwalden (passing loop at Hurdegaryp), double Feanwalden - 1.75 km
  east of Grijpskerk (junction J1), single J1 - Zuidhorn, double Zuidhorn - 8.3 km on (junction J2), single J2 -
  Groningen. The parallel track Zuidhorn - J2 has no ProRail reference in OSM yet (it is new); the 2025 timetable
  needs it (as single track 66 scheduled crossings would conflict), so it is double.
- **Validation.** 163 of the 181 days run conflict-free as scheduled under the layout and the headways below; the
  others are disrupted days whose record holds services changed on the day, or weekend Zuidhorn shuttles.
- **Minimum running times:** the 5th percentile of actual stop-to-stop running times (never above schedule).
  Passing points and junctions are interpolated by distance.
- **Headways:** opposing trains may enter a single-track section in the minute the other leaves it (the timetable
  plans crossings that way at minute resolution); same-direction trains 3 minutes apart at both ends (the
  timetable keeps at least 4).
- **Rolling-stock links:** first-in-first-out at each terminal with at least 4 minutes to turn. The circulation is
  not public; delay propagation in the record cannot separate links from single-track knock-on.

**Assumed (named in the task text where they matter):**
- trains never run ahead of the timetable, and run at minimum times when late;
- an unscheduled stop costs 1 minute braking and 1 accelerating (stopping trains lose 1-2 minutes per stop against
  fast trains in the record);
- a train set needs 4 minutes to turn, at terminals and at turn points alike; terminals have room for every train;
- **disruption effects by cause:** breakdowns, collisions, emergency calls, damaged track, power failures close
  the sections between the outermost stations named; signal and level-crossing failures make trains run on sight
  (+4 minutes per section); points failures take one track out at the passing station in the span. A disruption
  named for (nearly) every station of the line gets a location drawn with the task's seed;
- the train that broke down (or collided) is out of service for the window; its later runs are not in the task;
- **cost:** every minute of arrival delay at every scheduled stop served counts 1; every scheduled stop not served
  counts 30 (the next train of the same kind is 30 minutes later); lateness handed to a train set's first run after
  the window counts per stop of that run. No passenger counts are public for these stations, so stops count equally;
- no train may reach a stop more than 120 minutes late (dispatchers cancel instead).
