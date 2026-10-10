# RLSimEnv

Simulation RL environments rebuilt from real operational records and graded against a proven optimum.

| environment | the job | data | status |
|---|---|---|---|
| [**RailSimEnv v1**](railsim-v1/) | dispatch Arriva's trains on the single-track Groningen - Leeuwarden line after a real 2025 breakdown, collision, signal or points failure: hold, short-turn and cancel trains so opposing trains cross safely | Rijden de Treinen train archive and disruptions (CC BY 4.0), OpenStreetMap (2025) | 45 eval tasks chosen for hardness (best model 0.556) + 159 train tasks, OpenEnv server, proven CP-SAT references, 3D replay, eval explorer, [Space](https://huggingface.co/spaces/bpHigh/RailSimEnv) |

The recipe:
1. Take one real job and rebuild it from the records it leaves behind.
2. Stack the disruptions a practitioner meets.
3. Prove the best answer with an exact solver.
4. Grade the agent deterministically against it. The grader is pure Python, and the agent acts only through MCP tools.



## Licence

The code and the documentation are MIT. The data keep their sources' licences, which the MIT licence does not replace:

| data | files | licence |
|---|---|---|
| trains, timetables, delays, cancellations, platforms, disruptions ([Rijden de Treinen](https://www.rijdendetreinen.nl/en/open-data) open data), and what is derived from them | `railsim-v1/data/gnlw/` (`services_2025h1.csv.gz`, `disruptions_2025h1.csv`, `disruption_profiles.csv`, `runtimes.json`, `links_2025h1.csv`); the task packs in `railsim-v1/envs/rail_ops/tasks/` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), credit "Rijden de Treinen" |
| the line's layout and the 3D replay's map (track, platforms, buildings, land cover, water, roads, trees) | `railsim-v1/data/gnlw/network.json` (also copied into every task); `railsim-v1/envs/rail_ops/openenv/rail_openenv/web/twin/twin.json.gz` | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/), © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) |

Anyone reusing or adapting the OpenStreetMap-derived files must keep them under ODbL and credit OpenStreetMap; the 3D replay and the train graph show both credits on screen.
