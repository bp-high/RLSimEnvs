---
title: RailSimEnv v1
emoji: 🚆
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
license: mit
pinned: false
short_description: Dispatch trains on a real single-track Dutch line
tags: [openenv, simulation, rl-environment, railway, scheduling, operations-research]
---

# RailSimEnv v1 · OpenEnv

Dispatch Arriva's trains on the single-track Groningen – Leeuwarden line (Netherlands) after a real 2025 disruption:
a broken-down train, a collision at a level crossing, a signal or points failure, on its real day and time, with every
train where it really was. Hold, short-turn and cancel trains so opposing trains cross safely. One graded submit per
episode against the proven CP-SAT optimum. Serves the rail-v1 packs: 45 eval tasks chosen for hardness and 159 train
tasks on other days.

| page | what |
|---|---|
| `/web` | OpenEnv's web UI. **Explore**: the eval explorer, the 3D replay and the train graph. **Playground**: reset, step and the MCP tools by hand |
| `/explorer/` | the leaderboard and every model's plan on every eval task; pick one to watch it |
| `/twin/` | the 3D replay: each model's plan on every eval task, on the line's real track from OpenStreetMap (plus a few drawn sidings outside each terminal, where a set waits when both platform tracks are taken), with signals, level crossings, closures, the train graph and the cost as it accrues |
| `/viewer/` | the train graph (time across, distance down) of any plan, with the plan's holds, short-turns and rule breaks |
| `/reset`, `/step`, `/state`, `/ws`, `/mcp` | OpenEnv |
| `/rail_ops/...` | OpenEnv Task API (`splits`, `num_tasks`, `task`, `task_range`) |
| `/api/...` | read-only JSON for the pages |

## Tools

| tool | returns |
|---|---|
| `get_situation()` | the line, the closures, every train's timetable and position, the train sets and their chains of runs |
| `check_plan(plan)` | simulates a draft: every train's times, delay and lost stops, every rule broken, the cost; 10 per episode |
| `submit_plan(plan)` | ends the episode with one grade |

`plan` lists only the trains you change: `[{"train": "37413", "to": "HDG"}, {"train": "37324", "from": "HDG"},
{"train": "37419", "hold": {"GK": "08:36"}}, {"train": "37330", "cancel": true}]`. Trains left alone run as early as
the physics allow, never ahead of the timetable. `reset(task_id=...)`, `reset(split=..., index=...)` or
`reset(seed=...)`.

**Rules:** no two opposing trains in a single-track section at once; 3-minute headways for trains going the same way;
never more trains at a station than tracks (the terminals have room for every set); no train in a closed section;
runs start where their train set is, and every set ends the window where its next run leaves; no stop more than 2
hours late.

**Cost**, in stop-minutes: every minute of delay at every stop served, 30 per stop not served, and lateness handed on
to the next window.

## Reward

Graded once, on `submit_plan`, by `RailPlanRubric` (every other action scores 0.0). A plan that breaks any rule scores
0.2 × the share of runs it leaves clean. A legal plan scores 0.2 + 0.8 × exp(−gap / 0.5), with
gap = (cost − optimum) / (optimum − floor + 100), so the proven CP-SAT optimum scores 1.0. No submission scores 0. The
children `runs_clean` and `cost_quality` are reported in the final observation's `metadata.rubric`.

## Results on the 45 eval tasks

One rollout per model, the same rules for every model (12 turns, 64k output tokens per turn):

| model | mean reward (95% CI) | legal plans | at the optimum |
|---|---|---:|---:|
| Qwen3.8-2.4T-A95B | **0.556** (0.452-0.665) | 39 | 12 |
| DeepSeek-V4.1-Flash | **0.534** (0.427-0.643) | 33 | 12 |
| Claude Sonnet 5.5 | **0.467** (0.384-0.553) | 45 | 4 |
| GLM-5.3-Flash | **0.360** (0.251-0.477) | 24 | 6 |
| the real dispatchers (real tier, scored as if legal) | 0.502 | | |
| greedy dispatcher | 0.180 | | |
| change nothing | 0.105 | 0 | 0 |

Every plan is in the eval explorer, and replays in 3D exactly as it was graded.

## Connect an agent

```python
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.mcp_client import MCPToolClient

env = MCPToolClient("https://bphigh-railsimenv.hf.space").sync()
obs = env.reset(split="eval", index=0)
rules = obs.observation.metadata["instructions"]          # the system prompt
situation = env.step(CallToolAction(tool_name="get_situation", arguments={}))
plan = [{"train": "37413", "to": "HDG"}, {"train": "37324", "from": "HDG"}]
print(env.step(CallToolAction(tool_name="check_plan", arguments={"plan": plan})).observation)
```

## Licences

The code is MIT. The data keep their sources' licences:

| data | source | licence |
|---|---|---|
| trains, timetables, delays, cancellations, platforms; disruptions | [Rijden de Treinen](https://www.rijdendetreinen.nl/en/open-data) open data | CC BY 4.0 ("Rijden de Treinen") |
| track layout; the 3D replay's map (track, platforms, buildings, land cover, water, roads) | © OpenStreetMap contributors | ODbL 1.0 |

The OpenStreetMap-derived data in this Space (the 3D map, `openenv/rail_openenv/web/twin/twin.json.gz`, and the line
layout inside every task in `tasks/`) is licensed under the [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/),
not MIT, and stays so when reused or adapted. The 3D replay and the train graph credit OpenStreetMap and Rijden de
Treinen on screen.

The rolling-stock circulation, the disruption effects by cause and the cost weights are modelling choices.
