# RailSimEnv v1: design notes

The details behind the [README](README.md): the line, the physics and the rules, what the agent decides, the cost
and the reward, how the reference is proven, how tasks are built and how the eval set is chosen for hardness, and
how the grader is kept honest. Sources and licences are in [`data/gnlw/SOURCE.md`](data/gnlw/SOURCE.md).

The recipe: take one real job, rebuild it from the records it leaves behind, and grade the agent against an answer
that can be proven best. The problem is hard in kind: trains on single track block each other in both directions, and
every decision moves trains you have not looked at yet.

| | RailSimEnv |
|---|---|
| the world | 11 points on 55 km of line; single-track sections, passing stations with 1-2 tracks |
| the decisions | per train: holds, short-turn, cancel |
| the coupling | trains coupled three ways: opposing trains on single track, following trains (headway), and each train set's chain of runs |
| the disruptions | the record's breakdowns, collisions, signal and points failures, plus disruptions drawn from it |
| the proof | CP-SAT disjunctive model with ordering ties, per-train-set floors |
| the picture | a 3D map of the line (OSM track, stations, level crossings) replaying any plan, a train graph, an eval explorer |

## The line

Arriva runs the Groningen - Leeuwarden line (Staatslijn B, 55 km) with stopping trains (Stoptrein, every station)
and fast trains (Sneltrein, Feanwalden plus Buitenpost or Zuidhorn): two of each per hour in each direction, so
eight trains an hour share the single track between Leeuwarden and Feanwalden.

```
LW ---single--- LWC[1] ---single--- HDG[2 turn] ---single--- FWD[2] ===double=== DWE[1 each way] ===double===
BP[2 turn] ===double=== GK[1 each way] ===double=== J1 ---single--- ZH[2 turn] ===double=== J2 ---single--- GN
```

- [n] tracks at a station; [1 each way] a station on double track (no overtaking); turn: trains can be short-turned
  there (the three stations where they really were in 2025).
- J1 and J2 are junctions where double track becomes single; trains cannot stop there.
- The layout comes from OSM's per-track ways and ProRail section references. The 2025 timetable confirms it: 163
  of 181 days run conflict-free as scheduled.

## A task

One real day, cut at a "now" minute (`start`), with every run that leaves before `end` (a 150-240 minute window) and
every run already moving. The record gives each moving train's place (from its last actual departure), each train
set's chain of runs (rolling-stock links), and the timetable. A task is refused when the record cannot be replayed
honestly up to `start`: for example, a train was already cut short before it, or the plain timetable conflicts in
the window.

## The physics (deterministic)

- Every train set works its runs in order. A run starts where its set is, no earlier than when the set is ready
  (`turn_min` = 4 minutes after its last arrival), its scheduled departure, or the plan's hold.
- A train runs at its minimum running times (the 5th percentile of the record), but never reaches a point before
  its timetable minute there. A late train makes up time; an on-time train runs to the timetable.
- It never leaves a scheduled stop early. Where it has no stop it passes, unless held past its passing minute: then
  it stops, losing a minute braking and a minute accelerating. Junctions cannot hold a train.
- A closed section stops nothing by itself. The rules forbid being in it, so the plan must hold, turn or cancel.
  A slow section adds minutes to every train entering it; a track out of use removes one platform track.

## Rules

A plan is infeasible if any is broken:

| rule | |
|---|---|
| single track | two trains going opposite ways are never in a single-track section at once; the second enters at the earliest in the minute the first leaves |
| headway | trains going the same way are 3 minutes apart at both ends of every section (no overtaking between points) |
| platforms | never more trains at a station than tracks (one per direction on double track; the terminals have room for every set); a set waiting between runs keeps a track |
| closed | no train in a closed section while it is closed (a train already stuck inside waits there) |
| continuity | a run starts where its set is; short-turns only at turn points; a moving train cannot be cancelled; each set ends the window where its next run leaves |
| max delay | no stop reached more than 120 minutes late |

## What the agent decides

A plan lists only the trains it changes (an empty plan changes nothing):

```json
[{"train": "37413", "to": "HDG"}, {"train": "37424", "from": "HDG", "hold": {"HDG": "07:31"}},
 {"train": "37326", "hold": {"FWD": "07:55"}}, {"train": "37419", "cancel": true}]
```

- `hold` makes a train wait at a station (to let an opposing train through, or for a section to reopen).
- `to` and `from` short-turn a train set: it ends one run early and starts its next from there.
- `cancel` drops a run; its set stays put, so its next run usually goes too.

These are the levers a real treindienstleider pulls. The record shows the line's dispatchers turning trains at
Hurdegaryp, Zuidhorn and Buitenpost and cancelling them outright.

## Cost and reward

Cost, in stop-minutes:
- every minute of arrival delay at every scheduled stop served;
- 30 for every scheduled stop not served;
- for each train set, its lateness for its first run after the window, times that run's stops.

The reward:

| submitted plan | reward |
|---|---|
| unreadable, malformed or infeasible | 0.2 x clean runs / (runs + entry problems), below 0.2 |
| feasible | 0.2 + 0.8 x exp(-g / 0.5), g = (cost - optimum) / (optimum - floor + 100) |

- The **floor** is each train set's best cost alone with the closures: what the disruption forces however well
  trains are sequenced. It is computed exactly by the same solver without the interaction rules.
- `gap_k` = 100 stop-minutes (about three unserved stops). An earlier draft used 30 (one unserved stop); a pilot
  showed it turns a plan 20 % over the optimum into the 0.2 floor, a cliff with no gradient.

## The reference, proven

`solve.py` builds one CP-SAT model of exactly the physics above:
- one option per run: in full, short-turned or started at each turn point, or cancelled;
- arrival and departure minutes per point, with the recurrences `simulate` uses (including max(run, timetable));
- a Boolean per passing point for an unscheduled stop;
- each set's position and readiness carried run to run;
- single-track and headway rules as pairwise disjunctions, pruned where the time windows cannot meet;
- platforms as cumulatives, closures as disjunctions;
- max delay as a domain bound, which also bounds every time window.

Two exact strengthenings make larger windows provable:
1. **Ordering ties.** Where two trains cannot pass each other (a junction, the one-track halt, a station with one
   track each way), their order is the same on both sides, so the two ordering Booleans are tied.
2. **Per-set floors.** Each set's cost is at least its cost alone, and the objective at least their sum, so the
   bound starts at the floor and only the interactions are left to prove.

Large windows get warm-started extensions: twice, from the best plan, with twice the time. A task is kept only if
its reference is within 1 % of the proven bound on the avoidable scale (cost − floor + gap_k). The solver's plan is
always re-checked by the pure-Python grader, and its cost must equal the solver's objective; one such mismatch found
a simulator bug during development.

## Tasks and tiers

| tier | what goes wrong |
|---|---|
| `real` | the record's own disruption on its own day. The window starts at the announcement, or up to 45 min earlier if trains were already being cut short |
| `blockage` | a closure drawn from the 2024-2025 record (cause, duration, place), maybe a late train set |
| `loop` | a points failure (a passing track out of use), maybe slow running and a late set |
| `compound` | a closure plus maybe a lost track and slow running, and 1-3 late sets, over 3-4 hours |

Eval tasks come only from six held-out ISO weeks (4, 7, 11, 17, 22, 25; 42 days); train tasks from the other 139
days. A candidate is kept if it has at least 15 runs, its reference is proven (above), the naive plan scores ≤ 0.6
and the greedy dispatchers ≤ 0.85.

## Choosing hard eval tasks

The user asked for an eval where Sonnet 5.5 lands below 0.7, so the eval is chosen for hardness. It must not be
chosen by peeking at Sonnet's eval results, so selection never sees them:

1. **Features** (`hardness.py`, model-free):
   - size: runs, sets;
   - conflicts if nothing is changed;
   - how much the optimum changes: holds, short-turns, cancellations, distinct turn points;
   - avoidable cost;
   - the greedy dispatcher's gap;
   - minutes of each disruption kind;
   - whether a closure is on single track or mid-line (turns needed on both sides);
   - late sets.
2. **Calibration** on the train split: Sonnet 5.5 plays 16 train tasks spread over tiers and sizes.
3. **Fit** (`select.py`): a ridge regression from standardized features to Sonnet's calibration reward, with
   leave-one-out error; features whose sign is unstable are dropped.
4. **Select**: predict every pool task (built on held-out days only), keep the hardest 45 by prediction, with at most
   3 per day and every tier represented.

The run that made `rail-v1-eval` (`tasks/rail-v1-eval/selection.json`):
- **Calibration:** Sonnet 5.5 averaged 0.557 on the 16 train tasks.
- **Fit:** 10 of the 15 features kept stable signs. Leave-one-out mean absolute error was 0.134 and leave-one-out
  Spearman 0.67. The signs make sense:
  - more distinct turn points, decisions, holds, short-turns and slow running predict a lower reward;
  - a lost passing track (`loop`) predicts a higher one.
- **Pool:** 72 tasks (from 134 candidates on 29 held-out days, all proven optimal).
- **Selection:** 45 chosen, predicted mean 0.443 against 0.563 for the pool. By tier: blockage 20, compound 17,
  loop 4, real 4.

**After the eval run** (a check only; nothing was re-selected):
- Sonnet 5.5 scored 0.467 on the 45, close to the predicted level.
- Per-task ranking did not validate: Spearman 0.08 between predicted and actual reward within the 45. The ranking
  is fitted on 16 points and tested on a range-restricted set (all low predictions).
- What the selection reliably did is shift the mix: fewer `loop` tasks (Sonnet 0.74 on them in calibration), more
  closures.
- The headline (under 0.7) holds mostly because the environment is hard: 0.557 on 16 train tasks spread over tiers and sizes.
- A better predictor needs more calibration episodes, and features that capture what Sonnet actually got wrong:
  it cancels whole trains where the optimum short-turns them, and leaves knock-on delay unrecovered.

The pilot that motivated this (`pilot` runs, 3 real tasks) is the evidence that hardness is real decision-making
and not formatting:
- Sonnet solved a simple end-of-line closure perfectly.
- It missed a non-obvious short-turn at Zuidhorn on a 2-hour closure (20 % over the optimum).
- It never short-turned in a slow-running scenario where the optimum does.

## The map replay

`/twin/` replays any plan on a 3D model of the line. It is a picture of a plan, never part of the grade:
- **Times:** `rail_core.replay` runs the grader's own `simulate` and `evaluate`. It returns every run's arrival,
  departure and passing minute, the plan's orders, the rule breaks, and the cost as it accrues. The accrual is
  event by event: a stop's delay when the train arrives, 30 for a stop not served when it was due, lateness handed
  on at the window's end. These events add up to the grader's cost (asserted).
- **Places:** `tools/twin/build_twin.py` builds the line from OpenStreetMap. For each direction it finds a shortest
  path over the track graph that never reverses at a switch, through the platform each direction uses in the 2025
  record. It records where every network point sits on that path, so a network km maps to a place on the track.
  Inside a station's zone a train standing on the other track blends onto the other path: between the switches of
  a loop, or over a crossover where the line is double.
- **Motion:** between simulated minutes a train moves on a monotone cubic: zero speed at stops, its passing speed
  kept through points where it does not stop. Every car follows the track by its two bogies.
- **Derived for the picture only:**
  - terminal tracks, by the record's usual split;
  - which track a train takes at a passing station;
  - level crossing barriers;
  - signals at each single-track entry, red while the section holds a train or is closed. They show the rule the
    grader enforces, not the line's real signalling.

  `web/twin/SOURCES.md` lists what is real, derived and drawn, including one substitution at Groningen, where OSM
  maps the station after its 2026 rebuild.

## How it is kept honest

- **Independent reference checker** (`core/tests/test_grader_robust.py::ref_evaluate`). It re-implements the physics
  and every rule on a minute grid without importing `check.py`. It agrees with the grader on feasibility, cost and
  which rules break, over hundreds of perturbed plans (holds, short-turns, cancellations, removals).
- **Hostile plans** never crash grading and never pass: garbage JSON, NaN and inf times, unknown trains and stations,
  duplicates, 50x-repeated entries.
- **The optimum holds**:
  - no perturbed plan beats the proven bound;
  - the floor is below every feasible plan;
  - the stored optimum scores 1.0 through the server (the oracle over a real WebSocket MCP session).
- **The timetable validates the layout**: clean days run conflict-free as scheduled.
- **Lazy policies stay low**: the naive plan is infeasible, and the greedy dispatchers score ≤ 0.85 (usually about 0.2).
- **The replay is the grade**: every submitted plan in the committed runs replays to the reward the eval recorded
  (61 plans).
- **The picture keeps the simulated times**: `tests/viewer/*.test.mjs` (run by `test_viewer.py` under node) checks
  the viewer's motion against 112 replays: 17 tasks under the optimum and two baselines, plus the 61 model plans.
  - Every train is at its point at every simulated minute.
  - None runs backwards or impossibly fast.
  - No two trains share a station track in a legal plan.
  - The cost events add up to the plan's cost.

## Simulated, not real

- The rolling-stock circulation (first-in-first-out at terminals) is derived.
- The disruption effects by cause (closed / slow / track out) are assumptions.
- Station weights are equal: no passenger counts are public.
- Terminals have room for everyone.
- The world is clairvoyant: the closure's end is known when the task starts.

v2 (live simulation) would:
- reveal closure ends and delays over time, using the record's announcement times;
- execute the standing plan as the clock runs;
- add the rest of the Groningen network (Roodeschool, Delfzijl, Veendam, Nieuweschans) sharing Groningen and Sauwerd.
