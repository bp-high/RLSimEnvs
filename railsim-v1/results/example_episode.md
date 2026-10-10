# One episode, tool by tool

Claude Sonnet 5.5 (effort high) on the **train** task `gnlw-2025-05-01-0800-180m-blockage-0`: 1 May 2025 from 08:00,
30 runs and 7 train sets. A train broke down between De Westereen and Grijpskerk, closing the double track there
08:10-08:36. It is from the calibration run (`results/rollouts/calib-sonnet-5-5`), so it says nothing about the
eval tasks.

**What the agent sees.** The situation (abridged):

```
LW[terminal] ---single--- LWC[1] ---single--- HDG[2 turn] ---single--- FWD[2] ===double=== DWE[1 each way] ===double===
BP[2 turn] ===double=== GK[1 each way] ===double=== J1 ---single--- ZH[2 turn] ===double=== J2 ---single--- GN[terminal]

WHAT HAS HAPPENED
- CLOSED: every section between DWE and GK, 08:10 to 08:36 (broken down train).

MINIMUM RUNNING TIMES (minutes from each point to the next, for each stopping pattern)
P3 Stoptrein (stopping) LW->GN: LW 3 LWC 4 HDG 3 FWD 3 DWE 5 BP 7 GK 1 J1* 4 ZH 6 J2* 2 GN
...
TRAINS (timetable: arrival/departure at stops, passing minute at points marked *)
37419 stop GN->LW [P4, set T07] RUNNING (running, reaches ZH at 08:02; +0 min on its timetable there): GN 07:54, ...
...
TRAIN SETS (each works its runs in this order)
T07: running 37419; runs 37419 -> 37430 -> ...; must then be at GN for ... at ...
```

**Turn 1.** `check_plan([])`: change nothing, to see what breaks. 8 violations: trains 37426, 37328 and 37419 would
run into the closed sections.

**Turn 2.** Holds only: every train caught by the closure waits it out at the nearest station
(`37419` at GK until 08:36, `37426` at DWE until 08:36, `37328` and `37428` at FWD, `37319` and `37421` at GN).
The result: 40 violations. A platoon of late trains now arrives together at the single track out of Leeuwarden, and
the headways and crossings there break.

**Turn 3.** Sonnet changes strategy: "a cleaner strategy with fewer disruptions". It short-turns two westbound
trains at Hurdegaryp (`37419`, `37319`) and one eastbound at Zuidhorn (`37426`), starts their sets' next runs from
there (`37330`, `37430` from HDG; `37423` from ZH), and retimes the rest. 3 violations remain (two single-track
crossings, at J2-GN and at HDG-FWD).

**Turn 4.** It fixes those with two more holds (`37323` at GN, `37430` at GK). The plan is feasible: cost 675
(435 delay + 8 unserved stops).

**Turn 5.** It checks whether running `37426` through to Groningen would be cheaper than the short-turn, decides
not, sees "not much more savings available", and submits on turn 5 of 12 with 6 of its 10 checks unused.
**Reward 0.429.**

**What the optimum does** (cost 543, proven; floor 432):
- the same three short-turns, but with much better sequencing: 303 minutes of delay instead of 435;
- it holds `37421` at GK, BP and FWD, so it crosses the eastbound platoon at Feanwalden instead of meeting it on the
  single track;
- it releases `37328` from FWD at 08:40 and holds it again at GK;
- it spreads the recovery over the next hour (`37427` at BP, `37334` at ZH, `37429` at ZH and FWD), where Sonnet's
  plan leaves the knock-on delay to compound.

Sonnet's errors were not legality: the plan was clean. It lost on judgement. It stopped early, and it never
explored holds that make one train late so that several others are not. That is the gap the eval measures.
