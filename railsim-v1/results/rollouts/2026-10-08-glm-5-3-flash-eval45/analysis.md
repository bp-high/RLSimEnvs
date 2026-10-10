# GLM-5.3-Flash on rail-v1-eval (45 tasks, one rollout each)

`zai-org/GLM-5.3-Flash` through Hugging Face's router, served by Together.
The harness is `rail_openenv.agent_openai`: the same episode rules as the Claude runs (12 turns, 64,000 output tokens
per turn, two turns in a row without a tool call end the episode, no submit scores 0). The model's own reasoning
setting was used (no `reasoning_effort`), and its reasoning went back with each assistant message.

## In one paragraph

GLM-5.3-Flash scores **0.360** (95% CI 0.251-0.477), against 0.467 for Claude Sonnet 5.5 on the same tasks. The gap
is almost entirely episodes that never submit: 16 of 45 ended after two turns in a row cut at the 64k output limit
with no tool call. GLM tries to simulate the timetable in its head instead of letting `check_plan` do it; 96 of its
275 turns hit the limit. When it does submit, it is as good as Sonnet: 0.559 on those 29 tasks against Sonnet's
0.509, 24 of 29 plans legal, 6 at the proven optimum, and legal plans 6 % over the optimum's cost (median; Sonnet 17 %).

## Numbers

| | GLM-5.3-Flash | Claude Sonnet 5.5 |
|---|---:|---:|
| mean reward (95% CI) | **0.360** (0.251-0.477) | **0.467** (0.384-0.553) |
| median | 0.200 | 0.325 |
| below 0.7 | 32 / 45 | 33 / 45 |
| ended without a submit | 16 | 0 |
| submitted plans that break a rule | 5 / 29 | 0 / 45 |
| at the proven optimum | 6 | 4 |
| legal plans: cost over the optimum (median, mean) | 6 %, 19 % | 17 %, 42 % |
| on the 29 tasks GLM submitted: mean | 0.559 | 0.509 |
| head to head over 45 tasks (better / worse / tie) | 12 / 29 / 4 | |
| real, blockage, loop, compound | 0.420, 0.278, 0.250, 0.468 | 0.457, 0.421, 0.538, 0.507 |
| median turns, median seconds per episode | 5, 1,611 s | 6, 264 s |

## How the run went, and what was rerun

- **Provider.** Same model, very different speed: Together about 100 output tokens/s, Novita about 40, DeepInfra
  about 22. Fireworks refused the DeepPPT organisation (402, pay-as-you-go not enabled). All 45 episodes ran on
  Together, all 45 in parallel.
- **Connection drops (6 episodes).** The sandbox's outbound proxy drained its tunnels three times during the run
  (`server_draining`). Every stream open at that moment died, and the client's retries cover only a request's start.
  Those 6 episodes were rerun from scratch; the harness now reruns just the dropped turn.
- **Wall clock (2 episodes).** The runner's 3,600 s safety limit never bound Sonnet (its longest episode took 930 s).
  It cut two GLM episodes after 9-10 of their 12 turns. They were rerun with 7,200 s, so the 12-turn budget binds,
  as it did for Sonnet. They scored 0.000 and 0.194 on the rerun.
- **A container restart** killed the first reruns part-way; they were started again. Every rerun is a fresh attempt
  chosen by what happened to it (an infrastructure failure), never by its score. The 16 episodes that ended without a
  submit are the model's own behaviour under the same rules as Sonnet, and were not rerun.

## From eval/analyze.py

- episodes 45, mean reward 0.360, median 0.200
- predicted mean 0.443; Spearman(predicted, actual) -0.10
- reward < 0.7 on 32/45; at the optimum on 6
- feasible plans cost 6% more than the optimum (median), 19% (mean)
- where the extra cost went (sum over feasible plans, plan minus optimum): delay +1731, unserved +900, handed_on +145; cancellations +7, short-turns -24
- real tier: cheaper than the real dispatchers' outcome on 1/2 feasible plans

| task | tier | runs | predicted | reward | cost | optimum | floor |
|---|---|---:|---:|---:|---:|---:|---:|
| gnlw-2025-01-21-0626-180m-real-0 | real | 29 | 0.53 | 0.000 | None | 445 | 356 |
| gnlw-2025-01-22-1600-240m-compound-0 | compound | 38 | 0.54 | 0.000 | None | 670 | 627 |
| gnlw-2025-01-22-1800-150m-loop-0 | loop | 25 | 0.58 | 0.000 | None | 397 | 336 |
| gnlw-2025-02-13-1530-180m-blockage-0 | blockage | 30 | 0.43 | 0.000 | None | 648 | 588 |
| gnlw-2025-02-14-1730-210m-blockage-0 | blockage | 33 | 0.29 | 0.000 | None | 875 | 776 |
| gnlw-2025-02-15-0700-240m-compound-0 | compound | 32 | 0.52 | 0.000 | None | 858 | 785 |
| gnlw-2025-02-15-1630-150m-loop-0 | loop | 26 | 0.38 | 0.000 | None | 275 | 234 |
| gnlw-2025-03-10-0700-210m-blockage-0 | blockage | 34 | 0.52 | 0.000 | None | 455 | 386 |
| gnlw-2025-03-11-0730-150m-blockage-0 | blockage | 26 | 0.60 | 0.000 | None | 2043 | 1851 |
| gnlw-2025-03-12-1530-180m-blockage-0 | blockage | 30 | 0.47 | 0.000 | None | 806 | 730 |
| gnlw-2025-03-12-1730-210m-loop-0 | loop | 33 | 0.44 | 0.000 | None | 166 | 158 |
| gnlw-2025-03-16-1700-210m-compound-0 | compound | 16 | 0.25 | 0.000 | None | 529 | 529 |
| gnlw-2025-04-23-1630-150m-blockage-0 | blockage | 26 | 0.46 | 0.000 | None | 1926 | 1782 |
| gnlw-2025-04-26-1700-180m-blockage-0 | blockage | 30 | 0.42 | 0.000 | None | 447 | 433 |
| gnlw-2025-05-27-1830-180m-blockage-0 | blockage | 27 | 0.59 | 0.000 | None | 596 | 560 |
| gnlw-2025-05-30-0630-180m-blockage-0 | blockage | 29 | 0.40 | 0.000 | None | 454 | 385 |
| gnlw-2025-05-27-1700-180m-blockage-0 | blockage | 30 | 0.53 | 0.060 | None | 791 | 670 |
| gnlw-2025-01-20-1800-180m-compound-0 | compound | 24 | 0.26 | 0.067 | None | 664 | 539 |
| gnlw-2025-01-23-0800-210m-compound-0 | compound | 34 | 0.47 | 0.100 | None | 894 | 841 |
| gnlw-2025-05-31-0800-210m-compound-0 | compound | 32 | 0.60 | 0.188 | None | 1179 | 1017 |
| gnlw-2025-02-12-0900-180m-real-0 | real | 31 | 0.19 | 0.194 | None | 643 | 529 |
| gnlw-2025-02-16-1530-210m-compound-0 | compound | 18 | 0.53 | 0.200 | 679 | 273 | 269 |
| gnlw-2025-04-26-0630-180m-blockage-0 | blockage | 20 | 0.23 | 0.200 | 1692 | 1047 | 980 |
| gnlw-2025-05-27-0830-210m-blockage-0 | blockage | 34 | 0.50 | 0.205 | 1121 | 795 | 768 |
| gnlw-2025-04-25-1730-210m-blockage-0 | blockage | 33 | 0.57 | 0.213 | 1607 | 1370 | 1355 |
| gnlw-2025-01-23-0900-180m-compound-0 | compound | 30 | 0.42 | 0.237 | 1236 | 853 | 704 |
| gnlw-2025-01-25-0730-180m-compound-0 | compound | 26 | 0.49 | 0.260 | 1541 | 1302 | 1218 |
| gnlw-2025-04-27-0830-210m-blockage-0 | blockage | 18 | 0.25 | 0.315 | 1380 | 1283 | 1283 |
| gnlw-2025-01-25-1700-180m-blockage-0 | blockage | 30 | 0.36 | 0.375 | 430 | 320 | 275 |
| gnlw-2025-02-13-0954-180m-real-0 | real | 30 | 0.40 | 0.484 | 233 | 172 | 154 |
| gnlw-2025-02-15-0630-180m-blockage-0 | blockage | 20 | 0.30 | 0.543 | 2419 | 2300 | 2119 |
| gnlw-2025-01-26-1700-210m-compound-0 | compound | 16 | 0.27 | 0.559 | 670 | 618 | 588 |
| gnlw-2025-05-26-1630-150m-blockage-0 | blockage | 26 | 0.57 | 0.722 | 290 | 254 | 185 |
| gnlw-2025-05-28-1830-210m-compound-0 | compound | 31 | 0.58 | 0.740 | 466 | 436 | 383 |
| gnlw-2025-02-16-1730-240m-compound-0 | compound | 19 | 0.45 | 0.777 | 673 | 656 | 652 |
| gnlw-2025-04-27-0800-210m-compound-0 | compound | 15 | 0.32 | 0.847 | 393 | 382 | 378 |
| gnlw-2025-05-26-1800-150m-blockage-0 | blockage | 25 | 0.50 | 0.957 | 1018 | 1014 | 968 |
| gnlw-2025-05-26-0730-150m-blockage-0 | blockage | 26 | 0.46 | 0.971 | 859 | 857 | 848 |
| gnlw-2025-05-31-0900-180m-compound-0 | compound | 30 | 0.36 | 0.986 | 227 | 226 | 216 |
| gnlw-2025-01-25-0630-180m-loop-0 | loop | 20 | 0.46 | 1.000 | 116 | 116 | 102 |
| gnlw-2025-02-11-0900-150m-blockage-0 | blockage | 23 | 0.41 | 1.000 | 725 | 725 | 721 |
| gnlw-2025-02-16-1700-210m-compound-0 | compound | 16 | 0.53 | 1.000 | 796 | 796 | 787 |
| gnlw-2025-03-13-0706-180m-real-0 | real | 26 | 0.50 | 1.000 | 378 | 378 | 378 |
| gnlw-2025-03-16-1630-180m-compound-0 | compound | 15 | 0.51 | 1.000 | 1259 | 1259 | 1256 |
| gnlw-2025-06-22-1600-240m-compound-0 | compound | 18 | 0.51 | 1.000 | 668 | 668 | 668 |

The predicted rewards are Sonnet's hardness predictor, fitted on Sonnet's calibration episodes; they are not a
prediction for GLM.

## Every task, against Sonnet 5.5

| task | tier | GLM-5.3-Flash | how it ended | Sonnet 5.5 |
|---|---|---:|---|---:|
| gnlw-2025-01-21-0626-180m-real-0 | real | 0.000 | no submit (64k cuts) | 0.374 |
| gnlw-2025-01-22-1600-240m-compound-0 | compound | 0.000 | no submit (64k cuts) | 0.719 |
| gnlw-2025-01-22-1800-150m-loop-0 | loop | 0.000 | no submit (64k cuts) | 0.744 |
| gnlw-2025-02-13-1530-180m-blockage-0 | blockage | 0.000 | no submit (64k cuts) | 0.258 |
| gnlw-2025-02-14-1730-210m-blockage-0 | blockage | 0.000 | no submit (64k cuts) | 0.200 |
| gnlw-2025-02-15-0700-240m-compound-0 | compound | 0.000 | no submit (64k cuts) | 0.225 |
| gnlw-2025-02-15-1630-150m-loop-0 | loop | 0.000 | no submit (64k cuts) | 0.325 |
| gnlw-2025-03-10-0700-210m-blockage-0 | blockage | 0.000 | no submit (64k cuts) | 0.200 |
| gnlw-2025-03-11-0730-150m-blockage-0 | blockage | 0.000 | no submit (64k cuts) | 0.427 |
| gnlw-2025-03-12-1530-180m-blockage-0 | blockage | 0.000 | no submit (64k cuts) | 0.221 |
| gnlw-2025-03-12-1730-210m-loop-0 | loop | 0.000 | no submit (64k cuts) | 0.212 |
| gnlw-2025-03-16-1700-210m-compound-0 | compound | 0.000 | no submit (64k cuts) | 0.953 |
| gnlw-2025-04-23-1630-150m-blockage-0 | blockage | 0.000 | no submit (64k cuts) | 0.265 |
| gnlw-2025-04-26-1700-180m-blockage-0 | blockage | 0.000 | no submit (64k cuts) | 0.225 |
| gnlw-2025-05-27-1830-180m-blockage-0 | blockage | 0.000 | no submit (64k cuts) | 0.692 |
| gnlw-2025-05-30-0630-180m-blockage-0 | blockage | 0.000 | no submit (64k cuts) | 0.214 |
| gnlw-2025-05-27-1700-180m-blockage-0 | blockage | 0.060 | submitted, illegal | 0.312 |
| gnlw-2025-01-20-1800-180m-compound-0 | compound | 0.067 | submitted, illegal | 0.200 |
| gnlw-2025-01-23-0800-210m-compound-0 | compound | 0.100 | submitted, illegal | 0.395 |
| gnlw-2025-05-31-0800-210m-compound-0 | compound | 0.188 | submitted, illegal | 0.221 |
| gnlw-2025-02-12-0900-180m-real-0 | real | 0.194 | submitted, illegal | 0.200 |
| gnlw-2025-02-16-1530-210m-compound-0 | compound | 0.200 | submitted, legal | 0.216 |
| gnlw-2025-04-26-0630-180m-blockage-0 | blockage | 0.200 | submitted, legal | 0.269 |
| gnlw-2025-05-27-0830-210m-blockage-0 | blockage | 0.205 | submitted, legal | 0.832 |
| gnlw-2025-04-25-1730-210m-blockage-0 | blockage | 0.213 | submitted, legal | 0.200 |
| gnlw-2025-01-23-0900-180m-compound-0 | compound | 0.237 | submitted, legal | 0.411 |
| gnlw-2025-01-25-0730-180m-compound-0 | compound | 0.260 | submitted, legal | 0.333 |
| gnlw-2025-04-27-0830-210m-blockage-0 | blockage | 0.315 | submitted, legal | 0.315 |
| gnlw-2025-01-25-1700-180m-blockage-0 | blockage | 0.375 | submitted, legal | 0.674 |
| gnlw-2025-02-13-0954-180m-real-0 | real | 0.484 | submitted, legal | 0.251 |
| gnlw-2025-02-15-0630-180m-blockage-0 | blockage | 0.543 | submitted, legal | 0.263 |
| gnlw-2025-01-26-1700-210m-compound-0 | compound | 0.559 | submitted, legal | 0.352 |
| gnlw-2025-05-26-1630-150m-blockage-0 | blockage | 0.722 | submitted, legal | 0.253 |
| gnlw-2025-05-28-1830-210m-compound-0 | compound | 0.740 | submitted, legal | 0.280 |
| gnlw-2025-02-16-1730-240m-compound-0 | compound | 0.777 | submitted, legal | 0.205 |
| gnlw-2025-04-27-0800-210m-compound-0 | compound | 0.847 | submitted, legal | 0.777 |
| gnlw-2025-05-26-1800-150m-blockage-0 | blockage | 0.957 | submitted, legal | 0.682 |
| gnlw-2025-05-26-0730-150m-blockage-0 | blockage | 0.971 | submitted, legal | 0.985 |
| gnlw-2025-05-31-0900-180m-compound-0 | compound | 0.986 | submitted, legal | 1.000 |
| gnlw-2025-01-25-0630-180m-loop-0 | loop | 1.000 | optimal | 0.871 |
| gnlw-2025-02-11-0900-150m-blockage-0 | blockage | 1.000 | optimal | 0.941 |
| gnlw-2025-02-16-1700-210m-compound-0 | compound | 1.000 | optimal | 1.000 |
| gnlw-2025-03-13-0706-180m-real-0 | real | 1.000 | optimal | 1.000 |
| gnlw-2025-03-16-1630-180m-compound-0 | compound | 1.000 | optimal | 0.331 |
| gnlw-2025-06-22-1600-240m-compound-0 | compound | 1.000 | optimal | 1.000 |
