# Qwen3.8-2.4T-A95B on rail-v1-eval (45 tasks, one rollout each)

`Qwen/Qwen3.8-2.4T-A95B` through Hugging Face's router, served by Together.
Same harness and rules as the GLM-5.3-Flash run (`rail_openenv.agent_openai`): 12 turns, 64,000 output tokens per
turn, the model's own reasoning setting, its reasoning sent back with each assistant message, a 7,200 s wall clock.

## In one paragraph

Qwen3.8-2.4T scores **0.556** (95% CI 0.452-0.665), the best of the three models run so far (Sonnet 5.5 0.467,
GLM-5.3-Flash 0.360). It submitted on all 45 tasks: it tests drafts with `check_plan` turn after turn instead of
simulating by hand (20 of 419 turns reached the 64k cap, against GLM's 96 of 275). 39 plans are legal and 12 reach
the proven optimum; legal plans cost 5 % more than the optimum (median). Its losses are 6 illegal plans and a long
tail of legal but costly ones. It is also the slowest: a median of 11 turns and 21 minutes per episode.

## Numbers

| | Qwen3.8-2.4T | Claude Sonnet 5.5 | GLM-5.3-Flash |
|---|---:|---:|---:|
| mean reward (95% CI) | **0.556** (0.452-0.665) | **0.467** (0.384-0.553) | **0.360** (0.251-0.477) |
| median | 0.534 | 0.325 | 0.200 |
| below 0.7 | 27 / 45 | 33 / 45 | 32 / 45 |
| ended without a submit | 0 | 0 | 16 |
| submitted plans that break a rule | 6 | 0 | 5 |
| at the proven optimum | 12 | 4 | 6 |
| legal plans: cost over the optimum (median, mean) | 5 %, 49 % | 17 %, 42 % | 6 %, 19 % |
| head to head with Qwen (Qwen better / worse / tie) | | 22 / 20 / 3 | 32 / 8 / 5 |
| real, blockage, loop, compound | 0.546, 0.481, 0.787, 0.593 | 0.457, 0.421, 0.538, 0.507 | 0.420, 0.278, 0.250, 0.468 |
| beat the real dispatchers (real tier) | 3 of 4 | 2 of 4 | 1 of 4 |
| median turns, median seconds per episode | 11, 1,273 s | 6, 264 s | 5, 1,611 s |

Qwen beats Sonnet on the mean, but head to head the two are close (22 better, 20 worse, 3 tied): the gap comes from
Qwen's 12 optimal plans and its strength on `loop` tasks (0.787), not from winning most tasks.

## How the run went

- **Provider.** Two-task pilots on DeepInfra and Together (stopped before any episode finished) showed Together
  streaming 114-141 tokens/s and DeepInfra 43-51. All episodes ran on Together.
- **47 streams were too many.** The first start ran all 45 episodes at once (plus a 2-episode rerun). Together
  answered with dynamic rate limits (429) and the router's circuit breaker paused the provider (503); the client's
  short retries gave up and 9 episodes died. The run was stopped and restarted 20 wide, keeping the 2 episodes that
  had finished (both 1.0); the harness now waits out a 429 or 503 for up to about 4.5 minutes. In the restarted run
  22 turns were recovered that way and no episode was lost.
- **Context.** Reasoning goes back each turn, so prompts grow: the longest reached 436k tokens, inside Together's
  1M-token window (DeepInfra's is 262k). No episode hit the window or the wall clock.

## From eval/analyze.py

- episodes 45, mean reward 0.556, median 0.534
- predicted mean 0.443; Spearman(predicted, actual) -0.08
- reward < 0.7 on 27/45; at the optimum on 12
- feasible plans cost 5% more than the optimum (median), 49% (mean)
- where the extra cost went (sum over feasible plans, plan minus optimum): delay +7413, unserved +4650, handed_on +1405; cancellations +24, short-turns -20
- real tier: cheaper than the real dispatchers' outcome on 3/4 feasible plans

| task | tier | runs | predicted | reward | cost | optimum | floor |
|---|---|---:|---:|---:|---:|---:|---:|
| gnlw-2025-03-12-1530-180m-blockage-0 | blockage | 30 | 0.47 | 0.047 | None | 806 | 730 |
| gnlw-2025-05-30-0630-180m-blockage-0 | blockage | 29 | 0.40 | 0.055 | None | 454 | 385 |
| gnlw-2025-02-14-1730-210m-blockage-0 | blockage | 33 | 0.29 | 0.085 | None | 875 | 776 |
| gnlw-2025-02-15-1630-150m-loop-0 | loop | 26 | 0.38 | 0.146 | None | 275 | 234 |
| gnlw-2025-05-31-0800-210m-compound-0 | compound | 32 | 0.60 | 0.175 | None | 1179 | 1017 |
| gnlw-2025-03-11-0730-150m-blockage-0 | blockage | 26 | 0.60 | 0.185 | None | 2043 | 1851 |
| gnlw-2025-02-12-0900-180m-real-0 | real | 31 | 0.19 | 0.200 | 4091 | 643 | 529 |
| gnlw-2025-05-27-1700-180m-blockage-0 | blockage | 30 | 0.53 | 0.200 | 5393 | 791 | 670 |
| gnlw-2025-01-23-0900-180m-compound-0 | compound | 30 | 0.42 | 0.200 | 2325 | 853 | 704 |
| gnlw-2025-05-27-1830-180m-blockage-0 | blockage | 27 | 0.59 | 0.200 | 1186 | 596 | 560 |
| gnlw-2025-02-16-1730-240m-compound-0 | compound | 19 | 0.45 | 0.200 | 1044 | 656 | 652 |
| gnlw-2025-05-27-0830-210m-blockage-0 | blockage | 34 | 0.50 | 0.205 | 1118 | 795 | 768 |
| gnlw-2025-02-15-0700-240m-compound-0 | compound | 32 | 0.52 | 0.205 | 1295 | 858 | 785 |
| gnlw-2025-01-22-1600-240m-compound-0 | compound | 38 | 0.54 | 0.225 | 918 | 670 | 627 |
| gnlw-2025-01-20-1800-180m-compound-0 | compound | 24 | 0.26 | 0.233 | 1024 | 664 | 539 |
| gnlw-2025-01-23-0800-210m-compound-0 | compound | 34 | 0.47 | 0.239 | 1126 | 894 | 841 |
| gnlw-2025-01-25-1700-180m-blockage-0 | blockage | 30 | 0.36 | 0.251 | 519 | 320 | 275 |
| gnlw-2025-05-28-1830-210m-compound-0 | compound | 31 | 0.58 | 0.256 | 639 | 436 | 383 |
| gnlw-2025-01-21-0626-180m-real-0 | real | 29 | 0.53 | 0.311 | 632 | 445 | 356 |
| gnlw-2025-02-13-1530-180m-blockage-0 | blockage | 30 | 0.43 | 0.348 | 783 | 648 | 588 |
| gnlw-2025-02-15-0630-180m-blockage-0 | blockage | 20 | 0.30 | 0.385 | 2506 | 2300 | 2119 |
| gnlw-2025-02-11-0900-150m-blockage-0 | blockage | 23 | 0.41 | 0.473 | 781 | 725 | 721 |
| gnlw-2025-04-26-0630-180m-blockage-0 | blockage | 20 | 0.23 | 0.534 | 1120 | 1047 | 980 |
| gnlw-2025-03-10-0700-210m-blockage-0 | blockage | 34 | 0.52 | 0.562 | 522 | 455 | 386 |
| gnlw-2025-04-23-1630-150m-blockage-0 | blockage | 26 | 0.46 | 0.589 | 2014 | 1926 | 1782 |
| gnlw-2025-01-25-0730-180m-compound-0 | compound | 26 | 0.49 | 0.631 | 1359 | 1302 | 1218 |
| gnlw-2025-02-13-0954-180m-real-0 | real | 30 | 0.40 | 0.673 | 203 | 172 | 154 |
| gnlw-2025-02-16-1700-210m-compound-0 | compound | 16 | 0.53 | 0.796 | 812 | 796 | 787 |
| gnlw-2025-04-25-1730-210m-blockage-0 | blockage | 33 | 0.57 | 0.806 | 1386 | 1370 | 1355 |
| gnlw-2025-04-26-1700-180m-blockage-0 | blockage | 30 | 0.42 | 0.837 | 460 | 447 | 433 |
| gnlw-2025-05-26-1630-150m-blockage-0 | blockage | 26 | 0.57 | 0.902 | 265 | 254 | 185 |
| gnlw-2025-01-26-1700-210m-compound-0 | compound | 16 | 0.27 | 0.929 | 624 | 618 | 588 |
| gnlw-2025-05-26-1800-150m-blockage-0 | blockage | 25 | 0.50 | 0.957 | 1018 | 1014 | 968 |
| gnlw-2025-01-22-1800-150m-loop-0 | loop | 25 | 0.58 | 1.000 | 397 | 397 | 336 |
| gnlw-2025-01-25-0630-180m-loop-0 | loop | 20 | 0.46 | 1.000 | 116 | 116 | 102 |
| gnlw-2025-02-16-1530-210m-compound-0 | compound | 18 | 0.53 | 1.000 | 273 | 273 | 269 |
| gnlw-2025-03-12-1730-210m-loop-0 | loop | 33 | 0.44 | 1.000 | 166 | 166 | 158 |
| gnlw-2025-03-13-0706-180m-real-0 | real | 26 | 0.50 | 1.000 | 378 | 378 | 378 |
| gnlw-2025-03-16-1630-180m-compound-0 | compound | 15 | 0.51 | 1.000 | 1259 | 1259 | 1256 |
| gnlw-2025-03-16-1700-210m-compound-0 | compound | 16 | 0.25 | 1.000 | 529 | 529 | 529 |
| gnlw-2025-04-27-0800-210m-compound-0 | compound | 15 | 0.32 | 1.000 | 382 | 382 | 378 |
| gnlw-2025-04-27-0830-210m-blockage-0 | blockage | 18 | 0.25 | 1.000 | 1283 | 1283 | 1283 |
| gnlw-2025-05-26-0730-150m-blockage-0 | blockage | 26 | 0.46 | 1.000 | 857 | 857 | 848 |
| gnlw-2025-05-31-0900-180m-compound-0 | compound | 30 | 0.36 | 1.000 | 226 | 226 | 216 |
| gnlw-2025-06-22-1600-240m-compound-0 | compound | 18 | 0.51 | 1.000 | 668 | 668 | 668 |

The predicted rewards are Sonnet's hardness predictor, fitted on Sonnet's calibration episodes; they are not a
prediction for Qwen.

## Every task, against Sonnet 5.5 and GLM-5.3-Flash

| task | tier | Qwen3.8-2.4T | how it ended | turns | Sonnet 5.5 | GLM-5.3-Flash |
|---|---|---:|---|---:|---:|---:|
| gnlw-2025-03-12-1530-180m-blockage-0 | blockage | 0.047 | illegal plan | 12 | 0.221 | 0.000 |
| gnlw-2025-05-30-0630-180m-blockage-0 | blockage | 0.055 | illegal plan | 12 | 0.214 | 0.000 |
| gnlw-2025-02-14-1730-210m-blockage-0 | blockage | 0.085 | illegal plan | 12 | 0.200 | 0.000 |
| gnlw-2025-02-15-1630-150m-loop-0 | loop | 0.146 | illegal plan | 12 | 0.325 | 0.000 |
| gnlw-2025-05-31-0800-210m-compound-0 | compound | 0.175 | illegal plan | 12 | 0.221 | 0.188 |
| gnlw-2025-03-11-0730-150m-blockage-0 | blockage | 0.185 | illegal plan | 11 | 0.427 | 0.000 |
| gnlw-2025-02-12-0900-180m-real-0 | real | 0.200 | legal | 12 | 0.200 | 0.194 |
| gnlw-2025-05-27-1700-180m-blockage-0 | blockage | 0.200 | legal | 11 | 0.312 | 0.060 |
| gnlw-2025-01-23-0900-180m-compound-0 | compound | 0.200 | legal | 11 | 0.411 | 0.237 |
| gnlw-2025-05-27-1830-180m-blockage-0 | blockage | 0.200 | legal | 9 | 0.692 | 0.000 |
| gnlw-2025-02-16-1730-240m-compound-0 | compound | 0.200 | legal | 12 | 0.205 | 0.777 |
| gnlw-2025-05-27-0830-210m-blockage-0 | blockage | 0.205 | legal | 9 | 0.832 | 0.205 |
| gnlw-2025-02-15-0700-240m-compound-0 | compound | 0.205 | legal | 10 | 0.225 | 0.000 |
| gnlw-2025-01-22-1600-240m-compound-0 | compound | 0.225 | legal | 11 | 0.719 | 0.000 |
| gnlw-2025-01-20-1800-180m-compound-0 | compound | 0.233 | legal | 11 | 0.200 | 0.067 |
| gnlw-2025-01-23-0800-210m-compound-0 | compound | 0.239 | legal | 11 | 0.395 | 0.100 |
| gnlw-2025-01-25-1700-180m-blockage-0 | blockage | 0.251 | legal | 11 | 0.674 | 0.375 |
| gnlw-2025-05-28-1830-210m-compound-0 | compound | 0.256 | legal | 12 | 0.280 | 0.740 |
| gnlw-2025-01-21-0626-180m-real-0 | real | 0.311 | legal | 12 | 0.374 | 0.000 |
| gnlw-2025-02-13-1530-180m-blockage-0 | blockage | 0.348 | legal | 11 | 0.258 | 0.000 |
| gnlw-2025-02-15-0630-180m-blockage-0 | blockage | 0.385 | legal | 9 | 0.263 | 0.543 |
| gnlw-2025-02-11-0900-150m-blockage-0 | blockage | 0.473 | legal | 5 | 0.941 | 1.000 |
| gnlw-2025-04-26-0630-180m-blockage-0 | blockage | 0.534 | legal | 11 | 0.269 | 0.200 |
| gnlw-2025-03-10-0700-210m-blockage-0 | blockage | 0.562 | legal | 11 | 0.200 | 0.000 |
| gnlw-2025-04-23-1630-150m-blockage-0 | blockage | 0.589 | legal | 12 | 0.265 | 0.000 |
| gnlw-2025-01-25-0730-180m-compound-0 | compound | 0.631 | legal | 9 | 0.333 | 0.260 |
| gnlw-2025-02-13-0954-180m-real-0 | real | 0.673 | legal | 11 | 0.251 | 0.484 |
| gnlw-2025-02-16-1700-210m-compound-0 | compound | 0.796 | legal | 5 | 1.000 | 1.000 |
| gnlw-2025-04-25-1730-210m-blockage-0 | blockage | 0.806 | legal | 7 | 0.200 | 0.213 |
| gnlw-2025-04-26-1700-180m-blockage-0 | blockage | 0.837 | legal | 9 | 0.225 | 0.000 |
| gnlw-2025-05-26-1630-150m-blockage-0 | blockage | 0.902 | legal | 10 | 0.253 | 0.722 |
| gnlw-2025-01-26-1700-210m-compound-0 | compound | 0.929 | legal | 11 | 0.352 | 0.559 |
| gnlw-2025-05-26-1800-150m-blockage-0 | blockage | 0.957 | legal | 9 | 0.682 | 0.957 |
| gnlw-2025-01-22-1800-150m-loop-0 | loop | 1.000 | optimal | 10 | 0.744 | 0.000 |
| gnlw-2025-01-25-0630-180m-loop-0 | loop | 1.000 | optimal | 5 | 0.871 | 1.000 |
| gnlw-2025-02-16-1530-210m-compound-0 | compound | 1.000 | optimal | 4 | 0.216 | 0.200 |
| gnlw-2025-03-12-1730-210m-loop-0 | loop | 1.000 | optimal | 8 | 0.212 | 0.000 |
| gnlw-2025-03-13-0706-180m-real-0 | real | 1.000 | optimal | 5 | 1.000 | 1.000 |
| gnlw-2025-03-16-1630-180m-compound-0 | compound | 1.000 | optimal | 5 | 0.331 | 1.000 |
| gnlw-2025-03-16-1700-210m-compound-0 | compound | 1.000 | optimal | 12 | 0.953 | 0.000 |
| gnlw-2025-04-27-0800-210m-compound-0 | compound | 1.000 | optimal | 9 | 0.777 | 0.847 |
| gnlw-2025-04-27-0830-210m-blockage-0 | blockage | 1.000 | optimal | 5 | 0.315 | 0.315 |
| gnlw-2025-05-26-0730-150m-blockage-0 | blockage | 1.000 | optimal | 4 | 0.985 | 0.971 |
| gnlw-2025-05-31-0900-180m-compound-0 | compound | 1.000 | optimal | 6 | 1.000 | 0.986 |
| gnlw-2025-06-22-1600-240m-compound-0 | compound | 1.000 | optimal | 3 | 1.000 | 1.000 |
