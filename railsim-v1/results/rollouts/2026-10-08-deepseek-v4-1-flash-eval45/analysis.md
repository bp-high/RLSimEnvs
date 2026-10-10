# DeepSeek-V4.1-Flash on rail-v1-eval (45 tasks, one rollout each)

`deepseek-ai/DeepSeek-V4.1-Flash` through Hugging Face's router, served by Baseten. Same harness and rules as the
Qwen and GLM runs (`rail_openenv.agent_openai`): 12 turns, 64,000 output tokens per turn, the model's own reasoning
setting, its reasoning sent back with each assistant message, a 7,200 s wall clock. 22 episodes in flight at a time.
The episode on 2025-01-25 17:00 is the 1-task pilot's, reused: it ran on the same code and settings.

## In one paragraph

DeepSeek-V4.1-Flash scores **0.534** (95% CI 0.427-0.643), second of four, close behind Qwen3.8-2.4T (0.556) and
ahead of Sonnet 5.5 (0.467) and GLM-5.3-Flash (0.360), for a fifth of Qwen's cost and half its time. It submitted on
all 45 tasks, never reached the 64k cap, and reached the proven optimum on 12, as many as Qwen. Its weakness is
legality: 12 of its 45 plans break a rule (Qwen 6, Sonnet 0), which is why its mean trails Qwen's although its legal
plans are closer to the optimum (4 % over, median; 17 % mean against Qwen's 49 %).

## Numbers

| | DeepSeek-V4.1-Flash | Qwen3.8-2.4T | Claude Sonnet 5.5 | GLM-5.3-Flash |
|---|---:|---:|---:|---:|
| mean reward (95% CI) | **0.534** (0.427-0.643) | **0.556** (0.452-0.665) | **0.467** (0.384-0.553) | **0.360** (0.251-0.477) |
| median | 0.497 | 0.534 | 0.325 | 0.200 |
| below 0.7 | 29 / 45 | 27 / 45 | 33 / 45 | 32 / 45 |
| ended without a submit | 0 | 0 | 0 | 16 |
| submitted plans that break a rule | 12 | 6 | 0 | 5 |
| at the proven optimum | 12 | 12 | 4 | 6 |
| legal plans: cost over the optimum (median, mean) | 4 %, 17 % | 5 %, 49 % | 17 %, 42 % | 6 %, 19 % |
| head to head with DeepSeek (DeepSeek better / worse / tie) | | 13 / 22 / 10 | 20 / 20 / 5 | 31 / 7 / 7 |
| real, blockage, loop, compound | 0.578, 0.429, 0.685, 0.611 | 0.546, 0.481, 0.787, 0.593 | 0.457, 0.421, 0.538, 0.507 | 0.420, 0.278, 0.250, 0.468 |
| beat the real dispatchers (real tier) | 2 of 4 | 3 of 4 | 2 of 4 | 1 of 4 |
| median turns, median seconds per episode | 11, 629 s | 11, 1,273 s | 6, 264 s | 5, 1,611 s |

## How the run went

- **Speed.** Baseten streamed about 300-370 tokens/s per stream with 22 in flight; the median episode took 10.5
  minutes and the whole run about 35 minutes.
- **Throttling, absorbed.** The router's circuit breaker paused Baseten 126 times (503) and Baseten rate-limited 7
  times (429), across 40 episodes. Each came at the start of a request, so no tokens were lost; the harness's patient
  retry (up to about 4.5 minutes per turn) waited them out and no episode failed. Before that retry existed, these
  would have killed most of the run.
- **Context.** Prompts reached 344k tokens (the reasoning goes back each turn), inside Baseten's 1M window.

## From eval/analyze.py

- episodes 45, mean reward 0.534, median 0.497
- predicted mean 0.443; Spearman(predicted, actual) 0.15
- reward < 0.7 on 29/45; at the optimum on 12
- feasible plans cost 4% more than the optimum (median), 17% (mean)
- where the extra cost went (sum over feasible plans, plan minus optimum): delay +2060, unserved +1740, handed_on +806; cancellations +4, short-turns -16
- real tier: cheaper than the real dispatchers' outcome on 2/3 feasible plans

| task | tier | runs | predicted | reward | cost | optimum | floor |
|---|---|---:|---:|---:|---:|---:|---:|
| gnlw-2025-05-30-0630-180m-blockage-0 | blockage | 29 | 0.40 | 0.069 | None | 454 | 385 |
| gnlw-2025-03-10-0700-210m-blockage-0 | blockage | 34 | 0.52 | 0.071 | None | 455 | 386 |
| gnlw-2025-02-14-1730-210m-blockage-0 | blockage | 33 | 0.29 | 0.079 | None | 875 | 776 |
| gnlw-2025-01-23-0900-180m-compound-0 | compound | 30 | 0.42 | 0.087 | None | 853 | 704 |
| gnlw-2025-02-13-1530-180m-blockage-0 | blockage | 30 | 0.43 | 0.093 | None | 648 | 588 |
| gnlw-2025-03-12-1530-180m-blockage-0 | blockage | 30 | 0.47 | 0.100 | None | 806 | 730 |
| gnlw-2025-01-20-1800-180m-compound-0 | compound | 24 | 0.26 | 0.117 | None | 664 | 539 |
| gnlw-2025-01-25-1700-180m-blockage-0 | blockage | 30 | 0.36 | 0.120 | None | 320 | 275 |
| gnlw-2025-01-23-0800-210m-compound-0 | compound | 34 | 0.47 | 0.147 | None | 894 | 841 |
| gnlw-2025-05-27-1700-180m-blockage-0 | blockage | 30 | 0.53 | 0.167 | None | 791 | 670 |
| gnlw-2025-02-15-1630-150m-loop-0 | loop | 26 | 0.38 | 0.185 | None | 275 | 234 |
| gnlw-2025-02-12-0900-180m-real-0 | real | 31 | 0.19 | 0.187 | None | 643 | 529 |
| gnlw-2025-02-16-1730-240m-compound-0 | compound | 19 | 0.45 | 0.200 | 2283 | 656 | 652 |
| gnlw-2025-04-26-0630-180m-blockage-0 | blockage | 20 | 0.23 | 0.201 | 1646 | 1047 | 980 |
| gnlw-2025-02-15-0700-240m-compound-0 | compound | 32 | 0.52 | 0.203 | 1345 | 858 | 785 |
| gnlw-2025-03-11-0730-150m-blockage-0 | blockage | 26 | 0.60 | 0.219 | 2591 | 2043 | 1851 |
| gnlw-2025-01-25-0730-180m-compound-0 | compound | 26 | 0.49 | 0.299 | 1494 | 1302 | 1218 |
| gnlw-2025-01-22-1600-240m-compound-0 | compound | 38 | 0.54 | 0.314 | 809 | 670 | 627 |
| gnlw-2025-04-27-0830-210m-blockage-0 | blockage | 18 | 0.25 | 0.315 | 1380 | 1283 | 1283 |
| gnlw-2025-02-15-0630-180m-blockage-0 | blockage | 20 | 0.30 | 0.383 | 2507 | 2300 | 2119 |
| gnlw-2025-05-31-0800-210m-compound-0 | compound | 32 | 0.60 | 0.422 | 1347 | 1179 | 1017 |
| gnlw-2025-04-23-1630-150m-blockage-0 | blockage | 26 | 0.46 | 0.480 | 2054 | 1926 | 1782 |
| gnlw-2025-04-25-1730-210m-blockage-0 | blockage | 33 | 0.57 | 0.497 | 1427 | 1370 | 1355 |
| gnlw-2025-02-13-0954-180m-real-0 | real | 30 | 0.40 | 0.526 | 225 | 172 | 154 |
| gnlw-2025-05-27-0830-210m-blockage-0 | blockage | 34 | 0.50 | 0.547 | 848 | 795 | 768 |
| gnlw-2025-01-22-1800-150m-loop-0 | loop | 25 | 0.58 | 0.557 | 462 | 397 | 336 |
| gnlw-2025-01-21-0626-180m-real-0 | real | 29 | 0.53 | 0.598 | 511 | 445 | 356 |
| gnlw-2025-05-28-1830-210m-compound-0 | compound | 31 | 0.58 | 0.693 | 473 | 436 | 383 |
| gnlw-2025-04-26-1700-180m-blockage-0 | blockage | 30 | 0.42 | 0.698 | 474 | 447 | 433 |
| gnlw-2025-05-27-1830-180m-blockage-0 | blockage | 27 | 0.59 | 0.746 | 622 | 596 | 560 |
| gnlw-2025-05-26-1800-150m-blockage-0 | blockage | 25 | 0.50 | 0.879 | 1026 | 1014 | 968 |
| gnlw-2025-01-26-1700-210m-compound-0 | compound | 16 | 0.27 | 0.907 | 626 | 618 | 588 |
| gnlw-2025-05-26-1630-150m-blockage-0 | blockage | 26 | 0.57 | 0.911 | 264 | 254 | 185 |
| gnlw-2025-01-25-0630-180m-loop-0 | loop | 20 | 0.46 | 1.000 | 116 | 116 | 102 |
| gnlw-2025-02-11-0900-150m-blockage-0 | blockage | 23 | 0.41 | 1.000 | 725 | 725 | 721 |
| gnlw-2025-02-16-1530-210m-compound-0 | compound | 18 | 0.53 | 1.000 | 273 | 273 | 269 |
| gnlw-2025-02-16-1700-210m-compound-0 | compound | 16 | 0.53 | 1.000 | 796 | 796 | 787 |
| gnlw-2025-03-12-1730-210m-loop-0 | loop | 33 | 0.44 | 1.000 | 166 | 166 | 158 |
| gnlw-2025-03-13-0706-180m-real-0 | real | 26 | 0.50 | 1.000 | 378 | 378 | 378 |
| gnlw-2025-03-16-1630-180m-compound-0 | compound | 15 | 0.51 | 1.000 | 1259 | 1259 | 1256 |
| gnlw-2025-03-16-1700-210m-compound-0 | compound | 16 | 0.25 | 1.000 | 529 | 529 | 529 |
| gnlw-2025-04-27-0800-210m-compound-0 | compound | 15 | 0.32 | 1.000 | 382 | 382 | 378 |
| gnlw-2025-05-26-0730-150m-blockage-0 | blockage | 26 | 0.46 | 1.000 | 857 | 857 | 848 |
| gnlw-2025-05-31-0900-180m-compound-0 | compound | 30 | 0.36 | 1.000 | 226 | 226 | 216 |
| gnlw-2025-06-22-1600-240m-compound-0 | compound | 18 | 0.51 | 1.000 | 668 | 668 | 668 |

The predicted rewards are Sonnet's hardness predictor, fitted on Sonnet's calibration episodes; they are not a
prediction for DeepSeek.

## Every task, against the other three models

| task | tier | DeepSeek-V4.1-Flash | how it ended | turns | Qwen3.8-2.4T | Sonnet 5.5 | GLM-5.3-Flash |
|---|---|---:|---|---:|---:|---:|---:|
| gnlw-2025-05-30-0630-180m-blockage-0 | blockage | 0.069 | illegal plan | 11 | 0.055 | 0.214 | 0.000 |
| gnlw-2025-03-10-0700-210m-blockage-0 | blockage | 0.071 | illegal plan | 11 | 0.562 | 0.200 | 0.000 |
| gnlw-2025-02-14-1730-210m-blockage-0 | blockage | 0.079 | illegal plan | 12 | 0.085 | 0.200 | 0.000 |
| gnlw-2025-01-23-0900-180m-compound-0 | compound | 0.087 | illegal plan | 12 | 0.200 | 0.411 | 0.237 |
| gnlw-2025-02-13-1530-180m-blockage-0 | blockage | 0.093 | illegal plan | 11 | 0.348 | 0.258 | 0.000 |
| gnlw-2025-03-12-1530-180m-blockage-0 | blockage | 0.100 | illegal plan | 12 | 0.047 | 0.221 | 0.000 |
| gnlw-2025-01-20-1800-180m-compound-0 | compound | 0.117 | illegal plan | 11 | 0.233 | 0.200 | 0.067 |
| gnlw-2025-01-25-1700-180m-blockage-0 | blockage | 0.120 | illegal plan | 12 | 0.251 | 0.674 | 0.375 |
| gnlw-2025-01-23-0800-210m-compound-0 | compound | 0.147 | illegal plan | 11 | 0.239 | 0.395 | 0.100 |
| gnlw-2025-05-27-1700-180m-blockage-0 | blockage | 0.167 | illegal plan | 12 | 0.200 | 0.312 | 0.060 |
| gnlw-2025-02-15-1630-150m-loop-0 | loop | 0.185 | illegal plan | 12 | 0.146 | 0.325 | 0.000 |
| gnlw-2025-02-12-0900-180m-real-0 | real | 0.187 | illegal plan | 11 | 0.200 | 0.200 | 0.194 |
| gnlw-2025-02-16-1730-240m-compound-0 | compound | 0.200 | legal | 12 | 0.200 | 0.205 | 0.777 |
| gnlw-2025-04-26-0630-180m-blockage-0 | blockage | 0.201 | legal | 12 | 0.534 | 0.269 | 0.200 |
| gnlw-2025-02-15-0700-240m-compound-0 | compound | 0.203 | legal | 12 | 0.205 | 0.225 | 0.000 |
| gnlw-2025-03-11-0730-150m-blockage-0 | blockage | 0.219 | legal | 11 | 0.185 | 0.427 | 0.000 |
| gnlw-2025-01-25-0730-180m-compound-0 | compound | 0.299 | legal | 11 | 0.631 | 0.333 | 0.260 |
| gnlw-2025-01-22-1600-240m-compound-0 | compound | 0.314 | legal | 11 | 0.225 | 0.719 | 0.000 |
| gnlw-2025-04-27-0830-210m-blockage-0 | blockage | 0.315 | legal | 4 | 1.000 | 0.315 | 0.315 |
| gnlw-2025-02-15-0630-180m-blockage-0 | blockage | 0.383 | legal | 10 | 0.385 | 0.263 | 0.543 |
| gnlw-2025-05-31-0800-210m-compound-0 | compound | 0.422 | legal | 11 | 0.175 | 0.221 | 0.188 |
| gnlw-2025-04-23-1630-150m-blockage-0 | blockage | 0.480 | legal | 8 | 0.589 | 0.265 | 0.000 |
| gnlw-2025-04-25-1730-210m-blockage-0 | blockage | 0.497 | legal | 11 | 0.806 | 0.200 | 0.213 |
| gnlw-2025-02-13-0954-180m-real-0 | real | 0.526 | legal | 10 | 0.673 | 0.251 | 0.484 |
| gnlw-2025-05-27-0830-210m-blockage-0 | blockage | 0.547 | legal | 10 | 0.205 | 0.832 | 0.205 |
| gnlw-2025-01-22-1800-150m-loop-0 | loop | 0.557 | legal | 11 | 1.000 | 0.744 | 0.000 |
| gnlw-2025-01-21-0626-180m-real-0 | real | 0.598 | legal | 12 | 0.311 | 0.374 | 0.000 |
| gnlw-2025-05-28-1830-210m-compound-0 | compound | 0.693 | legal | 9 | 0.256 | 0.280 | 0.740 |
| gnlw-2025-04-26-1700-180m-blockage-0 | blockage | 0.698 | legal | 12 | 0.837 | 0.225 | 0.000 |
| gnlw-2025-05-27-1830-180m-blockage-0 | blockage | 0.746 | legal | 7 | 0.200 | 0.692 | 0.000 |
| gnlw-2025-05-26-1800-150m-blockage-0 | blockage | 0.879 | legal | 8 | 0.957 | 0.682 | 0.957 |
| gnlw-2025-01-26-1700-210m-compound-0 | compound | 0.907 | legal | 10 | 0.929 | 0.352 | 0.559 |
| gnlw-2025-05-26-1630-150m-blockage-0 | blockage | 0.911 | legal | 10 | 0.902 | 0.253 | 0.722 |
| gnlw-2025-01-25-0630-180m-loop-0 | loop | 1.000 | optimal | 5 | 1.000 | 0.871 | 1.000 |
| gnlw-2025-02-11-0900-150m-blockage-0 | blockage | 1.000 | optimal | 6 | 0.473 | 0.941 | 1.000 |
| gnlw-2025-02-16-1530-210m-compound-0 | compound | 1.000 | optimal | 10 | 1.000 | 0.216 | 0.200 |
| gnlw-2025-02-16-1700-210m-compound-0 | compound | 1.000 | optimal | 7 | 0.796 | 1.000 | 1.000 |
| gnlw-2025-03-12-1730-210m-loop-0 | loop | 1.000 | optimal | 7 | 1.000 | 0.212 | 0.000 |
| gnlw-2025-03-13-0706-180m-real-0 | real | 1.000 | optimal | 7 | 1.000 | 1.000 | 1.000 |
| gnlw-2025-03-16-1630-180m-compound-0 | compound | 1.000 | optimal | 4 | 1.000 | 0.331 | 1.000 |
| gnlw-2025-03-16-1700-210m-compound-0 | compound | 1.000 | optimal | 9 | 1.000 | 0.953 | 0.000 |
| gnlw-2025-04-27-0800-210m-compound-0 | compound | 1.000 | optimal | 7 | 1.000 | 0.777 | 0.847 |
| gnlw-2025-05-26-0730-150m-blockage-0 | blockage | 1.000 | optimal | 7 | 1.000 | 0.985 | 0.971 |
| gnlw-2025-05-31-0900-180m-compound-0 | compound | 1.000 | optimal | 7 | 1.000 | 1.000 | 0.986 |
| gnlw-2025-06-22-1600-240m-compound-0 | compound | 1.000 | optimal | 5 | 1.000 | 1.000 | 1.000 |
