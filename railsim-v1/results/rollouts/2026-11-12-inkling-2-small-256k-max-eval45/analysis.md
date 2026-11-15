# Inkling 2 Small (base), 262,144-token context, maximum effort, on rail-v1-eval (45 tasks, one rollout each)

`thinkingmachines/Inkling-2-Small:peft:262144` through the Tinker API (`rail_openenv.agent_tinker`), the base model:
effort max (0.99), **32,768 output tokens per turn**, a 262,144-token context, 12 turns, temperature 1. Run to decide
whether to do RL in this setting.

| 45 eval tasks | base, 64K, high, 8K/turn | base, 64K, max, 8K/turn | after RL (`rl-a`), 64K, high, 8K/turn | **base, 256K, max, 32K/turn** |
|---|---:|---:|---:|---:|
| mean reward (95% CI) | 0.119 (0.082-0.166) | 0.111 (0.054-0.187) | 0.186 (0.142-0.242) | **0.259** (0.192-0.337) |
| median | 0.088 | | 0.138 | 0.200 |
| submitted a plan | 34 | 24 | 45 | 43 |
| legal plans | 9 | 7 | 13 | 25 |
| at the proven optimum | 0 | 1 | 1 | 2 |
| turns cut at the per-turn limit | 32 % | 36 % | 4 % | 13 % |
| median turns; median minutes per episode | 6; 4 | 5; 5 | 6; 3 | 10; 27 |
| output tokens per episode | 22,900 | 22,700 | 11,700 | 106,200 |

Paired over tasks, this run against:
- the base model at high, 64K: **+0.140** (95% interval +0.080 to +0.208), better on 33, worse on 8, same on 4;
- the trained model: **+0.073** (+0.005 to +0.137), better on 32, worse on 8, same on 5;
- the base model at max, 64K: +0.148 (+0.080 to +0.219), better on 38, worse on 4, same on 3.

By tier: real 0.107, blockage 0.215, loop 0.597, compound 0.268.

## What it shows

- **The budget was the bottleneck, not the effort.** At max effort with 8K a turn the model did no better than at
  high; with 32K a turn and room for ten turns it more than doubles. It finishes its reasoning (13 % of turns cut,
  against a third), checks more drafts and submits a legal plan on 25 tasks (9 before).
- **The legal plans are still costly:** 37 % over the optimum (median; 113 % mean). 18 submitted plans break a rule.
- **It costs far more per episode:** 106k output tokens, 694k prompt tokens (81 % cache reads), contexts up to
  233k tokens (143k on average at the end), 27 minutes (median; 43 at the 90th percentile). The run took 63 minutes,
  45 episodes in flight; 31.2M input and 4.8M output tokens. No errors.
