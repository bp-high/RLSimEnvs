# Inkling 2 Small after RL on rail-v1-eval (45 tasks, one rollout each)

`thinkingmachines/Inkling-2-Small` with the adapter of the training run `rl-a` (`results/train/rl-a`, its final
checkpoint, step 23), sampled through the Tinker API with `rail_openenv.agent_tinker`. The "before" is the base model
on the same harness and settings (`results/rollouts/2026-11-12-inkling-2-small-base-eval45`): effort high, 12 turns,
8,192 output tokens per turn, a 65,536-token context, temperature 1. These two runs compare with each other, not with
the board's other rows, which had a million tokens of context and 64,000 per turn.

## In one paragraph

RL on the train split raised the mean reward on the 45 held-out eval tasks from **0.119** (95% CI 0.082-0.166) to
**0.186** (0.142-0.242): **+0.067 per task** (95% interval +0.024 to +0.115, paired bootstrap over tasks), better on
30 tasks, worse on 12, the same on 3. Most of it is the model learning to work inside its budget: before, a third of
its turns were cut at the token limit and 11 of 45 episodes ended without a plan; after, 4 % of turns are cut and
every episode submits, on half the output tokens. Legal plans went from 9 to 13 and one reached the proven optimum.
The plans themselves are still weak: 32 of 45 break a rule, and the legal ones cost more over the optimum than
before.

## Numbers

| | before RL | after RL |
|---|---:|---:|
| mean reward (95% CI) | **0.119** (0.082-0.166) | **0.186** (0.142-0.242) |
| median | 0.088 | 0.138 |
| submitted a plan | 34 / 45 | 45 / 45 |
| ended without a plan (two turns without a tool call; out of turns) | 5; 6 | 0; 0 |
| legal plans | 9 | 13 |
| submitted plans that break a rule | 25 | 32 |
| rule breaks in an illegal plan (median) | 27 | 16.5 |
| at the proven optimum | 0 | 1 |
| legal plans: cost over the optimum (median, mean) | 36 %, 64 % | 69 %, 81 % |
| turns cut at the 8,192-token limit | 84 of 263 | 10 of 284 |
| `check_plan` calls per episode | 2.4 | 4.1 |
| median turns, median seconds per episode | 6, 240 s | 6, 188 s |
| input tokens (cache reads), output tokens | 5.89M (70 %), 1.03M | 6.14M (72 %), 0.53M |

| tier | tasks | before | after | difference |
|---|---:|---:|---:|---:|
| blockage | 20 | 0.080 | 0.153 | +0.073 |
| compound | 17 | 0.120 | 0.157 | +0.037 |
| loop | 4 | 0.355 | 0.309 | -0.046 |
| real | 4 | 0.078 | 0.353 | +0.275 |

The loop and real tiers have 4 tasks each: their means move with one task (the real tier's with the plan that
reached the optimum).

## The training run

`train/train_rl.py --run rl-a`: a rank-8 LoRA, 16 train tasks x 4 episodes a step, rewards centred within a task's
group, one optimizer step a batch, 23 steps (1,472 episodes; 1,432 trained on, after dropping 10 groups whose rewards
were all equal). 143 train tasks were trained on; 16 more were held out and played once at steps 0, 6, 12, 18 and 23:

| step | mean reward | submitted | legal plans | turns cut at the limit | checks per episode | output tokens per episode |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.061 | 62% | 6% | 36% | 1.75 | 21,182 |
| 6 | 0.083 | 75% | 0% | 33% | 1.88 | 21,245 |
| 12 | 0.125 | 94% | 12% | 16% | 2.81 | 16,586 |
| 18 | 0.195 | 100% | 19% | 11% | 3.25 | 13,470 |
| 23 | 0.169 | 100% | 25% | 2% | 3.81 | 10,312 |

On those 16 tasks step 23 beats step 0 by +0.108 a task (standard error 0.030; better on 13, worse on 3).

- **The learning rate mattered.** Steps 0-5 ran at 4e-5 and moved the held-out tasks by +0.022 (se 0.020): nothing
  measurable. From step 6 it was 1e-4, and the submit rate and the cut turns moved within four steps.
- **What it cost in tokens:** 28.4M sampled, 197.1M prefilled (72 % cache hits) and 62.9M trained on, validation
  included. An episode is one training sequence of 43,900 tokens on average.
- **Time:** 3.0 hours of sampling, 0.5 of training and 0.5 of validation, in two sittings (the container restarted
  during step 6; the run resumed from its saved state). Steps took 8 to 17 minutes at first and 4 to 5 at the end,
  as the model wrote less.

## What changed, and what did not

- **Budgeting, learned.** The base model at effort high reasons past 8,192 tokens on a third of its turns; each cut
  turn spends a turn and an eighth of the context and calls no tool. The trained model writes half as much, checks
  4.1 drafts instead of 2.4, and always submits.
- **Legality, a little.** 13 legal plans against 9, and illegal plans break 16.5 rules (median) against 27. Eight of
  the 13 legal plans score 0.20 to 0.21: legal, but far from the optimum (more cancellations per legal plan, 2.4
  against 1.7).
- **Not learned: `get_situation`.** All 45 trained episodes still begin by calling it (38 before), though its text
  is already in the first message; it costs about 4,000 tokens of context each time.
- **Twelve tasks got worse**, two of them by more than 0.1: with one rollout per task, a single task's change is
  mostly noise; the paired mean is the evidence.

## Every task

| task | tier | before | after | difference |
|---|---|---:|---:|---:|
| gnlw-2025-01-22-1800-150m-loop-0 | loop | 0.485 | 0.168 | -0.317 |
| gnlw-2025-03-16-1630-180m-compound-0 | compound | 0.304 | 0.200 | -0.104 |
| gnlw-2025-05-27-1830-180m-blockage-0 | blockage | 0.156 | 0.059 | -0.096 |
| gnlw-2025-01-25-0630-180m-loop-0 | loop | 0.804 | 0.716 | -0.088 |
| gnlw-2025-02-16-1700-210m-compound-0 | compound | 0.200 | 0.150 | -0.050 |
| gnlw-2025-02-15-0700-240m-compound-0 | compound | 0.094 | 0.056 | -0.037 |
| gnlw-2025-01-26-1700-210m-compound-0 | compound | 0.226 | 0.200 | -0.026 |
| gnlw-2025-03-12-1530-180m-blockage-0 | blockage | 0.080 | 0.067 | -0.013 |
| gnlw-2025-05-27-1700-180m-blockage-0 | blockage | 0.093 | 0.080 | -0.013 |
| gnlw-2025-01-23-0800-210m-compound-0 | compound | 0.059 | 0.047 | -0.012 |
| gnlw-2025-04-26-0630-180m-blockage-0 | blockage | 0.080 | 0.070 | -0.010 |
| gnlw-2025-05-28-1830-210m-compound-0 | compound | 0.123 | 0.116 | -0.006 |
| gnlw-2025-04-27-0830-210m-blockage-0 | blockage | 0.315 | 0.315 | +0.000 |
| gnlw-2025-05-31-0800-210m-compound-0 | compound | 0.056 | 0.056 | +0.000 |
| gnlw-2025-06-22-1600-240m-compound-0 | compound | 0.200 | 0.200 | +0.000 |
| gnlw-2025-04-27-0800-210m-compound-0 | compound | 0.200 | 0.200 | +0.000 |
| gnlw-2025-05-30-0630-180m-blockage-0 | blockage | 0.117 | 0.124 | +0.007 |
| gnlw-2025-02-15-1630-150m-loop-0 | loop | 0.131 | 0.138 | +0.008 |
| gnlw-2025-01-20-1800-180m-compound-0 | compound | 0.125 | 0.133 | +0.008 |
| gnlw-2025-02-16-1730-240m-compound-0 | compound | 0.158 | 0.168 | +0.011 |
| gnlw-2025-05-26-0730-150m-blockage-0 | blockage | 0.185 | 0.200 | +0.015 |
| gnlw-2025-01-23-0900-180m-compound-0 | compound | 0.000 | 0.027 | +0.027 |
| gnlw-2025-05-26-1630-150m-blockage-0 | blockage | 0.115 | 0.154 | +0.038 |
| gnlw-2025-01-21-0626-180m-real-0 | real | 0.090 | 0.131 | +0.041 |
| gnlw-2025-03-10-0700-210m-blockage-0 | blockage | 0.088 | 0.135 | +0.047 |
| gnlw-2025-01-22-1600-240m-compound-0 | compound | 0.074 | 0.121 | +0.047 |
| gnlw-2025-02-14-1730-210m-blockage-0 | blockage | 0.000 | 0.073 | +0.073 |
| gnlw-2025-04-25-1730-210m-blockage-0 | blockage | 0.030 | 0.115 | +0.085 |
| gnlw-2025-05-26-1800-150m-blockage-0 | blockage | 0.072 | 0.160 | +0.088 |
| gnlw-2025-01-25-0730-180m-compound-0 | compound | 0.038 | 0.131 | +0.092 |
| gnlw-2025-02-13-1530-180m-blockage-0 | blockage | 0.000 | 0.100 | +0.100 |
| gnlw-2025-02-15-0630-180m-blockage-0 | blockage | 0.000 | 0.120 | +0.120 |
| gnlw-2025-03-11-0730-150m-blockage-0 | blockage | 0.000 | 0.123 | +0.123 |
| gnlw-2025-04-26-1700-180m-blockage-0 | blockage | 0.080 | 0.204 | +0.124 |
| gnlw-2025-02-12-0900-180m-real-0 | real | 0.000 | 0.129 | +0.129 |
| gnlw-2025-04-23-1630-150m-blockage-0 | blockage | 0.000 | 0.131 | +0.131 |
| gnlw-2025-03-16-1700-210m-compound-0 | compound | 0.062 | 0.200 | +0.138 |
| gnlw-2025-02-13-0954-180m-real-0 | real | 0.000 | 0.153 | +0.153 |
| gnlw-2025-02-16-1530-210m-compound-0 | compound | 0.000 | 0.156 | +0.156 |
| gnlw-2025-05-27-0830-210m-blockage-0 | blockage | 0.024 | 0.200 | +0.176 |
| gnlw-2025-01-25-1700-180m-blockage-0 | blockage | 0.000 | 0.180 | +0.180 |
| gnlw-2025-03-12-1730-210m-loop-0 | loop | 0.000 | 0.212 | +0.212 |
| gnlw-2025-02-11-0900-150m-blockage-0 | blockage | 0.174 | 0.452 | +0.278 |
| gnlw-2025-05-31-0900-180m-compound-0 | compound | 0.113 | 0.505 | +0.392 |
| gnlw-2025-03-13-0706-180m-real-0 | real | 0.222 | 1.000 | +0.778 |
