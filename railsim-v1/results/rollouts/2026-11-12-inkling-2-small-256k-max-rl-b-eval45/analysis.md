# Inkling 2 Small 256K at max effort, after RL (`rl-b`), on rail-v1-eval (45 tasks, one rollout each)

`thinkingmachines/Inkling-2-Small:peft:262144` with the adapter of training run `rl-b` (`results/train/rl-b`, the
sampler checkpoint saved at its step-12 validation), sampled through the Tinker API: effort max (0.99), 32,768 output
tokens per turn, a 262,144-token context, 12 turns, temperature 1. The "before" is the base model on the same settings
(`2026-11-12-inkling-2-small-256k-max-eval45`, 0.259).

## In one paragraph

RL moved the mean from **0.259** (95% CI 0.192-0.337) to **0.316** (0.231-0.407): **+0.056 per task**, but the 95%
interval (-0.008 to +0.126, paired over tasks) still includes zero; better on 22, worse on 13, the same on 10. The
direction agrees with the training batches (steps 6-11 averaged 0.31 against 0.22 for steps 0-5) and not with the
16 held-out train tasks (0.246 at step 0, 0.244 at step 12). What changed is plan quality rather than legality: legal
plans are as frequent (58 % against 56 %) but cheaper, 23 % over the optimum (median; 61 % mean) against 37 %
(113 %), and three reach the proven optimum (two before). Unlike the 64K run, there was no budget habit to learn here.

## Numbers

| | before RL | after RL (`rl-b`) |
|---|---:|---:|
| mean reward (95% CI) | 0.259 (0.192-0.337) | **0.316** (0.231-0.407) |
| median | 0.200 | 0.200 |
| submitted a plan | 43 | 44 |
| legal plans | 25 | 26 |
| submitted plans that break a rule | 18 | 18 |
| at the proven optimum | 2 | 3 |
| legal plans: cost over the optimum (median, mean) | 37 %, 113 % | 23 %, 61 % |
| turns cut at the 32,768-token limit | 52 of 403 | 69 of 396 |
| `check_plan` calls per episode | 6.0 | 5.4 |
| median minutes per episode | 27 | 21 |
| input tokens (cache reads), output tokens | 31.2M (81 %), 4.78M | 31.3M (80 %), 5.16M |

| tier | tasks | before | after | difference |
|---|---:|---:|---:|---:|
| blockage | 20 | 0.215 | 0.212 | -0.003 |
| compound | 17 | 0.268 | 0.324 | +0.056 |
| loop | 4 | 0.597 | 0.715 | +0.119 |
| real | 4 | 0.107 | 0.398 | +0.291 |

The real and loop tiers have four tasks each; their means move with one task.

## The training run

`train/train_rl.py --run rl-b`: rank-8 LoRA, 8 train tasks x 4 episodes a step, lr 1e-4, 12 steps (0-11), stopped by
its budget. Batch rewards by step: 0.192, 0.269, 0.248, 0.218, 0.184, 0.230, 0.364, 0.290, 0.237, 0.296, 0.385, 0.282.
The 16 held-out train tasks: 0.246 (step 0), 0.236 (6), 0.244 (12); at step 12, 11 legal plans against 9.

- **Tokens:** 46.5M sampled, 57.7M + 247.1M cached prefilled, 57.5M trained, validation included. An episode is one
  training sequence of about 143k tokens.
- **Time:** 6.0 hours of sampling, 0.5 of training, over 9 hours of wall time.
- **Three container restarts** lost about 35 credits of sampling and one trained step (step 5 was trained twice). Since
  then the state is saved right after each step and finished episodes are cached on disk.
- **A grader crash found by the policy:** a plan starting a run at its own last point crashed `check_plan` (KeyError).
  Fixed in `check.py` during the run; the training process ran the old code until step 6, so a few drafts got an
  error message instead of a rule-break report.

## Every task

| task | tier | before | after | difference |
|---|---|---:|---:|---:|
| gnlw-2025-05-31-0900-180m-compound-0 | compound | 0.756 | 0.200 | -0.556 |
| gnlw-2025-04-23-1630-150m-blockage-0 | blockage | 0.419 | 0.219 | -0.200 |
| gnlw-2025-05-26-0730-150m-blockage-0 | blockage | 0.351 | 0.200 | -0.150 |
| gnlw-2025-04-26-1700-180m-blockage-0 | blockage | 0.250 | 0.120 | -0.130 |
| gnlw-2025-05-28-1830-210m-compound-0 | compound | 0.123 | 0.000 | -0.123 |
| gnlw-2025-04-27-0800-210m-compound-0 | compound | 0.438 | 0.365 | -0.073 |
| gnlw-2025-05-27-1700-180m-blockage-0 | blockage | 0.153 | 0.093 | -0.060 |
| gnlw-2025-01-22-1600-240m-compound-0 | compound | 0.200 | 0.163 | -0.037 |
| gnlw-2025-05-26-1630-150m-blockage-0 | blockage | 0.169 | 0.138 | -0.031 |
| gnlw-2025-01-21-0626-180m-real-0 | real | 0.097 | 0.069 | -0.028 |
| gnlw-2025-02-13-1530-180m-blockage-0 | blockage | 0.100 | 0.080 | -0.020 |
| gnlw-2025-01-23-0900-180m-compound-0 | compound | 0.080 | 0.067 | -0.013 |
| gnlw-2025-01-23-0800-210m-compound-0 | compound | 0.112 | 0.100 | -0.012 |
| gnlw-2025-01-20-1800-180m-compound-0 | compound | 0.200 | 0.200 | +0.000 |
| gnlw-2025-01-25-0630-180m-loop-0 | loop | 1.000 | 1.000 | +0.000 |
| gnlw-2025-02-11-0900-150m-blockage-0 | blockage | 0.860 | 0.860 | +0.000 |
| gnlw-2025-02-15-1630-150m-loop-0 | loop | 0.177 | 0.177 | +0.000 |
| gnlw-2025-02-16-1730-240m-compound-0 | compound | 0.200 | 0.200 | +0.000 |
| gnlw-2025-03-12-1730-210m-loop-0 | loop | 0.890 | 0.890 | +0.000 |
| gnlw-2025-03-16-1700-210m-compound-0 | compound | 0.200 | 0.200 | +0.000 |
| gnlw-2025-04-25-1730-210m-blockage-0 | blockage | 0.200 | 0.200 | +0.000 |
| gnlw-2025-04-27-0830-210m-blockage-0 | blockage | 0.315 | 0.315 | +0.000 |
| gnlw-2025-06-22-1600-240m-compound-0 | compound | 1.000 | 1.000 | +0.000 |
| gnlw-2025-05-27-0830-210m-blockage-0 | blockage | 0.200 | 0.201 | +0.001 |
| gnlw-2025-03-11-0730-150m-blockage-0 | blockage | 0.200 | 0.202 | +0.002 |
| gnlw-2025-02-16-1530-210m-compound-0 | compound | 0.200 | 0.205 | +0.005 |
| gnlw-2025-02-14-1730-210m-blockage-0 | blockage | 0.103 | 0.109 | +0.006 |
| gnlw-2025-01-25-1700-180m-blockage-0 | blockage | 0.167 | 0.187 | +0.020 |
| gnlw-2025-02-15-0630-180m-blockage-0 | blockage | 0.240 | 0.261 | +0.021 |
| gnlw-2025-05-26-1800-150m-blockage-0 | blockage | 0.205 | 0.232 | +0.027 |
| gnlw-2025-03-10-0700-210m-blockage-0 | blockage | 0.076 | 0.106 | +0.029 |
| gnlw-2025-01-25-0730-180m-compound-0 | compound | 0.154 | 0.200 | +0.046 |
| gnlw-2025-05-30-0630-180m-blockage-0 | blockage | 0.103 | 0.172 | +0.069 |
| gnlw-2025-02-15-0700-240m-compound-0 | compound | 0.106 | 0.188 | +0.081 |
| gnlw-2025-03-12-1530-180m-blockage-0 | blockage | 0.060 | 0.147 | +0.087 |
| gnlw-2025-05-27-1830-180m-blockage-0 | blockage | 0.133 | 0.228 | +0.095 |
| gnlw-2025-02-12-0900-180m-real-0 | real | 0.000 | 0.097 | +0.097 |
| gnlw-2025-01-26-1700-210m-compound-0 | compound | 0.204 | 0.328 | +0.124 |
| gnlw-2025-05-31-0800-210m-compound-0 | compound | 0.056 | 0.181 | +0.125 |
| gnlw-2025-04-26-0630-180m-blockage-0 | blockage | 0.000 | 0.180 | +0.180 |
| gnlw-2025-02-13-0954-180m-real-0 | real | 0.201 | 0.489 | +0.289 |
| gnlw-2025-01-22-1800-150m-loop-0 | loop | 0.320 | 0.794 | +0.474 |
| gnlw-2025-03-16-1630-180m-compound-0 | compound | 0.324 | 0.912 | +0.588 |
| gnlw-2025-02-16-1700-210m-compound-0 | compound | 0.200 | 1.000 | +0.800 |
| gnlw-2025-03-13-0706-180m-real-0 | real | 0.131 | 0.938 | +0.808 |
