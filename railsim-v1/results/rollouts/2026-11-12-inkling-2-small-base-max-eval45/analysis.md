# Inkling 2 Small (base) at maximum effort on rail-v1-eval (45 tasks, one rollout each)

The base model at effort `max` (the renderer's 0.99, its highest preset), otherwise as the high-effort baseline
(`2026-11-12-inkling-2-small-base-eval45`): 12 turns, 8,192 output tokens per turn, a 65,536-token context. Run to
decide whether a next RL run should train at maximum effort.

| | high (0.9) | max (0.99) |
|---|---:|---:|
| mean reward (95% CI) | **0.119** (0.082-0.166) | **0.111** (0.054-0.187) |
| submitted a plan | 34 | 24 |
| ended with two turns in a row without a tool call | 5 | 15 |
| legal plans; at the optimum | 9; 0 | 7; 1 |
| turns cut at the 8,192-token limit | 84 of 263 (32 %) | 83 of 228 (36 %) |
| output tokens, median seconds per episode | 22,900, 240 s | 22,700, 280 s |

Paired over tasks: -0.008 (95% interval -0.066 to +0.056), better on 14, worse on 23, the same on 8. No difference
in the mean; but more reasoning per turn inside the same 8,192 tokens means more turns that never call a tool, and
21 of 45 episodes end without a plan (11 at high). The trained model (`...-rl-a-eval45`, high) scores 0.186.

**For a next run:** at 8,192 tokens a turn, maximum effort is not worth training at; high is the better start, and
what RL fixed at high (cut turns, no submit) is worse at max. Max might pay with a larger per-turn budget (16k or
32k), which in a 65,536-token context leaves room for only two or three turns: that needs the 262,144-token variant,
which trains at twice the rate.
