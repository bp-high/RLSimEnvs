# Inkling 2 Small before RL on rail-v1-eval (45 tasks, one rollout each)

The base model `thinkingmachines/Inkling-2-Small`, sampled through the Tinker API with `rail_openenv.agent_tinker`:
effort high, 12 turns, 8,192 output tokens per turn, a 65,536-token context, temperature 1. It is the "before" of
`results/rollouts/2026-11-12-inkling-2-small-rl-a-eval45`, whose `analysis.md` has the comparison and every task.
It does not compare with the board's other rows, which had a million tokens of context and 64,000 per turn.

- **0.119** (95% CI 0.082-0.166), median 0.088; by tier real 0.078, blockage 0.080, loop 0.355, compound 0.120.
- **34 of 45 submitted.** 5 episodes ended after two turns in a row without a tool call and 6 ran out of turns, all
  after turns cut at the 8,192-token limit: 84 of 263 turns were cut.
- **9 legal plans, none optimal.** 25 submitted plans break a rule (27 rule breaks, median). Legal plans cost 36 %
  more than the optimum (median; 64 % mean).
- 38 episodes began by calling `get_situation`, whose text is already in the first message.
- 2.4 `check_plan` calls per episode; median 6 turns and 240 s. 5.89M input tokens (70 % cache reads), 1.03M output
  tokens. No errors, no retries; the whole run took 7 minutes, 45 episodes in flight.
