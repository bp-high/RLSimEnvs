# RL on RailSimEnv through Tinker

Fine-tune a Tinker-hosted model on RailSimEnv's train split with reinforcement learning, and measure it on the eval
split before and after. Built for Thinking Machines' **Inkling 2 Small** (`thinkingmachines/Inkling-2-Small`), a
rank-8 LoRA trained through the [Tinker](https://tinker-docs.thinkingmachines.ai/) API.

| file | |
|---|---|
| `../envs/rail_ops/openenv/rail_openenv/agent_tinker.py` | the episode: `RailEnv` (what RL steps through), `play`, and `run_episode` for the eval |
| `train_rl.py` | the RL loop, with a spending cap |
| `../eval/run_eval.py` | `tinker:` model specs: the base model, or a checkpoint of a training run |

## Result: Inkling 2 Small, run `rl-a`

| 45 eval tasks, one rollout each | before RL | after RL |
|---|---:|---:|
| mean reward (95% CI) | **0.119** (0.082-0.166) | **0.186** (0.142-0.242) |
| submitted a plan | 34 | 45 |
| legal plans | 9 | 13 |
| turns cut at the 8,192-token limit | 84 of 263 | 10 of 284 |
| output tokens per episode | 22,900 | 11,700 |

+0.067 per task (95% interval +0.024 to +0.115, paired over tasks). 23 steps, 1,472 episodes, 62.9M trained tokens;
3 hours of sampling and half an hour of training. The held-out train tasks went from 0.061 to 0.169, and from 62 %
of episodes submitting to all of them. Details and every task:
`../results/rollouts/2026-11-12-inkling-2-small-rl-a-eval45/analysis.md`; the run: `../results/train/rl-a/`.

What that run taught about the settings:
- **lr 1e-4, not 4e-5.** Six steps at 4e-5 moved nothing measurable; at 1e-4 the behaviour changed within four.
  The default here is now 1e-4.
- **Save the state every step** (`--save-every 1`, now the default): a state save takes 30 s, and the container
  restarted once mid-run.
- **Compare tasks, not means.** One rollout per task is noisy; `eval/compare.py` pairs the two runs.

## Setup

```bash
cd railsim-v1/envs/rail_ops
uv pip install --python .venv/bin/python tinker tinker-cookbook     # pulls PyTorch; about 6 GB
export TINKER_API_KEY=...
```

## Before, train, after

```bash
cd railsim-v1
# 1. the base model on the 45 eval tasks
envs/rail_ops/.venv/bin/python eval/run_eval.py --run 2026-11-12-inkling-2-small-base-eval45 \
    --models tinker:thinkingmachines/Inkling-2-Small@high --split eval --concurrency 45

# 2. RL on the train split (resumes if interrupted)
envs/rail_ops/.venv/bin/python train/train_rl.py --run rl-a --max-credits 150

# 3. the trained checkpoint on the same 45 tasks
envs/rail_ops/.venv/bin/python eval/run_eval.py --run 2026-11-12-inkling-2-small-rl-a-eval45 \
    --models "tinker:thinkingmachines/Inkling-2-Small#rl-a@high" --split eval --concurrency 45
```

```bash
# 4. the two runs, task by task
envs/rail_ops/.venv/bin/python eval/compare.py 2026-11-12-inkling-2-small-base-eval45 \
    2026-11-12-inkling-2-small-rl-a-eval45
```

`#rl-a` is the last sampler checkpoint in `results/train/rl-a/checkpoints.jsonl`; `#rl-a:000012` names one, and
`#tinker://...` is a sampler path.

## The episode

The same rules as the other harnesses (12 turns, the task's rules as the system prompt, `check_plan` 10 times,
`submit_plan` once, two turns without a tool call end the episode, no submit scores 0), with what Tinker's API adds:

- **A 65,536-token context.** The other rows on the board ran with a million. A check result is 3,000 to 10,000
  tokens, so the note after each turn says how much context is left, and when another check would no longer fit
  beside the final answer the next turn is announced as the last. A prompt with no room left ends the episode
  (`context_length`).
- **8,192 output tokens per turn** (64,000 for the other rows, which would not fit).
- **Effort high** (the model's default, the renderer's 0.9), temperature 1.

So these rows compare with each other, base against trained, not with the rest of the board.

The eval and the training loop play the same class (`RailEnv`), so the model is evaluated on exactly the prompts it
was trained on. Each turn's prompt is the previous prompt, the tokens the model sampled and the new messages, token
for token: an episode is one training sequence, at most one context long.

## The training loop

One step:
1. draw 16 train tasks (a seeded order over the pack, reshuffled each pass);
2. play each 4 times with the current weights;
3. centre the rewards within each task's group (the advantage), and drop groups whose rewards are all equal;
4. one optimizer step on every sampled token, weighted by its episode's advantage (`importance_sampling`);
5. refresh the sampling weights.

16 of the 159 train tasks are held out (seeded) and played once every 6 steps: a learning curve that never touches
the eval split. The reward is the environment's own (0 without a submit, below 0.2 for a plan that breaks a rule,
0.2 to 1.0 for a legal one).

| flag | default | |
|---|---|---|
| `--groups`, `--group-size` | 16, 4 | 64 episodes a step |
| `--steps` | 36 | the budget usually ends the run first |
| `--lr`, `--lora-rank` | 1e-4, 8 | checkpoint size grows with the rank (below) |
| `--save-every` | 1 | steps between saved states: a restart resumes from the last |
| `--effort` | high | train at the effort you evaluate at: it is part of the prompt |
| `--val-tasks`, `--val-every` | 16, 6 | |
| `--max-credits` / `--max-train-tokens` | required | stop before a step that would pass it |

## Spending

`--max-credits` counts every token the run samples, prefills and trains on at the rates Tinker publishes (fetched
when the run starts) and stops before a step that would pass the cap; it then saves the final checkpoint. Training
is the larger part: every episode is trained on once, at up to one context of tokens. Files record tokens and time,
never prices.

Checkpoints are large, and a free Tinker plan holds 100 GB or 100 checkpoints. Measured at rank 32: a training state
is 50 GB and a sampler checkpoint 17 GB, so a second state does not fit beside the first and the run would stop.
Rank 8, the default, is a quarter of that. The loop keeps the newest training state, the two newest sampler
checkpoints (saved at validations and at the end) and the best on validation, and deletes the rest. The final ones
have no expiry: delete them when done (`tinker checkpoint list`, `tinker checkpoint delete`).

## Files of a run

`results/train/<run>/`: `config.json`, `metrics.jsonl` (a row per step and per validation: rewards, how episodes
ended, tokens, seconds), `episodes.jsonl` (a row per episode), `checkpoints.jsonl` (the `tinker://` paths),
`rollouts/` (a few transcripts per step; git-ignored).
