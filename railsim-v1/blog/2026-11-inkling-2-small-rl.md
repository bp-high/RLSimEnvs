# Teaching a small model to run a railway: two days of RL on Inkling 2 Small

*November 2026. RailSimEnv, a train-dispatching environment built from real 2025 records, and Thinking Machines'
Inkling 2 Small, fine-tuned through the Tinker API.*

## The job

Between Leeuwarden and Groningen in the north of the Netherlands runs 55 km of mostly single track. Trains going
opposite ways can only pass each other at stations with two tracks. Eight trains an hour share it. When something
goes wrong (a broken-down train, a collision at a level crossing, a signal failure) a dispatcher has a few minutes to
re-plan: hold this train at that station, turn that one back early, cancel another.

RailSimEnv turns real days from 2025 into that job. The model reads the situation, tries draft plans with a
`check_plan` tool (which simulates every train and lists every broken rule), and submits one plan. We know the best
possible plan for every task, proven by a solver, so the score is honest:

- a plan that breaks a rule (two trains head-on on a single track, say) scores under 0.2;
- a legal plan scores between 0.2 and 1.0, and 1.0 means it matched the proven optimum.

Big models do fine but not great on the 45 hard evaluation tasks: Qwen3.8 scores 0.556, DeepSeek-V4.1-Flash 0.534,
Claude Sonnet 5.5 0.467, GLM-5.3-Flash 0.360. A simple rule-based dispatcher gets 0.180.

## The model and the setup

Inkling 2 Small is a mixture-of-experts model: 594 billion parameters, 17 billion active per token. Tinker trains a
LoRA adapter on it (a small set of extra weights) while Thinking Machines runs the hardware. We used rank 8, which
keeps checkpoints manageable (more on that below).

The training loop is plain policy-gradient RL:

1. pick a handful of training tasks (other days than the evaluation ones);
2. let the model play each one 4 times;
3. compare the 4 scores: attempts that did better than their siblings get pushed up, worse ones pushed down;
4. one update, then repeat with the new weights.

Every token the model wrote in an episode, its reasoning, its tool calls and its final plan, is one training
sequence, and the reward is the environment's own score. No human labels, no judge model.

## What happened

### Round 1: a 64K context, 8K tokens per turn

Tinker's standard Inkling 2 Small has a 65,536-token context. A single `check_plan` report can be 10,000 tokens, so we
gave the model 8,192 tokens of output per turn and told it, after every turn, how much room it had left.

**Before training: 0.119.** Worse than the rule-based dispatcher. Reading the transcripts showed why: the model
thought too long. A third of its turns hit the 8,192-token limit in the middle of reasoning, before it called any
tool, and 11 of 45 episodes ended with no plan at all.

We tried the obvious fix, more effort (maximum instead of high): **0.111**. More thinking in the same small box made
it worse, with twice as many episodes ending without a plan.

**After 23 RL steps: 0.186.** On the same 45 tasks, +0.067 per task (95% interval +0.024 to +0.115): better on 30,
worse on 12. But look at *what* it learned:

| | before | after |
|---|---:|---:|
| episodes that submitted a plan | 34 of 45 | 45 of 45 |
| turns cut off at the token limit | 32 % | 4 % |
| output tokens per episode | 22,900 | 11,700 |
| legal plans | 9 | 13 |

It learned to budget. Shorter reasoning, more drafts checked, and always a plan at the end. The plans themselves
barely improved. That is what RL does first: it finds the cheapest way to stop losing points.

One thing that mattered a lot: the learning rate. The first six steps at 4e-5 changed nothing measurable. At 1e-4 the
behaviour moved within four steps.

### Round 2: give it room to think

If the problem was room, give it room. Tinker also offers Inkling 2 Small with a 262,144-token context. We ran it at
maximum effort with 32,768 output tokens per turn.

**Before any training: 0.259.** More than double the 64K baseline, and better than the model we had just trained. 25
of 45 plans were legal (9 before) and two hit the proven optimum. The lesson from round 1 held: the bottleneck was the
budget, not the model.

So we trained this one too: 12 RL steps (8 tasks x 4 attempts per step, because each episode is now about 143,000
tokens long and much more expensive to train on).

**After 12 RL steps: 0.316.** +0.056 per task, better on 22 tasks, worse on 13. This time it wasn't about budget (the
model already submitted almost every time) but about plan quality: legal plans now cost 23 % more than the optimum on
the median, against 37 % before, and three reached the optimum.

**But it is not settled.** The 95% interval is -0.008 to +0.126, so it just touches zero. And the 16 training tasks we
held out to track progress did not move at all (0.246 before, 0.244 after), while the training batches climbed from
about 0.22 to 0.31. One more rollout of each evaluation would settle it.

## All the numbers

| Inkling 2 Small on the 45 eval tasks | context | per turn | effort | mean reward |
|---|---|---|---|---:|
| base | 64K | 8K | high | 0.119 |
| base | 64K | 8K | max | 0.111 |
| after RL (23 steps) | 64K | 8K | high | 0.186 |
| base | 256K | 32K | max | 0.259 |
| after RL (12 steps) | 256K | 32K | max | **0.316** |

For scale: the best trained Inkling sits between the rule-based dispatcher (0.180) and GLM-5.3-Flash (0.360), a model
of similar active size, which ran with four times the room (a million tokens of context and 64,000 per turn).

## Things we noticed along the way

**Reasoning room beat reasoning effort.** Turning effort up without more room per turn made things worse. Giving it
4x the room per turn doubled the score with no training at all. The biggest gain of the whole project came from a
setting, not from RL.

**RL learns the easy lesson first.** On the 64K model the improvement was almost entirely "stop running out of
tokens, always submit". Useful, but it is format, not skill. On the long-context model, where that lesson was already
learned, the gains came slower and were about better plans.

**Small samples lie in both directions.** Training batches draw different tasks each step, so a batch can swing from
0.18 to 0.38 by luck. Our held-out set of 16 tasks and the 45-task evaluation sometimes disagreed. Comparing the same
tasks before and after (paired), rather than two averages, was the only way to see real effects.

**The model found a bug in the grader.** During training it tried a plan that starts a train at its own final
station, so there was nothing left to run. The simulator crashed instead of calling it a broken rule. RL is very good
at finding the corners of an environment nobody tested. Fixed, with a test.

**Some habits stuck.** Every episode, before and after training, begins by calling `get_situation`, though the
situation is already in the first message. That costs about 4,000 tokens of context each time and training never
removed it.

**Checkpoints are big.** A rank-32 adapter's training state for this model is 50 GB, and a sampler copy 17 GB.
Tinker's free storage is 100 GB, so two training states would not fit side by side. Rank 8 brought it to 12.5 GB.

**The machine running the loop is not the machine doing the training.** Tinker never failed us. The cloud container
that ran our loop restarted three times in one night, each time throwing away the episodes it was in the middle of
playing. We changed the loop to save its state the moment a step finishes and to keep every finished episode on disk,
so a restart now costs only the few episodes in flight.

## What it took

| | 64K run | 256K run |
|---|---:|---:|
| RL steps | 23 | 12 |
| episodes played (training) | 1,472 | 384 |
| tokens sampled | 28.4M | 46.5M |
| tokens trained on | 62.9M | 57.5M |
| hours of sampling | 3.0 | 6.0 |

## What's next

1. **Settle round 2:** one more rollout of the 256K evaluations, before and after.
2. **Train longer at 256K:** round 1 needed about 12 steps before anything moved; round 2 stopped at 12.
3. **Try high effort at 256K:** maybe the room matters and maximum effort does not.
4. **Tell the model the situation is already there,** or let RL learn to skip `get_situation`.

Everything is open: the environment and its tasks, the training script (`railsim-v1/train/train_rl.py`), every
evaluation run with each plan the model submitted, and a Hugging Face Space
([bpHigh/RailSimEnv](https://huggingface.co/spaces/bpHigh/RailSimEnv)) whose first tab replays all of Inkling's plans
on a 3D map of the line.

*Data: Rijden de Treinen (CC BY 4.0). Map: © OpenStreetMap contributors (ODbL).*
