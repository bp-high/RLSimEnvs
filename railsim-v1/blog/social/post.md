# X / Twitter thread: Inkling 2 Small on RailSimEnv

Media in this folder: `inkling-railsim-30s.mp4` (31 s, 1920×1080, no audio; rebuild with `make_video.py`) and
`results.png` (the results chart, a still from the video).

**Before posting:** the write-up Space (bpHigh/RailSimEnv-Inkling) is still **private**; make it public first, or
drop its link from post 7. Character counts are as X counts them (each link counts as 23).

## 1/7  (271 characters)

**Attach:** `inkling-railsim-30s.mp4`

```text
We post-trained Inkling 2 Small with reinforcement learning to dispatch real Dutch trains during disruptions.

On 45 held-out tasks: 0.119 → 0.316 mean reward (1.0 = a solver-proven optimum).

The biggest early gain came from giving it room to think, not from training. 🧵
```

## 2/7  (264 characters)



```text
The job: 55 km of mostly single track between Leeuwarden and Groningen. A train breaks down and the next 3 hours need re-planning: hold, turn back, cancel.

RailSimEnv rebuilds real 2025 days from public train records and grades every plan against the proven best.
```

## 3/7  (213 characters)



```text
Round 1, 64K context: 0.119 → 0.186 after 23 RL steps.

Most of that gain was budgeting: shorter reasoning, more drafts checked, always a plan at the end. Turning effort up without more room made it worse (0.111).
```

## 4/7  (243 characters)

**Attach:** `results.png`

```text
Round 2, 262K context: 0.259 before any training. More room alone more than doubled the score.

12 RL steps later: 0.316. Better on 22 tasks, worse on 13, same on 10. Legal plans now cost 23% more than the optimum on the median, down from 37%.
```

## 5/7  (205 characters)



```text
Best case: litter closes half the line in the evening rush.

Before RL the model cancels 5 trains (0.20). After RL it cancels 1 and turns 6 back where they can still run: exactly the proven optimum (1.00).
```

## 6/7  (253 characters)



```text
The honest part: the 95% interval for the round-2 gain is −0.008 to +0.126, so it is not settled yet.

And on a short 26-minute closure, the trained model now waits instead of turning trains back (0.76 → 0.20). One more rollout of both evals would tell.
```

## 7/7  (249 characters)



```text
Everything is open: the environment and a 3D replay of every plan:
huggingface.co/spaces/bpHigh/RailSimEnv

Write-up: huggingface.co/spaces/bpHigh/RailSimEnv-Inkling

Trained with LoRA on Thinking Machines' Tinker API. Data: Rijden de Treinen (CC BY 4.0), map © OpenStreetMap contributors.
```

## Alt text

**Video:** "A 30-second summary. Title: Post-training Inkling 2 Small to run a railway. A 3D replay of a train held
by a breakdown near Leeuwarden. A bar chart of mean reward on 45 held-out tasks: 64K context 0.119, with RL 0.186;
262K context 0.259, with RL 0.316; a rule-based dispatcher scores 0.180. Side-by-side 3D replays of the same task
before and after RL: on 13 March the base plan's late train runs into conflicts while the trained plan's train is
already back in Leeuwarden; on 16 February the base plan holds a train at Buitenpost and cancels its next runs, the
trained plan turns it and runs it on time. A caveat: the gain is +0.056 per task with a 95% interval of −0.008 to
+0.126, and on a short closure the trained model waits too long. Ends with links to the RailSimEnv Space."

**Results image:** "Bar chart, mean reward on 45 held-out tasks (1.0 = the proven optimum on every task): 64K context
0.119, 64K with RL 0.186, 262K context 0.259, 262K with RL 0.316. A dashed line marks a rule-based dispatcher at 0.180."

## Single-post version (no thread)

```text
We post-trained Inkling 2 Small with RL to dispatch real Dutch trains during disruptions.

45 held-out tasks: 0.119 → 0.316 mean reward (1.0 = solver-proven optimum). Room to think did more than training; RL then made the plans cheaper.

huggingface.co/spaces/bpHigh/RailSimEnv
```
