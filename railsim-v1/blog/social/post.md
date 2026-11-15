# X / Twitter post: Inkling 2 Small on RailSimEnv

One post, with the video attached. Media in this folder: `inkling-railsim-30s.mp4` (31 s, 1920×1080, no audio;
rebuild with `make_video.py`). `results.png` (the results chart, a still from the video) is there if you would rather
post an image.

## Post (280 characters as X counts them; the link counts as 23)

**Attach:** `inkling-railsim-30s.mp4`

```text
We post-trained Inkling 2 Small with RL to dispatch real Dutch trains during disruptions.

45 held-out tasks: 0.119 → 0.316 mean reward (1.0 = solver-proven optimum). When litter closed half the line, it went from cancelling 5 trains to the exact optimum.

huggingface.co/spaces/bpHigh/RailSimEnv
```

The link is the public environment Space. The write-up (bpHigh/RailSimEnv-Inkling) is still private; make it public
before adding its link.

## Alt text for the video

"A 30-second summary. Title: Post-training Inkling 2 Small to run a railway. A 3D replay of a train held by a
breakdown near Leeuwarden. A bar chart of mean reward on 45 held-out tasks: 64K context 0.119, with RL 0.186; 262K
context 0.259, with RL 0.316; a rule-based dispatcher scores 0.180. Side-by-side 3D replays of the same task before
and after RL: on 13 March the base plan's late train runs into conflicts while the trained plan's train is already back
in Leeuwarden; on 16 February the base plan holds a train at Buitenpost and cancels its next runs, the trained plan
turns it and runs it on time. Then the numbers behind the gain: +0.056 per task, better on 22 tasks and worse on 13,
with a 95% interval of −0.008 to +0.126, and a short closure where the trained model waits too long. Ends with links
to the RailSimEnv Space."
