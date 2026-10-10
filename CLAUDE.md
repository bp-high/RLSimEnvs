# RLSimEnv

Simulation RL environments rebuilt from real records and graded against a proven optimum. The current environment is
**RailSimEnv v1** (`railsim-v1/`): dispatching Arriva's trains on the single-track Groningen - Leeuwarden line
(Netherlands) on real January-June 2025 days.


Quick facts:
- **Code** lives in `railsim-v1/envs/rail_ops/`: `core/` (rail_core) and `openenv/` (rail_openenv).
- **Tests:** `cd railsim-v1/envs/rail_ops && .venv/bin/python -m pytest core/tests openenv/tests -q`. Set up the venv
  first, as in `claude-sessions/CONTEXT.md`.
- **Evals:** `railsim-v1/eval/run_eval.py`. Claude models through `rail_openenv/agent.py`; open models as
  `hf:<model>:<provider>` through Hugging Face's router (`agent_openai.py`, billed with `--bill-to`). The results
  table is in `railsim-v1/README.md`; each run's `index.json`, `board.md` and `analysis.md` are committed.
- **Packs:** changing generator, physics, rule or solver code changes task content, so the packs in `tasks/` must be
  rebuilt (and the eval re-selected). Prompt changes do not.
- **Space:** [bpHigh/RailSimEnv](https://huggingface.co/spaces/bpHigh/RailSimEnv) (private), a Gradio-SDK Space on
  ZeroGPU serving the whole server. Rebuild and push with `railsim-v1/space/build.py --push bpHigh/RailSimEnv -m "..."`.
  Nothing on the Hub may mention Claude or these notes (build.py refuses), and its commits carry no attribution lines.
- **No model prices** anywhere (docs, boards, code): costs depend on prompt caching and are not comparable. Report
  tokens and time instead.
- **Data licences:** Rijden de Treinen data is CC BY 4.0 (credit "Rijden de Treinen"); OSM-derived files are ODbL.
