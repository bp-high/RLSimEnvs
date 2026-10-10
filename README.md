# RLSimEnv

Simulation RL environments rebuilt from real operational records and graded against a proven optimum.

| environment | the job | data | status |
|---|---|---|---|
| [**RailSimEnv v1**](railsim-v1/) | dispatch Arriva's trains on the single-track Groningen - Leeuwarden line after a real 2025 breakdown, collision, signal or points failure: hold, short-turn and cancel trains so opposing trains cross safely | Rijden de Treinen train archive and disruptions (CC BY 4.0), OpenStreetMap (2025) | 45 eval tasks chosen for hardness (best model 0.556) + 159 train tasks, OpenEnv server, proven CP-SAT references, 3D replay, eval explorer, [Space](https://huggingface.co/spaces/bpHigh/RailSimEnv) |

The recipe:
1. Take one real job and rebuild it from the records it leaves behind.
2. Stack the disruptions a practitioner meets.
3. Prove the best answer with an exact solver.
4. Grade the agent deterministically against it. The grader is pure Python, and the agent acts only through MCP tools.
