# Train Games AI

Unified workspace for four game AI training dashboards:

| Project | Description | Quick start |
| --- | --- | --- |
| `chess-ai` | Chess training dashboard with optional Stockfish support. | `cd chess-ai && ./run_web_dashboard.sh` |
| `snake-ai` | Snake reinforcement-learning trainers, evaluators, and dashboard. | `cd snake-ai && ./run_web_dashboard.sh` |
| `soccer-ai` | Browser soccer simulator plus backend RL trainer. | `cd soccer-ai && ./run_web_dashboard.sh` |
| `tetris-ai` | Browser Tetris simulator plus backend afterstate RL trainer. | `cd tetris-ai && ./run_web_dashboard.sh` |

Each subproject keeps its own README and runtime instructions.

## Repository Notes

This repository tracks source code, dashboards, scripts, docs, and small static
assets. Generated training output is intentionally ignored:

- TensorBoard logs
- runtime state
- policy replay output
- model checkpoints and `.zip` bundles
- Python bytecode/cache directories

Keep large trained models outside Git or publish them through a release/artifact
store when they need to be shared.
