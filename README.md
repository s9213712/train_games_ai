# Train Games AI

Unified workspace for four game AI training dashboards:

| Project | Description | Quick start |
| --- | --- | --- |
| `chess-ai` | Chess training dashboard with optional Stockfish support. | `cd chess-ai && ./run_web_dashboard.sh` |
| `snake-ai` | Snake reinforcement-learning trainers, evaluators, and dashboard. | `cd snake-ai && ./run_web_dashboard.sh` |
| `soccer-ai` | Browser soccer simulator plus backend RL trainer. | `cd soccer-ai && ./run_web_dashboard.sh` |
| `tetris-ai` | Browser Tetris simulator plus backend afterstate RL trainer. | `cd tetris-ai && ./run_web_dashboard.sh` |

Each subproject keeps its own README and runtime instructions.

## Training Integrity

Every dashboard now treats a training chunk as a candidate transaction. A
candidate is not served or checkpointed merely because parameters changed:
the policy must change its deterministic behavior, improve on a paired
evaluation, and satisfy a separate promotion check without regressing the
protected reference. An unaccepted policy is never served or written as the
authoritative checkpoint; accepted state is restored after rejection, while
attempt counters (and Soccer's explicitly private in-memory staging) may still
advance to avoid replaying the same proposal forever. Training and evaluation
seed namespaces are disjoint, protected-best objectives are compared only
under the same frozen protocol, and stale checkpoint formats are quarantined
instead of silently treated as new evidence.

The browser-only Hamiltonian/mutation previews in Snake, Soccer, and Tetris are
labelled separately from backend learning. They do not increment accepted
training progress or qualify a checkpoint.

## Runtime Model Audit

After local training, run:

```bash
python3 scripts/audit_usable_models.py
```

The audit does not trust embedded "improved" flags. It reloads each protected
artifact, compares its deterministic behavior against a fresh or built-in
baseline on independent same-seed episodes/positions, validates the current
checkpoint protocol and acceptance evidence, and writes
`runtime/usable_model_audit_latest.json`. `overall_training_verified` is false
until all four runtime artifacts pass that strict check; an old artifact can be
safe because the dashboard quarantines it while still not counting as verified
training.

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
