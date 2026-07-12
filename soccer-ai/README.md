# Soccer AI Trainer

Browser-based soccer training dashboard inspired by the local `snake-ai` project shape.

## Run

```bash
./run_web_dashboard.sh
```

Open the printed local URL. The dashboard has two deliberately separate layers:

- Browser Mutation Demo: an 11v11 visual simulation with ephemeral random weight mutation. It does not call the RL API, update the backend policy, or write a checkpoint.
- Checkpointed Backend RL: a Python tactical soccer environment and policy-gradient trainer exposed through `/api/rl/*`, with one compressed match minute per environment step.

The simulator implements the core formal match structure: kickoff, two halves, stoppage-time allowance, minimum-player abandonment, substitutions, advantage, dropped balls, offside, direct and indirect free kicks, fouls, handball, yellow/red cards, penalties, throw-ins, corners, goal kicks, goalkeeper penalty-area behavior, goalkeeper saves, back-pass handling, second-touch restart offences, goalkeeper hold-time sanctions, and goal restarts. The page shows live rule events, stamina, discipline, match metrics, cumulative win rate, reward history, set pieces, active player counts, substitutions, saves, and current policy weights.

Players have role-specific physical and technical attributes: stamina, recovery, acceleration, max speed, kick power, passing skill, tackling skill, and foul risk. Sprinting and repeated actions drain stamina; tired players run slower and kick/pass less accurately. Teams can make limited substitutions to restore tired outfield players.

## Backend RL training

Run the local server:

```bash
python3 server.py --host 127.0.0.1 --port 7872
```

Then open `http://127.0.0.1:7872/` and use the Backend RL controls:

- `Start Backend RL` / `Pause Backend RL` runs policy-gradient training in a background thread.
- `10 Episodes` performs a deterministic manual training batch.
- `Reset` clears the current trainer state and runtime outputs.
- `Load Replay` / replay transport controls load and inspect the latest backend tactical replay on the pitch canvas.
- `Evaluate` runs frozen-policy matches without changing weights, Elo, history, or opponent pool.
- Learning rate, gamma, sampling temperature, self-play, league play, and the optional state-based coach can be changed from the dashboard. These settings are persisted in `soccer_policy.json` and restored on restart.

Backend training uses pure reinforcement learning by default (`coach_enabled=false`). The optional coach is not a constant possession label: it selects among conserve, low block, direct attack, counter, high press, possession, and balanced from the observed score, time, ball position, possession, and stamina. Its effective update rate is capped at 25% of the policy-gradient learning rate and both configured/effective rates are reported by the state API.

Runtime artifacts are written under `runtime/`:

- `soccer_policy.json`: last policy that passed both acceptance stages, plus resumable trainer state.
- `soccer_policy.best.json`: protected policy promoted under the fixed canonical context.
- `training_metrics.jsonl`: episode-level score, win/loss/draw, xG, possession, stamina, discipline, and top action probabilities.
- `latest_replay.json`: latest tactical replay frames for inspection or future visual replay playback.
- `latest_evaluation.json`: latest frozen-policy evaluation summary.

The backend environment models ball-out-of-play boundaries instead of clamping every touch inside the pitch. Overhit attacks can become goal kicks, deflected defensive actions can become corners, wide misses and saves are separated, throw-ins are offside-exempt, and goal kicks/corners/throw-ins have different tactical profiles. Stoppages from goals, cards, injuries, substitutions, free kicks, goal kicks, corners, and throw-ins feed into added time.

## Learning approach

Football does not provide a single correct action for each state, so the backend's default and primary update is reinforcement learning from shaped football rewards. The optional state-based coach described above is an explicit auxiliary signal, not a claim of ground-truth labels. Each action receives a shaped reward made from football signals:

- score: goals and final result
- xG and shot quality: chance creation without rewarding harmless possession too much
- territory and possession: controlled field position
- defense: tackles, interceptions, and clearances
- discipline: fouls, offsides, yellow/red cards, and injury risk
- stamina: discourages reckless all-match sprinting

When self-play is enabled, the red side is no longer only a fixed script. The trainer samples opponents from a small league:

- scripted red for rule grounding and stable regression checks
- current red policy for direct self-play
- frozen historical blue snapshots so the main policy cannot forget older counter-strategies

The dashboard reports league Elo, opponent type, pool size, per-episode reward terms, frozen evaluation, and a tactical replay. These numbers are diagnostics, not proof of strong football play. A useful review loop is: train a batch, evaluate without learning, load the replay, then inspect whether goals/xG came from plausible pressure, transitions, set pieces, or defensive errors.

## Acceptance guard

Guarded batches do not treat a weight change as learning. Before training, the trainer freezes the red policy, league snapshot pool, and exact opponent schedule. Baseline and candidate first play identical, fresh validation seeds that rotate after every attempted batch, so rejected proposals cannot adapt to a repeatedly reused gate. Only the blue candidate policy differs. A candidate can be served or saved only when all of these deterministic checks pass:

- its greedy actions change on a material fraction of baseline holdout states;
- the football-weighted fresh-gate objective improves by the configured margin and at least the non-lowerable built-in 1.0-point minimum effect;
- match points do not regress.
- a separate fixed 64-match canonical holdout confirms the same minimum improvement and match-point non-regression against the currently accepted policy.

The objective prioritizes win/draw match points, then goal difference and xG difference, with shaped reward carrying only a small weight. Behavior-identical candidates are rejected even if their raw weights differ. The served policy, red, league, Elo, history, replay, and checkpoint state is rolled back; a monotonic rollout counter advances so the next attempt trains and validates on fresh matches. A candidate is retained privately only when its sole rejection reason is unchanged holdout behavior, so a saturated policy can accumulate enough updates to cross an action boundary. This staged candidate exists only in memory, is never checkpointed or served, and is discarded on restart, configuration changes, or any rejection after its actions change.

Training, fresh-gate validation, and fixed canonical validation use disjoint versioned string seed namespaces (`soccer-ai/train/v1/...`, `soccer-ai/fresh-gate/v1/...`, and `soccer-ai/canonical-promotion/v1/...`). They cannot collide with each other or integer seeds used by independent audits. The canonical context also fixes its own red policy and opponent schedule; it is never used for training rollouts or the repeated first-stage gate. Historical best-checkpoint promotion compares only scores from this exact context; an incomparable context is a direct rejection and cannot overwrite the best model. Configuration writes and shutdown wait for the guard transaction, and checkpoints are atomically replaced. `soccer_policy.json` is the authoritative accepted-policy checkpoint; `soccer_policy.best.json` is only an optional derivative, so failure to write it cannot invalidate an already durable main checkpoint. Checkpoints older than version 4 are rejected rather than loaded as served policies. The acceptance guard is mandatory.

The backend API is intentionally simple:

- `GET /api/rl/state`
- `GET /api/rl/replay/latest`
- `POST /api/rl/start`
- `POST /api/rl/pause`
- `POST /api/rl/reset`
- `POST /api/rl/config`
- `POST /api/rl/step` with `{"episodes": 10}`

This is still an approximate simulator, not a certified IFAB Laws of the Game engine. It intentionally abstracts referee positioning, assistant-referee mechanics, equipment inspection, VAR protocol, all misconduct subcategories, exact wall management, exact stoppage-time accounting, and competition-specific substitution rules.
