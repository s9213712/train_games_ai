# Soccer AI Trainer

Browser-based soccer training dashboard inspired by the local `snake-ai` project shape.

## Run

```bash
./run_web_dashboard.sh
```

Open the printed local URL. The dashboard has two training layers:

- Browser match view: an 11v11 continuous-control soccer simulation with live animation and browser-side policy evolution.
- Backend RL panel: a Python tactical soccer environment and policy-gradient trainer exposed through `/api/rl/*`, with one compressed match minute per environment step.

The simulator implements the core formal match structure: kickoff, two halves, stoppage-time allowance, minimum-player abandonment, substitutions, advantage, dropped balls, offside, direct and indirect free kicks, fouls, handball, yellow/red cards, penalties, throw-ins, corners, goal kicks, goalkeeper penalty-area behavior, goalkeeper saves, back-pass handling, second-touch restart offences, goalkeeper hold-time sanctions, and goal restarts. The page shows live rule events, stamina, discipline, match metrics, cumulative win rate, reward history, set pieces, active player counts, substitutions, saves, and current policy weights.

Players have role-specific physical and technical attributes: stamina, recovery, acceleration, max speed, kick power, passing skill, tackling skill, and foul risk. Sprinting and repeated actions drain stamina; tired players run slower and kick/pass less accurately. Teams can make limited substitutions to restore tired outfield players.

## Backend RL training

Run the local server:

```bash
python3 server.py --host 127.0.0.1 --port 7872
```

Then open `http://127.0.0.1:7872/` and use the Backend RL controls:

- `Start` / `Pause` runs training in a background thread.
- `10 Episodes` performs a deterministic manual training batch.
- `Reset` clears the current trainer state and runtime outputs.
- `Load Replay` / replay transport controls load and inspect the latest backend tactical replay on the pitch canvas.
- `Evaluate` runs frozen-policy matches without changing weights, Elo, history, or opponent pool.
- Learning rate and self-play can be changed from the dashboard before starting or stepping training.

Runtime artifacts are written under `runtime/`:

- `soccer_policy.json`: learned softmax policy weights.
- `training_metrics.jsonl`: episode-level score, win/loss/draw, xG, possession, stamina, discipline, and top action probabilities.
- `latest_replay.json`: latest tactical replay frames for inspection or future visual replay playback.
- `latest_evaluation.json`: latest frozen-policy evaluation summary.

The backend environment models ball-out-of-play boundaries instead of clamping every touch inside the pitch. Overhit attacks can become goal kicks, deflected defensive actions can become corners, wide misses and saves are separated, throw-ins are offside-exempt, and goal kicks/corners/throw-ins have different tactical profiles. Stoppages from goals, cards, injuries, substitutions, free kicks, goal kicks, corners, and throw-ins feed into added time.

## Learning approach

Football does not provide a single correct action for each state, so the backend uses reinforcement learning rather than supervised labels. Each action receives a shaped reward made from football signals:

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

The backend API is intentionally simple:

- `GET /api/rl/state`
- `GET /api/rl/replay/latest`
- `POST /api/rl/start`
- `POST /api/rl/pause`
- `POST /api/rl/reset`
- `POST /api/rl/config`
- `POST /api/rl/step` with `{"episodes": 10}`

This is still an approximate simulator, not a certified IFAB Laws of the Game engine. It intentionally abstracts referee positioning, assistant-referee mechanics, equipment inspection, VAR protocol, all misconduct subcategories, exact wall management, exact stoppage-time accounting, and competition-specific substitution rules.
