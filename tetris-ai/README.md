# Tetris AI Trainer

Browser-based Tetris training dashboard inspired by the local `snake-ai` project shape.

## Run

```bash
./run_web_dashboard.sh
```

Open the printed local URL. The dashboard now has two layers:

- Browser preview: the original lightweight mutation-based feature search.
- Backend RL: a Python afterstate Tetris environment with temporal-difference value learning.

The backend enumerates every legal rotation/column placement, includes modern hold-piece actions, scores each action as immediate reward plus the resulting afterstate value, adds a bounded known-next-piece lookahead for the best candidate moves, and updates a linear value function from TD targets. This is closer to standard Tetris RL practice than mutating weights only inside the page: the learned policy is saved, metrics are written, and evaluation can run without changing the model.

## Backend RL

Run:

```bash
python3 server.py --host 127.0.0.1 --port 7871
```

Dashboard controls:

- `RL Start` / `RL Pause`: background backend training.
- `Guarded 10`: manual backend training batch; trains a candidate policy and accepts it only if frozen evaluation stays near or above the previous policy.
- `Evaluate`: frozen-policy evaluation; does not update weights.
- `Load Replay` / replay controls: inspect the latest backend episode on the board canvas.

Runtime outputs:

- `runtime/tetris_policy.json`: TD value weights and config.
- `runtime/tetris_policy.best.json`: best guarded policy accepted by fixed-seed objective.
- `runtime/tetris_policy.best_score.json`: best single-episode score policy.
- `runtime/training_metrics.jsonl`: score, lines, reward, TD error, features, and weights per episode.
- `runtime/latest_replay.json`: latest training episode frames.
- `runtime/latest_evaluation.json`: latest frozen evaluation summary.

The feature vector includes cleared lines, aggregate height, max height, holes, covered holes, bumpiness, wells, right-side well depth, eroded cells, ready-to-tetris rows, row/column transitions, current piece, and next piece. Hold is modeled at the action layer so older checkpoints remain usable: the policy can choose a normal placement or a hold-then-place candidate without changing the saved value-vector dimension. The trainer uses a slowly updated target value function for bootstrap estimates so TD updates do not chase their own rapidly moving predictions. Greedy action selection also includes the immediate reward, so a four-line clear is not undervalued just because the post-clear board no longer contains a ready tetris well. The default lookahead is intentionally bounded (`0.10`, top 4 candidates) so it can use the visible next piece as a tie-breaker without making each training episode too slow.

The default TD rate is intentionally conservative (`0.0005`). Earlier higher-rate runs learned basic survival but tended to drift into short-term line clears with few tetrises, and later caused value drift after the 15-feature upgrade. The trainer now keeps an elite policy anchor: if a training episode falls far below the historical best, the live value function is nudged back toward the best saved policy. The current shaping gives credit to safe right-side well preparation and four-line clears, while the checkpoint loader remaps older 11-feature policies into the newer 15-feature representation. The dashboard reports average tetrises separately because score can improve without learning the four-line-clear strategy.

Manual batch training is guarded. The trainer snapshots the current policy, trains a candidate policy without writing it immediately, evaluates baseline and candidate with fixed seeds, and rolls back if the candidate objective falls below the acceptance threshold. This keeps exploratory batches from overwriting a good policy after a few unlucky or unstable TD updates. The dashboard uses smaller guarded batches now because hold-enabled games survive much longer than the earlier one-piece-only environment.

For a stricter multi-seed benchmark, run:

```bash
python3 evaluate_policy.py --episodes 100 --seeds 1000:1099 --future-hold
```

The CLI prints JSON with average, median, p10/p90 score, average tetrises, top-out rate, piece-cap hit rate, and per-seed rows. It is intended for release/best-policy checks rather than the fast interactive dashboard guard.

The hold-enabled evaluator avoids duplicate terminal-check move enumeration and caches duplicate future-board lookahead calls inside each action selection. This keeps the stronger hold policy usable in the interactive dashboard without changing the policy semantics.

`Future Hold` is an optional stronger lookahead mode. When enabled, the second lookahead layer also considers holding on the next turn. It improves some frozen evaluations substantially, but it is slower, so the dashboard leaves it off by default for interactive guarded training. `Strong Eval` runs a short Future Hold evaluation without changing the training config.

Replay now keeps the full backend episode up to the environment piece limit instead of only the final tail. This is important for hold-enabled runs because strong games often reach 900 pieces; the replay slider can now show the opening, middle game, and endgame of the same evaluation.
