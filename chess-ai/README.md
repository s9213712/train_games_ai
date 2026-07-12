# Chess AI Trainer

Interactive chess training dashboard using `python-chess` and an optional Stockfish UCI engine as opponent and mentor.

The student uses both reward-shaped update signals and teacher guidance. Material and
position changes, opponent replies, and final game results update the heuristic
weights. When the student chooses a different move from the teacher, a small
ranking update also moves the weights toward the teacher move's feature vector
and away from the chosen move. The dashboard reports teacher agreement and
teacher-update counts so the teacher-guided part of training is explicit.
Training is intentionally described as a five-feature heuristic ranker, not a
neural network or a TD-value model. Candidate chunks are checked on
White-to-move guard positions that match the side controlled by the student.
They are retained only when chosen-move regret improves on both the guard and
the separate fixed holdout, while a third promotion-audit set does not regress.
These three sets are fixed validation sets used repeatedly for candidate
selection; they are not described as statistically independent test data.
Behavior-identical candidates are rolled back and reported as rejected rather
than counted as training progress.
The fifth feature measures the position after the opponent's best deterministic
reply; its coefficient starts at zero and is learned from teacher-ranking
updates, giving the student enough capacity to improve beyond a purely
one-move heuristic.

At startup, saved checkpoints are benchmarked against the built-in default
policy on all three sets. A checkpoint that performs worse is rejected,
preventing a stale or drifted model from silently replacing a stronger
baseline. Candidate chunks are also transaction-like: intermediate weights are
never written to a checkpoint, and shutdown/configuration waits for the chunk
to either pass or roll back. A fourth, disjoint position set is never consulted
by the trainer and is reserved for the repository-level offline audit. Current
checkpoint schema, protocol, accepted-guard evidence, and a policy fingerprint
must all agree before a saved policy can load. The displayed student weights
are read-only so a manual edit cannot inherit old training evidence.

## Run

```bash
./run_web_dashboard.sh
```

Open the printed local URL.

## Stockfish

The dashboard looks for Stockfish in this order:

1. `STOCKFISH_PATH`
2. `HTML_LEARNING_CHESS_STOCKFISH_PATH`
3. `stockfish` on `PATH`

Example:

```bash
STOCKFISH_PATH=/path/to/stockfish ./run_web_dashboard.sh
```

If Stockfish is not available, the page still runs with a deterministic material/mobility fallback teacher and clearly reports that Stockfish is offline.
