# Chess AI Trainer

Interactive chess training dashboard using `python-chess` and an optional Stockfish UCI engine as opponent and mentor.

The student uses both reinforcement signals and teacher guidance. Material and
position changes, opponent replies, and final game results update the heuristic
weights. When the student chooses a different move from the teacher, a small
ranking update also moves the weights toward the teacher move's feature vector
and away from the chosen move. The dashboard reports teacher agreement and
teacher-update counts so the teacher-guided part of training is explicit.
Training chunks are guarded with a fixed FEN probe against the deterministic
fallback teacher: if the candidate weights do not reduce the average teacher
score gap, the weights are rolled back.

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
