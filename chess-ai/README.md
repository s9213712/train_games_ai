# Chess AI Trainer

Interactive chess training dashboard using `python-chess` and an optional Stockfish UCI engine as opponent and mentor.

The student does not train by fitting Stockfish's move output. Stockfish is used
as a live opponent, line viewer, and diagnostic benchmark. The student policy is
updated from reinforcement signals: material/position change after its own moves,
the opponent reply, and the final game result. The dashboard still reports
Stockfish agreement as a diagnostic, but that agreement is not used as the
training target.

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
