const board = document.getElementById("board");
const ctx = board.getContext("2d");
const chart = document.getElementById("chart");
const chartCtx = chart.getContext("2d");

const el = {
  runState: document.getElementById("runState"),
  eventText: document.getElementById("eventText"),
  start: document.getElementById("startBtn"),
  pause: document.getElementById("pauseBtn"),
  reset: document.getElementById("resetBtn"),
  step: document.getElementById("stepBtn"),
  burst: document.getElementById("burstBtn"),
  game: document.getElementById("game"),
  ply: document.getElementById("ply"),
  turn: document.getElementById("turn"),
  reward: document.getElementById("reward"),
  best: document.getElementById("best"),
  teacherState: document.getElementById("teacherState"),
  winRate: document.getElementById("winRate"),
  matchRate: document.getElementById("matchRate"),
  samples: document.getElementById("samples"),
  trainLoss: document.getElementById("trainLoss"),
  strength: document.getElementById("strength"),
  updates: document.getElementById("updates"),
  lines: document.getElementById("lines"),
  moves: document.getElementById("moves"),
  weights: document.getElementById("weights"),
  guard: document.getElementById("guardBox"),
  stockfish: document.getElementById("stockfishBox"),
  error: document.getElementById("errorBox"),
  teacherDepth: document.getElementById("teacherDepth"),
  chunkMoves: document.getElementById("chunkMoves"),
  learningRate: document.getElementById("learningRate"),
  exploration: document.getElementById("exploration"),
  mutation: document.getElementById("mutation"),
  apply: document.getElementById("applyBtn"),
  wMaterial: document.getElementById("wMaterial"),
  wMobility: document.getElementById("wMobility"),
  wCenter: document.getElementById("wCenter"),
  wKing: document.getElementById("wKing"),
  wReply: document.getElementById("wReply"),
};

const PIECES = {
  P: "♙", N: "♘", B: "♗", R: "♖", Q: "♕", K: "♔",
  p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚",
};

let state = null;
let polling = false;

async function api(path, body = null) {
  const options = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

function boardArray(fen) {
  const rows = fen.split(" ")[0].split("/");
  return rows.map((row) => {
    const out = [];
    for (const ch of row) {
      if (/\d/.test(ch)) {
        for (let i = 0; i < Number(ch); i += 1) out.push("");
      } else {
        out.push(ch);
      }
    }
    return out;
  });
}

function drawBoard() {
  if (!state) return;
  const cells = boardArray(state.fen);
  const size = board.width / 8;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "48px serif";
  for (let r = 0; r < 8; r += 1) {
    for (let c = 0; c < 8; c += 1) {
      ctx.fillStyle = (r + c) % 2 ? "#7d8f69" : "#d7c6a1";
      ctx.fillRect(c * size, r * size, size, size);
      const piece = cells[r][c];
      if (piece) {
        ctx.fillStyle = piece === piece.toUpperCase() ? "#f8fafc" : "#111827";
        ctx.fillText(PIECES[piece], c * size + size / 2, r * size + size / 2 + 2);
      }
    }
  }
  const best = state.teacher?.best_move || "";
  if (best.length >= 4) {
    highlightSquare(best.slice(0, 2), "rgba(245, 158, 11, 0.45)");
    highlightSquare(best.slice(2, 4), "rgba(245, 158, 11, 0.45)");
  }
}

function highlightSquare(name, color) {
  const file = name.charCodeAt(0) - 97;
  const rank = 8 - Number(name[1]);
  const size = board.width / 8;
  if (file < 0 || file > 7 || rank < 0 || rank > 7) return;
  ctx.fillStyle = color;
  ctx.fillRect(file * size, rank * size, size, size);
}

function drawChart() {
  chartCtx.fillStyle = "#11161c";
  chartCtx.fillRect(0, 0, chart.width, chart.height);
  const data = (state?.history || []).slice(-80);
  if (!data.length) return;
  const max = Math.max(...data.map((d) => d.reward), 1);
  const min = Math.min(...data.map((d) => d.reward), -1);
  chartCtx.strokeStyle = "#6c8fdb";
  chartCtx.lineWidth = 2;
  chartCtx.beginPath();
  data.forEach((d, i) => {
    const x = 18 + (i / Math.max(1, data.length - 1)) * (chart.width - 36);
    const y = chart.height - 18 - ((d.reward - min) / Math.max(1, max - min)) * (chart.height - 36);
    if (i === 0) chartCtx.moveTo(x, y);
    else chartCtx.lineTo(x, y);
  });
  chartCtx.stroke();
  chartCtx.fillStyle = "#9aa8b8";
  chartCtx.font = "12px ui-monospace, monospace";
  chartCtx.fillText(`min ${Math.round(min)}  max ${Math.round(max)}`, 14, 18);
}

function fillInputs() {
  if (!state) return;
  const cfg = state.config;
  const w = state.weights;
  if (document.activeElement !== el.teacherDepth) el.teacherDepth.value = cfg.teacher_depth;
  if (document.activeElement !== el.chunkMoves) el.chunkMoves.value = cfg.chunk_moves;
  if (document.activeElement !== el.learningRate) el.learningRate.value = Number(cfg.learning_rate ?? 0).toFixed(3);
  if (document.activeElement !== el.exploration) el.exploration.value = Math.round(cfg.exploration * 100);
  if (document.activeElement !== el.mutation) el.mutation.value = Math.round(cfg.mutation * 100);
  for (const [element, key] of [[el.wMaterial, "material"], [el.wMobility, "mobility"], [el.wCenter, "center"], [el.wKing, "king_safety"], [el.wReply, "reply_safety"]]) {
    if (document.activeElement !== element) element.value = Number(w[key]).toFixed(2);
  }
}

function render() {
  if (!state) return;
  el.runState.textContent = state.running ? "training + verifying" : "paused";
  el.eventText.textContent = state.last_event || "ready";
  el.game.textContent = state.game;
  el.ply.textContent = state.ply;
  el.turn.textContent = state.turn;
  el.reward.textContent = Math.round(state.reward);
  el.best.textContent = Math.round(state.best_reward);
  el.teacherState.textContent = state.teacher?.source || "none";
  const learning = state.learning || {};
  el.winRate.textContent = `${Math.round((learning.win_rate || 0) * 100)}%`;
  el.matchRate.textContent = `${Math.round((learning.match_rate || 0) * 100)}%`;
  el.samples.textContent = learning.samples || 0;
  el.trainLoss.textContent = Number(learning.update_signal_ema || 0).toFixed(3);
  el.strength.textContent = `${learning.accepted_chunks || 0} / ${learning.rejected_chunks || 0}`;
  el.updates.textContent = learning.updates || 0;
  el.lines.innerHTML = (state.teacher?.lines || []).map((line, index) => (
    `<div>${index + 1}. ${line.move} <span>${line.score_cp ?? 0} cp</span> <span>${(line.pv || []).join(" ")}</span></div>`
  )).join("") || "<div>No teacher line yet</div>";
  el.moves.innerHTML = (state.moves || []).slice().reverse().map((move) => (
    `<div>${move.ply}. ${move.san} <span>${move.uci}</span> <span>teacher ${move.teacher || "-"}</span> <span>${move.match ? "teacher match" : `signal ${Number(move.loss || 0).toFixed(2)}`}</span>${move.fallback ? " <span>safety fallback</span>" : ""}</div>`
  )).join("") || "<div>No moves yet</div>";
  el.weights.innerHTML = Object.entries(state.weights).map(([key, val]) => (
    `<div><span>${key}</span><strong>${Number(val).toFixed(2)}</strong></div>`
  )).join("");
  const guard = state.guard || {};
  if (guard.baseline && guard.candidate) {
    const holdoutBefore = Number(guard.holdout_baseline?.avg_gap ?? 0).toFixed(1);
    const holdoutAfter = Number(guard.holdout_candidate?.avg_gap ?? 0).toFixed(1);
    const auditBefore = Number(guard.audit_baseline?.avg_gap ?? 0).toFixed(1);
    const auditAfter = Number(guard.audit_candidate?.avg_gap ?? 0).toFixed(1);
    el.guard.className = `stockfish ${guard.accepted ? "ok" : "warn"}`;
    el.guard.textContent = `${guard.accepted ? "ACCEPTED" : "REJECTED"}: guard ${Number(guard.baseline.avg_gap).toFixed(1)}→${Number(guard.candidate.avg_gap).toFixed(1)}, holdout ${holdoutBefore}→${holdoutAfter}, audit ${auditBefore}→${auditAfter}. ${guard.reason || ""}`;
  } else {
    el.guard.className = "stockfish warn";
    el.guard.textContent = guard.reason || "No candidate has completed all three checks yet.";
  }
  const sf = state.stockfish || {};
  el.stockfish.className = `stockfish ${sf.connected ? "ok" : "warn"}`;
  el.stockfish.textContent = sf.connected
    ? `Stockfish connected: ${sf.path}`
    : sf.available
      ? `Stockfish configured but not connected yet: ${sf.path}${sf.error ? ` (${sf.error})` : ""}`
      : "Stockfish not found. Set STOCKFISH_PATH to use it as the live teacher/opponent.";
  el.error.hidden = !state.last_error;
  el.error.textContent = state.last_error || "";
  drawBoard();
  drawChart();
  fillInputs();
}

async function refresh() {
  if (polling) return;
  polling = true;
  try {
    state = await api("/api/state");
    render();
  } catch (err) {
    el.error.hidden = false;
    el.error.textContent = err.message;
  } finally {
    polling = false;
  }
}

async function postAndRender(path, body = {}) {
  state = await api(path, body);
  render();
}

function readConfig() {
  return {
    teacher_depth: Number(el.teacherDepth.value),
    chunk_moves: Number(el.chunkMoves.value),
    learning_rate: Number(el.learningRate.value),
    exploration: Number(el.exploration.value) / 100,
    mutation: Number(el.mutation.value) / 100,
  };
}

el.start.addEventListener("click", () => postAndRender("/api/start", {}));
el.pause.addEventListener("click", () => postAndRender("/api/pause", {}));
el.reset.addEventListener("click", () => postAndRender("/api/reset", {}));
el.step.addEventListener("click", () => postAndRender("/api/step", { count: 1 }));
el.burst.addEventListener("click", () => postAndRender("/api/step", { count: 80 }));
el.apply.addEventListener("click", () => postAndRender("/api/config", readConfig()));
for (const input of [el.teacherDepth, el.chunkMoves, el.learningRate, el.exploration, el.mutation]) {
  input.addEventListener("change", () => postAndRender("/api/config", readConfig()));
}

refresh();
setInterval(refresh, 600);
