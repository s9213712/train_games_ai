const COLS = 10;
const ROWS = 20;
const COLORS = ["#000000", "#38bdf8", "#f59e0b", "#2563eb", "#eab308", "#22c55e", "#a855f7", "#ef4444"];
const PIECES = [
  [[[0, 0], [0, 1], [0, 2], [0, 3]]],
  [[[0, 0], [1, 0], [1, 1], [1, 2]]],
  [[[0, 2], [1, 0], [1, 1], [1, 2]]],
  [[[0, 0], [0, 1], [1, 0], [1, 1]]],
  [[[0, 1], [0, 2], [1, 0], [1, 1]]],
  [[[0, 1], [1, 0], [1, 1], [1, 2]]],
  [[[0, 0], [0, 1], [1, 1], [1, 2]]],
].map((base) => rotations(base[0]));

const el = {
  board: document.getElementById("board"),
  next: document.getElementById("nextPiece"),
  chart: document.getElementById("chart"),
  runState: document.getElementById("runState"),
  eventText: document.getElementById("eventText"),
  start: document.getElementById("startBtn"),
  pause: document.getElementById("pauseBtn"),
  reset: document.getElementById("resetBtn"),
  step: document.getElementById("stepBtn"),
  fast: document.getElementById("fastBtn"),
  speed: document.getElementById("speed"),
  epsilon: document.getElementById("epsilon"),
  mutation: document.getElementById("mutation"),
  fps: document.getElementById("fps"),
  applyWeights: document.getElementById("applyWeightsBtn"),
  episode: document.getElementById("episode"),
  pieces: document.getElementById("pieces"),
  lines: document.getElementById("lines"),
  score: document.getElementById("score"),
  reward: document.getElementById("reward"),
  bestReward: document.getElementById("bestReward"),
  weights: document.getElementById("weights"),
  error: document.getElementById("errorBox"),
  wLines: document.getElementById("wLines"),
  wHeight: document.getElementById("wHeight"),
  wHoles: document.getElementById("wHoles"),
  wBumpiness: document.getElementById("wBumpiness"),
  wWells: document.getElementById("wWells"),
};

const ctx = el.board.getContext("2d");
const nextCtx = el.next.getContext("2d");
const chartCtx = el.chart.getContext("2d");

let state;
let running = false;
let lastRender = 0;
let history = [];
let bestWeights = null;
const backendReplay = {
  frames: [],
  index: 0,
  mode: false,
  playing: false,
  lastTick: 0,
  latest: null,
};

function rotations(shape) {
  const out = [];
  let current = normalize(shape);
  for (let i = 0; i < 4; i += 1) {
    const key = current.map((p) => p.join(",")).join(";");
    if (!out.some((r) => r.map((p) => p.join(",")).join(";") === key)) out.push(current);
    current = normalize(current.map(([r, c]) => [c, -r]));
  }
  return out;
}

function normalize(shape) {
  const minR = Math.min(...shape.map(([r]) => r));
  const minC = Math.min(...shape.map(([, c]) => c));
  return shape.map(([r, c]) => [r - minR, c - minC]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

function emptyBoard() {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(0));
}

function resetAll() {
  state = {
    board: emptyBoard(),
    current: randPiece(),
    next: randPiece(),
    weights: { lines: 7.6, height: -0.58, holes: -4.8, bumpiness: -0.62, wells: -0.38 },
    candidateWeights: null,
    episode: 0,
    pieces: 0,
    lines: 0,
    score: 0,
    reward: 0,
    bestReward: -Infinity,
    baselineReward: -Infinity,
    alive: true,
  };
  bestWeights = { ...state.weights };
  history = [];
  fillWeightInputs();
  setEvent("reset");
  draw();
}

function randPiece() {
  const id = Math.floor(Math.random() * PIECES.length);
  return { id: id + 1, rotations: PIECES[id] };
}

function collide(board, shape, row, col) {
  return shape.some(([r, c]) => {
    const rr = row + r;
    const cc = col + c;
    return cc < 0 || cc >= COLS || rr >= ROWS || (rr >= 0 && board[rr][cc]);
  });
}

function dropRow(board, shape, col) {
  let row = -4;
  while (!collide(board, shape, row + 1, col)) row += 1;
  return row;
}

function place(board, shape, row, col, id) {
  const next = board.map((line) => line.slice());
  for (const [r, c] of shape) {
    const rr = row + r;
    if (rr < 0) return null;
    next[rr][col + c] = id;
  }
  return clearLines(next);
}

function clearLines(board) {
  const kept = board.filter((line) => line.some((cell) => !cell));
  const cleared = ROWS - kept.length;
  while (kept.length < ROWS) kept.unshift(Array(COLS).fill(0));
  return { board: kept, cleared };
}

function features(board, cleared) {
  const heights = [];
  let holes = 0;
  let wells = 0;
  for (let c = 0; c < COLS; c += 1) {
    let seen = false;
    let h = 0;
    for (let r = 0; r < ROWS; r += 1) {
      if (board[r][c]) {
        if (!seen) h = ROWS - r;
        seen = true;
      } else if (seen) {
        holes += 1;
      }
    }
    heights.push(h);
  }
  for (let c = 0; c < COLS; c += 1) {
    const left = c === 0 ? ROWS : heights[c - 1];
    const right = c === COLS - 1 ? ROWS : heights[c + 1];
    if (heights[c] < left && heights[c] < right) wells += Math.min(left, right) - heights[c];
  }
  let bumpiness = 0;
  for (let c = 0; c < COLS - 1; c += 1) bumpiness += Math.abs(heights[c] - heights[c + 1]);
  return { lines: cleared, height: heights.reduce((a, b) => a + b, 0), holes, bumpiness, wells };
}

function value(feat, weights) {
  return Object.entries(feat).reduce((sum, [key, val]) => sum + (weights[key] || 0) * val, 0);
}

function bestMove() {
  const moves = [];
  for (const shape of state.current.rotations) {
    const width = Math.max(...shape.map(([, c]) => c)) + 1;
    for (let col = 0; col <= COLS - width; col += 1) {
      const row = dropRow(state.board, shape, col);
      const result = place(state.board, shape, row, col, state.current.id);
      if (!result) continue;
      const feat = features(result.board, result.cleared);
      moves.push({ shape, col, row, result, feat, value: value(feat, state.weights) });
    }
  }
  if (!moves.length) return null;
  if (Math.random() < Number(el.epsilon.value) / 100) return moves[Math.floor(Math.random() * moves.length)];
  return moves.sort((a, b) => b.value - a.value)[0];
}

function stepGame() {
  if (!state.alive) nextEpisode();
  const move = bestMove();
  if (!move) {
    state.alive = false;
    nextEpisode();
    return;
  }
  state.board = move.result.board;
  state.current = state.next;
  state.next = randPiece();
  state.pieces += 1;
  state.lines += move.result.cleared;
  const lineScore = [0, 100, 300, 500, 800][move.result.cleared] || 0;
  state.score += lineScore + 2;
  const feat = features(state.board, move.result.cleared);
  const reward = lineScore + 1 - feat.holes * 2 - feat.height * 0.08 - feat.bumpiness * 0.15;
  state.reward += reward;
  if (state.board[0].some(Boolean)) {
    state.reward -= 260;
    state.alive = false;
    nextEpisode();
  }
}

function nextEpisode() {
  const episodeReward = Math.round(state.reward);
  history.push({ episode: state.episode, reward: episodeReward, score: state.score, lines: state.lines });
  if (episodeReward > state.bestReward) {
    state.bestReward = episodeReward;
    bestWeights = { ...state.weights };
    state.baselineReward = episodeReward;
    setEvent("new best");
  } else if (Math.random() < 0.22) {
    state.weights = { ...bestWeights };
  }
  mutateWeights();
  state.board = emptyBoard();
  state.current = randPiece();
  state.next = randPiece();
  state.episode += 1;
  state.pieces = 0;
  state.lines = 0;
  state.score = 0;
  state.reward = 0;
  state.alive = true;
  fillWeightInputs();
}

function mutateWeights() {
  const scale = Number(el.mutation.value) / 100;
  for (const key of Object.keys(state.weights)) {
    state.weights[key] += (Math.random() * 2 - 1) * scale;
  }
}

function fillWeightInputs() {
  el.wLines.value = state.weights.lines.toFixed(2);
  el.wHeight.value = state.weights.height.toFixed(2);
  el.wHoles.value = state.weights.holes.toFixed(2);
  el.wBumpiness.value = state.weights.bumpiness.toFixed(2);
  el.wWells.value = state.weights.wells.toFixed(2);
}

function applyWeights() {
  state.weights = {
    lines: Number(el.wLines.value),
    height: Number(el.wHeight.value),
    holes: Number(el.wHoles.value),
    bumpiness: Number(el.wBumpiness.value),
    wells: Number(el.wWells.value),
  };
  setEvent("weights applied");
  draw();
}

function drawBoard() {
  if (backendReplay.mode && backendReplay.frames.length) {
    drawReplayBoard(backendReplay.frames[backendReplay.index]);
    return;
  }
  const cell = el.board.width / COLS;
  ctx.fillStyle = "#080a0d";
  ctx.fillRect(0, 0, el.board.width, el.board.height);
  ctx.strokeStyle = "#1b222c";
  for (let r = 0; r <= ROWS; r += 1) {
    ctx.beginPath();
    ctx.moveTo(0, r * cell + 0.5);
    ctx.lineTo(el.board.width, r * cell + 0.5);
    ctx.stroke();
  }
  for (let c = 0; c <= COLS; c += 1) {
    ctx.beginPath();
    ctx.moveTo(c * cell + 0.5, 0);
    ctx.lineTo(c * cell + 0.5, el.board.height);
    ctx.stroke();
  }
  state.board.forEach((row, r) => row.forEach((id, c) => {
    if (!id) return;
    ctx.fillStyle = COLORS[id];
    ctx.fillRect(c * cell + 2, r * cell + 2, cell - 4, cell - 4);
  }));
}

function drawNext() {
  if (backendReplay.mode && backendReplay.frames.length) {
    drawReplayNext(backendReplay.frames[backendReplay.index]);
    return;
  }
  nextCtx.fillStyle = "#080a0d";
  nextCtx.fillRect(0, 0, el.next.width, el.next.height);
  const shape = state.next.rotations[0];
  const size = 24;
  const offsetX = 18;
  const offsetY = 30;
  nextCtx.fillStyle = COLORS[state.next.id];
  shape.forEach(([r, c]) => nextCtx.fillRect(offsetX + c * size, offsetY + r * size, size - 3, size - 3));
}

function drawChart() {
  chartCtx.fillStyle = "#0d1117";
  chartCtx.fillRect(0, 0, el.chart.width, el.chart.height);
  const data = history.slice(-80);
  if (!data.length) return;
  const max = Math.max(...data.map((d) => d.reward), 1);
  const min = Math.min(...data.map((d) => d.reward), -1);
  chartCtx.strokeStyle = "#22a06b";
  chartCtx.lineWidth = 2;
  chartCtx.beginPath();
  data.forEach((d, i) => {
    const x = 18 + (i / Math.max(1, data.length - 1)) * (el.chart.width - 36);
    const y = el.chart.height - 18 - ((d.reward - min) / Math.max(1, max - min)) * (el.chart.height - 36);
    if (i === 0) chartCtx.moveTo(x, y);
    else chartCtx.lineTo(x, y);
  });
  chartCtx.stroke();
  chartCtx.fillStyle = "#9aa8b8";
  chartCtx.font = "12px ui-monospace, monospace";
  chartCtx.fillText(`min ${Math.round(min)}  max ${Math.round(max)}`, 14, 18);
}

function drawWeights() {
  el.weights.innerHTML = Object.entries(state.weights).map(([key, val]) => (
    `<div><span>${key}</span><strong>${val.toFixed(2)}</strong></div>`
  )).join("");
}

function draw() {
  drawBoard();
  drawNext();
  drawChart();
  drawWeights();
  if (backendReplay.mode && backendReplay.frames.length) {
    drawReplayHud(backendReplay.frames[backendReplay.index]);
    return;
  }
  el.runState.textContent = running ? "running" : "paused";
  el.episode.textContent = state.episode;
  el.pieces.textContent = state.pieces;
  el.lines.textContent = state.lines;
  el.score.textContent = state.score;
  el.reward.textContent = Math.round(state.reward);
  el.bestReward.textContent = Number.isFinite(state.bestReward) ? Math.round(state.bestReward) : 0;
}

function setEvent(text) {
  el.eventText.textContent = text;
}

function loop(ts) {
  if (running) {
    backendReplay.mode = false;
    backendReplay.playing = false;
    const steps = Math.max(1, Number(el.speed.value));
    for (let i = 0; i < steps; i += 1) stepGame();
    const minDelay = 1000 / Number(el.fps.value);
    if (ts - lastRender > minDelay) {
      draw();
      lastRender = ts;
    }
  }
  if (backendReplay.playing && backendReplay.frames.length) {
    if (ts - backendReplay.lastTick > 180) {
      backendReplay.index = Math.min(backendReplay.frames.length - 1, backendReplay.index + 1);
      backendReplay.lastTick = ts;
      if (backendReplay.index >= backendReplay.frames.length - 1) backendReplay.playing = false;
      syncReplaySlider();
      draw();
    }
  }
  requestAnimationFrame(loop);
}

el.start.addEventListener("click", () => { running = true; setEvent("training"); draw(); });
el.pause.addEventListener("click", () => { running = false; setEvent("paused"); draw(); });
el.reset.addEventListener("click", () => { running = false; resetAll(); });
el.step.addEventListener("click", () => { stepGame(); draw(); });
el.fast.addEventListener("click", () => { for (let i = 0; i < 1000; i += 1) stepGame(); draw(); });
el.applyWeights.addEventListener("click", applyWeights);

resetAll();
requestAnimationFrame(loop);

const rl = {
  start: document.getElementById("rlStartBtn"),
  pause: document.getElementById("rlPauseBtn"),
  step: document.getElementById("rlStepBtn"),
  reset: document.getElementById("rlResetBtn"),
  rate: document.getElementById("rlRate"),
  epsilon: document.getElementById("rlEpsilon"),
  lookahead: document.getElementById("rlLookahead"),
  futureHold: document.getElementById("rlFutureHold"),
  replay: document.getElementById("rlReplayBtn"),
  replayPlay: document.getElementById("rlReplayPlayBtn"),
  replayPrev: document.getElementById("rlReplayPrevBtn"),
  replayNext: document.getElementById("rlReplayNextBtn"),
  replaySlider: document.getElementById("rlReplaySlider"),
  eval: document.getElementById("rlEvalBtn"),
  strongEval: document.getElementById("rlStrongEvalBtn"),
  box: document.getElementById("rlBox"),
};

async function rlApi(path, body = null) {
  const options = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

function rlConfig() {
  return {
    learning_rate: Number(rl.rate.value),
    epsilon: Number(rl.epsilon.value),
    lookahead_weight: Number(rl.lookahead.value),
    lookahead_include_hold: Boolean(rl.futureHold.checked),
  };
}

function renderRl(data) {
  if (!rl.box || !data) return;
  const latest = data.latest || {};
  const record = data.record || {};
  const evalData = data.evaluation || {};
  const guard = data.guard || {};
  const weights = data.weights || {};
  const terms = latest.reward_terms || {};
  const guardText = guard.episodes
    ? `${guard.accepted ? "accepted" : "rejected"} · ${Number(guard.baseline_score || 0).toFixed(0)} -> ${Number(guard.candidate_score || 0).toFixed(0)} · T ${Number(guard.baseline_tetrises || 0).toFixed(2)} -> ${Number(guard.candidate_tetrises || 0).toFixed(2)}`
    : "none";
  rl.box.innerHTML = [
    `<div>Mode <strong>${data.running ? "training" : "paused"}</strong> · Episode <strong>${data.episode || 0}</strong></div>`,
    `<div>Avg <strong>${Number(record.avg_score || 0).toFixed(1)}</strong> score · <strong>${Number(record.avg_lines || 0).toFixed(1)}</strong> lines · T <strong>${Number(record.avg_tetrises || 0).toFixed(2)}</strong></div>`,
    `<div>Best <strong>${record.best_score || 0}</strong> score · <strong>${record.best_lines || 0}</strong> lines</div>`,
    `<div>Latest <strong>${latest.score || 0}</strong> score · <strong>${latest.lines || 0}</strong> lines · T <strong>${latest.tetrises || 0}</strong> · H <strong>${latest.holds || 0}</strong> · ${latest.pieces || 0} pieces</div>`,
    `<div>Eval <strong>${Number(evalData.avg_score || 0).toFixed(1)}</strong> score · <strong>${Number(evalData.avg_lines || 0).toFixed(1)}</strong> lines · T <strong>${Number(evalData.avg_tetrises || 0).toFixed(2)}</strong> · H <strong>${Number(evalData.avg_holds || 0).toFixed(1)}</strong> · best <strong>${evalData.best_score || 0}</strong>${evalData.future_hold ? " · strong" : ""}</div>`,
    `<div>Config <strong>lookahead ${Number((data.config || {}).lookahead_weight || 0).toFixed(2)} · top ${Number((data.config || {}).lookahead_candidates || 0)} · future hold ${(data.config || {}).lookahead_include_hold ? "on" : "off"}</strong></div>`,
    `<div>Guard <strong>${guardText}</strong></div>`,
    `<div>Reward <strong>lines ${Number(terms.lines || 0).toFixed(2)} · well ${Number(terms.well || 0).toFixed(2)} · eroded ${Number(terms.eroded || 0).toFixed(2)} · shape ${Number(terms.shape || 0).toFixed(2)}</strong></div>`,
    `<div>Weights <strong>L ${Number(weights.lines || 0).toFixed(2)} H ${Number(weights.height || 0).toFixed(2)} Holes ${Number(weights.holes || 0).toFixed(2)} Well ${Number(weights.right_well || 0).toFixed(2)}</strong></div>`,
    `<div>${data.last_event || "ready"}</div>`,
  ].join("");
  if (data.config) {
    if (document.activeElement !== rl.rate) rl.rate.value = Number(data.config.learning_rate || 0.0005).toFixed(4);
    if (document.activeElement !== rl.epsilon) rl.epsilon.value = Number(data.config.epsilon || 0.015).toFixed(3);
    if (document.activeElement !== rl.lookahead) rl.lookahead.value = Number(data.config.lookahead_weight || 0.1).toFixed(2);
    if (document.activeElement !== rl.futureHold) rl.futureHold.checked = Boolean(data.config.lookahead_include_hold);
  }
}

function drawReplayBoard(frame) {
  const board = frame.board || [];
  const cell = el.board.width / COLS;
  ctx.fillStyle = "#080a0d";
  ctx.fillRect(0, 0, el.board.width, el.board.height);
  ctx.strokeStyle = "#1b222c";
  for (let r = 0; r <= ROWS; r += 1) {
    ctx.beginPath();
    ctx.moveTo(0, r * cell + 0.5);
    ctx.lineTo(el.board.width, r * cell + 0.5);
    ctx.stroke();
  }
  for (let c = 0; c <= COLS; c += 1) {
    ctx.beginPath();
    ctx.moveTo(c * cell + 0.5, 0);
    ctx.lineTo(c * cell + 0.5, el.board.height);
    ctx.stroke();
  }
  board.forEach((row, r) => row.forEach((id, c) => {
    if (!id) return;
    ctx.fillStyle = COLORS[id] || "#eef3f8";
    ctx.fillRect(c * cell + 2, r * cell + 2, cell - 4, cell - 4);
  }));
  ctx.fillStyle = "rgba(8,10,13,0.78)";
  ctx.fillRect(12, 12, 220, 70);
  ctx.fillStyle = "#eef3f8";
  ctx.font = "14px ui-sans-serif";
  ctx.fillText(`Backend replay ${backendReplay.index + 1}/${backendReplay.frames.length}`, 22, 34);
  ctx.fillText(`Score ${frame.score || 0} · Lines ${frame.lines || 0} · P ${frame.pieces || 0}`, 22, 56);
  ctx.fillStyle = "#9aa8b8";
  ctx.font = "11px ui-sans-serif";
  ctx.fillText(frame.last_event || "event", 22, 74);
}

function drawReplayNext(frame) {
  nextCtx.fillStyle = "#080a0d";
  nextCtx.fillRect(0, 0, el.next.width, el.next.height);
  nextCtx.fillStyle = "#9aa8b8";
  nextCtx.font = "12px ui-sans-serif";
  nextCtx.fillText(`next ${frame.next_piece ?? "-"}`, 18, 28);
  nextCtx.fillText(`hold ${frame.hold_piece ?? "-"}`, 18, 48);
}

function drawReplayHud(frame) {
  el.runState.textContent = backendReplay.playing ? "replay" : "replay paused";
  el.episode.textContent = backendReplay.latest?.episode ?? "-";
  el.pieces.textContent = frame.pieces || 0;
  el.lines.textContent = frame.lines || 0;
  el.score.textContent = frame.score || 0;
  el.reward.textContent = Number(frame.reward || 0).toFixed(1);
  el.bestReward.textContent = `${backendReplay.index + 1}/${backendReplay.frames.length}`;
  setEvent(`${frame.last_event || "backend replay"} · hold ${frame.hold_piece ?? "-"}`);
}

function syncReplaySlider() {
  if (!rl.replaySlider) return;
  rl.replaySlider.max = String(Math.max(0, backendReplay.frames.length - 1));
  rl.replaySlider.value = String(backendReplay.index);
}

async function loadBackendReplay() {
  const payload = await rlApi("/api/rl/replay/latest");
  backendReplay.frames = Array.isArray(payload.frames) ? payload.frames : [];
  backendReplay.latest = payload.latest || null;
  backendReplay.index = Math.max(0, backendReplay.frames.length - 1);
  backendReplay.mode = backendReplay.frames.length > 0;
  backendReplay.playing = false;
  running = false;
  syncReplaySlider();
  draw();
  return payload;
}

function stepReplay(delta) {
  if (!backendReplay.frames.length) return;
  backendReplay.mode = true;
  backendReplay.playing = false;
  backendReplay.index = Math.max(0, Math.min(backendReplay.frames.length - 1, backendReplay.index + delta));
  syncReplaySlider();
  draw();
}

async function refreshRl() {
  try {
    renderRl(await rlApi("/api/rl/state"));
  } catch (err) {
    if (rl.box) rl.box.innerHTML = `<div>${err.message}</div>`;
  }
}

if (rl.start) {
  rl.start.addEventListener("click", async () => renderRl(await rlApi("/api/rl/start", rlConfig())));
  rl.pause.addEventListener("click", async () => renderRl(await rlApi("/api/rl/pause", {})));
  rl.step.addEventListener("click", async () => {
    renderRl(await rlApi("/api/rl/step", { ...rlConfig(), episodes: 10, eval_episodes: 2 }));
    await loadBackendReplay();
  });
  rl.reset.addEventListener("click", async () => renderRl(await rlApi("/api/rl/reset", {})));
  rl.rate.addEventListener("change", async () => renderRl(await rlApi("/api/rl/config", rlConfig())));
  rl.epsilon.addEventListener("change", async () => renderRl(await rlApi("/api/rl/config", rlConfig())));
  rl.lookahead.addEventListener("change", async () => renderRl(await rlApi("/api/rl/config", rlConfig())));
  rl.futureHold.addEventListener("change", async () => renderRl(await rlApi("/api/rl/config", rlConfig())));
  rl.replay.addEventListener("click", async () => { await loadBackendReplay(); });
  rl.replayPlay.addEventListener("click", async () => {
    if (!backendReplay.frames.length) await loadBackendReplay();
    backendReplay.mode = backendReplay.frames.length > 0;
    backendReplay.playing = !backendReplay.playing;
    backendReplay.lastTick = performance.now();
    if (backendReplay.index >= backendReplay.frames.length - 1) backendReplay.index = 0;
    syncReplaySlider();
  });
  rl.replayPrev.addEventListener("click", () => stepReplay(-1));
  rl.replayNext.addEventListener("click", () => stepReplay(1));
  rl.replaySlider.addEventListener("input", () => {
    backendReplay.index = Number(rl.replaySlider.value || 0);
    backendReplay.mode = backendReplay.frames.length > 0;
    backendReplay.playing = false;
    draw();
  });
  rl.eval.addEventListener("click", async () => {
    const payload = await rlApi("/api/rl/evaluate", { episodes: 20 });
    renderRl(payload.state);
  });
  rl.strongEval.addEventListener("click", async () => {
    const payload = await rlApi("/api/rl/evaluate", { episodes: 2, lookahead_include_hold: true });
    renderRl(payload.state);
  });
  refreshRl();
  setInterval(refreshRl, 1200);
}
