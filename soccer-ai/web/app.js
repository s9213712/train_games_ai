const W = 900;
const H = 540;
const FIELD = { left: 22, right: 878, top: 22, bottom: 518 };
const GOAL_W = 124;
const PENALTY_DEPTH = 118;
const PENALTY_W = 300;
const PLAYER_R = 8;
const BALL_R = 6;
const DT = 1 / 30;
const ROLE_LABELS = ["GK", "LB", "CB", "CB", "RB", "DM", "CM", "CM", "LW", "ST", "RW"];
const MIN_PLAYERS = 7;
const FORMATION = [
  { role: 0, depth: 34, y: 0 },
  { role: 1, depth: 150, y: -185 },
  { role: 2, depth: 150, y: -62 },
  { role: 3, depth: 150, y: 62 },
  { role: 4, depth: 150, y: 185 },
  { role: 5, depth: 270, y: 0 },
  { role: 6, depth: 385, y: -75 },
  { role: 7, depth: 385, y: 75 },
  { role: 8, depth: 535, y: -178 },
  { role: 9, depth: 595, y: 0 },
  { role: 10, depth: 535, y: 178 },
];
const ROLE_STATS = [
  { maxSpeed: 92, accel: 255, staminaDrain: 0.62, recovery: 0.038, kick: 0.86, pass: 0.72, tackle: 0.55, foul: 0.65, handling: 1.0 },
  { maxSpeed: 122, accel: 300, staminaDrain: 0.82, recovery: 0.033, kick: 0.82, pass: 0.72, tackle: 0.72, foul: 0.55 },
  { maxSpeed: 112, accel: 280, staminaDrain: 0.76, recovery: 0.035, kick: 0.78, pass: 0.7, tackle: 0.82, foul: 0.62 },
  { maxSpeed: 112, accel: 280, staminaDrain: 0.76, recovery: 0.035, kick: 0.78, pass: 0.7, tackle: 0.82, foul: 0.62 },
  { maxSpeed: 122, accel: 300, staminaDrain: 0.82, recovery: 0.033, kick: 0.82, pass: 0.72, tackle: 0.72, foul: 0.55 },
  { maxSpeed: 116, accel: 285, staminaDrain: 0.78, recovery: 0.036, kick: 0.8, pass: 0.78, tackle: 0.78, foul: 0.58 },
  { maxSpeed: 118, accel: 294, staminaDrain: 0.82, recovery: 0.034, kick: 0.84, pass: 0.86, tackle: 0.64, foul: 0.48 },
  { maxSpeed: 118, accel: 294, staminaDrain: 0.82, recovery: 0.034, kick: 0.84, pass: 0.86, tackle: 0.64, foul: 0.48 },
  { maxSpeed: 132, accel: 330, staminaDrain: 0.95, recovery: 0.031, kick: 0.9, pass: 0.78, tackle: 0.5, foul: 0.45 },
  { maxSpeed: 124, accel: 305, staminaDrain: 0.9, recovery: 0.032, kick: 1.0, pass: 0.74, tackle: 0.48, foul: 0.5 },
  { maxSpeed: 132, accel: 330, staminaDrain: 0.95, recovery: 0.031, kick: 0.9, pass: 0.78, tackle: 0.5, foul: 0.45 },
];

const el = {
  pitch: document.getElementById("pitch"),
  chart: document.getElementById("chart"),
  runState: document.getElementById("runState"),
  eventText: document.getElementById("eventText"),
  start: document.getElementById("startBtn"),
  pause: document.getElementById("pauseBtn"),
  reset: document.getElementById("resetBtn"),
  step: document.getElementById("stepBtn"),
  burst: document.getElementById("burstBtn"),
  speed: document.getElementById("speed"),
  epsilon: document.getElementById("epsilon"),
  mutation: document.getElementById("mutation"),
  matchSeconds: document.getElementById("matchSeconds"),
  match: document.getElementById("match"),
  clock: document.getElementById("clock"),
  blueScore: document.getElementById("blueScore"),
  redScore: document.getElementById("redScore"),
  reward: document.getElementById("reward"),
  best: document.getElementById("best"),
  winRate: document.getElementById("winRate"),
  record: document.getElementById("record"),
  shots: document.getElementById("shots"),
  possession: document.getElementById("possession"),
  fouls: document.getElementById("fouls"),
  cards: document.getElementById("cards"),
  stamina: document.getElementById("stamina"),
  advantage: document.getElementById("advantage"),
  restart: document.getElementById("restart"),
  rules: document.getElementById("rules"),
  weights: document.getElementById("weights"),
  applyWeights: document.getElementById("applyWeightsBtn"),
  wChase: document.getElementById("wChase"),
  wGoal: document.getElementById("wGoal"),
  wSpacing: document.getElementById("wSpacing"),
  wPress: document.getElementById("wPress"),
  wShoot: document.getElementById("wShoot"),
};

const ctx = el.pitch.getContext("2d");
const chartCtx = el.chart.getContext("2d");
let state;
let running = false;
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

function resetAll() {
  state = {
    match: 0,
    t: 0,
    half: 1,
    blueScore: 0,
    redScore: 0,
    reward: 0,
    best: -Infinity,
    record: { wins: 0, losses: 0, draws: 0 },
    shots: 0,
    passes: 0,
    restarts: 0,
    offsides: 0,
    handballs: 0,
    fouls: { blue: 0, red: 0 },
    cards: { blueY: 0, redY: 0, blueR: 0, redR: 0 },
    substitutions: { blue: 0, red: 0 },
    advantage: null,
    lastKick: null,
    lastRestart: "",
    bluePossessionTicks: 0,
    ticks: 0,
    lastTouch: "blue",
    phase: "kickoff",
    restart: null,
    weights: { chase: 2.1, goal: 1.25, spacing: 1.15, press: 1.0, shoot: 1.15 },
    players: [],
    ball: { x: W / 2, y: H / 2, vx: 0, vy: 0, owner: null },
  };
  bestWeights = { ...state.weights };
  history = [];
  resetPositions("blue", true);
  fillInputs();
  setEvent("kickoff");
  draw();
}

function player(team, spec, x, y) {
  const stats = ROLE_STATS[spec.role];
  return {
    team,
    x,
    y,
    vx: 0,
    vy: 0,
    role: spec.role,
    label: ROLE_LABELS[spec.role],
    cooldown: 0,
    stamina: 1,
    maxSpeed: stats.maxSpeed,
    accel: stats.accel,
    staminaDrain: stats.staminaDrain,
    recovery: stats.recovery,
    holdTimer: 0,
    kickPower: stats.kick,
    passSkill: stats.pass,
    tackleSkill: stats.tackle,
    foulRisk: stats.foul,
    handling: stats.handling || 0,
    yellow: 0,
    sentOff: false,
  };
}

function resetPositions(kickoffTeam = null, resetCards = false) {
  const oldCards = new Map((state.players || []).map((p) => [`${p.team}:${p.role}`, { yellow: p.yellow, sentOff: p.sentOff }]));
  state.players = [];
  for (const team of ["blue", "red"]) {
    for (const spec of FORMATION) {
      const pos = homePosition(team, spec.role);
      const p = player(team, spec, pos.x, pos.y);
      if (!resetCards) {
        const cards = oldCards.get(`${team}:${spec.role}`);
        if (cards) {
          p.yellow = cards.yellow;
          p.sentOff = cards.sentOff;
        }
      }
      state.players.push(p);
    }
  }
  state.ball = { x: W / 2, y: H / 2, vx: 0, vy: 0, owner: null };
  state.restart = null;
  state.advantage = null;
  if (kickoffTeam) {
    setRestart("kickoff", kickoffTeam, { x: W / 2, y: H / 2 }, { delay: 0.65 });
  }
}

function rand(min, max) {
  return min + Math.random() * (max - min);
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function norm(x, y) {
  const d = Math.hypot(x, y) || 1;
  return { x: x / d, y: y / d };
}

function otherTeam(team) {
  return team === "blue" ? "red" : "blue";
}

function attackSign(team) {
  return team === "blue" ? 1 : -1;
}

function ownGoalX(team) {
  return team === "blue" ? FIELD.left : FIELD.right;
}

function attackGoalX(team) {
  return team === "blue" ? FIELD.right : FIELD.left;
}

function activePlayers(team = null) {
  return state.players.filter((p) => !p.sentOff && (!team || p.team === team));
}

function teamPlayers(team) {
  return activePlayers(team);
}

function homePosition(team, role) {
  const spec = FORMATION[role];
  const sign = attackSign(team);
  const baseX = team === "blue" ? FIELD.left : FIELD.right;
  return {
    x: clamp(baseX + sign * spec.depth, FIELD.left + PLAYER_R, FIELD.right - PLAYER_R),
    y: clamp(H / 2 + spec.y, FIELD.top + PLAYER_R, FIELD.bottom - PLAYER_R),
  };
}

function nearest(team, point, excludeGoalkeeper = false) {
  return teamPlayers(team)
    .filter((p) => !excludeGoalkeeper || p.role !== 0)
    .sort((a, b) => dist(a, point) - dist(b, point))[0];
}

function inPenaltyArea(teamDefending, point) {
  const inY = Math.abs(point.y - H / 2) <= PENALTY_W / 2;
  if (teamDefending === "blue") return inY && point.x <= FIELD.left + PENALTY_DEPTH;
  return inY && point.x >= FIELD.right - PENALTY_DEPTH;
}

function staminaFactor(p) {
  return 0.58 + p.stamina * 0.42;
}

function ballSpeed() {
  return Math.hypot(state.ball.vx, state.ball.vy);
}

function averageStamina(team) {
  const players = teamPlayers(team);
  if (!players.length) return 0;
  return players.reduce((sum, p) => sum + p.stamina, 0) / players.length;
}

function openSpaceScore(p) {
  const nearestOpponent = nearest(otherTeam(p.team), p);
  const pressure = nearestOpponent ? dist(nearestOpponent, p) : 220;
  return clamp(pressure / 210, 0, 1);
}

function secondLastDefenderLine(attackingTeam) {
  const defenders = teamPlayers(otherTeam(attackingTeam)).map((p) => p.x).sort((a, b) => a - b);
  if (defenders.length < 2) return attackingTeam === "blue" ? FIELD.right : FIELD.left;
  return attackingTeam === "blue" ? defenders[defenders.length - 2] : defenders[1];
}

function isOffsideTarget(passer, receiver) {
  if (!passer || !receiver || passer.team !== receiver.team || receiver.role === 0) return false;
  if (state.phase === "throw-in" || state.phase === "corner" || state.phase === "goal kick") return false;
  const sign = attackSign(receiver.team);
  const inOppHalf = receiver.team === "blue" ? receiver.x > W / 2 : receiver.x < W / 2;
  if (!inOppHalf) return false;
  const aheadOfBall = sign * (receiver.x - state.ball.x) > 4;
  const line = secondLastDefenderLine(receiver.team);
  const beyondLine = receiver.team === "blue" ? receiver.x > line : receiver.x < line;
  return aheadOfBall && beyondLine;
}

function separationOffset(p) {
  let ox = 0;
  let oy = 0;
  for (const other of activePlayers()) {
    if (other === p) continue;
    const dx = p.x - other.x;
    const dy = p.y - other.y;
    const d = Math.hypot(dx, dy);
    const desired = other.team === p.team ? 33 : 21;
    if (d > 0 && d < desired) {
      const force = (desired - d) / desired;
      ox += (dx / d) * force * 48;
      oy += (dy / d) * force * 48;
    }
  }
  return { x: ox, y: oy };
}

function withSpacing(p, target, scale = 1) {
  if (p.role === 0 || state.restart) {
    return {
      x: clamp(target.x, FIELD.left + PLAYER_R, FIELD.right - PLAYER_R),
      y: clamp(target.y, FIELD.top + PLAYER_R, FIELD.bottom - PLAYER_R),
    };
  }
  const offset = separationOffset(p);
  return {
    x: clamp(target.x + offset.x * scale, FIELD.left + PLAYER_R, FIELD.right - PLAYER_R),
    y: clamp(target.y + offset.y * scale, FIELD.top + PLAYER_R, FIELD.bottom - PLAYER_R),
  };
}

function supportPosition(team, role) {
  const sign = attackSign(team);
  const ball = state.ball;
  const home = homePosition(team, role);
  if (role === 0) {
    return { x: ownGoalX(team) + sign * 30, y: clamp(ball.y, H / 2 - 92, H / 2 + 92) };
  }
  if (role >= 1 && role <= 4) {
    return {
      x: clamp(ball.x - sign * (role === 1 || role === 4 ? 110 : 145), FIELD.left + 50, FIELD.right - 50),
      y: clamp(home.y * 0.7 + ball.y * 0.3, FIELD.top + 35, FIELD.bottom - 35),
    };
  }
  if (role === 5) return { x: clamp(ball.x - sign * 96, FIELD.left + 55, FIELD.right - 55), y: clamp(ball.y, FIELD.top + 48, FIELD.bottom - 48) };
  if (role === 6 || role === 7) return { x: clamp(ball.x + sign * 40, FIELD.left + 55, FIELD.right - 55), y: clamp(home.y * 0.5 + ball.y * 0.5, FIELD.top + 45, FIELD.bottom - 45) };
  if (role === 9) return { x: clamp(ball.x + sign * 142, FIELD.left + 55, FIELD.right - 55), y: clamp(ball.y, FIELD.top + 55, FIELD.bottom - 55) };
  return { x: clamp(ball.x + sign * 122, FIELD.left + 55, FIELD.right - 55), y: clamp(home.y * 0.65 + ball.y * 0.35, FIELD.top + 32, FIELD.bottom - 32) };
}

function defensivePosition(team, role) {
  const sign = attackSign(team);
  const ball = state.ball;
  const home = homePosition(team, role);
  if (role === 0) return { x: ownGoalX(team) + sign * 26, y: clamp(ball.y, H / 2 - 86, H / 2 + 86) };
  if (role >= 1 && role <= 4) {
    return { x: clamp(ball.x - sign * 115, FIELD.left + 42, FIELD.right - 42), y: clamp(home.y * 0.58 + ball.y * 0.42, FIELD.top + 34, FIELD.bottom - 34) };
  }
  if (role === 5) return { x: clamp(ball.x - sign * 75, FIELD.left + 55, FIELD.right - 55), y: clamp(ball.y, FIELD.top + 42, FIELD.bottom - 42) };
  if (role === 6 || role === 7) return { x: clamp(ball.x - sign * 38, FIELD.left + 55, FIELD.right - 55), y: clamp(home.y * 0.5 + ball.y * 0.5, FIELD.top + 42, FIELD.bottom - 42) };
  return { x: clamp(home.x - sign * 35, FIELD.left + 55, FIELD.right - 55), y: clamp(home.y * 0.7 + ball.y * 0.3, FIELD.top + 36, FIELD.bottom - 36) };
}

function pressureGroup(team) {
  const ball = state.ball;
  return teamPlayers(team)
    .filter((p) => p.role !== 0)
    .sort((a, b) => dist(a, ball) - dist(b, ball));
}

function restartTarget(p) {
  const restart = state.restart;
  if (!restart) return null;
  if (p === state.ball.owner) return restart.spot;
  if (p.team === restart.team) {
    const sign = attackSign(p.team);
    const home = homePosition(p.team, p.role);
    if (restart.type === "corner") {
      return p.role >= 2 && p.role <= 9
        ? { x: attackGoalX(p.team) - sign * rand(45, 115), y: H / 2 + rand(-95, 95) }
        : home;
    }
    if (restart.type === "penalty") {
      return p.role === 9 ? { x: restart.spot.x - sign * 18, y: restart.spot.y } : home;
    }
    return {
      x: clamp(restart.spot.x + sign * (p.role >= 8 ? 145 : p.role >= 5 ? 72 : -85), FIELD.left + 40, FIELD.right - 40),
      y: clamp(home.y * 0.55 + restart.spot.y * 0.45, FIELD.top + 36, FIELD.bottom - 36),
    };
  }
  const d = dist(p, restart.spot);
  if (d < 72) {
    const away = norm(p.x - restart.spot.x, p.y - restart.spot.y);
    return { x: clamp(restart.spot.x + away.x * 72, FIELD.left + 28, FIELD.right - 28), y: clamp(restart.spot.y + away.y * 72, FIELD.top + 28, FIELD.bottom - 28) };
  }
  return defensivePosition(p.team, p.role);
}

function targetForTeam(p, team, weights) {
  if (p.sentOff) return { x: p.x, y: p.y };
  const restartMove = restartTarget(p);
  if (restartMove) return restartMove;

  const ball = state.ball;
  const owner = ball.owner;
  const pressers = pressureGroup(team);
  const pressIndex = pressers.indexOf(p);
  const sign = attackSign(team);

  if (owner === p) {
    const goal = { x: attackGoalX(team), y: H / 2 };
    return withSpacing(p, {
      x: clamp(p.x + sign * (66 + weights.goal * 17), FIELD.left + 38, FIELD.right - 38),
      y: p.y * 0.82 + goal.y * 0.18,
    }, 0.35);
  }
  if (owner?.team === team) return withSpacing(p, supportPosition(team, p.role), 0.72 + weights.spacing * 0.16);
  if (owner?.team === otherTeam(team)) {
    if (pressIndex >= 0 && pressIndex < 2) return withSpacing(p, { x: ball.x, y: ball.y }, 0.45 + weights.press * 0.08);
    if (pressIndex === 2) return withSpacing(p, { x: ball.x - sign * 58, y: clamp(ball.y + (p.y < ball.y ? -48 : 48), FIELD.top + 38, FIELD.bottom - 38) }, 0.65);
    return withSpacing(p, defensivePosition(team, p.role), 0.85);
  }
  if (pressIndex === 0) return withSpacing(p, { x: ball.x, y: ball.y }, 0.42 + weights.chase * 0.06);
  if (pressIndex === 1) return withSpacing(p, { x: ball.x - sign * 48, y: ball.y }, 0.7);
  const blended = p.role <= 5 ? defensivePosition(team, p.role) : supportPosition(team, p.role);
  return withSpacing(p, blended, 0.92);
}

function redWeights() {
  return { chase: 1.9, goal: 1.08, spacing: 1.05, press: 0.95, shoot: 1.03 };
}

function movePlayer(p, target, speed) {
  if (p.sentOff) return;
  const dx = target.x - p.x;
  const dy = target.y - p.y;
  const distance = Math.hypot(dx, dy);
  const dir = norm(dx, dy);
  const tacticalDemand = clamp(speed / 130, 0.45, 1.15);
  const maxSpeed = p.maxSpeed * staminaFactor(p) * tacticalDemand;
  const desiredSpeed = Math.min(maxSpeed, distance / Math.max(DT, 0.001));
  const currentSpeed = Math.hypot(p.vx, p.vy);
  const nextSpeed = currentSpeed + clamp(desiredSpeed - currentSpeed, -p.accel * DT, p.accel * DT);
  p.vx = dir.x * nextSpeed;
  p.vy = dir.y * nextSpeed;
  p.x = clamp(p.x + p.vx * DT, FIELD.left + PLAYER_R, FIELD.right - PLAYER_R);
  p.y = clamp(p.y + p.vy * DT, FIELD.top + PLAYER_R, FIELD.bottom - PLAYER_R);
  const effort = clamp(nextSpeed / Math.max(1, p.maxSpeed), 0, 1.25);
  if (effort > 0.56) p.stamina = clamp(p.stamina - p.staminaDrain * effort * effort * DT / 48, 0.16, 1);
  else p.stamina = clamp(p.stamina + p.recovery * DT, 0.16, 1);
  p.cooldown = Math.max(0, p.cooldown - DT);
}

function stepSim() {
  state.t += DT;
  state.ticks += 1;
  if (state.restart) state.restart.delay = Math.max(0, state.restart.delay - DT);
  updateAdvantage();
  maybeDroppedBall();
  maybeSubstitute("blue");
  maybeSubstitute("red");
  for (const p of activePlayers()) {
    const weights = p.team === "blue" ? state.weights : redWeights();
    const target = targetForTeam(p, p.team, weights);
    const roleBoost = p.role === 0 ? 0.76 : p.role >= 8 ? 1.06 : 1.0;
    movePlayer(p, target, (p.team === "blue" ? 112 : 108) * roleBoost);
  }
  updateBall();
  state.reward += rewardTick();
  if (teamPlayers("blue").length < MIN_PLAYERS || teamPlayers("red").length < MIN_PLAYERS) {
    finishMatch(teamPlayers("blue").length < MIN_PLAYERS ? "red" : "blue", "too few players");
    return;
  }
  const limit = Number(el.matchSeconds.value);
  if (state.t >= limit / 2 && state.half === 1) {
    state.half = 2;
    resetPositions(state.match % 2 ? "blue" : "red");
    setEvent("half-time kickoff");
  }
  if (state.t >= limit) finishMatch();
}

function maybeSubstitute(team) {
  if (state.substitutions[team] >= 5 || state.restart) return;
  const tired = teamPlayers(team)
    .filter((p) => p.role !== 0 && p.stamina < 0.28)
    .sort((a, b) => a.stamina - b.stamina)[0];
  if (!tired || Math.random() > 0.006) return;
  tired.stamina = 0.92;
  tired.cooldown = 0.4;
  tired.vx = 0;
  tired.vy = 0;
  state.substitutions[team] += 1;
  setEvent(`${team} substitution`);
}

function updateAdvantage() {
  if (!state.advantage) return;
  state.advantage.timer -= DT;
  const team = state.advantage.team;
  const sign = attackSign(team);
  const gained = sign * (state.ball.x - state.advantage.startX) > 62 || state.ball.owner?.team === team;
  if (gained && state.advantage.timer <= 0.9) {
    setEvent(`${team} advantage`);
    state.advantage = null;
    return;
  }
  if (state.advantage.timer <= 0) {
    const spot = state.advantage.spot;
    setRestart(state.advantage.restartType, team, spot, { delay: 0.65, indirect: state.advantage.indirect });
    setEvent(`${team} advantage recalled`);
    state.advantage = null;
  }
}

function maybeDroppedBall() {
  if (state.restart || !state.ball.owner || Math.random() > 0.00018) return;
  setRestart("dropped ball", state.ball.owner.team, { x: state.ball.x, y: state.ball.y }, { delay: 0.45, indirect: true });
  setEvent("dropped ball");
}

function updateBall() {
  const ball = state.ball;
  if (ball.owner?.sentOff) ball.owner = null;
  if (ball.owner) {
    const owner = ball.owner;
    const sign = attackSign(owner.team);
    ball.x = owner.x + sign * 13;
    ball.y = owner.y;
    state.lastTouch = owner.team;
    if (owner.team === "blue") state.bluePossessionTicks += 1;
    checkGoalkeeperHold(owner);
    if (!state.restart) maybeTackle(owner);
    maybeKick(owner);
  } else {
    ball.x += ball.vx * DT;
    ball.y += ball.vy * DT;
    ball.vx *= 0.988;
    ball.vy *= 0.988;
    collectLooseBall();
  }
  applyRestartRules();
}

function checkGoalkeeperHold(owner) {
  for (const p of activePlayers()) {
    if (p !== owner) p.holdTimer = 0;
  }
  if (owner.role !== 0 || !inPenaltyArea(owner.team, owner) || state.restart) {
    owner.holdTimer = 0;
    return;
  }
  owner.holdTimer += DT;
  if (owner.holdTimer < 1.2) return;
  const attacking = otherTeam(owner.team);
  const cornerX = attackGoalX(attacking) - attackSign(attacking) * 26;
  const cornerY = owner.y < H / 2 ? FIELD.top + 18 : FIELD.bottom - 18;
  setRestart("corner", attacking, { x: cornerX, y: cornerY }, { delay: 0.75 });
  owner.holdTimer = 0;
  setEvent(`${owner.team} goalkeeper eight seconds`);
}

function collectLooseBall() {
  for (const p of activePlayers().slice().sort((a, b) => dist(a, state.ball) - dist(b, state.ball))) {
    const keeperReach = p.role === 0 && inPenaltyArea(p.team, state.ball) ? 12 : 0;
    if (dist(p, state.ball) < PLAYER_R + BALL_R + 4 + keeperReach) {
      if (state.lastKick?.player === p && state.lastKick.kind !== "deflection") {
        callTechnicalOffence(p.team, otherTeam(p.team), { x: p.x, y: p.y }, "indirect free kick", "second touch");
        return;
      }
      if (p.role === 0 && inPenaltyArea(p.team, state.ball) && illegalGoalkeeperHandling(p)) {
        callTechnicalOffence(p.team, otherTeam(p.team), { x: p.x, y: p.y }, "indirect free kick", "back-pass");
        return;
      }
      if (p.role !== 0 && ballSpeed() > 150 && Math.random() < 0.012) {
        callHandball(p);
        return;
      }
      state.ball.owner = p;
      state.ball.vx = 0;
      state.ball.vy = 0;
      state.lastTouch = p.team;
      if (!state.lastKick || state.lastKick.player !== p) state.lastKick = null;
      setEvent(`${p.team} possession`);
      break;
    }
  }
}

function illegalGoalkeeperHandling(keeper) {
  const last = state.lastKick;
  if (!last || last.team !== keeper.team) return false;
  return last.kind === "pass" || last.kind === "throw-in";
}

function callHandball(player) {
  state.handballs += 1;
  const attacking = otherTeam(player.team);
  if (player.team === "blue") state.reward -= 24;
  else state.reward += 18;
  const spot = { x: clamp(state.ball.x, FIELD.left + 12, FIELD.right - 12), y: clamp(state.ball.y, FIELD.top + 12, FIELD.bottom - 12) };
  if (inPenaltyArea(player.team, spot)) {
    setRestart("penalty", attacking, { x: player.team === "blue" ? FIELD.left + 88 : FIELD.right - 88, y: H / 2 }, { delay: 0.9 });
  } else {
    setRestart("direct free kick", attacking, spot, { delay: 0.7 });
  }
  setEvent(`${player.team} handball`);
}

function callTechnicalOffence(offendingTeam, restartTeam, spot, type, reason) {
  const restartSpot = { x: clamp(spot.x, FIELD.left + 12, FIELD.right - 12), y: clamp(spot.y, FIELD.top + 12, FIELD.bottom - 12) };
  setRestart(type, restartTeam, restartSpot, { delay: 0.65, indirect: true });
  if (offendingTeam === "blue") state.reward -= 16;
  else state.reward += 12;
  setEvent(`${offendingTeam} ${reason}`);
}

function maybeTackle(owner) {
  const defender = nearest(otherTeam(owner.team), owner, false);
  if (!defender || defender.cooldown > 0 || dist(defender, owner) > 22) return;
  const support = teamPlayers(defender.team).filter((p) => p !== defender && dist(p, owner) < 64).length;
  const closingSpeed = Math.hypot(defender.vx - owner.vx, defender.vy - owner.vy);
  const reckless = dist(defender, owner) < 13 || closingSpeed > 126;
  const foulChance = clamp(0.02 + defender.foulRisk * 0.035 + (reckless ? 0.085 : 0) + Math.max(0, 1 - openSpaceScore(owner)) * 0.04, 0.018, 0.2);
  if (Math.random() < foulChance) {
    callFoul(defender, owner);
    return;
  }
  const tackleChance = clamp(0.05 + defender.tackleSkill * 0.07 + support * 0.03 - openSpaceScore(owner) * 0.04, 0.035, 0.19);
  if (Math.random() < tackleChance) {
    state.ball.owner = defender;
    defender.cooldown = 0.55;
    owner.cooldown = 0.45;
    state.reward += defender.team === "blue" ? 10 : -10;
    setEvent(`${defender.team} legal tackle`);
  }
}

function callFoul(defender, victim) {
  const attacking = victim.team;
  const defending = defender.team;
  state.fouls[defending] += 1;
  defender.cooldown = 1.1;
  victim.cooldown = 0.45;
  if (defending === "blue") state.reward -= 32;
  else state.reward += 24;

  const contactSpeed = Math.hypot(defender.vx - victim.vx, defender.vy - victim.vy);
  const excessive = contactSpeed > 185 || Math.random() < 0.018;
  const reckless = contactSpeed > 125 || Math.random() < 0.18;
  if (excessive) giveCard(defender, "red");
  else if (reckless) giveCard(defender, "yellow");
  const spot = { x: clamp(victim.x, FIELD.left + 12, FIELD.right - 12), y: clamp(victim.y, FIELD.top + 12, FIELD.bottom - 12) };
  const restartType = inPenaltyArea(defending, spot) ? "penalty" : "direct free kick";
  if (victim.team === state.ball.owner?.team && openSpaceScore(victim) > 0.58 && !inPenaltyArea(defending, spot)) {
    state.advantage = { team: attacking, spot, startX: state.ball.x, timer: 1.6, restartType, indirect: false };
    setEvent(`${attacking} advantage`);
    return;
  }
  if (inPenaltyArea(defending, spot)) {
    setRestart("penalty", attacking, { x: defending === "blue" ? FIELD.left + 88 : FIELD.right - 88, y: H / 2 }, { delay: 0.9 });
  } else {
    setRestart("direct free kick", attacking, spot, { delay: 0.75 });
  }
  setEvent(`${defending} foul`);
}

function giveCard(p, card) {
  if (card === "red" || p.yellow >= 1) {
    p.sentOff = true;
    if (p.team === "blue") state.cards.blueR += 1;
    else state.cards.redR += 1;
    if (state.ball.owner === p) state.ball.owner = null;
    setEvent(`${p.team} red card`);
    return;
  }
  p.yellow += 1;
  if (p.team === "blue") state.cards.blueY += 1;
  else state.cards.redY += 1;
}

function maybeKick(owner) {
  if (owner.cooldown > 0) return;
  if (state.restart) {
    if (state.restart.delay <= 0 && owner === state.ball.owner) setPieceKick(owner);
    return;
  }
  const sign = attackSign(owner.team);
  const weights = owner.team === "blue" ? state.weights : redWeights();
  const goal = { x: attackGoalX(owner.team), y: H / 2 };
  const dGoal = Math.hypot(goal.x - owner.x, goal.y - owner.y);
  const angleOk = Math.abs(owner.y - H / 2) < 150;
  const pressure = 1 - openSpaceScore(owner);
  const shootChance = clamp((weights.shoot * 0.105) + (1 - dGoal / W) * 0.72 - pressure * 0.14, 0.015, 0.56);

  if (angleOk && dGoal < 220 && Math.random() < shootChance) {
    shoot(owner, goal);
    return;
  }

  const passTarget = bestPassTarget(owner);
  const passChance = clamp(0.06 + pressure * 0.36 + (passTarget?.score || 0) * 0.24, 0.04, 0.58);
  if (passTarget && Math.random() < passChance) {
    if (isOffsideTarget(owner, passTarget.player)) {
      callOffside(owner.team, passTarget.player);
      return;
    }
    pass(owner, passTarget.player);
    return;
  }

  owner.cooldown = 0.18;
  state.ball.x = clamp(state.ball.x + sign * 2.5, FIELD.left, FIELD.right);
}

function setPieceKick(owner) {
  const restart = state.restart;
  if (!restart) return;
  if (restart.type === "penalty") {
    shoot(owner, { x: attackGoalX(owner.team), y: H / 2 });
    state.restart = null;
    state.phase = "open";
    return;
  }
  const target = bestPassTarget(owner, restart.type === "corner" ? 390 : 460);
  if (target) pass(owner, target.player, restart.type === "throw-in" ? 220 : restart.type === "corner" ? 360 : 330);
  else {
    const sign = attackSign(owner.team);
    releaseBall(owner, { x: owner.x + sign * 180, y: H / 2 }, 310);
  }
  state.restart = null;
  state.phase = "open";
}

function shoot(owner, goal) {
  const fatigueError = (1 - owner.stamina) * 34;
  const skillError = (1 - owner.kickPower) * 46;
  const dir = norm(goal.x - owner.x, goal.y + rand(-30 - fatigueError - skillError, 30 + fatigueError + skillError) - owner.y);
  releaseBall(owner, { x: owner.x + dir.x * 100, y: owner.y + dir.y * 100 }, rand(455, 610), "shot");
  owner.cooldown = 0.9;
  owner.stamina = clamp(owner.stamina - 0.018, 0.16, 1);
  state.phase = "open";
  if (owner.team === "blue") {
    state.shots += 1;
    state.reward += 16;
  }
  setEvent(`${owner.team} shot`);
}

function bestPassTarget(owner, maxDistance = 330) {
  const sign = attackSign(owner.team);
  const mates = teamPlayers(owner.team).filter((p) => p !== owner && p.role !== 0);
  let best = null;
  for (const mate of mates) {
    const forward = sign * (mate.x - owner.x);
    const distance = dist(owner, mate);
    if (distance < 36 || distance > maxDistance) continue;
    const offsidePenalty = isOffsideTarget(owner, mate) ? -1.2 : 0;
    const open = openSpaceScore(mate);
    const lane = 1 - clamp(Math.abs(mate.y - owner.y) / 270, 0, 1);
    const score = open * 0.48 + clamp(forward / 260, -0.28, 1) * 0.34 + lane * 0.18 + offsidePenalty;
    if (!best || score > best.score) best = { player: mate, score };
  }
  return best;
}

function pass(owner, mate, speed = 315) {
  const lead = attackSign(owner.team) * 20;
  const fatigueError = (1 - owner.stamina) * 26;
  const skillError = (1 - owner.passSkill) * 38;
  const target = { x: mate.x + lead + rand(-skillError, skillError), y: mate.y + rand(-fatigueError - skillError * 0.45, fatigueError + skillError * 0.45) };
  const kind = state.restart?.type === "throw-in" ? "throw-in" : "pass";
  releaseBall(owner, target, rand(speed * 0.88, speed * 1.08), kind);
  owner.cooldown = 0.6;
  owner.stamina = clamp(owner.stamina - 0.008, 0.16, 1);
  state.phase = "open";
  state.passes += 1;
  if (owner.team === "blue") state.reward += 6 + openSpaceScore(mate) * 8;
  setEvent(`${owner.team} pass`);
}

function releaseBall(owner, target, speed, kind = "kick") {
  const dir = norm(target.x - owner.x, target.y - owner.y);
  const power = (0.64 + owner.kickPower * 0.36) * staminaFactor(owner);
  state.ball.owner = null;
  state.ball.vx = dir.x * speed * power;
  state.ball.vy = dir.y * speed * power;
  state.lastTouch = owner.team;
  state.lastKick = { team: owner.team, player: owner, kind, restart: state.lastRestart };
}

function callOffside(attackingTeam, offender) {
  state.offsides += 1;
  if (attackingTeam === "blue") state.reward -= 28;
  else state.reward += 16;
  setRestart("offside free kick", otherTeam(attackingTeam), { x: offender.x, y: offender.y }, { delay: 0.7 });
  setEvent(`${attackingTeam} offside`);
}

function applyRestartRules() {
  const ball = state.ball;
  const inGoalY = Math.abs(ball.y - H / 2) < GOAL_W / 2;
  if (ball.x > FIELD.right + BALL_R && inGoalY) {
    scoreGoal("blue");
    return;
  }
  if (ball.x < FIELD.left - BALL_R && inGoalY) {
    scoreGoal("red");
    return;
  }
  if (ball.x < FIELD.left - BALL_R || ball.x > FIELD.right + BALL_R) {
    const defendingSide = ball.x < FIELD.left ? "blue" : "red";
    const restartTeam = state.lastTouch === defendingSide ? otherTeam(defendingSide) : defendingSide;
    const type = state.lastTouch === defendingSide ? "corner" : "goal kick";
    setRestart(type, restartTeam, {
      x: ball.x < FIELD.left ? FIELD.left + 26 : FIELD.right - 26,
      y: type === "corner" ? (ball.y < H / 2 ? FIELD.top + 18 : FIELD.bottom - 18) : H / 2,
    }, { delay: 0.75 });
    return;
  }
  if (ball.y < FIELD.top - BALL_R || ball.y > FIELD.bottom + BALL_R) {
    setRestart("throw-in", otherTeam(state.lastTouch), {
      x: clamp(ball.x, FIELD.left + 35, FIELD.right - 35),
      y: ball.y < FIELD.top ? FIELD.top + 10 : FIELD.bottom - 10,
    }, { delay: 0.55 });
  }
}

function setRestart(type, team, spot, options = {}) {
  state.restarts += 1;
  const taker = chooseRestartTaker(type, team, spot);
  state.restart = { type, team, spot, delay: options.delay ?? 0.6 };
  state.lastRestart = type;
  state.lastKick = null;
  state.ball = { x: spot.x, y: spot.y, vx: 0, vy: 0, owner: taker };
  if (taker) {
    taker.x = clamp(spot.x - attackSign(team) * (type === "corner" ? 0 : 10), FIELD.left + PLAYER_R, FIELD.right - PLAYER_R);
    taker.y = clamp(spot.y, FIELD.top + PLAYER_R, FIELD.bottom - PLAYER_R);
    taker.cooldown = options.delay ?? 0.6;
  }
  state.lastTouch = team;
  state.phase = type;
  state.reward += team === "blue" ? 2 : -2;
}

function chooseRestartTaker(type, team, spot) {
  if (type === "goal kick") return teamPlayers(team).find((p) => p.role === 0) || nearest(team, spot, false);
  if (type === "penalty") return teamPlayers(team).find((p) => p.role === 9) || nearest(team, spot, true);
  if (type === "corner" || type === "throw-in") {
    return teamPlayers(team)
      .filter((p) => p.role !== 0 && (p.role === 1 || p.role === 4 || p.role === 8 || p.role === 10))
      .sort((a, b) => dist(a, spot) - dist(b, spot))[0] || nearest(team, spot, true);
  }
  return nearest(team, spot, true) || nearest(team, spot, false);
}

function scoreGoal(team) {
  if (team === "blue") {
    state.blueScore += 1;
    state.reward += 520;
  } else {
    state.redScore += 1;
    state.reward -= 500;
  }
  setEvent(`${team} goal`);
  resetPositions(otherTeam(team));
}

function rewardTick() {
  const ball = state.ball;
  const progress = (ball.x - W / 2) / W;
  const nearestBlue = nearest("blue", ball);
  const pressure = nearestBlue ? 1 - clamp(dist(nearestBlue, ball) / 250, 0, 1) : 0;
  const possession = ball.owner?.team === "blue" ? 0.16 : ball.owner?.team === "red" ? -0.14 : 0;
  const shape = teamShapeReward("blue") - teamShapeReward("red") * 0.55;
  const discipline = state.fouls.blue * -0.002 + state.fouls.red * 0.0015 + state.offsides * -0.001;
  return progress * 0.18 + pressure * 0.02 + possession + shape * 0.01 + discipline;
}

function teamShapeReward(team) {
  const players = teamPlayers(team).filter((p) => p.role !== 0);
  let total = 0;
  let pairs = 0;
  for (let i = 0; i < players.length; i += 1) {
    for (let j = i + 1; j < players.length; j += 1) {
      const d = dist(players[i], players[j]);
      total += d > 38 && d < 245 ? 1 : -0.35;
      pairs += 1;
    }
  }
  return pairs ? total / pairs : 0;
}

function finishMatch(forcedWinner = "", reason = "full time") {
  let outcome = "draw";
  if (forcedWinner === "blue" && state.blueScore <= state.redScore) state.blueScore = state.redScore + 1;
  if (forcedWinner === "red" && state.redScore <= state.blueScore) state.redScore = state.blueScore + 1;
  if (state.blueScore > state.redScore) {
    state.record.wins += 1;
    outcome = "win";
  } else if (state.blueScore < state.redScore) {
    state.record.losses += 1;
    outcome = "loss";
  } else {
    state.record.draws += 1;
  }
  const result = Math.round(
    state.reward
    + (state.blueScore - state.redScore) * 260
    + state.passes * 0.8
    - state.fouls.blue * 10
    - state.offsides * 8
    - state.cards.blueR * 80
  );
  history.push({ match: state.match, reward: result, blue: state.blueScore, red: state.redScore, outcome, reason });
  if (result > state.best) {
    state.best = result;
    bestWeights = { ...state.weights };
    setEvent("new best");
  } else if (Math.random() < 0.25) {
    state.weights = { ...bestWeights };
  }
  mutateWeights();
  state.match += 1;
  state.t = 0;
  state.half = 1;
  state.blueScore = 0;
  state.redScore = 0;
  state.reward = 0;
  state.shots = 0;
  state.passes = 0;
  state.restarts = 0;
  state.offsides = 0;
  state.handballs = 0;
  state.fouls = { blue: 0, red: 0 };
  state.cards = { blueY: 0, redY: 0, blueR: 0, redR: 0 };
  state.substitutions = { blue: 0, red: 0 };
  state.advantage = null;
  state.lastKick = null;
  state.lastRestart = "";
  state.bluePossessionTicks = 0;
  state.ticks = 0;
  resetPositions(state.match % 2 ? "red" : "blue", true);
  fillInputs();
}

function mutateWeights() {
  const scale = Number(el.mutation.value) / 100;
  const eps = Number(el.epsilon.value) / 100;
  for (const key of Object.keys(state.weights)) {
    if (Math.random() < 0.36 + eps * 0.42) state.weights[key] = clamp(state.weights[key] + rand(-scale, scale), 0.1, 5);
  }
}

function fillInputs() {
  el.wChase.value = state.weights.chase.toFixed(2);
  el.wGoal.value = state.weights.goal.toFixed(2);
  el.wSpacing.value = state.weights.spacing.toFixed(2);
  el.wPress.value = state.weights.press.toFixed(2);
  el.wShoot.value = state.weights.shoot.toFixed(2);
}

function applyWeights() {
  state.weights = {
    chase: Number(el.wChase.value),
    goal: Number(el.wGoal.value),
    spacing: Number(el.wSpacing.value),
    press: Number(el.wPress.value),
    shoot: Number(el.wShoot.value),
  };
  setEvent("weights applied");
  draw();
}

function drawPitch() {
  ctx.fillStyle = "#0e5130";
  ctx.fillRect(0, 0, W, H);
  for (let i = 0; i < 12; i += 1) {
    ctx.fillStyle = i % 2 ? "rgba(255,255,255,0.025)" : "rgba(0,0,0,0.045)";
    ctx.fillRect((W / 12) * i, 0, W / 12, H);
  }
  ctx.strokeStyle = "rgba(238,244,247,0.74)";
  ctx.lineWidth = 3;
  ctx.strokeRect(FIELD.left, FIELD.top, FIELD.right - FIELD.left, FIELD.bottom - FIELD.top);
  ctx.beginPath();
  ctx.moveTo(W / 2, FIELD.top);
  ctx.lineTo(W / 2, FIELD.bottom);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(W / 2, H / 2, 62, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeRect(FIELD.left, H / 2 - GOAL_W / 2, 46, GOAL_W);
  ctx.strokeRect(FIELD.right - 46, H / 2 - GOAL_W / 2, 46, GOAL_W);
  ctx.strokeRect(FIELD.left, H / 2 - PENALTY_W / 2, PENALTY_DEPTH, PENALTY_W);
  ctx.strokeRect(FIELD.right - PENALTY_DEPTH, H / 2 - PENALTY_W / 2, PENALTY_DEPTH, PENALTY_W);
  ctx.strokeRect(FIELD.left, H / 2 - 92, 48, 184);
  ctx.strokeRect(FIELD.right - 48, H / 2 - 92, 48, 184);
  ctx.fillStyle = "rgba(238,244,247,0.18)";
  ctx.fillRect(0, H / 2 - GOAL_W / 2, FIELD.left, GOAL_W);
  ctx.fillRect(FIELD.right, H / 2 - GOAL_W / 2, W - FIELD.right, GOAL_W);
  if (state.restart) {
    ctx.fillStyle = "rgba(255, 224, 140, 0.25)";
    ctx.beginPath();
    ctx.arc(state.restart.spot.x, state.restart.spot.y, 72, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawPlayers() {
  for (const p of state.players) {
    if (p.sentOff) continue;
    ctx.fillStyle = p.team === "blue" ? "#38a3ff" : "#f45b69";
    ctx.beginPath();
    ctx.arc(p.x, p.y, PLAYER_R, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#071017";
    ctx.font = "7px ui-sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(p.label, p.x, p.y + 0.4);
    if (p.yellow) {
      ctx.fillStyle = "#f7c948";
      ctx.fillRect(p.x + 5, p.y - 10, 4, 6);
    }
  }
  const ball = state.ball;
  ctx.fillStyle = "#f7f3dc";
  ctx.beginPath();
  ctx.arc(ball.x, ball.y, BALL_R, 0, Math.PI * 2);
  ctx.fill();
}

function drawChart() {
  chartCtx.fillStyle = "#11161c";
  chartCtx.fillRect(0, 0, el.chart.width, el.chart.height);
  const data = history.slice(-80);
  if (!data.length) return;
  const max = Math.max(...data.map((d) => d.reward), 1);
  const min = Math.min(...data.map((d) => d.reward), -1);
  chartCtx.strokeStyle = "#38a3ff";
  chartCtx.lineWidth = 2;
  chartCtx.beginPath();
  data.forEach((d, i) => {
    const x = 18 + (i / Math.max(1, data.length - 1)) * (el.chart.width - 36);
    const y = el.chart.height - 18 - ((d.reward - min) / Math.max(1, max - min)) * (el.chart.height - 36);
    if (i === 0) chartCtx.moveTo(x, y);
    else chartCtx.lineTo(x, y);
  });
  chartCtx.stroke();
  chartCtx.fillStyle = "#9faab4";
  chartCtx.font = "12px ui-monospace, monospace";
  chartCtx.fillText(`min ${Math.round(min)}  max ${Math.round(max)}`, 14, 18);
}

function drawWeights() {
  el.weights.innerHTML = Object.entries(state.weights).map(([key, val]) => (
    `<div><span>${key}</span><strong>${val.toFixed(2)}</strong></div>`
  )).join("");
}

function drawHud() {
  if (backendReplay.mode && backendReplay.frames.length) {
    drawReplayHud(backendReplay.frames[backendReplay.index]);
    return;
  }
  el.runState.textContent = running ? "running" : "paused";
  el.match.textContent = state.match;
  const limit = Number(el.matchSeconds.value);
  const minute = Math.min(90, Math.floor((state.t / Math.max(1, limit)) * 90));
  el.clock.textContent = `${minute}' H${state.half}`;
  el.blueScore.textContent = state.blueScore;
  el.redScore.textContent = state.redScore;
  el.reward.textContent = Math.round(state.reward);
  el.best.textContent = Number.isFinite(state.best) ? Math.round(state.best) : 0;
  const played = state.record.wins + state.record.losses + state.record.draws;
  const winRate = played ? Math.round((state.record.wins / played) * 100) : 0;
  el.winRate.textContent = `${winRate}%`;
  el.record.textContent = `${state.record.wins}-${state.record.losses}-${state.record.draws}`;
  el.shots.textContent = `${state.shots} / O${state.offsides}`;
  const pct = state.ticks ? Math.round((state.bluePossessionTicks / state.ticks) * 100) : 0;
  el.possession.textContent = `${pct}%`;
  el.fouls.textContent = `${state.fouls.blue}-${state.fouls.red} H${state.handballs}`;
  el.cards.textContent = `B ${state.cards.blueY}Y/${state.cards.blueR}R S${state.substitutions.blue}`;
  el.stamina.textContent = `${Math.round(averageStamina("blue") * 100)}%`;
  el.advantage.textContent = state.advantage ? state.advantage.team : "off";
  el.restart.textContent = state.restart ? state.restart.type : state.phase;
  el.rules.textContent = "17 laws approx";
}

function draw() {
  if (backendReplay.mode && backendReplay.frames.length) {
    drawReplayFrame(backendReplay.frames[backendReplay.index]);
  } else {
    drawPitch();
    drawPlayers();
  }
  drawChart();
  drawWeights();
  drawHud();
}

function setEvent(text) {
  el.eventText.textContent = text;
}

function loop() {
  if (running) {
    backendReplay.mode = false;
    backendReplay.playing = false;
    const steps = Number(el.speed.value);
    for (let i = 0; i < steps; i += 1) stepSim();
  }
  if (backendReplay.playing && backendReplay.frames.length) {
    const now = performance.now();
    if (now - backendReplay.lastTick > 260) {
      backendReplay.index = Math.min(backendReplay.frames.length - 1, backendReplay.index + 1);
      backendReplay.lastTick = now;
      if (backendReplay.index >= backendReplay.frames.length - 1) backendReplay.playing = false;
      syncReplaySlider();
    }
  }
  draw();
  requestAnimationFrame(loop);
}

el.start.addEventListener("click", () => { running = true; setEvent("training"); });
el.pause.addEventListener("click", () => { running = false; setEvent("paused"); });
el.reset.addEventListener("click", () => { running = false; resetAll(); });
el.step.addEventListener("click", () => { stepSim(); draw(); });
el.burst.addEventListener("click", () => { for (let i = 0; i < 1500; i += 1) stepSim(); draw(); });
el.applyWeights.addEventListener("click", applyWeights);

resetAll();
requestAnimationFrame(loop);

const rl = {
  start: document.getElementById("rlStartBtn"),
  pause: document.getElementById("rlPauseBtn"),
  step: document.getElementById("rlStepBtn"),
  reset: document.getElementById("rlResetBtn"),
  rate: document.getElementById("rlRate"),
  selfPlay: document.getElementById("rlSelfPlay"),
  league: document.getElementById("rlLeague"),
  replay: document.getElementById("rlReplayBtn"),
  replayPlay: document.getElementById("rlReplayPlayBtn"),
  replayPrev: document.getElementById("rlReplayPrevBtn"),
  replayNext: document.getElementById("rlReplayNextBtn"),
  replaySlider: document.getElementById("rlReplaySlider"),
  eval: document.getElementById("rlEvalBtn"),
  evalOpponent: document.getElementById("rlEvalOpponent"),
  box: document.getElementById("rlBox"),
};

async function rlApi(path, body = null) {
  const options = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

function replayX(nx = 0) {
  return FIELD.left + ((Number(nx) + 1) / 2) * (FIELD.right - FIELD.left);
}

function replayY(ny = 0) {
  return FIELD.top + ((Number(ny) + 1) / 2) * (FIELD.bottom - FIELD.top);
}

function replayPlayerShape(team, frame) {
  const ball = frame.ball || { x: 0, y: 0 };
  const ballX = replayX(ball.x || 0);
  const ballY = replayY(ball.y || 0);
  const possession = frame.possession_side || "blue";
  const sign = team === "blue" ? 1 : -1;
  const baseX = team === "blue" ? FIELD.left : FIELD.right;
  const attacking = possession === team ? 1 : 0;
  return FORMATION.map((spec) => {
    const home = homePosition(team, spec.role);
    const push = attacking ? 0.18 : -0.08;
    const x = clamp(home.x + (ballX - W / 2) * (0.08 + attacking * 0.09) + sign * push * 90, FIELD.left + 14, FIELD.right - 14);
    const y = clamp(home.y + (ballY - H / 2) * (0.08 + attacking * 0.05), FIELD.top + 14, FIELD.bottom - 14);
    return { x, y, role: spec.role, label: ROLE_LABELS[spec.role] };
  });
}

function drawReplayFrame(frame) {
  const savedRestart = state.restart;
  state.restart = null;
  drawPitch();
  state.restart = savedRestart;
  const ball = frame.ball || { x: 0, y: 0 };
  const ballX = replayX(ball.x || 0);
  const ballY = replayY(ball.y || 0);
  for (const team of ["blue", "red"]) {
    const players = replayPlayerShape(team, frame);
    for (const p of players) {
      const owns = (frame.possession_side || "blue") === team;
      ctx.fillStyle = team === "blue" ? "#38a3ff" : "#f45b69";
      ctx.globalAlpha = owns ? 0.96 : 0.62;
      ctx.beginPath();
      ctx.arc(p.x, p.y, PLAYER_R, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#071017";
      ctx.font = "7px ui-sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(p.label, p.x, p.y + 0.4);
    }
  }
  ctx.strokeStyle = frame.possession_side === "blue" ? "rgba(56,163,255,0.8)" : "rgba(244,91,105,0.8)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(ballX, ballY, 22, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = "#f7f3dc";
  ctx.beginPath();
  ctx.arc(ballX, ballY, BALL_R, 0, Math.PI * 2);
  ctx.fill();
  const score = frame.score || { blue: 0, red: 0 };
  const terms = frame.reward_terms || {};
  ctx.fillStyle = "rgba(7,16,23,0.78)";
  ctx.fillRect(32, 32, 330, 86);
  ctx.fillStyle = "#eef4f7";
  ctx.font = "15px ui-sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(`Backend replay ${backendReplay.index + 1}/${backendReplay.frames.length}`, 46, 56);
  ctx.fillText(`${frame.minute || 0}'  ${score.blue || 0}-${score.red || 0}  ${frame.phase || "open"}`, 46, 80);
  ctx.fillStyle = "#9faab4";
  ctx.font = "12px ui-sans-serif";
  ctx.fillText(`${frame.last_event || "event"} · reward ${Number(frame.reward || 0).toFixed(2)}`, 46, 102);
  ctx.fillText(`xG ${frame.xg?.blue ?? 0}-${frame.xg?.red ?? 0} · score ${Number(terms.score || 0).toFixed(2)} · xG ${Number(terms.xg || 0).toFixed(2)}`, 46, 120);
}

function drawReplayHud(frame) {
  const score = frame.score || {};
  const xg = frame.xg || {};
  const discipline = frame.discipline || {};
  const restarts = frame.restarts || {};
  const match = frame.match || {};
  el.runState.textContent = backendReplay.playing ? "replay" : "replay paused";
  el.match.textContent = backendReplay.latest?.episode ?? "-";
  el.clock.textContent = `${frame.minute || 0}'`;
  el.blueScore.textContent = score.blue ?? 0;
  el.redScore.textContent = score.red ?? 0;
  el.reward.textContent = Number(frame.reward || 0).toFixed(1);
  el.best.textContent = `xG ${xg.blue ?? 0}-${xg.red ?? 0}`;
  el.winRate.textContent = `${backendReplay.frames.length}f`;
  el.record.textContent = `${backendReplay.index + 1}/${backendReplay.frames.length}`;
  el.shots.textContent = `${frame.shots?.blue ?? 0}-${frame.shots?.red ?? 0}`;
  el.possession.textContent = `${Math.round((frame.possession?.blue || 0) * 100)}%`;
  el.fouls.textContent = `${discipline.blue_fouls || 0}-${discipline.red_fouls || 0}`;
  el.cards.textContent = `Y ${discipline.blue_yellows || 0}-${discipline.red_yellows || 0} R ${discipline.blue_reds || 0}-${discipline.red_reds || 0}`;
  el.stamina.textContent = `${Math.round((frame.stamina?.blue || 0) * 100)}%`;
  el.advantage.textContent = frame.possession_side || "-";
  el.restart.textContent = `${frame.phase || "open"} +${match.added_time || 0}`;
  el.rules.textContent = `${restarts.blue_corners || 0}-${restarts.red_corners || 0} C`;
  setEvent(frame.last_event || "backend replay");
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
  backendReplay.index = clamp(backendReplay.index + delta, 0, backendReplay.frames.length - 1);
  syncReplaySlider();
  draw();
}

function rlConfig() {
  return {
    learning_rate: Number(rl.rate.value),
    self_play: Boolean(rl.selfPlay.checked),
    league_enabled: Boolean(rl.league?.checked),
  };
}

function renderRl(data) {
  if (!rl.box || !data) return;
  const latest = data.latest || {};
  const record = data.record || {};
  const score = latest.score || {};
  const xg = latest.xg || {};
  const discipline = latest.discipline || {};
  const restarts = latest.restarts || {};
  const match = latest.match || {};
  const goalkeeping = latest.goalkeeping || {};
  const rewards = latest.reward_terms || {};
  const league = data.league || {};
  const evaluation = data.evaluation || {};
  const opponent = latest.opponent || league.last_opponent || {};
  const evalRecord = evaluation.record || {};
  const top = (latest.top_actions || []).map((row) => `${row.action} ${Math.round(row.prob * 100)}%`).join(" · ");
  const rewardLine = ["score", "xg", "shot_quality", "territory", "defense", "discipline", "result"]
    .map((key) => `${key.replace("_", " ")} ${Number(rewards[key] || 0).toFixed(2)}`)
    .join(" · ");
  rl.box.innerHTML = [
    `<div>Mode <strong>${data.running ? "training" : "paused"}</strong> · Episode <strong>${data.episode || 0}</strong></div>`,
    `<div>Record <strong>${record.wins || 0}-${record.losses || 0}-${record.draws || 0}</strong> · Win <strong>${Math.round((record.win_rate || 0) * 100)}%</strong></div>`,
    `<div>League Elo <strong>${Math.round(league.elo || 1000)}</strong> · Pool <strong>${league.pool_size || 0}</strong> · Opp <strong>${opponent.name || "-"}</strong></div>`,
    `<div>Eval <strong>${evalRecord.wins || 0}-${evalRecord.losses || 0}-${evalRecord.draws || 0}</strong> · GD <strong>${Number(evaluation.avg_goal_diff || 0).toFixed(2)}</strong> · xGD <strong>${Number(evaluation.avg_xg_diff || 0).toFixed(2)}</strong></div>`,
    `<div>Latest <strong>${score.blue ?? 0}-${score.red ?? 0}</strong> · xG <strong>${xg.blue ?? 0}-${xg.red ?? 0}</strong> · +<strong>${match.added_time || 0}</strong></div>`,
    `<div>Set pieces <strong>C ${restarts.blue_corners || 0}-${restarts.red_corners || 0}</strong> · FK <strong>${restarts.blue_free_kicks || 0}-${restarts.red_free_kicks || 0}</strong> · GK <strong>${restarts.blue_goal_kicks || 0}-${restarts.red_goal_kicks || 0}</strong></div>`,
    `<div>Discipline <strong>F ${discipline.blue_fouls || 0}-${discipline.red_fouls || 0}</strong> · O <strong>${discipline.blue_offsides || 0}-${discipline.red_offsides || 0}</strong> · YC <strong>${discipline.blue_yellows || 0}-${discipline.red_yellows || 0}</strong></div>`,
    `<div>Players <strong>${match.active_players?.blue || 11}-${match.active_players?.red || 11}</strong> · Subs <strong>${match.substitutions?.blue || 0}-${match.substitutions?.red || 0}</strong> · Saves <strong>${goalkeeping.blue_saves || 0}-${goalkeeping.red_saves || 0}</strong></div>`,
    `<div>Reward <strong>${rewardLine}</strong></div>`,
    `<div>Policy <strong>${top || "-"}</strong></div>`,
    `<div>${data.last_event || "ready"}</div>`,
  ].join("");
  if (data.config) {
    if (document.activeElement !== rl.rate) rl.rate.value = Number(data.config.learning_rate || 0.012).toFixed(3);
    if (document.activeElement !== rl.selfPlay) rl.selfPlay.checked = Boolean(data.config.self_play);
    if (rl.league && document.activeElement !== rl.league) rl.league.checked = Boolean(data.config.league_enabled);
  }
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
    renderRl(await rlApi("/api/rl/step", { episodes: 10 }));
    await loadBackendReplay();
  });
  rl.reset.addEventListener("click", async () => renderRl(await rlApi("/api/rl/reset", {})));
  rl.rate.addEventListener("change", async () => renderRl(await rlApi("/api/rl/config", rlConfig())));
  rl.selfPlay.addEventListener("change", async () => renderRl(await rlApi("/api/rl/config", rlConfig())));
  rl.league.addEventListener("change", async () => renderRl(await rlApi("/api/rl/config", rlConfig())));
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
    const payload = await rlApi("/api/rl/evaluate", { episodes: 20, opponent: rl.evalOpponent.value || "mixed" });
    renderRl(payload.state);
  });
  refreshRl();
  setInterval(refreshRl, 1200);
}
