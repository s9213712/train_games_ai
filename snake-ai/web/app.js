const fields = {
  agent: document.getElementById("agent"),
  strategy: document.getElementById("strategy"),
  device: document.getElementById("device"),
  boardPreset: document.getElementById("boardPreset"),
  boardSize: document.getElementById("boardSize"),
  seed: document.getElementById("seed"),
  learningRate: document.getElementById("learningRate"),
  gamma: document.getElementById("gamma"),
  entCoef: document.getElementById("entCoef"),
  clipRange: document.getElementById("clipRange"),
  foodTimePenalty: document.getElementById("foodTimePenalty"),
  foodStepLimitMultiplier: document.getElementById("foodStepLimitMultiplier"),
  foodRewardBonus: document.getElementById("foodRewardBonus"),
  distanceRewardScale: document.getElementById("distanceRewardScale"),
  loopPenalty: document.getElementById("loopPenalty"),
  loopWindow: document.getElementById("loopWindow"),
  oscillationPenalty: document.getElementById("oscillationPenalty"),
  oscillationWindow: document.getElementById("oscillationWindow"),
  chunkTimesteps: document.getElementById("chunkTimesteps"),
  previewSteps: document.getElementById("previewSteps"),
  completeEpisodePreview: document.getElementById("completeEpisodePreview"),
  deterministicPreview: document.getElementById("deterministicPreview"),
  trainingEnabled: document.getElementById("trainingEnabled"),
  numEnvs: document.getElementById("numEnvs"),
  nSteps: document.getElementById("nSteps"),
  batchSize: document.getElementById("batchSize"),
  nEpochs: document.getElementById("nEpochs"),
  cnnChannels: document.getElementById("cnnChannels"),
  cnnKernelSizes: document.getElementById("cnnKernelSizes"),
  cnnStrides: document.getElementById("cnnStrides"),
  cnnFeaturesDim: document.getElementById("cnnFeaturesDim"),
  cnnChannelFirst: document.getElementById("cnnChannelFirst"),
};

const board = document.getElementById("board");
const boardCtx = board.getContext("2d");
const chart = document.getElementById("chart");
const chartCtx = chart.getContext("2d");
const stateEl = document.getElementById("runState");
const eventEl = document.getElementById("eventText");
const trainedStepsEl = document.getElementById("trainedSteps");
const bestScoreEl = document.getElementById("bestScore");
const bestStepsEl = document.getElementById("bestSteps");
const scoreEl = document.getElementById("score");
const lengthEl = document.getElementById("length");
const foodCountEl = document.getElementById("foodCount");
const stepsPerFoodEl = document.getElementById("stepsPerFood");
const loopRevisitsEl = document.getElementById("loopRevisits");
const oscillationsEl = document.getElementById("oscillations");
const rewardEl = document.getElementById("reward");
const fpsEl = document.getElementById("fps");
const errorBox = document.getElementById("errorBox");
const playbackSpeed = document.getElementById("playbackSpeed");
const playbackLabel = document.getElementById("playbackLabel");
const downloadModelBtn = document.getElementById("downloadModelBtn");
const importModelBtn = document.getElementById("importModelBtn");
const moveDeviceBtn = document.getElementById("moveDeviceBtn");
const modelFile = document.getElementById("modelFile");
const modelStatus = document.getElementById("modelStatus");
const cnnSummary = document.getElementById("cnnSummary");
const deviceStatus = document.getElementById("deviceStatus");

let frames = [];
let frameVersion = -1;
let frameIndex = 0;
let pendingFrames = null;
let pendingFrameVersion = -1;
let frameDelayMs = 250;
let lastFrameAt = 0;
const dirtyFields = new Set();

function updatePlaybackSpeed() {
  const fps = Number(playbackSpeed.value);
  frameDelayMs = 1000 / Math.max(1, fps);
  playbackLabel.textContent = `${fps} fps`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

function filenameFromDisposition(disposition) {
  const match = /filename="?([^"]+)"?/i.exec(disposition || "");
  return match ? match[1] : "snake-ai-model.snakeai.zip";
}

function resetPlaybackFrames() {
  frames = [];
  frameIndex = 0;
  pendingFrames = null;
  pendingFrameVersion = -1;
  frameVersion = -1;
}

function readConfig() {
  return {
    agent: fields.agent.value,
    strategy: fields.strategy.value,
    device: fields.device.value,
    board_size: Number(fields.boardSize.value),
    seed: Number(fields.seed.value),
    learning_rate: Number(fields.learningRate.value),
    strategy: fields.strategy.value,
    gamma: Number(fields.gamma.value),
    ent_coef: Number(fields.entCoef.value),
    clip_range: Number(fields.clipRange.value),
    food_time_penalty: Number(fields.foodTimePenalty.value),
    food_step_limit_multiplier: Number(fields.foodStepLimitMultiplier.value),
    food_reward_bonus: Number(fields.foodRewardBonus.value),
    distance_reward_scale: Number(fields.distanceRewardScale.value),
    loop_penalty: Number(fields.loopPenalty.value),
    loop_window: Number(fields.loopWindow.value),
    oscillation_penalty: Number(fields.oscillationPenalty.value),
    oscillation_window: Number(fields.oscillationWindow.value),
    chunk_timesteps: Number(fields.chunkTimesteps.value),
    preview_steps: Number(fields.previewSteps.value),
    complete_episode_preview: fields.completeEpisodePreview.checked,
    deterministic_preview: fields.deterministicPreview.checked,
    training_enabled: fields.trainingEnabled.checked,
    num_envs: Number(fields.numEnvs.value),
    n_steps: Number(fields.nSteps.value),
    batch_size: Number(fields.batchSize.value),
    n_epochs: Number(fields.nEpochs.value),
    cnn_channels: fields.cnnChannels.value,
    cnn_kernel_sizes: fields.cnnKernelSizes.value,
    cnn_strides: fields.cnnStrides.value,
    cnn_features_dim: Number(fields.cnnFeaturesDim.value),
    cnn_channel_first: fields.cnnChannelFirst.checked,
  };
}

function readLiveConfig() {
  return {
    learning_rate: Number(fields.learningRate.value),
    gamma: Number(fields.gamma.value),
    ent_coef: Number(fields.entCoef.value),
    clip_range: Number(fields.clipRange.value),
    food_time_penalty: Number(fields.foodTimePenalty.value),
    food_step_limit_multiplier: Number(fields.foodStepLimitMultiplier.value),
    food_reward_bonus: Number(fields.foodRewardBonus.value),
    distance_reward_scale: Number(fields.distanceRewardScale.value),
    loop_penalty: Number(fields.loopPenalty.value),
    loop_window: Number(fields.loopWindow.value),
    oscillation_penalty: Number(fields.oscillationPenalty.value),
    oscillation_window: Number(fields.oscillationWindow.value),
    chunk_timesteps: Number(fields.chunkTimesteps.value),
    preview_steps: Number(fields.previewSteps.value),
    complete_episode_preview: fields.completeEpisodePreview.checked,
    deterministic_preview: fields.deterministicPreview.checked,
    training_enabled: fields.trainingEnabled.checked,
    n_epochs: Number(fields.nEpochs.value),
  };
}

function fillConfig(config) {
  setField(fields.agent, config.agent);
  setField(fields.strategy, config.strategy);
  setField(fields.device, config.device);
  if (document.activeElement !== fields.boardPreset) {
    const boardSize = Number(config.board_size);
    fields.boardPreset.value = boardSize === 12 || boardSize === 21 ? String(boardSize) : "custom";
  }
  setField(fields.boardSize, config.board_size);
  setField(fields.seed, config.seed);
  setField(fields.learningRate, config.learning_rate);
  setField(fields.gamma, config.gamma);
  setField(fields.entCoef, config.ent_coef);
  setField(fields.clipRange, config.clip_range);
  setField(fields.foodTimePenalty, config.food_time_penalty);
  setField(fields.foodStepLimitMultiplier, config.food_step_limit_multiplier);
  setField(fields.foodRewardBonus, config.food_reward_bonus);
  setField(fields.distanceRewardScale, config.distance_reward_scale);
  setField(fields.loopPenalty, config.loop_penalty);
  setField(fields.loopWindow, config.loop_window);
  setField(fields.oscillationPenalty, config.oscillation_penalty);
  setField(fields.oscillationWindow, config.oscillation_window);
  setField(fields.chunkTimesteps, config.chunk_timesteps);
  setField(fields.previewSteps, config.preview_steps);
  if (document.activeElement !== fields.completeEpisodePreview) {
    fields.completeEpisodePreview.checked = config.complete_episode_preview;
  }
  if (document.activeElement !== fields.deterministicPreview) {
    fields.deterministicPreview.checked = config.deterministic_preview;
  }
  if (document.activeElement !== fields.trainingEnabled) {
    fields.trainingEnabled.checked = Boolean(config.training_enabled);
  }
  setField(fields.nSteps, config.n_steps);
  setField(fields.numEnvs, config.num_envs);
  setField(fields.batchSize, config.batch_size);
  setField(fields.nEpochs, config.n_epochs);
  setField(fields.cnnChannels, config.cnn_channels);
  setField(fields.cnnKernelSizes, config.cnn_kernel_sizes);
  setField(fields.cnnStrides, config.cnn_strides);
  setField(fields.cnnFeaturesDim, config.cnn_features_dim);
  if (config.cnn_channel_first !== undefined && document.activeElement !== fields.cnnChannelFirst) {
    fields.cnnChannelFirst.checked = config.cnn_channel_first;
  }
}

function setField(element, value) {
  if (!element) {
    return;
  }
  if (value === undefined || value === null) {
    return;
  }
  if (document.activeElement !== element && !dirtyFields.has(element)) {
    element.value = value;
  }
}

function markFieldDirty(event) {
  dirtyFields.add(event.currentTarget);
  modelStatus.textContent = "Model settings changed. Press Reset With Config to rebuild the model.";
}

function clearDirtyFields() {
  dirtyFields.clear();
}

function drawBoard(frame) {
  const size = frame?.board_size || Number(fields.boardSize.value) || 12;
  const cell = board.width / size;
  boardCtx.fillStyle = "#070809";
  boardCtx.fillRect(0, 0, board.width, board.height);

  boardCtx.strokeStyle = "#171b20";
  boardCtx.lineWidth = 1;
  for (let i = 0; i <= size; i += 1) {
    const p = Math.round(i * cell) + 0.5;
    boardCtx.beginPath();
    boardCtx.moveTo(p, 0);
    boardCtx.lineTo(p, board.height);
    boardCtx.stroke();
    boardCtx.beginPath();
    boardCtx.moveTo(0, p);
    boardCtx.lineTo(board.width, p);
    boardCtx.stroke();
  }

  if (!frame) return;

  if (frame.ate) {
    boardCtx.fillStyle = "rgba(64, 196, 99, 0.18)";
    boardCtx.fillRect(0, 0, board.width, board.height);
  }

  const [fr, fc] = frame.food;
  boardCtx.fillStyle = "#f04444";
  boardCtx.fillRect(fc * cell + 4, fr * cell + 4, cell - 8, cell - 8);

  frame.snake.forEach(([r, c], idx) => {
    const t = idx / Math.max(1, frame.snake.length - 1);
    boardCtx.fillStyle = idx === 0 ? "#50d66b" : `rgb(${50 + 45 * t}, ${165 - 60 * t}, ${115 - 35 * t})`;
    boardCtx.fillRect(c * cell + 3, r * cell + 3, cell - 6, cell - 6);
  });

  if (frame.done) {
    boardCtx.fillStyle = "rgba(0,0,0,0.5)";
    boardCtx.fillRect(0, 0, board.width, board.height);
    boardCtx.fillStyle = "#ffb4ba";
    boardCtx.font = "700 34px system-ui";
    boardCtx.textAlign = "center";
    boardCtx.fillText("Episode Done", board.width / 2, board.height / 2);
  }
}

function drawChart(history) {
  chartCtx.clearRect(0, 0, chart.width, chart.height);
  chartCtx.fillStyle = "#111418";
  chartCtx.fillRect(0, 0, chart.width, chart.height);
  chartCtx.strokeStyle = "#29313a";
  chartCtx.lineWidth = 1;
  for (let y = 20; y < chart.height; y += 35) {
    chartCtx.beginPath();
    chartCtx.moveTo(0, y);
    chartCtx.lineTo(chart.width, y);
    chartCtx.stroke();
  }
  if (!history || history.length < 2) return;

  const maxScore = Math.max(10, ...history.map((h) => h.preview_score || 0));
  const pad = 16;
  const xStep = (chart.width - pad * 2) / Math.max(1, history.length - 1);
  chartCtx.strokeStyle = "#40c463";
  chartCtx.lineWidth = 3;
  chartCtx.beginPath();
  history.forEach((h, idx) => {
    const x = pad + idx * xStep;
    const y = chart.height - pad - ((h.preview_score || 0) / maxScore) * (chart.height - pad * 2);
    if (idx === 0) chartCtx.moveTo(x, y);
    else chartCtx.lineTo(x, y);
  });
  chartCtx.stroke();
}

function renderCnnArchitecture(architecture) {
  const cnn = architecture?.cnn;
  if (!cnn || !cnn.layers?.length) {
    cnnSummary.textContent = "No CNN architecture configured.";
    return;
  }
  const layers = cnn.layers
    .map((layer) => {
      return `L${layer.index}: ${layer.in_channels}->${layer.out_channels}, k${layer.kernel_size}, s${layer.stride}`;
    })
    .join(" | ");
  cnnSummary.textContent = `${layers} | FC ${cnn.features_dim}`;
}

function renderDeviceStatus(data) {
  const info = data.device_info || {};
  const cudaOption = fields.device.querySelector('option[value="cuda"]');
  if (cudaOption) {
    cudaOption.disabled = !info.cuda_available;
    cudaOption.textContent = info.cuda_available ? "CUDA" : "CUDA (unavailable)";
  }

  const requested = fields.device.value || data.config?.device || "cpu";
  const actual = data.actual_device || "not initialized";
  if (info.cuda_available) {
    const gpu = info.cuda_name || `${info.cuda_devices} CUDA device(s)`;
    if (data.actual_device && requested !== data.actual_device) {
      deviceStatus.textContent = `CUDA available: ${gpu}. Requested ${requested}, actual ${actual}. Use Reset With Config to rebuild the model.`;
      deviceStatus.className = "device-status warn";
    } else {
      deviceStatus.textContent = `CUDA available: ${gpu}. Actual: ${actual}.`;
      deviceStatus.className = "device-status ok";
    }
  } else if (requested === "cuda") {
    deviceStatus.textContent = `CUDA unavailable in PyTorch. Actual: ${actual}; training will run on CPU.`;
    deviceStatus.className = "device-status warn";
  } else {
    deviceStatus.textContent = `CUDA unavailable in PyTorch. Actual: ${actual}.`;
    deviceStatus.className = "device-status";
  }
}

function updateStatus(data) {
  fillConfig(data.config);
  renderCnnArchitecture(data.architecture);
  renderDeviceStatus(data);
  stateEl.textContent = data.running ? "running" : "paused";
  eventEl.textContent = data.last_event || "ready";
  trainedStepsEl.textContent = data.trained_steps;
  bestScoreEl.textContent = data.best?.score ?? 0;
  bestStepsEl.textContent = data.best?.steps ?? 0;
  const lastFrame = frames[frameIndex] || frames[frames.length - 1];
  scoreEl.textContent = lastFrame?.score ?? 0;
  lengthEl.textContent = lastFrame?.length ?? lastFrame?.snake?.length ?? 3;
  foodCountEl.textContent = lastFrame?.food_count ?? 0;
  const lastHistory = data.history[data.history.length - 1];
  stepsPerFoodEl.textContent = lastHistory?.avg_steps_per_food ?? 0;
  loopRevisitsEl.textContent = lastHistory?.loop_revisits ?? lastFrame?.loop_revisit_count ?? 0;
  oscillationsEl.textContent = lastHistory?.oscillations ?? lastFrame?.oscillation_count ?? 0;
  rewardEl.textContent = Number(lastFrame?.reward || 0).toFixed(4);
  fpsEl.textContent = lastHistory?.fps ?? 0;
  drawChart(data.history);

  if (data.last_error) {
    errorBox.hidden = false;
    errorBox.textContent = data.last_error;
  } else {
    errorBox.hidden = true;
  }

  if (data.frame_version !== frameVersion && data.frames.length) {
    if (!frames.length) {
      frameVersion = data.frame_version;
      frames = data.frames;
      frameIndex = 0;
    } else {
      pendingFrameVersion = data.frame_version;
      pendingFrames = data.frames;
    }
  }
}

async function poll() {
  try {
    const data = await api("/api/status");
    updateStatus(data);
  } catch (error) {
    errorBox.hidden = false;
    errorBox.textContent = error.message;
  }
}

function animate() {
  const now = performance.now();
  if (now - lastFrameAt < frameDelayMs) {
    requestAnimationFrame(animate);
    return;
  }
  lastFrameAt = now;

  const frame = frames[frameIndex] || frames[frames.length - 1];
  drawBoard(frame);
  if (frame) {
    scoreEl.textContent = frame.score;
    lengthEl.textContent = frame.length ?? frame.snake.length;
    foodCountEl.textContent = frame.food_count ?? 0;
    loopRevisitsEl.textContent = frame.loop_revisit_count ?? 0;
    oscillationsEl.textContent = frame.oscillation_count ?? 0;
    rewardEl.textContent = Number(frame.reward || 0).toFixed(4);
    frameIndex = (frameIndex + 1) % frames.length;
    if (frameIndex === 0 && pendingFrames) {
      frames = pendingFrames;
      frameVersion = pendingFrameVersion;
      pendingFrames = null;
      pendingFrameVersion = -1;
    }
  }
  requestAnimationFrame(animate);
}

playbackSpeed.addEventListener("input", updatePlaybackSpeed);

if (fields.boardPreset) {
  fields.boardPreset.addEventListener("change", (event) => {
    if (event.currentTarget.value !== "custom") {
      fields.boardSize.value = event.currentTarget.value;
    }
    markFieldDirty({ currentTarget: fields.boardSize });
  });
}

fields.boardSize.addEventListener("input", () => {
  const value = Number(fields.boardSize.value);
  if (fields.boardPreset) {
    fields.boardPreset.value = value === 12 || value === 21 ? String(value) : "custom";
  }
});

document.getElementById("startBtn").addEventListener("click", async () => {
  await api("/api/start", { method: "POST" });
  await poll();
});

document.getElementById("pauseBtn").addEventListener("click", async () => {
  await api("/api/pause", { method: "POST" });
  await poll();
});

document.getElementById("resetBtn").addEventListener("click", async () => {
  const data = await api("/api/reset", { method: "POST", body: JSON.stringify({}) });
  clearDirtyFields();
  resetPlaybackFrames();
  updateStatus(data);
  modelStatus.textContent = "Model reset with the current backend config.";
});

document.getElementById("applyBtn").addEventListener("click", async () => {
  await api("/api/settings", { method: "POST", body: JSON.stringify(readLiveConfig()) });
  await poll();
});

document.getElementById("resetConfigBtn").addEventListener("click", async () => {
  const data = await api("/api/reset", { method: "POST", body: JSON.stringify(readConfig()) });
  clearDirtyFields();
  resetPlaybackFrames();
  updateStatus(data);
  modelStatus.textContent = "Model rebuilt with the selected config. Press Start to train.";
});

downloadModelBtn.addEventListener("click", async () => {
  try {
    modelStatus.textContent = "Preparing download...";
    const response = await fetch("/api/model/download");
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || `Download failed: ${response.status}`);
    }
    const blob = await response.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filenameFromDisposition(response.headers.get("Content-Disposition"));
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
    modelStatus.textContent = "Model bundle downloaded.";
  } catch (error) {
    modelStatus.textContent = error.message;
  }
});

importModelBtn.addEventListener("click", async () => {
  const file = modelFile.files[0];
  if (!file) {
    modelStatus.textContent = "Choose a model bundle first.";
    return;
  }

  try {
    modelStatus.textContent = "Importing model...";
    const formData = new FormData();
    formData.append("model", file);
    const response = await fetch("/api/model/upload", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || `Import failed: ${response.status}`);
    }
    resetPlaybackFrames();
    clearDirtyFields();
    updateStatus(data);
    modelStatus.textContent = `Imported at ${data.trained_steps} steps. Press Start to continue.`;
  } catch (error) {
    modelStatus.textContent = error.message;
  }
});

moveDeviceBtn.addEventListener("click", async () => {
  try {
    modelStatus.textContent = "Moving model device...";
    const data = await api("/api/model/device", {
      method: "POST",
      body: JSON.stringify({ device: fields.device.value }),
    });
    updateStatus(data);
    modelStatus.textContent = `Model moved to ${data.actual_device || fields.device.value}. Press Start to continue.`;
  } catch (error) {
    modelStatus.textContent = error.message;
  }
});

[
  fields.agent,
  fields.strategy,
  fields.device,
  fields.boardPreset,
  fields.boardSize,
  fields.seed,
  fields.nSteps,
  fields.numEnvs,
  fields.batchSize,
  fields.cnnChannels,
  fields.cnnKernelSizes,
  fields.cnnStrides,
  fields.cnnFeaturesDim,
  fields.cnnChannelFirst,
].forEach((field) => {
  if (!field) return;
  field.addEventListener("input", markFieldDirty);
  field.addEventListener("change", markFieldDirty);
});

poll();
setInterval(poll, 750);
updatePlaybackSpeed();
animate();
