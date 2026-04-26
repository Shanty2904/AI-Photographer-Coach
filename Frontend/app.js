// ── Constants ─────────────────────────────────────────────────────────────────
const TARGET_CAPTURE_FPS = 24;
const FRAME_INTERVAL_MS  = 1000 / TARGET_CAPTURE_FPS;
const FRAME_WIDTH        = 640;
const FRAME_HEIGHT       = 480;
const JPEG_QUALITY       = 0.48;
const MAX_IN_FLIGHT      = 1;
const COOLDOWN_MS        = 7000; // ms to show each suggestion before evaluating

// ── DOM refs ──────────────────────────────────────────────────────────────────
const video         = document.querySelector("#camera");
const canvas        = document.querySelector("#frameCanvas");
const context       = canvas.getContext("2d", { willReadFrequently: true });
const captureCanvas = document.querySelector("#captureCanvas");
const captureCtx    = captureCanvas.getContext("2d");

const backendUrlInput = document.querySelector("#backendUrl");
const connectButton   = document.querySelector("#connectButton");
const cameraButton    = document.querySelector("#cameraButton");
const captureButton   = document.querySelector("#captureButton");

const serverStatus  = document.querySelector("#serverStatus");
const readyState    = document.querySelector("#readyState");
const coachMessage  = document.querySelector("#coachMessage");
const scoreMeter    = document.querySelector("#scoreMeter");
const scoreValue    = document.querySelector("#scoreValue");
const gradeValue    = document.querySelector("#gradeValue");
const weakestValue  = document.querySelector("#weakestValue");
const historyList   = document.querySelector("#historyList");

// Suggestion controls
const suggestionControls = document.querySelector("#suggestionControls");
const cooldownBar        = document.querySelector("#cooldownBar");
const cooldownTimer      = document.querySelector("#cooldownTimer");
const skipBtn            = document.querySelector("#skipBtn");

// Rotation HUD
const rotationHud  = document.querySelector("#rotationHud");
const tiltDegree   = document.querySelector("#tiltDegree");
const tiltHint     = document.querySelector("#tiltHint");
const dialNeedle   = document.querySelector("#dialNeedle");
const dialArcPath  = document.querySelector("#dialArcPath");

// Report modal
const reportModal       = document.querySelector("#reportModal");
const reportImage       = document.querySelector("#reportImage");
const reportScore       = document.querySelector("#reportScore");
const reportGrade       = document.querySelector("#reportGrade");
const reportTilt        = document.querySelector("#reportTilt");
const reportHistoryList = document.querySelector("#reportHistoryList");
const closeReportBtn    = document.querySelector("#closeReportBtn");
const downloadBtn       = document.querySelector("#downloadBtn");

// Onboarding
const onboardingOverlay = document.querySelector("#onboardingOverlay");
const startAppBtn       = document.querySelector("#startAppBtn");

// ── App state ─────────────────────────────────────────────────────────────────
let stream          = null;
let socket          = null;
let captureTimer    = null;
let inFlightFrames  = 0;
let useHttpFallback = false;
let isPaused        = false;
let lastTiltAngle   = 0;

// Suggestion state machine
let activeSuggestion  = null;  // { message, category, scoreAtStart, weakestAtStart }
let suggestionLocked  = false; // blocks new suggestions during cooldown
let suggestionTimer   = null;
let countdownInterval = null;  // drives the visible countdown
let lastScore         = 0;
let lastWeakest       = "";
const appliedTips     = new Set(); // confirmed improvements only

// ── Onboarding ────────────────────────────────────────────────────────────────
startAppBtn.addEventListener("click", () => {
  onboardingOverlay.classList.add("hidden");
});

// ── Status helpers ────────────────────────────────────────────────────────────
function setStatus(message, state = "") {
  serverStatus.textContent = message;
  serverStatus.className   = "connection-status" + (state === "ready" ? " connected" : "");
  serverStatus.style.color = "";
  if (state === "error")   serverStatus.style.color = "var(--danger)";
  if (state === "warning") serverStatus.style.color = "var(--warn)";
}

function getBackendUrl() {
  return backendUrlInput.value.trim().replace(/\/$/, "");
}
function getWebSocketUrl() {
  const url    = new URL(getBackendUrl());
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = "/ws/live";
  url.search   = "";
  return url.toString();
}

// ── Backend health check ──────────────────────────────────────────────────────
async function checkBackend() {
  try {
    const res     = await fetch(`${getBackendUrl()}/health`, { cache: "no-store" });
    const payload = await res.json();
    if (!res.ok || payload.status !== "ok") throw new Error();
    setStatus("Connected", "ready");
    return true;
  } catch {
    setStatus("Backend offline", "error");
    return false;
  }
}

// ── Camera ────────────────────────────────────────────────────────────────────
async function startCamera() {
  if (stream) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width:  { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { min: TARGET_CAPTURE_FPS, ideal: 30 },
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    cameraButton.textContent = "📷 Camera On";
    cameraButton.disabled    = true;
    captureButton.disabled   = false;
    coachMessage.textContent = "Analyzing frame…";
    startCaptureLoop();
  } catch (err) {
    readyState.textContent   = "Camera blocked";
    coachMessage.textContent = "Camera access denied. Please allow permissions.";
    console.error("Camera error:", err);
  }
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectSocket() {
  if (socket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(socket.readyState)) {
    socket.close();
  }
  useHttpFallback = false;
  socket          = new WebSocket(getWebSocketUrl());
  setStatus("Connecting…", "warning");

  socket.addEventListener("open",    () => setStatus("Connected via WebSocket", "ready"));
  socket.addEventListener("message", (event) => {
    inFlightFrames = Math.max(0, inFlightFrames - 1);
    const payload  = JSON.parse(event.data);
    if (payload.error) {
      readyState.textContent   = "Error";
      readyState.classList.remove("ready");
      coachMessage.textContent = payload.error;
      return;
    }
    if (!isPaused) renderCoach(payload);
  });
  socket.addEventListener("close", () => { inFlightFrames = 0; useHttpFallback = true; setStatus("Connected via HTTP", "ready"); });
  socket.addEventListener("error", () => { useHttpFallback = true; setStatus("Connected via HTTP", "ready"); });
}

// ── Capture loop ──────────────────────────────────────────────────────────────
function startCaptureLoop() {
  if (captureTimer) clearInterval(captureTimer);
  isPaused     = false;
  captureTimer = setInterval(captureAndSendFrame, FRAME_INTERVAL_MS);
}

function captureAndSendFrame() {
  if (!video.videoWidth || isPaused) return;
  if (inFlightFrames >= MAX_IN_FLIGHT) return;

  const { sx, sy, sw, sh } = cropToTarget(video.videoWidth, video.videoHeight, FRAME_WIDTH, FRAME_HEIGHT);
  context.drawImage(video, sx, sy, sw, sh, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
  const image   = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
  const payload = { image, width: FRAME_WIDTH, height: FRAME_HEIGHT, include_details: false, include_tip: false, client_capture_fps: 24 };

  if (!useHttpFallback && socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
    inFlightFrames++;
    return;
  }
  sendFrameOverHttp(payload);
}

function cropToTarget(vW, vH, tW, tH) {
  const videoRatio  = vW / vH;
  const targetRatio = tW / tH;
  let sx = 0, sy = 0, sw = vW, sh = vH;
  if (videoRatio > targetRatio) { sw = vH * targetRatio; sx = (vW - sw) / 2; }
  else                          { sh = vW / targetRatio; sy = (vH - sh) / 2; }
  return { sx, sy, sw, sh };
}

async function sendFrameOverHttp(payload) {
  inFlightFrames++;
  try {
    const res    = await fetch(`${getBackendUrl()}/analyze/live`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const result = await res.json();
    if (!res.ok || result.error) throw new Error(result.error || `${res.status}`);
    if (!isPaused) renderCoach(result);
    setStatus("Connected via HTTP", "ready");
  } catch {
    setStatus("Backend offline", "error");
    readyState.textContent   = "Disconnected";
    readyState.classList.remove("ready");
    coachMessage.textContent = "AI Analysis paused. Connect to the backend.";
  } finally {
    inFlightFrames = Math.max(0, inFlightFrames - 1);
  }
}

// ── Rotation HUD ──────────────────────────────────────────────────────────────
function updateRotationHud(tiltAngle) {
  if (tiltAngle === undefined || tiltAngle === null) return;
  lastTiltAngle = tiltAngle;

  const abs = Math.abs(tiltAngle);
  tiltDegree.textContent = abs.toFixed(1) + "°";

  const clampedAngle = Math.max(-45, Math.min(45, tiltAngle));
  dialNeedle.setAttribute("transform", `rotate(${clampedAngle}, 30, 28)`);

  const maxDash = 75;
  const fill    = Math.min(maxDash, (abs / 45) * maxDash);
  dialArcPath.style.strokeDashoffset = maxDash - fill;

  const isLevel = abs <= 2.0;
  if (isLevel) {
    rotationHud.classList.add("level");
    tiltDegree.style.color   = "";
    dialNeedle.style.stroke  = "var(--success)";
    dialArcPath.style.stroke = "var(--success)";
    tiltHint.textContent     = "Leveled ✓";
  } else {
    rotationHud.classList.remove("level");
    const dir = tiltAngle > 0 ? "↺ CCW" : "↻ CW";
    dialNeedle.style.stroke  = abs > 5 ? "var(--danger)" : "var(--warn)";
    dialArcPath.style.stroke = abs > 5 ? "var(--danger)" : "var(--warn)";
    tiltHint.textContent     = `Rotate ${abs.toFixed(1)}° ${dir}`;
  }
}

// ── Suggestion system ─────────────────────────────────────────────────────────

/** Returns true if a message is a real actionable suggestion (not a "hold steady" type). */
function isActionableSuggestion(msg, ready) {
  if (!msg || ready) return false;
  const lower = msg.toLowerCase();
  return !lower.includes("steady") && !lower.includes("looks good") && !lower.includes("analyzing");
}

/** Start showing a new suggestion and begin the cooldown timer. */
function startSuggestion(message, category, score, weakest) {
  if (suggestionLocked) return;
  if (!isActionableSuggestion(message, false)) return;

  activeSuggestion = { message, category, scoreAtStart: score, weakestAtStart: weakest };
  suggestionLocked = true;

  coachMessage.textContent = message;
  suggestionControls.style.display = "flex";

  // Animate cooldown bar: reset then shrink
  cooldownBar.style.transition = "none";
  cooldownBar.style.width      = "100%";
  requestAnimationFrame(() => requestAnimationFrame(() => {
    cooldownBar.style.transition = `width ${COOLDOWN_MS}ms linear`;
    cooldownBar.style.width      = "0%";
  }));

  // Visible countdown ticker
  let remaining = Math.round(COOLDOWN_MS / 1000);
  cooldownTimer.textContent = remaining + "s";
  clearInterval(countdownInterval);
  countdownInterval = setInterval(() => {
    remaining--;
    cooldownTimer.textContent = remaining > 0 ? remaining + "s" : "…";
    if (remaining <= 0) clearInterval(countdownInterval);
  }, 1000);

  clearTimeout(suggestionTimer);
  suggestionTimer = setTimeout(evaluateSuggestion, COOLDOWN_MS);
}

/** Called when the cooldown expires — checks if the user adjusted, then unlocks. */
function evaluateSuggestion() {
  if (!activeSuggestion) { clearSuggestionState(); return; }

  const scoreImproved    = lastScore > activeSuggestion.scoreAtStart + 0.4;
  const categoryChanged  = lastWeakest !== "" && lastWeakest !== activeSuggestion.weakestAtStart;

  if ((scoreImproved || categoryChanged) && !appliedTips.has(activeSuggestion.message)) {
    appliedTips.add(activeSuggestion.message);
    addImprovementToUI(activeSuggestion.message);
  }

  clearSuggestionState();
}

/** Called when user clicks Skip — dismisses without storing. */
function skipSuggestion() {
  clearTimeout(suggestionTimer);
  clearSuggestionState();
}

/** Resets all suggestion state and hides controls. */
function clearSuggestionState() {
  activeSuggestion = null;
  suggestionLocked = false;
  suggestionControls.style.display = "none";
  cooldownBar.style.transition = "none";
  cooldownBar.style.width = "0%";
  clearInterval(countdownInterval);
  cooldownTimer.textContent = "";
}

/** Adds a confirmed improvement to the sidebar list. */
function addImprovementToUI(message) {
  const isEmpty = historyList.querySelector(".empty-state");
  if (isEmpty) historyList.innerHTML = "";
  const li = document.createElement("li");
  li.textContent = message;
  li.style.animation = "slideIn .35s cubic-bezier(.16,1,.3,1)";
  historyList.prepend(li);
}

skipBtn.addEventListener("click", skipSuggestion);

// ── Render coach data ─────────────────────────────────────────────────────────
function renderCoach(payload) {
  const live         = payload.live || {};
  const score        = Number(live.score || 0);
  const scorePercent = Math.max(0, Math.min(100, score * 10));
  const ready        = Boolean(live.ready_to_capture);

  // Track latest values for suggestion evaluation
  lastScore   = score;
  lastWeakest = live.weakest_category || "";

  // Update status pill
  readyState.textContent = ready ? "✓ Perfect Framing!" : "Analyzing…";
  readyState.className   = `status-indicator${ready ? " ready" : ""}`;

  // --- Suggestion logic ---
  if (suggestionLocked) {
    // If user reached perfect framing while a suggestion is active, count it as applied immediately
    if (ready && activeSuggestion && !appliedTips.has(activeSuggestion.message)) {
      appliedTips.add(activeSuggestion.message);
      addImprovementToUI(activeSuggestion.message);
      clearTimeout(suggestionTimer);
      clearSuggestionState();
    }
    // Keep showing the active suggestion — don't overwrite coachMessage
  } else {
    // Not locked: update coach message and potentially trigger a new suggestion
    const msg = live.message || "Looking good. Keep it steady.";
    coachMessage.textContent = msg;
    if (isActionableSuggestion(msg, ready)) {
      startSuggestion(msg, live.weakest_category || "", score, live.weakest_category || "");
    }
  }

  // Update score display
  scoreValue.textContent = live.score ?? "--";
  gradeValue.textContent = live.grade  || "--";
  weakestValue.textContent = (live.weakest_category || "None").replaceAll("_", " ");

  scoreMeter.style.width      = `${scorePercent}%`;
  scoreMeter.style.background = ready ? "var(--success)" : score >= 6 ? "var(--warn)" : "var(--danger)";
  scoreMeter.style.boxShadow  = ready
    ? "0 0 14px var(--success-glow)"
    : score >= 6
    ? "0 0 14px var(--warn-glow)"
    : "0 0 14px var(--danger-glow)";

  // Update rotation HUD
  const analysis = payload.analysis || {};
  const horizon  = analysis.horizon  || {};
  if (horizon.tilt_angle !== undefined) {
    updateRotationHud(horizon.tilt_angle);
  } else {
    const match = live.message && live.message.match(/([\d.]+)°/);
    if (match) {
      const deg = parseFloat(match[1]);
      const dir = live.message.toLowerCase().includes("counter") ? 1 : -1;
      updateRotationHud(deg * dir);
    }
  }
}

// ── Capture & Report ──────────────────────────────────────────────────────────
captureButton.addEventListener("click", () => {
  if (!video.videoWidth) return;
  isPaused = true;
  clearInterval(captureTimer);

  const tW = 1280, tH = 720;
  const { sx, sy, sw, sh } = cropToTarget(video.videoWidth, video.videoHeight, tW, tH);
  captureCtx.drawImage(video, sx, sy, sw, sh, 0, 0, tW, tH);
  const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.95);

  reportImage.src         = dataUrl;
  reportScore.textContent = scoreValue.textContent;
  reportGrade.textContent = gradeValue.textContent;
  reportTilt.textContent  = Math.abs(lastTiltAngle).toFixed(1) + "°";

  reportHistoryList.innerHTML = "";
  if (appliedTips.size === 0) {
    const li = document.createElement("li");
    li.textContent = "No specific adjustments were needed. Perfect shot!";
    reportHistoryList.appendChild(li);
  } else {
    appliedTips.forEach(tip => {
      const li = document.createElement("li");
      li.textContent = tip;
      reportHistoryList.appendChild(li);
    });
  }

  reportModal.classList.add("active");
});

closeReportBtn.addEventListener("click", () => {
  reportModal.classList.remove("active");
  startCaptureLoop();
});

// ── Download as JPG ───────────────────────────────────────────────────────────
downloadBtn.addEventListener("click", () => {
  const jpegDataUrl = reportImage.src.startsWith("data:image/jpeg")
    ? reportImage.src
    : (() => {
        const img       = new Image();
        img.src         = reportImage.src;
        const tmpCanvas = document.createElement("canvas");
        tmpCanvas.width  = img.naturalWidth  || 1280;
        tmpCanvas.height = img.naturalHeight || 720;
        tmpCanvas.getContext("2d").drawImage(img, 0, 0);
        return tmpCanvas.toDataURL("image/jpeg", 0.95);
      })();

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const a         = document.createElement("a");
  a.href          = jpegDataUrl;
  a.download      = `AI_Photo_Coach_${timestamp}.jpg`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});

// ── Wire buttons ──────────────────────────────────────────────────────────────
connectButton.addEventListener("click", async () => {
  await checkBackend();
  connectSocket();
});
cameraButton.addEventListener("click", startCamera);

// ── Boot ──────────────────────────────────────────────────────────────────────
checkBackend().then(ok => { if (ok) connectSocket(); });
