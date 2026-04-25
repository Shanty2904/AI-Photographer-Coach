/* =================================================================
   AI PHOTOGRAPHER COACH — Frontend application logic
   Works with the new self-explanatory UI (index.html v2).
================================================================= */

const TARGET_CAPTURE_FPS  = 24;
const FRAME_INTERVAL_MS   = 1000 / TARGET_CAPTURE_FPS;
const FRAME_WIDTH         = 640;
const FRAME_HEIGHT        = 480;
const JPEG_QUALITY        = 0.48;
const MAX_IN_FLIGHT       = 1;

/* ── DOM refs ─────────────────────────────────────────────────── */
const video             = document.querySelector("#camera");
const canvas            = document.querySelector("#frameCanvas");
const ctx               = canvas.getContext("2d", { willReadFrequently: true });
const backendUrlInput   = document.querySelector("#backendUrl");
const connectButton     = document.querySelector("#connectButton");
const cameraButton      = document.querySelector("#cameraButton");
const switchCameraButton= document.querySelector("#switchCameraButton");
const reconnectButton   = document.querySelector("#reconnectButton");
const floatControls     = document.querySelector("#floatControls");
const setupPanel        = document.querySelector("#setupPanel");
const viewfinderGrid    = document.querySelector("#viewfinderGrid");

// Status / connection badge
const serverStatus      = document.querySelector("#serverStatus");
const statusDot         = document.querySelector("#statusDot");   // unused separately, badge class handles colour
const statusText        = document.querySelector("#statusText");

// Readiness
const readinessBadge    = document.querySelector("#readinessBadge");
const readinessIcon     = document.querySelector("#readinessIcon");
const readinessLabel    = document.querySelector("#readinessLabel");

// Score arc
const arcFill           = document.querySelector("#arcFill");
const arcGrade          = document.querySelector("#gradeValue");
const scoreValueEl      = document.querySelector("#scoreValue");

// Coach panel
const coachMessage      = document.querySelector("#coachMessage");
const scoreMeter        = document.querySelector("#scoreMeter");
const scoreBarVal       = document.querySelector("#scoreBarVal");
const weakestValue      = document.querySelector("#weakestValue");

// Metrics
const captureFpsEl      = document.querySelector("#captureFps");
const analysisFpsEl     = document.querySelector("#analysisFps");
const latencyEl         = document.querySelector("#latencyValue");

// Setup steps
const step1El           = document.querySelector("#step1");
const step2El           = document.querySelector("#step2");

/* ── SVG arc constants ─────────────────────────────────────────
   Circle: r=32 → circumference = 2πr ≈ 201.06
─────────────────────────────────────────────────────────────── */
const ARC_CIRCUMFERENCE = 2 * Math.PI * 32;


/* ── Auto-detect backend URL ───────────────────────────────────── */
(function setDefaultBackendUrl() {
  const loc = window.location;
  backendUrlInput.value = loc.protocol === "file:"
    ? "http://localhost:8000"
    : `${loc.protocol}//${loc.host}`;
})();


/* ── State ─────────────────────────────────────────────────────── */
let stream                = null;
let socket                = null;
let captureTimer          = null;
let inFlightFrames        = 0;
let capturedFrames        = 0;
let analyzedFrames        = 0;
let useHttpFallback       = false;
let captureWindowStartAt  = performance.now();
let analysisWindowStartAt = performance.now();
let measuredCaptureFps    = 0;
let backCameras           = [];
let currentCameraIndex    = 0;


/* ════════════════════════════════════════════════════════════════
   STATUS / CONNECTION BADGE
════════════════════════════════════════════════════════════════ */
function setStatus(message, state = "") {
  statusText.textContent = message;
  serverStatus.className = `connection-badge ${state}`.trim();
}


/* ════════════════════════════════════════════════════════════════
   BACKEND CHECK
════════════════════════════════════════════════════════════════ */
function getBackendUrl() {
  return backendUrlInput.value.trim().replace(/\/$/, "");
}

function getWebSocketUrl() {
  const url      = new URL(getBackendUrl());
  url.protocol   = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname   = "/ws/live";
  url.search     = "";
  return url.toString();
}

async function checkBackend() {
  setStatus("Connecting to AI engine…");
  try {
    const res     = await fetch(`${getBackendUrl()}/health`, { cache: "no-store" });
    const payload = await res.json();
    if (!res.ok || payload.status !== "ok") throw new Error("Backend did not return ok");
    setStatus(`AI engine ready · ${payload.live_capture_target_fps || 24} fps target`, "ready");
    step1El.classList.add("done");
    return true;
  } catch {
    setStatus("AI engine offline — check the address", "error");
    return false;
  }
}


/* ════════════════════════════════════════════════════════════════
   CAMERA MANAGEMENT
════════════════════════════════════════════════════════════════ */
function _cameraScore(label) {
  const l = label.toLowerCase();
  if (l.includes("ultra") || l.includes("ultrawide")) return 40;
  if (l.includes("wide"))                              return 30;
  if (l.includes("telephoto") || l.includes("zoom"))  return 20;
  if (l.includes("front"))                             return 99;
  return 10;
}

async function _enumerateBackCameras() {
  const devices  = await navigator.mediaDevices.enumerateDevices();
  const videos   = devices.filter(d => d.kind === "videoinput");
  const hasLabels= videos.some(d => d.label);
  const back     = hasLabels
    ? videos.filter(d => !d.label.toLowerCase().includes("front"))
    : videos;
  back.sort((a, b) => _cameraScore(a.label) - _cameraScore(b.label));
  backCameras = back;
}

async function _openCamera(index) {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  const device           = backCameras[index];
  const videoConstraints = device
    ? { deviceId: { exact: device.deviceId }, width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { min: TARGET_CAPTURE_FPS, ideal: 30 } }
    : { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720  }, frameRate: { min: TARGET_CAPTURE_FPS, ideal: 30 } };

  stream          = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
  video.srcObject = stream;
  await video.play();

  cameraButton.textContent = "Camera on";
  if (backCameras.length > 1) {
    switchCameraButton.title       = `Lens ${index + 1} of ${backCameras.length}`;
  }
}

async function startCamera() {
  if (stream) return;
  try {
    stream          = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
    video.srcObject = stream;
    await video.play();

    await _enumerateBackCameras();
    if (backCameras.length > 0) {
      currentCameraIndex = 0;
      await _openCamera(0);
    } else {
      cameraButton.textContent = "Camera on";
    }

    // Transition UI from setup → live mode
    setupPanel.classList.add("hidden");
    floatControls.style.display = "flex";
    if (backCameras.length < 2) switchCameraButton.style.display = "none";
    step2El.classList.add("done");

    // Show viewfinder grid
    setTimeout(() => viewfinderGrid.classList.add("visible"), 400);

    // Show readiness badge (waiting state)
    readinessBadge.classList.add("visible");

    startCaptureLoop();
  } catch {
    setReadiness("Camera blocked", "🚫", false);
    coachMessage.textContent = "Please allow camera access and reload this page.";
  }
}

async function switchCamera() {
  if (backCameras.length < 2) return;
  currentCameraIndex = (currentCameraIndex + 1) % backCameras.length;
  try   { await _openCamera(currentCameraIndex); }
  catch { currentCameraIndex = (currentCameraIndex + 1) % backCameras.length; await _openCamera(currentCameraIndex); }
}


/* ════════════════════════════════════════════════════════════════
   WEBSOCKET / SOCKET CONNECTION
════════════════════════════════════════════════════════════════ */
function connectSocket() {
  if (socket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(socket.readyState)) {
    socket.close();
  }
  useHttpFallback = false;
  socket          = new WebSocket(getWebSocketUrl());
  setStatus("Connecting to AI engine…");

  socket.addEventListener("open", () => {
    setStatus("AI engine connected — live coaching active", "ready");
    connectButton.textContent = "Reconnect";
  });

  socket.addEventListener("message", (event) => {
    inFlightFrames = Math.max(0, inFlightFrames - 1);
    analyzedFrames += 1;
    updateAnalysisFps();
    const payload = JSON.parse(event.data);
    if (payload.error) {
      setReadiness("AI error", "⚠️", false);
      coachMessage.textContent = payload.error;
      return;
    }
    renderCoach(payload);
  });

  socket.addEventListener("close", () => {
    inFlightFrames  = 0;
    useHttpFallback = true;
    setStatus("AI engine connected via HTTP fallback", "ready");
  });

  socket.addEventListener("error", () => {
    useHttpFallback = true;
    setStatus("AI engine connected via HTTP fallback", "ready");
  });
}


/* ════════════════════════════════════════════════════════════════
   FRAME CAPTURE LOOP
════════════════════════════════════════════════════════════════ */
function startCaptureLoop() {
  if (captureTimer) clearInterval(captureTimer);
  captureTimer = setInterval(captureAndSendFrame, FRAME_INTERVAL_MS);
}

function captureAndSendFrame() {
  if (!video.videoWidth) return;

  capturedFrames += 1;
  updateCaptureFps();

  if (inFlightFrames >= MAX_IN_FLIGHT) return;

  const vr = video.videoWidth / video.videoHeight;
  const tr = FRAME_WIDTH / FRAME_HEIGHT;
  let sx = 0, sy = 0, sw = video.videoWidth, sh = video.videoHeight;
  if (vr > tr) { sw = video.videoHeight * tr; sx = (video.videoWidth - sw) / 2; }
  else         { sh = video.videoWidth  / tr; sy = (video.videoHeight - sh) / 2; }

  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
  const image = canvas.toDataURL("image/jpeg", JPEG_QUALITY);

  const payload = {
    image,
    width: FRAME_WIDTH,
    height: FRAME_HEIGHT,
    include_details: false,
    include_tip: false,
    client_capture_fps: measuredCaptureFps,
  };

  if (!useHttpFallback && socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
    inFlightFrames += 1;
    return;
  }
  sendFrameOverHttp(payload);
}

async function sendFrameOverHttp(payload) {
  inFlightFrames += 1;
  try {
    const res    = await fetch(`${getBackendUrl()}/analyze/live`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await res.json();
    if (!res.ok || result.error) throw new Error(result.error || `Backend returned ${res.status}`);
    analyzedFrames += 1;
    updateAnalysisFps();
    renderCoach(result);
    setStatus("AI engine connected via HTTP", "ready");
  } catch (err) {
    setStatus("AI engine unavailable", "error");
    setReadiness("Reconnect AI", "⚠️", false);
    coachMessage.textContent = err.message || "Live coaching temporarily unavailable.";
  } finally {
    inFlightFrames = Math.max(0, inFlightFrames - 1);
  }
}


/* ════════════════════════════════════════════════════════════════
   FPS TRACKING
════════════════════════════════════════════════════════════════ */
function updateCaptureFps() {
  const now     = performance.now();
  const elapsed = now - captureWindowStartAt;
  if (elapsed >= 1000) {
    measuredCaptureFps          = capturedFrames / (elapsed / 1000);
    captureFpsEl.textContent    = measuredCaptureFps.toFixed(1);
    capturedFrames              = 0;
    captureWindowStartAt        = now;
  }
}

function updateAnalysisFps() {
  const now     = performance.now();
  const elapsed = now - analysisWindowStartAt;
  if (elapsed >= 1000) {
    analysisFpsEl.textContent   = (analyzedFrames / (elapsed / 1000)).toFixed(1);
    analyzedFrames              = 0;
    analysisWindowStartAt       = now;
  }
}


/* ════════════════════════════════════════════════════════════════
   RENDER COACH — update every UI element from server payload
════════════════════════════════════════════════════════════════ */
function setReadiness(label, icon, isReady) {
  readinessIcon.textContent  = icon;
  readinessLabel.textContent = label;
  readinessBadge.classList.toggle("ready", isReady);
}

function setArc(scorePercent, ready, score) {
  // Circumference = 201.06; offset 0 = full circle, offset 201 = empty
  const offset = ARC_CIRCUMFERENCE - (ARC_CIRCUMFERENCE * scorePercent / 100);
  arcFill.style.strokeDashoffset = offset.toFixed(1);

  // Colour
  arcFill.classList.remove("ready", "warn", "danger");
  arcGrade.classList.remove("ready-color", "warn-color", "danger-color");
  if (ready) {
    arcFill.classList.add("ready");
    arcGrade.classList.add("ready-color");
  } else if (score >= 6) {
    arcFill.classList.add("warn");
    arcGrade.classList.add("warn-color");
  } else {
    arcFill.classList.add("danger");
    arcGrade.classList.add("danger-color");
  }
}

function renderCoach(payload) {
  const live    = payload.live    || {};
  const capture = payload.capture || {};
  const score   = Number(live.score || 0);
  const pct     = Math.max(0, Math.min(100, score * 10));
  const ready   = Boolean(live.ready_to_capture);
  const grade   = live.grade || "--";
  const weakest = (live.weakest_category || "").replaceAll("_", " ") || "–";
  const message = live.message || "Hold steady while the next frame is analysed.";

  /* Readiness badge */
  if (ready) {
    setReadiness("Ready to capture!", "✅", true);
  } else {
    setReadiness("Adjust before capturing", "🎯", false);
  }

  /* Coach message */
  coachMessage.textContent = message;

  /* Score bar */
  scoreMeter.style.width        = `${pct}%`;
  scoreMeter.style.background   = ready ? "var(--ready)" : score >= 6 ? "var(--warn)" : "var(--danger)";
  scoreBarVal.textContent       = `${score.toFixed(1)} / 10`;

  /* Focus tag */
  weakestValue.textContent = weakest || "–";

  /* Score arc + grade */
  scoreValueEl.textContent = score.toFixed(1);
  arcGrade.textContent     = grade;
  setArc(pct, ready, score);

  /* Metrics */
  if (capture.client_capture_fps != null) {
    captureFpsEl.textContent = Number(capture.client_capture_fps).toFixed(1);
  }
  if (capture.server_analysis_fps != null) {
    analysisFpsEl.textContent = Number(capture.server_analysis_fps).toFixed(1);
  }
  latencyEl.textContent = capture.processing_ms ? `${capture.processing_ms} ms` : "--";
}


/* ════════════════════════════════════════════════════════════════
   EVENT LISTENERS
════════════════════════════════════════════════════════════════ */
connectButton.addEventListener("click", async () => {
  const ok = await checkBackend();
  if (ok) connectSocket();
});

reconnectButton.addEventListener("click", async () => {
  const ok = await checkBackend();
  if (ok) connectSocket();
});

cameraButton.addEventListener("click", startCamera);
switchCameraButton.addEventListener("click", switchCamera);


/* ════════════════════════════════════════════════════════════════
   BOOT — auto-connect on page load
════════════════════════════════════════════════════════════════ */
checkBackend().then((ok) => {
  if (ok) connectSocket();
});
