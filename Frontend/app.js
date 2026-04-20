const TARGET_CAPTURE_FPS = 24;
const FRAME_INTERVAL_MS = 1000 / TARGET_CAPTURE_FPS;
const FRAME_WIDTH = 640;
const FRAME_HEIGHT = 480;
const JPEG_QUALITY = 0.48;
const MAX_IN_FLIGHT = 1;

const video = document.querySelector("#camera");
const canvas = document.querySelector("#frameCanvas");
const context = canvas.getContext("2d", { willReadFrequently: true });
const backendUrlInput = document.querySelector("#backendUrl");
const connectButton = document.querySelector("#connectButton");
const cameraButton = document.querySelector("#cameraButton");
const serverStatus = document.querySelector("#serverStatus");
const readyState = document.querySelector("#readyState");
const coachMessage = document.querySelector("#coachMessage");
const scoreMeter = document.querySelector("#scoreMeter");
const scoreValue = document.querySelector("#scoreValue");
const gradeValue = document.querySelector("#gradeValue");
const weakestValue = document.querySelector("#weakestValue");
const captureFpsValue = document.querySelector("#captureFps");
const analysisFpsValue = document.querySelector("#analysisFps");
const latencyValue = document.querySelector("#latencyValue");

let stream = null;
let socket = null;
let captureTimer = null;
let inFlightFrames = 0;
let capturedFrames = 0;
let analyzedFrames = 0;
let useHttpFallback = false;
let captureWindowStartedAt = performance.now();
let analysisWindowStartedAt = performance.now();
let measuredCaptureFps = 0;

function setStatus(message, state = "") {
  serverStatus.textContent = message;
  serverStatus.className = `status-pill ${state}`.trim();
}

function getBackendUrl() {
  return backendUrlInput.value.trim().replace(/\/$/, "");
}

function getWebSocketUrl() {
  const url = new URL(getBackendUrl());
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = "/ws/live";
  url.search = "";
  return url.toString();
}

async function checkBackend() {
  try {
    const response = await fetch(`${getBackendUrl()}/health`, { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok || payload.status !== "ok") {
      throw new Error("Backend did not return ok");
    }
    setStatus(`Backend ready, target ${payload.live_capture_target_fps || 24}fps`, "ready");
    return true;
  } catch (error) {
    setStatus("Backend offline", "error");
    return false;
  }
}

async function startCamera() {
  if (stream) {
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { min: TARGET_CAPTURE_FPS, ideal: 30 },
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    cameraButton.textContent = "Camera on";
    startCaptureLoop();
  } catch (error) {
    readyState.textContent = "Camera blocked";
    coachMessage.textContent = "Allow camera access and reload this page.";
  }
}

function connectSocket() {
  if (socket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(socket.readyState)) {
    socket.close();
  }

  useHttpFallback = false;
  socket = new WebSocket(getWebSocketUrl());
  setStatus("Connecting backend");

  socket.addEventListener("open", () => {
    setStatus("Live coaching connected", "ready");
    connectButton.textContent = "Reconnect";
  });

  socket.addEventListener("message", (event) => {
    inFlightFrames = Math.max(0, inFlightFrames - 1);
    analyzedFrames += 1;
    updateAnalysisFps();

    const payload = JSON.parse(event.data);
    if (payload.error) {
      readyState.textContent = "Backend error";
      readyState.classList.remove("ready");
      coachMessage.textContent = payload.error;
      return;
    }

    renderCoach(payload);
  });

  socket.addEventListener("close", () => {
    inFlightFrames = 0;
    useHttpFallback = true;
    setStatus("Live coaching connected (HTTP)", "ready");
  });

  socket.addEventListener("error", () => {
    useHttpFallback = true;
    setStatus("Live coaching connected (HTTP)", "ready");
  });
}

function startCaptureLoop() {
  if (captureTimer) {
    clearInterval(captureTimer);
  }

  captureTimer = setInterval(captureAndSendFrame, FRAME_INTERVAL_MS);
}

function captureAndSendFrame() {
  if (!video.videoWidth) {
    return;
  }

  capturedFrames += 1;
  updateCaptureFps();

  if (inFlightFrames >= MAX_IN_FLIGHT) {
    return;
  }

  const videoRatio = video.videoWidth / video.videoHeight;
  const targetRatio = FRAME_WIDTH / FRAME_HEIGHT;
  let sx = 0;
  let sy = 0;
  let sw = video.videoWidth;
  let sh = video.videoHeight;

  if (videoRatio > targetRatio) {
    sw = video.videoHeight * targetRatio;
    sx = (video.videoWidth - sw) / 2;
  } else {
    sh = video.videoWidth / targetRatio;
    sy = (video.videoHeight - sh) / 2;
  }

  context.drawImage(video, sx, sy, sw, sh, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
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
    const response = await fetch(`${getBackendUrl()}/analyze/live`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok || result.error) {
      throw new Error(result.error || `Backend returned ${response.status}`);
    }
    analyzedFrames += 1;
    updateAnalysisFps();
    renderCoach(result);
    setStatus("Live coaching connected (HTTP)", "ready");
  } catch (error) {
    setStatus("Backend unavailable", "error");
    readyState.textContent = "Backend error";
    readyState.classList.remove("ready");
    coachMessage.textContent = error.message || "Live coaching is temporarily unavailable.";
  } finally {
    inFlightFrames = Math.max(0, inFlightFrames - 1);
  }
}

function updateCaptureFps() {
  const now = performance.now();
  const elapsed = now - captureWindowStartedAt;
  if (elapsed >= 1000) {
    measuredCaptureFps = capturedFrames / (elapsed / 1000);
    captureFpsValue.textContent = measuredCaptureFps.toFixed(1);
    capturedFrames = 0;
    captureWindowStartedAt = now;
  }
}

function updateAnalysisFps() {
  const now = performance.now();
  const elapsed = now - analysisWindowStartedAt;
  if (elapsed >= 1000) {
    analysisFpsValue.textContent = (analyzedFrames / (elapsed / 1000)).toFixed(1);
    analyzedFrames = 0;
    analysisWindowStartedAt = now;
  }
}

function renderCoach(payload) {
  const live = payload.live || {};
  const capture = payload.capture || {};
  const score = Number(live.score || 0);
  const scorePercent = Math.max(0, Math.min(100, score * 10));
  const ready = Boolean(live.ready_to_capture);

  readyState.textContent = ready ? "Ready to capture" : "Adjust before capture";
  readyState.classList.toggle("ready", ready);
  coachMessage.textContent = live.message || "Hold steady while the next frame is checked.";
  scoreValue.textContent = live.score ?? "--";
  gradeValue.textContent = live.grade || "--";
  weakestValue.textContent = (live.weakest_category || "Live").replaceAll("_", " ");
  scoreMeter.style.width = `${scorePercent}%`;
  scoreMeter.style.background = ready ? "var(--ready)" : score >= 6 ? "var(--warn)" : "var(--alert)";

  if (capture.client_capture_fps) {
    captureFpsValue.textContent = Number(capture.client_capture_fps).toFixed(1);
  }
  if (capture.server_analysis_fps) {
    analysisFpsValue.textContent = Number(capture.server_analysis_fps).toFixed(1);
  }
  latencyValue.textContent = capture.processing_ms ? `${capture.processing_ms}ms` : "--";
}

connectButton.addEventListener("click", async () => {
  await checkBackend();
  connectSocket();
});

cameraButton.addEventListener("click", startCamera);

checkBackend().then((ok) => {
  if (ok) {
    connectSocket();
  }
});
