import { useEffect, useRef, useState } from "react";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1"
).replace(/\/$/, "");
const CAMERA_WIDTH = Number(import.meta.env.VITE_CAMERA_WIDTH ?? 640);
const CAMERA_HEIGHT = Number(import.meta.env.VITE_CAMERA_HEIGHT ?? 360);
const DEFAULT_PROCESS_WIDTH = Number(import.meta.env.VITE_PROCESS_WIDTH ?? 96);
const DEFAULT_PROCESS_HEIGHT = Number(import.meta.env.VITE_PROCESS_HEIGHT ?? 54);
const CAPTURE_QUALITY = Number(import.meta.env.VITE_CAPTURE_QUALITY ?? 0.82);
const UPSCALE_OUTSCALE = Number(import.meta.env.VITE_UPSCALE_OUTSCALE ?? 4);
const OUTPUT_FORMAT = (import.meta.env.VITE_OUTPUT_FORMAT ?? "jpeg").toLowerCase();
const CAPTURE_MIME_TYPE = OUTPUT_FORMAT === "png" ? "image/png" : "image/jpeg";
const CAPTURE_EXTENSION = OUTPUT_FORMAT === "png" ? "png" : "jpeg";
const MIN_PROCESS_DIMENSION = 16;
const MAX_PROCESS_DIMENSION = 2048;
const UPSCALE_METHOD_OPTIONS = [
  { value: "realesrgan", label: "Real-ESRGAN" },
  { value: "bicubic", label: "Bicubic" },
];
const DEFAULT_UPSCALE_METHOD = UPSCALE_METHOD_OPTIONS[0].value;

function formatMs(value) {
  return value == null ? "n/a" : `${value.toFixed(1)} ms`;
}

function formatFps(value) {
  return value <= 0 ? "0.0 fps" : `${value.toFixed(1)} fps`;
}

function formatMethodLabel(value) {
  return (
    UPSCALE_METHOD_OPTIONS.find((option) => option.value === value)?.label ?? value
  );
}

function normalizeProcessDimension(value, fallback) {
  const parsedValue = Number.parseInt(value, 10);
  if (!Number.isFinite(parsedValue)) {
    return fallback;
  }
  return Math.min(
    MAX_PROCESS_DIMENSION,
    Math.max(MIN_PROCESS_DIMENSION, parsedValue),
  );
}

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const requestInFlightRef = useRef(false);
  const cameraActiveRef = useRef(false);
  const loopTimerRef = useRef(null);
  const lastFrameUrlRef = useRef(null);
  const fpsTimestampsRef = useRef([]);
  const mountedRef = useRef(true);
  const processWidthRef = useRef(DEFAULT_PROCESS_WIDTH);
  const processHeightRef = useRef(DEFAULT_PROCESS_HEIGHT);
  const upscaleMethodRef = useRef(DEFAULT_UPSCALE_METHOD);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [sourceStatus, setSourceStatus] = useState("Idle");
  const [backendStatus, setBackendStatus] = useState("Checking backend...");
  const [processedFrameUrl, setProcessedFrameUrl] = useState(null);
  const [processingTimeMs, setProcessingTimeMs] = useState(null);
  const [roundTripMs, setRoundTripMs] = useState(null);
  const [processedFps, setProcessedFps] = useState(0);
  const [lastResolution, setLastResolution] = useState("n/a");
  const [lastError, setLastError] = useState("");
  const [processWidthInput, setProcessWidthInput] = useState(
    String(DEFAULT_PROCESS_WIDTH),
  );
  const [processHeightInput, setProcessHeightInput] = useState(
    String(DEFAULT_PROCESS_HEIGHT),
  );
  const [upscaleMethod, setUpscaleMethod] = useState(DEFAULT_UPSCALE_METHOD);

  const processWidth = normalizeProcessDimension(
    processWidthInput,
    DEFAULT_PROCESS_WIDTH,
  );
  const processHeight = normalizeProcessDimension(
    processHeightInput,
    DEFAULT_PROCESS_HEIGHT,
  );

  useEffect(() => {
    mountedRef.current = true;
    void checkBackendHealth();

    return () => {
      mountedRef.current = false;
      stopProcessingLoop();
      stopCameraTracks();
      releaseCurrentFrameUrl();
    };
  }, []);

  useEffect(() => {
    processWidthRef.current = processWidth;
  }, [processWidth]);

  useEffect(() => {
    processHeightRef.current = processHeight;
  }, [processHeight]);

  useEffect(() => {
    upscaleMethodRef.current = upscaleMethod;
  }, [upscaleMethod]);

  async function checkBackendHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) {
        throw new Error(`Backend responded with ${response.status}`);
      }
      const payload = await response.json();
      if (mountedRef.current) {
        setBackendStatus(`Backend ready: ${payload.model_name} on ${payload.device}`);
      }
    } catch (error) {
      if (mountedRef.current) {
        setBackendStatus(
          error instanceof Error ? error.message : "Backend health check failed.",
        );
      }
    }
  }

  function stopCameraTracks() {
    if (!streamRef.current) {
      return;
    }
    for (const track of streamRef.current.getTracks()) {
      track.stop();
    }
    streamRef.current = null;
  }

  function stopProcessingLoop() {
    if (loopTimerRef.current) {
      window.clearTimeout(loopTimerRef.current);
      loopTimerRef.current = null;
    }
    requestInFlightRef.current = false;
  }

  function releaseCurrentFrameUrl() {
    if (lastFrameUrlRef.current) {
      URL.revokeObjectURL(lastFrameUrlRef.current);
      lastFrameUrlRef.current = null;
    }
  }

  function commitProcessWidth() {
    setProcessWidthInput(String(processWidth));
  }

  function commitProcessHeight() {
    setProcessHeightInput(String(processHeight));
  }

  function resetProcessSize() {
    setProcessWidthInput(String(DEFAULT_PROCESS_WIDTH));
    setProcessHeightInput(String(DEFAULT_PROCESS_HEIGHT));
  }

  async function startCamera() {
    setLastError("");
    setSourceStatus("Requesting camera...");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: CAMERA_WIDTH },
          height: { ideal: CAMERA_HEIGHT },
          facingMode: "user",
        },
        audio: false,
      });

      streamRef.current = stream;
      const videoElement = videoRef.current;
      if (!videoElement) {
        throw new Error("Video element is not available.");
      }

      videoElement.srcObject = stream;
      await videoElement.play();

      cameraActiveRef.current = true;
      setCameraReady(true);
      setCameraActive(true);
      setSourceStatus("Camera live");
      scheduleNextFrame(0);
    } catch (error) {
      cameraActiveRef.current = false;
      stopCameraTracks();
      setCameraReady(false);
      setCameraActive(false);
      setSourceStatus("Camera unavailable");
      setLastError(error instanceof Error ? error.message : "Unable to start camera.");
    }
  }

  function stopCamera() {
    cameraActiveRef.current = false;
    stopProcessingLoop();
    stopCameraTracks();
    setCameraActive(false);
    setCameraReady(false);
    setSourceStatus("Stream stopped");
  }

  function scheduleNextFrame(delayMs) {
    stopProcessingLoop();
    loopTimerRef.current = window.setTimeout(() => {
      void captureAndSendFrame();
    }, delayMs);
  }

  async function captureAndSendFrame() {
    if (!cameraActiveRef.current || requestInFlightRef.current) {
      return;
    }

    const videoElement = videoRef.current;
    const canvasElement = canvasRef.current;
    if (!videoElement || !canvasElement) {
      scheduleNextFrame(0);
      return;
    }

    if (videoElement.readyState < 2 || videoElement.videoWidth === 0) {
      scheduleNextFrame(250);
      return;
    }

    requestInFlightRef.current = true;

    try {
      const context = canvasElement.getContext("2d");
      if (!context) {
        throw new Error("Canvas 2D context is not available.");
      }

      canvasElement.width = processWidthRef.current;
      canvasElement.height = processHeightRef.current;
      context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

      const frameBlob = await new Promise((resolve, reject) => {
        canvasElement.toBlob(
          (blob) => {
            if (blob) {
              resolve(blob);
              return;
            }
            reject(new Error("Unable to encode camera frame."));
          },
          CAPTURE_MIME_TYPE,
          OUTPUT_FORMAT === "png" ? undefined : CAPTURE_QUALITY,
        );
      });

      const requestStartedAt = performance.now();
      const requestQuery = new URLSearchParams({
        output_format: CAPTURE_EXTENSION,
        outscale: String(UPSCALE_OUTSCALE),
        method: upscaleMethodRef.current,
      });
      const response = await fetch(
        `${API_BASE_URL}/upscale?${requestQuery.toString()}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/octet-stream",
          },
          body: frameBlob,
        },
      );

      if (!response.ok) {
        throw new Error(`Upscale request failed with ${response.status}`);
      }

      const responseBlob = await response.blob();
      const inferenceHeader = response.headers.get("X-Inference-Time-Ms");
      const outputWidth = response.headers.get("X-Output-Width");
      const outputHeight = response.headers.get("X-Output-Height");

      const objectUrl = URL.createObjectURL(responseBlob);
      releaseCurrentFrameUrl();
      lastFrameUrlRef.current = objectUrl;

      const requestEndedAt = performance.now();
      const now = Date.now();
      fpsTimestampsRef.current = fpsTimestampsRef.current
        .filter((timestamp) => now - timestamp < 1000)
        .concat(now);

      if (mountedRef.current) {
        setProcessedFrameUrl(objectUrl);
        setProcessingTimeMs(
          inferenceHeader == null ? null : Number.parseFloat(inferenceHeader),
        );
        setRoundTripMs(requestEndedAt - requestStartedAt);
        setProcessedFps(fpsTimestampsRef.current.length);
        setLastResolution(
          outputWidth && outputHeight ? `${outputWidth}×${outputHeight}` : "n/a",
        );
        setLastError("");
        setBackendStatus(
          `Streaming via ${formatMethodLabel(upscaleMethodRef.current)}`,
        );
      }
    } catch (error) {
      if (mountedRef.current) {
        setLastError(
          error instanceof Error ? error.message : "Frame processing failed.",
        );
      }
    } finally {
      requestInFlightRef.current = false;
      if (cameraActiveRef.current) {
        scheduleNextFrame(0);
      }
    }
  }

  return (
    <div className="page-shell">
      <header className="hero-bar">
        <div>
          <p className="eyebrow">Project Practice</p>
          <h1>Real-Time Super Resolution Console</h1>
        </div>
        <div className="control-row">
          <button
            className="primary-button"
            onClick={cameraActive ? stopCamera : startCamera}
            type="button"
          >
            {cameraActive ? "Stop stream" : "Start camera"}
          </button>
        </div>
      </header>

      <section className="stats-grid">
        <article className="stat-card">
          <span className="stat-label">Current FPS</span>
          <strong>{formatFps(processedFps)}</strong>
        </article>
        <article className="stat-card">
          <span className="stat-label">Input frame</span>
          <strong>
            {processWidth}×{processHeight}
          </strong>
        </article>
        <article className="stat-card">
          <span className="stat-label">Previous processing</span>
          <strong>{formatMs(processingTimeMs)}</strong>
        </article>
        <article className="stat-card">
          <span className="stat-label">Round trip</span>
          <strong>{formatMs(roundTripMs)}</strong>
        </article>
        <article className="stat-card">
          <span className="stat-label">Output frame</span>
          <strong>{lastResolution}</strong>
        </article>
      </section>

      <section className="settings-panel">
        <div className="settings-copy">
          <span className="status-tag">Send Size</span>
          <strong>
            {processWidth}×{processHeight}
          </strong>
          <p>These values and the selected method are applied to the next frame.</p>
        </div>
        <label className="settings-field">
          <span>Method</span>
          <select
            value={upscaleMethod}
            onChange={(event) => setUpscaleMethod(event.target.value)}
          >
            {UPSCALE_METHOD_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label className="settings-field">
          <span>Width</span>
          <input
            type="number"
            min={MIN_PROCESS_DIMENSION}
            max={MAX_PROCESS_DIMENSION}
            step="1"
            inputMode="numeric"
            value={processWidthInput}
            onChange={(event) => setProcessWidthInput(event.target.value)}
            onBlur={commitProcessWidth}
          />
        </label>
        <label className="settings-field">
          <span>Height</span>
          <input
            type="number"
            min={MIN_PROCESS_DIMENSION}
            max={MAX_PROCESS_DIMENSION}
            step="1"
            inputMode="numeric"
            value={processHeightInput}
            onChange={(event) => setProcessHeightInput(event.target.value)}
            onBlur={commitProcessHeight}
          />
        </label>
        <button
          className="secondary-button"
          onClick={resetProcessSize}
          type="button"
        >
          Reset
        </button>
      </section>

      <section className="status-strip">
        <div>
          <span className="status-tag">Source</span>
          <span>{sourceStatus}</span>
        </div>
        <div>
          <span className="status-tag">Backend</span>
          <span>{backendStatus}</span>
        </div>
        <div>
          <span className="status-tag">Preset</span>
          <span>
            {formatMethodLabel(upscaleMethod)}, camera {CAMERA_WIDTH}×
            {CAMERA_HEIGHT}, process {processWidth}×{processHeight}, adaptive
            max fps, x{UPSCALE_OUTSCALE}
          </span>
        </div>
      </section>

      {lastError ? <p className="error-banner">{lastError}</p> : null}

      <main className="frame-grid">
        <article className="frame-card">
          <div className="frame-head">
            <span>Input camera</span>
            <span>{cameraReady ? "live" : "waiting"}</span>
          </div>
          <div className="frame-surface">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="frame-video"
            />
          </div>
        </article>

        <article className="frame-card">
          <div className="frame-head">
            <span>Upscaled output</span>
            <span>{processedFrameUrl ? "receiving" : "waiting"}</span>
          </div>
          <div className="frame-surface">
            {processedFrameUrl ? (
              <img
                alt="Upscaled frame"
                className="frame-image"
                src={processedFrameUrl}
              />
            ) : (
              <div className="frame-placeholder">
                First processed frame will appear here.
              </div>
            )}
          </div>
        </article>
      </main>

      <canvas ref={canvasRef} className="hidden-canvas" />
    </div>
  );
}
