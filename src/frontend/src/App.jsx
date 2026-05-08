import { useEffect, useRef, useState } from "react";

const DEFAULT_API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";
const API_URL_STORAGE_KEY = "realesrgan.apiBaseUrl";
const MODEL_ID_STORAGE_KEY = "realesrgan.modelId";
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
const FALLBACK_MODEL_OPTIONS = [
  { id: "realesrgan_x4plus", name: "Real-ESRGAN x4plus", kind: "torch" },
  { id: "srcnn_rgb", name: "SRCNN RGB", kind: "torch" },
  { id: "bicubic", name: "Бикубическая интерполяция", kind: "bicubic" },
];
const FALLBACK_MODEL_ID = FALLBACK_MODEL_OPTIONS[0].id;

function normalizeApiBaseUrl(value) {
  return value.trim().replace(/\/$/, "");
}

function readStoredApiBaseUrl() {
  try {
    return window.localStorage.getItem(API_URL_STORAGE_KEY);
  } catch {
    return null;
  }
}

function writeStoredApiBaseUrl(value) {
  try {
    window.localStorage.setItem(API_URL_STORAGE_KEY, value);
  } catch {
    // The app can still use the value for the current session.
  }
}

function readStoredModelId() {
  try {
    return window.localStorage.getItem(MODEL_ID_STORAGE_KEY);
  } catch {
    return null;
  }
}

function writeStoredModelId(value) {
  try {
    window.localStorage.setItem(MODEL_ID_STORAGE_KEY, value);
  } catch {
    // The selected model still applies for the current session.
  }
}

function getInitialApiBaseUrl() {
  const query = new URLSearchParams(window.location.search);
  return normalizeApiBaseUrl(
    query.get("api") ??
      query.get("apiBaseUrl") ??
      readStoredApiBaseUrl() ??
      DEFAULT_API_BASE_URL,
  );
}

function formatMs(value) {
  return value == null ? "н/д" : `${value.toFixed(1)} мс`;
}

function formatFps(value) {
  return value <= 0 ? "0.0 к/с" : `${value.toFixed(1)} к/с`;
}

function formatModelLabel(value, models) {
  return models.find((model) => model.id === value)?.name ?? value;
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

function getUpscaledDimension(value) {
  return Math.max(1, Math.round(value * UPSCALE_OUTSCALE));
}

export default function App() {
  const videoRef = useRef(null);
  const previewCanvasRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const requestInFlightRef = useRef(false);
  const cameraActiveRef = useRef(false);
  const loopTimerRef = useRef(null);
  const previewFrameRef = useRef(null);
  const lastFrameUrlRef = useRef(null);
  const fpsTimestampsRef = useRef([]);
  const mountedRef = useRef(true);
  const apiBaseUrlRef = useRef(getInitialApiBaseUrl());
  const processWidthRef = useRef(DEFAULT_PROCESS_WIDTH);
  const processHeightRef = useRef(DEFAULT_PROCESS_HEIGHT);
  const selectedModelIdRef = useRef(readStoredModelId() ?? FALLBACK_MODEL_ID);

  const [apiBaseUrl, setApiBaseUrl] = useState(apiBaseUrlRef.current);
  const [apiBaseUrlInput, setApiBaseUrlInput] = useState(apiBaseUrlRef.current);
  const [modelOptions, setModelOptions] = useState(FALLBACK_MODEL_OPTIONS);
  const [selectedModelId, setSelectedModelId] = useState(
    selectedModelIdRef.current,
  );
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [sourceStatus, setSourceStatus] = useState("Ожидание");
  const [backendStatus, setBackendStatus] = useState("Проверка backend...");
  const [processedFrameUrl, setProcessedFrameUrl] = useState(null);
  const [processingTimeMs, setProcessingTimeMs] = useState(null);
  const [roundTripMs, setRoundTripMs] = useState(null);
  const [processedFps, setProcessedFps] = useState(0);
  const [lastResolution, setLastResolution] = useState("н/д");
  const [lastError, setLastError] = useState("");
  const [processWidthInput, setProcessWidthInput] = useState(
    String(DEFAULT_PROCESS_WIDTH),
  );
  const [processHeightInput, setProcessHeightInput] = useState(
    String(DEFAULT_PROCESS_HEIGHT),
  );

  const processWidth = normalizeProcessDimension(
    processWidthInput,
    DEFAULT_PROCESS_WIDTH,
  );
  const processHeight = normalizeProcessDimension(
    processHeightInput,
    DEFAULT_PROCESS_HEIGHT,
  );
  const frameAspectRatio = `${processWidth} / ${processHeight}`;

  useEffect(() => {
    mountedRef.current = true;
    void checkBackendHealth();

    return () => {
      mountedRef.current = false;
      stopProcessingLoop();
      stopPreviewLoop();
      stopCameraTracks();
      clearPreviewCanvas();
      releaseCurrentFrameUrl();
    };
  }, []);

  useEffect(() => {
    apiBaseUrlRef.current = apiBaseUrl;
  }, [apiBaseUrl]);

  useEffect(() => {
    processWidthRef.current = processWidth;
  }, [processWidth]);

  useEffect(() => {
    processHeightRef.current = processHeight;
  }, [processHeight]);

  useEffect(() => {
    selectedModelIdRef.current = selectedModelId;
    writeStoredModelId(selectedModelId);
  }, [selectedModelId]);

  async function checkBackendHealth(baseUrl = apiBaseUrlRef.current) {
    try {
      const [healthResponse, modelsResponse] = await Promise.all([
        fetch(`${baseUrl}/health`),
        fetch(`${baseUrl}/models`),
      ]);
      if (!healthResponse.ok) {
        throw new Error(`Backend ответил кодом ${healthResponse.status}`);
      }
      const payload = await healthResponse.json();
      if (modelsResponse.ok) {
        const catalog = await modelsResponse.json();
        const models = catalog.models?.length
          ? catalog.models
          : FALLBACK_MODEL_OPTIONS;
        const currentExists = models.some(
          (model) => model.id === selectedModelIdRef.current,
        );
        const nextModelId = currentExists
          ? selectedModelIdRef.current
          : catalog.default_model_id;
        if (mountedRef.current) {
          setModelOptions(models);
          setSelectedModelId(nextModelId);
        }
      }
      if (mountedRef.current) {
        setBackendStatus(
          `Backend готов: ${payload.model_name}, устройство ${payload.device}`,
        );
      }
    } catch (error) {
      if (mountedRef.current) {
        setBackendStatus(
          error instanceof Error
            ? error.message
            : "Не удалось проверить backend.",
        );
      }
    }
  }

  function commitApiBaseUrl() {
    const nextApiBaseUrl = normalizeApiBaseUrl(
      apiBaseUrlInput || DEFAULT_API_BASE_URL,
    );
    apiBaseUrlRef.current = nextApiBaseUrl;
    setApiBaseUrl(nextApiBaseUrl);
    setApiBaseUrlInput(nextApiBaseUrl);
    writeStoredApiBaseUrl(nextApiBaseUrl);
    setBackendStatus("Проверка backend...");
    void checkBackendHealth(nextApiBaseUrl);
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

  function stopPreviewLoop() {
    if (previewFrameRef.current) {
      window.cancelAnimationFrame(previewFrameRef.current);
      previewFrameRef.current = null;
    }
  }

  function releaseCurrentFrameUrl() {
    if (lastFrameUrlRef.current) {
      URL.revokeObjectURL(lastFrameUrlRef.current);
      lastFrameUrlRef.current = null;
    }
  }

  function clearPreviewCanvas() {
    const previewCanvasElement = previewCanvasRef.current;
    const context = previewCanvasElement?.getContext("2d");
    if (!previewCanvasElement || !context) {
      return;
    }
    context.clearRect(
      0,
      0,
      previewCanvasElement.width,
      previewCanvasElement.height,
    );
  }

  function drawSourcePreview() {
    const videoElement = videoRef.current;
    const previewCanvasElement = previewCanvasRef.current;

    if (
      !cameraActiveRef.current ||
      !videoElement ||
      !previewCanvasElement ||
      videoElement.readyState < 2 ||
      videoElement.videoWidth === 0
    ) {
      previewFrameRef.current = window.requestAnimationFrame(drawSourcePreview);
      return;
    }

    const context = previewCanvasElement.getContext("2d");
    if (context) {
      const previewWidth = getUpscaledDimension(processWidthRef.current);
      const previewHeight = getUpscaledDimension(processHeightRef.current);

      if (previewCanvasElement.width !== previewWidth) {
        previewCanvasElement.width = previewWidth;
      }
      if (previewCanvasElement.height !== previewHeight) {
        previewCanvasElement.height = previewHeight;
      }

      context.drawImage(videoElement, 0, 0, previewWidth, previewHeight);
    }

    previewFrameRef.current = window.requestAnimationFrame(drawSourcePreview);
  }

  function startPreviewLoop() {
    stopPreviewLoop();
    previewFrameRef.current = window.requestAnimationFrame(drawSourcePreview);
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
    setSourceStatus("Запрос доступа к камере...");

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
        throw new Error("Видеоэлемент недоступен.");
      }

      videoElement.srcObject = stream;
      await videoElement.play();

      cameraActiveRef.current = true;
      setCameraReady(true);
      setCameraActive(true);
      setSourceStatus("Камера включена");
      startPreviewLoop();
      scheduleNextFrame(0);
    } catch (error) {
      cameraActiveRef.current = false;
      stopPreviewLoop();
      stopCameraTracks();
      clearPreviewCanvas();
      setCameraReady(false);
      setCameraActive(false);
      setSourceStatus("Камера недоступна");
      setLastError(
        error instanceof Error ? error.message : "Не удалось запустить камеру.",
      );
    }
  }

  function stopCamera() {
    cameraActiveRef.current = false;
    stopProcessingLoop();
    stopPreviewLoop();
    stopCameraTracks();
    clearPreviewCanvas();
    setCameraActive(false);
    setCameraReady(false);
    setSourceStatus("Поток остановлен");
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
        throw new Error("2D-контекст canvas недоступен.");
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
            reject(new Error("Не удалось закодировать кадр с камеры."));
          },
          CAPTURE_MIME_TYPE,
          OUTPUT_FORMAT === "png" ? undefined : CAPTURE_QUALITY,
        );
      });

      const requestStartedAt = performance.now();
      const requestQuery = new URLSearchParams({
        output_format: CAPTURE_EXTENSION,
        outscale: String(UPSCALE_OUTSCALE),
        model_id: selectedModelIdRef.current,
      });
      const response = await fetch(
        `${apiBaseUrlRef.current}/upscale?${requestQuery.toString()}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/octet-stream",
          },
          body: frameBlob,
        },
      );

      if (!response.ok) {
        throw new Error(`Запрос апскейлинга завершился кодом ${response.status}`);
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
          outputWidth && outputHeight ? `${outputWidth}x${outputHeight}` : "н/д",
        );
        setLastError("");
        setBackendStatus(
          `Поток через ${formatModelLabel(
            selectedModelIdRef.current,
            modelOptions,
          )}`,
        );
      }
    } catch (error) {
      if (mountedRef.current) {
        setLastError(
          error instanceof Error ? error.message : "Не удалось обработать кадр.",
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
          <p className="eyebrow">Учебная практика</p>
          <h1>Консоль повышения разрешения в реальном времени</h1>
        </div>
        <div className="control-row">
          <button
            className="primary-button"
            onClick={cameraActive ? stopCamera : startCamera}
            type="button"
          >
            {cameraActive ? "Остановить поток" : "Запустить камеру"}
          </button>
        </div>
      </header>

      <section className="stats-grid">
        <article className="stat-card">
          <span className="stat-label">Текущий FPS</span>
          <strong>{formatFps(processedFps)}</strong>
        </article>
        <article className="stat-card">
          <span className="stat-label">Входной кадр</span>
          <strong>
            {processWidth}x{processHeight}
          </strong>
        </article>
        <article className="stat-card">
          <span className="stat-label">Последняя обработка</span>
          <strong>{formatMs(processingTimeMs)}</strong>
        </article>
        <article className="stat-card">
          <span className="stat-label">Полный цикл</span>
          <strong>{formatMs(roundTripMs)}</strong>
        </article>
        <article className="stat-card">
          <span className="stat-label">Выходной кадр</span>
          <strong>{lastResolution}</strong>
        </article>
      </section>

      <section className="settings-panel">
        <div className="settings-copy">
          <span className="status-tag">Размер отправки</span>
          <strong>
            {processWidth}x{processHeight}
          </strong>
          <p>Эти значения и выбранная модель применятся к следующему кадру.</p>
        </div>
        <label className="settings-field">
          <span>Модель</span>
          <select
            value={selectedModelId}
            onChange={(event) => setSelectedModelId(event.target.value)}
          >
            {modelOptions.map((option) => (
              <option key={option.id} value={option.id}>
                {option.name}
              </option>
            ))}
          </select>
        </label>
        <label className="settings-field api-url-field">
          <span>URL backend</span>
          <input
            type="url"
            value={apiBaseUrlInput}
            onChange={(event) => setApiBaseUrlInput(event.target.value)}
            onBlur={commitApiBaseUrl}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.currentTarget.blur();
              }
            }}
          />
        </label>
        <label className="settings-field">
          <span>Ширина</span>
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
          <span>Высота</span>
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
          Сбросить
        </button>
      </section>

      <section className="status-strip">
        <div>
          <span className="status-tag">Источник</span>
          <span>{sourceStatus}</span>
        </div>
        <div>
          <span className="status-tag">Backend</span>
          <span>{backendStatus}</span>
        </div>
        <div>
          <span className="status-tag">Параметры</span>
          <span>
            {formatModelLabel(selectedModelId, modelOptions)}, камера{" "}
            {CAMERA_WIDTH}x{CAMERA_HEIGHT}, обработка {processWidth}x
            {processHeight}, API {apiBaseUrl}, адаптивный максимум FPS, x
            {UPSCALE_OUTSCALE}
          </span>
        </div>
      </section>

      {lastError ? <p className="error-banner">{lastError}</p> : null}

      <main className="frame-grid">
        <article className="frame-card">
          <div className="frame-head">
            <span>Камера</span>
            <span>{cameraReady ? "в эфире" : "ожидание"}</span>
          </div>
          <div className="frame-surface" style={{ aspectRatio: frameAspectRatio }}>
            <canvas ref={previewCanvasRef} className="frame-canvas" />
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="stream-source-video"
            />
            {!cameraReady ? (
              <div className="frame-placeholder">
                Поток камеры появится здесь.
              </div>
            ) : null}
          </div>
        </article>

        <article className="frame-card">
          <div className="frame-head">
            <span>Результат апскейлинга</span>
            <span>{processedFrameUrl ? "получение" : "ожидание"}</span>
          </div>
          <div className="frame-surface" style={{ aspectRatio: frameAspectRatio }}>
            {processedFrameUrl ? (
              <img
                alt="Кадр после апскейлинга"
                className="frame-image"
                src={processedFrameUrl}
              />
            ) : (
              <div className="frame-placeholder">
                Первый обработанный кадр появится здесь.
              </div>
            )}
          </div>
        </article>
      </main>

      <canvas ref={canvasRef} className="hidden-canvas" />
    </div>
  );
}
