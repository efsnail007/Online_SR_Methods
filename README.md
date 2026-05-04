# Online_SR_Methods

Web-приложение для повышения разрешения кадров с камеры в реальном времени.
Backend построен на FastAPI, frontend - на React/Vite.

## Запуск

Все основные переменные окружения лежат в корневом `.env`.

Backend через Docker:

```powershell
docker compose build --no-cache backend-cpu
docker compose up backend-cpu
```

CUDA-вариант:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
docker compose build --no-cache backend-cuda
docker compose up backend-cuda
```

Frontend:

```powershell
docker compose up frontend
```

Или локально:

```powershell
cd src/frontend
npm install
npm run dev -- --host 0.0.0.0
```

По умолчанию frontend ждёт API на `http://localhost:8000/api/v1`.
Адрес можно поменять в `.env`, через query-параметр
`http://localhost:5173/?api=http://192.168.1.10:8000/api/v1`,
или прямо в поле `Backend URL` в интерфейсе.

## Модели

Backend теперь использует каталог моделей. UI получает список через:

```text
GET /api/v1/models
```

Инференс принимает `model_id`:

```text
POST /api/v1/upscale?model_id=bicubic&outscale=4&output_format=jpeg
```

Минимальная конфигурация задаётся через `.env`:

```dotenv
BACKEND_DEFAULT_MODEL_ID=realesrgan_x4plus
BACKEND_STARTUP_MODEL_IDS=realesrgan_x4plus
BACKEND_MODEL_WEIGHTS_PATH=src/backend/assets/weights/RealESRGAN_x4plus.pth
# BACKEND_MODEL_CATALOG_PATH=src/backend/models.json
```

Если `BACKEND_MODEL_CATALOG_PATH` не задан, backend автоматически регистрирует:

- `realesrgan_x4plus` - PyTorch Real-ESRGAN x4plus;
- `bicubic` - встроенный baseline без весов.

Пример каталога лежит в `src/backend/models.example.json`.
Чтобы добавить ONNX-модель, скопируйте пример в `src/backend/models.json`,
укажите путь к `.onnx`, включите переменную `BACKEND_MODEL_CATALOG_PATH` и
установите optional extra:

```powershell
poetry install --extras onnx
```

Поддерживаемые `kind`:

- `bicubic` - встроенный bicubic runtime;
- `torch` с `architecture=realesrgan_x4plus` - текущая PyTorch-модель;
- `onnx` - generic NCHW RGB runtime через `onnxruntime`.

Новые форматы добавляются отдельным runtime-классом в
`src/backend/app/ml/model_runtime.py` и одной веткой в `create_runtime`.

## Poetry-варианты PyTorch

```powershell
poetry install --extras cpu
poetry install --extras cuda
```

В Docker это разведено по файлам:

- `src/backend/Dockerfile.cpu` устанавливает CPU-зависимости;
- `src/backend/Dockerfile.cuda` устанавливает CUDA-зависимости.
