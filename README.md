# Веб-сервис детекции AI-сгенерированных текстов

1. загрузка изображения с текстом;
2. OCR-распознавание в текстовый формат;
3. ручное редактирование распознанного текста;
4. запуск модели детекции;
5. вывод процента AI-генерации;
6. сохранение результата в БД: `id`, `text`, `ai_probability`, `verdict`, `created_at`.

## Быстрый запуск без Docker

### 1. Установите Tesseract OCR

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-rus
```

macOS:

```bash
brew install tesseract tesseract-lang
```

### 2. Установите зависимости

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 3. Запустите сервис

```bash
uvicorn app.main:app --reload
```

Откройте: http://127.0.0.1:8000

Swagger API: http://127.0.0.1:8000/docs

## Подключение реальной модели детекции

По умолчанию включена заглушка `USE_MOCK_DETECTOR=true`, чтобы веб-часть запускалась без весов модели.

Для подключения `r_detect_retrain` из `llm-detect-ai`:

```env
USE_MOCK_DETECTOR=false
MODEL_BASE_PATH=mistralai/Mistral-7B-v0.1
LORA_ADAPTER_PATH=./models/r_detect_retrain/best
THRESHOLD=0.73
MODEL_MAX_LENGTH=256
AI_CLASS_INDEX=1
LOAD_IN_4BIT=true
```

После этого эндпоинт `/api/detect` будет использовать реальную модель.

Подробно: [docs/ML_INTEGRATION.md](docs/ML_INTEGRATION.md).

## Запуск в Docker

```bash
docker compose up --build
```

Сервис будет доступен на http://127.0.0.1:8000

## Предзагрузка Mistral

Docker монтирует `./hf-cache` в контейнер как кэш Hugging Face, а `.dockerignore` исключает этот кэш из сборки образа. Поэтому веса не должны скачиваться заново или копироваться в image после каждой пересборки. Чтобы скачать их заранее:

```bash
docker compose run --rm -it ai-detector-web python -u scripts/preload_model.py
```

Флаги `-it` дают контейнеру интерактивный терминал, а `python -u` отключает буферизацию вывода. В таком режиме прогресс Hugging Face/TQDM обновляется динамически в одной строке. В обычном `docker compose up` Docker пишет логи построчно, поэтому progress bar часто выглядит как зависший или плохо перерисовывается.

После успешной предзагрузки можно включить режим без сетевых запросов:

```env
MODEL_LOCAL_FILES_ONLY=true
```

Важно: это убирает повторное скачивание, но не отменяет загрузку модели с диска в память при каждом старте контейнера. Если контейнер пересоздаётся или приложение запускается с `--reload`, Mistral всё равно будет заново загружаться в RAM/VRAM.

Если загрузка зависает на `Fetching 2 files`, обычно это недокачанные shard-файлы модели. В проекте для Docker отключён Xet downloader (`HF_HUB_DISABLE_XET=1`), после этого повторите команду предзагрузки.
