# Подключение ML-части

1. пользователь загружает изображение или PDF;
2. `/api/ocr` распознаёт текст через Tesseract;
3. текст можно исправить в интерфейсе;
4. `/api/detect` отправляет исправленный текст в детектор;
5. результат сохраняется в SQLite: `id`, `text`, `ai_probability`, `ai_percent`, `verdict`, `confidence`.

## Что взять из `llm-detect-ai`

В отчёте `DETECTION_METRICS_REPORT.md` самый простой вариант для подключения к этому сервису - `r_detect_retrain`:

- base model: `mistralai/Mistral-7B-v0.1`;
- LoRA adapter: `models/r_detect_retrain/best`;
- max length: `256`;
- рекомендуемый threshold: `0.73`.

Именно под такой формат в проекте уже есть `TransformerDetector` в `app/services/detector.py`.

## Как включить реальную модель

1. Положите веса рядом с веб-проектом, например:

```text
ai_text_detector_web/
├── models/
│   └── r_detect_retrain/
│       └── best/
│           ├── adapter_config.json
│           └── adapter_model.safetensors
```

2. Создайте `.env` из примера:

```bash
cp .env.example .env
```

3. Укажите модель и выключите mock:

```env
USE_MOCK_DETECTOR=false
MODEL_BASE_PATH=mistralai/Mistral-7B-v0.1
LORA_ADAPTER_PATH=./models/r_detect_retrain/best
THRESHOLD=0.73
MODEL_MAX_LENGTH=256
AI_CLASS_INDEX=1
MODEL_INVERT_PROBABILITY=false
LOAD_IN_4BIT=true
```

4. Запустите сервис:

```bash
uvicorn app.main:app --reload
```

При старте приложение загрузит токенизатор, базовую модель и LoRA-адаптер. После этого `/api/detect` будет возвращать вероятность AI-генерации из реальной модели, а не из заглушки.

Если LoRA adapter сохранён с classification head на один выход, сервис автоматически создаёт base model с `num_labels=1` и считает вероятность через sigmoid. В этом режиме `AI_CLASS_INDEX` не используется.

Если на заведомо AI-тексте модель стабильно выдаёт почти `0%`, а на человеческом почти `100%`, включите `MODEL_INVERT_PROBABILITY=true`. Это инвертирует итоговую вероятность без изменения весов модели.

## Предзагрузка base model

Чтобы Mistral не скачивалась во время старта приложения, сначала скачайте её в постоянный Hugging Face cache:

```bash
docker compose run --rm -it ai-detector-web python -u scripts/preload_model.py
```

Для динамического progress bar нужны `-it` и запуск отдельной командой. В `docker compose up` прогресс больших файлов часто отображается плохо, потому что Docker превращает обновления одной строки в обычные логи.

В Docker этот кэш лежит в `./hf-cache`, потому что `docker-compose.yml` монтирует его в `/root/.cache/huggingface`. После успешной загрузки можно запретить приложению ходить в сеть:

```env
MODEL_LOCAL_FILES_ONLY=true
```

Если в `./hf-cache` остались файлы `*.incomplete`, значит скачивание было прервано. Повторный запуск `scripts/preload_model.py` должен продолжить загрузку.

## Если подключать лучший ensemble / Embedding KNN

Embedding KNN и ensemble из отчёта не являются одной обычной `AutoModelForSequenceClassification`: там нужны embedding-модель, обученный KNN/мета-классификатор и сохранённые артефакты. Для них лучше сделать отдельный ML-сервис с контрактом:

```http
POST /predict
Content-Type: application/json

{"text": "Текст для проверки"}
```

Ответ:

```json
{"ai_probability": 0.94}
```

После этого в веб-проекте достаточно добавить ещё одну реализацию протокола `Detector`, которая вызывает этот ML-сервис по HTTP. Такой вариант чище: веб-сервис отвечает за загрузку, OCR, БД и UI, а ML-сервис - за тяжёлую модель и её зависимости.
