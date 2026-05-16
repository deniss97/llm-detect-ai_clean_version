# OCR-пайплайн

1. пользователь загружает изображение или PDF;
2. `/api/ocr` распознаёт текст через TrOCR;
3. текст можно исправить в интерфейсе;
4. `/api/detect` отправляет исправленный текст в лёгкий локальный детектор;
5. результат сохраняется в PostgreSQL: `id`, `text`, `ai_probability`, `ai_percent`, `verdict`, `confidence`.

## Что используется

OCR сделан по приложенному ноутбуку:

- OpenCV читает изображение;
- изображение переводится в grayscale;
- применяется `cv2.adaptiveThreshold(..., THRESH_BINARY_INV, blockSize=11, C=10)`;
- строки выделяются горизонтальной проекцией;
- каждая строка отправляется в `cyrillic-trocr/trocr-handwritten-cyrillic`;
- результат склеивается построчно.

Детектор AI-текста может работать в старом режиме одной PEFT/LoRA-модели или в режиме ансамбля обновлённых моделей из `MODEL_ZOO.md`.

Старый режим использует адаптер `./models/r_detect_qwen3` поверх `Qwen/Qwen2.5-7B-Instruct`.
Если в папке адаптера есть tokenizer-файлы, backend загружает tokenizer оттуда, а базовую модель берёт из `MODEL_BASE_PATH`.

Настройки в `.env`:

```env
OCR_MODEL_NAME=cyrillic-trocr/trocr-handwritten-cyrillic
OCR_MODEL_LOCAL_FILES_ONLY=false
OCR_PRELOAD_ON_STARTUP=false
OCR_MIN_LINE_HEIGHT=20
OCR_LINE_THRESHOLD_RATIO=0.02
OCR_LINE_PADDING=15
USE_MOCK_DETECTOR=false
MODEL_BASE_PATH=Qwen/Qwen2.5-7B-Instruct
LORA_ADAPTER_PATH=./models/r_detect_qwen3
MODEL_INVERT_PROBABILITY=false
```

Для обновлённого ансамбля:

```env
USE_MOCK_DETECTOR=false
USE_ENSEMBLE_DETECTOR=true
THRESHOLD=0.5
ENSEMBLE_QWEN_ADAPTER_PATH=./models/r_detect_qwen3
ENSEMBLE_T_LITE_ADAPTER_PATH=./models/r_detect_t_lite_v2
```

Ансамбль усредняет score с весами из `MODEL_ZOO.md`. Embedding-модель `r_embed_final` подключается как KNN по embedding-пространству: reference-база берётся из `./datasets/final_prepared/final_train.csv`, а вероятность AI считается по меткам ближайших соседей. Когда embedding-score включён, `ENSEMBLE_USE_META_LEARNER=true` загружает `./models/meta_learner_3models.pkl` через scikit-learn и использует его `predict_proba`.

## Предзагрузка модели

Чтобы OCR-модель не скачивалась во время первого запроса, сначала скачайте её в постоянный Hugging Face cache:

```bash
docker compose run --rm -it ai-detector-web python -u scripts/preload_model.py
```

Для динамического progress bar нужны `-it` и запуск отдельной командой. В `docker compose up` прогресс больших файлов часто отображается плохо, потому что Docker превращает обновления одной строки в обычные логи.

В Docker этот кэш лежит в `./hf-cache`, потому что `docker-compose.yml` монтирует его в `/root/.cache/huggingface`. После успешной загрузки можно запретить приложению ходить в сеть:

```env
OCR_MODEL_LOCAL_FILES_ONLY=true
```

Если в `./hf-cache` остались файлы `*.incomplete`, значит скачивание было прервано. Повторный запуск `scripts/preload_model.py` должен продолжить загрузку.

При `OCR_PRELOAD_ON_STARTUP=true` FastAPI загрузит модель во время старта контейнера. Это делает старт дольше, зато первый пользовательский запрос `/api/ocr` уже не будет скачивать веса.
