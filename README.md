# Веб-сервис проверки и журнала сочинений

1. загрузка изображения или PDF с текстом;
2. OCR-распознавание рукописной кириллицы через TrOCR;
3. ручное редактирование распознанного текста;
4. запуск лёгкой локальной проверки текста;
5. вывод процента риска AI-генерации;
6. сохранение результата в БД: `id`, `text`, `ai_probability`, `verdict`, `created_at`.
7. сохранение сочинения в журнал класса с оценкой, типом работы и цветовой пометкой риска AI-генерации.

## Быстрый запуск без Docker

### 1. Установите зависимости

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

По умолчанию приложение использует PostgreSQL. Для запуска без Docker поднимите локальный PostgreSQL и укажите `DATABASE_URL`, например:

```env
DATABASE_URL=postgresql+psycopg://ai_detector:ai_detector@localhost:5432/ai_detector
```

OCR-модель по умолчанию: `cyrillic-trocr/trocr-handwritten-cyrillic`. При первом распознавании она скачивается из Hugging Face. PDF-файлы распознаются постранично: приложение рендерит страницы в изображения и затем запускает тот же TrOCR-пайплайн. Ограничения задаются переменными `PDF_MAX_PAGES` и `PDF_RENDER_DPI`.

Первый запуск OCR может выглядеть как долгий запрос в браузере, потому что скачивается `model.safetensors`. Чтобы не ждать внутри интерфейса, заранее выполните предзагрузку из раздела ниже.

### 2. Запустите сервис

```bash
uvicorn app.main:app --reload
```

Откройте: http://127.0.0.1:8000

Swagger API: http://127.0.0.1:8000/docs

## Журнал и роли

Во вкладке `Журнал` оценки отображаются таблицей по датам. Для сочинений цвет оценки зависит от процента AI-генерации:

- красный: `>= THRESHOLD * 100`, высокий риск генерации;
- жёлтый: от `40%` до порога, возможна генерация;
- зелёный: ниже `40%`, генерация не обнаружена.

Наведение на оценку показывает тип работы и комментарий. Клик по оценке сочинения открывает окно с оцифровкой текста и результатами модели. Для длинного периода есть выбор диапазона `14/30/90/180 дней`, таблица прокручивается горизонтально и считает средний балл.

Обычные оценки можно добавлять прямо из вкладки журнала: ответ на уроке, диктант, контрольная или другая работа. Такие оценки отображаются нейтральным цветом, потому что к ним не привязан результат AI-детекции.

Новые классы можно импортировать во вкладке `Журнал` из CSV. Поддерживаются заголовки `class`/`class_name`/`класс` и `student`/`full_name`/`фио`; без заголовков ожидаются две колонки: класс, ученик. Импортированные классы привязываются к текущему пользователю и по умолчанию видны только ему.

Пример CSV:

```csv
class,student
10А,Иванова Мария Дмитриевна
10А,Петров Иван Сергеевич
11Б,Сидорова Анна Сергеевна
```

или:

```csv
класс,фио
10А,Иванова Мария Дмитриевна
10А,Петров Иван Сергеевич
```

Чтобы дать доступ к классу другому пользователю через Keycloak:

1. Откройте Keycloak Admin Console.
2. Перейдите в realm `school` → `Users` → нужный пользователь → `Attributes`.
3. Добавьте атрибут `school_classes`.
4. В значение укажите `access_key` класса. По умолчанию это название класса, например `10А`.
5. Для нескольких классов добавьте несколько значений атрибута или перечислите их через запятую, например `10А,11Б`.
6. Пользователю нужно выйти и войти заново, чтобы новый атрибут попал в JWT.

Без авторизации приложение работает от имени демо-учителя `DEMO_TEACHER_USERNAME=demo.teacher` и показывает только его классы. При включённом Keycloak открытие http://127.0.0.1:8000 сразу перенаправляет на страницу входа Keycloak. После успешного входа backend сохраняет сессию в cookie, в правом верхнем углу UI показывает аккаунт пользователя и даёт выйти через меню аккаунта.

Учитель видит классы по данным Keycloak: если он создал класс, совпадает `owner_subject` класса с `sub` пользователя; если доступ выдан атрибутом Keycloak, `access_key` класса должен быть в claim/user attribute `school_classes`. Пользователь с ролью `ADMIN_ROLE` видит все классы.

## Keycloak

В Docker Keycloak поднимается отдельным сервисом `keycloak` на http://localhost:8080. Realm импортируется из `keycloak/realm-school.json`.

Демо-пользователи:

- `demo.teacher` / `teacher` - роль `teacher`, видит класс `8А`;
- `second.teacher` / `teacher` - роль `teacher`, видит класс `9Б`;
- `admin.teacher` / `admin` - роль `admin`, видит все классы.

Админка Keycloak: http://localhost:8080/admin, логин `admin`, пароль `admin`.

При ручной настройке Keycloak:

1. Создайте realm `school` и public client `ai-detector-web` для браузерного входа по Authorization Code + PKCE.
2. В настройках клиента добавьте `Valid redirect URIs`, например:

```text
http://127.0.0.1:8000/*
```

3. Добавьте Web origins для адреса приложения, например:

```text
http://127.0.0.1:8000
```

4. В `.env` включите авторизацию:

```env
AUTH_ENABLED=true
KEYCLOAK_PUBLIC_BASE_URL="http://localhost:8080"
KEYCLOAK_INTERNAL_BASE_URL="http://keycloak:8080"
KEYCLOAK_REALM="school"
KEYCLOAK_CLIENT_ID="ai-detector-web"
ADMIN_ROLE="admin"
ADMIN_USERNAMES="admin.teacher,admin.teacher@example.test"
TEACHER_ROLE="teacher"
```

`KEYCLOAK_PUBLIC_BASE_URL` нужен браузеру для редиректа на страницу входа. `KEYCLOAK_INTERNAL_BASE_URL` нужен backend-контейнеру, чтобы скачать JWKS-ключи Keycloak внутри docker compose сети.

По умолчанию `KEYCLOAK_AUDIENCE` пустой, поэтому backend не проверяет `aud` в access token. Если хотите строгую проверку audience, настройте audience mapper в Keycloak и укажите значение в `KEYCLOAK_AUDIENCE`.

## Запуск в Docker

```bash
docker compose up --build
```

Docker Compose поднимает PostgreSQL, Keycloak, Redis, FastAPI web-сервис, отдельный Celery worker и Nginx.

Схема контейнеров:

- `nginx` - внешняя точка входа, слушает `localhost:80` и проксирует запросы в FastAPI;
- `ai-detector-web` - FastAPI backend, авторизация Keycloak, REST API и UI;
- `ai-detector-worker` - отдельный inference-слой для тяжёлых OCR и AI-detection задач;
- `redis` - брокер очереди Celery, доступен внутри compose-сети как `redis:6379`;
- `postgres` - основная БД SQLAlchemy, хранит результаты, журнал и таблицу `jobs`;
- `keycloak` - авторизация пользователей и роли.

PostgreSQL доступен внутри compose-сети как `postgres:5432`, а с хоста как `localhost:5432`.

Основной адрес приложения через reverse proxy:

```text
http://localhost/
```

Прямой порт FastAPI `8000` оставлен для отладки:

```text
http://127.0.0.1:8000
```

Проверка health endpoint:

```bash
curl http://localhost/api/health
```

### Асинхронные задачи OCR и AI-детекции

Тяжёлые операции теперь можно запускать через очередь. FastAPI создаёт запись в таблице `jobs`, отправляет задачу в Celery через Redis, а worker выполняет OCR или detector inference и обновляет статус в PostgreSQL.

Статусы задач:

- `queued` - задача создана и отправлена в очередь;
- `running` - worker начал выполнение;
- `finished` - задача завершилась, поле `result` содержит результат;
- `failed` - задача завершилась ошибкой, поле `error_message` содержит текст ошибки.

Пример async detect job:

```bash
curl -X POST http://localhost/api/detect/jobs \
  -H "Content-Type: application/json" \
  -d '{"text":"Это пример достаточно длинного текста для проверки через асинхронную очередь."}'
```

Ответ содержит `job_id`:

```json
{
  "job_id": 1,
  "status": "queued",
  "task_type": "detect"
}
```

Проверка статуса:

```bash
curl http://localhost/api/jobs/1
```

Пример async OCR job:

```bash
curl -X POST http://localhost/api/ocr/jobs \
  -F "image=@essay.pdf"
```

Синхронные endpoints `/api/ocr` и `/api/detect` сохранены для совместимости.

## Модель детекции

Детектор подключён как LoRA-адаптер `./models/r_detect_qwen3` поверх базовой модели `Qwen/Qwen2.5-7B-Instruct`.
Папка адаптера смонтирована в Docker через `./models:/app/models`, поэтому для реального детектора в `.env` должны быть включены:

```env
USE_MOCK_DETECTOR=false
MODEL_BASE_PATH=Qwen/Qwen2.5-7B-Instruct
LORA_ADAPTER_PATH=./models/r_detect_qwen3
MODEL_INVERT_PROBABILITY=false
```

При первом запуске базовая Qwen-модель скачивается в `./hf-cache`. После успешной загрузки можно поставить `MODEL_LOCAL_FILES_ONLY=true`, чтобы приложение не ходило в сеть за весами.

### Ансамбль обновлённых моделей

Локальные веса из `MODEL_ZOO.md` ожидаются в папке `./models`:

- `./models/r_detect_qwen3`
- `./models/r_detect_t_lite_v2`
- `./models/r_embed_final`
- `./models/meta_learner_3models.pkl`
- `./datasets/final_prepared/final_train.csv`

Для запуска ансамбля из двух detection-моделей:

```env
USE_MOCK_DETECTOR=false
USE_ENSEMBLE_DETECTOR=true
THRESHOLD=0.5
ENSEMBLE_QWEN_BASE_PATH=Qwen/Qwen2.5-7B-Instruct
ENSEMBLE_QWEN_ADAPTER_PATH=./models/r_detect_qwen3
ENSEMBLE_T_LITE_BASE_PATH=t-tech/T-lite-it-1.0
ENSEMBLE_T_LITE_ADAPTER_PATH=./models/r_detect_t_lite_v2
```

Embedding-модель `USER-bge-m3` обучена как ranking/embedding-модель. В inference она используется как KNN по embedding-пространству: тексты из `final_train.csv` кодируются как reference-база, а вероятность AI считается по меткам ближайших соседей.

```env
EMBEDDING_ENABLED=true
EMBEDDING_REFERENCE_DATASET_PATH=./datasets/final_prepared/final_train.csv
EMBEDDING_KNN_K=5
```

После этого можно включить logistic meta learner из `meta_learner_3models.pkl`:

```env
ENSEMBLE_USE_META_LEARNER=true
ENSEMBLE_META_LEARNER_PATH=./models/meta_learner_3models.pkl
ENSEMBLE_META_FEATURE_ORDER=t_lite,qwen,embed
```

## Предзагрузка TrOCR

Docker монтирует `./hf-cache` в контейнер как кэш Hugging Face, а `.dockerignore` исключает этот кэш из сборки образа. Поэтому веса не должны скачиваться заново или копироваться в image после каждой пересборки. Чтобы скачать их заранее:

```bash
docker compose run --rm -it ai-detector-web python -u scripts/preload_model.py
```

Флаги `-it` дают контейнеру интерактивный терминал, а `python -u` отключает буферизацию вывода. В таком режиме прогресс Hugging Face/TQDM обновляется динамически в одной строке. В обычном `docker compose up` Docker пишет логи построчно, поэтому progress bar часто выглядит как зависший или плохо перерисовывается.

После успешной предзагрузки можно включить режим без сетевых запросов:

```env
OCR_MODEL_LOCAL_FILES_ONLY=true
```

Если хотите, чтобы приложение загружало модель сразу при старте, а не при первом файле:

```env
OCR_PRELOAD_ON_STARTUP=true
```

Важно: это убирает повторное скачивание, но не отменяет загрузку модели с диска в память при каждом старте контейнера. Если контейнер пересоздаётся или приложение запускается с `--reload`, TrOCR всё равно будет заново загружаться в RAM/VRAM.

Если загрузка зависает на `Fetching files`, обычно это недокачанные файлы модели. В проекте для Docker отключён Xet downloader (`HF_HUB_DISABLE_XET=1`), после этого повторите команду предзагрузки.
