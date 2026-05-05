# Веб-сервис детекции AI-сгенерированных текстов

1. загрузка изображения с текстом;
2. OCR-распознавание в текстовый формат;
3. ручное редактирование распознанного текста;
4. запуск модели детекции;
5. вывод процента AI-генерации;
6. сохранение результата в БД: `id`, `text`, `ai_probability`, `verdict`, `created_at`.
7. сохранение сочинения в журнал класса с оценкой, типом работы и цветовой пометкой риска AI-генерации.

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

## Журнал и роли

Во вкладке `Журнал` оценки отображаются таблицей по датам. Для сочинений цвет оценки зависит от процента AI-генерации:

- красный: `>= THRESHOLD * 100`, высокий риск генерации;
- жёлтый: от `40%` до порога, возможна генерация;
- зелёный: ниже `40%`, генерация не обнаружена.

Наведение на оценку показывает тип работы и комментарий. Клик по оценке сочинения открывает окно с оцифровкой текста и результатами модели. Для длинного периода есть выбор диапазона `14/30/90/180 дней`, таблица прокручивается горизонтально и считает средний балл.

Обычные оценки можно добавлять прямо из вкладки журнала: ответ на уроке, диктант, контрольная или другая работа. Такие оценки отображаются нейтральным цветом, потому что к ним не привязан результат AI-детекции.

Без авторизации приложение работает от имени демо-учителя `DEMO_TEACHER_USERNAME=demo.teacher` и показывает только его классы. При включённом Keycloak открытие http://127.0.0.1:8000 сразу перенаправляет на страницу входа Keycloak. После успешного входа backend сохраняет сессию в cookie, в правом верхнем углу UI показывает аккаунт пользователя и даёт выйти через меню аккаунта.

Учитель видит только классы, где `school_classes.teacher_username` совпадает с `preferred_username` из JWT. Пользователь с ролью `ADMIN_ROLE` видит все классы.

## Keycloak

В Docker Keycloak поднимается отдельным сервисом `keycloak` на http://localhost:8080. Realm импортируется из `keycloak/realm-school.json`.

Демо-пользователи:

- `demo.teacher` / `TeacherDemo!2026` - роль `teacher`, видит класс `8А`;
- `second.teacher` / `TeacherDemo!2026` - роль `teacher`, видит класс `9Б`;
- `admin.teacher` / `AdminDemo!2026` - роль `admin`, видит все классы.

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
