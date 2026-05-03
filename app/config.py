from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Text Detector Service"
    database_url: str = "sqlite:///./data/results.db"

    ocr_lang: str = "rus+eng"
    threshold: float = 0.73
    max_text_length: int = 20_000
    upload_max_bytes: int = 8 * 1024 * 1024

    # true — безопасная заглушка, false — загрузка реальной модели
    use_mock_detector: bool = True
    model_base_path: str | None = None
    lora_adapter_path: str | None = None
    load_in_4bit: bool = True
    model_max_length: int = 256
    ai_class_index: int = 1
    allow_cpu_model_load: bool = False
    model_local_files_only: bool = False
    hugging_face_hub_token: str | None = None
    hf_token: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
