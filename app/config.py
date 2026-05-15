from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Text Detector Service"
    database_url: str = "sqlite:///./data/results.db"

    ocr_model_name: str = "cyrillic-trocr/trocr-handwritten-cyrillic"
    ocr_model_local_files_only: bool = False
    ocr_preload_on_startup: bool = False
    ocr_min_line_height: int = 20
    ocr_line_threshold_ratio: float = 0.02
    ocr_line_padding: int = 15
    threshold: float = 0.5
    max_text_length: int = 20_000
    upload_max_bytes: int = 8 * 1024 * 1024
    pdf_max_pages: int = 20
    pdf_render_dpi: int = 200

    use_mock_detector: bool = True
    model_base_path: str | None = None
    lora_adapter_path: str | None = None
    use_ensemble_detector: bool = False
    ensemble_qwen_base_path: str = "Qwen/Qwen2.5-7B-Instruct"
    ensemble_qwen_adapter_path: str = "./models/r_detect_qwen3"
    ensemble_t_lite_base_path: str = "t-tech/T-lite-it-1.0"
    ensemble_t_lite_adapter_path: str = "./models/r_detect_t_lite_v2"
    ensemble_qwen_weight: float = 0.3360
    ensemble_t_lite_weight: float = 0.3279
    ensemble_embed_weight: float = 0.3360
    ensemble_use_meta_learner: bool = False
    ensemble_meta_learner_path: str = "./models/meta_learner_3models.pkl"
    ensemble_meta_feature_order: str = "t_lite,qwen,embed"
    embedding_enabled: bool = False
    embedding_base_path: str = "deepvk/USER-bge-m3"
    embedding_weights_path: str = "./models/r_embed_final/model.safetensors"
    embedding_reference_dataset_path: str = "./datasets/final_prepared/final_train.csv"
    embedding_ai_reference_texts_path: str | None = None
    embedding_human_reference_texts_path: str | None = None
    embedding_knn_k: int = 5
    embedding_max_length: int = 512
    load_in_4bit: bool = True
    model_max_length: int = 256
    ai_class_index: int = 1
    model_invert_probability: bool = False
    allow_cpu_model_load: bool = False
    model_local_files_only: bool = False
    hugging_face_hub_token: str | None = None
    hf_token: str | None = None

    auth_enabled: bool = False
    keycloak_base_url: str | None = None
    keycloak_public_base_url: str | None = None
    keycloak_internal_base_url: str | None = None
    keycloak_realm: str | None = None
    keycloak_client_id: str | None = None
    keycloak_audience: str | None = None
    keycloak_algorithms: str = "RS256"
    admin_role: str = "admin"
    admin_usernames: str = "admin.teacher,admin.teacher@example.test"
    teacher_role: str = "teacher"
    demo_teacher_username: str = "demo.teacher"
    class_access_claim: str = "school_classes"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
