import logging
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.config import Settings

logger = logging.getLogger(__name__)


def prepare_peft_adapter_path(adapter_path: str) -> str:
    path = Path(adapter_path)
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        return adapter_path

    config = json.loads(config_path.read_text(encoding="utf-8"))
    modules_to_save = config.get("modules_to_save") or []
    if "classification_head" not in modules_to_save:
        return adapter_path

    try:
        from safetensors.torch import load_file, save_file
    except Exception:
        logger.exception("Не удалось импортировать safetensors для проверки PEFT adapter.")
        return adapter_path

    state_dict = load_file(str(weights_path), device="cpu")
    old_key = "base_model.model.classification_head.weight"
    new_key = "base_model.model.score.weight"
    if old_key not in state_dict or new_key in state_dict:
        return adapter_path

    fixed_dir = Path(tempfile.mkdtemp(prefix="peft_adapter_score_"))
    for item in path.iterdir():
        if item.name in {"adapter_config.json", "adapter_model.safetensors"}:
            continue
        target = fixed_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    config["modules_to_save"] = [
        "score" if module == "classification_head" else module
        for module in modules_to_save
    ]
    (fixed_dir / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    state_dict[new_key] = state_dict.pop(old_key)
    save_file(state_dict, str(fixed_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    logger.info("Используем совместимую копию PEFT adapter: %s", fixed_dir)
    return str(fixed_dir)


def detect_adapter_num_labels(adapter_path: str | None) -> int | None:
    if not adapter_path:
        return None

    weights_path = Path(adapter_path) / "adapter_model.safetensors"
    if not weights_path.exists():
        return None

    try:
        from safetensors.torch import load_file
    except Exception:
        logger.exception("Не удалось импортировать safetensors для определения num_labels adapter.")
        return None

    state_dict = load_file(str(weights_path), device="cpu")
    for key in [
        "base_model.model.score.weight",
        "base_model.model.classification_head.weight",
    ]:
        weight = state_dict.get(key)
        if weight is not None and len(weight.shape) == 2:
            return int(weight.shape[0])
    return None


class Detector(Protocol):
    mode: str

    def predict_probability(self, text: str) -> float:
        """Вернуть вероятность AI-сгенерированного текста от 0 до 1."""


@dataclass
class MockDetector:
    """Детерминированная заглушка для проверки веб-части без весов модели.

    Не использовать как научный результат. Она нужна только для разработки UI/API.
    """

    mode: str = "mock"

    def predict_probability(self, text: str) -> float:
        words = re.findall(r"[А-Яа-яA-Za-zЁё]+", text.lower())
        if len(words) < 20:
            return 0.5

        unique_ratio = len(set(words)) / max(len(words), 1)
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        avg_sentence_words = len(words) / max(len(sentences), 1)

        # Простые признаки: шаблонность, слишком ровная длина предложений,
        # низкое лексическое разнообразие слегка повышают score.
        score = 0.45
        score += (0.52 - unique_ratio) * 0.9
        score += min(max((avg_sentence_words - 12) / 50, -0.12), 0.18)
        score += min(max((avg_word_len - 5) / 30, -0.05), 0.08)

        # Стабилизация результата в диапазоне 0.02..0.98
        return float(max(0.02, min(0.98, score)))


class TransformerDetector:
    mode: str = "transformers_peft"

    def __init__(self, settings: Settings):
        if not settings.model_base_path:
            raise ValueError("MODEL_BASE_PATH не задан.")
        if settings.lora_adapter_path and not Path(settings.lora_adapter_path).exists():
            raise ValueError(f"LORA_ADAPTER_PATH не найден: {settings.lora_adapter_path}.")

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_info()

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and not settings.allow_cpu_model_load:
            raise RuntimeError(
                "CUDA/GPU не обнаружена, а загрузка Mistral-7B на CPU отключена. "
                "Для CPU-запуска добавьте ALLOW_CPU_MODEL_LOAD=true."
            )

        hf_token = settings.hugging_face_hub_token or settings.hf_token or None
        hub_kwargs = {
            "token": hf_token,
            "local_files_only": settings.model_local_files_only,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_base_path,
            use_fast=True,
            trust_remote_code=True,
            **hub_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None
        if settings.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except Exception:
                logger.exception("4-bit конфигурация недоступна, грузим без bitsandbytes.")

        kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        elif self.device == "cuda":
            kwargs["torch_dtype"] = torch.float16

        adapter_num_labels = detect_adapter_num_labels(settings.lora_adapter_path)
        if adapter_num_labels is not None:
            kwargs["num_labels"] = adapter_num_labels

        logger.info("Загружаем детектор: base=%s, lora=%s", settings.model_base_path, settings.lora_adapter_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            settings.model_base_path,
            **hub_kwargs,
            **kwargs,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if settings.lora_adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                prepare_peft_adapter_path(settings.lora_adapter_path),
                device_map="auto" if self.device == "cuda" else None,
            )

        if self.device == "cpu":
            self.model.to(self.device)

        self.model.eval()
        self.max_length = settings.model_max_length
        self.ai_class_index = settings.ai_class_index
        self.invert_probability = settings.model_invert_probability

    def predict_probability(self, text: str) -> float:
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with self.torch.no_grad():
            logits = self.model(**inputs).logits

        if logits.shape[-1] == 1:
            prob = self.torch.sigmoid(logits.squeeze()).item()
        else:
            class_index = min(max(self.ai_class_index, 0), logits.shape[-1] - 1)
            prob = self.torch.softmax(logits, dim=-1)[0, class_index].item()

        if math.isnan(prob):
            raise RuntimeError("Модель вернула NaN.")
        if self.invert_probability:
            prob = 1.0 - prob
        return float(max(0.0, min(1.0, prob)))


def build_detector(settings: Settings) -> Detector:
    if settings.use_mock_detector:
        logger.info("USE_MOCK_DETECTOR=true: используем эвристический детектор.")
        return MockDetector(mode="heuristic")
    return TransformerDetector(settings)
