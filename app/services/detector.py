import logging
import csv
import json
import math
import pickle
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

    def __init__(
        self,
        settings: Settings,
        model_base_path: str | None = None,
        lora_adapter_path: str | None = None,
        mode: str | None = None,
    ):
        base_path = model_base_path or settings.model_base_path
        adapter_path = lora_adapter_path if lora_adapter_path is not None else settings.lora_adapter_path
        if mode is not None:
            self.mode = mode

        if not base_path:
            raise ValueError("MODEL_BASE_PATH не задан.")
        if adapter_path and not Path(adapter_path).exists():
            raise ValueError(f"LORA_ADAPTER_PATH не найден: {adapter_path}.")

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
        tokenizer_path = base_path
        if adapter_path and (Path(adapter_path) / "tokenizer_config.json").exists():
            tokenizer_path = adapter_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
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

        adapter_num_labels = detect_adapter_num_labels(adapter_path)
        if adapter_num_labels is not None:
            kwargs["num_labels"] = adapter_num_labels

        logger.info("Загружаем детектор: base=%s, lora=%s", base_path, adapter_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_path,
            **hub_kwargs,
            **kwargs,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                prepare_peft_adapter_path(adapter_path),
                device_map="auto" if self.device == "cuda" else None,
            )

        if self.device == "cpu":
            self.model.to(self.device)

        self.model.eval()
        self.max_length = settings.model_max_length
        self.ai_class_index = settings.ai_class_index
        self.invert_probability = settings.model_invert_probability
        logger.info("Детектор загружен: base=%s, tokenizer=%s", base_path, tokenizer_path)

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


class EmbeddingReferenceDetector:
    mode: str = "embedding_reference"

    def __init__(self, settings: Settings):
        import torch
        import torch.nn.functional as functional
        from safetensors.torch import load_file
        from sklearn.neighbors import NearestNeighbors
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.functional = functional
        self.nearest_neighbors_cls = NearestNeighbors
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and not settings.allow_cpu_model_load:
            raise RuntimeError(
                "CUDA/GPU не обнаружена, а загрузка embedding-модели на CPU отключена. "
                "Для CPU-запуска добавьте ALLOW_CPU_MODEL_LOAD=true."
            )

        hf_token = settings.hugging_face_hub_token or settings.hf_token or None
        hub_kwargs = {
            "token": hf_token,
            "local_files_only": settings.model_local_files_only,
        }
        tokenizer_path = Path(settings.embedding_weights_path).parent
        if not (tokenizer_path / "tokenizer_config.json").exists():
            tokenizer_path = Path(settings.embedding_base_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code=True,
            **hub_kwargs,
        )
        self.model = AutoModel.from_pretrained(
            settings.embedding_base_path,
            trust_remote_code=True,
            **hub_kwargs,
        )
        weights_path = Path(settings.embedding_weights_path)
        if weights_path.exists():
            state_dict = load_file(str(weights_path), device="cpu")
            self.model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(f"EMBEDDING_WEIGHTS_PATH не найден: {weights_path}.")

        self.model.to(self.device)
        self.model.eval()
        self.max_length = settings.embedding_max_length
        self.knn_k = max(1, settings.embedding_knn_k)
        reference_texts, reference_labels = self._load_reference_dataset(settings)
        self.reference_labels = reference_labels
        self.reference_embeddings = self._encode_batched(reference_texts, batch_size=16).cpu().numpy()
        n_neighbors = min(self.knn_k, len(self.reference_labels))
        self.knn = self.nearest_neighbors_cls(n_neighbors=n_neighbors, metric="cosine")
        self.knn.fit(self.reference_embeddings)
        logger.info("Embedding reference detector загружен: %s", weights_path)

    def _load_reference_dataset(self, settings: Settings) -> tuple[list[str], list[int]]:
        dataset_path = Path(settings.embedding_reference_dataset_path)
        if dataset_path.exists():
            texts: list[str] = []
            labels: list[int] = []
            with dataset_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                if "text" not in fieldnames:
                    raise ValueError(f"Embedding reference dataset должен иметь колонку text: {dataset_path}.")
                label_column = "generated" if "generated" in fieldnames else "label"
                if label_column not in fieldnames:
                    raise ValueError(
                        "Embedding reference dataset должен иметь колонку generated или label: "
                        f"{dataset_path}."
                    )

                for row in reader:
                    text = (row.get("text") or "").strip()
                    label_value = (row.get(label_column) or "").strip().lower()
                    if not text:
                        continue
                    if label_value in {"1", "true", "ai", "generated"}:
                        labels.append(1)
                        texts.append(text)
                    elif label_value in {"0", "false", "human"}:
                        labels.append(0)
                        texts.append(text)

            if texts and any(label == 1 for label in labels) and any(label == 0 for label in labels):
                return texts, labels
            raise ValueError(f"Embedding reference dataset не содержит оба класса: {dataset_path}.")

        if not settings.embedding_ai_reference_texts_path or not settings.embedding_human_reference_texts_path:
            raise ValueError(
                "Для embedding-score нужны EMBEDDING_REFERENCE_DATASET_PATH или "
                "EMBEDDING_AI_REFERENCE_TEXTS_PATH + EMBEDDING_HUMAN_REFERENCE_TEXTS_PATH."
            )

        ai_texts = self._read_reference_texts(settings.embedding_ai_reference_texts_path)
        human_texts = self._read_reference_texts(settings.embedding_human_reference_texts_path)
        return human_texts + ai_texts, [0] * len(human_texts) + [1] * len(ai_texts)

    def _read_reference_texts(self, path_value: str | None) -> list[str]:
        if not path_value:
            return []

        path = Path(path_value)
        if not path.exists():
            raise ValueError(f"Файл reference-текстов не найден: {path}.")

        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(item).strip() for item in data if str(item).strip()]
            raise ValueError(f"JSON reference-файл должен содержать список строк: {path}.")

        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if "text" not in (reader.fieldnames or []):
                    raise ValueError(f"CSV reference-файл должен иметь колонку text: {path}.")
                return [row["text"].strip() for row in reader if row.get("text", "").strip()]

        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _encode(self, texts: list[str]):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with self.torch.no_grad():
            output = self.model(**encoded)

        token_embeddings = output[0]
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = self.torch.sum(token_embeddings * mask, 1) / self.torch.clamp(mask.sum(1), min=1e-9)
        return self.functional.normalize(embeddings, p=2, dim=1)

    def _encode_batched(self, texts: list[str], batch_size: int):
        if not texts:
            raise ValueError("Reference-тексты для embedding-score пустые.")
        batches = []
        for index in range(0, len(texts), batch_size):
            batches.append(self._encode(texts[index : index + batch_size]).detach())
        return self.torch.cat(batches, dim=0)

    def predict_probability(self, text: str) -> float:
        embedding = self._encode([text]).cpu().numpy()
        _, indices = self.knn.kneighbors(embedding)
        neighbor_labels = [self.reference_labels[index] for index in indices[0]]
        probability = sum(neighbor_labels) / len(neighbor_labels)
        return float(max(0.0, min(1.0, probability)))


class EnsembleDetector:
    mode: str = "ensemble"

    def __init__(self, settings: Settings):
        self.members: list[tuple[str, Detector, float]] = [
            (
                "qwen",
                TransformerDetector(
                    settings,
                    model_base_path=settings.ensemble_qwen_base_path,
                    lora_adapter_path=settings.ensemble_qwen_adapter_path,
                    mode="qwen2.5_7b_lora",
                ),
                settings.ensemble_qwen_weight,
            ),
            (
                "t_lite",
                TransformerDetector(
                    settings,
                    model_base_path=settings.ensemble_t_lite_base_path,
                    lora_adapter_path=settings.ensemble_t_lite_adapter_path,
                    mode="t_lite_7b_lora",
                ),
                settings.ensemble_t_lite_weight,
            ),
        ]
        self.use_meta_learner = settings.ensemble_use_meta_learner
        self.meta_learner = None
        self.meta_feature_order = [
            name.strip()
            for name in settings.ensemble_meta_feature_order.split(",")
            if name.strip()
        ]

        if settings.embedding_enabled:
            self.members.append(
                (
                    "embed",
                    EmbeddingReferenceDetector(settings),
                    settings.ensemble_embed_weight,
                )
            )
        elif self.use_meta_learner:
            raise ValueError("ENSEMBLE_USE_META_LEARNER=true требует EMBEDDING_ENABLED=true.")

        if self.use_meta_learner:
            if len(self.meta_feature_order) != 3:
                raise ValueError("ENSEMBLE_META_FEATURE_ORDER должен содержать ровно 3 имени признаков.")
            meta_learner_path = Path(settings.ensemble_meta_learner_path)
            if not meta_learner_path.exists():
                raise ValueError(f"ENSEMBLE_META_LEARNER_PATH не найден: {meta_learner_path}.")

            # Pickle безопасен только для доверенного локального файла.
            with meta_learner_path.open("rb") as handle:
                self.meta_learner = pickle.load(handle)
            if not hasattr(self.meta_learner, "predict_proba"):
                raise ValueError("Meta learner должен поддерживать predict_proba.")
            logger.info("Meta learner ансамбля загружен: %s", meta_learner_path)

        names = ",".join(name for name, _, _ in self.members)
        self.mode = f"ensemble:{names}"
        logger.info("Ансамбль загружен: %s", self.mode)

    def predict_probability(self, text: str) -> float:
        probabilities = {
            name: detector.predict_probability(text)
            for name, detector, _ in self.members
        }

        if self.use_meta_learner:
            if self.meta_learner is None:
                raise RuntimeError("Meta learner не загружен.")
            features = [[probabilities[name] for name in self.meta_feature_order]]
            probability = self.meta_learner.predict_proba(features)[0][1]
            return float(max(0.0, min(1.0, probability)))

        total_weight = sum(weight for name, _, weight in self.members if name in probabilities)
        if total_weight <= 0:
            raise RuntimeError("Сумма весов ансамбля должна быть больше 0.")

        probability = sum(
            probabilities[name] * weight
            for name, _, weight in self.members
        ) / total_weight
        return float(max(0.0, min(1.0, probability)))


def build_detector(settings: Settings) -> Detector:
    if settings.use_mock_detector:
        logger.info("USE_MOCK_DETECTOR=true: используем эвристический детектор.")
        return MockDetector(mode="heuristic")
    if settings.use_ensemble_detector:
        logger.info("USE_ENSEMBLE_DETECTOR=true: используем ансамбль моделей.")
        return EnsembleDetector(settings)
    return TransformerDetector(settings)
