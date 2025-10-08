
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from config import GOEMOTIONS_MODEL

@lru_cache(maxsize=2)#For caching results
def load_goemotions_pipeline(model_name: str = GOEMOTIONS_MODEL, device: int = -1):
    """
    Load the GoEmotions pipeline on CPU/GPU.
    """
    if device in (-1, 0, 1, 2, 3):
        chosen_device = device
    else:
        chosen_device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, device_map=None, torch_dtype=None
    )

    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=chosen_device,  # CPU=-1, GPU=0
    )

def _extract_label_score(item: Any) -> Optional[Tuple[str, float]]:
    #Extracting scores 
    if isinstance(item, dict):
        label = item.get("label")
        score = item.get("score")
        if label is None:
            return None
        try:
            return str(label).lower(), float(score or 0.0)
        except (TypeError, ValueError):
            return None
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        label, score = item[0], item[1]
        try:
            return str(label).lower(), float(score)
        except (TypeError, ValueError):
            return None
    return None

#detecting emotions based on the top labels

def detect_emotions(text: str, threshold: float = 0.35, device: int = -1) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"predictions": [], "top_label": (None, 0.0), "raw": None}

    pipe = load_goemotions_pipeline(device=device)
    raw = pipe(text, truncation=True)

    scores_raw = raw[0] if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list) else raw

    normalized: List[Tuple[str, float]] = [
        parsed for item in scores_raw
        if (parsed := _extract_label_score(item)) is not None
    ]

    selected: List[Tuple[str, float]] = [
        (label, score) for label, score in normalized if score >= threshold
    ]

    if not selected and normalized:
        top = max(normalized, key=lambda x: x[1])
        selected = [top]

    top_label = selected[0] if selected else (None, 0.0)
    return {"predictions": selected, "top_label": top_label, "raw": scores_raw}
