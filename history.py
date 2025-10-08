
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from config import HISTORY_FILE

def _ensure_file():
    p = Path(HISTORY_FILE)
    if not p.exists():
        df = pd.DataFrame(columns=[
            "timestamp", "prompt", "used_emotions", "generated",
            "target_words", "temperature", "mode"
        ])
        df.to_csv(p, index=False)

def save_history(item: Dict[str, Any]):
    _ensure_file()
    df = load_history(limit=None)
    # Ensure all expected columns exist
    for col in ["timestamp","prompt","used_emotions","generated","target_words","temperature","mode"]:
        if col not in df.columns:
            df[col] = None
    df = pd.concat([pd.DataFrame([item]), df], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

def load_history(limit: int = 200) -> pd.DataFrame:
    _ensure_file()
    df = pd.read_csv(HISTORY_FILE)
    if limit:
        return df.head(limit)
    return df
