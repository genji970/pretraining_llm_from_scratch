import os
import random
from pathlib import Path
import json
import typing

import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_progress(path: Path) -> dict[str, int | str]:
    if not path.exists():
        return {
            "trained_documents": 0,
            "source_rows_consumed": 0,
            "next_chunk_id": 0,
            "global_step": 0,
            "checkpoint_path": "",
        }
    return json.loads(path.read_text(encoding="utf-8"))

def save_progress_atomic(path: Path, progress: dict[str, int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)