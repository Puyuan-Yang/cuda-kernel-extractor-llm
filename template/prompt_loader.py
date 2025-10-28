from pathlib import Path
from typing import Optional


def load_prompt(path: str, encoding: str = "utf-8") -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return file_path.read_text(encoding=encoding)


def ensure_prompt(path: str, default_content: str = "", encoding: str = "utf-8") -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        file_path.write_text(default_content, encoding=encoding)

