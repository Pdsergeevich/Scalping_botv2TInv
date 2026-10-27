"""Безопасное получение секретов (токен T-Invest) без утечки в код и логи.

Приоритет источников:
1. явный аргумент ``--token``;
2. переменная окружения (``tInvest.tokenEnvVar``, по умолчанию ``TINVEST_TOKEN``);
3. файл, на который указывает ``TINVEST_TOKEN_FILE``;
4. локальный файл ``./secrets/tinvest_token.txt`` (удобно, когда токен нельзя
   передать через окружение отдельного процесса);
5. поле ``tInvest.token`` в ``appsettings.json``.

Файл читается построчно: берётся первая непустая строка без комментария ``#``.
Никаких печатей токена в логи — только признак «задан» и длина.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_TOKEN_FILE = Path("secrets") / "tinvest_token.txt"
TOKEN_FILE_ENV_VAR = "TINVEST_TOKEN_FILE"


def _clean(token: str) -> str:
    """Убирает BOM, кавычки и переводы строк — PowerShell пишет UTF-8 с BOM."""
    return token.replace("\ufeff", "").strip().strip('"').strip("'").strip()


def _read_token_file(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    for encoding in ("utf-8-sig", "utf-16", "cp1251"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return ""
    for line in text.splitlines():
        cleaned = _clean(line)
        if cleaned and not cleaned.startswith("#"):
            return cleaned
    return ""


def read_token(
    explicit: str = "",
    env_var: str = "TINVEST_TOKEN",
    base_dir: Optional[str | Path] = None,
    file_hint: Optional[str | Path] = None,
) -> str:
    """Возвращает токен из первого доступного источника (или пустую строку)."""
    if explicit and _clean(explicit):
        return _clean(explicit)

    from_env = _clean(os.environ.get(env_var, ""))
    if from_env:
        return from_env

    candidates = []
    env_file = os.environ.get(TOKEN_FILE_ENV_VAR, "").strip()
    if env_file:
        candidates.append(Path(env_file))
    if file_hint:
        candidates.append(Path(file_hint))
    root = Path(base_dir) if base_dir is not None else Path(".")
    candidates.append(root / DEFAULT_TOKEN_FILE)

    for candidate in candidates:
        if candidate.is_file():
            token = _read_token_file(candidate)
            if token:
                return token
    return ""


def mask(token: str) -> str:
    """Безопасное представление токена для логов."""
    if not token:
        return "не задан"
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}…{token[-2:]} (len={len(token)})"


def token_source(explicit: str = "", env_var: str = "TINVEST_TOKEN",
                 base_dir: Optional[str | Path] = None) -> str:
    """Откуда взят токен — для диагностики preflight."""
    if explicit:
        return "аргумент --token"
    if os.environ.get(env_var, "").strip():
        return f"переменная окружения {env_var}"
    env_file = os.environ.get(TOKEN_FILE_ENV_VAR, "").strip()
    if env_file and Path(env_file).is_file():
        return f"файл {env_file} ({TOKEN_FILE_ENV_VAR})"
    root = Path(base_dir) if base_dir is not None else Path(".")
    local = root / DEFAULT_TOKEN_FILE
    if local.is_file() and _read_token_file(local):
        return f"локальный файл {local}"
    return "не найден"
