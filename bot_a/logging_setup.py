"""Логирование в консоль и файл (аналог Serilog Console + File sinks)."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LEVELS = {
    "verbose": logging.DEBUG,
    "debug": logging.DEBUG,
    "information": logging.INFO,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.CRITICAL,
    "critical": logging.CRITICAL,
}

_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)-7s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: str = "Information",
    log_dir: str = "./logs",
    file_name: str = "bot_a.log",
    console_stream=None,
) -> logging.Logger:
    """Настраивает корневой логгер и возвращает логгер приложения."""
    _make_console_safe()
    numeric_level = _LEVELS.get(level.strip().lower(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)
    console = logging.StreamHandler(stream=console_stream or sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(numeric_level)
    root.addHandler(console)

    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        directory / file_name,
        maxBytes=20 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    logging.getLogger("asyncio").setLevel(logging.WARNING)
    return logging.getLogger("bot_a")



def _make_console_safe() -> None:
    """Не даём логу упасть на символах вне кодовой страницы консоли (—, «», ₽)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except Exception:
            pass
