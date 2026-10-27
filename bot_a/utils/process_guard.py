"""Предохранитель ресурсов для тяжёлых вычислений.

Гриды бэктестов на 16-поточном CPU в 6–15 процессов давали 100 % загрузку
всех ядер, из-за чего отваливался дисплей (TDR/питание). Модуль ограничивает
параллелизм, число BLAS-потоков и понижает приоритет процесса, чтобы расчёты
не мешали остальной работе.
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Any, Dict, Optional

#: Сколько процессов разрешаем по умолчанию — держим машину отзывчивой.
DEFAULT_MAX_WORKERS = 2
#: Потолок, даже если в CLI явно попросили больше.
HARD_MAX_WORKERS = 4

_BLAS_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

_BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def limit_blas_threads(threads: int = 1) -> None:
    """Не даёт каждому воркеру плодить BLAS-потоки по числу ядер.

    Значения выставляются через `setdefault`, чтобы не затирать явную настройку
    пользователя. Вызывать до импорта numpy/pandas.
    """
    value = str(max(int(threads), 1))
    for name in _BLAS_VARIABLES:
        os.environ.setdefault(name, value)


def lower_priority() -> bool:
    """Понижает приоритет текущего процесса, чтобы система оставалась отзывчивой.

    Возвращает `True`, если приоритет изменён. Ошибки подавляются: в контейнерах
    и под ограниченным пользователем вызов может быть запрещён.
    """
    try:
        if sys.platform == "win32":
            handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
            ok = ctypes.windll.kernel32.SetPriorityClass(  # type: ignore[attr-defined]
                handle, _BELOW_NORMAL_PRIORITY_CLASS
            )
            return bool(ok)
        os.nice(10)  # type: ignore[attr-defined]
        return True
    except Exception:  # noqa: BLE001 — снижение приоритета не критично для расчётов
        return False


def cpu_limit() -> int:
    """Безопасное число воркеров по умолчанию для этой машины."""
    cores = os.cpu_count() or 1
    return max(1, min(DEFAULT_MAX_WORKERS, cores // 4 or 1))


def hard_limit() -> int:
    """Максимум воркеров, который разрешаем даже по явному запросу."""
    cores = os.cpu_count() or 1
    return max(1, min(HARD_MAX_WORKERS, cores // 2 or 1))


def resolve_jobs(requested: Optional[int] = None, cap: Optional[int] = None) -> int:
    """Приводит запрошенное число процессов к безопасному значению.

    * `None` или `<= 0` — `cpu_limit()` (по умолчанию 2, а не `cpu_count - 1`);
    * явный запрос обрезается до `hard_limit()`;
    * `cap` (жёсткое ограничение вызывающего) применяется последним.
    """
    ceiling = hard_limit() if cap is None else max(1, min(int(cap), hard_limit()))
    if requested is None or int(requested) <= 0:
        return max(1, min(cpu_limit(), ceiling))
    return max(1, min(int(requested), ceiling))


def worker_init() -> None:
    """Инициализатор воркера `ProcessPoolExecutor`: тихий режим в каждом процессе."""
    limit_blas_threads(1)
    lower_priority()


def env_report() -> Dict[str, Any]:
    """Краткая сводка настроек — удобно писать в лог перед гридом."""
    return {
        "jobs": resolve_jobs(None),
        "cpu": os.cpu_count() or 1,
        "blasThreads": os.environ.get("OMP_NUM_THREADS", "unset"),
    }
