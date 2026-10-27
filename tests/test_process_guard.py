"""Тесты предохранителя ресурсов (v2.6.3-quiet)."""

from __future__ import annotations

import os

import pytest

from bot_a.utils import process_guard
from bot_a.utils.process_guard import (
    DEFAULT_MAX_WORKERS,
    HARD_MAX_WORKERS,
    cpu_limit,
    hard_limit,
    limit_blas_threads,
    lower_priority,
    resolve_jobs,
    worker_init,
)


def test_default_jobs_is_small_and_bounded():
    """По умолчанию не больше двух процессов — раньше бралось cpu_count - 1 (15)."""
    jobs = resolve_jobs(None)
    assert 1 <= jobs <= DEFAULT_MAX_WORKERS
    assert jobs <= (os.cpu_count() or 1)


def test_explicit_jobs_is_clamped_to_hard_limit():
    assert resolve_jobs(64) == hard_limit()
    assert resolve_jobs(hard_limit() + 5) == hard_limit()
    assert hard_limit() <= HARD_MAX_WORKERS


def test_jobs_one_stays_one():
    assert resolve_jobs(1) == 1
    assert resolve_jobs(0) == resolve_jobs(None)
    assert resolve_jobs(-7) == resolve_jobs(None)


def test_caller_cap_wins():
    assert resolve_jobs(8, cap=1) == 1
    assert resolve_jobs(None, cap=1) == 1


def test_cpu_limits_positive_on_any_machine():
    assert cpu_limit() >= 1
    assert hard_limit() >= cpu_limit()


def test_limit_blas_threads_sets_env_once(monkeypatch):
    for name in process_guard._BLAS_VARIABLES:  # noqa: SLF001
        monkeypatch.delenv(name, raising=False)
    limit_blas_threads(1)
    for name in process_guard._BLAS_VARIABLES:  # noqa: SLF001
        assert os.environ[name] == "1"

    os.environ["OMP_NUM_THREADS"] = "3"
    limit_blas_threads(1)
    assert os.environ["OMP_NUM_THREADS"] == "3", "явная настройка пользователя не затирается"


def test_lower_priority_returns_bool_and_never_raises(monkeypatch):
    assert isinstance(lower_priority(), bool)

    def boom(*_args, **_kwargs):
        raise OSError("нет прав")

    monkeypatch.setattr(process_guard.os, "nice", boom, raising=False)
    if process_guard.sys.platform != "win32":
        assert lower_priority() is False


def test_worker_init_applies_both_guards(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(process_guard, "limit_blas_threads", lambda *_a: calls.append("blas"))
    monkeypatch.setattr(process_guard, "lower_priority", lambda: calls.append("prio"))

    worker_init()

    assert calls == ["blas", "prio"]


@pytest.mark.parametrize("requested", [None, 1, 2, 3, 8, 100])
def test_resolve_jobs_never_exceeds_hard_limit(requested):
    assert 1 <= resolve_jobs(requested) <= hard_limit()
